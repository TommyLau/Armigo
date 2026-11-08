// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <armigo/graphics/texflush.h>
#include <armigo/memory/write_tracker.h>
#include <armigo/nvn/driver.h>

#include "nvn/resource_layout.h"
#include "platform/metal/nvn_resources.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

namespace nvn = armigo::nvn;
namespace metal = armigo::platform::metal::nvn_driver;
using namespace armigo::nvn::api;

unsigned failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);             \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

class Pages {
public:
  explicit Pages(std::size_t count)
      : size_(count * static_cast<std::size_t>(getpagesize())) {
    memory_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory_ == MAP_FAILED)
      memory_ = nullptr;
  }

  ~Pages() {
    if (memory_)
      munmap(memory_, size_);
  }

  Pages(const Pages &) = delete;
  Pages &operator=(const Pages &) = delete;

  [[nodiscard]] std::byte *bytes() const {
    return static_cast<std::byte *>(memory_);
  }
  [[nodiscard]] std::size_t size() const { return size_; }

private:
  void *memory_ = nullptr;
  std::size_t size_ = 0;
};

bool complete(id<MTLCommandBuffer> commands) {
  if (!commands)
    return false;
  [commands waitUntilCompleted];
  return commands.status == MTLCommandBufferStatusCompleted &&
         commands.error == nil;
}

void check_buffer_pool(id<MTLDevice> device, metal::Resources &resources,
                       id<MTLCommandQueue> queue) {
  Pages pages(2);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  constexpr std::size_t displacement = 37;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes() + displacement);
  pool.size = pages.size() - displacement;
  CHECK(resources.register_pool(pool));

  auto span =
      resources.buffer_at(reinterpret_cast<std::uint64_t>(pool.memory), 64);
  CHECK(span.buffer != nil);
  CHECK(span.offset == displacement);
  CHECK(span.buffer.contents == pages.bytes());

  id<MTLCommandBuffer> pending = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [pending blitCommandEncoder];
  CHECK(blit != nil);
  [blit fillBuffer:span.buffer range:NSMakeRange(span.offset, 64) value:0x5a];
  [blit endEncoding];
  [pending commit];
  unsigned completion_calls = 0;
  resources.set_completion_hook([&] {
    ++completion_calls;
    return complete(pending);
  });
  resources.note_gpu_write(reinterpret_cast<std::uint64_t>(pool.memory), 64);
  resources.invalidate_pool(pool, 0, 64);
  CHECK(completion_calls == 1);
  CHECK(std::all_of(pool.memory, pool.memory + 64,
                    [](std::uint8_t value) { return value == 0x5a; }));
  resources.release_pool(pool);
  const auto address = reinterpret_cast<std::uint64_t>(pool.memory);
  CHECK(resources.buffer_at(address, 64).buffer == nil);
  CHECK(resources.register_pool(pool));
  const auto replacement = resources.buffer_at(address, 64);
  CHECK(replacement.buffer && replacement.buffer != span.buffer);
  CHECK(replacement.offset == displacement);
  CHECK(resources.buffer_at(address + 16, 16).buffer == replacement.buffer);
  CHECK(resources.buffer_at(address, pool.size + 1).buffer == nil);
  CHECK(resources.buffer_at(UINT64_MAX - 8, 16).buffer == nil);
  resources.release_pool(pool);

  // buffer_address hands out the pool's buffer unretained, so a released
  // pool's buffer stays alive until the replay worker begins its next walk.
  __weak id<MTLBuffer> released = nil;
  @autoreleasepool {
    CHECK(resources.register_pool(pool));
    released = resources.buffer_address(address, 64).buffer;
    CHECK(released != nil);
    resources.release_pool(pool);
  }
  CHECK(released != nil);
  resources.begin_walk();
  CHECK(released == nil);
  resources.set_completion_hook({});
  static_cast<void>(device);
}

nvn::TextureState make_texture(nvn::MemoryPoolState &pool,
                               std::uint64_t serial) {
  nvn::TextureState texture{};
  texture.description.target = NVN_TEXTURE_TARGET_2D;
  texture.description.format = NVN_FORMAT_RGBA8;
  texture.description.samples = 1;
  texture.description.width = 4;
  texture.description.height = 4;
  texture.description.depth = 1;
  texture.description.levels = 1;
  const auto layout = nvn::resource_layout(texture.description);
  if (!layout)
    return texture;
  texture.size = layout->total_size;
  texture.layer_size = layout->layer_stride;
  texture.memory = pool.memory;
  texture.pool = &pool;
  texture.offset = 0;
  texture.serial = serial;
  return texture;
}

nvn::TexelTransfer whole_rgba_transfer() {
  nvn::TexelTransfer transfer;
  transfer.region = {0, 0, 0, 4, 4, 1};
  transfer.row_stride = 16;
  transfer.image_stride = 64;
  return transfer;
}

// Texture uploads are batched on the resources' transfer queue: a reader on
// another queue commits them first, as the replay does before its segments.
std::vector<std::byte> read_native_rgba(metal::Resources& resources, id<MTLDevice> device,
                                        id<MTLCommandQueue> queue,
                                        id<MTLTexture> texture) {
  CHECK(resources.flush_uploads());
  constexpr std::size_t row = 256;
  id<MTLBuffer> output =
      [device newBufferWithLength:row * 4 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> commands = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
  [blit copyFromTexture:texture
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(4, 4, 1)
                      toBuffer:output
             destinationOffset:0
        destinationBytesPerRow:row
      destinationBytesPerImage:row * 4];
  [blit endEncoding];
  [commands commit];
  CHECK(complete(commands));
  std::vector<std::byte> result(64);
  for (std::size_t y = 0; y < 4; ++y)
    std::memcpy(result.data() + y * 16,
                static_cast<const std::byte *>(output.contents) + y * row, 16);
  return result;
}

// Colour textures stay losslessly compressible: no shader-write or
// format-view usage. A mip-mapped colour texture keeps format views, which
// leaves it uncompressed for mip chains that sample a level next to the one
// they render.
// Guest-memory tables lease their pools: a GPU-inaccessible pool is never
// listed, an unchanged pool set is not listed again, and a released pool
// waits for the submissions that leased it. A table leased while the release
// waits already leaves the pool out. With no lease pending, a release does not
// wait.
void check_guest_memory_claims(metal::Resources &resources) {
  Pages visible_pages(1), hidden_pages(1), later_pages(1);
  CHECK(visible_pages.bytes() && hidden_pages.bytes() && later_pages.bytes());
  if (!visible_pages.bytes() || !hidden_pages.bytes() || !later_pages.bytes())
    return;
  nvn::MemoryPoolState visible{}, hidden{}, later{};
  visible.memory = reinterpret_cast<std::uint8_t *>(visible_pages.bytes());
  visible.size = visible_pages.size();
  hidden.memory = reinterpret_cast<std::uint8_t *>(hidden_pages.bytes());
  hidden.size = hidden_pages.size();
  hidden.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS;
  later.memory = reinterpret_cast<std::uint8_t *>(later_pages.bytes());
  later.size = later_pages.size();
  CHECK(resources.register_pool(visible));
  CHECK(resources.register_pool(hidden));
  const auto visible_begin = reinterpret_cast<std::uint64_t>(visible.memory);
  const auto listed = [&](const std::vector<metal::Resources::GuestMemoryPool> &pools, std::uint64_t begin) {
    return std::ranges::any_of(pools, [&](const auto &pool) { return pool.begin == begin; });
  };
  std::vector<metal::Resources::GuestMemoryPool> pools;
  const auto generation = resources.claim_guest_memory_pools(0, pools);
  CHECK(generation != 0 && pools.size() == 1 && listed(pools, visible_begin));
  CHECK(!listed(pools, reinterpret_cast<std::uint64_t>(hidden.memory)));
  std::vector<metal::Resources::GuestMemoryPool> unchanged(1);
  CHECK(resources.claim_guest_memory_pools(generation, unchanged) == generation && unchanged.size() == 1 &&
        unchanged.front().begin == 0);

  // While the release waits, a new lease leaves the pool out; retiring both
  // ends the wait.
  std::atomic<bool> claimed{false};
  std::uint64_t during_generation = 0;
  std::vector<metal::Resources::GuestMemoryPool> during;
  std::thread retiring([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    during_generation = resources.claim_guest_memory_pools(generation, during);
    claimed = true;
    resources.complete_gpu_writes(resources.gpu_write_generation());
  });
  resources.release_pool(visible);
  retiring.join();
  CHECK(claimed && during_generation != generation && !listed(during, visible_begin));
  CHECK(resources.buffer_at(visible_begin, 64).buffer == nil);
  // With no lease outstanding a release does not wait.
  CHECK(resources.register_pool(later));
  const auto start = std::chrono::steady_clock::now();
  resources.release_pool(later);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
  resources.release_pool(hidden);
}

// A guest-memory lease holds back the release of its pools until the
// submission that took it retires, and only that: the generic replay
// completion hook (which a held queue can block) is not used. A release whose
// wait fails keeps the pool registered, and its range cannot be registered
// again. (CPU invalidation does not wait: as on the console, the guest's sync
// wait orders its reads after the GPU work, whose stores land in its pages.)
void check_guest_lease_waits(id<MTLDevice> device) {
  Pages first_pages(1), second_pages(1);
  CHECK(first_pages.bytes() && second_pages.bytes());
  if (!first_pages.bytes() || !second_pages.bytes())
    return;
  metal::Resources resources(device);
  unsigned hook_calls = 0;
  resources.set_completion_hook([&] {
    ++hook_calls;
    return false;
  });
  nvn::MemoryPoolState first{}, second{};
  first.memory = reinterpret_cast<std::uint8_t *>(first_pages.bytes());
  first.size = first_pages.size();
  second.memory = reinterpret_cast<std::uint8_t *>(second_pages.bytes());
  second.size = second_pages.size();
  CHECK(resources.register_pool(first));
  CHECK(resources.register_pool(second));
  const auto first_begin = reinterpret_cast<std::uint64_t>(first.memory);
  const auto second_begin = reinterpret_cast<std::uint64_t>(second.memory);
  std::vector<metal::Resources::GuestMemoryPool> pools;
  // Completes every write noted so far after a delay, from another thread,
  // as the replay's retirement does.
  const auto retire_later = [&](std::atomic<bool> &retired) {
    return std::thread([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      retired = true;
      resources.complete_gpu_writes(resources.gpu_write_generation());
    });
  };

  // Release waits for the lease without the generic hook, then removes.
  CHECK(resources.claim_guest_memory_pools(0, pools) != 0 && pools.size() == 2);
  std::atomic<bool> retired{false};
  auto retiring = retire_later(retired);
  resources.release_pool(first);
  CHECK(retired && hook_calls == 0);
  CHECK(resources.buffer_at(first_begin, 64).buffer == nil);
  retiring.join();

  // A lease that never retires: the release gives up and keeps the pool.
  resources.set_lease_wait_timeout(std::chrono::milliseconds(100));
  resources.claim_guest_memory_pools(0, pools);
  resources.release_pool(second);
  CHECK(resources.buffer_at(second_begin, 64).buffer != nil);
  CHECK(!resources.register_pool(second));
  resources.complete_gpu_writes(resources.gpu_write_generation());
  resources.set_completion_hook({});
}

// A GPU write found once it completed (the guest-memory write log) reaches a
// texture with storage of its own: its next use uploads the bytes the GPU
// stored, which its CPU write tracking never saw, once, and nothing when the
// stores left its bytes as uploaded. The write needs no
// write generation, so publishing it before the replay completes its
// submission's generation refuses nothing. Such a texture's data comes from
// its backing until the GPU writes the texture itself.
void check_completed_gpu_write(id<MTLDevice> device, metal::Resources &resources, id<MTLCommandQueue> queue) {
  Pages pages(1);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  std::memset(pool.memory, 0x10, pool.size);
  CHECK(resources.register_pool(pool));
  nvn::TextureState texture = make_texture(pool, 9811);
  resources.register_texture(texture);
  id<MTLTexture> first = resources.texture(texture, nullptr);
  CHECK(first != nil);
  const auto before = read_native_rgba(resources, device, queue, first);
  CHECK(std::ranges::all_of(before, [](std::byte value) { return value == std::byte{0x10}; }));
  // The GPU stores into the pool: the CPU tracker does not see it.
  const auto address = reinterpret_cast<std::uint64_t>(pool.memory);
  const auto span = resources.buffer_at(address, texture.size);
  CHECK(span.buffer != nil);
  id<MTLCommandBuffer> commands = [queue commandBuffer];
  id<MTLBlitCommandEncoder> fill = [commands blitCommandEncoder];
  [fill fillBuffer:span.buffer range:NSMakeRange(span.offset, texture.size) value:0x60];
  [fill endEncoding];
  [commands commit];
  CHECK(complete(commands));
  CHECK(resources.backing_feeds_texture(address, texture.size));
  // An unrelated write is still pending: its generation is incomplete.
  Pages other(1);
  nvn::MemoryPoolState other_pool{};
  other_pool.memory = reinterpret_cast<std::uint8_t *>(other.bytes());
  other_pool.size = other.size();
  other_pool.flags = pool.flags;
  CHECK(other.bytes() && resources.register_pool(other_pool));
  resources.note_gpu_write(reinterpret_cast<std::uint64_t>(other_pool.memory), 64);
  resources.note_completed_gpu_write(address, pool.size);
  const auto uploads = resources.upload_counts();
  const auto after = read_native_rgba(resources, device, queue, resources.texture(texture, nullptr));
  CHECK(std::ranges::all_of(after, [](std::byte value) { return value == std::byte{0x60}; }));
  const auto refreshed = resources.upload_counts();
  CHECK(refreshed.gpu + refreshed.cpu == uploads.gpu + uploads.cpu + 1);
  CHECK(resources.texture(texture, nullptr) != nil);
  const auto again = resources.upload_counts();
  CHECK(again.gpu + again.cpu == refreshed.gpu + refreshed.cpu);
  // Stores into the pool that leave the texture's bytes as uploaded upload
  // nothing; stores that change them upload once.
  resources.note_completed_gpu_write(address, pool.size);
  CHECK(resources.texture(texture, nullptr) != nil);
  const auto unchanged = resources.upload_counts();
  CHECK(unchanged.gpu + unchanged.cpu == again.gpu + again.cpu);
  commands = [queue commandBuffer];
  fill = [commands blitCommandEncoder];
  [fill fillBuffer:span.buffer range:NSMakeRange(span.offset, texture.size) value:0x70];
  [fill endEncoding];
  [commands commit];
  CHECK(complete(commands));
  resources.note_completed_gpu_write(address, pool.size);
  const auto changed = read_native_rgba(resources, device, queue, resources.texture(texture, nullptr));
  CHECK(std::ranges::all_of(changed, [](std::byte value) { return value == std::byte{0x70}; }));
  const auto rewritten = resources.upload_counts();
  CHECK(rewritten.gpu + rewritten.cpu == unchanged.gpu + unchanged.cpu + 1);
  resources.complete_gpu_writes(resources.gpu_write_generation());
  // Once the GPU wrote the texture, its data no longer comes from the pool.
  resources.note_gpu_write(texture);
  CHECK(!resources.backing_feeds_texture(address, texture.size));
  // Pool bytes with no texture registered feed the next texture made there.
  CHECK(resources.backing_feeds_texture(reinterpret_cast<std::uint64_t>(other_pool.memory), 64));
  resources.complete_gpu_writes(resources.gpu_write_generation());
  resources.release_pool(other_pool);
  resources.release_pool(pool);
}

void check_texture_usage(metal::Resources &resources) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  const auto storage_of = [](id<MTLTexture> texture) {
    while (texture.parentTexture)
      texture = texture.parentTexture;
    return texture;
  };
  nvn::TextureState single = make_texture(pool, 61);
  resources.register_texture(single);
  id<MTLTexture> plain = storage_of(resources.texture(single, nullptr));
  CHECK(plain != nil);
  CHECK((plain.usage & MTLTextureUsageRenderTarget) &&
        !(plain.usage & (MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView)));
  // Its first storage image write gives it writable storage in place of the
  // old, which the caller copies; a second asks for nothing.
  id<MTLTexture> previous = nil;
  id<MTLTexture> writable = resources.upgrade_for_image_write(single.serial, previous);
  CHECK(writable && previous == plain && writable != plain &&
        (writable.usage & MTLTextureUsageShaderWrite) && writable.width == plain.width &&
        writable.pixelFormat == plain.pixelFormat && writable.mipmapLevelCount == plain.mipmapLevelCount);
  CHECK(storage_of(resources.texture(single, nullptr)) == writable);
  CHECK(resources.upgrade_for_image_write(single.serial, previous) == writable && previous == nil);
  CHECK(resources.upgrade_for_image_write(single.serial + 1000, previous) == nil);
  nvn::TextureState chain = make_texture(pool, 62);
  chain.description.levels = 3;
  const auto layout = nvn::resource_layout(chain.description);
  CHECK(layout.has_value());
  if (layout) {
    chain.size = layout->total_size;
    chain.layer_size = layout->layer_stride;
    chain.offset = pool.size / 2;
    chain.memory = pool.memory + chain.offset;
    resources.register_texture(chain);
    id<MTLTexture> mips = storage_of(resources.texture(chain, nullptr));
    CHECK(mips != nil && mips.mipmapLevelCount == 3);
    CHECK((mips.usage & MTLTextureUsagePixelFormatView) != 0);
  }
  resources.release_pool(pool);
}

// A sampler ID registered again between two recorded bindings resolves, for
// each binding, to the parameters registered when it was recorded.
void check_sampler_registration_order(metal::Resources &resources) {
  nvn::SamplerParameters first{};
  first.min_filter = NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR;
  first.mag_filter = NVN_MAG_FILTER_LINEAR;
  first.wrap_s = first.wrap_t = first.wrap_r = NVN_WRAP_MODE_CLAMP_TO_EDGE;
  first.compare_mode = NVN_COMPARE_MODE_NONE;
  first.reduction_filter = NVN_SAMPLER_REDUCTION_AVERAGE;
  first.max_anisotropy = 1.0f;
  auto second = first;
  first.min_lod = first.max_lod = 0.0f;
  second.min_lod = second.max_lod = 2.0f;
  resources.register_sampler_descriptor(41, first);
  const auto first_binding = nvn::sampler_registrations.load();
  id<MTLSamplerState> level_zero = resources.sampler_for_id(41, first_binding);
  CHECK(level_zero != nil);
  resources.register_sampler_descriptor(41, second);
  const auto second_binding = nvn::sampler_registrations.load();
  id<MTLSamplerState> level_two = resources.sampler_for_id(41, second_binding);
  CHECK(level_two != nil && level_two != level_zero);
  CHECK(resources.sampler_for_id(41, first_binding) == level_zero);
  CHECK(resources.sampler_for_id(41) == level_two);
  resources.register_sampler_descriptor(41, first);
  CHECK(resources.sampler_for_id(41, second_binding) == level_two);
  CHECK(resources.sampler_for_id(41, first_binding) == level_zero);
  // Registering the same parameters again takes no number.
  const auto unchanged = nvn::sampler_registrations.load();
  resources.register_sampler_descriptor(41, first);
  CHECK(nvn::sampler_registrations.load() == unchanged);
}

// A resolved texture descriptor stays cached while the write clock moves for
// stores outside its guest range, and resolves again after a store into it.
void check_texture_slot_revalidation(metal::Resources &resources) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  nvn::TextureState state = make_texture(pool, 71);
  resources.register_texture(state);
  resources.register_texture_descriptor(20, state, nullptr);
  id<MTLTexture> first = resources.texture_for_id(20);
  CHECK(first != nil);
  const auto misses = [&] { return resources.texture_lookup_counts().misses; };
  const auto resolved = misses();
  CHECK(resources.texture_for_id(20) == first);
  CHECK(misses() == resolved);

  auto *elsewhere = pool.memory + pool.size - 64;
  const auto clock = armigo::memory::WriteTracker::now();
  armigo::memory::WriteTracker::note_store(elsewhere, 16);
  CHECK(armigo::memory::WriteTracker::now() != clock);
  CHECK(resources.texture_for_id(20) == first);
  CHECK(misses() == resolved);

  armigo::memory::WriteTracker::note_store(state.memory, 16);
  CHECK(resources.texture_for_id(20) != nil);
  CHECK(misses() == resolved + 1);
  resources.release_pool(pool);
}

// A cached descriptor lookup survives the registration of other IDs and moves
// with its own: re-registering an ID, replacing a descriptor pool or releasing
// a memory pool each make it resolve again.
void check_descriptor_stamps(metal::Resources &resources) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  nvn::TextureState first = make_texture(pool, 9101);
  nvn::TextureState second = make_texture(pool, 9102);
  second.offset = 4096;
  second.memory = pool.memory + second.offset;
  resources.register_texture(first);
  resources.register_texture(second);
  resources.register_texture_descriptor(40, first, nullptr);
  resources.register_texture_descriptor(41, second, nullptr);
  using Stamp = metal::Resources::DescriptorStamp;
  Stamp stamp_40, stamp_41;
  id<MTLTexture> view_40 = resources.texture_for_id(40, stamp_40);
  id<MTLTexture> view_41 = resources.texture_for_id(41, stamp_41);
  CHECK(view_40 != nil && view_41 != nil && view_40 != view_41);
  CHECK(stamp_40.valid && stamp_41.valid);
  CHECK(resources.texture_stamp_current(40, stamp_40));
  CHECK(resources.texture_stamp_current(41, stamp_41));
  // A lookup that found nothing has no stamp.
  Stamp missing;
  CHECK(resources.texture_for_id(42, missing) == nil && !missing.valid);
  CHECK(!resources.texture_stamp_current(42, missing));

  // Registering other IDs moves the global generation but not these stamps.
  const auto generation = resources.registration_generation();
  resources.register_texture_descriptor(42, second, nullptr);
  resources.register_texture_descriptor(43, first, nullptr);
  CHECK(resources.registration_generation() != generation);
  CHECK(resources.texture_stamp_current(40, stamp_40));
  CHECK(resources.texture_stamp_current(41, stamp_41));

  // The same ID registered again, even with the same texture, resolves again;
  // with another texture it yields that texture.
  resources.register_texture_descriptor(40, second, nullptr);
  CHECK(!resources.texture_stamp_current(40, stamp_40));
  CHECK(resources.texture_stamp_current(41, stamp_41));
  Stamp renewed_40;
  id<MTLTexture> view_second = resources.texture_for_id(40, renewed_40);
  CHECK(view_second != nil && view_second != view_40 && renewed_40.valid);
  CHECK(renewed_40.slot != stamp_40.slot);
  CHECK(resources.texture_stamp_current(40, renewed_40));
  CHECK(resources.texture_for_id(40) == view_second);

  Stamp stamp_43;
  CHECK(resources.texture_for_id(43, stamp_43) != nil);
  CHECK(resources.texture_stamp_current(43, stamp_43));

  // Replacing the descriptor pool drops every ID: nothing resolved before
  // stands, and the old IDs are not registered in the new pool.
  nvn::DescriptorPoolState first_pool{&pool, 0, 300};
  nvn::DescriptorPoolState second_pool{&pool, 16384, 300};
  CHECK(resources.register_texture_pool(first_pool));
  resources.register_texture_descriptor(44, first, nullptr);
  Stamp stamp_44;
  CHECK(resources.texture_for_id(44, stamp_44) != nil);
  CHECK(resources.texture_stamp_current(44, stamp_44));
  resources.release_texture_pool(first_pool);
  CHECK(!resources.texture_stamp_current(44, stamp_44));
  CHECK(resources.register_texture_pool(second_pool));
  CHECK(resources.texture_for_id(44, stamp_44) == nil && !stamp_44.valid);
  resources.register_texture_descriptor(44, first, nullptr);
  CHECK(resources.texture_for_id(44, stamp_44) != nil);
  CHECK(resources.texture_stamp_current(44, stamp_44));
  // Re-registering the texture entry behind an ID moves its serial stamp,
  // though the ID itself was not registered again.
  resources.register_texture(first);
  CHECK(!resources.texture_stamp_current(44, stamp_44));
  CHECK(resources.texture_for_id(44, stamp_44) != nil);
  CHECK(resources.texture_stamp_current(44, stamp_44));
  // Releasing any memory pool, even one that does not hold the descriptor
  // pool, moves the rare stamp.
  {
    Pages other_pages(1);
    CHECK(other_pages.bytes() != nullptr);
    nvn::MemoryPoolState other{};
    other.memory = reinterpret_cast<std::uint8_t *>(other_pages.bytes());
    other.size = other_pages.size();
    other.flags =
        NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
    CHECK(resources.register_pool(other));
    CHECK(resources.texture_for_id(44, stamp_44) != nil);
    CHECK(resources.texture_stamp_current(44, stamp_44));
    resources.release_pool(other);
    CHECK(!resources.texture_stamp_current(44, stamp_44));
    CHECK(resources.texture_for_id(44, stamp_44) != nil);
    CHECK(resources.texture_stamp_current(44, stamp_44));
  }
  // The pool's memory going away moves every stamp (the rare stamp).
  resources.release_texture_pool(second_pool);
  resources.release_pool(pool);
  CHECK(!resources.texture_stamp_current(44, stamp_44));
  resources.release_texture(first);
  resources.release_texture(second);
}

// The thread's sampler lookups follow the same rules: another ID's registration
// does not make the cached sampler resolve again, the ID's own registration and
// a replaced sampler pool do, and a binding recorded earlier still gets the
// parameters registered then.
void check_sampler_stamps(metal::Resources &resources) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  CHECK(resources.register_pool(pool));
  nvn::DescriptorPoolState first_pool{&pool, 0, 300};
  nvn::DescriptorPoolState second_pool{&pool, 16384, 300};
  CHECK(resources.register_sampler_pool(first_pool));
  nvn::SamplerParameters nearest{};
  nearest.min_filter = NVN_MIN_FILTER_NEAREST;
  nearest.mag_filter = NVN_MAG_FILTER_NEAREST;
  nearest.wrap_s = nearest.wrap_t = nearest.wrap_r = NVN_WRAP_MODE_CLAMP_TO_EDGE;
  nearest.compare_mode = NVN_COMPARE_MODE_NONE;
  nearest.reduction_filter = NVN_SAMPLER_REDUCTION_AVERAGE;
  nearest.max_lod = 1000.0f;
  nearest.max_anisotropy = 1.0f;
  auto linear = nearest;
  linear.min_filter = NVN_MIN_FILTER_LINEAR;
  linear.mag_filter = NVN_MAG_FILTER_LINEAR;
  resources.register_sampler_descriptor(60, nearest);
  resources.register_sampler_descriptor(61, linear);
  const auto misses = [&] { return resources.texture_lookup_counts().sampler_misses; };
  const auto before_binding = nvn::sampler_registrations.load();
  id<MTLSamplerState> sampler_60 = resources.sampler_for_id(60, before_binding);
  id<MTLSamplerState> sampler_61 = resources.sampler_for_id(61, before_binding);
  CHECK(sampler_60 != nil && sampler_61 != nil && sampler_60 != sampler_61);
  const auto resolved = misses();
  CHECK(resources.sampler_for_id(60, before_binding) == sampler_60);
  CHECK(resources.sampler_for_id(61, before_binding) == sampler_61);
  CHECK(misses() == resolved);

  // Other IDs, and the same parameters again, leave the cached samplers alone.
  resources.register_sampler_descriptor(62, linear);
  resources.register_sampler_descriptor(61, linear);
  CHECK(resources.sampler_for_id(60, nvn::sampler_registrations.load()) == sampler_60);
  CHECK(resources.sampler_for_id(61, nvn::sampler_registrations.load()) == sampler_61);
  CHECK(misses() == resolved);

  // An ID registered again resolves to the new sampler for later bindings and
  // to the old one for bindings recorded before.
  resources.register_sampler_descriptor(60, linear);
  const auto after_binding = nvn::sampler_registrations.load();
  id<MTLSamplerState> renewed = resources.sampler_for_id(60, after_binding);
  CHECK(renewed != nil && renewed != sampler_60);
  CHECK(resources.sampler_for_id(60, before_binding) == sampler_60);
  CHECK(resources.sampler_for_id(60, after_binding) == renewed);
  CHECK(resources.sampler_for_id(60) == renewed);
  CHECK(resources.sampler_for_id(61, after_binding) == sampler_61);

  // A replaced sampler pool drops every ID, cached or not.
  resources.release_sampler_pool(first_pool);
  CHECK(resources.sampler_for_id(60, after_binding) == nil);
  CHECK(resources.sampler_for_id(61, after_binding) == nil);
  CHECK(resources.register_sampler_pool(second_pool));
  CHECK(resources.sampler_for_id(61, after_binding) == nil);
  resources.register_sampler_descriptor(61, nearest);
  id<MTLSamplerState> replaced = resources.sampler_for_id(61, nvn::sampler_registrations.load());
  CHECK(replaced != nil && replaced != sampler_61);
  resources.release_sampler_pool(second_pool);
  resources.release_pool(pool);
}

// A recorded descriptor can outlive its texture object. Recreating its storage
// after pool retirement and a superseded-entry sweep must move the generation
// even though the descriptor ID is never registered again.
void check_recreated_texture_generation(id<MTLDevice> device,
                                        id<MTLCommandQueue> queue) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes()) return;
  metal::Resources resources(device);
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  auto state = make_texture(pool, 1);
  std::vector<std::byte> red(64), green(64);
  for (unsigned pixel = 0; pixel < 16; ++pixel) {
    red[pixel * 4] = red[pixel * 4 + 3] = std::byte{255};
    green[pixel * 4 + 1] = green[pixel * 4 + 3] = std::byte{255};
  }
  const auto transfer = whole_rgba_transfer();
  CHECK(nvn::transfer_pool_texels(state, transfer, red.data(), true));
  resources.register_texture(state);
  resources.register_texture_descriptor(20, state, nullptr);
  id<MTLTexture> first = resources.texture_for_id(20);
  CHECK(first != nil);
  CHECK(read_native_rgba(resources, device, queue, first) == red);

  const auto before_release = resources.registration_generation();
  resources.release_texture(state);
  resources.release_pool(pool);
  CHECK(resources.registration_generation() != before_release);
  CHECK(resources.register_pool(pool));
  // Registrations over unrelated storage drive the bounded retirement sweep.
  // The original descriptor remains registered throughout.
  for (std::uint64_t serial = 2; serial < 258; ++serial) {
    auto other = make_texture(pool, serial);
    other.offset = pool.size / 2;
    other.memory = pool.memory + other.offset;
    resources.register_texture(other);
  }
  CHECK(nvn::transfer_pool_texels(state, transfer, green.data(), true));
  const auto before_recreation = resources.registration_generation();
  id<MTLTexture> recreated = resources.texture_for_id(20);
  CHECK(recreated != nil && recreated != first);
  CHECK(resources.registration_generation() != before_recreation);
  CHECK(read_native_rgba(resources, device, queue, recreated) == green);
  CHECK(read_native_rgba(resources, device, queue, first) == red);
  resources.release_pool(pool);
}

// A texture object re-created over the storage another object resolved takes
// that storage over when the sweep drops the older one. The newer entry then
// has storage but never derived its format and layout, and sampling it must
// not depend on them being derived: its sampled view over the inherited
// storage is the same as the older object's.
void check_sampled_view_over_inherited_storage(id<MTLDevice> device,
                                               id<MTLCommandQueue> queue) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes()) return;
  metal::Resources resources(device);
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  auto older = make_texture(pool, 1);
  std::vector<std::byte> red(64);
  for (unsigned pixel = 0; pixel < 16; ++pixel)
    red[pixel * 4] = red[pixel * 4 + 3] = std::byte{255};
  CHECK(nvn::transfer_pool_texels(older, whole_rgba_transfer(), red.data(), true));
  resources.register_texture(older);
  resources.register_texture_descriptor(20, older, nullptr);
  id<MTLTexture> first = resources.texture_for_id(20);
  CHECK(first != nil);

  auto newer = make_texture(pool, 2);
  resources.register_texture(newer);
  resources.register_texture_descriptor(21, newer, nullptr);
  // Registrations over unrelated storage drive the bounded sweep, which hands
  // the older entry's storage to the newer one and drops the older.
  for (std::uint64_t serial = 3; serial < 259; ++serial) {
    auto other = make_texture(pool, serial);
    other.offset = pool.size / 2;
    other.memory = pool.memory + other.offset;
    resources.register_texture(other);
  }
  id<MTLTexture> inherited = resources.texture_for_id(21);
  CHECK(inherited != nil);
  if (inherited) CHECK(read_native_rgba(resources, device, queue, inherited) == red);
  nvn::TextureViewState view{};
  view.sets_format = 1;
  view.format = NVN_FORMAT_RGBA8;
  resources.register_texture_descriptor(22, newer, &view);
  CHECK(resources.texture_for_id(22) != nil);
  resources.release_pool(pool);
}

// An upload certifies the texture for a later texel flush only when its pool
// has a CPU mapping: a flush of a pool without one returns at once, so no
// certificate is made for it.
void check_upload_certificates_follow_cpu_mapping(id<MTLDevice> device) {
  Pages cached_pages(4), hidden_pages(4);
  CHECK(cached_pages.bytes() != nullptr && hidden_pages.bytes() != nullptr);
  if (!cached_pages.bytes() || !hidden_pages.bytes()) return;
  metal::Resources resources(device);
  nvn::MemoryPoolState cached{}, hidden{};
  cached.memory = reinterpret_cast<std::uint8_t *>(cached_pages.bytes());
  cached.size = cached_pages.size();
  cached.flags = NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  hidden.memory = reinterpret_cast<std::uint8_t *>(hidden_pages.bytes());
  hidden.size = hidden_pages.size();
  hidden.flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED |
                 NVN_MEMORY_POOL_FLAGS_COMPRESSIBLE;
  CHECK(resources.register_pool(cached));
  CHECK(resources.register_pool(hidden));
  const auto before = armigo::graphics::texflush::certificate_count();
  auto visible = make_texture(cached, 7101);
  resources.register_texture(visible);
  CHECK(resources.texture(visible, nullptr) != nil);
  CHECK(armigo::graphics::texflush::certificate_count() == before + 1);
  auto target = make_texture(hidden, 7102);
  resources.register_texture(target);
  CHECK(resources.texture(target, nullptr) != nil);
  CHECK(resources.flush_uploads());
  CHECK(armigo::graphics::texflush::certificate_count() == before + 1);
  resources.release_texture(visible);
  resources.release_texture(target);
  resources.release_pool(cached);
  resources.release_pool(hidden);
  armigo::graphics::texflush::forget(7101);
}

void check_texture_resources(id<MTLDevice> device, metal::Resources &resources,
                             id<MTLCommandQueue> queue) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(!armigo::memory::WriteTracker::tracked(pool.memory, pool.size));
  CHECK(resources.register_pool(pool));
  CHECK(armigo::memory::WriteTracker::tracked(pool.memory, pool.size));

  nvn::TextureState state = make_texture(pool, 41);
  CHECK(state.size != 0);
  CHECK(state.size <= pool.size);
  resources.register_texture(state);
  const auto transfer = whole_rgba_transfer();
  std::vector<std::byte> pattern(64);
  for (std::size_t i = 0; i < pattern.size(); ++i)
    pattern[i] = std::byte((i * 13 + 7) & 0xff);
  CHECK(resources.write_texels(state, transfer, pattern.data()));

  std::vector<std::byte> round_trip(64);
  CHECK(resources.read_texels(state, transfer, round_trip.data()));
  CHECK(round_trip == pattern);

  id<MTLTexture> texture = resources.texture(state, nullptr);
  CHECK(texture != nil);
  CHECK(texture.width == 4);
  CHECK(texture.height == 4);
  CHECK(read_native_rgba(resources, device, queue, texture) == pattern);

  std::vector<std::byte> uncached_pattern(64);
  for (std::size_t i = 0; i < uncached_pattern.size(); ++i)
    uncached_pattern[i] = std::byte((i * 29 + 3) & 0xff);
  CHECK(nvn::transfer_pool_texels(state, transfer, uncached_pattern.data(),
                                  true));
  CHECK(resources.texture(state, nullptr) == texture);
  CHECK(read_native_rgba(resources, device, queue, texture) == uncached_pattern);
  pattern = uncached_pattern;

  // A failed protection attempt cannot certify a snapshot, even when no
  // subsequent store changes the global clock on already writable pages.
  armigo::memory::WriteTracker::set_protect_hook([](void* address, std::size_t size, int protection) {
    return (protection & PROT_WRITE) && mprotect(address, size, protection) == 0;
  });
  pattern[0] ^= std::byte{0x7f};
  CHECK(nvn::transfer_pool_texels(state, transfer, pattern.data(), true));
  CHECK(resources.texture(state, nullptr) == texture);
  CHECK(read_native_rgba(resources, device, queue, texture) == pattern);
  pattern[1] ^= std::byte{0x3f};
  CHECK(nvn::transfer_pool_texels(state, transfer, pattern.data(), true));
  CHECK(resources.texture(state, nullptr) == texture);
  CHECK(read_native_rgba(resources, device, queue, texture) == pattern);
  armigo::memory::WriteTracker::set_protect_hook(nullptr);

  std::reverse(pattern.begin(), pattern.end());
  CHECK(resources.write_texels(state, transfer, pattern.data()));
  CHECK(read_native_rgba(resources, device, queue, texture) == pattern);

  resources.register_texture_descriptor(7, state, nullptr);
  CHECK(resources.texture_for_id(7).parentTexture == texture);

  std::vector<std::byte> gpu_pattern(64);
  for (std::size_t i = 0; i < gpu_pattern.size(); ++i)
    gpu_pattern[i] = std::byte((i * 17 + 91) & 0xff);
  CHECK(nvn::transfer_pool_texels(state, transfer, gpu_pattern.data(), true));
  std::vector<std::byte> encoded_gpu(state.size);
  std::memcpy(encoded_gpu.data(), state.memory, encoded_gpu.size());
  CHECK(nvn::transfer_pool_texels(state, transfer, pattern.data(), true));
  CHECK(resources.texture(state, nullptr) == texture);
  CHECK(read_native_rgba(resources, device, queue, texture) == pattern);

  metal::BufferSpan backing = resources.buffer_at(
      reinterpret_cast<std::uint64_t>(state.memory), state.size);
  CHECK(backing.buffer != nil);
  id<MTLBuffer> gpu_source =
      [device newBufferWithBytes:encoded_gpu.data()
                          length:encoded_gpu.size()
                         options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> backing_write = [queue commandBuffer];
  id<MTLBlitCommandEncoder> backing_blit = [backing_write blitCommandEncoder];
  [backing_blit copyFromBuffer:gpu_source
                  sourceOffset:0
                      toBuffer:backing.buffer
             destinationOffset:backing.offset
                          size:encoded_gpu.size()];
  [backing_blit endEncoding];
  [backing_write commit];
  unsigned backing_completion_calls = 0;
  resources.set_completion_hook([&] {
    ++backing_completion_calls;
    return complete(backing_write);
  });
  resources.note_gpu_write(reinterpret_cast<std::uint64_t>(state.memory),
                           state.size);
  CHECK(resources.texture_for_id(7).parentTexture == texture);
  CHECK(backing_completion_calls == 1);
  CHECK(read_native_rgba(resources, device, queue, texture) == gpu_pattern);

  nvn::TextureViewState view{};
  view.sets_levels = 1;
  view.base_level = 0;
  view.level_count = 1;
  view.sets_layers = 1;
  view.min_layer = 0;
  view.layer_count = 1;
  resources.register_texture_descriptor(8, state, &view);
  id<MTLTexture> viewed = resources.texture_for_id(8);
  CHECK(viewed != nil);
  CHECK(viewed.parentTexture != nil);

  nvn::SamplerParameters sampler{};
  sampler.min_filter = NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR;
  sampler.mag_filter = NVN_MAG_FILTER_LINEAR;
  sampler.wrap_s = NVN_WRAP_MODE_REPEAT;
  sampler.wrap_t = NVN_WRAP_MODE_CLAMP_TO_EDGE;
  sampler.wrap_r = NVN_WRAP_MODE_MIRRORED_REPEAT;
  sampler.compare_mode = NVN_COMPARE_MODE_NONE;
  sampler.reduction_filter = NVN_SAMPLER_REDUCTION_AVERAGE;
  sampler.min_lod = 0.0f;
  sampler.max_lod = 16.0f;
  sampler.max_anisotropy = 1.0f;
  resources.register_sampler_descriptor(3, sampler);
  CHECK(resources.sampler_for_id(3) != nil);

  // A texture of another type than a shader declares is bound as a view in
  // the declared type, kept in the descriptor's slot; a type without such a
  // view keeps the texture.
  MTLTextureDescriptor *cube_descriptor =
      [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                            size:8
                                                       mipmapped:YES];
  id<MTLTexture> cube = [device newTextureWithDescriptor:cube_descriptor];
  id<MTLTexture> cube_array = resources.fitted_texture(7, cube, MTLTextureTypeCubeArray);
  CHECK(cube_array.textureType == MTLTextureTypeCubeArray && cube_array.arrayLength == 1 &&
        cube_array.mipmapLevelCount == cube.mipmapLevelCount && cube_array.parentTexture == cube);
  CHECK(resources.fitted_texture(7, cube, MTLTextureTypeCubeArray) == cube_array);
  id<MTLTexture> face = resources.fitted_texture(7, cube, MTLTextureType2D);
  CHECK(face.textureType == MTLTextureType2D && face.parentTexture == cube);
  MTLTextureDescriptor *volume_descriptor = [MTLTextureDescriptor new];
  volume_descriptor.textureType = MTLTextureType3D;
  volume_descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
  volume_descriptor.width = volume_descriptor.height = volume_descriptor.depth = 4;
  id<MTLTexture> volume = [device newTextureWithDescriptor:volume_descriptor];
  CHECK(resources.fitted_texture(7, volume, MTLTextureType2D) == nil);

  nvn::TextureState rgbx = make_texture(pool, 42);
  rgbx.description.format = NVN_FORMAT_RGBX8;
  rgbx.swizzle =
      nvn::pack_swizzle(NVN_TEXTURE_SWIZZLE_A, NVN_TEXTURE_SWIZZLE_G,
                        NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_R);
  resources.register_texture(rgbx);
  resources.register_texture_descriptor(9, rgbx, nullptr);
  id<MTLTexture> rgbx_view = resources.texture_for_id(9);
  CHECK(rgbx_view != nil);
  if (rgbx_view) {
    const auto channels = rgbx_view.swizzle;
    CHECK(channels.red == MTLTextureSwizzleOne);
    CHECK(channels.green == MTLTextureSwizzleGreen);
    CHECK(channels.blue == MTLTextureSwizzleBlue);
    CHECK(channels.alpha == MTLTextureSwizzleRed);
  }

  CHECK(resources.flush_uploads());
  id<MTLCommandBuffer> pending = [queue commandBuffer];
  auto *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = texture;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
  id<MTLRenderCommandEncoder> render =
      [pending renderCommandEncoderWithDescriptor:pass];
  CHECK(render != nil);
  [render endEncoding];
  [pending commit];
  unsigned completion_calls = 0;
  resources.set_completion_hook([&] {
    ++completion_calls;
    return complete(pending);
  });
  resources.note_gpu_write(state);
  // Invalidation does not read a GPU-written texture back; an explicit texel
  // read waits for the write and reads it.
  resources.invalidate_pool(pool, 0, state.size);
  CHECK(completion_calls == 0);

  std::fill(round_trip.begin(), round_trip.end(), std::byte{0});
  CHECK(resources.read_texels(state, transfer, round_trip.data()));
  CHECK(completion_calls == 1);
  for (std::size_t pixel = 0; pixel < 16; ++pixel) {
    const auto channel = [&](std::size_t component) {
      return std::to_integer<unsigned>(round_trip[pixel * 4 + component]);
    };
    CHECK(channel(0) >= 63 && channel(0) <= 64);
    CHECK(channel(1) >= 127 && channel(1) <= 128);
    CHECK(channel(2) >= 191 && channel(2) <= 192);
    CHECK(channel(3) == 255);
  }

  id<MTLCommandBuffer> partial_pending = [queue commandBuffer];
  auto *partial_pass = [MTLRenderPassDescriptor renderPassDescriptor];
  partial_pass.colorAttachments[0].texture = texture;
  partial_pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  partial_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  partial_pass.colorAttachments[0].clearColor =
      MTLClearColorMake(0.125, 0.25, 0.5, 1.0);
  id<MTLRenderCommandEncoder> partial_render =
      [partial_pending renderCommandEncoderWithDescriptor:partial_pass];
  CHECK(partial_render != nil);
  [partial_render endEncoding];
  [partial_pending commit];
  resources.set_completion_hook([&] { return complete(partial_pending); });
  resources.note_gpu_write(state);
  const std::array<std::byte, 4> cpu_pixel{std::byte{0xff}, std::byte{0x00},
                                           std::byte{0x00}, std::byte{0xff}};
  nvn::TexelTransfer one_pixel;
  one_pixel.region = {0, 0, 0, 1, 1, 1};
  one_pixel.row_stride = 4;
  one_pixel.image_stride = 4;
  CHECK(nvn::transfer_pool_texels(
      state, one_pixel, const_cast<std::byte *>(cpu_pixel.data()), true));
  resources.flush_pool(pool, 0, cpu_pixel.size());
  const auto partial_result = read_native_rgba(resources, device, queue, texture);
  CHECK(std::equal(cpu_pixel.begin(), cpu_pixel.end(), partial_result.begin()));
  for (std::size_t pixel = 1; pixel < 16; ++pixel) {
    CHECK(std::to_integer<unsigned>(partial_result[pixel * 4]) >= 31 &&
          std::to_integer<unsigned>(partial_result[pixel * 4]) <= 32);
    CHECK(std::to_integer<unsigned>(partial_result[pixel * 4 + 1]) >= 63 &&
          std::to_integer<unsigned>(partial_result[pixel * 4 + 1]) <= 64);
    CHECK(std::to_integer<unsigned>(partial_result[pixel * 4 + 2]) >= 127 &&
          std::to_integer<unsigned>(partial_result[pixel * 4 + 2]) <= 128);
    CHECK(std::to_integer<unsigned>(partial_result[pixel * 4 + 3]) == 255);
  }

  // A flush over GPU-written content the CPU did not store into since the
  // texture's last snapshot keeps what the GPU wrote.
  id<MTLCommandBuffer> gpu_only_pending = [queue commandBuffer];
  auto *gpu_only_pass = [MTLRenderPassDescriptor renderPassDescriptor];
  gpu_only_pass.colorAttachments[0].texture = texture;
  gpu_only_pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  gpu_only_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  gpu_only_pass.colorAttachments[0].clearColor = MTLClearColorMake(0.75, 0.5, 0.25, 1.0);
  id<MTLRenderCommandEncoder> gpu_only_render =
      [gpu_only_pending renderCommandEncoderWithDescriptor:gpu_only_pass];
  CHECK(gpu_only_render != nil);
  [gpu_only_render endEncoding];
  [gpu_only_pending commit];
  [gpu_only_pending waitUntilCompleted];
  resources.set_completion_hook([&] { return complete(gpu_only_pending); });
  resources.note_gpu_write(state);
  resources.flush_pool(pool, 0, state.size);
  const auto gpu_only_result = read_native_rgba(resources, device, queue, texture);
  for (std::size_t pixel = 0; pixel < 16; ++pixel) {
    CHECK(std::to_integer<unsigned>(gpu_only_result[pixel * 4]) >= 191 &&
          std::to_integer<unsigned>(gpu_only_result[pixel * 4]) <= 192);
    CHECK(std::to_integer<unsigned>(gpu_only_result[pixel * 4 + 1]) >= 127 &&
          std::to_integer<unsigned>(gpu_only_result[pixel * 4 + 1]) <= 128);
    CHECK(std::to_integer<unsigned>(gpu_only_result[pixel * 4 + 2]) >= 63 &&
          std::to_integer<unsigned>(gpu_only_result[pixel * 4 + 2]) <= 64);
  }

  resources.release_texture(state);
  resources.release_texture(rgbx);
  nvn::MemoryPoolState copied_pool = pool;
  resources.release_pool(copied_pool);
  CHECK(resources.buffer_at(reinterpret_cast<std::uint64_t>(pool.memory), 4)
            .buffer == nil);
  CHECK(resources.texture(state, nullptr) == nil);
  resources.set_completion_hook({});
  CHECK(!armigo::memory::WriteTracker::tracked(pool.memory, pool.size));
}

void check_cached_pool_flush(id<MTLDevice> device, metal::Resources &resources,
                             id<MTLCommandQueue> queue) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  armigo::memory::WriteTracker::add_range(pool.memory, pool.size);
  CHECK(resources.register_pool(pool));

  nvn::TextureState state = make_texture(pool, 51);
  resources.register_texture(state);
  auto alias = state;
  alias.serial = 52;
  resources.register_texture(alias);
  nvn::DescriptorPoolState first_pool{&pool, 0, 300};
  nvn::DescriptorPoolState second_pool{&pool, 16384, 300};
  CHECK(resources.register_texture_pool(first_pool));
  CHECK(resources.register_sampler_pool(first_pool));
  CHECK(!resources.register_texture_pool(second_pool));
  CHECK(!resources.register_sampler_pool(second_pool));
  nvn::SamplerParameters pooled{};
  pooled.min_filter = NVN_MIN_FILTER_LINEAR;
  pooled.mag_filter = NVN_MAG_FILTER_LINEAR;
  pooled.wrap_s = pooled.wrap_t = pooled.wrap_r = NVN_WRAP_MODE_CLAMP_TO_EDGE;
  pooled.compare_mode = NVN_COMPARE_MODE_NONE;
  pooled.reduction_filter = NVN_SAMPLER_REDUCTION_AVERAGE;
  pooled.max_lod = 1000.0f;
  pooled.max_anisotropy = 1.0f;
  resources.register_sampler_descriptor(5, pooled);
  CHECK(resources.sampler_for_id(5) != nil);
  const auto transfer = whole_rgba_transfer();
  std::vector<std::byte> before(64, std::byte{0x19});
  std::vector<std::byte> after(64, std::byte{0xc7});
  CHECK(resources.write_texels(state, transfer, before.data()));
  id<MTLTexture> texture = resources.texture(state, nullptr);
  CHECK(texture != nil);
  id<MTLTexture> aliased_texture = resources.texture(alias, nullptr);
  // Objects over the same storage with the same description share it.
  CHECK(aliased_texture == texture);
  resources.register_texture_descriptor(256, state, nullptr);
  CHECK(resources.texture_for_id(256).parentTexture == texture);
  CHECK(read_native_rgba(resources, device, queue, texture) == before);

  CHECK(nvn::transfer_pool_texels(state, transfer, after.data(), true));
  CHECK(resources.texture(state, nullptr) == texture);
  CHECK(read_native_rgba(resources, device, queue, texture) == before);
  resources.flush_pool(pool, 0, state.size);
  CHECK(read_native_rgba(resources, device, queue, texture) == after);

  id<MTLCommandBuffer> source_clear = [queue commandBuffer];
  auto *source_pass = [MTLRenderPassDescriptor renderPassDescriptor];
  source_pass.colorAttachments[0].texture = texture;
  source_pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  source_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  source_pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 1, 1);
  auto source_encoder =
      [source_clear renderCommandEncoderWithDescriptor:source_pass];
  CHECK(source_encoder != nil);
  [source_encoder endEncoding];
  [source_clear commit];
  resources.set_completion_hook([&] { return complete(source_clear); });
  resources.note_gpu_write(state);
  std::array<std::byte, 4> red{std::byte{255}, std::byte{0}, std::byte{0},
                               std::byte{255}};
  nvn::TexelTransfer pixel;
  pixel.region = {0, 0, 0, 1, 1, 1};
  CHECK(nvn::transfer_pool_texels(state, pixel, red.data(), true));
  CHECK(resources.write_texels(
      alias, pixel, reinterpret_cast<const std::byte *>(state.memory)));
  std::vector<std::byte> coherent_cpu(64);
  CHECK(resources.read_texels(state, transfer, coherent_cpu.data()));
  CHECK(std::equal(red.begin(), red.end(), coherent_cpu.begin()));
  CHECK(resources.texture(state, nullptr) == texture);
  const auto preserved_source = read_native_rgba(resources, device, queue, texture);
  CHECK(std::equal(red.begin(), red.end(), preserved_source.begin()));
  CHECK(preserved_source[4] == std::byte{0} &&
        preserved_source[6] == std::byte{255});

  id<MTLCommandBuffer> clear = [queue commandBuffer];
  auto *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = texture;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 1, 0, 1);
  auto encoder = [clear renderCommandEncoderWithDescriptor:pass];
  CHECK(encoder != nil);
  [encoder endEncoding];
  [clear commit];
  resources.set_completion_hook([&] { return complete(clear); });
  resources.note_gpu_write(state);
  CHECK(resources.texture(alias, nullptr) == texture);
  CHECK(resources.read_texels(alias, transfer, coherent_cpu.data()));
  CHECK(coherent_cpu[0] == std::byte{0} && coherent_cpu[1] == std::byte{255});
  CHECK(resources.texture(alias, nullptr) == aliased_texture);
  auto pixels = read_native_rgba(resources, device, queue, aliased_texture);
  for (std::size_t pixel = 0; pixel < 16; ++pixel) {
    CHECK(pixels[pixel * 4] == std::byte{0});
    CHECK(pixels[pixel * 4 + 1] == std::byte{255});
    CHECK(pixels[pixel * 4 + 2] == std::byte{0});
    CHECK(pixels[pixel * 4 + 3] == std::byte{255});
  }
  CHECK(resources.write_texels(alias, transfer, before.data()));
  CHECK(resources.texture(state, nullptr) == texture);
  CHECK(read_native_rgba(resources, device, queue, texture) == before);
  CHECK(resources.sampler_for_id(5) != nil);
  resources.release_texture_pool(first_pool);
  resources.release_sampler_pool(first_pool);
  CHECK(resources.register_texture_pool(second_pool));
  CHECK(resources.register_sampler_pool(second_pool));
  CHECK(resources.texture_for_id(256) == nil);
  // Releasing the pool retires its samplers, cached lookups included.
  CHECK(resources.sampler_for_id(5) == nil);
  resources.release_texture(state);
  resources.release_texture(alias);
  resources.release_pool(pool);
  // Pool release also retires its descriptor namespaces before storage reuse.
  CHECK(resources.register_pool(pool));
  CHECK(resources.register_texture_pool(first_pool));
  resources.release_texture_pool(first_pool);
  resources.release_pool(pool);
  resources.set_completion_hook({});
  armigo::memory::WriteTracker::remove_range(pool.memory);
}

// A texture in a CPU-cached pool that the CPU writes after its first use, with
// no flush: by default the write waits for a flush call; with the title policy
// for natively flushing titles it shows at the texture's next use.
void check_cached_pool_cpu_write_after_creation(id<MTLDevice> device, metal::Resources &resources,
                                                id<MTLCommandQueue> queue) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes())
    return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  CHECK(armigo::memory::WriteTracker::tracked(pool.memory, pool.size));
  nvn::TextureState state = make_texture(pool, 461);
  resources.register_texture(state);
  const auto transfer = whole_rgba_transfer();
  const auto pattern_of = [](unsigned seed) {
    std::vector<std::byte> pattern(64);
    for (std::size_t i = 0; i < pattern.size(); ++i)
      pattern[i] = std::byte((i * seed + 5) & 0xff);
    return pattern;
  };
  auto first = pattern_of(11), second = pattern_of(47);
  CHECK(nvn::transfer_pool_texels(state, transfer, first.data(), true));
  id<MTLTexture> texture = resources.texture(state, nullptr);
  CHECK(texture != nil);
  if (texture) CHECK(read_native_rgba(resources, device, queue, texture) == first);
  // The second write lands after the texture's first use. By default a
  // cached pool's writes wait for a flush; a title whose cache maintenance
  // runs natively selects the policy that takes them at the next use.
  CHECK(nvn::transfer_pool_texels(state, transfer, second.data(), true));
  texture = resources.texture(state, nullptr);
  CHECK(texture != nil && read_native_rgba(resources, device, queue, texture) == first);
  nvn::set_title_policy({.cpu_cached_texture_writes = true});
  texture = resources.texture(state, nullptr);
  CHECK(texture != nil);
  if (texture) CHECK(read_native_rgba(resources, device, queue, texture) == second);
  nvn::set_title_policy({});
  resources.release_texture(state);
  resources.release_pool(pool);
}

void check_depth_storage(id<MTLDevice> device, metal::Resources &resources,
                          id<MTLCommandQueue> queue) {
  Pages pages(2);
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  CHECK(resources.register_pool(pool));
  auto state = make_texture(pool, 60);
  state.description.format = NVN_FORMAT_DEPTH24_STENCIL8;
  state.description.flags = NVN_TEXTURE_FLAGS_LINEAR;
  state.description.stride = 256;
  const auto layout = nvn::resource_layout(state.description);
  CHECK(layout.has_value());
  if (!layout) return;
  state.size = layout->total_size;
  state.layer_size = layout->layer_stride;
  resources.register_texture(state);
  const auto transfer = whole_rgba_transfer();
  std::array<std::uint32_t, 16> initial{};
  initial.fill(0x80000027u);
  CHECK(resources.write_texels(state, transfer,
                                reinterpret_cast<const std::byte *>(initial.data())));
  id<MTLTexture> storage = resources.texture(state, nullptr);
  CHECK(storage != nil && storage.pixelFormat == MTLPixelFormatDepth32Float_Stencil8);
  CHECK(storage.buffer == nil);
  if (!storage) return;
  nvn::TextureViewState stencil{};
  stencil.sets_depth_stencil = 1;
  stencil.depth_stencil_mode = NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL;
  resources.register_texture_descriptor(60, state, &stencil);
  CHECK(resources.texture_for_id(60).pixelFormat == MTLPixelFormatX32_Stencil8);
  CHECK(resources.texture(state, &stencil).pixelFormat == MTLPixelFormatDepth32Float_Stencil8);
  resources.register_texture_descriptor(61, state, nullptr);
  CHECK(resources.texture_for_id(61).pixelFormat == MTLPixelFormatDepth32Float_Stencil8);
  stencil.sets_swizzle = 1;
  stencil.swizzle[0] = stencil.swizzle[1] = stencil.swizzle[2] = NVN_TEXTURE_SWIZZLE_R;
  stencil.swizzle[3] = NVN_TEXTURE_SWIZZLE_ONE;
  resources.register_texture_descriptor(62, state, &stencil);
  CHECK(resources.texture_for_id(62).swizzle.green == MTLTextureSwizzleRed);

  // Read uploaded planes from the GPU before replacing them with a render clear.
  id<MTLBuffer> planes = [device newBufferWithLength:2048 options:MTLResourceStorageModeShared];
  CHECK(resources.flush_uploads());
  auto upload_check = [queue commandBuffer];
  auto blit = [upload_check blitCommandEncoder];
  [blit copyFromTexture:storage sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(4, 4, 1) toBuffer:planes destinationOffset:0
       destinationBytesPerRow:256 destinationBytesPerImage:1024 options:MTLBlitOptionDepthFromDepthStencil];
  [blit copyFromTexture:storage sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(4, 4, 1) toBuffer:planes destinationOffset:1024
       destinationBytesPerRow:256 destinationBytesPerImage:1024 options:MTLBlitOptionStencilFromDepthStencil];
  [blit endEncoding];
  [upload_check commit];
  CHECK(complete(upload_check));
  float uploaded_depth = 0;
  std::memcpy(&uploaded_depth, planes.contents, sizeof(uploaded_depth));
  CHECK(uploaded_depth >= 0.5f && uploaded_depth < 0.50001f);
  CHECK(static_cast<const std::uint8_t *>(planes.contents)[1024] == 0x27);

  auto clear = [queue commandBuffer];
  auto *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.depthAttachment.texture = storage;
  pass.depthAttachment.loadAction = MTLLoadActionClear;
  pass.depthAttachment.storeAction = MTLStoreActionStore;
  pass.depthAttachment.clearDepth = 0.25;
  pass.stencilAttachment.texture = storage;
  pass.stencilAttachment.loadAction = MTLLoadActionClear;
  pass.stencilAttachment.storeAction = MTLStoreActionStore;
  pass.stencilAttachment.clearStencil = 0x39;
  auto encoder = [clear renderCommandEncoderWithDescriptor:pass];
  CHECK(encoder != nil);
  [encoder endEncoding];
  [clear commit];
  resources.set_completion_hook([&] { return complete(clear); });
  resources.note_gpu_write(state);
  std::array<std::uint32_t, 16> returned{};
  CHECK(resources.read_texels(state, transfer, reinterpret_cast<std::byte *>(returned.data())));
  for (auto value : returned) {
    CHECK((value & 255u) == 0x39);
    CHECK((value >> 8) >= 0x3fffff && (value >> 8) <= 0x400000);
  }
  resources.release_texture(state);
  resources.release_pool(pool);
  resources.set_completion_hook({});
}

void check_subresources(id<MTLDevice> device, metal::Resources &resources,
                         id<MTLCommandQueue> queue) {
  Pages pages(8);
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  CHECK(resources.register_pool(pool));
  auto state = make_texture(pool, 70);
  state.description.target = NVN_TEXTURE_TARGET_2D_ARRAY;
  state.description.width = state.description.height = 8;
  state.description.depth = 2;
  state.description.levels = 3;
  const auto layout = nvn::resource_layout(state.description);
  CHECK(layout.has_value());
  if (!layout) return;
  state.size = layout->total_size;
  state.layer_size = layout->layer_stride;
  resources.register_texture(state);
  nvn::TexelTransfer transfer{};
  transfer.level = 1;
  transfer.first_layer = 1;
  transfer.region = {0, 0, 0, 4, 4, 1};
  std::array<std::byte, 64> pattern{};
  for (std::size_t i = 0; i < pattern.size(); ++i) pattern[i] = std::byte(i * 3);
  CHECK(resources.write_texels(state, transfer, pattern.data()));
  nvn::TextureViewState view{};
  view.sets_levels = view.sets_layers = view.sets_target = 1;
  view.base_level = 1;
  view.level_count = 1;
  view.min_layer = 1;
  view.layer_count = 0;
  view.target = NVN_TEXTURE_TARGET_2D;
  id<MTLTexture> selected = resources.texture(state, &view);
  CHECK(selected != nil && selected.width == 4);
  if (selected) CHECK(read_native_rgba(resources, device, queue, selected) ==
                      std::vector<std::byte>(pattern.begin(), pattern.end()));
  view.layer_count = -1;
  CHECK(resources.texture(state, &view) == nil);
  view.layer_count = 1;
  view.target = NVN_TEXTURE_TARGET_3D;
  CHECK(resources.texture(state, &view) == nil);
  view.target = NVN_TEXTURE_TARGET_2D;
  view.sets_format = 1;
  view.format = NVN_FORMAT_RGBA32F;
  CHECK(resources.texture(state, &view) == nil);
  resources.release_texture(state);
  resources.release_pool(pool);
}

void check_layout_round_trips(id<MTLDevice> device, metal::Resources &resources,
                               id<MTLCommandQueue> queue) {
  const std::array descriptions{
      nvn::TextureDescription{NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_DXT1, 0, 1, 8, 8, 1, 3, 0, 0},
      nvn::TextureDescription{NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_6x5, 0, 1, 12, 10, 1, 1, 0, 0},
      nvn::TextureDescription{NVN_TEXTURE_TARGET_3D, NVN_FORMAT_R8, 0, 1, 8, 8, 11, 4, 0, 0},
      nvn::TextureDescription{NVN_TEXTURE_TARGET_1D_ARRAY, NVN_FORMAT_R8, 0, 1, 8, 3, 1, 1, 0, 0},
      nvn::TextureDescription{NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, NVN_TEXTURE_FLAGS_LINEAR, 1, 4, 4, 1, 1, 256, 0},
      nvn::TextureDescription{NVN_TEXTURE_TARGET_BUFFER, NVN_FORMAT_R32UI, 0, 1, 64, 1, 1, 1, 0, 0}};
  std::uint64_t serial = 80;
  for (const auto &description : descriptions) {
    Pages pages(8);
    nvn::MemoryPoolState pool{};
    pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
    pool.size = pages.size();
    CHECK(resources.register_pool(pool));
    auto state = make_texture(pool, serial++);
    state.description = description;
    const auto layout = nvn::resource_layout(description);
    CHECK(layout.has_value());
    if (!layout) continue;
    state.size = layout->total_size;
    state.layer_size = layout->layer_stride;
    resources.register_texture(state);
    std::vector<std::vector<std::byte>> expected;
    for (std::uint32_t layer = 0; layer < layout->layers; ++layer) {
      for (std::uint32_t level = 0; level < layout->levels; ++level) {
        const auto &mip = layout->mips[level];
        std::vector<std::byte> bytes(mip.linear_bytes);
        for (std::size_t i = 0; i < bytes.size(); ++i)
          bytes[i] = std::byte((i * 19 + layer * 13 + level * 7) & 255);
        nvn::TexelTransfer transfer{};
        transfer.level = level;
        transfer.first_layer = layer;
        transfer.region = {0, 0, 0, static_cast<int>(mip.width),
                            static_cast<int>(mip.height), static_cast<int>(mip.depth)};
        CHECK(resources.write_texels(state, transfer, bytes.data()));
        expected.push_back(std::move(bytes));
      }
    }
    id<MTLTexture> native = resources.texture(state, nullptr);
    CHECK(native != nil);
    if (native) {
      // Exercise GPU readback of every mip, layer and volume slice. Staging
      // rows have independent padding so pool and linear layouts cannot agree
      // accidentally through a round trip that only touches the CPU copy.
      std::size_t subresource = 0;
      for (std::uint32_t layer = 0; layer < layout->layers; ++layer) {
        for (std::uint32_t level = 0; level < layout->levels; ++level, ++subresource) {
          const auto &mip = layout->mips[level];
          const auto row = (mip.linear_row_bytes + 255) & ~std::size_t{255};
          const auto image = row * mip.block_rows;
          id<MTLBuffer> output = [device newBufferWithLength:image * mip.depth
                                                       options:MTLResourceStorageModeShared];
          CHECK(resources.flush_uploads());
          auto commands = [queue commandBuffer];
          auto blit = [commands blitCommandEncoder];
          [blit copyFromTexture:native sourceSlice:layer sourceLevel:level
                   sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(mip.width, mip.height, mip.depth)
                       toBuffer:output destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:image];
          [blit endEncoding];
          [commands commit];
          CHECK(complete(commands));
          const auto *bytes = static_cast<const std::byte *>(output.contents);
          for (std::uint32_t z = 0; z < mip.depth; ++z)
            for (std::uint32_t y = 0; y < mip.block_rows; ++y)
              CHECK(std::memcmp(bytes + z * image + y * row,
                  expected[subresource].data() + (z * mip.block_rows + y) * mip.linear_row_bytes,
                  mip.linear_row_bytes) == 0);
        }
      }
    }
    resources.release_texture(state);
    resources.release_pool(pool);
  }
}

void check_in_place_write_completion(id<MTLDevice> device,
                                     metal::Resources &resources,
                                     id<MTLCommandQueue> queue, bool image) {
  // A completed GPU write into in-place (linear) storage is published to the
  // CPU side, so a block-linear object over the same bytes re-uploads them
  // without waiting for the replay again.
  Pages pages(2);
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  CHECK(resources.register_pool(pool));
  auto linear = make_texture(pool, 90);
  linear.description.flags = NVN_TEXTURE_FLAGS_LINEAR;
  linear.description.stride = 256;
  const auto layout = nvn::resource_layout(linear.description);
  CHECK(layout.has_value());
  if (!layout) return;
  linear.size = layout->total_size;
  linear.layer_size = layout->layer_stride;
  resources.register_texture(linear);
  auto blocks = make_texture(pool, 91);
  if (image) blocks.description.flags = NVN_TEXTURE_FLAGS_IMAGE;
  resources.register_texture(blocks);
  id<MTLTexture> target = resources.texture(linear, nullptr);
  CHECK(target != nil && resources.texture_is_in_place(linear));
  CHECK(resources.texture(blocks, nullptr) != nil &&
        !resources.texture_is_in_place(blocks));
  if (!target) return;
  resources.register_texture_descriptor(30, blocks, nullptr);
  CHECK(resources.texture_for_id(30) != nil);
  const auto writes_before = armigo::memory::WriteTracker::now();

  unsigned completion_calls = 0;
  resources.set_completion_hook([&] {
    ++completion_calls;
    return true;
  });
  id<MTLCommandBuffer> clear = [queue commandBuffer];
  auto *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 1, 0, 1);
  auto encoder = [clear renderCommandEncoderWithDescriptor:pass];
  CHECK(encoder != nil);
  [encoder endEncoding];
  [clear commit];
  CHECK(complete(clear));
  resources.note_gpu_write(linear);
  resources.complete_gpu_writes(resources.gpu_write_generation());
  CHECK(armigo::memory::WriteTracker::now() != writes_before);

  // Row 0 of a 4x4 block-linear RGBA8 surface is the first 16 bytes, as in
  // the linear layout.
  // A descriptor resolved before the GPU write must publish its new content
  // without another guest descriptor registration, including image storage.
  id<MTLTexture> refreshed = resources.texture_for_id(30);
  CHECK(refreshed != nil);
  CHECK(completion_calls == 0);
  if (refreshed) {
    const auto pixels = read_native_rgba(resources, device, queue, refreshed);
    for (std::size_t pixel = 0; pixel < 4; ++pixel) {
      CHECK(pixels[pixel * 4] == std::byte{0});
      CHECK(pixels[pixel * 4 + 1] == std::byte{255});
      CHECK(pixels[pixel * 4 + 2] == std::byte{0});
      CHECK(pixels[pixel * 4 + 3] == std::byte{255});
    }
  }
  resources.set_completion_hook({});
  resources.release_texture(linear);
  resources.release_texture(blocks);
  resources.release_pool(pool);
}

// The tightly packed bytes of every level and layer of a native texture,
// layer-major, then level, depth slice and block row, as the layout's linear
// images. Pending uploads are committed first.
std::vector<std::byte>
read_texture_bytes(metal::Resources &resources, id<MTLDevice> device,
                   id<MTLCommandQueue> queue, id<MTLTexture> texture,
                   const armigo::graphics::TextureLayoutDescription &layout,
                   bool is_3d) {
  CHECK(resources.flush_uploads());
  std::vector<std::byte> bytes;
  for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
    for (std::uint32_t level = 0; level < layout.levels; ++level) {
      const auto &mip = layout.mips[level];
      const auto row = (mip.linear_row_bytes + 255) & ~std::size_t{255};
      const auto image = row * mip.block_rows;
      id<MTLBuffer> output = [device newBufferWithLength:image * mip.depth
                                                   options:MTLResourceStorageModeShared];
      auto commands = [queue commandBuffer];
      auto blit = [commands blitCommandEncoder];
      [blit copyFromTexture:texture
                       sourceSlice:is_3d ? 0 : layer
                       sourceLevel:level
                      sourceOrigin:MTLOriginMake(0, 0, 0)
                        sourceSize:MTLSizeMake(mip.width, mip.height, mip.depth)
                          toBuffer:output
                 destinationOffset:0
            destinationBytesPerRow:row
          destinationBytesPerImage:image];
      [blit endEncoding];
      [commands commit];
      CHECK(complete(commands));
      const auto *read = static_cast<const std::byte *>(output.contents);
      for (std::uint32_t z = 0; z < mip.depth; ++z)
        for (std::uint32_t y = 0; y < mip.block_rows; ++y)
          bytes.insert(bytes.end(), read + z * image + y * row,
                       read + z * image + y * row + mip.linear_row_bytes);
    }
  }
  return bytes;
}

struct DeswizzleCase {
  nvn::TextureDescription description{};
  std::size_t offset = 0;
  std::vector<std::byte> expected;
};

nvn::TextureDescription deswizzle_description(std::int32_t target, std::int32_t format,
                                              std::uint32_t width, std::uint32_t height,
                                              std::uint32_t depth, std::uint32_t levels,
                                              std::int32_t packaged_block_height = 0) {
  nvn::TextureDescription description{};
  description.target = target;
  description.format = format;
  description.samples = 1;
  description.width = width;
  description.height = height;
  description.depth = depth;
  description.levels = levels;
  description.packaged_block_height = packaged_block_height;
  return description;
}

// Uploads every case into one batch of a fresh Resources and returns the
// bytes each native texture holds afterwards, with the upload counts.
std::vector<std::vector<std::byte>>
upload_cases(id<MTLDevice> device, id<MTLCommandQueue> queue,
             nvn::MemoryPoolState &pool, const std::vector<DeswizzleCase> &cases,
             metal::Resources::UploadCounts &counts) {
  metal::Resources resources(device);
  resources.wait_for_gpu_deswizzle(std::chrono::seconds(10));
  std::vector<std::vector<std::byte>> results(cases.size());
  CHECK(resources.register_pool(pool));
  std::vector<nvn::TextureState> states;
  std::vector<id<MTLTexture>> textures;
  std::uint64_t serial = 300;
  for (const auto &entry : cases) {
    nvn::TextureState state{};
    state.description = entry.description;
    const auto layout = nvn::resource_layout(entry.description);
    CHECK(layout.has_value());
    if (!layout) return results;
    state.size = layout->total_size;
    state.layer_size = layout->layer_stride;
    state.offset = static_cast<std::int64_t>(entry.offset);
    state.memory = pool.memory + entry.offset;
    state.pool = &pool;
    state.serial = serial++;
    resources.register_texture(state);
    id<MTLTexture> texture = resources.texture(state, nullptr);
    CHECK(texture != nil);
    states.push_back(state);
    textures.push_back(texture);
  }
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto layout = nvn::resource_layout(cases[index].description);
    if (textures[index])
      results[index] = read_texture_bytes(
          resources, device, queue, textures[index], *layout,
          cases[index].description.target == NVN_TEXTURE_TARGET_3D);
  }
  counts = resources.upload_counts();
  for (const auto &state : states)
    resources.release_texture(state);
  resources.release_pool(pool);
  return results;
}

// Textures uploaded from block-linear pool storage are deswizzled on the GPU
// into exactly the bytes the CPU deswizzle produces: every block height, mip
// chains, arrays, cube maps, volumes, compressed blocks, depth-only formats
// and sizes that do not fill their GOBs, all in one upload batch. The same uploads with the CPU
// path forced produce the same textures.
void check_gpu_deswizzle_uploads(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  std::vector<DeswizzleCase> cases;
  const auto add = [&](nvn::TextureDescription description) {
    cases.push_back({description, 0, {}});
  };
  for (std::int32_t packaged = 1; packaged <= 6; ++packaged)
    add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 70, 300, 1, 1,
                              packaged));
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 37, 23, 1, 3));
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 64, 1, 7));
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_R8, 5, 3, 1, 1));
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 20, 12, 3, 2));
  add(deswizzle_description(NVN_TEXTURE_TARGET_CUBEMAP, NVN_FORMAT_RGBA16F, 16, 16, 1, 3));
  add(deswizzle_description(NVN_TEXTURE_TARGET_1D_ARRAY, NVN_FORMAT_R8, 70, 3, 1, 1));
  add(deswizzle_description(NVN_TEXTURE_TARGET_3D, NVN_FORMAT_R8, 8, 8, 11, 4));
  add(deswizzle_description(NVN_TEXTURE_TARGET_3D, NVN_FORMAT_RGBA8, 33, 17, 9, 3));
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_6x5, 50, 23, 1, 2));
  // Depth alone keeps its bytes, so it takes the GPU path too.
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_DEPTH16, 45, 21, 1, 1));
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_DEPTH32F, 40, 24, 3, 1));
  add(deswizzle_description(NVN_TEXTURE_TARGET_CUBEMAP, NVN_FORMAT_DEPTH32F, 16, 16, 1, 3));
  add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_DEPTH16, 64, 32, 1, 4));
  if (device.supportsBCTextureCompression) {
    add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_DXT1, 36, 20, 1, 3));
    add(deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_BPTC_UNORM, 24, 24, 1, 2));
  }

  // The pool starts inside a page, so the pool buffer begins before it, and
  // each texture sits at its own 512-byte aligned offset in the pool.
  constexpr std::size_t pool_displacement = 256;
  std::size_t end = 512;
  for (auto &entry : cases) {
    const auto layout = nvn::resource_layout(entry.description);
    CHECK(layout.has_value());
    if (!layout) return;
    entry.offset = end;
    end = (end + layout->total_size + 511) & ~std::size_t{511};
  }
  const auto page = static_cast<std::size_t>(getpagesize());
  Pages pages((pool_displacement + end + page - 1) / page + 1);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes()) return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes() + pool_displacement);
  pool.size = end;
  pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  std::uint32_t random = 0x6a09e667u;
  for (std::size_t i = 0; i < pool.size; ++i) {
    random = random * 1664525u + 1013904223u;
    pool.memory[i] = static_cast<std::uint8_t>(random >> 24);
  }
  for (auto &entry : cases) {
    const auto layout = nvn::resource_layout(entry.description);
    nvn::TextureState state{};
    state.description = entry.description;
    state.size = layout->total_size;
    state.memory = pool.memory + entry.offset;
    const bool is_3d = entry.description.target == NVN_TEXTURE_TARGET_3D;
    for (std::uint32_t layer = 0; layer < layout->layers; ++layer) {
      for (std::uint32_t level = 0; level < layout->levels; ++level) {
        const auto &mip = layout->mips[level];
        nvn::TexelTransfer transfer{};
        transfer.level = level;
        transfer.first_layer = is_3d ? 0 : layer;
        transfer.region = {0, 0, 0, static_cast<int>(mip.width),
                           static_cast<int>(mip.height),
                           static_cast<int>(is_3d ? mip.depth : 1)};
        std::vector<std::byte> level_bytes(mip.linear_bytes);
        CHECK(nvn::transfer_pool_texels(state, transfer, level_bytes.data(), false));
        entry.expected.insert(entry.expected.end(), level_bytes.begin(),
                              level_bytes.end());
      }
    }
  }

  metal::Resources::UploadCounts gpu_counts, cpu_counts;
  const auto gpu = upload_cases(device, queue, pool, cases, gpu_counts);
  CHECK(gpu_counts.gpu == cases.size() && gpu_counts.cpu == 0);
  CHECK(setenv("ARMIGO_NVN_CPU_DESWIZZLE", "1", 1) == 0);
  const auto cpu = upload_cases(device, queue, pool, cases, cpu_counts);
  CHECK(unsetenv("ARMIGO_NVN_CPU_DESWIZZLE") == 0);
  CHECK(cpu_counts.gpu == 0 && cpu_counts.cpu == cases.size());
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (gpu[index] != cases[index].expected || cpu[index] != gpu[index])
      std::fprintf(stderr, "GPU deswizzle case %zu (format %#x, target %d) differs\n",
                   index, static_cast<unsigned>(cases[index].description.format),
                   cases[index].description.target);
    CHECK(!cases[index].expected.empty());
    CHECK(gpu[index] == cases[index].expected);
    CHECK(cpu[index] == gpu[index]);
  }
}

// Storage the upload converts keeps the CPU path: packed depth/stencil, in
// block-linear and in pitch storage. A colour texture uploaded in the same
// batch still takes the GPU path.
void check_cpu_upload_paths(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes()) return;
  metal::Resources resources(device);
  resources.wait_for_gpu_deswizzle(std::chrono::seconds(10));
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  CHECK(resources.register_pool(pool));
  std::vector<nvn::TextureState> depths;
  for (const bool pitch : {false, true}) {
    auto state = make_texture(pool, pitch ? 401 : 400);
    state.description.format = NVN_FORMAT_DEPTH24_STENCIL8;
    if (pitch) {
      state.description.flags = NVN_TEXTURE_FLAGS_LINEAR;
      state.description.stride = 256;
    }
    const auto layout = nvn::resource_layout(state.description);
    CHECK(layout.has_value());
    if (!layout) return;
    state.size = layout->total_size;
    state.layer_size = layout->layer_stride;
    state.offset = pitch ? 4096 : 0;
    state.memory = pool.memory + state.offset;
    std::array<std::uint32_t, 16> packed{};
    packed.fill(0x80000027u);
    CHECK(nvn::transfer_pool_texels(state, whole_rgba_transfer(),
                                    reinterpret_cast<std::byte *>(packed.data()), true));
    resources.register_texture(state);
    depths.push_back(state);
  }
  auto colour = make_texture(pool, 402);
  colour.offset = 8192;
  colour.memory = pool.memory + colour.offset;
  std::vector<std::byte> pattern(64);
  for (std::size_t i = 0; i < pattern.size(); ++i)
    pattern[i] = std::byte((i * 41 + 5) & 0xff);
  CHECK(nvn::transfer_pool_texels(colour, whole_rgba_transfer(), pattern.data(), true));
  resources.register_texture(colour);

  std::vector<id<MTLTexture>> storages;
  for (const auto &state : depths) {
    storages.push_back(resources.texture(state, nullptr));
    CHECK(storages.back() != nil &&
          storages.back().pixelFormat == MTLPixelFormatDepth32Float_Stencil8);
  }
  id<MTLTexture> colour_texture = resources.texture(colour, nullptr);
  CHECK(colour_texture != nil);
  const auto counts = resources.upload_counts();
  CHECK(counts.cpu == 2 && counts.gpu == 1);
  if (colour_texture)
    CHECK(read_native_rgba(resources, device, queue, colour_texture) == pattern);
  for (id<MTLTexture> storage : storages) {
    if (!storage) continue;
    id<MTLBuffer> planes = [device newBufferWithLength:2048 options:MTLResourceStorageModeShared];
    CHECK(resources.flush_uploads());
    auto commands = [queue commandBuffer];
    auto blit = [commands blitCommandEncoder];
    [blit copyFromTexture:storage sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(4, 4, 1) toBuffer:planes destinationOffset:0
         destinationBytesPerRow:256 destinationBytesPerImage:1024
                        options:MTLBlitOptionDepthFromDepthStencil];
    [blit copyFromTexture:storage sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(4, 4, 1) toBuffer:planes destinationOffset:1024
         destinationBytesPerRow:256 destinationBytesPerImage:1024
                        options:MTLBlitOptionStencilFromDepthStencil];
    [blit endEncoding];
    [commands commit];
    CHECK(complete(commands));
    const auto *bytes = static_cast<const std::uint8_t *>(planes.contents);
    for (std::size_t y = 0; y < 4; ++y)
      for (std::size_t x = 0; x < 4; ++x) {
        float depth = 0;
        std::memcpy(&depth, bytes + y * 256 + x * 4, sizeof(depth));
        CHECK(depth >= 0.5f && depth < 0.50001f);
        CHECK(bytes[1024 + y * 256 + x] == 0x27);
      }
  }
  for (const auto &state : depths)
    resources.release_texture(state);
  resources.release_texture(colour);
  resources.release_pool(pool);
}

// One texture uploaded twice in one batch keeps the second upload, on both
// paths: the batch orders the two copies into the same texture.
void check_repeated_upload_in_batch(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  for (const bool cpu_only : {false, true}) {
    Pages pages(4);
    CHECK(pages.bytes() != nullptr);
    if (!pages.bytes()) return;
    if (cpu_only) CHECK(setenv("ARMIGO_NVN_CPU_DESWIZZLE", "1", 1) == 0);
    metal::Resources resources(device);
    resources.wait_for_gpu_deswizzle(std::chrono::seconds(10));
    if (cpu_only) CHECK(unsetenv("ARMIGO_NVN_CPU_DESWIZZLE") == 0);
    nvn::MemoryPoolState pool{};
    pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
    pool.size = pages.size();
    pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
    CHECK(resources.register_pool(pool));
    auto state = make_texture(pool, 500);
    std::vector<std::byte> first(64), second(64);
    for (std::size_t i = 0; i < 64; ++i) {
      first[i] = std::byte((i * 7 + 1) & 0xff);
      second[i] = std::byte((i * 11 + 200) & 0xff);
    }
    CHECK(nvn::transfer_pool_texels(state, whole_rgba_transfer(), first.data(), true));
    resources.register_texture(state);
    id<MTLTexture> texture = resources.texture(state, nullptr);
    CHECK(texture != nil);
    CHECK(nvn::transfer_pool_texels(state, whole_rgba_transfer(), second.data(), true));
    CHECK(resources.texture(state, nullptr) == texture);
    const auto counts = resources.upload_counts();
    CHECK((cpu_only ? counts.cpu : counts.gpu) == 2);
    if (texture) CHECK(read_native_rgba(resources, device, queue, texture) == second);
    resources.release_texture(state);
    resources.release_pool(pool);
  }
}

// Upload batches are ordered after on the GPU by a fence per replay segment:
// the fence ranges tile the batch values, a segment's range settles the batches
// in it, and a present's flush still waits for uploads made outside a walk.
void check_upload_fences(id<MTLDevice> device) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes()) return;
  metal::Resources resources(device);
  resources.wait_for_gpu_deswizzle(std::chrono::seconds(10));
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  std::vector<std::byte> bytes(64, std::byte{0x5a});

  // The first fence with uploads in flight orders them on the CPU, once per
  // Resources; spend it here so the fences below take the event path.
  auto primer = make_texture(pool, 599);
  primer.memory = pool.memory + 8192;
  primer.offset = 8192;
  CHECK(nvn::transfer_pool_texels(primer, whole_rgba_transfer(), bytes.data(), true));
  resources.register_texture(primer);
  CHECK(resources.texture(primer, nullptr) != nil);
  metal::UploadFence primed;
  resources.commit_uploads(primed);
  CHECK(resources.unsettled_uploads() == 0);

  // An upload from a thread that has not begun a walk is external.
  auto first = make_texture(pool, 600);
  CHECK(nvn::transfer_pool_texels(first, whole_rgba_transfer(), bytes.data(), true));
  resources.register_texture(first);
  CHECK(resources.texture(first, nullptr) != nil);
  metal::UploadFence one;
  resources.commit_uploads(one);
  CHECK(one.floor == primed.value && one.value > one.floor);
  CHECK(resources.flush_uploads());
  CHECK(resources.unsettled_uploads() == 0);
  CHECK(!resources.uploads_failed(one));

  // The next fence starts where the last ended, also with nothing to wait for.
  metal::UploadFence two;
  resources.commit_uploads(two);
  CHECK(two.floor == one.value && two.value == one.value && two.event == nil);

  // An upload from the walking thread is waited for through its fence's range.
  resources.begin_walk();
  auto second = make_texture(pool, 601);
  second.memory = pool.memory + 4096;
  second.offset = 4096;
  CHECK(nvn::transfer_pool_texels(second, whole_rgba_transfer(), bytes.data(), true));
  resources.register_texture(second);
  CHECK(resources.texture(second, nullptr) != nil);
  metal::UploadFence three;
  resources.commit_uploads(three);
  CHECK(three.floor == two.value && three.value > three.floor);
  CHECK(!resources.uploads_failed(three));
  CHECK(resources.unsettled_uploads() == 0);
  resources.release_texture(primer);
  resources.release_texture(first);
  resources.release_texture(second);
  resources.release_pool(pool);
}

// The first segment that would wait for the upload event is ordered after its
// batch on the CPU: a batch still running on the transfer queue when the replay
// queue commits its first event wait stalls on the GPU for 4 s. That fence
// carries no event and leaves nothing unsettled, and the uploaded content is
// already in the texture; later batches use the event as usual.
void check_first_event_wait_follows_completion(id<MTLDevice> device,
                                               id<MTLCommandQueue> queue) {
  Pages pages(4);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes()) return;
  metal::Resources resources(device);
  resources.wait_for_gpu_deswizzle(std::chrono::seconds(10));
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  CHECK(resources.register_pool(pool));
  std::vector<std::byte> bytes(64);
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = std::byte((i * 5 + 3) & 0xff);

  resources.begin_walk();
  auto first = make_texture(pool, 700);
  CHECK(nvn::transfer_pool_texels(first, whole_rgba_transfer(), bytes.data(), true));
  resources.register_texture(first);
  id<MTLTexture> texture = resources.texture(first, nullptr);
  CHECK(texture != nil);
  metal::UploadFence fence;
  resources.commit_uploads(fence);
  CHECK(fence.value >= 1 && fence.floor == 0);
  CHECK(fence.event == nil);
  CHECK(resources.unsettled_uploads() == 0);
  CHECK(!resources.uploads_failed(fence));
  if (texture) CHECK(read_native_rgba(resources, device, queue, texture) == bytes);

  // The next batch is ordered through the event.
  auto second = make_texture(pool, 701);
  second.memory = pool.memory + 4096;
  second.offset = 4096;
  CHECK(nvn::transfer_pool_texels(second, whole_rgba_transfer(), bytes.data(), true));
  resources.register_texture(second);
  CHECK(resources.texture(second, nullptr) != nil);
  metal::UploadFence next;
  resources.commit_uploads(next);
  CHECK(next.floor == fence.value && next.value > next.floor);
  CHECK(!resources.uploads_failed(next));
  CHECK(resources.unsettled_uploads() == 0);
  resources.release_texture(first);
  resources.release_texture(second);
  resources.release_pool(pool);
}

// A GPU-path upload that fills the batch's staging budget commits the batch
// early with its deswizzles and copies; the next upload opens a new batch.
// Releasing the pool waits for both, the committed one included, so storage
// reused right after the release cannot reach the textures.
void check_gpu_upload_batches(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  // 4096x4096 RGBA8 stages 4096 rows of 16384 bytes: the whole 64 MiB
  // budget of one batch.
  std::array<DeswizzleCase, 2> cases{{
      {deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 4096, 4096, 1, 1), 0, {}},
      {deswizzle_description(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 37, 23, 1, 3), 0, {}},
  }};
  std::size_t end = 0;
  for (auto &entry : cases) {
    const auto layout = nvn::resource_layout(entry.description);
    CHECK(layout.has_value());
    if (!layout) return;
    entry.offset = end;
    end = (end + layout->total_size + 511) & ~std::size_t{511};
  }
  const auto page = static_cast<std::size_t>(getpagesize());
  Pages pages((end + page - 1) / page);
  CHECK(pages.bytes() != nullptr);
  if (!pages.bytes()) return;
  nvn::MemoryPoolState pool{};
  pool.memory = reinterpret_cast<std::uint8_t *>(pages.bytes());
  pool.size = pages.size();
  pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  std::uint32_t random = 0xbb67ae85u;
  for (std::size_t i = 0; i < pool.size; ++i) {
    random = random * 1664525u + 1013904223u;
    pool.memory[i] = static_cast<std::uint8_t>(random >> 24);
  }
  metal::Resources resources(device);
  resources.wait_for_gpu_deswizzle(std::chrono::seconds(10));
  CHECK(resources.register_pool(pool));
  std::vector<nvn::TextureState> states;
  std::vector<id<MTLTexture>> textures;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    auto &entry = cases[index];
    const auto layout = nvn::resource_layout(entry.description);
    nvn::TextureState state{};
    state.description = entry.description;
    state.size = layout->total_size;
    state.layer_size = layout->layer_stride;
    state.offset = static_cast<std::int64_t>(entry.offset);
    state.memory = pool.memory + entry.offset;
    state.pool = &pool;
    state.serial = 600 + index;
    for (std::uint32_t level = 0; level < layout->levels; ++level) {
      const auto &mip = layout->mips[level];
      nvn::TexelTransfer transfer{};
      transfer.level = level;
      transfer.region = {0, 0, 0, static_cast<int>(mip.width),
                         static_cast<int>(mip.height), 1};
      std::vector<std::byte> level_bytes(mip.linear_bytes);
      CHECK(nvn::transfer_pool_texels(state, transfer, level_bytes.data(), false));
      entry.expected.insert(entry.expected.end(), level_bytes.begin(), level_bytes.end());
    }
    resources.register_texture(state);
    textures.push_back(resources.texture(state, nullptr));
    CHECK(textures.back() != nil);
    states.push_back(state);
  }
  const auto counts = resources.upload_counts();
  CHECK(counts.gpu == 2 && counts.cpu == 0);
  for (const auto &state : states)
    resources.release_texture(state);
  resources.release_pool(pool);
  std::memset(pool.memory, 0xee, pool.size);
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (!textures[index]) continue;
    const auto layout = nvn::resource_layout(cases[index].description);
    CHECK(read_texture_bytes(resources, device, queue, textures[index], *layout, false) ==
          cases[index].expected);
  }
}

} // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !device.hasUnifiedMemory) {
      std::fputs("NVN resource tests require a unified-memory Metal device.\n",
                 stderr);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    CHECK(queue != nil);
    metal::Resources resources(device);
    resources.wait_for_gpu_deswizzle(std::chrono::seconds(10));
    check_buffer_pool(device, resources, queue);
    check_guest_memory_claims(resources);
    check_guest_lease_waits(device);
    check_completed_gpu_write(device, resources, queue);
    check_texture_resources(device, resources, queue);
    check_texture_usage(resources);
    check_texture_slot_revalidation(resources);
    check_descriptor_stamps(resources);
    check_sampler_stamps(resources);
    check_recreated_texture_generation(device, queue);
    check_sampled_view_over_inherited_storage(device, queue);
    check_upload_certificates_follow_cpu_mapping(device);
    check_sampler_registration_order(resources);
    check_cached_pool_flush(device, resources, queue);
    check_cached_pool_cpu_write_after_creation(device, resources, queue);
    check_depth_storage(device, resources, queue);
    check_subresources(device, resources, queue);
    check_layout_round_trips(device, resources, queue);
    check_in_place_write_completion(device, resources, queue, false);
    check_in_place_write_completion(device, resources, queue, true);
    check_gpu_deswizzle_uploads(device, queue);
    check_cpu_upload_paths(device, queue);
    check_repeated_upload_in_batch(device, queue);
    check_gpu_upload_batches(device, queue);
    check_upload_fences(device);
    check_first_event_wait_follows_completion(device, queue);
    if (failures) {
      std::fprintf(stderr, "%u NVN resource checks failed\n", failures);
      return 1;
    }
    std::puts("NVN Metal resource checks passed");
    return 0;
  }
}
