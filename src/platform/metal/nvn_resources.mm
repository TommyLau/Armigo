// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#import "nvn_resources.h"

#include <armigo/graphics/texflush.h>
#include <armigo/memory/write_tracker.h>

#include "nvn/resource_layout.h"
#include "nvn_deswizzle.h"
#include "texture_format.h"

#include <algorithm>
#include <condition_variable>
#include <atomic>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

namespace armigo::platform::metal::nvn_driver {
namespace {

namespace graphics = armigo::graphics;
namespace nvn = armigo::nvn;
using namespace armigo::nvn::api;

// Diagnostics of the hot paths repeat for every draw. Each distinct message is
// written a few times in full and then only counted: a skipped draw is already
// reported through the replay statistics, while thousands of identical lines
// cost the encoder more than the work they describe.
namespace {
constexpr unsigned report_repeat = 4;
struct ReportOnce {
  const char *message = nullptr;
  unsigned shown = 0, suppressed = 0;
};
// The last diagnostic raised on this thread and the last failed texture
// lookup, for the replay's per-skip detail.
thread_local const char *last_report = nullptr;
thread_local std::string texture_failure;
bool report_permitted(const char *message, bool flush_tail) {
  last_report = message;
  static thread_local std::array<ReportOnce, 256> seen{};
  for (auto &entry : seen) {
    if (!entry.message || entry.message == message) {
      if (!entry.message) entry.message = message;
      ++entry.shown;
      const bool show = entry.shown <= report_repeat;
      if (!show) ++entry.suppressed;
      if (show || flush_tail) {
        if (entry.shown == report_repeat + 1 && flush_tail) {
          armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: %s: further occurrences suppressed\n",
                       message);
          return false;
        }
        return show;
      }
      return false;
    }
  }
  return true;
}
} // namespace

void report(const char *message) {
  if (!report_permitted(message, false)) return;
  armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: %s\n", message);
}

void report_id(const char *message, std::int64_t id) {
  if (!report_permitted(message, false)) return;
  armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: %s: %lld\n", message,
               static_cast<long long>(id));
}

// A texture state's identity and shape, for diagnostics.
std::string describe_texture(const nvn::TextureState &texture) {
  char line[224];
  std::snprintf(line, sizeof line,
                "serial %llu tag %#x target %d format %d %ux%ux%u levels %u memory %s size %llu",
                static_cast<unsigned long long>(texture.serial), texture.tag,
                static_cast<int>(texture.description.target),
                static_cast<int>(texture.description.format),
                texture.description.width, texture.description.height,
                texture.description.depth, texture.description.levels,
                texture.memory ? "set" : "null",
                static_cast<unsigned long long>(texture.size));
  return line;
}

void report_texture(const char *message, const nvn::TextureState &texture) {
  if (!report_permitted(message, false)) return;
  armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: %s: %s\n", message,
               describe_texture(texture).c_str());
}

void report_serial(const char *message, std::uint64_t serial) {
  if (!report_permitted(message, false)) return;
  armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: %s: %llu\n", message,
               static_cast<unsigned long long>(serial));
}

void report_serial_format(const char *message, std::uint64_t serial,
                          std::int32_t format) {
  if (!report_permitted(message, false)) return;
  armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: %s: %llu (format %d)\n", message,
               static_cast<unsigned long long>(serial), format);
}

bool add_overflow(std::uint64_t left, std::uint64_t right,
                  std::uint64_t &result) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return true;
  result = left + right;
  return false;
}

bool overlaps(std::uint64_t first, std::uint64_t first_size,
              std::uint64_t second, std::uint64_t second_size) {
  std::uint64_t first_end = 0;
  std::uint64_t second_end = 0;
  if (!first_size || !second_size ||
      add_overflow(first, first_size, first_end) ||
      add_overflow(second, second_size, second_end))
    return false;
  return first < second_end && second < first_end;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (!alignment ||
      value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    return 0;
  return (value + alignment - 1) & ~(alignment - 1);
}

bool is_three_dimensional(const nvn::TextureDescription &description) {
  return description.target == NVN_TEXTURE_TARGET_3D;
}

// Whether two formats differ only in sRGB encoding: Metal creates such a view
// without format-view usage.
bool srgb_variants(MTLPixelFormat a, MTLPixelFormat b) {
  constexpr std::pair<MTLPixelFormat, MTLPixelFormat> pairs[] = {
      {MTLPixelFormatR8Unorm, MTLPixelFormatR8Unorm_sRGB},
      {MTLPixelFormatRG8Unorm, MTLPixelFormatRG8Unorm_sRGB},
      {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB},
      {MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm_sRGB},
      {MTLPixelFormatBGR10_XR, MTLPixelFormatBGR10_XR_sRGB},
      {MTLPixelFormatBGRA10_XR, MTLPixelFormatBGRA10_XR_sRGB},
  };
  for (const auto &[linear, srgb] : pairs)
    if ((a == linear && b == srgb) || (a == srgb && b == linear))
      return true;
  return false;
}

bool format_has_implicit_one_alpha(std::int32_t format) {
  switch (format) {
  case NVN_FORMAT_RGB5:
  case NVN_FORMAT_RGBX8_SRGB:
  case NVN_FORMAT_RGBX8:
  case NVN_FORMAT_RGBX8SN:
  case NVN_FORMAT_RGBX8UI:
  case NVN_FORMAT_RGBX8I:
  case NVN_FORMAT_RGBX16F:
  case NVN_FORMAT_RGBX16:
  case NVN_FORMAT_RGBX16SN:
  case NVN_FORMAT_RGBX16UI:
  case NVN_FORMAT_RGBX16I:
  case NVN_FORMAT_RGBX32F:
  case NVN_FORMAT_RGBX32UI:
  case NVN_FORMAT_RGBX32I:
  case NVN_FORMAT_RGB_DXT1:
  case NVN_FORMAT_RGB_DXT1_SRGB:
    return true;
  default:
    return false;
  }
}

MTLTextureType texture_type(std::int32_t target) {
  switch (target) {
  case NVN_TEXTURE_TARGET_1D:
    return MTLTextureType1D;
  case NVN_TEXTURE_TARGET_2D:
  case NVN_TEXTURE_TARGET_RECTANGLE:
    return MTLTextureType2D;
  case NVN_TEXTURE_TARGET_3D:
    return MTLTextureType3D;
  case NVN_TEXTURE_TARGET_1D_ARRAY:
    return MTLTextureType1DArray;
  case NVN_TEXTURE_TARGET_2D_ARRAY:
    return MTLTextureType2DArray;
  case NVN_TEXTURE_TARGET_2D_MULTISAMPLE:
    return MTLTextureType2DMultisample;
  case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
    return MTLTextureType2DMultisampleArray;
  case NVN_TEXTURE_TARGET_CUBEMAP:
    return MTLTextureTypeCube;
  case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY:
    return MTLTextureTypeCubeArray;
  case NVN_TEXTURE_TARGET_BUFFER:
    return MTLTextureTypeTextureBuffer;
  default:
    return static_cast<MTLTextureType>(NSUIntegerMax);
  }
}

std::optional<graphics::TextureSwizzle> texture_swizzle(unsigned selector) {
  switch (selector) {
  case 0:
    return graphics::TextureSwizzle::Zero;
  case 1:
    return graphics::TextureSwizzle::OneFloat;
  case 2:
    return graphics::TextureSwizzle::Red;
  case 3:
    return graphics::TextureSwizzle::Green;
  case 4:
    return graphics::TextureSwizzle::Blue;
  case 5:
    return graphics::TextureSwizzle::Alpha;
  default:
    return std::nullopt;
  }
}

bool view_equal(const nvn::TextureViewState &left,
                const nvn::TextureViewState &right) {
  return left.min_layer == right.min_layer &&
         left.layer_count == right.layer_count &&
         left.base_level == right.base_level &&
         left.level_count == right.level_count && left.target == right.target &&
         left.sets_levels == right.sets_levels &&
         left.sets_layers == right.sets_layers &&
         left.sets_format == right.sets_format &&
         left.sets_swizzle == right.sets_swizzle &&
         left.sets_depth_stencil == right.sets_depth_stencil &&
         left.sets_target == right.sets_target &&
         left.depth_stencil_mode == right.depth_stencil_mode &&
         left.format == right.format &&
         std::equal(std::begin(left.swizzle), std::end(left.swizzle),
                    std::begin(right.swizzle));
}

nvn::TexelTransfer
whole_level_transfer(const nvn::TextureState &texture,
                     const graphics::TextureLayoutDescription &layout,
                     std::uint32_t level, std::uint32_t layer) {
  const auto &mip = layout.mips[level];
  nvn::TexelTransfer transfer;
  transfer.level = level;
  transfer.first_layer = is_three_dimensional(texture.description) ? 0 : layer;
  transfer.region = {
      0,
      0,
      0,
      static_cast<std::int32_t>(mip.width),
      static_cast<std::int32_t>(mip.height),
      static_cast<std::int32_t>(
          is_three_dimensional(texture.description) ? mip.depth : 1)};
  transfer.row_stride = mip.linear_row_bytes;
  transfer.image_stride = mip.linear_row_bytes * mip.block_rows;
  return transfer;
}

struct PreparedPlane {
  std::size_t subresource = 0;
  bool stencil = false;
  std::size_t offset = 0;
  std::size_t row_bytes = 0;
  std::size_t image_bytes = 0;
  std::size_t payload_row = 0;
};

struct HostSubresource {
  std::uint32_t level = 0;
  std::uint32_t layer = 0;
  std::vector<std::byte> baseline;
  MetalTexturePlanes planes;
};

} // namespace

using DescriptorIdentity =
    std::tuple<std::uint64_t, std::int64_t, std::int32_t>;
static std::optional<DescriptorIdentity>
descriptor_identity(const nvn::DescriptorPoolState &pool) {
  if (!pool.pool || !pool.pool->memory || pool.offset < 0 || pool.count <= 0)
    return std::nullopt;
  return DescriptorIdentity{reinterpret_cast<std::uint64_t>(pool.pool->memory),
                            pool.offset, pool.count};
}

struct Resources::Impl {
  struct Pool {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t buffer_begin = 0;
    std::uint32_t flags = 0;
    id<MTLBuffer> buffer = nil;
    bool owns_tracking = false;
    // Textures registered over this pool's bytes. A range lookup walks only
    // these instead of an index over every texture in the process; the
    // reference keeps the same per-pool ownership records.
    std::vector<std::uint64_t> texture_serials;
    // A texture in this pool has storage of its own, so a GPU write through the
    // pool's memory may have to reach it. Pools that hold no such texture (the
    // uniform windows, geometry and command memory) skip the texture search of
    // every buffer write they receive.
    bool has_separate_textures = false;
    // The buffer's length and base addresses, read once at registration.
    std::uint64_t length = 0, gpu = 0;
    std::byte *cpu = nullptr;
  };

  struct CachedView {
    bool sampled = false;
    bool has_view = false;
    nvn::TextureViewState view{};
    id<MTLTexture> texture = nil;
  };

  // Metal storage and its coherence state. Texture objects initialized over
  // the same guest storage with the same description share one: an NVN
  // texture object only describes storage, and games re-create objects over
  // storage they keep using (transient render targets every frame).
  struct Storage {
    id<MTLTexture> storage = nil;
    std::vector<CachedView> views;
    bool in_place = false;
    bool dirty = false;
    std::uint64_t dirty_generation = 0;
    std::uint64_t backing_write_generation = 0;
    std::uint64_t uploaded_backing_generation = 0;
    std::uint64_t cpu_stamp = 0;
    mutable std::uint64_t cpu_seen = 0;
    bool cpu_snapshot_valid = false;
    bool backing_conflict = false;
    bool host_copy_stale = false;
    // Listed in dirty_storages; cleared when retirement drops the entry.
    bool retire_queued = false;
  };

  struct Texture {
    nvn::TextureState state{};
    std::optional<graphics::TextureFormat> format;
    std::optional<graphics::TextureLayoutDescription> layout;
    std::shared_ptr<Storage> shared = std::make_shared<Storage>();
    bool live = true;
    Storage *operator->() { return shared.get(); }
    const Storage *operator->() const { return shared.get(); }
  };

  std::atomic<unsigned> dirty_textures{0};

  // Storage a GPU write marked dirty, with the serial that marked it. Retiring
  // a submission visits these instead of the whole texture table.
  struct DirtyStorage {
    std::uint64_t serial = 0;
    std::weak_ptr<Storage> storage;
  };
  std::vector<DirtyStorage> dirty_storages;
  std::vector<const Texture *> completed_in_place;

  // A texture whose storage a GPU write still owns. The count answers "is there
  // anything to reconcile" without walking the texture table.
  void mark_dirty(const std::shared_ptr<Storage> &storage, std::uint64_t serial) {
    storage->dirty = true;
    storage->dirty_generation = ++gpu_write_generation;
    dirty_textures.fetch_add(1, std::memory_order_relaxed);
    if (!storage->retire_queued) {
      storage->retire_queued = true;
      dirty_storages.push_back({serial, storage});
    }
  }
  void clear_dirty(Storage &storage) {
    if (storage.dirty) dirty_textures.fetch_sub(1, std::memory_order_relaxed);
    storage.dirty = false;
    storage.dirty_generation = 0;
  }

  // Shared storage by guest address, then description.
  std::unordered_map<std::uint64_t,
                     std::vector<std::pair<nvn::TextureDescription,
                                           std::weak_ptr<Storage>>>>
      shared_storage;

  std::shared_ptr<Storage> find_shared_locked(const Texture &texture) {
    const auto found = shared_storage.find(
        reinterpret_cast<std::uint64_t>(texture.state.memory));
    if (found == shared_storage.end())
      return nullptr;
    for (auto &[description, weak] : found->second) {
      if (std::memcmp(&description, &texture.state.description,
                      sizeof description) != 0)
        continue;
      auto shared = weak.lock();
      if (shared && shared->storage)
        return shared;
    }
    return nullptr;
  }

  void remember_shared_locked(const Texture &texture) {
    auto &entries = shared_storage[reinterpret_cast<std::uint64_t>(
        texture.state.memory)];
    std::erase_if(entries, [&](const auto &entry) {
      return entry.second.expired() ||
             std::memcmp(&entry.first, &texture.state.description,
                         sizeof entry.first) == 0;
    });
    entries.emplace_back(texture.state.description, texture.shared);
  }

  // Every texture entry's byte range by value, in one contiguous array, so a
  // range lookup over all textures does not walk the node table. A range is
  // appended whenever an entry's state is assigned and the array is rebuilt at
  // each sweep: it holds every current range, and superseded ones until the
  // next sweep.
  struct TextureRange {
    std::uint64_t begin = 0, end = 0, serial = 0;
  };
  std::vector<TextureRange> texture_ranges;
  void note_texture_range_locked(const nvn::TextureState &state) {
    const auto begin = reinterpret_cast<std::uint64_t>(state.memory);
    if (begin && state.size)
      texture_ranges.push_back({begin, begin + state.size, state.serial});
  }

  // The entry for a recorded texture state, recreated over its shared
  // storage when the sweep removed it.
  std::unordered_map<std::uint64_t, Texture>::iterator
  entry_iterator_locked(const nvn::TextureState &state) {
    auto found = textures.find(state.serial);
    if (found == textures.end()) {
      ++texture_generation;
      touch_serial_locked(state.serial);
      found = textures.emplace(state.serial, Texture{state}).first;
      found->second.live = false;
      note_texture_range_locked(state);
      note_recreated_locked(state.serial);
    }
    return found;
  }

  // Keeps the newest texture entry per guest storage and description; older
  // ones hand their storage over when the newest has none yet. Games re-create
  // texture objects over the storage they keep using every frame, so each
  // registration records its serial as the newest over its storage, and the
  // serial it replaces becomes a candidate. Every 256 registrations the
  // candidates still superseded are dropped and the range lists compacted;
  // nothing walks the whole texture table to find them.
  std::uint64_t registrations = 0;
  std::unordered_map<std::uint64_t,
                     std::vector<std::pair<nvn::TextureDescription, std::uint64_t>>>
      newest_serials;
  std::vector<std::uint64_t> superseded_candidates;
  // Entries the pass removed, destroyed by the caller after it releases the
  // lock: dropping an entry can free its Metal storage.
  using TextureNode = std::unordered_map<std::uint64_t, Texture>::node_type;
  std::vector<TextureNode> dropped_textures;
  // Serials dropped since the range lists were last compacted.
  std::size_t stale_ranges = 0;

  // The newest serial registered over a texture state's storage, or 0.
  std::uint64_t newest_serial_locked(const nvn::TextureState &state) const {
    const auto found = newest_serials.find(reinterpret_cast<std::uint64_t>(state.memory));
    if (found == newest_serials.end())
      return 0;
    for (const auto &[description, serial] : found->second)
      if (std::memcmp(&description, &state.description, sizeof description) == 0)
        return serial;
    return 0;
  }

  void note_registered_locked(const nvn::TextureState &state) {
    if (!state.memory)
      return;
    auto &bucket = newest_serials[reinterpret_cast<std::uint64_t>(state.memory)];
    for (auto &[description, serial] : bucket) {
      if (std::memcmp(&description, &state.description, sizeof description) != 0)
        continue;
      if (serial != state.serial)
        superseded_candidates.push_back(std::min(serial, state.serial));
      serial = std::max(serial, state.serial);
      return;
    }
    bucket.push_back({state.description, state.serial});
  }

  // An entry recreated for a serial that recorded commands still name is
  // dropped again once a newer serial holds its storage.
  void note_recreated_locked(std::uint64_t serial) {
    superseded_candidates.push_back(serial);
  }

  void sweep_superseded_locked() {
    auto candidates = std::move(superseded_candidates);
    superseded_candidates.clear();
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    bool dropped_any = false;
    for (const auto serial : candidates) {
      const auto found = textures.find(serial);
      if (found == textures.end())
        continue;
      // A released pool's textures have no storage left to hand over.
      if (!found->second.state.memory && !found->second.live) {
        touch_serial_locked(serial);
        dropped_textures.push_back(textures.extract(found));
        dropped_any = true;
        continue;
      }
      const auto keep = newest_serial_locked(found->second.state);
      if (!keep || keep == serial)
        continue;
      const auto successor = textures.find(keep);
      if (successor == textures.end())
        continue;
      if (!successor->second->storage && found->second->storage) {
        successor->second.shared = found->second.shared;
        touch_serial_locked(keep);
      }
      touch_serial_locked(serial);
      dropped_textures.push_back(textures.extract(found));
      dropped_any = true;
    }
    if (!dropped_any)
      return;
    stale_ranges += dropped_textures.size();
    if (stale_ranges < 1024 && stale_ranges * 4 < textures.size())
      return;
    stale_ranges = 0;
    // The range index and each pool's texture list drop the serials that no
    // longer have an entry; both would otherwise keep every serial ever
    // registered.
    std::erase_if(texture_ranges, [&](const TextureRange &range) {
      const auto found = textures.find(range.serial);
      return found == textures.end() ||
             reinterpret_cast<std::uint64_t>(found->second.state.memory) != range.begin;
    });
    for (auto &pool : pools)
      std::erase_if(pool.texture_serials, [&](std::uint64_t serial) {
        return !textures.contains(serial);
      });
  }

  struct TextureDescriptor {
    nvn::TextureState texture{};
    bool has_view = false;
    nvn::TextureViewState view{};
  };

  struct Sampler {
    nvn::SamplerParameters parameters{};
    id<MTLSamplerState> state = nil;
    // The sampler registration that set these parameters, and the earlier
    // registrations of this ID that bindings recorded before it still use.
    std::uint64_t registration = 0;
    struct Version {
      std::uint64_t registration = 0;
      nvn::SamplerParameters parameters{};
      id<MTLSamplerState> state = nil;
    };
    std::vector<Version> earlier;
    // The newest registration dropped from `earlier`: a binding older than
    // the versions kept cannot be resolved as recorded.
    std::uint64_t dropped = 0;
  };

  struct Range {
    std::uint64_t begin = 0;
    std::uint64_t size = 0;
    std::uint64_t generation = 0;
  };

  id<MTLDevice> device = nil;
  id<MTLCommandQueue> transfer_queue = nil;
  // Texture uploads not yet committed: one batch on the transfer queue,
  // committed before any other transfer and before the replay commits work
  // that may read the textures (flush_uploads), so the uploads stop costing
  // a GPU round trip each. A batch is one compute pass holding the GPU
  // deswizzles of every texture uploaded from its memory pool, then the
  // staging-to-texture copies in upload order: copies are recorded and
  // encoded when the batch is committed.
  struct UploadCopy {
    id<MTLBuffer> staging = nil;
    std::size_t offset = 0, row_bytes = 0, image_bytes = 0;
    MTLSize size{};
    id<MTLTexture> texture = nil;
    NSUInteger slice = 0, level = 0;
    MTLBlitOption options = MTLBlitOptionNone;
    // The first copy of one upload: a texture uploaded again in the same
    // batch starts a new blit pass there (pass_destinations).
    bool first = false;
  };
  id<MTLCommandBuffer> upload_batch = nil;
  id<MTLComputeCommandEncoder> upload_compute = nil;
  std::vector<UploadCopy> upload_copies;
  // A batch committed to the transfer queue and not yet settled. The GPU
  // orders the replay's consuming segment after it through upload_event
  // (the batch signals its value when it completes, or its completion
  // handler does if it failed); the CPU only observes its outcome.
  struct CommittedUpload {
    id<MTLCommandBuffer> commands = nil;
    std::uint64_t value = 0;
    std::size_t bytes = 0;
    // Storage the batch uploads: if it fails, it uploads again at its next
    // use instead of keeping missing content.
    std::vector<std::pair<std::uint64_t, std::weak_ptr<Storage>>> storage;
    // The batch reads memory pool storage when it runs: releasing a pool
    // waits for it.
    bool reads_pools = false;
    // Its copies could not be encoded: it fails as a whole.
    bool encode_failed = false;
    // It holds an upload made outside the replay worker's walk (a texel
    // write, a pool flush, a descriptor lookup): no segment is known to be
    // the one that reads it, so a present must wait for it on the CPU.
    bool external = false;
  };
  std::vector<CommittedUpload> committed_uploads;
  // Signalled to a committed batch's value by the batch; the replay queue
  // waits for it on the GPU. upload_value is the newest value assigned and
  // ordered_value the newest one a replay segment has been told to wait for.
  id<MTLSharedEvent> upload_event = nil;
  std::uint64_t upload_value = 0, ordered_value = 0;
  // Whether the first segment that would wait for the event has been
  // ordered after its batches on the CPU (see commit_uploads).
  bool event_primed = false;
  // The newest value handed to a segment's fence, and the values of failed
  // batches no segment has been told about yet.
  std::uint64_t fenced_value = 0;
  std::vector<std::uint64_t> failed_uploads;
  // The thread that begins replay walks: uploads made on it belong to a
  // submission, any other is external.
  std::thread::id walk_thread;
  bool open_external = false;
  // Staging bytes of committed, unsettled batches; a commit that would leave
  // more than upload_inflight_limit in flight waits for the oldest first.
  std::size_t inflight_bytes = 0;
  // Storage the open batch uploads.
  std::vector<std::pair<std::uint64_t, std::weak_ptr<Storage>>> batch_storage;
  // Textures the blit pass being encoded copies into: a second upload of one
  // of them starts a new pass, so the two uploads stay ordered.
  std::vector<const void *> pass_destinations;
  std::size_t batch_bytes = 0;
  // A failed upload batch, reported once by the next flush_uploads.
  bool upload_failed = false;
  // The open batch reads memory pool storage when it runs.
  bool upload_reads_pools = false;
  static constexpr std::size_t upload_batch_limit = std::size_t{64} << 20;
  static constexpr std::size_t upload_inflight_limit = std::size_t{256} << 20;
  // The deswizzle kernel, compiled in the background when the resources are
  // created; uploads take the CPU path until it is ready. A failed
  // compilation is reported once and leaves every upload on the CPU, as does
  // a failed batch that read pools on the GPU; ARMIGO_NVN_CPU_DESWIZZLE=1
  // keeps every upload on the CPU as well.
  struct DeswizzleSlot {
    std::mutex mutex;
    std::condition_variable done_signal;
    bool done = false;
    id<MTLComputePipelineState> pipeline = nil;
  };
  std::shared_ptr<DeswizzleSlot> deswizzle_slot = std::make_shared<DeswizzleSlot>();
  bool gpu_deswizzle_disabled = false;
  const bool cpu_deswizzle_only = [] {
    const char *value = std::getenv("ARMIGO_NVN_CPU_DESWIZZLE");
    return value && *value && std::strcmp(value, "0") != 0;
  }();
  std::atomic<std::uint64_t> gpu_uploads{0}, cpu_uploads{0};
  std::mutex mutex;
  CompletionHook completion_hook;
  std::vector<Pool> pools;
  // Buffers of pools released since the replay worker's current walk began.
  std::vector<id<MTLBuffer>> released_buffers;
  inline static std::atomic<std::uint64_t> next_pool_generation{1};
  std::atomic<std::uint64_t> pool_generation{next_pool_generation.fetch_add(1)};
  std::atomic<std::uint64_t> texture_generation{1};
  std::atomic<std::uint64_t> texture_lookups{0}, texture_misses{0}, sampler_misses{0};
  mutable std::ptrdiff_t last_pool = -1;

  const Pool* find_pool(std::uint64_t address, std::uint64_t end) const {
    const auto contains = [&](const Pool& pool) {
      return address >= pool.begin && end <= pool.end;
    };
    if (last_pool >= 0 && std::size_t(last_pool) < pools.size() &&
        contains(pools[last_pool])) return &pools[last_pool];
    const auto after = std::upper_bound(pools.begin(), pools.end(), address,
        [](std::uint64_t value, const Pool& pool) { return value < pool.begin; });
    if (after != pools.begin() && contains(*std::prev(after))) {
      last_pool = std::prev(after) - pools.begin();
      return &pools[last_pool];
    }
    // Overlapping registrations can contain a range beyond the nearest start.
    for (std::size_t index = 0; index < pools.size(); ++index)
      if (contains(pools[index])) { last_pool = index; return &pools[index]; }
    return nullptr;
  }
  std::unordered_map<std::uint64_t, Texture> textures;
  // Visits the registered textures whose bytes overlap [begin, end). The
  // textures that can overlap a range are the ones registered in the pool that
  // contains it, so nothing is sorted and no index is rebuilt when a texture is
  // created.
  template <class Visitor>
  void visit_texture_ranges_locked(std::uint64_t begin, std::uint64_t end, Visitor visitor) {
    const auto* pool = find_pool(begin, end);
    if (!pool) return;
    for (const auto serial : pool->texture_serials) {
      const auto found = textures.find(serial);
      if (found == textures.end()) continue;
      auto& texture = found->second;
      const auto texture_begin = reinterpret_cast<std::uint64_t>(texture.state.memory);
      if (!texture_begin || !texture.state.size) continue;
      if (texture_begin < end && begin < texture_begin + texture.state.size)
        visitor(texture);
    }
  }
  // The texture descriptor pool by ID, each slot with the sampled view last
  // resolved for it. A slot changes only when its ID is registered again or
  // its texture's storage changes: a per-serial stamp (shared by hash) moves
  // when that texture's entry is created, swept or re-registered, and a rare
  // stamp moves for changes not tied to one serial (pool release). Guest
  // stores move the write stamp, which revalidates the view against its
  // storage instead of dropping it.
  struct TextureSlot {
    bool registered = false;
    TextureDescriptor descriptor;
    id<MTLTexture> sampled = nil;
    std::uint64_t rare = 0, serial_stamp = 0, write_stamp = 0;
    // The view in a shader's declared type, and the texture and type it was
    // made for. The view retains that texture, so the pointer stays unique.
    id<MTLTexture> fitted = nil;
    id<MTLTexture> __unsafe_unretained fitted_of = nil;
    MTLTextureType fitted_type = MTLTextureType2D;
  };
  std::vector<TextureSlot> texture_slots;
  // The stamps are written under the lock and read without it by
  // texture_stamp_current: a reader that sees an old value only resolves
  // again.
  std::array<std::atomic<std::uint64_t>, 1u << 16> serial_stamps{};
  std::uint64_t serial_stamp_clock = 1;
  std::atomic<std::uint64_t> rare_stamp{1};
  static unsigned serial_stamp_index(std::uint64_t serial) {
    return static_cast<unsigned>((serial * 0x9e3779b97f4a7c15ull) >> 48);
  }
  std::uint64_t serial_stamp_locked(std::uint64_t serial) const {
    return serial_stamps[serial_stamp_index(serial)].load(std::memory_order_relaxed);
  }
  void touch_serial_locked(std::uint64_t serial) {
    serial_stamps[serial_stamp_index(serial)].store(++serial_stamp_clock,
                                                    std::memory_order_relaxed);
  }
  // Registrations of each descriptor ID, readable without the lock. A texture
  // or sampler binding resolved for an ID stays valid until that ID is
  // registered again, so registering one ID does not make the bindings of the
  // others resolve again. Chunks are made under the lock and never move or
  // go away while the resources live.
  class DescriptorVersions {
   public:
    static constexpr std::uint32_t chunk_bits = 10, chunk_size = 1u << chunk_bits;
    static constexpr std::uint32_t chunk_count = 1u << 10;
    static constexpr std::int32_t limit = static_cast<std::int32_t>(chunk_size * chunk_count);
    // 0 when the ID was never registered or is out of range.
    std::uint32_t load(std::int32_t id) const {
      if (id < 0 || id >= limit) return 0;
      const auto* chunk = chunks_[static_cast<std::uint32_t>(id) >> chunk_bits].load(
          std::memory_order_acquire);
      return chunk ? chunk[static_cast<std::uint32_t>(id) & (chunk_size - 1)].load(
                         std::memory_order_relaxed) : 0;
    }
    // Under the resources lock; false for an ID the table does not cover.
    bool bump(std::int32_t id) {
      if (id < 0 || id >= limit) return false;
      auto& entry = chunks_[static_cast<std::uint32_t>(id) >> chunk_bits];
      auto* chunk = entry.load(std::memory_order_relaxed);
      if (!chunk) {
        owned_.push_back(std::make_unique<std::atomic<std::uint32_t>[]>(chunk_size));
        chunk = owned_.back().get();
        entry.store(chunk, std::memory_order_release);
      }
      chunk[static_cast<std::uint32_t>(id) & (chunk_size - 1)].fetch_add(
          1, std::memory_order_release);
      return true;
    }
   private:
    std::array<std::atomic<std::atomic<std::uint32_t>*>, chunk_count> chunks_{};
    std::vector<std::unique_ptr<std::atomic<std::uint32_t>[]>> owned_;
  };
  DescriptorVersions texture_versions, sampler_versions;
  // Moves when a descriptor pool is registered or released: every descriptor
  // resolved before is then gone. One counter for every instance, so a cache
  // shared across instances never matches another's value.
  inline static std::atomic<std::uint64_t> next_descriptor_epoch{1};
  std::atomic<std::uint64_t> descriptor_epoch{next_descriptor_epoch.fetch_add(1)};
  void new_descriptor_epoch_locked() {
    descriptor_epoch.store(next_descriptor_epoch.fetch_add(1), std::memory_order_relaxed);
  }
  // The stamp of a slot whose sampled view is current.
  void stamp_locked(std::int32_t id, const TextureSlot &slot, DescriptorStamp &stamp) const {
    stamp.slot = texture_versions.load(id);
    stamp.epoch = descriptor_epoch.load(std::memory_order_relaxed);
    stamp.rare = slot.rare;
    stamp.serial = slot.descriptor.texture.serial;
    stamp.serial_stamp = slot.serial_stamp;
    stamp.valid = stamp.slot != 0 && slot.sampled;
  }
  std::unordered_map<std::int32_t, Sampler> samplers;
  std::optional<DescriptorIdentity> texture_pool_identity;
  std::optional<DescriptorIdentity> sampler_pool_identity;
  std::vector<Range> pending_buffer_writes;
  std::uint64_t gpu_write_generation = 0;
  std::uint64_t completed_write_generation = 0;

  explicit Impl(id<MTLDevice> supplied) : device(supplied) {
    if (!device) {
      report("cannot create resources without a Metal device");
      return;
    }
    if (!device.hasUnifiedMemory)
      report("zero-copy memory pools require unified Metal memory");
    transfer_queue = [device newCommandQueue];
    if (!transfer_queue)
      report("failed to create the texture transfer queue");
    upload_event = [device newSharedEvent];
    if (!upload_event)
      report("failed to create the texture upload event");
    start_deswizzle_compile();
  }

  Resources::BufferSpan buffer_at_locked(std::uint64_t address,
                                         std::uint64_t size, bool diagnose) {
    std::uint64_t end = 0;
    if (!size || add_overflow(address, size, end)) {
      if (diagnose)
        report("buffer range is empty or overflows");
      return {nil, 0};
    }
    struct AddressCache {
      const Impl* owner = nullptr;
      std::uint64_t generation = 0, begin = 0, end = 0, base = 0;
      id<MTLBuffer> __unsafe_unretained buffer = nil;
    };
    static thread_local std::array<AddressCache, 4096> cache;
    auto& hit = cache[(address >> 14) & (cache.size() - 1)];
    if (hit.owner == this && hit.generation == pool_generation &&
        address >= hit.begin && end <= hit.end)
      return {hit.buffer, address - hit.base};
    if (const auto* pool = find_pool(address, end)) {
      hit = {this, pool_generation, pool->begin, pool->end,
             pool->buffer_begin, pool->buffer};
      return {pool->buffer, address - pool->buffer_begin};
    }
    if (diagnose) {
      // An unresolvable range is a binding the replay serves from the zero
      // block, so it repeats for every draw that binds it.
      // One warning per launch; later hits stay at trace so a repeated miss
      // does not fill the player log. The occurrence count is on the trace line.
      static std::atomic<unsigned> shown{0};
      const unsigned seen = shown.fetch_add(1, std::memory_order_relaxed);
      if (seen == 0) {
        armigo::diag::warning(
            "nvn_resources: GPU range %#llx+%#llx is not in a registered memory pool\n",
            static_cast<unsigned long long>(address),
            static_cast<unsigned long long>(size));
      } else if (armigo::diag::enabled(armigo::diag::Level::trace)) {
        armigo::diag::trace(
            "nvn_resources: GPU range %#llx+%#llx is not in a registered memory pool (occurrence %u)\n",
            static_cast<unsigned long long>(address),
            static_cast<unsigned long long>(size), seen + 1);
      }
    }
    return {nil, 0};
  }

  const Pool *pool_for_range_locked(std::uint64_t address,
                                    std::uint64_t size) const {
    std::uint64_t end = 0;
    if (!size || add_overflow(address, size, end))
      return nullptr;
    return find_pool(address, end);
  }

  bool cpu_cached_locked(const Texture &texture) const {
    const auto *pool = pool_for_range_locked(
        reinterpret_cast<std::uint64_t>(texture.state.memory),
        texture.state.size);
    return pool && (pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED);
  }

  bool range_for_pool(const nvn::MemoryPoolState &pool, std::uint64_t offset,
                      std::uint64_t size, Range &result) {
    if (offset > pool.size || size > pool.size - offset) {
      report("memory pool range exceeds the pool");
      return false;
    }
    const auto base = reinterpret_cast<std::uint64_t>(pool.memory);
    if (add_overflow(base, offset, result.begin)) {
      report("memory pool address overflows");
      return false;
    }
    result.size = size;
    return true;
  }

  bool pending_buffer_in_range_locked(Range range) const {
    if (std::any_of(pending_buffer_writes.begin(), pending_buffer_writes.end(),
                    [&](const Range &pending) {
                      return pending.generation > completed_write_generation &&
                             overlaps(range.begin, range.size, pending.begin,
                                      pending.size);
                    }))
      return true;
    return false;
  }

  bool pending_in_range_locked(Range range) const {
    if (pending_buffer_in_range_locked(range))
      return true;
    for (const auto &[serial, texture] : textures) {
      static_cast<void>(serial);
      if (texture->dirty &&
          texture->dirty_generation > completed_write_generation &&
          overlaps(range.begin, range.size,
                   reinterpret_cast<std::uint64_t>(texture.state.memory),
                   texture.state.size))
        return true;
    }
    return false;
  }

  bool begin_cpu_snapshot_locked(Texture &texture, std::uint64_t &stamp) {
    if (!texture.state.memory || !texture.state.size)
      return false;
    stamp = memory::WriteTracker::now();
    if (!memory::WriteTracker::protect_checked(texture.state.memory,
                                               texture.state.size))
      stamp = 0;
    return true;
  }

  void finish_cpu_snapshot_locked(Texture &texture, std::uint64_t stamp) {
    texture->cpu_stamp = stamp;
    texture->cpu_seen = stamp;
    texture->cpu_snapshot_valid = true;
    texture->host_copy_stale = false;
    texture->uploaded_backing_generation = std::max(
        texture->uploaded_backing_generation, texture->backing_write_generation);
  }

  void record_cpu_snapshot_locked(Texture &texture) {
    std::uint64_t stamp = 0;
    begin_cpu_snapshot_locked(texture, stamp);
    finish_cpu_snapshot_locked(texture, stamp);
  }

  void invalidate_aliases_locked(const Texture &changed) {
    const Texture *one[] = {&changed};
    invalidate_aliases_locked(std::span<const Texture *const>(one));
  }

  // Marks every other texture whose bytes overlap any of the changed textures.
  // Candidates come from the contiguous range index; each is checked against
  // its entry's current state.
  void invalidate_aliases_locked(std::span<const Texture *const> changed) {
    if (changed.empty())
      return;
    for (const auto &range : texture_ranges) {
      for (const auto *texture : changed) {
        const auto begin = reinterpret_cast<std::uint64_t>(texture->state.memory);
        if (range.serial == texture->state.serial || !texture->state.size ||
            range.begin >= begin + texture->state.size || begin >= range.end)
          continue;
        const auto found = textures.find(range.serial);
        if (found != textures.end() &&
            overlaps(begin, texture->state.size,
                     reinterpret_cast<std::uint64_t>(found->second.state.memory),
                     found->second.state.size))
          found->second->host_copy_stale = true;
        break;
      }
    }
  }

  bool cpu_changed_locked(const Texture &texture) const {
    if (!texture->cpu_snapshot_valid || !texture->cpu_stamp)
      return true;
    const auto now = memory::WriteTracker::now();
    if (texture->cpu_seen == now)
      return false;
    if (memory::WriteTracker::written_since(texture.state.memory,
                                            texture.state.size,
                                            texture->cpu_stamp))
      return true;
    texture->cpu_seen = now;
    return false;
  }

  bool await_replay(const char *operation) {
    CompletionHook hook;
    std::uint64_t target_generation = 0;
    {
      std::lock_guard lock(mutex);
      hook = completion_hook;
      target_generation = gpu_write_generation;
    }
    if (!hook) {
      armigo::diag::write(armigo::diag::Level::warning, 
                   "nvn_resources: %s requires a replay completion hook\n",
                   operation);
      return false;
    }
    bool complete = false;
    try {
      complete = hook();
    } catch (...) {
      report("replay completion hook raised an exception");
      return false;
    }
    if (!complete) {
      armigo::diag::write(armigo::diag::Level::warning, 
                   "nvn_resources: replay completion failed before %s\n",
                   operation);
      return false;
    }
    std::lock_guard lock(mutex);
    complete_writes_locked(target_generation);
    return true;
  }

  // Every GPU write noted up to target_generation has completed: publish the
  // buffer writes to the write tracker, drop them from the pending list and
  // settle in-place textures.
  void complete_writes_locked(std::uint64_t target_generation) {
    if (target_generation <= completed_write_generation &&
        std::none_of(pending_buffer_writes.begin(), pending_buffer_writes.end(),
                     [&](const Range &range) { return range.generation <= target_generation; }))
      return;
    completed_write_generation =
        std::max(completed_write_generation, target_generation);
    for (const auto &range : pending_buffer_writes) {
      if (range.generation <= target_generation)
        memory::WriteTracker::note_store(
            reinterpret_cast<const void *>(range.begin), range.size);
    }
    std::erase_if(pending_buffer_writes, [&](const Range &range) {
      return range.generation <= target_generation;
    });
    // In-place storage holds the GPU's writes in guest memory already; the
    // completed write only has to be published to the CPU side. Aliases of all
    // completed textures are marked in one walk of the texture table.
    completed_in_place.clear();
    std::erase_if(dirty_storages, [&](const DirtyStorage &entry) {
      const auto storage = entry.storage.lock();
      if (!storage || !storage->dirty) {
        if (storage) storage->retire_queued = false;
        return true;
      }
      if (!storage->in_place || storage->dirty_generation > target_generation)
        return false;
      auto *texture = texture_for_storage_locked(entry.serial, storage);
      if (!texture)
        return false;
      memory::WriteTracker::note_store(texture->state.memory,
                                       texture->state.size);
      clear_dirty(*storage);
      record_cpu_snapshot_locked(*texture);
      completed_in_place.push_back(texture);
      storage->retire_queued = false;
      return true;
    });
    invalidate_aliases_locked(completed_in_place);
  }

  // A registered texture over this storage: the one that marked it, or, when
  // that serial was re-created over other storage, any other one.
  Texture *texture_for_storage_locked(std::uint64_t serial,
                                      const std::shared_ptr<Storage> &storage) {
    if (const auto found = textures.find(serial);
        found != textures.end() && found->second.shared == storage)
      return &found->second;
    for (auto &[other, texture] : textures) {
      static_cast<void>(other);
      if (texture.shared == storage)
        return &texture;
    }
    return nullptr;
  }

  // Commits the open upload batch without waiting for it. The deswizzles
  // run first, then the copies: the blit passes wait for the compute pass's
  // staging writes (both use tracked buffers). Copies keep their upload
  // order, and a texture uploaded again in the batch starts a new blit pass,
  // so its two uploads stay ordered.
  void commit_uploads_locked() {
    if (!upload_batch)
      return;
    if (upload_compute) {
      [upload_compute endEncoding];
      upload_compute = nil;
    }
    id<MTLBlitCommandEncoder> encoder = nil;
    bool encode_failed = false;
    for (const auto &copy : upload_copies) {
      if (copy.first) {
        const auto *destination = (__bridge const void *)copy.texture;
        if (encoder && std::find(pass_destinations.begin(), pass_destinations.end(),
                                 destination) != pass_destinations.end()) {
          [encoder endEncoding];
          encoder = nil;
          pass_destinations.clear();
        }
        pass_destinations.push_back(destination);
      }
      if (!encoder) {
        encoder = [upload_batch blitCommandEncoder];
        if (!encoder) {
          report("failed to allocate texture upload encoder");
          encode_failed = true;
          break;
        }
      }
      [encoder copyFromBuffer:copy.staging
                 sourceOffset:copy.offset
            sourceBytesPerRow:copy.row_bytes
          sourceBytesPerImage:copy.image_bytes
                   sourceSize:copy.size
                    toTexture:copy.texture
             destinationSlice:copy.slice
             destinationLevel:copy.level
            destinationOrigin:MTLOriginMake(0, 0, 0)
                      options:copy.options];
    }
    [encoder endEncoding];
    upload_copies.clear();
    CommittedUpload committed;
    committed.commands = upload_batch;
    committed.value = ++upload_value;
    committed.bytes = batch_bytes;
    committed.storage = std::move(batch_storage);
    committed.reads_pools = upload_reads_pools;
    committed.encode_failed = encode_failed;
    committed.external = open_external;
    open_external = false;
    if (upload_event) {
      // The batch signals the replay's wait when it completes; a batch that
      // fails may not, so its completion handler raises the event as well
      // and the queue waiting on it never hangs.
      id<MTLSharedEvent> event = upload_event;
      const std::uint64_t value = committed.value;
      [upload_batch encodeSignalEvent:event value:value];
      [upload_batch addCompletedHandler:^(id<MTLCommandBuffer>) {
        if (event.signaledValue < value)
          event.signaledValue = value;
      }];
    }
    [upload_batch commit];
    inflight_bytes += committed.bytes;
    committed_uploads.push_back(std::move(committed));
    batch_storage.clear();
    upload_reads_pools = false;
    upload_batch = nil;
    pass_destinations.clear();
    batch_bytes = 0;
  }

  // Records the outcome of a completed batch. A failed batch makes the
  // textures it uploads upload again at their next use, and is reported by
  // the next flush_uploads.
  bool settle_upload_locked(CommittedUpload &batch) {
    const bool failed =
        batch.encode_failed || !transfer_completed(batch.commands, "texture upload");
    inflight_bytes -= std::min(inflight_bytes, batch.bytes);
    if (failed) {
      upload_failed = true;
      failed_uploads.push_back(batch.value);
      // The retries of a failed batch that deswizzled on the GPU use the CPU.
      if (batch.reads_pools && !gpu_deswizzle_disabled) {
        gpu_deswizzle_disabled = true;
        armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: a texture upload batch that deswizzled "
                             "on the GPU failed; uploads deswizzle on the CPU from now on\n");
      }
      for (const auto &[serial, storage] : batch.storage)
        if (const auto shared = storage.lock()) {
          shared->cpu_snapshot_valid = false;
          shared->host_copy_stale = true;
          graphics::texflush::forget(serial);
        }
    }
    batch.commands = nil;
    return !failed;
  }

  // Settles the committed batches the GPU has finished, without waiting.
  void poll_uploads_locked() {
    std::erase_if(committed_uploads, [&](CommittedUpload &batch) {
      if (batch.commands.status < MTLCommandBufferStatusCompleted)
        return false;
      settle_upload_locked(batch);
      return true;
    });
  }

  // Waits for the oldest committed batches until at most the in-flight
  // staging budget remains.
  void bound_inflight_uploads_locked() {
    while (inflight_bytes > upload_inflight_limit && !committed_uploads.empty()) {
      [committed_uploads.front().commands waitUntilCompleted];
      settle_upload_locked(committed_uploads.front());
      committed_uploads.erase(committed_uploads.begin());
    }
  }

  // Whether the open batch or a committed one not yet settled reads memory
  // pool storage when it runs.
  bool uploads_read_pools_locked() const {
    return upload_reads_pools ||
           std::any_of(committed_uploads.begin(), committed_uploads.end(),
                       [](const CommittedUpload &batch) { return batch.reads_pools; });
  }

  // Commits and waits for every pending upload. A failure among them stays
  // with the replay's next flush_uploads; the result is whether this call
  // settled only successful batches.
  bool finish_uploads_locked() {
    commit_uploads_locked();
    bool ok = true;
    for (CommittedUpload &batch : committed_uploads) {
      [batch.commands waitUntilCompleted];
      ok &= settle_upload_locked(batch);
    }
    committed_uploads.clear();
    return ok;
  }

  bool open_upload_batch_locked() {
    if (!upload_batch)
      upload_batch = [transfer_queue commandBuffer];
    if (!upload_batch)
      report("failed to allocate texture upload command buffer");
    return upload_batch != nil;
  }

  // Ends the upload of one texture into the open batch: the batch re-uploads
  // the storage if it fails, and commits early once it holds
  // upload_batch_limit staging bytes.
  void note_batched_upload_locked(Texture &texture, std::size_t staging_bytes) {
    batch_storage.emplace_back(texture.state.serial, texture.shared);
    if (std::this_thread::get_id() != walk_thread)
      open_external = true;
    batch_bytes += staging_bytes;
    if (batch_bytes >= upload_batch_limit) {
      commit_uploads_locked();
      bound_inflight_uploads_locked();
    }
  }

  bool transfer_completed(id<MTLCommandBuffer> commands, const char *operation) {
    if (commands.status != MTLCommandBufferStatusCompleted || commands.error) {
      armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: %s failed: %s\n", operation,
                   commands.error
                       ? commands.error.localizedDescription.UTF8String
                       : "incomplete Metal command buffer");
      return false;
    }
    return true;
  }

  bool run_transfer(id<MTLCommandBuffer> commands, const char *operation) {
    // Pending uploads finish first; a failure among them stays with the
    // replay's next flush instead of failing this transfer.
    finish_uploads_locked();
    if (!commands) {
      armigo::diag::write(armigo::diag::Level::warning, 
                   "nvn_resources: failed to allocate command buffer for %s\n",
                   operation);
      return false;
    }
    [commands commit];
    [commands waitUntilCompleted];
    return transfer_completed(commands, operation);
  }

  bool texture_description_locked(Texture &texture) {
    if (!texture.format)
      texture.format = nvn::resource_format(texture.state.description.format);
    if (!texture.format) {
      report_serial_format("unsupported texture format for serial",
                           texture.state.serial,
                           texture.state.description.format);
      return false;
    }
    if (!texture.layout)
      texture.layout = nvn::resource_layout(texture.state.description);
    if (!texture.layout) {
      report_serial("unsupported texture layout for serial",
                    texture.state.serial);
      return false;
    }
    return true;
  }

  bool prepare_subresources_locked(Texture &texture,
                                   std::vector<HostSubresource> &subresources,
                                   std::vector<PreparedPlane> &planes,
                                   std::size_t &transfer_size) {
    if (!texture_description_locked(texture))
      return false;
    const auto info = graphics::texture_format_info(*texture.format);
    for (std::uint32_t layer = 0; layer < texture.layout->layers; ++layer) {
      for (std::uint32_t level = 0; level < texture.layout->levels; ++level) {
        const auto &mip = texture.layout->mips[level];
        HostSubresource host;
        host.level = level;
        host.layer = layer;
        host.baseline.resize(mip.linear_bytes);
        const auto transfer =
            whole_level_transfer(texture.state, *texture.layout, level, layer);
        if (!nvn::transfer_pool_texels(texture.state, transfer,
                                       host.baseline.data(), false)) {
          report_serial("failed to read guest texture storage for serial",
                        texture.state.serial);
          return false;
        }
        auto converted = metal_texture_planes(*texture.format, host.baseline);
        if (auto *error = std::get_if<graphics::TextureError>(&converted)) {
          armigo::diag::write(armigo::diag::Level::warning, 
                       "nvn_resources: texture plane conversion failed for "
                       "serial %llu: %s\n",
                       static_cast<unsigned long long>(texture.state.serial),
                       error->message());
          return false;
        }
        host.planes = std::get<MetalTexturePlanes>(std::move(converted));
        const std::size_t index = subresources.size();
        subresources.push_back(std::move(host));
        const auto add_plane = [&](bool stencil, std::size_t unit) {
          const std::size_t payload_row = std::size_t{mip.block_columns} * unit;
          const std::size_t row = align_up(payload_row, 256);
          if (!row ||
              mip.block_rows > std::numeric_limits<std::size_t>::max() / row)
            return false;
          const std::size_t image = row * mip.block_rows;
          if (mip.depth >
              (std::numeric_limits<std::size_t>::max() - transfer_size) / image)
            return false;
          planes.push_back(
              {index, stencil, transfer_size, row, image, payload_row});
          transfer_size += image * mip.depth;
          return true;
        };
        const std::size_t color_unit =
            info.depth
                ? (*texture.format == graphics::TextureFormat::D16Unorm ? 2u
                                                                        : 4u)
                : info.bytes_per_block;
        if (!add_plane(false, color_unit) ||
            (info.stencil && !add_plane(true, 1))) {
          report("texture staging extent overflows");
          return false;
        }
      }
    }
    return true;
  }

  // The CPU path: deswizzle and convert every subresource on the CPU, copy
  // its rows into shared staging and record the copies into the batch.
  bool encode_cpu_upload_locked(Texture &texture, std::size_t &transfer_size) {
    std::vector<HostSubresource> subresources;
    std::vector<PreparedPlane> planes;
    transfer_size = 0;
    if (!prepare_subresources_locked(texture, subresources, planes,
                                     transfer_size))
      return false;
    id<MTLBuffer> staging =
        [device newBufferWithLength:std::max<std::size_t>(transfer_size, 1)
                            options:MTLResourceStorageModeShared];
    if (!staging) {
      report("failed to allocate texture upload staging buffer");
      return false;
    }
    auto *staging_bytes = static_cast<std::byte *>(staging.contents);
    for (const auto &plane : planes) {
      const auto &subresource = subresources[plane.subresource];
      const auto &mip = texture.layout->mips[subresource.level];
      const auto &source = plane.stencil ? subresource.planes.stencil
                                         : subresource.planes.color_or_depth;
      for (std::uint32_t z = 0; z < mip.depth; ++z)
        for (std::uint32_t row = 0; row < mip.block_rows; ++row)
          std::memcpy(staging_bytes + plane.offset +
                          std::size_t{z} * plane.image_bytes +
                          std::size_t{row} * plane.row_bytes,
                      source.data() + (std::size_t{z} * mip.block_rows + row) *
                                          plane.payload_row,
                      plane.payload_row);
    }
    if (!open_upload_batch_locked())
      return false;
    const bool packed =
        texture->storage.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    for (const auto &plane : planes) {
      const auto &subresource = subresources[plane.subresource];
      const auto &mip = texture.layout->mips[subresource.level];
      const MTLBlitOption option =
          packed ? (plane.stencil ? MTLBlitOptionStencilFromDepthStencil
                                  : MTLBlitOptionDepthFromDepthStencil)
                 : MTLBlitOptionNone;
      upload_copies.push_back(
          {staging, plane.offset, plane.row_bytes, plane.image_bytes,
           MTLSizeMake(mip.width, mip.height, mip.depth), texture->storage,
           is_three_dimensional(texture.state.description) ? 0
                                                           : subresource.layer,
           subresource.level, option, &plane == &planes.front()});
    }
    cpu_uploads.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  id<MTLComputePipelineState> deswizzle_kernel_locked() {
    if (gpu_deswizzle_disabled)
      return nil;
    std::lock_guard lock(deswizzle_slot->mutex);
    return deswizzle_slot->pipeline;
  }

  void start_deswizzle_compile() {
    if (cpu_deswizzle_only || !device) {
      std::lock_guard lock(deswizzle_slot->mutex);
      deswizzle_slot->done = true;
      return;
    }
    const auto slot = deswizzle_slot;
    id<MTLDevice> compile_device = device;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      auto kernel = make_deswizzle_kernel(compile_device);
      if (!kernel.pipeline)
        armigo::diag::write(armigo::diag::Level::warning, 
                     "nvn_resources: GPU texture deswizzle is unavailable, "
                     "texture uploads deswizzle on the CPU: %s\n",
                     kernel.error.c_str());
      std::lock_guard lock(slot->mutex);
      slot->pipeline = kernel.pipeline;
      slot->done = true;
      slot->done_signal.notify_all();
    });
  }

  enum class GpuUpload { Encoded, CpuPath, Failed };

  // The GPU path, taken when the upload would copy the guest bytes unchanged:
  // block-linear storage inside a registered memory pool, in a format with
  // no plane conversion. One dispatch per level and layer (all depth slices
  // of a 3D level at once) deswizzles straight from the pool's buffer into
  // private staging with the CPU path's staging layout, and the same copies
  // follow. The GPU reads the pool when the batch runs, as the hardware reads
  // texture memory when work executes; the CPU snapshot stamp is taken
  // before encoding, so a later store still triggers another upload.
  GpuUpload encode_gpu_upload_locked(Texture &texture,
                                     std::size_t &transfer_size) {
    if (cpu_deswizzle_only ||
        texture.layout->storage != graphics::TextureStorageLayout::BlockLinear ||
        !metal_texture_planes_identity(*texture.format))
      return GpuUpload::CpuPath;
    const auto info = graphics::texture_format_info(*texture.format);
    if (info.bytes_per_block != texture.layout->bytes_per_block)
      return GpuUpload::CpuPath;
    const auto memory = reinterpret_cast<std::uint64_t>(texture.state.memory);
    const auto *pool = pool_for_range_locked(memory, texture.state.size);
    if (!pool || !pool->buffer || memory < pool->buffer_begin)
      return GpuUpload::CpuPath;
    const std::uint64_t source_offset = memory - pool->buffer_begin;
    if (source_offset > pool->length ||
        texture.state.size > pool->length - source_offset)
      return GpuUpload::CpuPath;
    std::vector<graphics::BlockLinearLevelAddressing> levels;
    if (!nvn::pool_texture_addressing(texture.state, levels) ||
        levels.size() != std::size_t{texture.layout->layers} *
                             texture.layout->levels)
      return GpuUpload::CpuPath;
    // The staging layout of prepare_subresources_locked: layer-major, then
    // level, each level's rows padded to 256 bytes and its depth slices
    // consecutive.
    struct Planned {
      std::size_t offset = 0, row_bytes = 0, image_bytes = 0;
    };
    std::vector<Planned> planned;
    planned.reserve(levels.size());
    transfer_size = 0;
    for (std::size_t index = 0; index < levels.size(); ++index) {
      const auto &mip = texture.layout->mips[index % texture.layout->levels];
      const auto &level = levels[index];
      const std::size_t payload_row =
          std::size_t{mip.block_columns} * info.bytes_per_block;
      const std::size_t row = align_up(payload_row, 256);
      if (!row || level.row_bytes != payload_row ||
          level.rows != mip.block_rows || level.slices != mip.depth ||
          level.guest_base > texture.state.size ||
          level.guest_size > texture.state.size - level.guest_base ||
          !deswizzle_supported(level, row) ||
          mip.block_rows > std::numeric_limits<std::size_t>::max() / row)
        return GpuUpload::CpuPath;
      const std::size_t image = row * mip.block_rows;
      if (mip.depth >
          (std::numeric_limits<std::size_t>::max() - transfer_size) / image)
        return GpuUpload::CpuPath;
      planned.push_back({transfer_size, row, image});
      transfer_size += image * mip.depth;
    }
    id<MTLComputePipelineState> kernel = deswizzle_kernel_locked();
    if (!kernel)
      return GpuUpload::CpuPath;
    id<MTLBuffer> staging =
        [device newBufferWithLength:std::max<std::size_t>(transfer_size, 1)
                            options:MTLResourceStorageModePrivate];
    if (!staging) {
      report("failed to allocate texture upload staging buffer");
      return GpuUpload::Failed;
    }
    if (!open_upload_batch_locked())
      return GpuUpload::Failed;
    if (!upload_compute)
      upload_compute = [upload_batch
          computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    if (!upload_compute) {
      report("failed to allocate texture deswizzle encoder");
      return GpuUpload::CpuPath;
    }
    // Every dispatch writes its own staging rows and only reads the pool, so
    // the dispatches of a batch may run concurrently.
    for (std::size_t index = 0; index < levels.size(); ++index) {
      const auto &plan = planned[index];
      encode_deswizzle(upload_compute, kernel, pool->buffer, source_offset,
                       staging, plan.offset, plan.row_bytes, levels[index]);
    }
    for (std::size_t index = 0; index < levels.size(); ++index) {
      const auto level = static_cast<std::uint32_t>(index % texture.layout->levels);
      const auto layer = static_cast<std::uint32_t>(index / texture.layout->levels);
      const auto &mip = texture.layout->mips[level];
      const auto &plan = planned[index];
      upload_copies.push_back(
          {staging, plan.offset, plan.row_bytes, plan.image_bytes,
           MTLSizeMake(mip.width, mip.height, mip.depth), texture->storage,
           is_three_dimensional(texture.state.description) ? 0 : layer, level,
           MTLBlitOptionNone, index == 0});
    }
    upload_reads_pools = true;
    gpu_uploads.fetch_add(1, std::memory_order_relaxed);
    return GpuUpload::Encoded;
  }

  bool upload_texture_locked(Texture &texture) {
    if (texture->in_place) {
      record_cpu_snapshot_locked(texture);
      return true;
    }
    if (!texture->storage || !transfer_queue)
      return false;
    std::uint64_t stamp = 0;
    if (!begin_cpu_snapshot_locked(texture, stamp))
      return false;
    if (!texture_description_locked(texture))
      return false;
    std::size_t transfer_size = 0;
    const auto gpu = encode_gpu_upload_locked(texture, transfer_size);
    if (gpu == GpuUpload::Failed ||
        (gpu == GpuUpload::CpuPath &&
         !encode_cpu_upload_locked(texture, transfer_size)))
      return false;
    finish_cpu_snapshot_locked(texture, stamp);
    // Certify the layers just published, so a later texel flush can verify
    // them instead of uploading and waiting again. A pool without a CPU
    // mapping is never flushed (the flush returns at once), so its textures
    // need no certificate.
    const bool flushable =
        !texture.state.pool || nvn::pool_cpu_mapped(*texture.state.pool);
    if (flushable && texture.layout && texture.state.memory)
      graphics::texflush::note(
          texture.state.serial,
          reinterpret_cast<const std::byte *>(texture.state.memory),
          static_cast<std::uint64_t>(texture.layout->layer_stride),
          texture.layout->layers, 0, texture.layout->layers, true);
    // The texture now holds the guest bytes (the copy is ordered before every
    // later use), so it no longer holds an unfinished GPU write: a pool flush
    // over its bytes has nothing to wait for.
    clear_dirty(*texture.shared);
    texture->uploaded_backing_generation = texture->backing_write_generation;
    (void)stamp;
    note_batched_upload_locked(texture, transfer_size);
    return true;
  }

  bool readback_texture_locked(Texture &texture) {
    if (!texture->dirty)
      return true;
    if (texture->dirty_generation > completed_write_generation) {
      report_serial(
          "texture has a concurrent GPU write during readback for serial",
          texture.state.serial);
      return false;
    }
    if (texture->in_place) {
      clear_dirty(*texture.shared);
      return true;
    }
    if (!texture->storage || !transfer_queue) {
      report_serial("cannot read back texture serial", texture.state.serial);
      return false;
    }
    std::vector<HostSubresource> subresources;
    std::vector<PreparedPlane> planes;
    std::size_t transfer_size = 0;
    if (!prepare_subresources_locked(texture, subresources, planes,
                                     transfer_size))
      return false;
    id<MTLBuffer> staging =
        [device newBufferWithLength:std::max<std::size_t>(transfer_size, 1)
                            options:MTLResourceStorageModeShared];
    if (!staging) {
      report("failed to allocate texture readback staging buffer");
      return false;
    }
    id<MTLCommandBuffer> commands = [transfer_queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commands blitCommandEncoder];
    if (!encoder) {
      report("failed to allocate texture readback encoder");
      return false;
    }
    const bool packed =
        texture->storage.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    for (const auto &plane : planes) {
      const auto &subresource = subresources[plane.subresource];
      const auto &mip = texture.layout->mips[subresource.level];
      const MTLBlitOption option =
          packed ? (plane.stencil ? MTLBlitOptionStencilFromDepthStencil
                                  : MTLBlitOptionDepthFromDepthStencil)
                 : MTLBlitOptionNone;
      [encoder copyFromTexture:texture->storage
                       sourceSlice:is_three_dimensional(
                                       texture.state.description)
                                       ? 0
                                       : subresource.layer
                       sourceLevel:subresource.level
                      sourceOrigin:MTLOriginMake(0, 0, 0)
                        sourceSize:MTLSizeMake(mip.width, mip.height, mip.depth)
                          toBuffer:staging
                 destinationOffset:plane.offset
            destinationBytesPerRow:plane.row_bytes
          destinationBytesPerImage:plane.image_bytes
                           options:option];
    }
    [encoder endEncoding];
    if (!run_transfer(commands, "texture readback"))
      return false;
    const auto *staging_bytes =
        static_cast<const std::byte *>(staging.contents);
    for (const auto &plane : planes) {
      auto &subresource = subresources[plane.subresource];
      const auto &mip = texture.layout->mips[subresource.level];
      auto &destination = plane.stencil ? subresource.planes.stencil
                                        : subresource.planes.color_or_depth;
      for (std::uint32_t z = 0; z < mip.depth; ++z)
        for (std::uint32_t row = 0; row < mip.block_rows; ++row)
          std::memcpy(destination.data() +
                          (std::size_t{z} * mip.block_rows + row) *
                              plane.payload_row,
                      staging_bytes + plane.offset +
                          std::size_t{z} * plane.image_bytes +
                          std::size_t{row} * plane.row_bytes,
                      plane.payload_row);
    }
    memory::WriteTracker::prepare_host_write(texture.state.memory,
                                             texture.state.size);
    for (auto &subresource : subresources) {
      auto merged = metal_merge_texture_planes(
          *texture.format, subresource.planes, subresource.baseline);
      if (auto *error = std::get_if<graphics::TextureError>(&merged)) {
        armigo::diag::write(armigo::diag::Level::warning, 
                     "nvn_resources: texture readback conversion failed for "
                     "serial %llu: %s\n",
                     static_cast<unsigned long long>(texture.state.serial),
                     error->message());
        return false;
      }
      auto linear = std::get<std::vector<std::byte>>(std::move(merged));
      const auto transfer = whole_level_transfer(
          texture.state, *texture.layout, subresource.level, subresource.layer);
      if (!nvn::transfer_pool_texels(texture.state, transfer, linear.data(),
                                     true)) {
        memory::WriteTracker::note_store(texture.state.memory,
                                         texture.state.size);
        report_serial("failed to update guest storage for texture serial",
                      texture.state.serial);
        return false;
      }
    }
    memory::WriteTracker::note_store(texture.state.memory, texture.state.size);
    clear_dirty(*texture.shared);
    record_cpu_snapshot_locked(texture);
    invalidate_aliases_locked(texture);
    return true;
  }

  bool refresh_texture_locked(Texture &texture) {
    if (!texture->storage || texture->in_place || texture->dirty)
      return true;
    if (texture->backing_conflict) {
      report_serial(
          "GPU buffer write overlaps an unreconciled texture write for serial",
          texture.state.serial);
      return false;
    }
    if (texture->backing_write_generation > completed_write_generation) {
      report_serial(
          "texture backing still has an incomplete GPU write for serial",
          texture.state.serial);
      return false;
    }
    const bool backing_changed =
        texture->backing_write_generation > texture->uploaded_backing_generation;
    const bool cpu_changed =
        !cpu_cached_locked(texture) && cpu_changed_locked(texture);
    if (!backing_changed && !cpu_changed && !texture->host_copy_stale)
      return true;
    if (!upload_texture_locked(texture)) {
      report_serial("failed to refresh texture serial", texture.state.serial);
      return false;
    }
    return true;
  }

  bool await_texture_backing(const nvn::TextureState &state,
                             const char *operation) {
    bool pending = false;
    {
      std::lock_guard lock(mutex);
      // Shared buffer views are consumed on the GPU. Queue barriers establish
      // visibility without making their CPU backing a replay-thread dependency.
      const auto entry = textures.find(state.serial);
      if (state.description.target == NVN_TEXTURE_TARGET_BUFFER ||
          (entry != textures.end() && entry->second->in_place)) return true;
      const Range range{reinterpret_cast<std::uint64_t>(state.memory),
                        state.size};
      pending = pending_buffer_in_range_locked(range);
    }
    return !pending || await_replay(operation);
  }

  id<MTLTexture> make_buffer_texture_locked(Texture &texture) {
    const auto format = nvn::resource_format(texture.state.description.format);
    if (!format) {
      report_serial("unsupported buffer texture format for serial",
                    texture.state.serial);
      return nil;
    }
    const auto mapped = metal_pixel_format(*format);
    if (auto *error = std::get_if<graphics::TextureError>(&mapped)) {
      armigo::diag::write(armigo::diag::Level::warning, 
          "nvn_resources: buffer texture format failed for serial %llu: %s\n",
          static_cast<unsigned long long>(texture.state.serial),
          error->message());
      return nil;
    }
    const auto info = graphics::texture_format_info(*format);
    std::uint64_t bytes = 0;
    if (!info.bytes_per_block ||
        texture.state.description.width >
            std::numeric_limits<std::uint64_t>::max() / info.bytes_per_block) {
      report_serial("buffer texture extent overflows for serial",
                    texture.state.serial);
      return nil;
    }
    bytes =
        std::uint64_t{texture.state.description.width} * info.bytes_per_block;
    const auto pixel = std::get<MetalSampledFormat>(mapped).pixel_format;
    const auto alignment =
        [device minimumLinearTextureAlignmentForPixelFormat:pixel];
    if (!alignment) {
      report_serial("buffer texture format has no Metal alignment for serial",
                    texture.state.serial);
      return nil;
    }
    const auto row = align_up(static_cast<std::size_t>(bytes), alignment);
    auto span = buffer_at_locked(
        reinterpret_cast<std::uint64_t>(texture.state.memory), row, true);
    if (!span.buffer)
      return nil;
    if (span.offset % alignment) {
      report_serial("buffer texture storage is not Metal aligned for serial",
                    texture.state.serial);
      return nil;
    }
    auto *descriptor = [MTLTextureDescriptor
        textureBufferDescriptorWithPixelFormat:pixel
                                         width:texture.state.description.width
                               resourceOptions:span.buffer.resourceOptions
                                         usage:MTLTextureUsageShaderRead |
                                               MTLTextureUsageShaderWrite |
                                               MTLTextureUsagePixelFormatView];
    id<MTLTexture> native = [span.buffer newTextureWithDescriptor:descriptor
                                                           offset:span.offset
                                                      bytesPerRow:row];
    if (!native)
      report_serial("failed to create buffer texture for serial",
                    texture.state.serial);
    texture.format = format;
    texture->in_place = native != nil;
    return native;
  }

  id<MTLTexture> make_linear_texture_locked(Texture &texture) {
    if (!texture_description_locked(texture) || texture.layout->levels != 1 ||
        texture.layout->layers != 1 ||
        (texture.state.description.target != NVN_TEXTURE_TARGET_2D &&
         texture.state.description.target != NVN_TEXTURE_TARGET_RECTANGLE) ||
        is_three_dimensional(texture.state.description) ||
        graphics::is_block_compressed(
            graphics::texture_format_info(*texture.format)))
      return nil;
    // Depth formats keep separate storage even when the guest layout is
    // linear: Metal depth textures cannot view a buffer, and packed
    // depth/stencil bytes need plane conversion.
    if (graphics::texture_format_info(*texture.format).depth)
      return nil;
    const auto mapped = metal_storage_format(*texture.format);
    if (!std::holds_alternative<MTLPixelFormat>(mapped))
      return nil;
    const auto pixel = std::get<MTLPixelFormat>(mapped);
    auto span =
        buffer_at_locked(reinterpret_cast<std::uint64_t>(texture.state.memory),
                         texture.layout->total_size, false);
    if (!span.buffer)
      return nil;
    const auto alignment =
        [device minimumLinearTextureAlignmentForPixelFormat:pixel];
    const auto row = texture.layout->pitch_bytes;
    if (!alignment || span.offset % alignment || row % alignment)
      return nil;
    auto *descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:pixel
                                     width:texture.state.description.width
                                    height:std::max(
                                               texture.state.description.height,
                                               1u)
                                 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.resourceOptions = span.buffer.resourceOptions;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                       MTLTextureUsageRenderTarget |
                       MTLTextureUsagePixelFormatView;
    id<MTLTexture> native = [span.buffer newTextureWithDescriptor:descriptor
                                                           offset:span.offset
                                                      bytesPerRow:row];
    if (native)
      texture->in_place = true;
    return native;
  }

  id<MTLTexture> ensure_texture_locked(Texture &texture) {
    if (texture->storage)
      return texture->storage;
    if (!texture.state.serial) {
      report_texture("texture has no serial", texture.state);
      return nil;
    }
    if (!texture.state.memory || !texture.state.size ||
        !buffer_at_locked(reinterpret_cast<std::uint64_t>(texture.state.memory),
                          texture.state.size, false)
             .buffer) {
      // Distinct from the other report_serial diagnostics: this miss repeats
      // for every texture whose storage was never registered. Warn once.
      static std::atomic<unsigned> pool_misses{0};
      const unsigned seen = pool_misses.fetch_add(1, std::memory_order_relaxed);
      if (seen == 0) {
        armigo::diag::warning(
            "nvn_resources: texture storage is not in a registered memory pool for serial: %llu\n",
            static_cast<unsigned long long>(texture.state.serial));
      } else if (armigo::diag::enabled(armigo::diag::Level::trace)) {
        armigo::diag::trace(
            "nvn_resources: texture storage is not in a registered memory pool for serial: %llu (occurrence %u)\n",
            static_cast<unsigned long long>(texture.state.serial), seen + 1);
      }
      return nil;
    }
    if (texture.state.description.target == NVN_TEXTURE_TARGET_BUFFER) {
      texture->storage = make_buffer_texture_locked(texture);
      return texture->storage;
    }
    if (texture.state.description.samples > 1 ||
        texture.state.description.target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE ||
        texture.state.description.target ==
            NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY) {
      report_serial("multisample texture storage is unsupported for serial",
                    texture.state.serial);
      return nil;
    }
    if (!texture_description_locked(texture))
      return nil;
    if (auto shared = find_shared_locked(texture)) {
      texture.shared = std::move(shared);
      return texture->storage;
    }
    const std::size_t alignment =
        nvn::resource_alignment(texture.state.description);
    if (!alignment || texture.state.offset < 0 ||
        std::uint64_t(texture.state.offset) % alignment) {
      report_serial("texture storage alignment is invalid for serial",
                    texture.state.serial);
      return nil;
    }
    if (texture.state.description.flags &
        (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) {
      texture->storage = make_linear_texture_locked(texture);
      if (texture->storage) {
        remember_shared_locked(texture);
        return texture->storage;
      }
      report_serial(
          "pitch-linear texture requires copied Metal storage for serial",
          texture.state.serial);
    }
    if (auto *pool = const_cast<Pool *>(find_pool(
            reinterpret_cast<std::uint64_t>(texture.state.memory),
            reinterpret_cast<std::uint64_t>(texture.state.memory) +
                texture.state.size)))
      pool->has_separate_textures = true;
    const auto mapped = metal_storage_format(*texture.format);
    if (auto *error = std::get_if<graphics::TextureError>(&mapped)) {
      armigo::diag::write(armigo::diag::Level::warning, 
          "nvn_resources: Metal format mapping failed for serial %llu: %s\n",
          static_cast<unsigned long long>(texture.state.serial),
          error->message());
      return nil;
    }
    const MTLTextureType type = texture_type(texture.state.description.target);
    if (type == static_cast<MTLTextureType>(NSUIntegerMax)) {
      report_serial("unknown texture target for serial", texture.state.serial);
      return nil;
    }
    auto *descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = type;
    descriptor.pixelFormat = std::get<MTLPixelFormat>(mapped);
    descriptor.width = texture.state.description.width;
    descriptor.height =
        texture.state.description.target == NVN_TEXTURE_TARGET_1D ||
                texture.state.description.target == NVN_TEXTURE_TARGET_1D_ARRAY
            ? 1
            : std::max(texture.state.description.height, 1u);
    descriptor.depth = is_three_dimensional(texture.state.description)
                           ? std::max(texture.state.description.depth, 1u)
                           : 1;
    if (type == MTLTextureType1DArray || type == MTLTextureType2DArray)
      descriptor.arrayLength = texture.layout->layers;
    else if (type == MTLTextureTypeCubeArray)
      descriptor.arrayLength = texture.layout->layers / 6;
    else
      descriptor.arrayLength = 1;
    descriptor.mipmapLevelCount = texture.layout->levels;
    descriptor.sampleCount = 1;
    descriptor.storageMode = MTLStorageModePrivate;
    // Apple GPUs keep a colour texture losslessly compressed only without
    // shader-write and format-view usage (either one drops the compression
    // metadata). Colour textures are sampled and rendered; views that change
    // only the swizzle or the sRGB encoding need neither. Guest images are
    // written by shaders and may be viewed as another format of the same size,
    // so they get both. Block-compressed storage is never losslessly
    // compressed, and packed depth/stencil needs a format view for its
    // stencil plane, so both keep format views.
    // A mip-mapped colour target keeps format views as well, which leaves it
    // uncompressed: mip chains render one level while sampling a view of all
    // its levels, and on compressed storage such a view returns undefined
    // values while a level is being rendered.
    const auto info = graphics::texture_format_info(*texture.format);
    const bool compressed = graphics::is_block_compressed(info);
    const bool image = !compressed && !info.depth &&
                       (texture.state.description.flags & NVN_TEXTURE_FLAGS_IMAGE);
    const bool mip_chain = !compressed && !info.depth && texture.layout->levels > 1;
    descriptor.usage = MTLTextureUsageShaderRead;
    if (compressed || info.depth || image || mip_chain)
      descriptor.usage |= MTLTextureUsagePixelFormatView;
    if (!compressed)
      descriptor.usage |= MTLTextureUsageRenderTarget;
    if (image)
      descriptor.usage |= MTLTextureUsageShaderWrite;
    texture->storage = [device newTextureWithDescriptor:descriptor];
    if (!texture->storage) {
      report_serial("failed to allocate Metal texture for serial",
                    texture.state.serial);
      return nil;
    }
    if (!upload_texture_locked(texture)) {
      texture->storage = nil;
      return nil;
    }
    remember_shared_locked(texture);
    return texture->storage;
  }

  // Textures that share storage under different descriptions are separate
  // GPU textures: GPU-written content does not carry over from one to the
  // other. Games reuse such storage for a new purpose (transient render
  // targets) and write it before reading.
  id<MTLTexture> texture_view_locked(Texture &texture,
                                     const nvn::TextureViewState *view,
                                     bool sampled_view) {
    id<MTLTexture> storage = ensure_texture_locked(texture);
    if (!storage)
      return nil;
    if (!refresh_texture_locked(texture))
      return nil;
    if (!sampled_view && !view)
      return storage;
    for (const auto &cached : texture->views) {
      if (cached.sampled != sampled_view || cached.has_view != (view != nullptr))
        continue;
      if (!view || view_equal(cached.view, *view))
        return cached.texture;
    }
    if (texture.state.description.target == NVN_TEXTURE_TARGET_BUFFER) {
      if (view || texture.state.swizzle ||
          format_has_implicit_one_alpha(texture.state.description.format)) {
        report_serial("buffer texture sampling overrides are unsupported for serial",
                      texture.state.serial);
        return nil;
      }
      texture->views.push_back({sampled_view, false, {}, storage});
      return storage;
    }

    // An entry that took over storage another entry created has not derived
    // its format and layout yet: the storage check above returns early.
    if (!texture_description_locked(texture))
      return nil;
    nvn::TextureDescription selected = texture.state.description;
    if (view && view->sets_format)
      selected.format = view->format;
    const auto selected_format = nvn::resource_format(selected.format);
    if (!selected_format) {
      report_serial("texture view format is unsupported for serial",
                    texture.state.serial);
      return nil;
    }
    graphics::TextureDescriptor sampled;
    sampled.format = *selected_format;
    const auto packed = texture.state.swizzle;
    for (unsigned component = 0; component < 4; ++component) {
      unsigned selector = view && view->sets_swizzle
                              ? view->swizzle[component]
                              : nvn::swizzle_channel(packed, component);
      if (selector == NVN_TEXTURE_SWIZZLE_A &&
          format_has_implicit_one_alpha(selected.format))
        selector = NVN_TEXTURE_SWIZZLE_ONE;
      const auto swizzle = texture_swizzle(selector);
      if (!swizzle) {
        report_serial("texture view swizzle is unsupported for serial",
                      texture.state.serial);
        return nil;
      }
      sampled.swizzle[component] = *swizzle;
    }
    MTLPixelFormat pixel = storage.pixelFormat;
    MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(
        MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue,
        MTLTextureSwizzleAlpha);
    const unsigned aspect = view && view->sets_depth_stencil
                                ? view->depth_stencil_mode
                                : texture.state.depth_stencil_mode;
    const auto info = graphics::texture_format_info(*selected_format);
    const auto storage_info = graphics::texture_format_info(*texture.format);
    if (info.bytes_per_block != storage_info.bytes_per_block ||
        info.block_width != storage_info.block_width ||
        info.block_height != storage_info.block_height ||
        info.depth != storage_info.depth ||
        (info.depth && *selected_format != *texture.format)) {
      report_serial("texture view changes incompatible storage geometry for serial",
                    texture.state.serial);
      return nil;
    }
    if (!sampled_view) {
      const auto mapped = metal_storage_format(*selected_format);
      if (!std::holds_alternative<MTLPixelFormat>(mapped)) {
        armigo::diag::write(armigo::diag::Level::warning, 
                     "nvn_resources: texture storage format has no Metal pixel "
                     "format: serial %llu format %d target %d\n",
                     static_cast<unsigned long long>(texture.state.serial),
                     static_cast<int>(selected.format),
                     static_cast<int>(selected.target));
        return nil;
      }
      pixel = std::get<MTLPixelFormat>(mapped);
    } else if (info.depth && aspect > NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL) {
      report_serial("unsupported depth/stencil sampling mode for serial", texture.state.serial);
      return nil;
    } else if (info.depth && aspect == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL) {
      if (!info.stencil || storage.pixelFormat != MTLPixelFormatDepth32Float_Stencil8) {
        report_serial("texture has no stencil sampling plane for serial",
                      texture.state.serial);
        return nil;
      }
      pixel = MTLPixelFormatX32_Stencil8;
    } else if (info.depth) {
      // Metal samples depth directly from packed depth/stencil storage;
      // casting that storage to Depth32Float is not a legal texture view.
      pixel = storage.pixelFormat;
    } else {
      auto mapped = metal_sampled_format(sampled);
      if (auto *error = std::get_if<graphics::TextureError>(&mapped)) {
        armigo::diag::write(armigo::diag::Level::warning, 
            "nvn_resources: texture sampling view failed for serial %llu: %s\n",
            static_cast<unsigned long long>(texture.state.serial),
            error->message());
        return nil;
      }
      const auto &result = std::get<MetalSampledFormat>(mapped);
      pixel = result.pixel_format;
      swizzle = result.swizzle;
    }
    if (sampled_view && info.depth) {
      constexpr std::array native_selectors{
          MTLTextureSwizzleZero, MTLTextureSwizzleOne, MTLTextureSwizzleRed,
          MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha};
      std::array<MTLTextureSwizzle, 4> channels{};
      for (unsigned component = 0; component < 4; ++component)
        channels[component] = native_selectors[view && view->sets_swizzle
            ? view->swizzle[component] : nvn::swizzle_channel(packed, component)];
      swizzle = MTLTextureSwizzleChannelsMake(channels[0], channels[1], channels[2], channels[3]);
    }

    const std::uint32_t total_levels =
        static_cast<std::uint32_t>(storage.mipmapLevelCount);
    const std::uint32_t first_level =
        view && view->sets_levels ? view->base_level : 0;
    std::uint32_t level_count =
        view && view->sets_levels
            ? (view->level_count
                   ? view->level_count
                   : total_levels - std::min(first_level, total_levels))
            : total_levels;
    if (first_level >= total_levels || !level_count ||
        level_count > total_levels - first_level) {
      report_serial("texture view mip range is invalid for serial",
                    texture.state.serial);
      return nil;
    }
    std::uint32_t total_layers = texture.layout->layers;
    const bool three_dimensional =
        is_three_dimensional(texture.state.description);
    // A three-dimensional texture addresses an explicit layer range as depth
    // slices, so such a view is a two-dimensional view of those slices. Without
    // a layer range the texture stays three-dimensional and has one slice.
    const bool slice_view =
        three_dimensional && view && view->sets_layers;
    if (slice_view) total_layers = texture.state.description.depth;
    const std::int32_t signed_layer =
        view && view->sets_layers ? view->min_layer : 0;
    if (signed_layer < 0 || (view && view->sets_layers && view->layer_count < 0)) {
      report_serial("texture view has a negative layer range for serial",
                    texture.state.serial);
      return nil;
    }
    const std::uint32_t first_layer = signed_layer;
    std::uint32_t layer_count =
        view && view->sets_layers
            ? (view->layer_count > 0
                   ? static_cast<std::uint32_t>(view->layer_count)
                   : total_layers - std::min(first_layer, total_layers))
            : total_layers;
    if (first_layer >= total_layers || !layer_count ||
        layer_count > total_layers - first_layer) {
      report_serial("texture view layer range is invalid for serial",
                    texture.state.serial);
      return nil;
    }
    const std::int32_t selected_target = view && view->sets_target
                                             ? view->target
                                             : texture.state.description.target;
    MTLTextureType type = texture_type(selected_target);
    // Metal has no two-dimensional view of a three-dimensional texture: a
    // render target in one attaches a view of the whole selected level, and
    // the attachment's depth plane selects the view's layer.
    const bool volume_target = !sampled_view && three_dimensional;
    if (slice_view && !volume_target)
      type = layer_count == 1 ? MTLTextureType2D : MTLTextureType2DArray;
    if (!sampled_view) {
      // A render target renders into the base level of its view, and layered
      // targets address faces and layers as array slices: a cubemap face or a
      // subset of faces attaches as a two-dimensional (array) view.
      level_count = 1;
      if (volume_target)
        type = MTLTextureType3D;
      else if (type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray ||
               type == MTLTextureType2DArray || type == MTLTextureType2D)
        type = layer_count == 1 ? MTLTextureType2D : MTLTextureType2DArray;
    }
    if (type == static_cast<MTLTextureType>(NSUIntegerMax) ||
        type == MTLTextureTypeTextureBuffer ||
        type == MTLTextureType2DMultisample ||
        type == MTLTextureType2DMultisampleArray) {
      report_serial("texture view target is unsupported for serial",
                    texture.state.serial);
      return nil;
    }
    const auto dimension = [](MTLTextureType value) {
      if (value == MTLTextureType1D || value == MTLTextureType1DArray) return 1;
      if (value == MTLTextureType3D) return 3;
      return 2;
    };
    if (dimension(type) != dimension(storage.textureType)) {
      report_serial("texture view changes storage dimensionality for serial", texture.state.serial);
      return nil;
    }
    if (type == MTLTextureType1D || type == MTLTextureType2D)
      layer_count = 1;
    if (type == MTLTextureTypeCube && layer_count != 6) {
      report_serial("cubemap view requires exactly six faces for serial", texture.state.serial);
      return nil;
    }
    if ((type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray) &&
        (first_layer % 6 || layer_count % 6)) {
      report_serial("cubemap view layer range is invalid for serial",
                    texture.state.serial);
      return nil;
    }
    // Only a guest view that changes the format of a non-image colour texture
    // reaches this; recreating the storage with format views would be needed.
    if (pixel != storage.pixelFormat && !(storage.usage & MTLTextureUsagePixelFormatView) &&
        !srgb_variants(pixel, storage.pixelFormat)) {
      report_serial("texture view reinterprets storage created without format views for serial",
                    texture.state.serial);
      return nil;
    }
    id<MTLTexture> native = [storage
        newTextureViewWithPixelFormat:pixel
                          textureType:type
                               levels:NSMakeRange(first_level, level_count)
                               slices:volume_target ? NSMakeRange(0, 1)
                                                    : NSMakeRange(first_layer, layer_count)
                              swizzle:swizzle];
    if (!native) {
      report_serial("failed to create Metal texture view for serial",
                    texture.state.serial);
      return nil;
    }
    CachedView cached;
    cached.sampled = sampled_view;
    cached.has_view = view != nullptr;
    if (view)
      cached.view = *view;
    cached.texture = native;
    texture->views.push_back(cached);
    return native;
  }

  id<MTLSamplerState> sampler_locked(std::int32_t id, Sampler &sampler) {
    if (!sampler.state)
      sampler.state = new_sampler_locked(id, sampler.parameters);
    return sampler.state;
  }

  id<MTLSamplerState> new_sampler_locked(std::int32_t id, const nvn::SamplerParameters &p) {
    auto *descriptor = [MTLSamplerDescriptor new];
    switch (p.mag_filter) {
    case NVN_MAG_FILTER_NEAREST:
      descriptor.magFilter = MTLSamplerMinMagFilterNearest;
      break;
    case NVN_MAG_FILTER_LINEAR:
      descriptor.magFilter = MTLSamplerMinMagFilterLinear;
      break;
    default:
      report_id("unknown sampler magnification filter for descriptor", id);
      return nil;
    }
    switch (p.min_filter) {
    case NVN_MIN_FILTER_NEAREST:
    case NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST:
    case NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR:
      descriptor.minFilter = MTLSamplerMinMagFilterNearest;
      break;
    case NVN_MIN_FILTER_LINEAR:
    case NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST:
    case NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR:
      descriptor.minFilter = MTLSamplerMinMagFilterLinear;
      break;
    default:
      report_id("unknown sampler minification filter for descriptor", id);
      return nil;
    }
    switch (p.min_filter) {
    case NVN_MIN_FILTER_NEAREST:
    case NVN_MIN_FILTER_LINEAR:
      descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
      break;
    case NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST:
    case NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST:
      descriptor.mipFilter = MTLSamplerMipFilterNearest;
      break;
    case NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR:
    case NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR:
      descriptor.mipFilter = MTLSamplerMipFilterLinear;
      break;
    default:
      break;
    }
    const auto address_mode =
        [&](std::uint8_t wrap) -> std::optional<MTLSamplerAddressMode> {
      switch (wrap) {
      case NVN_WRAP_MODE_CLAMP:
      case NVN_WRAP_MODE_CLAMP_TO_EDGE:
        return MTLSamplerAddressModeClampToEdge;
      case NVN_WRAP_MODE_REPEAT:
        return MTLSamplerAddressModeRepeat;
      case NVN_WRAP_MODE_MIRROR_CLAMP:
      case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE:
        return MTLSamplerAddressModeMirrorClampToEdge;
      case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER:
      case NVN_WRAP_MODE_CLAMP_TO_BORDER:
        return MTLSamplerAddressModeClampToBorderColor;
      case NVN_WRAP_MODE_MIRRORED_REPEAT:
        return MTLSamplerAddressModeMirrorRepeat;
      default:
        return std::nullopt;
      }
    };
    const auto s = address_mode(p.wrap_s);
    const auto t = address_mode(p.wrap_t);
    const auto r = address_mode(p.wrap_r);
    if (!s || !t || !r) {
      report_id("unknown sampler wrap mode for descriptor", id);
      return nil;
    }
    descriptor.sAddressMode = *s;
    descriptor.tAddressMode = *t;
    descriptor.rAddressMode = *r;
    if (p.compare_mode == NVN_COMPARE_MODE_NONE) {
      descriptor.compareFunction = MTLCompareFunctionNever;
    } else if (p.compare_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE &&
               p.compare_function >= NVN_COMPARE_FUNC_NEVER &&
               p.compare_function <= NVN_COMPARE_FUNC_ALWAYS) {
      descriptor.compareFunction = static_cast<MTLCompareFunction>(
          p.compare_function - NVN_COMPARE_FUNC_NEVER);
    } else {
      report_id("unknown sampler comparison state for descriptor", id);
      return nil;
    }
    switch (p.reduction_filter) {
    case NVN_SAMPLER_REDUCTION_AVERAGE:
      descriptor.reductionMode = MTLSamplerReductionModeWeightedAverage;
      break;
    case NVN_SAMPLER_REDUCTION_MIN:
      descriptor.reductionMode = MTLSamplerReductionModeMinimum;
      break;
    case NVN_SAMPLER_REDUCTION_MAX:
      descriptor.reductionMode = MTLSamplerReductionModeMaximum;
      break;
    default:
      report_id("unknown sampler reduction filter for descriptor", id);
      return nil;
    }
    // Metal has no LOD snap and only three border colours: snap is ignored and
    // a custom border becomes the nearest preset (opaque white or black by its
    // alpha and red, else transparent black). Anisotropy is clamped to 1-16.
    descriptor.borderColor =
        p.border[3] > 0.5f ? (p.border[0] > 0.5f ? MTLSamplerBorderColorOpaqueWhite
                                                  : MTLSamplerBorderColorOpaqueBlack)
                           : MTLSamplerBorderColorTransparentBlack;
    descriptor.lodMinClamp = p.min_lod;
    descriptor.lodMaxClamp = p.max_lod;
    descriptor.lodBias = p.lod_bias;
    descriptor.maxAnisotropy = static_cast<NSUInteger>(std::clamp(p.max_anisotropy, 1.0f, 16.0f));
    descriptor.normalizedCoordinates = YES;
    descriptor.supportArgumentBuffers = YES;
    auto state = [device newSamplerStateWithDescriptor:descriptor];
    if (!state)
      report_id("failed to create Metal sampler for descriptor", id);
    return state;
  }

  bool reconcile_range(Range range, const char *operation) {
    bool needs_wait = false;
    {
      std::lock_guard lock(mutex);
      needs_wait = pending_in_range_locked(range);
    }
    if (needs_wait && !await_replay(operation))
      return false;
    std::lock_guard lock(mutex);
    if (pending_in_range_locked(range)) {
      armigo::diag::write(armigo::diag::Level::warning,  "nvn_resources: concurrent GPU write prevents %s\n",
                   operation);
      return false;
    }
    bool success = true;
    for (auto &[serial, texture] : textures) {
      static_cast<void>(serial);
      if (!texture->dirty ||
          !overlaps(range.begin, range.size,
                    reinterpret_cast<std::uint64_t>(texture.state.memory),
                    texture.state.size))
        continue;
      if (texture->backing_conflict) {
        report_serial("cannot reconcile overlapping GPU buffer and texture "
                      "writes for serial",
                      texture.state.serial);
        success = false;
        continue;
      }
      if (!readback_texture_locked(texture))
        success = false;
    }
    return success;
  }
};

Resources::Resources(id<MTLDevice> device)
    : impl_(std::make_unique<Impl>(device)) {}

Resources::~Resources() {
  {
    std::lock_guard lock(impl_->mutex);
    impl_->finish_uploads_locked();
  }
  for (const auto& pool : impl_->pools)
    if (pool.owns_tracking)
      memory::WriteTracker::remove_range(reinterpret_cast<void*>(pool.begin));
}

void Resources::set_completion_hook(CompletionHook hook) {
  std::lock_guard lock(impl_->mutex);
  impl_->completion_hook = std::move(hook);
}

// Pool generations come from one counter for every instance, and descriptor
// generations stay below 2^40: each state has its own value, so a cache
// stamped with one can never match a later state.
std::uint64_t Resources::registration_generation() const {
  constexpr unsigned descriptor_bits = 40;
  return (impl_->pool_generation.load(std::memory_order_acquire) << descriptor_bits) |
         (impl_->texture_generation.load(std::memory_order_acquire) &
          ((std::uint64_t{1} << descriptor_bits) - 1));
}

Resources::BufferSpan Resources::buffer_at(std::uint64_t gpu_address,
                                           std::uint64_t size) {
  if (!size || gpu_address > UINT64_MAX - size) return {nil, 0};
  const auto end = gpu_address + size;
  // A GPU address resolves to the pool that contains it and to the offset
  // inside that pool's zero-copy buffer. Pools are stable for as long as they
  // exist, so each thread keeps the last hit per address block and validates it
  // against the pool generation; the pool table only moves when a pool is added
  // or removed. Textures are not consulted here: their coherence is established
  // where they are used, so an ordinary buffer binding takes no lock and does no
  // search over texture storage.
  struct Cache {
    const Impl *owner = nullptr;
    std::uint64_t generation = 0, begin = 0, end = 0, base = 0;
    id<MTLBuffer> __unsafe_unretained buffer = nil;
  };
  static thread_local std::array<Cache, 4096> cache;
  auto &hit = cache[(gpu_address >> 14) & (cache.size() - 1)];
  const auto generation =
      impl_->pool_generation.load(std::memory_order_acquire);
  if (hit.owner == impl_.get() && hit.generation == generation &&
      gpu_address >= hit.begin && end <= hit.end)
    return {hit.buffer, gpu_address - hit.base};
  std::lock_guard lock(impl_->mutex);
  const auto *pool = impl_->find_pool(gpu_address, end);
  if (!pool) return impl_->buffer_at_locked(gpu_address, size, true);
  hit = {impl_.get(), generation, pool->begin, pool->end, pool->buffer_begin,
         pool->buffer};
  return {pool->buffer, gpu_address - pool->buffer_begin};
}

Resources::BufferAddress Resources::buffer_address(std::uint64_t gpu_address,
                                                   std::uint64_t size) {
  if (!size || gpu_address > UINT64_MAX - size) return {};
  const auto end = gpu_address + size;
  // The cache of buffer_at with the pool's length and base addresses, so a
  // binding resolves without retaining the buffer or asking it for them.
  struct Cache {
    const Impl *owner = nullptr;
    std::uint64_t generation = 0, begin = 0, end = 0, base = 0, length = 0, gpu = 0;
    std::byte *cpu = nullptr;
    id<MTLBuffer> __unsafe_unretained buffer = nil;
  };
  static thread_local std::array<Cache, 4096> cache;
  auto &hit = cache[(gpu_address >> 14) & (cache.size() - 1)];
  const auto generation =
      impl_->pool_generation.load(std::memory_order_acquire);
  if (hit.owner != impl_.get() || hit.generation != generation ||
      gpu_address < hit.begin || end > hit.end) {
    std::lock_guard lock(impl_->mutex);
    const auto *pool = impl_->find_pool(gpu_address, end);
    if (!pool) {
      static_cast<void>(impl_->buffer_at_locked(gpu_address, size, true));
      return {};
    }
    hit = {impl_.get(), generation, pool->begin, pool->end, pool->buffer_begin,
           pool->length, pool->gpu, pool->cpu, pool->buffer};
  }
  const auto offset = gpu_address - hit.base;
  return {hit.buffer, offset, hit.length, hit.gpu + offset,
          hit.cpu ? hit.cpu + offset : nullptr};
}

std::string Resources::take_texture_failure() {
  return std::exchange(texture_failure, {});
}

id<MTLTexture> Resources::texture(const nvn::TextureState &texture,
                                  const nvn::TextureViewState *view) {
  last_report = nullptr;
  id<MTLTexture> result = texture_lookup(texture, view);
  if (!result)
    texture_failure = describe_texture(texture) + (view ? " view" : "") + " (" +
                      (last_report ? last_report : "no diagnostic") + ")";
  return result;
}

id<MTLTexture> Resources::texture_lookup(const nvn::TextureState &texture,
                                         const nvn::TextureViewState *view) {
  if (!impl_->await_texture_backing(texture, "texture use")) {
    static std::atomic<unsigned> shown{0};
    if (shown.fetch_add(1, std::memory_order_relaxed) < 8)
      armigo::diag::write(armigo::diag::Level::warning, 
                   "nvn_resources: texture use waited on incomplete backing: serial %llu "
                   "geometry %ux%ux%u target %d\n",
                   static_cast<unsigned long long>(texture.serial),
                   texture.description.width, texture.description.height,
                   texture.description.depth,
                   static_cast<int>(texture.description.target));
    return nil;
  }
  std::lock_guard lock(impl_->mutex);
  auto found = impl_->textures.find(texture.serial);
  if (found == impl_->textures.end()) {
    // Superseded entries are swept; the recorded state recreates them over
    // the same shared storage.
    if (!texture.serial) {
      report_texture("texture has no serial", texture);
      return nil;
    }
    ++impl_->texture_generation;
    impl_->touch_serial_locked(texture.serial);
    found = impl_->textures.emplace(texture.serial, Impl::Texture{texture}).first;
    found->second.live = false;
    impl_->note_texture_range_locked(texture);
    impl_->note_recreated_locked(texture.serial);
  }
  return impl_->texture_view_locked(found->second, view, false);
}

Resources::AttachmentFormat
Resources::peek_attachment(const nvn::TextureState &texture,
                           const nvn::TextureViewState *view) {
  std::lock_guard lock(impl_->mutex);
  // The storage a texture object uses: its own once resolved, otherwise the
  // storage already shared by another object over the same memory and
  // description (games re-create texture objects every frame, so the object a
  // later pass names usually has none yet).
  std::shared_ptr<Impl::Storage> shared;
  if (const auto found = impl_->textures.find(texture.serial);
      found != impl_->textures.end() && found->second->storage) {
    shared = found->second.shared;
  } else {
    const Impl::Texture probe{texture};
    shared = impl_->find_shared_locked(probe);
  }
  if (!shared || !shared->storage) return {};
  if (!view || !view->sets_format)
    return {shared->storage.pixelFormat, shared->storage.sampleCount};
  for (const auto &cached : shared->views)
    if (!cached.sampled && cached.has_view && view_equal(cached.view, *view) && cached.texture)
      return {cached.texture.pixelFormat, cached.texture.sampleCount};
  return {};
}

id<MTLTexture> Resources::texture_for_id(std::int32_t texture_id) {
  DescriptorStamp stamp;
  return texture_for_id(texture_id, stamp);
}

bool Resources::texture_stamp_current(std::int32_t texture_id,
                                      const DescriptorStamp &stamp) const {
  return stamp.valid &&
         impl_->descriptor_epoch.load(std::memory_order_relaxed) == stamp.epoch &&
         impl_->texture_versions.load(texture_id) == stamp.slot &&
         impl_->rare_stamp.load(std::memory_order_relaxed) == stamp.rare &&
         impl_->serial_stamp_locked(stamp.serial) == stamp.serial_stamp;
}

id<MTLTexture> Resources::texture_for_id(std::int32_t texture_id,
                                         DescriptorStamp &stamp) {
  stamp = {};
  impl_->texture_lookups.fetch_add(1, std::memory_order_relaxed);
  nvn::TextureState state{};
  {
    std::lock_guard lock(impl_->mutex);
    if (texture_id < 0 ||
        static_cast<std::size_t>(texture_id) >= impl_->texture_slots.size() ||
        !impl_->texture_slots[texture_id].registered) {
      report_id("texture descriptor is not registered", texture_id);
      return nil;
    }
    auto &slot = impl_->texture_slots[texture_id];
    if (slot.sampled && slot.rare == impl_->rare_stamp.load(std::memory_order_relaxed) &&
        slot.serial_stamp ==
            impl_->serial_stamp_locked(slot.descriptor.texture.serial)) {
      impl_->stamp_locked(texture_id, slot, stamp);
      const auto now = memory::WriteTracker::now();
      if (slot.write_stamp == now)
        return slot.sampled;
      // The clock also moves for writes elsewhere, mostly GPU writes noted at
      // completion. A slot whose own guest range saw no write since its stamp,
      // and has no GPU write pending into it, still holds: restamp it with
      // the clock read before the check.
      const auto &texture = slot.descriptor.texture;
      const Impl::Range range{reinterpret_cast<std::uint64_t>(texture.memory),
                              texture.size};
      if (!memory::WriteTracker::written_since(texture.memory, texture.size,
                                               slot.write_stamp) &&
          !impl_->pending_buffer_in_range_locked(range)) {
        slot.write_stamp = now;
        return slot.sampled;
      }
    }
    stamp = {};
    state = slot.descriptor.texture;
  }
  impl_->texture_misses.fetch_add(1, std::memory_order_relaxed);
  if (!impl_->await_texture_backing(state, "texture descriptor use"))
    return nil;
  std::lock_guard lock(impl_->mutex);
  if (static_cast<std::size_t>(texture_id) >= impl_->texture_slots.size())
    return nil;
  auto &slot = impl_->texture_slots[texture_id];
  if (!slot.registered)
    return nil;
  const auto serial = slot.descriptor.texture.serial;
  // The write stamp is taken before resolving: a store meanwhile makes the
  // next lookup revalidate.
  const auto write_stamp = memory::WriteTracker::now();
  auto texture = impl_->textures.find(serial);
  if (texture == impl_->textures.end()) {
    ++impl_->texture_generation;
    impl_->touch_serial_locked(serial);
    texture = impl_->textures.emplace(serial, Impl::Texture{slot.descriptor.texture})
                  .first;
    texture->second.live = false;
    impl_->note_recreated_locked(serial);
  }
  const auto rare = impl_->rare_stamp.load(std::memory_order_relaxed);
  const auto serial_stamp = impl_->serial_stamp_locked(serial);
  auto *resolved = impl_->texture_view_locked(
      texture->second, slot.descriptor.has_view ? &slot.descriptor.view : nullptr,
      true);
  slot.sampled = resolved;
  slot.rare = rare;
  slot.serial_stamp = serial_stamp;
  slot.write_stamp = write_stamp;
  // The stamps taken before the resolution, as the slot keeps them.
  impl_->stamp_locked(texture_id, slot, stamp);
  return resolved;
}

id<MTLTexture> Resources::fitted_texture(std::int32_t texture_id, id<MTLTexture> texture,
                                         MTLTextureType type) {
  const MTLTextureType have = texture.textureType;
  const bool cube = have == MTLTextureTypeCube || have == MTLTextureTypeCubeArray;
  const bool layered = have == MTLTextureType2DArray || cube;
  const NSUInteger layers = texture.arrayLength * (cube ? 6 : 1);
  NSUInteger slices = 0;
  if (type == MTLTextureType2D && layered) slices = 1;
  else if (type == MTLTextureType2DArray && have == MTLTextureType2D) slices = 1;
  else if (type == MTLTextureType2DArray && layered) slices = layers;
  else if (type == MTLTextureTypeCube && layered && layers >= 6) slices = 6;
  else if (type == MTLTextureTypeCubeArray && layered && layers >= 6) slices = layers / 6 * 6;
  if (!slices) {
    report_id("sampled texture type has no view in the shader's type for descriptor", texture_id);
    return nil;
  }
  std::lock_guard lock(impl_->mutex);
  auto *slot = texture_id >= 0 && static_cast<std::size_t>(texture_id) < impl_->texture_slots.size()
                   ? &impl_->texture_slots[texture_id] : nullptr;
  if (slot && slot->fitted && slot->fitted_of == texture && slot->fitted_type == type)
    return slot->fitted;
  id<MTLTexture> view = [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                                   textureType:type
                                                        levels:NSMakeRange(0, texture.mipmapLevelCount)
                                                        slices:NSMakeRange(0, slices)
                                                       swizzle:texture.swizzle];
  if (!view) {
    report_id("texture view in the shader's type cannot be created for descriptor", texture_id);
    return nil;
  }
  if (slot) {
    slot->fitted = view;
    slot->fitted_of = texture;
    slot->fitted_type = type;
  }
  return view;
}

Resources::TextureLookupCounts Resources::texture_lookup_counts() const {
  return {impl_->texture_lookups.load(std::memory_order_relaxed),
          impl_->texture_misses.load(std::memory_order_relaxed),
          impl_->sampler_misses.load(std::memory_order_relaxed)};
}

id<MTLSamplerState> Resources::sampler_for_id(std::int32_t sampler_id) {
  return sampler_for_id(sampler_id, UINT64_MAX);
}

id<MTLSamplerState> Resources::sampler_for_id(std::int32_t sampler_id,
                                              std::uint64_t registrations) {
  // Valid until this ID is registered again or a pool changes: the epoch is
  // unique to one instance, so the table can serve every instance of a thread.
  struct ResolvedSampler {
    std::int32_t slot = -1;
    std::uint32_t version = 0;
    std::uint64_t epoch = 0;
    std::uint64_t since = 0;
    id<MTLSamplerState> __unsafe_unretained sampler = nil;
  };
  static thread_local std::array<ResolvedSampler, 1024> resolved;
  auto &cache = resolved[static_cast<std::uint32_t>(sampler_id) & (resolved.size() - 1)];
  if (cache.slot == sampler_id && registrations >= cache.since &&
      cache.epoch == impl_->descriptor_epoch.load(std::memory_order_relaxed) &&
      cache.version == impl_->sampler_versions.load(sampler_id))
    return cache.sampler;
  impl_->sampler_misses.fetch_add(1, std::memory_order_relaxed);
  std::uint64_t since = 0, epoch = 0;
  std::uint32_t version = 0;
  bool current = true;
  auto *sampler = [&]() -> id<MTLSamplerState> {
  std::lock_guard lock(impl_->mutex);
  const auto found = impl_->samplers.find(sampler_id);
  if (found == impl_->samplers.end()) {
    report_id("sampler descriptor is not registered", sampler_id);
    return nil;
  }
  auto &entry = found->second;
  since = entry.registration;
  epoch = impl_->descriptor_epoch.load(std::memory_order_relaxed);
  version = impl_->sampler_versions.load(sampler_id);
  // A binding recorded before the latest registration of this ID samples
  // with the parameters registered when it was recorded.
  if (registrations < entry.registration) {
    for (auto version = entry.earlier.rbegin(); version != entry.earlier.rend(); ++version) {
      if (version->registration > registrations)
        continue;
      current = false;
      if (!version->state)
        version->state = impl_->new_sampler_locked(sampler_id, version->parameters);
      return version->state;
    }
    if (entry.dropped && !entry.earlier.empty() &&
        registrations < entry.earlier.front().registration)
      report_id("sampler version of an older binding was no longer kept; using the newest for "
                "descriptor", sampler_id);
  }
  return impl_->sampler_locked(sampler_id, entry);
  }();
  // The stamps were read under the lock with the entry: a registration that
  // lands after it moves the version and makes the next lookup resolve again.
  if (sampler && current && version) cache = {sampler_id, version, epoch, since, sampler};
  return sampler;
}

void Resources::note_gpu_write(const nvn::TextureState &texture) {
  std::lock_guard lock(impl_->mutex);
  std::shared_ptr<Impl::Storage> storage;
  if (const auto found = impl_->textures.find(texture.serial);
      found != impl_->textures.end())
    storage = found->second.shared;
  else
    storage = impl_->find_shared_locked(Impl::Texture{texture});
  if (!storage) {
    report_serial("GPU write refers to an unknown texture serial",
                  texture.serial);
    return;
  }
  impl_->mark_dirty(storage, texture.serial);
}

std::uint64_t Resources::gpu_write_generation() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->gpu_write_generation;
}

void Resources::complete_gpu_writes(std::uint64_t generation) {
  std::lock_guard lock(impl_->mutex);
  impl_->complete_writes_locked(generation);
}

void Resources::note_gpu_write(std::uint64_t gpu_address, std::uint64_t size) {
  std::uint64_t end = 0;
  if (!size || add_overflow(gpu_address, size, end)) {
    report("GPU buffer write range is empty or overflows");
    return;
  }
  std::lock_guard lock(impl_->mutex);
  if (!impl_->buffer_at_locked(gpu_address, size, true).buffer)
    return;
  const auto generation = ++impl_->gpu_write_generation;
  impl_->pending_buffer_writes.push_back({gpu_address, size, generation});
  const auto *pool = impl_->find_pool(gpu_address, end);
  if (pool && !pool->has_separate_textures) return;
  impl_->visit_texture_ranges_locked(gpu_address, end, [&](Impl::Texture& texture) {
    // The GPU wrote this texture privately and a later buffer write lands in
    // its storage: the two cannot be merged. Games reach this when they reuse
    // the memory of a texture they no longer use, so the conflict is reported
    // where the texture is used again (refresh or CPU reconciliation), not here.
    if (texture->dirty && !texture->in_place) texture->backing_conflict = true;
    texture->backing_write_generation = generation;
  });
}

bool Resources::texture_is_in_place(const nvn::TextureState &texture) {
  const std::lock_guard lock(impl_->mutex);
  const auto entry = impl_->textures.find(texture.serial);
  return entry != impl_->textures.end() && entry->second->in_place;
}

bool Resources::register_pool(const nvn::MemoryPoolState &pool) {
  if (!impl_->device || !impl_->device.hasUnifiedMemory || !pool.memory ||
      !pool.size) {
    report("memory pool requires a unified device and nonempty storage");
    return false;
  }
  const std::uint64_t begin = reinterpret_cast<std::uint64_t>(pool.memory);
  std::uint64_t end = 0;
  if (add_overflow(begin, pool.size, end)) {
    report("memory pool address range overflows");
    return false;
  }
  const std::uint64_t page = static_cast<std::uint64_t>(getpagesize());
  const std::uint64_t buffer_begin = begin & ~(page - 1);
  if (end > std::numeric_limits<std::uint64_t>::max() - (page - 1)) {
    report("aligned memory pool range overflows");
    return false;
  }
  const std::uint64_t buffer_end = (end + page - 1) & ~(page - 1);
  const std::uint64_t length = buffer_end - buffer_begin;
  if (length > std::numeric_limits<NSUInteger>::max()) {
    report("memory pool exceeds the Metal buffer size range");
    return false;
  }
  id<MTLBuffer> buffer = [impl_->device
      newBufferWithBytesNoCopy:reinterpret_cast<void *>(buffer_begin)
                        length:static_cast<NSUInteger>(length)
                       options:MTLResourceStorageModeShared
                   deallocator:nil];
  if (!buffer) {
    report("failed to create zero-copy Metal memory pool buffer");
    return false;
  }
  std::lock_guard lock(impl_->mutex);
  for (const auto &existing : impl_->pools) {
    if (existing.begin == begin && existing.end == end) {
      report("memory pool is already registered");
      return false;
    }
  }
  const bool owns_tracking = !memory::WriteTracker::tracked(pool.memory, pool.size);
  if (owns_tracking)
    memory::WriteTracker::add_range(pool.memory, pool.size);
  impl_->pool_generation = Impl::next_pool_generation.fetch_add(1);
  impl_->last_pool = -1;
  Impl::Pool entry{begin, end, buffer_begin, pool.flags, std::move(buffer), owns_tracking};
  entry.length = entry.buffer.length;
  entry.gpu = entry.buffer.gpuAddress;
  entry.cpu = static_cast<std::byte *>(entry.buffer.contents);
  impl_->pools.push_back(std::move(entry));
  std::stable_sort(impl_->pools.begin(), impl_->pools.end(),
                   [](const Impl::Pool &left, const Impl::Pool &right) {
                     return left.begin < right.begin;
                   });
  return true;
}

bool Resources::wait_for_gpu_deswizzle(std::chrono::milliseconds timeout) {
  const auto slot = impl_->deswizzle_slot;
  std::unique_lock lock(slot->mutex);
  slot->done_signal.wait_for(lock, timeout, [&] { return slot->done; });
  return slot->pipeline != nil;
}

bool Resources::flush_uploads() {
  std::lock_guard lock(impl_->mutex);
  impl_->commit_uploads_locked();
  // A batch a replay segment is already ordered after needs no CPU wait: the
  // segment's completion covers it. A batch holding uploads made outside a
  // walk (and anything not yet ordered) finishes here: the caller, a present
  // on any queue, is not ordered after the segment that happened to commit it.
  for (auto &batch : impl_->committed_uploads)
    if (batch.external || batch.value > impl_->ordered_value)
      [batch.commands waitUntilCompleted];
  impl_->poll_uploads_locked();
  const bool finished = !impl_->upload_failed;
  impl_->upload_failed = false;
  return finished;
}

void Resources::commit_uploads(UploadFence &fence) {
  std::lock_guard lock(impl_->mutex);
  fence = {};
  impl_->commit_uploads_locked();
  impl_->bound_inflight_uploads_locked();
  impl_->poll_uploads_locked();
  // The first segment ordered after uploads still in flight waits for its
  // batches on the CPU instead of through the event. Measured at launch: an
  // upload batch on the transfer queue sometimes ran 4.00 s (about half of
  // warm launches) while the replay queue held its first event-ordered
  // segment, and the first frame stayed black; with this CPU wait 13 of 13
  // launches had no stall. The cause in the driver is not established. The
  // wait costs the in-flight batches' remaining latency (about 5 ms), once per
  // Resources; the batches complete before the segment commits, a failure is
  // still reported through this fence's range, and later fences use the event.
  if (!impl_->event_primed && !impl_->committed_uploads.empty()) {
    impl_->event_primed = true;
    impl_->finish_uploads_locked();
  }
  fence.floor = impl_->fenced_value;
  fence.value = impl_->fenced_value = impl_->upload_value;
  if (!impl_->committed_uploads.empty()) {
    if (impl_->upload_event) {
      fence.event = impl_->upload_event;
      impl_->ordered_value = impl_->upload_value;
    } else {
      impl_->finish_uploads_locked();
    }
  }
}

bool Resources::uploads_failed(const UploadFence &fence) {
  std::lock_guard lock(impl_->mutex);
  // The segment carrying the fence has run, so these batches have completed
  // or are about to report their status.
  std::erase_if(impl_->committed_uploads, [&](Impl::CommittedUpload &batch) {
    if (batch.value <= fence.floor || batch.value > fence.value)
      return false;
    [batch.commands waitUntilCompleted];
    impl_->settle_upload_locked(batch);
    return true;
  });
  bool failed = false;
  std::erase_if(impl_->failed_uploads, [&](std::uint64_t value) {
    if (value <= fence.floor || value > fence.value)
      return false;
    failed = true;
    return true;
  });
  return failed;
}

std::size_t Resources::unsettled_uploads() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->committed_uploads.size();
}

Resources::UploadCounts Resources::upload_counts() const {
  return {impl_->gpu_uploads.load(std::memory_order_relaxed),
          impl_->cpu_uploads.load(std::memory_order_relaxed)};
}

void Resources::begin_walk() {
  // The walk during which they were released has ended, and with it every use
  // of their addresses. They are released after the lock is.
  std::vector<id<MTLBuffer>> released;
  std::lock_guard lock(impl_->mutex);
  impl_->walk_thread = std::this_thread::get_id();
  released.swap(impl_->released_buffers);
}

void Resources::release_pool(const nvn::MemoryPoolState &pool) {
  Impl::Range range;
  if (!impl_->range_for_pool(pool, 0, pool.size, range))
    return;
  const auto end = range.begin + range.size;
  {
    std::lock_guard lock(impl_->mutex);
    const bool registered = std::any_of(
        impl_->pools.begin(), impl_->pools.end(), [&](const Impl::Pool &entry) {
          return entry.begin == range.begin && entry.end == end;
        });
    if (!registered) {
      report("released memory pool was not registered");
      return;
    }
  }
  const bool coherent = impl_->reconcile_range(range, "memory pool release");
  std::lock_guard lock(impl_->mutex);
  // Uploads deswizzled from pool storage read it when their batch runs: the
  // open batch and every committed one not yet waited for finish while the
  // released storage still holds the texture bytes. A failure among them
  // stays with the replay's next flush, as for other transfers.
  if (impl_->uploads_read_pools_locked())
    impl_->finish_uploads_locked();
  impl_->pool_generation = Impl::next_pool_generation.fetch_add(1);
  ++impl_->texture_generation;
  ++impl_->rare_stamp;
  impl_->last_pool = -1;
  const auto removed =
      std::erase_if(impl_->pools, [&](const Impl::Pool &entry) {
        if (entry.begin != range.begin || entry.end != end) return false;
        if (entry.owns_tracking)
          memory::WriteTracker::remove_range(reinterpret_cast<void*>(entry.begin));
        if (entry.buffer) impl_->released_buffers.push_back(entry.buffer);
        return true;
      });
  if (!removed)
    report("released memory pool was not registered");
  std::erase_if(impl_->pending_buffer_writes, [&](const Impl::Range &pending) {
    return overlaps(range.begin, range.size, pending.begin, pending.size);
  });
  for (auto &[serial, texture] : impl_->textures) {
    if (!overlaps(range.begin, range.size,
                  reinterpret_cast<std::uint64_t>(texture.state.memory),
                  texture.state.size))
      continue;
    texture.state.pool = nullptr;
    texture.state.memory = nullptr;
    texture->storage = nil;
    texture->views.clear();
    impl_->note_recreated_locked(serial);
    impl_->clear_dirty(*texture.shared);
    texture->cpu_snapshot_valid = false;
  }
  if (impl_->texture_pool_identity &&
      std::get<0>(*impl_->texture_pool_identity) == range.begin) {
    impl_->texture_pool_identity.reset();
    impl_->texture_slots.clear();
    impl_->new_descriptor_epoch_locked();
  }
  if (impl_->sampler_pool_identity &&
      std::get<0>(*impl_->sampler_pool_identity) == range.begin) {
    impl_->sampler_pool_identity.reset();
    impl_->samplers.clear();
    impl_->new_descriptor_epoch_locked();
  }
  if (!coherent)
    report("memory pool released without authoritative GPU texture readback");
}

void Resources::flush_pool(const nvn::MemoryPoolState &pool,
                           std::uint64_t offset, std::uint64_t size) {
  Impl::Range range;
  if (!impl_->range_for_pool(pool, offset, size, range) || !size)
    return;
  // A flush publishes CPU writes for the copies that read these bytes. A pool
  // that holds no texture with storage of its own has nothing to publish: the
  // guest's bytes are already what the GPU reads through the pool's buffer, and
  // the guest wrote them itself, so no page record is touched here.
  if (const auto *entry = impl_->find_pool(range.begin, range.begin + range.size);
      !entry || !entry->has_separate_textures)
    return;
  // On the hardware a flush only writes back the CPU's own stores: a texture
  // the GPU has written keeps that content unless the CPU also stored into
  // it. Such a texture is read back, the pages the CPU wrote since the
  // texture's last snapshot are restored over the readback, and it is
  // uploaded again below; other written textures are left alone.
  const std::uint64_t page = static_cast<std::uint64_t>(getpagesize());
  std::vector<Impl::Range> merged, restore;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->visit_texture_ranges_locked(range.begin, range.begin + range.size,
        [&](Impl::Texture& texture) {
          if (!texture->dirty) return;
          const auto memory = reinterpret_cast<std::uint64_t>(texture.state.memory);
          const auto first = std::max(memory, range.begin);
          const auto end = std::min(memory + texture.state.size, range.begin + range.size);
          const auto stamp = texture->cpu_snapshot_valid ? texture->cpu_stamp : 0;
          bool written = false;
          for (auto begin = first; begin < end;) {
            const auto stop = std::min((begin & ~(page - 1)) + page, end);
            if (!stamp || memory::WriteTracker::written_since(
                              reinterpret_cast<const void *>(begin), stop - begin, stamp)) {
              written = true;
              if (!restore.empty() && restore.back().begin + restore.back().size == begin)
                restore.back().size += stop - begin;
              else
                restore.push_back({begin, stop - begin});
            }
            begin = stop;
          }
          if (written) merged.push_back({memory, texture.state.size});
        });
  }
  if (!merged.empty()) {
    std::vector<std::vector<std::byte>> cpu_bytes;
    try {
      for (const auto &span : restore)
        cpu_bytes.emplace_back(reinterpret_cast<const std::byte *>(span.begin),
                               reinterpret_cast<const std::byte *>(span.begin) + span.size);
    } catch (...) {
      report("failed to preserve flushed CPU bytes");
      return;
    }
    for (const auto &span : merged)
      if (!impl_->reconcile_range(span, "memory pool flush"))
        return;
    {
      // As for a texel write: no GPU deswizzle may read the restored bytes
      // while they are written.
      std::lock_guard lock(impl_->mutex);
      if (impl_->uploads_read_pools_locked())
        impl_->finish_uploads_locked();
    }
    for (std::size_t index = 0; index < restore.size(); ++index) {
      auto *target = reinterpret_cast<void *>(restore[index].begin);
      memory::WriteTracker::prepare_host_write(target, restore[index].size);
      std::memcpy(target, cpu_bytes[index].data(), restore[index].size);
      memory::WriteTracker::note_store(target, restore[index].size);
    }
  }
  std::lock_guard lock(impl_->mutex);
  impl_->visit_texture_ranges_locked(range.begin, range.begin + range.size, [&](Impl::Texture& texture) {
    if (texture->in_place || !texture->storage || texture->dirty) return;
    if (!impl_->upload_texture_locked(texture))
      report_serial("failed to upload flushed texture serial", texture.state.serial);
  });
}

void Resources::invalidate_pool(const nvn::MemoryPoolState &pool,
                                std::uint64_t offset, std::uint64_t size) {
  Impl::Range range;
  if (!impl_->range_for_pool(pool, offset, size, range) || !size)
    return;
  // Native invalidation orders the CPU cache; it does not wait for GPU work.
  // Pending GPU buffer writes still owe their bytes to a CPU read. A texture
  // the GPU wrote keeps its content in its own storage, as the GPU's texture
  // writes never reach the pool: an invalidation does not read it back
  // (explicit texel reads do).
  std::atomic_thread_fence(std::memory_order_seq_cst);
  bool pending = false;
  {
    std::lock_guard lock(impl_->mutex);
    pending = impl_->pending_buffer_in_range_locked(range);
  }
  if (!pending) return;
  impl_->reconcile_range(range, "memory pool invalidation");
}

void Resources::register_texture(const nvn::TextureState &texture) {
  if (!texture.serial) {
    report("cannot register a texture without a serial");
    return;
  }
  // Declared before the lock, so the entries a pass drops are destroyed
  // after it is released.
  std::vector<Impl::TextureNode> dropped;
  std::lock_guard lock(impl_->mutex);
  ++impl_->texture_generation;
  impl_->touch_serial_locked(texture.serial);
  if (texture.memory && texture.size) {
    const auto begin = reinterpret_cast<std::uint64_t>(texture.memory);
    if (auto* pool = const_cast<Impl::Pool*>(impl_->find_pool(begin, begin + texture.size)))
      pool->texture_serials.push_back(texture.serial);
  }
  const auto [found, inserted] = impl_->textures.try_emplace(texture.serial);
  if (!inserted && found->second.live) {
    report_serial("texture serial is already registered", texture.serial);
    return;
  }
  if (!inserted && found->second->storage) {
    found->second.live = true;
    return;
  }
  found->second = {};
  found->second.state = texture;
  impl_->note_texture_range_locked(texture);
  impl_->note_registered_locked(texture);
  if (++impl_->registrations % 256 == 0) {
    impl_->sweep_superseded_locked();
    dropped = std::move(impl_->dropped_textures);
    impl_->dropped_textures.clear();
  }
}

void Resources::release_texture(const nvn::TextureState &texture) {
  std::lock_guard lock(impl_->mutex);
  const auto found = impl_->textures.find(texture.serial);
  if (found == impl_->textures.end()) {
    return;  // swept after a newer object took over its storage
  }
  // Recorded commands retain serial snapshots. Storage is kept until the
  // backend is destroyed because this interface has no replay retirement
  // fence for proving that every such command is gone.
  found->second.live = false;
}

bool Resources::register_texture_pool(const nvn::DescriptorPoolState &pool) {
  const auto identity = descriptor_identity(pool);
  if (!identity) {
    report("invalid texture descriptor pool backing");
    return false;
  }
  std::lock_guard lock(impl_->mutex);
  std::uint64_t address = 0;
  if (add_overflow(std::get<0>(*identity),
                   static_cast<std::uint64_t>(pool.offset), address) ||
      !impl_
           ->buffer_at_locked(
               address, static_cast<std::uint64_t>(pool.count) * 32, false)
           .buffer) {
    report("descriptor pool is outside registered GPU storage");
    return false;
  }
  auto &active = impl_->texture_pool_identity;
  if (active && active != identity) {
    report("multiple active texture descriptor pools require replay pool "
           "selection");
    return false;
  }
  if (!active) {
    ++impl_->texture_generation;
    impl_->texture_slots.clear();
    impl_->new_descriptor_epoch_locked();
  }
  active = identity;
  return true;
}

void Resources::release_texture_pool(const nvn::DescriptorPoolState &pool) {
  const auto identity = descriptor_identity(pool);
  std::lock_guard lock(impl_->mutex);
  if (identity && impl_->texture_pool_identity == identity) {
    impl_->texture_pool_identity.reset();
    ++impl_->texture_generation;
    impl_->texture_slots.clear();
    impl_->new_descriptor_epoch_locked();
  }
}

bool Resources::register_sampler_pool(const nvn::DescriptorPoolState &pool) {
  const auto identity = descriptor_identity(pool);
  if (!identity) {
    report("invalid sampler descriptor pool backing");
    return false;
  }
  std::lock_guard lock(impl_->mutex);
  std::uint64_t address = 0;
  if (add_overflow(std::get<0>(*identity),
                   static_cast<std::uint64_t>(pool.offset), address) ||
      !impl_
           ->buffer_at_locked(
               address, static_cast<std::uint64_t>(pool.count) * 32, false)
           .buffer) {
    report("descriptor pool is outside registered GPU storage");
    return false;
  }
  auto &active = impl_->sampler_pool_identity;
  if (active && active != identity) {
    report("multiple active sampler descriptor pools require replay pool "
           "selection");
    return false;
  }
  if (!active) {
    ++impl_->texture_generation;
    impl_->samplers.clear();
    impl_->new_descriptor_epoch_locked();
  }
  active = identity;
  return true;
}

void Resources::release_sampler_pool(const nvn::DescriptorPoolState &pool) {
  const auto identity = descriptor_identity(pool);
  std::lock_guard lock(impl_->mutex);
  if (identity && impl_->sampler_pool_identity == identity) {
    impl_->sampler_pool_identity.reset();
    ++impl_->texture_generation;
    impl_->samplers.clear();
    impl_->new_descriptor_epoch_locked();
  }
}

void Resources::register_texture_descriptor(std::int32_t id,
                                            const nvn::TextureState &texture,
                                            const nvn::TextureViewState *view) {
  if (id < 0) {
    report_id("negative texture descriptor ID", id);
    return;
  }
  Impl::TextureDescriptor descriptor;
  descriptor.texture = texture;
  descriptor.has_view = view != nullptr;
  if (view)
    descriptor.view = *view;
  std::lock_guard lock(impl_->mutex);
  ++impl_->texture_generation;
  if (static_cast<std::size_t>(id) >= impl_->texture_slots.size())
    impl_->texture_slots.resize(static_cast<std::size_t>(id) + 1024);
  auto &slot = impl_->texture_slots[id];
  slot = {};
  slot.registered = true;
  slot.descriptor = descriptor;
  impl_->texture_versions.bump(id);
}

void Resources::register_sampler_descriptor(
    std::int32_t id, const nvn::SamplerParameters &sampler) {
  if (id < 0) {
    report_id("negative sampler descriptor ID", id);
    return;
  }
  std::lock_guard lock(impl_->mutex);
  auto &entry = impl_->samplers[id];
  // The same parameters again resolve to the same sampler: the registration
  // keeps its number and nothing bound needs resolving again.
  if (entry.registration &&
      std::memcmp(&entry.parameters, &sampler, sizeof sampler) == 0)
    return;
  ++impl_->texture_generation;
  impl_->sampler_versions.bump(id);
  const auto registration =
      nvn::sampler_registrations.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (entry.registration) {
    // Bindings recorded since the previous registration keep its parameters
    // until they replay. The replay trails the guest by a few frames at most.
    constexpr std::size_t kept = 64;
    if (entry.earlier.size() == kept) {
      entry.dropped = entry.earlier.front().registration;
      entry.earlier.erase(entry.earlier.begin());
    }
    entry.earlier.push_back({entry.registration, entry.parameters, entry.state});
  }
  entry.parameters = sampler;
  entry.state = nil;
  entry.registration = registration;
}

bool Resources::read_texels(const nvn::TextureState &texture,
                            const nvn::TexelTransfer &transfer,
                            std::byte *out) {
  if (!out) {
    report("texture read destination is null");
    return false;
  }
  Impl::Range access{};
  {
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->entry_iterator_locked(texture);
    if (found == impl_->textures.end()) {
      report_serial("texture read refers to an unknown serial", texture.serial);
      return false;
    }
    access = {reinterpret_cast<std::uint64_t>(found->second.state.memory),
              found->second.state.size};
  }
  if (!impl_->reconcile_range(access, "texture CPU read"))
    return false;
  std::lock_guard lock(impl_->mutex);
  auto found = impl_->entry_iterator_locked(texture);
  if (found == impl_->textures.end())
    return false;
  const Impl::Range current{
      reinterpret_cast<std::uint64_t>(found->second.state.memory),
      found->second.state.size};
  if (impl_->pending_in_range_locked(current)) {
    report_serial("concurrent GPU write prevents texture CPU read for serial",
                  texture.serial);
    return false;
  }
  if (!impl_->readback_texture_locked(found->second))
    return false;
  if (!nvn::transfer_pool_texels(found->second.state, transfer, out, false)) {
    report_serial("texture CPU read transfer failed for serial",
                  texture.serial);
    return false;
  }
  return true;
}

bool Resources::write_texels(const nvn::TextureState &texture,
                             const nvn::TexelTransfer &transfer,
                             const std::byte *in) {
  if (!in) {
    report("texture write source is null");
    return false;
  }
  nvn::TextureState stored{};
  {
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->entry_iterator_locked(texture);
    if (found == impl_->textures.end()) {
      report_serial("texture write refers to an unknown serial",
                    texture.serial);
      return false;
    }
    stored = found->second.state;
  }
  std::vector<std::byte> source;
  nvn::TexelTransfer packed_transfer;
  if (!nvn::snapshot_texel_input(stored, transfer, in, source,
                                 packed_transfer)) {
    report("failed to capture the CPU texel write source");
    return false;
  }
  if (source.empty())
    return true;
  const Impl::Range access{reinterpret_cast<std::uint64_t>(stored.memory),
                           stored.size};
  if (!impl_->reconcile_range(access, "texture CPU write"))
    return false;
  std::lock_guard lock(impl_->mutex);
  auto found = impl_->entry_iterator_locked(texture);
  if (found == impl_->textures.end())
    return false;
  const Impl::Range current{
      reinterpret_cast<std::uint64_t>(found->second.state.memory),
      found->second.state.size};
  if (impl_->pending_in_range_locked(current)) {
    report_serial("concurrent GPU write prevents texture CPU write for serial",
                  texture.serial);
    return false;
  }
  if (!impl_->readback_texture_locked(found->second))
    return false;
  // A batch that deswizzles from the pool on the GPU must not read the bytes
  // this write replaces while it runs.
  if (impl_->uploads_read_pools_locked())
    impl_->finish_uploads_locked();
  memory::WriteTracker::prepare_host_write(found->second.state.memory,
                                           found->second.state.size);
  if (!nvn::transfer_pool_texels(found->second.state, packed_transfer,
                                 source.data(), true)) {
    memory::WriteTracker::note_store(found->second.state.memory,
                                     found->second.state.size);
    report_serial("texture CPU write transfer failed for serial",
                  texture.serial);
    return false;
  }
  memory::WriteTracker::note_store(found->second.state.memory,
                                   found->second.state.size);
  impl_->invalidate_aliases_locked(found->second);
  if (found->second->storage && !found->second->in_place &&
      !impl_->upload_texture_locked(found->second)) {
    report_serial("texture CPU write upload failed for serial", texture.serial);
    return false;
  }
  return true;
}

} // namespace armigo::platform::metal::nvn_driver
