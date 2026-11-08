// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <chrono>

#import <Metal/Metal.h>

#include "nvn/backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace armigo::platform::metal::nvn_driver {

// What the replay queue waits for on the GPU before it runs work that may read
// textures uploaded by committed batches: the event reaching the value means
// those batches have completed. A null event needs no wait.
//
// The values (floor, value] are the upload batches this fence is responsible
// for: a batch that failed in that range is the failure of the segment that
// carries it (Resources::uploads_failed).
struct UploadFence {
  id<MTLEvent> event = nil;
  std::uint64_t value = 0, floor = 0;
};

class Resources final : public armigo::nvn::ResourceBackend {
public:
  struct BufferSpan {
    id<MTLBuffer> buffer;
    std::uint64_t offset;
  };
  // A buffer range by its pool's raw pointers, for binding paths: the buffer
  // belongs to the pool and is not retained, and its length and GPU and CPU
  // addresses come from the per-thread lookup cache instead of Objective-C
  // property calls.
  struct BufferAddress {
    id<MTLBuffer> __unsafe_unretained buffer = nil;
    std::uint64_t offset = 0, length = 0, gpu = 0;
    std::byte *cpu = nullptr;
  };

  // The hook must wait until every replay submission made before the call is
  // complete. It returns false if completion cannot be established.
  using CompletionHook = std::function<bool()>;

  explicit Resources(id<MTLDevice> device);
  ~Resources() override;

  Resources(const Resources &) = delete;
  Resources &operator=(const Resources &) = delete;

  void set_completion_hook(CompletionHook hook);
  // How long a pool release waits for the submissions that leased
  // guest-memory tables over its range (CPU reconciliation does not wait for
  // leases: the guest's own sync waits order its reads).
  void set_lease_wait_timeout(std::chrono::nanoseconds timeout);

  // Moves whenever a pool, texture, sampler or descriptor registration changes,
  // so a cached binding that resolved a handle or address can tell that it must
  // be resolved again.
  [[nodiscard]] std::uint64_t registration_generation() const;
  [[nodiscard]] bool overlaps_texture_backing(std::uint64_t address, std::uint64_t size) const;

  // A registered pool as shaders' guest-memory table lists it: its guest
  // range, its zero-copy buffer and the buffer's device address of `begin`.
  struct GuestMemoryPool {
    std::uint64_t begin = 0, end = 0, gpu = 0;
    id<MTLBuffer> buffer = nil;
  };
  // Leases the pools a submission's guest-memory table names until the
  // submission retires: a pool released meanwhile waits for it. Returns the
  // pool-set generation, which moves whenever a pool is registered or starts
  // its release; when it differs from `known`, `pools` is replaced with that
  // set's GPU-accessible pools.
  std::uint64_t claim_guest_memory_pools(std::uint64_t known, std::vector<GuestMemoryPool> &pools);
  [[nodiscard]] std::uint64_t pool_generation() const noexcept;

  [[nodiscard]] BufferSpan buffer_at(std::uint64_t gpu_address,
                                     std::uint64_t size);
  [[nodiscard]] BufferAddress buffer_address(std::uint64_t gpu_address,
                                             std::uint64_t size);
  // Attachment and copy storage preserves every aspect and ignores sampling swizzles.
  [[nodiscard]] id<MTLTexture>
  texture(const armigo::nvn::TextureState &texture,
          const armigo::nvn::TextureViewState *view);
  // What the most recent texture() call on this thread that returned nil was
  // given, and the last resource diagnostic it raised; cleared by the call.
  // For skip diagnostics only.
  [[nodiscard]] static std::string take_texture_failure();
  // The format and sample count of a render attachment over this texture when
  // its storage and (for a view that sets a format) view already exist;
  // invalid otherwise. Creates nothing.
  struct AttachmentFormat {
    MTLPixelFormat format = MTLPixelFormatInvalid;
    NSUInteger samples = 0;
  };
  [[nodiscard]] AttachmentFormat
  peek_attachment(const armigo::nvn::TextureState &texture,
                  const armigo::nvn::TextureViewState *view);
  // Descriptor sampling applies the selected component mapping and depth/stencil aspect.
  [[nodiscard]] id<MTLTexture> texture_for_id(std::int32_t texture_id);
  // Gives the storage of texture `serial` shader-write usage for a storage
  // image write (a texture registered as an image without the IMAGE flag):
  // new storage with that usage replaces it for every alias, and its cached
  // views and descriptor lookups resolve again. The serial stamp and the rare
  // stamp both move with that replacement. `previous` is the storage
  // replaced, whose contents the caller copies before any use of the new
  // one; nil when the storage was writable already. Nil when no such
  // texture has storage.
  [[nodiscard]] id<MTLTexture> upgrade_for_image_write(std::uint64_t serial, __strong id<MTLTexture> &previous);
  // What a descriptor lookup depended on: the registrations of its ID, the
  // descriptor pools, its texture's storage entry. A caller that kept the
  // result may use it again while texture_stamp_current holds, without the
  // resources lock and without other IDs' registrations moving it; guest
  // stores to the texture's memory are the caller's to watch. Not valid when
  // the lookup found no texture.
  struct DescriptorStamp {
    std::uint64_t epoch = 0, rare = 0, serial = 0, serial_stamp = 0;
    // Guest storage read by this descriptor, including separately backed views.
    std::uint64_t backing_begin = 0, backing_size = 0;
    std::uint32_t slot = 0;
    bool valid = false;
  };
  // texture_for_id, also stamping the result.
  [[nodiscard]] id<MTLTexture> texture_for_id(std::int32_t texture_id, DescriptorStamp &stamp);
  [[nodiscard]] bool texture_stamp_current(std::int32_t texture_id,
                                           const DescriptorStamp &stamp) const;
  // Texture descriptor lookups since creation, and those that resolved the
  // descriptor again instead of using the cached view.
  struct TextureLookupCounts {
    std::uint64_t lookups = 0, misses = 0;
    // Sampler lookups that took the resource lock instead of a cached result.
    std::uint64_t sampler_misses = 0;
  };
  [[nodiscard]] TextureLookupCounts texture_lookup_counts() const;
  [[nodiscard]] id<MTLSamplerState> sampler_for_id(std::int32_t sampler_id);
  // The sampler as registered when a binding recorded after `registrations`
  // sampler registrations (nvn::sampler_registrations) was made.
  [[nodiscard]] id<MTLSamplerState> sampler_for_id(std::int32_t sampler_id,
                                                   std::uint64_t registrations);
  // A view of a descriptor's sampled texture in the type a shader declares,
  // when that differs from the texture's own type: a Metal argument of another
  // texture type reads undefined values. Layered and two-dimensional types
  // convert to each other (a two-dimensional view is the first layer); the view
  // is kept in the descriptor's slot until another type or texture needs it.
  // Nil when the types have no such view.
  [[nodiscard]] id<MTLTexture> fitted_texture(std::int32_t texture_id, id<MTLTexture> texture,
                                              MTLTextureType type);

  // Whether this texture shares the guest's bytes directly (a linear,
  // buffer-backed Metal texture) instead of a private tiled copy.
  [[nodiscard]] bool texture_is_in_place(const armigo::nvn::TextureState &texture);
  void note_gpu_write(const armigo::nvn::TextureState &texture);
  void note_gpu_write(std::uint64_t gpu_address, std::uint64_t size);
  // A GPU write found once the work that made it completed (a guest-memory
  // write log names whole pools): CPU snapshots see the range changed, and
  // textures with storage of their own over it compare their backing with the
  // bytes of their last upload at their next use, uploading again only when
  // they differ. A texture the GPU wrote keeps its storage.
  void note_completed_gpu_write(std::uint64_t gpu_address, std::uint64_t size);
  // Whether a texture's data over the range comes from guest memory at its
  // next use: a texture there has storage of its own the GPU has not written,
  // or none is registered yet.
  [[nodiscard]] bool backing_feeds_texture(std::uint64_t gpu_address, std::uint64_t size);
  // The generation of the newest GPU write noted so far, and the completion
  // of every GPU write noted up to a generation once its work has finished.
  [[nodiscard]] std::uint64_t gpu_write_generation() const;
  void complete_gpu_writes(std::uint64_t generation);
  // The replay worker starts walking a submission. buffer_address hands out
  // unretained buffers, so a released pool's buffer stays alive until the walk
  // after its release begins.
  void begin_walk();
  // Commits the batched texture uploads and waits for them (see the replay's
  // ReplayResources::flush_uploads).
  bool flush_uploads();
  // Commits the batched texture uploads, normally without waiting for them:
  // the fence orders the replay's next committed segment after them on the
  // GPU. The first fence with batches still in flight waits for them on the
  // CPU instead, once per Resources (see the definition). The
  // staging bytes of unsettled batches stay within a fixed budget: a commit
  // beyond it waits for the oldest batch.
  void commit_uploads(UploadFence &fence);
  // Whether a batch in the fence's range failed. The replay asks when the
  // segment that carries the fence retires, and fails that segment: the
  // textures its draws sampled are missing content. Each failed batch is
  // reported to exactly one fence.
  [[nodiscard]] bool uploads_failed(const UploadFence &fence);
  // Committed upload batches not yet settled.
  [[nodiscard]] std::size_t unsettled_uploads() const;
  // Waits up to timeout for the GPU deswizzle kernel, which compiles in the
  // background; returns whether it is available.
  bool wait_for_gpu_deswizzle(std::chrono::milliseconds timeout);
  // Texture uploads since creation: those deswizzled on the GPU straight from
  // their memory pool and those prepared on the CPU.
  struct UploadCounts {
    std::uint64_t gpu = 0, cpu = 0;
  };
  [[nodiscard]] UploadCounts upload_counts() const;

  bool register_pool(const armigo::nvn::MemoryPoolState &pool) override;
  void release_pool(const armigo::nvn::MemoryPoolState &pool) override;
  void prepare_pool_write(const armigo::nvn::MemoryPoolState &pool,
                          std::uint64_t offset, std::uint64_t size) override;
  void flush_pool(const armigo::nvn::MemoryPoolState &pool,
                  std::uint64_t offset, std::uint64_t size) override;
  void invalidate_pool(const armigo::nvn::MemoryPoolState &pool,
                       std::uint64_t offset, std::uint64_t size) override;

  void register_texture(const armigo::nvn::TextureState &texture) override;
  void release_texture(const armigo::nvn::TextureState &texture) override;
  bool
  register_texture_pool(const armigo::nvn::DescriptorPoolState &pool) override;
  bool
  register_sampler_pool(const armigo::nvn::DescriptorPoolState &pool) override;
  void
  release_texture_pool(const armigo::nvn::DescriptorPoolState &pool) override;
  void
  release_sampler_pool(const armigo::nvn::DescriptorPoolState &pool) override;

  void register_texture_descriptor(
      std::int32_t id, const armigo::nvn::TextureState &texture,
      const armigo::nvn::TextureViewState *view) override;
  void register_sampler_descriptor(
      std::int32_t id, const armigo::nvn::SamplerParameters &sampler) override;

  bool read_texels(const armigo::nvn::TextureState &texture,
                   const armigo::nvn::TexelTransfer &transfer,
                   std::byte *out) override;
  bool write_texels(const armigo::nvn::TextureState &texture,
                    const armigo::nvn::TexelTransfer &transfer,
                    const std::byte *in) override;

private:
  id<MTLTexture> texture_lookup(const armigo::nvn::TextureState &texture,
                                const armigo::nvn::TextureViewState *view);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using BufferSpan = Resources::BufferSpan;
using BufferAddress = Resources::BufferAddress;

} // namespace armigo::platform::metal::nvn_driver
