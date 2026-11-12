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

#include "nvn_submission_owner.h"
#include "nvn_border_sampler.h"

#include "nvn/backend.h"
#include "nvn_descriptor_identity.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

// False under ARMIGO_NVN_IMAGE_ATOMICS=0: atomic storage images are bound as
// any other storage image, with no R32 storage of their own.
bool image_atomics_enabled() noexcept;

// Logical upload batch range owned by a consuming replay segment. Queue
// publication supplies readiness; the range supplies error attribution.
struct UploadFence {
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

  explicit Resources(id<MTLDevice> device,
                     std::shared_ptr<Metal4SubmissionOwner> submission = {});
  [[nodiscard]] std::shared_ptr<Metal4SubmissionOwner> submission_owner() const;
  void set_upload_publication(UploadPublication publication);
  struct UploadObservation {
    enum class Kind { batch, allocator_wait } kind;
    std::uint64_t batch = 0;
    id<MTLBuffer> staging = nil;
    id<MTLTexture> target = nil;
  };
  // Synthetic-test observations run under the resource lock and must not
  // re-enter Resources. Batch observations expose representative allocations.
  using UploadObserver = std::function<void(const UploadObservation&)>;
  void set_upload_observer(UploadObserver observer);
  struct NativeSnapshot {
    id<MTLTexture> texture = nil;
    std::shared_ptr<const void> allocation;
    std::function<std::string()> producer_error;
    std::function<void()> release_read;
  };
  // Resolve and pin one physical allocation under the same resource lock.
  [[nodiscard]] NativeSnapshot capture_native(const armigo::nvn::TextureState& texture);
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
  // pool-set generation, which moves when a pool is registered, when one
  // starts its release, or when a handed-out address covers a range the table
  // did not. When it differs from `known`, `pools` is replaced with the
  // GPU-accessible pools that meet a handed-out address.
  // ARMIGO_NVN_ADDRESSED_POOLS=0 lists every GPU-accessible pool.
  std::uint64_t claim_guest_memory_pools(std::uint64_t known, std::vector<GuestMemoryPool> &pools);
  [[nodiscard]] std::uint64_t pool_generation() const noexcept;
  // Pools left out of the latest claimed table because no handed-out address
  // meets them, and the bytes those pools hold. A pool the GPU cannot access
  // is not counted. Both are zero when the addressed-pool filter is off.
  struct GuestMemoryAdmission {
    std::size_t omitted_pools = 0;
    std::uint64_t omitted_bytes = 0;
  };
  [[nodiscard]] GuestMemoryAdmission guest_memory_admission() const;
  // True when `address` lies in a GPU-accessible pool the table would leave
  // out. False when the filter is off, or no such pool holds the address.
  [[nodiscard]] bool address_in_unaddressed_pool(std::uint64_t address) const;
  // The registered pools nearest below and above `address` (begin, end,
  // flags, releasing), for a guest-memory miss report.
  [[nodiscard]] std::string describe_guest_address(std::uint64_t address) const;

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
  // Storage for an image atomic. R32Uint/R32Sint storage gains shader-write
  // usage when it lacks it and is remembered as atomic-capable. A 32-bit
  // colour texture the GPU has written (not one whose texels still come from
  // guest memory) is replaced by R32 storage, signed when `is_signed` is set;
  // its original format stays a view of that storage, every alias moves with
  // it, and `previous` is the storage whose contents the caller copies.
  // `previous` is nil when nothing was replaced. Nil when this storage cannot
  // run the atomic.
  [[nodiscard]] id<MTLTexture> upgrade_for_image_atomic(std::uint64_t serial, bool is_signed,
                                                        __strong id<MTLTexture> &previous);
  // Root storage was created with shader-atomic usage and is R32 integer.
  // Metal does not report that usage on the texture.
  [[nodiscard]] bool image_atomic_capable(std::uint64_t serial);
  // True when upgrade_for_image_atomic would replace this storage (`previous`
  // would be set). Signedness selects the R32 format and does not change
  // whether the storage is replaced. False when this serial has no storage.
  [[nodiscard]] bool image_atomic_needs_upgrade(std::uint64_t serial, bool is_signed);
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
  // Resolve the image's backing before making its format-changing view.
  [[nodiscard]] id<MTLTexture> image_base_for_id(std::int32_t texture_id, DescriptorStamp& stamp);
  // An identity-channel image view; sRGB images use their linear twin.
  [[nodiscard]] id<MTLTexture> raw_image_view(std::int32_t texture_id, id<MTLTexture> texture);
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
  // Consumes the unsupported-state diagnostic from this thread's last sampler lookup.
  [[nodiscard]] std::string take_sampler_failure();
  // A combined mipmapped sampler whose integer min/max clamp selects one level.
  [[nodiscard]] std::optional<std::uint32_t> sampler_pinned_level(
      std::int32_t sampler_id, std::uint64_t registrations = UINT64_MAX) const;
  [[nodiscard]] id<MTLSamplerState> sampler_for_id(std::int32_t sampler_id);
  // The sampler visible after `registrations` publications. Replay supplies
  // its immutable submission boundary; explicit historical lookups stay strict.
  [[nodiscard]] id<MTLSamplerState> sampler_for_id(std::int32_t sampler_id,
                                                   std::uint64_t registrations);
  // Exact primary state and color for a reflected coverage-v1 sampling pair.
  // The binding registration selects parameters atomically with the state.
  [[nodiscard]] CoverageSampler coverage_sampler_for_id(std::int32_t sampler_id,
                                                        std::uint64_t registrations);
  // A view of a descriptor's sampled texture in the type a shader declares,
  // when that differs from the texture's own type: a Metal argument of another
  // texture type reads undefined values. Layered and two-dimensional types
  // convert to each other (a two-dimensional view is the first layer); the view
  // is kept in the descriptor's slot until another type or texture needs it.
  // A floating-point typed-handle element reads zero when no view fits;
  // other unsupported pairs return nil.
  [[nodiscard]] id<MTLTexture> fitted_texture(std::int32_t texture_id, id<MTLTexture> texture,
                                              MTLTextureType type, bool typed_handle = false,
                                              MTLPixelFormat empty_format = MTLPixelFormatRGBA8Unorm);

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
  // The generation of the newest GPU write or retired upload storage, and
  // completion up to that generation once the submission has finished.
  [[nodiscard]] std::uint64_t gpu_write_generation() const;
  void complete_gpu_writes(std::uint64_t generation);
  // The replay worker starts walking a submission. buffer_address hands out
  // unretained buffers, so a released pool's buffer stays alive until the walk
  // after its release begins.
  void begin_walk();
  void end_walk();
  // Publishes pending producers to their reserved positions without a GPU wait.
  bool publish_uploads();
  // Commits the batched texture uploads and waits for them (see the replay's
  // ReplayResources::flush_uploads).
  bool flush_uploads();
  // Seals upload producers into the replay publication stream. The logical
  // range attributes failures to the next consuming segment; allocator reuse
  // and the staging budget wait for actual upload hardware completion.
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
  // Uploads of a texture whose storage the GPU had written and the CPU
  // rewrote afterwards.
  [[nodiscard]] std::uint64_t cpu_rewrites_after_gpu() const;

  bool register_pool(const armigo::nvn::MemoryPoolState &pool) override;
  void release_pool(const armigo::nvn::MemoryPoolState &pool) override;
  void note_addressed(std::uint64_t gpu_address, std::uint64_t size) override;
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
  void clear_texture_descriptor(std::int32_t id) override;
  void register_sampler_descriptor(
      std::int32_t id, const armigo::nvn::SamplerParameters &sampler) override;

  // Descriptor registration and storage-replacement notes. Generation 1 is the
  // epoch before the first note. A missing note was overwritten or has not
  // been written. kind 0 is a texture ID, 1 a sampler ID, 2 a memory pool
  // added or removed.
  struct RegistrationNote {
    std::int32_t id = 0;
    std::uint32_t kind = 0;
  };
  [[nodiscard]] std::uint64_t registration_log_generation() const;
  [[nodiscard]] bool read_registration_note(std::uint64_t generation, RegistrationNote &note) const;
  [[nodiscard]] std::uint64_t storage_log_generation() const;
  [[nodiscard]] bool read_storage_note(std::uint64_t generation, std::uint64_t &serial) const;
  [[nodiscard]] std::vector<std::int32_t> descriptor_ids_for_serial(std::uint64_t serial) const;

  // One past the highest registered descriptor, or 0 when none is registered.
  // None of these create a texture or a sampler.
  [[nodiscard]] bool texture_id_empty(std::int32_t id) const;
  [[nodiscard]] std::int32_t registered_texture_count() const;
  [[nodiscard]] std::int32_t sampler_count() const;
  [[nodiscard]] bool texture_registered(std::int32_t id) const;
  [[nodiscard]] bool texture_storage_made(std::int32_t id) const;
  [[nodiscard]] bool texture_storage_shared(std::int32_t id) const;
  [[nodiscard]] bool texture_compressible(std::int32_t id) const;
  [[nodiscard]] std::uint64_t texture_storage_size(std::int32_t id) const;
  [[nodiscard]] bool texture_storage_live(std::int32_t id) const;
  [[nodiscard]] std::uint64_t texture_serial(std::int32_t id) const;
  [[nodiscard]] std::uint64_t texture_descriptor_key(std::int32_t id) const;
  [[nodiscard]] bool texture_is_depth(std::int32_t id) const;
  [[nodiscard]] std::uint64_t texture_storage_key(std::int32_t id) const;
  [[nodiscard]] bool texture_read_identity(std::int32_t id, TextureReadIdentity &identity) const;
  [[nodiscard]] std::uint64_t texture_empty_generation() const;
  [[nodiscard]] std::uint64_t texture_pool_generation() const;
  void note_pool_depth_read(std::uint64_t serial, std::uint64_t storage_key);
  [[nodiscard]] bool pool_depth_read(std::uint64_t serial, std::uint64_t storage_key) const;

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
