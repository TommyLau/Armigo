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

#import <Metal/Metal.h>

#include "nvn_submission_owner.h"
#include "nvn_border_sampler.h"

#include "../../nvn/backend.h"
#include "nvn_descriptor_identity.h"

#include <chrono>
#include <cstddef>
#include <optional>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __has_include("nvn_resources.h")
#include "nvn_resources.h"
#define ARMIGO_NVN_REPLAY_HAS_RESOURCES 1
#endif

namespace armigo::platform {
class MetalShaderPackage;
}

namespace armigo::platform::metal::nvn_driver {

#if !defined(ARMIGO_NVN_REPLAY_HAS_RESOURCES)

// Temporary seam for the resource lane. The production resource owner can
// replace this declaration by providing nvn_resources.h with the same ABI.
// Tests implement it directly, so replay never manufactures guest resources.
struct BufferSpan {
    id<MTLBuffer> buffer = nil;
    std::uint64_t offset = 0;
};
struct BufferAddress {
    id<MTLBuffer> __unsafe_unretained buffer = nil;
    std::uint64_t offset = 0, length = 0, gpu = 0;
    std::byte* cpu = nullptr;
};
struct UploadFence {
    std::uint64_t value = 0, floor = 0;
};

#endif

// Injectable lookup boundary; the production overload forwards to Resources.
class ReplayResources {
public:
    virtual ~ReplayResources() = default;
    // For this provider's entire lifetime, every successfully resolved buffer
    // byte has CPU address equal to its NVN address, including overlapping
    // registrations. Unclassified or remapped providers must return false.
    [[nodiscard]] virtual bool buffer_addresses_are_cpu() const { return false; }

    // Resource coherence may request completion while replay resolves a binding.
    virtual void set_completion_hook(std::function<bool()> hook) { (void)hook; }
    // Describes the most recent failed texture lookup on this thread, for
    // skip diagnostics; empty when the implementation keeps no record.
    [[nodiscard]] virtual std::string take_texture_failure() { return {}; }
    [[nodiscard]] virtual BufferSpan
    buffer_at(std::uint64_t address, std::uint64_t size) = 0;
    // Leases the GPU-accessible pools for a submission's guest-memory table
    // (Resources::claim_guest_memory_pools): the pool-set generation, with
    // `pools` replaced when it differs from `known`. Resources lists a pool
    // only once a handed-out address meets it, unless
    // ARMIGO_NVN_ADDRESSED_POOLS=0. 0 when the provider lists no pools; the
    // replay refuses stages that read guest memory then, and when the set is
    // empty.
    virtual std::uint64_t claim_guest_memory_pools(std::uint64_t, std::vector<Resources::GuestMemoryPool>& pools) {
        pools.clear();
        return 0;
    }
    // Pools the latest claim left out as never addressed. Zero when the
    // provider does not filter.
    [[nodiscard]] virtual Resources::GuestMemoryAdmission guest_memory_admission() const { return {}; }
    // True when `address` lies in a pool the guest-memory table left out
    // because nothing had handed out an address in it.
    [[nodiscard]] virtual bool address_in_unaddressed_pool(std::uint64_t address) const {
        (void)address;
        return false;
    }
    [[nodiscard]] virtual std::string describe_guest_address(std::uint64_t address) const {
        (void)address;
        return {};
    }
    // The same range by raw pointers; the buffer must outlive the binding
    // that uses it, as a registered pool's buffer does.
    [[nodiscard]] virtual BufferAddress
    buffer_address(std::uint64_t address, std::uint64_t size) {
        const auto span = buffer_at(address, size);
        if (!span.buffer) return {};
        return {span.buffer, span.offset, span.buffer.length, span.buffer.gpuAddress + span.offset,
                span.buffer.contents ? static_cast<std::byte*>(span.buffer.contents) + span.offset
                                     : nullptr};
    }
    [[nodiscard]] virtual id<MTLTexture>
    texture(const nvn::TextureState& state, const nvn::TextureViewState* view) = 0;
    [[nodiscard]] virtual id<MTLTexture> texture_for_id(std::int32_t id) = 0;
    using DescriptorStamp = Resources::DescriptorStamp;
    // texture_for_id with the stamp the result stays valid under (see
    // Resources::DescriptorStamp). Without one the caller resolves again
    // whenever registration_generation moves.
    [[nodiscard]] virtual id<MTLTexture> texture_for_id_stamped(std::int32_t id, DescriptorStamp& stamp) {
        stamp = {};
        return texture_for_id(id);
    }
    // Providers with descriptor views resolve the parent backing here. A
    // provider whose descriptors already name backings uses its normal lookup.
    [[nodiscard]] virtual id<MTLTexture> image_base_for_id(std::int32_t id, DescriptorStamp& stamp) {
        return texture_for_id_stamped(id, stamp);
    }
    // Required format or channel conversion; providers without it refuse the use.
    [[nodiscard]] virtual id<MTLTexture> raw_image_view(std::int32_t, id<MTLTexture>) {
        return nil;
    }
    [[nodiscard]] virtual bool texture_stamp_current(std::int32_t id, const DescriptorStamp& stamp) const {
        (void)id; (void)stamp;
        return false;
    }
    // Resources::upgrade_for_image_write; providers without it cannot.
    [[nodiscard]] virtual id<MTLTexture> upgrade_for_image_write(std::uint64_t, __strong id<MTLTexture>& previous) {
        previous = nil;
        return nil;
    }
    // Resources::upgrade_for_image_atomic; providers without it cannot.
    [[nodiscard]] virtual id<MTLTexture> upgrade_for_image_atomic(std::uint64_t, bool,
                                                                  __strong id<MTLTexture>& previous) {
        previous = nil;
        return nil;
    }
    // Resources::image_atomic_capable. Metal does not report shader-atomic usage.
    [[nodiscard]] virtual bool image_atomic_capable(std::uint64_t) { return false; }
    // True when upgrade_for_image_atomic would replace this storage. Signedness
    // selects the destination format and does not change the result. Providers
    // without storage report false.
    [[nodiscard]] virtual bool image_atomic_needs_upgrade(std::uint64_t, bool) { return false; }
    [[nodiscard]] virtual std::string take_sampler_failure() { return {}; }
    [[nodiscard]] virtual std::optional<std::uint32_t> sampler_pinned_level(
        std::int32_t, std::uint64_t) const { return {}; }
    [[nodiscard]] virtual id<MTLSamplerState> sampler_for_id(std::int32_t id) = 0;
    // The sampler visible at the submission's registration boundary.
    [[nodiscard]] virtual id<MTLSamplerState> sampler_for_id(std::int32_t id, std::uint64_t registrations) {
        (void)registrations;
        return sampler_for_id(id);
    }
    [[nodiscard]] virtual CoverageSampler coverage_sampler_for_id(std::int32_t id,
                                                                  std::uint64_t registrations) {
        return {sampler_for_id(id, registrations), {}, false};
    }
    // The pixel format and sample count a render attachment over this texture
    // would have, when its storage already exists; invalid when creating it
    // would be needed. Never creates or uploads anything (the pipeline
    // look-ahead calls it for draws the replay has not reached).
    struct AttachmentFormat {
        MTLPixelFormat format = MTLPixelFormatInvalid;
        NSUInteger samples = 0;
    };
    [[nodiscard]] virtual AttachmentFormat peek_attachment(const nvn::TextureState& state,
                                                           const nvn::TextureViewState* view) {
        (void)state; (void)view;
        return {};
    }
    // A view in the declared type, or a zero for an incompatible floating-point
    // typed-handle element. A nil source requests the typed zero entry; a nil
    // result means that this resource provider cannot provide the view.
    [[nodiscard]] virtual id<MTLTexture> fitted_texture(std::int32_t texture_id, id<MTLTexture> texture,
                                                        MTLTextureType type, bool typed_handle = false,
                                              MTLPixelFormat empty_format = MTLPixelFormatRGBA8Unorm) {
        (void)texture_id; (void)texture; (void)type; (void)typed_handle; (void)empty_format;
        return nil;
    }
    // The replay worker starts walking a submission. Buffers it resolved while
    // walking an earlier one may be released from then on.
    virtual void begin_walk() {}
    virtual void end_walk() {}
    [[nodiscard]] virtual std::shared_ptr<Metal4SubmissionOwner> submission_owner() const { return {}; }
    virtual void set_upload_publication(UploadPublication) {}
    virtual bool publish_uploads() { return flush_uploads(); }
    // Explicit CPU consumers wait for upload hardware completion.
    virtual bool flush_uploads() { return true; }
    // Seals upload producers into the shared publication stream. The logical
    // range reports failures when its consuming replay segment retires.
    virtual void commit_uploads(UploadFence& fence) {
        fence = {};
        flush_uploads();
    }
    // Whether an upload batch the fence is responsible for failed; asked when
    // the segment carrying the fence retires.
    virtual bool uploads_failed(const UploadFence& fence) {
        (void)fence;
        return false;
    }
    virtual void note_gpu_write(const nvn::TextureState& state) = 0;
    // Resources::note_completed_gpu_write and backing_feeds_texture; a
    // provider without texture storage of its own has nothing to update, and
    // an unclassified one waits for every guest-memory store.
    virtual void note_completed_gpu_write(std::uint64_t, std::uint64_t) {}
    [[nodiscard]] virtual bool backing_feeds_texture(std::uint64_t, std::uint64_t) { return true; }
    virtual void note_gpu_write(std::uint64_t address, std::uint64_t size) = 0;
    // Generation of the newest noted GPU write, and completion of every write
    // noted up to a generation once the submissions that made it retired.
    [[nodiscard]] virtual std::uint64_t gpu_write_generation() const { return 0; }
    virtual void complete_gpu_writes(std::uint64_t generation) { (void)generation; }
    // Moves whenever a registration changes what a cached binding would resolve.
    [[nodiscard]] virtual std::uint64_t registration_generation() const { return 0; }
    // An unclassified provider must retain the ordered report path.
    [[nodiscard]] virtual bool overlaps_texture_backing(std::uint64_t, std::uint64_t) const { return true; }
    // Whether the texture shares the guest's bytes directly (linear storage) or
    // a private tiled copy backs it: a GPU write to it writes guest memory.
    [[nodiscard]] virtual bool texture_is_in_place(const nvn::TextureState&) { return false; }
    // Descriptor lookups and those that resolved the descriptor again.
    // Diagnostics only.
    struct LookupCounts {
        std::uint64_t lookups = 0, misses = 0;
    };
    [[nodiscard]] virtual LookupCounts texture_lookup_counts() const { return {}; }

    // Descriptor registration and storage-replacement notes. Generation 1 is
    // the epoch before the first note. A missing note was overwritten or has
    // not been written.
    struct RegistrationNote {
        std::int32_t id = 0;
        std::uint32_t kind = 0;
    };
    [[nodiscard]] virtual std::uint64_t registration_log_generation() const { return 1; }
    [[nodiscard]] virtual bool read_registration_note(std::uint64_t, RegistrationNote&) const { return false; }
    [[nodiscard]] virtual std::uint64_t storage_log_generation() const { return 1; }
    [[nodiscard]] virtual bool read_storage_note(std::uint64_t, std::uint64_t&) const { return false; }
    [[nodiscard]] virtual std::vector<std::int32_t> descriptor_ids_for_serial(std::uint64_t) const { return {}; }

    // One past the highest registered descriptor, or 0 when none is registered.
    // The queries do not create textures or samplers.
    // Only reference-defined empty IDs may bypass a failed descriptor lookup.
    [[nodiscard]] virtual bool texture_id_empty(std::int32_t) const { return false; }
    [[nodiscard]] virtual std::int32_t registered_texture_count() const { return 0; }
    [[nodiscard]] virtual std::int32_t sampler_count() const { return 0; }
    [[nodiscard]] virtual bool texture_registered(std::int32_t) const { return false; }
    [[nodiscard]] virtual bool texture_storage_made(std::int32_t) const { return false; }
    [[nodiscard]] virtual bool texture_storage_shared(std::int32_t) const { return false; }
    [[nodiscard]] virtual bool texture_compressible(std::int32_t) const { return false; }
    [[nodiscard]] virtual std::uint64_t texture_storage_size(std::int32_t) const { return 1; }
    [[nodiscard]] virtual bool texture_storage_live(std::int32_t) const { return false; }
    [[nodiscard]] virtual std::uint64_t texture_serial(std::int32_t) const { return 0; }
    [[nodiscard]] virtual std::uint64_t texture_descriptor_key(std::int32_t) const { return 0; }
    // Pool-capable providers capture this identity in one lock, including
    // historical empty transitions that registration-log gaps cannot erase.
    [[nodiscard]] virtual bool texture_read_identity(std::int32_t, TextureReadIdentity&) const { return false; }
    [[nodiscard]] virtual std::uint64_t texture_empty_generation() const { return 1; }
    [[nodiscard]] virtual bool texture_is_depth(std::int32_t) const { return false; }
    [[nodiscard]] virtual std::uint64_t texture_storage_key(std::int32_t) const { return 0; }
    [[nodiscard]] virtual std::uint64_t texture_pool_generation() const { return 1; }
    virtual void note_pool_depth_read(std::uint64_t, std::uint64_t) {}
    [[nodiscard]] virtual bool pool_depth_read(std::uint64_t, std::uint64_t) const { return false; }
};

#if !defined(ARMIGO_NVN_REPLAY_HAS_RESOURCES)
using Resources = ReplayResources;
#endif

enum class ReplayErrorCode : std::uint8_t {
    none,
    invalid_command,
    missing_resource,
    missing_program,
    unsupported,
    allocation,
    encoding,
    execution,
    // A required resource cannot be approximated or recovered by skipping work.
    unsupported_resource,
};

struct ReplayError {
    ReplayErrorCode code = ReplayErrorCode::none;
    std::string detail;
    explicit operator bool() const noexcept { return code != ReplayErrorCode::none; }
};

struct ReplayStatistics {
    std::uint64_t report_fast = 0, report_ordered = 0;
    std::uint64_t submissions = 0;
    // Earlier submissions not yet retired that the latest encode ordered
    // against.
    std::uint64_t unretired_history = 0;
    std::uint64_t render_passes = 0;
    // Render passes that began after only the previous pass's fragment work
    // (game barrier policy).
    std::uint64_t relaxed_pass_barriers = 0;
    std::uint64_t draws = 0;
    std::uint64_t dispatches = 0;
    std::uint64_t copies = 0;
    std::uint64_t clears = 0;
    std::uint64_t graphics_pipelines = 0;
    // Of those, pipelines the look-ahead started before their draw was reached.
    std::uint64_t predicted_pipelines = 0;
    // Of those, ones a draw has used.
    std::uint64_t predicted_used = 0;
    // Counter reports whose GPU time fell outside the window sampled around
    // their submission; they carry the completion time instead.
    std::uint64_t counter_timestamp_fallbacks = 0;
    std::uint64_t compute_pipelines = 0;
    // Time the replay worker spent encoding submissions.
    std::uint64_t encode_ns = 0;
    // Thread CPU time excludes blocking waits; submit includes record capture.
    std::uint64_t encode_cpu_ns = 0;
    std::uint64_t submit_cpu_ns = 0;
    // Thread CPU time of the encode thread, which replays the worker's Metal
    // calls; zero when the worker encodes itself.
    std::uint64_t encode_thread_cpu_ns = 0;
    // Draws and dispatches left out because replay cannot encode them; the
    // rest of their submission still runs.
    std::uint64_t skipped = 0;
    // Dispatches with an empty grid: they do no work and are not encoded.
    std::uint64_t empty_dispatches = 0;
    // GPU-predicated conditional blocks whose predicate ran as a kernel (also
    // counted in dispatches) or as a draw first in the block's render pass.
    std::uint64_t predicate_kernels = 0;
    std::uint64_t predicate_draws = 0;
    // Transfer-encoder operations encoded without a barrier because their
    // guest accesses do not conflict with the operations before them.
    std::uint64_t transfer_barriers_skipped = 0;
    std::uint64_t upload_arena_allocations = 0;
    std::uint64_t upload_arena_reuses = 0;
    std::uint64_t upload_bytes = 0;
    std::uint64_t synchronize_gpu_waits = 0;
    // Texture uses that first waited for guest-memory stores to be published.
    std::uint64_t guest_write_texture_waits = 0;
    // Guest-memory store prediction: logs read, pools learned so far, pools
    // found after a submission had been encoded from the learned set, and
    // submissions encoded that way.
    std::uint64_t guest_write_logs = 0;
    std::uint64_t guest_write_pools_learned = 0;
    std::uint64_t guest_write_misses = 0;
    std::uint64_t guest_write_predicted_submissions = 0;
    std::uint64_t uniform_versions = 0;
    std::uint64_t uniform_version_bytes = 0;
    std::uint64_t compute_passes = 0;
    std::uint64_t feedback_snapshots = 0;
    std::uint64_t feedback_snapshot_reuses = 0;
    std::uint64_t feedback_read_pass_splits = 0;
    std::uint64_t feedback_mip_view_narrows = 0;
    std::uint64_t new_buffer_allocations = 0;
    std::uint64_t completion_cpu_ns = 0;
    // Time from commit to the completion callback, summed over submissions.
    std::uint64_t gpu_ns = 0;
    // Stage argument tables built, and draws that reused the open pass's table.
    std::uint64_t stage_binds = 0;
    std::uint64_t stage_reuses = 0;
    // Of those, binds that rewrote only a stage's uniform buffers.
    std::uint64_t stage_uniform_binds = 0;
    // Texture and sampler pool tables published for native pool bindings.
    std::uint64_t pool_table_builds = 0;
    std::uint64_t pool_table_patches = 0;
    std::uint64_t pool_table_kept = 0;
    std::uint64_t pool_ids_read = 0;
    std::uint64_t pool_textures_made = 0;
    std::uint64_t pool_textures_left = 0;
    std::uint64_t pool_walk_us = 0;
    // Passes ended because a pool-reading fragment stage had read the pass's
    // depth attachment. Counted whether or not pass-reason logging is enabled.
    std::uint64_t pass_break_depth_sampled = 0;
    // Textures whose storage moved to R32 so an image atomic could run.
    std::uint64_t image_atomic_upgrades = 0;
    // Storage replacements that happened while a stage was already binding.
    // A slot written earlier in that bind can still name the old storage.
    std::uint64_t image_atomic_late_upgrades = 0;
    // Atomic image binds that used the stand-in texture.
    std::uint64_t image_atomic_sinks = 0;
    // Storage image binds whose handle named no texture and took a sink.
    std::uint64_t image_sinks = 0;
};

// Where the driver reads and writes its pipeline warm-up state. Every path is
// optional; an empty one turns that part off.
struct PipelineWarmupConfig {
    // Resolves the stage identities the manifests name.
    std::shared_ptr<MetalShaderPackage> package;
    // Read at launch; the rows are built in order in the background.
    std::string seed_manifest;
    // Read at launch like the seed, and appended with every pipeline this
    // session created that no manifest held.
    std::string user_manifest;
    // Receives every shader stage and pipeline in first-use order (a manifest
    // is made by pointing this at a new file).
    std::string record_manifest;
    // Holds the Metal binary archives of built pipelines. The replay keeps its
    // archives under a subdirectory named for the package, OS build and GPU,
    // and takes each pipeline from them without compiling when they hold it;
    // a pipeline they lack is compiled and logged as an archive miss.
    std::string archive_root;
    // A packed app's archive directory (read only), tried before archive_root
    // when the session does not capture: the archives built when the app was
    // packed, for the packing machine's OS build and GPU.
    std::string archive_bundle_root;
    // Also captures every pipeline this session builds into a new archive in
    // that subdirectory (the offline archive build; captured compiles bypass
    // Metal's own shader cache, so a playing session leaves this off).
    bool archive_capture = false;
    // Builds open-form manifest rows (O), which name program pairs no session
    // recorded: the offline archive build only. Other sessions, capturing ones
    // included, skip those rows.
    bool archive_open_rows = false;
    // Names the archive directory when there is no package to name it (tests).
    std::string archive_identity;
    unsigned warm_threads = 3;
    bool warm = true;
};

// The outcome of a pipeline warm-up run to completion, for the offline archive
// build: manifest rows and how each ended, and the archive file written.
struct PipelineWarmupReport {
    std::uint64_t rows = 0, built = 0, unresolved = 0, undecodable = 0, failed = 0;
    // Pipelines taken from an archive (exact or open form), of those the open
    // forms specialized, and compiles because no archive held the pipeline.
    std::uint64_t archive_hits = 0, archive_open_hits = 0, archive_misses = 0;
    bool finished = false;
    std::string archive;
    // A capturing session failed to write one of its archive files, so the
    // archive lacks pipelines it compiled.
    bool archive_incomplete = false;
    // Render pipelines ("<vertex> <fragment>") a capturing session compiled
    // but could not archive, and the pipelines the warm-up missed in the
    // loaded archives (the first 65536).
    std::vector<std::string> unarchived, warmup_missed;
};

class MetalReplayBackend : public nvn::ReplayBackend {
public:
    ~MetalReplayBackend() override = default;
    // Capture frame resources in CPU submission order, after the named walk,
    // before a later walk can publish another allocation. False cancels a
    // pending capture during shutdown; the callback never waits for the GPU.
    virtual bool enqueue_frame_capture(std::uint64_t after_fence,
                                       std::function<void(bool)> capture) = 0;
    [[nodiscard]] virtual std::shared_ptr<Metal4SubmissionOwner> submission_owner() const = 0;
    virtual void track_native_submission(const std::shared_ptr<Metal4NativeSubmission>& submission) = 0;

    // After guest producers have stopped: close submission admission, drain
    // admitted work and logging tails, and join replay/encode before mappings
    // or command registries can retire. A timeout retains owners for retry.
    virtual bool stop_submission_workers(
        std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt) = 0;

    // Starts the pipeline warm-up, manifest recording and archive described by
    // the configuration. Call once, before the first submission.
    virtual void configure_pipeline_warmup(PipelineWarmupConfig config) { (void)config; }
    // Waits until the warm-up (if any) has built every manifest row, writes the
    // captured archive, and reports.
    [[nodiscard]] virtual PipelineWarmupReport finish_pipeline_warmup() { return {}; }
    [[nodiscard]] virtual ReplayError last_error() const = 0;
    // Native consumers call after their ordered retirement dependency, before
    // making a frame visible. Public NVN fence semantics stay separate.
    [[nodiscard]] virtual std::function<std::string()> retain_native_producer_result(std::uint64_t fence) = 0;
    [[nodiscard]] virtual ReplayStatistics statistics() const noexcept = 0;
    // Test seam. Caps how many guest-memory pools stores may be predicted
    // from. Zero restores 4096. Past the cap, stores count as a write anywhere.
    virtual void set_learned_guest_write_cap(std::size_t) {}
    // Why draws and dispatches were left out, with a count per reason.
    [[nodiscard]] virtual std::vector<std::pair<std::string, std::uint64_t>> skip_reasons() const = 0;
    // Test seams for pool tables. The allocation failure skips creating a table
    // and fails the draw. The buffer is the texture table the latest lookup bound.
    virtual void set_pool_table_allocation_failure(bool) {}
    [[nodiscard]] virtual id<MTLBuffer> current_texture_pool_buffer() const { return nil; }
};

enum class ReplayThread { worker, encode, guest_present };

// Optional policy adapter, invoked on the thread being scheduled. It can
// observe or reject declarations; false must preserve the current host policy.
// The context must outlive the backend. An empty adapter uses the native policy.
struct ReplaySchedulingPolicy {
    void* context = nullptr;
    bool (*apply)(void*, ReplayThread, std::uint64_t period_ns,
                  std::uint64_t computation_ns) = nullptr;
    // Optional observation on the encode thread, after selecting a submission's
    // waiting budget. It does not replace the policy or alter command ordering.
    void (*observe_encode_spin)(void*, std::uint64_t fence, std::uint64_t period_ns,
                                std::uint64_t spin_ns) = nullptr;
    // Optional lifecycle observations, using the same context lifetime above.
    // They must not call back into replay; joined may be observed on retries.
    void (*observe_shutdown_workers_joined)(void*) = nullptr;
    void (*observe_retired_log)(void*, std::uint64_t fence) = nullptr;
    // On the committing thread, after all actual residency declarations.
    void (*observe_residency_sets)(void*, std::uint64_t fence, std::uint64_t segment, std::size_t count) = nullptr;
    // Test seam: called while a submission's unmet event waits were read and
    // its parts' holds are not registered yet (queue and hold locks held).
    void (*observe_split)(void*) = nullptr;
    // Test observations under replay locks; they must not call back into replay.
    // A wait is reported only at its first unmet predecessor snapshot.
    void (*observe_event_wait)(void*, const volatile std::uint32_t* event,
                               std::uint32_t target, std::uint64_t predecessor) = nullptr;
    // False is after the signal store, true is after hold evaluation. Both
    // observations precede retiring the signal's pending writer.
    void (*observe_event_signal)(void*, const volatile std::uint32_t* event,
                                 std::uint64_t sequence, bool evaluated) = nullptr;
};

// The queue must belong to device and both objects must outlive the backend.
// Resources is borrowed and must outlive every submitted fence.
[[nodiscard]] std::unique_ptr<MetalReplayBackend>
create_replay_backend(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                      ReplayResources& resources, ReplaySchedulingPolicy scheduling = {});
#if defined(ARMIGO_NVN_REPLAY_HAS_RESOURCES)
[[nodiscard]] std::unique_ptr<MetalReplayBackend>
create_replay_backend(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                      Resources& resources, ReplaySchedulingPolicy scheduling = {});
#endif

} // namespace armigo::platform::metal::nvn_driver

#if defined(ARMIGO_NVN_REPLAY_HAS_RESOURCES)
#undef ARMIGO_NVN_REPLAY_HAS_RESOURCES
#endif
