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

#include "../../nvn/backend.h"

#include <chrono>
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
    id<MTLEvent> event = nil;
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
    [[nodiscard]] virtual bool texture_stamp_current(std::int32_t id, const DescriptorStamp& stamp) const {
        (void)id; (void)stamp;
        return false;
    }
    [[nodiscard]] virtual id<MTLSamplerState> sampler_for_id(std::int32_t id) = 0;
    // The sampler as registered when a binding was recorded after
    // `registrations` sampler registrations.
    [[nodiscard]] virtual id<MTLSamplerState> sampler_for_id(std::int32_t id, std::uint64_t registrations) {
        (void)registrations;
        return sampler_for_id(id);
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
    // A view of a descriptor's texture in a shader's declared type; nil keeps
    // the texture as it is.
    [[nodiscard]] virtual id<MTLTexture> fitted_texture(std::int32_t texture_id, id<MTLTexture> texture,
                                                        MTLTextureType type) {
        (void)texture_id; (void)texture; (void)type;
        return nil;
    }
    // The replay worker starts walking a submission. Buffers it resolved while
    // walking an earlier one may be released from then on.
    virtual void begin_walk() {}
    // Commits the texture uploads made while resolving bindings and waits for
    // them, before the replay commits work that may read those textures.
    virtual bool flush_uploads() { return true; }
    // As flush_uploads, but the uploads are not waited for on the CPU: the
    // fence is what the replay queue waits for on the GPU before committing
    // the work that may read them. The default completes them first.
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
};

struct ReplayError {
    ReplayErrorCode code = ReplayErrorCode::none;
    std::string detail;
    explicit operator bool() const noexcept { return code != ReplayErrorCode::none; }
};

struct ReplayStatistics {
    std::uint64_t report_fast = 0, report_ordered = 0;
    std::uint64_t submissions = 0;
    std::uint64_t render_passes = 0;
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
    std::uint64_t uniform_versions = 0;
    std::uint64_t uniform_version_bytes = 0;
    std::uint64_t compute_passes = 0;
    std::uint64_t new_buffer_allocations = 0;
    std::uint64_t completion_cpu_ns = 0;
    // Time from commit to the completion callback, summed over submissions.
    std::uint64_t gpu_ns = 0;
    // Stage argument tables built, and draws that reused the open pass's table.
    std::uint64_t stage_binds = 0;
    std::uint64_t stage_reuses = 0;
    // Of those, binds that rewrote only a stage's uniform buffers.
    std::uint64_t stage_uniform_binds = 0;
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
};

class MetalReplayBackend : public nvn::ReplayBackend {
public:
    ~MetalReplayBackend() override = default;

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
    [[nodiscard]] virtual ReplayStatistics statistics() const noexcept = 0;
    // Why draws and dispatches were left out, with a count per reason.
    [[nodiscard]] virtual std::vector<std::pair<std::string, std::uint64_t>> skip_reasons() const = 0;
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
