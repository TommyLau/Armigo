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

#include "bindings.h"
#include "pipeline_state.h"
#include <armigo/graphics/renderer.h>

namespace armigo::platform::metal {

struct ResidentAttachments {
    graphics::RenderPassKey key;
    std::array<std::optional<graphics::ResidentTexture>, 8> colors;
    std::optional<graphics::ResidentTexture> depth, stencil;
};
struct PreparedDraw {
    graphics::DrawPlan draw;
    graphics::RenderState state;
    ResidentPipeline pipeline;
    graphics::RetainedResource depth_stencil;
    Bindings bindings;
    bool shader_depth_bias = false;
    // Feedback textures and argument-buffer slices share the per-packet staging budget.
    std::uint64_t feedback_bytes_limit = graphics::max_texture_backing_size;
};
struct PreparedClear { graphics::ClearPlan clear; };
// A compute launch encoded between render passes of the same batch, as a
// guest dispatch() does: grid and block come from the queue meta data.
struct PreparedDispatch {
    std::array<std::uint32_t, 3> grid{}, block{};
    ResidentComputePipeline pipeline;
    Bindings bindings;
    std::uint64_t argument_bytes_limit = graphics::max_texture_backing_size;
};
struct PreparedRender {
    graphics::RenderRequest request;
    // Only completions without guest effects or registry-wide coherence may
    // retire while a newer batch is still open.
    bool prefix_safe_completion = false;
    graphics::RenderTargetPlan targets;
    ResidentAttachments attachments;
    // Union of binding and attachment accesses, normalized to backing ranges.
    // append creates its token and atomically accepts these accesses before any
    // encoding; rejection publishes no partial version edits.
    std::vector<graphics::GpuAccessRequest> accesses;
    // Exact prepared storage for accesses, in the same order. Writable entries
    // implement ResourceMemoryStorage and cover the complete normalized range;
    // read entries still retain the allocation consumed by this packet.
    std::vector<graphics::RetainedResource> access_storage;
    std::variant<PreparedDraw, PreparedClear, PreparedDispatch> operation;
};

// L10: one ordered queue first. Compatible RenderPassKey values may share an
// encoder; transfers, attachment changes, feedback hazards and ordering/fetch
// boundaries break it. Keep guest-live contents with Load/Store; a validated
// full clear can fold into initial load, partial/masked clears are operations.
// append validates the complete immutable packet before any encoding for that
// packet. Native preparation and encoding run in order on a serial worker;
// flush hands off the batch without joining; the worker closes and commits it.
// Pending commits count toward the in-flight limit. Retirement waits for commit
// completion. After coordinator actions and registry-lease release, the worker
// destroys retired packet graphs through a bounded cleanup ring. A full ring
// retains the in-flight slot until a later poll; poll does not join cleanup.
// Successful drain, joined statistics and destruction include pending cleanup.
// Entry points must be externally serialized. Encoding failures poison the
// queue and surface at the next append, flush or poll. Drain still retires work.
// It retains all resources, snapshots, stages, access owners, actions and
// allocator storage until SubmissionCompletion retires them. Encode real
// Metal barriers/fences for dependencies; queue order is not a blanket hazard
// proof. Completion handlers capture SubmissionCompletion, record feedback and
// wake the coordinator only. Device/encoding/GPU errors withhold guest effects.
// A draw repeating the last appended draw's state: its segments and the
// buffer bindings whose contents changed since.
struct RepeatDraw {
    std::vector<graphics::DrawSegment> segments;
    std::vector<BufferBinding> changed;
    // Vertex stream and index buffer rebinds (same layouts as the last draw).
    std::vector<BufferBinding> vertex_buffers;
    std::optional<graphics::ResidentBuffer> index_buffer;
    // The source draw's state with the viewports and scissors in effect for
    // this repeat, when they differ from the source's.
    std::shared_ptr<const graphics::RenderState> state;
    // The pipeline this repeat draws with when its vertex attribute layout
    // differs from the source draw's (same stages, streams and outputs).
    std::shared_ptr<const ResidentPipeline> pipeline;
};

class SubmissionQueue {
public:
    virtual ~SubmissionQueue() = default;
    virtual graphics::ResidencyDomain domain() const noexcept = 0;
    virtual graphics::RendererResult<graphics::SubmissionReceipt> append(
        PreparedRender prepared) = 0;
    // Encodes a repeat of the last appended draw into its open batch; false
    // when there is none (the caller then prepares a full packet).
    virtual bool append_repeat(RepeatDraw repeat) {
        static_cast<void>(repeat);
        return false;
    }
    virtual graphics::RendererStatus flush(graphics::SubmissionBoundary boundary) = 0;
    virtual graphics::RendererStatus poll() = 0;
    virtual graphics::RendererStatus drain(std::uint64_t steady_deadline_ns) = 0;
    // A held queue may prepare and encode bounded later work, but its native
    // commits wait until release. Entry points remain externally serialized.
    virtual graphics::RendererStatus set_commit_hold(bool held) {
        if (!held) return graphics::renderer_success();
        return graphics::RendererError({graphics::CommandStatus::Unsupported,
                                        "submission queue does not support commit hold"});
    }
    virtual graphics::RendererStatus abort_commit_hold(graphics::CommandResult error) {
        static_cast<void>(error);
        return graphics::RendererError({graphics::CommandStatus::Unsupported,
                                        "submission queue does not support commit hold abort"});
    }
    virtual bool commit_held() const noexcept { return false; }
    virtual bool can_accept_while_held() const noexcept { return true; }
    // Flushes, then retires committed work in order until fewer than
    // `limit` submissions remain in flight.
    virtual graphics::RendererStatus retire_below(std::size_t limit, std::uint64_t steady_deadline_ns) {
        static_cast<void>(limit);
        return drain(steady_deadline_ns);
    }
};

// A transient drain/retirement result while a native commit is deliberately
// held. The caller may release the hold and retry without poisoning the queue.
[[nodiscard]] graphics::RendererStatus commit_hold_wait();
[[nodiscard]] bool commit_hold_blocked(const graphics::RendererStatus& status) noexcept;

struct SubmissionStatistics {
    std::uint64_t accepted = 0;
    std::uint64_t command_buffers = 0;
    std::uint64_t render_passes = 0;
    std::uint64_t draws = 0;
    std::uint64_t argument_tables = 0;
    std::uint64_t argument_pages = 0, argument_page_reuses = 0;
    std::uint64_t argument_slices = 0, argument_slice_reuses = 0;
    std::uint64_t command_bundles = 0, command_bundle_reuses = 0;
};

// Native tests may delay encoding or replace its result, and delay or replace
// feedback after GPU quiescence. Production factories omit this control. Hooks
// run off the caller thread and must not call guest code, publish effects or
// reenter the queue.
class SubmissionTestControl {
public:
    virtual ~SubmissionTestControl() = default;
    virtual graphics::RendererStatus before_encode(graphics::SubmissionId) {
        return graphics::renderer_success();
    }
    virtual graphics::RendererStatus before_commit(graphics::SubmissionId) {
        return graphics::renderer_success();
    }
    virtual graphics::CommandResult after_gpu_quiescence(
        graphics::CommandResult actual) = 0;
};

class DeviceContext;
[[nodiscard]] graphics::RendererResult<std::shared_ptr<SubmissionQueue>>
create_submission_queue(std::shared_ptr<DeviceContext> device,
                        std::shared_ptr<graphics::ResourceMemory> memory,
                        std::uint32_t max_commands_per_submission,
                        std::uint32_t max_inflight_submissions);
[[nodiscard]] graphics::RendererResult<std::shared_ptr<SubmissionQueue>>
create_submission_queue_for_testing(
    std::shared_ptr<DeviceContext> device,
    std::shared_ptr<graphics::ResourceMemory> memory,
    std::uint32_t max_commands_per_submission,
    std::uint32_t max_inflight_submissions,
    std::shared_ptr<SubmissionTestControl> control);
[[nodiscard]] SubmissionStatistics submission_statistics(
    const SubmissionQueue& queue, bool wait_for_encoding = true) noexcept;

} // namespace armigo::platform::metal
