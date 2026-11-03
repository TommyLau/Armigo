// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/render_state.h>
#include <armigo/graphics/draw_plan.h>

#include <memory>
#include <new>
#include <utility>

namespace armigo::graphics {

struct RenderRequest {
    SubmissionId id;
    // Kept alive through admission; enqueue owns an isolated copy on acceptance.
    std::shared_ptr<const RenderStop> snapshot;
    std::shared_ptr<ResourceMemory> memory;
    std::vector<CompletionToken> dependencies;
    std::shared_ptr<CompletionActions> completion;
    // Completion actions inspect only this packet and never require registry quiescence.
    bool prefix_safe_completion = false;
};

// L10 encodes; L11 owns channel ordering and delayed guest effects. enqueue
// completes all admission, live shader/descriptor checks and retention before
// returning a receipt. Failure schedules no work/effects for THIS request.
// Success never changes the existing synchronous RenderCallbacks contract.
// Monotonic queue/sequence IDs must not be reused; dependencies form an acyclic
// ordered graph. Unknown/failed dependencies stop; pending ones become real
// waits/barriers. Never run later effects past an earlier failed request.
// Persistent lazy resources require verified native tracking; otherwise flush
// resolves GPU writes and invalidates CPU-backed cache contents conservatively.
class Renderer {
public:
    virtual ~Renderer() = default;
    virtual RendererResult<SubmissionReceipt> enqueue(RenderRequest request) = 0;
    // Transfers exclusive ownership of a frozen snapshot. Callers must not
    // retain mutable aliases. This replaces request.snapshot; ordinary enqueue
    // continues to isolate the accepted snapshot from mutable caller aliases.
    // Backends without an ownership fast path retain their usual admission.
    virtual RendererResult<SubmissionReceipt> enqueue_owned(
        RenderRequest request, std::unique_ptr<const RenderStop> snapshot) {
        try {
            request.snapshot = std::move(snapshot);
            return enqueue(std::move(request));
        } catch (const std::bad_alloc&) {
            return RendererError({CommandStatus::BudgetExceeded,
                                  "Renderer snapshot ownership allocation failed"});
        }
    }

    // Commit accepted work through the named boundary; success means commit
    // only. Callers observe tokens for actual completion. Batching is allowed
    // across compatible passes, never across required fetch/effect boundaries.
    virtual RendererStatus flush(SubmissionBoundary boundary) = 0;
    // Run ready coherence/effect actions in channel order on the coordinator.
    // No borrowed callbacks or nvdrv reentry from Metal completion handlers.
    virtual RendererStatus poll() = 0;
    // Deadline is a steady-clock nanosecond time point. Pending timeout returns
    // Stopped and retains all work; only success proves quiescent teardown.
    // A GPU failure can be quiescent but still returns its causal error; the
    // caller must retain the renderer until a subsequent successful drain.
    virtual RendererStatus drain(std::uint64_t steady_deadline_ns) = 0;
};

} // namespace armigo::graphics
