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

#include <armigo/graphics/renderer_types.h>
#include <optional>
#include <vector>

namespace armigo::graphics {

enum class SubmissionPhase : std::uint8_t {
    Accepted, Submitted, GpuComplete, Publishing, Complete, Failed
};
struct CompletionSnapshot {
    SubmissionId id;
    SubmissionPhase phase = SubmissionPhase::Accepted;
    std::optional<RendererError> error;
    [[nodiscard]] bool guest_visible_complete() const noexcept {
        return phase == SubmissionPhase::Complete;
    }
};
namespace detail { struct SubmissionState; }

// Polling does not wait, publish effects, or retire storage. Dropping a token
// does not cancel work. A timeout is a caller observation, not a state change.
class CompletionToken {
public:
    CompletionToken() = default;
    [[nodiscard]] bool valid() const noexcept { return bool(state_); }
    [[nodiscard]] CompletionSnapshot snapshot() const;
    // The submission identity, fixed at creation: reading it takes no lock.
    [[nodiscard]] SubmissionId id() const noexcept;
private:
    friend class SubmissionCompletion;
    explicit CompletionToken(std::shared_ptr<detail::SubmissionState> state)
        : state_(std::move(state)) {}
    std::shared_ptr<detail::SubmissionState> state_;
};

// Coordinator-owned actions, never borrowed ioctl callbacks. Called outside the
// completion lock, once, in this order. prepare_publication() reserves and
// validates the ENTIRE ordered effect batch; failure publishes nothing.
// publish() is its infallible atomic commit, not arbitrary fallible callbacks.
// Coherence includes required boundary writeback in untracked mode. Tracked
// mode may preserve GPU authority, with CPU access intercepted by L3.
class CompletionActions {
public:
    virtual ~CompletionActions() = default;
    virtual RendererStatus make_coherent() = 0;
    virtual RendererStatus prepare_publication() = 0;
    virtual void publish() noexcept = 0;
    // Failure retirement first transfers every authoritative dirty allocation
    // and failed effect owner to registry/recovery ownership. Failure retains
    // all submission leases. This hook must never publish successful effects.
    virtual RendererStatus preserve_failed_authority() = 0;
};

// The backend MUST capture this value in its completion handler before commit.
// Its shared state retains every resource until actual GPU quiescence AND the
// coordinator's finish() (or explicit failed retirement). Renderer shutdown
// drains pending handlers; destroying
// the caller's token or renderer is never permission to free in-flight storage.
// feedback() only records owned status. It must not touch guest memory, publish
// fences or reenter the driver. finish() belongs to the serialized coordinator.
// Duplicate/out-of-order transitions fail without changing the first outcome.
// Failed authoritative GPU resources additionally remain owned by the registry;
// retirement here cannot discard the only dirty copy.
class SubmissionCompletion {
public:
    SubmissionCompletion(SubmissionId id, std::vector<RetainedResource> resources,
                         std::shared_ptr<CompletionActions> actions);
    [[nodiscard]] CompletionToken token() const { return CompletionToken(state_); }
    // Reject before commit when admission/encoding fails and no GPU can access
    // this packet. Submitted work can fail only through quiescent feedback.
    [[nodiscard]] RendererStatus reject(CommandResult error);
    [[nodiscard]] RendererStatus submitted();
    [[nodiscard]] RendererStatus feedback(CommandResult gpu_result);
    [[nodiscard]] RendererStatus finish();
    // Only after failed finish and proven GPU quiescence. Calls
    // preserve_failed_authority() before dropping any leases. Error remains sticky; no publication/retry.
    [[nodiscard]] RendererStatus release_failed();
private:
    std::shared_ptr<detail::SubmissionState> state_;
};

// Acceptance owns an immutable snapshot and all referenced access owners.
// It is distinct from commit (Submitted), GPU feedback and coherent completion.
struct SubmissionReceipt { CompletionToken completion; };
enum class SubmissionBoundary : std::uint8_t {
    SubmissionEnd, Dependency, CpuAccess, Capture, ResourceBudget, Shutdown
};

} // namespace armigo::graphics
