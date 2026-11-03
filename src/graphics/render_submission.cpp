// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/render_submission.h>

#include <mutex>

namespace armigo::graphics {
namespace detail {
struct SubmissionState {
    // Immutable after construction, before the state is shared.
    SubmissionId id{};
    mutable std::mutex mutex;
    CompletionSnapshot snapshot;
    std::vector<RetainedResource> resources;
    std::shared_ptr<CompletionActions> actions;
    bool feedback_received = false;
    bool retired = false;
    bool releasing_failure = false;
};
} // namespace detail
namespace {
RendererError transition_error(const char* detail) {
    return RendererError({CommandStatus::Stopped, detail});
}
}

SubmissionId CompletionToken::id() const noexcept {
    return state_ ? state_->id : SubmissionId{};
}

CompletionSnapshot CompletionToken::snapshot() const {
    if (!state_) {
        return {{}, SubmissionPhase::Failed,
                transition_error("invalid renderer completion token")};
    }
    std::lock_guard lock(state_->mutex);
    return state_->snapshot;
}

SubmissionCompletion::SubmissionCompletion(
    SubmissionId id, std::vector<RetainedResource> resources,
    std::shared_ptr<CompletionActions> actions)
    : state_(std::make_shared<detail::SubmissionState>()) {
    state_->id = id;
    state_->snapshot.id = id;
    state_->resources = std::move(resources);
    state_->actions = std::move(actions);
    if (id.queue == 0 || id.sequence == 0 || !state_->actions) {
        state_->snapshot.phase = SubmissionPhase::Failed;
        state_->snapshot.error = transition_error("invalid submission identity or actions");
        state_->feedback_received = true;
    }
}

RendererStatus SubmissionCompletion::reject(CommandResult error) {
    std::lock_guard lock(state_->mutex);
    if (state_->snapshot.phase != SubmissionPhase::Accepted || error.ok()) {
        return transition_error("only unsubmitted work can be rejected with an error");
    }
    state_->snapshot.phase = SubmissionPhase::Failed;
    state_->snapshot.error.emplace(error);
    state_->feedback_received = true;
    return renderer_success();
}

RendererStatus SubmissionCompletion::submitted() {
    std::lock_guard lock(state_->mutex);
    if (state_->snapshot.phase != SubmissionPhase::Accepted) {
        return transition_error("submission is not awaiting commit");
    }
    state_->snapshot.phase = SubmissionPhase::Submitted;
    return renderer_success();
}

RendererStatus SubmissionCompletion::feedback(CommandResult gpu_result) {
    std::lock_guard lock(state_->mutex);
    if (state_->snapshot.phase != SubmissionPhase::Submitted) {
        return transition_error("GPU feedback is duplicate or precedes commit");
    }
    if (gpu_result.ok()) {
        state_->snapshot.phase = SubmissionPhase::GpuComplete;
    } else {
        state_->snapshot.error.emplace(gpu_result);
        state_->snapshot.phase = SubmissionPhase::Failed;
    }
    state_->feedback_received = true;
    return renderer_success();
}

RendererStatus SubmissionCompletion::finish() {
    std::shared_ptr<CompletionActions> actions;
    std::vector<RetainedResource> retired_resources;
    std::optional<RendererError> prior_error;
    {
        std::lock_guard lock(state_->mutex);
        if (state_->retired) {
            if (state_->snapshot.error) return *state_->snapshot.error;
            return renderer_success();
        }
        if (!state_->feedback_received ||
            state_->snapshot.phase == SubmissionPhase::Publishing) {
            return transition_error("submission has not reached coordinator retirement");
        }
        prior_error = state_->snapshot.error;
        actions = state_->actions;
        state_->snapshot.phase = SubmissionPhase::Publishing;
    }

    RendererStatus outcome = prior_error ? RendererStatus(*prior_error) : renderer_success();
    try {
        if (!prior_error) {
            outcome = actions->make_coherent();
            if (std::holds_alternative<std::monostate>(outcome)) {
                outcome = actions->prepare_publication();
                if (std::holds_alternative<std::monostate>(outcome)) actions->publish();
            }
        }
    } catch (...) {
        outcome = RendererError({CommandStatus::Stopped,
                                 "renderer completion action threw an exception"});
    }
    {
        std::lock_guard lock(state_->mutex);
        if (const auto* error = std::get_if<RendererError>(&outcome)) {
            state_->snapshot.error = *error;
            state_->snapshot.phase = SubmissionPhase::Failed;
        } else {
            state_->snapshot.phase = SubmissionPhase::Complete;
        }
        state_->retired = true;
        if (!state_->snapshot.error) {
            retired_resources.swap(state_->resources);
            state_->actions.reset();
        }
    }
    // Destruct retained resources and actions outside the completion mutex.
    return outcome;
}

RendererStatus SubmissionCompletion::release_failed() {
    std::vector<RetainedResource> resources;
    std::shared_ptr<CompletionActions> actions;
    {
        std::lock_guard lock(state_->mutex);
        if (!state_->retired || !state_->feedback_received ||
            state_->snapshot.phase != SubmissionPhase::Failed || state_->releasing_failure) {
            return transition_error("failed submission is not ready for resource handoff");
        }
        if (!state_->actions) {
            if (!state_->resources.empty()) {
                return transition_error("failed resources have no recovery action owner");
            }
            return renderer_success();
        }
        state_->releasing_failure = true;
        actions = state_->actions;
    }
    RendererStatus outcome;
    try {
        outcome = actions->preserve_failed_authority();
    } catch (...) {
        outcome = transition_error("failed resource handoff threw an exception");
    }
    {
        std::lock_guard lock(state_->mutex);
        state_->releasing_failure = false;
        if (std::holds_alternative<std::monostate>(outcome)) {
            resources.swap(state_->resources);
            state_->actions.reset();
        }
    }
    return outcome;
}

} // namespace armigo::graphics
