// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/memory/guest_memory_model.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace armigo::runtime {

class ProcessRuntime;

// Host orchestration outcomes, never invented Horizon Results. Implementers
// preserve the first causal failure and its stage, including during teardown.
// Firmware 20.5.0 is the sole baseline; partial coverage is not success.
// An unsupported SVC/IPC,
// preparation gap or shader miss stops the run; it cannot count as success.
// Diagnostic addresses/identities are external report data, never fixtures.
enum class LaunchStage {
    inputs, profile, preparation, modules, memory, threads, services,
    graphics, presentation, shutdown,
};
enum class LaunchError {
    invalid_input, permission_denied, unsupported, artifact_mismatch,
    resource_limit, host_failure, guest_fault, cancelled, deadline_expired,
};
struct LaunchFailure {
    LaunchStage stage = LaunchStage::inputs;
    LaunchError code = LaunchError::invalid_input;
    std::string operation;
    std::string detail;
    std::optional<std::uint64_t> thread_id;
    std::optional<memory::GuestAddress> guest_pc;
    std::optional<std::uint32_t> command;
    // Present only when an existing subsystem supplied a real guest Result.
    std::optional<std::uint32_t> guest_result;
    // A failed construction may still own undrained consumers. This owner
    // accepts shutdown retries; its resources stay retained until one succeeds.
    std::shared_ptr<ProcessRuntime> cleanup_owner;
};

// Default construction is failure, so an unfilled result never means success.
template<class T> using LaunchResult = std::variant<LaunchFailure, T>;
using LaunchStatus = LaunchResult<std::monostate>;

struct RunStatistics {
    std::uint64_t queued_frames = 0;
    std::uint64_t completed_frames = 0;
    std::uint64_t completed_draws = 0;
    std::chrono::nanoseconds elapsed{};
};
struct RunSuccess {
    // Reached an explicit nonzero completed-presentation target, then drained.
    std::uint64_t frame_target = 0;
};
struct GuestExit {
    // ExitProcess does not imply a POSIX exit code. Populate only if the
    // admitted entry ABI actually supplies a guest status value.
    std::optional<std::int64_t> status;
};
struct RunTimeout { std::chrono::nanoseconds limit{}; };
enum class StopReason { user_request, window_closed, watchdog_timeout };
struct RunCancelled { StopReason reason = StopReason::user_request; };
using RunOutcome = std::variant<LaunchFailure, RunSuccess, GuestExit, RunTimeout, RunCancelled>;
struct RunResult {
    RunOutcome outcome;
    RunStatistics statistics;
    // If bounded teardown fails, the runtime must remain owned and alive.
    // A shell must use an external process supervisor for a hard deadline;
    // returning timeout never licenses freeing memory under native threads.
    bool quiescent = false;
    std::optional<LaunchFailure> teardown_failure;
};

} // namespace armigo::runtime
