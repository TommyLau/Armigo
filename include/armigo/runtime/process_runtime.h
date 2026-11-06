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

#include <armigo/runtime/launch_description.h>
#include <armigo/runtime/launch_plan.h>
#include <armigo/runtime/presentation.h>
#include <armigo/runtime/thread_runtime.h>
#include <memory>
#include <functional>
#include <stop_token>

namespace armigo::runtime {

class ServiceRegistry;
struct ServiceRegistryDependencies;

// Lane F owns the process facade; lane I uses it from a worker while AppKit
// pumps on the main thread. Reuse ModuleSet, ProcessRecord, HandleTable, IPC,
// shared arbitration and service objects. Firmware 20.5.0 is the sole baseline.
// The launch flows from collected inputs through preparation to real
// presentation, and an evidence gap stops the run. No SDK host overrides,
// guest fork path, fake shader/presentation success
// or hidden private input dependencies. Unsupported dependencies fail explicitly.
struct RuntimeDependencies {
    ProcessMemoryOptions memory_options;
    // Identity supplied by the installed build, not untrusted artifact input.
    PreparationOptions preparation_options;
    std::shared_ptr<GuestClock> clock;
    std::shared_ptr<PresentationClock> presentation_clock;
    std::shared_ptr<platform::Presenter> presenter;
    // Synthetic-test construction seam, rejected for production profiles.
    // Empty uses the production registry; consumed during initialize.
    std::function<LaunchResult<std::shared_ptr<ServiceRegistry>>(ServiceRegistryDependencies)> create_services;
};
class ProcessRuntime {
public:
    virtual ~ProcessRuntime() = default;
    // Construction validates all inputs and owns resources but executes no guest
    // code. Failure publishes no running thread or executable entry capability.
    // If bounded rollback fails, LaunchFailure::cleanup_owner retains the
    // incomplete runtime for shutdown retry without replacing the original error.
    // Dependencies must be nonnull, share one clock, and match the selected mode.
    static LaunchResult<std::unique_ptr<ProcessRuntime>> create(
        const LaunchDescription& description, RuntimeDependencies dependencies);
    // Single-use synchronous run, on a dedicated host worker. All callbacks,
    // mappings and module/site owners survive through teardown. Timeout/cancel
    // are distinct outcomes; a completed frame target requires real VI releases.
    virtual RunResult run(const RunLimits& limits, const ReportDestination& output,
                           std::stop_token stop = {}) = 0;
    // Thread-safe and idempotent. Does not pretend to join running native code.
    virtual void request_stop(StopReason reason = StopReason::user_request) noexcept = 0;
    // Stops submissions, wakes waits, drains GPU consumers, joins guest workers,
    // then releases services and memory. May be retried after deadline failure.
    virtual LaunchStatus shutdown(std::optional<ClockInstant> deadline) = 0;
};
// The capture policy shared by the frame writer and presentation readback,
// so the two cannot disagree: output captures a completed frame when it names
// a capture directory and the frame number is a multiple of
// ARMIGO_CAPTURE_EVERY (default 1, read once per process). Presentation reads
// back only these frames; the GPU still consumes every frame.
[[nodiscard]] bool captures_frame(const ReportDestination& output, std::uint64_t frame_number);
// Never destroy a nonquiescent runtime. Shells retain it after a timed-out run;
// a hard watchdog belongs to an external spawn/exec supervisor, never guest fork.

} // namespace armigo::runtime
