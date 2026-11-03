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

#include <armigo/execution/guest_dispatch.h>
#include <armigo/horizon/startup_queries.h>
#include <armigo/horizon/ipc.h>
#include <armigo/runtime/guest_memory.h>
#include <string>

namespace armigo::runtime {

class GuestClock;

// Connects the supported Horizon startup queries to prepared SVC sites.
// The queries object must outlive every scope using the returned supervisor.
// Unsupported operations fail the gate entry; they never become success stubs.
execution::GuestSupervisor query_supervisor(horizon::StartupQueries& queries) noexcept;

// One invoking thread's initial query and synchronous IPC boundary. The caller
// supplies validated, pinned IPC backing and trusted process identity. All
// referenced objects and backing must outlive scopes using binding(). Pending
// requests remain explicit failures to resume until a scheduler is connected.
// What the guest asked svcSleepThread for: a positive duration in nanoseconds, a
// reschedule (the yield types 0, -1 and -2), or a value the kernel ignores (any other
// negative value).
enum class GuestSleep { duration, yield, ignore };

GuestSleep classify_guest_sleep(std::int64_t nanoseconds) noexcept;

class StartupSupervisor {
public:
    StartupSupervisor(horizon::StartupQueries& queries, horizon::IpcDispatcher& ipc,
                      horizon::IpcCallContext caller, GuestClock* clock = nullptr,
                      std::shared_ptr<GuestMemory> memory = {})
        : queries_(&queries), ipc_(&ipc), caller_(caller), clock_(clock),
          memory_(std::move(memory)) {}
    StartupSupervisor(horizon::StartupQueries& queries, horizon::IpcDispatcher& ipc,
                      horizon::IpcCallContext caller, std::shared_ptr<GuestMemory> memory,
                      GuestClock* clock = nullptr)
        : StartupSupervisor(queries, ipc, caller, clock, std::move(memory)) {}
    const std::optional<LaunchFailure>& memory_failure() const noexcept { return memory_failure_; }
    execution::GuestSupervisor binding() noexcept;

private:
    static execution::GateStatus dispatch(std::uint16_t, execution::GateCall&,
                                          execution::GuestThreadContext&, void*);
    horizon::StartupQueries* queries_;
    horizon::IpcDispatcher* ipc_;
    horizon::IpcCallContext caller_;
    // Borrowed process clock; required for cancellation-aware production sleeps.
    GuestClock* clock_;
    std::shared_ptr<GuestMemory> memory_;
    std::optional<LaunchFailure> memory_failure_;
    // The detail of the last unsupported IPC request; failure messages point into it.
    std::string ipc_failure_detail_;
    // Holds the description of one guest break; the gate keeps the pointer in
    // its failure message, so it has to outlive the call.
    char break_message_[192]{};
};

} // namespace armigo::runtime
