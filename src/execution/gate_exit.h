// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>

#include <armigo/execution/guest_preempt.h>

namespace armigo::execution::detail {

// Returned to assembly only after the complete C++ dispatcher has returned.
struct GateExitTarget {
    std::uintptr_t continuation;
    void* record;
};

// One synchronous guest invocation. Defined in the gate library so ordinary
// gates have no link dependency on the optional guest-entry bridge.
class GuestExitScope {
public:
    // Optional host-owned field populated by the entry bridge before guest
    // execution. It must remain live and hold its aligned saved host SP until
    // this scope ends. A null pointer retains ordinary caller-stack dispatch.
    // The guest stack range [stack_low, stack_high) is published to a bound
    // GuestPreemptSlot for the duration of the invocation.
    GuestExitScope(std::uintptr_t continuation, void* record,
                   const std::uint64_t* saved_host_sp = nullptr,
                   std::uint64_t stack_low = 0, std::uint64_t stack_high = 0) noexcept;
    ~GuestExitScope();
    GuestExitScope(const GuestExitScope&) = delete;
    GuestExitScope& operator=(const GuestExitScope&) = delete;
    static const GateExitTarget* target(std::uint32_t depth) noexcept;
    // Zero keeps the current stack. Direct reentry during this invocation's
    // callback must not rewind its host stack; a nested invocation has its own
    // scope and saved SP. One is an invalid configured-SP sentinel for assembly.
    static std::uintptr_t callback_stack() noexcept;

private:
    GateExitTarget target_;
    GuestExitScope* outer_;
    const void* owner_;
    std::uint32_t entry_depth_;
    const std::uint64_t* saved_host_sp_;
    PreemptPublication outer_publication_;
};

} // namespace armigo::execution::detail
