// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Guest faults. Guest code runs natively, so a guest data abort, undefined
// instruction or breakpoint arrives as a host signal, whose default action
// ends the whole process without a report. The handler installed here ends
// the faulting guest invocation instead, the way an asynchronous stop does
// (guest_preempt.h), and records the fault in the thread's slot; the runtime
// then reports it and stops the process.
//
// A fault qualifies under the rules of a stop: its pc is in registered guest
// code, and the thread is bound to a slot whose innermost invocation is
// published with a guest stack that holds the stack pointer. Every other
// fault, including guest code under a host callback (GuestPreemptHold), goes
// to the handler installed before this one. When that is the default action,
// one line describing the fault is written to stderr first.

#pragma once

#include <armigo/execution/guest_preempt.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace armigo::execution {

// Installs the handler for SIGSEGV, SIGBUS, SIGILL and SIGTRAP once per
// process; later calls do nothing. A handler that must see its own faults
// first (the write tracker) is installed afterwards and chains to this one.
void install_guest_fault_handler() noexcept;
bool guest_fault_handler_installed() noexcept;

// Lists guest code for the handler while alive. The process holds a fixed
// number of ranges; a registration that does not fit lists nothing.
class GuestCodeRegistration {
public:
    explicit GuestCodeRegistration(std::span<const GuestCodeRange> ranges);
    ~GuestCodeRegistration();
    GuestCodeRegistration(const GuestCodeRegistration&) = delete;
    GuestCodeRegistration& operator=(const GuestCodeRegistration&) = delete;
    bool registered() const noexcept { return registered_; }

private:
    std::vector<std::uint16_t> entries_;
    bool registered_ = false;
};

// The fault that ended the slot's last invocation, cleared by the call.
// Called by the bound thread after its invocation returned as exited.
std::optional<GuestFault> take_guest_fault(GuestPreemptSlot& slot) noexcept;

// "SIGSEGV" and similar, or "signal".
const char* describe_guest_fault_signal(int signal) noexcept;

}  // namespace armigo::execution
