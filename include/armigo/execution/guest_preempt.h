// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Asynchronous stop of a guest thread that runs guest code without reaching a
// gate. Process termination ends every thread, including one executing user
// code; a native guest thread has no supervisor boundary to stop at, so the
// terminating process redirects it to the entry bridge's exit path, as a gate
// exit_thread does.
//
// A thread qualifies only while its pc is inside an executable range of the
// guest modules (original text and the per-module gate copies) and its stack
// pointer is on the guest stack of its innermost invocation. Everything else is
// host code, however it got there, and is never touched: NVN entries and the
// write-tracker signal handler run on the guest stack, and so do dyld, the
// shared cache's stub islands and the signal trampoline. The caller retries.
// A thread that runs guest code on a stack other than the one its invocation
// published never qualifies either; the deadline then reports it as before.

#pragma once

#include <armigo/execution/gate_frame_layout.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace armigo::execution {

// A native fault that ended a guest invocation (guest_fault.h): the signal,
// the faulting address and syndrome, and the guest registers at the fault.
struct GuestFault {
    int signal = 0;
    int code = 0;                // siginfo si_code
    std::uint64_t address = 0;   // data address, or the instruction for SIGILL/SIGTRAP
    std::uint32_t syndrome = 0;  // ESR_EL1 as reported to the thread
    std::uint64_t pc = 0;
    std::uint64_t sp = 0;
    std::uint64_t pstate = 0;
    std::array<std::uint64_t, 31> x{};  // x0-x30; x29 is fp and x30 is lr
    std::uint64_t fpcr = 0;
    std::uint64_t fpsr = 0;
};

// One per guest host-thread, owned by the runtime for the thread's lifetime.
struct GuestPreemptSlot {
    std::mutex mutex;
    // Mach thread of the bound host thread; zero while unbound.
    std::atomic<std::uint32_t> thread{0};
    // Innermost GuestEntryRecord and its guest stack [low, high); published by
    // the invocation scopes of the bound thread.
    std::atomic<void*> record{nullptr};
    std::atomic<std::uint64_t> stack_low{0};
    std::atomic<std::uint64_t> stack_high{0};
    // Gate frame handed to the exit routine, which reads only FPCR and FPSR.
    alignas(16) std::byte frame[ARMIGO_GATE_FRAME_SIZE]{};
    // Written by the guest fault handler on the bound thread itself before it
    // ends the innermost invocation; nonzero signal marks `fault` as set.
    std::atomic<int> fault_signal{0};
    GuestFault fault;
};

// Binds the calling host thread to a slot until destroyed.
class GuestPreemptBinding {
public:
    explicit GuestPreemptBinding(GuestPreemptSlot& slot) noexcept;
    ~GuestPreemptBinding();
    GuestPreemptBinding(const GuestPreemptBinding&) = delete;
    GuestPreemptBinding& operator=(const GuestPreemptBinding&) = delete;

private:
    GuestPreemptSlot& slot_;
    GuestPreemptSlot* previous_;
};

// A half-open executable range of the guest modules' mappings.
struct GuestCodeRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

enum class GuestPreemptResult : std::uint8_t {
    stopped,       // redirected to the exit path; the invocation returns as exited
    not_in_guest,  // running host code or no live invocation; retry later
    unavailable,   // no bound thread, or the thread could not be inspected
};

// Safe to call from any thread, repeatedly, except from the bound thread
// itself (unavailable). Never blocks on the target. `guest_code` is computed
// once by the caller from the loaded module set.
GuestPreemptResult preempt_guest_thread(GuestPreemptSlot& slot,
                                        std::span<const GuestCodeRange> guest_code) noexcept;

// While alive, the calling thread publishes no invocation: guest code that runs
// on top of live host frames (a host callback into the guest) must not be
// unwound by an asynchronous stop, which would abandon those frames. The
// outer invocation is restored on destruction.
class GuestPreemptHold {
public:
    GuestPreemptHold() noexcept;
    ~GuestPreemptHold();
    GuestPreemptHold(const GuestPreemptHold&) = delete;
    GuestPreemptHold& operator=(const GuestPreemptHold&) = delete;

private:
    struct Saved {
        void* record;
        std::uint64_t stack_low;
        std::uint64_t stack_high;
    } saved_;
};

namespace detail {
// Called by invocation scopes of the bound thread.
struct PreemptPublication {
    void* record;
    std::uint64_t stack_low;
    std::uint64_t stack_high;
};
PreemptPublication publish_invocation(void* record, std::uint64_t stack_low,
                                      std::uint64_t stack_high) noexcept;
void restore_invocation(const PreemptPublication& previous) noexcept;
// Slot bound to the calling thread, or null.
GuestPreemptSlot* bound_slot() noexcept;
}  // namespace detail

}  // namespace armigo::execution
