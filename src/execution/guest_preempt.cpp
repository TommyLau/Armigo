// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================


#include <armigo/execution/gate.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/execution/guest_preempt.h>

#include <pthread.h>

#include <mach/mach.h>
#include <mach/thread_act.h>

extern "C" void armigo_execution_guest_entry_exit(
    armigo::execution::GuestEntryRecord*, armigo::execution::GateFrame*) noexcept;

namespace armigo::execution {

GuestPreemptResult preempt_guest_thread(GuestPreemptSlot& slot,
                                        std::span<const GuestCodeRange> guest_code) noexcept {
    // Held across the suspension: the bound thread cannot unbind (and its
    // mach port cannot be recycled) while it is being inspected.
    const std::lock_guard lock(slot.mutex);
    const auto thread = slot.thread.load(std::memory_order_acquire);
    if (thread == 0 || thread == pthread_mach_thread_np(pthread_self())) return GuestPreemptResult::unavailable;

    if (thread_suspend(thread) != KERN_SUCCESS) return GuestPreemptResult::unavailable;
    // Nothing between here and thread_resume may allocate, log or lock: the
    // target may hold any host lock when it is in host code.
    auto result = GuestPreemptResult::unavailable;
    arm_thread_state64_t state{};
    mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
    if (thread_get_state(thread, ARM_THREAD_STATE64, reinterpret_cast<thread_state_t>(&state), &count) ==
        KERN_SUCCESS) {
        void* const record = slot.record.load(std::memory_order_acquire);
        const std::uint64_t low = slot.stack_low.load(std::memory_order_acquire);
        const std::uint64_t high = slot.stack_high.load(std::memory_order_acquire);
        const std::uint64_t pc = arm_thread_state64_get_pc(state);
        const std::uint64_t sp = arm_thread_state64_get_sp(state);
        bool in_guest = false;
        for (const auto& range : guest_code)
            if (pc >= range.begin && pc < range.end) { in_guest = true; break; }
        if (record == nullptr || !in_guest || sp < low || sp > high || (pc & 3) != 0) {
            result = GuestPreemptResult::not_in_guest;
        } else {
            arm_neon_state64_t neon{};
            mach_msg_type_number_t neon_count = ARM_NEON_STATE64_COUNT;
            if (thread_get_state(thread, ARM_NEON_STATE64, reinterpret_cast<thread_state_t>(&neon),
                                 &neon_count) == KERN_SUCCESS) {
                // The exit routine reads only FPCR and FPSR from its gate frame.
                auto* frame = slot.frame;
                const std::uint64_t fpsr = neon.__fpsr, fpcr = neon.__fpcr;
                __builtin_memcpy(frame + ARMIGO_GATE_FRAME_FPSR_OFFSET, &fpsr, sizeof fpsr);
                __builtin_memcpy(frame + ARMIGO_GATE_FRAME_FPCR_OFFSET, &fpcr, sizeof fpcr);
                state.__x[0] = reinterpret_cast<std::uint64_t>(record);
                state.__x[1] = reinterpret_cast<std::uint64_t>(frame);
                arm_thread_state64_set_pc_fptr(
                    state, reinterpret_cast<void*>(&armigo_execution_guest_entry_exit));
                if (thread_set_state(thread, ARM_THREAD_STATE64, reinterpret_cast<thread_state_t>(&state),
                                     ARM_THREAD_STATE64_COUNT) == KERN_SUCCESS)
                    result = GuestPreemptResult::stopped;
            }
        }
    }
    thread_resume(thread);
    return result;
}

}  // namespace armigo::execution
