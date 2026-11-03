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

#include <mach/mach.h>
#include "../platform/mach_vm.h"

#include <cstddef>
#include <limits>

#include "gate_exit.h"

extern "C" void armigo_execution_guest_entry_exit(
    armigo::execution::GuestEntryRecord*, armigo::execution::GateFrame*) noexcept;

namespace armigo::execution {
namespace {

static_assert(offsetof(GuestEntryRecord, function) == ARMIGO_GUEST_ENTRY_RECORD_FUNCTION_OFFSET);
static_assert(offsetof(GuestEntryRecord, stack_top) == ARMIGO_GUEST_ENTRY_RECORD_STACK_TOP_OFFSET);
static_assert(offsetof(GuestEntryRecord, x0) == ARMIGO_GUEST_ENTRY_RECORD_X0_OFFSET);
static_assert(offsetof(GuestEntryRecord, x1) == ARMIGO_GUEST_ENTRY_RECORD_X1_OFFSET);
static_assert(offsetof(GuestEntryRecord, fpcr) == ARMIGO_GUEST_ENTRY_RECORD_FPCR_OFFSET);
static_assert(offsetof(GuestEntryRecord, result_x0) == ARMIGO_GUEST_ENTRY_RECORD_RESULT_X0_OFFSET);
static_assert(offsetof(GuestEntryRecord, result_x1) == ARMIGO_GUEST_ENTRY_RECORD_RESULT_X1_OFFSET);
static_assert(offsetof(GuestEntryRecord, result_fpcr) ==
              ARMIGO_GUEST_ENTRY_RECORD_RESULT_FPCR_OFFSET);
static_assert(offsetof(GuestEntryRecord, result_fpsr) ==
              ARMIGO_GUEST_ENTRY_RECORD_RESULT_FPSR_OFFSET);
static_assert(offsetof(GuestEntryRecord, host_sp) == ARMIGO_GUEST_ENTRY_RECORD_HOST_SP_OFFSET);
static_assert(offsetof(GuestEntryRecord, completion) == ARMIGO_GUEST_ENTRY_RECORD_COMPLETION_OFFSET);
static_assert(offsetof(GuestEntryRecord, mode) == ARMIGO_GUEST_ENTRY_RECORD_MODE_OFFSET);
static_assert(offsetof(GuestEntryRecord, fpsr) == ARMIGO_GUEST_ENTRY_RECORD_FPSR_OFFSET);
static_assert(sizeof(GuestEntryRecord) == ARMIGO_GUEST_ENTRY_RECORD_SIZE);
static_assert(ARMIGO_GUEST_ENTRY_STACK_RESERVE % 16 == 0);
static_assert(ARMIGO_GUEST_ENTRY_HOST_FRAME_SIZE % 16 == 0);

std::uint64_t read_fpcr() noexcept {
    std::uint64_t value;
    asm volatile("mrs %0, fpcr" : "=r"(value));
    return value;
}

// The function must start inside one mapping that is executable and not
// writable. This rejects data, the guest stack itself, and code still in its
// writable preparation state; it cannot prove that the code was prepared.
bool executable_code(std::uint64_t function) noexcept {
    mach_vm_address_t address = function;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const kern_return_t result =
        mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (result != KERN_SUCCESS) return false;
    if (address > function || function - address >= size) return false;  // a later region
    return (info.protection & VM_PROT_EXECUTE) != 0 && (info.protection & VM_PROT_WRITE) == 0;
}

bool writable_stack(std::uint64_t base, std::uint64_t end) noexcept {
    auto cursor = base;
    while (cursor < end) {
        mach_vm_address_t address = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        const auto result = mach_vm_region(
            mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info), &count, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        if (result != KERN_SUCCESS || address > cursor || size == 0 ||
            size > std::numeric_limits<std::uint64_t>::max() - address ||
            (info.protection & (VM_PROT_READ | VM_PROT_WRITE)) !=
                (VM_PROT_READ | VM_PROT_WRITE))
            return false;
        const auto region_end = address + size;
        if (region_end <= cursor) return false;
        cursor = region_end;
    }
    return true;
}

GuestEntryResult failure(GuestEntryError error) noexcept {
    GuestEntryResult result;
    result.error = error;
    return result;
}

}  // namespace

const char* describe(GuestEntryError error) noexcept {
    switch (error) {
    case GuestEntryError::none: return "none";
    case GuestEntryError::null_function: return "function address is zero";
    case GuestEntryError::misaligned_function: return "function address is not 4-byte aligned";
    case GuestEntryError::function_not_executable:
        return "function is not in an executable, non-writable mapping";
    case GuestEntryError::misaligned_stack_top: return "stack top is not 16-byte aligned";
    case GuestEntryError::stack_top_out_of_range: return "stack top is outside the guarded stack";
    case GuestEntryError::insufficient_stack: return "stack has less than the required space";
    case GuestEntryError::invalid_fpcr: return "FPCR value sets unsupported bits";
    case GuestEntryError::no_gate_scope: return "no gate scope is active on this thread";
    case GuestEntryError::stack_busy: return "stack is already in use";
    case GuestEntryError::stack_not_writable:
        return "borrowed thread stack is not entirely readable and writable";
    case GuestEntryError::invalid_process_landing:
        return "original-process transfer landing is invalid";
    }
    return "unknown guest entry error";
}

GuestEntryResult invoke_guest_entry(platform::GuardedStack& stack,
                                    const GuestEntryRequest& request) noexcept {
    if (request.function == 0) return failure(GuestEntryError::null_function);
    if ((request.function & 3) != 0) return failure(GuestEntryError::misaligned_function);
    if (request.fpcr && (*request.fpcr & ~kGuestEntryFpcrMask) != 0) {
        return failure(GuestEntryError::invalid_fpcr);
    }

    const std::uint64_t top = request.stack_top.value_or(stack.aligned_top());
    if ((top & 15) != 0) return failure(GuestEntryError::misaligned_stack_top);
    if (top > stack.aligned_top() ||
        top < stack.usable_base() + ARMIGO_GUEST_ENTRY_STACK_RESERVE) {
        return failure(GuestEntryError::stack_top_out_of_range);
    }
    const std::uint64_t entry_sp = top - ARMIGO_GUEST_ENTRY_STACK_RESERVE;
    if (entry_sp - stack.usable_base() < request.minimum_stack_bytes) {
        return failure(GuestEntryError::insufficient_stack);
    }
    if (!executable_code(request.function)) {
        return failure(GuestEntryError::function_not_executable);
    }
    if (request.require_gate_scope && !GateScope::thread_has_scope()) {
        return failure(GuestEntryError::no_gate_scope);
    }

    const platform::GuardedStackUse use = stack.try_acquire();
    if (!use.acquired()) return failure(GuestEntryError::stack_busy);

    GuestEntryRecord record;
    record.function = request.function;
    record.stack_top = top;
    record.x0 = request.x0;
    record.x1 = request.x1;
    record.fpcr = request.fpcr.value_or(read_fpcr());
    record.fpsr = request.fpsr;
    const detail::GuestExitScope exit_scope(
        reinterpret_cast<std::uintptr_t>(&armigo_execution_guest_entry_exit), &record,
        &record.host_sp, stack.usable_base(), stack.aligned_top());
    armigo_execution_guest_entry_bridge(&record);

    GuestEntryResult result;
    result.x0 = record.result_x0;
    result.x1 = record.result_x1;
    result.fpcr = record.result_fpcr;
    result.fpsr = record.result_fpsr;
    result.completion = static_cast<GuestEntryCompletion>(record.completion);
    return result;
}

GuestEntryResult invoke_guest_thread(std::span<std::byte> pinned_stack,
                                     const GuestThreadEntryRequest& request) noexcept {
    if (request.function == 0) return failure(GuestEntryError::null_function);
    if ((request.function & 3) != 0) return failure(GuestEntryError::misaligned_function);
    if (request.fpcr && (*request.fpcr & ~kGuestEntryFpcrMask) != 0)
        return failure(GuestEntryError::invalid_fpcr);
    const auto sp = request.stack_pointer;
    if ((sp & 15) != 0) return failure(GuestEntryError::misaligned_stack_top);
    const auto base = reinterpret_cast<std::uintptr_t>(pinned_stack.data());
    if (pinned_stack.empty() || base == 0 ||
        pinned_stack.size() > std::numeric_limits<std::uint64_t>::max() - base ||
        sp < base || sp > base + pinned_stack.size())
        return failure(GuestEntryError::stack_top_out_of_range);
    if (sp == base) return failure(GuestEntryError::insufficient_stack);
    if (!writable_stack(base, base + pinned_stack.size()))
        return failure(GuestEntryError::stack_not_writable);
    if (!executable_code(request.function))
        return failure(GuestEntryError::function_not_executable);
    if (request.original_process) {
        if (!request.transfer_function || (request.transfer_function & 3) != 0 ||
            !executable_code(request.transfer_function))
            return failure(GuestEntryError::invalid_process_landing);
    } else if (request.transfer_function != 0) {
        return failure(GuestEntryError::invalid_process_landing);
    }
    if (request.returns && (request.original_process ||
                            sp - base < ARMIGO_GUEST_ENTRY_STACK_RESERVE + kGuestEntryDefaultMinimumStack))
        return failure(request.original_process ? GuestEntryError::invalid_process_landing
                                                : GuestEntryError::insufficient_stack);
    if (request.require_gate_scope && !GateScope::thread_has_scope())
        return failure(GuestEntryError::no_gate_scope);

    GuestEntryRecord record;
    record.function = request.original_process ? request.transfer_function : request.function;
    record.stack_top = sp;
    record.x0 = request.x0;
    record.x1 = request.x1;
    record.fpcr = request.fpcr.value_or(read_fpcr());
    record.fpsr = request.fpsr;
    record.mode = request.original_process ? ARMIGO_GUEST_ENTRY_MODE_PROCESS
                  : request.returns        ? ARMIGO_GUEST_ENTRY_MODE_CALL
                                           : ARMIGO_GUEST_ENTRY_MODE_THREAD;
    const detail::GuestExitScope exit_scope(
        reinterpret_cast<std::uintptr_t>(&armigo_execution_guest_entry_exit), &record,
        &record.host_sp, base, base + pinned_stack.size());
    armigo_execution_guest_entry_bridge(&record);
    GuestEntryResult result;
    result.x0 = record.result_x0;
    result.x1 = record.result_x1;
    result.fpcr = record.result_fpcr;
    result.fpsr = record.result_fpsr;
    result.completion = static_cast<GuestEntryCompletion>(record.completion);
    return result;
}

}  // namespace armigo::execution
