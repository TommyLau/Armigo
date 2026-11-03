// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate.h>
#include <armigo/execution/gate_code.h>
#include <armigo/platform/guest_stack.h>
#include <armigo/platform/host_module.h>

#include "gate_exit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <pthread.h>

namespace {
using namespace armigo;
using namespace armigo::execution;

struct Record {
    std::uint64_t guest_top = 0;
    std::uint64_t gate = 0;
    std::uint64_t host_sp = 0;
    std::uint64_t exited_frame = 0;
};
static_assert(offsetof(Record, host_sp) == 16 && offsetof(Record, exited_frame) == 24);
extern "C" {
std::uint64_t armigo_test_callback_stack_enter(Record*);
void armigo_test_callback_stack_exit();
void armigo_test_callback_stack_reenter();
}

enum class Mode { resume, exit, direct_reentry, nested_entry };
struct Context {
    Mode mode = Mode::resume;
    std::uint64_t guest_base = 0;
    std::uint64_t guest_top = 0;
    std::uint64_t gate = 0;
    unsigned expected_depth = 1;
    unsigned calls = 0;
    bool valid = true;
};

bool run(Mode, std::uint64_t, unsigned = 1);

GateStatus handle(GateCall& call, void* raw) {
    auto& context = *static_cast<Context*>(raw);
    // Large enough that accidentally running this handler on four KiB faults.
    // Volatile access keeps the native stack allocation live around nested work.
    volatile std::array<unsigned char, 32768> probe{};
    auto* storage = reinterpret_cast<volatile unsigned char*>(&probe);
    for (std::size_t i = 0; i != sizeof(probe); ++i) storage[i] = static_cast<unsigned char>(i ^ 0x5a);
    const auto here = reinterpret_cast<std::uintptr_t>(&probe);
    const auto host_top = reinterpret_cast<std::uintptr_t>(pthread_get_stackaddr_np(pthread_self()));
    const auto host_base = host_top - pthread_get_stacksize_np(pthread_self());
    ++context.calls;
    context.valid &= here >= host_base && here + sizeof(probe) <= host_top &&
                     !(here >= context.guest_base && here < context.guest_top) &&
                     call.depth() == context.expected_depth;
    if (context.mode != Mode::direct_reentry || context.calls == 1) {
        context.valid &= call.sp() == context.guest_top && call.x(0) == 7 && call.x(1) == 11;
    }
    if (context.mode == Mode::direct_reentry && context.calls == 1) {
        context.expected_depth = 2;
        armigo_test_callback_stack_reenter();
        context.expected_depth = 1;
    } else if (context.mode == Mode::direct_reentry) {
        context.valid &= !call.can_exit_thread() && detail::GuestExitScope::callback_stack() == 0;
    } else if (context.mode == Mode::nested_entry) {
        context.valid &= run(Mode::exit, context.gate, context.expected_depth + 1);
    }
    for (std::size_t i = 0; i != sizeof(probe); ++i)
        context.valid &= storage[i] == static_cast<unsigned char>(i ^ 0x5a);
    if (context.mode == Mode::exit) {
        context.valid &= call.can_exit_thread();
        return GateStatus::exit_thread;
    }
    call.set_x(0, 42);
    return GateStatus::resume;
}

bool run(Mode mode, std::uint64_t gate, unsigned depth) {
    auto allocation = platform::GuardedStack::create({.usable_size = 4096});
    if (!allocation.ok()) return false;
    const auto base = allocation.stack->usable_base();
    // Native pages may exceed four KiB; the lower guard still abuts this range.
    // Only this first logical four-KiB span is made available to the fixture.
    std::memset(reinterpret_cast<void*>(base), 0xa5, 4096);
    Record record{base + 4096, gate};
    Context context{mode, base, record.guest_top, gate, depth};
    GateScope scope(handle, &context);
    detail::GuestExitScope entry(reinterpret_cast<std::uintptr_t>(&armigo_test_callback_stack_exit),
                                &record, &record.host_sp);
    const auto result = armigo_test_callback_stack_enter(&record);
    // Allow only the fixed frame, wrapper saves and bounded pre-pivot TLS
    // accessor spill area on the guest stack, never the C++ snapshot/handler.
    constexpr std::size_t guest_budget = sizeof(GateFrame) + 32 + 512;
    for (std::size_t i = 0; i != 4096 - guest_budget; ++i)
        context.valid &= reinterpret_cast<const unsigned char*>(base)[i] == 0xa5;
    context.valid &= result == (mode == Mode::exit ? 99u : 42u) &&
                     context.calls == (mode == Mode::direct_reentry ? 2u : 1u) &&
                     scope.failure_count() == 0;
    if (mode == Mode::exit)
        context.valid &= record.exited_frame == record.guest_top - sizeof(GateFrame);
    else context.valid &= record.exited_frame == 0;
    if (!context.valid) std::fprintf(stderr, "Host callback stack scenario failed: mode%u depth%u.\n",
                                     static_cast<unsigned>(mode), depth);
    return context.valid;
}
} // namespace

int main() {
    const auto linked = reinterpret_cast<std::uintptr_t>(&armigo_execution_gate_entry);
    const auto code = make_local_gate_code();
    const std::array regions = {platform::ModuleRegion{0, code.size(),
        memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(code, regions);
    if (!mapped.ok() || !mapped.module->seal().ok()) return 1;
    const std::array gates = {linked, mapped.module->base_address()};
    for (const auto gate : gates)
        for (const auto mode : {Mode::resume, Mode::exit, Mode::direct_reentry, Mode::nested_entry})
            if (!run(mode, gate)) return 1;
    std::puts("Linked/copied gates pivot four-KiB guest callbacks to the host stack, preserving reentry and exit.");
    return 0;
}
