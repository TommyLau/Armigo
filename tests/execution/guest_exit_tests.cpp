// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate_code.h>
#include <armigo/execution/gate_patch.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
extern const std::byte armigo_test_exit_begin[], armigo_test_exit_end[];
extern const std::byte armigo_test_exit_normal[], armigo_test_exit_svc[];
extern const std::byte armigo_test_exit_nested_svc[], armigo_test_exit_direct_return[];
extern const std::byte armigo_test_exit_thread[];
void armigo_test_exit_direct_gate();
void armigo_test_exit_host_probe(void (*)(void*), void*, void*);
}

namespace {
using namespace armigo;
using namespace armigo::execution;
using namespace armigo::platform;
int failures = 0;
#define CHECK(x) do { if (!(x)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (false)

std::uint64_t pointer(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }
struct CallbackCleanup { unsigned& count; ~CallbackCleanup() { ++count; } };
struct Run {
    GuardedStack* outer;
    GuardedStack* inner;
    std::uint64_t exit_entry;
    unsigned callbacks = 0;
    unsigned destructors = 0;
    bool nested = false;
    bool direct_nested = false;
    bool exit_outer = false;
};

GateStatus supervisor(std::uint16_t svc, GateCall& call, GuestThreadContext& thread, void* raw) {
    auto& run = *static_cast<Run*>(raw);
    CallbackCleanup cleanup{run.destructors};
    ++run.callbacks;
    if (svc == 0xa) {
        thread.set_thread_pointer(0x222);
        CHECK(call.can_exit_thread() == (!run.direct_nested && run.outer->in_use()));
        return GateStatus::exit_thread;
    }
    CHECK(svc == 0x7f);
    if (run.direct_nested) {
        CHECK(call.can_exit_thread());
        armigo_test_exit_direct_gate();
        CHECK(call.can_exit_thread());
    } else if (run.nested) {
        std::array<std::uint64_t, 2> marker{};
        const auto inner = invoke_guest_entry(*run.inner, {.function = run.exit_entry,
                                                         .x0 = pointer(marker.data())});
        CHECK(inner.exited() && !inner.returned());
        CHECK(marker[0] == 1 && marker[1] == 0);
        CHECK(!run.inner->in_use() && run.outer->in_use());
        CHECK(GateScope::thread_gate_depth() == 1);
        CHECK(run.destructors == 1);
    }
    return run.exit_outer ? GateStatus::exit_thread : GateStatus::resume;
}

struct Probe {
    std::array<std::uint64_t, 10> x;
    std::array<std::uint64_t, 8> d;
    std::uint64_t frame_matches, fpcr, fpsr;
};
static_assert(sizeof(Probe) == 168);
struct Invocation {
    GuardedStack& stack;
    std::uint64_t entry;
    std::array<std::uint64_t, 2> marker{};
    GuestEntryResult result;
};
void invoke(void* raw) {
    auto& invocation = *static_cast<Invocation*>(raw);
    invocation.result = invoke_guest_entry(invocation.stack, {
        .function = invocation.entry, .x0 = pointer(invocation.marker.data())});
}

struct ThreadInvocation {
    std::span<std::byte> stack;
    std::uint64_t entry;
    std::uint64_t sp;
    std::array<std::uint64_t, 12> report{};
    GuestEntryResult result;
};
void invoke_thread(void* raw) {
    auto& invocation = *static_cast<ThreadInvocation*>(raw);
    invocation.result = invoke_guest_thread(invocation.stack, {
        .function = invocation.entry, .x0 = pointer(invocation.report.data()),
        .x1 = 0x123456789abcdef0, .stack_pointer = invocation.sp});
}

} // namespace

int main() {
    const auto begin = pointer(armigo_test_exit_begin);
    const auto size = pointer(armigo_test_exit_end) - begin;
    CHECK(size < 1024);
    if (failures) return 1;
    std::vector<std::byte> bytes(4096);
    std::memcpy(bytes.data(), armigo_test_exit_begin, size);
    const auto gate = make_local_gate_code();
    CHECK(gate.size() < 1024);
    if (failures) return 1;
    std::memcpy(bytes.data() + 1024, gate.data(), gate.size());
    std::vector<GuestGateSite> relative_sites;
    for (const auto original : {armigo_test_exit_svc, armigo_test_exit_nested_svc}) {
        const auto offset = pointer(original) - begin;
        std::uint32_t word;
        std::memcpy(&word, bytes.data() + offset, 4);
        const auto patch = make_gate_site_patch(word, offset, 2048 + relative_sites.size() * 32,
                                                {1024, gate.size()});
        CHECK(patch.ok());
        if (!patch.ok()) return 1;
        std::memcpy(bytes.data() + offset, &patch.patch->replacement_instruction, 4);
        std::memcpy(bytes.data() + patch.patch->trampoline_address,
                    patch.patch->trampoline.data(), 16);
        relative_sites.push_back(patch.patch->dispatch_site);
    }
    const std::array regions = {ModuleRegion{0, bytes.size(), memory::Permission::read |
                                                           memory::Permission::execute}};
    auto mapped = HostModule::create(bytes, regions);
    CHECK(mapped.ok() && mapped.module->seal().ok());
    if (failures) return 1;
    const auto base = mapped.module->base_address();
    for (auto& site : relative_sites) site.return_address += base;
    relative_sites.push_back({pointer(armigo_test_exit_direct_return), aarch64::encode_svc(0xa).word});
    const auto table = GuestSiteTable::build(relative_sites);
    CHECK(table.ok());
    auto outer = GuardedStack::create({.usable_size = 1u << 20});
    auto inner = GuardedStack::create({.usable_size = 1u << 20});
    CHECK(outer.ok() && inner.ok());
    ThreadSlotKey key;
    CHECK(key.valid());
    if (failures) return 1;
    GuestThreadContext context;
    context.set_thread_pointer(0x100);
    set_thread_guest_x18(0xabc);
    const auto normal = base + pointer(armigo_test_exit_normal) - begin;

    // Exit from a nested guest function on the copied gate, with a host ABI probe.
    {
        Run run{outer.stack.get(), inner.stack.get(), base};
        GuestDispatchScope scope(table.table, context, {supervisor, &run});
        ThreadSlotBinding binding(key, context.bound_register_cell());
        CHECK(binding.native_access().has_value());
        Invocation invocation{*outer.stack, base, {}, {}};
        Probe probe{};
        armigo_test_exit_host_probe(invoke, &invocation, &probe);
        CHECK(invocation.result.exited() && invocation.result.ok() && !invocation.result.returned());
        CHECK(invocation.result.x0 == 0 && invocation.result.x1 == 0);
        CHECK(invocation.result.fpcr == 0 && invocation.result.fpsr == 1);
        CHECK(invocation.marker[0] == 1 && invocation.marker[1] == 0);
        CHECK(run.callbacks == 1 && run.destructors == 1);
        for (unsigned i = 0; i < probe.x.size(); ++i) CHECK(probe.x[i] == i + 19);
        for (unsigned i = 0; i < probe.d.size(); ++i) CHECK(probe.d[i] == i + 8);
        CHECK(probe.frame_matches == 1 && probe.fpcr == 0x400000 && probe.fpsr == 0x11);
        CHECK(!outer.stack->in_use() && GateScope::thread_gate_depth() == 0);
        CHECK(context.thread_pointer() == 0x222 && scope.failure_count() == 0);
    }
    CHECK(!context.bound() && thread_guest_x18() == 0xabc && key.active_bindings() == 0);

    // The original thread stack contract is a 4 KiB span and an exact SP, not
    // a GuardedStack top with a bridge record deducted from it.
    {
        auto allocation = GuardedStack::create({.usable_size = 4096});
        CHECK(allocation.ok());
        if (!allocation.ok()) return 1;
        auto& stack = *allocation.stack;
        const auto held = stack.try_acquire();
        CHECK(held.acquired());
        auto* bytes = reinterpret_cast<std::byte*>(stack.usable_base());
        std::memset(bytes, 0xa5, stack.usable_size());
        const auto base = stack.aligned_top() - 4096;
        ThreadInvocation invocation{
            {reinterpret_cast<std::byte*>(base), 4096},
            mapped.module->base_address() + pointer(armigo_test_exit_thread) - begin,
            base + 0xfe0, {}, {}};
        Run run{&stack, nullptr, invocation.entry};
        GuestDispatchScope scope(table.table, context, {supervisor, &run});
        ThreadSlotBinding binding(key, context.bound_register_cell());
        Probe probe{};
        armigo_test_exit_host_probe(invoke_thread, &invocation, &probe);
        CHECK(invocation.result.exited() && !invocation.result.returned());
        CHECK(invocation.result.x0 == 0 && invocation.result.x1 == 0);
        CHECK(invocation.result.fpcr == 0 && invocation.result.fpsr == 1);
        CHECK(invocation.report[0] == invocation.sp && invocation.report[1] == 0);
        CHECK(invocation.report[2] != 0 && invocation.report[3] == 0x123456789abcdef0);
        for (unsigned i = 4; i < invocation.report.size(); ++i)
            CHECK(invocation.report[i] == 0xa5a5a5a5a5a5a5a5);
        // Neither entry nor exit is allowed to wipe the 32 bytes above SP.
        for (std::size_t i = 0xfe0; i < 4096; ++i)
            CHECK(invocation.stack[i] == std::byte{0xa5});
        // No gate frame or callback may underflow the caller's 4 KiB span.
        for (std::size_t i = 0; i < base - stack.usable_base(); ++i)
            CHECK(bytes[i] == std::byte{0xa5});
        for (unsigned i = 0; i < probe.x.size(); ++i) CHECK(probe.x[i] == i + 19);
        for (unsigned i = 0; i < probe.d.size(); ++i) CHECK(probe.d[i] == i + 8);
        CHECK(probe.frame_matches == 1 && probe.fpcr == 0x400000 && probe.fpsr == 0x11);
        CHECK(run.callbacks == 1 && run.destructors == 1 && scope.failure_count() == 0);
        CHECK(GateScope::thread_gate_depth() == 0 && stack.in_use());
    }
    CHECK(!context.bound() && thread_guest_x18() == 0xabc && key.active_bindings() == 0);

    // The linked gate entry uses the same completion path as the copied gate.
    {
        Run run{outer.stack.get(), inner.stack.get(), base};
        GuestDispatchScope scope(table.table, context, {supervisor, &run});
        const auto result = invoke_guest_entry(*outer.stack, {
            .function = reinterpret_cast<std::uintptr_t>(&armigo_test_exit_direct_gate)});
        CHECK(result.exited() && run.destructors == 1 && scope.failure_count() == 0);
        CHECK(!outer.stack->in_use() && GateScope::thread_gate_depth() == 0);
    }

    // A nested invocation exits only itself; its host callback can then resume
    // the outer invocation, or request an independent outer exit.
    for (bool exit_outer : {false, true}) {
        Run run{outer.stack.get(), inner.stack.get(), base};
        run.nested = true;
        run.exit_outer = exit_outer;
        GuestDispatchScope scope(table.table, context, {supervisor, &run});
        ThreadSlotBinding binding(key, context.bound_register_cell());
        const auto result = invoke_guest_entry(*outer.stack, {.function = normal});
        CHECK(result.ok() && result.exited() == exit_outer && result.returned() != exit_outer);
        if (!exit_outer) CHECK(result.x0 == 42 && result.x1 == 43);
        CHECK(run.callbacks == 2 && run.destructors == 2);
        CHECK(!outer.stack->in_use() && !inner.stack->in_use());
        CHECK(GateScope::thread_gate_depth() == 0 && scope.failure_count() == 0);
    }

    // Without an invocation, exit is a failed gate and supervisor state rolls back.
    {
        context.set_thread_pointer(0x100);
        Run run{outer.stack.get(), inner.stack.get(), base};
        GuestDispatchScope scope(table.table, context, {supervisor, &run},
                                 GateFailurePolicy::resume_unchanged);
        armigo_test_exit_direct_gate();
        CHECK(run.destructors == 1 && scope.failure_count() == 1);
        CHECK(scope.gate_scope().failure_count() == 1 && context.thread_pointer() == 0x100);
    }
    // A raw nested gate is not a nested invocation: exiting it would skip the
    // outer supervisor's C++ frames, so it must fail and return normally.
    {
        Run run{outer.stack.get(), inner.stack.get(), base};
        run.direct_nested = true;
        GuestDispatchScope scope(table.table, context, {supervisor, &run},
                                 GateFailurePolicy::resume_unchanged);
        const auto result = invoke_guest_entry(*outer.stack, {.function = normal});
        CHECK(result.returned() && result.x0 == 42);
        CHECK(run.callbacks == 2 && run.destructors == 2 && scope.failure_count() == 1);
    }
    // Normal return still works after every exit and rejected exit path.
    {
        Run run{outer.stack.get(), inner.stack.get(), base};
        GuestDispatchScope scope(table.table, context, {supervisor, &run});
        const auto result = invoke_guest_entry(*outer.stack, {.function = normal});
        CHECK(result.returned() && result.x0 == 42 && result.x1 == 43);
        CHECK(run.destructors == 1 && !outer.stack->in_use());
    }
    CHECK(!context.bound() && key.active_bindings() == 0 && !GateScope::thread_has_scope());
    std::printf("Guest exit failures: %d\n", failures);
    return failures ? 1 : 0;
}
