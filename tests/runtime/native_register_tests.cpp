// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/platform/thread_slot.h>

#include <array>
#include <cstdio>
#include <thread>

extern "C" void armigo_test_cell_round_trip(std::uint64_t, std::uint64_t, std::uint64_t*);
extern "C" const std::byte armigo_test_cell_gate_return[];
extern "C" const std::byte armigo_test_cell_guarded_entry[];
extern "C" std::uint64_t armigo_test_thread_slot_load(std::uint64_t);
extern "C" std::uint64_t armigo_test_thread_slot_load_word(std::uint64_t, std::uint64_t);

namespace {
using namespace armigo::execution;
using namespace armigo::platform;

bool expect(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

struct Run {
    ThreadSlotKey& key;
    std::shared_ptr<const GuestSiteTable> table;
    std::uint64_t seed;
    bool fail = false;
    bool nest = false;
    bool correct = true;
};

GateStatus serve(std::uint16_t immediate, GateCall& call, GuestThreadContext& context, void* raw) {
    auto& run = *static_cast<Run*>(raw);
    const auto offset = run.key.native_access()->tsd_byte_offset;
    auto* const cell = context.bound_register_cell();
    run.correct &= immediate == 1 && cell != nullptr &&
        armigo_test_thread_slot_load(offset) == reinterpret_cast<std::uintptr_t>(cell) &&
        call.guest_x18() == run.seed && context.thread_pointer() == run.seed + 1 &&
        context.read_only_thread_pointer() == run.seed + 2;
    if (run.nest) {
        GuestThreadContext nested;
        Run nested_run{run.key, run.table, 300};
        GuestDispatchScope nested_dispatch(run.table, nested, {serve, &nested_run});
        ThreadSlotBinding nested_binding(run.key, nested.bound_register_cell());
        if (!nested_binding.native_access()) {
            run.correct = false;
            return GateStatus::failed;
        }
        std::array<std::uint64_t, 4> output{};
        armigo_test_cell_round_trip(offset, nested_run.seed, output.data());
        run.correct &= nested_run.correct && output == std::array<std::uint64_t, 4>{7, 310, 311, 312};
        // An outer GateCall keeps its own cell even while the slot and current
        // gate select a different context. Native reads still see the inner one.
        call.set_guest_x18(run.seed + 10);
        run.correct &= context.guest_x18() == run.seed + 10 &&
            armigo_test_thread_slot_load_word(offset, 0) == 310;
    } else {
        call.set_guest_x18(run.seed + 10);
    }
    run.correct &= armigo_test_thread_slot_load(offset) == reinterpret_cast<std::uintptr_t>(cell);
    context.set_thread_pointer(run.seed + 11);
    context.set_read_only_thread_pointer(run.seed + 12);
    call.set_x(0, 7);
    return run.fail ? GateStatus::failed : GateStatus::resume;
}

bool round_trip(ThreadSlotKey& key, const std::shared_ptr<const GuestSiteTable>& table,
                GuestThreadContext& context, bool fail, bool nest,
                GuardedStack* stack = nullptr) {
    Run run{key, table, 100, fail, nest};
    GuestDispatchScope dispatch(table, context, {serve, &run}, GateFailurePolicy::resume_unchanged);
    if (!expect(dispatch.setup_error() == GuestDispatchFailureKind::none,
                "Guest dispatch binding failed.")) return false;
    ThreadSlotBinding binding(key, context.bound_register_cell());
    const auto access = binding.native_access();
    if (!expect(access.has_value(), "Native register cell publication failed.")) return false;
    std::array<std::uint64_t, 4> output{};
    if (stack != nullptr) {
        const std::array arguments = {run.seed, reinterpret_cast<std::uint64_t>(output.data())};
        const auto entered = invoke_guest_entry(*stack, {
            .function = reinterpret_cast<std::uintptr_t>(armigo_test_cell_guarded_entry),
            .x0 = access->tsd_byte_offset,
            .x1 = reinterpret_cast<std::uintptr_t>(arguments.data())});
        if (!expect(entered.ok(), "Guarded native register entry failed.")) return false;
    } else {
        armigo_test_cell_round_trip(access->tsd_byte_offset, run.seed, output.data());
    }
    const std::array<std::uint64_t, 4> expected = fail ?
        std::array<std::uint64_t, 4>{access->tsd_byte_offset, 100, 101, 102} :
        std::array<std::uint64_t, 4>{7, 110, 111, 112};
    return expect(run.correct && output == expected &&
                      dispatch.failure_count() == (fail ? 1u : 0u),
                  "Native and gate register cells disagree.");
}
} // namespace

int main() {
    ThreadSlotKey key;
    if (!expect(key.valid(), "Host thread slot is unavailable.")) return 1;
    const auto offset = key.native_access()->tsd_byte_offset;
    const std::array sites = {GuestGateSite{
        reinterpret_cast<std::uintptr_t>(armigo_test_cell_gate_return),
        armigo::aarch64::encode_svc(1).word}};
    const auto built = GuestSiteTable::build(sites);
    if (!expect(built.ok(), "Authored gate site was rejected.")) return 1;
    GuestThreadContext context;
    set_thread_guest_x18(900);
    for (const bool fail : {false, true}) {
        for (const bool nest : {false, true}) {
            if (!round_trip(key, built.table, context, fail, nest)) return 1;
            if (!expect(armigo_test_thread_slot_load(offset) == 0 &&
                            thread_guest_x18() == 900 && !context.bound(),
                        "Unbinding left native or gate storage selected.")) return 1;
        }
    }
    auto guarded = GuardedStack::create({.usable_size = 1u << 20});
    if (!expect(guarded.ok(), "Guarded composition stack allocation failed.")) return 1;
    for (const bool fail : {false, true}) {
        for (const bool nest : {false, true}) {
            if (!round_trip(key, built.table, context, fail, nest, guarded.stack.get())) return 1;
            if (!expect(!guarded.stack->in_use() && !context.bound() &&
                            armigo_test_thread_slot_load(offset) == 0 &&
                            thread_guest_x18() == 900,
                        "Guarded entry retained a stack, context, or native slot binding.")) return 1;
        }
    }

    // Join orders cell ownership transfer. The new host thread sees the
    // previous guest state, while its own default virtual register is separate.
    bool migrated = false;
    std::thread next([&] {
        set_thread_guest_x18(800);
        {
            GuestDispatchScope dispatch(built.table, context);
            ThreadSlotBinding binding(key, context.bound_register_cell());
            migrated = binding.native_access().has_value() &&
                armigo_test_thread_slot_load_word(offset, 0) == 100 &&
                armigo_test_thread_slot_load_word(offset, 1) == 101 &&
                armigo_test_thread_slot_load_word(offset, 2) == 102 &&
                thread_guest_x18() == 100;
        }
        migrated &= armigo_test_thread_slot_load(offset) == 0 && thread_guest_x18() == 800;
    });
    next.join();
    if (!expect(migrated, "A guest register cell did not survive host-thread rebinding.")) return 1;

    // Failed ownership acquisition must never publish another thread's cell.
    {
        GuestDispatchScope dispatch(built.table, context);
        ThreadSlotBinding binding(key, context.bound_register_cell());
        bool refused = false;
        std::thread foreign([&] {
            GuestDispatchScope rejected(built.table, context, {},
                                        GateFailurePolicy::resume_unchanged);
            ThreadSlotBinding invalid(key, context.bound_register_cell());
            refused = rejected.setup_error() == GuestDispatchFailureKind::context_bound_elsewhere &&
                !invalid.native_access() && armigo_test_thread_slot_load(offset) == 0;
        });
        foreign.join();
        if (!expect(refused, "A rejected context binding exposed foreign storage.")) return 1;
    }
    std::puts("Native slot and gate share one guest register cell.");
    return 0;
}
