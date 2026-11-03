// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../../src/horizon/thread_lifecycle.h"
#include <armigo/horizon/startup_queries.h>
#include <atomic>
#include <cstdio>
#include <thread>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #value); return false; } } while (false)

namespace {
using namespace armigo::horizon;
bool contracts() {
    const auto create = resolve_thread_creation_core(-2, 2, 7);
    CHECK(create.result == 0 && create.ideal_core == 2 && create.affinity_mask == 4);
    CHECK(resolve_thread_creation_core(-1, 0, 7).result == 0xe201);
    CHECK(resolve_thread_creation_core(-3, 0, 7).result == 0xe201);
    CHECK(resolve_thread_creation_core(64, 0, 7).result == 0xe201);
    CHECK(resolve_thread_creation_core(3, 0, 7).result == 0xe201);
    CHECK(resolve_thread_creation_core(63, 0, 1ull << 63).affinity_mask == (1ull << 63));
    const auto default_core = resolve_thread_core_mask(-2, ~0ull, 1, 2, 7);
    CHECK(default_core.result == 0 && default_core.ideal_core == 2 && default_core.affinity_mask == 4);
    CHECK(resolve_thread_core_mask(-1, 0, 0, 0, 7).result == 0xe801);
    CHECK(resolve_thread_core_mask(-4, 8, 0, 0, 7).result == 0xe201);
    CHECK(resolve_thread_core_mask(2, 2, 0, 0, 7).result == 0xe801);
    CHECK(resolve_thread_core_mask(64, 2, 0, 0, 7).result == 0xe201);
    CHECK(resolve_thread_core_mask(-3, 2, 0, 0, 7).result == 0xe801);
    CHECK(resolve_thread_core_mask(-3, 2, -1, 0, 7).ideal_core == -1);
    CHECK(resolve_thread_core_mask(-3, 3, 1, 0, 7).ideal_core == 1);
    CHECK(select_virtual_processor(2, 7, 1) == 2);
    CHECK(select_virtual_processor(-1, 6, 2) == 2);
    CHECK(select_virtual_processor(-1, 2, 2) == 1);
    CHECK(!select_virtual_processor(-2, 7));
    CHECK(!select_virtual_processor(64, ~0ull));
    CHECK(!select_virtual_processor(-1, 0));
    return true;
}

bool coherent_foreign_rebind() {
    ProcessConfiguration configuration;
    configuration.heap_address = 0x1000000000;
    configuration.alias_address = 0x2000000000;
    configuration.stack_address = 0x6000000000;
    configuration.allowed_core_mask = 7;
    auto process = ProcessRecord::create(configuration, {0, 0, 0, 0, 0x100000, 0});
    CHECK(process.ok());
    ThreadRecord record(process.process, 1, 24);
    CHECK(record.configure_scheduling(1, 2, 1));
    CHECK(!record.configure_scheduling(2, 2, 1));
    CHECK(!record.configure_scheduling(1, 10, 1));
    CHECK(!record.configure_scheduling(-2, 7, 1));
    CHECK(record.scheduling_state().ideal_core == 1);
    std::atomic<bool> failed = false;
    std::atomic<bool> done = false;
    std::thread reader([&] {
        while (!done.load()) {
            const auto state = record.scheduling_state();
            if (!state.current_processor ||
                state.affinity_mask != (1ull << *state.current_processor) ||
                state.ideal_core != static_cast<std::int32_t>(*state.current_processor))
                failed = true;
        }
    });
    for (unsigned i = 0; i < 10000; ++i) {
        const auto core = i % 3;
        if (!record.configure_scheduling(core, 1ull << core, core)) failed = true;
    }
    done = true;
    reader.join();
    CHECK(!failed);
    record.unbind_current_processor();
    CHECK(!record.scheduling_state().current_processor);
    return true;
}
}

int main() { return contracts() && coherent_foreign_rebind() ? 0 : 1; }
