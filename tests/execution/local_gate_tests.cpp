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
#include <armigo/execution/embedded_patch.h>
#include <cstring>
#include <armigo/platform/host_module.h>

#include <array>
#include <cstdio>

extern "C" std::uint64_t armigo_test_local_gate(std::uintptr_t gate, std::uint64_t input);
extern "C" void armigo_test_local_gate_return();

namespace {

struct Context {
    std::uintptr_t entry = 0;
    unsigned calls = 0;
    bool valid = true;
};

armigo::execution::GateStatus handle(armigo::execution::GateCall& call, void* opaque) {
    auto& context = *static_cast<Context*>(opaque);
    ++context.calls;
    context.valid &= call.return_address() ==
                     reinterpret_cast<std::uintptr_t>(&armigo_test_local_gate_return);
    context.valid &= call.x(16) == context.entry && call.x(9) == 123;
    context.valid &= call.sp() == call.x(2) && call.frame().x[18] == 0;
    call.set_x(0, call.x(0) + 7);
    call.set_x(3, context.entry);
    call.set_x(9, 321);
    return armigo::execution::GateStatus::resume;
}

} // namespace

int main() {
    using namespace armigo;
    for (bool embedded : {false, true}) {
        const auto planned = execution::make_embedded_gate_code(0, 0x4008);
        if (!planned) return 1;
        auto code = embedded ? *planned : execution::make_local_gate_code();
        if (code.empty() || code.size() % 8 != 0) return 1;
        const auto code_size = code.size();
        std::vector<platform::ModuleRegion> regions{{0, code_size,
            memory::Permission::read | memory::Permission::execute}};
        if (embedded) {
            code.resize(0x8000);
            const auto dispatcher = execution::gate_dispatcher_address();
            std::memcpy(code.data() + 0x4008, &dispatcher, 8);
            regions.push_back({0x4000, 0x4000, memory::Permission::read | memory::Permission::write});
        }
        std::array<platform::HostModuleResult, 2> copies;
        for (auto& copy : copies) {
            copy = platform::HostModule::create(code, regions);
            if (!copy.ok() || !copy.module->seal().ok()) return 1;
        }
        if (copies[0].module->base_address() == copies[1].module->base_address()) return 1;
        for (auto& copy : copies) {
            const auto entry = copy.module->executable_address(0);
            if (!entry) return 1;
            Context context{*entry};
            execution::GateScope scope(handle, &context);
            for (std::uint64_t input : {0ull, 42ull, 0x1234567890ull}) {
                if (armigo_test_local_gate(*entry, input) != input + 7) return 1;
            }
            if (!context.valid || context.calls != 3 || scope.failure_count() != 0) return 1;
        }
    }
    std::puts("Independently mapped native gates preserve state and reach their bound dispatcher.");
    return 0;
}
