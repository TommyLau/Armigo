// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/register_effects.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/thread_register_patch.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace armigo;
using namespace armigo::execution;
using aarch64::system_register_encoding;

constexpr auto tpidr_el0 = system_register_encoding(3, 3, 13, 0, 2);
constexpr auto tpidrro_el0 = system_register_encoding(3, 3, 13, 0, 3);
constexpr auto cntvct_el0 = system_register_encoding(3, 3, 14, 0, 2);

bool expect(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

bool rejected(std::uint32_t word, std::uint64_t address, std::uint64_t trampoline,
              std::uint32_t slot, ThreadRegisterPatchError expected) {
    const auto result = make_thread_register_read_patch(word, address, trampoline, slot);
    if (!expect(!result.ok() && result.error == expected, "Incorrect thread register refusal.")) {
        std::fprintf(stderr, "word=%08x expected=%u\n", word, static_cast<unsigned>(expected));
        return false;
    }
    return true;
}

bool test_refusals() {
    if (!expect(make_thread_register_read_patch(aarch64::encode_mrs(1, tpidrro_el0).word, 0x10000000,
                                                0x17fffff8, 32760).ok(),
                "Valid branch and slot encoding boundaries were rejected.")) return false;
    const auto mrs = [](unsigned rt, std::uint16_t reg) { return aarch64::encode_mrs(rt, reg).word; };
    return rejected(mrs(1, tpidrro_el0), 1, 4096, 0, ThreadRegisterPatchError::invalid_layout) &&
           rejected(mrs(1, tpidrro_el0), 0, 4098, 0, ThreadRegisterPatchError::invalid_layout) &&
           rejected(mrs(1, tpidrro_el0), 0, 4, 0, ThreadRegisterPatchError::overlapping_regions) &&
           rejected(mrs(1, tpidrro_el0), 0, 4096, 4, ThreadRegisterPatchError::invalid_thread_slot) &&
           rejected(mrs(1, tpidrro_el0), 0, 4096, 32768,
                    ThreadRegisterPatchError::invalid_thread_slot) &&
           rejected(mrs(1, tpidrro_el0), 0, 1ull << 28, 0,
                    ThreadRegisterPatchError::unreachable_branch) &&
           // The entry branch reaches its limit but the resume branch cannot.
           rejected(mrs(1, tpidrro_el0), 0x10000000, 0x17fffffc, 0,
                    ThreadRegisterPatchError::unreachable_branch) &&
           rejected(mrs(31, tpidrro_el0), 0, 4096, 0,
                    ThreadRegisterPatchError::unsupported_instruction) &&
           rejected(mrs(18, tpidr_el0), 0, 4096, 0,
                    ThreadRegisterPatchError::unsupported_instruction) &&
           rejected(mrs(1, cntvct_el0), 0, 4096, 0,
                    ThreadRegisterPatchError::unsupported_instruction) &&
           rejected(aarch64::encode_msr(tpidr_el0, 1).word, 0, 4096, 0,
                    ThreadRegisterPatchError::unsupported_instruction) &&
           rejected(aarch64::encode_svc(1).word, 0, 4096, 0,
                    ThreadRegisterPatchError::unsupported_instruction) &&
           rejected(0xd503201f, 0, 4096, 0, ThreadRegisterPatchError::unsupported_instruction);
}

// Each case is MRS Xrt, <register>; STR Xrt, [X0]; RET, with callee-saved
// registers and x30 parked in x16 around the read. Offsets are 0x40 apart; the trampolines follow at 0x800.
struct Case { unsigned rt; std::uint16_t reg; };

}  // namespace

int main() {
    if (!test_refusals()) return 1;
    platform::ThreadSlotKey key;
    if (!expect(key.valid(), "Native thread slot probe failed.")) return 1;
    const auto slot = key.native_access()->tsd_byte_offset;

    const Case cases[] = {{1, tpidrro_el0}, {1, tpidr_el0}, {9, tpidrro_el0}, {17, tpidr_el0},
                          {28, tpidrro_el0}, {29, tpidr_el0}, {30, tpidrro_el0}, {30, tpidr_el0}};
    std::vector<std::byte> code(4096);
    auto put = [&](std::size_t offset, std::uint32_t word) {
        std::memcpy(code.data() + offset, &word, 4);
    };
    for (std::size_t i = 0; i < std::size(cases); ++i) {
        const std::size_t base = i * 0x40;
        const unsigned rt = cases[i].rt;
        std::size_t at = base;
        const bool parked = rt >= 19;                      // callee-saved or link register
        if (parked) put(at, 0xaa0003f0u | (rt << 16)), at += 4;  // MOV X16, Xrt
        const auto site = at;
        put(at, aarch64::encode_mrs(rt, cases[i].reg).word), at += 4;
        put(at, 0xf9000000u | rt), at += 4;                // STR Xrt, [X0]
        if (parked) put(at, 0xaa1003e0u | rt), at += 4;    // MOV Xrt, X16
        put(at, 0xd65f03c0u);                              // RET
        std::uint32_t word;
        std::memcpy(&word, code.data() + site, 4);
        const auto planned = make_thread_register_read_patch(word, site, 0x800 + i * 16, slot);
        if (!expect(planned.ok(), "An authored thread register read was rejected.")) return 1;
        const auto& patch = *planned.patch;
        if (!expect(patch.expected_instruction == word && patch.destination_register == rt,
                    "Invalid thread register patch identity.")) return 1;
        for (const auto emitted : patch.trampoline) {
            const auto effects = aarch64::decode_register_effects(emitted);
            if (!expect(aarch64::proven_free_of(effects, 18),
                        "Generated trampoline uses hardware x18 or has unknown effects.")) return 1;
        }
        if (!expect(aarch64::decode(patch.trampoline[0]).opcode == aarch64::Opcode::mrs &&
                        aarch64::decode(patch.trampoline[3]).opcode == aarch64::Opcode::b,
                    "Unexpected trampoline entry or exit.")) return 1;
        put(site, patch.replacement_instruction);
        std::memcpy(code.data() + patch.trampoline_address, patch.trampoline.data(), 16);
    }
    const std::array regions = {platform::ModuleRegion{
        0, code.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(code, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "Prepared fixture mapping failed.")) return 1;

    const auto table = GuestSiteTable::build({});
    if (!expect(table.ok(), "An empty dispatch table was rejected.")) return 1;
    GuestThreadContext context;
    context.set_thread_pointer(0x1111222233334444ull);
    context.set_read_only_thread_pointer(0x5555666677778888ull);
    context.set_guest_x18(0x99);
    GuestDispatchScope dispatch(table.table, context);
    if (!expect(dispatch.setup_error() == GuestDispatchFailureKind::none,
                "Guest context binding failed.")) return 1;
    platform::ThreadSlotBinding binding(key, context.bound_register_cell());
    if (!expect(binding.native_access().has_value(), "Guest register publication failed.")) return 1;
    for (std::size_t i = 0; i < std::size(cases); ++i) {
        const auto entry = mapped.module->executable_address(i * 0x40);
        if (!expect(entry.has_value(), "Prepared fixture has no executable entry.")) return 1;
        std::uint64_t out = 0;
        reinterpret_cast<void (*)(std::uint64_t*)>(*entry)(&out);
        const auto expected = cases[i].reg == tpidr_el0 ? context.thread_pointer()
                                                        : context.read_only_thread_pointer();
        if (!expect(out == expected, "A thread register read returned the wrong guest value.")) {
            std::fprintf(stderr, "case %zu: got %llx expected %llx\n", i,
                         static_cast<unsigned long long>(out),
                         static_cast<unsigned long long>(expected));
            return 1;
        }
    }
    // The inline read observes the same cell a gate handler would: a changed
    // value is seen by the next execution, and x18 is not disturbed.
    context.set_read_only_thread_pointer(0xabcdef);
    std::uint64_t out = 0;
    reinterpret_cast<void (*)(std::uint64_t*)>(*mapped.module->executable_address(0))(&out);
    if (!expect(out == 0xabcdef && context.guest_x18() == 0x99,
                "The inline read did not follow the cell or changed guest x18.")) return 1;
    std::puts("Thread pointer reads executed inline through the guest register cell.");
    return 0;
}
