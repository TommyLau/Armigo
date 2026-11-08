// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/embedded_patch.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/aarch64/instruction.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {
using namespace armigo;
using namespace armigo::execution;
constexpr ScalarX18Policy contract{true, true, true};
constexpr std::size_t key_offset = 0x4000, appendix = 0x8000, image_size = 0xc000;
constexpr std::uint32_t ret = 0xd65f03c0;

bool expect(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

struct Fixture {
    std::vector<std::byte> bytes = std::vector<std::byte>(image_size);
    std::size_t cursor = appendix;
    void word(std::size_t offset, std::uint32_t value) {
        std::memcpy(bytes.data() + offset, &value, 4);
    }
    bool install(std::size_t offset, const std::optional<EmbeddedPatch>& patch) {
        if (!patch) return false;
        word(offset, patch->replacement_instruction);
        std::memcpy(bytes.data() + cursor, patch->trampoline.data(), patch->trampoline.size() * 4);
        cursor += patch->trampoline.size() * 4;
        return true;
    }
    bool scalar(std::size_t offset, std::uint32_t value) {
        return install(offset, make_embedded_scalar_x18_patch(value, offset, cursor, key_offset, contract));
    }
    platform::HostModuleResult map() {
        const auto rx = memory::Permission::read | memory::Permission::execute;
        const auto rw = memory::Permission::read | memory::Permission::write;
        const std::array regions = {platform::ModuleRegion{0, key_offset, rx},
            platform::ModuleRegion{key_offset, 0x4000, rw},
            platform::ModuleRegion{appendix, 0x4000, rx}};
        return platform::HostModule::create(bytes, regions);
    }
};

// Rebind only the key page between executions, using independently allocated
// pthread keys. The executable bytes remain identical across both bindings.
template<class Verify>
bool execute(Fixture& fixture, GuestThreadContext& context, Verify verify) {
    auto first = fixture.map();
    auto second = fixture.map();
    if (!expect(first.ok() && second.ok(), "Embedded fixture mapping failed.")) return false;
    platform::ThreadSlotKey keys[2];
    for (auto* module : {&*first.module, &*second.module}) {
        if (!expect(module->seal().ok(), "Embedded fixture sealing failed.")) return false;
        for (auto& key : keys) {
            if (!expect(key.native_access().has_value(), "Native slot probe failed.")) return false;
            const std::uint64_t slot = key.native_access()->tsd_byte_offset;
            std::memcpy(reinterpret_cast<void*>(module->base_address() + key_offset), &slot, 8);
            platform::ThreadSlotBinding binding(key, context.bound_register_cell());
            if (!expect(binding.native_access().has_value(), "Embedded slot binding failed.")) return false;
            if (!verify(module->base_address())) return false;
            if (!expect(std::memcmp(reinterpret_cast<void*>(module->base_address()),
                    fixture.bytes.data(), key_offset) == 0 &&
                    std::memcmp(reinterpret_cast<void*>(module->base_address() + appendix),
                    fixture.bytes.data() + appendix, 0x4000) == 0,
                    "Executable bytes changed during embedded execution.")) return false;
        }
    }
    return expect(first.module->base_address() != second.module->base_address() &&
                  keys[0].native_access()->tsd_byte_offset != keys[1].native_access()->tsd_byte_offset,
                  "Embedded rebasing test did not use distinct bases and slots.");
}

bool scalar(GuestThreadContext& context) {
    Fixture fixture;
    if (!fixture.scalar(0, 0xaa0003f2) || // MOV X18,X0
        !fixture.scalar(4, 0x91001e52) || // ADD X18,X18,#7
        !fixture.scalar(8, 0xaa1203e0)) return false; // MOV X0,X18
    fixture.word(12, ret);
    return execute(fixture, context, [&](std::uintptr_t base) {
        const auto run = reinterpret_cast<std::uint64_t(*)(std::uint64_t)>(base);
        for (const auto input : {0ull, 0xfffffffffffffff8ull, 0x123456789abcdef0ull}) {
            if (!expect(run(input) == input + 7 && context.guest_x18() == input + 7,
                        "Embedded scalar x18 result differs.")) return false;
        }
        return true;
    });
}

// SP pairs past the top of imm7 run as two single-register transfers through
// the embedded trampolines too.
bool far_pair(GuestThreadContext& context) {
    Fixture fixture;
    fixture.word(0, 0xd10883ff);                      // SUB SP,SP,#0x220
    if (!fixture.scalar(4, 0xaa0003f2) ||             // MOV X18,X0
        !fixture.scalar(8, 0xa91f4be1) ||             // STP X1,X18,[SP,#0x1f0]
        !fixture.scalar(12, 0xa95f4be0)) return false;  // LDP X0,X18,[SP,#0x1f0]
    fixture.word(16, 0x910883ff);                     // ADD SP,SP,#0x220
    fixture.word(20, ret);
    return execute(fixture, context, [&](std::uintptr_t base) {
        const auto run = reinterpret_cast<std::uint64_t(*)(std::uint64_t, std::uint64_t)>(base);
        return expect(run(0x0123456789abcdefull, 0xfedcba9876543210ull) == 0xfedcba9876543210ull &&
                          context.guest_x18() == 0x0123456789abcdefull,
                      "Embedded split stack pair moved the wrong values.");
    });
}

bool addresses(GuestThreadContext& context) {
    for (bool page : {false, true}) {
        Fixture fixture;
        const auto encoded = page ? aarch64::encode_adrp(18, 0, 0x1000)
                                  : aarch64::encode_adr(18, 0, 0x104);
        if (!fixture.scalar(0, encoded.word) || !fixture.scalar(4, 0xaa1203e0)) return false;
        fixture.word(8, ret);
        if (!execute(fixture, context, [&](std::uintptr_t base) {
            return expect(reinterpret_cast<std::uint64_t(*)()>(base)() == base + (page ? 0x1000 : 0x104),
                          "Embedded address generation retained a planning address.");
        })) return false;
    }
    return true;
}

bool conditional(GuestThreadContext& context) {
    for (bool test_bit : {false, true}) {
        Fixture fixture;
        const auto word = test_bit ? aarch64::encode_tbz(false, 18, 63, 0, 12).word
                                  : aarch64::encode_cbz(false, true, 18, 0, 12).word;
        if (!fixture.install(0, make_embedded_conditional_x18_patch(
            word, 0, fixture.cursor, key_offset, contract))) return false;
        fixture.word(4, 0xd2800020); // MOV X0,#1
        fixture.word(8, ret);
        fixture.word(12, 0xd2800040); // MOV X0,#2
        fixture.word(16, ret);
        if (!execute(fixture, context, [&](std::uintptr_t base) {
            for (auto input : {0ull, 1ull, 0x8000000000000000ull}) {
                context.set_guest_x18(input);
                const auto expected = (test_bit ? (input >> 63) == 0 : input == 0) ? 2u : 1u;
                if (!expect(reinterpret_cast<std::uint64_t(*)()>(base)() == expected &&
                            context.guest_x18() == input, "Embedded conditional branch differs.")) return false;
            }
            return true;
        })) return false;
    }
    return true;
}

bool indirect(GuestThreadContext& context) {
    Fixture fixture;
    if (!fixture.install(0, make_embedded_indirect_x18_table_patch(
        aarch64::encode_br(18).word, 0, fixture.cursor, key_offset, 0x100, 4, contract))) return false;
    for (std::size_t i = 0; i < 4; ++i) {
        fixture.word(0x100 + i * 4, aarch64::encode_b(0x100 + i * 4, 0x200 + i * 8).word);
        fixture.word(0x200 + i * 8, 0xd2800000u | (static_cast<unsigned>(i) << 5));
        fixture.word(0x204 + i * 8, ret);
    }
    return execute(fixture, context, [&](std::uintptr_t base) {
        for (std::size_t i = 0; i < 4; ++i) {
            context.set_guest_x18(base + 0x100 + i * 4);
            if (!expect(reinterpret_cast<std::uint64_t(*)()>(base)() == i,
                        "Embedded indirect table branch differs.")) return false;
        }
        return true;
    });
}

bool thread_registers(GuestThreadContext& context) {
    context.set_thread_pointer(0x123456789abcdef0);
    context.set_read_only_thread_pointer(0xfedcba9876543210);
    for (bool read_only : {false, true}) {
        for (unsigned reg : {0u, 1u, 17u}) {
            Fixture fixture;
            const auto word = aarch64::encode_mrs(reg,
                aarch64::system_register_encoding(3, 3, 13, 0, read_only ? 3 : 2)).word;
            if (!fixture.install(0, make_embedded_thread_register_patch(
                word, 0, fixture.cursor, key_offset, contract))) return false;
            fixture.word(4, 0xaa0003e0u | (reg << 16)); // MOV X0,Xreg
            fixture.word(8, ret);
            if (!execute(fixture, context, [&](std::uintptr_t base) {
                return expect(reinterpret_cast<std::uint64_t(*)()>(base)() ==
                    (read_only ? 0xfedcba9876543210ull : 0x123456789abcdef0ull),
                    "Embedded thread-register read differs.");
            })) return false;
        }
    }
    return true;
}

std::uint64_t host_add_seven(std::uint64_t value) { return value + 7; }
std::uint64_t host_add_nine(std::uint64_t value) { return value + 9; }

bool host_branch() {
    Fixture fixture;
    if (!fixture.install(0, make_embedded_host_branch(0, fixture.cursor, key_offset + 16))) return false;
    auto mapped = fixture.map();
    if (!mapped.ok() || !mapped.module->seal().ok()) return false;
    const auto base = mapped.module->base_address();
    for (bool alternate : {false, true}) {
        const auto target = reinterpret_cast<std::uintptr_t>(alternate ? &host_add_nine : &host_add_seven);
        std::memcpy(reinterpret_cast<void*>(base + key_offset + 16), &target, 8);
        if (!expect(reinterpret_cast<std::uint64_t(*)(std::uint64_t)>(base)(123) ==
                    (alternate ? 132 : 130), "Embedded host branch lost its arguments or return address.")) return false;
    }
    return expect(std::memcmp(reinterpret_cast<void*>(base), fixture.bytes.data(), key_offset) == 0 &&
        std::memcmp(reinterpret_cast<void*>(base + appendix), fixture.bytes.data() + appendix, 0x4000) == 0,
        "Host-target rebinding changed embedded executable bytes.");
}

bool refusals() {
    const auto scalar = [](std::uint64_t key, ScalarX18Policy policy = contract) {
        return make_embedded_scalar_x18_patch(0xaa1203e0, 0x100, 0x8000, key, policy);
    };
    return expect(!scalar(0x4001) && !scalar(0x8000) && !scalar(0x100) &&
        !scalar(0x100008000ull) && !scalar(0x4000, {}) &&
        !make_embedded_thread_register_patch(0xd53bd040, 0x100, 0x8000, 0x4000, {}),
        "Embedded preparation accepted an invalid key or missing contract.");
}
} // namespace

int main() {
    GuestThreadContext context;
    const auto table = GuestSiteTable::build({});
    if (!table.ok()) return 1;
    GuestDispatchScope dispatch(table.table, context);
    if (dispatch.setup_error() != GuestDispatchFailureKind::none) return 1;
    return refusals() && scalar(context) && far_pair(context) && addresses(context) && conditional(context) &&
        indirect(context) && thread_registers(context) && host_branch() ? 0 : 1;
}
