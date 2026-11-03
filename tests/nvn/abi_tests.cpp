// ============================================================================
// Armigo Tests - NVN Direct-Call ABI Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "entry_table.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace armigo::nvn;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

extern "C" std::uint64_t armigo_nvn_guest_blr(GenericFunction target,
                                               const std::uint64_t* registers,
                                               const std::uint64_t* float_registers);

std::uint32_t observed_boolean = 0;
bool observed_value = false;

void set_blend_enable(NVNcolorState*, int, std::uint32_t enabled) {
    observed_boolean = enabled;
    observed_value = guest_bool(enabled);
}

NVNboolean get_blend_enable(const NVNcolorState*, int) { return static_cast<NVNboolean>(0xab); }

std::array<std::uint64_t, 8> observed_registers{};
void draw_elements_instanced(NVNcommandBuffer* command_buffer,
                             NVNdrawPrimitive primitive,
                             NVNindexType index_type,
                             int count,
                             NVNbufferAddress index_buffer,
                             int base_vertex,
                             int instance_count,
                             int base_instance) {
    observed_registers = {reinterpret_cast<std::uintptr_t>(command_buffer),
                          static_cast<std::uint32_t>(primitive),
                          static_cast<std::uint32_t>(index_type),
                          static_cast<std::uint32_t>(count),
                          index_buffer,
                          static_cast<std::uint32_t>(base_vertex),
                          static_cast<std::uint32_t>(instance_count),
                          static_cast<std::uint32_t>(base_instance)};
}

std::array<float, 3> observed_floats{};
void set_polygon_offset(NVNcommandBuffer*, float factor, float units, float clamp) {
    observed_floats = {factor, units, clamp};
}

constexpr std::size_t narrow_parameter_count = 0
#define ARMIGO_NVN_TYPED(name, result, parameters) \
    + (EntryAbi<result (*) parameters>::has_narrow_integer_parameter ? 1u : 0u)
    ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
#undef ARMIGO_NVN_TYPED
    ;

constexpr std::size_t narrow_result_count = 0
#define ARMIGO_NVN_TYPED(name, result, parameters) \
    + (EntryAbi<result (*) parameters>::has_narrow_integer_result ? 1u : 0u)
    ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
#undef ARMIGO_NVN_TYPED
    ;

constexpr std::size_t stack_signature_count = 0
#define ARMIGO_NVN_TYPED(name, result, parameters) \
    + (EntryAbi<result (*) parameters>::needs_stack_shim ? 1u : 0u)
    ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
#undef ARMIGO_NVN_TYPED
    ;

} // namespace

int main() {
    CHECK(narrow_parameter_count == 0);
    CHECK(narrow_result_count == 45);
    CHECK(stack_signature_count == 0);

    setenv("ARMIGO_NVN_COUNT", "0", 1);
    select_api_version({55, 15});
    install<Entry::nvnColorStateSetBlendEnable>(&set_blend_enable);
    install<Entry::nvnColorStateGetBlendEnable>(&get_blend_enable);

    NVNcolorState state{};
    std::array<std::uint64_t, 8> registers{};
    std::array<std::uint64_t, 8> float_registers{};
    registers[0] = reinterpret_cast<std::uintptr_t>(&state);
    registers[1] = 3;
    registers[2] = 0xdeadbeef00000100ull;
    auto target = armigo_nvn_bootstrap_loader("nvnColorStateSetBlendEnable");
    CHECK(target != nullptr);
    armigo_nvn_guest_blr(target, registers.data(), float_registers.data());
    CHECK(observed_boolean == 0x100u);
    CHECK(!observed_value);

    registers[2] = 0xcafebabe0000aa01ull;
    armigo_nvn_guest_blr(target, registers.data(), float_registers.data());
    CHECK(observed_boolean == 0xaa01u);
    CHECK(observed_value);

    const auto result_target = armigo_nvn_bootstrap_loader("nvnColorStateGetBlendEnable");
    const auto raw_result = armigo_nvn_guest_blr(result_target, registers.data(), float_registers.data());
    CHECK((raw_result & 0xffu) == 0xabu);

    setenv("ARMIGO_NVN_COUNT", "1", 1);
    install<Entry::nvnCommandBufferDrawElementsInstanced>(&draw_elements_instanced);
    const auto draw_target = armigo_nvn_bootstrap_loader("nvnCommandBufferDrawElementsInstanced");
    registers = {0x1000, 4, 2, 37, 0x123456789abcdef0ull, 11, 13, 17};
    armigo_nvn_guest_blr(draw_target, registers.data(), float_registers.data());
    CHECK(observed_registers == registers);

    install<Entry::nvnCommandBufferSetPolygonOffsetClamp>(&set_polygon_offset);
    const auto offset_target = armigo_nvn_bootstrap_loader("nvnCommandBufferSetPolygonOffsetClamp");
    registers = {};
    float_registers[0] = std::bit_cast<std::uint32_t>(1.5f);
    float_registers[1] = std::bit_cast<std::uint32_t>(-2.0f);
    float_registers[2] = std::bit_cast<std::uint32_t>(3.25f);
    armigo_nvn_guest_blr(offset_target, registers.data(), float_registers.data());
    CHECK(observed_floats == (std::array{1.5f, -2.0f, 3.25f}));

    for (const auto* enabled : {"0", "1"}) {
        setenv("ARMIGO_NVN_COUNT", enabled, 1);
        const auto float_stub = reinterpret_cast<EntrySignature<Entry::nvnSamplerGetLodBias>::type>(
            armigo_nvn_bootstrap_loader("nvnSamplerGetLodBias"));
        CHECK(float_stub(nullptr) == 0.0f);
        const auto pointer_stub = reinterpret_cast<EntrySignature<Entry::nvnTextureGetMemoryPool>::type>(
            armigo_nvn_bootstrap_loader("nvnTextureGetMemoryPool"));
        CHECK(pointer_stub(nullptr) == nullptr);
    }

    std::printf("NVN ABI failures: %u\n", failures);
    return failures ? 1 : 0;
}
