// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/register_effects.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/scalar_x18_patch.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

extern "C" const std::byte armigo_test_scalar_x18_begin[];
extern "C" const std::byte armigo_test_scalar_x18_end[];
extern "C" const std::uint32_t armigo_test_scalar_x18_offsets[10];
extern "C" const std::byte armigo_test_adrp_x18_begin[];
extern "C" const std::byte armigo_test_adrp_x18_end[];
extern "C" const std::uint32_t armigo_test_adrp_x18_offset;
extern "C" const std::byte armigo_test_adr_x18_begin[];
extern "C" const std::byte armigo_test_adr_x18_end[];
extern "C" const std::uint32_t armigo_test_adr_x18_offsets[3];
extern "C" const std::uint32_t armigo_test_adr_x18_target;
extern "C" const std::byte armigo_test_sp_x18_begin[];
extern "C" const std::byte armigo_test_sp_x18_end[];
extern "C" const std::uint32_t armigo_test_sp_x18_offsets[6];
extern "C" const std::byte armigo_test_sp_pair_x18_begin[];
extern "C" const std::byte armigo_test_sp_pair_x18_end[];
extern "C" const std::uint32_t armigo_test_sp_pair_x18_offsets[8];
extern "C" const std::byte armigo_test_sp_far_pair_x18_begin[];
extern "C" const std::byte armigo_test_sp_far_pair_x18_end[];
extern "C" const std::uint32_t armigo_test_sp_far_pair_x18_offsets[4];

namespace {
using namespace armigo;
using namespace armigo::execution;
constexpr ScalarX18Policy contract{true, true, true};

std::span<const std::uint32_t> trampoline_words(const ScalarX18Patch& patch) {
    return {patch.trampoline.data(), patch.trampoline_word_count};
}

bool expect(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

bool rejected(std::uint32_t word, std::uint64_t address, std::uint64_t trampoline,
              std::uint32_t slot, ScalarX18Policy policy, ScalarX18PatchError expected) {
    const auto result = make_scalar_x18_patch(word, address, trampoline, slot, policy);
    if (!expect(!result.ok() && result.error == expected, "Incorrect x18 patch refusal.")) {
        std::fprintf(stderr, "word=%08x expected=%u actual=%u\n", word,
                     static_cast<unsigned>(expected),
                     result.ok() ? 0u : static_cast<unsigned>(result.error));
        return false;
    }
    return true;
}
bool test_adrp_boundaries() {
    constexpr std::uint64_t origin = 0x200000000;
    const auto positive = aarch64::encode_adrp(18, origin, origin + 0xfffff000);
    const auto negative = aarch64::encode_adrp(18, origin, origin - 0x100000000);
    if (!expect(positive.ok() && negative.ok(), "ADRP boundary fixture encoding failed.")) return false;
    return rejected(positive.word, origin, origin - 4096, 0, contract,
                    ScalarX18PatchError::substitution_failed) &&
           rejected(negative.word, origin, origin + 4096, 0, contract,
                    ScalarX18PatchError::substitution_failed) &&
           rejected(aarch64::encode_adrp(18, 4096, 0).word, 0, 8192, 0, contract,
                    ScalarX18PatchError::substitution_failed) &&
           rejected(aarch64::encode_adrp(18, 0, 4096).word,
                    std::numeric_limits<std::uint64_t>::max() - 4095,
                    std::numeric_limits<std::uint64_t>::max() - 16383, 0, contract,
                    ScalarX18PatchError::substitution_failed) &&
           rejected(aarch64::encode_adrp(17, 0, 4096).word, 0, 8192, 0, contract,
                    ScalarX18PatchError::register_not_used);
}

bool test_native_adrp(platform::ThreadSlotKey& key, GuestThreadContext& context) {
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_test_adrp_x18_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_test_adrp_x18_end);
    const auto site = armigo_test_adrp_x18_offset;
    if (!expect(end > begin && end - begin < 256 && site + 4 <= end - begin,
                "Invalid ADRP fixture extent.")) return false;
    // The executable ADRP slot is at trampoline+16, on the page AFTER the
    // trampoline's first instruction. Include both signs and sign reversals.
    struct Placement { std::size_t original, trampoline, target; };
    const std::array placements = {
        Placement{0x2000, 0x9ff0, 0xf000}, Placement{0x6000, 0x9ff0, 0x1000},
        Placement{0x2000, 0x9ff0, 0x6000}, Placement{0x6000, 0x1ff0, 0x4000}};
    for (const auto placement : placements) {
        std::vector<std::byte> code(0x10000);
        std::memcpy(code.data() + placement.original, armigo_test_adrp_x18_begin, end - begin);
        const std::array regions = {platform::ModuleRegion{
            0, code.size(), memory::Permission::read | memory::Permission::execute}};
        auto mapped = platform::HostModule::create(code, regions);
        if (!expect(mapped.ok(), "ADRP fixture mapping failed.")) return false;
        const auto base = mapped.module->base_address();
        const auto original = aarch64::encode_adrp(18, base + placement.original + site,
                                                 base + placement.target);
        if (!expect(original.ok(), "ADRP fixture target is not encodable.")) return false;
        auto bytes = mapped.module->preparation_bytes();
        std::memcpy(bytes.data() + placement.original + site, &original.word, 4);
        std::uint32_t preimage;
        std::memcpy(&preimage, bytes.data() + placement.original + site, 4);
        const auto result = make_scalar_x18_patch(preimage, base + placement.original + site,
            base + placement.trampoline, key.native_access()->tsd_byte_offset, contract);
        if (!expect(result.ok(), "An authored ADRP X18 site was rejected.")) return false;
        const auto& patch = *result.patch;
        const auto rewritten = aarch64::decode(patch.trampoline[4]);
        if (!expect(patch.expected_instruction == preimage &&
                        rewritten.opcode == aarch64::Opcode::adrp &&
                        rewritten.reg == patch.value_register &&
                        aarch64::referenced_address(rewritten, base + placement.trampoline + 16) ==
                            base + placement.target,
                    "ADRP preimage, destination or page target changed.")) return false;
        for (const auto word : trampoline_words(patch)) {
            if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(word), 18),
                        "ADRP trampoline touches physical x18 or has unknown effects.")) return false;
        }
        std::memcpy(bytes.data() + placement.original + site, &patch.replacement_instruction, 4);
        std::memcpy(bytes.data() + placement.trampoline, patch.trampoline.data(), 32);
        bytes = {};
        if (!expect(mapped.module->seal().ok(), "ADRP fixture sealing failed.")) return false;
        const auto entry = mapped.module->executable_address(placement.original);
        if (!expect(entry.has_value(), "ADRP fixture has no executable entry.")) return false;
        context.set_guest_x18(UINT64_MAX);
        std::array<std::uint64_t, 9> output{};
        reinterpret_cast<void (*)(std::uint64_t*)>(*entry)(output.data());
        if (!expect(context.guest_x18() == base + placement.target &&
                        output[0] == reinterpret_cast<std::uintptr_t>(output.data()) &&
                        output[1] == 0x3456 && output[2] == output[4] &&
                        output[3] == output[6] && output[5] == output[7] &&
                        output[8] == 0x60000000,
                    "ADRP changed its target, native registers, SP, LR, x18 or NZCV.")) return false;
    }
    return true;
}
bool test_sp_boundaries() {
    for (const auto opcode : {0x39000000u, 0x39400000u, 0x79000000u, 0x79400000u, 0xb9000000u,
                              0xb9400000u, 0xb9800000u, 0xf9000000u, 0xf9400000u}) {
        const unsigned size = opcode >> 30;
        const unsigned scale = size == 3 ? 8 : (size == 0 ? 1 : (size == 1 ? 2 : 4));
        const unsigned last = 4095 - 16 / scale;
        const auto word = opcode | (last << 10) | (31u << 5) | 18;
        const auto result = make_scalar_x18_patch(word, 0, 4096, 0, contract);
        if (!expect(result.ok() &&
                        aarch64::decode_register_effects(result.patch->trampoline[4]).memory.offset ==
                            4095 * scale,
                    "SP offset boundary did not compensate exactly 16 bytes.")) return false;
        if (!rejected(word + (1u << 10), 0, 4096, 0, contract,
                      ScalarX18PatchError::substitution_failed)) return false;
    }
    // Signed pair offsets keep their scaled imm7 and move by two elements; the
    // lowest imm7 stays in range after the move.
    for (const auto opcode : {0x29000000u, 0x29400000u, 0xa9000000u, 0xa9400000u, 0x69400000u}) {
        const unsigned scale = (opcode >> 31) ? 8 : 4;
        for (const unsigned immediate : {63u - 16 / scale, 64u}) {  // 64: imm7 -64
            const auto word = opcode | ((immediate & 0x7fu) << 15) | (10u << 10) | (31u << 5) | 18;
            const auto result = make_scalar_x18_patch(word, 0, 4096, 0, contract);
            if (!expect(result.ok() && result.patch->trampoline_word_count == 8,
                        "An in-range pair offset was rejected.")) return false;
            const auto after = aarch64::decode_register_effects(result.patch->trampoline[4]);
            const std::int64_t expected = immediate == 64u ? -64 * std::int64_t{scale} + 16 : 63 * std::int64_t{scale};
            if (!expect(after.memory.offset == expected && after.memory.elements == 2 &&
                            after.memory.addressing == aarch64::AddressingMode::signed_offset,
                        "Pair offset compensation changed the access.")) return false;
        }
    }
    // Past the top of imm7 (0x1f0 and 0x1f8 for X pairs, 0xf0 and 0xfc for W
    // pairs and LDPSW) a pair becomes two single-register transfers at the same
    // addresses, each register as the pair moved it; x18 only in its own half.
    for (const auto opcode : {0x29000000u, 0x29400000u, 0xa9000000u, 0xa9400000u, 0x69400000u, 0xa8000000u,
                              0xa8400000u, 0x28000000u, 0x28400000u}) {
        const unsigned scale = (opcode >> 31) ? 8 : 4;
        for (const unsigned immediate : {64u - 16 / scale, 63u})
            for (const bool second : {false, true}) {
                const unsigned rt = second ? 10 : 18, rt2 = second ? 18 : 10;
                const auto word = opcode | (immediate << 15) | (rt2 << 10) | (31u << 5) | rt;
                const auto result = make_scalar_x18_patch(word, 0, 4096, 0, contract);
                if (!expect(result.ok() && result.patch->trampoline_word_count == 9,
                            "A pair past the top of imm7 was not split.")) return false;
                const auto& patch = *result.patch;
                const auto original = aarch64::decode_register_effects(word);
                for (unsigned half = 0; half != 2; ++half) {
                    const auto access = aarch64::decode_register_effects(patch.trampoline[4 + half]);
                    const unsigned reg = (half == 0) == !second ? patch.value_register : 10;
                    const auto bit = aarch64::register_bit(reg);
                    if (!expect(access.memory.elements == 1 &&
                                    access.memory.element_bytes == original.memory.element_bytes &&
                                    access.memory.operation == original.memory.operation &&
                                    access.memory.offset == immediate * scale + 16 + half * scale &&
                                    access.memory.addressing == aarch64::AddressingMode::unsigned_offset &&
                                    !access.memory.writeback &&
                                    has(access.memory.attributes, aarch64::MemoryAttribute::sign_extend) ==
                                        has(original.memory.attributes, aarch64::MemoryAttribute::sign_extend) &&
                                    ((access.reads | access.writes) & bit) != 0 &&
                                    aarch64::proven_free_of(access, 18),
                                "A split pair half transfers the wrong register, width or address.")) return false;
                }
            }
    }
    // A pre-index pair past the top splits too; its epilogue still applies the
    // guest's SP update.
    {
        const auto result = make_scalar_x18_patch(0xa99f2bf2u, 0, 4096, 0, contract); // STP X18,X10,[SP,#0x1f0]!
        if (!expect(result.ok() && result.patch->trampoline_word_count == 10,
                    "A pre-index pair past the top of imm7 was not split.")) return false;
        const auto& patch = *result.patch;
        const auto first = aarch64::decode_register_effects(patch.trampoline[4]);
        const auto second = aarch64::decode_register_effects(patch.trampoline[5]);
        const auto restore = aarch64::decode_register_effects(patch.trampoline[7]);
        // ADD SP, SP, #0x200: the spill's 16 bytes plus the guest's 0x1f0.
        if (!expect(first.memory.offset == 0x200 && first.memory.elements == 1 && !first.memory.writeback &&
                        second.memory.offset == 0x208 && second.memory.elements == 1 && !second.memory.writeback &&
                        ((second.reads & aarch64::register_bit(10)) != 0) &&
                        restore.operation == aarch64::Operation::ldp && !restore.memory.writeback &&
                        patch.trampoline[8] == 0x910803ffu,
                    "A pre-index pair past the top of imm7 lost its halves or its exact SP update.")) return false;
    }
    // A load into one register twice (CONSTRAINED UNPREDICTABLE) is refused.
    if (!rejected(0xa95fcbf2u, 0, 4096, 0, contract,  // LDP X18,X18,[SP,#0x1f8]
                  ScalarX18PatchError::unsupported_instruction)) return false;
    // An unscaled offset moves the access but not SP, so the plain epilogue
    // still restores the spilled frame.
    const auto unscaled = make_scalar_x18_patch(0xf84043f2u, 0, 4096, 0, contract); // LDUR X18,[SP,#4]
    if (!expect(unscaled.ok() && unscaled.patch->trampoline_word_count == 8 &&
                    aarch64::decode_register_effects(unscaled.patch->trampoline[4]).memory.offset == 20,
                "An unscaled SP access was not compensated exactly.")) return false;
    // Pre-index writeback moves SP by the access offset, which is not a
    // multiple of the scratch pair's width: the epilogue updates SP itself.
    const auto writeback = make_scalar_x18_patch(0xf8405ff2u, 0, 4096, 0, contract); // LDR X18,[SP,#5]!
    if (!expect(writeback.ok() && writeback.patch->trampoline_word_count == 9,
                "A pre-index SP access did not select the long epilogue.")) return false;
    {
        const auto& patch = *writeback.patch;
        const auto access = aarch64::decode_register_effects(patch.trampoline[4]);
        const auto restore = aarch64::decode_register_effects(patch.trampoline[6]);
        const auto adjust = aarch64::decode_register_effects(patch.trampoline[7]);
        if (!expect(access.memory.offset == 21 && access.memory.addressing ==
                        aarch64::AddressingMode::unscaled_offset &&
                        restore.operation == aarch64::Operation::ldp &&
                        restore.memory.offset == 0 && !restore.memory.writeback &&
                        adjust.operation == aarch64::Operation::add_immediate &&
                        aarch64::writes_stack_pointer(adjust) &&
                        aarch64::reads_stack_pointer(adjust),
                    "The long epilogue did not restore scratch registers and SP exactly.")) return false;
        for (const auto emitted : trampoline_words(patch)) {
            if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(emitted), 18),
                        "SP trampoline touches physical x18 or has unknown effects.")) return false;
        }
    }
    return rejected(0xf85f83f2, 0, 4096, 0, contract, // LDUR X18, [SP, #-8]
                    ScalarX18PatchError::stack_pointer_operand) &&
           rejected(0xf81f8ff2, 0, 4096, 0, contract, // STR X18, [SP, #-8]!
                    ScalarX18PatchError::stack_pointer_operand) &&
           rejected(0xa9bf27f2, 0, 4096, 0, contract, // STP X18, X9, [SP, #-16]!
                    ScalarX18PatchError::stack_pointer_operand) &&
           rejected(0xf8726bf2, 0, 4096, 0, contract, // LDR X18, [SP, X18]
                    ScalarX18PatchError::stack_pointer_operand) &&
           rejected(0x9100025f, 0, 4096, 0, contract, // ADD SP, X18, #0
                    ScalarX18PatchError::stack_pointer_operand);
}

bool test_fmov_patch() {
    // FMOV between a general register and a SIMD&FP register is prepared by
    // substituting the general-register field; the scalar FP register is not
    // part of the general register mask.
    auto result = make_scalar_x18_patch(0x1E2600B2u, 0, 4096, 0, contract); // FMOV W18, S5
    if (!expect(result.ok() && result.patch->trampoline_word_count == 8,
                "FMOV W18,S5 was not prepared.")) return false;
    auto access = aarch64::decode_register_effects(result.patch->trampoline[4]);
    if (!expect(access.operation == aarch64::Operation::fmov_general &&
                    access.operands[1].number == result.patch->value_register,
                "FMOV W18,S5 did not substitute the general-register field.")) return false;
    for (const auto emitted : trampoline_words(*result.patch)) {
        if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(emitted), 18),
                    "FMOV trampoline touches physical x18.")) return false;
    }
    result = make_scalar_x18_patch(0x1E270245u, 0, 4096, 0, contract); // FMOV S5, W18
    if (!expect(result.ok() && result.patch->trampoline_word_count == 8,
                "FMOV S5,W18 was not prepared.")) return false;
    access = aarch64::decode_register_effects(result.patch->trampoline[4]);
    if (!expect(access.operation == aarch64::Operation::fmov_general &&
                    access.operands[0].number == result.patch->value_register,
                "FMOV S5,W18 did not substitute the general-register field.")) return false;
    // The SDK's float-to-int conversions write general registers too; FCVTMS
    // with a virtual-x18 destination must prepare the same way.
    result = make_scalar_x18_patch(0x1E300052u, 0, 4096, 0, contract); // FCVTMS W18, S2
    if (!expect(result.ok() && result.patch->trampoline_word_count == 8,
                "FCVTMS W18,S2 was not prepared.")) return false;
    access = aarch64::decode_register_effects(result.patch->trampoline[4]);
    return expect(access.operation == aarch64::Operation::fcvt_general &&
                      access.operands[1].number == result.patch->value_register,
                  "FCVTMS W18,S2 did not substitute the general-register field.");
}

bool test_native_adr(platform::ThreadSlotKey& key, GuestThreadContext& context) {
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_test_adr_x18_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_test_adr_x18_end);
    const auto site = armigo_test_adr_x18_offsets[1];
    const auto target = armigo_test_adr_x18_target;
    if (!expect(end > begin && end - begin < 256 && site + 4 <= end - begin &&
                    target <= end - begin,
                "Invalid ADR fixture extent.")) return false;
    std::vector<std::byte> code(4096);
    std::memcpy(code.data(), armigo_test_adr_x18_begin, end - begin);
    for (unsigned index = 0; index != 3; ++index) {
        const auto offset = armigo_test_adr_x18_offsets[index];
        std::uint32_t original;
        std::memcpy(&original, code.data() + offset, 4);
        const auto result = make_scalar_x18_patch(original, offset, 2048 + 48 * index,
                                                  key.native_access()->tsd_byte_offset, contract);
        if (!expect(result.ok(), "An authored ADR fixture site was rejected.")) return false;
        const auto& patch = *result.patch;
        if (index == 1) {
            // ADR cannot reach an appended trampoline, so the absolute target
            // is materialized with MOVZ/MOVK and must be exact.
            std::uint64_t materialized = 0;
            for (unsigned half = 0; half != 4; ++half) {
                const auto effects = aarch64::decode_register_effects(patch.trampoline[4 + half]);
                if (!expect(effects.operation == (half == 0 ? aarch64::Operation::movz
                                                            : aarch64::Operation::movk) &&
                                effects.writes == aarch64::register_bit(patch.value_register) &&
                                !effects.reads_flags && !effects.writes_flags,
                            "ADR materialization is not a plain MOVZ/MOVK to the value register.")) {
                    return false;
                }
                materialized |= static_cast<std::uint64_t>((patch.trampoline[4 + half] >> 5) & 0xFFFFu)
                                << (16 * half);
            }
            if (!expect(patch.expected_instruction == original && materialized == target,
                        "ADR byte target changed.")) return false;
        }
        for (const auto word : trampoline_words(patch)) {
            if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(word), 18),
                        "ADR trampoline touches physical x18 or has unknown effects.")) return false;
        }
        std::memcpy(code.data() + offset, &patch.replacement_instruction, 4);
        std::memcpy(code.data() + patch.trampoline_address, patch.trampoline.data(),
                    patch.trampoline_word_count * 4);
    }
    const std::array regions = {platform::ModuleRegion{
        0, code.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(code, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "ADR fixture mapping failed.")) return false;
    const auto entry = mapped.module->executable_address(0);
    if (!expect(entry.has_value(), "ADR fixture has no entry.")) return false;
    const std::uint64_t expected = target;
    context.set_guest_x18(UINT64_MAX);
    std::array<std::uint64_t, 5> output{};
    reinterpret_cast<void (*)(std::uint64_t*, std::uint64_t)>(*entry)(output.data(), 0);
    if (!expect(context.guest_x18() == expected && output[0] == expected &&
                    output[1] == UINT64_MAX && output[2] == output[4] &&
                    output[3] == 0x60000000,
                "ADR changed its byte target, x18, x30 or NZCV.")) return false;
    return true;
}

bool test_native_sp(platform::ThreadSlotKey& key, GuestThreadContext& context) {
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_test_sp_x18_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_test_sp_x18_end);
    if (!expect(end > begin && end - begin < 1024, "Invalid SP fixture extent.")) return false;
    std::vector<std::byte> code(4096);
    std::memcpy(code.data(), armigo_test_sp_x18_begin, end - begin);
    for (unsigned i = 0; i != 6; ++i) {
        const auto offset = armigo_test_sp_x18_offsets[i];
        if (!expect(offset % 4 == 0 && offset + 4 <= end - begin,
                    "Invalid SP fixture patch offset.")) return false;
        std::uint32_t original;
        std::memcpy(&original, code.data() + offset, 4);
        const auto result = make_scalar_x18_patch(original, offset, 1024 + 48 * i,
                                                  key.native_access()->tsd_byte_offset, contract);
        if (!expect(result.ok(), "An unsigned SP x18 access was rejected.")) return false;
        const auto& patch = *result.patch;
        const auto before = aarch64::decode_register_effects(original);
        const auto after = aarch64::decode_register_effects(patch.trampoline[4]);
        if (!expect(patch.expected_instruction == original && after.memory.offset == before.memory.offset + 16 &&
                        after.memory.element_bytes == before.memory.element_bytes &&
                        after.memory.operation == before.memory.operation,
                    "SP access preimage, width or address compensation changed.")) return false;
        for (const auto word : trampoline_words(patch)) {
            if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(word), 18),
                        "SP trampoline touches physical x18 or has unknown effects.")) return false;
        }
        std::memcpy(code.data() + offset, &patch.replacement_instruction, 4);
        std::memcpy(code.data() + patch.trampoline_address, patch.trampoline.data(),
                    patch.trampoline_word_count * 4);
    }
    const std::array regions = {platform::ModuleRegion{
        0, code.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(code, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "SP fixture mapping failed.")) return false;
    const auto entry = mapped.module->executable_address(0);
    if (!expect(entry.has_value(), "SP fixture has no entry.")) return false;
    for (const std::uint64_t input : {0xfedcba9876543210ull, 0x0000000180000001ull}) {
        context.set_guest_x18(UINT64_MAX);
        std::array<std::uint64_t, 12> output{};
        reinterpret_cast<void (*)(std::uint64_t*, std::uint64_t)>(*entry)(output.data(), input);
        const auto low = static_cast<std::uint32_t>(input);
        if (!expect(context.guest_x18() == low &&
                        output[0] == reinterpret_cast<std::uintptr_t>(output.data()) &&
                        output[1] == input && output[2] == (std::uint64_t{low} << 32 | 0x50607080) &&
                        output[3] == input && output[4] == low && output[5] == 0x60000000 &&
                        output[6] == output[7] && output[8] == output[9] && output[10] == output[11],
                    "SP x18 accesses changed stack data, registers, flags or write width.")) return false;
    }
    return true;
}
// Runs the split pairs: stored values read back natively, loaded values in
// their registers (LDPSW sign-extended), guest x18 and SP as without the
// trampolines.
bool test_native_sp_far_pair(platform::ThreadSlotKey& key, GuestThreadContext& context) {
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_test_sp_far_pair_x18_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_test_sp_far_pair_x18_end);
    if (!expect(end > begin && end - begin < 1024, "Invalid far SP pair fixture extent.")) return false;
    std::vector<std::byte> code(4096);
    std::memcpy(code.data(), armigo_test_sp_far_pair_x18_begin, end - begin);
    for (unsigned i = 0; i != 4; ++i) {
        const auto offset = armigo_test_sp_far_pair_x18_offsets[i];
        if (!expect(offset % 4 == 0 && offset + 4 <= end - begin, "Invalid far SP pair fixture offset.")) return false;
        std::uint32_t original;
        std::memcpy(&original, code.data() + offset, 4);
        const auto result = make_scalar_x18_patch(original, offset, 2048 + 48 * i,
                                                  key.native_access()->tsd_byte_offset, contract);
        if (!expect(result.ok() && result.patch->trampoline_word_count == 9, "A far stack pair was not split."))
            return false;
        const auto& patch = *result.patch;
        std::memcpy(code.data() + offset, &patch.replacement_instruction, 4);
        std::memcpy(code.data() + patch.trampoline_address, patch.trampoline.data(), patch.trampoline_word_count * 4);
    }
    const std::array regions = {platform::ModuleRegion{
        0, code.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(code, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "Far SP pair fixture mapping failed.")) return false;
    const auto entry = mapped.module->executable_address(0);
    if (!expect(entry.has_value(), "Far SP pair fixture has no entry.")) return false;
    constexpr std::uint64_t guest = 0x0123456789abcdefull, x5 = 0x8000000180000002ull;
    context.set_guest_x18(guest);
    std::array<std::uint64_t, 8> output{};
    reinterpret_cast<void (*)(std::uint64_t*, std::uint64_t)>(*entry)(output.data(), 1);
    return expect(output[0] == x5 && output[1] == guest && output[2] == guest && output[3] == 0x80000002u &&
                      output[4] == 0x7ffffffeu && output[5] == 0xffffffff80000002ull && output[6] == output[7] &&
                      context.guest_x18() == 0x7ffffffeu,
                  "Split stack pairs changed data, sign extension, SP or guest x18.");
}

bool test_native_sp_pair(platform::ThreadSlotKey& key, GuestThreadContext& context) {
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_test_sp_pair_x18_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_test_sp_pair_x18_end);
    if (!expect(end > begin && end - begin < 1024, "Invalid SP pair fixture extent.")) return false;
    std::vector<std::byte> code(4096);
    std::memcpy(code.data(), armigo_test_sp_pair_x18_begin, end - begin);
    for (unsigned i = 0; i != 8; ++i) {
        const auto offset = armigo_test_sp_pair_x18_offsets[i];
        if (!expect(offset % 4 == 0 && offset + 4 <= end - begin,
                    "Invalid SP pair fixture patch offset.")) return false;
        std::uint32_t original;
        std::memcpy(&original, code.data() + offset, 4);
        const auto result = make_scalar_x18_patch(original, offset, 2048 + 48 * i,
                                                  key.native_access()->tsd_byte_offset, contract);
        if (!expect(result.ok(), "A stack pair or unscaled x18 access was rejected.")) return false;
        const auto& patch = *result.patch;
        const auto before = aarch64::decode_register_effects(original);
        const auto after = aarch64::decode_register_effects(patch.trampoline[4]);
        const auto writeback = before.memory.addressing == aarch64::AddressingMode::pre_index ||
                               before.memory.addressing == aarch64::AddressingMode::post_index;
        if (!expect(patch.expected_instruction == original &&
                        after.memory.offset ==
                            (before.memory.addressing == aarch64::AddressingMode::post_index
                                 ? 16
                                 : before.memory.offset + 16) &&
                        after.memory.element_bytes == before.memory.element_bytes &&
                        after.memory.elements == before.memory.elements && !after.memory.writeback &&
                        (!writeback || after.memory.addressing != before.memory.addressing),
                    "Stack access preimage, width or address compensation changed.")) return false;
        for (const auto word : trampoline_words(patch)) {
            if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(word), 18),
                        "Stack pair trampoline touches physical x18 or has unknown effects.")) return false;
        }
        std::memcpy(code.data() + offset, &patch.replacement_instruction, 4);
        std::memcpy(code.data() + patch.trampoline_address, patch.trampoline.data(),
                    patch.trampoline_word_count * 4);
    }
    const std::array regions = {platform::ModuleRegion{
        0, code.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(code, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "SP pair fixture mapping failed.")) return false;
    const auto entry = mapped.module->executable_address(0);
    if (!expect(entry.has_value(), "SP pair fixture has no entry.")) return false;
    context.set_guest_x18(UINT64_MAX);
    std::array<std::uint64_t, 10> output{};
    reinterpret_cast<void (*)(std::uint64_t*, std::uint64_t)>(*entry)(output.data(), 1);
    if (!expect(output[0] == 0x34 && output[1] == 0 && output[2] == 0x34 && output[3] == 0 &&
                    output[4] == output[5] && output[6] == 0x60000000 && output[7] == output[9] &&
                    context.guest_x18() == 0,
                "Stack pair and unscaled x18 accesses changed data, SP, flags or x30.")) return false;
    return true;
}
} // namespace

int main() {
    if (!test_adrp_boundaries() || !test_sp_boundaries() || !test_fmov_patch()) return 1;
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_test_scalar_x18_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_test_scalar_x18_end);
    if (!expect(end > begin && end - begin < 1024, "Invalid authored fixture extent.")) return 1;
    std::vector<std::byte> code(4096);
    std::memcpy(code.data(), armigo_test_scalar_x18_begin, end - begin);
    std::uint32_t first;
    std::memcpy(&first, code.data() + armigo_test_scalar_x18_offsets[0], sizeof(first));
    if (!rejected(first, 1, 4096, 0, contract, ScalarX18PatchError::invalid_layout) ||
        !rejected(first, 0, 4, 0, contract, ScalarX18PatchError::overlapping_regions) ||
        !rejected(first, 0, 4096, 3, contract, ScalarX18PatchError::invalid_thread_slot) ||
        !rejected(first, 0, 4096, 32768, contract, ScalarX18PatchError::invalid_thread_slot) ||
        !rejected(first, 0, 4096, 0, {}, ScalarX18PatchError::missing_execution_contract) ||
        !rejected(first, 0, 4096, 0, {true, false, true},
                  ScalarX18PatchError::missing_execution_contract) ||
        !rejected(first, 0, 4096, 0, {false, true, true},
                  ScalarX18PatchError::missing_execution_contract) ||
        !rejected(first, 0, 4096, 0, {true, true, false},
                  ScalarX18PatchError::missing_execution_contract) ||
        !rejected(first, 0, 1ull << 28, 0, contract, ScalarX18PatchError::unreachable_branch) ||
        // Entry B reaches its positive limit, but the final B cannot return.
        !rejected(first, 0x10000000, 0x17fffffc, 0, contract,
                  ScalarX18PatchError::unreachable_branch) ||
        !rejected(first, 0, std::numeric_limits<std::uint64_t>::max() - 3, 0, contract,
                  ScalarX18PatchError::invalid_layout) ||
        !rejected(0xffffffff, 0, 4096, 0, contract, ScalarX18PatchError::unsupported_instruction) ||
        !rejected(aarch64::encode_svc(1).word, 0, 4096, 0, contract,
                  ScalarX18PatchError::unsupported_instruction) ||
        !rejected(aarch64::encode_br(18).word, 0, 4096, 0, contract,
                  ScalarX18PatchError::unsupported_instruction) ||

        !rejected(0xf81f83f2, 0, 4096, 0, contract, // STUR X18, [SP, #-8]
                  ScalarX18PatchError::stack_pointer_operand) ||
        !rejected(0x9100025f, 0, 4096, 0, contract, // ADD SP, X18, #0
                  ScalarX18PatchError::stack_pointer_operand) ||
        !rejected(0xc85f7c12, 0, 4096, 0, contract, // LDXR X18, [X0]
                  ScalarX18PatchError::exclusive_instruction) ||
        !rejected(0xd503201f, 0, 4096, 0, contract, // NOP
                  ScalarX18PatchError::register_not_used)) return 1;
    if (!expect(make_scalar_x18_patch(first, 0x10000000, 0x08000000, 32760, contract).ok(),
                "Valid branch and slot encoding boundaries were rejected.")) return 1;

    platform::ThreadSlotKey key;
    if (!expect(key.valid(), "Native thread slot probe failed.")) return 1;
    const auto slot = key.native_access()->tsd_byte_offset;
    for (std::size_t index = 0; index < 10; ++index) {
        const auto offset = armigo_test_scalar_x18_offsets[index];
        if (!expect(offset % 4 == 0 && offset + 4 <= end - begin,
                    "Invalid authored patch offset.")) return 1;
        std::uint32_t word;
        std::memcpy(&word, code.data() + offset, sizeof(word));
        const auto planned = make_scalar_x18_patch(word, offset, 1024 + index * 32, slot, contract);
        if (!expect(planned.ok(), "An authored scalar instruction was rejected.")) return 1;
        const auto& patch = *planned.patch;
        if (!expect(patch.expected_instruction == word && patch.value_register != patch.cell_register,
                    "Invalid patch identity or scratch selection.")) return 1;
        for (const auto emitted : trampoline_words(patch)) {
            if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(emitted), 18),
                        "Generated trampoline uses hardware x18 or has unknown effects.")) return 1;
        }
        std::memcpy(code.data() + offset, &patch.replacement_instruction, 4);
        std::memcpy(code.data() + patch.trampoline_address, patch.trampoline.data(),
                    patch.trampoline_word_count * 4);
    }
    // All offsets remain identical after the anonymous mapping is placed.
    const std::array regions = {platform::ModuleRegion{
        0, code.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(code, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "Prepared fixture mapping failed."))
        return 1;
    const auto entry = mapped.module->executable_address(0);
    if (!expect(entry.has_value(), "Prepared fixture has no executable entry.")) return 1;
    const auto table = GuestSiteTable::build({});
    if (!expect(table.ok(), "An empty dispatch table was rejected.")) return 1;
    GuestThreadContext context;
    context.set_guest_x18(UINT64_MAX);
    GuestDispatchScope dispatch(table.table, context);
    if (!expect(dispatch.setup_error() == GuestDispatchFailureKind::none,
                "Guest context binding failed.")) return 1;
    platform::ThreadSlotBinding binding(key, context.bound_register_cell());
    if (!expect(binding.native_access().has_value(), "Guest register publication failed.")) return 1;
    using Entry = std::uint64_t (*)(std::uint64_t*, std::uint64_t);
    const auto run = reinterpret_cast<Entry>(*entry);
    std::array<std::uint64_t, 6> output{};
    const auto result = run(output.data(), 3);
    if (!expect(result == 4 && context.guest_x18() == 4 &&
                    output == std::array<std::uint64_t, 6>{8, 3, 4, 0x1234, 0x5678, 0},
                "Prepared scalar x18 code changed data, scratch registers, or flags.")) return 1;
    if (!test_native_adrp(key, context) || !test_native_adr(key, context) ||
        !test_native_sp(key, context) || !test_native_sp_pair(key, context) ||
        !test_native_sp_far_pair(key, context)) return 1;
    std::puts("Scalar, ADRP and unsigned SP accesses executed through native x18 trampolines.");
    return 0;
}
