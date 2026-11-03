// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Uses the system assembler and linker as an independent oracle: the words in
// assembler_fixture.S were encoded by the toolchain from mnemonics and labels,
// and the address table was resolved by the linker from the same labels.

#include <armigo/aarch64/instruction.h>

#include "test_support.h"

#include <cstdint>
#include <cstring>

extern "C" {
extern const std::uint32_t armigo_aarch64_fixture_code[];
extern const std::uint64_t armigo_aarch64_fixture_table[];
extern const std::uint64_t armigo_aarch64_fixture_table_end[];
extern const std::uint64_t armigo_aarch64_fixture_endpoints[];
extern const std::uint64_t armigo_aarch64_fixture_endpoints_end[];
extern const std::uint64_t armigo_aarch64_fixture_data[];
}

namespace {

using namespace armigo::aarch64;

struct Expectation {
    Opcode opcode;
    std::uint8_t reg;
    SystemRegister system_register = SystemRegister::other;
};

// Mirrors the order of L0..L32 in the fixture.
constexpr Expectation expected[] = {
    {Opcode::b, 0},
    {Opcode::bl, 0},
    {Opcode::b_cond, 0},
    {Opcode::bc_cond, 0},
    {Opcode::cbz, 18},
    {Opcode::cbnz, 3},
    {Opcode::tbz, 7},
    {Opcode::tbnz, 18},
    {Opcode::adr, 18},
    {Opcode::adr, 1},
    {Opcode::adrp, 2},
    {Opcode::ldr_literal_w, 1},
    {Opcode::ldr_literal_x, 2},
    {Opcode::ldrsw_literal, 18},
    {Opcode::prfm_literal, 0},
    {Opcode::ldr_literal_s, 4},
    {Opcode::ldr_literal_d, 5},
    {Opcode::ldr_literal_q, 6},
    {Opcode::br, 18},
    {Opcode::blr, 9},
    {Opcode::ret, 30},
    {Opcode::svc, 0},
    {Opcode::brk, 0},
    {Opcode::mrs, 18, SystemRegister::tpidr_el0},
    {Opcode::msr_register, 1, SystemRegister::tpidr_el0},
    {Opcode::mrs, 2, SystemRegister::tpidrro_el0},
    {Opcode::mrs, 3, SystemRegister::cntpct_el0},
    {Opcode::mrs, 4, SystemRegister::cntvct_el0},
    {Opcode::mrs, 5, SystemRegister::cntfrq_el0},
    {Opcode::mrs, 6, SystemRegister::cntvctss_el0},
    {Opcode::mrs, 7, SystemRegister::nzcv},
    {Opcode::msr_register, 8, SystemRegister::fpcr},
    {Opcode::mrs, 9, SystemRegister::ctr_el0},
};

struct Endpoint {
    Opcode opcode;
    std::int64_t displacement;
};

// Mirrors E0..E9.
constexpr Endpoint endpoints[] = {
    {Opcode::b, 0x7fffffc},
    {Opcode::bl, -0x8000000},
    {Opcode::b_cond, 0xffffc},
    {Opcode::cbz, -0x100000},
    {Opcode::tbnz, 0x7ffc},
    {Opcode::tbz, -0x8000},
    {Opcode::adr, 0xfffff},
    {Opcode::adr, -0x100000},
    {Opcode::ldr_literal_x, 0xffffc},
    {Opcode::ldr_literal_q, -0x100000},
};

std::uint32_t word_at(std::uint64_t address) {
    std::uint32_t word = 0;
    std::memcpy(&word, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address)),
                sizeof(word));
    return word;
}

void labeled_instructions() {
    const std::size_t entries =
        static_cast<std::size_t>(armigo_aarch64_fixture_table_end - armigo_aarch64_fixture_table) / 2;
    CHECK_EQ(entries, std::size(expected));
    CHECK_EQ(armigo_aarch64_fixture_table[0],
             reinterpret_cast<std::uintptr_t>(armigo_aarch64_fixture_code) + 4);
    for (std::size_t index = 0; index < entries && index < std::size(expected); ++index) {
        const std::uint64_t pc = armigo_aarch64_fixture_table[index * 2];
        const std::uint64_t reference = armigo_aarch64_fixture_table[index * 2 + 1];
        const auto instruction = decode(word_at(pc));
        const auto& expect = expected[index];
        if (!CHECK(instruction.opcode == expect.opcode)) {
            std::fprintf(stderr, "  entry %zu word 0x%08x\n", index, instruction.word);
            continue;
        }
        if (expect.opcode != Opcode::prfm_literal && expect.opcode != Opcode::svc &&
            expect.opcode != Opcode::brk && expect.opcode != Opcode::b &&
            expect.opcode != Opcode::bl && expect.opcode != Opcode::b_cond &&
            expect.opcode != Opcode::bc_cond) {
            CHECK_EQ(instruction.reg, expect.reg);
        }
        if (is_pc_relative(expect.opcode)) {
            const auto address = referenced_address(instruction, pc);
            CHECK(address.has_value());
            const std::uint64_t want =
                expect.opcode == Opcode::adrp ? (reference & ~std::uint64_t{0xFFF}) : reference;
            CHECK_EQ(address.value_or(0), want);
        } else {
            CHECK(!referenced_address(instruction, pc));
        }
        if (expect.opcode == Opcode::mrs || expect.opcode == Opcode::msr_register) {
            CHECK(identify_system_register(instruction.system_register) == expect.system_register);
        }
    }
    // The ADRP fixture symbol is deliberately not page-aligned.
    CHECK((reinterpret_cast<std::uintptr_t>(armigo_aarch64_fixture_data) & 0xFFF) != 0);

    const auto tbz = decode(word_at(armigo_aarch64_fixture_table[6 * 2]));
    CHECK_EQ(tbz.test_bit, 31);
    CHECK(!tbz.wide);
    const auto tbnz = decode(word_at(armigo_aarch64_fixture_table[7 * 2]));
    CHECK_EQ(tbnz.test_bit, 63);
    CHECK(tbnz.wide);
    CHECK(decode(word_at(armigo_aarch64_fixture_table[2 * 2])).condition == Condition::hi);
    CHECK(decode(word_at(armigo_aarch64_fixture_table[3 * 2])).condition == Condition::eq);
    CHECK_EQ(decode(word_at(armigo_aarch64_fixture_table[21 * 2])).immediate, 0x7f);
    CHECK_EQ(decode(word_at(armigo_aarch64_fixture_table[22 * 2])).immediate, 0x18);
}

void encoders_match_assembler() {
    // Re-encoding each assembled PC-relative instruction from its decoded
    // fields at its own address must reproduce the assembler's word.
    const std::size_t entries =
        static_cast<std::size_t>(armigo_aarch64_fixture_table_end - armigo_aarch64_fixture_table) / 2;
    for (std::size_t index = 0; index < entries; ++index) {
        const std::uint64_t pc = armigo_aarch64_fixture_table[index * 2];
        const auto word = word_at(pc);
        const auto i = decode(word);
        const auto target = referenced_address(i, pc);
        Encoded e;
        switch (i.opcode) {
        case Opcode::b: e = encode_b(pc, *target); break;
        case Opcode::bl: e = encode_bl(pc, *target); break;
        case Opcode::b_cond: e = encode_b_cond(i.condition, pc, *target); break;
        case Opcode::bc_cond: e = encode_b_cond(i.condition, pc, *target, true); break;
        case Opcode::cbz:
        case Opcode::cbnz: e = encode_cbz(i.opcode == Opcode::cbnz, i.wide, i.reg, pc, *target); break;
        case Opcode::tbz:
        case Opcode::tbnz: e = encode_tbz(i.opcode == Opcode::tbnz, i.reg, i.test_bit, pc, *target); break;
        case Opcode::adr: e = encode_adr(i.reg, pc, *target); break;
        case Opcode::adrp: e = encode_adrp(i.reg, pc, *target); break;
        case Opcode::prfm_literal: e = encode_load_literal(i.opcode, i.prefetch_operation, pc, *target); break;
        case Opcode::ldr_literal_w:
        case Opcode::ldr_literal_x:
        case Opcode::ldrsw_literal:
        case Opcode::ldr_literal_s:
        case Opcode::ldr_literal_d:
        case Opcode::ldr_literal_q: e = encode_load_literal(i.opcode, i.reg, pc, *target); break;
        case Opcode::br: e = encode_br(i.reg); break;
        case Opcode::blr: e = encode_blr(i.reg); break;
        case Opcode::ret: e = encode_ret(i.reg); break;
        case Opcode::svc: e = encode_svc(i.immediate); break;
        case Opcode::brk: e = encode_brk(i.immediate); break;
        case Opcode::mrs: e = encode_mrs(i.reg, i.system_register); break;
        case Opcode::msr_register: e = encode_msr(i.system_register, i.reg); break;
        default: e.error = EncodeError::invalid_operand; break;
        }
        CHECK(e.ok());
        CHECK_EQ(e.word, word);
    }
}

void assembled_endpoints() {
    const std::size_t entries = static_cast<std::size_t>(armigo_aarch64_fixture_endpoints_end -
                                                         armigo_aarch64_fixture_endpoints);
    CHECK_EQ(entries, std::size(endpoints));
    for (std::size_t index = 0; index < entries && index < std::size(endpoints); ++index) {
        const auto i = decode(word_at(armigo_aarch64_fixture_endpoints[index]));
        CHECK(i.opcode == endpoints[index].opcode);
        CHECK_EQ(i.displacement, endpoints[index].displacement);
        const auto range = *displacement_range(i.opcode);
        CHECK(i.displacement == range.minimum || i.displacement == range.maximum);
    }
}

} // namespace

int main() {
    labeled_instructions();
    encoders_match_assembler();
    assembled_endpoints();
    return armigo::aarch64::test::finish("aarch64.assembler_fixture");
}
