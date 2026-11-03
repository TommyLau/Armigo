// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Decoder and encoder checks against hand-derived instruction words. Expected
// words were assembled by hand from the field layouts in the Arm A64
// encoding tables, independently of the implementation's formulas.

#include <armigo/aarch64/instruction.h>

#include "test_support.h"

#include <cstdint>
#include <limits>

namespace {

using namespace armigo::aarch64;

constexpr std::uint64_t x18 = RegisterMask{1} << 18;
constexpr std::uint64_t x30 = RegisterMask{1} << 30;

void immediate_branches() {
    auto i = decode(0x14000000u);  // b .
    CHECK(i.opcode == Opcode::b);
    CHECK_EQ(i.displacement, 0);
    CHECK_EQ(i.writes, 0);

    i = decode(0x15FFFFFFu);  // b .+0x7fffffc
    CHECK(i.opcode == Opcode::b);
    CHECK_EQ(i.displacement, (std::int64_t{1} << 27) - 4);

    i = decode(0x16000000u);  // b .-0x8000000
    CHECK_EQ(i.displacement, -(std::int64_t{1} << 27));

    i = decode(0x17FFFFFFu);  // b .-4
    CHECK_EQ(i.displacement, -4);

    i = decode(0x94000001u);  // bl .+4
    CHECK(i.opcode == Opcode::bl);
    CHECK_EQ(i.displacement, 4);
    CHECK_EQ(i.writes, x30);
    CHECK(writes_link_address(Opcode::bl));
    CHECK(!writes_link_address(Opcode::b));

    CHECK_EQ(displacement_field_mask(Opcode::b), 0x03FFFFFFu);
}

void conditional_branches() {
    auto i = decode(0x54000001u);  // b.ne .
    CHECK(i.opcode == Opcode::b_cond);
    CHECK(i.condition == Condition::ne);
    CHECK_EQ(i.displacement, 0);

    i = decode(0x547FFFEBu);  // b.lt .+0xffffc
    CHECK(i.condition == Condition::lt);
    CHECK_EQ(i.displacement, (1 << 20) - 4);

    i = decode(0x5480000Eu);  // b.al .-0x100000
    CHECK(i.condition == Condition::al);
    CHECK_EQ(i.displacement, -(1 << 20));

    i = decode(0x5400003Fu);  // bc.nv .+4
    CHECK(i.opcode == Opcode::bc_cond);
    CHECK(i.condition == Condition::nv);
    CHECK_EQ(i.displacement, 4);

    // o1 (bit 24) set is unallocated.
    CHECK(decode(0x55000000u).opcode == Opcode::unclassified);

    CHECK(invert(Condition::eq) == Condition::ne);
    CHECK(invert(Condition::hs) == Condition::lo);
    CHECK(invert(Condition::gt) == Condition::le);
}

void compare_and_test_branches() {
    auto i = decode(0xB4000012u);  // cbz x18, .
    CHECK(i.opcode == Opcode::cbz);
    CHECK(i.wide);
    CHECK_EQ(i.reg, 18);
    CHECK_EQ(i.reads, x18);

    i = decode(0x35FFFFE3u);  // cbnz w3, .-4
    CHECK(i.opcode == Opcode::cbnz);
    CHECK(!i.wide);
    CHECK_EQ(i.reg, 3);
    CHECK_EQ(i.displacement, -4);

    i = decode(0xB400001Fu);  // cbz xzr, .
    CHECK_EQ(i.reg, 31);
    CHECK_EQ(i.reads, 0);

    i = decode(0xB7FBFFE5u);  // tbnz x5, #63, .+0x7ffc
    CHECK(i.opcode == Opcode::tbnz);
    CHECK(i.wide);
    CHECK_EQ(i.test_bit, 63);
    CHECK_EQ(i.reg, 5);
    CHECK_EQ(i.displacement, (1 << 15) - 4);

    i = decode(0x36040007u);  // tbz w7, #0, .-0x8000
    CHECK(i.opcode == Opcode::tbz);
    CHECK(!i.wide);
    CHECK_EQ(i.test_bit, 0);
    CHECK_EQ(i.displacement, -(1 << 15));

    i = decode(0x36F80012u);  // tbz w18, #31, .
    CHECK_EQ(i.test_bit, 31);
    CHECK_EQ(i.reads, x18);

    CHECK_EQ(displacement_field_mask(Opcode::tbz), 0x0007FFE0u);
    CHECK_EQ(displacement_field_mask(Opcode::cbnz), 0x00FFFFE0u);
}

void address_generation() {
    auto i = decode(0x30000012u);  // adr x18, .+1
    CHECK(i.opcode == Opcode::adr);
    CHECK_EQ(i.displacement, 1);
    CHECK_EQ(i.writes, x18);

    i = decode(0x707FFFE0u);  // adr x0, .+0xfffff
    CHECK_EQ(i.displacement, (1 << 20) - 1);
    i = decode(0x10800000u);  // adr x0, .-0x100000
    CHECK_EQ(i.displacement, -(1 << 20));
    i = decode(0x70FFFFE1u);  // adr x1, .-1
    CHECK_EQ(i.displacement, -1);
    CHECK_EQ(i.reg, 1);

    i = decode(0xF07FFFE2u);  // adrp x2, page+0xfffff000
    CHECK(i.opcode == Opcode::adrp);
    CHECK_EQ(i.displacement, (std::int64_t{1} << 32) - 4096);
    i = decode(0x90800003u);  // adrp x3, page-0x100000000
    CHECK_EQ(i.displacement, -(std::int64_t{1} << 32));

    // ADRP refers to the page of pc, so the low 12 bits of pc do not matter.
    i = decode(0xB0000004u);  // adrp x4, page+0x1000
    CHECK_EQ(*referenced_address(i, 0x10FFCu), 0x11000u);
    CHECK_EQ(*referenced_address(i, 0x10000u), 0x11000u);

    CHECK_EQ(displacement_field_mask(Opcode::adr), 0x60FFFFE0u);
}

void literal_loads() {
    struct Case {
        std::uint32_t word;
        Opcode opcode;
        bool wide;
        std::uint64_t writes;
    };
    const Case cases[] = {
        {0x18000021u, Opcode::ldr_literal_w, false, 1u << 1},  // ldr w1, .+4
        {0x58000022u, Opcode::ldr_literal_x, true, 1u << 2},   // ldr x2, .+4
        {0x98000032u, Opcode::ldrsw_literal, true, x18},       // ldrsw x18, .+4
        {0x1C000024u, Opcode::ldr_literal_s, false, 0},        // ldr s4, .+4
        {0x5C000025u, Opcode::ldr_literal_d, false, 0},        // ldr d5, .+4
        {0x9C000026u, Opcode::ldr_literal_q, false, 0},        // ldr q6, .+4
    };
    for (const auto& c : cases) {
        const auto i = decode(c.word);
        CHECK(i.opcode == c.opcode);
        CHECK_EQ(i.wide, c.wide);
        CHECK_EQ(i.writes, c.writes);
        CHECK_EQ(i.displacement, 4);
        CHECK_EQ(i.reg, c.word & 31u);
    }

    auto i = decode(0xD8000032u);  // prfm #18, .+4: Rt is an operation, not x18
    CHECK(i.opcode == Opcode::prfm_literal);
    CHECK_EQ(i.prefetch_operation, 18);
    CHECK_EQ(i.writes | i.reads, 0);

    i = decode(0x587FFFE0u);  // ldr x0, .+0xffffc
    CHECK_EQ(i.displacement, (1 << 20) - 4);
    i = decode(0x18800000u);  // ldr w0, .-0x100000
    CHECK_EQ(i.displacement, -(1 << 20));

    // opc=11 with V=1 is unallocated.
    CHECK(decode(0xDC000000u).opcode == Opcode::unclassified);
}

void register_branches() {
    auto i = decode(0xD61F0240u);  // br x18
    CHECK(i.opcode == Opcode::br);
    CHECK_EQ(i.reg, 18);
    CHECK_EQ(i.reads, x18);

    i = decode(0xD63F0120u);  // blr x9
    CHECK(i.opcode == Opcode::blr);
    CHECK_EQ(i.reads, 1u << 9);
    CHECK_EQ(i.writes, x30);
    CHECK(writes_link_address(Opcode::blr));

    i = decode(0xD65F03C0u);  // ret
    CHECK(i.opcode == Opcode::ret);
    CHECK_EQ(i.reads, x30);
    CHECK(!is_pc_relative(Opcode::ret));
    CHECK_EQ(displacement_field_mask(Opcode::br), 0);

    // Pointer-authenticating and exception-return forms are not classified.
    CHECK(decode(0xD61F081Fu).opcode == Opcode::unclassified);  // braaz x0
    CHECK(decode(0xD65F0BFFu).opcode == Opcode::unclassified);  // retaa
    CHECK(decode(0xD69F03E0u).opcode == Opcode::unclassified);  // eret
    CHECK(decode(0xD61F0241u).opcode == Opcode::unclassified);  // br with op4 != 0
}

void exceptions() {
    auto i = decode(0xD4000021u);  // svc #1
    CHECK(i.opcode == Opcode::svc);
    CHECK_EQ(i.immediate, 1);
    CHECK_EQ(i.reads | i.writes, 0);
    i = decode(0xD41FFFE1u);  // svc #0xffff
    CHECK_EQ(i.immediate, 0xFFFF);
    CHECK(decode(0xD4000002u).opcode == Opcode::hvc);
    CHECK(decode(0xD4000003u).opcode == Opcode::smc);
    i = decode(0xD4200300u);  // brk #0x18
    CHECK(i.opcode == Opcode::brk);
    CHECK_EQ(i.immediate, 0x18);
    CHECK(decode(0xD4400000u).opcode == Opcode::hlt);

    CHECK(decode(0xD4000000u).opcode == Opcode::unclassified);  // svc space with LL=00
    CHECK(decode(0xD4000005u).opcode == Opcode::unclassified);  // op2 != 0
    CHECK(decode(0xD4600000u).opcode == Opcode::unclassified);  // tcancel
    CHECK(decode(0xD4A00001u).opcode == Opcode::unclassified);  // dcps1
}

void system_registers() {
    struct Case {
        std::uint32_t word;
        Opcode opcode;
        SystemRegister reg;
    };
    const Case cases[] = {
        {0xD53BD052u, Opcode::mrs, SystemRegister::tpidr_el0},           // mrs x18, tpidr_el0
        {0xD51BD041u, Opcode::msr_register, SystemRegister::tpidr_el0},  // msr tpidr_el0, x1
        {0xD53BD062u, Opcode::mrs, SystemRegister::tpidrro_el0},
        {0xD51BD063u, Opcode::msr_register, SystemRegister::tpidrro_el0},
        {0xD53BE003u, Opcode::mrs, SystemRegister::cntfrq_el0},
        {0xD53BE024u, Opcode::mrs, SystemRegister::cntpct_el0},
        {0xD53BE045u, Opcode::mrs, SystemRegister::cntvct_el0},
        {0xD53BE0A6u, Opcode::mrs, SystemRegister::cntpctss_el0},
        {0xD53BE0C7u, Opcode::mrs, SystemRegister::cntvctss_el0},
        {0xD53B4208u, Opcode::mrs, SystemRegister::nzcv},
        {0xD51B4409u, Opcode::msr_register, SystemRegister::fpcr},
        {0xD53B442Au, Opcode::mrs, SystemRegister::fpsr},
        {0xD53B002Bu, Opcode::mrs, SystemRegister::ctr_el0},
        {0xD53B00ECu, Opcode::mrs, SystemRegister::dczid_el0},
        {0xD538000Du, Opcode::mrs, SystemRegister::other},  // mrs x13, midr_el1
        {0xD5300000u, Opcode::mrs, SystemRegister::other},  // op0=2 debug space
    };
    for (const auto& c : cases) {
        const auto i = decode(c.word);
        CHECK(i.opcode == c.opcode);
        CHECK(identify_system_register(i.system_register) == c.reg);
        CHECK_EQ(i.reg, c.word & 31u);
        const auto bit = register_bit(c.word & 31u);
        CHECK_EQ(c.opcode == Opcode::mrs ? i.writes : i.reads, bit);
        if (c.reg != SystemRegister::other) {
            CHECK_EQ(*encoding_of(c.reg), i.system_register);
        }
    }
    CHECK_EQ(decode(0xD53BD040u).system_register, system_register_encoding(3, 3, 13, 0, 2));
    CHECK(!encoding_of(SystemRegister::other));

    // MSR (immediate), SYS, hints, and barriers have op0 < 2 and are not
    // system register moves.
    CHECK(decode(0xD503201Fu).opcode == Opcode::unclassified);  // nop
    CHECK(decode(0xD50B7420u).opcode == Opcode::unclassified);  // dc zva, x0
    CHECK(decode(0xD50040BFu).opcode == Opcode::unclassified);  // msr spsel, #0
    CHECK(decode(0xD5033BBFu).opcode == Opcode::unclassified);  // dmb ish
}

void unclassified_words() {
    const std::uint32_t words[] = {
        0x00000000u,  // udf #0
        0x8B120000u,  // add x0, x0, x18
        0xF9400252u,  // ldr x18, [x18]
        0xA9BF7BFDu,  // stp x29, x30, [sp, #-16]!
        0xD2800000u,  // movz x0, #0
        0xFFFFFFFFu,
    };
    for (const auto word : words) {
        const auto i = decode(word);
        CHECK(!i.classified());
        CHECK_EQ(i.word, word);
        CHECK_EQ(i.reads | i.writes, 0);
        CHECK(!referenced_address(i, 0x1000));
    }
}

void referenced_addresses() {
    const auto b = decode(0x17FFFFFFu);  // b .-4
    CHECK_EQ(*referenced_address(b, 0x1000), 0xFFCu);
    CHECK(!referenced_address(b, 0));            // leaves the address space below zero
    CHECK(!referenced_address(b, 0x1002));       // misaligned pc
    const auto forward = decode(0x14000002u);    // b .+8
    CHECK(!referenced_address(forward, 0xFFFFFFFFFFFFFFFCull));
    CHECK_EQ(*referenced_address(forward, 0xFFFFFFFFFFFFFFF4ull), 0xFFFFFFFFFFFFFFFCull);
    const auto adr = decode(0x70FFFFE1u);        // adr x1, .-1
    CHECK_EQ(*referenced_address(adr, 0x2000), 0x1FFFu);
}

void ranges() {
    const auto b = *displacement_range(Opcode::bl);
    CHECK_EQ(b.minimum, -(std::int64_t{1} << 27));
    CHECK_EQ(b.maximum, (std::int64_t{1} << 27) - 4);
    const auto tb = *displacement_range(Opcode::tbnz);
    CHECK_EQ(tb.minimum, -(1 << 15));
    CHECK_EQ(tb.maximum, (1 << 15) - 4);
    const auto adr = *displacement_range(Opcode::adr);
    CHECK_EQ(adr.granularity, 1);
    CHECK_EQ(adr.maximum, (1 << 20) - 1);
    const auto adrp = *displacement_range(Opcode::adrp);
    CHECK_EQ(adrp.granularity, 4096);
    CHECK_EQ(adrp.minimum, -(std::int64_t{1} << 32));
    CHECK(!displacement_range(Opcode::svc));
    CHECK(!displacement_range(Opcode::unclassified));
}

void encoders_produce_hand_derived_words() {
    CHECK_EQ(encode_b(0x1000, 0x1000).word, 0x14000000u);
    CHECK_EQ(encode_b(0x1000, 0x0FFC).word, 0x17FFFFFFu);
    CHECK_EQ(encode_bl(0x1000, 0x1004).word, 0x94000001u);
    CHECK_EQ(encode_b_cond(Condition::ne, 0x1000, 0x1000).word, 0x54000001u);
    CHECK_EQ(encode_b_cond(Condition::nv, 0x1000, 0x1004, true).word, 0x5400003Fu);
    CHECK_EQ(encode_cbz(false, true, 18, 0x1000, 0x1000).word, 0xB4000012u);
    CHECK_EQ(encode_cbz(true, false, 3, 0x1000, 0x0FFC).word, 0x35FFFFE3u);
    CHECK_EQ(encode_tbz(true, 5, 63, 0x1000, 0x1000 + 0x7FFC).word, 0xB7FBFFE5u);
    CHECK_EQ(encode_tbz(false, 7, 0, 0x10000, 0x10000 - 0x8000).word, 0x36040007u);
    CHECK_EQ(encode_adr(18, 0x1000, 0x1001).word, 0x30000012u);
    CHECK_EQ(encode_adr(1, 0x2000, 0x1FFF).word, 0x70FFFFE1u);
    CHECK_EQ(encode_adrp(4, 0x10FFC, 0x11234).word, 0xB0000004u);
    CHECK_EQ(encode_load_literal(Opcode::ldr_literal_x, 2, 0x1000, 0x1004).word, 0x58000022u);
    CHECK_EQ(encode_load_literal(Opcode::prfm_literal, 18, 0x1000, 0x1004).word, 0xD8000032u);
    CHECK_EQ(encode_load_literal(Opcode::ldr_literal_q, 6, 0x1000, 0x1004).word, 0x9C000026u);
    CHECK_EQ(encode_br(18).word, 0xD61F0240u);
    CHECK_EQ(encode_blr(9).word, 0xD63F0120u);
    CHECK_EQ(encode_ret().word, 0xD65F03C0u);
    CHECK_EQ(encode_svc(0xFFFF).word, 0xD41FFFE1u);
    CHECK_EQ(encode_brk(0x18).word, 0xD4200300u);
    CHECK_EQ(encode_mrs(18, *encoding_of(SystemRegister::tpidr_el0)).word, 0xD53BD052u);
    CHECK_EQ(encode_msr(*encoding_of(SystemRegister::fpcr), 9).word, 0xD51B4409u);
    CHECK_EQ(encode_add_immediate(true, 0, 0, 0x123).word, 0x91048C00u);       // add x0, x0, #0x123
    CHECK_EQ(encode_add_immediate(false, 31, 1, 1, true).word, 0x1140043Fu);   // add wsp, w1, #1, lsl #12
    CHECK_EQ(encode_load_unsigned_offset(Opcode::ldr_literal_x, 1, 2, 0x7FF8).word, 0xF97FFC41u);
    CHECK_EQ(encode_load_unsigned_offset(Opcode::ldr_literal_w, 1, 31, 4).word, 0xB94007E1u);
    CHECK_EQ(encode_load_unsigned_offset(Opcode::ldrsw_literal, 3, 3, 8).word, 0xB9800863u);
    CHECK_EQ(encode_load_unsigned_offset(Opcode::prfm_literal, 0, 4, 0).word, 0xF9800080u);
    CHECK_EQ(encode_load_unsigned_offset(Opcode::ldr_literal_s, 5, 6, 4).word, 0xBD4004C5u);
    CHECK_EQ(encode_load_unsigned_offset(Opcode::ldr_literal_d, 7, 8, 8).word, 0xFD400507u);
    CHECK_EQ(encode_load_unsigned_offset(Opcode::ldr_literal_q, 9, 10, 0xFFF0).word, 0x3DFFFD49u);
}

void encoders_reject_unencodable_operands() {
    // Endpoints are accepted; one step beyond is rejected, never truncated.
    const std::uint64_t pc = 0x40000000;
    CHECK(encode_b(pc, pc + (1u << 27) - 4).ok());
    CHECK(encode_b(pc, pc - (1u << 27)).ok());
    CHECK(encode_b(pc, pc + (1u << 27)).error == EncodeError::out_of_range);
    CHECK(encode_bl(pc, pc - (1u << 27) - 4).error == EncodeError::out_of_range);
    CHECK(encode_b_cond(Condition::eq, pc, pc + (1u << 20) - 4).ok());
    CHECK(encode_b_cond(Condition::eq, pc, pc + (1u << 20)).error == EncodeError::out_of_range);
    CHECK(encode_cbz(false, true, 0, pc, pc - (1u << 20)).ok());
    CHECK(encode_cbz(false, true, 0, pc, pc - (1u << 20) - 4).error == EncodeError::out_of_range);
    CHECK(encode_tbz(false, 0, 0, pc, pc + (1u << 15) - 4).ok());
    CHECK(encode_tbz(false, 0, 0, pc, pc + (1u << 15)).error == EncodeError::out_of_range);
    CHECK(encode_tbz(false, 0, 0, pc, pc - (1u << 15) - 4).error == EncodeError::out_of_range);
    CHECK(encode_adr(0, pc, pc + (1u << 20) - 1).ok());
    CHECK(encode_adr(0, pc, pc + (1u << 20)).error == EncodeError::out_of_range);
    CHECK(encode_adr(0, pc, pc - (1u << 20)).ok());
    CHECK(encode_adr(0, pc, pc - (1u << 20) - 1).error == EncodeError::out_of_range);
    const std::uint64_t high = 0x400000000ull;
    CHECK(encode_adrp(0, high, high + (1ull << 32) - 4096).ok());
    CHECK(encode_adrp(0, high, high + (1ull << 32) + 0xFFF).ok() == false);
    CHECK(encode_adrp(0, high, high - (1ull << 32)).ok());
    CHECK(encode_adrp(0, high, high - (1ull << 32) - 1).error == EncodeError::out_of_range);
    CHECK(encode_load_literal(Opcode::ldr_literal_w, 0, pc, pc + (1u << 20)).error ==
          EncodeError::out_of_range);

    // Distances that do not fit int64 are out of range, not wrapped.
    CHECK(encode_b(0, 0xFFFFFFFFFFFFFFFCull).error == EncodeError::out_of_range);
    CHECK(encode_b(0xFFFFFFFFFFFFFFFCull, 0).error == EncodeError::out_of_range);

    CHECK(encode_b(pc + 2, pc).error == EncodeError::misaligned_address);
    CHECK(encode_b(pc, pc + 2).error == EncodeError::misaligned_address);
    CHECK(encode_load_literal(Opcode::ldr_literal_x, 0, pc, pc + 6).error ==
          EncodeError::misaligned_address);
    CHECK(encode_adr(0, pc + 1, pc).error == EncodeError::misaligned_address);

    CHECK(encode_cbz(false, true, 32, pc, pc).error == EncodeError::invalid_register);
    CHECK(encode_tbz(false, 0, 64, pc, pc).error == EncodeError::invalid_operand);
    CHECK(encode_br(32).error == EncodeError::invalid_register);
    CHECK(encode_b_cond(static_cast<Condition>(16), pc, pc).error == EncodeError::invalid_operand);
    CHECK(encode_load_literal(Opcode::b, 0, pc, pc).error == EncodeError::invalid_operand);
    CHECK(encode_mrs(0, system_register_encoding(1, 0, 7, 5, 1)).error ==
          EncodeError::invalid_operand);
    CHECK(encode_mrs(32, *encoding_of(SystemRegister::nzcv)).error ==
          EncodeError::invalid_register);
    CHECK(encode_add_immediate(true, 0, 0, 0x1000).error == EncodeError::out_of_range);
    CHECK(encode_load_unsigned_offset(Opcode::ldr_literal_x, 0, 1, 4).error ==
          EncodeError::misaligned_address);
    CHECK(encode_load_unsigned_offset(Opcode::ldr_literal_x, 0, 1, 0x8000).error ==
          EncodeError::out_of_range);
    CHECK(encode_load_unsigned_offset(Opcode::adr, 0, 1, 0).error == EncodeError::invalid_operand);
}

// Encoding a random in-range target and decoding the word must reproduce the
// target and leave only the displacement field different from the template.
void encode_decode_round_trip() {
    armigo::aarch64::test::Lcg random{0x5eed};
    for (int iteration = 0; iteration < 20000; ++iteration) {
        const std::uint64_t pc = 0x200000000ull + (random.next() % 0x40000000ull) * 4;
        const auto pick = [&](std::int64_t minimum, std::int64_t maximum, std::int64_t step) {
            const auto steps = static_cast<std::uint64_t>((maximum - minimum) / step) + 1;
            return minimum + static_cast<std::int64_t>(random.next() % steps) * step;
        };
        const auto target_for = [&](Opcode op) {
            const auto range = *displacement_range(op);
            return static_cast<std::uint64_t>(
                static_cast<std::int64_t>(op == Opcode::adrp ? (pc & ~0xFFFull) : pc) +
                pick(range.minimum, range.maximum, range.granularity));
        };
        const unsigned reg = static_cast<unsigned>(random.next() % 32);

        std::uint64_t target = target_for(Opcode::b);
        auto e = encode_b(pc, target);
        CHECK(e.ok() && decode(e.word).opcode == Opcode::b &&
              *referenced_address(decode(e.word), pc) == target);
        CHECK_EQ(e.word & ~displacement_field_mask(Opcode::b), 0x14000000u);

        target = target_for(Opcode::b_cond);
        const auto condition = static_cast<Condition>(random.next() % 16);
        e = encode_b_cond(condition, pc, target);
        CHECK(e.ok() && decode(e.word).condition == condition &&
              *referenced_address(decode(e.word), pc) == target);

        target = target_for(Opcode::cbnz);
        e = encode_cbz(true, false, reg, pc, target);
        CHECK(e.ok() && decode(e.word).opcode == Opcode::cbnz && decode(e.word).reg == reg &&
              *referenced_address(decode(e.word), pc) == target);

        target = target_for(Opcode::tbz);
        const unsigned bit = static_cast<unsigned>(random.next() % 64);
        e = encode_tbz(false, reg, bit, pc, target);
        CHECK(e.ok() && decode(e.word).test_bit == bit && decode(e.word).wide == (bit >= 32) &&
              *referenced_address(decode(e.word), pc) == target);

        target = target_for(Opcode::adr);
        e = encode_adr(reg, pc, target);
        CHECK(e.ok() && *referenced_address(decode(e.word), pc) == target);

        target = target_for(Opcode::adrp) | (random.next() & 0xFFF);
        e = encode_adrp(reg, pc, target);
        CHECK(e.ok() && *referenced_address(decode(e.word), pc) == (target & ~0xFFFull));

        target = target_for(Opcode::ldrsw_literal);
        e = encode_load_literal(Opcode::ldrsw_literal, reg, pc, target);
        CHECK(e.ok() && decode(e.word).opcode == Opcode::ldrsw_literal &&
              *referenced_address(decode(e.word), pc) == target);
    }
}

} // namespace

int main() {
    immediate_branches();
    conditional_branches();
    compare_and_test_branches();
    address_generation();
    literal_loads();
    register_branches();
    exceptions();
    system_registers();
    unclassified_words();
    referenced_addresses();
    ranges();
    encoders_produce_hand_derived_words();
    encoders_reject_unencodable_operands();
    encode_decode_round_trip();
    return armigo::aarch64::test::finish("aarch64.instruction");
}
