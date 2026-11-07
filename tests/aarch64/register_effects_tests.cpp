// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Register-effect checks against instruction words composed from the field
// layouts of the Arm A64 encoding tables. The composition helpers below are
// written independently of the implementation, and a sample of their words
// is pinned to words produced by the system assembler from mnemonics.
// Expected operands, roles and register-31 meanings are written out per
// instruction from the instruction descriptions, not computed.

#include <armigo/aarch64/register_effects.h>
#include <armigo/aarch64/register_substitution.h>

#include "test_support.h"

#include <cstdint>
#include <initializer_list>
#include <optional>

namespace {

using namespace armigo::aarch64;

constexpr auto ZR = Register31::zero_register;
constexpr auto SP = Register31::stack_pointer;
constexpr auto NA = Register31::not_applicable;
constexpr auto R = Access::read;
constexpr auto W = Access::write;
constexpr auto RW = Access::read_write;
constexpr auto GPR = RegisterFile::general;
constexpr auto VEC = RegisterFile::vector;
using Role = OperandRole;

struct Op {
    std::uint8_t field;
    std::uint8_t number;
    Register31 at_31;
    std::uint8_t bits;
    Access access;
    Role role;
    bool conditional = false;
    RegisterFile file = GPR;
};

std::uint32_t with_field(std::uint32_t word, unsigned lsb, unsigned value) {
    return (word & ~(std::uint32_t{0x1F} << lsb)) | (std::uint32_t{value & 0x1F} << lsb);
}

// ---------------------------------------------------------------------------
// Field composition from the encoding tables.

std::uint32_t add_sub_imm(unsigned sf, unsigned op, unsigned s, unsigned sh, unsigned imm12,
                          unsigned rn, unsigned rd) {
    return sf << 31 | op << 30 | s << 29 | 0b100010u << 23 | sh << 22 | imm12 << 10 | rn << 5 | rd;
}
std::uint32_t logical_imm(unsigned sf, unsigned opc, unsigned n, unsigned immr, unsigned imms,
                          unsigned rn, unsigned rd) {
    return sf << 31 | opc << 29 | 0b100100u << 23 | n << 22 | immr << 16 | imms << 10 | rn << 5 |
           rd;
}
std::uint32_t move_wide(unsigned sf, unsigned opc, unsigned hw, unsigned imm16, unsigned rd) {
    return sf << 31 | opc << 29 | 0b100101u << 23 | hw << 21 | imm16 << 5 | rd;
}
std::uint32_t bitfield(unsigned sf, unsigned opc, unsigned n, unsigned immr, unsigned imms,
                       unsigned rn, unsigned rd) {
    return sf << 31 | opc << 29 | 0b100110u << 23 | n << 22 | immr << 16 | imms << 10 | rn << 5 |
           rd;
}
std::uint32_t extract(unsigned sf, unsigned op21, unsigned n, unsigned o0, unsigned rm,
                      unsigned imms, unsigned rn, unsigned rd) {
    return sf << 31 | op21 << 29 | 0b100111u << 23 | n << 22 | o0 << 21 | rm << 16 | imms << 10 |
           rn << 5 | rd;
}
std::uint32_t logical_shifted(unsigned sf, unsigned opc, unsigned shift, unsigned n, unsigned rm,
                              unsigned imm6, unsigned rn, unsigned rd) {
    return sf << 31 | opc << 29 | 0b01010u << 24 | shift << 22 | n << 21 | rm << 16 | imm6 << 10 |
           rn << 5 | rd;
}
std::uint32_t add_sub_shifted(unsigned sf, unsigned op, unsigned s, unsigned shift, unsigned rm,
                              unsigned imm6, unsigned rn, unsigned rd) {
    return sf << 31 | op << 30 | s << 29 | 0b01011u << 24 | shift << 22 | rm << 16 | imm6 << 10 |
           rn << 5 | rd;
}
std::uint32_t add_sub_extended(unsigned sf, unsigned op, unsigned s, unsigned opt, unsigned rm,
                               unsigned option, unsigned imm3, unsigned rn, unsigned rd) {
    return sf << 31 | op << 30 | s << 29 | 0b01011u << 24 | opt << 22 | 1u << 21 | rm << 16 |
           option << 13 | imm3 << 10 | rn << 5 | rd;
}
std::uint32_t carry(unsigned sf, unsigned op, unsigned s, unsigned rm, unsigned op3, unsigned rn,
                    unsigned rd) {
    return sf << 31 | op << 30 | s << 29 | 0b11010000u << 21 | rm << 16 | op3 << 10 | rn << 5 | rd;
}
std::uint32_t cond_compare(unsigned sf, unsigned op, unsigned s, unsigned rm_imm5, unsigned cond,
                           unsigned immediate, unsigned o2, unsigned rn, unsigned o3,
                           unsigned nzcv) {
    return sf << 31 | op << 30 | s << 29 | 0b11010010u << 21 | rm_imm5 << 16 | cond << 12 |
           immediate << 11 | o2 << 10 | rn << 5 | o3 << 4 | nzcv;
}
std::uint32_t cond_select(unsigned sf, unsigned op, unsigned s, unsigned rm, unsigned cond,
                          unsigned op2, unsigned rn, unsigned rd) {
    return sf << 31 | op << 30 | s << 29 | 0b11010100u << 21 | rm << 16 | cond << 12 | op2 << 10 |
           rn << 5 | rd;
}
std::uint32_t two_source(unsigned sf, unsigned s, unsigned rm, unsigned opcode, unsigned rn,
                         unsigned rd) {
    return sf << 31 | s << 29 | 0b11010110u << 21 | rm << 16 | opcode << 10 | rn << 5 | rd;
}
std::uint32_t one_source(unsigned sf, unsigned s, unsigned opcode2, unsigned opcode, unsigned rn,
                         unsigned rd) {
    return sf << 31 | 1u << 30 | s << 29 | 0b11010110u << 21 | opcode2 << 16 | opcode << 10 |
           rn << 5 | rd;
}
std::uint32_t three_source(unsigned sf, unsigned op54, unsigned op31, unsigned rm, unsigned o0,
                           unsigned ra, unsigned rn, unsigned rd) {
    return sf << 31 | op54 << 29 | 0b11011u << 24 | op31 << 21 | rm << 16 | o0 << 15 | ra << 10 |
           rn << 5 | rd;
}

// Load/store register: unsigned offset, imm9 forms (kind: 0 unscaled,
// 1 post-index, 2 unprivileged, 3 pre-index) and register offset.
std::uint32_t ls_unsigned(unsigned size, unsigned v, unsigned opc, unsigned imm12, unsigned rn,
                          unsigned rt) {
    return size << 30 | 0b111u << 27 | v << 26 | 1u << 24 | opc << 22 | imm12 << 10 | rn << 5 | rt;
}
std::uint32_t ls_imm9(unsigned size, unsigned v, unsigned opc, int imm9, unsigned kind, unsigned rn,
                      unsigned rt) {
    return size << 30 | 0b111u << 27 | v << 26 | opc << 22 |
           (static_cast<std::uint32_t>(imm9) & 0x1FFu) << 12 | kind << 10 | rn << 5 | rt;
}
std::uint32_t ls_register(unsigned size, unsigned v, unsigned opc, unsigned rm, unsigned option,
                          unsigned s, unsigned rn, unsigned rt) {
    return size << 30 | 0b111u << 27 | v << 26 | opc << 22 | 1u << 21 | rm << 16 | option << 13 |
           s << 12 | 0b10u << 10 | rn << 5 | rt;
}
// mode: 0 no-allocate, 1 post-index, 2 signed offset, 3 pre-index.
std::uint32_t ls_pair(unsigned opc, unsigned v, unsigned mode, unsigned l, int imm7, unsigned rt2,
                      unsigned rn, unsigned rt) {
    return opc << 30 | 0b101u << 27 | v << 26 | mode << 23 | l << 22 |
           (static_cast<std::uint32_t>(imm7) & 0x7Fu) << 15 | rt2 << 10 | rn << 5 | rt;
}

// Load/store exclusive and ordered: size o2 L o1 Rs o0 Rt2 Rn Rt.
std::uint32_t exclusive(unsigned size, unsigned o2, unsigned l, unsigned o1, unsigned rs,
                        unsigned o0, unsigned rt2, unsigned rn, unsigned rt) {
    return size << 30 | 0b001000u << 24 | o2 << 23 | l << 22 | o1 << 21 | rs << 16 | o0 << 15 |
           rt2 << 10 | rn << 5 | rt;
}

// ---------------------------------------------------------------------------
// Checks.

bool same_operand(const RegisterOperand& actual, const Op& expected) {
    return actual.field == expected.field && actual.number == expected.number &&
           actual.at_31 == expected.at_31 && actual.bits == expected.bits &&
           actual.access == expected.access && actual.role == expected.role &&
           actual.conditional == expected.conditional && actual.file == expected.file;
}

RegisterMask mask_of(std::initializer_list<Op> ops, bool want_write) {
    RegisterMask mask = 0;
    for (const auto& op : ops) {
        if (op.file != GPR || op.number >= 31) {
            continue;
        }
        if (want_write ? includes_write(op.access) : includes_read(op.access)) {
            mask |= register_bit(op.number);
        }
    }
    return mask;
}

// Substitutes 18 and 31 into every explicit general-register field and checks
// that only that operand changes, with the expected register-31 meaning. Then
// writes 18 into each remaining standard field position and checks that any
// word still decoding as the same operation does not use x18.
void check_field_semantics(std::uint32_t word, Operation operation,
                           std::initializer_list<Op> ops, int line) {
    unsigned explicit_fields = 0;
    std::size_t index = 0;
    for (const auto& op : ops) {
        const std::size_t position = index++;
        if (op.field == implicit_operand) {
            continue;
        }
        explicit_fields |= 1u << op.field;
        if (op.file != GPR) {
            // V18 is never x18. The word may become a pair-load hazard
            // (Rt == Rt2), so inspect operands rather than the status.
            const auto e = decode_register_effects(with_field(word, op.field, 18));
            bool names_x18 = false;
            for (std::size_t other = 0; other < e.operand_count; ++other) {
                names_x18 = names_x18 || e.operands[other].names_general_register(18);
            }
            if (!CHECK(e.operation == operation && !names_x18 && (e.reads >> 18 & 1u) == 0 &&
                       (e.writes >> 18 & 1u) == 0)) {
                std::fprintf(stderr, "  line %d: vector field %u = 18\n", line, op.field);
            }
            continue;
        }
        const auto e18 = decode_register_effects(with_field(word, op.field, 18));
        const bool ok18 = CHECK(e18.operation == operation) &&
                          CHECK(e18.operands[position].number == 18) &&
                          CHECK(register_fields(e18, 18) == std::optional<std::uint32_t>{
                                                                std::uint32_t{0x1F} << op.field}) &&
                          CHECK(!proven_free_of(e18, 18)) &&
                          CHECK(((e18.reads >> 18) & 1u) == (includes_read(op.access) ? 1u : 0u)) &&
                          CHECK(((e18.writes >> 18) & 1u) == (includes_write(op.access) ? 1u : 0u));
        if (!ok18) {
            std::fprintf(stderr, "  line %d: field %u = 18, word 0x%08x\n", line, op.field,
                         with_field(word, op.field, 18));
        }

        const auto e31 = decode_register_effects(with_field(word, op.field, 31));
        const auto& operand = e31.operands[position];
        const bool ok31 =
            CHECK(e31.decoded()) && CHECK(e31.operation == operation) &&
            CHECK(operand.number == 31) &&
            CHECK(op.at_31 == SP ? operand.names_stack_pointer() : operand.names_zero_register()) &&
            CHECK(op.at_31 != SP || reads_stack_pointer(e31) || !includes_read(op.access)) &&
            CHECK(op.at_31 != SP || writes_stack_pointer(e31) || !includes_write(op.access));
        if (!ok31) {
            std::fprintf(stderr, "  line %d: field %u = 31, word 0x%08x\n", line, op.field,
                         with_field(word, op.field, 31));
        }
    }
    for (unsigned lsb : {0u, 5u, 10u, 16u}) {
        if (explicit_fields & (1u << lsb)) {
            continue;
        }
        const auto e = decode_register_effects(with_field(word, lsb, 18));
        if (e.decoded() && e.operation == operation) {
            if (!CHECK(proven_free_of(e, 18))) {
                std::fprintf(stderr, "  line %d: non-register field %u = 18\n", line, lsb);
            }
        }
    }
}

RegisterEffects expect_effects(std::uint32_t word, Operation operation,
                               std::initializer_list<Op> ops, int line) {
    const auto e = decode_register_effects(word);
    bool ok = CHECK(e.status == EffectStatus::decoded) && CHECK(e.operation == operation) &&
              CHECK_EQ(e.operand_count, ops.size());
    std::size_t index = 0;
    for (const auto& op : ops) {
        if (!ok || index >= e.operand_count) {
            break;
        }
        ok = CHECK(same_operand(e.operands[index], op));
        ++index;
    }
    ok = ok && CHECK_EQ(e.reads, mask_of(ops, false)) && CHECK_EQ(e.writes, mask_of(ops, true));
    if (!ok) {
        std::fprintf(stderr, "  line %d: word 0x%08x\n", line, word);
        return e;
    }
    check_field_semantics(word, operation, ops, line);
    return e;
}

#define EXPECT_EFFECTS(word, operation, ...) \
    expect_effects((word), (operation), {__VA_ARGS__}, __LINE__)

void expect_status(std::uint32_t word, EffectStatus status, int line) {
    const auto e = decode_register_effects(word);
    bool ok = CHECK(e.status == status);
    if (status == EffectStatus::unknown || status == EffectStatus::reserved) {
        ok = ok && CHECK(e.operation == Operation::none) && CHECK_EQ(e.operand_count, 0) &&
             CHECK_EQ(e.reads, 0) && CHECK_EQ(e.writes, 0) &&
             CHECK(e.memory.operation == MemoryOperation::none);
    }
    for (unsigned reg = 0; reg < 31; ++reg) {
        ok = ok && CHECK(!proven_free_of(e, reg)) && CHECK(!register_fields(e, reg).has_value());
    }
    if (!ok) {
        std::fprintf(stderr, "  line %d: word 0x%08x\n", line, word);
    }
}

#define EXPECT_STATUS(word, status) expect_status((word), EffectStatus::status, __LINE__)

// ---------------------------------------------------------------------------

void composition_matches_assembler() {
    // Words produced by the system assembler from the quoted mnemonics.
    CHECK_EQ(add_sub_imm(1, 0, 0, 0, 4, 31, 18), 0x910013F2u);       // add x18, sp, #4
    CHECK_EQ(logical_imm(1, 0, 1, 0, 7, 1, 18), 0x92401C32u);        // and x18, x1, #0xff
    CHECK_EQ(move_wide(1, 3, 3, 1, 18), 0xF2E00032u);                // movk x18, #1, lsl #48
    CHECK_EQ(bitfield(0, 1, 0, 29, 3, 2, 18), 0x331D0C52u);          // bfi w18, w2, #3, #4
    CHECK_EQ(extract(1, 0, 1, 0, 2, 63, 1, 18), 0x93C2FC32u);        // extr x18, x1, x2, #63
    CHECK_EQ(logical_shifted(0, 3, 3, 1, 2, 31, 1, 18), 0x6AE27C32u);  // bics w18, w1, w2, ror #31
    CHECK_EQ(add_sub_shifted(1, 0, 0, 2, 2, 63, 1, 18), 0x8B82FC32u);  // add x18, x1, x2, asr #63
    // add x18, sp, w2, uxtw #4
    CHECK_EQ(add_sub_extended(1, 0, 0, 0, 2, 2, 4, 31, 18), 0x8B2253F2u);
    CHECK_EQ(carry(0, 1, 1, 2, 0, 1, 18), 0x7A020032u);              // sbcs w18, w1, w2
    CHECK_EQ(cond_compare(1, 1, 1, 2, 1, 0, 0, 18, 0, 4), 0xFA421244u);  // ccmp x18, x2, #4, ne
    CHECK_EQ(cond_select(1, 0, 0, 2, 11, 0, 1, 18), 0x9A82B032u);    // csel x18, x1, x2, lt
    CHECK_EQ(two_source(1, 0, 2, 0x17, 1, 18), 0x9AC25C32u);         // crc32cx w18, w1, x2
    CHECK_EQ(one_source(1, 0, 0, 3, 1, 18), 0xDAC00C32u);            // rev x18, x1
    CHECK_EQ(three_source(1, 0, 1, 2, 0, 3, 1, 18), 0x9B220C32u);    // smaddl x18, w1, w2, x3
    CHECK_EQ(ls_unsigned(3, 0, 1, 1, 31, 18), 0xF94007F2u);          // ldr x18, [sp, #8]
    CHECK_EQ(ls_imm9(3, 0, 1, -8, 1, 1, 18), 0xF85F8432u);           // ldr x18, [x1], #-8
    CHECK_EQ(ls_imm9(3, 0, 1, 8, 2, 1, 18), 0xF8408832u);            // ldtr x18, [x1, #8]
    CHECK_EQ(ls_register(1, 0, 2, 18, 6, 1, 2, 1), 0x78B2D841u);     // ldrsh x1, [x2, w18, sxtw #1]
    CHECK_EQ(ls_pair(2, 0, 3, 0, -2, 18, 31, 1), 0xA9BF4BE1u);       // stp x1, x18, [sp, #-16]!
    CHECK_EQ(ls_pair(2, 1, 3, 0, -64, 1, 18, 18), 0xADA00652u);      // stp q18, q1, [x18, #-1024]!
    CHECK_EQ(exclusive(3, 0, 1, 0, 31, 1, 31, 18, 1), 0xC85FFE41u);  // ldaxr x1, [x18]
    CHECK_EQ(exclusive(3, 0, 0, 1, 18, 0, 2, 3, 1), 0xC8320861u);    // stxp w18, x1, x2, [x3]
    CHECK_EQ(exclusive(1, 1, 0, 0, 31, 1, 31, 31, 18), 0x489FFFF2u); // stlrh w18, [sp]
}

void add_subtract_immediate() {
    // add x5, x7, #4: Rd and Rn are SP when 31.
    auto e = EXPECT_EFFECTS(add_sub_imm(1, 0, 0, 0, 4, 7, 5), Operation::add_immediate,
                            Op{0, 5, SP, 64, W, Role::destination},
                            Op{5, 7, SP, 64, R, Role::source});
    CHECK(!e.reads_flags && !e.writes_flags);
    // subs w5, w7, #1, lsl #12: Rd is WZR when 31 (CMP), Rn is WSP.
    e = EXPECT_EFFECTS(add_sub_imm(0, 1, 1, 1, 1, 7, 5), Operation::subs_immediate,
                       Op{0, 5, ZR, 32, W, Role::destination}, Op{5, 7, SP, 32, R, Role::source});
    CHECK(e.writes_flags && !e.reads_flags);
    EXPECT_EFFECTS(add_sub_imm(1, 0, 1, 0, 0, 3, 4), Operation::adds_immediate,
                   Op{0, 4, ZR, 64, W, Role::destination}, Op{5, 3, SP, 64, R, Role::source});
    EXPECT_EFFECTS(add_sub_imm(0, 1, 0, 0, 0xFFF, 3, 4), Operation::sub_immediate,
                   Op{0, 4, SP, 32, W, Role::destination}, Op{5, 3, SP, 32, R, Role::source});

    // mov sp, x18 (ADD #0) writes SP and reads x18.
    e = decode_register_effects(add_sub_imm(1, 0, 0, 0, 0, 18, 31));
    CHECK(writes_stack_pointer(e) && !reads_stack_pointer(e));
    CHECK_EQ(register_fields(e, 18).value_or(0), 0x3E0u);
    // cmp x18, #1: x18 is read, nothing is written.
    e = decode_register_effects(0xF100065Fu);
    CHECK(e.operation == Operation::subs_immediate);
    CHECK_EQ(e.reads, RegisterMask{1} << 18);
    CHECK_EQ(e.writes, 0);
    CHECK(!writes_stack_pointer(e) && !reads_stack_pointer(e));
}

void logical_immediate() {
    // and x5, x7, #0xff: Rd is SP when 31, Rn is XZR.
    EXPECT_EFFECTS(logical_imm(1, 0, 1, 0, 7, 7, 5), Operation::and_immediate,
                   Op{0, 5, SP, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source});
    EXPECT_EFFECTS(logical_imm(0, 1, 0, 1, 0, 7, 5), Operation::orr_immediate,
                   Op{0, 5, SP, 32, W, Role::destination}, Op{5, 7, ZR, 32, R, Role::source});
    EXPECT_EFFECTS(logical_imm(1, 2, 0, 0, 0x3C, 7, 5), Operation::eor_immediate,
                   Op{0, 5, SP, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source});
    // ands (TST when Rd = 31): Rd is the zero register.
    auto e = EXPECT_EFFECTS(logical_imm(0, 3, 0, 1, 1, 7, 5), Operation::ands_immediate,
                            Op{0, 5, ZR, 32, W, Role::destination},
                            Op{5, 7, ZR, 32, R, Role::source});
    CHECK(e.writes_flags);

    // Reserved bitmask immediates and N = 1 in a 32-bit form.
    EXPECT_STATUS(logical_imm(0, 0, 1, 0, 7, 1, 18), reserved);      // sf = 0, N = 1
    EXPECT_STATUS(logical_imm(1, 0, 1, 0, 0x3F, 1, 18), reserved);   // 64 ones
    EXPECT_STATUS(logical_imm(0, 0, 0, 0, 0x1F, 1, 18), reserved);   // 32 ones
    EXPECT_STATUS(logical_imm(1, 1, 0, 0, 0x3D, 1, 18), reserved);   // 2-bit element, all ones
    EXPECT_STATUS(logical_imm(1, 1, 0, 0, 0x3E, 1, 18), reserved);   // element length 1
    EXPECT_STATUS(logical_imm(0, 1, 0, 0, 0x3F, 1, 18), reserved);   // N:NOT(imms) = 0
    // Largest valid runs next to the reserved ones.
    CHECK(decode_register_effects(logical_imm(1, 0, 1, 0, 0x3E, 1, 2)).decoded());
    CHECK(decode_register_effects(logical_imm(1, 0, 0, 0, 0x3C, 1, 2)).decoded());
    CHECK(decode_register_effects(logical_imm(0, 0, 0, 0, 0x1E, 1, 2)).decoded());
}

void move_wide_immediate() {
    EXPECT_EFFECTS(move_wide(1, 0, 3, 0xFFFF, 5), Operation::movn,
                   Op{0, 5, ZR, 64, W, Role::destination});
    EXPECT_EFFECTS(move_wide(0, 2, 1, 7, 5), Operation::movz,
                   Op{0, 5, ZR, 32, W, Role::destination});
    // MOVK keeps the other 48 bits: Rd is read and written.
    EXPECT_EFFECTS(move_wide(1, 3, 2, 7, 5), Operation::movk,
                   Op{0, 5, ZR, 64, RW, Role::destination});
    EXPECT_EFFECTS(move_wide(0, 3, 0, 7, 5), Operation::movk,
                   Op{0, 5, ZR, 32, RW, Role::destination});
    // imm16 bits that look like register fields are not registers.
    const auto e = decode_register_effects(move_wide(1, 2, 0, 18u | 18u << 5 | 18u << 11, 5));
    CHECK(e.decoded() && proven_free_of(e, 18));

    EXPECT_STATUS(move_wide(1, 1, 0, 7, 5), reserved);  // opc = 01
    EXPECT_STATUS(move_wide(0, 2, 2, 7, 5), reserved);  // 32-bit, hw = 2
    EXPECT_STATUS(move_wide(0, 3, 3, 7, 5), reserved);  // 32-bit, hw = 3
}

void bitfield_and_extract() {
    EXPECT_EFFECTS(bitfield(1, 0, 1, 3, 7, 7, 5), Operation::sbfm,
                   Op{0, 5, ZR, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source});
    // BFM (BFI, BFXIL) keeps the unselected bits of Rd.
    EXPECT_EFFECTS(bitfield(0, 1, 0, 29, 3, 7, 5), Operation::bfm,
                   Op{0, 5, ZR, 32, RW, Role::destination}, Op{5, 7, ZR, 32, R, Role::source});
    EXPECT_EFFECTS(bitfield(0, 2, 0, 28, 27, 7, 5), Operation::ubfm,  // lsl w5, w7, #4
                   Op{0, 5, ZR, 32, W, Role::destination}, Op{5, 7, ZR, 32, R, Role::source});
    EXPECT_STATUS(bitfield(1, 0, 0, 3, 7, 7, 5), reserved);   // sf = 1, N = 0
    EXPECT_STATUS(bitfield(0, 0, 1, 3, 7, 7, 5), reserved);   // sf = 0, N = 1
    EXPECT_STATUS(bitfield(0, 2, 0, 32, 7, 7, 5), reserved);  // 32-bit immr >= 32
    EXPECT_STATUS(bitfield(0, 2, 0, 3, 32, 7, 5), reserved);  // 32-bit imms >= 32
    EXPECT_STATUS(bitfield(1, 3, 1, 3, 7, 7, 5), reserved);   // opc = 11

    // extr (ROR immediate when Rn = Rm).
    EXPECT_EFFECTS(extract(1, 0, 1, 0, 9, 63, 7, 5), Operation::extr,
                   Op{0, 5, ZR, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source},
                   Op{16, 9, ZR, 64, R, Role::source});
    EXPECT_EFFECTS(extract(0, 0, 0, 0, 9, 31, 7, 5), Operation::extr,
                   Op{0, 5, ZR, 32, W, Role::destination}, Op{5, 7, ZR, 32, R, Role::source},
                   Op{16, 9, ZR, 32, R, Role::source});
    EXPECT_STATUS(extract(0, 0, 0, 0, 9, 32, 7, 5), reserved);  // 32-bit lsb >= 32
    EXPECT_STATUS(extract(1, 0, 0, 0, 9, 3, 7, 5), reserved);   // sf = 1, N = 0
    EXPECT_STATUS(extract(0, 0, 1, 0, 9, 3, 7, 5), reserved);   // sf = 0, N = 1
    EXPECT_STATUS(extract(1, 0, 1, 1, 9, 3, 7, 5), unknown);    // o0 = 1
    EXPECT_STATUS(extract(1, 1, 1, 0, 9, 3, 7, 5), unknown);    // op21 != 00
}

void logical_and_add_sub_shifted() {
    static constexpr Operation logical[] = {
        Operation::and_shifted, Operation::bic_shifted, Operation::orr_shifted,
        Operation::orn_shifted, Operation::eor_shifted, Operation::eon_shifted,
        Operation::ands_shifted, Operation::bics_shifted,
    };
    for (unsigned opc = 0; opc < 4; ++opc) {
        for (unsigned n = 0; n < 2; ++n) {
            for (unsigned sf = 0; sf < 2; ++sf) {
                const unsigned bits = sf ? 64 : 32;
                const auto e = EXPECT_EFFECTS(logical_shifted(sf, opc, 3, n, 9, 31, 7, 5),
                                              logical[opc * 2 + n],
                                              Op{0, 5, ZR, static_cast<std::uint8_t>(bits), W,
                                                 Role::destination},
                                              Op{5, 7, ZR, static_cast<std::uint8_t>(bits), R,
                                                 Role::source},
                                              Op{16, 9, ZR, static_cast<std::uint8_t>(bits), R,
                                                 Role::source});
                CHECK(e.writes_flags == (opc == 3));
            }
        }
    }
    // mov x5, x9 (ORR with Rn = XZR) reads only x9.
    auto e = decode_register_effects(logical_shifted(1, 1, 0, 0, 9, 0, 31, 5));
    CHECK_EQ(e.reads, RegisterMask{1} << 9);
    EXPECT_STATUS(logical_shifted(0, 0, 0, 0, 9, 32, 7, 5), reserved);  // 32-bit imm6 >= 32

    static constexpr Operation arithmetic[] = {Operation::add_shifted, Operation::adds_shifted,
                                               Operation::sub_shifted, Operation::subs_shifted};
    for (unsigned op = 0; op < 2; ++op) {
        for (unsigned s = 0; s < 2; ++s) {
            e = EXPECT_EFFECTS(add_sub_shifted(1, op, s, 2, 9, 63, 7, 5), arithmetic[op * 2 + s],
                               Op{0, 5, ZR, 64, W, Role::destination},
                               Op{5, 7, ZR, 64, R, Role::source},
                               Op{16, 9, ZR, 64, R, Role::source});
            CHECK(e.writes_flags == (s == 1));
            EXPECT_EFFECTS(add_sub_shifted(0, op, s, 0, 9, 31, 7, 5), arithmetic[op * 2 + s],
                           Op{0, 5, ZR, 32, W, Role::destination},
                           Op{5, 7, ZR, 32, R, Role::source}, Op{16, 9, ZR, 32, R, Role::source});
        }
    }
    EXPECT_STATUS(add_sub_shifted(1, 0, 0, 3, 9, 1, 7, 5), reserved);   // ROR shift
    EXPECT_STATUS(add_sub_shifted(0, 0, 0, 0, 9, 32, 7, 5), reserved);  // 32-bit imm6 >= 32
}

void add_sub_extended_register() {
    // add x5, x7, w9, uxtw #4: Rd and Rn are SP when 31, Rm is a W register.
    EXPECT_EFFECTS(add_sub_extended(1, 0, 0, 0, 9, 2, 4, 7, 5), Operation::add_extended,
                   Op{0, 5, SP, 64, W, Role::destination}, Op{5, 7, SP, 64, R, Role::source},
                   Op{16, 9, ZR, 32, R, Role::source});
    // sub x5, x7, x9, sxtx: UXTX/SXTX name an X register.
    EXPECT_EFFECTS(add_sub_extended(1, 1, 0, 0, 9, 7, 0, 7, 5), Operation::sub_extended,
                   Op{0, 5, SP, 64, W, Role::destination}, Op{5, 7, SP, 64, R, Role::source},
                   Op{16, 9, ZR, 64, R, Role::source});
    EXPECT_EFFECTS(add_sub_extended(1, 0, 0, 0, 9, 3, 2, 7, 5), Operation::add_extended,
                   Op{0, 5, SP, 64, W, Role::destination}, Op{5, 7, SP, 64, R, Role::source},
                   Op{16, 9, ZR, 64, R, Role::source});
    // adds/subs: Rd is the zero register (CMN/CMP) but Rn is still SP.
    auto e = EXPECT_EFFECTS(add_sub_extended(1, 1, 1, 0, 9, 1, 2, 7, 5), Operation::subs_extended,
                            Op{0, 5, ZR, 64, W, Role::destination},
                            Op{5, 7, SP, 64, R, Role::source},
                            Op{16, 9, ZR, 32, R, Role::source});
    CHECK(e.writes_flags);
    // 32-bit forms always name W registers, even with option UXTX.
    EXPECT_EFFECTS(add_sub_extended(0, 0, 1, 0, 9, 3, 0, 7, 5), Operation::adds_extended,
                   Op{0, 5, ZR, 32, W, Role::destination}, Op{5, 7, SP, 32, R, Role::source},
                   Op{16, 9, ZR, 32, R, Role::source});
    // cmp sp, w18, uxth #2 reads SP and x18 and writes nothing.
    e = decode_register_effects(0xEB322BFFu);
    CHECK(e.operation == Operation::subs_extended);
    CHECK(reads_stack_pointer(e) && !writes_stack_pointer(e));
    CHECK_EQ(e.reads, RegisterMask{1} << 18);
    CHECK_EQ(e.writes, 0);

    EXPECT_STATUS(add_sub_extended(1, 0, 0, 0, 9, 2, 5, 7, 5), reserved);  // shift 5
    EXPECT_STATUS(add_sub_extended(1, 0, 0, 0, 9, 2, 7, 7, 5), reserved);  // shift 7
    EXPECT_STATUS(add_sub_extended(1, 0, 0, 1, 9, 2, 0, 7, 5), unknown);   // opt = 01
    EXPECT_STATUS(add_sub_extended(1, 0, 0, 2, 9, 2, 0, 7, 5), unknown);   // opt = 10
}

void carry_compare_select() {
    static constexpr Operation carries[] = {Operation::adc, Operation::adcs, Operation::sbc,
                                            Operation::sbcs};
    for (unsigned op = 0; op < 2; ++op) {
        for (unsigned s = 0; s < 2; ++s) {
            const auto e = EXPECT_EFFECTS(carry(1, op, s, 9, 0, 7, 5), carries[op * 2 + s],
                                          Op{0, 5, ZR, 64, W, Role::destination},
                                          Op{5, 7, ZR, 64, R, Role::source},
                                          Op{16, 9, ZR, 64, R, Role::source});
            CHECK(e.reads_flags && e.writes_flags == (s == 1));
        }
    }
    EXPECT_STATUS(carry(1, 0, 0, 9, 1, 7, 5), unknown);     // RMIF space
    EXPECT_STATUS(carry(0, 0, 1, 9, 2, 7, 13), unknown);    // SETF space
    EXPECT_STATUS(carry(1, 0, 0, 9, 0x08, 7, 5), unknown);  // checked-pointer space

    // ccmp x7, x9, #4, ne: both registers are read only if the condition holds.
    auto e = EXPECT_EFFECTS(cond_compare(1, 1, 1, 9, 1, 0, 0, 7, 0, 4), Operation::ccmp_register,
                            Op{5, 7, ZR, 64, R, Role::source, true},
                            Op{16, 9, ZR, 64, R, Role::source, true});
    CHECK(e.reads_flags && e.writes_flags);
    EXPECT_EFFECTS(cond_compare(0, 0, 1, 9, 14, 0, 0, 7, 0, 15), Operation::ccmn_register,
                   Op{5, 7, ZR, 32, R, Role::source, true},
                   Op{16, 9, ZR, 32, R, Role::source, true});
    // Immediate forms: bits 20..16 are imm5, not a register.
    e = EXPECT_EFFECTS(cond_compare(1, 1, 1, 18, 0, 1, 0, 7, 0, 0), Operation::ccmp_immediate,
                       Op{5, 7, ZR, 64, R, Role::source, true});
    CHECK(proven_free_of(e, 18));
    EXPECT_EFFECTS(cond_compare(0, 0, 1, 31, 0, 1, 0, 7, 0, 0), Operation::ccmn_immediate,
                   Op{5, 7, ZR, 32, R, Role::source, true});
    EXPECT_STATUS(cond_compare(1, 1, 0, 9, 1, 0, 0, 7, 0, 4), reserved);  // S = 0
    EXPECT_STATUS(cond_compare(1, 1, 1, 9, 1, 0, 1, 7, 0, 4), reserved);  // o2 = 1
    EXPECT_STATUS(cond_compare(1, 1, 1, 9, 1, 1, 0, 7, 1, 4), reserved);  // o3 = 1

    static constexpr Operation selects[] = {Operation::csel, Operation::csinc, Operation::csinv,
                                            Operation::csneg};
    for (unsigned op = 0; op < 2; ++op) {
        for (unsigned op2 = 0; op2 < 2; ++op2) {
            e = EXPECT_EFFECTS(cond_select(1, op, 0, 9, 11, op2, 7, 5), selects[op * 2 + op2],
                               Op{0, 5, ZR, 64, W, Role::destination},
                               Op{5, 7, ZR, 64, R, Role::source, true},
                               Op{16, 9, ZR, 64, R, Role::source, true});
            CHECK(e.reads_flags && !e.writes_flags);
        }
    }
    // cset x18, hi (CSINC x18, xzr, xzr, ls) writes x18 and reads nothing.
    e = decode_register_effects(0x9A9F97F2u);
    CHECK(e.operation == Operation::csinc);
    CHECK_EQ(e.reads, 0);
    CHECK_EQ(e.writes, RegisterMask{1} << 18);
    EXPECT_STATUS(cond_select(1, 0, 1, 9, 11, 0, 7, 5), reserved);  // S = 1
    EXPECT_STATUS(cond_select(1, 0, 0, 9, 11, 2, 7, 5), reserved);  // op2 = 10
    EXPECT_STATUS(cond_select(1, 1, 0, 9, 11, 3, 7, 5), reserved);  // op2 = 11
}

void one_two_three_source() {
    struct TwoSource {
        unsigned opcode;
        Operation operation;
    };
    static constexpr TwoSource shifts[] = {
        {0x02, Operation::udiv}, {0x03, Operation::sdiv}, {0x08, Operation::lslv},
        {0x09, Operation::lsrv}, {0x0A, Operation::asrv}, {0x0B, Operation::rorv},
    };
    for (const auto& entry : shifts) {
        for (unsigned sf = 0; sf < 2; ++sf) {
            const auto bits = static_cast<std::uint8_t>(sf ? 64 : 32);
            EXPECT_EFFECTS(two_source(sf, 0, 9, entry.opcode, 7, 5), entry.operation,
                           Op{0, 5, ZR, bits, W, Role::destination},
                           Op{5, 7, ZR, bits, R, Role::source},
                           Op{16, 9, ZR, bits, R, Role::source});
        }
    }
    static constexpr Operation crcs[] = {
        Operation::crc32b, Operation::crc32h, Operation::crc32w, Operation::crc32x,
        Operation::crc32cb, Operation::crc32ch, Operation::crc32cw, Operation::crc32cx,
    };
    for (unsigned index = 0; index < 8; ++index) {
        const unsigned size = index & 3;
        const unsigned sf = size == 3 ? 1 : 0;
        EXPECT_EFFECTS(two_source(sf, 0, 9, 0x10 | index, 7, 5), crcs[index],
                       Op{0, 5, ZR, 32, W, Role::destination}, Op{5, 7, ZR, 32, R, Role::source},
                       Op{16, 9, ZR, static_cast<std::uint8_t>(size == 3 ? 64 : 32), R,
                          Role::source});
        EXPECT_STATUS(two_source(sf ^ 1, 0, 9, 0x10 | index, 7, 5), reserved);
    }
    EXPECT_STATUS(two_source(1, 0, 9, 0x00, 7, 5), unknown);  // SUBP (memory tagging)
    EXPECT_STATUS(two_source(1, 0, 9, 0x04, 7, 5), unknown);  // IRG
    EXPECT_EFFECTS(two_source(1, 0, 9, 0x0C, 7, 5), Operation::pacga,
                   Op{0, 5, ZR, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source},
                   Op{16, 9, SP, 64, R, Role::source});
    EXPECT_STATUS(two_source(1, 0, 9, 0x18, 7, 5), unknown);  // SMAX (register)
    EXPECT_STATUS(two_source(1, 1, 9, 0x02, 7, 5), unknown);  // S = 1

    struct OneSource {
        unsigned sf;
        unsigned opcode;
        Operation operation;
    };
    static constexpr OneSource ones[] = {
        {0, 0, Operation::rbit}, {1, 0, Operation::rbit}, {0, 1, Operation::rev16},
        {1, 1, Operation::rev16}, {0, 2, Operation::rev}, {1, 2, Operation::rev32},
        {1, 3, Operation::rev}, {0, 4, Operation::clz}, {1, 4, Operation::clz},
        {0, 5, Operation::cls}, {1, 5, Operation::cls},
    };
    for (const auto& entry : ones) {
        const auto bits = static_cast<std::uint8_t>(entry.sf ? 64 : 32);
        EXPECT_EFFECTS(one_source(entry.sf, 0, 0, entry.opcode, 7, 5), entry.operation,
                       Op{0, 5, ZR, bits, W, Role::destination},
                       Op{5, 7, ZR, bits, R, Role::source});
    }
    EXPECT_STATUS(one_source(0, 0, 0, 3, 7, 5), reserved);  // REV with sf = 0, opc = 11
    EXPECT_STATUS(one_source(1, 0, 0, 6, 7, 5), unknown);   // CTZ (CSSC)
    EXPECT_EFFECTS(one_source(1, 0, 1, 0, 7, 5), Operation::pacia,
                   Op{0, 5, ZR, 64, RW, Role::destination}, Op{5, 7, SP, 64, R, Role::source});
    EXPECT_STATUS(one_source(1, 1, 0, 0, 7, 5), unknown);   // S = 1

    // madd/msub: Ra is an accumulator in bits 14..10.
    EXPECT_EFFECTS(three_source(1, 0, 0, 9, 0, 11, 7, 5), Operation::madd,
                   Op{0, 5, ZR, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source},
                   Op{16, 9, ZR, 64, R, Role::source}, Op{10, 11, ZR, 64, R, Role::accumulator});
    EXPECT_EFFECTS(three_source(0, 0, 0, 9, 1, 11, 7, 5), Operation::msub,
                   Op{0, 5, ZR, 32, W, Role::destination}, Op{5, 7, ZR, 32, R, Role::source},
                   Op{16, 9, ZR, 32, R, Role::source}, Op{10, 11, ZR, 32, R, Role::accumulator});
    // mul x5, x7, x9 is madd with Ra = XZR.
    auto e = decode_register_effects(three_source(1, 0, 0, 9, 0, 31, 7, 5));
    CHECK_EQ(e.reads, (RegisterMask{1} << 7) | (RegisterMask{1} << 9));
    struct Long {
        unsigned op31;
        unsigned o0;
        Operation operation;
    };
    static constexpr Long longs[] = {
        {1, 0, Operation::smaddl}, {1, 1, Operation::smsubl},
        {5, 0, Operation::umaddl}, {5, 1, Operation::umsubl},
    };
    for (const auto& entry : longs) {
        EXPECT_EFFECTS(three_source(1, 0, entry.op31, 9, entry.o0, 11, 7, 5), entry.operation,
                       Op{0, 5, ZR, 64, W, Role::destination}, Op{5, 7, ZR, 32, R, Role::source},
                       Op{16, 9, ZR, 32, R, Role::source},
                       Op{10, 11, ZR, 64, R, Role::accumulator});
        EXPECT_STATUS(three_source(0, 0, entry.op31, 9, entry.o0, 11, 7, 5), reserved);
    }
    EXPECT_EFFECTS(three_source(1, 0, 2, 9, 0, 31, 7, 5), Operation::smulh,
                   Op{0, 5, ZR, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source},
                   Op{16, 9, ZR, 64, R, Role::source});
    EXPECT_EFFECTS(three_source(1, 0, 6, 9, 0, 31, 7, 5), Operation::umulh,
                   Op{0, 5, ZR, 64, W, Role::destination}, Op{5, 7, ZR, 64, R, Role::source},
                   Op{16, 9, ZR, 64, R, Role::source});
    // SMULH Ra is should-be-one: other values are CONSTRAINED UNPREDICTABLE,
    // and an x18-looking value there is never treated as an operand.
    e = decode_register_effects(three_source(1, 0, 2, 9, 0, 18, 7, 5));
    CHECK(e.status == EffectStatus::unpredictable);
    CHECK(has(e.notes, EffectNote::should_be_one_field));
    CHECK(!proven_free_of(e, 18) && !proven_free_of(e, 1));
    EXPECT_STATUS(three_source(0, 0, 2, 9, 0, 31, 7, 5), reserved);  // 32-bit SMULH
    EXPECT_STATUS(three_source(1, 0, 2, 9, 1, 31, 7, 5), unknown);
    EXPECT_STATUS(three_source(1, 0, 3, 9, 0, 11, 7, 5), unknown);
    EXPECT_STATUS(three_source(1, 1, 0, 9, 0, 11, 7, 5), unknown);   // op54 != 00
}

void hints_and_barriers() {
    for (std::uint32_t word : {0xD503201Fu, 0xD5033BBFu, 0xD5033F9Fu, 0xD5033FDFu, 0xD5033F5Fu}) {
        const auto e = decode_register_effects(word);
        CHECK(e.decoded() && e.operand_count == 0 && proven_free_of(e, 18));
    }
    CHECK(decode_register_effects(0xD503201Fu).operation == Operation::nop);
    CHECK(decode_register_effects(0xD5033BBFu).operation == Operation::dmb);
    CHECK(decode_register_effects(0xD5033F9Fu).operation == Operation::dsb);
    CHECK(decode_register_effects(0xD5033FDFu).operation == Operation::isb);
    CHECK(decode_register_effects(0xD5033F5Fu).operation == Operation::clrex);
    EXPECT_STATUS(0xD503233Fu, unknown);  // PACIASP: reads SP, writes x30
    EXPECT_STATUS(0xD503241Fu, unknown);  // BTI
    EXPECT_STATUS(0xD5033BB2u, unknown);  // DMB encoding with Rt = 18
    EXPECT_STATUS(0xD50330FFu, unknown);  // SB
    EXPECT_STATUS(0xD503223Fu, unknown);  // PSB CSYNC
}

void single_register_loads_and_stores() {
    // ldr x5, [x7, #8]: base is SP when 31, transfer is XZR when 31.
    auto e = EXPECT_EFFECTS(ls_unsigned(3, 0, 1, 1, 7, 5), Operation::ldr,
                            Op{0, 5, ZR, 64, W, Role::transfer}, Op{5, 7, SP, 64, R, Role::base});
    CHECK(e.memory.operation == MemoryOperation::load);
    CHECK(e.memory.addressing == AddressingMode::unsigned_offset);
    CHECK_EQ(e.memory.offset, 8);
    CHECK_EQ(e.memory.element_bytes, 8);
    CHECK(!e.memory.writeback);

    struct Form {
        unsigned size;
        unsigned opc;
        Operation operation;
        std::uint8_t bits;
        Access access;
        bool sign;
    };
    static constexpr Form forms[] = {
        {0, 0, Operation::strb, 32, R, false}, {0, 1, Operation::ldrb, 32, W, false},
        {0, 2, Operation::ldrsb, 64, W, true}, {0, 3, Operation::ldrsb, 32, W, true},
        {1, 0, Operation::strh, 32, R, false}, {1, 1, Operation::ldrh, 32, W, false},
        {1, 2, Operation::ldrsh, 64, W, true}, {1, 3, Operation::ldrsh, 32, W, true},
        {2, 0, Operation::str, 32, R, false},  {2, 1, Operation::ldr, 32, W, false},
        {2, 2, Operation::ldrsw, 64, W, true}, {3, 0, Operation::str, 64, R, false},
        {3, 1, Operation::ldr, 64, W, false},
    };
    for (const auto& form : forms) {
        const unsigned bytes = 1u << form.size;
        e = EXPECT_EFFECTS(ls_unsigned(form.size, 0, form.opc, 3, 7, 5), form.operation,
                           Op{0, 5, ZR, form.bits, form.access, Role::transfer},
                           Op{5, 7, SP, 64, R, Role::base});
        CHECK_EQ(e.memory.offset, 3 * bytes);
        CHECK_EQ(e.memory.element_bytes, bytes);
        CHECK(has(e.memory.attributes, MemoryAttribute::sign_extend) == form.sign);
        // Unscaled, unprivileged, pre-index and post-index forms.
        e = EXPECT_EFFECTS(ls_imm9(form.size, 0, form.opc, -256, 0, 7, 5), form.operation,
                           Op{0, 5, ZR, form.bits, form.access, Role::transfer},
                           Op{5, 7, SP, 64, R, Role::base});
        CHECK(e.memory.addressing == AddressingMode::unscaled_offset);
        CHECK_EQ(e.memory.offset, -256);
        e = EXPECT_EFFECTS(ls_imm9(form.size, 0, form.opc, 255, 2, 7, 5), form.operation,
                           Op{0, 5, ZR, form.bits, form.access, Role::transfer},
                           Op{5, 7, SP, 64, R, Role::base});
        CHECK(has(e.memory.attributes, MemoryAttribute::unprivileged));
        CHECK_EQ(e.memory.offset, 255);
        for (unsigned kind : {1u, 3u}) {
            e = EXPECT_EFFECTS(ls_imm9(form.size, 0, form.opc, -16, kind, 7, 5), form.operation,
                               Op{0, 5, ZR, form.bits, form.access, Role::transfer},
                               Op{5, 7, SP, 64, RW, Role::base});
            CHECK(e.memory.writeback);
            CHECK(e.memory.addressing ==
                  (kind == 1 ? AddressingMode::post_index : AddressingMode::pre_index));
            CHECK_EQ(e.memory.offset, -16);
        }
        // Register offset: Rm is the zero register when 31.
        e = EXPECT_EFFECTS(ls_register(form.size, 0, form.opc, 9, 6, 1, 7, 5), form.operation,
                           Op{0, 5, ZR, form.bits, form.access, Role::transfer},
                           Op{5, 7, SP, 64, R, Role::base}, Op{16, 9, ZR, 32, R, Role::index});
        CHECK(e.memory.extend == IndexExtend::sxtw);
        CHECK_EQ(e.memory.index_shift, form.size);
    }
    // Register-offset extends and the X-register index forms.
    e = decode_register_effects(ls_register(3, 0, 1, 9, 3, 0, 7, 5));
    CHECK(e.memory.extend == IndexExtend::lsl && e.memory.index_shift == 0);
    CHECK_EQ(e.operands[2].bits, 64);
    e = decode_register_effects(ls_register(3, 0, 1, 9, 2, 1, 7, 5));
    CHECK(e.memory.extend == IndexExtend::uxtw && e.memory.index_shift == 3);
    CHECK_EQ(e.operands[2].bits, 32);
    e = decode_register_effects(ls_register(3, 0, 1, 9, 7, 1, 7, 5));
    CHECK(e.memory.extend == IndexExtend::sxtx && e.memory.index_shift == 3);
    CHECK_EQ(e.operands[2].bits, 64);
    for (unsigned option : {0u, 1u, 4u, 5u}) {
        EXPECT_STATUS(ls_register(3, 0, 1, 9, option, 0, 7, 5), reserved);
    }

    // Prefetch: Rt is the prefetch operation, never a register.
    e = EXPECT_EFFECTS(ls_unsigned(3, 0, 2, 2, 7, 18), Operation::prfm,
                       Op{5, 7, SP, 64, R, Role::base});
    CHECK(e.memory.operation == MemoryOperation::prefetch);
    CHECK_EQ(e.memory.offset, 16);
    CHECK(proven_free_of(e, 18));
    e = EXPECT_EFFECTS(ls_imm9(3, 0, 2, -1, 0, 7, 18), Operation::prfm,
                       Op{5, 7, SP, 64, R, Role::base});
    CHECK(e.memory.addressing == AddressingMode::unscaled_offset && proven_free_of(e, 18));
    e = EXPECT_EFFECTS(ls_register(3, 0, 2, 9, 3, 1, 7, 18), Operation::prfm,
                       Op{5, 7, SP, 64, R, Role::base}, Op{16, 9, ZR, 64, R, Role::index});
    CHECK_EQ(e.memory.index_shift, 3);
    // Range-prefetch extensions do not share PRFM's index-width semantics.
    EXPECT_STATUS(0xF8AADB7Bu, unknown);  // rprfm #0x2b, x10, [x27]
    for (unsigned prefetch = 24; prefetch < 32; ++prefetch) {
        for (unsigned option : {2u, 3u, 6u, 7u}) {
            const auto range = decode_register_effects(
                ls_register(3, 0, 2, 18, option, 1, 7, prefetch));
            CHECK(range.status == EffectStatus::unknown);
            CHECK(!proven_free_of(range, 18));
        }
    }
    EXPECT_STATUS(ls_imm9(3, 0, 2, 8, 1, 7, 5), reserved);   // prefetch post-index
    EXPECT_STATUS(ls_imm9(3, 0, 2, 8, 3, 7, 5), reserved);   // prefetch pre-index
    EXPECT_STATUS(ls_imm9(3, 0, 2, 8, 2, 7, 5), reserved);   // prefetch unprivileged
    for (unsigned kind : {0u, 1u, 2u, 3u}) {
        EXPECT_STATUS(ls_imm9(2, 0, 3, 8, kind, 7, 5), reserved);  // size 10, opc 11
        EXPECT_STATUS(ls_imm9(3, 0, 3, 8, kind, 7, 5), reserved);  // size 11, opc 11
    }
    EXPECT_STATUS(ls_unsigned(2, 0, 3, 1, 7, 5), reserved);
    EXPECT_STATUS(ls_unsigned(3, 0, 3, 1, 7, 5), reserved);
    // Atomic memory operations and pointer-authenticated loads share the
    // class space but are not decoded.
    EXPECT_STATUS(ls_register(3, 0, 0, 9, 0, 0, 7, 5) & ~0x800u, unknown);  // LDADD
    EXPECT_STATUS(0xF8200425u, unknown);  // LDRAA x5, [x1]
}

void vector_transfers() {
    struct Form {
        unsigned size;
        unsigned opc;
        std::uint8_t bits;
    };
    static constexpr Form forms[] = {{0, 0, 8}, {0, 1, 8}, {1, 0, 16}, {1, 1, 16}, {2, 0, 32},
                                     {2, 1, 32}, {3, 0, 64}, {3, 1, 64}, {0, 2, 128}, {0, 3, 128}};
    for (const auto& form : forms) {
        const bool load = (form.opc & 1) != 0;
        const Operation operation = load ? Operation::ldr_vector : Operation::str_vector;
        const Access access = load ? W : R;
        // ldr q18, [x7, #...]: V18 is not x18.
        auto e = EXPECT_EFFECTS(ls_unsigned(form.size, 1, form.opc, 2, 7, 18), operation,
                                Op{0, 18, NA, form.bits, access, Role::transfer, false, VEC},
                                Op{5, 7, SP, 64, R, Role::base});
        CHECK(proven_free_of(e, 18));
        CHECK_EQ(e.memory.offset, 2 * (form.bits / 8));
        EXPECT_EFFECTS(ls_imm9(form.size, 1, form.opc, -3, 3, 7, 18), operation,
                       Op{0, 18, NA, form.bits, access, Role::transfer, false, VEC},
                       Op{5, 7, SP, 64, RW, Role::base});
        e = EXPECT_EFFECTS(ls_register(form.size, 1, form.opc, 9, 3, 1, 7, 18), operation,
                           Op{0, 18, NA, form.bits, access, Role::transfer, false, VEC},
                           Op{5, 7, SP, 64, R, Role::base}, Op{16, 9, ZR, 64, R, Role::index});
        CHECK_EQ(e.memory.index_shift, form.bits == 8 ? 0 : (form.bits == 16  ? 1
                                                              : form.bits == 32 ? 2
                                                              : form.bits == 64 ? 3
                                                                                : 4));
        // Writeback with the same number in both files is not an overlap.
        e = decode_register_effects(ls_imm9(form.size, 1, form.opc, 16, 1, 18, 18));
        CHECK(e.decoded());
        EXPECT_STATUS(ls_imm9(form.size, 1, form.opc, 8, 2, 7, 5), reserved);  // unprivileged
    }
    for (unsigned size = 1; size < 4; ++size) {
        EXPECT_STATUS(ls_unsigned(size, 1, 2, 1, 7, 5), reserved);
        EXPECT_STATUS(ls_unsigned(size, 1, 3, 1, 7, 5), reserved);
    }
}

void scalar_fp_moves() {
    // The helper rejects a word that names x18, so its field probes use a
    // non-18 general register and the x18 forms are checked directly below.
    EXPECT_EFFECTS(0x1E2600A3u, Operation::fmov_general,  // fmov w3, s5
                   Op{5, 5, NA, 32, R, Role::source, false, VEC},
                   Op{0, 3, ZR, 32, W, Role::destination});
    EXPECT_EFFECTS(0x1E270065u, Operation::fmov_general,  // fmov s5, w3
                   Op{5, 3, ZR, 32, R, Role::source},
                   Op{0, 5, NA, 32, W, Role::destination, false, VEC});
    EXPECT_EFFECTS(0x9E6600A3u, Operation::fmov_general,  // fmov x3, d5
                   Op{5, 5, NA, 64, R, Role::source, false, VEC},
                   Op{0, 3, ZR, 64, W, Role::destination});
    EXPECT_EFFECTS(0x9E670065u, Operation::fmov_general,  // fmov d5, x3
                   Op{5, 3, ZR, 64, R, Role::source},
                   Op{0, 5, NA, 64, W, Role::destination, false, VEC});
    EXPECT_EFFECTS(0x1EE600A3u, Operation::fmov_general,  // fmov w3, h5
                   Op{5, 5, NA, 16, R, Role::source, false, VEC},
                   Op{0, 3, ZR, 32, W, Role::destination});
    EXPECT_EFFECTS(0x1EE70065u, Operation::fmov_general,  // fmov h5, w3
                   Op{5, 3, ZR, 32, R, Role::source},
                   Op{0, 5, NA, 16, W, Role::destination, false, VEC});

    // Scalar FCVT to a general register. The SDK's float-to-int conversions use
    // round-toward-minus-infinity and round-toward-plus-infinity forms; the
    // destination width follows the sf bit.
    EXPECT_EFFECTS(0x1E300043u, Operation::fcvt_general,  // fcvtms w3, s2
                   Op{5, 2, NA, 32, R, Role::source, false, VEC},
                   Op{0, 3, ZR, 32, W, Role::destination});
    EXPECT_EFFECTS(0x1E2800B1u, Operation::fcvt_general,  // fcvtps w17, s5
                   Op{5, 5, NA, 32, R, Role::source, false, VEC},
                   Op{0, 17, ZR, 32, W, Role::destination});
    EXPECT_EFFECTS(0x9E780043u, Operation::fcvt_general,  // fcvtzs x3, d2
                   Op{5, 2, NA, 64, R, Role::source, false, VEC},
                   Op{0, 3, ZR, 64, W, Role::destination});
    EXPECT_EFFECTS(0x1E700043u, Operation::fcvt_general,  // fcvtms w3, d2
                   Op{5, 2, NA, 64, R, Role::source, false, VEC},
                   Op{0, 3, ZR, 32, W, Role::destination});
    EXPECT_EFFECTS(0x0E040E02u, Operation::dup_general,  // dup v2.2s, w16
                   Op{5, 16, ZR, 32, R, Role::source},
                   Op{0, 2, NA, 64, W, Role::destination, false, VEC});
    EXPECT_EFFECTS(0x4E040E02u, Operation::dup_general,  // dup v2.4s, w16
                   Op{5, 16, ZR, 32, R, Role::source},
                   Op{0, 2, NA, 128, W, Role::destination, false, VEC});
    EXPECT_EFFECTS(0x1EE00043u, Operation::fcvt_general,  // fcvtns w3, h2
                   Op{5, 2, NA, 16, R, Role::source, false, VEC},
                   Op{0, 3, ZR, 32, W, Role::destination});

    // x18 forms: the vector register does not enter the general mask and the
    // general register is the only x18 use.
    auto e = decode_register_effects(0x1E2600B2u);  // fmov w18, s5
    CHECK(e.decoded() && e.operation == Operation::fmov_general && e.operand_count == 2);
    CHECK(e.operands[0].file == VEC && e.operands[0].number == 5 && e.operands[0].bits == 32);
    CHECK(e.operands[1].file == GPR && e.operands[1].number == 18 && e.operands[1].bits == 32);
    CHECK_EQ(e.reads, 0);
    CHECK_EQ(e.writes, std::uint32_t{1} << 18);
    e = decode_register_effects(0x1E270245u);  // fmov s5, w18
    CHECK(e.decoded() && e.operation == Operation::fmov_general && e.operand_count == 2);
    CHECK(e.operands[0].file == GPR && e.operands[0].number == 18 && e.operands[0].bits == 32);
    CHECK(e.operands[1].file == VEC && e.operands[1].number == 5 && e.operands[1].bits == 32);
    CHECK_EQ(e.reads, std::uint32_t{1} << 18);
    CHECK_EQ(e.writes, 0);
    CHECK(proven_free_of(e, 3));

    // Substitution renames only the general-register field.
    auto sub = substitute_register(0x1E2600B2u, 18, 3, UnchangedWord::reject);
    CHECK(sub.ok() && sub.changed());
    CHECK_EQ(sub.word, 0x1E2600A3u);
    sub = substitute_register(0x1E270245u, 18, 3, UnchangedWord::reject);
    CHECK(sub.ok() && sub.changed());
    CHECK_EQ(sub.word, 0x1E270065u);
}

void structure_encoding_boundaries() {
    // Each allocated multiple-structure opcode names this many vector registers.
    constexpr unsigned counts[] = {4, 0, 4, 0, 3, 0, 3, 1, 2, 0, 2, 0, 0, 0, 0, 0};
    for (unsigned q = 0; q < 2; ++q) {
        for (unsigned load = 0; load < 2; ++load) {
            for (unsigned post = 0; post < 2; ++post) {
                for (unsigned opcode = 0; opcode < 16; ++opcode) {
                    for (unsigned size = 0; size < 4; ++size) {
                        const auto word = 0x0C000000u | q << 30 | post << 23 | load << 22 |
                            (post ? 18u << 16 : 0) | opcode << 12 | size << 10 | 18u << 5 | 31u;
                        const bool valid = counts[opcode] != 0 &&
                            !(q == 0 && size == 3 && (opcode == 0 || opcode == 4 || opcode == 8));
                        if (!valid) {
                            EXPECT_STATUS(word, reserved);
                            CHECK(substitute_register(word, 18, 9, UnchangedWord::reject).error ==
                                  SubstitutionError::reserved_encoding);
                            continue;
                        }
                        const auto e = decode_register_effects(word);
                        CHECK(e.decoded());
                        CHECK_EQ(e.memory.elements, counts[opcode]);
                        CHECK_EQ(e.memory.element_bytes, q ? 16 : 8);
                        CHECK_EQ(e.reads, register_bit(18));
                        CHECK_EQ(e.writes, post ? register_bit(18) : 0);
                        CHECK_EQ(e.operand_count, counts[opcode] + 1 + post);
                        CHECK(register_fields(e, 18) == std::optional<std::uint32_t>{
                            (0x1Fu << 5) | (post ? 0x1Fu << 16 : 0)});
                        for (unsigned i = 0; i < counts[opcode]; ++i) {
                            CHECK(e.operands[i].file == VEC);
                            CHECK_EQ(e.operands[i].number, (31 + i) % 32);
                            CHECK_EQ(e.operands[i].field, i == 0 ? 0 : implicit_operand);
                        }
                        CHECK(substitute_register(word, 18, 9, UnchangedWord::reject).ok());
                        EXPECT_STATUS(word | 1u << 21, unknown);
                        if (!post) EXPECT_STATUS(word | 1u << 16, unknown);
                    }
                }
            }
        }
    }
    // Single structures: lane selector constraints apply independently of Q,
    // R (register count), L and the two post-index increment encodings.
    for (unsigned opcode = 0; opcode < 8; ++opcode) {
        for (unsigned size = 0; size < 4; ++size) {
            for (unsigned sbit = 0; sbit < 2; ++sbit) {
                for (unsigned load = 0; load < 2; ++load) {
                    const bool valid = opcode < 2 ||
                        (opcode < 4 && (size == 0 || size == 2)) ||
                        (opcode < 6 && opcode >= 4 && (size == 0 || (size == 1 && sbit == 0))) ||
                        (opcode >= 6 && load == 1 && sbit == 0);
                    for (unsigned q = 0; q < 2; ++q) {
                        for (unsigned r = 0; r < 2; ++r) {
                            for (unsigned mode = 0; mode < 3; ++mode) {
                                const auto word = 0x0D000000u | q << 30 | load << 22 | r << 21 |
                                    opcode << 13 | sbit << 12 | size << 10 | 31u << 5 | 18u |
                                    (mode ? 1u << 23 : 0) |
                                    (mode == 1 ? 31u << 16 : mode == 2 ? 18u << 16 : 0);
                                if (!valid) {
                                    EXPECT_STATUS(word, reserved);
                                    continue;
                                }
                                const auto e = decode_register_effects(word);
                                CHECK(e.decoded());
                                CHECK_EQ(e.memory.elements, 1 + 2 * (opcode % 2) + r);
                                CHECK(reads_stack_pointer(e));
                                CHECK(writes_stack_pointer(e) == (mode != 0));
                                CHECK_EQ(e.reads, mode == 2 ? register_bit(18) : 0);
                                CHECK_EQ(e.writes, 0);
                                CHECK(e.operands[0].access == (load == 0 ? R : opcode < 6 ? RW : W));
                                const unsigned bytes = opcode < 2 ? 1 : opcode < 4 ? 2 :
                                    opcode < 6 ? (size == 0 ? 4 : 8) : 1u << size;
                                CHECK_EQ(e.memory.element_bytes, bytes);
                                CHECK_EQ(e.memory.offset, mode == 1 ? bytes * e.memory.elements : 0);
                                CHECK_EQ(e.memory.writeback, mode != 0);
                                CHECK(substitute_register(word, 18, 9, UnchangedWord::accept).ok());
                            }
                        }
                    }
                }
            }
        }
    }
    // RCpc3 uses the otherwise absent Rm/o2 bit in a no-writeback form.
    EXPECT_STATUS(0x0D419400u, unknown);  // LDAP1 D[0], [X0]
    EXPECT_STATUS(0x0D019400u, unknown);  // STL1 D[0], [X0]
}

void simd_copy_encoding_boundaries() {
    constexpr unsigned opcodes[] = {1, 3, 5, 7};
    constexpr Operation operations[] = {Operation::dup_general, Operation::ins_general,
                                        Operation::smov, Operation::umov};
    // Bit n says imm5=n is allocated. These list the encoding-table patterns,
    // including DUP's ignored upper immediate bits.
    constexpr std::uint32_t valid[4][2] = {
        {0xFEFEFEFEu, 0xFFFEFFFEu}, {0, 0xFFFEFFFEu},
        {0xEEEEEEEEu, 0xFEFEFEFEu}, {0xFEFEFEFEu, 0x01000100u}};
    for (unsigned form = 0; form < 4; ++form) {
        for (unsigned q = 0; q < 2; ++q) {
            for (unsigned imm5 = 0; imm5 < 32; ++imm5) {
                const auto word = 0x0E000400u | q << 30 | imm5 << 16 | opcodes[form] << 11 |
                                  18u << 5 | 18u;
                if (((valid[form][q] >> imm5) & 1u) == 0) {
                    EXPECT_STATUS(word, reserved);
                    continue;
                }
                const auto e = decode_register_effects(word);
                CHECK(e.decoded() && e.operation == operations[form]);
                CHECK_EQ(e.reads, form < 2 ? register_bit(18) : 0);
                CHECK_EQ(e.writes, form < 2 ? 0 : register_bit(18));
                const auto& gpr = e.operands[form < 2 ? 0 : 1];
                CHECK_EQ(gpr.bits, form < 2 ? (imm5 == 8 || imm5 == 24 ? 64 : 32) : q ? 64 : 32);
                CHECK(gpr.at_31 == ZR);
                CHECK(register_fields(e, 18) == std::optional<std::uint32_t>{form < 2 ? 0x3E0u : 0x1Fu});
                CHECK(substitute_register(word, 18, 9, UnchangedWord::reject).ok());
                const auto zero = decode_register_effects(with_field(word, gpr.field, 31));
                CHECK(zero.decoded() && zero.reads == 0 && zero.writes == 0);
                CHECK(!reads_stack_pointer(zero) && !writes_stack_pointer(zero));
            }
        }
    }
    // INS (element) is decoded on its own; mismatched lane sizes are a
    // reserved combination of the copy class.
    EXPECT_STATUS(0x4E181E52u | 1u << 29, reserved);
}

void fp_general_encoding_boundaries() {
    constexpr unsigned conversions[] = {0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 25};
    for (unsigned sf = 0; sf < 2; ++sf) {
        for (unsigned type = 0; type < 4; ++type) {
            for (unsigned op : conversions) {
                const auto word = 0x1E200000u | sf << 31 | type << 22 | op << 16 | 18u << 5 | 18u;
                if (type == 2) {
                    EXPECT_STATUS(word, reserved);
                    continue;
                }
                const auto e = decode_register_effects(word);
                CHECK(e.decoded());
                CHECK_EQ(e.reads, op == 2 || op == 3 ? register_bit(18) : 0);
                CHECK_EQ(e.writes, op == 2 || op == 3 ? 0 : register_bit(18));
                CHECK_EQ(e.operands[op == 2 || op == 3 ? 0 : 1].bits, sf ? 64 : 32);
                CHECK(substitute_register(word, 18, 9, UnchangedWord::reject).ok());
                // Scalar conversions exist only in the sf = 0, U = 0, p = 0
                // rows. Setting bit10 turns the word into the FCCMP frame.
                EXPECT_STATUS(word | 1u << 29, reserved);
                EXPECT_STATUS(word | 1u << 30, reserved);
                if (sf == 0) {
                    const auto fccmp = decode_register_effects(word | 1u << 10);
                    CHECK(fccmp.decoded() &&
                          fccmp.operation == Operation::fp_conditional_compare);
                } else {
                    EXPECT_STATUS(word | 1u << 10, reserved);
                }
                if (op != 2 && op != 3 && op != 24 && op != 25) continue;
                for (unsigned scale = 0; scale < 64; ++scale) {
                    const auto fixed = (word & ~(1u << 21)) | scale << 10;
                    if (sf == 0 && scale < 32) {
                        EXPECT_STATUS(fixed, reserved);
                    } else {
                        CHECK(decode_register_effects(fixed).decoded());
                        CHECK(substitute_register(fixed, 18, 9, UnchangedWord::reject).ok());
                    }
                }
            }
            for (unsigned op : {6u, 7u, 14u, 15u}) {
                const auto word = 0x1E200000u | sf << 31 | type << 22 | op << 16 | 18u << 5 | 18u;
                const bool valid = op >= 14 ? sf == 1 && type == 2 :
                    type == 3 || (sf == 0 && type == 0) || (sf == 1 && type == 1);
                if (!valid) {
                    EXPECT_STATUS(word, reserved);
                } else {
                    const auto e = decode_register_effects(word);
                    CHECK(e.decoded() && e.operation == Operation::fmov_general);
                    CHECK_EQ(e.reads, op % 2 ? register_bit(18) : 0);
                    CHECK_EQ(e.writes, op % 2 ? 0 : register_bit(18));
                    if (op == 15) CHECK(e.operands[1].access == RW && e.operands[1].bits == 128);
                    CHECK(substitute_register(word, 18, 9, UnchangedWord::reject).ok());
                }
            }
        }
    }
    EXPECT_STATUS(0x1E7E0252u, unknown);  // FJCVTZS belongs to an unsupported extension.
    EXPECT_STATUS(0x1E2E0252u, reserved); // Top-half move requires X and ftype=10.
}

void writeback_hazards() {
    // Load and store writeback with Rt == Rn (n != 31) is CONSTRAINED
    // UNPREDICTABLE; the assembler refuses ldr x1, [x1, #8]! and friends.
    for (unsigned kind : {1u, 3u}) {
        for (unsigned opc : {0u, 1u}) {
            const auto e = decode_register_effects(ls_imm9(3, 0, opc, 8, kind, 18, 18));
            CHECK(e.status == EffectStatus::unpredictable);
            CHECK(has(e.notes, EffectNote::writeback_overlaps_transfer));
            CHECK(!proven_free_of(e, 18) && !register_fields(e, 18).has_value());
            CHECK(!proven_free_of(e, 1));
        }
    }
    // Rt = 31 with Rn = 31 names XZR and SP: no overlap.
    auto e = decode_register_effects(ls_imm9(3, 0, 0, -16, 3, 31, 31));
    CHECK(e.decoded() && reads_stack_pointer(e) && writes_stack_pointer(e));
    CHECK(has(e.notes, EffectNote::stack_pointer_base));
    CHECK_EQ(e.reads, 0);
    // Without writeback the same registers are fine.
    CHECK(decode_register_effects(ls_unsigned(3, 0, 1, 1, 18, 18)).decoded());
    CHECK(decode_register_effects(ls_imm9(3, 0, 1, 8, 0, 18, 18)).decoded());
    CHECK(decode_register_effects(ls_imm9(3, 0, 1, 8, 2, 18, 18)).decoded());
    // Prefetch "Rt" never overlaps the base.
    CHECK(decode_register_effects(ls_unsigned(3, 0, 2, 0, 18, 18)).decoded());
}

void register_pairs() {
    // ldp x5, x11, [x7, #16]: Rt and Rt2 are XZR when 31, Rn is SP.
    auto e = EXPECT_EFFECTS(ls_pair(2, 0, 2, 1, 2, 11, 7, 5), Operation::ldp,
                            Op{0, 5, ZR, 64, W, Role::transfer},
                            Op{10, 11, ZR, 64, W, Role::transfer},
                            Op{5, 7, SP, 64, R, Role::base});
    CHECK(e.memory.addressing == AddressingMode::signed_offset);
    CHECK_EQ(e.memory.offset, 16);
    CHECK_EQ(e.memory.elements, 2);
    // stp w5, w11, [x7, #-256]!
    e = EXPECT_EFFECTS(ls_pair(0, 0, 3, 0, -64, 11, 7, 5), Operation::stp,
                       Op{0, 5, ZR, 32, R, Role::transfer}, Op{10, 11, ZR, 32, R, Role::transfer},
                       Op{5, 7, SP, 64, RW, Role::base});
    CHECK(e.memory.writeback && e.memory.addressing == AddressingMode::pre_index);
    CHECK_EQ(e.memory.offset, -256);
    e = EXPECT_EFFECTS(ls_pair(2, 0, 1, 1, 63, 11, 7, 5), Operation::ldp,
                       Op{0, 5, ZR, 64, W, Role::transfer}, Op{10, 11, ZR, 64, W, Role::transfer},
                       Op{5, 7, SP, 64, RW, Role::base});
    CHECK(e.memory.addressing == AddressingMode::post_index);
    CHECK_EQ(e.memory.offset, 504);
    // ldpsw: 4-byte elements, sign-extended into X registers.
    e = EXPECT_EFFECTS(ls_pair(1, 0, 2, 1, -1, 11, 7, 5), Operation::ldpsw,
                       Op{0, 5, ZR, 64, W, Role::transfer}, Op{10, 11, ZR, 64, W, Role::transfer},
                       Op{5, 7, SP, 64, R, Role::base});
    CHECK_EQ(e.memory.offset, -4);
    CHECK(has(e.memory.attributes, MemoryAttribute::sign_extend));
    // Non-temporal pairs.
    e = EXPECT_EFFECTS(ls_pair(2, 0, 0, 0, 1, 11, 7, 5), Operation::stnp,
                       Op{0, 5, ZR, 64, R, Role::transfer}, Op{10, 11, ZR, 64, R, Role::transfer},
                       Op{5, 7, SP, 64, R, Role::base});
    CHECK(has(e.memory.attributes, MemoryAttribute::non_temporal) && !e.memory.writeback);
    EXPECT_EFFECTS(ls_pair(0, 0, 0, 1, 1, 11, 7, 5), Operation::ldnp,
                   Op{0, 5, ZR, 32, W, Role::transfer}, Op{10, 11, ZR, 32, W, Role::transfer},
                   Op{5, 7, SP, 64, R, Role::base});
    // Vector pairs: S, D and Q.
    for (unsigned opc = 0; opc < 3; ++opc) {
        const auto bits = static_cast<std::uint8_t>(32u << opc);
        // ldp q18, q11, [x7, #-16]!: V18 is not x18.
        e = EXPECT_EFFECTS(ls_pair(opc, 1, 3, 1, -1, 11, 7, 18), Operation::ldp_vector,
                           Op{0, 18, NA, bits, W, Role::transfer, false, VEC},
                           Op{10, 11, NA, bits, W, Role::transfer, false, VEC},
                           Op{5, 7, SP, 64, RW, Role::base});
        CHECK(proven_free_of(e, 18));
        CHECK_EQ(e.memory.offset, -(4 << opc));
        e = EXPECT_EFFECTS(ls_pair(opc, 1, 0, 0, 1, 11, 7, 5), Operation::stnp_vector,
                           Op{0, 5, NA, bits, R, Role::transfer, false, VEC},
                           Op{10, 11, NA, bits, R, Role::transfer, false, VEC},
                           Op{5, 7, SP, 64, R, Role::base});
        CHECK_EQ(e.memory.offset, 4 << opc);
        EXPECT_EFFECTS(ls_pair(opc, 1, 2, 0, 1, 11, 7, 5), Operation::stp_vector,
                       Op{0, 5, NA, bits, R, Role::transfer, false, VEC},
                       Op{10, 11, NA, bits, R, Role::transfer, false, VEC},
                       Op{5, 7, SP, 64, R, Role::base});
        EXPECT_EFFECTS(ls_pair(opc, 1, 0, 1, 1, 11, 7, 5), Operation::ldnp_vector,
                       Op{0, 5, NA, bits, W, Role::transfer, false, VEC},
                       Op{10, 11, NA, bits, W, Role::transfer, false, VEC},
                       Op{5, 7, SP, 64, R, Role::base});
    }

    // Hazards: Rt == Rt2 on loads, and writeback overlapping a transfer.
    e = decode_register_effects(ls_pair(2, 0, 2, 1, 0, 18, 1, 18));
    CHECK(e.status == EffectStatus::unpredictable);
    CHECK(has(e.notes, EffectNote::pair_load_same_register));
    CHECK(!proven_free_of(e, 3));
    e = decode_register_effects(ls_pair(0, 1, 2, 1, 0, 5, 1, 5));  // ldp s5, s5, [x1]
    CHECK(e.status == EffectStatus::unpredictable);
    CHECK(decode_register_effects(ls_pair(2, 0, 2, 0, 0, 18, 1, 18)).decoded());  // stp x18, x18
    for (unsigned mode : {1u, 3u}) {
        for (unsigned l : {0u, 1u}) {
            e = decode_register_effects(ls_pair(2, 0, mode, l, 2, 2, 18, 18));  // Rt == Rn
            CHECK(e.status == EffectStatus::unpredictable);
            CHECK(has(e.notes, EffectNote::writeback_overlaps_transfer));
            e = decode_register_effects(ls_pair(2, 0, mode, l, 2, 18, 18, 2));  // Rt2 == Rn
            CHECK(e.status == EffectStatus::unpredictable);
            CHECK(has(e.notes, EffectNote::writeback_overlaps_transfer));
        }
    }
    // stp x29, x30, [sp, #-16]! with Rt = Rt2 = 31 and Rn = SP is not a hazard.
    e = decode_register_effects(ls_pair(2, 0, 3, 0, -2, 31, 31, 31));
    CHECK(e.decoded() && writes_stack_pointer(e));
    // Vector writeback never overlaps the general base.
    CHECK(decode_register_effects(ls_pair(1, 1, 3, 0, 2, 2, 18, 18)).decoded());

    // Neighbors that are extensions or unallocated.
    EXPECT_STATUS(ls_pair(1, 0, 2, 0, 1, 11, 7, 5), unknown);  // STGP
    EXPECT_STATUS(ls_pair(1, 0, 0, 1, 1, 11, 7, 5), unknown);  // LDNP with opc = 01
    EXPECT_STATUS(ls_pair(3, 0, 2, 1, 1, 11, 7, 5), unknown);  // LDTP space
    EXPECT_STATUS(ls_pair(3, 1, 2, 1, 1, 11, 7, 5), unknown);
}

void exclusive_and_ordered() {
    const auto exclusive_attribute = static_cast<MemoryAttributes>(MemoryAttribute::exclusive);
    const auto acquire = static_cast<MemoryAttributes>(MemoryAttribute::acquire);
    const auto release = static_cast<MemoryAttributes>(MemoryAttribute::release);

    // Load-exclusive: Rt written, Rn is SP when 31; Rs and Rt2 are unused.
    static constexpr Operation loads[2][4] = {
        {Operation::ldxrb, Operation::ldxrh, Operation::ldxr, Operation::ldxr},
        {Operation::ldaxrb, Operation::ldaxrh, Operation::ldaxr, Operation::ldaxr},
    };
    static constexpr Operation stores[2][4] = {
        {Operation::stxrb, Operation::stxrh, Operation::stxr, Operation::stxr},
        {Operation::stlxrb, Operation::stlxrh, Operation::stlxr, Operation::stlxr},
    };
    for (unsigned o0 = 0; o0 < 2; ++o0) {
        for (unsigned size = 0; size < 4; ++size) {
            const auto bits = static_cast<std::uint8_t>(size == 3 ? 64 : 32);
            auto e = EXPECT_EFFECTS(exclusive(size, 0, 1, 0, 31, o0, 31, 7, 5), loads[o0][size],
                                    Op{0, 5, ZR, bits, W, Role::transfer},
                                    Op{5, 7, SP, 64, R, Role::base});
            CHECK(e.memory.operation == MemoryOperation::load);
            CHECK(e.memory.addressing == AddressingMode::base_register);
            CHECK_EQ(e.memory.element_bytes, 1u << size);
            CHECK_EQ(e.memory.attributes, exclusive_attribute | (o0 ? acquire : 0));
            // Store-exclusive: Rs is a written 32-bit status, WZR when 31.
            e = EXPECT_EFFECTS(exclusive(size, 0, 0, 0, 9, o0, 31, 7, 5), stores[o0][size],
                               Op{0, 5, ZR, bits, R, Role::transfer},
                               Op{5, 7, SP, 64, R, Role::base},
                               Op{16, 9, ZR, 32, W, Role::status});
            CHECK(e.memory.operation == MemoryOperation::store);
            CHECK_EQ(e.memory.attributes, exclusive_attribute | (o0 ? release : 0));
        }
    }
    // Pairs: sz is bit 30; Rt2 is a second transfer register.
    for (unsigned sz = 0; sz < 2; ++sz) {
        const auto bits = static_cast<std::uint8_t>(sz ? 64 : 32);
        auto e = EXPECT_EFFECTS(exclusive(2 | sz, 0, 1, 1, 31, 0, 11, 7, 5), Operation::ldxp,
                                Op{0, 5, ZR, bits, W, Role::transfer},
                                Op{10, 11, ZR, bits, W, Role::transfer},
                                Op{5, 7, SP, 64, R, Role::base});
        CHECK_EQ(e.memory.elements, 2);
        CHECK_EQ(e.memory.element_bytes, bits / 8);
        e = EXPECT_EFFECTS(exclusive(2 | sz, 0, 1, 1, 31, 1, 11, 7, 5), Operation::ldaxp,
                           Op{0, 5, ZR, bits, W, Role::transfer},
                           Op{10, 11, ZR, bits, W, Role::transfer},
                           Op{5, 7, SP, 64, R, Role::base});
        CHECK(has(e.memory.attributes, MemoryAttribute::acquire));
        EXPECT_EFFECTS(exclusive(2 | sz, 0, 0, 1, 9, 0, 11, 7, 5), Operation::stxp,
                       Op{0, 5, ZR, bits, R, Role::transfer},
                       Op{10, 11, ZR, bits, R, Role::transfer}, Op{5, 7, SP, 64, R, Role::base},
                       Op{16, 9, ZR, 32, W, Role::status});
        e = EXPECT_EFFECTS(exclusive(2 | sz, 0, 0, 1, 9, 1, 11, 7, 5), Operation::stlxp,
                           Op{0, 5, ZR, bits, R, Role::transfer},
                           Op{10, 11, ZR, bits, R, Role::transfer},
                           Op{5, 7, SP, 64, R, Role::base}, Op{16, 9, ZR, 32, W, Role::status});
        CHECK(has(e.memory.attributes, MemoryAttribute::release));
    }
    // Load-acquire and store-release (not exclusive).
    static constexpr Operation acquires[] = {Operation::ldarb, Operation::ldarh, Operation::ldar,
                                             Operation::ldar};
    static constexpr Operation releases[] = {Operation::stlrb, Operation::stlrh, Operation::stlr,
                                             Operation::stlr};
    for (unsigned size = 0; size < 4; ++size) {
        const auto bits = static_cast<std::uint8_t>(size == 3 ? 64 : 32);
        auto e = EXPECT_EFFECTS(exclusive(size, 1, 1, 0, 31, 1, 31, 7, 5), acquires[size],
                                Op{0, 5, ZR, bits, W, Role::transfer},
                                Op{5, 7, SP, 64, R, Role::base});
        CHECK_EQ(e.memory.attributes, acquire);
        e = EXPECT_EFFECTS(exclusive(size, 1, 0, 0, 31, 1, 31, 7, 5), releases[size],
                           Op{0, 5, ZR, bits, R, Role::transfer},
                           Op{5, 7, SP, 64, R, Role::base});
        CHECK_EQ(e.memory.attributes, release);
    }

    // Status overlaps (the assembler rejects these as unpredictable).
    auto e = decode_register_effects(exclusive(3, 0, 0, 0, 5, 0, 31, 7, 5));  // stxr w5, x5
    CHECK(e.status == EffectStatus::unpredictable);
    CHECK(has(e.notes, EffectNote::status_overlaps_operand));
    e = decode_register_effects(exclusive(3, 0, 0, 0, 7, 0, 31, 7, 5));  // stxr w7, x5, [x7]
    CHECK(e.status == EffectStatus::unpredictable);
    e = decode_register_effects(exclusive(3, 0, 0, 1, 11, 0, 11, 7, 5));  // stxp w11, x5, x11
    CHECK(e.status == EffectStatus::unpredictable);
    e = decode_register_effects(exclusive(3, 0, 0, 0, 31, 0, 31, 7, 31));  // stxr wzr, xzr
    CHECK(e.status == EffectStatus::unpredictable);
    // Rs == Rn == 31 names WZR and SP: not an overlap.
    e = decode_register_effects(exclusive(3, 0, 0, 0, 31, 0, 31, 31, 5));  // stxr wzr, x5, [sp]
    CHECK(e.decoded() && has(e.notes, EffectNote::stack_pointer_base));
    // Pair load with Rt == Rt2.
    e = decode_register_effects(exclusive(3, 0, 1, 1, 31, 0, 5, 7, 5));
    CHECK(e.status == EffectStatus::unpredictable);
    CHECK(has(e.notes, EffectNote::pair_load_same_register));
    // Should-be-one fields: an x18-looking value there is not an operand,
    // and the word is never treated as x18-free.
    for (std::uint32_t word : {
             exclusive(3, 0, 1, 0, 18, 0, 31, 7, 5),   // ldxr with Rs = 18
             exclusive(3, 0, 1, 0, 31, 0, 18, 7, 5),   // ldxr with Rt2 = 18
             exclusive(3, 0, 0, 0, 9, 0, 18, 7, 5),    // stxr with Rt2 = 18
             exclusive(3, 0, 1, 1, 18, 0, 11, 7, 5),   // ldxp with Rs = 18
             exclusive(3, 1, 1, 0, 18, 1, 31, 7, 5),   // ldar with Rs = 18
             exclusive(3, 1, 0, 0, 31, 1, 18, 7, 5),   // stlr with Rt2 = 18
         }) {
        e = decode_register_effects(word);
        CHECK(e.status == EffectStatus::unpredictable);
        CHECK(has(e.notes, EffectNote::should_be_one_field));
        CHECK(!proven_free_of(e, 18) && !proven_free_of(e, 3));
    }

    EXPECT_STATUS(exclusive(3, 1, 0, 1, 1, 0, 31, 3, 2), unknown);  // CAS
    EXPECT_STATUS(exclusive(1, 0, 0, 1, 0, 0, 31, 4, 2), unknown);  // CASP
    EXPECT_STATUS(exclusive(3, 1, 1, 0, 31, 0, 31, 2, 1), unknown); // LDLAR
    EXPECT_STATUS(exclusive(3, 1, 0, 0, 31, 0, 31, 2, 1), unknown); // STLLR
}

void classified_integration() {
    // Effects of classified words come from instruction.h without change.
    test::Lcg random{0x5EEDu};
    std::uint64_t compared = 0;
    for (int index = 0; index < 400'000; ++index) {
        // Bias toward the classified encoding groups.
        std::uint32_t word = static_cast<std::uint32_t>(random.next());
        switch (index % 8) {
        case 0:
            word = (word & 0x83FFFFFFu) | 0x14000000u;  // B/BL
            break;
        case 1:
            word = (word & 0xFF00FFFFu) | 0x34000000u;  // CBZ/CBNZ
            break;
        case 2:
            word = (word & 0x001FFFFFu) | 0xD5100000u;  // MRS/MSR
            break;
        case 3:
            word = (word & 0x8100FFFFu) | 0x36000000u;  // TBZ/TBNZ
            break;
        case 4:
            word = (word & 0x60FFFFFFu) | 0x10000000u | (word & 0x80000000u);  // ADR/ADRP
            break;
        case 5:
            word = (word & 0xC4FFFFFFu) | 0x18000000u;  // load literal
            break;
        case 6:  // BR/BLR/RET
            word = 0xD61F0000u | ((word % 3) << 21) | (((word >> 8) & 0x1Fu) << 5);
            break;
        default:  // exception generation
            word = (word & 0x00FFFFE3u) | 0xD4000000u;
            break;
        }
        const Instruction instruction = decode(word);
        if (!instruction.classified()) {
            continue;
        }
        const auto e = decode_register_effects(word);
        ++compared;
        CHECK(e.status == EffectStatus::decoded);
        CHECK(e.operation == Operation::classified);
        CHECK(e.opcode == instruction.opcode);
        CHECK_EQ(e.reads, instruction.reads);
        CHECK_EQ(e.writes, instruction.writes);
    }
    CHECK(compared > 100'000);

    // cbz x18: field 0 read, XZR at 31.
    auto e = decode_register_effects(0xB4000012u);
    CHECK(e.control_flow == ControlFlow::conditional_branch);
    CHECK_EQ(register_fields(e, 18).value_or(0), 0x1Fu);
    CHECK(decode_register_effects(0xB400001Fu).operands[0].names_zero_register());
    // tbz w18, #3: a 32-bit view.
    e = decode_register_effects(0x36180012u);
    CHECK_EQ(e.operands[0].bits, 32);
    // br x18 / blr x18 / ret x18 read field 5.
    e = decode_register_effects(0xD61F0240u);
    CHECK(e.control_flow == ControlFlow::indirect_branch);
    CHECK_EQ(register_fields(e, 18).value_or(0), 0x3E0u);
    e = decode_register_effects(0xD63F0240u);
    CHECK(e.control_flow == ControlFlow::indirect_call);
    CHECK_EQ(e.operand_count, 2);
    CHECK(e.operands[1].field == implicit_operand && e.operands[1].role == Role::link);
    CHECK(!register_fields(e, 30).has_value());  // x30 is implicit, not a field
    CHECK(!proven_free_of(e, 30));
    CHECK(proven_free_of(decode_register_effects(0xD63F0200u), 18));  // blr x16
    // ret reads x30 through field 5; ret x18 reads x18.
    e = decode_register_effects(0xD65F03C0u);
    CHECK(e.control_flow == ControlFlow::return_);
    CHECK_EQ(e.reads, RegisterMask{1} << 30);
    CHECK_EQ(register_fields(e, 30).value_or(0), 0x3E0u);
    CHECK_EQ(register_fields(decode_register_effects(0xD65F0240u), 18).value_or(0), 0x3E0u);
    // bl: implicit link only.
    e = decode_register_effects(0x94000001u);
    CHECK(e.control_flow == ControlFlow::call && proven_free_of(e, 18) && !proven_free_of(e, 30));
    // b.ne reads NZCV only.
    e = decode_register_effects(0x54000001u);
    CHECK(e.reads_flags && e.operand_count == 0 && proven_free_of(e, 18));
    // adr x18 writes; adrp with Rd = 31 writes XZR.
    e = decode_register_effects(0x10000012u);
    CHECK_EQ(register_fields(e, 18).value_or(0), 0x1Fu);
    CHECK(decode_register_effects(0x9000001Fu).operands[0].names_zero_register());
    // Literal loads: GPR and vector transfers.
    e = decode_register_effects(0x98000012u);  // ldrsw x18, literal
    CHECK(e.memory.addressing == AddressingMode::literal);
    CHECK(has(e.memory.attributes, MemoryAttribute::sign_extend));
    CHECK_EQ(e.writes, RegisterMask{1} << 18);
    e = decode_register_effects(0x9C000012u);  // ldr q18, literal
    CHECK(e.operands[0].file == VEC && proven_free_of(e, 18));
    e = decode_register_effects(0xD8000012u);  // prfm #18, literal
    CHECK(e.operand_count == 0 && proven_free_of(e, 18));
    // Exception generation never claims register freedom.
    for (std::uint32_t word : {0xD4000001u, 0xD4000002u, 0xD4000003u, 0xD4200000u, 0xD4400000u}) {
        e = decode_register_effects(word);
        CHECK(e.decoded() && e.control_flow == ControlFlow::exception);
        CHECK(has(e.notes, EffectNote::handler_defined_registers));
        CHECK(!proven_free_of(e, 18) && !register_fields(e, 18).has_value());
    }
    // mrs x18, tpidr_el0 / msr nzcv, x1 / mrs x2, nzcv.
    e = decode_register_effects(0xD53BD052u);
    CHECK_EQ(register_fields(e, 18).value_or(0), 0x1Fu);
    CHECK_EQ(e.writes, RegisterMask{1} << 18);
    e = decode_register_effects(0xD51B4201u);
    CHECK(e.writes_flags && !e.reads_flags);
    e = decode_register_effects(0xD53B4202u);
    CHECK(e.reads_flags && !e.writes_flags);
}

// Properties over random words: unknown and reserved words make no register
// claim; decoded operands match their fields; and every explicit general
// field is a pure register field (any other value 0..30 decodes the same
// operation with the same operand layout).
void random_word_properties() {
    test::Lcg random{0xA64u};
    std::uint64_t decoded = 0;
    std::uint64_t unknown = 0;
    std::uint64_t reserved = 0;
    std::uint64_t unpredictable = 0;
    for (int index = 0; index < 300'000; ++index) {
        std::uint32_t word = static_cast<std::uint32_t>(random.next());
        // Bias half the words toward the decoded encoding groups.
        switch (index % 7) {
        case 0:
            word = (word & ~0x1C000000u) | 0x10000000u;  // data processing (immediate)
            break;
        case 4:
            word = (word & ~0x3F000000u) | 0x08000000u;  // load/store exclusive and ordered
            break;
        case 1:
            word = (word & ~0x0E000000u) | 0x0A000000u;  // data processing (register)
            break;
        case 2:
            word = (word & ~0x3A000000u) | 0x38000000u;  // load/store register
            break;
        case 3:
            word = (word & ~0x3A000000u) | 0x28000000u;  // load/store pair
            break;
        default:
            break;
        }
        const auto e = decode_register_effects(word);
        CHECK_EQ(e.word, word);
        switch (e.status) {
        case EffectStatus::unknown:
        case EffectStatus::reserved:
            (e.status == EffectStatus::unknown ? unknown : reserved) += 1;
            if (!CHECK(e.operation == Operation::none && e.operand_count == 0 && e.reads == 0 &&
                       e.writes == 0 && !proven_free_of(e, 18))) {
                std::fprintf(stderr, "  word 0x%08x\n", word);
            }
            continue;
        case EffectStatus::unpredictable:
            ++unpredictable;
            CHECK(e.notes != 0 && !proven_free_of(e, 18));
            break;
        case EffectStatus::decoded:
            ++decoded;
            break;
        }
        RegisterMask reads = 0;
        RegisterMask writes = 0;
        for (std::size_t op = 0; op < e.operand_count; ++op) {
            const auto& operand = e.operands[op];
            if (operand.field != implicit_operand) {
                CHECK_EQ(operand.number, (word >> operand.field) & 0x1Fu);
            }
            CHECK((operand.file == GPR) == (operand.at_31 != NA));
            if (operand.file == GPR && operand.number < 31) {
                reads |= includes_read(operand.access) ? register_bit(operand.number) : 0;
                writes |= includes_write(operand.access) ? register_bit(operand.number) : 0;
            }
        }
        CHECK_EQ(e.reads, reads);
        CHECK_EQ(e.writes, writes);
        if (e.status != EffectStatus::decoded || e.operation == Operation::classified ||
            index % 8 != 0) {
            continue;
        }
        // Field independence for a sample of decoded words.
        for (std::size_t op = 0; op < e.operand_count; ++op) {
            const auto& operand = e.operands[op];
            if (operand.field == implicit_operand) {
                continue;
            }
            for (unsigned value : {0u, 18u, 30u, 31u}) {
                const auto changed =
                    decode_register_effects(with_field(word, operand.field, value));
                // Structure post-index Rm=31 selects an immediate increment,
                // rather than a zero-register operand. Substitution never emits 31.
                if (value == 31 && operand.role == Role::index &&
                    e.memory.addressing == AddressingMode::post_index) {
                    CHECK(changed.decoded() && changed.operation == e.operation);
                    CHECK_EQ(changed.operand_count + 1, e.operand_count);
                    CHECK_EQ(changed.memory.offset,
                             changed.memory.element_bytes * changed.memory.elements);
                    continue;
                }
                const bool same_shape = changed.operation == e.operation &&
                                        changed.operand_count == e.operand_count &&
                                        changed.operands[op].field == operand.field &&
                                        changed.operands[op].role == operand.role &&
                                        changed.operands[op].at_31 == operand.at_31 &&
                                        changed.operands[op].access == operand.access;
                // Changing a register may create a writeback or pair overlap.
                const bool status_ok = changed.status == EffectStatus::decoded ||
                                       changed.status == EffectStatus::unpredictable;
                if (!CHECK(same_shape && status_ok)) {
                    std::fprintf(stderr, "  word 0x%08x field %u value %u\n", word, operand.field,
                                 value);
                }
            }
        }
    }
    std::printf("random words: %llu decoded, %llu unpredictable, %llu reserved, %llu unknown\n",
                static_cast<unsigned long long>(decoded),
                static_cast<unsigned long long>(unpredictable),
                static_cast<unsigned long long>(reserved),
                static_cast<unsigned long long>(unknown));
    CHECK(decoded > 50'000 && reserved > 1'000 && unknown > 50'000 && unpredictable > 100);
}

void register_queries() {
    // Register numbers outside 0..30 never yield a claim.
    const auto e = decode_register_effects(add_sub_imm(1, 0, 0, 0, 4, 7, 5));
    CHECK(!proven_free_of(e, 31) && !register_fields(e, 31).has_value());
    CHECK(!proven_free_of(e, 64));
    CHECK(proven_free_of(e, 18));
    CHECK_EQ(register_fields(e, 18).value_or(1), 0);
    // x18 in two fields at once: both fields are reported.
    const auto twice = decode_register_effects(add_sub_shifted(1, 0, 0, 0, 1, 0, 18, 18));
    CHECK_EQ(register_fields(twice, 18).value_or(0), 0x3FFu);
}

// ---------------------------------------------------------------------------
// Advanced SIMD and scalar floating-point data processing (op0 = x111),
// composed from the encoding tables.

std::uint32_t simd_same(unsigned q, unsigned u, unsigned size, unsigned opcode, unsigned rm,
                        unsigned rn, unsigned rd) {
    return q << 30 | u << 29 | 0b1110u << 24 | size << 22 | 1u << 21 | rm << 16 | opcode << 11 |
           1u << 10 | rn << 5 | rd;
}
std::uint32_t simd_same_fp(unsigned q, unsigned u, unsigned bit23, unsigned bit22,
                           unsigned opcode, unsigned rm, unsigned rn, unsigned rd) {
    return q << 30 | u << 29 | 0b1110u << 24 | bit23 << 23 | bit22 << 22 | 1u << 21 | rm << 16 |
           opcode << 11 | 1u << 10 | rn << 5 | rd;
}
std::uint32_t simd_different(unsigned q, unsigned u, unsigned size, unsigned opcode, unsigned rm,
                             unsigned rn, unsigned rd) {
    return q << 30 | u << 29 | 0b1110u << 24 | size << 22 | 1u << 21 | rm << 16 | opcode << 11 |
           rn << 5 | rd;
}
std::uint32_t simd_tagged(unsigned q, unsigned u, unsigned bit23, unsigned bit22, unsigned opcode6,
                          unsigned tag, unsigned rn, unsigned rd) {
    return q << 30 | u << 29 | 0b1110u << 24 | bit23 << 23 | bit22 << 22 | 1u << 21 | tag << 16 |
           opcode6 << 10 | rn << 5 | rd;
}
// Bits 28:21 = 01110 0 00 for copy, table, EXT and permute rows.
std::uint32_t simd_copy_extract(unsigned q, unsigned u, unsigned opcode6, unsigned rm,
                                unsigned rn, unsigned rd) {
    return q << 30 | u << 29 | 0x0E000000u | rm << 16 | opcode6 << 10 | rn << 5 | rd;
}
std::uint32_t simd_shift_imm(unsigned q, unsigned u, unsigned immh, unsigned immb,
                             unsigned opcode6, unsigned rn, unsigned rd) {
    return q << 30 | u << 29 | 0x0F000000u | immh << 19 | immb << 16 | opcode6 << 10 | rn << 5 |
           rd;
}
std::uint32_t simd_modified_imm(unsigned q, unsigned u, unsigned imm, unsigned opcode6,
                                unsigned rd) {
    return q << 30 | u << 29 | 0x0F000000u | imm << 12 | opcode6 << 10 | rd;
}
std::uint32_t simd_scalar_shift(unsigned u, unsigned immh, unsigned immb, unsigned opcode6,
                                unsigned rn, unsigned rd) {
    return 1u << 30 | u << 29 | 0x1F000000u | immh << 19 | immb << 16 | opcode6 << 10 | rn << 5 |
           rd;
}
std::uint32_t fp_3source(unsigned type, unsigned rm, unsigned o1, unsigned ra, unsigned o0,
                         unsigned rn, unsigned rd) {
    return 0x1F000000u | type << 22 | o1 << 21 | rm << 16 | o0 << 15 | ra << 10 | rn << 5 | rd;
}
std::uint32_t scalar_p1_misc(unsigned u, unsigned bit23, unsigned bit22, unsigned opcode6,
                             unsigned tag, unsigned rn, unsigned rd) {
    return 1u << 30 | u << 29 | 0x1E000000u | bit23 << 23 | bit22 << 22 | 1u << 21 | tag << 16 |
           opcode6 << 10 | rn << 5 | rd;
}

Op vector_destination(unsigned number, unsigned bits, Access access) {
    return Op{0, static_cast<std::uint8_t>(number), NA, static_cast<std::uint8_t>(bits), access,
              Role::destination, false, VEC};
}
Op vector_source(unsigned field, unsigned number, unsigned bits) {
    return Op{static_cast<std::uint8_t>(field), static_cast<std::uint8_t>(number), NA,
              static_cast<std::uint8_t>(bits), R, Role::source, false, VEC};
}

void simd_vector_groups() {
    // Integer three same: ADD V0.4S, V1.4S, V2.4S.
    auto e = EXPECT_EFFECTS(simd_same(1, 0, 2, 16, 2, 1, 0), Operation::simd_data,
                            vector_destination(0, 128, W), vector_source(5, 1, 128),
                            vector_source(16, 2, 128));
    CHECK(!e.reads_flags && !e.writes_flags);
    // SQADD keeps a doubleword row; the byte form uses the low halves.
    EXPECT_EFFECTS(simd_same(1, 0, 3, 1, 2, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));
    EXPECT_EFFECTS(simd_same(0, 1, 0, 0, 2, 1, 0), Operation::simd_data,
                   vector_destination(0, 64, W), vector_source(5, 1, 64),
                   vector_source(16, 2, 64));
    // Bitwise rows (opcode 3, size selects the operation).
    e = EXPECT_EFFECTS(simd_same(1, 1, 1, 3, 2, 1, 0), Operation::simd_data,
                       vector_destination(0, 128, RW), vector_source(5, 1, 128),
                       vector_source(16, 2, 128));  // BSL reads Vd
    EXPECT_EFFECTS(simd_same(1, 0, 0, 3, 2, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));  // AND writes Vd
    // Multiply-accumulate rows read Vd.
    EXPECT_EFFECTS(simd_same(1, 0, 2, 18, 2, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, RW), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));
    // Floating-point three same and accumulate.
    EXPECT_EFFECTS(simd_same_fp(1, 0, 0, 0, 26, 2, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));  // FADD
    EXPECT_EFFECTS(simd_same_fp(1, 0, 0, 0, 25, 2, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, RW), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));  // FMLA
    EXPECT_EFFECTS(simd_same_fp(1, 1, 0, 0, 31, 2, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));  // FDIV
    // FP16 three same rows live in the bit21 = 0 frame.
    auto fp16 = [](unsigned q, unsigned u, unsigned size, unsigned opcode, unsigned rm,
                   unsigned rn, unsigned rd) {
        return q << 30 | u << 29 | 0b1110u << 24 | size << 22 | rm << 16 | opcode << 11 |
               1u << 10 | rn << 5 | rd;
    };
    EXPECT_EFFECTS(fp16(1, 0, 1, 2, 2, 1, 0), Operation::simd_data,  // FADD .8H
                   vector_destination(0, 128, W), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));
    EXPECT_EFFECTS(fp16(1, 0, 3, 1, 2, 1, 0), Operation::simd_data,  // FMLS .8H
                   vector_destination(0, 128, RW), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));
    EXPECT_STATUS(fp16(1, 0, 1, 5, 2, 1, 0), unknown);  // FP16 opcodes beyond the verified rows

    // Three different: long, narrow and accumulate-long shapes.
    EXPECT_EFFECTS(simd_different(1, 0, 0, 0, 2, 1, 0), Operation::simd_data,  // SADDL
                   vector_destination(0, 128, W), vector_source(5, 1, 64),
                   vector_source(16, 2, 64));
    EXPECT_EFFECTS(simd_different(1, 0, 0, 8, 2, 1, 0), Operation::simd_data,  // ADDHN
                   vector_destination(0, 64, W), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));
    EXPECT_EFFECTS(simd_different(1, 0, 0, 16, 2, 1, 0), Operation::simd_data,  // SMLAL
                   vector_destination(0, 128, RW), vector_source(5, 1, 64),
                   vector_source(16, 2, 64));
    EXPECT_EFFECTS(simd_different(1, 0, 0, 2, 2, 1, 0), Operation::simd_data,  // SADDW
                   vector_destination(0, 128, W), vector_source(5, 1, 128),
                   vector_source(16, 2, 64));

    // Tagged one/two-register rows.
    EXPECT_EFFECTS(simd_tagged(1, 0, 0, 0, 2, 0, 1, 0), Operation::simd_data,  // REV64
                   vector_destination(0, 128, W), vector_source(5, 1, 128));
    EXPECT_EFFECTS(simd_tagged(1, 0, 0, 0, 0x0E, 0, 1, 0), Operation::simd_data,  // SUQADD
                   vector_destination(0, 128, RW), vector_source(5, 1, 128));
    e = EXPECT_EFFECTS(simd_tagged(1, 0, 0, 0, 0x22, 0, 1, 0), Operation::simd_data,
                       vector_destination(0, 128, W), vector_source(5, 1, 128));  // CMGT #0
    CHECK(!e.reads_flags && !e.writes_flags);  // vector compares never touch NZCV
    EXPECT_EFFECTS(simd_tagged(1, 0, 1, 0, 0x32, 0, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128));  // FCMGT #0.0
    // Across-lane reductions write an element-width result.
    EXPECT_EFFECTS(simd_tagged(1, 0, 1, 0, 0x2E, 0x11, 1, 0), Operation::simd_data,
                   vector_destination(0, 32, W), vector_source(5, 1, 128));  // ADDV S0, V1.4S
    EXPECT_EFFECTS(simd_tagged(0, 0, 0, 1, 0x0E, 0x10, 1, 0), Operation::simd_data,
                   vector_destination(0, 16, W), vector_source(5, 1, 64));  // SADDLV H0, V1.4H
    EXPECT_EFFECTS(simd_tagged(1, 1, 1, 0, 0x3E, 0x10, 1, 0), Operation::simd_data,
                   vector_destination(0, 32, W), vector_source(5, 1, 128));  // FMAXV S0, V1.4S
    // AES rounds.
    EXPECT_EFFECTS(0x4E284820u, Operation::simd_data,
                   vector_destination(0, 128, RW), vector_source(5, 1, 128));  // AESE V0.16B, V1.16B

    // Permute, EXT, table lookup and copy element.
    auto permute = [](unsigned q, unsigned size, unsigned opcode, unsigned rm, unsigned rn,
                      unsigned rd) {
        return q << 30 | 0x0E000000u | size << 22 | rm << 16 | opcode << 11 | rn << 5 | rd;
    };
    EXPECT_EFFECTS(permute(1, 0, 7, 2, 1, 0), Operation::simd_data, vector_destination(0, 128, W),
                   vector_source(5, 1, 128), vector_source(16, 2, 128));  // ZIP1
    auto ext = [](unsigned q, unsigned imm4, unsigned rm, unsigned rn, unsigned rd) {
        return q << 30 | 1u << 29 | 0x0E000000u | rm << 16 | imm4 << 11 | rn << 5 | rd;
    };
    EXPECT_EFFECTS(ext(1, 5, 2, 1, 0), Operation::simd_data, vector_destination(0, 128, W),
                   vector_source(5, 1, 128), vector_source(16, 2, 128));  // EXT
    auto table = [](unsigned q, unsigned len, unsigned rm, unsigned rn, unsigned rd) {
        return q << 30 | 0x0E000000u | len << 13 | rm << 16 | rn << 5 | rd;
    };
    e = EXPECT_EFFECTS(table(1, 1, 3, 1, 0), Operation::simd_data, vector_destination(0, 128, W),
                       vector_source(5, 1, 128),
                       Op{implicit_operand, 2, NA, 128, R, Role::source, false, VEC},
                       vector_source(16, 3, 128));  // TBL with two table registers
    (void)e;
    e = EXPECT_EFFECTS(table(1, 0, 2, 1, 0) | 1u << 12, Operation::simd_data,
                       vector_destination(0, 128, RW), vector_source(5, 1, 128),
                       vector_source(16, 2, 128));  // TBX reads Vd
    // DUP element: the lane lies in the upper half exactly for index >= 8
    // bytes into the register; imm5 = (index << (size code + 1)) | (1 << size code).
    EXPECT_EFFECTS(0x4E140420u, Operation::simd_data, vector_destination(0, 128, W),
                   vector_source(5, 1, 128));  // DUP V0.4S, V1.S[2]
    EXPECT_EFFECTS(simd_copy_extract(1, 0, 1, 0x0A, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 64));  // DUP V0.8H, V1.H[2]
    EXPECT_EFFECTS(simd_copy_extract(1, 0, 1, 0x15, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128));  // DUP V0.16B, V1.B[10]
    // INS element reads and writes Vd; the source index must fit the
    // destination element size.
    EXPECT_EFFECTS(0x6E0C4420u, Operation::simd_data, vector_destination(0, 128, RW),
                   vector_source(5, 1, 128));  // INS V0.S[1], V1.S[2]
    EXPECT_EFFECTS(simd_copy_extract(1, 1, 1, 0x18, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, RW), vector_source(5, 1, 64));  // INS V0.D[1], V1.D[0]
    EXPECT_STATUS(0x6E181E52u, reserved);  // INS with a noncanonical source field
    EXPECT_STATUS(simd_copy_extract(1, 0, 0x11, 0x10, 1, 0), reserved);  // TBL frame bit12
    // SDOT/UDOT accumulate.
    EXPECT_EFFECTS(simd_same(1, 1, 2, 0x12, 2, 1, 0) & ~(1u << 21), Operation::simd_data,
                   vector_destination(0, 128, RW), vector_source(5, 1, 128),
                   vector_source(16, 2, 128));  // UDOT V0.4S, V1.16B, V2.16B

    // Shift by immediate.
    EXPECT_EFFECTS(simd_shift_imm(1, 0, 4, 1, 0x01, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128));  // SSHR #7
    EXPECT_EFFECTS(simd_shift_imm(1, 0, 4, 1, 0x05, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, RW), vector_source(5, 1, 128));  // SSRA
    EXPECT_EFFECTS(simd_shift_imm(0, 0, 3, 1, 0x29, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 64));  // SSHLL (long)
    EXPECT_EFFECTS(simd_shift_imm(1, 0, 4, 1, 0x21, 1, 0), Operation::simd_data,
                   vector_destination(0, 128, W), vector_source(5, 1, 128));  // SHRN (narrow)
    // immh = 0 with a shift opcode leaves the modified-immediate frame;
    // an opcode outside both tables is reserved.
    EXPECT_STATUS(simd_shift_imm(1, 0, 0, 1, 0x07, 1, 0), reserved);
    EXPECT_STATUS(simd_shift_imm(1, 0, 8, 1, 0x29, 1, 0), reserved);  // widening needs <= S

    // Modified immediate: MOVI, MVNI, ORR, BIC and FMOV immediate rows.
    EXPECT_EFFECTS(simd_modified_imm(1, 0, 0x010, 0x01, 0), Operation::simd_data,
                   vector_destination(0, 128, W));  // MOVI
    EXPECT_EFFECTS(simd_modified_imm(1, 1, 0x010, 0x05, 0), Operation::simd_data,
                   vector_destination(0, 128, RW));  // BIC (immediate) reads Vd
    EXPECT_EFFECTS(simd_modified_imm(1, 0, 0x010, 0x3D, 0), Operation::simd_data,
                   vector_destination(0, 128, W));  // FMOV V0.4S, #imm
    EXPECT_EFFECTS(simd_modified_imm(1, 1, 0x010, 0x3D, 0), Operation::simd_data,
                   vector_destination(0, 128, W));  // FMOV V0.8H, #imm
    EXPECT_STATUS(simd_modified_imm(0, 1, 0x010, 0x3D, 0), reserved);  // FMOV H immediate needs Q
    EXPECT_STATUS(simd_modified_imm(1, 0, 0x010, 0x03, 0), reserved);  // unallocated cmode
}

void scalar_simd_fp_groups() {
    // Scalar integer three same (p = 1): ADD S0, S1, S2 with a free size field.
    EXPECT_EFFECTS(scalar_p1_misc(0, 0, 2, 0x21, 2, 1, 0), Operation::simd_scalar_data,
                   vector_destination(0, 32, W), vector_source(5, 1, 32),
                   vector_source(16, 2, 32));
    // Scalar FP zero compares write a vector compare result, not NZCV.
    auto e = EXPECT_EFFECTS(scalar_p1_misc(1, 1, 0, 0x32, 0, 1, 0), Operation::simd_scalar_data,
                            vector_destination(0, 32, W),
                            vector_source(5, 1, 32));  // FCMGT S0, S1, #0.0
    CHECK(!e.reads_flags && !e.writes_flags);
    // FRECPE/FRSQRTE/FRECPX.
    auto reciprocal = [](unsigned u, unsigned bit22, unsigned opcode6, unsigned rn, unsigned rd) {
        return 1u << 30 | u << 29 | 0x1E000000u | 1u << 23 | bit22 << 22 | 1u << 21 | 1u << 16 |
               opcode6 << 10 | rn << 5 | rd;
    };
    EXPECT_EFFECTS(reciprocal(0, 0, 0x36, 1, 0), Operation::simd_scalar_data,
                   vector_destination(0, 32, W), vector_source(5, 1, 32));  // FRECPE S0, S1
    EXPECT_EFFECTS(reciprocal(1, 1, 0x36, 1, 0), Operation::simd_scalar_data,
                   vector_destination(0, 64, W), vector_source(5, 1, 64));  // FRSQRTE D0, D1
    EXPECT_EFFECTS(reciprocal(0, 1, 0x3E, 1, 0), Operation::simd_scalar_data,
                   vector_destination(0, 64, W), vector_source(5, 1, 64));  // FRECPX D0, D1
    // FRECPS/FRSQRTS: two-source rows with a free Rm.
    auto recps = [](unsigned type, unsigned rm, unsigned rn, unsigned rd) {
        return 1u << 30 | 0x1E000000u | type << 22 | 1u << 21 | rm << 16 | 0x3F << 10 | rn << 5 |
               rd;
    };
    EXPECT_EFFECTS(recps(0, 2, 1, 0), Operation::simd_scalar_data, vector_destination(0, 32, W),
                   vector_source(5, 1, 32), vector_source(16, 2, 32));  // FRECPS S0, S1, S2
    EXPECT_EFFECTS(recps(3, 2, 1, 0), Operation::simd_scalar_data, vector_destination(0, 64, W),
                   vector_source(5, 1, 64), vector_source(16, 2, 64));  // FRSQRTS D0, D1, D2

    // Scalar shift by immediate.
    EXPECT_EFFECTS(simd_scalar_shift(0, 8, 5, 0x01, 1, 0), Operation::simd_scalar_data,
                   vector_destination(0, 64, W), vector_source(5, 1, 64));  // SSHR D0, D1, #3
    EXPECT_EFFECTS(simd_scalar_shift(1, 8, 5, 0x11, 1, 0), Operation::simd_scalar_data,
                   vector_destination(0, 64, RW), vector_source(5, 1, 64));  // SRI
    EXPECT_EFFECTS(simd_scalar_shift(0, 8, 5, 0x3F, 1, 0), Operation::simd_scalar_data,
                   vector_destination(0, 64, W), vector_source(5, 1, 64));  // FCVTZS fixed
    EXPECT_STATUS(simd_scalar_shift(0, 8, 5, 0x15, 1, 0), reserved);  // SHL needs U = 1 rows?? SHL is U = 0/opcode 0x15 exists

    // Scalar floating-point: one source, two source, compare, conditional
    // compare and select, immediate, three source.
    auto fp1 = [](unsigned type, unsigned tag, unsigned opcode6, unsigned rn, unsigned rd) {
        return 0x1E000000u | type << 22 | 1u << 21 | tag << 16 | opcode6 << 10 | rn << 5 | rd;
    };
    EXPECT_EFFECTS(fp1(0, 0, 0x10, 1, 0), Operation::fp_data, vector_destination(0, 32, W),
                   vector_source(5, 1, 32));  // FMOV S0, S1
    EXPECT_EFFECTS(fp1(0, 0, 0x30, 1, 0), Operation::fp_data, vector_destination(0, 32, W),
                   vector_source(5, 1, 32));  // FABS S0, S1
    EXPECT_EFFECTS(fp1(0, 2, 0x30, 1, 0), Operation::fp_data, vector_destination(0, 64, W),
                   vector_source(5, 1, 32));  // FCVT D0, S1
    EXPECT_EFFECTS(fp1(0, 3, 0x30, 1, 0), Operation::fp_data, vector_destination(0, 16, W),
                   vector_source(5, 1, 32));  // FCVT H0, S1
    EXPECT_EFFECTS(fp1(3, 2, 0x10, 1, 0), Operation::fp_data, vector_destination(0, 32, W),
                   vector_source(5, 1, 16));  // FCVT S0, H1
    EXPECT_STATUS(fp1(2, 0, 0x10, 1, 0), reserved);  // ftype 10
    auto fp2 = [](unsigned type, unsigned opcode6, unsigned rm, unsigned rn, unsigned rd) {
        return 0x1E000000u | type << 22 | 1u << 21 | rm << 16 | opcode6 << 10 | rn << 5 | rd;
    };
    EXPECT_EFFECTS(fp2(0, 0x0A, 2, 1, 0), Operation::fp_data, vector_destination(0, 32, W),
                   vector_source(5, 1, 32), vector_source(16, 2, 32));  // FADD S0, S1, S2
    EXPECT_EFFECTS(fp2(1, 0x06, 2, 1, 0), Operation::fp_data, vector_destination(0, 64, W),
                   vector_source(5, 1, 64), vector_source(16, 2, 64));  // FDIV D0, D1, D2
    EXPECT_EFFECTS(fp2(0, 0x22, 2, 1, 0), Operation::fp_data, vector_destination(0, 32, W),
                   vector_source(5, 1, 32), vector_source(16, 2, 32));  // FNMUL S0, S1, S2
    // FCMP writes NZCV and names no destination.
    e = EXPECT_EFFECTS(fp2(0, 0x08, 2, 1, 0), Operation::fp_compare, vector_source(5, 1, 32),
                       vector_source(16, 2, 32));
    CHECK(!e.reads_flags && e.writes_flags);
    e = EXPECT_EFFECTS(fp2(0, 0x08, 0, 1, 8), Operation::fp_compare, vector_source(5, 1, 32));
    CHECK(e.writes_flags);  // FCMP S0, #0.0
    EXPECT_EFFECTS(fp2(0, 0x08, 0, 1, 0x18), Operation::fp_compare,
                   vector_source(5, 1, 32));  // FCMPE S0, #0.0
    // FCCMP reads and writes NZCV conditionally.
    auto fccmp = [](unsigned type, unsigned rm, unsigned cond, unsigned bit4, unsigned nzcv,
                    unsigned rn) {
        return 0x1E000000u | type << 22 | 1u << 21 | rm << 16 | cond << 12 | 1u << 10 | bit4 << 4 |
               nzcv | rn << 5;
    };
    e = EXPECT_EFFECTS(fccmp(0, 2, 0, 0, 2, 1), Operation::fp_conditional_compare,
                       Op{5, 1, NA, 32, R, Role::source, true, VEC},
                       Op{16, 2, NA, 32, R, Role::source, true, VEC});
    CHECK(e.reads_flags && e.writes_flags);
    // FCSEL reads NZCV; its sources are conditional.
    e = EXPECT_EFFECTS(fp2(0, 0x13, 2, 1, 0), Operation::fp_conditional_select,
                       vector_destination(0, 32, W), Op{5, 1, NA, 32, R, Role::source, true, VEC},
                       Op{16, 2, NA, 32, R, Role::source, true, VEC});
    CHECK(e.reads_flags && !e.writes_flags);
    // FMOV immediate.
    EXPECT_EFFECTS(0x1E2E1000u, Operation::fp_immediate, vector_destination(0, 32, W));
    EXPECT_EFFECTS(0x1E6E1000u, Operation::fp_immediate, vector_destination(0, 64, W));
    // FMADD family.
    auto fmadd = [](unsigned type, unsigned rm, unsigned o1, unsigned ra, unsigned o0, unsigned rn,
                    unsigned rd) {
        return 0x1F000000u | type << 22 | o1 << 21 | rm << 16 | o0 << 15 | ra << 10 | rn << 5 |
               rd;
    };
    EXPECT_EFFECTS(fmadd(0, 2, 0, 3, 0, 1, 0), Operation::fp_data, vector_destination(0, 32, W),
                   vector_source(5, 1, 32), vector_source(16, 2, 32),
                   Op{10, 3, NA, 32, R, Role::accumulator, false, VEC});  // FMADD S0, S1, S2, S3
    e = decode_register_effects(fmadd(0, 2, 0, 3, 0, 1, 0));
    CHECK(e.operands[3].role == Role::accumulator);
    EXPECT_EFFECTS(fmadd(3, 2, 1, 3, 1, 1, 0), Operation::fp_data, vector_destination(0, 16, W),
                   vector_source(5, 1, 16), vector_source(16, 2, 16),
                   Op{10, 3, NA, 16, R, Role::accumulator, false, VEC});  // FNMSUB H0, H1, H2, H3
    EXPECT_STATUS(fmadd(2, 2, 0, 3, 0, 1, 0), reserved);  // ftype 10

    // Scalar crypto without general registers.
    auto sha1h = [](unsigned opcode6, unsigned rn, unsigned rd) {
        return 1u << 30 | 0x1E000000u | 1u << 21 | 8u << 16 | opcode6 << 10 | rn << 5 |
               rd;
    };
    e = EXPECT_EFFECTS(sha1h(0x02, 1, 0), Operation::simd_data, vector_destination(0, 32, W),
                       vector_source(5, 1, 32));  // SHA1H S0, S1
    CHECK(proven_free_of(e, 18));
    auto sha3reg = [](unsigned opcode6, unsigned rm, unsigned rn, unsigned rd) {
        return 1u << 30 | 0x1E000000u | rm << 16 | opcode6 << 10 | rn << 5 | rd;
    };
    EXPECT_EFFECTS(sha3reg(0x00, 2, 1, 0), Operation::simd_data, vector_destination(0, 128, RW),
                   vector_source(5, 1, 32), vector_source(16, 2, 128));  // SHA1C Q0, S1, V2.4S

    // Every decoded word of the scalar FP groups proves x18 freedom.
    for (std::uint32_t word :
         {fp1(0, 0, 0x10, 1, 0), fp2(0, 0x0A, 2, 1, 0), fp2(0, 0x08, 2, 1, 0),
          fccmp(0, 2, 0, 0, 2, 1), fp2(0, 0x13, 2, 1, 0), 0x1E2E1000u}) {
        const auto effects = decode_register_effects(word);
        CHECK(effects.decoded());
        CHECK(proven_free_of(effects, 18));
    }
}

// Properties over the newly covered top-level patterns: a decoded word never
// names a general register and touches NZCV only in the scalar floating-point
// compare groups.
void simd_coverage_properties() {
    test::Lcg random{0x51DFu};
    struct Pattern {
        std::uint32_t mask;
        std::uint32_t value;
    };
    static constexpr Pattern patterns[] = {
        {0x1F000000u, 0x0E000000u},  // vector data processing
        {0x1F000000u, 0x0F000000u},  // shift by immediate, modified immediate
        {0x9F000000u, 0x1E000000u},  // scalar floating point (p = 0)
        {0x9F000000u, 0x1F000000u},  // FP three-source and scalar shift
        {0x9F000000u, 0x5E000000u},  // scalar rows, p = 1, U = 0
        {0x9F000000u, 0x7E000000u},  // scalar rows, p = 1, U = 1
    };
    constexpr int sample = 200'000;
    for (const auto& pattern : patterns) {
        std::uint64_t decoded = 0;
        std::uint64_t flagged = 0;
        for (int index = 0; index < sample; ++index) {
            std::uint32_t word = static_cast<std::uint32_t>(random.next());
            word = (word & ~pattern.mask) | pattern.value;
            const auto e = decode_register_effects(word);
            if (e.status != EffectStatus::decoded) {
                continue;
            }
            if (e.operation == Operation::fmov_general ||
                e.operation == Operation::fcvt_general ||
                e.operation == Operation::scvtf_general ||
                e.operation == Operation::ucvtf_general ||
                e.operation == Operation::dup_general ||
                e.operation == Operation::ins_general ||
                e.operation == Operation::smov || e.operation == Operation::umov) {
                // The general-register moves, conversions and copies were
                // already covered; their operands name general registers by
                // design.
                continue;
            }
            ++decoded;
            CHECK(e.operation == Operation::simd_data ||
                  e.operation == Operation::simd_scalar_data || e.operation == Operation::fp_data ||
                  e.operation == Operation::fp_compare ||
                  e.operation == Operation::fp_conditional_compare ||
                  e.operation == Operation::fp_conditional_select ||
                  e.operation == Operation::fp_immediate);
            bool names_general = false;
            for (std::size_t operand = 0; operand < e.operand_count; ++operand) {
                const auto& op = e.operands[operand];
                names_general = names_general || op.is_general();
                CHECK(op.file == RegisterFile::vector);
                if (op.field != implicit_operand) {
                    CHECK_EQ(op.number, (word >> op.field) & 0x1Fu);
                }
            }
            // No general register is ever named: the masks stay empty.
            CHECK(!names_general);
            CHECK_EQ(e.reads, RegisterMask{0});
            CHECK_EQ(e.writes, RegisterMask{0});
            CHECK(proven_free_of(e, 18) && proven_free_of(e, 9) && proven_free_of(e, 30));
            if (e.reads_flags || e.writes_flags) {
                ++flagged;
                CHECK(e.operation == Operation::fp_compare ||
                      e.operation == Operation::fp_conditional_compare ||
                      e.operation == Operation::fp_conditional_select);
            } else {
                CHECK(e.operation != Operation::fp_compare &&
                      e.operation != Operation::fp_conditional_compare &&
                      e.operation != Operation::fp_conditional_select);
            }
        }
        CHECK(decoded > 1'000);
    }
}

} // namespace

int main() {
    composition_matches_assembler();
    add_subtract_immediate();
    logical_immediate();
    move_wide_immediate();
    bitfield_and_extract();
    logical_and_add_sub_shifted();
    add_sub_extended_register();
    carry_compare_select();
    one_two_three_source();
    hints_and_barriers();
    single_register_loads_and_stores();
    vector_transfers();
    scalar_fp_moves();
    structure_encoding_boundaries();
    simd_copy_encoding_boundaries();
    fp_general_encoding_boundaries();
    writeback_hazards();
    register_pairs();
    exclusive_and_ordered();
    simd_vector_groups();
    scalar_simd_fp_groups();
    simd_coverage_properties();
    classified_integration();
    random_word_properties();
    register_queries();
    return armigo::aarch64::test::finish("aarch64_register_effects_tests");
}
