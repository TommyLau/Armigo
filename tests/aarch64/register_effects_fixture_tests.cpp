// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Uses the system assembler as an independent oracle: every word in
// register_effects_fixture.S was encoded from a mnemonic, and the expected
// effects below were written from the same mnemonics.

#include <armigo/aarch64/register_effects.h>
#include <armigo/aarch64/register_substitution.h>

#include "test_support.h"

#include <cstdint>
#include <initializer_list>
#include <optional>

extern "C" {
extern const std::uint32_t armigo_simd_effect_pairs[];
extern const std::uint32_t armigo_simd_effect_pairs_end[];
extern const std::uint32_t armigo_register_effects_fixture[];
extern const std::uint32_t armigo_register_effects_fixture_end[];
extern const std::uint32_t armigo_simd_data_fixture[];
extern const std::uint32_t armigo_simd_data_fixture_end[];
}

namespace {

using namespace armigo::aarch64;

constexpr RegisterMask regs(std::initializer_list<unsigned> list) {
    RegisterMask mask = 0;
    for (unsigned reg : list) {
        mask |= RegisterMask{1} << reg;
    }
    return mask;
}

// x18 field masks.
constexpr std::uint32_t RD = 0x1Fu;
constexpr std::uint32_t RN = 0x1Fu << 5;
constexpr std::uint32_t RA = 0x1Fu << 10;
constexpr std::uint32_t RM = 0x1Fu << 16;
// register_fields() must make no claim (exception generation).
constexpr std::uint32_t NO_CLAIM = 0xFFFFFFFFu;

// Stack pointer and NZCV use.
constexpr std::uint8_t SPR = 1;
constexpr std::uint8_t SPW = 2;
constexpr std::uint8_t FR = 1;
constexpr std::uint8_t FW = 2;

struct Expected {
    Operation operation;
    RegisterMask reads;
    RegisterMask writes;
    std::uint32_t x18_fields;
    std::uint8_t stack_pointer = 0;
    std::uint8_t flags = 0;
};

#define E(operation, reads, writes, ...) \
    Expected { Operation::operation, regs reads, regs writes, __VA_ARGS__ }

// Mirrors the entry order of register_effects_fixture.S.
const Expected expected[] = {
    // Add/subtract (immediate).
    E(add_immediate, ({}), ({18}), RD, SPR),                 // 0 mov x18, sp
    E(add_immediate, ({18}), ({}), RN, SPW),                 // 1 mov sp, x18
    E(add_immediate, ({}), ({18}), RD, SPR),                 // 2 add w18, wsp, #4095
    E(adds_immediate, ({18}), ({18}), RD | RN, 0, FW),       // 3
    E(adds_immediate, ({18}), ({}), RN, 0, FW),              // 4 cmn w18, #7
    E(sub_immediate, ({}), ({}), 0, SPR | SPW),              // 5 sub sp, sp, #0x40
    E(subs_immediate, ({18}), ({}), RN, 0, FW),              // 6 cmp x18, #0
    // Logical (immediate).
    E(and_immediate, ({18}), ({18}), RD | RN),               // 7
    E(orr_immediate, ({1}), ({}), 0, SPW),                   // 8 orr sp, x1, #0x10
    E(orr_immediate, ({}), ({18}), RD),                      // 9 orr w18, wzr, ...
    E(eor_immediate, ({18}), ({1}), RN),                     // 10
    E(ands_immediate, ({18}), ({}), RN, 0, FW),              // 11 tst w18, #1
    // Move wide.
    E(movn, ({}), ({18}), RD),                               // 12
    E(movz, ({}), ({18}), RD),                               // 13
    E(movk, ({18}), ({18}), RD),                             // 14
    // Bitfield and extract.
    E(sbfm, ({1}), ({18}), RD),                              // 15 sxtw x18, w1
    E(sbfm, ({18}), ({1}), RN),                              // 16 asr x1, x18, #63
    E(bfm, ({1, 18}), ({18}), RD),                           // 17 bfxil x18, x1
    E(bfm, ({1, 18}), ({1}), RN),                            // 18 bfi w1, w18
    E(bfm, ({18}), ({18}), RD),                              // 19 bfc x18
    E(ubfm, ({18}), ({18}), RD | RN),                        // 20 lsr w18, w18, #1
    E(ubfm, ({1}), ({18}), RD),                              // 21 uxtb w18, w1
    E(extr, ({1, 18}), ({18}), RD | RN),                     // 22
    E(extr, ({2, 18}), ({1}), RM),                           // 23
    // Logical (shifted register).
    E(and_shifted, ({1, 2}), ({18}), RD),                    // 24
    E(bic_shifted, ({18, 2}), ({1}), RN),                    // 25
    E(orr_shifted, ({2, 18}), ({1}), RM),                    // 26
    E(orr_shifted, ({18}), ({1}), RM),                       // 27 mov x1, x18
    E(orn_shifted, ({18}), ({18}), RD | RM),                 // 28 mvn w18, w18
    E(eor_shifted, ({18}), ({18}), RD | RN | RM),            // 29
    E(eon_shifted, ({2, 3}), ({1}), 0),                      // 30
    E(ands_shifted, ({18, 1}), ({}), RN, 0, FW),             // 31 tst x18, x1, lsr #2
    E(bics_shifted, ({1, 2}), ({18}), RD, 0, FW),            // 32
    // Add/subtract (shifted register).
    E(add_shifted, ({1, 18}), ({18}), RD | RM),              // 33
    E(adds_shifted, ({18, 2}), ({1}), RN, 0, FW),            // 34
    E(sub_shifted, ({1}), ({18}), RD),                       // 35 neg x18, x1
    E(subs_shifted, ({18}), ({1}), RM, 0, FW),               // 36 negs w1, w18
    E(subs_shifted, ({1, 18}), ({}), RM, 0, FW),             // 37 cmp x1, x18, lsl #1
    // Add/subtract (extended register).
    E(add_extended, ({1}), ({18}), RD, SPR),                 // 38 add x18, sp, x1
    E(add_extended, ({18, 1}), ({}), RN, SPW),               // 39 add sp, x18, w1, uxtw #2
    E(add_extended, ({2, 18}), ({1}), RM),                   // 40
    E(adds_extended, ({1}), ({18}), RD, SPR, FW),            // 41
    E(sub_extended, ({18}), ({}), RM, SPR | SPW),            // 42 sub sp, sp, x18
    E(subs_extended, ({18}), ({}), RM, SPR, FW),             // 43 cmp sp, x18
    E(adds_extended, ({18, 1}), ({}), RN, 0, FW),            // 44 cmn w18, w1, uxtb
    // Add/subtract with carry.
    E(adc, ({1, 2}), ({18}), RD, 0, FR),                     // 45
    E(adcs, ({18, 2}), ({1}), RN, 0, FR | FW),               // 46
    E(sbc, ({2, 18}), ({1}), RM, 0, FR),                     // 47
    E(sbcs, ({1}), ({18}), RD, 0, FR | FW),                  // 48 ngcs w18, w1
    // Conditional compare and select.
    E(ccmp_register, ({18, 1}), ({}), RN, 0, FR | FW),       // 49
    E(ccmn_register, ({1, 18}), ({}), RM, 0, FR | FW),       // 50
    E(ccmp_immediate, ({18}), ({}), RN, 0, FR | FW),         // 51 ccmp x18, #18
    E(ccmn_immediate, ({1}), ({}), 0, 0, FR | FW),           // 52 ccmn x1, #18
    E(csel, ({18, 1}), ({18}), RD | RN, 0, FR),              // 53
    E(csinc, ({2, 18}), ({1}), RM, 0, FR),                   // 54
    E(csinc, ({1}), ({18}), RD, 0, FR),                      // 55 cinc x18, x1
    E(csinv, ({}), ({18}), RD, 0, FR),                       // 56 csetm w18
    E(csinv, ({18}), ({1}), RN | RM, 0, FR),                 // 57 cinv x1, x18
    E(csneg, ({1, 2}), ({18}), RD, 0, FR),                   // 58
    E(csneg, ({18}), ({1}), RN | RM, 0, FR),                 // 59 cneg w1, w18
    // Data processing (2 source).
    E(udiv, ({1, 2}), ({18}), RD),                           // 60
    E(sdiv, ({18, 2}), ({1}), RN),                           // 61
    E(lslv, ({2, 18}), ({1}), RM),                           // 62
    E(lsrv, ({18, 1}), ({18}), RD | RN),                     // 63
    E(asrv, ({18}), ({1}), RN | RM),                         // 64
    E(rorv, ({2, 18}), ({1}), RM),                           // 65
    E(crc32h, ({1, 2}), ({18}), RD),                         // 66
    E(crc32w, ({18, 2}), ({1}), RN),                         // 67
    E(crc32x, ({2, 18}), ({1}), RM),                         // 68
    E(crc32cb, ({2, 18}), ({1}), RM),                        // 69
    E(crc32ch, ({18}), ({18}), RD | RN | RM),                // 70
    E(crc32cw, ({2, 3}), ({1}), 0),                          // 71
    E(crc32cx, ({1, 2}), ({18}), RD),                        // 72
    // Data processing (1 source).
    E(rbit, ({1}), ({18}), RD),                              // 73
    E(rev16, ({18}), ({1}), RN),                             // 74
    E(rev32, ({18}), ({18}), RD | RN),                       // 75
    E(rev, ({1}), ({18}), RD),                               // 76
    E(rev, ({18}), ({1}), RN),                               // 77
    E(clz, ({18}), ({18}), RD | RN),                         // 78
    E(cls, ({18}), ({1}), RN),                               // 79
    // Data processing (3 source).
    E(madd, ({1, 2, 3}), ({18}), RD),                        // 80
    E(msub, ({18, 2, 3}), ({1}), RN),                        // 81
    E(madd, ({2, 18}), ({1}), RM),                           // 82 mul x1, x2, x18
    E(msub, ({1, 2}), ({18}), RD),                           // 83 mneg x18, x1, x2
    E(madd, ({2, 3, 18}), ({1}), RA),                        // 84
    E(smaddl, ({1, 2, 3}), ({18}), RD),                      // 85
    E(smaddl, ({18, 2}), ({1}), RN),                         // 86 smull x1, w18, w2
    E(smsubl, ({2, 3, 18}), ({1}), RA),                      // 87
    E(smsubl, ({2, 18}), ({1}), RM),                         // 88 smnegl x1, w2, w18
    E(umaddl, ({2, 3, 18}), ({1}), RA),                      // 89
    E(umaddl, ({1, 2}), ({18}), RD),                         // 90 umull x18, w1, w2
    E(umsubl, ({18, 2, 3}), ({1}), RN),                      // 91
    E(umsubl, ({18}), ({18}), RD | RN | RM),                 // 92 umnegl x18, w18, w18
    E(smulh, ({1, 2}), ({18}), RD),                          // 93
    E(umulh, ({2, 18}), ({1}), RM),                          // 94
    // Hints and barriers.
    E(nop, ({}), ({}), 0),                                   // 95
    E(dmb, ({}), ({}), 0),                                   // 96
    E(dsb, ({}), ({}), 0),                                   // 97
    E(isb, ({}), ({}), 0),                                   // 98
    E(clrex, ({}), ({}), 0),                                 // 99
    // Load/store register (unsigned offset).
    E(ldr, ({}), ({18}), RD, SPR),                           // 100
    E(ldr, ({18}), ({1}), RN),                               // 101
    E(str, ({18}), ({}), RD | RN),                           // 102 str x18, [x18, #8]
    E(strb, ({18}), ({}), RD, SPR),                          // 103
    E(ldrh, ({1}), ({18}), RD),                              // 104
    E(ldrsh, ({18}), ({1}), RN),                             // 105
    E(ldrsb, ({1}), ({18}), RD),                             // 106
    E(ldrsw, ({18}), ({1}), RN),                             // 107
    E(strh, ({18}), ({}), RN),                               // 108 strh wzr, [x18, #2]
    E(prfm, ({18}), ({}), RN),                               // 109
    E(prfm, ({1}), ({}), 0),                                 // 110 operation 18
    // Unscaled and unprivileged.
    E(ldr, ({1}), ({18}), RD),                               // 111 ldur
    E(str, ({1, 18}), ({}), RN),                             // 112 stur
    E(ldrb, ({}), ({18}), RD, SPR),                          // 113 ldurb
    E(ldrsw, ({18}), ({18}), RD | RN),                       // 114 ldursw
    E(prfm, ({18}), ({}), RN),                               // 115 prfum
    E(ldr, ({1}), ({18}), RD),                               // 116 ldtr
    E(strh, ({1, 18}), ({}), RN),                            // 117 sttrh
    E(ldrsb, ({1}), ({18}), RD),                             // 118 ldtrsb
    // Pre- and post-index.
    E(ldr, ({1}), ({1, 18}), RD),                            // 119 ldr x18, [x1, #8]!
    E(str, ({1, 18}), ({18}), RN),                           // 120 str w1, [x18], #-4
    E(stp, ({29, 30}), ({}), 0, SPR | SPW),                  // 121
    E(ldp, ({}), ({29, 30}), 0, SPR | SPW),                  // 122
    E(str, ({18}), ({}), RD, SPR | SPW),                     // 123
    E(ldr, ({}), ({18}), RD, SPR | SPW),                     // 124
    E(ldrb, ({18}), ({1, 18}), RN),                          // 125
    // Register offset.
    E(ldr, ({2, 18}), ({1}), RM),                            // 126
    E(ldr, ({1, 2}), ({18}), RD),                            // 127
    E(str, ({1, 18, 2}), ({}), RN),                          // 128
    E(ldrsw, ({18}), ({1}), RM, SPR),                        // 129
    E(ldrb, ({2, 18}), ({1}), RM),                           // 130
    E(strh, ({18, 1, 2}), ({}), RD),                         // 131
    E(prfm, ({1, 18}), ({}), RM),                            // 132
    // SIMD&FP transfers: V18 is not x18.
    E(ldr_vector, ({1}), ({}), 0),                           // 133 ldr q18, [x1]
    E(str_vector, ({18}), ({}), RN),                         // 134
    E(ldr_vector, ({18, 2}), ({}), RN),                      // 135
    E(str_vector, ({2, 18}), ({}), RM),                      // 136
    E(ldr_vector, ({18}), ({18}), RN),                       // 137 ldr s18, [x18], #4
    E(ldr_vector, ({18}), ({}), RN),                         // 138
    // Register pairs.
    E(ldp, ({2}), ({18, 1}), RD),                            // 139
    E(ldp, ({2}), ({1, 18}), RA),                            // 140
    E(stp, ({1, 2, 18}), ({}), RN),                          // 141
    E(stp, ({18, 1}), ({}), RD | RA),                        // 142 stp w18, w18, [x1]
    E(ldp, ({18}), ({1, 2, 18}), RN),                        // 143
    E(ldpsw, ({}), ({18, 1}), RD, SPR),                      // 144
    E(stnp, ({18, 1, 2}), ({}), RD),                         // 145
    E(ldnp, ({18}), ({1, 2}), RN),                           // 146
    E(ldp_vector, ({18}), ({}), RN),                         // 147 ldp q18, q1, [x18]
    E(stp_vector, ({18}), ({18}), RN),                       // 148
    E(ldnp_vector, ({1}), ({}), 0),                          // 149 ldnp s1, s18, [x1]
    E(stnp_vector, ({18}), ({}), RN),                        // 150
    // Classified by instruction.h.
    E(classified, ({18}), ({}), RD),                         // 151 cbz x18
    E(classified, ({18}), ({}), RN),                         // 152 br x18
    E(classified, ({18}), ({30}), RN),                       // 153 blr x18
    E(classified, ({30}), ({}), 0),                          // 154 ret
    E(classified, ({}), ({18}), RD),                         // 155 mrs x18, tpidr_el0
    E(classified, ({}), ({18}), RD),                         // 156 adrp x18
    E(classified, ({}), ({}), NO_CLAIM),                     // 157 svc #0
    // Exclusive and ordered accesses.
    E(ldxr, ({1}), ({18}), RD),                              // 158
    E(ldaxrb, ({18}), ({1}), RN),                            // 159
    E(stxr, ({1, 2}), ({18}), RM),                           // 160 status w18
    E(stlxrh, ({18}), ({1}), RD, SPR),                       // 161
    E(ldxp, ({2}), ({18, 1}), RD),                           // 162
    E(stlxp, ({2, 18, 3}), ({1}), RA),                       // 163
    E(stxp, ({3, 4, 18}), ({2}), RN),                        // 164
    E(ldar, ({1}), ({18}), RD),                              // 165
    E(stlr, ({1, 18}), ({}), RN),                            // 166
    E(ldarb, ({}), ({1}), 0, SPR),                           // 167
};

#undef E

// Entries whose mnemonic names register 18 only as W18, and the entry that
// names it as both X18 and W18; every other x18 operand is an X view.
constexpr std::size_t w18_entries[] = {
    2,  4,  9,  11, 13, 18, 20, 21, 25, 28, 34, 36, 40, 44, 46,  48,  50,  54,  56,  59,
    60, 63, 65, 66, 67, 69, 70, 72, 73, 77, 78, 81, 86, 88, 91, 103, 104, 113, 118, 127,
    129, 131, 142, 160, 161, 165,
};
constexpr std::size_t mixed_x18_entry = 92;  // umnegl x18, w18, w18

// Bit 0: some operand naming x18 is a 32-bit view; bit 1: a 64-bit view.
unsigned x18_views(const RegisterEffects& effects) {
    unsigned views = 0;
    for (std::size_t index = 0; index < effects.operand_count; ++index) {
        const auto& operand = effects.operands[index];
        if (operand.names_general_register(18)) {
            views |= operand.bits == 32 ? 1u : operand.bits == 64 ? 2u : 4u;
        }
    }
    return views;
}

unsigned expected_x18_views(std::size_t entry, const Expected& expect) {
    if (expect.x18_fields == 0 || expect.x18_fields == NO_CLAIM) {
        return 0;
    }
    if (entry == mixed_x18_entry) {
        return 3;
    }
    for (std::size_t w18 : w18_entries) {
        if (w18 == entry) {
            return 1;
        }
    }
    return 2;
}

void fixture_effects() {
    const auto entries = static_cast<std::size_t>(armigo_register_effects_fixture_end -
                                                  armigo_register_effects_fixture);
    CHECK_EQ(entries, std::size(expected));
    for (std::size_t index = 0; index < entries && index < std::size(expected); ++index) {
        const std::uint32_t word = armigo_register_effects_fixture[index];
        const auto e = decode_register_effects(word);
        const auto& expect = expected[index];
        const auto fields = register_fields(e, 18);
        const bool ok =
            CHECK(e.status == EffectStatus::decoded) && CHECK(e.operation == expect.operation) &&
            CHECK_EQ(e.reads, expect.reads) && CHECK_EQ(e.writes, expect.writes) &&
            CHECK(expect.x18_fields == NO_CLAIM ? !fields.has_value()
                                                : fields == std::optional<std::uint32_t>{
                                                                expect.x18_fields}) &&
            CHECK(proven_free_of(e, 18) == (expect.x18_fields == 0)) &&
            CHECK(reads_stack_pointer(e) == ((expect.stack_pointer & SPR) != 0)) &&
            CHECK(writes_stack_pointer(e) == ((expect.stack_pointer & SPW) != 0)) &&
            CHECK(e.reads_flags == ((expect.flags & FR) != 0)) &&
            CHECK(e.writes_flags == ((expect.flags & FW) != 0)) &&
            CHECK_EQ(x18_views(e), expected_x18_views(index, expect));
        if (!ok) {
            std::fprintf(stderr, "  entry %zu word 0x%08x\n", index, word);
        }
    }
}

struct SimdExpected {
    Operation operation;
    RegisterMask reads;
    RegisterMask writes;
    std::uint32_t fields;
    unsigned general_bits;
    unsigned vector_count;
    unsigned vector_bits;
    Access vector_access;
    unsigned bytes;
    unsigned offset;
    AddressingMode addressing;
    unsigned stack_pointer;
};

// Expected semantics follow the mnemonics, including retained vector lanes.
const SimdExpected simd_expected[] = {
    // st1 { v18.1d }, [x18]
    {Operation::st1_multiple, regs({18}), regs({}), RN, 64, 1, 64,
     Access::read, 8, 0, AddressingMode::base_register, 0},
    // st1 { v18.1d }, [x18], #8
    {Operation::st1_multiple, regs({18}), regs({18}), RN, 64, 1, 64,
     Access::read, 8, 8, AddressingMode::post_index, 0},
    // st1 { v18.1d }, [x1], x18
    {Operation::st1_multiple, regs({1, 18}), regs({1}), RM, 64, 1, 64,
     Access::read, 8, 0, AddressingMode::post_index, 0},
    // st1 { v18.2d, v19.2d }, [x18]
    {Operation::st1_multiple, regs({18}), regs({}), RN, 64, 2, 128,
     Access::read, 16, 0, AddressingMode::base_register, 0},
    // st1 { v18.2d, v19.2d }, [x18], #32
    {Operation::st1_multiple, regs({18}), regs({18}), RN, 64, 2, 128,
     Access::read, 16, 32, AddressingMode::post_index, 0},
    // st1 { v18.2d, v19.2d }, [x1], x18
    {Operation::st1_multiple, regs({1, 18}), regs({1}), RM, 64, 2, 128,
     Access::read, 16, 0, AddressingMode::post_index, 0},
    // st1 { v18.4h, v19.4h, v20.4h }, [x18]
    {Operation::st1_multiple, regs({18}), regs({}), RN, 64, 3, 64,
     Access::read, 8, 0, AddressingMode::base_register, 0},
    // st1 { v18.4h, v19.4h, v20.4h }, [x18], #24
    {Operation::st1_multiple, regs({18}), regs({18}), RN, 64, 3, 64,
     Access::read, 8, 24, AddressingMode::post_index, 0},
    // st1 { v18.4h, v19.4h, v20.4h }, [x1], x18
    {Operation::st1_multiple, regs({1, 18}), regs({1}), RM, 64, 3, 64,
     Access::read, 8, 0, AddressingMode::post_index, 0},
    // st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18]
    {Operation::st1_multiple, regs({18}), regs({}), RN, 64, 4, 128,
     Access::read, 16, 0, AddressingMode::base_register, 0},
    // st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18], #64
    {Operation::st1_multiple, regs({18}), regs({18}), RN, 64, 4, 128,
     Access::read, 16, 64, AddressingMode::post_index, 0},
    // st1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x1], x18
    {Operation::st1_multiple, regs({1, 18}), regs({1}), RM, 64, 4, 128,
     Access::read, 16, 0, AddressingMode::post_index, 0},
    // st2 { v18.2s, v19.2s }, [x18]
    {Operation::st2_multiple, regs({18}), regs({}), RN, 64, 2, 64,
     Access::read, 8, 0, AddressingMode::base_register, 0},
    // st2 { v18.2s, v19.2s }, [x18], #16
    {Operation::st2_multiple, regs({18}), regs({18}), RN, 64, 2, 64,
     Access::read, 8, 16, AddressingMode::post_index, 0},
    // st2 { v18.2s, v19.2s }, [x1], x18
    {Operation::st2_multiple, regs({1, 18}), regs({1}), RM, 64, 2, 64,
     Access::read, 8, 0, AddressingMode::post_index, 0},
    // st3 { v18.8h, v19.8h, v20.8h }, [x18]
    {Operation::st3_multiple, regs({18}), regs({}), RN, 64, 3, 128,
     Access::read, 16, 0, AddressingMode::base_register, 0},
    // st3 { v18.8h, v19.8h, v20.8h }, [x18], #48
    {Operation::st3_multiple, regs({18}), regs({18}), RN, 64, 3, 128,
     Access::read, 16, 48, AddressingMode::post_index, 0},
    // st3 { v18.8h, v19.8h, v20.8h }, [x1], x18
    {Operation::st3_multiple, regs({1, 18}), regs({1}), RM, 64, 3, 128,
     Access::read, 16, 0, AddressingMode::post_index, 0},
    // st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18]
    {Operation::st4_multiple, regs({18}), regs({}), RN, 64, 4, 128,
     Access::read, 16, 0, AddressingMode::base_register, 0},
    // st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18], #64
    {Operation::st4_multiple, regs({18}), regs({18}), RN, 64, 4, 128,
     Access::read, 16, 64, AddressingMode::post_index, 0},
    // st4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x1], x18
    {Operation::st4_multiple, regs({1, 18}), regs({1}), RM, 64, 4, 128,
     Access::read, 16, 0, AddressingMode::post_index, 0},
    // st1 { v18.b }[15], [x18]
    {Operation::st1_single, regs({18}), regs({}), RN, 64, 1, 128,
     Access::read, 1, 0, AddressingMode::base_register, 0},
    // st1 { v18.b }[15], [x18], #1
    {Operation::st1_single, regs({18}), regs({18}), RN, 64, 1, 128,
     Access::read, 1, 1, AddressingMode::post_index, 0},
    // st1 { v18.b }[15], [x1], x18
    {Operation::st1_single, regs({1, 18}), regs({1}), RM, 64, 1, 128,
     Access::read, 1, 0, AddressingMode::post_index, 0},
    // st2 { v18.h, v19.h }[6], [x18]
    {Operation::st2_single, regs({18}), regs({}), RN, 64, 2, 128,
     Access::read, 2, 0, AddressingMode::base_register, 0},
    // st2 { v18.h, v19.h }[6], [x18], #4
    {Operation::st2_single, regs({18}), regs({18}), RN, 64, 2, 128,
     Access::read, 2, 4, AddressingMode::post_index, 0},
    // st2 { v18.h, v19.h }[6], [x1], x18
    {Operation::st2_single, regs({1, 18}), regs({1}), RM, 64, 2, 128,
     Access::read, 2, 0, AddressingMode::post_index, 0},
    // st3 { v18.s, v19.s, v20.s }[3], [x18]
    {Operation::st3_single, regs({18}), regs({}), RN, 64, 3, 128,
     Access::read, 4, 0, AddressingMode::base_register, 0},
    // st3 { v18.s, v19.s, v20.s }[3], [x18], #12
    {Operation::st3_single, regs({18}), regs({18}), RN, 64, 3, 128,
     Access::read, 4, 12, AddressingMode::post_index, 0},
    // st3 { v18.s, v19.s, v20.s }[3], [x1], x18
    {Operation::st3_single, regs({1, 18}), regs({1}), RM, 64, 3, 128,
     Access::read, 4, 0, AddressingMode::post_index, 0},
    // st4 { v18.d, v19.d, v20.d, v21.d }[1], [x18]
    {Operation::st4_single, regs({18}), regs({}), RN, 64, 4, 128,
     Access::read, 8, 0, AddressingMode::base_register, 0},
    // st4 { v18.d, v19.d, v20.d, v21.d }[1], [x18], #32
    {Operation::st4_single, regs({18}), regs({18}), RN, 64, 4, 128,
     Access::read, 8, 32, AddressingMode::post_index, 0},
    // st4 { v18.d, v19.d, v20.d, v21.d }[1], [x1], x18
    {Operation::st4_single, regs({1, 18}), regs({1}), RM, 64, 4, 128,
     Access::read, 8, 0, AddressingMode::post_index, 0},
    // ld1 { v18.1d }, [x18]
    {Operation::ld1_multiple, regs({18}), regs({}), RN, 64, 1, 64,
     Access::write, 8, 0, AddressingMode::base_register, 0},
    // ld1 { v18.1d }, [x18], #8
    {Operation::ld1_multiple, regs({18}), regs({18}), RN, 64, 1, 64,
     Access::write, 8, 8, AddressingMode::post_index, 0},
    // ld1 { v18.1d }, [x1], x18
    {Operation::ld1_multiple, regs({1, 18}), regs({1}), RM, 64, 1, 64,
     Access::write, 8, 0, AddressingMode::post_index, 0},
    // ld1 { v18.2d, v19.2d }, [x18]
    {Operation::ld1_multiple, regs({18}), regs({}), RN, 64, 2, 128,
     Access::write, 16, 0, AddressingMode::base_register, 0},
    // ld1 { v18.2d, v19.2d }, [x18], #32
    {Operation::ld1_multiple, regs({18}), regs({18}), RN, 64, 2, 128,
     Access::write, 16, 32, AddressingMode::post_index, 0},
    // ld1 { v18.2d, v19.2d }, [x1], x18
    {Operation::ld1_multiple, regs({1, 18}), regs({1}), RM, 64, 2, 128,
     Access::write, 16, 0, AddressingMode::post_index, 0},
    // ld1 { v18.4h, v19.4h, v20.4h }, [x18]
    {Operation::ld1_multiple, regs({18}), regs({}), RN, 64, 3, 64,
     Access::write, 8, 0, AddressingMode::base_register, 0},
    // ld1 { v18.4h, v19.4h, v20.4h }, [x18], #24
    {Operation::ld1_multiple, regs({18}), regs({18}), RN, 64, 3, 64,
     Access::write, 8, 24, AddressingMode::post_index, 0},
    // ld1 { v18.4h, v19.4h, v20.4h }, [x1], x18
    {Operation::ld1_multiple, regs({1, 18}), regs({1}), RM, 64, 3, 64,
     Access::write, 8, 0, AddressingMode::post_index, 0},
    // ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18]
    {Operation::ld1_multiple, regs({18}), regs({}), RN, 64, 4, 128,
     Access::write, 16, 0, AddressingMode::base_register, 0},
    // ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x18], #64
    {Operation::ld1_multiple, regs({18}), regs({18}), RN, 64, 4, 128,
     Access::write, 16, 64, AddressingMode::post_index, 0},
    // ld1 { v18.16b, v19.16b, v20.16b, v21.16b }, [x1], x18
    {Operation::ld1_multiple, regs({1, 18}), regs({1}), RM, 64, 4, 128,
     Access::write, 16, 0, AddressingMode::post_index, 0},
    // ld2 { v18.2s, v19.2s }, [x18]
    {Operation::ld2_multiple, regs({18}), regs({}), RN, 64, 2, 64,
     Access::write, 8, 0, AddressingMode::base_register, 0},
    // ld2 { v18.2s, v19.2s }, [x18], #16
    {Operation::ld2_multiple, regs({18}), regs({18}), RN, 64, 2, 64,
     Access::write, 8, 16, AddressingMode::post_index, 0},
    // ld2 { v18.2s, v19.2s }, [x1], x18
    {Operation::ld2_multiple, regs({1, 18}), regs({1}), RM, 64, 2, 64,
     Access::write, 8, 0, AddressingMode::post_index, 0},
    // ld3 { v18.8h, v19.8h, v20.8h }, [x18]
    {Operation::ld3_multiple, regs({18}), regs({}), RN, 64, 3, 128,
     Access::write, 16, 0, AddressingMode::base_register, 0},
    // ld3 { v18.8h, v19.8h, v20.8h }, [x18], #48
    {Operation::ld3_multiple, regs({18}), regs({18}), RN, 64, 3, 128,
     Access::write, 16, 48, AddressingMode::post_index, 0},
    // ld3 { v18.8h, v19.8h, v20.8h }, [x1], x18
    {Operation::ld3_multiple, regs({1, 18}), regs({1}), RM, 64, 3, 128,
     Access::write, 16, 0, AddressingMode::post_index, 0},
    // ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18]
    {Operation::ld4_multiple, regs({18}), regs({}), RN, 64, 4, 128,
     Access::write, 16, 0, AddressingMode::base_register, 0},
    // ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x18], #64
    {Operation::ld4_multiple, regs({18}), regs({18}), RN, 64, 4, 128,
     Access::write, 16, 64, AddressingMode::post_index, 0},
    // ld4 { v18.4s, v19.4s, v20.4s, v21.4s }, [x1], x18
    {Operation::ld4_multiple, regs({1, 18}), regs({1}), RM, 64, 4, 128,
     Access::write, 16, 0, AddressingMode::post_index, 0},
    // ld1 { v18.b }[15], [x18]
    {Operation::ld1_single, regs({18}), regs({}), RN, 64, 1, 128,
     Access::read_write, 1, 0, AddressingMode::base_register, 0},
    // ld1 { v18.b }[15], [x18], #1
    {Operation::ld1_single, regs({18}), regs({18}), RN, 64, 1, 128,
     Access::read_write, 1, 1, AddressingMode::post_index, 0},
    // ld1 { v18.b }[15], [x1], x18
    {Operation::ld1_single, regs({1, 18}), regs({1}), RM, 64, 1, 128,
     Access::read_write, 1, 0, AddressingMode::post_index, 0},
    // ld2 { v18.h, v19.h }[6], [x18]
    {Operation::ld2_single, regs({18}), regs({}), RN, 64, 2, 128,
     Access::read_write, 2, 0, AddressingMode::base_register, 0},
    // ld2 { v18.h, v19.h }[6], [x18], #4
    {Operation::ld2_single, regs({18}), regs({18}), RN, 64, 2, 128,
     Access::read_write, 2, 4, AddressingMode::post_index, 0},
    // ld2 { v18.h, v19.h }[6], [x1], x18
    {Operation::ld2_single, regs({1, 18}), regs({1}), RM, 64, 2, 128,
     Access::read_write, 2, 0, AddressingMode::post_index, 0},
    // ld3 { v18.s, v19.s, v20.s }[3], [x18]
    {Operation::ld3_single, regs({18}), regs({}), RN, 64, 3, 128,
     Access::read_write, 4, 0, AddressingMode::base_register, 0},
    // ld3 { v18.s, v19.s, v20.s }[3], [x18], #12
    {Operation::ld3_single, regs({18}), regs({18}), RN, 64, 3, 128,
     Access::read_write, 4, 12, AddressingMode::post_index, 0},
    // ld3 { v18.s, v19.s, v20.s }[3], [x1], x18
    {Operation::ld3_single, regs({1, 18}), regs({1}), RM, 64, 3, 128,
     Access::read_write, 4, 0, AddressingMode::post_index, 0},
    // ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x18]
    {Operation::ld4_single, regs({18}), regs({}), RN, 64, 4, 128,
     Access::read_write, 8, 0, AddressingMode::base_register, 0},
    // ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x18], #32
    {Operation::ld4_single, regs({18}), regs({18}), RN, 64, 4, 128,
     Access::read_write, 8, 32, AddressingMode::post_index, 0},
    // ld4 { v18.d, v19.d, v20.d, v21.d }[1], [x1], x18
    {Operation::ld4_single, regs({1, 18}), regs({1}), RM, 64, 4, 128,
     Access::read_write, 8, 0, AddressingMode::post_index, 0},
    // ld1r { v18.16b }, [x18]
    {Operation::ld1r, regs({18}), regs({}), RN, 64, 1, 128,
     Access::write, 1, 0, AddressingMode::base_register, 0},
    // ld1r { v18.16b }, [x18], #1
    {Operation::ld1r, regs({18}), regs({18}), RN, 64, 1, 128,
     Access::write, 1, 1, AddressingMode::post_index, 0},
    // ld1r { v18.16b }, [x1], x18
    {Operation::ld1r, regs({1, 18}), regs({1}), RM, 64, 1, 128,
     Access::write, 1, 0, AddressingMode::post_index, 0},
    // ld2r { v18.4h, v19.4h }, [x18]
    {Operation::ld2r, regs({18}), regs({}), RN, 64, 2, 64,
     Access::write, 2, 0, AddressingMode::base_register, 0},
    // ld2r { v18.4h, v19.4h }, [x18], #4
    {Operation::ld2r, regs({18}), regs({18}), RN, 64, 2, 64,
     Access::write, 2, 4, AddressingMode::post_index, 0},
    // ld2r { v18.4h, v19.4h }, [x1], x18
    {Operation::ld2r, regs({1, 18}), regs({1}), RM, 64, 2, 64,
     Access::write, 2, 0, AddressingMode::post_index, 0},
    // ld3r { v18.4s, v19.4s, v20.4s }, [x18]
    {Operation::ld3r, regs({18}), regs({}), RN, 64, 3, 128,
     Access::write, 4, 0, AddressingMode::base_register, 0},
    // ld3r { v18.4s, v19.4s, v20.4s }, [x18], #12
    {Operation::ld3r, regs({18}), regs({18}), RN, 64, 3, 128,
     Access::write, 4, 12, AddressingMode::post_index, 0},
    // ld3r { v18.4s, v19.4s, v20.4s }, [x1], x18
    {Operation::ld3r, regs({1, 18}), regs({1}), RM, 64, 3, 128,
     Access::write, 4, 0, AddressingMode::post_index, 0},
    // ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x18]
    {Operation::ld4r, regs({18}), regs({}), RN, 64, 4, 64,
     Access::write, 8, 0, AddressingMode::base_register, 0},
    // ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x18], #32
    {Operation::ld4r, regs({18}), regs({18}), RN, 64, 4, 64,
     Access::write, 8, 32, AddressingMode::post_index, 0},
    // ld4r { v18.1d, v19.1d, v20.1d, v21.1d }, [x1], x18
    {Operation::ld4r, regs({1, 18}), regs({1}), RM, 64, 4, 64,
     Access::write, 8, 0, AddressingMode::post_index, 0},
    // ld4 { v31.16b, v0.16b, v1.16b, v2.16b }, [x18], x18
    {Operation::ld4_multiple, regs({18}), regs({18}), RN | RM, 64, 4, 128,
     Access::write, 16, 0, AddressingMode::post_index, 0},
    // st1 { v18.16b }, [sp], x18
    {Operation::st1_multiple, regs({18}), regs({}), RM, 64, 1, 128,
     Access::read, 16, 0, AddressingMode::post_index, 3},
    // ld1 { v18.b }[9], [sp], #1
    {Operation::ld1_single, regs({}), regs({}), 0, 64, 1, 128,
     Access::read_write, 1, 1, AddressingMode::post_index, 3},
    // st2 { v18.d, v19.d }[0], [sp]
    {Operation::st2_single, regs({}), regs({}), 0, 64, 2, 128,
     Access::read, 8, 0, AddressingMode::base_register, 1},
    // smov w18, v18.b[15]
    {Operation::smov, regs({}), regs({18}), RD, 32, 1, 128,
     Access::read, 0, 0, AddressingMode::none, 0},
    // smov x18, v18.h[4]
    {Operation::smov, regs({}), regs({18}), RD, 64, 1, 128,
     Access::read, 0, 0, AddressingMode::none, 0},
    // smov x18, v18.s[0]
    {Operation::smov, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // umov w18, v18.b[0]
    {Operation::umov, regs({}), regs({18}), RD, 32, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // umov w18, v18.h[7]
    {Operation::umov, regs({}), regs({18}), RD, 32, 1, 128,
     Access::read, 0, 0, AddressingMode::none, 0},
    // umov w18, v18.s[2]
    {Operation::umov, regs({}), regs({18}), RD, 32, 1, 128,
     Access::read, 0, 0, AddressingMode::none, 0},
    // umov x18, v18.d[1]
    {Operation::umov, regs({}), regs({18}), RD, 64, 1, 128,
     Access::read, 0, 0, AddressingMode::none, 0},
    // ins v18.b[15], w18
    {Operation::ins_general, regs({18}), regs({}), RN, 32, 1, 128,
     Access::read_write, 0, 0, AddressingMode::none, 0},
    // ins v18.h[4], w18
    {Operation::ins_general, regs({18}), regs({}), RN, 32, 1, 128,
     Access::read_write, 0, 0, AddressingMode::none, 0},
    // ins v18.s[0], w18
    {Operation::ins_general, regs({18}), regs({}), RN, 32, 1, 128,
     Access::read_write, 0, 0, AddressingMode::none, 0},
    // ins v18.d[1], x18
    {Operation::ins_general, regs({18}), regs({}), RN, 64, 1, 128,
     Access::read_write, 0, 0, AddressingMode::none, 0},
    // dup v18.8b, w18
    {Operation::dup_general, regs({18}), regs({}), RN, 32, 1, 64,
     Access::write, 0, 0, AddressingMode::none, 0},
    // dup v18.8h, w18
    {Operation::dup_general, regs({18}), regs({}), RN, 32, 1, 128,
     Access::write, 0, 0, AddressingMode::none, 0},
    // dup v18.2s, w18
    {Operation::dup_general, regs({18}), regs({}), RN, 32, 1, 64,
     Access::write, 0, 0, AddressingMode::none, 0},
    // dup v18.2d, x18
    {Operation::dup_general, regs({18}), regs({}), RN, 64, 1, 128,
     Access::write, 0, 0, AddressingMode::none, 0},
    // fcvtns w18, s18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtnu x18, d18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtas w18, h18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtau x18, s18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtps w18, d18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtpu x18, h18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtms w18, s18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtmu x18, d18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzs w18, h18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzu x18, s18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzs w18, s18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzs w18, s18, #32
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzs x18, d18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzs x18, d18, #64
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzs x18, h18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzs x18, h18, #1
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzu w18, s18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzu w18, s18, #32
    {Operation::fcvt_general, regs({}), regs({18}), RD, 32, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzu x18, d18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzu x18, d18, #64
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzu x18, h18
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fcvtzu x18, h18, #1
    {Operation::fcvt_general, regs({}), regs({18}), RD, 64, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // scvtf s18, w18
    {Operation::scvtf_general, regs({18}), regs({}), RN, 32, 1, 32,
     Access::write, 0, 0, AddressingMode::none, 0},
    // scvtf s18, w18, #32
    {Operation::scvtf_general, regs({18}), regs({}), RN, 32, 1, 32,
     Access::write, 0, 0, AddressingMode::none, 0},
    // scvtf d18, x18
    {Operation::scvtf_general, regs({18}), regs({}), RN, 64, 1, 64,
     Access::write, 0, 0, AddressingMode::none, 0},
    // scvtf d18, x18, #64
    {Operation::scvtf_general, regs({18}), regs({}), RN, 64, 1, 64,
     Access::write, 0, 0, AddressingMode::none, 0},
    // scvtf h18, x18
    {Operation::scvtf_general, regs({18}), regs({}), RN, 64, 1, 16,
     Access::write, 0, 0, AddressingMode::none, 0},
    // scvtf h18, x18, #1
    {Operation::scvtf_general, regs({18}), regs({}), RN, 64, 1, 16,
     Access::write, 0, 0, AddressingMode::none, 0},
    // ucvtf s18, w18
    {Operation::ucvtf_general, regs({18}), regs({}), RN, 32, 1, 32,
     Access::write, 0, 0, AddressingMode::none, 0},
    // ucvtf s18, w18, #32
    {Operation::ucvtf_general, regs({18}), regs({}), RN, 32, 1, 32,
     Access::write, 0, 0, AddressingMode::none, 0},
    // ucvtf d18, x18
    {Operation::ucvtf_general, regs({18}), regs({}), RN, 64, 1, 64,
     Access::write, 0, 0, AddressingMode::none, 0},
    // ucvtf d18, x18, #64
    {Operation::ucvtf_general, regs({18}), regs({}), RN, 64, 1, 64,
     Access::write, 0, 0, AddressingMode::none, 0},
    // ucvtf h18, x18
    {Operation::ucvtf_general, regs({18}), regs({}), RN, 64, 1, 16,
     Access::write, 0, 0, AddressingMode::none, 0},
    // ucvtf h18, x18, #1
    {Operation::ucvtf_general, regs({18}), regs({}), RN, 64, 1, 16,
     Access::write, 0, 0, AddressingMode::none, 0},
    // fmov w18, h18
    {Operation::fmov_general, regs({}), regs({18}), RD, 32, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fmov h18, w18
    {Operation::fmov_general, regs({18}), regs({}), RN, 32, 1, 16,
     Access::write, 0, 0, AddressingMode::none, 0},
    // fmov x18, h18
    {Operation::fmov_general, regs({}), regs({18}), RD, 64, 1, 16,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fmov h18, x18
    {Operation::fmov_general, regs({18}), regs({}), RN, 64, 1, 16,
     Access::write, 0, 0, AddressingMode::none, 0},
    // fmov w18, s18
    {Operation::fmov_general, regs({}), regs({18}), RD, 32, 1, 32,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fmov s18, w18
    {Operation::fmov_general, regs({18}), regs({}), RN, 32, 1, 32,
     Access::write, 0, 0, AddressingMode::none, 0},
    // fmov x18, d18
    {Operation::fmov_general, regs({}), regs({18}), RD, 64, 1, 64,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fmov d18, x18
    {Operation::fmov_general, regs({18}), regs({}), RN, 64, 1, 64,
     Access::write, 0, 0, AddressingMode::none, 0},
    // fmov x18, v18.d[1]
    {Operation::fmov_general, regs({}), regs({18}), RD, 64, 1, 128,
     Access::read, 0, 0, AddressingMode::none, 0},
    // fmov v18.d[1], x18
    {Operation::fmov_general, regs({18}), regs({}), RN, 64, 1, 128,
     Access::read_write, 0, 0, AddressingMode::none, 0},
};

void simd_fixture_effects() {
    CHECK_EQ(armigo_simd_effect_pairs_end - armigo_simd_effect_pairs, 2 * std::size(simd_expected));
    for (std::size_t i = 0; i < std::size(simd_expected); ++i) {
        const auto word = armigo_simd_effect_pairs[2 * i];
        const auto renamed = armigo_simd_effect_pairs[2 * i + 1];
        const auto& expect = simd_expected[i];
        const auto before = test::failures();
        const auto e = decode_register_effects(word);
        CHECK(e.decoded() && e.operation == expect.operation);
        CHECK_EQ(e.reads, expect.reads);
        CHECK_EQ(e.writes, expect.writes);
        CHECK(register_fields(e, 18) == std::optional<std::uint32_t>{expect.fields});
        CHECK_EQ(e.memory.element_bytes, expect.bytes);
        CHECK_EQ(e.memory.offset, expect.offset);
        CHECK(e.memory.addressing == expect.addressing);
        CHECK(e.memory.writeback == (expect.addressing == AddressingMode::post_index));
        CHECK(reads_stack_pointer(e) == ((expect.stack_pointer & SPR) != 0));
        CHECK(writes_stack_pointer(e) == ((expect.stack_pointer & SPW) != 0));
        CHECK(!e.reads_flags && !e.writes_flags);
        unsigned vector_count = 0;
        for (unsigned operand = 0; operand < e.operand_count; ++operand) {
            const auto& op = e.operands[operand];
            if (op.file == RegisterFile::vector) {
                ++vector_count;
                CHECK_EQ(op.bits, expect.vector_bits);
                CHECK(op.access == expect.vector_access);
            } else {
                CHECK_EQ(op.bits, expect.general_bits);
            }
        }
        CHECK_EQ(vector_count, expect.vector_count);
        if (expect.bytes != 0) {
            CHECK_EQ(e.memory.elements, expect.vector_count);
            CHECK(e.memory.operation == (expect.vector_access == Access::read
                ? MemoryOperation::store : MemoryOperation::load));
        }
        const auto rewritten = substitute_register(word, 18, 9, UnchangedWord::accept);
        CHECK(rewritten.ok());
        CHECK_EQ(rewritten.word, renamed);
        CHECK(verify_register_substitution(word, renamed, 18, 9, UnchangedWord::accept).ok());
        if (test::failures() != before) std::fprintf(stderr, "  SIMD pair %zu: 0x%08x\n", i, word);
    }
}

// Assembled SIMD data-processing words with no general registers: the
// expectation records the operation class, the operand count, the widths
// seen, and whether the word names NZCV. Every entry must prove freedom
// from every general register.
struct SimdDataExpected {
    Operation operation;
    std::uint8_t operands;
    std::uint8_t flag_bits;  // 1 reads NZCV, 2 writes NZCV
    unsigned destination_bits;
    Access destination_access;
};

const SimdDataExpected simd_data_expected[] = {
    {Operation::simd_data, 3, 0, 128, Access::write},      // 0 ADD
    {Operation::simd_data, 3, 0, 128, Access::write},      // 1 SQADD
    {Operation::simd_data, 3, 0, 128, Access::write},      // 2 ADDP
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 3 MLA
    {Operation::simd_data, 3, 0, 128, Access::write},      // 4 AND
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 5 BSL
    {Operation::simd_data, 3, 0, 128, Access::write},      // 6 FADD
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 7 FMLA
    {Operation::simd_data, 3, 0, 128, Access::write},      // 8 FABD half
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 9 FMLS half
    {Operation::simd_data, 3, 0, 128, Access::write},      // 10 SADDL
    {Operation::simd_data, 3, 0, 64, Access::write},       // 11 ADDHN
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 12 SMLAL
    {Operation::simd_data, 3, 0, 128, Access::write},      // 13 SADDW
    {Operation::simd_data, 2, 0, 128, Access::write},      // 14 REV64
    {Operation::simd_data, 2, 0, 128, Access::read_write}, // 15 SUQADD
    {Operation::simd_data, 2, 0, 128, Access::write},      // 16 CMGT zero
    {Operation::simd_data, 2, 0, 128, Access::write},      // 17 ABS
    {Operation::simd_data, 2, 0, 128, Access::write},      // 18 FCMGT zero
    {Operation::simd_data, 2, 0, 128, Access::write},      // 19 FNEG
    {Operation::simd_data, 2, 0, 32, Access::write},       // 20 ADDV
    {Operation::simd_data, 2, 0, 16, Access::write},       // 21 SMAXV
    {Operation::simd_data, 2, 0, 32, Access::write},       // 22 FMAXV
    {Operation::simd_data, 3, 0, 128, Access::write},      // 23 UZP1
    {Operation::simd_data, 3, 0, 128, Access::write},      // 24 ZIP2
    {Operation::simd_data, 3, 0, 128, Access::write},      // 25 EXT
    {Operation::simd_data, 4, 0, 128, Access::write},      // 26 TBL two registers
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 27 TBX
    {Operation::simd_data, 2, 0, 128, Access::write},      // 28 DUP element
    {Operation::simd_data, 2, 0, 128, Access::read_write}, // 29 INS element
    {Operation::simd_data, 2, 0, 128, Access::write},      // 30 SSHR
    {Operation::simd_data, 2, 0, 128, Access::read_write}, // 31 SSRA
    {Operation::simd_data, 2, 0, 128, Access::write},      // 32 SSHLL
    {Operation::simd_data, 2, 0, 64, Access::write},       // 33 SHRN
    {Operation::simd_data, 2, 0, 64, Access::write},       // 34 SQSHRN
    {Operation::simd_data, 1, 0, 128, Access::write},      // 35 MOVI
    {Operation::simd_data, 1, 0, 128, Access::write},      // 36 MVNI
    {Operation::simd_data, 1, 0, 64, Access::write},       // 37 ORR immediate
    {Operation::simd_data, 1, 0, 64, Access::write},       // 38 FMOV immediate
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 39 MLA indexed
    {Operation::simd_data, 3, 0, 128, Access::write},      // 40 SMULL indexed
    {Operation::simd_data, 3, 0, 128, Access::write},      // 41 SQDMULH indexed
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 42 FMLA indexed
    {Operation::simd_data, 3, 0, 128, Access::write},      // 43 FMUL half indexed
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 44 SDOT
    {Operation::simd_data, 2, 0, 128, Access::write},      // 45 AESE
    {Operation::simd_data, 3, 0, 128, Access::read_write}, // 46 SHA1C
    {Operation::simd_scalar_data, 3, 0, 8, Access::write},   // 47 ADD scalar (byte)
    {Operation::fp_data, 3, 0, 32, Access::write},           // 48 FADD
    {Operation::fp_data, 3, 0, 64, Access::write},           // 49 FDIV
    {Operation::simd_scalar_data, 3, 0, 32, Access::write},  // 50 FMULX (scalar three same)
    {Operation::fp_data, 2, 0, 32, Access::write},           // 51 FABS
    {Operation::fp_data, 2, 0, 64, Access::write},           // 52 FCVT D, S
    {Operation::fp_data, 2, 0, 16, Access::write},           // 53 FCVT H, S
    {Operation::fp_data, 2, 0, 32, Access::write},           // 54 FCVT S, H
    {Operation::fp_compare, 2, 2, 0, Access::read},          // 55 FCMP
    {Operation::fp_compare, 1, 2, 0, Access::read},          // 56 FCMPE zero
    {Operation::fp_conditional_compare, 2, 1 | 2, 0, Access::read}, // 57 FCCMP
    {Operation::fp_conditional_select, 3, 1, 32, Access::write},     // 58 FCSEL
    {Operation::fp_immediate, 1, 0, 32, Access::write},      // 59 FMOV immediate
    {Operation::fp_data, 4, 0, 32, Access::write},           // 60 FMADD
    {Operation::simd_scalar_data, 2, 0, 64, Access::write},  // 61 SSHR D
    {Operation::simd_scalar_data, 2, 0, 64, Access::read_write}, // 62 SRI
    {Operation::simd_scalar_data, 2, 0, 16, Access::write},  // 63 SQSHRUN
    {Operation::simd_scalar_data, 2, 0, 32, Access::write},  // 64 SQXTN
    {Operation::simd_scalar_data, 2, 0, 32, Access::write},  // 65 FRECPE
    {Operation::simd_scalar_data, 3, 0, 32, Access::write},  // 66 FRECPS
    {Operation::simd_scalar_data, 2, 0, 64, Access::write},  // 68 FRECPX
    {Operation::simd_scalar_data, 3, 0, 8, Access::write},   // 69 UQADD B
    {Operation::simd_scalar_data, 2, 0, 32, Access::write},  // 70 FCMGT zero
    {Operation::simd_scalar_data, 2, 0, 64, Access::write},  // 71 ADDP D
    {Operation::simd_data, 2, 0, 32, Access::write},         // 72 SHA1H
};

void simd_data_fixture_effects() {
    const auto entries = static_cast<std::size_t>(armigo_simd_data_fixture_end -
                                                 armigo_simd_data_fixture);
    CHECK_EQ(entries, std::size(simd_data_expected));
    for (std::size_t index = 0; index < entries && index < std::size(simd_data_expected);
         ++index) {
        const auto before = test::failures();
        const std::uint32_t word = armigo_simd_data_fixture[index];
        const auto e = decode_register_effects(word);
        const auto before40 = test::failures();
        if (index == 40) {
            for (std::size_t operand = 0; operand < e.operand_count; ++operand) {
                std::fprintf(stderr, "DBG40 op%zu f=%u n=%u b=%u a=%d\n", operand,
                             e.operands[operand].field, e.operands[operand].number,
                             e.operands[operand].bits, (int)e.operands[operand].access);
            }
        }

        const auto& expect = simd_data_expected[index];
        CHECK(e.status == EffectStatus::decoded);
        CHECK(e.operation == expect.operation);
        CHECK_EQ(e.operand_count, expect.operands);
        CHECK_EQ(e.reads, RegisterMask{0});
        CHECK_EQ(e.writes, RegisterMask{0});
        CHECK(e.reads_flags == ((expect.flag_bits & 1) != 0));
        CHECK(e.writes_flags == ((expect.flag_bits & 2) != 0));
        CHECK(proven_free_of(e, 18) && proven_free_of(e, 9) && proven_free_of(e, 30));
        for (unsigned reg = 0; reg < 31; ++reg) {
            CHECK(proven_free_of(e, reg));
        }
        bool saw_destination = false;
        for (std::size_t operand = 0; operand < e.operand_count; ++operand) {
            const auto& op = e.operands[operand];
            CHECK(op.file == RegisterFile::vector);
            if (op.field == 0 || op.field == implicit_operand) {
                saw_destination = saw_destination ||
                                  (includes_write(op.access) &&
                                   (expect.destination_bits == 0 || op.bits == expect.destination_bits || op.field == 16));
            }
        }
        CHECK(saw_destination == (expect.destination_access != Access::read));
        CHECK(e.memory.operation == MemoryOperation::none);
        CHECK(e.control_flow == ControlFlow::none);
        if (test::failures() != before) {
            std::fprintf(stderr, "  SIMD data entry %zu: word 0x%08x delta=%u\n", index, word, test::failures() - before);
        }
    }
}

} // namespace

int main() {
    fixture_effects();
    simd_fixture_effects();
    simd_data_fixture_effects();
    return armigo::aarch64::test::finish("aarch64_register_effects_fixture_tests");
}
