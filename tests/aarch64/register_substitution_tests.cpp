// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Portable checks of the register substitution contract. Words are composed
// from the A64 encoding tables by local helpers; expected rewrites are written
// independently of the substitution code. The assembler-encoded fixture
// (register_substitution_fixture_tests.cpp) covers mnemonics and execution.

#include <armigo/aarch64/register_substitution.h>

#include "test_support.h"

#include <cstdint>
#include <optional>

namespace {

using namespace armigo::aarch64;

constexpr std::uint32_t RD = 0x1Fu;
constexpr std::uint32_t RN = 0x1Fu << 5;
constexpr std::uint32_t RA = 0x1Fu << 10;
constexpr std::uint32_t RM = 0x1Fu << 16;

constexpr auto reject = UnchangedWord::reject;
constexpr auto accept = UnchangedWord::accept;

// ADD (immediate): sf 0 0 100010 sh imm12 Rn Rd.
constexpr std::uint32_t add_immediate(bool wide, unsigned rd, unsigned rn, unsigned imm12) {
    return (wide ? 0x91000000u : 0x11000000u) | (imm12 << 10) | (rn << 5) | rd;
}
// ADD (shifted register), LSL #0.
constexpr std::uint32_t add_shifted(bool wide, unsigned rd, unsigned rn, unsigned rm) {
    return (wide ? 0x8B000000u : 0x0B000000u) | (rm << 16) | (rn << 5) | rd;
}
// MADD X.
constexpr std::uint32_t madd(unsigned rd, unsigned rn, unsigned rm, unsigned ra) {
    return 0x9B000000u | (rm << 16) | (ra << 10) | (rn << 5) | rd;
}
// LDR X, pre-index.
constexpr std::uint32_t ldr_pre(unsigned rt, unsigned rn, unsigned imm9) {
    return 0xF8400C00u | ((imm9 & 0x1FFu) << 12) | (rn << 5) | rt;
}
// STXR X.
constexpr std::uint32_t stxr(unsigned rs, unsigned rt, unsigned rn) {
    return 0xC8007C00u | (rs << 16) | (rn << 5) | rt;
}
// LDR Q (unsigned offset).
constexpr std::uint32_t ldr_q(unsigned qt, unsigned rn) { return 0x3DC00000u | (rn << 5) | qt; }
// LDADD X (atomic memory operation; outside decoded coverage).
constexpr std::uint32_t ldadd(unsigned rs, unsigned rt, unsigned rn) {
    return 0xF8200000u | (rs << 16) | (rn << 5) | rt;
}
// LD64B Xt, [Xn]: not decoded.
constexpr std::uint32_t ld64b(unsigned rt, unsigned rn) {
    return 0xF83FD000u | (rn << 5) | rt;
}
// ORR (shifted register) X with Rn = XZR: MOV Xd, Xm.
constexpr std::uint32_t mov_register(unsigned rd, unsigned rm) {
    return 0xAA0003E0u | (rm << 16) | rd;
}
// CBZ X with imm19 = 2.
constexpr std::uint32_t cbz(unsigned rt) { return 0xB4000040u | rt; }
constexpr std::uint32_t blr(unsigned rn) { return 0xD63F0000u | (rn << 5); }
constexpr std::uint32_t ret(unsigned rn) { return 0xD65F0000u | (rn << 5); }
constexpr std::uint32_t bl = 0x94000010u;
constexpr std::uint32_t svc_0 = 0xD4000001u;
// AND (immediate) with sf = 0 and N = 1: reserved.
constexpr std::uint32_t reserved_logical = 0x12400000u | (18u << 5) | 18u;

void expect_error(std::uint32_t word, unsigned from, unsigned to, UnchangedWord unchanged,
                  SubstitutionError error) {
    const SubstitutedWord result = substitute_register(word, from, to, unchanged);
    CHECK_EQ(result.error, error);
    CHECK(!result.ok());
    CHECK_EQ(result.word, 0u);
    CHECK_EQ(result.fields, 0u);
}

void expect_rewrite(std::uint32_t word, unsigned from, unsigned to, std::uint32_t expected,
                    std::uint32_t fields) {
    for (UnchangedWord unchanged : {reject, accept}) {
        const SubstitutedWord result = substitute_register(word, from, to, unchanged);
        CHECK(result.ok());
        CHECK_EQ(result.word, expected);
        CHECK_EQ(result.fields, fields);
        CHECK(result.changed());
        CHECK(result.effects.decoded());
        CHECK_EQ(result.effects.word, expected);
        CHECK(proven_free_of(result.effects, from));
    }
}

void expect_unchanged(std::uint32_t word, unsigned from, unsigned to, SubstitutionError rejected) {
    expect_error(word, from, to, reject, rejected);
    const SubstitutedWord result = substitute_register(word, from, to, accept);
    CHECK(result.ok());
    CHECK(!result.changed());
    CHECK_EQ(result.word, word);
    CHECK_EQ(result.fields, 0u);
    CHECK(result.effects.decoded());
    CHECK_EQ(result.effects.reads, decode_register_effects(word).reads);
}

void test_register_numbers() {
    const std::uint32_t word = add_shifted(true, 18, 1, 2);
    for (UnchangedWord unchanged : {reject, accept}) {
        expect_error(word, 31, 9, unchanged, SubstitutionError::invalid_register);
        expect_error(word, 18, 31, unchanged, SubstitutionError::invalid_register);
        expect_error(word, 32, 9, unchanged, SubstitutionError::invalid_register);
        expect_error(word, 18, 255, unchanged, SubstitutionError::invalid_register);
        expect_error(word, 31, 31, unchanged, SubstitutionError::invalid_register);
    }
    // Register-31 operands are never renamed: MOV X18, SP and MOV X18, XZR.
    expect_rewrite(add_immediate(true, 18, 31, 0), 18, 9, add_immediate(true, 9, 31, 0), RD);
    expect_rewrite(mov_register(18, 31), 18, 9, mov_register(9, 31), RD);
    // Atomic memory operations rename their transfer, operand and base fields.
    expect_rewrite(ldadd(1, 18, 2), 18, 9, ldadd(1, 9, 2), RD);
    expect_rewrite(ldadd(18, 3, 18), 18, 9, ldadd(9, 3, 9), RM | RN);
    expect_unchanged(add_immediate(true, 31, 31, 16), 18, 9, SubstitutionError::register_absent);
}

void test_input_status() {
    for (UnchangedWord unchanged : {reject, accept}) {
        expect_error(ld64b(18, 2), 18, 9, unchanged, SubstitutionError::unknown_encoding);
        // Unknown words are refused even when the field pattern holds no 18.
        expect_error(ld64b(3, 2), 18, 9, unchanged, SubstitutionError::unknown_encoding);
        expect_error(0x00000000u, 0, 9, unchanged, SubstitutionError::unknown_encoding);
        expect_error(reserved_logical, 18, 9, unchanged, SubstitutionError::reserved_encoding);
        // LDR X18, [X18, #8]! is CONSTRAINED UNPREDICTABLE.
        expect_error(ldr_pre(18, 18, 8), 18, 9, unchanged,
                     SubstitutionError::unpredictable_encoding);
        expect_error(svc_0, 18, 9, unchanged, SubstitutionError::handler_defined_registers);
        expect_error(svc_0, 18, 18, unchanged, SubstitutionError::handler_defined_registers);
    }
}

void test_implicit_and_collisions() {
    for (UnchangedWord unchanged : {reject, accept}) {
        // X30 written by BL and BLR has no field.
        expect_error(bl, 30, 9, unchanged, SubstitutionError::implicit_operand);
        expect_error(blr(30), 30, 9, unchanged, SubstitutionError::implicit_operand);
        expect_error(blr(30), 30, 30, unchanged, SubstitutionError::implicit_operand);
        // Renaming into the implicit link register merges two operands.
        expect_error(blr(18), 18, 30, unchanged, SubstitutionError::register_collision);
        // Every other independently named operand.
        expect_error(add_shifted(true, 18, 9, 2), 18, 9, unchanged,
                     SubstitutionError::register_collision);
        expect_error(madd(1, 18, 2, 3), 18, 3, unchanged, SubstitutionError::register_collision);
        expect_error(madd(1, 18, 2, 3), 18, 1, unchanged, SubstitutionError::register_collision);
        // Writeback base against the transfer register, and the exclusive
        // status against the transfer register: either would also make the
        // output CONSTRAINED UNPREDICTABLE.
        expect_error(ldr_pre(1, 18, 8), 18, 1, unchanged, SubstitutionError::register_collision);
        expect_error(stxr(1, 18, 2), 18, 1, unchanged, SubstitutionError::register_collision);
        expect_error(stxr(18, 1, 2), 18, 2, unchanged, SubstitutionError::register_collision);
    }
    // Structure base and post-index are separate GPRs; vector numbers never
    // collide. The immediate-post sentinel 31 cannot be a substitution target.
    const std::uint32_t structure = 0x4CD27252u; // ld1 {v18.16b}, [x18], x18
    expect_rewrite(structure, 18, 9, 0x4CC97132u, RN | RM);
    const std::uint32_t indexed = 0x4CD27032u; // ld1 {v18.16b}, [x1], x18
    expect_error(indexed, 18, 1, reject, SubstitutionError::register_collision);
    expect_error(indexed, 18, 31, reject, SubstitutionError::invalid_register);
    CHECK_EQ(verify_register_substitution(indexed, indexed | RM, 18, 9, reject).error,
             SubstitutionError::effects_mismatch);
    // INS keeps its lane selector (imm5=18) and V18 destination.
    expect_rewrite(0x4E121E52u, 18, 9, 0x4E121D32u, RN);
    expect_unchanged(0x4E121FF2u, 18, 9, SubstitutionError::register_absent);
    expect_rewrite(blr(18), 18, 9, blr(9), RN);
    expect_rewrite(ret(30), 30, 9, ret(9), RN);
    expect_unchanged(bl, 18, 9, SubstitutionError::register_absent);
    expect_rewrite(ldr_pre(1, 18, 8), 18, 2, ldr_pre(1, 2, 8), RN);
    expect_rewrite(stxr(18, 1, 2), 18, 3, stxr(3, 1, 2), RM);
    expect_rewrite(stxr(1, 18, 2), 18, 3, stxr(1, 3, 2), RD);
}

void test_unchanged_policy() {
    // from == to.
    expect_unchanged(add_shifted(true, 18, 1, 2), 18, 18, SubstitutionError::same_register);
    // from absent; immediates whose bits coincide with a field holding 18.
    expect_unchanged(add_immediate(true, 1, 2, 18), 18, 9, SubstitutionError::register_absent);
    expect_unchanged(add_immediate(true, 1, 2, 18u | (18u << 5)), 18, 9,
                     SubstitutionError::register_absent);
    // from absent and to present: nothing to rename, so no collision.
    expect_unchanged(add_shifted(true, 9, 1, 2), 18, 9, SubstitutionError::register_absent);
    // A vector register numbered from is not a general register.
    expect_unchanged(ldr_q(18, 1), 18, 9, SubstitutionError::register_absent);
}

void test_rewrites() {
    expect_rewrite(add_shifted(true, 18, 18, 18), 18, 9, add_shifted(true, 9, 9, 9), RD | RN | RM);
    expect_rewrite(add_shifted(false, 18, 1, 2), 18, 9, add_shifted(false, 9, 1, 2), RD);
    expect_rewrite(add_immediate(true, 18, 18, 18), 18, 9, add_immediate(true, 9, 9, 18), RD | RN);
    expect_rewrite(madd(1, 2, 3, 18), 18, 9, madd(1, 2, 3, 9), RA);
    expect_rewrite(cbz(18), 18, 0, cbz(0), RD);
    expect_rewrite(add_shifted(true, 0, 1, 2), 0, 30, add_shifted(true, 30, 1, 2), RD);
    // The SIMD&FP transfer register keeps its number; only the base changes,
    // and a vector register numbered to is no collision.
    expect_rewrite(ldr_q(18, 18), 18, 9, ldr_q(18, 9), RN);
    expect_rewrite(ldr_q(9, 18), 18, 9, ldr_q(9, 9), RN);

    const SubstitutedWord w = substitute_register(add_shifted(false, 18, 1, 2), 18, 9, reject);
    CHECK_EQ(w.effects.operands[0].bits, 32u);
    CHECK_EQ(w.effects.writes, register_bit(9));
}

void test_verify() {
    const std::uint32_t original = add_immediate(true, 18, 18, 1);
    CHECK(verify_register_substitution(original, add_immediate(true, 9, 9, 1), 18, 9, reject).ok());
    CHECK_EQ(verify_register_substitution(original, add_immediate(true, 9, 9, 2), 18, 9, reject)
                 .error,
             SubstitutionError::changed_outside_fields);
    // Both from fields changed, but not both to to.
    CHECK_EQ(verify_register_substitution(original, add_immediate(true, 9, 10, 1), 18, 9, reject)
                 .error,
             SubstitutionError::effects_mismatch);
    CHECK_EQ(verify_register_substitution(original, add_immediate(true, 9, 31, 1), 18, 9, reject)
                 .error,
             SubstitutionError::effects_mismatch);
    // A rewrite that is CONSTRAINED UNPREDICTABLE.
    CHECK_EQ(verify_register_substitution(ldr_pre(18, 1, 8), ldr_pre(1, 1, 8), 18, 9, reject).error,
             SubstitutionError::output_not_decoded);
    // Unchanged substitutions admit only the original word.
    CHECK(verify_register_substitution(original, original, 18, 18, accept).ok());
    CHECK_EQ(verify_register_substitution(original, add_immediate(true, 9, 9, 1), 18, 18, accept)
                 .error,
             SubstitutionError::changed_outside_fields);
    CHECK_EQ(verify_register_substitution(original, original, 18, 18, reject).error,
             SubstitutionError::same_register);
    CHECK_EQ(verify_register_substitution(stxr(1, 18, 2), stxr(1, 1, 2), 18, 1, reject).error,
             SubstitutionError::register_collision);
    CHECK_EQ(verify_register_substitution(svc_0, svc_0, 18, 9, accept).error,
             SubstitutionError::handler_defined_registers);
}

// Every (from, to) pair on instructions with three distinct fields.
void test_all_register_pairs() {
    for (unsigned from = 0; from < 31; ++from) {
        for (unsigned to = 0; to < 31; ++to) {
            const unsigned a = (from + 1) % 31 == to ? (from + 3) % 31 : (from + 1) % 31;
            const unsigned b = (from + 2) % 31 == to ? (from + 4) % 31 : (from + 2) % 31;
            const std::uint32_t word = madd(from, a, from, b);
            const SubstitutedWord result = substitute_register(word, from, to, reject);
            if (from == to) {
                CHECK_EQ(result.error, SubstitutionError::same_register);
            } else if (to == a || to == b) {
                CHECK_EQ(result.error, SubstitutionError::register_collision);
            } else {
                CHECK(result.ok());
                CHECK_EQ(result.word, madd(to, a, to, b));
                CHECK_EQ(result.fields, RD | RM);
            }
        }
    }
}

bool equal_effects(const RegisterEffects& a, const RegisterEffects& b) {
    if (a.word != b.word || a.status != b.status || a.operation != b.operation ||
        a.operand_count != b.operand_count || a.reads != b.reads || a.writes != b.writes) {
        return false;
    }
    for (std::size_t index = 0; index < a.operand_count; ++index) {
        if (a.operands[index].number != b.operands[index].number ||
            a.operands[index].field != b.operands[index].field) {
            return false;
        }
    }
    return true;
}

// Class patterns from the decoder coverage table, so that random words land
// in decoded classes often enough.
struct Pattern {
    std::uint32_t mask;
    std::uint32_t value;
};
constexpr Pattern patterns[] = {
    {0x1F800000u, 0x11000000u}, {0x1F800000u, 0x12000000u}, {0x1F800000u, 0x12800000u},
    {0x1F800000u, 0x13000000u}, {0x1F000000u, 0x0A000000u}, {0x1F200000u, 0x0B000000u},
    {0x1F200000u, 0x0B200000u}, {0x1FE0FC00u, 0x1A000000u}, {0x1FE00000u, 0x1A400000u},
    {0x1FE00000u, 0x1A800000u}, {0x5FE00000u, 0x1AC00000u}, {0x1F000000u, 0x1B000000u},
    {0x3B000000u, 0x39000000u}, {0x3B200000u, 0x38000000u}, {0x3B200C00u, 0x38200800u},
    {0x3A000000u, 0x28000000u}, {0x3F000000u, 0x08000000u}, {0x7C000000u, 0x34000000u},
    {0xFE000000u, 0xD6000000u}, {0x9F000000u, 0x90000000u}, {0xFFE00000u, 0xD5300000u},
    {0x00000000u, 0x00000000u},
};

SubstitutionError status_error(EffectStatus status) {
    switch (status) {
    case EffectStatus::unknown:
        return SubstitutionError::unknown_encoding;
    case EffectStatus::reserved:
        return SubstitutionError::reserved_encoding;
    case EffectStatus::unpredictable:
        return SubstitutionError::unpredictable_encoding;
    case EffectStatus::decoded:
        break;
    }
    return SubstitutionError::none;
}

void test_random_properties() {
    test::Lcg random{0x5EB5'7171'0001ull};
    unsigned rewritten = 0;
    unsigned collisions = 0;
    for (unsigned iteration = 0; iteration < 400000; ++iteration) {
        const Pattern& pattern = patterns[random.next() % std::size(patterns)];
        const auto bits = static_cast<std::uint32_t>(random.next());
        const std::uint32_t word = (bits & ~pattern.mask) | pattern.value;
        // Favor registers the word actually names.
        const RegisterEffects effects = decode_register_effects(word);
        unsigned from = static_cast<unsigned>(random.next() % 31);
        if (effects.operand_count != 0 && random.next() % 4 != 0) {
            const auto& operand = effects.operands[random.next() % effects.operand_count];
            if (operand.number < 31) {
                from = operand.number;
            }
        }
        const unsigned to = static_cast<unsigned>(random.next() % 31);
        const SubstitutedWord result = substitute_register(word, from, to, accept);

        if (!effects.decoded()) {
            CHECK_EQ(result.error, status_error(effects.status));
            continue;
        }
        if (has(effects.notes, EffectNote::handler_defined_registers)) {
            CHECK_EQ(result.error, SubstitutionError::handler_defined_registers);
            continue;
        }
        const std::optional<std::uint32_t> fields = register_fields(effects, from);
        if (!fields) {
            CHECK_EQ(result.error, SubstitutionError::implicit_operand);
            continue;
        }
        if (from == to || *fields == 0) {
            CHECK(result.ok() && result.word == word && !result.changed());
            continue;
        }
        bool collides = false;
        for (std::size_t index = 0; index < effects.operand_count; ++index) {
            collides = collides || effects.operands[index].names_general_register(to);
        }
        if (collides) {
            ++collisions;
            CHECK_EQ(result.error, SubstitutionError::register_collision);
            continue;
        }
        if (!CHECK(result.ok())) {
            std::fprintf(stderr, "  word 0x%08x x%u -> x%u: error %u\n", word, from, to,
                         static_cast<unsigned>(result.error));
            continue;
        }
        ++rewritten;
        CHECK_EQ(result.fields, *fields);
        CHECK_EQ((result.word ^ word) & ~result.fields, 0u);
        for (unsigned low : {0u, 5u, 10u, 16u}) {
            if ((result.fields >> low & 0x1Fu) != 0) {
                CHECK_EQ(result.word >> low & 0x1Fu, to);
            }
        }
        CHECK(proven_free_of(result.effects, from));
        CHECK(register_fields(result.effects, to) == std::optional<std::uint32_t>{*fields});
        CHECK(equal_effects(result.effects, decode_register_effects(result.word)));
        const SubstitutedWord verified =
            verify_register_substitution(word, result.word, from, to, reject);
        CHECK(verified.ok() && verified.word == result.word && verified.fields == result.fields);
        // Renaming back restores the original word exactly.
        const SubstitutedWord back = substitute_register(result.word, to, from, reject);
        CHECK(back.ok() && back.word == word);
    }
    CHECK(rewritten > 50000);
    CHECK(collisions > 1000);
}

} // namespace

int main() {
    test_register_numbers();
    test_input_status();
    test_implicit_and_collisions();
    test_unchanged_policy();
    test_rewrites();
    test_verify();
    test_all_register_pairs();
    test_random_properties();
    return armigo::aarch64::test::finish("aarch64.register_substitution");
}
