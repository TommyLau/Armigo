// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Uses the system assembler as an independent oracle. Every original and
// expected rewrite in register_substitution_fixture.S was encoded from a
// mnemonic; the outcomes below were written from the same mnemonics.
//
// The native cases then execute an original instruction sequence and its
// assembled renaming from identical machine states, mapping x10's value to
// x12, and require identical results apart from that mapping. They show that
// the renamed fields mean what the decoder says; they do not show that any
// substitution is safe in a larger program.

#include <armigo/aarch64/register_substitution.h>

#include "test_support.h"

#include <cstdint>
#include <cstring>

extern "C" {
extern const std::uint32_t armigo_substitution_pairs[];
extern const std::uint32_t armigo_substitution_pairs_end[];
}

namespace {

using namespace armigo::aarch64;

using E = SubstitutionError;

struct PairExpectation {
    unsigned from;
    unsigned to;
    // Result under UnchangedWord::reject. Under UnchangedWord::accept,
    // same_register and register_absent become unchanged successes.
    SubstitutionError error;
};

// Mirrors the entry order of the pair table.
const PairExpectation pair_expectations[] = {
    {18, 9, E::none},                       // 0 add x18, x1, x2
    {18, 9, E::none},                       // 1 add w1, w18, w2
    {18, 9, E::none},                       // 2 add x18, x18, #1
    {18, 9, E::none},                       // 3 movk w18
    {18, 9, E::none},                       // 4 eor x18, x18, x18
    {18, 9, E::register_absent},            // 5 add x1, x2, #18
    {18, 9, E::register_absent},            // 6 movz x1, #0x12
    {18, 9, E::register_absent},            // 7 ccmp x1, #18
    {18, 9, E::register_absent},            // 8 ubfx x1, x2, #18, #4
    {18, 9, E::register_absent},            // 9 prfm pstl2keep
    {18, 9, E::register_absent},            // 10 tbz x1, #18
    {18, 9, E::none},                       // 11 lsr w18, w1, #18
    {18, 9, E::none},                       // 12 mov x18, sp
    {18, 9, E::none},                       // 13 mov sp, x18
    {18, 9, E::none},                       // 14 mov w18, wzr
    {18, 9, E::none},                       // 15 cmp x18, #0
    {18, 9, E::none},                       // 16 csel
    {18, 9, E::none},                       // 17 mul x1, x18, x18
    {18, 9, E::none},                       // 18 add x18, sp, w1, uxtw
    {18, 9, E::none},                       // 19 ldr x1, [x18, #8]!
    {18, 1, E::register_collision},         // 20
    {18, 9, E::none},                       // 21 str x18, [x1], #16
    {18, 9, E::none},                       // 22 ldp x18, x1, [sp, #16]
    {18, 1, E::register_collision},         // 23
    {18, 9, E::none},                       // 24 stp x18, x18, [sp, #-16]!
    {18, 1, E::register_collision},         // 25
    {18, 9, E::none},                       // 26 index
    {18, 9, E::none},                       // 27 index, W
    {18, 9, E::none},                       // 28 ldr x18, [x18]
    {18, 9, E::none},                       // 29 ldxr
    {18, 9, E::none},                       // 30 stxr w18
    {18, 1, E::register_collision},         // 31
    {18, 1, E::register_collision},         // 32
    {18, 9, E::none},                       // 33 ldr q18, [x18, #32]
    {18, 9, E::none},                       // 34 str d9, [x18]
    {18, 9, E::register_absent},            // 35 ldr s18, [x1]
    {30, 9, E::implicit_operand},           // 36 bl
    {18, 9, E::none},                       // 37 blr x18
    {18, 30, E::register_collision},        // 38
    {30, 9, E::implicit_operand},           // 39 blr x30
    {30, 9, E::none},                       // 40 ret
    {18, 9, E::none},                       // 41 br x18
    {18, 9, E::none},                       // 42 cbnz w18
    {18, 9, E::none},                       // 43 adr x18
    {18, 9, E::none},                       // 44 mrs
    {18, 9, E::none},                       // 45 msr
    {18, 9, E::register_collision},         // 46 add x18, x9, x2
    {18, 18, E::same_register},             // 47
    {18, 9, E::none},                       // 48 ldadd
    {18, 9, E::none},                       // 49 cas
    {30, 9, E::unknown_encoding},           // 50 paciasp
    {18, 9, E::none},                       // 51 fmov d18, x18
    {18, 9, E::handler_defined_registers},  // 52 svc
    {18, 9, E::handler_defined_registers},  // 53 brk #18
};

void test_pairs() {
    const std::size_t words =
        static_cast<std::size_t>(armigo_substitution_pairs_end - armigo_substitution_pairs);
    if (!CHECK_EQ(words, 2 * std::size(pair_expectations))) {
        return;
    }
    for (std::size_t index = 0; index < std::size(pair_expectations); ++index) {
        const PairExpectation& expected = pair_expectations[index];
        const std::uint32_t original = armigo_substitution_pairs[2 * index];
        const std::uint32_t rewritten = armigo_substitution_pairs[2 * index + 1];
        const bool unchanged_case =
            expected.error == E::same_register || expected.error == E::register_absent;
        const unsigned before = test::failures();

        const SubstitutedWord strict =
            substitute_register(original, expected.from, expected.to, UnchangedWord::reject);
        CHECK_EQ(strict.error, expected.error);
        const SubstitutedWord lenient =
            substitute_register(original, expected.from, expected.to, UnchangedWord::accept);
        if (unchanged_case) {
            CHECK(lenient.ok() && !lenient.changed());
            CHECK_EQ(lenient.word, original);
            CHECK_EQ(rewritten, original);
        } else {
            CHECK_EQ(lenient.error, expected.error);
        }
        if (expected.error == E::none) {
            CHECK_EQ(strict.word, rewritten);
            CHECK(strict.changed());
            CHECK(proven_free_of(strict.effects, expected.from));
            CHECK(verify_register_substitution(original, rewritten, expected.from, expected.to,
                                               UnchangedWord::reject)
                      .ok());
        } else {
            CHECK_EQ(rewritten, original);
        }
        if (test::failures() != before) {
            std::fprintf(stderr, "  pair entry %zu: 0x%08x -> 0x%08x\n", index, original,
                         rewritten);
        }
    }
}

// Native execution of renamed sequences.

struct State {
    std::uint64_t x[7];  // x9..x15
    std::uint64_t nzcv;
};

using Harness = void (*)(State*);

constexpr unsigned first_register = 9;
constexpr unsigned from_register = 10;
constexpr unsigned to_register = 12;
constexpr std::uint64_t sentinel = 0x5E17'1E15'DEAD'0010ull;

constexpr std::uint8_t reg(unsigned number) {
    return static_cast<std::uint8_t>(1u << (number - first_register));
}

struct NativeCase {
    const char* name;
    Harness original;
    const std::uint32_t* original_words;
    const std::uint32_t* original_end;
    Harness renamed;
    const std::uint32_t* renamed_words;
    const std::uint32_t* renamed_end;
    std::uint8_t pointers;  // registers holding a buffer address
    std::uint8_t small;     // registers holding a small buffer index
};

#define DECLARE_VARIANT(symbol)                    \
    extern "C" void symbol(State*);                \
    extern "C" const std::uint32_t symbol##_words[]; \
    extern "C" const std::uint32_t symbol##_end[];
#define DECLARE_CASE(name)                         \
    DECLARE_VARIANT(armigo_native_##name##_original) \
    DECLARE_VARIANT(armigo_native_##name##_renamed)

DECLARE_CASE(add_source)
DECLARE_CASE(add_destination)
DECLARE_CASE(w_read_write)
DECLARE_CASE(movk)
DECLARE_CASE(flags)
DECLARE_CASE(multiply)
DECLARE_CASE(immediate_fields)
DECLARE_CASE(load_writeback)
DECLARE_CASE(store_post_index)
DECLARE_CASE(pair)
DECLARE_CASE(index)
DECLARE_CASE(branch)
DECLARE_CASE(exclusive)
DECLARE_CASE(vector)

#define NATIVE_CASE(name, pointers, small)                                              \
    NativeCase {                                                                        \
        #name, armigo_native_##name##_original, armigo_native_##name##_original_words, \
            armigo_native_##name##_original_end, armigo_native_##name##_renamed,       \
            armigo_native_##name##_renamed_words, armigo_native_##name##_renamed_end,  \
            pointers, small                                                             \
    }

const NativeCase native_cases[] = {
    NATIVE_CASE(add_source, 0, 0),
    NATIVE_CASE(add_destination, 0, 0),
    NATIVE_CASE(w_read_write, 0, 0),
    NATIVE_CASE(movk, 0, 0),
    NATIVE_CASE(flags, 0, 0),
    NATIVE_CASE(multiply, 0, 0),
    NATIVE_CASE(immediate_fields, 0, 0),
    NATIVE_CASE(load_writeback, reg(10), 0),
    NATIVE_CASE(store_post_index, reg(9), 0),
    NATIVE_CASE(pair, reg(10) | reg(13), 0),
    NATIVE_CASE(index, reg(11), reg(10)),
    NATIVE_CASE(branch, 0, 0),
    NATIVE_CASE(exclusive, reg(10), 0),
    NATIVE_CASE(vector, reg(10) | reg(13), 0),
};

constexpr std::size_t buffer_words = 64;

struct Run {
    State state;
    std::uint64_t buffer[buffer_words];
};

void prepare(const NativeCase& c, unsigned variant, Run& run) {
    test::Lcg random{0xA11C'E5ED'0000ull + variant};
    for (std::size_t word = 0; word < buffer_words; ++word) {
        run.buffer[word] = random.next() * 0x9E37'79B9ull;
    }
    for (unsigned index = 0; index < 7; ++index) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1u << index);
        std::uint64_t value;
        if (c.pointers & bit) {
            value = reinterpret_cast<std::uint64_t>(&run.buffer[8 * (index + 1)]);
        } else if (c.small & bit) {
            value = variant == 0 ? 0 : (random.next() % 4);
        } else if (variant == 0) {
            value = 0;  // takes CBZ and makes zero-dependent paths visible
        } else if (variant == 1) {
            value = random.next() % 64;
        } else {
            value = (random.next() << 20) ^ random.next();
        }
        run.state.x[index] = value;
    }
    static constexpr std::uint64_t flags[] = {0, 0x6000'0000ull, 0x9000'0000ull};
    run.state.nzcv = flags[variant % 3];
}

void test_native_case(const NativeCase& c) {
    const unsigned before = test::failures();
    const std::size_t original_count = static_cast<std::size_t>(c.original_end - c.original_words);
    const std::size_t renamed_count = static_cast<std::size_t>(c.renamed_end - c.renamed_words);
    if (!CHECK_EQ(original_count, renamed_count)) {
        return;
    }
    // Static: renaming each original word yields the assembled renamed word.
    for (std::size_t index = 0; index < original_count; ++index) {
        const SubstitutedWord result = substitute_register(c.original_words[index], from_register,
                                                           to_register, UnchangedWord::accept);
        CHECK(result.ok());
        CHECK_EQ(result.word, c.renamed_words[index]);
    }

    constexpr unsigned from = from_register - first_register;
    constexpr unsigned to = to_register - first_register;
    for (unsigned variant = 0; variant < 3; ++variant) {
        static Run original;
        static Run renamed;
        prepare(c, variant, original);
        prepare(c, variant, renamed);
        // The renamed code finds x10's value in x12 and no longer names x10.
        // Each run has its own identically initialized buffer, so addresses
        // into the original buffer are rebased onto the renamed one.
        const std::uint64_t original_to = original.state.x[to];
        renamed.state.x[to] = original.state.x[from];
        renamed.state.x[from] = sentinel;
        const auto rebase = [&](std::uint64_t value) {
            const auto base = reinterpret_cast<std::uint64_t>(original.buffer);
            const auto limit = base + sizeof(original.buffer);
            if (value >= base && value <= limit) {
                return value - base + reinterpret_cast<std::uint64_t>(renamed.buffer);
            }
            return value;
        };
        renamed.state.x[to] = rebase(renamed.state.x[to]);

        c.original(&original.state);
        c.renamed(&renamed.state);

        CHECK_EQ(original.state.x[to], original_to);
        CHECK_EQ(renamed.state.x[from], sentinel);
        CHECK_EQ(renamed.state.x[to], rebase(original.state.x[from]));
        for (unsigned index = 0; index < 7; ++index) {
            if (index != from && index != to) {
                CHECK_EQ(renamed.state.x[index], rebase(original.state.x[index]));
            }
        }
        CHECK_EQ(renamed.state.nzcv, original.state.nzcv);
        for (std::size_t word = 0; word < buffer_words; ++word) {
            CHECK_EQ(renamed.buffer[word], rebase(original.buffer[word]));
        }
    }
    if (test::failures() != before) {
        std::fprintf(stderr, "  native case %s\n", c.name);
    }
}

} // namespace

int main() {
    test_pairs();
    for (const NativeCase& c : native_cases) {
        test_native_case(c);
    }
    return armigo::aarch64::test::finish("aarch64.register_substitution_fixture");
}
