// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Relocation checks. Single words must keep their referenced address and
// every bit outside the displacement field. Sequences are checked by a small
// model, written here from the architectural pseudocode, that runs the
// original instruction at its old address and the sequence at its new one and
// compares control flow, register results and memory access addresses.

#include <armigo/aarch64/relocation.h>

#include "test_support.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

using namespace armigo::aarch64;

constexpr std::uint64_t page_mask = ~std::uint64_t{0xFFF};

// ---------------------------------------------------------------------------
// Test-local model of the instructions a relocation may emit.

struct Access {
    std::uint64_t address = 0;
    std::uint32_t width = 0;
    bool operator==(const Access&) const = default;
};

struct Machine {
    std::array<std::uint64_t, 32> x{};
    std::uint32_t nzcv = 0;  // N=8, Z=4, C=2, V=1
    std::vector<Access> accesses;
};

constexpr std::uint64_t loaded_marker = 0x10AD10AD10AD10ADull;

bool condition_holds(unsigned cond, std::uint32_t flags) {
    const bool n = flags & 8, z = flags & 4, c = flags & 2, v = flags & 1;
    bool result = false;
    switch (cond >> 1) {
    case 0: result = z; break;
    case 1: result = c; break;
    case 2: result = n; break;
    case 3: result = v; break;
    case 4: result = c && !z; break;
    case 5: result = n == v; break;
    case 6: result = n == v && !z; break;
    default: return true;  // AL and NV
    }
    return (cond & 1) ? !result : result;
}

std::uint64_t sext(std::uint64_t value, unsigned bits) {
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    return (value ^ sign) - sign;
}

// Executes one word at pc and returns the next pc, or nullopt for a word the
// model does not cover. Field extraction is written out here rather than
// taken from the library decoder.
std::optional<std::uint64_t> step(Machine& m, std::uint32_t w, std::uint64_t pc) {
    const unsigned rd = w & 31;
    const unsigned rn = (w >> 5) & 31;
    const auto reg = [&](unsigned r) { return r == 31 ? 0 : m.x[r]; };
    const auto set = [&](unsigned r, std::uint64_t value) {
        if (r != 31) {
            m.x[r] = value;
        }
    };
    if ((w & 0x7C000000u) == 0x14000000u) {  // B, BL
        if (w >> 31) {
            m.x[30] = pc + 4;
        }
        return pc + sext(w & 0x3FFFFFF, 26) * 4;
    }
    if ((w & 0xFF000000u) == 0x54000000u) {  // B.cond, BC.cond
        return condition_holds(w & 15, m.nzcv) ? pc + sext((w >> 5) & 0x7FFFF, 19) * 4 : pc + 4;
    }
    if ((w & 0x7E000000u) == 0x34000000u) {  // CBZ, CBNZ
        const std::uint64_t value = (w >> 31) ? reg(rd) : (reg(rd) & 0xFFFFFFFFu);
        const bool taken = ((w >> 24) & 1) ? value != 0 : value == 0;
        return taken ? pc + sext((w >> 5) & 0x7FFFF, 19) * 4 : pc + 4;
    }
    if ((w & 0x7E000000u) == 0x36000000u) {  // TBZ, TBNZ
        const unsigned bit = ((w >> 31) << 5) | ((w >> 19) & 31);
        const bool set_bit = (reg(rd) >> bit) & 1;
        const bool taken = ((w >> 24) & 1) ? set_bit : !set_bit;
        return taken ? pc + sext((w >> 5) & 0x3FFF, 14) * 4 : pc + 4;
    }
    if ((w & 0x1F000000u) == 0x10000000u) {  // ADR, ADRP
        const std::uint64_t imm = sext((((w >> 5) & 0x7FFFF) << 2) | ((w >> 29) & 3), 21);
        set(rd, (w >> 31) ? (pc & page_mask) + (imm << 12) : pc + imm);
        return pc + 4;
    }
    if ((w & 0xFF800000u) == 0x91000000u) {  // ADD (immediate), 64-bit
        const std::uint64_t imm = ((w >> 10) & 0xFFF) << ((w >> 22) & 1 ? 12 : 0);
        if (rd == 31 || rn == 31) {
            return std::nullopt;  // SP forms are never expected here
        }
        set(rd, reg(rn) + imm);
        return pc + 4;
    }
    if ((w & 0x3B000000u) == 0x18000000u) {  // load (literal)
        const unsigned opc = w >> 30;
        const bool vector = (w >> 26) & 1;
        static constexpr std::uint32_t widths[2][4] = {{4, 8, 4, 8}, {4, 8, 16, 0}};
        m.accesses.push_back({pc + sext((w >> 5) & 0x7FFFF, 19) * 4, widths[vector][opc]});
        if (!vector && opc != 3) {
            set(rd, loaded_marker);
        }
        return pc + 4;
    }
    struct UnsignedLoad {
        std::uint32_t base;
        std::uint32_t width;
        bool general;
    };
    static constexpr UnsignedLoad unsigned_loads[] = {
        {0xB9400000u, 4, true},  {0xF9400000u, 8, true}, {0xB9800000u, 4, true},
        {0xF9800000u, 8, false}, {0xBD400000u, 4, false}, {0xFD400000u, 8, false},
        {0x3DC00000u, 16, false},
    };
    for (const auto& load : unsigned_loads) {
        if ((w & 0xFFC00000u) == load.base) {
            if (rn == 31) {
                return std::nullopt;
            }
            m.accesses.push_back({reg(rn) + ((w >> 10) & 0xFFF) * load.width, load.width});
            if (load.general) {
                set(rd, loaded_marker);
            }
            return pc + 4;
        }
    }
    if ((w & 0xFFFFFC1Fu) == 0xD61F0000u) {  // BR
        return reg(rn);
    }
    if ((w & 0xFFFFFC1Fu) == 0xD63F0000u) {  // BLR
        const std::uint64_t target = reg(rn);
        m.x[30] = pc + 4;
        return target;
    }
    return std::nullopt;
}

struct Outcome {
    Machine machine;
    std::uint64_t exit = 0;
    bool valid = false;
};

Outcome run_original(const Machine& start, std::uint32_t word, std::uint64_t from) {
    Outcome outcome{start};
    const auto next = step(outcome.machine, word, from);
    outcome.valid = next.has_value();
    outcome.exit = next.value_or(0);
    return outcome;
}

Outcome run_sequence(const Machine& start, std::span<const std::uint32_t> words, std::uint64_t to) {
    Outcome outcome{start};
    std::uint64_t pc = to;
    const std::uint64_t end = to + 4 * words.size();
    for (std::size_t steps = 0; pc >= to && pc < end; ++steps) {
        if (steps > words.size()) {
            return outcome;  // a loop inside a relocation sequence is a failure
        }
        const auto next = step(outcome.machine, words[(pc - to) / 4], pc);
        if (!next) {
            return outcome;
        }
        pc = *next;
    }
    outcome.exit = pc;
    outcome.valid = true;
    return outcome;
}

// Checks that the sequence at to behaves like word at from for the given
// machine state. Fall-through of the original maps to the sequence end.
// `clobbered` lists registers the sequence may change beyond the original.
bool equivalent(std::uint32_t word, std::uint64_t from, const RelocatedSequence& sequence,
                std::uint64_t to, const Machine& start, std::uint32_t clobbered = 0,
                bool link_may_differ = false) {
    const auto original = run_original(start, word, from);
    const auto moved = run_sequence(start, sequence.view(), to);
    if (!CHECK(original.valid && moved.valid)) {
        return false;
    }
    std::uint64_t expected_exit = original.exit;
    if (expected_exit == from + 4) {
        expected_exit = to + 4 * sequence.count;
    }
    bool ok = CHECK_EQ(moved.exit, expected_exit);
    for (unsigned r = 0; r < 31; ++r) {
        if ((clobbered >> r) & 1) {
            continue;
        }
        if (r == 30 && link_may_differ) {
            ok &= CHECK_EQ(moved.machine.x[30], to + 4 * sequence.count);
            continue;
        }
        ok &= CHECK_EQ(moved.machine.x[r], original.machine.x[r]);
    }
    ok &= CHECK(moved.machine.accesses == original.machine.accesses);
    return ok;
}

Machine seeded_machine(armigo::aarch64::test::Lcg& random) {
    Machine m;
    for (auto& value : m.x) {
        value = random.next() * 0x9E3779B97F4A7C15ull;
    }
    m.x[31] = 0;
    m.nzcv = static_cast<std::uint32_t>(random.next() & 15);
    return m;
}

// ---------------------------------------------------------------------------

struct Template {
    Opcode opcode;
    std::uint32_t word;  // displacement field zero
};

// Hand-written words with all non-displacement fields populated, verified
// against the assembler in assembler_fixture.S for the corresponding forms.
constexpr Template templates[] = {
    {Opcode::b, 0x14000000u},
    {Opcode::bl, 0x94000000u},
    {Opcode::b_cond, 0x5400000Cu},   // b.gt
    {Opcode::bc_cond, 0x54000013u},  // bc.lo
    {Opcode::cbz, 0xB4000012u},      // cbz x18
    {Opcode::cbnz, 0x35000007u},     // cbnz w7
    {Opcode::tbz, 0x36F80003u},      // tbz w3, #31
    {Opcode::tbnz, 0xB7E8001Du},     // tbnz x29, #61
    {Opcode::adr, 0x10000009u},      // adr x9
    {Opcode::adrp, 0x90000011u},     // adrp x17
    {Opcode::ldr_literal_w, 0x18000001u},
    {Opcode::ldr_literal_x, 0x5800001Eu},
    {Opcode::ldrsw_literal, 0x98000012u},
    {Opcode::prfm_literal, 0xD8000005u},
    {Opcode::ldr_literal_s, 0x1C00001Fu},
    {Opcode::ldr_literal_d, 0x5C000008u},
    {Opcode::ldr_literal_q, 0x9C000010u},
};

// Builds the template's word at from referring to target.
std::uint32_t at(const Template& t, std::uint64_t from, std::uint64_t target) {
    const auto relocated = relocate_instruction(t.word, from, from);  // validates the template
    CHECK(relocated.ok());
    Encoded e;
    const auto i = decode(t.word);
    switch (t.opcode) {
    case Opcode::adr: e = encode_adr(i.reg, from, target); break;
    case Opcode::adrp: e = encode_adrp(i.reg, from, target); break;
    default: {
        // Re-target through relocation from a site whose displacement is zero:
        // a zero-displacement word at `target` refers to `target` itself.
        const auto r = relocate_instruction(t.word, target, from);
        e.word = r.word;
        e.error = r.ok() ? EncodeError::none : EncodeError::out_of_range;
        break;
    }
    }
    CHECK(e.ok());
    return e.word;
}

void single_word_preserves_reference_and_fields() {
    armigo::aarch64::test::Lcg random{0xC0FFEE};
    for (const auto& t : templates) {
        const auto range = *displacement_range(t.opcode);
        const std::uint32_t mask = displacement_field_mask(t.opcode);
        CHECK_EQ(t.word & mask, 0);
        for (int iteration = 0; iteration < 2000; ++iteration) {
            const std::uint64_t from = 0x400000000ull + (random.next() % 0x4000000ull) * 4;
            const std::uint64_t page_base = t.opcode == Opcode::adrp ? (from & page_mask) : from;
            const auto steps = static_cast<std::uint64_t>((range.maximum - range.minimum) /
                                                          range.granularity) + 1;
            const std::uint64_t target = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(page_base) + range.minimum +
                static_cast<std::int64_t>(random.next() % steps) * range.granularity);
            const std::uint32_t word = at(t, from, target);
            const auto original = decode(word);

            // Any destination within a quarter of the range stays reachable.
            const std::int64_t spread = (range.maximum / 4) & ~std::int64_t{0xFFF};
            const std::uint64_t to = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(target & ~std::uint64_t{3}) - spread +
                static_cast<std::int64_t>(random.next() % static_cast<std::uint64_t>(spread / 2)) * 4);
            const auto moved = relocate_instruction(word, from, to);
            if (!CHECK(moved.ok())) {
                std::fprintf(stderr, "  opcode %u word 0x%08x from 0x%llx to 0x%llx error %u\n",
                             static_cast<unsigned>(t.opcode), word,
                             static_cast<unsigned long long>(from),
                             static_cast<unsigned long long>(to),
                             static_cast<unsigned>(moved.error));
                continue;
            }
            CHECK_EQ((moved.word ^ word) & ~mask, 0);
            CHECK_EQ(*referenced_address(decode(moved.word), to), *referenced_address(original, from));
            CHECK_EQ(moved.link_value_changed, t.opcode == Opcode::bl && from != to);
        }
    }
}

void single_word_reach_endpoints() {
    for (const auto& t : templates) {
        const auto range = *displacement_range(t.opcode);
        const std::uint64_t base = 0x200000000ull;
        const std::uint64_t from = base - 0x40;
        if (t.opcode == Opcode::adrp) {
            const std::uint64_t target = base;
            const std::uint32_t word = at(t, from, target);
            // Page differences: farthest destination pages on each side.
            const std::uint64_t far_low = target - static_cast<std::uint64_t>(range.maximum);
            const std::uint64_t far_high = target + static_cast<std::uint64_t>(-range.minimum);
            CHECK(relocate_instruction(word, from, far_low + 0xFFC).ok());
            CHECK(relocate_instruction(word, from, far_low - 4).error == RelocationError::out_of_range);
            CHECK(relocate_instruction(word, from, far_high).ok());
            CHECK(relocate_instruction(word, from, far_high + 0x1000).error ==
                  RelocationError::out_of_range);
            continue;
        }
        // displacement = target - to, so to = target - displacement. The
        // targets are chosen so both extreme destinations are aligned (ADR's
        // maximum is odd).
        const std::uint64_t target_max = base + static_cast<std::uint64_t>(range.maximum % 4);
        const std::uint64_t target_min = base;
        const std::uint32_t word_max = at(t, from, target_max);
        const std::uint32_t word_min = at(t, from, target_min);
        const std::uint64_t lowest = target_max - static_cast<std::uint64_t>(range.maximum);
        const std::uint64_t highest = target_min + static_cast<std::uint64_t>(-range.minimum);
        const auto low = relocate_instruction(word_max, from, lowest);
        const auto high = relocate_instruction(word_min, from, highest);
        CHECK(low.ok());
        CHECK(high.ok());
        CHECK_EQ(decode(low.word).displacement, range.maximum);
        CHECK_EQ(decode(high.word).displacement, range.minimum);
        CHECK(relocate_instruction(word_max, from, lowest - 4).error ==
              RelocationError::out_of_range);
        CHECK(relocate_instruction(word_min, from, highest + 4).error ==
              RelocationError::out_of_range);
    }
}

void single_word_rejections() {
    CHECK(relocate_instruction(0x8B120000u, 0x1000, 0x2000).error ==
          RelocationError::unclassified_instruction);  // add x0, x0, x18
    CHECK(relocate_instruction(0xD503201Fu, 0x1000, 0x2000).error ==
          RelocationError::unclassified_instruction);  // nop is not examined
    CHECK(relocate_instruction(0x14000000u, 0x1002, 0x2000).error ==
          RelocationError::misaligned_address);
    CHECK(relocate_instruction(0x14000000u, 0x1000, 0x2002).error ==
          RelocationError::misaligned_address);
    CHECK(relocate_instruction(0x17FFFFFFu, 0, 0x1000).error ==
          RelocationError::address_overflow);  // b .-4 at address zero
    CHECK(relocate_instruction(0x14000002u, 0xFFFFFFFFFFFFFFFCull, 0x1000).error ==
          RelocationError::address_overflow);

    // Position-independent classified words are returned unchanged.
    for (const std::uint32_t word : {0xD4000021u, 0xD53BD052u, 0xD61F0240u, 0xD65F03C0u}) {
        const auto r = relocate_instruction(word, 0x1000, 0x900000);
        CHECK(r.ok() && r.word == word && !r.link_value_changed);
    }
    const auto blr = relocate_instruction(0xD63F0120u, 0x1000, 0x2000);
    CHECK(blr.ok() && blr.word == 0xD63F0120u && blr.link_value_changed);
    CHECK(!relocate_instruction(0xD63F0120u, 0x1000, 0x1000).link_value_changed);
}

void sequence_single_word_when_reachable() {
    const auto r = relocate_instruction_sequence(0x54000040u, 0x1000, 0x2000);  // b.eq .+8
    CHECK(r.ok());
    CHECK_EQ(r.count, 1);
    CHECK_EQ(decode(r.words[0]).displacement, 0x1008 - 0x2000);
}

void sequence_branch_expansions() {
    armigo::aarch64::test::Lcg random{0xB5A7};
    const std::uint64_t from = 0x40000000;
    const std::uint64_t target = from + 0x80;
    const std::uint64_t to = from + 0x4000000;  // beyond 1 MiB and 32 KiB, within 128 MiB
    const Template* branches[] = {&templates[2], &templates[3], &templates[4],
                                  &templates[5], &templates[6], &templates[7]};
    for (const auto* t : branches) {
        const std::uint32_t word = at(*t, from, target);
        CHECK(relocate_instruction(word, from, to).error == RelocationError::out_of_range);
        const auto sequence = relocate_instruction_sequence(word, from, to);
        if (!CHECK(sequence.ok())) {
            continue;
        }
        CHECK_EQ(sequence.count, 2);
        const auto first = decode(sequence.words[0]);
        CHECK_EQ(first.displacement, 8);
        // The skip keeps every operand except the inverted sense.
        CHECK_EQ((sequence.words[0] ^ word) & ~displacement_field_mask(t->opcode),
                 t->opcode == Opcode::b_cond || t->opcode == Opcode::bc_cond ? 1u : 0x01000000u);
        CHECK(decode(sequence.words[1]).opcode == Opcode::b);
        for (int trial = 0; trial < 64; ++trial) {
            Machine m = seeded_machine(random);
            if (trial % 2 == 0) {
                m.x[decode(word).reg] = 0;  // exercise the zero/clear side
            }
            equivalent(word, from, sequence, to, m);
        }
    }

    // AL and NV always branch, so no inverted skip exists or is needed.
    for (const std::uint32_t cond : {14u, 15u}) {
        const std::uint32_t word = at(Template{Opcode::b_cond, 0x54000000u | cond}, from, target);
        const auto sequence = relocate_instruction_sequence(word, from, to);
        CHECK(sequence.ok() && sequence.count == 1 && decode(sequence.words[0]).opcode == Opcode::b);
        equivalent(word, from, sequence, to, Machine{});
    }

    // Beyond the reach of the unconditional branch the expansion fails too.
    const std::uint32_t word = at(templates[2], from, target);
    CHECK(relocate_instruction_sequence(word, from, target + (1ull << 27) + 4).error ==
          RelocationError::out_of_range);
    CHECK(relocate_instruction_sequence(word, from, target - (1ull << 27) + 4).ok());

    // B and BL have no register-free expansion.
    const std::uint32_t b = at(templates[0], from, target);
    CHECK(relocate_instruction_sequence(b, from, from + (1ull << 28)).error ==
          RelocationError::out_of_range);
    const std::uint32_t bl = at(templates[1], from, target);
    CHECK(relocate_instruction_sequence(bl, from, from + (1ull << 28)).error ==
          RelocationError::out_of_range);

    // Expansion words past the end of the address space are refused.
    CHECK(relocate_instruction_sequence(0x54000040u, 0xFFFFFFFFFF000000ull, 0xFFFFFFFFFFFFFFFCull)
              .error == RelocationError::address_overflow);
}

void sequence_address_and_load_expansions() {
    armigo::aarch64::test::Lcg random{0xADD5};
    const std::uint64_t from = 0x7000000000ull;
    // Targets lie within literal reach of from; their low 12 bits select the
    // ADRP-only, scaled-offset and explicit-add forms.
    for (const std::uint64_t target : {from + 0x12347ull, from - 0x99000ull, from + 0xF0004ull,
                                       from + 0x8008ull, from - 0x7FFF0ull}) {
        const std::uint64_t to = from + 0x10000000ull;
        // ADR (x9): ADRP + ADD, or ADRP alone when the low bits are zero.
        const std::uint32_t adr = at(templates[8], from, target);
        auto sequence = relocate_instruction_sequence(adr, from, to);
        CHECK(sequence.ok());
        CHECK_EQ(sequence.count, (target & 0xFFF) ? 2u : 1u);
        equivalent(adr, from, sequence, to, seeded_machine(random));

        for (std::size_t index = 10; index < std::size(templates); ++index) {
            const auto& t = templates[index];
            if ((target & 3) != 0) {
                continue;
            }
            const std::uint32_t word = at(t, from, target);
            const bool general = index <= 12;
            sequence = relocate_instruction_sequence(word, from, to);
            if (!general) {
                CHECK(sequence.error == RelocationError::scratch_register_required);
                CHECK(sequence.count == 0);
                CHECK(relocate_instruction_sequence(word, from, to, {.scratch_register = 18}).error ==
                      RelocationError::invalid_scratch_register);
                CHECK(relocate_instruction_sequence(word, from, to, {.scratch_register = 31}).error ==
                      RelocationError::invalid_scratch_register);
                sequence = relocate_instruction_sequence(word, from, to, {.scratch_register = 12});
            }
            if (!CHECK(sequence.ok())) {
                continue;
            }
            const std::uint32_t width = index == 16 ? 16 : (index == 11 || index == 13 || index == 15) ? 8 : 4;
            CHECK_EQ(sequence.count, (target & 0xFFF) % width == 0 ? 2u : 3u);
            equivalent(word, from, sequence, to, seeded_machine(random),
                       general ? 0 : (1u << 12));
        }
    }

    // Register 31 forms are refused rather than rewritten through SP.
    const std::uint64_t far = from + 0x20000000ull;
    CHECK(relocate_instruction_sequence(at(Template{Opcode::adr, 0x1000001Fu}, from, from + 8), from,
                                        far).error == RelocationError::unsupported_operand);
    CHECK(relocate_instruction_sequence(at(Template{Opcode::ldr_literal_x, 0x5800001Fu}, from,
                                           from + 8), from, far).error ==
          RelocationError::unsupported_operand);

    // ADRP and expanded ADR cannot reach beyond +/-4 GiB.
    const std::uint32_t adrp = at(templates[9], from, from + 0x1000);
    CHECK(relocate_instruction_sequence(adrp, from, from + 0x200000000ull).error ==
          RelocationError::out_of_range);
    const std::uint32_t adr = at(templates[8], from, from + 0x10);
    CHECK(relocate_instruction_sequence(adr, from, from + 0x200000000ull).error ==
          RelocationError::out_of_range);
}

void sequence_preserved_link_calls() {
    armigo::aarch64::test::Lcg random{0x11CC};
    const std::uint64_t from = 0x9000001234ull & ~3ull;
    const std::uint64_t target = from + 0x5000;
    const std::uint32_t bl = at(templates[1], from, target);
    const std::uint64_t to = from + 0x1000000;
    const SequenceOptions preserve{.preserve_link_address = true};

    const auto plain = relocate_instruction_sequence(bl, from, to);
    CHECK(plain.ok() && plain.count == 1 && plain.link_value_changed);
    equivalent(bl, from, plain, to, seeded_machine(random), 0, true);

    const auto kept = relocate_instruction_sequence(bl, from, to, preserve);
    CHECK(kept.ok() && !kept.link_value_changed);
    CHECK_EQ(kept.count, ((from + 4) & 0xFFF) ? 3u : 2u);
    const auto moved = run_sequence(seeded_machine(random), kept.view(), to);
    CHECK(moved.valid);
    CHECK_EQ(moved.exit, target);
    CHECK_EQ(moved.machine.x[30], from + 4);

    // Even when unreachable by one BL, the preserved form reaches.
    const std::uint64_t far = target + 0x7000000ull + 0x8000000ull;
    CHECK(relocate_instruction_sequence(bl, from, far).error == RelocationError::out_of_range);
    CHECK(relocate_instruction_sequence(bl, from, target + 0x7000000ull, preserve).ok());

    // BLR xN keeps its register and gets the original return address.
    const auto blr = relocate_instruction_sequence(0xD63F0120u, from, to, preserve);
    CHECK(blr.ok() && !blr.link_value_changed);
    Machine m = seeded_machine(random);
    const auto blr_moved = run_sequence(m, blr.view(), to);
    CHECK_EQ(blr_moved.exit, m.x[9]);
    CHECK_EQ(blr_moved.machine.x[30], from + 4);
    CHECK(relocate_instruction_sequence(0xD63F03C0u, from, to, preserve).error ==
          RelocationError::unsupported_operand);  // blr x30

    // In place, no expansion is needed.
    const auto same = relocate_instruction_sequence(bl, from, from, preserve);
    CHECK(same.ok() && same.count == 1 && same.words[0] == bl);
}

void sequence_storage_overflow() {
    constexpr std::uint64_t last_word = 0xFFFFFFFFFFFFFFFCull;
    constexpr std::uint64_t from = last_word - 0x200000;
    const std::array words = {
        encode_adr(0, from, from + 1).word,
        encode_load_literal(Opcode::ldr_literal_x, 0, from, from + 4).word,
        encode_load_literal(Opcode::ldr_literal_q, 1, from, from + 8).word,
        encode_blr(9).word,
    };
    const SequenceOptions options{.scratch_register = 12, .preserve_link_address = true};
    for (const auto word : words) {
        const auto result = relocate_instruction_sequence(word, from, last_word, options);
        CHECK(result.error == RelocationError::address_overflow);
        CHECK(result.count == 0);
        for (const auto emitted : result.words) CHECK(emitted == 0);
    }
    // A three-word unaligned SIMD load expansion must check its last append.
    const auto three = relocate_instruction_sequence(words[2], from, last_word - 4, options);
    CHECK(three.error == RelocationError::address_overflow && three.count == 0);
    // Two words fit when their final instruction starts at the last word.
    const auto fits = relocate_instruction_sequence(words[0], from, last_word - 4, options);
    CHECK(fits.ok() && fits.count == 2);
}

} // namespace

int main() {
    single_word_preserves_reference_and_fields();
    single_word_reach_endpoints();
    single_word_rejections();
    sequence_single_word_when_reachable();
    sequence_branch_expansions();
    sequence_address_and_load_expansions();
    sequence_preserved_link_calls();
    sequence_storage_overflow();
    return armigo::aarch64::test::finish("aarch64.relocation");
}
