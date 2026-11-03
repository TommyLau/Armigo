// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic instruction words only; no game bytes, captures or dumps.

#include "pipeline/decode/resource_fields.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;
using resource_decode::rule;

int failures = 0;
void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Builds a synthetic guest word: a 14-bit dispatch pattern ('1', '0', 'x') with
// fields OR-ed in at explicit bit positions.
struct Word {
    std::uint64_t bits{};
    Word(std::string_view pattern) {
        for (unsigned index = 0; index < pattern.size(); ++index) {
            if (pattern[index] == '1') {
                bits |= std::uint64_t{1} << (63 - index);
            }
        }
    }
    Word& set(unsigned first, unsigned width, std::uint64_t value) {
        const std::uint64_t mask = width >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << width) - 1;
        bits |= (value & mask) << first;
        return *this;
    }
    Word& bit(unsigned first, bool value = true) { return set(first, 1, value ? 1 : 0); }
    [[nodiscard]] operator std::uint64_t() const { return bits; }
};

// Lane-3 style dispatch over the published rules only.
std::optional<DecodedInstruction> decode_word(std::uint64_t bits, Pc pc = 8) {
    const RawInstruction raw{bits};
    for (const DecodeRule& candidate : resource_decode_rules()) {
        if ((bits & candidate.mask) == candidate.value) {
            Result<DecodedInstruction> result = candidate.decode(pc, raw);
            if (const auto* error = std::get_if<Diagnostic>(&result)) {
                std::cerr << "decode error at pc " << pc << ": " << error->message << '\n';
                return std::nullopt;
            }
            return std::get<DecodedInstruction>(std::move(result));
        }
    }
    return std::nullopt;
}

bool rejects(std::uint64_t bits, std::string_view detail) {
    const RawInstruction raw{bits};
    for (const DecodeRule& candidate : resource_decode_rules()) {
        if ((bits & candidate.mask) == candidate.value) {
            Result<DecodedInstruction> result = candidate.decode(8, raw);
            if (const auto* error = std::get_if<Diagnostic>(&result)) {
                check(error->phase == Phase::decode && error->code == DiagnosticCode::malformed_input &&
                          error->pc == 8,
                      std::string(detail) + " (diagnostic context)");
                return true;
            }
        }
    }
    check(false, std::string(detail) + " (expected rejection)");
    return false;
}

std::optional<std::uint64_t> field(const DecodedInstruction& instruction, DecodedField key) {
    Result<std::uint64_t> result = decoded_field(instruction, key);
    if (const auto* value = std::get_if<std::uint64_t>(&result)) {
        return *value;
    }
    return std::nullopt;
}

bool has_role(const DecodedInstruction& instruction, RegisterRole role, unsigned lanes, Access access) {
    unsigned count = 0;
    for (const RegisterAccess& entry : instruction.registers) {
        if (entry.role == role) {
            ++count;
            if (entry.access != access) return false;
        }
    }
    return count == lanes;
}

unsigned role_lanes(const DecodedInstruction& instruction, RegisterRole role) {
    unsigned count = 0;
    for (const RegisterAccess& entry : instruction.registers) {
        if (entry.role == role) ++count;
    }
    return count;
}

bool has_source(const DecodedInstruction& instruction, SourceRole role) {
    for (const DecodedSource& source : instruction.sources) {
        if (source.role == role) return true;
    }
    return false;
}

// --- Rule table integrity ----------------------------------------------------

void test_rule_table() {
    const auto rules = resource_decode_rules();
    check(rules.size() >= 60, "Resource decode rules cover the lane families");
    for (std::size_t outer = 0; outer < rules.size(); ++outer) {
        for (std::size_t inner = outer + 1; inner < rules.size(); ++inner) {
            const std::uint64_t shared = rules[outer].mask & rules[inner].mask;
            const bool disjoint = shared == 0 || (shared & (rules[outer].value ^ rules[inner].value)) != 0;
            check(disjoint, "Published decode rules are pairwise disjoint");
        }
        check(rules[outer].decode != nullptr, "Decode rules carry callbacks");
    }
    check(!decode_word(0).has_value(), "All-zero word matches no resource rule");
    check(!decode_word(~std::uint64_t{0}).has_value(), "All-ones word matches no resource rule");
}

// --- Flow control ------------------------------------------------------------

void test_flow() {
    // BRA with displacement -8 words relative to the scheduling base: the
    // normalized displacement already folds in the +8 scheduling base.
    const DecodedInstruction bra = *decode_word(Word("111000100100xx")
                                                    .set(20, 24, 0xfffff8)
                                                    .set(16, 3, 3)
                                                    .bit(19)
                                                    .set(0, 5, 1));
    check(bra.name == InstructionName::bra, "BRA decodes by name");
    check(field(bra, DecodedField::branch_displacement) == std::optional<std::uint64_t>{0}, "BRA displacement includes the scheduling base");
    check(field(bra, DecodedField::condition) == std::optional<std::uint64_t>{1}, "BRA preserves the condition selector");
    check(bra.predicate.index == 3 && bra.predicate.inverted && bra.predicate.enabled, "BRA predicate bits are exact");
    // Forward branch: pc 24 + displacement 56 = 80, crossing a scheduling word.
    const DecodedInstruction forward = *decode_word(Word("111000100100xx").set(20, 24, 48), 24);
    check(field(forward, DecodedField::branch_displacement) == std::optional<std::uint64_t>{56},
        "Forward BRA displacement is scheduling-relative");

    const DecodedInstruction brx = *decode_word(Word("111000100101xx").set(8, 8, 5).set(20, 24, 16));
    check(brx.name == InstructionName::brx, "BRX decodes by name");
    check(field(brx, DecodedField::target_register) == std::optional<std::uint64_t>{5}, "BRX records the target register");
    check(field(brx, DecodedField::branch_displacement) == std::optional<std::uint64_t>{24}, "BRX keeps its displacement base");
    check(has_source(brx, SourceRole::a) && role_lanes(brx, RegisterRole::source_a) == 1, "BRX reads its index register");

    const DecodedInstruction jmp = *decode_word(Word("111000100001xx").set(20, 32, 0x1234'5678));
    check(jmp.name == InstructionName::jmp, "JMP decodes by name");
    check(field(jmp, DecodedField::branch_absolute) == std::optional<std::uint64_t>{1}, "JMP is flagged absolute");
    check(field(jmp, DecodedField::branch_displacement) == std::optional<std::uint64_t>{0x1234'5678}, "JMP keeps the raw 32-bit immediate");
    const DecodedInstruction jmx = *decode_word(Word("111000100000xx").set(8, 8, 7));
    check(jmx.name == InstructionName::jmx, "JMX decodes by name");
    check(field(jmx, DecodedField::target_register) == std::optional<std::uint64_t>{7}, "JMX records the target register");

    const DecodedInstruction cal = *decode_word(Word("111000100110xx").set(20, 24, 0x40));
    check(cal.name == InstructionName::cal, "CAL decodes by name");
    check(!cal.predicate.enabled, "CAL is a NoPred encoding");
    check(field(cal, DecodedField::branch_displacement) == std::optional<std::uint64_t>{0x48}, "CAL target uses the scheduling base");
    for (const auto& [pattern, name] : {std::pair{"111000101001xx", InstructionName::ssy},
             {"111000101010xx", InstructionName::pbk}, {"111000101011xx", InstructionName::pcnt}}) {
        const DecodedInstruction push = *decode_word(Word(pattern).set(20, 24, 32));
        check(push.name == name, "Push instruction decodes by name");
        check(!push.predicate.enabled, "Push instructions are NoPred encodings");
        check(field(push, DecodedField::branch_displacement) == std::optional<std::uint64_t>{40},
            "Push target displacement includes the scheduling base");
    }
    for (const auto& [pattern, name] : {std::pair{"111000110100xx", InstructionName::brk},
             {"111000110101xx", InstructionName::cont}, {"111000110000xx", InstructionName::exit},
             {"111000110011xx", InstructionName::kil}, {"111000110010xx", InstructionName::ret},
             {"1111000011111x", InstructionName::sync}}) {
        const DecodedInstruction flow = *decode_word(Word(pattern).set(0, 5, 15));
        check(flow.name == name, "Pop/terminator decodes by name");
        check(flow.predicate.enabled, "Pop and terminator instructions stay predicated");
        check(field(flow, DecodedField::condition) == std::optional<std::uint64_t>{15},
            "Terminator preserves the condition selector");
    }
    const DecodedInstruction nop = *decode_word(Word("0101000010110x"));
    check(nop.name == InstructionName::nop, "NOP decodes by name");
    check(nop.registers.empty() && nop.fields.empty(), "NOP carries no operands");
    check(std::holds_alternative<Diagnostic>(decoded_field(nop, DecodedField::condition)),
        "NOP has no condition field to consume");

    const DecodedInstruction votevtg = *decode_word(Word("0101000011100x").set(48, 2, 1).set(20, 28, 0x123));
    check(votevtg.name == InstructionName::votevtg, "VOTE.VTG decodes by name");
    check(votevtg.registers.empty(), "VOTE.VTG reads and writes no register");

    const DecodedInstruction depbar = *decode_word(Word("1111000011110x").set(26, 3, 5));
    check(depbar.name == InstructionName::depbar, "DEPBAR decodes by name");
    check(field(depbar, DecodedField::barrier_id) == std::optional<std::uint64_t>{5}, "DEPBAR records the scoreboard id");

    const DecodedInstruction bar = *decode_word(Word("1111000010101x").set(32, 3, 2).set(20, 12, 0x123).set(8, 8, 9));
    check(bar.name == InstructionName::bar, "BAR decodes by name");
    check(field(bar, DecodedField::barrier_kind) == std::optional<std::uint64_t>{2}, "BAR preserves the barrier operation");
    check(field(bar, DecodedField::barrier_id) == std::optional<std::uint64_t>{0x123}, "BAR preserves the barrier id");
    check(has_source(bar, SourceRole::a) && has_source(bar, SourceRole::predicate), "BAR reads its register and predicate");

    const DecodedInstruction membar = *decode_word(Word("1110111110011x").set(8, 2, 2));
    check(membar.name == InstructionName::membar, "MEMBAR decodes by name");
    check(field(membar, DecodedField::memory_scope) == std::optional<std::uint64_t>{2}, "MEMBAR preserves the scope selector");
}

// --- Attribute load/store and interpolation ----------------------------------

void test_attributes() {
    const DecodedInstruction ald = *decode_word(
        Word("1110111111011x").set(47, 2, 3).set(20, 11, 0x180).set(0, 8, 10).set(8, 8, 2));
    check(ald.name == InstructionName::ald, "ALD decodes by name");
    check(field(ald, DecodedField::io_byte_offset) == std::optional<std::uint64_t>{0x180},
        "ALD attribute offset is already a byte offset");
    check(field(ald, DecodedField::memory_elements) == std::optional<std::uint64_t>{4}, "ALD 128-bit access covers four words");
    check(has_role(ald, RegisterRole::destination, 4, Access::write), "ALD writes four destination lanes");
    check(ald.registers[0].reg.index == 10 && ald.registers[3].reg.index == 13, "ALD destination lanes are consecutive");
    check(has_source(ald, SourceRole::a), "ALD reads its base register");

    const DecodedInstruction ast =
        *decode_word(Word("1110111111110x").set(47, 2, 1).set(20, 11, 0x40).set(0, 8, 30).set(8, 8, 4).set(39, 8, 6));
    check(ast.name == InstructionName::ast, "AST decodes by name");
    check(field(ast, DecodedField::io_byte_offset) == std::optional<std::uint64_t>{0x40}, "AST attribute offset is a byte offset");
    check(has_role(ast, RegisterRole::source_b, 2, Access::read), "AST reads a 64-bit data pair");
    check(has_role(ast, RegisterRole::source_c, 1, Access::read) && has_source(ast, SourceRole::c),
        "AST reads its index register");

    const DecodedInstruction ipa =
        *decode_word(Word("11100000xxxxxx").set(28, 10, 0x88).set(54, 2, 1).set(52, 2, 2).set(47, 3, 4).bit(50));
    check(ipa.name == InstructionName::ipa, "IPA decodes by name");
    check(field(ipa, DecodedField::io_byte_offset) == std::optional<std::uint64_t>{0x88}, "IPA offset is a byte offset");
    check(field(ipa, DecodedField::interpolation_mode) == std::optional<std::uint64_t>{1}, "IPA preserves its operation selector");
    check(field(ipa, DecodedField::interpolation_location) == std::optional<std::uint64_t>{2}, "IPA preserves its sample location");
    check(ipa.sources.size() == 4 && ipa.sources[3].modifiers.negate, "IPA records the inverted predicate source");
}

// --- Memory ------------------------------------------------------------------

void test_memory() {
    const DecodedInstruction ldc = *decode_word(
        Word("1110111110010x").set(48, 3, 2).set(36, 5, 3).set(20, 16, 0xfff8).set(0, 8, 12).set(8, 8, 5));
    check(ldc.name == InstructionName::ldc, "LDC decodes by name");
    check(ldc.encoding == Encoding::constant_buffer, "LDC uses the constant-buffer encoding");
    check(field(ldc, DecodedField::memory_space) ==
              std::optional<std::uint64_t>{static_cast<std::uint64_t>(AddressSpace::constant_buffer)},
        "LDC targets the constant-buffer space");
    check(field(ldc, DecodedField::memory_byte_offset) == std::optional<std::uint64_t>{static_cast<std::uint64_t>(-8)},
        "LDC offsets are signed bytes");
    check(field(ldc, DecodedField::memory_element_bytes) == std::optional<std::uint64_t>{2}, "LDC 16-bit load covers two bytes");
    check(resource_decode::ldc_cbuf_slot(ldc.raw) == 3, "LDC slot accessor preserves the encoded slot");

    const DecodedInstruction ldl = *decode_word(Word("1110111101000x").set(48, 3, 5).set(20, 24, 0x1000).set(0, 8, 20));
    check(field(ldl, DecodedField::memory_space) ==
              std::optional<std::uint64_t>{static_cast<std::uint64_t>(AddressSpace::local)},
        "LDL targets local memory");
    check(field(ldl, DecodedField::memory_byte_offset) == std::optional<std::uint64_t>{0x1000}, "LDL offsets are signed bytes");
    check(field(ldl, DecodedField::memory_elements) == std::optional<std::uint64_t>{2}, "LDL 64-bit load covers two words");
    check(has_role(ldl, RegisterRole::destination, 2, Access::write), "LDL writes two destination lanes");

    const DecodedInstruction lds = *decode_word(Word("1110111101001x").set(48, 3, 0).set(20, 24, 8));
    check(field(lds, DecodedField::memory_space) ==
              std::optional<std::uint64_t>{static_cast<std::uint64_t>(AddressSpace::shared)},
        "LDS targets shared memory");

    const DecodedInstruction stl =
        *decode_word(Word("1110111101010x").set(48, 3, 4).set(20, 24, 0xfffffe0ULL).set(8, 8, 3).set(0, 8, 40));
    check(field(stl, DecodedField::memory_byte_offset) == std::optional<std::uint64_t>{static_cast<std::uint64_t>(-32)},
        "STL offsets sign-extend as bytes");
    check(has_role(stl, RegisterRole::source_b, 1, Access::read), "STL reads its data register");

    const DecodedInstruction ldg = *decode_word(Word("1110111011010x").bit(45).set(48, 3, 4).set(8, 8, 2));
    check(ldg.name == InstructionName::ldg, "LDG decodes by name");
    check(field(ldg, DecodedField::memory_space) ==
              std::optional<std::uint64_t>{static_cast<std::uint64_t>(AddressSpace::storage)},
        "LDG targets storage memory");
    check(has_role(ldg, RegisterRole::source_a, 2, Access::read), "LDG extended address reads the address pair");
    const DecodedInstruction stg = *decode_word(Word("1110111011011x").set(48, 3, 6).set(8, 8, 2).set(0, 8, 9));
    check(has_role(stg, RegisterRole::source_a, 1, Access::read) &&
              has_role(stg, RegisterRole::source_b, 4, Access::read),
        "STG reads the address and a 128-bit data vector");

    const DecodedInstruction atom = *decode_word(
        Word("11101101xxxxxx").set(49, 3, 2).set(52, 4, 1).set(28, 20, 0xffff0).set(8, 8, 4).set(20, 8, 6).set(0, 8, 8));
    check(atom.name == InstructionName::atom, "ATOM decodes by name");
    check(field(atom, DecodedField::atomic_operation) == std::optional<std::uint64_t>{1}, "ATOM preserves its operation");
    check(field(atom, DecodedField::memory_byte_offset) == std::optional<std::uint64_t>{static_cast<std::uint64_t>(-16)},
        "ATOM offsets are signed bytes");
    check(field(atom, DecodedField::memory_elements) == std::optional<std::uint64_t>{2}, "ATOM 64-bit atomic covers two words");
    check(has_role(atom, RegisterRole::destination, 1, Access::write), "ATOM writes its result");

    const DecodedInstruction cas = *decode_word(Word("111011101111xx").set(8, 8, 4).set(20, 8, 6).set(0, 8, 8));
    check(cas.name == InstructionName::atom_cas, "ATOMCAS decodes by name");
    check(has_role(cas, RegisterRole::destination, 1, Access::read_write), "ATOMCAS destination is read-modify-write");

    const DecodedInstruction atoms = *decode_word(Word("11101100xxxxxx").set(30, 22, 1).set(28, 2, 0).set(52, 4, 5));
    check(atoms.name == InstructionName::atoms, "ATOMS decodes by name");
    check(field(atoms, DecodedField::memory_space) ==
              std::optional<std::uint64_t>{static_cast<std::uint64_t>(AddressSpace::shared)},
        "ATOMS targets shared memory");
    check(field(atoms, DecodedField::memory_byte_offset) == std::optional<std::uint64_t>{4},
        "Shared atomic word immediates normalize to bytes");
    const DecodedInstruction atomscas = *decode_word(Word("111011100xxxxx"));
    check(atomscas.name == InstructionName::atoms_cas, "ATOMSCAS decodes by name");
    check(has_role(atomscas, RegisterRole::destination, 1, Access::read_write), "ATOMSCAS destination is read-modify-write");

    const DecodedInstruction red = *decode_word(Word("1110101111111x").set(20, 3, 0).set(23, 3, 7).set(28, 20, 4));
    check(red.name == InstructionName::red, "RED decodes by name");
    check(role_lanes(red, RegisterRole::destination) == 0, "RED writes no destination");
    check(field(red, DecodedField::atomic_operation) == std::optional<std::uint64_t>{7}, "RED preserves its operation");
    check(field(red, DecodedField::memory_byte_offset) == std::optional<std::uint64_t>{4}, "RED offsets are signed bytes");
}

// --- Surface (image) operations ----------------------------------------------

void test_surfaces() {
    const DecodedInstruction suld =
        *decode_word(Word("1110101100001x").set(20, 3, 6).set(33, 3, 3).set(36, 13, 0x1234).set(8, 8, 2).set(0, 8, 10));
    check(suld.name == InstructionName::suld, "SULD decodes by name");
    check(resource_decode::texture_mode_of(field(suld, DecodedField::texture_mode).value()) == TextureMode::fetch,
        "Surface loads use the fetch mode");
    check(field(suld, DecodedField::texture_handle_byte_offset) == std::optional<std::uint64_t>{0x1234 * 4},
        "Surface handles normalize word offsets to bytes");
    check(field(suld, DecodedField::memory_elements) == std::optional<std::uint64_t>{4}, "SULD 128-bit load covers four words");
    const std::uint64_t dimension = field(suld, DecodedField::texture_dimension).value();
    check(resource_decode::texture_dimension_of(dimension) == TextureDimension::d2, "Surface dimension decodes to 2D");

    const DecodedInstruction suld_b = *decode_word(
        Word("1110101100000x").set(20, 4, 0b0011).set(33, 3, 3).set(39, 8, 7).set(8, 8, 2).set(0, 8, 10));
    check(suld_b.name == InstructionName::suld_b, "SULDB decodes by name");
    check(suld_b.encoding == Encoding::texture_bindless, "SULDB is bindless");
    check(has_role(suld_b, RegisterRole::source_c, 1, Access::read), "SULDB reads its handle register");
    check(field(suld_b, DecodedField::texture_component_mask) == std::optional<std::uint64_t>{0b0011},
        "SULDB preserves its component mask");
    check(has_role(suld_b, RegisterRole::destination, 2, Access::write), "SULDB writes one lane per component");

    const DecodedInstruction suld_d =
        *decode_word(Word("1110101100011x").set(20, 3, 4).set(33, 3, 3).set(30, 3, 5).set(36, 13, 8));
    check(suld_d.name == InstructionName::suld_d, "SULDD decodes by name");
    check(role_lanes(suld_d, RegisterRole::predicate_destination) == 1, "SULDD carries its predicate destination");

    const DecodedInstruction sust =
        *decode_word(Word("1110101100101x").set(20, 4, 0b1000).set(33, 3, 3).set(36, 13, 0).set(8, 8, 2).set(0, 8, 11));
    check(sust.name == InstructionName::sust, "SUST decodes by name");
    check(has_role(sust, RegisterRole::source_b, 1, Access::read), "SUST reads one data word per component");
    check(role_lanes(sust, RegisterRole::destination) == 0, "SUST writes no destination");

    const DecodedInstruction suatom = *decode_word(
        Word("11101010x0xxxx").set(33, 3, 3).set(51, 3, 0).set(29, 4, 0).set(36, 13, 0x40).set(8, 8, 2).set(20, 8, 6));
    check(suatom.name == InstructionName::suatom, "SUATOM decodes by name");
    check(field(suatom, DecodedField::atomic_operation) == std::optional<std::uint64_t>{0}, "SUATOM preserves its operation");

    const DecodedInstruction suatom_cas =
        *decode_word(Word("1110101111xxxx").set(33, 3, 3).set(51, 3, 0).set(30, 3, 2).set(8, 8, 2).set(20, 8, 6));
    check(suatom_cas.name == InstructionName::suatom_cas, "SUATOMCAS decodes through its disjoint rules");
    check(has_role(suatom_cas, RegisterRole::destination, 1, Access::read_write), "SUATOMCAS destination is read-modify-write");
    check(role_lanes(suatom_cas, RegisterRole::predicate_destination) == 1, "SUATOMCAS carries its predicate destination");
    const DecodedInstruction suatom_cas_b =
        *decode_word(Word("1110101011010x").set(33, 3, 3).set(36, 3, 0).set(39, 8, 7).set(30, 3, 2));
    check(suatom_cas_b.name == InstructionName::suatom_cas_b, "SUATOMCASB decodes by name");
    check(suatom_cas_b.encoding == Encoding::texture_bindless, "SUATOMCASB is bindless");
    const DecodedInstruction suatom_b2 = *decode_word(Word("1110101110xxxx").set(33, 3, 3));
    check(suatom_b2.name == InstructionName::suatom_b2, "SUATOMB2 keeps its own identity");

    const DecodedInstruction sured =
        *decode_word(Word("1110101101011x").set(20, 3, 0).set(24, 3, 2).set(33, 3, 3).set(36, 13, 4));
    check(sured.name == InstructionName::sured, "SURED decodes by name");
    check(field(sured, DecodedField::atomic_operation) == std::optional<std::uint64_t>{2}, "SURED preserves its operation");
    const DecodedInstruction sured_b = *decode_word(Word("1110101101010x").set(39, 8, 7));
    check(sured_b.name == InstructionName::sured_b, "SUREDB decodes by name");
}

// --- Sampled textures --------------------------------------------------------

void test_textures() {
    const DecodedInstruction tex = *decode_word(
        Word("11000xxxxxxxxx").set(28, 3, 2).set(55, 3, 3).set(31, 4, 0b0011).set(36, 13, 0x100).set(8, 8, 2).set(20, 8, 5));
    check(tex.name == InstructionName::tex, "TEX decodes by name");
    check(field(tex, DecodedField::texture_handle_byte_offset) == std::optional<std::uint64_t>{0x400},
        "Texture handles normalize word offsets to bytes");
    check(field(tex, DecodedField::texture_dimension) == std::optional<std::uint64_t>{2}, "TEX keeps the raw TexDim selector");
    check(field(tex, DecodedField::texture_mode) == std::optional<std::uint64_t>{3}, "TEX keeps the raw LOD selector");
    check(field(tex, DecodedField::texture_compare) == std::optional<std::uint64_t>{0} &&
              field(tex, DecodedField::texture_ndv) == std::optional<std::uint64_t>{0} &&
              field(tex, DecodedField::texture_lod_clamp) == std::optional<std::uint64_t>{0},
        "TEX publishes its boolean modifiers");
    check(field(tex, DecodedField::texture_offset) == std::optional<std::uint64_t>{0}, "TEX always carries its offset flag");
    check(role_lanes(tex, RegisterRole::source_a) == 2, "TEX reads a 2D coordinate pair");
    check(role_lanes(tex, RegisterRole::source_b) == 1, "TEX reads its explicit level");
    check(role_lanes(tex, RegisterRole::destination) == 2, "TEX writes one lane per mask bit");
    const DecodedInstruction implicit = *decode_word(
        Word("11000xxxxxxxxx").set(28, 3, 2).set(55, 3, 0).set(31, 4, 0b0001).set(8, 8, 2).set(20, 8, 5));
    check(field(implicit, DecodedField::texture_mode) == std::optional<std::uint64_t>{0} &&
              role_lanes(implicit, RegisterRole::source_b) == 0,
        "TEX without a LOD selector reads no level operand");
    check(role_lanes(tex, RegisterRole::predicate_destination) == 1, "TEX carries its predicate destination");

    const DecodedInstruction tex_shadow = *decode_word(
        Word("11000xxxxxxxxx").set(28, 3, 2).set(55, 3, 1).set(31, 4, 0b0001).bit(50).bit(35).bit(58).bit(54));
    check(field(tex_shadow, DecodedField::texture_compare) == std::optional<std::uint64_t>{1} &&
              field(tex_shadow, DecodedField::texture_ndv) == std::optional<std::uint64_t>{1} &&
              field(tex_shadow, DecodedField::texture_lod_clamp) == std::optional<std::uint64_t>{1} &&
              field(tex_shadow, DecodedField::texture_offset) == std::optional<std::uint64_t>{1},
        "TEX depth-compare, Ndv, clamp and offset bits stay explicit");
    check(role_lanes(tex_shadow, RegisterRole::source_b) == 2, "TEX level-zero mode reads only offset and compare");

    const DecodedInstruction texb = *decode_word(
        Word("1101111010xxxx").set(28, 3, 2).set(37, 3, 1).bit(36).set(31, 4, 0b0001).set(20, 8, 5));
    check(texb.name == InstructionName::tex_b, "TEXB decodes by name");
    check(texb.encoding == Encoding::texture_bindless, "TEXB is bindless");
    check(field(texb, DecodedField::texture_handle_byte_offset) == std::nullopt, "Bindless textures carry no handle offset");
    check(field(texb, DecodedField::texture_mode) == std::optional<std::uint64_t>{1}, "TEXB keeps the raw level-zero LOD selector");
    check(role_lanes(texb, RegisterRole::source_b) == 2, "TEXB reads its handle and packed offset register");

    // TEXS table: 2D target with a single destination keeps the raw selector.
    const DecodedInstruction texs = *decode_word(
        Word("1101100xxxxxxx").set(53, 4, 1).set(50, 3, 0).set(28, 8, 255).set(36, 13, 0x80).set(8, 8, 2).set(20, 8, 5));
    check(texs.name == InstructionName::texs, "TEXS decodes by name");
    check(field(texs, DecodedField::texture_target) == std::optional<std::uint64_t>{1}, "TEXS keeps the raw target selector");
    check(field(texs, DecodedField::texture_component_mask) == std::optional<std::uint64_t>{0},
        "TEXS keeps its encoded 3-bit mask selector");
    check(has_source(texs, SourceRole::b), "TEXS always publishes its Rb operand slot");
    check(role_lanes(texs, RegisterRole::second_destination) == 0, "TEXS with Dest2=RZ writes no second destination");
    // TEXS dual destinations use the alternate mask row at emission.
    const DecodedInstruction texs_dual = *decode_word(
        Word("1101100xxxxxxx").set(53, 4, 1).set(50, 3, 0).set(28, 8, 40).set(36, 13, 0x80));
    check(role_lanes(texs_dual, RegisterRole::destination) == 2 &&
              role_lanes(texs_dual, RegisterRole::second_destination) == 1,
        "TEXS splits results between Dest and Dest2");
    // Array targets keep the layer on Ra and remaining coordinates on Rb.
    const DecodedInstruction texs_array = *decode_word(Word("1101100xxxxxxx").set(53, 4, 9).set(50, 3, 0).set(28, 8, 255));
    check(field(texs_array, DecodedField::texture_target) == std::optional<std::uint64_t>{9},
        "TEXS array target keeps its selector");
    check(role_lanes(texs_array, RegisterRole::source_a) == 2 && role_lanes(texs_array, RegisterRole::source_b) == 2,
        "TEXS array targets read the layer and shadow registers from the short groups");

    const DecodedInstruction texs_f16 = *decode_word(Word("1101000xxxxxxx").set(53, 4, 1).set(50, 3, 0).set(28, 8, 255));
    check(texs_f16.name == InstructionName::texs_f16, "TEXSF16 keeps its own name");

    const DecodedInstruction tlds = *decode_word(
        Word("1101101xxxxxxx").set(53, 4, 0x5).set(50, 3, 0).set(28, 8, 255).set(36, 13, 0x10));
    check(tlds.name == InstructionName::tlds, "TLDS decodes by name");
    check(field(tlds, DecodedField::texture_target) == std::optional<std::uint64_t>{5}, "TLDS keeps the raw target selector");
    check(role_lanes(tlds, RegisterRole::source_a) == 2 && role_lanes(tlds, RegisterRole::source_b) == 1,
        "TLDS level target reads two coordinates and its level register");
    const DecodedInstruction tlds_ms = *decode_word(Word("1101101xxxxxxx").set(53, 4, 0x6).set(50, 3, 0).set(28, 8, 255));
    check(field(tlds_ms, DecodedField::texture_target) == std::optional<std::uint64_t>{6},
        "TLDS multisample target keeps its selector");
    check(role_lanes(tlds_ms, RegisterRole::source_b) == 1, "TLDS multisample target reads its sample register");
    const DecodedInstruction tlds_f16 = *decode_word(Word("1101001xxxxxxx").set(53, 4, 0x2).set(50, 3, 0).set(28, 8, 255));
    check(tlds_f16.name == InstructionName::tlds_f16, "TLDSF16 keeps its own name");

    const DecodedInstruction tld4 =
        *decode_word(Word("110010xxxxxxxx").set(28, 3, 2).set(56, 2, 2).set(54, 2, 1).set(31, 4, 0b1111));
    check(tld4.name == InstructionName::tld4, "TLD4 decodes by name");
    check(field(tld4, DecodedField::gather_component) == std::optional<std::uint64_t>{2}, "TLD4 preserves its gather component");
    check(field(tld4, DecodedField::texture_offset) == std::optional<std::uint64_t>{1}, "TLD4 records its offset mode");
    check(field(tld4, DecodedField::texture_dimension) == std::optional<std::uint64_t>{2}, "TLD4 keeps the raw TexDim selector");
    check(field(tld4, DecodedField::texture_compare) == std::optional<std::uint64_t>{0} &&
              field(tld4, DecodedField::texture_ndv) == std::optional<std::uint64_t>{0} &&
              field(tld4, DecodedField::texture_lod_clamp) == std::optional<std::uint64_t>{0},
        "TLD4 publishes its boolean modifiers");
    const DecodedInstruction tld4b = *decode_word(Word("1101111011xxxx").set(38, 2, 3).set(36, 2, 2));
    check(tld4b.name == InstructionName::tld4_b, "TLD4B decodes by name");
    check(field(tld4b, DecodedField::gather_component) == std::optional<std::uint64_t>{3}, "TLD4B reads its component from the bindless bit block");
    check(field(tld4b, DecodedField::texture_offset) == std::optional<std::uint64_t>{2}, "TLD4B keeps its packed-offset mode");

    const DecodedInstruction tld4s = *decode_word(
        Word("1101111100xxxx").set(52, 2, 1).set(50, 1, 0).set(28, 8, 40).set(36, 13, 0x20));
    check(tld4s.name == InstructionName::tld4s, "TLD4S decodes by name");
    check(field(tld4s, DecodedField::gather_component) == std::optional<std::uint64_t>{1}, "TLD4S preserves its gather component");
    check(field(tld4s, DecodedField::texture_compare) == std::optional<std::uint64_t>{0} &&
              field(tld4s, DecodedField::texture_offset) == std::optional<std::uint64_t>{0},
        "TLD4S publishes explicit compare and offset booleans");
    check(role_lanes(tld4s, RegisterRole::destination) == 2 &&
              role_lanes(tld4s, RegisterRole::second_destination) == 2,
        "TLD4S splits RGBA results across two registers");
    const DecodedInstruction tld4s_f16 = *decode_word(Word("1101111110xxxx").set(28, 8, 255));
    check(tld4s_f16.name == InstructionName::tld4s_f16, "TLD4SF16 keeps its own name");
    check(role_lanes(tld4s_f16, RegisterRole::destination) == 2 &&
              role_lanes(tld4s_f16, RegisterRole::second_destination) == 0,
        "TLD4SF16 with Dest2=RZ writes one packed pair");

    const DecodedInstruction tmml = *decode_word(Word("1101111101011x").set(28, 3, 2).set(36, 13, 4));
    check(tmml.name == InstructionName::tmml, "TMML decodes by name");
    check(field(tmml, DecodedField::texture_dimension) == std::optional<std::uint64_t>{2}, "TMML keeps the raw TexDim selector");
    check(field(tmml, DecodedField::texture_ndv) == std::optional<std::uint64_t>{0}, "TMML publishes its Ndv flag");
    check(has_source(tmml, SourceRole::b) && role_lanes(tmml, RegisterRole::source_b) == 0,
        "TMML keeps Rb as an unused formal slot");
    const DecodedInstruction tmmlb = *decode_word(Word("1101111101100x").set(20, 8, 6));
    check(tmmlb.name == InstructionName::tmml_b, "TMMLB decodes by name");
    check(role_lanes(tmmlb, RegisterRole::source_b) == 1, "TMMLB reads its handle register");

    const DecodedInstruction txq = *decode_word(
        Word("1101111101001x").set(22, 6, 1).set(31, 4, 0b0011).set(36, 13, 0x40));
    check(txq.name == InstructionName::txq, "TXQ decodes by name");
    check(field(txq, DecodedField::texture_query) == std::optional<std::uint64_t>{1}, "TXQ keeps the raw query selector");
    check(role_lanes(txq, RegisterRole::destination) == 2, "TXQ writes one lane per mask bit");
    const DecodedInstruction txqb = *decode_word(Word("1101111101010x").set(22, 6, 1).set(8, 8, 4));
    check(txqb.name == InstructionName::txq_b, "TXQB decodes by name");
    check(txqb.encoding == Encoding::texture_bindless, "TXQB is bindless");
    check(role_lanes(txqb, RegisterRole::source_a) == 2, "TXQB reads its handle and LOD registers");

    const DecodedInstruction txd = *decode_word(Word("1101111000xxxx").set(28, 3, 2).set(31, 4, 0b0010));
    check(txd.name == InstructionName::txd, "TXD decodes by name");
    check(field(txd, DecodedField::texture_dimension) == std::optional<std::uint64_t>{2}, "TXD keeps the raw TexDim selector");
    check(field(txd, DecodedField::texture_offset) == std::optional<std::uint64_t>{0} &&
              field(txd, DecodedField::texture_lod_clamp) == std::optional<std::uint64_t>{0},
        "TXD publishes explicit offset and clamp booleans");
    check(role_lanes(txd, RegisterRole::source_a) == 3, "TXD reads coordinates plus its packed parameter register");
    check(role_lanes(txd, RegisterRole::source_b) == 4, "TXD reads two gradient words per axis");
    const DecodedInstruction txdb = *decode_word(Word("1101111001xxxx").set(8, 8, 3));
    check(txdb.name == InstructionName::txd_b, "TXDB decodes by name");

    const DecodedInstruction txa = *decode_word(Word("1101111101000x").set(31, 4, 0b0001));
    check(txa.name == InstructionName::txa, "TXA decodes by name");
    check(field(txa, DecodedField::texture_handle_byte_offset) == std::optional<std::uint64_t>{0},
        "TXA records its bound handle offset");
}

// --- Rejection of malformed fields -------------------------------------------

void test_rejection() {
    rejects(Word("1110111110010x").set(48, 3, 6).bits, "LDC size selector 6 is malformed");
    rejects(Word("1110111110010x").set(48, 3, 7).bits, "LDC size selector 7 is malformed");
    rejects(Word("1110111101000x").set(48, 3, 7).bits, "LDL size selector 7 is malformed");
    rejects(Word("1110111101011x").set(48, 3, 7).bits, "STS size selector 7 is malformed");
    rejects(Word("11101101xxxxxx").set(49, 3, 6).bits, "ATOM size selector 6 is malformed");
    rejects(Word("11101101xxxxxx").set(49, 3, 2).set(52, 4, 9).bits, "ATOM operation selector 9 is malformed");
    rejects(Word("11101101xxxxxx").set(49, 3, 2).set(52, 4, 12).bits, "ATOM operation selector 12 is malformed");
    rejects(Word("11101100xxxxxx").set(52, 4, 9).bits, "ATOMS operation selector 9 is malformed");
    rejects(Word("1110101111111x").set(20, 3, 6).bits, "RED size selector 6 is malformed");
    rejects(Word("1110101100000x").set(20, 4, 0).bits, "SULDB with an empty component mask is malformed");
    rejects(Word("1110101100101x").set(20, 4, 0).bits, "SUST with an empty component mask is malformed");
    rejects(Word("1110101100001x").set(33, 3, 6).bits, "SULD with an undefined dimension is malformed");
    rejects(Word("11101010x0xxxx").set(29, 4, 9).bits, "SUATOM with an undefined operation is malformed");
    rejects(Word("11000xxxxxxxxx").set(55, 3, 4).bits, "TEX with LOD selector 4 is malformed");
    rejects(Word("11000xxxxxxxxx").set(55, 3, 5).bits, "TEX with LOD selector 5 is malformed");
    rejects(Word("110010xxxxxxxx").set(54, 2, 3).bits, "TLD4 with offset selector 3 is malformed");
    rejects(Word("1101100xxxxxxx").set(53, 4, 14).bits, "TEXS with undefined target 14 is malformed");
    rejects(Word("1101101xxxxxxx").set(53, 4, 3).bits, "TLDS with undefined target 3 is malformed");
    rejects(Word("1101100xxxxxxx").set(53, 4, 1).set(50, 3, 5).set(28, 8, 40).bits,
        "TEXS dual destination with an unsplittable write mask is malformed");
}

} // namespace

int main() {
    test_rule_table();
    test_flow();
    test_attributes();
    test_memory();
    test_surfaces();
    test_textures();
    test_rejection();
    if (failures) {
        std::cerr << failures << " decode resource checks failed\n";
        return 1;
    }
    std::cout << "Shader pipeline decode resource passed\n";
    return 0;
}
