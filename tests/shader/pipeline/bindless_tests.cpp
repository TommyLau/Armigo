// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Bindless texture handle resolution, end to end: independently authored
// synthetic instruction words run through decode, emit_ir, SSA construction
// and legalization. No game bytes, captures or dumps are used.

#include "pipeline/phases.h"
#include "pipeline/dispatch.h"

#include <iostream>
#include <map>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
void check(bool value, std::string_view label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
template<class T> T take(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) {
        std::cerr << error->message << '\n';
        throw 1;
    }
    return std::get<T>(std::move(result));
}
template<class T> bool rejected(Result<T> result, DiagnosticCode code, std::string_view label) {
    const auto* error = std::get_if<Diagnostic>(&result);
    if (!error || error->code != code) {
        std::cerr << "FAIL: " << label;
        if (error) std::cerr << " (unexpected: " << error->message << ')';
        std::cerr << '\n';
        ++failures;
        return false;
    }
    return true;
}

TextureDescriptor descriptor() {
    return {{TextureDimension::d2, false, false}, ScalarType::f32,
        ComponentFormat::floating, true, false, 1, "synthetic-format-f32"};
}
class Facts final : public FactSource {
public:
    std::map<FactKey, FactValue> values;
    Facts() {
        values[{FactKind::constant_buffer_bound_mask}] = std::uint32_t{1};
        values[{FactKind::constant_buffer_size, 0}] = std::uint32_t{256};
        values[{FactKind::local_memory_bytes}] = std::uint32_t{64};
        values[fact_query::texture({0, 16}).key()] = descriptor();
        values[fact_query::texture({0, 20}).key()] = descriptor();
    }
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (auto it = values.find(key); it != values.end())
            return Known<FactValue>{it->second, {OriginKind::explicit_option, "synthetic-bindless"}};
        return Unknown{"Synthetic fact unavailable"};
    }
};
MetalProfile profile() { return {"synthetic-metal", 1, 31, 32, 16, true, false, true, false, false, true}; }

std::uint64_t pattern(std::string_view bits) {
    std::uint64_t result{};
    for (std::size_t i = 0; i < bits.size(); ++i) if (bits[i] == '1') result |= std::uint64_t{1} << (63 - i);
    return result;
}
std::uint64_t place(unsigned first, unsigned width, std::uint64_t value) {
    return (value & ((std::uint64_t{1} << width) - 1)) << first;
}
// MOV32I: register := 32-bit literal.
std::uint64_t mov32i(unsigned dest, std::uint32_t literal) {
    return pattern("000000010000") | dest | place(16, 3, 7) | place(20, 32, literal);
}
// MOV with a constant-buffer source operand (slot, word offset).
std::uint64_t mov_cb(unsigned dest, unsigned slot, unsigned word_offset) {
    return pattern("0100110010011") | dest | place(16, 3, 7) |
        place(20, 14, word_offset) | place(34, 5, slot);
}
// MOV between registers; the register source occupies the Rb slot.
std::uint64_t mov_reg(unsigned dest, unsigned source) {
    return pattern("0101110010011") | dest | place(16, 3, 7) | place(20, 8, source);
}
// LDC: B32 load from a static constant-buffer byte address.
std::uint64_t ldc(unsigned dest, unsigned byte_offset) {
    return pattern("1110111110010x") | dest | place(8, 8, 255) | place(16, 3, 7) |
        place(20, 16, byte_offset) | place(48, 3, 4);
}
// LDL: local memory load (handle provenance from memory must fail).
std::uint64_t ldl(unsigned dest) {
    return pattern("1110111011010x") | dest | place(8, 8, 255) | place(16, 3, 7) | place(48, 3, 4);
}
// LOP32I: AND (0) / OR (1) against a 32-bit literal.
std::uint64_t lop32i(unsigned dest, unsigned source, unsigned operation, std::uint32_t literal) {
    return pattern("000001") | dest | place(8, 8, source) | place(16, 3, 7) |
        place(20, 32, literal) | place(53, 2, operation);
}
// LOP between registers.
std::uint64_t lop(unsigned dest, unsigned left, unsigned right, unsigned operation) {
    return pattern("0101110001000") | dest | place(8, 8, left) | place(16, 3, 7) |
        place(20, 8, right) | place(41, 2, operation);
}
// ISETP: P0 := left < right (unsigned), combined with the always-true PT.
std::uint64_t isetp(unsigned left, unsigned right) {
    return pattern("010110110110") | place(8, 8, left) | place(16, 3, 7) |
        place(20, 8, right) | place(39, 3, 7) | place(49, 3, 1);
}
std::uint64_t bra(unsigned from, unsigned to, unsigned predicate) {
    return pattern("111000100100") | 15 | place(16, 3, predicate) |
        place(20, 24, static_cast<std::uint64_t>(to - from - 8) & 0xFFFFFF);
}
std::uint64_t exit_word() { return pattern("111000110000") | place(16, 3, 7) | 15; }
// TEX_B: 2D sample, mask XY, level-zero LOD, coordinates on Ra, handle on Rb.
std::uint64_t texb(unsigned ra, unsigned rb, unsigned dest) {
    return pattern("1101111010xxxx") | place(28, 3, 2) | place(31, 4, 0b0011) | place(37, 3, 1) |
        place(16, 3, 7) | place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TLD_B: integer fetch with the given dimension selector, mask XY.
std::uint64_t tldb(unsigned dimension, unsigned ra, unsigned rb, unsigned dest) {
    return pattern("11011101xxxxxx") | place(28, 3, dimension) | place(31, 4, 0b0011) |
        place(16, 3, 7) | place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TLD4_B: 2D gather of component 1 with one packed offset word behind the handle.
std::uint64_t tld4b(unsigned ra, unsigned rb, unsigned dest) {
    return pattern("1101111011xxxx") | place(28, 3, 2) | place(31, 4, 0b1111) |
        place(36, 2, 1) | place(38, 2, 1) | place(16, 3, 7) | place(51, 3, 7) |
        place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TMML_B: 2D LOD query, mask XY.
std::uint64_t tmmlb(unsigned ra, unsigned rb, unsigned dest) {
    return pattern("1101111101100x") | place(28, 3, 2) | place(31, 4, 0b0011) |
        place(16, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TXD_B: 2D gradient sample; the handle leads Ra, gradients start at Rb.
std::uint64_t txdb(unsigned ra, unsigned rb, unsigned dest) {
    return pattern("1101111001xxxx") | place(28, 3, 2) | place(31, 4, 0b0011) |
        place(16, 3, 7) | place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TXQ_B: size query over the given component mask; the handle leads Ra.
std::uint64_t txqb(unsigned mask, unsigned ra, unsigned dest) {
    return pattern("1101111101010x") | place(22, 6, 1) | place(31, 4, mask) |
        place(16, 3, 7) | place(8, 8, ra) | place(0, 8, dest);
}

// Instruction words only. A scheduling word opens the stream (entry is byte 8),
// zero words land on scheduling slots (pc % 32 == 0), and a trailing scheduling
// word closes an aligned stream.
StageImage image(std::initializer_list<std::uint64_t> words) {
    StageImage result;
    result.stage = Stage::fragment;
    auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) result.bytes.push_back(std::byte((word >> (i * 8)) & 255));
    };
    append(0);
    for (auto word : words) {
        if (result.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    if (result.bytes.size() % 32 == 0) append(0);
    return result;
}
RegisterProgram registers(const StageImage& input, Facts& facts) {
    FactReader reader(facts);
    return take(emit_ir(take(decode(input, reader, {})), reader, {}));
}
LegalProgram legalized(const StageImage& input, Facts& facts) {
    FactReader reader(facts);
    return take(legalize(take(construct_ssa(registers(input, facts), {})), reader, profile()));
}
// Straight-line program with the 2D coordinate pair prepared and the form word
// consuming its operands from R20/R21 (or the handle registers it names).
StageImage form_program(std::uint64_t form) {
    return image({mov32i(20, 0), ldc(4, 16), mov32i(21, 0), form, exit_word()});
}
// Same phases without unwrapping, for cases that must fail at legalization.
Result<LegalProgram> legalization(const StageImage& input, Facts& facts) {
    FactReader reader(facts);
    auto ssa = take(construct_ssa(registers(input, facts), {}));
    return legalize(std::move(ssa), reader, profile());
}

std::size_t texture_index(const Program& p) {
    for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (std::holds_alternative<TextureInfo>(p.operations[i].info)) return i;
    throw 1;
}
const TextureInfo& texture_info(const Program& p) { return std::get<TextureInfo>(p.operations[texture_index(p)].info); }
const TextureUse& texture_use(const LegalProgram& legal) {
    for (const auto& use : legal.requirements.uses) if (const auto* t = std::get_if<TextureUse>(&use)) return *t;
    throw 1;
}

void each_form_resolves() {
    // Per-form programs: the handle register carries the LDC word, and every
    // operand register the form consumes is defined.
    const std::array<std::pair<StageImage, std::string_view>, 6> forms{{
        {image({ldc(4, 16), mov32i(20, 0), mov32i(21, 0), texb(20, 4, 80), exit_word()}), "TEX_B"},
        {image({ldc(4, 16), mov32i(20, 0), mov32i(21, 0), tldb(2, 20, 4, 80), exit_word()}), "TLD_B"},
        {image({ldc(4, 16), mov32i(5, 0), mov32i(20, 0), mov32i(21, 0), tld4b(20, 4, 80), exit_word()}), "TLD4_B"},
        {image({ldc(4, 16), mov32i(20, 0), mov32i(21, 0), tmmlb(20, 4, 80), exit_word()}), "TMML_B"},
        {image({ldc(20, 16), mov32i(4, 0), mov32i(5, 0), mov32i(6, 0),
                mov32i(7, 0), mov32i(21, 0), mov32i(22, 0), txdb(20, 4, 80), exit_word()}), "TXD_B"},
        {image({ldc(20, 16), mov32i(21, 0), txqb(0b0011, 20, 80), exit_word()}), "TXQ_B"},
    }};
    for (const auto& [program, label] : forms) {
        Facts facts;
        auto legal = legalized(program, facts);
        const auto& info = texture_info(legal.ir);
        check(std::get<TextureKey>(info.texture) == TextureKey{0, 16}, std::string{label} + " resolves to the cbuf word");
        check(info.descriptor == descriptor(), std::string{label} + " adopts the descriptor fact");
        check(info.resource.has_value(), std::string{label} + " receives a resolved texture resource");
        // Fetches and size queries use no sampler; every sampled form pairs
        // the combined handle with its sampler.
        const bool sampled = info.mode != TextureMode::fetch && info.mode < TextureMode::query_size;
        check(sampled == info.sampler_resource.has_value(), std::string{label} + " takes a sampler exactly when it samples");
        if (sampled) check(std::get<TextureKey>(*info.sampler) == TextureKey{0, 16},
            std::string{label} + " pairs the combined handle with its sampler");
        const auto& use = texture_use(legal);
        check(use.key == TextureKey{0, 16} && use.descriptor == descriptor(),
            std::string{label} + " plans the resolved texture");
    }
    // The sampled form keeps descriptor-derived floating results.
    Facts facts;
    auto legal = legalized(forms[0].first, facts);
    check(legal.ir.operations[texture_index(legal.ir)].numeric ==
              NumericMode{Domain::floating, ScalarType::f32, Rounding::nearest_even},
        "Descriptor-typed bindless results take the descriptor numeric");
}

void handle_provenance_paths() {
    // Handle read as a constant-buffer operand of the move itself.
    Facts facts;
    auto legal = legalized(image({mov_cb(4, 0, 4), mov32i(20, 0), mov32i(21, 0), texb(20, 4, 80), exit_word()}), facts);
    check(std::get<TextureKey>(texture_info(legal.ir).texture) == TextureKey{0, 16}, "Cbuf-operand MOV handle resolves");
    // Handle copied from an LDC load through a register move.
    legal = legalized(image({mov32i(20, 0), ldc(4, 16), mov_reg(5, 4), mov32i(21, 0), texb(20, 5, 80), exit_word()}), facts);
    check(std::get<TextureKey>(texture_info(legal.ir).texture) == TextureKey{0, 16}, "Copied LDC handle resolves");
    // Handle merged from both routes by a phi with identical provenance.
    // Instruction slots run 8,16,24 | 40,48,56 | 72,80,88 | 104,112 across the
    // automatically inserted scheduling words.
    legal = legalized(image({
        mov32i(10, 0), mov32i(11, 1), mov32i(20, 0), mov32i(21, 0),
        isetp(10, 11), bra(56, 88, 0), mov_cb(4, 0, 4), bra(80, 104, 7),
        ldc(4, 16), texb(20, 4, 80), exit_word()}), facts);
    check(std::get<TextureKey>(texture_info(legal.ir).texture) == TextureKey{0, 16}, "Phi handle with identical provenance resolves");
    const auto& info = texture_info(legal.ir);
    check(info.coordinates.size() == 2 && info.coordinates[0] == InputIndex{0} && info.coordinates[1] == InputIndex{1},
        "Resolved phi program drops the handle input and remaps roles");
}

void separate_sampler_resolves() {
    Facts facts;
    // Combined handle: texture word masked to its low half OR sampler word in
    // the high half, both from constant-buffer words.
    auto legal = legalized(image({
        mov_cb(4, 0, 4), lop32i(5, 4, 0, 0x000fffff), mov_cb(6, 0, 5), lop32i(7, 6, 0, 0xfff00000),
        lop(8, 5, 7, 1), mov32i(20, 0), mov32i(21, 0),
        texb(20, 8, 80), exit_word()}), facts);
    const auto& info = texture_info(legal.ir);
    check(std::get<TextureKey>(info.texture) == TextureKey{0, 16} &&
              std::get<TextureKey>(*info.sampler) == TextureKey{0, 20},
        "Low/high handle split resolves distinct texture and sampler keys");
    bool sampler_use{};
    for (const auto& use : legal.requirements.uses)
        if (const auto* s = std::get_if<SamplerUse>(&use); s && s->key == TextureKey{0, 20}) sampler_use = true;
    check(sampler_use, "Separate sampler key is reflected");
}

void descriptor_dependent_resolution() {
    // A signed fetch adopts the signed component type of its descriptor.
    Facts facts;
    facts.values[fact_query::texture({0, 16}).key()] = TextureDescriptor{{TextureDimension::d2, false, false},
        ScalarType::s32, ComponentFormat::signed_integer, true, false, 1, "synthetic-format-s32"};
    auto legal = legalized(form_program(tldb(2, 20, 4, 80)), facts);
    check(legal.ir.operations[texture_index(legal.ir)].numeric ==
              NumericMode{Domain::signed_integer, ScalarType::s32, Rounding::not_applicable},
        "Signed fetch results adopt the descriptor numeric");
    // A buffer fetch resolves without the level the instruction always stages.
    Facts buffer_facts;
    buffer_facts.values[fact_query::texture({0, 16}).key()] = TextureDescriptor{{TextureDimension::buffer, false, false},
        ScalarType::f32, ComponentFormat::floating, true, false, 1, "synthetic-format-buffer"};
    legal = legalized(image({ldc(4, 16), mov32i(20, 0), tldb(0, 20, 4, 80), exit_word()}), buffer_facts);
    check(std::get<TextureKey>(texture_info(legal.ir).texture) == TextureKey{0, 16} &&
              !texture_info(legal.ir).level_or_bias,
        "Buffer fetch drops the staged level once the descriptor resolves");
    // A size-query component absent from the descriptor shape is never guessed.
    Facts query_facts;
    auto absent = legalization(image({ldc(20, 16), mov32i(21, 0), txqb(0b0100, 20, 80), exit_word()}), query_facts);
    rejected(absent, DiagnosticCode::conflicting_fact, "Absent size-query component fails");
}

void unresolvable_handles_fail() {
    Facts facts;
    // Raw literal bits are not descriptor provenance.
    auto raw = legalization(image({mov32i(4, 0x30), mov32i(20, 0), mov32i(21, 0), texb(20, 4, 80), exit_word()}), facts);
    rejected(raw, DiagnosticCode::unresolved_resource, "Literal handle bits fail at legalize");
    // A handle computed from arbitrary words has no provenance either.
    auto computed = legalization(image({mov32i(4, 0x30), lop32i(5, 4, 0, 0xfffff), mov32i(20, 0),
        mov32i(21, 0), texb(20, 5, 80), exit_word()}), facts);
    rejected(computed, DiagnosticCode::unresolved_resource, "Computed handle fails at legalize");
    // A handle read from local memory is not a descriptor address.
    auto memory = legalization(image({ldl(4), mov32i(20, 0), mov32i(21, 0), texb(20, 4, 80), exit_word()}), facts);
    rejected(memory, DiagnosticCode::unresolved_resource, "Memory handle fails at legalize");
}
} // namespace

int main() {
    for (auto test : {each_form_resolves, handle_provenance_paths, separate_sampler_resolves,
                      descriptor_dependent_resolution, unresolvable_handles_fail}) {
        try {
            test();
        } catch (...) {
            ++failures;
            std::cerr << "Test could not complete\n";
        }
    }
    if (failures) {
        std::cerr << failures << " bindless resolution checks failed\n";
        return 1;
    }
    std::cout << "Shader pipeline bindless handle resolution passed\n";
    return 0;
}
