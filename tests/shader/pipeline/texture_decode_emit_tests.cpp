// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture decode/emit alignment: real synthetic instruction words flow through
// the published resource decode rules and then the registered texture emitter
// into the recording context. Words are independently authored. No game bytes,
// captures or dumps are used.

#include "pipeline/dispatch.h"
#include "pipeline/decode/resource_fields.h"
#include "pipeline/emit/context_ir.h"

#include <array>
#include <bit>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;

int failures{};
bool check(bool value, std::string_view message) {
    if (!value) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
    return value;
}
template<class T> T expect(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) {
        ++failures; std::cerr << "Unexpected diagnostic: " << error->message << '\n';
        return {};
    }
    return std::move(std::get<T>(result));
}

// Builds a synthetic guest word from a 14-bit dispatch pattern ('1', '0', 'x')
// with fields OR-ed in at explicit bit positions.
struct Word {
    std::uint64_t bits{};
    Word(std::string_view pattern) {
        for (unsigned index = 0; index < pattern.size(); ++index) {
            if (pattern[index] == '1') bits |= std::uint64_t{1} << (63 - index);
        }
    }
    Word& set(unsigned first, unsigned width, std::uint64_t value) {
        const std::uint64_t mask = (std::uint64_t{1} << width) - 1;
        bits |= (value & mask) << first;
        return *this;
    }
    Word& bit(unsigned first, bool value = true) { return set(first, 1, value ? 1 : 0); }
    [[nodiscard]] operator std::uint64_t() const { return bits; }
};

// Shared synthetic operands: coordinates start at R20, auxiliaries at R40,
// results land in R80/R100, and the bound handle sits at word offset 20 so the
// descriptor fact lookup (slot 3, byte 80) succeeds.
constexpr unsigned coord_reg = 20, aux_reg = 40, dest_reg = 80, dest2_reg = 100, handle_words = 20;

class Facts final : public FactSource {
public:
    TextureDescriptor descriptor{{TextureDimension::d2, false, false}, ScalarType::f32,
                                 ComponentFormat::floating, true, false, 1, "synthetic-v1"};
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::texture_handle_cbuf)
            return Known<FactValue>{std::uint32_t{3}, {OriginKind::explicit_option, "synthetic-slot"}};
        if (key.kind == FactKind::texture_descriptor && key.index == 3 && key.byte_offset == handle_words * 4)
            return Known<FactValue>{descriptor, {OriginKind::explicit_option, "synthetic-descriptor"}};
        return Unknown{"Synthetic texture fact is absent"};
    }
};

// Lane-3 style dispatch over the published resource rules only.
std::optional<DecodedInstruction> decode_word(std::uint64_t bits) {
    const RawInstruction raw{bits};
    for (const DecodeRule& candidate : resource_decode_rules()) {
        if ((bits & candidate.mask) == candidate.value) {
            Result<DecodedInstruction> result = candidate.decode(8, raw);
            if (const auto* error = std::get_if<Diagnostic>(&result)) {
                std::cerr << "decode error: " << error->message << '\n';
                return std::nullopt;
            }
            return std::get<DecodedInstruction>(std::move(result));
        }
    }
    std::cerr << "decode error: no resource rule matched\n";
    return std::nullopt;
}

bool decodes(std::uint64_t bits, std::string_view message) {
    const auto decoded = decode_word(bits);
    if (!decoded) { check(false, message); return false; }
    return true;
}

bool decode_rejects(std::uint64_t bits, std::string_view message) {
    const RawInstruction raw{bits};
    for (const DecodeRule& candidate : resource_decode_rules()) {
        if ((bits & candidate.mask) == candidate.value) {
            const Result<DecodedInstruction> result = candidate.decode(8, raw);
            const auto* error = std::get_if<Diagnostic>(&result);
            check(error && error->phase == Phase::decode && error->code == DiagnosticCode::malformed_input,
                  message);
            return error != nullptr;
        }
    }
    check(false, message);
    return false;
}

struct Emission { Status status; Program program; };
Emission emit(const DecodedInstruction& decoded, const Facts& facts = {}) {
    FactReader reader(facts);
    DecodedProgram program_in;
    program_in.stage = Stage::fragment;
    program_in.entry = FunctionId{0};
    program_in.code_bytes = 24;
    program_in.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}});
    DecodedBlock block;
    block.function = FunctionId{0};
    block.begin_pc = 8;
    block.end_pc = 16;
    program_in.blocks.push_back(block);
    RegisterEmitter context(program_in, reader, Limits{});
    expect(context.begin_function(FunctionId{0}));
    expect(context.begin_block(BlockId{0}));
    context.begin_instruction(decoded);
    auto status = dispatch_instruction(context);
    if (std::holds_alternative<std::monostate>(status)) {
        expect(context.terminate(Return{}));
        expect(context.finish_function(BlockId{0}));
    }
    auto program = context.take_program();
    if (std::holds_alternative<std::monostate>(status))
        expect(validate(program, IrForm::registers, Phase::emit_ir));
    return {std::move(status), std::move(program)};
}
bool success(const Emission& e, std::string_view message) {
    if (const auto* error = std::get_if<Diagnostic>(&e.status)) {
        std::cerr << "Emission failed (" << message << "): " << error->message << '\n';
        ++failures;
        return false;
    }
    return true;
}
void emit_rejects(std::uint64_t bits, DiagnosticCode code, std::string_view message, const Facts& facts = {}) {
    const auto decoded = decode_word(bits);
    if (!decoded) { check(false, message); return; }
    const auto e = emit(*decoded, facts);
    const auto* error = std::get_if<Diagnostic>(&e.status);
    if (!error) { check(false, message); return; }
    if (error->code != code || error->phase != Phase::emit_ir || error->pc != 8) {
        std::cerr << "Diagnostic mismatch: code " << static_cast<int>(error->code) << " phase "
                  << static_cast<int>(error->phase) << " pc " << error->pc.value_or(Pc{}) << " message "
                  << error->message << '\n';
        check(false, message);
    }
}

std::vector<const Operation*> textures(const Program& p) {
    std::vector<const Operation*> result;
    for (const auto& op : p.operations) if (std::holds_alternative<TextureInfo>(op.info)) result.push_back(&op);
    return result;
}
const Operation* definition(const Program& p, ValueId id) {
    for (const auto& op : p.operations) for (const auto& output : op.outputs)
        if (const auto* value = std::get_if<ValueId>(&output); value && *value == id) return &op;
    return nullptr;
}
Operand origin(const Program& p, Operand value) {
    for (unsigned depth = 0; depth < 16; ++depth) {
        const auto* id = std::get_if<ValueId>(&value);
        if (!id) return value;
        const auto* def = definition(p, *id);
        if (!def || def->opcode != Opcode::copy) return value;
        value = def->inputs[0];
    }
    return value;
}
bool reads(const Program& p, const Operation& op, InputIndex index, unsigned expected) {
    const auto value = origin(p, op.inputs.at(index.value));
    const auto* r = std::get_if<Register>(&value);
    return r && r->file == RegisterFile::gpr && r->index == expected;
}
bool literal(const Program& p, const Operation& op, InputIndex index, unsigned expected) {
    const auto value = origin(p, op.inputs.at(index.value));
    const auto* bits = std::get_if<LiteralBits>(&value);
    return bits && bits->bits == expected;
}
std::vector<unsigned> writes(const Program& p) {
    std::vector<unsigned> result;
    for (const auto& op : p.operations) if (op.opcode == Opcode::copy && !op.outputs.empty())
        if (const auto* r = std::get_if<Register>(&op.outputs[0]); r && r->file == RegisterFile::gpr)
            result.push_back(r->index);
    return result;
}

// TEXS/TLDS: target at 53..56, mask selector at 50..52, Dest at 0..8,
// Dest2 at 28..36, bound handle at 36..49, Ra at 8..16, Rb at 20..28.
Word texs_word(const char* pattern, unsigned target, unsigned mask, unsigned dest2) {
    Word word{pattern};
    word.set(53, 4, target).set(50, 3, mask).set(0, 8, dest_reg).set(28, 8, dest2)
        .set(36, 13, handle_words).set(8, 8, coord_reg).set(20, 8, aux_reg);
    return word;
}

void test_texs_rows() {
    struct Row { unsigned target, dimension; bool array, shadow; std::vector<unsigned> coords;
                 int layer, compare, lod; TextureMode mode; };
    const std::array<Row, 14> rows{{
        {0,0,false,false,{coord_reg},-1,-1,0,TextureMode::sample_level},
        {1,1,false,false,{coord_reg,aux_reg},-1,-1,-1,TextureMode::sample_implicit},
        {2,1,false,false,{coord_reg,aux_reg},-1,-1,0,TextureMode::sample_level},
        {3,1,false,false,{coord_reg,coord_reg+1},-1,-1,aux_reg,TextureMode::sample_level},
        {4,1,false,true,{coord_reg,coord_reg+1},-1,aux_reg,-1,TextureMode::sample_implicit},
        {5,1,false,true,{coord_reg,coord_reg+1},-1,aux_reg,aux_reg+1,TextureMode::sample_level},
        {6,1,false,true,{coord_reg,coord_reg+1},-1,aux_reg,0,TextureMode::sample_level},
        {7,1,true,false,{coord_reg+1,aux_reg},coord_reg,-1,-1,TextureMode::sample_implicit},
        {8,1,true,false,{coord_reg+1,aux_reg},coord_reg,-1,0,TextureMode::sample_level},
        {9,1,true,true,{coord_reg+1,aux_reg},coord_reg,aux_reg+1,0,TextureMode::sample_level},
        {10,2,false,false,{coord_reg,coord_reg+1,aux_reg},-1,-1,-1,TextureMode::sample_implicit},
        {11,2,false,false,{coord_reg,coord_reg+1,aux_reg},-1,-1,0,TextureMode::sample_level},
        {12,3,false,false,{coord_reg,coord_reg+1,aux_reg},-1,-1,-1,TextureMode::sample_implicit},
        {13,3,false,false,{coord_reg,coord_reg+1,aux_reg},-1,-1,aux_reg+1,TextureMode::sample_level}
    }};
    for (const auto& row : rows) for (auto name : {std::pair{"1101100xxxxxxx", InstructionName::texs},
                                                   std::pair{"1101000xxxxxxx", InstructionName::texs_f16}}) {
        const auto decoded = decode_word(texs_word(name.first, row.target, 4, dest2_reg));
        if (!decoded) { check(false, "TEXS row decodes"); continue; }
        check(decoded->name == name.second, "TEXS row keeps its name");
        Facts facts;
        facts.descriptor.shape = {static_cast<TextureDimension>(row.dimension), row.array, false};
        facts.descriptor.comparison = row.shadow;
        const auto e = emit(*decoded, facts);
        if (!success(e, "TEXS row emits")) continue;
        const auto ops = textures(e.program);
        check(ops.size() == 1, "TEXS row has one sample");
        const auto& op = *ops[0];
        const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == row.mode && t.coordinates.size() == row.coords.size(), "TEXS target semantics survive decoding");
        for (unsigned n = 0; n < row.coords.size(); ++n) check(reads(e.program, op, t.coordinates[n], row.coords[n]),
            "TEXS coordinate routing follows the target table");
        check(t.array_layer.has_value() == row.array && t.compare.has_value() == row.shadow,
            "TEXS array and compare roles follow the target table");
        if (row.layer >= 0) check(reads(e.program, op, *t.array_layer, row.layer), "TEXS layer source");
        if (row.compare >= 0) check(reads(e.program, op, *t.compare, row.compare), "TEXS comparison source");
        check(t.level_or_bias.has_value() == (row.lod >= 0), "TEXS LOD presence");
        if (row.lod == 0) check(literal(e.program, op, *t.level_or_bias, 0), "TEXS zero LOD");
        if (row.lod > 0) check(reads(e.program, op, *t.level_or_bias, row.lod), "TEXS explicit LOD");
        check(writes(e.program) == (name.second == InstructionName::texs ?
              std::vector<unsigned>{80, 81, 100, 101} : std::vector<unsigned>{80, 100}),
            "TEXS splits results across Dest and Dest2");
        check(t.component_mask == 15 && op.outputs.size() == 4, "TEXS dual-destination mask covers four components");
    }
    // Dest2 encoded as RZ keeps the single-destination mask row.
    const auto single = decode_word(texs_word("1101100xxxxxxx", 1, 4, 255));
    if (check(single.has_value(), "TEXS RZ Dest2 decodes")) {
        const auto e = emit(*single);
        if (success(e, "TEXS RZ Dest2 emits")) {
            check(writes(e.program) == std::vector<unsigned>{80, 81}, "TEXS RZ Dest2 packs two results into Dest");
            check(std::get<TextureInfo>(textures(e.program)[0]->info).component_mask == 3,
                "TEXS RZ Dest2 uses the single-destination mask row");
        }
    }
    decode_rejects(texs_word("1101100xxxxxxx", 14, 4, dest2_reg), "Reserved TEXS target 14 fails decode");
    decode_rejects(texs_word("1101100xxxxxxx", 15, 4, dest2_reg), "Reserved TEXS target 15 fails decode");
    decode_rejects(texs_word("1101100xxxxxxx", 1, 5, dest2_reg), "Reserved dual TEXS mask fails decode");
}

void test_tlds_rows() {
    struct Row { unsigned target, dimension; bool array, ms; std::vector<unsigned> coords;
                 int lod, sample, offset_first; };
    const std::array<Row, 9> rows{{
        {0,0,false,false,{coord_reg},0,-1,-1}, {1,0,false,false,{coord_reg},aux_reg,-1,-1},
        {2,1,false,false,{coord_reg,aux_reg},0,-1,-1}, {4,1,false,false,{coord_reg,coord_reg+1},0,-1,aux_reg},
        {5,1,false,false,{coord_reg,coord_reg+1},aux_reg,-1,-1},
        {6,1,false,true,{coord_reg,coord_reg+1},-1,aux_reg,-1},
        {7,2,false,false,{coord_reg,coord_reg+1,aux_reg},0,-1,-1},
        {8,1,true,false,{aux_reg,aux_reg+1},0,-1,-1},
        {12,1,false,false,{coord_reg,coord_reg+1},aux_reg,-1,aux_reg+1}
    }};
    for (const auto& row : rows) for (auto name : {std::pair{"1101101xxxxxxx", InstructionName::tlds},
                                                   std::pair{"1101001xxxxxxx", InstructionName::tlds_f16}}) {
        const auto decoded = decode_word(texs_word(name.first, row.target, 4, dest2_reg));
        if (!decoded) { check(false, "TLDS row decodes"); continue; }
        check(decoded->name == name.second, "TLDS row keeps its name");
        Facts facts;
        facts.descriptor.shape = {static_cast<TextureDimension>(row.dimension), row.array, row.ms};
        const auto e = emit(*decoded, facts);
        if (!success(e, "TLDS row emits")) continue;
        const auto& op = *textures(e.program)[0];
        const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == TextureMode::fetch && t.coordinates.size() == row.coords.size(), "TLDS fetch shape");
        for (unsigned n = 0; n < row.coords.size(); ++n) check(reads(e.program, op, t.coordinates[n], row.coords[n]),
            "TLDS coordinate routing follows the target table");
        if (row.array) check(t.array_layer && reads(e.program, op, *t.array_layer, coord_reg), "TLDS layer comes from Ra");
        check(t.level_or_bias.has_value() == (row.lod >= 0), "TLDS LOD presence");
        if (row.lod == 0) check(literal(e.program, op, *t.level_or_bias, 0), "TLDS zero LOD");
        if (row.lod > 0) check(reads(e.program, op, *t.level_or_bias, row.lod), "TLDS explicit LOD");
        if (row.sample >= 0) check(t.sample_index && reads(e.program, op, *t.sample_index, row.sample), "TLDS sample register");
        check(t.offsets.size() == (row.offset_first >= 0 ? 2U : 0U), "TLDS offset count");
        for (unsigned n = 0; n < t.offsets.size(); ++n) {
            const auto* extract = definition(e.program, std::get<ValueId>(op.inputs[t.offsets[n].value]));
            check(extract && extract->opcode == Opcode::bit_extract &&
                      reads(e.program, *extract, InputIndex{0}, row.offset_first) &&
                      literal(e.program, *extract, InputIndex{1}, n * 4) && literal(e.program, *extract, InputIndex{2}, 4),
                "TLDS packed offsets extract signed four-bit fields");
        }
    }
    for (unsigned target : {3, 9, 10, 11, 13, 14, 15})
        decode_rejects(texs_word("1101101xxxxxxx", target, 4, dest2_reg), "Reserved TLDS target fails decode");
    // A 1D LodZero fetch against a buffer descriptor omits the mip level.
    Facts buffer;
    buffer.descriptor.shape.dimension = TextureDimension::buffer;
    const auto decoded = decode_word(texs_word("1101101xxxxxxx", 0, 4, dest2_reg));
    if (check(decoded.has_value(), "TLDS buffer row decodes")) {
        const auto e = emit(*decoded, buffer);
        if (success(e, "TLDS buffer row emits"))
            check(!std::get<TextureInfo>(textures(e.program)[0]->info).level_or_bias, "Buffer fetch omits the mip level");
    }
}

// TEX: TexDim at 28..31, mask at 31..35, Dc at 50, DestPred at 51..54,
// Aoffi at 54, Lod at 55..58, Lc at 58, Ndv at 35, handle at 36..49.
Word tex_word(unsigned dimension, unsigned mask, unsigned lod, bool shadow, bool offset, bool ndv, bool clamp) {
    Word word{"11000xxxxxxxxx"};
    word.set(28, 3, dimension).set(31, 4, mask).set(55, 3, lod).set(51, 3, 7)
        .set(36, 13, handle_words).set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    word.bit(50, shadow).bit(54, offset).bit(35, ndv).bit(58, clamp);
    return word;
}

void test_tex() {
    for (unsigned dimension = 0; dimension < 8; ++dimension) for (bool shadow : {false, true})
        for (unsigned lod : {0, 1, 2, 3}) {
            const auto decoded = decode_word(tex_word(dimension, 0b0011, lod, shadow, true, false, false));
            if (!decoded) { check(false, "TEX row decodes"); continue; }
            Facts facts;
            facts.descriptor.shape = {static_cast<TextureDimension>(dimension / 2), bool(dimension & 1), false};
            facts.descriptor.comparison = shadow;
            const auto e = emit(*decoded, facts);
            if (!success(e, "TEX row emits")) continue;
            const auto& op = *textures(e.program)[0];
            const auto& t = std::get<TextureInfo>(op.info);
            const unsigned count = dimension / 2 == 0 ? 1 : dimension / 2 == 1 ? 2 : 3;
            const unsigned first_coord = coord_reg + (dimension & 1);
            check(t.coordinates.size() == count && t.offsets.size() == count, "TEX dimensional operand counts");
            for (unsigned n = 0; n < count; ++n) check(reads(e.program, op, t.coordinates[n], first_coord + n),
                "TEX coordinates follow the array layer");
            check(t.array_layer.has_value() == bool(dimension & 1), "TEX array layer presence");
            if (dimension & 1) check(reads(e.program, op, *t.array_layer, coord_reg), "TEX layer reads Ra first");
            check(t.compare.has_value() == shadow, "TEX compare presence");
            if (shadow) check(reads(e.program, op, *t.compare, aux_reg + 1 + (lod > 1 ? 1 : 0)), "TEX compare follows level and offsets");
            check(t.mode == (lod == 0 ? TextureMode::sample_implicit : lod == 2 ? TextureMode::sample_bias
                                                                                : TextureMode::sample_level),
                "TEX LOD selector maps to the sample mode");
            if (lod > 1) check(t.level_or_bias && reads(e.program, op, *t.level_or_bias, aux_reg), "TEX explicit LOD source");
            if (lod == 1) check(t.level_or_bias && literal(e.program, op, *t.level_or_bias, 0), "TEX level-zero LOD");
            check(writes(e.program) == std::vector<unsigned>{80, 81}, "TEX writes one lane per mask bit");
        }
    check(decode_rejects(tex_word(2, 0b0011, 4, false, false, false, false), "TEX undefined LOD 4 fails decode"),
        "TEX undefined LOD 4 fails decode");
    check(decode_rejects(tex_word(2, 0b0011, 5, false, false, false, false), "TEX undefined LOD 5 fails decode"),
        "TEX undefined LOD 5 fails decode");
    emit_rejects(tex_word(2, 0b0011, 6, false, false, false, false), DiagnosticCode::unsupported_modifier,
        "TEX alternate bias mode has no proven lowering");
    emit_rejects(tex_word(2, 0b0011, 7, false, false, false, false), DiagnosticCode::unsupported_modifier,
        "TEX alternate level mode has no proven lowering");
    emit_rejects(tex_word(2, 0b0011, 0, false, false, true, false), DiagnosticCode::unsupported_modifier,
        "TEX Ndv modifier has no proven lowering");
    emit_rejects(tex_word(2, 0b0011, 0, false, false, false, true), DiagnosticCode::unsupported_modifier,
        "TEX LOD clamp has no proven lowering");
}

// TLD: TexDim at 28..31, mask at 31..35, Ms at 50, DestPred at 51..54,
// Cl at 54, LL at 55, Toff at 35, handle at 36..49.
Word tld_word(unsigned dimension, bool level, bool offset, bool multisample, bool clamp) {
    Word word{"11011100xxxxxx"};
    word.set(28, 3, dimension).set(31, 4, 0b0011).set(51, 3, 7)
        .set(36, 13, handle_words).set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    word.bit(50, multisample).bit(54, clamp).bit(55, level).bit(35, offset);
    return word;
}

void test_tld() {
    // Level-zero and explicit-level fetches; offsets occupy Rb before the LOD.
    for (unsigned dimension : {2, 3}) for (bool level : {false, true}) {
        const auto decoded = decode_word(tld_word(dimension, level, true, false, false));
        if (!decoded) { check(false, "TLD row decodes"); continue; }
        Facts facts;
        facts.descriptor.shape = {static_cast<TextureDimension>(dimension / 2), bool(dimension & 1), false};
        const auto e = emit(*decoded, facts);
        if (!success(e, "TLD row emits")) continue;
        const auto& op = *textures(e.program)[0];
        const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == TextureMode::fetch, "TLD decodes to a fetch");
        check(t.level_or_bias.has_value(), "TLD always stages a LOD operand");
        if (level) check(reads(e.program, op, *t.level_or_bias, aux_reg), "TLD explicit LOD reads Rb");
        else check(literal(e.program, op, *t.level_or_bias, 0), "TLD level-zero fetch");
        check(t.offsets.size() == 2, "TLD offsets follow the coordinate count");
        check(!t.sample_index.has_value(), "TLD without Ms reads no sample index");
    }
    // Multisample fetches stage the packed offsets before the sample index.
    for (unsigned dimension : {2, 3}) {
        const auto decoded = decode_word(tld_word(dimension, false, true, true, false));
        if (!decoded) { check(false, "TLD MS row decodes"); continue; }
        Facts facts;
        facts.descriptor.shape = {static_cast<TextureDimension>(dimension / 2), bool(dimension & 1), true};
        const auto e = emit(*decoded, facts);
        if (!success(e, "TLD MS row emits")) continue;
        const auto& op = *textures(e.program)[0];
        const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == TextureMode::fetch && !t.level_or_bias, "TLD multisample fetch omits the mip level");
        check(t.sample_index && reads(e.program, op, *t.sample_index, aux_reg + 1),
            "TLD sample index follows the packed offsets");
    }
    emit_rejects(tld_word(2, false, false, false, true), DiagnosticCode::unsupported_modifier,
        "TLD clamp has no proven lowering");
    Facts multisample;
    multisample.descriptor.shape.multisample = true;
    emit_rejects(tld_word(2, true, true, true, false), DiagnosticCode::unsupported_modifier,
        "Multisample fetch with explicit LOD is unsupported", multisample);
}

// TLD4: TexDim at 28..31, mask at 31..35, Dc at 50, DestPred at 51..54,
// Toff at 54..56, TexComp at 56..58, Lc at 58, Ndv at 35, handle at 36..49.
Word tld4_word(unsigned component, unsigned offset, bool shadow, bool ndv, bool clamp) {
    Word word{"110010xxxxxxxx"};
    word.set(28, 3, 2).set(31, 4, 0b1111).set(56, 2, component).set(54, 2, offset).set(51, 3, 7)
        .set(36, 13, handle_words).set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    word.bit(50, shadow).bit(35, ndv).bit(58, clamp);
    return word;
}

void test_tld4() {
    for (unsigned component : {0, 1, 2, 3}) for (unsigned offset : {0, 1, 2}) {
        const auto decoded = decode_word(tld4_word(component, offset, false, false, false));
        if (!decoded) { check(false, "TLD4 row decodes"); continue; }
        const auto e = emit(*decoded);
        if (!success(e, "TLD4 row emits")) continue;
        const auto& op = *textures(e.program)[0];
        const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == TextureMode::gather && t.gather_component == component, "TLD4 gather component survives decoding");
        check(t.component_mask == 15, "TLD4 gathers into a full result mask");
        check(t.offsets.size() == (offset == 2 ? 8U : offset * 2), "TLD4 offset-set packing follows TexOffset");
        for (unsigned n = 0; n < t.offsets.size(); ++n) {
            const auto* extract = definition(e.program, std::get<ValueId>(op.inputs[t.offsets[n].value]));
            check(extract && reads(e.program, *extract, InputIndex{0}, aux_reg + n / 4) &&
                      literal(e.program, *extract, InputIndex{1}, (n % 4) * 8) &&
                      literal(e.program, *extract, InputIndex{2}, 6),
                "TLD4 offsets extract signed six-bit fields from Rb");
        }
    }
    // Depth-compare gathers force component zero and read the reference last.
    const auto shadow = decode_word(tld4_word(0, 1, true, false, false));
    if (check(shadow.has_value(), "TLD4 shadow row decodes")) {
        Facts facts;
        facts.descriptor.comparison = true;
        const auto e = emit(*shadow, facts);
        if (success(e, "TLD4 shadow row emits")) {
            const auto& t = std::get<TextureInfo>(textures(e.program)[0]->info);
            check(t.compare && reads(e.program, *textures(e.program)[0], *t.compare, aux_reg + 1),
                "TLD4 shadow reference follows the packed offsets");
        }
    }
    emit_rejects(tld4_word(0, 0, false, true, false), DiagnosticCode::unsupported_modifier,
        "TLD4 Ndv modifier has no proven lowering");
    check(decode_rejects(tld4_word(0, 3, false, false, false), "TLD4 undefined offset selector fails decode"),
        "TLD4 undefined offset selector fails decode");
}

// TLD4S: component at 52..54, Aoffi at 51, Dc at 50, Dest2 at 28..36,
// handle at 36..49.
Word tld4s_word(const char* pattern, unsigned component, bool offset, bool shadow, unsigned dest2) {
    Word word{pattern};
    word.set(52, 2, component).set(28, 8, dest2).set(36, 13, handle_words)
        .set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    word.bit(51, offset).bit(50, shadow);
    return word;
}

void test_tld4s() {
    // Shadow gathers force component zero; the reference must read zero.
    for (unsigned component : {0, 1, 2, 3}) for (bool offset : {false, true}) for (bool shadow : {false, true}) {
        if (shadow && component != 0) continue;
        for (unsigned dest2 : {dest2_reg, 255u})
            for (auto name : {std::pair{"1101111100xxxx", InstructionName::tld4s},
                              std::pair{"1101111110xxxx", InstructionName::tld4s_f16}}) {
            const auto decoded = decode_word(tld4s_word(name.first, component, offset, shadow, dest2));
            if (!decoded) { check(false, "TLD4S row decodes"); continue; }
            check(decoded->name == name.second, "TLD4S row keeps its name");
            Facts facts;
            facts.descriptor.comparison = shadow;
            const auto e = emit(*decoded, facts);
            if (!success(e, "TLD4S row emits")) continue;
            const auto& op = *textures(e.program)[0];
            const auto& t = std::get<TextureInfo>(op.info);
            check(t.mode == TextureMode::gather && t.gather_component == component, "TLD4S gather component survives decoding");
            check(t.coordinates.size() == 2, "TLD4S reads a 2D coordinate pair");
            if (shadow || offset) {
                check(reads(e.program, op, t.coordinates[0], coord_reg) &&
                          reads(e.program, op, t.coordinates[1], coord_reg + 1),
                    "TLD4S shadow and offset gathers split coordinates across Ra");
            } else {
                check(reads(e.program, op, t.coordinates[0], coord_reg) && reads(e.program, op, t.coordinates[1], aux_reg),
                    "TLD4S plain gathers take y from Rb");
            }
            check(t.offsets.size() == (offset ? 2U : 0U), "TLD4S offset presence follows the Aoffi flag");
            if (offset) {
                const auto* extract = definition(e.program, std::get<ValueId>(op.inputs[t.offsets[0].value]));
                check(extract && reads(e.program, *extract, InputIndex{0}, aux_reg),
                    "TLD4S packed offsets read Rb before the reference");
            }
            check(t.compare.has_value() == shadow, "TLD4S compare presence follows the Dc flag");
            if (shadow) check(reads(e.program, op, *t.compare, aux_reg + (offset ? 1 : 0)), "TLD4S compare reads Rb");
        }
    }
    // Reserved gather components under depth compare fail explicitly.
    Facts shadow_facts;
    shadow_facts.descriptor.comparison = true;
    emit_rejects(tld4s_word("1101111100xxxx", 1, false, true, dest2_reg), DiagnosticCode::unsupported_modifier,
        "Shadow gather component must be zero", shadow_facts);
}

// TMML: TexDim at 28..31, mask at 31..35, Ndv at 35, handle at 36..49.
Word tmml_word(bool ndv) {
    Word word{"1101111101011x"};
    word.set(28, 3, 2).set(31, 4, 0b0011).set(36, 13, handle_words)
        .set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    word.bit(35, ndv);
    return word;
}

void test_tmml() {
    const auto decoded = decode_word(tmml_word(false));
    if (check(decoded.has_value(), "TMML decodes")) {
        const auto e = emit(*decoded);
        if (success(e, "TMML emits")) {
            const auto ops = textures(e.program);
            check(ops.size() == 2, "TMML expands to both LOD components");
            for (unsigned n = 0; n < ops.size(); ++n) {
                const auto& t = std::get<TextureInfo>(ops[n]->info);
                check(t.mode == TextureMode::query_lod && t.query_component == (n ^ 1), "TMML reverses the LOD pair");
                check(reads(e.program, *ops[n], t.coordinates[0], coord_reg) &&
                          reads(e.program, *ops[n], t.coordinates[1], coord_reg + 1),
                    "TMML coordinates come from Ra");
            }
            check(writes(e.program) == std::vector<unsigned>{80, 81}, "TMML writes its mask lanes in order");
        }
    }
    emit_rejects(tmml_word(true), DiagnosticCode::unsupported_modifier, "TMML Ndv modifier has no proven lowering");
}

// TXQ: query at 22..28, mask at 31..35, handle at 36..49. TXQ has no Rb
// operand, so bits 20..27 stay clear of the query selector.
Word txq_word(unsigned query, unsigned mask) {
    Word word{"1101111101001x"};
    word.set(22, 6, query).set(31, 4, mask).set(36, 13, handle_words)
        .set(8, 8, coord_reg).set(0, 8, dest_reg);
    return word;
}

void test_txq() {
    const auto decoded = decode_word(txq_word(1, 0b1111));
    if (check(decoded.has_value(), "TXQ dimension query decodes")) {
        Facts facts;
        facts.descriptor.shape.dimension = TextureDimension::d3;
        const auto e = emit(*decoded, facts);
        if (success(e, "TXQ dimension query emits")) {
            const auto ops = textures(e.program);
            check(ops.size() == 4, "TXQ dimensions expand per mask lane");
            for (unsigned n = 0; n < ops.size(); ++n) {
                const auto& t = std::get<TextureInfo>(ops[n]->info);
                check(t.mode == (n == 3 ? TextureMode::query_levels : TextureMode::query_size),
                    "TXQ W component queries the level count");
                check(t.query_component == (n == 3 ? 0 : n), "TXQ query component follows the mask lane");
                if (n < 3) check(reads(e.program, *ops[n], *t.level_or_bias, coord_reg), "TXQ size reads its LOD register");
                else check(!t.level_or_bias, "TXQ level count has no LOD operand");
            }
            check(writes(e.program) == std::vector<unsigned>{80, 81, 82, 83}, "TXQ writes one lane per mask bit");
        }
    }
    // Texture-type query against a multisample descriptor reports the samples.
    const auto samples = decode_word(txq_word(2, 0b0100));
    if (check(samples.has_value(), "TXQ texture-type query decodes")) {
        Facts facts;
        facts.descriptor.shape.multisample = true;
        const auto e = emit(*samples, facts);
        if (success(e, "TXQ texture-type query emits"))
            check(std::get<TextureInfo>(textures(e.program)[0]->info).mode == TextureMode::query_samples,
                "TXQ texture-type query reports sample counts");
    }
    emit_rejects(txq_word(5, 0b0001), DiagnosticCode::unsupported_modifier, "Unsupported TXQ query selector fails");
}

// TXD: TexDim at 28..31, mask at 31..35, Toff at 35, Lc at 50, DestPred at
// 51..54, handle at 36..49.
Word txd_word(unsigned dimension, bool offset, bool clamp) {
    Word word{"1101111000xxxx"};
    word.set(28, 3, dimension).set(31, 4, 0b0011).set(51, 3, 7)
        .set(36, 13, handle_words).set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    word.bit(35, offset).bit(50, clamp);
    return word;
}

void test_txd() {
    for (unsigned dimension : {0, 1, 2, 3, 4}) {
        const auto decoded = decode_word(txd_word(dimension, true, false));
        if (!decoded) { check(false, "TXD row decodes"); continue; }
        Facts facts;
        facts.descriptor.shape = {static_cast<TextureDimension>(dimension / 2), bool(dimension & 1), false};
        const auto e = emit(*decoded, facts);
        if (!success(e, "TXD row emits")) continue;
        const auto& op = *textures(e.program)[0];
        const auto& t = std::get<TextureInfo>(op.info);
        const unsigned count = dimension / 2 == 0 ? 1 : dimension / 2 == 1 ? 2 : 3;
        check(t.mode == TextureMode::sample_gradients, "TXD decodes to explicit gradients");
        check(t.gradients_x.size() == count && t.gradients_y.size() == count, "TXD gradient counts follow the shape");
        for (unsigned n = 0; n < count; ++n)
            check(reads(e.program, op, t.gradients_x[n], aux_reg + n) &&
                      reads(e.program, op, t.gradients_y[n], aux_reg + count + n), "TXD gradient vector order");
        check(t.offsets.size() == count, "TXD offsets follow the Toff flag");
        if (dimension & 1) {
            const auto* layer = definition(e.program, std::get<ValueId>(op.inputs[t.array_layer->value]));
            check(layer && layer->opcode == Opcode::bit_and, "TXD array layer unwraps the packed parameter");
        }
        check(writes(e.program) == std::vector<unsigned>{80, 81}, "TXD writes one lane per mask bit");
    }
    emit_rejects(txd_word(2, false, true), DiagnosticCode::unsupported_modifier, "TXD LOD clamp is unsupported");
}

void test_bindless() {
    // Every bindless form decodes with Encoding::texture_bindless and stages
    // the handle register as a dynamic input operand in guest source order: Rb
    // first for TEXB/TLDB/TLD4B/TMMLB, Ra first for TXDB/TXQB.
    // Bindless operand fields live at their own bit positions (TEXB Lodb at
    // 37..40, TLD4B Toff at 36..38, TexComp at 38..40, Lcb at 40). Shape facts
    // are deferred to legalization, so emission consumes no facts.
    Word texb{"1101111010xxxx"};
    texb.set(28, 3, 2).set(31, 4, 0b0011).set(37, 3, 1).set(51, 3, 7)
        .set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    Word tldb{"11011101xxxxxx"};
    tldb.set(28, 3, 2).set(31, 4, 0b0011).set(51, 3, 7)
        .set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    Word tld4b{"1101111011xxxx"};
    tld4b.set(28, 3, 2).set(31, 4, 0b1111).set(38, 2, 1).set(36, 2, 1).set(51, 3, 7)
        .set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    Word tmmlb{"1101111101100x"};
    tmmlb.set(28, 3, 2).set(31, 4, 0b0011).set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    Word txdb{"1101111001xxxx"};
    txdb.set(28, 3, 2).set(31, 4, 0b0011).set(51, 3, 7)
        .set(8, 8, coord_reg).set(20, 8, aux_reg).set(0, 8, dest_reg);
    Word txqb{"1101111101010x"};
    txqb.set(22, 6, 1).set(31, 4, 0b0011).set(8, 8, coord_reg).set(0, 8, dest_reg);
    const auto decoded_texb = decode_word(texb);
    if (check(decoded_texb.has_value(), "TEXB decodes")) {
        check(decoded_texb->encoding == Encoding::texture_bindless, "TEXB stays bindless");
        const auto e = emit(*decoded_texb);
        if (success(e, "TEXB emits a dynamic-handle sample")) {
            const auto& op = *textures(e.program)[0];
            const auto& t = std::get<TextureInfo>(op.info);
            check(std::holds_alternative<DynamicHandle>(t.texture) &&
                      reads(e.program, op, std::get<DynamicHandle>(t.texture).input, aux_reg),
                "TEXB stages the handle from the first Rb register");
            check(t.mode == TextureMode::sample_level && t.coordinates.size() == 2 &&
                      reads(e.program, op, t.coordinates[0], coord_reg) &&
                      reads(e.program, op, t.coordinates[1], coord_reg + 1),
                "TEXB coordinates follow the dimension selector");
            check(t.level_or_bias && literal(e.program, op, *t.level_or_bias, 0),
                "TEXB level-zero LOD is explicit");
            check(writes(e.program) == std::vector<unsigned>{80, 81}, "TEXB writes one lane per mask bit");
        }
    }
    const auto decoded_tldb = decode_word(tldb);
    if (check(decoded_tldb.has_value(), "TLDB decodes")) {
        const auto e = emit(*decoded_tldb);
        if (success(e, "TLDB emits a dynamic-handle fetch")) {
            const auto& op = *textures(e.program)[0];
            const auto& t = std::get<TextureInfo>(op.info);
            check(std::holds_alternative<DynamicHandle>(t.texture) &&
                      reads(e.program, op, std::get<DynamicHandle>(t.texture).input, aux_reg),
                "TLDB stages the handle from the first Rb register");
            check(t.mode == TextureMode::fetch && t.level_or_bias &&
                      literal(e.program, op, *t.level_or_bias, 0),
                "TLDB fetch keeps its explicit zero level");
        }
    }
    const auto decoded_tld4b = decode_word(tld4b);
    if (check(decoded_tld4b.has_value(), "TLD4B decodes")) {
        const auto e = emit(*decoded_tld4b);
        if (success(e, "TLD4B emits a dynamic-handle gather")) {
            const auto& op = *textures(e.program)[0];
            const auto& t = std::get<TextureInfo>(op.info);
            check(std::holds_alternative<DynamicHandle>(t.texture) &&
                      reads(e.program, op, std::get<DynamicHandle>(t.texture).input, aux_reg),
                "TLD4B stages the handle before the packed offsets");
            check(t.mode == TextureMode::gather && t.gather_component == 1, "TLD4B gather component survives decoding");
            check(t.offsets.size() == 2, "TLD4B offset set follows the TexOffset selector");
            for (unsigned n = 0; n < t.offsets.size(); ++n) {
                const auto* extract = definition(e.program, std::get<ValueId>(op.inputs[t.offsets[n].value]));
                check(extract && reads(e.program, *extract, InputIndex{0}, aux_reg + 1) &&
                          literal(e.program, *extract, InputIndex{1}, n * 8) &&
                          literal(e.program, *extract, InputIndex{2}, 6),
                    "TLD4B offsets shift past the handle register");
            }
        }
    }
    const auto decoded_tmmlb = decode_word(tmmlb);
    if (check(decoded_tmmlb.has_value(), "TMMLB decodes")) {
        const auto e = emit(*decoded_tmmlb);
        if (success(e, "TMMLB emits dynamic-handle LOD queries")) {
            const auto ops = textures(e.program);
            check(ops.size() == 2, "TMMLB expands to both LOD components");
            for (unsigned n = 0; n < ops.size(); ++n) {
                const auto& t = std::get<TextureInfo>(ops[n]->info);
                check(std::holds_alternative<DynamicHandle>(t.texture) &&
                          reads(e.program, *ops[n], std::get<DynamicHandle>(t.texture).input, aux_reg),
                    "TMMLB stages the handle from Rb");
                check(t.mode == TextureMode::query_lod && t.query_component == (n ^ 1),
                    "TMMLB reverses the LOD pair");
            }
        }
    }
    const auto decoded_txdb = decode_word(txdb);
    if (check(decoded_txdb.has_value(), "TXDB decodes")) {
        const auto e = emit(*decoded_txdb);
        if (success(e, "TXDB emits a dynamic-handle gradient sample")) {
            const auto& op = *textures(e.program)[0];
            const auto& t = std::get<TextureInfo>(op.info);
            check(std::holds_alternative<DynamicHandle>(t.texture) &&
                      reads(e.program, op, std::get<DynamicHandle>(t.texture).input, coord_reg),
                "TXDB stages the handle from the first Ra register");
            check(t.mode == TextureMode::sample_gradients && t.coordinates.size() == 2 &&
                      reads(e.program, op, t.coordinates[0], coord_reg + 1) &&
                      reads(e.program, op, t.coordinates[1], coord_reg + 2),
                "TXDB coordinates shift past the handle");
            check(t.gradients_x.size() == 2 && t.gradients_y.size() == 2 &&
                      reads(e.program, op, t.gradients_x[0], aux_reg) &&
                      reads(e.program, op, t.gradients_y[1], aux_reg + 3),
                "TXDB gradients follow the coordinate count");
        }
    }
    const auto decoded_txqb = decode_word(txqb);
    if (check(decoded_txqb.has_value(), "TXQB decodes")) {
        const auto e = emit(*decoded_txqb);
        if (success(e, "TXQB emits dynamic-handle size queries")) {
            const auto ops = textures(e.program);
            check(ops.size() == 2, "TXQB expands per mask lane");
            for (unsigned n = 0; n < ops.size(); ++n) {
                const auto& t = std::get<TextureInfo>(ops[n]->info);
                check(std::holds_alternative<DynamicHandle>(t.texture) &&
                          reads(e.program, *ops[n], std::get<DynamicHandle>(t.texture).input, coord_reg),
                    "TXQB stages the handle from the first Ra register");
                check(t.mode == TextureMode::query_size && t.query_component == n &&
                          t.level_or_bias && reads(e.program, *ops[n], *t.level_or_bias, coord_reg + 1),
                    "TXQB size queries read the LOD register after the handle");
                check(ops[n]->numeric.scalar == ScalarType::u32, "TXQB keeps integer query results");
            }
        }
    }
}
} // namespace

int main() {
    test_texs_rows();
    test_tlds_rows();
    test_tex();
    test_tld();
    test_tld4();
    test_tld4s();
    test_tmml();
    test_txq();
    test_txd();
    test_bindless();
    if (failures) {
        std::cerr << failures << " texture decode/emit alignment checks failed\n";
        return 1;
    }
    std::cout << "Shader pipeline texture decode/emit alignment passed\n";
    return 0;
}
