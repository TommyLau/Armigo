// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture behavior tests are independently authored using synthetic decoded
// operands.

#include "pipeline/dispatch.h"
#include "pipeline/emit/context_ir.h"

#include <array>
#include <bit>
#include <iostream>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
void check(bool value, std::string_view message) {
    if (!value) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template<class T> T expect(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) {
        ++failures; std::cerr << "Unexpected diagnostic: " << error->message << '\n';
        return {};
    }
    return std::move(std::get<T>(result));
}
Register reg(unsigned index) { return {RegisterFile::gpr, static_cast<std::uint16_t>(index)}; }
class Facts final : public FactSource {
public:
    TextureDescriptor descriptor{{TextureDimension::d2, false, false}, ScalarType::f32,
                                 ComponentFormat::floating, true, false, 1, "synthetic-v1"};
    bool known_slot{true}, known_descriptor{true};
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::texture_handle_cbuf && known_slot)
            return Known<FactValue>{std::uint32_t{3}, {OriginKind::explicit_option, "synthetic-slot"}};
        if (key.kind == FactKind::texture_descriptor && key.index == 3 && key.byte_offset == 80 && known_descriptor)
            return Known<FactValue>{descriptor, {OriginKind::explicit_option, "synthetic-descriptor"}};
        return Unknown{"Synthetic texture fact is absent"};
    }
};
void set(DecodedInstruction& i, DecodedField key, unsigned value) {
    for (auto& field : i.fields) if (field.field == key) { field.bits = value; return; }
    i.fields.push_back({key, value});
}
void erase(DecodedInstruction& i, DecodedField key) {
    std::erase_if(i.fields, [key](const auto& field) { return field.field == key; });
}
DecodedInstruction instruction(InstructionName name, unsigned mask = 15) {
    DecodedInstruction i;
    i.pc = 8; i.name = name; i.encoding = Encoding::texture_bound;
    i.sources = {{SourceRole::a, reg(20), {}}, {SourceRole::b, reg(40), {}}};
    i.registers = {{RegisterRole::destination, 0, reg(80), Access::write},
                   {RegisterRole::second_destination, 0, reg(100), Access::write}};
    for (auto [key, value] : std::array<std::pair<DecodedField, unsigned>, 12>{{
        {DecodedField::texture_dimension, 2}, {DecodedField::texture_mode, 0},
        {DecodedField::texture_component_mask, mask}, {DecodedField::texture_handle_byte_offset, 80},
        {DecodedField::texture_offset, 0}, {DecodedField::texture_compare, 0},
        {DecodedField::texture_multisample, 0}, {DecodedField::texture_lod_clamp, 0},
        {DecodedField::texture_ndv, 0}, {DecodedField::texture_clamp, 0},
        {DecodedField::gather_component, 0}, {DecodedField::texture_query, 1}}}) set(i, key, value);
    return i;
}
struct Emission { Status status; Program program; std::vector<Dependency> dependencies; };
Emission emit(const DecodedInstruction& i, const Facts& facts = {}, Limits limits = {}) {
    FactReader reader(facts, limits);
    DecodedProgram decoded;
    decoded.stage = Stage::fragment;
    decoded.entry = FunctionId{0}; decoded.code_bytes = 24;
    decoded.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}});
    DecodedBlock block; block.function = FunctionId{0}; block.begin_pc = 8; block.end_pc = 16;
    decoded.blocks.push_back(block);
    RegisterEmitter context(decoded, reader, limits);
    expect(context.begin_function(FunctionId{0}));
    expect(context.begin_block(BlockId{0}));
    context.begin_instruction(i);
    auto status = dispatch_instruction(context);
    if (std::holds_alternative<std::monostate>(status)) {
        expect(context.terminate(Return{}));
        expect(context.finish_function(BlockId{0}));
    }
    auto program = context.take_program();
    if (std::holds_alternative<std::monostate>(status)) expect(validate(program, IrForm::registers, Phase::emit_ir));
    return {std::move(status), std::move(program), {reader.dependencies().begin(), reader.dependencies().end()}};
}
bool success(const Emission& e) {
    if (auto* error = std::get_if<Diagnostic>(&e.status)) {
        std::cerr << "Emission failed: " << error->message << '\n'; return false;
    }
    return true;
}
void reject(const Emission& e, DiagnosticCode code, std::string_view message) {
    const auto* error = std::get_if<Diagnostic>(&e.status);
    check(error && error->code == code && error->phase == Phase::emit_ir && error->pc == 8, message);
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
bool reads(const Program& p, const Operation& op, InputIndex index, unsigned index_expected) {
    const auto value = origin(p, op.inputs.at(index.value));
    const auto* r = std::get_if<Register>(&value);
    return r && *r == reg(index_expected);
}
bool literal(const Program& p, const Operation& op, InputIndex index, unsigned expected) {
    const auto value = origin(p, op.inputs.at(index.value));
    const auto* bits = std::get_if<LiteralBits>(&value);
    return bits && bits->bits == expected;
}
std::vector<unsigned> writes(const Program& p) {
    std::vector<unsigned> result;
    for (const auto& op : p.operations) if (op.opcode == Opcode::copy && !op.outputs.empty())
        if (const auto* r = std::get_if<Register>(&op.outputs[0])) result.push_back(r->index);
    return result;
}
void coordinate_registers(const Emission& e, const Operation& op, std::initializer_list<unsigned> expected) {
    const auto& info = std::get<TextureInfo>(op.info);
    check(info.coordinates.size() == expected.size(), "Coordinate count");
    unsigned index = 0;
    for (auto r : expected) {
        if (index < info.coordinates.size()) check(reads(e.program, op, info.coordinates[index], r), "Coordinate register routing");
        ++index;
    }
}

void test_texs_rows() {
    // Expected source registers and roles are stated separately for every row.
    struct Row { unsigned target, dimension; bool array, shadow; std::vector<unsigned> coords;
                 int layer, compare, lod; TextureMode mode; };
    const std::array<Row, 14> rows{{
        {0,0,false,false,{20},-1,-1,0,TextureMode::sample_level},
        {1,1,false,false,{20,40},-1,-1,-1,TextureMode::sample_implicit},
        {2,1,false,false,{20,40},-1,-1,0,TextureMode::sample_level},
        {3,1,false,false,{20,21},-1,-1,40,TextureMode::sample_level},
        {4,1,false,true,{20,21},-1,40,-1,TextureMode::sample_implicit},
        {5,1,false,true,{20,21},-1,40,41,TextureMode::sample_level},
        {6,1,false,true,{20,21},-1,40,0,TextureMode::sample_level},
        {7,1,true,false,{21,40},20,-1,-1,TextureMode::sample_implicit},
        {8,1,true,false,{21,40},20,-1,0,TextureMode::sample_level},
        {9,1,true,true,{21,40},20,41,0,TextureMode::sample_level},
        {10,2,false,false,{20,21,40},-1,-1,-1,TextureMode::sample_implicit},
        {11,2,false,false,{20,21,40},-1,-1,0,TextureMode::sample_level},
        {12,3,false,false,{20,21,40},-1,-1,-1,TextureMode::sample_implicit},
        {13,3,false,false,{20,21,40},-1,-1,41,TextureMode::sample_level}
    }};
    for (const auto& row : rows) for (auto name : {InstructionName::texs, InstructionName::texs_f16}) {
        auto i = instruction(name, 4); set(i, DecodedField::texture_target, row.target);
        Facts facts; facts.descriptor.shape = {static_cast<TextureDimension>(row.dimension), row.array, false};
        facts.descriptor.comparison = row.shadow;
        auto e = emit(i, facts); if (!success(e)) { check(false, "TEXS row emits"); continue; }
        const auto ops = textures(e.program); check(ops.size() == 1, "TEXS has one sample");
        const auto& op = *ops[0]; const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == row.mode && t.coordinates.size() == row.coords.size(), "TEXS target semantics");
        for (unsigned n = 0; n < row.coords.size(); ++n) check(reads(e.program, op, t.coordinates[n], row.coords[n]), "TEXS coordinate source");
        check(t.array_layer.has_value() == row.array && t.compare.has_value() == row.shadow, "TEXS array and compare roles");
        if (row.layer >= 0) check(reads(e.program, op, *t.array_layer, row.layer), "TEXS layer source");
        if (row.compare >= 0) check(reads(e.program, op, *t.compare, row.compare), "TEXS comparison source");
        check(t.level_or_bias.has_value() == (row.lod >= 0), "TEXS LOD presence");
        if (row.lod == 0) check(literal(e.program, op, *t.level_or_bias, 0), "TEXS zero LOD");
        if (row.lod > 0) check(reads(e.program, op, *t.level_or_bias, row.lod), "TEXS explicit LOD");
        check(writes(e.program) == (name == InstructionName::texs ? std::vector<unsigned>{80,81,100,101} :
              std::vector<unsigned>{80,100}), "TEXS split output destinations");
        check(t.component_mask == 15 && op.outputs.size() == 4, "TEXS full component output");
    }
}

void test_tlds_rows() {
    struct Row { unsigned target, dimension; bool array, ms; std::vector<unsigned> coords; int lod, sample, offset; };
    const std::array<Row, 9> rows{{
        {0,0,false,false,{20},0,-1,-1}, {1,0,false,false,{20},40,-1,-1},
        {2,1,false,false,{20,40},0,-1,-1}, {4,1,false,false,{20,21},0,-1,40},
        {5,1,false,false,{20,21},40,-1,-1}, {6,1,false,true,{20,21},-1,40,-1},
        {7,2,false,false,{20,21,40},0,-1,-1}, {8,1,true,false,{40,41},0,-1,-1},
        {12,1,false,false,{20,21},40,-1,41}
    }};
    for (const auto& row : rows) for (auto name : {InstructionName::tlds, InstructionName::tlds_f16}) {
        auto i = instruction(name, 4); set(i, DecodedField::texture_target, row.target);
        Facts facts; facts.descriptor.shape = {static_cast<TextureDimension>(row.dimension), row.array, row.ms};
        auto e = emit(i, facts); if (!success(e)) { check(false, "TLDS row emits"); continue; }
        const auto& op = *textures(e.program)[0]; const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == TextureMode::fetch && t.coordinates.size() == row.coords.size(), "TLDS shape and mode");
        for (unsigned n = 0; n < row.coords.size(); ++n) check(reads(e.program, op, t.coordinates[n], row.coords[n]), "TLDS coordinate source");
        if (row.array) check(t.array_layer && reads(e.program, op, *t.array_layer, 20), "TLDS layer comes from A");
        check(t.level_or_bias.has_value() == (row.lod >= 0), "TLDS LOD presence");
        if (row.lod == 0) check(literal(e.program, op, *t.level_or_bias, 0), "TLDS zero LOD");
        if (row.lod > 0) check(reads(e.program, op, *t.level_or_bias, row.lod), "TLDS explicit LOD");
        if (row.sample >= 0) check(t.sample_index && reads(e.program, op, *t.sample_index, row.sample), "TLDS sample register");
        check(t.offsets.size() == (row.offset >= 0 ? 2U : 0U), "TLDS offset count");
        for (unsigned n = 0; n < t.offsets.size(); ++n) {
            const auto* extract = definition(e.program, std::get<ValueId>(op.inputs[t.offsets[n].value]));
            check(extract && extract->opcode == Opcode::bit_extract && extract->numeric.domain == Domain::signed_integer,
                  "TLDS offsets are signed extracts");
            if (extract) check(reads(e.program, *extract, InputIndex{0}, row.offset) &&
                literal(e.program, *extract, InputIndex{1}, n * 4) && literal(e.program, *extract, InputIndex{2}, 4), "TLDS packed offset bit range");
        }
    }
    for (unsigned target : {3,9,10,11,13,14,15}) {
        auto i = instruction(InstructionName::tlds, 4); set(i, DecodedField::texture_target, target);
        reject(emit(i), DiagnosticCode::unsupported_modifier, "Reserved TLDS row fails");
    }
    Facts buffer; buffer.descriptor.shape.dimension = TextureDimension::buffer;
    auto i = instruction(InstructionName::tlds, 4); set(i, DecodedField::texture_target, 0);
    auto e = emit(i, buffer); check(success(e), "TLDS buffer selection uses fact");
    if (success(e)) check(!std::get<TextureInfo>(textures(e.program)[0]->info).level_or_bias, "Buffer fetch omits mip level");
}

void test_regular() {
    for (unsigned dimension = 0; dimension < 8; ++dimension) for (bool shadow : {false,true})
        for (unsigned lod = 0; lod < 4; ++lod) {
            auto i = instruction(InstructionName::tex, 5);
            set(i, DecodedField::texture_dimension, dimension); set(i, DecodedField::texture_compare, shadow);
            set(i, DecodedField::texture_mode, lod); set(i, DecodedField::texture_offset, 1);
            Facts f; f.descriptor.shape = {static_cast<TextureDimension>(dimension / 2), bool(dimension & 1), false};
            f.descriptor.comparison = shadow;
            auto e = emit(i, f); if (!success(e)) { check(false, "TEX dimension/array/shadow/LOD emits"); continue; }
            const auto& op = *textures(e.program)[0]; const auto& t = std::get<TextureInfo>(op.info);
            const unsigned count = dimension / 2 == 0 ? 1 : dimension / 2 == 1 ? 2 : 3;
            check(t.coordinates.size() == count && t.offsets.size() == count, "TEX dimensional operand count");
            for (unsigned n = 0; n < count; ++n) check(reads(e.program, op, t.coordinates[n], 20 + (dimension & 1) + n), "TEX coordinates follow array layer");
            if (dimension & 1) check(t.array_layer && reads(e.program, op, *t.array_layer, 20), "TEX array layer first");
            if (shadow) check(t.compare && reads(e.program, op, *t.compare, lod > 1 ? 42 : 41), "TEX compare follows LOD and offsets");
            check(t.mode == (lod == 0 ? TextureMode::sample_implicit : lod == 2 ? TextureMode::sample_bias : TextureMode::sample_level), "TEX LOD mode");
            if (lod > 1) check(t.level_or_bias && reads(e.program, op, *t.level_or_bias, 40), "TEX explicit LOD/bias source");
            if (lod == 1) check(t.level_or_bias && literal(e.program, op, *t.level_or_bias, 0), "TEX level zero");
            check(writes(e.program) == std::vector<unsigned>({80,81}), "Sparse TEX mask packs destinations");
            check(includes(op.effects, Effects::derivative) == (lod == 0 || lod == 2), "Derivative effects follow LOD mode");
        }
    for (unsigned lod : {4,5,6,7}) {
        auto i = instruction(InstructionName::tex); set(i, DecodedField::texture_mode, lod);
        reject(emit(i), DiagnosticCode::unsupported_modifier, "Unknown or unproven LOD modes fail");
    }
    for (bool ms : {false,true}) {
        auto i = instruction(InstructionName::tld); set(i, DecodedField::texture_multisample, ms);
        set(i, DecodedField::texture_mode, ms ? 0 : 1); set(i, DecodedField::texture_offset, 1);
        Facts f; f.descriptor.shape.multisample = ms;
        auto e = emit(i, f); check(success(e), "TLD emits");
        if (!success(e)) continue;
        const auto& op = *textures(e.program)[0]; const auto& t = std::get<TextureInfo>(op.info);
        check(t.mode == TextureMode::fetch && !includes(op.effects, Effects::derivative), "Fetch has no implicit derivative");
        if (ms) check(t.sample_index && reads(e.program, op, *t.sample_index, 41) && !t.level_or_bias, "TLD MS sample after offsets");
        else check(t.level_or_bias && reads(e.program, op, *t.level_or_bias, 40), "TLD explicit integer LOD");
    }
}

void test_gather_and_gradients() {
    for (unsigned component = 0; component < 4; ++component) for (unsigned offset = 0; offset <= 2; ++offset) {
        auto i = instruction(InstructionName::tld4); set(i, DecodedField::gather_component, component); set(i, DecodedField::texture_offset, offset);
        auto e = emit(i); check(success(e), "Gather emits all components and offset forms"); if (!success(e)) continue;
        const auto& op = *textures(e.program)[0]; const auto& t = std::get<TextureInfo>(op.info);
        check(t.gather_component == component && t.component_mask == 15, "Gather component independent of result mask");
        check(t.offsets.size() == (offset == 2 ? 8U : offset * 2), "Gather four-offset packing");
        for (unsigned n = 0; n < t.offsets.size(); ++n) {
            const auto* extract = definition(e.program, std::get<ValueId>(op.inputs[t.offsets[n].value]));
            check(extract && reads(e.program, *extract, InputIndex{0}, 40 + n / 4) &&
                literal(e.program, *extract, InputIndex{1}, (n % 4) * 8) && literal(e.program, *extract, InputIndex{2}, 6), "Gather signed six-bit offsets");
        }
    }
    for (bool shadow : {false,true}) for (bool offset : {false,true}) for (auto name : {InstructionName::tld4s, InstructionName::tld4s_f16}) {
        auto i = instruction(name); set(i, DecodedField::texture_compare, shadow); set(i, DecodedField::texture_offset, offset);
        Facts f; f.descriptor.comparison = shadow;
        auto e = emit(i, f); check(success(e), "Compact gather emits"); if (!success(e)) continue;
        const auto& op = *textures(e.program)[0]; const auto& t = std::get<TextureInfo>(op.info);
        if (shadow || offset) coordinate_registers(e, op, {20,21});
        else coordinate_registers(e, op, {20,40});
        check(t.offsets.size() == (offset ? 2U : 0U), "Compact gather offsets");
        if (shadow) check(t.compare && reads(e.program, op, *t.compare, offset ? 41 : 40), "Compact gather compare source");
    }
    for (unsigned dimension = 0; dimension < 8; ++dimension) {
        auto i = instruction(InstructionName::txd); set(i, DecodedField::texture_dimension, dimension); set(i, DecodedField::texture_offset, 1);
        Facts f; f.descriptor.shape = {static_cast<TextureDimension>(dimension / 2), bool(dimension & 1), false};
        auto e = emit(i, f); check(success(e), "TXD emits"); if (!success(e)) continue;
        const auto& op = *textures(e.program)[0]; const auto& t = std::get<TextureInfo>(op.info);
        const unsigned count = dimension / 2 == 0 ? 1 : dimension / 2 == 1 ? 2 : 3;
        check(t.gradients_x.size() == count && t.gradients_y.size() == count && t.offsets.size() == count, "TXD operand role counts");
        for (unsigned n = 0; n < count; ++n) {
            check(reads(e.program, op, t.gradients_x[n], 40+2*n) && reads(e.program, op, t.gradients_y[n], 40+2*n+1), "TXD gradient pairs per coordinate");
            const auto* extract = definition(e.program, std::get<ValueId>(op.inputs[t.offsets[n].value]));
            check(extract && reads(e.program, *extract, InputIndex{0}, 20+count) && literal(e.program, *extract, InputIndex{1}, 16+n*4), "TXD offsets in packed A parameter");
        }
        if (dimension & 1) {
            const auto* layer = definition(e.program, std::get<ValueId>(op.inputs[t.array_layer->value]));
            check(layer && layer->opcode == Opcode::bit_and && literal(e.program, *layer, InputIndex{1}, 65535), "TXD masks the array layer to 16 bits");
        }
    }
}

void test_queries() {
    auto i = instruction(InstructionName::tmml, 3);
    i.registers[0].reg = reg(20); // Destinations overlap the coordinate registers.
    auto e = emit(i); check(success(e), "TMML emits");
    if (success(e)) {
        const auto ops = textures(e.program); check(ops.size() == 2, "TMML emits both scalar queries");
        for (unsigned n = 0; n < ops.size(); ++n) {
            const auto& t = std::get<TextureInfo>(ops[n]->info);
            check(t.query_component == (n ^ 1) && t.mode == TextureMode::query_lod, "TMML reverses host LOD components");
            coordinate_registers(e, *ops[n], {20,21});
            check(std::holds_alternative<ValueId>(ops[n]->inputs[0]), "TMML coordinates snapshotted before aliasing writes");
        }
        unsigned multiplies{}, truncations{}, converts{};
        for (const auto& op : e.program.operations) {
            if (op.opcode == Opcode::multiply) { ++multiplies; check(literal(e.program, op, InputIndex{1}, 0x43800000), "TMML scales by 256"); }
            if (op.opcode == Opcode::truncate) {
                ++truncations;
                check(!op.numeric.flush_inputs && !op.numeric.flush_output,
                      "TMML truncation preserves unflushed floating semantics");
            }
            if (op.opcode == Opcode::convert) {
                ++converts;
                check(op.numeric.rounding == Rounding::toward_zero && op.numeric.flush_inputs &&
                      std::get<ConversionInfo>(op.info).destination == ScalarType::s32,
                      "TMML fixed-point signed conversion");
                const auto* input = definition(e.program, std::get<ValueId>(op.inputs[0]));
                check(input && input->opcode == Opcode::truncate,
                      "Only an integral TMML value reaches the flushing conversion");
            }
        }
        check(multiplies == 2 && truncations == 2 && converts == 2, "TMML fixed-point expansion");
    }
    for (unsigned mask : {4,8,15}) { i = instruction(InstructionName::tmml, mask); reject(emit(i), DiagnosticCode::unsupported_modifier, "TMML unknown components fail"); }
    i = instruction(InstructionName::txq);
    Facts dimensions; dimensions.descriptor.shape.dimension = TextureDimension::d3;
    e = emit(i, dimensions); check(success(e), "TXQ dimensions emit");
    if (success(e)) {
        const auto ops = textures(e.program); check(ops.size() == 4, "TXQ dimensions expand to four queries");
        for (unsigned n = 0; n < ops.size(); ++n) {
            const auto& t = std::get<TextureInfo>(ops[n]->info);
            check(t.mode == (n == 3 ? TextureMode::query_levels : TextureMode::query_size), "TXQ W is level count");
            check(t.query_component == (n == 3 ? 0 : n) && ops[n]->numeric.scalar == ScalarType::u32, "TXQ component and integer result");
            if (n < 3) check(t.level_or_bias && reads(e.program, *ops[n], *t.level_or_bias, 20), "TXQ size LOD source");
            else check(!t.level_or_bias, "TXQ level count has no LOD operand");
        }
    }
    for (auto dimension : {TextureDimension::d1,TextureDimension::d2,TextureDimension::d3,TextureDimension::cube,TextureDimension::buffer})
        for (bool array : {false,true}) {
            Facts shape; shape.descriptor.shape = {dimension,array,false};
            const unsigned axes = dimension == TextureDimension::d1 || dimension == TextureDimension::buffer ? 1 : dimension == TextureDimension::d3 ? 3 : 2;
            for (unsigned component = 0; component < 3; ++component) {
                auto q = instruction(InstructionName::txq, 1U << component);
                auto result = emit(q, shape);
                if (component >= axes + unsigned(array)) reject(result,DiagnosticCode::unsupported_modifier,"TXQ missing size axis fails");
                else check(success(result),"TXQ spatial axis or array length emits");
            }
        }
    Facts f; f.descriptor.shape.multisample = true;
    i = instruction(InstructionName::txq, 4); set(i, DecodedField::texture_query, 2);
    e = emit(i, f); check(success(e), "TXQ sample count emits");
    if (success(e)) check(std::get<TextureInfo>(textures(e.program)[0]->info).mode == TextureMode::query_samples, "TXQ sample mode");
    set(i, DecodedField::texture_component_mask, 5); reject(emit(i, f), DiagnosticCode::unsupported_modifier, "TXQ unsupported texture-type components fail");
    for (unsigned query : {0,3,5,16,18,20,22,63}) {
        i = instruction(InstructionName::txq); set(i, DecodedField::texture_query, query);
        reject(emit(i), DiagnosticCode::unsupported_modifier, "Unsupported TXQ query fails");
    }
}

void test_masks_and_failures() {
    const std::array<unsigned,8> single{1,2,4,8,3,9,10,12};
    const std::array<unsigned,5> dual{7,11,13,14,15};
    for (bool two : {false,true}) for (unsigned mask = 0; mask < 8; ++mask) for (auto name : {InstructionName::texs,InstructionName::texs_f16}) {
        auto i = instruction(name, mask); set(i, DecodedField::texture_target, 1);
        i.registers[1].reg = reg(two ? 100 : 255);
        auto e = emit(i);
        if (two && mask >= 5) { reject(e, DiagnosticCode::unsupported_modifier, "Reserved dual mask fails"); continue; }
        check(success(e), "Compact mask emits"); if (!success(e)) continue;
        const auto& op = *textures(e.program)[0]; const auto& t = std::get<TextureInfo>(op.info);
        const auto expected = two ? dual[mask] : single[mask];
        check(t.component_mask == expected && op.outputs.size() == static_cast<unsigned>(std::popcount(expected)), "Compact component mask lookup");
        if (name == InstructionName::texs_f16) {
            unsigned pair{};
            for (const auto& pack : e.program.operations) if (pack.opcode == Opcode::pack_half2) {
                check(pack.numeric.rounding == Rounding::nearest_even, "Half pack rounding retained");
                for (unsigned n = 0; n < 2; ++n) {
                    const auto component = pair * 2 + n;
                    if (component < op.outputs.size()) check(std::get<ValueId>(pack.inputs[n]) == std::get<ValueId>(op.outputs[component]), "Half pack ascending components");
                    else check(literal(e.program, pack, InputIndex{static_cast<std::uint16_t>(n)}, 0), "Half pack unused lane is zero");
                }
                ++pair;
            }
            check(pair == 2, "Both compact half pairs explicitly packed");
        }
    }
    for (unsigned mask = 1; mask < 16; ++mask) {
        auto i = instruction(InstructionName::tex, mask); i.registers[0].reg = reg(254);
        auto e = emit(i); check(success(e), "Physical register boundary emits complete texture operation");
        if (success(e)) check(textures(e.program)[0]->outputs.size() == static_cast<unsigned>(std::popcount(mask)) && writes(e.program) == std::vector<unsigned>{254}, "R255 discards writes without losing texture results");
    }
    auto i = instruction(InstructionName::tex, 0); reject(emit(i), DiagnosticCode::unsupported_modifier, "Empty mask is explicit diagnostic");
    i = instruction(InstructionName::tex); erase(i, DecodedField::texture_dimension);
    reject(emit(i), DiagnosticCode::unsupported_modifier, "Missing selector is not defaulted");
    i = instruction(InstructionName::tex); i.fields.push_back({DecodedField::texture_dimension, 2});
    auto duplicate = emit(i); check(std::holds_alternative<Diagnostic>(duplicate.status), "Duplicate selector fails");
    for (auto name : {InstructionName::tex_b,InstructionName::tld_b,InstructionName::tld4_b,InstructionName::tmml_b,InstructionName::txd_b,InstructionName::txq_b})
        reject(emit(instruction(name)), DiagnosticCode::malformed_input, "Bindless name with a bound encoding fails");
    auto bound_named = instruction(InstructionName::tex);
    bound_named.encoding = Encoding::texture_bindless;
    reject(emit(bound_named), DiagnosticCode::malformed_input, "Bound name with a bindless encoding fails");
    auto bindless = instruction(InstructionName::tex_b, 3);
    bindless.encoding = Encoding::texture_bindless;
    erase(bindless, DecodedField::texture_handle_byte_offset);
    auto dynamic = emit(bindless);
    check(success(dynamic), "Bindless TEX stages the handle register as a dynamic operand");
    if (success(dynamic)) {
        const auto& op = *textures(dynamic.program)[0];
        const auto& t = std::get<TextureInfo>(op.info);
        check(std::holds_alternative<DynamicHandle>(t.texture) &&
                  reads(dynamic.program, op, std::get<DynamicHandle>(t.texture).input, 40),
            "Bindless handle input names the first Rb register");
        check(t.descriptor == std::nullopt && dynamic.dependencies.empty(),
            "Bindless emission consumes no descriptor facts and defers the shape");
    }
    for (auto key : {DecodedField::texture_ndv,DecodedField::texture_lod_clamp}) {
        i = instruction(InstructionName::tex); set(i,key,1); reject(emit(i),DiagnosticCode::unsupported_modifier,"Unproven modifier fails");
    }
    i = instruction(InstructionName::tld); set(i,DecodedField::texture_clamp,1); reject(emit(i),DiagnosticCode::unsupported_modifier,"TLD clamp fails");
    i = instruction(InstructionName::tex); i.registers.push_back({RegisterRole::predicate_destination,0,{RegisterFile::predicate,0},Access::write});
    reject(emit(i),DiagnosticCode::unsupported_modifier,"Residency output cannot be dropped");
    i = instruction(InstructionName::tex); i.sources[0].value = reg(255);
    reject(emit(i),DiagnosticCode::unsupported_modifier,"Source sequence overflow is not synthesized zero");
    Facts unknown; unknown.known_slot = false; i = instruction(InstructionName::tex);
    reject(emit(i,unknown),DiagnosticCode::missing_fact,"Unknown default CB slot fails");
    unknown.known_slot = true; unknown.known_descriptor = false;
    reject(emit(i,unknown),DiagnosticCode::missing_fact,"Unknown descriptor fails");
    Facts conflict; conflict.descriptor.shape.dimension = TextureDimension::d3;
    reject(emit(i,conflict),DiagnosticCode::conflicting_fact,"Instruction/descriptor dimension mismatch fails");
    auto e = emit(i); check(e.dependencies.size() == 2, "Default slot and descriptor facts recorded");
    set(i,DecodedField::texture_cbuf_slot,3); e = emit(i);
    check(success(e) && e.dependencies.size() == 1, "Explicit CB slot avoids default fact");
    i.numeric = NumericMode{Domain::floating,ScalarType::f32,Rounding::toward_negative,true,true,true,true};
    e = emit(i); check(success(e),"Numeric metadata accepted without folding");
    if (success(e)) check(textures(e.program)[0]->numeric == *i.numeric,"All numeric modifiers preserved exactly");
    for (auto name : {InstructionName::tmml, InstructionName::txq}) {
        i = instruction(name, 1);
        i.numeric = name == InstructionName::tmml ?
            NumericMode{Domain::floating,ScalarType::f32,Rounding::nearest_even,false,false,false,true} :
            NumericMode{Domain::unsigned_integer,ScalarType::u32,Rounding::not_applicable,false,false,false,true};
        reject(emit(i),DiagnosticCode::unsupported_modifier,"Query numeric modifier cannot be discarded");
    }
    i = instruction(InstructionName::texs_f16, 4); set(i,DecodedField::texture_target,1);
    i.numeric = NumericMode{Domain::floating,ScalarType::f32,Rounding::toward_negative,true,true,true,true};
    e = emit(i); check(success(e),"Compact half metadata emits");
    if (success(e)) for (const auto& op : e.program.operations)
        if (op.opcode == Opcode::pack_half2 || op.opcode == Opcode::texture_sample)
            check(op.numeric == *i.numeric,"Half conversion preserves numeric modifiers exactly");
    Limits limits; limits.operations = 2; i = instruction(InstructionName::tex);
    reject(emit(i,{},limits),DiagnosticCode::limit_exceeded,"Context limits propagate");
}
} // namespace
// SUATOM: a 32-bit integer surface atomic lowers to the
// image atomic operation with its encoded kind, returns the old value in the
// destination, and derives its numeric interpretation from the descriptor.
void test_suatom() {
    Facts facts;
    facts.descriptor = {{TextureDimension::d2, false, false}, ScalarType::u32,
                        ComponentFormat::unsigned_integer, true, false, 1, "synthetic-v1"};
    auto i = instruction(InstructionName::suatom);
    set(i, DecodedField::atomic_operation, 5); // And in guest operation order.
    set(i, DecodedField::memory_element_bytes, 4);
    set(i, DecodedField::memory_elements, 1);
    set(i, DecodedField::texture_dimension, unsigned(TextureDimension::d2));
    const auto emitted = emit(i, facts);
    if (!success(emitted)) { ++failures; return; }
    const auto ops = textures(emitted.program);
    check(ops.size() == 1, "SUATOM emits exactly one texture operation");
    const auto& op = *ops.front();
    check(op.opcode == Opcode::image_atomic, "SUATOM lowers to the image atomic opcode");
    const auto& info = std::get<TextureInfo>(op.info);
    check(info.mode == TextureMode::fetch && info.atomic == AtomicKind::bit_and,
        "SUATOM keeps the fetch mode and the encoded atomic kind");
    check(std::holds_alternative<TextureKey>(info.texture) &&
        std::get<TextureKey>(info.texture) == TextureKey{3, 80},
        "A bound SUATOM resolves its handle against the texture handle buffer");
    check(info.data.size() == 1 && info.coordinates.size() == 2 && !info.array_layer,
        "SUATOM reads one data word and two planar coordinates");
    check(op.numeric.scalar == ScalarType::u32,
        "SUATOM takes its numeric interpretation from the descriptor");
    check(op.outputs.size() == 1, "SUATOM writes the old value to one destination");
    // Floating descriptors have no atomic lowering and remain a
    // structured rejection instead of a substituted format.
    Facts floating;
    const auto rejected = emit(i, floating);
    const auto* error = std::get_if<Diagnostic>(&rejected.status);
    check(error && error->code == DiagnosticCode::unsupported_capability &&
        error->phase == Phase::emit_ir, "SUATOM rejects a floating descriptor component");
}

int main() {
    test_texs_rows(); test_tlds_rows(); test_regular(); test_gather_and_gradients();
    test_queries(); test_masks_and_failures(); test_suatom();
    if (failures) { std::cerr << failures << " texture emitter checks failed\n"; return 1; }
    std::cout << "Texture emitter checks passed\n";
}
