// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic tests for the vertex input class analysis.
// These tests pin Armigo's code-derived property:
// float when any use interprets the register word as floating, uint/sint when
// every use keeps the raw integer bits, and a structured diagnostic when the
// interpretations mix. Guest words below are synthetic encodings.

#include "pipeline/phases.h"
#include "pipeline/dispatch.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
void check(bool value, std::string_view label) {
    if (!value) { std::cerr << "FAIL: " << label << '\n'; ++failures; }
}
template <class T> const T* success(const Result<T>& result, std::string_view label) {
    const auto* value = std::get_if<T>(&result);
    check(value != nullptr, label);
    if (const auto* error = std::get_if<Diagnostic>(&result)) std::cerr << "  " << error->message << '\n';
    return value;
}
template <class T> const Diagnostic* rejected(const Result<T>& result, DiagnosticCode code, std::string_view label) {
    const auto* error = std::get_if<Diagnostic>(&result);
    check(error && error->code == code, label);
    if (error && error->code != code) std::cerr << "  unexpected error: " << error->message << '\n';
    return error;
}
class Facts final : public FactSource {
public:
    std::map<FactKey, FactValue> values;
    Knowledge<FactValue> lookup(FactKey key) const override {
        auto it = values.find(key);
        if (it == values.end()) return Unknown{"Synthetic fact is unavailable"};
        return Known<FactValue>{it->second, {OriginKind::explicit_option, "vertex-class-synthetic"}};
    }
    template <FactKind K> void set(FactQuery<K> key, typename FactTraits<K>::value_type value) {
        values[key.key()] = std::move(value);
    }
};
MetalProfile profile() { return {"synthetic-metal", 1, 31, 32, 16, true, false, true, false, false, true}; }
Operand word(std::uint32_t value) { return LiteralBits{value, ValueType::word32}; }
NumericMode floating() { return {Domain::floating, ScalarType::f32, Rounding::nearest_even}; }

Program empty(Stage stage = Stage::vertex) {
    Program p;
    p.stage = stage; p.entry = FunctionId{0};
    p.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
    p.blocks.push_back({FunctionId{0}, {8, {}}, {}, {}, {}, Return{}});
    return p;
}
ValueId add(Program& p, Operation operation) {
    operation.effects = required_effects(operation);
    ValueId result{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({storage_type(operation.numeric.scalar)});
    operation.outputs = {result};
    p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back(std::move(operation));
    return result;
}
ValueId add_to(Program& p, BlockId block, Operation operation) {
    operation.effects = required_effects(operation);
    ValueId result{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({storage_type(operation.numeric.scalar)});
    operation.outputs = {result};
    p.blocks[block.value].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back(std::move(operation));
    return result;
}
// Keeps a chain alive by returning it: terminator uses root the values, so
// cleanup's dead-code pass cannot remove the uses the analysis must see.
void keep(Program& p, std::initializer_list<ValueId> values) {
    p.functions[0].results.assign(values.size(), ValueType::word32);
    p.blocks[0].terminator = Return{{values.begin(), values.end()}};
}

// A raw generic vertex input word as the fact-less ALD emitter produces it.
ValueId input_load(Program& p, unsigned location) {
    IoInfo info; info.semantic = IoSemantic::generic;
    info.location = static_cast<std::uint16_t>(location);
    info.component_mask = 1;
    info.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
    info.type = {ScalarType::u32, 4};
    return add(p, {{40, InstructionName::ald}, Opcode::load_io, {}, {Domain::bits, ScalarType::u32}, {}, {}, info});
}
ValueId float_multiply(Program& p, Operand value) {
    return add(p, {{40, InstructionName::fmul}, Opcode::multiply, {}, floating(), {value, word(0x3f000000)}, {}, {}});
}
ValueId bit_and(Program& p, Operand value) {
    return add(p, {{40, InstructionName::lop}, Opcode::bit_and, {}, {}, {value, word(255)}, {}, {}});
}
ValueId signed_shift(Program& p, Operand value) {
    return add(p, {{40, InstructionName::shr}, Opcode::shift_right, {},
        {Domain::signed_integer, ScalarType::s32}, {value, word(1)}, {}, {}});
}
ValueId bits_select(Program& p, Operand value) {
    const auto condition = ValueId{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({ValueType::predicate});
    p.functions[0].parameters.push_back(condition);
    return add(p, {{40, InstructionName::sel}, Opcode::select, {}, {}, {condition, value, value}, {}, {}});
}
ValueId unsigned_to_float(Program& p, Operand value) {
    return add(p, {{40, InstructionName::i2f}, Opcode::convert, {}, floating(),
        {value}, {}, ConversionInfo{ScalarType::u32, ScalarType::f32}});
}
void export_raw(Program& p, Operand value) {
    IoInfo info; info.semantic = IoSemantic::generic;
    info.location = 0; info.component_mask = 1;
    info.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
    info.type = {ScalarType::f32, 4};
    info.data = {InputIndex{0}};
    auto store = Operation{{40, InstructionName::ast}, Opcode::store_io,
        required_effects(Operation{{}, Opcode::store_io, {}, {}, {}, {}, info}),
        floating(), {value}, {}, info};
    p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back(std::move(store));
}
StructuredProgram structured(const LegalProgram& legal) {
    return {legal.ir, legal.requirements, legal.profile, {BlockRegion{BlockId{0}}}, {{FunctionId{0}, RegionId{0}}}};
}
Result<LegalProgram> run(Program p, Facts& f) {
    FactReader reader(f);
    return legalize({std::move(p)}, reader, profile());
}
const IoUse* reflected(const LegalProgram& legal, unsigned location) {
    for (const auto& use : legal.requirements.io)
        if (use.semantic == IoSemantic::generic && use.location == location && use.access == Access::read) return &use;
    return nullptr;
}

const IoUse* reflected(const LegalProgram& legal, IoSemantic semantic,
                       Access access, unsigned location) {
    for (const auto& use : legal.requirements.io)
        if (use.semantic == semantic && use.access == access && use.location == location)
            return &use;
    return nullptr;
}

// --- legalization-level classification --------------------------------------

void float_input() {
    Facts f;
    auto p = empty();
    const auto loaded = input_load(p, 0);
    float_multiply(p, loaded);
    auto result = run(p, f);
    if (const auto* legal = success(result, "Floating use classifies the input as float")) {
        const auto* use = reflected(*legal, 0);
        check(use && use->vertex_class == VertexAttributeClass::float_class &&
              use->type == IoType{ScalarType::f32, 4} && !use->vertex_format.has_value(),
            "Float input reflects the float class and float4 type");
        check(std::none_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(),
                [](const auto& d) { return d.key.kind == FactKind::vertex_attribute_format; }),
            "Class derivation records no pipeline-state dependency");
        success(plan_resources(structured(*legal), profile()), "Class-based reflection plans");
    }

    p = empty();
    input_load(p, 0);
    auto classified1 = run(p, f);
    if (const auto* legal = success(classified1, "An unread input stays in the float class")) {
        const auto* use = reflected(*legal, 0);
        check(use && use->vertex_class == VertexAttributeClass::float_class, "Unread input defaults to float");
    }
}

void integer_input() {
    Facts f;
    auto p = empty();
    const auto loaded = input_load(p, 3);
    keep(p, {bit_and(p, loaded)});
    auto classified2 = run(p, f);
    if (const auto* legal = success(classified2, "Bit use classifies the input as uint")) {
        const auto* use = reflected(*legal, 3);
        check(use && use->vertex_class == VertexAttributeClass::uint_class &&
              use->type == IoType{ScalarType::u32, 4},
            "Uint input reflects the uint class and uint4 type");
        const auto& load = legal->ir.operations[0];
        check(load.numeric == NumericMode{Domain::unsigned_integer, ScalarType::u32},
            "Classified uint load carries the unsigned interpretation");
    }

    p = empty();
    const auto copies = input_load(p, 4);
    ValueId moved = copies;
    for (unsigned index = 0; index != 3; ++index) {
        auto copy = Operation{{40, InstructionName::mov}, Opcode::copy, {}, {}, {moved}, {}, {}};
        copy.effects = required_effects(copy);
        const auto result = ValueId{static_cast<std::uint32_t>(p.values.size())};
        p.values.push_back({ValueType::word32});
        copy.outputs = {result};
        p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
        p.operations.push_back(std::move(copy));
        moved = result;
    }
    keep(p, {bit_and(p, moved)});
    auto classified3 = run(p, f);
    if (const auto* legal = success(classified3, "Bit-preserving copies stay transparent")) {
        const auto* use = reflected(*legal, 4);
        check(use && use->vertex_class == VertexAttributeClass::uint_class,
            "Copies do not force the float class");
    }

    p = empty();
    const auto signed_input = input_load(p, 5);
    keep(p, {signed_shift(p, signed_input)});
    auto classified4 = run(p, f);
    if (const auto* legal = success(classified4, "Signed use classifies the input as sint")) {
        const auto* use = reflected(*legal, 5);
        check(use && use->vertex_class == VertexAttributeClass::sint_class &&
              use->type == IoType{ScalarType::s32, 4},
            "Sint input reflects the sint class and int4 type");
    }

    p = empty();
    const auto selected = input_load(p, 6);
    keep(p, {float_multiply(p, bits_select(p, selected))});
    auto classified5 = run(p, f);
    if (const auto* legal = success(classified5, "A raw-word select stays transparent")) {
        const auto* use = reflected(*legal, 6);
        check(use && use->vertex_class == VertexAttributeClass::float_class,
            "Raw select before floating use classifies float");
    }

    p = empty();
    const auto widened = input_load(p, 7);
    keep(p, {unsigned_to_float(p, widened)});
    auto classified6 = run(p, f);
    if (const auto* legal = success(classified6, "An integer conversion source classifies uint")) {
        const auto* use = reflected(*legal, 7);
        check(use && use->vertex_class == VertexAttributeClass::uint_class,
            "u32 conversion source keeps the uint class");
    }

    p = empty();
    const auto raw_export = input_load(p, 8);
    export_raw(p, raw_export);
    auto classified7 = run(p, f);
    if (const auto* legal = success(classified7, "Exporting the raw word reads it as float")) {
        const auto* use = reflected(*legal, 8);
        check(use && use->vertex_class == VertexAttributeClass::float_class,
            "A raw export to a float varying is a floating use");
    }
}

void mixed_use_rejected() {
    Facts f;
    auto p = empty();
    const auto loaded = input_load(p, 2);
    keep(p, {float_multiply(p, loaded), bit_and(p, loaded)});
    auto mixed = run(p, f);
    if (const auto* error = rejected(mixed, DiagnosticCode::unsupported_capability,
            "Mixed interpretations are a structured diagnostic"))
        check(error->message.find("2") != std::string::npos, "Mixed-use diagnostic names the location");

    p = empty();
    const auto loaded_reversed = input_load(p, 1);
    keep(p, {bit_and(p, loaded_reversed), float_multiply(p, loaded_reversed)});
    rejected(run(p, f), DiagnosticCode::unsupported_capability, "Order does not matter for mixed use");
}

void fragment_inputs_are_not_vertex_classified() {
    Facts f;
    constexpr Interpolation perspective{InterpolationMode::perspective,
                                        InterpolationLocation::center};
    f.set(fact_query::interpolation(4, 0), perspective);
    f.set(FactQuery<FactKind::fragment_output_type>{0},
          IoType{ScalarType::u32, 4});

    auto p = empty(Stage::fragment);
    IoInfo input;
    input.semantic = IoSemantic::generic;
    input.location = 4;
    input.component_mask = 1;
    input.interpolation = perspective;
    input.type = {ScalarType::f32, 4};
    const auto varying = add(p, {{40, InstructionName::ipa}, Opcode::load_io, {},
                                 floating(), {}, {}, input});

    IoInfo output;
    output.semantic = IoSemantic::color;
    output.location = 0;
    output.component_mask = 1;
    output.type = {ScalarType::u32, 4};
    output.data = {InputIndex{0}};
    Operation store{{44, InstructionName::ast}, Opcode::store_io, {},
                    {Domain::unsigned_integer, ScalarType::u32}, {varying}, {}, output};
    store.effects = required_effects(store);
    p.blocks[0].operations.push_back(OperationId{
        static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back(std::move(store));

    auto result = run(std::move(p), f);
    const auto* legal = success(result,
        "Raw fragment export does not classify its floating varying as a vertex input");
    if (!legal) return;
    const auto* reflected_input = reflected(*legal, IoSemantic::generic,
                                            Access::read, 4);
    const auto* reflected_output = reflected(*legal, IoSemantic::color,
                                             Access::write, 0);
    check(reflected_input && reflected_input->type == IoType{ScalarType::f32, 4} &&
          !reflected_input->vertex_class && !reflected_input->vertex_format,
          "Fragment varying retains its floating type and interpolation contract");
    check(reflected_output && reflected_output->type == IoType{ScalarType::u32, 4},
          "Fragment colour output retains raw uint4 storage");

    auto direct = structured(*legal);
    auto plan = plan_resources(direct, profile());
    const auto* planned = success(plan, "Adaptive fragment regression plans");
    if (!planned) return;
    EmitOptions options;
    options.fragment_output_types_from_render_targets = true;
    options.fragment_position =
        FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    auto artifact = emit_msl(direct, *planned, options);
    const auto* generated = success(artifact, "Adaptive fragment regression emits MSL");
    if (!generated) return;
    check(generated->source.find(
              "float generic4_0 [[user(locn4_0), center_perspective]];") !=
              std::string::npos,
          "Adaptive fragment MSL declares its varying as floating");
    check(generated->source.find("= as_type<uint>(in.generic4_0);") !=
              std::string::npos &&
          generated->source.find("armigo_color0_raw.x = value") !=
              std::string::npos,
          "Adaptive fragment MSL passes floating varying bits into raw colour storage");
}

void analysis_budget() {
    Facts f;
    auto p = empty();
    const auto loaded = input_load(p, 0);
    float_multiply(p, loaded);
    p.limits.analysis_steps = 1;
    rejected(run(p, f), DiagnosticCode::limit_exceeded, "Class analysis respects its work budget");
}

void fact_path_preserved() {
    Facts f;
    // Old pipeline-state path: the supplied format keeps the exact scalar and
    // the reflected format, including its per-format validation.
    f.set(FactQuery<FactKind::vertex_attribute_format>{1}, AttributeFormat{VertexFormat::f32, 4});
    auto p = empty();
    IoInfo info; info.semantic = IoSemantic::generic; info.location = 1;
    info.component_mask = 1;
    info.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
    info.type = {ScalarType::f32, 4};
    add(p, {{40, InstructionName::ald}, Opcode::load_io, {}, floating(), {}, {}, info});
    auto classified8 = run(p, f);
    if (const auto* legal = success(classified8, "Supplied format fact keeps the old path")) {
        const auto* use = reflected(*legal, 1);
        check(use && use->vertex_format == AttributeFormat{VertexFormat::f32, 4} &&
              !use->vertex_class.has_value(),
            "Fact path reflects vertex_format instead of a class");
        check(std::any_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(),
                [](const auto& d) { return d.key.kind == FactKind::vertex_attribute_format; }),
            "Fact path consumes the format as a dependency");
    }

    f.set(FactQuery<FactKind::vertex_attribute_format>{2}, AttributeFormat{VertexFormat::u32, 4});
    p = empty();
    IoInfo mismatch; mismatch.semantic = IoSemantic::generic; mismatch.location = 2;
    mismatch.component_mask = 1;
    mismatch.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
    mismatch.type = {ScalarType::f32, 4};
    add(p, {{40, InstructionName::ald}, Opcode::load_io, {}, floating(), {}, {}, mismatch});
    rejected(run(p, f), DiagnosticCode::unsupported_modifier,
        "Integer format into a floating load still fails explicitly");

    // Vertex-stage generic interpolation without a fact is unsupported.
    p = empty();
    IoInfo varying; varying.semantic = IoSemantic::generic; varying.location = 3;
    varying.component_mask = 1;
    varying.interpolation = {InterpolationMode::perspective, InterpolationLocation::center};
    varying.type = {ScalarType::f32, 4};
    varying.reciprocal_w = InputIndex{0};
    add(p, {{40, InstructionName::ipa}, Opcode::interpolate, {}, floating(), {word(0)}, {}, varying});
    rejected(run(p, f), DiagnosticCode::unsupported_capability,
        "Vertex generic interpolation needs a format fact");

    f.set(FactQuery<FactKind::vertex_attribute_format>{3}, AttributeFormat{VertexFormat::f32, 4});
    auto classified9 = run(p, f);
    if (const auto* legal = success(classified9, "Supplied fact covers vertex interpolation")) {
        const auto* use = reflected(*legal, 3);
        check(use && use->vertex_format.has_value(), "Interpolated fact path reflects the format");
    }
}

void merged_across_blocks() {
    Facts f;
    auto p = empty();
    p.blocks.resize(4);
    p.functions[0].blocks = {BlockId{0}, BlockId{1}, BlockId{2}, BlockId{3}};
    for (auto& block : p.blocks) block.function = FunctionId{0};
    const auto condition = ValueId{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({ValueType::predicate});
    p.functions[0].parameters.push_back(condition);
    p.blocks[0].terminator = Branch{condition, BlockId{1}, BlockId{2}};
    p.blocks[1].predecessors = {BlockId{0}};
    p.blocks[2].predecessors = {BlockId{0}};
    const auto left = add_to(p, BlockId{1}, [&] {
        IoInfo info; info.semantic = IoSemantic::generic; info.location = 9;
        info.component_mask = 1;
        info.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
        info.type = {ScalarType::u32, 4};
        return Operation{{40, InstructionName::ald}, Opcode::load_io, {}, {Domain::bits, ScalarType::u32}, {}, {}, info};
    }());
    const auto right = add_to(p, BlockId{2}, [&] {
        IoInfo info; info.semantic = IoSemantic::generic; info.location = 9;
        info.component_mask = 4;
        info.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
        info.type = {ScalarType::u32, 4};
        return Operation{{44, InstructionName::ald}, Opcode::load_io, {}, {Domain::bits, ScalarType::u32}, {}, {}, info};
    }());
    p.blocks[1].terminator = Jump{BlockId{3}};
    p.blocks[2].terminator = Jump{BlockId{3}};
    p.blocks[3].predecessors = {BlockId{1}, BlockId{2}};
    const auto merged = ValueId{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({ValueType::word32});
    p.blocks[3].phis = {{{48, {}}, merged, {{BlockId{1}, left}, {BlockId{2}, right}}}};
    auto use = Operation{{48, InstructionName::fmul}, Opcode::multiply, {}, floating(), {merged, word(0x3f000000)}, {}, {}};
    const auto product = ValueId{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({ValueType::word32});
    use.outputs = {product};
    use.effects = required_effects(use);
    p.blocks[3].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back(std::move(use));
    p.functions[0].results = {ValueType::word32};
    p.blocks[3].terminator = Return{{product}};
    auto classified10 = run(p, f);
    if (const auto* legal = success(classified10, "Phi-merged reads of one location classify together")) {
        const auto* use_reflection = reflected(*legal, 9);
        check(use_reflection && use_reflection->vertex_class == VertexAttributeClass::float_class &&
              use_reflection->mask == 5,
            "Both phi branches feed the shared float class and merged mask");
    }
}

// --- full pipeline: decode through emit_msl ---------------------------------

constexpr std::uint64_t pattern(std::string_view bits) {
    std::uint64_t value{};
    for (std::size_t index = 0; index < bits.size(); ++index)
        if (bits[index] == '1') value |= std::uint64_t{1} << (63 - index);
    return value;
}
std::uint64_t encoded_ald(unsigned destination, unsigned offset) {
    return pattern("1110111111011x") | destination | (255ull << 8) | (7ull << 16) |
        (std::uint64_t{offset} << 20);
}
std::uint64_t encoded_ast(unsigned source, unsigned offset, unsigned words = 1) {
    return pattern("1110111111110x") | source | (255ull << 8) | (7ull << 16) |
        (std::uint64_t{offset} << 20) | (std::uint64_t{words - 1} << 47);
}
std::uint64_t encoded_fmul32i(unsigned destination, unsigned source, std::uint32_t bits) {
    return pattern("00011110") | destination | (std::uint64_t{source} << 8) | (7ull << 16) |
        (std::uint64_t{bits} << 20);
}
std::uint64_t encoded_i2f(unsigned destination, unsigned source) {
    // D.F32 (format 2 at bit 8), src.U32 (format 2 at bit 10), register B at 20.
    return pattern("0101110010111") | destination | (2ull << 8) | (2ull << 10) | (7ull << 16) |
        (std::uint64_t{source} << 20);
}
std::uint64_t encoded_mov32i(unsigned destination, std::uint32_t bits) {
    return pattern("000000010000") | destination | (7ull << 16) | (std::uint64_t{bits} << 20);
}
StageImage stage_image(std::initializer_list<std::uint64_t> instructions) {
    StageImage image;
    image.stage = Stage::vertex;
    auto append = [&](std::uint64_t word) {
        for (unsigned byte = 0; byte < 8; ++byte)
            image.bytes.push_back(std::byte((word >> (byte * 8)) & 0xff));
    };
    for (auto instruction : instructions) {
        if (image.bytes.size() % 32 == 0) append(0);
        append(instruction);
    }
    if (image.bytes.size() % 32 == 0) append(0);
    append(pattern("111000110000") | (7ull << 16) | 15);
    return image;
}

void full_pipeline() {
    Facts f;
    f.set(FactQuery<FactKind::depth_convention>{}, DepthConvention::zero_to_one);
    // R4 reads attribute 0 and feeds a float multiply; R9 reads attribute 1 and
    // is only converted from its unsigned integer bits. Neither location has a
    // vertex_attribute_format fact.
    const StageImage image = stage_image({
        encoded_ald(4, 0x80),
        encoded_fmul32i(5, 4, 0x3f000000),
        encoded_mov32i(6, 0), encoded_mov32i(7, 0), encoded_mov32i(8, 0x3f800000),
        encoded_ast(5, 0x70, 4),
        encoded_ald(9, 0x90),
        encoded_i2f(10, 9),
        encoded_mov32i(11, 0), encoded_mov32i(12, 0), encoded_mov32i(13, 0x3f800000),
        encoded_ast(10, 0x80, 4)});
    FactReader reader(f);
    auto decoded = decode(image, reader, {});
    if (!success(decoded, "Vertex class fixture decodes")) return;
    auto depth = reader.require(FactQuery<FactKind::depth_convention>{}, Phase::emit_ir);
    check(std::holds_alternative<DepthConvention>(depth), "Vertex class fixture records its depth convention");
    auto emitted = emit_ir(std::get<DecodedProgram>(std::move(decoded)), reader, {});
    if (!success(emitted, "Vertex class fixture emits IR")) return;
    auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(emitted)), {});
    if (!success(ssa, "Vertex class fixture reaches SSA")) return;
    auto legal = legalize(std::get<SsaProgram>(std::move(ssa)), reader, profile());
    const auto* program = success(legal, "Vertex class fixture legalizes without format facts");
    if (!program) return;
    const auto* float_input = reflected(*program, 0);
    const auto* int_input = reflected(*program, 1);
    check(float_input && float_input->vertex_class == VertexAttributeClass::float_class &&
          float_input->type == IoType{ScalarType::f32, 4},
        "Pipeline float input reflects the float class");
    check(int_input && int_input->vertex_class == VertexAttributeClass::uint_class &&
          int_input->type == IoType{ScalarType::u32, 4},
        "Pipeline integer input reflects the uint class");
    check(std::none_of(program->ir.dependencies.begin(), program->ir.dependencies.end(),
            [](const auto& d) { return d.key.kind == FactKind::vertex_attribute_format; }),
        "Offline vertex package records no attribute format dependency");
    // The general structurizer's flag-dispatch boundary for straight-line
    // bodies belongs to another lane; like the emit IO tests, emission checks
    // run on the direct single-block region, which is the same structured
    // contract for an unconditional program.
    const auto& legalized = *program;
    StructuredProgram direct{legalized.ir, legalized.requirements, legalized.profile,
        {BlockRegion{BlockId{0}}}, {{FunctionId{0}, RegionId{0}}}};
    auto plan = plan_resources(direct, profile());
    if (!success(plan, "Vertex class fixture plans")) return;
    auto artifact = emit_msl(direct, std::get<ResourcePlan>(plan), {});
    const auto* generated = success(artifact, "Vertex class fixture emits MSL");
    if (!generated) return;
    const auto& source = generated->source;
    check(source.find("float4 generic0 [[attribute(0)]]") != std::string::npos,
        "Float input declares float4 at its attribute index");
    check(source.find("uint4 generic1 [[attribute(1)]]") != std::string::npos,
        "Integer input declares uint4 at its attribute index");
    check(source.find("= as_type<uint>(in.generic0.x);") != std::string::npos,
        "Float input load reinterprets the float component as a raw word");
    check(source.find("= in.generic1.x;") != std::string::npos,
        "Integer input load keeps the raw uint component");
}

void full_pipeline_mixed_rejected() {
    Facts f;
    f.set(FactQuery<FactKind::depth_convention>{}, DepthConvention::zero_to_one);
    // Attribute 0 feeds a float multiply and a raw integer conversion, which
    // no single vertex descriptor declaration can serve.
    const StageImage image = stage_image({
        encoded_ald(4, 0x80),
        encoded_fmul32i(5, 4, 0x3f000000),
        encoded_i2f(6, 4),
        encoded_mov32i(7, 0), encoded_mov32i(8, 0x3f800000),
        encoded_ast(5, 0x70, 4),
        encoded_ast(6, 0x80)});
    FactReader reader(f);
    auto decoded = decode(image, reader, {});
    if (!success(decoded, "Mixed fixture decodes")) return;
    auto depth = reader.require(FactQuery<FactKind::depth_convention>{}, Phase::emit_ir);
    check(std::holds_alternative<DepthConvention>(depth), "Mixed fixture records its depth convention");
    auto emitted = emit_ir(std::get<DecodedProgram>(std::move(decoded)), reader, {});
    if (!success(emitted, "Mixed fixture emits IR")) return;
    auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(emitted)), {});
    if (!success(ssa, "Mixed fixture reaches SSA")) return;
    rejected(legalize(std::get<SsaProgram>(std::move(ssa)), reader, profile()),
        DiagnosticCode::unsupported_capability, "Mixed pipeline fixture is a structured diagnostic");
}

} // namespace

int main() {
    float_input();
    integer_input();
    mixed_use_rejected();
    fragment_inputs_are_not_vertex_classified();
    analysis_budget();
    fact_path_preserved();
    merged_across_blocks();
    full_pipeline();
    full_pipeline_mixed_rejected();
    if (failures != 0) {
        std::cerr << failures << " vertex class checks failed\n";
        return 1;
    }
    std::cout << "shader.pipeline_vertex_class: all checks passed\n";
    return 0;
}
