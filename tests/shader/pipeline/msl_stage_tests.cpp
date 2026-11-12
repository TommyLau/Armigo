// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// These independently authored synthetic programs exercise MSL stage-emitter
// contracts.
#include "pipeline/phases.h"
#include "pipeline/msl/matrix_plan.h"
#include "vertex_builtins_fixture.h"
#include <armigo/shader/vertex_fetch.h>
#include <array>
#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
std::filesystem::path fixture_directory;
bool compile_metal{};
void check(bool value, const std::string& message) { if (!value) { std::cerr << "FAIL: " << message << '\n'; ++failures; } }
Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
Operand fp(float value) { return word(std::bit_cast<std::uint32_t>(value)); }
Operand predicate(bool value) { return LiteralBits{value ? 1u : 0u, ValueType::predicate}; }
NumericMode floating() { return {Domain::floating, ScalarType::f32, Rounding::nearest_even}; }
struct Fixture {
    StructuredProgram p;
    ResourcePlan plan;
    EmitOptions options;
    Fixture(Stage stage = Stage::fragment) {
        p.ir.stage = stage; p.ir.entry = FunctionId{0};
        p.ir.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
        p.ir.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
        p.regions = {BlockRegion{BlockId{0}}}; p.functions = {{FunctionId{0}, RegionId{0}}};
        p.profile = {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        plan.stage = stage; plan.profile = p.profile;
        if (stage == Stage::vertex) {
            p.ir.dependencies.push_back({{FactKind::depth_convention}, DepthConvention::negative_one_to_one, {OriginKind::explicit_option, "synthetic-depth"}});
            plan.dependencies = p.ir.dependencies;
            io(IoSemantic::position, Access::write, 15, {ScalarType::f32, 4});
        }
    }
    void io(IoSemantic semantic, Access access, unsigned mask, IoType type, unsigned location = 0, Interpolation interpolation = {}) {
        IoUse use; use.semantic = semantic; use.access = access; use.mask = mask; use.type = type; use.location = location; use.interpolation = interpolation;
        if (semantic == IoSemantic::generic && p.ir.stage == Stage::vertex && access == Access::read) use.vertex_format = AttributeFormat{VertexFormat::f32, type.components};
        p.requirements.io.push_back(use); plan.io.push_back(use);
        if (semantic == IoSemantic::instance_id && access == Access::read)
            plan.draw_support_buffer_index = 0;
    }
    ResourceId resource(ResourceUse use, MetalNamespace space, unsigned index, unsigned count = 1) {
        ResourceId id{static_cast<std::uint32_t>(p.requirements.uses.size())};
        p.requirements.uses.push_back(use); plan.resources.push_back({id, std::move(use), MetalRange{space, index, count}}); return id;
    }
    ResourceId constant_buffer(unsigned slot, unsigned window, unsigned index) {
        auto id = resource(BufferUse{ConstantBufferSource{static_cast<std::uint8_t>(slot)}, Access::read,
            {SizeKind::constant_buffer_window, window, 4, ConstantBufferPolicy::padded_zero_words_v2}},
            MetalNamespace::buffer, index);
        resource(BufferUse{SupportBufferSource{std::string(constant_buffer_support_schema), constant_buffer_support_version},
            Access::read, {SizeKind::accessed_extent, constant_buffer_support_bytes, 4}},
            MetalNamespace::buffer, 30);
        return id;
    }
    OperationId operation(Operation op) {
        op.effects = required_effects(op); op.source = {40 + static_cast<Pc>(p.ir.operations.size() * 8), InstructionName::mov};
        OperationId id{static_cast<std::uint32_t>(p.ir.operations.size())};
        p.ir.operations.push_back(std::move(op)); p.ir.blocks[0].operations.push_back(id); return id;
    }
    ValueId value(ValueType type = ValueType::word32, ValueKind kind = ValueKind::temporary) {
        ValueId id{static_cast<std::uint32_t>(p.ir.values.size())}; p.ir.values.push_back({type, kind}); return id;
    }
    void store(IoSemantic semantic, std::vector<Operand> data, IoType type = {ScalarType::f32, 4}, unsigned location = 0) {
        IoInfo info; info.semantic = semantic; info.type = type; info.location = location; info.component_mask = (1u << data.size()) - 1;
        for (unsigned i = 0; i < data.size(); ++i) info.data.push_back(InputIndex{static_cast<std::uint16_t>(i)});
        operation({{}, Opcode::store_io, {}, {}, std::move(data), {}, info});
    }
    void isolate_operations() {
        p.ir.blocks.clear(); p.ir.functions[0].blocks.clear();
        for (std::uint32_t i = 0; i < p.ir.operations.size(); ++i) {
            p.ir.functions[0].blocks.push_back(BlockId{i});
            BasicBlock block; block.function = FunctionId{0}; block.operations = {OperationId{i}};
            if (i) block.predecessors = {BlockId{i - 1}};
            block.terminator = i + 1 < p.ir.operations.size() ? Terminator{Jump{BlockId{i + 1}}} : Terminator{Return{}};
            p.ir.blocks.push_back(std::move(block));
        }
    }
    Result<MslArtifact> emit() { return emit_msl(p, plan, options); }
};
std::string quote(const std::string& value) {
    std::string result = "'"; for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c); return result + "'";
}
std::string accepted(Fixture& f, const std::string& label) {
    auto result = f.emit();
    if (const auto* d = std::get_if<Diagnostic>(&result)) { check(false, label + ": " + d->message); return {}; }
    auto artifact = std::get<MslArtifact>(result);
    check(artifact.reflection.dependencies == f.plan.dependencies && artifact.reflection.fragment_position == f.options.fragment_position, label + " reflection");
    if (!fixture_directory.empty()) {
        auto path = fixture_directory / (label + ".metal"); std::ofstream(path) << artifact.source;
        if (compile_metal) {
            const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" + quote((fixture_directory / "module-cache").string()) + " -c " + quote(path.string()) + " -o " + quote((fixture_directory / (label + ".air")).string());
            check(std::system(command.c_str()) == 0, label + " Metal compilation");
        }
    }
    return artifact.source;
}
// Owner decision: a read whose definition does not reach every structured path
// observes the value's
// zero-initialized declaration instead of being refused.
void zero_read(Fixture& f, const std::string& label) {
    const auto source = accepted(f, label);
    check(source.find(" = {};") != std::string::npos, label + ": values start at zero");
}
void rejected(Fixture f, const std::string& label, std::optional<DiagnosticCode> code = {}) {
    auto result = f.emit(); const auto* d = std::get_if<Diagnostic>(&result);
    check(d && d->phase == Phase::emit_msl && (!code || d->code == *code), label);
    if (d && code && d->code != *code) std::cerr << "  " << d->message << '\n';
}
void basic_tests() {
    Fixture vertex(Stage::vertex);
    vertex.store(IoSemantic::position, {fp(0), fp(0), fp(0.5f), fp(2)});
    auto source = accepted(vertex, "vertex_depth");
    check(source.find("armigo_vertex_main") != std::string::npos && source.find("out.position.z = armigo_depth_sum * 0.5f") != std::string::npos, "Vertex entry and depth adapter");
    check(source.find("[[point_size]]") != std::string::npos && source.find(" = 1.0f;") != std::string::npos &&
          source.find("function_constant(10)") == std::string::npos,
          "A vertex stage without its own point size writes the fixed 1.0 without a function constant");
    check(source.find("float4 position [[position]];") != std::string::npos, "A vertex position is not invariant by default");
    auto invariant = vertex; invariant.options.invariant_position = true;
    check(accepted(invariant, "vertex_invariant").find("float4 position [[position, invariant]];") != std::string::npos,
          "The invariant option marks the vertex position");
    auto missing = vertex; missing.p.ir.dependencies.clear(); missing.plan.dependencies.clear(); rejected(missing, "Missing depth fact", DiagnosticCode::missing_fact);
    vertex.p.ir.dependencies[0].value = DepthConvention::zero_to_one; vertex.plan.dependencies = vertex.p.ir.dependencies;
    source = accepted(vertex, "vertex_zero_depth"); check(source.find("armigo_depth_sum") == std::string::npos, "Zero-to-one depth remains unchanged");
    Fixture fragment;
    fragment.io(IoSemantic::color, Access::write, 15, {ScalarType::f32, 4}, 2);
    fragment.store(IoSemantic::color, {fp(1), fp(0), fp(0.25f), fp(1)}, {ScalarType::f32, 4}, 2);
    source = accepted(fragment, "fragment_color"); check(source.find("[[color(2)]]") != std::string::npos, "MRT binding");
    Fixture adaptive;
    adaptive.options.fragment_output_types_from_render_targets = true;
    adaptive.io(IoSemantic::color, Access::write, 3, {ScalarType::u32, 4}, 2);
    adaptive.store(IoSemantic::color, {word(0x3f800000), word(0x7fc12345)},
                   {ScalarType::u32, 4}, 2);
    source = accepted(adaptive, "adaptive_fragment_color");
    check(source.find("constant uint armigo_color2_type [[function_constant(2)]];") != std::string::npos &&
          source.find("float4 color2_f32 [[color(2), function_constant(armigo_color2_is_f32)]];") != std::string::npos &&
          source.find("uint4 color2_u32 [[color(2), function_constant(armigo_color2_is_u32)]];") != std::string::npos &&
          source.find("int4 color2_s32 [[color(2), function_constant(armigo_color2_is_s32)]];") != std::string::npos,
          "Adaptive fragment output declarations");
    check(source.find("uint4 armigo_color2_raw = uint4(0);") != std::string::npos &&
          source.find("armigo_color2_raw.x = 1065353216u;") != std::string::npos &&
          source.find("out.color2_f32 = as_type<float4>(armigo_color2_raw);") != std::string::npos &&
          source.find("out.color2_u32 = armigo_color2_raw;") != std::string::npos &&
          source.find("out.color2_s32 = as_type<int4>(armigo_color2_raw);") != std::string::npos,
          "Adaptive fragment output preserves raw words and initializes unwritten lanes");
    auto invalid_adaptive = fragment;
    invalid_adaptive.options.fragment_output_types_from_render_targets = true;
    rejected(invalid_adaptive, "Adaptive fragment output rejects typed backing",
             DiagnosticCode::conflicting_fact);
    auto malformed = fragment; malformed.p.regions = {SequenceRegion{{RegionId{0}}}}; rejected(malformed, "Cyclic AST");
    malformed = fragment; malformed.p.ir.limits.source_bytes = 80; rejected(malformed, "Source growth bound", DiagnosticCode::limit_exceeded);
    malformed = fragment; malformed.p.ir.stage = malformed.plan.stage = static_cast<Stage>(99); rejected(malformed,"Unknown stage cannot default to fragment",DiagnosticCode::unsupported_capability);
    malformed = fragment; malformed.plan.profile.identity = "different"; rejected(malformed, "Profile mismatch");
    malformed = fragment; malformed.options.entry_point = "main){}"; rejected(malformed, "Unsafe entry point");
    for (const auto* name : {"fragment", "vertex", "float", "return", "float4", "ArmigoOutput", "__internal"}) {
        malformed = fragment; malformed.options.entry_point = name; rejected(malformed, "Reserved entry point");
    }
    malformed = fragment; malformed.p.ir.operations[0].numeric.saturate = true; rejected(malformed, "Unsupported numeric modifier");
    fragment.operation({{}, Opcode::discard, {}, {}, {}, {}, {}}); accepted(fragment, "discard");

    // S2R SR_YDIRECTION: the consuming stage declares the armigo_y_negate
    // function constant and reads the sign-selected word like an entry-level
    // builtin; a stage without the read declares nothing.
    Fixture direction;
    direction.io(IoSemantic::y_direction, Access::read, 1, {ScalarType::u32, 1});
    IoInfo load_info;
    load_info.semantic = IoSemantic::y_direction;
    load_info.type = {ScalarType::u32, 1};
    load_info.component_mask = 1;
    const auto sign = direction.value();
    direction.operation({{}, Opcode::load_io, {}, NumericMode{Domain::unsigned_integer, ScalarType::u32},
                         {}, {Operand{sign}}, load_info});
    direction.io(IoSemantic::color, Access::write, 15, {ScalarType::u32, 4}, 2);
    direction.store(IoSemantic::color, {Operand{sign}, word(0), word(0), word(0)}, {ScalarType::u32, 4}, 2);
    source = accepted(direction, "fragment_y_direction");
    check(source.find("constant bool armigo_y_negate [[function_constant(9)]];") != std::string::npos &&
              source.find("constant uint builtin23 = armigo_y_negate ? 0xbf800000u : 0x3f800000u;") != std::string::npos &&
              source.find("= builtin23;") != std::string::npos,
          "Y-direction function constant and selection");
    check(source.find("color2") != std::string::npos, "Y-direction stage keeps its other outputs");
    Fixture no_direction;
    no_direction.io(IoSemantic::color, Access::write, 15, {ScalarType::f32, 4});
    no_direction.store(IoSemantic::color, {fp(1), fp(0), fp(0.25f), fp(1)});
    source = accepted(no_direction, "fragment_without_y_direction");
    check(source.find("armigo_y_negate") == std::string::npos && source.find("builtin23") == std::string::npos,
          "No y-direction declaration without the read");
}
void depth_bias_tests() {
    Fixture fragment;
    fragment.options.emulate_depth_bias = true;
    auto mandated = fragment;
    mandated.p.ir.api_mandated_early_fragment_tests = true;
    rejected(mandated, "API early tests cannot be replaced by late fragment depth-bias emulation",
             DiagnosticCode::unsupported_capability);
    fragment.operation({{}, Opcode::discard, {}, {}, {}, {}, {}});
    fragment.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}}},
        BlockRegion{BlockId{0}}, IfRegion{predicate(true), RegionId{3}, RegionId{4}},
        ReturnRegion{}, ReturnRegion{}};
    auto source = accepted(fragment, "depth_bias_early_returns");
    check(source.find("constant bool armigo_has_depth [[function_constant(8)]];") != std::string::npos &&
          source.find("float armigo_depth [[depth(any), function_constant(armigo_has_depth)]];") != std::string::npos &&
          source.find("constant float4& armigo_depth_bias [[buffer(30)]]") != std::string::npos &&
          source.find("float4 armigo_position [[position]]") != std::string::npos,
          "Depth bias adds a dedicated uniform, position input and conditional depth output");
    const auto derivatives = source.find("max(abs(dfdx(armigo_z)), abs(dfdy(armigo_z)))");
    const auto depth_write = source.find("out.armigo_depth = armigo_depth_bias.w != 0.0f ? clamp(armigo_z + armigo_depth_offset, 0.0f, 1.0f) : armigo_z;");
    check(derivatives != std::string::npos && depth_write != std::string::npos &&
          source.find("if (armigo_has_depth) {") < derivatives &&
          derivatives < source.find("discard_fragment()") && depth_write < source.find("if (true)") &&
          source.find("return out;", source.find("return out;") + 1) != std::string::npos,
          "Depth bias is evaluated before discard and divergence and survives every return");
    check(source.find("armigo_depth_offset += armigo_depth_bias.y * armigo_depth_slope;") <
              source.find("min(armigo_depth_offset, armigo_depth_bias.z)") &&
          source.find("else if (armigo_depth_bias.z < 0.0f) armigo_depth_offset = max(armigo_depth_offset, armigo_depth_bias.z);") != std::string::npos,
          "Signed one-sided clamp applies once to the combined offset");
    check(source.find("float armigo_depth_offset = armigo_depth_bias.x;") <
              source.find("if (armigo_depth_bias.y != 0.0f) {") &&
          source.find("if (armigo_depth_bias.y != 0.0f) {") < derivatives,
          "Constant-only fallback does not depend on reconstructed derivatives");
    auto artifact = std::get<MslArtifact>(fragment.emit());
    check(artifact.reflection.emulate_depth_bias && artifact.reflection.io.empty(),
          "Host depth-bias capability does not invent guest position or depth IO");

    Fixture adaptive;
    adaptive.options.emulate_depth_bias = true;
    adaptive.options.fragment_output_types_from_render_targets = true;
    adaptive.io(IoSemantic::color, Access::write, 15, {ScalarType::u32, 4});
    adaptive.store(IoSemantic::color, {word(0), word(0), word(0), word(0)},
                   {ScalarType::u32, 4});
    source = accepted(adaptive, "depth_bias_conditional_color_and_depth");
    check(source.find("float armigo_depth [[depth(any), function_constant(armigo_has_depth)]];") != std::string::npos &&
          source.find("function_constant(armigo_color0_is_f32)") != std::string::npos &&
          source.find("constant bool armigo_has_depth [[function_constant(8)]];") != std::string::npos &&
          source.find("if (armigo_has_depth) {") < source.find("out.armigo_depth ="),
          "Depth output specialization is independent of adaptive color outputs");

    Fixture position;
    position.options.emulate_depth_bias = true;
    position.options.fragment_position = FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    position.io(IoSemantic::fragment_coord, Access::read, 8, {ScalarType::f32, 4});
    source = accepted(position, "depth_bias_existing_position");
    check(source.find("const float armigo_z = in.position.z;") != std::string::npos &&
          source.find("armigo_position [[position]]") == std::string::npos,
          "Depth bias reuses existing position without duplicating its builtin");
    artifact = std::get<MslArtifact>(position.emit());
    check(artifact.reflection.io.size() == 1 && artifact.reflection.io[0].mask == 8,
          "Depth-bias input preserves the guest position component mask");

    Fixture position_color;
    position_color.options.emulate_depth_bias = true;
    position_color.io(IoSemantic::fragment_coord, Access::read, 4, {ScalarType::f32, 4});
    position_color.io(IoSemantic::color, Access::write, 15, {ScalarType::f32, 4});
    const auto z = position_color.value();
    IoInfo position_z;
    position_z.semantic = IoSemantic::fragment_coord;
    position_z.type = {ScalarType::f32, 4};
    position_z.component_mask = 4;
    position_color.operation({{}, Opcode::load_io, {}, {}, {}, {z}, position_z});
    position_color.store(IoSemantic::color, {z, fp(0), fp(0), fp(1)});
    source = accepted(position_color, "depth_bias_position_z_color");
    const auto export_depth = source.find("out.armigo_depth =");
    const auto update_position = source.find("in.position.z = out.armigo_depth;");
    const auto guest_read = source.find("value0 = as_type<uint>(in.position.z);");
    check(export_depth != std::string::npos && update_position != std::string::npos &&
          guest_read != std::string::npos && export_depth < update_position &&
          update_position < guest_read &&
          source.find("if (armigo_has_depth) {") < update_position,
          "Guest fragment Z reads observe the biased exported depth before guest code");

    Fixture collision;
    collision.options.emulate_depth_bias = true;
    collision.constant_buffer(0, 16, 0);
    rejected(collision, "Depth-bias binding collision is rejected", DiagnosticCode::conflicting_fact);
    std::get<MetalRange>(collision.plan.resources.back().location).first = 29;
    accepted(collision, "depth_bias_relocated_support");

    Fixture guest_depth;
    guest_depth.options.emulate_depth_bias = true;
    guest_depth.io(IoSemantic::fragment_depth, Access::write, 1, {ScalarType::f32, 1});
    guest_depth.store(IoSemantic::fragment_depth, {fp(0.5f)}, {ScalarType::f32, 1});
    rejected(guest_depth, "Guest depth export cannot be overwritten", DiagnosticCode::unsupported_capability);
    Fixture vertex(Stage::vertex);
    vertex.options.emulate_depth_bias = true;
    rejected(vertex, "Depth-bias ABI is fragment-only", DiagnosticCode::unsupported_capability);
}
void io_tests() {
    Fixture f;
    f.options.fragment_position = f.plan.fragment_position = FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    f.io(IoSemantic::generic, Access::read, 1, {ScalarType::f32, 4}, 3, {InterpolationMode::perspective, InterpolationLocation::center});
    f.io(IoSemantic::generic, Access::read, 2, {ScalarType::f32, 4}, 3, {InterpolationMode::linear, InterpolationLocation::centroid});
    f.io(IoSemantic::fragment_coord, Access::read, 8, {ScalarType::f32, 4});
    f.io(IoSemantic::color, Access::write, 15, {ScalarType::f32, 4});
    auto w = f.value(), color = f.value();
    IoInfo position; position.semantic = IoSemantic::fragment_coord; position.type = {ScalarType::f32, 4}; position.component_mask = 8;
    f.operation({{}, Opcode::load_io, {}, {}, {}, {w}, position});
    IoInfo varying; varying.semantic = IoSemantic::generic; varying.location = 3; varying.type = {ScalarType::f32, 4}; varying.reciprocal_w = InputIndex{0};
    f.operation({{}, Opcode::interpolate, {}, floating(), {w}, {color}, varying});
    f.store(IoSemantic::color, {color, fp(0), fp(0), fp(1)});
    auto source = accepted(f, "perspective_components");
    check(source.find("center_perspective") != std::string::npos && source.find("centroid_no_perspective") != std::string::npos &&
          source.find("armigo_multiply_rne(as_type<uint>(in.generic3_0), interpolation_factor1)") != std::string::npos, "Perspective factor preserved without cancellation");
    f.options.fragment_position = f.plan.fragment_position = FragmentPositionConvention::unspecified;
    rejected(f, "Position convention is mandatory", DiagnosticCode::missing_fact);
    Fixture v(Stage::vertex); v.io(IoSemantic::generic, Access::read, 7, {ScalarType::f32, 3}, 5);
    v.io(IoSemantic::generic, Access::write, 3, {ScalarType::f32, 4}, 3);
    v.store(IoSemantic::position, {fp(0),fp(0),fp(0),fp(1)});
    v.store(IoSemantic::generic, {fp(1),fp(2)}, {ScalarType::f32,4}, 3);
    accepted(v, "vertex_attributes");
    // Fact-less vertex inputs declare their code-derived class; the load reads
    // the attribute vector component and converts it to the raw IR word.
    Fixture classes(Stage::vertex);
    classes.store(IoSemantic::position, {fp(0), fp(0), fp(0), fp(1)});
    auto io_use = [&](VertexAttributeClass cls, unsigned location) {
        IoUse use; use.semantic = IoSemantic::generic; use.access = Access::read; use.mask = 1;
        use.location = static_cast<std::uint16_t>(location);
        use.type = {vertex_class_scalar(cls), 4}; use.vertex_class = cls;
        classes.p.requirements.io.push_back(use); classes.plan.io.push_back(use);
    };
    io_use(VertexAttributeClass::float_class, 0);
    io_use(VertexAttributeClass::uint_class, 1);
    io_use(VertexAttributeClass::sint_class, 2);
    classes.io(IoSemantic::generic, Access::write, 1, {ScalarType::f32, 4}, 4);
    auto as_float = classes.value(), as_uint = classes.value(), as_sint = classes.value();
    IoInfo input_info; input_info.semantic = IoSemantic::generic; input_info.component_mask = 1;
    input_info.type = {ScalarType::f32, 4}; input_info.location = 0;
    classes.operation({{}, Opcode::load_io, {}, floating(), {}, {as_float}, input_info});
    input_info.type = {ScalarType::u32, 4}; input_info.location = 1;
    classes.operation({{}, Opcode::load_io, {}, {}, {}, {as_uint}, input_info});
    input_info.type = {ScalarType::s32, 4}; input_info.location = 2;
    classes.operation({{}, Opcode::load_io, {}, {}, {}, {as_sint}, input_info});
    classes.store(IoSemantic::generic, {as_float}, {ScalarType::f32, 4}, 4);
    auto class_source = accepted(classes, "vertex_class_declarations");
    check(class_source.find("float4 generic0 [[attribute(0)]]") != std::string::npos &&
          class_source.find("uint4 generic1 [[attribute(1)]]") != std::string::npos &&
          class_source.find("int4 generic2 [[attribute(2)]]") != std::string::npos,
        "Class-based vertex inputs declare float4/uint4/int4 at their attribute index");
    check(class_source.find("value0 = as_type<uint>(in.generic0.x);") != std::string::npos &&
          class_source.find("value1 = in.generic1.x;") != std::string::npos &&
          class_source.find("value2 = as_type<uint>(in.generic2.x);") != std::string::npos,
        "Vertex input loads apply the class bit interpretation");
    auto contradiction = classes;
    contradiction.plan.io[0].vertex_class = contradiction.p.requirements.io[0].vertex_class =
        VertexAttributeClass::uint_class;
    rejected(contradiction, "Class contradicting its IO type is rejected");
    // Shader-side vertex fetch: the same inputs are filled from the fetch
    // table at entry, with no stage_in and no attribute declarations.
    auto fetching = classes;
    fetching.plan.vertex_fetch_buffer_index = 4;
    auto fetch_source = accepted(fetching, "vertex_fetch_declarations");
    check(fetch_source.find("[[attribute(") == std::string::npos &&
          fetch_source.find("[[stage_in]]") == std::string::npos &&
          fetch_source.find("constant ArmigoVertexFetch& armigo_vertex_fetch [[buffer(4)]]") != std::string::npos &&
          fetch_source.find("uint armigo_fetch_base_instance [[base_instance]]") != std::string::npos,
        "A fetching vertex stage takes the fetch table instead of vertex attributes");
    check(fetch_source.find("in.generic0 = armigo_vertex_fetch_float(armigo_vertex_fetch, 0u, armigo_fetch_vertex_id, "
                            "armigo_fetch_instance_id, armigo_fetch_base_instance);") != std::string::npos &&
          fetch_source.find("in.generic1 = armigo_vertex_fetch_uint(armigo_vertex_fetch, 1u,") != std::string::npos &&
          fetch_source.find("in.generic2 = armigo_vertex_fetch_int(armigo_vertex_fetch, 2u,") != std::string::npos &&
          fetch_source.find("value1 = in.generic1.x;") != std::string::npos,
        "A fetching vertex stage fills each input from the table by its class");
    // A stage with only float inputs leaves the integer fetch functions
    // unused, and the package build compiles with warnings as errors.
    if (!fixture_directory.empty() && compile_metal) {
        const auto path = fixture_directory / "vertex_fetch_float_only.metal";
        std::ofstream(path) << "#include <metal_stdlib>\nusing namespace metal;\n" << armigo::shader::vertex_fetch_msl()
                            << "\nvertex float4 armigo_vertex_main(uint vertex_id [[vertex_id]], "
                               "uint instance_id [[instance_id]], constant ArmigoVertexFetch& fetch [[buffer(0)]]) {\n"
                               "    return armigo_vertex_fetch_float(fetch, 0u, vertex_id, instance_id, 0u);\n}\n";
        const auto command = "xcrun -sdk macosx metal -std=metal4.0 -Werror -Wno-unused-variable "
                             "-Wno-sometimes-uninitialized -fmodules-cache-path=" +
            quote((fixture_directory / "module-cache").string()) + " -c " + quote(path.string()) + " -o " +
            quote((fixture_directory / "vertex_fetch_float_only.air").string());
        check(std::system(command.c_str()) == 0, "Float-only vertex fetch compiles with warnings as errors");
    }
    auto overlapping = fetching;
    overlapping.io(IoSemantic::instance_id, Access::read, 1, {ScalarType::u32, 1});  // draw support at 0
    overlapping.plan.vertex_fetch_buffer_index = 0;
    rejected(overlapping, "A fetch table overlapping another buffer is rejected", DiagnosticCode::conflicting_fact);
    auto described = fetching;
    described.io(IoSemantic::generic, Access::read, 1, {ScalarType::f32, 4}, 6);
    rejected(described, "Vertex fetch with a pipeline-state format input is rejected", DiagnosticCode::conflicting_fact);
}
void memory_tests() {
    Fixture f;
    auto buffer = f.constant_buffer(5, 64, 7);
    auto result = f.value(); MemoryInfo memory; memory.constant_buffer_slot = 5; memory.byte_offset = 5; memory.element_bytes = 2; memory.resource = buffer;
    f.operation({{}, Opcode::load, {}, {}, {}, {result}, memory});
    auto source = accepted(f, "constant_bytes");
    check(source.find("constant uint* resource0 [[buffer(7)]]") != std::string::npos &&
          source.find("constant uint* resource1 [[buffer(30)]]") != std::string::npos &&
          source.find("resource1[5u]") == std::string::npos && source.find("65535u") != std::string::npos,
          "Padded constant-buffer addressing and planned bindings");
    auto bad = f; std::get<MemoryInfo>(bad.p.ir.operations[0].info).resource.reset(); rejected(bad, "Unresolved memory", DiagnosticCode::unresolved_resource);
    bad = f; std::get<MemoryInfo>(bad.p.ir.operations[0].info).byte_offset = 63;
    source = accepted(bad, "Runtime guarded buffer load");
    check(source.find("min(63ul, 64ul)") != std::string::npos &&
        source.find("armigo_cbuf_word") == std::string::npos, "Window boundary uses a direct padded load");
    bad = f; bad.plan.resources.push_back(bad.plan.resources[0]); rejected(bad, "Duplicate binding");
    memory.byte_offset = 0; memory.dynamic_byte_offset = InputIndex{0};
    f.p.ir.operations[0].info = memory; f.p.ir.operations[0].inputs = {word(3)}; accepted(f, "constant_dynamic");
    f.p.ir.operations[0].numeric = {Domain::signed_integer, ScalarType::s16}; accepted(f, "constant_signed");
}
Fixture texture_fixture(TextureMode mode, TextureShape shape = {}, bool compare = false) {
    Fixture f;
    TextureDescriptor descriptor; descriptor.format_namespace = "synthetic-format-v1"; descriptor.shape = shape; descriptor.comparison = compare; descriptor.normalized_coordinates = true;
    descriptor.format = compare ? ComponentFormat::depth : ComponentFormat::floating;
    TextureUse texture; texture.key = {2,16}; texture.descriptor = descriptor; texture.modes = {mode};
    auto resource = f.resource(texture, MetalNamespace::texture, 4);
    const bool query = mode >= TextureMode::query_lod, fetch = mode == TextureMode::fetch, lod = mode == TextureMode::query_lod;
    TextureInfo info; info.texture = texture.key; info.descriptor = descriptor; info.resource = resource; info.mode = mode;
    std::vector<Operand> inputs;
    auto add = [&](Operand value) { InputIndex i{static_cast<std::uint16_t>(inputs.size())}; inputs.push_back(value); return i; };
    if ((!query && !fetch) || lod) {
        auto sampler = f.resource(SamplerUse{texture.key, compare, true, 1}, MetalNamespace::sampler, 6);
        info.sampler_resource = sampler; info.sampler = texture.key;
    }
    unsigned dimensions = shape.dimension == TextureDimension::d1 || shape.dimension == TextureDimension::buffer ? 1 : shape.dimension == TextureDimension::d2 ? 2 : 3;
    if (!query || lod) for (unsigned i = 0; i < dimensions; ++i) info.coordinates.push_back(add(fetch ? word(i) : fp(0.25f)));
    if (shape.array && (!query || lod)) info.array_layer = add(word(1));
    if (compare && !query && !fetch) info.compare = add(fp(0.5f));
    if (mode == TextureMode::sample_level || mode == TextureMode::sample_bias) info.level_or_bias = add(fp(1));
    if ((fetch || mode == TextureMode::query_size) && !shape.multisample && shape.dimension != TextureDimension::buffer) info.level_or_bias = add(word(0));
    if (fetch && shape.multisample) info.sample_index = add(word(0));
    if (mode == TextureMode::sample_gradients) {
        for (unsigned i = 0; i < dimensions; ++i) info.gradients_x.push_back(add(fp(0.125f)));
        for (unsigned i = 0; i < dimensions; ++i) info.gradients_y.push_back(add(fp(0.0625f)));
    }
    info.component_mask = query || (compare && mode != TextureMode::gather) ? 1 : 15;
    std::vector<Operand> outputs;
    for (int i = 0; i < std::popcount(info.component_mask); ++i) outputs.push_back(f.value());
    auto numeric = query && !lod ? NumericMode{Domain::unsigned_integer, ScalarType::u32} : floating();
    f.operation({{}, query ? Opcode::texture_query : Opcode::texture_sample, {}, numeric, std::move(inputs), std::move(outputs), info});
    return f;
}
void argument_buffer_tests() {
    auto f = texture_fixture(TextureMode::sample_level);
    for (auto& use : f.p.requirements.uses) {
        if (auto* texture = std::get_if<TextureUse>(&use)) { texture->array_length = 17; texture->dynamically_indexed = true; }
        else std::get<SamplerUse>(use).array_length = 17;
    }
    auto& op = f.p.ir.operations[0];
    std::get<TextureInfo>(op.info).array_index = InputIndex{static_cast<std::uint16_t>(op.inputs.size())};
    op.inputs.push_back(word(16));
    auto depth = std::get<TextureUse>(f.p.requirements.uses[0]);
    depth.array_length = 1; depth.key = {2, 32};
    depth.descriptor.format = ComponentFormat::depth; depth.descriptor.comparison = true;
    f.p.requirements.uses.push_back(depth);
    auto storage = depth;
    storage.key = {2, 48}; storage.kind = ResourceTextureKind::storage_image;
    storage.descriptor.format = ComponentFormat::unsigned_integer; storage.descriptor.scalar = ScalarType::u32;
    storage.descriptor.comparison = false; storage.access = Access::read_write;
    storage.modes = {TextureMode::fetch};
    f.p.requirements.uses.push_back(storage);
    const auto result = plan_resources(f.p, f.p.profile);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        check(false, "Argument-buffer fixture planning: " + diagnostic->message); return;
    }
    f.plan = std::get<ResourcePlan>(result);
    const auto source = accepted(f, "argument_buffer_arrays_and_types");
    check(source.find("struct ArmigoStageResources {") != std::string::npos &&
        source.find("const device ArmigoStageResources& resources [[buffer(0)]]") != std::string::npos &&
        source.find("[[id(") == std::string::npos && source.find("[[sampler(") == std::string::npos &&
        source.find("[[texture(") == std::string::npos, "Overflow stage uses Metal direct-layout argument-buffer ABI");
    check(source.find("array<texture2d<float>, 17> resource0;") != std::string::npos &&
        source.find("array<sampler, 17> resource1;") != std::string::npos &&
        source.find("depth2d<float> resource2;") != std::string::npos &&
        source.find("texture2d<uint, access::read_write> resource3;") != std::string::npos &&
        source.find("resources.resource0[16u].sample(resources.resource1[16u]") != std::string::npos,
        "Typed members retain arrays, depth, scalar/access types, and indexed sample operands");
    check(source.find("resource0;") < source.find("resource2;") &&
        source.find("resource3;") < source.find("resource1;"), "Declarations follow byte offsets, not ResourceId order");
    auto bad = f;
    bad.plan.argument_buffer.reset(); rejected(bad, "Member without container is rejected");
    bad = f; bad.options.entry_point = "ArmigoStageResources";
    rejected(bad, "Entry cannot shadow the stage resource type");
    bad = f; bad.plan.argument_buffer->abi_version = 2; rejected(bad, "Unknown argument-buffer ABI is rejected");
    bad = f; bad.plan.argument_buffer->alignment = 16; rejected(bad, "Argument-buffer alignment is exact");
    bad = f; bad.plan.argument_buffer->byte_extent -= 8; rejected(bad, "Truncated argument-buffer extent is rejected");
    bad = f; bad.plan.argument_buffer->byte_extent += 8; rejected(bad, "Trailing argument-buffer padding is rejected");
    bad = f; bad.plan.argument_buffer->buffer_index = 31; rejected(bad, "Argument-buffer index exceeds profile");
    bad = f; std::get<ArgumentBufferMember>(bad.plan.resources[0].location).stride = 16;
    rejected(bad, "Wrong member stride is rejected");
    bad = f; std::get<ArgumentBufferMember>(bad.plan.resources[0].location).count = 16;
    rejected(bad, "Wrong member count is rejected");
    bad = f; std::get<ArgumentBufferMember>(bad.plan.resources[0].location).byte_offset = 1;
    rejected(bad, "Misaligned member is rejected");
    bad = f; std::get<ArgumentBufferMember>(bad.plan.resources[1].location).byte_offset = 0;
    rejected(bad, "Overlapping members are rejected");
    bad = f; std::get<ArgumentBufferMember>(bad.plan.resources[0].location).byte_offset = UINT32_MAX - 7u;
    rejected(bad, "Member offset overflow is rejected");
    bad = f; bad.plan.resources[2].location = MetalRange{MetalNamespace::texture, 0, 1};
    rejected(bad, "Mixed direct/indirect textures are rejected");
    bad = f; bad.plan.argument_buffer->buffer_index = 30; bad.options.emulate_depth_bias = true;
    rejected(bad, "Container colliding with depth-bias binding is rejected", DiagnosticCode::conflicting_fact);
    bad.plan.argument_buffer->buffer_index = 29;
    accepted(bad, "argument_buffer_relocated_depth_bias");
    auto buffers = f;
    buffers.resource(BufferUse{StorageBufferSource{{2, 64}}, Access::read, {SizeKind::full_binding, 16, 4}},
        MetalNamespace::buffer, 0);
    rejected(buffers, "Container colliding with direct buffer is rejected");
    std::get<MetalRange>(buffers.plan.resources.back().location).first = 1;
    accepted(buffers, "argument_buffer_with_direct_storage");
    buffers.plan.resources.back().location = ArgumentBufferMember{0, 1, 8};
    rejected(buffers, "Ordinary buffer tagged as argument member is rejected");
    auto direct = texture_fixture(TextureMode::sample_level);
    std::get<MetalRange>(direct.plan.resources[0].location).first = UINT32_MAX;
    rejected(direct, "Direct binding uint32 overflow is rejected");
}

void texture_tests() {
    for (auto mode : {TextureMode::sample_implicit, TextureMode::sample_level, TextureMode::sample_bias, TextureMode::sample_gradients,
                     TextureMode::gather, TextureMode::fetch, TextureMode::query_lod, TextureMode::query_size, TextureMode::query_levels}) {
        auto f = texture_fixture(mode); auto label = "texture_mode_" + std::to_string(static_cast<unsigned>(mode));
        auto source = accepted(f, label); check(source.find("[[texture(4)]]") != std::string::npos, label + " binding");
        const auto artifact = emit_msl(f.p, f.plan, f.options);
        const auto* emitted = std::get_if<MslArtifact>(&artifact);
        const bool supported_border = mode <= TextureMode::sample_gradients;
        check(emitted && bool(emitted->reflection.custom_border) == supported_border,
              "Only ordinary 2D samples declare the versioned custom-border binding");
        if (supported_border && emitted && emitted->reflection.custom_border) {
            const auto& support = *emitted->reflection.custom_border;
            check(support.pairs.size() == 1 && support.pairs[0].texture == ResourceId{0} &&
                  support.pairs[0].sampler == ResourceId{1},
                  "Custom border reflection preserves the texture/sampler pair");
            const auto original = source.find("resource0.sample(");
            const auto end = source.find(";", original);
            const auto suffix = source.substr(original + std::string("resource0").size(),
                end - original - std::string("resource0").size());
            check(source.find("armigo_border[0].coverage" + suffix + ".x") != std::string::npos,
                  "Coverage repeats exactly the original LOD/bias/gradient operands");
            check(source.find("armigo_border[0].enabled != 0u") != std::string::npos,
                  "Native fixed-border samples bypass correction");
            check(source.find("function_constant(12)") == std::string::npos &&
                  source.find("if (armigo_border[0].enabled != 0u)") != std::string::npos,
                  "A stage that may be built unspecialized reads its border records unconditionally");
            auto typed = texture_fixture(mode);
            typed.options.fragment_output_types_from_render_targets = true;
            const auto typed_source = accepted(typed, label + "_typed_outputs");
            check(typed_source.find("armigo_border_records_ [[function_constant(12)]]") != std::string::npos &&
                  typed_source.find("? armigo_border_records_ : true") != std::string::npos &&
                  typed_source.find("if (armigo_border_records && armigo_border[0].enabled != 0u)") != std::string::npos,
                  "A specialized fragment reads its border records unless draw time disables them");
        }
        if (mode == TextureMode::query_lod) {
            check(source.find("calculate_clamped_lod") != std::string::npos && source.find("calculate_unclamped_lod") == std::string::npos,
                  "Host LOD component zero is clamped");
            std::get<TextureInfo>(f.p.ir.operations[0].info).query_component = 1;
            source = accepted(f, "texture_lod_unclamped");
            check(source.find("calculate_unclamped_lod") != std::string::npos && source.find("calculate_clamped_lod") == std::string::npos,
                  "Host LOD component one is unclamped");
        }
    }
    for (auto mode : {TextureMode::sample_implicit, TextureMode::sample_level, TextureMode::sample_bias, TextureMode::sample_gradients, TextureMode::gather}) {
        auto f = texture_fixture(mode, {}, true);
        const auto source = accepted(f, "compare_" + std::to_string(static_cast<unsigned>(mode)));
        check(source.find("ArmigoCustomBorder") == std::string::npos,
              "Comparison samplers have no custom-border capability");
    }
    for (auto shape : {TextureShape{TextureDimension::d1}, TextureShape{TextureDimension::d2,true}, TextureShape{TextureDimension::d3}, TextureShape{TextureDimension::cube}, TextureShape{TextureDimension::cube,true}}) {
        auto f = texture_fixture(shape.dimension == TextureDimension::d1 ? TextureMode::sample_implicit : TextureMode::sample_level, shape);
        const auto source = accepted(f, "shape_" + std::to_string(static_cast<unsigned>(shape.dimension)) + (shape.array ? "_array" : ""));
        check(source.find("ArmigoCustomBorder") == std::string::npos,
              "Other dimensions and array textures have no custom-border capability");
    }
    auto multisample = texture_fixture(TextureMode::fetch, {TextureDimension::d2,false,true}); accepted(multisample, "multisample_read");
    multisample = texture_fixture(TextureMode::query_samples, {TextureDimension::d2,true,true}); accepted(multisample, "multisample_count");
    {
        // A per-texel gather offset set lowers as four gathers: result
        // component k takes quadrant k of the gather issued at offsets[k].
        auto gather = texture_fixture(TextureMode::gather);
        auto& info = std::get<TextureInfo>(gather.p.ir.operations[0].info);
        const std::array<std::array<std::int32_t, 2>, 4> offsets{{{ -8, 7 }, { 7, -8 }, { -8, -8 }, { 7, 7 }}};
        for (const auto& offset : offsets) for (auto value : offset) {
            info.offsets.push_back(InputIndex{static_cast<std::uint16_t>(gather.p.ir.operations[0].inputs.size())});
            gather.p.ir.operations[0].inputs.push_back(word(static_cast<std::uint32_t>(value)));
        }
        const auto source = accepted(gather, "per_texel_gather_offsets");
        check(source.find(".gather(") != std::string::npos, "Per-texel gather lowers to gathers");
        // Each result texel is the (i0, j0) texel of its own footprint, which
        // Metal's gather returns as w.
        std::size_t texels = 0;
        for (auto at = source.find("component::x).w"); at != std::string::npos; at = source.find("component::x).w", at + 1)) ++texels;
        check(texels == 4, "Per-texel gather takes the base texel of each offset footprint");
        for (const auto& offset : offsets)
            check(source.find("int2(" + std::to_string(offset[0]) + ", " + std::to_string(offset[1]) + ")") != std::string::npos,
                "Per-texel gather keeps its own offsets");
        auto ranged = gather;
        auto& range_info = std::get<TextureInfo>(ranged.p.ir.operations[0].info);
        auto& last = ranged.p.ir.operations[0].inputs[range_info.offsets.back().value];
        last = word(8);
        rejected(ranged, "Per-texel gather offsets stay inside the Metal range", DiagnosticCode::unsupported_modifier);
        auto shadow = texture_fixture(TextureMode::gather, {}, true);
        auto& shadow_info = std::get<TextureInfo>(shadow.p.ir.operations[0].info);
        for (unsigned n = 0; n < 8; ++n) {
            shadow_info.offsets.push_back(InputIndex{static_cast<std::uint16_t>(shadow.p.ir.operations[0].inputs.size())});
            shadow.p.ir.operations[0].inputs.push_back(word(0));
        }
        rejected(shadow, "Shadow gathers keep one offset", DiagnosticCode::unsupported_capability);
    }
    for (auto shape : {TextureShape{TextureDimension::d2,true},TextureShape{TextureDimension::cube},TextureShape{TextureDimension::cube,true}}) {
        for (bool compare : {false,true}) for (auto mode : {TextureMode::sample_implicit,TextureMode::sample_bias,TextureMode::sample_gradients,TextureMode::gather,TextureMode::query_lod}) {
            auto f = texture_fixture(mode,shape,compare);
            accepted(f,"matrix_" + std::to_string(static_cast<unsigned>(shape.dimension)) + (shape.array ? "a" : "s") +
                       (compare ? "c" : "r") + std::to_string(static_cast<unsigned>(mode)));
        }
    }
    auto bad = texture_fixture(TextureMode::query_lod); bad.p.profile.supports_query_lod = bad.plan.profile.supports_query_lod = false; rejected(bad, "LOD profile capability", DiagnosticCode::unsupported_capability);
    bad = texture_fixture(TextureMode::query_lod); std::get<TextureInfo>(bad.p.ir.operations[0].info).query_component = 2;
    rejected(bad, "LOD component outside the pair");
    bad = texture_fixture(TextureMode::sample_implicit); std::get<TextureInfo>(bad.p.ir.operations[0].info).resource.reset(); rejected(bad, "Unresolved texture", DiagnosticCode::unresolved_resource);
    bad = texture_fixture(TextureMode::sample_level); std::get<TextureInfo>(bad.p.ir.operations[0].info).level_or_bias.reset(); rejected(bad, "Required LOD cannot disappear");
    bad = texture_fixture(TextureMode::fetch); bad.p.ir.operations[0].numeric.scalar = ScalarType::f64; rejected(bad, "No FP64 substitution");
}

void shadow_pipeline_tests() {
    struct Facts final : FactSource {
        bool array{};
        Knowledge<FactValue> lookup(FactKey key) const override {
            if (key == FactKey{FactKind::texture_handle_cbuf})
                return Known<FactValue>{std::uint32_t{2}, {OriginKind::explicit_option, "synthetic-shadow-slot"}};
            if (key == fact_query::texture({2, 32}).key()) {
                TextureDescriptor descriptor{{TextureDimension::d2, array, false}, ScalarType::f32,
                    ComponentFormat::depth, true, true, 0, "synthetic-shadow-v1"};
                return Known<FactValue>{descriptor, {OriginKind::explicit_option, "synthetic-shadow-descriptor"}};
            }
            return Unknown{"Synthetic shadow fact is absent"};
        }
    } facts;
    auto pattern = [](std::string_view bits) {
        std::uint64_t word{};
        for (std::size_t i = 0; i < bits.size(); ++i)
            if (bits[i] == '1') word |= std::uint64_t{1} << (63 - i);
        return word;
    };
    auto field = [](unsigned start, std::uint64_t value) { return value << start; };
    const auto operands = field(0, 80) | field(8, 20) | field(16, 7) | field(20, 40) | field(36, 8);
    struct Row { std::string label; std::uint64_t word; bool array{}, gather{}; unsigned outputs; };
    std::vector<Row> rows;
    for (unsigned target : {4, 5, 6, 9})
        rows.push_back({"shadow_texs_" + std::to_string(target), pattern("1101100xxxxxxx") | operands |
            field(28, 255) | field(50, 4) | field(53, target), target == 9, false, 2});
    for (unsigned lod : {0, 1, 2, 3})
        rows.push_back({"shadow_tex_" + std::to_string(lod), pattern("11000xxxxxxxxx") | operands |
            field(28, 2) | field(31, 15) | field(50, 1) | field(51, 7) | field(55, lod), false, false, 4});
    rows.push_back({"shadow_tld4", pattern("110010xxxxxxxx") | operands |
        field(28, 2) | field(31, 15) | field(50, 1) | field(51, 7), false, true, 4});
    for (const auto& row : rows) {
        try {
            StageImage image; image.stage = Stage::fragment;
            auto append = [&](std::uint64_t word) {
                for (unsigned i = 0; i < 8; ++i) image.bytes.push_back(std::byte((word >> (i * 8)) & 255));
            };
            auto instruction = [&](std::uint64_t word) {
                if (image.bytes.size() % 32 == 0) append(0);
                append(word);
            };
            for (unsigned reg : {20, 21, 40, 41})
                instruction(pattern("000000010000") | field(0, reg) | field(16, 7) |
                    field(20, std::bit_cast<std::uint32_t>(0.5f)));
            instruction(row.word);
            instruction(pattern("111000110000") | field(16, 7) | 15);
            if (image.bytes.size() % 32 == 0) append(0);
            facts.array = row.array;
            FactReader reader(facts);
            auto take = []<class T>(Result<T> result) -> T {
                if (const auto* d = std::get_if<Diagnostic>(&result)) throw std::runtime_error(d->message);
                return std::get<T>(std::move(result));
            };
            Fixture f;
            auto decoded = take(decode(image, reader, {}));
            auto registers = take(emit_ir(decoded, reader, {}));
            auto ssa = take(construct_ssa(std::move(registers), {}));
            auto legal = take(legalize(std::move(ssa), reader, f.p.profile));
            f.p = take(structure(std::move(legal), {}));
            f.plan = take(plan_resources(f.p, f.p.profile));
            const auto source = accepted(f, row.label);
            check(source.find(row.array ? "depth2d_array<float>" : "depth2d<float>") != std::string::npos,
                row.label + " uses a depth texture");
            check(source.find(row.gather ? ".gather_compare(" : ".sample_compare(") != std::string::npos,
                row.label + " preserves comparison sampling");
            unsigned samples = 0;
            for (std::size_t i = 0; i < f.p.ir.operations.size(); ++i) {
                const auto& op = f.p.ir.operations[i];
                if (op.opcode != Opcode::texture_sample) continue;
                ++samples;
                const auto& info = std::get<TextureInfo>(op.info);
                check(info.compare && op.outputs.size() == row.outputs, row.label + " preserves selected outputs");
                const auto temp = "texture_result" + std::to_string(i);
                for (std::size_t lane = 0; lane < op.outputs.size(); ++lane) {
                    const auto value = std::get<ValueId>(op.outputs[lane]);
                    const auto result = temp + (row.gather ? "." + std::string(1, "xyzw"[lane]) : "");
                    check(source.find("value" + std::to_string(value.value) + " = as_type<uint>(" + result + ");") != std::string::npos,
                        row.label + " preserves result lane " + std::to_string(lane));
                }
            }
            check(samples == 1, row.label + " retains exactly one sample");
        } catch (const std::exception& error) { check(false, row.label + ": " + error.what()); }
    }
}
void vertex_builtin_tests() {
    for (unsigned ids : {1u, 2u, 3u}) {
        auto result = armigo::shader::test::vertex_builtins::guest_vertex(ids);
        const auto label = "decoded_vertex_builtins_" + std::to_string(ids);
        const auto* artifact = std::get_if<MslArtifact>(&result);
        check(artifact != nullptr, label + " translates from guest instructions");
        if (!artifact) {
            std::cerr << "  " << std::get<Diagnostic>(result).message << '\n';
            continue;
        }
        const auto& source = artifact->source;
        check((source.find("[[vertex_id]]") != std::string::npos) == bool(ids & 1),
            label + " emits only requested vertex ID");
        check(source.find("[[base_instance]]") == std::string::npos,
            label + " never subtracts the per-call native base");
        check(artifact->reflection.draw_support_buffer_index.has_value() == bool(ids & 2),
            label + " reflects draw support only when InstanceId is consumed");
        if (ids & 2) check(source.find("armigo_instance_index - armigo_guest_first_instance") != std::string::npos,
            label + " removes the original guest first instance");
        check(source.find("[[base_vertex]]") == std::string::npos,
            label + " retains first/base vertex in VertexId");
        unsigned reflected = 0;
        for (const auto& io : artifact->reflection.io)
            if (io.semantic == IoSemantic::vertex_id || io.semantic == IoSemantic::instance_id) {
                ++reflected;
                check(io.type == IoType{ScalarType::u32, 1} && io.mask == 1 && io.location == 0 &&
                      io.access == Access::read, label + " reflects unsigned scalar input IDs");
            }
        check(reflected == ((ids & 1) != 0) + ((ids & 2) != 0), label + " reflects both requested IDs");
        check(artifact->reflection.resources.empty() && artifact->reflection.dependencies.size() == 1 &&
              artifact->reflection.dependencies[0].key.kind == FactKind::depth_convention,
            label + " does not specialize on dynamic draw parameters");
        if (!fixture_directory.empty()) {
            const auto path = fixture_directory / (label + ".metal");
            std::ofstream(path) << source;
            if (compile_metal) {
                const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
                    quote((fixture_directory / "module-cache").string()) + " -c " + quote(path.string()) +
                    " -o " + quote((fixture_directory / (label + ".air")).string());
                check(std::system(command.c_str()) == 0, label + " Metal compilation");
            }
        }
    }
}
void custom_border_vertex_tests() {
    // A vertex stage sampling with an explicit LOD plans the same coverage-v1
    // pair as a fragment stage, but the support buffer must avoid the native
    // vertex-stream window (constant buffers plus 18 + ordinal) that the
    // package tool and the runtime reserve for vertex streams.
    Fixture f(Stage::vertex);
    f.io(IoSemantic::generic, Access::read, 7, {ScalarType::f32, 3}, 5);
    for (unsigned index = 0; index <= 17; ++index)
        f.resource(BufferUse{StorageBufferSource{{2, 64}}, Access::read,
            {SizeKind::full_binding, 16, 4}}, MetalNamespace::buffer, index);
    TextureDescriptor descriptor; descriptor.format_namespace = "synthetic-format-v1";
    descriptor.comparison = false; descriptor.normalized_coordinates = true;
    descriptor.format = ComponentFormat::floating;
    TextureUse texture; texture.key = {2, 16}; texture.descriptor = descriptor;
    texture.modes = {TextureMode::sample_level};
    const auto texture_id = f.resource(texture, MetalNamespace::texture, 4);
    const auto sampler_id = f.resource(SamplerUse{texture.key, false, true, 1}, MetalNamespace::sampler, 6);
    TextureInfo info; info.texture = texture.key; info.descriptor = descriptor;
    info.resource = texture_id; info.mode = TextureMode::sample_level;
    info.sampler_resource = sampler_id; info.sampler = texture.key;
    info.coordinates = {InputIndex{0}, InputIndex{1}};
    info.level_or_bias = InputIndex{2};
    info.component_mask = 15;
    std::vector<Operand> outputs;
    for (int i = 0; i < 4; ++i) outputs.push_back(f.value());
    f.operation({{}, Opcode::texture_sample, {}, floating(),
        {fp(0.25f), fp(0.25f), fp(1)}, std::move(outputs), info});
    f.store(IoSemantic::position, {fp(0), fp(0), fp(0), fp(1)});
    auto source = accepted(f, "custom_border_vertex_explicit_lod");
    check(source.find("const device ArmigoCustomBorder* armigo_border [[buffer(19)]]") != std::string::npos,
          "Vertex custom-border support avoids the native vertex-stream window");
    const auto original = source.find("resource18.sample(");
    const auto end = source.find(";", original);
    const auto suffix = source.substr(original + std::string("resource18").size(),
        end - original - std::string("resource18").size());
    check(original != std::string::npos && suffix.find("level(") != std::string::npos &&
              source.find("armigo_border[0].coverage" + suffix + ".x") != std::string::npos,
          "Vertex coverage sample repeats the original explicit-LOD operands");
    const auto artifact = std::get<MslArtifact>(f.emit());
    check(artifact.reflection.custom_border &&
              artifact.reflection.custom_border->buffer_index == 19 &&
              artifact.reflection.custom_border->pairs.size() == 1 &&
              artifact.reflection.custom_border->pairs[0].texture == ResourceId{18} &&
              artifact.reflection.custom_border->pairs[0].sampler == ResourceId{19},
          "Vertex custom-border reflection preserves the pair and relocated index");
}
void expanded_tests() {
    Fixture vertex(Stage::vertex);
    vertex.io(IoSemantic::vertex_id, Access::read, 1, {ScalarType::u32,1});
    vertex.io(IoSemantic::instance_id, Access::read, 1, {ScalarType::u32,1});
    vertex.io(IoSemantic::point_size, Access::write, 1, {ScalarType::f32,1});
    vertex.io(IoSemantic::layer, Access::write, 1, {ScalarType::u32,1});
    vertex.io(IoSemantic::viewport_index, Access::write, 1, {ScalarType::u32,1});
    vertex.store(IoSemantic::position,{fp(0),fp(0),fp(0),fp(1)});
    vertex.store(IoSemantic::point_size,{fp(1)},{ScalarType::f32,1});
    vertex.store(IoSemantic::layer,{word(0)},{ScalarType::u32,1});
    vertex.store(IoSemantic::viewport_index,{word(0)},{ScalarType::u32,1});
    IoInfo vertex_id; vertex_id.semantic = IoSemantic::vertex_id; vertex_id.type = {ScalarType::u32,1};
    vertex.operation({{},Opcode::load_io,{}, {},{},{vertex.value()},vertex_id});
    auto builtin_source = accepted(vertex,"vertex_builtins");
    check(builtin_source.find("value0 = builtin4;") != std::string::npos, "Vertex builtin loads use entry parameters");
    {
        auto bad = vertex;
        bad.plan.draw_support_buffer_index.reset();
        rejected(bad, "InstanceId requires draw support reflection");
        bad = vertex;
        bad.plan.draw_support_buffer_index = 31;
        rejected(bad, "Draw support must fit the Metal buffer namespace");
        bad = vertex;
        bad.constant_buffer(0, 16, 0);
        rejected(bad, "Draw support cannot overlap a constant buffer");
        bad = vertex;
        bad.io(IoSemantic::generic, Access::read, 15, {ScalarType::f32, 4});
        bad.plan.draw_support_buffer_index = 18;
        rejected(bad, "Draw support cannot overlap a native vertex stream");
    }
    vertex_builtin_tests();
    for (auto semantic : {IoSemantic::vertex_id, IoSemantic::instance_id}) {
        auto bad = vertex;
        for (auto* uses : {&bad.plan.io, &bad.p.requirements.io})
            for (auto& io : *uses) if (io.semantic == semantic) io.type.scalar = ScalarType::s32;
        rejected(bad, "Signed vertex builtins are rejected");
    }
    Fixture fragment;
    fragment.io(IoSemantic::front_facing,Access::read,1,{ScalarType::boolean,1});
    fragment.io(IoSemantic::point_coord,Access::read,3,{ScalarType::f32,2});
    fragment.io(IoSemantic::sample_id,Access::read,1,{ScalarType::u32,1});
    fragment.io(IoSemantic::sample_mask,Access::read,1,{ScalarType::u32,1});
    fragment.io(IoSemantic::sample_mask,Access::write,1,{ScalarType::u32,1});
    fragment.io(IoSemantic::fragment_depth,Access::write,1,{ScalarType::f32,1});
    fragment.store(IoSemantic::sample_mask,{word(1)},{ScalarType::u32,1});
    fragment.store(IoSemantic::fragment_depth,{fp(0.5f)},{ScalarType::f32,1});
    accepted(fragment,"fragment_builtins");
    {
        // A point-coordinate read flips y under the lower-left origin constant.
        Fixture sprite;
        sprite.io(IoSemantic::point_coord, Access::read, 3, {ScalarType::f32, 2});
        IoInfo coord; coord.semantic = IoSemantic::point_coord; coord.type = {ScalarType::f32, 2}; coord.component_mask = 3;
        const auto u = sprite.value(); const auto v = sprite.value();
        sprite.operation({{}, Opcode::load_io, {}, {}, {}, {Operand{u}, Operand{v}}, coord});
        sprite.io(IoSemantic::color, Access::write, 15, {ScalarType::f32, 4});
        sprite.store(IoSemantic::color, {Operand{u}, Operand{v}, fp(0.0f), fp(1.0f)});
        const auto sprite_source = accepted(sprite, "fragment_point_coord_origin");
        check(sprite_source.find("constant bool armigo_point_coord_lower_left [[function_constant(11)]];") !=
                  std::string::npos &&
              sprite_source.find("armigo_point_coord_lower_left ? 1.0f - ") != std::string::npos,
              "Point coordinates flip y under the lower-left origin constant");
    }
    fragment = Fixture{};
    fragment.io(IoSemantic::generic,Access::read,15,{ScalarType::u32,4},2,{InterpolationMode::flat,InterpolationLocation::center});
    accepted(fragment,"flat_integer");
    // A Metal gather takes no level argument in any stage; outside the
    // fragment stage it still reads the base level without derivatives.
    for (bool offset : {false, true}) {
        auto f = texture_fixture(TextureMode::gather);
        f.p.ir.stage = f.plan.stage = Stage::compute;
        const ComputeConfiguration launch{ComputeLocalSize{{32, 1, 1}}, 0};
        const EvidenceOrigin origin{OriginKind::explicit_option, "synthetic-compute"};
        f.p.requirements.compute = f.plan.compute = launch;
        f.p.ir.dependencies = {{FactKey{FactKind::compute_local_size}, FactValue{ComputeLocalSize{{32, 1, 1}}}, origin},
                               {FactKey{FactKind::compute_shared_memory_bytes}, FactValue{std::uint32_t{0}}, origin}};
        f.plan.dependencies = f.p.ir.dependencies;
        if (offset) {
            auto& op = f.p.ir.operations[0]; auto& info = std::get<TextureInfo>(op.info);
            info.offsets = {InputIndex{static_cast<std::uint16_t>(op.inputs.size())},InputIndex{static_cast<std::uint16_t>(op.inputs.size()+1)}};
            op.inputs.push_back(word(1)); op.inputs.push_back(word(0xffffffffu));
        }
        const auto gather_source = accepted(f, offset ? "compute_gather_offset" : "compute_gather");
        check(gather_source.find(".gather(") != std::string::npos &&
              gather_source.find("level(") == std::string::npos,
              "Non-fragment gathers carry no level argument");
    }
    for (auto mode : {TextureMode::sample_level,TextureMode::sample_gradients,TextureMode::gather}) {
        auto f = texture_fixture(mode);
        auto& op = f.p.ir.operations[0]; auto& info = std::get<TextureInfo>(op.info);
        info.offsets = {InputIndex{static_cast<std::uint16_t>(op.inputs.size())},InputIndex{static_cast<std::uint16_t>(op.inputs.size()+1)}};
        op.inputs.push_back(word(0xffffffffu)); op.inputs.push_back(word(1));
        accepted(f,"offset_"+std::to_string(static_cast<unsigned>(mode)));
    }
    auto arrays = texture_fixture(TextureMode::sample_level);
    for (auto& use : arrays.p.requirements.uses) {
        if (auto* texture = std::get_if<TextureUse>(&use)) { texture->array_length = 2; texture->dynamically_indexed = true; }
        else std::get<SamplerUse>(use).array_length = 2;
    }
    for (auto& resource : arrays.plan.resources) { resource.use = arrays.p.requirements.uses[resource.resource.value]; std::get<MetalRange>(resource.location).count = 2; }
    auto& array_op = arrays.p.ir.operations[0];
    std::get<TextureInfo>(array_op.info).array_index = InputIndex{static_cast<std::uint16_t>(array_op.inputs.size())}; array_op.inputs.push_back(word(1));
    accepted(arrays,"resource_arrays");
    for (bool write : {false,true}) {
        auto f = texture_fixture(TextureMode::fetch);
        auto& use = std::get<TextureUse>(f.p.requirements.uses[0]); use.kind = ResourceTextureKind::storage_image; use.access = write ? Access::write : Access::read;
        f.plan.resources[0].use = use;
        auto& op = f.p.ir.operations[0]; auto& info = std::get<TextureInfo>(op.info);
        op.inputs.pop_back(); info.level_or_bias.reset(); op.opcode = write ? Opcode::image_store : Opcode::image_load;
        if (write) {
            op.outputs.clear(); f.p.ir.values.clear();
            for (unsigned i = 0; i < 4; ++i) { info.data.push_back(InputIndex{static_cast<std::uint16_t>(op.inputs.size())}); op.inputs.push_back(fp(0.25f)); }
        }
        op.effects = required_effects(op); accepted(f,write ? "image_store" : "image_load");
    }
    auto integer = texture_fixture(TextureMode::fetch);
    std::get<TextureUse>(integer.p.requirements.uses[0]).descriptor.scalar = ScalarType::u32;
    std::get<TextureUse>(integer.p.requirements.uses[0]).descriptor.format = ComponentFormat::unsigned_integer;
    integer.plan.resources[0].use = integer.p.requirements.uses[0];
    std::get<TextureInfo>(integer.p.ir.operations[0].info).descriptor = std::get<TextureUse>(integer.p.requirements.uses[0]).descriptor;
    integer.p.ir.operations[0].numeric = {Domain::unsigned_integer,ScalarType::u32};
    accepted(integer,"integer_fetch");
    auto bad = texture_fixture(TextureMode::sample_level,{TextureDimension::d1});
    rejected(bad,"Unsupported 1D LOD is not dropped",DiagnosticCode::unsupported_capability);
    auto one = texture_fixture(TextureMode::fetch,{TextureDimension::d1}); accepted(one,"one_dimensional_read");
    one = texture_fixture(TextureMode::query_size,{TextureDimension::d1}); accepted(one,"one_dimensional_size");
    Fixture memory;
    auto buffer = memory.resource(BufferUse{StorageBufferSource{{0,8}},Access::read_write,{SizeKind::full_binding,64,4}},MetalNamespace::buffer,3);
    MemoryInfo info; info.space = AddressSpace::storage; info.resource = buffer; info.alignment = 4; info.access = Access::write; info.data = {InputIndex{0}};
    memory.operation({{},Opcode::store,{}, {},{word(0x12345678)}, {},info});
    info.access = Access::read; info.data.clear(); memory.operation({{},Opcode::load,{}, {},{},{memory.value()},info});
    accepted(memory,"storage_buffer");
}
void review_regressions() {
    Fixture exported;
    exported.io(IoSemantic::color, Access::write, 15, {ScalarType::f32, 4});
    exported.store(IoSemantic::color, {fp(1), fp(0), fp(0), fp(1)});
    std::get<IoInfo>(exported.p.ir.operations[0].info).output_storage = true;
    accepted(exported, "explicit_fragment_output_storage");
    Fixture f;
    f.options.fragment_position = FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    f.io(IoSemantic::generic,Access::read,1,{ScalarType::f32,1});
    IoInfo io; io.type = {ScalarType::f32,1};
    f.operation({{},Opcode::interpolate,{},floating(),{},{f.value()},io});
    rejected(f,"Perspective interpolation requires explicit factor");
    io.reciprocal_w = InputIndex{0}; f.p.ir.operations[0].info = io; f.p.ir.operations[0].inputs = {fp(0.5f)};
    accepted(f,"option_recorded_in_reflection");
    auto linear = f;
    std::get<IoInfo>(linear.p.ir.operations[0].info).interpolation.mode = InterpolationMode::linear;
    linear.plan.io[0].interpolation.mode = linear.p.requirements.io[0].interpolation.mode = InterpolationMode::linear;
    rejected(linear,"Linear interpolation cannot apply PositionW");
    Fixture v(Stage::vertex); v.store(IoSemantic::position,{fp(0),fp(0),fp(0),fp(1)});
    v.p.regions = {IfRegion{predicate(true),RegionId{1},{}},BlockRegion{BlockId{0}}};
    accepted(v,"literal_guard_position");
    std::get<IfRegion>(v.p.regions[0]).condition = predicate(false);
    rejected(v,"Conditional position export is undefined",DiagnosticCode::undefined_read);
    Fixture output_read(Stage::vertex);
    output_read.io(IoSemantic::generic,Access::write,1,{ScalarType::f32,4},2);
    output_read.store(IoSemantic::position,{fp(0),fp(0),fp(0),fp(1)});
    output_read.store(IoSemantic::generic,{fp(0.75f)},{ScalarType::f32,4},2);
    IoInfo output_info; output_info.semantic = IoSemantic::generic;
    output_info.location = 2; output_info.type = {ScalarType::f32,4};
    output_info.output_storage = true;
    output_read.operation({{},Opcode::load_io,{}, {},{}, {output_read.value()},output_info});
    auto output_source = accepted(output_read,"readable_vertex_output");
    check(output_source.find("value0 = as_type<uint>(out.generic2_0);") != std::string::npos,
        "Output-space loads read the writable stage interface");
    Fixture undefined_output(Stage::vertex);
    undefined_output.io(IoSemantic::generic,Access::write,1,{ScalarType::f32,4},2);
    undefined_output.store(IoSemantic::position,{fp(0),fp(0),fp(0),fp(1)});
    undefined_output.operation({{},Opcode::load_io,{}, {},{}, {undefined_output.value()},output_info});
    undefined_output.store(IoSemantic::generic,{fp(0.75f)},{ScalarType::f32,4},2);
    rejected(undefined_output,"Output read before write",DiagnosticCode::undefined_read);
    Fixture memory;
    auto cb = memory.constant_buffer(0,16,0);
    MemoryInfo m; m.resource = cb; m.constant_buffer_slot = 0; m.dynamic_byte_offset = InputIndex{0};
    memory.operation({{},Opcode::load,{}, {},{word(16)},{memory.value()},m});
    rejected(memory,"Literal dynamic buffer offset must fit the reflected window");
    memory.p.ir.operations[0].inputs = {word(12)}; accepted(memory,"last_buffer_word");
    std::get<BufferUse>(memory.plan.resources[0].use).size.alignment = std::get<BufferUse>(memory.p.requirements.uses[0]).size.alignment = 1;
    rejected(memory,"Raw-word buffer needs base alignment");
    Fixture dynamic;
    cb = dynamic.constant_buffer(0,64,0);
    auto offset = dynamic.value(), loaded = dynamic.value();
    m.dynamic_byte_offset.reset(); m.resource = cb;
    dynamic.operation({{},Opcode::load,{}, {},{},{offset},m});
    m.dynamic_byte_offset = InputIndex{0};
    dynamic.operation({{},Opcode::load,{}, {},{offset},{loaded},m});
    rejected(dynamic,"Nonliteral buffer offset needs a proof below the full window");
    std::get<MemoryInfo>(dynamic.p.ir.operations[1].info).dynamic_byte_offset_upper_bound = 60;
    accepted(dynamic,"bounded_dynamic_buffer");
    std::get<MemoryInfo>(dynamic.p.ir.operations[1].info).dynamic_byte_offset_upper_bound = 64;
    rejected(dynamic,"Dynamic offset proof must fit the reflected window");
    auto depth = texture_fixture(TextureMode::sample_level);
    auto& texture = std::get<TextureUse>(depth.p.requirements.uses[0]); texture.descriptor.format = ComponentFormat::depth;
    depth.plan.resources[0].use = texture;
    auto& op = depth.p.ir.operations[0]; auto& t = std::get<TextureInfo>(op.info);
    t.descriptor = texture.descriptor; t.component_mask = 1; op.outputs.resize(1); depth.p.ir.values.resize(1);
    auto depth_source = accepted(depth,"depth_without_comparison");
    check(depth_source.find("const float texture_result0") != std::string::npos && depth_source.find("texture_result0.x") == std::string::npos,"Raw depth sample stays scalar");
    auto invalid_depth = depth;
    std::get<TextureInfo>(invalid_depth.p.ir.operations[0].info).component_mask = 2;
    rejected(invalid_depth, "Raw depth non-X selection remains unsupported");
    std::get<TextureInfo>(invalid_depth.p.ir.operations[0].info).component_mask = 3;
    invalid_depth.p.ir.operations[0].outputs.push_back(invalid_depth.value());
    rejected(invalid_depth, "Raw depth vector selection remains unsupported");
    Fixture outer;
    outer.p.regions = {SequenceRegion{{RegionId{1},RegionId{2}}},BlockRegion{BlockId{0}},LoopRegion{RegionId{3},RegionId{8}},
        LoopRegion{RegionId{4},{}},IfRegion{predicate(true),RegionId{5},RegionId{6}},ContinueRegion{RegionId{2}},
        BreakRegion{RegionId{2}},SequenceRegion{},SequenceRegion{{RegionId{7}}}};
    accepted(outer,"outer_loop_transfers");
    outer.p.regions = {SequenceRegion{{RegionId{1},RegionId{2}}},BlockRegion{BlockId{0}},LoopRegion{RegionId{3},RegionId{4}},
        SequenceRegion{},LoopRegion{RegionId{5},{}},IfRegion{predicate(true),RegionId{6},RegionId{7}},
        ContinueRegion{RegionId{2}},BreakRegion{RegionId{2}}};
    accepted(outer,"continuing_outer_loop_transfers");
}
void flow_tests() {
    Fixture f;
    f.io(IoSemantic::color, Access::write, 15, {ScalarType::f32,4}); f.store(IoSemantic::color, {fp(1),fp(0),fp(0),fp(1)});
    f.p.regions = {SequenceRegion{{RegionId{1},RegionId{2},RegionId{8}}}, BlockRegion{BlockId{0}}, LoopRegion{RegionId{3}, RegionId{7}},
        IfRegion{predicate(true), RegionId{4}, RegionId{5}}, ContinueRegion{RegionId{2}}, BreakRegion{RegionId{2}},
        SequenceRegion{}, SequenceRegion{{RegionId{6}}}, ReturnRegion{}};
    auto source = accepted(f, "structured_loop");
    check(source.find("do {") != std::string::npos && source.find("armigo_break_2 = true") != std::string::npos && source.find("} else {") != std::string::npos, "Loop continue reaches continuing region");
    auto bad = f; std::get<ContinueRegion>(bad.p.regions[4]).loop = RegionId{99}; rejected(bad, "Invalid loop transfer", DiagnosticCode::invalid_ir);
    bad = f; bad.p.ir.limits.nesting_depth = 2; rejected(bad, "Nesting limit", DiagnosticCode::limit_exceeded);
    Fixture dataflow;
    dataflow.io(IoSemantic::color, Access::write, 1, {ScalarType::u32,1});
    auto cb = dataflow.constant_buffer(0,16,0);
    auto value = dataflow.value(ValueType::word32,ValueKind::mutable_local);
    MemoryInfo m; m.constant_buffer_slot = 0; m.resource = cb;
    dataflow.operation({{},Opcode::load,{}, {}, {},{value},m});
    dataflow.store(IoSemantic::color,{value},{ScalarType::u32,1});
    dataflow.p.ir.blocks.push_back({FunctionId{0},{},{},{},{OperationId{1}},Return{}});
    dataflow.p.ir.blocks[0].operations = {OperationId{0}}; dataflow.p.ir.functions[0].blocks.push_back(BlockId{1});
    dataflow.p.regions = {SequenceRegion{{RegionId{1},RegionId{3}}},IfRegion{predicate(true),RegionId{2},{}},BlockRegion{BlockId{0}},BlockRegion{BlockId{1}}};
    accepted(dataflow,"literal_guard_definition");
    std::get<IfRegion>(dataflow.p.regions[1]).condition = predicate(false);
    zero_read(dataflow, "Conditional uninitialized local (reads zero)");
}

void definedness_tests() {
    Fixture f;
    f.io(IoSemantic::front_facing, Access::read, 1, {ScalarType::boolean, 1});
    f.io(IoSemantic::color, Access::write, 1, {ScalarType::u32, 1});
    const auto condition = f.value(ValueType::predicate), inverse = f.value(ValueType::predicate);
    const auto value = f.value(ValueType::word32, ValueKind::mutable_local);
    IoInfo facing; facing.semantic = IoSemantic::front_facing; facing.type = {ScalarType::boolean, 1};
    const NumericMode logical{Domain::boolean, ScalarType::boolean};
    f.operation({{}, Opcode::load_io, {}, {}, {}, {condition}, facing});
    f.operation({{}, Opcode::copy, {}, {}, {word(7)}, {value}, {}});
    f.operation({{}, Opcode::logical_not, {}, logical, {condition}, {inverse}, {}});
    f.operation({{}, Opcode::copy, {}, {}, {word(11)}, {value}, {}});
    f.store(IoSemantic::color, {value}, {ScalarType::u32, 1});
    f.isolate_operations();
    f.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}, RegionId{4}, RegionId{5}, RegionId{7}}},
        BlockRegion{BlockId{0}}, IfRegion{condition, RegionId{3}, {}}, BlockRegion{BlockId{1}},
        BlockRegion{BlockId{2}}, IfRegion{inverse, RegionId{6}, {}}, BlockRegion{BlockId{3}}, BlockRegion{BlockId{4}}};
    accepted(f, "complementary_definitions");
    auto missing = f;
    std::get<IfRegion>(missing.p.regions[5]).condition = condition;
    zero_read(missing, "Noncomplementary definitions remain undefined (reads zero)");

    auto outputs = f;
    IoInfo color; color.semantic = IoSemantic::color; color.type = {ScalarType::u32, 1}; color.data = {InputIndex{0}};
    for (unsigned i : {1u, 3u}) {
        auto& op = outputs.p.ir.operations[i]; op.opcode = Opcode::store_io;
        op.outputs.clear(); op.info = color; op.effects = required_effects(op);
    }
    outputs.p.ir.operations[4].inputs = {word(1)};
    // Output coverage must come from the two guards, before the final return.
    outputs.p.ir.operations[4].opcode = Opcode::copy;
    outputs.p.ir.operations[4].outputs = {value}; outputs.p.ir.operations[4].info = {};
    outputs.p.ir.operations[4].effects = Effects::none;
    accepted(outputs, "complementary_output_writes");
    std::get<IfRegion>(outputs.p.regions[5]).condition = condition;
    rejected(outputs, "Missing guarded output stays rejected", DiagnosticCode::undefined_read);

    Fixture loop;
    loop.io(IoSemantic::color, Access::write, 1, {ScalarType::u32, 1});
    const auto flag = loop.value(ValueType::predicate, ValueKind::mutable_local);
    const auto carried = loop.value(ValueType::word32, ValueKind::mutable_local);
    loop.operation({{}, Opcode::copy, {}, logical, {predicate(false)}, {flag}, {}});
    loop.operation({{}, Opcode::copy, {}, {}, {word(19)}, {carried}, {}});
    loop.operation({{}, Opcode::copy, {}, logical, {predicate(true)}, {flag}, {}});
    loop.store(IoSemantic::color, {carried}, {ScalarType::u32, 1});
    loop.isolate_operations();
    loop.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}, RegionId{9}}}, BlockRegion{BlockId{0}},
        LoopRegion{RegionId{3}, {}}, SequenceRegion{{RegionId{4}, RegionId{6}, RegionId{7}, RegionId{8}}},
        IfRegion{flag, RegionId{5}, {}}, BreakRegion{RegionId{2}}, BlockRegion{BlockId{1}},
        BlockRegion{BlockId{2}}, ContinueRegion{RegionId{2}}, BlockRegion{BlockId{3}}};
    accepted(loop, "loop_carried_definition");
    auto skipped = loop;
    skipped.p.ir.operations[0].inputs = {predicate(true)};
    zero_read(skipped, "First-iteration break skips definition (reads zero)");

    // The first iteration skips this read, but a later iteration reaches it.
    auto late = loop;
    late.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}}}, BlockRegion{BlockId{0}},
        LoopRegion{RegionId{3}, {}}, SequenceRegion{{RegionId{4}, RegionId{8}, RegionId{9}}},
        IfRegion{flag, RegionId{5}, {}}, SequenceRegion{{RegionId{6}, RegionId{7}}},
        BlockRegion{BlockId{3}}, BlockRegion{BlockId{1}}, BlockRegion{BlockId{2}}, ContinueRegion{RegionId{2}}};
    // This loop has no exit to a return, so it is checked as emitted MSL only.
    {
        const auto result = late.emit();
        const auto* artifact = std::get_if<MslArtifact>(&result);
        check(artifact && artifact->source.find(" = {};") != std::string::npos,
              "Undefined read reached on a later iteration reads zero");
    }
    auto nested = loop;
    nested.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}, RegionId{12}}}, BlockRegion{BlockId{0}},
        LoopRegion{RegionId{3}, RegionId{11}}, SequenceRegion{{RegionId{4}, RegionId{6}}},
        IfRegion{flag, RegionId{5}, {}}, BreakRegion{RegionId{2}}, LoopRegion{RegionId{7}, RegionId{8}},
        SequenceRegion{}, SequenceRegion{{RegionId{9}, RegionId{10}}}, BlockRegion{BlockId{1}},
        ContinueRegion{RegionId{2}}, BlockRegion{BlockId{2}}, BlockRegion{BlockId{3}}};
    accepted(nested, "nested_continuing_definition");
    auto bypassed = nested;
    bypassed.p.regions.push_back(IfRegion{predicate(false), RegionId{9}, {}});
    std::get<SequenceRegion>(bypassed.p.regions[8]).children = {RegionId{13}, RegionId{10}};
    zero_read(bypassed, "Outer continue bypasses a nested definition (reads zero)");
    auto changed = f;
    changed.p.ir.values[condition.value].kind = ValueKind::mutable_local;
    changed.p.ir.operations[2].outputs = {condition};
    std::get<IfRegion>(changed.p.regions[5]).condition = condition;
    accepted(changed, "aliased_predicate_complement");
    changed.p.ir.operations[2].opcode = Opcode::copy;
    changed.p.ir.operations[2].inputs = {predicate(false)};
    changed.p.ir.operations[2].numeric = logical;
    zero_read(changed, "Overwritten predicates do not retain stale facts (reads zero)");
    auto aliases = f;
    aliases.p.ir.values[condition.value].kind = ValueKind::mutable_local;
    aliases.p.ir.operations[2].opcode = Opcode::copy;
    aliases.p.ir.operations[3] = aliases.p.ir.operations[0];
    aliases.store(IoSemantic::color, {word(1)}, {ScalarType::u32, 1});
    aliases.isolate_operations();
    aliases.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}, RegionId{3}, RegionId{4}, RegionId{6}, RegionId{8}}},
        BlockRegion{BlockId{0}}, BlockRegion{BlockId{2}}, BlockRegion{BlockId{3}},
        IfRegion{inverse, RegionId{5}, {}}, BlockRegion{BlockId{1}},
        IfRegion{condition, RegionId{7}, {}}, BlockRegion{BlockId{4}}, BlockRegion{BlockId{5}}};
    zero_read(aliases, "An overwritten atom is independent of its surviving aliases (reads zero)");

    // Reusing independent guards used to retain every truth assignment even
    // though neither arm changed a must fact. Complement aliases exercise the
    // parity retained when opposite constant environments are coalesced.
    for (const unsigned count : {8u, 16u, 32u, 64u}) {
        Fixture independent;
        independent.io(IoSemantic::front_facing, Access::read, 1, {ScalarType::boolean, 1});
        independent.io(IoSemantic::color, Access::write, 1, {ScalarType::u32, 1});
        std::vector<ValueId> conditions, complements;
        for (unsigned i = 0; i < count; ++i) {
            const auto condition_id = independent.value(ValueType::predicate);
            const auto complement_id = independent.value(ValueType::predicate);
            independent.operation({{}, Opcode::load_io, {}, {}, {}, {condition_id}, facing});
            independent.operation({{}, Opcode::logical_not, {}, logical,
                                   {condition_id}, {complement_id}, {}});
            conditions.push_back(condition_id);
            complements.push_back(complement_id);
        }
        independent.store(IoSemantic::color, {word(1)}, {ScalarType::u32, 1});
        independent.isolate_operations();
        independent.p.regions = {SequenceRegion{}};
        std::vector<RegionId> children;
        for (std::uint32_t i = 0; i < 2 * count; ++i) {
            children.push_back(RegionId{static_cast<std::uint32_t>(independent.p.regions.size())});
            independent.p.regions.push_back(BlockRegion{BlockId{i}});
        }
        auto guard = [&](ValueId condition_id) {
            const RegionId body{static_cast<std::uint32_t>(independent.p.regions.size())};
            independent.p.regions.push_back(SequenceRegion{});
            children.push_back(RegionId{static_cast<std::uint32_t>(independent.p.regions.size())});
            independent.p.regions.push_back(IfRegion{condition_id, body, {}});
        };
        for (auto condition_id : conditions) guard(condition_id);
        for (auto complement_id : complements) guard(complement_id);
        children.push_back(RegionId{static_cast<std::uint32_t>(independent.p.regions.size())});
        independent.p.regions.push_back(BlockRegion{BlockId{2 * count}});
        std::get<SequenceRegion>(independent.p.regions[0]).children = std::move(children);
        independent.p.ir.limits.analysis_steps = 20000ull * count;
        accepted(independent, "independent_guard_coalescing_" + std::to_string(count));
    }

    // Distinct live must-facts cannot be coalesced. Crossing the environment
    // cap must widen conservatively and report the real undefined reads.
    Fixture widened;
    widened.io(IoSemantic::front_facing, Access::read, 1, {ScalarType::boolean, 1});
    constexpr unsigned widened_count = 8;
    std::vector<ValueId> widened_conditions, widened_values;
    for (unsigned i = 0; i < widened_count; ++i) {
        const auto condition_id = widened.value(ValueType::predicate);
        widened.operation({{}, Opcode::load_io, {}, {}, {}, {condition_id}, facing});
        widened_conditions.push_back(condition_id);
    }
    for (unsigned i = 0; i < widened_count; ++i) {
        const auto value_id = widened.value(ValueType::word32, ValueKind::mutable_local);
        widened.operation({{}, Opcode::copy, {}, {}, {word(i)}, {value_id}, {}});
        widened_values.push_back(value_id);
    }
    for (auto value_id : widened_values)
        widened.operation({{}, Opcode::copy, {}, {}, {value_id}, {widened.value()}, {}});
    widened.isolate_operations();
    widened.p.regions = {SequenceRegion{}};
    std::vector<RegionId> widened_children;
    for (std::uint32_t i = 0; i < widened_count; ++i) {
        widened_children.push_back(RegionId{static_cast<std::uint32_t>(widened.p.regions.size())});
        widened.p.regions.push_back(BlockRegion{BlockId{i}});
    }
    for (std::uint32_t i = 0; i < widened_count; ++i) {
        const RegionId body{static_cast<std::uint32_t>(widened.p.regions.size())};
        widened.p.regions.push_back(BlockRegion{BlockId{widened_count + i}});
        widened_children.push_back(RegionId{static_cast<std::uint32_t>(widened.p.regions.size())});
        widened.p.regions.push_back(IfRegion{widened_conditions[i], body, {}});
    }
    for (auto condition_id : widened_conditions) {
        const RegionId body{static_cast<std::uint32_t>(widened.p.regions.size())};
        widened.p.regions.push_back(SequenceRegion{});
        widened_children.push_back(RegionId{static_cast<std::uint32_t>(widened.p.regions.size())});
        widened.p.regions.push_back(IfRegion{condition_id, body, {}});
    }
    for (std::uint32_t i = 0; i < widened_count; ++i) {
        widened_children.push_back(RegionId{static_cast<std::uint32_t>(widened.p.regions.size())});
        widened.p.regions.push_back(BlockRegion{BlockId{2 * widened_count + i}});
    }
    std::get<SequenceRegion>(widened.p.regions[0]).children = std::move(widened_children);
    widened.p.ir.limits.analysis_steps = 400000;
    zero_read(widened, "Widened environments remain conservative (reads zero)");

    Fixture widened_output;
    widened_output.io(IoSemantic::front_facing, Access::read, 1, {ScalarType::boolean, 1});
    std::vector<ValueId> output_conditions;
    for (unsigned i = 0; i < widened_count; ++i) {
        widened_output.io(IoSemantic::color, Access::write, 1, {ScalarType::u32, 1}, i);
        const auto condition_id = widened_output.value(ValueType::predicate);
        widened_output.operation({{}, Opcode::load_io, {}, {}, {}, {condition_id}, facing});
        output_conditions.push_back(condition_id);
    }
    for (unsigned i = 0; i < widened_count; ++i)
        widened_output.store(IoSemantic::color, {word(i)}, {ScalarType::u32, 1}, i);
    IoInfo widened_output_read;
    widened_output_read.semantic = IoSemantic::color;
    widened_output_read.location = 0;
    widened_output_read.type = {ScalarType::u32, 1};
    widened_output_read.output_storage = true;
    widened_output.operation({{}, Opcode::load_io, {}, {}, {},
                              {widened_output.value()}, widened_output_read});
    widened_output.isolate_operations();
    widened_output.p.regions = {SequenceRegion{}};
    std::vector<RegionId> output_children;
    for (std::uint32_t i = 0; i < widened_count; ++i) {
        output_children.push_back(RegionId{static_cast<std::uint32_t>(widened_output.p.regions.size())});
        widened_output.p.regions.push_back(BlockRegion{BlockId{i}});
    }
    for (std::uint32_t i = 0; i < widened_count; ++i) {
        const RegionId body{static_cast<std::uint32_t>(widened_output.p.regions.size())};
        widened_output.p.regions.push_back(BlockRegion{BlockId{widened_count + i}});
        output_children.push_back(RegionId{static_cast<std::uint32_t>(widened_output.p.regions.size())});
        widened_output.p.regions.push_back(IfRegion{output_conditions[i], body, {}});
    }
    for (auto condition_id : output_conditions) {
        const RegionId body{static_cast<std::uint32_t>(widened_output.p.regions.size())};
        widened_output.p.regions.push_back(SequenceRegion{});
        output_children.push_back(RegionId{static_cast<std::uint32_t>(widened_output.p.regions.size())});
        widened_output.p.regions.push_back(IfRegion{condition_id, body, {}});
    }
    output_children.push_back(RegionId{static_cast<std::uint32_t>(widened_output.p.regions.size())});
    widened_output.p.regions.push_back(BlockRegion{BlockId{2 * widened_count}});
    std::get<SequenceRegion>(widened_output.p.regions[0]).children = std::move(output_children);
    widened_output.p.ir.limits.analysis_steps = 400000;
    const auto widened_output_result = widened_output.emit();
    const auto* widened_output_error = std::get_if<Diagnostic>(&widened_output_result);
    check(widened_output_error && widened_output_error->code == DiagnosticCode::undefined_read &&
          widened_output_error->message.find("read before it is written") != std::string::npos,
          "Widened output facts remain conservative");

    // The precise states prove that p && q && (p xor q) is unreachable. The
    // widened parity summary may admit it, but that abstract path must not
    // witness a vertex write and enable output initialization.
    Fixture widened_witness(Stage::vertex);
    std::vector<ValueId> witness_inputs, witness_conditions, witness_flags;
    for (unsigned i = 0; i < widened_count; ++i) {
        widened_witness.io(IoSemantic::generic, Access::read, 1, {ScalarType::u32, 1}, i);
        const auto input_id = widened_witness.value();
        const auto condition_id = widened_witness.value(ValueType::predicate);
        IoInfo input;
        input.semantic = IoSemantic::generic;
        input.location = static_cast<std::uint16_t>(i);
        input.type = {ScalarType::u32, 1};
        input.component_mask = 1;
        widened_witness.operation({{}, Opcode::load_io, {}, {}, {}, {input_id}, input});
        widened_witness.operation({{}, Opcode::compare, {}, {Domain::unsigned_integer, ScalarType::u32},
                                   {input_id, word(0)}, {condition_id}, ComparisonInfo{Compare::not_equal, false}});
        witness_inputs.push_back(input_id);
        witness_conditions.push_back(condition_id);
    }
    // Each guarded body defines a predicate that stays live, so the arms
    // differ in their environments and the partitions cannot coalesce.
    for (unsigned i = 0; i < widened_count; ++i) {
        const auto flag_id = widened_witness.value(ValueType::predicate);
        widened_witness.operation({{}, Opcode::compare, {}, {Domain::unsigned_integer, ScalarType::u32},
                                   {witness_inputs[i], word(1)}, {flag_id}, ComparisonInfo{Compare::not_equal, false}});
        witness_flags.push_back(flag_id);
    }
    const auto unequal = widened_witness.value(ValueType::predicate);
    widened_witness.operation({{}, Opcode::logical_xor, {}, logical,
                               {witness_conditions[0], witness_conditions[1]}, {unequal}, {}});
    widened_witness.store(IoSemantic::position, {fp(0), fp(0), fp(0), fp(1)});
    widened_witness.isolate_operations();
    widened_witness.p.regions = {SequenceRegion{}};
    std::vector<RegionId> witness_children;
    for (std::uint32_t i = 0; i < 2 * widened_count; ++i) {
        witness_children.push_back(RegionId{static_cast<std::uint32_t>(widened_witness.p.regions.size())});
        widened_witness.p.regions.push_back(BlockRegion{BlockId{i}});
    }
    for (std::uint32_t i = 0; i < widened_count; ++i) {
        const RegionId body{static_cast<std::uint32_t>(widened_witness.p.regions.size())};
        widened_witness.p.regions.push_back(BlockRegion{BlockId{2 * widened_count + i}});
        witness_children.push_back(RegionId{static_cast<std::uint32_t>(widened_witness.p.regions.size())});
        widened_witness.p.regions.push_back(IfRegion{witness_conditions[i], body, {}});
    }
    for (auto condition_id : witness_conditions) {
        const RegionId body{static_cast<std::uint32_t>(widened_witness.p.regions.size())};
        widened_witness.p.regions.push_back(SequenceRegion{});
        witness_children.push_back(RegionId{static_cast<std::uint32_t>(widened_witness.p.regions.size())});
        widened_witness.p.regions.push_back(IfRegion{condition_id, body, {}});
    }
    witness_children.push_back(RegionId{static_cast<std::uint32_t>(widened_witness.p.regions.size())});
    widened_witness.p.regions.push_back(BlockRegion{BlockId{3 * widened_count}});
    const RegionId write{static_cast<std::uint32_t>(widened_witness.p.regions.size())};
    widened_witness.p.regions.push_back(BlockRegion{BlockId{3 * widened_count + 1}});
    const RegionId unequal_guard{static_cast<std::uint32_t>(widened_witness.p.regions.size())};
    widened_witness.p.regions.push_back(IfRegion{unequal, write, {}});
    const RegionId second_guard{static_cast<std::uint32_t>(widened_witness.p.regions.size())};
    widened_witness.p.regions.push_back(IfRegion{witness_conditions[1], unequal_guard, {}});
    witness_children.push_back(RegionId{static_cast<std::uint32_t>(widened_witness.p.regions.size())});
    widened_witness.p.regions.push_back(IfRegion{witness_conditions[0], second_guard, {}});
    const RegionId returned{static_cast<std::uint32_t>(widened_witness.p.regions.size())};
    widened_witness.p.regions.push_back(ReturnRegion{});
    witness_children.push_back(RegionId{static_cast<std::uint32_t>(widened_witness.p.regions.size())});
    widened_witness.p.regions.push_back(IfRegion{predicate(true), returned, {}});
    for (auto flag_id : witness_flags) {
        const RegionId body{static_cast<std::uint32_t>(widened_witness.p.regions.size())};
        widened_witness.p.regions.push_back(SequenceRegion{});
        witness_children.push_back(RegionId{static_cast<std::uint32_t>(widened_witness.p.regions.size())});
        widened_witness.p.regions.push_back(IfRegion{flag_id, body, {}});
    }
    std::get<SequenceRegion>(widened_witness.p.regions[0]).children = std::move(witness_children);
    widened_witness.p.ir.limits.analysis_steps = 600000;
    const auto widened_witness_result = widened_witness.emit();
    const auto* widened_witness_error = std::get_if<Diagnostic>(&widened_witness_result);
    check(widened_witness_error && widened_witness_error->code == DiagnosticCode::undefined_read &&
          widened_witness_error->message.find("Stage output is not written") != std::string::npos,
          "Widened paths do not witness vertex writes");
    if (widened_witness_error && widened_witness_error->message.find("Stage output is not written") == std::string::npos)
        std::cerr << "  " << widened_witness_error->message << '\n';

    Fixture unlowered;
    unlowered.operation({{}, Opcode::copy, {}, {}, {word(1)}, {Register{RegisterFile::gpr, 0}}, {}});
    rejected(unlowered, "Register destinations are rejected before analysis", DiagnosticCode::invalid_ir);
    auto bounded = f; bounded.p.ir.limits.analysis_steps = 10;
    rejected(bounded, "Definedness analysis is bounded", DiagnosticCode::limit_exceeded);
    Fixture output_budget;
    for (unsigned i = 0; i < 64; ++i) {
        output_budget.io(IoSemantic::generic, Access::write, 1, {ScalarType::f32, 1}, i);
        output_budget.store(IoSemantic::generic, {fp(1)}, {ScalarType::f32, 1}, i);
    }
    output_budget.p.ir.limits.analysis_steps = 2000;
    rejected(output_budget, "Written output facts consume analysis work", DiagnosticCode::limit_exceeded);
}

struct GuestFlow {
    struct Fixup { std::size_t index; std::string target; };
    std::vector<std::uint64_t> words;
    std::map<std::string, std::size_t> labels;
    std::vector<Fixup> fixups;
    static Pc pc(std::size_t i) { return static_cast<Pc>((i / 3) * 32 + (i % 3) * 8 + 8); }
    static std::uint64_t guard(std::uint64_t word, unsigned predicate, bool inverted) {
        return (word & ~(15ull << 16)) | (std::uint64_t{predicate} << 16) | (std::uint64_t{inverted} << 19);
    }
    void mov(unsigned destination, std::uint32_t bits, unsigned predicate = 7, bool inverted = false) {
        words.push_back(guard(0x0100000000000000ull | destination | (std::uint64_t{bits} << 20), predicate, inverted));
    }
    void store(unsigned reg, unsigned offset, unsigned count = 1, unsigned predicate = 7, bool inverted = false) {
        words.push_back(guard(0xeff000000000ff00ull | reg | (std::uint64_t{offset} << 20) |
            (std::uint64_t{count - 1} << 47), predicate, inverted));
    }
    void condition(unsigned destination) {
        // ISETP.LT.AND Pn, PT, R10, R11, PT.
        words.push_back(0x5b60000000000000ull | (std::uint64_t{destination} << 3) | 7ull |
            (10ull << 8) | (7ull << 16) | (11ull << 20) | (7ull << 39) | (1ull << 49));
    }
    void branch(std::string target, unsigned predicate = 7) {
        fixups.push_back({words.size(), std::move(target)});
        words.push_back(guard(0xe24000000000000full, predicate, false));
    }
    void label(std::string name) { labels.emplace(std::move(name), words.size()); }
    StageImage image() {
        for (const auto& fixup : fixups) {
            const auto displacement = (std::int64_t(pc(labels.at(fixup.target))) - pc(fixup.index) - 8) & 0xffffff;
            words[fixup.index] |= std::uint64_t(displacement) << 20;
        }
        StageImage image; image.stage = Stage::vertex;
        auto append = [&](std::uint64_t word) {
            for (unsigned i = 0; i < 8; ++i) image.bytes.push_back(std::byte((word >> (8 * i)) & 255));
        };
        for (auto word : words) {
            if (image.bytes.size() % 32 == 0) append(0);
            append(word);
        }
        return image;
    }
};

void decoded_definedness_tests() {
    struct Facts final : FactSource {
        Knowledge<FactValue> lookup(FactKey key) const override {
            if (key.kind == FactKind::depth_convention)
                return Known<FactValue>{DepthConvention::zero_to_one,
                    {OriginKind::explicit_option, "synthetic-structured-definedness"}};
            return Unknown{"No additional facts for the synthetic control-flow program"};
        }
    } source;
    auto take = []<class T>(Result<T> result) -> T {
        if (const auto* d = std::get_if<Diagnostic>(&result)) throw std::runtime_error(d->message);
        return std::get<T>(std::move(result));
    };
    for (unsigned variant = 0; variant < 7; ++variant) {
        const bool dispatch = variant >= 2 && variant <= 4, entry_output = variant == 3;
        const bool missing_output = variant == 4 || variant == 6;
        const auto label = "decoded_definedness_" + std::to_string(variant);
        try {
            GuestFlow guest;
            guest.words.push_back(0xefd800000007ff0aull | (0x2fcull << 20));
            guest.mov(11, 4);
            for (unsigned p = 0; p < 4; ++p) guest.condition(p);
            guest.mov(1, 0); guest.mov(2, 0); guest.mov(3, 0x3f800000);
            if (variant == 5) {
                guest.mov(0, 0x3f800000);
                for (unsigned i = 0; i < 256; ++i) {
                    guest.mov(0, 0x3f000000, 0);
                    guest.mov(0, 0x3f800000, 0, true);
                }
                guest.store(0, 0x70, 4);
            } else if (variant == 6) {
                guest.mov(0, 0);
                guest.store(0, 0x70); guest.store(1, 0x74); guest.store(2, 0x78);
                guest.words.push_back(GuestFlow::guard(0xe30000000000000full, 0, false));
                guest.store(3, 0x7c);
            } else if (!dispatch) {
                guest.mov(0, 0x3f000000, 0);
                guest.mov(0, 0x3f800000, 0, true);
                if (variant == 0) guest.store(0, 0x70, 4);
                else {
                    guest.store(0, 0x70, 1, 0);
                    guest.store(0, 0x70, 1, 0, true);
                    guest.store(1, 0x74); guest.store(2, 0x78); guest.store(3, 0x7c);
                }
            } else {
                guest.mov(0, 0x3f000000);
                if (entry_output) guest.store(0, 0x70, 4);
                guest.label("outer"); guest.branch("exit", 0);
                guest.label("inner"); guest.branch("outer_tail", 1);
                guest.branch("inner_tail", 2);
                guest.mov(0, 0x3f800000);
                if (missing_output) guest.store(0, 0x80);
                guest.branch("exit", 3); guest.branch("outer");
                guest.label("inner_tail"); guest.branch("inner");
                guest.label("outer_tail"); guest.branch("outer");
                guest.label("exit");
                if (!entry_output) guest.store(0, 0x70, 4);
            }
            guest.words.push_back(0xe30000000007000full);
            FactReader facts(source);
            auto decoded = take(decode(guest.image(), facts, {}));
            auto registers = take(emit_ir(decoded, facts, {}));
            auto ssa = take(construct_ssa(std::move(registers), {}));
            const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
            auto legal = take(legalize(std::move(ssa), facts, profile));
            auto structured = take(structure(std::move(legal), {}));
            if (variant == 5) structured.ir.limits.analysis_steps = 1u << 21;
            if (dispatch) {
                const auto& entry = structured.ir.blocks[structured.ir.functions[0].entry.value];
                check(!entry.operations.empty() && !structured.ir.operations[entry.operations[0].value].source.pc,
                    label + " exercises pending-target dispatch fallback");
            }
            auto plan = take(plan_resources(structured, profile));
            auto result = emit_msl(structured, plan, {});
            if (missing_output) {
                // armigo-output-init-v1 replaces the rejection these variants
                // used to pin: an output written on some live path is
                // initialized at entry, so the skipped write is covered.
                const auto* error = std::get_if<Diagnostic>(&result);
                check(!error, label + " accepts a partially written output under armigo-output-init-v1");
                if (!error) {
                    const auto& artifact = std::get<MslArtifact>(result);
                    check(artifact.source.find("out.position.w = 1.0f;") != std::string::npos,
                        label + " initializes position W at entry");
                    if (variant == 4)
                        check(artifact.source.find("out.generic0_0 = float(0);") != std::string::npos,
                            label + " initializes the path-only generic output at entry");
                }
                continue;
            }
            const auto artifact = take(std::move(result));
            check(artifact.source.find("value") != std::string::npos, label + " reaches MSL emission");
            if (!fixture_directory.empty()) {
                const auto path = fixture_directory / (label + ".metal");
                std::ofstream(path) << artifact.source;
                if (compile_metal) {
                    const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
                        quote((fixture_directory / "module-cache").string()) + " -c " + quote(path.string()) +
                        " -o " + quote((fixture_directory / (label + ".air")).string());
                    check(std::system(command.c_str()) == 0, label + " Metal compilation");
                }
            }
        } catch (const std::exception& error) { check(false, label + ": " + error.what()); }
    }
}
}
// Matrix planning is charged to the emitter's analysis budget: at least one
// step for each operation in each of its passes, and a spent budget stops it.
void matrix_plan_budget_tests() {
    StructuredProgram p;
    p.ir.stage = Stage::compute;
    constexpr unsigned count = 40;
    BasicBlock block;
    for (unsigned i = 0; i < count; ++i) {
        Operation op;
        op.opcode = Opcode::matrix_multiply_add;
        op.info = SubgroupInfo{};
        op.inputs.push_back(LiteralBits{0});
        for (unsigned word = 0; word < 8; ++word) {
            op.inputs.push_back(ValueId{static_cast<std::uint32_t>(p.ir.values.size())});
            p.ir.values.push_back({});
        }
        for (unsigned word = 0; word < 2; ++word) {
            op.outputs.push_back(ValueId{static_cast<std::uint32_t>(p.ir.values.size())});
            p.ir.values.push_back({});
        }
        block.operations.push_back(OperationId{static_cast<std::uint32_t>(p.ir.operations.size())});
        p.ir.operations.push_back(std::move(op));
    }
    p.ir.blocks.push_back(block);
    std::uint64_t steps = 0;
    const auto plan = plan_matrices(p, [&] { ++steps; });
    check(plan.multiplies.size() == count && steps >= 5 * count, "Matrix planning charges each operation of each pass");
    struct Spent {};
    std::uint64_t left = 10;
    bool stopped = false;
    try {
        (void)plan_matrices(p, [&] { if (!left--) throw Spent{}; });
    } catch (const Spent&) { stopped = true; }
    check(stopped, "A spent analysis budget stops matrix planning");
}

int main(int argc, char** argv) {
    if (argc > 1) { fixture_directory = argv[1]; std::filesystem::create_directories(fixture_directory); }
    if (argc > 2) compile_metal = std::string(argv[2]) == "--metal";
    basic_tests(); depth_bias_tests(); io_tests(); memory_tests(); texture_tests(); custom_border_vertex_tests(); argument_buffer_tests(); shadow_pipeline_tests(); expanded_tests(); review_regressions(); flow_tests(); definedness_tests(); decoded_definedness_tests();
    matrix_plan_budget_tests();
    if (failures) std::cerr << failures << " MSL stage tests failed\n";
    else std::cout << "MSL stage tests passed\n";
    return failures ? 1 : 0;
}
