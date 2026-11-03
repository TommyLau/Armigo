// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "test_support.h"
#include "graph_backend_fixture.h"

#include <armigo/shader/graphics_msl.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace armigo::shader::test {
namespace {

namespace gx = graphics;
namespace gm = graphics_msl;
using Stage = shader_input::Stage;

const gm::EmittedStage* accepted(const gm::EmitResult& result) {
    const auto* stage = std::get_if<gm::EmittedStage>(&result);
    ARMIGO_CHECK(stage != nullptr);
    return stage;
}
void rejected(const gm::EmitResult& result, gm::EmitError::Code code) {
    const auto* error = std::get_if<gm::EmitError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error) ARMIGO_CHECK(error->code == code);
}
bool contains(const std::string& text, const std::string& value) {
    return text.find(value) != std::string::npos;
}
bool ordered(const std::string& text, const std::string& first,
             const std::string& second) {
    const auto first_at = text.find(first);
    const auto second_at = text.find(second);
    return first_at != std::string::npos && second_at != std::string::npos &&
           first_at < second_at;
}
// True when `statement` appears inside the body of the first guard whose
// opening text is `guard`, tracked by brace depth, so a statement hoisted
// after the body — including one hidden inside a later guard with the same
// text — does not satisfy the check. Unit assertion only; no source parser.
bool guard_contains(const std::string& text, const std::string& guard,
                    const std::string& statement) {
    const auto open = text.find(guard);
    if (open == std::string::npos) return false;
    const auto body_open = text.find('{', open);
    if (body_open == std::string::npos) return false;
    for (std::size_t at = body_open + 1, depth = 1; at < text.size(); ++at) {
        if (text[at] == '{') {
            ++depth;
        } else if (text[at] == '}') {
            if (--depth == 0) return false;
        } else if (text.compare(at, statement.size(), statement) == 0) {
            return true;
        }
    }
    return false;
}

void put_sph_header(gx::GraphSpec& spec, Stage stage) {
    const auto header = shader_input::sph_stage_type_for(stage) << 10;
    for (unsigned byte = 0; byte < 4; ++byte) {
        spec.sph[byte] = std::byte(header >> (8 * byte));
    }
}

// Independent synthetic graphs assembled directly through the typed
// constructor; no decoder output is involved. fragment_input0 is the raw SPH
// interpolation byte for generic input vector 0 (two bits per lane).
std::optional<gx::Program> make_program(
    Stage stage, std::vector<gx::Node> nodes, std::vector<gx::Export> exports,
    std::vector<gx::BasicBlock> blocks, std::uint8_t fragment_input0 = 0) {
    gx::GraphSpec spec;
    spec.stage = stage;
    put_sph_header(spec, stage);
    if (stage == Stage::Vertex) {
        spec.sph[0x18] = std::byte{0x01};  // generic input vector 0 declared
        spec.sph[0x35] = std::byte{0xf0};  // position outputs declared
    } else {
        spec.sph[0x18] = std::byte{fragment_input0};
        spec.sph[0x48] = std::byte{0x0f};  // color target 0 RGBA declared
    }
    spec.nodes = std::move(nodes);
    spec.exports = std::move(exports);
    spec.blocks = std::move(blocks);
    const auto result = gx::make_graph(spec);
    const auto* program = std::get_if<gx::Program>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return std::nullopt;
    return std::move(*program);
}

// Diamond: entry compares an attribute word against zero, the arms carry one
// scalar and one predicate definition each, and the join selects both with
// predecessor-labelled phis.
std::optional<gx::Program> diamond_program() {
    std::vector<gx::Node> nodes{
        {8, gx::AttributeLoad{32}},
        {16, gx::Literal{0}},
        {24, gx::WordNotEqual{gx::ScalarId{0}, gx::ScalarId{1}}},
        {32, gx::Literal{0x3f800000u}},
        {40, gx::PredicateLiteral{true}},
        {48, gx::Literal{0x40400000u}},
        {56, gx::PredicateNot{gx::PredicateId{2}}},
        {64, gx::ScalarPhi{{
                 {gx::BlockId{1}, gx::ScalarId{3}},
                 {gx::BlockId{2}, gx::ScalarId{5}}}}},
        {72, gx::PredicatePhi{{
                 {gx::BlockId{1}, gx::PredicateId{4}},
                 {gx::BlockId{2}, gx::PredicateId{6}}}}},
    };
    std::vector<gx::Export> exports;
    for (unsigned lane = 0; lane < 4; ++lane) {
        exports.push_back({gx::OutputKind::Position, static_cast<std::uint16_t>(lane),
                           gx::ScalarId{7}});
    }
    std::vector<gx::BasicBlock> blocks{
        {0, 3, gx::BranchIf{gx::PredicateId{2}, gx::BlockId{1}, gx::BlockId{2}}},
        {3, 5, gx::Jump{gx::BlockId{3}}},
        {5, 7, gx::Jump{gx::BlockId{3}}},
        {7, 9, gx::Return{}},
    };
    return make_program(Stage::Vertex, std::move(nodes), std::move(exports),
                        std::move(blocks));
}

// Fragment diamond with sampling: entry derives the branch predicate from a
// constant-buffer word and fixes both sample coordinates, the taken arm
// samples handle offset 32 and extracts its red and green lanes, the join
// samples handle offset 36 and extracts its blue and alpha lanes, and scalar
// phis carry the arm lanes into the exports.
std::optional<gx::Program> branched_sample_program(
    std::uint32_t arm_handle = 32, std::uint32_t join_handle = 36) {
    std::vector<gx::Node> nodes{
        {8, gx::ConstantLoad{0, 0}},
        {16, gx::Literal{0}},
        {24, gx::WordNotEqual{gx::ScalarId{0}, gx::ScalarId{1}}},
        {32, gx::Literal{0x3e800000u}},  // 0.25f u coordinate
        {40, gx::Literal{0x3e800000u}},  // 0.25f v coordinate
        {48, gx::TextureSample{gx::ScalarId{3}, gx::ScalarId{4}, arm_handle, true}},
        {56, gx::SampleComponent{gx::SampleId{5}, 0}},
        {64, gx::SampleComponent{gx::SampleId{5}, 1}},
        {72, gx::Literal{0x3f800000u}},
        {80, gx::Literal{0x00000000u}},
        {88, gx::ScalarPhi{{
                 {gx::BlockId{1}, gx::ScalarId{6}},
                 {gx::BlockId{2}, gx::ScalarId{8}}}}},
        {96, gx::ScalarPhi{{
                 {gx::BlockId{1}, gx::ScalarId{7}},
                 {gx::BlockId{2}, gx::ScalarId{9}}}}},
        {104, gx::Literal{0x3f400000u}},  // 0.75f u coordinate
        {112, gx::Literal{0x3e800000u}},  // 0.25f v coordinate
        {120, gx::TextureSample{gx::ScalarId{12}, gx::ScalarId{13}, join_handle, false}},
        {128, gx::SampleComponent{gx::SampleId{14}, 2}},
        {136, gx::SampleComponent{gx::SampleId{14}, 3}},
    };
    std::vector<gx::Export> exports{
        {gx::OutputKind::Color, 0, gx::ScalarId{10}},
        {gx::OutputKind::Color, 1, gx::ScalarId{11}},
        {gx::OutputKind::Color, 2, gx::ScalarId{15}},
        {gx::OutputKind::Color, 3, gx::ScalarId{16}},
    };
    std::vector<gx::BasicBlock> blocks{
        {0, 5, gx::BranchIf{gx::PredicateId{2}, gx::BlockId{1}, gx::BlockId{2}}},
        {5, 8, gx::Jump{gx::BlockId{3}}},
        {8, 10, gx::Jump{gx::BlockId{3}}},
        {10, 17, gx::Return{}},
    };
    return make_program(Stage::Fragment, std::move(nodes), std::move(exports),
                        std::move(blocks));
}

// One-block fragment graph: generic input vector 0 lane x is loaded directly
// and exported on every color channel. No Interpolate node exists, so the SPH
// interpolation byte is the only mode source for the input.
std::optional<gx::Program> direct_input_program(std::uint8_t interpolation_modes) {
    std::vector<gx::Export> exports;
    for (unsigned lane = 0; lane < 4; ++lane) {
        exports.push_back({gx::OutputKind::Color, static_cast<std::uint16_t>(lane),
                           gx::ScalarId{0}});
    }
    return make_program(Stage::Fragment,
                        std::vector<gx::Node>{{8, gx::AttributeLoad{32}}},
                        std::move(exports),
                        std::vector<gx::BasicBlock>{{0, 1, gx::Return{}}},
                        interpolation_modes);
}

}  // namespace

ARMIGO_TEST(graphics_control_flow, msl_emits_single_block_graph) {
    const auto result = backend_graph_fixture();
    const auto* program = std::get_if<gx::Program>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return;
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const gm::EmitResult output = gm::emit(*program, options);
    const gm::EmittedStage* stage = accepted(output);
    if (!stage) return;
    ARMIGO_CHECK(stage->entry_point == "armigo_vertex_main");
    ARMIGO_CHECK(stage->vertex_depth == gm::VertexDepthConvention::NegativeOneToOne);
    ARMIGO_CHECK(stage->interface.position_output_mask == 0xf);
    ARMIGO_CHECK(stage->constant_buffers.empty());
    ARMIGO_CHECK(stage->textures.empty());
    // The entry block executes under its own activity guard.
    ARMIGO_CHECK(contains(stage->source, "bool a0 = true;"));
    ARMIGO_CHECK(contains(stage->source, "if (a0) {"));
    ARMIGO_CHECK(contains(stage->source, "v0 = 0x3f800000u;"));
    ARMIGO_CHECK(contains(stage->source, "out.position.x = as_type<float>(v0);"));
    ARMIGO_CHECK(contains(stage->source, "armigo_depth_sum = out.position.z + out.position.w"));
    ARMIGO_CHECK(contains(stage->source, "return out;"));
    ARMIGO_CHECK(ordered(stage->source, "if (a0) {", "v0 = 0x3f800000u;"));
    ARMIGO_CHECK(stage->source ==
                 std::get<gm::EmittedStage>(gm::emit(*program, options)).source);
}

ARMIGO_TEST(graphics_control_flow, msl_emits_branch_guards_and_edge_phis) {
    const auto program = diamond_program();
    if (!program) return;
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    options.vertex_attributes.push_back({0, 0, gm::VertexFormat::Float32x3});
    const gm::EmitResult output = gm::emit(*program, options);
    const gm::EmittedStage* stage = accepted(output);
    if (!stage) return;
    // Typed storage: predicate and scalar identities use distinct types.
    ARMIGO_CHECK(contains(stage->source, "[[maybe_unused]] bool p2 = false;"));
    ARMIGO_CHECK(contains(stage->source, "[[maybe_unused]] uint v7 = 0u;"));
    ARMIGO_CHECK(contains(stage->source, "[[maybe_unused]] bool p8 = false;"));
    // The branch is a real guard on the predicate value, selecting exactly one
    // successor flag per invocation.
    ARMIGO_CHECK(contains(stage->source, "if (p2) {"));
    ARMIGO_CHECK(contains(stage->source, "a1 = true;"));
    ARMIGO_CHECK(contains(stage->source, "a2 = true;"));
    // Every block body executes only under its activity guard.
    ARMIGO_CHECK(ordered(stage->source, "if (a1) {", "v3 = 0x3f800000u;"));
    ARMIGO_CHECK(ordered(stage->source, "if (a2) {", "v5 = 0x40400000u;"));
    ARMIGO_CHECK(ordered(stage->source, "if (a2) {", "p6 = !p2;"));
    // Phi values are assigned on the taken predecessor edge, before the join
    // region runs; the join itself computes nothing.
    ARMIGO_CHECK(ordered(stage->source, "if (a1) {", "v7 = v3;"));
    ARMIGO_CHECK(ordered(stage->source, "v7 = v3;", "if (a3) {"));
    ARMIGO_CHECK(ordered(stage->source, "if (a1) {", "p8 = p4;"));
    ARMIGO_CHECK(ordered(stage->source, "if (a2) {", "v7 = v5;"));
    ARMIGO_CHECK(ordered(stage->source, "if (a2) {", "p8 = p6;"));
    ARMIGO_CHECK(!contains(stage->source, "select("));
    ARMIGO_CHECK(contains(stage->source, "out.position.x = as_type<float>(v7);"));
    ARMIGO_CHECK(stage->source ==
                 std::get<gm::EmittedStage>(gm::emit(*program, options)).source);
}

ARMIGO_TEST(graphics_control_flow, msl_emits_graph_texture_samples) {
    const auto program = branched_sample_program();
    if (!program) return;
    const gm::EmitResult output = gm::emit(*program);
    const gm::EmittedStage* stage = accepted(output);
    if (!stage) return;
    // Reflection is identical to straight-line emission: unique handle
    // offsets sorted, one texture and one sampler slot per offset.
    ARMIGO_CHECK(stage->textures.size() == 2);
    if (stage->textures.size() != 2) return;
    ARMIGO_CHECK(stage->textures[0].handle_byte_offset == 32);
    ARMIGO_CHECK(stage->textures[0].texture_index == 0);
    ARMIGO_CHECK(stage->textures[0].sampler_index == 0);
    ARMIGO_CHECK(stage->textures[1].handle_byte_offset == 36);
    ARMIGO_CHECK(stage->textures[1].texture_index == 1);
    ARMIGO_CHECK(stage->textures[1].sampler_index == 1);
    ARMIGO_CHECK(contains(stage->source, "texture2d<float> tex1 [[texture(1)]]"));
    ARMIGO_CHECK(stage->constant_buffers.size() == 1);
    if (stage->constant_buffers.size() != 1) return;
    ARMIGO_CHECK(stage->constant_buffers[0].guest_binding == 0);
    ARMIGO_CHECK(stage->constant_buffers[0].metal_buffer_index == 0);
    ARMIGO_CHECK(stage->constant_buffers[0].minimum_size == 4);
    // Typed storage: samples declare float4 tuple locals, scalars stay uint.
    ARMIGO_CHECK(contains(stage->source, "[[maybe_unused]] float4 s5 = float4(0.0f);"));
    ARMIGO_CHECK(contains(stage->source, "[[maybe_unused]] float4 s14 = float4(0.0f);"));
    ARMIGO_CHECK(contains(stage->source, "[[maybe_unused]] uint v10 = 0u;"));
    ARMIGO_CHECK(!contains(stage->source, "uint v5 ="));
    ARMIGO_CHECK(!contains(stage->source, "uint v14 ="));
    // The arm sample and its component extraction sit at their emitted
    // location inside the arm guard; the untaken arm emits no sample text.
    ARMIGO_CHECK(guard_contains(stage->source, "if (a1) {",
        "s5 = tex0.sample(samp0, float2(as_type<float>(v3), as_type<float>(v4)));"));
    ARMIGO_CHECK(guard_contains(stage->source, "if (a1) {",
                 "v6 = as_type<uint>(s5[0]);"));
    ARMIGO_CHECK(guard_contains(stage->source, "if (a1) {",
                 "v7 = as_type<uint>(s5[1]);"));
    ARMIGO_CHECK(!guard_contains(stage->source, "if (a2) {", "tex0.sample("));
    // The join sample sits inside the join guard and extracts its lanes there.
    ARMIGO_CHECK(guard_contains(stage->source, "if (a3) {",
        "s14 = tex1.sample(samp1, float2(as_type<float>(v12), as_type<float>(v13)));"));
    ARMIGO_CHECK(guard_contains(stage->source, "if (a3) {",
                 "v15 = as_type<uint>(s14[2]);"));
    ARMIGO_CHECK(guard_contains(stage->source, "if (a3) {",
                 "v16 = as_type<uint>(s14[3]);"));
    // Program sample order is preserved and no eager select replaces guards.
    ARMIGO_CHECK(ordered(stage->source, "s5 = tex0.sample(", "s14 = tex1.sample("));
    ARMIGO_CHECK(!contains(stage->source, "select("));
    // Phi lanes from the sampled arm still arrive on the predecessor edge.
    ARMIGO_CHECK(guard_contains(stage->source, "if (a1) {", "v10 = v6;"));
    ARMIGO_CHECK(guard_contains(stage->source, "if (a1) {", "v11 = v7;"));
    ARMIGO_CHECK(guard_contains(stage->source, "if (a2) {", "v10 = v8;"));
    ARMIGO_CHECK(guard_contains(stage->source, "if (a2) {", "v11 = v9;"));
    ARMIGO_CHECK(contains(stage->source, "out.color.w = as_type<float>(v16);"));
    ARMIGO_CHECK(stage->source ==
                 std::get<gm::EmittedStage>(gm::emit(*program)).source);
}

ARMIGO_TEST(graphics_control_flow, msl_graph_texture_reflection_matches_straight_line) {
    // Reverse encounter order and repeated handles exercise sorting and
    // deduplication through both emission paths with the same sample metadata.
    for (const auto handles : {std::array{36u, 32u}, std::array{32u, 32u}}) {
        const auto graph = branched_sample_program(handles[0], handles[1]);
        if (!graph) return;
        maxwell::DecodedProgram decoded;
        decoded.stage = Stage::Fragment;
        decoded.sph = graph->sph();
        const auto append = [&](maxwell::OpcodeFamily family,
                                maxwell::InstructionPayload payload) {
            const auto index = decoded.instructions.size();
            decoded.instructions.push_back({
                static_cast<std::uint32_t>((index / 3) * 32 + (index % 3 + 1) * 8),
                0, family, {7, false}, std::move(payload)});
        };
        for (const auto& node : graph->nodes()) {
            const auto* sample = std::get_if<gx::TextureSample>(&node.operation);
            if (!sample) continue;
            const auto x = std::get<gx::Literal>(graph->nodes()[sample->x.index].operation);
            const auto y = std::get<gx::Literal>(graph->nodes()[sample->y.index].operation);
            append(maxwell::OpcodeFamily::Mov32Immediate,
                   maxwell::Mov32Immediate{{4}, 15, x.bits});
            append(maxwell::OpcodeFamily::Mov32Immediate,
                   maxwell::Mov32Immediate{{5}, 15, y.bits});
            append(maxwell::OpcodeFamily::TextureSample,
                   maxwell::TextureSample2D{{0}, {2}, {4}, {5},
                                           sample->handle_byte_offset, sample->no_dependency});
        }
        append(maxwell::OpcodeFamily::Exit, maxwell::Exit{15});
        const auto lowered = gx::lower(decoded);
        const auto* straight = std::get_if<gx::Program>(&lowered);
        ARMIGO_CHECK(straight != nullptr);
        if (!straight) return;
        const auto graph_output = gm::emit(*graph);
        const auto straight_output = gm::emit(*straight);
        const auto* graph_stage = accepted(graph_output);
        const auto* straight_stage = accepted(straight_output);
        if (!graph_stage || !straight_stage) return;
        ARMIGO_CHECK(graph_stage->textures.size() == straight_stage->textures.size());
        if (graph_stage->textures.size() != straight_stage->textures.size()) return;
        ARMIGO_CHECK(graph_stage->textures.size() == (handles[0] == handles[1] ? 1u : 2u));
        for (std::size_t index = 0; index < graph_stage->textures.size(); ++index) {
            const auto& actual = graph_stage->textures[index];
            const auto& expected = straight_stage->textures[index];
            ARMIGO_CHECK(actual.handle_byte_offset == expected.handle_byte_offset);
            ARMIGO_CHECK(actual.texture_index == expected.texture_index);
            ARMIGO_CHECK(actual.sampler_index == expected.sampler_index);
        }
        gm::EmitOptions limited;
        limited.max_texture_resources = 0;
        rejected(gm::emit(*graph, limited), gm::EmitError::Code::ResourceLimit);
    }
}

ARMIGO_TEST(graphics_control_flow, msl_requires_graph_profile_options) {
    const auto result = backend_graph_fixture();
    const auto* program = std::get_if<gx::Program>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return;
    // A vertex graph without an explicit clip-depth convention is a profile
    // rejection, exactly as for straight-line emission.
    rejected(gm::emit(*program), gm::EmitError::Code::DepthConventionRequired);
}

ARMIGO_TEST(graphics_control_flow, msl_derives_fragment_sph_interpolation) {
    // Without an Interpolate node, the emitted qualifier and the reflected
    // interface mode come from the actual SPH lane modes: flat, perspective,
    // screen-linear, and all-four-lanes flat.
    struct Case { std::uint8_t modes; const char* field; gx::Interpolation mode; };
    const std::array<Case, 4> cases{{
        {0x01, "float4 generic0 [[user(locn0), flat]];", gx::Interpolation::Constant},
        {0x02, "float4 generic0 [[user(locn0)]];", gx::Interpolation::Perspective},
        {0x03, "float4 generic0 [[user(locn0), center_no_perspective]];",
         gx::Interpolation::ScreenLinear},
        {0x55, "float4 generic0 [[user(locn0), flat]];", gx::Interpolation::Constant},
    }};
    for (const auto& selected : cases) {
        const auto program = direct_input_program(selected.modes);
        if (!program) return;
        const gm::EmitResult output = gm::emit(*program);
        const gm::EmittedStage* stage = accepted(output);
        if (!stage) return;
        ARMIGO_CHECK(stage->interface.generic_inputs.size() == 1);
        if (stage->interface.generic_inputs.size() != 1) return;
        ARMIGO_CHECK(stage->interface.generic_inputs[0].slot == 0);
        ARMIGO_CHECK(stage->interface.generic_inputs[0].component_mask == 0x1);
        ARMIGO_CHECK(stage->interface.generic_inputs[0].interpolation == selected.mode);
        ARMIGO_CHECK(contains(stage->source, selected.field));
        ARMIGO_CHECK(stage->source ==
                     std::get<gm::EmittedStage>(gm::emit(*program)).source);
    }
}

ARMIGO_TEST(graphics_control_flow, msl_rejects_mixed_sph_component_modes) {
    // Lane x declares perspective and lane y flat inside one vector slot. One
    // MSL qualifier covers the whole float4, so emission rejects the
    // combination explicitly instead of choosing one component's mode for the
    // rest of the vector.
    const auto program = direct_input_program(0x06);
    if (!program) return;
    rejected(gm::emit(*program), gm::EmitError::Code::UnsupportedInterface);
}

}  // namespace armigo::shader::test
