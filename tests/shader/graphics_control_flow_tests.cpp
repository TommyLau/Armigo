// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>
#include "test_support.h"

#include <array>
#include <utility>
#include <variant>

namespace armigo::shader::test {
namespace {
namespace gx = graphics;
namespace gm = graphics_msl;
using Stage = shader_input::Stage;
using Code = gx::Error::Code;
using MslCode = gm::EmitError::Code;
using gx::BlockId;
using gx::PredicateId;
using gx::ScalarId;

// Synthetic node ids used by the shared diamond fixture below.
enum NodeSlot {
    WordA = 0, WordB = 1, WordsDiffer = 2, Invert = 3,
    LeftWord = 4, LeftFlag = 5, RightWord = 6, RightFlag = 7,
    JoinWord = 8, JoinFlag = 9,
};

// Synthetic node ids for the fragment sampling fixture.
enum FragmentSlot {
    CoordX = 0, CoordY = 1, SceneSample = 2,
    RedLane = 3, GreenLane = 4, BlueLane = 5, AlphaLane = 6,
};

void put_sph_word(maxwell::DecodedProgram& program, unsigned at,
                  std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        program.sph[at + i] = std::byte(value >> (i * 8));
    }
}

void put_sph_word(std::array<std::byte, 80>& sph, unsigned at,
                  std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        sph[at + i] = std::byte(value >> (i * 8));
    }
}

// Valid Vertex header declaring all four Position components, matching the
// interface contract the legacy Builder enforces on lowered programs.
void put_vertex_sph(gx::GraphSpec& spec) {
    put_sph_word(spec.sph, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    spec.sph[0x35] = std::byte(0xf0);
}

// Valid Fragment header declaring all four Color components.
void put_fragment_sph(gx::GraphSpec& spec) {
    put_sph_word(spec.sph, 0, shader_input::sph_stage_type_for(Stage::Fragment) << 10);
    put_sph_word(spec.sph, 0x48, 0xf);
}

// Valid Fragment header that additionally declares the Position.x input so a
// node can interpolate it.
void put_fragment_input_sph(gx::GraphSpec& spec) {
    put_fragment_sph(spec);
    spec.sph[0x17] = std::byte(1 << 4);
}

gx::Node node(std::uint32_t pc, gx::Operation operation) {
    return {pc, std::move(operation)};
}

// Diamond: entry compares two words, inverted condition picks one of two
// (word, predicate) pairs, and the join merges both with typed phis.
gx::GraphSpec diamond() {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_vertex_sph(spec);
    spec.nodes.push_back(node(8, gx::Literal{0x3f800000}));
    spec.nodes.push_back(node(16, gx::Literal{1}));
    spec.nodes.push_back(node(24, gx::WordNotEqual{ScalarId{WordA}, ScalarId{WordB}}));
    spec.nodes.push_back(node(32, gx::PredicateNot{PredicateId{WordsDiffer}}));
    spec.nodes.push_back(node(40, gx::Literal{10}));
    spec.nodes.push_back(node(48, gx::PredicateLiteral{true}));
    spec.nodes.push_back(node(56, gx::Literal{20}));
    spec.nodes.push_back(node(64, gx::PredicateLiteral{false}));
    spec.nodes.push_back(node(72, gx::ScalarPhi{{
        {{1}, {LeftWord}}, {{2}, {RightWord}}}}));
    spec.nodes.push_back(node(80, gx::PredicatePhi{{
        {{1}, {LeftFlag}}, {{2}, {RightFlag}}}}));
    spec.blocks.push_back({0, 4,
        gx::BranchIf{PredicateId{WordsDiffer}, BlockId{1}, BlockId{2}}});
    spec.blocks.push_back({4, 6, gx::Jump{{3}}});
    spec.blocks.push_back({6, 8, gx::Jump{{3}}});
    spec.blocks.push_back({8, 10, gx::Return{}});
    for (std::uint16_t component = 0; component < 4; ++component) {
        spec.exports.push_back({gx::OutputKind::Position, component, {JoinWord}});
    }
    return spec;
}

// Straight-line fragment program: samples a synthetic 2D texture and exports
// one Color component per sampled lane.
gx::GraphSpec fragment_sample() {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    put_fragment_sph(spec);
    spec.nodes.push_back(node(8, gx::Literal{0x3f800000}));
    spec.nodes.push_back(node(16, gx::Literal{0x40000000}));
    spec.nodes.push_back(node(24, gx::TextureSample{
        {CoordX}, {CoordY}, 0, false}));
    spec.nodes.push_back(node(32, gx::SampleComponent{{SceneSample}, 0}));
    spec.nodes.push_back(node(40, gx::SampleComponent{{SceneSample}, 1}));
    spec.nodes.push_back(node(48, gx::SampleComponent{{SceneSample}, 2}));
    spec.nodes.push_back(node(56, gx::SampleComponent{{SceneSample}, 3}));
    spec.blocks.push_back({0, 7, gx::Return{}});
    spec.exports.push_back({gx::OutputKind::Color, 0, {RedLane}});
    spec.exports.push_back({gx::OutputKind::Color, 1, {GreenLane}});
    spec.exports.push_back({gx::OutputKind::Color, 2, {BlueLane}});
    spec.exports.push_back({gx::OutputKind::Color, 3, {AlphaLane}});
    return spec;
}

// Straight-line Vertex program exporting Position from node 1, with node 0
// available for well-defined secondary exports. `operation` is node 1.
gx::GraphSpec vertex_with_operation(gx::Operation operation) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_vertex_sph(spec);
    spec.nodes.push_back(node(8, gx::Literal{0x3f800000}));
    spec.nodes.push_back(node(16, std::move(operation)));
    spec.blocks.push_back({0, 2, gx::Return{}});
    spec.exports.push_back({gx::OutputKind::Position, 0, {1}});
    for (std::uint16_t component = 1; component < 4; ++component) {
        spec.exports.push_back({gx::OutputKind::Position, component, {0}});
    }
    return spec;
}

const gx::Program* as_program(const gx::GraphResult& result) {
    return std::get_if<gx::Program>(&result);
}

const gx::Error* as_error(const gx::GraphResult& result) {
    return std::get_if<gx::Error>(&result);
}

// Operation metadata may map to either diagnostic class; the constructor must
// reject it rather than admit it into a validated Program.
void check_op_metadata_rejection(const gx::GraphResult& result) {
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr);
    if (error == nullptr) return;
    const auto code = static_cast<unsigned>(error->code);
    ARMIGO_CHECK(code == static_cast<unsigned>(Code::UnsupportedModifier) ||
                 code == static_cast<unsigned>(Code::InvalidInstruction));
}

}  // namespace

ARMIGO_TEST(graphics_control_flow, valid_diamond) {
    const auto result = gx::make_graph(diamond());
    const auto* program = as_program(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    ARMIGO_CHECK_EQ(static_cast<unsigned>(program->stage()),
                    static_cast<unsigned>(Stage::Vertex));
    ARMIGO_CHECK(!program->blocks().empty());
    const auto entry = program->entry();
    ARMIGO_CHECK(entry.has_value());
    ARMIGO_CHECK_EQ(entry->index, 0u);
    ARMIGO_CHECK_EQ(program->blocks().size(), 4u);
    ARMIGO_CHECK_EQ(static_cast<unsigned>(program->nodes()[WordsDiffer].type()),
                    static_cast<unsigned>(gx::ValueType::Predicate));
    ARMIGO_CHECK_EQ(static_cast<unsigned>(program->nodes()[JoinFlag].type()),
                    static_cast<unsigned>(gx::ValueType::Predicate));
    ARMIGO_CHECK_EQ(static_cast<unsigned>(program->nodes()[JoinWord].type()),
                    static_cast<unsigned>(gx::ValueType::Raw32));
    ARMIGO_CHECK_EQ(program->exports().size(), 4u);
    ARMIGO_CHECK(std::holds_alternative<gx::Return>(
        program->blocks()[3].terminator));
}

ARMIGO_TEST(graphics_control_flow, fragment_sample_admitted) {
    const auto result = gx::make_graph(fragment_sample());
    const auto* program = as_program(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    ARMIGO_CHECK_EQ(static_cast<unsigned>(program->stage()),
                    static_cast<unsigned>(Stage::Fragment));
    ARMIGO_CHECK_EQ(program->blocks().size(), 1u);
    ARMIGO_CHECK_EQ(program->exports().size(), 4u);
}

ARMIGO_TEST(graphics_control_flow, straight_line_graph) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_vertex_sph(spec);
    spec.nodes.push_back(node(8, gx::Literal{7}));
    spec.blocks.push_back({0, 1, gx::Return{}});
    for (std::uint16_t component = 0; component < 4; ++component) {
        spec.exports.push_back({gx::OutputKind::Position, component, {0}});
    }
    const auto result = gx::make_graph(spec);
    const auto* program = as_program(result);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    ARMIGO_CHECK_EQ(program->blocks().size(), 1u);
    const auto entry = program->entry();
    ARMIGO_CHECK(entry.has_value());
    ARMIGO_CHECK_EQ(entry->index, 0u);
}

ARMIGO_TEST(graphics_control_flow, legacy_lowering_keeps_empty_blocks) {
    maxwell::DecodedProgram decoded;
    decoded.stage = Stage::Vertex;
    put_sph_word(decoded, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    decoded.sph[0x35] |= std::byte(1 << 4);
    const auto index = decoded.instructions.size();
    decoded.instructions.push_back({
        static_cast<std::uint32_t>((index / 3) * 32 + (index % 3 + 1) * 8),
        0, maxwell::OpcodeFamily::Mov32Immediate,
        {7, false}, maxwell::Mov32Immediate{{1}, 15, 0x3f800000}});
    const auto second = decoded.instructions.size();
    decoded.instructions.push_back({
        static_cast<std::uint32_t>((second / 3) * 32 + (second % 3 + 1) * 8),
        0, maxwell::OpcodeFamily::AttributeStore,
        {7, false}, maxwell::AttributeAccess{{1}, {255}, {255}, 28 * 4, 0, false}});
    const auto third = decoded.instructions.size();
    decoded.instructions.push_back({
        static_cast<std::uint32_t>((third / 3) * 32 + (third % 3 + 1) * 8),
        0, maxwell::OpcodeFamily::Exit, {7, false}, maxwell::Exit{15}});
    const auto lowered = gx::lower(decoded);
    const auto* program = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    ARMIGO_CHECK(program->blocks().empty());
    ARMIGO_CHECK(!program->entry().has_value());
    // The legacy path still executes.
    const gx::Inputs inputs;
    const auto evaluated = gx::evaluate(*program, inputs);
    ARMIGO_CHECK(std::holds_alternative<gx::Evaluation>(evaluated));
}

ARMIGO_TEST(graphics_control_flow, rejects_missing_blocks) {
    auto spec = diamond();
    spec.blocks.clear();
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr);
    if (error == nullptr) return;
    ARMIGO_CHECK_EQ(static_cast<unsigned>(error->code),
                    static_cast<unsigned>(Code::GraphFormat));
}

ARMIGO_TEST(graphics_control_flow, rejects_unsupported_stage) {
    auto spec = diamond();
    spec.stage = Stage::Compute;
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr);
    if (error == nullptr) return;
    ARMIGO_CHECK_EQ(static_cast<unsigned>(error->code),
                    static_cast<unsigned>(Code::UnsupportedStage));
}

ARMIGO_TEST(graphics_control_flow, rejects_zero_sph) {
    auto spec = diamond();
    spec.sph.fill(std::byte{0});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::HeaderMismatch);
}

ARMIGO_TEST(graphics_control_flow, rejects_sph_stage_mismatch) {
    auto spec = diamond();
    put_sph_word(spec.sph, 0,
                 shader_input::sph_stage_type_for(Stage::Fragment) << 10);
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::HeaderMismatch);
}

ARMIGO_TEST(graphics_control_flow, rejects_missing_declared_export) {
    auto spec = diamond();
    spec.exports.clear();
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::MissingOutput);
}

ARMIGO_TEST(graphics_control_flow, rejects_partial_position_exports) {
    auto spec = diamond();
    spec.exports.resize(1);
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::MissingOutput);
}

ARMIGO_TEST(graphics_control_flow, rejects_duplicate_export) {
    auto spec = diamond();
    spec.exports.push_back({gx::OutputKind::Position, 0, {JoinWord}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphFormat);
}

ARMIGO_TEST(graphics_control_flow, rejects_vertex_color_export) {
    auto spec = diamond();
    spec.exports.push_back({gx::OutputKind::Color, 0, {JoinWord}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::UnsupportedInterface);
}

ARMIGO_TEST(graphics_control_flow, rejects_fragment_position_export) {
    auto spec = fragment_sample();
    spec.exports.push_back({gx::OutputKind::Position, 0, {RedLane}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::UnsupportedInterface);
}

ARMIGO_TEST(graphics_control_flow, rejects_position_component_out_of_range) {
    auto spec = diamond();
    spec.exports.push_back({gx::OutputKind::Position, 4, {JoinWord}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::UnsupportedInterface);
}

ARMIGO_TEST(graphics_control_flow, rejects_branch_in_empty_block) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_vertex_sph(spec);
    spec.blocks.push_back({0, 0,
        gx::BranchIf{PredicateId{0}, BlockId{1}, BlockId{1}}});
    spec.blocks.push_back({0, 0, gx::Return{}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphFormat);
}

ARMIGO_TEST(graphics_control_flow, rejects_entry_scalar_phi_without_arms) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_vertex_sph(spec);
    spec.nodes.push_back(node(8, gx::Literal{1}));
    spec.nodes.push_back(node(16, gx::ScalarPhi{}));
    spec.blocks.push_back({0, 2, gx::Return{}});
    spec.exports.push_back({gx::OutputKind::Position, 0, {1}});
    for (std::uint16_t component = 1; component < 4; ++component) {
        spec.exports.push_back({gx::OutputKind::Position, component, {0}});
    }
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphDataFlow);
}

ARMIGO_TEST(graphics_control_flow, rejects_entry_predicate_phi_without_arms) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_vertex_sph(spec);
    spec.nodes.push_back(node(8, gx::PredicatePhi{}));
    spec.nodes.push_back(node(16, gx::PredicateNot{PredicateId{0}}));
    spec.blocks.push_back({0, 2, gx::Return{}});
    for (std::uint16_t component = 0; component < 4; ++component) {
        spec.exports.push_back({gx::OutputKind::Position, component, {0}});
    }
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphDataFlow);
}

ARMIGO_TEST(graphics_control_flow, rejects_constant_load_binding_out_of_range) {
    const auto result = gx::make_graph(
        vertex_with_operation(gx::ConstantLoad{18, 0}));
    check_op_metadata_rejection(result);
}

ARMIGO_TEST(graphics_control_flow, rejects_constant_load_unaligned_offset) {
    const auto result = gx::make_graph(
        vertex_with_operation(gx::ConstantLoad{0, 2}));
    check_op_metadata_rejection(result);
}

ARMIGO_TEST(graphics_control_flow, rejects_fma_without_addend) {
    const auto result = gx::make_graph(vertex_with_operation(
        gx::FloatArithmetic{gx::Arithmetic::FusedMultiplyAdd,
                            {0}, {0}, {}, false}));
    check_op_metadata_rejection(result);
}

ARMIGO_TEST(graphics_control_flow, rejects_invalid_arithmetic_operation) {
    const auto result = gx::make_graph(vertex_with_operation(
        gx::FloatArithmetic{static_cast<gx::Arithmetic>(4),
                            {0}, {0}, {}, false}));
    check_op_metadata_rejection(result);
}

ARMIGO_TEST(graphics_control_flow, rejects_invalid_interpolation_mode) {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    put_fragment_input_sph(spec);
    spec.nodes.push_back(node(8, gx::AttributeLoad{28}));
    spec.nodes.push_back(node(16, gx::Interpolate{
        {0}, {}, static_cast<gx::Interpolation>(0), 0x7ff, {}}));
    spec.blocks.push_back({0, 2, gx::Return{}});
    for (std::uint16_t component = 0; component < 4; ++component) {
        spec.exports.push_back({gx::OutputKind::Color, component, {0}});
    }
    check_op_metadata_rejection(gx::make_graph(spec));
}

ARMIGO_TEST(graphics_control_flow, rejects_sample_component_lane_four) {
    auto spec = fragment_sample();
    spec.nodes[AlphaLane] = node(56, gx::SampleComponent{{SceneSample}, 4});
    check_op_metadata_rejection(gx::make_graph(spec));
}

ARMIGO_TEST(graphics_control_flow, rejects_range_gap) {
    auto spec = diamond();
    spec.blocks[1].begin = 5;
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphFormat);
}

ARMIGO_TEST(graphics_control_flow, rejects_range_overlap) {
    auto spec = diamond();
    spec.blocks[0].end = 7;
    spec.blocks[1].end = 8;
    spec.blocks[2].begin = 8;
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphFormat);
}

ARMIGO_TEST(graphics_control_flow, rejects_undefined_reference) {
    auto spec = diamond();
    spec.nodes[WordsDiffer] = node(24, gx::WordNotEqual{{WordA}, {999}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphFormat);
}

ARMIGO_TEST(graphics_control_flow, rejects_backward_target) {
    auto spec = diamond();
    spec.blocks[1] = {4, 6, gx::Jump{{0}}};
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphShape);
}

ARMIGO_TEST(graphics_control_flow, rejects_target_out_of_bounds) {
    auto spec = diamond();
    spec.blocks[1] = {4, 6, gx::Jump{{99}}};
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphFormat);
}

ARMIGO_TEST(graphics_control_flow, rejects_multiple_returns) {
    auto spec = diamond();
    spec.blocks[2] = {6, 8, gx::Return{}};
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphShape);
}

ARMIGO_TEST(graphics_control_flow, rejects_unreachable_block) {
    auto spec = diamond();
    spec.blocks[0].terminator = gx::BranchIf{PredicateId{WordsDiffer}, BlockId{3}, BlockId{3}};
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphShape);
}

ARMIGO_TEST(graphics_control_flow, rejects_block_limit) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_vertex_sph(spec);
    for (std::size_t index = 0; index <= gx::max_blocks; ++index) {
        const auto next = static_cast<std::uint32_t>(index + 1);
        spec.blocks.push_back({0, 0,
            index == gx::max_blocks ? gx::Terminator{gx::Return{}}
                                    : gx::Terminator{gx::Jump{{next}}}});
    }
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr && error->code == Code::Limit);
}

ARMIGO_TEST(graphics_control_flow, rejects_branch_condition_type) {
    auto spec = diamond();
    spec.blocks[0].terminator = gx::BranchIf{PredicateId{WordA}, BlockId{1}, BlockId{2}};
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphTypes);
}

ARMIGO_TEST(graphics_control_flow, rejects_predicate_operand_type) {
    auto spec = diamond();
    spec.nodes[Invert] = node(32, gx::PredicateNot{PredicateId{WordA}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphTypes);
}

ARMIGO_TEST(graphics_control_flow, rejects_phi_arm_type) {
    auto spec = diamond();
    spec.nodes[JoinFlag] = node(80, gx::PredicatePhi{{
        {{1}, {LeftWord}}, {{2}, {RightWord}}}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphTypes);
}

ARMIGO_TEST(graphics_control_flow, rejects_missing_phi_arm) {
    auto spec = diamond();
    spec.nodes[JoinWord] = node(72, gx::ScalarPhi{{{ {1}, {LeftWord} }}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphDataFlow);
}

ARMIGO_TEST(graphics_control_flow, rejects_duplicate_phi_arm) {
    auto spec = diamond();
    spec.nodes[JoinWord] = node(72, gx::ScalarPhi{{
        {{1}, {LeftWord}}, {{1}, {LeftWord}}}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphDataFlow);
}

ARMIGO_TEST(graphics_control_flow, rejects_wrong_phi_arm) {
    auto spec = diamond();
    spec.nodes[JoinWord] = node(72, gx::ScalarPhi{{
        {{0}, {WordA}}, {{2}, {RightWord}}}});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphDataFlow);
}

ARMIGO_TEST(graphics_control_flow, rejects_nondominating_use) {
    auto spec = diamond();
    // LeftWord reads RightWord's literal, defined only on the other arm.
    spec.nodes[LeftWord] = node(40, gx::WordAnd{{RightWord}, 0xff});
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphDataFlow);
}

ARMIGO_TEST(graphics_control_flow, rejects_export_not_dominating_return) {
    auto spec = diamond();
    spec.exports[0].value = {LeftWord};
    const auto result = gx::make_graph(spec);
    const auto* error = as_error(result);
    ARMIGO_CHECK(error != nullptr &&
                 error->code == Code::GraphDataFlow);
}



ARMIGO_TEST(graphics_control_flow, evaluate_still_accepts_straight_line) {
    // The graph guard must not disturb legacy straight-line programs.
    maxwell::DecodedProgram decoded;
    decoded.stage = Stage::Vertex;
    put_sph_word(decoded, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    decoded.sph[0x35] |= std::byte(1 << 4);
    const auto index = decoded.instructions.size();
    decoded.instructions.push_back({
        static_cast<std::uint32_t>((index / 3) * 32 + (index % 3 + 1) * 8),
        0, maxwell::OpcodeFamily::Mov32Immediate,
        {7, false}, maxwell::Mov32Immediate{{1}, 15, 0x3f800000}});
    const auto second = decoded.instructions.size();
    decoded.instructions.push_back({
        static_cast<std::uint32_t>((second / 3) * 32 + (second % 3 + 1) * 8),
        0, maxwell::OpcodeFamily::AttributeStore,
        {7, false}, maxwell::AttributeAccess{{1}, {255}, {255}, 28 * 4, 0, false}});
    const auto third = decoded.instructions.size();
    decoded.instructions.push_back({
        static_cast<std::uint32_t>((third / 3) * 32 + (third % 3 + 1) * 8),
        0, maxwell::OpcodeFamily::Exit, {7, false}, maxwell::Exit{15}});
    const auto legacy = gx::lower(decoded);
    const auto* program = std::get_if<gx::Program>(&legacy);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    const gx::Inputs inputs;
    const auto evaluated = gx::evaluate(*program, inputs);
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr);
    if (outputs == nullptr) return;
    ARMIGO_CHECK_EQ(outputs->outputs.size(), 1u);
    if (!outputs->outputs.empty()) {
        ARMIGO_CHECK_EQ(outputs->outputs[0].bits, 0x3f800000u);
    }
}

}  // namespace armigo::shader::test
