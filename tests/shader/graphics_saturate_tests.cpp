// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic typed-SSA saturation hook tests. Every expected bit below is
// authored from the saturate_float32 policy (a sign bit or NaN becomes +0,
// a value above one becomes 1) and that helper's contract; no decoder output and no
// implementation-derived oracle is involved. CPU and MSL agreement here
// verifies the shared Armigo contract only, never Maxwell hardware behavior.

#include "test_support.h"

#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "graphics_builder.h"
#include "graphics_families.h"

namespace armigo::shader::test {
namespace {

namespace gx = graphics;
namespace gm = graphics_msl;
using Stage = shader_input::Stage;
using gx::BlockId;
using gx::PredicateId;
using gx::ScalarId;

void put_sph_word(std::array<std::byte, 80>& sph, unsigned at,
                  std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) sph[at + i] = std::byte(value >> (i * 8));
}

gx::GraphSpec vertex_spec() {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_sph_word(spec.sph, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    spec.sph[0x18] = std::byte(0x01);  // generic input vector 0 declared
    spec.sph[0x35] = std::byte(0xf0);
    return spec;
}

// Fragment header declaring the first `lanes` color output lanes.
gx::GraphSpec fragment_spec(unsigned lanes) {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    put_sph_word(spec.sph, 0, shader_input::sph_stage_type_for(Stage::Fragment) << 10);
    const auto mask = static_cast<std::uint32_t>(
        lanes >= 32 ? ~std::uint32_t{0} : (std::uint32_t{1} << lanes) - 1);
    put_sph_word(spec.sph, 0x48, mask);
    return spec;
}

std::optional<gx::Program> build(gx::GraphSpec spec) {
    const auto result = gx::make_graph(spec);
    const auto* program = std::get_if<gx::Program>(&result);
    if (program == nullptr) {
        record_failure(__FILE__, __LINE__,
                       std::string("fixture graph rejected: ") +
                           std::get<gx::Error>(result).message());
    }
    if (program == nullptr) return std::nullopt;
    return std::move(*program);
}

const gm::EmittedStage* accepted(const gm::EmitResult& result) {
    const auto* stage = std::get_if<gm::EmittedStage>(&result);
    ARMIGO_CHECK(stage != nullptr);
    return stage;
}

bool contains(const std::string& text, const std::string& value) {
    return text.find(value) != std::string::npos;
}

std::size_t occurrences(const std::string& text, const std::string& value) {
    std::size_t count = 0;
    for (std::size_t at = text.find(value); at != std::string::npos;
         at = text.find(value, at + value.size())) {
        ++count;
    }
    return count;
}

std::size_t ordered_at(const std::string& text, const std::string& first) {
    return text.find(first);
}

void expect_bits(const gx::EvaluateResult& result, unsigned index,
                 std::uint32_t bits) {
    const auto* evaluation = std::get_if<gx::Evaluation>(&result);
    if (evaluation == nullptr) {
        record_failure(__FILE__, __LINE__, std::get<gx::Error>(result).message());
        return;
    }
    if (index >= evaluation->outputs.size()) {
        record_failure(__FILE__, __LINE__, "missing graphics output slot");
        return;
    }
    ARMIGO_CHECK_EQ(evaluation->outputs[index].bits, bits);
}

}  // namespace

// One block, one saturation: the constructor accepts the node and evaluation
// saturates the literal word exactly once.
ARMIGO_TEST(graphics_saturate, graph_accepts_and_evaluates_saturate) {
    auto spec = fragment_spec(1);
    spec.nodes.push_back({8, gx::Literal{0x40800000u}});  // 4.0f
    spec.nodes.push_back({16, gx::Saturate{ScalarId{0}}});
    spec.blocks.push_back({0, 2, gx::Return{}});
    spec.exports.push_back({gx::OutputKind::Color, 0, ScalarId{1}});
    const auto program = build(std::move(spec));
    if (!program) return;
    const auto result = gx::evaluate(*program, {});
    expect_bits(result, 0, 0x3f800000u);
}

// Operand checks: a predicate- or sample-typed operand is a type error, an
// out-of-range operand is a format error, and a forward reference is a
// data-flow error.
ARMIGO_TEST(graphics_saturate, graph_rejects_wrong_typed_saturate_operands) {
    auto predicate_operand = vertex_spec();
    predicate_operand.nodes.push_back({8, gx::PredicateLiteral{true}});
    predicate_operand.nodes.push_back({16, gx::Saturate{ScalarId{0}}});
    predicate_operand.blocks.push_back({0, 2, gx::Return{}});
    for (unsigned lane = 0; lane < 4; ++lane) {
        predicate_operand.exports.push_back(
            {gx::OutputKind::Position, static_cast<std::uint16_t>(lane), ScalarId{1}});
    }
    auto rejected = gx::make_graph(predicate_operand);
    ARMIGO_CHECK(std::holds_alternative<gx::Error>(rejected));
    if (std::holds_alternative<gx::Error>(rejected)) {
        ARMIGO_CHECK(std::get<gx::Error>(rejected).code == gx::Error::Code::GraphTypes);
    }

    auto sample_operand = fragment_spec(1);
    sample_operand.nodes.push_back({8, gx::Literal{0x3f800000u}});
    sample_operand.nodes.push_back({16, gx::Literal{0x40000000u}});
    sample_operand.nodes.push_back({24, gx::TextureSample{ScalarId{0}, ScalarId{1}, 0, true}});
    sample_operand.nodes.push_back({32, gx::Saturate{ScalarId{2}}});
    sample_operand.blocks.push_back({0, 4, gx::Return{}});
    sample_operand.exports.push_back({gx::OutputKind::Color, 0, ScalarId{3}});
    rejected = gx::make_graph(sample_operand);
    ARMIGO_CHECK(std::holds_alternative<gx::Error>(rejected));
    if (std::holds_alternative<gx::Error>(rejected)) {
        ARMIGO_CHECK(std::get<gx::Error>(rejected).code == gx::Error::Code::GraphTypes);
    }

    auto out_of_range = fragment_spec(1);
    out_of_range.nodes.push_back({8, gx::Saturate{ScalarId{99}}});
    out_of_range.blocks.push_back({0, 1, gx::Return{}});
    out_of_range.exports.push_back({gx::OutputKind::Color, 0, ScalarId{0}});
    rejected = gx::make_graph(out_of_range);
    ARMIGO_CHECK(std::holds_alternative<gx::Error>(rejected));
    if (std::holds_alternative<gx::Error>(rejected)) {
        ARMIGO_CHECK(std::get<gx::Error>(rejected).code == gx::Error::Code::GraphFormat);
    }

    auto forward = fragment_spec(1);
    forward.nodes.push_back({8, gx::Saturate{ScalarId{1}}});
    forward.nodes.push_back({16, gx::Literal{0x3f800000u}});
    forward.blocks.push_back({0, 2, gx::Return{}});
    forward.exports.push_back({gx::OutputKind::Color, 0, ScalarId{0}});
    rejected = gx::make_graph(forward);
    ARMIGO_CHECK(std::holds_alternative<gx::Error>(rejected));
    if (std::holds_alternative<gx::Error>(rejected)) {
        ARMIGO_CHECK(std::get<gx::Error>(rejected).code == gx::Error::Code::GraphDataFlow);
    }
}

// Authored bit edges of the documented policy, evaluated through one graph:
// every result word is written out literally, no oracle recomputes it.
ARMIGO_TEST(graphics_saturate, saturate_applies_authored_bit_edges) {
    struct Case { std::uint32_t input; std::uint32_t expected; };
    const std::array<Case, 13> cases{{
        {0x7fc00000u, 0x00000000u},  // quiet NaN, positive sign
        {0xffc00000u, 0x00000000u},  // quiet NaN, negative sign
        {0x7f800001u, 0x00000000u},  // NaN with smallest payload
        {0x80000000u, 0x00000000u},  // negative zero normalizes to +0
        {0x00000000u, 0x00000000u},  // +0 keeps its bits
        {0x7f800000u, 0x3f800000u},  // +infinity maps to one
        {0xff800000u, 0x00000000u},  // -infinity maps to +0
        {0x3f7fffffu, 0x3f7fffffu},  // largest word below one keeps its bits
        {0x3f800000u, 0x3f800000u},  // exactly one keeps its bits
        {0x3f800001u, 0x3f800000u},  // smallest word above one clamps
        {0xbf000000u, 0x00000000u},  // negative normal maps to +0
        {0x00800000u, 0x00800000u},  // minimum positive normal keeps its bits
        {0x00000001u, 0x00000001u},  // positive subnormal keeps its bits
    }};
    auto spec = fragment_spec(static_cast<unsigned>(cases.size()));
    for (unsigned index = 0; index < cases.size(); ++index) {
        spec.nodes.push_back({8 * (2 * index + 1), gx::Literal{cases[index].input}});
        spec.nodes.push_back({8 * (2 * index + 2), gx::Saturate{ScalarId{2 * index}}});
        spec.exports.push_back({gx::OutputKind::Color,
            static_cast<std::uint16_t>(index), ScalarId{2 * index + 1}});
    }
    spec.blocks.push_back({0, static_cast<std::uint32_t>(2 * cases.size()), gx::Return{}});
    const auto program = build(std::move(spec));
    if (!program) return;
    const auto result = gx::evaluate(*program, {});
    for (unsigned index = 0; index < cases.size(); ++index) {
        expect_bits(result, index, cases[index].expected);
    }
}

// The saturation node sits in a branch arm; each outcome selects the value
// produced by the arm actually taken.
ARMIGO_TEST(graphics_saturate, taken_and_untaken_arms_saturate_their_own_words) {
    auto spec = vertex_spec();
    spec.nodes.push_back({8, gx::AttributeLoad{32}});
    spec.nodes.push_back({16, gx::Literal{0}});
    spec.nodes.push_back({24, gx::WordNotEqual{ScalarId{0}, ScalarId{1}}});
    spec.nodes.push_back({40, gx::Literal{0x40800000u}});  // taken arm: 4.0f
    spec.nodes.push_back({48, gx::Saturate{ScalarId{3}}});
    spec.nodes.push_back({56, gx::PredicateLiteral{true}});
    spec.nodes.push_back({64, gx::Literal{0xbf800000u}});  // untaken arm: -1.5f
    spec.nodes.push_back({72, gx::Saturate{ScalarId{6}}});
    spec.nodes.push_back({80, gx::ScalarPhi{{
        {gx::BlockId{1}, gx::ScalarId{4}},
        {gx::BlockId{2}, gx::ScalarId{7}}}}});
    spec.blocks.push_back({0, 3,
        gx::BranchIf{PredicateId{2}, BlockId{1}, BlockId{2}}});
    spec.blocks.push_back({3, 6, gx::Jump{BlockId{3}}});
    spec.blocks.push_back({6, 8, gx::Jump{BlockId{3}}});
    spec.blocks.push_back({8, 9, gx::Return{}});
    for (unsigned lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Position,
            static_cast<std::uint16_t>(lane), ScalarId{8}});
    }
    const auto program = build(std::move(spec));
    if (!program) return;
    gx::Inputs taken;
    const std::array nonzero = {gx::AttributeInput{32, 5u}};
    taken.attributes = nonzero;
    expect_bits(gx::evaluate(*program, taken), 0, 0x3f800000u);
    gx::Inputs zero;
    const std::array zero_word = {gx::AttributeInput{32, 0u}};
    zero.attributes = zero_word;
    expect_bits(gx::evaluate(*program, zero), 0, 0x00000000u);
}

// Composition: the controlled multiply produces its word first, then the
// distinct saturation node clamps it. Authored products and results.
ARMIGO_TEST(graphics_saturate, saturate_composes_after_controlled_multiply) {
    auto spec = fragment_spec(4);
    spec.nodes.push_back({8, gx::Literal{0x40000000u}});   // 2.0f
    spec.nodes.push_back({16, gx::Literal{0x40000000u}});  // 2.0f
    spec.nodes.push_back({24, gx::FloatArithmetic{gx::Arithmetic::MultiplyNearest,
        ScalarId{0}, ScalarId{1}, std::nullopt, false}});  // 4.0f
    spec.nodes.push_back({32, gx::Saturate{ScalarId{2}}});  // -> 1.0f
    spec.nodes.push_back({40, gx::Literal{0x40000000u}});   // 2.0f
    spec.nodes.push_back({48, gx::Literal{0xbf000000u}});   // -0.5f
    spec.nodes.push_back({56, gx::FloatArithmetic{gx::Arithmetic::MultiplyNearest,
        ScalarId{4}, ScalarId{5}, std::nullopt, false}});  // -1.0f
    spec.nodes.push_back({64, gx::Saturate{ScalarId{6}}});  // -> +0.0f
    spec.nodes.push_back({72, gx::Literal{0x3e800000u}});   // 0.25f
    spec.blocks.push_back({0, 9, gx::Return{}});
    spec.exports.push_back({gx::OutputKind::Color, 0, ScalarId{3}});
    spec.exports.push_back({gx::OutputKind::Color, 1, ScalarId{2}});
    spec.exports.push_back({gx::OutputKind::Color, 2, ScalarId{7}});
    spec.exports.push_back({gx::OutputKind::Color, 3, ScalarId{8}});
    const auto program = build(std::move(spec));
    if (!program) return;
    const auto result = gx::evaluate(*program, {});
    expect_bits(result, 0, 0x3f800000u);
    expect_bits(result, 1, 0x40800000u);
    expect_bits(result, 2, 0x00000000u);
    expect_bits(result, 3, 0x3e800000u);
}

// The MSL helper is emitted only for programs that contain a Saturate node;
// programs without one keep their helper-free source bytes, and reflection
// gains no resources from saturation.
ARMIGO_TEST(graphics_saturate, msl_helper_is_conditional_and_reflection_unchanged) {
    const auto plain = gx::make_graph([] {
        gx::GraphSpec spec;
        spec.stage = Stage::Vertex;
        put_sph_word(spec.sph, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
        spec.sph[0x35] = std::byte(0xf0);
        spec.nodes.push_back({8, gx::Literal{0x3f800000u}});
        spec.blocks.push_back({0, 1, gx::Return{}});
        for (unsigned lane = 0; lane < 4; ++lane) {
            spec.exports.push_back({gx::OutputKind::Position,
                static_cast<std::uint16_t>(lane), ScalarId{0}});
        }
        return spec;
    }());
    const auto* plain_program = std::get_if<gx::Program>(&plain);
    ARMIGO_CHECK(plain_program != nullptr);
    if (!plain_program) return;
    gm::EmitOptions vertex_options;
    vertex_options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    // Own the emit result: accepted() returns a pointer into it, so the
    // result must outlive every use of that pointer.
    const auto plain_result = gm::emit(*plain_program, vertex_options);
    const auto* plain_emitted = accepted(plain_result);
    if (!plain_emitted) return;
    ARMIGO_CHECK(!contains(plain_emitted->source, "armigo_saturate"));
    ARMIGO_CHECK(plain_emitted->source ==
                 std::get<gm::EmittedStage>(gm::emit(*plain_program, vertex_options)).source);

    auto spec = fragment_spec(4);
    spec.nodes.push_back({8, gx::Literal{0x40000000u}});
    spec.nodes.push_back({16, gx::Literal{0x40000000u}});
    spec.nodes.push_back({24, gx::FloatArithmetic{gx::Arithmetic::MultiplyNearest,
        ScalarId{0}, ScalarId{1}, std::nullopt, false}});
    spec.nodes.push_back({32, gx::Saturate{ScalarId{2}}});
    spec.blocks.push_back({0, 4, gx::Return{}});
    for (unsigned lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Color,
            static_cast<std::uint16_t>(lane), ScalarId{3}});
    }
    const auto program = build(std::move(spec));
    if (!program) return;
    const auto saturated_result = gm::emit(*program, {});
    const auto* emitted = accepted(saturated_result);
    if (!emitted) return;
    ARMIGO_CHECK(occurrences(emitted->source,
                  "inline uint armigo_saturate_float32(uint bits)") == 1);
    // The graph body assigns under the block activity guard, after the
    // source arithmetic computed the multiplied word in the same block.
    ARMIGO_CHECK(contains(emitted->source,
                  "v3 = armigo_saturate_float32(v2);"));
    ARMIGO_CHECK(ordered_at(emitted->source,
                  "v2 = armigo_mul_rn_ftz(v0, v1);") <
                 ordered_at(emitted->source,
                  "v3 = armigo_saturate_float32(v2);"));
    ARMIGO_CHECK(emitted->constant_buffers.empty());
    ARMIGO_CHECK(emitted->textures.empty());
    ARMIGO_CHECK(emitted->interface.color_output_mask == 0xf);
    ARMIGO_CHECK(emitted->source ==
                 std::get<gm::EmittedStage>(gm::emit(*program, {})).source);
}

// The checked family facade routes through Builder::scalar and appends a
// well-formed Saturate node referencing the supplied operand.
ARMIGO_TEST(graphics_saturate, family_lower_facade_appends_saturate_nodes) {
    gx::Builder builder{gx::Limits{}};
    std::array<std::byte, 80> sph{};
    put_sph_word(sph, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    builder.begin_graph(Stage::Vertex, sph);
    const auto source = builder.scalar(gx::Literal{0x40000000u});
    gx::detail::FamilyLower lower{builder};
    // begin_graph already installed the zero literal as node 0.
    ARMIGO_CHECK_EQ(source.index, 1u);
    const auto saturated = lower.saturate(source);
    ARMIGO_CHECK_EQ(saturated.index, 2u);
    const auto* operation =
        std::get_if<gx::Saturate>(&builder.program.nodes()[2].operation);
    ARMIGO_CHECK(operation != nullptr);
    if (operation != nullptr) {
        ARMIGO_CHECK_EQ(operation->operand.index, source.index);
    }
    const auto again = lower.saturate(saturated);
    ARMIGO_CHECK_EQ(again.index, 3u);
}

}  // namespace armigo::shader::test
