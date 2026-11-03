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
#include <armigo/shader/maxwell.h>
#include "test_support.h"

#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

namespace armigo::shader::test {
namespace {
namespace gx = graphics;
namespace mx = maxwell;
using Stage = shader_input::Stage;
using gx::BlockId;
using gx::PredicateId;
using gx::ScalarId;

std::uint32_t fbits(float value) { return std::bit_cast<std::uint32_t>(value); }

void put_sph_word(std::array<std::byte, 80>& sph, unsigned at,
                  std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) sph[at + i] = std::byte(value >> (i * 8));
}

// Vertex header declaring the four Position outputs, generic input 0, and
// generic output 0 under the graph interface contract.
gx::GraphSpec vertex_spec() {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    put_sph_word(spec.sph, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    spec.sph[0x35] = std::byte(0xf0);
    spec.sph[0x18] = std::byte(0x01);
    spec.sph[0x36] = std::byte(0x01);
    return spec;
}

// Fragment header declaring Color0 lanes 0..3.
gx::GraphSpec fragment_spec() {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    put_sph_word(spec.sph, 0, shader_input::sph_stage_type_for(Stage::Fragment) << 10);
    put_sph_word(spec.sph, 0x48, 0xf);
    return spec;
}

gx::Node node(std::uint32_t pc, gx::Operation operation) {
    return {pc, std::move(operation)};
}

// Diamond: the entry condition reads an attribute word, the chosen arm defines
// one word and one predicate, both joins select by actual predecessor, and the
// inverted joined predicate picks the final word. Node slots follow the block
// layout below.
enum DiamondSlot {
    DiamondInput = 0, DiamondZero = 1, DiamondDiffers = 2,
    LeftWord = 3, LeftFlag = 4, RightWord = 5, RightFlag = 6,
    JoinWord = 7, JoinFlag = 8, Inverted = 9,
    HighWord = 10, LowWord = 11, Tail = 12,
};

gx::GraphSpec diamond() {
    auto spec = vertex_spec();
    spec.nodes.push_back(node(8, gx::AttributeLoad{32}));
    spec.nodes.push_back(node(16, gx::Literal{0}));
    spec.nodes.push_back(node(24, gx::WordNotEqual{{DiamondInput}, {DiamondZero}}));
    spec.nodes.push_back(node(40, gx::Literal{10}));
    spec.nodes.push_back(node(48, gx::PredicateLiteral{true}));
    spec.nodes.push_back(node(56, gx::Literal{20}));
    spec.nodes.push_back(node(64, gx::PredicateLiteral{false}));
    spec.nodes.push_back(node(72, gx::ScalarPhi{{
        {{BlockId{1}}, {LeftWord}}, {{BlockId{2}}, {RightWord}}}}));
    spec.nodes.push_back(node(80, gx::PredicatePhi{{
        {{BlockId{1}}, {LeftFlag}}, {{BlockId{2}}, {RightFlag}}}}));
    spec.nodes.push_back(node(88, gx::PredicateNot{{JoinFlag}}));
    spec.nodes.push_back(node(96, gx::Literal{0x11111111u}));
    spec.nodes.push_back(node(104, gx::Literal{0x22222222u}));
    spec.nodes.push_back(node(112, gx::ScalarPhi{{
        {{BlockId{4}}, {HighWord}}, {{BlockId{5}}, {LowWord}}}}));
    spec.blocks.push_back({0, 3,
        gx::BranchIf{PredicateId{DiamondDiffers}, BlockId{1}, BlockId{2}}});
    spec.blocks.push_back({3, 5, gx::Jump{BlockId{3}}});
    spec.blocks.push_back({5, 7, gx::Jump{BlockId{3}}});
    spec.blocks.push_back({7, 10,
        gx::BranchIf{PredicateId{Inverted}, BlockId{5}, BlockId{4}}});
    spec.blocks.push_back({10, 11, gx::Jump{BlockId{6}}});
    spec.blocks.push_back({11, 12, gx::Jump{BlockId{6}}});
    spec.blocks.push_back({12, 13, gx::Return{}});
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Position, lane, ScalarId{Tail}});
    }
    spec.exports.push_back({gx::OutputKind::Generic, 0, ScalarId{JoinWord}});
    return spec;
}

// Diamond whose true edge passes through an empty block before the join, so
// the phi predecessor is the post-empty-block jump source.
enum BridgeSlot {
    BridgeInput = 0, BridgeZero = 1, BridgeDiffers = 2,
    BridgeLeft = 3, BridgeRight = 4, BridgeJoin = 5,
};

gx::GraphSpec empty_bridge() {
    auto spec = vertex_spec();
    spec.nodes.push_back(node(8, gx::AttributeLoad{32}));
    spec.nodes.push_back(node(16, gx::Literal{0}));
    spec.nodes.push_back(node(24, gx::WordNotEqual{{BridgeInput}, {BridgeZero}}));
    spec.nodes.push_back(node(32, gx::Literal{30}));
    spec.nodes.push_back(node(40, gx::Literal{40}));
    spec.nodes.push_back(node(48, gx::ScalarPhi{{
        {{BlockId{2}}, {BridgeLeft}}, {{BlockId{3}}, {BridgeRight}}}}));
    spec.blocks.push_back({0, 3,
        gx::BranchIf{PredicateId{BridgeDiffers}, BlockId{1}, BlockId{3}}});
    spec.blocks.push_back({3, 3, gx::Jump{BlockId{2}}});
    spec.blocks.push_back({3, 4, gx::Jump{BlockId{4}}});
    spec.blocks.push_back({4, 5, gx::Jump{BlockId{4}}});
    spec.blocks.push_back({5, 6, gx::Return{}});
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Position, lane, ScalarId{BridgeJoin}});
    }
    spec.exports.push_back({gx::OutputKind::Generic, 0, ScalarId{BridgeJoin}});
    return spec;
}

// The guarded arm reads past the end of a one-word constant buffer, then
// samples through the callback; the skipped arm exports a literal. Which
// resources run depends only on the branch chosen at run time.
enum GuardedSlot {
    GuardedControl = 0, GuardedZero = 1, GuardedDiffers = 2,
    GuardedX = 3, GuardedY = 4, GuardedLoad = 5, GuardedSample = 6,
    GuardedRed = 7, GuardedSafe = 8, GuardedJoin = 9,
};

gx::GraphSpec guarded_resources() {
    auto spec = fragment_spec();
    spec.nodes.push_back(node(8, gx::ConstantLoad{1, 0}));
    spec.nodes.push_back(node(16, gx::Literal{0}));
    spec.nodes.push_back(node(24, gx::WordNotEqual{{GuardedControl}, {GuardedZero}}));
    spec.nodes.push_back(node(32, gx::Literal{fbits(0.25f)}));
    spec.nodes.push_back(node(40, gx::Literal{fbits(0.5f)}));
    spec.nodes.push_back(node(48, gx::ConstantLoad{0, 4}));
    spec.nodes.push_back(node(56,
        gx::TextureSample{{GuardedX}, {GuardedY}, 0x34, true}));
    spec.nodes.push_back(node(64, gx::SampleComponent{{GuardedSample}, 0}));
    spec.nodes.push_back(node(72, gx::Literal{0x40490fdbu}));
    spec.nodes.push_back(node(80, gx::ScalarPhi{{
        {{BlockId{1}}, {GuardedRed}}, {{BlockId{2}}, {GuardedSafe}}}}));
    spec.blocks.push_back({0, 3,
        gx::BranchIf{PredicateId{GuardedDiffers}, BlockId{1}, BlockId{2}}});
    spec.blocks.push_back({3, 8, gx::Jump{BlockId{3}}});
    spec.blocks.push_back({8, 9, gx::Jump{BlockId{3}}});
    spec.blocks.push_back({9, 10, gx::Return{}});
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Color, lane, ScalarId{GuardedJoin}});
    }
    return spec;
}

// Two samples execute back to back in one block; each pair of exports reads
// different lanes of different samples, so both callback results must be
// distinct, complete, and ordered.
enum SequentialSlot {
    SequentialX = 0, SequentialY = 1, FirstSample = 2, FirstRed = 3,
    FirstGreen = 4, SecondSample = 5, SecondRed = 6, SecondGreen = 7,
};

gx::GraphSpec sequential_samples() {
    auto spec = fragment_spec();
    spec.nodes.push_back(node(8, gx::Literal{fbits(0.25f)}));
    spec.nodes.push_back(node(16, gx::Literal{fbits(0.75f)}));
    spec.nodes.push_back(node(24,
        gx::TextureSample{{SequentialX}, {SequentialY}, 0, true}));
    spec.nodes.push_back(node(32, gx::SampleComponent{{FirstSample}, 0}));
    spec.nodes.push_back(node(40, gx::SampleComponent{{FirstSample}, 1}));
    spec.nodes.push_back(node(48,
        gx::TextureSample{{SequentialX}, {SequentialY}, 8, false}));
    spec.nodes.push_back(node(56, gx::SampleComponent{{SecondSample}, 0}));
    spec.nodes.push_back(node(64, gx::SampleComponent{{SecondSample}, 1}));
    spec.blocks.push_back({0, 8, gx::Return{}});
    spec.exports.push_back({gx::OutputKind::Color, 0, ScalarId{FirstRed}});
    spec.exports.push_back({gx::OutputKind::Color, 1, ScalarId{FirstGreen}});
    spec.exports.push_back({gx::OutputKind::Color, 2, ScalarId{SecondRed}});
    spec.exports.push_back({gx::OutputKind::Color, 3, ScalarId{SecondGreen}});
    return spec;
}

// Sampler that counts calls and fails in the configured way.
struct Sequence {
    unsigned calls = 0;
    enum class When { Never, First, Second } fail = When::Never;
    bool fail_by_throw = true;
    bool poison_rounding = false;
    std::array<std::uint32_t, 4> first{};
    std::array<std::uint32_t, 4> second{};

    bool operator()(const gx::TextureRequest&,
                    std::array<std::uint32_t, 4>& out) {
        ++calls;
        if (poison_rounding) std::fesetround(FE_UPWARD);
        const bool failing = fail == When::First ||
            (fail == When::Second && calls >= 2);
        if (failing && fail_by_throw) {
            throw std::runtime_error("Synthetic sampler failure");
        }
        if (failing) return false;
        out = calls == 1 ? first : second;
        return true;
    }
};

const gx::Program* as_program(const gx::GraphResult& result) {
    return std::get_if<gx::Program>(&result);
}

gx::EvaluateResult run(const gx::GraphSpec& spec, const gx::Inputs& inputs = {}) {
    const auto made = gx::make_graph(spec);
    const auto* program = as_program(made);
    if (program == nullptr) {
        const auto& error = std::get<gx::Error>(made);
        record_failure(__FILE__, __LINE__,
                       std::string("graph fixture was rejected: ") + error.message());
        return error;
    }
    return gx::evaluate(*program, inputs);
}

void expect_output(const gx::EvaluateResult& result, unsigned index,
                   gx::OutputKind kind, unsigned component, std::uint32_t bits) {
    const auto* evaluation = std::get_if<gx::Evaluation>(&result);
    if (evaluation == nullptr) {
        record_failure(__FILE__, __LINE__, std::get<gx::Error>(result).message());
        return;
    }
    if (index >= evaluation->outputs.size()) {
        record_failure(__FILE__, __LINE__, "missing graphics output slot");
        return;
    }
    const auto& output = evaluation->outputs[index];
    ARMIGO_CHECK(output.kind == kind);
    ARMIGO_CHECK_EQ(output.component, static_cast<std::uint16_t>(component));
    ARMIGO_CHECK_EQ(output.bits, bits);
}

void expect_position(const gx::EvaluateResult& result, unsigned first,
                     std::uint32_t bits) {
    for (unsigned lane = 0; lane < 4; ++lane) {
        expect_output(result, first + lane, gx::OutputKind::Position, lane, bits);
    }
}

void expect_color(const gx::EvaluateResult& result, unsigned first,
                  std::uint32_t bits) {
    for (unsigned lane = 0; lane < 4; ++lane) {
        expect_output(result, first + lane, gx::OutputKind::Color, lane, bits);
    }
}

void expect_error(const gx::EvaluateResult& result, gx::Error::Code code) {
    const auto* error = std::get_if<gx::Error>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) ARMIGO_CHECK(error->code == code);
}

}  // namespace

ARMIGO_TEST(graph_cpu_execution, diamond_selects_actual_predecessor_values) {
    auto spec = diamond();
    gx::Inputs inputs;
    const std::array left = {gx::AttributeInput{32, 5u}};
    inputs.attributes = left;
    const auto taken = run(spec, inputs);
    expect_position(taken, 0, 0x11111111u);
    expect_output(taken, 4, gx::OutputKind::Generic, 0, 10u);
    const std::array right = {gx::AttributeInput{32, 0u}};
    inputs.attributes = right;
    const auto skipped = run(spec, inputs);
    expect_position(skipped, 0, 0x22222222u);
    expect_output(skipped, 4, gx::OutputKind::Generic, 0, 20u);
}

ARMIGO_TEST(graph_cpu_execution, raw_high_bit_words_compare_unequal) {
    auto spec = diamond();
    gx::Inputs inputs;
    // Negative zero equals zero as float, yet the raw words differ, so the
    // word-not-equal condition still takes the true arm.
    const std::array negative_zero = {gx::AttributeInput{32, 0x80000000u}};
    inputs.attributes = negative_zero;
    const auto result = run(spec, inputs);
    expect_position(result, 0, 0x11111111u);
    expect_output(result, 4, gx::OutputKind::Generic, 0, 10u);
}

ARMIGO_TEST(graph_cpu_execution, empty_block_forwards_to_join) {
    auto spec = empty_bridge();
    gx::Inputs inputs;
    const std::array through_empty = {gx::AttributeInput{32, 7u}};
    inputs.attributes = through_empty;
    const auto taken = run(spec, inputs);
    expect_position(taken, 0, 30u);
    const std::array around = {gx::AttributeInput{32, 0u}};
    inputs.attributes = around;
    const auto skipped = run(spec, inputs);
    expect_position(skipped, 0, 40u);
}

ARMIGO_TEST(graph_cpu_execution, skipped_arm_avoids_resources_and_callbacks) {
    auto spec = guarded_resources();
    Sequence sampler;
    gx::Inputs inputs;
    const std::array control = {std::uint32_t{0}};
    inputs.constant_buffers[1] = control;
    inputs.sample_2d_implicit_lod =
        [&sampler](const gx::TextureRequest& request,
                   std::array<std::uint32_t, 4>& out) { return sampler(request, out); };
    // The false arm runs without touching the missing guarded buffer, the
    // absent sampler input, or the callback.
    const auto result = run(spec, inputs);
    expect_color(result, 0, 0x40490fdbu);
    ARMIGO_CHECK_EQ(sampler.calls, 0u);
}

ARMIGO_TEST(graph_cpu_execution, executed_arm_reports_resource_failures) {
    auto spec = guarded_resources();
    Sequence sampler;
    gx::Inputs inputs;
    const std::array control = {std::uint32_t{1}};
    inputs.constant_buffers[1] = control;
    inputs.sample_2d_implicit_lod =
        [&sampler](const gx::TextureRequest& request,
                   std::array<std::uint32_t, 4>& out) { return sampler(request, out); };
    // The guarded load runs first and its bounds failure is reported.
    expect_error(run(spec, inputs), gx::Error::Code::ConstantBounds);
    ARMIGO_CHECK_EQ(sampler.calls, 0u);
    // With the guarded buffer present, the callback runs exactly once; both
    // failure forms of the callback become SamplingFailed.
    const std::array guarded = {std::uint32_t{7}, std::uint32_t{8}};
    inputs.constant_buffers[0] = guarded;
    sampler.fail = Sequence::When::First;
    expect_error(run(spec, inputs), gx::Error::Code::SamplingFailed);
    ARMIGO_CHECK_EQ(sampler.calls, 1u);
    sampler.fail_by_throw = false;
    expect_error(run(spec, inputs), gx::Error::Code::SamplingFailed);
    ARMIGO_CHECK_EQ(sampler.calls, 2u);
    // A successful callback feeds the join through the sampled lane.
    sampler.calls = 0;
    sampler.fail = Sequence::When::Never;
    sampler.first = {fbits(2.0f), fbits(3.0f), fbits(4.0f), fbits(5.0f)};
    const auto sampled = run(spec, inputs);
    expect_color(sampled, 0, fbits(2.0f));
    ARMIGO_CHECK_EQ(sampler.calls, 1u);
}

ARMIGO_TEST(graph_cpu_execution, sequential_samples_preserve_order_and_environment) {
    auto spec = sequential_samples();
    Sequence sampler;
    gx::Inputs inputs;
    inputs.sample_2d_implicit_lod =
        [&sampler](const gx::TextureRequest& request,
                   std::array<std::uint32_t, 4>& out) { return sampler(request, out); };
    std::fenv_t original;
    ARMIGO_CHECK_EQ(std::fegetenv(&original), 0);
    ARMIGO_CHECK_EQ(std::fesetround(FE_DOWNWARD), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID);
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    sampler.first = {fbits(10.0f), fbits(11.0f), fbits(12.0f), fbits(13.0f)};
    sampler.second = {fbits(20.0f), fbits(21.0f), fbits(22.0f), fbits(23.0f)};
    const auto sampled = run(spec, inputs);
    expect_output(sampled, 0, gx::OutputKind::Color, 0, fbits(10.0f));
    expect_output(sampled, 1, gx::OutputKind::Color, 1, fbits(11.0f));
    expect_output(sampled, 2, gx::OutputKind::Color, 2, fbits(20.0f));
    expect_output(sampled, 3, gx::OutputKind::Color, 3, fbits(21.0f));
    ARMIGO_CHECK_EQ(sampler.calls, 2u);
    ARMIGO_CHECK_EQ(std::fegetround(), FE_DOWNWARD);
    ARMIGO_CHECK_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    // A failure on the second callback still proves the first one executed.
    sampler.calls = 0;
    sampler.fail = Sequence::When::Second;
    expect_error(run(spec, inputs), gx::Error::Code::SamplingFailed);
    ARMIGO_CHECK_EQ(sampler.calls, 2u);
    ARMIGO_CHECK_EQ(std::fegetround(), FE_DOWNWARD);
    ARMIGO_CHECK_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    // A callback that changes the rounding mode fails the postcheck.
    sampler.calls = 0;
    sampler.fail = Sequence::When::Never;
    sampler.poison_rounding = true;
    expect_error(run(spec, inputs), gx::Error::Code::FloatingEnvironment);
    ARMIGO_CHECK_EQ(sampler.calls, 1u);
    ARMIGO_CHECK_EQ(std::fegetround(), FE_DOWNWARD);
    ARMIGO_CHECK_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    ARMIGO_CHECK_EQ(std::fesetenv(&original), 0);
}

ARMIGO_TEST(graph_cpu_execution, legacy_straight_line_arithmetic_unchanged) {
    mx::DecodedProgram decoded;
    decoded.stage = Stage::Vertex;
    put_sph_word(decoded.sph, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    decoded.sph[0x35] |= std::byte(1 << 4);
    const auto append = [&decoded](mx::OpcodeFamily family,
                                   mx::InstructionPayload payload) {
        const auto i = decoded.instructions.size();
        decoded.instructions.push_back(
            {static_cast<std::uint32_t>((i / 3) * 32 + (i % 3 + 1) * 8),
             0, family, {7, false}, std::move(payload)});
    };
    append(mx::OpcodeFamily::Mov32Immediate,
           mx::Mov32Immediate{{1}, 15, fbits(3.0f)});
    append(mx::OpcodeFamily::FloatMultiplyRegister,
           mx::FloatMultiply{{2}, {1}, mx::RegisterOperand{1},
                              mx::RoundingMode::NearestEven,
                              mx::FlushMode::FlushToZero});
    append(mx::OpcodeFamily::FloatAddRegister,
           mx::FloatAdd{{2}, {2}, mx::RegisterOperand{1},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero, false});
    append(mx::OpcodeFamily::AttributeStore,
           mx::AttributeAccess{{2}, {255}, {255}, 28 * 4, 0, false});
    append(mx::OpcodeFamily::Exit, mx::Exit{15});
    const auto lowered = gx::lower(decoded);
    const auto* program = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    ARMIGO_CHECK(program->blocks().empty());
    ARMIGO_CHECK(!program->entry().has_value());
    const auto result = gx::evaluate(*program, {});
    expect_output(result, 0, gx::OutputKind::Position, 0, fbits(12.0f));
}

}  // namespace armigo::shader::test
