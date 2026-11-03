// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Contract tests for the registered pure-operation table: registry shape,
// CPU reference results, graph validation, MSL text in both emitters, and
// the FamilyLower facade (operations and the predicate register file).
// Synthetic graphs and headers only.

#include "graphics_builder.h"
#include "graphics_families.h"
#include "graphics_operations.h"
#include "test_support.h"

#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace gx = graphics;
namespace gm = graphics_msl;
using Stage = shader_input::Stage;
using gx::PureKind;

void put_word(std::array<std::byte, 80>& data, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        data[offset + i] = std::byte(static_cast<unsigned char>(value >> (i * 8)));
    }
}

std::array<std::byte, 80> vertex_header() {
    std::array<std::byte, 80> data{};
    put_word(data, 0, shader_input::sph_stage_type_for(Stage::Vertex) << 10);
    data[0x35] = std::byte{0xf0};  // position x, y, z, w outputs
    return data;
}

gx::Node pure(std::uint32_t pc, PureKind kind, std::array<std::uint32_t, 3> operands,
              std::uint32_t immediate = 0) {
    return {pc, gx::Pure{kind, operands, immediate}};
}

// Vertex graph exporting four Raw32 nodes as position x/y/z/w.
gx::GraphSpec vertex_spec(std::vector<gx::Node> nodes, std::array<std::uint32_t, 4> outputs) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    spec.sph = vertex_header();
    spec.nodes = std::move(nodes);
    for (std::uint16_t lane = 0; lane < 4; ++lane) {
        spec.exports.push_back({gx::OutputKind::Position, lane, gx::ScalarId{outputs[lane]}});
    }
    spec.blocks = {{0, static_cast<std::uint32_t>(spec.nodes.size()), gx::Return{}}};
    return spec;
}

std::array<std::uint32_t, 4> evaluate(const gx::GraphSpec& spec) {
    std::array<std::uint32_t, 4> result{};
    const auto graph = gx::make_graph(spec);
    const auto* program = std::get_if<gx::Program>(&graph);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return result;
    const auto evaluated = gx::evaluate(*program, {});
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr);
    if (outputs == nullptr) return result;
    for (const auto& output : outputs->outputs) result[output.component] = output.bits;
    return result;
}

gx::Error::Code rejection(const gx::GraphSpec& spec) {
    const auto graph = gx::make_graph(spec);
    const auto* error = std::get_if<gx::Error>(&graph);
    ARMIGO_CHECK(error != nullptr);
    return error == nullptr ? gx::Error::Code::InvalidInstruction : error->code;
}

std::string emit(const gx::Program& program) {
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const auto emitted = gm::emit(program, options);
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    return stage == nullptr ? std::string{} : stage->source;
}

bool contains(const std::string& text, const std::string& value) {
    return text.find(value) != std::string::npos;
}

template<class Action>
void rejects(const Action& action, gx::Error::Code code) {
    try {
        action();
    } catch (const gx::BuilderFailure& failure) {
        ARMIGO_CHECK(failure.error.code == code);
        return;
    }
    record_failure(__FILE__, __LINE__, "expected BuilderFailure was not thrown");
}

// Two runtime-free predicates from literal comparisons, then every
// combinator observed through word selection between 1.0 and 2.0.
gx::GraphSpec predicate_spec(bool first, bool second) {
    return vertex_spec({
        {8, gx::Literal{0x3f800000u}},                      // 0: 1.0
        {16, gx::Literal{0x40000000u}},                     // 1: 2.0
        {24, gx::PredicateLiteral{first}},                  // 2
        {32, gx::PredicateLiteral{second}},                 // 3
        pure(40, PureKind::PredicateAnd, {2, 3, 0}),        // 4
        pure(48, PureKind::PredicateOr, {2, 3, 0}),         // 5
        pure(56, PureKind::PredicateXor, {2, 3, 0}),        // 6
        pure(64, PureKind::PredicateSelect, {2, 3, 6}),     // 7
        pure(72, PureKind::Select, {4, 0, 1}),              // 8
        pure(80, PureKind::Select, {5, 0, 1}),              // 9
        pure(88, PureKind::Select, {6, 0, 1}),              // 10
        pure(96, PureKind::Select, {7, 0, 1}),              // 11
    }, {8, 9, 10, 11});
}

}  // namespace

ARMIGO_TEST(operation_registry, every_kind_is_registered_once) {
    const std::array kinds{PureKind::FloatNegate, PureKind::FloatAbsolute, PureKind::Select,
                           PureKind::PredicateAnd, PureKind::PredicateOr,
                           PureKind::PredicateXor, PureKind::PredicateSelect};
    // Later families append kinds; these seven must stay registered as defined.
    ARMIGO_CHECK(gx::detail::registered_operations().size() >= kinds.size());
    for (const auto kind : kinds) {
        const auto* entry = gx::detail::find_operation(kind);
        ARMIGO_CHECK(entry != nullptr);
        if (entry == nullptr) continue;
        ARMIGO_CHECK(entry->kind == kind);
        ARMIGO_CHECK(gx::pure_result_type(kind) == entry->result);
        ARMIGO_CHECK(entry->admits(0));
        ARMIGO_CHECK(!entry->admits(1));
    }
    ARMIGO_CHECK(gx::detail::find_operation(static_cast<PureKind>(0x7fff)) == nullptr);
    ARMIGO_CHECK(gx::pure_result_type(PureKind::PredicateAnd) == gx::ValueType::Predicate);
    ARMIGO_CHECK(gx::pure_result_type(PureKind::Select) == gx::ValueType::Raw32);
}

ARMIGO_TEST(operation_registry, sign_operations_are_exact_bit_operations) {
    // NaN payloads, signed zeros and subnormals pass through untouched apart
    // from the sign bit: no flush and no canonicalization.
    const auto result = evaluate(vertex_spec({
        {8, gx::Literal{0x7fc12345u}},
        {16, gx::Literal{0x80000001u}},
        {24, gx::Literal{0x00000000u}},
        {32, gx::Literal{0xff812345u}},
        pure(40, PureKind::FloatNegate, {0, 0, 0}),
        pure(48, PureKind::FloatAbsolute, {1, 0, 0}),
        pure(56, PureKind::FloatNegate, {2, 0, 0}),
        pure(64, PureKind::FloatAbsolute, {3, 0, 0}),
    }, {4, 5, 6, 7}));
    ARMIGO_CHECK_EQ(result[0], 0xffc12345u);
    ARMIGO_CHECK_EQ(result[1], 0x00000001u);
    ARMIGO_CHECK_EQ(result[2], 0x80000000u);
    ARMIGO_CHECK_EQ(result[3], 0x7f812345u);
}

ARMIGO_TEST(operation_registry, predicate_combinators_follow_truth_tables) {
    for (const bool first : {false, true}) {
        for (const bool second : {false, true}) {
            const auto result = evaluate(predicate_spec(first, second));
            const auto pick = [](bool value) { return value ? 0x3f800000u : 0x40000000u; };
            ARMIGO_CHECK_EQ(result[0], pick(first && second));
            ARMIGO_CHECK_EQ(result[1], pick(first || second));
            ARMIGO_CHECK_EQ(result[2], pick(first != second));
            ARMIGO_CHECK_EQ(result[3], pick(first ? second : (first != second)));
        }
    }
}

ARMIGO_TEST(operation_registry, graph_validator_rejects_malformed_operations) {
    const auto base = [] {
        return std::vector<gx::Node>{{8, gx::Literal{1}}, {16, gx::PredicateLiteral{true}}};
    };
    auto unknown = base();
    unknown.push_back(pure(24, static_cast<PureKind>(0x7fff), {0, 0, 0}));
    ARMIGO_CHECK(rejection(vertex_spec(unknown, {0, 0, 0, 0})) == gx::Error::Code::GraphFormat);

    auto immediate = base();
    immediate.push_back(pure(24, PureKind::FloatNegate, {0, 0, 0}, 1));
    ARMIGO_CHECK(rejection(vertex_spec(immediate, {2, 2, 2, 2})) ==
                 gx::Error::Code::GraphFormat);

    auto unused = base();
    unused.push_back(pure(24, PureKind::FloatNegate, {0, 1, 0}));
    ARMIGO_CHECK(rejection(vertex_spec(unused, {2, 2, 2, 2})) == gx::Error::Code::GraphFormat);

    auto word_for_predicate = base();
    word_for_predicate.push_back(pure(24, PureKind::Select, {0, 0, 0}));
    ARMIGO_CHECK(rejection(vertex_spec(word_for_predicate, {2, 2, 2, 2})) ==
                 gx::Error::Code::GraphTypes);

    auto predicate_for_word = base();
    predicate_for_word.push_back(pure(24, PureKind::FloatAbsolute, {1, 0, 0}));
    ARMIGO_CHECK(rejection(vertex_spec(predicate_for_word, {2, 2, 2, 2})) ==
                 gx::Error::Code::GraphTypes);

    auto forward = base();
    forward.push_back(pure(24, PureKind::FloatNegate, {3, 0, 0}));
    forward.push_back({32, gx::Literal{2}});
    ARMIGO_CHECK(rejection(vertex_spec(forward, {2, 2, 2, 2})) ==
                 gx::Error::Code::GraphDataFlow);

    auto predicate_export = base();
    predicate_export.push_back(pure(24, PureKind::PredicateAnd, {1, 1, 0}));
    ARMIGO_CHECK(rejection(vertex_spec(predicate_export, {2, 0, 0, 0})) ==
                 gx::Error::Code::GraphTypes);
}

ARMIGO_TEST(operation_registry, graph_msl_renders_registered_expressions) {
    const auto graph = gx::make_graph(predicate_spec(true, false));
    const auto* program = std::get_if<gx::Program>(&graph);
    ARMIGO_CHECK(program != nullptr);
    if (program == nullptr) return;
    const auto source = emit(*program);
    ARMIGO_CHECK(contains(source, "p4 = (p2 && p3);"));
    ARMIGO_CHECK(contains(source, "p5 = (p2 || p3);"));
    ARMIGO_CHECK(contains(source, "p6 = (p2 != p3);"));
    ARMIGO_CHECK(contains(source, "p7 = (p2 ? p3 : p6);"));
    ARMIGO_CHECK(contains(source, "v8 = (p4 ? v0 : v1);"));
    ARMIGO_CHECK(contains(source, "p2 = true;"));

    const auto signs = gx::make_graph(vertex_spec({
        {8, gx::Literal{0x3f800000u}},
        pure(16, PureKind::FloatNegate, {0, 0, 0}),
        pure(24, PureKind::FloatAbsolute, {1, 0, 0}),
    }, {1, 2, 0, 0}));
    const auto* sign_program = std::get_if<gx::Program>(&signs);
    ARMIGO_CHECK(sign_program != nullptr);
    if (sign_program == nullptr) return;
    const auto sign_source = emit(*sign_program);
    ARMIGO_CHECK(contains(sign_source, "v1 = (v0 ^ 0x80000000u);"));
    ARMIGO_CHECK(contains(sign_source, "v2 = (v1 & 0x7fffffffu);"));
}

// Writes both fixtures for shader.forward_lower_msl_compile, which compiles
// every emitted forward fixture with the Metal toolchain.
ARMIGO_TEST(forward_lower_msl, registered_operation_fixtures) {
    const char* directory = std::getenv("ARMIGO_FORWARD_MSL_DIR");
    const auto predicates = gx::make_graph(predicate_spec(true, false));
    const auto signs = gx::make_graph(vertex_spec({
        {8, gx::Literal{0x3f800000u}},
        pure(16, PureKind::FloatNegate, {0, 0, 0}),
        pure(24, PureKind::FloatAbsolute, {1, 0, 0}),
    }, {1, 2, 0, 0}));
    for (const auto& [name, graph] : {std::pair{"operation_predicates", &predicates},
                                      std::pair{"operation_signs", &signs}}) {
        const auto* program = std::get_if<gx::Program>(graph);
        ARMIGO_CHECK(program != nullptr);
        if (program == nullptr) continue;
        const auto source = emit(*program);
        if (directory == nullptr || *directory == '\0') continue;
        std::error_code filesystem_error;
        std::filesystem::create_directories(directory, filesystem_error);
        std::ofstream output{std::string{directory} + "/" + name + ".metal",
                             std::ios::binary | std::ios::trunc};
        ARMIGO_CHECK(static_cast<bool>(output));
        output << source;
    }
}

ARMIGO_TEST(operation_registry, facade_builds_checked_operations) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, vertex_header());
    gx::detail::FamilyLower lower(builder);
    const auto one = lower.literal(0x3f800000u);
    const auto negative = lower.operation(PureKind::FloatNegate, {one});
    const auto* node = std::get_if<gx::Pure>(&builder.program.nodes()[negative.index].operation);
    ARMIGO_CHECK(node != nullptr && node->kind == PureKind::FloatNegate &&
                 node->operands[0] == one.index && node->operands[1] == 0);

    const auto truth = lower.predicate_literal(true);
    const auto both = lower.predicate_operation(PureKind::PredicateAnd, {truth, truth});
    const auto chosen = lower.operation(PureKind::Select, {both, one, negative});
    ARMIGO_CHECK(builder.program.nodes()[chosen.index].type() == gx::ValueType::Raw32);

    // Wrong result kind, arity, operand type, and immediate are invalid.
    rejects([&] { (void)lower.operation(PureKind::PredicateAnd, {truth, truth}); },
            gx::Error::Code::InvalidInstruction);
    rejects([&] { (void)lower.predicate_operation(PureKind::Select, {truth, one, one}); },
            gx::Error::Code::InvalidInstruction);
    rejects([&] { (void)lower.operation(PureKind::Select, {truth, one}); },
            gx::Error::Code::InvalidInstruction);
    rejects([&] { (void)lower.operation(PureKind::Select, {one, one, one}); },
            gx::Error::Code::InvalidInstruction);
    rejects([&] { (void)lower.operation(PureKind::FloatNegate, {one}, 1); },
            gx::Error::Code::InvalidInstruction);
    rejects([&] { (void)lower.operation(PureKind::FloatNegate, {gx::ScalarId{9999}}); },
            gx::Error::Code::InvalidInstruction);
}

ARMIGO_TEST(operation_registry, evaluator_rejects_forged_unused_slots) {
    // A node appended without Builder::pure or graph validation must still be
    // rejected by the CPU reference, matching the validator and the emitter.
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, vertex_header());
    const auto one = builder.append(gx::Literal{0x3f800000u});
    builder.append(gx::Pure{PureKind::FloatNegate, {one, one, 0}, 0});
    const auto evaluated = gx::evaluate(builder.program, {});
    const auto* error = std::get_if<gx::Error>(&evaluated);
    ARMIGO_CHECK(error != nullptr && error->code == gx::Error::Code::InvalidInstruction);
}

ARMIGO_TEST(operation_registry, facade_owns_the_predicate_file) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, vertex_header());
    gx::detail::FamilyLower lower(builder);
    rejects([&] { (void)lower.read_predicate(0); }, gx::Error::Code::UninitializedRegister);
    rejects([&] { (void)lower.read_predicate(8); }, gx::Error::Code::InvalidInstruction);

    const auto pt = lower.read_predicate(7);
    const auto* literal =
        std::get_if<gx::PredicateLiteral>(&builder.program.nodes()[pt.index].operation);
    ARMIGO_CHECK(literal != nullptr && literal->value);

    const auto falsity = lower.predicate_literal(false);
    lower.write_predicate(3, falsity);
    ARMIGO_CHECK(lower.read_predicate(3).index == falsity.index);
    ARMIGO_CHECK(builder.predicates[3].has_value());
    lower.write_predicate(7, falsity);  // PT discards
    ARMIGO_CHECK(!builder.predicates[0].has_value());

    const auto inverted = lower.negate_predicate(falsity);
    const auto* negation =
        std::get_if<gx::PredicateNot>(&builder.program.nodes()[inverted.index].operation);
    ARMIGO_CHECK(negation != nullptr && negation->operand.index == falsity.index);

    const auto word = lower.literal(0);
    rejects([&] { lower.write_predicate(1, gx::PredicateId{word.index}); },
            gx::Error::Code::InvalidInstruction);
    rejects([&] { (void)lower.negate_predicate(gx::PredicateId{word.index}); },
            gx::Error::Code::InvalidInstruction);
}

}  // namespace armigo::shader::test
