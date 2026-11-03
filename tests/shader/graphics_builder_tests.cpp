// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Contract tests for the private graphics Builder bridge: begin_graph setup,
// structured BuilderFailure reporting, and snapshot-scoped export collection.
// Synthetic shader headers only; no forward lowering or graph execution.

#include "graphics_builder.h"
#include "test_support.h"

#include <array>
#include <cstdint>
#include <utility>
#include <variant>

namespace armigo::shader::test {
namespace {
namespace gx = graphics;
namespace mx = maxwell;
using Stage = shader_input::Stage;
using Family = mx::OpcodeFamily;

void put_word(std::array<std::byte, 80>& data, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        data[offset + i] = std::byte(static_cast<unsigned char>(value >> (i * 8)));
    }
}

// Minimal synthetic SPH: only the stage type field, everything else zero.
std::array<std::byte, 80> sph(Stage stage) {
    std::array<std::byte, 80> data{};
    put_word(data, 0, static_cast<std::uint32_t>(
                          shader_input::sph_stage_type_for(stage)) << 10);
    return data;
}

// Declares vertex position output 28 (component bit in header byte 0x35).
std::array<std::byte, 80> sph_with_position_output() {
    auto data = sph(Stage::Vertex);
    data[0x35] |= std::byte{1u << 4};
    return data;
}

// Declares fragment color target 0 with all four lanes (RGBA).
std::array<std::byte, 80> sph_with_color0_rgba() {
    auto data = sph(Stage::Fragment);
    put_word(data, 0x48, 15);
    return data;
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
}  // namespace

ARMIGO_TEST(graphics_builder, begin_graph_prepares_fresh_program) {
    gx::Builder builder(gx::Limits{});
    const auto header = sph(Stage::Vertex);
    builder.begin_graph(Stage::Vertex, header);
    ARMIGO_CHECK(builder.program.stage() == Stage::Vertex);
    ARMIGO_CHECK(builder.program.sph() == header);
    ARMIGO_CHECK_EQ(builder.program.nodes().size(), 1u);
    ARMIGO_CHECK(builder.program.blocks().empty());
    ARMIGO_CHECK(builder.program.exports().empty());
    const auto* literal =
        std::get_if<gx::Literal>(&builder.program.nodes()[0].operation);
    ARMIGO_CHECK(literal != nullptr);
    if (literal) ARMIGO_CHECK_EQ(literal->bits, 0u);
    ARMIGO_CHECK_EQ(builder.zero.index, 0u);
}

ARMIGO_TEST(graphics_builder, begin_graph_rejects_bad_stage_and_header) {
    const auto header = sph(Stage::Vertex);
    rejects([&] {
        gx::Builder(gx::Limits{}).begin_graph(Stage::Geometry, header);
    }, gx::Error::Code::UnsupportedStage);
    // A fragment header does not match the requested vertex stage.
    rejects([&] {
        gx::Builder(gx::Limits{}).begin_graph(Stage::Vertex, sph(Stage::Fragment));
    }, gx::Error::Code::HeaderMismatch);
}

ARMIGO_TEST(graphics_builder, begin_graph_reports_zero_node_limit) {
    rejects([&] {
        gx::Builder(gx::Limits{0}).begin_graph(Stage::Vertex, sph(Stage::Vertex));
    }, gx::Error::Code::Limit);
}

ARMIGO_TEST(graphics_builder, begin_graph_rejects_reused_builder) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, sph(Stage::Vertex));
    rejects([&] {
        builder.begin_graph(Stage::Vertex, sph(Stage::Vertex));
    }, gx::Error::Code::InvalidInstruction);
}

ARMIGO_TEST(graphics_builder, instruction_failure_arrives_as_builder_failure) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, sph_with_position_output());
    builder.pc = 8;
    const mx::Instruction bad_add{8, 0, Family::FloatAddRegister, {7, false},
        mx::FloatAdd{{2}, {2}, mx::RegisterOperand{255},
                     mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero,
                     true}};
    rejects([&] { builder.instruction(bad_add); },
            gx::Error::Code::UninitializedRegister);
}

ARMIGO_TEST(graphics_builder, run_reports_through_builder_failure) {
    mx::DecodedProgram decoded;
    decoded.stage = Stage::Vertex;
    decoded.sph = sph_with_position_output();
    gx::Builder builder(gx::Limits{});
    rejects([&] { std::ignore = builder.run(decoded); },
            gx::Error::Code::MissingExit);
}

ARMIGO_TEST(graphics_builder, collect_exports_reports_only_current_snapshot) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, sph_with_position_output());
    const auto first_id = builder.scalar(gx::Literal{0x11111111u});
    builder.stored[28] = first_id;
    const auto first = builder.collect_exports();
    ARMIGO_CHECK_EQ(first.size(), 1u);
    ARMIGO_CHECK(first[0].kind == gx::OutputKind::Position);
    ARMIGO_CHECK_EQ(first[0].component, 0u);
    ARMIGO_CHECK_EQ(first[0].value.index, first_id.index);
    const auto second_id = builder.scalar(gx::Literal{0x22222222u});
    builder.stored[28] = second_id;
    const auto second = builder.collect_exports();
    ARMIGO_CHECK_EQ(second.size(), 1u);
    ARMIGO_CHECK_EQ(second[0].value.index, second_id.index);
    ARMIGO_CHECK(builder.program.exports().empty());
    // Nodes and builder state survive collection: the zero literal and both
    // snapshot nodes are retained.
    ARMIGO_CHECK_EQ(builder.program.nodes().size(), 3u);
    const auto* snapshot_literal = std::get_if<gx::Literal>(
        &builder.program.nodes()[second_id.index].operation);
    ARMIGO_CHECK(snapshot_literal != nullptr);
    if (snapshot_literal) ARMIGO_CHECK_EQ(snapshot_literal->bits, 0x22222222u);
}

ARMIGO_TEST(graphics_builder, collect_exports_collects_fragment_color_snapshot) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Fragment, sph_with_color0_rgba());
    std::array<gx::ScalarId, 4> lanes{};
    for (unsigned reg = 0; reg < 4; ++reg) {
        lanes[reg] = builder.scalar(gx::Literal{0x01010101u + reg});
        builder.write(mx::RegisterOperand{static_cast<std::uint8_t>(reg)},
                      lanes[reg]);
    }
    const auto first = builder.collect_exports();
    ARMIGO_CHECK_EQ(first.size(), 4u);
    for (unsigned lane = 0; lane < 4; ++lane) {
        ARMIGO_CHECK(first[lane].kind == gx::OutputKind::Color);
        ARMIGO_CHECK_EQ(first[lane].component, lane);
        ARMIGO_CHECK_EQ(first[lane].value.index, lanes[lane].index);
    }
    ARMIGO_CHECK(builder.program.exports().empty());
    const auto replaced = builder.scalar(gx::Literal{0x55555555u});
    builder.write(mx::RegisterOperand{0}, replaced);
    const auto second = builder.collect_exports();
    ARMIGO_CHECK_EQ(second.size(), 4u);
    for (unsigned lane = 0; lane < 4; ++lane) {
        ARMIGO_CHECK(second[lane].kind == gx::OutputKind::Color);
        ARMIGO_CHECK_EQ(second[lane].component, lane);
        ARMIGO_CHECK_EQ(second[lane].value.index,
                        lane == 0 ? replaced.index : lanes[lane].index);
    }
    ARMIGO_CHECK(builder.program.exports().empty());
    // Nodes survive collection: the zero literal, the four lane nodes, and
    // the replacement R0 node are all retained.
    ARMIGO_CHECK_EQ(builder.program.nodes().size(), 6u);
}

ARMIGO_TEST(graphics_builder, collect_exports_reports_missing_output_precisely) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, sph_with_position_output());
    try {
        std::ignore = builder.collect_exports();
        record_failure(__FILE__, __LINE__, "missing output was not reported");
    } catch (const gx::BuilderFailure& failure) {
        ARMIGO_CHECK(failure.error.code == gx::Error::Code::MissingOutput);
        ARMIGO_CHECK_EQ(failure.error.detail, 28u);
    }
}
}  // namespace armigo::shader::test
