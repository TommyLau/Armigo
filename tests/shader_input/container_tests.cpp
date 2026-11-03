// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_input/container.h>

#include "synthetic.h"
#include "test_support.h"

#include <algorithm>
#include <array>
#include <limits>

namespace {

using namespace armigo::shader_input;
using Bytes = std::vector<std::uint8_t>;

void put64(Bytes& bytes, std::uint32_t at, std::uint64_t value) {
    test::put_le32(bytes, at, static_cast<std::uint32_t>(value));
    test::put_le32(bytes, at + 4, static_cast<std::uint32_t>(value >> 32));
}

void place(Bytes& bytes, std::size_t at, const Bytes& payload) {
    std::copy(payload.begin(), payload.end(), bytes.begin() + at);
}

void shader(Bytes& bytes, std::uint32_t control_at, std::uint32_t code_at,
            Stage stage) {
    test::CodeSpec spec;
    spec.stage = static_cast<std::uint32_t>(stage);
    place(bytes, control_at, test::build_control(test::control_for(spec)));
    place(bytes, code_at, test::build_code(spec));
}

Bytes bnsh() {
    Bytes bytes(0x7000);
    std::memcpy(bytes.data(), "BNSH\0\0\0\0", 8);
    std::memcpy(bytes.data() + 0x60, "grsc", 4);
    test::put_le32(bytes, 0x7c, 3);
    put64(bytes, 0x80, 0x200);
    put64(bytes, 0x210, 0x300);
    put64(bytes, 0x250, 0x340);
    const std::array stages = {Stage::Vertex, Stage::TessellationControl,
        Stage::TessellationEvaluation, Stage::Geometry, Stage::Fragment,
        Stage::Compute};
    for (std::uint32_t i = 0; i < stages.size(); ++i) {
        const auto descriptor = 0x400u + i * 0x20u;
        const auto control = 0x1000u + i * 0x1000u;
        const auto code = control + 0x800u;
        put64(bytes, 0x308 + i * 8, descriptor);
        put64(bytes, descriptor + 8, control);
        put64(bytes, descriptor + 16, code);
        shader(bytes, control, code, stages[i]);
        test::put_le32(bytes, descriptor + 24,
                      stages[i] == Stage::Compute ? 48 : 176);
        test::put_le32(bytes, descriptor + 28, format::control_min_size);
    }
    // A second variation shares only vertex/fragment stages with the first.
    put64(bytes, 0x348, 0x400);
    put64(bytes, 0x368, 0x480);
    return bytes;
}

void entry(Bytes& bytes, std::uint32_t at, std::uint32_t control,
           std::uint32_t code_offset) {
    test::put_le32(bytes, at, code_offset);
    test::put_le32(bytes, at + 4, 0);
    test::put_le32(bytes, at + 8, format::control_min_size);
    test::put_le32(bytes, at + 12, control - at);
}

Bytes bahs() {
    Bytes bytes(0x2600);
    std::memcpy(bytes.data(), "BAHS", 4);
    test::put_le32(bytes, 0x18, 0x400);
    test::put_le32(bytes, 0x1c, 0x2200);
    shader(bytes, 0x200, 0x2200, Stage::Vertex);
    shader(bytes, 0xb00, 0x2300, Stage::Fragment);
    entry(bytes, 0x40, 0x200, 0);
    entry(bytes, 0x1c00, 0xb00, 0x100);
    entry(bytes, 0x1c10, 0x200, 0);
    return bytes;
}

template <typename Result>
void rejected(const Result& result, ContainerError::Code expected) {
    const auto* error = std::get_if<ContainerError>(&result);
    ARMIGO_SI_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_SI_CHECK(error->code == expected);
        ARMIGO_SI_CHECK(error->message()[0] != '\0');
    }
}

template <typename Container>
const Container* accepted(const std::variant<Container, ContainerError>& result) {
    const auto* value = std::get_if<Container>(&result);
    ARMIGO_SI_CHECK(value != nullptr);
    return value;
}

}  // namespace

ARMIGO_SI_TEST(bnsh, all_slots_and_variation_pairing) {
    const auto bytes = bnsh();
    const auto result = parse_bnsh(test::span_of(bytes));
    const auto* value = accepted(result);
    if (value == nullptr) return;
    ARMIGO_SI_CHECK_EQ(value->variation_count, 3u);
    ARMIGO_SI_CHECK_EQ(value->stages.size(), 8u);
    const std::array expected = {Stage::Vertex, Stage::TessellationControl,
        Stage::TessellationEvaluation, Stage::Geometry, Stage::Fragment,
        Stage::Compute};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& stage = value->stages[i];
        ARMIGO_SI_CHECK(stage.shader.stage == expected[i]);
        ARMIGO_SI_CHECK(stage.variation == 0u);
        ARMIGO_SI_CHECK(stage.slot == static_cast<StageSlot>(i));
        ARMIGO_SI_CHECK(stage.shader.control.data == bytes.data() + stage.control_offset);
        ARMIGO_SI_CHECK(stage.shader.code.data == bytes.data() + stage.code_offset);
    }
    ARMIGO_SI_CHECK(value->stages[6].variation == 1u);
    ARMIGO_SI_CHECK(value->stages[7].variation == 1u);
    ARMIGO_SI_CHECK(value->stages[6].slot == StageSlot::Vertex);
    ARMIGO_SI_CHECK(value->stages[7].slot == StageSlot::Fragment);
}

ARMIGO_SI_TEST(bnsh, offsets_are_relative_to_supplied_container) {
    const auto original = bnsh();
    Bytes outer(original.size() + 37, 0xab);
    place(outer, 37, original);
    const auto result = parse_bnsh({outer.data() + 37, original.size()});
    const auto* value = accepted(result);
    if (value != nullptr) {
        ARMIGO_SI_CHECK(value->stages[0].shader.control.data == outer.data() + 37 + 0x1000);
    }
}

ARMIGO_SI_TEST(bnsh, header_failures) {
    auto bytes = bnsh();
    rejected(parse_bnsh({nullptr, 2}), ContainerError::Code::InvalidSpan);
    for (std::size_t size = 0; size < 0x88; ++size) {
        rejected(parse_bnsh({bytes.data(), size}), ContainerError::Code::Truncated);
    }
    bytes[7] = 1;
    rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::Magic);
    bytes[7] = 0;
    bytes[0x60] = 0;
    rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::MissingResourceBlock);
}

ARMIGO_SI_TEST(bnsh, rejects_every_pointer_level_and_overflow) {
    for (const auto at : {0x80u, 0x210u, 0x308u, 0x408u, 0x410u}) {
        for (const auto pointer : {std::uint64_t{0x6fff},
                                  std::numeric_limits<std::uint64_t>::max()}) {
            auto bytes = bnsh();
            put64(bytes, at, pointer);
            rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::Range);
        }
    }
}

ARMIGO_SI_TEST(bnsh, rejects_payload_size_disagreement) {
    auto bytes = bnsh();
    test::put_le32(bytes, 0x41c, format::control_min_size + 8);
    rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::ControlSizeMismatch);
    bytes = bnsh();
    test::put_le32(bytes, 0x418, 184);
    rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::CodeSizeMismatch);
    test::put_le32(bytes, 0x418, 168);
    rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::ShaderInvalid);
    bytes = bnsh();
    test::put_le32(bytes, 0x41c, 0x700);
    rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::ShaderInvalid);
}

ARMIGO_SI_TEST(bnsh, rejects_slot_stage_disagreement) {
    auto bytes = bnsh();
    put64(bytes, 0x308, 0x480);
    rejected(parse_bnsh(test::span_of(bytes)), ContainerError::Code::SlotStageMismatch);
}

ARMIGO_SI_TEST(bnsh, retains_nested_shader_error) {
    auto bytes = bnsh();
    test::put_le32(bytes, 0x1004, 9);
    const auto result = parse_bnsh(test::span_of(bytes));
    rejected(result, ContainerError::Code::ShaderInvalid);
    const auto* error = std::get_if<ContainerError>(&result);
    if (error != nullptr) {
        ARMIGO_SI_CHECK(error->shader_error.has_value());
        ARMIGO_SI_CHECK(error->shader_error->code == ParseError::Code::ControlTag);
        ARMIGO_SI_CHECK_EQ(error->offset, 0x400u);
    }
}

ARMIGO_SI_TEST(bnsh, variation_and_stage_limits) {
    const auto bytes = bnsh();
    ContainerLimits limits;
    limits.max_variations = 2;
    rejected(parse_bnsh(test::span_of(bytes), limits), ContainerError::Code::VariationLimit);
    limits.max_variations = 3;
    limits.max_stages = 7;
    rejected(parse_bnsh(test::span_of(bytes), limits), ContainerError::Code::StageLimit);
    limits.max_stages = 8;
    const auto result = parse_bnsh(test::span_of(bytes), limits);
    accepted(result);
}

ARMIGO_SI_TEST(bnsh, zero_variations) {
    auto bytes = bnsh();
    test::put_le32(bytes, 0x7c, 0);
    put64(bytes, 0x80, 0);
    const auto result = parse_bnsh(test::span_of(bytes));
    const auto* value = accepted(result);
    if (value != nullptr) ARMIGO_SI_CHECK(value->stages.empty());
}

ARMIGO_SI_TEST(bahs, signed_entry_pointers_and_deduplication) {
    const auto bytes = bahs();
    const auto result = parse_bahs(test::span_of(bytes));
    const auto* value = accepted(result);
    if (value == nullptr) return;
    ARMIGO_SI_CHECK_EQ(value->stages.size(), 2u);
    ARMIGO_SI_CHECK_EQ(value->candidate_count, 3u);
    ARMIGO_SI_CHECK_EQ(value->duplicate_candidates, 1u);
    ARMIGO_SI_CHECK_EQ(value->rejected_candidates, 0u);
    ARMIGO_SI_CHECK_EQ(value->scanned_positions, (0x2200u - 16u) / 4u + 1u);
    ARMIGO_SI_CHECK_EQ(value->stages[1].descriptor_offset, 0x1c00u);
    ARMIGO_SI_CHECK_EQ(value->stages[1].control_offset, 0xb00u);
    ARMIGO_SI_CHECK_EQ(value->stages[1].code_offset, 0x2300u);
    ARMIGO_SI_CHECK(value->stages[1].shader.stage == Stage::Fragment);
    ARMIGO_SI_CHECK(!value->stages[1].variation.has_value());
    ARMIGO_SI_CHECK(!value->stages[1].slot.has_value());
    ARMIGO_SI_CHECK(value->stages[1].shader.control.data == bytes.data() + 0xb00);
}

ARMIGO_SI_TEST(bahs, shared_control_with_distinct_code_is_preserved) {
    auto bytes = bahs();
    entry(bytes, 0x1c20, 0x200, 0x200);
    test::CodeSpec spec;
    place(bytes, 0x2400, test::build_code(spec));
    const auto result = parse_bahs(test::span_of(bytes));
    const auto* value = accepted(result);
    if (value == nullptr) return;
    ARMIGO_SI_CHECK_EQ(value->stages.size(), 3u);
    ARMIGO_SI_CHECK_EQ(value->stages[2].control_offset, 0x200u);
    ARMIGO_SI_CHECK_EQ(value->stages[2].code_offset, 0x2400u);
}

ARMIGO_SI_TEST(bahs, malformed_recognized_candidates_are_counted) {
    auto bytes = bahs();
    // One out-of-region code and one structurally invalid code, both with
    // correctly identified control structures, must remain visible as rejects.
    entry(bytes, 0x1c20, 0xb00, 0x400);
    entry(bytes, 0x1c30, 0xb00, 0x200);
    const auto result = parse_bahs(test::span_of(bytes));
    const auto* value = accepted(result);
    if (value == nullptr) return;
    ARMIGO_SI_CHECK_EQ(value->rejected_candidates, 2u);
    ARMIGO_SI_CHECK_EQ(value->candidate_count, 5u);
    ARMIGO_SI_CHECK_EQ(value->stages.size(), 2u);
}

ARMIGO_SI_TEST(bahs, invalid_control_and_stage_are_rejected) {
    for (const auto field : {format::control_version_offset, format::stage_field}) {
        auto bytes = bahs();
        test::put_le32(bytes, 0xb00 + field, 99);
        const auto result = parse_bahs(test::span_of(bytes));
        const auto* value = accepted(result);
        if (value == nullptr) continue;
        ARMIGO_SI_CHECK_EQ(value->rejected_candidates, 1u);
        ARMIGO_SI_CHECK_EQ(value->stages.size(), 1u);
    }
}

ARMIGO_SI_TEST(bahs, recognition_requires_complete_consistent_signature) {
    for (const auto field : {0u, 4u, 8u, 12u}) {
        auto bytes = bahs();
        test::put_le32(bytes, 0x1c00 + field, field == 12 ? 0x80000000u : 1u);
        const auto result = parse_bahs(test::span_of(bytes));
        const auto* value = accepted(result);
        if (value == nullptr) continue;
        ARMIGO_SI_CHECK_EQ(value->candidate_count, 2u);
        ARMIGO_SI_CHECK_EQ(value->rejected_candidates, 0u);
    }
    auto bytes = bahs();
    test::put_le32(bytes, 0xb00 + format::control_size_offset, 0x720);
    const auto result = parse_bahs(test::span_of(bytes));
    const auto* value = accepted(result);
    if (value != nullptr) ARMIGO_SI_CHECK_EQ(value->candidate_count, 2u);
}

ARMIGO_SI_TEST(bahs, header_and_code_region_bounds) {
    auto bytes = bahs();
    rejected(parse_bahs({nullptr, 1}), ContainerError::Code::InvalidSpan);
    for (std::size_t size = 0; size < 0x20; ++size) {
        rejected(parse_bahs({bytes.data(), size}), ContainerError::Code::Truncated);
    }
    bytes[0] = 0;
    rejected(parse_bahs(test::span_of(bytes)), ContainerError::Code::Magic);
    for (const auto start : {0u, 0x1fu, 0x2500u, 0xffffffffu}) {
        bytes = bahs();
        test::put_le32(bytes, 0x1c, start);
        rejected(parse_bahs(test::span_of(bytes)), ContainerError::Code::Range);
    }
    bytes = bahs();
    test::put_le32(bytes, 0x18, 0xffffffffu);
    rejected(parse_bahs(test::span_of(bytes)), ContainerError::Code::Range);
}

ARMIGO_SI_TEST(bahs, scan_candidate_and_stage_limits) {
    auto bytes = bahs();
    ContainerLimits limits;
    limits.max_scan_bytes = 0x21ff;
    rejected(parse_bahs(test::span_of(bytes), limits), ContainerError::Code::ScanLimit);
    limits.max_scan_bytes = 0x2200;
    limits.max_candidates = 2;
    rejected(parse_bahs(test::span_of(bytes), limits), ContainerError::Code::CandidateLimit);
    limits.max_candidates = 3;
    limits.max_stages = 1;
    rejected(parse_bahs(test::span_of(bytes), limits), ContainerError::Code::StageLimit);
    limits.max_stages = 2;
    const auto result = parse_bahs(test::span_of(bytes), limits);
    accepted(result);
    entry(bytes, 0x1c20, 0xb00, 0x400);
    rejected(parse_bahs(test::span_of(bytes), limits), ContainerError::Code::CandidateLimit);
}

ARMIGO_SI_TEST(bahs, final_scan_position_is_included) {
    auto bytes = bahs();
    entry(bytes, 0x21f0, 0xb00, 0x100);
    const auto result = parse_bahs(test::span_of(bytes));
    const auto* value = accepted(result);
    if (value != nullptr) {
        ARMIGO_SI_CHECK_EQ(value->duplicate_candidates, 2u);
    }
}
