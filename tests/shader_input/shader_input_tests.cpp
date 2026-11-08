// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic structural tests for the native shader input parser. All fixtures
// are independently authored byte patterns; no game content is used.

#include "synthetic.h"
#include "test_support.h"

#include <vector>

namespace {

using namespace armigo::shader_input;

}  // namespace

// ---------------------------------------------------------------------------
// control group: prefix, magic, tag, version, size bounds, truncation, tails.
// ---------------------------------------------------------------------------

ARMIGO_SI_TEST(control, accepts_versions_9_14_15_and_16) {
    for (const std::uint32_t version : {9u, 14u, 15u, 16u}) {
        const test::CodeSpec code;
        test::ControlSpec spec = test::control_for(code);
        spec.version = version;
        const std::vector<std::uint8_t> ctl = test::build_control(spec);
        const std::vector<std::uint8_t> bin = test::build_code(code);
        const ParsedShader parsed = test::must_parse(
            {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
        ARMIGO_SI_CHECK_EQ(parsed.version, version);
    }
}

ARMIGO_SI_TEST(control, rejects_wrong_magic) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.magic = 0x98761235u;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                      ParseError::Code::ControlMagic, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(control, rejects_wrong_tag) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.tag = 2;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                      ParseError::Code::ControlTag, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(control, rejects_unsupported_version) {
    for (const std::uint32_t version : {0u, 8u, 10u, 13u, 17u}) {
        test::CodeSpec code;
        test::ControlSpec spec = test::control_for(code);
        spec.version = version;
        const std::vector<std::uint8_t> ctl = test::build_control(spec);
        const std::vector<std::uint8_t> bin = test::build_code(code);
        test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                          ParseError::Code::ControlVersion, __FILE__,
                          __LINE__);
    }
}

ARMIGO_SI_TEST(control, rejects_span_below_prefix) {
    test::CodeSpec code;
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    for (const std::size_t size : {std::size_t{0}, std::size_t{0x23}}) {
        test::must_reject({{ctl.data(), size}, test::span_of(bin), {}},
                          ParseError::Code::ControlSpanTooSmall, __FILE__,
                          __LINE__);
    }
}

ARMIGO_SI_TEST(control, rejects_declared_size_below_minimum) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.size_field = format::control_min_size - 1;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                      ParseError::Code::ControlSizeTooSmall, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(control, rejects_declared_size_above_ceiling) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.size_field = limits::control_max_size + 1;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                      ParseError::Code::ControlSizeTooLarge, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(control, accepts_declared_size_at_ceiling) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.size_field = limits::control_max_size;
    spec.buffer_size = limits::control_max_size;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.control_size, limits::control_max_size);
    ARMIGO_SI_CHECK_EQ(parsed.control.size,
                       static_cast<std::size_t>(limits::control_max_size));
}

ARMIGO_SI_TEST(control, rejects_declared_size_beyond_span) {
    test::CodeSpec code;
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    // One byte short of the declared 0x718.
    test::must_reject({{ctl.data(), ctl.size() - 1}, test::span_of(bin), {}},
                      ParseError::Code::ControlSizeBeyondSpan, __FILE__,
                      __LINE__);
    // Exactly the declared size is accepted.
    test::must_parse({{ctl.data(), format::control_min_size},
                      test::span_of(bin),
                      {}},
                     __FILE__, __LINE__);
}

ARMIGO_SI_TEST(control, accepts_backing_larger_than_declared) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.size_field = format::control_min_size;
    spec.buffer_size = format::control_min_size + 0x48;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.control.size,
                       static_cast<std::size_t>(format::control_min_size));
    ARMIGO_SI_CHECK(parsed.control.data == ctl.data());
}

ARMIGO_SI_TEST(control, preserves_opaque_tail) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.buffer_size = format::extended_metadata_min_control_size;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK(test::bytes_equal(
        parsed.control, test::slice(test::span_of(ctl), 0, spec.buffer_size)));
    // Bytes beyond the parsed header fields are preserved verbatim.
    ARMIGO_SI_CHECK(test::bytes_equal(
        {parsed.control.data + format::control_min_size,
         parsed.control.size - format::control_min_size},
        test::slice(test::span_of(ctl), format::control_min_size,
                    spec.buffer_size - format::control_min_size)));
}

ARMIGO_SI_TEST(control, rejects_null_control_data) {
    const test::CodeSpec code;
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({{nullptr, 0x100}, test::span_of(bin), {}},
                      ParseError::Code::NullControlData, __FILE__, __LINE__);
}

// ---------------------------------------------------------------------------
// code group: stages, layout bounds, wrapper/SPH agreement, C1 handling.
// ---------------------------------------------------------------------------

ARMIGO_SI_TEST(code, accepts_all_graphics_stages) {
    for (const std::uint32_t stage : {0u, 1u, 2u, 3u, 4u}) {
        const test::CodeSpec code_spec{stage, 64, 32};
        const std::vector<std::uint8_t> ctl =
            test::build_control(test::control_for(code_spec));
        const std::vector<std::uint8_t> bin = test::build_code(code_spec);
        const ParsedShader parsed = test::must_parse(
            {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
        ARMIGO_SI_CHECK_EQ(static_cast<std::uint32_t>(parsed.stage), stage);
        ARMIGO_SI_CHECK(!parsed.is_compute());
        ARMIGO_SI_CHECK_EQ(parsed.code_offset, format::graphics_code_offset);
        ARMIGO_SI_CHECK_EQ(parsed.program_offset,
                           format::graphics_program_offset);
        ARMIGO_SI_CHECK_EQ(parsed.program_size, 64u);
        ARMIGO_SI_CHECK_EQ(parsed.sph_stage_type(),
                           sph_stage_type_for(parsed.stage));
        ARMIGO_SI_CHECK_EQ(parsed.wrapper.size,
                           static_cast<std::size_t>(format::code_wrapper_size));
        ARMIGO_SI_CHECK_EQ(parsed.sph.size,
                           static_cast<std::size_t>(format::sph_size));
        // The SPH span starts right after the wrapper.
        ARMIGO_SI_CHECK(parsed.sph.data ==
                        parsed.code.data + format::code_wrapper_size);
    }
}

ARMIGO_SI_TEST(code, accepts_compute_without_wrapper) {
    const test::CodeSpec code_spec{5u, 40, 24};
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code_spec));
    const std::vector<std::uint8_t> bin = test::build_code(code_spec);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK(parsed.is_compute());
    ARMIGO_SI_CHECK_EQ(parsed.code_offset, 0u);
    ARMIGO_SI_CHECK_EQ(parsed.program_offset, 0u);
    ARMIGO_SI_CHECK(parsed.wrapper.empty());
    ARMIGO_SI_CHECK(parsed.sph.empty());
    ARMIGO_SI_CHECK_EQ(parsed.sph_stage_type(), 0u);
    ARMIGO_SI_CHECK_EQ(parsed.program.size, static_cast<std::size_t>(40));
}

ARMIGO_SI_TEST(code, rejects_stage_out_of_range) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.stage = 6;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                      ParseError::Code::StageOutOfRange, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(code, rejects_zero_code_size) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.code_size = 0;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    test::must_reject({test::span_of(ctl), {}, {}},
                      ParseError::Code::CodeSizeZero, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(code, rejects_code_size_above_ceiling) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.code_size = limits::code_max_size + 8;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    test::must_reject({test::span_of(ctl), {}, {}},
                      ParseError::Code::CodeSizeTooLarge, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(code, rejects_misaligned_code_size) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.code_size = 0x89;  // 0x88 minimum plus one stray byte.
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    test::must_reject({test::span_of(ctl), {}, {}},
                      ParseError::Code::CodeSizeMisaligned, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(code, rejects_c1_offset_beyond_code_size) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.c1_offset = spec.code_size + 8;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    test::must_reject({test::span_of(ctl), {}, {}},
                      ParseError::Code::C1OffsetBeyondCodeSize, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(code, rejects_code_offset_beyond_c1_offset) {
    test::CodeSpec code{5u, 40, 24};  // Compute: code offset must be 0.
    test::ControlSpec spec = test::control_for(code);
    spec.code_offset = 0x10;
    spec.c1_offset = 8;
    spec.code_size = 0x10;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    test::must_reject({test::span_of(ctl), {}, {}},
                      ParseError::Code::CodeOffsetBeyondC1Offset, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(code, rejects_misaligned_c1_offset) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.c1_offset = format::graphics_min_c1_offset + 4;  // 0x8c
    spec.code_size = 0x90;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    test::must_reject({test::span_of(ctl), {}, {}},
                      ParseError::Code::C1OffsetMisaligned, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(code, rejects_graphics_layout_violations) {
    // Wrong wrapper size: code offset 0x28.
    {
        test::CodeSpec code;
        test::ControlSpec spec = test::control_for(code);
        spec.code_offset = 0x28;
        spec.c1_offset = 0x88;
        spec.code_size = 0x98;
        const std::vector<std::uint8_t> ctl = test::build_control(spec);
        test::must_reject({test::span_of(ctl), {}, {}},
                          ParseError::Code::GraphicsHeaderLayout, __FILE__,
                          __LINE__);
    }
    // C1 offset below 0x88 leaves no room for wrapper, SPH, and a program.
    {
        test::CodeSpec code;
        test::ControlSpec spec = test::control_for(code);
        spec.c1_offset = 0x80;
        spec.code_size = 0x88;
        const std::vector<std::uint8_t> ctl = test::build_control(spec);
        test::must_reject({test::span_of(ctl), {}, {}},
                          ParseError::Code::GraphicsHeaderLayout, __FILE__,
                          __LINE__);
    }
}

ARMIGO_SI_TEST(code, rejects_compute_layout_violations) {
    // Compute with a nonzero code offset.
    {
        test::CodeSpec code{5u, 40, 24};
        test::ControlSpec spec = test::control_for(code);
        spec.code_offset = 8;
        spec.c1_offset = 0x10;
        spec.code_size = 0x20;
        const std::vector<std::uint8_t> ctl = test::build_control(spec);
        test::must_reject({test::span_of(ctl), {}, {}},
                          ParseError::Code::ComputeHeaderLayout, __FILE__,
                          __LINE__);
    }
    // Compute with an empty program region (C1 offset below 8).
    {
        test::CodeSpec code{5u, 40, 24};
        test::ControlSpec spec = test::control_for(code);
        spec.code_offset = 0;
        spec.c1_offset = 0;
        spec.code_size = 8;
        const std::vector<std::uint8_t> ctl = test::build_control(spec);
        test::must_reject({test::span_of(ctl), {}, {}},
                          ParseError::Code::ComputeHeaderLayout, __FILE__,
                          __LINE__);
    }
}

ARMIGO_SI_TEST(code, rejects_wrapper_magic) {
    test::CodeSpec code;
    code.wrapper_magic = 0x12345679u;
    test::ControlSpec spec = test::control_for(code);
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                      ParseError::Code::WrapperMagic, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(code, rejects_sph_stage_mismatch) {
    // Fragment control stage paired with the vertex SPH type.
    test::CodeSpec code{1u, 64, 32};
    code.sph_type = 1;
    test::ControlSpec spec = test::control_for(code);
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin), {}},
                      ParseError::Code::SphStageMismatch, __FILE__, __LINE__);
    // The fragment SPH type (5) is accepted for the same control.
    code.sph_type = 5;
    const std::vector<std::uint8_t> fixed = test::build_code(code);
    test::must_parse({test::span_of(ctl), test::span_of(fixed), {}}, __FILE__,
                     __LINE__);
}

ARMIGO_SI_TEST(code, rejects_code_span_truncation) {
    test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    // Backing one byte short of the declared code size.
    test::must_reject({test::span_of(ctl),
                       {bin.data(), bin.size() - 1},
                       {}},
                      ParseError::Code::CodeSpanTooSmall, __FILE__, __LINE__);
    // Larger backing than declared is accepted and trimmed to code_size.
    std::vector<std::uint8_t> padded = bin;
    padded.resize(bin.size() + 16, 0);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(padded), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.code.size,
                       static_cast<std::size_t>(spec.code_size));
}

ARMIGO_SI_TEST(code, accepts_empty_c1_segment) {
    test::CodeSpec code{0u, 8, 0};  // Minimal graphics program, no C1 bytes.
    test::ControlSpec spec = test::control_for(code);
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.c1_size, 0u);
    ARMIGO_SI_CHECK(parsed.c1.empty());
    ARMIGO_SI_CHECK(parsed.c1.data == parsed.code.data + parsed.code_size);
    ARMIGO_SI_CHECK_EQ(parsed.program.size, static_cast<std::size_t>(8));
}

ARMIGO_SI_TEST(code, accepts_compute_minimal) {
    test::CodeSpec code{5u, 8, 0};
    test::ControlSpec spec = test::control_for(code);
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.program.size, static_cast<std::size_t>(8));
    ARMIGO_SI_CHECK_EQ(parsed.scheduling_group_count(), 1u);
    ARMIGO_SI_CHECK_EQ(parsed.instruction_slot_count(), 0u);
}

ARMIGO_SI_TEST(code, rejects_null_code_data) {
    test::CodeSpec code;
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    test::must_reject({test::span_of(ctl), {nullptr, 0x100}, {}},
                      ParseError::Code::NullCodeData, __FILE__, __LINE__);
}

// ---------------------------------------------------------------------------
// structure group: scheduling groups, instruction word slots, span identity.
// ---------------------------------------------------------------------------

ARMIGO_SI_TEST(structure, enumerates_scheduling_word_positions) {
    // 72 program bytes = two full groups plus a bare 8-byte trailing group.
    const test::CodeSpec code{0u, 72, 16};
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.scheduling_group_count(), 3u);
    const std::uint32_t expected_offsets[3] = {0x80, 0xa0, 0xc0};
    for (std::uint32_t group = 0; group < 3; ++group) {
        const ByteSpan word = parsed.scheduling_word(group);
        ARMIGO_SI_CHECK_EQ(word.size,
                           static_cast<std::size_t>(
                               format::scheduling_word_size));
        ARMIGO_SI_CHECK(word.data ==
                        parsed.code.data + expected_offsets[group]);
        // Scheduling words are exposed verbatim, as opaque data.
        ARMIGO_SI_CHECK(test::bytes_equal(
            word, test::slice(test::span_of(bin), expected_offsets[group],
                              format::scheduling_word_size)));
    }
}

ARMIGO_SI_TEST(structure, enumerates_instruction_word_pcs) {
    const test::CodeSpec code{0u, 72, 16};
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.instruction_slot_count(), 6u);
    const std::uint32_t expected_pcs[6] = {8, 16, 24, 40, 48, 56};
    const std::uint32_t expected_code_offsets[6] = {0x88, 0x90, 0x98,
                                                    0xa8, 0xb0, 0xb8};
    for (std::uint32_t i = 0; i < 6; ++i) {
        const InstructionSlot slot = parsed.instruction_slot(i);
        ARMIGO_SI_CHECK_EQ(slot.pc, expected_pcs[i]);
        ARMIGO_SI_CHECK_EQ(slot.code_offset, expected_code_offsets[i]);
    }
}

ARMIGO_SI_TEST(structure, truncated_trailing_group_variants) {
    struct Variant {
        std::uint32_t program_bytes;
        std::uint32_t groups;
        std::uint32_t slots;
    };
    const Variant variants[] = {
        {8u, 1u, 0u},    {16u, 1u, 1u},   {24u, 1u, 2u},   {32u, 1u, 3u},
        {40u, 2u, 3u},   {48u, 2u, 4u},   {56u, 2u, 5u},   {64u, 2u, 6u},
        {72u, 3u, 6u},
    };
    for (const Variant& variant : variants) {
        const test::CodeSpec code{0u, variant.program_bytes, 8};
        const std::vector<std::uint8_t> ctl =
            test::build_control(test::control_for(code));
        const std::vector<std::uint8_t> bin = test::build_code(code);
        const ParsedShader parsed = test::must_parse(
            {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
        ARMIGO_SI_CHECK_EQ(parsed.scheduling_group_count(), variant.groups);
        ARMIGO_SI_CHECK_EQ(parsed.instruction_slot_count(), variant.slots);
        // Every slot offset is a multiple of 8, at least 8, in program bounds,
        // and never inside a scheduling word.
        for (std::uint32_t i = 0; i < variant.slots; ++i) {
            const InstructionSlot slot = parsed.instruction_slot(i);
            ARMIGO_SI_CHECK(slot.pc >= 8);
            ARMIGO_SI_CHECK_EQ(slot.pc % 8u, 0u);
            ARMIGO_SI_CHECK(slot.pc < variant.program_bytes);
            ARMIGO_SI_CHECK_EQ((slot.pc - 8u) % 32u < 24u, true);
        }
    }
}

ARMIGO_SI_TEST(structure, compute_program_starts_at_zero) {
    const test::CodeSpec code{5u, 40, 24};
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK(parsed.program.data == parsed.code.data);
    ARMIGO_SI_CHECK_EQ(parsed.scheduling_group_count(), 2u);
    ARMIGO_SI_CHECK_EQ(parsed.instruction_slot_count(), 3u);
    for (std::uint32_t i = 0; i < 3; ++i) {
        const InstructionSlot slot = parsed.instruction_slot(i);
        ARMIGO_SI_CHECK_EQ(slot.code_offset, slot.pc);
    }
    // The first scheduling word sits at code byte 0.
    ARMIGO_SI_CHECK(parsed.scheduling_word(0).data == parsed.code.data);
}

ARMIGO_SI_TEST(structure, out_of_range_access_is_inert) {
    const test::CodeSpec code{0u, 72, 16};
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK(parsed.scheduling_word(parsed.scheduling_group_count())
                        .empty());
    ARMIGO_SI_CHECK(parsed.scheduling_word(0xffffffffu).empty());
    const InstructionSlot zeroed = parsed.instruction_slot(
        parsed.instruction_slot_count());
    ARMIGO_SI_CHECK_EQ(zeroed.pc, 0u);
    ARMIGO_SI_CHECK_EQ(zeroed.code_offset, 0u);
    const InstructionSlot huge = parsed.instruction_slot(0xffffffffu);
    ARMIGO_SI_CHECK_EQ(huge.pc, 0u);
    ARMIGO_SI_CHECK_EQ(huge.code_offset, 0u);
}

ARMIGO_SI_TEST(structure, spans_borrow_input_buffers) {
    const test::CodeSpec code{2u, 64, 32};
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK(parsed.control.data == ctl.data());
    ARMIGO_SI_CHECK(parsed.code.data == bin.data());
    ARMIGO_SI_CHECK(parsed.wrapper.data == bin.data());
    ARMIGO_SI_CHECK(parsed.sph.data == bin.data() + 0x30);
    ARMIGO_SI_CHECK(parsed.program.data ==
                    bin.data() + format::graphics_program_offset);
    ARMIGO_SI_CHECK(parsed.c1.data ==
                    bin.data() + parsed.c1_offset);
    // Region contents are preserved verbatim from the borrowed buffers.
    ARMIGO_SI_CHECK(test::bytes_equal(
        parsed.program,
        test::slice(test::span_of(bin), parsed.program_offset, 64)));
    ARMIGO_SI_CHECK(test::bytes_equal(
        parsed.c1, test::slice(test::span_of(bin), parsed.c1_offset, 32)));
    ARMIGO_SI_CHECK(test::bytes_equal(
        parsed.sph, test::slice(test::span_of(bin), 0x30, format::sph_size)));
}

ARMIGO_SI_TEST(structure, regions_partition_the_code_buffer) {
    const test::CodeSpec code{4u, 96, 40};
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    const ParsedShader parsed = test::must_parse(
        {test::span_of(ctl), test::span_of(bin), {}}, __FILE__, __LINE__);
    ARMIGO_SI_CHECK_EQ(parsed.wrapper.size +
                           parsed.sph.size + parsed.program.size +
                           parsed.c1.size,
                       parsed.code.size);
    // Every scheduling word lies inside the program region; every slot lies
    // after its group's scheduling word.
    for (std::uint32_t group = 0;
         group < parsed.scheduling_group_count(); ++group) {
        const ByteSpan word = parsed.scheduling_word(group);
        ARMIGO_SI_CHECK(word.data >= parsed.program.data);
        ARMIGO_SI_CHECK(word.data + word.size <=
                        parsed.program.data + parsed.program.size);
    }
}

// ---------------------------------------------------------------------------
// metadata group: extended-descriptor override handling.
// ---------------------------------------------------------------------------

ARMIGO_SI_TEST(metadata, accepts_absent_extended_metadata) {
    const test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.buffer_size = format::extended_metadata_min_control_size;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    // Absent metadata never touches the extended comparison.
    test::must_parse({test::span_of(ctl), test::span_of(bin), {}}, __FILE__,
                     __LINE__);
}

ARMIGO_SI_TEST(metadata, accepts_identical_extended_metadata) {
    const test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.buffer_size = format::extended_metadata_min_control_size;
    spec.extended.assign(format::extended_metadata_size, 0);
    for (std::uint32_t i = 0; i < format::extended_metadata_size; ++i) {
        spec.extended[i] = static_cast<std::uint8_t>(0x40u + i);
    }
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_parse(
        {test::span_of(ctl), test::span_of(bin),
         test::span_of(test::matching_extended(spec))},
        __FILE__, __LINE__);
}

ARMIGO_SI_TEST(metadata, rejects_differing_extended_metadata) {
    const test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.buffer_size = format::extended_metadata_min_control_size;
    spec.extended.assign(format::extended_metadata_size, 0x11);
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    std::vector<std::uint8_t> different =
        test::matching_extended(spec);
    different[format::extended_metadata_size - 1] ^= 0xff;
    test::must_reject({test::span_of(ctl), test::span_of(bin),
                       test::span_of(different)},
                      ParseError::Code::ExtendedMetadataMismatch, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(metadata, rejects_wrong_extended_metadata_size) {
    const test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.buffer_size = format::extended_metadata_min_control_size;
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    std::vector<std::uint8_t> short_meta(format::extended_metadata_size - 1,
                                         0);
    std::vector<std::uint8_t> long_meta(format::extended_metadata_size + 1, 0);
    test::must_reject({test::span_of(ctl), test::span_of(bin),
                       test::span_of(short_meta)},
                      ParseError::Code::ExtendedMetadataSize, __FILE__,
                      __LINE__);
    test::must_reject({test::span_of(ctl), test::span_of(bin),
                       test::span_of(long_meta)},
                      ParseError::Code::ExtendedMetadataSize, __FILE__,
                      __LINE__);
}

ARMIGO_SI_TEST(metadata, rejects_short_control_for_extended_metadata) {
    const test::CodeSpec code;
    test::ControlSpec spec = test::control_for(code);
    spec.buffer_size = format::extended_metadata_min_control_size;
    spec.size_field = format::control_min_size;  // Declared below 0x780.
    spec.extended.assign(format::extended_metadata_size, 0x22);
    const std::vector<std::uint8_t> ctl = test::build_control(spec);
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject(
        {test::span_of(ctl), test::span_of(bin),
         test::span_of(test::matching_extended(spec))},
        ParseError::Code::ExtendedMetadataShortControl, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(metadata, rejects_null_extended_metadata) {
    const test::CodeSpec code;
    const std::vector<std::uint8_t> ctl =
        test::build_control(test::control_for(code));
    const std::vector<std::uint8_t> bin = test::build_code(code);
    test::must_reject({test::span_of(ctl), test::span_of(bin),
                       {nullptr, format::extended_metadata_size}},
                      ParseError::Code::NullExtendedMetadata, __FILE__,
                      __LINE__);
}
