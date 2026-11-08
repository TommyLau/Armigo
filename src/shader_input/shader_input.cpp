// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Structural validation for the native shader input containers.
//
// The acceptance set mirrors the primary implementation's generic capture
// bounds logic exactly; only diagnostic granularity differs, because this
// parser reports a specific structured code per rejected condition while the
// primary source groups them into combined messages.

#include <armigo/shader_input/shader_input.h>

#include <cstring>

namespace armigo::shader_input {
namespace {

std::uint32_t read_u32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) |
           static_cast<std::uint32_t>(p[1]) << 8 |
           static_cast<std::uint32_t>(p[2]) << 16 |
           static_cast<std::uint32_t>(p[3]) << 24;
}

ParseError error(ParseError::Code code, std::uint64_t value = 0,
                 std::uint64_t bound = 0) noexcept {
    return ParseError{code, value, bound};
}

}  // namespace

const char* ParseError::message() const noexcept {
    switch (code) {
        case Code::NullControlData:
            return "control data pointer is null but the span size is not zero";
        case Code::NullCodeData:
            return "code data pointer is null but the span size is not zero";
        case Code::NullExtendedMetadata:
            return "extended metadata pointer is null but the span size is "
                   "not zero";
        case Code::ControlSpanTooSmall:
            return "control span is truncated below the prefix";
        case Code::ControlMagic:
            return "control magic does not match";
        case Code::ControlTag:
            return "control tag word is not 1";
        case Code::ControlVersion:
            return "control version is not 9, 14, 15 or 16";
        case Code::ControlSizeTooSmall:
            return "declared control size is below the minimum";
        case Code::ControlSizeTooLarge:
            return "declared control size exceeds the conservative ceiling";
        case Code::ControlSizeBeyondSpan:
            return "declared control size exceeds the provided span";
        case Code::ExtendedMetadataSize:
            return "extended metadata is present but not exactly 24 bytes";
        case Code::ExtendedMetadataShortControl:
            return "extended metadata requires a larger control structure";
        case Code::ExtendedMetadataMismatch:
            return "extended metadata differs from the control; override "
                   "semantics are unsupported";
        case Code::StageOutOfRange:
            return "control stage number exceeds 5";
        case Code::CodeSizeZero:
            return "declared code size is zero";
        case Code::CodeSizeTooLarge:
            return "declared code size exceeds the conservative ceiling";
        case Code::CodeSizeMisaligned:
            return "declared code size is not a multiple of 8";
        case Code::C1OffsetBeyondCodeSize:
            return "C1 offset exceeds the declared code size";
        case Code::CodeOffsetBeyondC1Offset:
            return "code offset exceeds the C1 offset";
        case Code::C1OffsetMisaligned:
            return "C1 offset is not a multiple of 8";
        case Code::ComputeHeaderLayout:
            return "compute stage requires code offset 0 and C1 offset >= 8";
        case Code::GraphicsHeaderLayout:
            return "graphics stage requires code offset 0x30 and C1 offset "
                   ">= 0x88";
        case Code::CodeSpanTooSmall:
            return "declared code size exceeds the provided span";
        case Code::WrapperMagic:
            return "graphics code wrapper magic does not match";
        case Code::SphStageMismatch:
            return "SPH stage type does not match the control stage";
    }
    return "unknown parse error";
}

std::uint32_t ParsedShader::scheduling_group_count() const noexcept {
    const std::uint32_t remainder = program_size % format::scheduling_group_size;
    return program_size / format::scheduling_group_size +
           (remainder != 0 ? 1u : 0u);
}

ByteSpan ParsedShader::scheduling_word(std::uint32_t group) const noexcept {
    const std::uint64_t offset =
        static_cast<std::uint64_t>(group) * format::scheduling_group_size;
    if (offset + format::scheduling_word_size > program_size) {
        return {};
    }
    return {program.data + offset, format::scheduling_word_size};
}

std::uint32_t ParsedShader::instruction_slot_count() const noexcept {
    const std::uint32_t full =
        program_size / format::scheduling_group_size;
    const std::uint32_t remainder =
        program_size % format::scheduling_group_size;
    // A truncated trailing group of 8 bytes holds only its scheduling word;
    // 16- and 24-byte remainders hold one and two instruction slots.
    const std::uint32_t partial_slots =
        remainder >= 2 * format::instruction_word_size
            ? remainder / format::instruction_word_size - 1u
            : 0u;
    return full * 3u + partial_slots;
}

InstructionSlot ParsedShader::instruction_slot(
    std::uint32_t index) const noexcept {
    const std::uint32_t full =
        program_size / format::scheduling_group_size;
    const std::uint32_t remainder =
        program_size % format::scheduling_group_size;
    const std::uint32_t partial_slots =
        remainder >= 2 * format::instruction_word_size
            ? remainder / format::instruction_word_size - 1u
            : 0u;
    if (index >= full * 3u + partial_slots) {
        return {};
    }
    std::uint32_t offset;
    if (index < full * 3u) {
        offset = format::scheduling_group_size * (index / 3u) +
                 format::instruction_word_size * (index % 3u + 1u);
    } else {
        offset = format::scheduling_group_size * full +
                 format::instruction_word_size *
                     (1u + (index - full * 3u));
    }
    return InstructionSlot{offset, program_offset + offset};
}

std::uint32_t ParsedShader::sph_stage_type() const noexcept {
    if (sph.empty()) {
        return 0u;
    }
    return (read_u32(sph.data) >> format::sph_stage_shift) &
           format::sph_stage_mask;
}

ParseResult parse_shader(const ShaderInput& input) noexcept {
    // Input contract checks, then the primary implementation's validation
    // order: control prefix, control size bounds, extended metadata, code
    // layout fields, code span coverage, and finally the graphics wrapper
    // and SPH stage agreement.
    if (input.control.data == nullptr && input.control.size != 0) {
        return error(ParseError::Code::NullControlData);
    }
    if (input.control.size < format::control_prefix_size) {
        return error(ParseError::Code::ControlSpanTooSmall,
                     input.control.size, format::control_prefix_size);
    }
    const std::uint32_t magic = read_u32(input.control.data);
    const std::uint32_t tag =
        read_u32(input.control.data + 4u);
    const std::uint32_t version =
        read_u32(input.control.data + format::control_version_offset);
    const std::uint32_t declared_control_size =
        read_u32(input.control.data + format::control_size_offset);

    if (magic != format::control_magic) {
        return error(ParseError::Code::ControlMagic, magic,
                     format::control_magic);
    }
    if (tag != format::control_tag) {
        return error(ParseError::Code::ControlTag, tag, format::control_tag);
    }
    if (!format::is_accepted_version(version)) {
        return error(ParseError::Code::ControlVersion, version, 0);
    }
    if (declared_control_size < format::control_min_size) {
        return error(ParseError::Code::ControlSizeTooSmall,
                     declared_control_size, format::control_min_size);
    }
    if (declared_control_size > limits::control_max_size) {
        return error(ParseError::Code::ControlSizeTooLarge,
                     declared_control_size, limits::control_max_size);
    }
    if (declared_control_size > input.control.size) {
        return error(ParseError::Code::ControlSizeBeyondSpan,
                     declared_control_size, input.control.size);
    }

    const bool has_extended =
        input.extended_metadata.data != nullptr ||
        input.extended_metadata.size != 0;
    if (has_extended) {
        if (input.extended_metadata.data == nullptr) {
            return error(ParseError::Code::NullExtendedMetadata);
        }
        if (input.extended_metadata.size !=
            format::extended_metadata_size) {
            return error(ParseError::Code::ExtendedMetadataSize,
                         input.extended_metadata.size,
                         format::extended_metadata_size);
        }
        if (declared_control_size <
            format::extended_metadata_min_control_size) {
            return error(ParseError::Code::ExtendedMetadataShortControl,
                         declared_control_size,
                         format::extended_metadata_min_control_size);
        }
        if (std::memcmp(input.extended_metadata.data,
                        input.control.data + format::extended_metadata_offset,
                        format::extended_metadata_size) != 0) {
            return error(ParseError::Code::ExtendedMetadataMismatch, 0, 0);
        }
    }

    const std::uint32_t stage =
        read_u32(input.control.data + format::stage_field);
    const std::uint32_t code_size =
        read_u32(input.control.data + format::code_size_field);
    const std::uint32_t code_offset =
        read_u32(input.control.data + format::code_offset_field);
    const std::uint32_t c1_offset =
        read_u32(input.control.data + format::c1_offset_field);

    if (stage > static_cast<std::uint32_t>(Stage::Compute)) {
        return error(ParseError::Code::StageOutOfRange, stage, 5);
    }
    if (input.code.data == nullptr && input.code.size != 0) {
        return error(ParseError::Code::NullCodeData);
    }
    if (code_size == 0) {
        return error(ParseError::Code::CodeSizeZero, 0, 0);
    }
    if (code_size > limits::code_max_size) {
        return error(ParseError::Code::CodeSizeTooLarge, code_size,
                     limits::code_max_size);
    }
    if ((code_size % format::instruction_word_size) != 0) {
        return error(ParseError::Code::CodeSizeMisaligned, code_size,
                     format::instruction_word_size);
    }
    if (c1_offset > code_size) {
        return error(ParseError::Code::C1OffsetBeyondCodeSize, c1_offset,
                     code_size);
    }
    if (code_offset > c1_offset) {
        return error(ParseError::Code::CodeOffsetBeyondC1Offset, code_offset,
                     c1_offset);
    }
    if ((c1_offset % format::instruction_word_size) != 0) {
        return error(ParseError::Code::C1OffsetMisaligned, c1_offset,
                     format::instruction_word_size);
    }
    const bool compute = stage == static_cast<std::uint32_t>(Stage::Compute);
    if (compute) {
        if (code_offset != format::compute_code_offset ||
            c1_offset < format::compute_min_c1_offset) {
            return error(ParseError::Code::ComputeHeaderLayout, code_offset,
                         c1_offset);
        }
    } else {
        if (code_offset != format::graphics_code_offset ||
            c1_offset < format::graphics_min_c1_offset) {
            return error(ParseError::Code::GraphicsHeaderLayout, code_offset,
                         c1_offset);
        }
    }
    if (input.code.size < code_size) {
        return error(ParseError::Code::CodeSpanTooSmall, code_size,
                     input.code.size);
    }
    if (!compute) {
        if (read_u32(input.code.data) != format::code_wrapper_magic) {
            return error(ParseError::Code::WrapperMagic,
                         read_u32(input.code.data),
                         format::code_wrapper_magic);
        }
        const std::uint32_t sph_word = read_u32(input.code.data + code_offset);
        const std::uint32_t sph_type =
            (sph_word >> format::sph_stage_shift) & format::sph_stage_mask;
        if (sph_type != sph_stage_type_for(static_cast<Stage>(stage))) {
            return error(ParseError::Code::SphStageMismatch, sph_type,
                         sph_stage_type_for(static_cast<Stage>(stage)));
        }
    }

    ParsedShader parsed;
    parsed.version = version;
    parsed.stage = static_cast<Stage>(stage);
    parsed.control_size = declared_control_size;
    parsed.code_size = code_size;
    parsed.code_offset = code_offset;
    parsed.c1_offset = c1_offset;
    parsed.c1_size = code_size - c1_offset;
    parsed.program_offset =
        compute ? format::compute_program_offset
                : format::graphics_program_offset;
    parsed.program_size = c1_offset - parsed.program_offset;
    parsed.control = {input.control.data, declared_control_size};
    parsed.code = {input.code.data, code_size};
    if (!compute) {
        parsed.wrapper = {input.code.data, format::code_wrapper_size};
        parsed.sph = {input.code.data + code_offset, format::sph_size};
    }
    parsed.program = {input.code.data + parsed.program_offset,
                      parsed.program_size};
    parsed.c1 = {input.code.data + c1_offset, parsed.c1_size};
    return parsed;
}

}  // namespace armigo::shader_input
