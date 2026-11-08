// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Structural parser for the native shader input containers consumed by
// Armigo's independent Maxwell decoder.
//
// The parser validates a borrowed control buffer and a borrowed code buffer
// and exposes immutable spans for the control structure (including its opaque
// tail), the graphics wrapper, the SPH header, the program region, and the
// trailing C1 constant segment. It also enumerates 32-byte scheduling groups
// and the instruction word positions inside them. It deliberately does not
// decode opcodes, scheduling tokens, or any SPH field beyond the stage type.
//
// Lifetime contract: every span produced by this parser borrows the caller's
// buffers. parse_shader allocates no memory and copies no bytes; a parsed
// result is only valid while the input buffers remain alive and unmodified.
#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

namespace armigo::shader_input {

// Immutable borrowed view over bytes owned by the caller.
struct ByteSpan {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
};

// Container format facts re-verified against the pinned primary
// implementation (generic capture header and bounds logic). These are facts
// of the shader input container, not statements about any game title.
namespace format {

inline constexpr std::uint32_t control_magic = 0x98761234u;
inline constexpr std::uint32_t control_tag = 1u;  // LE32 at offset 4.
inline constexpr std::uint32_t control_prefix_size = 0x24u;
inline constexpr std::uint32_t control_version_offset = 0x8u;
inline constexpr std::uint32_t control_size_offset = 0x20u;
inline constexpr std::uint32_t control_min_size = 0x718u;
inline constexpr std::uint32_t c1_offset_field = 0x700u;
inline constexpr std::uint32_t code_size_field = 0x704u;
inline constexpr std::uint32_t code_offset_field = 0x708u;
inline constexpr std::uint32_t stage_field = 0x714u;
inline constexpr std::uint32_t extended_metadata_offset = 0x768u;
inline constexpr std::uint32_t extended_metadata_size = 24u;
// A control that must back the extended metadata comparison needs at least
// this many bytes.
inline constexpr std::uint32_t extended_metadata_min_control_size = 0x780u;

inline constexpr std::uint32_t code_wrapper_magic = 0x12345678u;
inline constexpr std::uint32_t code_wrapper_size = 0x30u;
inline constexpr std::uint32_t sph_size = 0x50u;
inline constexpr std::uint32_t sph_stage_shift = 10u;
inline constexpr std::uint32_t sph_stage_mask = 0xfu;

// The program region is a sequence of 32-byte groups: one 8-byte scheduling
// word followed by up to three 8-byte instruction word slots. A trailing
// group may be truncated to 8, 16, or 24 bytes; a bare 8-byte group carries
// only its scheduling word.
inline constexpr std::uint32_t scheduling_group_size = 32u;
inline constexpr std::uint32_t scheduling_word_size = 8u;
inline constexpr std::uint32_t instruction_word_size = 8u;

inline constexpr std::uint32_t graphics_code_offset = code_wrapper_size;
inline constexpr std::uint32_t graphics_program_offset =
    code_wrapper_size + sph_size;  // 0x80
inline constexpr std::uint32_t graphics_min_c1_offset = 0x88u;
inline constexpr std::uint32_t compute_code_offset = 0u;
inline constexpr std::uint32_t compute_program_offset = 0u;
inline constexpr std::uint32_t compute_min_c1_offset = 8u;

// Version 15 (SDK 13 drivers) keeps version 14's fields: the C1 offset, code
// size, code offset and stage words and the extended metadata are where 14 has
// them. Version 16 (SDK 16 GLSLC) keeps them too: in every version-16 GLSLC
// output examined, the control size, code size, code offset, C1 offset and
// stage words agree with the output's own section headers.
[[nodiscard]] constexpr bool is_accepted_version(std::uint32_t version) {
    return version == 9u || version == 14u || version == 15u || version == 16u;
}

}  // namespace format

// Conservative resource ceilings. These bound parse cost and are explicit
// policy choices, not format facts: nothing in the container format itself
// establishes them.
namespace limits {

inline constexpr std::uint32_t control_max_size = 1024u * 1024u;
inline constexpr std::uint32_t code_max_size = 16u * 1024u * 1024u;

}  // namespace limits

// Stage number stored in the control word at 0x714. The names follow the
// stage ordering implied by the primary implementation's control-to-SPH
// mapping; the parser never interprets stage semantics beyond that mapping.
enum class Stage : std::uint32_t {
    Vertex = 0,
    Fragment = 1,
    Geometry = 2,
    TessellationControl = 3,
    TessellationEvaluation = 4,
    Compute = 5,
};

// SPH stage type nibble for a graphics control stage, taken from the primary
// implementation's mapping {1, 5, 4, 2, 3}. Returns 0 for Compute, which
// carries no SPH header at all.
[[nodiscard]] constexpr std::uint32_t sph_stage_type_for(Stage stage) {
    switch (stage) {
        case Stage::Vertex:
            return 1u;
        case Stage::Fragment:
            return 5u;
        case Stage::Geometry:
            return 4u;
        case Stage::TessellationControl:
            return 2u;
        case Stage::TessellationEvaluation:
            return 3u;
        case Stage::Compute:
            break;
    }
    return 0u;
}

struct ParseError {
    enum class Code : std::uint8_t {
        // Input contract violations.
        NullControlData,
        NullCodeData,
        NullExtendedMetadata,
        // Control prefix and structure validation.
        ControlSpanTooSmall,
        ControlMagic,
        ControlTag,
        ControlVersion,
        ControlSizeTooSmall,
        ControlSizeTooLarge,
        ControlSizeBeyondSpan,
        // Extended-descriptor metadata validation.
        ExtendedMetadataSize,
        ExtendedMetadataShortControl,
        ExtendedMetadataMismatch,
        // Code layout validation.
        StageOutOfRange,
        CodeSizeZero,
        CodeSizeTooLarge,
        CodeSizeMisaligned,
        C1OffsetBeyondCodeSize,
        CodeOffsetBeyondC1Offset,
        C1OffsetMisaligned,
        ComputeHeaderLayout,
        GraphicsHeaderLayout,
        CodeSpanTooSmall,
        WrapperMagic,
        SphStageMismatch,
    };

    Code code = Code::NullControlData;
    // Offending observed value and the bound it violated, where meaningful;
    // otherwise zero. Both are zero for contract violations that carry no
    // numeric detail.
    std::uint64_t value = 0;
    std::uint64_t bound = 0;

    [[nodiscard]] const char* message() const noexcept;
};

// One instruction word slot inside the program region.
struct InstructionSlot {
    // Byte offset of the instruction word relative to the start of the
    // program region. Slots start at 8 (after the first scheduling word), so
    // a zeroed slot is never a valid result.
    std::uint32_t pc = 0;
    // Byte offset of the same word within the whole code buffer.
    std::uint32_t code_offset = 0;
};

struct ParsedShader {
    std::uint32_t version = 0;  // Accepted control versions are 9 and 14.
    Stage stage = Stage::Vertex;
    std::uint32_t control_size = 0;  // Declared size from control[0x20].
    std::uint32_t code_size = 0;     // Declared size from control[0x704].
    // SPH offset from control[0x708]: 0x30 for graphics stages, 0 for
    // compute. For graphics this is the wrapper size.
    std::uint32_t code_offset = 0;
    // Program end / C1 start from control[0x700].
    std::uint32_t c1_offset = 0;
    std::uint32_t c1_size = 0;
    // Byte offset and size of the instruction stream inside the code buffer:
    // [0x80, c1_offset) for graphics, [0, c1_offset) for compute.
    std::uint32_t program_offset = 0;
    std::uint32_t program_size = 0;

    // Borrowed spans; see the lifetime contract at the top of this header.
    ByteSpan control;  // control_size bytes, opaque tail included.
    ByteSpan code;     // code_size bytes.
    ByteSpan wrapper;  // Graphics only: [0, 0x30); empty for compute.
    ByteSpan sph;      // Graphics only: 0x50 bytes at code_offset; else empty.
    ByteSpan program;  // Instruction stream: program_size bytes.
    ByteSpan c1;       // Trailing constant segment: c1_size bytes, may be empty.

    [[nodiscard]] bool is_compute() const noexcept {
        return stage == Stage::Compute;
    }

    // Number of 32-byte scheduling groups covering the program region,
    // including a possibly truncated trailing group.
    [[nodiscard]] std::uint32_t scheduling_group_count() const noexcept;

    // The opaque 8-byte scheduling word that starts group `group`. Returns an
    // empty span when `group` is out of range. The word's encoding carries
    // scheduling semantics that only the decoder may interpret.
    [[nodiscard]] ByteSpan scheduling_word(std::uint32_t group) const noexcept;

    // Total number of instruction word slots present in the program region.
    [[nodiscard]] std::uint32_t instruction_slot_count() const noexcept;

    // The `index`-th instruction word slot in program order (slots within a
    // group are ordered by ascending offset). Returns a zeroed slot when
    // `index` is out of range; zeroed slots are never valid.
    [[nodiscard]] InstructionSlot instruction_slot(
        std::uint32_t index) const noexcept;

    // Raw SPH stage type nibble for graphics stages; 0 when there is no SPH
    // (compute). Verified to match sph_stage_type_for(stage) during parsing.
    [[nodiscard]] std::uint32_t sph_stage_type() const noexcept;
};

using ParseResult = std::variant<ParsedShader, ParseError>;

struct ShaderInput {
    // Borrowed control bytes. Must cover at least the 0x24-byte prefix to be
    // examined, and at least control[0x20] bytes for the declared structure.
    ByteSpan control;
    // Borrowed code bytes. Must cover at least control[0x704] bytes.
    ByteSpan code;
    // Optional extended-descriptor metadata: when present (non-null data or
    // nonzero size) it must be exactly 24 bytes and byte-identical to
    // control[0x768..0x780). A differing override is rejected, never ignored.
    ByteSpan extended_metadata;
};

// Validates `input` and returns either the parsed structure or a structured
// error. Every slice is validated before it is exposed; out-of-range access
// on the result returns empty spans or zeroed slots instead of reading out of
// bounds. No allocation, no byte copies, no game-specific data.
[[nodiscard]] ParseResult parse_shader(const ShaderInput& input) noexcept;

}  // namespace armigo::shader_input
