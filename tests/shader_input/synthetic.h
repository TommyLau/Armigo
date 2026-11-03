// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic shader input fixtures. Every byte is a
// deterministic made-up pattern; no game-derived content is used anywhere.
#pragma once

#include <armigo/shader_input/shader_input.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace armigo::shader_input::test {

inline void put_le32(std::vector<std::uint8_t>& bytes, std::uint32_t offset,
                     std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        bytes[offset + i] =
            static_cast<std::uint8_t>(value >> (8u * i));
    }
}

inline std::uint32_t get_le32(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           static_cast<std::uint32_t>(bytes[1]) << 8 |
           static_cast<std::uint32_t>(bytes[2]) << 16 |
           static_cast<std::uint32_t>(bytes[3]) << 24;
}

// Fills bytes with a synthetic pattern derived from the absolute offset so
// distinct regions never compare equal by accident.
inline void fill_synthetic(std::vector<std::uint8_t>& bytes,
                           std::uint32_t from, std::uint32_t to) {
    for (std::uint32_t i = from; i < to; ++i) {
        bytes[i] = static_cast<std::uint8_t>((i * 31u + 7u) & 0xffu);
    }
}

struct ControlSpec {
    std::uint32_t magic = format::control_magic;
    std::uint32_t tag = format::control_tag;
    std::uint32_t version = 14;
    // Size written to control[0x20]; ~0u means "use buffer_size".
    std::uint32_t size_field = ~0u;
    std::uint32_t buffer_size = format::control_min_size;
    std::uint32_t stage = 0;
    std::uint32_t code_size = 0;
    std::uint32_t code_offset = format::graphics_code_offset;
    std::uint32_t c1_offset = format::graphics_min_c1_offset;
    // When non-empty, written at 0x768 regardless of buffer_size.
    std::vector<std::uint8_t> extended;
};

inline std::vector<std::uint8_t> build_control(const ControlSpec& spec) {
    const std::uint32_t size =
        spec.extended.empty()
            ? spec.buffer_size
            : (spec.buffer_size < format::extended_metadata_min_control_size
                   ? format::extended_metadata_min_control_size
                   : spec.buffer_size);
    std::vector<std::uint8_t> bytes(size, 0);
    fill_synthetic(bytes, 0, size);
    put_le32(bytes, 0, spec.magic);
    put_le32(bytes, 4, spec.tag);
    put_le32(bytes, format::control_version_offset, spec.version);
    put_le32(bytes, format::control_size_offset,
             spec.size_field == ~0u ? size : spec.size_field);
    put_le32(bytes, format::c1_offset_field, spec.c1_offset);
    put_le32(bytes, format::code_size_field, spec.code_size);
    put_le32(bytes, format::code_offset_field, spec.code_offset);
    put_le32(bytes, format::stage_field, spec.stage);
    if (!spec.extended.empty()) {
        std::memcpy(bytes.data() + format::extended_metadata_offset,
                    spec.extended.data(), format::extended_metadata_size);
    }
    return bytes;
}

// Builds the 24 synthetic extended metadata bytes that correspond to a
// control buffer built from `spec`.
inline std::vector<std::uint8_t> matching_extended(const ControlSpec& spec) {
    const std::vector<std::uint8_t> control = build_control(spec);
    return std::vector<std::uint8_t>(
        control.begin() + format::extended_metadata_offset,
        control.begin() + format::extended_metadata_offset +
            format::extended_metadata_size);
}

struct CodeSpec {
    std::uint32_t stage = 0;  // Control stage number 0..5.
    // Program region size in bytes; multiple of 8, at least 8.
    std::uint32_t program_bytes = 32;
    std::uint32_t c1_bytes = 16;
    // SPH stage type written into the header; ~0u means "correct for stage".
    std::uint32_t sph_type = ~0u;
    std::uint32_t wrapper_magic = format::code_wrapper_magic;
};

inline std::vector<std::uint8_t> build_code(const CodeSpec& spec) {
    const bool compute =
        spec.stage == static_cast<std::uint32_t>(Stage::Compute);
    const std::uint32_t program_offset =
        compute ? format::compute_program_offset
                : format::graphics_program_offset;
    const std::uint32_t total =
        program_offset + spec.program_bytes + spec.c1_bytes;
    std::vector<std::uint8_t> bytes(total, 0);
    fill_synthetic(bytes, 0, total);
    if (!compute) {
        put_le32(bytes, 0, spec.wrapper_magic);
        const std::uint32_t sph_type =
            spec.sph_type == ~0u
                ? sph_stage_type_for(static_cast<Stage>(spec.stage))
                : spec.sph_type;
        const std::uint32_t sph_word = get_le32(
            bytes.data() + format::code_wrapper_size);
        put_le32(bytes, format::code_wrapper_size,
                 (sph_word & ~(format::sph_stage_mask
                               << format::sph_stage_shift)) |
                     (sph_type << format::sph_stage_shift));
    }
    return bytes;
}

// Control spec values matching a code buffer built from `code_spec`.
inline ControlSpec control_for(const CodeSpec& code_spec) {
    ControlSpec spec;
    spec.stage = code_spec.stage;
    spec.version = 14;
    const bool compute =
        code_spec.stage == static_cast<std::uint32_t>(Stage::Compute);
    spec.code_offset =
        compute ? format::compute_code_offset : format::graphics_code_offset;
    const std::uint32_t program_offset =
        compute ? format::compute_program_offset
                : format::graphics_program_offset;
    spec.c1_offset = program_offset + code_spec.program_bytes;
    spec.code_size = spec.c1_offset + code_spec.c1_bytes;
    return spec;
}

inline ByteSpan span_of(const std::vector<std::uint8_t>& bytes) {
    return {bytes.data(), bytes.size()};
}

inline bool bytes_equal(ByteSpan span,
                        const std::vector<std::uint8_t>& expected) {
    return span.size == expected.size() &&
           (expected.empty() ||
            std::memcmp(span.data, expected.data(), expected.size()) == 0);
}

// Extracts `span` (expressed relative to `base`) as a vector for comparison.
inline std::vector<std::uint8_t> slice(ByteSpan base, std::uint32_t offset,
                                       std::uint32_t size) {
    return std::vector<std::uint8_t>(base.data + offset,
                                     base.data + offset + size);
}

}  // namespace armigo::shader_input::test
