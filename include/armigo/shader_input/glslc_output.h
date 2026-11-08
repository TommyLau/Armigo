// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Reader for GLSLC compiler outputs compiled into a module image. An output
// starts with a four-byte magic, declares its whole length and a table of
// section headers; each GPU code section locates one stage's control and code
// inside the section's data.
#pragma once

#include <armigo/shader_input/gfd_cache.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::shader_input {

namespace glslc_format {

inline constexpr std::uint32_t magic = 0x19866891u;
inline constexpr std::size_t size_field = 0x44;
inline constexpr std::size_t section_count_field = 0x4c;
inline constexpr std::size_t first_section = 0x90;
inline constexpr std::size_t section_stride = 0x90;
// Section header: size, data offset (from the output) and type.
inline constexpr std::size_t section_size_field = 0x0;
inline constexpr std::size_t section_data_field = 0x4;
inline constexpr std::size_t section_type_field = 0x8;
inline constexpr std::uint32_t gpu_code_section = 0;
// GPU code section header, after the common fields: control and code offsets
// from the section's data, then the code and control lengths.
inline constexpr std::size_t control_offset_field = 0x30;
inline constexpr std::size_t code_offset_field = 0x34;
inline constexpr std::size_t code_size_field = 0x38;
inline constexpr std::size_t control_size_field = 0x3c;

}  // namespace glslc_format

// Resource policy, not a format limit.
inline constexpr std::uint32_t glslc_section_limit = 16;

struct GlslcScan {
    // Every GPU code section of every output, in image order.
    std::vector<OwnedShaderStage> stages;
    // Number of outputs whose header and sections validated.
    std::size_t outputs = 0;
    // Outputs whose header is plausible but whose sections do not validate.
    std::vector<std::string> rejected;
    // Offsets of shader-control magics that no extracted stage accounts for.
    std::vector<std::uint64_t> uncovered_controls;
};

// Scans a module image for GLSLC outputs. An occurrence of the magic whose
// declared length does not fit the image or cannot hold a section table is not
// an output; its bytes are still checked by the control coverage pass, so a
// stage that sits in an output the scan did not recognize is reported there.
[[nodiscard]] GlslcScan scan_glslc_outputs(std::span<const std::byte> image, std::string_view label);

}  // namespace armigo::shader_input
