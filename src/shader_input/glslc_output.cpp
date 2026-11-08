// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_input/glslc_output.h>

#include <algorithm>
#include <array>
#include <unordered_set>
#include <utility>
#include <variant>

namespace armigo::shader_input {
namespace {

namespace gf = glslc_format;

std::uint32_t read_le32(std::span<const std::byte> input, std::size_t offset) {
    return std::to_integer<std::uint32_t>(input[offset]) |
           (std::to_integer<std::uint32_t>(input[offset + 1]) << 8) |
           (std::to_integer<std::uint32_t>(input[offset + 2]) << 16) |
           (std::to_integer<std::uint32_t>(input[offset + 3]) << 24);
}

bool within(std::size_t position, std::size_t count, std::size_t end) {
    return count <= end && position <= end - count;
}

std::array<std::byte, 4> le_bytes(std::uint32_t value) {
    return {std::byte(value & 0xff), std::byte((value >> 8) & 0xff), std::byte((value >> 16) & 0xff),
            std::byte(value >> 24)};
}

// The stages of one output, or the reason it does not validate.
std::variant<std::vector<OwnedShaderStage>, std::string> parse_output(std::span<const std::byte> output,
                                                                      std::uint32_t sections,
                                                                      const std::string& where) {
    std::vector<OwnedShaderStage> stages;
    for (std::uint32_t index = 0; index < sections; ++index) {
        const std::size_t header = gf::first_section + gf::section_stride * index;
        const std::uint32_t size = read_le32(output, header + gf::section_size_field);
        const std::uint32_t data = read_le32(output, header + gf::section_data_field);
        if (read_le32(output, header + gf::section_type_field) != gf::gpu_code_section) continue;
        if (!within(data, size, output.size())) {
            return "section " + std::to_string(index) + " runs past the output";
        }
        const auto section = output.subspan(data, size);
        const std::uint32_t control_at = read_le32(output, header + gf::control_offset_field);
        const std::uint32_t code_at = read_le32(output, header + gf::code_offset_field);
        const std::uint32_t code_size = read_le32(output, header + gf::code_size_field);
        const std::uint32_t control_size = read_le32(output, header + gf::control_size_field);
        if (!within(control_at, control_size, section.size()) || !within(code_at, code_size, section.size())) {
            return "section " + std::to_string(index) + " control or code runs past its data";
        }
        const auto control = section.subspan(control_at, control_size);
        const auto code = section.subspan(code_at, code_size);
        const ShaderInput input{{reinterpret_cast<const std::uint8_t*>(control.data()), control.size()},
                                {reinterpret_cast<const std::uint8_t*>(code.data()), code.size()},
                                {}};
        const auto parsed = parse_shader(input);
        if (const auto* failed = std::get_if<ParseError>(&parsed)) {
            return "section " + std::to_string(index) + ": " + failed->message();
        }
        const auto& shader = std::get<ParsedShader>(parsed);
        if (shader.control_size != control.size() || shader.code_size != code.size()) {
            return "section " + std::to_string(index) + " lengths disagree with its control";
        }
        OwnedShaderStage stage;
        stage.source = where + "#" + std::to_string(index);
        stage.stage = shader.stage;
        stage.control.assign(control.begin(), control.end());
        stage.code.assign(code.begin(), code.end());
        stages.push_back(std::move(stage));
    }
    if (stages.empty()) return std::string("no GPU code section");
    return stages;
}

}  // namespace

GlslcScan scan_glslc_outputs(std::span<const std::byte> image, std::string_view label) {
    GlslcScan scan;
    std::unordered_set<std::string> covered;
    const auto magic = le_bytes(gf::magic);
    auto cursor = image.begin();
    while (cursor != image.end()) {
        const auto found = std::search(cursor, image.end(), magic.begin(), magic.end());
        if (found == image.end()) break;
        cursor = found + static_cast<std::ptrdiff_t>(magic.size());
        const auto offset = static_cast<std::size_t>(found - image.begin());
        if (!within(offset, gf::first_section, image.size())) continue;
        const std::uint32_t size = read_le32(image, offset + gf::size_field);
        const std::uint32_t sections = read_le32(image, offset + gf::section_count_field);
        // Not an output: the declared length leaves no room for one section
        // header or runs past the image.
        if (size < gf::first_section + gf::section_stride || !within(offset, size, image.size())) continue;
        const auto where = std::string(label) + "@" + std::to_string(offset);
        if (sections == 0 || sections > glslc_section_limit ||
            gf::first_section + gf::section_stride * sections > size) {
            scan.rejected.push_back(where + ": section table does not fit the output");
            continue;
        }
        auto parsed = parse_output(image.subspan(offset, size), sections, where);
        if (auto* reason = std::get_if<std::string>(&parsed)) {
            scan.rejected.push_back(where + ": " + *reason);
            continue;
        }
        ++scan.outputs;
        for (auto& stage : std::get<std::vector<OwnedShaderStage>>(parsed)) {
            covered.emplace(reinterpret_cast<const char*>(stage.control.data()), stage.control.size());
            scan.stages.push_back(std::move(stage));
        }
    }
    const auto control = le_bytes(format::control_magic);
    cursor = image.begin();
    while (cursor != image.end()) {
        const auto found = std::search(cursor, image.end(), control.begin(), control.end());
        if (found == image.end()) break;
        cursor = found + static_cast<std::ptrdiff_t>(control.size());
        const auto offset = static_cast<std::size_t>(found - image.begin());
        std::size_t declared = 0;
        if (within(offset, format::control_size_offset + 4, image.size())) {
            declared = read_le32(image, offset + format::control_size_offset);
        }
        const bool accounted = declared != 0 && within(offset, declared, image.size()) &&
            covered.contains(std::string(reinterpret_cast<const char*>(image.data() + offset), declared));
        if (!accounted) scan.uncovered_controls.push_back(offset);
    }
    return scan;
}

}  // namespace armigo::shader_input
