// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic GLSLC compiler outputs inside a module image. Every byte is
// constructed here; nothing is taken from a game.
#include "synthetic.h"
#include "test_support.h"

#include <armigo/shader_input/glslc_output.h>

#include <cstring>
#include <vector>

namespace {
namespace si = armigo::shader_input;
namespace gf = armigo::shader_input::glslc_format;
using armigo::shader_input::test::build_code;
using armigo::shader_input::test::build_control;
using armigo::shader_input::test::CodeSpec;
using armigo::shader_input::test::control_for;

struct Section {
    std::uint32_t type = gf::gpu_code_section;
    std::vector<std::uint8_t> control;
    std::vector<std::uint8_t> code;
};

void put(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[offset + index] = static_cast<std::byte>(value >> (8u * index));
    }
}

Section stage(std::uint32_t number) {
    CodeSpec spec;
    spec.stage = number;
    return {gf::gpu_code_section, build_control(control_for(spec)), build_code(spec)};
}

// One output: header, section table, then each section's data (control
// followed by code for a GPU code section).
std::vector<std::byte> output(const std::vector<Section>& sections) {
    std::size_t size = gf::first_section + gf::section_stride * sections.size();
    for (const auto& section : sections) size += section.control.size() + section.code.size() + 16;
    std::vector<std::byte> out(size, std::byte{0});
    put(out, 0, gf::magic);
    put(out, gf::size_field, static_cast<std::uint32_t>(size));
    put(out, gf::section_count_field, static_cast<std::uint32_t>(sections.size()));
    std::size_t data = gf::first_section + gf::section_stride * sections.size();
    for (std::size_t index = 0; index < sections.size(); ++index) {
        const auto& section = sections[index];
        const std::size_t header = gf::first_section + gf::section_stride * index;
        const auto length = section.control.size() + section.code.size() + 16;
        put(out, header + gf::section_size_field, static_cast<std::uint32_t>(length));
        put(out, header + gf::section_data_field, static_cast<std::uint32_t>(data));
        put(out, header + gf::section_type_field, section.type);
        // Section data starts with 16 bytes the GPU code header skips.
        put(out, header + gf::control_offset_field, 16);
        put(out, header + gf::code_offset_field, static_cast<std::uint32_t>(16 + section.control.size()));
        put(out, header + gf::code_size_field, static_cast<std::uint32_t>(section.code.size()));
        put(out, header + gf::control_size_field, static_cast<std::uint32_t>(section.control.size()));
        std::memcpy(out.data() + data + 16, section.control.data(), section.control.size());
        std::memcpy(out.data() + data + 16 + section.control.size(), section.code.data(), section.code.size());
        data += length;
    }
    return out;
}

void append(std::vector<std::byte>& image, const std::vector<std::byte>& bytes) {
    image.insert(image.end(), bytes.begin(), bytes.end());
    image.resize((image.size() + 15) & ~std::size_t{15}, std::byte{0});
}

}  // namespace

ARMIGO_SI_TEST(glslc, single_stage_outputs) {
    Section reflection;
    reflection.type = 3;
    reflection.control.assign(32, 0x5a);
    std::vector<std::byte> image(64, std::byte{0});
    append(image, output({stage(0), reflection}));
    const auto second = image.size();
    append(image, output({stage(1)}));
    const auto scan = si::scan_glslc_outputs(image, "main");
    ARMIGO_SI_CHECK(scan.rejected.empty());
    ARMIGO_SI_CHECK(scan.uncovered_controls.empty());
    ARMIGO_SI_CHECK_EQ(scan.outputs, std::size_t{2});
    ARMIGO_SI_CHECK_EQ(scan.stages.size(), std::size_t{2});
    if (scan.stages.size() == 2) {
        ARMIGO_SI_CHECK(scan.stages[0].stage == si::Stage::Vertex);
        ARMIGO_SI_CHECK(scan.stages[1].stage == si::Stage::Fragment);
        ARMIGO_SI_CHECK(scan.stages[0].source == "main@64#0");
        ARMIGO_SI_CHECK(scan.stages[1].source == "main@" + std::to_string(second) + "#0");
        const auto expected = stage(1);
        ARMIGO_SI_CHECK(scan.stages[1].control.size() == expected.control.size() &&
                        std::memcmp(scan.stages[1].control.data(), expected.control.data(),
                                    expected.control.size()) == 0);
        ARMIGO_SI_CHECK(scan.stages[1].code.size() == expected.code.size() &&
                        std::memcmp(scan.stages[1].code.data(), expected.code.data(), expected.code.size()) == 0);
    }
}

ARMIGO_SI_TEST(glslc, multi_stage_output) {
    std::vector<std::byte> image;
    append(image, output({stage(0), stage(1)}));
    const auto scan = si::scan_glslc_outputs(image, "main");
    ARMIGO_SI_CHECK(scan.rejected.empty() && scan.uncovered_controls.empty());
    ARMIGO_SI_CHECK_EQ(scan.outputs, std::size_t{1});
    ARMIGO_SI_CHECK_EQ(scan.stages.size(), std::size_t{2});
    if (scan.stages.size() == 2) ARMIGO_SI_CHECK(scan.stages[1].source == "main@0#1");
}

ARMIGO_SI_TEST(glslc, implausible_magic_is_not_an_output) {
    std::vector<std::byte> image(0x200, std::byte{0});
    put(image, 0x10, gf::magic);  // declared length 0
    put(image, 0x100, gf::magic);
    put(image, 0x100 + gf::size_field, 0x10000);  // past the image
    const auto scan = si::scan_glslc_outputs(image, "main");
    ARMIGO_SI_CHECK(scan.rejected.empty() && scan.stages.empty() && scan.uncovered_controls.empty());
    ARMIGO_SI_CHECK_EQ(scan.outputs, std::size_t{0});
}

ARMIGO_SI_TEST(glslc, malformed_outputs_are_rejected) {
    std::vector<std::byte> image;
    auto no_sections = output({stage(0)});
    put(no_sections, gf::section_count_field, 0);
    append(image, no_sections);
    auto past = output({stage(0)});
    put(past, gf::first_section + gf::code_size_field, 0x100000);
    append(image, past);
    auto mismatched = output({stage(0)});
    // The code length no longer matches control[0x704].
    put(mismatched, gf::first_section + gf::code_size_field,
        static_cast<std::uint32_t>(stage(0).code.size() - 8));
    append(image, mismatched);
    Section reflection;
    reflection.type = 3;
    append(image, output({reflection}));
    const auto scan = si::scan_glslc_outputs(image, "main");
    ARMIGO_SI_CHECK_EQ(scan.rejected.size(), std::size_t{4});
    ARMIGO_SI_CHECK_EQ(scan.outputs, std::size_t{0});
    ARMIGO_SI_CHECK(scan.stages.empty());
    // The three controls inside the rejected outputs are not accounted for.
    ARMIGO_SI_CHECK_EQ(scan.uncovered_controls.size(), std::size_t{3});
}

ARMIGO_SI_TEST(glslc, loose_control_is_uncovered) {
    std::vector<std::byte> image;
    append(image, output({stage(0)}));
    const auto loose = image.size();
    const auto extra = stage(1);
    for (const auto value : extra.control) image.push_back(static_cast<std::byte>(value));
    const auto scan = si::scan_glslc_outputs(image, "main");
    ARMIGO_SI_CHECK(scan.rejected.empty());
    ARMIGO_SI_CHECK_EQ(scan.stages.size(), std::size_t{1});
    ARMIGO_SI_CHECK_EQ(scan.uncovered_controls.size(), std::size_t{1});
    ARMIGO_SI_CHECK(!scan.uncovered_controls.empty() && scan.uncovered_controls.front() == loose);
}
