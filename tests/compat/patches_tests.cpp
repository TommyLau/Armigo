// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/compat/patches.h>
#include <armigo/aarch64/instruction.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)
const char* json = R"({"version":1,"patches":[{
"title_id":"0x0000000000000123",
"main_build_id":"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
"module":"frame-rate","original_callee":"0x80",
"calls":[{"offset":"0x20","expected_instruction":"0x94000018"}],
"layout":{"framework_pointer":"0x200","framework_period":"0x10",
"override_flag":"0x0","ratio":"0x4","interval":"0x8","base_interval":"0xc"}}]})";
armigo::compat::PatchReadResult parse(const std::string& contents) {
    const auto pattern = (std::filesystem::temp_directory_path() / "armigo-compat-test-XXXXXX").string();
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    const int fd = mkstemp(path.data());
    if (fd < 0) throw std::runtime_error("Cannot create synthetic patch fixture.");
    close(fd);
    { std::ofstream out(path.data()); out << contents; }
    auto result = armigo::compat::read_patch_descriptions(path.data());
    unlink(path.data());
    return result;
}
}
int main() {
    using namespace armigo;
    auto read = parse(json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(read));
    if (!std::holds_alternative<compat::PatchDescriptions>(read)) return 1;
    const auto patches = std::get<compat::PatchDescriptions>(read);
    const auto& patch = patches.front();
    CHECK(compat::match_patch(patches, patch.identity) == &patch);
    auto other = patch.identity;
    ++other.title_id;
    CHECK(compat::match_patch(patches, other) == nullptr);
    other = patch.identity; other.main_build_id.back() ^= std::byte{1};
    CHECK(compat::match_patch(patches, other) == nullptr);
    CHECK(compat::match_patch({}, patch.identity) == nullptr);
    loader::NsoModuleImage image;
    image.bytes.resize(0x400);
    image.segments[0].image_offset = 0;
    image.segments[0].content_size = 0x100;
    image.segments[2].image_offset = 0x200;
    image.segments[2].mapped_size = 0x200;
    std::memcpy(image.bytes.data() + 0x20, &patch.calls[0].expected_instruction, 4);
    const auto before = image.bytes;
    CHECK(compat::validate_patch(patch, image).empty());
    image.bytes[0x20] ^= std::byte{1};
    CHECK(compat::validate_patch(patch, image).find("preimage mismatch") != std::string::npos);
    image.bytes = before;
    auto invalid = patch;
    invalid.original_callee += 4;
    CHECK(!compat::validate_patch(invalid, image).empty());
    invalid = patch; invalid.calls[0].offset = 0x200;
    CHECK(!compat::validate_patch(invalid, image).empty());
    invalid = patch; invalid.calls.push_back(invalid.calls.front());
    CHECK(!compat::validate_patch(invalid, image).empty());
    invalid = patch; invalid.layout.framework_pointer = 0x3fc;
    CHECK(!compat::validate_patch(invalid, image).empty());
    // An optional frame step must be an aligned float inside module data.
    CHECK(patch.layout.frame_step == 0);
    auto stepped = patch; stepped.layout.frame_step = 0x3fc;
    CHECK(compat::validate_patch(stepped, image).empty());
    for (std::uint64_t offset : {std::uint64_t{0x3fe}, std::uint64_t{0x400}, std::uint64_t{0x1fc},
                                 std::uint64_t{0x200}, std::uint64_t{0x204}}) {
        stepped.layout.frame_step = offset;
        CHECK(compat::validate_patch(stepped, image).find("frame step") != std::string::npos);
    }
    stepped.layout.frame_step = 0x208;
    CHECK(compat::validate_patch(stepped, image).empty());
    CHECK(image.bytes == before);
    CHECK(std::holds_alternative<std::string>(parse("{}")));
    CHECK(std::holds_alternative<std::string>(parse("not JSON")));
    auto replace = [](std::string value, const char* from, const char* to) {
        value.replace(value.find(from), std::strlen(from), to); return value;
    };
    CHECK(std::holds_alternative<std::string>(parse(replace(json, "frame-rate", "unknown"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(json, "0x94000018", "0x100000000"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(json, "0x0000000000000123", "0x0000000000000000"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(json, "framework_period", "unknown_field"))));
    {
        const auto step_json = replace(json, R"("base_interval":"0xc"})", R"("base_interval":"0xc","frame_step":"0x300"})");
        const auto stepped_read = parse(step_json);
        CHECK(std::holds_alternative<compat::PatchDescriptions>(stepped_read));
        if (const auto* parsed = std::get_if<compat::PatchDescriptions>(&stepped_read))
            CHECK(parsed->front().layout.frame_step == 0x300);
        CHECK(std::holds_alternative<std::string>(parse(replace(json, R"("base_interval":"0xc"})",
                                                                R"("base_interval":"0xc","frame_step":"0x0"})"))));
    }
    // Instruction words: parsed with or without a module, validated against
    // executable text and their preimage, never overlapping a call hook.
    const auto words_json = replace(json, R"("override_flag":"0x0","ratio":"0x4","interval":"0x8","base_interval":"0xc"}})",
        R"("override_flag":"0x0","ratio":"0x4","interval":"0x8","base_interval":"0xc"},"words":[{"offset":"0x40","expected_instruction":"0x32170108","replacement_instruction":"0xd503201f"}]})");
    auto with_words = parse(words_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(with_words));
    if (std::holds_alternative<compat::PatchDescriptions>(with_words)) {
        const auto worded = std::get<compat::PatchDescriptions>(with_words).front();
        CHECK(worded.words.size() == 1 && worded.words[0].offset == 0x40 &&
              worded.words[0].expected_instruction == 0x32170108u &&
              worded.words[0].replacement_instruction == 0xd503201fu);
        const std::uint32_t orr = 0x32170108u;
        std::memcpy(image.bytes.data() + 0x40, &orr, 4);
        CHECK(compat::validate_patch(worded, image).empty());
        image.bytes[0x40] ^= std::byte{1};
        CHECK(compat::validate_patch(worded, image).find("word preimage mismatch") != std::string::npos);
        image.bytes[0x40] ^= std::byte{1};
        auto outside = worded; outside.words[0].offset = 0x200;
        CHECK(!compat::validate_patch(outside, image).empty());
        auto overlap = worded; overlap.words[0].offset = 0x20; overlap.words[0].expected_instruction = patch.calls[0].expected_instruction;
        CHECK(compat::validate_patch(overlap, image).find("overlaps a call hook") != std::string::npos);
        auto words_only = worded; words_only.module.clear(); words_only.calls.clear();
        CHECK(compat::validate_patch(words_only, image).empty());
    }
    CHECK(std::holds_alternative<std::string>(parse(replace(words_json, "replacement_instruction", "replacement"))));
    auto driver_only = parse(replace(words_json, R"("replacement_instruction":"0xd503201f")",
                                     R"("replacement_instruction":"0xd503201f","nvn_driver_only":true)"));
    CHECK(std::holds_alternative<compat::PatchDescriptions>(driver_only) &&
          std::get<compat::PatchDescriptions>(driver_only).front().words[0].nvn_driver_only);
    CHECK(std::holds_alternative<std::string>(parse(replace(words_json, R"("replacement_instruction":"0xd503201f")",
                                                            R"("replacement_instruction":"0xd503201f","nvn_driver_only":1)"))));
    // Resolution sites: the swapchain size takes one pair or a list of pairs,
    // one per copy of its initializer.
    const std::string resolution = R"(,"resolution":{"tv_size":TV,"screen_widths":["0x60","0x64"],)"
        R"("display_size":["0x68","0x6c"],"window_crop":["0x70","0x74"],"floor":["0x78","0x7c"],)"
        R"("render_target_pool":"0x80","graphics_heap":"0x84","process_heap":"0x88","root_heap":"0x8c",)"
        R"("load_scale":"0x90","swapchain_textures":3,"pool_size":[1600,900],"minimum_render_width":1024}})";
    const auto with_resolution = [&](const std::string& tv) {
        auto sites = resolution;
        sites.replace(sites.find("TV"), 2, tv);
        const auto record = R"("replacement_instruction":"0xd503201f"}])" + sites;
        return parse(replace(words_json, R"("replacement_instruction":"0xd503201f"}]})", record.c_str()));
    };
    using Pairs = std::vector<std::array<std::uint64_t, 2>>;
    const Pairs one{{0x50, 0x54}}, two{{0x50, 0x54}, {0x58, 0x5c}};
    auto single = with_resolution(R"(["0x50","0x54"])");
    CHECK(std::holds_alternative<compat::PatchDescriptions>(single) &&
          std::get<compat::PatchDescriptions>(single).front().resolution->tv_size == one);
    auto copies = with_resolution(R"([["0x50","0x54"],["0x58","0x5c"]])");
    CHECK(std::holds_alternative<compat::PatchDescriptions>(copies) &&
          std::get<compat::PatchDescriptions>(copies).front().resolution->tv_size == two);
    CHECK(std::holds_alternative<std::string>(with_resolution("[]")));
    CHECK(std::holds_alternative<std::string>(with_resolution(R"([["0x50"]])")));
    std::puts(failures ? "Compatibility patch tests failed." : "Compatibility patch tests passed.");
    return failures ? 1 : 0;
}
