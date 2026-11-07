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
"calls":[{"offset":"0x20","expected_instruction":"0x94000018","note":"Synthetic interval-update hook."}],
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
    CHECK(patch.shader_inputs.containers == compat::ShaderContainerSet::Bnsh);
    CHECK(patch.shader_inputs.vertex_depth == compat::ShaderVertexDepth::NegativeOneToOne);
    CHECK(patch.shader_inputs.caches.empty());
    CHECK(compat::match_patch(patches, patch.identity) == &patch);
    auto other = patch.identity;
    ++other.title_id;
    CHECK(compat::match_patch(patches, other) == nullptr);
    other = patch.identity; other.main_build_id.back() ^= std::byte{1};
    CHECK(compat::match_patch(patches, other) == nullptr);
    CHECK(compat::match_patch({}, patch.identity) == nullptr);
    CHECK(patch.note.empty());
    CHECK(patch.calls[0].note == "Synthetic interval-update hook.");
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
    const auto counter_json = replace(json,
        R"("layout":{"framework_pointer":"0x200","framework_period":"0x10",)"
        "\n" R"("override_flag":"0x0","ratio":"0x4","interval":"0x8","base_interval":"0xc"})",
        R"("layout":{"variant":"fractional-counter-30","framework_pointer":"0x200","framework_period":"0x10",)"
        R"("object_register":22,"state":"0x0","active_state":2,"counter":"0x48"})");
    {
        auto phase_json = replace(json, R"("base_interval":"0xc"})",
            R"("base_interval":"0xc","frame_step":"0x208"})");
        phase_json = replace(phase_json, R"("calls":[)",
            R"("frame_rate":{"default":60,"display_refresh_rate":60},"frame_rates":[60,120],"calls":[)");
        phase_json = replace(phase_json, R"("note":"Synthetic interval-update hook."}])",
            R"("note":"Synthetic interval-update hook."},{"offset":"0x24","expected_instruction":"0x9400001b","frame_rate":120,"note":"Synthetic phase scheduler.","phases":{"original_callee":"0x90","counter":"0x0","phase_count":10}}])");
        auto read_phases = parse(phase_json);
        CHECK(std::holds_alternative<compat::PatchDescriptions>(read_phases));
        if (const auto* parsed = std::get_if<compat::PatchDescriptions>(&read_phases)) {
            const auto& phased = parsed->front();
            auto phase_image = image;
            const std::uint32_t branch = 0x9400001b;
            std::memcpy(phase_image.bytes.data() + 0x24, &branch, 4);
            CHECK(compat::validate_patch(phased, phase_image).empty());
            CHECK(compat::call_callee(phased, phased.calls[0]) == 0x80);
            CHECK(compat::call_callee(phased, phased.calls[1]) == 0x90);
            for (const auto rate : {"", "0", "60"}) {
                auto selected = compat::select_patch_calls(phased, rate);
                CHECK(std::get<std::vector<compat::CallPatch>>(selected).size() == 1);
                CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(phased, rate)).empty());
            }
            CHECK(std::get<std::vector<compat::CallPatch>>(compat::select_patch_calls(phased, "120")).size() == 2);
            auto invalid_phase = phased;
            invalid_phase.calls[1].phases->original_callee = 0x80;
            CHECK(!compat::validate_patch(invalid_phase, phase_image).empty());
            invalid_phase = phased; invalid_phase.layout.frame_step = 0;
            CHECK(!compat::validate_patch(invalid_phase, phase_image).empty());
        }
        for (const auto count : {"0", "65536", "true", "1.5", "-1"})
            CHECK(std::holds_alternative<std::string>(parse(replace(phase_json, R"("phase_count":10)",
                (std::string("\"phase_count\":") + count).c_str()))));
        CHECK(std::holds_alternative<std::string>(parse(replace(phase_json, R"("counter":"0x0")", R"("counter":"0x1")"))));
        CHECK(std::holds_alternative<std::string>(parse(replace(phase_json, R"(,"frame_step":"0x208")", ""))));
        CHECK(std::holds_alternative<std::string>(parse(replace(phase_json, R"("phase_count":10)", R"("unknown":10)"))));
    }
    const auto counter_read = parse(counter_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(counter_read));
    if (const auto* descriptions = std::get_if<compat::PatchDescriptions>(&counter_read)) {
        CHECK(descriptions->front().layout.variant == titles::FrameRateVariant::FractionalCounter30);
        CHECK(descriptions->front().layout.object_register == 22);
        CHECK(compat::validate_patch(descriptions->front(), image).empty());
    }
    const auto countdown_read = parse(replace(counter_json, "fractional-counter-30", "fractional-countdown-30"));
    CHECK(std::holds_alternative<compat::PatchDescriptions>(countdown_read));
    if (const auto* descriptions = std::get_if<compat::PatchDescriptions>(&countdown_read)) {
        CHECK(descriptions->front().layout.variant == titles::FrameRateVariant::FractionalCountdown30);
        CHECK(compat::validate_patch(descriptions->front(), image).empty());
    }
    const auto repeat_json = replace(replace(counter_json, "fractional-counter-30", "fractional-repeat-30"),
        R"("counter":"0x48")", R"("counter":"0x48","repeat_flag":"0x4c","repeat_frames":5)");
    const auto repeat_read = parse(repeat_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(repeat_read));
    if (const auto* descriptions = std::get_if<compat::PatchDescriptions>(&repeat_read)) {
        CHECK(descriptions->front().layout.variant == titles::FrameRateVariant::FractionalRepeat30);
        CHECK(descriptions->front().layout.repeat_flag == 0x4c && descriptions->front().layout.repeat_frames == 5);
        CHECK(compat::validate_patch(descriptions->front(), image).empty());
    }
    CHECK(std::holds_alternative<std::string>(parse(replace(repeat_json, R"("repeat_frames":5)", R"("repeat_frames":0)"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(repeat_json, R"("repeat_flag":"0x4c")", R"("repeat_flag":"0x49")"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(repeat_json, R"(,"repeat_flag":"0x4c")", ""))));
    CHECK(std::holds_alternative<std::string>(parse(replace(repeat_json, "fractional-repeat-30", "fractional-countdown-30"))));
    for (const auto* value : {"31", "-1", "true", "22.5"})
        CHECK(std::holds_alternative<std::string>(parse(replace(counter_json,
            R"("object_register":22)", (std::string("\"object_register\":") + value).c_str()))));
    CHECK(std::holds_alternative<std::string>(parse(replace(counter_json, "fractional-counter-30", "unknown"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(counter_json,
        R"("counter":"0x48")", R"("counter":"0x0")"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(counter_json,
        R"("counter":"0x48")", R"("counter":"0x49")"))));
    // A call note is required, a string of 1 to 200 characters. The description
    // note is the same string when present and may be omitted.
    CHECK(std::holds_alternative<std::string>(parse(replace(json, R"(,"note":"Synthetic interval-update hook.")", ""))));
    CHECK(std::holds_alternative<std::string>(parse(replace(json, "Synthetic interval-update hook.", ""))));
    CHECK(std::holds_alternative<std::string>(parse(replace(json, R"("note":"Synthetic interval-update hook.")", R"("note":7)"))));
    {
        const std::string longest(200, 'n');
        const std::string over(201, 'n');
        const auto accepted = parse(replace(json, "Synthetic interval-update hook.", longest.c_str()));
        CHECK(std::holds_alternative<compat::PatchDescriptions>(accepted));
        if (const auto* parsed = std::get_if<compat::PatchDescriptions>(&accepted))
            CHECK(parsed->front().calls[0].note.size() == 200);
        CHECK(std::holds_alternative<std::string>(parse(replace(json, "Synthetic interval-update hook.", over.c_str()))));
        const auto described = parse(replace(json, R"("module":"frame-rate")",
                                             R"("module":"frame-rate","note":"Synthetic description.")"));
        CHECK(std::holds_alternative<compat::PatchDescriptions>(described));
        if (const auto* parsed = std::get_if<compat::PatchDescriptions>(&described))
            CHECK(parsed->front().note == "Synthetic description.");
        CHECK(std::holds_alternative<std::string>(parse(replace(json, R"("module":"frame-rate")",
                                                                R"("module":"frame-rate","note":"")"))));
        CHECK(std::holds_alternative<std::string>(parse(replace(json, R"("module":"frame-rate")",
                                                                (R"("module":"frame-rate","note":")" + over + "\"").c_str()))));
    }
    // Instruction words: parsed with or without a module, validated against
    // executable text and their preimage, never overlapping a call hook.
    const auto words_json = replace(json, R"("override_flag":"0x0","ratio":"0x4","interval":"0x8","base_interval":"0xc"}})",
        R"("override_flag":"0x0","ratio":"0x4","interval":"0x8","base_interval":"0xc"},"words":[{"offset":"0x40","note":"Synthetic word.","expected_instruction":"0x32170108","replacement_instruction":"0xd503201f"}]})");
    auto with_words = parse(words_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(with_words));
    if (std::holds_alternative<compat::PatchDescriptions>(with_words)) {
        const auto worded = std::get<compat::PatchDescriptions>(with_words).front();
        CHECK(worded.words.size() == 1 && worded.words[0].offset == 0x40 &&
              worded.words[0].expected_instruction == 0x32170108u &&
              worded.words[0].replacement_instruction == 0xd503201fu &&
              worded.words[0].note == "Synthetic word.");
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
    CHECK(std::holds_alternative<std::string>(parse(replace(words_json, R"(,"note":"Synthetic word.")", ""))));
    CHECK(std::holds_alternative<std::string>(parse(replace(words_json, R"("note":"Synthetic word.")", R"("note":"")"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(words_json, "Synthetic word.", std::string(201, 'w').c_str()))));
    auto driver_only = parse(replace(words_json, R"("replacement_instruction":"0xd503201f")",
                                     R"("replacement_instruction":"0xd503201f","nvn_driver_only":true)"));
    CHECK(std::holds_alternative<compat::PatchDescriptions>(driver_only) &&
          std::get<compat::PatchDescriptions>(driver_only).front().words[0].nvn_driver_only);
    CHECK(std::holds_alternative<std::string>(parse(replace(words_json, R"("replacement_instruction":"0xd503201f")",
                                                            R"("replacement_instruction":"0xd503201f","nvn_driver_only":1)"))));
    const auto rate_json = replace(words_json, R"("words":[)",
        R"("frame_rate":{"default":30,"display_refresh_rate":60},"frame_rates":[30,60],"words":[)");
    const auto conditional_json = replace(rate_json, R"("note":"Synthetic word.")",
        R"("note":"Synthetic word.","frame_rate":60)");
    const auto conditional = parse(conditional_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(conditional));
    if (const auto* descriptions = std::get_if<compat::PatchDescriptions>(&conditional)) {
        auto chosen = descriptions->front();
        CHECK(chosen.frame_rate->default_rate == 30 && chosen.frame_rate->display_refresh_rate == 60);
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(chosen)).empty());
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(chosen, "0")).empty());
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(chosen, "30")).empty());
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(chosen, "60")).size() == 1);
        for (const auto* invalid_rate : {"auto", "30.0", "-1", " 60", "60x"})
            CHECK(std::holds_alternative<std::string>(compat::select_patch_words(chosen, invalid_rate)));
        CHECK(std::holds_alternative<std::string>(compat::select_patch_words(chosen, "120")));
        auto common = chosen.words.front(); common.offset += 4; common.frame_rate = 0;
        chosen.words.push_back(common);
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(chosen, "30")).size() == 1);
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(chosen, "60")).size() == 2);
        chosen.frame_rate->default_rate = 60;
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(chosen, "0")).size() == 2);
        // Inactive words still require the original executable preimage.
        chosen.frame_rate->default_rate = 30;
        chosen.words.resize(1);
        auto wrong_image = image;
        wrong_image.bytes[chosen.words.front().offset] ^= std::byte{1};
        CHECK(compat::validate_patch(chosen, wrong_image).find("preimage mismatch") != std::string::npos);
    }
    CHECK(std::holds_alternative<std::string>(parse(replace(words_json, R"("note":"Synthetic word.")",
        R"("note":"Synthetic word.","frame_rate":60)"))));
    for (const auto* invalid_rate : {"0", "120", "true", "30.5", "\"30\""}) {
        CHECK(std::holds_alternative<std::string>(parse(replace(conditional_json, R"("default":30)",
            (std::string("\"default\":") + invalid_rate).c_str()))));
    }
    for (const auto* invalid_rate : {"0", "90", "true", "30.5", "\"30\""}) {
        CHECK(std::holds_alternative<std::string>(parse(replace(conditional_json, R"("frame_rate":60)",
            (std::string("\"frame_rate\":") + invalid_rate).c_str()))));
    }
    // Private modes can share an instruction site only when their conditions
    // are mutually exclusive and they identify the same original instruction.
    const std::string private_word = R"({"offset":"0x40","note":"Synthetic private mode.",)"
        R"("expected_instruction":"0x32170108","replacement_instruction":"0xd503203f","frame_rate":120},)";
    const auto variants_json = replace(replace(conditional_json, R"("frame_rates":[30,60])",
        R"("frame_rates":[30,60,120])"), R"("words":[)", (R"("words":[)" + private_word).c_str());
    const auto experimental_json = replace(variants_json, R"("display_refresh_rate":60)",
        R"("display_refresh_rate":60,"experimental_rates":[120])");
    const auto experimental = parse(experimental_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(experimental));
    if (const auto* parsed = std::get_if<compat::PatchDescriptions>(&experimental))
        CHECK(parsed->front().frame_rate->experimental_rates == std::vector<std::uint32_t>({120}));
    CHECK(std::holds_alternative<std::string>(parse(replace(experimental_json,
        R"("experimental_rates":[120])", R"("experimental_rates":[120,120])"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(experimental_json,
        R"("frame_rates":[30,60,120])", R"("frame_rates":[30,60])"))));
    const auto variants = parse(variants_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(variants));
    if (const auto* descriptions = std::get_if<compat::PatchDescriptions>(&variants)) {
        auto modes = descriptions->front();
        CHECK(compat::validate_patch(modes, image).empty());
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(modes, "30")).empty());
        const auto sixty = std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(modes, "60"));
        const auto high = std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(modes, "120"));
        CHECK(sixty.size() == 1 && sixty.front().replacement_instruction == 0xd503201fu);
        CHECK(high.size() == 1 && high.front().replacement_instruction == 0xd503203fu);
        auto high_only = modes;
        high_only.words.resize(1);
        high_only.frame_rate->supported_rates = {30, 120};
        CHECK(std::holds_alternative<std::string>(compat::select_patch_words(high_only, "60")));
        high_only.frame_rate->supported_rates = {30, 60, 120};
        CHECK(std::holds_alternative<std::string>(compat::select_patch_words(high_only, "60")));
        high_only.words.front().nvn_driver_only = true;
        std::uint32_t admitted = 0;
        CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(
            compat::select_patch_words(high_only, "120", true, &admitted)));
        CHECK(admitted == 120);
        CHECK(std::holds_alternative<std::string>(compat::select_patch_words(high_only, "120", false, &admitted)));
        CHECK(admitted == 0);
        CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(high_only, "30", false)).empty());
        modes.words.front().frame_rate = 60;
        CHECK(!compat::validate_patch(modes, image).empty());
        CHECK(std::holds_alternative<std::string>(compat::select_patch_words(modes, "60")));
    }
    CHECK(std::holds_alternative<std::string>(parse(replace(variants_json, R"("frame_rate":120)", R"("frame_rate":60)"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(variants_json, R"(,"frame_rate":120)", ""))));
    CHECK(std::holds_alternative<std::string>(parse(replace(variants_json,
        R"("expected_instruction":"0x32170108")", R"("expected_instruction":"0xd503201f")"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(rate_json, "display_refresh_rate", "refresh"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(rate_json, R"(,"frame_rates":[30,60])", ""))));
    for (const char* supported : {"[]", "[60]", "[30,30]", "[30,90]", "[30,true]"})
        CHECK(std::holds_alternative<std::string>(parse(replace(rate_json, "[30,60]", supported))));
    CHECK(std::holds_alternative<std::string>(parse(replace(variants_json, "[30,60,120]", "[30,60]"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(rate_json,
        R"("display_refresh_rate":60)", R"("display_refresh_rate":120)"))));
    CHECK(std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(patch, "unrelated")).empty());
    // Variable is an explicitly declared hidden mode; it cannot become a
    // default/display rate, and fixed selections keep their original call set.
    const auto variable_json = replace(replace(variants_json, "[30,60,120]", "[30,60,120,\"variable\"]"),
        R"("note":"Synthetic interval-update hook.")",
        R"("note":"Synthetic interval-update hook.","frame_rate":"variable","nvn_driver_only":true)");
    const auto variable = parse(variable_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(variable));
    if (const auto* descriptions = std::get_if<compat::PatchDescriptions>(&variable)) {
        const auto& mode = descriptions->front();
        CHECK(compat::validate_patch(mode, image).empty());
        for (const char* rate : {"", "0", "30", "60", "120"}) {
            const auto calls = compat::select_patch_calls(mode, rate);
            CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
            if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls)) CHECK(selected->empty());
        }
        std::uint32_t admitted = 99;
        const auto words = compat::select_patch_words(mode, "variable", true, &admitted);
        CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(words) && admitted == 0);
        const auto calls = compat::select_patch_calls(mode, "variable");
        CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
        if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls))
            CHECK(selected->size() == 1 && selected->front().offset == patch.calls.front().offset &&
                  selected->front().expected_instruction == patch.calls.front().expected_instruction);
        CHECK(std::holds_alternative<std::string>(compat::select_patch_calls(mode, "variable", false)));
        CHECK(std::holds_alternative<std::string>(compat::select_patch_words(mode, "4294967295")));
        auto undeclared = mode; undeclared.frame_rate->supported_rates.pop_back();
        CHECK(!compat::validate_patch(undeclared, image).empty());
        CHECK(std::holds_alternative<std::string>(compat::select_patch_calls(undeclared, "variable")));
    }
    const auto variable_word_json = replace(variable_json, R"("frame_rate":120)", R"("frame_rate":"variable")");
    const auto variable_word = parse(variable_word_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(variable_word));
    if (const auto* descriptions = std::get_if<compat::PatchDescriptions>(&variable_word)) {
        const auto selected = compat::select_patch_words(descriptions->front(), "variable");
        CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(selected));
        if (const auto* words = std::get_if<std::vector<compat::WordPatch>>(&selected))
            CHECK(words->size() == 1 && words->front().replacement_instruction == 0xd503203fu);
    }
    CHECK(std::holds_alternative<std::string>(parse(replace(variable_json,
        R"("default":30)", R"("default":"variable")"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(variable_json,
        R"("display_refresh_rate":60)", R"("display_refresh_rate":"variable")"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(variable_json,
        "[30,60,120,\"variable\"]", "[30,60,120]"))));
    CHECK(std::holds_alternative<std::string>(parse(replace(variable_json,
        R"("nvn_driver_only":true)", R"("nvn_driver_only":1)"))));
    CHECK(std::holds_alternative<std::string>(compat::select_patch_calls(
        std::get<compat::PatchDescriptions>(variants).front(), "variable")));
    for (const char* rate : {"", "0", "30", "60", "120"}) {
        const auto selected = compat::select_patch_calls(std::get<compat::PatchDescriptions>(variants).front(), rate);
        CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(selected));
        if (const auto* calls = std::get_if<std::vector<compat::CallPatch>>(&selected))
            CHECK(calls->size() == 1 && calls->front().offset == patch.calls.front().offset &&
                  calls->front().expected_instruction == patch.calls.front().expected_instruction);
    }
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
    const char* shader_json = R"({"version":1,"patches":[{
"title_id":"0x0000000000000123",
"main_build_id":"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
"shader_inputs":{"containers":"gfd-cache","vertex_depth":"zero-to-one",
"caches":["BASE/ONE.GSC","BASE/TWO.GSC"],
"fallbacks":[{"cache":"BASE/TWO.GSC","alternatives":["BASE/ALT.GSC"]}],
"ignored_needle_offsets":["0x10"]}}]})";
    auto shader_read = parse(shader_json);
    CHECK(std::holds_alternative<compat::PatchDescriptions>(shader_read));
    if (const auto* parsed = std::get_if<compat::PatchDescriptions>(&shader_read)) {
        const auto& selected = parsed->front().shader_inputs;
        CHECK(selected.containers == compat::ShaderContainerSet::GfdCache);
        CHECK(selected.vertex_depth == compat::ShaderVertexDepth::ZeroToOne);
        CHECK(selected.caches.size() == 2 && selected.caches[1] == "BASE/TWO.GSC");
        CHECK(selected.fallbacks.size() == 1 && selected.fallbacks[0].cache == "BASE/TWO.GSC" &&
              selected.fallbacks[0].alternatives.size() == 1 &&
              selected.fallbacks[0].alternatives[0] == "BASE/ALT.GSC");
        CHECK(selected.ignored_needle_offsets.size() == 1 && selected.ignored_needle_offsets[0] == 0x10);
    }
    const auto shader_of = [](const char* body) {
        return parse(std::string(R"({"version":1,"patches":[{
"title_id":"0x0000000000000123",
"main_build_id":"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
"shader_inputs":)") + body + "}]}");
    };
    CHECK(std::holds_alternative<compat::PatchDescriptions>(shader_of(
        R"({"containers":"bnsh","vertex_depth":"minus-one-to-one"})")));
    CHECK(std::holds_alternative<std::string>(shader_of(R"({"containers":"bnsh"})")));
    CHECK(std::holds_alternative<std::string>(shader_of(
        R"({"containers":"other","vertex_depth":"zero-to-one"})")));
    CHECK(std::holds_alternative<std::string>(shader_of(
        R"({"containers":"bnsh","vertex_depth":"minus-one-to-one","caches":["BASE/ONE.GSC"]})")));
    CHECK(std::holds_alternative<std::string>(shader_of(
        R"({"containers":"gfd-cache","vertex_depth":"zero-to-one","caches":["BASE/ONE.GSC"],"fallbacks":[{"cache":"BASE/MISSING.GSC","alternatives":["BASE/ALT.GSC"]}]})")));
    CHECK(std::holds_alternative<std::string>(shader_of(
        R"({"containers":"gfd-cache","vertex_depth":"zero-to-one","extra":true})")));
    std::puts(failures ? "Compatibility patch tests failed." : "Compatibility patch tests passed.");
    return failures ? 1 : 0;
}
