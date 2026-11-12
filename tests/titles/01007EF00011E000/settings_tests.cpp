// ============================================================================
// Armigo Tests - Title 01007EF00011E000 Settings
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Checks the committed settings of title 01007EF00011E000.
// Usage: armigo_01007EF00011E000_settings_tests TITLES_DIR

#include <armigo/compat/title_settings.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace {

using namespace armigo;

unsigned failures = 0;
#define CHECK(value) \
    do { \
        if (!(value)) { \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); \
            ++failures; \
        } \
    } while (false)

void check_descriptions(const compat::TitleSettings& settings) {
    for (const auto& patch : settings.patches) {
        CHECK(patch.calls.size() == 2);
        CHECK(patch.frame_rate && patch.frame_rate->default_rate == 60);
        CHECK(patch.frame_rate->supported_rates == std::vector<std::uint32_t>({30, 60, 120}));
        CHECK(patch.frame_rate->experimental_rates == std::vector<std::uint32_t>({120}));
        CHECK(patch.resolution && patch.resolution->gpu_budget_rate == 120);
        const auto baseline = std::get<std::vector<compat::WordPatch>>(compat::select_patch_words(patch, "60"));
        CHECK(baseline.size() == 26);
        for (const char* rate : {"", "0", "30", "60", "120"}) {
            const auto words = compat::select_patch_words(patch, rate);
            CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(words));
            if (const auto* selected = std::get_if<std::vector<compat::WordPatch>>(&words))
                CHECK(std::equal(selected->begin(), selected->end(), baseline.begin(), baseline.end(),
                    [](const auto& left, const auto& right) {
                        return left.offset == right.offset &&
                            left.expected_instruction == right.expected_instruction &&
                            left.replacement_instruction == right.replacement_instruction && !left.frame_rate;
                    }));
            const auto calls = compat::select_patch_calls(patch, rate);
            CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
            if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls))
                CHECK(std::equal(selected->begin(), selected->end(), patch.calls.begin(), patch.calls.end(),
                    [](const auto& left, const auto& right) {
                        return left.offset == right.offset &&
                            left.expected_instruction == right.expected_instruction;
                    }));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fputs("usage: armigo_01007EF00011E000_settings_tests TITLES_DIR\n", stderr);
        return 2;
    }
    const std::filesystem::path folder = std::filesystem::path(argv[1]) / "01007EF00011E000";
    auto read = armigo::compat::read_title_settings(folder, 0x01007EF00011E000ull);
    if (const auto* error = std::get_if<std::string>(&read)) {
        std::fprintf(stderr, "%s: %s\n", folder.c_str(), error->c_str());
        return 1;
    }
    check_descriptions(std::get<armigo::compat::TitleSettings>(read));
    if (failures != 0) {
        std::fprintf(stderr, "%u failures\n", failures);
        return 1;
    }
    std::puts("01007EF00011E000 settings checks passed");
    return 0;
}
