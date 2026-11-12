// ============================================================================
// Armigo Tests - Title 01004B10157F2000 Settings
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Checks the committed settings of title 01004B10157F2000.
// Usage: armigo_01004B10157F2000_settings_tests TITLES_DIR

#include <armigo/compat/title_settings.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
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

// Exercise the declared initializer/return words on a synthetic register bank.
// A counter's floating representation must not escape into an integer state.
void check_counter_state_return(const std::vector<compat::WordPatch>& words) {
    const auto move = [](std::uint32_t word)
        -> std::optional<std::pair<unsigned, std::uint32_t>> {
        if ((word & 0xff800000u) != 0x52800000u) return {};
        const unsigned shift = ((word >> 21) & 3u) * 16;
        if (shift >= 32) return {};
        return std::pair{word & 31u, ((word >> 5) & 0xffffu) << shift};
    };
    unsigned checked = 0;
    for (const auto& initial : words) {
        const auto original = move(initial.expected_instruction);
        const auto replacement = move(initial.replacement_instruction);
        if (!original || !replacement || original->first != replacement->first ||
            replacement->second != std::bit_cast<std::uint32_t>(static_cast<float>(original->second)))
            continue;
        const auto branch = std::find_if(words.begin(), words.end(), [&](const auto& word) {
            return word.offset == initial.offset + 12 &&
                   (word.replacement_instruction & 0xfc000000u) == 0x14000000u;
        });
        if (branch == words.end()) continue;
        ++checked;
        std::int64_t displacement = branch->replacement_instruction & 0x03ffffffu;
        if (displacement & (1ll << 25)) displacement -= 1ll << 26;
        const auto target = static_cast<std::uint64_t>(static_cast<std::int64_t>(branch->offset) + displacement * 4);
        const auto restore = std::find_if(words.begin(), words.end(), [&](const auto& word) {
            return word.offset == target;
        });
        CHECK(restore != words.end());
        if (restore == words.end()) continue;
        const auto restored = move(restore->replacement_instruction);
        CHECK(restored && *restored == *original);
        CHECK(restore->expected_instruction == restore->replacement_instruction);
        if (!restored) continue;
        for (std::uint32_t seed : {0u, UINT32_MAX}) {
            std::array<std::uint32_t, 32> registers;
            registers.fill(seed);
            registers[replacement->first] = replacement->second;
            const auto counter_bits = registers[original->first];
            registers[restored->first] = restored->second;
            const auto state = registers[original->first];
            CHECK(std::bit_cast<float>(counter_bits) == static_cast<float>(original->second));
            CHECK(state == original->second);
        }
    }
    CHECK(checked == 1);
}

void check_descriptions(const compat::TitleSettings& settings) {
    for (const auto& patch : settings.patches) {
        // The hidden experiment must leave all admitted fixed call sets
        // empty and retain their established instruction counts.
        for (const auto& [rate, count] : std::array<std::pair<const char*, std::size_t>, 5>{{
                 {"", 0}, {"0", 0}, {"30", 0}, {"60", 17}, {"120", 18}}}) {
            const auto words = compat::select_patch_words(patch, rate);
            const auto calls = compat::select_patch_calls(patch, rate);
            CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(words));
            CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
            if (const auto* selected = std::get_if<std::vector<compat::WordPatch>>(&words))
                CHECK(selected->size() == count);
            if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls))
                CHECK(selected->empty());
        }
        const auto calls = compat::select_patch_calls(patch, "variable");
        CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
        if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls))
            CHECK(!selected->empty());
        const auto variable_words = compat::select_patch_words(patch, "variable");
        CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(variable_words));
        if (const auto* selected = std::get_if<std::vector<compat::WordPatch>>(&variable_words))
            check_counter_state_return(*selected);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fputs("usage: armigo_01004B10157F2000_settings_tests TITLES_DIR\n", stderr);
        return 2;
    }
    const std::filesystem::path folder = std::filesystem::path(argv[1]) / "01004B10157F2000";
    auto read = armigo::compat::read_title_settings(folder, 0x01004B10157F2000ull);
    if (const auto* error = std::get_if<std::string>(&read)) {
        std::fprintf(stderr, "%s: %s\n", folder.c_str(), error->c_str());
        return 1;
    }
    check_descriptions(std::get<armigo::compat::TitleSettings>(read));
    if (failures != 0) {
        std::fprintf(stderr, "%u failures\n", failures);
        return 1;
    }
    std::puts("01004B10157F2000 settings checks passed");
    return 0;
}
