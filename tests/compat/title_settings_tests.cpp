// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Checks every title folder of the repository (argv[1]: titles/) and the
// title-settings lookup on synthetic folders.

#include <armigo/compat/title_settings.h>

#include "pso_manifest.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <unistd.h>
#include <string>
#include <array>
#include <algorithm>
#include <bit>
#include <optional>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)

void fail(const std::filesystem::path& where, const std::string& why) {
    std::fprintf(stderr, "%s: %s\n", where.c_str(), why.c_str());
    ++failures;
}

// Exercise the declared initializer/return words on a synthetic register bank.
// A counter's floating representation must not escape into an integer state.
void check_counter_state_return(const std::vector<armigo::compat::WordPatch>& words) {
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

// One title folder: named by its title id, a profile that reads and names
// that title, pipeline lists whose every row parses, and nothing else.
void check_folder(const std::filesystem::path& folder) {
    namespace fs = std::filesystem;
    using namespace armigo;
    const auto title = compat::parse_title_id_text(folder.filename().string());
    if (!title) return fail(folder, "a title folder is named by its title id: 16 uppercase hexadecimal digits");
    auto read = compat::read_title_settings(folder, *title);
    if (const auto* error = std::get_if<std::string>(&read)) return fail(folder, *error);
    const auto& settings = std::get<compat::TitleSettings>(read);
    std::set<std::string> expected{"profile.json"};
    for (const auto& patch : settings.patches) {
        bool zero = true;
        for (const auto byte : patch.identity.main_build_id) zero &= byte == std::byte{0};
        if (zero) fail(folder, "a build id is all zeros");
        if (*title == 0x01004b10157f2000ull) {
            // The hidden experiment must leave all admitted fixed call sets
            // empty and retain their established instruction counts.
            for (const auto& [rate, count] : std::array<std::pair<const char*, std::size_t>, 5>{{
                     {"", 0}, {"0", 0}, {"30", 0}, {"60", 17}, {"120", 18}}}) {
                const auto words = compat::select_patch_words(patch, rate);
                const auto calls = compat::select_patch_calls(patch, rate);
                CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(words));
                CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
                if (const auto* selected = std::get_if<std::vector<compat::WordPatch>>(&words)) CHECK(selected->size() == count);
                if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls)) CHECK(selected->empty());
            }
            const auto calls = compat::select_patch_calls(patch, "variable");
            CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
            if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls)) CHECK(!selected->empty());
            const auto variable_words = compat::select_patch_words(patch, "variable");
            CHECK(std::holds_alternative<std::vector<compat::WordPatch>>(variable_words));
            if (const auto* selected = std::get_if<std::vector<compat::WordPatch>>(&variable_words))
                check_counter_state_return(*selected);
        }
        if (*title == 0x01007ef00011e000ull) {
            CHECK(patch.calls.size() == 2);
            for (const char* rate : {"", "0", "30", "60", "120"}) {
                const auto calls = compat::select_patch_calls(patch, rate);
                CHECK(std::holds_alternative<std::vector<compat::CallPatch>>(calls));
                if (const auto* selected = std::get_if<std::vector<compat::CallPatch>>(&calls))
                    CHECK(std::equal(selected->begin(), selected->end(), patch.calls.begin(), patch.calls.end(),
                        [](const auto& left, const auto& right) {
                            return left.offset == right.offset && left.expected_instruction == right.expected_instruction;
                        }));
            }
        }
        if (patch.pso_manifest.empty()) continue;
        const fs::path list = patch.pso_manifest;
        expected.insert(list.filename().string());
        std::ifstream in(list);
        std::string line;
        std::size_t rows = 0, number = 0;
        while (std::getline(in, line)) {
            ++number;
            if (line.empty() || line.front() == '#') continue;
            if (!platform::metal::nvn_driver::parse_pso_row(line))
                fail(list, "line " + std::to_string(number) + " is not a pipeline row");
            ++rows;
        }
        if (rows == 0) fail(list, "a pipeline list has no rows");
    }
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(folder, error))
        if (!expected.contains(entry.path().filename().string()))
            fail(entry.path(), "a title folder holds only its profile and the pipeline lists it names");
}

void write(const std::filesystem::path& file, const std::string& text) {
    std::filesystem::create_directories(file.parent_path());
    std::ofstream(file) << text;
}

std::string profile(const char* title, const char* extra = "") {
    std::ostringstream out;
    out << R"({"version":1,"patches":[{"title_id":")" << title
        << R"(","main_build_id":"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20")" << extra
        << "}]}";
    return out.str();
}

void check_lookup() {
    namespace fs = std::filesystem;
    using namespace armigo;
    CHECK(compat::title_id_text(0x01007ef00011e000) == "01007EF00011E000");
    CHECK(compat::parse_title_id_text("01007EF00011E000") == 0x01007ef00011e000ull);
    CHECK(!compat::parse_title_id_text("01007ef00011e000"));
    CHECK(!compat::parse_title_id_text("1007EF00011E000"));

    const auto root = fs::temp_directory_path() / ("armigo-title-settings-" + std::to_string(::getpid()));
    fs::remove_all(root);
    write(root / "0100000000001000/profile.json", profile("0x0100000000001000", R"(,"pipelines":"pipelines.tsv")"));
    write(root / "0100000000001000/pipelines.tsv", "# rows\nS\t0.ab.0\n");
    write(root / "0100000000002000/profile.json", profile("0x0100000000003000"));
    write(root / "0100000000004000/profile.json", profile("0x0100000000004000", R"(,"pipelines":"missing.tsv")"));
    write(root / "0100000000005000/profile.json", profile("0x0100000000005000", R"(,"pipelines":"../escape.tsv")"));

    ::setenv("ARMIGO_TITLES", root.c_str(), 1);
    auto found = compat::find_title_settings(0x0100000000001000);
    CHECK(std::holds_alternative<std::optional<compat::TitleSettings>>(found));
    if (auto* settings = std::get_if<std::optional<compat::TitleSettings>>(&found); settings && *settings) {
        CHECK((*settings)->folder == root / "0100000000001000");
        CHECK((*settings)->patches.size() == 1);
        CHECK((*settings)->patches.front().pso_manifest == (root / "0100000000001000/pipelines.tsv").string());
    } else {
        CHECK(false);
    }
    // No folder: no settings, not an error.
    found = compat::find_title_settings(0x0100000000009000);
    CHECK(std::holds_alternative<std::optional<compat::TitleSettings>>(found) &&
          !std::get<std::optional<compat::TitleSettings>>(found));
    // A folder that describes another title, names a missing list or one
    // outside the folder is an error.
    CHECK(std::holds_alternative<std::string>(compat::find_title_settings(0x0100000000002000)));
    CHECK(std::holds_alternative<std::string>(compat::find_title_settings(0x0100000000004000)));
    CHECK(std::holds_alternative<std::string>(compat::find_title_settings(0x0100000000005000)));
    // "off" turns title settings off.
    ::setenv("ARMIGO_TITLES", "off", 1);
    found = compat::find_title_settings(0x0100000000001000);
    CHECK(std::holds_alternative<std::optional<compat::TitleSettings>>(found) &&
          !std::get<std::optional<compat::TitleSettings>>(found));
    ::unsetenv("ARMIGO_TITLES");

    CHECK(compat::title_save_directory(0x0100000000001000).filename() == "0100000000001000");
    CHECK(compat::title_save_directory(0x0100000000001000).parent_path().filename() == "Armigo");
    CHECK(compat::title_cache_directory(0x0100000000001000).filename() == "0100000000001000");
    fs::remove_all(root);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fputs("usage: armigo_title_settings_tests TITLES_DIR\n", stderr);
        return 2;
    }
    std::size_t folders = 0;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(argv[1], error)) {
        if (!entry.is_directory()) {
            fail(entry.path(), "titles/ holds only title folders");
            continue;
        }
        check_folder(entry.path());
        ++folders;
    }
    if (error || folders == 0) fail(argv[1], "no title folders");
    check_lookup();
    if (failures) std::fprintf(stderr, "%u failures\n", failures);
    else std::printf("%zu title folders checked\n", folders);
    return failures ? 1 : 0;
}
