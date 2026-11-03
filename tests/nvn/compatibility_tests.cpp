// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/compat/patches.h>
#include <cstdio>
#include <fstream>
#include <unistd.h>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)
armigo::compat::PatchReadResult parse(const std::string& flag, const std::string& extra = "") {
    // Independently authored identity used only for parser tests.
    const auto contents = std::string(R"({"version":1,"patches":[{"title_id":"0x0000000000000001","main_build_id":")") +
        std::string(64, '0') + R"(","nvn_driver":)" + flag + extra + "}]}";
    auto pattern = (std::filesystem::temp_directory_path() / "armigo-nvn-compat-XXXXXX").string();
    const int fd = mkstemp(pattern.data());
    if (fd < 0) return std::string("Cannot create synthetic compatibility file.");
    close(fd);
    { std::ofstream out(pattern); out << contents; }
    auto result = armigo::compat::read_patch_descriptions(pattern);
    unlink(pattern.c_str());
    return result;
}
}
int main() {
    using namespace armigo::compat;
    for (const auto flag : {"true", "false"}) {
        const auto result = parse(flag);
        CHECK(std::holds_alternative<PatchDescriptions>(result));
        if (const auto* descriptions = std::get_if<PatchDescriptions>(&result)) {
            CHECK(descriptions->size() == 1);
            CHECK(descriptions->front().nvn_driver == (std::string_view(flag) == "true"));
            CHECK(descriptions->front().module.empty());
            CHECK(descriptions->front().calls.empty());
            CHECK(validate_patch(descriptions->front(), {}).empty());
        }
    }
    for (const auto flag : {"1", "0", "null", "[]", "{}", "\"true\""})
        CHECK(std::holds_alternative<std::string>(parse(flag)));
    {
        // The title's pipeline list is an optional file name in the profile's
        // folder; the description holds its path there.
        const auto folder = std::filesystem::temp_directory_path();
        const auto list = folder / ("armigo-nvn-compat-list-" + std::to_string(getpid()) + ".tsv");
        { std::ofstream out(list); out << "S\t0.ab.0\n"; }
        const auto result = parse("true", ",\"pipelines\":\"" + list.filename().string() + "\"");
        const auto* descriptions = std::get_if<PatchDescriptions>(&result);
        CHECK(descriptions && descriptions->front().pso_manifest == list.string());
        std::filesystem::remove(list);
        const auto absent = parse("true");
        const auto* plain = std::get_if<PatchDescriptions>(&absent);
        CHECK(plain && plain->front().pso_manifest.empty());
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":\"\"")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":7")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":\"missing.tsv\"")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":\"../x.tsv\"")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pso_manifest\":\"/a.tsv\"")));
    }
    CHECK(std::holds_alternative<std::string>(parse("true", ",\"unknown\":0")));
    CHECK(std::holds_alternative<std::string>(parse("true", ",\"calls\":[]")));
    std::printf("NVN compatibility failures: %u\n", failures);
    return failures ? 1 : 0;
}
