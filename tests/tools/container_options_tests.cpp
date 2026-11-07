// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Runner-shell container option tests: argument validation for
// --container/--keys/--cache and the metadata-signature report field.

#include "runner_shell.h"

#include <filesystem>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using armigo::platform::PresentationMode;
using armigo::tools::parse_runner_arguments;
using armigo::tools::RunnerArguments;

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

[[nodiscard]] std::vector<char*> as_arguments(const std::vector<std::string>& values) {
    std::vector<char*> pointers;
    pointers.reserve(values.size());
    for (const std::string& value : values) {
        pointers.push_back(const_cast<char*>(value.c_str()));
    }
    return pointers;
}

[[nodiscard]] std::filesystem::path make_temp_directory(const std::string& name) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

} // namespace

int main() {
    const auto directory = make_temp_directory("armigo-container-cli-test");
    const std::filesystem::path container = directory / "game.nsp";
    const std::filesystem::path keys = directory / "prod.keys";
    const std::filesystem::path cache = directory / "cache";
    const std::filesystem::path shaders = directory / "shaders";
    std::filesystem::create_directories(shaders);
    { FILE* file = std::fopen(container.c_str(), "wb"); if (file) std::fputs("x", file); std::fclose(file); }
    { FILE* file = std::fopen(keys.c_str(), "wb"); if (file) std::fputs("k = 00\n", file); std::fclose(file); }

    {
        const auto parsed = parse_runner_arguments(
            PresentationMode::headless,
            as_arguments({"--container", container.string(), "--keys", keys.string(),
                          "--shaders", shaders.string()}));
        CHECK(parsed.ok);
        CHECK(parsed.arguments.container_file == container);
        CHECK(parsed.arguments.keys_file == keys);
        CHECK(!parsed.arguments.cache_directory.has_value());
    }
    {
        const auto parsed = parse_runner_arguments(
            PresentationMode::headless,
            as_arguments({"--container", container.string(), "--keys", keys.string(),
                          "--cache", cache.string(), "--shaders", shaders.string()}));
        CHECK(parsed.ok);
        CHECK(parsed.arguments.cache_directory == cache);
    }
    {
        // Container input excludes extracted directories.
        const auto parsed = parse_runner_arguments(
            PresentationMode::headless,
            as_arguments({"--container", container.string(), "--keys", keys.string(),
                          "--exefs", shaders.string(), "--shaders", shaders.string()}));
        CHECK(!parsed.ok);
    }
    {
        // Keys are mandatory with container input.
        const auto parsed = parse_runner_arguments(
            PresentationMode::headless,
            as_arguments({"--container", container.string(), "--shaders",
                          shaders.string()}));
        CHECK(!parsed.ok);
    }
    {
        const auto parsed = parse_runner_arguments(
            PresentationMode::headless,
            as_arguments({"--keys", keys.string(), "--shaders", shaders.string()}));
        CHECK(!parsed.ok);
    }
    {
        // Container input is headless-only in this increment.
        const auto parsed = parse_runner_arguments(
            PresentationMode::windowed,
            as_arguments({"--container", container.string(), "--keys", keys.string(),
                          "--shaders", shaders.string()}));
        CHECK(!parsed.ok);
    }
    {
        // Missing files are usage errors before any run starts.
        const auto parsed = parse_runner_arguments(
            PresentationMode::headless,
            as_arguments({"--container", directory.string() + "/absent.nsp", "--keys",
                          keys.string(), "--shaders", shaders.string()}));
        CHECK(!parsed.ok);
    }

    {
        const auto parsed = parse_runner_arguments(PresentationMode::headless,
            as_arguments({"--nup", container.string()}));
        CHECK(parsed.ok && parsed.arguments.nup_file == container);
        CHECK(!parsed.arguments.keys_file);
        for (const auto flag : {"--keys", "--container", "--exefs", "--romfs", "--cache", "--update"}) {
            const auto mixed = parse_runner_arguments(PresentationMode::headless,
                as_arguments({"--nup", container.string(), flag, container.string()}));
            CHECK(!mixed.ok);
        }
        CHECK(!parse_runner_arguments(PresentationMode::windowed,
            as_arguments({"--nup", container.string()})).ok);
        CHECK(!parse_runner_arguments(PresentationMode::headless,
            as_arguments({"--nup", (directory / "absent.nup").string()})).ok);
    }

    // The report carries the verification state; extraction failures leave
    // it not_verified and a verified container run sets it.
    {
        armigo::tools::ShellReport report;
        armigo::tools::ShellReportFields fields = armigo::tools::report_fields(report);
        CHECK(!fields.metadata_signature_verified);
        report.metadata_signature_verified = true;
        fields = armigo::tools::report_fields(report);
        CHECK(fields.metadata_signature_verified);
    }

    std::filesystem::remove_all(directory);
    if (failures == 0) {
        std::printf("container option tests passed\n");
        return 0;
    }
    std::printf("%d container option test check(s) failed\n", failures);
    return 1;
}
