// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/exefs_directory.h>
#include <armigo/runtime/preparation_format.h>
#include <armigo/runtime/prepared_modules.h>
#include <armigo/runtime/launch_plan.h>
#include <cstdlib>

#include <cstdio>
#include <filesystem>
#include <string_view>
#include <variant>

namespace {

struct Arguments {
    std::filesystem::path exefs;
    std::filesystem::path output;
    bool verify = false;
    bool synthetic = false;
    bool embedded = false;
    std::filesystem::path patches;
};

void usage(FILE* stream) {
    std::fprintf(stream,
                 "Usage: armigo-prepare [--verify] [--synthetic-test | --embedded [--patches FILE]] --exefs DIR --output DIR\n"
                 "Prepare an ExeFS directory for the installed Armigo runtime, or verify\n"
                 "an existing prepared output without executing guest code.\n");
}

bool parse_arguments(int argc, char** argv, Arguments& arguments) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help") {
            usage(stdout);
            return false;
        }
        if (argument == "--embedded") {
            arguments.embedded = true;
            continue;
        }
        if (argument == "--synthetic-test") {
            arguments.synthetic = true;
            continue;
        }
        if (argument == "--verify") {
            arguments.verify = true;
            continue;
        }
        if ((argument == "--exefs" || argument == "--output" || argument == "--patches") && index + 1 < argc) {
            auto& destination = argument == "--exefs" ? arguments.exefs :
                                argument == "--patches" ? arguments.patches : arguments.output;
            if (!destination.empty()) {
                std::fprintf(stderr, "armigo-prepare: duplicate option: %.*s\n",
                             static_cast<int>(argument.size()), argument.data());
                return false;
            }
            destination = argv[++index];
            continue;
        }
        std::fprintf(stderr, "armigo-prepare: invalid argument: %.*s\n",
                     static_cast<int>(argument.size()), argument.data());
        return false;
    }
    if ((arguments.embedded && arguments.synthetic) || (!arguments.embedded && !arguments.patches.empty())) {
        std::fputs("armigo-prepare: --embedded conflicts with --synthetic-test; --patches requires --embedded\n", stderr);
        return false;
    }
    if (arguments.exefs.empty() || arguments.output.empty()) {
        std::fputs("armigo-prepare: --exefs and --output are required\n", stderr);
        return false;
    }
    return true;
}

int report_failure(const armigo::runtime::LaunchFailure& failure) {
    std::fprintf(stderr, "armigo-prepare: %s: %s\n", failure.operation.c_str(),
                 failure.detail.c_str());
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    Arguments arguments;
    if (!parse_arguments(argc, argv, arguments)) {
        if (argc == 2 && std::string_view(argv[1]) == "--help") return 0;
        usage(stderr);
        return 2;
    }

    auto collected = armigo::loader::read_exefs_directory(
        arguments.exefs, {.collect_all_subsdk = arguments.synthetic});
    if (const auto* failure = std::get_if<armigo::loader::ExeFsDirectoryFailure>(&collected)) {
        std::fprintf(stderr, "armigo-prepare: read ExeFS: %s: %s\n",
                     armigo::loader::to_string(failure->code), failure->detail.c_str());
        return 1;
    }
    const auto& input = std::get<armigo::loader::ExeFsDirectoryInput>(collected);
    for (const auto& name : input.ignored_modules)
        std::fprintf(stderr, "armigo-prepare: ignored %s after the first missing subsdk slot.\n",
                     name.c_str());
    if (arguments.embedded) {
        using namespace armigo::runtime;
        if (!input.process_metadata) {
            std::fputs("armigo-prepare: embedded preparation requires main.npdm\n", stderr);
            return 1;
        }
        auto derived = derive_launch_profile(*input.process_metadata,
            NpdmTrustPolicy::structurally_validated_extracted_input);
        if (const auto* failure = std::get_if<LaunchFailure>(&derived)) return report_failure(*failure);
        armigo::compat::PatchDescriptions patches;
        auto patch_file = arguments.patches;
        if (patch_file.empty()) if (const char* path = std::getenv("ARMIGO_COMPAT_PATCHES")) patch_file = path;
        if (!patch_file.empty()) {
            auto parsed = armigo::compat::read_patch_descriptions(patch_file);
            if (const auto* error = std::get_if<std::string>(&parsed)) {
                std::fprintf(stderr, "armigo-prepare: compatibility settings: %s\n", error->c_str());
                return 1;
            }
            patches = std::get<armigo::compat::PatchDescriptions>(std::move(parsed));
        }
        const auto& profile = std::get<LaunchProfile>(derived);
        auto key = std::make_shared<armigo::platform::ThreadSlotKey>();
        if (arguments.verify) {
            auto opened = open_embedded_modules(arguments.output);
            if (const auto* failure = std::get_if<LaunchFailure>(&opened)) return report_failure(*failure);
            const auto& files = std::get<EmbeddedNativeFiles>(opened);
            auto loaded = load_native_launch(input, profile, key, patches,
                {true, nullptr, &files.artifacts(), files.backings()});
            if (const auto* failure = std::get_if<LaunchFailure>(&loaded)) return report_failure(*failure);
            std::printf("Verified %zu embedded module(s).\n", files.artifacts().modules.size());
            return 0;
        }
        EmbeddedNativeArtifacts artifacts;
        auto loaded = load_native_launch(input, profile, key, patches, {true, &artifacts});
        if (const auto* failure = std::get_if<LaunchFailure>(&loaded)) return report_failure(*failure);
        auto written = write_embedded_modules(arguments.output, artifacts);
        if (const auto* failure = std::get_if<LaunchFailure>(&written)) return report_failure(*failure);
        std::printf("Prepared %zu embedded module(s).\n", artifacts.modules.size());
        return 0;
    }
    const armigo::runtime::PreparationOptions options{
        .required_preparation_identity = armigo::runtime::installed_preparation_identity(),
        .max_artifact_bytes = armigo::runtime::default_prepared_artifact_limit,
        .mode = arguments.synthetic ? armigo::runtime::PreparationMode::synthetic_test
                                    : armigo::runtime::PreparationMode::automatic,
    };

    if (arguments.verify) {
        auto result = armigo::runtime::read_prepared_modules(arguments.output, input, options);
        if (const auto* failure = std::get_if<armigo::runtime::LaunchFailure>(&result)) {
            return report_failure(*failure);
        }
        std::printf("Verified %zu prepared module(s).\n",
                    std::get<armigo::runtime::PreparedModuleArtifacts>(result).modules.size());
        return 0;
    }

    auto prepared = armigo::runtime::prepare_modules(input, options);
    if (const auto* failure = std::get_if<armigo::runtime::LaunchFailure>(&prepared)) {
        return report_failure(*failure);
    }
    const auto& artifacts = std::get<armigo::runtime::PreparedModuleArtifacts>(prepared);
    auto written = armigo::runtime::write_prepared_modules(arguments.output, artifacts);
    if (const auto* failure = std::get_if<armigo::runtime::LaunchFailure>(&written)) {
        return report_failure(*failure);
    }
    std::printf("Prepared %zu module(s).\n", artifacts.modules.size());
    return 0;
}
