// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pipelines.h"
#include "bundle.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <system_error>
#include <vector>

extern char** environ;

namespace armigo::pack {

std::variant<std::string, armigo::platform::PipelineArchiveResult> build_pack_pipelines(
    const std::filesystem::path& package, const std::vector<std::filesystem::path>& manifests,
    const std::filesystem::path& cache_root, const std::string& expected_chain_id) {
    if (expected_chain_id.empty()) {
        return std::string("This build has no shader chain, so a pipeline archive cannot be built.");
    }
    if (manifests.empty()) {
        return std::string("The matched title settings have no pipeline list.");
    }
    std::error_code error;
    if (!std::filesystem::is_directory(package, error)) {
        return "The shader package is missing: " + package.string() + ".";
    }
    for (const std::filesystem::path& manifest : manifests) {
        if (!std::filesystem::is_regular_file(manifest, error)) {
            return "The pipeline list is missing: " + manifest.string() + ".";
        }
    }
    std::filesystem::create_directories(cache_root, error);
    if (error) {
        return "Cannot create the pipeline directory " + cache_root.string() + ".";
    }
    armigo::platform::PipelineArchiveBuild build;
    build.package = package;
    build.expected_chain_id = expected_chain_id;
    build.manifests = manifests;
    build.cache_root = cache_root;
    auto result = armigo::platform::build_pipeline_archive(build);
    if (!result.ok || result.archive.empty()) {
        std::string message = result.error.empty() ? "The pipeline archive was not written." : result.error;
        message += " (rows " + std::to_string(result.rows) + ", built " + std::to_string(result.built) +
                   ", unresolved " + std::to_string(result.unresolved) + ", failed " +
                   std::to_string(result.failed) + ")";
        return message;
    }
    return result;
}

std::string warm_pack_pipelines(const std::filesystem::path& app,
                                const std::vector<std::filesystem::path>& manifests) {
    namespace fs = std::filesystem;
    const auto executable = packed_application_executable(app);
    if (const auto* failure = std::get_if<std::string>(&executable)) return *failure;
    const fs::path& player = std::get<fs::path>(executable);
    const fs::path resources = app / "Contents" / "Resources" / "Armigo";
    std::error_code error;
    if (!fs::is_regular_file(player, error)) return "The packed player is missing: " + player.string() + ".";
    if (!fs::is_directory(resources / "shaders", error)) return "The packed shader package is missing.";
    if (!fs::is_directory(resources / "pipelines", error)) return "The packed pipeline archive is missing.";
    if (manifests.empty()) return "The packed player has no pipeline list.";
    for (const fs::path& manifest : manifests) {
        if (!fs::is_regular_file(manifest, error))
            return "The pipeline list is missing: " + manifest.string() + ".";
    }

    std::vector<std::string> arguments{
        player.string(), "pso-archive", "--shaders", (resources / "shaders").string()};
    for (const fs::path& manifest : manifests) {
        arguments.emplace_back("--manifest");
        arguments.push_back(manifest.string());
    }
    arguments.emplace_back("--cache");
    arguments.push_back((resources / "pipelines").string());
    arguments.emplace_back("--warm-only");
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (std::string& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    pid_t pid = 0;
    if (const int spawned = posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), environ); spawned != 0)
        return "Cannot start the packed player's Metal cache warm-up: " + std::string(std::strerror(spawned)) + ".";
    int status = 0;
    pid_t waited = 0;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) return "Cannot wait for the packed player's Metal cache warm-up.";
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        return "The packed player's Metal cache warm-up failed; see its diagnostics above.";
    return {};
}

} // namespace armigo::pack
