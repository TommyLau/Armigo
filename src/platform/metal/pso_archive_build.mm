// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/pso_archive_build.h>

#include <armigo/platform/metal_shader_package.h>

#include "nvn_replay.h"
#include "pso_manifest.h"

#import <Metal/Metal.h>

#include <signal.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <system_error>
#include <vector>

namespace armigo::platform {
namespace {

using namespace armigo::platform::metal::nvn_driver;

unsigned performance_cores() {
    int cores = 0;
    std::size_t size = sizeof cores;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nullptr, 0) != 0 || cores <= 0) return 4;
    return static_cast<unsigned>(cores);
}

PipelineArchiveResult failure(std::string message) {
    PipelineArchiveResult result;
    result.error = std::move(message);
    return result;
}

std::string os_build() {
    char build[64]{};
    std::size_t size = sizeof build - 1;
    sysctlbyname("kern.osversion", build, &size, nullptr, 0);
    return build;
}

// What an archive directory was built for, kept beside its archive so a later
// build can tell this package's archives for another OS build or GPU apart.
std::string identity_text(const MetalShaderPackage& package, id<MTLDevice> device) {
    return "package " + package.content_id() + "\nchain " + package.chain_id() + "\nos " + os_build() +
           "\ngpu " + (device.name ? std::string(device.name.UTF8String) : std::string()) + "\n";
}

// The value of one "name value" line of an archive directory's identity.
std::string identity_field(const std::string& text, const std::string& name) {
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);)
        if (line.rfind(name + " ", 0) == 0) return line.substr(name.size() + 1);
    return {};
}

bool build_running(const std::string& pid_text) {
    char* end = nullptr;
    const auto pid = std::strtol(pid_text.c_str(), &end, 10);
    return pid > 0 && end && *end == 0 && (pid == getpid() || kill(static_cast<pid_t>(pid), 0) == 0);
}

// Removes what earlier builds left: staging and previous directories of builds
// no longer running, archive directories without an identity (written before
// identities were kept), and archive directories of this package and GPU for
// another OS build or shader chain. Another package's archives and another
// GPU's are kept, and so is an archive directory changed in the last minute:
// a build swapping it in or a capturing session may not have written its
// identity yet.
void prune(const std::filesystem::path& root, const std::string& current) {
    std::error_code error;
    const auto recent = [](const std::filesystem::directory_entry& entry) {
        std::error_code failed;
        const auto written = entry.last_write_time(failed);
        return failed || std::filesystem::file_time_type::clock::now() - written < std::chrono::minutes(1);
    };
    std::vector<std::filesystem::path> unowned_previous;
    bool other_build = false;
    for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
        if (!entry.is_directory()) continue;
        const auto name = entry.path().filename().string();
        const auto previous = name.find(".previous");
        if (name.rfind(".archive-build-", 0) == 0) {
            const auto pid = name.substr(15);
            if (!build_running(pid)) std::filesystem::remove_all(entry.path(), error);
            else if (pid != std::to_string(getpid())) other_build = true;
        } else if (previous != std::string::npos && name.rfind("archive-", 0) == 0) {
            const auto owner = name.substr(previous + 9);
            // A previous directory names the build that set it aside; one
            // without a name is from an older build.
            if (owner.empty()) unowned_previous.push_back(entry.path());
            else if (owner[0] == '-' && !build_running(owner.substr(1))) std::filesystem::remove_all(entry.path(), error);
        } else if (name.rfind("archive-", 0) == 0 && !recent(entry)) {
            std::ifstream in(entry.path() / "identity.txt");
            if (!in) {
                std::filesystem::remove_all(entry.path(), error);
                continue;
            }
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (text != current && identity_field(text, "package") == identity_field(current, "package") &&
                identity_field(text, "gpu") == identity_field(current, "gpu"))
                std::filesystem::remove_all(entry.path(), error);
        }
    }
    if (!other_build)
        for (const auto& path : unowned_previous) std::filesystem::remove_all(path, error);
}

} // namespace

// The build runs the driver's own pipeline warm-up over the manifests with
// archive capture on, in a staging root, so it compiles exactly the pipeline
// descriptors the replay asks the archive for at run time. The staged archive
// directory then replaces the one in the cache root. A warm-only build instead
// reads the existing archive and leaves it untouched.
PipelineArchiveResult build_pipeline_archive(const PipelineArchiveBuild& build) {
    const auto begin = std::chrono::steady_clock::now();
    MetalShaderPackage::Options options;
    if (build.expected_chain_id.empty()) return failure("this build accepts no shader package");
    options.expected_chain_id = build.expected_chain_id;
    options.allow_development_subset = true;
    options.lazy_entries = true;
    options.prefetch_threads = std::clamp(performance_cores(), 4u, 12u);
    auto opened = MetalShaderPackage::open(build.package, std::move(options));
    if (const auto* error = std::get_if<MetalShaderPackageError>(&opened))
        return failure("the shader package does not open: " + error->detail);
    auto package = std::get<std::shared_ptr<MetalShaderPackage>>(std::move(opened));

    // One manifest in row order: duplicates keep their first place.
    std::vector<PsoRow> rows;
    for (const auto& path : build.manifests) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return failure("cannot read the manifest " + path.string());
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        rows = merge_pso_manifests(rows, parse_pso_manifest(text));
    }
    if (rows.empty()) return failure("the manifests hold no rows");

    std::error_code error;
    std::filesystem::create_directories(build.cache_root, error);
    const auto staging = build.cache_root / (".archive-build-" + std::to_string(getpid()));
    std::filesystem::remove_all(staging, error);
    std::filesystem::create_directories(staging, error);
    if (error) return failure("cannot create " + staging.string() + ": " + error.message());
    const auto merged = staging / "manifest.tsv";
    {
        std::ofstream out(merged, std::ios::binary);
        for (const auto& row : rows) out << row.text() << '\n';
        if (!out) return failure("cannot write " + merged.string());
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::filesystem::remove_all(staging, error);
        return failure("no Metal device");
    }
    const auto identity = identity_text(*package, device);
    if (!build.warm_only) prune(build.cache_root, identity);
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    if (!queue) {
        std::filesystem::remove_all(staging, error);
        return failure("no Metal 4 command queue");
    }
    auto resources = std::make_unique<Resources>(device);
    auto replay = create_replay_backend(device, queue, *resources);
    if (!replay) {
        std::filesystem::remove_all(staging, error);
        return failure("no replay backend");
    }
    PipelineWarmupConfig config;
    config.package = package;
    config.seed_manifest = merged.string();
    config.archive_root = build.warm_only ? build.cache_root.string() : staging.string();
    config.archive_capture = !build.warm_only;
    config.archive_open_rows = !build.warm_only;
    config.warm_threads = build.threads ? build.threads : performance_cores();
    config.warm = true;
    replay->configure_pipeline_warmup(config);
    const auto report = replay->finish_pipeline_warmup();
    replay.reset();

    PipelineArchiveResult result;
    result.rows = report.rows;
    result.built = report.built;
    result.unresolved = report.unresolved;
    result.undecodable = report.undecodable;
    result.failed = report.failed;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    if (!report.finished || report.archive_incomplete || (!build.warm_only && report.archive.empty())) {
        std::filesystem::remove_all(staging, error);
        return failure(!report.finished ? "the warm-up did not finish"
                       : report.archive_incomplete ? "an archive file was not written (see the log)"
                                                   : "no archive was written");
    }
    if (build.warm_only) {
        std::filesystem::remove_all(staging, error);
        if (report.archive_misses || report.failed) {
            result.error = "the existing archive did not satisfy every warm-up row (" +
                           std::to_string(report.archive_misses) + " archive misses, " +
                           std::to_string(report.failed) + " failed rows)";
            return result;
        }
        result.ok = true;
        return result;
    }
    // Swap the staged archive directory in for the cache root's.
    const std::filesystem::path written(report.archive);
    const auto directory = written.parent_path();
    {
        std::ofstream out(directory / "identity.txt", std::ios::binary);
        out << identity;
    }
    const auto target = build.cache_root / directory.filename();
    const auto previous =
        build.cache_root / (directory.filename().string() + ".previous-" + std::to_string(getpid()));
    std::filesystem::remove_all(previous, error);
    if (std::filesystem::exists(target)) std::filesystem::rename(target, previous, error);
    std::filesystem::rename(directory, target, error);
    if (error) {
        const auto message = error.message();
        // Put the previous archive back, so a failed build leaves what was there.
        std::error_code ignored;
        if (std::filesystem::exists(previous)) std::filesystem::rename(previous, target, ignored);
        std::filesystem::remove_all(staging, ignored);
        return failure("cannot move the archive into " + target.string() + ": " + message);
    }
    std::filesystem::remove_all(previous, error);
    std::filesystem::remove_all(staging, error);
    result.archive = target / written.filename();
    // A large build writes several archive files (one per chunk of pipelines).
    for (const auto& entry : std::filesystem::directory_iterator(target, error)) {
        if (entry.path().extension() != ".marchive") continue;
        result.archive_bytes += entry.file_size(error);
        ++result.archive_files;
    }
    result.ok = true;
    return result;
}

} // namespace armigo::platform
