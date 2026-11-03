// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "runner_shell.h"

#include <armigo/compat/title_settings.h>
#include <armigo/platform/pso_archive_build.h>
#include <armigo/runtime/shader_chain.h>

#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

namespace armigo::tools {

int run_pso_archive_command(int argc, char* argv[]) {
    platform::PipelineArchiveBuild build;
    std::filesystem::path title_cache;
    build.expected_chain_id = runtime::expected_shader_chain_id();
    for (int index = 2; index < argc; ++index) {
        const std::string_view flag = argv[index];
        if (flag == "--warm-only") {
            build.warm_only = true;
            continue;
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "armigo pso-archive: %s needs a value\n", argv[index]);
            return 2;
        }
        const char* value = argv[++index];
        if (flag == "--shaders") build.package = value;
        else if (flag == "--manifest") build.manifests.emplace_back(value);
        else if (flag == "--cache") build.cache_root = value;
        else if (flag == "--title") {
            const auto title = compat::parse_title_id_text(value);
            if (!title) {
                std::fprintf(stderr, "armigo pso-archive: --title takes 16 uppercase hexadecimal digits\n");
                return 2;
            }
            title_cache = compat::title_cache_directory(*title);
        } else if (flag == "--threads") build.threads = static_cast<unsigned>(std::strtoul(value, nullptr, 10));
        else {
            std::fprintf(stderr, "armigo pso-archive: unknown option %s\n", argv[index - 1]);
            return 2;
        }
    }
    if (build.package.empty() || build.manifests.empty()) {
        std::fputs("Usage: armigo pso-archive --shaders DIR --manifest FILE... [--title ID | --cache DIR] "
                   "[--threads N] [--warm-only]\n", stderr);
        return 2;
    }
    if (build.cache_root.empty()) {
        const char* cache = std::getenv("ARMIGO_PSO_CACHE");
        if (cache && *cache) {
            if (std::string_view(cache) == "off") {
                std::fputs("armigo pso-archive: ARMIGO_PSO_CACHE is off, so the driver reads no archive; "
                           "give --cache DIR\n", stderr);
                return 2;
            }
            build.cache_root = cache;
        } else if (!title_cache.empty()) build.cache_root = title_cache / "pso";
        else {
            std::fputs("armigo pso-archive: name the title (--title) or the cache directory (--cache)\n",
                       stderr);
            return 2;
        }
    }
    @autoreleasepool {
        const auto result = platform::build_pipeline_archive(build);
        const auto escape = [](const std::string& text) {
            std::string out;
            for (const char c : text) {
                if (c == '"' || c == '\\') out += '\\';
                if (static_cast<unsigned char>(c) < 0x20) { out += ' '; continue; }
                out += c;
            }
            return out;
        };
        std::printf("{\"ok\":%s,\"rows\":%llu,\"built\":%llu,\"unresolved\":%llu,\"undecodable\":%llu,"
                    "\"failed\":%llu,\"seconds\":%.1f,\"archive_bytes\":%llu,\"archive_files\":%llu,"
                    "\"archive\":\"%s\",\"error\":\"%s\"}\n",
                    result.ok ? "true" : "false", (unsigned long long)result.rows, (unsigned long long)result.built,
                    (unsigned long long)result.unresolved, (unsigned long long)result.undecodable,
                    (unsigned long long)result.failed, result.seconds, (unsigned long long)result.archive_bytes,
                    (unsigned long long)result.archive_files, escape(result.archive.string()).c_str(),
                    escape(result.error).c_str());
        return result.ok ? 0 : 1;
    }
}

} // namespace armigo::tools
