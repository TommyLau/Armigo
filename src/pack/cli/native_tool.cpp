// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "native_shaders.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {
void print_counts(const char* label, const std::map<std::string, std::uint64_t>& counts) {
    std::cout << label;
    bool first = true;
    for (const auto& [key, value] : counts) {
        if (!value) continue;
        std::cout << (first ? ' ' : ',') << key << '=' << value;
        first = false;
    }
    if (first) std::cout << " none";
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv) {
    armigo::pack::NativeShaderRequest request;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        auto take = [&](std::filesystem::path& destination) -> bool {
            if (index + 1 >= argc) return false;
            destination = argv[++index];
            return true;
        };
        if (argument == "--nup") {
            if (!take(request.nup)) break;
        } else if (argument == "--output") {
            if (!take(request.output)) break;
        } else if (argument == "--translate-tool") {
            if (!take(request.translate_tool)) break;
        } else if (argument == "--compile-cache") {
            if (!take(request.compile_cache)) break;
        } else if (argument == "--dictionary") {
            if (index + 1 >= argc) break;
            request.dictionary_member = argv[++index];
        } else if (argument == "--member") {
            if (index + 1 >= argc) break;
            request.members.emplace_back(argv[++index]);
        } else if (argument == "--identity") {
            if (index + 1 >= argc) {
                std::cerr << "--identity requires a stage identity\n";
                return 2;
            }
            const std::string identity = argv[++index];
            if (identity.size() != 64 || identity.find_first_not_of("0123456789abcdef") != std::string::npos) {
                std::cerr << "--identity requires 64 lowercase hex digits\n";
                return 2;
            }
            request.identities.push_back(identity);
        } else if (argument == "--jobs") {
            if (index + 1 >= argc) break;
            request.jobs = static_cast<unsigned>(std::atoi(argv[++index]));
        } else if (argument == "--enumerate-only") {
            request.enumerate_only = true;
        } else if (argument == "--translate-only") {
            request.translate_only = true;
        } else if (argument == "--no-sources") {
            request.keep_sources = false;
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            return 2;
        }
    }
    if (request.enumerate_only && request.translate_only) {
        std::cerr << "--enumerate-only and --translate-only cannot be combined\n";
        return 2;
    }
    if (request.nup.empty() || request.output.empty() || request.translate_tool.empty() || request.jobs < 1 || request.jobs > 8) {
        std::cerr << "Usage: armigo-native-shader-package --nup FILE --output DIR --translate-tool PATH "
                     "[--dictionary MEMBER] [--member PATH]... [--identity HEX]... [--jobs N] [--compile-cache DIR] "
                     "[--enumerate-only] [--translate-only] [--no-sources]\n";
        return 2;
    }
    auto built = armigo::pack::build_native_shader_package(request);
    if (const auto* failure = std::get_if<std::string>(&built)) {
        std::cerr << "Native shader package failed: " << *failure << '\n';
        return 1;
    }
    const auto& report = std::get<armigo::pack::NativeShaderReport>(built);
    std::cout << "Native shader package: enumerated " << report.enumerated << ", unique " << report.unique
              << ", translated " << report.translated << ", compiled " << report.compiled
              << ", duplicates " << report.duplicates << '\n';
    print_counts("Stages:", report.by_stage);
    print_counts("Translated:", report.translated_by_stage);
    print_counts("Compiled:", report.compiled_by_stage);
    print_counts("Refusals:", report.refusals);
    return 0;
}
