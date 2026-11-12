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
#include "settings.h"

#include <armigo/loader/nup.h>

#include <mach-o/dyld.h>

#include <array>
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
std::filesystem::path current_executable() {
    std::uint32_t size = 4096;
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        buffer.assign(size, '\0');
        if (_NSGetExecutablePath(buffer.data(), &size) != 0) return {};
    }
    std::error_code error;
    auto path = std::filesystem::weakly_canonical(buffer.c_str(), error);
    return error ? std::filesystem::path(buffer.c_str()) : path;
}

// Without an explicit selection, the title's settings give it: the archive's
// title and main-module build select the profile, as a pack does. A title
// without settings is read whole, with no dictionary.
void apply_title_inputs(armigo::pack::NativeShaderRequest& request) {
    namespace ld = armigo::loader;
    auto opened = ld::NupArchive::open(request.nup);
    if (std::holds_alternative<ld::NupFailure>(opened)) return;
    const auto archive = std::get<std::shared_ptr<ld::NupArchive>>(std::move(opened));
    auto selected = archive->select();
    if (std::holds_alternative<ld::NupFailure>(selected)) return;
    const auto selection = std::get<ld::NupSelection>(std::move(selected));
    auto read = archive->read_exefs(selection);
    if (std::holds_alternative<ld::NupFailure>(read)) return;
    const auto& exefs = std::get<ld::ExeFsDirectoryInput>(read);
    std::array<std::byte, 32> build_id{};
    bool found = false;
    for (const auto& module : exefs.modules) {
        if (module.name != "main" || module.bytes.size() < 0x60) continue;
        std::copy_n(module.bytes.begin() + 0x40, build_id.size(), build_id.begin());
        found = true;
    }
    if (!found) return;
    auto match = armigo::pack::match_title_build(selection.title_id, build_id, current_executable());
    if (const auto* failure = std::get_if<std::string>(&match)) {
        std::cerr << "Native shader title settings: " << *failure << " Reading every member without a dictionary.\n";
        return;
    }
    const auto& matched = std::get<armigo::pack::TitleMatch>(match);
    const auto& inputs = matched.description().shader_inputs;
    if (inputs.containers != armigo::compat::ShaderContainerSet::Native) return;
    request.dictionary_member = inputs.dictionary_member;
    request.members = inputs.members;
    request.identities = inputs.identities;
    std::cerr << "Native shader title settings: " << matched.folder.string() << '\n';
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
        } else if (argument == "--source-dir") {
            if (index + 1 >= argc) break;
            request.source_directory = argv[++index];
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
                     "[--enumerate-only] [--translate-only] [--no-sources] [--source-dir DIR]\n";
        return 2;
    }
    if (request.dictionary_member.empty() && request.members.empty() && request.identities.empty())
        apply_title_inputs(request);
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
