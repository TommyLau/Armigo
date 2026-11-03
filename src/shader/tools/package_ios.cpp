// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/shader/ios_package.h>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace {

namespace fs = std::filesystem;

constexpr std::string_view usage =
    "Usage: armigo-shader-package-ios --input MACOS_PACKAGE --output IOS_PACKAGE\n"
    "  [--compile-cache DIR] [--jobs 1..8]\n";

unsigned jobs(std::string_view text) {
    unsigned value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        value < 1 || value > 8) {
        throw std::runtime_error("--jobs must be between 1 and 8");
    }
    return value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        armigo::shader::package::IosBuildOptions options;
        options.jobs = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
        for (int index = 1; index < argc; ++index) {
            const std::string_view option = argv[index];
            if (option == "--help") {
                std::cout << usage;
                return 0;
            }
            if (index + 1 == argc) throw std::runtime_error("Missing value for " + std::string(option));
            const std::string_view value = argv[++index];
            if (value.empty()) throw std::runtime_error("Empty option value");
            if (option == "--input") {
                if (!options.macos_package.empty()) throw std::runtime_error("Duplicate --input");
                options.macos_package = value;
            } else if (option == "--output") {
                if (!options.output.empty()) throw std::runtime_error("Duplicate --output");
                options.output = value;
            } else if (option == "--compile-cache") {
                if (options.compile_cache) throw std::runtime_error("Duplicate --compile-cache");
                options.compile_cache = fs::path(value);
            } else if (option == "--jobs") {
                options.jobs = jobs(value);
            } else {
                throw std::runtime_error("Unknown option: " + std::string(option));
            }
        }
        if (options.macos_package.empty() || options.output.empty()) {
            armigo::diag::error("%.*s", static_cast<int>(usage.size()), usage.data());
            return 2;
        }
        auto built = armigo::shader::package::build_ios_package(options);
        if (const auto* error = std::get_if<std::string>(&built)) {
            armigo::diag::error("armigo-shader-package-ios: %s\n", error->c_str());
            return 1;
        }
        std::cout << "Converted " << std::get<armigo::shader::package::IosBuildReport>(built).converted_stages
                  << " shader stages for iOS 26.0 (Metal 4).\n";
        return 0;
    } catch (const std::exception& error) {
        armigo::diag::error("armigo-shader-package-ios: %s\n", error.what());
        return 2;
    }
}
