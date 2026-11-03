// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/module_set.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <new>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace {

void usage() {
    std::puts("Usage: armigo-module-check [--] FILE...\n"
              "Validate, map, link, and seal an ordered set of external NSO files.\n"
              "No module code or initializers are executed. No files are written.\n"
              "Inputs must be lawfully owned or authorized. See DISCLAIMER.md.");
}

bool read_input(const char* path, std::size_t ordinal, std::uint64_t limit,
                std::vector<std::byte>& bytes) {
    // Opening nonblocking before checking the descriptor prevents a FIFO
    // from hanging before file-type and size validation.
    const int descriptor = ::open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (descriptor < 0) {
        std::fprintf(stderr, "Input %zu: cannot open the file.\n", ordinal);
        return false;
    }
    struct CloseFile {
        int descriptor;
        ~CloseFile() { ::close(descriptor); }
    } close_file{descriptor};
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
        std::fprintf(stderr, "Input %zu: a regular file is required.\n", ordinal);
        return false;
    }
    if (status.st_size < 0 || static_cast<std::uint64_t>(status.st_size) > limit) {
        std::fprintf(stderr, "Input %zu: file size is unavailable or exceeds the input limit.\n",
                     ordinal);
        return false;
    }
    bytes.resize(static_cast<std::size_t>(status.st_size));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto received = ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) {
            std::fprintf(stderr, "Input %zu: read failed or the file size changed.\n", ordinal);
            return false;
        }
        offset += static_cast<std::size_t>(received);
    }
    std::byte extra;
    ssize_t received;
    do {
        received = ::read(descriptor, &extra, 1);
    } while (received < 0 && errno == EINTR);
    if (received == 0) return true;
    std::fprintf(stderr, "Input %zu: read failed or the file size changed.\n", ordinal);
    return false;
}

int check(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        usage();
        return 0;
    }
    const int first = argc > 1 && std::string_view(argv[1]) == "--" ? 2 : 1;
    const armigo::runtime::ModuleLoadOptions options;
    if (first >= argc || argc - first > static_cast<int>(options.link_limits.max_modules)) {
        usage();
        return 2;
    }
    for (int index = first; index < argc; ++index) {
        if (first == 1 && std::string_view(argv[index]).starts_with("-")) {
            std::fputs("Unknown option. Use -- before filenames that begin with a dash.\n", stderr);
            return 2;
        }
    }

    const auto count = static_cast<std::size_t>(argc - first);
    std::vector<std::vector<std::byte>> inputs(count);
    std::vector<std::string> names;
    std::vector<armigo::runtime::NsoSource> sources;
    names.reserve(count);
    sources.reserve(count);
    constexpr std::uint64_t max_total_input = 2ull * 1024 * 1024 * 1024;
    std::uint64_t remaining = max_total_input;
    for (std::size_t index = 0; index < count; ++index) {
        const auto limit = std::min(remaining, options.reader.limits.max_input_bytes);
        if (!read_input(argv[first + index], index + 1, limit, inputs[index])) return 1;
        remaining -= inputs[index].size();
        names.push_back("module-" + std::to_string(index + 1));
        sources.push_back({names.back(), inputs[index]});
    }

    const auto loaded = armigo::runtime::ModuleSet::load_nso(sources, options);
    if (!loaded.ok()) {
        std::fprintf(stderr, "Module-set validation failed (error %u)",
                     static_cast<unsigned>(loaded.error));
        if (loaded.module_index) std::fprintf(stderr, " at input %zu", *loaded.module_index + 1);
        std::fprintf(stderr, ": %s\n", loaded.message.c_str());
        if (loaded.unresolved_count != 0) {
            std::fprintf(stderr, "Unresolved strong references: %zu.\n", loaded.unresolved_count);
        }
        return 1;
    }
    std::printf("Validated %zu modules and %zu relocation records; no module code executed.\n",
                loaded.modules->module_count(), loaded.modules->relocation_count());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return check(argc, argv);
    } catch (const std::bad_alloc&) {
        std::fputs("Module-set validation failed: allocation limit reached.\n", stderr);
    } catch (const std::length_error&) {
        std::fputs("Module-set validation failed: container capacity exceeded.\n", stderr);
    }
    return 1;
}
