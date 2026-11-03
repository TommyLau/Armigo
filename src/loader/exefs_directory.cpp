// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/exefs_directory.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <string_view>
#include <system_error>
#include <utility>

namespace armigo::loader {
namespace {

// Admitted module names. "main" is required; the others are collected when
// present. Numbered subsdk collection stops at the first absent slot.
constexpr std::array<std::string_view, 13> kModuleNames{
    "main",    "rtld",   "sdk",    "subsdk0", "subsdk1", "subsdk2", "subsdk3",
    "subsdk4", "subsdk5", "subsdk6", "subsdk7", "subsdk8", "subsdk9"};

// Process metadata is admitted alongside the modules and returned unparsed.
constexpr std::string_view kProcessMetadataName = "main.npdm";

// The required module name.
constexpr std::string_view kMainModuleName = "main";

ExeFsDirectoryResult failure(ExeFsDirectoryError code, std::string detail) {
    return ExeFsDirectoryResult{ExeFsDirectoryFailure{code, std::move(detail)}};
}

// A candidate observed during the sizing pass. It is read only after every
// candidate passed the bounds, so no data is read for a directory that
// fails admission.
struct Candidate {
    std::string name;
    std::uintmax_t size = 0;
    bool is_process_metadata = false;
};

bool is_process_metadata_name(std::string_view name) noexcept {
    return name == kProcessMetadataName;
}

bool is_admitted_module(std::string_view name) noexcept {
    return std::find(kModuleNames.begin(), kModuleNames.end(), name) != kModuleNames.end();
}

std::string size_detail(const std::string& name, std::uintmax_t size, std::uintmax_t limit) {
    return name + ": " + std::to_string(size) + " bytes exceeds the limit of " +
           std::to_string(limit) + " bytes";
}

// Reads the candidate completely and rejects a length that changed between
// sizing and reading: fewer bytes is a truncation, an available extra byte
// is growth. Returns nothing and fills `code` and `detail` on failure.
std::optional<std::vector<std::byte>> read_candidate(const std::filesystem::path& path,
                                                     const Candidate& candidate,
                                                     ExeFsDirectoryError& code,
                                                     std::string& detail) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        code = ExeFsDirectoryError::read_failed;
        detail = candidate.name + ": could not be opened";
        return std::nullopt;
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(candidate.size));
    if (candidate.size != 0) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(candidate.size));
    }
    if (input.bad()) {
        code = ExeFsDirectoryError::read_failed;
        detail = candidate.name + ": read failed";
        return std::nullopt;
    }
    const auto actual = static_cast<std::uintmax_t>(input.gcount());
    if (actual != candidate.size) {
        code = ExeFsDirectoryError::size_changed;
        detail = candidate.name + ": expected " + std::to_string(candidate.size) +
                 " bytes but read " + std::to_string(actual);
        return std::nullopt;
    }
    char extra = 0;
    input.read(&extra, 1);
    if (input.gcount() != 0) {
        code = ExeFsDirectoryError::size_changed;
        detail = candidate.name + ": grew past " + std::to_string(candidate.size) +
                 " bytes while reading";
        return std::nullopt;
    }
    return bytes;
}

ExeFsDirectoryResult collect(const std::filesystem::path& directory,
                             const ExeFsReadLimits& limits) {
    std::error_code error;
    std::vector<Candidate> modules;
    std::optional<Candidate> metadata;
    // Sum of the sizes admitted so far. The invariant
    // `sized_total <= limits.max_total_bytes` makes the aggregate check
    // below a subtraction, so it cannot overflow.
    std::uintmax_t sized_total = 0;
    bool has_main = false;

    // Determine the first absent numbered slot before examining file sizes or
    // opening any candidate. Later subsdk files are not loader inputs.
    std::set<std::string> present;
    for (std::filesystem::directory_iterator entry(directory, error), end; entry != end;
         entry.increment(error)) {
        present.insert(entry->path().filename().generic_string());
    }
    if (error)
        return failure(ExeFsDirectoryError::directory_unavailable,
                       "the directory could not be listed: " + error.message());
    unsigned subsdk_count = 0;
    while (subsdk_count < 10 && present.contains("subsdk" + std::to_string(subsdk_count)))
        ++subsdk_count;
    ExeFsDirectoryInput input;
    for (unsigned slot = limits.collect_all_subsdk ? 10 : subsdk_count; slot < 10; ++slot) {
        const auto name = "subsdk" + std::to_string(slot);
        if (present.contains(name)) input.ignored_modules.push_back(name);
    }

    for (std::filesystem::directory_iterator entry(directory, error), end; entry != end;
         entry.increment(error)) {
        const std::string name = entry->path().filename().generic_string();
        if (std::find(input.ignored_modules.begin(), input.ignored_modules.end(), name) !=
            input.ignored_modules.end()) continue;
        const bool is_metadata = is_process_metadata_name(name);
        if (!is_metadata && !is_admitted_module(name)) {
            // Ordinary entries outside the profile are ignored unseen.
            continue;
        }

        std::error_code status_error;
        if (entry->symlink_status(status_error).type() == std::filesystem::file_type::symlink) {
            return failure(ExeFsDirectoryError::candidate_not_regular,
                           name + " is a symbolic link");
        }
        if (status_error) {
            return failure(ExeFsDirectoryError::candidate_stat_failed,
                           name + ": " + status_error.message());
        }
        if (entry->status(status_error).type() != std::filesystem::file_type::regular) {
            return failure(ExeFsDirectoryError::candidate_not_regular,
                           name + " is not a regular file");
        }
        if (status_error) {
            return failure(ExeFsDirectoryError::candidate_stat_failed,
                           name + ": " + status_error.message());
        }
        const std::uintmax_t size = entry->file_size(status_error);
        if (status_error) {
            return failure(ExeFsDirectoryError::candidate_stat_failed,
                           name + ": " + status_error.message());
        }
        if (size > static_cast<std::uintmax_t>(limits.max_file_bytes) ||
            size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
            return failure(ExeFsDirectoryError::candidate_too_large,
                           size_detail(name, size, limits.max_file_bytes));
        }
        if (size > static_cast<std::uintmax_t>(limits.max_total_bytes) - sized_total) {
            return failure(ExeFsDirectoryError::total_too_large,
                           size_detail(name, size, limits.max_total_bytes));
        }

        sized_total += size;
        has_main = has_main || (!is_metadata && name == kMainModuleName);
        Candidate candidate{name, size, is_metadata};
        if (is_metadata) {
            metadata = std::move(candidate);
        } else {
            modules.push_back(std::move(candidate));
        }
    }
    if (error) {
        return failure(ExeFsDirectoryError::directory_unavailable,
                       "the directory could not be listed: " + error.message());
    }
    if (!has_main) {
        return failure(ExeFsDirectoryError::main_module_missing,
                       "the required module \"main\" is not present");
    }

    // Byte-wise name order. Names in a directory are unique, so the order
    // is total and deterministic regardless of enumeration order.
    std::sort(modules.begin(), modules.end(),
              [](const Candidate& lhs, const Candidate& rhs) { return lhs.name < rhs.name; });

    input.modules.reserve(modules.size());
    for (const Candidate& module : modules) {
        ExeFsDirectoryError code = ExeFsDirectoryError::read_failed;
        std::string detail;
        std::optional<std::vector<std::byte>> bytes =
            read_candidate(directory / module.name, module, code, detail);
        if (!bytes) {
            return failure(code, std::move(detail));
        }
        ExeFsFile file;
        file.name = module.name;
        file.bytes = std::move(*bytes);
        input.modules.push_back(std::move(file));
    }
    if (metadata) {
        ExeFsDirectoryError code = ExeFsDirectoryError::read_failed;
        std::string detail;
        std::optional<std::vector<std::byte>> bytes =
            read_candidate(directory / metadata->name, *metadata, code, detail);
        if (!bytes) {
            return failure(code, std::move(detail));
        }
        input.process_metadata = std::move(*bytes);
    }
    return ExeFsDirectoryResult{std::move(input)};
}

} // namespace

const char* to_string(ExeFsDirectoryError error) noexcept {
    switch (error) {
    case ExeFsDirectoryError::directory_unavailable:
        return "ExeFS directory could not be listed";
    case ExeFsDirectoryError::main_module_missing:
        return "required ExeFS module \"main\" is missing";
    case ExeFsDirectoryError::candidate_not_regular:
        return "ExeFS candidate is not a regular file";
    case ExeFsDirectoryError::candidate_stat_failed:
        return "ExeFS candidate could not be examined";
    case ExeFsDirectoryError::candidate_too_large:
        return "ExeFS candidate exceeds the per-file size limit";
    case ExeFsDirectoryError::total_too_large:
        return "ExeFS candidates exceed the aggregate size limit";
    case ExeFsDirectoryError::size_changed:
        return "ExeFS candidate length changed while reading";
    case ExeFsDirectoryError::read_failed:
        return "ExeFS candidate could not be read";
    case ExeFsDirectoryError::allocation_failed:
        return "host memory allocation failed";
    }
    return "unknown ExeFS directory error";
}

ExeFsDirectoryResult read_exefs_directory(const std::filesystem::path& directory,
                                          const ExeFsReadLimits& limits) {
    try {
        return collect(directory, limits);
    } catch (const std::bad_alloc&) {
        return failure(ExeFsDirectoryError::allocation_failed,
                       "host memory ran out while collecting the directory");
    } catch (const std::length_error&) {
        return failure(ExeFsDirectoryError::allocation_failed,
                       "a collected size exceeded the host container limit");
    }
}

} // namespace armigo::loader
