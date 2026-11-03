// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The ExeFS directory collector must transport exactly the admitted files
// of a synthetic directory, in deterministic name order, into owned bytes,
// while ignoring unrelated entries, rejecting non-regular candidates, and
// enforcing the read limits. Fixtures are independently authored arbitrary
// pattern bytes; none of them are valid NSO containers or NPDM metadata,
// which this boundary does not claim to validate.

#include <armigo/loader/exefs_directory.h>

#include "test_support.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace fs = std::filesystem;

using armigo::loader::ExeFsDirectoryError;
using armigo::loader::ExeFsDirectoryFailure;
using armigo::loader::ExeFsDirectoryInput;
using armigo::loader::ExeFsDirectoryResult;
using armigo::loader::ExeFsFile;
using armigo::loader::ExeFsReadLimits;
using armigo::loader::read_exefs_directory;

// Independently authored arbitrary bytes: a position-and-seed pattern that
// is not a valid container of any kind.
std::vector<std::byte> pattern(std::size_t size, unsigned seed) {
    std::vector<std::byte> bytes(size);
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<std::byte>((i * 7 + seed * 31 + 5) & 0xFF);
    }
    return bytes;
}

// A private scratch root that is removed again when the test ends.
class ScratchRoot {
public:
    ScratchRoot()
        : root_(fs::temp_directory_path() / "armigo-exefs-directory-tests") {
        fs::remove_all(root_);
        fs::create_directories(root_);
    }
    ~ScratchRoot() {
        std::error_code ignored;
        fs::remove_all(root_, ignored);
    }
    ScratchRoot(const ScratchRoot&) = delete;
    ScratchRoot& operator=(const ScratchRoot&) = delete;

    const fs::path& path() const noexcept { return root_; }

private:
    fs::path root_;
};

void write_file(const fs::path& path, const std::vector<std::byte>& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    ARMIGO_CHECK(out.good());
}

// Reads a host file back for comparison with the collected bytes.
std::vector<std::byte> read_file_back(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    ARMIGO_CHECK(in.good());
    const auto size = static_cast<std::size_t>(in.tellg());
    std::vector<std::byte> bytes(size);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    ARMIGO_CHECK(in.gcount() == static_cast<std::streamsize>(size));
    return bytes;
}

const ExeFsDirectoryFailure& as_failure(const ExeFsDirectoryResult& result) {
    static const ExeFsDirectoryFailure none{};
    if (std::holds_alternative<ExeFsDirectoryFailure>(result)) {
        return std::get<ExeFsDirectoryFailure>(result);
    }
    ARMIGO_CHECK(false && "expected a failure, got an input");
    return none;
}

const ExeFsFile* find_module(const ExeFsDirectoryInput& input, std::string_view name) {
    for (const ExeFsFile& file : input.modules) {
        if (file.name == name) {
            return &file;
        }
    }
    return nullptr;
}

// The full admitted roster collected in deterministic name order with owned
// bytes, while unrelated entries are ignored without disturbing the result.
void full_roster_sorted_and_owned() {
    ScratchRoot root;
    // Created deliberately out of name order.
    const std::pair<std::string, std::size_t> plan[] = {
        {"subsdk9", 5}, {"rtld", 7}, {"main", 64}, {"sdk", 3},  {"subsdk3", 2},
        {"subsdk0", 9}, {"subsdk1", 1}, {"subsdk2", 4}, {"subsdk4", 6},
        {"subsdk5", 8}, {"subsdk6", 2}, {"subsdk7", 3}, {"subsdk8", 1},
    };
    std::map<std::string, std::vector<std::byte>> written;
    unsigned seed = 0;
    for (const auto& [name, size] : plan) {
        const std::vector<std::byte> bytes = pattern(size, seed++);
        write_file(root.path() / name, bytes);
        written.emplace(name, bytes);
    }
    const std::vector<std::byte> metadata = pattern(40, 90);
    write_file(root.path() / "main.npdm", metadata);

    // Unrelated entries that must be ignored without being read.
    write_file(root.path() / "unrelated.bin", pattern(11, 91));
    write_file(root.path() / "main.npdm.bak", pattern(3, 92));
    write_file(root.path() / "subsdk10" / "nested.bin", pattern(2, 93));
    std::error_code link_error;
    fs::create_symlink("missing-target", root.path() / "dangling", link_error);
    ARMIGO_CHECK(!link_error);

    const ExeFsDirectoryResult result = read_exefs_directory(root.path());
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(result));
    if (!std::holds_alternative<ExeFsDirectoryInput>(result)) {
        return;
    }
    const ExeFsDirectoryInput& input = std::get<ExeFsDirectoryInput>(result);
    ARMIGO_CHECK(input.modules.size() == 13);
    const char* expected[] = {"main",    "rtld",    "sdk",     "subsdk0", "subsdk1",
                              "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
                              "subsdk7", "subsdk8", "subsdk9"};
    for (std::size_t i = 0; i < input.modules.size() && i < 13; ++i) {
        ARMIGO_CHECK(input.modules[i].name == expected[i]);
    }
    for (const ExeFsFile& file : input.modules) {
        const auto it = written.find(file.name);
        ARMIGO_CHECK(it != written.end());
        ARMIGO_CHECK(it == written.end() || file.bytes == it->second);
    }
    ARMIGO_CHECK(input.process_metadata.has_value());
    ARMIGO_CHECK(*input.process_metadata == metadata);
    ARMIGO_CHECK(find_module(input, "main.npdm") == nullptr);

    // The result owns its bytes: mutating it cannot affect the tree, and a
    // second collection returns the original content again.
    ExeFsDirectoryInput owned = std::get<ExeFsDirectoryInput>(result);
    if (!owned.modules.empty()) {
        owned.modules.front().bytes.assign(1, std::byte{0});
    }
    const ExeFsDirectoryResult second = read_exefs_directory(root.path());
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(second));
    if (std::holds_alternative<ExeFsDirectoryInput>(second)) {
        const ExeFsFile* main_again = find_module(std::get<ExeFsDirectoryInput>(second), "main");
        ARMIGO_CHECK(main_again != nullptr);
        if (main_again != nullptr) {
            ARMIGO_CHECK(main_again->bytes == written.at("main"));
        }
    }
}

// The tree is only read: every file keeps its bytes and the directory keeps
// exactly its original entries.
void inputs_remain_unchanged() {
    ScratchRoot root;
    const std::vector<std::byte> main_bytes = pattern(50, 1);
    const std::vector<std::byte> rtld_bytes = pattern(4, 2);
    const std::vector<std::byte> metadata_bytes = pattern(20, 3);
    write_file(root.path() / "main", main_bytes);
    write_file(root.path() / "rtld", rtld_bytes);
    write_file(root.path() / "main.npdm", metadata_bytes);

    std::map<std::string, std::vector<std::byte>> before;
    std::vector<std::string> entries_before;
    for (const fs::directory_entry& entry : fs::directory_iterator(root.path())) {
        entries_before.push_back(entry.path().filename().generic_string());
        before.emplace(entries_before.back(), read_file_back(entry.path()));
    }

    const ExeFsDirectoryResult result = read_exefs_directory(root.path());
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(result));

    std::size_t seen = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(root.path())) {
        const std::string name = entry.path().filename().generic_string();
        const auto it = before.find(name);
        ARMIGO_CHECK(it != before.end());
        if (it == before.end()) {
            continue;
        }
        ARMIGO_CHECK(it->second == read_file_back(entry.path()));
        ++seen;
    }
    ARMIGO_CHECK(seen == entries_before.size());
}

// Optional modules and metadata are absent-safe; an empty metadata file is
// present-but-empty, distinct from an absent one; an empty "main" is
// admitted because this boundary does not judge content.
void optional_files_are_optional() {
    ScratchRoot root;
    const std::vector<std::byte> main_bytes = pattern(10, 4);
    write_file(root.path() / "main", main_bytes);

    const ExeFsDirectoryResult minimal = read_exefs_directory(root.path());
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(minimal));
    if (std::holds_alternative<ExeFsDirectoryInput>(minimal)) {
        const ExeFsDirectoryInput& input = std::get<ExeFsDirectoryInput>(minimal);
        ARMIGO_CHECK(input.modules.size() == 1);
        ARMIGO_CHECK(!input.modules.empty() && input.modules[0].name == "main");
        ARMIGO_CHECK(input.modules.empty() || input.modules[0].bytes == main_bytes);
        ARMIGO_CHECK(!input.process_metadata.has_value());
    }

    write_file(root.path() / "main.npdm", {});
    const ExeFsDirectoryResult empty_metadata = read_exefs_directory(root.path());
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(empty_metadata));
    if (std::holds_alternative<ExeFsDirectoryInput>(empty_metadata)) {
        const ExeFsDirectoryInput& input = std::get<ExeFsDirectoryInput>(empty_metadata);
        ARMIGO_CHECK(input.process_metadata.has_value());
        ARMIGO_CHECK(input.process_metadata->empty());
    }

    write_file(root.path() / "main", {});
    const ExeFsDirectoryResult empty_main = read_exefs_directory(root.path());
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(empty_main));
    if (std::holds_alternative<ExeFsDirectoryInput>(empty_main)) {
        const ExeFsFile* main_file = find_module(std::get<ExeFsDirectoryInput>(empty_main), "main");
        ARMIGO_CHECK(main_file != nullptr);
        ARMIGO_CHECK(main_file != nullptr && main_file->bytes.empty());
    }
}

// "main" is required; the name match is exact and case-sensitive.
void main_is_required() {
    ScratchRoot root;
    write_file(root.path() / "rtld", pattern(4, 5));
    write_file(root.path() / "subsdk0", pattern(4, 6));
    write_file(root.path() / "main.npdm", pattern(8, 7));
    const ExeFsDirectoryResult without_main = read_exefs_directory(root.path());
    ARMIGO_CHECK(as_failure(without_main).code == ExeFsDirectoryError::main_module_missing);

    ScratchRoot cased;
    // On a case-insensitive host the name differs by case only when "main"
    // itself is absent, as it is here.
    write_file(cased.path() / "Main", pattern(4, 8));
    const ExeFsDirectoryResult wrong_case = read_exefs_directory(cased.path());
    ARMIGO_CHECK(as_failure(wrong_case).code == ExeFsDirectoryError::main_module_missing);
}

// Symbolic-link candidates are rejected even when they point at regular
// files, for required and optional names alike.
void symlink_candidates_rejected() {
    ScratchRoot root;
    write_file(root.path() / "real-main", pattern(16, 9));
    std::error_code link_error;
    fs::create_symlink("real-main", root.path() / "main", link_error);
    ARMIGO_CHECK(!link_error);
    if (link_error) {
        return;
    }
    const ExeFsDirectoryResult linked_main = read_exefs_directory(root.path());
    const ExeFsDirectoryFailure& failure = as_failure(linked_main);
    ARMIGO_CHECK(failure.code == ExeFsDirectoryError::candidate_not_regular);
    ARMIGO_CHECK(failure.detail.find("main") != std::string::npos);

    ScratchRoot optional_linked;
    write_file(optional_linked.path() / "main", pattern(8, 10));
    write_file(optional_linked.path() / "real-rtld", pattern(8, 11));
    std::error_code rtld_error;
    fs::create_symlink("real-rtld", optional_linked.path() / "rtld", rtld_error);
    ARMIGO_CHECK(!rtld_error);
    if (rtld_error) {
        return;
    }
    const ExeFsDirectoryResult linked_rtld = read_exefs_directory(optional_linked.path());
    ARMIGO_CHECK(as_failure(linked_rtld).code == ExeFsDirectoryError::candidate_not_regular);
}

// A directory occupying an admitted name is rejected.
void directory_candidates_rejected() {
    ScratchRoot root;
    write_file(root.path() / "main", pattern(8, 12));
    fs::create_directories(root.path() / "sdk");
    const ExeFsDirectoryResult result = read_exefs_directory(root.path());
    const ExeFsDirectoryFailure& failure = as_failure(result);
    ARMIGO_CHECK(failure.code == ExeFsDirectoryError::candidate_not_regular);
    ARMIGO_CHECK(failure.detail.find("sdk") != std::string::npos);
}

// A candidate above the per-file limit is rejected before reading; a
// candidate exactly at the limit is admitted.
void per_file_limit() {
    ScratchRoot root;
    write_file(root.path() / "main", pattern(17, 13));
    ExeFsReadLimits limits;
    limits.max_file_bytes = 16;
    limits.max_total_bytes = 1 << 20;
    const ExeFsDirectoryResult too_large = read_exefs_directory(root.path(), limits);
    const ExeFsDirectoryFailure& failure = as_failure(too_large);
    ARMIGO_CHECK(failure.code == ExeFsDirectoryError::candidate_too_large);
    ARMIGO_CHECK(failure.detail.find("17") != std::string::npos);

    write_file(root.path() / "main", pattern(16, 13));
    const ExeFsDirectoryResult exact = read_exefs_directory(root.path(), limits);
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(exact));
}

// The aggregate limit covers all admitted files including the metadata; the
// boundary is exact.
void aggregate_limit() {
    ScratchRoot root;
    write_file(root.path() / "main", pattern(10, 14));
    write_file(root.path() / "rtld", pattern(10, 15));
    ExeFsReadLimits limits;
    limits.max_file_bytes = 64;
    limits.max_total_bytes = 20;
    // Modules exactly at the aggregate limit are admitted.
    const ExeFsDirectoryResult modules_at_limit =
        read_exefs_directory(root.path(), limits);
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(modules_at_limit));

    // The metadata counts toward the same aggregate.
    write_file(root.path() / "main.npdm", pattern(1, 16));
    const ExeFsDirectoryResult metadata_over = read_exefs_directory(root.path(), limits);
    ARMIGO_CHECK(as_failure(metadata_over).code == ExeFsDirectoryError::total_too_large);

    write_file(root.path() / "main", pattern(5, 14));
    write_file(root.path() / "rtld", pattern(4, 15));
    const ExeFsDirectoryResult exact = read_exefs_directory(root.path(), limits);
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(exact));
    if (std::holds_alternative<ExeFsDirectoryInput>(exact)) {
        ARMIGO_CHECK(std::get<ExeFsDirectoryInput>(exact).modules.size() == 2);
    }

    // Any limit below the collected total fails, whatever the order.
    limits.max_total_bytes = 9;
    const ExeFsDirectoryResult over = read_exefs_directory(root.path(), limits);
    ARMIGO_CHECK(as_failure(over).code == ExeFsDirectoryError::total_too_large);
}

// A missing path and a path that names a regular file are both refused.
void unavailable_directories() {
    ScratchRoot root;
    const ExeFsDirectoryResult missing =
        read_exefs_directory(root.path() / "does-not-exist");
    ARMIGO_CHECK(as_failure(missing).code == ExeFsDirectoryError::directory_unavailable);

    write_file(root.path() / "plain-file", pattern(4, 17));
    const ExeFsDirectoryResult not_a_directory =
        read_exefs_directory(root.path() / "plain-file");
    ARMIGO_CHECK(as_failure(not_a_directory).code ==
                 ExeFsDirectoryError::directory_unavailable);
}

// An unreadable ordinary file outside the profile must not disturb the
// collection, because unrelated entries are never opened. When the tests
// run with privileges that can read such a file anyway, the check weakens
// to the plain successful collection.
void unreadable_unrelated_file_is_ignored() {
    ScratchRoot root;
    write_file(root.path() / "main", pattern(8, 18));
    write_file(root.path() / "locked.bin", pattern(32, 19));
    std::error_code permission_error;
    fs::permissions(root.path() / "locked.bin", fs::perms::none, permission_error);
    ARMIGO_CHECK(!permission_error);

    const ExeFsDirectoryResult result = read_exefs_directory(root.path());
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(result));
    if (std::holds_alternative<ExeFsDirectoryInput>(result)) {
        ARMIGO_CHECK(find_module(std::get<ExeFsDirectoryInput>(result), "main") != nullptr);
    }

    fs::permissions(root.path() / "locked.bin", fs::perms::owner_read, permission_error);
    ARMIGO_CHECK(!permission_error);
}

void orphan_subsdk_files_are_not_opened_or_sized() {
    ScratchRoot root;
    write_file(root.path() / "main", pattern(4, 1));
    write_file(root.path() / "sdk", pattern(4, 2));
    // A dangling symlink and an oversized file are both outside admission.
    fs::create_symlink("absent", root.path() / "subsdk1");
    write_file(root.path() / "subsdk9", pattern(128, 3));
    const auto result = read_exefs_directory(root.path(),
        {.max_file_bytes = 4, .max_total_bytes = 8});
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(result));
    if (const auto* input = std::get_if<ExeFsDirectoryInput>(&result)) {
        ARMIGO_CHECK(input->modules.size() == 2);
        ARMIGO_CHECK(input->ignored_modules == std::vector<std::string>({"subsdk1", "subsdk9"}));
    }
    fs::remove(root.path() / "subsdk1");
    write_file(root.path() / "subsdk1", pattern(4, 4));
    const auto synthetic = read_exefs_directory(root.path(), {.collect_all_subsdk = true});
    ARMIGO_CHECK(std::holds_alternative<ExeFsDirectoryInput>(synthetic));
    if (const auto* input = std::get_if<ExeFsDirectoryInput>(&synthetic)) {
        ARMIGO_CHECK(input->modules.size() == 4);
        ARMIGO_CHECK(input->ignored_modules.empty());
    }
}

// Default limits and error text.
void defaults_and_diagnostics() {
    const ExeFsReadLimits defaults;
    ARMIGO_CHECK(defaults.max_file_bytes == std::size_t{256} * 1024 * 1024);
    ARMIGO_CHECK(defaults.max_total_bytes == std::size_t{512} * 1024 * 1024);

    for (unsigned raw = 0; raw <= static_cast<unsigned>(
                                        ExeFsDirectoryError::allocation_failed);
         ++raw) {
        const auto error = static_cast<ExeFsDirectoryError>(raw);
        ARMIGO_CHECK(armigo::loader::to_string(error) != nullptr);
    }
}

} // namespace

int main() {
    full_roster_sorted_and_owned();
    inputs_remain_unchanged();
    optional_files_are_optional();
    main_is_required();
    symlink_candidates_rejected();
    directory_candidates_rejected();
    per_file_limit();
    aggregate_limit();
    unavailable_directories();
    unreadable_unrelated_file_is_ignored();
    orphan_subsdk_files_are_not_opened_or_sized();
    defaults_and_diagnostics();
    return armigo::loader::test::finish("loader.exefs_directory");
}
