// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/loader/exefs_directory.h>
#include <array>
#include <map>
#include <memory>
#include <span>
#include <set>

namespace armigo::loader {

struct NupFailure {
    std::string operation;
    std::string detail;
};

struct NupFile {
    std::uint64_t size = 0;
    std::uint64_t offset = 0;
    std::uint64_t stored_size = 0;
    std::array<std::byte, 32> hash{};
    std::uint8_t codec = 0;
};

struct NupPackage {
    std::uint64_t title_id = 0;
    std::uint32_t version = 0;
    std::uint8_t kind = 0;
    bool standalone = false;
    std::uint32_t root = 0;
    std::uint32_t exefs = ~std::uint32_t{0};
    std::uint32_t romfs = ~std::uint32_t{0};
    std::vector<std::byte> npdm;
    // The package's display version (metadata type 4), empty when absent.
    std::string display_version;
};

struct NupSelection {
    std::uint64_t title_id = 0;
    std::uint32_t version = 0;
    bool update_applied = false;
    // The selected package's display version, empty when the package has none.
    std::string display_version;
    // Paths are relative, slash-separated and validated. Payloads remain in the archive.
    std::map<std::string, std::uint32_t> exefs;
    std::map<std::string, std::uint32_t> romfs;
    std::set<std::string> romfs_directories;
    std::vector<std::byte> npdm;
};

// Opening verifies the bounded TOC, not all payloads. Hashes detect corruption,
// not authenticity. The caller must keep the opened file unchanged while in use.
class NupArchive {
public:
    static std::variant<NupFailure, std::shared_ptr<NupArchive>> open(
        const std::filesystem::path& path);
    ~NupArchive();
    NupArchive(const NupArchive&) = delete;
    NupArchive& operator=(const NupArchive&) = delete;
    const std::vector<NupPackage>& packages() const noexcept { return packages_; }
    const std::vector<NupFile>& files() const noexcept { return files_; }
    // Zero selects the sole base application; multiple bases require an explicit ID.
    std::variant<NupFailure, NupSelection> select(std::uint64_t title_id = 0) const;
    // Reads a raw extent without allocating its complete contents or verifying its hash.
    std::variant<NupFailure, std::size_t> read(
        std::uint32_t file, std::uint64_t offset, std::span<std::byte> output) const;
    std::optional<NupFailure> verify(std::uint32_t file) const;
    // Nonempty selected RomFS members whose final component ends in ".zs" must
    // start with a zstd frame, standard or skippable. Empty members are left
    // out. The result lists the mismatch paths. A member that cannot be read
    // fails the call; a matching member hash does not prove the frame.
    std::variant<NupFailure, std::vector<std::string>> zs_frame_mismatches(
        const NupSelection& selection) const;
    // Collects and verifies only selected executable members, under the usual limits.
    std::variant<NupFailure, ExeFsDirectoryInput> read_exefs(
        const NupSelection& selection, const ExeFsReadLimits& limits = {}) const;

private:
    NupArchive() = default;
    struct Directory { std::string name; std::vector<std::uint32_t> children; };
    std::optional<NupFailure> parse();
    void collect(std::uint32_t directory, const std::string& prefix,
                 std::map<std::string, std::uint32_t>& output) const;
    int descriptor_ = -1;
    std::uint64_t size_ = 0;
    std::vector<Directory> directories_;
    std::vector<std::string> names_;
    std::vector<NupFile> files_;
    std::vector<NupPackage> packages_;
};

} // namespace armigo::loader
