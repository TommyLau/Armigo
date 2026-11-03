// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Program-container extraction: locates the program NCA of an NSP (PFS0) or
// XCI (gamecard with an HFS0 root partition), verifies its signatures and
// content hash chains, and extracts the ExeFS and RomFS trees below a
// caller-owned output directory. With an update container, the update's
// program NCA (matched by title id, titlekey decryption through a common
// ticket or the user's rights-id table) supplies the ExeFS, and its
// BKTR-patched RomFS layers over the base program NCA's RomFS. Written for
// Armigo from the PFS0/HFS0/XCI/IVFC/BKTR on-disk formats.

#include <armigo/loader/container.h>

#include "container_nca.h"
#include "container_scan.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace armigo::loader {
namespace {

constexpr std::uint32_t kPfs0Magic = 0x30534650; // "PFS0"
constexpr std::uint32_t kHfs0Magic = 0x30534648; // "HFS0"
constexpr std::uint32_t kHeadMagic = 0x44414548; // "HEAD"

bool read_plain(std::FILE* file, std::uint64_t offset, std::span<std::byte> out) noexcept {
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
        return false;
    }
    return std::fread(out.data(), 1, out.size(), file) == out.size();
}

// One entry of a plaintext PFS0 or HFS0 partition; the shared leading
// fields (offset, size, name offset) have the same layout in both.
struct PlainEntry {
    std::string name;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

struct PlainPartition {
    std::uint64_t data_offset = 0; // absolute file offset of the first data byte
    std::vector<PlainEntry> entries;
};

// Reads a PFS0 or HFS0 partition table at `base` from plaintext file data.
[[nodiscard]] bool read_plain_partition(std::FILE* file, std::uint64_t base, bool hfs0,
                                        PlainPartition& partition,
                                        ContainerFailure& failure) {
    std::array<std::byte, 0x10> header{};
    if (!read_plain(file, base, header)) {
        failure = {ContainerError::partition_invalid,
                   "Reading the partition header failed; the container may be truncated."};
        return false;
    }
    const std::uint32_t magic = container_detail::read_u32_le(header, 0);
    if (magic != (hfs0 ? kHfs0Magic : kPfs0Magic)) {
        failure = {ContainerError::partition_invalid,
                   "The partition table has an unexpected magic."};
        return false;
    }
    const std::uint32_t file_count = container_detail::read_u32_le(header, 4);
    const std::uint32_t string_table_size = container_detail::read_u32_le(header, 8);
    const std::size_t entry_size = hfs0 ? 0x40 : 0x18;
    // Gamecard partitions may be empty (an unused normal or logo partition).
    if (file_count > 0x10000 || string_table_size > (1u << 24)) {
        failure = {ContainerError::partition_invalid,
                   "The partition table header is malformed."};
        return false;
    }
    std::vector<std::byte> entries(std::size_t{entry_size} * file_count);
    if (!read_plain(file, base + 0x10, entries)) {
        failure = {ContainerError::partition_invalid, "Reading the partition entries failed."};
        return false;
    }
    std::vector<std::byte> string_table(string_table_size);
    if (!read_plain(file, base + 0x10 + entries.size(), string_table)) {
        failure = {ContainerError::partition_invalid, "Reading the partition names failed."};
        return false;
    }
    partition.data_offset = base + 0x10 + entries.size() + string_table.size();
    partition.entries.reserve(file_count);
    for (std::uint32_t i = 0; i < file_count; ++i) {
        const std::size_t entry = std::size_t{i} * entry_size;
        PlainEntry item;
        item.offset = container_detail::read_u64_le(entries, entry);
        item.size = container_detail::read_u64_le(entries, entry + 8);
        const std::uint32_t name_offset = container_detail::read_u32_le(entries, entry + 0x10);
        if (name_offset >= string_table.size()) {
            failure = {ContainerError::partition_invalid,
                       "A partition entry name leaves the name table."};
            return false;
        }
        const auto* begin = reinterpret_cast<const char*>(string_table.data()) + name_offset;
        const auto* end =
            reinterpret_cast<const char*>(string_table.data()) + string_table.size();
        const auto* terminator =
            static_cast<const char*>(std::memchr(begin, '\0', static_cast<std::size_t>(end - begin)));
        item.name.assign(begin, terminator ? terminator : end);
        if (item.name.empty() || item.name.find('/') != std::string::npos ||
            item.name.find('\\') != std::string::npos || item.name == "." ||
            item.name == "..") {
            failure = {ContainerError::partition_invalid,
                       "A partition entry has an unusable name."};
            return false;
        }
        partition.entries.push_back(std::move(item));
    }
    return true;
}

[[nodiscard]] bool checked_partition_offset(std::uint64_t base, std::uint64_t relative,
                                            std::uint64_t& offset, ContainerFailure& failure) {
    if (relative > ~std::uint64_t{0} - base) {
        failure = {ContainerError::partition_invalid,
                   "A partition entry offset overflows the container address space."};
        return false;
    }
    offset = base + relative;
    return true;
}

// Appends the candidate NCA offsets of one partition to `candidates`, and
// the ticket files' extents to `tickets` when requested.
void collect_partition_candidates(std::FILE* file, std::uint64_t base, bool hfs0,
                                  std::vector<std::uint64_t>& candidates,
                                  std::vector<container_detail::ContainerTicketRef>* tickets,
                                  ContainerFailure& failure, bool& ok) {
    PlainPartition partition;
    if (!read_plain_partition(file, base, hfs0, partition, failure)) {
        ok = false;
        return;
    }
    constexpr std::string_view kTicketSuffix = ".tik";
    for (const PlainEntry& entry : partition.entries) {
        std::uint64_t offset = 0;
        if (!checked_partition_offset(partition.data_offset, entry.offset, offset, failure)) {
            ok = false;
            return;
        }
        candidates.push_back(offset);
        if (tickets != nullptr && entry.name.size() > kTicketSuffix.size() &&
            entry.name.compare(entry.name.size() - kTicketSuffix.size(), kTicketSuffix.size(),
                               kTicketSuffix) == 0) {
            tickets->push_back({offset, entry.size});
        }
    }
    ok = true;
}

// Appends the candidate NCA offsets of the XCI root partition, searching
// the gamecard partitions in the order program content appears in them.
[[nodiscard]] bool collect_xci_candidates(std::FILE* file, std::uint64_t root_offset,
                                          std::vector<std::uint64_t>& candidates,
                                          std::vector<container_detail::ContainerTicketRef>* tickets,
                                          ContainerFailure& failure) {
    PlainPartition root;
    if (!read_plain_partition(file, root_offset, true, root, failure)) {
        return false;
    }
    static constexpr std::string_view kSearchOrder[] = {"secure", "normal", "update"};
    for (const std::string_view wanted : kSearchOrder) {
        const auto match = std::find_if(root.entries.begin(), root.entries.end(),
                                        [&](const PlainEntry& entry) {
                                            return entry.name == wanted;
                                        });
        if (match == root.entries.end()) {
            continue;
        }
        std::uint64_t partition_base = 0;
        if (!checked_partition_offset(root.data_offset, match->offset, partition_base, failure)) {
            return false;
        }
        bool ok = false;
        collect_partition_candidates(file, partition_base, true, candidates, tickets, failure, ok);
        if (!ok) {
            return false;
        }
    }
    return true;
}

// Writes a directory entry and returns false when the host refused.
[[nodiscard]] bool make_directories(const std::filesystem::path& path) noexcept {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        return false;
    }
    std::filesystem::permissions(path, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace, error);
    return !error;
}

// Copies one ExeFS entry from its NCA section to the output directory.
[[nodiscard]] bool extract_exefs_file(const container_detail::SectionView& view,
                                      const container_detail::Pfs0Partition& partition,
                                      std::size_t index, const std::filesystem::path& target,
                                      std::uint64_t max_file_bytes,
                                      ContainerFailure& failure) {
    const std::uint64_t size = partition.entry_size(index);
    if (size > max_file_bytes) {
        failure = {ContainerError::section_invalid,
                   "ExeFS entry '" + std::string(partition.entry_name(index)) +
                       "' exceeds the configured file limit."};
        return false;
    }
    std::ofstream output(target, std::ios::binary | std::ios::trunc);
    if (!output) {
        failure = {ContainerError::io_failure,
                   "Could not create '" + target.string() + "'."};
        return false;
    }
    constexpr std::uint64_t kChunk = 1u << 20;
    std::vector<std::byte> buffer(std::size_t{1} << 20);
    for (std::uint64_t done = 0; done < size;) {
        const std::uint64_t take = std::min(kChunk, size - done);
        if (!container_detail::read_section_view(
                view, partition.entry_file_offset(index) + done,
                std::span<std::byte>(buffer.data(), std::size_t(take)))) {
            failure = {ContainerError::section_invalid,
                       "Reading ExeFS entry '" + std::string(partition.entry_name(index)) +
                           "' failed."};
            return false;
        }
        output.write(reinterpret_cast<const char*>(buffer.data()),
                     static_cast<std::streamsize>(take));
        if (!output) {
            failure = {ContainerError::io_failure,
                       "Writing '" + target.string() + "' failed."};
            return false;
        }
        done += take;
    }
    return true;
}

// One RomFS directory node on the walk stack.
struct RomFsWalk {
    std::filesystem::path directory;
    std::uint32_t child = 0xFFFFFFFF; // first child directory
    std::uint32_t file = 0xFFFFFFFF;  // first file
};

// Reads one RomFS metadata entry field from a meta table image.
[[nodiscard]] bool extract_romfs_tree(const container_detail::SectionView& view,
                                      std::uint64_t romfs_offset,
                                      const std::filesystem::path& target,
                                      ContainerFailure& failure) {
    const container_detail::NcaSection& section = *view.section;
    std::array<std::byte, 0x50> header{};
    if (!container_detail::read_section_view(view, romfs_offset, header) ||
        container_detail::read_u64_le(header, 0) != 0x50) {
        failure = {ContainerError::section_invalid,
                   "The RomFS image header is missing or malformed."};
        return false;
    }
    const std::uint64_t dir_meta_offset = container_detail::read_u64_le(header, 0x18);
    const std::uint64_t dir_meta_size = container_detail::read_u64_le(header, 0x20);
    const std::uint64_t file_meta_offset = container_detail::read_u64_le(header, 0x38);
    const std::uint64_t file_meta_size = container_detail::read_u64_le(header, 0x40);
    const std::uint64_t data_offset = container_detail::read_u64_le(header, 0x48);
    // The hashed level-6 extent from the IVFC superblock bounds the image;
    // its parse-time check keeps it inside the section, so every read below
    // stays within the verified hash chain.
    const std::uint64_t romfs_size = section.superblock.romfs_size;
    if (dir_meta_size == 0 || dir_meta_size > romfs_size ||
        dir_meta_offset > romfs_size - dir_meta_size || file_meta_size > romfs_size ||
        file_meta_offset > romfs_size - file_meta_size || data_offset > romfs_size) {
        failure = {ContainerError::section_invalid,
                   "The RomFS metadata tables leave the RomFS image."};
        return false;
    }
    std::vector<std::byte> directories(dir_meta_size);
    if (!container_detail::read_section_view(view, romfs_offset + dir_meta_offset,
                                             directories)) {
        failure = {ContainerError::section_invalid, "Reading the RomFS directories failed."};
        return false;
    }
    std::vector<std::byte> files(file_meta_size);
    if (!container_detail::read_section_view(view, romfs_offset + file_meta_offset, files)) {
        failure = {ContainerError::section_invalid, "Reading the RomFS file entries failed."};
        return false;
    }

    auto dir_field = [&directories](std::uint32_t entry, std::size_t offset) -> std::uint32_t {
        return container_detail::read_u32_le(directories, entry + offset);
    };
    auto dir_name = [&directories](std::uint32_t entry) -> std::string_view {
        const std::uint32_t name_size = container_detail::read_u32_le(directories, entry + 0x14);
        if (entry + 0x18 + name_size > directories.size()) {
            return {};
        }
        return {reinterpret_cast<const char*>(directories.data()) + entry + 0x18, name_size};
    };
    auto file_field = [&files](std::uint32_t entry, std::size_t offset) -> std::uint32_t {
        return container_detail::read_u32_le(files, entry + offset);
    };
    auto file_name = [&files](std::uint32_t entry) -> std::string_view {
        const std::uint32_t name_size = container_detail::read_u32_le(files, entry + 0x1C);
        if (entry + 0x20 + name_size > files.size()) {
            return {};
        }
        return {reinterpret_cast<const char*>(files.data()) + entry + 0x20, name_size};
    };

    constexpr std::uint32_t kEmpty = 0xFFFFFFFF;
    constexpr std::uint64_t kChunk = 1u << 20;
    std::vector<std::byte> buffer(std::size_t{1} << 20);

    // The walk must cover every directory and file the tables link. Entries
    // never shrink below their fixed part, so the table sizes bound how many
    // distinct entries exist; a well-formed tree links each entry exactly
    // once, so exceeding a bound means a cycle or overlapping entries, not a
    // tree too large to walk. Depth is unbounded: the stack holds pending
    // siblings across levels, which for wide trees grows far past any depth
    // the directory names themselves express.
    const std::uint64_t dir_capacity = directories.size() / 0x18;
    const std::uint64_t file_capacity = files.size() / 0x20;
    std::uint64_t dir_visits = 0;
    std::uint64_t file_visits = 0;

    // The root directory is entry 0 of the directory metadata.
    if (directories.size() < 0x18) {
        failure = {ContainerError::section_invalid,
                   "The RomFS directory metadata has no root entry."};
        return false;
    }
    std::vector<RomFsWalk> stack;
    RomFsWalk root;
    root.directory = target;
    root.child = dir_field(0, 0x08);
    root.file = dir_field(0, 0x0C);
    stack.push_back(root);
    while (!stack.empty()) {
        RomFsWalk current = stack.back();
        stack.pop_back();
        for (std::uint32_t child = current.child; child != kEmpty;) {
            if (std::size_t{child} + 0x18 > directories.size()) {
                failure = {ContainerError::section_invalid,
                           "A RomFS directory entry leaves the metadata table."};
                return false;
            }
            if (++dir_visits > dir_capacity || stack.size() > dir_capacity) {
                failure = {ContainerError::section_invalid,
                           "The RomFS directory graph is malformed."};
                return false;
            }
            const std::string_view name = dir_name(child);
            if (!sanitize_romfs_name(name)) {
                failure = {ContainerError::section_invalid,
                           "A RomFS directory has an unusable name."};
                return false;
            }
            RomFsWalk next;
            next.directory = current.directory / std::filesystem::path(name);
            if (!make_directories(next.directory)) {
                failure = {ContainerError::io_failure,
                           "Could not create '" + next.directory.string() + "'."};
                return false;
            }
            next.child = dir_field(child, 0x08);
            next.file = dir_field(child, 0x0C);
            stack.push_back(next);
            child = dir_field(child, 0x04); // sibling
        }
        for (std::uint32_t entry = current.file; entry != kEmpty;) {
            if (std::size_t{entry} + 0x20 > files.size()) {
                failure = {ContainerError::section_invalid,
                           "A RomFS file entry leaves the metadata table."};
                return false;
            }
            if (++file_visits > file_capacity) {
                failure = {ContainerError::section_invalid,
                           "The RomFS file graph is malformed."};
                return false;
            }
            const std::string_view name = file_name(entry);
            if (!sanitize_romfs_name(name)) {
                failure = {ContainerError::section_invalid,
                           "A RomFS file has an unusable name."};
                return false;
            }
            const std::uint64_t relative = container_detail::read_u64_le(files, entry + 0x08);
            const std::uint64_t size = container_detail::read_u64_le(files, entry + 0x10);
            if (relative > romfs_size - data_offset || size > romfs_size - data_offset - relative) {
                failure = {ContainerError::section_invalid,
                           "A RomFS file extends past the RomFS image."};
                return false;
            }
            const std::filesystem::path target_file =
                current.directory / std::filesystem::path(name);
            std::ofstream output(target_file, std::ios::binary | std::ios::trunc);
            if (!output) {
                failure = {ContainerError::io_failure,
                           "Could not create '" + target_file.string() + "'."};
                return false;
            }
            for (std::uint64_t done = 0; done < size;) {
                const std::uint64_t take = std::min(kChunk, size - done);
                if (!container_detail::read_section_view(
                        view, romfs_offset + data_offset + relative + done,
                        std::span<std::byte>(buffer.data(), std::size_t(take)))) {
                    failure = {ContainerError::section_invalid,
                               "Reading RomFS file '" + std::string(name) + "' failed."};
                    return false;
                }
                output.write(reinterpret_cast<const char*>(buffer.data()),
                             static_cast<std::streamsize>(take));
                if (!output) {
                    failure = {ContainerError::io_failure,
                               "Writing '" + target_file.string() + "' failed."};
                    return false;
                }
                done += take;
            }
            entry = file_field(entry, 0x04); // sibling
        }
    }
    return true;
}

} // namespace

bool sanitize_romfs_name(std::string_view name) noexcept {
    return !name.empty() && name.size() <= 255 && name != "." && name != ".." &&
           name.find('/') == std::string_view::npos &&
           name.find('\\') == std::string_view::npos &&
           name.find('\0') == std::string_view::npos;
}

const char* to_string(ContainerError error) noexcept {
    switch (error) {
        case ContainerError::container_unavailable: return "container_unavailable";
        case ContainerError::key_missing: return "key_missing";
        case ContainerError::unknown_container_format: return "unknown_container_format";
        case ContainerError::partition_invalid: return "partition_invalid";
        case ContainerError::program_nca_missing: return "program_nca_missing";
        case ContainerError::unsupported_nca: return "unsupported_nca";
        case ContainerError::ticket_invalid: return "ticket_invalid";
        case ContainerError::signature_invalid: return "signature_invalid";
        case ContainerError::npdm_invalid: return "npdm_invalid";
        case ContainerError::section_invalid: return "section_invalid";
        case ContainerError::io_failure: return "io_failure";
        case ContainerError::allocation_failed: return "allocation_failed";
        case ContainerError::content_hash_invalid: return "content_hash_invalid";
    }
    return "unknown";
}

ContainerTrustAnchors ContainerTrustAnchors::production() {
    // The fixed public verification keys of the NCA format: public-key
    // constants, not secrets. They identify which private key must have
    // produced a signature and carry no decryption capability.
    static constexpr std::array<std::uint8_t, 256> kNcaHeaderModulus = {
        0xBF, 0xBE, 0x40, 0x6C, 0xF4, 0xA7, 0x80, 0xE9, 0xF0, 0x7D, 0x0C, 0x99, 0x61, 0x1D,
        0x77, 0x2F, 0x96, 0xBC, 0x4B, 0x9E, 0x58, 0x38, 0x1B, 0x03, 0xAB, 0xB1, 0x75, 0x49,
        0x9F, 0x2B, 0x4D, 0x58, 0x34, 0xB0, 0x05, 0xA3, 0x75, 0x22, 0xBE, 0x1A, 0x3F, 0x03,
        0x73, 0xAC, 0x70, 0x68, 0xD1, 0x16, 0xB9, 0x04, 0x46, 0x5E, 0xB7, 0x07, 0x91, 0x2F,
        0x07, 0x8B, 0x26, 0xDE, 0xF6, 0x00, 0x07, 0xB2, 0xB4, 0x51, 0xF8, 0x0D, 0x0A, 0x5E,
        0x58, 0xAD, 0xEB, 0xBC, 0x9A, 0xD6, 0x49, 0xB9, 0x64, 0xEF, 0xA7, 0x82, 0xB5, 0xCF,
        0x6D, 0x70, 0x13, 0xB0, 0x0F, 0x85, 0xF6, 0xA9, 0x08, 0xAA, 0x4D, 0x67, 0x66, 0x87,
        0xFA, 0x89, 0xFF, 0x75, 0x90, 0x18, 0x1E, 0x6B, 0x3D, 0xE9, 0x8A, 0x68, 0xC9, 0x26,
        0x04, 0xD9, 0x80, 0xCE, 0x3F, 0x5E, 0x92, 0xCE, 0x01, 0xFF, 0x06, 0x3B, 0xF2, 0xC1,
        0xA9, 0x0C, 0xCE, 0x02, 0x6F, 0x16, 0xBC, 0x92, 0x42, 0x0A, 0x41, 0x64, 0xCD, 0x52,
        0xB6, 0x34, 0x4D, 0xAE, 0xC0, 0x2E, 0xDE, 0xA4, 0xDF, 0x27, 0x68, 0x3C, 0xC1, 0xA0,
        0x60, 0xAD, 0x43, 0xF3, 0xFC, 0x86, 0xC1, 0x3E, 0x6C, 0x46, 0xF7, 0x7C, 0x29, 0x9F,
        0xFA, 0xFD, 0xF0, 0xE3, 0xCE, 0x64, 0xE7, 0x35, 0xF2, 0xF6, 0x56, 0x56, 0x6F, 0x6D,
        0xF1, 0xE2, 0x42, 0xB0, 0x83, 0x40, 0xA5, 0xC3, 0x20, 0x2B, 0xCC, 0x9A, 0xAE, 0xCA,
        0xED, 0x4D, 0x70, 0x30, 0xA8, 0x70, 0x1C, 0x70, 0xFD, 0x13, 0x63, 0x29, 0x02, 0x79,
        0xEA, 0xD2, 0xA7, 0xAF, 0x35, 0x28, 0x32, 0x1C, 0x7B, 0xE6, 0x2F, 0x1A, 0xAA, 0x40,
        0x7E, 0x32, 0x8C, 0x27, 0x42, 0xFE, 0x82, 0x78, 0xEC, 0x0D, 0xEB, 0xE6, 0x83, 0x4B,
        0x6D, 0x81, 0x04, 0x40, 0x1A, 0x9E, 0x9A, 0x67, 0xF6, 0x72, 0x29, 0xFA, 0x04, 0xF0,
        0x9D, 0xE4, 0xF4, 0x03,
    };
    static constexpr std::array<std::uint8_t, 256> kAcidModulus = {
        0xDD, 0xC8, 0xDD, 0xF2, 0x4E, 0x6D, 0xF0, 0xCA, 0x9E, 0xC7, 0x5D, 0xC7, 0x7B, 0xAD,
        0xFE, 0x7D, 0x23, 0x89, 0x69, 0xB6, 0xF2, 0x06, 0xA2, 0x02, 0x88, 0xE1, 0x55, 0x91,
        0xAB, 0xCB, 0x4D, 0x50, 0x2E, 0xFC, 0x9D, 0x94, 0x76, 0xD6, 0x4C, 0xD8, 0xFF, 0x10,
        0xFA, 0x5E, 0x93, 0x0A, 0xB4, 0x57, 0xAC, 0x51, 0xC7, 0x16, 0x66, 0xF4, 0x1A, 0x54,
        0xC2, 0xC5, 0x04, 0x3D, 0x1B, 0xFE, 0x30, 0x20, 0x8A, 0xAC, 0x6F, 0x6F, 0xF5, 0xC7,
        0xB6, 0x68, 0xB8, 0xC9, 0x40, 0x6B, 0x42, 0xAD, 0x11, 0x21, 0xE7, 0x8B, 0xE9, 0x75,
        0x01, 0x86, 0xE4, 0x48, 0x9B, 0x0A, 0x0A, 0xF8, 0x7F, 0xE8, 0x87, 0xF2, 0x82, 0x01,
        0xE6, 0xA3, 0x0F, 0xE4, 0x66, 0xAE, 0x83, 0x3F, 0x4E, 0x9F, 0x5E, 0x01, 0x30, 0xA4,
        0x00, 0xB9, 0x9A, 0xAE, 0x5F, 0x03, 0xCC, 0x18, 0x60, 0xE5, 0xEF, 0x3B, 0x5E, 0x15,
        0x16, 0xFE, 0x1C, 0x82, 0x78, 0xB5, 0x2F, 0x47, 0x7C, 0x06, 0x66, 0x88, 0x5D, 0x35,
        0xA2, 0x67, 0x20, 0x10, 0xE7, 0x6C, 0x43, 0x68, 0xD3, 0xE4, 0x5A, 0x68, 0x2A, 0x5A,
        0xE2, 0x6D, 0x73, 0xB0, 0x31, 0x53, 0x1C, 0x20, 0x09, 0x44, 0xF5, 0x1A, 0x9D, 0x22,
        0xBE, 0x12, 0xA1, 0x77, 0x11, 0xE2, 0xA1, 0xCD, 0x40, 0x9A, 0xA2, 0x8B, 0x60, 0x9B,
        0xEF, 0xA0, 0xD3, 0x48, 0x63, 0xA2, 0xF8, 0xA3, 0x2C, 0x08, 0x56, 0x52, 0x2E, 0x60,
        0x19, 0x67, 0x5A, 0xA7, 0x9F, 0xDC, 0x3F, 0x3F, 0x69, 0x2B, 0x31, 0x6A, 0xB7, 0x88,
        0x4A, 0x14, 0x84, 0x80, 0x33, 0x3C, 0x9D, 0x44, 0xB7, 0x3F, 0x4C, 0xE1, 0x75, 0xEA,
        0x37, 0xEA, 0xE8, 0x1E, 0x7C, 0x77, 0xB7, 0xC6, 0x1A, 0xA2, 0xF0, 0x9F, 0x10, 0x61,
        0xCD, 0x7B, 0x5B, 0x32, 0x4C, 0x37, 0xEF, 0xB1, 0x71, 0x68, 0x53, 0x0A, 0xED, 0x51,
        0x7D, 0x35, 0x22, 0xFD,
    };
    ContainerTrustAnchors anchors;
    anchors.nca_header_moduli.push_back(kNcaHeaderModulus);
    anchors.acid_moduli.push_back(kAcidModulus);
    return anchors;
}

// One scanned container: candidate NCA offsets and ticket extents.
struct ScannedContainer {
    std::vector<std::uint64_t> candidates;
    std::vector<container_detail::ContainerTicketRef> tickets;
};

// Detects the container format and records the candidate NCA offsets and
// ticket files of its partitions.
[[nodiscard]] bool scan_container(std::FILE* file, ScannedContainer& scan,
                                  ContainerFailure& failure) {
    std::array<std::byte, 0x10> magic{};
    std::array<std::byte, 0x10> head{};
    const bool has_pfs0 = read_plain(file, 0, magic) &&
                          container_detail::read_u32_le(magic, 0) == kPfs0Magic;
    const bool has_head = read_plain(file, 0x100, head) &&
                          container_detail::read_u32_le(head, 0) == kHeadMagic;
    if (has_pfs0) {
        bool ok = false;
        collect_partition_candidates(file, 0, false, scan.candidates, &scan.tickets, failure, ok);
        return ok;
    }
    if (has_head) {
        std::array<std::byte, 8> offset_bytes{};
        if (!read_plain(file, 0x130, offset_bytes)) {
            failure = {ContainerError::partition_invalid,
                       "Reading the gamecard root partition offset failed."};
            return false;
        }
        const std::uint64_t root_offset = container_detail::read_u64_le(offset_bytes, 0);
        return collect_xci_candidates(file, root_offset, scan.candidates, &scan.tickets, failure);
    }
    if (std::fseek(file, 0, SEEK_END) == 0 && std::ftell(file) >= 0xC00) {
        scan.candidates.push_back(0);
        return true;
    }
    failure = {ContainerError::unknown_container_format,
               "The file is neither an NSP, XCI, nor standalone NCA container."};
    return false;
}

bool scan_container_entries(std::FILE* file, ContainerScan& scan, ContainerFailure& failure) {
    auto collect = [&](std::uint64_t base, bool hfs0) -> bool {
        PlainPartition partition;
        if (!read_plain_partition(file, base, hfs0, partition, failure)) {
            return false;
        }
        constexpr std::string_view kTicketSuffix = ".tik";
        for (const PlainEntry& entry : partition.entries) {
            std::uint64_t offset = 0;
            if (!checked_partition_offset(partition.data_offset, entry.offset, offset, failure)) {
                return false;
            }
            scan.entries.push_back({entry.name, offset, entry.size});
            if (entry.name.size() > kTicketSuffix.size() &&
                entry.name.compare(entry.name.size() - kTicketSuffix.size(), kTicketSuffix.size(),
                                   kTicketSuffix) == 0) {
                scan.tickets.push_back({offset, entry.size});
            }
        }
        return true;
    };

    std::array<std::byte, 0x10> magic{};
    std::array<std::byte, 0x10> head{};
    const bool has_pfs0 = read_plain(file, 0, magic) &&
                          container_detail::read_u32_le(magic, 0) == kPfs0Magic;
    const bool has_head = read_plain(file, 0x100, head) &&
                          container_detail::read_u32_le(head, 0) == kHeadMagic;
    if (has_pfs0) {
        return collect(0, false);
    }
    if (has_head) {
        std::array<std::byte, 8> offset_bytes{};
        if (!read_plain(file, 0x130, offset_bytes)) {
            failure = {ContainerError::partition_invalid,
                       "Reading the gamecard root partition offset failed."};
            return false;
        }
        const std::uint64_t root_offset = container_detail::read_u64_le(offset_bytes, 0);
        PlainPartition root;
        if (!read_plain_partition(file, root_offset, true, root, failure)) {
            return false;
        }
        static constexpr std::string_view kSearchOrder[] = {"secure", "normal", "update"};
        for (const std::string_view wanted : kSearchOrder) {
            const auto match = std::find_if(root.entries.begin(), root.entries.end(),
                                            [&](const PlainEntry& entry) {
                                                return entry.name == wanted;
                                            });
            if (match == root.entries.end()) {
                continue;
            }
            std::uint64_t partition_base = 0;
            if (!checked_partition_offset(root.data_offset, match->offset, partition_base,
                                          failure) ||
                !collect(partition_base, true)) {
                return false;
            }
        }
        return true;
    }
    if (std::fseek(file, 0, SEEK_END) == 0) {
        const long size = std::ftell(file);
        if (size >= 0xC00) {
            scan.entries.push_back({"content.nca", 0, static_cast<std::uint64_t>(size)});
            return true;
        }
    }
    failure = {ContainerError::unknown_container_format,
               "The file is neither an NSP, XCI, nor standalone NCA container."};
    return false;
}

// Opens the first verified program NCA of a scanned container. The search
// stops at the first verified program NCA; a program NCA that fails
// admission aborts with its structured cause.
[[nodiscard]] bool open_program(std::FILE* file, const ScannedContainer& scan,
                                const ContainerExtractionOptions& options,
                                std::uint64_t wanted_title_id, bool patch_sections_allowed,
                                container_detail::NcaProgram& program,
                                ContainerFailure& failure) {
    const container_detail::TitleKeySource title_keys{file, &scan.tickets,
                                                      &options.title_keys};
    for (const std::uint64_t base : scan.candidates) {
        const container_detail::CandidateOutcome outcome =
            container_detail::open_program_nca_candidate(file, base, options.keys, title_keys,
                                                         options.anchors, wanted_title_id,
                                                         patch_sections_allowed, program,
                                                         failure);
        if (outcome == container_detail::CandidateOutcome::failure) {
            return false;
        }
        if (outcome == container_detail::CandidateOutcome::verified) {
            return true;
        }
    }
    failure = {ContainerError::program_nca_missing,
               wanted_title_id == 0
                   ? "No program NCA was found in the container."
                   : "No program NCA matching the base title id was found in the update "
                     "container."};
    return false;
}

const container_detail::NcaSection* find_section(const container_detail::NcaProgram& program,
                                                 std::uint8_t partition_type,
                                                 std::uint8_t fs_type) noexcept {
    for (const container_detail::NcaSection& section : program.sections) {
        if (section.partition_type == partition_type && section.fs_type == fs_type) {
            return &section;
        }
    }
    return nullptr;
}

std::variant<ContainerFailure, ContainerExtraction> extract_program_container_impl(
    const std::filesystem::path& container_file, const ContainerExtractionOptions& options) {
    std::FILE* file = std::fopen(container_file.c_str(), "rb");
    if (!file) {
        return ContainerFailure{ContainerError::container_unavailable,
                                "Could not open the container file."};
    }
    // Opened later when an update container layers over the base.
    std::FILE* update_handle = nullptr;
    auto close = [&] {
        if (update_handle) std::fclose(update_handle);
        if (file) std::fclose(file);
    };

    // Base container: the same admission as before; its own verification is
    // unchanged whether or not an update layers over it.
    ContainerFailure failure;
    ScannedContainer base_scan;
    if (!scan_container(file, base_scan, failure)) {
        close();
        return failure;
    }
    container_detail::NcaProgram base_program;
    if (!open_program(file, base_scan, options, 0, false, base_program, failure)) {
        close();
        return failure;
    }

    // The update program supplies the active content: its ExeFS, and its
    // RomFS section (BKTR-patched over the base section) when present.
    const container_detail::NcaProgram* content_program = &base_program;
    const container_detail::NcaSection* base_romfs =
        find_section(base_program, container_detail::kPartitionRomFs,
                     container_detail::kFsTypeRomFs);
    const container_detail::NcaSection* content_romfs = base_romfs;
    container_detail::SectionView romfs_view;
    romfs_view.file = file;
    romfs_view.program = &base_program;
    romfs_view.section = base_romfs;

    // Output directories are created only after both containers verify, so
    // a failed admission never leaves a partial extraction tree behind.
    const std::filesystem::path exefs_directory = options.output_directory / "exefs";
    const std::filesystem::path romfs_directory = options.output_directory / "romfs";
    auto ensure_output_directories = [&]() -> bool {
        if (!make_directories(options.output_directory)) {
            failure = {ContainerError::io_failure,
                       "Could not create the extraction output directory."};
            return false;
        }
        if (!make_directories(exefs_directory) || !make_directories(romfs_directory)) {
            failure = {ContainerError::io_failure,
                       "Could not create the extraction directories."};
            return false;
        }
        return true;
    };

    const bool has_update = !options.update_container_file.empty();
    // The update program and its container file outlive the block below:
    // the patched-RomFS view and section pointers refer to them.
    container_detail::NcaProgram update_program;
    if (has_update) {
        update_handle = std::fopen(options.update_container_file.c_str(), "rb");
        if (!update_handle) {
            close();
            return ContainerFailure{ContainerError::container_unavailable,
                                    "Could not open the update container file."};
        }
        std::FILE* update_file = update_handle;
        ScannedContainer update_scan;
        if (!scan_container(update_file, update_scan, failure) ||
            !open_program(update_file, update_scan, options, base_program.title_id, true,
                          update_program, failure)) {
            close();
            return failure;
        }
        const container_detail::NcaSection* update_exefs = find_section(
            update_program, container_detail::kPartitionPfs0, container_detail::kFsTypePfs0);
        const container_detail::NcaSection* update_romfs = find_section(
            update_program, container_detail::kPartitionRomFs, container_detail::kFsTypeRomFs);
        if (!update_exefs) {
            close();
            return ContainerFailure{ContainerError::section_invalid,
                                    "The update program NCA lacks an ExeFS section."};
        }
        if (update_romfs) {
            if (!base_romfs) {
                close();
                return ContainerFailure{
                    ContainerError::section_invalid,
                    "The update program NCA has a RomFS section but the base program NCA does "
                    "not; the patched image has no base to layer over."};
            }
            if (update_romfs->is_bktr()) {
                // The patched image's IVFC chain verifies over the layered
                // bytes exactly as an unpatched RomFS verifies.
                const container_detail::SectionView patched{
                    &update_program, update_romfs, &base_program, base_romfs, update_file,
                    file};
                if (!container_detail::verify_section_view_chain(patched, failure)) {
                    close();
                    return failure;
                }
            }
            // A plain CTR update RomFS verified inside its own open; it
            // replaces the base image without a patch layer.
            content_program = &update_program;
            content_romfs = update_romfs;
            romfs_view.program = &update_program;
            romfs_view.section = update_romfs;
            if (update_romfs->is_bktr()) {
                romfs_view.base_program = &base_program;
                romfs_view.base_section = base_romfs;
                romfs_view.base_file = file;
            }
        }
        // ExeFS extraction and the patched RomFS reads use the update file
        // while it stays open; the base file serves the base ranges.
        if (!ensure_output_directories()) {
            close();
            return failure;
        }
        container_detail::Pfs0Partition partition;
        if (!container_detail::read_pfs0_partition(update_program, *update_exefs, update_file,
                                                   update_exefs->superblock_offset, partition,
                                                   failure)) {
            close();
            return failure;
        }
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < partition.file_count; ++i) {
            total += partition.entry_size(i);
            if (total > options.limits.max_total_bytes) {
                close();
                return ContainerFailure{ContainerError::section_invalid,
                                        "The ExeFS entries exceed the configured total limit."};
            }
            const std::string name(partition.entry_name(i));
            if (!sanitize_romfs_name(name)) {
                close();
                return ContainerFailure{ContainerError::section_invalid,
                                        "An ExeFS entry has an unusable name."};
            }
            const container_detail::SectionView exefs_view{&update_program, update_exefs, nullptr,
                                                           nullptr, update_file};
            if (!extract_exefs_file(exefs_view, partition, i, exefs_directory / name,
                                    options.limits.max_file_bytes, failure)) {
                close();
                return failure;
            }
        }
        // The update file stays open: the patched RomFS reads below map
        // patch ranges through it and base ranges through the base file.
        romfs_view.file = update_handle;
    }

    const container_detail::NcaSection* exefs = find_section(
        *content_program, container_detail::kPartitionPfs0, container_detail::kFsTypePfs0);
    if (!exefs || !content_romfs) {
        close();
        return ContainerFailure{ContainerError::section_invalid,
                                "The program NCA lacks an ExeFS or RomFS section."};
    }

    if (!has_update) {
        if (!ensure_output_directories()) {
            close();
            return failure;
        }
        container_detail::Pfs0Partition partition;
        if (!container_detail::read_pfs0_partition(base_program, *exefs, file,
                                                   exefs->superblock_offset, partition,
                                                   failure)) {
            close();
            return failure;
        }
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < partition.file_count; ++i) {
            total += partition.entry_size(i);
            if (total > options.limits.max_total_bytes) {
                close();
                return ContainerFailure{ContainerError::section_invalid,
                                        "The ExeFS entries exceed the configured total limit."};
            }
            const std::string name(partition.entry_name(i));
            if (!sanitize_romfs_name(name)) {
                close();
                return ContainerFailure{ContainerError::section_invalid,
                                        "An ExeFS entry has an unusable name."};
            }
            const container_detail::SectionView exefs_view{&base_program, exefs, nullptr, nullptr,
                                                           file};
            if (!extract_exefs_file(exefs_view, partition, i, exefs_directory / name,
                                    options.limits.max_file_bytes, failure)) {
                close();
                return failure;
            }
        }
    }
    if (!extract_romfs_tree(romfs_view, content_romfs->romfs_offset, romfs_directory, failure)) {
        close();
        return failure;
    }
    close();
    ContainerExtraction extraction;
    extraction.exefs_directory = exefs_directory;
    extraction.romfs_directory = romfs_directory;
    extraction.update_applied = has_update;
    extraction.title_id = base_program.title_id;
    return extraction;
}

std::variant<ContainerFailure, ContainerExtraction> extract_program_container(
    const std::filesystem::path& container_file, const ContainerExtractionOptions& options) {
    // Allocation failures from the host are structured failures, matching
    // the loader's other entry points.
    try {
        return extract_program_container_impl(container_file, options);
    } catch (const std::bad_alloc&) {
        return ContainerFailure{ContainerError::allocation_failed,
                                "Host memory for container extraction could not be allocated."};
    }
}

namespace {

// Public data, then data. Content type is tested before title id, so a
// program NCA that shares the container is skipped rather than admitted.
constexpr int kDataContentTypes[] = {5, 4};

std::variant<ContainerFailure, DataRomFsExtraction> extract_data_romfs_impl(
    const std::filesystem::path& container_file, const ContainerExtractionOptions& options,
    std::uint64_t title_id) {
    if (title_id == 0) {
        return ContainerFailure{ContainerError::program_nca_missing,
                                "An add-on data RomFS requires a title id."};
    }
    std::FILE* file = std::fopen(container_file.c_str(), "rb");
    if (!file) {
        return ContainerFailure{ContainerError::container_unavailable,
                                "Could not open the add-on container file."};
    }
    auto close = [&] { std::fclose(file); };
    ContainerFailure failure;
    ScannedContainer scan;
    if (!scan_container(file, scan, failure)) {
        close();
        return failure;
    }
    const container_detail::TitleKeySource title_keys{file, &scan.tickets, &options.title_keys};
    container_detail::NcaProgram program;
    bool opened = false;
    for (const int content_type : kDataContentTypes) {
        for (const std::uint64_t base : scan.candidates) {
            const container_detail::CandidateOutcome outcome =
                container_detail::open_nca_candidate(file, base, options.keys, title_keys,
                                                     options.anchors, content_type, title_id,
                                                     false, false, program, failure);
            if (outcome == container_detail::CandidateOutcome::failure) {
                close();
                return failure;
            }
            if (outcome == container_detail::CandidateOutcome::verified) {
                opened = true;
                break;
            }
        }
        if (opened) {
            break;
        }
    }
    if (!opened) {
        close();
        return ContainerFailure{ContainerError::program_nca_missing,
                                "No data NCA matching the add-on title id was found in the "
                                "container."};
    }
    const container_detail::NcaSection* romfs = find_section(
        program, container_detail::kPartitionRomFs, container_detail::kFsTypeRomFs);
    if (!romfs) {
        close();
        return ContainerFailure{ContainerError::section_invalid,
                                "The add-on data NCA has no RomFS section."};
    }
    if (!make_directories(options.output_directory)) {
        close();
        return ContainerFailure{ContainerError::io_failure,
                                "Could not create the add-on RomFS directory."};
    }
    const container_detail::SectionView view{&program, romfs, nullptr, nullptr, file};
    if (!extract_romfs_tree(view, romfs->romfs_offset, options.output_directory, failure)) {
        close();
        return failure;
    }
    DataRomFsExtraction extraction;
    extraction.romfs_directory = options.output_directory;
    extraction.title_id = program.title_id;
    extraction.content_type = program.content_type;
    close();
    return extraction;
}

} // namespace

std::variant<ContainerFailure, DataRomFsExtraction> extract_data_romfs(
    const std::filesystem::path& container_file, const ContainerExtractionOptions& options,
    std::uint64_t title_id) {
    try {
        return extract_data_romfs_impl(container_file, options, title_id);
    } catch (const std::bad_alloc&) {
        return ContainerFailure{ContainerError::allocation_failed,
                                "Host memory for add-on extraction could not be allocated."};
    }
}

} // namespace armigo::loader
