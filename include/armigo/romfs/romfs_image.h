// ============================================================================
// Armigo Tools - RomFS image builder
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independent implementation of the public RomFS image layout used for the
// read-only file-system image this project exposes to guest clients:
//
//   header(0x50, ten u64 fields) | directory hash table | directory metadata table |
//   file hash table | file metadata table | file data region (16-byte aligned)
//
// Directory metadata entries are {parent, next sibling, first child directory,
// first file, bucket chain, name length, name padded to 4}. File metadata entries
// are {parent, next file, data offset, data length, bucket chain, name length,
// name padded to 4}. Metadata links and hash-table bucket values are offsets
// relative to their metadata table base; file data offsets are relative to the
// image's file data region. The bucket chain field points at the next entry in
// the same hash bucket and 0xFFFFFFFF terminates the chain. The path hash is the
// public RomFS hash: `parent_offset ^ 123456789`, then rotate right by 5 and xor
// each name byte. A hash table has one bucket per `hash_table_count(entries)`
// value and stores the head entry offset of each bucket. The root directory is
// entry zero of the directory metadata table with an empty name.
//
// The tables are emitted in one pre-order walk so that the assigned offsets and
// the written bytes cannot disagree. The builder contains no title-specific
// behaviour.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::romfs {

inline constexpr std::uint32_t absent = 0xFFFFFFFFu;
inline constexpr std::uint32_t header_length = 0x50;

struct ImageEntry {
    std::string path;  // slash separated, relative to the image root
    std::uint32_t meta_offset = 0;  // relative to the entry's metadata table
    std::uint64_t data_offset = 0;  // relative to the image file data region
    std::uint64_t size = 0;
    std::uint32_t hash = 0;
    bool directory = false;
};

struct ImageLayout {
    std::uint32_t directory_hash_offset = 0;
    std::uint32_t directory_hash_size = 0;
    std::uint32_t directory_meta_offset = 0;
    std::uint32_t directory_meta_size = 0;
    std::uint32_t file_hash_offset = 0;
    std::uint32_t file_hash_size = 0;
    std::uint32_t file_meta_offset = 0;
    std::uint32_t file_meta_size = 0;
    std::uint32_t file_data_offset = 0;
    std::vector<ImageEntry> directories;  // root first
    std::vector<ImageEntry> files;
};

inline std::uint32_t path_hash(std::uint32_t parent, std::string_view name) noexcept {
    std::uint32_t hash = parent ^ 123456789u;
    for (const unsigned char byte : name) {
        hash = (hash >> 5) | (hash << 27);
        hash ^= byte;
    }
    return hash;
}

inline std::uint32_t hash_table_count(std::uint32_t entries) noexcept {
    if (entries < 3) return 3;
    if (entries < 19) return entries | 1;
    auto count = entries;
    while (count % 2 == 0 || count % 3 == 0 || count % 5 == 0 || count % 7 == 0 ||
           count % 11 == 0 || count % 13 == 0 || count % 17 == 0) {
        ++count;
    }
    return count;
}

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept {
    return (value + alignment - 1) / alignment * alignment;
}

namespace detail {

struct Node {
    std::string name;
    bool directory = false;
    std::uint64_t size = 0;
    std::filesystem::path source;
    std::uint32_t hash = 0;
    std::vector<std::unique_ptr<Node>> children;
    std::uint32_t meta = 0;      // relative to the node's metadata table
    std::uint64_t data = 0;      // relative to the image file data region
    std::uint32_t first_child = absent;
    std::uint32_t first_file = absent;
    std::uint32_t next = absent;
};

inline bool load(const std::filesystem::path& directory, Node& node, std::string& error) {
    std::error_code code;
    std::vector<std::filesystem::directory_entry> entries;
    for (std::filesystem::directory_iterator it(directory, code), end; !code && it != end;
         it.increment(code)) {
        entries.push_back(*it);
    }
    if (code) {
        error = "cannot read " + directory.string() + ": " + code.message();
        return false;
    }
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
        return left.path().filename().string() < right.path().filename().string();
    });
    for (const auto& entry : entries) {
        auto child = std::make_unique<Node>();
        child->name = entry.path().filename().string();
        child->source = entry.path();
        const bool is_directory = entry.is_directory(code);
        if (code) {
            error = "cannot inspect " + entry.path().string() + ": " + code.message();
            return false;
        }
        if (is_directory) {
            child->directory = true;
            if (!load(entry.path(), *child, error)) return false;
        } else if (entry.is_regular_file(code) && !code) {
            child->size = entry.file_size(code);
            if (code) {
                error = "cannot size " + entry.path().string() + ": " + code.message();
                return false;
            }
        } else {
            continue;
        }
        node.children.push_back(std::move(child));
    }
    return true;
}

// Assigns relative metadata offsets in the same pre-order the bytes are written
// in, resolves the sibling, child and file chains, and records each path hash.
inline void assign(Node& node, std::uint32_t& directory_cursor, std::uint32_t& file_cursor) {
    node.meta = directory_cursor;
    directory_cursor += static_cast<std::uint32_t>(align_up(24 + node.name.size(), 4));
    Node* previous_directory = nullptr;
    for (auto& child : node.children) {
        if (!child->directory) continue;
        assign(*child, directory_cursor, file_cursor);
        if (!previous_directory) node.first_child = child->meta;
        else previous_directory->next = child->meta;
        previous_directory = child.get();
    }
    Node* previous_file = nullptr;
    for (auto& child : node.children) {
        if (child->directory) continue;
        child->meta = file_cursor;
        file_cursor += static_cast<std::uint32_t>(align_up(32 + child->name.size(), 4));
        if (!previous_file) node.first_file = child->meta;
        else previous_file->next = child->meta;
        previous_file = child.get();
    }
    for (auto& child : node.children) child->hash = path_hash(node.meta, child->name);
    node.hash = path_hash(0, node.name);
}

inline void assign_data(Node& node, std::uint64_t& cursor) {
    for (auto& child : node.children) {
        if (child->directory) {
            assign_data(*child, cursor);
            continue;
        }
        child->data = cursor;
        cursor = align_up(cursor + child->size, 0x10);
    }
}

inline void collect(const Node& node, const std::string& prefix, ImageLayout& layout) {
    if (prefix.empty()) {
        layout.directories.push_back(
            ImageEntry{.path = {}, .meta_offset = 0, .hash = node.hash, .directory = true});
    } else {
        layout.directories.push_back(
            ImageEntry{.path = prefix, .meta_offset = node.meta, .hash = node.hash, .directory = true});
    }
    for (const auto& child : node.children) {
        if (!child->directory) continue;
        collect(*child, prefix.empty() ? child->name : prefix + "/" + child->name, layout);
    }
    for (const auto& child : node.children) {
        if (child->directory) continue;
        layout.files.push_back(
            ImageEntry{.path = prefix.empty() ? child->name : prefix + "/" + child->name,
                       .meta_offset = child->meta,
                       .data_offset = child->data,
                       .size = child->size,
                       .hash = child->hash});
    }
}

inline void put32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
inline void put64(std::vector<std::byte>& bytes, std::size_t offset, std::uint64_t value) {
    for (unsigned i = 0; i != 8; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
inline void put_name(std::vector<std::byte>& bytes, std::size_t offset, const std::string& name) {
    for (std::size_t i = 0; i != name.size(); ++i)
        bytes[offset + i] = static_cast<std::byte>(static_cast<unsigned char>(name[i]));
}

// Writes the metadata with relative links. The bucket chain field is written by
// build() after the chain and bucket heads are known.
inline void write_entries(const Node& node, std::uint32_t parent, std::vector<std::byte>& bytes,
                          std::uint32_t directory_base, std::uint32_t file_base) {
    put32(bytes, directory_base + node.meta + 0, parent);
    put32(bytes, directory_base + node.meta + 4, node.next);
    put32(bytes, directory_base + node.meta + 8, node.first_child);
    put32(bytes, directory_base + node.meta + 12, node.first_file);
    put32(bytes, directory_base + node.meta + 16, 0);
    put32(bytes, directory_base + node.meta + 20, static_cast<std::uint32_t>(node.name.size()));
    put_name(bytes, directory_base + node.meta + 24, node.name);
    for (const auto& child : node.children)
        if (child->directory) write_entries(*child, node.meta, bytes, directory_base, file_base);
    for (const auto& child : node.children) {
        if (child->directory) continue;
        put32(bytes, file_base + child->meta + 0, node.meta);
        put32(bytes, file_base + child->meta + 4, child->next);
        put64(bytes, file_base + child->meta + 8, child->data);
        put64(bytes, file_base + child->meta + 16, child->size);
        put32(bytes, file_base + child->meta + 24, 0);
        put32(bytes, file_base + child->meta + 28, static_cast<std::uint32_t>(child->name.size()));
        put_name(bytes, file_base + child->meta + 32, child->name);
    }
}

} // namespace detail

// One file in a virtual image: where its data lives in the image and which
// host file supplies it.
struct VirtualFile {
    std::string path;         // slash separated, relative to the image root
    std::uint64_t offset = 0; // absolute image offset of the file data
    std::uint64_t size = 0;
};

// Header and metadata tables in memory, file data left in the host files. The
// storage layer serves reads from the files at their image offsets without
// copying the data region.
struct VirtualImage {
    std::vector<std::byte> tables; // header, hash tables and metadata tables
    std::uint64_t data_offset = 0;
    std::uint64_t size = 0;        // header/tables + all file data and padding
    std::vector<VirtualFile> files;
};

// Builds the table bytes and the file map for `root`. The files are only
// opened when their data is read; nothing in the data region is copied.
inline std::optional<VirtualImage> build_virtual(const std::filesystem::path& root,
                                                 ImageLayout& layout,
                                                 std::string& error) {
    detail::Node tree;
    tree.directory = true;
    if (!detail::load(root, tree, error)) return std::nullopt;

    std::uint32_t directory_meta_cursor = 0;
    std::uint32_t file_meta_cursor = 0;
    detail::assign(tree, directory_meta_cursor, file_meta_cursor);
    layout.directory_meta_size = directory_meta_cursor;
    layout.file_meta_size = file_meta_cursor;

    std::uint64_t data_cursor = 0;
    detail::assign_data(tree, data_cursor);
    layout.directories.clear();
    layout.files.clear();
    detail::collect(tree, {}, layout);

    const auto directory_buckets =
        hash_table_count(static_cast<std::uint32_t>(layout.directories.size()));
    const auto file_buckets = hash_table_count(static_cast<std::uint32_t>(layout.files.size()));
    layout.directory_hash_size = directory_buckets * 4;
    layout.file_hash_size = file_buckets * 4;
    layout.directory_hash_offset = header_length;
    layout.directory_meta_offset = layout.directory_hash_offset + layout.directory_hash_size;
    layout.file_hash_offset = layout.directory_meta_offset + layout.directory_meta_size;
    layout.file_meta_offset = layout.file_hash_offset + layout.file_hash_size;
    layout.file_data_offset =
        static_cast<std::uint32_t>(align_up(layout.file_meta_offset + layout.file_meta_size, 0x10));

    VirtualImage image;
    image.data_offset = layout.file_data_offset;
    image.size = layout.file_data_offset + data_cursor;
    image.tables.resize(layout.file_data_offset, std::byte{0});
    auto& bytes = image.tables;
    detail::put64(bytes, 0x00, header_length);
    detail::put64(bytes, 0x08, layout.directory_hash_offset);
    detail::put64(bytes, 0x10, layout.directory_hash_size);
    detail::put64(bytes, 0x18, layout.directory_meta_offset);
    detail::put64(bytes, 0x20, layout.directory_meta_size);
    detail::put64(bytes, 0x28, layout.file_hash_offset);
    detail::put64(bytes, 0x30, layout.file_hash_size);
    detail::put64(bytes, 0x38, layout.file_meta_offset);
    detail::put64(bytes, 0x40, layout.file_meta_size);
    detail::put64(bytes, 0x48, layout.file_data_offset);
    for (std::uint32_t bucket = 0; bucket != directory_buckets; ++bucket)
        detail::put32(bytes, layout.directory_hash_offset + bucket * 4, absent);
    for (std::uint32_t bucket = 0; bucket != file_buckets; ++bucket)
        detail::put32(bytes, layout.file_hash_offset + bucket * 4, absent);
    detail::write_entries(tree, 0, bytes, layout.directory_meta_offset, layout.file_meta_offset);

    std::vector<std::uint32_t> directory_heads(directory_buckets, absent);
    for (const auto& directory : layout.directories) {
        const auto bucket = directory.hash % directory_buckets;
        detail::put32(bytes, layout.directory_meta_offset + directory.meta_offset + 16,
                      directory_heads[bucket]);
        detail::put32(bytes, layout.directory_hash_offset + bucket * 4, directory.meta_offset);
        directory_heads[bucket] = directory.meta_offset;
    }
    std::vector<std::uint32_t> file_heads(file_buckets, absent);
    for (const auto& file : layout.files) {
        const auto bucket = file.hash % file_buckets;
        detail::put32(bytes, layout.file_meta_offset + file.meta_offset + 24, file_heads[bucket]);
        detail::put32(bytes, layout.file_hash_offset + bucket * 4, file.meta_offset);
        file_heads[bucket] = file.meta_offset;
    }

    image.files.reserve(layout.files.size());
    for (const auto& file : layout.files) {
        image.files.push_back(VirtualFile{.path = file.path,
                                          .offset = layout.file_data_offset + file.data_offset,
                                          .size = file.size});
    }
    return image;
}

// Builds one contiguous image for `root`. The file data is copied into the
// returned bytes; build_virtual() is the non-copying form used by the runtime
// storage.
inline std::optional<std::vector<std::byte>> build(const std::filesystem::path& root,
                                                   ImageLayout& layout, std::string& error) {
    const auto image = build_virtual(root, layout, error);
    if (!image) return std::nullopt;
    std::vector<std::byte> bytes(static_cast<std::size_t>(image->size), std::byte{0});
    std::copy(image->tables.begin(), image->tables.end(), bytes.begin());
    for (const auto& file : image->files) {
        if (file.size == 0) continue;
        std::ifstream source(root / file.path, std::ios::binary);
        if (!source) {
            error = "cannot open " + (root / file.path).string();
            return std::nullopt;
        }
        source.read(reinterpret_cast<char*>(bytes.data() + file.offset),
                    static_cast<std::streamsize>(file.size));
        if (!source) {
            error = "cannot read " + (root / file.path).string();
            return std::nullopt;
        }
    }
    (void)layout;
    return bytes;
}

} // namespace armigo::romfs
