// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Builds an image from a synthetic tree and parses it back independently: the
// reader in this file does not share code with the builder beyond the byte
// layout, so an offset or link error surfaces here. The lookup helpers implement
// the public RomFS path hash and bucket chain walk used by guest clients, which
// is the only way a file name is resolved at runtime.

#include <armigo/romfs/romfs_image.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (false)

constexpr std::uint32_t entry_absent = 0xFFFFFFFFu;

std::uint32_t get32(const std::vector<std::byte>& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, 4);
    return value;
}
std::uint64_t get64(const std::vector<std::byte>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, 8);
    return value;
}
std::string name_at(const std::vector<std::byte>& bytes, std::size_t offset, std::size_t length) {
    std::string name(length, '\0');
    std::memcpy(name.data(), bytes.data() + offset, length);
    return name;
}

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

std::uint32_t client_path_hash(std::uint32_t parent, const std::string& name) {
    std::uint32_t hash = parent ^ 123456789u;
    for (const unsigned char byte : name) {
        hash = (hash >> 5) | (hash << 27);
        hash ^= byte;
    }
    return hash;
}

std::uint32_t client_bucket_count(std::uint32_t entries) {
    if (entries < 3) return 3;
    if (entries < 19) return entries | 1;
    auto count = entries;
    while (count % 2 == 0 || count % 3 == 0 || count % 5 == 0 || count % 7 == 0 ||
           count % 11 == 0 || count % 13 == 0 || count % 17 == 0) {
        ++count;
    }
    return count;
}

struct ImageView {
    const std::vector<std::byte>& bytes;
    std::uint32_t directory_hash = 0;
    std::uint32_t directory_hash_size = 0;
    std::uint32_t directory_meta = 0;
    std::uint32_t file_hash = 0;
    std::uint32_t file_hash_size = 0;
    std::uint32_t file_meta = 0;
    std::uint32_t data = 0;

    std::uint32_t find_directory(std::uint32_t parent, const std::string& name) const {
        const auto bucket = client_path_hash(parent, name) % (directory_hash_size / 4);
        for (auto current = get32(bytes, directory_hash + bucket * 4); current != entry_absent;
             current = get32(bytes, directory_meta + current + 16)) {
            const auto length = get32(bytes, directory_meta + current + 20);
            if (name_at(bytes, directory_meta + current + 24, length) == name) return current;
        }
        return entry_absent;
    }
    std::uint32_t find_file(std::uint32_t parent, const std::string& name) const {
        const auto bucket = client_path_hash(parent, name) % (file_hash_size / 4);
        for (auto current = get32(bytes, file_hash + bucket * 4); current != entry_absent;
             current = get32(bytes, file_meta + current + 24)) {
            const auto length = get32(bytes, file_meta + current + 28);
            if (name_at(bytes, file_meta + current + 32, length) == name) return current;
        }
        return entry_absent;
    }
    std::string file_text(std::uint32_t file) const {
        const auto offset = get64(bytes, file_meta + file + 8);
        const auto size = get64(bytes, file_meta + file + 16);
        return name_at(bytes, data + offset, size);
    }
};

void test_round_trip() {
    const auto root = std::filesystem::temp_directory_path() / "armigo-romfs-fixture";
    std::filesystem::remove_all(root);
    write_file(root / "alpha.txt", "alpha");
    write_file(root / "Game" / "data.bin", std::string(40, 'x'));
    write_file(root / "Game" / "sub" / "deep.txt", "deep");
    write_file(root / "Game" / "sub" / "zeta.txt", "zeta");
    write_file(root / "Pack" / "empty.bin", "");

    armigo::romfs::ImageLayout layout;
    std::string error;
    const auto image = armigo::romfs::build(root, layout, error);
    CHECK(image.has_value());
    if (!image) {
        std::fprintf(stderr, "build error: %s\n", error.c_str());
        return;
    }
    // Header: ten u64 fields, the shape nn::fs::RomFsFileSystem::Initialize reads.
    CHECK(get64(*image, 0x00) == 0x50);
    const ImageView view{*image,
                         static_cast<std::uint32_t>(get64(*image, 0x08)),
                         static_cast<std::uint32_t>(get64(*image, 0x10)),
                         static_cast<std::uint32_t>(get64(*image, 0x18)),
                         static_cast<std::uint32_t>(get64(*image, 0x28)),
                         static_cast<std::uint32_t>(get64(*image, 0x30)),
                         static_cast<std::uint32_t>(get64(*image, 0x38)),
                         static_cast<std::uint32_t>(get64(*image, 0x48))};
    const auto directory_meta_size = static_cast<std::uint32_t>(get64(*image, 0x20));
    const auto file_meta_size = static_cast<std::uint32_t>(get64(*image, 0x40));
    CHECK(view.directory_hash == 0x50);
    CHECK(view.directory_hash_size > 0 && view.file_hash_size > 0);
    // Every region must stay inside the image and be contiguous.
    CHECK(view.directory_hash + view.directory_hash_size == view.directory_meta);
    CHECK(view.directory_meta + directory_meta_size == view.file_hash);
    CHECK(view.file_hash + view.file_hash_size == view.file_meta);
    CHECK(view.file_meta + file_meta_size <= view.data);
    CHECK(view.data % 16 == 0);
    // The bucket counts follow the public RomFS table sizing rule.
    CHECK(view.directory_hash_size / 4 ==
          client_bucket_count(static_cast<std::uint32_t>(layout.directories.size())));
    CHECK(view.file_hash_size / 4 ==
          client_bucket_count(static_cast<std::uint32_t>(layout.files.size())));

    // Root metadata links are relative offsets into the directory table.
    CHECK(get32(*image, view.directory_meta + 8) != entry_absent);
    CHECK(get32(*image, view.directory_meta + 12) != entry_absent);

    // Resolve names only through the client hash and chain walk.
    const auto alpha = view.find_file(0, "alpha.txt");
    CHECK(alpha != entry_absent);
    CHECK(view.file_text(alpha) == "alpha");

    const auto game = view.find_directory(0, "Game");
    CHECK(game != entry_absent);
    const auto data = view.find_file(game, "data.bin");
    CHECK(data != entry_absent);
    CHECK(view.file_text(data) == std::string(40, 'x'));

    const auto sub = view.find_directory(game, "sub");
    CHECK(sub != entry_absent);
    const auto deep = view.find_file(sub, "deep.txt");
    CHECK(deep != entry_absent);
    CHECK(view.file_text(deep) == "deep");
    const auto zeta = view.find_file(sub, "zeta.txt");
    CHECK(zeta != entry_absent);
    CHECK(view.file_text(zeta) == "zeta");

    const auto pack = view.find_directory(0, "Pack");
    CHECK(pack != entry_absent);
    const auto empty = view.find_file(pack, "empty.bin");
    CHECK(empty != entry_absent);
    CHECK(view.file_text(empty).empty());

    // A name that does not exist must terminate at the end of its chain.
    CHECK(view.find_file(0, "missing.txt") == entry_absent);
    CHECK(view.find_directory(0, "Missing") == entry_absent);

    // Every file entry's data must lie inside the image.
    for (const auto& file : layout.files) {
        CHECK(view.data + file.data_offset + file.size <= image->size());
    }
    std::filesystem::remove_all(root);
}
} // namespace

int main() {
    test_round_trip();
    std::printf("RomFS image failures: %u\n", failures);
    return failures ? 1 : 0;
}
