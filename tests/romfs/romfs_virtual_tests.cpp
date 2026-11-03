// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The directory-backed storage must serve exactly the bytes of the contiguous
// image the builder produces, read on demand and across file/padding
// boundaries, without holding the data region.

#include <armigo/romfs/romfs_virtual.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

namespace {
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (false)

void write_file(const std::filesystem::path& path, const std::string& bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void test_virtual_matches_contiguous() {
    const auto root = std::filesystem::temp_directory_path() / "armigo-romfs-virtual";
    std::filesystem::remove_all(root);
    write_file(root / "alpha.txt", "alpha");
    write_file(root / "Dir" / "beta.bin", std::string(97, 'b'));
    write_file(root / "Dir" / "empty.bin", "");
    write_file(root / "Other" / "gamma.bin", std::string(300, 'g'));

    armigo::romfs::ImageLayout layout;
    std::string error;
    const auto virtual_image = armigo::romfs::build_virtual(root, layout, error);
    CHECK(virtual_image.has_value());
    if (!virtual_image) {
        std::fprintf(stderr, "build_virtual error: %s\n", error.c_str());
        return;
    }
    CHECK(virtual_image->tables.size() == virtual_image->data_offset);
    CHECK(virtual_image->size >= virtual_image->data_offset);
    CHECK(virtual_image->files.size() == layout.files.size());
    // Every file map entry matches its metadata entry and lies inside the image.
    for (const auto& file : virtual_image->files) {
        CHECK(file.offset >= virtual_image->data_offset);
        CHECK(file.offset + file.size <= virtual_image->size);
    }

    armigo::romfs::ImageLayout contiguous_layout;
    const auto contiguous = armigo::romfs::build(root, contiguous_layout, error);
    CHECK(contiguous.has_value());
    if (!contiguous) return;
    CHECK(contiguous->size() == virtual_image->size);
    CHECK(std::memcmp(contiguous->data(), virtual_image->tables.data(),
                      virtual_image->tables.size()) == 0);

    std::string open_error;
    const auto storage = armigo::romfs::DirectoryStorage::open(root, open_error);
    CHECK(storage != nullptr);
    if (!storage) {
        std::fprintf(stderr, "open error: %s\n", open_error.c_str());
        return;
    }
    CHECK(storage->size() == contiguous->size());

    // Read the whole virtual image in awkward chunk sizes, crossing files and
    // alignment padding, and compare every byte with the contiguous image.
    for (const std::size_t chunk : {1u, 7u, 96u, 97u, 301u, 4096u}) {
        std::uint64_t offset = 0;
        bool same = true;
        std::vector<std::byte> buffer(chunk);
        while (offset < storage->size()) {
            const auto got = storage->read(offset, buffer);
            same = same && got.has_value();
            if (!got) break;
            same = same && *got == std::min<std::uint64_t>(chunk, storage->size() - offset);
            same = same && std::memcmp(buffer.data(), contiguous->data() + offset, *got) == 0;
            if (!same) break;
            offset += *got;
            if (*got == 0) break;
        }
        CHECK(same && offset == storage->size());
    }
    // A read at the end returns zero bytes; a read past the end returns zero.
    std::array<std::byte, 8> tail{};
    CHECK(storage->read(storage->size(), tail) == std::size_t{0});
    CHECK(storage->read(storage->size() + 100, tail) == std::size_t{0});
    // Reading beyond the end of a range returns only the available bytes.
    const auto short_read = storage->read(storage->size() - 3, tail);
    CHECK(short_read == std::size_t{3});

    // A host file that disappears after the tables were built surfaces as an
    // explicit I/O failure, not as zero-filled data. (A file already open when
    // it disappears keeps serving its data, as any open file does.)
    const auto fresh = armigo::romfs::DirectoryStorage::open(root, open_error);
    CHECK(fresh != nullptr);
    if (!fresh) return;
    std::filesystem::remove(root / "Other" / "gamma.bin");
    const auto missing = fresh->read(fresh->size() - 300, tail);
    CHECK(!missing.has_value());
    std::filesystem::remove_all(root);
}

// More files than the storage keeps open, read by several threads at once at
// random places, must give the bytes of the contiguous image.
void test_concurrent_reads_across_many_files() {
    const auto root = std::filesystem::temp_directory_path() / "armigo-romfs-concurrent";
    std::filesystem::remove_all(root);
    for (unsigned file = 0; file != 700; ++file) {
        std::string bytes(37 + file * 3, static_cast<char>('a' + file % 26));
        bytes[bytes.size() / 2] = static_cast<char>(file);
        write_file(root / ("d" + std::to_string(file % 7)) / ("f" + std::to_string(file) + ".bin"), bytes);
    }
    std::string error;
    armigo::romfs::ImageLayout layout;
    const auto contiguous = armigo::romfs::build(root, layout, error);
    const auto storage = armigo::romfs::DirectoryStorage::open(root, error);
    CHECK(contiguous.has_value() && storage != nullptr);
    if (!contiguous || !storage) return;
    std::atomic<unsigned> mismatches{0};
    std::vector<std::thread> readers;
    for (unsigned thread = 0; thread != 4; ++thread)
        readers.emplace_back([&, thread] {
            std::uint32_t state = 7 + thread;
            std::vector<std::byte> buffer(5000);
            for (unsigned round = 0; round != 3000; ++round) {
                state = state * 1664525u + 1013904223u;
                const std::uint64_t offset = (state >> 8) % storage->size();
                const auto got = storage->read(offset, buffer);
                const auto want = std::min<std::uint64_t>(buffer.size(), storage->size() - offset);
                if (!got || *got != want || std::memcmp(buffer.data(), contiguous->data() + offset, *got) != 0)
                    ++mismatches;
            }
        });
    for (auto& reader : readers) reader.join();
    CHECK(mismatches == 0);
    std::filesystem::remove_all(root);
}

// Under a low soft limit on open files (a process started from Finder gets
// 256) the storage keeps few files open and every read still succeeds.
void test_reads_within_a_low_descriptor_limit() {
    rlimit saved{};
    CHECK(getrlimit(RLIMIT_NOFILE, &saved) == 0);
    rlimit low = saved;
    low.rlim_cur = 64;
    CHECK(setrlimit(RLIMIT_NOFILE, &low) == 0);
    const auto root = std::filesystem::temp_directory_path() / "armigo-romfs-low-limit";
    std::filesystem::remove_all(root);
    for (unsigned file = 0; file != 300; ++file)
        write_file(root / ("f" + std::to_string(file) + ".bin"), std::string(64 + file, static_cast<char>('a' + file % 26)));
    std::string error;
    armigo::romfs::ImageLayout layout;
    const auto contiguous = armigo::romfs::build(root, layout, error);
    const auto storage = armigo::romfs::DirectoryStorage::open(root, error);
    CHECK(contiguous.has_value() && storage != nullptr);
    if (contiguous && storage) {
        const auto open_files = [] {
            unsigned count = 0;
            for (const auto& entry : std::filesystem::directory_iterator("/dev/fd")) {
                static_cast<void>(entry);
                ++count;
            }
            return count;
        };
        const unsigned before = open_files();
        std::atomic<unsigned> failures_seen{0};
        std::vector<std::thread> readers;
        for (unsigned thread = 0; thread != 4; ++thread)
            readers.emplace_back([&, thread] {
                std::vector<std::byte> buffer(700);
                for (unsigned round = 0; round != 600; ++round) {
                    const std::uint64_t offset = (std::uint64_t{round} * 7919 + thread * 104729) % storage->size();
                    const auto got = storage->read(offset, buffer);
                    const auto want = std::min<std::uint64_t>(buffer.size(), storage->size() - offset);
                    if (!got || *got != want || std::memcmp(buffer.data(), contiguous->data() + offset, *got) != 0)
                        ++failures_seen;
                }
            });
        for (auto& reader : readers) reader.join();
        CHECK(failures_seen == 0);
        // At most an eighth of the limit stays open (plus the directory scan's own).
        CHECK(open_files() <= before + 64 / 8 + 1);
    }
    CHECK(setrlimit(RLIMIT_NOFILE, &saved) == 0);
    std::filesystem::remove_all(root);
}

void test_memory_storage() {
    std::vector<std::byte> bytes(32);
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::byte>(index);
    armigo::romfs::MemoryStorage storage(std::move(bytes));
    std::array<std::byte, 16> buffer{};
    CHECK(storage.size() == 32);
    auto got = storage.read(10, buffer);
    CHECK(got == std::size_t{16});
    CHECK(buffer[0] == std::byte{10} && buffer[15] == std::byte{25});
    got = storage.read(24, buffer);
    CHECK(got == std::size_t{8});
    CHECK(buffer[7] == std::byte{31});
    CHECK(storage.read(32, buffer) == std::size_t{0});
}
}  // namespace

int main() {
    test_reads_within_a_low_descriptor_limit();
    test_virtual_matches_contiguous();
    test_concurrent_reads_across_many_files();
    test_memory_storage();
    std::printf("RomFS virtual failures: %u\n", failures);
    return failures ? 1 : 0;
}
