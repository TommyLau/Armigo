// ============================================================================
// Armigo RomFS - Directory-backed virtual storage
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/romfs/romfs_image.h>
#include <armigo/romfs/storage.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace armigo::romfs {

// Serves a RomFS image built from an extracted host directory. The header and
// metadata tables are built in memory; each file's data is read from its host
// file at the image offset, so the data region is never copied or held whole.
class DirectoryStorage final : public Storage {
public:
    // Builds the tables and file map from `root`. Returns null and fills
    // `error` when the directory cannot be read or a file cannot be sized.
    static std::shared_ptr<DirectoryStorage> open(const std::filesystem::path& root,
                                                  std::string& error);
    ~DirectoryStorage() override;
    std::uint64_t size() const noexcept override;
    // Reads of different files, and of one file, run at the same time: a file
    // stays open after its first read and is read at its offset, so the host's
    // read-ahead sees a sequential reader as one.
    std::optional<std::size_t> read(std::uint64_t offset,
                                    std::span<std::byte> destination) override;

    const VirtualImage& image() const noexcept { return image_; }

private:
    DirectoryStorage(std::filesystem::path root, VirtualImage image);
    // The descriptor of file `index`, or -1 when it cannot be opened.
    // `temporary` is set when the caller must close it (the cache is full).
    int descriptor(std::size_t index, bool& temporary);
    std::filesystem::path root_;
    VirtualImage image_;
    std::mutex mutex_;  // guards descriptors_
    std::vector<int> descriptors_;  // by file index, -1 until first read
    std::size_t cached_ = 0;
    std::size_t cached_limit_ = 0;  // descriptors kept open, from the process's limit
};

// A small in-memory store for tests and synthetic images.
class MemoryStorage final : public Storage {
public:
    explicit MemoryStorage(std::vector<std::byte> bytes) : bytes_(std::move(bytes)) {}
    std::uint64_t size() const noexcept override { return bytes_.size(); }
    std::optional<std::size_t> read(std::uint64_t offset,
                                    std::span<std::byte> destination) override;
    const std::vector<std::byte>& bytes() const noexcept { return bytes_; }

private:
    std::vector<std::byte> bytes_;
};

}  // namespace armigo::romfs
