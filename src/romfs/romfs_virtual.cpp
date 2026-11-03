// ============================================================================
// Armigo RomFS - Directory-backed virtual storage
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/romfs/romfs_virtual.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

namespace armigo::romfs {
namespace {

// Files stay open after their first read, up to an eighth of the process's
// soft descriptor limit and at most 512, so the rest of the process (shader
// package, saves, the GPU driver) keeps the others. Beyond the budget a file
// is opened for each read and closed again.
std::size_t descriptor_budget() {
    rlimit limit{};
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) return 32;
    if (limit.rlim_cur == RLIM_INFINITY) return 512;
    return std::min<std::size_t>(512, static_cast<std::size_t>(limit.rlim_cur / 8));
}

} // namespace

DirectoryStorage::DirectoryStorage(std::filesystem::path root, VirtualImage image)
    : root_(std::move(root)), image_(std::move(image)) {
    std::sort(image_.files.begin(), image_.files.end(),
              [](const VirtualFile& left, const VirtualFile& right) {
                  return left.offset < right.offset;
              });
    descriptors_.assign(image_.files.size(), -1);
    cached_limit_ = descriptor_budget();
}

std::shared_ptr<DirectoryStorage> DirectoryStorage::open(const std::filesystem::path& root,
                                                         std::string& error) {
    ImageLayout layout;
    auto image = build_virtual(root, layout, error);
    if (!image) return nullptr;
    return std::shared_ptr<DirectoryStorage>(
        new DirectoryStorage(root, std::move(*image)));
}

std::uint64_t DirectoryStorage::size() const noexcept {
    return image_.size;
}

DirectoryStorage::~DirectoryStorage() {
    for (const auto descriptor : descriptors_)
        if (descriptor >= 0) ::close(descriptor);
}

int DirectoryStorage::descriptor(std::size_t index, bool& temporary) {
    temporary = false;
    {
        const std::lock_guard lock(mutex_);
        if (descriptors_[index] >= 0) return descriptors_[index];
    }
    const int opened = ::open((root_ / image_.files[index].path).c_str(), O_RDONLY | O_CLOEXEC);
    if (opened < 0) return -1;
    const std::lock_guard lock(mutex_);
    if (descriptors_[index] >= 0) {  // another reader opened it meanwhile
        ::close(opened);
        return descriptors_[index];
    }
    if (cached_ >= cached_limit_) {
        temporary = true;
        return opened;
    }
    descriptors_[index] = opened;
    ++cached_;
    return opened;
}

std::optional<std::size_t> DirectoryStorage::read(std::uint64_t offset,
                                                  std::span<std::byte> destination) {
    if (offset >= image_.size || destination.empty()) return std::size_t{0};
    const auto end = std::min<std::uint64_t>(offset + destination.size(), image_.size);
    const auto wanted = static_cast<std::size_t>(end - offset);
    // The header and metadata tables live in memory and are served first; the
    // file data region follows. Padding between files reads as zero, as an
    // image does.
    std::fill(destination.begin(), destination.begin() + static_cast<std::ptrdiff_t>(wanted),
              std::byte{0});
    if (offset < image_.data_offset) {
        const auto table_end = std::min<std::uint64_t>(end, image_.data_offset);
        const auto table_count = static_cast<std::size_t>(table_end - offset);
        std::memcpy(destination.data(), image_.tables.data() + offset, table_count);
    }
    auto file = std::lower_bound(image_.files.begin(), image_.files.end(), offset,
                                 [](const VirtualFile& entry, std::uint64_t at) {
                                     return entry.offset + entry.size <= at;
                                 });
    for (; file != image_.files.end() && file->offset < end; ++file) {
        const auto file_begin = std::max(offset, file->offset);
        const auto file_end = std::min(end, file->offset + file->size);
        if (file_end <= file_begin) continue;
        bool temporary = false;
        const int fd = descriptor(static_cast<std::size_t>(file - image_.files.begin()), temporary);
        if (fd < 0) return std::nullopt;
        auto count = static_cast<std::size_t>(file_end - file_begin);
        auto* into = reinterpret_cast<char*>(destination.data() + (file_begin - offset));
        auto at = static_cast<off_t>(file_begin - file->offset);
        bool complete = true;
        while (count != 0) {
            const auto got = ::pread(fd, into, count, at);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) { complete = false; break; }
            into += got; at += got; count -= static_cast<std::size_t>(got);
        }
        if (temporary) ::close(fd);
        if (!complete) return std::nullopt;
    }
    return wanted;
}

std::optional<std::size_t> MemoryStorage::read(std::uint64_t offset,
                                               std::span<std::byte> destination) {
    if (offset >= bytes_.size() || destination.empty()) return std::size_t{0};
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(destination.size(), bytes_.size() - offset));
    std::memcpy(destination.data(), bytes_.data() + offset, count);
    return count;
}

}  // namespace armigo::romfs
