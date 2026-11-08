// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/nup.h>
#include "sha256.h"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace armigo::loader {
namespace {
constexpr std::uint32_t absent = ~std::uint32_t{0};
constexpr std::uint64_t max_toc = 256ull * 1024 * 1024;
constexpr std::uint64_t max_metadata = 16ull * 1024 * 1024;
bool range(std::uint64_t offset, std::uint64_t length, std::uint64_t size) {
    return offset <= size && length <= size - offset;
}
std::uint64_t le(std::span<const std::byte> bytes, std::size_t pos, unsigned count) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i)
        value |= std::uint64_t{std::to_integer<unsigned char>(bytes[pos + i])} << (8 * i);
    return value;
}
bool read_at(int fd, std::uint64_t offset, std::span<std::byte> output) {
    while (!output.empty()) {
        const auto count = pread(fd, output.data(), std::min(output.size(), std::size_t{1} << 20),
                                 static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        output = output.subspan(static_cast<std::size_t>(count));
        offset += count;
    }
    return true;
}
bool component(const std::string& name) {
    return !name.empty() && name != "." && name != ".." &&
        name.find_first_of("/\\") == std::string::npos && name.find('\0') == std::string::npos;
}
NupFailure bad(std::string operation, std::string detail) {
    return {"nup." + std::move(operation), std::move(detail)};
}
} // namespace

NupArchive::~NupArchive() { if (descriptor_ >= 0) close(descriptor_); }

std::variant<NupFailure, std::shared_ptr<NupArchive>> NupArchive::open(
    const std::filesystem::path& path) {
    try {
        auto archive = std::shared_ptr<NupArchive>(new NupArchive);
        archive->descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat info{};
        if (archive->descriptor_ < 0 || fstat(archive->descriptor_, &info) != 0 ||
            !S_ISREG(info.st_mode) || info.st_size < 0)
            return bad("open", "The archive is not a readable regular file.");
        archive->size_ = static_cast<std::uint64_t>(info.st_size);
        if (auto failure = archive->parse()) return *failure;
        return archive;
    } catch (const std::bad_alloc&) {
        return bad("allocation", "Archive metadata exceeds available host memory.");
    }
}

std::optional<NupFailure> NupArchive::parse() {
    std::array<std::byte, 0x80> header{};
    if (!read_at(descriptor_, 0, header)) return bad("header", "The header is truncated.");
    const auto toc_offset = le(header, 8, 8), toc_size = le(header, 16, 8);
    const auto blob = le(header, 0x38, 8);
    if (le(header, 0, 4) != 0x0150554e || le(header, 4, 2) != 0x80 ||
        le(header, 6, 2) != 1 || le(header, 0x4c, 4) != 0)
        return bad("header", "Unsupported archive magic, header version, size or flags.");
    if (le(header, 0x40, 8) != size_ || toc_offset < header.size() ||
        !range(toc_offset, toc_size, blob) || blob > size_ || blob % 16 ||
        toc_size < 0x30 || toc_size > max_toc)
        return bad("bounds", "The archive or table bounds are invalid.");
    std::vector<std::byte> toc(toc_size);
    if (!read_at(descriptor_, toc_offset, toc)) return bad("toc", "The table is truncated.");
    const auto hash = detail::sha256(toc);
    if (!hash || !std::equal(hash->begin(), hash->end(), header.begin() + 0x18))
        return bad("toc_hash", "The table SHA-256 does not match.");
    const auto packages = le(toc, 4, 4), dirs = le(toc, 8, 8), files = le(toc, 16, 8);
    const auto pool = le(toc, 24, 8), pool_size = le(toc, 32, 8);
    if (le(toc, 0, 4) != 0x01434f54 || packages == 0 ||
        packages != le(header, 0x48, 4) || packages > toc_size / 0x40 ||
        dirs > toc_size / 0x18 || files > toc_size / 0x50)
        return bad("counts", "Invalid table magic or record counts.");
    const auto records_end = 0x30 + packages * 0x40 + dirs * 0x18 + files * 0x50;
    if (records_end > pool || !range(pool, pool_size, toc_size) || (pool - records_end) % 4)
        return bad("pool", "The record arrays or string pool are out of bounds.");
    const auto children = (pool - records_end) / 4;
    auto name_at = [&](std::size_t pos, std::string& result) {
        const auto offset = le(toc, pos, 4), length = le(toc, pos + 4, 4);
        if (!range(offset, length, pool_size) || length > 255) return false;
        result.assign(reinterpret_cast<const char*>(toc.data() + pool + offset), length);
        return result.empty() || component(result);
    };
    std::vector<std::pair<std::uint64_t, std::uint64_t>> metadata_ranges;
    for (std::size_t i = 0, pos = 0x30; i < packages; ++i, pos += 0x40) {
        NupPackage package;
        package.title_id = le(toc, pos, 8);
        package.root = le(toc, pos + 8, 4);
        package.exefs = le(toc, pos + 12, 4);
        package.romfs = le(toc, pos + 16, 4);
        package.version = le(toc, pos + 20, 4);
        package.kind = le(toc, pos + 24, 1);
        const auto flags = le(toc, pos + 25, 1);
        package.standalone = flags & 1;
        if (package.kind > 2 || flags > 1 || package.root >= dirs ||
            (package.exefs != absent && package.exefs >= dirs) ||
            (package.romfs != absent && package.romfs >= dirs))
            return bad("package", "A package has invalid kind, flags or directory indices.");
        const auto offset = le(toc, pos + 32, 8), length = le(toc, pos + 40, 4);
        if (length == 0 && offset != 0)
            return bad("metadata", "Empty metadata has a nonzero offset.");
        if (length) {
            if (length > max_metadata || offset < toc_offset + toc_size ||
                !range(offset, length, size_))
                return bad("metadata", "The package metadata bounds are invalid.");
            metadata_ranges.emplace_back(offset, offset + length);
            std::vector<std::byte> bytes(length);
            if (!read_at(descriptor_, offset, bytes)) return bad("metadata", "Metadata is truncated.");
            for (std::size_t at = 0; at < bytes.size();) {
                if (!range(at, 8, bytes.size())) return bad("metadata", "A TLV header is truncated.");
                const auto type = le(bytes, at, 2), count = le(bytes, at + 4, 4);
                if (le(bytes, at + 2, 2) != 0 || !range(at + 8, count, bytes.size()))
                    return bad("metadata", "A TLV payload is invalid.");
                at += 8;
                if (type == 1) {
                    if (!package.npdm.empty()) return bad("metadata", "Duplicate NPDM metadata.");
                    package.npdm.assign(bytes.begin() + at, bytes.begin() + at + count);
                }
                at += count;
            }
        }
        packages_.push_back(std::move(package));
    }
    directories_.resize(dirs);
    std::vector<std::uint32_t> parents(dirs + files, 0);
    std::size_t pos = 0x30 + packages * 0x40;
    for (auto& directory : directories_) {
        const auto first = le(toc, pos + 8, 4), count = le(toc, pos + 12, 4);
        if (!name_at(pos, directory.name) || !range(first, count, children))
            return bad("directory", "A directory name or child range is invalid.");
        for (std::uint64_t slot = first; slot < first + count; ++slot) {
            const auto child = le(toc, records_end + slot * 4, 4);
            if (child >= dirs + files || ++parents[child] > 1)
                return bad("tree", "A child index is invalid or has multiple parents.");
            directory.children.push_back(child);
        }
        pos += 0x18;
    }
    names_.resize(files);
    files_.resize(files);
    for (std::size_t i = 0; i < files; ++i, pos += 0x50) {
        auto& file = files_[i];
        file.size = le(toc, pos + 8, 8);
        file.offset = le(toc, pos + 16, 8);
        file.stored_size = le(toc, pos + 24, 8);
        std::copy_n(toc.begin() + pos + 32, 32, file.hash.begin());
        file.codec = le(toc, pos + 64, 1);
        if (!name_at(pos, names_[i]) || !component(names_[i]) || file.codec > 1 ||
            le(toc, pos + 65, 1) != 0 ||
            (file.codec == 0 && file.size != file.stored_size) ||
            (file.size == 0 && (file.offset || file.stored_size)) ||
            (file.size && (file.offset < blob || file.offset % 16 || !file.stored_size ||
                          !range(file.offset, file.stored_size, size_))))
            return bad("extent", "A file name, codec or extent is invalid.");
        for (auto [start, end] : metadata_ranges)
            if (file.stored_size && file.offset < end && start < file.offset + file.stored_size)
                return bad("extent", "A file extent overlaps package metadata.");
    }
    std::vector<unsigned char> state(dirs, 0);
    std::function<bool(std::uint32_t, unsigned)> visit = [&](std::uint32_t index, unsigned depth) {
        if (depth > 128 || state[index] == 1) return false;
        if (state[index] == 2) return true;
        state[index] = 1;
        std::set<std::string> names;
        for (auto child : directories_[index].children) {
            const auto& name = child < dirs ? directories_[child].name : names_[child - dirs];
            if (!component(name) || !names.insert(name).second) return false;
            if (child < dirs && !visit(child, depth + 1)) return false;
        }
        state[index] = 2;
        return true;
    };
    // Walk package roots first so the depth limit covers the complete path.
    std::vector<std::uint32_t> owners(dirs, absent);
    for (std::uint32_t i = 0; i < packages_.size(); ++i) {
        const auto& package = packages_[i];
        if (parents[package.root] || !visit(package.root, 1))
            return bad("tree", "A package tree has a cycle, duplicate name or excessive depth.");
        std::function<bool(std::uint32_t)> own = [&](std::uint32_t dir) {
            if (owners[dir] != absent) return false;
            owners[dir] = i;
            for (auto child : directories_[dir].children)
                if (child < dirs && !own(child)) return false;
            return true;
        };
        if (!own(package.root) ||
            (package.exefs != absent && owners[package.exefs] != i) ||
            (package.romfs != absent && owners[package.romfs] != i))
            return bad("tree", "Package directory ownership is inconsistent.");
    }
    if (std::find(state.begin(), state.end(), 0) != state.end() ||
        std::find(parents.begin() + dirs, parents.end(), 0) != parents.end())
        return bad("tree", "The archive contains unreachable records.");
    return {};
}

void NupArchive::collect(std::uint32_t directory, const std::string& prefix,
                         std::map<std::string, std::uint32_t>& output) const {
    if (directory == absent) return;
    for (auto child : directories_[directory].children) {
        if (child < directories_.size()) {
            collect(child, prefix + directories_[child].name + "/", output);
        } else {
            const auto file = child - static_cast<std::uint32_t>(directories_.size());
            output[prefix + names_[file]] = file;
        }
    }
}

std::variant<NupFailure, NupSelection> NupArchive::select(std::uint64_t title_id) const {
    const NupPackage* base = nullptr;
    for (const auto& package : packages_) {
        if (package.kind != 0 || (title_id && package.title_id != title_id)) continue;
        if (base) return bad("selection", "Multiple base applications require an unambiguous title ID.");
        base = &package;
    }
    if (!base) return bad("selection", "A matching base application is required.");
    const NupPackage* update = nullptr;
    for (const auto& package : packages_) {
        if (package.kind != 1 || package.title_id != (base->title_id | 0x800)) continue;
        if (update && package.version == update->version)
            return bad("selection", "Multiple updates have the same version.");
        if (!update || package.version > update->version) update = &package;
    }
    const auto& active = update ? *update : *base;
    if (active.exefs == absent) return bad("selection", "The selected package has no ExeFS.");
    NupSelection result;
    result.title_id = base->title_id;
    result.version = active.version;
    result.update_applied = update != nullptr;
    result.npdm = active.npdm;
    collect(active.exefs, {}, result.exefs);
    auto apply_tree = [&](std::uint32_t root) {
        std::function<void(std::uint32_t, const std::string&)> apply =
            [&](std::uint32_t dir, const std::string& prefix) {
                if (dir == absent) return;
                for (const auto child : directories_[dir].children) {
                    if (child < directories_.size()) {
                        const auto path = prefix + directories_[child].name;
                        result.romfs.erase(path);
                        result.romfs_directories.insert(path);
                        apply(child, path + "/");
                    } else {
                        const auto index = child - static_cast<std::uint32_t>(directories_.size());
                        const auto path = prefix + names_[index];
                        // A file replaces the directory at that path, including empty directories.
                        result.romfs_directories.erase(path);
                        const auto below = path + "/";
                        for (auto it = result.romfs_directories.lower_bound(below);
                             it != result.romfs_directories.end() && it->starts_with(below);)
                            it = result.romfs_directories.erase(it);
                        for (auto it = result.romfs.lower_bound(below);
                             it != result.romfs.end() && it->first.starts_with(below);)
                            it = result.romfs.erase(it);
                        result.romfs[path] = index;
                    }
                }
            };
        apply(root, {});
    };
    if (!update || !update->standalone) apply_tree(base->romfs);
    if (update) apply_tree(update->romfs);
    return result;
}

std::variant<NupFailure, std::size_t> NupArchive::read(
    std::uint32_t index, std::uint64_t offset, std::span<std::byte> output) const {
    if (index >= files_.size()) return bad("file", "The file index is invalid.");
    const auto& file = files_[index];
    if (file.codec != 0) return bad("codec", "Zstandard NUP extents are not implemented.");
    const auto count = offset >= file.size ? 0 : std::min<std::uint64_t>(output.size(), file.size - offset);
    if (count && !read_at(descriptor_, file.offset + offset, output.first(count)))
        return bad("read", "The extent could not be read completely.");
    return static_cast<std::size_t>(count);
}

std::variant<NupFailure, std::vector<std::string>> NupArchive::zs_frame_mismatches(
    const NupSelection& selection) const {
    auto zs_name = [](const std::string& path) {
        const auto slash = path.rfind('/');
        const auto base = slash == std::string::npos ? std::string::size_type{0} : slash + 1;
        const auto dot = path.rfind('.');
        if (dot == std::string::npos || dot <= base || path.size() - dot != 3) return false;
        const auto lower = [](unsigned char c) {
            return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
        };
        return lower(static_cast<unsigned char>(path[dot + 1])) == 'z' &&
            lower(static_cast<unsigned char>(path[dot + 2])) == 's';
    };
    auto frame = [](std::span<const std::byte> head) {
        if (head.size() != 4) return false;
        const auto byte = [&](std::size_t index) {
            return std::to_integer<unsigned char>(head[index]);
        };
        if (byte(0) == 0x28 && byte(1) == 0xb5 && byte(2) == 0x2f && byte(3) == 0xfd) return true;
        return (byte(0) & 0xf0) == 0x50 && byte(1) == 0x2a && byte(2) == 0x4d && byte(3) == 0x18;
    };
    std::vector<std::string> damaged;
    for (const auto& [path, index] : selection.romfs) {
        if (!zs_name(path)) continue;
        if (index >= files_.size()) return bad("file", "The file index is invalid.");
        if (files_[index].size == 0) continue;
        std::array<std::byte, 4> head{};
        auto read_head = read(index, 0, head);
        if (const auto* failure = std::get_if<NupFailure>(&read_head)) return *failure;
        const auto count = std::get<std::size_t>(read_head);
        if (!frame(std::span<const std::byte>(head.data(), count))) damaged.push_back(path);
    }
    return damaged;
}

std::optional<NupFailure> NupArchive::verify(std::uint32_t index) const {
    if (index >= files_.size()) return bad("file", "The file index is invalid.");
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) return bad("hash", "SHA-256 initialization failed.");
    std::array<std::byte, 65536> buffer;
    const auto& file = files_[index];
    if (file.codec != 0) return bad("codec", "Zstandard NUP extents are not implemented.");
    for (std::uint64_t offset = 0; offset < file.size;) {
        auto result = read(index, offset, buffer);
        if (auto failure = std::get_if<NupFailure>(&result)) return *failure;
        const auto count = std::get<std::size_t>(result);
        if (!count || CC_SHA256_Update(&context, buffer.data(), static_cast<CC_LONG>(count)) != 1)
            return bad("hash", "SHA-256 input failed.");
        offset += count;
    }
    std::array<std::byte, 32> digest{};
    if (CC_SHA256_Final(reinterpret_cast<unsigned char*>(digest.data()), &context) != 1 || digest != file.hash)
        return bad("file_hash", "The member SHA-256 does not match.");
    return {};
}

std::variant<NupFailure, ExeFsDirectoryInput> NupArchive::read_exefs(
    const NupSelection& selection, const ExeFsReadLimits& limits) const {
    ExeFsDirectoryInput result;
    std::set<std::string> admitted{"rtld", "main", "sdk", "main.npdm"};
    bool gap = false;
    for (unsigned i = 0; i < 10; ++i) {
        const auto name = "subsdk" + std::to_string(i);
        if (!selection.exefs.contains(name)) gap = true;
        else if (gap && !limits.collect_all_subsdk) result.ignored_modules.push_back(name);
        else admitted.insert(name);
    }
    if (!selection.exefs.contains("main")) return bad("exefs", "The selected ExeFS has no main module.");
    std::uint64_t total = 0;
    for (const auto& [name, index] : selection.exefs) {
        if (!admitted.contains(name)) continue;
        if (index >= files_.size()) return bad("file", "The selected file index is invalid.");
        const auto& file = files_[index];
        if (file.size > limits.max_file_bytes || !range(total, file.size, limits.max_total_bytes))
            return bad("limits", "The selected ExeFS exceeds the collection limits.");
        total += file.size;
        std::vector<std::byte> bytes(file.size);
        auto read_result = read(index, 0, bytes);
        if (auto failure = std::get_if<NupFailure>(&read_result)) return *failure;
        const auto hash = detail::sha256(bytes);
        if (!hash || *hash != file.hash) return bad("file_hash", "The executable member SHA-256 does not match.");
        if (name == "main.npdm") result.process_metadata = std::move(bytes);
        else result.modules.push_back({name, std::move(bytes)});
    }
    if (result.process_metadata && !selection.npdm.empty() && *result.process_metadata != selection.npdm)
        return bad("metadata", "ExeFS and metadata NPDM bytes disagree.");
    if (!result.process_metadata && !selection.npdm.empty()) {
        if (selection.npdm.size() > limits.max_file_bytes ||
            !range(total, selection.npdm.size(), limits.max_total_bytes))
            return bad("limits", "The metadata NPDM exceeds the collection limits.");
        result.process_metadata = selection.npdm;
    }
    return result;
}

} // namespace armigo::loader
