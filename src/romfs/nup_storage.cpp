// ============================================================================
// Armigo RomFS - Directory-backed virtual storage
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/romfs/nup_storage.h>
#include <limits>
#include <map>

namespace armigo::romfs {
std::shared_ptr<NupStorage> NupStorage::open(std::shared_ptr<loader::NupArchive> archive,
                                          const loader::NupSelection& selection,
                                          std::string& error) {
    try {
        if (!archive) { error = "A NUP archive is required."; return {}; }
        auto result = std::make_shared<NupStorage>();
        result->archive_ = std::move(archive);
        detail::Node tree;
        tree.directory = true;
        std::map<std::string, detail::Node*> directories{{"", &tree}};
        std::uint64_t metadata_bytes = 0, payload_bytes = 0;
        auto add = [&](const std::string& path, bool directory, std::uint64_t size) {
            const auto slash = path.rfind('/');
            const auto parent = slash == std::string::npos ? "" : path.substr(0, slash);
            const auto name = slash == std::string::npos ? path : path.substr(slash + 1);
            auto owner = directories.find(parent);
            if (owner == directories.end() || name.empty() || name == "." || name == ".." ||
                directories.contains(path)) return false;
            metadata_bytes += name.size() + 64;
            if (metadata_bytes > (std::uint64_t{1} << 30) ||
                size > std::numeric_limits<std::uint64_t>::max() - payload_bytes - 15) return false;
            if (!directory) payload_bytes = align_up(payload_bytes + size, 16);
            auto node = std::make_unique<detail::Node>();
            node->name = name; node->directory = directory; node->size = size;
            if (directory && !directories.emplace(path, node.get()).second) return false;
            owner->second->children.push_back(std::move(node));
            return true;
        };
        for (const auto& directory : selection.romfs_directories)
            if (!add(directory, true, 0)) { error = "NUP RomFS directory layout is invalid or too large."; return {}; }
        for (const auto& [path, index] : selection.romfs) {
            if (index >= result->archive_->files().size()) { error = "Invalid NUP RomFS member."; return {}; }
            const auto& file = result->archive_->files()[index];
            if (file.codec != 0) { error = "Zstandard NUP RomFS extents are not implemented."; return {}; }
            if (!add(path, false, file.size)) { error = "NUP RomFS file layout is invalid or too large."; return {}; }
        }
        for (auto& [path, directory] : directories)
            std::sort(directory->children.begin(), directory->children.end(),
                [](const auto& a, const auto& b) { return a->name < b->name; });
        if (payload_bytes > std::numeric_limits<std::uint64_t>::max() - metadata_bytes - 4096) {
            error = "The NUP RomFS image size overflows.";
            return {};
        }
        ImageLayout layout;
        result->image_ = build_virtual_tree(tree, layout);
        std::sort(result->image_.files.begin(), result->image_.files.end(),
            [](const auto& a, const auto& b) { return a.offset < b.offset; });
        for (const auto& file : result->image_.files) result->members_.push_back(selection.romfs.at(file.path));
        return result;
    } catch (const std::bad_alloc&) {
        error = "NUP RomFS metadata exceeds available host memory.";
        return {};
    }
}

std::optional<std::size_t> NupStorage::read(std::uint64_t offset, std::span<std::byte> output) {
    if (offset >= image_.size || output.empty()) return std::size_t{0};
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(output.size(), image_.size - offset));
    const auto end = offset + count;
    std::fill_n(output.begin(), count, std::byte{});
    if (offset < image_.data_offset) {
        const auto table_size = static_cast<std::size_t>(std::min(end, image_.data_offset) - offset);
        std::memcpy(output.data(), image_.tables.data() + offset, table_size);
    }
    auto file = std::lower_bound(image_.files.begin(), image_.files.end(), offset,
        [](const auto& file, std::uint64_t at) { return file.offset + file.size <= at; });
    for (; file != image_.files.end() && file->offset < end; ++file) {
        const auto begin = std::max(offset, file->offset), stop = std::min(end, file->offset + file->size);
        if (begin == stop) continue;
        const auto member = members_[file - image_.files.begin()];
        auto read = archive_->read(member, begin - file->offset, output.subspan(begin - offset, stop - begin));
        if (!std::holds_alternative<std::size_t>(read) || std::get<std::size_t>(read) != stop - begin)
            return std::nullopt;
    }
    return count;
}
} // namespace armigo::romfs
