// ============================================================================
// Armigo Runtime - Native Process Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/addon_content.h>

#include <armigo/romfs/romfs_virtual.h>

#include <algorithm>
#include <optional>
#include <string>

namespace armigo::runtime {
namespace {

LaunchFailure invalid(const std::filesystem::path& directory, const std::string& detail) {
    LaunchFailure failure;
    failure.stage = LaunchStage::inputs;
    failure.code = LaunchError::invalid_input;
    failure.operation = "open-addon-content";
    failure.detail = "Add-on content directory " + directory.string() + ": " + detail;
    return failure;
}

std::optional<std::uint64_t> data_id_from_name(const std::filesystem::path& directory) {
    auto named = directory;
    if (named.filename().empty()) named = named.parent_path();
    if (named.filename() == "romfs") named = named.parent_path();
    const auto name = named.filename().string();
    if (name.size() != 16) return std::nullopt;
    std::uint64_t value = 0;
    for (const char c : name) {
        unsigned digit = 0;
        if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') digit = static_cast<unsigned>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = static_cast<unsigned>(c - 'A' + 10);
        else return std::nullopt;
        value = value << 4 | digit;
    }
    return value;
}

}  // namespace

LaunchResult<std::shared_ptr<const services::AddOnContentCatalog>> open_addon_content(
    std::uint64_t application_id, std::span<const std::filesystem::path> directories) {
    auto catalog = std::make_shared<services::AddOnContentCatalog>();
    catalog->base_id = addon_content_base_id(application_id);
    for (const auto& directory : directories) {
        const auto data_id = data_id_from_name(directory);
        if (!data_id)
            return invalid(directory, "the directory name must be the add-on data id as 16 hexadecimal digits.");
        if (catalog->base_id < application_id || *data_id <= catalog->base_id || *data_id - catalog->base_id > 0xfff)
            return invalid(directory, "the data id does not belong to this application's add-on content.");
        if (catalog->find(*data_id))
            return invalid(directory, "the data id is given more than once.");
        std::string error;
        auto storage = romfs::DirectoryStorage::open(directory, error);
        if (!storage) return invalid(directory, error);
        catalog->items.push_back({*data_id, std::move(storage)});
    }
    std::sort(catalog->items.begin(), catalog->items.end(),
              [](const auto& a, const auto& b) { return a.data_id < b.data_id; });
    return std::shared_ptr<const services::AddOnContentCatalog>(std::move(catalog));
}

}  // namespace armigo::runtime
