// ============================================================================
// Armigo Compatibility - Title Settings
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/compat/title_settings.h>

#import <Foundation/Foundation.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string_view>

namespace armigo::compat {
namespace {

std::filesystem::path user_directory(NSSearchPathDirectory kind) {
    @autoreleasepool {
        NSArray<NSString*>* paths = NSSearchPathForDirectoriesInDomains(kind, NSUserDomainMask, YES);
        if (paths.count == 0) return {};
        return std::filesystem::path(paths.firstObject.fileSystemRepresentation);
    }
}

// The title roots that hold <TitleID>/ folders, in search order.
std::vector<std::filesystem::path> title_roots() {
    if (const char* root = std::getenv("ARMIGO_TITLES"); root && *root) {
        if (std::string_view(root) == "off") return {};
        return {std::filesystem::path(root)};
    }
    std::vector<std::filesystem::path> roots;
    @autoreleasepool {
        NSBundle* bundle = NSBundle.mainBundle;
        if (NSString* executable = bundle.executablePath)
            roots.push_back(std::filesystem::path(executable.fileSystemRepresentation).parent_path() / "titles");
        if (NSString* resources = bundle.resourcePath) {
            auto in_resources = std::filesystem::path(resources.fileSystemRepresentation) / "titles";
            if (roots.empty() || roots.front() != in_resources) roots.push_back(std::move(in_resources));
        }
    }
    return roots;
}

} // namespace

std::string title_id_text(std::uint64_t title_id) {
    char text[17];
    std::snprintf(text, sizeof text, "%016llX", static_cast<unsigned long long>(title_id));
    return text;
}

std::optional<std::uint64_t> parse_title_id_text(const std::string& text) {
    if (text.size() != 16) return std::nullopt;
    for (const char digit : text)
        if (!((digit >= '0' && digit <= '9') || (digit >= 'A' && digit <= 'F'))) return std::nullopt;
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return std::nullopt;
    return value;
}

std::variant<std::string, TitleSettings> read_title_settings(const std::filesystem::path& folder,
                                                             std::uint64_t title_id) {
    const auto profile = folder / "profile.json";
    std::error_code error;
    if (!std::filesystem::is_regular_file(profile, error))
        return "Title settings " + folder.string() + " have no profile.json.";
    auto read = read_patch_descriptions(profile);
    if (auto* failure = std::get_if<std::string>(&read))
        return "Title settings " + profile.string() + ": " + *failure;
    TitleSettings settings{folder, std::get<PatchDescriptions>(std::move(read))};
    if (settings.patches.empty()) return "Title settings " + profile.string() + " describe no build.";
    for (const auto& patch : settings.patches)
        if (patch.identity.title_id != title_id)
            return "Title settings " + profile.string() + " describe title " +
                title_id_text(patch.identity.title_id) + ", not " + title_id_text(title_id) + ".";
    return settings;
}

std::filesystem::path packed_resources() {
    @autoreleasepool {
        NSString* resources = NSBundle.mainBundle.resourcePath;
        if (!resources) return {};
        auto armigo = std::filesystem::path(resources.fileSystemRepresentation) / "Armigo";
        std::error_code error;
        if (!std::filesystem::is_directory(armigo, error)) return {};
        return armigo;
    }
}

TitleSettingsResult find_title_settings(std::uint64_t title_id) {
    std::error_code error;
    if (const auto packed = packed_resources(); !packed.empty()) {
        // A packed app carries the settings of the title it was packed for;
        // another title run from it looks in the title roots instead.
        const auto folder = packed / "title";
        auto read = read_patch_descriptions(folder / "profile.json");
        if (const auto* failure = std::get_if<std::string>(&read);
            failure && std::filesystem::is_directory(folder, error))
            return "Packed title settings " + folder.string() + ": " + *failure;
        if (const auto* patches = std::get_if<PatchDescriptions>(&read);
            patches && !patches->empty() && patches->front().identity.title_id == title_id) {
            auto settings = read_title_settings(folder, title_id);
            if (auto* failure = std::get_if<std::string>(&settings)) return *failure;
            return std::optional(std::get<TitleSettings>(std::move(settings)));
        }
    }
    const auto name = title_id_text(title_id);
    for (const auto& root : title_roots()) {
        const auto folder = root / name;
        if (!std::filesystem::is_directory(folder, error)) continue;
        auto read = read_title_settings(folder, title_id);
        if (auto* failure = std::get_if<std::string>(&read)) return *failure;
        return std::optional(std::get<TitleSettings>(std::move(read)));
    }
    return std::optional<TitleSettings>{};
}

std::filesystem::path title_save_directory(std::uint64_t title_id) {
    const auto documents = user_directory(NSDocumentDirectory);
    if (documents.empty()) return {};
    return documents / "Armigo" / "Saves" / title_id_text(title_id);
}

std::optional<std::string> prepare_save_directory(
    const std::filesystem::path& saves, const std::filesystem::path& legacy) {
    if (saves.empty()) return "The save folder is unavailable.";
    std::error_code error;
    const bool exists = std::filesystem::exists(saves, error);
    if (error) return "The save folder could not be checked.";
    if (exists) {
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(saves, error)) || error ||
            !std::filesystem::is_directory(saves, error) || error)
            return "The save folder is unavailable.";
        return std::nullopt;
    }
    // A maintenance client uploads and verifies a complete sibling staging
    // folder, then publishes this small marker last. Partial uploads stay inert.
    const auto staging = std::filesystem::path(saves.string() + ".import");
    const auto ready = std::filesystem::path(saves.string() + ".import-ready");
    const auto ready_status = std::filesystem::symlink_status(ready, error);
    if (error == std::errc::no_such_file_or_directory) error.clear();
    if (error) return "The saved data import could not be checked.";
    if (std::filesystem::exists(ready_status)) {
        if (!std::filesystem::is_regular_file(ready_status) ||
            std::filesystem::file_size(ready, error) != 31 || error)
            return "The saved data import is incomplete.";
        std::ifstream input(ready, std::ios::binary);
        std::string marker(31, '\0');
        if (!input.read(marker.data(), marker.size()) || marker != "Armigo verified save import v1\n")
            return "The saved data import is incomplete.";
        const auto status = std::filesystem::symlink_status(staging, error);
        if (error || !std::filesystem::is_directory(status))
            return "The saved data import is incomplete.";
        if (::renamex_np(staging.c_str(), saves.c_str(), RENAME_EXCL) != 0)
            return "The saved data import could not be completed. Existing saves were preserved.";
        std::filesystem::remove(ready, error);
        if (error) return "Saved data was imported, but import cleanup could not finish.";
        return std::nullopt;
    }
    const bool has_legacy = std::filesystem::exists(legacy, error);
    if (error) return "Existing saved data could not be checked.";
    if (has_legacy) return "Your existing saved data needs to be imported before you can play.";
    std::filesystem::create_directories(saves, error);
    if (error) return "The save folder could not be created.";
    return std::nullopt;
}

std::filesystem::path title_cache_directory(std::uint64_t title_id) {
    const auto caches = user_directory(NSCachesDirectory);
    if (caches.empty()) return {};
    return caches / "Armigo" / title_id_text(title_id);
}

} // namespace armigo::compat
