// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "bundle.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string_view>
#include <system_error>

namespace armigo::pack {
namespace {

std::string xml_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char character : text) {
        switch (character) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += static_cast<char>(character); break;
        }
    }
    return out;
}

std::string json_escape(std::string_view text) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(text.size());
    for (const unsigned char character : text) {
        switch (character) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (character < 0x20) {
                    out += "\\u00";
                    out += kDigits[character >> 4];
                    out += kDigits[character & 0x0f];
                } else {
                    out += static_cast<char>(character);
                }
                break;
        }
    }
    return out;
}

bool write_file(const std::filesystem::path& path, std::string_view bytes, std::string& error) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        error = "Cannot write " + path.string() + ".";
        return false;
    }
    return true;
}

// The app localization that shows a Horizon language's title.
std::string_view localization(std::string_view language) {
    if (language == "en-US") return "en";
    if (language == "pt") return "pt-PT";
    if (language == "zh-TW") return "zh-Hant";
    if (language == "zh-CN") return "zh-Hans";
    for (const std::string_view code : {"en-GB", "ja", "fr", "de", "es-419", "es", "it", "nl", "fr-CA", "ru", "ko",
                                        "zh-Hant", "zh-Hans", "pt-BR"})
        if (language == code) return code;
    return {};
}

// The app's localizations, in the title's language order, with the Horizon
// language each one shows.
std::vector<std::pair<std::string, std::string>> app_localizations(const BundleSpec& spec) {
    std::vector<std::pair<std::string, std::string>> result;
    for (const std::string& language : spec.languages) {
        const auto name = localization(language);
        if (name.empty()) continue;
        const auto existing = std::find_if(result.begin(), result.end(),
                                           [&](const auto& entry) { return entry.first == name; });
        if (existing == result.end()) result.emplace_back(name, language);
    }
    return result;
}

std::string bundle_localizations(const BundleSpec& spec) {
    std::string result = "<key>CFBundleLocalizations</key><array>";
    for (const auto& [name, language] : app_localizations(spec)) {
        result += "<string>" + xml_escape(name) + "</string>";
    }
    result += "</array>\n";
    return result;
}

std::string strings_escape(std::string_view text) {
    std::string result;
    for (const char character : text) {
        if (character == '"' || character == '\\') result += '\\';
        if (character == '\n') {
            result += "\\n";
            continue;
        }
        result += character;
    }
    return result;
}

} // namespace

std::string info_plist(const BundleSpec& spec) {
    const auto display = xml_escape(spec.display_name);
    const auto name = xml_escape(spec.bundle_name);
    const auto identifier = xml_escape(spec.bundle_identifier);
    if (spec.platform == "ios") {
        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
               "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
               "<plist version=\"1.0\"><dict>\n"
               "<key>CFBundleDevelopmentRegion</key><string>en</string>\n" + bundle_localizations(spec) +
               "<key>CFBundleExecutable</key><string>ArmigoPlayer</string>\n"
               "<key>CFBundleIdentifier</key><string>" + identifier + "</string>\n"
               "<key>CFBundleName</key><string>" + name + "</string>\n"
               "<key>CFBundleDisplayName</key><string>" + display + "</string>\n"
               "<key>CFBundlePackageType</key><string>APPL</string>\n"
               "<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>\n"
               "<key>CFBundleVersion</key><string>1</string>\n"
               "<key>CFBundleShortVersionString</key><string>0.1.0</string>\n"
               "<key>MinimumOSVersion</key><string>26.0</string>\n"
               "<key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>\n"
               "<key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>\n"
               "<key>UIRequiredDeviceCapabilities</key><array><string>arm64</string><string>metal</string></array>\n"
               "<key>UISupportedInterfaceOrientations</key><array><string>UIInterfaceOrientationLandscapeLeft</string>"
               "<string>UIInterfaceOrientationLandscapeRight</string></array>\n"
               "<key>UIRequiresFullScreen</key><true/>\n"
               "<key>LSApplicationCategoryType</key><string>public.app-category.games</string>\n"
               "<key>LSSupportsGameMode</key><true/>\n"
               "<key>GCSupportsGameMode</key><true/>\n"
               "<key>UILaunchScreen</key><dict/>\n"
               "<key>UIApplicationSceneManifest</key><dict><key>UIApplicationSupportsMultipleScenes</key><false/></dict>\n"
               "<key>GCSupportsControllerUserInteraction</key><true/>\n"
               "<key>GCSupportedGameControllers</key><array><dict><key>ProfileName</key><string>ExtendedGamepad</string></dict></array>\n"
               "<key>CFBundleIconName</key><string>AppIcon</string>\n"
               "<key>CFBundleIconFiles</key><array><string>AppIcon</string></array>\n"
               "<key>UIFileSharingEnabled</key><true/>\n"
               "</dict></plist>\n";
    }
    return std::string(
               "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
               "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
               "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
               "<plist version=\"1.0\">\n"
               "<dict>\n"
               "    <key>CFBundleDevelopmentRegion</key>\n"
               "    <string>en</string>\n"
               "    <key>CFBundleExecutable</key>\n"
               "    <string>Armigo</string>\n"
               "    <key>CFBundleIdentifier</key>\n"
               "    <string>") +
           identifier +
           "</string>\n" + bundle_localizations(spec) +
           "    <key>CFBundleInfoDictionaryVersion</key>\n"
           "    <string>6.0</string>\n"
           "    <key>CFBundleName</key>\n"
           "    <string>" +
           name +
           "</string>\n"
           "    <key>CFBundleDisplayName</key>\n"
           "    <string>" +
           display +
           "</string>\n"
           "    <key>LSHasLocalizedDisplayName</key>\n"
           "    <true/>\n"
           "    <key>CFBundleIconFile</key>\n"
           "    <string>AppIcon</string>\n"
           "    <key>CFBundlePackageType</key>\n"
           "    <string>APPL</string>\n"
           "    <key>CFBundleShortVersionString</key>\n"
           "    <string>0.1.0</string>\n"
           "    <key>CFBundleVersion</key>\n"
           "    <string>0.1.0</string>\n"
           "    <key>LSMinimumSystemVersion</key>\n"
           "    <string>26.0</string>\n"
           "    <key>LSApplicationCategoryType</key>\n"
           "    <string>public.app-category.games</string>\n"
           "    <key>LSSupportsGameMode</key>\n"
           "    <true/>\n"
           "    <key>GCSupportsGameMode</key>\n"
           "    <true/>\n"
           "    <key>GCSupportsControllerUserInteraction</key>\n"
           "    <true/>\n"
           "    <key>NSHighResolutionCapable</key>\n"
           "    <true/>\n"
           "    <key>NSPrincipalClass</key>\n"
           "    <string>NSApplication</string>\n"
           "</dict>\n"
           "</plist>\n";
}

std::string settings_bundle(const BundleSpec&) {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        "<plist version=\"1.0\"><dict>\n"
        "<key>StringsTable</key><string>Root</string>\n"
        "<key>PreferenceSpecifiers</key><array>\n"
        "<dict><key>Type</key><string>PSGroupSpecifier</string><key>Title</key><string>Display</string></dict>\n"
        "<dict><key>Type</key><string>PSMultiValueSpecifier</string><key>Title</key><string>Console Mode</string>"
        "<key>Key</key><string>console_mode</string><key>DefaultValue</key><string>docked</string>"
        "<key>Titles</key><array><string>Docked</string><string>Handheld</string></array>"
        "<key>Values</key><array><string>docked</string><string>handheld</string></array></dict>\n"
        "<dict><key>Type</key><string>PSMultiValueSpecifier</string><key>Title</key><string>Resolution</string>"
        "<key>Key</key><string>resolution</string><key>DefaultValue</key><string>auto</string>"
        "<key>Titles</key><array><string>Auto (1080p)</string><string>720p</string><string>900p</string>"
        "<string>1080p</string><string>1440p</string><string>Native</string></array>"
        "<key>Values</key><array><string>auto</string><string>720p</string><string>900p</string>"
        "<string>1080p</string><string>1440p</string><string>native</string></array></dict>\n"
        "<dict><key>Type</key><string>PSMultiValueSpecifier</string><key>Title</key><string>Frame Rate</string>"
        "<key>Key</key><string>frame_rate</string><key>DefaultValue</key><integer>60</integer>"
        "<key>Titles</key><array><string>Auto</string><string>30 fps</string><string>60 fps</string></array>"
        "<key>Values</key><array><integer>0</integer><integer>30</integer><integer>60</integer></array></dict>\n"
        "<dict><key>Type</key><string>PSGroupSpecifier</string><key>Title</key><string>Overlay</string>"
        "<key>FooterText</key><string>Game language follows this app's language in system Settings. "
        "Changes take effect the next time the game starts.</string></dict>\n"
        "<dict><key>Type</key><string>PSToggleSwitchSpecifier</string><key>Title</key><string>Performance Overlay</string>"
        "<key>Key</key><string>performance_overlay</string><key>DefaultValue</key><false/></dict>\n"
        "</array></dict></plist>\n";
}

std::vector<std::pair<std::string, std::string>> localized_names(const BundleSpec& spec) {
    std::vector<std::pair<std::string, std::string>> result;
    for (const auto& [name, language] : app_localizations(spec)) {
        std::string title = spec.display_name;
        for (const auto& [code, localized] : spec.localized_titles)
            if (localization(code) == name && !localized.empty()) title = localized;
        const auto escaped = strings_escape(title);
        result.emplace_back(name + ".lproj",
            "\"CFBundleDisplayName\" = \"" + escaped + "\";\n\"CFBundleName\" = \"" + escaped + "\";\n");
    }
    return result;
}

std::string language_manifest(const BundleSpec& spec) {
    char mask[11];
    std::snprintf(mask, sizeof(mask), "0x%08x", spec.supported_language_flag);
    std::string result = std::string(mask) + "\n";
    for (const std::string& language : spec.languages) result += language + "\n";
    return result;
}

std::string pack_json(const BundleSpec& spec) {
    return std::string("{\n"
                       "  \"title_id\": \"") +
           json_escape(spec.title_id) +
           "\",\n"
           "  \"build_id\": \"" +
           json_escape(spec.build_id) +
           "\",\n"
           "  \"title_version\": " +
           std::to_string(spec.title_version) +
           ",\n"
           "  \"platform\": \"" +
           json_escape(spec.platform) +
           "\",\n"
           "  \"commit\": \"" +
           json_escape(spec.commit) +
           "\"\n"
           "}\n";
}

namespace {

bool write_localized_names(const BundleSpec& spec, const std::filesystem::path& resources, std::string& error) {
    for (const auto& [directory, strings] : localized_names(spec)) {
        std::error_code created;
        std::filesystem::create_directories(resources / directory, created);
        if (created) {
            error = "Cannot create " + (resources / directory).string() + ".";
            return false;
        }
        if (!write_file(resources / directory / "InfoPlist.strings", strings, error)) return false;
    }
    return true;
}

} // namespace

std::variant<std::string, std::filesystem::path> assemble_application(const BundleSpec& spec) {
    std::error_code error;
    if (spec.icon.empty()) {
        return std::string("The title has no icon.");
    }
    if (!std::filesystem::is_regular_file(spec.player, error)) {
        return "The linker-signed player is missing: " + spec.player.string() + ".";
    }
    if (!std::filesystem::is_directory(spec.title_folder, error)) {
        return "The matched title settings are missing: " + spec.title_folder.string() + ".";
    }
    const bool ios = spec.platform == "ios";
    const auto contents = ios ? spec.app_path : spec.app_path / "Contents";
    const auto executable_directory = ios ? spec.app_path : contents / "MacOS";
    const auto resources = ios ? spec.app_path : contents / "Resources";
    const auto armigo = resources / "Armigo";
    const auto title = armigo / "title";
    std::filesystem::create_directories(executable_directory, error);
    if (error) {
        return "Cannot create " + executable_directory.string() + ".";
    }
    std::filesystem::create_directories(title, error);
    if (error) {
        return "Cannot create " + title.string() + ".";
    }
    if (ios) {
        std::filesystem::create_directories(resources / "Settings.bundle", error);
        if (error) return std::string("Cannot create the iOS Settings bundle.");
    }
    std::string write_error;
    if (!write_file(contents / "Info.plist", info_plist(spec), write_error) ||
        !write_file(armigo / "pack.json", pack_json(spec), write_error) ||
        !write_file(armigo / "languages.txt", language_manifest(spec), write_error) ||
        (ios && !write_file(resources / "Settings.bundle" / "Root.plist", settings_bundle(spec), write_error)) ||
        !write_localized_names(spec, resources, write_error) ||
        !write_file(resources / (ios ? "AppIcon.png" : "AppIcon.icns"),
                    std::string_view(reinterpret_cast<const char*>(spec.icon.data()), spec.icon.size()),
                    write_error)) {
        return write_error;
    }
    if (ios && !write_file(spec.app_path.parent_path() / "entitlements.plist",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        "<plist version=\"1.0\"><dict>\n"
        "<key>com.apple.developer.kernel.increased-memory-limit</key><true/>\n"
        "<key>com.apple.developer.kernel.extended-virtual-addressing</key><true/>\n"
        "<key>com.apple.developer.sustained-execution</key><true/>\n"
        "</dict></plist>\n", write_error)) {
        return write_error;
    }
    const auto executable = executable_directory / (ios ? "ArmigoPlayer" : "Armigo");
    std::filesystem::copy_file(spec.player, executable, std::filesystem::copy_options::overwrite_existing,
                               error);
    if (error) {
        return "Cannot copy the player to " + executable.string() + ".";
    }
    std::filesystem::permissions(executable,
                                 std::filesystem::perms::owner_all | std::filesystem::perms::group_read |
                                     std::filesystem::perms::group_exec | std::filesystem::perms::others_read |
                                     std::filesystem::perms::others_exec,
                                 std::filesystem::perm_options::replace, error);
    if (error) {
        return std::string("Cannot mark the player executable.");
    }
    std::filesystem::copy(spec.title_folder, title,
                          std::filesystem::copy_options::recursive |
                              std::filesystem::copy_options::overwrite_existing |
                              std::filesystem::copy_options::skip_symlinks,
                          error);
    if (error) {
        return std::string("Cannot copy title settings into the app.");
    }
    return spec.app_path;
}

} // namespace armigo::pack
