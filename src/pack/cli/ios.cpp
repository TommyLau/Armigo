// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "ios.h"

#include <spawn.h>
#include <sys/clonefile.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>
#include <system_error>

extern char** environ;

namespace armigo::pack {
namespace {

namespace fs = std::filesystem;

struct RemoveOnFailure {
    fs::path path;
    bool keep = false;
    ~RemoveOnFailure() {
        if (!keep && !path.empty()) {
            std::error_code error;
            fs::remove_all(path, error);
        }
    }
};

std::string read_text(const fs::path& path, std::uintmax_t limit) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (error || size > limit) {
        return {};
    }
    std::string text(size, '\0');
    std::ifstream input(path, std::ios::binary);
    if (!input.read(text.data(), static_cast<std::streamsize>(text.size()))) {
        return {};
    }
    return text;
}

std::optional<std::string> json_string(std::string_view text, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    auto cursor = text.find(needle);
    if (cursor == std::string_view::npos) {
        return std::nullopt;
    }
    cursor = text.find(':', cursor + needle.size());
    if (cursor == std::string_view::npos) {
        return std::nullopt;
    }
    cursor = text.find('"', cursor + 1);
    if (cursor == std::string_view::npos) {
        return std::nullopt;
    }
    std::string value;
    for (++cursor; cursor < text.size(); ++cursor) {
        const char character = text[cursor];
        if (character == '"') {
            return value;
        }
        if (character == '\\') {
            if (++cursor >= text.size()) {
                return std::nullopt;
            }
            const char escaped = text[cursor];
            if (escaped != '"' && escaped != '\\' && escaped != '/') {
                return std::nullopt;
            }
            value += escaped;
        } else {
            if (static_cast<unsigned char>(character) < 0x20) {
                return std::nullopt;
            }
            value += character;
        }
    }
    return std::nullopt;
}

std::optional<std::uint32_t> json_number(std::string_view text, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    auto cursor = text.find(needle);
    if (cursor == std::string_view::npos) {
        return std::nullopt;
    }
    cursor = text.find(':', cursor + needle.size());
    if (cursor == std::string_view::npos) {
        return std::nullopt;
    }
    ++cursor;
    while (cursor < text.size() && (text[cursor] == ' ' || text[cursor] == '\t' || text[cursor] == '\n')) {
        ++cursor;
    }
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(text.data() + cursor, text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr == text.data() + cursor) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::string> cmake_cache_value(const fs::path& cache, std::string_view key) {
    std::ifstream input(cache);
    std::string line;
    const std::string prefix = std::string(key) + ":";
    while (std::getline(input, line)) {
        if (!line.starts_with(prefix)) {
            continue;
        }
        const auto equals = line.find('=');
        if (equals != std::string::npos) {
            return line.substr(equals + 1);
        }
    }
    return std::nullopt;
}

std::variant<std::string, PackedIdentity> identity_from_manifest(const fs::path& manifest) {
    const std::string text = read_text(manifest, 4096);
    if (text.empty()) {
        return "The packed app manifest is missing or too large: " + manifest.string() + ".";
    }
    const auto title = json_string(text, "title_id");
    const auto build = json_string(text, "build_id");
    const auto platform = json_string(text, "platform");
    const auto version = json_number(text, "title_version");
    if (!title || !build || !platform || !version || title->size() != 16 || build->size() != 64) {
        return "The packed app manifest has no valid title and build identity: " + manifest.string() + ".";
    }
    return PackedIdentity{*title, *build, *platform, *version};
}

std::string clone_directory(const fs::path& source, const fs::path& destination) {
    std::error_code error;
    if (!fs::is_directory(source, error) || fs::exists(destination, error)) {
        return "The reuse game directory is missing or its destination already exists.";
    }
    fs::create_directories(destination, error);
    if (error) {
        return "Cannot create the cloned game directory: " + destination.string() + ".";
    }
    fs::recursive_directory_iterator cursor(source, fs::directory_options::none, error);
    const fs::recursive_directory_iterator end;
    while (!error && cursor != end) {
        const auto relative = cursor->path().lexically_relative(source);
        const auto output = destination / relative;
        const auto status = cursor->symlink_status(error);
        if (error) {
            break;
        }
        if (fs::is_directory(status)) {
            fs::create_directory(output, error);
        } else if (fs::is_regular_file(status)) {
            if (clonefile(cursor->path().c_str(), output.c_str(), 0) != 0) {
                return "APFS clone failed for " + cursor->path().string() + ": " + std::strerror(errno) + ".";
            }
        } else {
            return "The reuse game contains an unsupported non-regular entry: " + cursor->path().string() + ".";
        }
        cursor.increment(error);
    }
    if (error) {
        return "Cannot enumerate the reuse game: " + source.string() + ".";
    }
    return {};
}

} // namespace

std::string run_pack_tool(const std::vector<std::string>& arguments, bool default_title_rate) {
    if (arguments.empty() || arguments.front().empty()) {
        return "Cannot run an empty pack tool command.";
    }
    std::vector<char*> pointers;
    pointers.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
        pointers.push_back(const_cast<char*>(argument.c_str()));
    }
    pointers.push_back(nullptr);
    std::vector<std::string> environment;
    std::vector<char*> environment_pointers;
    if (default_title_rate) {
        for (char** entry = environ; *entry; ++entry)
            if (!std::string_view(*entry).starts_with("ARMIGO_COMPAT_FRAME_RATE=")) environment.emplace_back(*entry);
        environment.emplace_back("ARMIGO_COMPAT_FRAME_RATE=0");
        for (auto& entry : environment) environment_pointers.push_back(entry.data());
        environment_pointers.push_back(nullptr);
    }
    pid_t child = 0;
    const int spawned = posix_spawnp(&child, pointers.front(), nullptr, nullptr, pointers.data(),
                                    default_title_rate ? environment_pointers.data() : environ);
    if (spawned != 0) {
        return "Cannot start " + arguments.front() + ": " + std::strerror(spawned) + ".";
    }
    int status = 0;
    pid_t waited = 0;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != child) {
        return "Cannot wait for " + arguments.front() + ".";
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return arguments.front() + " failed" +
               (WIFEXITED(status) ? " with exit code " + std::to_string(WEXITSTATUS(status)) : " without exiting") + ".";
    }
    return {};
}

std::variant<std::string, PackedIdentity> read_packed_identity(const fs::path& macos_app) {
    auto identity = identity_from_manifest(macos_app / "Contents" / "Resources" / "Armigo" / "pack.json");
    if (auto* parsed = std::get_if<PackedIdentity>(&identity); parsed && parsed->platform != "macos") {
        return "The reuse app is not a macOS pack: " + macos_app.string() + ".";
    }
    return identity;
}

std::variant<std::string, std::optional<fs::path>> select_macos_reuse(
    const fs::path& requested, const fs::path& output_directory, const fs::path& destination_app,
    const std::string& title_id) {
    if (!requested.empty()) {
        return std::optional<fs::path>(requested);
    }
    std::error_code error;
    std::vector<fs::path> matches;
    if (!fs::exists(output_directory, error)) {
        return std::optional<fs::path>{};
    }
    fs::directory_iterator cursor(output_directory, fs::directory_options::skip_permission_denied, error);
    const fs::directory_iterator end;
    while (!error && cursor != end) {
        const fs::path candidate = cursor->path();
        if (candidate.extension() == ".app" && candidate.lexically_normal() != destination_app.lexically_normal()) {
            auto identity = read_packed_identity(candidate);
            if (const auto* packed = std::get_if<PackedIdentity>(&identity); packed && packed->title_id == title_id) {
                matches.push_back(candidate);
            }
        }
        cursor.increment(error);
    }
    if (error) {
        return "Cannot search the output directory for a reusable macOS pack: " + output_directory.string() + ".";
    }
    if (matches.size() > 1) {
        return std::string("More than one macOS pack for this title is available; select one with --reuse.");
    }
    if (matches.empty()) {
        return std::optional<fs::path>{};
    }
    return std::optional<fs::path>(matches.front());
}

std::variant<std::string, PackedIdentity> clone_macos_game(
    const fs::path& macos_app, const fs::path& destination_game, const std::string& expected_title_id) {
    auto identity = read_packed_identity(macos_app);
    if (auto* failure = std::get_if<std::string>(&identity)) {
        return *failure;
    }
    if (std::get<PackedIdentity>(identity).title_id != expected_title_id) {
        return "The reuse app has a different title id: " + macos_app.string() + ".";
    }
    RemoveOnFailure cleanup{destination_game};
    const std::string cloned = clone_directory(
        macos_app / "Contents" / "Resources" / "Armigo" / "game", destination_game);
    if (!cloned.empty()) {
        return cloned;
    }
    cleanup.keep = true;
    return identity;
}

ShaderStagePaths shader_stage_paths(
    const fs::path& staging, const fs::path& resource_root, const fs::path& reused_app, bool ios) {
    if (!ios) {
        return {staging, resource_root / "shaders"};
    }
    if (!reused_app.empty()) {
        return {staging / "enumeration",
                reused_app / "Contents" / "Resources" / "Armigo" / "shaders"};
    }
    return {staging / "enumeration", staging / "macos-shaders"};
}

std::variant<std::string, std::vector<fs::path>> reused_addon_directories(const fs::path& game) {
    std::vector<fs::path> directories;
    const fs::path addons = game / "addons";
    std::error_code error;
    // symlink_status does not follow a link. A missing path is zero add-ons.
    // exists() would follow a dangling link and report the same absence.
    const fs::file_status link = fs::symlink_status(addons, error);
    if (link.type() == fs::file_type::not_found) {
        return directories;
    }
    if (error || link.type() == fs::file_type::none) {
        return std::string("Cannot enumerate reused add-on content.");
    }
    error.clear();
    if (link.type() == fs::file_type::symlink) {
        const fs::file_status target = fs::status(addons, error);
        if (error || target.type() != fs::file_type::directory) {
            return std::string("Cannot enumerate reused add-on content.");
        }
    } else if (link.type() != fs::file_type::directory) {
        return std::string("Cannot enumerate reused add-on content.");
    }
    error.clear();
    for (fs::directory_iterator cursor(addons, error), end; !error && cursor != end; cursor.increment(error)) {
        std::error_code entry_error;
        if (cursor->is_directory(entry_error)) {
            directories.push_back(cursor->path());
        } else if (entry_error) {
            error = entry_error;
            break;
        }
    }
    if (error) return std::string("Cannot enumerate reused add-on content.");
    return directories;
}

std::string validate_reused_selection(
    const PackedIdentity& identity, const std::vector<fs::path>& reused_addons,
    const std::string& title_id, const std::string& build_id, std::uint32_t title_version,
    const std::vector<std::string>& selected_addons) {
    if (identity.title_id != title_id || identity.build_id != build_id ||
        identity.title_version != title_version) {
        return "The reuse app title, build or version does not match the selected inputs.";
    }
    std::vector<std::string> actual;
    actual.reserve(reused_addons.size());
    for (const fs::path& addon : reused_addons) {
        actual.push_back(addon.filename().string());
    }
    std::sort(actual.begin(), actual.end());
    std::vector<std::string> expected = selected_addons;
    std::sort(expected.begin(), expected.end());
    if (actual != expected) {
        return "The reuse app add-on content does not match the selected inputs.";
    }
    return {};
}

std::variant<std::string, fs::path> prepare_ios_player_build(
    const fs::path& requested, const fs::path& packer_executable, const fs::path& host_shader_translator) {
    const fs::path tools = packer_executable.parent_path();
    fs::path build = requested;
    if (build.empty()) {
        if (const char* environment = std::getenv("ARMIGO_IOS_PLAYER_BUILD")) {
            build = environment;
        } else {
            build = tools.parent_path() / "ios-release";
        }
    }
    build = fs::absolute(build).lexically_normal();
    const fs::path cache = build / "CMakeCache.txt";
    std::error_code error;
    if (fs::is_regular_file(cache, error)) {
        const auto system = cmake_cache_value(cache, "CMAKE_SYSTEM_NAME");
        if (!system || *system != "iOS") {
            return "The iOS player build directory is not configured for iOS: " + build.string() + ".";
        }
        return build;
    }
    auto source = cmake_cache_value(tools / "CMakeCache.txt", "CMAKE_HOME_DIRECTORY");
    if (!source) {
        source = tools.parent_path().parent_path();
    }
    if (!fs::is_regular_file(fs::path(*source) / "CMakeLists.txt", error)) {
        return std::string("Cannot locate the Armigo source directory; configure the iOS player with the ios-release preset.");
    }
    const std::string configured = run_pack_tool({
        "cmake", "--preset", "ios-release", "-S", *source, "-B", build.string(),
        "-DARMIGO_HOST_SHADER_TRANSLATE=" + fs::absolute(host_shader_translator).string()});
    if (!configured.empty()) {
        return configured;
    }
    return build;
}

std::string create_ios_ipa(const fs::path& app_path) {
    const fs::path output = app_path.parent_path() / (app_path.stem().string() + ".ipa");
    std::error_code error;
    if (fs::exists(output, error)) {
        return "The IPA already exists: " + output.string() + ".";
    }
    const std::string unique = std::to_string(getpid());
    const fs::path staging = app_path.parent_path() / (".ipa-stage-" + unique);
    const fs::path temporary = app_path.parent_path() / ("." + app_path.stem().string() + "-" + unique + ".ipa");
    RemoveOnFailure staging_cleanup{staging};
    RemoveOnFailure ipa_cleanup{temporary};
    const std::string cloned = clone_directory(app_path, staging / "Payload" / app_path.filename());
    if (!cloned.empty()) {
        return cloned;
    }
    const std::string packed = run_pack_tool({
        "/usr/bin/env", "-C", staging.string(), "/usr/bin/zip", "-X", "-0", "-r",
        temporary.string(), "Payload"});
    if (!packed.empty()) {
        return packed;
    }
    fs::rename(temporary, output, error);
    if (error) {
        return "Cannot publish the IPA: " + output.string() + ".";
    }
    ipa_cleanup.keep = true;
    return {};
}

} // namespace armigo::pack
