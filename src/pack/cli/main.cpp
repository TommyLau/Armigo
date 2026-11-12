// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "bundle.h"
#include "command.h"
#include "firmware.h"
#include "icon.h"
#include "ios.h"
#include "native_shaders.h"
#include "pipelines.h"
#include "package_tools.h"
#include "pack_flow.h"
#include "run.h"
#include "settings.h"
#include "shaders.h"
#include "space.h"

#include <armigo/compat/title_settings.h>
#include <armigo/loader/container.h>
#include <armigo/loader/nup.h>
#include <armigo/pack/ios_link.h>
#include <armigo/pack/naming.h>
#include <armigo/runtime/shader_chain.h>
#include <armigo/shader/ios_package.h>

#include "inputs.h"
#include "pack_commit.h"

#include <mach-o/dyld.h>
#include <sys/clonefile.h>
#include <unistd.h>

#include <cctype>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace pack = armigo::pack;
namespace loader = armigo::loader;

constexpr std::uint64_t kLowSpaceBytes = 4ull << 30;

struct PartialOutput {
    fs::path app;
    fs::path sidecar;
    fs::path staging;
    // Further outputs removed on failure (a shaders-only pack's folders).
    std::vector<fs::path> others;
    bool keep = false;

    ~PartialOutput() {
        if (keep) {
            return;
        }
        std::error_code error;
        for (const fs::path& other : others) {
            fs::remove_all(other, error);
        }
        if (!app.empty()) {
            fs::remove_all(app, error);
        }
        if (!sidecar.empty()) {
            fs::remove_all(sidecar, error);
        }
        if (!staging.empty()) {
            fs::remove_all(staging, error);
        }
    }
};

struct Step {
    const char* name;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    ~Step() {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("%s: %.1f s\n", name, seconds);
        std::fflush(stdout);
    }
};

void print_error(std::string_view message) {
    armigo::diag::write(armigo::diag::Level::error,  "%s\n", std::string(message).c_str());
}

std::string format_gib(std::uint64_t bytes) {
    char text[64];
    std::snprintf(text, sizeof text, "%.2f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return text;
}

fs::path current_executable() {
    std::uint32_t size = 4096;
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        buffer.assign(size, '\0');
        if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
            return {};
        }
    }
    std::error_code error;
    auto path = fs::weakly_canonical(buffer.c_str(), error);
    return error ? fs::path(buffer.c_str()) : path;
}

std::string keys_error(const loader::ContainerKeysFailure& failure) {
    const char* code = "keys";
    switch (failure.code) {
        case loader::ContainerKeysFailure::Code::file_unavailable: code = "keys file"; break;
        case loader::ContainerKeysFailure::Code::malformed_entry: code = "keys file"; break;
        case loader::ContainerKeysFailure::Code::bad_hex_value: code = "keys file"; break;
        case loader::ContainerKeysFailure::Code::duplicate_key: code = "keys file"; break;
    }
    if (failure.detail.empty()) {
        return std::string(code) + " could not be read.";
    }
    return std::string(code) + ": " + failure.detail;
}

std::string container_error(const loader::ContainerFailure& failure) {
    return std::string(loader::to_string(failure.code)) + ": " + failure.detail;
}

std::uint64_t directory_bytes(const fs::path& root) {
    std::uint64_t sum = 0;
    std::error_code error;
    if (!fs::exists(root, error)) {
        return 0;
    }
    fs::recursive_directory_iterator cursor(root, fs::directory_options::skip_permission_denied, error);
    const fs::recursive_directory_iterator end;
    while (!error && cursor != end) {
        if (cursor->is_regular_file(error)) {
            const auto size = cursor->file_size(error);
            if (!error) {
                sum += static_cast<std::uint64_t>(size);
            }
        }
        error.clear();
        cursor.increment(error);
    }
    return sum;
}

std::uint64_t program_and_data_bytes(const pack::ContentSet& set) {
    std::uint64_t sum = 0;
    for (const pack::ContentRecord& record : set.content_records) {
        if (record.type != pack::ContentRecordType::program && record.type != pack::ContentRecordType::data) {
            continue;
        }
        if (sum > UINT64_MAX - record.size) {
            return UINT64_MAX;
        }
        sum += record.size;
    }
    return sum;
}

void print_content(const char* role, const pack::ContentSet& set) {
    std::printf("%s %s version %u\n  %s\n", role, pack::format_title_id(set.title_id).c_str(), set.version,
                set.location.file.c_str());
}

// Firmware NCAs are inputs too, but only their meta records matter: the
// content they name is read through those records, so the other NCAs of a
// firmware folder are counted rather than listed as refused.
void print_inputs(const pack::Inputs& inputs, const fs::path& firmware) {
    print_content("Game", inputs.application);
    if (inputs.patch) {
        print_content("Update", *inputs.patch);
    }
    for (const pack::ContentSet& addon : inputs.add_on_content) {
        print_content("Add-on", addon);
    }
    std::size_t firmware_content = 0;
    for (const pack::RefusedInput& refused : inputs.refused) {
        if (!firmware.empty() && refused.file.parent_path() == firmware) {
            ++firmware_content;
            continue;
        }
        std::printf("Refused %s: %s\n", refused.file.c_str(), refused.reason.c_str());
    }
    if (firmware_content)
        std::printf("Firmware: %zu content NCAs, read through their meta records.\n", firmware_content);
    for (const pack::SupersededUpdate& update : inputs.superseded_updates) {
        std::printf("Ignored update %s version %u: %s\n", pack::format_title_id(update.update.title_id).c_str(),
                    update.update.version, update.reason.c_str());
    }
    for (const pack::SupersededUpdate& update : inputs.duplicate_updates) {
        std::printf("Ignored update %s version %u: %s\n", pack::format_title_id(update.update.title_id).c_str(),
                    update.update.version, update.reason.c_str());
    }
    for (const pack::SupersededUpdate& update : inputs.superseded_add_on_content) {
        std::printf("Ignored add-on %s version %u: %s\n", pack::format_title_id(update.update.title_id).c_str(),
                    update.update.version, update.reason.c_str());
    }
    std::printf("Keys accepted.\n");
    std::fflush(stdout);
}

std::string require_space(const fs::path& output) {
    const pack::SpaceCheck check = pack::check_free_space(output, 0);
    if (!check.error.empty()) {
        return check.error;
    }
    std::printf("Free space: %s\n", format_gib(check.available).c_str());
    std::fflush(stdout);
    if (check.available < kLowSpaceBytes) {
        return "Free space is below 4 GiB (" + format_gib(check.available) + "); packing stopped.";
    }
    return {};
}

// A shaders-only pack writes into a folder that is new or empty.
std::string prepare_shaders_output(const fs::path& output) {
    std::error_code error;
    if (fs::is_symlink(output, error)) {
        return "The shaders-only output cannot be a symbolic link: " + output.string();
    }
    if (fs::exists(output, error) &&
        (!fs::is_directory(output, error) || fs::directory_iterator(output, error) != fs::directory_iterator{})) {
        return "The shaders-only output must be a new or empty folder: " + output.string();
    }
    fs::create_directories(output, error);
    if (error) {
        return "Cannot create " + output.string();
    }
    return {};
}

// Ends a shaders-only pack: the extracted game is work data, the package and
// pipeline archive are the result.
int finish_shaders_only(PartialOutput& partial, const fs::path& output, const fs::path& game,
                        std::chrono::steady_clock::time_point started) {
    std::error_code error;
    fs::remove_all(game, error);
    if (error) {
        print_error("Cannot remove the extracted game work folder " + game.string() + ": " + error.message());
        return 1;
    }
    partial.keep = true;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("Packed shaders %s (%s) in %.1f s\n", output.c_str(), format_gib(directory_bytes(output)).c_str(),
                seconds);
    std::fflush(stdout);
    return 0;
}

std::optional<std::array<std::byte, 32>> read_main_build(const fs::path& exefs, std::string& error) {
    const fs::path main_file = fs::is_regular_file(exefs / "main") ? exefs / "main" : exefs / "main.nso";
    auto read = pack::read_nso_build_id(main_file);
    if (auto* failure = std::get_if<std::string>(&read)) {
        error = *failure;
        return std::nullopt;
    }
    return std::get<std::array<std::byte, 32>>(read);
}

bool nup_extension(const fs::path& game) {
    std::string extension = game.extension().string();
    for (char& character : extension) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return extension == ".nup";
}

std::string nup_error(const loader::NupFailure& failure) {
    if (failure.detail.empty()) return failure.operation;
    return failure.operation + ": " + failure.detail;
}

bool write_bytes(const fs::path& path, const std::vector<std::byte>& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    return static_cast<bool>(output);
}

int report_free_space(const fs::path& output, std::uint64_t expected) {
    Step step{"Check free space"};
    const pack::SpaceCheck space = pack::check_free_space(output, expected);
    if (!space.error.empty()) {
        print_error(space.error);
        return 1;
    }
    std::printf("Free space %s; this pack expects %s.\n", format_gib(space.available).c_str(),
                format_gib(space.expected).c_str());
    if (!space.enough) {
        print_error("Not enough free space: " + format_gib(space.available) + " available, " +
                    format_gib(space.expected) + " required.");
        return 1;
    }
    return 0;
}

struct PackSession {
    const pack::Command& command;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    fs::path executable;
    fs::path player;
    fs::path translator;
    fs::path preparer;
    bool ios = false;
    fs::path firmware_input;
    fs::path app_path;
    fs::path kept_app;
    fs::path assembly;
    fs::path armigo_root;
    fs::path game;
    std::string title_id;
    std::uint64_t raw_title_id = 0;
    std::uint32_t title_version = 0;
    std::array<std::byte, 32> build_id{};
    std::optional<fs::path> reuse_app;
    std::optional<pack::PackedIdentity> reused_identity;
    std::vector<fs::path> addon_directories;
    std::vector<fs::path> manifests;
    fs::path linked_player;
    std::string executable_name;
    pack::TitleMatch matched;
    const armigo::compat::PatchDescription* description = nullptr;
    // firmware_partial is declared first, so partial is destroyed before it.
    PartialOutput firmware_partial;
    PartialOutput partial;

    explicit PackSession(const pack::Command& command_in) : command(command_in) {}
};

// The app is assembled under a hidden name that is not a bundle and moved into
// place only when complete: LaunchServices may record an app while it is still
// being written, and the Finder then keeps showing it as one that cannot run.
int bind_output(PackSession& session) {
    if (!session.command.shaders_only) {
        std::error_code exists_error;
        if (fs::exists(session.kept_app, exists_error)) {
            print_error("The packed app already exists: " + session.kept_app.string());
            return 1;
        }
    }
    session.assembly = pack::assembly_path(session.app_path, static_cast<std::uint64_t>(getpid()));
    if (!session.command.shaders_only) {
        std::error_code exists_error;
        if (fs::exists(session.assembly, exists_error)) {
            print_error("The app assembly folder already exists: " + session.assembly.string());
            return 1;
        }
    }
    return 0;
}

struct PackInput {
    virtual ~PackInput() = default;
    virtual pack::InputCapabilities capabilities() const = 0;
    virtual int read_keys(const pack::Command& command) = 0;
    virtual int load_content(PackSession& session) = 0;
    virtual int announce(PackSession& session) = 0;
    virtual int estimate_space(PackSession& session) = 0;
    virtual int install_game(PackSession& session) = 0;
    virtual int collect_addons(PackSession& session) = 0;
    virtual int build_shaders(PackSession& session, const fs::path& compile_cache) = 0;
    virtual int extract_firmware(PackSession& session) = 0;
    virtual int fill_bundle_identity(PackSession& session, pack::BundleSpec& spec) = 0;
};

// XCI, NSP and an extracted folder. Firmware is embedded and iOS is available.
struct GamePackInput : PackInput {
    loader::ContainerKeys keys;
    loader::ContainerKeys title_keys;
    pack::Inputs inputs;
    pack::PackedNames names;
    std::vector<std::byte> icon_bytes;

    pack::InputCapabilities capabilities() const override { return pack::game_input_capabilities(); }
    int read_keys(const pack::Command& command) override;
    int load_content(PackSession& session) override;
    int announce(PackSession& session) override;
    int estimate_space(PackSession& session) override;
    int install_game(PackSession& session) override;
    int collect_addons(PackSession& session) override;
    int build_shaders(PackSession& session, const fs::path& compile_cache) override;
    int extract_firmware(PackSession& session) override;
    int fill_bundle_identity(PackSession& session, pack::BundleSpec& spec) override;
};

// A .nup title is not a cartridge: the archive stays whole, its ExeFS is
// written beside it, and admitted native programs are packaged before launch.
// Keys are read so a bad keys file fails immediately; the archive is not
// decrypted. Firmware fonts are not embedded.
struct NupPackInput : PackInput {
    loader::ExeFsDirectoryInput exefs;
    std::vector<std::byte> icon_bytes;

    pack::InputCapabilities capabilities() const override { return pack::nup_input_capabilities(); }
    int read_keys(const pack::Command& command) override;
    int load_content(PackSession& session) override;
    int announce(PackSession& session) override;
    int estimate_space(PackSession& session) override;
    int install_game(PackSession& session) override;
    int collect_addons(PackSession&) override { return 0; }
    int build_shaders(PackSession& session, const fs::path& compile_cache) override;
    int extract_firmware(PackSession&) override { return 0; }
    int fill_bundle_identity(PackSession& session, pack::BundleSpec& spec) override;
};

int GamePackInput::read_keys(const pack::Command& command) {
    Step step{"Read keys"};
    auto read = loader::read_container_keys_file(command.keys);
    if (auto* failure = std::get_if<loader::ContainerKeysFailure>(&read)) {
        print_error(keys_error(*failure));
        return 1;
    }
    keys = std::get<loader::ContainerKeys>(std::move(read));
    if (!command.title_keys.empty()) {
        auto read_title = loader::read_container_keys_file(command.title_keys);
        if (auto* failure = std::get_if<loader::ContainerKeysFailure>(&read_title)) {
            print_error(keys_error(*failure));
            return 1;
        }
        title_keys = std::get<loader::ContainerKeys>(std::move(read_title));
    }
    return 0;
}

int GamePackInput::load_content(PackSession& session) {
    {
        Step step{"Classify inputs"};
        std::vector<fs::path> paths;
        paths.push_back(session.command.game);
        paths.insert(paths.end(), session.command.updates.begin(), session.command.updates.end());
        paths.insert(paths.end(), session.command.dlc.begin(), session.command.dlc.end());
        if (!session.firmware_input.empty()) {
            paths.push_back(session.firmware_input);
        }
        pack::InputsOptions options;
        options.keys_file = session.command.keys;
        auto classified = pack::classify_inputs(paths, options);
        if (auto* failure = std::get_if<pack::InputsFailure>(&classified)) {
            print_error(failure->detail);
            return 1;
        }
        inputs = std::get<pack::Inputs>(std::move(classified));
        print_inputs(inputs, session.firmware_input);
    }
    session.raw_title_id = inputs.application.title_id;
    session.title_id = pack::format_title_id(session.raw_title_id);
    session.title_version = inputs.patch ? inputs.patch->version : inputs.application.version;
    return 0;
}

int GamePackInput::announce(PackSession& session) {
    const pack::Command& command = session.command;
    if (!command.shaders_only) {
        auto named = pack::names_from_control(inputs.application.title_id, inputs.control.nacp);
        if (auto* failure = std::get_if<std::string>(&named)) {
            print_error(*failure);
            return 1;
        }
        names = std::get<pack::PackedNames>(std::move(named));
        std::printf("Title %s\n  %s.app\n", names.display_name.c_str(), names.file_name.c_str());
        for (const std::string& warning : names.warnings) {
            std::printf("Control data: %s\n", warning.c_str());
        }
        std::fflush(stdout);
    }
    session.executable_name = names.executable_name;
    session.app_path = command.output / (names.file_name + ".app");
    // A kept iOS app sits beside the macOS app of the same title, which an iOS
    // pack reuses from the same output folder; the IPA keeps the plain name.
    session.kept_app = session.ios ? command.output / (names.file_name + " (iOS).app") : session.app_path;
    return 0;
}

int GamePackInput::estimate_space(PackSession& session) {
    const bool ios = session.ios;
    std::uint64_t expected_bytes = pack::kPackMarginBytes;
    if (ios || !session.reuse_app) {
        std::vector<pack::ContentSize> sizes;
        sizes.push_back({inputs.application.location.file, program_and_data_bytes(inputs.application)});
        if (inputs.patch) {
            sizes.push_back({inputs.patch->location.file, program_and_data_bytes(*inputs.patch)});
        }
        for (const pack::ContentSet& addon : inputs.add_on_content) {
            sizes.push_back({addon.location.file, program_and_data_bytes(addon)});
        }
        const auto content = pack::sum_content_bytes(sizes);
        if (auto* failure = std::get_if<std::string>(&content)) {
            print_error(*failure);
            return 1;
        }
        const std::uint64_t content_bytes = std::get<std::uint64_t>(content);
        const std::uint64_t copies = ios && !session.reuse_app ? 2 : 1;
        if (content_bytes > UINT64_MAX / copies) {
            print_error("The packed input size overflows the free-space estimate.");
            return 1;
        }
        const auto expected = pack::expected_pack_bytes(content_bytes * copies);
        if (auto* failure = std::get_if<std::string>(&expected)) {
            print_error(*failure);
            return 1;
        }
        expected_bytes = std::get<std::uint64_t>(expected);
    }
    return report_free_space(session.command.output, expected_bytes);
}

int GamePackInput::install_game(PackSession& session) {
    if (session.reuse_app) {
        Step step{"Clone game data"};
        auto cloned = pack::clone_macos_game(*session.reuse_app, session.game, session.title_id);
        if (auto* failure = std::get_if<std::string>(&cloned)) {
            print_error(*failure);
            return 1;
        }
        session.reused_identity = std::get<pack::PackedIdentity>(std::move(cloned));
        std::printf("Reused %s\n", session.reuse_app->c_str());
        return 0;
    }
    Step step{"Extract game"};
    loader::ContainerExtractionOptions extraction;
    extraction.keys = keys;
    extraction.title_keys = title_keys;
    extraction.output_directory = session.game;
    if (inputs.patch) {
        extraction.update_container_file = inputs.patch->location.file;
    }
    auto extracted = loader::extract_program_container(inputs.application.location.file, extraction);
    if (auto* failure = std::get_if<loader::ContainerFailure>(&extracted)) {
        print_error(container_error(*failure));
        return 1;
    }
    return 0;
}

int GamePackInput::collect_addons(PackSession& session) {
    if (session.reused_identity) {
        Step step{"Collect reused add-on content"};
        auto collected = pack::reused_addon_directories(session.game);
        if (const auto* failure = std::get_if<std::string>(&collected)) {
            print_error(*failure);
            return 1;
        }
        session.addon_directories = std::get<std::vector<fs::path>>(std::move(collected));
    } else {
        Step step{"Extract add-on content"};
        loader::ContainerExtractionOptions extraction;
        extraction.keys = keys;
        extraction.title_keys = title_keys;
        for (const pack::ContentSet& addon : inputs.add_on_content) {
            const std::string data_id = pack::format_title_id(addon.title_id);
            const fs::path destination = session.game / "addons" / data_id;
            extraction.output_directory = destination;
            extraction.update_container_file.clear();
            auto extracted = loader::extract_data_romfs(addon.location.file, extraction, addon.title_id);
            if (auto* failure = std::get_if<loader::ContainerFailure>(&extracted)) {
                print_error(data_id + ": " + container_error(*failure));
                return 1;
            }
            session.addon_directories.push_back(destination);
            std::printf("Add-on %s\n  %s\n", data_id.c_str(), destination.c_str());
        }
    }
    if (session.reused_identity) {
        std::vector<std::string> selected_addons;
        selected_addons.reserve(inputs.add_on_content.size());
        for (const pack::ContentSet& addon : inputs.add_on_content) {
            selected_addons.push_back(pack::format_title_id(addon.title_id));
        }
        const std::string mismatch = pack::validate_reused_selection(
            *session.reused_identity, session.addon_directories, session.title_id,
            pack::format_build_id(session.build_id), session.title_version, selected_addons);
        if (!mismatch.empty()) {
            print_error(mismatch);
            return 1;
        }
    }
    return 0;
}

int GamePackInput::build_shaders(PackSession& session, const fs::path& compile_cache) {
    const bool reuse_shaders = session.ios && session.reuse_app.has_value();
    const auto shader_paths = pack::shader_stage_paths(session.partial.staging, session.armigo_root,
                                                        session.reuse_app.value_or(fs::path{}), session.ios);
    const fs::path& macos_shaders = shader_paths.macos_package;
    if (reuse_shaders) {
        std::printf("Reusing macOS shader package %s\n", macos_shaders.c_str());
    } else {
        if (const std::string low = require_space(session.command.output); !low.empty()) {
            print_error(low);
            return 1;
        }
        const auto& selected = session.description->shader_inputs;
        if (selected.containers == armigo::compat::ShaderContainerSet::Native) {
            print_error("The matched title settings select native shader inputs, which only a .nup title has.");
            return 1;
        }
        Step step{"Translate shaders"};
        pack::ShaderPackInput shader_input;
        shader_input.settings = selected;
        shader_input.hoist_implicit_samples = session.description->nvn_policy.hoist_implicit_samples;
        auto built = pack::build_shader_package(session.game / "exefs", session.game / "romfs",
                                                 session.addon_directories, shader_paths.enumeration, macos_shaders,
                                                 compile_cache, session.translator, shader_input,
                                                 session.command.restamp.empty()
                                                     ? std::vector<fs::path>{}
                                                     : std::vector<fs::path>{session.command.restamp});
        if (auto* failure = std::get_if<std::string>(&built)) {
            print_error(*failure);
            return 1;
        }
        const auto& shaders = std::get<pack::ShaderReport>(built);
        std::printf("Shader stages translated: %zu (enumerated %zu, unsupported %zu, structural errors %zu).\n",
                    shaders.translated, shaders.enumerated, shaders.unsupported, shaders.structural_errors);
        if (!session.command.restamp.empty()) {
            std::printf("Restamp: %zu of %zu Metal libraries reused from %s; %zu compiled or cached.\n",
                        shaders.reused, shaders.libraries, session.command.restamp.c_str(),
                        shaders.libraries - shaders.reused);
        }
    }
    if (session.ios) {
        Step step{"Build iOS shader package"};
        armigo::shader::package::IosBuildOptions options;
        options.macos_package = macos_shaders;
        options.output = session.armigo_root / "shaders";
        options.jobs = 8;
        if (!compile_cache.empty()) {
            options.compile_cache = compile_cache;
        }
        auto converted = armigo::shader::package::build_ios_package(options);
        if (auto* failure = std::get_if<std::string>(&converted)) {
            print_error(*failure);
            return 1;
        }
        std::printf("iOS shader stages compiled: %zu.\n",
                    std::get<armigo::shader::package::IosBuildReport>(converted).converted_stages);
    }
    return 0;
}

int GamePackInput::extract_firmware(PackSession& session) {
    Step step{"Extract firmware resources"};
    auto resources = pack::extract_firmware_resources(inputs, session.firmware_input, keys,
                                                      session.armigo_root / "system");
    if (const auto* failure = std::get_if<std::string>(&resources)) {
        print_error(*failure);
        return 1;
    }
    const auto& firmware = std::get<pack::FirmwareResources>(resources);
    std::printf("Firmware version %s.\n", firmware.version.c_str());
    return 0;
}

int GamePackInput::fill_bundle_identity(PackSession& session, pack::BundleSpec& spec) {
    std::vector<pack::ControlIcon> icons;
    icons.reserve(inputs.control.icons.size());
    for (const pack::NamedData& icon : inputs.control.icons) {
        icons.push_back(pack::ControlIcon{icon.name, icon.bytes});
    }
    const auto selected = pack::select_control_icon(icons, names.language);
    if (auto* failure = std::get_if<std::string>(&selected)) {
        print_error(*failure);
        return 1;
    }
    const auto& jpeg = std::get<pack::ControlIcon>(selected).bytes;
    auto icon = session.ios ? pack::jpeg_to_ios_png(jpeg) : pack::jpeg_to_icns(jpeg);
    if (auto* failure = std::get_if<std::string>(&icon)) {
        print_error(*failure);
        return 1;
    }
    icon_bytes = std::get<std::vector<std::byte>>(std::move(icon));
    spec.display_name = names.display_name;
    spec.file_name = names.file_name;
    spec.executable_name = names.executable_name;
    spec.bundle_identifier = names.bundle_identifier;
    spec.icon = icon_bytes;
    spec.languages = names.languages;
    spec.localized_titles = names.localized_titles;
    spec.supported_language_flag = names.supported_language_flag;
    return 0;
}

int NupPackInput::read_keys(const pack::Command& command) {
    Step step{"Read keys"};
    auto read = loader::read_container_keys_file(command.keys);
    if (const auto* failure = std::get_if<loader::ContainerKeysFailure>(&read)) {
        print_error(keys_error(*failure));
        return 1;
    }
    std::printf("Keys accepted.\n");
    std::fflush(stdout);
    return 0;
}

int NupPackInput::load_content(PackSession& session) {
    Step step{"Read package"};
    auto opened = loader::NupArchive::open(session.command.game);
    if (const auto* failure = std::get_if<loader::NupFailure>(&opened)) {
        print_error(nup_error(*failure));
        return 1;
    }
    const auto archive = std::get<std::shared_ptr<loader::NupArchive>>(std::move(opened));
    auto selected = archive->select();
    if (const auto* failure = std::get_if<loader::NupFailure>(&selected)) {
        print_error(nup_error(*failure));
        return 1;
    }
    const auto selection = std::get<loader::NupSelection>(std::move(selected));
    auto read = archive->read_exefs(selection);
    if (const auto* failure = std::get_if<loader::NupFailure>(&read)) {
        print_error(nup_error(*failure));
        return 1;
    }
    exefs = std::get<loader::ExeFsDirectoryInput>(std::move(read));
    // A member hash can already describe damaged bytes. Stop before the
    // archive is placed when a nonempty .zs member is not a zstd frame.
    auto mismatches = archive->zs_frame_mismatches(selection);
    if (const auto* failure = std::get_if<loader::NupFailure>(&mismatches)) {
        print_error(nup_error(*failure));
        return 1;
    }
    const auto& damaged = std::get<std::vector<std::string>>(mismatches);
    if (!damaged.empty()) {
        std::string message = std::to_string(damaged.size()) +
            " compressed member(s) do not start with a zstd frame:";
        const std::size_t shown = damaged.size() < 20 ? damaged.size() : 20;
        for (std::size_t i = 0; i < shown; ++i) message += "\n  " + damaged[i];
        if (damaged.size() > shown) {
            message += "\n  " + std::to_string(damaged.size() - shown) + " more";
        }
        print_error(message);
        return 1;
    }
    session.raw_title_id = selection.title_id;
    session.title_id = pack::format_title_id(selection.title_id);
    session.title_version = selection.version;
    return 0;
}

int NupPackInput::announce(PackSession& session) {
    const pack::Command& command = session.command;
    std::printf("Title %s\n  %s\n", session.title_id.c_str(),
                command.shaders_only ? command.output.c_str() : (session.title_id + ".app").c_str());
    std::fflush(stdout);
    session.executable_name = session.title_id;
    session.app_path = command.output / (session.title_id + ".app");
    session.kept_app = session.app_path;
    return 0;
}

int NupPackInput::estimate_space(PackSession& session) {
    std::error_code size_error;
    const auto bytes = fs::file_size(session.command.game, size_error);
    if (size_error) {
        return 0;
    }
    const auto expected = pack::expected_pack_bytes(static_cast<std::uint64_t>(bytes));
    if (const auto* failure = std::get_if<std::string>(&expected)) {
        print_error(*failure);
        return 1;
    }
    return report_free_space(session.command.output, std::get<std::uint64_t>(expected));
}

int NupPackInput::install_game(PackSession& session) {
    const fs::path exefs_dir = session.game / "exefs";
    {
        Step step{"Write executable files"};
        std::error_code error;
        fs::create_directories(exefs_dir, error);
        if (error) {
            print_error("Cannot create " + exefs_dir.string());
            return 1;
        }
        for (const loader::ExeFsFile& module : exefs.modules) {
            if (module.name.find('/') != std::string::npos || module.name.find('\\') != std::string::npos) {
                print_error("The package named an executable member that is not a file name.");
                return 1;
            }
            if (!write_bytes(exefs_dir / module.name, module.bytes)) {
                print_error("Cannot write " + (exefs_dir / module.name).string());
                return 1;
            }
        }
        if (exefs.process_metadata && !write_bytes(exefs_dir / "main.npdm", *exefs.process_metadata)) {
            print_error("Cannot write " + (exefs_dir / "main.npdm").string());
            return 1;
        }
    }
    if (session.command.shaders_only) {
        return 0;
    }
    Step step{"Place package"};
    const fs::path destination = session.game / "title.nup";
    std::error_code error;
    fs::create_directories(destination.parent_path(), error);
    if (error) {
        print_error("Cannot create " + destination.parent_path().string());
        return 1;
    }
    if (::clonefile(session.command.game.c_str(), destination.c_str(), 0) != 0) {
        error.clear();
        fs::create_hard_link(session.command.game, destination, error);
        if (error) {
            error.clear();
            fs::copy_file(session.command.game, destination, error);
            if (error) {
                print_error("Cannot place the package at " + destination.string() + ": " + error.message());
                return 1;
            }
        }
    }
    return 0;
}

int NupPackInput::build_shaders(PackSession& session, const fs::path& compile_cache) {
    if (const std::string low = require_space(session.command.output); !low.empty()) {
        print_error(low);
        return 1;
    }
    Step step{"Translate native shaders"};
    pack::NativeShaderRequest request;
    request.nup = session.command.game;
    request.output = session.armigo_root / "shaders";
    request.translate_tool = session.translator;
    request.compile_cache = compile_cache;
    request.jobs = 8;
    request.keep_sources = false;
    // The title's settings name the dictionary member and any member or
    // stage selection; without them every member and program is read.
    if (const auto& selected = session.description->shader_inputs;
        selected.containers == armigo::compat::ShaderContainerSet::Native) {
        request.dictionary_member = selected.dictionary_member;
        request.members = selected.members;
        request.identities = selected.identities;
    }
    auto built = pack::build_native_shader_package(request);
    if (const auto* failure = std::get_if<std::string>(&built)) {
        print_error(*failure);
        return 1;
    }
    const auto& report = std::get<pack::NativeShaderReport>(built);
    std::printf("Native shader stages translated: %llu (enumerated %llu, unique %llu, compiled %llu).\n",
                static_cast<unsigned long long>(report.translated),
                static_cast<unsigned long long>(report.enumerated),
                static_cast<unsigned long long>(report.unique),
                static_cast<unsigned long long>(report.compiled));
    return 0;
}

int NupPackInput::fill_bundle_identity(PackSession& session, pack::BundleSpec& spec) {
    icon_bytes = {std::byte{0x69}, std::byte{0x63}, std::byte{0x6e}, std::byte{0x73}};
    spec.display_name = session.title_id;
    spec.file_name = session.title_id;
    spec.executable_name = session.title_id;
    spec.bundle_identifier = "com.tommylau.armigo." + session.title_id;
    spec.icon = icon_bytes;
    spec.languages = {"en-US"};
    spec.supported_language_flag = 1;
    return 0;
}

// One pack pipeline. An input adapter supplies the kind-specific reads, the
// game placement and the shader strategy; every other step lives here.
int pack_flow(const pack::Command& command) {
    // The XCI/NSP timer includes the tool check. The .nup timer starts once
    // that check has passed, which is where its own pack began.
    auto started = std::chrono::steady_clock::now();
    if (pack::inside_git_work_tree(command.output)) {
        print_error("Refusing to write a packed app inside a git work tree: " + command.output.string());
        return 1;
    }
    const fs::path executable = current_executable();
    if (executable.empty()) {
        print_error("Cannot locate the packer executable.");
        return 1;
    }
    const bool ios = command.platform == "ios";
    const fs::path tools = executable.parent_path();
    const fs::path player = tools / "Armigo.linker";
    const fs::path translator = tools / "armigo-shader-translate";
    const fs::path preparer = tools / "armigo-prepare";
    {
        Step step{"Check packer tools"};
        std::error_code error;
        if (!ios && !command.shaders_only && !fs::is_regular_file(player, error)) {
            print_error("The linker-signed player is missing: " + player.string());
            return 1;
        }
        if (!fs::is_regular_file(translator, error)) {
            print_error("The shader translator is missing next to the packer.");
            return 1;
        }
        if (ios && !fs::is_regular_file(preparer, error)) {
            print_error("The embedded preparation tool is missing next to the packer.");
            return 1;
        }
    }

    const bool native = nup_extension(command.game);
    std::unique_ptr<PackInput> input;
    if (native) {
        input = std::make_unique<NupPackInput>();
    } else {
        input = std::make_unique<GamePackInput>();
    }
    const pack::InputCapabilities caps = input->capabilities();
    if (!caps.ios && ios) {
        print_error("A .nup title packs for macOS only.");
        return 1;
    }
    if (native) {
        started = std::chrono::steady_clock::now();
    }
    if (caps.native_shaders && !command.restamp.empty()) {
        print_error("--restamp reuses a Switch 1 shader package; a .nup title has none to reuse yet.");
        return 1;
    }
    if (!caps.updates_and_addons && (!command.updates.empty() || !command.dlc.empty())) {
        print_error("A .nup pack does not take updates or add-on content.");
        return 1;
    }

    PackSession session(command);
    session.started = started;
    session.executable = executable;
    session.player = player;
    session.translator = translator;
    session.preparer = preparer;
    session.ios = ios;
    session.linked_player = player;

    if (caps.native_shaders) {
        if (command.shaders_only) {
            if (const std::string failure = prepare_shaders_output(command.output); !failure.empty()) {
                print_error(failure);
                return 1;
            }
        }
        if (const std::string low = require_space(command.output); !low.empty()) {
            print_error(low);
            return 1;
        }
        if (input->read_keys(command) != 0) return 1;
        if (!command.shaders_only) {
            std::error_code error;
            if (!fs::exists(command.firmware, error) || error) {
                print_error("The firmware path does not exist: " + command.firmware.string());
                return 1;
            }
            std::printf("Firmware system data is not embedded for a .nup pack.\n");
            std::fflush(stdout);
        }
        if (input->load_content(session) != 0) return 1;
        if (input->announce(session) != 0) return 1;
        if (bind_output(session) != 0) return 1;
        if (input->estimate_space(session) != 0) return 1;
    } else {
        if (input->read_keys(command) != 0) return 1;
        if (command.shaders_only) {
            if (const std::string failure = prepare_shaders_output(command.output); !failure.empty()) {
                print_error(failure);
                return 1;
            }
        }
        if (session.ios) {
            // The title's settings decide whether it packs for iOS; that is
            // known from its content metadata, before any firmware or game
            // content is unpacked.
            std::vector<fs::path> paths{command.game};
            paths.insert(paths.end(), command.updates.begin(), command.updates.end());
            paths.insert(paths.end(), command.dlc.begin(), command.dlc.end());
            pack::InputsOptions options;
            options.keys_file = command.keys;
            auto classified = pack::classify_inputs(paths, options);
            if (const auto* failure = std::get_if<pack::InputsFailure>(&classified)) {
                print_error(failure->detail);
                return 1;
            }
            const std::uint64_t title = std::get<pack::Inputs>(classified).application.title_id;
            if (const std::string refused = pack::check_title_platform(title, "ios", executable); !refused.empty()) {
                print_error(refused);
                return 1;
            }
        }
        session.firmware_partial.staging = command.output / ".firmware-input";
        if (!command.shaders_only) {
            Step step{"Prepare firmware"};
            auto prepared = pack::prepare_firmware_input(command.firmware, session.firmware_partial.staging);
            if (const auto* failure = std::get_if<std::string>(&prepared)) {
                print_error(*failure);
                return 1;
            }
            session.firmware_input = std::get<fs::path>(prepared);
        }
        if (input->load_content(session) != 0) return 1;
        if (input->announce(session) != 0) return 1;
        if (bind_output(session) != 0) return 1;
        if (command.shaders_only) {
            std::printf("Title %s\n  %s\n", session.title_id.c_str(), command.output.c_str());
            std::fflush(stdout);
        }
        if (session.ios) {
            Step step{"Find reusable game data"};
            auto selected = pack::select_macos_reuse(command.reuse, command.output, session.kept_app,
                                                      session.title_id);
            if (auto* failure = std::get_if<std::string>(&selected)) {
                print_error(*failure);
                return 1;
            }
            session.reuse_app = std::get<std::optional<fs::path>>(std::move(selected));
        }
        if (input->estimate_space(session) != 0) return 1;
    }

    if (session.ios) {
        session.partial.sidecar = command.output / "entitlements.plist";
    }
    if (!caps.native_shaders) {
        session.partial.staging = command.output / (session.ios ? ".pack-stages-ios" : ".pack-stages");
    }
    // A shaders-only pack builds the same package and archive straight into the
    // output folder. A full XCI/NSP pack extracts into the app; a full .nup pack
    // writes executable files there and places the archive beside them.
    session.armigo_root = command.shaders_only ? command.output
                                              : session.ios ? session.assembly / "Armigo"
                                                            : session.assembly / "Contents" / "Resources" / "Armigo";
    session.game = command.shaders_only ? command.output / ".pack-game" : session.armigo_root / "game";
    if (command.shaders_only) {
        session.partial.others = {session.game, session.armigo_root / "shaders", session.armigo_root / "pipelines"};
    } else {
        session.partial.app = session.assembly;
    }
    if (input->install_game(session) != 0) return 1;

    {
        Step step{"Match title settings"};
        std::string build_error;
        const auto read_build = read_main_build(session.game / "exefs", build_error);
        if (!read_build) {
            print_error(build_error);
            return 1;
        }
        session.build_id = *read_build;
        if (session.reused_identity &&
            session.reused_identity->build_id != pack::format_build_id(session.build_id)) {
            print_error("The reuse app build id does not match its cloned ExeFS.");
            return 1;
        }
        auto match = pack::match_title_build(session.raw_title_id, session.build_id, session.executable);
        if (auto* failure = std::get_if<std::string>(&match)) {
            print_error(*failure);
            return 1;
        }
        session.matched = std::get<pack::TitleMatch>(std::move(match));
        session.description = &session.matched.description();
        std::printf("Title settings %s\n", session.matched.folder.c_str());
        const auto& platforms = session.description->platforms;
        if (std::find(platforms.begin(), platforms.end(), command.platform) == platforms.end()) {
            print_error("Build " + pack::format_build_id(session.build_id) + " of title " + session.title_id +
                        " does not support packing for " + command.platform + " in its title settings.");
            return 1;
        }
    }
    if (caps.pipeline_list_required && !session.ios && session.description->pso_manifest.empty()) {
        print_error("The matched title settings have no pipeline list.");
        return 1;
    }
    if (input->collect_addons(session) != 0) return 1;

    const fs::path compile_cache = pack::default_shader_compile_cache(session.title_id);
    if (input->build_shaders(session, compile_cache) != 0) return 1;

    session.manifests = pack::select_pack_manifests(*session.description, session.armigo_root / "shaders", caps);
    if (!session.ios) {
        if (session.manifests.empty()) {
            std::printf("Pipeline warm-up skipped: the title settings have no pipeline list.\n");
            std::fflush(stdout);
        } else {
            if (const std::string low = require_space(command.output); !low.empty()) {
                print_error(low);
                return 1;
            }
            Step step{"Build pipeline archive"};
            const std::string chain{armigo::runtime::expected_shader_chain_id()};
            if (chain.empty()) {
                print_error("This build has no shader chain, so shaders and pipelines cannot be packed.");
                return 1;
            }
            auto built = pack::build_pack_pipelines(session.armigo_root / "shaders", session.manifests,
                                                     session.armigo_root / "pipelines", chain);
            if (const auto* failure = std::get_if<std::string>(&built)) {
                print_error(*failure);
                return 1;
            }
            const auto& pipelines = std::get<armigo::platform::PipelineArchiveResult>(built);
            std::printf("Pipeline archive: %llu rows, %llu built, %llu unresolved, %llu failed, %s\n",
                        static_cast<unsigned long long>(pipelines.rows),
                        static_cast<unsigned long long>(pipelines.built),
                        static_cast<unsigned long long>(pipelines.unresolved),
                        static_cast<unsigned long long>(pipelines.failed), pipelines.archive.c_str());
        }
    }
    if (command.shaders_only) {
        return finish_shaders_only(session.partial, command.output, session.game, session.started);
    }
    if (caps.embeds_firmware && input->extract_firmware(session) != 0) return 1;

    if (session.ios) {
        const fs::path embedded = session.partial.staging / "embedded";
        {
            Step step{"Capture embedded guest code"};
            const std::string failure = pack::run_pack_tool(
                {session.preparer.string(), "--embedded", "--exefs", (session.game / "exefs").string(), "--output",
                 embedded.string(), "--patches", (session.matched.folder / "profile.json").string()},
                true);
            if (!failure.empty()) {
                print_error(failure);
                return 1;
            }
        }
        {
            Step step{"Verify embedded guest code"};
            const std::string failure = pack::run_pack_tool(
                {session.preparer.string(), "--embedded", "--verify", "--exefs", (session.game / "exefs").string(),
                 "--output", embedded.string(), "--patches", (session.matched.folder / "profile.json").string()},
                true);
            if (!failure.empty()) {
                print_error(failure);
                return 1;
            }
        }
        Step step{"Link iOS player"};
        auto build = pack::prepare_ios_player_build(command.ios_build, session.executable, session.translator);
        if (auto* failure = std::get_if<std::string>(&build)) {
            print_error(*failure);
            return 1;
        }
        session.linked_player = session.partial.staging / session.executable_name;
        pack::IosLinkOptions options;
        options.player_build_directory = std::get<fs::path>(build);
        options.embedded_directory = embedded;
        options.work_directory = session.partial.staging / "generated";
        options.output_executable = session.linked_player;
        auto linked = pack::link_ios_player(options);
        if (auto* failure = std::get_if<std::string>(&linked)) {
            print_error(*failure);
            return 1;
        }
        const auto& report = std::get<pack::IosLinkReport>(linked);
        std::printf("Embedded modules: %zu; executable bytes: %zu.\n", report.modules, report.executable_bytes);
    }

    {
        Step step{"Assemble application"};
        pack::BundleSpec spec;
        spec.app_path = session.assembly;
        spec.title_version = session.title_version;
        spec.development = command.development;
        spec.development_metal_hud = command.development_metal_hud;
        if (input->fill_bundle_identity(session, spec) != 0) return 1;
        pack::CommonBundleFields fields;
        fields.title_id = session.title_id;
        fields.build_id = pack::format_build_id(session.build_id);
        fields.commit = ARMIGO_PACK_COMMIT;
        fields.player = session.linked_player;
        fields.title_folder = session.matched.folder;
        fields.platform = command.platform;
        // The current title-mode selector is a macOS launch feature. iOS embeds
        // the profile's default preparation and keeps its existing UI. A .nup pack
        // is macOS only, so its frame rates are applied whenever the profile has them.
        fields.apply_frame_rates = !session.ios;
        pack::fill_common_bundle_fields(spec, fields, *session.description);
        auto assembled = pack::assemble_application(spec);
        if (const auto* failure = std::get_if<std::string>(&assembled)) {
            print_error(*failure);
            return 1;
        }
    }

    if (!session.ios) {
        if (const std::string failure = pack::publish_application(session.assembly, session.app_path);
            !failure.empty()) {
            print_error(failure);
            return 1;
        }
        session.partial.app = session.app_path;
        if (!session.manifests.empty()) {
            // The warm-up runs the player from its final path, so its Metal
            // caches belong to the app's own bundle identifier. A list packaged
            // with the shaders moved with the app, so the lists are chosen
            // again from the published app.
            Step step{"Warm player Metal caches"};
            const auto manifests = pack::warm_up_manifests(*session.description, session.app_path, caps);
            const std::string failure = pack::warm_pack_pipelines(session.app_path, manifests);
            if (!failure.empty()) {
                print_error(failure);
                return 1;
            }
        }
        if (const std::string failure = pack::register_application(session.app_path); !failure.empty()) {
            std::printf("Warning: %s The Finder may show the app as unable to run until it is opened once.\n",
                        failure.c_str());
        }
    } else if (!command.no_ipa) {
        Step step{"Create unsigned IPA"};
        const std::string failure = pack::create_ios_ipa(session.assembly, session.app_path);
        if (!failure.empty()) {
            print_error(failure);
            return 1;
        }
    }

    fs::path output_path = session.app_path;
    if (session.ios) {
        std::error_code error;
        fs::remove_all(session.partial.staging, error);
        session.partial.staging.clear();
        if (!command.no_ipa) {
            output_path = session.app_path.parent_path() / (session.app_path.stem().string() + ".ipa");
        }
        if (!command.keep_app && !command.no_ipa) {
            fs::remove_all(session.assembly, error);
            if (error) {
                print_error("Cannot remove the temporary iOS app after creating the IPA.");
                return 1;
            }
            fs::remove(session.partial.sidecar, error);
            if (error) {
                print_error("Cannot remove the temporary iOS entitlements after creating the IPA.");
                return 1;
            }
            session.partial.app.clear();
            session.partial.sidecar.clear();
        } else {
            if (const std::string failure = pack::publish_application(session.assembly, session.kept_app);
                !failure.empty()) {
                print_error(failure);
                return 1;
            }
            session.partial.app = session.kept_app;
            if (command.no_ipa) output_path = session.kept_app;
        }
    }
    session.partial.keep = true;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - session.started).count();
    std::error_code size_error;
    const auto bytes = session.ios && !command.no_ipa ? fs::file_size(output_path, size_error)
                                                       : directory_bytes(output_path);
    if (size_error) {
        print_error("Cannot measure the packed output.");
        return 1;
    }
    std::printf("Packed %s (%s) in %.1f s\n", output_path.c_str(), format_gib(bytes).c_str(), seconds);
    std::fflush(stdout);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const auto parsed = pack::parse_command(argc, argv);
    if (const auto* failure = std::get_if<pack::CommandFailure>(&parsed)) {
        print_error(failure->detail);
        armigo::diag::error("%s", pack::command_usage().c_str());
        return failure->exit_code;
    }
    const pack::Command command = std::get<pack::Command>(parsed);
    if (command.help) {
        std::fputs(pack::command_usage().c_str(), stdout);
        return 0;
    }
    if (!command.compare.empty()) {
        const auto compared = pack::compare_packages(command.compare[0], command.compare[1]);
        if (const auto* failure = std::get_if<std::string>(&compared)) {
            print_error(*failure);
            return 1;
        }
        std::fputs(pack::format_comparison(std::get<pack::PackageComparison>(compared)).c_str(), stdout);
        return 0;
    }
    if (command.run) return pack::run_game(command, current_executable());
    try {
        return pack_flow(command);
    } catch (const std::bad_alloc&) {
        print_error("Host memory for packing could not be allocated.");
        return 1;
    } catch (const std::exception& failure) {
        print_error(failure.what());
        return 1;
    }
}
