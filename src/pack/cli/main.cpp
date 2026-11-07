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
#include "pipelines.h"
#include "settings.h"
#include "shaders.h"
#include "space.h"

#include <armigo/compat/title_settings.h>
#include <armigo/loader/container.h>
#include <armigo/pack/ios_link.h>
#include <armigo/pack/naming.h>
#include <armigo/runtime/shader_chain.h>
#include <armigo/shader/ios_package.h>

#include "inputs.h"
#include "pack_commit.h"

#include <mach-o/dyld.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
    bool keep = false;

    ~PartialOutput() {
        if (keep) {
            return;
        }
        std::error_code error;
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

std::optional<std::array<std::byte, 32>> read_main_build(const fs::path& exefs, std::string& error) {
    const fs::path main_file = fs::is_regular_file(exefs / "main") ? exefs / "main" : exefs / "main.nso";
    auto read = pack::read_nso_build_id(main_file);
    if (auto* failure = std::get_if<std::string>(&read)) {
        error = *failure;
        return std::nullopt;
    }
    return std::get<std::array<std::byte, 32>>(read);
}

int pack_game(const pack::Command& command) {
    const auto started = std::chrono::steady_clock::now();
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
        if (!ios && !fs::is_regular_file(player, error)) {
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

    loader::ContainerKeys keys;
    loader::ContainerKeys title_keys;
    {
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
    }

    PartialOutput firmware_partial;
    firmware_partial.staging = command.output / ".firmware-input";
    fs::path firmware_input;
    {
        Step step{"Prepare firmware"};
        auto prepared = pack::prepare_firmware_input(command.firmware, firmware_partial.staging);
        if (const auto* failure = std::get_if<std::string>(&prepared)) {
            print_error(*failure);
            return 1;
        }
        firmware_input = std::get<fs::path>(prepared);
    }

    pack::Inputs inputs;
    {
        Step step{"Classify inputs"};
        std::vector<fs::path> paths;
        paths.push_back(command.game);
        paths.insert(paths.end(), command.updates.begin(), command.updates.end());
        paths.insert(paths.end(), command.dlc.begin(), command.dlc.end());
        paths.push_back(firmware_input);
        pack::InputsOptions options;
        options.keys_file = command.keys;
        auto classified = pack::classify_inputs(paths, options);
        if (auto* failure = std::get_if<pack::InputsFailure>(&classified)) {
            print_error(failure->detail);
            return 1;
        }
        inputs = std::get<pack::Inputs>(std::move(classified));
        print_inputs(inputs, firmware_input);
    }

    pack::PackedNames names;
    {
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
    const fs::path app_path = command.output / (names.file_name + ".app");
    const std::uint32_t title_version = inputs.patch ? inputs.patch->version : inputs.application.version;
    std::error_code exists_error;
    if (fs::exists(app_path, exists_error)) {
        print_error("The packed app already exists: " + app_path.string());
        return 1;
    }
    const std::string title_id = pack::format_title_id(inputs.application.title_id);
    std::optional<fs::path> reuse_app;
    if (ios) {
        Step step{"Find reusable game data"};
        auto selected = pack::select_macos_reuse(command.reuse, command.output, app_path, title_id);
        if (auto* failure = std::get_if<std::string>(&selected)) {
            print_error(*failure);
            return 1;
        }
        reuse_app = std::get<std::optional<fs::path>>(std::move(selected));
    }

    std::uint64_t expected_bytes = pack::kPackMarginBytes;
    if (ios || !reuse_app) {
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
        const std::uint64_t copies = ios && !reuse_app ? 2 : 1;
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
    {
        Step step{"Check free space"};
        const pack::SpaceCheck space = pack::check_free_space(command.output, expected_bytes);
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
    }

    PartialOutput partial;
    partial.app = app_path;
    if (ios) {
        partial.sidecar = command.output / "entitlements.plist";
    }
    partial.staging = command.output / (ios ? ".pack-stages-ios" : ".pack-stages");
    const fs::path armigo_root = ios ? app_path / "Armigo" : app_path / "Contents" / "Resources" / "Armigo";
    const fs::path game = armigo_root / "game";
    std::optional<pack::PackedIdentity> reused_identity;
    loader::ContainerExtractionOptions extraction;
    extraction.keys = keys;
    extraction.title_keys = title_keys;
    extraction.output_directory = game;
    if (inputs.patch) {
        extraction.update_container_file = inputs.patch->location.file;
    }
    if (reuse_app) {
        Step step{"Clone game data"};
        auto cloned = pack::clone_macos_game(*reuse_app, game, title_id);
        if (auto* failure = std::get_if<std::string>(&cloned)) {
            print_error(*failure);
            return 1;
        }
        reused_identity = std::get<pack::PackedIdentity>(std::move(cloned));
        std::printf("Reused %s\n", reuse_app->c_str());
    } else {
        Step step{"Extract game"};
        auto extracted = loader::extract_program_container(inputs.application.location.file, extraction);
        if (auto* failure = std::get_if<loader::ContainerFailure>(&extracted)) {
            print_error(container_error(*failure));
            return 1;
        }
    }

    std::array<std::byte, 32> build_id{};
    pack::TitleMatch matched;
    armigo::compat::TitleSettings settings;
    const armigo::compat::PatchDescription* description = nullptr;
    {
        Step step{"Match title settings"};
        std::string build_error;
        auto read_build = read_main_build(game / "exefs", build_error);
        if (!read_build) {
            print_error(build_error);
            return 1;
        }
        build_id = *read_build;
        if (reused_identity && reused_identity->build_id != pack::format_build_id(build_id)) {
            print_error("The reuse app build id does not match its cloned ExeFS.");
            return 1;
        }
        auto match = pack::match_title_build(inputs.application.title_id, build_id, executable);
        if (auto* failure = std::get_if<std::string>(&match)) {
            print_error(*failure);
            return 1;
        }
        matched = std::get<pack::TitleMatch>(match);
        auto read_settings = armigo::compat::read_title_settings(matched.folder, inputs.application.title_id);
        if (auto* failure = std::get_if<std::string>(&read_settings)) {
            print_error(*failure);
            return 1;
        }
        settings = std::get<armigo::compat::TitleSettings>(std::move(read_settings));
        for (const armigo::compat::PatchDescription& patch : settings.patches) {
            if (patch.identity.main_build_id == build_id) {
                description = &patch;
                break;
            }
        }
        if (description == nullptr) {
            print_error("No title settings describe build id " + pack::format_build_id(build_id) + ".");
            return 1;
        }
        std::printf("Title settings %s\n", matched.folder.c_str());
    }
    if (!ios && description->pso_manifest.empty()) {
        print_error("The matched title settings have no pipeline list.");
        return 1;
    }

    std::vector<fs::path> addon_directories;
    if (reused_identity) {
        Step step{"Collect reused add-on content"};
        auto collected = pack::reused_addon_directories(game);
        if (const auto* failure = std::get_if<std::string>(&collected)) {
            print_error(*failure);
            return 1;
        }
        addon_directories = std::get<std::vector<fs::path>>(std::move(collected));
    } else {
        Step step{"Extract add-on content"};
        for (const pack::ContentSet& addon : inputs.add_on_content) {
            const std::string data_id = pack::format_title_id(addon.title_id);
            const fs::path destination = game / "addons" / data_id;
            extraction.output_directory = destination;
            extraction.update_container_file.clear();
            auto extracted = loader::extract_data_romfs(addon.location.file, extraction, addon.title_id);
            if (auto* failure = std::get_if<loader::ContainerFailure>(&extracted)) {
                print_error(data_id + ": " + container_error(*failure));
                return 1;
            }
            addon_directories.push_back(destination);
            std::printf("Add-on %s\n  %s\n", data_id.c_str(), destination.c_str());
        }
    }
    if (reused_identity) {
        std::vector<std::string> selected_addons;
        selected_addons.reserve(inputs.add_on_content.size());
        for (const pack::ContentSet& addon : inputs.add_on_content) {
            selected_addons.push_back(pack::format_title_id(addon.title_id));
        }
        const std::string mismatch = pack::validate_reused_selection(
            *reused_identity, addon_directories, title_id, pack::format_build_id(build_id), title_version,
            selected_addons);
        if (!mismatch.empty()) {
            print_error(mismatch);
            return 1;
        }
    }

    fs::path compile_cache;
    if (std::getenv("ARMIGO_SHADER_COMPILE_CACHE") == nullptr) {
        if (const char* home = std::getenv("HOME")) {
            compile_cache = fs::path(home) / "Library" / "Caches" / "Armigo" / title_id / "compile-cache";
        }
    }
    const bool reuse_shaders = ios && reuse_app.has_value();
    const auto shader_paths = pack::shader_stage_paths(
        partial.staging, armigo_root, reuse_app.value_or(fs::path{}), ios);
    const fs::path& macos_shaders = shader_paths.macos_package;
    if (reuse_shaders) {
        std::printf("Reusing macOS shader package %s\n", macos_shaders.c_str());
    } else {
        if (auto low = require_space(command.output); !low.empty()) {
            print_error(low);
            return 1;
        }
        Step step{"Translate shaders"};
        pack::ShaderPackInput shader_input;
        const auto& selected = description->shader_inputs;
        if (selected.containers == armigo::compat::ShaderContainerSet::GfdCache) {
            shader_input.containers = pack::ShaderPackInput::Containers::GfdCache;
        }
        if (selected.vertex_depth == armigo::compat::ShaderVertexDepth::ZeroToOne) {
            shader_input.vertex_depth = pack::ShaderPackInput::VertexDepth::ZeroToOne;
        }
        shader_input.caches = selected.caches;
        shader_input.fallbacks.reserve(selected.fallbacks.size());
        for (const auto& fallback : selected.fallbacks) {
            shader_input.fallbacks.emplace_back(fallback.cache, fallback.alternatives);
        }
        shader_input.ignored_needle_offsets = selected.ignored_needle_offsets;
        auto built = pack::build_shader_package(game / "exefs", game / "romfs", addon_directories,
                                                 shader_paths.enumeration, macos_shaders, compile_cache, translator,
                                                 shader_input);
        if (auto* failure = std::get_if<std::string>(&built)) {
            print_error(*failure);
            return 1;
        }
        const auto& shaders = std::get<pack::ShaderReport>(built);
        std::printf("Shader stages translated: %zu (enumerated %zu, unsupported %zu, structural errors %zu).\n",
                    shaders.translated, shaders.enumerated, shaders.unsupported, shaders.structural_errors);
    }
    if (ios) {
        Step step{"Build iOS shader package"};
        armigo::shader::package::IosBuildOptions options;
        options.macos_package = macos_shaders;
        options.output = armigo_root / "shaders";
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

    if (!ios) {
        if (auto low = require_space(command.output); !low.empty()) {
            print_error(low);
            return 1;
        }
        Step step{"Build pipeline archive"};
        const std::string chain{armigo::runtime::expected_shader_chain_id()};
        if (chain.empty()) {
            print_error("This build has no shader chain, so shaders and pipelines cannot be packed.");
            return 1;
        }
        auto built = pack::build_pack_pipelines(armigo_root / "shaders", description->pso_manifest,
                                                 armigo_root / "pipelines", chain);
        if (auto* failure = std::get_if<std::string>(&built)) {
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

    {
        Step step{"Extract firmware resources"};
        auto resources = pack::extract_firmware_resources(inputs, firmware_input, keys, armigo_root / "system");
        if (const auto* failure = std::get_if<std::string>(&resources)) {
            print_error(*failure);
            return 1;
        }
        const auto& firmware = std::get<pack::FirmwareResources>(resources);
        std::printf("Firmware version %s.\n", firmware.version.c_str());
        if (firmware.version != "20.5.0") {
            std::printf("Warning: firmware %s is outside the 20.5.0 compatibility baseline.\n",
                        firmware.version.c_str());
        }
    }

    fs::path linked_player = player;
    if (ios) {
        const fs::path embedded = partial.staging / "embedded";
        {
            Step step{"Capture embedded guest code"};
            const std::string failure = pack::run_pack_tool({
                preparer.string(), "--embedded", "--exefs", (game / "exefs").string(),
                "--output", embedded.string(), "--patches", (matched.folder / "profile.json").string()}, true);
            if (!failure.empty()) {
                print_error(failure);
                return 1;
            }
        }
        {
            Step step{"Verify embedded guest code"};
            const std::string failure = pack::run_pack_tool({
                preparer.string(), "--embedded", "--verify", "--exefs", (game / "exefs").string(),
                "--output", embedded.string(), "--patches", (matched.folder / "profile.json").string()}, true);
            if (!failure.empty()) {
                print_error(failure);
                return 1;
            }
        }
        Step step{"Link iOS player"};
        auto build = pack::prepare_ios_player_build(command.ios_build, executable, translator);
        if (auto* failure = std::get_if<std::string>(&build)) {
            print_error(*failure);
            return 1;
        }
        linked_player = partial.staging / "Armigo";
        pack::IosLinkOptions options;
        options.player_build_directory = std::get<fs::path>(build);
        options.embedded_directory = embedded;
        options.work_directory = partial.staging / "generated";
        options.output_executable = linked_player;
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
        auto icon = ios ? pack::jpeg_to_ios_png(jpeg) : pack::jpeg_to_icns(jpeg);
        if (auto* failure = std::get_if<std::string>(&icon)) {
            print_error(*failure);
            return 1;
        }
        const auto& icon_bytes = std::get<std::vector<std::byte>>(icon);
        pack::BundleSpec spec;
        spec.app_path = app_path;
        spec.display_name = names.display_name;
        spec.bundle_name = names.file_name;
        spec.bundle_identifier = names.bundle_identifier;
        spec.title_id = title_id;
        spec.build_id = pack::format_build_id(build_id);
        spec.title_version = title_version;
        spec.platform = command.platform;
        spec.commit = ARMIGO_PACK_COMMIT;
        spec.player = linked_player;
        spec.title_folder = matched.folder;
        spec.icon = icon_bytes;
        spec.languages = names.languages;
        spec.localized_titles = names.localized_titles;
        spec.supported_language_flag = names.supported_language_flag;
        // The current title-mode selector is a macOS launch feature. iOS
        // embeds the profile's default preparation and keeps its existing UI.
        if (!ios && description->frame_rate) {
            spec.default_frame_rate = description->frame_rate->default_rate;
            spec.display_refresh_rate = description->frame_rate->display_refresh_rate;
            spec.supported_frame_rates = description->frame_rate->supported_rates;
            spec.experimental_frame_rates = description->frame_rate->experimental_rates;
        }
        auto assembled = pack::assemble_application(spec);
        if (auto* failure = std::get_if<std::string>(&assembled)) {
            print_error(*failure);
            return 1;
        }
    }

    if (!ios) {
        Step step{"Warm player Metal caches"};
        const std::string failure = pack::warm_pack_pipelines(app_path, description->pso_manifest);
        if (!failure.empty()) {
            print_error(failure);
            return 1;
        }
    } else if (!command.no_ipa) {
        Step step{"Create unsigned IPA"};
        const std::string failure = pack::create_ios_ipa(app_path);
        if (!failure.empty()) {
            print_error(failure);
            return 1;
        }
    }

    fs::path output_path = app_path;
    if (ios) {
        std::error_code error;
        fs::remove_all(partial.staging, error);
        partial.staging.clear();
        if (!command.no_ipa) output_path = app_path.parent_path() / (app_path.stem().string() + ".ipa");
        if (!command.keep_app && !command.no_ipa) {
            fs::remove_all(app_path, error);
            if (error) {
                print_error("Cannot remove the temporary iOS app after creating the IPA.");
                return 1;
            }
            fs::remove(partial.sidecar, error);
            if (error) {
                print_error("Cannot remove the temporary iOS entitlements after creating the IPA.");
                return 1;
            }
            partial.app.clear();
            partial.sidecar.clear();
        }
    }
    partial.keep = true;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::error_code size_error;
    const auto bytes = ios && !command.no_ipa ? fs::file_size(output_path, size_error) : directory_bytes(output_path);
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
    try {
        return pack_game(command);
    } catch (const std::bad_alloc&) {
        print_error("Host memory for packing could not be allocated.");
        return 1;
    } catch (const std::exception& failure) {
        print_error(failure.what());
        return 1;
    }
}
