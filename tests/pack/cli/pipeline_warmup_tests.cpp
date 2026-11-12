// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "bundle.h"
#include "pipelines.h"
#include <armigo/pack/naming.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <unistd.h>

namespace fs = std::filesystem;
namespace pack = armigo::pack;

namespace {
int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); ++failures; } } while (false)

void write(const fs::path& path, std::string_view text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

std::string read(const fs::path& path) {
    std::ifstream stream(path);
    return {(std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>()};
}

int warm_player(int argc, char** argv) {
    const auto resources = fs::path(argv[0]).parent_path().parent_path() / "Resources/Armigo";
    const std::vector<std::string> expected{
        "pso-archive", "--shaders", (resources / "shaders").string(),
        "--manifest", (resources / "title/seed pipelines.tsv").string(),
        "--manifest", (resources / "shaders/native/compute-pipelines.tsv").string(),
        "--cache", (resources / "pipelines").string(), "--warm-only"};
    if (argc != static_cast<int>(expected.size() + 1)) return 2;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (argv[i + 1] != expected[i]) return 3;
    write(resources / "pipelines/warmed.txt", fs::path(argv[0]).filename().string());
    return fs::exists(resources / "pipelines/fail-warm") ? 17 : 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "pso-archive") return warm_player(argc, argv);
    const fs::path root = fs::temp_directory_path() / ("armigo-pack-warmup-" + std::to_string(getpid()));
    struct Cleanup { fs::path root; ~Cleanup() { fs::remove_all(root); } } cleanup{root};
    std::vector<std::byte> control(pack::nacp::control_file_size);
    const std::string title = "Synthetic Warm-Up: One";
    std::memcpy(control.data(), title.data(), title.size());
    control[0x302c] = std::byte{1};
    const auto named = pack::names_from_control(0x0100000000000001, control);
    CHECK(std::holds_alternative<pack::PackedNames>(named));
    if (!std::holds_alternative<pack::PackedNames>(named)) return 1;
    const auto& names = std::get<pack::PackedNames>(named);
    pack::BundleSpec spec;
    const auto app = root / (names.file_name + ".app");
    spec.app_path = pack::assembly_path(app, getpid());
    spec.display_name = names.display_name;
    spec.file_name = names.file_name;
    spec.executable_name = names.executable_name;
    spec.bundle_identifier = names.bundle_identifier;
    spec.title_id = "0100000000000001";
    spec.platform = "macos";
    spec.languages = names.languages;
    spec.localized_titles = names.localized_titles;
    spec.supported_language_flag = names.supported_language_flag;
    spec.player = fs::absolute(argv[0]);
    spec.title_folder = root / "title";
    write(spec.title_folder / "profile.json", "{\"version\":1}\n");
    write(spec.title_folder / "seed pipelines.tsv", "synthetic title pipeline\n");
    static constexpr std::array<std::byte, 4> icon{
        std::byte{'i'}, std::byte{'c'}, std::byte{'n'}, std::byte{'s'}};
    spec.icon = icon;
    const auto assembled = pack::assemble_application(spec);
    CHECK(std::holds_alternative<fs::path>(assembled));
    if (!std::holds_alternative<fs::path>(assembled)) return 1;
    CHECK(pack::publish_application(spec.app_path, app).empty());
    const auto resources = app / "Contents/Resources/Armigo";
    fs::create_directories(resources / "pipelines");
    write(resources / "shaders/native/compute-pipelines.tsv", "synthetic compute pipeline\n");
    const std::vector<fs::path> manifests{
        resources / "title/seed pipelines.tsv", resources / "shaders/native/compute-pipelines.tsv"};
    const auto player = app / "Contents/MacOS" / names.executable_name;
    CHECK(fs::is_regular_file(player));
    CHECK(!fs::exists(app / "Contents/MacOS/Armigo"));
    const auto warmed = pack::warm_pack_pipelines(app, manifests);
    if (!warmed.empty()) std::printf("%s\n", warmed.c_str());
    CHECK(warmed.empty());
    CHECK(read(resources / "pipelines/warmed.txt") == names.executable_name);

    write(resources / "pipelines/fail-warm", "fail\n");
    CHECK(pack::warm_pack_pipelines(app, manifests).find("warm-up failed") != std::string::npos);
    fs::remove(resources / "pipelines/fail-warm");
    fs::remove(resources / "pipelines/warmed.txt");

    // A legacy executable must not replace missing or invalid bundle metadata.
    fs::copy_file(player, app / "Contents/MacOS/Armigo");
    const auto plist = app / "Contents/Info.plist";
    for (const auto& value : {std::string{}, std::string("<key>CFBundleExecutable</key><integer>1</integer>"),
                              std::string("<key>CFBundleExecutable</key><string>../outside</string>")}) {
        write(plist, "<plist version=\"1.0\"><dict>" + value + "</dict></plist>");
        CHECK(!pack::warm_pack_pipelines(app, manifests).empty());
        CHECK(!fs::exists(resources / "pipelines/warmed.txt"));
    }
    write(plist, "not a property list");
    CHECK(!pack::warm_pack_pipelines(app, manifests).empty());
    fs::remove(plist);
    CHECK(!pack::warm_pack_pipelines(app, manifests).empty());
    write(plist, pack::info_plist(spec));
    fs::remove(player);
    CHECK(pack::warm_pack_pipelines(app, manifests).find("packed player is missing") != std::string::npos);
    CHECK(!fs::exists(resources / "pipelines/warmed.txt"));
    return failures ? 1 : 0;
}
