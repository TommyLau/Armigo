// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/ios_package.h>
#include <armigo/shader/package.h>
#include <armigo/shader/source_version.h>
#include <armigo/shader_input/shader_input.h>
#include <armigo/shader_translation/stage_translation.h>

#include "../translation/stage_fixture.h"

#include <array>
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <variant>
#include <vector>

extern char** environ;

namespace {

namespace fs = std::filesystem;
namespace package = armigo::shader::package;
namespace si = armigo::shader_input;
namespace tr = armigo::shader::translation;

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

void write(const fs::path& path, std::string_view data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(data.data(), static_cast<std::streamsize>(data.size()));
    CHECK(stream.good());
}

void write(const fs::path& path, const std::vector<std::byte>& data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    CHECK(stream.good());
}

std::string read(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::vector<std::byte> bytes(const fs::path& path) {
    const std::string data = read(path);
    std::vector<std::byte> result(data.size());
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}

std::string manifest_value(const std::string& manifest, std::string_view key) {
    const std::string prefix = "\"" + std::string(key) + "\":\"";
    const auto begin = manifest.find(prefix);
    CHECK(begin != std::string::npos);
    if (begin == std::string::npos) return {};
    const auto end = manifest.find('"', begin + prefix.size());
    CHECK(end != std::string::npos);
    return end == std::string::npos ? std::string{} : manifest.substr(begin + prefix.size(), end - begin - prefix.size());
}

void store_u32(std::vector<std::byte>& data, std::size_t offset, std::uint32_t value) {
    std::memcpy(data.data() + offset, &value, sizeof value);
}

struct Fixture {
    fs::path root = fs::temp_directory_path() / ("armigo-ios-package-" + std::to_string(::getpid()));
    fs::path bin = root / "bin";
    fs::path log = root / "xcrun.log";
    fs::path cache = root / "compile-cache";
    fs::path macos = root / "macos-package";
    fs::path stage = root / "translated-stage";

    Fixture() {
        fs::remove_all(root);
        fs::create_directories(bin);
        const fs::path translate = bin / "translate";
        write(translate, "#!/bin/sh\n[ \"$1\" = --version ] && printf '" +
                         std::string(armigo::shader::source_version) + "\\n'\n");
        ::chmod(translate.c_str(), 0755);
        write(bin / "metal-tool", R"SH(#!/bin/sh
printf '%s\n' "metal $*" >> "$FAKE_XCRUN_LOG"
if [ "$FAKE_FAIL" = metal ]; then printf 'synthetic failure\n' >&2; exit 7; fi
out=
source=
previous=
for argument in "$@"; do
    if [ "$previous" = -o ]; then out=$argument; fi
    if [ "$previous" = -c ]; then source=$argument; fi
    previous=$argument
done
printf AIR > "$out"
cat "$source" >> "$out"
)SH");
        write(bin / "metallib-tool", R"SH(#!/bin/sh
printf '%s\n' "metallib $*" >> "$FAKE_XCRUN_LOG"
if [ "$FAKE_FAIL" = metallib ]; then printf 'synthetic failure\n' >&2; exit 7; fi
out=
previous=
for argument in "$@"; do
    if [ "$previous" = -o ]; then out=$argument; fi
    previous=$argument
done
printf LIB > "$out"
cat "$1" >> "$out"
)SH");
        ::chmod((bin / "metal-tool").c_str(), 0755);
        ::chmod((bin / "metallib-tool").c_str(), 0755);
        const fs::path xcrun = bin / "xcrun";
        write(xcrun, R"SH(#!/bin/sh
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$2" = macosx ] || [ "$2" = iphoneos ]; then
    if [ "$3" = --find ] && [ "$4" = metal ]; then printf '%s\n' "$ROOT/metal-tool"; exit; fi
    if [ "$3" = --find ] && [ "$4" = metallib ]; then printf '%s\n' "$ROOT/metallib-tool"; exit; fi
    if [ "$3 $4" = "metal --version" ]; then printf 'Apple metal synthetic 4.0\n'; exit; fi
    if [ "$3" = --show-sdk-version ]; then printf '26.1\n'; exit; fi
    if [ "$3" = --show-sdk-build-version ]; then printf '26B77\n'; exit; fi
fi
printf 'unexpected xcrun query: %s\n' "$*" >&2
exit 9
)SH");
        ::chmod(xcrun.c_str(), 0755);
        const std::string path = bin.string() + ":/usr/bin:/bin";
        ::setenv("PATH", path.c_str(), 1);
        ::setenv("FAKE_XCRUN_LOG", log.c_str(), 1);
        ::unsetenv("FAKE_FAIL");
        ::unsetenv("ARMIGO_SHADER_COMPILE_CACHE");

        std::vector<std::byte> control(si::format::control_min_size);
        store_u32(control, 0, si::format::control_magic);
        store_u32(control, 4, si::format::control_tag);
        store_u32(control, si::format::control_version_offset, 14);
        store_u32(control, si::format::control_size_offset, static_cast<std::uint32_t>(control.size()));
        store_u32(control, si::format::c1_offset_field, 112);
        store_u32(control, si::format::code_size_field, 112);
        store_u32(control, si::format::code_offset_field, 0);
        store_u32(control, si::format::stage_field, static_cast<std::uint32_t>(si::Stage::Compute));
        std::vector<std::byte> code(112);
        const std::uint64_t exit = 0xE30000000007000Full;
        std::memcpy(code.data() + 8, &exit, sizeof exit);
        tr::StageOptions translated_options;
        translated_options.texture_handle_cbuf = 2;
        translated_options.texture_descriptors_derived = true;
        translated_options.compute_local_size = std::array<std::uint32_t, 3>{4, 4, 2};
        translated_options.compute_shared_memory_bytes = 0;
        const auto translated = tr::translate_stage(control, code, translated_options);
        fs::create_directory(stage);
        write(stage / "program.bin", translated.program);
        write(stage / "source.metal", translated.source);
        write(stage / "reflection.json", translated.reflection);

        const auto graphics = armigo::test::stage_fixture::fragment_stage();
        tr::StageOptions graphics_options;
        graphics_options.texture_handle_cbuf = 2;
        graphics_options.texture_descriptors_derived = true;
        const auto graphics_translated = tr::translate_stage(graphics.control, graphics.code, graphics_options);
        const fs::path graphics_stage = root / "graphics-stage";
        fs::create_directory(graphics_stage);
        write(graphics_stage / "program.bin", graphics_translated.program);
        write(graphics_stage / "source.metal", graphics_translated.source);
        write(graphics_stage / "reflection.json", graphics_translated.reflection);

        package::BuildOptions build;
        build.stages = {stage, graphics_stage};
        build.translate_tool = translate;
        build.output = macos;
        build.compile_cache = cache;
        build.jobs = 2;
        const auto built = package::build(build);
        if (const auto* error = std::get_if<std::string>(&built)) std::printf("macOS build: %s\n", error->c_str());
        CHECK(std::holds_alternative<package::BuildReport>(built));
    }

    ~Fixture() { fs::remove_all(root); }

    package::IosBuildOptions options(fs::path output) {
        package::IosBuildOptions result;
        result.macos_package = macos;
        result.output = root / output;
        result.jobs = 2;
        return result;
    }
};

void test_identity_and_cache(Fixture& fixture) {
    auto options = fixture.options("ios-a");
    options.compile_cache = fixture.cache;
    const auto built = package::build_ios_package(options);
    if (const auto* error = std::get_if<std::string>(&built)) std::printf("iOS build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::IosBuildReport>(built));
    if (const auto* report = std::get_if<package::IosBuildReport>(&built)) CHECK(report->converted_stages == 2);
    const std::string macos_manifest = read(fixture.macos / "manifest.json");
    const std::string ios_manifest = read(options.output / "manifest.json");
    // The player and packer use one synthetic translator, SDK and tool set.
    // Compare their real canonical profiles and generated admission header.
    std::array<std::string, 4> parity_arguments{ARMIGO_TEST_PYTHON, ARMIGO_IOS_CHAIN_PARITY_SCRIPT,
        (fixture.bin / "translate").string(), (options.output / "manifest.json").string()};
    std::array<char*, 5> parity_argv{parity_arguments[0].data(), parity_arguments[1].data(),
        parity_arguments[2].data(), parity_arguments[3].data(), nullptr};
    pid_t child = 0;
    const int spawned = ::posix_spawn(&child, parity_argv[0], nullptr, nullptr, parity_argv.data(), environ);
    CHECK(spawned == 0);
    if (spawned == 0) {
        int status = 0; pid_t observed;
        do { observed = ::waitpid(child, &status, 0); } while (observed == -1 && errno == EINTR);
        CHECK(observed == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    CHECK(manifest_value(macos_manifest, "platform") == "macos-metal4");
    CHECK(manifest_value(ios_manifest, "platform") == "ios-metal4");
    CHECK(manifest_value(macos_manifest, "chain_id") != manifest_value(ios_manifest, "chain_id"));
    CHECK(manifest_value(macos_manifest, "source_version") == manifest_value(ios_manifest, "source_version"));
    CHECK(ios_manifest.find("-mios-version-min=26.0") != std::string::npos);
    CHECK(ios_manifest.find("-mmacosx-version-min=26.0") == std::string::npos);
    const std::string variant = manifest_value(ios_manifest, "variant_id");
    CHECK(variant == manifest_value(macos_manifest, "variant_id"));
    const fs::path input_stage = fixture.macos / "stages" / variant;
    const fs::path output_stage = options.output / "stages" / variant;
    CHECK(bytes(input_stage / "program.bin") == bytes(output_stage / "program.bin"));
    CHECK(read(input_stage / "reflection.json") == read(output_stage / "reflection.json"));
    CHECK(!fs::exists(output_stage / "source.metal"));
    CHECK(read(output_stage / "shader.metallib").starts_with("LIBAIR"));
    CHECK(read(fixture.log).find("-std=metal4.0 -mios-version-min=26.0 -fwrapv -Werror") != std::string::npos);
    std::size_t cache_entries = 0;
    for (const auto& entry : fs::directory_iterator(fixture.cache)) {
        if (entry.path().extension() == ".metallib") ++cache_entries;
    }
    CHECK(cache_entries == 4);

    const std::string cold_commands = read(fixture.log);
    options.output = fixture.root / "ios-b";
    CHECK(std::holds_alternative<package::IosBuildReport>(package::build_ios_package(options)));
    CHECK(read(fixture.log) == cold_commands);
    CHECK(read(options.output / "manifest.json") == ios_manifest);

    options.macos_package = options.output;
    options.output = fixture.root / "wrong-platform";
    const auto wrong = package::build_ios_package(options);
    CHECK(std::holds_alternative<std::string>(wrong));
    if (const auto* error = std::get_if<std::string>(&wrong)) {
        CHECK(error->find("requires a macos-metal4 package") != std::string::npos);
    }
    CHECK(!fs::exists(options.output));
}

std::size_t count_occurrences(std::string_view text, std::string_view needle) {
    std::size_t count = 0;
    for (std::size_t at = 0; (at = text.find(needle, at)) != std::string_view::npos; at += needle.size()) ++count;
    return count;
}

// A macOS manifest that records sample hoisting converts, and the iOS manifest
// carries the same field. A manifest without it stays without it. The field is
// outside the profile, so the two iOS chain ids match.
void test_sample_hoisting_carries(Fixture& fixture) {
    package::BuildOptions build;
    build.stages = {fixture.stage, fixture.root / "graphics-stage"};
    build.translate_tool = fixture.bin / "translate";
    build.output = fixture.root / "macos-hoisted";
    build.jobs = 2;
    build.hoist_implicit_samples = true;
    const auto macos_built = package::build(build);
    if (const auto* error = std::get_if<std::string>(&macos_built))
        std::printf("hoisted macOS build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::BuildReport>(macos_built));
    const std::string hoisted_macos = read(build.output / "manifest.json");
    const std::string plain_macos = read(fixture.macos / "manifest.json");
    CHECK(count_occurrences(plain_macos, "sample_hoisting") == 0);
    CHECK(hoisted_macos.find("\"sample_hoisting\":\"implicit\"") != std::string::npos);
    CHECK(count_occurrences(hoisted_macos, "sample_hoisting") == 1);
    CHECK(manifest_value(hoisted_macos, "chain_id") == manifest_value(plain_macos, "chain_id"));

    auto hoisted = fixture.options("ios-hoisted");
    hoisted.macos_package = build.output;
    const auto hoisted_built = package::build_ios_package(hoisted);
    if (const auto* error = std::get_if<std::string>(&hoisted_built))
        std::printf("hoisted iOS build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::IosBuildReport>(hoisted_built));
    const std::string hoisted_ios = read(hoisted.output / "manifest.json");
    CHECK(hoisted_ios.find("\"sample_hoisting\":\"implicit\"") != std::string::npos);
    CHECK(count_occurrences(hoisted_ios, "sample_hoisting") == 1);

    auto plain = fixture.options("ios-plain");
    const auto plain_built = package::build_ios_package(plain);
    if (const auto* error = std::get_if<std::string>(&plain_built))
        std::printf("plain iOS build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::IosBuildReport>(plain_built));
    const std::string plain_ios = read(plain.output / "manifest.json");
    CHECK(count_occurrences(plain_ios, "sample_hoisting") == 0);
    CHECK(manifest_value(hoisted_ios, "chain_id") == manifest_value(plain_ios, "chain_id"));
}

void test_compile_failure(Fixture& fixture) {
    auto options = fixture.options("failed-output");
    fs::create_directory(options.output);
    ::setenv("FAKE_FAIL", "metal", 1);
    const auto failed = package::build_ios_package(options);
    ::unsetenv("FAKE_FAIL");
    CHECK(std::holds_alternative<std::string>(failed));
    if (const auto* error = std::get_if<std::string>(&failed)) {
        CHECK(error->find("synthetic failure") != std::string::npos);
    }
    CHECK(fs::is_empty(options.output));
    std::size_t staging_count = 0;
    for (const auto& entry : fs::directory_iterator(fixture.root)) {
        if (entry.path().filename().string().starts_with(".armigo-ios-package-")) ++staging_count;
    }
    CHECK(staging_count == 0);
}

} // namespace

int main() {
    {
        Fixture fixture;
        test_identity_and_cache(fixture);
    }
    {
        Fixture fixture;
        test_sample_hoisting_carries(fixture);
    }
    {
        Fixture fixture;
        test_compile_failure(fixture);
    }
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("shader.ios_package checks passed\n");
    return 0;
}
