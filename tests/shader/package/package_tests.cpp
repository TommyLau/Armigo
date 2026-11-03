// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/package.h>

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <cerrno>
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

namespace {

namespace fs = std::filesystem;
namespace package = armigo::shader::package;

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

std::string digest(std::span<const std::byte> bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> value{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), value.data());
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (const unsigned char byte : value) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}

std::string digest(std::string_view text) {
    return digest({reinterpret_cast<const std::byte*>(text.data()), text.size()});
}

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

struct Fixture {
    fs::path root;
    fs::path bin;
    fs::path inputs;
    fs::path log;
    fs::path translate;

    Fixture() {
        root = fs::temp_directory_path() / ("armigo-native-package-" + std::to_string(::getpid()));
        fs::remove_all(root);
        bin = root / "bin";
        inputs = root / "inputs";
        log = root / "xcrun.log";
        fs::create_directories(bin);
        fs::create_directories(inputs);
        translate = bin / "translate";
        write(translate, "#!/bin/sh\n[ \"$1\" = --version ] && printf '" + std::string(64, 'a') + "\\n'\n");
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
if [ "$1 $2 $3 $4" = "--sdk macosx --find metal" ]; then printf '%s\n' "$ROOT/metal-tool"; exit; fi
if [ "$1 $2 $3 $4" = "--sdk macosx --find metallib" ]; then printf '%s\n' "$ROOT/metallib-tool"; exit; fi
if [ "$1 $2 $3 $4" = "--sdk macosx metal --version" ]; then printf 'Apple metal synthetic 4.0\n'; exit; fi
if [ "$1 $2 $3" = "--sdk macosx --show-sdk-version" ]; then printf '26.1\n'; exit; fi
if [ "$1 $2 $3" = "--sdk macosx --show-sdk-build-version" ]; then printf '26B77\n'; exit; fi
printf '%s\n' "$*" >> "$FAKE_XCRUN_LOG"
tool=$3
if [ "$FAKE_FAIL" = "$tool" ]; then printf 'synthetic failure\n' >&2; exit 7; fi
out=
source=
previous=
for argument in "$@"; do
    if [ "$previous" = -o ]; then out=$argument; fi
    if [ "$previous" = -c ]; then source=$argument; fi
    previous=$argument
done
if [ "$tool" = metal ]; then
    printf AIR > "$out"
    cat "$source" >> "$out"
else
    printf LIB > "$out"
    cat "$4" >> "$out"
fi
)SH");
        ::chmod(xcrun.c_str(), 0755);
        const std::string path = bin.string() + ":/usr/bin:/bin";
        ::setenv("PATH", path.c_str(), 1);
        ::setenv("FAKE_XCRUN_LOG", log.c_str(), 1);
        ::unsetenv("FAKE_FAIL");
        ::unsetenv("ARMIGO_SHADER_COMPILE_CACHE");
    }

    ~Fixture() { fs::remove_all(root); }

    fs::path stage(std::string_view name, unsigned seed = 17) {
        const fs::path directory = inputs / name;
        fs::create_directory(directory);
        std::vector<std::byte> program(160);
        for (std::size_t index = 0; index < program.size(); ++index) {
            program[index] = static_cast<std::byte>((index * 29 + seed) & 255);
        }
        const std::uint32_t sph = 1 << 10;
        std::memcpy(program.data(), &sph, sizeof sph);
        const std::string source = "#include <metal_stdlib>\n// " + std::string(name) + "\n";
        const std::string metadata =
            "{\"schema\":\"armigo-shader-stage-v1\",\"source_version\":\"" + std::string(64, 'a') +
            "\",\"program_key\":\"" + digest(program) + "\",\"program_size\":160," +
            "\"program_file\":\"program.bin\",\"source_file\":\"source.metal\",\"source_digest\":\"" +
            digest(source) + "\",\"stage\":\"vertex\",\"entry_point\":\"armigo_vertex_main\"," +
            "\"vertex_depth\":\"minus-one-to-one\",\"fragment_position\":\"unspecified\"," +
            "\"vertex_attributes\":[{\"generic_slot\":0,\"attribute_index\":2,\"format\":\"float32x3\"}]," +
            "\"constant_buffers\":[],\"textures\":[],\"interface\":{\"position_input_mask\":0," +
            "\"position_output_mask\":15,\"color_output_mask\":0,\"generic_inputs\":[{\"slot\":0," +
            "\"component_mask\":7,\"interpolation\":\"screen-linear\"}],\"generic_outputs\":[{\"slot\":0," +
            "\"component_mask\":3,\"interpolation\":\"screen-linear\"}]}}\n";
        write(directory / "program.bin", program);
        write(directory / "source.metal", source);
        write(directory / "reflection.json", metadata);
        return directory;
    }

    package::BuildOptions options(fs::path output) {
        package::BuildOptions result;
        result.translate_tool = translate;
        result.output = root / output;
        result.jobs = 2;
        return result;
    }
};

void test_deterministic_package(Fixture& fixture) {
    const fs::path stage = fixture.stage("vertex");
    auto options = fixture.options("package-a");
    options.stages = {stage, stage, fixture.stage("second", 18), fixture.stage("third", 19),
                      fixture.stage("fourth", 20), fixture.stage("fifth", 21)};
    const auto built = package::build(options);
    if (const auto* error = std::get_if<std::string>(&built)) std::printf("build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::BuildReport>(built));
    if (const auto* report = std::get_if<package::BuildReport>(&built)) {
        CHECK(report->input_stages == 5);
        CHECK(report->unique_stages == 5);
    }
    const std::string variant = "1cd06a449ca44560d3f15f646a5aeb3a8c3a2d40637f57e2016291bb2043e57c";
    const std::string manifest = read(options.output / "manifest.json");
    CHECK(manifest.find("\"chain_id\":\"239cb9f212404965aa640e5ab22406012d8777a211ccada708cac17be5afabcf\"") !=
          std::string::npos);
    CHECK(manifest.find("\"variant_id\":\"" + variant + "\"") != std::string::npos);
    CHECK(manifest.find("\"prefix_key\":\"d8d18c6edafd31857672a329c60b3c204fac4747fec24533d857614ef88df848\"") !=
          std::string::npos);
    CHECK(manifest.find("\"program_size\":160") != std::string::npos);
    CHECK(manifest.find("\"complete_inventory\":false") != std::string::npos);
    CHECK(manifest.find("\"runnable\":false") != std::string::npos);
    CHECK(manifest.back() == '\n');
    const fs::path published = options.output / "stages" / variant;
    CHECK(bytes(published / "program.bin") == bytes(stage / "program.bin"));
    CHECK(read(published / "reflection.json") == read(stage / "reflection.json"));
    const std::string library = read(published / "shader.metallib");
    CHECK(library.starts_with("LIBAIR#include <metal_stdlib>"));
    const std::string commands = read(fixture.log);
    CHECK(commands.find("-std=metal4.0 -mmacosx-version-min=26.0 -Werror -Wno-unused-variable "
                        "-Wno-sometimes-uninitialized -c") != std::string::npos);
    CHECK(commands.find("metallib") != std::string::npos);

    options.output = fixture.root / "package-b";
    options.compile_cache = fixture.root / "compile-cache";
    CHECK(std::holds_alternative<package::BuildReport>(package::build(options)));
    CHECK(read(options.output / "manifest.json") == manifest);
    CHECK(bytes(options.output / "stages" / variant / "shader.metallib") == bytes(published / "shader.metallib"));
    const std::string cold_commands = read(fixture.log);
    options.output = fixture.root / "package-c";
    CHECK(std::holds_alternative<package::BuildReport>(package::build(options)));
    CHECK(read(options.output / "manifest.json") == manifest);
    CHECK(read(fixture.log) == cold_commands);
}

void test_failures(Fixture& fixture) {
    const fs::path stage = fixture.stage("invalid");
    write(stage / "reflection.json", "{\"schema\":1,\"schema\":2}\n");
    auto options = fixture.options("invalid-output");
    options.stages = {stage};
    const auto malformed = package::build(options);
    CHECK(std::holds_alternative<std::string>(malformed));
    CHECK(std::get<std::string>(malformed).find("invalid JSON") != std::string::npos);
    CHECK(!fs::exists(options.output));

    const fs::path valid = fixture.stage("valid");
    options.stages = {valid};
    options.output = fixture.root / "compile-failure";
    fs::create_directory(options.output);
    ::setenv("FAKE_FAIL", "metal", 1);
    const auto failed = package::build(options);
    ::unsetenv("FAKE_FAIL");
    CHECK(std::holds_alternative<std::string>(failed));
    CHECK(std::get<std::string>(failed).find("synthetic failure") != std::string::npos);
    CHECK(fs::is_empty(options.output));
    std::size_t staging_count = 0;
    for (const auto& entry : fs::directory_iterator(fixture.root)) {
        if (entry.path().filename().string().starts_with(".armigo-package-")) ++staging_count;
    }
    CHECK(staging_count == 0);

    options.jobs = 0;
    CHECK(std::holds_alternative<std::string>(package::build(options)));
    options.jobs = 1;
    options.output = valid / "inside";
    CHECK(std::holds_alternative<std::string>(package::build(options)));
}

} // namespace

int main() {
    {
        Fixture fixture;
        test_deterministic_package(fixture);
    }
    {
        Fixture fixture;
        test_failures(fixture);
    }
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("shader.package checks passed\n");
    return 0;
}
