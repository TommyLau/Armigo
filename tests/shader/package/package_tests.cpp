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

// A v2 fragment whose texture, sampler and descriptor fact all name slot 31.
// `programs` is the handle_programs JSON value, or null when the field is absent.
fs::path write_handle_fragment(Fixture& fixture, std::string_view name, const char* programs,
                               std::string_view descriptor_offset) {
    const fs::path directory = fixture.inputs / name;
    fs::create_directory(directory);
    std::vector<std::byte> program(160);
    for (std::size_t index = 0; index < program.size(); ++index)
        program[index] = static_cast<std::byte>((index * 31 + 11) & 255);
    const std::uint32_t sph = 5u << 10;
    std::memcpy(program.data(), &sph, sizeof sph);
    const std::string source = "#include <metal_stdlib>\n// " + std::string(name) + "\n";
    const std::string version(64, 'a');
    const std::string programs_field = programs ? ",\"handle_programs\":" + std::string(programs) : "";
    const std::string metadata =
        "{\"schema\":\"armigo-shader-stage-v2\",\"source_version\":\"" + version +
        "\",\"program_key\":\"" + digest(std::span<const std::byte>(program)) + "\",\"program_size\":160,"
        "\"program_file\":\"program.bin\",\"source_file\":\"source.metal\",\"source_digest\":\"" +
        digest(source) + "\",\"stage\":\"fragment\",\"entry_point\":\"armigo_fragment_main\","
        "\"vertex_depth\":\"unspecified\",\"fragment_position\":\"unspecified\",\"c1_digest\":null,"
        "\"profile\":{\"identity\":\"armigo-macos-metal4\",\"version\":1},"
        "\"constant_buffers\":[],\"support_buffers\":[],\"textures\":["
        "{\"cbuf_slot\":31,\"handle_byte_offset\":0,\"kind\":\"texture\",\"dimension\":\"d2\","
        "\"array\":false,\"multisample\":false,\"scalar\":\"f32\",\"component_format\":\"unorm\","
        "\"normalized_coordinates\":true,\"comparison\":false,\"texture_index\":0},"
        "{\"cbuf_slot\":31,\"handle_byte_offset\":0,\"kind\":\"sampler\",\"comparison\":false,"
        "\"normalized_coordinates\":true,\"sampler_index\":0}],"
        "\"vertex_streams\":[],\"vertex_inputs\":[],"
        "\"interface\":{\"position_input_mask\":0,\"position_output_mask\":0,\"color_output_mask\":15,"
        "\"color_outputs\":[{\"location\":0,\"component_mask\":15,\"scalar\":\"f32\",\"components\":4}],"
        "\"writes_fragment_depth\":false,\"writes_sample_mask\":false,\"generic_inputs\":[],"
        "\"generic_outputs\":[]},\"dependencies\":[{\"kind\":\"fragment_output_type\",\"index\":0,"
        "\"byte_offset\":0,\"component\":0,\"origin\":\"derived\",\"value\":{\"scalar\":\"f32\","
        "\"components\":4}},{\"kind\":\"texture_descriptor\",\"index\":31,\"byte_offset\":" +
        std::string(descriptor_offset) + ",\"component\":0,\"origin\":\"derived\",\"value\":{"
        "\"dimension\":\"d2\",\"array\":false,\"multisample\":false,\"scalar\":\"f32\","
        "\"component_format\":\"unorm\",\"normalized_coordinates\":true,\"comparison\":false}}]" +
        programs_field + "}\n";
    write(directory / "program.bin", program);
    write(directory / "source.metal", source);
    write(directory / "reflection.json", metadata);
    return directory;
}

void expect_package_refusal(Fixture& fixture, const fs::path& stage, std::string_view output,
                            std::string_view message) {
    auto options = fixture.options(std::string(output));
    options.stages = {stage};
    const auto built = package::build(options);
    const auto* error = std::get_if<std::string>(&built);
    if (!error || error->find(message) == std::string::npos) {
        std::printf("FAIL package %s: expected \"%s\" in \"%s\"\n", std::string(output).c_str(),
                    std::string(message).c_str(), error ? error->c_str() : "<build succeeded>");
        ++failures;
    }
    CHECK(!fs::exists(options.output));
}

void test_handle_program_packages(Fixture& fixture) {
    // A v1 reflection has no handle_programs field. Its variant id stays the
    // one pinned before that field existed.
    const fs::path vertex = fixture.stage("vertex");
    CHECK(read(vertex / "reflection.json").find("handle_programs") == std::string::npos);
    auto identity = fixture.options("handle-identity");
    identity.stages = {vertex};
    const auto identity_built = package::build(identity);
    if (const auto* error = std::get_if<std::string>(&identity_built))
        std::printf("identity build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::BuildReport>(identity_built));
    const std::string variant = "1cd06a449ca44560d3f15f646a5aeb3a8c3a2d40637f57e2016291bb2043e57c";
    CHECK(read(identity.output / "manifest.json").find("\"variant_id\":\"" + variant + "\"") != std::string::npos);

    const fs::path valid = write_handle_fragment(fixture, "handle-valid", "[[4294967295,0]]", "0");
    auto accepted = fixture.options("handle-valid-out");
    accepted.stages = {valid};
    const auto built = package::build(accepted);
    if (const auto* error = std::get_if<std::string>(&built))
        std::printf("handle program build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::BuildReport>(built));
    if (std::holds_alternative<package::BuildReport>(built)) {
        std::string packaged;
        for (const auto& entry : fs::recursive_directory_iterator(accepted.output)) {
            if (entry.path().filename() == "reflection.json") packaged = read(entry.path());
        }
        const std::string reflection = read(valid / "reflection.json");
        CHECK(packaged == reflection);
        CHECK(packaged.find("\"cbuf_slot\":31") != std::string::npos);
        CHECK(packaged.find("\"handle_programs\":[[4294967295,0]]") != std::string::npos);
        CHECK(packaged.find("\"kind\":\"texture_descriptor\"") != std::string::npos);
    }

    expect_package_refusal(fixture, write_handle_fragment(fixture, "handle-missing", nullptr, "0"),
                           "handle-missing-out", "texture provenance reflection is invalid");
    expect_package_refusal(fixture, write_handle_fragment(fixture, "handle-empty", "[]", "0"),
                           "handle-empty-out", "texture provenance reflection is invalid");
    expect_package_refusal(fixture, write_handle_fragment(fixture, "handle-range", "[[4294967295,0]]", "4"),
                           "handle-range-out", "texture descriptor dependency key is invalid");
    expect_package_refusal(fixture, write_handle_fragment(fixture, "handle-object", "{}", "0"),
                           "handle-object-out", "handle_programs must be an array");
    expect_package_refusal(fixture, write_handle_fragment(fixture, "handle-element", "[1]", "0"),
                           "handle-element-out", "handle program must be an array");
    expect_package_refusal(fixture, write_handle_fragment(fixture, "handle-word", "[[-1]]", "0"),
                           "handle-word-out", "handle program word is invalid");
}

std::string profile_text_of(std::string_view manifest) {
    constexpr std::string_view marker = "\"profile_text\":\"";
    const auto start = manifest.find(marker);
    CHECK(start != std::string_view::npos);
    std::string result;
    if (start == std::string_view::npos) return result;
    for (std::size_t index = start + marker.size(); index < manifest.size(); ++index) {
        const char character = manifest[index];
        if (character == '\\') {
            if (++index >= manifest.size()) break;
            result += manifest[index];
            continue;
        }
        if (character == '"') break;
        result += character;
    }
    return result;
}

std::size_t profile_key_count(std::string_view profile) {
    std::size_t count = 0;
    for (std::size_t index = 0; index + 1 < profile.size(); ++index) {
        if (profile[index] == '"' && profile[index + 1] == ':') ++count;
    }
    return count;
}

// The profile keys every package records. Sample hoisting is manifest metadata
// and must not change this set or the chain id.
constexpr std::string_view baseline_profile_keys[] = {
    "compile_options",
    "deployment_target",
    "identity",
    "metal_binary_sha256",
    "metal_language",
    "metal_version",
    "metallib_binary_sha256",
    "platform",
    "sdk_build",
    "sdk_version",
    "source_version",
    "translation_profile",
};

void test_sample_hoisting_manifest(Fixture& fixture) {
    const fs::path stage = fixture.stage("hoist-vertex");
    auto plain = fixture.options("hoist-off");
    plain.stages = {stage};
    plain.hoist_implicit_samples = false;
    const auto plain_built = package::build(plain);
    if (const auto* error = std::get_if<std::string>(&plain_built))
        std::printf("hoist-off build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::BuildReport>(plain_built));
    const std::string plain_manifest = read(plain.output / "manifest.json");
    const std::string plain_profile = profile_text_of(plain_manifest);
    CHECK(plain_manifest.find("sample_hoisting") == std::string::npos);
    CHECK(plain_profile.find("sample_hoisting") == std::string::npos);
    CHECK(profile_key_count(plain_profile) == std::size(baseline_profile_keys));
    for (const std::string_view key : baseline_profile_keys) {
        CHECK(plain_profile.find(std::string("\"") + std::string(key) + "\":") != std::string::npos);
    }

    auto hoisted = fixture.options("hoist-on");
    hoisted.stages = {stage};
    hoisted.hoist_implicit_samples = true;
    const auto hoisted_built = package::build(hoisted);
    if (const auto* error = std::get_if<std::string>(&hoisted_built))
        std::printf("hoist-on build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::BuildReport>(hoisted_built));
    const std::string hoisted_manifest = read(hoisted.output / "manifest.json");
    const std::string hoisted_profile = profile_text_of(hoisted_manifest);
    if (hoisted_profile != plain_profile)
        std::printf("hoisted profile: %s\n", hoisted_profile.c_str());
    CHECK(hoisted_profile == plain_profile);
    CHECK(profile_key_count(hoisted_profile) == std::size(baseline_profile_keys));
    CHECK(hoisted_manifest.find("\"sample_hoisting\":\"implicit\"") != std::string::npos);
    constexpr std::string_view hoisting_needle = "sample_hoisting";
    std::size_t occurrences = 0;
    for (std::size_t at = 0; (at = hoisted_manifest.find(hoisting_needle, at)) != std::string::npos;
         at += hoisting_needle.size())
        ++occurrences;
    CHECK(occurrences == 1);
    const auto chain_of = [](const std::string& manifest) {
        constexpr std::string_view marker = "\"chain_id\":\"";
        const auto start = manifest.find(marker);
        return start == std::string::npos ? std::string() : manifest.substr(start + marker.size(), 64);
    };
    // The field is outside the profile, so both packages keep the chain the plain profile always had.
    CHECK(chain_of(plain_manifest) == "239cb9f212404965aa640e5ab22406012d8777a211ccada708cac17be5afabcf");
    CHECK(chain_of(hoisted_manifest) == "239cb9f212404965aa640e5ab22406012d8777a211ccada708cac17be5afabcf");
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
    {
        Fixture fixture;
        test_handle_program_packages(fixture);
    }
    {
        Fixture fixture;
        test_sample_hoisting_manifest(fixture);
    }
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("shader.package checks passed\n");
    return 0;
}
