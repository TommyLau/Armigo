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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

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

void write_bytes(const fs::path& path, std::string_view data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(data.data(), static_cast<std::streamsize>(data.size()));
    CHECK(stream.good());
}

void write_bytes(const fs::path& path, const std::vector<std::byte>& data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    CHECK(stream.good());
}

std::string nested_kernel() {
    constexpr int depth = 300;
    std::string source = "#include <metal_stdlib>\nusing namespace metal;\n"
                         "kernel void armigo_compute_main(device uint* out [[buffer(0)]]) {\n    uint value = ";
    source.append(static_cast<std::size_t>(depth), '(');
    source += "0u";
    source.append(static_cast<std::size_t>(depth), ')');
    source += ";\n    out[0] = value;\n}\n";
    return source;
}

int run(const std::vector<std::string>& arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    pid_t pid = 0;
    if (posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ) != 0) return -1;
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}

std::string capture(const std::vector<std::string>& arguments) {
    int pipe_fd[2] = {-1, -1};
    if (pipe(pipe_fd) != 0) return {};
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return {};
    posix_spawn_file_actions_addclose(&actions, pipe_fd[0]);
    posix_spawn_file_actions_adddup2(&actions, pipe_fd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_fd[1]);
    std::vector<char*> argv;
    for (const std::string& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    pid_t pid = 0;
    const int spawned = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipe_fd[1]);
    std::string output;
    if (spawned == 0) {
        std::array<char, 512> buffer{};
        for (;;) {
            const auto count = ::read(pipe_fd[0], buffer.data(), buffer.size());
            if (count <= 0) break;
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        int status = 0;
        waitpid(pid, &status, 0);
    }
    close(pipe_fd[0]);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    return output;
}

} // namespace

int main() {
    ::unsetenv("ARMIGO_SHADER_COMPILE_CACHE");
    const fs::path root = fs::temp_directory_path() / ("armigo-bracket-depth-" + std::to_string(::getpid()));
    fs::remove_all(root);
    const fs::path stage = root / "stage";
    fs::create_directories(stage);
    const fs::path translate = root / "translate";
    write_bytes(translate, "#!/bin/sh\n[ \"$1\" = --version ] && printf '" + std::string(64, 'a') + "\\n'\n");
    ::chmod(translate.c_str(), 0755);

    std::vector<std::byte> program(160);
    const std::uint32_t sph = 1u << 10;
    std::memcpy(program.data(), &sph, sizeof sph);
    const std::string source = nested_kernel();
    const std::string metadata =
        "{\"schema\":\"armigo-shader-stage-v1\",\"source_version\":\"" + std::string(64, 'a') +
        "\",\"program_key\":\"" + digest(std::span<const std::byte>(program)) + "\",\"program_size\":160,"
        "\"program_file\":\"program.bin\",\"source_file\":\"source.metal\",\"source_digest\":\"" +
        digest(source) + "\",\"stage\":\"vertex\",\"entry_point\":\"armigo_vertex_main\","
        "\"vertex_depth\":\"minus-one-to-one\",\"fragment_position\":\"unspecified\","
        "\"vertex_attributes\":[{\"generic_slot\":0,\"attribute_index\":2,\"format\":\"float32x3\"}],"
        "\"constant_buffers\":[],\"textures\":[],\"interface\":{\"position_input_mask\":0,"
        "\"position_output_mask\":15,\"color_output_mask\":0,\"generic_inputs\":[{\"slot\":0,"
        "\"component_mask\":7,\"interpolation\":\"screen-linear\"}],\"generic_outputs\":[{\"slot\":0,"
        "\"component_mask\":3,\"interpolation\":\"screen-linear\"}]}}\n";
    write_bytes(stage / "program.bin", program);
    write_bytes(stage / "source.metal", source);
    write_bytes(stage / "reflection.json", metadata);

    package::BuildOptions options;
    options.translate_tool = translate;
    options.output = root / "package";
    options.jobs = 1;
    options.stages = {stage};
    const auto built = package::build(options);
    if (const auto* error = std::get_if<std::string>(&built)) std::printf("build: %s\n", error->c_str());
    CHECK(std::holds_alternative<package::BuildReport>(built));

    const std::string metal = capture({"xcrun", "--sdk", "macosx", "--find", "metal"});
    CHECK(!metal.empty());
    const fs::path air = root / "without.air";
    const int rejected = run({metal, "-std=metal4.0", "-mmacosx-version-min=26.0", "-Werror",
                              "-Wno-unused-variable", "-Wno-sometimes-uninitialized", "-c",
                              (stage / "source.metal").string(), "-o", air.string()});
    CHECK(rejected != 0);

    fs::remove_all(root);
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("shader.package bracket depth passed\n");
    return 0;
}
