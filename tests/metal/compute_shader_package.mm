// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic compute-stage package: opening, compute
// lookup by (program, local size, shared memory), and the structured miss
// that names the program and both sides' sizes. Fixture assembly follows the
// synthetic package idioms of tests/metal/shader_package.mm.

#import <Foundation/Foundation.h>

#include <armigo/platform/metal_shader_package.h>
#include "compute_qmd.h"
#include <armigo/shader/program_identity.h>
#include <armigo/shader/source_version.h>
#include "sha256.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using namespace armigo;
using platform::CompiledComputeStageResult;
using platform::MetalShaderPackage;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

std::string hash(std::span<const std::byte> bytes) {
    const auto digest = loader::detail::sha256(bytes);
    CHECK(digest.has_value());
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (auto byte : *digest) {
        const auto value = std::to_integer<unsigned>(byte);
        result += digits[value >> 4];
        result += digits[value & 15];
    }
    return result;
}

std::string hash(std::string_view text) {
    return hash(std::as_bytes(std::span(text.data(), text.size())));
}

void write(const fs::path& path, std::string_view text) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    CHECK(bool(stream));
}

// A compute program is opaque bytes; identity is its digest alone (no SPH).
std::vector<std::byte> compute_program(std::byte fill) {
    return std::vector<std::byte>(
        shader::program_identity::prefix_size, fill);
}

std::string compute_reflection(std::span<const std::byte> bytes, const std::string& local_size,
                               const std::string& shared_memory) {
    return std::string("{\"schema\":\"armigo-shader-stage-compute-v1\",\"source_version\":\"") +
           shader::source_version + "\",\"program_key\":\"" + hash(bytes) +
           "\",\"program_size\":" + std::to_string(bytes.size()) +
           ",\"program_file\":\"program.bin\",\"source_file\":\"source.metal\","
           "\"source_digest\":\"" + std::string(64, '0') + "\",\"stage\":\"compute\","
           "\"entry_point\":\"armigo_compute_main\",\"local_size\":" + local_size +
           ",\"shared_memory_bytes\":" + shared_memory +
           ",\"profile\":{\"identity\":\"armigo-macos-metal4\",\"version\":3},"
           "\"storage_buffers\":[{\"cbuf_slot\":1,\"descriptor_byte_offset\":0,"
           "\"metal_buffer_index\":8,\"access\":\"read_write\",\"alignment\":4}],"
           "\"textures\":[{\"cbuf_slot\":2,\"handle_byte_offset\":0,\"kind\":\"texture\","
           "\"dimension\":\"d2\",\"array\":false,\"multisample\":false,\"scalar\":\"f32\","
           "\"component_format\":\"floating\",\"normalized_coordinates\":true,"
           "\"comparison\":false,\"texture_index\":0},{\"cbuf_slot\":2,"
           "\"handle_byte_offset\":0,\"kind\":\"sampler\",\"comparison\":false,"
           "\"normalized_coordinates\":true,\"sampler_index\":0}]}\n";
}

struct Fixture {
    fs::path root;
    std::string chain;
    std::vector<std::byte> program;
    std::string variant;
};

Fixture package(const fs::path& root, const std::string& local_size,
                const std::string& shared_memory) {
    Fixture result{root, {}, compute_program(std::byte{0x5c}), hash("compute variant")};
    const std::string profile = std::string("{\"platform\":\"macos-metal4\",\"source_version\":\"") +
                                shader::source_version + "\",\"tool_version\":\"sdk/metal\"}";
    result.chain = hash("armigo-metal-chain-v1\n" + profile);
    fs::create_directories(root / "stages" / result.variant);
    const auto reflected = compute_reflection(result.program, local_size, shared_memory);
    write(root / "stages" / result.variant / "program.bin", "");
    {
        std::ofstream stream(root / "stages" / result.variant / "program.bin", std::ios::binary);
        stream.write(reinterpret_cast<const char*>(result.program.data()),
                     static_cast<std::streamsize>(result.program.size()));
    }
    write(root / "stages" / result.variant / "reflection.json", reflected);
    write(root / "stages" / result.variant / "shader.metallib", "synthetic compute metallib");
    const std::string entry = std::string("{\"stage\":\"compute\",\"program_key\":\"") +
        hash(result.program) + "\",\"prefix_key\":\"" + hash(result.program) +
        "\",\"program_size\":" + std::to_string(result.program.size()) +
        ",\"directory\":\"stages/" + result.variant + "\",\"reflection_digest\":\"" +
        hash(reflected) + "\",\"library_digest\":\"" + hash(std::string("synthetic compute metallib")) +
        "\",\"library_size\":26,\"variant_id\":\"" + result.variant + "\"}";
    const std::string manifest = std::string("{\"schema\":\"armigo-metal-package-v1\","
        "\"source_version\":\"") + shader::source_version + "\",\"chain_id\":\"" +
        result.chain + "\",\"platform\":\"macos-metal4\",\"profile_text\":";
    NSString* profile_string = [NSString stringWithUTF8String:profile.c_str()];
    NSData* escaped_data =
        [NSJSONSerialization dataWithJSONObject:@[profile_string] options:0 error:nil];
    std::string escaped(static_cast<const char*>(escaped_data.bytes), escaped_data.length);
    escaped = escaped.substr(1, escaped.size() - 2);
    write(root / "manifest.json", manifest + escaped +
        ",\"complete_inventory\":false,\"runnable\":false,\"entries\":[" + entry + "]}\n");
    return result;
}

struct Backing {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x8000);
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> output) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < 0x1000 || address - 0x1000 > self.bytes.size() ||
            output.size() > self.bytes.size() - (address - 0x1000)) {
            return false;
        }
        std::memcpy(output.data(), self.bytes.data() + (address - 0x1000), output.size());
        return true;
    }
    static bool write(void*, std::uint64_t, std::span<const std::byte>) noexcept { return true; }
};

graphics::GpuAddressSpace memory(Backing& backing) {
    const graphics::Mapping mapping{0x1000, 0x8000, 0, 0x1000, 0x8000, 0x1000};
    return std::get<graphics::GpuAddressSpace>(graphics::make_address_space(
        {&mapping, 1}, {Backing::read, Backing::write, &backing}));
}

// Publishes the synthetic queue meta data at guest 0x2000 and returns the
// decoded launch inside a compute RenderStop bound to the program at 0x1000.
graphics::RenderStop compute_stop(const Fixture& fixture, Backing& backing,
                                  std::array<std::uint32_t, 3> grid,
                                  std::array<std::uint32_t, 3> block,
                                  std::uint32_t shared_memory_words) {
    std::memcpy(backing.bytes.data(), fixture.program.data(), fixture.program.size());
    std::array<std::uint32_t, 32> words{};
    words[12] = grid[0];
    words[13] = grid[1] | (grid[2] << 16);
    words[17] = shared_memory_words;
    words[18] = block[0] << 16;
    words[19] = block[1] | (block[2] << 16);
    words[20] = 1;  // constant buffer 0 valid (unused by lookup)
    std::memcpy(backing.bytes.data() + 0x1000, words.data(), sizeof(words));
    graphics::RenderStop stop;
    stop.kind = graphics::RenderKind::Compute;
    stop.engine = graphics::engine_class::compute;
    stop.compute_descriptor_address = 0x2000;
    std::memcpy(stop.compute_descriptor.data(), words.data(), sizeof(words));
    stop.has_compute_descriptor = true;
    stop.compute = armigo::graphics::decode_compute_qmd(stop.compute_descriptor);
    stop.programs[0] = {0x1000, 0, 0, true};
    return stop;
}

const platform::CompiledGraphicsStage* stage_of(const CompiledComputeStageResult& result) {
    const auto* pair_failure = std::get_if<graphics::CommandResult>(&result);
    if (pair_failure) return nullptr;
    return std::get<std::shared_ptr<const platform::CompiledGraphicsStage>>(result).get();
}

}  // namespace

int main() {
    @autoreleasepool {
    const fs::path base = fs::temp_directory_path() / "armigo-compute-package-test";
    fs::remove_all(base);

    // Without development-subset consent the incomplete package refuses.
    {
        auto fixture = package(base / "refusal", "[4,1,1]", "1024");
        const auto refused = platform::MetalShaderPackage::open(
            fixture.root, {fixture.chain, false});
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(refused));
    }

    auto fixture = package(base / "good", "[4,1,1]", "1024");
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    if (auto* error = std::get_if<platform::MetalShaderPackageError>(&opened)) {
        std::fprintf(stderr, "package open failed: %s\n", error->detail.c_str());
        ++failures;
        return 1;
    }
    auto& loaded = std::get<std::shared_ptr<MetalShaderPackage>>(opened);

    Backing backing;
    auto space = memory(backing);

    // Matching launch shape resolves the compute stage with its bindings.
    {
        const auto stop = compute_stop(fixture, backing, {2, 3, 1}, {4, 1, 1}, 1024);
        const auto result = loaded->lookup_compute(stop, space);
        const auto* stage = stage_of(result);
        CHECK(stage != nullptr);
        if (stage) {
            CHECK(stage->stage == armigo::shader_input::Stage::Compute);
            CHECK(stage->entry_point == "armigo_compute_main");
            CHECK((stage->compute_local_size == std::array<std::uint32_t, 3>{4, 1, 1}));
            CHECK(stage->compute_shared_memory_bytes == 1024);
            CHECK(stage->storage_buffers.size() == 1);
            CHECK(stage->storage_buffers[0].cbuf_slot == 1);
            CHECK(stage->storage_buffers[0].metal_buffer_index == 8);
            CHECK(stage->storage_buffers[0].access ==
                  platform::CompiledShaderResourceAccess::ReadWrite);
            CHECK(stage->textures.size() == 1);
            CHECK(stage->textures[0].texture_index == 0);
            CHECK(stage->textures[0].sampler_index == 0);
            CHECK(stage->texture_facts.size() == 1);
            const std::string library = "synthetic compute metallib";
            CHECK(stage->metallib ==
                  std::vector<std::byte>(std::as_bytes(std::span(library)).begin(),
                                         std::as_bytes(std::span(library)).end()));
        }
    }

    // A mismatching local size or shared memory is a structured miss naming
    // the program and both sides' sizes.
    {
        const auto stop = compute_stop(fixture, backing, {2, 3, 1}, {8, 8, 8}, 1024);
        const auto result = loaded->lookup_compute(stop, space);
        const auto* failure = std::get_if<graphics::CommandResult>(&result);
        CHECK(failure != nullptr);
        CHECK(failure && failure->status == graphics::CommandStatus::Unsupported);
        CHECK(failure && failure->detail.find("8x8x8") != std::string_view::npos);
        CHECK(failure && failure->detail.find("shared memory 1024") != std::string_view::npos);
    }
    // A larger shared allocation than the variant declares still uses it (the
    // launch allocates in hardware granules); a smaller one is a structured miss.
    {
        const auto stop = compute_stop(fixture, backing, {2, 3, 1}, {4, 1, 1}, 2048);
        const auto result = loaded->lookup_compute(stop, space);
        CHECK(std::get_if<graphics::CommandResult>(&result) == nullptr);
    }
    {
        const auto stop = compute_stop(fixture, backing, {2, 3, 1}, {4, 1, 1}, 512);
        const auto result = loaded->lookup_compute(stop, space);
        const auto* failure = std::get_if<graphics::CommandResult>(&result);
        CHECK(failure != nullptr);
        CHECK(failure && failure->status == graphics::CommandStatus::Unsupported);
        CHECK(failure && failure->detail.find("shared memory 512") != std::string_view::npos);
    }

    // A program absent from guest memory is a structured lookup failure.
    {
        auto stop = compute_stop(fixture, backing, {2, 3, 1}, {4, 1, 1}, 1024);
        stop.programs[0].address = 0x7000;  // never written
        const auto result = loaded->lookup_compute(stop, space);
        const auto* failure = std::get_if<graphics::CommandResult>(&result);
        CHECK(failure != nullptr);
        CHECK(failure && failure->status == graphics::CommandStatus::Unsupported);
        CHECK(failure && failure->detail.find("compute lookup failed") != std::string_view::npos);
    }

    // A non-compute stop is refused explicitly.
    {
        auto stop = compute_stop(fixture, backing, {2, 3, 1}, {4, 1, 1}, 1024);
        stop.kind = graphics::RenderKind::Draw;
        const auto result = loaded->lookup_compute(stop, space);
        const auto* failure = std::get_if<graphics::CommandResult>(&result);
        CHECK(failure != nullptr);
    }

    fs::remove_all(base);
    if (failures == 0) std::printf("compute shader package tests passed\n");
    else std::fprintf(stderr, "%u compute package checks failed\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
