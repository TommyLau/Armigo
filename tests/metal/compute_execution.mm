// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Real-GPU compute dispatch execution: two independently authored synthetic
// kernels - one writing a storage buffer from a constant buffer and its
// global ID, one sampling a 2D texture with an explicit LOD - driven through
// execute_metal_compute with synthetic queue meta data. Fixture byte layouts
// for TIC/TSC descriptors follow the published word positions used by
// tests/graphics/texture_tests.cpp.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_dispatch.h"
#include "compute_qmd.h"
#include <sys/mman.h>
#include <armigo/platform/metal_engine_renderer.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using namespace armigo;
using namespace armigo::graphics;
using platform::CompiledComputeStageResult;
using platform::MetalComputeContext;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

constexpr std::uint64_t guest_base = 0x10000;
constexpr std::size_t guest_size = 0x10000;

struct Backing {
    std::array<std::byte, guest_size> bytes{};
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> output) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < guest_base || address - guest_base > self.bytes.size() ||
            output.size() > self.bytes.size() - (address - guest_base)) {
            return false;
        }
        std::memcpy(output.data(), self.bytes.data() + (address - guest_base), output.size());
        return true;
    }
    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> input) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < guest_base || address - guest_base > self.bytes.size() ||
            input.size() > self.bytes.size() - (address - guest_base)) {
            return false;
        }
        std::memcpy(self.bytes.data() + (address - guest_base), input.data(), input.size());
        return true;
    }

    void put32(std::uint64_t address, std::uint32_t value) {
        std::memcpy(bytes.data() + (address - guest_base), &value, sizeof(value));
    }
};

GpuAddressSpace make_memory(Backing& backing) {
    const Mapping mapping{guest_base, guest_size, 0, guest_base, guest_size, 0x1000};
    return std::get<GpuAddressSpace>(make_address_space(
        {&mapping, 1}, {Backing::read, Backing::write, &backing}));
}

void put32(std::span<std::byte> bytes, std::size_t byte, std::uint32_t value) {
    std::memcpy(bytes.data() + byte, &value, sizeof(value));
}

// Builds the 256-byte queue meta data for the launch: grid, block, shared
// memory words, and up to two enabled constant buffers (slot 0 at guest
// `cbuf0`, slot 1 at `cbuf1`, 16 bytes each).
std::array<std::byte, 256> descriptor(std::array<std::uint32_t, 3> grid,
                                      std::array<std::uint32_t, 3> block,
                                      std::uint32_t shared_memory_bytes,
                                      std::uint64_t cbuf0 = 0, std::uint64_t cbuf1 = 0,
                                      std::uint32_t cbuf0_size = 16) {
    std::array<std::byte, 256> bytes{};
    auto* words = reinterpret_cast<std::uint32_t*>(bytes.data());
    words[12] = grid[0];
    words[13] = grid[1] | (grid[2] << 16);
    words[17] = shared_memory_bytes;
    words[18] = block[0] << 16;
    words[19] = block[1] | (block[2] << 16);
    words[20] = (cbuf0 ? 1u : 0u) | (cbuf1 ? (1u << 1) : 0u);
    if (cbuf0) {
        words[29] = static_cast<std::uint32_t>(cbuf0);
        words[30] = std::uint32_t(cbuf0 >> 32) | (cbuf0_size << 15);
    }
    if (cbuf1) {
        words[31] = static_cast<std::uint32_t>(cbuf1);
        words[32] = std::uint32_t(cbuf1 >> 32) | (16u << 15);
    }
    return bytes;
}

RenderStop make_stop(std::span<const std::byte, 256> qmd, std::uint32_t sampler_pool,
                     std::uint32_t texture_pool) {
    RenderStop stop;
    stop.kind = RenderKind::Compute;
    stop.engine = engine_class::compute;
    stop.has_compute_descriptor = true;
    stop.compute_descriptor_address = guest_base + 0x8000;
    std::copy(qmd.begin(), qmd.end(), stop.compute_descriptor.begin());
    stop.compute = decode_compute_qmd(stop.compute_descriptor);
    stop.programs[0] = {guest_base + 0x10, 0, 0, true};
    stop.registers[0x557] = sampler_pool;  // sampler (TSC) pool address
    stop.registers[0x559] = 0;             // sampler pool maximum index
    stop.registers[0x55d] = texture_pool;  // texture (TIC) pool address
    stop.registers[0x55f] = 0;             // texture pool maximum index
    return stop;
}

// Compiles MSL into metallib bytes through the platform toolchain, exactly
// like the offline package builder does. Returns empty on any failure.
std::vector<std::byte> build_metallib(const fs::path& work, std::string_view source) {
    const auto source_path = work / "kernel.metal";
    const auto air_path = work / "kernel.air";
    const auto library_path = work / "kernel.metallib";
    {
        std::ofstream stream(source_path);
        stream.write(source.data(), static_cast<std::streamsize>(source.size()));
    }
    for (const std::string command :
         {"xcrun --sdk macosx metal -c '" + source_path.string() + "' -o '" +
              air_path.string() + "'",
          "xcrun --sdk macosx metallib '" + air_path.string() + "' -o '" +
              library_path.string() + "'"}) {
        if (std::system(command.c_str()) != 0) {
            std::fprintf(stderr, "metallib build failed: %s\n", command.c_str());
            return {};
        }
    }
    std::ifstream stream(library_path, std::ios::binary);
    const std::string contents{std::istreambuf_iterator<char>(stream),
                               std::istreambuf_iterator<char>()};
    std::vector<std::byte> result;
    result.insert(result.end(), std::as_bytes(std::span(contents.data(), contents.size())).begin(),
                  std::as_bytes(std::span(contents.data(), contents.size())).end());
    return result;
}

struct StageHolder {
    std::shared_ptr<const platform::CompiledGraphicsStage> stage;
    static CompiledComputeStageResult lookup(void* raw, const RenderStop&,
                                             const GpuAddressSpace&) noexcept {
        auto& self = *static_cast<StageHolder*>(raw);
        return self.stage;
    }
};

platform::CompiledGraphicsStage storage_stage(std::vector<std::byte> metallib) {
    platform::CompiledGraphicsStage stage;
    stage.stage = armigo::shader_input::Stage::Compute;
    stage.entry_point = "armigo_compute_main";
    stage.metallib = std::move(metallib);
    stage.compute_local_size = {4, 1, 1};
    stage.compute_shared_memory_bytes = 0;
    stage.storage_buffers.push_back({1, 0, 8,
                                     platform::CompiledShaderResourceAccess::ReadWrite, 4});
    return stage;
}

platform::CompiledGraphicsStage texture_stage(std::vector<std::byte> metallib) {
    platform::CompiledGraphicsStage stage;
    stage.stage = armigo::shader_input::Stage::Compute;
    stage.entry_point = "armigo_compute_main";
    stage.metallib = std::move(metallib);
    stage.compute_local_size = {4, 1, 1};
    stage.compute_shared_memory_bytes = 0;
    // The output buffer descriptor rides in constant buffer 0 at offset 8;
    // the texture handle word is constant buffer 0 at offset 0.
    stage.storage_buffers.push_back({0, 8, 8,
                                     platform::CompiledShaderResourceAccess::Write, 4});
    platform::DerivedTextureFact fact{};
    fact.cbuf_slot = 0;
    fact.dimension = platform::TextureFactDimension::d2;
    stage.texture_facts.push_back(fact);
    stage.textures.push_back({0, 0, 0});
    return stage;
}

// Reflection binds constant buffers through bounded windows with
// an armigo-cb-sizes support buffer, as the translator emits for compute.
platform::CompiledGraphicsStage windowed_storage_stage(std::vector<std::byte> metallib) {
    auto stage = storage_stage(std::move(metallib));
    stage.constant_buffer_windows = {{0, 16, 0}, {1, 16, 1}};
    stage.constant_buffer_support_index = 14;
    return stage;
}

// The persistent engine encodes the launch into its batch like a draw and
// writes the storage result back at completion (or, with deferred writeback,
// at the next coherence point), through its residency registry.
void test_persistent_dispatch(const std::vector<std::byte>& metallib, bool deferred) {
    Backing backing;
    std::shared_ptr<void> owner = std::make_shared<int>(0);
    const Mapping mapping{guest_base, guest_size, 0, guest_base, guest_size, 0x1000, owner};
    auto space = std::get<GpuAddressSpace>(make_address_space(
        {&mapping, 1}, {Backing::read, Backing::write, &backing}));
    backing.put32(0x10100, 11);
    backing.put32(0x10104, 22);
    backing.put32(0x10108, 900);
    backing.put32(0x10140, 0x10200);
    backing.put32(0x10144, 0);
    backing.put32(0x10148, 16);
    const auto qmd = descriptor({4, 1, 1}, {4, 1, 1}, 0, 0x10100, 0x10140);
    const auto stop = make_stop(qmd, 0, 0);

    auto holder = std::make_shared<StageHolder>();
    holder->stage = std::make_shared<const platform::CompiledGraphicsStage>(
        windowed_storage_stage(metallib));
    platform::MetalEngineOptions options;
    options.mode = platform::MetalEngineMode::Persistent;
    options.compute_lookup = StageHolder::lookup;
    options.lookup_context = holder.get();
    options.lookup_owner = holder;
    options.defer_gpu_writeback = deferred;
    options.statistics = std::make_shared<platform::MetalEngineStatistics>();
    auto created = platform::create_metal_engine_backend(std::move(options));
    if (const auto* error = std::get_if<RendererError>(&created)) {
        const auto result = error->command();
        std::fprintf(stderr, "persistent engine: %.*s\n", int(result.detail.size()),
                     result.detail.data());
        ++failures;
        return;
    }
    auto backend = std::get<RenderCallbacks>(std::move(created));
    const auto result = backend.dispatch(backend.context, stop, space);
    if (!result.ok()) {
        std::fprintf(stderr, "persistent dispatch failed: %.*s\n",
                     static_cast<int>(result.detail.size()), result.detail.data());
        ++failures;
        return;
    }
    if (deferred) {
        CHECK(backend.coherence != nullptr);
        if (backend.coherence) CHECK(backend.coherence(backend.context, space, 0, 0).ok());
    }
    std::array<std::uint32_t, 4> output{};
    std::memcpy(output.data(), backing.bytes.data() + (0x10200 - guest_base), sizeof(output));
    for (unsigned index = 0; index < 4; ++index) CHECK(output[index] == 900u + index);
}

// Host-addressable guest memory (native execution): the storage binding is a
// zero-copy view of the backing, so the shader's results land in guest memory
// and the coherence boundary writes nothing back through the callbacks.
struct HostBacking {
    std::byte* host = nullptr;
    unsigned writes = 0;
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> output) noexcept {
        auto& self = *static_cast<HostBacking*>(raw);
        const auto base = reinterpret_cast<std::uint64_t>(self.host);
        if (address < base || address - base > guest_size || output.size() > guest_size - (address - base))
            return false;
        std::memcpy(output.data(), self.host + (address - base), output.size());
        return true;
    }
    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> input) noexcept {
        auto& self = *static_cast<HostBacking*>(raw);
        const auto base = reinterpret_cast<std::uint64_t>(self.host);
        if (address < base || address - base > guest_size || input.size() > guest_size - (address - base))
            return false;
        ++self.writes;
        std::memcpy(self.host + (address - base), input.data(), input.size());
        return true;
    }
    void put32(std::uint64_t gpu_address, std::uint32_t value) {
        std::memcpy(host + (gpu_address - guest_base), &value, sizeof(value));
    }
};

void test_persistent_dispatch_in_place(const std::vector<std::byte>& metallib) {
    void* mapped = ::mmap(nullptr, guest_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(mapped != MAP_FAILED);
    if (mapped == MAP_FAILED) return;
    HostBacking backing{static_cast<std::byte*>(mapped)};
    std::shared_ptr<void> owner = std::make_shared<int>(0);
    const Mapping mapping{guest_base, guest_size, 0, reinterpret_cast<std::uint64_t>(mapped), guest_size,
                          0x1000, owner};
    auto space = std::get<GpuAddressSpace>(make_address_space(
        {&mapping, 1}, {HostBacking::read, HostBacking::write, &backing}));
    backing.put32(0x10100, 11);
    backing.put32(0x10104, 22);
    backing.put32(0x10108, 900);
    backing.put32(0x10140, 0x10200);
    backing.put32(0x10144, 0);
    backing.put32(0x10148, 16);
    const auto qmd = descriptor({4, 1, 1}, {4, 1, 1}, 0, 0x10100, 0x10140);
    const auto stop = make_stop(qmd, 0, 0);
    auto holder = std::make_shared<StageHolder>();
    holder->stage = std::make_shared<const platform::CompiledGraphicsStage>(
        windowed_storage_stage(metallib));
    platform::MetalEngineOptions options;
    options.mode = platform::MetalEngineMode::Persistent;
    options.compute_lookup = StageHolder::lookup;
    options.lookup_context = holder.get();
    options.lookup_owner = holder;
    options.defer_gpu_writeback = true;
    options.guest_memory_host_addressable = true;
    options.statistics = std::make_shared<platform::MetalEngineStatistics>();
    auto created = platform::create_metal_engine_backend(std::move(options));
    CHECK(std::holds_alternative<RenderCallbacks>(created));
    if (std::holds_alternative<RenderCallbacks>(created)) {
        auto backend = std::get<RenderCallbacks>(std::move(created));
        CHECK(backend.dispatch(backend.context, stop, space).ok());
        CHECK(backend.coherence(backend.context, space, 0, 0).ok());
        CHECK(backing.writes == 0);
        std::array<std::uint32_t, 4> output{};
        std::memcpy(output.data(), backing.host + (0x10200 - guest_base), sizeof(output));
        for (unsigned index = 0; index < 4; ++index) CHECK(output[index] == 900u + index);
    }
    ::munmap(mapped, guest_size);
}

// A persistent launch with an empty grid dimension completes as a no-op.
void test_persistent_empty_grid(const std::vector<std::byte>& metallib) {
    Backing backing;
    std::shared_ptr<void> owner = std::make_shared<int>(0);
    const Mapping mapping{guest_base, guest_size, 0, guest_base, guest_size, 0x1000, owner};
    auto space = std::get<GpuAddressSpace>(make_address_space(
        {&mapping, 1}, {Backing::read, Backing::write, &backing}));
    backing.put32(0x10100, 11);
    backing.put32(0x10104, 22);
    backing.put32(0x10108, 900);
    backing.put32(0x10140, 0x10200);
    backing.put32(0x10144, 0);
    backing.put32(0x10148, 16);
    const auto qmd = descriptor({0, 1, 1}, {4, 1, 1}, 0, 0x10100, 0x10140);
    const auto stop = make_stop(qmd, 0, 0);
    auto holder = std::make_shared<StageHolder>();
    holder->stage = std::make_shared<const platform::CompiledGraphicsStage>(
        windowed_storage_stage(metallib));
    platform::MetalEngineOptions options;
    options.mode = platform::MetalEngineMode::Persistent;
    options.compute_lookup = StageHolder::lookup;
    options.lookup_context = holder.get();
    options.lookup_owner = holder;
    options.defer_gpu_writeback = true;
    options.statistics = std::make_shared<platform::MetalEngineStatistics>();
    auto created = platform::create_metal_engine_backend(options);
    CHECK(std::holds_alternative<RenderCallbacks>(created));
    if (!std::holds_alternative<RenderCallbacks>(created)) return;
    auto backend = std::get<RenderCallbacks>(std::move(created));
    CHECK(backend.dispatch(backend.context, stop, space).ok());
    CHECK(backend.coherence(backend.context, space, 0, 0).ok());
    CHECK(options.statistics->completed_dispatches == 1);
    std::array<std::uint32_t, 4> output{};
    std::memcpy(output.data(), backing.bytes.data() + (0x10200 - guest_base), sizeof(output));
    for (const auto value : output) CHECK(value == 0u);
}

// TIC fixture: 32x8 RGBA8 pitch-linear at a guest address (format word 8 with
// component selectors, address words 1-2, layout word 2 bits 21..23 = 2, and
// the pitch in word 3).
std::array<std::byte, 32> tic(std::uint64_t address) {
    std::array<std::byte, 32> bytes{};
    std::uint32_t word0 = 8;
    for (unsigned component = 0; component < 4; ++component) {
        word0 |= 2u << (7 + component * 3);
        word0 |= (component + 2u) << (19 + component * 3);
    }
    put32(bytes, 0, word0);
    put32(bytes, 4, static_cast<std::uint32_t>(address));
    put32(bytes, 8, static_cast<std::uint32_t>(address >> 32) | (2u << 21));
    put32(bytes, 12, 128 >> 5);  // pitch layout stores row stride / 32
    put32(bytes, 16, (32 - 1) | (1u << 23) | (3u << 29));
    put32(bytes, 20, (8 - 1) | (1u << 31));
    put32(bytes, 24, 1u);
    return bytes;
}

// TSC fixture: nearest filtering with clamped LOD range covering zero.
std::array<std::byte, 32> tsc() {
    std::array<std::byte, 32> bytes{};
    std::uint32_t word0 = 2u | (2u << 3) | (2u << 6);
    word0 |= 1u << 15;
    put32(bytes, 0, word0);
    put32(bytes, 4, 2u | (2u << 4) | (3u << 6) | (1u << 8));
    put32(bytes, 8, 0 | (std::uint32_t{0x380} << 12));
    put32(bytes, 28, std::bit_cast<std::uint32_t>(1.0f));
    return bytes;
}

}  // namespace

int main() {
    @autoreleasepool {
    id<MTLDevice> probe = MTLCreateSystemDefaultDevice();
    if (!probe || ![probe supportsFamily:MTLGPUFamilyMetal4]) {
        std::fprintf(stderr, "No Metal 4 device; compute execution tests skipped.\n");
        return 1;
    }
    const fs::path work = fs::temp_directory_path() / "armigo-compute-execution-test";
    fs::create_directories(work);

    // Kernel one: storage output[i] = constant + global id.
    const auto storage_metallib = build_metallib(work, R"(
#include <metal_stdlib>
using namespace metal;
kernel void armigo_compute_main(constant uint* constants [[buffer(0)]],
                                device uint* output [[buffer(8)]],
                                uint3 id [[thread_position_in_grid]]) {
    output[id.x] = constants[2] + id.x;
}
)");
    // Kernel two: explicit-LOD sample of a 2D texture into a storage buffer.
    const auto texture_metallib = build_metallib(work, R"(
#include <metal_stdlib>
using namespace metal;
kernel void armigo_compute_main(texture2d<float, access::sample> source [[texture(0)]],
                                sampler states [[sampler(0)]],
                                device float* output [[buffer(8)]],
                                uint3 id [[thread_position_in_grid]]) {
    float4 pixel = source.sample(states, float2(0.5 / 32.0, 0.5 / 8.0), level(0));
    output[id.x] = pixel.r * 255.0 + pixel.g * 1000.0;
}
)");
    if (storage_metallib.empty() || texture_metallib.empty()) {
        fs::remove_all(work);
        return 1;
    }

    Backing backing;
    auto space = make_memory(backing);

    // Constant buffer 0 at 0x10100: three words, the third is the addend.
    backing.put32(0x10100, 11);
    backing.put32(0x10104, 22);
    backing.put32(0x10108, 700);
    // Constant buffer 1 at 0x10140: the storage descriptor for output at
    // 0x10200, 16 bytes.
    backing.put32(0x10140, 0x10200);
    backing.put32(0x10144, 0);
    backing.put32(0x10148, 16);

    const auto qmd = descriptor({4, 1, 1}, {4, 1, 1}, 0, 0x10100, 0x10140);
    const auto stop = make_stop(qmd, 0, 0);

    StageHolder holder;
    holder.stage = std::make_shared<const platform::CompiledGraphicsStage>(
        storage_stage(storage_metallib));
    MetalComputeContext context{StageHolder::lookup, &holder, nullptr,
                                30'000'000'000ull};
    const auto result = platform::execute_metal_compute(&context, stop, space);
    if (!result.ok()) {
        std::fprintf(stderr, "storage dispatch failed: %.*s\n",
                     static_cast<int>(result.detail.size()), result.detail.data());
        ++failures;
    } else {
        std::array<std::uint32_t, 4> output{};
        std::memcpy(output.data(), backing.bytes.data() + (0x10200 - guest_base),
                    sizeof(output));
        for (unsigned index = 0; index < 4; ++index) {
            CHECK(output[index] == 700u + index);
        }
    }

    // Texture path: TIC/TSC pools in guest memory, one entry each; the
    // texture backing holds a gradient whose first texel is (16, 32, 48, 255).
    backing.put32(0x10300, 0);  // handle word 0: texture 0, sampler 0
    const auto tic_entry = tic(0x10400);
    const auto tsc_entry = tsc();
    std::memcpy(backing.bytes.data() + (0x10500 - guest_base), tic_entry.data(), 32);
    std::memcpy(backing.bytes.data() + (0x10600 - guest_base), tsc_entry.data(), 32);
    std::memcpy(backing.bytes.data() + (0x10400 - guest_base),
                std::array<std::uint8_t, 4>{16, 32, 48, 255}.data(), 4);
    // Constant buffer 0: handle word at offset 0, output descriptor at 8.
    backing.put32(0x10100, 0);
    backing.put32(0x10108, 0x10200);
    backing.put32(0x1010c, 0);
    backing.put32(0x1010c + 4, 16);

    const auto texture_qmd = descriptor({4, 1, 1}, {4, 1, 1}, 0, 0x10100, 0, 32);
    const auto texture_stop = make_stop(texture_qmd, 0x10600, 0x10500);
    holder.stage = std::make_shared<const platform::CompiledGraphicsStage>(
        texture_stage(texture_metallib));
    const auto sampled = platform::execute_metal_compute(&context, texture_stop, space);
    if (!sampled.ok()) {
        std::fprintf(stderr, "texture dispatch failed: status %u detail '%.*s'\n",
                     static_cast<unsigned>(sampled.status),
                     static_cast<int>(sampled.detail.size()), sampled.detail.data());
        ++failures;
    } else {
        std::array<float, 4> output{};
        std::memcpy(output.data(), backing.bytes.data() + (0x10200 - guest_base),
                    sizeof(output));
        for (unsigned index = 0; index < 4; ++index) {
            const float expected = 16.0f / 255.0f * 255.0f + 32.0f / 255.0f * 1000.0f;
            if (std::abs(output[index] - expected) > 1e-3f) {
                std::fprintf(stderr, "FAIL texture output[%u]: %f expected %f\n", index,
                             output[index], expected);
                ++failures;
            }
        }
    }

    test_persistent_dispatch(storage_metallib, false);
    test_persistent_dispatch(storage_metallib, true);
    test_persistent_empty_grid(storage_metallib);
    test_persistent_dispatch_in_place(storage_metallib);

    // Invalid launches are structured rejections, never successes.
    {
        const auto broken = descriptor({0, 1, 1}, {4, 1, 1}, 0);
        const auto invalid = make_stop(broken, 0, 0);
        const auto refused = platform::validate_metal_compute_state(invalid);
        CHECK(!refused.ok());
    }

    fs::remove_all(work);
    if (failures == 0) std::printf("compute execution tests passed\n");
    else std::fprintf(stderr, "%u compute execution checks failed\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
