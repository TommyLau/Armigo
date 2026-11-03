// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Foundation/Foundation.h>

#include <armigo/platform/metal_engine_renderer.h>
#include "engine_renderer.h"
#include "native_pipeline_state.h"
#include <armigo/graphics/texture_layout.h>
#include <armigo/graphics/twod_blit.h>#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;
namespace metal_backend = armigo::platform::metal;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

// Guest backing of one 64 KiB mapping: GPU range [0x100000, 0x110000) aliased
// to the guest address 0x800000 through retained ownership.
struct Backing {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x10000, std::byte{0x19});
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < 0x800000 || address - 0x800000 > self.bytes.size() ||
            out.size() > self.bytes.size() - (address - 0x800000))
            return false;
        std::memcpy(out.data(), self.bytes.data() + (address - 0x800000), out.size());
        return true;
    }
    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < 0x800000 || address - 0x800000 > self.bytes.size() ||
            in.size() > self.bytes.size() - (address - 0x800000))
            return false;
        std::memcpy(self.bytes.data() + (address - 0x800000), in.data(), in.size());
        return true;
    }
    GpuAddressSpace memory() {
        Mapping mapping{0x100000, bytes.size(), 0, 0x800000, bytes.size(), 0x1000, owner};
        return std::get<GpuAddressSpace>(make_address_space({&mapping, 1}, {read, write, this}));
    }
    std::shared_ptr<void> owner;
};

constexpr std::uint64_t source_gpu_address = 0x100000;
constexpr std::uint64_t destination_gpu_address = 0x101000;

RenderStop blit_stop(std::uint32_t operation, std::uint32_t sample_mode) {
    RenderStop stop{};
    stop.kind = RenderKind::Blit;
    stop.engine = engine_class::twod;
    stop.original = {0x237, 5, 0, 0, true};
    auto& registers = stop.registers;
    // Destination: 8x8 or smaller block-linear RGBA8 surface.
    registers[0x80] = 0xd5;  // A8B8G8R8 -> R8G8B8A8Unorm
    registers[0x81] = 0;     // block linear
    registers[0x83] = 1;     // depth
    registers[0x86] = 8;
    registers[0x87] = 8;
    registers[0x88] = std::uint32_t(destination_gpu_address >> 32);
    registers[0x89] = std::uint32_t(destination_gpu_address);
    // Source: block-linear by default; pitch tests rewrite the block.
    registers[0x8c] = 0xd5;
    registers[0x8d] = 0;
    registers[0x8f] = 1;
    registers[0x92] = 8;
    registers[0x93] = 8;
    registers[0x94] = std::uint32_t(source_gpu_address >> 32);
    registers[0x95] = std::uint32_t(source_gpu_address);
    registers[0xab] = operation;
    registers[0x223] = sample_mode;
    registers[0x22c] = 0;  // destination x0
    registers[0x22d] = 0;  // destination y0
    registers[0x22e] = 8;  // destination width
    registers[0x22f] = 8;  // destination height
    // du/dx = dv/dy = 1.0, source origin (0, 0).
    registers[0x231] = 1;
    registers[0x233] = 1;
    return stop;
}

TextureLayoutDescription block_linear_layout(unsigned width, unsigned height) {
    TextureLayoutDescription description{};
    description.width = width;
    description.height = height;
    description.depth = 1;
    description.levels = 1;
    description.layers = 1;
    description.bytes_per_block = 4;
    description.storage = TextureStorageLayout::BlockLinear;
    return std::get<TextureLayoutDescription>(calculate_texture_size(description));
}

void set_block_linear_source(Backing& backing, unsigned width, unsigned height,
                             const std::vector<std::byte>& linear) {
    const auto converted =
        convert_texture_to_block_linear(block_linear_layout(width, height), linear);
    const auto& storage = std::get<std::vector<std::byte>>(converted);
    std::memcpy(backing.bytes.data(), storage.data(), storage.size());
}

std::vector<std::byte> read_block_linear(Backing& backing, unsigned width, unsigned height) {
    const auto converted = convert_texture_to_linear(
        block_linear_layout(width, height),
        std::span<const std::byte>(backing.bytes.data() + 0x1000, 0x200));
    return std::get<std::vector<std::byte>>(converted);
}

// A 4x4 RGBA8 gradient with one distinct byte lane value per pixel.
std::vector<std::byte> gradient(unsigned width, unsigned height) {
    std::vector<std::byte> linear(std::size_t{width} * height * 4);
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            const std::size_t offset = (std::size_t{y} * width + x) * 4;
            linear[offset] = std::byte(x * 16 + y * 4 + 4);
            linear[offset + 1] = std::byte(x * 8 + y * 2 + 2);
            linear[offset + 2] = std::byte(x * 4 + y + 1);
            linear[offset + 3] = std::byte(0x80);
        }
    }
    return linear;
}

void test_same_size_block_linear_copy(RenderCallbacks& backend) {
    auto backing = std::make_shared<Backing>();
    backing->owner = backing;
    auto memory = backing->memory();
    const auto source = gradient(8, 8);
    set_block_linear_source(*backing, 8, 8, source);
    const auto stop = blit_stop(3, 0);
    const auto result = backend.blit(backend.context, stop, memory);
    if (!result.ok()) {
        std::fprintf(stderr, "Same-size blit: %.*s\n", int(result.detail.size()),
                     result.detail.data());
    }
    CHECK(result.ok());
    // The copy preserves payload bytes; padded GOB columns keep their prior
    // guest bytes, so compare the decoded linear images instead.
    const auto written = read_block_linear(*backing, 8, 8);
    CHECK(written == source);
    if (written != source) {
        const auto mismatch = std::mismatch(written.begin(), written.end(), source.begin());
        const auto offset = std::size_t(mismatch.first - written.begin());
        std::fprintf(stderr, "First payload mismatch at %#zx: source %02x dest %02x\n", offset,
            std::to_integer<unsigned>(*mismatch.second),
            std::to_integer<unsigned>(*mismatch.first));
    }
    // The copy did not disturb storage beyond the destination surface.
    CHECK(backing->bytes[0x1400] == std::byte{0x19});
}

void test_pitch_linear_source(RenderCallbacks& backend) {
    auto backing = std::make_shared<Backing>();
    backing->owner = backing;
    auto memory = backing->memory();
    const auto source = gradient(8, 8);
    // Pitch-linear source: 8 pixels per row, 32-byte stride.
    for (unsigned y = 0; y < 8; ++y)
        std::memcpy(backing->bytes.data() + std::size_t{y} * 32, source.data() + std::size_t{y} * 32, 32);
    auto stop = blit_stop(3, 0);
    stop.registers[0x8d] = 1;  // pitch layout
    stop.registers[0x91] = 32; // pitch bytes
    const auto result = backend.blit(backend.context, stop, memory);
    if (!result.ok()) {
        std::fprintf(stderr, "Pitch source blit: %.*s\n", int(result.detail.size()),
                     result.detail.data());
    }
    CHECK(result.ok());
    CHECK(read_block_linear(*backing, 8, 8) == source);
}

void test_linear_downscale_against_cpu_reference(RenderCallbacks& backend) {
    auto backing = std::make_shared<Backing>();
    backing->owner = backing;
    auto memory = backing->memory();
    const auto source = gradient(4, 4);
    set_block_linear_source(*backing, 4, 4, source);
    auto stop = blit_stop(3, 0x10);  // bilinear filter
    stop.registers[0x92] = 4;        // source 4x4
    stop.registers[0x93] = 4;
    stop.registers[0x22e] = 2;       // destination 2x2
    stop.registers[0x22f] = 2;
    stop.registers[0x231] = 2;       // du/dx 2.0
    stop.registers[0x233] = 2;       // dv/dy 2.0
    const auto result = backend.blit(backend.context, stop, memory);
    if (!result.ok()) {
        std::fprintf(stderr, "Downscale blit: %.*s\n", int(result.detail.size()),
                     result.detail.data());
    }
    CHECK(result.ok());
    // CPU reference: each destination pixel averages the four source texels
    // its center lands between.
    const auto linear = read_block_linear(*backing, 8, 8);
    for (unsigned channel = 0; channel < 4; ++channel) {
        for (unsigned y = 0; y < 2; ++y) {
            for (unsigned x = 0; x < 2; ++x) {
                unsigned sum = 0;
                for (unsigned sy = 0; sy < 2; ++sy)
                    for (unsigned sx = 0; sx < 2; ++sx)
                        sum += unsigned(source[((std::size_t{y} * 2 + sy) * 4 +
                                                (std::size_t{x} * 2 + sx)) * 4 + channel]);
                const auto expected = std::byte(sum / 4);
                const auto actual =
                    linear[(std::size_t{y} * 8 + x) * 4 + channel];
                const int difference = int(std::to_integer<unsigned>(actual)) -
                                       int(std::to_integer<unsigned>(expected));
                if (std::abs(difference) > 1) {
                    std::fprintf(stderr,
                                 "FAIL downscale pixel %u,%u channel %u: expected %u got %u\n",
                                 x, y, channel, std::to_integer<unsigned>(expected),
                                 std::to_integer<unsigned>(actual));
                    ++failures;
                }
            }
        }
    }
}

void test_unsupported_operation(RenderCallbacks& backend) {
    auto backing = std::make_shared<Backing>();
    backing->owner = backing;
    auto memory = backing->memory();
    const auto before = backing->bytes;
    const auto stop = blit_stop(4, 0);  // Rop
    const auto result = backend.blit(backend.context, stop, memory);
    CHECK(result.status == CommandStatus::Unsupported);
    if (result.detail.find("operation") == std::string_view::npos)
        std::fprintf(stderr, "Unsupported blit detail: %.*s\n", int(result.detail.size()),
                     result.detail.data());
    CHECK(result.detail.find("operation") != std::string_view::npos);
    CHECK(backing->bytes == before);
}

void test_point_upscale_quad(RenderCallbacks& backend) {
    // Full-rect quad path (2x point upscale): isolates quad rasterization
    // from the direct-copy blit encoder.
    auto backing = std::make_shared<Backing>();
    backing->owner = backing;
    auto memory = backing->memory();
    const auto source = gradient(4, 4);
    set_block_linear_source(*backing, 4, 4, source);
    auto stop = blit_stop(3, 0);
    stop.registers[0x92] = 4;
    stop.registers[0x93] = 4;
    stop.registers[0x230] = 0x80000000;  // du/dx 0.5
    stop.registers[0x231] = 0;
    stop.registers[0x232] = 0x80000000;  // dv/dy 0.5
    stop.registers[0x233] = 0;
    const auto result = backend.blit(backend.context, stop, memory);
    if (!result.ok()) {
        std::fprintf(stderr, "Point upscale blit: %.*s\n", int(result.detail.size()),
                     result.detail.data());
    }
    CHECK(result.ok());
    // Point samples land on the even source texel of each 2x2 destination
    // block: dest (x, y) = source (x / 2, y / 2).
    const auto linear = read_block_linear(*backing, 8, 8);
    for (unsigned y = 0; y < 8; ++y) {
        for (unsigned x = 0; x < 8; ++x) {
            const auto expected = std::array<std::byte, 4>{
                source[(std::size_t{y / 2} * 4 + x / 2) * 4],
                source[(std::size_t{y / 2} * 4 + x / 2) * 4 + 1],
                source[(std::size_t{y / 2} * 4 + x / 2) * 4 + 2],
                source[(std::size_t{y / 2} * 4 + x / 2) * 4 + 3]};
            for (unsigned channel = 0; channel < 4; ++channel) {
                const auto actual = linear[(std::size_t{y} * 8 + x) * 4 + channel];
                const int difference = int(std::to_integer<unsigned>(actual)) -
                                       int(std::to_integer<unsigned>(expected[channel]));
                if (std::abs(difference) > 1) {
                    std::fprintf(stderr, "FAIL upscale pixel %u,%u channel %u\n", x, y, channel);
                    ++failures;
                }
            }
        }
    }
}

}  // namespace

int main() {
    @autoreleasepool {
        armigo::platform::MetalEngineOptions options;
        options.mode = armigo::platform::MetalEngineMode::Persistent;
        options.statistics = std::make_shared<armigo::platform::MetalEngineStatistics>();
        auto created = metal_backend::create_engine_backend(std::move(options),
            metal_backend::create_pipeline_state_cache);
        if (const auto* error = std::get_if<RendererError>(&created)) {
            const auto result = error->command();
            std::fprintf(stderr, "Engine factory: %.*s\n", int(result.detail.size()),
                         result.detail.data());
            CHECK(false);
            return failures != 0 ? 1 : 0;
        }
        auto backend = std::get<RenderCallbacks>(std::move(created));
        CHECK(backend.blit != nullptr);
        if (!backend.blit) return 1;
        test_same_size_block_linear_copy(backend);
        test_pitch_linear_source(backend);
        test_point_upscale_quad(backend);
        test_linear_downscale_against_cpu_reference(backend);
        test_unsupported_operation(backend);
        if (failures != 0) {
            std::fprintf(stderr, "%u twod blit GPU checks failed\n", failures);
            return 1;
        }
        return 0;
    }
}
