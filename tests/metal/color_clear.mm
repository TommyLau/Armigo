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
#import <Metal/Metal.h>

#include <armigo/platform/metal_color_clear.h>
#include <armigo/graphics/color_clear.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace armigo::graphics;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); ++failures; } } while (false)

struct Backing {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x100000, std::byte{0xa5});
    bool fail_write = false;
    unsigned writes = 0;
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < 0x800000 || address - 0x800000 > self.bytes.size() ||
            out.size() > self.bytes.size() - (address - 0x800000)) return false;
        std::memcpy(out.data(), self.bytes.data() + address - 0x800000, out.size());
        return true;
    }
    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        ++self.writes;
        if (self.fail_write || address < 0x800000 || address - 0x800000 > self.bytes.size() ||
            in.size() > self.bytes.size() - (address - 0x800000)) return false;
        std::memcpy(self.bytes.data() + address - 0x800000, in.data(), in.size());
        return true;
    }
    GpuAddressSpace memory() {
        const Mapping mapping{0x100000, bytes.size(), 0, 0x800000, bytes.size(), 0x1000};
        return std::get<GpuAddressSpace>(make_address_space({&mapping, 1}, {read, write, this}));
    }
};

RenderStop snapshot(bool tiled, ColorTargetFormat format, unsigned block_height = 0,
                    ColorClearRegion region = {}) {
    RenderStop stop{};
    stop.engine = engine_class::threed;
    stop.kind = RenderKind::Clear;
    stop.registers[0x674] = 0x3c;
    stop.registers[0x556] = 1;
    stop.registers[0xdf] = 1;
    stop.registers[0x3fd] = 37u << 16;
    stop.registers[0x3fe] = 139u << 16;
    stop.registers[0x206] = 1;
    if (region.width != 0 || region.height != 0) {
        stop.registers[0x43e] = 0x100;
        stop.registers[0x380] = 1;
        stop.registers[0x381] = region.x | ((region.x + region.width) << 16);
        stop.registers[0x382] = region.y | ((region.y + region.height) << 16);
    }
    auto& target = stop.color_targets[0];
    target.address = 0x100000;
    target.width = tiled ? 37 : 192;
    target.height = 139;
    target.depth = 1;
    target.format = static_cast<std::uint32_t>(format);
    target.tile_mode = tiled ? block_height << 4 : 0x1000;
    constexpr std::array color{1.0f, 0.5f, 0.0f, 1.0f};
    for (unsigned i = 0; i < color.size(); ++i)
        stop.registers[0x360 + i] = std::bit_cast<std::uint32_t>(color[i]);
    return stop;
}

std::uint32_t expected_word(ColorTargetFormat format, const std::array<float, 4>& color) {
    if (format == ColorTargetFormat::R8G8B8A8Srgb) {
        // The GPU stores sRGB-encoded RGB and plain UNORM alpha.
        std::uint32_t word = 0;
        for (unsigned component = 0; component < 4; ++component) {
            const float value = std::clamp(color[component], 0.0f, 1.0f);
            const float encoded = component == 3
                ? value
                : (value <= 0.0031308f ? 12.92f * value
                                       : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f);
            word |= unsigned(std::lround(encoded * 255.0f)) << (8 * component);
        }
        return word;
    }
    if (format == ColorTargetFormat::R8G8B8A8Unorm) {
        std::uint32_t word = 0;
        for (unsigned component = 0; component < 4; ++component) {
            word |= unsigned(std::lround(std::clamp(color[component], 0.0f, 1.0f) * 255.0f)) <<
                    (8 * component);
        }
        return word;
    }
    std::uint32_t word = 0;
    unsigned shift = 0;
    for (unsigned component = 0; component < 4; ++component) {
        const unsigned bits = component == 3 ? 2u : 10u;
        word |= unsigned(std::lround(std::clamp(color[component], 0.0f, 1.0f) * float((1u << bits) - 1)))
                << shift;
        shift += bits;
    }
    return word;
}

bool rgb_close(std::uint32_t actual, std::uint32_t expected, unsigned tolerance) {
    for (unsigned shift = 0; shift < 24; shift += 8) {
        const int difference = int((actual >> shift) & 0xff) - int((expected >> shift) & 0xff);
        if (std::abs(difference) > int(tolerance)) return false;
    }
    return true;
}

void verify_clear(bool tiled, ColorTargetFormat format, unsigned block_height,
                  ColorClearRegion region = {}, const std::array<float, 4>& color =
                                                    {1.0f, 0.5f, 0.0f, 1.0f}) {
    Backing backing;
    auto memory = backing.memory();
    auto stop = snapshot(tiled, format, block_height, region);
    for (unsigned component = 0; component < color.size(); ++component)
        stop.registers[0x360 + component] = std::bit_cast<std::uint32_t>(color[component]);
    const auto planned = plan_color_clear(stop);
    CHECK(std::holds_alternative<ColorClearPlan>(planned));
    if (!std::holds_alternative<ColorClearPlan>(planned)) return;
    const auto& plan = std::get<ColorClearPlan>(planned);
    const ColorClearRegion cleared = region.width != 0 || region.height != 0
        ? region
        : ColorClearRegion{0, 0, plan.width, plan.height};
    CHECK(plan.region.x == cleared.x && plan.region.y == cleared.y);
    CHECK(plan.region.width == cleared.width && plan.region.height == cleared.height);
    armigo::platform::MetalClearStatistics stats;
    const auto result = armigo::platform::execute_metal_color_clear(&stats, stop, memory);
    if (!result.ok()) std::fprintf(stderr, "%.*s\n", int(result.detail.size()), result.detail.data());
    CHECK(result.ok());
    CHECK(stats.completed_clears == 1);
    CHECK(stats.written_pixels == std::uint64_t{cleared.width} * cleared.height);
    const std::uint32_t expected = expected_word(format, color);
    // The tolerance applies to sRGB-encoded RGB only; alpha stays exact.
    const unsigned tolerance = format == ColorTargetFormat::R8G8B8A8Srgb ? 2u : 0u;
    std::vector<bool> touched(backing.bytes.size(), false);
    for (std::uint32_t y = 0; y < cleared.height; ++y) {
        for (std::uint32_t x = 0; x < cleared.width; ++x) {
            const auto offset =
                std::get<std::size_t>(pixel_byte_offset(plan, cleared.x + x, cleared.y + y));
            std::uint32_t actual = 0;
            std::memcpy(&actual, backing.bytes.data() + offset, 4);
            if (tolerance == 0) {
                CHECK(actual == expected);
            } else {
                // sRGB encoding precision differs between hosts for RGB only;
                // the alpha channel is plain UNORM and must be exact.
                CHECK(rgb_close(actual, expected, tolerance));
                CHECK(((actual >> 24) & 0xff) == ((expected >> 24) & 0xff));
            }
            for (unsigned b = 0; b < 4; b++) touched[offset + b] = true;
        }
    }
    for (std::size_t i = 0; i < touched.size(); ++i)
        if (!touched[i]) CHECK(backing.bytes[i] == std::byte{0xa5});
}

void verify_falcon_then_clear() {
    Backing backing;
    auto memory = backing.memory();
    const auto stop = snapshot(true, ColorTargetFormat::R8G8B8A8Unorm);
    EngineFrontEnd engines;
    armigo::platform::MetalClearStatistics stats;
    CompletionCallbacks completion{};
    completion.rendering = {armigo::platform::execute_metal_color_clear, &stats};
    const auto write = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call({method, 0, argument, 0x4000, true},
                            engine_class::threed, memory, completion);
    };
    for (const auto method : {0x3fdu, 0x3feu, 0x206u,
                              0x360u, 0x361u, 0x362u, 0x363u})
        CHECK(write(method, stop.registers[method]).ok());
    const auto& target = stop.color_targets[0];
    CHECK(write(0x201, static_cast<std::uint32_t>(target.address)).ok());
    CHECK(write(0x202, target.width).ok());
    CHECK(write(0x203, target.height).ok());
    CHECK(write(0x204, target.format).ok());
    CHECK(write(0x205, target.tile_mode).ok());
    for (std::uint32_t index = 0; index < 29; ++index) {
        CHECK(write(0xd00, 0).ok());
        CHECK(write(0xd01, index).ok());
        CHECK(write(0xd02, ~index).ok());
        CHECK(write(0x8c4, index * 3).ok());
    }
    CHECK(stats.completed_clears == 0);
    CHECK(write(0x674, 0x3c).ok());
    CHECK(stats.completed_clears == 1);
    CHECK(stats.written_pixels == 37 * 139);
    CHECK(!engines.rendering_stop());
    const auto plan = std::get<ColorClearPlan>(plan_color_clear(stop));
    for (std::uint32_t y = 0; y < plan.height; ++y) {
        for (std::uint32_t x = 0; x < plan.width; ++x) {
            const auto offset = std::get<std::size_t>(pixel_byte_offset(plan, x, y));
            std::uint32_t actual = 0;
            std::memcpy(&actual, backing.bytes.data() + offset, 4);
            CHECK(actual == 0xff0080ffu);
        }
    }
    CHECK(write(0x585, 1).status == CommandStatus::Stopped);
    CHECK(stats.completed_clears == 1);
}

void verify_failure_gates() {
    Backing backing;
    auto memory = backing.memory();
    auto stop = snapshot(true, ColorTargetFormat::R8G8B8A8Unorm);
    armigo::platform::MetalClearStatistics stats;
    stop.color_targets[0].address = 0x200000;
    CHECK(armigo::platform::execute_metal_color_clear(&stats, stop, memory).status ==
          CommandStatus::MemoryFault);
    CHECK(backing.writes == 0);
    CHECK(stats.written_pixels == 0);
    stop.color_targets[0].address = 0x100000;
    backing.fail_write = true;
    CHECK(armigo::platform::execute_metal_color_clear(&stats, stop, memory).status ==
          CommandStatus::MemoryFault);
    CHECK(stats.completed_clears == 0);
    CHECK(stats.written_pixels == 0);
    CHECK(backing.writes == 1);

    // A scissored clear reports the same gates: the writeback halts before
    // completion, and the partial region never reports success.
    Backing region_backing;
    auto region_memory = region_backing.memory();
    const ColorClearRegion region{13, 20, 13, 40};
    const auto scissored = snapshot(true, ColorTargetFormat::R8G8B8A8Srgb, 1, region);
    CHECK(std::holds_alternative<ColorClearPlan>(plan_color_clear(scissored)));
    region_backing.fail_write = true;
    CHECK(armigo::platform::execute_metal_color_clear(&stats, scissored, region_memory).status ==
          CommandStatus::MemoryFault);
    CHECK(stats.completed_clears == 0);
    CHECK(stats.written_pixels == 0);
    CHECK(region_backing.writes == 1);
}
}  // namespace

int main() {
    for (const auto format : {ColorTargetFormat::R8G8B8A8Unorm,
                             ColorTargetFormat::A2B10G10R10Unorm}) {
        verify_clear(false, format, 0);
        for (unsigned height : {0u, 1u, 4u}) verify_clear(true, format, height);
    }
    // Partial regions with an unaligned nonzero origin crossing the tiled GOB
    // boundary at x = 16 pixels (the origin sits mid 16-byte segment) and the
    // block boundary at y = 32 rows, and spanning pitch padding columns in
    // linear storage. Mid-range linear RGB exercises the GPU's sRGB encoding;
    // the UNORM variants compare exactly.
    constexpr ColorClearRegion region{13, 20, 13, 40};
    constexpr std::array<float, 4> mid_color{0.5f, 0.25f, 0.75f, 0.5f};
    for (const auto format : {ColorTargetFormat::R8G8B8A8Srgb,
                              ColorTargetFormat::R8G8B8A8Unorm}) {
        verify_clear(false, format, 0, region, mid_color);
        verify_clear(true, format, 1, region, mid_color);
    }
    verify_falcon_then_clear();
    verify_failure_gates();
    return failures == 0 ? 0 : 1;
}
