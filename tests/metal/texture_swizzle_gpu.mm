// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic GPU tests compare both Metal texture-swizzle directions against the
// production CPU converters, including byte-exact preservation of padding.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "texture_swizzle_gpu.h"

#include <armigo/graphics/texture_layout.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;
namespace metal = armigo::platform::metal;

unsigned failures = 0;

#define CHECK(...)                                                             \
    do {                                                                       \
        if (!(__VA_ARGS__)) {                                                  \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__);   \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

constexpr std::byte linear_sentinel{0xAA};
constexpr std::byte guest_sentinel{0x55};

// Position-unique synthetic payload byte, independent of the layout code.
std::byte pattern_byte(std::uint64_t layer, std::uint64_t level, std::uint64_t z,
                       std::uint64_t row, std::uint64_t column) {
    const std::uint64_t value = (layer * 0x9E3779B97F4A7C15ull) ^
        (level * 0xC2B2AE3D27D4EB4Full) ^ (z * 0x165667B19E3779F9ull) ^
        (row * 0x27D4EB2F165667C5ull) ^ (column * 0x85EBCA77C2B2AE63ull);
    return std::byte(static_cast<unsigned char>(value >> 33));
}

// Synchronous single-encoder runner with the same commit and feedback shape
// the texture cache transfers use.
bool execute(id<MTLDevice> device, id<MTL4CommandQueue> queue,
             NSArray<id<MTLAllocation>>* transfers,
             const std::function<void(id<MTL4ComputeCommandEncoder>)>& encode) {
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    auto* descriptor = [MTLResidencySetDescriptor new];
    descriptor.initialCapacity = transfers.count;
    NSError* error = nil;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:descriptor error:&error];
    if (!allocator || !commands || !residency) return false;
    for (id<MTLAllocation> allocation in transfers) [residency addAllocation:allocation];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    if (!encoder) return false;
    [encoder barrierAfterQueueStages:MTLStageAll
                        beforeStages:(MTLStageBlit | MTLStageDispatch)
                   visibilityOptions:MTL4VisibilityOptionDevice];
    encode(encoder);
    [encoder barrierAfterStages:(MTLStageBlit | MTLStageDispatch)
              beforeQueueStages:MTLStageAll
              visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder endEncoding];
    [commands endCommandBuffer];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block bool completed = false;
    auto* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        completed = feedback.error == nil;
        dispatch_semaphore_signal(done);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)))
        return false;
    return completed;
}

struct Case {
    std::uint32_t width, height, depth, levels, layers;
    std::size_t bytes_per_block;
    std::uint32_t gob_height_log2, gob_depth_log2;
};

void run_case(id<MTLDevice> device, id<MTL4CommandQueue> queue,
              metal::GpuTextureSwizzle& swizzle, const Case& item) {
    TextureLayoutDescription description{};
    description.width = item.width;
    description.height = item.height;
    description.depth = item.depth;
    description.levels = item.levels;
    description.layers = item.layers;
    description.bytes_per_block = item.bytes_per_block;
    description.gob_block_height_log2 = item.gob_height_log2;
    description.gob_block_depth_log2 = item.gob_depth_log2;
    description.storage = TextureStorageLayout::BlockLinear;
    const auto calculated = calculate_texture_size(description);
    if (!std::holds_alternative<TextureLayoutDescription>(calculated)) {
        CHECK(false && "layout calculation failed");
        return;
    }
    const auto& layout = std::get<TextureLayoutDescription>(calculated);

    // Synthetic guest storage: every byte of the texture's extents carries a
    // position-unique value.
    std::vector<std::byte> guest(layout.total_size);
    for (std::size_t i = 0; i < guest.size(); ++i)
        guest[i] = std::byte(static_cast<unsigned char>(i * 131 + (i >> 8) * 29 + 3));

    // Linear placement with an odd pitch gap so the kernel's partial-segment
    // and padding handling is exercised on every level.
    constexpr std::size_t pitch_gap = 37;
    struct Placement {
        std::uint32_t level, layer;
        std::size_t offset, pitch, row_bytes, rows;
    };
    std::vector<Placement> placements;
    std::size_t linear_bytes = 0;
    for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
        for (std::uint32_t level = 0; level < layout.levels; ++level) {
            const auto& mip = layout.mips[level];
            const std::size_t pitch = mip.linear_row_bytes + pitch_gap;
            placements.push_back({level, layer, linear_bytes, pitch,
                                  mip.linear_row_bytes, mip.block_rows});
            linear_bytes += mip.depth * mip.block_rows * pitch;
        }
    }

    id<MTLBuffer> guest_buffer = [device newBufferWithLength:layout.total_size
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> linear_buffer = [device newBufferWithLength:linear_bytes
                                                      options:MTLResourceStorageModeShared];
    if (!guest_buffer || !linear_buffer) {
        CHECK(false && "staging allocation failed");
        return;
    }

    std::vector<metal::TextureSwizzleExtent> extents(placements.size());
    for (std::size_t slot = 0; slot < placements.size(); ++slot) {
        const auto& place = placements[slot];
        const auto& mip = layout.mips[place.level];
        auto& extent = extents[slot];
        extent.guest_offset = std::uint64_t{place.layer} * layout.layer_stride + mip.offset;
        extent.guest_size = mip.size;
        extent.linear_offset = place.offset;
        extent.linear_pitch = static_cast<std::uint32_t>(place.pitch);
        extent.row_bytes = place.row_bytes;
        extent.rows = place.rows;
        extent.slices = mip.depth;
        extent.gob_block_height_log2 = mip.gob_block_height_log2;
        extent.gob_block_depth_log2 = mip.gob_block_depth_log2;
        // Independent GOB column count: one GOB per 64 payload bytes (the
        // sparse-tile alignment is not exercised by these layouts).
        extent.gobs_x = static_cast<std::uint32_t>((place.row_bytes + 63) >> 6);
    }
    if (auto ready = swizzle.reserve(extents.size());
        std::holds_alternative<RendererError>(ready)) {
        CHECK(false && "parameter reservation failed");
        return;
    }
    NSArray<id<MTLAllocation>>* transfer_allocations =
        @[guest_buffer, linear_buffer, swizzle.parameters_buffer()];

    // --- Deswizzle: guest storage must decode to the CPU converter's image.
    std::memcpy(guest_buffer.contents, guest.data(), guest.size());
    std::memset(linear_buffer.contents,
                static_cast<int>(linear_sentinel), linear_bytes);
    const bool decoded = execute(device, queue, transfer_allocations,
        [&](id<MTL4ComputeCommandEncoder> encoder) {
            for (std::size_t slot = 0; slot < extents.size(); ++slot)
                CHECK(std::holds_alternative<std::monostate>(
                    swizzle.encode_deswizzle(encoder, guest_buffer, linear_buffer,
                                             static_cast<std::uint32_t>(slot),
                                             extents[slot])));
        });
    CHECK(decoded && "deswizzle execution failed");
    if (decoded) {
        for (std::size_t slot = 0; slot < placements.size(); ++slot) {
            const auto& place = placements[slot];
            const auto converted =
                convert_level_to_linear(layout, guest, place.level, place.layer);
            if (!std::holds_alternative<TextureLevelData>(converted)) {
                CHECK(false && "CPU deswizzle oracle failed");
                continue;
            }
            const auto& expected = std::get<TextureLevelData>(converted);
            const auto* staged = static_cast<const std::byte*>(linear_buffer.contents) +
                                 place.offset;
            const auto* source = expected.bytes.data();
            for (std::uint32_t z = 0; z < expected.depth; ++z) {
                for (std::uint32_t row = 0; row < place.rows; ++row) {
                    const std::byte* gpu_row =
                        staged + (std::size_t{z} * place.rows + row) * place.pitch;
                    const std::byte* cpu_row = source +
                        (std::size_t{z} * place.rows + row) * expected.row_bytes;
                    if (std::memcmp(gpu_row, cpu_row, expected.row_bytes) != 0) {
                        CHECK(false && "GPU deswizzle differs from CPU conversion");
                        return;
                    }
                    for (std::size_t pad = expected.row_bytes; pad < place.pitch; ++pad)
                        if (gpu_row[pad] != linear_sentinel) {
                            CHECK(false && "GPU deswizzle wrote row padding");
                            return;
                        }
                }
            }
        }
    }

    // --- Swizzle: linear pattern must tile exactly like the CPU inverse,
    // including preserving bytes outside the covered level extents.
    for (std::size_t slot = 0; slot < placements.size(); ++slot) {
        const auto& place = placements[slot];
        auto* staged = static_cast<std::byte*>(linear_buffer.contents) + place.offset;
        for (std::uint32_t z = 0; z < layout.mips[place.level].depth; ++z)
            for (std::uint32_t row = 0; row < place.rows; ++row)
                for (std::size_t column = 0; column < place.row_bytes; ++column)
                    staged[(std::size_t{z} * place.rows + row) * place.pitch + column] =
                        pattern_byte(place.layer, place.level, z, row, column);
    }
    // The CPU oracle starts from the same sentinel image and writes each
    // level with the production inverse converter.
    std::vector<std::byte> oracle(guest.size(), guest_sentinel);
    std::vector<std::byte> pattern(linear_bytes);
    for (const auto& place : placements) {
        const auto& mip = layout.mips[place.level];
        std::vector<std::byte> level_pattern(
            std::size_t{mip.depth} * place.rows * place.row_bytes);
        for (std::uint32_t z = 0; z < mip.depth; ++z)
            for (std::uint32_t row = 0; row < place.rows; ++row)
                for (std::size_t column = 0; column < place.row_bytes; ++column)
                    level_pattern[(std::size_t{z} * place.rows + row) * place.row_bytes +
                                  column] = pattern_byte(place.layer, place.level, z,
                                                         row, column);
        if (std::holds_alternative<TextureLayoutError>(write_level_from_linear(
                layout, level_pattern, oracle, place.level, place.layer))) {
            CHECK(false && "CPU swizzle oracle failed");
            return;
        }
    }
    std::memset(guest_buffer.contents, static_cast<int>(guest_sentinel), guest.size());
    const bool encoded = execute(device, queue, transfer_allocations,
        [&](id<MTL4ComputeCommandEncoder> encoder) {
            for (std::size_t slot = 0; slot < extents.size(); ++slot)
                CHECK(std::holds_alternative<std::monostate>(
                    swizzle.encode_swizzle(encoder, linear_buffer, guest_buffer,
                                           static_cast<std::uint32_t>(slot),
                                           extents[slot])));
        });
    CHECK(encoded && "swizzle execution failed");
    if (encoded && std::memcmp(guest_buffer.contents, oracle.data(), oracle.size()) != 0) {
        CHECK(false && "GPU swizzle differs from CPU conversion");
    }
}

} // namespace

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::puts("Skipping texture swizzle GPU tests: no Metal 4 device.");
        return 0;
    }
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    const auto created = metal::GpuTextureSwizzle::create(device);
    if (!std::holds_alternative<std::shared_ptr<metal::GpuTextureSwizzle>>(created)) {
        std::fprintf(stderr, "texture swizzle pipeline creation failed\n");
        return 1;
    }
    auto& swizzle = std::get<std::shared_ptr<metal::GpuTextureSwizzle>>(created);

    const std::vector<Case> cases = {
        // bytes-per-block coverage with odd sizes, block heights 1-32 GOBs.
        {100, 7, 1, 1, 1, 1, 0, 0},
        {37, 29, 1, 3, 2, 1, 5, 0},
        {5, 3, 1, 2, 1, 2, 1, 0},
        {64, 16, 1, 1, 1, 2, 2, 0},
        {13, 11, 1, 2, 2, 4, 3, 0},
        {1, 1, 1, 1, 1, 4, 1, 0},
        {23, 9, 1, 2, 1, 8, 4, 0},
        {10, 6, 1, 1, 1, 12, 2, 0},
        {19, 19, 1, 2, 1, 12, 1, 0},
        {7, 5, 1, 2, 1, 16, 2, 0},
        {200, 50, 1, 2, 2, 8, 3, 0},
        {64, 24, 1, 2, 1, 4, 2, 0},
        {17, 32, 1, 1, 1, 2, 1, 0},
        {300, 200, 1, 9, 1, 4, 3, 0},
        // 3D with depth blocks and shrinking GOB depth.
        {32, 16, 3, 2, 1, 4, 1, 1},
        {9, 9, 5, 1, 1, 1, 2, 2},
        {5, 5, 2, 3, 1, 2, 0, 1},
        // Array large enough to force parameter buffer growth past 16 slots.
        {21, 13, 1, 1, 24, 4, 1, 0},
    };
    for (const auto& item : cases) run_case(device, queue, *swizzle, item);

    // An extent beyond the staging buffer must fail with a structured error
    // instead of encoding dispatches that touch memory.
    metal::TextureSwizzleExtent oversized;
    oversized.guest_offset = 0;
    oversized.guest_size = 512;
    oversized.linear_offset = 0;
    oversized.linear_pitch = 64;
    oversized.row_bytes = 64;
    oversized.rows = 8;
    oversized.slices = 1;
    oversized.gobs_x = 1;
    id<MTLBuffer> small_buffer = [device newBufferWithLength:256
                                                     options:MTLResourceStorageModeShared];
    id<MTL4CommandAllocator> probe_allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> probe_commands = [device newCommandBuffer];
    [probe_commands beginCommandBufferWithAllocator:probe_allocator];
    id<MTL4ComputeCommandEncoder> probe = [probe_commands computeCommandEncoder];
    if (auto status = swizzle->encode_deswizzle(probe, small_buffer, small_buffer, 0, oversized);
        !std::holds_alternative<RendererError>(status)) {
        CHECK(false && "oversized extent must return a structured error");
    }
    [probe endEncoding];
    [probe_commands endCommandBuffer];

    if (failures == 0) std::puts("Verified GPU texture deswizzle and swizzle against CPU conversions.");
    return failures == 0 ? 0 : 1;
}
