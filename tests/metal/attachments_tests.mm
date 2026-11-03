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

#include <armigo/graphics/render_target.h>
#include <attachments.h>

#include <algorithm>
#include <array>
#include <bit>
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
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

constexpr std::uint32_t rt_control(std::uint32_t count,
                                   const std::array<std::uint32_t, 8>& permutation) noexcept
{
    std::uint32_t packed = count & 0xf;
    for (std::uint32_t index = 0; index < permutation.size(); ++index) {
        packed |= (permutation[index] & 7) << (4 + index * 3);
    }
    return packed;
}

constexpr std::uint32_t sentinel_word = 0xa53c7e19u;

id<MTLDevice> ensure_device()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory) {
        std::fprintf(stderr, "FAIL: unified-memory Metal 4 device is unavailable\n");
        std::exit(1);
    }
    return device;
}

id<MTLTexture> make_texture(id<MTLDevice> device, MTLPixelFormat format,
                            NSUInteger width, NSUInteger height)
{
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
        width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    return [device newTextureWithDescriptor:descriptor];
}

// Builds a render target plan through the portable planner: the control word
// names count and permutation, the format words name the per-physical-target
// guest formats, and every slot shares the extent.
RenderTargetPlan plan_from_registers(std::uint32_t control,
                                     const std::array<std::uint32_t, 8>& formats,
                                     std::uint32_t width, std::uint32_t height)
{
    RenderStop stop{};
    stop.kind = RenderKind::Draw;
    stop.engine = engine_class::threed;
    stop.registers[0x574] = 0;
    stop.registers[0x487] = control;
    for (std::uint32_t physical = 0; physical < 8; ++physical) {
        if (formats[physical] == 0) continue;
        const std::size_t base = 0x200 + physical * 0x10;
        // Address zero is a null binding; give each target its own storage.
        const std::uint64_t address = 0x100000ull * (physical + 1);
        stop.registers[base] = std::uint32_t(address >> 32);
        stop.registers[base + 1] = std::uint32_t(address);
        stop.registers[base + 2] = width;
        stop.registers[base + 3] = height;
        stop.registers[base + 4] = formats[physical];
        stop.registers[base + 5] = 0;
        stop.registers[base + 6] = 1;
        // The frontend snapshot also carries the decoded target description.
        TargetDescription& target = stop.color_targets[physical];
        target.address = address;
        target.width = width;
        target.height = height;
        target.format = formats[physical];
        target.depth = 1;
    }
    auto planned = plan_render_targets(stop);
    if (!std::holds_alternative<RenderTargetPlan>(planned)) {
        std::fprintf(stderr, "FAIL: render target plan rejected\n");
        ++failures;
        return RenderTargetPlan{};
    }
    return std::get<RenderTargetPlan>(planned);
}

RenderTargetPlan depth_only_plan(std::uint32_t format, std::uint32_t width,
                                 std::uint32_t height)
{
    RenderStop stop{};
    stop.kind = RenderKind::Draw;
    stop.engine = engine_class::threed;
    stop.registers[0x574] = 0;
    stop.registers[0x487] = rt_control(0, {0, 0, 0, 0, 0, 0, 0, 0});
    stop.registers[0x3f9] = 0x100000;
    stop.registers[0x3fa] = format;
    stop.registers[0x3fb] = 0;
    stop.registers[0x3fc] = 0x1000;
    stop.registers[0x48a] = width;
    stop.registers[0x48b] = height;
    stop.registers[0x48c] = (1u << 16) | 1;
    stop.registers[0x54e] = 1;
    stop.depth.address = 0x100000;
    stop.depth.format = format;
    stop.depth.tile_mode = 0;
    stop.depth.width = width;
    stop.depth.height = height;
    stop.depth.enabled = true;
    auto planned = plan_render_targets(stop);
    if (!std::holds_alternative<RenderTargetPlan>(planned)) {
        std::fprintf(stderr, "FAIL: depth plan rejected\n");
        ++failures;
        return RenderTargetPlan{};
    }
    return std::get<RenderTargetPlan>(planned);
}

// One round trip over private textures: each texture is prefilled from shared
// staging (the sentinel pattern), the planned pass runs as an empty encoder
// (load actions perform the folded clears), then textures are copied into
// shared readback buffers. Returns the readback words in transfer order.
struct ColorTransfer {
    id<MTLTexture> texture = nil;
    id<MTLBuffer> staging = nil;
    id<MTLBuffer> output = nil;
    std::size_t row_bytes = 0;
    std::size_t size = 0;
};

bool commit_and_wait(id<MTLDevice> device, NSMutableArray* retained,
                     id<MTLResidencySet> residency, id<MTL4CommandBuffer> commands)
{
    static_cast<void>(residency);
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    [retained addObject:queue];
    [commands endCommandBuffer];
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    auto* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(finished);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(finished,
                                dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC))) {
        std::fprintf(stderr, "FAIL: GPU round trip timed out\n");
        ++failures;
        return false;
    }
    if (execution_error) {
        std::fprintf(stderr, "FAIL: GPU round trip failed: %s\n",
                     execution_error.localizedDescription.UTF8String);
        ++failures;
        return false;
    }
    return true;
}

std::vector<std::byte> fill_words(std::uint32_t word, std::size_t count)
{
    std::vector<std::byte> bytes(count * 4);
    for (std::size_t index = 0; index < count; ++index) {
        std::memcpy(bytes.data() + index * 4, &word, 4);
    }
    return bytes;
}

std::vector<std::vector<std::byte>> run_color_round_trip(
    id<MTLDevice> device,
    MTL4RenderPassDescriptor * pass,
    std::vector<ColorTransfer>& transfers)
{
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    auto* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 64;
    NSError* error = nil;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    NSMutableArray* retained = [NSMutableArray arrayWithObjects:
        allocator, commands, residency, nil];
    if (!allocator || !commands || !residency) {
        std::fprintf(stderr, "FAIL: round trip allocation failed\n");
        ++failures;
        return {};
    }
    for (ColorTransfer& transfer : transfers) {
        [retained addObject:transfer.texture];
        [residency addAllocation:transfer.texture];
        // RGBA16F texels are eight bytes; every other colour format here is four.
        const std::size_t texel_bytes =
            transfer.texture.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : 4;
        const std::vector<std::byte> prefill =
            fill_words(sentinel_word, std::size_t(transfer.texture.width) *
                                          transfer.texture.height * (texel_bytes / 4));
        transfer.staging = [device newBufferWithBytes:prefill.data()
            length:prefill.size() options:MTLResourceStorageModeShared];
        transfer.row_bytes = std::size_t(transfer.texture.width) * texel_bytes;
        transfer.size = transfer.row_bytes * transfer.texture.height;
        transfer.output = [device newBufferWithLength:transfer.size
            options:MTLResourceStorageModeShared];
        [retained addObjectsFromArray:@[transfer.staging, transfer.output]];
        [residency addAllocation:transfer.staging];
        [residency addAllocation:transfer.output];
    }
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> upload = [commands computeCommandEncoder];
    if (!upload) {
        std::fprintf(stderr, "FAIL: upload encoder creation failed\n");
        ++failures;
        return {};
    }
    for (const ColorTransfer& transfer : transfers) {
        [upload copyFromBuffer:transfer.staging sourceOffset:0
            sourceBytesPerRow:transfer.row_bytes
            sourceBytesPerImage:transfer.size
            sourceSize:MTLSizeMake(transfer.texture.width, transfer.texture.height, 1)
            toTexture:transfer.texture destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [upload endEncoding];
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) {
        std::fprintf(stderr, "FAIL: render encoder creation failed\n");
        ++failures;
        return {};
    }
    // The uploaded sentinels must be visible to the pass load actions.
    [render barrierAfterQueueStages:MTLStageBlit beforeStages:MTLStageAll
        visibilityOptions:MTL4VisibilityOptionDevice];
    [render endEncoding];
    id<MTL4ComputeCommandEncoder> download = [commands computeCommandEncoder];
    if (!download) {
        std::fprintf(stderr, "FAIL: readback encoder creation failed\n");
        ++failures;
        return {};
    }
    [download barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
        visibilityOptions:MTL4VisibilityOptionDevice];
    for (const ColorTransfer& transfer : transfers) {
        [download copyFromTexture:transfer.texture sourceSlice:0 sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
            sourceSize:MTLSizeMake(transfer.texture.width, transfer.texture.height, 1)
            toBuffer:transfer.output destinationOffset:0
            destinationBytesPerRow:transfer.row_bytes
            destinationBytesPerImage:transfer.size];
    }
    [download endEncoding];
    if (!commit_and_wait(device, retained, residency, commands)) return {};
    std::vector<std::vector<std::byte>> blocks;
    for (const ColorTransfer& transfer : transfers) {
        const auto* bytes = static_cast<const std::byte*>(transfer.output.contents);
        blocks.emplace_back(bytes, bytes + transfer.size);
    }
    return blocks;
}

std::uint32_t read_word(const std::vector<std::byte>& block, std::size_t word_index)
{
    std::uint32_t word = 0;
    std::memcpy(&word, block.data() + word_index * 4, 4);
    return word;
}

void expect_uniform(const std::vector<std::byte>& block, std::size_t word_count,
                    const std::function<std::uint32_t(std::size_t)>& expected,
                    const char* what)
{
    for (std::size_t word_index = 0; word_index < word_count; ++word_index) {
        const std::uint32_t actual = read_word(block, word_index);
        if (actual != expected(word_index)) {
            std::fprintf(stderr, "FAIL: %s word %zu wrote %08x expected %08x\n",
                         what, word_index, actual, expected(word_index));
            ++failures;
            return;
        }
    }
}

template <std::size_t slots>
metal::RenderPassClearValues
slot_colors(const std::array<std::array<float, 4>, slots>& colors)
{
    metal::RenderPassClearValues values{};
    for (std::size_t slot = 0; slot < slots; ++slot) {
        for (std::size_t component = 0; component < 4; ++component) {
            values.color[slot][component] = colors[slot][component];
        }
    }
    return values;
}

void test_eight_distinct_outputs(id<MTLDevice> device)
{
    RenderTargetPlan plan = plan_from_registers(rt_control(8, {0, 1, 2, 3, 4, 5, 6, 7}),
        {0xd5, 0xd5, 0xd5, 0xd5, 0xd5, 0xd5, 0xd5, 0xd5}, 64, 64);
    if (plan.colors[0] == std::nullopt) return;
    std::array<std::array<float, 4>, 8> colors{};
    for (std::size_t slot = 0; slot < 8; ++slot) {
        plan.colors[slot]->load = AttachmentLoad::Clear;
        plan.colors[slot]->physical_target = std::uint8_t(slot);
        colors[slot] = {slot * 0.125f, 1.0f - slot * 0.125f,
                        slot % 2 ? 0.75f : 0.25f, slot == 7 ? 1.0f : 0.5f};
    }
    metal::NativeRenderTargets targets;
    std::vector<ColorTransfer> transfers(8);
    for (std::size_t slot = 0; slot < 8; ++slot) {
        targets.colors[slot].texture =
            make_texture(device, MTLPixelFormatRGBA8Unorm, 64, 64);
        transfers[slot].texture = targets.colors[slot].texture;
    }
    const metal::RenderPassClearValues values = slot_colors(colors);
    const auto built = metal::make_render_pass(plan, targets, &values);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor *>(built));
    if (!std::holds_alternative<MTL4RenderPassDescriptor *>(built)) return;
    const auto blocks =
        run_color_round_trip(device, std::get<MTL4RenderPassDescriptor *>(built), transfers);
    CHECK(blocks.size() == 8);
    if (blocks.size() != 8) return;
    for (std::size_t slot = 0; slot < 8; ++slot) {
        const std::uint32_t expected =
            std::uint32_t(std::lround(colors[slot][0] * 255.0f)) |
            std::uint32_t(std::lround(colors[slot][1] * 255.0f)) << 8 |
            std::uint32_t(std::lround(colors[slot][2] * 255.0f)) << 16 |
            std::uint32_t(std::lround(colors[slot][3] * 255.0f)) << 24;
        expect_uniform(blocks[slot], 64 * 64, [expected](std::size_t) { return expected; },
                       "eight-output slot");
    }
    // Eight distinct outputs wrote eight distinct words.
    for (std::size_t left = 0; left < 8; ++left) {
        for (std::size_t right = left + 1; right < 8; ++right) {
            CHECK(read_word(blocks[left], 0) != read_word(blocks[right], 0));
        }
    }
}

void test_sparse_and_permuted_attachments(id<MTLDevice> device)
{
    // Logical slots 0, 1, 2 map to physical targets 5, 0, 7; every other
    // logical slot is a hole.
    RenderTargetPlan plan = plan_from_registers(rt_control(3, {5, 0, 7, 0, 0, 0, 0, 0}),
        {0xd5, 0, 0, 0, 0, 0xd5, 0, 0xd5}, 64, 64);
    if (plan.colors[0] == std::nullopt) return;
    const std::array<std::array<float, 4>, 3> colors{
        std::array<float, 4>{1.0f, 0.0f, 0.0f, 1.0f},
        std::array<float, 4>{0.0f, 1.0f, 0.0f, 1.0f},
        std::array<float, 4>{0.0f, 0.0f, 1.0f, 1.0f}};
    metal::NativeRenderTargets targets;
    std::vector<ColorTransfer> transfers(3);
    for (std::size_t logical = 0; logical < 3; ++logical) {
        plan.colors[logical]->load = AttachmentLoad::Clear;
        targets.colors[logical].texture =
            make_texture(device, MTLPixelFormatRGBA8Unorm, 64, 64);
        transfers[logical].texture = targets.colors[logical].texture;
    }
    const metal::RenderPassClearValues values = slot_colors(colors);
    const auto built = metal::make_render_pass(plan, targets, &values);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor *>(built));
    if (!std::holds_alternative<MTL4RenderPassDescriptor *>(built)) return;
    const auto blocks =
        run_color_round_trip(device, std::get<MTL4RenderPassDescriptor *>(built), transfers);
    CHECK(blocks.size() == 3);
    if (blocks.size() != 3) return;
    // Each logical slot's own texture carries its own clear color, so the
    // permutation routed the writes without compacting holes.
    const std::array<std::uint32_t, 3> expected{0xff0000ffu, 0xff00ff00u, 0xffff0000u};
    for (std::size_t logical = 0; logical < 3; ++logical) {
        expect_uniform(blocks[logical], 64 * 64,
                       [expected, logical](std::size_t) { return expected[logical]; },
                       "permuted slot");
    }
}

void test_typed_formats(id<MTLDevice> device)
{
    // Five typed formats share one pass: normalized 8-bit, packed 10-10-10,
    // 16-bit float, 32-bit float and a 32-bit unsigned integer target.
    RenderTargetPlan plan = plan_from_registers(
        rt_control(5, {0, 1, 2, 3, 4, 0, 0, 0}),
        {0xd5, 0xd1, 0xca, 0xe5, 0xe4, 0, 0, 0}, 32, 32);
    if (plan.colors[0] == std::nullopt) return;
    const std::array<MTLPixelFormat, 5> formats{
        MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGB10A2Unorm,
        MTLPixelFormatRGBA16Float, MTLPixelFormatR32Float, MTLPixelFormatR32Uint};
    const std::array<std::array<float, 4>, 5> colors{
        std::array<float, 4>{0.25f, 0.5f, 0.75f, 1.0f},
        std::array<float, 4>{1.0f, 0.5f, 0.25f, 0.25f},
        std::array<float, 4>{0.5f, 0.25f, 1.0f, 0.75f},
        std::array<float, 4>{0.125f, 0.0f, 0.0f, 0.0f},
        std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}};
    metal::RenderPassClearValues values = slot_colors(colors);
    // The integer target takes the raw cleared words as exact doubles.
    values.color[4] = {42.0, 0.0, 0.0, 0.0};
    metal::NativeRenderTargets targets;
    std::vector<ColorTransfer> transfers(5);
    for (std::size_t slot = 0; slot < 5; ++slot) {
        plan.colors[slot]->load = AttachmentLoad::Clear;
        targets.colors[slot].texture = make_texture(device, formats[slot], 32, 32);
        transfers[slot].texture = targets.colors[slot].texture;
    }
    const auto built = metal::make_render_pass(plan, targets, &values);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor *>(built));
    if (!std::holds_alternative<MTL4RenderPassDescriptor *>(built)) return;
    const auto blocks =
        run_color_round_trip(device, std::get<MTL4RenderPassDescriptor *>(built), transfers);
    CHECK(blocks.size() == 5);
    if (blocks.size() != 5) return;
    // Expected words: 8-bit unorm, 10-10-10 packing (R low bits, A top bits),
    // two f16 words per pixel, f32 bit pattern and the raw u32 word.
    const std::uint16_t f16_r = 0x3800, f16_g = 0x3400, f16_b = 0x3c00, f16_a = 0x3a00;
    const std::uint32_t f16_rg = std::uint32_t(f16_r) | std::uint32_t(f16_g) << 16;
    const std::uint32_t f16_ba = std::uint32_t(f16_b) | std::uint32_t(f16_a) << 16;
    expect_uniform(blocks[0], 32 * 32, [](std::size_t) { return 0xffbf8040u; },
                   "rgba8 slot");
    // 0.5*1023 rounds to 512 and 0.25*1023 to 256 (measured driver behavior).
    expect_uniform(blocks[1], 32 * 32, [](std::size_t) { return 0x500803ffu; },
                   "rgb10a2 slot");
    for (std::size_t word_index = 0; word_index < 2 * 32 * 32; ++word_index) {
        const std::uint32_t actual = read_word(blocks[2], word_index);
        const std::uint32_t want =
            word_index % 2 == 0 ? f16_rg : f16_ba;
        if (actual != want) {
            std::fprintf(stderr, "FAIL: rgba16f slot word %zu wrote %08x expected %08x\n",
                         word_index, actual, want);
            ++failures;
            return;
        }
    }
    expect_uniform(blocks[3], 32 * 32,
                   [](std::size_t) { return std::bit_cast<std::uint32_t>(0.125f); },
                   "r32f slot");
    expect_uniform(blocks[4], 32 * 32, [](std::size_t) { return 42u; }, "r32u slot");
}

// Depth/stencil round trip: prefill, run the planned pass, then read the
// depth aspect into an R32Float texture and the stencil aspect into an
// X32_Stencil8 texture through the dedicated blit options.
struct DepthReadback {
    bool ok = false;
    std::vector<std::byte> depth;
    std::vector<std::byte> stencil;
};

DepthReadback run_depth_round_trip(id<MTLDevice> device,
                                   MTL4RenderPassDescriptor * pass,
                                   id<MTLTexture> depth_texture,
                                   std::uint32_t width, std::uint32_t height,
                                   std::uint32_t prefill_word,
                                   bool read_stencil,
                                   std::vector<std::byte> prefill_bytes = {})
{
    DepthReadback result;
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    auto* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 16;
    NSError* error = nil;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    NSMutableArray* retained = [NSMutableArray arrayWithObjects:
        allocator, commands, residency, nil];
    if (!allocator || !commands || !residency) {
        std::fprintf(stderr, "FAIL: depth round trip allocation failed\n");
        ++failures;
        return result;
    }
    [retained addObject:depth_texture];
    [residency addAllocation:depth_texture];
    const std::vector<std::byte> prefill = prefill_bytes.empty()
        ? fill_words(prefill_word, std::size_t(width) * height)
        : std::move(prefill_bytes);
    id<MTLBuffer> staging = [device newBufferWithBytes:prefill.data()
        length:prefill.size() options:MTLResourceStorageModeShared];
    // The aspects blit straight into shared buffers: the depth aspect as
    // R32Float words and the stencil aspect as words with the stencil value.
    const std::size_t depth_row_bytes = std::size_t(width) * 4;
    id<MTLBuffer> depth_output = [device newBufferWithLength:depth_row_bytes * height
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> stencil_output = nil;
    if (read_stencil) {
        // The stencil aspect reads back as one byte per pixel.
        stencil_output = [device newBufferWithLength:std::size_t(width) * height
            options:MTLResourceStorageModeShared];
    }
    if (!staging || !depth_output || (read_stencil && !stencil_output)) {
        std::fprintf(stderr, "FAIL: depth round trip resource allocation failed\n");
        ++failures;
        return result;
    }
    [retained addObjectsFromArray:@[staging, depth_output]];
    [residency addAllocation:staging];
    [residency addAllocation:depth_output];
    if (read_stencil) {
        [retained addObject:stencil_output];
        [residency addAllocation:stencil_output];
    }
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> upload = [commands computeCommandEncoder];
    if (!upload) {
        std::fprintf(stderr, "FAIL: depth upload encoder creation failed\n");
        ++failures;
        return result;
    }
    [upload copyFromBuffer:staging sourceOffset:0
        sourceBytesPerRow:prefill.size() / height
        sourceBytesPerImage:prefill.size()
        sourceSize:MTLSizeMake(width, height, 1)
        toTexture:depth_texture destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [upload endEncoding];
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) {
        std::fprintf(stderr, "FAIL: depth render encoder creation failed\n");
        ++failures;
        return result;
    }
    [render barrierAfterQueueStages:MTLStageBlit beforeStages:MTLStageAll
        visibilityOptions:MTL4VisibilityOptionDevice];
    [render endEncoding];
    id<MTL4ComputeCommandEncoder> download = [commands computeCommandEncoder];
    if (!download) {
        std::fprintf(stderr, "FAIL: depth readback encoder creation failed\n");
        ++failures;
        return result;
    }
    [download barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
        visibilityOptions:MTL4VisibilityOptionDevice];
    [download copyFromTexture:depth_texture sourceSlice:0 sourceLevel:0
        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
        toBuffer:depth_output destinationOffset:0
        destinationBytesPerRow:depth_row_bytes destinationBytesPerImage:0];
    if (read_stencil) {
        [download copyFromTexture:depth_texture sourceSlice:0 sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
            toBuffer:stencil_output destinationOffset:0
            destinationBytesPerRow:width destinationBytesPerImage:0
            options:MTLBlitOptionStencilFromDepthStencil];
    }
    [download endEncoding];
    if (!commit_and_wait(device, retained, residency, commands)) return result;
    const auto* depth_bytes = static_cast<const std::byte*>(depth_output.contents);
    result.depth.assign(depth_bytes, depth_bytes + depth_row_bytes * height);
    if (read_stencil) {
        const auto* stencil_bytes = static_cast<const std::byte*>(stencil_output.contents);
        result.stencil.assign(stencil_bytes,
                              stencil_bytes + std::size_t(width) * height);
    }
    result.ok = true;
    return result;
}

void test_depth_only_pass(id<MTLDevice> device)
{
    RenderTargetPlan plan = depth_only_plan(0x0a, 64, 32);  // Zf32 -> D32Float
    if (!plan.depth) return;
    plan.depth->load = AttachmentLoad::Clear;
    metal::NativeRenderTargets targets;
    targets.depth.texture = make_texture(device, MTLPixelFormatDepth32Float, 64, 32);
    metal::RenderPassClearValues values{};
    values.depth = 0.5;
    const auto built = metal::make_render_pass(plan, targets, &values);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor *>(built));
    if (!std::holds_alternative<MTL4RenderPassDescriptor *>(built)) return;
    const DepthReadback readback =
        run_depth_round_trip(device, std::get<MTL4RenderPassDescriptor *>(built),
                             targets.depth.texture, 64, 32,
                             std::bit_cast<std::uint32_t>(0.25f), false);
    CHECK(readback.ok);
    if (!readback.ok) return;
    expect_uniform(readback.depth, 64 * 32,
                   [](std::size_t) { return std::bit_cast<std::uint32_t>(0.5f); },
                   "depth-only pass");
}

void test_attachmentless_descriptor()
{
    RenderTargetPlan plan;
    plan.area = {0, 0, 192, 192};
    metal::NativeRenderTargets targets;
    auto built = metal::make_render_pass(plan, targets, nullptr);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor*>(built));
    if (!std::holds_alternative<MTL4RenderPassDescriptor*>(built)) return;
    auto* pass = std::get<MTL4RenderPassDescriptor*>(built);
    CHECK(pass.renderTargetWidth == 192 && pass.renderTargetHeight == 192);
    CHECK(pass.defaultRasterSampleCount == 1);
    CHECK(pass.depthAttachment.texture == nil && pass.stencilAttachment.texture == nil);
    for (NSUInteger slot = 0; slot != 8; ++slot) CHECK(pass.colorAttachments[slot].texture == nil);

    plan.area = {0, 0, 65535, 32768};
    built = metal::make_render_pass(plan, targets, nullptr);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor*>(built));
    if (!std::holds_alternative<MTL4RenderPassDescriptor*>(built)) return;
    pass = std::get<MTL4RenderPassDescriptor*>(built);
    const auto bounded = metal::render_pass_area(plan);
    CHECK(bounded.width == 16384 && bounded.height == 16384);
    CHECK(pass.renderTargetWidth == bounded.width && pass.renderTargetHeight == bounded.height);

    plan.samples = 2;
    built = metal::make_render_pass(plan, targets, nullptr);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor*>(built));
    if (std::holds_alternative<MTL4RenderPassDescriptor*>(built))
        CHECK(std::get<MTL4RenderPassDescriptor*>(built).defaultRasterSampleCount == 2);
}

void test_depth_stencil_pass(id<MTLDevice> device)
{
    RenderTargetPlan plan = depth_only_plan(0x19, 64, 32);  // Zf32X24S8
    if (!plan.depth) return;
    CHECK(plan.depth.has_value() && plan.stencil.has_value());
    plan.depth->load = AttachmentLoad::Clear;
    plan.stencil->load = AttachmentLoad::Clear;
    metal::NativeRenderTargets targets;
    targets.depth.texture =
        make_texture(device, MTLPixelFormatDepth32Float_Stencil8, 64, 32);
    targets.stencil.texture = targets.depth.texture;
    metal::RenderPassClearValues values{};
    values.depth = 0.75;
    values.stencil = 0x3c;
    const auto built = metal::make_render_pass(plan, targets, &values);
    CHECK(std::holds_alternative<MTL4RenderPassDescriptor *>(built));
    if (!std::holds_alternative<MTL4RenderPassDescriptor *>(built)) return;
    // Prefill: depth 0.25 with stencil byte 0xa7; the packed format stores
    // eight bytes per pixel.
    std::vector<std::byte> prefill =
        fill_words(std::bit_cast<std::uint32_t>(0.25f), 64 * 32 * 2);
    for (std::size_t pixel = 0; pixel < 64 * 32; ++pixel) {
        prefill[pixel * 8 + 4] = std::byte{0xa7};
    }
    const DepthReadback readback =
        run_depth_round_trip(device, std::get<MTL4RenderPassDescriptor *>(built),
                             targets.depth.texture, 64, 32, 0, true,
                             std::move(prefill));
    CHECK(readback.ok);
    if (!readback.ok) return;
    expect_uniform(readback.depth, 64 * 32,
                   [](std::size_t) { return std::bit_cast<std::uint32_t>(0.75f); },
                   "depth/stencil depth aspect");
    // The stencil readback carries one byte per pixel.
    for (std::size_t pixel = 0; pixel < 64 * 32; ++pixel) {
        if (std::to_integer<unsigned>(readback.stencil[pixel]) != 0x3c) {
            std::fprintf(stderr, "FAIL: depth/stencil stencil aspect pixel %zu wrote %02x\n",
                         pixel, std::to_integer<unsigned>(readback.stencil[pixel]));
            ++failures;
            return;
        }
    }
}

}  // namespace

int main() {
    id<MTLDevice> device = ensure_device();
    test_eight_distinct_outputs(device);
    test_sparse_and_permuted_attachments(device);
    test_typed_formats(device);
    test_depth_only_pass(device);
    test_depth_stencil_pass(device);
        test_attachmentless_descriptor();
    return failures == 0 ? 0 : 1;
}
