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
#include <clear_encoder.h>
#include <pass_counters.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
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

constexpr std::uint32_t pack_halves(std::uint32_t low, std::uint32_t high) noexcept
{
    return low | (high << 16);
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

id<MTLTexture> make_shared_texture(id<MTLDevice> device, MTLPixelFormat format,
                                   NSUInteger width, NSUInteger height)
{
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
        width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite |
                       MTLTextureUsageShaderRead;
    return [device newTextureWithDescriptor:descriptor];
}

id<MTLTexture> make_shared_array_texture(id<MTLDevice> device, MTLPixelFormat format,
                                         NSUInteger width, NSUInteger height,
                                         NSUInteger layers)
{
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
        width:width height:height mipmapped:NO];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.arrayLength = layers;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite |
                       MTLTextureUsageShaderRead;
    return [device newTextureWithDescriptor:descriptor];
}

void prefill_slice(id<MTLTexture> texture, NSUInteger slice,
                   const std::vector<std::byte>& bytes, std::size_t bytes_per_pixel = 4)
{
    [texture replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
        mipmapLevel:0 slice:slice withBytes:bytes.data()
        bytesPerRow:std::size_t(texture.width) * bytes_per_pixel
        bytesPerImage:std::size_t(texture.width) * texture.height * bytes_per_pixel];
}

std::vector<std::byte> read_color_slice(id<MTLTexture> texture, NSUInteger slice)
{
    std::vector<std::byte> bytes(std::size_t(texture.width) * texture.height * 4);
    [texture getBytes:bytes.data()
        bytesPerRow:std::size_t(texture.width) * 4
        bytesPerImage:std::size_t(texture.width) * texture.height * 4
        fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
        mipmapLevel:0 slice:slice];
    return bytes;
}

void prefill(id<MTLTexture> texture, const std::vector<std::byte>& bytes,
             std::size_t bytes_per_pixel = 4)
{
    [texture replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
        mipmapLevel:0 withBytes:bytes.data()
        bytesPerRow:std::size_t(texture.width) * bytes_per_pixel];
}

std::vector<std::byte> fill_words(std::uint32_t word, std::size_t count)
{
    std::vector<std::byte> bytes(count * 4);
    for (std::size_t index = 0; index < count; ++index) {
        std::memcpy(bytes.data() + index * 4, &word, 4);
    }
    return bytes;
}

std::uint32_t read_word(const std::vector<std::byte>& block, std::size_t word_index)
{
    std::uint32_t word = 0;
    std::memcpy(&word, block.data() + word_index * 4, 4);
    return word;
}

// Commits one encoded command buffer and waits for its feedback. All
// submissions in one test share a queue: separate Metal queues give no
// cross-queue ordering, so a readback on another queue would race the clear.
bool submit_and_wait(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                     id<MTL4CommandBuffer> commands,
                     NSMutableArray* retained)
{
    static_cast<void>(device);
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
        std::fprintf(stderr, "FAIL: clear submission timed out\n");
        ++failures;
        return false;
    }
    if (execution_error) {
        std::fprintf(stderr, "FAIL: clear submission failed: %s\n",
                     execution_error.localizedDescription.UTF8String);
        ++failures;
        return false;
    }
    return true;
}

// Encodes the plan, submits it, and returns true on clean completion.
bool run_clear(id<MTLDevice> device, id<MTL4CommandQueue> queue,
               const ClearPlan& plan, const metal::ClearBindings& bindings)
{
    // Reuse pixel-preservation checks with sampling enabled by the caller.
    metal::PassCounters counters;
    auto measured_bindings = bindings;
    if (metal::PassCounters::enabled()) measured_bindings.pass_counters = &counters;
    const auto encoded = metal::encode_clear(device, plan, measured_bindings);
    if (!std::holds_alternative<metal::EncodedClear>(encoded)) {
        const auto& error = std::get<armigo::graphics::RendererError>(encoded);
        const auto command = error.command();
        std::fprintf(stderr, "FAIL: encode_clear rejected: %.*s\n",
                     int(command.detail.size()), command.detail.data());
        ++failures;
        return false;
    }
    const auto value = std::get<metal::EncodedClear>(encoded);
    const bool completed = submit_and_wait(device, queue, value.commands, value.retained);
    if (completed && metal::PassCounters::enabled()) {
        CHECK(counters.passes.size() == 1);
        CHECK(!counters.passes.empty() && counters.passes[0].ended);
        counters.report(device, 1, true, -1);
    }
    return completed;
}

// Reads the depth aspect of a depth texture into shared bytes through the
// depth blit option.
std::vector<std::byte> read_depth(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                                  id<MTLTexture> texture,
                                  std::uint32_t width, std::uint32_t height,
                                  NSUInteger slice = 0)
{
    id<MTLBuffer> output = [device newBufferWithLength:std::size_t(width) * height * 4
        options:MTLResourceStorageModeShared];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    auto* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 8;
    NSError* error = nil;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    NSMutableArray* retained = [NSMutableArray arrayWithObjects:
        allocator, commands, residency, output, nil];
    if (!output || !allocator || !commands || !residency) {
        std::fprintf(stderr, "FAIL: depth readback allocation failed\n");
        ++failures;
        return {};
    }
    [residency addAllocation:texture];
    [residency addAllocation:output];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    if (!copy) {
        std::fprintf(stderr, "FAIL: depth readback encoder creation failed\n");
        ++failures;
        return {};
    }
    const NSUInteger bytes_per_image =
        texture.textureType == MTLTextureType2DArray
            ? NSUInteger(width) * height * 4 : 0;
    [copy barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
              visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:texture sourceSlice:slice sourceLevel:0
        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
        toBuffer:output destinationOffset:0
        destinationBytesPerRow:std::size_t(width) * 4
        destinationBytesPerImage:bytes_per_image];
    [copy endEncoding];
    [commands endCommandBuffer];
    if (!submit_and_wait(device, queue, commands, retained)) return {};
    const auto* bytes = static_cast<const std::byte*>(output.contents);
    return {bytes, bytes + std::size_t(width) * height * 4};
}

// Reads the stencil aspect into shared bytes; each word carries the stencil
// value in its low byte.
std::vector<std::byte> read_stencil(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                                    id<MTLTexture> texture,
                                    std::uint32_t width, std::uint32_t height)
{
    // The stencil aspect reads back as one byte per pixel.
    id<MTLBuffer> output = [device newBufferWithLength:std::size_t(width) * height
        options:MTLResourceStorageModeShared];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    auto* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 8;
    NSError* error = nil;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    NSMutableArray* retained = [NSMutableArray arrayWithObjects:
        allocator, commands, residency, output, nil];
    if (!output || !allocator || !commands || !residency) {
        std::fprintf(stderr, "FAIL: stencil readback allocation failed\n");
        ++failures;
        return {};
    }
    [residency addAllocation:texture];
    [residency addAllocation:output];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    if (!copy) {
        std::fprintf(stderr, "FAIL: stencil readback encoder creation failed\n");
        ++failures;
        return {};
    }
    [copy barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
              visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:texture sourceSlice:0 sourceLevel:0
        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
        toBuffer:output destinationOffset:0
        destinationBytesPerRow:width destinationBytesPerImage:0
        options:MTLBlitOptionStencilFromDepthStencil];
    [copy endEncoding];
    [commands endCommandBuffer];
    if (!submit_and_wait(device, queue, commands, retained)) return {};
    const auto* bytes = static_cast<const std::byte*>(output.contents);
    return {bytes, bytes + std::size_t(width) * height};
}

// A color clear stop over a pitch RGBA8 target of 64x32 texels; the target's
// array-mode register carries the programmed layer count.
RenderStop make_color_clear_stop(std::uint32_t argument, std::uint32_t clear_control,
                                 std::uint32_t array_layers = 1)
{
    RenderStop stop{};
    stop.kind = RenderKind::Clear;
    stop.engine = engine_class::threed;
    stop.registers[0x674] = argument;
    stop.registers[0xdf] = 1;
    stop.registers[0x574] = 0;
    stop.registers[0x43e] = clear_control;
    stop.registers[0x3fd] = pack_halves(0, 64);
    stop.registers[0x3fe] = pack_halves(0, 32);
    constexpr std::size_t base = 0x200 + 2 * 0x10;
    stop.registers[base + 2] = 256;  // pitch stride for 64 four-byte texels
    stop.registers[base + 3] = 32;
    stop.registers[base + 4] = 0xd5;
    stop.registers[base + 5] = 0x1000;
    stop.registers[base + 6] = array_layers;
    stop.color_targets[2].address = 0x1000000;
    stop.color_targets[2].width = 256;
    stop.color_targets[2].height = 32;
    stop.color_targets[2].format = 0xd5;
    stop.color_targets[2].tile_mode = 0x1000;
    stop.color_targets[2].depth = array_layers;
    const std::array<float, 4> color{0.0f, 0.0f, 1.0f, 0.0f};
    for (std::size_t index = 0; index < 4; ++index) {
        stop.registers[0x360 + index] = std::bit_cast<std::uint32_t>(color[index]);
    }
    return stop;
}

ClearPlan require_color_clear_plan(RenderStop stop)
{
    auto planned = plan_render_clear(stop);
    if (!std::holds_alternative<ClearPlan>(planned)) {
        std::fprintf(stderr, "FAIL: color clear plan rejected\n");
        ++failures;
        return ClearPlan{};
    }
    return std::get<ClearPlan>(planned);
}

ClearPlan color_clear_plan(std::uint32_t argument, std::uint32_t clear_control)
{
    return require_color_clear_plan(make_color_clear_stop(argument, clear_control));
}

// A depth/stencil clear plan over a 64x32 block-linear target; the array-mode
// register carries the programmed layer count (bit 16 disables array mode).
ClearPlan depth_clear_plan(std::uint32_t argument, std::uint32_t clear_control,
                           std::uint32_t format, bool scissored,
                           std::uint32_t array_layers = 1)
{
    RenderStop stop{};
    stop.kind = RenderKind::Clear;
    stop.engine = engine_class::threed;
    stop.registers[0x674] = argument;
    stop.registers[0xdf] = 1;
    stop.registers[0x574] = 0;
    stop.registers[0x43e] = clear_control | (scissored ? 0x100u : 0u);
    stop.registers[0x3fd] = pack_halves(0, 64);
    stop.registers[0x3fe] = pack_halves(0, 32);
    if (scissored) {
        stop.registers[0x380] = 1;
        stop.registers[0x381] = pack_halves(8, 40);
        stop.registers[0x382] = pack_halves(4, 24);
    }
    stop.registers[0x3f9] = 0x2000000;
    stop.registers[0x3fa] = format;
    stop.registers[0x3fb] = 0;
    stop.registers[0x3fc] = 0x1000;
    stop.registers[0x48a] = 64;
    stop.registers[0x48b] = 32;
    stop.registers[0x48c] = (array_layers > 1 ? 0u : (1u << 16)) | array_layers;
    stop.registers[0x54e] = 1;
    stop.depth.address = 0x2000000;
    stop.depth.format = format;
    stop.depth.tile_mode = 0;
    stop.depth.width = 64;
    stop.depth.height = 32;
    stop.depth.enabled = true;
    stop.registers[0x364] = std::bit_cast<std::uint32_t>(0.75f);
    stop.registers[0x368] = 0x5a;
    auto planned = plan_render_clear(stop);
    if (!std::holds_alternative<ClearPlan>(planned)) {
        std::fprintf(stderr, "FAIL: depth clear plan rejected\n");
        ++failures;
        return ClearPlan{};
    }
    return std::get<ClearPlan>(planned);
}

void expect_uniform(const std::vector<std::byte>& block, std::size_t word_count,
                    std::uint32_t expected, const char* what)
{
    for (std::size_t word_index = 0; word_index < word_count; ++word_index) {
        if (read_word(block, word_index) != expected) {
            std::fprintf(stderr, "FAIL: %s word %zu wrote %08x expected %08x\n",
                         what, word_index, read_word(block, word_index), expected);
            ++failures;
            return;
        }
    }
}

void test_full_color_clear_folds(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    ClearPlan plan = color_clear_plan(0xf << 2 | 2 << 6, 0);
    CHECK(plan.fold_into_load);
    id<MTLTexture> texture = make_shared_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32);
    prefill(texture, fill_words(sentinel_word, 64 * 32));
    metal::ClearBindings bindings;
    bindings.color = texture;
    CHECK(run_clear(device, queue, plan, bindings));
    std::vector<std::byte> bytes(std::size_t(64) * 32 * 4);
    [texture getBytes:bytes.data() bytesPerRow:64 * 4
        fromRegion:MTLRegionMake2D(0, 0, 64, 32) mipmapLevel:0];
    // 0.0, 0.0, 1.0, 0.0 -> blue full: byte 2 of each RGBA8 word.
    expect_uniform(bytes, 64 * 32, 0x00ff0000u, "folded clear");
}

void test_scissored_clear_preserves_untouched_pixels(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    RenderStop stop = make_color_clear_stop(0xf << 2 | 2 << 6, 0x100);
    stop.registers[0x380] = 1;
    stop.registers[0x381] = pack_halves(13, 26);
    stop.registers[0x382] = pack_halves(20, 32);
    ClearPlan plan = require_color_clear_plan(std::move(stop));
    CHECK(plan.region == (RenderArea{13, 20, 13, 12}));
    CHECK(!plan.fold_into_load);
    id<MTLTexture> texture = make_shared_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32);
    prefill(texture, fill_words(sentinel_word, 64 * 32));
    metal::ClearBindings bindings;
    bindings.color = texture;
    CHECK(run_clear(device, queue, plan, bindings));
    std::vector<std::byte> bytes(std::size_t(64) * 32 * 4);
    [texture getBytes:bytes.data() bytesPerRow:64 * 4
        fromRegion:MTLRegionMake2D(0, 0, 64, 32) mipmapLevel:0];
    for (std::uint32_t y = 0; y < 32; ++y) {
        for (std::uint32_t x = 0; x < 64; ++x) {
            const bool cleared = x >= 13 && x < 26 && y >= 20;
            const std::uint32_t actual =
                read_word(bytes, std::size_t(y) * 64 + x);
            const std::uint32_t expected = cleared ? 0x00ff0000u : sentinel_word;
            if (actual != expected) {
                std::fprintf(stderr, "FAIL: scissored pixel %u,%u wrote %08x expected %08x\n",
                             x, y, actual, expected);
                ++failures;
                return;
            }
        }
    }
}

void test_masked_clear_preserves_channels(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    // Mask 0x5 names R and B only; G and A keep their prior bytes.
    ClearPlan plan = color_clear_plan(0x5 << 2 | 2 << 6, 0);
    CHECK(!plan.fold_into_load);
    CHECK(plan.color_mask == 0x5);
    id<MTLTexture> texture = make_shared_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32);
    const std::uint32_t pattern = 0x44332211u;
    prefill(texture, fill_words(pattern, 64 * 32));
    metal::ClearBindings bindings;
    bindings.color = texture;
    CHECK(run_clear(device, queue, plan, bindings));
    std::vector<std::byte> bytes(std::size_t(64) * 32 * 4);
    [texture getBytes:bytes.data() bytesPerRow:64 * 4
        fromRegion:MTLRegionMake2D(0, 0, 64, 32) mipmapLevel:0];
    // R forced to 0, B forced to 0xff, G and A preserved.
    expect_uniform(bytes, 64 * 32, 0x44ff2200u, "masked clear");
}

void test_scissored_depth_clear(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    ClearPlan plan = depth_clear_plan(1, 0, 0x0a, true);
    CHECK(!plan.fold_into_load);
    CHECK(plan.clear_depth && !plan.clear_stencil);
    id<MTLTexture> texture =
        make_shared_texture(device, MTLPixelFormatDepth32Float, 64, 32);
    prefill(texture, fill_words(std::bit_cast<std::uint32_t>(0.25f), 64 * 32));
    metal::ClearBindings bindings;
    bindings.depth = texture;
    CHECK(run_clear(device, queue, plan, bindings));
    const auto depth = read_depth(device, queue, texture, 64, 32);
    CHECK(depth.size() == std::size_t(64) * 32 * 4);
    if (depth.size() != std::size_t(64) * 32 * 4) return;
    for (std::uint32_t y = 0; y < 32; ++y) {
        for (std::uint32_t x = 0; x < 64; ++x) {
            const bool cleared = x >= 8 && x < 40 && y >= 4 && y < 24;
            const float value = std::bit_cast<float>(
                read_word(depth, std::size_t(y) * 64 + x));
            const float expected = cleared ? 0.75f : 0.25f;
            if (value != expected) {
                std::fprintf(stderr, "FAIL: scissored depth %u,%u wrote %g expected %g\n",
                             x, y, value, expected);
                ++failures;
                return;
            }
        }
    }
}

void test_depth_stencil_fold_and_stencil_mask(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    // A full packed clear folds into the load actions.
    ClearPlan fold = depth_clear_plan(3, 0, 0x19, false);
    CHECK(fold.fold_into_load);
    id<MTLTexture> texture =
        make_shared_texture(device, MTLPixelFormatDepth32Float_Stencil8, 64, 32);
    // Eight bytes per pixel: depth 0.25, stencil byte 0xa7.
    std::vector<std::byte> packed_prefill =
        fill_words(std::bit_cast<std::uint32_t>(0.25f), 64 * 32 * 2);
    for (std::size_t pixel = 0; pixel < 64 * 32; ++pixel) {
        packed_prefill[pixel * 8 + 4] = std::byte{0xa7};
    }
    prefill(texture, packed_prefill, 8);
    metal::ClearBindings bindings;
    bindings.depth = texture;
    bindings.stencil = texture;
    CHECK(run_clear(device, queue, fold, bindings));
    const auto depth = read_depth(device, queue, texture, 64, 32);
    CHECK(depth.size() == std::size_t(64) * 32 * 4);
    if (depth.size() == std::size_t(64) * 32 * 4) {
        expect_uniform(depth, 64 * 32, std::bit_cast<std::uint32_t>(0.75f), "packed depth");
    }
    const auto stencil = read_stencil(device, queue, texture, 64, 32);
    CHECK(stencil.size() == std::size_t(64) * 32);
    if (stencil.size() == std::size_t(64) * 32) {
        for (std::size_t pixel = 0; pixel < 64 * 32; ++pixel) {
            if (std::to_integer<unsigned>(stencil[pixel]) != 0x5a) {
                std::fprintf(stderr, "FAIL: packed stencil pixel %zu wrote %02x\n",
                             pixel, std::to_integer<unsigned>(stencil[pixel]));
                ++failures;
                return;
            }
        }
    }

    // A partial stencil write mask preserves the masked bits: the front mask
    // 0x0f leaves the high nibble of the prior stencil value.
    ClearPlan masked = depth_clear_plan(2, 0x1, 0x19, false);
    CHECK(!masked.fold_into_load);
    masked.stencil_write_mask = 0x0f;
    id<MTLTexture> masked_texture =
        make_shared_texture(device, MTLPixelFormatDepth32Float_Stencil8, 64, 32);
    metal::ClearBindings masked_bindings;
    masked_bindings.stencil = masked_texture;
    // Apple silicon splits the packed format into separate planes, so the
    // seed value comes from a prior full clear instead of CPU prefill: the
    // seed pass writes stencil 0xa7 (the plan's own value) everywhere.
    ClearPlan seed = depth_clear_plan(3, 0, 0x19, false);
    seed.depth_value = 0.25;
    seed.stencil_value = 0xa7;
    CHECK(run_clear(device, queue, seed, {.color = nil, .depth = masked_texture,
                                   .stencil = masked_texture}));
    CHECK(run_clear(device, queue, masked, masked_bindings));
    const auto masked_stencil = read_stencil(device, queue, masked_texture, 64, 32);
    CHECK(masked_stencil.size() == std::size_t(64) * 32);
    if (masked_stencil.size() == std::size_t(64) * 32) {
        for (std::size_t pixel = 0; pixel < 64 * 32; ++pixel) {
            const unsigned actual = std::to_integer<unsigned>(masked_stencil[pixel]);
            if (actual != 0xaa) {
                std::fprintf(stderr, "FAIL: masked stencil pixel %zu wrote %02x expected aa\n",
                             pixel, actual);
                ++failures;
                return;
            }
        }
    }
}

void test_clear_rejections_fail_closed(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    // A plan whose region exceeds the texture is rejected before encoding.
    ClearPlan plan = color_clear_plan(0xf << 2 | 2 << 6, 0);
    plan.region = RenderArea{0, 0, 65, 32};
    metal::ClearBindings bindings;
    bindings.color = make_shared_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32);
    // Reuse pixel-preservation checks with sampling enabled by the caller.
    metal::PassCounters counters;
    auto measured_bindings = bindings;
    if (metal::PassCounters::enabled()) measured_bindings.pass_counters = &counters;
    const auto encoded = metal::encode_clear(device, plan, measured_bindings);
    CHECK(std::holds_alternative<armigo::graphics::RendererError>(encoded));
    // A pixel-format contradiction is rejected before encoding.
    ClearPlan contradicting = color_clear_plan(0xf << 2 | 2 << 6, 0);
    contradicting.target.texture.format = TextureFormat::R32Float;
    const auto mismatched =
        metal::encode_clear(device, contradicting, bindings);
    CHECK(std::holds_alternative<armigo::graphics::RendererError>(mismatched));
    // A missing aspect binding is rejected before encoding.
    ClearPlan depthless = depth_clear_plan(1, 0, 0x0a, false);
    const auto unbound = metal::encode_clear(device, depthless, bindings);
    CHECK(std::holds_alternative<armigo::graphics::RendererError>(unbound));
}

void test_single_layer_of_depth_array_clears_only_that_layer(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    // Layer field 2 of a four-layer depth array clears exactly that layer.
    ClearPlan plan = depth_clear_plan(1 | 2 << 10, 0, 0x0a, false, 4);
    CHECK(plan.target.layer == 2 && plan.target.layer_count == 1);
    CHECK(plan.target.texture.layers == 4);
    CHECK(!plan.fold_into_load && !plan.fold_layers_into_load);
    id<MTLTexture> texture =
        make_shared_array_texture(device, MTLPixelFormatDepth32Float, 64, 32, 4);
    for (NSUInteger slice = 0; slice < 4; ++slice) {
        prefill_slice(texture, slice,
                      fill_words(std::bit_cast<std::uint32_t>(0.25f), 64 * 32));
    }
    metal::ClearBindings bindings;
    bindings.depth = texture;
    bindings.slice = plan.target.layer;
    CHECK(run_clear(device, queue, plan, bindings));
    for (NSUInteger slice = 0; slice < 4; ++slice) {
        const auto depth = read_depth(device, queue, texture, 64, 32, slice);
        CHECK(depth.size() == std::size_t(64) * 32 * 4);
        if (depth.size() != std::size_t(64) * 32 * 4) return;
        const float expected =
            slice == 2 ? 0.75f : 0.25f;
        for (std::size_t pixel = 0; pixel < std::size_t(64) * 32; ++pixel) {
            const float value =
                std::bit_cast<float>(read_word(depth, pixel));
            if (value != expected) {
                std::fprintf(stderr,
                             "FAIL: depth array slice %zu pixel %zu wrote %g expected %g\n",
                             slice, pixel, value, expected);
                ++failures;
                return;
            }
        }
    }
}

void test_combined_color_and_depth_clear(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    // One argument selects colour target 2 and the depth aspect; the plan
    // keeps both attachments and folds into one pass with both load actions.
    auto make = [&](bool scissored) {
        RenderStop stop = make_color_clear_stop(1 | 0xf << 2 | 2 << 6,
                                                scissored ? 0x100u : 0u);
        stop.registers[0x3f9] = 0x2000000;
        stop.registers[0x3fa] = 0x0a;
        stop.registers[0x3fb] = 0;
        stop.registers[0x3fc] = 0x1000;
        stop.registers[0x48a] = 64;
        stop.registers[0x48b] = 32;
        stop.registers[0x48c] = 1u << 16;
        stop.registers[0x54e] = 1;
        stop.depth.address = 0x2000000;
        stop.depth.format = 0x0a;
        stop.depth.tile_mode = 0;
        stop.depth.width = 64;
        stop.depth.height = 32;
        stop.depth.enabled = true;
        stop.registers[0x364] = std::bit_cast<std::uint32_t>(0.75f);
        if (scissored) {
            stop.registers[0x380] = 1;
            stop.registers[0x381] = pack_halves(8, 40);
            stop.registers[0x382] = pack_halves(4, 24);
        }
        auto planned = plan_render_clear(stop);
        if (!std::holds_alternative<ClearPlan>(planned)) {
            std::fprintf(stderr, "FAIL: combined clear plan rejected\n");
            ++failures;
            return ClearPlan{};
        }
        return std::get<ClearPlan>(planned);
    };
    // Run the scissored combined draw first: the first combined submission
    // exposes order-dependent behaviour, if any.
    ClearPlan scissored = make(true);
    CHECK(!scissored.fold_into_load);
    id<MTLTexture> scissor_color =
        make_shared_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32);
    id<MTLTexture> scissor_depth =
        make_shared_texture(device, MTLPixelFormatDepth32Float, 64, 32);
    prefill(scissor_color, fill_words(sentinel_word, 64 * 32));
    prefill(scissor_depth, fill_words(std::bit_cast<std::uint32_t>(0.25f), 64 * 32));
    metal::ClearBindings scissor_bindings;
    scissor_bindings.color = scissor_color;
    scissor_bindings.depth = scissor_depth;
    CHECK(run_clear(device, queue, scissored, scissor_bindings));
    const auto scissor_color_bytes = read_color_slice(scissor_color, 0);
    const auto scissor_depth_bytes = read_depth(device, queue, scissor_depth, 64, 32);
    if (scissor_color_bytes.size() == std::size_t(64) * 32 * 4 &&
        scissor_depth_bytes.size() == std::size_t(64) * 32 * 4) {
        for (std::uint32_t y = 0; y < 32; ++y) {
            for (std::uint32_t x = 0; x < 64; ++x) {
                const bool cleared = x >= 8 && x < 40 && y >= 4 && y < 24;
                const std::size_t pixel = std::size_t(y) * 64 + x;
                const std::uint32_t color_word = read_word(scissor_color_bytes, pixel);
                const float depth_value =
                    std::bit_cast<float>(read_word(scissor_depth_bytes, pixel));
                const bool color_ok = color_word == (cleared ? 0x00ff0000u : sentinel_word);
                const bool depth_ok = depth_value == (cleared ? 0.75f : 0.25f);
                if (!color_ok || !depth_ok) {
                    std::fprintf(stderr,
                                 "FAIL: scissored combined pixel %u,%u color %08x depth %g\n",
                                 x, y, color_word, depth_value);
                    ++failures;
                    return;
                }
            }
        }
    }

    ClearPlan fold = make(false);
    CHECK(fold.color_mask == 0xf && fold.clear_depth && !fold.clear_stencil);
    CHECK(fold.depth_stencil.has_value());
    // A combined clear is a real single-pass operation, not a load fold.
    CHECK(!fold.fold_into_load);
    id<MTLTexture> color = make_shared_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32);
    id<MTLTexture> depth = make_shared_texture(device, MTLPixelFormatDepth32Float, 64, 32);
    prefill(color, fill_words(sentinel_word, 64 * 32));
    prefill(depth, fill_words(std::bit_cast<std::uint32_t>(0.25f), 64 * 32));
    metal::ClearBindings bindings;
    bindings.color = color;
    bindings.depth = depth;
    CHECK(run_clear(device, queue, fold, bindings));
    std::vector<std::byte> color_bytes(std::size_t(64) * 32 * 4);
    [color getBytes:color_bytes.data() bytesPerRow:64 * 4
        fromRegion:MTLRegionMake2D(0, 0, 64, 32) mipmapLevel:0];
    expect_uniform(color_bytes, 64 * 32, 0x00ff0000u, "combined color");
    const auto depth_bytes = read_depth(device, queue, depth, 64, 32);
    if (depth_bytes.size() == std::size_t(64) * 32 * 4) {
        expect_uniform(depth_bytes, 64 * 32, std::bit_cast<std::uint32_t>(0.75f),
                       "combined depth");

    }

}

void test_layered_color_clear_covers_every_slice(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    // Layer zero clears the whole programmed colour array through the
    // layered pass load action.
    ClearPlan plan;
    {
        RenderStop stop = make_color_clear_stop(0xf << 2 | 2 << 6, 0, 4);
        auto planned = plan_render_clear(stop);
        if (!std::holds_alternative<ClearPlan>(planned)) {
            std::fprintf(stderr, "FAIL: layered color clear plan rejected\n");
            ++failures;
            return;
        }
        plan = std::get<ClearPlan>(planned);
    }
    CHECK(plan.target.layer == 0 && plan.target.layer_count == 4);
    CHECK(plan.target.texture.layers == 4);
    CHECK(!plan.fold_into_load && plan.fold_layers_into_load);
    id<MTLTexture> texture =
        make_shared_array_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32, 4);
    for (NSUInteger slice = 0; slice < 4; ++slice) {
        prefill_slice(texture, slice, fill_words(sentinel_word, 64 * 32));
    }
    metal::ClearBindings bindings;
    bindings.color = texture;
    bindings.slice = plan.target.layer;
    CHECK(run_clear(device, queue, plan, bindings));
    for (NSUInteger slice = 0; slice < 4; ++slice) {
        const auto bytes = read_color_slice(texture, slice);
        expect_uniform(bytes, 64 * 32, 0x00ff0000u, "layered color clear");
    }
}

void test_scissored_layered_clear_preserves_outside_layers(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    // A scissored layer-zero clear of a four-layer colour array draws once
    // per layer inside a layered pass, writing only the scissored region.
    RenderStop stop = make_color_clear_stop(0xf << 2 | 2 << 6, 0x100, 4);
    stop.registers[0x380] = 1;
    stop.registers[0x381] = pack_halves(13, 26);
    stop.registers[0x382] = pack_halves(20, 32);
    auto planned = plan_render_clear(stop);
    if (!std::holds_alternative<ClearPlan>(planned)) {
        std::fprintf(stderr, "FAIL: scissored layered clear plan rejected\n");
        ++failures;
        return;
    }
    const ClearPlan plan = std::get<ClearPlan>(planned);
    CHECK(plan.region == (RenderArea{13, 20, 13, 12}));
    CHECK(plan.target.layer_count == 4);
    CHECK(!plan.fold_into_load && !plan.fold_layers_into_load);
    id<MTLTexture> texture =
        make_shared_array_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32, 4);
    for (NSUInteger slice = 0; slice < 4; ++slice) {
        prefill_slice(texture, slice, fill_words(sentinel_word, 64 * 32));
    }
    metal::ClearBindings bindings;
    bindings.color = texture;
    bindings.slice = plan.target.layer;
    CHECK(run_clear(device, queue, plan, bindings));
    for (NSUInteger slice = 0; slice < 4; ++slice) {
        const auto bytes = read_color_slice(texture, slice);
        if (bytes.size() != std::size_t(64) * 32 * 4) {
            std::fprintf(stderr, "FAIL: layered scissor readback slice %zu\n", slice);
            ++failures;
            return;
        }
        for (std::uint32_t y = 0; y < 32; ++y) {
            for (std::uint32_t x = 0; x < 64; ++x) {
                const bool cleared = x >= 13 && x < 26 && y >= 20;
                const std::uint32_t actual =
                    read_word(bytes, std::size_t(y) * 64 + x);
                const std::uint32_t expected = cleared ? 0x00ff0000u : sentinel_word;
                if (actual != expected) {
                    std::fprintf(stderr,
                                 "FAIL: layered scissor slice %zu pixel %u,%u wrote %08x\n",
                                 slice, x, y, actual);
                    ++failures;
                    return;
                }
            }
        }
    }
}

// A configuration prepared on another thread (a warm-up row) is not compiled
// again by the clear that needs it, threads asking for one configuration
// together compile it once, and a configuration naming no clear function is
// refused. Runs first, before another test compiles the masked configuration.
void test_prepared_clear_pipelines(id<MTLDevice> device, id<MTL4CommandQueue> queue)
{
    (void)metal::take_clear_compiles();
    // R and B of RGBA8, as test_masked_clear_preserves_channels clears.
    const metal::ClearPipelineConfig masked{"clear_vertex", "clear_color_frag", MTLPixelFormatRGBA8Unorm,
                                            MTLColorWriteMaskRed | MTLColorWriteMaskBlue,
                                            MTLPixelFormatInvalid, MTLPixelFormatInvalid};
    bool prepared = false;
    std::vector<metal::ClearCompile> compiled;
    std::thread([&] {
        prepared = metal::prepare_clear_pipeline(device, masked);
        compiled = metal::take_clear_compiles();
    }).join();
    CHECK(prepared);
    CHECK(std::any_of(compiled.begin(), compiled.end(), [&](const metal::ClearCompile& compile) {
        return compile.pipeline == masked && !compile.wait;
    }));
    ClearPlan plan = color_clear_plan(0x5 << 2 | 2 << 6, 0);
    CHECK(plan.color_mask == 0x5);
    id<MTLTexture> texture = make_shared_texture(device, MTLPixelFormatRGBA8Unorm, 64, 32);
    metal::ClearBindings bindings;
    bindings.color = texture;
    CHECK(run_clear(device, queue, plan, bindings));
    CHECK(metal::take_clear_compiles().empty());

    const metal::ClearPipelineConfig wide{"clear_vertex", "clear_color_frag", MTLPixelFormatRGBA16Float,
                                          MTLColorWriteMaskGreen, MTLPixelFormatInvalid, MTLPixelFormatInvalid};
    std::atomic<unsigned> compiles{0}, refused{0};
    std::vector<std::thread> threads;
    for (int index = 0; index < 4; ++index)
        threads.emplace_back([&] {
            if (!metal::prepare_clear_pipeline(device, wide)) ++refused;
            for (const auto& compile : metal::take_clear_compiles())
                if (compile.pipeline == wide && !compile.wait) ++compiles;
        });
    for (auto& thread : threads) thread.join();
    CHECK(refused == 0 && compiles == 1);
    auto unknown = wide;
    unknown.fragment = "clear_unknown_frag";
    CHECK(!metal::prepare_clear_pipeline(device, unknown));
    auto odd = wide;
    odd.samples = 3;
    CHECK(!metal::prepare_clear_pipeline(device, odd));

    // A partial clear of a multisampled target builds its pipeline at the
    // target's sample count.
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                         width:64 height:32 mipmapped:NO];
    descriptor.textureType = MTLTextureType2DMultisample;
    descriptor.sampleCount = 4;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    metal::ClearBindings multisampled;
    multisampled.color = [device newTextureWithDescriptor:descriptor];
    CHECK(run_clear(device, queue, plan, multisampled));
    const auto multisampled_compiles = metal::take_clear_compiles();
    CHECK(std::any_of(multisampled_compiles.begin(), multisampled_compiles.end(),
                      [&](const metal::ClearCompile& compile) {
        return compile.pipeline && !compile.wait && compile.pipeline->samples == 4 &&
               compile.pipeline->color == MTLPixelFormatRGBA8Unorm;
    }));
}

}  // namespace

int main() {
    id<MTLDevice> device = ensure_device();
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    test_prepared_clear_pipelines(device, queue);
    test_full_color_clear_folds(device, queue);
    test_scissored_clear_preserves_untouched_pixels(device, queue);
    test_masked_clear_preserves_channels(device, queue);
    test_scissored_depth_clear(device, queue);
    test_depth_stencil_fold_and_stencil_mask(device, queue);
    test_clear_rejections_fail_closed(device, queue);
    test_single_layer_of_depth_array_clears_only_that_layer(device, queue);
    test_combined_color_and_depth_clear(device, queue);
    test_layered_color_clear_covers_every_slice(device, queue);
    test_scissored_layered_clear_preserves_outside_layers(device, queue);
    return failures == 0 ? 0 : 1;
}
