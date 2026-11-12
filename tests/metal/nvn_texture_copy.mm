// ============================================================================
// Armigo Tests - NVN Texture Copy Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Metal/Metal.h>

#include "nvn_resources.h"
#include "nvn_replay.h"
#include "entry_table.h"
#include "commands.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>

namespace {
using namespace armigo;
using namespace armigo::nvn;
namespace replay = platform::metal::nvn_driver;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (false)

template<Entry E> auto load() {
    return reinterpret_cast<typename EntrySignature<E>::type>(implemented_entry(E));
}

class CopyResources final : public replay::ReplayResources {
public:
    static constexpr std::uint64_t base = 0x10000;
    explicit CopyResources(id<MTLDevice> device)
        : memory([device newBufferWithLength:65536 options:MTLResourceStorageModeShared]) {}
    replay::BufferSpan buffer_at(std::uint64_t address, std::uint64_t size) override {
        if (address < base || address - base > memory.length || size > memory.length - (address - base)) return {};
        return {memory, address - base};
    }
    id<MTLTexture> texture(const TextureState& state, const TextureViewState* view) override {
        const auto found = textures.find(state.serial);
        if (found == textures.end()) return nil;
        id<MTLTexture> texture = found->second;
        if (!view || !view->sets_levels) return texture;
        ++level_views;
        return [texture newTextureViewWithPixelFormat:texture.pixelFormat textureType:MTLTextureType2D
            levels:NSMakeRange(view->base_level, view->level_count) slices:NSMakeRange(0, 1)];
    }
    id<MTLTexture> texture_for_id(std::int32_t) override { CHECK(false); return nil; }
    id<MTLSamplerState> sampler_for_id(std::int32_t) override { CHECK(false); return nil; }
    void note_gpu_write(const TextureState&) override { texture_writes.fetch_add(1); }
    void note_gpu_write(std::uint64_t, std::uint64_t) override { buffer_writes.fetch_add(1); }
    id<MTLBuffer> memory;
    std::unordered_map<std::uint64_t, id<MTLTexture>> textures;
    std::atomic<unsigned> texture_writes{0}, buffer_writes{0};
    unsigned level_views = 0;
};

struct Recorder {
    NVNdevice device{};
    NVNcommandBuffer commands{};
    NVNmemoryPool pool{};
    std::array<std::byte, 65536> donated{};
    Recorder() {
        auto& backing = *reinterpret_cast<MemoryPoolState*>(&pool);
        backing.memory = reinterpret_cast<std::uint8_t*>(donated.data());
        backing.size = donated.size();
        CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device));
        load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 0, donated.size());
        load<Entry::nvnCommandBufferAddControlMemory>()(&commands, donated.data(), donated.size());
    }
};

TextureState texture_state(std::uint64_t serial, unsigned width, unsigned height, bool hdr, unsigned levels) {
    TextureState state{};
    state.serial = serial;
    state.description.target = NVN_TEXTURE_TARGET_2D;
    state.description.format = hdr ? NVN_FORMAT_R11G11B10F : NVN_FORMAT_RGBA16F;
    state.description.width = width; state.description.height = height;
    state.description.depth = 1; state.description.levels = levels; state.description.samples = 1;
    return state;
}

NVNtexture texture_object(TextureState& snapshot) {
    auto state = snapshot;
    state.snapshot = &snapshot;
    NVNtexture object{};
    std::memcpy(&object, &state, sizeof state);
    return object;
}

using HalfPixel = std::array<_Float16, 4>;
HalfPixel pattern(unsigned x, unsigned y) {
    return {(_Float16)(x + 10 * y), (_Float16)(float(x) * 0.25f), (_Float16)y, (_Float16)1};
}

// Authored positive powers of two have exact encodings in both packed fields.
std::uint32_t hdr_pattern(unsigned x, unsigned y) {
    constexpr std::array<std::uint32_t, 4> rg{0, 15u << 6, 16u << 6, 17u << 6};
    const auto blue = (15u + (x + y) % 3u) << 5;
    return rg[x % 4] | (rg[y % 4] << 11) | (blue << 22);
}

// Cases: nearest, linear, preserved partial target, packed HDR, selected mips,
// and a GPU clear followed by scaling and a dependent texture-to-buffer copy.
void test_scaled_copy(id<MTLDevice> device, id<MTL4CommandQueue> queue, unsigned mode) {
    const bool hdr = mode == 3, mip = mode == 4, partial = mode == 2, linear = mode == 1;
    const unsigned source_width = mip ? 16 : 8, source_height = mip ? 8 : 4;
    const unsigned target_width = mip ? 8 : partial ? 6 : 4, target_height = mip ? 4 : partial ? 4 : 2;
    const unsigned width = partial ? 6 : 4, height = partial ? 4 : 2;
    CopyResources resources(device);
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
        hdr ? MTLPixelFormatRG11B10Float : MTLPixelFormatRGBA16Float
        width:source_width height:source_height mipmapped:mip];
    descriptor.mipmapLevelCount = mip ? 2 : 1;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    auto source = [device newTextureWithDescriptor:descriptor];
    descriptor.width = target_width; descriptor.height = target_height;
    auto destination = [device newTextureWithDescriptor:descriptor];
    CHECK(source && destination);
    if (!source || !destination) return;
    const HalfPixel sentinel{99, 98, 97, 1};
    for (unsigned level = 0; level != (mip ? 2u : 1u); ++level) {
        const unsigned sw = source_width >> level, sh = source_height >> level;
        const unsigned dw = target_width >> level, dh = target_height >> level;
        if (hdr) {
            std::vector<std::uint32_t> input(sw * sh), output(dw * dh, 0);
            for (unsigned y = 0; y != sh; ++y) for (unsigned x = 0; x != sw; ++x) input[y * sw + x] = hdr_pattern(x, y);
            [source replaceRegion:MTLRegionMake2D(0, 0, sw, sh) mipmapLevel:level withBytes:input.data() bytesPerRow:sw * 4];
            [destination replaceRegion:MTLRegionMake2D(0, 0, dw, dh) mipmapLevel:level withBytes:output.data() bytesPerRow:dw * 4];
        } else {
            std::vector<HalfPixel> input(sw * sh), output(dw * dh, sentinel);
            for (unsigned y = 0; y != sh; ++y) for (unsigned x = 0; x != sw; ++x)
                input[y * sw + x] = mip && !level ? HalfPixel{-50, -50, -50, 1} : pattern(x, y);
            [source replaceRegion:MTLRegionMake2D(0, 0, sw, sh) mipmapLevel:level withBytes:input.data() bytesPerRow:sw * sizeof(HalfPixel)];
            [destination replaceRegion:MTLRegionMake2D(0, 0, dw, dh) mipmapLevel:level withBytes:output.data() bytesPerRow:dw * sizeof(HalfPixel)];
        }
    }
    resources.textures = {{1, source}, {2, destination}};
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto source_state = texture_state(1, source_width, source_height, hdr, mip ? 2 : 1);
    auto target_state = texture_state(2, target_width, target_height, hdr, mip ? 2 : 1);
    auto source_object = texture_object(source_state), target_object = texture_object(target_state);
    TextureViewState level{};
    level.sets_levels = 1; level.base_level = 1; level.level_count = 1;
    NVNtextureView view{};
    std::memcpy(&view, &level, sizeof level);
    const NVNcopyRegion source_region{partial ? 2 : 0, 0, 0, partial ? 4 : 8, 4, 1};
    const NVNcopyRegion target_region{partial ? 1 : 0, partial ? 1 : 0, 0, partial ? 2 : 4, 2, 1};
    const NVNcopyRegion whole_target{0, 0, 0, int(width), int(height), 1};
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    if (mode == 5) {
        const NVNcopyRegion whole_source{0, 0, 0, 8, 4, 1};
        const float color[]{0.125f, 0.25f, 0.5f, 1};
        load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &source_object, nullptr, &whole_source, color, 0xf);
    }
    load<Entry::nvnCommandBufferCopyTextureToTexture>()(&recorder.commands, &source_object, mip ? &view : nullptr,
        &source_region, &target_object, mip ? &view : nullptr, &target_region,
        linear ? NVN_COPY_FLAGS_LINEAR_FILTER_BIT : NVN_COPY_FLAGS_NONE);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, mip ? &view : nullptr,
        &whole_target, CopyResources::base + 0x4000, NVN_COPY_FLAGS_NONE);
    const auto handle = load<Entry::nvnCommandBufferEndRecording>()(&recorder.commands);
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    CHECK(backend->statistics().copies == 2 && backend->statistics().skipped == 0);
    const auto* bytes = static_cast<const std::byte*>(resources.memory.contents) + 0x4000;
    for (unsigned y = 0; y != height; ++y) for (unsigned x = 0; x != width; ++x) {
        if (hdr) {
            const auto* output = reinterpret_cast<const std::uint32_t*>(bytes);
            CHECK(output[y * width + x] == hdr_pattern(2 * x + 1, 2 * y + 1));
        } else {
            HalfPixel expected;
            if (mode == 5) expected = {(_Float16)0.125f, (_Float16)0.25f, (_Float16)0.5f, 1};
            else if (partial && (x < 1 || x >= 3 || y < 1 || y >= 3)) expected = sentinel;
            else if (linear) expected = {(_Float16)(2 * x + 20 * y + 5.5f),
                (_Float16)((2 * x + 0.5f) * 0.25f), (_Float16)(2 * y + 0.5f), 1};
            else expected = pattern(partial ? 2 + 2 * (x - 1) + 1 : 2 * x + 1,
                                    partial ? 2 * (y - 1) + 1 : 2 * y + 1);
            const auto* output = reinterpret_cast<const HalfPixel*>(bytes);
            for (unsigned channel = 0; channel != 4; ++channel)
                CHECK(float(output[y * width + x][channel]) == float(expected[channel]));
        }
    }
    if (mip) {
        HalfPixel unchanged{};
        [destination getBytes:&unchanged bytesPerRow:sizeof(HalfPixel) fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
        CHECK(unchanged == sentinel && resources.level_views >= 3);
    }
    CHECK(resources.texture_writes.load() >= (mode == 5 ? 2u : 1u));
    CHECK(resources.buffer_writes.load() >= 1);
    CHECK(backend->stop_submission_workers(std::chrono::steady_clock::now() + std::chrono::seconds(3)));
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    std::fprintf(stderr, "scaled_copy_case mode=%u completed\n", mode);
}

// Two copies share cached pipeline/argument resources but own distinct arena
// parameters and sampler bindings until the GPU has consumed both records.
void test_copy_parameter_lifetime(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    CopyResources resources(device);
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
        width:8 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    auto source = [device newTextureWithDescriptor:descriptor];
    descriptor.width = 4; descriptor.height = 2;
    auto first = [device newTextureWithDescriptor:descriptor], second = [device newTextureWithDescriptor:descriptor];
    CHECK(source && first && second);
    if (!source || !first || !second) return;
    std::array<HalfPixel, 32> input{};
    for (unsigned y = 0; y != 4; ++y) for (unsigned x = 0; x != 8; ++x) input[y * 8 + x] = pattern(x, y);
    [source replaceRegion:MTLRegionMake2D(0, 0, 8, 4) mipmapLevel:0 withBytes:input.data() bytesPerRow:8 * sizeof(HalfPixel)];
    resources.textures = {{1, source}, {2, first}, {3, second}};
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto source_state = texture_state(1, 8, 4, false, 1);
    auto first_state = texture_state(2, 4, 2, false, 1), second_state = texture_state(3, 4, 2, false, 1);
    auto source_object = texture_object(source_state), first_object = texture_object(first_state), second_object = texture_object(second_state);
    const NVNcopyRegion source_region{0, 0, 0, 8, 4, 1}, other_region{4, 0, 0, 4, 4, 1}, target{0, 0, 0, 4, 2, 1};
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferCopyTextureToTexture>()(&recorder.commands, &source_object, nullptr, &source_region,
        &first_object, nullptr, &target, NVN_COPY_FLAGS_NONE);
    load<Entry::nvnCommandBufferCopyTextureToTexture>()(&recorder.commands, &source_object, nullptr, &other_region,
        &second_object, nullptr, &target, NVN_COPY_FLAGS_LINEAR_FILTER_BIT);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &first_object, nullptr, &target,
        CopyResources::base + 0x4000, NVN_COPY_FLAGS_NONE);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &second_object, nullptr, &target,
        CopyResources::base + 0x5000, NVN_COPY_FLAGS_NONE);
    const auto handle = load<Entry::nvnCommandBufferEndRecording>()(&recorder.commands);
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* a = reinterpret_cast<const HalfPixel*>(static_cast<const std::byte*>(resources.memory.contents) + 0x4000);
    const auto* b = reinterpret_cast<const HalfPixel*>(static_cast<const std::byte*>(resources.memory.contents) + 0x5000);
    for (unsigned y = 0; y != 2; ++y) for (unsigned x = 0; x != 4; ++x) {
        CHECK(a[y * 4 + x] == pattern(2 * x + 1, 2 * y + 1));
        const HalfPixel expected{(_Float16)(9 + x + 20 * y), (_Float16)(1 + x * 0.25f), (_Float16)(2 * y + 0.5f), 1};
        CHECK(b[y * 4 + x] == expected);
    }
    CHECK(backend->statistics().copies == 4 && backend->statistics().skipped == 0);
    CHECK(resources.texture_writes.load() >= 2 && resources.buffer_writes.load() >= 2);
    CHECK(backend->stop_submission_workers(std::chrono::steady_clock::now() + std::chrono::seconds(3)));
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Separate images in one linear pool, including separate no-copy wrappers,
// are valid when their complete texture storage intervals are disjoint.
void test_disjoint_buffer_copy(id<MTLDevice> device, id<MTL4CommandQueue> queue, unsigned mode) {
    CopyResources resources(device);
    auto source_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
        width:8 height:4 mipmapped:NO];
    source_descriptor.storageMode = MTLStorageModeShared;
    source_descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    source_descriptor.allowGPUOptimizedContents = NO;
    MTLTextureDescriptor* target_descriptor = [source_descriptor copy]; target_descriptor.width = 4; target_descriptor.height = 2;
    auto backing = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLBuffer> other = backing;
    NSUInteger source_offset = 0;
    if (mode == 2) {
        void* bytes = mmap(nullptr, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(bytes != MAP_FAILED);
        if (bytes == MAP_FAILED) return;
        auto allocation = std::shared_ptr<void>(bytes, [](void* value) { munmap(value, 65536); });
        backing = [device newBufferWithBytesNoCopy:bytes length:65536 options:MTLResourceStorageModeShared
            deallocator:^(void*, NSUInteger) { static_cast<void>(allocation.get()); }];
        other = [device newBufferWithBytesNoCopy:static_cast<std::byte*>(bytes) + 16384 length:49152
            options:MTLResourceStorageModeShared deallocator:^(void*, NSUInteger) { static_cast<void>(allocation.get()); }];
        source_offset = 16384;
    }
    auto source = [backing newTextureWithDescriptor:source_descriptor offset:source_offset bytesPerRow:256];
    auto destination = [other newTextureWithDescriptor:target_descriptor offset:2048 bytesPerRow:256];
    CHECK(source && destination && source.buffer && destination.buffer);
    if (!source || !destination) return;
    CHECK(mode == 2 ? source.buffer != destination.buffer : source.buffer == destination.buffer);
    std::array<HalfPixel, 32> input;
    for (unsigned y = 0; y != 4; ++y) for (unsigned x = 0; x != 8; ++x) input[y * 8 + x] = pattern(x, y);
    const HalfPixel sentinel{9, 8, 7, 1}; std::array<HalfPixel, 8> old; old.fill(sentinel);
    [source replaceRegion:MTLRegionMake2D(0, 0, 8, 4) mipmapLevel:0 withBytes:input.data() bytesPerRow:8 * sizeof(HalfPixel)];
    [destination replaceRegion:MTLRegionMake2D(0, 0, 4, 2) mipmapLevel:0 withBytes:old.data() bytesPerRow:4 * sizeof(HalfPixel)];
    resources.textures = {{1, source}, {2, destination}};
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto source_state = texture_state(1, 8, 4, false, 1), target_state = texture_state(2, 4, 2, false, 1);
    auto source_object = texture_object(source_state), target_object = texture_object(target_state);
    const bool partial = mode == 1;
    const NVNcopyRegion input_region{partial ? 2 : 0, 0, 0, partial ? 4 : 8, 4, 1};
    const NVNcopyRegion output_region{partial ? 1 : 0, 0, 0, partial ? 2 : 4, 2, 1}, whole{0, 0, 0, 4, 2, 1};
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferCopyTextureToTexture>()(&recorder.commands, &source_object, nullptr, &input_region,
        &target_object, nullptr, &output_region, NVN_COPY_FLAGS_NONE);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        CopyResources::base + 0x4000, NVN_COPY_FLAGS_NONE);
    const auto handle = load<Entry::nvnCommandBufferEndRecording>()(&recorder.commands);
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* data = reinterpret_cast<const HalfPixel*>(static_cast<const std::byte*>(resources.memory.contents) + 0x4000);
    for (unsigned y = 0; y != 2; ++y) for (unsigned x = 0; x != 4; ++x) {
        const auto expected = partial && (x == 0 || x == 3) ? sentinel :
            pattern(partial ? 2 + 2 * (x - 1) + 1 : 2 * x + 1, 2 * y + 1);
        CHECK(data[y * 4 + x] == expected);
    }
    CHECK(backend->statistics().copies == 2 && backend->statistics().skipped == 0);
    CHECK(resources.texture_writes.load() >= 1 && resources.buffer_writes.load() >= 1);
    CHECK(backend->stop_submission_workers(std::chrono::steady_clock::now() + std::chrono::seconds(3)));
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Unsupported scaled forms must remain diagnosed without writing a target.
void test_unsupported_copy(id<MTLDevice> device, id<MTL4CommandQueue> queue, unsigned mode) {
    const bool integer = mode == 0, alias = mode == 2, multisample = mode == 3, volume = mode == 4, depth = mode == 5;
    const bool buffer_alias = mode == 6 || mode == 7;
    const auto format = integer ? MTLPixelFormatR32Uint : depth ? MTLPixelFormatDepth32Float : MTLPixelFormatRGBA16Float;
    CopyResources resources(device);
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:8 height:4 mipmapped:NO];
    descriptor.storageMode = multisample || depth ? MTLStorageModePrivate : MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    if (multisample) { descriptor.textureType = MTLTextureType2DMultisample; descriptor.sampleCount = 4; }
    if (volume) { descriptor.textureType = MTLTextureType3D; descriptor.depth = 2; }
    auto source = [device newTextureWithDescriptor:descriptor];
    auto target_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:4 height:2 mipmapped:NO];
    target_descriptor.storageMode = depth ? MTLStorageModePrivate : MTLStorageModeShared;
    target_descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    auto destination = [device newTextureWithDescriptor:target_descriptor];
    if (buffer_alias) {
        descriptor.allowGPUOptimizedContents = NO;
        target_descriptor.allowGPUOptimizedContents = NO;
        auto backing = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        id<MTLBuffer> other = backing;
        NSUInteger source_offset = 0;
        if (mode == 7) {
            void* bytes = mmap(nullptr, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            CHECK(bytes != MAP_FAILED);
            if (bytes == MAP_FAILED) return;
            auto allocation = std::shared_ptr<void>(bytes, [](void* value) { munmap(value, 65536); });
            backing = [device newBufferWithBytesNoCopy:bytes length:65536 options:MTLResourceStorageModeShared
                deallocator:^(void*, NSUInteger) { static_cast<void>(allocation.get()); }];
            other = [device newBufferWithBytesNoCopy:static_cast<std::byte*>(bytes) + 16384 length:49152
                options:MTLResourceStorageModeShared deallocator:^(void*, NSUInteger) { static_cast<void>(allocation.get()); }];
            source_offset = 16384;
        }
        source = [backing newTextureWithDescriptor:descriptor offset:source_offset bytesPerRow:256];
        destination = [other newTextureWithDescriptor:target_descriptor offset:256 bytesPerRow:256];
        CHECK(source && destination && source != destination && !source.parentTexture && !destination.parentTexture);
        CHECK(source.buffer && destination.buffer);
        CHECK(mode == 7 ? source.buffer != destination.buffer : source.buffer == destination.buffer);
    }
    if (alias && source) {
        destination = [source newTextureViewWithPixelFormat:format];
        source = [source newTextureViewWithPixelFormat:format];
    }
    CHECK(source && destination);
    if (!source || !destination) return;
    const HalfPixel sentinel{9, 8, 7, 1};
    std::array<HalfPixel, 8> half_pixels; half_pixels.fill(sentinel);
    std::array<std::uint32_t, 8> uint_pixels; uint_pixels.fill(0x12345678u);
    if (!depth) [destination replaceRegion:MTLRegionMake2D(0, 0, 4, 2) mipmapLevel:0
        withBytes:integer ? static_cast<const void*>(uint_pixels.data()) : static_cast<const void*>(half_pixels.data())
        bytesPerRow:integer ? 16 : 4 * sizeof(HalfPixel)];
    resources.textures = {{1, source}, {2, destination}};
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto source_state = texture_state(1, 8, 4, false, 1), target_state = texture_state(2, 4, 2, false, 1);
    if (integer) source_state.description.format = target_state.description.format = NVN_FORMAT_R32UI;
    if (depth) source_state.description.format = target_state.description.format = NVN_FORMAT_DEPTH32F;
    if (volume) { source_state.description.target = NVN_TEXTURE_TARGET_3D; source_state.description.depth = 2; }
    if (multisample) source_state.description.samples = 4;
    auto source_object = texture_object(source_state), target_object = texture_object(target_state);
    const NVNcopyRegion input{0, 0, 0, 8, 4, 1}, target{0, 0, 0, 4, 2, 1};
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferCopyTextureToTexture>()(&recorder.commands, &source_object, nullptr, &input,
        &target_object, nullptr, &target, mode == 1 ? NVN_COPY_FLAGS_MIRROR_X_BIT : NVN_COPY_FLAGS_NONE);
    const auto handle = load<Entry::nvnCommandBufferEndRecording>()(&recorder.commands);
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(backend->statistics().copies == 0 && backend->statistics().skipped == 1);
    const auto reasons = backend->skip_reasons();
    const char* expected = mode == 1 ? "flags" : alias || buffer_alias ? "staging" : "single-sample 2D color";
    CHECK(reasons.size() == 1 && reasons.front().first.find(expected) != std::string::npos && reasons.front().second == 1);
    CHECK(resources.texture_writes.load() == 0 && resources.buffer_writes.load() == 0);
    if (!depth) {
        if (integer) {
            std::array<std::uint32_t, 8> pixels{};
            [destination getBytes:pixels.data() bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 2) mipmapLevel:0];
            CHECK(pixels == uint_pixels);
        } else {
            std::array<HalfPixel, 8> pixels{};
            [destination getBytes:pixels.data() bytesPerRow:4 * sizeof(HalfPixel) fromRegion:MTLRegionMake2D(0, 0, 4, 2) mipmapLevel:0];
            CHECK(pixels == half_pixels);
        }
    }
    CHECK(backend->stop_submission_workers(std::chrono::steady_clock::now() + std::chrono::seconds(3)));
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}
} // namespace

int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory) return 77;
        select_api_version({55, 15});
        register_command_entries();
        for (unsigned mode = 0; mode != 6; ++mode) test_scaled_copy(device, [device newMTL4CommandQueue], mode);
        test_copy_parameter_lifetime(device, [device newMTL4CommandQueue]);
        for (unsigned mode = 0; mode != 3; ++mode) test_disjoint_buffer_copy(device, [device newMTL4CommandQueue], mode);
        for (unsigned mode = 0; mode != 8; ++mode) test_unsupported_copy(device, [device newMTL4CommandQueue], mode);
        std::fprintf(stderr, "Metal scaled-copy contract failures: %u\n", failures);
        return failures ? 1 : 0;
    }
}
