// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic GPU tests for the persistent Metal texture
// cache. Synthetic guest bytes use independent scalar GOB addressing; every
// texel is checked by GPU sampling against its original patterned value.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device_context.h"
#include "texture_cache.h"
#include "texture_residency.h"

#include <armigo/graphics/resource_memory_registry.h>
#include <armigo/graphics/texture_layout.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
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

bool succeeded(const RendererStatus& result) {
    return std::holds_alternative<std::monostate>(result);
}

template <class T>
T require_value(RendererResult<T> result) {
    CHECK(std::holds_alternative<T>(result));
    if (const auto* error = std::get_if<RendererError>(&result)) {
        const CommandResult command = error->command();
        std::fprintf(stderr, "Cannot continue: %.*s\n",
                     static_cast<int>(command.detail.size()),
                     command.detail.data());
        std::abort();
    }
    return std::get<T>(std::move(result));
}

template <class T>
T require_layout(TextureLayoutResult<T> result) {
    CHECK(std::holds_alternative<T>(result));
    if (const auto* error = std::get_if<TextureLayoutError>(&result)) {
        std::fprintf(stderr, "Cannot continue: %s\n", error->message());
        std::abort();
    }
    return std::get<T>(std::move(result));
}

RendererError synthetic_error(const char* detail) {
    return RendererError({CommandStatus::MemoryFault, detail});
}

struct Backend final : ResourceMemoryBackend {
    struct Allocation {
        BackingId id;
        std::vector<std::byte> bytes;
    };

    Allocation& add(BackingId id, std::size_t size) {
        allocations.push_back({id, std::vector<std::byte>(size)});
        return allocations.back();
    }

    Allocation* find(BackingId id) {
        const auto found = std::find_if(
            allocations.begin(), allocations.end(),
            [&](const Allocation& allocation) { return allocation.id == id; });
        return found == allocations.end() ? nullptr : &*found;
    }

    RendererStatus read_cpu(BackingRange range,
                            std::span<std::byte> destination) override {
        ++reads;
        Allocation* allocation = find(range.backing);
        if (!allocation || destination.size() != range.size ||
            range.offset > allocation->bytes.size() ||
            range.size > allocation->bytes.size() - range.offset) {
            return synthetic_error("synthetic texture CPU read failed");
        }
        std::copy_n(allocation->bytes.begin() + range.offset,
                    destination.size(), destination.begin());
        return renderer_success();
    }

    RendererStatus write_cpu(BackingRange range,
                             std::span<const std::byte> source) override {
        ++writes;
        Allocation* allocation = find(range.backing);
        if (!allocation || source.size() != range.size ||
            range.offset > allocation->bytes.size() ||
            range.size > allocation->bytes.size() - range.offset) {
            return synthetic_error("synthetic texture CPU write failed");
        }
        std::copy(source.begin(), source.end(),
                  allocation->bytes.begin() + range.offset);
        return renderer_success();
    }

    RendererStatus wait_gpu(CompletionToken token) override {
        ++waits;
        if (!token.valid()) {
            return synthetic_error("synthetic texture GPU wait has no token");
        }
        const SubmissionPhase phase = token.snapshot().phase;
        if (phase != SubmissionPhase::GpuComplete &&
            phase != SubmissionPhase::Publishing &&
            phase != SubmissionPhase::Complete) {
            return synthetic_error("synthetic texture GPU wait is not complete");
        }
        return renderer_success();
    }

    std::vector<Allocation> allocations;
    std::atomic<unsigned> reads = 0;
    std::atomic<unsigned> writes = 0;
    std::atomic<unsigned> waits = 0;
};

struct NoopActions final : CompletionActions {
    RendererStatus make_coherent() override { return renderer_success(); }
    RendererStatus prepare_publication() override { return renderer_success(); }
    void publish() noexcept override {}
    RendererStatus preserve_failed_authority() override {
        return renderer_success();
    }
};

struct Context final : metal::DeviceContext {
    explicit Context(id<MTLDevice> native_device)
        : native_device_(native_device), native_queue_([native_device newMTL4CommandQueue]) {
        identity_.device = 0x4c34544558545552ull;
        identity_.compiler_chain = "metal-4-test";
        identity_.capability_profile = "synthetic-texture-cache";
    }

    id<MTLDevice> device() const noexcept override { return native_device_; }
    id<MTL4CommandQueue> queue() const noexcept override { return native_queue_; }
    const metal::PipelineContextKey& identity() const noexcept override {
        return identity_;
    }

private:
    id<MTLDevice> native_device_ = nil;
    id<MTL4CommandQueue> native_queue_ = nil;
    metal::PipelineContextKey identity_;
};

struct TextureAsset {
    TextureDescriptor descriptor;
    TextureLayoutDescription layout;
    RetainedRange retained;
    std::vector<std::vector<std::array<std::uint8_t, 4>>> expected;
};

// Synthetic write-tracking registry, mirroring the buffer cache tests:
// forwards every operation to a real registry and overrides only the optional
// CPU write stamp contract. touch() advances the store generation for the
// whole address space or, with a range, for exactly the stored ranges.
class StampRegistry final : public ResourceMemoryRegistry {
public:
    explicit StampRegistry(std::shared_ptr<ResourceMemoryRegistry> inner)
        : inner_(std::move(inner)) {}

    std::uint64_t cpu_write_stamp() const noexcept override { return stamp_; }
    bool cpu_written_since(BackingRange range, std::uint64_t stamp) const noexcept override {
        if (stamp >= whole_since_) return false;
        if (whole_dirty_) return true;
        for (const auto& dirty : dirty_) {
            if (dirty.backing == range.backing &&
                dirty.offset < range.offset + range.size &&
                range.offset < dirty.offset + dirty.size) return true;
        }
        return false;
    }
    void touch() {
        ++stamp_;
        whole_since_ = stamp_;
        whole_dirty_ = true;
    }
    void touch(BackingRange range) {
        ++stamp_;
        whole_since_ = stamp_;
        whole_dirty_ = false;
        dirty_.push_back(range);
    }

    RendererStatus register_backing(BackingRegistration backing) override {
        return inner_->register_backing(backing);
    }
    RendererStatus register_mapping(MappingRegistration mapping) override {
        return inner_->register_mapping(mapping);
    }
    RendererStatus unmap(MappingId mapping) override { return inner_->unmap(mapping); }
    RendererStatus retire_backing(BackingId backing) override {
        return inner_->retire_backing(backing);
    }
    RendererResult<RetainedRange> retain(std::uint64_t gpu_address, std::uint64_t size,
                                         ResourceAccess access) override {
        return inner_->retain(gpu_address, size, access);
    }
    const GpuAddressSpace& address_space() const noexcept override {
        return inner_->address_space();
    }
    CoherenceMode coherence_mode() const noexcept override {
        return inner_->coherence_mode();
    }
    RendererResult<std::vector<DirtyRange>> inspect(BackingRange range) override {
        return inner_->inspect(range);
    }
    RendererStatus resolve(BackingRange range, ResolveReason reason) override {
        return inner_->resolve(range, reason);
    }
    RendererResult<std::unique_ptr<CpuReadTransaction>> prepare_cpu_read(
        BackingRange range, ResolveReason reason) override {
        return inner_->prepare_cpu_read(range, reason);
    }
    RendererResult<std::unique_ptr<CpuWriteTransaction>> prepare_cpu_write(
        BackingRange range) override {
        return inner_->prepare_cpu_write(range);
    }
    RendererResult<std::shared_ptr<GpuAccessReservation>> reserve_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer) override {
        return inner_->reserve_accesses(requests, consumer);
    }
    RendererStatus invalidate_mappings(std::span<const MappingId> mappings,
                                       std::span<const BackingId> discarded) override {
        return inner_->invalidate_mappings(mappings, discarded);
    }
    bool has_backing(BackingId backing) const noexcept override {
        return inner_->has_backing(backing);
    }
    RendererStatus validate_retained(const RetainedRange& range,
                                     ResourceAccess access) override {
        return inner_->validate_retained(range, access);
    }
    RendererStatus read_retained(const RetainedRange& retained, BackingRange bytes,
                                 std::span<std::byte> destination) override {
        return inner_->read_retained(retained, bytes, destination);
    }
    RendererResult<std::vector<ResourceStorageRange>> inspect_storage(
        BackingRange range) override {
        return inner_->inspect_storage(range);
    }
    RendererStatus submission_boundary() override {
        return inner_->submission_boundary();
    }
    std::uint64_t cpu_cache_epoch() const noexcept override {
        return inner_->cpu_cache_epoch();
    }

private:
    std::shared_ptr<ResourceMemoryRegistry> inner_;
    std::uint64_t stamp_ = 1, whole_since_ = 0;
    bool whole_dirty_ = false;
    std::vector<BackingRange> dirty_;
};

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Fixture-only address oracle: enumerate bytes within GOB tiles rather than
// invoking the production forward or inverse converters.
void store_fixture_level(const TextureLayoutDescription& layout,
                         std::span<const std::byte> linear,
                         std::span<std::byte> guest,
                         std::uint32_t level, std::uint32_t layer) {
    const auto& mip = layout.mips[level];
    const std::size_t gob_height = std::size_t{1} << mip.gob_block_height_log2;
    const std::size_t gob_depth = std::size_t{1} << mip.gob_block_depth_log2;
    const std::size_t tiles_x = (mip.linear_row_bytes + 63) / 64;
    const std::size_t tiles_y = (mip.block_rows + 8 * gob_height - 1) / (8 * gob_height);
    for (std::size_t z = 0; z < mip.depth; ++z) {
        for (std::size_t y = 0; y < mip.block_rows; ++y) {
            for (std::size_t x = 0; x < mip.linear_row_bytes; ++x) {
                std::size_t offset = y * layout.pitch_bytes + x;
                if (layout.storage == TextureStorageLayout::BlockLinear) {
                    const std::size_t tile =
                        ((z / gob_depth * tiles_y + y / (8 * gob_height)) * tiles_x + x / 64);
                    const std::size_t gob = (z % gob_depth) * gob_height + (y / 8) % gob_height;
                    const std::size_t within = (x & 15) | ((y & 1) << 4) |
                        ((x & 16) << 1) | ((y & 6) << 5) | ((x & 32) << 3);
                    offset = tile * 512 * gob_height * gob_depth + gob * 512 + within;
                }
                offset += layer * layout.layer_stride + mip.offset;
                CHECK(offset < guest.size());
                if (offset < guest.size())
                    guest[offset] = linear[(z * mip.block_rows + y) * mip.linear_row_bytes + x];
            }
        }
    }
}

struct Fixture {
    Fixture(id<MTLDevice> device, std::uint64_t budget = 32u * 1024u * 1024u,
            std::function<std::shared_ptr<ResourceMemoryRegistry>(
                std::shared_ptr<ResourceMemoryRegistry>)> wrap = {})
        : backend(std::make_shared<Backend>()), context(std::make_shared<Context>(device)) {
        memory = require_value(make_resource_memory(
            {0x4c34, backend, CoherenceMode::SubmissionBoundary, 0x1000}));
        if (wrap) memory = wrap(memory);
        cache = require_value(metal::create_texture_cache(context, memory, budget));
    }

    TextureAsset add_texture(TextureType type, std::uint32_t width,
                             std::uint32_t height, std::uint32_t depth,
                             std::uint32_t levels, std::uint32_t layers,
                             TextureLayout storage = TextureLayout::BlockLinear,
                             TextureFormat format = TextureFormat::R8G8B8A8Unorm,
                             std::uint32_t gob_height_log2 = 1,
                             std::uint32_t gob_depth_log2 = 1,
                             bool depth_blocks_on_any_target = false) {
        const TextureFormatInfo info = texture_format_info(format);
        TextureLayoutDescription description{};
        description.width = width;
        description.height = height;
        description.depth = depth;
        description.levels = levels;
        description.layers = layers;
        description.block_width = info.block_width;
        description.block_height = info.block_height;
        description.bytes_per_block = info.bytes_per_block;
        description.gob_block_height_log2 = storage == TextureLayout::BlockLinear ? gob_height_log2 : 0;
        description.gob_block_depth_log2 =
            storage == TextureLayout::BlockLinear &&
                    (type == TextureType::Texture3D || depth_blocks_on_any_target)
                ? gob_depth_log2
                : 0;
        description.storage = storage == TextureLayout::Pitch
                                  ? TextureStorageLayout::Pitch
                                  : TextureStorageLayout::BlockLinear;
        description.pitch_bytes = storage == TextureLayout::Pitch
                                      ? align_up(std::size_t{width} * info.bytes_per_block, 64)
                                      : 0;
        TextureAsset asset;
        asset.layout = require_layout(calculate_texture_size(description));

        const std::uint64_t allocation = next_allocation++;
        const BackingId backing{0x4c34, allocation, 1};
        const MappingId mapping{0x4c34, allocation, 1};
        const std::size_t registered_size = align_up(asset.layout.total_size, 0x1000);
        Backend::Allocation& bytes = backend->add(backing, registered_size);
        std::fill(bytes.bytes.begin(), bytes.bytes.end(), std::byte{0xd3});
        const std::uint64_t address = allocation * 0x100000;
        CHECK(succeeded(memory->register_backing(
            {backing, registered_size, backend})));
        CHECK(succeeded(memory->register_mapping(
            {mapping, backing,
             {address, registered_size, 0, address + 0x40000000,
              registered_size, 0x1000}})));

        asset.expected.resize(std::size_t{levels} * layers);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            for (std::uint32_t level = 0; level < levels; ++level) {
                const auto& mip = asset.layout.mips[level];
                std::vector<std::byte> linear(mip.linear_bytes);
                auto& expected =
                    asset.expected[std::size_t{layer} * levels + level];
                expected.resize(std::size_t{mip.width} * mip.height * mip.depth);
                if (format == TextureFormat::R8G8B8A8Unorm ||
                    format == TextureFormat::R8G8B8A8Srgb) {
                    for (std::size_t texel = 0; texel < expected.size(); ++texel) {
                        const std::uint32_t value =
                            static_cast<std::uint32_t>(texel + level * 65536 + layer * 1048576 + 1) *
                            2654435761u;
                        for (unsigned component = 0; component < 4; ++component) {
                            const auto byte = static_cast<std::uint8_t>(value >> (component * 8));
                            expected[texel][component] = byte;
                            linear[texel * 4 + component] = std::byte{byte};
                        }
                    }
                } else if (format == TextureFormat::Bc1RgbaUnorm) {
                    // One solid block colour per depth slice (red, green, blue,
                    // white), so 3D slice placement is observable; 2D textures
                    // stay solid red.
                    constexpr std::array<std::uint16_t, 4> colors565{0xf800, 0x07e0, 0x001f, 0xffff};
                    constexpr std::array<std::array<std::uint8_t, 4>, 4> colors{{
                        {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 255, 255}}};
                    const std::size_t slice_bytes = linear.size() / mip.depth;
                    const std::size_t slice_texels = std::size_t{mip.width} * mip.height;
                    for (std::uint32_t z = 0; z < mip.depth; ++z) {
                        const auto color = colors565[z % colors565.size()];
                        const std::array<std::byte, 8> block{
                            std::byte(color & 0xff), std::byte(color >> 8),
                            std::byte{0x00}, std::byte{0x00},
                            std::byte{0x00}, std::byte{0x00},
                            std::byte{0x00}, std::byte{0x00}};
                        for (std::size_t offset = 0; offset < slice_bytes; offset += block.size())
                            std::copy(block.begin(), block.end(), linear.begin() + z * slice_bytes + offset);
                        std::fill(expected.begin() + z * slice_texels, expected.begin() + (z + 1) * slice_texels,
                                  colors[z % colors.size()]);
                    }
                }
                store_fixture_level(asset.layout, linear, bytes.bytes, level, layer);
            }
        }

        TextureDescriptor& descriptor = asset.descriptor;
        descriptor.address = address;
        descriptor.backing_extent = asset.layout.total_size;
        descriptor.row_bytes = storage == TextureLayout::Pitch
                                   ? asset.layout.pitch_bytes
                                   : align_up(asset.layout.mips[0].linear_row_bytes, 64);
        descriptor.storage_height = storage == TextureLayout::Pitch
                                        ? asset.layout.mips[0].block_rows
                                        : align_up(asset.layout.mips[0].block_rows,
                                                   std::size_t{8} <<
                                                       asset.layout.mips[0].gob_block_height_log2);
        descriptor.layer_stride = asset.layout.layer_stride;
        descriptor.width = width;
        descriptor.height = height;
        descriptor.depth = depth;
        descriptor.levels = levels;
        descriptor.layers = layers;
        descriptor.samples = 1;
        descriptor.block_height_log2 =
            static_cast<std::uint8_t>(asset.layout.gob_block_height_log2);
        descriptor.block_depth_log2 =
            static_cast<std::uint8_t>(asset.layout.gob_block_depth_log2);
        descriptor.format = format;
        descriptor.layout = storage;
        descriptor.type = type;
        descriptor.srgb = format == TextureFormat::R8G8B8A8Srgb;
        descriptor.normalized_coordinates = type != TextureType::Texture2DRect;
        const TextureLayoutDescription rebuilt =
            require_layout(texture_view_layout(descriptor));
        CHECK(rebuilt.total_size <= asset.layout.total_size);
        descriptor.backing_extent = rebuilt.total_size;
        asset.layout.total_size = rebuilt.total_size;
        CHECK(rebuilt.layer_stride == asset.layout.layer_stride);
        CHECK(rebuilt.mips.size() == asset.layout.mips.size());
        asset.retained = require_value(memory->retain(
            address, asset.layout.total_size, ResourceAccess::ReadWrite));
        return asset;
    }

    TextureRequest request(const TextureAsset& asset,
                           ResourceAccess access = ResourceAccess::Read) const {
        TextureStorageKey storage{asset.retained.bytes, asset.descriptor};
        TextureViewKey view{storage};
        view.level_count = asset.descriptor.levels;
        view.layer_count = asset.descriptor.layers;
        return {asset.retained, view, access};
    }

    std::shared_ptr<Backend> backend;
    std::shared_ptr<ResourceMemoryRegistry> memory;
    std::shared_ptr<Context> context;
    std::shared_ptr<TextureCache> cache;
    std::uint64_t next_allocation = 1;
};

const char* kSamplingSource = R"metal(
#include <metal_stdlib>
using namespace metal;

struct Probe {
    uint level; uint layer; uint z; uint x; uint y;
    float x_coordinate; float y_coordinate; float z_coordinate;
};

constexpr sampler nearest_sampler(coord::normalized, address::clamp_to_edge,
                                  filter::nearest, mip_filter::nearest);

float3 cube_direction(uint face, float x, float y) {
    float u = 2.0 * x - 1.0;
    float v = 2.0 * y - 1.0;
    switch (face) {
        case 0: return float3(1.0, -v, -u);
        case 1: return float3(-1.0, -v, u);
        case 2: return float3(u, 1.0, v);
        case 3: return float3(u, -1.0, -v);
        case 4: return float3(u, -v, 1.0);
        default: return float3(-u, -v, -1.0);
    }
}

kernel void sample_1d(texture1d<float, access::sample> source [[texture(0)]],
                      device const Probe* probes [[buffer(0)]],
                      device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = source.sample(nearest_sampler, probes[tid].x_coordinate);
}
kernel void sample_1d_array(texture1d_array<float, access::sample> source [[texture(0)]],
                            device const Probe* probes [[buffer(0)]],
                            device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = source.sample(nearest_sampler, probes[tid].x_coordinate, probes[tid].layer);
}
kernel void sample_2d(texture2d<float, access::sample> source [[texture(0)]],
                      device const Probe* probes [[buffer(0)]],
                      device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = source.sample(nearest_sampler, float2(probes[tid].x_coordinate, probes[tid].y_coordinate),
                                level(float(probes[tid].level)));
}
kernel void sample_2d_array(texture2d_array<float, access::sample> source [[texture(0)]],
                            device const Probe* probes [[buffer(0)]],
                            device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = source.sample(nearest_sampler, float2(probes[tid].x_coordinate, probes[tid].y_coordinate), probes[tid].layer,
                                level(float(probes[tid].level)));
}
kernel void sample_3d(texture3d<float, access::sample> source [[texture(0)]],
                      device const Probe* probes [[buffer(0)]],
                      device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = source.sample(nearest_sampler,
                                float3(probes[tid].x_coordinate, probes[tid].y_coordinate, probes[tid].z_coordinate),
                                level(float(probes[tid].level)));
}
kernel void sample_cube(texturecube<float, access::sample> source [[texture(0)]],
                        device const Probe* probes [[buffer(0)]],
                        device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = source.sample(nearest_sampler, cube_direction(probes[tid].layer, probes[tid].x_coordinate, probes[tid].y_coordinate),
                                level(float(probes[tid].level)));
}
kernel void sample_cube_array(texturecube_array<float, access::sample> source [[texture(0)]],
                              device const Probe* probes [[buffer(0)]],
                              device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = source.sample(nearest_sampler,
                                cube_direction(probes[tid].layer % 6, probes[tid].x_coordinate, probes[tid].y_coordinate),
                                probes[tid].layer / 6,
                                level(float(probes[tid].level)));
}

// Depth admission probe: the comparison result (x) and the plain depth read
// (y) of one texel, so a colour-coded alias binding can be checked against a
// direct depth compare of the same storage.
kernel void probe_depth(depth2d<float, access::sample> source [[texture(0)]],
                        device const Probe* probes [[buffer(0)]],
                        device float4* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    constexpr sampler compare_sampler(coord::normalized, address::clamp_to_edge,
                                      filter::nearest, mip_filter::nearest,
                                      compare_func::less_equal);
    constexpr sampler plain_sampler(coord::normalized, address::clamp_to_edge,
                                    filter::nearest, mip_filter::nearest);
    const float2 uv = float2(probes[tid].x_coordinate, probes[tid].y_coordinate);
    output[tid] = float4(
        source.sample_compare(compare_sampler, uv, probes[tid].z_coordinate,
                              level(float(probes[tid].level))),
        source.sample(plain_sampler, uv, level(float(probes[tid].level))),
        0.0, 0.0);
}
)metal";

struct Probe {
    std::uint32_t level;
    std::uint32_t layer;
    std::uint32_t z;
    std::uint32_t x;
    std::uint32_t y;
    float x_coordinate;
    float y_coordinate;
    float z_coordinate;
};

struct SamplingHarness {
    explicit SamplingHarness(const std::shared_ptr<Context>& context)
        : context(context) {
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        library = [context->device()
            newLibraryWithSource:[NSString stringWithUTF8String:kSamplingSource]
                         options:options error:&error];
        MTL4CompilerDescriptor* descriptor = [MTL4CompilerDescriptor new];
        compiler = [context->device() newCompilerWithDescriptor:descriptor error:&error];
        if (!library || !compiler) {
            std::fprintf(stderr, "Texture sampling compiler setup failed: %s\n",
                         error.localizedDescription.UTF8String);
        }
    }

    id<MTLComputePipelineState> pipeline(const char* name) const {
        if (!library || !compiler) return nil;
        NSError* error = nil;
        MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = [NSString stringWithUTF8String:name];
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        id<MTLComputePipelineState> result = [compiler
            newComputePipelineStateWithDescriptor:descriptor
                               compilerTaskOptions:nil error:&error];
        if (!result) {
            std::fprintf(stderr, "Texture sampling pipeline %s failed: %s\n",
                         name, error.localizedDescription.UTF8String);
        }
        return result;
    }

    bool sample(
        const TextureAsset& asset, const metal::TextureResource& texture,
        const char* function, bool decode_srgb = false,
        std::array<TextureSwizzle, 4> expected_swizzle = {
            TextureSwizzle::Red, TextureSwizzle::Green,
            TextureSwizzle::Blue, TextureSwizzle::Alpha},
        const std::function<void()>& while_pending = {}) {
        std::vector<Probe> probes;
        for (std::uint32_t layer = 0; layer < asset.descriptor.layers; ++layer) {
            for (std::uint32_t level = 0; level < asset.descriptor.levels; ++level) {
                const auto& mip = asset.layout.mips[level];
                for (std::uint32_t z = 0; z < mip.depth; ++z) {
                    for (std::uint32_t y = 0; y < mip.height; ++y) {
                        for (std::uint32_t x = 0; x < mip.width; ++x) {
                            probes.push_back({level, layer, z, x, y,
                                (static_cast<float>(x) + 0.5f) / mip.width,
                                (static_cast<float>(y) + 0.5f) / mip.height,
                                (static_cast<float>(z) + 0.5f) / mip.depth});
                        }
                    }
                }
            }
        }
        id<MTLComputePipelineState> state = pipeline(function);
        id<MTLBuffer> probe_buffer = [context->device()
            newBufferWithBytes:probes.data() length:probes.size() * sizeof(Probe)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [context->device()
            newBufferWithLength:probes.size() * sizeof(std::array<float, 4>)
                        options:MTLResourceStorageModeShared];
        MTL4ArgumentTableDescriptor* table_descriptor =
            [MTL4ArgumentTableDescriptor new];
        table_descriptor.maxBufferBindCount = 2;
        table_descriptor.maxTextureBindCount = 1;
        NSError* error = nil;
        id<MTL4ArgumentTable> table = [context->device()
            newArgumentTableWithDescriptor:table_descriptor error:&error];
        MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
        residency_descriptor.initialCapacity = 3;
        id<MTLResidencySet> residency = [context->device()
            newResidencySetWithDescriptor:residency_descriptor error:&error];
        id<MTL4CommandAllocator> allocator = [context->device() newCommandAllocator];
        id<MTL4CommandBuffer> commands = [context->device() newCommandBuffer];
        if (!state || !probe_buffer || !output || !table || !residency ||
            !allocator || !commands) {
            std::fputs("Texture sampling dispatch allocation failed.\n", stderr);
            return false;
        }
        const float unwritten = std::numeric_limits<float>::quiet_NaN();
        std::fill_n(static_cast<std::array<float, 4>*>(output.contents), probes.size(),
                    std::array<float, 4>{unwritten, unwritten, unwritten, unwritten});
        [table setAddress:probe_buffer.gpuAddress atIndex:0];
        [table setAddress:output.gpuAddress atIndex:1];
        [table setTexture:texture.texture().gpuResourceID atIndex:0];
        [residency addAllocation:texture.allocation()];
        [residency addAllocation:probe_buffer];
        [residency addAllocation:output];
        [residency commit];
        NSArray* retained = @[library, compiler, state, probe_buffer, output, table,
                              residency, allocator, commands, texture.texture(),
                              texture.allocation()];

        [commands beginCommandBufferWithAllocator:allocator];
        [commands useResidencySet:residency];
        id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        if (!encoder) return false;
        [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch
                      visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder setComputePipelineState:state];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(probes.size(), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(probes.size(), 32),
                                               1, 1)];
        [encoder endEncoding];
        [commands endCommandBuffer];

        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block NSError* execution_error = nil;
        MTL4CommitOptions* commit = [MTL4CommitOptions new];
        [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            static_cast<void>(retained.count);
            execution_error = feedback.error;
            dispatch_semaphore_signal(done);
        }];
        id<MTL4CommandBuffer> submitted[] = {commands};
        [context->queue() commit:submitted count:1 options:commit];
        if (while_pending) {
            CHECK(dispatch_semaphore_wait(done, DISPATCH_TIME_NOW) != 0);
            while_pending();
        }
        if (dispatch_semaphore_wait(
                done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0 ||
            execution_error) {
            std::fprintf(stderr, "Texture sampling execution failed: %s\n",
                         execution_error.localizedDescription.UTF8String);
            return false;
        }

        const auto* sampled =
            static_cast<const std::array<float, 4>*>(output.contents);
        const auto srgb_to_linear = [](float value) {
            return value <= 0.04045f
                       ? value / 12.92f
                       : std::pow((value + 0.055f) / 1.055f, 2.4f);
        };
        for (std::size_t index = 0; index < probes.size(); ++index) {
            const auto& probe = probes[index];
            const auto& mip = asset.layout.mips[probe.level];
            const auto encoded = asset.expected[
                std::size_t{probe.layer} * asset.descriptor.levels + probe.level][
                    (std::size_t{probe.z} * mip.height + probe.y) * mip.width + probe.x];
            std::array<float, 4> source{};
            for (unsigned component = 0; component < 4; ++component) {
                source[component] =
                    static_cast<float>(encoded[component]) / 255.0f;
                if (decode_srgb && component < 3) {
                    source[component] = srgb_to_linear(source[component]);
                }
            }
            for (unsigned component = 0; component < 4; ++component) {
                float value = 0.0f;
                switch (expected_swizzle[component]) {
                    case TextureSwizzle::Red: value = source[0]; break;
                    case TextureSwizzle::Green: value = source[1]; break;
                    case TextureSwizzle::Blue: value = source[2]; break;
                    case TextureSwizzle::Alpha: value = source[3]; break;
                    case TextureSwizzle::OneSInt:
                    case TextureSwizzle::OneFloat: value = 1.0f; break;
                    case TextureSwizzle::Zero: break;
                }
                if (!std::isfinite((*sampled)[component]) ||
                    std::fabs((*sampled)[component] - value) > 1.5f / 255.0f) {
                    std::fprintf(stderr,
                        "Texture sample %s mip %u layer %u xyz (%u,%u,%u) component %u: "
                        "expected %.6f, got %.6f.\n", function,
                        probes[index].level, probes[index].layer,
                        probes[index].x, probes[index].y, probes[index].z, component, value,
                        (*sampled)[component]);
                    ++failures;
                }
            }
            ++sampled;
        }
        return true;
    }

    std::shared_ptr<Context> context;
    id<MTLLibrary> library = nil;
    id<MTL4Compiler> compiler = nil;
};

std::shared_ptr<const metal::TextureResource> native_texture(
    const ResidentTexture& resident) {
    auto texture = std::dynamic_pointer_cast<const metal::TextureResource>(
        resident.storage);
    CHECK(texture != nullptr);
    return texture;
}


bool is_error(const RendererStatus& result,
              CommandStatus expected = CommandStatus::Unsupported) {
    const auto* error = std::get_if<RendererError>(&result);
    CHECK(error != nullptr);
    if (!error) return false;
    CHECK(error->command().status == expected);
    return true;
}

template <class T>
bool is_error(const RendererResult<T>& result) {
    const auto* error = std::get_if<RendererError>(&result);
    CHECK(error != nullptr);
    if (!error) return false;
    return true;
}

// A 2D texture whose descriptor carries two GOBs in Z keeps that block-linear
// layout: every texel samples back from its Z-blocked guest address.
void test_two_dimensional_depth_blocks(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2D, 73, 45, 1, 1, 1, TextureLayout::BlockLinear,
        TextureFormat::R8G8B8A8Unorm, 1, 1, true);
    CHECK(asset.descriptor.block_depth_log2 == 1);
    const ResidentTexture resident = require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto native = native_texture(resident);
    CHECK(native != nullptr);
    if (native) CHECK(sampling.sample(asset, *native, "sample_2d"));
}

void test_reuse_versions_views_and_aliases(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2DArray, 4, 4, 1, 3, 3);
    const TextureRequest full = fixture.request(asset);
    const ResidentTexture first = require_value(fixture.cache->acquire(full));
    const ResidentTexture second = require_value(fixture.cache->acquire(full));
    const auto first_native = native_texture(first);
    const auto second_native = native_texture(second);
    CHECK(first_native && second_native);
    if (first_native && second_native) {
        CHECK(first_native->allocation() == second_native->allocation());
        CHECK(first_native->texture() == second_native->texture());
        CHECK(sampling.sample(asset, *second_native, "sample_2d_array"));
    }
    auto statistics = metal::texture_cache_statistics(*fixture.cache);
    CHECK(statistics.uploads == 1);
    CHECK(statistics.hits == 1);

    TextureRequest subset = full;
    subset.key.first_level = 1;
    subset.key.level_count = 2;
    subset.key.first_layer = 1;
    subset.key.layer_count = 1;
    const ResidentTexture subview = require_value(fixture.cache->acquire(subset));
    const auto subview_native = native_texture(subview);
    if (subview_native) {
        CHECK(subview_native->texture().mipmapLevelCount == 2);
        CHECK(subview_native->texture().arrayLength == 1);
        CHECK(subview_native->texture().width == 2);
        CHECK(subview_native->texture().height == 2);
        CHECK(subview_native->allocation() == first_native->allocation());
    }

    TextureRequest srgb = full;
    srgb.key.storage.descriptor.format = TextureFormat::R8G8B8A8Srgb;
    srgb.key.storage.descriptor.srgb = true;
    const ResidentTexture srgb_resident =
        require_value(fixture.cache->acquire(srgb));
    const auto srgb_native = native_texture(srgb_resident);
    if (srgb_native) {
        CHECK(srgb_native->allocation() == first_native->allocation());
        CHECK(srgb_native->texture().pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
        TextureAsset srgb_asset = asset;
        srgb_asset.descriptor.format = TextureFormat::R8G8B8A8Srgb;
        srgb_asset.descriptor.srgb = true;
        CHECK(sampling.sample(srgb_asset, *srgb_native,
                              "sample_2d_array", true));
    }
    TextureRequest swizzled = full;
    swizzled.key.swizzle = {
        TextureSwizzle::Blue, TextureSwizzle::OneFloat,
        TextureSwizzle::Red, TextureSwizzle::Zero};
    const ResidentTexture swizzled_resident =
        require_value(fixture.cache->acquire(swizzled));
    const auto swizzled_native = native_texture(swizzled_resident);
    if (swizzled_native) {
        CHECK(swizzled_native->allocation() == first_native->allocation());
        CHECK(sampling.sample(asset, *swizzled_native, "sample_2d_array",
                              false, swizzled.key.swizzle));
    }
    statistics = metal::texture_cache_statistics(*fixture.cache);
    CHECK(statistics.uploads == 1);
    CHECK(statistics.hits == 4);

    auto write = require_value(
        fixture.memory->prepare_cpu_write(asset.retained.bytes));
    Backend::Allocation* bytes = fixture.backend->find(asset.retained.bytes.backing);
    CHECK(bytes != nullptr);
    if (bytes) {
        const auto& mip = asset.layout.mips[0];
        std::vector<std::byte> linear(mip.linear_bytes);
        const std::array<std::uint8_t, 4> changed{233, 19, 37, 241};
        for (std::size_t offset = 0; offset < linear.size(); offset += 4) {
            for (unsigned component = 0; component < 4; ++component) {
                linear[offset + component] = std::byte{changed[component]};
            }
        }
        CHECK(std::holds_alternative<std::monostate>(write_level_from_linear(
            asset.layout, linear, bytes->bytes, 0, 0)));
        std::fill(asset.expected[0].begin(), asset.expected[0].end(), changed);
    }
    write->commit();
    const ResidentTexture refreshed =
        require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto refreshed_native = native_texture(refreshed);
    CHECK(refreshed.uploaded_version.cpu_write > first.uploaded_version.cpu_write);
    if (refreshed_native) {
        CHECK(refreshed_native->allocation() != first_native->allocation());
        CHECK(sampling.sample(asset, *refreshed_native, "sample_2d_array"));
    }
    statistics = metal::texture_cache_statistics(*fixture.cache);
    CHECK(statistics.uploads == 2);

    const std::uint64_t prior_epoch = fixture.memory->cpu_cache_epoch();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() == prior_epoch + 1);
    const ResidentTexture after_boundary =
        require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto boundary_native = native_texture(after_boundary);
    if (boundary_native) {
        CHECK(boundary_native->allocation() == refreshed_native->allocation());
    }
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 2);
    std::puts("Verified upload reuse, view ranges, aliases, versions and epochs.");
}

void test_cpu_epoch_snapshots(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    auto asset = fixture.add_texture(TextureType::Texture2DArray, 4, 4, 1, 3, 3);
    auto resident = require_value(fixture.cache->acquire(fixture.request(asset)));
    auto native = native_texture(resident);
    const auto initial = metal::texture_cache_statistics(*fixture.cache);
    // Certificates replaced the packed baseline in residency accounting: only
    // the native allocation is charged.
    CHECK(initial.resident_bytes == native->allocation().allocatedSize);
    for (unsigned iteration = 0; iteration < 4; ++iteration) {
        CHECK(succeeded(fixture.memory->submission_boundary()));
        const unsigned reads = fixture.backend->reads;
        auto reused = require_value(fixture.cache->acquire(fixture.request(asset)));
        CHECK(native_texture(reused)->allocation() == native->allocation());
        CHECK(fixture.backend->reads == reads + 1);
        // A second acquisition in this epoch needs no backing read.
        require_value(fixture.cache->acquire(fixture.request(asset)));
        CHECK(fixture.backend->reads == reads + 1);
        const auto stats = metal::texture_cache_statistics(*fixture.cache);
        CHECK(stats.uploads == initial.uploads);
        CHECK(stats.upload_bytes == initial.upload_bytes);
        CHECK(stats.resident_bytes == initial.resident_bytes);
    }

    // Changes outside the first mip/layer must replace storage from the same
    // single snapshot used for comparison, while older consumers stay intact.
    for (const auto [level, layer] : {std::pair{2u, 0u}, std::pair{0u, 2u}}) {
        const auto old_asset = asset;
        const auto old_resident = resident;
        const auto old_native = native;
        const auto before = metal::texture_cache_statistics(*fixture.cache);
        const auto& mip = asset.layout.mips[level];
        const std::array<std::uint8_t, 4> changed{31, 97, 173, 255};
        std::vector<std::byte> linear(mip.linear_bytes);
        for (std::size_t offset = 0; offset < linear.size(); ++offset)
            linear[offset] = std::byte{changed[offset % 4]};
        auto* bytes = fixture.backend->find(asset.retained.bytes.backing);
        CHECK(bytes != nullptr);
        store_fixture_level(asset.layout, linear, bytes->bytes, level, layer);
        auto& expected = asset.expected[layer * asset.descriptor.levels + level];
        std::fill(expected.begin(), expected.end(), changed);
        CHECK(succeeded(fixture.memory->submission_boundary()));
        const unsigned reads = fixture.backend->reads;
        auto subset = fixture.request(asset);
        subset.key.level_count = 1;
        subset.key.layer_count = 1;
        const auto subview = require_value(fixture.cache->acquire(subset));
        resident = require_value(fixture.cache->acquire(fixture.request(asset)));
        native = native_texture(resident);
        CHECK(fixture.backend->reads == reads + 1);
        CHECK(native_texture(subview)->allocation() == native->allocation());
        CHECK(native->allocation() != old_native->allocation());
        CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == before.uploads + 1);
        CHECK(sampling.sample(asset, *native, "sample_2d_array"));
        CHECK(sampling.sample(old_asset, *old_native, "sample_2d_array"));
        CHECK(resident.uploaded_version == old_resident.uploaded_version);
    }

    const auto before_store = metal::texture_cache_statistics(*fixture.cache);
    const auto old_native = native;
    const auto prior_version = resident.uploaded_version;
    auto write = require_value(fixture.memory->prepare_cpu_write(asset.retained.bytes));
    write->commit();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const unsigned reads = fixture.backend->reads;
    resident = require_value(fixture.cache->acquire(fixture.request(asset)));
    native = native_texture(resident);
    CHECK(fixture.backend->reads == reads + 1);
    CHECK(native->allocation() != old_native->allocation());
    CHECK(resident.uploaded_version.cpu_write > prior_version.cpu_write);
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == before_store.uploads + 1);
    CHECK(sampling.sample(asset, *native, "sample_2d_array"));
    CHECK(sampling.sample(asset, *old_native, "sample_2d_array"));
    std::puts("Verified unchanged CPU epochs, later mip/layer changes and tracked same-value stores.");
}

void test_cpu_epoch_padding_and_budget(id<MTLDevice> device) {
    Fixture fixture(device);
    const auto asset = fixture.add_texture(TextureType::Texture2D, 19, 3, 1, 1, 1,
                                           TextureLayout::Pitch);
    const auto first = require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto original = native_texture(first);
    auto* bytes = fixture.backend->find(asset.retained.bytes.backing);
    const auto baseline = bytes->bytes;
    // Only change padding in a later row; texel sampling alone cannot detect it.
    const auto padding = asset.layout.pitch_bytes + asset.layout.mips[0].linear_row_bytes;
    CHECK(padding < asset.descriptor.backing_extent);
    bytes->bytes[padding] ^= std::byte{1};
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const unsigned reads = fixture.backend->reads;
    const auto replacement = require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto native = native_texture(replacement);
    CHECK(fixture.backend->reads == reads + 1);
    CHECK(native->allocation() != original->allocation());
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 2);
    std::vector<std::byte> packed(asset.descriptor.backing_extent);
    CHECK(succeeded(native->read_gpu(asset.retained.bytes, packed)));
    CHECK(std::equal(packed.begin(), packed.end(), bytes->bytes.begin()));
    CHECK(succeeded(original->read_gpu(asset.retained.bytes, packed)));
    CHECK(std::equal(packed.begin(), packed.end(), baseline.begin()));

    // Residency accounting charges only the native allocations now; the
    // established readback base is host-side bookkeeping.
    const auto cost = original->allocation().allocatedSize;
    CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes == cost * 2);
    auto tight_cache = require_value(metal::create_texture_cache(
        fixture.context, fixture.memory, cost - 1));
    CHECK(is_error(tight_cache->acquire(fixture.request(asset))));
    CHECK(metal::texture_cache_statistics(*tight_cache).resident_bytes == 0);
    CHECK(metal::texture_cache_statistics(*tight_cache).uploads == 0);
    CHECK(is_error(fixture.cache->trim(cost * 2 - 1), CommandStatus::Stopped));
    std::puts("Verified row-padding snapshots, retained old consumers and packed-baseline budget accounting.");
}

// Stamp certificates: a proven-clean stamp skips every guest access, a
// suspected stamp hashes one snapshot, and only the suspected layer/level
// unit whose hash differs re-uploads — in place while no consumer pins the
// storage.
void test_stamp_certificates(id<MTLDevice> device) {
    std::shared_ptr<StampRegistry> tracked;
    Fixture fixture(device, 32u * 1024u * 1024u, [&](auto inner) {
        tracked = std::make_shared<StampRegistry>(std::move(inner));
        return tracked;
    });
    SamplingHarness sampling(fixture.context);
    auto asset = fixture.add_texture(TextureType::Texture2DArray, 4, 4, 1, 3, 3);
    auto resident = require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto allocation = native_texture(resident)->allocation();
    const auto established = metal::texture_cache_statistics(*fixture.cache);
    const auto fast_hits = metal::texture_cache_fast_path_hits(*fixture.cache);

    // The stamp proof from the upload skips the revalidation read entirely.
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const unsigned reads = fixture.backend->reads;
    auto reused = require_value(fixture.cache->acquire(fixture.request(asset)));
    CHECK(native_texture(reused)->allocation() == allocation);
    CHECK(fixture.backend->reads == reads);
    auto stats = metal::texture_cache_statistics(*fixture.cache);
    CHECK(stats.revalidations_skipped == established.revalidations_skipped);
    CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == fast_hits + 1);
    CHECK(stats.uploads == established.uploads);
    CHECK(stats.reuploads == 0);

    // A whole-range store suspicion hashes one snapshot; identical bytes
    // re-upload nothing and refresh the certificate.
    tracked->touch();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const unsigned hashed_reads = fixture.backend->reads;
    auto hashed = require_value(fixture.cache->acquire(fixture.request(asset)));
    CHECK(native_texture(hashed)->allocation() == allocation);
    CHECK(fixture.backend->reads == hashed_reads + 1);
    stats = metal::texture_cache_statistics(*fixture.cache);
    CHECK(stats.revalidations_hashed == 1);
    CHECK(stats.reuploads == 0);
    // The refreshed certificate answers clean across the next boundary.
    CHECK(succeeded(fixture.memory->submission_boundary()));
    require_value(fixture.cache->acquire(fixture.request(asset)));
    CHECK(fixture.backend->reads == hashed_reads + 1);
    CHECK(metal::texture_cache_statistics(*fixture.cache).revalidations_skipped ==
          established.revalidations_skipped);
    CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == fast_hits + 2);

    // Only the suspected layer/level unit is hashed and re-uploaded, in
    // place, once every consumer view is released.
    const std::uint32_t level = 1, layer = 2;
    const auto& mip = asset.layout.mips[level];
    std::vector<std::byte> linear(mip.linear_bytes);
    const std::array<std::uint8_t, 4> changed{17, 89, 5, 200};
    for (std::size_t offset = 0; offset < linear.size(); ++offset)
        linear[offset] = std::byte{changed[offset % 4]};
    auto* bytes = fixture.backend->find(asset.retained.bytes.backing);
    CHECK(bytes != nullptr);
    store_fixture_level(asset.layout, linear, bytes->bytes, level, layer);
    std::fill(asset.expected[std::size_t{layer} * asset.descriptor.levels + level].begin(),
              asset.expected[std::size_t{layer} * asset.descriptor.levels + level].end(),
              changed);
    const std::size_t start = asset.layout.layer_stride * layer + mip.offset;
    const std::size_t end = level + 1 < asset.descriptor.levels
        ? asset.layout.layer_stride * layer + asset.layout.mips[level + 1].offset
        : asset.layout.layer_stride * (layer + 1);
    tracked->touch({asset.retained.bytes.backing, start, end - start});
    CHECK(succeeded(fixture.memory->submission_boundary()));
    resident = {};
    reused = {};
    hashed = {};
    const auto before = metal::texture_cache_statistics(*fixture.cache);
    const unsigned patched_reads = fixture.backend->reads;
    auto patched = require_value(fixture.cache->acquire(fixture.request(asset)));
    CHECK(fixture.backend->reads == patched_reads + 1);
    CHECK(native_texture(patched)->allocation() == allocation);
    stats = metal::texture_cache_statistics(*fixture.cache);
    CHECK(stats.uploads == before.uploads + 1);
    CHECK(stats.upload_bytes == before.upload_bytes + (end - start));
    CHECK(stats.reuploads == 1);
    CHECK(stats.reupload_bytes == end - start);
    CHECK(sampling.sample(asset, *native_texture(patched), "sample_2d_array"));
    std::puts("Verified stamp certificates: skip, hash-only and single-layer re-upload.");
}

// Without a tracker the registry answers written-since for every range:
// epoch revalidation hashes once and never reports a stamp skip.
void test_default_registry_hashes_revalidations(id<MTLDevice> device) {
    Fixture fixture(device);
    auto asset = fixture.add_texture(TextureType::Texture2DArray, 4, 4, 1, 2, 2);
    auto resident = require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto allocation = native_texture(resident)->allocation();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const unsigned reads = fixture.backend->reads;
    auto reused = require_value(fixture.cache->acquire(fixture.request(asset)));
    CHECK(native_texture(reused)->allocation() == allocation);
    CHECK(fixture.backend->reads == reads + 1);
    const auto stats = metal::texture_cache_statistics(*fixture.cache);
    CHECK(stats.revalidations_hashed == 1);
    CHECK(stats.revalidations_skipped == 0);
    CHECK(stats.reuploads == 0);
    std::puts("Verified default registry hashing without stamp skips.");
}

struct SubmittedWrite {
    explicit SubmittedWrite(std::uint64_t sequence,
                            std::vector<RetainedResource> resources)
        : completion({0x4c34, sequence}, std::move(resources),
                     std::make_shared<NoopActions>()),
          token(completion.token()) {}

    SubmissionCompletion completion;
    CompletionToken token;
};

bool clear_texture(const std::shared_ptr<Context>& context,
                   const metal::TextureResource& texture,
                   SubmittedWrite& submission, MTLClearColor color,
                   bool depth = false, double clear_depth = 1.0,
                   bool stencil_only = false,
                   std::uint32_t clear_stencil = 0,
                   const std::function<void()>& while_pending = {}) {
    id<MTL4CommandAllocator> allocator = [context->device() newCommandAllocator];
    id<MTL4CommandBuffer> commands = [context->device() newCommandBuffer];
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.initialCapacity = 1;
    NSError* error = nil;
    id<MTLResidencySet> residency = [context->device()
        newResidencySetWithDescriptor:descriptor error:&error];
    if (!allocator || !commands || !residency) return false;
    [residency addAllocation:texture.allocation()];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    if (depth) {
        pass.depthAttachment.texture = texture.texture();
        pass.depthAttachment.loadAction =
            stencil_only ? MTLLoadActionLoad : MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.depthAttachment.clearDepth = clear_depth;
        pass.stencilAttachment.texture = texture.texture();
        pass.stencilAttachment.loadAction =
            stencil_only ? MTLLoadActionClear : MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
        pass.stencilAttachment.clearStencil = clear_stencil;
    } else {
        pass.colorAttachments[0].texture = texture.texture();
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = color;
    }
    id<MTL4RenderCommandEncoder> encoder =
        [commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) return false;
    [encoder endEncoding];
    [commands endCommandBuffer];
    NSArray* retained = @[allocator, commands, residency, texture.texture(),
                          texture.allocation()];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        const CommandResult result = feedback.error
            ? CommandResult{CommandStatus::Stopped,
                            "synthetic texture clear failed"}
            : CommandResult{};
        CHECK(succeeded(submission.completion.feedback(result)));
        dispatch_semaphore_signal(done);
    }];
    CHECK(succeeded(submission.completion.submitted()));
    id<MTL4CommandBuffer> submitted[] = {commands};
    [context->queue() commit:submitted count:1 options:options];
    if (while_pending) {
        CHECK(dispatch_semaphore_wait(done, DISPATCH_TIME_NOW) != 0);
        while_pending();
    }
    if (dispatch_semaphore_wait(
            done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0 ||
        execution_error) {
        return false;
    }
    CHECK(succeeded(submission.completion.finish()));
    return true;
}

void commit_gpu_write(Fixture& fixture, const TextureAsset& asset,
                      const ResidentTexture& resident,
                      SubmittedWrite& submission) {
    const std::array accesses{
        GpuAccessRequest{asset.retained, ResourceAccess::Write}};
    auto reservation = require_value(
        fixture.memory->reserve_accesses(accesses, submission.token));
    CHECK(succeeded(reservation->commit({resident.storage})));
}

// The bounded identical-request shortcut must be observable only while every
// live admission fact remains unchanged. These cases exercise the independent
// CPU tracker, registry GPU authority, incompatible aliases and mapping IDs.
void test_identical_request_fast_path(id<MTLDevice> device) {
    {
        std::shared_ptr<StampRegistry> tracked;
        Fixture fixture(device, 32u * 1024u * 1024u, [&](auto inner) {
            tracked = std::make_shared<StampRegistry>(std::move(inner));
            return tracked;
        });
        auto asset = fixture.add_texture(
            TextureType::Texture2D, 8, 8, 1, 1, 1);
        const TextureRequest request = fixture.request(asset);
        const ResidentTexture first = require_value(fixture.cache->acquire(request));
        const auto first_native = native_texture(first);
        for (unsigned repeat = 0; repeat < 5; ++repeat) {
            const auto resident = require_value(fixture.cache->acquire(request));
            CHECK(native_texture(resident)->allocation() == first_native->allocation());
        }
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == 5);
        tracked->touch({asset.retained.bytes.backing,
                        asset.retained.bytes.offset + asset.retained.bytes.size, 1});
        const auto clean = require_value(fixture.cache->acquire(request));
        CHECK(clean.storage == first.storage);
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == 6);
        auto malformed = request;
        malformed.key.storage.descriptor.swizzle[0] = static_cast<TextureSwizzle>(255);
        CHECK(is_error(fixture.cache->acquire(malformed)));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == 6);

        auto* backing = fixture.backend->find(asset.retained.bytes.backing);
        CHECK(backing != nullptr);
        if (backing) backing->bytes[0] ^= std::byte{1};
        tracked->touch({asset.retained.bytes.backing,
                        asset.retained.bytes.offset, 1});
        const auto before_write = metal::texture_cache_fast_path_hits(*fixture.cache);
        // A miss before the boundary must not certify older uploaded bytes
        // with the new store stamp and hide the later epoch revalidation.
        require_value(fixture.cache->acquire(request));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_write);
        CHECK(succeeded(fixture.memory->submission_boundary()));
        const auto refreshed = require_value(fixture.cache->acquire(request));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_write);
        CHECK(native_texture(refreshed)->allocation() != first_native->allocation());
        require_value(fixture.cache->acquire(request));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_write + 1);
    }

    {
        Fixture fixture(device, 32u * 1024u * 1024u, [](auto inner) {
            return std::make_shared<StampRegistry>(std::move(inner));
        });
        auto asset = fixture.add_texture(
            TextureType::Texture2D, 8, 8, 1, 1, 1);
        const TextureRequest writable =
            fixture.request(asset, ResourceAccess::Write);
        const ResidentTexture target =
            require_value(fixture.cache->acquire(writable));
        const auto native = native_texture(target);
        require_value(fixture.cache->acquire(writable));
        const auto before_write = metal::texture_cache_fast_path_hits(*fixture.cache);

        SubmittedWrite writer(90, {target.storage});
        commit_gpu_write(fixture, asset, target, writer);
        const ResidentTexture pending =
            require_value(fixture.cache->acquire(writable));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_write);
        CHECK(pending.uploaded_version.gpu_write > target.uploaded_version.gpu_write);
        CHECK(std::ranges::any_of(pending.dependencies, [&](const CompletionToken& token) {
            return token.id() == writer.token.id();
        }));
        require_value(fixture.cache->acquire(writable));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_write);
        CHECK(native && clear_texture(fixture.context, *native, writer,
                                      MTLClearColorMake(0.25, 0.5, 0.75, 1.0)));
        const ResidentTexture completed =
            require_value(fixture.cache->acquire(writable));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_write);
        CHECK(completed.dependencies.empty());
        require_value(fixture.cache->acquire(writable));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_write + 1);
    }

    {
        Fixture fixture(device, 32u * 1024u * 1024u, [](auto inner) {
            return std::make_shared<StampRegistry>(std::move(inner));
        });
        auto asset = fixture.add_texture(
            TextureType::Texture2D, 8, 8, 1, 1, 1);
        const TextureRequest sampled = fixture.request(asset);
        const ResidentTexture original =
            require_value(fixture.cache->acquire(sampled));
        require_value(fixture.cache->acquire(sampled));

        TextureAsset alias = asset;
        alias.descriptor.format = TextureFormat::R32Float;
        const TextureRequest alias_write =
            fixture.request(alias, ResourceAccess::Write);
        const ResidentTexture alias_target =
            require_value(fixture.cache->acquire(alias_write));
        const auto alias_native = native_texture(alias_target);
        SubmittedWrite alias_writer(91, {alias_target.storage});
        commit_gpu_write(fixture, alias, alias_target, alias_writer);
        CHECK(alias_native && clear_texture(
            fixture.context, *alias_native, alias_writer,
            MTLClearColorMake(0.5, 0, 0, 1)));

        const auto before_alias =
            metal::texture_cache_fast_path_hits(*fixture.cache);
        const ResidentTexture after_alias =
            require_value(fixture.cache->acquire(sampled));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_alias);
        CHECK(after_alias.uploaded_version.gpu_write >
              original.uploaded_version.gpu_write);
    }

    {
        Fixture fixture(device, 32u * 1024u * 1024u, [](auto inner) {
            return std::make_shared<StampRegistry>(std::move(inner));
        });
        const auto asset = fixture.add_texture(
            TextureType::Texture2D, 8, 8, 1, 1, 1);
        const TextureRequest request =
            fixture.request(asset, ResourceAccess::ReadWrite);
        require_value(fixture.cache->acquire(request));
        require_value(fixture.cache->acquire(request));
        const auto before_mapping =
            metal::texture_cache_fast_path_hits(*fixture.cache);
        const std::array removed{asset.retained.mapping};
        CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
        CHECK(is_error(fixture.cache->acquire(request)));
        CHECK(metal::texture_cache_fast_path_hits(*fixture.cache) == before_mapping);
    }
    std::puts("Verified bounded texture request fast hits and coherence invalidation.");
}

// A render target described differently over the same bytes (a smaller
// target of a downsampling chain) supersedes the larger target's GPU content:
// no readback, no guest write, ordered after the producer.
void test_attachment_supersedes_other_description(id<MTLDevice> device) {
    Fixture fixture(device);
    TextureAsset large = fixture.add_texture(TextureType::Texture2D, 32, 32, 1, 1, 1);
    const ResidentTexture large_target =
        require_value(fixture.cache->acquire(fixture.request(large, ResourceAccess::Write)));
    SubmittedWrite writer(1, {large_target.storage});
    commit_gpu_write(fixture, large, large_target, writer);

    TextureAsset small = fixture.add_texture(TextureType::Texture2D, 16, 16, 1, 1, 1);
    small.descriptor.address = large.descriptor.address;
    small.retained = require_value(fixture.memory->retain(
        small.descriptor.address, small.descriptor.backing_extent, ResourceAccess::ReadWrite));
    const unsigned reads = fixture.backend->reads;
    const unsigned writes = fixture.backend->writes;
    const auto readbacks = metal::texture_cache_statistics(*fixture.cache).readbacks;
    const ResidentTexture small_target =
        require_value(fixture.cache->acquire(fixture.request(small, ResourceAccess::ReadWrite)));
    CHECK(fixture.backend->reads == reads);
    CHECK(fixture.backend->writes == writes);
    CHECK(metal::texture_cache_statistics(*fixture.cache).readbacks == readbacks);
    CHECK(native_texture(small_target) &&
          native_texture(small_target)->allocation() != native_texture(large_target)->allocation());
    CHECK(std::ranges::any_of(small_target.dependencies, [&](const CompletionToken& token) {
        return token.snapshot().id == writer.token.snapshot().id;
    }));
    std::puts("Verified attachment superseding another description's GPU content.");
}

// A cube face rendered through its own 2D target at the face's bytes is
// brought into the existing cube by one planned GPU copy: no guest readback,
// no writeback and no wait for the producer, which becomes a dependency.
void test_rendered_cube_face_composes_by_copy(id<MTLDevice> device) {
    Fixture fixture(device);
    TextureAsset cube = fixture.add_texture(TextureType::Cubemap, 16, 16, 1, 1, 6);
    const ResidentTexture first = require_value(fixture.cache->acquire(fixture.request(cube)));
    CHECK(first.transfers.empty());
    const auto cube_native = native_texture(first);

    TextureAsset face = fixture.add_texture(TextureType::Texture2D, 16, 16, 1, 1, 1);
    face.descriptor.address = cube.descriptor.address + 2 * cube.layout.layer_stride;
    face.retained = require_value(fixture.memory->retain(
        face.descriptor.address, face.descriptor.backing_extent, ResourceAccess::Write));
    const ResidentTexture target =
        require_value(fixture.cache->acquire(fixture.request(face, ResourceAccess::Write)));
    SubmittedWrite writer(1, {target.storage});
    commit_gpu_write(fixture, face, target, writer);

    const unsigned reads = fixture.backend->reads;
    const unsigned writes = fixture.backend->writes;
    const ResidentTexture sampled = require_value(fixture.cache->acquire(fixture.request(cube)));
    CHECK(fixture.backend->reads == reads);
    CHECK(fixture.backend->writes == writes);
    CHECK(sampled.transfers.size() == 1);
    const auto copy = sampled.transfers.empty() ? nullptr
        : std::dynamic_pointer_cast<const metal::SubresourceCopy>(sampled.transfers[0]);
    CHECK(copy != nullptr);
    if (copy) {
        CHECK(copy->layer == 2 && copy->level == 0 && copy->width == 16 && copy->height == 16);
        CHECK(cube_native && copy->destination == cube_native->allocation());
        CHECK(copy->source == native_texture(target)->allocation());
    }
    CHECK(std::ranges::any_of(sampled.dependencies, [&](const CompletionToken& token) {
        return token.snapshot().id == writer.token.snapshot().id;
    }));
    // The plan is offered until a packet encodes it, then not repeated.
    const ResidentTexture before_encode = require_value(fixture.cache->acquire(fixture.request(cube)));
    CHECK(before_encode.transfers.size() == 1 &&
          before_encode.transfers[0] == sampled.transfers[0]);
    if (copy) copy->encoded = true;
    const ResidentTexture after_encode = require_value(fixture.cache->acquire(fixture.request(cube)));
    CHECK(after_encode.transfers.empty());
    std::puts("Verified rendered cube face composition by planned GPU copy.");
}

void test_colour_rendered_depth_layer_composes_through_staging(id<MTLDevice> device) {
    Fixture fixture(device);
    TextureAsset cascades = fixture.add_texture(TextureType::Texture2DArray, 16, 16, 1, 1, 3,
                                                TextureLayout::BlockLinear, TextureFormat::D16Unorm);
    TextureRequest depth = fixture.request(cascades);
    depth.key.aspect = TextureAspect::Depth;
    const ResidentTexture first = require_value(fixture.cache->acquire(depth));
    const auto cascades_native = native_texture(first);

    // A wider R16Unorm target with the same GOB columns renders layer 1.
    TextureAsset layer = fixture.add_texture(TextureType::Texture2D, 32, 16, 1, 1, 1,
                                             TextureLayout::BlockLinear, TextureFormat::R16Unorm);
    layer.descriptor.address = cascades.descriptor.address + cascades.layout.layer_stride;
    layer.retained = require_value(fixture.memory->retain(
        layer.descriptor.address, layer.descriptor.backing_extent, ResourceAccess::Write));
    const ResidentTexture target =
        require_value(fixture.cache->acquire(fixture.request(layer, ResourceAccess::Write)));
    SubmittedWrite writer(1, {target.storage});
    commit_gpu_write(fixture, layer, target, writer);

    const unsigned reads = fixture.backend->reads;
    const ResidentTexture sampled = require_value(fixture.cache->acquire(depth));
    CHECK(fixture.backend->reads == reads);
    CHECK(sampled.transfers.size() == 1);
    const auto copy = sampled.transfers.empty() ? nullptr
        : std::dynamic_pointer_cast<const metal::SubresourceCopy>(sampled.transfers[0]);
    CHECK(copy != nullptr);
    if (copy) {
        CHECK(copy->layer == 1 && copy->level == 0 && copy->width == 16 && copy->height == 16);
        CHECK(cascades_native && copy->destination == cascades_native->allocation());
        CHECK(copy->source.pixelFormat == MTLPixelFormatR16Unorm &&
              copy->destination.pixelFormat == MTLPixelFormatDepth16Unorm);
        CHECK(copy->staging != nil && copy->bytes_per_row >= 32 &&
              copy->staging.length >= copy->bytes_per_row * 16);
    }
    std::puts("Verified a colour-rendered depth array layer composes through a staging buffer.");
}

void verify_render_target_writeback(Fixture& fixture, TextureAsset& asset,
                                    std::uint32_t layer,
                                    std::span<const std::byte> clear_texel,
                                    std::uint64_t sequence) {
    auto* backing = fixture.backend->find(asset.retained.bytes.backing);
    CHECK(backing != nullptr);
    if (!backing || clear_texel.empty()) return;

    const auto before = backing->bytes;
    auto expected = before;
    const auto& mip = asset.layout.mips[0];
    std::vector<std::byte> linear(mip.linear_bytes);
    CHECK(linear.size() % clear_texel.size() == 0);
    for (std::size_t offset = 0; offset < linear.size();
         offset += clear_texel.size()) {
        std::copy(clear_texel.begin(), clear_texel.end(),
                  linear.begin() + offset);
    }
    store_fixture_level(asset.layout, linear, expected, 0, layer);

    std::vector<std::byte> payload_mask(asset.layout.total_size);
    std::vector<std::byte> payload(mip.linear_bytes, std::byte{0xff});
    store_fixture_level(asset.layout, payload, payload_mask, 0, layer);

    TextureRequest writable = fixture.request(asset, ResourceAccess::Write);
    writable.key.first_layer = layer;
    writable.key.layer_count = 1;
    // Render-target state and a later TIC carry different non-storage words.
    // Compatible storage must still retain one native allocation.
    writable.key.storage.descriptor.raw_words[0] = 0x52540000u | layer;
    writable.key.storage.descriptor.normalized_coordinates = false;
    const ResidentTexture target =
        require_value(fixture.cache->acquire(writable));
    const auto target_native = native_texture(target);
    CHECK(target_native != nullptr);
    if (!target_native) return;

    SubmittedWrite writer(sequence, {target.storage});
    commit_gpu_write(fixture, asset, target, writer);
    CHECK(clear_texture(fixture.context, *target_native, writer,
                        MTLClearColorMake(0.25, 0.5, 0.75, 1.0)));

    const ResidentTexture sampled =
        require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto sampled_native = native_texture(sampled);
    CHECK(sampled_native &&
          sampled_native->allocation() == target_native->allocation());
    CHECK(succeeded(fixture.cache->resolve(writable.key,
                                           ResolveReason::Capture)));
    CHECK(backing->bytes == expected);
    for (std::size_t offset = 0; offset < before.size(); ++offset) {
        if (offset >= payload_mask.size() ||
            payload_mask[offset] == std::byte{0}) {
            CHECK(backing->bytes[offset] == before[offset]);
        }
    }
}

void test_render_target_layout_writeback(id<MTLDevice> device) {
    Fixture fixture(device);
    const std::array<std::byte, 4> rgba{
        std::byte{64}, std::byte{128}, std::byte{191}, std::byte{255}};

    // Height 129 prevents the layout calculator from shrinking any requested
    // GOB block height, so this covers every effective RT block height.
    for (std::uint32_t block_height_log2 = 0; block_height_log2 <= 5;
         ++block_height_log2) {
        const std::uint32_t layers = block_height_log2 == 5 ? 2 : 1;
        TextureAsset asset = fixture.add_texture(
            layers == 1 ? TextureType::Texture2D
                        : TextureType::Texture2DArray,
            19, 129, 1, 1, layers, TextureLayout::BlockLinear,
            TextureFormat::R8G8B8A8Unorm, block_height_log2);
        CHECK(asset.layout.mips[0].gob_block_height_log2 ==
              block_height_log2);
        verify_render_target_writeback(
            fixture, asset, layers - 1, rgba, 100 + block_height_log2);
    }

    TextureAsset pitch = fixture.add_texture(
        TextureType::Texture2DArray, 19, 13, 1, 1, 2,
        TextureLayout::Pitch);
    CHECK(pitch.layout.pitch_bytes == 128);
    CHECK(pitch.layout.mips[0].linear_row_bytes == 76);
    verify_render_target_writeback(fixture, pitch, 1, rgba, 106);

    // A depth-one 3D target retains the encoded depth block. Metal uses
    // depth plane zero, while guest rows retain their block-depth Y strides.
    TextureAsset depth_one = fixture.add_texture(
        TextureType::Texture3D, 64, 16, 1, 1, 1,
        TextureLayout::BlockLinear, TextureFormat::R8Unorm, 0, 4);
    CHECK(depth_one.layout.total_size == 8704);
    const std::array<std::byte, 1> red{std::byte{64}};
    verify_render_target_writeback(fixture, depth_one, 0, red, 107);
    const auto* backing = fixture.backend->find(depth_one.retained.bytes.backing);
    CHECK(backing != nullptr);
    if (backing) {
        CHECK(std::all_of(backing->bytes.begin(), backing->bytes.begin() + 512,
                          [](std::byte value) { return value == std::byte{64}; }));
        CHECK(std::all_of(backing->bytes.begin() + 512,
                          backing->bytes.begin() + 8192,
                          [](std::byte value) { return value == std::byte{0xd3}; }));
        CHECK(std::all_of(backing->bytes.begin() + 8192,
                          backing->bytes.begin() + 8704,
                          [](std::byte value) { return value == std::byte{64}; }));
        CHECK(backing->bytes[0] == std::byte{64});
        CHECK(backing->bytes[511] == std::byte{64});
        CHECK(backing->bytes[8192] == std::byte{64});
        CHECK(backing->bytes[8703] == std::byte{64});
    }
    std::puts("Verified RT pitch padding, every GOB height, array stride and depth-one 3D writeback.");
}

void test_render_to_sample_and_incompatible_alias(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2D, 4, 4, 1, 1, 1);
    TextureRequest writable = fixture.request(asset, ResourceAccess::Write);
    const ResidentTexture target = require_value(fixture.cache->acquire(writable));
    const auto native = native_texture(target);
    SubmittedWrite writer(1, {target.storage});
    commit_gpu_write(fixture, asset, target, writer);
    const unsigned pending_reads = fixture.backend->reads;
    const unsigned pending_writes = fixture.backend->writes;
    const auto pending_stats =
        metal::texture_cache_statistics(*fixture.cache);
    const ResidentTexture pending_sample =
        require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto pending_native = native_texture(pending_sample);
    CHECK(pending_sample.dependencies.size() == 1);
    if (pending_sample.dependencies.size() == 1) {
        CHECK(pending_sample.dependencies[0].snapshot().id ==
              writer.token.snapshot().id);
        CHECK(pending_sample.dependencies[0].snapshot().phase ==
              SubmissionPhase::Accepted);
    }
    CHECK(pending_native && native &&
          pending_native->allocation() == native->allocation());
    CHECK(fixture.backend->reads == pending_reads);
    CHECK(fixture.backend->writes == pending_writes);
    const auto after_pending =
        metal::texture_cache_statistics(*fixture.cache);
    CHECK(after_pending.uploads == pending_stats.uploads);
    CHECK(after_pending.upload_bytes == pending_stats.upload_bytes);
    CHECK(after_pending.readbacks == pending_stats.readbacks);
    const MTLClearColor clear = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
    CHECK(native && clear_texture(fixture.context, *native, writer, clear));

    const unsigned reads_before = fixture.backend->reads;
    const unsigned writes_before = fixture.backend->writes;
    const auto stats_before = metal::texture_cache_statistics(*fixture.cache);
    const ResidentTexture sampled =
        require_value(fixture.cache->acquire(fixture.request(asset)));
    const auto sampled_native = native_texture(sampled);
    CHECK(fixture.backend->reads == reads_before);
    CHECK(fixture.backend->writes == writes_before);
    CHECK(metal::texture_cache_statistics(*fixture.cache).readbacks ==
          stats_before.readbacks);
    CHECK(sampled_native && sampled_native->allocation() == native->allocation());
    std::fill(asset.expected[0].begin(), asset.expected[0].end(),
              std::array<std::uint8_t, 4>{64, 128, 191, 255});
    if (pending_native) {
        CHECK(sampling.sample(asset, *pending_native, "sample_2d"));
    }
    if (sampled_native) {
        CHECK(sampling.sample(asset, *sampled_native, "sample_2d"));
    }

    // The producer has finished, so an incompatible alias writes the
    // rendered bytes back for its own range and uploads a new allocation.
    TextureRequest incompatible = fixture.request(asset);
    incompatible.key.storage.descriptor.format = TextureFormat::R32Uint;
    const ResidentTexture reinterpreted =
        require_value(fixture.cache->acquire(incompatible));
    CHECK(fixture.backend->writes == writes_before + 1);
    CHECK(metal::texture_cache_statistics(*fixture.cache).readbacks ==
          stats_before.readbacks + 1);
    const auto reinterpreted_native = native_texture(reinterpreted);
    CHECK(reinterpreted_native != nullptr);
    CHECK(reinterpreted_native->allocation() != native->allocation());
    std::puts("Verified render-to-sample reuse and explicit alias resolution.");
}

// Depth-only clear for non-stencil depth storage (D16Unorm/D32Float), which
// Metal refuses to pair with a stencil attachment.
bool clear_depth_texture(const std::shared_ptr<Context>& context,
                         const metal::TextureResource& texture,
                         SubmittedWrite& submission, double depth) {
    id<MTL4CommandAllocator> allocator = [context->device() newCommandAllocator];
    id<MTL4CommandBuffer> commands = [context->device() newCommandBuffer];
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.initialCapacity = 1;
    id<MTLResidencySet> residency = [context->device()
        newResidencySetWithDescriptor:descriptor error:nil];
    if (!allocator || !commands || !residency) return false;
    [residency addAllocation:texture.allocation()];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    pass.depthAttachment.texture = texture.texture();
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.depthAttachment.clearDepth = depth;
    id<MTL4RenderCommandEncoder> encoder =
        [commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) return false;
    [encoder endEncoding];
    [commands endCommandBuffer];
    NSArray* retained = @[allocator, commands, residency, texture.texture(),
                          texture.allocation()];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        const CommandResult result = feedback.error
            ? CommandResult{CommandStatus::Stopped,
                            "synthetic depth clear failed"}
            : CommandResult{};
        CHECK(succeeded(submission.completion.feedback(result)));
        dispatch_semaphore_signal(done);
    }];
    CHECK(succeeded(submission.completion.submitted()));
    id<MTL4CommandBuffer> submitted[] = {commands};
    [context->queue() commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(
            done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0 ||
        execution_error) {
        return false;
    }
    CHECK(succeeded(submission.completion.finish()));
    return true;
}

// Runs the probe_depth kernel over one full level and checks the comparison
// results and plain depth reads against the caller's per-texel expectations.
bool probe_depth(const SamplingHarness& harness,
                 const metal::TextureResource& texture, std::uint32_t width,
                 std::uint32_t height, float reference,
                 const std::vector<float>& expected_compare,
                 const std::vector<float>& expected_value,
                 float value_tolerance) {
    std::vector<Probe> probes;
    probes.reserve(std::size_t{width} * height);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
            probes.push_back({0, 0, 0, x, y,
                (static_cast<float>(x) + 0.5f) / width,
                (static_cast<float>(y) + 0.5f) / height, reference});
    id<MTLComputePipelineState> state = harness.pipeline("probe_depth");
    id<MTLBuffer> probe_buffer = [harness.context->device()
        newBufferWithBytes:probes.data() length:probes.size() * sizeof(Probe)
                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [harness.context->device()
        newBufferWithLength:probes.size() * sizeof(std::array<float, 4>)
                    options:MTLResourceStorageModeShared];
    MTL4ArgumentTableDescriptor* table_descriptor =
        [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    table_descriptor.maxTextureBindCount = 1;
    id<MTL4ArgumentTable> table = [harness.context->device()
        newArgumentTableWithDescriptor:table_descriptor error:nil];
    MTLResidencySetDescriptor* residency_descriptor =
        [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 3;
    id<MTLResidencySet> residency = [harness.context->device()
        newResidencySetWithDescriptor:residency_descriptor error:nil];
    id<MTL4CommandAllocator> allocator =
        [harness.context->device() newCommandAllocator];
    id<MTL4CommandBuffer> commands =
        [harness.context->device() newCommandBuffer];
    if (!state || !probe_buffer || !output || !table || !residency ||
        !allocator || !commands) {
        std::fputs("Depth probe dispatch allocation failed.\n", stderr);
        return false;
    }
    [table setAddress:probe_buffer.gpuAddress atIndex:0];
    [table setAddress:output.gpuAddress atIndex:1];
    [table setTexture:texture.texture().gpuResourceID atIndex:0];
    [residency addAllocation:texture.allocation()];
    [residency addAllocation:probe_buffer];
    [residency addAllocation:output];
    [residency commit];
    NSArray* retained = @[state, probe_buffer, output, table, residency,
                          allocator, commands, texture.texture(),
                          texture.allocation()];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    if (!encoder) return false;
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch
                  visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder setComputePipelineState:state];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(probes.size(), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(probes.size(), 32),
                                           1, 1)];
    [encoder endEncoding];
    [commands endCommandBuffer];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* commit = [MTL4CommitOptions new];
    [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(done);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [harness.context->queue() commit:submitted count:1 options:commit];
    if (dispatch_semaphore_wait(
            done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0 ||
        execution_error) {
        std::fprintf(stderr, "Depth probe execution failed: %s\n",
                     execution_error.localizedDescription.UTF8String);
        return false;
    }
    const auto* sampled = static_cast<const std::array<float, 4>*>(output.contents);
    for (std::size_t index = 0; index < probes.size(); ++index) {
        if (std::fabs((*sampled)[0] - expected_compare[index]) > 0.25f) {
            std::fprintf(stderr, "Depth compare probe %zu: expected %.6f, got %.6f.\n",
                         index, expected_compare[index], (*sampled)[0]);
            ++failures;
        }
        if (!std::isfinite((*sampled)[1]) ||
            std::fabs((*sampled)[1] - expected_value[index]) > value_tolerance) {
            std::fprintf(stderr, "Depth read probe %zu: expected %.6f, got %.6f.\n",
                         index, expected_value[index], (*sampled)[1]);
            ++failures;
        }
        ++sampled;
    }
    return true;
}

// A colour-coded sampler
// descriptor whose format aliases the depth storage (D16Unorm/R16Unorm,
// D32Float/R32Float) resolves onto the existing depth target's allocation —
// no copy, no re-upload, GPU-newer authority kept with its producer token.
void test_depth_alias_sampler_lookup(id<MTLDevice> device) {
    for (const auto& pair : std::vector<std::pair<TextureFormat, TextureFormat>>{
             {TextureFormat::D16Unorm, TextureFormat::R16Unorm},
             {TextureFormat::D32Float, TextureFormat::R32Float}}) {
        Fixture fixture(device);
        TextureAsset asset = fixture.add_texture(
            TextureType::Texture2D, 8, 8, 1, 1, 1, TextureLayout::BlockLinear,
            pair.first);
        TextureRequest writable = fixture.request(asset, ResourceAccess::Write);
        writable.key.aspect = TextureAspect::Depth;
        const ResidentTexture target = require_value(fixture.cache->acquire(writable));
        const auto native = native_texture(target);
        CHECK(native && native->allocation().pixelFormat ==
                  (pair.first == TextureFormat::D16Unorm
                       ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float));
        SubmittedWrite writer(70, {target.storage});
        commit_gpu_write(fixture, asset, target, writer);
        CHECK(native && clear_depth_texture(fixture.context, *native, writer, 0.25));
        TextureRequest sampled = fixture.request(asset);
        sampled.key.storage.descriptor.format = pair.second;
        sampled.alias = TextureAliasScope::Sampler;
        const unsigned reads_before = fixture.backend->reads;
        const unsigned writes_before = fixture.backend->writes;
        const auto stats_before = metal::texture_cache_statistics(*fixture.cache);
        const ResidentTexture aliased = require_value(fixture.cache->acquire(sampled));
        const auto aliased_native = native_texture(aliased);
        CHECK(fixture.backend->reads == reads_before);
        CHECK(fixture.backend->writes == writes_before);
        const auto after = metal::texture_cache_statistics(*fixture.cache);
        CHECK(after.uploads == stats_before.uploads &&
              after.readbacks == stats_before.readbacks);
        CHECK(aliased_native && native &&
              aliased_native->allocation() == native->allocation());
        // The clear has completed on this queue, so no pending producer
        // dependency is carried; a still-accepted write would have to be.
        for (const auto& dependency : aliased.dependencies) {
            CHECK(dependency.snapshot().id == writer.token.snapshot().id);
        }
        // The colour-coded request binds the depth storage representation:
        // Metal has no colour view of a depth texture.
        if (aliased_native) {
            CHECK(aliased_native->texture().pixelFormat ==
                  (pair.first == TextureFormat::D16Unorm
                       ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float));
        }
        // Without the sampler scope the finished GPU-newer depth is written
        // back and the colour request gets its own allocation.
        TextureRequest unaliased = sampled;
        unaliased.alias = TextureAliasScope::None;
        const auto unaliased_result = fixture.cache->acquire(unaliased);
        CHECK(std::holds_alternative<ResidentTexture>(unaliased_result));
        if (const auto* resident = std::get_if<ResidentTexture>(&unaliased_result)) {
            const auto native = native_texture(*resident);
            if (native && aliased_native) CHECK(native->allocation() != aliased_native->allocation());
        }
    }
    std::puts("Verified sampler depth/colour alias reuse of depth target storage.");
}

// Compare sampling through a colour-coded descriptor equals a direct depth
// compare: rendered depth storage stays authoritative, and CPU-uploaded
// colour-format bytes are allocated and sampled as their depth equivalent.
void test_depth_alias_compare_sampling(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    // CPU-uploaded R16Unorm data compare-sampled through the rewritten
    // depth-equivalent key gives the depth interpretation of the same bytes.
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2D, 8, 8, 1, 1, 1, TextureLayout::BlockLinear,
        TextureFormat::R16Unorm);
    Backend::Allocation* bytes = fixture.backend->find(asset.retained.bytes.backing);
    CHECK(bytes != nullptr);
    const auto& mip = asset.layout.mips[0];
    std::vector<std::byte> linear(mip.linear_bytes);
    std::vector<float> expected_compare(std::size_t{mip.width} * mip.height);
    std::vector<float> expected_value(expected_compare.size());
    for (std::size_t texel = 0; texel < expected_value.size(); ++texel) {
        const auto raw = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(texel * 2654435761u + 1) >> 16);
        linear[texel * 2] = std::byte(raw & 0xffu);
        linear[texel * 2 + 1] = std::byte(raw >> 8);
        expected_value[texel] = static_cast<float>(raw) / 65535.0f;
        expected_compare[texel] = 0.5f <= expected_value[texel] ? 1.0f : 0.0f;
    }
    if (bytes) {
        CHECK(std::holds_alternative<std::monostate>(
            write_level_from_linear(asset.layout, linear, bytes->bytes, 0, 0)));
    }
    // The compare-sampled binding arrives as the depth-equivalent storage key.
    TextureRequest depth_keyed = fixture.request(asset);
    depth_keyed.key.storage.descriptor.format = TextureFormat::D16Unorm;
    depth_keyed.key.aspect = TextureAspect::Depth;
    depth_keyed.alias = TextureAliasScope::Sampler;
    const ResidentTexture resident = require_value(fixture.cache->acquire(depth_keyed));
    const auto native = native_texture(resident);
    CHECK(native && native->allocation().pixelFormat == MTLPixelFormatDepth16Unorm);
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 1);
    if (native) {
        CHECK(probe_depth(sampling, *native, mip.width, mip.height, 0.5f,
                          expected_compare, expected_value, 1.5f / 255.0f));
    }
    // Rendered depth storage sampled through the R16Unorm descriptor with a
    // compare sampler equals the direct depth compare of the same texture.
    TextureAsset target = fixture.add_texture(
        TextureType::Texture2D, 8, 8, 1, 1, 1, TextureLayout::BlockLinear,
        TextureFormat::D16Unorm);
    TextureRequest writable = fixture.request(target, ResourceAccess::Write);
    writable.key.aspect = TextureAspect::Depth;
    const ResidentTexture written = require_value(fixture.cache->acquire(writable));
    const auto written_native = native_texture(written);
    SubmittedWrite writer(71, {written.storage});
    commit_gpu_write(fixture, target, written, writer);
    CHECK(written_native &&
          clear_depth_texture(fixture.context, *written_native, writer, 0.25));
    TextureRequest coded = fixture.request(target);
    coded.key.storage.descriptor.format = TextureFormat::R16Unorm;
    coded.alias = TextureAliasScope::Sampler;
    const ResidentTexture aliased = require_value(fixture.cache->acquire(coded));
    const auto aliased_native = native_texture(aliased);
    CHECK(aliased_native && written_native &&
          aliased_native->allocation() == written_native->allocation());
    const auto quantized = static_cast<float>(std::lround(0.25 * 65535.0)) / 65535.0f;
    const auto probes = std::size_t{mip.width} * mip.height;
    if (aliased_native) {
        CHECK(probe_depth(sampling, *aliased_native, mip.width, mip.height, 0.2f,
                          std::vector<float>(probes, 1.0f),
                          std::vector<float>(probes, quantized), 1.5f / 255.0f));
        CHECK(probe_depth(sampling, *aliased_native, mip.width, mip.height, 0.3f,
                          std::vector<float>(probes, 0.0f),
                          std::vector<float>(probes, quantized), 1.5f / 255.0f));
    }
    std::puts("Verified compare sampling through colour-coded depth-alias descriptors.");
}

// A depth-aliased allocation keeps exactly its own TIC's layout: a colour-
// coded descriptor whose GOB block height and extent differ names a different
// guest layout, and serving it from the padded depth allocation converted a
// source span sized by the second TIC through the first TIC's layout ("texture
// upload layout conversion failed"). The alias may change the pixel format
// only; a different layout gets its own allocation.
void test_depth_alias_layout_matches_descriptor(id<MTLDevice> device) {
    std::shared_ptr<StampRegistry> tracked;
    Fixture fixture(device, 32u * 1024u * 1024u, [&](auto inner) {
        tracked = std::make_shared<StampRegistry>(std::move(inner));
        return tracked;
    });
    // The compare-sampled bind arrives as the depth-equivalent key of a
    // block-linear 192x192 R16Unorm TIC whose block height pads the storage
    // to 256 rows; CPU contents establish it with no prior depth storage.
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2D, 192, 192, 1, 1, 1, TextureLayout::BlockLinear,
        TextureFormat::R16Unorm, 4);
    CHECK(asset.descriptor.backing_extent == 98304);
    TextureRequest depth_keyed = fixture.request(asset);
    depth_keyed.key.storage.descriptor.format = TextureFormat::D16Unorm;
    depth_keyed.key.aspect = TextureAspect::Depth;
    depth_keyed.alias = TextureAliasScope::Sampler;
    ResidentTexture depth_resident =
        require_value(fixture.cache->acquire(depth_keyed));
    auto depth_native = native_texture(depth_resident);
    CHECK(depth_native &&
          depth_native->allocation().pixelFormat == MTLPixelFormatDepth16Unorm);
    const id<MTLTexture> depth_allocation =
        depth_native ? depth_native->allocation() : nil;
    // An identical TIC without the compare sampler shares the depth
    // allocation: the alias changes the pixel format alone.
    TextureRequest same_tic = fixture.request(asset);
    same_tic.alias = TextureAliasScope::Sampler;
    ResidentTexture shared = require_value(fixture.cache->acquire(same_tic));
    {
        const auto shared_view = native_texture(shared);
        CHECK(shared_view && shared_view->allocation() == depth_allocation);
    }
    shared = {};
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 1);
    // A second TIC over the same address names the unpadded layout: block
    // height one keeps 192 rows, so its extent is the unpadded 73,728 bytes.
    TextureDescriptor coded = asset.descriptor;
    coded.format = TextureFormat::R16Unorm;
    coded.block_height_log2 = 1;
    const auto coded_layout = require_layout(texture_view_layout(coded));
    CHECK(coded_layout.total_size == 73728);
    coded.backing_extent = coded_layout.total_size;
    coded.row_bytes =
        (coded_layout.mips[0].linear_row_bytes + 63) & ~std::size_t{63};
    const auto alignment = std::size_t{8} << 1;
    coded.storage_height =
        (coded_layout.mips[0].block_rows + alignment - 1) & ~(alignment - 1);
    coded.layer_stride = coded_layout.layer_stride;
    // New CPU bytes and a registry suspicion drive the revalidation that used
    // to feed the smaller source span into the padded allocation's layout.
    auto* bytes = fixture.backend->find(asset.retained.bytes.backing);
    CHECK(bytes != nullptr);
    if (bytes) {
        std::fill(bytes->bytes.begin(), bytes->bytes.end(), std::byte{0x77});
    }
    tracked->touch();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    // Drop every consumer view so the padded allocation is not pinned.
    depth_resident = {};
    depth_native = nullptr;
    auto retained = require_value(fixture.memory->retain(
        coded.address, coded.backing_extent, ResourceAccess::Read));
    TextureStorageKey storage{retained.bytes, coded};
    TextureViewKey view{storage};
    view.level_count = 1;
    view.layer_count = 1;
    TextureRequest sampled{retained, view, ResourceAccess::Read,
                           TextureAliasScope::Sampler};
    const auto before = metal::texture_cache_statistics(*fixture.cache);
    const auto resident = require_value(fixture.cache->acquire(sampled));
    const auto native = native_texture(resident);
    CHECK(native && native->allocation().pixelFormat == MTLPixelFormatR16Unorm);
    CHECK(resident.key.storage.bytes.size == 73728);
    CHECK(native && native->allocation() != depth_allocation);
    const auto after = metal::texture_cache_statistics(*fixture.cache);
    CHECK(after.uploads == before.uploads + 1);
    CHECK(after.upload_bytes == before.upload_bytes + 73728);
    CHECK(after.reuploads == before.reuploads);
    // The padded depth allocation still serves its own exact key, re-uploaded
    // in place from the changed guest bytes.
    const auto depth_again = require_value(fixture.cache->acquire(depth_keyed));
    const auto depth_again_native = native_texture(depth_again);
    CHECK(depth_again_native &&
          depth_again_native->allocation() == depth_allocation);
    CHECK(metal::texture_cache_statistics(*fixture.cache).reuploads ==
          before.reuploads + 1);
    std::puts("Verified depth-aliased allocations keep their TIC's layout.");
}

void test_delayed_producer_and_inflight_eviction(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    TextureAsset asset = fixture.add_texture(TextureType::Texture2D, 73, 45, 1, 1, 1);
    const TextureRequest request = fixture.request(asset, ResourceAccess::Write);
    ResidentTexture target = require_value(fixture.cache->acquire(request));
    auto native = native_texture(target);
    SubmittedWrite writer(23, {target.storage});
    commit_gpu_write(fixture, asset, target, writer);
    id<MTLSharedEvent> gate = [device newSharedEvent];
    CHECK(gate != nil);
    if (!gate || !native) return;
    [fixture.context->queue() waitForEvent:gate value:1];
    const auto before = metal::texture_cache_statistics(*fixture.cache);
    const unsigned reads_before = fixture.backend->reads;
    const unsigned writes_before = fixture.backend->writes;
    std::fill(asset.expected[0].begin(), asset.expected[0].end(),
              std::array<std::uint8_t, 4>{64, 128, 191, 255});
    bool sampled_while_pending = false;
    CHECK(clear_texture(fixture.context, *native, writer,
        MTLClearColorMake(0.25, 0.5, 0.75, 1.0), false, 1.0, false, 0, [&] {
            CHECK(gate.signaledValue == 0);
            CHECK(writer.token.snapshot().phase == SubmissionPhase::Submitted);
            ResidentTexture pending = require_value(fixture.cache->acquire(fixture.request(asset)));
            auto pending_native = native_texture(pending);
            CHECK(pending.dependencies.size() == 1);
            if (pending.dependencies.size() == 1) {
                CHECK(pending.dependencies[0].snapshot().id == writer.token.snapshot().id);
                CHECK(pending.dependencies[0].snapshot().phase == SubmissionPhase::Submitted);
            }
            CHECK(pending_native && pending_native->allocation() == native->allocation());
            SubmittedWrite consumer(24, {pending.storage});
            const std::array accesses{GpuAccessRequest{asset.retained, ResourceAccess::Read}};
            auto reservation = require_value(fixture.memory->reserve_accesses(accesses, consumer.token));
            CHECK(succeeded(reservation->commit({pending.storage})));
            CHECK(succeeded(consumer.completion.submitted()));
            if (pending_native) {
                CHECK(sampling.sample(asset, *pending_native, "sample_2d", false,
                    {TextureSwizzle::Red, TextureSwizzle::Green,
                     TextureSwizzle::Blue, TextureSwizzle::Alpha}, [&] {
                        // Both commands are committed while the queue is gated.
                        // Drop client handles so completion/registry pins alone
                        // must keep the allocation charged and unevictable.
                        CHECK(writer.token.snapshot().phase == SubmissionPhase::Submitted);
                        CHECK(gate.signaledValue == 0);
                        target.storage.reset();
                        native.reset();
                        pending.storage.reset();
                        pending_native.reset();
                        CHECK(is_error(fixture.cache->trim(0), CommandStatus::Stopped));
                        CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes == before.resident_bytes);
                        CHECK(fixture.backend->reads == reads_before);
                        CHECK(fixture.backend->writes == writes_before);
                        sampled_while_pending = true;
                        // Keep the producer genuinely in flight after the
                        // consumer dispatch, then unblock without a CPU wait.
                        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC),
                                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                            gate.signaledValue = 1;
                        });
                    }));
            }
            // Also unblock the queue on a setup failure so teardown can finish.
            gate.signaledValue = 1;
            CHECK(succeeded(consumer.completion.feedback({})));
            CHECK(succeeded(consumer.completion.finish()));
        }));
    CHECK(sampled_while_pending);
    CHECK(writer.token.snapshot().phase == SubmissionPhase::Complete);
    const auto after = metal::texture_cache_statistics(*fixture.cache);
    CHECK(after.uploads == before.uploads && after.readbacks == before.readbacks);
    CHECK(succeeded(fixture.cache->resolve(request.key, ResolveReason::Capture)));
    CHECK(succeeded(fixture.cache->trim(0)));
    CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes == 0);
    std::puts("Verified sampling behind a delayed in-flight producer and eviction pinning.");
}

void test_resolved_incompatible_alias_versions(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2DArray, 4, 4, 1, 1, 2, TextureLayout::Pitch);
    TextureRequest layer_one = fixture.request(asset, ResourceAccess::Write);
    layer_one.key.first_layer = 1;
    layer_one.key.layer_count = 1;
    const ResidentTexture original = require_value(fixture.cache->acquire(layer_one));
    const auto native = native_texture(original);
    TextureAsset written_layer = asset;
    written_layer.retained = require_value(fixture.memory->retain(
        asset.descriptor.address + asset.layout.layer_stride,
        asset.layout.layer_stride, ResourceAccess::Write));
    SubmittedWrite first_writer(20, {original.storage});
    commit_gpu_write(fixture, written_layer, original, first_writer);
    CHECK(native && clear_texture(fixture.context, *native, first_writer,
                                  MTLClearColorMake(0.25, 0.5, 0.75, 1.0)));

    // A separate, incompatible float allocation overwrites only layer zero.
    TextureAsset alias = asset;
    alias.descriptor.layers = 1;
    alias.descriptor.backing_extent = asset.layout.layer_stride;
    alias.descriptor.format = TextureFormat::R32Float;
    alias.retained = require_value(fixture.memory->retain(
        asset.descriptor.address, asset.layout.layer_stride, ResourceAccess::Write));
    const TextureRequest alias_request = fixture.request(alias, ResourceAccess::Write);
    const ResidentTexture alias_target = require_value(fixture.cache->acquire(alias_request));
    const auto alias_native = native_texture(alias_target);
    SubmittedWrite alias_writer(21, {alias_target.storage});
    commit_gpu_write(fixture, alias, alias_target, alias_writer);
    CHECK(alias_native && clear_texture(fixture.context, *alias_native, alias_writer,
                                        MTLClearColorMake(0.5, 0, 0, 0)));
    CHECK(succeeded(fixture.cache->resolve(alias_request.key, ResolveReason::IncompatibleAlias)));
    const auto ranges = require_value(fixture.memory->inspect(asset.retained.bytes));
    CHECK(ranges.size() == 2);
    CHECK(!ranges[0].gpu_dirty && ranges[1].gpu_dirty);
    CHECK(ranges[0].version.gpu_write > ranges[1].version.gpu_write);
    CHECK(ranges[0].version.cpu_write == ranges[1].version.cpu_write);

    // Because each storage and description has its own texture, bytes the
    // incompatible alias rendered do not transfer: the array keeps its own
    // layer zero, without an upload, and its own rendered layer one.
    const auto before = metal::texture_cache_statistics(*fixture.cache);
    const ResidentTexture refreshed = require_value(fixture.cache->acquire(fixture.request(asset)));
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == before.uploads);
    const auto refreshed_native = native_texture(refreshed);
    CHECK(refreshed_native && refreshed_native->allocation() == native->allocation());
    std::fill(asset.expected[1].begin(), asset.expected[1].end(),
              std::array<std::uint8_t, 4>{64, 128, 191, 255});
    CHECK(refreshed_native && sampling.sample(asset, *refreshed_native, "sample_2d_array"));
    std::puts("Verified a sampled description keeps its contents over an incompatible alias.");
}

void test_prefix_mip_upload_and_writeback(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    const TextureAsset allocated = fixture.add_texture(
        TextureType::Texture2DArray, 73, 45, 1, 7, 2,
        TextureLayout::BlockLinear, TextureFormat::R8G8B8A8Unorm, 2);
    CHECK(allocated.layout.layer_stride == 30720);
    TextureAsset prefix = allocated;
    prefix.descriptor.levels = 3;
    prefix.layout = require_layout(texture_view_layout(prefix.descriptor));
    CHECK(prefix.layout.layer_stride == 30720);
    CHECK(prefix.layout.total_size == 59392);
    prefix.descriptor.backing_extent = prefix.layout.total_size;
    prefix.retained = require_value(fixture.memory->retain(
        prefix.descriptor.address, prefix.descriptor.backing_extent, ResourceAccess::ReadWrite));
    prefix.expected.clear();
    for (std::uint32_t layer = 0; layer < 2; ++layer)
        for (std::uint32_t level = 0; level < 3; ++level)
            prefix.expected.push_back(allocated.expected[layer * 7 + level]);
    const ResidentTexture uploaded = require_value(fixture.cache->acquire(fixture.request(prefix)));
    const auto uploaded_native = native_texture(uploaded);
    CHECK(uploaded_native && sampling.sample(prefix, *uploaded_native, "sample_2d_array"));

    auto* backing = fixture.backend->find(prefix.retained.bytes.backing);
    CHECK(backing != nullptr);
    if (!backing) return;
    auto expected_guest = backing->bytes;
    const std::array<std::uint8_t, 4> color{64, 128, 191, 255};
    std::vector<std::byte> linear(prefix.layout.mips[0].linear_bytes);
    for (std::size_t texel = 0; texel < linear.size() / 4; ++texel)
        for (unsigned component = 0; component < 4; ++component)
            linear[texel * 4 + component] = std::byte{color[component]};
    store_fixture_level(allocated.layout, linear, expected_guest, 0, 1);

    TextureRequest writable = fixture.request(prefix, ResourceAccess::Write);
    writable.key.first_layer = 1;
    writable.key.layer_count = 1;
    writable.key.level_count = 1;
    const ResidentTexture target = require_value(fixture.cache->acquire(writable));
    const auto native = native_texture(target);
    SubmittedWrite writer(25, {target.storage});
    // Resolve the whole prefix so every conversion runs, including preserved
    // layers, mip levels and the omitted tail between the two array layers.
    commit_gpu_write(fixture, prefix, target, writer);
    CHECK(native && clear_texture(fixture.context, *native, writer,
                                  MTLClearColorMake(0.25, 0.5, 0.75, 1.0)));
    CHECK(succeeded(fixture.cache->resolve(writable.key, ResolveReason::Capture)));
    CHECK(backing->bytes == expected_guest);
    CHECK(std::equal(backing->bytes.begin() + 28672, backing->bytes.begin() + 30720,
                     expected_guest.begin() + 28672));
    std::fill(prefix.expected[3].begin(), prefix.expected[3].end(), color);
    const ResidentTexture refreshed = require_value(fixture.cache->acquire(fixture.request(prefix)));
    const auto refreshed_native = native_texture(refreshed);
    CHECK(refreshed_native && sampling.sample(prefix, *refreshed_native, "sample_2d_array"));
    std::puts("Verified 73x45 layered prefix GPU upload and writeback preserve allocated mip tails.");
}

void test_shared_cache_eviction(id<MTLDevice> device) {
    Fixture fixture(device);
    auto second_cache = require_value(metal::create_texture_cache(
        fixture.context, fixture.memory, 32u * 1024u * 1024u));
    TextureAsset asset = fixture.add_texture(TextureType::Texture2D, 4, 4, 1, 1, 1);
    const TextureRequest request = fixture.request(asset, ResourceAccess::Write);
    {
        const ResidentTexture target = require_value(fixture.cache->acquire(request));
        const auto native = native_texture(target);
        SubmittedWrite writer(22, {target.storage});
        commit_gpu_write(fixture, asset, target, writer);
        CHECK(native && clear_texture(fixture.context, *native, writer,
                                      MTLClearColorMake(0.25, 0.5, 0.75, 1)));
        const ResidentTexture imported = require_value(second_cache->acquire(fixture.request(asset)));
        const auto imported_native = native_texture(imported);
        CHECK(imported_native && imported_native->allocation() == native->allocation());
        CHECK(metal::texture_cache_statistics(*second_cache).uploads == 0);
        CHECK(succeeded(fixture.cache->resolve(request.key, ResolveReason::Capture)));
        CHECK(is_error(fixture.cache->trim(0), CommandStatus::Stopped));
        CHECK(is_error(second_cache->trim(0), CommandStatus::Stopped));
    }
    CHECK(succeeded(fixture.cache->trim(0)));
    CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes == 0);
    CHECK(succeeded(second_cache->trim(0)));
    CHECK(metal::texture_cache_statistics(*second_cache).resident_bytes == 0);
    const ResidentTexture admitted = require_value(second_cache->acquire(fixture.request(asset)));
    CHECK(admitted.storage != nullptr);
    std::puts("Verified shared cache ownership does not permanently pin clean allocations.");
}

void test_resolved_depth_residency(id<MTLDevice> device) {
    Fixture fixture(device);
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2D, 2, 2, 1, 1, 1,
        TextureLayout::BlockLinear, TextureFormat::D24UnormS8Uint);
    auto request = fixture.request(asset, ResourceAccess::Write);
    request.key.aspect = TextureAspect::DepthStencil;
    auto target = require_value(fixture.cache->acquire(request));
    auto native = native_texture(target);
    id<MTLTexture> allocation = native->allocation();
    SubmittedWrite writer(31, {target.storage});
    commit_gpu_write(fixture, asset, target, writer);
    CHECK(clear_texture(fixture.context, *native, writer,
                        MTLClearColorMake(0, 0, 0, 0), true, 0.25));
    BackingRange prefix = asset.retained.bytes;
    prefix.size = 4;
    CHECK(succeeded(fixture.memory->resolve(prefix, ResolveReason::Capture)));
    auto partial = require_value(fixture.cache->acquire(request));
    CHECK(native_texture(partial)->allocation() == allocation);
    CHECK(succeeded(fixture.memory->submission_boundary()));
    auto resolved = require_value(fixture.cache->acquire(request));
    CHECK(native_texture(resolved)->allocation() == allocation);
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 1);
    partial.storage.reset();
    resolved.storage.reset();
    target.storage.reset();
    native.reset();
    // Packed D24 cannot reconstruct the original floating depth. Budget
    // pressure must preserve it or stop explicitly, never silently re-upload.
    CHECK(is_error(fixture.cache->trim(0), CommandStatus::Stopped));
    auto retained = require_value(fixture.cache->acquire(request));
    CHECK(native_texture(retained)->allocation() == allocation);
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 1);

    // Remapping a resolved D24 target must preserve native float precision,
    // including while callers still hold passive views of the old mapping.
    auto remapped = fixture.memory->address_space().mappings().front();
    const std::array removed{asset.retained.mapping};
    CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    remapped.gpu_base += 0x800000;
    const MappingId new_mapping{removed[0].address_space, removed[0].mapping, 2};
    CHECK(succeeded(fixture.memory->register_mapping(
        {new_mapping, asset.retained.bytes.backing, remapped})));
    asset.descriptor.address = remapped.gpu_base;
    asset.retained = require_value(fixture.memory->retain(remapped.gpu_base,
        asset.descriptor.backing_extent, ResourceAccess::ReadWrite));
    request = fixture.request(asset, ResourceAccess::Write);
    request.key.aspect = TextureAspect::DepthStencil;
    auto alias = require_value(fixture.cache->acquire(request));
    CHECK(native_texture(alias)->allocation() == allocation);
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 1);
    alias.storage.reset();

    // A tracked same-value CPU store still supersedes native GPU precision.
    auto write = require_value(fixture.memory->prepare_cpu_write(asset.retained.bytes));
    write->commit();
    std::vector<std::byte> stale_read(asset.retained.bytes.size);
    CHECK(succeeded(native_texture(retained)->read_gpu(asset.retained.bytes, stale_read)));
    auto replaced = require_value(fixture.cache->acquire(request));
    CHECK(native_texture(replaced)->allocation() != allocation);
    CHECK(metal::texture_cache_statistics(*fixture.cache).uploads == 2);
    retained.storage.reset();
    replaced.storage.reset();
    CHECK(succeeded(fixture.cache->trim(0)));
    std::puts("Verified resolved D24 precision, partial resolves and CPU version replacement.");
}

void test_depth_snapshot_epoch_and_retirement(id<MTLDevice> device) {
    for (const bool retire : {false, true}) {
        Fixture fixture(device);
        TextureAsset asset = fixture.add_texture(
            TextureType::Texture2D, 2, 2, 1, 1, 1,
            TextureLayout::BlockLinear, TextureFormat::D24UnormS8Uint);
        auto request = fixture.request(asset, ResourceAccess::Write);
        request.key.aspect = TextureAspect::DepthStencil;
        auto target = require_value(fixture.cache->acquire(request));
        auto native = native_texture(target);
        SubmittedWrite writer(32, {target.storage});
        commit_gpu_write(fixture, asset, target, writer);
        CHECK(clear_texture(fixture.context, *native, writer,
                            MTLClearColorMake(0, 0, 0, 0), true, 0.25));
        CHECK(succeeded(fixture.memory->submission_boundary()));
        const auto backing = asset.retained.bytes.backing;
        const auto mapping = asset.retained.mapping;
        target.storage.reset();
        native.reset();
        request.range.lease.reset();
        asset.retained.lease.reset();
        // Do not reacquire before trim: it must validate the changed epoch
        // independently, and must never treat an unmapped VA as a CPU change.
        if (retire) CHECK(succeeded(fixture.memory->unmap(mapping)));
        CHECK(is_error(fixture.cache->trim(0), CommandStatus::Stopped));
        if (retire) {
            CHECK(succeeded(fixture.memory->retire_backing(backing)));
        } else {
            auto* bytes = fixture.backend->find(backing);
            CHECK(bytes != nullptr);
            if (bytes) bytes->bytes[0] ^= std::byte{1};
            CHECK(succeeded(fixture.memory->submission_boundary()));
        }
        CHECK(succeeded(fixture.cache->trim(0)));
        CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes == 0);
    }
    std::puts("Verified depth snapshot epoch validation and retired-backing eviction.");
}

void test_depth_stencil_preservation(id<MTLDevice> device) {
    Fixture fixture(device);
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2D, 2, 2, 1, 1, 1,
        TextureLayout::BlockLinear, TextureFormat::D32FloatS8Uint);
    Backend::Allocation* bytes = fixture.backend->find(asset.retained.bytes.backing);
    CHECK(bytes != nullptr);
    const auto& mip = asset.layout.mips[0];
    std::vector<std::byte> linear(mip.linear_bytes);
    for (std::size_t texel = 0; texel < linear.size() / 8; ++texel) {
        const float initial_depth = 0.8f;
        std::memcpy(linear.data() + texel * 8, &initial_depth, sizeof(initial_depth));
        linear[texel * 8 + 4] = std::byte{0x7b};
        linear[texel * 8 + 5] = std::byte{0xa5};
        linear[texel * 8 + 6] = std::byte{0x5a};
        linear[texel * 8 + 7] = std::byte{0xc3};
    }
    if (bytes) {
        CHECK(std::holds_alternative<std::monostate>(write_level_from_linear(
            asset.layout, linear, bytes->bytes, 0, 0)));
    }
    TextureRequest writable = fixture.request(asset, ResourceAccess::Write);
    writable.key.aspect = TextureAspect::DepthStencil;
    const ResidentTexture target = require_value(fixture.cache->acquire(writable));
    const auto native = native_texture(target);
    CHECK(native != nullptr);
    if (native) {
        CHECK(native->allocation().pixelFormat ==
              MTLPixelFormatDepth32Float_Stencil8);
    }
    SubmittedWrite writer(2, {target.storage});
    commit_gpu_write(fixture, asset, target, writer);
    CHECK(native && clear_texture(fixture.context, *native, writer,
                                  MTLClearColorMake(0, 0, 0, 0), true, 0.25));
    CHECK(succeeded(fixture.cache->resolve(
        writable.key, ResolveReason::Capture)));
    CHECK(bytes != nullptr);
    if (bytes) {
        const TextureLevelData resolved = require_layout(convert_level_to_linear(
            asset.layout, bytes->bytes, 0, 0));
        for (std::size_t texel = 0; texel < resolved.bytes.size() / 8; ++texel) {
            float depth = 0.0f;
            std::memcpy(&depth, resolved.bytes.data() + texel * 8, sizeof(depth));
            CHECK(std::fabs(depth - 0.25f) <= 1.0e-6f);
            CHECK(resolved.bytes[texel * 8 + 4] == std::byte{0x7b});
            CHECK(resolved.bytes[texel * 8 + 5] == std::byte{0xa5});
            CHECK(resolved.bytes[texel * 8 + 6] == std::byte{0x5a});
            CHECK(resolved.bytes[texel * 8 + 7] == std::byte{0xc3});
        }
    }

    TextureAsset stencil_asset = fixture.add_texture(
        TextureType::Texture2D, 2, 2, 1, 1, 1,
        TextureLayout::BlockLinear, TextureFormat::D32FloatS8Uint);
    Backend::Allocation* stencil_bytes =
        fixture.backend->find(stencil_asset.retained.bytes.backing);
    const auto& stencil_mip = stencil_asset.layout.mips[0];
    std::vector<std::byte> stencil_linear(stencil_mip.linear_bytes);
    for (std::size_t texel = 0; texel < stencil_linear.size() / 8; ++texel) {
        const float initial_depth = 0.625f;
        std::memcpy(stencil_linear.data() + texel * 8, &initial_depth,
                    sizeof(initial_depth));
        stencil_linear[texel * 8 + 4] = std::byte{0x11};
        stencil_linear[texel * 8 + 5] = std::byte{0xde};
        stencil_linear[texel * 8 + 6] = std::byte{0xad};
        stencil_linear[texel * 8 + 7] = std::byte{0xbe};
    }
    CHECK(stencil_bytes != nullptr);
    if (stencil_bytes) {
        CHECK(std::holds_alternative<std::monostate>(write_level_from_linear(
            stencil_asset.layout, stencil_linear, stencil_bytes->bytes, 0, 0)));
    }
    TextureRequest stencil_writable =
        fixture.request(stencil_asset, ResourceAccess::Write);
    stencil_writable.key.aspect = TextureAspect::DepthStencil;
    const ResidentTexture stencil_target =
        require_value(fixture.cache->acquire(stencil_writable));
    const auto stencil_native = native_texture(stencil_target);
    SubmittedWrite stencil_writer(4, {stencil_target.storage});
    commit_gpu_write(fixture, stencil_asset, stencil_target, stencil_writer);
    CHECK(stencil_native && clear_texture(
        fixture.context, *stencil_native, stencil_writer,
        MTLClearColorMake(0, 0, 0, 0), true, 1.0, true, 0x2d));
    CHECK(succeeded(fixture.cache->resolve(
        stencil_writable.key, ResolveReason::Capture)));
    if (stencil_bytes) {
        const TextureLevelData resolved = require_layout(convert_level_to_linear(
            stencil_asset.layout, stencil_bytes->bytes, 0, 0));
        for (std::size_t texel = 0; texel < resolved.bytes.size() / 8; ++texel) {
            float depth = 0.0f;
            std::memcpy(&depth, resolved.bytes.data() + texel * 8,
                        sizeof(depth));
            CHECK(std::fabs(depth - 0.625f) <= 1.0e-6f);
            CHECK(resolved.bytes[texel * 8 + 4] == std::byte{0x2d});
            CHECK(resolved.bytes[texel * 8 + 5] == std::byte{0xde});
            CHECK(resolved.bytes[texel * 8 + 6] == std::byte{0xad});
            CHECK(resolved.bytes[texel * 8 + 7] == std::byte{0xbe});
        }
    }
    TextureAsset packed_asset = fixture.add_texture(
        TextureType::Texture2D, 2, 2, 1, 1, 1,
        TextureLayout::BlockLinear, TextureFormat::D24UnormS8Uint);
    Backend::Allocation* packed_bytes =
        fixture.backend->find(packed_asset.retained.bytes.backing);
    const auto& packed_mip = packed_asset.layout.mips[0];
    std::vector<std::byte> packed_linear(packed_mip.linear_bytes);
    for (std::size_t texel = 0; texel < packed_linear.size() / 4; ++texel) {
        packed_linear[texel * 4] = std::byte{0x39};
        packed_linear[texel * 4 + 1] = std::byte{0x00};
        packed_linear[texel * 4 + 2] = std::byte{0x00};
        packed_linear[texel * 4 + 3] = std::byte{0xc0};
    }
    CHECK(packed_bytes != nullptr);
    if (packed_bytes) {
        CHECK(std::holds_alternative<std::monostate>(write_level_from_linear(
            packed_asset.layout, packed_linear, packed_bytes->bytes, 0, 0)));
    }
    TextureRequest packed_writable =
        fixture.request(packed_asset, ResourceAccess::Write);
    packed_writable.key.aspect = TextureAspect::DepthStencil;
    const ResidentTexture packed_target =
        require_value(fixture.cache->acquire(packed_writable));
    const auto packed_native = native_texture(packed_target);
    if (packed_native) {
        CHECK(packed_native->allocation().pixelFormat ==
              MTLPixelFormatDepth32Float_Stencil8);
    }
    SubmittedWrite packed_writer(7, {packed_target.storage});
    commit_gpu_write(fixture, packed_asset, packed_target, packed_writer);
    CHECK(packed_native && clear_texture(
        fixture.context, *packed_native, packed_writer,
        MTLClearColorMake(0, 0, 0, 0), true, 0.25));
    CHECK(succeeded(fixture.cache->resolve(
        packed_writable.key, ResolveReason::Capture)));
    if (packed_bytes) {
        const TextureLevelData resolved = require_layout(convert_level_to_linear(
            packed_asset.layout, packed_bytes->bytes, 0, 0));
        for (std::size_t texel = 0; texel < resolved.bytes.size() / 4; ++texel) {
            CHECK(resolved.bytes[texel * 4] == std::byte{0x39});
            CHECK(resolved.bytes[texel * 4 + 1] == std::byte{0x00});
            CHECK(resolved.bytes[texel * 4 + 2] == std::byte{0x00});
            CHECK(resolved.bytes[texel * 4 + 3] == std::byte{0x40});
        }
    }
    CHECK(metal::texture_cache_statistics(*fixture.cache).readbacks == 3);
    std::puts("Verified depth/stencil clears and guest padding preservation.");
}

void test_invalidation_and_partial_resolve(id<MTLDevice> device) {
    {
        Fixture fixture(device);
        TextureAsset asset = fixture.add_texture(
            TextureType::Texture2D, 4, 4, 1, 1, 1);
        const TextureRequest writable =
            fixture.request(asset, ResourceAccess::Write);
        ResidentTexture target =
            require_value(fixture.cache->acquire(writable));
        auto native = native_texture(target);
        SubmittedWrite writer(5, {target.storage});
        commit_gpu_write(fixture, asset, target, writer);
        CHECK(native && clear_texture(
            fixture.context, *native, writer,
            MTLClearColorMake(0.125, 0.25, 0.5, 1.0)));

        const auto before = metal::texture_cache_statistics(*fixture.cache);
        const unsigned reads_before = fixture.backend->reads;
        CHECK(succeeded(fixture.cache->invalidate(asset.retained.bytes)));
        CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes ==
              before.resident_bytes);
        ResidentTexture recovered =
            require_value(fixture.cache->acquire(fixture.request(asset)));
        auto recovered_native = native_texture(recovered);
        CHECK(recovered_native && native &&
              recovered_native->allocation() == native->allocation());
        CHECK(fixture.backend->reads == reads_before);
        const auto after = metal::texture_cache_statistics(*fixture.cache);
        CHECK(after.uploads == before.uploads);
        CHECK(after.readbacks == before.readbacks);
        CHECK(after.resident_bytes == before.resident_bytes);

        recovered_native.reset();
        native.reset();
        recovered.storage.reset();
        target.storage.reset();
        CHECK(is_error(fixture.cache->trim(0), CommandStatus::Stopped));
        CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes ==
              before.resident_bytes);
    }

    {
        Fixture fixture(device);
        TextureAsset asset = fixture.add_texture(
            TextureType::Texture2D, 4, 4, 1, 1, 1);
        ResidentTexture first =
            require_value(fixture.cache->acquire(fixture.request(asset)));
        auto first_native = native_texture(first);
        id<MTLTexture> old_allocation =
            first_native ? first_native->allocation() : nil;
        const auto before = metal::texture_cache_statistics(*fixture.cache);
        first_native.reset();
        first.storage.reset();
        CHECK(succeeded(fixture.cache->invalidate(asset.retained.bytes)));

        TextureRequest srgb = fixture.request(asset);
        srgb.key.storage.descriptor.format = TextureFormat::R8G8B8A8Srgb;
        srgb.key.storage.descriptor.srgb = true;
        const ResidentTexture refreshed =
            require_value(fixture.cache->acquire(srgb));
        const auto refreshed_native = native_texture(refreshed);
        CHECK(refreshed_native != nullptr);
        if (refreshed_native) {
            CHECK(refreshed_native->allocation() != old_allocation);
            CHECK(refreshed_native->texture().pixelFormat ==
                  MTLPixelFormatRGBA8Unorm_sRGB);
        }
        const auto after = metal::texture_cache_statistics(*fixture.cache);
        CHECK(after.uploads == before.uploads + 1);
        CHECK(after.upload_bytes == before.upload_bytes +
                                          asset.descriptor.backing_extent);
        CHECK(after.readbacks == before.readbacks);
    }

    {
        Fixture fixture(device);
        TextureAsset asset = fixture.add_texture(
            TextureType::Texture2D, 4, 4, 1, 1, 1, TextureLayout::Pitch);
        const TextureRequest writable =
            fixture.request(asset, ResourceAccess::Write);
        const ResidentTexture target =
            require_value(fixture.cache->acquire(writable));
        const auto native = native_texture(target);
        SubmittedWrite writer(6, {target.storage});
        commit_gpu_write(fixture, asset, target, writer);
        CHECK(native && clear_texture(
            fixture.context, *native, writer,
            MTLClearColorMake(0.2, 0.4, 0.6, 1.0)));

        Backend::Allocation* bytes =
            fixture.backend->find(asset.retained.bytes.backing);
        CHECK(bytes != nullptr);
        const auto original = asset.expected[0][1];
        BackingRange first_texel = asset.retained.bytes;
        first_texel.size = 4;
        CHECK(succeeded(fixture.memory->resolve(
            first_texel, ResolveReason::Capture)));
        if (bytes) {
            const std::array<std::byte, 4> clear{
                std::byte{51}, std::byte{102}, std::byte{153}, std::byte{255}};
            CHECK(std::equal(clear.begin(), clear.end(), bytes->bytes.begin()));
            for (unsigned component = 0; component < 4; ++component) {
                CHECK(bytes->bytes[4 + component] ==
                      std::byte{original[component]});
            }
        }
        const auto intervals =
            require_value(fixture.memory->inspect(asset.retained.bytes));
        CHECK(std::any_of(intervals.begin(), intervals.end(),
                          [](const DirtyRange& range) {
                              return range.gpu_dirty;
                          }));
        CHECK(metal::texture_cache_statistics(*fixture.cache).readbacks == 1);
    }
    std::puts("Verified dirty and clean invalidation plus partial native resolve.");
}

void test_mapping_invalidation_and_remap(id<MTLDevice> device) {
    for (const bool dirty : {false, true}) {
        Fixture fixture(device);
        SamplingHarness sampling(fixture.context);
        auto asset = fixture.add_texture(TextureType::Texture2D, 4, 4, 1, 1, 1);
        const auto original_request = fixture.request(asset, ResourceAccess::ReadWrite);
        const auto original_mapping = fixture.memory->address_space().mappings().front();
        const auto original = require_value(fixture.cache->acquire(original_request));
        const auto native = native_texture(original);
        const auto old_bytes = fixture.backend->find(asset.retained.bytes.backing)->bytes;
        SubmittedWrite writer(70, {original.storage});
        if (dirty) {
            commit_gpu_write(fixture, asset, original, writer);
            CHECK(clear_texture(fixture.context, *native, writer, MTLClearColorMake(1, 0, 0, 1)));
            std::fill(asset.expected[0].begin(), asset.expected[0].end(),
                      std::array<std::uint8_t, 4>{255, 0, 0, 255});
            CHECK(fixture.backend->find(asset.retained.bytes.backing)->bytes == old_bytes);
        }
        auto alias_mapping = original_mapping;
        alias_mapping.gpu_base += 0x800000;
        const MappingId alias_id{0x4c34, 100, 1};
        CHECK(succeeded(fixture.memory->register_mapping(
            {alias_id, asset.retained.bytes.backing, alias_mapping})));
        const std::array removed{asset.retained.mapping};
        const auto writes = fixture.backend->writes.load();
        CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
        CHECK((fixture.backend->writes > writes) == dirty);
        CHECK(is_error(fixture.cache->acquire(original_request)));
        CHECK(is_error(fixture.memory->retain(original_mapping.gpu_base,
            asset.retained.bytes.size, ResourceAccess::Read)));
        // The surviving alias still names the original bytes and native image.
        auto alias_asset = asset;
        alias_asset.descriptor.address = alias_mapping.gpu_base;
        alias_asset.retained = require_value(fixture.memory->retain(alias_mapping.gpu_base,
            asset.retained.bytes.size, ResourceAccess::ReadWrite));
        auto alias = require_value(fixture.cache->acquire(fixture.request(alias_asset)));
        CHECK(sampling.sample(alias_asset, *native_texture(alias), "sample_2d"));
        // Restore the same allocation at the old VA with a fresh mapping identity.
        const MappingId restored{removed[0].address_space, removed[0].mapping, 2};
        CHECK(succeeded(fixture.memory->register_mapping(
            {restored, asset.retained.bytes.backing, original_mapping})));
        asset.retained = require_value(fixture.memory->retain(original_mapping.gpu_base,
            asset.retained.bytes.size, ResourceAccess::ReadWrite));
        auto restored_texture = require_value(fixture.cache->acquire(fixture.request(asset)));
        CHECK(sampling.sample(asset, *native_texture(restored_texture), "sample_2d"));
        CHECK(is_error(fixture.cache->acquire(original_request)));

        // Reuse the VA for a different allocation while the old alias survives.
        const std::array replaced{restored};
        CHECK(succeeded(fixture.memory->invalidate_mappings(replaced)));
        auto replacement = fixture.add_texture(TextureType::Texture2D, 4, 4, 1, 1, 1);
        const auto replacement_original = replacement.retained.mapping;
        Mapping replacement_mapping{};
        for (const auto& mapping : fixture.memory->address_space().mappings())
            if (mapping.gpu_base == replacement.descriptor.address) replacement_mapping = mapping;
        const std::array replacement_removed{replacement_original};
        CHECK(succeeded(fixture.memory->invalidate_mappings(replacement_removed)));
        replacement_mapping.gpu_base = original_mapping.gpu_base;
        const MappingId replacement_id{removed[0].address_space, removed[0].mapping, 3};
        CHECK(succeeded(fixture.memory->register_mapping(
            {replacement_id, replacement.retained.bytes.backing, replacement_mapping})));
        replacement.descriptor.address = original_mapping.gpu_base;
        replacement.retained = require_value(fixture.memory->retain(original_mapping.gpu_base,
            replacement.descriptor.backing_extent, ResourceAccess::ReadWrite));
        auto fresh = require_value(fixture.cache->acquire(fixture.request(replacement)));
        CHECK(native_texture(fresh)->allocation() != native->allocation());
        CHECK(sampling.sample(replacement, *native_texture(fresh), "sample_2d"));
        CHECK(sampling.sample(alias_asset, *native_texture(alias), "sample_2d"));
        CHECK(is_error(fixture.cache->acquire(fixture.request(asset))));
    }
    std::puts("Verified clean and GPU-dirty texture remaps, retained aliases and fresh VA reuse.");
}

void test_srgb_and_staging_preflight(id<MTLDevice> device) {
    Fixture fixture(device);
    TextureAsset useful = fixture.add_texture(
        TextureType::Texture2D, 4, 4, 1, 1, 1);
    const TextureRequest useful_request = fixture.request(useful);
    const ResidentTexture useful_resident =
        require_value(fixture.cache->acquire(useful_request));
    const auto useful_native = native_texture(useful_resident);

    const unsigned mismatch_reads = fixture.backend->reads;
    const auto mismatch_stats =
        metal::texture_cache_statistics(*fixture.cache);
    TextureRequest unorm_marked_srgb = useful_request;
    unorm_marked_srgb.key.storage.descriptor.srgb = true;
    CHECK(is_error(fixture.cache->acquire(unorm_marked_srgb)));
    TextureRequest srgb_marked_linear = useful_request;
    srgb_marked_linear.key.storage.descriptor.format =
        TextureFormat::R8G8B8A8Srgb;
    srgb_marked_linear.key.storage.descriptor.srgb = false;
    CHECK(is_error(fixture.cache->acquire(srgb_marked_linear)));
    CHECK(fixture.backend->reads == mismatch_reads);
    auto after_mismatch = metal::texture_cache_statistics(*fixture.cache);
    CHECK(after_mismatch.uploads == mismatch_stats.uploads);
    CHECK(after_mismatch.upload_bytes == mismatch_stats.upload_bytes);
    CHECK(after_mismatch.readbacks == mismatch_stats.readbacks);
    CHECK(after_mismatch.resident_bytes == mismatch_stats.resident_bytes);

    constexpr std::uint32_t width = 1;
    constexpr std::uint32_t height = 8193;
    constexpr std::uint32_t layers = 256;
    constexpr std::uint64_t extent =
        std::uint64_t{width} * height * layers;
    constexpr BackingId backing{0x4c34, 0xf000, 1};
    constexpr MappingId mapping{0x4c34, 0xf000, 1};
    constexpr std::uint64_t address = 0xf0000000;
    const std::uint64_t registered_size = align_up(extent, 0x1000);
    fixture.backend->add(backing, registered_size);
    CHECK(succeeded(fixture.memory->register_backing(
        {backing, registered_size, fixture.backend})));
    CHECK(succeeded(fixture.memory->register_mapping(
        {mapping, backing,
         {address, registered_size, 0, address + 0x40000000,
          registered_size, 0x1000}})));

    TextureDescriptor oversized{};
    oversized.address = address;
    oversized.backing_extent = extent;
    oversized.row_bytes = width;
    oversized.storage_height = height;
    oversized.layer_stride = height;
    oversized.width = width;
    oversized.height = height;
    oversized.depth = 1;
    oversized.levels = 1;
    oversized.layers = layers;
    oversized.samples = 1;
    oversized.format = TextureFormat::R8Unorm;
    oversized.layout = TextureLayout::Pitch;
    oversized.type = TextureType::Texture2DArray;
    oversized.normalized_coordinates = true;
    const RetainedRange retained = require_value(fixture.memory->retain(
        address, extent, ResourceAccess::Read));
    TextureViewKey oversized_view{{retained.bytes, oversized}};
    oversized_view.layer_count = layers;
    const TextureRequest oversized_request{
        retained, oversized_view, ResourceAccess::Read};

    const unsigned reads_before = fixture.backend->reads;
    const auto before = metal::texture_cache_statistics(*fixture.cache);
    CHECK(is_error(fixture.cache->acquire(oversized_request)));
    CHECK(is_error(fixture.cache->acquire(oversized_request)));
    CHECK(fixture.backend->reads == reads_before);
    const auto after_failures =
        metal::texture_cache_statistics(*fixture.cache);
    CHECK(after_failures.uploads == before.uploads);
    CHECK(after_failures.upload_bytes == before.upload_bytes);
    CHECK(after_failures.readbacks == before.readbacks);
    CHECK(after_failures.hits == before.hits);
    CHECK(after_failures.resident_bytes == before.resident_bytes);

    const ResidentTexture useful_hit =
        require_value(fixture.cache->acquire(useful_request));
    const auto useful_hit_native = native_texture(useful_hit);
    CHECK(useful_native && useful_hit_native);
    if (useful_native && useful_hit_native) {
        CHECK(useful_hit_native->allocation() == useful_native->allocation());
        CHECK(useful_hit_native->texture() == useful_native->texture());
    }
    CHECK(fixture.backend->reads == reads_before);
    const auto final = metal::texture_cache_statistics(*fixture.cache);
    CHECK(final.uploads == before.uploads);
    CHECK(final.upload_bytes == before.upload_bytes);
    CHECK(final.readbacks == before.readbacks);
    CHECK(final.hits == before.hits + 1);
    CHECK(final.resident_bytes == before.resident_bytes);
    std::puts("Verified sRGB identity and staging expansion preflight failures.");
}

void test_compressed_srgb_aliases(id<MTLDevice> device) {
    Fixture fixture(device);
    struct FormatPair {
        TextureFormat linear;
        TextureFormat srgb;
        MTLPixelFormat native_srgb;
    };
    const std::array pairs{
        FormatPair{TextureFormat::Etc2RgbUnorm, TextureFormat::Etc2RgbSrgb,
                   MTLPixelFormatETC2_RGB8_sRGB},
        FormatPair{TextureFormat::Etc2RgbPtaUnorm,
                   TextureFormat::Etc2RgbPtaSrgb,
                   MTLPixelFormatETC2_RGB8A1_sRGB},
        FormatPair{TextureFormat::Etc2RgbaUnorm, TextureFormat::Etc2RgbaSrgb,
                   MTLPixelFormatEAC_RGBA8_sRGB},
        FormatPair{TextureFormat::Astc4x4Unorm, TextureFormat::Astc4x4Srgb,
                   MTLPixelFormatASTC_4x4_sRGB},
        FormatPair{TextureFormat::Astc5x4Unorm, TextureFormat::Astc5x4Srgb,
                   MTLPixelFormatASTC_5x4_sRGB},
        FormatPair{TextureFormat::Astc5x5Unorm, TextureFormat::Astc5x5Srgb,
                   MTLPixelFormatASTC_5x5_sRGB},
        FormatPair{TextureFormat::Astc6x5Unorm, TextureFormat::Astc6x5Srgb,
                   MTLPixelFormatASTC_6x5_sRGB},
        FormatPair{TextureFormat::Astc6x6Unorm, TextureFormat::Astc6x6Srgb,
                   MTLPixelFormatASTC_6x6_sRGB},
        FormatPair{TextureFormat::Astc8x5Unorm, TextureFormat::Astc8x5Srgb,
                   MTLPixelFormatASTC_8x5_sRGB},
        FormatPair{TextureFormat::Astc8x6Unorm, TextureFormat::Astc8x6Srgb,
                   MTLPixelFormatASTC_8x6_sRGB},
        FormatPair{TextureFormat::Astc8x8Unorm, TextureFormat::Astc8x8Srgb,
                   MTLPixelFormatASTC_8x8_sRGB},
        FormatPair{TextureFormat::Astc10x5Unorm, TextureFormat::Astc10x5Srgb,
                   MTLPixelFormatASTC_10x5_sRGB},
        FormatPair{TextureFormat::Astc10x6Unorm, TextureFormat::Astc10x6Srgb,
                   MTLPixelFormatASTC_10x6_sRGB},
        FormatPair{TextureFormat::Astc10x8Unorm, TextureFormat::Astc10x8Srgb,
                   MTLPixelFormatASTC_10x8_sRGB},
        FormatPair{TextureFormat::Astc10x10Unorm,
                   TextureFormat::Astc10x10Srgb,
                   MTLPixelFormatASTC_10x10_sRGB},
        FormatPair{TextureFormat::Astc12x10Unorm,
                   TextureFormat::Astc12x10Srgb,
                   MTLPixelFormatASTC_12x10_sRGB},
        FormatPair{TextureFormat::Astc12x12Unorm,
                   TextureFormat::Astc12x12Srgb,
                   MTLPixelFormatASTC_12x12_sRGB},
    };
    const auto initial = metal::texture_cache_statistics(*fixture.cache);
    for (std::size_t index = 0; index < pairs.size(); ++index) {
        const FormatPair& pair = pairs[index];
        TextureAsset asset = fixture.add_texture(
            TextureType::Texture2D, 12, 12, 1, 1, 1,
            TextureLayout::BlockLinear, pair.linear);
        const ResidentTexture linear =
            require_value(fixture.cache->acquire(fixture.request(asset)));
        const auto linear_native = native_texture(linear);
        const auto after_linear =
            metal::texture_cache_statistics(*fixture.cache);
        CHECK(after_linear.uploads == initial.uploads + index + 1);

        TextureRequest srgb = fixture.request(asset);
        srgb.key.storage.descriptor.format = pair.srgb;
        srgb.key.storage.descriptor.srgb = true;
        const ResidentTexture srgb_view =
            require_value(fixture.cache->acquire(srgb));
        const auto srgb_native = native_texture(srgb_view);
        CHECK(linear_native && srgb_native);
        if (linear_native && srgb_native) {
            CHECK(srgb_native->allocation() == linear_native->allocation());
            CHECK(srgb_native->texture().pixelFormat == pair.native_srgb);
        }
        const auto after_srgb =
            metal::texture_cache_statistics(*fixture.cache);
        CHECK(after_srgb.uploads == after_linear.uploads);
        CHECK(after_srgb.upload_bytes == after_linear.upload_bytes);
        CHECK(after_srgb.readbacks == after_linear.readbacks);
        CHECK(after_srgb.hits == initial.hits + index + 1);
    }
    std::printf("Verified %zu ETC2/ASTC linear-sRGB alias pairs.\n",
                pairs.size());
}

void test_negative_admission_and_dirty_budget(id<MTLDevice> device) {
    Fixture fixture(device);
    TextureAsset asset = fixture.add_texture(
        TextureType::Texture2DArray, 4, 4, 1, 3, 3);
    const TextureRequest valid = fixture.request(asset);

    TextureRequest bad_range = valid;
    --bad_range.range.bytes.size;
    CHECK(is_error(fixture.cache->acquire(bad_range)));
    TextureRequest bad_dimensions = valid;
    bad_dimensions.key.storage.descriptor.depth = 2;
    CHECK(is_error(fixture.cache->acquire(bad_dimensions)));
    TextureRequest bad_swizzle = valid;
    bad_swizzle.key.swizzle[2] = static_cast<TextureSwizzle>(0xff);
    CHECK(is_error(fixture.cache->acquire(bad_swizzle)));
    TextureRequest bad_samples = valid;
    bad_samples.key.storage.descriptor.samples = 4;
    CHECK(is_error(fixture.cache->acquire(bad_samples)));
    TextureRequest bad_levels = valid;
    bad_levels.key.first_level = 2;
    bad_levels.key.level_count = 2;
    CHECK(is_error(fixture.cache->acquire(bad_levels)));

    TextureAsset cube = fixture.add_texture(
        TextureType::CubemapArray, 4, 4, 1, 2, 12);
    TextureRequest second_cube = fixture.request(cube);
    second_cube.key.first_layer = 6;
    second_cube.key.layer_count = 6;
    const ResidentTexture cube_view =
        require_value(fixture.cache->acquire(second_cube));
    const auto cube_native = native_texture(cube_view);
    if (cube_native) CHECK(cube_native->texture().arrayLength == 1);
    TextureRequest bad_cube = fixture.request(cube);
    bad_cube.key.first_layer = 3;
    bad_cube.key.layer_count = 6;
    CHECK(is_error(fixture.cache->acquire(bad_cube)));

    TextureAsset one_d = fixture.add_texture(
        TextureType::Texture1D, 4, 1, 1, 1, 1);
    TextureRequest forged_mips = fixture.request(one_d);
    forged_mips.key.storage.descriptor.levels = 2;
    forged_mips.key.level_count = 2;
    CHECK(is_error(fixture.cache->acquire(forged_mips)));

    Fixture foreign(device);
    TextureAsset foreign_asset = foreign.add_texture(
        TextureType::Texture2D, 4, 4, 1, 1, 1);
    CHECK(is_error(fixture.cache->acquire(foreign.request(foreign_asset))));

    TextureAsset dirty_asset = fixture.add_texture(
        TextureType::Texture2D, 4, 4, 1, 1, 1);
    const TextureRequest dirty_write =
        fixture.request(dirty_asset, ResourceAccess::Write);
    ResidentTexture dirty =
        require_value(fixture.cache->acquire(dirty_write));
    SubmittedWrite writer(3, {dirty.storage});
    commit_gpu_write(fixture, dirty_asset, dirty, writer);
    auto dirty_native = native_texture(dirty);
    CHECK(dirty_native && clear_texture(
        fixture.context, *dirty_native, writer,
        MTLClearColorMake(1.0, 0.0, 0.0, 1.0)));
    dirty_native.reset();
    dirty.storage.reset();
    CHECK(is_error(fixture.cache->trim(0), CommandStatus::Stopped));
    CHECK(metal::texture_cache_statistics(*fixture.cache).resident_bytes > 0);
    std::puts("Verified malformed requests, foreign memory and dirty budget pinning.");
}

void test_factory_rejections(id<MTLDevice> device) {
    auto backend = std::make_shared<Backend>();
    auto memory = require_value(make_resource_memory(
        ResourceMemoryOptions{0x4c34, backend,
                              CoherenceMode::SubmissionBoundary, 0x1000}));
    auto context = std::make_shared<Context>(device);
    CHECK(is_error(metal::create_texture_cache({}, memory, 4096)));
    CHECK(is_error(metal::create_texture_cache(context, {}, 4096)));
    CHECK(is_error(metal::create_texture_cache(context, memory, 0)));
}

void test_all_targets_and_layouts(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    struct Target {
        TextureType type;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t depth;
        std::uint32_t layers;
        const char* function;
    };
    const std::array targets{
        Target{TextureType::Texture1D, 73, 1, 1, 1, "sample_1d"},
        Target{TextureType::Texture1DArray, 73, 1, 1, 3, "sample_1d_array"},
        Target{TextureType::Texture2D, 73, 45, 1, 1, "sample_2d"},
        Target{TextureType::Texture2DArray, 73, 45, 1, 3, "sample_2d_array"},
        Target{TextureType::Texture3D, 35, 19, 5, 1, "sample_3d"},
        Target{TextureType::Cubemap, 35, 35, 1, 6, "sample_cube"},
        Target{TextureType::CubemapArray, 35, 35, 1, 12, "sample_cube_array"},
    };
    for (const Target& target : targets) {
        const std::uint32_t levels =
            target.type == TextureType::Texture1D ||
                    target.type == TextureType::Texture1DArray
                ? 1
                : 3;
        TextureAsset asset = fixture.add_texture(
            target.type, target.width, target.height, target.depth, levels, target.layers);
        const ResidentTexture resident =
            require_value(fixture.cache->acquire(fixture.request(asset)));
        auto native = native_texture(resident);
        if (native) CHECK(sampling.sample(asset, *native, target.function));
    }

    TextureAsset rectangle = fixture.add_texture(
        TextureType::Texture2DRect, 73, 45, 1, 1, 1, TextureLayout::Pitch);
    const ResidentTexture rectangle_resident =
        require_value(fixture.cache->acquire(fixture.request(rectangle)));
    auto rectangle_native = native_texture(rectangle_resident);
    if (rectangle_native) {
        CHECK(sampling.sample(rectangle, *rectangle_native, "sample_2d"));
    }
    TextureAsset pitch_array = fixture.add_texture(
        TextureType::Texture2DArray, 73, 45, 1, 1, 3, TextureLayout::Pitch);
    const ResidentTexture pitch_array_resident =
        require_value(fixture.cache->acquire(fixture.request(pitch_array)));
    auto pitch_array_native = native_texture(pitch_array_resident);
    if (pitch_array_native) {
        CHECK(sampling.sample(pitch_array, *pitch_array_native,
                              "sample_2d_array"));
    }
    std::size_t compressed_uploads = 0;
    if (device.supportsBCTextureCompression) {
        TextureAsset bc1 = fixture.add_texture(
            TextureType::Texture2DArray, 8, 8, 1, 4, 3,
            TextureLayout::BlockLinear, TextureFormat::Bc1RgbaUnorm);
        const ResidentTexture bc1_resident =
            require_value(fixture.cache->acquire(fixture.request(bc1)));
        const auto bc1_native = native_texture(bc1_resident);
        if (bc1_native) {
            CHECK(bc1_native->texture().pixelFormat == MTLPixelFormatBC1_RGBA);
            CHECK(sampling.sample(bc1, *bc1_native, "sample_2d_array"));
        }
        compressed_uploads = 1;
        // Apple Silicon samples BC-compressed 3D textures natively.
        TextureAsset bc1_3d = fixture.add_texture(
            TextureType::Texture3D, 16, 8, 4, 2, 1,
            TextureLayout::BlockLinear, TextureFormat::Bc1RgbaUnorm);
        const ResidentTexture bc1_3d_resident =
            require_value(fixture.cache->acquire(fixture.request(bc1_3d)));
        const auto bc1_3d_native = native_texture(bc1_3d_resident);
        if (bc1_3d_native) {
            CHECK(bc1_3d_native->texture().pixelFormat == MTLPixelFormatBC1_RGBA);
            CHECK(bc1_3d_native->texture().textureType == MTLTextureType3D);
            CHECK(sampling.sample(bc1_3d, *bc1_3d_native, "sample_3d"));
        }
        compressed_uploads = 2;
    }

    const auto statistics = metal::texture_cache_statistics(*fixture.cache);
    CHECK(statistics.uploads == targets.size() + 2 + compressed_uploads);
    CHECK(statistics.upload_bytes > 0);
    CHECK(statistics.readbacks == 0);
    CHECK(statistics.resident_bytes > 0);
    std::printf("Sampled every texture target across all synthetic mip/layer views.\n");
}

// Same-dimensions arrayness views: a
// layer-zero native view in the admitted target serves a binding whose target
// differs from the storage only by arrayness. Every family samples correctly
// through the view, and shapes Metal cannot view fail with structured errors.
void test_same_dimensions_target_views(id<MTLDevice> device) {
    Fixture fixture(device);
    SamplingHarness sampling(fixture.context);
    // Probe assets describe the view's extent; expectations are copied from
    // the storage layers the view covers. Views are single-level here.
    const auto view_asset = [](const TextureAsset& storage, TextureType type,
                               std::uint32_t layers, std::uint32_t first_layer) {
        TextureAsset view;
        view.descriptor = storage.descriptor;
        view.descriptor.type = type;
        view.descriptor.layers = layers;
        view.descriptor.normalized_coordinates = type != TextureType::Texture2DRect;
        view.layout = storage.layout;
        view.retained = storage.retained;
        const std::uint32_t levels = storage.descriptor.levels;
        for (std::uint32_t layer = 0; layer < layers; ++layer)
            view.expected.push_back(
                storage.expected[std::size_t{first_layer + layer} * levels]);
        return view;
    };
    const auto view_request = [](const TextureAsset& storage, TextureType target,
                                 std::uint32_t first_layer, std::uint32_t layer_count) {
        TextureStorageKey storage_key{storage.retained.bytes, storage.descriptor};
        TextureViewKey key{storage_key};
        key.level_count = storage.descriptor.levels;
        key.layer_count = storage.descriptor.layers;
        key.view_target = target;
        key.first_layer = first_layer;
        key.layer_count = layer_count;
        return TextureRequest{storage.retained, key, ResourceAccess::Read};
    };
    const auto sample_view = [&](const char* what, const TextureAsset& storage,
                                 TextureType target, const char* function,
                                 std::uint32_t first_layer, std::uint32_t layer_count,
                                 MTLTextureType expected_native) {
        const auto resident = require_value(
            fixture.cache->acquire(view_request(storage, target, first_layer, layer_count)));
        const auto native = native_texture(resident);
        const bool typed = native != nullptr &&
                           native->texture().textureType == expected_native;
        if (!typed) std::fprintf(stderr, "view %s: native type mismatch\n", what);
        CHECK(typed);
        if (native)
            CHECK(sampling.sample(view_asset(storage, target, layer_count, first_layer),
                                  *native, function));
    };
    const TextureAsset array =
        fixture.add_texture(TextureType::Texture2DArray, 73, 45, 1, 1, 3);
    sample_view("2D view of 2D-array storage", array, TextureType::Texture2D,
                "sample_2d", 0, 1, MTLTextureType2D);
    const TextureAsset plain =
        fixture.add_texture(TextureType::Texture2D, 73, 45, 1, 1, 1);
    sample_view("2D-array view of 2D storage", plain, TextureType::Texture2DArray,
                "sample_2d_array", 0, 1, MTLTextureType2DArray);
    const TextureAsset array1d =
        fixture.add_texture(TextureType::Texture1DArray, 73, 1, 1, 1, 3);
    sample_view("1D view of 1D-array storage", array1d, TextureType::Texture1D,
                "sample_1d", 0, 1, MTLTextureType1D);
    const TextureAsset plain1d =
        fixture.add_texture(TextureType::Texture1D, 73, 1, 1, 1, 1);
    sample_view("1D-array view of 1D storage", plain1d, TextureType::Texture1DArray,
                "sample_1d_array", 0, 1, MTLTextureType1DArray);
    const TextureAsset cube =
        fixture.add_texture(TextureType::Cubemap, 35, 35, 1, 1, 6);
    sample_view("cube-array view of cube storage", cube, TextureType::CubemapArray,
                "sample_cube_array", 0, 6, MTLTextureTypeCubeArray);
    const TextureAsset cubes =
        fixture.add_texture(TextureType::CubemapArray, 35, 35, 1, 1, 12);
    sample_view("cube view of the first cube-array group", cubes,
                TextureType::Cubemap, "sample_cube", 0, 6, MTLTextureTypeCube);
    sample_view("cube view of the second cube-array group", cubes,
                TextureType::Cubemap, "sample_cube", 6, 6, MTLTextureTypeCube);
    const TextureAsset pitch_array = fixture.add_texture(
        TextureType::Texture2DArray, 73, 45, 1, 1, 2, TextureLayout::Pitch);
    sample_view("2D view of pitch 2D-array storage", pitch_array,
                TextureType::Texture2D, "sample_2d", 0, 1, MTLTextureType2D);
    // Shapes Metal cannot view must return structured failures: cross-family
    // targets, non-single-layer 2D views and incomplete cube face groups.
    CHECK(is_error(fixture.cache->acquire(
        view_request(plain, TextureType::Texture3D, 0, 1))));
    CHECK(is_error(fixture.cache->acquire(
        view_request(cube, TextureType::Texture2D, 0, 1))));
    CHECK(is_error(fixture.cache->acquire(
        view_request(plain, TextureType::Texture2DArray, 0, 2))));
    CHECK(is_error(fixture.cache->acquire(
        view_request(cube, TextureType::CubemapArray, 3, 6))));
    CHECK(is_error(fixture.cache->acquire(
        view_request(cube, TextureType::CubemapArray, 0, 12))));
}

}  // namespace

int main() {
    @autoreleasepool {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::fputs("Texture cache GPU tests require a Metal 4 device.\n", stderr);
            return 1;
        }
        test_factory_rejections(device);
        test_all_targets_and_layouts(device);
        test_same_dimensions_target_views(device);
        test_reuse_versions_views_and_aliases(device);
        test_identical_request_fast_path(device);
        test_two_dimensional_depth_blocks(device);
        test_rendered_cube_face_composes_by_copy(device);
        test_colour_rendered_depth_layer_composes_through_staging(device);
        test_attachment_supersedes_other_description(device);
        test_cpu_epoch_snapshots(device);
        test_cpu_epoch_padding_and_budget(device);
        test_stamp_certificates(device);
        test_default_registry_hashes_revalidations(device);
        test_render_target_layout_writeback(device);
        test_render_to_sample_and_incompatible_alias(device);
        test_depth_alias_sampler_lookup(device);
        test_depth_alias_compare_sampling(device);
        test_depth_alias_layout_matches_descriptor(device);
        test_delayed_producer_and_inflight_eviction(device);
        test_resolved_incompatible_alias_versions(device);
        test_prefix_mip_upload_and_writeback(device);
        test_shared_cache_eviction(device);
        test_resolved_depth_residency(device);
        test_depth_snapshot_epoch_and_retirement(device);
        test_depth_stencil_preservation(device);
        test_invalidation_and_partial_resolve(device);
        test_mapping_invalidation_and_remap(device);
        test_srgb_and_staging_preflight(device);
        test_compressed_srgb_aliases(device);
        test_negative_admission_and_dirty_budget(device);
        std::printf("Metal texture cache failures: %u\n", failures);
        return failures == 0 ? 0 : 1;
    }
}
