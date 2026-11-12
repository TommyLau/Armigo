// ============================================================================
// Armigo NVN - Null Device
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
// Stage-A null device: the NVN object surface a title touches before any
// GPU work exists. Devices, queues, windows, programs, syncs, memory pools,
// buffers, textures, samplers and descriptor pools keep full state in place
// (objects.h); texture storage sizes follow the console block-linear layout
// so the game's pool allocations are correct. Command buffers accept the
// whole recording surface and record nothing; a queue submit produces a fence
// value that is complete the moment it is issued, so syncs are immediate.
// Entries outside this stage stay on the entry table's reporting stubs.
//
// Object states live in the storage the game allocated; games copy objects
// by value, so nothing host-side is keyed by an object address. Serials come
// from one process-wide counter shared by every object type that issues them.

#include "entry_table.h"
#include "redirect.h"
#include "objects.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace armigo::nvn {
namespace {

// ---------------------------------------------------------------- shared helpers

std::atomic<std::uint64_t> serial_counter{1};
std::atomic<std::uint64_t> fence_counter{0};

std::uint64_t new_serial() noexcept { return serial_counter.fetch_add(1, std::memory_order_relaxed); }

// The fence value of the most recent submission; every issued fence is
// complete immediately in the null device.
std::uint64_t latest_fence() noexcept { return fence_counter.load(std::memory_order_acquire); }

// Reports a condition the null device accepts without acting on it, once.
void report_once(std::atomic_flag& reported, const char* what) noexcept {
    if (!reported.test_and_set(std::memory_order_relaxed))
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN null device: %s\n", what);
}

template <typename State, typename Storage>
State* state_of(Storage* object) noexcept {
    return reinterpret_cast<State*>(object);
}
template <typename State, typename Storage>
const State* state_of(const Storage* object) noexcept {
    return reinterpret_cast<const State*>(object);
}

// Silent acceptance of a call whose effects belong to a later stage: debug
// labels and every recording command.
template <typename... Params>
void accept_nothing(Params...) noexcept {}

constexpr std::uint32_t tag_device = object_tag('D', 'E', 'V', 0);
constexpr std::uint32_t tag_queue = object_tag('Q', 'U', 'E', 'U');
constexpr std::uint32_t tag_pool = object_tag('P', 'O', 'O', 'L');
constexpr std::uint32_t tag_texture = object_tag('T', 'E', 'X', 0);
constexpr std::uint32_t tag_sampler = object_tag('S', 'M', 'P', 0);
constexpr std::uint32_t tag_program = object_tag('P', 'R', 'O', 'G');
constexpr std::uint32_t tag_sync = object_tag('S', 'Y', 'N', 'C');
constexpr std::uint32_t tag_window = object_tag('W', 'I', 'N', 'D');
constexpr std::uint32_t tag_command_buffer = object_tag('C', 'M', 'D', 'B');
constexpr std::uint32_t tag_buffer = object_tag('B', 'U', 'F', 0);

// ---------------------------------------------------------------- device properties

// Property values of the target console GPU; the API version reported is the
// selected one (55.15 when none is selected). Values
// outside the table answer zero; a query the table does not describe is
// reported once instead of guessed.
constexpr std::size_t device_info_count = 0x62;
constexpr auto device_info_values = [] {
    std::array<int, device_info_count> values{};
    values[NVN_DEVICE_INFO_API_MAJOR_VERSION] = 55;
    values[NVN_DEVICE_INFO_API_MINOR_VERSION] = 15;
    values[NVN_DEVICE_INFO_UNIFORM_BUFFER_BINDINGS_PER_STAGE] = 14;
    values[NVN_DEVICE_INFO_MAX_UNIFORM_BUFFER_SIZE] = 65536;
    values[NVN_DEVICE_INFO_UNIFORM_BUFFER_ALIGNMENT] = 256;
    values[NVN_DEVICE_INFO_COLOR_BUFFER_BINDINGS] = 8;
    values[NVN_DEVICE_INFO_VERTEX_BUFFER_BINDINGS] = 16;
    values[NVN_DEVICE_INFO_TRANSFORM_FEEDBACK_BUFFER_BINDINGS] = 4;
    values[NVN_DEVICE_INFO_SHADER_STORAGE_BUFFER_BINDINGS_PER_STAGE] = 16;
    values[NVN_DEVICE_INFO_TEXTURE_BINDINGS_PER_STAGE] = 32;
    values[NVN_DEVICE_INFO_COUNTER_ALIGNMENT] = 16;
    values[NVN_DEVICE_INFO_TRANSFORM_FEEDBACK_BUFFER_ALIGNMENT] = 4;
    values[NVN_DEVICE_INFO_TRANSFORM_FEEDBACK_CONTROL_ALIGNMENT] = 4;
    values[NVN_DEVICE_INFO_INDIRECT_DRAW_ALIGNMENT] = 4;
    values[NVN_DEVICE_INFO_VERTEX_ATTRIBUTES] = 16;
    values[NVN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE] = 32;
    values[NVN_DEVICE_INFO_SAMPLER_DESCRIPTOR_SIZE] = 32;
    values[NVN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS] = 256;
    values[NVN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS] = 256;
    values[NVN_DEVICE_INFO_COMMAND_BUFFER_COMMAND_ALIGNMENT] = 4;
    values[NVN_DEVICE_INFO_COMMAND_BUFFER_CONTROL_ALIGNMENT] = 8;
    values[NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE] = 512;
    values[NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE] = 256;
    values[NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_MINIMUM] = 2;
    values[NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_RECOMMENDED] = 256;
    values[NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_ALIGNMENT] = 4096;
    values[NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_GRANULARITY] = 131072;
    values[NVN_DEVICE_INFO_MAX_TEXTURE_ANISOTROPY] = 16;
    values[NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_X] = 1536;
    values[NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Y] = 1024;
    values[NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Z] = 64;
    values[NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_THREADS] = 1536;
    values[NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_X] = 65535;
    values[NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Y] = 65535;
    values[NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Z] = 65535;
    values[NVN_DEVICE_INFO_IMAGE_BINDINGS_PER_STAGE] = 8;
    values[NVN_DEVICE_INFO_MAX_TEXTURE_POOL_SIZE] = 1048576;
    values[NVN_DEVICE_INFO_MAX_SAMPLER_POOL_SIZE] = 4096;
    values[NVN_DEVICE_INFO_MAX_VIEWPORTS] = 16;
    values[NVN_DEVICE_INFO_MEMPOOL_TEXTURE_OBJECT_PAGE_ALIGNMENT] = 4096;
    values[NVN_DEVICE_INFO_SUPPORTS_MIN_MAX_FILTERING] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_STENCIL8_FORMAT] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_ASTC_FORMATS] = 1;
    values[NVN_DEVICE_INFO_L2_SIZE] = 262144;
    values[NVN_DEVICE_INFO_MAX_TEXTURE_LEVELS] = 15;
    values[NVN_DEVICE_INFO_MAX_TEXTURE_LAYERS] = 2048;
    values[NVN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MAJOR_VERSION] = 1;
    values[NVN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MAJOR_VERSION] = 1;
    values[NVN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MINOR_VERSION] = 16;
    values[NVN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MINOR_VERSION] = 5;
    values[NVN_DEVICE_INFO_SUPPORTS_CONSERVATIVE_RASTER] = 1;
    values[NVN_DEVICE_INFO_SUBPIXEL_BITS] = 8;
    values[NVN_DEVICE_INFO_MAX_SUBPIXEL_BIAS_BITS] = 8;
    values[NVN_DEVICE_INFO_INDIRECT_DISPATCH_ALIGNMENT] = 4;
    values[NVN_DEVICE_INFO_ZCULL_SAVE_RESTORE_ALIGNMENT] = 32;
    values[NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_COMPUTE_SCALE_FACTOR_MINIMUM] = 128;
    values[NVN_DEVICE_INFO_LINEAR_TEXTURE_STRIDE_ALIGNMENT] = 32;
    values[NVN_DEVICE_INFO_LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT] = 128;
    values[NVN_DEVICE_INFO_MEMORY_POOL_PAGE_SIZE] = 65536;
    values[NVN_DEVICE_INFO_SUPPORTS_ZERO_FROM_UNMAPPED_VIRTUAL_POOL_PAGES] = 1;
    values[NVN_DEVICE_INFO_UNIFORM_BUFFER_UPDATE_ALIGNMENT] = 4;
    values[NVN_DEVICE_INFO_MAX_TEXTURE_SIZE] = 16384;
    values[NVN_DEVICE_INFO_MAX_BUFFER_TEXTURE_SIZE] = 134217728;
    values[NVN_DEVICE_INFO_MAX_3D_TEXTURE_SIZE] = 2048;
    values[NVN_DEVICE_INFO_MAX_CUBE_MAP_TEXTURE_SIZE] = 16384;
    values[NVN_DEVICE_INFO_MAX_RECTANGLE_TEXTURE_SIZE] = 16384;
    values[NVN_DEVICE_INFO_SUPPORTS_PASSTHROUGH_GEOMETRY_SHADERS] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_VIEWPORT_SWIZZLE] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_SPARSE_TILED_PACKAGED_TEXTURES] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_ADVANCED_BLEND_MODES] = 1;
    values[NVN_DEVICE_INFO_MAX_PRESENT_INTERVAL] = 5;
    values[NVN_DEVICE_INFO_SUPPORTS_DRAW_TEXTURE] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_TARGET_INDEPENDENT_RASTERIZATION] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_COVERAGE_TO_COLOR] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_POST_DEPTH_COVERAGE] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_IMAGES_USING_TEXTURE_HANDLES] = 1;
    values[NVN_DEVICE_INFO_SUPPORTS_SAMPLE_LOCATIONS] = 1;
    values[NVN_DEVICE_INFO_MAX_SAMPLE_LOCATION_TABLE_ENTRIES] = 16;
    values[NVN_DEVICE_INFO_SHADER_CODE_MEMORY_POOL_PADDING_SIZE] = 1024;
    values[NVN_DEVICE_INFO_MAX_PATCH_SIZE] = 32;
    values[NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_GRANULARITY] = 4096;
    values[NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_MIN_SIZE] = 65536;
    values[NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_DEFAULT_SIZE] = 65536;
    values[NVN_DEVICE_INFO_QUEUE_COMPUTE_MEMORY_GRANULARITY] = 256;
    values[NVN_DEVICE_INFO_QUEUE_COMPUTE_MEMORY_MIN_SIZE] = 16384;
    values[NVN_DEVICE_INFO_QUEUE_COMPUTE_MEMORY_DEFAULT_SIZE] = 262144;
    values[NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_MIN_FLUSH_THRESHOLD] = 4096;
    values[NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_SHADER_INTERLOCK] = 1;
    values[NVN_DEVICE_INFO_MAX_TEXTURES_PER_WINDOW] = 4;
    values[NVN_DEVICE_INFO_MIN_TEXTURES_PER_WINDOW] = 2;
    values[NVN_DEVICE_INFO_SUPPORTS_DEBUG_LAYER] = 0;
    values[NVN_DEVICE_INFO_QUEUE_CONTROL_MEMORY_MIN_SIZE] = 4096;
    values[NVN_DEVICE_INFO_QUEUE_CONTROL_MEMORY_DEFAULT_SIZE] = 16384;
    values[NVN_DEVICE_INFO_QUEUE_CONTROL_MEMORY_GRANULARITY] = 4096;
    values[NVN_DEVICE_INFO_SEPARATE_TEXTURE_BINDINGS_PER_STAGE] = 128;
    values[NVN_DEVICE_INFO_SEPARATE_SAMPLER_BINDINGS_PER_STAGE] = 32;
    values[NVN_DEVICE_INFO_DEBUG_GROUPS_MAX_DOMAIN_ID] = 65535;
    values[NVN_DEVICE_INFO_EVENTS_SUPPORT_REDUCTION_OPERATIONS] = 1;
    return values;
}();

// ---------------------------------------------------------------- texture storage layout

// One GOB is 64 bytes by 8 rows; the block-linear layout stacks GOBs into
// blocks whose height (and depth, for 3D) is a power-of-two GOB count.
struct FormatBlock {
    std::uint32_t bytes, width, height;  // per block; 1x1 for uncompressed
};

constexpr FormatBlock format_block(std::int32_t format) noexcept {
    if (format >= 0x79 && format <= 0x94) {  // ASTC: 16-byte blocks, footprint by format order
        constexpr std::uint8_t astc[14][2] = {{4, 4},  {5, 4},  {5, 5},  {6, 5},  {6, 6},  {8, 5},  {8, 6},
                                              {8, 8},  {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12}};
        const unsigned kind = static_cast<unsigned>(format - 0x79) % 14;
        return {16, astc[kind][0], astc[kind][1]};
    }
    if (format >= 0x41 && format <= 0x50) {  // DXT / RGTC / BPTC: 4x4 blocks
        const bool eight = format <= 0x42 || (format >= 0x45 && format <= 0x46) || (format >= 0x49 && format <= 0x4a);
        return {eight ? 8u : 16u, 4, 4};
    }
    constexpr struct {
        std::int32_t lo, hi;
        std::uint32_t bytes;
    } ranges[] = {
        {0x01, 0x04, 1},   {0x05, 0x09, 2},   {0x0a, 0x0c, 4},   {0x0d, 0x10, 2},   {0x11, 0x15, 4},
        {0x16, 0x18, 8},   {0x19, 0x1c, 3},   {0x1d, 0x21, 6},   {0x22, 0x24, 12},  {0x25, 0x28, 4},
        {0x29, 0x2d, 8},   {0x2e, 0x30, 16},  {0x31, 0x31, 1},   {0x32, 0x32, 2},   {0x33, 0x35, 4},
        {0x36, 0x36, 8},   {0x37, 0x38, 4},   {0x39, 0x3c, 2},   {0x3d, 0x40, 4},   {0x51, 0x52, 1},
        {0x53, 0x54, 2},   {0x55, 0x56, 4},   {0x57, 0x58, 2},   {0x59, 0x5a, 4},   {0x5b, 0x5c, 8},
        {0x5d, 0x5e, 3},   {0x5f, 0x60, 6},   {0x61, 0x62, 12},  {0x63, 0x64, 4},   {0x65, 0x66, 8},
        {0x67, 0x68, 16},  {0x69, 0x6c, 4},   {0x6d, 0x70, 4},   {0x71, 0x75, 8},   {0x76, 0x78, 16},
        {0x95, 0x98, 2},   {0x99, 0x9c, 4},
    };
    for (const auto& range : ranges)
        if (format >= range.lo && format <= range.hi) return {range.bytes, 1, 1};
    return {0, 1, 1};  // unknown format: no storage
}

struct TextureShape {
    std::uint32_t width = 1, height = 1, depth = 1;
    unsigned dimensions_per_layer = 2;  // 1, 2 or 3 addressable dimensions
    bool layered = false;
};

TextureShape texture_shape(const TextureDescription& description) noexcept {
    TextureShape shape;
    shape.width = description.width;
    shape.height = description.height;
    shape.depth = description.depth;
    switch (description.target) {
    case NVN_TEXTURE_TARGET_1D:
        shape.dimensions_per_layer = 1;
        shape.height = 1;
        shape.depth = 1;
        break;
    case NVN_TEXTURE_TARGET_1D_ARRAY:  // the layer count arrives in height
        shape.dimensions_per_layer = 1;
        shape.depth = description.height;
        shape.height = 1;
        shape.layered = true;
        break;
    case NVN_TEXTURE_TARGET_3D:
        shape.dimensions_per_layer = 3;
        break;
    case NVN_TEXTURE_TARGET_2D_ARRAY:
    case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
        shape.dimensions_per_layer = 2;
        shape.layered = true;
        break;
    case NVN_TEXTURE_TARGET_CUBEMAP:
        shape.dimensions_per_layer = 2;
        shape.depth = 6;
        shape.layered = true;
        break;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY:
        shape.dimensions_per_layer = 2;
        shape.depth = description.depth * 6;
        shape.layered = true;
        break;
    default:  // 2D, 2D multisample, rectangle
        shape.depth = 1;
        break;
    }
    if (shape.depth == 0) shape.depth = 1;
    return shape;
}

void sample_scale(std::int32_t samples, std::uint32_t& x, std::uint32_t& y) noexcept {
    x = y = 1;
    if (samples == 2) x = 2;
    else if (samples == 4) x = y = 2;
    else if (samples == 8) { x = 4; y = 2; }
}

std::uint32_t mip_size(std::uint32_t size, unsigned level) noexcept {
    if (level >= 32) return 1;
    size >>= level;
    return size ? size : 1;
}

std::uint32_t block_count(std::uint32_t size, std::uint32_t block) noexcept { return (size + block - 1) / block; }

// Block height (in GOBs, log2) chosen from the row count: 16 or more rows of
// GOBs want a 16-GOB block, and so on down.
unsigned pick_tile(std::uint32_t gobs) noexcept {
    return gobs >= 16 ? 4u : gobs >= 8 ? 3u : gobs >= 4 ? 2u : gobs >= 2 ? 1u : 0u;
}

// A level too small to fill its block uses a smaller one: shrink the block
// while the previous size already covers the level's extent.
unsigned adjusted_tile(unsigned shift, std::uint32_t unit, std::uint32_t extent) noexcept {
    if (shift == 0) return 0;
    std::uint32_t covered = unit << (shift - 1);
    if (covered >= extent) {
        while (--shift) {
            covered >>= 1;
            if (covered < extent) break;
        }
    }
    return shift;
}

std::uint64_t level_bytes(const TextureDescription& description, const TextureShape& shape, const FormatBlock& block,
                          unsigned level, unsigned tile_height, unsigned tile_depth) noexcept {
    if (level > 32) return 0;
    std::uint32_t sample_x, sample_y;
    sample_scale(description.samples, sample_x, sample_y);
    std::uint64_t offset = 0;
    for (unsigned i = 0; i < level; ++i) {
        const std::uint32_t width_blocks = block_count(mip_size(shape.width * sample_x, i), block.width);
        const std::uint32_t height_blocks =
            shape.dimensions_per_layer >= 2 ? block_count(mip_size(shape.height * sample_y, i), block.height) : 1;
        const std::uint32_t depth = shape.dimensions_per_layer >= 3 ? mip_size(shape.depth, i) : 1;
        const std::uint32_t width_bytes = width_blocks * block.bytes;
        const unsigned height_tile = adjusted_tile(tile_height, 8, height_blocks);
        const unsigned depth_tile = adjusted_tile(tile_depth, 1, depth);
        const std::uint32_t width_gobs = (width_bytes + 63) / 64;
        const std::uint32_t height_rows = (height_blocks + 7) / 8;
        const std::uint32_t width_tiles = width_gobs;
        const std::uint32_t height_tiles = (height_rows + (1u << height_tile) - 1) >> height_tile;
        const std::uint32_t depth_tiles = (depth + (1u << depth_tile) - 1) >> depth_tile;
        offset += static_cast<std::uint64_t>(width_tiles) * height_tiles * depth_tiles << (9 + height_tile + depth_tile);
    }
    return offset;
}

struct TextureLayout {
    std::uint64_t size = 0, layer_size = 0;
    std::uint32_t alignment = 512, layers = 1, block_bytes = 0;
    unsigned tile_height = 0, tile_depth = 0;  // GOBs per block, log2
};

TextureLayout texture_storage_layout(const TextureDescription& description) noexcept {
    TextureLayout layout;
    const auto maximum_width = description.target == NVN_TEXTURE_TARGET_BUFFER
        ? device_info_values[NVN_DEVICE_INFO_MAX_BUFFER_TEXTURE_SIZE]
        : device_info_values[NVN_DEVICE_INFO_MAX_TEXTURE_SIZE];
    const auto maximum_depth = description.target == NVN_TEXTURE_TARGET_3D
        ? device_info_values[NVN_DEVICE_INFO_MAX_3D_TEXTURE_SIZE]
        : device_info_values[NVN_DEVICE_INFO_MAX_TEXTURE_LAYERS];
    if (description.levels > 32 || description.packaged_block_height < 0 ||
        description.packaged_block_height > 6 || description.width > std::uint32_t(maximum_width) ||
        description.height > std::uint32_t(device_info_values[NVN_DEVICE_INFO_MAX_TEXTURE_SIZE]) ||
        description.depth > std::uint32_t(maximum_depth)) {
        static std::atomic_flag reported;
        report_once(reported, "texture layout parameters exceed the supported device limits");
        return layout;
    }
    const FormatBlock block = format_block(description.format);
    const TextureShape shape = texture_shape(description);
    layout.block_bytes = block.bytes;
    layout.layers = shape.layered ? shape.depth : 1;
    if (block.bytes == 0) return layout;  // unknown format: zero size, GOB alignment

    if (description.target == NVN_TEXTURE_TARGET_BUFFER) {
        layout.layer_size = layout.size = static_cast<std::uint64_t>(block.bytes) * description.width;
        layout.alignment = block.bytes == 12 ? 4 : block.bytes;
        return layout;
    }

    // Pitch layout for linear textures and linear render targets: one level,
    // row pitch from the stride or aligned to the layout's pitch alignment.
    if (description.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) {
        layout.alignment = (description.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) ? 128 : 32;
        const std::uint32_t stride = description.stride
                                         ? description.stride
                                         : (block.bytes * block_count(shape.width, block.width) + layout.alignment - 1) &
                                               ~(layout.alignment - 1);
        layout.layer_size = layout.size = static_cast<std::uint64_t>(stride) * block_count(shape.height, block.height);
        return layout;
    }

    if (shape.dimensions_per_layer == 2) {
        std::uint32_t sample_x, sample_y;
        sample_scale(description.samples, sample_x, sample_y);
        const std::uint32_t sampled_rows = block_count(shape.height * sample_y, block.height);
        // A packaged layout fixes the block height (byte 0 carries log2 plus
        // one); otherwise it follows the row count with one half-row of slack.
        layout.tile_height =
            description.packaged_block_height
                ? static_cast<unsigned>(description.packaged_block_height) - 1
                : pick_tile((sampled_rows + sampled_rows / 2 + 7) / 8);
    } else if (shape.dimensions_per_layer == 3) {
        layout.tile_depth = pick_tile(shape.depth + shape.depth / 2);
    }

    const unsigned levels = description.levels ? description.levels : 1;
    std::uint64_t layer_size = level_bytes(description, shape, block, levels, layout.tile_height, layout.tile_depth);
    if (shape.layered) {
        // Layers start block-aligned: shrink the block to the base level and
        // round the layer size up to the remaining block.
        const std::uint32_t block_rows = block_count(shape.height, block.height);
        unsigned height_tile = layout.tile_height, depth_tile = layout.tile_depth;
        while (height_tile && block_rows <= (8u << (height_tile - 1))) --height_tile;
        while (depth_tile && shape.depth <= (1u << (depth_tile - 1))) --depth_tile;
        const unsigned shift = height_tile + depth_tile + 9;
        const std::uint64_t blocks = layer_size >> shift;
        if (layer_size != blocks << shift) layer_size = (blocks + 1) << shift;
    }
    layout.layer_size = layer_size;
    layout.size = shape.layered ? layer_size * shape.depth : layer_size;
    layout.alignment = 512;
    return layout;
}

std::uint64_t texture_level_bytes(const TextureDescription& description, unsigned level) noexcept {
    const FormatBlock block = format_block(description.format);
    if (block.bytes == 0 || (description.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) ||
        description.target == NVN_TEXTURE_TARGET_BUFFER)
        return 0;  // pitch and buffer layouts have a single level
    const TextureShape shape = texture_shape(description);
    const TextureLayout layout = texture_storage_layout(description);
    return level_bytes(description, shape, block, level, layout.tile_height, layout.tile_depth);
}

// ---------------------------------------------------------------- device

void device_builder_set_defaults(NVNdeviceBuilder* builder) noexcept {
    if (!builder) return;
    *state_of<DeviceBuilderState>(builder) = DeviceBuilderState{};
}

void device_builder_set_flags(NVNdeviceBuilder* builder, int flags) noexcept {
    if (!builder) return;
    state_of<DeviceBuilderState>(builder)->flags = static_cast<std::uint32_t>(flags);
}

NVNboolean device_initialize(NVNdevice* device, const NVNdeviceBuilder* builder) noexcept {
    if (!device) return 0;
    auto* state = state_of<DeviceState>(device);
    *state = DeviceState{};
    state->tag = tag_device;
    state->flags = builder ? state_of<DeviceBuilderState>(builder)->flags : 0;
    const auto version = selected_api_version();
    state->gob_order = version && version->major == 231 ? 1 : 0;
    return 1;  // window origin and depth mode keep the API defaults
}

void device_finalize(NVNdevice* device) noexcept {
    if (device) state_of<DeviceState>(device)->tag = 0;
}

PFNNVNGENERICFUNCPTRPROC device_get_proc_address(const NVNdevice*, const char* name) noexcept {
    return armigo_nvn_bootstrap_loader(name);
}

void device_get_integer(const NVNdevice*, NVNdeviceInfo info, int* out) noexcept {
    if (!out) return;
    static std::atomic_flag reported_unknown;
    if (info < 0 || static_cast<unsigned>(info) >= device_info_count) {
        report_once(reported_unknown, "unknown device property queried, returning zero");
        *out = 0;
        return;
    }
    *out = device_info_values[static_cast<std::size_t>(info)];
    if (const auto version = selected_api_version()) {
        if (info == NVN_DEVICE_INFO_API_MAJOR_VERSION) *out = version->major;
        if (info == NVN_DEVICE_INFO_API_MINOR_VERSION) *out = version->minor;
    }
}

NVNtextureHandle device_get_texture_handle(const NVNdevice*, int texture, int sampler) noexcept {
    return combined_handle(texture, sampler);
}

NVNtextureHandle device_get_texel_fetch_handle(const NVNdevice*, int texture) noexcept {
    return texture_handle(texture);
}

NVNimageHandle device_get_image_handle(const NVNdevice*, int id) noexcept { return static_cast<NVNimageHandle>(id); }

NVNseparateTextureHandle device_get_separate_texture_handle(const NVNdevice*, int id) noexcept {
    return NVNseparateTextureHandle{texture_handle(id)};
}

NVNseparateSamplerHandle device_get_separate_sampler_handle(const NVNdevice*, int id) noexcept {
    return NVNseparateSamplerHandle{sampler_handle(id)};
}

void device_install_debug_callback(NVNdevice* device, const PFNNVNDEBUGCALLBACKPROC callback, void* data,
                                   std::uint32_t enable) noexcept {
    if (!device) return;
    auto* state = state_of<DeviceState>(device);
    state->debug_callback = guest_bool(enable) ? reinterpret_cast<void*>(callback) : nullptr;
    state->debug_data = data;
}

void device_set_window_origin_mode(NVNdevice* device, NVNwindowOriginMode mode) noexcept {
    if (device) state_of<DeviceState>(device)->window_origin = static_cast<std::uint32_t>(mode);
}

NVNwindowOriginMode device_get_window_origin_mode(const NVNdevice* device) noexcept {
    return device ? static_cast<NVNwindowOriginMode>(state_of<DeviceState>(device)->window_origin)
                  : NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
}

void device_set_depth_mode(NVNdevice* device, NVNdepthMode mode) noexcept {
    if (device) state_of<DeviceState>(device)->depth_mode = static_cast<std::uint32_t>(mode);
}

NVNdepthMode device_get_depth_mode(const NVNdevice* device) noexcept {
    return device ? static_cast<NVNdepthMode>(state_of<DeviceState>(device)->depth_mode) : NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
}

// Fast-clear registration is a hardware hint the null device has no use for;
// normal clears are unaffected and the registration is declined.
NVNboolean decline_fast_clear(const char* what) noexcept {
    static std::atomic_flag reported;
    report_once(reported, what);
    return 0;
}
NVNboolean device_decline_fast_clear(NVNdevice*, const float*, NVNformat) noexcept {
    return decline_fast_clear("fast-clear color registration declined; normal clears are unaffected");
}
NVNboolean device_decline_fast_cleari(NVNdevice*, const int*, NVNformat) noexcept {
    return decline_fast_clear("fast-clear color registration declined; normal clears are unaffected");
}
NVNboolean device_decline_fast_clearui(NVNdevice*, const std::uint32_t*, NVNformat) noexcept {
    return decline_fast_clear("fast-clear color registration declined; normal clears are unaffected");
}
NVNboolean device_decline_fast_clear_depth(NVNdevice*, float) noexcept {
    return decline_fast_clear("fast-clear depth registration declined; normal clears are unaffected");
}

// ---------------------------------------------------------------- memory pools

void memory_pool_builder_set_defaults(NVNmemoryPoolBuilder* builder) noexcept {
    if (!builder) return;
    *state_of<MemoryPoolBuilderState>(builder) = MemoryPoolBuilderState{};
}

void memory_pool_builder_set_device(NVNmemoryPoolBuilder* builder, NVNdevice* device) noexcept {
    if (builder) state_of<MemoryPoolBuilderState>(builder)->device = device;
}

void memory_pool_builder_set_storage(NVNmemoryPoolBuilder* builder, void* memory, std::size_t size) noexcept {
    if (!builder) return;
    auto* state = state_of<MemoryPoolBuilderState>(builder);
    state->memory = memory;
    state->size = size;
}

void memory_pool_builder_set_flags(NVNmemoryPoolBuilder* builder, int flags) noexcept {
    if (builder) state_of<MemoryPoolBuilderState>(builder)->flags = static_cast<std::uint32_t>(flags);
}

std::size_t memory_pool_builder_get_size(const NVNmemoryPoolBuilder* builder) noexcept {
    return builder ? state_of<MemoryPoolBuilderState>(builder)->size : 0;
}

NVNmemoryPoolFlags memory_pool_builder_get_flags(const NVNmemoryPoolBuilder* builder) noexcept {
    return builder ? static_cast<NVNmemoryPoolFlags>(state_of<MemoryPoolBuilderState>(builder)->flags)
                   : NVNmemoryPoolFlags{0};
}

NVNboolean memory_pool_initialize(NVNmemoryPool* pool, const NVNmemoryPoolBuilder* builder) noexcept {
    if (!pool) return 0;
    static std::atomic_flag reported_live;
    static std::atomic_flag reported_storageless;
    auto* state = state_of<MemoryPoolState>(pool);
    const auto* source = builder ? state_of<MemoryPoolBuilderState>(builder) : nullptr;
    if (state->tag == tag_pool) report_once(reported_live, "live pool reinitialized without finalize");
    *state = MemoryPoolState{};
    state->tag = tag_pool;
    state->flags = source ? source->flags : 0;
    state->memory = source ? static_cast<std::uint8_t*>(source->memory) : nullptr;
    state->size = source ? source->size : 0;
    if (!state->memory && !(state->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL))
        report_once(reported_storageless, "pool without storage is not virtual; accepted with no backing");
    return 1;
}

void memory_pool_finalize(NVNmemoryPool* pool) noexcept {
    if (pool) state_of<MemoryPoolState>(pool)->tag = 0;
}

void* memory_pool_map(const NVNmemoryPool* pool) noexcept {
    if (!pool) return nullptr;
    const auto* state = state_of<MemoryPoolState>(pool);
    if (state->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return nullptr;
    return state->memory;
}

void memory_pool_flush_mapped_range(const NVNmemoryPool* pool, ptrdiff_t offset, std::size_t size) noexcept {
    if (!pool) return;
    auto* state = state_of<MemoryPoolState>(const_cast<NVNmemoryPool*>(pool));
    // Only CPU-cached pools flush through the data cache; the latest range
    // wins, ordered by its serial (end zero while the range is incomplete).
    if (state->tag != tag_pool || !(state->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) || offset < 0 || size == 0) return;
    state->flush_end = 0;
    state->flush_begin = static_cast<std::uint64_t>(offset);
    state->flush_serial = new_serial();
    state->flush_end = static_cast<std::uint64_t>(offset) + size;
}

void memory_pool_invalidate_mapped_range(const NVNmemoryPool*, ptrdiff_t, std::size_t) noexcept {
    // The null device moves nothing between CPU and GPU; ordering with GPU
    // work is moot until submissions exist.
}

NVNbufferAddress memory_pool_get_buffer_address(const NVNmemoryPool* pool) noexcept {
    if (!pool) return 0;
    const auto* state = state_of<MemoryPoolState>(pool);
    // GPU addresses equal guest addresses; physical-only pools expose none.
    if (state->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) return 0;
    return reinterpret_cast<NVNbufferAddress>(state->memory);
}

std::size_t memory_pool_get_size(const NVNmemoryPool* pool) noexcept {
    return pool ? state_of<MemoryPoolState>(pool)->size : 0;
}

NVNmemoryPoolFlags memory_pool_get_flags(const NVNmemoryPool* pool) noexcept {
    return pool ? static_cast<NVNmemoryPoolFlags>(state_of<MemoryPoolState>(pool)->flags) : NVNmemoryPoolFlags{0};
}

// ---------------------------------------------------------------- descriptor pools

NVNboolean texture_pool_initialize(NVNtexturePool* texture_pool, const NVNmemoryPool* pool, ptrdiff_t offset,
                                   int count) noexcept {
    if (!texture_pool) return 0;
    *state_of<DescriptorPoolState>(texture_pool) =
        DescriptorPoolState{pool ? state_of<MemoryPoolState>(pool) : nullptr, offset, count};
    return 1;
}

void texture_pool_finalize(NVNtexturePool* texture_pool) noexcept {
    if (texture_pool) *state_of<DescriptorPoolState>(texture_pool) = DescriptorPoolState{};
}

void texture_pool_register_texture(const NVNtexturePool*, int, const NVNtexture*, const NVNtextureView*) noexcept {
    // Descriptor slots become live with the resource backend.
}

void texture_pool_register_image(const NVNtexturePool*, int, const NVNtexture*, const NVNtextureView*) noexcept {}

const NVNmemoryPool* texture_pool_get_memory_pool(const NVNtexturePool* texture_pool) noexcept {
    if (!texture_pool) return nullptr;
    const auto* state = state_of<DescriptorPoolState>(texture_pool);
    return reinterpret_cast<const NVNmemoryPool*>(state->pool);
}

ptrdiff_t texture_pool_get_memory_offset(const NVNtexturePool* texture_pool) noexcept {
    return texture_pool ? state_of<DescriptorPoolState>(texture_pool)->offset : 0;
}

int texture_pool_get_size(const NVNtexturePool* texture_pool) noexcept {
    return texture_pool ? state_of<DescriptorPoolState>(texture_pool)->count : 0;
}

NVNboolean sampler_pool_initialize(NVNsamplerPool* sampler_pool, const NVNmemoryPool* pool, ptrdiff_t offset,
                                   int count) noexcept {
    if (!sampler_pool) return 0;
    *state_of<DescriptorPoolState>(sampler_pool) =
        DescriptorPoolState{pool ? state_of<MemoryPoolState>(pool) : nullptr, offset, count};
    return 1;
}

void sampler_pool_finalize(NVNsamplerPool* sampler_pool) noexcept {
    if (sampler_pool) *state_of<DescriptorPoolState>(sampler_pool) = DescriptorPoolState{};
}

void sampler_pool_register_sampler(const NVNsamplerPool*, int, const NVNsampler*) noexcept {}

void sampler_pool_register_sampler_builder(const NVNsamplerPool*, int, const NVNsamplerBuilder*) noexcept {}

const NVNmemoryPool* sampler_pool_get_memory_pool(const NVNsamplerPool* sampler_pool) noexcept {
    if (!sampler_pool) return nullptr;
    const auto* state = state_of<DescriptorPoolState>(sampler_pool);
    return reinterpret_cast<const NVNmemoryPool*>(state->pool);
}

ptrdiff_t sampler_pool_get_memory_offset(const NVNsamplerPool* sampler_pool) noexcept {
    return sampler_pool ? state_of<DescriptorPoolState>(sampler_pool)->offset : 0;
}

int sampler_pool_get_size(const NVNsamplerPool* sampler_pool) noexcept {
    return sampler_pool ? state_of<DescriptorPoolState>(sampler_pool)->count : 0;
}

// ---------------------------------------------------------------- buffers

struct BufferBuilderState {
    NVNdevice* device;
    MemoryPoolState* pool;
    std::int64_t offset;
    std::uint64_t size;
};
static_assert(sizeof(BufferBuilderState) <= sizeof(NVNbufferBuilder), "buffer builder exceeds its storage");

struct BufferState {
    std::uint32_t tag, reserved;
    NVNdevice* device;
    MemoryPoolState* pool;
    std::int64_t offset;
    std::uint64_t size;
    std::uint64_t serial;
};
static_assert(sizeof(BufferState) <= sizeof(NVNbuffer), "buffer exceeds its storage");
static_assert(sizeof(BufferState) == 0x30, "Buffer storage is 48 bytes");

void buffer_builder_set_device(NVNbufferBuilder* builder, NVNdevice* device) noexcept {
    if (builder) state_of<BufferBuilderState>(builder)->device = device;
}

void buffer_builder_set_defaults(NVNbufferBuilder* builder) noexcept {
    if (!builder) return;
    auto* state = state_of<BufferBuilderState>(builder);
    NVNdevice* device = state->device;
    *state = BufferBuilderState{};
    state->device = device;
}

void buffer_builder_set_storage(NVNbufferBuilder* builder, NVNmemoryPool* pool, ptrdiff_t offset,
                                std::size_t size) noexcept {
    if (!builder) return;
    auto* state = state_of<BufferBuilderState>(builder);
    state->pool = pool ? state_of<MemoryPoolState>(pool) : nullptr;
    state->offset = offset;
    state->size = size;
}

NVNdevice* buffer_builder_get_device(const NVNbufferBuilder* builder) noexcept {
    return builder ? state_of<BufferBuilderState>(builder)->device : nullptr;
}

NVNmemoryPool* buffer_builder_get_memory_pool(const NVNbufferBuilder* builder) noexcept {
    if (!builder) return nullptr;
    return reinterpret_cast<NVNmemoryPool*>(state_of<BufferBuilderState>(builder)->pool);
}

ptrdiff_t buffer_builder_get_memory_offset(const NVNbufferBuilder* builder) noexcept {
    return builder ? state_of<BufferBuilderState>(builder)->offset : 0;
}

std::size_t buffer_builder_get_size(const NVNbufferBuilder* builder) noexcept {
    return builder ? state_of<BufferBuilderState>(builder)->size : 0;
}

NVNboolean buffer_initialize(NVNbuffer* buffer, const NVNbufferBuilder* builder) noexcept {
    if (!buffer || !builder) return 0;
    static std::atomic_flag reported_no_pool;
    static std::atomic_flag reported_physical;
    static std::atomic_flag reported_outside;
    const auto* source = state_of<BufferBuilderState>(builder);
    if (!source->pool || source->pool->tag != tag_pool) {
        report_once(reported_no_pool, "buffer builder has no initialized memory pool");
        return 0;
    }
    if (source->pool->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) {
        report_once(reported_physical, "buffers cannot use physical-only pools");
        return 0;
    }
    if (source->offset < 0 || static_cast<std::uint64_t>(source->offset) > source->pool->size ||
        source->size > source->pool->size - static_cast<std::uint64_t>(source->offset)) {
        report_once(reported_outside, "buffer storage lies outside its pool");
        return 0;
    }
    *state_of<BufferState>(buffer) =
        BufferState{tag_buffer, 0, source->device, source->pool, source->offset, source->size, new_serial()};
    return 1;
}

void buffer_finalize(NVNbuffer* buffer) noexcept {
    if (buffer) state_of<BufferState>(buffer)->tag = 0;
}

void* buffer_map(const NVNbuffer* buffer) noexcept {
    if (!buffer) return nullptr;
    const auto* state = state_of<BufferState>(buffer);
    if (state->tag != tag_buffer || !state->pool || !state->pool->memory) return nullptr;
    if (state->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return nullptr;
    return state->pool->memory + state->offset;
}

NVNbufferAddress buffer_get_address(const NVNbuffer* buffer) noexcept {
    if (!buffer) return 0;
    const auto* state = state_of<BufferState>(buffer);
    if (state->tag != tag_buffer || !state->pool || !state->pool->memory) return 0;
    return reinterpret_cast<NVNbufferAddress>(state->pool->memory + state->offset);
}

void buffer_flush_mapped_range(const NVNbuffer*, ptrdiff_t, std::size_t) noexcept {}
void buffer_invalidate_mapped_range(const NVNbuffer*, ptrdiff_t, std::size_t) noexcept {}

NVNmemoryPool* buffer_get_memory_pool(const NVNbuffer* buffer) noexcept {
    if (!buffer) return nullptr;
    return reinterpret_cast<NVNmemoryPool*>(state_of<BufferState>(buffer)->pool);
}

ptrdiff_t buffer_get_memory_offset(const NVNbuffer* buffer) noexcept {
    return buffer ? state_of<BufferState>(buffer)->offset : 0;
}

std::size_t buffer_get_size(const NVNbuffer* buffer) noexcept {
    return buffer ? state_of<BufferState>(buffer)->size : 0;
}

std::uint64_t buffer_get_debug_id(const NVNbuffer* buffer) noexcept {
    return buffer ? state_of<BufferState>(buffer)->serial : 0;
}

// ---------------------------------------------------------------- texture builder

void texture_builder_set_defaults(NVNtextureBuilder* builder) noexcept {
    if (!builder) return;
    auto* state = state_of<TextureBuilderState>(builder);
    NVNdevice* device = state->device;
    *state = TextureBuilderState{};
    state->device = device;
    state->description.target = NVN_TEXTURE_TARGET_2D;
    state->description.width = state->description.height = state->description.depth = state->description.levels = 1;
}

void texture_builder_set_device(NVNtextureBuilder* builder, NVNdevice* device) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->device = device;
}

void texture_builder_set_flags(NVNtextureBuilder* builder, int flags) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.flags = flags;
}

void texture_builder_set_target(NVNtextureBuilder* builder, NVNtextureTarget target) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.target = target;
}

void texture_builder_set_width(NVNtextureBuilder* builder, int width) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.width = static_cast<std::uint32_t>(width);
}

void texture_builder_set_height(NVNtextureBuilder* builder, int height) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.height = static_cast<std::uint32_t>(height);
}

void texture_builder_set_depth(NVNtextureBuilder* builder, int depth) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.depth = static_cast<std::uint32_t>(depth);
}

void texture_builder_set_size_1d(NVNtextureBuilder* builder, int width) noexcept {
    if (!builder) return;
    auto* description = &state_of<TextureBuilderState>(builder)->description;
    description->width = static_cast<std::uint32_t>(width);
    description->height = description->depth = 1;
}

void texture_builder_set_size_2d(NVNtextureBuilder* builder, int width, int height) noexcept {
    if (!builder) return;
    auto* description = &state_of<TextureBuilderState>(builder)->description;
    description->width = static_cast<std::uint32_t>(width);
    description->height = static_cast<std::uint32_t>(height);
    description->depth = 1;
}

void texture_builder_set_size_3d(NVNtextureBuilder* builder, int width, int height, int depth) noexcept {
    if (!builder) return;
    auto* description = &state_of<TextureBuilderState>(builder)->description;
    description->width = static_cast<std::uint32_t>(width);
    description->height = static_cast<std::uint32_t>(height);
    description->depth = static_cast<std::uint32_t>(depth);
}

void texture_builder_set_levels(NVNtextureBuilder* builder, int levels) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.levels = static_cast<std::uint32_t>(levels);
}

void texture_builder_set_format(NVNtextureBuilder* builder, NVNformat format) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.format = format;
}

void texture_builder_set_samples(NVNtextureBuilder* builder, int samples) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.samples = samples;
}

void texture_builder_set_swizzle(NVNtextureBuilder* builder, NVNtextureSwizzle r, NVNtextureSwizzle g,
                                 NVNtextureSwizzle b, NVNtextureSwizzle a) noexcept {
    if (!builder) return;
    static std::atomic_flag reported_swizzle;
    if (static_cast<unsigned>(r) > 5 || static_cast<unsigned>(g) > 5 || static_cast<unsigned>(b) > 5 ||
        static_cast<unsigned>(a) > 5) {
        report_once(reported_swizzle, "invalid texture swizzle ignored");
        return;
    }
    state_of<TextureBuilderState>(builder)->swizzle = pack_swizzle(r, g, b, a);
}

void texture_builder_set_depth_stencil_mode(NVNtextureBuilder* builder, NVNtextureDepthStencilMode mode) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->depth_stencil_mode = static_cast<std::uint32_t>(mode);
}

std::size_t texture_builder_get_storage_size(const NVNtextureBuilder* builder) noexcept {
    if (!builder) return 0;
    static std::atomic_flag reported_format;
    const TextureDescription& description = state_of<TextureBuilderState>(builder)->description;
    const TextureLayout layout = texture_storage_layout(description);
    if (layout.block_bytes == 0) report_once(reported_format, "storage size asked for an unknown format");
    return layout.size;
}

std::size_t texture_builder_get_storage_alignment(const NVNtextureBuilder* builder) noexcept {
    if (!builder) return 512;
    return texture_storage_layout(state_of<TextureBuilderState>(builder)->description).alignment;
}

void texture_builder_set_storage(NVNtextureBuilder* builder, NVNmemoryPool* pool, ptrdiff_t offset) noexcept {
    if (!builder) return;
    auto* state = state_of<TextureBuilderState>(builder);
    state->pool = pool ? state_of<MemoryPoolState>(pool) : nullptr;
    state->offset = offset;
}

void texture_builder_set_packaged_texture_layout(NVNtextureBuilder* builder,
                                                 const NVNpackagedTextureLayout* layout) noexcept {
    if (!builder) return;
    auto* state = state_of<TextureBuilderState>(builder);
    state->has_packaged_layout = layout != nullptr;
    if (layout) {
        std::memcpy(state->packaged_layout, layout->layout, sizeof state->packaged_layout);
        state->description.packaged_block_height = layout->layout[0] + 1;
    } else {
        state->description.packaged_block_height = 0;
    }
}

void texture_builder_set_stride(NVNtextureBuilder* builder, ptrdiff_t stride) noexcept {
    if (builder) state_of<TextureBuilderState>(builder)->description.stride = static_cast<std::uint32_t>(stride);
}

NVNtextureFlags texture_builder_get_flags(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<NVNtextureFlags>(state_of<TextureBuilderState>(builder)->description.flags)
                   : NVN_TEXTURE_FLAGS_DISPLAY;
}

NVNtextureTarget texture_builder_get_target(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<NVNtextureTarget>(state_of<TextureBuilderState>(builder)->description.target)
                   : NVN_TEXTURE_TARGET_2D;
}

int texture_builder_get_width(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<int>(state_of<TextureBuilderState>(builder)->description.width) : 0;
}

int texture_builder_get_height(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<int>(state_of<TextureBuilderState>(builder)->description.height) : 0;
}

int texture_builder_get_depth(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<int>(state_of<TextureBuilderState>(builder)->description.depth) : 0;
}

int texture_builder_get_levels(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<int>(state_of<TextureBuilderState>(builder)->description.levels) : 0;
}

NVNformat texture_builder_get_format(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<NVNformat>(state_of<TextureBuilderState>(builder)->description.format) : NVN_FORMAT_NONE;
}

int texture_builder_get_samples(const NVNtextureBuilder* builder) noexcept {
    return builder ? state_of<TextureBuilderState>(builder)->description.samples : 0;
}

void texture_builder_get_swizzle(const NVNtextureBuilder* builder, NVNtextureSwizzle* r, NVNtextureSwizzle* g,
                                 NVNtextureSwizzle* b, NVNtextureSwizzle* a) noexcept {
    const std::uint16_t packed = builder ? state_of<TextureBuilderState>(builder)->swizzle : 0;
    if (r) *r = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 0));
    if (g) *g = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 1));
    if (b) *b = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 2));
    if (a) *a = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 3));
}

NVNtextureDepthStencilMode texture_builder_get_depth_stencil_mode(const NVNtextureBuilder* builder) noexcept {
    return builder
               ? static_cast<NVNtextureDepthStencilMode>(state_of<TextureBuilderState>(builder)->depth_stencil_mode)
               : NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
}

ptrdiff_t texture_builder_get_stride(const NVNtextureBuilder* builder) noexcept {
    return builder ? static_cast<ptrdiff_t>(state_of<TextureBuilderState>(builder)->description.stride) : 0;
}

// The null device keeps no compression metadata of its own.
std::size_t texture_builder_get_zcull_storage_size(const NVNtextureBuilder*) noexcept { return 0; }

ptrdiff_t texture_builder_get_memory_offset(const NVNtextureBuilder* builder) noexcept {
    return builder ? state_of<TextureBuilderState>(builder)->offset : 0;
}

// ---------------------------------------------------------------- texture objects

NVNboolean texture_initialize(NVNtexture* texture, const NVNtextureBuilder* builder) noexcept {
    if (!texture || !builder) return 0;
    static std::atomic_flag reported_live;
    auto* state = state_of<TextureState>(texture);
    const auto* source = state_of<TextureBuilderState>(builder);
    const TextureLayout layout = texture_storage_layout(source->description);
    static std::atomic_flag reported_range;
    if (!layout.size || (source->pool && (source->offset < 0 ||
        static_cast<std::uint64_t>(source->offset) > source->pool->size ||
        layout.size > source->pool->size - static_cast<std::uint64_t>(source->offset)))) {
        report_once(reported_range, "texture storage is invalid or lies outside its pool");
        return 0;
    }
    if (state->tag == tag_texture)
        report_once(reported_live, "live texture reinitialized without finalize; the old serial is dropped");
    *state = TextureState{};
    state->tag = tag_texture;
    state->description = source->description;
    state->pool = source->pool;
    state->offset = source->offset;
    state->has_packaged_layout = source->has_packaged_layout;
    std::memcpy(state->packaged_layout, source->packaged_layout, sizeof state->packaged_layout);
    state->swizzle = source->swizzle;
    state->depth_stencil_mode = source->depth_stencil_mode & 3u;
    state->size = layout.size;
    state->layer_size = layout.layer_size;
    state->memory = source->pool && source->pool->memory ? source->pool->memory + source->offset : nullptr;
    state->serial = new_serial();
    state->snapshot = nullptr;  // recordings appear with the replay stage
    return 1;
}

void texture_finalize(NVNtexture* texture) noexcept {
    if (!texture) return;
    auto* state = state_of<TextureState>(texture);
    state->tag = 0;
    state->serial = 0;
}

std::size_t texture_get_zcull_storage_size(const NVNtexture*) noexcept { return 0; }

ptrdiff_t texture_get_view_offset(const NVNtexture* texture, const NVNtextureView* view) noexcept {
    if (!texture || !view) return 0;
    const auto* state = state_of<TextureState>(texture);
    const auto* view_state = state_of<TextureViewState>(view);
    std::uint64_t offset =
        view_state->sets_levels ? texture_level_bytes(state->description, view_state->base_level) : 0;
    if (view_state->sets_layers) offset += static_cast<std::int64_t>(view_state->min_layer) * state->layer_size;
    return static_cast<ptrdiff_t>(offset);
}

NVNtextureFlags texture_get_flags(const NVNtexture* texture) noexcept {
    return texture ? static_cast<NVNtextureFlags>(state_of<TextureState>(texture)->description.flags)
                   : NVN_TEXTURE_FLAGS_DISPLAY;
}

NVNtextureTarget texture_get_target(const NVNtexture* texture) noexcept {
    return texture ? static_cast<NVNtextureTarget>(state_of<TextureState>(texture)->description.target)
                   : NVN_TEXTURE_TARGET_2D;
}

int texture_get_width(const NVNtexture* texture) noexcept {
    return texture ? static_cast<int>(state_of<TextureState>(texture)->description.width) : 0;
}

int texture_get_height(const NVNtexture* texture) noexcept {
    return texture ? static_cast<int>(state_of<TextureState>(texture)->description.height) : 0;
}

int texture_get_depth(const NVNtexture* texture) noexcept {
    return texture ? static_cast<int>(state_of<TextureState>(texture)->description.depth) : 0;
}

int texture_get_levels(const NVNtexture* texture) noexcept {
    return texture ? static_cast<int>(state_of<TextureState>(texture)->description.levels) : 0;
}

NVNformat texture_get_format(const NVNtexture* texture) noexcept {
    return texture ? static_cast<NVNformat>(state_of<TextureState>(texture)->description.format) : NVN_FORMAT_NONE;
}

int texture_get_samples(const NVNtexture* texture) noexcept {
    return texture ? state_of<TextureState>(texture)->description.samples : 0;
}

void texture_get_swizzle(const NVNtexture* texture, NVNtextureSwizzle* r, NVNtextureSwizzle* g, NVNtextureSwizzle* b,
                         NVNtextureSwizzle* a) noexcept {
    const std::uint16_t packed = texture ? state_of<TextureState>(texture)->swizzle : 0;
    if (r) *r = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 0));
    if (g) *g = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 1));
    if (b) *b = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 2));
    if (a) *a = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 3));
}

NVNtextureDepthStencilMode texture_get_depth_stencil_mode(const NVNtexture* texture) noexcept {
    return texture
               ? static_cast<NVNtextureDepthStencilMode>(state_of<TextureState>(texture)->depth_stencil_mode)
               : NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
}

ptrdiff_t texture_get_stride(const NVNtexture* texture) noexcept {
    return texture ? static_cast<ptrdiff_t>(state_of<TextureState>(texture)->description.stride) : 0;
}

NVNtextureAddress texture_get_texture_address(const NVNtexture* texture) noexcept {
    return texture ? reinterpret_cast<NVNtextureAddress>(state_of<TextureState>(texture)->memory) : 0;
}

ptrdiff_t texture_get_memory_offset(const NVNtexture* texture) noexcept {
    return texture ? state_of<TextureState>(texture)->offset : 0;
}

int texture_get_storage_size(const NVNtexture* texture) noexcept {
    return texture ? static_cast<int>(state_of<TextureState>(texture)->size) : 0;
}

NVNboolean texture_compare(const NVNtexture* left, const NVNtexture* right) noexcept {
    if (!left || !right) return left == right ? 1 : 0;
    const auto* a = state_of<TextureState>(left);
    const auto* b = state_of<TextureState>(right);
    return std::memcmp(&a->description, &b->description, sizeof(TextureDescription)) == 0 && a->memory == b->memory &&
                   a->swizzle == b->swizzle && a->depth_stencil_mode == b->depth_stencil_mode
               ? 1
               : 0;
}

std::uint64_t texture_get_debug_id(const NVNtexture* texture) noexcept {
    return texture ? state_of<TextureState>(texture)->serial : 0;
}

void texture_flush_texels(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*) noexcept {}
void texture_invalidate_texels(const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*) noexcept {}

// ---------------------------------------------------------------- texture views

void texture_view_set_defaults(NVNtextureView* view) noexcept {
    if (view) *state_of<TextureViewState>(view) = TextureViewState{};
}

void texture_view_set_levels(NVNtextureView* view, int base, int count) noexcept {
    if (!view) return;
    auto* state = state_of<TextureViewState>(view);
    state->sets_levels = 1;
    state->base_level = static_cast<std::uint8_t>(base);
    state->level_count = static_cast<std::uint8_t>(count);
}

void texture_view_set_layers(NVNtextureView* view, int minimum, int count) noexcept {
    if (!view) return;
    auto* state = state_of<TextureViewState>(view);
    state->sets_layers = 1;
    state->min_layer = static_cast<std::int16_t>(minimum);
    state->layer_count = static_cast<std::int16_t>(count);
}

void texture_view_set_format(NVNtextureView* view, NVNformat format) noexcept {
    if (!view) return;
    auto* state = state_of<TextureViewState>(view);
    state->sets_format = 1;
    state->format = format;
}

void texture_view_set_swizzle(NVNtextureView* view, NVNtextureSwizzle r, NVNtextureSwizzle g, NVNtextureSwizzle b,
                              NVNtextureSwizzle a) noexcept {
    if (!view) return;
    static std::atomic_flag reported_swizzle;
    if (static_cast<unsigned>(r) > 5 || static_cast<unsigned>(g) > 5 || static_cast<unsigned>(b) > 5 ||
        static_cast<unsigned>(a) > 5) {
        report_once(reported_swizzle, "invalid texture view swizzle ignored");
        return;
    }
    auto* state = state_of<TextureViewState>(view);
    state->sets_swizzle = 1;
    state->swizzle[0] = static_cast<std::uint8_t>(r);
    state->swizzle[1] = static_cast<std::uint8_t>(g);
    state->swizzle[2] = static_cast<std::uint8_t>(b);
    state->swizzle[3] = static_cast<std::uint8_t>(a);
}

void texture_view_set_depth_stencil_mode(NVNtextureView* view, NVNtextureDepthStencilMode mode) noexcept {
    if (!view) return;
    auto* state = state_of<TextureViewState>(view);
    state->sets_depth_stencil = 1;
    state->depth_stencil_mode = static_cast<std::uint8_t>(mode) & 3;
}

void texture_view_set_target(NVNtextureView* view, NVNtextureTarget target) noexcept {
    if (!view) return;
    auto* state = state_of<TextureViewState>(view);
    state->sets_target = 1;
    state->target = static_cast<std::uint8_t>(target);
}

NVNboolean texture_view_get_levels(const NVNtextureView* view, int* base, int* count) noexcept {
    if (!view) return 0;
    const auto* state = state_of<TextureViewState>(view);
    if (base) *base = state->base_level;
    if (count) *count = state->level_count;
    return state->sets_levels;
}

NVNboolean texture_view_get_layers(const NVNtextureView* view, int* minimum, int* count) noexcept {
    if (!view) return 0;
    const auto* state = state_of<TextureViewState>(view);
    if (minimum) *minimum = state->min_layer;
    if (count) *count = state->layer_count;
    return state->sets_layers;
}

NVNboolean texture_view_get_format(const NVNtextureView* view, NVNformat* format) noexcept {
    if (!view) return 0;
    const auto* state = state_of<TextureViewState>(view);
    if (format) *format = static_cast<NVNformat>(state->format);
    return state->sets_format;
}

NVNboolean texture_view_get_swizzle(const NVNtextureView* view, NVNtextureSwizzle* r, NVNtextureSwizzle* g,
                                    NVNtextureSwizzle* b, NVNtextureSwizzle* a) noexcept {
    if (!view) return 0;
    const auto* state = state_of<TextureViewState>(view);
    if (r) *r = static_cast<NVNtextureSwizzle>(state->swizzle[0]);
    if (g) *g = static_cast<NVNtextureSwizzle>(state->swizzle[1]);
    if (b) *b = static_cast<NVNtextureSwizzle>(state->swizzle[2]);
    if (a) *a = static_cast<NVNtextureSwizzle>(state->swizzle[3]);
    return state->sets_swizzle;
}

NVNboolean texture_view_get_depth_stencil_mode(const NVNtextureView* view, NVNtextureDepthStencilMode* mode) noexcept {
    if (!view) return 0;
    const auto* state = state_of<TextureViewState>(view);
    if (mode) *mode = static_cast<NVNtextureDepthStencilMode>(state->depth_stencil_mode);
    return state->sets_depth_stencil;
}

NVNboolean texture_view_get_target(const NVNtextureView* view, NVNtextureTarget* target) noexcept {
    if (!view) return 0;
    const auto* state = state_of<TextureViewState>(view);
    if (target) *target = static_cast<NVNtextureTarget>(state->target);
    return state->sets_target;
}

NVNboolean texture_view_compare(const NVNtextureView* left, const NVNtextureView* right) noexcept {
    if (!left || !right) return left == right ? 1 : 0;
    return std::memcmp(state_of<TextureViewState>(left), state_of<TextureViewState>(right), sizeof(TextureViewState)) == 0
               ? 1
               : 0;
}

// ---------------------------------------------------------------- samplers

void sampler_builder_set_defaults(NVNsamplerBuilder* builder) noexcept {
    if (!builder) return;
    auto* state = state_of<SamplerBuilderState>(builder);
    NVNdevice* device = state->device;
    *state = SamplerBuilderState{};
    state->device = device;
    SamplerParameters& parameters = state->parameters;
    parameters.min_filter = NVN_MIN_FILTER_LINEAR;
    parameters.mag_filter = NVN_MAG_FILTER_LINEAR;
    parameters.wrap_s = parameters.wrap_t = parameters.wrap_r = NVN_WRAP_MODE_REPEAT;
    parameters.compare_function = NVN_COMPARE_FUNC_LESS;
    parameters.min_lod = 0.0f;
    parameters.max_lod = 1000.0f;
    parameters.max_anisotropy = 1.0f;
}

void sampler_builder_set_device(NVNsamplerBuilder* builder, NVNdevice* device) noexcept {
    if (builder) state_of<SamplerBuilderState>(builder)->device = device;
}

void sampler_builder_set_min_mag_filter(NVNsamplerBuilder* builder, NVNminFilter min, NVNmagFilter mag) noexcept {
    if (!builder) return;
    SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    parameters.min_filter = static_cast<std::uint8_t>(min);
    parameters.mag_filter = static_cast<std::uint8_t>(mag);
}

void sampler_builder_set_wrap_mode(NVNsamplerBuilder* builder, NVNwrapMode s, NVNwrapMode t, NVNwrapMode r) noexcept {
    if (!builder) return;
    SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    parameters.wrap_s = static_cast<std::uint8_t>(s);
    parameters.wrap_t = static_cast<std::uint8_t>(t);
    parameters.wrap_r = static_cast<std::uint8_t>(r);
}

void sampler_builder_set_lod_clamp(NVNsamplerBuilder* builder, float minimum, float maximum) noexcept {
    if (!builder) return;
    SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    parameters.min_lod = minimum;
    parameters.max_lod = maximum;
}

void sampler_builder_set_lod_bias(NVNsamplerBuilder* builder, float bias) noexcept {
    if (builder) state_of<SamplerBuilderState>(builder)->parameters.lod_bias = bias;
}

void sampler_builder_set_compare(NVNsamplerBuilder* builder, NVNcompareMode mode, NVNcompareFunc function) noexcept {
    if (!builder) return;
    SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    parameters.compare_mode = static_cast<std::uint8_t>(mode);
    parameters.compare_function = static_cast<std::uint8_t>(function);
}

void sampler_builder_set_border_color(NVNsamplerBuilder* builder, const float* color) noexcept {
    if (builder && color) std::memcpy(state_of<SamplerBuilderState>(builder)->parameters.border, color, 16);
}

void sampler_builder_set_border_colori(NVNsamplerBuilder* builder, const int* color) noexcept {
    sampler_builder_set_border_color(builder, reinterpret_cast<const float*>(color));
}

void sampler_builder_set_border_colorui(NVNsamplerBuilder* builder, const std::uint32_t* color) noexcept {
    sampler_builder_set_border_color(builder, reinterpret_cast<const float*>(color));
}

void sampler_builder_set_max_anisotropy(NVNsamplerBuilder* builder, float anisotropy) noexcept {
    if (builder) state_of<SamplerBuilderState>(builder)->parameters.max_anisotropy = anisotropy;
}

// Reduction filters and LOD snap have no portable effect here; both are
// accepted and forgotten.
void sampler_builder_set_reduction_filter(NVNsamplerBuilder*, NVNsamplerReduction) noexcept {}
void sampler_builder_set_lod_snap(NVNsamplerBuilder*, float) noexcept {}
NVNsamplerReduction sampler_builder_get_reduction_filter(const NVNsamplerBuilder*) noexcept {
    return NVN_SAMPLER_REDUCTION_AVERAGE;
}
float sampler_builder_get_lod_snap(const NVNsamplerBuilder*) noexcept { return 0.0f; }

void sampler_builder_get_min_mag_filter(const NVNsamplerBuilder* builder, NVNminFilter* min, NVNmagFilter* mag) noexcept {
    if (!builder) return;
    const SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    if (min) *min = static_cast<NVNminFilter>(parameters.min_filter);
    if (mag) *mag = static_cast<NVNmagFilter>(parameters.mag_filter);
}

void sampler_builder_get_wrap_mode(const NVNsamplerBuilder* builder, NVNwrapMode* s, NVNwrapMode* t,
                                   NVNwrapMode* r) noexcept {
    if (!builder) return;
    const SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    if (s) *s = static_cast<NVNwrapMode>(parameters.wrap_s);
    if (t) *t = static_cast<NVNwrapMode>(parameters.wrap_t);
    if (r) *r = static_cast<NVNwrapMode>(parameters.wrap_r);
}

void sampler_builder_get_lod_clamp(const NVNsamplerBuilder* builder, float* minimum, float* maximum) noexcept {
    if (!builder) return;
    const SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    if (minimum) *minimum = parameters.min_lod;
    if (maximum) *maximum = parameters.max_lod;
}

float sampler_builder_get_lod_bias(const NVNsamplerBuilder* builder) noexcept {
    return builder ? state_of<SamplerBuilderState>(builder)->parameters.lod_bias : 0.0f;
}

void sampler_builder_get_compare(const NVNsamplerBuilder* builder, NVNcompareMode* mode,
                                 NVNcompareFunc* function) noexcept {
    if (!builder) return;
    const SamplerParameters& parameters = state_of<SamplerBuilderState>(builder)->parameters;
    if (mode) *mode = static_cast<NVNcompareMode>(parameters.compare_mode);
    if (function) *function = static_cast<NVNcompareFunc>(parameters.compare_function);
}

void sampler_builder_get_border_color(const NVNsamplerBuilder* builder, float* color) noexcept {
    if (builder && color) std::memcpy(color, state_of<SamplerBuilderState>(builder)->parameters.border, 16);
}

void sampler_builder_get_border_colori(const NVNsamplerBuilder* builder, int* color) noexcept {
    sampler_builder_get_border_color(builder, reinterpret_cast<float*>(color));
}

void sampler_builder_get_border_colorui(const NVNsamplerBuilder* builder, std::uint32_t* color) noexcept {
    sampler_builder_get_border_color(builder, reinterpret_cast<float*>(color));
}

float sampler_builder_get_max_anisotropy(const NVNsamplerBuilder* builder) noexcept {
    return builder ? state_of<SamplerBuilderState>(builder)->parameters.max_anisotropy : 0.0f;
}

NVNboolean sampler_initialize(NVNsampler* sampler, const NVNsamplerBuilder* builder) noexcept {
    if (!sampler || !builder) return 0;
    auto* state = state_of<SamplerState>(sampler);
    const auto* source = state_of<SamplerBuilderState>(builder);
    *state = SamplerState{};
    state->tag = tag_sampler;
    state->parameters = source->parameters;
    state->device = source->device;
    return 1;
}

void sampler_finalize(NVNsampler* sampler) noexcept {
    if (sampler) state_of<SamplerState>(sampler)->tag = 0;
}

void sampler_get_min_mag_filter(const NVNsampler* sampler, NVNminFilter* min, NVNmagFilter* mag) noexcept {
    if (!sampler) return;
    const SamplerParameters& parameters = state_of<SamplerState>(sampler)->parameters;
    if (min) *min = static_cast<NVNminFilter>(parameters.min_filter);
    if (mag) *mag = static_cast<NVNmagFilter>(parameters.mag_filter);
}

void sampler_get_wrap_mode(const NVNsampler* sampler, NVNwrapMode* s, NVNwrapMode* t, NVNwrapMode* r) noexcept {
    if (!sampler) return;
    const SamplerParameters& parameters = state_of<SamplerState>(sampler)->parameters;
    if (s) *s = static_cast<NVNwrapMode>(parameters.wrap_s);
    if (t) *t = static_cast<NVNwrapMode>(parameters.wrap_t);
    if (r) *r = static_cast<NVNwrapMode>(parameters.wrap_r);
}

void sampler_get_lod_clamp(const NVNsampler* sampler, float* minimum, float* maximum) noexcept {
    if (!sampler) return;
    const SamplerParameters& parameters = state_of<SamplerState>(sampler)->parameters;
    if (minimum) *minimum = parameters.min_lod;
    if (maximum) *maximum = parameters.max_lod;
}

float sampler_get_lod_bias(const NVNsampler* sampler) noexcept {
    return sampler ? state_of<SamplerState>(sampler)->parameters.lod_bias : 0.0f;
}

void sampler_get_compare(const NVNsampler* sampler, NVNcompareMode* mode, NVNcompareFunc* function) noexcept {
    if (!sampler) return;
    const SamplerParameters& parameters = state_of<SamplerState>(sampler)->parameters;
    if (mode) *mode = static_cast<NVNcompareMode>(parameters.compare_mode);
    if (function) *function = static_cast<NVNcompareFunc>(parameters.compare_function);
}

void sampler_get_border_color(const NVNsampler* sampler, float* color) noexcept {
    if (sampler && color) std::memcpy(color, state_of<SamplerState>(sampler)->parameters.border, 16);
}

float sampler_get_max_anisotropy(const NVNsampler* sampler) noexcept {
    return sampler ? state_of<SamplerState>(sampler)->parameters.max_anisotropy : 0.0f;
}

NVNsamplerReduction sampler_get_reduction_filter(const NVNsampler*) noexcept { return NVN_SAMPLER_REDUCTION_AVERAGE; }

bool same_parameters(const SamplerParameters& a, const SamplerParameters& b) noexcept {
    return a.min_filter == b.min_filter && a.mag_filter == b.mag_filter && a.wrap_s == b.wrap_s &&
           a.wrap_t == b.wrap_t && a.wrap_r == b.wrap_r && a.compare_mode == b.compare_mode &&
           a.compare_function == b.compare_function && a.min_lod == b.min_lod && a.max_lod == b.max_lod &&
           a.lod_bias == b.lod_bias && a.max_anisotropy == b.max_anisotropy &&
           std::memcmp(a.border, b.border, sizeof a.border) == 0;
}

NVNboolean sampler_compare(const NVNsampler* left, const NVNsampler* right) noexcept {
    if (!left || !right) return left == right ? 1 : 0;
    const auto* a = state_of<SamplerState>(left);
    const auto* b = state_of<SamplerState>(right);
    return a->device == b->device && same_parameters(a->parameters, b->parameters) ? 1 : 0;
}

// ---------------------------------------------------------------- programs

NVNboolean program_initialize(NVNprogram* program, NVNdevice* device) noexcept {
    if (!program) return 0;
    auto* state = state_of<ProgramState>(program);
    *state = ProgramState{};
    state->tag = tag_program;
    state->device = device;
    return 1;
}

void program_finalize(NVNprogram* program) noexcept {
    if (!program) return;
    auto* state = state_of<ProgramState>(program);
    state->tag = 0;
    state->serial = 0;
}

NVNboolean program_set_shaders(NVNprogram* program, int count, const NVNshaderData* stages) noexcept {
    if (!program) return 0;
    static std::atomic_flag reported_range;
    auto* state = state_of<ProgramState>(program);
    if (count < 0 || count > 6 || (count > 0 && !stages)) {
        report_once(reported_range, "program shader stage count or range invalid; program unchanged");
        return 0;
    }
    // The null device accepts the stage descriptors as given; translation
    // finds them when the shader package stage arrives.
    for (int i = 0; i < count; ++i) state->stages[i] = stages[i];
    state->stage_count = static_cast<std::uint32_t>(count);
    state->serial = new_serial();
    state->snapshot = nullptr;
    return 1;
}

NVNboolean program_set_shaders_ext(NVNprogram* program, int count, const NVNshaderDataExt* stages) noexcept {
    if (!program) return 0;
    static std::atomic_flag reported_range;
    auto* state = state_of<ProgramState>(program);
    if (count < 0 || count > 6 || (count > 0 && !stages)) {
        report_once(reported_range, "program shader stage count or range invalid; program unchanged");
        return 0;
    }
    for (int i = 0; i < count; ++i) {
        state->stages[i].data = stages[i].data;
        state->stages[i].control = stages[i].control;
    }
    state->stage_count = static_cast<std::uint32_t>(count);
    state->serial = new_serial();
    state->snapshot = nullptr;
    return 1;
}

// ---------------------------------------------------------------- queues

void queue_builder_set_defaults(NVNqueueBuilder* builder) noexcept {
    if (!builder) return;
    auto* state = state_of<QueueBuilderState>(builder);
    NVNdevice* device = state->device;
    *state = QueueBuilderState{};
    state->device = device;
    state->command_memory_size = 65536;
    state->compute_memory_size = 262144;
    state->control_memory_size = 16384;
    state->flush_threshold = 4096;
}

void queue_builder_set_device(NVNqueueBuilder* builder, NVNdevice* device) noexcept {
    if (builder) state_of<QueueBuilderState>(builder)->device = device;
}

void queue_builder_set_flags(NVNqueueBuilder* builder, int flags) noexcept {
    if (builder) state_of<QueueBuilderState>(builder)->flags = static_cast<std::uint32_t>(flags);
}

void queue_builder_set_command_memory_size(NVNqueueBuilder* builder, std::size_t size) noexcept {
    if (builder) state_of<QueueBuilderState>(builder)->command_memory_size = size;
}

void queue_builder_set_compute_memory_size(NVNqueueBuilder* builder, std::size_t size) noexcept {
    if (builder) state_of<QueueBuilderState>(builder)->compute_memory_size = size;
}

void queue_builder_set_control_memory_size(NVNqueueBuilder* builder, std::size_t size) noexcept {
    if (builder) state_of<QueueBuilderState>(builder)->control_memory_size = size;
}

void queue_builder_set_command_flush_threshold(NVNqueueBuilder* builder, std::size_t threshold) noexcept {
    if (builder) state_of<QueueBuilderState>(builder)->flush_threshold = threshold;
}

std::size_t queue_builder_get_queue_memory_size(const NVNqueueBuilder* builder) noexcept {
    if (!builder) return 0;
    const auto* state = state_of<QueueBuilderState>(builder);
    // The queue memory block holds the command, compute and control memory,
    // page aligned.
    return (state->command_memory_size + state->compute_memory_size + state->control_memory_size + 4095) &
           ~static_cast<std::size_t>(4095);
}

void queue_builder_set_queue_memory(NVNqueueBuilder* builder, void* memory, std::size_t size) noexcept {
    if (!builder) return;
    auto* state = state_of<QueueBuilderState>(builder);
    state->memory = memory;
    state->memory_size = size;
}

NVNboolean queue_initialize(NVNqueue* queue, const NVNqueueBuilder* builder) noexcept {
    if (!queue) return 0;
    auto* state = state_of<QueueState>(queue);
    *state = QueueState{};
    state->tag = tag_queue;
    state->device = builder ? state_of<QueueBuilderState>(builder)->device : nullptr;
    return 1;
}

void queue_finalize(NVNqueue* queue) noexcept {
    if (queue) state_of<QueueState>(queue)->tag = 0;
}

NVNqueueGetErrorResult queue_get_error(NVNqueue*, NVNqueueErrorInfo* info) noexcept {
    if (info) std::memset(info, 0, sizeof *info);
    return NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR;
}

std::size_t queue_get_total_command_memory_used(NVNqueue*) noexcept { return 0; }
std::size_t queue_get_total_control_memory_used(NVNqueue*) noexcept { return 0; }
std::size_t queue_get_total_compute_memory_used(NVNqueue*) noexcept { return 0; }
void queue_reset_memory_usage_counts(NVNqueue*) noexcept {}

void queue_submit_commands(NVNqueue* queue, int count, const NVNcommandHandle*) noexcept {
    static std::atomic_flag reported_negative;
    if (count < 0) {
        report_once(reported_negative, "negative command count submitted; ignored");
        return;
    }
    if (queue) {
        auto* state = state_of<QueueState>(queue);
        if (state->tag == tag_queue) ++state->submits;
    }
    if (count > 0) fence_counter.fetch_add(1, std::memory_order_release);
}

void queue_flush(NVNqueue*) noexcept {}

void queue_finish(NVNqueue*) noexcept {
    // Nothing is pending: every issued fence was complete on submission.
}

void queue_fence_sync(NVNqueue*, NVNsync* sync, NVNsyncCondition, int) noexcept {
    if (!sync) return;
    auto* state = state_of<SyncState>(sync);
    state->fenced = 1;
    state->fence = latest_fence();
}

NVNboolean queue_wait_sync(NVNqueue*, const NVNsync* sync) noexcept {
    if (!sync) return 0;
    return state_of<SyncState>(sync)->fenced ? 1 : 0;
}

void queue_present_texture(NVNqueue*, NVNwindow* window, int index) noexcept {
    if (!window) return;
    auto* state = state_of<WindowState>(window);
    if (state->tag != tag_window) return;
    ++state->presents;
    if (index >= 0 && index < state->texture_count && index < 4)
        state->present_ticket[index] = new_serial();  // the null compositor consumes nothing
}

NVNqueueAcquireTextureResult queue_acquire_texture(NVNqueue*, NVNwindow* window, int* index) noexcept {
    if (!window) return NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
    auto* state = state_of<WindowState>(window);
    const int next = state->texture_count > 0 ? static_cast<int>(state->next++ % unsigned(state->texture_count)) : 0;
    if (index) *index = next;
    return NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_SUCCESS;
}

// ---------------------------------------------------------------- windows

void window_builder_set_defaults(NVNwindowBuilder* builder) noexcept {
    if (!builder) return;
    auto* state = state_of<WindowBuilderState>(builder);
    NVNdevice* device = state->device;
    *state = WindowBuilderState{};
    state->device = device;
    state->present_interval = 1;
}

void window_builder_set_device(NVNwindowBuilder* builder, NVNdevice* device) noexcept {
    if (builder) state_of<WindowBuilderState>(builder)->device = device;
}

void window_builder_set_native_window(NVNwindowBuilder* builder, NVNnativeWindow window) noexcept {
    if (builder) state_of<WindowBuilderState>(builder)->native_window = window;
}

void window_builder_set_textures(NVNwindowBuilder* builder, int count, NVNtexture* const* textures) noexcept {
    if (!builder) return;
    static std::atomic_flag reported_count;
    if (count < 0 || count > 4) {
        report_once(reported_count, "window texture count outside zero to four; clamped");
        count = count < 0 ? 0 : 4;
    }
    auto* state = state_of<WindowBuilderState>(builder);
    state->texture_count = count;
    for (int i = 0; i < count; ++i) state->textures[i] = textures ? textures[i] : nullptr;
}

void window_builder_set_present_interval(NVNwindowBuilder* builder, int interval) noexcept {
    if (builder) state_of<WindowBuilderState>(builder)->present_interval = interval;
}

NVNnativeWindow window_builder_get_native_window(const NVNwindowBuilder* builder) noexcept {
    return builder ? state_of<WindowBuilderState>(builder)->native_window : nullptr;
}

int window_builder_get_present_interval(const NVNwindowBuilder* builder) noexcept {
    return builder ? state_of<WindowBuilderState>(builder)->present_interval : 1;
}

NVNboolean window_initialize(NVNwindow* window, const NVNwindowBuilder* builder) noexcept {
    if (!window || !builder) return 0;
    const auto* source = state_of<WindowBuilderState>(builder);
    auto* state = state_of<WindowState>(window);
    *state = WindowState{};
    state->tag = tag_window;
    state->native_window = source->native_window;
    state->texture_count = source->texture_count;
    state->present_interval = source->present_interval;
    for (int i = 0; i < 4; ++i) state->textures[i] = source->textures[i];
    return 1;
}

void window_finalize(NVNwindow* window) noexcept {
    if (window) state_of<WindowState>(window)->tag = 0;
}

int window_get_present_interval(const NVNwindow* window) noexcept {
    return window ? state_of<WindowState>(window)->present_interval : 0;
}

void window_set_present_interval(NVNwindow* window, int interval) noexcept {
    if (window) state_of<WindowState>(window)->present_interval = interval;
}

void window_set_crop(NVNwindow* window, int x, int y, int width, int height) noexcept {
    if (!window) return;
    auto* state = state_of<WindowState>(window);
    state->crop[0] = x;
    state->crop[1] = y;
    state->crop[2] = width;
    state->crop[3] = height;
}

void window_get_crop(const NVNwindow* window, NVNrectangle* crop) noexcept {
    if (!crop) return;
    const auto* state = window ? state_of<WindowState>(window) : nullptr;
    crop->x = state ? state->crop[0] : 0;
    crop->y = state ? state->crop[1] : 0;
    crop->width = state ? state->crop[2] : 0;
    crop->height = state ? state->crop[3] : 0;
}

NVNnativeWindow window_get_native_window(const NVNwindow* window) noexcept {
    return window ? state_of<WindowState>(window)->native_window : nullptr;
}

NVNwindowAcquireTextureResult window_acquire_texture(NVNwindow* window, NVNsync* sync, int* index) noexcept {
    if (!window) return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
    auto* state = state_of<WindowState>(window);
    // Acquire hands out the swapchain textures in turn; with no compositor
    // every image is available immediately, and the sync fences the (already
    // complete) submissions made so far.
    const int next = state->texture_count > 0 ? static_cast<int>(state->next++ % unsigned(state->texture_count)) : 0;
    if (index) *index = next;
    if (sync) queue_fence_sync(nullptr, sync, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
    return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS;
}

// ---------------------------------------------------------------- sync

NVNboolean sync_initialize(NVNsync* sync, NVNdevice*) noexcept {
    if (!sync) return 0;
    *state_of<SyncState>(sync) = SyncState{tag_sync, 0, 0};
    return 1;
}

void sync_finalize(NVNsync* sync) noexcept {
    if (sync) state_of<SyncState>(sync)->tag = 0;
}

NVNsyncWaitResult sync_wait(const NVNsync* sync, std::uint64_t) noexcept {
    if (!sync) return NVN_SYNC_WAIT_RESULT_FAILED;
    const auto* state = state_of<SyncState>(sync);
    if (!state->fenced) return NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;  // never fenced: never signaled
    return NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED;  // every fence completed on submission
}

// ---------------------------------------------------------------- command buffers

struct NullCommandBufferState {
    CommandBufferState common;
    std::size_t command_capacity = 0;
    std::size_t control_capacity = 0;
};
static_assert(sizeof(NullCommandBufferState) <= sizeof(NVNcommandBuffer));

NVNboolean command_buffer_initialize(NVNcommandBuffer* commands, NVNdevice* device) noexcept {
    if (!commands) return 0;
    auto* storage = state_of<NullCommandBufferState>(commands);
    *storage = NullCommandBufferState{};
    auto* state = &storage->common;
    state->tag = tag_command_buffer;
    state->device = device;
    return 1;
}

void command_buffer_finalize(NVNcommandBuffer* commands) noexcept {
    if (commands) state_of<CommandBufferState>(commands)->tag = 0;
}

void command_buffer_set_memory_callback(NVNcommandBuffer* commands, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC callback) noexcept {
    if (commands) state_of<CommandBufferState>(commands)->memory_callback = callback;
}

void command_buffer_set_memory_callback_data(NVNcommandBuffer* commands, void* data) noexcept {
    if (commands) state_of<CommandBufferState>(commands)->callback_data = data;
}

PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC command_buffer_get_memory_callback(const NVNcommandBuffer* commands) noexcept {
    return commands ? state_of<CommandBufferState>(commands)->memory_callback : nullptr;
}

// Donated memory is accepted and left untouched: the null device records
// nothing, so no segment is ever exhausted.
void command_buffer_add_command_memory(NVNcommandBuffer* commands, const NVNmemoryPool*, ptrdiff_t,
                                       std::size_t size) noexcept {
    if (commands) state_of<NullCommandBufferState>(commands)->command_capacity = size;
}
void command_buffer_add_control_memory(NVNcommandBuffer* commands, void*, std::size_t size) noexcept {
    if (commands) state_of<NullCommandBufferState>(commands)->control_capacity = size;
}

std::size_t command_buffer_get_command_memory_size(const NVNcommandBuffer* commands) noexcept {
    return commands ? state_of<NullCommandBufferState>(commands)->command_capacity : 0;
}
std::size_t command_buffer_get_command_memory_used(const NVNcommandBuffer*) noexcept { return 0; }
std::size_t command_buffer_get_command_memory_free(const NVNcommandBuffer* commands) noexcept {
    return command_buffer_get_command_memory_size(commands);
}
std::size_t command_buffer_get_control_memory_size(const NVNcommandBuffer* commands) noexcept {
    return commands ? state_of<NullCommandBufferState>(commands)->control_capacity : 0;
}
std::size_t command_buffer_get_control_memory_used(const NVNcommandBuffer*) noexcept { return 0; }
std::size_t command_buffer_get_control_memory_free(const NVNcommandBuffer* commands) noexcept {
    return command_buffer_get_control_memory_size(commands);
}

void command_buffer_begin_recording(NVNcommandBuffer* commands) noexcept {
    if (commands) state_of<CommandBufferState>(commands)->recording = 1;
}

NVNcommandHandle command_buffer_end_recording(NVNcommandBuffer* commands) noexcept {
    if (!commands) return 0;
    state_of<CommandBufferState>(commands)->recording = 0;
    return new_serial();  // an opaque identity; the null queue retains no resources
}

NVNboolean command_buffer_is_recording(const NVNcommandBuffer* commands) noexcept {
    return commands && state_of<CommandBufferState>(commands)->recording ? 1 : 0;
}

// Recording a fence takes effect at once: an empty command stream is complete
// the moment it would be submitted.
void command_buffer_fence_sync(NVNcommandBuffer*, NVNsync* sync, NVNsyncCondition, int) noexcept {
    queue_fence_sync(nullptr, sync, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
}

ptrdiff_t command_buffer_get_copy_row_stride(const NVNcommandBuffer*) noexcept { return 0; }
ptrdiff_t command_buffer_get_copy_image_stride(const NVNcommandBuffer*) noexcept { return 0; }

// Recording commands that change GPU state or issue GPU work: accepted and
// dropped, the null device executes nothing.
template <typename... Params>
void record_nothing(Params...) noexcept {}

} // namespace

// Installs the stage-A null device entry points into the entry table. Call
// once during driver setup, before the game resolves any entry point; the
// functions are plain host entry points and may also be resolved through
// nvnDeviceGetProcAddress afterwards.
void register_null_entries() noexcept {
    // Device.
    install<Entry::nvnDeviceBuilderSetDefaults>(&device_builder_set_defaults);
    install<Entry::nvnDeviceBuilderSetFlags>(&device_builder_set_flags);
    install<Entry::nvnDeviceInitialize>(&device_initialize);
    install<Entry::nvnDeviceFinalize>(&device_finalize);
    install<Entry::nvnDeviceGetProcAddress>(&device_get_proc_address);
    install<Entry::nvnDeviceGetInteger>(&device_get_integer);
    install<Entry::nvnDeviceGetTextureHandle>(&device_get_texture_handle);
    install<Entry::nvnDeviceGetTexelFetchHandle>(&device_get_texel_fetch_handle);
    install<Entry::nvnDeviceGetImageHandle>(&device_get_image_handle);
    install<Entry::nvnDeviceGetSeparateTextureHandle>(&device_get_separate_texture_handle);
    install<Entry::nvnDeviceGetSeparateSamplerHandle>(&device_get_separate_sampler_handle);
    install<Entry::nvnDeviceInstallDebugCallback>(&device_install_debug_callback);
    install<Entry::nvnDeviceSetWindowOriginMode>(&device_set_window_origin_mode);
    install<Entry::nvnDeviceGetWindowOriginMode>(&device_get_window_origin_mode);
    install<Entry::nvnDeviceSetDepthMode>(&device_set_depth_mode);
    install<Entry::nvnDeviceGetDepthMode>(&device_get_depth_mode);
    install<Entry::nvnDeviceRegisterFastClearColor>(&device_decline_fast_clear);
    install<Entry::nvnDeviceRegisterFastClearColori>(&device_decline_fast_cleari);
    install<Entry::nvnDeviceRegisterFastClearColorui>(&device_decline_fast_clearui);
    install<Entry::nvnDeviceRegisterFastClearDepth>(&device_decline_fast_clear_depth);
    install<Entry::nvnDeviceFinalizeCommandHandle>(&accept_nothing<NVNdevice*, NVNcommandHandle>);

    // Debug labels carry no state in any stage.
    install<Entry::nvnDeviceSetDebugLabel>(&accept_nothing<NVNdevice*, const char*>);
    install<Entry::nvnQueueSetDebugLabel>(&accept_nothing<NVNqueue*, const char*>);
    install<Entry::nvnWindowSetDebugLabel>(&accept_nothing<NVNwindow*, const char*>);
    install<Entry::nvnProgramSetDebugLabel>(&accept_nothing<NVNprogram*, const char*>);
    install<Entry::nvnMemoryPoolSetDebugLabel>(&accept_nothing<NVNmemoryPool*, const char*>);
    install<Entry::nvnTexturePoolSetDebugLabel>(&accept_nothing<NVNtexturePool*, const char*>);
    install<Entry::nvnSamplerPoolSetDebugLabel>(&accept_nothing<NVNsamplerPool*, const char*>);
    install<Entry::nvnTextureSetDebugLabel>(&accept_nothing<NVNtexture*, const char*>);
    install<Entry::nvnSamplerSetDebugLabel>(&accept_nothing<NVNsampler*, const char*>);
    install<Entry::nvnBufferSetDebugLabel>(&accept_nothing<NVNbuffer*, const char*>);
    install<Entry::nvnCommandBufferSetDebugLabel>(&accept_nothing<NVNcommandBuffer*, const char*>);
    install<Entry::nvnSyncSetDebugLabel>(&accept_nothing<NVNsync*, const char*>);

    // Memory pools.
    install<Entry::nvnMemoryPoolBuilderSetDefaults>(&memory_pool_builder_set_defaults);
    install<Entry::nvnMemoryPoolBuilderSetDevice>(&memory_pool_builder_set_device);
    install<Entry::nvnMemoryPoolBuilderSetStorage>(&memory_pool_builder_set_storage);
    install<Entry::nvnMemoryPoolBuilderSetFlags>(&memory_pool_builder_set_flags);
    install<Entry::nvnMemoryPoolBuilderGetSize>(&memory_pool_builder_get_size);
    install<Entry::nvnMemoryPoolBuilderGetFlags>(&memory_pool_builder_get_flags);
    install<Entry::nvnMemoryPoolInitialize>(&memory_pool_initialize);
    install<Entry::nvnMemoryPoolFinalize>(&memory_pool_finalize);
    install<Entry::nvnMemoryPoolMap>(&memory_pool_map);
    install<Entry::nvnMemoryPoolFlushMappedRange>(&memory_pool_flush_mapped_range);
    install<Entry::nvnMemoryPoolInvalidateMappedRange>(&memory_pool_invalidate_mapped_range);
    install<Entry::nvnMemoryPoolGetBufferAddress>(&memory_pool_get_buffer_address);
    install<Entry::nvnMemoryPoolGetSize>(&memory_pool_get_size);
    install<Entry::nvnMemoryPoolGetFlags>(&memory_pool_get_flags);

    // Descriptor pools.
    install<Entry::nvnTexturePoolInitialize>(&texture_pool_initialize);
    install<Entry::nvnTexturePoolFinalize>(&texture_pool_finalize);
    install<Entry::nvnTexturePoolRegisterTexture>(&texture_pool_register_texture);
    install<Entry::nvnTexturePoolRegisterImage>(&texture_pool_register_image);
    install<Entry::nvnTexturePoolGetMemoryPool>(&texture_pool_get_memory_pool);
    install<Entry::nvnTexturePoolGetMemoryOffset>(&texture_pool_get_memory_offset);
    install<Entry::nvnTexturePoolGetSize>(&texture_pool_get_size);
    install<Entry::nvnSamplerPoolInitialize>(&sampler_pool_initialize);
    install<Entry::nvnSamplerPoolFinalize>(&sampler_pool_finalize);
    install<Entry::nvnSamplerPoolRegisterSampler>(&sampler_pool_register_sampler);
    install<Entry::nvnSamplerPoolRegisterSamplerBuilder>(&sampler_pool_register_sampler_builder);
    install<Entry::nvnSamplerPoolGetMemoryPool>(&sampler_pool_get_memory_pool);
    install<Entry::nvnSamplerPoolGetMemoryOffset>(&sampler_pool_get_memory_offset);
    install<Entry::nvnSamplerPoolGetSize>(&sampler_pool_get_size);

    // Buffers.
    install<Entry::nvnBufferBuilderSetDevice>(&buffer_builder_set_device);
    install<Entry::nvnBufferBuilderSetDefaults>(&buffer_builder_set_defaults);
    install<Entry::nvnBufferBuilderSetStorage>(&buffer_builder_set_storage);
    install<Entry::nvnBufferBuilderGetDevice>(&buffer_builder_get_device);
    install<Entry::nvnBufferBuilderGetMemoryPool>(&buffer_builder_get_memory_pool);
    install<Entry::nvnBufferBuilderGetMemoryOffset>(&buffer_builder_get_memory_offset);
    install<Entry::nvnBufferBuilderGetSize>(&buffer_builder_get_size);
    install<Entry::nvnBufferInitialize>(&buffer_initialize);
    install<Entry::nvnBufferFinalize>(&buffer_finalize);
    install<Entry::nvnBufferMap>(&buffer_map);
    install<Entry::nvnBufferGetAddress>(&buffer_get_address);
    install<Entry::nvnBufferFlushMappedRange>(&buffer_flush_mapped_range);
    install<Entry::nvnBufferInvalidateMappedRange>(&buffer_invalidate_mapped_range);
    install<Entry::nvnBufferGetMemoryPool>(&buffer_get_memory_pool);
    install<Entry::nvnBufferGetMemoryOffset>(&buffer_get_memory_offset);
    install<Entry::nvnBufferGetSize>(&buffer_get_size);
    install<Entry::nvnBufferGetDebugID>(&buffer_get_debug_id);

    // Texture builder.
    install<Entry::nvnTextureBuilderSetDevice>(&texture_builder_set_device);
    install<Entry::nvnTextureBuilderSetDefaults>(&texture_builder_set_defaults);
    install<Entry::nvnTextureBuilderSetFlags>(&texture_builder_set_flags);
    install<Entry::nvnTextureBuilderSetTarget>(&texture_builder_set_target);
    install<Entry::nvnTextureBuilderSetWidth>(&texture_builder_set_width);
    install<Entry::nvnTextureBuilderSetHeight>(&texture_builder_set_height);
    install<Entry::nvnTextureBuilderSetDepth>(&texture_builder_set_depth);
    install<Entry::nvnTextureBuilderSetSize1D>(&texture_builder_set_size_1d);
    install<Entry::nvnTextureBuilderSetSize2D>(&texture_builder_set_size_2d);
    install<Entry::nvnTextureBuilderSetSize3D>(&texture_builder_set_size_3d);
    install<Entry::nvnTextureBuilderSetLevels>(&texture_builder_set_levels);
    install<Entry::nvnTextureBuilderSetFormat>(&texture_builder_set_format);
    install<Entry::nvnTextureBuilderSetSamples>(&texture_builder_set_samples);
    install<Entry::nvnTextureBuilderSetSwizzle>(&texture_builder_set_swizzle);
    install<Entry::nvnTextureBuilderSetDepthStencilMode>(&texture_builder_set_depth_stencil_mode);
    install<Entry::nvnTextureBuilderGetStorageSize>(&texture_builder_get_storage_size);
    install<Entry::nvnTextureBuilderGetStorageAlignment>(&texture_builder_get_storage_alignment);
    install<Entry::nvnTextureBuilderSetStorage>(&texture_builder_set_storage);
    install<Entry::nvnTextureBuilderSetPackagedTextureLayout>(&texture_builder_set_packaged_texture_layout);
    install<Entry::nvnTextureBuilderSetStride>(&texture_builder_set_stride);
    install<Entry::nvnTextureBuilderGetFlags>(&texture_builder_get_flags);
    install<Entry::nvnTextureBuilderGetTarget>(&texture_builder_get_target);
    install<Entry::nvnTextureBuilderGetWidth>(&texture_builder_get_width);
    install<Entry::nvnTextureBuilderGetHeight>(&texture_builder_get_height);
    install<Entry::nvnTextureBuilderGetDepth>(&texture_builder_get_depth);
    install<Entry::nvnTextureBuilderGetLevels>(&texture_builder_get_levels);
    install<Entry::nvnTextureBuilderGetFormat>(&texture_builder_get_format);
    install<Entry::nvnTextureBuilderGetSamples>(&texture_builder_get_samples);
    install<Entry::nvnTextureBuilderGetSwizzle>(&texture_builder_get_swizzle);
    install<Entry::nvnTextureBuilderGetDepthStencilMode>(&texture_builder_get_depth_stencil_mode);
    install<Entry::nvnTextureBuilderGetStride>(&texture_builder_get_stride);
    install<Entry::nvnTextureBuilderGetZCullStorageSize>(&texture_builder_get_zcull_storage_size);
    install<Entry::nvnTextureBuilderGetMemoryOffset>(&texture_builder_get_memory_offset);

    // Texture objects.
    install<Entry::nvnTextureInitialize>(&texture_initialize);
    install<Entry::nvnTextureFinalize>(&texture_finalize);
    install<Entry::nvnTextureGetZCullStorageSize>(&texture_get_zcull_storage_size);
    install<Entry::nvnTextureGetViewOffset>(&texture_get_view_offset);
    install<Entry::nvnTextureGetFlags>(&texture_get_flags);
    install<Entry::nvnTextureGetTarget>(&texture_get_target);
    install<Entry::nvnTextureGetWidth>(&texture_get_width);
    install<Entry::nvnTextureGetHeight>(&texture_get_height);
    install<Entry::nvnTextureGetDepth>(&texture_get_depth);
    install<Entry::nvnTextureGetLevels>(&texture_get_levels);
    install<Entry::nvnTextureGetFormat>(&texture_get_format);
    install<Entry::nvnTextureGetSamples>(&texture_get_samples);
    install<Entry::nvnTextureGetSwizzle>(&texture_get_swizzle);
    install<Entry::nvnTextureGetDepthStencilMode>(&texture_get_depth_stencil_mode);
    install<Entry::nvnTextureGetStride>(&texture_get_stride);
    install<Entry::nvnTextureGetTextureAddress>(&texture_get_texture_address);
    install<Entry::nvnTextureGetMemoryOffset>(&texture_get_memory_offset);
    install<Entry::nvnTextureGetStorageSize>(&texture_get_storage_size);
    install<Entry::nvnTextureCompare>(&texture_compare);
    install<Entry::nvnTextureGetDebugID>(&texture_get_debug_id);
    install<Entry::nvnTextureFlushTexels>(&texture_flush_texels);
    install<Entry::nvnTextureInvalidateTexels>(&texture_invalidate_texels);

    // Texture views.
    install<Entry::nvnTextureViewSetDefaults>(&texture_view_set_defaults);
    install<Entry::nvnTextureViewSetLevels>(&texture_view_set_levels);
    install<Entry::nvnTextureViewSetLayers>(&texture_view_set_layers);
    install<Entry::nvnTextureViewSetFormat>(&texture_view_set_format);
    install<Entry::nvnTextureViewSetSwizzle>(&texture_view_set_swizzle);
    install<Entry::nvnTextureViewSetDepthStencilMode>(&texture_view_set_depth_stencil_mode);
    install<Entry::nvnTextureViewSetTarget>(&texture_view_set_target);
    install<Entry::nvnTextureViewGetLevels>(&texture_view_get_levels);
    install<Entry::nvnTextureViewGetLayers>(&texture_view_get_layers);
    install<Entry::nvnTextureViewGetFormat>(&texture_view_get_format);
    install<Entry::nvnTextureViewGetSwizzle>(&texture_view_get_swizzle);
    install<Entry::nvnTextureViewGetDepthStencilMode>(&texture_view_get_depth_stencil_mode);
    install<Entry::nvnTextureViewGetTarget>(&texture_view_get_target);
    install<Entry::nvnTextureViewCompare>(&texture_view_compare);

    // Samplers.
    install<Entry::nvnSamplerBuilderSetDevice>(&sampler_builder_set_device);
    install<Entry::nvnSamplerBuilderSetDefaults>(&sampler_builder_set_defaults);
    install<Entry::nvnSamplerBuilderSetMinMagFilter>(&sampler_builder_set_min_mag_filter);
    install<Entry::nvnSamplerBuilderSetWrapMode>(&sampler_builder_set_wrap_mode);
    install<Entry::nvnSamplerBuilderSetLodClamp>(&sampler_builder_set_lod_clamp);
    install<Entry::nvnSamplerBuilderSetLodBias>(&sampler_builder_set_lod_bias);
    install<Entry::nvnSamplerBuilderSetCompare>(&sampler_builder_set_compare);
    install<Entry::nvnSamplerBuilderSetBorderColor>(&sampler_builder_set_border_color);
    install<Entry::nvnSamplerBuilderSetBorderColori>(&sampler_builder_set_border_colori);
    install<Entry::nvnSamplerBuilderSetBorderColorui>(&sampler_builder_set_border_colorui);
    install<Entry::nvnSamplerBuilderSetMaxAnisotropy>(&sampler_builder_set_max_anisotropy);
    install<Entry::nvnSamplerBuilderSetReductionFilter>(&sampler_builder_set_reduction_filter);
    install<Entry::nvnSamplerBuilderSetLodSnap>(&sampler_builder_set_lod_snap);
    install<Entry::nvnSamplerBuilderGetMinMagFilter>(&sampler_builder_get_min_mag_filter);
    install<Entry::nvnSamplerBuilderGetWrapMode>(&sampler_builder_get_wrap_mode);
    install<Entry::nvnSamplerBuilderGetLodClamp>(&sampler_builder_get_lod_clamp);
    install<Entry::nvnSamplerBuilderGetLodBias>(&sampler_builder_get_lod_bias);
    install<Entry::nvnSamplerBuilderGetCompare>(&sampler_builder_get_compare);
    install<Entry::nvnSamplerBuilderGetBorderColor>(&sampler_builder_get_border_color);
    install<Entry::nvnSamplerBuilderGetBorderColori>(&sampler_builder_get_border_colori);
    install<Entry::nvnSamplerBuilderGetBorderColorui>(&sampler_builder_get_border_colorui);
    install<Entry::nvnSamplerBuilderGetMaxAnisotropy>(&sampler_builder_get_max_anisotropy);
    install<Entry::nvnSamplerBuilderGetReductionFilter>(&sampler_builder_get_reduction_filter);
    install<Entry::nvnSamplerBuilderGetLodSnap>(&sampler_builder_get_lod_snap);
    install<Entry::nvnSamplerInitialize>(&sampler_initialize);
    install<Entry::nvnSamplerFinalize>(&sampler_finalize);
    install<Entry::nvnSamplerGetMinMagFilter>(&sampler_get_min_mag_filter);
    install<Entry::nvnSamplerGetWrapMode>(&sampler_get_wrap_mode);
    install<Entry::nvnSamplerGetLodClamp>(&sampler_get_lod_clamp);
    install<Entry::nvnSamplerGetLodBias>(&sampler_get_lod_bias);
    install<Entry::nvnSamplerGetCompare>(&sampler_get_compare);
    install<Entry::nvnSamplerGetBorderColor>(&sampler_get_border_color);
    install<Entry::nvnSamplerGetMaxAnisotropy>(&sampler_get_max_anisotropy);
    install<Entry::nvnSamplerGetReductionFilter>(&sampler_get_reduction_filter);
    install<Entry::nvnSamplerCompare>(&sampler_compare);

    // Programs.
    install<Entry::nvnProgramInitialize>(&program_initialize);
    install<Entry::nvnProgramFinalize>(&program_finalize);
    install<Entry::nvnProgramSetShaders>(&program_set_shaders);
    install<Entry::nvnProgramSetShadersExt>(&program_set_shaders_ext);

    // Queues.
    install<Entry::nvnQueueBuilderSetDevice>(&queue_builder_set_device);
    install<Entry::nvnQueueBuilderSetDefaults>(&queue_builder_set_defaults);
    install<Entry::nvnQueueBuilderSetFlags>(&queue_builder_set_flags);
    install<Entry::nvnQueueBuilderSetCommandMemorySize>(&queue_builder_set_command_memory_size);
    install<Entry::nvnQueueBuilderSetComputeMemorySize>(&queue_builder_set_compute_memory_size);
    install<Entry::nvnQueueBuilderSetControlMemorySize>(&queue_builder_set_control_memory_size);
    install<Entry::nvnQueueBuilderGetQueueMemorySize>(&queue_builder_get_queue_memory_size);
    install<Entry::nvnQueueBuilderSetQueueMemory>(&queue_builder_set_queue_memory);
    install<Entry::nvnQueueBuilderSetCommandFlushThreshold>(&queue_builder_set_command_flush_threshold);
    install<Entry::nvnQueueInitialize>(&queue_initialize);
    install<Entry::nvnQueueFinalize>(&queue_finalize);
    install<Entry::nvnQueueGetError>(&queue_get_error);
    install<Entry::nvnQueueGetTotalCommandMemoryUsed>(&queue_get_total_command_memory_used);
    install<Entry::nvnQueueGetTotalControlMemoryUsed>(&queue_get_total_control_memory_used);
    install<Entry::nvnQueueGetTotalComputeMemoryUsed>(&queue_get_total_compute_memory_used);
    install<Entry::nvnQueueResetMemoryUsageCounts>(&queue_reset_memory_usage_counts);
    install<Entry::nvnQueueSubmitCommands>(&queue_submit_commands);
    install<Entry::nvnQueueFlush>(&queue_flush);
    install<Entry::nvnQueueFinish>(&queue_finish);
    install<Entry::nvnQueueFenceSync>(&queue_fence_sync);
    install<Entry::nvnQueueWaitSync>(&queue_wait_sync);
    install<Entry::nvnQueuePresentTexture>(&queue_present_texture);
    install<Entry::nvnQueueAcquireTexture>(&queue_acquire_texture);

    // Windows.
    install<Entry::nvnWindowBuilderSetDevice>(&window_builder_set_device);
    install<Entry::nvnWindowBuilderSetDefaults>(&window_builder_set_defaults);
    install<Entry::nvnWindowBuilderSetNativeWindow>(&window_builder_set_native_window);
    install<Entry::nvnWindowBuilderSetTextures>(&window_builder_set_textures);
    install<Entry::nvnWindowBuilderSetPresentInterval>(&window_builder_set_present_interval);
    install<Entry::nvnWindowBuilderGetNativeWindow>(&window_builder_get_native_window);
    install<Entry::nvnWindowBuilderGetPresentInterval>(&window_builder_get_present_interval);
    install<Entry::nvnWindowInitialize>(&window_initialize);
    install<Entry::nvnWindowFinalize>(&window_finalize);
    install<Entry::nvnWindowGetPresentInterval>(&window_get_present_interval);
    install<Entry::nvnWindowSetPresentInterval>(&window_set_present_interval);
    install<Entry::nvnWindowSetCrop>(&window_set_crop);
    install<Entry::nvnWindowGetCrop>(&window_get_crop);
    install<Entry::nvnWindowGetNativeWindow>(&window_get_native_window);
    install<Entry::nvnWindowAcquireTexture>(&window_acquire_texture);

    // Sync objects.
    install<Entry::nvnSyncInitialize>(&sync_initialize);
    install<Entry::nvnSyncFinalize>(&sync_finalize);
    install<Entry::nvnSyncWait>(&sync_wait);

    // Command buffers: structure kept, recording accepted and dropped.
    install<Entry::nvnCommandBufferInitialize>(&command_buffer_initialize);
    install<Entry::nvnCommandBufferFinalize>(&command_buffer_finalize);
    install<Entry::nvnCommandBufferSetMemoryCallback>(&command_buffer_set_memory_callback);
    install<Entry::nvnCommandBufferSetMemoryCallbackData>(&command_buffer_set_memory_callback_data);
    install<Entry::nvnCommandBufferGetMemoryCallback>(&command_buffer_get_memory_callback);
    install<Entry::nvnCommandBufferAddCommandMemory>(&command_buffer_add_command_memory);
    install<Entry::nvnCommandBufferAddControlMemory>(&command_buffer_add_control_memory);
    install<Entry::nvnCommandBufferGetCommandMemorySize>(&command_buffer_get_command_memory_size);
    install<Entry::nvnCommandBufferGetCommandMemoryUsed>(&command_buffer_get_command_memory_used);
    install<Entry::nvnCommandBufferGetCommandMemoryFree>(&command_buffer_get_command_memory_free);
    install<Entry::nvnCommandBufferGetControlMemorySize>(&command_buffer_get_control_memory_size);
    install<Entry::nvnCommandBufferGetControlMemoryUsed>(&command_buffer_get_control_memory_used);
    install<Entry::nvnCommandBufferGetControlMemoryFree>(&command_buffer_get_control_memory_free);
    install<Entry::nvnCommandBufferBeginRecording>(&command_buffer_begin_recording);
    install<Entry::nvnCommandBufferEndRecording>(&command_buffer_end_recording);
    install<Entry::nvnCommandBufferIsRecording>(&command_buffer_is_recording);
    install<Entry::nvnCommandBufferFenceSync>(&command_buffer_fence_sync);
    install<Entry::nvnCommandBufferGetCopyRowStride>(&command_buffer_get_copy_row_stride);
    install<Entry::nvnCommandBufferGetCopyImageStride>(&command_buffer_get_copy_image_stride);

    // Recording surface: every typed command-buffer entry that only changes
    // GPU state or issues GPU work.
    install<Entry::nvnCommandBufferCallCommands>(&record_nothing<NVNcommandBuffer*, int, const NVNcommandHandle*>);
    install<Entry::nvnCommandBufferCopyCommands>(&record_nothing<NVNcommandBuffer*, int, const NVNcommandHandle*>);
    install<Entry::nvnCommandBufferBindBlendState>(&record_nothing<NVNcommandBuffer*, const NVNblendState*>);
    install<Entry::nvnCommandBufferBindChannelMaskState>(&record_nothing<NVNcommandBuffer*, const NVNchannelMaskState*>);
    install<Entry::nvnCommandBufferBindColorState>(&record_nothing<NVNcommandBuffer*, const NVNcolorState*>);
    install<Entry::nvnCommandBufferBindMultisampleState>(&record_nothing<NVNcommandBuffer*, const NVNmultisampleState*>);
    install<Entry::nvnCommandBufferBindPolygonState>(&record_nothing<NVNcommandBuffer*, const NVNpolygonState*>);
    install<Entry::nvnCommandBufferBindDepthStencilState>(&record_nothing<NVNcommandBuffer*, const NVNdepthStencilState*>);
    install<Entry::nvnCommandBufferBindVertexAttribState>(&record_nothing<NVNcommandBuffer*, int, const NVNvertexAttribState*>);
    install<Entry::nvnCommandBufferBindVertexStreamState>(&record_nothing<NVNcommandBuffer*, int, const NVNvertexStreamState*>);
    install<Entry::nvnCommandBufferBindProgram>(&record_nothing<NVNcommandBuffer*, const NVNprogram*, int>);
    install<Entry::nvnCommandBufferBindVertexBuffer>(&record_nothing<NVNcommandBuffer*, int, NVNbufferAddress, std::size_t>);
    install<Entry::nvnCommandBufferBindVertexBuffers>(&record_nothing<NVNcommandBuffer*, int, int, const NVNbufferRange*>);
    install<Entry::nvnCommandBufferBindUniformBuffer>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, NVNbufferAddress, std::size_t>);
    install<Entry::nvnCommandBufferBindUniformBuffers>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, int, const NVNbufferRange*>);
    install<Entry::nvnCommandBufferBindTransformFeedbackBuffer>(
        &record_nothing<NVNcommandBuffer*, int, NVNbufferAddress, std::size_t>);
    install<Entry::nvnCommandBufferBindTransformFeedbackBuffers>(
        &record_nothing<NVNcommandBuffer*, int, int, const NVNbufferRange*>);
    install<Entry::nvnCommandBufferBindStorageBuffer>(&record_nothing<NVNcommandBuffer*, NVNshaderStage, int, NVNbufferAddress, std::size_t>);
    install<Entry::nvnCommandBufferBindStorageBuffers>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, int, const NVNbufferRange*>);
    install<Entry::nvnCommandBufferBindTexture>(&record_nothing<NVNcommandBuffer*, NVNshaderStage, int, NVNtextureHandle>);
    install<Entry::nvnCommandBufferBindTextures>(&record_nothing<NVNcommandBuffer*, NVNshaderStage, int, int, const NVNtextureHandle*>);
    install<Entry::nvnCommandBufferBindImage>(&record_nothing<NVNcommandBuffer*, NVNshaderStage, int, NVNimageHandle>);
    install<Entry::nvnCommandBufferBindImages>(&record_nothing<NVNcommandBuffer*, NVNshaderStage, int, int, const NVNimageHandle*>);
    install<Entry::nvnCommandBufferSetPatchSize>(&record_nothing<NVNcommandBuffer*, int>);
    install<Entry::nvnCommandBufferSetInnerTessellationLevels>(&record_nothing<NVNcommandBuffer*, const float*>);
    install<Entry::nvnCommandBufferSetOuterTessellationLevels>(&record_nothing<NVNcommandBuffer*, const float*>);
    install<Entry::nvnCommandBufferSetPrimitiveRestart>(&record_nothing<NVNcommandBuffer*, std::uint32_t, int>);
    install<Entry::nvnCommandBufferBeginTransformFeedback>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress>);
    install<Entry::nvnCommandBufferEndTransformFeedback>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress>);
    install<Entry::nvnCommandBufferPauseTransformFeedback>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress>);
    install<Entry::nvnCommandBufferResumeTransformFeedback>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress>);
    install<Entry::nvnCommandBufferDrawTransformFeedback>(&record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNbufferAddress>);
    install<Entry::nvnCommandBufferDrawArrays>(&record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, int, int>);
    install<Entry::nvnCommandBufferDrawElements>(&record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, int, NVNbufferAddress>);
    install<Entry::nvnCommandBufferDrawElementsBaseVertex>(
        &record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, int, NVNbufferAddress, int>);
    install<Entry::nvnCommandBufferDrawArraysInstanced>(&record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, int, int, int, int>);
    install<Entry::nvnCommandBufferDrawElementsInstanced>(
        &record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, int, NVNbufferAddress, int, int, int>);
    install<Entry::nvnCommandBufferDrawArraysIndirect>(&record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNbufferAddress>);
    install<Entry::nvnCommandBufferDrawElementsIndirect>(
        &record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, NVNbufferAddress, NVNbufferAddress>);
    install<Entry::nvnCommandBufferMultiDrawArraysIndirectCount>(
        &record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNbufferAddress, NVNbufferAddress, int, ptrdiff_t>);
    install<Entry::nvnCommandBufferMultiDrawElementsIndirectCount>(
        &record_nothing<NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, NVNbufferAddress, NVNbufferAddress, NVNbufferAddress, int, ptrdiff_t>);
    install<Entry::nvnCommandBufferClearColor>(&record_nothing<NVNcommandBuffer*, int, const float*, int>);
    install<Entry::nvnCommandBufferClearColori>(&record_nothing<NVNcommandBuffer*, int, const int*, int>);
    install<Entry::nvnCommandBufferClearColorui>(&record_nothing<NVNcommandBuffer*, int, const std::uint32_t*, int>);
    install<Entry::nvnCommandBufferClearDepthStencil>(&record_nothing<NVNcommandBuffer*, float, std::uint32_t, int, int>);
    install<Entry::nvnCommandBufferDispatchCompute>(&record_nothing<NVNcommandBuffer*, int, int, int>);
    install<Entry::nvnCommandBufferDispatchComputeIndirect>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress>);
    install<Entry::nvnCommandBufferSetViewport>(&record_nothing<NVNcommandBuffer*, int, int, int, int>);
    install<Entry::nvnCommandBufferSetViewports>(&record_nothing<NVNcommandBuffer*, int, int, const float*>);
    install<Entry::nvnCommandBufferSetViewportSwizzles>(&record_nothing<NVNcommandBuffer*, int, int, const NVNviewportSwizzle*>);
    install<Entry::nvnCommandBufferSetScissor>(&record_nothing<NVNcommandBuffer*, int, int, int, int>);
    install<Entry::nvnCommandBufferSetScissors>(&record_nothing<NVNcommandBuffer*, int, int, const int*>);
    install<Entry::nvnCommandBufferSetDepthRange>(&record_nothing<NVNcommandBuffer*, float, float>);
    install<Entry::nvnCommandBufferSetDepthBounds>(&record_nothing<NVNcommandBuffer*, std::uint32_t, float, float>);
    install<Entry::nvnCommandBufferSetDepthRanges>(&record_nothing<NVNcommandBuffer*, int, int, const float*>);
    install<Entry::nvnCommandBufferSetTiledCacheAction>(&record_nothing<NVNcommandBuffer*, NVNtiledCacheAction>);
    install<Entry::nvnCommandBufferSetTiledCacheTileSize>(&record_nothing<NVNcommandBuffer*, int, int>);
    install<Entry::nvnCommandBufferBindSeparateTexture>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, NVNseparateTextureHandle>);
    install<Entry::nvnCommandBufferBindSeparateSampler>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, NVNseparateSamplerHandle>);
    install<Entry::nvnCommandBufferBindSeparateTextures>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, int, const NVNseparateTextureHandle*>);
    install<Entry::nvnCommandBufferBindSeparateSamplers>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, int, const NVNseparateSamplerHandle*>);
    install<Entry::nvnCommandBufferSetStencilValueMask>(&record_nothing<NVNcommandBuffer*, NVNface, int>);
    install<Entry::nvnCommandBufferSetStencilMask>(&record_nothing<NVNcommandBuffer*, NVNface, int>);
    install<Entry::nvnCommandBufferSetStencilRef>(&record_nothing<NVNcommandBuffer*, NVNface, int>);
    install<Entry::nvnCommandBufferSetBlendColor>(&record_nothing<NVNcommandBuffer*, const float*>);
    install<Entry::nvnCommandBufferSetPointSize>(&record_nothing<NVNcommandBuffer*, float>);
    install<Entry::nvnCommandBufferSetLineWidth>(&record_nothing<NVNcommandBuffer*, float>);
    install<Entry::nvnCommandBufferSetPolygonOffsetClamp>(&record_nothing<NVNcommandBuffer*, float, float, float>);
    install<Entry::nvnCommandBufferSetAlphaRef>(&record_nothing<NVNcommandBuffer*, float>);
    install<Entry::nvnCommandBufferSetSampleMask>(&record_nothing<NVNcommandBuffer*, int>);
    install<Entry::nvnCommandBufferSetRasterizerDiscard>(&record_nothing<NVNcommandBuffer*, std::uint32_t>);
    install<Entry::nvnCommandBufferSetDepthClamp>(&record_nothing<NVNcommandBuffer*, std::uint32_t>);
    install<Entry::nvnCommandBufferSetConservativeRasterEnable>(&record_nothing<NVNcommandBuffer*, std::uint32_t>);
    install<Entry::nvnCommandBufferSetConservativeRasterDilate>(&record_nothing<NVNcommandBuffer*, float>);
    install<Entry::nvnCommandBufferSetSubpixelPrecisionBias>(&record_nothing<NVNcommandBuffer*, int, int>);
    install<Entry::nvnCommandBufferCopyBufferToTexture>(
        &record_nothing<NVNcommandBuffer*, NVNbufferAddress, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, int>);
    install<Entry::nvnCommandBufferCopyTextureToBuffer>(
        &record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, NVNbufferAddress, int>);
    install<Entry::nvnCommandBufferCopyTextureToTexture>(
        &record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const NVNtexture*,
                        const NVNtextureView*, const NVNcopyRegion*, int>);
    install<Entry::nvnCommandBufferCopyBufferToBuffer>(
        &record_nothing<NVNcommandBuffer*, NVNbufferAddress, NVNbufferAddress, std::size_t, int>);
    install<Entry::nvnCommandBufferClearBuffer>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress, std::size_t, std::uint32_t>);
    install<Entry::nvnCommandBufferClearTexture>(
        &record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const float*, int>);
    install<Entry::nvnCommandBufferClearTexturei>(
        &record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const int*, int>);
    install<Entry::nvnCommandBufferClearTextureui>(
        &record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const std::uint32_t*, int>);
    install<Entry::nvnCommandBufferUpdateUniformBuffer>(
        &record_nothing<NVNcommandBuffer*, NVNbufferAddress, std::size_t, ptrdiff_t, std::size_t, const void*>);
    install<Entry::nvnCommandBufferReportCounter>(&record_nothing<NVNcommandBuffer*, NVNcounterType, NVNbufferAddress>);
    install<Entry::nvnCommandBufferResetCounter>(&record_nothing<NVNcommandBuffer*, NVNcounterType>);
    install<Entry::nvnCommandBufferReportValue>(&record_nothing<NVNcommandBuffer*, std::uint32_t, NVNbufferAddress>);
    install<Entry::nvnCommandBufferSetRenderEnable>(&record_nothing<NVNcommandBuffer*, std::uint32_t>);
    install<Entry::nvnCommandBufferSetRenderEnableConditional>(
        &record_nothing<NVNcommandBuffer*, NVNconditionalRenderMode, NVNbufferAddress>);
    install<Entry::nvnCommandBufferSetRenderTargets>(
        &record_nothing<NVNcommandBuffer*, int, const NVNtexture* const*, const NVNtextureView* const*, const NVNtexture*,
                        const NVNtextureView*>);
    install<Entry::nvnCommandBufferDiscardColor>(&record_nothing<NVNcommandBuffer*, int>);
    install<Entry::nvnCommandBufferDiscardDepthStencil>(&record_nothing<NVNcommandBuffer*>);
    install<Entry::nvnCommandBufferDownsample>(&record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtexture*>);
    install<Entry::nvnCommandBufferTiledDownsample>(&record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtexture*>);
    install<Entry::nvnCommandBufferDownsampleTextureView>(
        &record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNtexture*, const NVNtextureView*>);
    install<Entry::nvnCommandBufferTiledDownsampleTextureView>(
        &record_nothing<NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNtexture*, const NVNtextureView*>);
    install<Entry::nvnCommandBufferBarrier>(&record_nothing<NVNcommandBuffer*, int>);
    install<Entry::nvnCommandBufferWaitSync>(&record_nothing<NVNcommandBuffer*, const NVNsync*>);
    install<Entry::nvnCommandBufferSetTexturePool>(&record_nothing<NVNcommandBuffer*, const NVNtexturePool*>);
    install<Entry::nvnCommandBufferSetSamplerPool>(&record_nothing<NVNcommandBuffer*, const NVNsamplerPool*>);
    install<Entry::nvnCommandBufferSetShaderScratchMemory>(
        &record_nothing<NVNcommandBuffer*, const NVNmemoryPool*, ptrdiff_t, std::size_t>);
    install<Entry::nvnCommandBufferSaveZCullData>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress, std::size_t>);
    install<Entry::nvnCommandBufferRestoreZCullData>(&record_nothing<NVNcommandBuffer*, NVNbufferAddress, std::size_t>);
    install<Entry::nvnCommandBufferSetCopyRowStride>(&record_nothing<NVNcommandBuffer*, ptrdiff_t>);
    install<Entry::nvnCommandBufferSetCopyImageStride>(&record_nothing<NVNcommandBuffer*, ptrdiff_t>);
    install<Entry::nvnCommandBufferDrawTexture>(
        &record_nothing<NVNcommandBuffer*, NVNtextureHandle, const NVNdrawTextureRegion*, const NVNdrawTextureRegion*>);
    install<Entry::nvnCommandBufferSetProgramSubroutines>(
        &record_nothing<NVNcommandBuffer*, NVNprogram*, NVNshaderStage, const int, const int, const int*>);
    install<Entry::nvnCommandBufferBindCoverageModulationTable>(&record_nothing<NVNcommandBuffer*, const float*>);
    install<Entry::nvnCommandBufferResolveDepthBuffer>(&record_nothing<NVNcommandBuffer*>);
    install<Entry::nvnCommandBufferPushDebugGroupStatic>(&record_nothing<NVNcommandBuffer*, std::uint32_t, const char*>);
    install<Entry::nvnCommandBufferPushDebugGroupDynamic>(&record_nothing<NVNcommandBuffer*, std::uint32_t, const char*>);
    install<Entry::nvnCommandBufferPushDebugGroup>(&record_nothing<NVNcommandBuffer*, std::uint32_t, const char*>);
    install<Entry::nvnCommandBufferPopDebugGroup>(&record_nothing<NVNcommandBuffer*>);
    install<Entry::nvnCommandBufferPopDebugGroupId>(&record_nothing<NVNcommandBuffer*, std::uint32_t>);
    install<Entry::nvnCommandBufferInsertDebugMarkerStatic>(&record_nothing<NVNcommandBuffer*, std::uint32_t, const char*>);
    install<Entry::nvnCommandBufferInsertDebugMarkerDynamic>(&record_nothing<NVNcommandBuffer*, std::uint32_t, const char*>);
    install<Entry::nvnCommandBufferInsertDebugMarker>(&record_nothing<NVNcommandBuffer*, const char*>);
    install<Entry::nvnCommandBufferWaitEvent>(&record_nothing<NVNcommandBuffer*, const NVNevent*, NVNeventWaitMode, std::uint32_t>);
    install<Entry::nvnCommandBufferSignalEvent>(
        &record_nothing<NVNcommandBuffer*, const NVNevent*, NVNeventSignalMode, NVNeventSignalLocation, int, std::uint32_t>);
    install<Entry::nvnCommandBufferBindTexturesDeferred>(
        &record_nothing<NVNcommandBuffer*, NVNshaderStage, int, int, const NVNtextureHandle*>);

    // Boot: the loader entry itself, so a resolved bootstrap pointer behaves
    // like the redirected SDK entry.
    install<Entry::nvnBootstrapLoader>(&armigo_nvn_bootstrap_loader);
}

} // namespace armigo::nvn
