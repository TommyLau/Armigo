// ============================================================================
// Armigo NVN - Driver Object Layouts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The driver's state for every NVN object lives in place, inside the storage
// the game allocated for it (the API reserves a fixed size per object type).
// Each layout here is checked against that size. Games may copy objects by
// value, so host-side resources (Metal textures, pipelines) are found through
// a serial number stored in the object, never through the object's address.
//
// Ownership between the driver subsystems:
//   device, memory pools, buffers, textures, samplers, descriptor pools: resources
//   programs, state objects, command buffers, command records: commands
//   queues, sync objects, events, windows: present

#pragma once

#include <armigo/nvn/api_types.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace armigo::nvn {

using namespace api;

// Resource creation and CPU publications share one monotonic ordering domain.
inline std::uint64_t next_resource_serial() noexcept {
    static std::atomic<std::uint64_t> serial{1};
    return serial.fetch_add(1, std::memory_order_relaxed);
}

constexpr std::uint32_t object_tag(char a, char b, char c, char d) noexcept {
    return std::uint32_t(std::uint8_t(a)) | std::uint32_t(std::uint8_t(b)) << 8 |
           std::uint32_t(std::uint8_t(c)) << 16 | std::uint32_t(std::uint8_t(d)) << 24;
}

// A layout must fit the storage the API reserves for its object type.
#define ARMIGO_NVN_FITS(State, Storage) \
    static_assert(sizeof(State) <= sizeof(Storage), #State " exceeds the storage of " #Storage)

// ---------------------------------------------------------------- device

struct DeviceBuilderState {
    std::uint32_t flags;
};
ARMIGO_NVN_FITS(DeviceBuilderState, NVNdeviceBuilder);

struct DeviceState {
    std::uint32_t tag, flags, window_origin, depth_mode;
    void* debug_callback;
    void* debug_data;
};
ARMIGO_NVN_FITS(DeviceState, NVNdevice);

// ---------------------------------------------------------------- queues

struct QueueBuilderState {
    NVNdevice* device;
    std::uint32_t flags;
    std::uint64_t command_memory_size, compute_memory_size, control_memory_size, flush_threshold;
    void* memory;
    std::uint64_t memory_size;
};
ARMIGO_NVN_FITS(QueueBuilderState, NVNqueueBuilder);

struct QueueState {
    std::uint32_t tag;
    NVNdevice* device;
    std::uint64_t submits;
};
ARMIGO_NVN_FITS(QueueState, NVNqueue);

// ---------------------------------------------------------------- memory

struct MemoryPoolBuilderState {
    NVNdevice* device;
    void* memory;
    std::uint64_t size;
    std::uint32_t flags;
};
ARMIGO_NVN_FITS(MemoryPoolBuilderState, NVNmemoryPoolBuilder);

// GPU addresses equal guest addresses: a pool's GPU base is its memory.
// flush_begin/flush_end hold the latest FlushMappedRange (end 0 when none),
// ordered by flush_serial.
struct MemoryPoolState {
    std::uint32_t tag, flags;
    std::uint8_t* memory;
    std::uint64_t size;
    std::uint64_t flush_begin, flush_end, flush_serial;
};
ARMIGO_NVN_FITS(MemoryPoolState, NVNmemoryPool);

// Whether the CPU maps the pool: neither CPU_NO_ACCESS nor VIRTUAL. Texel
// flushes of a pool without a mapping do nothing, so nothing certifies its
// textures for them either.
inline bool pool_cpu_mapped(const MemoryPoolState& pool) noexcept {
    return !(pool.flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL));
}

// Texture and sampler descriptor pools.
struct DescriptorPoolState {
    const MemoryPoolState* pool;
    std::int64_t offset;
    std::int32_t count;
};
ARMIGO_NVN_FITS(DescriptorPoolState, NVNtexturePool);
ARMIGO_NVN_FITS(DescriptorPoolState, NVNsamplerPool);

// ---------------------------------------------------------------- textures

// Channel selectors packed as 3 bits each (ZERO=0, ONE=1, R/G/B/A=2..5).
// Zero means identity; bit 12 marks an explicit mapping.
constexpr std::uint16_t pack_swizzle(unsigned r, unsigned g, unsigned b, unsigned a) noexcept {
    if (r == 2 && g == 3 && b == 4 && a == 5) return 0;
    return std::uint16_t(0x1000u | r | (g << 3) | (b << 6) | (a << 9));
}
constexpr unsigned swizzle_channel(std::uint16_t packed, unsigned channel) noexcept {
    return packed & 0x1000u ? (packed >> (3 * channel)) & 7u : channel + 2;
}

struct TextureDescription {
    std::int32_t target, format, flags, samples;  // samples 0 or 1: single-sampled
    std::uint32_t width, height, depth, levels, stride;
    // Block-height hint from a packaged layout: log2 of the block height + 1;
    // 0 selects the automatic block height.
    std::int32_t packaged_block_height;
};

struct TextureBuilderState {
    NVNdevice* device;
    TextureDescription description;
    MemoryPoolState* pool;
    std::int64_t offset;
    std::uint8_t has_packaged_layout, packaged_layout[8];
    std::uint16_t swizzle;
    std::uint32_t depth_stencil_mode;
};
ARMIGO_NVN_FITS(TextureBuilderState, NVNtextureBuilder);
// Games allocate texture builders with no more room than this; rarely used
// builder values (packaged data, GL names) live outside the object.
static_assert(sizeof(TextureBuilderState) == 80);

struct TextureState {
    std::uint32_t tag;
    TextureDescription description;
    std::uint64_t size, layer_size;
    std::uint8_t* memory;
    MemoryPoolState* pool;
    std::int64_t offset;
    std::uint8_t has_packaged_layout, packaged_layout[8];
    std::uint16_t swizzle;
    std::uint32_t depth_stencil_mode;
    // Serial assigned at initialization; host tables are keyed by it.
    std::uint64_t serial;
    // Immutable snapshot of this serial that command records point at, so a
    // later rewrite of the game's object never changes recorded commands.
    const void* snapshot;
};
ARMIGO_NVN_FITS(TextureState, NVNtexture);
static_assert(sizeof(TextureState) <= 0x80, "Texture storage is 128 bytes");

// The game reads view fields in place: this layout is the API's own.
struct TextureViewState {
    std::int16_t min_layer, layer_count;
    std::uint8_t base_level, level_count, target;
    std::uint8_t sets_levels : 1, sets_layers : 1, sets_format : 1, sets_swizzle : 1, sets_depth_stencil : 1,
        sets_target : 1, depth_stencil_mode : 2;
    std::int32_t format;
    std::uint8_t swizzle[4];
};
ARMIGO_NVN_FITS(TextureViewState, NVNtextureView);
static_assert(sizeof(TextureViewState) == 16 && offsetof(TextureViewState, format) == 8 &&
              offsetof(TextureViewState, swizzle) == 12);

// Retires the texture identities of a memory pool that is being released, so
// a later pool at the same address starts new identities (textures.cpp).
void retire_pool_textures(const MemoryPoolState* pool) noexcept;

// ---------------------------------------------------------------- samplers

struct SamplerParameters {
    std::uint8_t min_filter, mag_filter, wrap_s, wrap_t, wrap_r, compare_mode, compare_function, reduction_filter;
    float min_lod, max_lod, lod_bias, max_anisotropy, border[4];
    float lod_snap;
};

struct SamplerBuilderState {
    NVNdevice* device;
    SamplerParameters parameters;
};
ARMIGO_NVN_FITS(SamplerBuilderState, NVNsamplerBuilder);

// The object's bytes are a pure function of the sampler's state and device: a
// game may hash or compare them (a sampler cache keyed on the raw object does),
// so nothing unique to the object lives in them. nvnSamplerGetDebugID derives
// its value from the object's address instead.
struct SamplerState {
    std::uint32_t tag;
    SamplerParameters parameters;
    NVNdevice* device;
    std::uint64_t reserved;
};
ARMIGO_NVN_FITS(SamplerState, NVNsampler);
static_assert(offsetof(SamplerState, parameters) == 4 && sizeof(SamplerState) <= 64);

// Every sampler registration takes the next number. A recorded texture or
// sampler binding keeps the number current when it was recorded, so the replay
// resolves its sampler as registered at that point of the guest's commands:
// a game may register one sampler ID again between the draws of one command
// buffer.
// Draws and dispatches stamp the number (binding records too), and handles read
// from guest memory resolve the same way as bound ones.
// Known difference from resolving at GPU execution: a command buffer recorded
// once and submitted again after its sampler was registered anew keeps the
// version it was recorded with (until 64 newer versions push it out, after
// which the newest is used and reported). Title 01007EF00011E000 meets none:
// its pre-recorded command buffers never use a sampler registered again.
inline std::atomic<std::uint64_t> sampler_registrations{0};

// ---------------------------------------------------------------- descriptor handles

// Texture IDs occupy 20 bits and sampler IDs 12 bits. Handles that carry a
// texture set bit 32 after sign-extending the packed 32-bit value; that bit is
// not part of the sampler ID.
constexpr std::uint64_t combined_handle(std::int32_t texture, std::int32_t sampler) noexcept {
    const std::uint32_t packed = std::uint32_t(texture) | (std::uint32_t(sampler) << 20);
    return std::uint64_t(std::int64_t(std::int32_t(packed))) | 0x100000000ull;
}
constexpr std::uint64_t texture_handle(std::int32_t texture) noexcept {
    return std::uint64_t(std::int64_t(texture)) | 0x100000000ull;
}
constexpr std::uint64_t sampler_handle(std::int32_t sampler) noexcept {
    return std::uint64_t(std::int64_t(sampler)) << 20;
}
constexpr unsigned handle_texture_id(std::uint64_t handle) noexcept { return unsigned(handle & 0xfffffu); }
constexpr unsigned handle_sampler_id(std::uint64_t handle) noexcept { return unsigned((handle >> 20) & 0xfffu); }

// ---------------------------------------------------------------- programs

struct ProgramState {
    std::uint32_t tag, stage_count;
    NVNshaderData stages[6];
    std::uint64_t serial;
    const void* snapshot;
    NVNdevice* device;
};
ARMIGO_NVN_FITS(ProgramState, NVNprogram);
static_assert(sizeof(ProgramState) == 128, "Program storage is 128 bytes");

// ---------------------------------------------------------------- sync, events, windows

// A sync object is a queue fence value; waits return once the GPU passed it.
struct SyncState {
    std::uint32_t tag, fenced;
    std::uint64_t fence;
};
ARMIGO_NVN_FITS(SyncState, NVNsync);

struct EventBuilderState {
    const MemoryPoolState* pool;
    std::int64_t offset;
};
ARMIGO_NVN_FITS(EventBuilderState, NVNeventBuilder);

struct EventState {
    std::uint32_t tag;
    volatile std::uint32_t* value;
};
ARMIGO_NVN_FITS(EventState, NVNevent);

struct WindowBuilderState {
    NVNdevice* device;
    void* native_window;
    std::int32_t texture_count, present_interval;
    NVNtexture* textures[4];
};
ARMIGO_NVN_FITS(WindowBuilderState, NVNwindowBuilder);

// available_at[i]: monotonic time (ns) at which image i may be acquired again;
// present_ticket[i]: the latest present of image i.
struct WindowState {
    std::uint32_t tag;
    void* native_window;
    std::int32_t texture_count, present_interval, next, crop[4];
    NVNtexture* textures[4];
    std::uint64_t presents, available_at[4], pace_target, present_ticket[4];
};
ARMIGO_NVN_FITS(WindowState, NVNwindow);

// ---------------------------------------------------------------- command buffers

struct CommandList;

// Commands are recorded into the command memory the game adds; the command
// handle is the start of a recording. When memory runs out, the game's memory
// callback is asked for more.
struct CommandBufferState {
    std::uint32_t tag, recording;
    std::uint8_t* cursor;
    std::uint8_t* end;
    std::uint8_t* begin_record;
    PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC memory_callback;
    void* callback_data;
    NVNdevice* device;
    std::uint8_t* segment_base;
    // Game-donated segment and the position a native driver would have
    // written to, while records live in host blocks.
    std::uint8_t* donated_base;
    std::uint8_t* donated_end;
    std::uint8_t* donated_cursor;
    CommandList* list;
    // The out-of-memory callback ran for this segment without adding memory.
    std::uint32_t callback_exhausted;
    // The control-memory callback could not provide more memory; a record of
    // the current recording could not be made.
    std::uint8_t control_exhausted, recording_failed;
    // Game-donated control memory and the position a native driver would have
    // used, and the strides SetCopyRowStride/SetCopyImageStride last set.
    std::uint8_t* control_base;
    std::uint8_t* control_cursor;
    std::uint8_t* control_end;
    std::int64_t copy_row_stride, copy_image_stride;
};
ARMIGO_NVN_FITS(CommandBufferState, NVNcommandBuffer);

// ---------------------------------------------------------------- state objects

// Defaults follow the API defaults. Layouts are packed into the object storage.
struct ColorStateBits {
    std::uint32_t blend_enable : 8, logic_op : 4, alpha_test : 4;
};
ARMIGO_NVN_FITS(ColorStateBits, NVNcolorState);

// Four bits per target (r=1 g=2 b=4 a=8); a set bit enables the write.
struct ChannelMaskStateBits {
    std::uint32_t mask;
};
ARMIGO_NVN_FITS(ChannelMaskStateBits, NVNchannelMaskState);

struct BlendStateBits {
    std::uint64_t target : 3, source_color : 7, destination_color : 7, source_alpha : 7, destination_alpha : 7,
        color_equation : 3, alpha_equation : 3;
};
ARMIGO_NVN_FITS(BlendStateBits, NVNblendState);

struct MultisampleStateBits {
    std::uint8_t enable : 1, alpha_to_coverage : 1, alpha_to_coverage_dither : 1,
        coverage_to_color_enable : 1, sample_locations_enable : 1, sample_locations_grid : 1;
    std::uint8_t samples, raster_samples, coverage_modulation, coverage_to_color_output;
    // Coordinates use the API's 1/16-pixel grid: x in the low nibble, y in the high nibble.
    std::uint8_t sample_locations[16];
    std::uint16_t sample_location_mask;
};
ARMIGO_NVN_FITS(MultisampleStateBits, NVNmultisampleState);
static_assert(sizeof(MultisampleStateBits) == 24);

struct PolygonStateBits {
    std::uint32_t cull : 2, front_face : 1, mode : 2, offset_enables : 3;
};
ARMIGO_NVN_FITS(PolygonStateBits, NVNpolygonState);

struct DepthStencilStateBits {
    std::uint64_t depth_test : 1, depth_write : 1, depth_function : 4, stencil_test : 1, front_function : 4,
        back_function : 4, front_fail : 4, front_depth_fail : 4, front_pass : 4, back_fail : 4, back_depth_fail : 4,
        back_pass : 4;
};
ARMIGO_NVN_FITS(DepthStencilStateBits, NVNdepthStencilState);

struct VertexAttributeStateBits {
    std::uint32_t format : 8, offset : 19, stream : 5;
};
ARMIGO_NVN_FITS(VertexAttributeStateBits, NVNvertexAttribState);

struct VertexStreamStateBits {
    std::uint32_t stride, divisor;
};
ARMIGO_NVN_FITS(VertexStreamStateBits, NVNvertexStreamState);

#undef ARMIGO_NVN_FITS

} // namespace armigo::nvn
