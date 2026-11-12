// ============================================================================
// Armigo NVN - Device Entry Points
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
// Device builder and device entry points. The device owns no host
// resources; its state lives in place (objects.h) and holds the defaults
// the rest of the driver reads (window origin, depth mode, debug callback).
// DeviceGetInteger reports the selected API version (55.15 or 55.12, whose
// entry points are the same) and the measured Metal capabilities of the
// constant table below for every other property; the descriptor handle entry
// points are pure
// packing: texture IDs occupy 20 bits, sampler IDs 12, with the texture
// marker bit above 32 (objects.h). Object pointers are taken as given; the
// getters of a null device return the API default instead of stopping the
// host, and each such misuse is reported once.

#include "entry_table.h"
#include "objects.h"

#include <array>
#include <atomic>
#include <cstdio>

namespace armigo::nvn {
namespace {

using namespace api;

constexpr std::uint32_t live_device_tag = object_tag('D', 'E', 'V', 0);

// Reports the first occurrence of a misuse or unsupported case.
template <std::size_t Site>
void report_once(const char* text) noexcept {
    static std::atomic<bool> reported{false};
    if (!reported.exchange(true, std::memory_order_relaxed))
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: %s\n", text);
}

DeviceBuilderState* builder_of(NVNdeviceBuilder* builder) noexcept {
    return reinterpret_cast<DeviceBuilderState*>(builder);
}

DeviceState* state_of(NVNdevice* device) noexcept { return reinterpret_cast<DeviceState*>(device); }
const DeviceState* state_of(const NVNdevice* device) noexcept {
    return reinterpret_cast<const DeviceState*>(device);
}

// Device properties of the implemented API version (55.15), indexed by
// NVNdeviceInfo. Index 0x00/0x01 must stay equal to the version the entry
// table selects; entries the target hardware measures as supported are 1.
constexpr std::size_t device_info_count = 0x62;
constexpr std::array<int, device_info_count> make_device_info() {
    std::array<int, device_info_count> values{};
    const auto set = [&values](NVNdeviceInfo info, int value) {
        values[static_cast<std::size_t>(info)] = value;
    };
    set(NVN_DEVICE_INFO_API_MAJOR_VERSION, 55);
    set(NVN_DEVICE_INFO_API_MINOR_VERSION, 15);
    set(NVN_DEVICE_INFO_UNIFORM_BUFFER_BINDINGS_PER_STAGE, 14);
    set(NVN_DEVICE_INFO_MAX_UNIFORM_BUFFER_SIZE, 65536);
    set(NVN_DEVICE_INFO_UNIFORM_BUFFER_ALIGNMENT, 256);
    set(NVN_DEVICE_INFO_COLOR_BUFFER_BINDINGS, 8);
    set(NVN_DEVICE_INFO_VERTEX_BUFFER_BINDINGS, 16);
    set(NVN_DEVICE_INFO_TRANSFORM_FEEDBACK_BUFFER_BINDINGS, 4);
    set(NVN_DEVICE_INFO_SHADER_STORAGE_BUFFER_BINDINGS_PER_STAGE, 16);
    set(NVN_DEVICE_INFO_TEXTURE_BINDINGS_PER_STAGE, 32);
    set(NVN_DEVICE_INFO_COUNTER_ALIGNMENT, 16);
    set(NVN_DEVICE_INFO_TRANSFORM_FEEDBACK_BUFFER_ALIGNMENT, 4);
    set(NVN_DEVICE_INFO_TRANSFORM_FEEDBACK_CONTROL_ALIGNMENT, 4);
    set(NVN_DEVICE_INFO_INDIRECT_DRAW_ALIGNMENT, 4);
    set(NVN_DEVICE_INFO_VERTEX_ATTRIBUTES, 16);
    set(NVN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE, 32);
    set(NVN_DEVICE_INFO_SAMPLER_DESCRIPTOR_SIZE, 32);
    set(NVN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS, 256);
    set(NVN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS, 256);
    set(NVN_DEVICE_INFO_COMMAND_BUFFER_COMMAND_ALIGNMENT, 4);
    set(NVN_DEVICE_INFO_COMMAND_BUFFER_CONTROL_ALIGNMENT, 8);
    set(NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE, 512);
    set(NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE, 256);
    set(NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_MINIMUM, 2);
    set(NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_RECOMMENDED, 256);
    set(NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_ALIGNMENT, 4096);
    set(NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_GRANULARITY, 131072);
    set(NVN_DEVICE_INFO_MAX_TEXTURE_ANISOTROPY, 16);
    set(NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_X, 1536);
    set(NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Y, 1024);
    set(NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Z, 64);
    set(NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_THREADS, 1536);
    set(NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_X, 65535);
    set(NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Y, 65535);
    set(NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Z, 65535);
    set(NVN_DEVICE_INFO_IMAGE_BINDINGS_PER_STAGE, 8);
    set(NVN_DEVICE_INFO_MAX_TEXTURE_POOL_SIZE, 1048576);
    set(NVN_DEVICE_INFO_MAX_SAMPLER_POOL_SIZE, 4096);
    set(NVN_DEVICE_INFO_MAX_VIEWPORTS, 16);
    set(NVN_DEVICE_INFO_MEMPOOL_TEXTURE_OBJECT_PAGE_ALIGNMENT, 4096);
    set(NVN_DEVICE_INFO_SUPPORTS_MIN_MAX_FILTERING, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_STENCIL8_FORMAT, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_ASTC_FORMATS, 1);
    set(NVN_DEVICE_INFO_L2_SIZE, 262144);
    set(NVN_DEVICE_INFO_MAX_TEXTURE_LEVELS, 15);
    set(NVN_DEVICE_INFO_MAX_TEXTURE_LAYERS, 2048);
    set(NVN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MAJOR_VERSION, 1);
    set(NVN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MAJOR_VERSION, 1);
    set(NVN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MINOR_VERSION, 16);
    set(NVN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MINOR_VERSION, 5);
    set(NVN_DEVICE_INFO_SUPPORTS_CONSERVATIVE_RASTER, 1);
    set(NVN_DEVICE_INFO_SUBPIXEL_BITS, 8);
    set(NVN_DEVICE_INFO_MAX_SUBPIXEL_BIAS_BITS, 8);
    set(NVN_DEVICE_INFO_INDIRECT_DISPATCH_ALIGNMENT, 4);
    set(NVN_DEVICE_INFO_ZCULL_SAVE_RESTORE_ALIGNMENT, 32);
    set(NVN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_COMPUTE_SCALE_FACTOR_MINIMUM, 128);
    set(NVN_DEVICE_INFO_LINEAR_TEXTURE_STRIDE_ALIGNMENT, 32);
    set(NVN_DEVICE_INFO_LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT, 128);
    set(NVN_DEVICE_INFO_MEMORY_POOL_PAGE_SIZE, 65536);
    set(NVN_DEVICE_INFO_SUPPORTS_ZERO_FROM_UNMAPPED_VIRTUAL_POOL_PAGES, 1);
    set(NVN_DEVICE_INFO_UNIFORM_BUFFER_UPDATE_ALIGNMENT, 4);
    set(NVN_DEVICE_INFO_MAX_TEXTURE_SIZE, 16384);
    set(NVN_DEVICE_INFO_MAX_BUFFER_TEXTURE_SIZE, 134217728);
    set(NVN_DEVICE_INFO_MAX_3D_TEXTURE_SIZE, 2048);
    set(NVN_DEVICE_INFO_MAX_CUBE_MAP_TEXTURE_SIZE, 16384);
    set(NVN_DEVICE_INFO_MAX_RECTANGLE_TEXTURE_SIZE, 16384);
    set(NVN_DEVICE_INFO_SUPPORTS_PASSTHROUGH_GEOMETRY_SHADERS, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_VIEWPORT_SWIZZLE, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_SPARSE_TILED_PACKAGED_TEXTURES, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_ADVANCED_BLEND_MODES, 1);
    set(NVN_DEVICE_INFO_MAX_PRESENT_INTERVAL, 5);
    set(NVN_DEVICE_INFO_SUPPORTS_DRAW_TEXTURE, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_TARGET_INDEPENDENT_RASTERIZATION, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_COVERAGE_TO_COLOR, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_POST_DEPTH_COVERAGE, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_IMAGES_USING_TEXTURE_HANDLES, 1);
    set(NVN_DEVICE_INFO_SUPPORTS_SAMPLE_LOCATIONS, 1);
    set(NVN_DEVICE_INFO_MAX_SAMPLE_LOCATION_TABLE_ENTRIES, 16);
    set(NVN_DEVICE_INFO_SHADER_CODE_MEMORY_POOL_PADDING_SIZE, 1024);
    set(NVN_DEVICE_INFO_MAX_PATCH_SIZE, 32);
    set(NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_GRANULARITY, 4096);
    set(NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_MIN_SIZE, 65536);
    set(NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_DEFAULT_SIZE, 65536);
    set(NVN_DEVICE_INFO_QUEUE_COMPUTE_MEMORY_GRANULARITY, 256);
    set(NVN_DEVICE_INFO_QUEUE_COMPUTE_MEMORY_MIN_SIZE, 16384);
    set(NVN_DEVICE_INFO_QUEUE_COMPUTE_MEMORY_DEFAULT_SIZE, 262144);
    set(NVN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_MIN_FLUSH_THRESHOLD, 4096);
    set(NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_SHADER_INTERLOCK, 1);
    set(NVN_DEVICE_INFO_MAX_TEXTURES_PER_WINDOW, 4);
    set(NVN_DEVICE_INFO_MIN_TEXTURES_PER_WINDOW, 2);
    set(NVN_DEVICE_INFO_SUPPORTS_DEBUG_LAYER, 0);
    set(NVN_DEVICE_INFO_QUEUE_CONTROL_MEMORY_MIN_SIZE, 4096);
    set(NVN_DEVICE_INFO_QUEUE_CONTROL_MEMORY_DEFAULT_SIZE, 16384);
    set(NVN_DEVICE_INFO_QUEUE_CONTROL_MEMORY_GRANULARITY, 4096);
    set(NVN_DEVICE_INFO_SEPARATE_TEXTURE_BINDINGS_PER_STAGE, 128);
    set(NVN_DEVICE_INFO_SEPARATE_SAMPLER_BINDINGS_PER_STAGE, 32);
    set(NVN_DEVICE_INFO_DEBUG_GROUPS_MAX_DOMAIN_ID, 65535);
    set(NVN_DEVICE_INFO_EVENTS_SUPPORT_REDUCTION_OPERATIONS, 1);
    return values;
}
constexpr std::array<int, device_info_count> device_info = make_device_info();

void device_builder_set_defaults(NVNdeviceBuilder* builder) noexcept {
    *builder_of(builder) = DeviceBuilderState{};
}

int device_builder_get_flags(const NVNdeviceBuilder* builder) noexcept {
    return static_cast<int>(reinterpret_cast<const DeviceBuilderState*>(builder)->flags);
}

void device_builder_set_flags(NVNdeviceBuilder* builder, int flags) noexcept {
    builder_of(builder)->flags = static_cast<std::uint32_t>(flags);
}

NVNboolean device_initialize(NVNdevice* device, const NVNdeviceBuilder* builder) noexcept {
    if (!device || !builder) {
        report_once<__LINE__>("nvnDeviceInitialize requires a device and a builder");
        return 0;
    }
    DeviceState* state = state_of(device);
    state->tag = live_device_tag;
    state->flags = reinterpret_cast<const DeviceBuilderState*>(builder)->flags;
    state->window_origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
    state->depth_mode = NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
    state->debug_callback = nullptr;
    state->debug_data = nullptr;
    const auto version = selected_api_version();
    state->gob_order = version && version->major == 231 ? 1 : 0;
    return 1;
}

void device_finalize(NVNdevice* device) noexcept {
    if (DeviceState* state = state_of(device)) state->tag = 0;
}

void device_set_debug_label(NVNdevice*, const char*) noexcept {}

PFNNVNGENERICFUNCPTRPROC device_get_proc_address(const NVNdevice*, const char* name) noexcept {
    return armigo_nvn_bootstrap_loader(name);
}

void device_get_integer(const NVNdevice*, NVNdeviceInfo what, int* out) noexcept {
    if (!out) {
        report_once<__LINE__>("nvnDeviceGetInteger without a result address");
        return;
    }
    const auto index = static_cast<std::uint32_t>(what);
    if (index >= device_info.size()) {
        report_once<__LINE__>("nvnDeviceGetInteger with an unknown device info; returning 0");
        *out = 0;
        return;
    }
    *out = device_info[index];
    if (const auto version = selected_api_version()) {
        if (what == NVN_DEVICE_INFO_API_MAJOR_VERSION) *out = version->major;
        if (what == NVN_DEVICE_INFO_API_MINOR_VERSION) *out = version->minor;
    }
}

// Fast-clear registration names clear values the hardware may compress. Metal
// clears any value through a load action, so a registration on a live device
// succeeds and changes nothing else; clears themselves are unaffected.
bool live(const NVNdevice* device) noexcept {
    const DeviceState* state = state_of(device);
    return state && state->tag == live_device_tag;
}

NVNboolean device_register_fast_clear_color(NVNdevice* device, const float* color, NVNformat) noexcept {
    return live(device) && color;
}

NVNboolean device_register_fast_clear_colori(NVNdevice* device, const int* color, NVNformat) noexcept {
    return live(device) && color;
}

NVNboolean device_register_fast_clear_colorui(NVNdevice* device, const std::uint32_t* color, NVNformat) noexcept {
    return live(device) && color;
}

NVNboolean device_register_fast_clear_depth(NVNdevice* device, float) noexcept { return live(device); }

NVNtextureHandle device_get_texture_handle(const NVNdevice*, int texture, int sampler) noexcept {
    return combined_handle(texture, sampler);
}

NVNtextureHandle device_get_texel_fetch_handle(const NVNdevice*, int texture) noexcept {
    return texture_handle(texture);
}

NVNimageHandle device_get_image_handle(const NVNdevice*, int id) noexcept {
    return static_cast<NVNimageHandle>(static_cast<std::uint64_t>(static_cast<std::int64_t>(id)));
}

NVNseparateTextureHandle device_get_separate_texture_handle(const NVNdevice*, int id) noexcept {
    return NVNseparateTextureHandle{texture_handle(id)};
}

NVNseparateSamplerHandle device_get_separate_sampler_handle(const NVNdevice*, int id) noexcept {
    return NVNseparateSamplerHandle{sampler_handle(id)};
}

void device_install_debug_callback(NVNdevice* device, const PFNNVNDEBUGCALLBACKPROC callback, void* data,
                                   std::uint32_t enable) noexcept {
    if (DeviceState* state = state_of(device)) {
        state->debug_callback = guest_bool(enable) ? reinterpret_cast<void*>(callback) : nullptr;
        state->debug_data = data;
    }
}

void device_set_window_origin_mode(NVNdevice* device, NVNwindowOriginMode mode) noexcept {
    if (DeviceState* state = state_of(device)) state->window_origin = static_cast<std::uint32_t>(mode);
}

NVNwindowOriginMode device_get_window_origin_mode(const NVNdevice* device) noexcept {
    if (const DeviceState* state = state_of(device))
        return static_cast<NVNwindowOriginMode>(state->window_origin);
    report_once<__LINE__>("nvnDeviceGetWindowOriginMode without a device; returning the default");
    return NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
}

void device_set_depth_mode(NVNdevice* device, NVNdepthMode mode) noexcept {
    if (DeviceState* state = state_of(device)) state->depth_mode = static_cast<std::uint32_t>(mode);
}

NVNdepthMode device_get_depth_mode(const NVNdevice* device) noexcept {
    if (const DeviceState* state = state_of(device)) return static_cast<NVNdepthMode>(state->depth_mode);
    report_once<__LINE__>("nvnDeviceGetDepthMode without a device; returning the default");
    return NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
}

} // namespace

void register_device_entries() noexcept {
    install<Entry::nvnDeviceBuilderGetFlags>(&device_builder_get_flags);
    install<Entry::nvnDeviceBuilderSetDefaults>(&device_builder_set_defaults);
    install<Entry::nvnDeviceBuilderSetFlags>(&device_builder_set_flags);
    install<Entry::nvnDeviceInitialize>(&device_initialize);
    install<Entry::nvnDeviceFinalize>(&device_finalize);
    install<Entry::nvnDeviceSetDebugLabel>(&device_set_debug_label);
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
    install<Entry::nvnDeviceRegisterFastClearColor>(&device_register_fast_clear_color);
    install<Entry::nvnDeviceRegisterFastClearColori>(&device_register_fast_clear_colori);
    install<Entry::nvnDeviceRegisterFastClearColorui>(&device_register_fast_clear_colorui);
    install<Entry::nvnDeviceRegisterFastClearDepth>(&device_register_fast_clear_depth);
}

} // namespace armigo::nvn
