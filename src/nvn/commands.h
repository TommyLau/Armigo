// ============================================================================
// Armigo NVN - Command Records
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Command buffers record GPU work as compact records: a header {op, size}
// (size includes the header, rounded up to 8 bytes) followed by the payload.
// Records that only change GPU state are replayed at queue submit; records
// with CPU-visible effects (event signals, counter reports, buffer copies and
// uniform updates) are ordered with the GPU work around them. Texture and
// program pointers in payloads point at immutable per-serial snapshots taken
// at record time, so the game rewriting or re-initializing an object later
// never changes a recorded command.

#pragma once

#include "objects.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <memory>
#include <span>

namespace armigo::nvn {

struct RecordHeader {
    std::uint32_t op, size;
};

enum class Op : std::uint32_t {
    end, jump, begin, call,
    // CPU-visible side effects.
    signal_event, report_counter, copy_buffer, update_uniforms,
    set_render_targets, clear_color, clear_depth_stencil, viewports, scissor, depth_range,
    bind_program, bind_state, bind_vertex_attributes, bind_vertex_streams, bind_vertex_buffers,
    bind_uniform_buffer, bind_storage_buffer, bind_textures, bind_samplers,
    stencil, polygon_offset, draw, draw_indexed, dispatch, barrier,
    set_texture_pool, set_sampler_pool, copy_buffer_to_texture, copy_stride, blend_color, copy_texture_to_buffer,
    bind_images, copy_texture_to_texture, reset_counter, clear_texture, depth_clamp, line_width, point_size,
    bind_textures_deferred,
    // Appended operations keep the established record numbers stable.
    unsupported, wait_event, wait_sync, fence_sync, clear_buffer, report_value,
    render_enable, render_enable_conditional, discard_color, discard_depth_stencil, downsample,
    draw_indirect, draw_indexed_indirect, multi_draw_indirect, multi_draw_indexed_indirect,
    viewport_swizzles, scissors, depth_bounds, depth_ranges, bind_transform_feedback_buffer,
    transform_feedback, patch_size, tessellation_levels, primitive_restart, alpha_ref, sample_mask,
    rasterizer_discard, conservative_raster, conservative_raster_dilate, subpixel_precision_bias,
    shader_scratch_memory, zcull_data, copy_image_stride, draw_texture, program_subroutines,
    coverage_modulation_table, resolve_depth_buffer, debug_marker, clear_color_integer, dispatch_indirect,
    sampler_registrations,
    count
};

struct SetRenderTargets {
    std::int32_t count;
    const TextureState* color[8];
    TextureViewState color_views[8];
    std::uint8_t has_color_view[8];
    const TextureState* depth;
    TextureViewState depth_view;
    std::uint8_t has_depth_view;
};
struct ClearColor {
    std::int32_t index, mask;
    float rgba[4];
};
struct ClearDepthStencil {
    float depth;
    std::int32_t depth_mask, stencil, stencil_mask;
};
// x, y, width, height, near, far; near and far are replaced by depth_range.
struct Viewports {
    std::int32_t first, count;
    float values[16][6];
};
struct Scissor {
    std::int32_t x, y, width, height;
};
struct DepthRange {
    float near_value, far_value;
};
struct BindProgram {
    const ProgramState* program;
    std::int32_t stages;
};
// kind: 0 blend, 1 color, 2 channel mask, 3 multisample, 4 polygon, 5 depth/stencil.
struct BindState {
    std::int32_t kind;
    std::uint8_t bytes[0x18];
};
struct BindVertexAttributes {
    std::int32_t first, count;
    std::uint32_t states[16];
};
struct BindVertexStreams {
    std::int32_t first, count;
    VertexStreamStateBits states[16];
};
struct BindVertexBuffers {
    std::int32_t first, count;
    NVNbufferRange ranges[16];
};
// Uniform and storage buffers.
struct BindBuffer {
    std::int32_t stage, index;
    std::uint64_t address, size;
};
// Separate texture or sampler handles.
// Sampler registrations made before the next draw or dispatch was recorded.
struct SamplerRegistrations {
    std::uint64_t value;
};
struct BindHandles {
    std::int32_t stage, first, count;
    // Sampler registrations made before the binding was recorded (0 for a
    // deferred binding, whose draw carries them).
    std::uint64_t sampler_registrations;
    std::uint64_t handles[32];
};
// which: 0 value mask, 1 write mask, 2 reference value.
struct Stencil {
    std::int32_t face, value_mask, mask, reference, which;
};
struct PolygonOffset {
    float factor, units, clamp;
};
struct Draw {
    std::int32_t primitive, first, count, instances, base_instance;
};
struct DrawIndexed {
    std::int32_t primitive, index_type, count, base_vertex, base_instance, instances;
    std::uint64_t index_buffer;
};
// caller: the game's return address at the dispatch call (diagnostics).
struct Dispatch {
    std::int32_t x, y, z;
    std::uint64_t caller;
};
struct SetDescriptorPool {
    const DescriptorPoolState* pool;
};
struct CopyBufferToTexture {
    std::uint64_t source;
    const TextureState* texture;
    TextureViewState view;
    std::uint8_t has_view;
    NVNcopyRegion region;
    std::int32_t flags;
};
// Also used for texture-to-buffer copies: source then holds the destination
// buffer address in place of a texture.
struct CopyTextureToTexture {
    const TextureState* source;
    const TextureState* destination;
    TextureViewState source_view, destination_view;
    std::uint8_t has_source_view, has_destination_view;
    NVNcopyRegion source_region, destination_region;
    std::int32_t flags;
};
struct ClearTexture {
    const TextureState* texture;
    TextureViewState view;
    std::uint8_t has_view, is_integer;
    NVNcopyRegion region;
    std::int32_t mask;
    union {
        float f[4];
        std::uint32_t u[4];
    } color;
};
struct CopyStride {
    std::int64_t stride;
};
struct BlendColor {
    float rgba[4];
};
// Booleans are normalized to 0 or 1 at record time.
struct DepthClamp {
    std::int32_t enable;
};
// Stored as given; Metal draws lines one pixel wide.
struct LineWidth {
    float width;
};
struct PointSize {
    float size;
};
// The caller's handle array is recorded by pointer and never read at record
// time; queue submit copies exactly count handles into a per-submit snapshot
// that replay reads. The full int values are kept so a truncated range fails.
struct BindTexturesDeferred {
    std::int32_t stage, first, count, reserved;
    const std::uint64_t* handles;
};

// Payloads for commands whose records were added after the initial replay ABI.
struct Unsupported {
    std::uint32_t entry;
};
struct SignalEvent {
    volatile std::uint32_t* value;
    std::int32_t mode, location, flags;
    std::uint32_t signal_value;
};
struct WaitEvent {
    const volatile std::uint32_t* value;
    std::int32_t mode;
    std::uint32_t wait_value;
};
struct WaitSync {
    const SyncState* sync;
};
struct FenceSync {
    SyncState* sync;
    std::int32_t condition, flags;
};
struct ReportCounter {
    std::int32_t type;
    std::uint64_t address;
};
struct ResetCounter {
    std::int32_t type;
};
struct CopyBuffer {
    std::uint64_t source, destination, size;
    std::int32_t flags;
};
struct ClearBuffer {
    std::uint64_t address, size;
    std::uint32_t value;
};
struct UpdateUniforms {
    std::uint64_t destination, size;
    std::byte data[1];
};
struct ReportValue {
    std::uint32_t value;
    std::uint64_t address;
};
struct RenderEnable {
    std::int32_t enable;
};
struct RenderEnableConditional {
    std::int32_t mode;
    std::uint64_t address;
};
struct DiscardColor {
    std::int32_t index;
};
struct Downsample {
    const TextureState* source;
    const TextureState* destination;
    TextureViewState source_view, destination_view;
    std::uint8_t has_source_view, has_destination_view, tiled;
};
struct DrawIndirect {
    std::int32_t primitive;
    std::uint64_t indirect_buffer;
};
struct DrawIndexedIndirect {
    std::int32_t primitive, index_type;
    std::uint64_t index_buffer, indirect_buffer;
};
struct MultiDrawIndirect {
    std::int32_t primitive, max_draw_count;
    std::uint64_t indirect_buffer, parameter_buffer;
    std::int64_t stride;
};
struct MultiDrawIndexedIndirect {
    std::int32_t primitive, index_type, max_draw_count;
    std::uint64_t index_buffer, indirect_buffer, parameter_buffer;
    std::int64_t stride;
};
struct ViewportSwizzles {
    std::int32_t first, count;
    std::int32_t values[16][4];
};
struct Scissors {
    std::int32_t first, count;
    std::int32_t values[16][4];
};
struct DepthBounds {
    std::int32_t enable;
    float near_value, far_value;
};
struct DepthRanges {
    std::int32_t first, count;
    float values[16][2];
};
struct BindTransformFeedbackBuffer {
    std::int32_t index;
    std::uint64_t address, size;
};
struct TransformFeedback {
    std::int32_t action, argument;
    std::uint64_t address;
};
struct PatchSize {
    std::int32_t size;
};
struct TessellationLevels {
    std::uint8_t outer;
    float values[4];
};
struct PrimitiveRestart {
    std::uint32_t index;
    std::int32_t enable;
};
struct ScalarFloat {
    float value;
};
struct ScalarInt {
    std::int32_t value;
};
struct ConservativeRaster {
    std::int32_t enable;
};
struct SubpixelPrecisionBias {
    std::int32_t x_bits, y_bits;
};
struct ShaderScratchMemory {
    const MemoryPoolState* pool;
    std::int64_t offset;
    std::uint64_t size;
};
struct ZCullData {
    std::uint64_t address, size;
    std::uint8_t restore;
};
struct DrawTexture {
    std::uint64_t texture_handle;
    NVNdrawTextureRegion destination, source;
};
struct ProgramSubroutines {
    const ProgramState* program;
    std::int32_t stage, first, count;
    std::int32_t values[1];
};
struct CoverageModulationTable {
    float values[16];
};
struct DebugMarker {
    std::uint32_t domain_id;
    std::uint8_t action;
    char text[1];
};
struct ClearColorInteger {
    std::int32_t index, mask;
    std::uint8_t is_unsigned;
    std::uint32_t rgba[4];
};
struct DispatchIndirect {
    std::uint64_t address;
};

// A view is valid only while its visitor is running. Flexible payloads report
// their complete byte count so consumers can validate array lengths.
struct CommandRecordView {
    Op op;
    const void* payload;
    std::size_t payload_size;
};

using CommandVisitor = bool (*)(const CommandRecordView&, void* context);

enum class CommandWalkStatus : std::uint8_t {
    complete,
    stopped,
    invalid_handle,
    malformed_record,
    call_cycle,
    call_depth_exceeded,
    invalid_deferred_binding,
};

// Expands CallCommands in record order. Deferred texture handle arrays are
// copied at walk time and presented as ordinary bind_textures records.
[[nodiscard]] CommandWalkStatus walk_commands(NVNcommandHandle handle, CommandVisitor visitor, void* context) noexcept;
[[nodiscard]] std::string_view command_walk_status_name(CommandWalkStatus status) noexcept;

// A submission owns its complete call graph until fence retirement. Only deferred
// handle arrays and mutable object descriptions are captured on the submitter;
// ordinary command payloads are visited directly in their recording storage.
struct CommandSubmission;
[[nodiscard]] std::shared_ptr<const CommandSubmission> prepare_commands(
    std::span<const NVNcommandHandle> handles, CommandWalkStatus& status);
[[nodiscard]] CommandWalkStatus walk_commands(
    const CommandSubmission& submission, CommandVisitor visitor, void* context) noexcept;
void fence_commands(const CommandSubmission& submission, std::uint64_t fence) noexcept;

void register_state_entries() noexcept;
void register_program_entries() noexcept;
void register_command_entries() noexcept;

} // namespace armigo::nvn
