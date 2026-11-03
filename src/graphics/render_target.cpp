// ============================================================================
// Armigo Graphics - GPU Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Render-target planning decodes Maxwell 3D attachment registers, clear target,
// components, layers, aspects, scissor state, layout, formats, MSAA sample count,
// texture shape, and programmed array-layer spacing. Guest clear macros take
// the color layer count from 0x206 + target * 0x10 words and the depth layer
// count from 0x48c words, clearing the programmed array from the selected layer.
// Missing storage omits the attachment and its clear work. Framebuffer dimensions
// are clipped to bound attachments with an explicit attachment-free fallback.
// SET_ZT_SELECT at byte offset 0x1538 exposes TARGET_COUNT in bit zero;
// CLEAR_SURFACE at 0x19d0 independently selects depth and stencil aspects.

#include <armigo/graphics/render_target.h>

#include <armigo/graphics/color_clear.h>
#include <armigo/graphics/texture_format.h>
#include <armigo/graphics/texture_layout.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>

namespace armigo::graphics {
namespace {

// Maxwell 3D method offsets below follow the sequential guest register layout
// (the register file is uint words, so method = byte/4).
// They agree with the established Armigo frontend and planners, which program
// and consume the same offsets (color targets at 0x200 + 0x10*i, depth state
// at 0x3f8, screen scissor at 0x3fd, clear methods from color_clear.cpp).
constexpr std::size_t raster_enable_method = 0x0df;
constexpr std::size_t clear_colors_method = 0x360;  // four float words
constexpr std::size_t clear_depth_method = 0x364;
constexpr std::size_t clear_stencil_method = 0x368;
constexpr std::size_t scissor_base_method = 0x380;  // enable, x1|x2, y1|y2
constexpr std::size_t screen_scissor_method = 0x3fd;  // {x, width}, {y, height}
constexpr std::size_t clear_control_method = 0x43e;
constexpr std::size_t color_target_base_method = 0x200;  // 0x10 words each
constexpr std::size_t depth_state_method = 0x3f8;  // address, format, layout, layer size
constexpr std::size_t stencil_test_method = 0x4e0;  // 8 words, front mask at +7
constexpr std::size_t rt_control_method = 0x487;
constexpr std::size_t depth_size_method = 0x48a;  // width, height, depth|flags
constexpr std::size_t render_enable_override_method = 0x651;
constexpr std::size_t clear_surface_method = 0x674;
constexpr std::size_t msaa_mode_method = 0x574;
constexpr std::size_t depth_enable_method = 0x54e;

constexpr std::size_t total_targets = 8;

// MemoryLayout packed fields.
constexpr std::uint32_t layout_block_height_log2_mask = 0xf0;
constexpr std::uint32_t layout_block_depth_log2_shift = 8;
constexpr std::uint32_t layout_linear_bit = 1u << 12;
constexpr std::uint32_t layout_target_3d_bit = 1u << 16;
constexpr std::uint32_t layout_known_bits = 0x11fff;

// RtDepthStencilSize: the depth word packs the
// layer count and the layered flag; DepthDescription carries neither, so both
// are read from the immutable register image.
constexpr std::uint32_t depth_size_layered_flag_bit = 1u << 16;

// ClearFlags: bit 0 makes the clear respect the stencil
// test mask, bit 8 makes it respect the scissor. The remaining established
// profile bits (clear rectangle, viewport clip) stay explicit rejections.
constexpr std::uint32_t clear_control_stencil_mask_bit = 1u << 0;
constexpr std::uint32_t clear_control_scissor_bit = 1u << 8;
constexpr std::uint32_t known_clear_control = 0x1111;

constexpr std::uint32_t low_half(std::uint32_t value) noexcept
{
    return value & 0xffff;
}

constexpr std::uint32_t high_half(std::uint32_t value) noexcept
{
    return value >> 16;
}

constexpr bool checked_add(std::size_t left, std::size_t right,
                           std::size_t& sum) noexcept
{
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    sum = left + right;
    return true;
}

constexpr bool checked_multiply(std::size_t left, std::size_t right,
                                std::size_t& product) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

constexpr bool align_up(std::size_t value, std::size_t alignment,
                        std::size_t& aligned) noexcept
{
    if (alignment == 0) {
        return false;
    }
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        aligned = value;
        return true;
    }
    return checked_add(value, alignment - remainder, aligned);
}

RendererResult<RenderTargetPlan> plan_failure(const RenderStop& stop,
                                              std::size_t method,
                                              std::string_view detail) noexcept
{
    return RendererError{CommandResult{CommandStatus::Unsupported,
                                       detail,
                                       stop.original.address,
                                       0,
                                       0,
                                       static_cast<std::uint32_t>(method),
                                       stop.registers[method],
                                       engine_class::threed}};
}

RendererResult<ClearPlan> clear_plan_failure(
    const RenderStop& stop, std::string_view detail,
    std::size_t method = clear_surface_method) noexcept
{
    return RendererError{CommandResult{CommandStatus::Unsupported,
                                       detail,
                                       stop.original.address,
                                       0,
                                       0,
                                       static_cast<std::uint32_t>(method),
                                       stop.registers[method],
                                       engine_class::threed}};
}

// Render-target register format encodings mapped onto
// the shared decoded texture format table. X-padding variants map onto their
// RGBA counterpart; encodings without a table entry (the BGR orderings) stay
// structured rejections because Armigo's table has no channel-reordered entry.
bool decode_color_target_format(std::uint32_t encoded, TextureFormat& out) noexcept
{
    switch (encoded) {
        case 0xc0: out = TextureFormat::R32G32B32A32Float; return true;
        case 0xc1: out = TextureFormat::R32G32B32A32Sint; return true;
        case 0xc2: out = TextureFormat::R32G32B32A32Uint; return true;
        case 0xc3: out = TextureFormat::R32G32B32A32Float; return true;
        case 0xc4: out = TextureFormat::R32G32B32A32Sint; return true;
        case 0xc5: out = TextureFormat::R32G32B32A32Uint; return true;
        case 0xc6: out = TextureFormat::R16G16B16A16Unorm; return true;
        case 0xc7: out = TextureFormat::R16G16B16A16Snorm; return true;
        case 0xc8: out = TextureFormat::R16G16B16A16Sint; return true;
        case 0xc9: out = TextureFormat::R16G16B16A16Uint; return true;
        case 0xca: out = TextureFormat::R16G16B16A16Float; return true;
        case 0xcb: out = TextureFormat::R32G32Float; return true;
        case 0xcc: out = TextureFormat::R32G32Sint; return true;
        case 0xcd: out = TextureFormat::R32G32Uint; return true;
        case 0xce: out = TextureFormat::R16G16B16A16Float; return true;
        case 0xd1: out = TextureFormat::R10G10B10A2Unorm; return true;
        case 0xd2: out = TextureFormat::R10G10B10A2Uint; return true;
        case 0xd5: out = TextureFormat::R8G8B8A8Unorm; return true;
        case 0xd6: out = TextureFormat::R8G8B8A8Srgb; return true;
        case 0xd7: out = TextureFormat::R8G8B8A8Snorm; return true;
        case 0xd8: out = TextureFormat::R8G8B8A8Sint; return true;
        case 0xd9: out = TextureFormat::R8G8B8A8Uint; return true;
        case 0xda: out = TextureFormat::R16G16Unorm; return true;
        case 0xdb: out = TextureFormat::R16G16Snorm; return true;
        case 0xdc: out = TextureFormat::R16G16Sint; return true;
        case 0xdd: out = TextureFormat::R16G16Uint; return true;
        case 0xde: out = TextureFormat::R16G16Float; return true;
        case 0xe0: out = TextureFormat::R11G11B10Float; return true;
        case 0xe3: out = TextureFormat::R32Sint; return true;
        case 0xe4: out = TextureFormat::R32Uint; return true;
        case 0xe5: out = TextureFormat::R32Float; return true;
        case 0xe8: out = TextureFormat::R5G6B5Unorm; return true;
        case 0xea: out = TextureFormat::R8G8Unorm; return true;
        case 0xeb: out = TextureFormat::R8G8Snorm; return true;
        case 0xec: out = TextureFormat::R8G8Sint; return true;
        case 0xed: out = TextureFormat::R8G8Uint; return true;
        case 0xee: out = TextureFormat::R16Unorm; return true;
        case 0xef: out = TextureFormat::R16Snorm; return true;
        case 0xf0: out = TextureFormat::R16Sint; return true;
        case 0xf1: out = TextureFormat::R16Uint; return true;
        case 0xf2: out = TextureFormat::R16Float; return true;
        case 0xf3: out = TextureFormat::R8Unorm; return true;
        case 0xf4: out = TextureFormat::R8Snorm; return true;
        case 0xf5: out = TextureFormat::R8Sint; return true;
        case 0xf6: out = TextureFormat::R8Uint; return true;
        case 0xf9: out = TextureFormat::R8G8B8A8Unorm; return true;
        case 0xfa: out = TextureFormat::R8G8B8A8Srgb; return true;
        default: return false;
    }
}

constexpr bool color_target_has_alpha(std::uint32_t encoded) noexcept
{
    switch (encoded) {
        case 0xc3:
        case 0xc4:
        case 0xc5:
        case 0xce:
        case 0xf9:
        case 0xfa:
            return false;
        default:
            return true;
    }
}

// Depth-stencil register format encodings. The
// stencil-only encoding has no entry in Armigo's format table and stays a
// structured rejection.
bool decode_depth_target_format(std::uint32_t encoded, TextureFormat& out) noexcept
{
    switch (encoded) {
        case 0x0a: out = TextureFormat::D32Float; return true;
        case 0x13: out = TextureFormat::D16Unorm; return true;
        case 0x14: out = TextureFormat::D24UnormS8Uint; return true;
        case 0x15: out = TextureFormat::X8UintD24Unorm; return true;
        case 0x16: out = TextureFormat::S8UintD24Unorm; return true;
        case 0x19: out = TextureFormat::D32FloatS8Uint; return true;
        default: return false;
    }
}

// TextureMsaaMode decode. Unknown encodings are
// reserved; the established v1 profile admits single-sample passes only.
bool decode_msaa_mode(std::uint32_t encoded, std::uint32_t& samples) noexcept
{
    switch (encoded) {
        case 0: samples = 1; return true;
        case 2: samples = 4; return true;
        case 4: samples = 8; return true;
        case 5: samples = 2; return true;
        case 6: samples = 16; return true;
        default: return false;
    }
}

struct TargetGeometry {
    TextureLayout layout = TextureLayout::BlockLinear;
    std::uint32_t width = 0, height = 0;
    std::size_t pitch_bytes = 0;
    std::uint8_t block_height_log2 = 0, block_depth_log2 = 0;
    bool target_3d = false;
};

struct GeometryError {
    std::size_t register_offset = 5;
    std::string_view detail;
};

// Width is a byte pitch for linear targets and a texel count otherwise.
// A one-slice 3D target retains its encoded Z interleaving: even with depth
// one, successive X/Y GOBs can be separated by unused depth planes.
std::optional<GeometryError> decode_color_target_geometry(
    const TargetDescription& target, const TextureFormatInfo& info,
    TargetGeometry& out) noexcept
{
    const std::uint32_t tile_mode = target.tile_mode;
    if ((tile_mode & ~layout_known_bits) != 0) {
        return GeometryError{5, "render target memory layout has reserved bits"};
    }
    if ((tile_mode & 0xf) != 0) {
        return GeometryError{5, "render target GOB block width is unsupported"};
    }
    out.block_height_log2 =
        static_cast<std::uint8_t>((tile_mode & layout_block_height_log2_mask) >> 4);
    out.block_depth_log2 =
        static_cast<std::uint8_t>((tile_mode >> layout_block_depth_log2_shift) & 0xf);
    out.target_3d = (tile_mode & layout_target_3d_bit) != 0;
    if (out.block_height_log2 > 5) {
        return GeometryError{5, "render target GOB block height exceeds 32 GOBs"};
    }
    if (out.block_depth_log2 > 5) {
        return GeometryError{5, "render target GOB block depth exceeds 32 GOBs"};
    }
    if (out.target_3d && target.depth != 1) {
        return GeometryError{6, "render target 3D depth must be one slice"};
    }
    if (!out.target_3d && out.block_depth_log2 != 0) {
        return GeometryError{5, "render target GOB block depth requires a 3D target"};
    }
    if ((tile_mode & layout_linear_bit) != 0) {
        if (out.target_3d) {
            return GeometryError{5, "pitch-linear 3D render target is unsupported"};
        }
        if (out.block_height_log2 != 0 || out.block_depth_log2 != 0) {
            return GeometryError{5, "pitch-linear render target has nonzero GOB geometry"};
        }
        if (target.width % info.bytes_per_block != 0) {
            return GeometryError{2, "render target byte pitch is not a whole texel"};
        }
        out.layout = TextureLayout::Pitch;
        out.pitch_bytes = target.width;
        out.width = target.width / info.bytes_per_block;
    } else {
        out.layout = TextureLayout::BlockLinear;
        out.width = target.width;
    }
    out.height = target.height;
    if (out.width == 0) {
        return GeometryError{2, "render target width or pitch is empty"};
    }
    if (out.height == 0) {
        return GeometryError{3, "render target height is empty"};
    }
    return std::nullopt;
}

std::optional<GeometryError> decode_depth_target_geometry(
    std::uint32_t tile_mode, std::uint32_t width, std::uint32_t height,
    const TextureFormatInfo& info, TargetGeometry& out) noexcept
{
    if ((tile_mode & layout_linear_bit) != 0) {
        return GeometryError{5, "pitch-linear depth target is unsupported"};
    }
    if ((tile_mode & layout_target_3d_bit) != 0) {
        return GeometryError{5, "3D depth target is unsupported"};
    }
    TargetDescription target{};
    target.tile_mode = tile_mode;
    target.width = width;
    target.height = height;
    target.depth = 1;
    return decode_color_target_geometry(target, info, out);
}

// Builds the storage shape of a single-level attachment view with
// calculate_texture_size: the descriptor's backing extent, GOB-aligned row
// stride and storage row count come from the shared layout module.
bool build_attachment_descriptor(std::uint64_t address,
                                 const TargetGeometry& geometry,
                                 const TextureFormatInfo& info,
                                 std::uint32_t layers,
                                 std::uint32_t array_pitch,
                                 std::array<std::uint32_t, 8> raw_words,
                                 TextureDescriptor& out) noexcept
{
    TextureLayoutDescription description{};
    description.width = geometry.width;
    description.height = geometry.height;
    description.depth = 1;
    description.levels = 1;
    description.layers = layers;
    description.block_width = info.block_width;
    description.block_height = info.block_height;
    description.bytes_per_block = info.bytes_per_block;
    description.gob_block_height_log2 = geometry.block_height_log2;
    description.gob_block_depth_log2 = geometry.block_depth_log2;
    description.storage = geometry.layout == TextureLayout::Pitch
                              ? TextureStorageLayout::Pitch
                              : TextureStorageLayout::BlockLinear;
    description.pitch_bytes = geometry.pitch_bytes;
    const auto calculated = calculate_texture_size(std::move(description));
    if (const auto* layout_error = std::get_if<TextureLayoutError>(&calculated)) {
        static_cast<void>(layout_error);
        return false;
    }
    const TextureLayoutDescription& layout =
        std::get<TextureLayoutDescription>(calculated);
    const TextureLevelLayout& base = layout.mips[0];

    std::size_t row_bytes = 0;
    std::size_t storage_height = 0;
    if (geometry.layout == TextureLayout::Pitch) {
        row_bytes = geometry.pitch_bytes;
        storage_height = base.block_rows;
    } else {
        std::size_t logical_row = 0;
        if (!checked_multiply(std::size_t{base.block_columns},
                              info.bytes_per_block, logical_row) ||
            !align_up(logical_row, texture_gob_width_bytes, row_bytes) ||
            !align_up(std::size_t{base.block_rows},
                      texture_gob_height_rows << base.gob_block_height_log2,
                      storage_height)) {
            return false;
        }
    }
    std::size_t total_size = layout.total_size;
    std::size_t layer_stride = layout.layer_stride;
    // Maxwell programs block-linear layer spacing in words. Pitch targets
    // derive their layer spacing from pitch times height instead.
    if (layers > 1 && geometry.layout == TextureLayout::BlockLinear) {
        if (!checked_multiply(std::size_t{array_pitch}, 4, layer_stride) ||
            !checked_multiply(std::max(layer_stride, layout.layer_stride),
                              layers - 1, total_size) ||
            !checked_add(total_size, base.size, total_size)) {
            return false;
        }
        layer_stride = std::max(layer_stride, layout.layer_stride);
    }
    if (total_size == 0 || total_size > max_texture_backing_size) {
        return false;
    }
    std::size_t end = 0;
    if (!checked_add(std::size_t(address), total_size, end)) {
        return false;
    }

    out.raw_words = raw_words;
    out.address = address;
    out.backing_extent = total_size;
    out.row_bytes = row_bytes;
    out.storage_height = storage_height;
    out.layer_stride = layer_stride;
    out.width = geometry.width;
    out.height = geometry.height;
    out.depth = 1;
    out.levels = 1;
    out.layers = layers;
    out.base_level = 0;
    out.samples = 1;
    out.block_height_log2 = geometry.block_height_log2;
    out.block_depth_log2 = geometry.block_depth_log2;
    out.layout = geometry.layout;
    out.type = geometry.target_3d ? TextureType::Texture3D
                               : (layers > 1 ? TextureType::Texture2DArray
                                             : TextureType::Texture2D);
    return true;
}

// Reads the effective render-enable state. Only an unconditional enable is
// admitted; conditional rendering stops the planner (no guessed evaluation).
bool render_unconditionally(const RenderStop& stop) noexcept
{
    const std::uint32_t override_value = stop.registers[render_enable_override_method];
    return override_value == 1 ||
           (override_value == 0 && stop.effective_render_enable_mode == 1);
}

struct ClearRegion {
    std::uint32_t x = 0, y = 0, width = 0, height = 0;
    bool full = true;
};

// Derives the clear region from the clipped surface extent and the scissor
// state: the clear covers the
// screen scissor, intersected with scissor 0 when the clear honours it, and
// the bound target clips both. An enabled scissor intersects inclusively at
// the minimum corner and exclusively at the maximum corner. An empty
// intersection clears nothing.
bool decode_clear_region(const RenderStop& stop,
                         std::uint32_t surface_width,
                         std::uint32_t surface_height,
                         bool scissored,
                         ClearRegion& out) noexcept
{
    out = {0, 0, surface_width, surface_height, true};
    if (!scissored) {
        return true;
    }
    if (stop.registers[scissor_base_method] != 1) {
        return false;
    }
    const std::uint32_t horizontal = stop.registers[scissor_base_method + 1];
    const std::uint32_t vertical = stop.registers[scissor_base_method + 2];
    const std::uint32_t left = std::min(low_half(horizontal), surface_width);
    const std::uint32_t top = std::min(low_half(vertical), surface_height);
    const std::uint32_t right = std::min(high_half(horizontal), surface_width);
    const std::uint32_t bottom = std::min(high_half(vertical), surface_height);
    out = {left, top, right > left ? right - left : 0u, bottom > top ? bottom - top : 0u, false};
    return true;
}

// The surface extent of a clear comes from the screen scissor and clears
// exactly the programmed surface, so the origin stays zero and the
// extent nonempty in the admitted profile.
bool decode_surface_extent(const RenderStop& stop,
                           std::uint32_t& width,
                           std::uint32_t& height) noexcept
{
    const std::uint32_t horizontal = stop.registers[screen_scissor_method];
    const std::uint32_t vertical = stop.registers[screen_scissor_method + 1];
    if (low_half(horizontal) != 0 || low_half(vertical) != 0) {
        return false;
    }
    width = high_half(horizontal);
    height = high_half(vertical);
    return width != 0 && height != 0;
}

}  // namespace

RendererResult<RenderTargetPlan> plan_render_targets(const RenderStop& stop)
{
    if (stop.engine != engine_class::threed ||
        (stop.kind != RenderKind::Draw && stop.kind != RenderKind::Clear)) {
        return plan_failure(stop, 0, "render targets plan requires a 3D render stop");
    }

    std::uint32_t samples = 0;
    if (!decode_msaa_mode(stop.registers[msaa_mode_method], samples)) {
        return plan_failure(stop, msaa_mode_method,
                            "render target MSAA mode is a reserved encoding");
    }
    if (samples != 1) {
        return plan_failure(stop, msaa_mode_method,
                            "multisampled render targets are unsupported");
    }

    // RtControl: the low nibble names the active output count and three-bit
    // fields map each logical output onto its physical target. Slots beyond
    // the count and targets with a zero format or stride remain holes; the
    // permutation is never compacted.
    const std::uint32_t rt_control = stop.registers[rt_control_method];
    const std::uint32_t count = rt_control & 0xf;
    if (count > total_targets) {
        return plan_failure(stop, rt_control_method,
                            "render target count exceeds eight outputs");
    }

    RenderTargetPlan plan{};
    plan.samples = samples;
    bool any_attachment = false;
    std::uint64_t clip_width = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t clip_height = std::numeric_limits<std::uint64_t>::max();

    for (std::uint32_t logical = 0; logical < total_targets; ++logical) {
        if (logical >= count) {
            continue;
        }
        const std::uint32_t physical = (rt_control >> (4 + logical * 3)) & 7;
        const TargetDescription& target = stop.color_targets[physical];
        // Colors are disabled by a zero format or a zero width/stride register.
        // A null address is a binding without storage and stays a hole too.
        if (target.format == 0 || target.width == 0 || target.address == 0) {
            continue;
        }
        TextureFormat format = TextureFormat::R8G8B8A8Unorm;
        if (!decode_color_target_format(target.format, format)) {
            return plan_failure(stop,
                                color_target_base_method + physical * 0x10 + 4,
                                "color target format has no decoded table entry");
        }
        const TextureFormatInfo info = texture_format_info(format);
        TargetGeometry geometry{};
        if (const auto error = decode_color_target_geometry(target, info, geometry)) {
            return plan_failure(stop, color_target_base_method + physical * 0x10 +
                                     error->register_offset, error->detail);
        }
        const std::uint32_t layers = target.depth;
        if ((stop.registers[color_target_base_method + physical * 0x10 + 6] &
             0xffff0000u) != 0) {
            return plan_failure(stop, color_target_base_method + physical * 0x10 + 6,
                                "color target array mode has unsupported high bits");
        }
        if (layers == 0) {
            return plan_failure(stop,
                                color_target_base_method + physical * 0x10 + 6,
                                "color target names zero array layers");
        }
        if (target.base_layer >= layers) {
            return plan_failure(stop,
                                color_target_base_method + physical * 0x10 + 8,
                                "color target base layer exceeds its layer count");
        }
        AttachmentPlan attachment{};
        attachment.physical_target = static_cast<std::uint8_t>(physical);
        const std::size_t base = color_target_base_method + physical * 0x10;
        std::array<std::uint32_t, 8> raw{};
        for (std::size_t word = 0; word < raw.size(); ++word) {
            raw[word] = stop.registers[base + word];
        }
        if (!build_attachment_descriptor(target.address, geometry, info, layers,
                                         target.array_pitch, raw, attachment.texture)) {
            return plan_failure(stop,
                                color_target_base_method + physical * 0x10 + (layers > 1 ? 7 : 5),
                                "color target storage extent or layer stride is invalid");
        }
        attachment.texture.format = format;
        attachment.texture.srgb = format == TextureFormat::R8G8B8A8Srgb;
        attachment.aspect = TextureAspect::Color;
        attachment.has_alpha = color_target_has_alpha(target.format);
        attachment.level = 0;
        attachment.layer = target.base_layer;
        attachment.layer_count = 1;
        attachment.load = AttachmentLoad::Load;
        attachment.store = AttachmentStore::Store;
        plan.colors[logical] = attachment;
        any_attachment = true;
        clip_width = std::min<std::uint64_t>(clip_width, geometry.width);
        clip_height = std::min<std::uint64_t>(clip_height, geometry.height);
    }

    // Like a null color target, an enabled depth target without GPU storage is
    // absent. Stale format and geometry registers must not make it invalid.
    if (stop.depth.enabled && stop.depth.address != 0) {
        const std::uint32_t depth_width = stop.registers[depth_size_method];
        const std::uint32_t depth_height = stop.registers[depth_size_method + 1];
        const std::uint32_t depth_word = stop.registers[depth_size_method + 2];
        if (depth_width == 0 || depth_height == 0) {
            return plan_failure(stop, depth_size_method + (depth_width == 0 ? 0 : 1),
                                "depth target extent is empty");
        }
        TextureFormat format = TextureFormat::D32Float;
        if (!decode_depth_target_format(stop.depth.format, format)) {
            return plan_failure(stop, depth_state_method + 2,
                                "depth target format has no decoded table entry");
        }
        const TextureFormatInfo info = texture_format_info(format);
        if ((depth_word & 0xfffe0000u) != 0) {
            return plan_failure(stop, depth_size_method + 2,
                                "depth target array mode has unsupported flags");
        }
        // Bit 16 disables array interpretation of the low sixteen bits.
        const bool layered = (depth_word & depth_size_layered_flag_bit) == 0;
        const std::uint32_t layers = layered ? low_half(depth_word) : 1;
        if (layers == 0) {
            return plan_failure(stop, depth_size_method + 2,
                                "depth target names zero array layers");
        }
        TargetGeometry geometry{};
        if (const auto error = decode_depth_target_geometry(
                stop.depth.tile_mode, depth_width, depth_height, info, geometry)) {
            return plan_failure(stop, depth_state_method + 3, error->detail);
        }
        const std::size_t base = depth_state_method;
        std::array<std::uint32_t, 8> raw{};
        for (std::size_t word = 0; word < 5; ++word) {
            raw[word] = stop.registers[base + word];
        }
        raw[5] = depth_word;
        AttachmentPlan depth_attachment{};
        depth_attachment.physical_target = 0;
        if (!build_attachment_descriptor(stop.depth.address, geometry, info,
                                         layers, stop.depth.array_pitch, raw, depth_attachment.texture)) {
            return plan_failure(stop, depth_state_method + (layers > 1 ? 4 : 3),
                                "depth target storage extent or layer stride is invalid");
        }
        depth_attachment.texture.format = format;
        depth_attachment.level = 0;
        depth_attachment.layer = 0;
        depth_attachment.layer_count = 1;
        depth_attachment.load = AttachmentLoad::Load;
        depth_attachment.store = AttachmentStore::Store;
        if (info.depth) {
            depth_attachment.aspect = TextureAspect::Depth;
            plan.depth = depth_attachment;
        }
        if (info.stencil) {
            depth_attachment.aspect = TextureAspect::Stencil;
            plan.stencil = depth_attachment;
        }
        any_attachment = true;
        clip_width = std::min<std::uint64_t>(clip_width, geometry.width);
        clip_height = std::min<std::uint64_t>(clip_height, geometry.height);
    }

    if (!any_attachment) {
        std::uint32_t surface_width = 0;
        std::uint32_t surface_height = 0;
        if (!decode_surface_extent(stop, surface_width, surface_height)) {
            return plan_failure(stop, screen_scissor_method,
                                "attachment-less render pass has an invalid surface extent");
        }
        clip_width = surface_width;
        clip_height = surface_height;
    }
    plan.area = RenderArea{0, 0, clip_width, clip_height};
    return plan;
}

RendererResult<ClearPlan> plan_render_clear(const RenderStop& stop)
{
    if (stop.kind != RenderKind::Clear || stop.engine != engine_class::threed) {
        return clear_plan_failure(stop, "render stop is not a 3D clear");
    }

    const std::uint32_t clear = stop.registers[clear_surface_method];
    if ((clear & 0xfc000000U) != 0) {
        return clear_plan_failure(stop, "clear has unsupported argument bits");
    }
    const bool clear_depth = (clear & 1U) != 0;
    const bool clear_stencil = (clear & 2U) != 0;
    std::uint32_t component_mask = (clear >> 2) & 0xf;
    const std::uint32_t target_index = (clear >> 6) & 0xf;
    // CLEAR_SURFACE bits 10-25 name the first cleared layer. Layer zero clears
    // the target's whole programmed array because the
    // guest SDK clear macros read the layer count from the array-mode
    // registers. Layer-zero clears use the layered pass load action, while a
    // nonzero layer names exactly one layer.
    const std::uint32_t layer_field = (clear >> 10) & 0xffff;
    const auto layer_range = [](std::uint32_t field,
                                std::uint32_t layer_count) {
        return std::pair<std::uint32_t, std::uint32_t>{
            field, field == 0 ? layer_count : 1u};
    };
    if (component_mask == 0 && !clear_depth && !clear_stencil) {
        return clear_plan_failure(stop, "clear names no color or depth aspect");
    }
    if (target_index >= stop.color_targets.size()) {
        return clear_plan_failure(stop, "clear render-target index is out of range");
    }

    if (!render_unconditionally(stop)) {
        return clear_plan_failure(stop, "conditional or disabled clear is unsupported");
    }
    if (stop.registers[raster_enable_method] != 1) {
        return clear_plan_failure(stop, "clear with rasterization disabled is unsupported");
    }

    const std::uint32_t clear_control = stop.registers[clear_control_method];
    if ((clear_control & ~known_clear_control) != 0) {
        return clear_plan_failure(stop, "clear control has unsupported bits");
    }
    if ((clear_control & 0x10) != 0) {
        return clear_plan_failure(stop, "clear-rectangle clear is unsupported");
    }
    if ((clear_control & 0x1000) != 0) {
        return clear_plan_failure(stop, "viewport-clipped clear is unsupported");
    }
    std::uint32_t samples = 0;
    if (!decode_msaa_mode(stop.registers[msaa_mode_method], samples) || samples != 1) {
        return clear_plan_failure(stop, "multisampled clear is unsupported");
    }

    std::uint32_t surface_width = 0;
    std::uint32_t surface_height = 0;
    if (!decode_surface_extent(stop, surface_width, surface_height)) {
        return clear_plan_failure(stop, "screen scissor does not describe a full surface");
    }
    const bool scissored = (clear_control & clear_control_scissor_bit) != 0 &&
                           stop.registers[scissor_base_method] != 0;

    ClearPlan plan{};
    bool colour_admitted = false;
    bool depth_admitted = false;
    AttachmentPlan colour_attachment{};
    AttachmentPlan depth_attachment{};
    ClearRegion colour_region{};
    ClearRegion depth_region{};
    if (component_mask != 0) {
        const TargetDescription& target = stop.color_targets[target_index];
        if (target.format == 0 || target.width == 0) {
            return clear_plan_failure(stop, "clear target is disabled");
        }
        // A null colour binding has no attachment to clear, even when its
        // format and geometry registers retain a previous target's values.
        // Armigo still clears the depth/stencil aspects of such a
        // combined request, so the colour half drops out instead of failing.
        if (target.address == 0) {
            component_mask = 0;
        } else {
            TextureFormat format = TextureFormat::R8G8B8A8Unorm;
            if (!decode_color_target_format(target.format, format)) {
                return clear_plan_failure(stop,
                                          "clear target format has no decoded table entry");
            }
            const TextureFormatInfo info = texture_format_info(format);
            TargetGeometry geometry{};
            if (const auto error = decode_color_target_geometry(target, info, geometry)) {
                return clear_plan_failure(stop, error->detail,
                    color_target_base_method + target_index * 0x10 + error->register_offset);
            }
            if ((stop.registers[color_target_base_method + target_index * 0x10 + 6] &
                 0xffff0000u) != 0) {
                return clear_plan_failure(stop,
                    "color target array mode has unsupported high bits",
                    color_target_base_method + target_index * 0x10 + 6);
            }
            // The array-mode register (MacroHLE ColorLayerCountOffset) carries
            // the cleared layer count. Multi-slice 3D colour targets stay
            // unsupported; the geometry decoder admits one-slice 3D storage
            // only, which keeps Texture3D attachments single-slice.
            const std::uint32_t layers = target.depth;
            if (layers == 0) {
                return clear_plan_failure(stop, "clear target names zero array layers",
                    color_target_base_method + target_index * 0x10 + 6);
            }
            // The cleared range starts at the target's base layer plus the
            // argument's layer field: the whole array for layer zero, one
            // layer otherwise.
            const auto [colour_first, colour_count] = layer_range(layer_field, layers);
            const std::uint64_t first_layer =
                std::uint64_t{target.base_layer} + colour_first;
            if (first_layer + colour_count > target.depth) {
                return clear_plan_failure(stop, "clear layer range exceeds the color target",
                    color_target_base_method + target_index * 0x10 + 8);
            }
            // The programmed surface defines the clear extent and the bound
            // target clips it to the host framebuffer.
            const std::uint32_t clip_width = std::min(surface_width, geometry.width);
            const std::uint32_t clip_height = std::min(surface_height, target.height);
            ClearRegion region{};
            if (!decode_clear_region(stop, clip_width, clip_height, scissored, region)) {
                return clear_plan_failure(stop, "clear scissor state is malformed");
            }
            if (region.width == 0 || region.height == 0) {
                component_mask = 0;
            } else {
                for (std::size_t word = 0; word < plan.color_bits.size(); ++word) {
                    plan.color_bits[word] = stop.registers[clear_colors_method + word];
                    if (texture_numeric_class(format) == TextureNumericClass::Float &&
                        !std::isfinite(std::bit_cast<float>(plan.color_bits[word]))) {
                        return clear_plan_failure(stop, "clear color value is not finite");
                    }
                }
                if (!build_attachment_descriptor(target.address, geometry, info, layers,
                                                 target.array_pitch, {},
                                                 colour_attachment.texture)) {
                    return clear_plan_failure(stop, "clear target storage layout is invalid");
                }
                colour_attachment.texture.format = format;
                colour_attachment.texture.srgb = format == TextureFormat::R8G8B8A8Srgb;
                colour_attachment.physical_target = static_cast<std::uint8_t>(target_index);
                colour_attachment.aspect = TextureAspect::Color;
                colour_attachment.has_alpha = color_target_has_alpha(target.format);
                colour_attachment.level = 0;
                colour_attachment.layer = static_cast<std::uint32_t>(first_layer);
                colour_attachment.layer_count = colour_count;
                colour_region = region;
                colour_admitted = true;
            }
        }
    }

    if (clear_depth || clear_stencil) {
        // SET_ZT_SELECT with TARGET_COUNT zero selects no zeta attachment. A
        // null binding is likewise absent even if stale format and geometry
        // remain; a combined clear still performs its colour half.
        if (!stop.depth.enabled || stop.depth.address == 0) {
            plan.clear_depth = plan.clear_stencil = false;
        } else {
            TextureFormat format = TextureFormat::D32Float;
            if (!decode_depth_target_format(stop.depth.format, format)) {
                return clear_plan_failure(stop,
                                          "depth clear format has no decoded table entry");
            }
            const TextureFormatInfo info = texture_format_info(format);
            if (clear_depth && !info.depth) {
                return clear_plan_failure(stop,
                                          "depth clear names an aspect the format lacks");
            }
            if (clear_stencil && !info.stencil) {
                return clear_plan_failure(stop,
                                          "stencil clear names an aspect the format lacks");
            }
            const std::uint32_t depth_width = stop.registers[depth_size_method];
            const std::uint32_t depth_height = stop.registers[depth_size_method + 1];
            const std::uint32_t depth_word = stop.registers[depth_size_method + 2];
            if ((depth_word & 0xfffe0000u) != 0) {
                return clear_plan_failure(stop,
                                          "depth target array mode has unsupported flags",
                                          depth_size_method + 2);
            }
            // The depth array-mode register (MacroHLE ZetaLayerCountOffset)
            // carries the cleared layer count; bit 16 disables array mode.
            const bool layered = (depth_word & depth_size_layered_flag_bit) == 0;
            const std::uint32_t layers = layered ? low_half(depth_word) : 1;
            if (layers == 0) {
                return clear_plan_failure(stop, "depth target names zero array layers",
                                          depth_size_method + 2);
            }
            // The depth target binds layer zero, so the argument's layer
            // field alone selects the first cleared slice.
            const auto [depth_first, depth_count] = layer_range(layer_field, layers);
            if (std::uint64_t{depth_first} + depth_count > layers) {
                return clear_plan_failure(stop, "clear layer range exceeds the depth target",
                                          depth_size_method + 2);
            }
            // The depth target clips the programmed surface, as for colour
            // clears.
            ClearRegion region{};
            if (!decode_clear_region(stop, std::min(surface_width, depth_width),
                                     std::min(surface_height, depth_height), scissored,
                                     region)) {
                return clear_plan_failure(stop, "clear scissor state is malformed");
            }
            if (region.width == 0 || region.height == 0) {
                plan.clear_depth = plan.clear_stencil = false;
            } else {
                TargetGeometry geometry{};
                if (const auto error = decode_depth_target_geometry(
                        stop.depth.tile_mode, depth_width, depth_height, info, geometry)) {
                    return clear_plan_failure(stop, error->detail, depth_state_method + 3);
                }
                if (!build_attachment_descriptor(stop.depth.address, geometry, info, layers,
                                                 stop.depth.array_pitch, {},
                                                 depth_attachment.texture)) {
                    return clear_plan_failure(stop,
                                              "depth clear target storage layout is invalid");
                }
                depth_attachment.texture.format = format;
                depth_attachment.physical_target = 0;
                depth_attachment.level = 0;
                depth_attachment.layer = depth_first;
                depth_attachment.layer_count = depth_count;
                depth_region = region;
                depth_admitted = true;
            }
        }
    }

    if (!colour_admitted && !depth_admitted) {
        plan.no_op = true;
        return plan;
    }
    if (colour_admitted && depth_admitted &&
        colour_attachment.layer_count != depth_attachment.layer_count) {
        // Each aspect's layer count comes from its own array-mode register;
        // one plan can only express equal ranges.
        return clear_plan_failure(stop,
            "combined clear names differing color and depth layer counts");
    }

    // A combined clear keeps the colour attachment in target and the
    // depth/stencil descriptor alongside; single-aspect clears keep the
    // established shape with the cleared aspects named on the attachment.
    if (colour_admitted) {
        plan.target = colour_attachment;
        plan.color_mask = static_cast<std::uint8_t>(component_mask);
        plan.region = RenderArea{colour_region.x, colour_region.y,
                                 colour_region.width, colour_region.height};
        if (depth_admitted) plan.depth_stencil = depth_attachment;
    } else {
        plan.target = depth_attachment;
        plan.target.aspect = clear_depth && clear_stencil
            ? TextureAspect::DepthStencil
            : (clear_stencil ? TextureAspect::Stencil : TextureAspect::Depth);
        plan.region = RenderArea{depth_region.x, depth_region.y,
                                 depth_region.width, depth_region.height};
    }
    plan.clear_depth = clear_depth && depth_admitted;
    plan.clear_stencil = clear_stencil && depth_admitted;
    if (plan.clear_depth || plan.clear_stencil) {
        const float depth_value = std::bit_cast<float>(stop.registers[clear_depth_method]);
        if (!std::isfinite(depth_value)) {
            return clear_plan_failure(stop, "depth clear value is not finite");
        }
        plan.depth_value = static_cast<double>(depth_value);
        plan.stencil_value = stop.registers[clear_stencil_method];
        // With ClearFlags bit 0 set the clear respects the front stencil mask;
        // otherwise it writes all eight stencil bits.
        plan.stencil_write_mask =
            (clear_control & clear_control_stencil_mask_bit) != 0
                ? static_cast<std::uint8_t>(stop.registers[stencil_test_method + 7] & 0xff)
                : 0xff;
    }

    // A clear folds into a load action only when it is unconditional, covers
    // every cleared attachment's full extent and every aspect of its format;
    // a partial colour mask, a partial stencil write mask or a skipped aspect
    // preserves data and stays a real operation. The batch render pass binds
    // one slice, so only single-layer storage folds there; a full-range clear
    // of an arrayed attachment folds into a standalone layered pass load
    // action instead. A combined colour and depth/stencil clear never folds:
    // the tested Metal 4 implementation drops the depth load action of an
    // empty combined pass, so it encodes the single-pass full-screen draw the
    // combined clear requires.
    bool fold_ready = colour_admitted != depth_admitted;
    bool single_layer = true;
    bool full_range = true;
    if (colour_admitted) {
        fold_ready = fold_ready && component_mask == 0xf &&
                     colour_region.x == 0 && colour_region.y == 0 &&
                     colour_region.width == colour_attachment.texture.width &&
                     colour_region.height == colour_attachment.texture.height;
        single_layer = single_layer && colour_attachment.texture.layers == 1;
        full_range = full_range && colour_attachment.layer == 0 &&
                     colour_attachment.layer_count == colour_attachment.texture.layers;
    }
    if (depth_admitted) {
        const TextureFormatInfo info = texture_format_info(depth_attachment.texture.format);
        const bool stencil_mask_preserves =
            info.stencil && (!plan.clear_stencil ||
                             ((clear_control & clear_control_stencil_mask_bit) != 0 &&
                              (stop.registers[stencil_test_method + 7] & 0xff) != 0xff));
        const bool full_aspects = (!info.depth || plan.clear_depth) &&
                                  (!info.stencil || plan.clear_stencil) &&
                                  !stencil_mask_preserves;
        fold_ready = fold_ready && full_aspects &&
                     depth_region.x == 0 && depth_region.y == 0 &&
                     depth_region.width == depth_attachment.texture.width &&
                     depth_region.height == depth_attachment.texture.height;
        single_layer = single_layer && depth_attachment.texture.layers == 1;
        full_range = full_range && depth_attachment.layer == 0 &&
                     depth_attachment.layer_count == depth_attachment.texture.layers;
    }
    plan.fold_into_load = fold_ready && full_range && single_layer;
    plan.fold_layers_into_load = fold_ready && full_range && !single_layer;
    return plan;
}

}  // namespace armigo::graphics
