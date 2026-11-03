// ============================================================================
// Armigo Graphics - GPU Color Clear Planning
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/color_clear.h>
#include <bit>
#include <cmath>
#include <limits>

namespace armigo::graphics {
namespace {

constexpr std::size_t bytes_per_pixel = 4;
constexpr std::size_t gob_width_bytes = 64;
constexpr std::size_t gob_height_rows = 8;
constexpr std::size_t gob_size = gob_width_bytes * gob_height_rows;

constexpr std::size_t raster_enable_method = 0xdf;
constexpr std::size_t clear_surface_method = 0x674;
constexpr std::size_t clear_color_method = 0x360;
constexpr std::size_t scissor_method = 0x380;
constexpr std::size_t clear_control_method = 0x43e;
constexpr std::size_t sample_mode_method = 0x574;
constexpr std::size_t render_enable_override_method = 0x651;
constexpr std::size_t surface_clip_method = 0x3fd;

constexpr CommandResult failure(CommandStatus status,
                                std::string_view detail,
                                std::uint64_t address = 0,
                                std::uint32_t method = 0,
                                std::uint32_t argument = 0) noexcept
{
    return {status, detail, address, 0, 0, method, argument, engine_class::threed};
}

constexpr bool checked_multiply(std::size_t left, std::size_t right, std::size_t& product) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

constexpr bool checked_add(std::size_t left, std::size_t right, std::size_t& sum) noexcept
{
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    sum = left + right;
    return true;
}

constexpr bool align_up(std::size_t value, std::size_t alignment, std::size_t& aligned) noexcept
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

constexpr std::uint32_t low_half(std::uint32_t value) noexcept
{
    return value & 0xffff;
}

constexpr std::uint32_t high_half(std::uint32_t value) noexcept
{
    return value >> 16;
}

CommandResult stop_failure(const RenderStop& stop, std::string_view detail) noexcept
{
    return failure(CommandStatus::Unsupported,
                   detail,
                   stop.original.address,
                   static_cast<std::uint32_t>(clear_surface_method),
                   stop.registers[clear_surface_method]);
}

PixelByteOffsetResult invalid_plan(std::string_view detail) noexcept
{
    return failure(CommandStatus::InvalidEntry, detail);
}

}  // namespace

ColorClearResult plan_color_clear(const RenderStop& stop) noexcept
{
    if (stop.kind != RenderKind::Clear || stop.engine != engine_class::threed) {
        return stop_failure(stop, "render stop is not a 3D color clear");
    }

    const std::uint32_t clear = stop.registers[clear_surface_method];
    if ((clear & 0xfc000000U) != 0) {
        return stop_failure(stop, "color clear has unsupported argument bits");
    }
    if ((clear & 0x3U) != 0) {
        return stop_failure(stop, "depth or stencil clear is unsupported");
    }
    if ((clear & 0x3cU) != 0x3cU) {
        return stop_failure(stop, "color clear must write every RGBA component");
    }

    const std::uint32_t target_index = (clear >> 6) & 0xf;
    if (target_index >= stop.color_targets.size()) {
        return stop_failure(stop, "color clear render-target index is out of range");
    }
    if (((clear >> 10) & 0xffff) != 0) {
        return stop_failure(stop, "layered color clear is unsupported");
    }

    const std::uint32_t render_enable_mode = stop.effective_render_enable_mode;
    const std::uint32_t render_enable_override = stop.registers[render_enable_override_method];
    const bool render_unconditionally =
        render_enable_override == 1 || (render_enable_override == 0 && render_enable_mode == 1);
    if (!render_unconditionally) {
        return stop_failure(stop, "conditional or disabled color clear is unsupported");
    }
    if (stop.registers[raster_enable_method] != 1) {
        return stop_failure(stop, "color clear with rasterization disabled is unsupported");
    }

    const std::uint32_t clear_control = stop.registers[clear_control_method];
    constexpr std::uint32_t known_clear_control = 0x1111;
    if ((clear_control & ~known_clear_control) != 0) {
        return stop_failure(stop, "color clear control has unsupported bits");
    }
    if ((clear_control & 0x10) != 0) {
        return stop_failure(stop, "clear-rectangle color clear is unsupported");
    }
    if ((clear_control & 0x1000) != 0) {
        return stop_failure(stop, "viewport-clipped color clear is unsupported");
    }
    if (stop.registers[sample_mode_method] != 0) {
        return stop_failure(stop, "multisampled color clear is unsupported");
    }

    const TargetDescription& target = stop.color_targets[target_index];
    ColorTargetFormat format{};
    if (target.format == static_cast<std::uint32_t>(ColorTargetFormat::A2B10G10R10Unorm)) {
        format = ColorTargetFormat::A2B10G10R10Unorm;
    } else if (target.format == static_cast<std::uint32_t>(ColorTargetFormat::R8G8B8A8Unorm)) {
        format = ColorTargetFormat::R8G8B8A8Unorm;
    } else if (target.format == static_cast<std::uint32_t>(ColorTargetFormat::R8G8B8A8Srgb)) {
        format = ColorTargetFormat::R8G8B8A8Srgb;
    } else {
        return stop_failure(stop, "color clear target format is unsupported");
    }

    const std::uint32_t clip_horizontal = stop.registers[surface_clip_method];
    const std::uint32_t clip_vertical = stop.registers[surface_clip_method + 1];
    const std::uint32_t clip_x = low_half(clip_horizontal);
    const std::uint32_t clip_y = low_half(clip_vertical);
    const std::uint32_t width = high_half(clip_horizontal);
    const std::uint32_t height = high_half(clip_vertical);
    if (clip_x != 0 || clip_y != 0 || width == 0 || height == 0) {
        return stop_failure(stop, "surface clip does not describe a full nonempty target");
    }
    if (target.height != height) {
        return stop_failure(stop, "surface clip height does not match the color target");
    }

    // The clear applies to the full surface unless clear control selects the
    // scissor. The scissor rectangle is packed as inclusive minimum and
    // exclusive maximum coordinates and
    // clears exactly max-min texels when clear control uses the scissor
    // in host coordinates. A disabled
    // scissor leaves the clear full. Only nonempty, contained rectangles are
    // accepted in this profile; empty, inverted, or overshooting extents stay
    // explicit unsupported results.
    ColorClearRegion region{0, 0, width, height};
    if ((clear_control & 0x100) != 0 && stop.registers[scissor_method] != 0) {
        if (stop.registers[scissor_method] != 1) {
            return stop_failure(stop, "clear scissor state is unsupported");
        }
        const std::uint32_t horizontal = stop.registers[scissor_method + 1];
        const std::uint32_t vertical = stop.registers[scissor_method + 2];
        const std::uint32_t left = low_half(horizontal);
        const std::uint32_t top = low_half(vertical);
        const std::uint32_t right = high_half(horizontal);
        const std::uint32_t bottom = high_half(vertical);
        if (right <= left || bottom <= top || right > width || bottom > height) {
            return stop_failure(stop, "clear scissor rectangle is empty or not contained");
        }
        region = {left, top, right - left, bottom - top};
    }

    const std::size_t target_register = 0x200 + target_index * 0x10;
    if (stop.registers[target_register + 6] != 1 || target.depth != 1) {
        return stop_failure(stop, "arrayed or three-dimensional color target is unsupported");
    }
    if (target.base_layer != 0) {
        return stop_failure(stop, "color target base layer is unsupported");
    }

    ColorClearPlan plan{};
    plan.target_address = target.address;
    plan.target_index = target_index;
    plan.width = width;
    plan.height = height;
    plan.format = format;
    for (std::size_t index = 0; index < plan.color.size(); ++index) {
        plan.color[index] = std::bit_cast<float>(stop.registers[clear_color_method + index]);
        if (!std::isfinite(plan.color[index])) {
            return stop_failure(stop, "color clear value is not finite");
        }
    }
    plan.region = region;

    const std::uint32_t tile_mode = target.tile_mode;
    constexpr std::uint32_t pitch_layout_bit = 1U << 12;
    if ((tile_mode & pitch_layout_bit) != 0) {
        if (tile_mode != pitch_layout_bit) {
            return stop_failure(stop, "pitch color target has unsupported tile fields");
        }
        std::size_t logical_row_bytes = 0;
        if (!checked_multiply(width, bytes_per_pixel, logical_row_bytes) ||
            target.width < logical_row_bytes || (target.width % bytes_per_pixel) != 0) {
            return stop_failure(stop, "pitch color target stride is too small or misaligned");
        }
        plan.layout = ColorTargetLayout::Pitch;
        plan.pitch_bytes = target.width;
        plan.row_bytes = target.width;
        if (!checked_multiply(plan.row_bytes, height, plan.storage_size)) {
            return stop_failure(stop, "pitch color target storage size overflows");
        }
    } else {
        constexpr std::uint32_t known_block_linear_bits = 0x00000fff;
        const std::uint32_t block_width_log2 = tile_mode & 0xf;
        const std::uint32_t block_height_log2 = (tile_mode >> 4) & 0xf;
        const std::uint32_t block_depth_log2 = (tile_mode >> 8) & 0xf;
        if ((tile_mode & ~known_block_linear_bits) != 0 || block_width_log2 != 0 ||
            block_depth_log2 != 0 || block_height_log2 > 5) {
            return stop_failure(stop, "block-linear color target tile mode is unsupported");
        }
        if (target.width != width) {
            return stop_failure(stop, "surface clip width does not match the color target");
        }
        std::size_t logical_row_bytes = 0;
        if (!checked_multiply(width, bytes_per_pixel, logical_row_bytes) ||
            !align_up(logical_row_bytes, gob_width_bytes, plan.row_bytes)) {
            return stop_failure(stop, "block-linear color target row size overflows");
        }
        const std::size_t block_height_rows = gob_height_rows << block_height_log2;
        std::size_t storage_height = 0;
        if (!align_up(height, block_height_rows, storage_height) ||
            !checked_multiply(plan.row_bytes, storage_height, plan.storage_size)) {
            return stop_failure(stop, "block-linear color target storage size overflows");
        }
        plan.layout = ColorTargetLayout::BlockLinear;
        plan.block_height_log2 = block_height_log2;
    }

    if (plan.storage_size == 0 || plan.storage_size > max_color_clear_storage_size) {
        return stop_failure(stop, "color target storage exceeds the clear resource policy");
    }
    if (plan.storage_size - 1 > std::numeric_limits<std::uint64_t>::max() - plan.target_address) {
        return stop_failure(stop, "color target GPU address range overflows");
    }
    return plan;
}

PixelByteOffsetResult
pixel_byte_offset(const ColorClearPlan& plan, std::uint32_t x, std::uint32_t y) noexcept
{
    if (plan.width == 0 || plan.height == 0 || x >= plan.width || y >= plan.height) {
        return invalid_plan("color target pixel coordinate is out of bounds");
    }

    std::size_t logical_row_bytes = 0;
    if (!checked_multiply(plan.width, bytes_per_pixel, logical_row_bytes)) {
        return invalid_plan("color target logical row size overflows");
    }

    std::size_t offset = 0;
    if (plan.layout == ColorTargetLayout::Pitch) {
        std::size_t expected_storage_size = 0;
        if (plan.pitch_bytes != plan.row_bytes || plan.row_bytes < logical_row_bytes ||
            (plan.row_bytes % bytes_per_pixel) != 0 ||
            !checked_multiply(plan.row_bytes, plan.height, expected_storage_size) ||
            expected_storage_size != plan.storage_size ||
            !checked_multiply(y, plan.row_bytes, offset) ||
            !checked_add(offset, static_cast<std::size_t>(x) * bytes_per_pixel, offset)) {
            return invalid_plan("pitch color target plan is malformed");
        }
    } else if (plan.layout == ColorTargetLayout::BlockLinear) {
        if (plan.pitch_bytes != 0 || plan.block_height_log2 > 5) {
            return invalid_plan("block-linear color target plan is malformed");
        }
        const std::size_t block_height_gobs = std::size_t{1} << plan.block_height_log2;
        const std::size_t block_height_rows = gob_height_rows * block_height_gobs;
        std::size_t storage_height = 0;
        std::size_t expected_storage_size = 0;
        if (plan.row_bytes < logical_row_bytes || plan.row_bytes % gob_width_bytes != 0 ||
            !align_up(plan.height, block_height_rows, storage_height) ||
            !checked_multiply(plan.row_bytes, storage_height, expected_storage_size) ||
            expected_storage_size != plan.storage_size) {
            return invalid_plan("block-linear color target plan is malformed");
        }

        const std::size_t x_bytes = static_cast<std::size_t>(x) * bytes_per_pixel;
        const std::size_t gobs_per_row = plan.row_bytes / gob_width_bytes;
        const std::size_t block_size = gobs_per_row * gob_size * block_height_gobs;
        const std::size_t gob_y = y / gob_height_rows;
        offset = (gob_y / block_height_gobs) * block_size;
        offset += (x_bytes / gob_width_bytes) * gob_size * block_height_gobs;
        offset += (gob_y % block_height_gobs) * gob_size;
        offset += ((x_bytes & 0x3f) >> 5) << 8;
        offset += ((y & 0x7) >> 1) << 6;
        offset += ((x_bytes & 0x1f) >> 4) << 5;
        offset += (y & 1) << 4;
        offset += x_bytes & 0xf;
    } else {
        return invalid_plan("color target layout is invalid");
    }

    if (offset > plan.storage_size || bytes_per_pixel > plan.storage_size - offset) {
        return invalid_plan("color target pixel offset exceeds storage");
    }
    return offset;
}

}  // namespace armigo::graphics
