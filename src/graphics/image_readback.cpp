// ============================================================================
// Armigo Graphics - CPU Image Readback
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/image_readback.h>
#include <cstring>
#include <limits>
#include <new>

namespace armigo::graphics {
namespace {

constexpr std::size_t bytes_per_pixel = 4;
constexpr std::uint32_t max_readback_dimension = 16384;

constexpr ImageReadbackError failure(ImageReadbackError::Code code,
                                     std::string_view detail) noexcept
{
    return {code, detail};
}

constexpr bool checked_multiply(std::size_t left, std::size_t right, std::size_t& product) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

}  // namespace

Rgba8ReadbackResult readback_rgba8(const ColorClearPlan& plan,
                                   std::span<const std::byte> storage,
                                   std::array<std::int32_t, 4> crop) noexcept
{
    if (plan.format != ColorTargetFormat::R8G8B8A8Unorm) {
        return failure(ImageReadbackError::Code::InvalidLayout,
                       "image readback requires an R8G8B8A8Unorm color target");
    }
    if (plan.width == 0 || plan.height == 0) {
        return failure(ImageReadbackError::Code::InvalidLayout,
                       "image readback requires a nonempty color target");
    }
    if (plan.width > max_readback_dimension || plan.height > max_readback_dimension) {
        return failure(ImageReadbackError::Code::ExtentLimit,
                       "color target dimensions exceed the image readback limit");
    }
    if (plan.storage_size > max_color_clear_storage_size) {
        return failure(ImageReadbackError::Code::ExtentLimit,
                       "color target storage exceeds the image readback resource policy");
    }
    if (plan.layout == ColorTargetLayout::Pitch && plan.block_height_log2 != 0) {
        return failure(ImageReadbackError::Code::InvalidLayout,
                       "pitch color target has a block height");
    }

    // Probe the complete plan once so that malformed geometry is rejected before
    // crop selection or allocation. pixel_byte_offset owns the layout checks.
    if (!std::holds_alternative<std::size_t>(pixel_byte_offset(plan, 0, 0))) {
        return failure(ImageReadbackError::Code::InvalidLayout,
                       "color target plan geometry is malformed");
    }

    const std::int64_t width = plan.width;
    const std::int64_t height = plan.height;
    std::int64_t left = crop[0];
    std::int64_t top = crop[1];
    std::int64_t right = crop[2];
    std::int64_t bottom = crop[3];
    const bool full_surface_capture = left == 0 && top == 0 && right == 0 && bottom == 0;
    if (full_surface_capture) {
        left = 0;
        top = 0;
        right = width;
        bottom = height;
    } else if (left < 0 || top < 0 || right <= left || bottom <= top || right > width ||
               bottom > height) {
        return failure(ImageReadbackError::Code::InvalidCrop,
                       "image readback crop is empty or outside the color target");
    }

    const auto output_width = static_cast<std::uint32_t>(right - left);
    const auto output_height = static_cast<std::uint32_t>(bottom - top);
    std::size_t output_row_bytes = 0;
    std::size_t output_size = 0;
    if (!checked_multiply(output_width, bytes_per_pixel, output_row_bytes) ||
        !checked_multiply(output_row_bytes, output_height, output_size) ||
        output_size > max_color_clear_storage_size) {
        return failure(ImageReadbackError::Code::ExtentLimit,
                       "image readback output exceeds the resource policy");
    }

    if (storage.size() < plan.storage_size) {
        return failure(ImageReadbackError::Code::InputTooSmall,
                       "image readback storage is shorter than the color target");
    }

    try {
        Rgba8Image image{};
        image.width = output_width;
        image.height = output_height;
        image.pixels.resize(output_size);
        // The probe above validated the plan geometry, and the crop lies
        // inside the surface, so offsets are computed without per-pixel checks.
        if (plan.layout == ColorTargetLayout::Pitch) {
            for (std::uint32_t y = 0; y < output_height; ++y)
                std::memcpy(image.pixels.data() + std::size_t(y) * output_row_bytes,
                            storage.data() + std::size_t(top + y) * plan.row_bytes +
                                std::size_t(left) * bytes_per_pixel,
                            output_row_bytes);
            return image;
        }
        const std::size_t block_height_gobs = std::size_t{1} << plan.block_height_log2;
        const std::size_t gob_bytes = 512;
        const std::size_t block_size = plan.row_bytes / 64 * gob_bytes * block_height_gobs;
        for (std::uint32_t y = 0; y < output_height; ++y) {
            const std::size_t row = std::size_t(top) + y;
            const std::size_t gob_y = row / 8;
            const std::size_t row_base = (gob_y / block_height_gobs) * block_size +
                                         (gob_y % block_height_gobs) * gob_bytes +
                                         (((row & 0x7) >> 1) << 6) + ((row & 1) << 4);
            auto* out = image.pixels.data() + std::size_t(y) * output_row_bytes;
            for (std::uint32_t x = 0; x < output_width; ++x) {
                const std::size_t x_bytes = (std::size_t(left) + x) * bytes_per_pixel;
                const std::size_t offset = row_base + (x_bytes / 64) * gob_bytes * block_height_gobs +
                                           (((x_bytes & 0x3f) >> 5) << 8) +
                                           (((x_bytes & 0x1f) >> 4) << 5) + (x_bytes & 0xf);
                std::memcpy(out + std::size_t(x) * bytes_per_pixel, storage.data() + offset,
                            bytes_per_pixel);
            }
        }
        return image;
    } catch (const std::bad_alloc&) {
        return failure(ImageReadbackError::Code::AllocationFailure,
                       "image readback output allocation failed");
    }
}

}  // namespace armigo::graphics
