// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <algorithm>
#include <armigo/graphics/image_readback.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

constexpr std::size_t bytes_per_pixel = 4;
constexpr std::size_t guard_size = 16;
constexpr std::byte padding_byte{0x7b};
constexpr std::byte guard_byte{0x5a};

ColorClearPlan make_pitch_plan(std::uint32_t width,
                               std::uint32_t height,
                               std::uint32_t pitch_bytes)
{
    ColorClearPlan plan{};
    plan.target_address = 0xfffffffffffff000ULL;
    plan.width = width;
    plan.height = height;
    plan.row_bytes = pitch_bytes;
    plan.pitch_bytes = pitch_bytes;
    plan.storage_size = static_cast<std::size_t>(pitch_bytes) * height;
    plan.format = ColorTargetFormat::R8G8B8A8Unorm;
    plan.layout = ColorTargetLayout::Pitch;
    return plan;
}

ColorClearPlan make_block_linear_plan(std::uint32_t width,
                                      std::uint32_t height,
                                      std::uint32_t block_height_log2)
{
    const std::size_t logical_row_bytes = static_cast<std::size_t>(width) * bytes_per_pixel;
    const std::size_t row_bytes = (logical_row_bytes + 63) / 64 * 64;
    const std::size_t block_height_rows = std::size_t{8} << block_height_log2;
    const std::size_t storage_height =
        (static_cast<std::size_t>(height) + block_height_rows - 1) / block_height_rows *
        block_height_rows;
    ColorClearPlan plan{};
    plan.target_address = 0xfffffffffffff000ULL;
    plan.width = width;
    plan.height = height;
    plan.row_bytes = row_bytes;
    plan.pitch_bytes = 0;
    plan.storage_size = row_bytes * storage_height;
    plan.format = ColorTargetFormat::R8G8B8A8Unorm;
    plan.layout = ColorTargetLayout::BlockLinear;
    plan.block_height_log2 = block_height_log2;
    return plan;
}

std::array<std::byte, 4> expected_pixel(std::uint32_t x, std::uint32_t y)
{
    return {
        static_cast<std::byte>(x & 0xffU),
        static_cast<std::byte>(y & 0xffU),
        static_cast<std::byte>((x + y) & 0xffU),
        static_cast<std::byte>((x * 37U + y * 53U) & 0xffU),
    };
}

std::vector<std::byte> make_storage(const ColorClearPlan& plan, bool direct_pitch)
{
    std::vector<std::byte> storage(plan.storage_size + guard_size, guard_byte);
    std::fill_n(storage.begin(), plan.storage_size, padding_byte);
    for (std::uint32_t y = 0; y < plan.height; ++y) {
        for (std::uint32_t x = 0; x < plan.width; ++x) {
            std::size_t offset = 0;
            if (direct_pitch) {
                offset = static_cast<std::size_t>(y) * plan.row_bytes +
                         static_cast<std::size_t>(x) * bytes_per_pixel;
            } else {
                const PixelByteOffsetResult result = pixel_byte_offset(plan, x, y);
                CHECK(std::holds_alternative<std::size_t>(result));
                if (!std::holds_alternative<std::size_t>(result)) {
                    return storage;
                }
                offset = std::get<std::size_t>(result);
            }
            const std::array<std::byte, 4> pixel = expected_pixel(x, y);
            std::copy(pixel.begin(), pixel.end(),
                      storage.begin() + static_cast<std::ptrdiff_t>(offset));
        }
    }
    return storage;
}

void check_image(const Rgba8ReadbackResult& result,
                 std::uint32_t left,
                 std::uint32_t top,
                 std::uint32_t width,
                 std::uint32_t height)
{
    CHECK(std::holds_alternative<Rgba8Image>(result));
    if (!std::holds_alternative<Rgba8Image>(result)) {
        return;
    }
    const Rgba8Image& image = std::get<Rgba8Image>(result);
    CHECK(image.width == width);
    CHECK(image.height == height);
    CHECK(image.pixels.size() == static_cast<std::size_t>(width) * height * bytes_per_pixel);
    if (image.pixels.size() != static_cast<std::size_t>(width) * height * bytes_per_pixel) {
        return;
    }
    bool matches = true;
    for (std::uint32_t y = 0; y < height && matches; ++y) {
        for (std::uint32_t x = 0; x < width && matches; ++x) {
            const std::array<std::byte, 4> pixel = expected_pixel(left + x, top + y);
            const std::size_t offset =
                (static_cast<std::size_t>(y) * width + x) * bytes_per_pixel;
            for (std::size_t channel = 0; channel < pixel.size(); ++channel) {
                if (image.pixels[offset + channel] != pixel[channel]) {
                    matches = false;
                }
            }
        }
    }
    CHECK(matches);
}

void check_error(const Rgba8ReadbackResult& result, ImageReadbackError::Code expected)
{
    CHECK(std::holds_alternative<ImageReadbackError>(result));
    if (!std::holds_alternative<ImageReadbackError>(result)) {
        return;
    }
    const ImageReadbackError& error = std::get<ImageReadbackError>(result);
    CHECK(error.code == expected);
    CHECK(!error.detail.empty());
}

void test_pitch_crop_with_padding()
{
    const ColorClearPlan plan = make_pitch_plan(23, 11, 112);
    std::vector<std::byte> storage = make_storage(plan, true);
    const std::vector<std::byte> original = storage;
    const std::span<const std::byte> source(storage);

    check_image(readback_rgba8(plan, source, {3, 2, 17, 9}), 3, 2, 14, 7);
    CHECK(storage == original);

    check_image(readback_rgba8(plan, source, {}), 0, 0, 23, 11);
    CHECK(storage == original);

    check_image(readback_rgba8(plan, source, {0, 0, 23, 11}), 0, 0, 23, 11);
    CHECK(storage == original);

    check_image(readback_rgba8(plan, source, {0, 0, 23, 10}), 0, 0, 23, 10);
    check_image(readback_rgba8(plan, source, {22, 10, 23, 11}), 22, 10, 1, 1);
    check_image(readback_rgba8(plan, source, {0, 5, 1, 11}), 0, 5, 1, 6);

    CHECK(storage == original);
    CHECK(storage[plan.storage_size] == guard_byte);
    CHECK(storage.back() == guard_byte);
}

void test_block_linear_crop()
{
    for (const std::uint32_t block_height_log2 : {0U, 2U}) {
        const ColorClearPlan plan = make_block_linear_plan(23, 11, block_height_log2);
        std::vector<std::byte> storage = make_storage(plan, false);
        const std::vector<std::byte> original = storage;
        const std::span<const std::byte> source(storage);

        check_image(readback_rgba8(plan, source, {1, 5, 13, 10}), 1, 5, 12, 5);
        CHECK(storage == original);

        check_image(readback_rgba8(plan, source, {}), 0, 0, 23, 11);
        CHECK(storage == original);

        check_image(readback_rgba8(plan, source, {0, 0, 23, 11}), 0, 0, 23, 11);
        check_image(readback_rgba8(plan, source, {22, 10, 23, 11}), 22, 10, 1, 1);
        check_image(readback_rgba8(plan, source, {4, 0, 12, 3}), 4, 0, 8, 3);

        CHECK(storage == original);
        CHECK(storage[plan.storage_size] == guard_byte);
        CHECK(storage.back() == guard_byte);
    }
}

void test_crop_validation()
{
    const ColorClearPlan plan = make_pitch_plan(23, 11, 112);
    const std::vector<std::byte> storage = make_storage(plan, true);
    const std::span<const std::byte> source(storage);
    constexpr auto min = std::numeric_limits<std::int32_t>::min();
    constexpr auto max = std::numeric_limits<std::int32_t>::max();
    const std::array<std::array<std::int32_t, 4>, 14> invalid{{
        {-1, 0, 5, 5},
        {0, -1, 5, 5},
        {0, 0, -1, 5},
        {0, 0, 5, -1},
        {min, 0, 5, 5},
        {5, 0, 2, 5},
        {0, 5, 5, 2},
        {0, 0, 24, 5},
        {0, 0, 5, 12},
        {0, 0, 0, 5},
        {2, 3, 2, 5},
        {0, 0, 5, 0},
        {0, 0, max, max},
        {23, 0, 23, 11},
    }};
    for (const auto& crop : invalid) {
        check_error(readback_rgba8(plan, source, crop), ImageReadbackError::Code::InvalidCrop);
    }
    check_error(readback_rgba8(plan, source, {0, 11, 23, 11}),
                ImageReadbackError::Code::InvalidCrop);
    check_image(readback_rgba8(plan, source, {0, 0, 23, 11}), 0, 0, 23, 11);
}

void test_input_size()
{
    const ColorClearPlan plan = make_pitch_plan(23, 11, 112);
    const std::vector<std::byte> storage = make_storage(plan, true);
    check_error(readback_rgba8(plan, std::span<const std::byte>(storage.data(),
                                                                plan.storage_size - 1)),
                ImageReadbackError::Code::InputTooSmall);
    check_error(readback_rgba8(plan, {}), ImageReadbackError::Code::InputTooSmall);
}

void test_malformed_plans()
{
    const ColorClearPlan pitch = make_pitch_plan(23, 11, 112);
    const std::vector<std::byte> pitch_storage(pitch.storage_size, std::byte{0});

    ColorClearPlan plan = pitch;
    plan.format = ColorTargetFormat::A2B10G10R10Unorm;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.format = static_cast<ColorTargetFormat>(0xff);
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.width = 0;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.height = 0;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.pitch_bytes += 4;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.row_bytes = 88;
    plan.pitch_bytes = 88;
    plan.storage_size = 88 * 11;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.row_bytes = 93;
    plan.pitch_bytes = 93;
    plan.storage_size = 93 * 11;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.storage_size -= 1;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.layout = static_cast<ColorTargetLayout>(0x7f);
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    plan = pitch;
    plan.block_height_log2 = 1;
    check_error(readback_rgba8(plan, pitch_storage), ImageReadbackError::Code::InvalidLayout);

    const ColorClearPlan block = make_block_linear_plan(23, 11, 2);
    const std::vector<std::byte> block_storage(block.storage_size, std::byte{0});

    plan = block;
    plan.block_height_log2 = 6;
    check_error(readback_rgba8(plan, block_storage), ImageReadbackError::Code::InvalidLayout);

    plan = block;
    plan.pitch_bytes = 4;
    check_error(readback_rgba8(plan, block_storage), ImageReadbackError::Code::InvalidLayout);

    plan = block;
    plan.row_bytes += 4;
    check_error(readback_rgba8(plan, block_storage), ImageReadbackError::Code::InvalidLayout);

    plan = block;
    plan.storage_size -= 1;
    check_error(readback_rgba8(plan, block_storage), ImageReadbackError::Code::InvalidLayout);

    plan = block;
    plan.layout = static_cast<ColorTargetLayout>(0xfe);
    check_error(readback_rgba8(plan, block_storage), ImageReadbackError::Code::InvalidLayout);
}

void test_extent_limits()
{
    check_error(readback_rgba8(make_pitch_plan(16385, 1, 16385 * 4), {}),
                ImageReadbackError::Code::ExtentLimit);
    check_error(readback_rgba8(make_pitch_plan(1, 16385, 4), {}),
                ImageReadbackError::Code::ExtentLimit);
    check_error(readback_rgba8(make_pitch_plan(16384, 16384, 16384 * 4), {}),
                ImageReadbackError::Code::ExtentLimit);
    check_error(readback_rgba8(make_pitch_plan(16384, 4097, 16384 * 4), {}),
                ImageReadbackError::Code::ExtentLimit);

    ColorClearPlan plan = make_pitch_plan(23, 11, 112);
    plan.storage_size = 0;
    check_error(readback_rgba8(plan, {}), ImageReadbackError::Code::InvalidLayout);
}

void test_max_dimension()
{
    const ColorClearPlan plan = make_pitch_plan(16384, 1, 16384 * 4);
    const std::vector<std::byte> storage = make_storage(plan, true);
    check_image(readback_rgba8(plan, storage), 0, 0, 16384, 1);
}

}  // namespace

int main()
{
    test_pitch_crop_with_padding();
    test_block_linear_crop();
    test_crop_validation();
    test_input_size();
    test_malformed_plans();
    test_extent_limits();
    test_max_dimension();
    return failures == 0 ? 0 : 1;
}
