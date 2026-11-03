// ============================================================================
// Armigo Compatibility - Title Compatibility Semantics
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/titles/01007EF00011E000/frame_rate.h>

#include <armigo/horizon/ipc.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>

namespace armigo::titles::title_01007ef00011e000 {
namespace {

constexpr double ticks_per_second = 19'200'000.0;
constexpr std::uint32_t maximum_measured_rate = 300;
constexpr float minimum_ratio = 30.0f / maximum_measured_rate;
constexpr float maximum_ratio = 30.0f / 20.0f;
constexpr float minimum_step = 60.0f / maximum_measured_rate;
constexpr float maximum_step = 60.0f / 20.0f;

bool checked_address(std::uint64_t base, std::uint64_t offset, std::uint64_t& address) noexcept {
    if (offset > std::numeric_limits<std::uint64_t>::max() - base)
        return false;
    address = base + offset;
    return true;
}

template <typename T>
bool read_value(const horizon::GuestBuffers& buffers, std::uint64_t base, std::uint64_t offset,
                T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    std::uint64_t address = 0;
    if (!checked_address(base, offset, address))
        return false;
    std::array<std::byte, sizeof(T)> bytes{};
    if (!buffers.read_memory(address, bytes))
        return false;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return true;
}

template <typename T>
bool write_value(const horizon::GuestBuffers& buffers, std::uint64_t base, std::uint64_t offset,
                 const T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    std::uint64_t address = 0;
    if (!checked_address(base, offset, address))
        return false;
    const auto bytes = std::as_bytes(std::span{&value, std::size_t{1}});
    return buffers.write_memory(address, bytes);
}

// Reads the whole previous-frame period. False reports a memory error; a null
// framework or a nonpositive period leaves zero.
bool read_period(const FrameRateLayout& layout, std::uint64_t module_base,
                 const horizon::GuestBuffers& buffers, std::int64_t& period_ticks) noexcept {
    period_ticks = 0;
    std::uint64_t framework = 0;
    if (!read_value(buffers, module_base, layout.framework_pointer, framework))
        return false;
    if (framework != 0 && !read_value(buffers, framework, layout.framework_period, period_ticks))
        return false;
    if (period_ticks < 0)
        period_ticks = 0;
    return true;
}

}  // namespace

bool update_frame_rate(const FrameRateLayout& layout, std::uint64_t module_base,
                            std::uint64_t vfr_object, std::uint32_t interval,
                            const horizon::GuestBuffers& buffers,
                            std::uint32_t* maximum_rate) noexcept {
    if (maximum_rate) *maximum_rate = 0;
    // The frame step follows the measured period whatever the frame-rate
    // object's state, so counters keep time while the override is set too.
    std::int64_t period_ticks = 0;
    const bool period_read = layout.frame_step != 0;
    if (period_read) {
        if (!read_period(layout, module_base, buffers, period_ticks))
            return false;
        if (period_ticks > 0) {
            float step = static_cast<float>(static_cast<double>(period_ticks) / ticks_per_second * 60.0);
            if (step < minimum_step)
                step = minimum_step;
            if (step > maximum_step)
                step = maximum_step;
            if (!write_value(buffers, module_base, layout.frame_step, step))
                return false;
        }
    }

    std::uint8_t has_override = 0;
    if (!read_value(buffers, vfr_object, layout.override_flag, has_override))
        return false;
    if (has_override != 0 || interval == 0)
        return true;

    std::uint32_t base_interval = 0;
    if (!read_value(buffers, vfr_object, layout.base_interval, base_interval))
        return false;
    if (base_interval == 0)
        return true;

    if (!period_read && !read_period(layout, module_base, buffers, period_ticks))
        return false;
    if (period_ticks <= 0)
        return true;

    float ratio = static_cast<float>(static_cast<double>(period_ticks) / ticks_per_second * 60.0 /
                                     static_cast<double>(base_interval));
    if (ratio < minimum_ratio)
        ratio = minimum_ratio;
    if (ratio > maximum_ratio)
        ratio = maximum_ratio;

    // Writing the incoming interval first prevents the guest implementation
    // from recomputing the ratio during the immediately following call.
    if (!write_value(buffers, vfr_object, layout.interval, interval))
        return false;
    if (!write_value(buffers, vfr_object, layout.ratio, ratio))
        return false;
    if (maximum_rate) *maximum_rate = maximum_measured_rate;
    return true;
}

}  // namespace armigo::titles::title_01007ef00011e000
