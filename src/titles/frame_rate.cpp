// ============================================================================
// Armigo Compatibility - Title Compatibility Semantics
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/titles/frame_rate.h>

#include <armigo/horizon/ipc.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>

namespace armigo::titles {
namespace {

constexpr double ticks_per_second = 19'200'000.0;
constexpr std::uint32_t maximum_measured_rate = 300;
constexpr float minimum_ratio = 30.0f / maximum_measured_rate;
constexpr float maximum_ratio = 30.0f / 20.0f;
constexpr float minimum_step = 60.0f / maximum_measured_rate;
constexpr float maximum_step = 60.0f / 20.0f;

double measured_seconds(std::int64_t period_ticks) noexcept {
    return static_cast<double>(period_ticks) / ticks_per_second;
}

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

bool update_fractional_counter(const FrameRateLayout& layout, std::uint64_t module_base,
                               std::uint64_t object, const horizon::GuestBuffers& buffers) noexcept {
    std::uint32_t state = 0;
    if (!read_value(buffers, object, layout.state, state)) return false;
    if (state != layout.active_state) return true;
    std::int64_t ticks = 0;
    if (!read_period(layout, module_base, buffers, ticks)) return false;
    if (ticks <= 0) return true;
    // Match a native float-seconds getter followed by multiplication. The
    // interval-ratio variant keeps its distinct double-before-float order.
    const float seconds = static_cast<float>(measured_seconds(ticks));
    const float step = std::min(seconds, 0.05f) * 30.0f;
    float counter = 0;
    if (!read_value(buffers, object, layout.counter, counter) ||
        !std::isfinite(counter) || counter < 0) return false;
    if (layout.variant == FrameRateVariant::FractionalRepeat30) {
        if (layout.repeat_frames == 0) return false;
        if (counter == 0.0f) {
            std::uint8_t held = 0;
            if (!read_value(buffers, object, layout.repeat_flag, held)) return false;
            if (held != 0) counter = static_cast<float>(layout.repeat_frames);
        }
    }
    const float updated = layout.variant == FrameRateVariant::FractionalCounter30
        ? counter + step : std::max(0.0f, counter - step);
    if (!std::isfinite(updated)) return false;
    return write_value(buffers, object, layout.counter, updated);
}

}  // namespace

bool advance_frame_phases(const FramePhaseLayout& layout, std::uint64_t module_base,
                          std::uint64_t object, const horizon::GuestBuffers& buffers,
                          FramePhaseDispatch& dispatch) noexcept {
    dispatch = {};
    if (!layout.frame_step || layout.frame_step % 4 || layout.counter % 4 ||
        !layout.phase_count || layout.phase_count > 65535)
        return false;
    float step = 0;
    if (!read_value(buffers, module_base, layout.frame_step, step) ||
        !std::isfinite(step) || step < 0 || step > maximum_step)
        return false;
    if (step == 0) return true;
    float phase = 0;
    if (!read_value(buffers, object, layout.counter, phase) ||
        !std::isfinite(phase) || phase < 0 || phase >= layout.phase_count)
        return false;
    // Use the same float precision for crossing detection and persisted state.
    // Counting before rounding could lose a phase when the stored float rounds
    // onto a boundary. The published step limits this to at most two crossings.
    const float advanced = phase + step * 0.5f;
    const auto first = static_cast<std::uint32_t>(phase);
    const auto count = static_cast<std::uint32_t>(advanced) - first;
    const float wrapped = std::fmod(advanced, static_cast<float>(layout.phase_count));
    if (!write_value(buffers, object, layout.counter, wrapped)) return false;
    dispatch = {first, count};
    return true;
}

bool update_frame_rate(const FrameRateLayout& layout, std::uint64_t module_base,
                            std::uint64_t vfr_object, std::uint32_t interval,
                            const horizon::GuestBuffers& buffers,
                            std::uint32_t* maximum_rate) noexcept {
    if (maximum_rate) *maximum_rate = 0;
    if (layout.variant == FrameRateVariant::FractionalCounter30 ||
        layout.variant == FrameRateVariant::FractionalCountdown30 ||
        layout.variant == FrameRateVariant::FractionalRepeat30)
        return update_fractional_counter(layout, module_base, vfr_object, buffers);
    if (layout.variant != FrameRateVariant::IntervalRatio60) return false;
    // The frame step follows the measured period whatever the frame-rate
    // object's state, so counters keep time while the override is set too.
    std::int64_t period_ticks = 0;
    const bool period_read = layout.frame_step != 0;
    if (period_read) {
        if (!read_period(layout, module_base, buffers, period_ticks))
            return false;
        if (period_ticks > 0) {
            float step = static_cast<float>(measured_seconds(period_ticks) * 60.0);
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

    float ratio = static_cast<float>(measured_seconds(period_ticks) * 60.0 /
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

}  // namespace armigo::titles
