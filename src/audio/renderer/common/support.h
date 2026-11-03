// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "renderer/common/types.h"

#include <format>
#include <string>
#include <type_traits>

namespace armigo::audio::renderer {

template <typename T>
constexpr T align_up(T value, std::size_t alignment) noexcept {
    const auto mask = static_cast<T>(alignment - 1);
    return static_cast<T>((value + mask) & ~mask);
}

template <typename T>
constexpr T align_down(T value, std::size_t alignment) noexcept {
    return static_cast<T>(value & ~static_cast<T>(alignment - 1));
}

template <typename T>
constexpr bool is_aligned(T value, std::size_t alignment) noexcept {
    return (value & static_cast<T>(alignment - 1)) == 0;
}

// Four ASCII characters packed little-endian, as the renderer's magic values.
constexpr u32 make_magic(char a, char b, char c, char d) noexcept {
    return u32(static_cast<u8>(a)) | (u32(static_cast<u8>(b)) << 8) |
           (u32(static_cast<u8>(c)) << 16) | (u32(static_cast<u8>(d)) << 24);
}

// Host monotonic time, the renderer's clock for performance metrics.
u64 host_time_ns() noexcept;
inline u64 host_time_us() noexcept { return host_time_ns() / 1000; }

enum class LogLevel { trace, debug, info, warning, error, critical };

// Renderer diagnostics. Errors and warnings go to stderr with a bound on the
// total number of lines; ARMIGO_AUDIO_TRACE=1 also prints debug and trace
// lines and lifts the bound.
void log_message(LogLevel level, const std::string& text) noexcept;
bool log_enabled(LogLevel level) noexcept;

// Formats an argument for diagnostics: enumerations print their underlying value.
template <typename T>
constexpr decltype(auto) log_argument(const T& value) {
    if constexpr (std::is_enum_v<T>)
        return static_cast<std::underlying_type_t<T>>(value);
    else
        return (value);
}

template <typename... Args>
void log_format(LogLevel level, std::format_string<decltype(log_argument(std::declval<const Args&>()))...> format,
                const Args&... args) {
    if (!log_enabled(level)) return;
    try {
        log_message(level, std::format(format, log_argument(args)...));
    } catch (...) {
    }
}

}  // namespace armigo::audio::renderer

#define AUDIO_LOG_TRACE(...) ::armigo::audio::renderer::log_format(::armigo::audio::renderer::LogLevel::trace, __VA_ARGS__)
#define AUDIO_LOG_DEBUG(...) ::armigo::audio::renderer::log_format(::armigo::audio::renderer::LogLevel::debug, __VA_ARGS__)
#define AUDIO_LOG_INFO(...) ::armigo::audio::renderer::log_format(::armigo::audio::renderer::LogLevel::info, __VA_ARGS__)
#define AUDIO_LOG_WARNING(...) ::armigo::audio::renderer::log_format(::armigo::audio::renderer::LogLevel::warning, __VA_ARGS__)
#define AUDIO_LOG_ERROR(...) ::armigo::audio::renderer::log_format(::armigo::audio::renderer::LogLevel::error, __VA_ARGS__)
#define AUDIO_LOG_CRITICAL(...) ::armigo::audio::renderer::log_format(::armigo::audio::renderer::LogLevel::critical, __VA_ARGS__)
