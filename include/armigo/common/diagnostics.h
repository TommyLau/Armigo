// ============================================================================
// Armigo Diagnostics - Host Diagnostics
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace armigo::diag {

// Increasing verbosity. A message is emitted when its level is at or below
// the process minimum. Error is always recorded: the minimum cannot hide it.
enum class Level : int {
    error = 0,
    warning = 1,
    info = 2,
    trace = 3,
};

namespace detail {
inline std::atomic<int> minimum_level{static_cast<int>(Level::info)};
inline std::atomic<int> frame_stats{1};
inline std::atomic<bool> player_quiet{false};
} // namespace detail

// One relaxed load. A disabled call still evaluates its arguments, so callers
// that build strings check enabled() or frame_stats_enabled() first.
[[nodiscard]] inline bool enabled(Level level) noexcept {
    return (level == Level::error || !detail::player_quiet.load(std::memory_order_relaxed)) &&
           static_cast<int>(level) <= detail::minimum_level.load(std::memory_order_relaxed);
}

[[nodiscard]] inline bool frame_stats_enabled() noexcept {
    return !detail::player_quiet.load(std::memory_order_relaxed) &&
           detail::frame_stats.load(std::memory_order_relaxed) != 0;
}

[[nodiscard]] inline Level minimum_level() noexcept {
    return static_cast<Level>(detail::minimum_level.load(std::memory_order_relaxed));
}

// Printf-style. Each finished line (through a newline, or an explicit
// newline() after fragments) is one thread-safe write. Fragments without a
// newline stay on the calling thread until the line is finished.
void write(Level level, const char* format, ...) noexcept __attribute__((format(printf, 2, 3)));
void error(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));
void warning(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));
void info(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));
void trace(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));

// Per-frame counters. Independent of the level. Headless leaves this on.
void frame(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));

// An existing opt-in switch (ARMIGO_TRACE_*, ARMIGO_NVN_COUNT, and the other
// *_LOG gates) has already decided to emit. The line is written even when the
// process minimum is below trace, and it is not copied into the last-run log.
void opt_in(const char* format, ...) noexcept __attribute__((format(printf, 1, 2)));

// Finishes a line started by earlier fragments on this thread. Does nothing
// when those fragments were filtered out.
void newline() noexcept;

// Development tools, including headless: info and frame stats, unless the
// environment says otherwise. This is the process default.
void use_development_defaults() noexcept;

// Ordinary players emit errors only and never create diagnostic files.
// Explicit development tooling may opt into environment-controlled diagnostics.
void use_player_defaults(bool development_diagnostics = false) noexcept;

// Truncates `path` on the first open of a launch and appends warnings and
// errors until 1 MiB. A second open of the same path in this process keeps
// the file already opened. Info, trace, frame stats and opt-in lines stay
// on the standard error stream only.
bool open_last_run_log(const char* path) noexcept;

[[nodiscard]] std::optional<Level> parse_level(std::string_view text) noexcept;
[[nodiscard]] std::optional<bool> parse_frame_stats(std::string_view text) noexcept;

// Test seams. The sink replaces the standard error stream and must not log.
using TestingSink = void (*)(Level level, std::string_view line, void* context);
void testing_reset() noexcept;
void testing_set_level(Level level) noexcept;
void testing_set_frame_stats(bool enabled) noexcept;
void testing_set_sink(TestingSink sink, void* context) noexcept;
void testing_close_log() noexcept;
[[nodiscard]] std::uint64_t testing_last_run_bytes() noexcept;
// Times an exit-time write used the buffer that outlives the main thread's
// thread_local line state. Zero while that state is still alive.
[[nodiscard]] std::uint64_t testing_exit_pending_appends() noexcept;

} // namespace armigo::diag
