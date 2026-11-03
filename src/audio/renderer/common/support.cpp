// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/common/support.h"

#include <armigo/common/diagnostics.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace armigo::audio::renderer {
namespace {

bool trace_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("ARMIGO_AUDIO_TRACE");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

std::atomic<u32> logged_lines{0};
constexpr u32 max_logged_lines = 256;

}  // namespace

u64 host_time_ns() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

bool log_enabled(LogLevel level) noexcept {
    if (trace_enabled()) return true;
    if (level < LogLevel::warning) return false;
    return logged_lines.load(std::memory_order_relaxed) < max_logged_lines;
}

void log_message(LogLevel level, const std::string& text) noexcept {
    const auto line = logged_lines.fetch_add(1, std::memory_order_relaxed);
    if (!trace_enabled() && line >= max_logged_lines) return;
    // Info and trace are reached only when ARMIGO_AUDIO_TRACE is on (log_enabled).
    // opt_in keeps those lines visible without copying them into last-run.log.
    if (level >= LogLevel::error)
        armigo::diag::error("[audio renderer] error: %s\n", text.c_str());
    else if (level == LogLevel::warning)
        armigo::diag::warning("[audio renderer] warning: %s\n", text.c_str());
    else
        armigo::diag::opt_in("[audio renderer] info: %s\n", text.c_str());
    if (!trace_enabled() && line + 1 == max_logged_lines)
        armigo::diag::warning("[audio renderer] further diagnostics are suppressed (ARMIGO_AUDIO_TRACE=1 shows all)\n");
}

}  // namespace armigo::audio::renderer
