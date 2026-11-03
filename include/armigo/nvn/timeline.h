// ============================================================================
// Armigo NVN - Frame Timeline Trace
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>

namespace armigo::nvn {

// A frame's wall time is the sum of work that overlaps and work that does not:
// the GPU's own bus, the encode thread and the guest threads waiting for a
// fence. When the GPU is idle inside a frame, something serialized. This trace
// records the boundaries of every phase on one clock (nanoseconds) so the
// overlap can be read from the numbers instead of guessed.
//
// Off unless ARMIGO_NVN_TIMELINE names a file; writes one tab-separated record
// per event: time_ns, event, key, value.
[[nodiscard]] bool timeline_enabled() noexcept;
void timeline_event(const char* event, std::uint64_t key, std::uint64_t value) noexcept;
// Records an event that happened at time_ns on the same clock as timeline_now_ns.
void timeline_event_at(std::uint64_t time_ns, const char* event, std::uint64_t key,
                       std::uint64_t value) noexcept;
// Steady-clock nanoseconds, the clock every timeline record uses.
[[nodiscard]] std::uint64_t timeline_now_ns() noexcept;

} // namespace armigo::nvn
