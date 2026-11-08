// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The running process's guest clock timeline for subsystems below the runtime,
// such as the NVN driver, which put GPU times on the same clock the guest
// counter reads. The runtime publishes its clock while a process lives; both
// queries are empty otherwise.

#pragma once

#include <chrono>
#include <optional>

namespace armigo::execution {

struct GuestTimelineSource {
    // Nanoseconds since the clock's origin, the timeline the guest counter
    // reads, or nullopt when the clock cannot sample. Called under the
    // publication lock; it must not publish or withdraw a timeline.
    std::optional<std::chrono::nanoseconds> (*elapsed)(void* context) noexcept = nullptr;
    void* context = nullptr;
    // The origin on the host steady clock.
    std::chrono::steady_clock::time_point origin{};
};

// Nanoseconds since the published clock's origin.
std::optional<std::chrono::nanoseconds> guest_timeline_now() noexcept;
// The published clock's origin on the host steady clock.
std::optional<std::chrono::steady_clock::time_point> guest_timeline_origin() noexcept;

// Runtime only. Publishing replaces any earlier source. Withdrawing removes
// the source only while it is the one published for this context, and returns
// after any query still reading it.
void publish_guest_timeline(GuestTimelineSource source) noexcept;
void withdraw_guest_timeline(const void* context) noexcept;

} // namespace armigo::execution
