// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/guest_timeline.h>

#include <mutex>

namespace armigo::execution {
namespace {

std::mutex& publication_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

std::optional<GuestTimelineSource>& published() noexcept {
    static std::optional<GuestTimelineSource> source;
    return source;
}

} // namespace

std::optional<std::chrono::nanoseconds> guest_timeline_now() noexcept {
    const std::lock_guard lock(publication_mutex());
    const auto& source = published();
    if (!source || !source->elapsed) return std::nullopt;
    return source->elapsed(source->context);
}

std::optional<std::chrono::steady_clock::time_point> guest_timeline_origin() noexcept {
    const std::lock_guard lock(publication_mutex());
    const auto& source = published();
    if (!source) return std::nullopt;
    return source->origin;
}

void publish_guest_timeline(GuestTimelineSource source) noexcept {
    const std::lock_guard lock(publication_mutex());
    published() = source;
}

void withdraw_guest_timeline(const void* context) noexcept {
    const std::lock_guard lock(publication_mutex());
    if (published() && published()->context == context) published().reset();
}

} // namespace armigo::execution
