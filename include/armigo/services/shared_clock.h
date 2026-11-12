// ============================================================================
// Armigo Services - Shared Clock Initialization
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace armigo::services::shared_clock {

inline constexpr std::size_t page_size = 0x1000;
inline constexpr std::size_t backing_alignment = 8;
using SourceId = std::array<std::byte, 16>;

struct SystemClockContext {
    std::int64_t offset_seconds = 0;
    std::int64_t steady_seconds = 0;
    SourceId source_id{};
};

struct Configuration {
    // A sample of the guest counter timeline in nanoseconds (the Switch 1
    // 19.2 MHz and Switch 2 31.25 MHz counters share its origin). This is not
    // host wall time.
    std::int64_t counter_nanoseconds = 0;
    std::int64_t steady_offset_nanoseconds = 0;
    // Nonzero source identity for this initialized clock timeline. Requiring
    // it here lets an absent network context remain explicitly unavailable.
    SourceId steady_source_id{};
    // Reused only when its source matches the current steady source.
    std::optional<SystemClockContext> local_context;
    std::int64_t fallback_wall_seconds = 0;
    // nullopt serializes an unavailable zero context. A supplied context is
    // preserved, including a different source; this API does not synchronize
    // a network clock or assert that a caller's synchronization is genuine.
    std::optional<SystemClockContext> network_context;
    bool automatic_correction_enabled = false;
};

enum class InitializeResult {
    initialized,
    invalid_backing,
    invalid_source,
    time_out_of_range,
};

// Constructs the firmware startup page in caller-owned storage. Requires
// exactly page_size bytes, backing_alignment alignment and exclusive access
// before any guest/host reader can observe the page. Validation failures leave
// all bytes unchanged. No pointers, handles, threads or ownership are retained.
//
// After success, hand off the page through the caller's synchronization and
// expose a read-only guest mapping. Keep the backing alive for every mapping.
// Fixed offsets combine with advancing guest ticks; there are no page updates
// or continuous-adjustment algorithms here. Do not call this on a live page.
InitializeResult initialize(std::span<std::byte> backing,
                            const Configuration& configuration) noexcept;

}  // namespace armigo::services::shared_clock
