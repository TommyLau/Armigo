// ============================================================================
// Armigo NVN - Presentation and Synchronization Helpers
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "objects.h"

#include <cstdint>
#include <functional>
#include <stop_token>

namespace armigo::nvn {

using BlockingWaitHook = bool (*)(const std::function<bool()> &wait);

void set_blocking_wait_hook(BlockingWaitHook hook) noexcept;
[[nodiscard]] bool blocking_wait(const std::function<bool()> &wait,
                                  const char *tag = "wait");

[[nodiscard]] bool queue_valid(const NVNqueue *queue) noexcept;
[[nodiscard]] std::uint64_t queue_last_fence(const NVNqueue *queue) noexcept;
void queue_error(NVNqueue *queue, const char *message) noexcept;

[[nodiscard]] std::uint64_t
timestamp_ticks_from_nanoseconds(std::uint64_t nanoseconds) noexcept;
[[nodiscard]] std::uint64_t
timestamp_nanoseconds_from_ticks(std::uint64_t ticks) noexcept;
[[nodiscard]] std::uint64_t current_timestamp_ticks() noexcept;

// Replay uses the same atomic pool-memory operations as CPU event access.
[[nodiscard]] bool event_signal_value(const EventState &event,
                                      NVNeventSignalMode mode,
                                      std::uint32_t value) noexcept;
[[nodiscard]] bool event_wait_value(const EventState &event,
                                    NVNeventWaitMode mode, std::uint32_t value,
                                    std::uint64_t timeout_ns,
                                    std::stop_token stop = {});

void register_queue_entries() noexcept;
void register_sync_entries() noexcept;

} // namespace armigo::nvn
