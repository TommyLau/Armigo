// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "thread_lifecycle.h"
#include <armigo/horizon/startup_queries.h>
#include <bit>

namespace armigo::horizon {
namespace {
bool contains(std::uint64_t mask, std::int32_t core) noexcept {
    return core >= 0 && core < 64 && (mask & (std::uint64_t{1} << core));
}
}

ThreadCoreRequest resolve_thread_creation_core(std::int32_t requested,
    std::int32_t process_default, std::uint64_t allowed) noexcept {
    const auto core = requested == -2 ? process_default : requested;
    if (!contains(allowed, core)) return {0xe201};
    return {0, core, std::uint64_t{1} << core};
}

ThreadCoreRequest resolve_thread_core_mask(std::int32_t requested, std::uint64_t mask,
    std::int32_t previous_ideal, std::int32_t process_default,
    std::uint64_t allowed) noexcept {
    if (requested == -2) return resolve_thread_creation_core(-2, process_default, allowed);
    if (mask & ~allowed) return {0xe201};
    if (!mask) return {0xe801};
    if (requested >= 0) {
        if (requested >= 64) return {0xe201};
        if (!contains(mask, requested)) return {0xe801};
    } else if (requested == -3) {
        requested = previous_ideal;
        if (requested >= 0 && !contains(mask, requested)) return {0xe801};
    } else if (requested != -1) {
        return {0xe201};
    }
    return {0, requested, mask};
}

std::optional<std::uint32_t> select_virtual_processor(std::int32_t ideal,
    std::uint64_t mask, std::optional<std::uint32_t> current) noexcept {
    if (!mask) return {};
    if (ideal >= 0) {
        if (!contains(mask, ideal)) return {};
        return static_cast<std::uint32_t>(ideal);
    }
    if (ideal != -1) return {};
    if (current && *current < 64 && contains(mask, static_cast<std::int32_t>(*current)))
        return current;
    return static_cast<std::uint32_t>(std::countr_zero(mask));
}

ThreadSchedulingState ThreadRecord::scheduling_state() const noexcept {
    const std::lock_guard lock(scheduling_mutex_);
    return {ideal_core_, affinity_, current_processor_};
}

bool ThreadRecord::configure_scheduling(std::int32_t ideal, std::uint64_t mask,
                                       std::optional<std::uint32_t> processor) noexcept {
    if (!mask || (mask & ~process_->configuration().allowed_core_mask) ||
        (ideal != -1 && !contains(mask, ideal)) ||
        (processor && (*processor >= 64 || !contains(mask, static_cast<std::int32_t>(*processor))))) return false;
    const std::lock_guard lock(scheduling_mutex_);
    ideal_core_ = ideal;
    affinity_ = mask;
    current_processor_ = processor;
    return true;
}

void ThreadRecord::cancel_synchronization() noexcept {
    const std::lock_guard lock(guest_wait_mutex_);
    if (guest_wait_cancel_ && guest_wait_cancel_(guest_wait_context_)) {
        guest_cancellation_pending_ = false;
    } else {
        guest_cancellation_pending_ = true;
    }
}

bool ThreadRecord::synchronization_wait_active() const noexcept {
    const std::lock_guard lock(guest_wait_mutex_);
    return guest_wait_cancel_ != nullptr;
}

} // namespace armigo::horizon
