// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "guest_wait.h"
#include <array>
#include <exception>

namespace armigo::nvn {
namespace {
thread_local GuestWaitScope* current_wait = nullptr;
}
GuestWaitScope::GuestWaitScope(horizon::HandleTable& objects, horizon::ThreadRecord& caller,
    GuestWaitStopPolicy stop_policy, void* context) noexcept
    : objects_(objects), caller_(caller), stop_policy_(stop_policy), context_(context),
      previous_(current_wait) {
    current_wait = this;
}
GuestWaitScope::~GuestWaitScope() {
    if (current_wait != this) std::terminate();
    current_wait = previous_;
}
horizon::EventResult wait_guest_event(horizon::Handle event, std::int64_t timeout) noexcept {
    auto* scope = current_wait;
    if (!scope || !scope->stop_policy_)
        return {horizon::EventDisposition::unsupported, 0, {}, "NVN guest wait has no runtime binding."};
    try {
        const std::array handles{event};
        return horizon::wait_event(scope->objects_, handles, timeout, scope->caller_,
            scope->stop_policy_(scope->context_));
    } catch (...) {
        return {horizon::EventDisposition::unsupported, 0, {}, "NVN guest wait failed in the host runtime."};
    }
}
} // namespace armigo::nvn
