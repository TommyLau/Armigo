// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/event.h>

namespace armigo::nvn {

// The runtime supplies a policy which defers host cancellation while the guest
// owns an observed mutex grant. It is queried immediately before enrollment.
using GuestWaitStopPolicy = std::stop_token (*)(void*) noexcept;
class GuestWaitScope {
public:
    GuestWaitScope(horizon::HandleTable& objects, horizon::ThreadRecord& caller,
                   GuestWaitStopPolicy stop_policy, void* context) noexcept;
    ~GuestWaitScope();
    GuestWaitScope(const GuestWaitScope&) = delete;
    GuestWaitScope& operator=(const GuestWaitScope&) = delete;
private:
    friend horizon::EventResult wait_guest_event(horizon::Handle, std::int64_t) noexcept;
    horizon::HandleTable& objects_;
    horizon::ThreadRecord& caller_;
    GuestWaitStopPolicy stop_policy_;
    void* context_;
    GuestWaitScope* previous_;
};

// Host cancellation is a distinct disposition, never successful completion.
// Bind on the guest's host worker for the full direct-call lifetime. The caller,
// table and policy context must outlive the scope. Unbound calls fail explicitly.
[[nodiscard]] horizon::EventResult wait_guest_event(horizon::Handle event,
    std::int64_t timeout_nanoseconds) noexcept;

} // namespace armigo::nvn
