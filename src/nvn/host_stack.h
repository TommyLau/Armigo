// ============================================================================
// Armigo NVN - Host Stack Calls
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Direct entry points run on the calling guest thread's stack. The game
// allocates those stacks itself, usually from its heap without a guard page,
// and sizes them for its own driver, so host work that needs deep stacks
// (backend calls into Metal, locks, diagnostics) switches to the host stack
// the guest thread entered guest code from.

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

extern "C" void armigo_nvn_call_on_stack(void* context, void (*function)(void*), std::uintptr_t stack_top) noexcept;

namespace armigo::nvn {

// Returns a 16-byte aligned host stack top the calling thread may use below,
// or 0 to stay on the current stack.
using HostStackProvider = std::uintptr_t (*)() noexcept;
void set_host_stack_provider(HostStackProvider provider) noexcept;
[[nodiscard]] std::uintptr_t host_stack_top() noexcept;

// Runs body() on the host stack when one is available.
template <class Body>
decltype(auto) on_host_stack(Body&& body) {
    const std::uintptr_t top = host_stack_top();
    if (!top) return body();
    using Result = decltype(body());
    if constexpr (std::is_void_v<Result>) {
        armigo_nvn_call_on_stack(&body, [](void* context) { (*static_cast<std::remove_reference_t<Body>*>(context))(); },
                                 top);
    } else {
        struct Call {
            std::remove_reference_t<Body>* body;
            Result result{};
        } call{&body};
        armigo_nvn_call_on_stack(&call, [](void* context) {
            auto* c = static_cast<Call*>(context);
            c->result = (*c->body)();
        }, top);
        return call.result;
    }
}

} // namespace armigo::nvn
