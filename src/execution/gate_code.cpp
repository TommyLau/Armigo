// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate.h>
#include <armigo/execution/gate_code.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

extern "C" {
const armigo::execution::detail::GateExitTarget*
armigo_execution_gate_dispatch_on_host_stack(armigo::execution::GateFrame*) noexcept;
extern const std::byte armigo_execution_gate_template_begin[];
extern const std::byte armigo_execution_gate_template_dispatcher[];
extern const std::byte armigo_execution_gate_template_end[];
}

namespace armigo::execution {
namespace {

struct LinkedGateTemplate {
    std::size_t size = 0;
    std::size_t dispatcher_offset = 0;
};

LinkedGateTemplate linked_gate_template() {
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_execution_gate_template_begin);
    const auto binding =
        reinterpret_cast<std::uintptr_t>(armigo_execution_gate_template_dispatcher);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_execution_gate_template_end);
    if (end <= begin || end - begin > 65536 || binding < begin || binding > end ||
        end - binding != sizeof(std::uintptr_t) || begin % 8 != 0 || binding % 8 != 0) {
        throw std::logic_error("The native gate template has an invalid linked layout.");
    }
    return {end - begin, binding - begin};
}

} // namespace

LocalGateTemplate make_local_gate_template() {
    static_assert(sizeof(std::uintptr_t) == 8);
    const auto layout = linked_gate_template();
    LocalGateTemplate result{std::vector<std::byte>(layout.size), layout.dispatcher_offset};
    std::memcpy(result.code.data(), armigo_execution_gate_template_begin, result.code.size());
    std::fill_n(result.code.data() + result.dispatcher_offset, sizeof(std::uintptr_t),
                std::byte{0});
    return result;
}

bool bind_local_gate_dispatcher(std::span<std::byte> bytes, std::size_t dispatcher_offset,
                                std::uintptr_t dispatcher) noexcept {
    if (dispatcher == 0 || dispatcher_offset % alignof(std::uintptr_t) != 0 ||
        dispatcher_offset > bytes.size() || bytes.size() - dispatcher_offset < sizeof(dispatcher)) {
        return false;
    }
    std::memcpy(bytes.data() + dispatcher_offset, &dispatcher, sizeof(dispatcher));
    return true;
}

std::uintptr_t gate_dispatcher_address() noexcept {
    return reinterpret_cast<std::uintptr_t>(&armigo_execution_gate_dispatch_on_host_stack);
}

std::vector<std::byte> make_local_gate_code() {
    auto result = make_local_gate_template();
    if (!bind_local_gate_dispatcher(result.code, result.dispatcher_offset,
                                    gate_dispatcher_address())) {
        throw std::logic_error("The native gate template dispatcher field is invalid.");
    }
    return std::move(result.code);
}

} // namespace armigo::execution
