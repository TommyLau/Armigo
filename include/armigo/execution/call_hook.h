// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/gate.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace armigo::execution {

class GuestThreadContext;

struct CallHookCodeRegion {
    std::uint64_t address = 0;
    std::uint64_t size = 0;
};

// One direct guest call redirected through a host callback. return_address is
// the gate call's stub-local identity. call_return_address and target_address
// retain the architectural values of the original BL.
struct CallHookSite {
    std::uint64_t return_address = 0;
    std::uint64_t call_return_address = 0;
    std::uint64_t target_address = 0;
};

enum class CallHookTableErrorKind : std::uint8_t {
    none,
    too_many_sites,
    null_return_address,
    misaligned_return_address,
    invalid_call_return_address,
    invalid_target_address,
    duplicate_return_address,
};

const char* describe(CallHookTableErrorKind kind) noexcept;

struct CallHookTableError {
    CallHookTableErrorKind kind = CallHookTableErrorKind::none;
    std::size_t site_index = 0;
    std::uint64_t return_address = 0;
};

class CallHookTable;

struct CallHookTableBuild {
    std::shared_ptr<const CallHookTable> table;
    CallHookTableError error;
    bool ok() const noexcept {
        return table != nullptr && error.kind == CallHookTableErrorKind::none;
    }
};

// Immutable map keyed by the return address produced by the stub's gate call.
class CallHookTable {
public:
    static CallHookTableBuild build(std::span<const CallHookSite> sites,
                                    std::size_t max_sites = std::size_t{1} << 20);

    const CallHookSite* find(std::uint64_t return_address) const noexcept;
    std::span<const CallHookSite> sites() const noexcept { return sites_; }

    CallHookTable(const CallHookTable&) = delete;
    CallHookTable& operator=(const CallHookTable&) = delete;

private:
    explicit CallHookTable(std::vector<CallHookSite> sites) : sites_(std::move(sites)) {}

    std::vector<CallHookSite> sites_;
};

// The callback runs synchronously on the guest's host thread before the original
// callee. GateCall::x(30) is the gate identity and GateCall::sp() is the guest
// stack pointer minus CallHookPatch::stack_adjustment. The site's
// call_return_address is the original guest x30 value that the callee receives.
// A callback must not change x30 or sp. Other accepted GateCall changes are
// applied before control transfers to the original callee.
using CallHookHandler = GateStatus (*)(const CallHookSite& site, GateCall& call,
                                      GuestThreadContext& thread, void* context);

struct CallHook {
    CallHookHandler handler = nullptr;
    void* context = nullptr;
};

enum class CallHookPatchError : std::uint8_t {
    none,
    invalid_layout,
    overlapping_regions,
    unsupported_site,
    unreachable_branch,
};

const char* describe(CallHookPatchError error) noexcept;

struct CallHookPatch {
    std::uint64_t instruction_address = 0;
    std::uint64_t trampoline_address = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
    std::array<std::uint32_t, 4> trampoline{};
    CallHookSite site;

    static constexpr std::uint64_t stack_adjustment = 16;
};

struct CallHookPatchResult {
    CallHookPatchError error = CallHookPatchError::none;
    std::optional<CallHookPatch> patch;
    bool ok() const noexcept { return patch.has_value(); }
};

// Redirects one direct BL through a local gate and then to its original target.
// The replacement remains a BL, so x30 already contains the original site+4 on
// entry to the stub. The stub preserves that value around the gate and tail
// branches to the original callee, which therefore observes the exact guest LR.
// All addresses must share one namespace. The site and its continuation, stub
// and gate must be pairwise disjoint, and every generated branch must be in range.
CallHookPatchResult make_call_hook_patch(std::uint32_t instruction,
                                        std::uint64_t instruction_address,
                                        std::uint64_t trampoline_address,
                                        CallHookCodeRegion gate) noexcept;

} // namespace armigo::execution
