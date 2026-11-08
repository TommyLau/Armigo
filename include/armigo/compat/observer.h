// ============================================================================
// Armigo Compatibility - Guest Instruction Observers
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/call_hook.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::horizon { class GuestBuffers; }

namespace armigo::compat {

// A debugging aid, never a title setting: an instruction of the main module
// that, just before it executes, logs register and guest-memory values and
// then runs unchanged. ARMIGO_OBSERVE enables it for a load-time prepared
// launch only:
//
//   ARMIGO_OBSERVE='<offset> [if <expression>] [: <item>, <item>...]; ...'
//
// An item is an expression, "s:<expression>" (the C string at that address)
// or "regs" (x0..x30 and sp). Expressions use numbers, x0..x30, w0..w30, sp,
// [e] (u64 at e), w[e], h[e], b[e] (u32, u16, u8), parentheses, unary ! ~ -,
// and * + - << >> & ^ | == != < > (in that order of precedence, all unsigned).
// A read the guest cannot make makes the expression unknown; an unknown
// condition is false. Each point logs at most `limit` hits.
struct ObserveNode {
    enum class Kind : std::uint8_t { constant, x, w, sp, load, unary, binary };
    Kind kind = Kind::constant;
    // Unary: '!', '~', '-'. Binary: '*', '+', '-', 'l' (<<), 'r' (>>), '&',
    // '^', '|', 'e' (==), 'n' (!=), '<', '>'.
    char op = 0;
    std::uint8_t width = 8; // Bytes a load reads.
    std::uint64_t value = 0; // A constant or a register number.
    std::int32_t left = -1, right = -1;
};
struct ObserveExpression {
    std::vector<ObserveNode> nodes;
    std::int32_t root = -1;
};
struct ObserveItem {
    std::string text;
    ObserveExpression expression;
    bool string = false;
    bool registers = false;
};
struct ObservePoint {
    std::uint64_t offset = 0;
    // The module's word at offset, set when the launch selects the point.
    std::uint32_t expected_instruction = 0;
    std::string text;
    std::optional<ObserveExpression> condition;
    std::vector<ObserveItem> items;
    std::uint32_t limit = 64;
    std::shared_ptr<std::atomic<std::uint32_t>> hits = std::make_shared<std::atomic<std::uint32_t>>(0);
};

// What an expression reads: guest registers as they are before the observed
// instruction, and guest memory.
class ObserveState {
public:
    virtual ~ObserveState() = default;
    virtual std::uint64_t x(unsigned index) const noexcept = 0; // 0..30
    virtual std::uint64_t sp() const noexcept = 0;
    virtual bool read(std::uint64_t address, std::span<std::byte> out) const noexcept = 0;
};

std::variant<std::vector<ObservePoint>, std::string> parse_observe_points(std::string_view text);
std::optional<std::uint64_t> evaluate(const ObserveExpression& expression, const ObserveState& state) noexcept;
// The logged values of one hit, or nothing when the condition is false.
std::optional<std::string> describe_hit(const ObservePoint& point, const ObserveState& state);

// The observed instruction becomes a branch to a stub that saves x30, calls
// the gate, restores x30 and then runs the displaced instruction: a copy of
// it, or for TBZ, TBNZ, CBZ, CBNZ and B.cond the same test against a nearby
// branch to the original target. Other PC-relative instructions, branches,
// exclusive accesses and system instructions are refused.
inline constexpr std::size_t observe_stub_bytes = 24;
struct ObserveHook {
    std::uint32_t replacement_instruction = 0;
    std::array<std::uint32_t, observe_stub_bytes / 4> trampoline{};
    execution::CallHookSite site;
};
std::variant<ObserveHook, std::string> make_observe_hook(std::uint32_t instruction, std::uint64_t address,
                                                         std::uint64_t trampoline,
                                                         execution::CallHookCodeRegion gate) noexcept;

// Logs one hit of a point reached through its gate.
void observe_hit(const ObservePoint& point, const execution::GateCall& call, const horizon::GuestBuffers& memory) noexcept;

} // namespace armigo::compat
