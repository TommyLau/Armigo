// ============================================================================
// Armigo Shader - Shader Pipeline
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture handles a stage computes from constant-buffer words and a word of
// guest memory, the same value in every invocation of a draw: the host
// computes them at draw time. A handle program is a stack machine over 32-bit
// words; operands follow the operation code, and operations pop their right
// operand first. A texture or sampler whose key has the slot
// handle_program_slot takes its handle from program handle_byte_offset / 4
// of the stage's ResourcePlan::handle_programs instead of a constant-buffer
// word.

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace armigo::shader::pipeline {
inline constexpr std::uint8_t handle_program_slot = 31;
// A texture or sampler keyed at texture_pool_slot is read from a pool table
// by the handle word's IDs at run time (the project documentation); its
// resource, not its key, names the pool.
inline constexpr std::uint8_t texture_pool_slot = 30;

enum class HandleOp : std::uint32_t {
    immediate,         // value
    cbuf,              // slot, byte offset: the constant-buffer word
    cbuf_dynamic,      // slot: the word at the popped byte offset
    guest,             // byte displacement: the guest word at the popped high, then low, address words
    add, subtract, multiply, bit_and, bit_or, bit_xor, shift_left, shift_right, shift_right_signed,
    minimum, maximum, minimum_signed, maximum_signed,
    equal, not_equal, less, less_equal, greater, greater_equal,
    less_signed, less_equal_signed, greater_signed, greater_equal_signed,
    logical_and, logical_or, logical_not, bit_not,
    select,            // condition, true value, false value
    keep,              // slot: keeps the top value in the slot (shared subexpressions)
    recall,            // slot: the value kept there
};
inline constexpr std::size_t handle_program_slots = 64;

// Runs a handle program. read_cbuf(slot, byte_offset) and read_guest(address)
// return the words it reads (zero where nothing is there). Returns nothing
// for a malformed program.
template <class ReadCbuf, class ReadGuest>
std::optional<std::uint32_t> evaluate_handle_program(std::span<const std::uint32_t> code, ReadCbuf&& read_cbuf, ReadGuest&& read_guest) {
    std::array<std::uint32_t, 64> stack{};
    std::array<std::uint32_t, handle_program_slots> slots{};
    std::size_t size = 0;
    bool malformed = false;
    const auto pop = [&]() -> std::uint32_t {
        if (size == 0) { malformed = true; return 0; }
        return stack[--size];
    };
    const auto push = [&](std::uint32_t value) {
        if (size == stack.size()) { malformed = true; return; }
        stack[size++] = value;
    };
    for (std::size_t pc = 0; pc < code.size() && !malformed;) {
        const auto op = static_cast<HandleOp>(code[pc++]);
        const auto operand = [&]() -> std::uint32_t {
            if (pc >= code.size()) { malformed = true; return 0; }
            return code[pc++];
        };
        switch (op) {
        case HandleOp::immediate: push(operand()); continue;
        case HandleOp::cbuf: { const auto slot = operand(); push(read_cbuf(slot, operand())); continue; }
        case HandleOp::cbuf_dynamic: { const auto slot = operand(); push(read_cbuf(slot, pop())); continue; }
        case HandleOp::guest: {
            const auto displacement = static_cast<std::int32_t>(operand());
            const auto high = pop(), low = pop();
            push(read_guest(((std::uint64_t{high} << 32) | low) + static_cast<std::uint64_t>(std::int64_t{displacement})));
            continue;
        }
        case HandleOp::logical_not: push(pop() == 0 ? 1u : 0u); continue;
        case HandleOp::bit_not: push(~pop()); continue;
        case HandleOp::select: {
            const auto no = pop(), yes = pop(), condition = pop();
            push(condition != 0 ? yes : no);
            continue;
        }
        case HandleOp::keep: {
            const auto slot = operand();
            if (slot >= slots.size() || size == 0) { malformed = true; continue; }
            slots[slot] = stack[size - 1];
            continue;
        }
        case HandleOp::recall: {
            const auto slot = operand();
            if (slot >= slots.size()) { malformed = true; continue; }
            push(slots[slot]);
            continue;
        }
        default: break;
        }
        const auto b = pop(), a = pop();
        const auto sa = static_cast<std::int32_t>(a), sb = static_cast<std::int32_t>(b);
        switch (op) {
        case HandleOp::add: push(a + b); break;
        case HandleOp::subtract: push(a - b); break;
        case HandleOp::multiply: push(a * b); break;
        case HandleOp::bit_and: push(a & b); break;
        case HandleOp::bit_or: push(a | b); break;
        case HandleOp::bit_xor: push(a ^ b); break;
        // Counts wrap to 0-31, as the IR's shifts.
        case HandleOp::shift_left: push(a << (b & 31u)); break;
        case HandleOp::shift_right: push(a >> (b & 31u)); break;
        case HandleOp::shift_right_signed: push(static_cast<std::uint32_t>(sa >> (b & 31u))); break;
        case HandleOp::minimum: push(a < b ? a : b); break;
        case HandleOp::maximum: push(a > b ? a : b); break;
        case HandleOp::minimum_signed: push(static_cast<std::uint32_t>(sa < sb ? sa : sb)); break;
        case HandleOp::maximum_signed: push(static_cast<std::uint32_t>(sa > sb ? sa : sb)); break;
        case HandleOp::equal: push(a == b); break;
        case HandleOp::not_equal: push(a != b); break;
        case HandleOp::less: push(a < b); break;
        case HandleOp::less_equal: push(a <= b); break;
        case HandleOp::greater: push(a > b); break;
        case HandleOp::greater_equal: push(a >= b); break;
        case HandleOp::less_signed: push(sa < sb); break;
        case HandleOp::less_equal_signed: push(sa <= sb); break;
        case HandleOp::greater_signed: push(sa > sb); break;
        case HandleOp::greater_equal_signed: push(sa >= sb); break;
        case HandleOp::logical_and: push(a != 0 && b != 0); break;
        case HandleOp::logical_or: push(a != 0 || b != 0); break;
        default: malformed = true; break;
        }
    }
    if (malformed || size != 1) return std::nullopt;
    return stack[0];
}
} // namespace armigo::shader::pipeline
