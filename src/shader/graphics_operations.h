// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Private registry of pure graphics operations. Each kind lives in its own
// translation unit under operations/ and registers one immutable entry in
// graphics_operations.cpp. The validators, the CPU evaluator, and both MSL
// emitters dispatch through the entry, so a new kind never edits them.
#pragma once

#include <armigo/shader/graphics.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace armigo::shader::graphics::detail {

// Operand words in signature order. A Raw32 operand is its raw word; a
// Predicate operand is 1 for true and 0 for false. Unused slots are 0.
using OperationWords = std::array<std::uint32_t, 3>;
// MSL operand expressions in signature order: a uint expression for a Raw32
// operand and a bool expression for a Predicate operand. Unused slots are
// empty.
using OperationText = std::array<std::string, 3>;

struct OperationEntry {
    PureKind kind;
    std::uint8_t arity;
    // Raw32 or Predicate only; a sample tuple is never a pure operand.
    std::array<ValueType, 3> operands;
    ValueType result;
    // Accepts the kind-specific immediate; false is an invalid node.
    bool (*admits)(std::uint32_t immediate) noexcept;
    // Bit-exact CPU reference. A Predicate result is nonzero for true. It
    // runs inside the evaluator's floating scope (round to nearest, no
    // traps) and must not change floating control state.
    std::uint32_t (*evaluate)(const OperationWords& operands,
                              std::uint32_t immediate) noexcept;
    // MSL expression of the result type: uint for Raw32, bool for
    // Predicate. It may call only standard MSL or this kind's helper.
    std::string (*emit)(const OperationText& operands, std::uint32_t immediate);
    // MSL helper definitions emitted once per program that uses the kind,
    // or null when the expression needs none. Programs that do not use a
    // kind keep their source bytes unchanged.
    std::string_view (*helper)() noexcept;
};

// Immutable, append-only table of registered pure operations.
[[nodiscard]] std::span<const OperationEntry> registered_operations() noexcept;

// Registered entry for the kind, or null.
[[nodiscard]] const OperationEntry* find_operation(PureKind kind) noexcept;

// Shared admission for kinds that carry no immediate.
[[nodiscard]] bool admits_no_immediate(std::uint32_t immediate) noexcept;

// Unit definitions; each is defined beside its kind and declared here so the
// table stays the only registration point.
std::uint32_t evaluate_float_negate(const OperationWords& operands,
                                    std::uint32_t immediate) noexcept;
std::string emit_float_negate(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_float_absolute(const OperationWords& operands,
                                      std::uint32_t immediate) noexcept;
std::string emit_float_absolute(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_select(const OperationWords& operands,
                              std::uint32_t immediate) noexcept;
std::string emit_select(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_predicate_and(const OperationWords& operands,
                                     std::uint32_t immediate) noexcept;
std::string emit_predicate_and(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_predicate_or(const OperationWords& operands,
                                    std::uint32_t immediate) noexcept;
std::string emit_predicate_or(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_predicate_xor(const OperationWords& operands,
                                     std::uint32_t immediate) noexcept;
std::string emit_predicate_xor(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_predicate_select(const OperationWords& operands,
                                        std::uint32_t immediate) noexcept;
std::string emit_predicate_select(const OperationText& operands, std::uint32_t immediate);
bool admits_word_logic(std::uint32_t immediate) noexcept;
std::uint32_t evaluate_word_logic(const OperationWords& operands,
                                  std::uint32_t immediate) noexcept;
std::string emit_word_logic(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_word_is_zero(const OperationWords& operands,
                                    std::uint32_t immediate) noexcept;
std::string emit_word_is_zero(const OperationText& operands, std::uint32_t immediate);
std::uint32_t evaluate_float_minimum(const OperationWords& operands,
                                     std::uint32_t immediate) noexcept;
std::string emit_float_minimum(const OperationText& operands, std::uint32_t immediate);
std::string_view helper_float_minimum() noexcept;
std::uint32_t evaluate_float_maximum(const OperationWords& operands,
                                     std::uint32_t immediate) noexcept;
std::string emit_float_maximum(const OperationText& operands, std::uint32_t immediate);
std::string_view helper_float_maximum() noexcept;

// Float-to-integer conversion of one binary32 word; the immediate encodes
// the admitted profile (signedness, 16/32-bit destination, rounding mode).
bool admits_float_to_integer(std::uint32_t immediate) noexcept;
std::uint32_t evaluate_float_to_integer(const OperationWords& operands,
                                        std::uint32_t immediate) noexcept;
std::string emit_float_to_integer(const OperationText& operands, std::uint32_t immediate);
std::string_view helper_float_to_integer() noexcept;

// Float compare to predicate: two raw binary32 words, one IEEE comparison
// code and a flush-to-zero flag in the immediate (bit 4; code in bits 0..3).
// Ordered codes are false and the code-8..15 codes (NaN, the unordered
// variants, and T) are true when either input is a NaN. Flush-to-zero
// replaces zero and subnormal inputs with same-sign zero before comparing;
// NaNs and infinities pass through.
bool admits_float_compare_predicate(std::uint32_t immediate) noexcept;
std::uint32_t evaluate_float_compare_predicate(const OperationWords& operands,
                                               std::uint32_t immediate) noexcept;
std::string emit_float_compare_predicate(const OperationText& operands,
                                         std::uint32_t immediate);
std::string_view helper_float_compare_predicate() noexcept;
bool admits_integer_to_float(std::uint32_t immediate) noexcept;
std::uint32_t evaluate_integer_to_float(const OperationWords& operands,
                                        std::uint32_t immediate) noexcept;
std::string emit_integer_to_float(const OperationText& operands, std::uint32_t immediate);
std::string_view integer_to_float_msl_helper() noexcept;
bool admits_float_round_integral(std::uint32_t immediate) noexcept;
std::uint32_t evaluate_float_round_integral(const OperationWords& operands,
                                            std::uint32_t immediate) noexcept;
std::string emit_float_round_integral(const OperationText& operands,
                                      std::uint32_t immediate);
std::string_view helper_float_round_integral() noexcept;

}  // namespace armigo::shader::graphics::detail
