// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "graphics_operations.h"

#include <cstddef>
#include <iterator>

namespace armigo::shader::graphics {
namespace detail {
namespace {

constexpr ValueType R = ValueType::Raw32;
constexpr ValueType P = ValueType::Predicate;

// Append-only registry: one entry per operation kind. New kinds append a
// PureKind value, declare their unit functions in graphics_operations.h,
// define them under operations/, and add one entry here; the validators,
// evaluator, and emitters are not edited. The array size is deduced so an
// appended entry touches no other line.
constexpr OperationEntry kOperations[] = {
    {PureKind::FloatNegate, 1, {R, R, R}, R, admits_no_immediate,
     evaluate_float_negate, emit_float_negate, nullptr},
    {PureKind::FloatAbsolute, 1, {R, R, R}, R, admits_no_immediate,
     evaluate_float_absolute, emit_float_absolute, nullptr},
    {PureKind::Select, 3, {P, R, R}, R, admits_no_immediate,
     evaluate_select, emit_select, nullptr},
    {PureKind::PredicateAnd, 2, {P, P, R}, P, admits_no_immediate,
     evaluate_predicate_and, emit_predicate_and, nullptr},
    {PureKind::PredicateOr, 2, {P, P, R}, P, admits_no_immediate,
     evaluate_predicate_or, emit_predicate_or, nullptr},
    {PureKind::PredicateXor, 2, {P, P, R}, P, admits_no_immediate,
     evaluate_predicate_xor, emit_predicate_xor, nullptr},
    {PureKind::PredicateSelect, 3, {P, P, P}, P, admits_no_immediate,
     evaluate_predicate_select, emit_predicate_select, nullptr},
    {PureKind::WordLogic, 2, {R, R, R}, R, admits_word_logic,
     evaluate_word_logic, emit_word_logic, nullptr},
    {PureKind::WordIsZero, 1, {R, R, R}, P, admits_no_immediate,
     evaluate_word_is_zero, emit_word_is_zero, nullptr},
    {PureKind::FloatToInteger, 1, {R, R, R}, R, admits_float_to_integer,
     evaluate_float_to_integer, emit_float_to_integer, helper_float_to_integer},
    {PureKind::FloatMinimum, 2, {R, R, R}, R, admits_no_immediate,
     evaluate_float_minimum, emit_float_minimum, helper_float_minimum},
    {PureKind::FloatMaximum, 2, {R, R, R}, R, admits_no_immediate,
     evaluate_float_maximum, emit_float_maximum, helper_float_maximum},
    {PureKind::FloatComparePredicate, 2, {R, R, R}, P,
     admits_float_compare_predicate, evaluate_float_compare_predicate,
     emit_float_compare_predicate, helper_float_compare_predicate},
    {PureKind::IntegerToFloat, 1, {R, R, R}, R, admits_integer_to_float,
     evaluate_integer_to_float, emit_integer_to_float, integer_to_float_msl_helper},
    {PureKind::FloatRoundIntegral, 1, {R, R, R}, R, admits_float_round_integral,
     evaluate_float_round_integral, emit_float_round_integral,
     helper_float_round_integral},
};

constexpr bool entries_are_well_formed() {
    for (std::size_t i = 0; i < std::size(kOperations); ++i) {
        const auto& entry = kOperations[i];
        if (entry.arity == 0 || entry.arity > 3 || entry.admits == nullptr ||
            entry.evaluate == nullptr || entry.emit == nullptr ||
            entry.result == ValueType::Sample4) {
            return false;
        }
        for (std::size_t slot = 0; slot < entry.arity; ++slot) {
            if (entry.operands[slot] == ValueType::Sample4) return false;
        }
        for (std::size_t j = i + 1; j < std::size(kOperations); ++j) {
            if (entry.kind == kOperations[j].kind) return false;
        }
    }
    return true;
}

static_assert(entries_are_well_formed(),
              "registered operations must be well-formed and one-to-one");

}  // namespace

std::span<const OperationEntry> registered_operations() noexcept {
    return kOperations;
}

const OperationEntry* find_operation(PureKind kind) noexcept {
    for (const OperationEntry& entry : kOperations) {
        if (entry.kind == kind) return &entry;
    }
    return nullptr;
}

bool admits_no_immediate(std::uint32_t immediate) noexcept {
    return immediate == 0;
}

}  // namespace detail

ValueType pure_result_type(PureKind kind) noexcept {
    const auto* entry = detail::find_operation(kind);
    return entry == nullptr ? ValueType::Raw32 : entry->result;
}

}  // namespace armigo::shader::graphics
