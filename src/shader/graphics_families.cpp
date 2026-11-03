// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "graphics_families.h"

#include "family_registry_validation.h"

#include <array>
#include <cstddef>

namespace armigo::shader::graphics::detail {
namespace {

// Append-only registry: one entry per family translation unit. New families
// add a declaration to graphics_families.h, a definition under families/, and
// one entry here; nothing else in the lowerer changes.
constexpr LowerEntry kLowerers[] = {
    {maxwell::OpcodeFamily::MovRegister, lower_mov_register},
    {maxwell::OpcodeFamily::MovConstant, lower_mov_constant},
    {maxwell::OpcodeFamily::FloatFmaRegisterConstant, lower_fma_register_constant},
    {maxwell::OpcodeFamily::WordAndImmediate, lower_word_and_immediate},
    {maxwell::OpcodeFamily::FloatFmaConstantNegated, lower_fma_constant_negated},
    {maxwell::OpcodeFamily::FloatMultiplySaturateRegister,
     lower_fmul_saturate_register},
    {maxwell::OpcodeFamily::FloatFmaRegister, lower_ffma_register},
    {maxwell::OpcodeFamily::FloatMultiplyImmediate32,
     lower_fmul_immediate32},
    {maxwell::OpcodeFamily::FloatAddImmediate32, lower_fadd_immediate32},
    {maxwell::OpcodeFamily::RangeReductionRegister, lower_range_reduction},
    {maxwell::OpcodeFamily::RangeReductionConstant, lower_range_reduction},
    {maxwell::OpcodeFamily::DependencyBarrier, lower_dependency_barrier},
    {maxwell::OpcodeFamily::SelectByPredicateRegister, lower_select_by_predicate},
    {maxwell::OpcodeFamily::SelectByPredicateConstant, lower_select_by_predicate},
    {maxwell::OpcodeFamily::SelectByPredicateImmediate, lower_select_by_predicate},
    {maxwell::OpcodeFamily::WordLogicRegister, lower_word_logic},
    {maxwell::OpcodeFamily::WordLogicConstant, lower_word_logic},
    {maxwell::OpcodeFamily::WordLogicImmediate, lower_word_logic},
    {maxwell::OpcodeFamily::FloatToIntegerRegister, lower_float_to_integer},
    {maxwell::OpcodeFamily::FloatToIntegerConstant, lower_float_to_integer},
    {maxwell::OpcodeFamily::FloatMinMaxRegister, lower_float_min_max},
    {maxwell::OpcodeFamily::FloatMinMaxConstant, lower_float_min_max},
    {maxwell::OpcodeFamily::FloatMinMaxImmediate, lower_float_min_max},
    {maxwell::OpcodeFamily::FloatSetPredicateRegister, lower_float_set_predicate},
    {maxwell::OpcodeFamily::FloatSetPredicateConstant, lower_float_set_predicate},
    {maxwell::OpcodeFamily::FloatSetPredicateImmediate, lower_float_set_predicate},
    {maxwell::OpcodeFamily::ReconvergenceSetup, lower_ssy},
    {maxwell::OpcodeFamily::IntegerToFloatRegister, lower_integer_to_float},
    {maxwell::OpcodeFamily::IntegerToFloatConstant, lower_integer_to_float},
    {maxwell::OpcodeFamily::FloatRoundIntegralRegister,
     lower_float_round_integral},
    {maxwell::OpcodeFamily::FloatRoundIntegralConstant,
     lower_float_round_integral},
};

constexpr bool entries_are_well_formed() {
    return shader::detail::family_registry_valid(
        std::span<const LowerEntry>{kLowerers},
        [](const LowerEntry& entry) {
            return entry.lower != nullptr &&
                   entry.family != maxwell::OpcodeFamily::Unknown;
        });
}

static_assert(entries_are_well_formed(),
              "registered family lowerers must be well-formed and one-to-one");

}  // namespace

std::span<const LowerEntry> registered_lowerers() noexcept {
    return kLowerers;
}

}  // namespace armigo::shader::graphics::detail
