// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "maxwell_families.h"

#include "family_registry_validation.h"

#include <array>
#include <cstddef>

namespace armigo::shader::maxwell::detail {
namespace {

// Append-only registry: one entry per family translation unit. New families
// add a declaration to maxwell_families.h, a definition under families/, and
// one entry here; nothing else in the decoder changes.
constexpr FamilyEntry kFamilies[] = {
    {0xfff8, 0x5c98, OpcodeFamily::MovRegister, decode_mov_register},
    {0xfff8, 0x4c98, OpcodeFamily::MovConstant, decode_mov_constant},
    {0xff80, 0x5180, OpcodeFamily::FloatFmaRegisterConstant, decode_fma_register_constant},
    {0xfc00, 0x0400, OpcodeFamily::WordAndImmediate, decode_word_and_immediate},
    {0xfff0, 0x5b60, OpcodeFamily::IntegerSetPredicateRegister, decode_isetp_register},
    {0xfff8, 0x5090, OpcodeFamily::PredicateSetPredicate, decode_predicate_set_predicate},
    {0xff81, 0x4981, OpcodeFamily::FloatFmaConstantNegated, decode_fma_constant_negated},
    {0xfffc, 0x5c6c, OpcodeFamily::FloatMultiplySaturateRegister, decode_fmul_saturate_register},
    {0xff80, 0x5980, OpcodeFamily::FloatFmaRegister, decode_ffma_register},
    {0xff00, 0x1e00, OpcodeFamily::FloatMultiplyImmediate32, decode_fmul_immediate32},
    {0xfc00, 0x0800, OpcodeFamily::FloatAddImmediate32, decode_fadd_immediate32},
    {0xfff8, 0x5c90, OpcodeFamily::RangeReductionRegister, decode_range_reduction_register},
    {0xfff8, 0x4c90, OpcodeFamily::RangeReductionConstant, decode_range_reduction_constant},
    {0xfff8, 0xf0f0, OpcodeFamily::DependencyBarrier, decode_dependency_barrier},
    {0xfff8, 0x5ca0, OpcodeFamily::SelectByPredicateRegister, decode_select_register},
    {0xfff8, 0x4ca0, OpcodeFamily::SelectByPredicateConstant, decode_select_constant},
    {0xfef8, 0x38a0, OpcodeFamily::SelectByPredicateImmediate, decode_select_immediate},
    {0xfff8, 0x5c40, OpcodeFamily::WordLogicRegister, decode_logic_register},
    {0xfff8, 0x4c40, OpcodeFamily::WordLogicConstant, decode_logic_constant},
    {0xfef8, 0x3840, OpcodeFamily::WordLogicImmediate, decode_logic_immediate},
    {0xfff8, 0x5cb0, OpcodeFamily::FloatToIntegerRegister, decode_float_to_integer_register},
    {0xfff8, 0x4cb0, OpcodeFamily::FloatToIntegerConstant, decode_float_to_integer_constant},
    {0xfff8, 0x5c60, OpcodeFamily::FloatMinMaxRegister,
     decode_float_min_max_register},
    {0xfff8, 0x4c60, OpcodeFamily::FloatMinMaxConstant,
     decode_float_min_max_constant},
    {0xfef8, 0x3860, OpcodeFamily::FloatMinMaxImmediate,
     decode_float_min_max_immediate},
    {0xfff0, 0x5bb0, OpcodeFamily::FloatSetPredicateRegister, decode_float_set_predicate_register},
    {0xfff0, 0x4bb0, OpcodeFamily::FloatSetPredicateConstant, decode_float_set_predicate_constant},
    {0xfef0, 0x36b0, OpcodeFamily::FloatSetPredicateImmediate, decode_float_set_predicate_immediate},
    {0xfff8, 0x5cb8, OpcodeFamily::IntegerToFloatRegister,
     decode_integer_to_float_register},
    {0xfff8, 0x4cb8, OpcodeFamily::IntegerToFloatConstant,
     decode_integer_to_float_constant},
    {0xfff8, 0x5ca8, OpcodeFamily::FloatRoundIntegralRegister,
     decode_float_round_integral_register},
    {0xfff8, 0x4ca8, OpcodeFamily::FloatRoundIntegralConstant,
     decode_float_round_integral_constant},
};

constexpr bool entries_are_well_formed() {
    return shader::detail::family_registry_valid(
        std::span<const FamilyEntry>{kFamilies},
        [](const FamilyEntry& entry) {
            return entry.decode != nullptr &&
                   entry.family != OpcodeFamily::Unknown;
        });
}

static_assert(entries_are_well_formed(),
              "registered decoder families must be well-formed, one-to-one, "
              "and unambiguously matchable");

}  // namespace

std::span<const FamilyEntry> registered_families() noexcept {
    return kFamilies;
}

}  // namespace armigo::shader::maxwell::detail
