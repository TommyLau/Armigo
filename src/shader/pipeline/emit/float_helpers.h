// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo places every guest modifier into NumericMode or an explicit operation;
// unresolved selectors return diagnostics.

#pragma once

#include "pipeline/emitter_context.h"

#include <array>

namespace armigo::shader::pipeline::float_emit {
// Lane 7 shared helpers for the floating, half, conversion and multifunction
// emitters. Integer lanes own their own flag helpers; nothing here escapes the
// emit/float family.

// Forward-chaining for Result<T>: on failure the diagnostic propagates and the
// enclosing Status-returning emitter returns immediately.
#define ARMIGO_FLOAT_TRY(variable, expression)                                        \
    auto variable##_result = (expression);                                            \
    if (variable##_result.index() == 1) return std::move(std::get<1>(variable##_result)); \
    auto variable = std::move(std::get<0>(std::move(variable##_result)))

// Exact word32 patterns of the scale and comparison literals. Powers of two and
// zero are exactly representable, so no host floating-point rounding is involved.
inline constexpr std::uint32_t float_zero_bits = 0x00000000U;
inline constexpr std::uint32_t float_one_bits = 0x3F800000U;
inline constexpr std::uint32_t float_half_bits = 0x3F000000U;
inline constexpr std::uint32_t float_quarter_bits = 0x3E800000U;
inline constexpr std::uint32_t float_eighth_bits = 0x3E000000U;
inline constexpr std::uint32_t float_two_bits = 0x40000000U;
inline constexpr std::uint32_t float_four_bits = 0x40800000U;
inline constexpr std::uint32_t float_eight_bits = 0x41000000U;
static_assert(float_half_bits == std::bit_cast<std::uint32_t>(0.5F));
static_assert(float_two_bits == std::bit_cast<std::uint32_t>(2.0F));

[[nodiscard]] inline Operand word(std::uint32_t bits, ValueType type = ValueType::word32) {
    return LiteralBits{bits, type};
}

// The decoded NumericMode is authoritative: emitters carry it into operations
// exactly (rounding, flush, saturation, precision). A missing or nonfloating
// mode is a decoder contract violation, never a silent default.
[[nodiscard]] Result<NumericMode> float_mode(const EmitterContext& context);
[[nodiscard]] NumericMode without_saturation(const NumericMode& mode);
// Comparisons and NaN tests cannot carry output flush or saturation.
[[nodiscard]] NumericMode comparison_mode(const NumericMode& mode);
// Select operations only interleave already computed values.
[[nodiscard]] NumericMode plain_float_mode(const NumericMode& mode);

[[nodiscard]] Result<Operand> source_operand(EmitterContext& context,
                                             const DecodedInstruction& instruction, SourceRole role);
// Resolves a source and applies its absolute/negate modifiers in one step.
[[nodiscard]] Result<Operand> modified_source(EmitterContext& context,
                                              const DecodedInstruction& instruction,
                                              SourceRole role, const NumericMode& mode);
// Absolute first, then negation, mirroring the guest modifier order.
[[nodiscard]] Result<Operand> apply_source_modifiers(EmitterContext& context, Operand value,
                                                     const SourceModifiers& modifiers,
                                                     const NumericMode& mode);
// Word64 sources are explicitly packed from low and high word32 containers.
[[nodiscard]] Result<Operand> float64_source(EmitterContext& context, const DecodedSource& source);
// The predicate source (with negate meaning inversion) is a dedicated role.
[[nodiscard]] Result<Operand> predicate_operand(EmitterContext& context,
                                                const DecodedInstruction& instruction);
// The decoded modifiers of one source role (subword position, abs, negate).
[[nodiscard]] Result<SourceModifiers> source_modifiers(const DecodedInstruction& instruction,
                                                       SourceRole role);
[[nodiscard]] Result<Register> destination_register(const DecodedInstruction& instruction);
[[nodiscard]] Result<Register> second_destination_register(const DecodedInstruction& instruction);
// The guest write-CC bit (0/1). Absence of the decoded field is an error.
[[nodiscard]] Result<bool> write_cc_enabled(EmitterContext& context);
[[nodiscard]] Status write_destination(EmitterContext& context,
                                       const DecodedInstruction& instruction, Operand value);
// Explicit CC flag writes: zero is flag 0, sign is flag 1.
[[nodiscard]] Status write_float_flags(EmitterContext& context, Operand value,
                                       const NumericMode& mode, bool write_cc);
[[nodiscard]] Status write_integer_flags(EmitterContext& context, Operand value, bool write_cc);

// Floating comparison with guest NaN selectors. Selector values are the guest
// FCMP field (0..15): F, LT, EQ, LE, GT, NE, GE, NUM, NAN, LTU, EQU, LEU, GTU,
// NEU, GEU, T. NUM and NAN expand through is_nan tests because no relation
// expresses "ordered" alone; the others use ComparisonInfo with the unordered
// flag from selector bit 3.
[[nodiscard]] Result<Operand> float_comparison(EmitterContext& context, std::uint64_t selector,
                                               Operand left, Operand right,
                                               const NumericMode& mode);

// Predicate combination with the guest BOP field (0..2 AND/OR/XOR, 3 pass).
[[nodiscard]] Result<Operand> predicate_combine(EmitterContext& context, std::uint64_t selector,
                                                Operand input, Operand predicate);

// Half sources decode into two f32 lanes. Accepted SourceModifiers combinations:
// scalar f16 with subword full/half_low/half_high (F16/H0H0/H1H1) and scalar f32
// with subword full (F32 broadcast).
[[nodiscard]] Result<std::array<Operand, 2>> half_source_pair(EmitterContext& context,
                                                              const DecodedSource& source,
                                                              const NumericMode& mode);
// Resolves, unpacks and applies abs/negate modifiers to both half lanes.
[[nodiscard]] Result<std::array<Operand, 2>> modified_half_source_pair(
    EmitterContext& context, const DecodedInstruction& instruction, SourceRole role,
    const NumericMode& mode);
// Half output selector for HADD2/HMUL2/HFMA2 (guest FMT field): 0 F16 pack both,
// 1 F32 keep lane zero, 2 MrgH0 keep dest high, 3 MrgH1 keep dest low.
[[nodiscard]] Result<Operand> pack_half_result(EmitterContext& context, std::uint64_t selector,
                                               Operand low, Operand high, const NumericMode& mode,
                                               Register destination);
// Byte/half field extraction for the conversion families. bit_start comes from
// the source subword position; signedness selects the numeric domain.
[[nodiscard]] Result<Operand> extract_subword(EmitterContext& context, Operand value,
                                              std::uint32_t bit_start, std::uint32_t bit_count,
                                              bool is_signed);
[[nodiscard]] std::uint32_t subword_bit_start(Subword position);
} // namespace armigo::shader::pipeline::float_emit
