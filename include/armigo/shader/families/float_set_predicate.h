// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of the float set predicate (FSETP) family. This header declares
// only the shared operand types it needs; the decoder declarations stay in
// the private registry headers.
#pragma once

#include <armigo/shader/maxwell_operands.h>

#include <armigo/shader/families/predicate_logic.h>

#include <cstdint>
#include <variant>

namespace armigo::shader::maxwell {

// IEEE comparison code of the FSETP family, in encoding order. Codes F..GE
// are ordered: they are false when either input is a NaN. NUM is true when
// both inputs are numbers, NAN exactly when either is a NaN. The six
// U-suffixed codes are their base comparison OR unordered, so a NaN
// satisfies each of them; T is always true and F always false.
enum class FloatCompareCode : std::uint8_t {
    False = 0,
    LessThan = 1,
    Equal = 2,
    LessEqual = 3,
    GreaterThan = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Number = 7,
    Nan = 8,
    LessThanUnordered = 9,
    EqualUnordered = 10,
    LessEqualUnordered = 11,
    GreaterThanUnordered = 12,
    NotEqualUnordered = 13,
    GreaterEqualUnordered = 14,
    True = 15,
};

// Second source of the FSETP family: a register in the register form, a
// constant-buffer word in the buffer form, and a sign-and-magnitude short
// immediate in the immediate form.
using FloatSetPredicateSource =
    std::variant<RegisterOperand, ConstantBufferOperand, ShortFloatImmediate>;

// Decoded float set predicate. The encoding computes two predicate
// destinations from one comparison combined with one source predicate:
// destination A receives compare(a, b) combined with the predicate, and
// destination B receives NOT compare(a, b) combined with it. Every field
// below is carried as a decoding fact; combining and writing predicate
// values is downstream work.
//
// The predicate destinations are the paired 3-bit fields in encoding order
// (destination_a at bits 3..5, destination_b at bits 0..2), preserved as
// 0..7 including PT (7), whose writes downstream discard. The boolean
// predicate preserves its 3-bit index (PT included) and its inversion bit.
// The operand sign modifiers are preserved independently; both apply
// absolute value before negation, as pure sign-bit operations upstream of
// the comparison's own flush.
struct FloatSetPredicate {
    std::uint8_t destination_a = 0;
    std::uint8_t destination_b = 0;
    RegisterOperand source_a;
    FloatSetPredicateSource source_b;
    Predicate boolean_predicate;
    PredicateBooleanOperation boolean_operation = PredicateBooleanOperation::And;
    bool negate_a = false;
    bool absolute_a = false;
    bool negate_b = false;
    bool absolute_b = false;
    bool flush_to_zero = true;
    FloatCompareCode comparison = FloatCompareCode::False;
};

}  // namespace armigo::shader::maxwell
