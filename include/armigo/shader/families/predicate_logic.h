// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of the predicate-set-predicate family. This header declares only
// the shared operand types it needs; the decoder declaration stays in the
// private registry headers.
#pragma once

#include <armigo/shader/maxwell_operands.h>

#include <cstdint>

namespace armigo::shader::maxwell {

// Boolean operation combining two predicate values. The encodings follow the
// hardware order AND, OR, XOR.
enum class PredicateBooleanOperation : std::uint8_t {
    And = 0,
    Or = 1,
    Xor = 2,
};

// Decoded predicate-set-predicate instruction. The encoding pairs two
// boolean operations with three source predicates: lhs (first source) and
// rhs (second source) enter the first operation, gate (third source) enters
// the second. Every field below is carried as a decoding fact; combining and
// writing predicate values is downstream work that no decoder or lowering
// path implements yet.
//
// The predicate destinations are the paired 3-bit fields in encoding order
// (destination_a at bits 3..5, destination_b at bits 0..2), preserved as
// 0..7 including PT (7), whose writes downstream discard. Each source
// predicate preserves its 3-bit index 0..7 (PT included) and its own
// inversion bit. The boolean operations preserve both 2-bit fields; the
// fourth encoding of each is rejected with a structured diagnostic.
struct PredicateSet {
    std::uint8_t destination_a = 0;
    std::uint8_t destination_b = 0;
    Predicate lhs;
    Predicate rhs;
    Predicate gate;
    PredicateBooleanOperation first = PredicateBooleanOperation::And;
    PredicateBooleanOperation second = PredicateBooleanOperation::And;
};

}  // namespace armigo::shader::maxwell
