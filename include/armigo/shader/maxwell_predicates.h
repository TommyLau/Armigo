// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of the integer-set-predicate register family. This header declares
// only the shared operand types it needs; the decoder declaration stays in
// the private registry headers.
#pragma once

#include <armigo/shader/maxwell_operands.h>

#include <cstdint>

namespace armigo::shader::maxwell {

// Decoded integer-set-predicate instruction of the register encoding. The
// admitted profile fixes the comparison to not-equal and the boolean
// combination to AND with the PT source predicate; the signed field is
// retained raw because raw-word equality does not depend on it.
//
// The predicate destinations are the paired 3-bit fields in encoding order
// (destination_a at bits 3..5, destination_b at bits 0..2), preserved as
// 0..7 including PT (7), whose writes downstream discard. Both register
// operands are preserved as encoded, including RZ (255). This payload carries
// decoding facts only; defining predicate values for later instructions is
// downstream work that no decoder or lowering path implements yet.
struct IntegerSetPredicate {
    std::uint8_t destination_a = 0;
    std::uint8_t destination_b = 0;
    RegisterOperand lhs;
    RegisterOperand rhs;
    Predicate source;
    bool signed_compare = false;
};

}  // namespace armigo::shader::maxwell
