// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Payload of LOP: bitwise logic on register A and a register,
// constant-buffer or immediate B, with optional inversion of either input and
// an optional predicate result derived from the word.
#pragma once
#include <armigo/shader/maxwell_operands.h>
#include <cstdint>
#include <variant>
namespace armigo::shader::maxwell {
enum class LogicOperation : std::uint8_t { And, Or, Xor, PassB };
// F, T, Z (result == 0), NZ (result != 0), the predicate encoding order.
enum class LogicPredicate : std::uint8_t { False, True, Zero, NonZero };
// The immediate is a 20-bit signed value, sign-extended to 32 bits.
using LogicSource = std::variant<RegisterOperand, ConstantBufferOperand, std::uint32_t>;
struct WordLogic {
    RegisterOperand destination;
    RegisterOperand source_a;
    LogicSource source_b;
    LogicOperation operation = LogicOperation::And;
    bool invert_a = false;
    bool invert_b = false;
    LogicPredicate predicate_operation = LogicPredicate::False;
    std::uint8_t predicate_destination = 7;
};
}  // namespace armigo::shader::maxwell
