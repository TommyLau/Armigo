// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../maxwell_families.h"

#include "../maxwell_decode_helpers.h"

#include <utility>
#include <variant>

namespace armigo::shader::maxwell::detail {
namespace {

// Float set predicate, shared body of the register, constant-buffer, and
// immediate forms. The opcode patterns fix bits 63..52 (and, for the
// immediate form, bit 56 is the immediate sign), leaving the 4-bit compare
// code at 48..51 as the opcode-pattern wildcard; every code is admitted. The
// admitted profile restricts the 2-bit boolean operation to AND/OR/XOR and
// treats every undecoded field position as reserved except the paired 3-bit
// predicate destinations, both operand sign modifiers, the 3-bit boolean
// predicate with its inversion, the boolean operation, the flush-to-zero
// flag, and the common PT instruction predicate. Destinations and sources
// are preserved as encoded, including predicate PT (7) and register RZ
// (255). The immediate form reconstructs the binary32 word from its
// magnitude-at-20 and sign-at-56 encoding; its low 12 bits are zero.
DecodeOutcome decode_float_set_predicate(std::uint64_t word,
                                         std::uint32_t pc,
                                         OpcodeFamily family,
                                         std::uint64_t opcode_mask,
                                         std::uint64_t source_mask,
                                         FloatSetPredicateSource source_b,
                                         Predicate predicate) {
    const std::uint64_t allowed =
        opcode_mask | source_mask | bit_mask(6, 0) | bit_mask(4, 16) | bit_mask(1, 6) |
        bit_mask(1, 7) | bit_mask(8, 8) | bit_mask(3, 39) |
        bit_mask(1, 42) | bit_mask(1, 43) | bit_mask(1, 44) |
        bit_mask(2, 45) | bit_mask(1, 47) | bit_mask(4, 48);
    if (auto error = reserved(pc, family, word, allowed)) {
        return *error;
    }
    const std::uint64_t operation = field(word, 45, 2);
    if (operation > static_cast<std::uint64_t>(
                        PredicateBooleanOperation::Xor)) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::BooleanOperation, operation,
                       static_cast<std::uint64_t>(
                           PredicateBooleanOperation::Xor));
    }
    return Instruction{
        pc, word, family, predicate,
        FloatSetPredicate{
            static_cast<std::uint8_t>(field(word, 3, 3)),
            static_cast<std::uint8_t>(field(word, 0, 3)), reg(word, 8),
            std::move(source_b),
            Predicate{static_cast<std::uint8_t>(field(word, 39, 3)),
                      field(word, 42, 1) != 0},
            static_cast<PredicateBooleanOperation>(operation),
            field(word, 43, 1) != 0, field(word, 7, 1) != 0,
            field(word, 6, 1) != 0, field(word, 44, 1) != 0,
            field(word, 47, 1) != 0,
            static_cast<FloatCompareCode>(field(word, 48, 4))}};
}

}  // namespace

DecodeOutcome decode_float_set_predicate_register(std::uint64_t word,
                                                  std::uint32_t pc,
                                                  Predicate predicate) {
    return decode_float_set_predicate(
        word, pc, OpcodeFamily::FloatSetPredicateRegister, high_mask(0xfff0),
        bit_mask(8, 20), reg(word, 20), predicate);
}

DecodeOutcome decode_float_set_predicate_constant(std::uint64_t word,
                                                  std::uint32_t pc,
                                                  Predicate predicate) {
    const auto addend = constant_buffer(word, pc,
                                        OpcodeFamily::FloatSetPredicateConstant);
    if (const auto* error = std::get_if<DecodeError>(&addend)) {
        return *error;
    }
    return decode_float_set_predicate(
        word, pc, OpcodeFamily::FloatSetPredicateConstant, high_mask(0xfff0),
        bit_mask(19, 20), std::get<ConstantBufferOperand>(addend), predicate);
}

DecodeOutcome decode_float_set_predicate_immediate(std::uint64_t word,
                                                   std::uint32_t pc,
                                                   Predicate predicate) {
    return decode_float_set_predicate(
        word, pc, OpcodeFamily::FloatSetPredicateImmediate,
        high_mask(0xfef0) | bit_mask(1, 56), bit_mask(19, 20),
        ShortFloatImmediate{
            static_cast<std::uint32_t>(field(word, 20, 19) << 12) |
            static_cast<std::uint32_t>(field(word, 56, 1) << 31)},
        predicate);
}

}  // namespace armigo::shader::maxwell::detail
