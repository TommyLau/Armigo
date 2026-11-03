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
#include <variant>
namespace armigo::shader::maxwell::detail {
namespace {
// SEL register, constant-buffer and immediate forms: destination 0..7, A
// 8..15, the B source
// at 20.., predicate 39..41, negate-predicate 42. Every other low bit is
// reserved; the guard predicate 16..19 is validated by the shared decoder.
std::uint64_t common_fields() {
    return bit_mask(8, 0) | bit_mask(8, 8) | bit_mask(4, 16) | bit_mask(3, 39) |
           bit_mask(1, 42);
}
Instruction make(std::uint64_t word, std::uint32_t pc, OpcodeFamily family,
                 Predicate predicate, SelectSource source) {
    return Instruction{pc, word, family, predicate,
        SelectByPredicate{reg(word, 0), reg(word, 8), source,
                          static_cast<std::uint8_t>(field(word, 39, 3)),
                          field(word, 42, 1) != 0}};
}
}  // namespace
DecodeOutcome decode_select_register(std::uint64_t word, std::uint32_t pc,
                                     Predicate predicate) {
    constexpr auto family = OpcodeFamily::SelectByPredicateRegister;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfff8) | common_fields() | bit_mask(8, 20)))
        return *error;
    return make(word, pc, family, predicate, reg(word, 20));
}
DecodeOutcome decode_select_constant(std::uint64_t word, std::uint32_t pc,
                                     Predicate predicate) {
    constexpr auto family = OpcodeFamily::SelectByPredicateConstant;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfff8) | common_fields() | bit_mask(19, 20)))
        return *error;
    const auto source = constant_buffer(word, pc, family);
    if (const auto* error = std::get_if<DecodeError>(&source)) return *error;
    return make(word, pc, family, predicate, std::get<ConstantBufferOperand>(source));
}
DecodeOutcome decode_select_immediate(std::uint64_t word, std::uint32_t pc,
                                      Predicate predicate) {
    constexpr auto family = OpcodeFamily::SelectByPredicateImmediate;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfef8) | bit_mask(1, 56) | common_fields() |
                                  bit_mask(19, 20)))
        return *error;
    // The immediate uses a 19-bit magnitude field with the sign at bit 56,
    // two's-complement extended to 32 bits.
    auto value = static_cast<std::uint32_t>(field(word, 20, 19));
    if (field(word, 56, 1) != 0) value |= 0xfff80000u;
    return make(word, pc, family, predicate, value);
}
}  // namespace armigo::shader::maxwell::detail
