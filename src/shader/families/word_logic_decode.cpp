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
#include <array>
#include <variant>
namespace armigo::shader::maxwell::detail {
namespace {
// LOP register, constant-buffer and immediate forms: destination 0..7, A
// 8..15, B at 20.., invert A 39, invert B 40, operation
// 41..42, X 43, predicate operation 44..45, CC 47, predicate destination
// 48..50. X and CC are rejected; bit 46 is reserved.
std::uint64_t common_fields() {
    return bit_mask(8, 0) | bit_mask(8, 8) | bit_mask(4, 16) | bit_mask(2, 39) |
           bit_mask(2, 41) | bit_mask(1, 43) | bit_mask(2, 44) | bit_mask(1, 47) |
           bit_mask(3, 48);
}
std::optional<DecodeError> modifiers(std::uint64_t word, std::uint32_t pc,
                                     OpcodeFamily family) {
    if (auto error = unsupported(pc, family, DecodeField::ExtendedMode, field(word, 43, 1), 0))
        return error;
    return unsupported(pc, family, DecodeField::ConditionCode, field(word, 47, 1), 0);
}
Instruction make(std::uint64_t word, std::uint32_t pc, OpcodeFamily family,
                 Predicate predicate, LogicSource source) {
    return Instruction{pc, word, family, predicate,
        WordLogic{reg(word, 0), reg(word, 8), source,
                  static_cast<LogicOperation>(field(word, 41, 2)),
                  field(word, 39, 1) != 0, field(word, 40, 1) != 0,
                  static_cast<LogicPredicate>(field(word, 44, 2)),
                  static_cast<std::uint8_t>(field(word, 48, 3))}};
}
}  // namespace
DecodeOutcome decode_logic_register(std::uint64_t word, std::uint32_t pc,
                                    Predicate predicate) {
    constexpr auto family = OpcodeFamily::WordLogicRegister;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfff8) | common_fields() | bit_mask(8, 20)))
        return *error;
    if (auto error = modifiers(word, pc, family)) return *error;
    return make(word, pc, family, predicate, reg(word, 20));
}
DecodeOutcome decode_logic_constant(std::uint64_t word, std::uint32_t pc,
                                    Predicate predicate) {
    constexpr auto family = OpcodeFamily::WordLogicConstant;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfff8) | common_fields() | bit_mask(19, 20)))
        return *error;
    if (auto error = modifiers(word, pc, family)) return *error;
    const auto source = constant_buffer(word, pc, family);
    if (const auto* error = std::get_if<DecodeError>(&source)) return *error;
    return make(word, pc, family, predicate, std::get<ConstantBufferOperand>(source));
}
DecodeOutcome decode_logic_immediate(std::uint64_t word, std::uint32_t pc,
                                     Predicate predicate) {
    constexpr auto family = OpcodeFamily::WordLogicImmediate;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfef8) | bit_mask(1, 56) | common_fields() |
                                  bit_mask(19, 20)))
        return *error;
    if (auto error = modifiers(word, pc, family)) return *error;
    auto value = static_cast<std::uint32_t>(field(word, 20, 19));
    if (field(word, 56, 1) != 0) value |= 0xfff80000u;
    return make(word, pc, family, predicate, value);
}
}  // namespace armigo::shader::maxwell::detail
