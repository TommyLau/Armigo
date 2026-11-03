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
#include <optional>
#include <utility>
#include <variant>
namespace armigo::shader::maxwell::detail {
namespace {
// I2F register and constant-buffer forms: destination 0..7, float format
// 8..9, integer format 10..11, signed 13, the source at 20.., rounding
// 39..40, selector 41..42, negate 45, CC 47, absolute 49. The float format
// is admitted only as F32, the integer format as U16 and U32, the rounding
// as nearest-even and toward-zero, and the selector only as 0; CC and
// negate are rejected. Absolute is admitted only inside the signed 32-bit
// nearest-even profile.
std::uint64_t common_fields() {
    return bit_mask(8, 0) | bit_mask(2, 8) | bit_mask(2, 10) | bit_mask(1, 13) |
           bit_mask(4, 16) | bit_mask(2, 39) | bit_mask(2, 41) | bit_mask(1, 45) |
           bit_mask(1, 47) | bit_mask(1, 49);
}

std::optional<DecodeError> modifiers(std::uint64_t word, std::uint32_t pc,
                                     OpcodeFamily family) {
    if (auto error = unsupported(pc, family, DecodeField::FloatFormat,
                                 field(word, 8, 2), 2)) {
        return error;
    }
    const std::uint64_t int_format = field(word, 10, 2);
    if (int_format != 1 && int_format != 2) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::IntegerFormat, int_format, 2);
    }
    const std::uint64_t rounding = field(word, 39, 2);
    if (rounding != 0 && rounding != 3) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::Rounding, rounding, 0);
    }
    if (auto error = unsupported(pc, family, DecodeField::IntegerSelector,
                                 field(word, 41, 2), 0)) {
        return error;
    }
    if (auto error = unsupported(pc, family, DecodeField::NegateB,
                                 field(word, 45, 1), 0)) {
        return error;
    }
    return unsupported(pc, family, DecodeField::ConditionCode, field(word, 47, 1), 0);
}

std::optional<DecodeError> absolute_profile(std::uint64_t word, std::uint32_t pc,
                                            OpcodeFamily family) {
    if (field(word, 49, 1) == 0) return std::nullopt;
    const bool admitted = field(word, 10, 2) == 2 && field(word, 13, 1) != 0 &&
                          field(word, 39, 2) == 0;
    if (admitted) return std::nullopt;
    return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                   DecodeField::AbsoluteB, 1, 0);
}

Instruction make(std::uint64_t word, std::uint32_t pc, OpcodeFamily family,
                 Predicate predicate,
                 std::variant<RegisterOperand, ConstantBufferOperand> source) {
    return Instruction{pc, word, family, predicate,
        IntegerToFloat{reg(word, 0), std::move(source), field(word, 13, 1) != 0,
                       static_cast<std::uint8_t>(field(word, 10, 2) == 1 ? 16 : 32),
                       static_cast<RoundingMode>(field(word, 39, 2)),
                       field(word, 49, 1) != 0}};
}
}  // namespace

DecodeOutcome decode_integer_to_float_register(std::uint64_t word, std::uint32_t pc,
                                               Predicate predicate) {
    constexpr auto family = OpcodeFamily::IntegerToFloatRegister;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfff8) | common_fields() | bit_mask(8, 20)))
        return *error;
    if (auto error = modifiers(word, pc, family)) return *error;
    if (auto error = absolute_profile(word, pc, family)) return *error;
    return make(word, pc, family, predicate, reg(word, 20));
}

DecodeOutcome decode_integer_to_float_constant(std::uint64_t word, std::uint32_t pc,
                                               Predicate predicate) {
    constexpr auto family = OpcodeFamily::IntegerToFloatConstant;
    if (auto error = reserved(pc, family, word,
                              high_mask(0xfff8) | common_fields() | bit_mask(19, 20)))
        return *error;
    if (auto error = modifiers(word, pc, family)) return *error;
    if (auto error = absolute_profile(word, pc, family)) return *error;
    const auto source = constant_buffer(word, pc, family);
    if (const auto* error = std::get_if<DecodeError>(&source)) return *error;
    return make(word, pc, family, predicate, std::get<ConstantBufferOperand>(source));
}
}  // namespace armigo::shader::maxwell::detail
