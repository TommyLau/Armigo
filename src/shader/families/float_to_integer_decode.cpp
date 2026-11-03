// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// F2I decoders for the register (0xfff8/0x5cb0) and constant-buffer
// (0xfff8/0x4cb0) windows. Admitted fields are F32 source, half=0, FTZ input,
// absolute=0, CC=0, negate=0, and exactly these destination profiles:
// unsigned I16 nearest-even, signed I32 floor, signed I32 ceil, unsigned I32
// truncating, and signed I32 truncating. Every other field value, stray bit,
// or profile combination is an explicit UnsupportedModifier.
#include "../maxwell_families.h"
#include "../maxwell_decode_helpers.h"

#include <array>
#include <variant>

namespace armigo::shader::maxwell::detail {
namespace {

constexpr unsigned destination_format_i16 = 1;
constexpr unsigned destination_format_i32 = 2;
constexpr unsigned source_format_f32 = 2;

DecodeOutcome decode_float_to_integer(std::uint64_t word, std::uint32_t pc,
                                      Predicate predicate,
                                      OpcodeFamily family) {
    // Destination 0..7, destination format 8..9, source format 10..11,
    // signed 12, instruction predicate 16..19, B register 20..27 (the
    // constant form uses the buffer operand 20..38), rounding 39..40,
    // half 41, FTZ 44, absolute 45, CC 47, negate 49.
    const std::uint64_t common = high_mask(0xfff8) | bit_mask(8, 0) |
        bit_mask(2, 8) | bit_mask(2, 10) | bit_mask(1, 12) |
        bit_mask(4, 16) | bit_mask(2, 39) | bit_mask(1, 41) |
        bit_mask(1, 44) | bit_mask(1, 45) | bit_mask(1, 47) |
        bit_mask(1, 49);
    const auto allowed = family == OpcodeFamily::FloatToIntegerConstant
        ? common | bit_mask(19, 20)
        : common | bit_mask(8, 20);
    if (auto error = reserved(pc, family, word, allowed)) return *error;
    struct Rule { DecodeField field; unsigned shift, width; std::uint64_t expected; };
    constexpr std::array rules{
        Rule{DecodeField::SourceHalf, 41, 1, 0},
        Rule{DecodeField::FlushMode, 44, 1, 1},
        Rule{DecodeField::AbsoluteA, 45, 1, 0},
        Rule{DecodeField::ConditionCode, 47, 1, 0},
        Rule{DecodeField::NegateA, 49, 1, 0}};
    for (const auto& rule : rules)
        if (auto error = unsupported(pc, family, rule.field,
                                     field(word, rule.shift, rule.width),
                                     rule.expected))
            return *error;

    const auto destination_format = field(word, 8, 2);
    if (destination_format != destination_format_i16 &&
        destination_format != destination_format_i32) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::DestinationFormat, destination_format,
                       destination_format_i32);
    }
    if (auto error = unsupported(pc, family, DecodeField::SourceFormat,
                                 field(word, 10, 2), source_format_f32))
        return *error;

    // The admitted profiles: I32 signed with floor, ceil, or trunc; I32
    // unsigned with trunc; I16 unsigned with round-even. A combination
    // outside them reports on the rounding field.
    const std::uint64_t rounding = field(word, 39, 2);
    const bool is_signed = field(word, 12, 1) != 0;
    if (destination_format == destination_format_i16 && is_signed) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::Signedness, 1, 0);
    }
    const bool admitted =
        (destination_format == destination_format_i32 && is_signed &&
         rounding >= 1) ||
        (destination_format == destination_format_i32 && !is_signed &&
         rounding == 3) ||
        (destination_format == destination_format_i16 && !is_signed &&
         rounding == 0);
    if (!admitted) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::Rounding, rounding,
                       destination_format == destination_format_i16 ? 0 : 3);
    }

    FloatToIntegerSource source;
    if (family == OpcodeFamily::FloatToIntegerConstant) {
        auto buffer = constant_buffer(word, pc, family);
        if (auto* error = std::get_if<DecodeError>(&buffer)) return *error;
        source = std::get<ConstantBufferOperand>(buffer);
    } else {
        source = reg(word, 20);
    }
    return Instruction{pc, word, family, predicate,
                       FloatToInteger{reg(word, 0), std::move(source),
                                      static_cast<RoundingMode>(rounding),
                                      FlushMode::FlushToZero, is_signed,
                                      destination_format == destination_format_i16
                                          ? std::uint8_t{16}
                                          : std::uint8_t{32}}};
}

}  // namespace

DecodeOutcome decode_float_to_integer_register(std::uint64_t word,
                                               std::uint32_t pc,
                                               Predicate predicate) {
    return decode_float_to_integer(word, pc, predicate,
                                   OpcodeFamily::FloatToIntegerRegister);
}

DecodeOutcome decode_float_to_integer_constant(std::uint64_t word,
                                               std::uint32_t pc,
                                               Predicate predicate) {
    return decode_float_to_integer(word, pc, predicate,
                                   OpcodeFamily::FloatToIntegerConstant);
}

}  // namespace armigo::shader::maxwell::detail
