// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Shared bitfield, error, and operand primitives of the Maxwell decoder.
// Internal to the shader target: every entity is inline or constexpr so no
// external symbols are exported.
#pragma once

#include <armigo/shader/maxwell.h>

#include <cstdint>
#include <optional>
#include <variant>

namespace armigo::shader::maxwell::detail {

constexpr std::uint64_t bit_mask(unsigned width, unsigned shift) {
    return ((std::uint64_t{1} << width) - 1) << shift;
}

constexpr std::uint64_t field(std::uint64_t word, unsigned shift,
                              unsigned width) {
    return (word >> shift) & ((std::uint64_t{1} << width) - 1);
}

constexpr std::uint64_t high_mask(std::uint16_t mask) {
    return static_cast<std::uint64_t>(mask) << 48;
}

constexpr bool high_matches(std::uint64_t word, std::uint16_t mask,
                            std::uint16_t value) {
    return ((word >> 48) & mask) == value;
}

inline std::uint64_t read_word(const std::uint8_t* bytes) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return value;
}

inline DecodeError failure(DecodeError::Code code, std::uint32_t pc,
                           OpcodeFamily family = OpcodeFamily::Unknown,
                           DecodeField decode_field = DecodeField::None,
                           std::uint64_t value = 0,
                           std::uint64_t expected = 0) {
    return {code, pc, family, decode_field, value, expected, std::nullopt};
}

inline std::optional<DecodeError> unsupported(std::uint32_t pc,
                                              OpcodeFamily family,
                                              DecodeField decode_field,
                                              std::uint64_t value,
                                              std::uint64_t expected) {
    if (value == expected) {
        return std::nullopt;
    }
    return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                   decode_field, value, expected);
}

inline std::optional<DecodeError> reserved(std::uint32_t pc,
                                           OpcodeFamily family,
                                           std::uint64_t word,
                                           std::uint64_t allowed) {
    const std::uint64_t unexpected = word & ~allowed;
    if (unexpected == 0) {
        return std::nullopt;
    }
    return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                   DecodeField::ReservedBits, unexpected, 0);
}

inline RegisterOperand reg(std::uint64_t word, unsigned shift) {
    return {static_cast<std::uint8_t>(field(word, shift, 8))};
}

inline std::variant<ConstantBufferOperand, DecodeError> constant_buffer(
    std::uint64_t word, std::uint32_t pc, OpcodeFamily family) {
    const auto binding = static_cast<std::uint8_t>(field(word, 34, 5));
    if (binding >= 18) {
        return failure(DecodeError::Code::UnsupportedModifier, pc, family,
                       DecodeField::ConstantBinding, binding, 17);
    }
    return ConstantBufferOperand{
        binding, static_cast<std::uint32_t>(field(word, 20, 14) * 4)};
}

}  // namespace armigo::shader::maxwell::detail
