// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <cstdint>

namespace armigo::shader::maxwell {

struct Predicate {
    std::uint8_t index = 7;
    bool inverted = false;
};

struct RegisterOperand {
    std::uint8_t index = 255;
    [[nodiscard]] bool is_zero() const noexcept { return index == 255; }
};

struct ConstantBufferOperand {
    std::uint8_t binding = 0;
    std::uint32_t byte_offset = 0;
};

struct ShortFloatImmediate {
    std::uint32_t raw_bits = 0;
};

enum class RoundingMode : std::uint8_t {
    NearestEven = 0,
    NegativeInfinity = 1,
    PositiveInfinity = 2,
    Zero = 3,
};

enum class FlushMode : std::uint8_t {
    None = 0,
    FlushToZero = 1,
    FlushMultiplyZero = 2,
};

enum class OrderedComparison : std::uint8_t {
    LessThan = 1,
    GreaterThan = 4,
};

}  // namespace armigo::shader::maxwell
