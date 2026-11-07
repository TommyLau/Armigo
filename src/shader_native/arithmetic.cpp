// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "arithmetic.h"

namespace armigo::shader_native {
bool arithmetic_opcode(unsigned opcode) {
    switch (opcode) {
    case 0x220: case 0x820: case 0xa20:
    case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21:
    case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23:
    case 0x308: case 0x824: return true;
    default: return false;
    }
}
std::variant<Arithmetic, std::string> decode_arithmetic(Instruction raw) {
    const auto opcode = raw.bits(0, 12), form = raw.bits(9, 3);
    if (!arithmetic_opcode(opcode)) return "arithmetic opcode is not admitted";
    if (raw.words[3] & 0xc00001ffu) return "arithmetic extension bits are not admitted";
    Arithmetic result;
    result.destination = raw.bits(16, 8);
    auto general = [&](unsigned at, bool absolute = false, bool negate = false) {
        return ArithmeticSource{Register{RegisterFile::general, static_cast<std::uint8_t>(raw.bits(at, 8))}, absolute, negate};
    };
    if (opcode == 0x824) {
        if ((raw.words[2] & ~0x2ffu) != 0x078e0000u) return "integer multiply-add carry or modifier mode is not admitted";
        result.numeric = {ir::Domain::unsigned_integer, ir::ScalarType::u32};
        result.operation = ir::Opcode::multiply; result.integer_multiply_add = true;
        result.sources = {general(24), {Immediate{raw.words[1]}}, general(64)};
        return result;
    }
    result.numeric = {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
    if (opcode == 0x308) {
        const unsigned function = raw.bits(74, 6);
        if (raw.bits(24, 8) || raw.words[2] != (function << 10) || (raw.words[1] & ~0xc00000ffu))
            return "special-function operand or half mode is not admitted";
        if (function != 4 && function != 5 && function != 8) return "special-function selector is not admitted";
        result.operation = function == 4 ? ir::Opcode::reciprocal : function == 5 ? ir::Opcode::reciprocal_sqrt : ir::Opcode::square_root;
        result.sources = {general(32, raw.bits(62, 1), raw.bits(63, 1))};
        return result;
    }
    const bool multiply = (opcode & 0x1ff) == 0x20;
    const bool fma = (opcode & 0x1ff) == 0x23;
    if (raw.bits(78, 2)) return "floating arithmetic rounding mode is not admitted";
    const std::uint32_t allowed = multiply ? 0x0071e300u : fma ? 0x0001efffu : 0x0001e300u;
    if (raw.words[2] & ~allowed) return "floating arithmetic mode is not admitted";
    if (multiply && raw.bits(84, 3) != 4) return "floating multiply scale is not admitted";
    result.numeric.flush_inputs = result.numeric.flush_output = raw.bits(80, 1);
    result.numeric.saturate = raw.bits(77, 1);
    result.operation = multiply ? ir::Opcode::multiply : fma ? ir::Opcode::fused_multiply_add : ir::Opcode::add;
    result.sources.push_back(general(24, raw.bits(73, 1), raw.bits(72, 1)));
    auto wide = [&]() -> std::variant<ArithmeticSource, std::string> {
        if (form == 2 || form == 4) return ArithmeticSource{Immediate{raw.words[1]}};
        if (raw.words[1] & ~0xc7ffff00u) return "constant source address or alignment is not admitted";
        const auto bank = raw.bits(54, 5);
        if (bank >= 18) return "constant bank exceeds the established IR binding contract";
        return ArithmeticSource{Constant{static_cast<std::uint8_t>(bank), static_cast<std::uint16_t>(raw.bits(38, 16))}, bool(raw.bits(62, 1)), bool(raw.bits(63, 1))};
    };
    if (form == 1) {
        if (raw.words[1] & ~0xc00000ffu) return "floating register source mode is not admitted";
        result.sources.push_back(general(32, raw.bits(62, 1), raw.bits(63, 1)));
        if (fma) result.sources.push_back(general(64, raw.bits(74, 1), raw.bits(75, 1)));
    } else {
        auto source = wide();
        if (auto error = std::get_if<std::string>(&source)) return *error;
        auto value = std::get<ArithmeticSource>(source);
        if (fma && (form == 2 || form == 3)) {
            result.sources.push_back(general(64, raw.bits(74, 1), raw.bits(75, 1)));
            result.sources.push_back(value);
        } else {
            result.sources.push_back(value);
            if (fma) result.sources.push_back(general(64, raw.bits(74, 1), raw.bits(75, 1)));
        }
    }
    return result;
}
} // namespace armigo::shader_native
