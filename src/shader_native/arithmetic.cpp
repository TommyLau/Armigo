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

#include <algorithm>

namespace armigo::shader_native {
bool arithmetic_opcode(unsigned opcode) {
    switch (opcode) {
    case 0x220: case 0x820: case 0xa20:
    case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21:
    case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23:
    case 0xc20: case 0xc21: case 0xe21: case 0xc23: case 0xe23:
    case 0x308: case 0xb08: case 0xd08: return true;
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
    result.numeric = {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
    if ((opcode & 0x1ff) == 0x108) {
        // MUFU has no flush selector: like the FTZ arithmetic around it, the
        // unit flushes binary32 subnormals at its input and result. Metal's
        // functions flush the same way and define the approximation, as in
        // the Maxwell multifunction contract. SIN and COS take their angle in
        // turns; the frontend converts it to radians for the shared sine and
        // cosine.
        const unsigned function = raw.bits(74, 6);
        const std::uint32_t selector = opcode == 0xd08 ? 0x08000000u : 0;
        if (raw.bits(24, 8) || raw.words[2] != ((function << 10) | selector))
            return "special-function operand or half mode is not admitted";
        if (function > 5 && function != 8)
            return "special-function selector is not admitted";
        constexpr ir::Opcode operations[]{ir::Opcode::cosine, ir::Opcode::sine, ir::Opcode::exp2, ir::Opcode::log2,
            ir::Opcode::reciprocal, ir::Opcode::reciprocal_sqrt, ir::Opcode::copy, ir::Opcode::copy, ir::Opcode::square_root};
        result.operation = operations[function];
        result.numeric.flush_inputs = result.numeric.flush_output = true;
        if (opcode == 0x308) {
            if (raw.words[1] & ~0xc00000ffu) return "special-function register source mode is not admitted";
            result.sources = {general(32, raw.bits(62, 1), raw.bits(63, 1))};
        } else if (opcode == 0xd08) {
            if (raw.words[1] & ~0xc000003fu) return "special-function uniform source mode is not admitted";
            result.sources = {ArithmeticSource{Register{RegisterFile::uniform, static_cast<std::uint8_t>(raw.bits(32, 6))}, bool(raw.bits(62, 1)), bool(raw.bits(63, 1))}};
        } else {
            if (raw.words[1] & ~0xc7ffff00u) return "constant source address or alignment is not admitted";
            const auto bank = raw.bits(54, 5);
            if (bank >= 18) return "constant bank exceeds the established IR binding contract";
            result.sources = {ArithmeticSource{Constant{static_cast<std::uint8_t>(bank), static_cast<std::uint16_t>(raw.bits(38, 16))}, bool(raw.bits(62, 1)), bool(raw.bits(63, 1))}};
        }
        return result;
    }
    const bool multiply = (opcode & 0x1ff) == 0x20;
    const bool fma = (opcode & 0x1ff) == 0x23;
    // FMUL.RZ follows the Maxwell multiply contract: the native product
    // rounded to nearest, a documented departure of at most one ulp, with
    // both flush boundaries. Other directed modes stay refused.
    const auto rounding = raw.bits(78, 2);
    // A directed FADD with an unmodified RZ operand is exact: the mode only
    // decides the sign of a zero sum (-0 only toward minus). Other directed
    // adds round to nearest (the FMUL.RZ contract, at most one ulp).
    const bool zero_add = !multiply && !fma && rounding;
    // FMUL.RZ and FFMA.RZ follow the Maxwell toward-zero product contract:
    // the native result rounded to nearest, at most one ulp away.
    if (rounding && !((multiply || fma) && rounding == 3) && !zero_add) return "floating arithmetic rounding mode is not admitted";
    // Uniform-register forms 6 and 7 also set the uniform operand selector, bit 91.
    const bool uniform = form == 6 || form == 7;
    const std::uint32_t allowed = (multiply ? 0x0071e300u : fma ? 0x0001efffu : 0x0001e300u) | (uniform ? 0x08000000u : 0);
    if (raw.words[2] & ~allowed) return "floating arithmetic mode is not admitted";
    if (uniform && !raw.bits(91, 1)) return "uniform operand form lacks its selector";
    if (multiply && raw.bits(84, 3) != 4) {
        // Selectors 1-3 divide by 8, 4 and 2 and 5-7 multiply by 2, 4 and 8,
        // the Maxwell multiply scales in the same order around unit scale.
        constexpr std::uint32_t factors[]{0, 0x3e000000u, 0x3e800000u, 0x3f000000u, 0, 0x40000000u, 0x40800000u, 0x41000000u};
        const auto selector = raw.bits(84, 3);
        if (!selector) return "floating multiply scale is not admitted";
        result.scale = factors[selector];
    }
    result.numeric.flush_inputs = result.numeric.flush_output = raw.bits(80, 1);
    if (rounding && multiply) result.numeric.rounding = ir::Rounding::toward_zero;
    result.numeric.saturate = raw.bits(77, 1);
    result.operation = multiply ? ir::Opcode::multiply : fma ? ir::Opcode::fused_multiply_add : ir::Opcode::add;
    result.sources.push_back(general(24, raw.bits(73, 1), raw.bits(72, 1)));
    auto wide = [&]() -> std::variant<ArithmeticSource, std::string> {
        if (form == 2 || form == 4) return ArithmeticSource{Immediate{raw.words[1]}};
        if (uniform) {
            if (raw.words[1] & ~0xc000003fu) return "uniform register source mode is not admitted";
            return ArithmeticSource{Register{RegisterFile::uniform, static_cast<std::uint8_t>(raw.bits(32, 6))}, bool(raw.bits(62, 1)), bool(raw.bits(63, 1))};
        }
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
        if (fma && (form == 2 || form == 3 || form == 7)) {
            result.sources.push_back(general(64, raw.bits(74, 1), raw.bits(75, 1)));
            result.sources.push_back(value);
        } else {
            result.sources.push_back(value);
            if (fma) result.sources.push_back(general(64, raw.bits(74, 1), raw.bits(75, 1)));
        }
    }
    if (zero_add) {
        auto zero = std::find_if(result.sources.begin(), result.sources.end(), [](const ArithmeticSource& source) {
            const auto* r = std::get_if<Register>(&source.value);
            return r && r->file == RegisterFile::general && r->index == 255 && !source.absolute && !source.negate;
        });
        // Any other directed FADD follows the toward-zero product contract:
        // the native sum rounded to nearest, at most one ulp away.
        if (zero != result.sources.end() && !result.numeric.saturate) {
            result.sources.erase(zero);
            result.zero_sum_toward_minus = rounding == 1;
        }
    }
    return result;
}
} // namespace armigo::shader_native
