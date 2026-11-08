// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Native floating selection, comparison, rounding and conversion.
//
// The shared backend flushes binary32 subnormals at the integral roundings
// and comparisons but not at the extrema. Each admitted form is exact for
// its guest flush mode: a flushing minimum/maximum flushes its selected
// result, which equals selecting between flushed inputs (opposite-sign zero
// ties and NaN payloads follow the host, as in the shared extrema contract),
// and a non-flushing floor or
// ceiling restores the guest result of a subnormal input (-1 or +0 for
// floor, +1 or -0 for ceiling). Float-to-integer conversion rounds
// explicitly, clamps the unsigned lower bound and converts with the host
// cast, which saturates and maps NaN to zero like the guest NaN-to-zero
// form; narrow destinations clamp afterwards.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode signed_word{ir::Domain::signed_integer, ir::ScalarType::s32};
constexpr ir::NumericMode sign_mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
} // namespace

bool Builder::conversion(Fields& fields) {
    const unsigned base = opcode_ & 0x1ff;
    auto floating_source = [&](const AluSource& source) {
        ir::Operand value = source.value;
        if (source.absolute && fields.flag(*source.absolute)) value = temporary(ir::Opcode::absolute, {value}, sign_mode);
        if (source.negate && fields.flag(*source.negate)) value = temporary(ir::Opcode::negate, {value}, sign_mode);
        return value;
    };
    auto subnormal = [&](const ir::Operand& value) {
        auto exponent = bits(ir::Opcode::bit_and, {value, word(0x7f800000)});
        auto zero_exponent = temporary(ir::Opcode::compare, {exponent, word(0)}, unsigned_word, ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
        auto fraction = bits(ir::Opcode::bit_and, {value, word(0x007fffff)});
        auto nonzero = temporary(ir::Opcode::compare, {fraction, word(0)}, unsigned_word, ir::ComparisonInfo{ir::Compare::not_equal, false}, ir::ValueType::predicate);
        return logic(ir::Opcode::logical_and, {zero_exponent, nonzero});
    };
    auto negative = [&](const ir::Operand& value) {
        return temporary(ir::Opcode::compare, {value, word(0x80000000u)}, unsigned_word, ir::ComparisonInfo{ir::Compare::greater_equal, false}, ir::ValueType::predicate);
    };
    // Integral rounding in the guest direction, exact for either flush mode.
    auto integral = [&](ir::Operand value, unsigned direction, bool flush) {
        constexpr ir::Opcode operations[]{ir::Opcode::round, ir::Opcode::floor, ir::Opcode::ceiling, ir::Opcode::truncate};
        constexpr ir::Rounding roundings[]{ir::Rounding::nearest_even, ir::Rounding::toward_negative, ir::Rounding::toward_positive, ir::Rounding::toward_zero};
        ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, roundings[direction], flush, flush};
        auto rounded = temporary(operations[direction], {value}, mode);
        if (flush || direction == 0 || direction == 3) return rounded;
        auto sign = negative(value);
        const auto toward = direction == 1 ? bits(ir::Opcode::select, {sign, word(0xbf800000u), word(0)}, unsigned_word)
                                           : bits(ir::Opcode::select, {sign, word(0x80000000u), word(0x3f800000u)}, unsigned_word);
        return bits(ir::Opcode::select, {subnormal(value), toward, rounded}, unsigned_word);
    };
    switch (base) {
    case 0x03e: { // F2FP.PACK_AB: f16(A) in the high half, f16(B) in the low half.
        // Register form only, rounded to nearest even like the Maxwell
        // binary16 conversion; binary16 subnormal results are kept.
        if (((opcode_ >> 9) & 7) != 1) fail("float packing operand form is not admitted");
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        auto s = alu_sources(fields, 2, false);
        if (fields.take(64, 8) != 255) fail("float packing third operand is not admitted");
        if (fields.take(79, 2)) fail("float packing rounding other than nearest is not admitted");
        if (fields.flag(78) || fields.flag(72)) fail("float packing merge form is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        const ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
        write(d, temporary(ir::Opcode::pack_half2, {s[1].value, s[0].value}, mode, ir::ConversionInfo{ir::ScalarType::f32, ir::ScalarType::f16}));
        leave(fields); return true;
    }
    case 0x022: { // FSWZADD Rd, Ra, Rb, subop: a quad-lane signed sum.
        // The Maxwell swizzle-add contract: the fragment's quad lane L takes
        // the 2-bit code at bits 2L of the subop byte; code 0 is a + b, 1 is
        // b - a, 2 is a - b and 3 is b (0 * a + b). The rounding and the
        // flush mode ride every step; without FTZ, Metal's own flushing
        // applies (admitted, as Switch 1 translation emits it).
        if (opcode_ != 0x822) fail("swizzled add operand form is not admitted");
        if (source_.stage != Stage::fragment) fail("swizzled add outside a fragment stage is not admitted");
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        const auto a = reg(static_cast<unsigned>(fields.take(24, 8))), b = reg(static_cast<unsigned>(fields.take(64, 8)));
        const auto subop = static_cast<std::uint32_t>(fields.take(32, 8));
        fields.take(77, 1); // NDV: derivatives within the quad, no change to the sum
        if (fields.take(78, 2)) fail("swizzled add rounding other than nearest is not admitted");
        const bool ftz = fields.flag(80);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::IoInfo lane; lane.semantic = ir::IoSemantic::quad_lane_id; lane.type = {ir::ScalarType::u32, 1}; lane.component_mask = 1;
        auto quad = temporary(ir::Opcode::load_io, {}, unsigned_word, lane);
        auto shift = bits(ir::Opcode::shift_left, {bits(ir::Opcode::bit_and, {quad, word(3)}), word(1)});
        auto code = bits(ir::Opcode::bit_and, {bits(ir::Opcode::shift_right, {word(subop), shift}), word(3)});
        auto is = [&](std::uint32_t value) {
            return temporary(ir::Opcode::compare, {code, word(value)}, unsigned_word, ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
        };
        const auto c0 = is(0), c1 = is(1), c2 = is(2);
        auto sign = [&](std::array<std::uint32_t, 4> table) {
            auto third = bits(ir::Opcode::select, {c2, word(table[2]), word(table[3])});
            auto second = bits(ir::Opcode::select, {c1, word(table[1]), third});
            return bits(ir::Opcode::select, {c0, word(table[0]), second});
        };
        const auto sa = sign({0x3f800000u, 0xbf800000u, 0x3f800000u, 0u});
        const auto sb = sign({0x3f800000u, 0x3f800000u, 0xbf800000u, 0x3f800000u});
        ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even, ftz, ftz};
        metal_flush_ = !ftz;
        auto sum = temporary(ir::Opcode::add, {temporary(ir::Opcode::multiply, {a, sa}, mode), temporary(ir::Opcode::multiply, {b, sb}, mode)}, mode);
        metal_flush_ = false;
        write(d, sum);
        leave(fields); return true;
    }
    case 0x009: { // FMNMX: the predicate selects the minimum.
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        auto s = alu_sources(fields, 2, false);
        const auto a = floating_source(s[0]), b = floating_source(s[1]);
        auto minimum = predicate_source(fields, 87, false);
        // The host extrema may compare subnormals as zeros, so only the
        // flushing form is exact.
        if (!fields.flag(80)) fail("non-FTZ floating extrema require a subnormal-preserving backend", ir::DiagnosticCode::unsupported_capability);
        ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
        auto extremum = [&](bool lower) { return temporary(lower ? ir::Opcode::minimum : ir::Opcode::maximum, {a, b}, mode); };
        ir::Operand result;
        if (const auto* fixed = std::get_if<ir::LiteralBits>(&minimum)) result = extremum(fixed->bits != 0);
        else result = bits(ir::Opcode::select, {minimum, extremum(true), extremum(false)}, unsigned_word);
        result = bits(ir::Opcode::select, {subnormal(result), bits(ir::Opcode::bit_and, {result, word(0x80000000u)}), result}, unsigned_word);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write(d, result);
        leave(fields); return true;
    }
    case 0x008: { // FSEL: sign modifiers, then a raw-word selection.
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        auto s = alu_sources(fields, 2, false);
        const auto a = floating_source(s[0]), b = floating_source(s[1]);
        auto condition = predicate_source(fields, 87, false);
        ir::Operand result;
        if (const auto* fixed = std::get_if<ir::LiteralBits>(&condition)) result = fixed->bits ? a : b;
        else result = bits(ir::Opcode::select, {condition, a, b}, unsigned_word);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write(d, result);
        leave(fields); return true;
    }
    case 0x00a: case 0x00b: { // FSET.BF (1.0 or +0 result) and FSETP (two predicates)
        const bool to_predicate = base == 0x00b;
        enter(fields, false);
        const auto d = to_predicate ? 255u : static_cast<unsigned>(fields.take(16, 8));
        auto s = alu_sources(fields, 2, false);
        const auto a = floating_source(s[0]), b = floating_source(s[1]);
        const auto combine = static_cast<unsigned>(fields.take(74, 2)), function = static_cast<unsigned>(fields.take(76, 4));
        const bool flush = fields.flag(80);
        unsigned first = 7, second = 7;
        if (to_predicate) {
            first = static_cast<unsigned>(fields.take(81, 3)); second = static_cast<unsigned>(fields.take(84, 3));
            if (first == second && first != 7) fail("predicate destinations alias");
        }
        if (combine == 3) fail("floating predicate combination mode is not admitted");
        auto accumulator = predicate_source(fields, 87, false);
        auto mode = sign_mode; mode.flush_inputs = flush;
        ir::Operand comparison;
        if (function == 0 || function == 15) comparison = ir::LiteralBits{function == 15 ? 1u : 0u, ir::ValueType::predicate};
        else if (function == 7 || function == 8) {
            // NUM/NAN test the raw words; the flush mode is irrelevant.
            auto exact = mode; exact.flush_inputs = false;
            auto left = temporary(ir::Opcode::is_nan, {a}, exact, {}, ir::ValueType::predicate);
            auto right = temporary(ir::Opcode::is_nan, {b}, exact, {}, ir::ValueType::predicate);
            comparison = logic(ir::Opcode::logical_or, {left, right});
            if (function == 7) comparison = logic(ir::Opcode::logical_not, {comparison});
        } else {
            constexpr ir::Compare comparisons[]{ir::Compare::never, ir::Compare::less, ir::Compare::equal,
                ir::Compare::less_equal, ir::Compare::greater, ir::Compare::not_equal, ir::Compare::greater_equal};
            comparison = temporary(ir::Opcode::compare, {a, b}, mode, ir::ComparisonInfo{comparisons[function & 7], function >= 9}, ir::ValueType::predicate);
        }
        const auto combination = combine == 0 ? ir::Opcode::logical_and : combine == 1 ? ir::Opcode::logical_or : ir::Opcode::logical_xor;
        // Both results are formed before either predicate is written; the
        // accumulator may alias a destination.
        auto primary = logic(combination, {comparison, accumulator});
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        if (to_predicate) {
            auto secondary = logic(combination, {logic(ir::Opcode::logical_not, {comparison}), accumulator});
            write_predicate(first, primary); write_predicate(second, secondary);
        } else write(d, bits(ir::Opcode::select, {primary, word(0x3f800000u), word(0)}, unsigned_word));
        leave(fields); return true;
    }
    case 0x107: { // FRND.F32
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        auto s = alu_sources(fields, 1, false);
        if (fields.take(75, 2) != 2 || fields.take(84, 2) != 2) fail("integral rounding formats other than F32 are not admitted");
        const auto direction = static_cast<unsigned>(fields.take(78, 2));
        const bool flush = fields.flag(80);
        auto value = floating_source(s[1]);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write(d, integral(value, direction, flush));
        leave(fields); return true;
    }
    case 0x105: { // F2I.NTZ: round, saturate, NaN to zero.
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        auto s = alu_sources(fields, 1, false);
        const bool is_signed = fields.flag(72);
        const auto size = fields.take(75, 2);
        if (size != 1 && size != 2) fail("float-to-integer destination width is not admitted");
        if (!fields.flag(77)) fail("float-to-integer conversion without NaN-to-zero is not admitted");
        const auto direction = static_cast<unsigned>(fields.take(78, 2));
        const bool flush = fields.flag(80);
        if (fields.take(84, 2) != 2) fail("float-to-integer source formats other than F32 are not admitted");
        auto value = integral(floating_source(s[1]), direction, flush);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        if (!is_signed) value = temporary(ir::Opcode::maximum, {value, word(0)}, {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even});
        const auto carrier = is_signed ? ir::ScalarType::s32 : ir::ScalarType::u32;
        auto converted = temporary(ir::Opcode::convert, {value}, {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::toward_zero},
                                   ir::ConversionInfo{ir::ScalarType::f32, carrier});
        if (size == 1) converted = bits(ir::Opcode::clamp, {converted, word(is_signed ? 0xffff8000u : 0), word(is_signed ? 0x7fffu : 0xffffu)}, is_signed ? signed_word : unsigned_word);
        write(d, converted);
        leave(fields); return true;
    }
    case 0x043: { // F2IP.U8.F32 Rd, Ra, Rb, RZ
        // Rb's byte in bits 0-7 and Ra's in bits 8-15 (RZ leaves zero
        // there): each rounded in the direction bits 78-79 give (nearest
        // even, mode byte 0x04, or toward zero, 0xc4), saturated to an
        // unsigned byte, NaN to zero. Other modes and a third source (bits
        // 64-71) are refused.
        if (opcode_ != 0x243) return false;
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        const auto a = static_cast<unsigned>(fields.take(24, 8));
        const auto b = static_cast<unsigned>(fields.take(32, 8));
        if (fields.take(64, 8) != 255) fail("packed float-to-byte with a third source is not admitted");
        const auto mode = static_cast<unsigned>(fields.take(72, 8));
        if (mode != 0x04 && mode != 0xc4) fail("packed float-to-byte mode is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        const auto direction = mode >> 6;
        auto to_byte = [&](unsigned r) {
            auto value = temporary(ir::Opcode::maximum, {integral(reg(r), direction, false), word(0)},
                                   {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even});
            auto converted = temporary(ir::Opcode::convert, {value}, {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::toward_zero},
                                       ir::ConversionInfo{ir::ScalarType::f32, ir::ScalarType::u32});
            return bits(ir::Opcode::clamp, {converted, word(0), word(255)}, unsigned_word);
        };
        ir::Operand result = to_byte(b);
        if (a != 255) result = bits(ir::Opcode::bit_or, {result, bits(ir::Opcode::shift_left, {to_byte(a), word(8)})});
        write(d, result);
        leave(fields); return true;
    }
    case 0x106: case 0x045: { // I2F and I2FP to F32
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        auto s = alu_sources(fields, 1, false);
        const bool is_signed = fields.flag(74);
        if (fields.take(75, 2) != 2) fail("integer-to-float destinations other than F32 are not admitted");
        const auto direction = fields.take(78, 2);
        if (direction != 0 && direction != 3) fail("integer-to-float directed rounding toward infinity is not admitted");
        const auto size = fields.take(84, 2);
        if (size > 2 || (base == 0x045 && size != 2)) fail("integer-to-float source width is not admitted");
        // Narrow sources select a byte (B0-B3) or a halfword (H0-H1).
        unsigned part = 0;
        if (((opcode_ >> 9) & 7) != 4) {
            part = static_cast<unsigned>(fields.take(60, 2));
            if ((size == 1 && part > 1) || (size == 2 && part)) fail("integer-to-float subword selector is not admitted");
        } else if (size != 2) fail("narrow integer-to-float immediate source is not admitted");
        ir::Operand value = s[1].value;
        if (size < 2) value = bits(ir::Opcode::bit_extract, {value, word((size == 0 ? 8 : 16) * part), word(size == 0 ? 8 : 16)}, is_signed ? signed_word : unsigned_word);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, direction == 3 ? ir::Rounding::toward_zero : ir::Rounding::nearest_even};
        write(d, temporary(ir::Opcode::convert, {value}, mode, ir::ConversionInfo{is_signed ? ir::ScalarType::s32 : ir::ScalarType::u32, ir::ScalarType::f32}));
        leave(fields); return true;
    }
    default: return false;
    }
}
} // namespace armigo::shader_native::detail
