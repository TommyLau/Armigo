// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Packed binary16 arithmetic: HADD2, HMUL2, HFMA2 and HMNMX2.
//
// Each lane is widened, modified (absolute, then negate), computed in
// binary32 and, unless the result is scalar, packed nearest-even. FTZ
// flushes binary16 inputs at the widen and binary16 results at the pack;
// the lane operation carries that same flush and the unit clamp. Without
// FTZ the lane add or fused multiply-add is admitted with Metal's own
// binary32 subnormal flushing. A widened binary16 value is a normal
// binary32 number or a zero, so that flush does not change it; a source
// read as binary32 still flushes there.
//
// HFMA2's clamp, when its predicate holds, runs on the packed lanes: a
// negative lane, including negative zero and negative infinity, becomes
// +0, positive infinity is kept, and either NaN becomes the all-ones
// binary16 NaN. A binary32 maximum against zero would turn a NaN into +0.
// Saturation and a scalar result are refused together with that clamp.
//
// HMNMX2 compares the widened lanes and packs them. Binary16 subnormals
// stay, because they widen to normal binary32 values. Opposite-sign zero
// ties and a two-NaN payload follow the shared extrema contract. Flush,
// a scalar result and the NaN or sign modes are refused.
//
// HFMA2's register form reads the product's right factor from a register,
// with that register's own selector and sign. The immediate form is unchanged.
//
// HMUL2 multiplies the widened lanes and packs them. The register form
// reads the right factor from a register, with that register's selector
// and sign. The immediate form is the packed pair in the instruction
// word, with no selector or sign of its own. Saturation and FTZ follow
// the fused multiply-add. A denormal product, a scalar result and the
// other mode bits are refused.
//
// HSETP2 compares the two lanes and combines each result with the input
// predicate. With the conjunction bit, the first predicate is that
// conjunction and the second is its complement; otherwise each lane keeps
// its own predicate. Both results are formed before either predicate is
// written. FTZ, after absolute and negate, replaces a magnitude below the
// smallest normal binary16 with +0. FTZ uses the binary16 bound for every
// lane, a binary32 selector included. The immediate second source is the
// packed pair in the instruction word, with no selector or sign of its own.
// The register form reads that source from a register, with the register's
// selector, absolute and negate.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode sign_mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }

// A packed source. An immediate has no selector or sign modifier: those bits
// are the immediate. Selector 0 keeps both halves, 1 broadcasts the register
// as one binary32 value, 2 repeats the low half and 3 repeats the high half.
struct Pair {
    ir::Operand value;
    unsigned swizzle{};
    bool absolute{};
    bool negate{};
    bool immediate{};
};
} // namespace

bool Builder::half_precision(Fields& fields) {
    const unsigned base = opcode_ & 0x1ff;
    if (base == 0x034) {
        // HSETP2. The immediate form's second source is the packed pair.
        // The register form's second source is a register: selector at
        // bits 60-61, absolute at 62, negate at 63. A set bit in 40-59
        // or 64-68 is not a source.
        if (opcode_ != 0x434 && opcode_ != 0x234) return false;
        enter(fields, false);
        const auto ra = static_cast<unsigned>(fields.take(24, 8));
        const Pair first{reg(ra), static_cast<unsigned>(fields.take(74, 2)), fields.flag(73), fields.flag(72), false};
        const Pair second = opcode_ == 0x234
            ? Pair{reg(static_cast<unsigned>(fields.take(32, 8))), static_cast<unsigned>(fields.take(60, 2)), fields.flag(62), fields.flag(63), false}
            : Pair{word(static_cast<std::uint32_t>(fields.take(32, 32))), 0, false, false, true};
        const auto combine = static_cast<unsigned>(fields.take(69, 2));
        const bool conjunct = fields.flag(71);
        const auto function = static_cast<unsigned>(fields.take(76, 4));
        const bool ftz = fields.flag(80);
        const auto first_pred = static_cast<unsigned>(fields.take(81, 3));
        const auto second_pred = static_cast<unsigned>(fields.take(84, 3));
        if (combine == 3) fail("half predicate combination mode is not admitted");
        if (first_pred == second_pred && first_pred != 7) fail("predicate destinations alias");
        auto accumulator = predicate_source(fields, 87, false);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        // Widen without flushing. FTZ is applied after the sign modifiers,
        // so a subnormal keeps its magnitude until then.
        auto unpack = [&](ir::Operand packed) {
            if (p_.values.size() + 2 > p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
            const auto base_id = static_cast<std::uint32_t>(p_.values.size());
            p_.values.push_back({ir::ValueType::word32}); p_.values.push_back({ir::ValueType::word32});
            std::array<ir::Operand, 2> out{ir::ValueId{base_id}, ir::ValueId{base_id + 1}};
            operation({{}, ir::Opcode::unpack_half2, {}, sign_mode, {std::move(packed)}, {out[0], out[1]},
                       ir::ConversionInfo{ir::ScalarType::f16, ir::ScalarType::f32}});
            return out;
        };
        auto lanes = [&](const Pair& source) {
            auto modify = [&](ir::Operand value) {
                if (source.absolute) value = temporary(ir::Opcode::absolute, {value}, sign_mode);
                if (source.negate) value = temporary(ir::Opcode::negate, {value}, sign_mode);
                return value;
            };
            if (!source.immediate && source.swizzle == 1) {
                auto lane = modify(source.value);
                return std::array<ir::Operand, 2>{lane, lane};
            }
            auto halves = unpack(source.value);
            if (source.swizzle >= 2) {
                auto lane = modify(halves[source.swizzle - 2]);
                return std::array<ir::Operand, 2>{lane, lane};
            }
            return std::array<ir::Operand, 2>{modify(halves[0]), modify(halves[1])};
        };
        // The smallest normal binary16, as binary32. Anything smaller, including
        // a widened subnormal, becomes +0. A NaN is not smaller.
        auto flush_tiny = [&](ir::Operand value) {
            if (!ftz) return value;
            auto magnitude = temporary(ir::Opcode::absolute, {value}, sign_mode);
            auto tiny = temporary(ir::Opcode::compare, {magnitude, word(0x38800000u)},
                                  ir::NumericMode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even, true},
                                  ir::ComparisonInfo{ir::Compare::less, false}, ir::ValueType::predicate);
            return bits(ir::Opcode::select, {tiny, word(0), std::move(value)}, unsigned_word);
        };
        auto left = lanes(first);
        auto right = lanes(second);
        auto compare_lane = [&](unsigned lane) {
            auto a = flush_tiny(left[lane]);
            auto b = flush_tiny(right[lane]);
            if (function == 0 || function == 15)
                return ir::Operand{ir::LiteralBits{function == 15 ? 1u : 0u, ir::ValueType::predicate}};
            if (function == 7 || function == 8) {
                auto left_nan = temporary(ir::Opcode::is_nan, {a}, sign_mode, {}, ir::ValueType::predicate);
                auto right_nan = temporary(ir::Opcode::is_nan, {b}, sign_mode, {}, ir::ValueType::predicate);
                auto nan = logic(ir::Opcode::logical_or, {left_nan, right_nan});
                return function == 7 ? logic(ir::Opcode::logical_not, {nan}) : nan;
            }
            constexpr ir::Compare comparisons[]{ir::Compare::never, ir::Compare::less, ir::Compare::equal,
                ir::Compare::less_equal, ir::Compare::greater, ir::Compare::not_equal, ir::Compare::greater_equal};
            auto mode = sign_mode;
            // Widened binary16 lanes are normal binary32 values or zeros, so
            // this flush does not change them. A binary32 source still flushes.
            if (ftz) mode.flush_inputs = true;
            else metal_flush_ = true;
            auto comparison = temporary(ir::Opcode::compare, {std::move(a), std::move(b)}, mode,
                                        ir::ComparisonInfo{comparisons[function & 7], function >= 9}, ir::ValueType::predicate);
            metal_flush_ = false;
            return comparison;
        };
        const auto combination = combine == 0 ? ir::Opcode::logical_and : combine == 1 ? ir::Opcode::logical_or : ir::Opcode::logical_xor;
        auto low = logic(combination, {compare_lane(0), accumulator});
        auto high = logic(combination, {compare_lane(1), accumulator});
        if (conjunct) {
            auto both = logic(ir::Opcode::logical_and, {low, high});
            write_predicate(first_pred, both);
            write_predicate(second_pred, logic(ir::Opcode::logical_not, {both}));
        } else {
            write_predicate(first_pred, low);
            write_predicate(second_pred, high);
        }
        leave(fields); return true;
    }
    if (base != 0x030 && base != 0x031 && base != 0x032 && base != 0x040) return false;
    const bool add = opcode_ == 0x230, fma = opcode_ == 0x831 || opcode_ == 0x231,
               mul = opcode_ == 0x232 || opcode_ == 0x832, extremum = opcode_ == 0x240;
    if (!add && !fma && !mul && !extremum) fail("half arithmetic operand form is not admitted");
    enter(fields, false);
    const auto dest = static_cast<unsigned>(fields.take(16, 8));
    const auto ra = static_cast<unsigned>(fields.take(24, 8));
    const bool a_negate = fields.flag(72), a_absolute = fields.flag(73);
    const auto a_swizzle = static_cast<unsigned>(fields.take(74, 2));
    Pair second{};
    Pair third{};
    if (opcode_ == 0x831 || opcode_ == 0x832) {
        // The product's right factor is the 32-bit immediate. A fused
        // multiply-add also reads its addend from the upper register.
        second = {word(static_cast<std::uint32_t>(fields.take(32, 32))), 0, false, false, true};
        if (fma) third = {reg(static_cast<unsigned>(fields.take(64, 8))), static_cast<unsigned>(fields.take(81, 2)), fields.flag(83), fields.flag(84), false};
    } else if (fma) {
        // Register form: the right factor has its selector at bits 60-61 and
        // its sign at bits 62-63. Bits 40-59 are not a source.
        second = {reg(static_cast<unsigned>(fields.take(32, 8))), static_cast<unsigned>(fields.take(60, 2)), fields.flag(62), fields.flag(63), false};
        third = {reg(static_cast<unsigned>(fields.take(64, 8))), static_cast<unsigned>(fields.take(81, 2)), fields.flag(83), fields.flag(84), false};
    } else {
        second = {reg(static_cast<unsigned>(fields.take(32, 8))), static_cast<unsigned>(fields.take(60, 2)), fields.flag(62), fields.flag(63), false};
    }
    const bool saturate = fields.flag(77), scalar = fields.flag(78);
    const bool ftz = extremum ? false : fields.flag(80);
    bool clamp = false, exchange = false;
    ir::Operand enabled;
    if (add) {
        if (fields.flag(76) || fields.flag(79) || fields.take(81, 11)) fail("half addition mode is not admitted");
    } else if (fma) {
        // Bit 76 would be a denormal product mode. It is not the flush bit.
        if (fields.flag(76)) fail("half fused multiply-add denormal mode is not admitted");
        clamp = fields.flag(79);
        exchange = fields.flag(86);
        if (fields.flag(85) || fields.flag(91)) fail("half fused multiply-add alternate format is not admitted");
        if (clamp && (saturate || scalar)) fail("half fused multiply-add clamp conflicts with saturation or a scalar result");
        if (clamp) enabled = predicate_source(fields, 87, false);
        else if (fields.take(87, 4)) fail("half fused multiply-add predicate is not admitted without the clamp");
        // Swapping halves and then selecting one half is a different source
        // than the mixed-sign form, which exchanges the unmodified pair.
        if (opcode_ == 0x231 && exchange && (second.swizzle != 0 || second.absolute))
            fail("half fused multiply-add mixed-sign source is not admitted");
    } else if (mul) {
        // Bit 76 would be a denormal product mode. It is not the flush bit.
        // Bits 78-79 are a scalar result and an unused mode. Bits 81-91
        // are not a half-multiply field.
        if (fields.flag(76)) fail("half multiply denormal mode is not admitted");
        if (fields.take(78, 2) || fields.take(81, 11)) fail("half multiply mode is not admitted");
    } else {
        if (fields.take(76, 11) || fields.flag(91)) fail("half minimum/maximum mode is not admitted");
        if (a_swizzle == 1 || second.swizzle == 1) fail("half minimum/maximum binary32 source is not admitted");
        enabled = predicate_source(fields, 87, false);
    }
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");

    const Pair first{reg(ra), a_swizzle, a_absolute, a_negate, false};
    // Negated RZ is -0 in every lane. Added to a widened half, with no flush
    // and no saturation, it leaves that half unchanged, including a binary16
    // subnormal and a NaN payload, so the scalar result is the widened half.
    const bool widen = add && scalar && !ftz && !saturate && ra == 255 && a_negate && second.swizzle != 1;
    auto unpack = [&](ir::Operand packed) {
        if (p_.values.size() + 2 > p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const auto base_id = static_cast<std::uint32_t>(p_.values.size());
        p_.values.push_back({ir::ValueType::word32}); p_.values.push_back({ir::ValueType::word32});
        std::array<ir::Operand, 2> out{ir::ValueId{base_id}, ir::ValueId{base_id + 1}};
        const ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even, ftz, false};
        operation({{}, ir::Opcode::unpack_half2, {}, mode, {std::move(packed)}, {out[0], out[1]},
                   ir::ConversionInfo{ir::ScalarType::f16, ir::ScalarType::f32}});
        return out;
    };
    auto lanes = [&](const Pair& source) {
        auto modify = [&](ir::Operand value) {
            if (source.absolute) value = temporary(ir::Opcode::absolute, {value}, sign_mode);
            if (source.negate) value = temporary(ir::Opcode::negate, {value}, sign_mode);
            return value;
        };
        if (!source.immediate && source.swizzle == 1) {
            auto lane = modify(source.value);
            return std::array<ir::Operand, 2>{lane, lane};
        }
        auto halves = unpack(source.value);
        if (source.swizzle >= 2) {
            auto lane = modify(halves[source.swizzle - 2]);
            return std::array<ir::Operand, 2>{lane, lane};
        }
        return std::array<ir::Operand, 2>{modify(halves[0]), modify(halves[1])};
    };
    if (widen) {
        write(dest, lanes(second)[second.swizzle == 3 ? 1u : 0u]);
        leave(fields); return true;
    }
    auto left = lanes(first);
    auto right = lanes(second);
    if (exchange) right = {temporary(ir::Opcode::negate, {right[1]}, sign_mode), right[0]};
    std::array<ir::Operand, 2> addend{};
    if (fma) addend = lanes(third);
    auto binary = [&](unsigned lane) {
        // Metal flushes a binary32 subnormal at this operation even without
        // FTZ. Widened binary16 lanes are normal binary32 values or zeros.
        ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even, ftz, ftz, saturate};
        if (!ftz) metal_flush_ = true;
        auto value = fma ? temporary(ir::Opcode::fused_multiply_add, {left[lane], right[lane], addend[lane]}, mode)
                         : mul ? temporary(ir::Opcode::multiply, {left[lane], right[lane]}, mode)
                               : temporary(ir::Opcode::add, {left[lane], right[lane]}, mode);
        metal_flush_ = false;
        return value;
    };
    auto pack = [&](ir::Operand low, ir::Operand high) {
        const ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even, false, ftz};
        return temporary(ir::Opcode::pack_half2, {std::move(low), std::move(high)}, mode,
                         ir::ConversionInfo{ir::ScalarType::f32, ir::ScalarType::f16});
    };
    if (extremum) {
        const ir::NumericMode mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
        const auto* fixed = std::get_if<ir::LiteralBits>(&enabled);
        auto pick = [&](unsigned lane) {
            if (fixed) return temporary(fixed->bits ? ir::Opcode::minimum : ir::Opcode::maximum, {left[lane], right[lane]}, mode);
            auto lower = temporary(ir::Opcode::minimum, {left[lane], right[lane]}, mode);
            auto upper = temporary(ir::Opcode::maximum, {left[lane], right[lane]}, mode);
            return bits(ir::Opcode::select, {enabled, lower, upper}, unsigned_word);
        };
        write(dest, pack(pick(0), pick(1)));
        leave(fields); return true;
    }
    if (scalar) {
        write(dest, binary(0));
        leave(fields); return true;
    }
    auto packed = pack(binary(0), binary(1));
    if (clamp) {
        // Predicate false keeps the fused result, including negative zero.
        const auto* fixed = std::get_if<ir::LiteralBits>(&enabled);
        if (!fixed || fixed->bits) {
            auto lane = [&](std::uint32_t shift) {
                auto low = shift ? bits(ir::Opcode::shift_right, {packed, word(shift)}) : packed;
                low = bits(ir::Opcode::bit_and, {low, word(0xffff)});
                auto magnitude = bits(ir::Opcode::bit_and, {low, word(0x7fff)});
                auto nan = temporary(ir::Opcode::compare, {magnitude, word(0x7c00)}, unsigned_word,
                                     ir::ComparisonInfo{ir::Compare::greater, false}, ir::ValueType::predicate);
                auto negative = temporary(ir::Opcode::compare, {bits(ir::Opcode::bit_and, {low, word(0x8000)}), word(0)}, unsigned_word,
                                          ir::ComparisonInfo{ir::Compare::not_equal, false}, ir::ValueType::predicate);
                auto cleared = bits(ir::Opcode::select, {negative, word(0), low}, unsigned_word);
                return bits(ir::Opcode::select, {nan, word(0x7fff), cleared}, unsigned_word);
            };
            auto clamped = bits(ir::Opcode::bit_or, {lane(0), bits(ir::Opcode::shift_left, {lane(16), word(16)})});
            packed = fixed ? clamped : bits(ir::Opcode::select, {enabled, clamped, packed}, unsigned_word);
        }
    }
    write(dest, packed);
    leave(fields); return true;
}
} // namespace armigo::shader_native::detail
