// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Native integer, bitwise and predicate-logic instructions, in their vector
// and uniform-datapath forms. Operand slots follow the native ALU encoding:
// source A at bits 24-31, then the form selector (bits 9-11) places register,
// immediate, constant or uniform sources at bits 32-63 and 64-71. A uniform
// operation reads and writes only uniform registers and predicates.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
constexpr ir::NumericMode word_bits{ir::Domain::bits, ir::ScalarType::u32};
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode signed_word{ir::Domain::signed_integer, ir::ScalarType::s32};
constexpr ir::NumericMode boolean{ir::Domain::boolean, ir::ScalarType::boolean};
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
std::optional<std::uint32_t> literal(const ir::Operand& value) {
    if (const auto* bits = std::get_if<ir::LiteralBits>(&value); bits && bits->type == ir::ValueType::word32)
        return static_cast<std::uint32_t>(bits->bits);
    return std::nullopt;
}
// Base operation selectors (bits 0-8) of the native ALU families handled here.
enum Base : unsigned {
    select_base = 0x007, set_predicate = 0x00c, add3 = 0x010, scaled_add = 0x011, lookup3 = 0x012,
    absolute_base = 0x013, permute = 0x016, extremum = 0x017, funnel = 0x019, sign_extend = 0x01a,
    multiply_add = 0x024, multiply_wide = 0x025, multiply_high = 0x027,
    uniform_select = 0x087, uniform_set_predicate = 0x08c, uniform_add3 = 0x090, uniform_scaled_add = 0x091,
    uniform_lookup3 = 0x092, uniform_permute = 0x096, uniform_funnel = 0x099, uniform_sign_extend = 0x09a,
    uniform_multiply_add = 0x0a4, uniform_multiply_wide = 0x0a5, uniform_leading_one = 0x0bd, uniform_reverse = 0x0be, uniform_population = 0x0bf,
    leading_one = 0x100, reverse = 0x101, population = 0x109,
};
} // namespace

ir::Operand Builder::bits(ir::Opcode opcode, std::vector<ir::Operand> inputs, ir::NumericMode mode) {
    return temporary(opcode, std::move(inputs), mode, {}, ir::storage_type(mode.scalar));
}
ir::Operand Builder::logic(ir::Opcode opcode, std::vector<ir::Operand> inputs) {
    return temporary(opcode, std::move(inputs), boolean, {}, ir::ValueType::predicate);
}
// Three-input lookup table: source 0 selects table bits 4-7, source 1 bits
// 2-3 and 6-7, source 2 the odd bits. The expression is a Shannon expansion
// on the sources in order, with constant and single-variable cofactors folded.
ir::Operand Builder::lookup(unsigned table, const std::array<ir::Operand, 3>& sources, bool predicates) {
    const ir::Operand zero = predicates ? ir::Operand{ir::LiteralBits{0, ir::ValueType::predicate}} : word(0);
    const ir::Operand ones = predicates ? ir::Operand{ir::LiteralBits{1, ir::ValueType::predicate}} : word(~0u);
    auto apply = [&](ir::Opcode bitwise, ir::Opcode logical, std::vector<ir::Operand> inputs) {
        return predicates ? logic(logical, std::move(inputs)) : bits(bitwise, std::move(inputs));
    };
    auto invert = [&](ir::Operand value) { return apply(ir::Opcode::bit_not, ir::Opcode::logical_not, {std::move(value)}); };
    // Each node is the function's truth table over the remaining sources.
    auto build = [&](auto&& self, unsigned values, unsigned depth) -> ir::Operand {
        const unsigned width = 1u << (3 - depth), mask = (1u << width) - 1;
        values &= mask;
        if (values == 0) return zero;
        if (values == mask) return ones;
        const unsigned half = width / 2;
        const unsigned high = values >> half, low = values & ((1u << half) - 1), half_mask = (1u << half) - 1;
        const auto& top = sources[depth];
        if (high == low) return self(self, low, depth + 1);
        if (low == 0 && high == half_mask) return top;
        if (high == 0 && low == half_mask) return invert(top);
        if (low == 0) return apply(ir::Opcode::bit_and, ir::Opcode::logical_and, {top, self(self, high, depth + 1)});
        if (high == 0) return apply(ir::Opcode::bit_and, ir::Opcode::logical_and, {invert(top), self(self, low, depth + 1)});
        if (low == half_mask) return apply(ir::Opcode::bit_or, ir::Opcode::logical_or, {invert(top), self(self, high, depth + 1)});
        if (high == half_mask) return apply(ir::Opcode::bit_or, ir::Opcode::logical_or, {top, self(self, low, depth + 1)});
        auto low_value = self(self, low, depth + 1);
        if (high == (~low & half_mask)) return apply(ir::Opcode::bit_xor, ir::Opcode::logical_xor, {top, low_value});
        auto high_value = self(self, high, depth + 1);
        auto difference = apply(ir::Opcode::bit_xor, ir::Opcode::logical_xor, {high_value, low_value});
        return apply(ir::Opcode::bit_xor, ir::Opcode::logical_xor,
                     {low_value, apply(ir::Opcode::bit_and, ir::Opcode::logical_and, {top, difference})});
    };
    return build(build, table & 0xff, 0);
}
// The carry-out is formed with unsigned comparisons: the first sum wraps
// below its left operand, or a carry-in wraps the second sum to zero. (The
// shared IR's combined carry operation is not used.)
std::array<ir::Operand, 2> Builder::add_with_carry(ir::Operand left, ir::Operand right, ir::Operand carry) {
    auto compare = [&](ir::Operand a, ir::Operand b, ir::Compare relation) {
        return temporary(ir::Opcode::compare, {std::move(a), std::move(b)}, unsigned_word, ir::ComparisonInfo{relation, false}, ir::ValueType::predicate);
    };
    auto sum = bits(ir::Opcode::add, {left, right}, unsigned_word);
    auto out = compare(sum, left, ir::Compare::less);
    if (const auto* fixed = std::get_if<ir::LiteralBits>(&carry); fixed && !fixed->bits) return {sum, out};
    auto total = bits(ir::Opcode::add, {sum, bits(ir::Opcode::select, {carry, word(1), word(0)}, unsigned_word)}, unsigned_word);
    auto wrapped = logic(ir::Opcode::logical_and, {carry, compare(total, word(0), ir::Compare::equal)});
    return {total, logic(ir::Opcode::logical_or, {out, wrapped})};
}
std::array<AluSource, 3> Builder::alu_sources(Fields& fields, unsigned count, bool uniform) {
    const auto form = fields.raw().bits(9, 3);
    auto general = [&](unsigned at) { return reg(static_cast<unsigned>(fields.take(at, 8))); };
    auto uniform_register = [&](unsigned at) {
        const auto index = static_cast<unsigned>(fields.take(at, 6));
        if (fields.take(at + 6, 2)) fail("uniform register field extension is not admitted");
        return this->uniform(index);
    };
    auto vector = [&](unsigned at) { return uniform ? uniform_register(at) : general(at); };
    auto constant_source = [&]() -> ir::Operand {
        if (uniform) fail("uniform operation has no constant-buffer operand form");
        if (fields.flag(91)) fail("uniform-register indexed constant bank is not admitted");
        if (fields.take(32, 6)) fail("constant operand index field is not admitted");
        const auto offset = static_cast<unsigned>(fields.take(38, 16));
        return constant(static_cast<unsigned>(fields.take(54, 5)), offset);
    };
    auto immediate = [&]() -> ir::Operand { return word(static_cast<std::uint32_t>(fields.take(32, 32))); };
    std::array<AluSource, 3> s{};
    const bool third = count == 3;
    switch (form) {
    case 1:
        s[1] = {vector(32), 62, 63};
        if (third) s[2] = {vector(64), 74, 75};
        break;
    case 2:
        if (!third) fail("operand form requires a third source");
        s[1] = {vector(64), 74, 75}; s[2] = {immediate(), {}, {}};
        break;
    case 3:
        if (!third) fail("operand form requires a third source");
        s[1] = {vector(64), 74, 75}; s[2] = {constant_source(), 62, 63};
        break;
    case 4:
        s[1] = {immediate(), {}, {}};
        if (third) s[2] = {vector(64), 74, 75};
        break;
    case 5:
        s[1] = {constant_source(), 62, 63};
        if (third) s[2] = {vector(64), 74, 75};
        break;
    case 6:
        if (uniform || !fields.flag(91)) fail("uniform-register operand form is not admitted");
        s[1] = {uniform_register(32), 62, 63};
        if (third) s[2] = {general(64), 74, 75};
        break;
    case 7:
        if (!third || uniform || !fields.flag(91)) fail("uniform-register operand form is not admitted");
        s[1] = {general(64), 74, 75}; s[2] = {uniform_register(32), 62, 63};
        break;
    default: fail("operand form is not admitted");
    }
    if (count >= 2) s[0] = {vector(24), 73, 72};
    if (uniform && !fields.flag(91)) fail("uniform operation lacks its datapath selector");
    return s;
}
ir::Operand Builder::modified_integer(Fields& fields, const AluSource& source, bool allow_negate) {
    if (!allow_negate || !source.negate || !fields.flag(*source.negate)) return source.value;
    return bits(ir::Opcode::negate, {source.value}, signed_word);
}

bool Builder::integer(Fields& fields) {
    const unsigned base = opcode_ & 0x1ff, form = (opcode_ >> 9) & 7;
    const bool uniform = (base >= 0x080 && base <= 0x0bf) || opcode_ == 0x89c;
    auto destination = [&] { return static_cast<unsigned>(uniform ? fields.take(16, 6) : fields.take(16, 8)); };
    auto write_destination = [&](unsigned index, ir::Operand value) {
        if (uniform) write_uniform(index, std::move(value)); else write(index, std::move(value));
    };
    auto write_flag = [&](unsigned index, ir::Operand value) {
        if (uniform) write_uniform_predicate(index, std::move(value)); else write_predicate(index, std::move(value));
    };
    auto constant_false = [&](unsigned first) {
        if (fields.take(first, 3) != 7 || !fields.flag(first + 3)) fail("carry or predicate input is not the constant false form");
    };
    auto uniform_tail = [&] { if (uniform && fields.take(22, 2)) fail("uniform destination field extension is not admitted"); };
    auto as_word = [&](ir::Operand flag) { return bits(ir::Opcode::select, {std::move(flag), word(1), word(0)}, unsigned_word); };
    auto not_equal_zero = [&](ir::Operand value) {
        return temporary(ir::Opcode::compare, {std::move(value), word(0)}, unsigned_word, ir::ComparisonInfo{ir::Compare::not_equal, false}, ir::ValueType::predicate);
    };

    if (opcode_ == 0x81c || opcode_ == 0x89c) {
        // PLOP3: two lookup-table results over three predicates.
        enter(fields, uniform);
        const bool uniform_third = !uniform && fields.flag(67);
        std::array<ir::Operand, 3> sources{predicate_source(fields, 87, uniform), predicate_source(fields, 77, uniform),
                                           predicate_source(fields, 68, uniform || uniform_third)};
        const unsigned first_table = static_cast<unsigned>(fields.take(64, 3) | (fields.take(72, 5) << 3));
        const unsigned second_table = static_cast<unsigned>(fields.take(16, 8));
        const auto first = lookup(first_table, sources, true), second = lookup(second_table, sources, true);
        const auto first_target = static_cast<unsigned>(fields.take(81, 3)), second_target = static_cast<unsigned>(fields.take(84, 3));
        if (first_target == second_target && first_target != 7) fail("predicate destinations alias");
        write_flag(first_target, first); write_flag(second_target, second);
        leave(fields); return true;
    }
    if (opcode_ == 0x804) {
        // R2P PR, Ra, mask: P0-P6 take the bits of the selected byte of Ra
        // whose mask bit is set; the others keep their values.
        enter(fields, false);
        const auto source = reg(static_cast<unsigned>(fields.take(24, 8)));
        const auto mask = static_cast<unsigned>(fields.take(32, 32));
        const auto byte = static_cast<unsigned>(fields.take(76, 2));
        if (mask > 0x7f) fail("register-to-predicate mask beyond P6 is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        for (unsigned index = 0; index < 7; ++index) if (mask & (1u << index))
            write_predicate(index, not_equal_zero(bits(ir::Opcode::bit_and, {source, word(1u << (8 * byte + index))})));
        leave(fields); return true;
    }
    switch (base) {
    case add3: case uniform_add3: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 3, uniform);
        const bool extended = fields.flag(74);
        const auto first_out = static_cast<unsigned>(fields.take(81, 3)), second_out = static_cast<unsigned>(fields.take(84, 3));
        if (first_out == second_out && first_out != 7) fail("carry destinations alias");
        std::array<bool, 3> inverted{};
        for (unsigned i = 0; i < 3; ++i) inverted[i] = s[i].negate && fields.flag(*s[i].negate);
        std::array<ir::Operand, 2> carries{ir::LiteralBits{0, ir::ValueType::predicate}, ir::LiteralBits{0, ir::ValueType::predicate}};
        if (extended) { carries[0] = predicate_source(fields, 87, uniform); carries[1] = predicate_source(fields, 77, uniform); }
        else { constant_false(87); constant_false(77); }
        // A negated source contributes ~x + 1 as a 33-bit term; the extended
        // form adds ~x and the two carry predicates instead.
        ir::Operand result;
        std::optional<std::array<ir::Operand, 2>> flags;
        if (first_out == 7 && second_out == 7) {
            result = word(0);
            for (unsigned i = 0; i < 3; ++i) {
                auto term = inverted[i] ? bits(extended ? ir::Opcode::bit_not : ir::Opcode::negate, {s[i].value}, extended ? word_bits : signed_word) : s[i].value;
                result = literal(result) == 0u ? term : bits(ir::Opcode::add, {result, term}, unsigned_word);
            }
            // A constant carry adds its literal value; PT carries in one.
            for (auto& carry : carries) {
                if (const auto* fixed = std::get_if<ir::LiteralBits>(&carry)) {
                    if (fixed->bits) result = bits(ir::Opcode::add, {result, word(1)}, unsigned_word);
                } else result = bits(ir::Opcode::add, {result, as_word(carry)}, unsigned_word);
            }
        } else {
            // Three 32-bit carry additions sum the inverted terms and every
            // +1 (a negation's, or a carry input's); the result is the low
            // word and the three carry-outs count the multiples of 2^32.
            std::array<ir::Operand, 3> ones{carries[0], carries[1], ir::LiteralBits{0, ir::ValueType::predicate}};
            if (!extended) for (unsigned i = 0; i < 3; ++i) ones[i] = ir::LiteralBits{inverted[i] ? 1u : 0u, ir::ValueType::predicate};
            std::array<ir::Operand, 3> terms{};
            for (unsigned i = 0; i < 3; ++i) terms[i] = inverted[i] ? bits(ir::Opcode::bit_not, {s[i].value}) : s[i].value;
            auto [first_sum, first_carry] = add_with_carry(terms[0], terms[1], ones[0]);
            auto [second_sum, second_carry] = add_with_carry(first_sum, terms[2], ones[1]);
            auto [third_sum, third_carry] = add_with_carry(second_sum, word(0), ones[2]);
            result = third_sum;
            auto any = logic(ir::Opcode::logical_or, {logic(ir::Opcode::logical_or, {first_carry, second_carry}), third_carry});
            auto pair = logic(ir::Opcode::logical_or, {logic(ir::Opcode::logical_and, {first_carry, second_carry}),
                logic(ir::Opcode::logical_and, {third_carry, logic(ir::Opcode::logical_or, {first_carry, second_carry})})});
            flags = std::array<ir::Operand, 2>{any, pair};
        }
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write_destination(d, result);
        if (flags) { write_flag(first_out, (*flags)[0]); write_flag(second_out, (*flags)[1]); }
        leave(fields); return true;
    }
    case scaled_add: case uniform_scaled_add: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 3, uniform);
        const bool extended = fields.flag(74), high_part = fields.flag(80), inverted = fields.flag(72);
        // .SX32 (bit 73): with .HI the high word is A's sign extension, not a
        // source; the encoding leaves the high slot at RZ.
        const bool sign_extended = fields.flag(73);
        if (sign_extended && (!high_part || literal(s[2].value) != 0u))
            fail("sign-extended scaled add requires .HI and a zero high source");
        const auto shift = static_cast<unsigned>(fields.take(75, 5));
        // Bit 75 starts the shift field, so a source in the upper slot has no sign modifier.
        const bool negate_b = s[1].negate && *s[1].negate != 75 && fields.flag(*s[1].negate);
        const auto overflow_target = static_cast<unsigned>(fields.take(81, 3));
        if (!high_part && literal(s[2].value) != 0u) fail("scaled add without .HI requires a zero high source");
        ir::Operand carry = ir::LiteralBits{0, ir::ValueType::predicate};
        if (extended) carry = predicate_source(fields, 87, uniform); else constant_false(87);
        const ir::Operand high = sign_extended ? bits(ir::Opcode::shift_right, {s[0].value, word(31)}, signed_word) : s[2].value;
        ir::Operand shifted = high_part ? bits(ir::Opcode::funnel_shift_left, {s[0].value, high, word(shift)})
                                        : (shift ? bits(ir::Opcode::shift_left, {s[0].value, word(shift)}) : s[0].value);
        ir::Operand addend = s[1].value;
        std::vector<ir::Operand> overflow;
        if (inverted) {
            if (!extended) overflow.push_back(temporary(ir::Opcode::compare, {shifted, word(0)}, unsigned_word, ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate));
            shifted = bits(extended ? ir::Opcode::bit_not : ir::Opcode::negate, {shifted}, extended ? word_bits : signed_word);
        }
        if (negate_b) {
            if (!extended) overflow.push_back(temporary(ir::Opcode::compare, {addend, word(0)}, unsigned_word, ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate));
            addend = bits(extended ? ir::Opcode::bit_not : ir::Opcode::negate, {addend}, extended ? word_bits : signed_word);
        }
        ir::Operand result;
        const auto* fixed_carry = std::get_if<ir::LiteralBits>(&carry);
        if (overflow_target == 7 && fixed_carry && !fixed_carry->bits) {
            result = bits(ir::Opcode::add, {shifted, addend}, unsigned_word);
        } else {
            auto [sum, carry_out] = add_with_carry(shifted, addend, carry);
            result = sum; overflow.push_back(carry_out);
        }
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write_destination(d, result);
        if (overflow_target != 7) {
            ir::Operand flag = overflow.empty() ? ir::Operand{ir::LiteralBits{0, ir::ValueType::predicate}} : overflow[0];
            for (std::size_t i = 1; i < overflow.size(); ++i) flag = logic(ir::Opcode::logical_or, {flag, overflow[i]});
            write_flag(overflow_target, flag);
        }
        leave(fields); return true;
    }
    case lookup3: case uniform_lookup3: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 3, uniform);
        const auto table = static_cast<unsigned>(fields.take(72, 8));
        if (fields.flag(80)) fail("predicate conjunction output mode is not admitted");
        constant_false(87);
        const auto target = static_cast<unsigned>(fields.take(81, 3));
        const auto result = lookup(table, {s[0].value, s[1].value, s[2].value}, false);
        // The predicate output reports a nonzero result.
        std::optional<ir::Operand> flag;
        if (target != 7) flag = not_equal_zero(result);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write_destination(d, result);
        if (flag) write_flag(target, *flag);
        leave(fields); return true;
    }
    case absolute_base: {
        if (uniform) return false;
        enter(fields, false);
        const auto d = destination();
        auto s = alu_sources(fields, 1, false);
        write(d, bits(ir::Opcode::absolute, {s[1].value}, signed_word));
        leave(fields); return true;
    }
    case permute: case uniform_permute: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 3, uniform);
        if (fields.take(72, 3)) fail("byte permutation mode other than index is not admitted");
        const auto selector = literal(s[1].value);
        if (form != 4 || !selector) fail("byte permutation selector must be an immediate");
        ir::Operand result = word(0);
        for (unsigned byte = 0; byte < 4; ++byte) {
            const unsigned nibble = (*selector >> (4 * byte)) & 15;
            const auto& from = nibble & 4 ? s[2].value : s[0].value;
            ir::Operand part = nibble & 8
                ? bits(ir::Opcode::bit_extract, {from, word(8 * (nibble & 3) + 7), word(1)}, signed_word)
                : bits(ir::Opcode::bit_extract, {from, word(8 * (nibble & 3)), word(8)}, unsigned_word);
            if (nibble & 8) part = bits(ir::Opcode::bit_and, {part, word(0xff)});
            if (byte) part = bits(ir::Opcode::shift_left, {part, word(8 * byte)});
            result = byte ? bits(ir::Opcode::bit_or, {result, part}) : part;
        }
        write_destination(d, result);
        leave(fields); return true;
    }
    case extremum: {
        enter(fields, false);
        const auto d = destination();
        auto s = alu_sources(fields, 2, false);
        const auto mode = fields.flag(73) ? signed_word : unsigned_word;
        auto minimum = predicate_source(fields, 87, false);
        ir::Operand result;
        if (const auto* fixed = std::get_if<ir::LiteralBits>(&minimum))
            result = bits(fixed->bits ? ir::Opcode::minimum : ir::Opcode::maximum, {s[0].value, s[1].value}, mode);
        else result = bits(ir::Opcode::select, {minimum, bits(ir::Opcode::minimum, {s[0].value, s[1].value}, mode),
                                                bits(ir::Opcode::maximum, {s[0].value, s[1].value}, mode)}, unsigned_word);
        write(d, result);
        leave(fields); return true;
    }
    case funnel: case uniform_funnel: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 3, uniform);
        const auto type = static_cast<unsigned>(fields.take(73, 2));
        const bool wrap = fields.flag(75), right = fields.flag(76), high_part = fields.flag(80);
        // Types: 0 S64, 1 U64, 2 S32, 3 U32.
        const bool wide = type < 2, is_signed = type == 0 || type == 2;
        const auto& low = s[0].value;
        const auto& high = s[2].value;
        // The high:low pair shifts as one 64-bit value by the count wrapped
        // to 0-31 or clamped to 0-32, and one word of the result is kept.
        // Counts below 32 are funnel shifts; a count of 32 moves whole words.
        auto by = [&](const ir::Operand& count) -> ir::Operand {
            if (right) return high_part ? bits(ir::Opcode::shift_right, {high, count}, is_signed ? signed_word : unsigned_word)
                                        : bits(ir::Opcode::funnel_shift_right, {low, high, count});
            return high_part ? bits(ir::Opcode::funnel_shift_left, {low, high, count}) : bits(ir::Opcode::shift_left, {low, count});
        };
        auto whole = [&]() -> ir::Operand {
            if (right) return high_part ? (is_signed ? bits(ir::Opcode::shift_right, {high, word(31)}, signed_word) : word(0)) : high;
            return high_part ? low : word(0);
        };
        ir::Operand result;
        if (wide) {
            // A 64-bit type shifts by the count wrapped to 0-63 or clamped to
            // 0-64: counts from 32 move a word and shift it by the rest, and
            // 64 leaves zero, or the sign of a signed right shift.
            const auto sign = bits(ir::Opcode::shift_right, {high, word(31)}, signed_word);
            const auto fill = [&]() -> ir::Operand { return right && is_signed ? sign : word(0); };
            auto beyond = [&](const ir::Operand& rest) -> ir::Operand {
                if (right) return high_part ? fill() : bits(ir::Opcode::shift_right, {high, rest}, is_signed ? signed_word : unsigned_word);
                return high_part ? bits(ir::Opcode::shift_left, {low, rest}) : word(0);
            };
            if (auto fixed = literal(s[1].value)) {
                const auto count = wrap ? (*fixed & 63) : std::min(*fixed, 64u);
                result = count < 32 ? by(word(count)) : count == 32 ? whole() : count < 64 ? beyond(word(count - 32)) : fill();
            } else {
                const auto count = wrap ? bits(ir::Opcode::bit_and, {s[1].value, word(63)})
                                        : bits(ir::Opcode::minimum, {s[1].value, word(64)}, unsigned_word);
                auto below = [&](std::uint32_t limit) {
                    return temporary(ir::Opcode::compare, {count, word(limit)}, unsigned_word, ir::ComparisonInfo{ir::Compare::less, false}, ir::ValueType::predicate);
                };
                const auto low_count = bits(ir::Opcode::bit_and, {count, word(31)});
                const auto upper = bits(ir::Opcode::select, {below(64), beyond(low_count), fill()}, unsigned_word);
                result = bits(ir::Opcode::select, {below(32), by(low_count), upper}, unsigned_word);
            }
        } else if (auto fixed = literal(s[1].value)) {
            const auto count = wrap ? (*fixed & 31) : std::min(*fixed, 32u);
            result = count == 32 ? whole() : by(word(count));
        } else if (wrap) result = by(bits(ir::Opcode::bit_and, {s[1].value, word(31)}));
        else {
            auto full = temporary(ir::Opcode::compare, {s[1].value, word(32)}, unsigned_word, ir::ComparisonInfo{ir::Compare::greater_equal, false}, ir::ValueType::predicate);
            result = bits(ir::Opcode::select, {full, whole(), by(s[1].value)}, unsigned_word);
        }
        write_destination(d, result);
        leave(fields); return true;
    }
    case sign_extend: case uniform_sign_extend: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 2, uniform);
        const auto mode = fields.flag(73) ? signed_word : unsigned_word;
        // A width of zero yields zero and one of 32 or more the whole word.
        write_destination(d, bits(ir::Opcode::bit_extract, {s[0].value, word(0), s[1].value}, mode));
        leave(fields); return true;
    }
    case multiply_add: case uniform_multiply_add: case multiply_wide: case uniform_multiply_wide: case multiply_high: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 3, uniform);
        const bool is_signed = fields.flag(73);
        if (fields.take(81, 3) != 7) fail("multiply-add carry output is not admitted");
        // The extended form adds a carry predicate. It is admitted only with
        // a zero product and unmodified sources, where it is the addend plus
        // the carry under every reading of the product's upper half.
        const bool extended = (base == multiply_add || base == uniform_multiply_add) && fields.flag(74);
        ir::Operand carry = ir::LiteralBits{0, ir::ValueType::predicate};
        if (extended) carry = predicate_source(fields, 87, uniform); else constant_false(87);
        if (base == multiply_add || base == uniform_multiply_add) {
            std::array<ir::Operand, 3> v{};
            for (unsigned i = 0; i < 3; ++i) v[i] = modified_integer(fields, s[i], !extended);
            const bool zero_product = literal(v[0]) == 0u || literal(v[1]) == 0u;
            if (extended && !zero_product) fail("extended multiply-add with a nonzero product is not admitted");
            ir::Operand result = zero_product ? v[2] : bits(ir::Opcode::multiply, {v[0], v[1]}, unsigned_word);
            if (!zero_product && literal(v[2]) != 0u) result = bits(ir::Opcode::add, {result, v[2]}, unsigned_word);
            if (extended) result = bits(ir::Opcode::add, {result, as_word(carry)}, unsigned_word);
            if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
            write_destination(d, result);
        } else if (base == multiply_high) {
            // Only the zero-addend form, where every reading of the addend
            // agrees, is admitted.
            if (literal(s[2].value) != 0u) fail("high multiply-add with a nonzero addend is not admitted");
            if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
            write(d, bits(ir::Opcode::multiply_high, {s[0].value, s[1].value}, is_signed ? signed_word : unsigned_word));
        } else {
            // 64-bit product plus a register-pair addend. The uniform datapath
            // admits register pairs only: URZ is 63 and a pair must not cross
            // it. The vector datapath keeps RZ as 255 and its constant form.
            // Negation and a carry in or out stay refused.
            if (uniform ? (form != 1 && form != 4) : (form != 1 && form != 4 && form != 5 && form != 6))
                fail("wide multiply-add addend form is not admitted");
            const auto addend_index = static_cast<unsigned>(fields.raw().bits(64, uniform ? 6 : 8));
            const unsigned zero = uniform ? 63u : 255u;
            if (d != zero && d + 1 >= zero)
                fail(uniform ? "wide multiply-add destination pair crosses URZ" : "wide multiply-add destination pair crosses RZ");
            if (addend_index != zero && addend_index + 1 >= zero)
                fail(uniform ? "wide multiply-add addend pair crosses URZ" : "wide multiply-add addend pair crosses RZ");
            if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
            // Low and high product words, then the 64-bit addend with one carry.
            auto low_product = bits(ir::Opcode::multiply, {s[0].value, s[1].value}, unsigned_word);
            auto high_product = bits(ir::Opcode::multiply_high, {s[0].value, s[1].value}, is_signed ? signed_word : unsigned_word);
            auto [low, carry] = add_with_carry(low_product, s[2].value, ir::LiteralBits{0, ir::ValueType::predicate});
            ir::Operand high_addend = word(0);
            if (addend_index != zero) high_addend = uniform ? this->uniform(addend_index + 1) : reg(addend_index + 1);
            auto high = bits(ir::Opcode::add, {high_product, high_addend}, unsigned_word);
            high = bits(ir::Opcode::add, {high, as_word(carry)}, unsigned_word);
            if (d != zero) { write_destination(d, low); write_destination(d + 1, high); }
        }
        leave(fields); return true;
    }
    case set_predicate: case uniform_set_predicate: {
        enter(fields, uniform);
        auto s = alu_sources(fields, 2, uniform);
        const bool extended = fields.flag(72), is_signed = fields.flag(73);
        const auto combine = static_cast<unsigned>(fields.take(74, 2)), relation = static_cast<unsigned>(fields.take(76, 3));
        if (combine == 3) fail("predicate combination mode is not admitted");
        if (fields.flag(79)) fail("integer comparison selector extension is not admitted");
        ir::Operand lower = ir::LiteralBits{1, ir::ValueType::predicate};
        if (extended) lower = predicate_source(fields, 68, uniform);
        else if (fields.take(68, 3) != 7 || fields.flag(71)) fail("lower comparison input requires the extended form");
        const auto first = static_cast<unsigned>(fields.take(81, 3)), second = static_cast<unsigned>(fields.take(84, 3));
        if (first == second && first != 7) fail("predicate destinations alias");
        auto accumulator = predicate_source(fields, 87, uniform);
        ir::Operand comparison;
        if (relation == 0 || relation == 7) comparison = ir::LiteralBits{relation == 7 ? 1u : 0u, ir::ValueType::predicate};
        else {
            constexpr ir::Compare relations[]{ir::Compare::never, ir::Compare::less, ir::Compare::equal, ir::Compare::less_equal,
                                              ir::Compare::greater, ir::Compare::not_equal, ir::Compare::greater_equal};
            comparison = temporary(ir::Opcode::compare, {s[0].value, s[1].value}, is_signed ? signed_word : unsigned_word,
                                   ir::ComparisonInfo{relations[relation], false}, ir::ValueType::predicate);
            if (extended) {
                // Equal upper words defer to the lower-word comparison.
                auto equal = temporary(ir::Opcode::compare, {s[0].value, s[1].value}, unsigned_word, ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
                comparison = logic(ir::Opcode::logical_or, {logic(ir::Opcode::logical_and, {equal, lower}),
                                                            logic(ir::Opcode::logical_and, {logic(ir::Opcode::logical_not, {equal}), comparison})});
            }
        }
        const auto operation_code = combine == 0 ? ir::Opcode::logical_and : combine == 1 ? ir::Opcode::logical_or : ir::Opcode::logical_xor;
        auto complement = logic(ir::Opcode::logical_not, {comparison});
        auto primary = logic(operation_code, {comparison, accumulator});
        auto secondary = logic(operation_code, {complement, accumulator});
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write_flag(first, primary); write_flag(second, secondary);
        leave(fields); return true;
    }
    case select_base: case uniform_select: {
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 2, uniform);
        auto condition = predicate_source(fields, 87, uniform);
        ir::Operand result;
        if (const auto* fixed = std::get_if<ir::LiteralBits>(&condition)) result = fixed->bits ? s[0].value : s[1].value;
        else result = bits(ir::Opcode::select, {condition, s[0].value, s[1].value}, unsigned_word);
        write_destination(d, result);
        leave(fields); return true;
    }
    case population: case uniform_population: case leading_one: case uniform_leading_one: case reverse: case uniform_reverse: {
        const bool uniform_query = base == uniform_population || base == uniform_leading_one || base == uniform_reverse;
        if (uniform_query != uniform) return false;
        enter(fields, uniform);
        const auto d = destination(); uniform_tail();
        auto s = alu_sources(fields, 1, uniform);
        ir::Operand value = s[1].value;
        const bool count_ones = base == population || base == uniform_population, leading = base == leading_one || base == uniform_leading_one;
        // The complement modifier belongs to register, constant and uniform
        // sources; an immediate fills bits 32-63 itself.
        if ((count_ones || leading) && s[1].negate && fields.flag(*s[1].negate)) value = bits(ir::Opcode::bit_not, {value});
        ir::Operand result;
        if (count_ones) result = bits(ir::Opcode::bit_count, {value});
        else if (leading) {
            const bool shift_amount = fields.flag(74), is_signed = fields.flag(73);
            if (fields.take(81, 3) != 7) fail("leading-one predicate output is not admitted");
            // A negative signed source searches its complement, so 0 and -1
            // both report no bit. The unsigned search covers both forms.
            if (is_signed) value = bits(ir::Opcode::bit_xor, {value, bits(ir::Opcode::shift_right, {value, word(31)}, signed_word)});
            result = bits(ir::Opcode::find_msb, {value}, unsigned_word);
            // The shift-amount form reports the leading zero count instead.
            if (shift_amount) result = bits(ir::Opcode::subtract, {word(31), result}, unsigned_word);
        } else result = bits(ir::Opcode::bit_reverse, {value});
        write_destination(d, result);
        leave(fields); return true;
    }
    default: return false;
    }
}
} // namespace armigo::shader_native::detail
