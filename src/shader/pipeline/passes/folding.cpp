// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "internal.h"
#include <bit>

namespace armigo::shader::pipeline::passes::detail {
bool neutral(const Operation& op) {
    const auto& n = op.numeric;
    return op.effects == Effects::none && n.domain != Domain::floating &&
        n.rounding == Rounding::not_applicable && !n.flush_inputs && !n.flush_output &&
        !n.saturate && !n.precise;
}
namespace {
std::uint64_t mask(unsigned width) { return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1; }
bool same(const Operand& a, const Operand& b) {
    if (const auto* v = std::get_if<ValueId>(&a)) { const auto* w = std::get_if<ValueId>(&b); return w && *v == *w; }
    if (const auto* v = std::get_if<LiteralBits>(&a)) { const auto* w = std::get_if<LiteralBits>(&b); return w && *v == *w; }
    return false;
}
bool comparison(Compare relation, bool equal, bool less) {
    switch (relation) {
    case Compare::never: return false;
    case Compare::less: return less;
    case Compare::equal: return equal;
    case Compare::less_equal: return less || equal;
    case Compare::greater: return !less && !equal;
    case Compare::not_equal: return !equal;
    case Compare::greater_equal: return !less;
    case Compare::always: return true;
    }
    return false;
}
}
std::optional<Operand> simplify(const Operation& op, ValueType output) {
    if (!neutral(op) || op.outputs.size() != 1) return {};
    if (op.opcode == Opcode::copy) return op.inputs[0];
    const auto width = scalar_bits(op.numeric.scalar);
    // Narrow integer values need explicit canonical extension; leave those to emission.
    if (width != 32 && width != 64 && width != 1) return {};
    const auto all = mask(width);
    auto literal = [&](std::size_t index) { return std::get_if<LiteralBits>(&op.inputs[index]); };
    auto constant = [&](std::size_t index, std::uint64_t bits) {
        const auto* value = literal(index); return value && value->bits == bits;
    };
    auto result = [&](std::uint64_t bits) -> std::optional<Operand> {
        return LiteralBits{bits & (output == ValueType::word64 ? UINT64_MAX : output == ValueType::word32 ? UINT32_MAX : 1), output};
    };
    if (op.opcode == Opcode::select) {
        if (const auto* condition = literal(0)) return op.inputs[condition->bits ? 1 : 2];
        if (same(op.inputs[1], op.inputs[2])) return op.inputs[1];
        return {};
    }
    // Identities apply only at full storage width, so they cannot lose narrowing.
    if (width >= 32 && op.inputs.size() == 2) {
        switch (op.opcode) {
        case Opcode::add: case Opcode::bit_or: case Opcode::bit_xor:
            if (constant(0, 0)) return op.inputs[1];
            if (constant(1, 0)) return op.inputs[0];
            break;
        case Opcode::multiply:
            if (constant(0, 1)) return op.inputs[1];
            if (constant(1, 1)) return op.inputs[0];
            break;
        case Opcode::bit_and:
            if (constant(0, all)) return op.inputs[1];
            if (constant(1, all)) return op.inputs[0];
            if (constant(0, 0) || constant(1, 0)) return result(0);
            break;
        case Opcode::subtract: case Opcode::shift_left: case Opcode::shift_right:
            if (constant(1, 0)) return op.inputs[0];
            break;
        case Opcode::divide:
            if (constant(1, 1)) return op.inputs[0];
            break;
        default: break;
        }
        if (op.opcode == Opcode::bit_or && (constant(0, all) || constant(1, all))) return result(all);
    }
    for (const auto& input : op.inputs) if (!std::holds_alternative<LiteralBits>(input)) return {};
    const auto a = literal(0)->bits & all;
    const auto b = op.inputs.size() > 1 ? literal(1)->bits & all : 0;
    const auto c = op.inputs.size() > 2 ? literal(2)->bits : 0;
    const bool signed_mode = op.numeric.domain == Domain::signed_integer;
    const auto sign = std::uint64_t{1} << (width - 1);
    auto less = [&](std::uint64_t x, std::uint64_t y) { return signed_mode ? (x ^ sign) < (y ^ sign) : x < y; };
    auto arithmetic_right = [&](std::uint64_t value, unsigned count) {
        if (count == 0) return value;
        auto shifted = value >> count;
        if (signed_mode && (value & sign)) shifted |= all ^ (all >> count);
        return shifted;
    };
    switch (op.opcode) {
    case Opcode::absolute: return result(signed_mode && (a & sign) ? (0 - a) & all : a);
    case Opcode::negate: return result((0 - a) & all);
    case Opcode::add: return result((a + b) & all);
    case Opcode::subtract: return result((a - b) & all);
    case Opcode::multiply: return result((a * b) & all);
    case Opcode::divide: case Opcode::remainder: {
        // Neither division by zero nor signed overflow has a proven fallback result.
        if (b == 0 || (signed_mode && a == sign && b == all)) return {};
        const bool na = signed_mode && (a & sign), nb = signed_mode && (b & sign);
        const auto ma = na ? (0 - a) & all : a, mb = nb ? (0 - b) & all : b;
        auto value = op.opcode == Opcode::divide ? ma / mb : ma % mb;
        if (op.opcode == Opcode::divide ? na != nb : na) value = (0 - value) & all;
        return result(value);
    }
    case Opcode::minimum: return result(less(a, b) ? a : b);
    case Opcode::maximum: return result(less(a, b) ? b : a);
    case Opcode::clamp:
        if (less(c & all, b)) return {};
        return result(less(a, b) ? b : less(c & all, a) ? c & all : a);
    case Opcode::compare: {
        const auto& info = std::get<ComparisonInfo>(op.info);
        if (info.relation > Compare::always) return {};
        return result(comparison(info.relation, a == b, less(a, b)));
    }
    case Opcode::bit_and: return result(a & b);
    case Opcode::bit_or: return result(a | b);
    case Opcode::bit_xor: return result(a ^ b);
    case Opcode::bit_not: return result((~a) & all);
    case Opcode::logical_and: return result(a && b);
    case Opcode::logical_or: return result(a || b);
    case Opcode::logical_xor: return result(a != b);
    case Opcode::logical_not: return result(!a);
    case Opcode::predicate_to_mask: return result(literal(0)->bits ? UINT32_MAX : 0);
    case Opcode::mask_to_predicate: return result(literal(0)->bits != 0);
    case Opcode::shift_left: return result((a << (literal(1)->bits & (width - 1))) & all);
    case Opcode::shift_right: return result(arithmetic_right(a, literal(1)->bits & (width - 1)));
    case Opcode::funnel_shift_left: {
        const auto count = static_cast<unsigned>(c & (width - 1));
        return result(count ? ((b << count) | (a >> (width - count))) & all : b);
    }
    case Opcode::funnel_shift_right: {
        const auto count = static_cast<unsigned>(c & (width - 1));
        return result(count ? ((a >> count) | (b << (width - count))) & all : a);
    }
    case Opcode::bit_extract: {
        const auto start = literal(1)->bits;
        if (c == 0 || start >= width || c > width - start) return {};
        auto value = (a >> start) & mask(static_cast<unsigned>(c));
        if (signed_mode && (value & (std::uint64_t{1} << (c - 1)))) value |= all ^ mask(static_cast<unsigned>(c));
        return result(value);
    }
    case Opcode::bit_insert: {
        const auto count = literal(3)->bits;
        if (count == 0 || c >= width || count > width - c) return {};
        const auto field = mask(static_cast<unsigned>(count)) << c;
        return result(((a & ~field) | ((b << c) & field)) & all);
    }
    case Opcode::bit_count: return result(std::popcount(a));
    case Opcode::bit_reverse: {
        std::uint64_t reversed = 0;
        for (unsigned i = 0; i < width; ++i) reversed |= ((a >> i) & 1) << (width - 1 - i);
        return result(reversed);
    }
    case Opcode::pack_word64: return result(literal(0)->bits | (literal(1)->bits << 32));
    default: return {};
    }
}
} // namespace armigo::shader::pipeline::passes::detail
