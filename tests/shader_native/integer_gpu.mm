// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed native integer, predicate, uniform, conversion,
// multifunction and flow forms. Each program loads its inputs from constant bank 3, computes R0-R3
// and exports them; the expected words come from scalar reference models
// written from the documented instruction contracts.

#include "metal_harness.h"
#include <bit>
#include <cmath>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <tuple>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
using Words = std::vector<std::uint32_t>;
using Pixel = std::array<std::uint32_t, 4>;
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
constexpr unsigned PT = 7, NOT_PT = 15;
sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
// Register operands of the three ALU slots.
sn::Instruction alu(unsigned opcode, unsigned d, unsigned a, unsigned b, unsigned c, std::initializer_list<Field> extra = {}) {
    auto i = op(opcode, {{16, 8, d}, {24, 8, a}, {32, 8, b}, {64, 8, c}});
    for (auto [first, count, value] : extra) field(i, first, count, value);
    return i;
}
sn::Instruction load(unsigned d, unsigned index) { return op(0xa02, {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}}); }
// R7 = 1 for predicate observation through SEL.
sn::Instruction one() { return move(7, 1); }
sn::Instruction observe(unsigned d, unsigned p, bool inverted = false) {
    return op(0x807, {{16, 8, d}, {24, 8, 7}, {32, 32, 0}, {87, 3, p}, {90, 1, inverted}});
}
sn::Instruction no_carry(sn::Instruction i) { field(i, 77, 4, NOT_PT); field(i, 81, 3, PT); field(i, 84, 3, PT); field(i, 87, 4, NOT_PT); return i; }

struct Case {
    std::string name;
    std::vector<sn::Instruction> body;
    std::vector<Words> inputs;
    std::function<Pixel(const Words&)> expected;
    std::vector<std::uint32_t> constants{};
};
// Inputs land in R8-R15; R0-R3 start at zero so partial bodies stay defined.
Fixture program(const Case& c) {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    for (unsigned i = 0; i < 8; ++i) code.push_back(load(8 + i, i));
    code.push_back(one());
    code.insert(code.end(), c.body.begin(), c.body.end());
    code.push_back(exit_instruction());
    Fixture f(code, sn::Stage::fragment, c.constants); put(f.code, 0x48, 0xf);
    return f;
}

// Reference models.
std::uint32_t lut3(unsigned table, std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    std::uint32_t r = 0;
    for (unsigned bit = 0; bit < 32; ++bit) {
        const unsigned index = ((a >> bit & 1) << 2) | ((b >> bit & 1) << 1) | (c >> bit & 1);
        r |= ((table >> index) & 1u) << bit;
    }
    return r;
}
// A 64-bit type (wide) wraps the count to 0-63 or clamps it to 0-64.
std::uint32_t funnel(std::uint32_t low, std::uint32_t count, std::uint32_t high, bool is_signed, bool wrap, bool right, bool high_part,
                     bool wide = false) {
    const std::uint32_t s = wrap ? count & (wide ? 63 : 31) : std::min(count, wide ? 64u : 32u);
    const std::uint64_t x = std::uint64_t(high) << 32 | low;
    std::uint64_t shifted;
    if (s == 64) shifted = right && is_signed && std::int64_t(x) < 0 ? ~0ull : 0;
    else if (right) shifted = is_signed ? std::uint64_t(std::int64_t(x) >> s) : x >> s;
    else shifted = x << s;
    return high_part ? std::uint32_t(shifted >> 32) : std::uint32_t(shifted);
}
std::uint32_t permute(std::uint32_t a, std::uint32_t selector, std::uint32_t c) {
    std::uint32_t r = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
        const unsigned n = (selector >> (4 * byte)) & 15;
        std::uint32_t value = ((n & 4 ? c : a) >> (8 * (n & 3))) & 0xff;
        if (n & 8) value = value & 0x80 ? 0xff : 0;
        r |= value << (8 * byte);
    }
    return r;
}
float as_float(std::uint32_t bits) { return std::bit_cast<float>(bits); }
std::uint32_t bits_of(float value) { return std::bit_cast<std::uint32_t>(value); }
std::uint32_t flush(std::uint32_t bits) { return (bits & 0x7f800000u) ? bits : bits & 0x80000000u; }
bool is_nan(std::uint32_t bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }
float integral(float x, unsigned direction) {
    switch (direction) {
    case 0: return std::nearbyint(x);
    case 1: return std::floor(x);
    case 2: return std::ceil(x);
    default: return std::trunc(x);
    }
}
// Round, then saturate; NaN converts to zero.
std::uint32_t to_integer(std::uint32_t bits, unsigned direction, bool ftz, bool is_signed, unsigned width) {
    if (ftz) bits = flush(bits);
    if (is_nan(bits)) return 0;
    const double r = integral(as_float(bits), direction);
    const double low = is_signed ? -std::ldexp(1.0, int(width) - 1) : 0.0;
    const double high = is_signed ? std::ldexp(1.0, int(width) - 1) - 1 : std::ldexp(1.0, int(width)) - 1;
    const double clamped = std::min(std::max(r, low), high);
    return is_signed ? std::uint32_t(std::int32_t(clamped)) : std::uint32_t(clamped);
}
// Unsigned 32-bit to binary32 rounded toward zero.
std::uint32_t unsigned_toward_zero(std::uint32_t value) {
    float f = float(value);
    if (double(f) > double(value)) f = std::nextafter(f, 0.0f);
    return bits_of(f);
}

std::vector<Case> cases() {
    std::vector<Case> c;
    const Words edge{0, 1, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x12345678u, 0xfedcba98u, 5};
    auto rotations = [&](std::size_t count) {
        std::vector<Words> v;
        for (std::size_t r = 0; r < count; ++r) { Words w(8); for (unsigned i = 0; i < 8; ++i) w[i] = edge[(i + r * 3) % edge.size()] ^ std::uint32_t(r * 0x01010101u); v.push_back(w); }
        return v;
    };
    c.push_back({"iadd3 sign modifiers",
        {no_carry(alu(0x210, 0, 8, 9, 10)), no_carry(alu(0x210, 1, 8, 9, 10, {{72, 1, 1}, {75, 1, 1}})),
         no_carry(alu(0x210, 2, 8, 9, 255, {{63, 1, 1}})), no_carry(alu(0x210, 3, 255, 255, 11, {{75, 1, 1}}))},
        rotations(6), [](const Words& w) { return Pixel{w[0] + w[1] + w[2], 0u - w[0] + w[1] - w[2], w[0] - w[1], 0u - w[3]}; }});
    c.push_back({"iadd3 carry chain",
        {[] { auto i = no_carry(alu(0x210, 0, 8, 9, 10, {{72, 1, 1}})); field(i, 81, 3, 0); field(i, 84, 3, 1); return i; }(),
         observe(1, 0), observe(2, 1),
         [] { auto i = alu(0x210, 3, 11, 255, 255, {{74, 1, 1}}); field(i, 87, 4, 0); field(i, 77, 4, 1); field(i, 81, 3, PT); field(i, 84, 3, PT); return i; }()},
        rotations(8), [](const Words& w) {
            const std::uint64_t sum = std::uint64_t(~w[0]) + 1 + w[1] + w[2];
            const bool p0 = sum >= (1ull << 32), p1 = sum >= (2ull << 32);
            return Pixel{std::uint32_t(sum), p0 ? 1u : 0u, p1 ? 1u : 0u, w[3] + p0 + p1};
        }});
    // LEA.HI.SX32: the high word is A's sign extension (Rc stays RZ).
    c.push_back({"lea high sign-extended",
        {[] { auto i = alu(0x211, 0, 8, 9, 255, {{73, 1, 1}, {75, 5, 4}, {80, 1, 1}}); field(i, 81, 3, PT); field(i, 87, 4, NOT_PT); return i; }(),
         [] { auto i = alu(0x211, 1, 8, 9, 255, {{73, 1, 1}, {75, 5, 0}, {80, 1, 1}}); field(i, 81, 3, PT); field(i, 87, 4, NOT_PT); return i; }(),
         [] { auto i = alu(0x211, 2, 8, 10, 255, {{73, 1, 1}, {75, 5, 31}, {80, 1, 1}}); field(i, 81, 3, PT); field(i, 87, 4, NOT_PT); return i; }()},
        rotations(8), [](const Words& w) {
            const auto wide = std::uint64_t(std::int64_t(std::int32_t(w[0])));
            auto high = [&](unsigned shift) { return std::uint32_t((wide << shift) >> 32); };
            return Pixel{high(4) + w[1], high(0) + w[1], high(31) + w[2], 0};
        }});
    c.push_back({"lea forms",
        {[] { auto i = alu(0x211, 0, 8, 9, 255, {{75, 5, 5}}); field(i, 81, 3, 0); field(i, 87, 4, NOT_PT); return i; }(),
         observe(1, 0),
         [] { auto i = alu(0x211, 2, 8, 9, 10, {{75, 5, 7}, {80, 1, 1}}); field(i, 81, 3, PT); field(i, 87, 4, NOT_PT); return i; }(),
         [] { auto i = alu(0x211, 3, 8, 9, 255, {{72, 1, 1}, {75, 5, 3}}); field(i, 81, 3, PT); field(i, 87, 4, NOT_PT); return i; }()},
        rotations(8), [](const Words& w) {
            const std::uint32_t shifted = w[0] << 5;
            const bool overflow = std::uint64_t(shifted) + w[1] > 0xffffffffull;
            const std::uint32_t high = std::uint32_t((std::uint64_t(w[2]) << 32 | w[0]) >> 25);
            return Pixel{shifted + w[1], overflow ? 1u : 0u, high + w[1], (0u - (w[0] << 3)) + w[1]};
        }});
    c.push_back({"lop3 tables and nonzero predicate",
        {alu(0x212, 0, 8, 9, 10, {{72, 8, 0x96}, {81, 3, PT}, {87, 4, NOT_PT}}), alu(0x212, 1, 8, 9, 10, {{72, 8, 0xe8}, {81, 3, PT}, {87, 4, NOT_PT}}),
         alu(0x212, 2, 8, 9, 10, {{72, 8, 0xca}, {81, 3, PT}, {87, 4, NOT_PT}}), alu(0x212, 255, 8, 9, 255, {{72, 8, 0xc0}, {81, 3, 0}, {87, 4, NOT_PT}}),
         observe(3, 0)},
        rotations(8), [](const Words& w) { return Pixel{lut3(0x96, w[0], w[1], w[2]), lut3(0xe8, w[0], w[1], w[2]), lut3(0xca, w[0], w[1], w[2]), (w[0] & w[1]) ? 1u : 0u}; }});
    for (bool wrap : {false, true}) {
        auto shf = [&](unsigned d, unsigned type, bool right, bool high) { return alu(0x219, d, 8, 10, 9, {{73, 2, type}, {75, 1, wrap}, {76, 1, right}, {80, 1, high}}); };
        std::vector<Words> inputs;
        for (std::uint32_t count : {0u, 1u, 17u, 31u, 32u, 33u, 64u, 0xffffffffu}) inputs.push_back({0x89abcdefu, 0xf0123456u, count, 0, 0, 0, 0, 0});
        c.push_back({wrap ? "funnel shifts wrapped" : "funnel shifts clamped",
            {shf(0, 3, false, false), shf(1, 3, false, true), shf(2, 3, true, false), shf(3, 2, true, true)}, inputs,
            [wrap](const Words& w) { return Pixel{funnel(w[0], w[2], w[1], false, wrap, false, false), funnel(w[0], w[2], w[1], false, wrap, false, true),
                                                  funnel(w[0], w[2], w[1], false, wrap, true, false), funnel(w[0], w[2], w[1], true, wrap, true, true)}; }});
        // 64-bit types (0 S64, 1 U64): counts past 31 move a word.
        std::vector<Words> wide_inputs;
        for (std::uint32_t count : {0u, 1u, 31u, 32u, 33u, 47u, 63u, 64u, 65u, 0xffffffffu})
            for (std::uint32_t high : {0xf0123456u, 0x70123456u}) wide_inputs.push_back({0x89abcdefu, high, count, 0, 0, 0, 0, 0});
        c.push_back({wrap ? "64-bit funnel shifts wrapped" : "64-bit funnel shifts clamped",
            {shf(0, 1, false, false), shf(1, 1, false, true), shf(2, 1, true, false), shf(3, 0, true, true)}, wide_inputs,
            [wrap](const Words& w) { return Pixel{funnel(w[0], w[2], w[1], false, wrap, false, false, true), funnel(w[0], w[2], w[1], false, wrap, false, true, true),
                                                  funnel(w[0], w[2], w[1], false, wrap, true, false, true), funnel(w[0], w[2], w[1], true, wrap, true, true, true)}; }});
        c.push_back({wrap ? "64-bit signed right low word, wrapped" : "64-bit signed right low word, clamped",
            {shf(0, 0, true, false), shf(1, 1, true, true), shf(2, 0, false, true), shf(3, 0, false, false)}, wide_inputs,
            [wrap](const Words& w) { return Pixel{funnel(w[0], w[2], w[1], true, wrap, true, false, true), funnel(w[0], w[2], w[1], false, wrap, true, true, true),
                                                  funnel(w[0], w[2], w[1], true, wrap, false, true, true), funnel(w[0], w[2], w[1], true, wrap, false, false, true)}; }});
        // Immediate counts take the literal path.
        for (std::uint32_t count : {0u, 20u, 32u, 40u, 63u}) {
            auto immediate = [&](unsigned d, unsigned type, bool right, bool high) {
                return op(0x819, {{16, 8, d}, {24, 8, 8}, {32, 32, count}, {64, 8, 9}, {73, 2, type}, {75, 1, wrap}, {76, 1, right}, {80, 1, high}});
            };
            c.push_back({"64-bit funnel shift by " + std::to_string(count) + (wrap ? " wrapped" : " clamped"),
                {immediate(0, 1, false, false), immediate(1, 1, false, true), immediate(2, 1, true, false), immediate(3, 0, true, true)},
                {Words{0x89abcdefu, 0xf0123456u, 0, 0, 0, 0, 0, 0}, Words{0x89abcdefu, 0x70123456u, 0, 0, 0, 0, 0, 0}},
                [wrap, count](const Words& w) { return Pixel{funnel(w[0], count, w[1], false, wrap, false, false, true), funnel(w[0], count, w[1], false, wrap, false, true, true),
                                                             funnel(w[0], count, w[1], false, wrap, true, false, true), funnel(w[0], count, w[1], true, wrap, true, true, true)}; }});
        }
    }
    c.push_back({"byte permutation",
        {op(0x816, {{16, 8, 0}, {24, 8, 8}, {32, 32, 0x3210}, {64, 8, 9}}), op(0x816, {{16, 8, 1}, {24, 8, 8}, {32, 32, 0x7654}, {64, 8, 9}}),
         op(0x816, {{16, 8, 2}, {24, 8, 8}, {32, 32, 0x8c4a}, {64, 8, 9}}), op(0x816, {{16, 8, 3}, {24, 8, 8}, {32, 32, 0x1076}, {64, 8, 9}})},
        rotations(6), [](const Words& w) { return Pixel{permute(w[0], 0x3210, w[1]), permute(w[0], 0x7654, w[1]), permute(w[0], 0x8c4a, w[1]), permute(w[0], 0x1076, w[1])}; }});
    c.push_back({"integer extrema",
        {alu(0x217, 0, 8, 9, 0, {{87, 4, PT}}), alu(0x217, 1, 8, 9, 0, {{87, 4, NOT_PT}}),
         alu(0x217, 2, 8, 9, 0, {{73, 1, 1}, {87, 4, PT}}), alu(0x217, 3, 8, 9, 0, {{73, 1, 1}, {87, 4, NOT_PT}})},
        rotations(8), [](const Words& w) {
            const auto a = std::int32_t(w[0]), b = std::int32_t(w[1]);
            return Pixel{std::min(w[0], w[1]), std::max(w[0], w[1]), std::uint32_t(std::min(a, b)), std::uint32_t(std::max(a, b))};
        }});
    auto isetp = [](unsigned a, unsigned b, unsigned relation, bool is_signed, unsigned first, std::initializer_list<Field> extra = {}) {
        auto i = op(0x20c, {{24, 8, a}, {32, 8, b}, {68, 4, PT}, {73, 1, is_signed}, {76, 3, relation}, {81, 3, first}, {84, 3, PT}, {87, 4, PT}});
        for (auto [f, n, v] : extra) field(i, f, n, v);
        return i;
    };
    c.push_back({"integer comparisons",
        {isetp(8, 9, 1, false, 0), observe(0, 0), isetp(8, 9, 6, true, 0, {{84, 3, 1}}), observe(1, 0), observe(2, 1),
         isetp(8, 10, 1, false, 0), isetp(9, 11, 1, true, 0, {{72, 1, 1}, {68, 4, 0}}), observe(3, 0)},
        rotations(8), [](const Words& w) {
            const bool ge = std::int32_t(w[0]) >= std::int32_t(w[1]);
            const std::int64_t left = std::int64_t(std::uint64_t(w[1]) << 32 | w[0]), right = std::int64_t(std::uint64_t(w[3]) << 32 | w[2]);
            return Pixel{w[0] < w[1] ? 1u : 0u, ge ? 1u : 0u, ge ? 0u : 1u, left < right ? 1u : 0u};
        }});
    c.push_back({"selection and predicate tables",
        {isetp(8, 9, 1, false, 0), isetp(10, 11, 4, false, 1),
         op(0x81c, {{87, 4, 0}, {77, 4, 1}, {68, 4, PT}, {64, 3, 0x96 & 7}, {72, 5, 0x96 >> 3}, {16, 8, 0xe8}, {81, 3, 2}, {84, 3, 3}}),
         alu(0x207, 0, 8, 9, 0, {{87, 3, 2}}), observe(1, 3), alu(0x207, 2, 10, 11, 0, {{87, 3, 0}, {90, 1, 1}}),
         alu(0x21a, 3, 8, 12, 0, {{73, 1, 1}})},
        [] { std::vector<Words> v; for (std::uint32_t width : {0u, 1u, 8u, 31u, 32u, 40u}) for (std::uint32_t x : {0x80u, 0xfffffff0u, 0x12345u}) v.push_back({x, 0x100u, width * 3, 7, width, 0, 0, 0}); return v; }(),
        [](const Words& w) {
            const bool p0 = w[0] < w[1], p1 = w[2] > w[3];
            const bool p2 = p0 == p1;          // 0x96 (three-way parity) with PT as the third source
            const bool p3 = p0 || p1;           // 0xe8 majority with PT
            std::uint32_t extended;
            if (w[4] >= 32) extended = w[0]; else if (!w[4]) extended = 0;
            else extended = std::uint32_t(std::int32_t(w[0] << (32 - w[4])) >> (32 - w[4]));
            return Pixel{p2 ? w[0] : w[1], p3 ? 1u : 0u, p0 ? w[3] : w[2], extended};
        }});
    c.push_back({"bit queries",
        {op(0x309, {{16, 8, 0}, {32, 8, 8}, {63, 1, 1}}), op(0x300, {{16, 8, 1}, {32, 8, 8}, {73, 1, 1}, {81, 3, PT}}),
         op(0x300, {{16, 8, 2}, {32, 8, 8}, {74, 1, 1}, {81, 3, PT}}), op(0x301, {{16, 8, 3}, {32, 8, 8}})},
        rotations(8), [](const Words& w) {
            const auto x = w[0];
            const unsigned leading_signed = std::int32_t(x) < 0 ? std::countl_zero(~x) : std::countl_zero(x);
            std::uint32_t reversed = 0; for (unsigned b = 0; b < 32; ++b) reversed |= ((x >> b) & 1u) << (31 - b);
            return Pixel{std::uint32_t(std::popcount(~x)), 31u - leading_signed, std::uint32_t(std::countl_zero(x)), reversed};
        }});
    auto imad = [](unsigned opcode, unsigned d, unsigned a, unsigned b, unsigned cr, std::initializer_list<Field> extra = {}) {
        auto i = alu(opcode, d, a, b, cr, extra); field(i, 81, 3, PT); field(i, 87, 4, NOT_PT); return i;
    };
    c.push_back({"multiply-add widths",
        {imad(0x224, 0, 8, 9, 10), imad(0x224, 1, 8, 9, 10, {{75, 1, 1}}), imad(0x225, 2, 8, 9, 12, {{73, 1, 1}})},
        rotations(8), [](const Words& w) {
            const std::int64_t product = std::int64_t(std::int32_t(w[0])) * std::int32_t(w[1]);
            const std::uint64_t wide = std::uint64_t(product) + (std::uint64_t(w[5]) << 32 | w[4]);
            return Pixel{w[0] * w[1] + w[2], w[0] * w[1] - w[2], std::uint32_t(wide), std::uint32_t(wide >> 32)};
        }});
    c.push_back({"high multiply and absolute",
        {imad(0x227, 0, 8, 9, 255), imad(0x227, 1, 8, 9, 255, {{73, 1, 1}}), op(0x213, {{16, 8, 2}, {32, 8, 8}})},
        rotations(8), [](const Words& w) {
            const std::uint32_t high = std::uint32_t((std::uint64_t(w[0]) * w[1]) >> 32);
            const std::uint32_t signed_high = std::uint32_t(std::uint64_t(std::int64_t(std::int32_t(w[0])) * std::int32_t(w[1])) >> 32);
            const std::uint32_t absolute = std::int32_t(w[0]) < 0 ? 0u - w[0] : w[0];
            return Pixel{high, signed_high, absolute, 0};
        }});
    // Uniform datapath: ULDC.64, UIADD3 with a uniform carry, UISETP, USHF, guarded UMOV.
    c.push_back({"uniform datapath",
        {op(0xab9, {{16, 6, 4}, {73, 3, 5}, {54, 5, 3}, {38, 16, 8}}),
         [] { auto i = op(0x290, {{16, 6, 6}, {24, 6, 4}, {32, 6, 5}, {64, 6, 63}, {91, 1, 1}}); return no_carry(i); }(),
         [] { auto i = op(0x290, {{16, 6, 9}, {24, 6, 4}, {32, 6, 5}, {64, 6, 63}, {91, 1, 1}}); i = no_carry(i); field(i, 81, 3, 0); return i; }(),
         op(0x28c, {{24, 6, 4}, {32, 6, 5}, {68, 4, PT}, {76, 3, 1}, {81, 3, 1}, {84, 3, PT}, {87, 4, PT}, {91, 1, 1}}),
         op(0x882, {{16, 6, 7}, {32, 32, 0}, {12, 4, PT}}), op(0x882, {{16, 6, 7}, {32, 32, 1}, {12, 4, 1}}),
         op(0x899, {{16, 6, 8}, {24, 6, 63}, {32, 32, 3}, {64, 6, 5}, {73, 2, 3}, {76, 1, 1}, {80, 1, 1}, {91, 1, 1}}),
         op(0x882, {{16, 6, 10}, {32, 32, 0}}), op(0x882, {{16, 6, 10}, {32, 32, 1}, {12, 4, 0}}),
         op(0xc02, {{16, 8, 0}, {32, 32, 6}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 1}, {32, 32, 7}, {64, 32, 0x08000f00}}),
         op(0xc02, {{16, 8, 2}, {32, 32, 8}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 3}, {32, 32, 10}, {64, 32, 0x08000f00}})},
        rotations(8), [](const Words& w) {
            return Pixel{w[2] + w[3], w[2] < w[3] ? 1u : 0u, w[3] >> 3, std::uint64_t(w[2]) + w[3] > 0xffffffffull ? 1u : 0u};
        }});
    // Narrow ULDC: U8 and S8 at byte 1, U16 and S16 at the high half of
    // input word 2 (bank 3 offset 8), direct and indexed by UR10 = 1 or 2.
    auto uldc = [](unsigned d, unsigned width, unsigned offset) { return op(0xab9, {{16, 6, d}, {73, 3, width}, {54, 5, 3}, {38, 16, offset}}); };
    auto indexed = [](unsigned d, unsigned width, unsigned offset) {
        return op(0xabb, {{16, 6, d}, {24, 6, 10}, {73, 3, width}, {54, 5, 3}, {38, 16, offset}, {91, 1, 1}});
    };
    auto narrow_value = [](std::uint32_t word, unsigned width, unsigned byte) -> std::uint32_t {
        const std::uint32_t raw = width < 2 ? (word >> (8 * byte)) & 0xff : (word >> (8 * byte)) & 0xffff;
        if (width == 1) return std::uint32_t(std::int32_t(std::int8_t(raw)));
        if (width == 3) return std::uint32_t(std::int32_t(std::int16_t(raw)));
        return raw;
    };
    c.push_back({"narrow uniform constant loads",
        {uldc(4, 0, 9), uldc(5, 1, 9), uldc(6, 2, 10), uldc(7, 3, 10),
         op(0xc02, {{16, 8, 0}, {32, 32, 4}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 1}, {32, 32, 5}, {64, 32, 0x08000f00}}),
         op(0xc02, {{16, 8, 2}, {32, 32, 6}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 3}, {32, 32, 7}, {64, 32, 0x08000f00}})},
        rotations(8), [narrow_value](const Words& w) {
            return Pixel{narrow_value(w[2], 0, 1), narrow_value(w[2], 1, 1), narrow_value(w[2], 2, 2), narrow_value(w[2], 3, 2)};
        }});
    c.push_back({"indexed narrow uniform constant loads",
        {op(0x882, {{16, 6, 10}, {32, 32, 1}}), indexed(4, 0, 8), indexed(5, 1, 8), op(0x882, {{16, 6, 10}, {32, 32, 2}}),
         indexed(6, 2, 8), indexed(7, 3, 8),
         op(0xc02, {{16, 8, 0}, {32, 32, 4}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 1}, {32, 32, 5}, {64, 32, 0x08000f00}}),
         op(0xc02, {{16, 8, 2}, {32, 32, 6}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 3}, {32, 32, 7}, {64, 32, 0x08000f00}})},
        rotations(8), [narrow_value](const Words& w) {
            return Pixel{narrow_value(w[2], 0, 1), narrow_value(w[2], 1, 1), narrow_value(w[2], 2, 2), narrow_value(w[2], 3, 2)};
        }});
    // UP2UR places uniform predicates at their mask bits: UP1 is UR4 < UR5,
    // UP0 and UP2-UP6 are never written and read false.
    auto up2ur = [](unsigned d, std::uint32_t mask) { return op(0x883, {{16, 6, d}, {24, 8, 63}, {32, 32, mask}, {91, 1, 1}}); };
    c.push_back({"uniform predicates to a uniform register",
        {op(0xab9, {{16, 6, 4}, {73, 3, 5}, {54, 5, 3}, {38, 16, 8}}),
         op(0x28c, {{24, 6, 4}, {32, 6, 5}, {68, 4, PT}, {76, 3, 1}, {81, 3, 1}, {84, 3, PT}, {87, 4, PT}, {91, 1, 1}}),
         up2ur(6, 0x2), up2ur(7, 0x7d), up2ur(8, 0x7f), up2ur(9, 0x80),
         op(0xc02, {{16, 8, 0}, {32, 32, 6}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 1}, {32, 32, 7}, {64, 32, 0x08000f00}}),
         op(0xc02, {{16, 8, 2}, {32, 32, 8}, {64, 32, 0x08000f00}}), op(0xc02, {{16, 8, 3}, {32, 32, 9}, {64, 32, 0x08000f00}})},
        rotations(8), [](const Words& w) {
            const std::uint32_t up1 = w[2] < w[3] ? 2u : 0u;
            return Pixel{up1, 0, up1, 0};
        }});
    // Conversions on special and boundary values.
    const Words specials{0x3fc00000u, 0xbfc00000u, 0x40200000u, 0x7fc00000u, 0x7f800000u, 0xff800000u, 0x4f32d05eu, 0xcf32d05eu,
                         0x47800066u, 0x80000001u, 0x00000001u, 0x80000000u, 0x3f000000u, 0xbf000000u, 0x4b7fffffu, 0x00800000u};
    auto each = [&](auto make) { std::vector<Words> v; for (auto x : specials) v.push_back(make(x)); return v; };
    auto f2i = [](unsigned d, bool is_signed, unsigned width, unsigned direction, bool ftz) {
        return op(0x305, {{16, 8, d}, {32, 8, 8}, {72, 1, is_signed}, {75, 2, width == 16 ? 1u : 2u}, {77, 1, 1}, {78, 2, direction}, {80, 1, ftz}, {84, 2, 2}});
    };
    c.push_back({"float to integer",
        {f2i(0, true, 32, 3, true), f2i(1, false, 32, 0, true), f2i(2, true, 32, 1, false), f2i(3, false, 16, 0, true)},
        each([](std::uint32_t x) { return Words{x, 0, 0, 0, 0, 0, 0, 0}; }), [](const Words& w) {
            return Pixel{to_integer(w[0], 3, true, true, 32), to_integer(w[0], 0, true, false, 32), to_integer(w[0], 1, false, true, 32), to_integer(w[0], 0, true, false, 16)};
        }});
    // F2IP.U8.F32: Rb's byte at bits 0-7 and Ra's at 8-15, each rounded
    // (mode 0xc4 toward zero, 0x04 to nearest even), saturated to a byte,
    // NaN to zero; RZ as Ra leaves zero above.
    auto f2ip = [](unsigned d, unsigned a, unsigned mode) { return op(0x243, {{16, 8, d}, {24, 8, a}, {32, 8, 8}, {64, 8, 255}, {72, 8, mode}}); };
    c.push_back({"packed float to byte", {f2ip(0, 255, 0xc4), f2ip(1, 255, 0x04), f2ip(2, 9, 0xc4), f2ip(3, 9, 0x04)},
        [&] {
            std::vector<Words> v;
            for (std::size_t i = 0; i < specials.size(); ++i) v.push_back({specials[i], specials[(i + 5) % specials.size()], 0, 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) {
            const auto trunc = [](std::uint32_t x) { return to_integer(x, 3, false, false, 8); };
            const auto nearest = [](std::uint32_t x) { return to_integer(x, 0, false, false, 8); };
            return Pixel{trunc(w[0]), nearest(w[0]), trunc(w[0]) | trunc(w[1]) << 8, nearest(w[0]) | nearest(w[1]) << 8};
        }});
    auto frnd = [](unsigned d, unsigned direction, bool ftz) { return op(0x307, {{16, 8, d}, {32, 8, 8}, {75, 2, 2}, {78, 2, direction}, {80, 1, ftz}, {84, 2, 2}}); };
    c.push_back({"integral rounding",
        {frnd(0, 1, false), frnd(1, 2, false), frnd(2, 0, true), frnd(3, 3, true)},
        // NaN payloads are not canonicalized (a documented departure), so no NaN input here.
        [&] { std::vector<Words> v; for (auto x : specials) if (!is_nan(x)) v.push_back({x, 0, 0, 0, 0, 0, 0, 0}); return v; }(), [](const Words& w) {
            auto r = [&](unsigned direction, bool ftz) { return bits_of(integral(as_float(ftz ? flush(w[0]) : w[0]), direction)); };
            return Pixel{r(1, false), r(2, false), r(0, true), r(3, true)};
        }});
    c.push_back({"integer to float",
        {op(0x306, {{16, 8, 0}, {32, 8, 8}, {60, 2, 1}, {75, 2, 2}, {84, 2, 0}}), op(0x306, {{16, 8, 1}, {32, 8, 8}, {60, 2, 1}, {74, 1, 1}, {75, 2, 2}, {84, 2, 1}}),
         op(0x245, {{16, 8, 2}, {32, 8, 8}, {74, 1, 1}, {75, 2, 2}, {84, 2, 2}}), op(0x245, {{16, 8, 3}, {32, 8, 8}, {75, 2, 2}, {78, 2, 3}, {84, 2, 2}})},
        [] { std::vector<Words> v; for (std::uint32_t x : {0u, 1u, 0x80000000u, 0xffffffffu, 0x01000001u, 0x7fffffffu, 0xfe12ab34u, 0x00ff8000u}) v.push_back({x, 0, 0, 0, 0, 0, 0, 0}); return v; }(),
        [](const Words& w) {
            const auto x = w[0];
            return Pixel{bits_of(float((x >> 8) & 0xff)), bits_of(float(std::int16_t(x >> 16))), bits_of(float(std::int32_t(x))), unsigned_toward_zero(x)};
        }});
    c.push_back({"float selection and sets",
        {alu(0x209, 0, 8, 9, 0, {{80, 1, 1}, {87, 4, PT}}), alu(0x209, 1, 8, 9, 0, {{80, 1, 1}, {87, 4, NOT_PT}}),
         alu(0x20a, 2, 8, 9, 0, {{76, 4, 1}, {80, 1, 1}, {87, 4, PT}}),
         op(0x80b, {{24, 8, 8}, {32, 32, 0x3f000000u}, {76, 4, 4}, {80, 1, 1}, {81, 3, 0}, {84, 3, PT}, {87, 4, PT}}),
         alu(0x208, 3, 8, 9, 0, {{62, 1, 1}, {63, 1, 1}, {87, 3, 0}})},
        // Opposite-sign zero ties are a documented departure, so no input pair flushes to one.
        [&] {
            std::vector<Words> v;
            for (std::uint32_t x : {0x3fc00000u, 0xbf800000u, 0x00000003u, 0x80800000u, 0x7fc00000u, 0x40400000u})
                for (std::uint32_t y : {0x3f000000u, 0x00000002u, 0xc0000000u}) v.push_back({x, y, 0, 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) {
            auto pick = [&](bool minimum) {
                if (is_nan(w[0])) return w[1];
                if (is_nan(w[1])) return w[0];
                return as_float(w[0]) < as_float(w[1]) == minimum ? w[0] : w[1];
            };
            const auto a = flush(w[0]), b = flush(w[1]);
            const bool less = !is_nan(a) && !is_nan(b) && as_float(a) < as_float(b);
            const bool greater = !is_nan(a) && as_float(a) > 0.5f;
            return Pixel{flush(pick(true)), flush(pick(false)), less ? 0x3f800000u : 0u, greater ? w[0] : (w[1] | 0x80000000u)};
        }});
    c.push_back({"half widening",
        {op(0x230, {{16, 8, 0}, {24, 8, 255}, {72, 1, 1}, {32, 8, 8}, {60, 2, 2}, {78, 1, 1}}),
         op(0x230, {{16, 8, 1}, {24, 8, 255}, {72, 1, 1}, {32, 8, 8}, {60, 2, 3}, {78, 1, 1}})},
        [] { std::vector<Words> v; for (std::uint32_t x : {0x3c00bc00u, 0x7bff0001u, 0x80000400u, 0x3555c000u}) v.push_back({x, 0, 0, 0, 0, 0, 0, 0}); return v; }(),
        [](const Words& w) {
            auto widen = [](std::uint32_t half) { return bits_of(float(std::bit_cast<_Float16>(std::uint16_t(half)))); };
            return Pixel{widen(w[0] & 0xffff), widen(w[0] >> 16), 0, 0};
        }});
    // Constant carries: PT adds one in IADD3.X and LEA.X; a 32-bit immediate
    // keeps its top bit in POPC and FLO (no complement modifier there).
    c.push_back({"constant carries and immediate queries",
        {[] { auto i = alu(0x210, 0, 8, 9, 10, {{74, 1, 1}}); field(i, 87, 4, PT); field(i, 77, 4, NOT_PT); field(i, 81, 3, PT); field(i, 84, 3, PT); return i; }(),
         [] { auto i = alu(0x211, 1, 8, 9, 255, {{74, 1, 1}}); field(i, 81, 3, PT); field(i, 87, 4, PT); return i; }(),
         op(0x909, {{16, 8, 2}, {32, 32, 0x80000000u}}),
         op(0x900, {{16, 8, 3}, {32, 32, 0x80000000u}, {81, 3, PT}})},
        rotations(4), [](const Words& w) { return Pixel{w[0] + w[1] + w[2] + 1, w[0] + w[1] + 1, 1, 31}; }});
    // Multifunction operations flush their input and result; Metal defines
    // the approximation, so the vectors use exactly representable results.
    auto mufu = [](unsigned d, unsigned function, unsigned source) { return op(0x308, {{16, 8, d}, {32, 8, source}, {74, 6, function}}); };
    c.push_back({"multifunction reciprocal family",
        {mufu(0, 4, 8), mufu(1, 5, 8), op(0xb08, {{16, 8, 2}, {54, 5, 3}, {38, 16, 0}, {74, 6, 8}}), mufu(3, 3, 8)},
        [] {
            std::vector<Words> v;
            for (std::uint32_t x : {0x40800000u, 0x3e800000u, 0x41800000u, 0x3f800000u, 0u, 0x80000000u, 0x7f800000u, 0x00000100u, 0x7e800000u})
                v.push_back({x, 0, 0, 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) {
            const float x = as_float(flush(w[0]));
            return Pixel{flush(bits_of(1.0f / x)), flush(bits_of(float(1.0 / std::sqrt(double(x))))), bits_of(std::sqrt(x)), bits_of(std::log2(x))};
        }});
    c.push_back({"multifunction exponent",
        {mufu(0, 2, 8), op(0xab9, {{16, 6, 4}, {73, 3, 4}, {54, 5, 3}, {38, 16, 0}}), op(0xd08, {{16, 8, 1}, {32, 6, 4}, {74, 6, 2}, {91, 1, 1}})},
        [] {
            std::vector<Words> v;
            for (float x : {3.0f, -2.0f, 0.0f, -INFINITY, INFINITY, 127.0f, 128.0f, -126.0f, -127.0f, -149.0f, 1.0f})
                v.push_back({bits_of(x), 0, 0, 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) { const auto r = flush(bits_of(std::exp2(as_float(w[0])))); return Pixel{r, r, 0, 0}; }});
    // Multiply scales: D8, D2, M2 and M8 after the rounded, flushed product.
    auto scaled = [](unsigned d, unsigned selector) { return op(0x220, {{16, 8, d}, {24, 8, 8}, {32, 8, 9}, {80, 1, 1}, {84, 3, selector}}); };
    c.push_back({"multiply scales",
        {scaled(0, 1), scaled(1, 3), scaled(2, 5), scaled(3, 7)},
        [] {
            std::vector<Words> v;
            for (float x : {3.0f, -1.5f, 0x1p-120f, 0x1p125f}) for (float y : {5.0f, -0.75f, 0x1p-4f}) v.push_back({bits_of(x), bits_of(y), 0, 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) {
            const auto product = flush(bits_of(as_float(w[0]) * as_float(w[1])));
            auto r = [&](float factor) { return flush(bits_of(as_float(product) * factor)); };
            return Pixel{r(0.125f), r(0.5f), r(2.0f), r(8.0f)};
        }});
    // Literal boundary results distinguish product scaling from input scaling
    // and preserve the first product's overflow/FTZ before the second step.
    c.push_back({"multiply scales range boundaries",
        {scaled(0, 1), scaled(1, 3), scaled(2, 5), scaled(3, 7)},
        {{0x7e800000u, 0x3e000000u, 0, 0, 0, 0, 0, 0},
         {0x01000000u, 0x41000000u, 0, 0, 0, 0, 0, 0},
         {0x7e800000u, 0x41000000u, 0, 0, 0, 0, 0, 0},
         {0x01000000u, 0x3e000000u, 0, 0, 0, 0, 0, 0}},
        [](const Words& w) {
            if (w[0] == 0x7e800000u) return w[1] == 0x3e000000u
                ? Pixel{0x7b800000u, 0x7c800000u, 0x7d800000u, 0x7e800000u}
                : Pixel{0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u};
            return w[1] == 0x41000000u ? Pixel{0x01000000u, 0x02000000u, 0x03000000u, 0x04000000u}
                : Pixel{0, 0, 0, 0};
        }});
    c.push_back({"multiply scale before saturation",
        {[&] { auto i = scaled(0, 1); field(i, 77, 1, 1); return i; }()},
        {{0x40000000u, 0x40000000u, 0, 0, 0, 0, 0, 0}},
        [](const Words&) { return Pixel{0x3f000000u, 0, 0, 0}; }});
    c.push_back({"scaled product final saturation",
        {[&] { auto i = scaled(0, 7); field(i, 77, 1, 1); return i; }()},
        {{0x3f000000u, 0x3f000000u, 0, 0, 0, 0, 0, 0}},
        [](const Words&) { return Pixel{0x3f800000u, 0, 0, 0}; }});
    // FMUL.RZ follows the Maxwell contract: the native nearest-even product,
    // at most one ulp above the guest's toward-zero result.
    c.push_back({"toward-zero multiply contract",
        {op(0x220, {{16, 8, 0}, {24, 8, 8}, {32, 8, 9}, {78, 2, 3}, {80, 1, 1}, {84, 3, 4}})},
        {{0x3fc00000u, 0x3f800001u, 0, 0, 0, 0, 0, 0}, {0x40400000u, 0x3eaaaaabu, 0, 0, 0, 0, 0, 0}, {0x00800000u, 0x3f000000u, 0, 0, 0, 0, 0, 0}},
        [](const Words& w) {
            const auto nearest = flush(bits_of(as_float(w[0]) * as_float(w[1])));
            const double exact = double(as_float(w[0])) * double(as_float(w[1]));
            float toward = float(exact); if (std::fabs(double(toward)) > std::fabs(exact)) toward = std::nextafter(toward, 0.0f);
            const auto zero = flush(bits_of(toward));
            if (nearest != zero && nearest != zero + 1) throw std::runtime_error("toward-zero contract exceeds one ulp");
            return Pixel{nearest, 0, 0, 0};
        }});
    // Convergence barriers around a data-dependent branch, a register-pair
    // zero, the guarded call idiom and fragment discard.
    auto relative = [](unsigned from, unsigned to) { return std::uint64_t(std::int64_t(to) * 16 - std::int64_t(from) * 16 - 16) & ((1ull << 50) - 1); };
    const unsigned base = 13; // instruction index of the first body instruction
    c.push_back({"flow forms",
        {op(0x945, {{16, 4, 0}, {32, 50, relative(base, base + 4)}, {87, 4, PT}}),
         isetp(8, 9, 1, false, 0),
         op(0x947, {{12, 4, 0}, {32, 50, relative(base + 2, base + 4)}, {87, 4, PT}}),
         move(0, 0x111),
         op(0x941, {{16, 4, 0}, {87, 4, PT}}),
         op(0x805, {{16, 8, 2}, {72, 8, 0xff}, {80, 1, 1}}),
         op(0x944, {{12, 4, 0}, {32, 50, 16}, {86, 1, 1}, {87, 4, PT}}),
         move(1, 0x222),
         op(0x95b, {{12, 4, 8}, {87, 4, PT}})},
        rotations(8), [](const Words& w) {
            const bool p0 = w[0] < w[1];
            if (!p0) return Pixel{MetalHarness::cleared, MetalHarness::cleared, MetalHarness::cleared, MetalHarness::cleared};
            return Pixel{0, 0, 0, 0};
        }});
    // R2P: P0-P6 from a byte of R8 under a mask; unmasked predicates keep
    // the values an ISETP gave them.
    for (auto [byte, mask] : {std::pair{1u, 0x7fu}, std::pair{2u, 0x5au}}) {
        std::vector<sn::Instruction> body{isetp(9, 9, 0, false, 0), isetp(9, 9, 0, false, 3), isetp(9, 9, 0, false, 6),
            op(0x804, {{24, 8, 8}, {32, 32, mask}, {76, 2, byte}})};
        for (unsigned p : {0u, 3u, 4u, 6u}) body.push_back(observe(p == 0 ? 0 : p == 3 ? 1 : p == 4 ? 2 : 3, p));
        c.push_back({"R2P byte " + std::to_string(byte), body, rotations(8), [byte, mask](const Words& w) {
            Pixel out{};
            const unsigned predicates[4] = {0, 3, 4, 6};
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned p = predicates[i];
                // isetp(9, 9, 0, ...) with relation 0 sets its predicate to false.
                const bool kept = false;
                out[i] = (mask & (1u << p)) ? (w[0] >> (8 * byte + p)) & 1 : kept;
            }
            return out;
        }});
    }
    // FADD.FTZ with a directed rounding and an RZ operand: exact, the mode
    // only signing a zero sum (toward minus keeps -0, toward plus makes +0).
    c.push_back({"directed FADD with RZ", {op(0x221, {{16, 8, 0}, {24, 8, 255}, {32, 8, 8}, {78, 2, 1}, {80, 1, 1}}),
                                           op(0x221, {{16, 8, 1}, {24, 8, 255}, {32, 8, 8}, {78, 2, 2}, {80, 1, 1}})},
        [] {
            std::vector<Words> v;
            for (std::uint32_t x : {0u, 0x80000000u, 0x3fc00000u, 0xc0200000u, 0x00000100u, 0x80000100u, 0x7f7fffffu, 0xff800000u, 0xffc00001u})
                v.push_back({x, 0, 0, 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) {
            const bool zero = !(w[0] & 0x7f800000u), nan = (w[0] & 0x7fffffffu) > 0x7f800000u;
            if (nan) return Pixel{0x7fffffffu, 0x7fffffffu, 0, 0};
            return Pixel{zero ? (w[0] & 0x80000000u) : w[0], zero ? 0u : w[0], 0, 0};
        }});
    // FADD.RM/.RP/.RZ of two registers: the native sum rounded to nearest.
    for (unsigned mode : {1u, 2u, 3u})
        c.push_back({"directed FADD as nearest " + std::to_string(mode), {op(0x221, {{16, 8, 0}, {24, 8, 8}, {32, 8, 9}, {78, 2, mode}, {80, 1, 1}})},
            [] { std::vector<Words> v; for (float a : {1.0f, -3.5f, 0x1.000002p0f}) v.push_back({bits_of(a), bits_of(1e-8f), 0, 0, 0, 0, 0, 0}); return v; }(),
            [](const Words& w) { return Pixel{bits_of(as_float(w[0]) + as_float(w[1])), 0, 0, 0}; }});
    // FFMA without FTZ: admitted with Metal's flushing, exact (fused) for
    // normal operands and results.
    c.push_back({"FFMA without FTZ", {op(0x223, {{16, 8, 0}, {24, 8, 8}, {32, 8, 9}, {64, 8, 10}})},
        [] {
            std::vector<Words> v;
            for (float a : {1.5f, -3.25f, 0x1.000002p0f}) for (float b : {2.0f, 0x1.fffffep0f}) v.push_back({bits_of(a), bits_of(b), bits_of(-1.0f / 3.0f), 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) { return Pixel{bits_of(std::fma(as_float(w[0]), as_float(w[1]), as_float(w[2]))), 0, 0, 0}; }});
    // ULDC indexed by a uniform register (0xabb): UR4 = c[3][UR5 + 4], with
    // UR5 from an input word, then moved to R0.
    c.push_back({"indexed uniform constant load", {op(0xab9, {{16, 6, 5}, {73, 3, 4}, {54, 5, 3}, {38, 16, 28}}),
                                                   op(0xabb, {{16, 6, 4}, {24, 6, 5}, {73, 3, 4}, {54, 5, 3}, {38, 16, 4}, {91, 1, 1}}),
                                                   op(0xc02, {{16, 8, 0}, {32, 6, 4}, {72, 4, 15}, {91, 1, 1}})},
        [] { std::vector<Words> v; for (std::uint32_t at : {0u, 4u, 8u, 12u}) v.push_back({0x11, 0x22, 0x33, 0x44, 0x55, 0, 0, at}); return v; }(),
        [](const Words& w) { return Pixel{w[w[7] / 4 + 1], 0, 0, 0}; }});
    // LDC narrow widths: U8, S8, U16, S16 of the input words through an index.
    c.push_back({"narrow constant loads", {op(0xb82, {{16, 8, 0}, {24, 8, 15}, {54, 5, 3}, {38, 16, 1}, {73, 3, 0}}),
                                           op(0xb82, {{16, 8, 1}, {24, 8, 15}, {54, 5, 3}, {38, 16, 3}, {73, 3, 1}}),
                                           op(0xb82, {{16, 8, 2}, {24, 8, 15}, {54, 5, 3}, {38, 16, 2}, {73, 3, 2}}),
                                           op(0xb82, {{16, 8, 3}, {24, 8, 15}, {54, 5, 3}, {38, 16, 2}, {73, 3, 3}})},
        [] {
            std::vector<Words> v;
            for (std::uint32_t word : {0x8081f2f3u, 0x7f01a5c3u}) for (std::uint32_t at : {0u, 4u})
                v.push_back({word, word ^ 0xffffffffu, 0, 0, 0, 0, 0, at});
            return v;
        }(),
        [](const Words& w) {
            const auto x = w[w[7] / 4];
            return Pixel{(x >> 8) & 0xff, std::uint32_t(std::int32_t(std::int8_t(x >> 24))), x >> 16,
                         std::uint32_t(std::int32_t(std::int16_t(x >> 16)))};
        }});
    // FFMA.RZ: the native fma rounded to nearest (the FMUL.RZ contract, at
    // most one ulp from the toward-zero result), here without FTZ.
    c.push_back({"FFMA.RZ as nearest", {op(0x223, {{16, 8, 0}, {24, 8, 8}, {32, 8, 9}, {64, 8, 10}, {78, 2, 3}})},
        [] {
            std::vector<Words> v;
            for (float a : {1.5f, 0x1.000002p0f, 3.0f}) for (float b : {0x1.fffffep0f, -1.0f / 3.0f}) v.push_back({bits_of(a), bits_of(b), bits_of(0.1f), 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) { return Pixel{bits_of(std::fma(as_float(w[0]), as_float(w[1]), as_float(w[2]))), 0, 0, 0}; }});
    // F2FP.PACK_AB: binary16 of R8 in the high half, of R9 in the low half,
    // rounded to nearest even, with binary16 overflow and subnormals.
    auto half_bits = [](std::uint32_t word) {
        return std::uint32_t(std::bit_cast<std::uint16_t>(static_cast<_Float16>(std::bit_cast<float>(word))));
    };
    c.push_back({"F2FP.PACK_AB", {op(0x23e, {{16, 8, 0}, {24, 8, 8}, {32, 8, 9}, {64, 8, 255}}),
                                  op(0x23e, {{16, 8, 1}, {24, 8, 255}, {32, 8, 10}, {64, 8, 255}})},
        [] {
            std::vector<Words> v;
            const float values[] = {1.0f, -2.5f, 65504.0f, 65520.0f, 70000.0f, 0x1p-24f, 0x1.8p-24f, 0x1p-26f, 1e-10f, -0.0f, 3.14159265f, -INFINITY};
            for (unsigned i = 0; i < std::size(values); ++i)
                v.push_back({bits_of(values[i]), bits_of(values[(i + 5) % std::size(values)]), bits_of(values[(i + 7) % std::size(values)]), 0, 0, 0, 0, 0});
            return v;
        }(),
        [half_bits](const Words& w) { return Pixel{(half_bits(w[0]) << 16) | half_bits(w[1]), half_bits(w[2]), 0, 0}; }});
    // CALL and RET: a subroutine adding 2 to R0, called twice; each call
    // writes its return offset to R10 first, as the compiler does.
    {
        auto copy = [](unsigned d, unsigned a, unsigned b) { return no_carry(alu(0x210, d, a, b, 255)); };
        c.push_back({"call and return",
            {move(10, (base + 2) * 16), op(0x944, {{32, 50, relative(base + 1, base + 7)}, {86, 1, 1}, {87, 4, PT}}),
             copy(1, 0, 255),
             move(10, (base + 5) * 16), op(0x944, {{32, 50, relative(base + 4, base + 7)}, {86, 1, 1}, {87, 4, PT}}),
             copy(2, 0, 255),
             op(0x947, {{32, 50, relative(base + 6, base + 11)}, {87, 4, PT}}),
             copy(0, 0, 7), copy(0, 0, 7), move(11, 0),
             op(0x950, {{24, 8, 10}, {32, 50, relative(base + 10, 0)}, {86, 1, 1}, {87, 4, PT}})},
            rotations(1), [](const Words&) { return Pixel{4, 2, 4, 0}; }});
    }
    // A predicate-only UPLOP3 between the table load and the BRXU, its
    // second lookup-table byte equal to the selector's number (6), leaves
    // the load the selector.
    const std::vector<std::uint32_t> past_table{(base + 4) * 16, (base + 6) * 16, (base + 8) * 16, 5};
    std::vector<Words> past_selections;
    for (std::uint32_t entry = 0; entry < 3; ++entry)
        past_selections.push_back({past_table[0], past_table[1], past_table[2], past_table[3], entry * 4, 0, 0, 0});
    c.push_back({"uniform jump table past UPLOP3",
        {op(0xab9, {{16, 6, 5}, {73, 3, 4}, {54, 5, 3}, {38, 16, 16}}),
         op(0xabb, {{16, 6, 6}, {24, 6, 5}, {73, 3, 4}, {54, 5, 1}, {91, 1, 1}}),
         op(0x89c, {{87, 4, PT}, {77, 4, PT}, {68, 4, PT}, {16, 8, 6}, {81, 3, PT}, {84, 3, PT}}),
         op(0x958, {{24, 6, 6}, {32, 50, relative(base + 3, 0)}, {87, 4, PT}, {91, 1, 1}}),
         move(0, 0x11), op(0x947, {{32, 50, relative(base + 5, base + 10)}, {87, 4, PT}}),
         move(0, 0x22), op(0x947, {{32, 50, relative(base + 7, base + 10)}, {87, 4, PT}}),
         move(0, 0x33), op(0x947, {{32, 50, relative(base + 9, base + 10)}, {87, 4, PT}})},
        past_selections, [](const Words& w) { return Pixel{0x11u * (w[4] / 4 + 1), 0, 0, 0}; }, past_table});
    // BRXU through the high half of a 64-bit ULDC: UR7 = c[3][24] (input
    // word 6), UR4:UR5 = c[1][UR7], and BRXU UR5 takes the table a word on.
    const std::vector<std::uint32_t> high_table{7, (base + 3) * 16, (base + 5) * 16, (base + 7) * 16, 5};
    std::vector<Words> high_selections;
    for (std::uint32_t entry = 0; entry < 3; ++entry)
        high_selections.push_back({high_table[0], high_table[1], high_table[2], high_table[3], high_table[4], 0, entry * 4, 0});
    c.push_back({"uniform jump table through a pair's high half",
        {op(0xab9, {{16, 6, 7}, {73, 3, 4}, {54, 5, 3}, {38, 16, 24}}),
         op(0xabb, {{16, 6, 4}, {24, 6, 7}, {73, 3, 5}, {54, 5, 1}, {91, 1, 1}}),
         op(0x958, {{24, 6, 5}, {32, 50, relative(base + 2, 0)}, {87, 4, PT}, {91, 1, 1}}),
         move(0, 0x11), op(0x947, {{32, 50, relative(base + 4, base + 9)}, {87, 4, PT}}),
         move(0, 0x22), op(0x947, {{32, 50, relative(base + 6, base + 9)}, {87, 4, PT}}),
         move(0, 0x33), op(0x947, {{32, 50, relative(base + 8, base + 9)}, {87, 4, PT}})},
        high_selections, [](const Words& w) { return Pixel{0x11u * (w[6] / 4 + 1), 0, 0, 0}; }, high_table});
    // A guarded CALL over a BRA after writing its return offset is a call:
    // the callee sets R1 and returns to the BRA, which ends the program.
    c.push_back({"guarded call over a branch",
        {isetp(8, 9, 1, false, 0), move(10, (base + 3) * 16),
         op(0x944, {{12, 4, 0}, {32, 50, 16}, {86, 1, 1}, {87, 4, PT}}),
         op(0x947, {{32, 50, relative(base + 3, base + 7)}, {87, 4, PT}}),
         move(1, 0x222),
         op(0x950, {{24, 8, 10}, {32, 50, relative(base + 5, 0)}, {86, 1, 1}, {87, 4, PT}}),
         move(1, 0x333)},
        {{1, 2, 0, 0, 0, 0, 0, 0}, {2, 1, 0, 0, 0, 0, 0, 0}, {5, 5, 0, 0, 0, 0, 0, 0}},
        [](const Words& w) { return Pixel{0, w[0] < w[1] ? 0x222u : 0u, 0, 0}; }});
    // A guarded CALL over the next instruction, the loop's back-edge BRA,
    // leaves the loop: R1 counts up to the input word, with no call linkage.
    c.push_back({"loop exit through a guarded call",
        {move(1, 0), move(2, 1),
         no_carry(alu(0x210, 1, 1, 2, 255)),
         isetp(1, 8, 2, false, 0),
         op(0x944, {{12, 4, 0}, {32, 50, 16}, {86, 1, 1}, {87, 4, PT}}),
         op(0x947, {{32, 50, relative(base + 5, base + 2)}, {87, 4, PT}}),
         no_carry(alu(0x210, 0, 1, 255, 255))},
        {{1, 0, 0, 0, 0, 0, 0, 0}, {2, 0, 0, 0, 0, 0, 0, 0}, {5, 0, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0, 0}},
        [](const Words& w) { return Pixel{w[0], w[0], 1, 0}; }});
    // BRX through a jump table: LDC R6 = c[1][R12], BRX R6 with base 0, one
    // case per table entry, each storing a marker and branching to EXIT. The
    // harness binds every constant bank to the input words, so the inputs
    // repeat the program's table; the entry 5 ends it.
    const std::vector<std::uint32_t> table{(base + 2) * 16, (base + 4) * 16, (base + 6) * 16, 5};
    std::vector<Words> selections;
    for (std::uint32_t entry = 0; entry < 3; ++entry) selections.push_back({table[0], table[1], table[2], table[3], entry * 4, 0, 0, 0});
    c.push_back({"jump table",
        {op(0xb82, {{16, 8, 6}, {24, 8, 12}, {54, 5, 1}, {73, 3, 4}}),
         op(0x949, {{24, 8, 6}, {32, 50, relative(base + 1, 0)}, {87, 4, PT}}),
         move(0, 0x11), op(0x947, {{32, 50, relative(base + 3, base + 8)}, {87, 4, PT}}),
         move(0, 0x22), op(0x947, {{32, 50, relative(base + 5, base + 8)}, {87, 4, PT}}),
         move(0, 0x33), op(0x947, {{32, 50, relative(base + 7, base + 8)}, {87, 4, PT}})},
        selections, [](const Words& w) { return Pixel{0x11u * (w[4] / 4 + 1), 0, 0, 0}; }, table});
    // BRXU through the same kind of table, its selector in a uniform
    // register: UR5 = c[3][16] (input word 4), UR6 = c[1][UR5], BRXU UR6.
    const std::vector<std::uint32_t> uniform_table{(base + 3) * 16, (base + 5) * 16, (base + 7) * 16, 5};
    std::vector<Words> uniform_selections;
    for (std::uint32_t entry = 0; entry < 3; ++entry)
        uniform_selections.push_back({uniform_table[0], uniform_table[1], uniform_table[2], uniform_table[3], entry * 4, 0, 0, 0});
    c.push_back({"uniform jump table",
        {op(0xab9, {{16, 6, 5}, {73, 3, 4}, {54, 5, 3}, {38, 16, 16}}),
         op(0xabb, {{16, 6, 6}, {24, 6, 5}, {73, 3, 4}, {54, 5, 1}, {91, 1, 1}}),
         op(0x958, {{24, 6, 6}, {32, 50, relative(base + 2, 0)}, {87, 4, PT}, {91, 1, 1}}),
         move(0, 0x11), op(0x947, {{32, 50, relative(base + 4, base + 9)}, {87, 4, PT}}),
         move(0, 0x22), op(0x947, {{32, 50, relative(base + 6, base + 9)}, {87, 4, PT}}),
         move(0, 0x33), op(0x947, {{32, 50, relative(base + 8, base + 9)}, {87, 4, PT}})},
        uniform_selections, [](const Words& w) { return Pixel{0x11u * (w[4] / 4 + 1), 0, 0, 0}; }, uniform_table});
    return c;
}
}
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    unsigned checked = 0;
    for (const auto& test : cases()) {
        auto compiled = gpu.compile(program(test));
        for (const auto& input : test.inputs) {
            const auto got = gpu.run(compiled, input), want = test.expected(input);
            if (got != want) {
                std::fprintf(stderr, "%s: input %08x %08x %08x %08x: got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n", test.name.c_str(),
                    input[0], input[1], input[2], input[3], got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                throw std::runtime_error("native integer GPU result");
            }
            ++checked;
        }
    }
    // MUFU.SIN and MUFU.COS take turns: sin(2 pi y), cos(2 pi y). Metal
    // defines the approximation, so finite results are checked against the
    // double-precision reference within 2^-21, about the unit's own error;
    // zeros and flushed subnormals are exact. Infinite and NaN angles follow
    // the shared sine and cosine, which the Metal functions define.
    {
        std::vector<sn::Instruction> code;
        for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
        code.push_back(load(8, 0));
        code.push_back(op(0x308, {{16, 8, 0}, {32, 8, 8}, {74, 6, 1}}));
        code.push_back(op(0x308, {{16, 8, 1}, {32, 8, 8}, {74, 6, 0}}));
        code.push_back(exit_instruction());
        Fixture f(code); put(f.code, 0x48, 0xf);
        auto compiled = gpu.compile(f);
        for (float y : {0.125f, 0.25f, -0.375f, 0.5f, 0.7f, 1.0f, -3.3f, 123.456f, -12345.678f, 1.5e6f, 0x1p-30f, 0x1p24f}) {
            const auto got = gpu.run(compiled, {bits_of(y)});
            const double turns = double(y) - std::nearbyint(double(y));
            const double reference[2] = {std::sin(2 * M_PI * turns), std::cos(2 * M_PI * turns)};
            for (unsigned i = 0; i < 2; ++i) if (std::fabs(double(as_float(got[i])) - reference[i]) > 0x1p-21) {
                std::fprintf(stderr, "multifunction %s(%a): got %a, reference %a\n", i ? "cos" : "sin", double(y), double(as_float(got[i])), reference[i]);
                throw std::runtime_error("native sine and cosine accuracy");
            }
            checked += 2;
        }
        for (std::uint32_t input : {0u, 0x80000000u, 0x00000100u, 0x80000100u}) {
            const auto got = gpu.run(compiled, {input});
            if ((got[0] & 0x7fffffffu) != 0 || got[1] != 0x3f800000u) {
                std::fprintf(stderr, "multifunction zero %08x: got %08x %08x\n", input, got[0], got[1]);
                throw std::runtime_error("native sine and cosine of zero turns");
            }
            checked += 2;
        }
    }
    // SHFL across a 4x4 target, each fragment shuffling its own pixel
    // position: butterflies 1 and 2 reach the horizontal and vertical quad
    // neighbours, and index 0 within four-lane segments reaches the quad's
    // first pixel, in all four operand forms.
    {
        auto ipa = [](unsigned d, unsigned attribute) { return op(0x326, {{16, 8, d}, {32, 8, 255}, {64, 8, attribute / 4}, {81, 3, PT}}); };
        auto shfl = [](unsigned opcode, unsigned d, unsigned a, unsigned mode, std::initializer_list<Field> operands) {
            auto i = op(opcode, {{16, 8, d}, {24, 8, a}, {58, 2, mode}, {81, 3, PT}});
            for (auto [first, count, value] : operands) field(i, first, count, value);
            return i;
        };
        Fixture f({move(0, 0), move(1, 0), move(2, 0), move(3, 0), ipa(4, 0x70), ipa(5, 0x74), move(6, 1), move(7, 0x1c03),
                   shfl(0x389, 0, 4, 3, {{32, 8, 6}, {64, 8, 7}}),
                   shfl(0xf89, 1, 5, 3, {{53, 5, 2}, {40, 13, 0x1c03}}),
                   shfl(0x589, 2, 4, 0, {{32, 8, 255}, {40, 13, 0x1c03}}),
                   shfl(0x989, 3, 5, 0, {{53, 5, 0}, {64, 8, 7}}),
                   exit_instruction()});
        put(f.code, 0x48, 0xf); f.code[0x17] = std::byte{0x30};
        auto compiled = gpu.compile(f);
        const auto pixels = gpu.grid(compiled, {}, 4, 4);
        auto centre = [](unsigned v) { return bits_of(float(v) + 0.5f); };
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const Pixel want{centre(x ^ 1), centre(y ^ 1), centre(x & ~1u), centre(y & ~1u)};
            if (pixels[y * 4 + x] != want) {
                const auto& got = pixels[y * 4 + x];
                std::fprintf(stderr, "shuffle at (%u, %u): got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n", x, y,
                    got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                throw std::runtime_error("native warp shuffle");
            }
            ++checked;
        }
    }
    // A SHFL guarded on odd lanes: the even lanes it turns off still offer
    // their registers, so an odd lane's butterfly reads its even neighbour's
    // x, while the even lanes keep their destination and predicate. Then
    // source lanes outside the SIMD-group in every mode: each lane keeps its
    // own value with a false predicate.
    {
        auto ipa = [](unsigned d, unsigned attribute) { return op(0x326, {{16, 8, d}, {32, 8, 255}, {64, 8, attribute / 4}, {81, 3, PT}}); };
        auto shfl = [](unsigned opcode, unsigned d, unsigned p, unsigned a, unsigned mode, std::initializer_list<Field> operands, unsigned guard = PT) {
            auto i = op(opcode, {{16, 8, d}, {24, 8, a}, {58, 2, mode}, {81, 3, p}});
            for (auto [first, count, value] : operands) field(i, first, count, value);
            field(i, 12, 4, guard);
            return i;
        };
        auto lane_id = op(0x919, {{16, 8, 8}, {72, 8, 0}});
        auto parity = op(0x804, {{24, 8, 8}, {32, 32, 1}, {76, 2, 0}});
        auto predicates = [](unsigned d) { return op(0x803, {{16, 8, d}, {24, 8, 255}, {32, 32, 0x7f}, {76, 2, 0}}); };
        Fixture guarded({move(0, 0x5555), move(1, 0), move(2, 0), move(3, 0), ipa(4, 0x70), lane_id, parity,
                         shfl(0xf89, 0, 1, 4, 3, {{53, 5, 1}, {40, 13, 0x1f}}, 0), predicates(1), exit_instruction()});
        put(guarded.code, 0x48, 0xf); guarded.code[0x17] = std::byte{0x30};
        auto compiled = gpu.compile(guarded);
        auto pixels = gpu.grid(compiled, {}, 4, 4);
        auto centre = [](unsigned v) { return bits_of(float(v) + 0.5f); };
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const bool odd = x & 1;
            const Pixel want{odd ? centre(x ^ 1) : 0x5555u, odd ? 3u : 0u, 0, 0};
            if (pixels[y * 4 + x] != want) {
                const auto& got = pixels[y * 4 + x];
                std::fprintf(stderr, "guarded shuffle at (%u, %u): got %08x %08x, expected %08x %08x\n", x, y, got[0], got[1], want[0], want[1]);
                throw std::runtime_error("native guarded warp shuffle");
            }
            ++checked;
        }
        // Register lanes and counts past the SIMD-group: index and butterfly
        // (bounds 0x1f) for every value, up (no bounds) and down (bounds 0x1f)
        // for the values that do not wrap back into it. Each keeps the lane's
        // own value in R0, R2, R3 and R5 with P0-P3 false; R1 gathers them.
        for (const std::uint32_t far : {32u, 33u, 0x80000000u, 0xffffffffu}) {
            const bool shifts = far != 0xffffffffu;
            std::vector<sn::Instruction> code{move(0, 0), move(1, 0), move(2, 0), move(3, 0), move(5, 0), ipa(4, 0x70), move(6, far), move(7, 0x1f),
                                              shfl(0x389, 0, 0, 4, 0, {{32, 8, 6}, {64, 8, 7}}),
                                              shfl(0x389, 5, 3, 4, 3, {{32, 8, 6}, {64, 8, 7}})};
            if (shifts) {
                code.push_back(shfl(0x389, 2, 1, 4, 1, {{32, 8, 6}, {64, 8, 255}}));
                code.push_back(shfl(0x389, 3, 2, 4, 2, {{32, 8, 6}, {64, 8, 7}}));
            }
            code.push_back(op(0x212, {{16, 8, 0}, {24, 8, 0}, {32, 8, 5}, {64, 8, 255}, {72, 8, 0x96}, {81, 3, PT}, {87, 4, NOT_PT}}));
            code.push_back(predicates(1));
            code.push_back(exit_instruction());
            Fixture outside(code);
            put(outside.code, 0x48, 0xf); outside.code[0x17] = std::byte{0x30};
            auto out_compiled = gpu.compile(outside);
            pixels = gpu.grid(out_compiled, {}, 4, 4);
            for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
                // R0 holds the index result XOR the butterfly result: zero
                // when both kept the lane's own value.
                const Pixel want{0, 0, shifts ? centre(x) : 0, shifts ? centre(x) : 0};
                const auto& got = pixels[y * 4 + x];
                if (got != want) {
                    std::fprintf(stderr, "out-of-range shuffle %08x at (%u, %u): got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n", far, x, y,
                        got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                    throw std::runtime_error("native out-of-range warp shuffle");
                }
                ++checked;
            }
        }
    }
    // FSWZADD on the same grid: quad lane L = (x & 1) + 2 (y & 1) takes the
    // code at bits 2L. Subop 0xe4 gives lanes 0-3 codes 0-3 (a + b, b - a,
    // a - b, b) with FTZ; 0xeb (b, a - b, a - b, b) without FTZ, as the boot
    // programs use it.
    {
        auto ipa = [](unsigned d, unsigned attribute) { return op(0x326, {{16, 8, d}, {32, 8, 255}, {64, 8, attribute / 4}, {81, 3, PT}}); };
        auto fswzadd = [](unsigned d, unsigned subop, bool ftz) { return op(0x822, {{16, 8, d}, {24, 8, 4}, {32, 8, subop}, {64, 8, 5}, {80, 1, ftz}}); };
        Fixture f({move(0, 0), move(1, 0), move(2, 0), move(3, 0), ipa(4, 0x70), ipa(5, 0x74),
                   fswzadd(0, 0xe4, true), fswzadd(1, 0xeb, false), exit_instruction()});
        put(f.code, 0x48, 0xf); f.code[0x17] = std::byte{0x30};
        const auto pixels = gpu.grid(gpu.compile(f), {}, 4, 4);
        auto combine = [](unsigned code, float a, float b) {
            return code == 0 ? a + b : code == 1 ? b - a : code == 2 ? a - b : b;
        };
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const unsigned lane = (x & 1) + 2 * (y & 1);
            const float a = float(x) + 0.5f, b = float(y) + 0.5f;
            const Pixel want{bits_of(combine((0xe4u >> (2 * lane)) & 3, a, b)), bits_of(combine((0xebu >> (2 * lane)) & 3, a, b)), 0, 0};
            if (pixels[y * 4 + x] != want) {
                const auto& got = pixels[y * 4 + x];
                std::fprintf(stderr, "swizzled add at (%u, %u): got %08x %08x, expected %08x %08x\n", x, y, got[0], got[1], want[0], want[1]);
                throw std::runtime_error("native swizzled add");
            }
            ++checked;
        }
    }
    std::printf("Native integer, conversion and flow Metal execution passed (%u vectors)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
