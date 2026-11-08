// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed packed binary16 addition, multiplication, fused multiply-add
// and extrema.
// The register fused multiply-add uses the same lanes, with the product's
// right factor taken from a register. HSETP2 compares those lanes and
// combines each result with one input predicate; its flush, after absolute
// and negate, replaces a magnitude below the smallest normal binary16 with +0.
// Expectations come from the model below, not from the shader. Each half is
// widened exactly, absolute is applied before negate, and the lane is added
// or fused in binary32. Binary32 subnormals flush at that operation. The
// pair then packs round to nearest, ties to even. FTZ also flushes a
// binary16 subnormal at the widen and at the pack. A scalar result keeps
// the low lane in binary32. The fused clamp, when its predicate holds, runs
// after the pack: a negative lane becomes +0 and either NaN becomes the
// all-ones binary16 NaN.

#include "metal_harness.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace sn = armigo::shader_native;
using namespace sn::test;
namespace {
using Words = std::vector<std::uint32_t>;
using Pixel = std::array<std::uint32_t, 4>;
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
constexpr unsigned PT = 7, NOT_PT = 15;
constexpr std::uint32_t H1 = 0x3c00, HN1 = 0xbc00, H2 = 0x4000, HN2 = 0xc000;
constexpr std::uint32_t H_HALF = 0x3800, H_INF = 0x7c00, HN_INF = 0xfc00, H_NAN = 0x7e00, H_MAX = 0x7bff;
constexpr std::uint32_t H_SUB = 0x0001, H_NORM = 0x0400, H_M10 = 0x1400, H_M11 = 0x1000, H_1P = 0x3c01;

std::uint32_t pair(std::uint32_t low, std::uint32_t high) { return low | (high << 16); }
bool half_nan(std::uint32_t h) { return (h & 0x7fffu) > 0x7c00u; }
bool word_nan(std::uint32_t b) { return (b & 0x7fffffffu) > 0x7f800000u; }
std::uint32_t flush32(std::uint32_t b) { return (b & 0x7f800000u) ? b : (b & 0x80000000u); }
std::uint32_t flush16(std::uint32_t h) { h &= 0xffffu; return (h & 0x7c00u) ? h : (h & 0x8000u); }

// Exact widen. A binary16 subnormal normalizes; it is never a binary32 subnormal.
std::uint32_t f16_to_f32(std::uint32_t h) {
    h &= 0xffffu;
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const unsigned exp = (h >> 10) & 31u, frac = h & 0x3ffu;
    if (exp == 0) {
        if (!frac) return sign;
        unsigned magnitude = frac, shift = 0;
        while ((magnitude & 0x400u) == 0) { magnitude <<= 1; ++shift; }
        const int unbiased = -14 - int(shift);
        return sign | (std::uint32_t(unbiased + 127) << 23) | ((magnitude & 0x3ffu) << 13);
    }
    if (exp == 31) return sign | 0x7f800000u | (frac ? (frac << 13) : 0u);
    return sign | (std::uint32_t(exp - 15 + 127) << 23) | (frac << 13);
}

// Round to nearest, ties to even. A binary32 subnormal is below the smallest
// binary16 subnormal, so it packs to signed zero. A NaN payload is not kept.
std::uint32_t f32_to_f16(std::uint32_t b) {
    const std::uint32_t sign = (b >> 16) & 0x8000u;
    const unsigned exp = (b >> 23) & 0xffu;
    const std::uint32_t mant = b & 0x7fffffu;
    if (exp == 255) return sign | (mant ? 0x7e00u : 0x7c00u);
    if (exp == 0) return sign;
    int half_exp = int(exp) - 112;
    if (half_exp >= 31) return sign | 0x7c00u;
    if (half_exp <= 0) {
        const std::uint32_t significand = mant | 0x800000u;
        const int shift = 14 - half_exp;
        if (shift >= 25) return sign;
        std::uint32_t half_mant = significand >> shift;
        const std::uint32_t round = (significand >> (shift - 1)) & 1u;
        const std::uint32_t sticky = significand & ((1u << (shift - 1)) - 1u);
        if (round && (sticky || (half_mant & 1u))) ++half_mant;
        if (half_mant == 0x400u) return sign | 0x0400u;
        return sign | half_mant;
    }
    std::uint32_t half_mant = mant >> 13;
    const std::uint32_t round = (mant >> 12) & 1u;
    if (round && ((mant & 0xfffu) || (half_mant & 1u))) {
        ++half_mant;
        if (half_mant == 0x400u) {
            half_mant = 0;
            if (++half_exp >= 31) return sign | 0x7c00u;
        }
    }
    return sign | (std::uint32_t(half_exp) << 10) | half_mant;
}

std::uint32_t add32(std::uint32_t a, std::uint32_t b) {
    a = flush32(a); b = flush32(b);
    return flush32(std::bit_cast<std::uint32_t>(std::bit_cast<float>(a) + std::bit_cast<float>(b)));
}
std::uint32_t mul32(std::uint32_t a, std::uint32_t b) {
    a = flush32(a); b = flush32(b);
    return flush32(std::bit_cast<std::uint32_t>(std::bit_cast<float>(a) * std::bit_cast<float>(b)));
}
std::uint32_t fma32(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    a = flush32(a); b = flush32(b); c = flush32(c);
    return flush32(std::bit_cast<std::uint32_t>(std::fma(std::bit_cast<float>(a), std::bit_cast<float>(b), std::bit_cast<float>(c))));
}
std::uint32_t saturate32(std::uint32_t b) {
    if (word_nan(b)) return 0;
    const float value = std::bit_cast<float>(b);
    if (!(value > 0.f)) return 0;
    return value > 1.f ? 0x3f800000u : b;
}
std::uint32_t pack2(std::uint32_t low, std::uint32_t high, bool ftz) {
    auto one = [&](std::uint32_t b) { const auto h = f32_to_f16(b); return ftz ? flush16(h) : h; };
    return one(low) | (one(high) << 16);
}
std::uint32_t relu(std::uint32_t packed) {
    auto lane = [](std::uint32_t h) {
        h &= 0xffffu;
        if ((h & 0x7fffu) > 0x7c00u) return 0x7fffu;
        return (h & 0x8000u) ? 0u : h;
    };
    return lane(packed) | (lane(packed >> 16) << 16);
}

struct Mods {
    unsigned swizzle_a = 0, swizzle_b = 0, swizzle_c = 0;
    bool abs_a = false, neg_a = false, abs_b = false, neg_b = false, abs_c = false, neg_c = false;
    bool sat = false, ftz = false, scalar = false, exchange = false;
};
std::array<std::uint32_t, 2> source_lanes(std::uint32_t bits, unsigned swizzle, bool absolute, bool negate, bool ftz) {
    auto modify = [&](std::uint32_t b) {
        if (absolute) b &= 0x7fffffffu;
        if (negate) b ^= 0x80000000u;
        return b;
    };
    if (swizzle == 1) {
        const auto lane = modify(bits);
        return {lane, lane};
    }
    auto widen = [&](unsigned shift) {
        auto h = (bits >> shift) & 0xffffu;
        if (ftz) h = flush16(h);
        return modify(f16_to_f32(h));
    };
    if (swizzle >= 2) {
        const auto lane = widen(swizzle == 3 ? 16u : 0u);
        return {lane, lane};
    }
    return {widen(0), widen(16)};
}
std::uint32_t hadd(std::uint32_t a, std::uint32_t b, Mods m = {}) {
    const auto left = source_lanes(a, m.swizzle_a, m.abs_a, m.neg_a, m.ftz);
    const auto right = source_lanes(b, m.swizzle_b, m.abs_b, m.neg_b, m.ftz);
    auto lane = [&](unsigned i) {
        auto value = add32(left[i], right[i]);
        if (m.sat) value = saturate32(value);
        return value;
    };
    return m.scalar ? lane(0) : pack2(lane(0), lane(1), m.ftz);
}
std::uint32_t hmul(std::uint32_t a, std::uint32_t b, Mods m = {}) {
    const auto left = source_lanes(a, m.swizzle_a, m.abs_a, m.neg_a, m.ftz);
    const auto right = source_lanes(b, m.swizzle_b, m.abs_b, m.neg_b, m.ftz);
    auto lane = [&](unsigned i) {
        auto value = mul32(left[i], right[i]);
        if (m.sat) value = saturate32(value);
        return value;
    };
    return m.scalar ? lane(0) : pack2(lane(0), lane(1), m.ftz);
}
std::uint32_t hfma(std::uint32_t a, std::uint32_t imm, std::uint32_t c, Mods m = {}, bool clamp = false) {
    const auto left = source_lanes(a, m.swizzle_a, m.abs_a, m.neg_a, m.ftz);
    auto right = source_lanes(imm, 0, false, false, m.ftz);
    if (m.exchange) right = {right[1] ^ 0x80000000u, right[0]};
    const auto addend = source_lanes(c, m.swizzle_c, m.abs_c, m.neg_c, m.ftz);
    auto lane = [&](unsigned i) {
        auto value = fma32(left[i], right[i], addend[i]);
        if (m.sat) value = saturate32(value);
        return value;
    };
    if (m.scalar) return lane(0);
    const auto packed = pack2(lane(0), lane(1), m.ftz);
    return clamp ? relu(packed) : packed;
}
std::uint32_t hfma_reg(std::uint32_t a, std::uint32_t b, std::uint32_t c, Mods m = {}) {
    const auto left = source_lanes(a, m.swizzle_a, m.abs_a, m.neg_a, m.ftz);
    auto right = source_lanes(b, m.swizzle_b, m.abs_b, m.neg_b, m.ftz);
    if (m.exchange) right = {right[1] ^ 0x80000000u, right[0]};
    const auto addend = source_lanes(c, m.swizzle_c, m.abs_c, m.neg_c, m.ftz);
    auto lane = [&](unsigned i) {
        auto value = fma32(left[i], right[i], addend[i]);
        if (m.sat) value = saturate32(value);
        return value;
    };
    if (m.scalar) return lane(0);
    return pack2(lane(0), lane(1), m.ftz);
}
// After absolute and negate, a magnitude below the smallest normal binary16
// becomes +0. A NaN is left unchanged. The two predicate results are the low bits.
std::uint32_t flush_half_threshold(std::uint32_t bits) {
    if (word_nan(bits)) return bits;
    return (bits & 0x7fffffffu) < 0x38800000u ? 0u : bits;
}
bool half_relation(std::uint32_t a, std::uint32_t b, unsigned function) {
    if (function == 0) return false;
    if (function == 15) return true;
    const bool left_nan = word_nan(a), right_nan = word_nan(b);
    if (function == 7) return !left_nan && !right_nan;
    if (function == 8) return left_nan || right_nan;
    if (left_nan || right_nan) return function >= 9;
    const float left = std::bit_cast<float>(a), right = std::bit_cast<float>(b);
    switch (function & 7) {
    case 1: return left < right;
    case 2: return left == right;
    case 3: return left <= right;
    case 4: return left > right;
    case 5: return left != right;
    case 6: return left >= right;
    default: return false;
    }
}
std::uint32_t hsetp(std::uint32_t a, std::uint32_t imm, Mods m, unsigned function, unsigned combine, bool conjunct, bool predicate) {
    auto left = source_lanes(a, m.swizzle_a, m.abs_a, m.neg_a, false);
    // An immediate second source has no selector or sign. A register does.
    auto right = source_lanes(imm, m.swizzle_b, m.abs_b, m.neg_b, false);
    if (m.ftz) { left = {flush_half_threshold(left[0]), flush_half_threshold(left[1])};
                 right = {flush_half_threshold(right[0]), flush_half_threshold(right[1])}; }
    auto mix = [&](bool comparison) {
        if (combine == 0) return comparison && predicate;
        if (combine == 1) return comparison || predicate;
        return comparison != predicate;
    };
    const bool low = mix(half_relation(left[0], right[0], function));
    const bool high = mix(half_relation(left[1], right[1], function));
    if (conjunct) { const bool both = low && high; return both | ((!both) << 1); }
    return low | (high << 1);
}
std::uint32_t hminmax(std::uint32_t a, std::uint32_t b, bool minimum, Mods m = {}) {
    const auto left = source_lanes(a, m.swizzle_a, m.abs_a, m.neg_a, false);
    const auto right = source_lanes(b, m.swizzle_b, m.abs_b, m.neg_b, false);
    auto pick = [&](unsigned i) {
        const auto x = left[i], y = right[i];
        if (word_nan(x)) return y;
        if (word_nan(y)) return x;
        const float fx = std::bit_cast<float>(x), fy = std::bit_cast<float>(y);
        if (fx < fy) return minimum ? x : y;
        if (fy < fx) return minimum ? y : x;
        return x;
    };
    return pack2(pick(0), pick(1), false);
}

sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
sn::Instruction load(unsigned d, unsigned index) { return op(0xa02, {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}}); }
void put_mod(sn::Instruction& i, unsigned first, unsigned count, std::uint64_t value) { if (value) field(i, first, count, value); }
void put_signs(sn::Instruction& i, Mods m, bool third) {
    put_mod(i, 72, 1, m.neg_a); put_mod(i, 73, 1, m.abs_a); put_mod(i, 74, 2, m.swizzle_a);
    put_mod(i, 77, 1, m.sat); put_mod(i, 78, 1, m.scalar); put_mod(i, 80, 1, m.ftz);
    if (!third) { put_mod(i, 60, 2, m.swizzle_b); put_mod(i, 62, 1, m.abs_b); put_mod(i, 63, 1, m.neg_b); }
    else { put_mod(i, 81, 2, m.swizzle_c); put_mod(i, 83, 1, m.abs_c); put_mod(i, 84, 1, m.neg_c); put_mod(i, 86, 1, m.exchange); }
}
sn::Instruction hadd_i(unsigned d, unsigned a, unsigned b, Mods m = {}) {
    auto i = op(0x230, {{16, 8, d}, {24, 8, a}, {32, 8, b}}); put_signs(i, m, false); return i;
}
sn::Instruction hfma_i(unsigned d, unsigned a, std::uint32_t imm, unsigned c, Mods m = {}, bool clamp = false, unsigned pred = PT) {
    auto i = op(0x831, {{16, 8, d}, {24, 8, a}, {32, 32, imm}, {64, 8, c}});
    put_signs(i, m, true);
    if (clamp) { field(i, 79, 1, 1); field(i, 87, 4, pred); }
    return i;
}
sn::Instruction hmul_reg_i(unsigned d, unsigned a, unsigned b, Mods m = {}) {
    auto i = op(0x232, {{16, 8, d}, {24, 8, a}, {32, 8, b}});
    put_signs(i, m, false);
    return i;
}
sn::Instruction hmul_imm_i(unsigned d, unsigned a, std::uint32_t imm, Mods m = {}) {
    auto i = op(0x832, {{16, 8, d}, {24, 8, a}, {32, 32, imm}});
    put_mod(i, 72, 1, m.neg_a); put_mod(i, 73, 1, m.abs_a); put_mod(i, 74, 2, m.swizzle_a);
    put_mod(i, 77, 1, m.sat); put_mod(i, 78, 1, m.scalar); put_mod(i, 80, 1, m.ftz);
    return i;
}
sn::Instruction hfma_reg_i(unsigned d, unsigned a, unsigned b, unsigned c, Mods m = {}) {
    auto i = op(0x231, {{16, 8, d}, {24, 8, a}, {32, 8, b}, {64, 8, c}});
    put_mod(i, 72, 1, m.neg_a); put_mod(i, 73, 1, m.abs_a); put_mod(i, 74, 2, m.swizzle_a);
    put_mod(i, 60, 2, m.swizzle_b); put_mod(i, 62, 1, m.abs_b); put_mod(i, 63, 1, m.neg_b);
    put_mod(i, 81, 2, m.swizzle_c); put_mod(i, 83, 1, m.abs_c); put_mod(i, 84, 1, m.neg_c);
    put_mod(i, 77, 1, m.sat); put_mod(i, 78, 1, m.scalar); put_mod(i, 80, 1, m.ftz); put_mod(i, 86, 1, m.exchange);
    return i;
}
// Immediate HSETP2 into P0 and P1, then P2R of those predicates into Rd.
sn::Instruction hsetp_i(unsigned a, std::uint32_t imm, Mods m, unsigned function, unsigned combine, bool conjunct, unsigned pred) {
    auto i = op(0x434, {{24, 8, a}, {32, 32, imm}, {69, 2, combine}, {76, 4, function}, {81, 3, 0}, {84, 3, 1}, {87, 4, pred}});
    put_mod(i, 71, 1, conjunct); put_mod(i, 72, 1, m.neg_a); put_mod(i, 73, 1, m.abs_a);
    put_mod(i, 74, 2, m.swizzle_a); put_mod(i, 80, 1, m.ftz);
    return i;
}
// Register HSETP2. The second source has its own selector and sign.
sn::Instruction hsetp_reg_i(unsigned a, unsigned b, Mods m, unsigned function, unsigned combine, bool conjunct, unsigned pred) {
    auto i = op(0x234, {{24, 8, a}, {32, 8, b}, {69, 2, combine}, {76, 4, function}, {81, 3, 0}, {84, 3, 1}, {87, 4, pred}});
    put_mod(i, 71, 1, conjunct); put_mod(i, 72, 1, m.neg_a); put_mod(i, 73, 1, m.abs_a); put_mod(i, 74, 2, m.swizzle_a);
    put_mod(i, 60, 2, m.swizzle_b); put_mod(i, 62, 1, m.abs_b); put_mod(i, 63, 1, m.neg_b); put_mod(i, 80, 1, m.ftz);
    return i;
}
sn::Instruction p2r(unsigned d) { return op(0x803, {{16, 8, d}, {24, 8, 255}, {32, 32, 0x7f}}); }
sn::Instruction hminmax_i(unsigned d, unsigned a, unsigned b, unsigned pred, Mods m = {}) {
    auto i = op(0x240, {{16, 8, d}, {24, 8, a}, {32, 8, b}, {87, 4, pred}});
    put_mod(i, 72, 1, m.neg_a); put_mod(i, 73, 1, m.abs_a); put_mod(i, 74, 2, m.swizzle_a);
    put_mod(i, 60, 2, m.swizzle_b); put_mod(i, 62, 1, m.abs_b); put_mod(i, 63, 1, m.neg_b);
    return i;
}
// Unsigned less into P0, with the always-true accumulator.
sn::Instruction below(unsigned a, unsigned b) {
    return op(0x20c, {{24, 8, a}, {32, 8, b}, {68, 4, PT}, {76, 3, 1}, {81, 3, 0}, {84, 3, PT}, {87, 4, PT}});
}
Words words(std::initializer_list<std::uint32_t> in) { Words v(in); v.resize(8, 0); return v; }
Fixture program(const std::vector<sn::Instruction>& body) {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    for (unsigned i = 0; i < 8; ++i) code.push_back(load(8 + i, i));
    code.insert(code.end(), body.begin(), body.end());
    code.push_back(exit_instruction());
    Fixture f(code); put(f.code, 0x48, 0xf); return f;
}

struct Case {
    std::string name;
    std::vector<sn::Instruction> body;
    std::vector<Words> inputs;
    std::function<Pixel(const Words&)> expected;
    // Bit 2*r + half is a packed lane whose NaN payload is not compared.
    std::uint32_t nan_lanes = 0;
};
bool matches(const Pixel& got, const Pixel& want, std::uint32_t nan_lanes) {
    if (!nan_lanes) return got == want;
    for (unsigned r = 0; r < 4; ++r) for (unsigned half = 0; half < 2; ++half) {
        const auto shift = 16u * half;
        const auto g = (got[r] >> shift) & 0xffffu, e = (want[r] >> shift) & 0xffffu;
        if ((nan_lanes >> (r * 2 + half)) & 1u) { if (!half_nan(g)) return false; }
        else if (g != e) return false;
    }
    return true;
}

void self_check() {
    for (std::uint32_t h = 0; h < 65536; ++h) if (!half_nan(h) && f32_to_f16(f16_to_f32(h)) != h)
        throw std::runtime_error("binary16 round trip");
    check(f16_to_f32(H1) == 0x3f800000u && f16_to_f32(H_SUB) == 0x33800000u && f16_to_f32(H_MAX) == 0x477fe000u, "widen");
    check(f16_to_f32(0x8000) == 0x80000000u && f16_to_f32(H_NORM) == 0x38800000u && f16_to_f32(H_NAN) == 0x7fc00000u, "widen specials");
    check(f16_to_f32(H_M10) == 0x3a800000u && (f16_to_f32(H_SUB) & 0x7f800000u) != 0, "subnormal widen");
    check(f32_to_f16(0x3f801000u) == H1 && f32_to_f16(0x3f802000u) == H_1P && f32_to_f16(0x3f803000u) == 0x3c02u, "ties");
    check(f32_to_f16(0x33000000u) == 0 && f32_to_f16(0x33000001u) == H_SUB && f32_to_f16(0x477ff000u) == H_INF, "underflow and overflow");
    check(f32_to_f16(0x38000000u) == 0x0200u && f32_to_f16(0x40200000u) == 0x4100u && f32_to_f16(0x40400000u) == 0x4200u, "known halves");
    check(hadd(pair(H1, H2), pair(H_M11, HN1)) == 0x3c003c00u, "plain add");
    check(hadd(pair(H1, H2), pair(H_M11, HN1), {.swizzle_b = 2}) == 0x40003c00u, "low half twice");
    check(hadd(pair(H1, H2), pair(H_M11, HN1), {.swizzle_b = 3}) == 0x3c000000u, "high half twice");
    check(hadd(pair(H1, H2), pair(H_HALF, H_MAX), {.swizzle_a = 2, .swizzle_b = 2}) == 0x3e003e00u, "both low halves");
    check(hadd(pair(H_SUB, 0x0401), pair(H_SUB, 0x8400)) == 0x00010002u, "subnormal sum");
    check(hadd(pair(0x8000, H_MAX), pair(0x8000, H_MAX)) == 0x7c008000u, "negative zero and overflow");
    check(hadd(pair(0, H1), pair(0x8000, HN1)) == 0u, "opposite zeros");
    check(hadd(pair(H_INF, H1), pair(H1, HN_INF)) == 0xfc007c00u, "infinities");
    check(hadd(pair(H1, H_1P), pair(H_M10, H_M11)) == 0x3c023c01u, "packed ties");
    check(hadd(pair(HN2, H1), 0, {.abs_a = true}) == 0x3c004000u, "absolute");
    check(hadd(pair(HN2, H1), 0, {.neg_a = true}) == 0xbc004000u, "negate");
    check(hadd(pair(HN2, H1), 0, {.abs_a = true, .neg_a = true}) == 0xbc00c000u, "absolute then negate");
    check(hadd(0, pair(H2, HN1), {.neg_b = true}) == 0x3c00c000u && hadd(0, pair(HN2, H1), {.abs_b = true}) == 0x3c004000u, "second source sign");
    check(hadd(pair(HN2, H2), 0, {.sat = true}) == 0x3c000000u, "saturate");
    check(hadd(pair(0x8001, 0x0400), pair(0x0001, 0x8401)) == 0x80010000u, "tiny difference");
    check(hadd(pair(0x8001, 0x0400), pair(0x0001, 0x8401), {.ftz = true}) == 0x80000000u, "flushed tiny difference");
    check(hadd(0, pair(0x8000, H1), {.swizzle_b = 2, .neg_a = true, .scalar = true}) == 0x80000000u, "widened negative zero");
    check(hadd(0, pair(0x8000, H1), {.swizzle_b = 2, .scalar = true}) == 0u, "negative zero plus zero");
    check(hadd(0, pair(H_SUB, H_MAX), {.swizzle_b = 2, .neg_a = true, .scalar = true}) == 0x33800000u, "widened subnormal");
    check(hadd(0, pair(H_SUB, H_MAX), {.swizzle_b = 3, .neg_a = true, .scalar = true}) == 0x477fe000u, "widened maximum");
    check(hadd(0, pair(0x8001, 0), {.swizzle_b = 2, .neg_a = true, .scalar = true}) == 0xb3800000u, "widened negative subnormal");
    check(hadd(0, pair(0x8001, 0), {.swizzle_b = 2, .neg_a = true, .ftz = true, .scalar = true}) == 0x80000000u, "flushed negative subnormal");
    check(hadd(0, pair(0x8001, 0), {.swizzle_b = 2, .ftz = true, .scalar = true}) == 0u, "flushed subnormal plus zero");
    check(hadd(0x00000001u, 0x3f800000u, {.swizzle_a = 1, .swizzle_b = 1, .scalar = true}) == 0x3f800000u, "binary32 subnormal");
    check(hadd(0x80000001u, 0, {.swizzle_a = 1, .swizzle_b = 1, .scalar = true}) == 0u, "negative binary32 subnormal");
    check(hadd(0x80000001u, 0x80000001u, {.swizzle_a = 1, .swizzle_b = 1, .scalar = true}) == 0x80000000u, "two negative subnormals");
    check(hadd(0x3fc00000u, 0x40200000u, {.swizzle_a = 1, .swizzle_b = 1, .scalar = true}) == 0x40800000u, "binary32 sum");
    check(hfma(pair(H1, H_1P), pair(H1, H1), pair(H_M11, H_M11)) == 0x3c023c00u, "fused ties");
    check(hfma(pair(H1, H_1P), pair(H1, H1), pair(H_M11, H_M11), {.scalar = true}) == 0x3f801000u, "scalar keeps the tie bit");
    check(hfma(pair(H1, H1), pair(H1, H2), 0, {.exchange = true}) == 0x3c00c000u, "exchanged factor");
    check(hfma(pair(HN1, HN2), pair(H1, H1), pair(HN1, H1), {.neg_a = true, .abs_c = true}) == 0x42004000u, "fused signs");
    check(hfma(pair(H1, H2), pair(H1, H1), 0x3f000000u, {.swizzle_c = 1, .scalar = true}) == 0x3fc00000u, "binary32 addend");
    check(hfma(pair(H_NORM, 0x8400), pair(H_M10, H_M10), 0) == 0x80010001u, "tiny product");
    check(hfma(pair(H_NORM, 0x8400), pair(H_M10, H_M10), 0, {.ftz = true}) == 0x80000000u, "flushed tiny product");
    check(hfma(pair(H2, HN2), pair(H1, H1), 0, {.sat = true}) == 0x00003c00u, "fused saturate");
    check(hfma(pair(H1, H1), pair(HN2, H2), 0, {}, true) == 0x40000000u, "clamp negatives");
    check(hfma(0, pair(HN1, HN1), pair(0x8000, 0x8000)) == 0x80008000u, "fused negative zero");
    check(hfma(0, pair(HN1, HN1), pair(0x8000, 0x8000), {}, true) == 0u, "clamp negative zero");
    check(hfma(pair(H_INF, H1), 0, pair(0, H1), {}, true) == 0x3c007fffu, "clamp NaN");
    check(hfma(pair(H_INF, HN_INF), pair(H1, H1), 0, {}, true) == 0x00007c00u, "clamp infinities");
    check(hfma(pair(H1, H2), pair(H2, H_HALF), 0, {.swizzle_a = 2}) == 0x38004000u, "factor low half");
    check(hfma(pair(H1, H2), pair(H1, H1), pair(H1, H2), {.swizzle_c = 3}) == 0x44004200u, "addend high half");
    check(hfma(pair(H1, H2), pair(H1, H1), 0x3f000000u, {.swizzle_c = 1}) == 0x41003e00u, "packed binary32 addend");
    check(hfma(pair(H1, H2), pair(H1, H2), pair(H1, HN1), {.exchange = true}) == 0x3c00bc00u, "exchange with addend");
    check(hadd(pair(H_SUB, H_SUB), pair(H_SUB, 0x8001)) == 0x00000002u, "two subnormals");
    check(hadd(pair(H_SUB, H_SUB), pair(H_SUB, 0x8001), {.ftz = true}) == 0u, "flushed subnormal inputs");
    const Mods b_low{.swizzle_b = 2}, b_high{.swizzle_b = 3}, b_abs{.abs_b = true}, b_neg{.neg_b = true};
    const Mods f32_ftz{.swizzle_a = 1, .ftz = true};
    check(hsetp(pair(H2, H1), pair(H1, H2), b_low, 4, 0, false, true) == 1u, "low half of a register source");
    check(hsetp(pair(H2, H1), pair(H1, H2), b_high, 4, 0, false, true) == 0u, "high half of a register source");
    check(hsetp(pair(H1, H1), pair(HN1, HN1), b_abs, 2, 0, false, true) == 3u, "absolute register source");
    check(hsetp(pair(H1, H2), pair(HN1, HN2), b_neg, 2, 0, false, true) == 3u, "negated register source");
    check(hsetp(0x00800000u, 0, f32_ftz, 4, 0, false, true) == 0u, "binary32 below the binary16 bound flushes");
    check(hsetp(0x3f800000u, 0, f32_ftz, 4, 0, false, true) == 3u, "binary32 one stays greater");
    check(hminmax(pair(H1, H2), pair(H2, H_HALF), true) == 0x38003c00u, "minimum");
    check(hminmax(pair(H1, H2), pair(H2, H_HALF), false) == 0x40004000u, "maximum");
    check(hminmax(pair(H1, H2), pair(H2, H_HALF), true, {.swizzle_b = 2}) == 0x40003c00u, "replicated minimum");
    check(hminmax(pair(H_NAN, H1), pair(H2, H_NAN), true) == 0x3c004000u && hminmax(pair(H_NAN, H1), pair(H2, H_NAN), false) == 0x3c004000u, "one NaN");
    check(hminmax(pair(H_SUB, 0x0003), pair(0x0002, H_SUB), true) == 0x00010001u, "subnormal minimum");
    check(hminmax(pair(H_SUB, 0x0003), pair(0x0002, H_SUB), false) == 0x00030002u, "subnormal maximum");
    check(hminmax(pair(H_INF, HN_INF), pair(H1, H1), false) == 0x3c007c00u, "infinite maximum");
    check(hminmax(pair(HN2, H1), pair(H1, HN1), true, {.abs_a = true}) == 0xbc003c00u, "absolute minimum");
    const auto mixed = hadd(pair(H_INF, H1), pair(HN_INF, H2));
    check(half_nan(mixed) && (mixed >> 16) == 0x4200u, "infinite cancellation");
    check(hmul(pair(H1, H2), pair(H2, H_HALF)) == pair(H2, H1), "plain multiply");
    check(hmul(pair(HN2, H1), pair(HN1, H2), {.abs_b = true}) == pair(HN2, H2), "absolute factor");
    check(hmul(pair(H1, H2), pair(H_HALF, H_MAX), {.swizzle_a = 2, .swizzle_b = 2}) == pair(H_HALF, H_HALF), "replicated low product");
    check(hmul(pair(H1, H2), pair(H2, H_HALF), {.swizzle_a = 3}) == pair(0x4400, H1), "replicated high product");
    check(hmul(pair(H2, HN2), pair(H1, H1), {.sat = true}) == pair(H1, 0), "saturated product");
    check(hmul(pair(H1, H2), pair(H2, H1), {.scalar = true}) == 0x40000000u, "scalar product");
    check(hmul(pair(H_NORM, 0x8400), pair(H_M10, H_M10)) == 0x80010001u, "tiny product");
    check(hmul(pair(H_NORM, 0x8400), pair(H_M10, H_M10), {.ftz = true}) == 0x80000000u, "flushed tiny product");
    check(hfma_reg(pair(H1, H1), pair(H2, H2), 0) == pair(H2, H2), "register factor");
    check(hfma_reg(pair(H1, H1), pair(H2, HN2), 0, {.neg_b = true}) == pair(HN2, H2), "negated register factor");
    check(hsetp(pair(H2, H_HALF), pair(H1, H1), {}, 4, 0, false, true) == 1u, "half greater");
    check(hsetp(pair(H2, H_HALF), pair(H1, H1), {.swizzle_a = 2}, 4, 0, true, true) == 1u, "half conjunction");
    check(hsetp(pair(H_SUB, H1), 0, {.ftz = true}, 5, 0, false, true) == 2u, "flushed half inequality");
    check(hsetp(pair(H_SUB, H1), 0, {}, 5, 0, false, true) == 3u, "subnormal half inequality");
}

std::vector<Case> cases() {
    std::vector<Case> c;
    const Mods low{.swizzle_b = 2}, high{.swizzle_b = 3}, both{.swizzle_a = 2, .swizzle_b = 2};
    c.push_back({"packed addition",
        {hadd_i(0, 8, 9), hadd_i(1, 8, 9, low), hadd_i(2, 8, 9, high), hadd_i(3, 8, 9, both)},
        {words({pair(H1, H2), pair(H_M11, HN1)}), words({pair(H_SUB, 0x0401), pair(H_SUB, 0x8400)}),
         words({pair(0x8000, H_MAX), pair(0x8000, H_MAX)}), words({pair(0, H1), pair(0x8000, HN1)}),
         words({pair(H_INF, H1), pair(H1, HN_INF)}), words({pair(H1, H_1P), pair(H_M10, H_M11)})},
        [=](const Words& w) { return Pixel{hadd(w[0], w[1]), hadd(w[0], w[1], low), hadd(w[0], w[1], high), hadd(w[0], w[1], both)}; }});

    const Mods abs_a{.abs_a = true}, neg_b{.neg_b = true}, abs_neg{.abs_a = true, .neg_a = true}, abs_b{.abs_b = true};
    c.push_back({"source signs",
        {hadd_i(0, 8, 255, abs_a), hadd_i(1, 255, 9, neg_b), hadd_i(2, 8, 255, abs_neg), hadd_i(3, 255, 9, abs_b)},
        {words({pair(HN2, H1), pair(H2, HN1)}), words({pair(0x8001, H_NORM), pair(0x8400, H_SUB)})},
        [=](const Words& w) { return Pixel{hadd(w[0], 0, abs_a), hadd(0, w[1], neg_b), hadd(w[0], 0, abs_neg), hadd(0, w[1], abs_b)}; }});

    const Mods sat{.sat = true}, sat_high{.swizzle_a = 3, .sat = true, .scalar = true}, sat_low{.sat = true, .scalar = true};
    c.push_back({"saturation",
        {hadd_i(0, 8, 9, sat), hadd_i(1, 8, 255, sat_low), hadd_i(2, 8, 255, sat_high), hadd_i(3, 10, 255, sat_low)},
        {words({pair(HN2, H2), 0, pair(H_NAN, 0)}), words({pair(H_NORM, H_HALF), pair(0x8401, 0), pair(H_INF, 0)})},
        [=](const Words& w) { return Pixel{hadd(w[0], w[1], sat), hadd(w[0], 0, sat_low), hadd(w[0], 0, sat_high), hadd(w[2], 0, sat_low)}; }});

    const Mods ftz{.ftz = true};
    c.push_back({"flush to zero",
        {hadd_i(0, 8, 9), hadd_i(1, 8, 9, ftz), hadd_i(2, 10, 11), hadd_i(3, 10, 11, ftz)},
        {words({pair(0x8001, H_NORM), pair(H_SUB, 0x8401), pair(H_SUB, H_SUB), pair(H_SUB, 0x8001)})},
        [=](const Words& w) { return Pixel{hadd(w[0], w[1]), hadd(w[0], w[1], ftz), hadd(w[2], w[3]), hadd(w[2], w[3], ftz)}; }});

    const Mods widen_low{.swizzle_b = 2, .neg_a = true, .scalar = true}, widen_high{.swizzle_b = 3, .neg_a = true, .scalar = true};
    const Mods plus_low{.swizzle_b = 2, .scalar = true}, plus_high{.swizzle_b = 3, .scalar = true};
    c.push_back({"scalar widening",
        {hadd_i(0, 255, 8, widen_low), hadd_i(1, 255, 8, widen_high), hadd_i(2, 255, 8, plus_low), hadd_i(3, 255, 8, plus_high)},
        {words({0x3c00bc00u}), words({0x7bff0001u}), words({0x80000400u}), words({0x3555c000u}), words({pair(0x8000, H1)})},
        [=](const Words& w) { return Pixel{hadd(0, w[0], widen_low), hadd(0, w[0], widen_high), hadd(0, w[0], plus_low), hadd(0, w[0], plus_high)}; }});

    const Mods flush_low{.swizzle_b = 2, .ftz = true, .scalar = true}, flush_neg{.swizzle_b = 2, .neg_a = true, .ftz = true, .scalar = true};
    c.push_back({"scalar flush",
        {hadd_i(0, 255, 8, flush_low), hadd_i(1, 255, 8, flush_neg), hadd_i(2, 255, 8, widen_low), hadd_i(3, 255, 8, plus_low)},
        {words({pair(0x8001, H1)})},
        [=](const Words& w) { return Pixel{hadd(0, w[0], flush_low), hadd(0, w[0], flush_neg), hadd(0, w[0], widen_low), hadd(0, w[0], plus_low)}; }});

    const Mods raw{.swizzle_a = 1, .swizzle_b = 1, .scalar = true};
    c.push_back({"binary32 sources",
        {hadd_i(0, 8, 9, raw), hadd_i(1, 10, 255, raw), hadd_i(2, 10, 10, raw), hadd_i(3, 11, 12, raw)},
        {words({0x00000001u, 0x3f800000u, 0x80000001u, 0x3fc00000u, 0x40200000u})},
        [=](const Words& w) { return Pixel{hadd(w[0], w[1], raw), hadd(w[2], 0, raw), hadd(w[2], w[2], raw), hadd(w[3], w[4], raw)}; }});

    const Mods exchanged{.exchange = true}, signs{.neg_a = true, .abs_c = true}, as_f32{.scalar = true};
    c.push_back({"fused multiply-add",
        {hfma_i(0, 8, pair(H1, H1), 9), hfma_i(1, 10, pair(H1, H2), 255, exchanged),
         hfma_i(2, 11, pair(H1, H1), 12, signs), hfma_i(3, 8, pair(H1, H1), 9, as_f32)},
        {words({pair(H1, H_1P), pair(H_M11, H_M11), pair(H1, H1), pair(HN1, HN2), pair(HN1, H1)})},
        [=](const Words& w) { return Pixel{hfma(w[0], pair(H1, H1), w[1]), hfma(w[2], pair(H1, H2), 0, exchanged),
                                           hfma(w[3], pair(H1, H1), w[4], signs), hfma(w[0], pair(H1, H1), w[1], as_f32)}; }});

    const Mods factor{.swizzle_a = 2}, addend_high{.swizzle_c = 3}, addend_raw{.swizzle_c = 1};
    c.push_back({"fused selectors",
        {hfma_i(0, 8, pair(H2, H_HALF), 255, factor), hfma_i(1, 8, pair(H1, H1), 8, addend_high),
         hfma_i(2, 8, pair(H1, H1), 12, addend_raw), hfma_i(3, 8, pair(H1, H2), 10, exchanged)},
        {words({pair(H1, H2), 0, pair(H1, HN1), 0, 0x3f000000u})},
        [=](const Words& w) { return Pixel{hfma(w[0], pair(H2, H_HALF), 0, factor), hfma(w[0], pair(H1, H1), w[0], addend_high),
                                           hfma(w[0], pair(H1, H1), w[4], addend_raw), hfma(w[0], pair(H1, H2), w[2], exchanged)}; }});

    c.push_back({"fused flush and clamp",
        {hfma_i(0, 8, pair(H_M10, H_M10), 255), hfma_i(1, 8, pair(H_M10, H_M10), 255, ftz),
         hfma_i(2, 9, pair(H1, H1), 255, sat), hfma_i(3, 10, pair(HN2, H2), 255, {}, true)},
        {words({pair(H_NORM, 0x8400), pair(H2, HN2), pair(H1, H1)})},
        [=](const Words& w) { return Pixel{hfma(w[0], pair(H_M10, H_M10), 0), hfma(w[0], pair(H_M10, H_M10), 0, ftz),
                                           hfma(w[1], pair(H1, H1), 0, sat), hfma(w[2], pair(HN2, H2), 0, {}, true)}; }});

    c.push_back({"clamp specials",
        {hfma_i(0, 8, 0, 9, {}, true), hfma_i(1, 10, pair(H1, H1), 255, {}, true),
         hfma_i(2, 255, pair(HN1, HN1), 11, {}, true), hfma_i(3, 255, pair(HN1, HN1), 11, {}, true, NOT_PT)},
        {words({pair(H_INF, H1), pair(0, H1), pair(H_INF, HN_INF), pair(0x8000, 0x8000)})},
        [=](const Words& w) { return Pixel{hfma(w[0], 0, w[1], {}, true), hfma(w[2], pair(H1, H1), 0, {}, true),
                                           hfma(0, pair(HN1, HN1), w[3], {}, true), hfma(0, pair(HN1, HN1), w[3])}; }});

    c.push_back({"dynamic clamp",
        {below(12, 13), hfma_i(0, 255, pair(HN1, HN1), 8, {}, true, 0), hfma_i(1, 9, pair(HN2, H2), 255, {}, true, 0)},
        {words({pair(0x8000, 0x8000), pair(H1, H1), 0, 0, 0, 1}), words({pair(0x8000, 0x8000), pair(H1, H1), 0, 0, 1, 0})},
        [=](const Words& w) { const bool hold = w[4] < w[5];
            return Pixel{hfma(0, pair(HN1, HN1), w[0], {}, hold), hfma(w[1], pair(HN2, H2), 0, {}, hold), 0, 0}; }});

    const Mods replicate{.swizzle_b = 2};
    c.push_back({"half extrema",
        {hminmax_i(0, 8, 9, PT), hminmax_i(1, 8, 9, NOT_PT), hminmax_i(2, 8, 9, PT, replicate), hminmax_i(3, 8, 9, NOT_PT, abs_a)},
        {words({pair(H1, H2), pair(H2, H_HALF)}), words({pair(HN2, H1), pair(H1, HN1)})},
        [=](const Words& w) { return Pixel{hminmax(w[0], w[1], true), hminmax(w[0], w[1], false),
                                           hminmax(w[0], w[1], true, replicate), hminmax(w[0], w[1], false, abs_a)}; }});
    c.push_back({"extrema boundaries",
        {hminmax_i(0, 8, 9, PT), hminmax_i(1, 8, 9, NOT_PT), hminmax_i(2, 10, 11, PT), hminmax_i(3, 12, 13, NOT_PT)},
        {words({pair(H_SUB, 0x0003), pair(0x0002, H_SUB), pair(H_NAN, H1), pair(H2, H_NAN), pair(H_INF, HN_INF), pair(H1, H1)})},
        [=](const Words& w) { return Pixel{hminmax(w[0], w[1], true), hminmax(w[0], w[1], false),
                                           hminmax(w[2], w[3], true), hminmax(w[4], w[5], false)}; }});
    c.push_back({"dynamic extrema",
        {below(12, 13), hminmax_i(0, 8, 9, 0), hminmax_i(1, 8, 9, 0, replicate)},
        {words({pair(H1, H2), pair(H2, H_HALF), 0, 0, 0, 1}), words({pair(H1, H2), pair(H2, H_HALF), 0, 0, 1, 0})},
        [=](const Words& w) { const bool minimum = w[4] < w[5];
            return Pixel{hminmax(w[0], w[1], minimum), hminmax(w[0], w[1], minimum, replicate), 0, 0}; }});

    const Mods abs_factor{.abs_b = true}, low_abs{.swizzle_a = 2, .swizzle_b = 2, .abs_b = true};
    const Mods low_factor{.swizzle_b = 2}, high_source{.swizzle_a = 3};
    c.push_back({"packed half multiply",
        {hmul_reg_i(0, 8, 9, abs_factor), hmul_reg_i(1, 8, 9, low_abs),
         hmul_reg_i(2, 8, 9, low_factor), hmul_imm_i(3, 8, 0x39b839b8u, high_source)},
        {words({pair(H1, H2), pair(HN2, H_HALF)}), words({pair(HN1, H1), pair(H2, HN2)}),
         words({pair(H_SUB, H_NORM), pair(H_SUB, H_M10)})},
        [=](const Words& w) { return Pixel{hmul(w[0], w[1], abs_factor), hmul(w[0], w[1], low_abs),
                                           hmul(w[0], w[1], low_factor), hmul(w[0], 0x39b839b8u, high_source)}; }});
    const Mods mul_sat{.sat = true}, mul_ftz{.ftz = true}, mul_low{.swizzle_a = 2}, mul_neg{.neg_a = true};
    c.push_back({"half multiply modifiers",
        {hmul_reg_i(0, 8, 9, mul_sat), hmul_reg_i(1, 10, 11, mul_ftz),
         hmul_imm_i(2, 8, pair(H2, H1), mul_low), hmul_reg_i(3, 8, 9, mul_neg)},
        {words({pair(H2, HN2), pair(H1, H1), pair(H_NORM, 0x8400), pair(H_M10, H_M10)})},
        [=](const Words& w) { return Pixel{hmul(w[0], w[1], mul_sat), hmul(w[2], w[3], mul_ftz),
                                           hmul(w[0], pair(H2, H1), mul_low), hmul(w[0], w[1], mul_neg)}; }});
    const Mods factor_low{.swizzle_b = 2, .neg_b = true}, factor_high{.swizzle_a = 3, .swizzle_b = 3};
    const Mods factor_abs{.swizzle_b = 2, .swizzle_c = 2, .abs_b = true}, factor_mix{.swizzle_a = 2, .swizzle_b = 3, .swizzle_c = 2};
    c.push_back({"register fused multiply-add",
        {hfma_reg_i(0, 8, 9, 10, factor_low), hfma_reg_i(1, 8, 9, 10, factor_high),
         hfma_reg_i(2, 8, 9, 10, factor_abs), hfma_reg_i(3, 8, 9, 10, factor_mix)},
        {words({pair(H1, H2), pair(H2, H_HALF), pair(H1, HN1)})},
        [=](const Words& w) { return Pixel{hfma_reg(w[0], w[1], w[2], factor_low), hfma_reg(w[0], w[1], w[2], factor_high),
                                           hfma_reg(w[0], w[1], w[2], factor_abs), hfma_reg(w[0], w[1], w[2], factor_mix)}; }});
    const Mods low_twice{.swizzle_a = 2}, absolute{.abs_a = true}, negated{.neg_a = true}, flushed{.ftz = true};
    c.push_back({"half predicate compare",
        {hsetp_i(8, pair(H1, H1), {}, 4, 0, false, PT), p2r(0),
         hsetp_i(8, pair(H1, H1), low_twice, 4, 0, true, PT), p2r(1),
         hsetp_i(9, 0, flushed, 5, 0, false, PT), p2r(2),
         hsetp_i(9, 0, {}, 5, 0, false, PT), p2r(3)},
        {words({pair(H2, H_HALF), pair(H_SUB, H1)})},
        [=](const Words& w) { return Pixel{hsetp(w[0], pair(H1, H1), {}, 4, 0, false, true),
                                           hsetp(w[0], pair(H1, H1), low_twice, 4, 0, true, true),
                                           hsetp(w[1], 0, flushed, 5, 0, false, true),
                                           hsetp(w[1], 0, {}, 5, 0, false, true)}; }});
    const Mods b_low{.swizzle_b = 2}, b_high{.swizzle_b = 3}, b_abs{.abs_b = true}, b_neg{.neg_b = true};
    c.push_back({"half predicate register source",
        {hsetp_reg_i(8, 9, b_low, 4, 0, false, PT), p2r(0),
         hsetp_reg_i(8, 9, b_high, 4, 0, false, PT), p2r(1),
         hsetp_reg_i(8, 9, b_abs, 2, 0, false, PT), p2r(2),
         hsetp_reg_i(8, 9, b_neg, 2, 0, false, PT), p2r(3)},
        {words({pair(H2, H1), pair(H1, H2)}), words({pair(H1, H1), pair(HN1, HN1)}), words({pair(H1, H2), pair(HN1, HN2)})},
        [=](const Words& w) { return Pixel{hsetp(w[0], w[1], b_low, 4, 0, false, true),
                                           hsetp(w[0], w[1], b_high, 4, 0, false, true),
                                           hsetp(w[0], w[1], b_abs, 2, 0, false, true),
                                           hsetp(w[0], w[1], b_neg, 2, 0, false, true)}; }});
    const Mods f32_ftz{.swizzle_a = 1, .ftz = true};
    c.push_back({"half predicate binary32 flush",
        {hsetp_i(8, 0, f32_ftz, 4, 0, false, PT), p2r(0),
         hsetp_i(9, 0, f32_ftz, 4, 0, false, PT), p2r(1)},
        {words({0x00800000u, 0x3f800000u})},
        [=](const Words& w) { return Pixel{hsetp(w[0], 0, f32_ftz, 4, 0, false, true),
                                           hsetp(w[1], 0, f32_ftz, 4, 0, false, true), 0, 0}; }});
    c.push_back({"half predicate signs and combination",
        {hsetp_i(8, pair(H1, H1), absolute, 4, 0, false, PT), p2r(0),
         hsetp_i(9, pair(H1, H1), negated, 4, 0, false, PT), p2r(1),
         hsetp_i(10, pair(H1, H1), {}, 4, 1, false, NOT_PT), p2r(2),
         hsetp_i(10, pair(H1, H1), {}, 4, 0, false, NOT_PT), p2r(3)},
        {words({pair(HN2, H1), pair(H1, H2), pair(H2, H_HALF)})},
        [=](const Words& w) { return Pixel{hsetp(w[0], pair(H1, H1), absolute, 4, 0, false, true),
                                           hsetp(w[1], pair(H1, H1), negated, 4, 0, false, true),
                                           hsetp(w[2], pair(H1, H1), {}, 4, 1, false, false),
                                           hsetp(w[2], pair(H1, H1), {}, 4, 0, false, false)}; }});
    c.push_back({"unsigned NaN lanes",
        {hadd_i(0, 8, 9), hadd_i(1, 10, 11), hfma_i(2, 12, 0, 13), hadd_i(3, 8, 9, high)},
        {words({pair(H_INF, H1), pair(HN_INF, H2), pair(H1, H_NAN), pair(H1, H1), pair(H_INF, H1), pair(0, H1)})},
        [=](const Words& w) { return Pixel{hadd(w[0], w[1]), hadd(w[2], w[3]), hfma(w[4], 0, w[5]), hadd(w[0], w[1], high)}; },
        (1u << 0) | (1u << 3) | (1u << 4)});
    return c;
}
} // namespace

int main() { @autoreleasepool { try {
    self_check();
    MetalHarness gpu;
    unsigned checked = 0;
    for (const auto& test : cases()) {
        auto compiled = gpu.compile(program(test.body));
        for (const auto& input : test.inputs) {
            const auto got = gpu.run(compiled, input), want = test.expected(input);
            if (!matches(got, want, test.nan_lanes)) {
                std::fprintf(stderr, "%s: got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n", test.name.c_str(),
                    got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                throw std::runtime_error("native half-precision GPU result");
            }
            ++checked;
        }
    }
    std::printf("Native half-precision Metal execution passed (%u vectors)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
