// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/fma.h>

#include <array>
#include <bit>

namespace armigo::shader {
namespace {

// Exact unsigned magnitudes in units of 2^-298. Nine limbs cover every finite
// normal product and addend, including the sum before its final rounding.
using Magnitude = std::array<std::uint64_t, 9>;

std::uint32_t flush(std::uint32_t bits) noexcept {
    return (bits & 0x7f800000u) == 0 ? bits & 0x80000000u : bits;
}

void insert(Magnitude& destination, std::uint64_t value, unsigned shift) noexcept {
    const auto index = shift / 64;
    const auto offset = shift % 64;
    destination[index] = value << offset;
    if (offset != 0 && index + 1 < destination.size()) {
        destination[index + 1] = value >> (64 - offset);
    }
}

int compare(const Magnitude& lhs, const Magnitude& rhs) noexcept {
    for (int index = 8; index >= 0; --index) {
        if (lhs[index] != rhs[index]) return lhs[index] > rhs[index] ? 1 : -1;
    }
    return 0;
}

Magnitude combine(const Magnitude& lhs, const Magnitude& rhs, bool subtract) noexcept {
    Magnitude result{};
    std::uint64_t carry = 0;
    for (unsigned index = 0; index < result.size(); ++index) {
        if (subtract) {
            const auto difference = lhs[index] - rhs[index];
            result[index] = difference - carry;
            carry = (lhs[index] < rhs[index]) || (difference < carry);
        } else {
            const auto sum = lhs[index] + rhs[index];
            result[index] = sum + carry;
            carry = (sum < lhs[index]) || (result[index] < sum);
        }
    }
    return result;
}

int highest_bit(const Magnitude& value) noexcept {
    for (int index = 8; index >= 0; --index) {
        if (value[index] != 0) {
            return index * 64 + 63 - static_cast<int>(std::countl_zero(value[index]));
        }
    }
    return -1;
}

std::uint32_t round_right(const Magnitude& value, unsigned shift) noexcept {
    const auto index = shift / 64;
    const auto offset = shift % 64;
    auto retained = value[index] >> offset;
    if (offset != 0 && index + 1 < value.size()) {
        retained |= value[index + 1] << (64 - offset);
    }
    const unsigned guard_index = (shift - 1) / 64;
    const unsigned guard_offset = (shift - 1) % 64;
    const bool guard = ((value[guard_index] >> guard_offset) & 1) != 0;
    bool sticky = (value[guard_index] & ((std::uint64_t{1} << guard_offset) - 1)) != 0;
    for (unsigned i = 0; i < guard_index; ++i) sticky |= value[i] != 0;
    return static_cast<std::uint32_t>(retained) +
           (guard && (sticky || (retained & 1) != 0) ? 1u : 0u);
}

std::uint32_t encode(const Magnitude& value, std::uint32_t sign) noexcept {
    const int highest = highest_bit(value);
    if (highest < 0) return sign;
    int exponent = highest - 171;
    if (exponent >= 255) return sign | 0x7f800000u;
    if (exponent <= 0) {
        // Subnormal rounding precedes FTZ: a carry into minimum normal survives.
        const auto rounded = round_right(value, 149);
        return sign | (rounded == 0x00800000u ? rounded : 0u);
    }
    auto rounded = round_right(value, static_cast<unsigned>(highest - 23));
    if (rounded == 0x01000000u) {
        rounded >>= 1;
        ++exponent;
    }
    if (exponent >= 255) return sign | 0x7f800000u;
    return sign | (static_cast<std::uint32_t>(exponent) << 23) |
           (rounded & 0x007fffffu);
}

}  // namespace

std::uint32_t fma_rne_ftz(std::uint32_t a_bits, std::uint32_t b_bits,
                         std::uint32_t c_bits) noexcept {
    const auto a = flush(a_bits), b = flush(b_bits), c = flush(c_bits);
    const auto a_magnitude = a & 0x7fffffffu;
    const auto b_magnitude = b & 0x7fffffffu;
    const auto c_magnitude = c & 0x7fffffffu;
    const auto product_sign = (a ^ b) & 0x80000000u;
    const auto addend_sign = c & 0x80000000u;
    if (a_magnitude > 0x7f800000u || b_magnitude > 0x7f800000u ||
        c_magnitude > 0x7f800000u) return 0x7fc00000u;
    const bool product_infinite = a_magnitude == 0x7f800000u || b_magnitude == 0x7f800000u;
    if (product_infinite) {
        if (a_magnitude == 0 || b_magnitude == 0 ||
            (c_magnitude == 0x7f800000u && product_sign != addend_sign)) {
            return 0x7fc00000u;
        }
        return product_sign | 0x7f800000u;
    }
    if (c_magnitude == 0x7f800000u) return c;

    Magnitude product{}, addend{};
    if (a_magnitude != 0 && b_magnitude != 0) {
        const auto lhs = (a & 0x007fffffu) | 0x00800000u;
        const auto rhs = (b & 0x007fffffu) | 0x00800000u;
        const auto shift = ((a >> 23) & 255u) + ((b >> 23) & 255u) - 2;
        insert(product, std::uint64_t{lhs} * rhs, shift);
    }
    if (c_magnitude != 0) {
        insert(addend, (c & 0x007fffffu) | 0x00800000u, ((c >> 23) & 255u) + 148);
    }
    if (product_sign == addend_sign) return encode(combine(product, addend, false), product_sign);
    const int ordering = compare(product, addend);
    if (ordering == 0) return 0u;
    return ordering > 0 ? encode(combine(product, addend, true), product_sign)
                        : encode(combine(addend, product, true), addend_sign);
}

std::string emit_fma_msl() {
    return R"msl(inline void armigo_fma_insert(thread ulong* target, ulong value, uint shift) {
    const uint index = shift / 64u;
    const uint offset = shift % 64u;
    target[index] = value << offset;
    if (offset != 0u && index + 1u < 9u) target[index + 1u] = value >> (64u - offset);
}
inline uint armigo_fma_round(thread const ulong* value, uint shift) {
    const uint index = shift / 64u;
    const uint offset = shift % 64u;
    ulong retained = value[index] >> offset;
    if (offset != 0u && index + 1u < 9u) retained |= value[index + 1u] << (64u - offset);
    const uint guard_index = (shift - 1u) / 64u;
    const uint guard_offset = (shift - 1u) % 64u;
    const bool guard = ((value[guard_index] >> guard_offset) & 1) != 0;
    bool sticky = (value[guard_index] & ((ulong(1) << guard_offset) - 1)) != 0;
    for (uint i = 0; i < guard_index; ++i) sticky |= value[i] != 0;
    return uint(retained) + (guard && (sticky || (retained & 1) != 0) ? 1u : 0u);
}
inline uint armigo_fma_rne_ftz(uint a, uint b, uint c) {
    if ((a & 0x7f800000u) == 0u) a &= 0x80000000u;
    if ((b & 0x7f800000u) == 0u) b &= 0x80000000u;
    if ((c & 0x7f800000u) == 0u) c &= 0x80000000u;
    const uint am = a & 0x7fffffffu, bm = b & 0x7fffffffu, cm = c & 0x7fffffffu;
    const uint ps = (a ^ b) & 0x80000000u, cs = c & 0x80000000u;
    if (am > 0x7f800000u || bm > 0x7f800000u || cm > 0x7f800000u) return 0x7fc00000u;
    if (am == 0x7f800000u || bm == 0x7f800000u) {
        if (am == 0u || bm == 0u || (cm == 0x7f800000u && ps != cs)) return 0x7fc00000u;
        return ps | 0x7f800000u;
    }
    if (cm == 0x7f800000u) return c;
    ulong product[9] = {0}, addend[9] = {0}, magnitude[9];
    if (am != 0u && bm != 0u) {
        const ulong lhs = (a & 0x007fffffu) | 0x00800000u;
        const ulong rhs = (b & 0x007fffffu) | 0x00800000u;
        armigo_fma_insert(product, lhs * rhs, ((a >> 23) & 255u) + ((b >> 23) & 255u) - 2u);
    }
    if (cm != 0u) armigo_fma_insert(addend, (c & 0x007fffffu) | 0x00800000u, ((c >> 23) & 255u) + 148u);
    int ordering = 0;
    for (int i = 8; i >= 0; --i) {
        if (product[i] != addend[i]) { ordering = product[i] > addend[i] ? 1 : -1; break; }
    }
    if (ps != cs && ordering == 0) return 0u;
    uint sign = ps;
    if (ps != cs && ordering < 0) sign = cs;
    ulong carry = 0;
    for (uint i = 0; i < 9u; ++i) {
        if (ps == cs) {
            const ulong sum = product[i] + addend[i];
            magnitude[i] = sum + carry;
            carry = (sum < product[i]) || (magnitude[i] < sum);
        } else {
            const ulong lhs = ordering > 0 ? product[i] : addend[i];
            const ulong rhs = ordering > 0 ? addend[i] : product[i];
            const ulong difference = lhs - rhs;
            magnitude[i] = difference - carry;
            carry = (lhs < rhs) || (difference < carry);
        }
    }
    int highest = -1;
    for (int i = 8; i >= 0; --i) {
        if (magnitude[i] != 0) { highest = i * 64 + 63 - int(clz(magnitude[i])); break; }
    }
    if (highest < 0) return sign;
    int exponent = highest - 171;
    if (exponent >= 255) return sign | 0x7f800000u;
    if (exponent <= 0) {
        const uint rounded = armigo_fma_round(magnitude, 149u);
        return sign | (rounded == 0x00800000u ? rounded : 0u);
    }
    uint rounded = armigo_fma_round(magnitude, uint(highest - 23));
    if (rounded == 0x01000000u) { rounded >>= 1; ++exponent; }
    if (exponent >= 255) return sign | 0x7f800000u;
    return sign | (uint(exponent) << 23) | (rounded & 0x007fffffu);
}
)msl";
}

}  // namespace armigo::shader
