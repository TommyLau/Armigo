// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace armigo::audio::renderer {

// Signed binary fixed point with I integer and F fractional bits, stored in a
// two's-complement integer of exactly I + F bits. The renderer's DSP state uses
// these widths (Q14 and Q15 in 32- and 64-bit words), and its arithmetic must
// round exactly like the firmware's DSP code:
//  - 32-bit values multiply and divide in a 64-bit intermediate;
//  - 64-bit values have no wider intermediate, so multiplication splits each
//    operand into integer and fraction parts and division runs a shift-subtract
//    loop on the magnitudes.
// Conversions to integers round half up (to_int, to_long) or truncate toward
// negative infinity (the _floor variants).
template <std::size_t I, std::size_t F>
class FixedPoint;

namespace fixed_detail {

template <std::size_t Bits>
struct Storage;
template <>
struct Storage<16> {
    using type = std::int16_t;
    using wide = std::int32_t;
    static constexpr bool has_wide = true;
};
template <>
struct Storage<32> {
    using type = std::int32_t;
    using wide = std::int64_t;
    static constexpr bool has_wide = true;
};
template <>
struct Storage<64> {
    using type = std::int64_t;
    using wide = std::int64_t;
    static constexpr bool has_wide = false;
};

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

}  // namespace fixed_detail

template <std::size_t I, std::size_t F>
class FixedPoint {
public:
    static constexpr std::size_t fractional_bits = F;
    static constexpr std::size_t integer_bits = I;
    static constexpr std::size_t total_bits = I + F;
    using storage = fixed_detail::Storage<total_bits>;
    using base_type = typename storage::type;
    using unsigned_type = std::make_unsigned_t<base_type>;
    static constexpr base_type fractional_mask =
        static_cast<base_type>(~(static_cast<unsigned_type>(~unsigned_type(0)) << F));
    static constexpr base_type integer_mask = static_cast<base_type>(~fractional_mask);
    static constexpr base_type one = static_cast<base_type>(base_type(1) << F);

    constexpr FixedPoint() = default;
    template <fixed_detail::Arithmetic Number>
    constexpr FixedPoint(Number n) : data_(static_cast<base_type>(n * one)) {}

    // Widening conversion between formats: the integer and fraction parts are
    // placed separately, so a narrower fraction gains zero low bits.
    template <std::size_t I2, std::size_t F2>
    constexpr explicit FixedPoint(FixedPoint<I2, F2> other) {
        static_assert(I2 <= I && F2 <= F, "fixed point conversion can only widen");
        using T = FixedPoint<I2, F2>;
        const base_type fraction = static_cast<base_type>(other.data_ & T::fractional_mask);
        const base_type integer = static_cast<base_type>((other.data_ & T::integer_mask) >> F2);
        data_ = static_cast<base_type>((integer << F) | (fraction << (F - F2)));
    }

    static constexpr FixedPoint from_base(base_type n) {
        FixedPoint value;
        value.data_ = n;
        return value;
    }

    friend constexpr auto operator<=>(FixedPoint, FixedPoint) = default;
    friend constexpr bool operator==(FixedPoint, FixedPoint) = default;

    constexpr bool operator!() const { return !data_; }
    constexpr FixedPoint operator-() const { return from_base(static_cast<base_type>(-data_)); }
    constexpr FixedPoint operator+() const { return *this; }

    constexpr FixedPoint& operator+=(FixedPoint n) {
        data_ = static_cast<base_type>(data_ + n.data_);
        return *this;
    }
    constexpr FixedPoint& operator-=(FixedPoint n) {
        data_ = static_cast<base_type>(data_ - n.data_);
        return *this;
    }
    constexpr FixedPoint& operator*=(FixedPoint n) {
        data_ = multiply(data_, n.data_);
        return *this;
    }
    constexpr FixedPoint& operator/=(FixedPoint n) {
        data_ = divide(data_, n.data_);
        return *this;
    }
    template <std::integral Integer>
    constexpr FixedPoint& operator>>=(Integer n) {
        data_ >>= n;
        return *this;
    }
    template <std::integral Integer>
    constexpr FixedPoint& operator<<=(Integer n) {
        data_ = static_cast<base_type>(data_ << n);
        return *this;
    }

    constexpr void round_up() { data_ = static_cast<base_type>(data_ + ((data_ & fractional_mask) >> 1)); }
    constexpr int to_int() {
        round_up();
        return static_cast<int>((data_ & integer_mask) >> F);
    }
    constexpr unsigned int to_uint() {
        round_up();
        return static_cast<unsigned int>((data_ & integer_mask) >> F);
    }
    constexpr std::int64_t to_long() {
        round_up();
        return static_cast<std::int64_t>((data_ & integer_mask) >> F);
    }
    constexpr int to_int_floor() const { return static_cast<int>((data_ & integer_mask) >> F); }
    constexpr std::int64_t to_long_floor() const {
        return static_cast<std::int64_t>((data_ & integer_mask) >> F);
    }
    constexpr unsigned int to_uint_floor() const {
        return static_cast<unsigned int>((data_ & integer_mask) >> F);
    }
    constexpr float to_float() const { return static_cast<float>(data_) / one; }
    constexpr double to_double() const { return static_cast<double>(data_) / one; }
    constexpr base_type to_raw() const { return data_; }
    constexpr void clear_int() { data_ &= fractional_mask; }
    constexpr base_type get_frac() const { return data_ & fractional_mask; }

    base_type data_{};

private:
    static constexpr base_type multiply(base_type lhs, base_type rhs) {
        if constexpr (storage::has_wide) {
            using wide = typename storage::wide;
            wide product = static_cast<wide>(lhs) * static_cast<wide>(rhs);
            product >>= F;
            return static_cast<base_type>(product);
        } else {
            // Split multiplication: each operand is hi * 2^F + lo with lo in [0, 2^F).
            const base_type a_hi = (lhs & integer_mask) >> F;
            const base_type b_hi = (rhs & integer_mask) >> F;
            const base_type a_lo = lhs & fractional_mask;
            const base_type b_lo = rhs & fractional_mask;
            const auto wrap = [](unsigned_type value) { return static_cast<base_type>(value); };
            const unsigned_type x1 = static_cast<unsigned_type>(a_hi) * static_cast<unsigned_type>(b_hi);
            const unsigned_type x2 = static_cast<unsigned_type>(a_hi) * static_cast<unsigned_type>(b_lo);
            const unsigned_type x3 = static_cast<unsigned_type>(a_lo) * static_cast<unsigned_type>(b_hi);
            const base_type x4 = wrap(static_cast<unsigned_type>(a_lo) * static_cast<unsigned_type>(b_lo));
            return wrap((x1 << F) + (x3 + x2) + static_cast<unsigned_type>(x4 >> F));
        }
    }

    static constexpr base_type divide(base_type numerator, base_type denominator) {
        if constexpr (storage::has_wide) {
            using wide = typename storage::wide;
            wide t = numerator;
            t <<= F;
            return static_cast<base_type>(t / denominator);
        } else {
            if (denominator == 0) return 0;
            bool negative = false;
            if (numerator < 0) {
                negative = !negative;
                numerator = static_cast<base_type>(-numerator);
            }
            if (denominator < 0) {
                negative = !negative;
                denominator = static_cast<base_type>(-denominator);
            }
            constexpr int bits = static_cast<int>(total_bits);
            unsigned_type n = static_cast<unsigned_type>(numerator);
            unsigned_type d = static_cast<unsigned_type>(denominator);
            unsigned_type x = 1;
            unsigned_type answer = 0;
            while (n >= d && ((d >> (bits - 1)) & 1) == 0) {
                x <<= 1;
                d <<= 1;
            }
            while (x != 0) {
                if (n >= d) {
                    n -= d;
                    answer += x;
                }
                x >>= 1;
                d >>= 1;
            }
            unsigned_type l1 = n;
            unsigned_type l2 = static_cast<unsigned_type>(denominator);
            while ((l1 >> (bits - F)) > 0) {
                l1 >>= 1;
                l2 >>= 1;
            }
            const unsigned_type lo = l2 == 0 ? 0 : (l1 << F) / l2;
            const auto quotient = static_cast<base_type>((answer << F) | lo);
            return negative ? static_cast<base_type>(-quotient) : quotient;
        }
    }
};

// Mixed integer widths with the same fraction promote to the wider format.
template <std::size_t I1, std::size_t I2, std::size_t F>
    requires(I1 != I2)
constexpr auto operator+(FixedPoint<I1, F> lhs, FixedPoint<I2, F> rhs) {
    using T = std::conditional_t<(I1 >= I2), FixedPoint<I1, F>, FixedPoint<I2, F>>;
    return T::from_base(lhs.to_raw()) + T::from_base(rhs.to_raw());
}
template <std::size_t I1, std::size_t I2, std::size_t F>
    requires(I1 != I2)
constexpr auto operator-(FixedPoint<I1, F> lhs, FixedPoint<I2, F> rhs) {
    using T = std::conditional_t<(I1 >= I2), FixedPoint<I1, F>, FixedPoint<I2, F>>;
    return T::from_base(lhs.to_raw()) - T::from_base(rhs.to_raw());
}
template <std::size_t I1, std::size_t I2, std::size_t F>
    requires(I1 != I2)
constexpr auto operator*(FixedPoint<I1, F> lhs, FixedPoint<I2, F> rhs) {
    using T = std::conditional_t<(I1 >= I2), FixedPoint<I1, F>, FixedPoint<I2, F>>;
    return T::from_base(lhs.to_raw()) * T::from_base(rhs.to_raw());
}
template <std::size_t I1, std::size_t I2, std::size_t F>
    requires(I1 != I2)
constexpr auto operator/(FixedPoint<I1, F> lhs, FixedPoint<I2, F> rhs) {
    using T = std::conditional_t<(I1 >= I2), FixedPoint<I1, F>, FixedPoint<I2, F>>;
    return T::from_base(lhs.to_raw()) / T::from_base(rhs.to_raw());
}

template <std::size_t I, std::size_t F>
constexpr FixedPoint<I, F> operator+(FixedPoint<I, F> lhs, FixedPoint<I, F> rhs) {
    return lhs += rhs;
}
template <std::size_t I, std::size_t F>
constexpr FixedPoint<I, F> operator-(FixedPoint<I, F> lhs, FixedPoint<I, F> rhs) {
    return lhs -= rhs;
}
template <std::size_t I, std::size_t F>
constexpr FixedPoint<I, F> operator*(FixedPoint<I, F> lhs, FixedPoint<I, F> rhs) {
    return lhs *= rhs;
}
template <std::size_t I, std::size_t F>
constexpr FixedPoint<I, F> operator/(FixedPoint<I, F> lhs, FixedPoint<I, F> rhs) {
    return lhs /= rhs;
}

template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator+(FixedPoint<I, F> lhs, Number rhs) {
    return lhs += FixedPoint<I, F>(rhs);
}
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator-(FixedPoint<I, F> lhs, Number rhs) {
    return lhs -= FixedPoint<I, F>(rhs);
}
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator*(FixedPoint<I, F> lhs, Number rhs) {
    return lhs *= FixedPoint<I, F>(rhs);
}
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator/(FixedPoint<I, F> lhs, Number rhs) {
    return lhs /= FixedPoint<I, F>(rhs);
}
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator+(Number lhs, FixedPoint<I, F> rhs) {
    return FixedPoint<I, F>(lhs) += rhs;
}
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator-(Number lhs, FixedPoint<I, F> rhs) {
    return FixedPoint<I, F>(lhs) -= rhs;
}
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator*(Number lhs, FixedPoint<I, F> rhs) {
    return FixedPoint<I, F>(lhs) *= rhs;
}
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr FixedPoint<I, F> operator/(Number lhs, FixedPoint<I, F> rhs) {
    return FixedPoint<I, F>(lhs) /= rhs;
}

template <std::size_t I, std::size_t F, std::integral Integer>
constexpr FixedPoint<I, F> operator<<(FixedPoint<I, F> lhs, Integer rhs) {
    return lhs <<= rhs;
}
template <std::size_t I, std::size_t F, std::integral Integer>
constexpr FixedPoint<I, F> operator>>(FixedPoint<I, F> lhs, Integer rhs) {
    return lhs >>= rhs;
}

template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator>(FixedPoint<I, F> lhs, Number rhs) { return lhs > FixedPoint<I, F>(rhs); }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator<(FixedPoint<I, F> lhs, Number rhs) { return lhs < FixedPoint<I, F>(rhs); }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator>=(FixedPoint<I, F> lhs, Number rhs) { return lhs >= FixedPoint<I, F>(rhs); }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator<=(FixedPoint<I, F> lhs, Number rhs) { return lhs <= FixedPoint<I, F>(rhs); }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator==(FixedPoint<I, F> lhs, Number rhs) { return lhs == FixedPoint<I, F>(rhs); }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator!=(FixedPoint<I, F> lhs, Number rhs) { return lhs != FixedPoint<I, F>(rhs); }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator>(Number lhs, FixedPoint<I, F> rhs) { return FixedPoint<I, F>(lhs) > rhs; }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator<(Number lhs, FixedPoint<I, F> rhs) { return FixedPoint<I, F>(lhs) < rhs; }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator>=(Number lhs, FixedPoint<I, F> rhs) { return FixedPoint<I, F>(lhs) >= rhs; }
template <std::size_t I, std::size_t F, fixed_detail::Arithmetic Number>
constexpr bool operator<=(Number lhs, FixedPoint<I, F> rhs) { return FixedPoint<I, F>(lhs) <= rhs; }

static_assert(sizeof(FixedPoint<18, 14>) == 4);
static_assert(sizeof(FixedPoint<50, 14>) == 8);

}  // namespace armigo::audio::renderer
