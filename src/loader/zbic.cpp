// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "zbic.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>
#if ARMIGO_HAS_ZSTD
#include <zstd.h>
#endif

namespace armigo::loader::detail {
namespace {
using Bytes = std::vector<std::byte>;
struct Invalid {};
class Cursor {
public:
    explicit Cursor(std::span<const std::byte> bytes) : bytes_(bytes) {}
    std::uint64_t integer(unsigned count) {
        const auto bytes = take(count);
        std::uint64_t value = 0;
        for (unsigned i = 0; i < count; ++i)
            value |= std::uint64_t{std::to_integer<unsigned char>(bytes[i])} << (8 * i);
        return value;
    }
    std::span<const std::byte> take(std::size_t size) {
        if (size > bytes_.size()) throw Invalid{};
        const auto result = bytes_.first(size);
        bytes_ = bytes_.subspan(size);
        return result;
    }
    std::span<const std::byte> rest() const { return bytes_; }
private:
    std::span<const std::byte> bytes_;
};
void append(Bytes& output, std::span<const std::byte> bytes) {
    output.insert(output.end(), bytes.begin(), bytes.end());
}
void integer(Bytes& output, std::uint64_t value, unsigned count) {
    if (count < 8 && value >= (std::uint64_t{1} << (count * 8))) throw Invalid{};
    for (unsigned i = 0; i < count; ++i) output.push_back(std::byte((value >> (8 * i)) & 255));
}

// A reverse byte-normalized integer carries the interpolative cumulative counts.
Bytes entropy(Cursor& source, unsigned max_symbol, unsigned max_log) {
    const auto descriptor = source.integer(1);
    const auto low_probability = descriptor >> 7;
    const auto payload = source.take(descriptor & 127);
    std::size_t left = payload.size();
    auto refill = [&](std::uint64_t value) {
        while (left && value < (std::uint64_t{1} << 40))
            value = (value << 8) | std::to_integer<unsigned char>(payload[--left]);
        return value;
    };
    auto state = refill(0);
    const auto last_symbol = state % 52 + 1;
    state = refill(state / 52);
    const unsigned log = (state & 7) + 5;
    state = refill(state >> 3);
    if (last_symbol > max_symbol || log > max_log) throw Invalid{};
    const auto total = std::uint32_t{1} << log;
    const auto last_count = state % total + 1 + (low_probability ? last_symbol + 1 : 0);
    state = refill(state / total);
    const unsigned power = std::bit_ceil(static_cast<unsigned>(last_symbol + 1));
    std::array<std::uint32_t, 65> cumulative{};
    if (power >= cumulative.size()) throw Invalid{};
    cumulative[power] = last_count;
    auto interpolate = [&](auto&& self, unsigned first, unsigned last) -> void {
        if (last - first < 2) return;
        const auto middle = (first + last) / 2;
        const auto low = cumulative[first], high = cumulative[last];
        if (low == high) {
            std::fill(cumulative.begin() + first + 1, cumulative.begin() + last, low);
        } else {
            const auto count = high - low + 1;
            cumulative[middle] = low + state % count;
            state = refill(state / count);
        }
        self(self, middle, last);
        self(self, first, middle);
    };
    interpolate(interpolate, 0, power);
    if (left) throw Invalid{};
    std::vector<int> counts;
    unsigned sum = 0;
    for (unsigned i = 0; i <= last_symbol; ++i) {
        const int count = static_cast<int>(cumulative[i + 1]) -
                          static_cast<int>(cumulative[i]) - static_cast<int>(low_probability);
        if (count < -1) throw Invalid{};
        counts.push_back(count);
        sum += count < 0 ? -count : count;
    }
    if (sum != total) throw Invalid{};

    // Standard normalized-count coding is little-endian and shrinks its alphabet
    // bit width as the remaining probability mass decreases.
    Bytes output;
    std::uint64_t bits = 0;
    unsigned bit_count = 0;
    auto emit = [&](unsigned value, unsigned width) {
        if (width > 16 || value >= (1u << width)) throw Invalid{};
        bits |= std::uint64_t{value} << bit_count;
        bit_count += width;
        while (bit_count >= 8) {
            output.push_back(std::byte(bits & 255));
            bits >>= 8;
            bit_count -= 8;
        }
    };
    emit(log - 5, 4);
    int remaining = total + 1, threshold = total;
    unsigned width = log + 1;
    bool previous_zero = false;
    for (std::size_t symbol = 0; symbol < counts.size() && remaining > 1;) {
        if (previous_zero) {
            unsigned run = 0;
            while (symbol < counts.size() && counts[symbol] == 0) { ++run; ++symbol; }
            if (symbol == counts.size()) throw Invalid{};
            while (run >= 3) { emit(3, 2); run -= 3; }
            emit(run, 2);
        }
        int count = counts[symbol++];
        const int boundary = 2 * threshold - 1 - remaining;
        remaining -= count < 0 ? -count : count;
        if (remaining < 1) throw Invalid{};
        ++count;
        if (count >= threshold) count += boundary;
        emit(count, width - (count < boundary));
        previous_zero = count == 1;
        while (remaining < threshold) { --width; threshold >>= 1; }
    }
    if (remaining != 1) throw Invalid{};
    if (bit_count) output.push_back(std::byte(bits & 255));
    return output;
}

Bytes compressed_block(std::span<const std::byte> bytes) {
    Cursor source(bytes);
    const auto first = source.integer(1);
    const auto kind = first & 3, format = (first >> 2) & 3;
    Bytes output;
    if (kind < 2) {
        const unsigned header_size = (format == 0 || format == 2) ? 1 : (format == 1 ? 2 : 3);
        auto header = first;
        if (header_size > 1) header |= source.integer(header_size - 1) << 8;
        const auto size = header >> (header_size == 1 ? 3 : 4);
        integer(output, header, header_size);
        append(output, source.take(kind == 0 ? size : 1));
    } else {
        const unsigned header_size = format < 2 ? 3 : format + 2;
        const unsigned width = format < 2 ? 10 : (format == 2 ? 14 : 18);
        const auto header = first | (source.integer(header_size - 1) << 8);
        const auto size = (header >> 4) & ((std::uint64_t{1} << width) - 1);
        const auto encoded_size = header >> (4 + width);
        Cursor literals(source.take(encoded_size));
        Bytes converted;
        if (kind == 2) {
            const auto table_size = literals.integer(1);
            if (table_size < 128) {
                Cursor weights(literals.take(table_size));
                auto table = entropy(weights, 12, 6);
                append(table, weights.rest());
                if (table.size() >= 128) throw Invalid{};
                integer(converted, table.size(), 1);
                append(converted, table);
            } else {
                integer(converted, table_size, 1);
            }
        }
        append(converted, literals.rest());
        // Four-stream headers can grow into the next size format. A single-stream
        // literal section keeps its original format and has a 1023-byte limit.
        unsigned out_format = format, out_width = width, out_size = header_size;
        if (converted.size() >= (std::uint64_t{1} << out_width)) {
            if (format == 0 || format == 3) throw Invalid{};
            ++out_format; out_width += 4; ++out_size;
        }
        integer(output, kind | (out_format << 2) | (size << 4) |
                        (std::uint64_t{converted.size()} << (4 + out_width)), out_size);
        append(output, converted);
    }
    const auto first_count = source.integer(1);
    integer(output, first_count, 1);
    if (first_count >= 128) {
        const unsigned extra = first_count == 255 ? 2 : 1;
        append(output, source.take(extra));
    }
    if (first_count) {
        const auto modes = source.integer(1);
        if (modes & 3) throw Invalid{};
        integer(output, modes, 1);
        const unsigned shifts[] = {6, 4, 2};
        const unsigned symbols[] = {35, 31, 52};
        const unsigned logs[] = {9, 8, 9};
        for (unsigned i = 0; i < 3; ++i) {
            const auto mode = (modes >> shifts[i]) & 3;
            if (mode == 1) append(output, source.take(1));
            if (mode == 2) append(output, entropy(source, symbols[i], logs[i]));
        }
    } else if (!source.rest().empty()) throw Invalid{};
    append(output, source.rest());
    return output;
}

Bytes standard_frame(std::span<const std::byte> bytes) {
    Cursor source(bytes);
    if (source.integer(4) != 0x4349425a) throw Invalid{};
    const auto descriptor = source.integer(1);
    if (descriptor & 0x18) throw Invalid{};
    const bool single = descriptor & 0x20;
    const unsigned dictionary_sizes[] = {0, 1, 2, 4};
    const unsigned content_sizes[] = {single ? 1u : 0u, 2, 4, 8};
    Bytes output;
    integer(output, 0xfd2fb528, 4);
    integer(output, descriptor, 1);
    if (!single) append(output, source.take(1));
    const auto dictionary_size = dictionary_sizes[descriptor & 3];
    if (source.integer(dictionary_size) != 0) throw Invalid{};
    integer(output, 0, dictionary_size);
    append(output, source.take(content_sizes[descriptor >> 6]));
    for (;;) {
        const auto header = source.integer(3);
        const auto kind = (header >> 1) & 3, size = header >> 3;
        if (kind == 3 || size > 128 * 1024) throw Invalid{};
        const auto payload = source.take(kind == 1 ? 1 : size);
        if (kind == 2) {
            auto converted = compressed_block(payload);
            integer(output, (converted.size() << 3) | (header & 7), 3);
            append(output, converted);
        } else {
            integer(output, header, 3);
            append(output, payload);
        }
        if (header & 1) break;
    }
    if (descriptor & 4) append(output, source.take(4));
    if (!source.rest().empty()) throw Invalid{};
    return output;
}
} // namespace

std::optional<std::string> decompress_zbic(std::span<const std::byte> input,
                                         std::span<std::byte> output) {
#if ARMIGO_HAS_ZSTD
    try {
        const auto frame = standard_frame(input);
        const auto result = ZSTD_decompress(output.data(), output.size(), frame.data(), frame.size());
        if (ZSTD_isError(result)) return std::string("ZBIC decompression failed: ") + ZSTD_getErrorName(result);
        if (result != output.size()) return "ZBIC output size does not match the segment.";
        return {};
    } catch (const Invalid&) {
        return "ZBIC frame or entropy table is malformed or unsupported.";
    } catch (const std::bad_alloc&) {
        return "ZBIC conversion exceeded available host memory.";
    }
#else
    (void)input; (void)output;
    return "ZBIC requires a build with Zstandard support.";
#endif
}
} // namespace armigo::loader::detail
