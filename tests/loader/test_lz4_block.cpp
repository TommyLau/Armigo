// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/lz4_block.h>

#include "test_support.h"

#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace armigo::loader;
using armigo::loader::test::bytes_of;
using armigo::loader::test::lz4_encode;

Lz4Result decode(const std::vector<std::byte>& input, std::vector<std::byte>& output) {
    return decompress_lz4_block(input, output);
}

bool equals(const std::vector<std::byte>& bytes, std::string_view text) {
    return bytes.size() == text.size() && std::memcmp(bytes.data(), text.data(), text.size()) == 0;
}

void literal_only_blocks() {
    std::vector<std::byte> empty;
    ARMIGO_CHECK(decode(bytes_of({0x00}), empty).ok());

    std::vector<std::byte> out(3);
    const auto result = decode(bytes_of({0x30, 'a', 'b', 'c'}), out);
    ARMIGO_CHECK(result.ok() && result.input_offset == 4);
    ARMIGO_CHECK(equals(out, "abc"));
}

void overlapping_matches() {
    // "ab", then a match of length 6 at offset 2, then the final literal "x".
    std::vector<std::byte> out(9);
    ARMIGO_CHECK(decode(bytes_of({0x22, 'a', 'b', 0x02, 0x00, 0x10, 'x'}), out).ok());
    ARMIGO_CHECK(equals(out, "abababab" "x"));

    // Run-length expansion with offset 1 and an extended match length:
    // nibble 15 + extension 1 + minimum 4 = 20 copies after the seed byte.
    std::vector<std::byte> run(22);
    ARMIGO_CHECK(decode(bytes_of({0x1F, 'z', 0x01, 0x00, 0x01, 0x10, '!'}), run).ok());
    ARMIGO_CHECK(equals(run, std::string(21, 'z') + "!"));

    // Non-overlapping match copies an earlier region verbatim.
    std::vector<std::byte> copy(13);
    ARMIGO_CHECK(decode(bytes_of({0x80, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 0x08, 0x00,
                                  0x10, '.'}),
                        copy)
                     .ok());
    ARMIGO_CHECK(equals(copy, "abcdefgh" "abcd" "."));
}

void extended_literal_length() {
    // 15 + 255 + 10 = 280 literal bytes.
    std::vector<std::byte> input = bytes_of({0xF0, 0xFF, 0x0A});
    for (int i = 0; i < 280; ++i) {
        input.push_back(static_cast<std::byte>(i));
    }
    std::vector<std::byte> out(280);
    ARMIGO_CHECK(decode(input, out).ok());
    ARMIGO_CHECK(out[279] == static_cast<std::byte>(279 & 0xFF));
}

void malformed_blocks() {
    std::vector<std::byte> out(4);
    ARMIGO_CHECK(decode({}, out).error == Lz4Error::truncated_input);

    std::vector<std::byte> none;
    ARMIGO_CHECK(decode({}, none).error == Lz4Error::truncated_input);

    // Literal run longer than the remaining input.
    ARMIGO_CHECK(decode(bytes_of({0x40, 'a', 'b'}), out).error == Lz4Error::truncated_input);
    // Literal length extension missing.
    std::vector<std::byte> big(64);
    ARMIGO_CHECK(decode(bytes_of({0xF0}), big).error == Lz4Error::truncated_input);
    // Match offset cut to one byte.
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x01}), out).error == Lz4Error::truncated_input);
    // Match length extension missing.
    ARMIGO_CHECK(decode(bytes_of({0x1F, 'a', 0x01, 0x00}), big).error ==
                 Lz4Error::truncated_input);
    // Offset zero is never valid.
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x00, 0x00, 0x00}), out).error ==
                 Lz4Error::invalid_match_offset);
    // Offset reaching before the output start.
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x02, 0x00, 0x00}), out).error ==
                 Lz4Error::invalid_match_offset);
    // Literal run larger than the output.
    std::vector<std::byte> two(2);
    ARMIGO_CHECK(decode(bytes_of({0x30, 'a', 'b', 'c'}), two).error == Lz4Error::output_overflow);
    // Match larger than the output.
    std::vector<std::byte> five(5);
    ARMIGO_CHECK(decode(bytes_of({0x11, 'a', 0x01, 0x00, 0x00}), five).error ==
                 Lz4Error::output_overflow);
    // Output not filled.
    ARMIGO_CHECK(decode(bytes_of({0x20, 'a', 'b'}), out).error == Lz4Error::incomplete_output);
    // Block ends with a match.
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x01, 0x00}), five).error ==
                 Lz4Error::missing_final_literals);
    // Bytes after a literal run are a match offset; a match past a full
    // output overflows.
    std::vector<std::byte> one(1);
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x01, 0x00, 0x10, 'b'}), one).error ==
                 Lz4Error::output_overflow);
    // A single trailing byte after literals is a truncated offset.
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x00}), one).error == Lz4Error::truncated_input);
    // An empty literal-only final sequence after a match is legal.
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x01, 0x00, 0x00}), five).ok());
    // A non-empty final literal run after the output is full overflows.
    ARMIGO_CHECK(decode(bytes_of({0x10, 'a', 0x01, 0x00, 0x10, 'b'}), five).error ==
                 Lz4Error::output_overflow);

    // An enormous length extension is rejected without unbounded accumulation.
    std::vector<std::byte> hostile = bytes_of({0xF0});
    hostile.insert(hostile.end(), 100000, std::byte{0xFF});
    const auto result = decode(hostile, big);
    ARMIGO_CHECK(result.error == Lz4Error::output_overflow && result.input_offset < 16);
}

std::vector<std::byte> synthetic_payload(std::size_t size, std::uint32_t seed) {
    std::vector<std::byte> data(size);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < size; ++i) {
        state = state * 1664525u + 1013904223u;
        const std::uint32_t mode = (state >> 28) & 3u;
        if (mode == 0 && i >= 7) {
            data[i] = data[i - 7];
        } else if (mode == 1 && i >= 1) {
            data[i] = data[i - 1];
        } else {
            data[i] = static_cast<std::byte>(state >> 16);
        }
    }
    return data;
}

void round_trips() {
    for (std::size_t size : {0u, 1u, 5u, 13u, 64u, 300u, 4096u, 70000u}) {
        for (std::uint32_t seed : {1u, 7u}) {
            const auto original = synthetic_payload(size, seed);
            const auto encoded = lz4_encode(original);
            std::vector<std::byte> decoded(size);
            ARMIGO_CHECK(decompress_lz4_block(encoded, decoded).ok());
            ARMIGO_CHECK(decoded == original);

            std::vector<std::byte> larger(size + 1);
            ARMIGO_CHECK(decompress_lz4_block(encoded, larger).error ==
                         Lz4Error::incomplete_output);
            if (size != 0) {
                std::vector<std::byte> smaller(size - 1);
                ARMIGO_CHECK(decompress_lz4_block(encoded, smaller).error ==
                             Lz4Error::output_overflow);
                // Every truncation of a valid block must fail cleanly.
                for (std::size_t cut = 0; cut < encoded.size() && cut < 64; ++cut) {
                    std::vector<std::byte> partial(encoded.begin(), encoded.begin() + cut);
                    ARMIGO_CHECK(!decompress_lz4_block(partial, decoded).ok());
                }
            }
        }
    }
}

} // namespace

int main() {
    literal_only_blocks();
    overlapping_matches();
    extended_literal_length();
    malformed_blocks();
    round_trips();
    return armigo::loader::test::finish("loader.lz4_block");
}
