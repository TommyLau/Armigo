// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/lz4_block.h>

#include <cstring>

namespace armigo::loader {

namespace {

// Minimum match length encoded by a token's low nibble of zero.
constexpr std::size_t kMinimumMatchLength = 4;
// Nibble value indicating that additional length bytes follow.
constexpr std::size_t kLengthExtensionMarker = 15;

// Reads a token length extension. `length` starts at the nibble value; each
// extension byte adds up to 255 and a byte below 255 terminates the run.
// Accumulation stops early once the length exceeds `limit`, which keeps the
// arithmetic bounded and reports overflow before any copy.
Lz4Error read_extended_length(std::span<const std::byte> input, std::size_t& position,
                              std::size_t& length, std::size_t limit) noexcept {
    for (;;) {
        if (position >= input.size()) {
            return Lz4Error::truncated_input;
        }
        const auto value = static_cast<std::uint8_t>(input[position++]);
        length += value;
        if (length > limit) {
            return Lz4Error::output_overflow;
        }
        if (value != 0xFF) {
            return Lz4Error::success;
        }
    }
}

} // namespace

const char* to_string(Lz4Error error) noexcept {
    switch (error) {
    case Lz4Error::success:
        return "success";
    case Lz4Error::truncated_input:
        return "LZ4 input ended inside a sequence";
    case Lz4Error::output_overflow:
        return "LZ4 sequence exceeds the expected decompressed size";
    case Lz4Error::invalid_match_offset:
        return "LZ4 match offset is zero or precedes the output start";
    case Lz4Error::incomplete_output:
        return "LZ4 input ended before the expected decompressed size was produced";
    case Lz4Error::missing_final_literals:
        return "LZ4 block ends with a match instead of a literal-only sequence";
    }
    return "unknown LZ4 error";
}

Lz4Result decompress_lz4_block(std::span<const std::byte> input,
                               std::span<std::byte> output) noexcept {
    std::size_t in = 0;
    std::size_t out = 0;
    const std::size_t out_size = output.size();

    if (input.empty()) {
        // A block always contains at least one token, even for empty output.
        return {Lz4Error::truncated_input, 0};
    }

    while (in < input.size()) {
        const std::size_t token_offset = in;
        const auto token = static_cast<std::uint8_t>(input[in++]);

        std::size_t literal_length = token >> 4;
        if (literal_length == kLengthExtensionMarker) {
            const Lz4Error error =
                read_extended_length(input, in, literal_length, out_size - out);
            if (error != Lz4Error::success) {
                return {error, in};
            }
        }
        if (literal_length > out_size - out) {
            return {Lz4Error::output_overflow, token_offset};
        }
        if (literal_length > input.size() - in) {
            return {Lz4Error::truncated_input, input.size()};
        }
        if (literal_length != 0) {
            std::memcpy(output.data() + out, input.data() + in, literal_length);
            in += literal_length;
            out += literal_length;
        }

        if (in == input.size()) {
            // Literal-only final sequence.
            if (out != out_size) {
                return {Lz4Error::incomplete_output, in};
            }
            return {Lz4Error::success, in};
        }

        if (input.size() - in < 2) {
            return {Lz4Error::truncated_input, input.size()};
        }
        const std::size_t match_offset = static_cast<std::size_t>(
            static_cast<std::uint8_t>(input[in]) |
            (static_cast<std::uint8_t>(input[in + 1]) << 8));
        if (match_offset == 0 || match_offset > out) {
            return {Lz4Error::invalid_match_offset, in};
        }
        in += 2;

        std::size_t match_length = token & 0x0F;
        if (match_length == kLengthExtensionMarker) {
            const Lz4Error error =
                read_extended_length(input, in, match_length, out_size - out);
            if (error != Lz4Error::success) {
                return {error, in};
            }
        }
        match_length += kMinimumMatchLength;
        if (match_length > out_size - out) {
            return {Lz4Error::output_overflow, token_offset};
        }

        std::byte* destination = output.data() + out;
        const std::byte* source = destination - match_offset;
        if (match_offset >= match_length) {
            std::memcpy(destination, source, match_length);
        } else {
            // Overlapping copy: each produced byte may feed later bytes of the
            // same match, so the copy must proceed strictly forward.
            for (std::size_t i = 0; i < match_length; ++i) {
                destination[i] = source[i];
            }
        }
        out += match_length;

        if (in == input.size()) {
            return {Lz4Error::missing_final_literals, in};
        }
    }
    // Unreachable: the loop returns when the input is exhausted.
    return {Lz4Error::truncated_input, in};
}

} // namespace armigo::loader
