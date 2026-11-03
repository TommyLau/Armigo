// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Strict LZ4 block-format decoder.
//
// Decodes one raw LZ4 block (no frame header, no checksum) into a caller-owned
// output buffer whose size must equal the expected decompressed size exactly.
// The decoder never reads outside the input span, never writes outside the
// output span, and resolves match references only against bytes it has
// already produced in that output span.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace armigo::loader {

/// Structured failure reported by the LZ4 block decoder.
enum class Lz4Error : std::uint8_t {
    success = 0,
    /// The input ended inside a token, length extension, literal run, or
    /// match offset.
    truncated_input,
    /// A literal run or match would write past the end of the output buffer.
    output_overflow,
    /// A match offset is zero or refers to bytes before the output start.
    invalid_match_offset,
    /// The input was fully consumed before the output buffer was filled.
    incomplete_output,
    /// The block ends with a match instead of a literal-only final sequence.
    missing_final_literals,
};

/// English diagnostic text for an `Lz4Error` value. Never returns null.
const char* to_string(Lz4Error error) noexcept;

/// Result of one block decode.
///
/// `input_offset` is the input position at which the failure was detected,
/// or the input size on success.
struct Lz4Result {
    Lz4Error error = Lz4Error::success;
    std::size_t input_offset = 0;

    constexpr bool ok() const noexcept { return error == Lz4Error::success; }
};

/// Decodes `input` as one complete LZ4 block into `output`.
///
/// Success requires that every input byte is consumed and that exactly
/// `output.size()` bytes are produced. On failure the contents of `output`
/// are unspecified but no byte outside it has been written.
Lz4Result decompress_lz4_block(std::span<const std::byte> input,
                               std::span<std::byte> output) noexcept;

} // namespace armigo::loader
