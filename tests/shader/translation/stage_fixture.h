// ============================================================================
// Armigo Tests - Shader Stage Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

// An independently assembled shader container: shader control plus code with
// a header (SPH) and an instruction stream, the layout the translator reads.
namespace armigo::test::stage_fixture {

inline std::uint64_t field(std::uint64_t word, unsigned start, unsigned width, std::uint64_t value) {
    const auto mask = ((std::uint64_t{1} << width) - 1) << start;
    return (word & ~mask) | ((value << start) & mask);
}

// An encoding pattern written from bit 63 down, under the always-true P7.
inline std::uint64_t pattern(std::string_view bits) {
    std::uint64_t word = 0;
    for (std::size_t index = 0; index < bits.size(); ++index)
        if (bits[index] == '1') word |= std::uint64_t{1} << (63 - index);
    return word | (std::uint64_t{7} << 16);
}

// IPA into `reg` from the attribute byte offset; the SPH interpolation map
// supplies the mode.
inline std::uint64_t interpolate(unsigned reg, unsigned byte_offset) {
    auto word = pattern("11100000000000");
    word = field(word, 0, 8, reg);
    word = field(word, 8, 8, 255);
    word = field(word, 20, 8, 255);
    word = field(word, 28, 10, byte_offset);
    word = field(word, 39, 8, 255);
    word = field(word, 47, 3, 7);
    word = field(word, 52, 2, 0);
    return field(word, 54, 2, 1);
}

inline std::uint64_t exit_word() { return field(pattern("111000110000"), 0, 4, 15); }

inline void put32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
inline void put64(std::vector<std::byte>& bytes, std::size_t offset, std::uint64_t value) {
    for (unsigned i = 0; i != 8; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

struct Container {
    std::vector<std::byte> control, code;
    std::size_t program_begin = 0x30, program_end = 0;  // the identity bytes
};

// NOP: no effect; pads a program past the 112-byte identity prefix.
inline std::uint64_t nop_word() { return 0x50b0000000070f00ull; }

// A fragment stage that interpolates attribute 0x80 into R0 and exports
// colour target 0's first component.
inline Container fragment_stage() {
    std::vector<std::uint64_t> words{interpolate(0, 0x80), nop_word(), nop_word(), nop_word(), nop_word(),
                                     exit_word()};
    std::vector<std::byte> sph(0x50);
    put32(sph, 0, 5u << 10);
    sph[0x18] = std::byte{1};
    put32(sph, 0x48, 1);
    std::vector<std::byte> stream;
    for (std::size_t index = 0; index < words.size(); ++index) {
        if (index % 3 == 0) stream.resize(stream.size() + 8);
        stream.resize(stream.size() + 8);
        put64(stream, stream.size() - 8, words[index]);
    }
    Container result;
    result.program_end = 0x80 + stream.size();
    result.code.resize(0x30);
    result.code.insert(result.code.end(), sph.begin(), sph.end());
    result.code.insert(result.code.end(), stream.begin(), stream.end());
    result.code.resize(result.code.size() + 16, std::byte{0x6d});
    put32(result.code, 0, 0x12345678);
    result.control.resize(0x718);
    put32(result.control, 0, 0x98761234);
    put32(result.control, 4, 1);
    put32(result.control, 8, 14);
    put32(result.control, 0x20, 0x718);
    put32(result.control, 0x700, static_cast<std::uint32_t>(result.program_end));
    put32(result.control, 0x704, static_cast<std::uint32_t>(result.code.size()));
    put32(result.control, 0x708, 0x30);
    put32(result.control, 0x714, 1);
    return result;
}

} // namespace armigo::test::stage_fixture
