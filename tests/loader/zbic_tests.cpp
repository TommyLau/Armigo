// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "zbic.h"
#include "nso_fixture.h"
#include <armigo/loader/nso_module.h>
#include "zbic_fixture.h"
#include "test_support.h"
#include <string_view>

using namespace armigo::loader::test;
using armigo::loader::detail::decompress_zbic;
int main() {
    std::vector<std::byte> expected;
    for (unsigned i = 0; i < 128; ++i) {
        for (unsigned repeat = 0; repeat < 1 + (i * 13) % 7; ++repeat) {
            for (const char c : std::string_view("abracadabra")) expected.push_back(std::byte(c));
            expected.push_back(std::byte(65 + (i * 7) % 26));
        }
        expected.insert(expected.end(), i % 13, std::byte((i * 17) % 251));
    }
    const auto input = std::as_bytes(std::span(synthetic_zbic));
    std::vector<std::byte> output(expected.size());
    ARMIGO_CHECK(!decompress_zbic(input, output));
    ARMIGO_CHECK(output == expected);
    for (std::size_t size = 0; size < input.size(); ++size)
        ARMIGO_CHECK(decompress_zbic(input.first(size), output).has_value());
    ARMIGO_CHECK(decompress_zbic(input, std::span(output).first(output.size() - 1)).has_value());
    output.resize(output.size() + 1);
    ARMIGO_CHECK(decompress_zbic(input, output).has_value());
    auto raw = bytes_of({'Z', 'B', 'I', 'C', 0x20, 3, 25, 0, 0, 'a', 'b', 'c'});
    output.resize(3);
    ARMIGO_CHECK(!decompress_zbic(raw, output));
    ARMIGO_CHECK(output == bytes_of({'a', 'b', 'c'}));
    auto rle = bytes_of({'Z', 'B', 'I', 'C', 0x20, 16, 131, 0, 0, 'x'});
    output.assign(16, std::byte{});
    ARMIGO_CHECK(!decompress_zbic(rle, output));
    ARMIGO_CHECK(output == std::vector<std::byte>(16, std::byte{'x'}));
    raw[0] = std::byte{};
    ARMIGO_CHECK(decompress_zbic(raw, output).has_value());
    raw[0] = std::byte{'Z'};
    raw.push_back(std::byte{});
    ARMIGO_CHECK(decompress_zbic(raw, output).has_value());
    raw.pop_back(); raw[4] |= std::byte{8};
    ARMIGO_CHECK(decompress_zbic(raw, output).has_value());
    auto broken = std::vector<std::byte>(input.begin(), input.end());
    broken[10] = std::byte{0xff};
    output.resize(expected.size());
    ARMIGO_CHECK(decompress_zbic(broken, output).has_value());
    std::vector<std::byte> nso(0x100);
    put_u32(nso, 0, 0x304f534e);
    put_u32(nso, 0x0c, 0xff);
    for (unsigned i = 0; i < 3; ++i) {
        put_u32(nso, field_file_offset(i), nso.size());
        put_u32(nso, field_image_offset(i), i * 0x2000);
        put_u32(nso, field_size(i), expected.size());
        put_u32(nso, field_file_size(i), input.size());
        const auto hash = armigo::loader::detail::sha256(expected);
        std::copy(hash->begin(), hash->end(), nso.begin() + field_hash(i));
        nso.insert(nso.end(), input.begin(), input.end());
    }
    const auto module = armigo::loader::read_nso_module(nso);
    ARMIGO_CHECK(module.ok());
    if (module.ok()) {
        ARMIGO_CHECK(module.image->zbic_compression);
        for (const auto& segment : module.image->segments) {
            ARMIGO_CHECK(segment.hash == armigo::loader::HashCheck::verified);
            ARMIGO_CHECK(std::equal(expected.begin(), expected.end(), module.image->bytes.begin() + segment.image_offset));
        }
        ARMIGO_CHECK(module.image->segments[0].permission == armigo::loader::SegmentPermission::execute);
    }
    auto bad_hash = nso;
    bad_hash[field_hash(1)] ^= std::byte{1};
    ARMIGO_CHECK(armigo::loader::read_nso_module(bad_hash).error.code == armigo::loader::NsoError::hash_mismatch);
    nso[get_u32(nso, field_file_offset(0))] = std::byte{};
    ARMIGO_CHECK(armigo::loader::read_nso_module(nso).error.code == armigo::loader::NsoError::zbic_decompression_failed);
    return finish("loader.zbic");
}
