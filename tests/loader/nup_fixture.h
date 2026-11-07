// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/loader/nup.h>
#include "sha256.h"
#include "test_support.h"
#include <fstream>
#include <unistd.h>

namespace nup_fixture {
using namespace armigo::loader;
using Bytes = std::vector<std::byte>;
inline void put(Bytes& bytes, std::size_t pos, std::uint64_t value, unsigned count = 4) {
    for (unsigned i = 0; i < count; ++i) bytes[pos + i] = std::byte((value >> (8 * i)) & 255);
}
constexpr std::size_t toc = 0x80;
constexpr std::size_t package = toc + 0x30;
constexpr std::size_t directory = package + 2 * 0x40;
constexpr std::size_t file = directory + 6 * 0x18;
constexpr std::size_t child = file + 5 * 0x50;
constexpr std::size_t pool = child + 9 * 4;
constexpr std::size_t toc_end = pool + 40;
constexpr std::size_t blob = (toc_end + 16 + 15) & ~std::size_t{15};
inline void rehash(Bytes& bytes) {
    const auto hash = detail::sha256(std::span(bytes).subspan(toc, toc_end - toc));
    std::copy(hash->begin(), hash->end(), bytes.begin() + 0x18);
}
inline Bytes fixture(bool metadata_after = false) {
    Bytes bytes(blob + 5 * 16 + (metadata_after ? 12 : 0));
    put(bytes, 0, 0x0150554e); put(bytes, 4, 0x80, 2); put(bytes, 6, 1, 2);
    put(bytes, 8, toc, 8); put(bytes, 16, toc_end - toc, 8);
    put(bytes, 0x38, blob, 8); put(bytes, 0x40, bytes.size(), 8); put(bytes, 0x48, 2);
    put(bytes, toc, 0x01434f54); put(bytes, toc + 4, 2);
    put(bytes, toc + 8, 6, 8); put(bytes, toc + 16, 5, 8);
    put(bytes, toc + 24, pool - toc, 8); put(bytes, toc + 32, 40, 8);
    const char names[] = "rootexefsromfsmainoldkeepnew";
    std::memcpy(bytes.data() + pool, names, sizeof(names) - 1);
    for (unsigned i = 0; i < 2; ++i) {
        auto pos = package + i * 0x40;
        put(bytes, pos, 0x0400000000002000ull + i * 0x800, 8);
        put(bytes, pos + 8, i * 3); put(bytes, pos + 12, i * 3 + 1);
        put(bytes, pos + 16, i * 3 + 2); put(bytes, pos + 20, (i + 1) << 16);
        put(bytes, pos + 24, i, 1);
    }
    const auto meta = metadata_after ? blob + 5 * 16 : toc_end;
    put(bytes, package + 0x20, meta, 8); put(bytes, package + 0x28, 12);
    put(bytes, meta, 99, 2); put(bytes, meta + 4, 4); put(bytes, meta + 8, 1234);
    unsigned starts[] = {0, 2, 3, 5, 7, 8};
    unsigned counts[] = {2, 1, 2, 2, 1, 1};
    for (unsigned i = 0; i < 6; ++i) {
        auto pos = directory + i * 0x18;
        put(bytes, pos, (i % 3) == 0 ? 0 : (i % 3) == 1 ? 4 : 9);
        put(bytes, pos + 4, (i % 3) == 0 ? 4 : 5);
        put(bytes, pos + 8, starts[i]); put(bytes, pos + 12, counts[i]);
    }
    unsigned indices[] = {1, 2, 6, 7, 8, 4, 5, 9, 10};
    for (unsigned i = 0; i < 9; ++i) put(bytes, child + i * 4, indices[i]);
    unsigned name_offsets[] = {14, 18, 21, 14, 18};
    unsigned name_lengths[] = {4, 3, 4, 4, 3};
    for (unsigned i = 0; i < 5; ++i) {
        auto pos = file + i * 0x50;
        put(bytes, pos, name_offsets[i]); put(bytes, pos + 4, name_lengths[i]);
        put(bytes, pos + 8, 4, 8); put(bytes, pos + 16, blob + i * 16, 8);
        put(bytes, pos + 24, 4, 8); put(bytes, blob + i * 16, 0x11111111u * (i + 1));
        const auto hash = detail::sha256(std::span(bytes).subspan(blob + i * 16, 4));
        std::copy(hash->begin(), hash->end(), bytes.begin() + pos + 32);
    }
    rehash(bytes);
    return bytes;
}
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        auto name = (std::filesystem::temp_directory_path() / "armigo-nup-test-XXXXXX").string();
        const auto fd = mkstemp(name.data()); ARMIGO_CHECK(fd >= 0); close(fd); path = name;
    }
    ~Temporary() { std::filesystem::remove(path); }
    auto open(const Bytes& bytes) {
        { std::ofstream stream(path, std::ios::binary | std::ios::trunc);
          stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }
        return NupArchive::open(path);
    }
};

} // namespace nup_fixture
