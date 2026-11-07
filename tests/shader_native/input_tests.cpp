// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/input.h>
#include <zstd.h>
#define ZDICT_STATIC_LINKING_ONLY
#include <zdict.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

namespace sn = armigo::shader_native;
namespace {
void check(bool value) { if (!value) throw std::runtime_error("Synthetic input assertion failed"); }
template<class F> void rejects(F f) { bool rejected = false; try { f(); } catch (const sn::InputError&) { rejected = true; } check(rejected); }
void word(sn::Buffer& b, std::size_t at, std::uint64_t value, unsigned width = 4) {
    for (unsigned i = 0; i < width; ++i) b.at(at + i) = std::byte(value >> (i * 8));
}
void text(sn::Buffer& b, std::size_t at, const char* s, std::size_t count) { std::memcpy(b.data() + at, s, count); }
void control(sn::Buffer& b, std::size_t at, bool filled = true) {
    word(b, at, 0x98761234); word(b, at + 4, 0xc9); word(b, at + 8, 20);
    word(b, at + 0x30, filled ? 0xe0 : 0); word(b, at + 0x44, 0x100); word(b, at + 0xd4, 0xe0);
}
sn::Buffer bnsh() {
    sn::Buffer b(0x600);
    text(b, 0, "BNSH\0\0\0\0", 8); word(b, 0x1c, b.size()); text(b, 0x60, "grsc", 4);
    word(b, 0x7c, 3); word(b, 0x80, 0x90, 8);
    word(b, 0xa0, 0x180, 8); word(b, 0xe0, 0x180, 8);
    word(b, 0x188, 0x200, 8); word(b, 0x1a8, 0x200, 8);
    word(b, 0x208, 0x280, 8); word(b, 0x210, 0x400, 8);
    word(b, 0x218, 0x100); word(b, 0x21c, 0xe0); control(b, 0x280);
    return b;
}
sn::Buffer bahs() {
    sn::Buffer b(0x500);
    text(b, 0, "BAHS", 4); word(b, 4, 11); word(b, 8, b.size());
    word(b, 0x18, 0x100); word(b, 0x1c, 0x400); control(b, 0x100, false);
    for (auto at : {0x40u, 0x300u, 0x3f0u}) {
        word(b, at + 8, 0xe0); word(b, at + 12, std::uint32_t(0x100 - at));
    }
    return b;
}
sn::Buffer sarc(const sn::Buffer& payload) {
    sn::Buffer b(0x60 + payload.size());
    text(b, 0, "SARC", 4); word(b, 4, 0x14, 2); word(b, 6, 0xfeff, 2);
    word(b, 8, b.size()); word(b, 12, 0x60); word(b, 16, 0x100, 2);
    text(b, 0x14, "SFAT", 4); word(b, 0x18, 12, 2); word(b, 0x1a, 1, 2);
    word(b, 0x24, 0x01000000); word(b, 0x2c, payload.size());
    text(b, 0x30, "SFNT", 4); word(b, 0x34, 8, 2); text(b, 0x38, "item", 5);
    std::copy(payload.begin(), payload.end(), b.begin() + 0x60); return b;
}
sn::Buffer expand(sn::Bytes input, const sn::Dictionaries& d, std::size_t limit = 1 << 20, std::size_t stride = 7) {
    std::size_t offset = 0;
    return sn::read_zstandard([&](std::span<std::byte> output) {
        auto n = std::min({input.size() - offset, output.size(), stride});
        std::copy_n(input.begin() + offset, n, output.begin()); offset += n; return n;
    }, d, limit);
}
}
int main() {
    try {
        auto b = bnsh(); std::vector<sn::Record> records;
        sn::Enumerator e([&](const auto& r) { records.push_back(r); });
        e.scan(b, "root"); check(records.size() == 4); check(records[2].variation == 1); check(records[1].slot == 4);
        auto nested = sarc(sarc(b)); records.clear(); e.scan(nested, "root");
        check(records.size() == 4 && records[0].ancestry.size() == 3 && records[0].ancestry[1].offset == 0x60);
        auto embedded = b; embedded.insert(embedded.begin(), 43, std::byte{0x55}); records.clear(); e.scan(embedded, "effect");
        check(records.size() == 4 && records[0].container_offset == 43 && records[0].control_offset == 0x280);
        auto a = bahs(); records.clear(); e.scan(a, "archive"); check(records.size() == 1 && e.inventory().duplicate_candidates == 2);
        check(!records[0].variation && !records[0].slot);
        auto bad = b; word(bad, 0x80, ~std::uint64_t{}, 8); rejects([&]{ e.scan(bad, "bad"); });
        bad = b; word(bad, 0x1c, b.size() + 1); rejects([&]{ e.scan(bad, "bad"); });
        bad = b; word(bad, 0x21c, 0x10); rejects([&]{ e.scan(bad, "bad"); });
        bad = b; word(bad, 0x2b0, 0); rejects([&]{ e.scan(bad, "bad"); });
        bad = b; word(bad, 0x2c4, 0xff); rejects([&]{ e.scan(bad, "bad"); });
        bad = a; word(bad, 0x144, 0x101); rejects([&]{ e.scan(bad, "bad"); });
        bad = a; word(bad, 4, 12); rejects([&]{ e.scan(bad, "bad"); });
        sn::Limits limit; limit.records = 3; sn::Enumerator limited([](const auto&){}, limit); rejects([&]{ limited.scan(b, "limit"); });
        limit = {}; limit.depth = 1; sn::Enumerator shallow([](const auto&){}, limit); rejects([&]{ shallow.scan(nested, "limit"); });
        limit = {}; limit.variations = 2; sn::Enumerator few([](const auto&){}, limit); rejects([&]{ few.scan(b, "limit"); });
        limit = {}; limit.traversal_bytes = 100; sn::Enumerator small([](const auto&){}, limit); rejects([&]{ small.scan(b, "limit"); });
        check(sn::digest({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        sn::Buffer compressed(ZSTD_compressBound(b.size()));
        auto size = ZSTD_compress(compressed.data(), compressed.size(), b.data(), b.size(), 1); check(!ZSTD_isError(size)); compressed.resize(size);
        auto raw_read = [](sn::Bytes bytes, std::size_t ceiling, std::size_t stride) {
            std::size_t position = 0;
            return sn::read_raw_member([&](std::span<std::byte> output) {
                auto count = std::min({output.size(), bytes.size() - position, stride});
                std::copy_n(bytes.begin() + position, count, output.begin()); position += count; return count;
            }, bytes.size(), ceiling);
        };
        const auto compressed_archive = sarc(compressed);
        // A compressed shader is not discoverable through literal payload markers.
        check(std::search(compressed_archive.begin() + 0x60, compressed_archive.end(), b.begin(), b.begin() + 8) == compressed_archive.end());
        rejects([&]{ raw_read(compressed_archive, compressed_archive.size() - 1, 1); });
        auto malformed_archive = compressed_archive; word(malformed_archive, 8, 0xffffffff);
        rejects([&]{ raw_read(malformed_archive, malformed_archive.size() - 1, 3); });
        check(raw_read(b, b.size(), 5) == b);
        rejects([&]{ raw_read(b, b.size() - 1, 2); });
        sn::Buffer media(100, std::byte{0x55}); check(raw_read(media, 10, 3).empty());
        rejects([&]{ sn::read_raw_member([](std::span<std::byte>) { return std::size_t{0}; }, 100, 10); });
        check(expand(compressed, {}) == b); rejects([&]{ expand(compressed, {}, b.size() - 1); });
        auto truncated = compressed; truncated.pop_back(); rejects([&]{ expand(truncated, {}); });
        auto trailing = compressed; trailing.push_back(std::byte{}); rejects([&]{ expand(trailing, {}); });
        auto joined = compressed; joined.insert(joined.end(), compressed.begin(), compressed.end()); rejects([&]{ expand(joined, {}); });
        // Build an identified dictionary solely from deterministic synthetic samples.
        sn::Buffer samples(128 * 256), dictionary(2048), content(1024);
        std::array<std::size_t, 128> sizes{}; sizes.fill(256);
        for (std::size_t i = 0; i < samples.size(); ++i) samples[i] = std::byte((i * 17 + i / 256) % 251);
        std::copy_n(samples.begin(), content.size(), content.begin());
        ZDICT_params_t params{}; params.dictID = 1;
        auto ds = ZDICT_finalizeDictionary(dictionary.data(), dictionary.size(), content.data(), content.size(), samples.data(), sizes.data(), sizes.size(), params);
        check(!ZDICT_isError(ds)); dictionary.resize(ds);
        auto dictionaries = sn::read_dictionaries(sarc(dictionary), 1 << 20); check(dictionaries.contains(1));
        auto ctx = ZSTD_createCCtx(); compressed.resize(ZSTD_compressBound(b.size()));
        size = ZSTD_compress_usingDict(ctx, compressed.data(), compressed.size(), b.data(), b.size(), dictionary.data(), dictionary.size(), 3);
        ZSTD_freeCCtx(ctx); check(!ZSTD_isError(size)); compressed.resize(size);
        check(ZSTD_getDictID_fromFrame(compressed.data(), compressed.size()) == 1);
        check(expand(compressed, dictionaries) == b);
        std::size_t nested_records = 0;
        sn::Enumerator compressed_nested([&](const auto&) { ++nested_records; }, {}, dictionaries);
        compressed_nested.scan(sarc(compressed), "compressed-member"); check(nested_records == 4);
        rejects([&]{ expand(compressed, {}); });
        rejects([&]{ sn::read_dictionaries(sarc(dictionary), 4); });
        rejects([&]{ sn::read_dictionaries(sarc(b), 1 << 20); });
        std::cout << "Native input synthetic contracts passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
