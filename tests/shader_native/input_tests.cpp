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
        // Task and mesh stages at +0x40 and +0x48 of a program block count
        // only when they lead to a control section of that stage.
        {
            auto m = bnsh();
            word(m, 0x180 + 0x40, 0x300, 8); word(m, 0x180 + 0x48, 0x340, 8);
            word(m, 0x308, 0x380, 8); word(m, 0x310, 0x400, 8); word(m, 0x318, 0x100); word(m, 0x31c, 0xe0);
            word(m, 0x348, 0x500, 8); word(m, 0x350, 0x400, 8); word(m, 0x358, 0x100); word(m, 0x35c, 0xe0);
            control(m, 0x380); m[0x380 + 0x56] = std::byte{6};
            control(m, 0x500); m[0x500 + 0x56] = std::byte{7};
            records.clear(); e.scan(m, "mesh");
            check(records.size() == 8 && records[2].slot == 7 && records[3].slot == 8 && records[2].control_offset == 0x380 &&
                  records[3].control_offset == 0x500);
            m[0x500 + 0x56] = std::byte{6}; records.clear(); e.scan(m, "mesh");
            check(records.size() == 6);  // a task control in the mesh slot is not a mesh stage
            word(m, 0x180 + 0x48, 0x5000, 8); records.clear(); e.scan(m, "mesh");
            check(records.size() == 6);  // a pointer outside the container is not followed
        }
        auto nested = sarc(sarc(b)); records.clear(); e.scan(nested, "root");
        check(records.size() == 4 && records[0].ancestry.size() == 3 && records[0].ancestry[1].offset == 0x60);
        auto embedded = b; embedded.insert(embedded.begin(), 43, std::byte{0x55}); records.clear(); e.scan(embedded, "effect");
        check(records.size() == 4 && records[0].container_offset == 43 && records[0].control_offset == 0x280);
        // Directory program occupies file 0x48..0x228. The inline header starts at that end.
        sn::Buffer video(0x420);
        text(video, 0, "NDVE", 4); word(video, 0x0c, 0); word(video, 0x10, 0x3f8); word(video, 0x14, 2);
        word(video, 0x28, 1); word(video, 0x2c, 1); word(video, 0x30, 0x20);
        word(video, 0x38, 0x20); word(video, 0x3c, 0xe0); word(video, 0x40, 0x100);
        control(video, 0x48);
        word(video, 0x228, 1); word(video, 0x230, 0x18); word(video, 0x234, 0xe0); word(video, 0x238, 0x100);
        control(video, 0x240);
        records.clear();
        const auto unrecognized = e.inventory().unrecognized_spans;
        e.scan(video, "sequence");
        check(records.size() == 2 && records[0].control.size() == 0xe0 && records[0].code.size() == 0x100);
        check(records[1].code.size() == 0x100 && e.inventory().unrecognized_spans == unrecognized);
        // An NDSR pack: an 8-byte data entry, then one version-17 compute record.
        sn::Buffer pack(0x208);
        text(pack, 0, "NDSR", 4);
        word(pack, 0x10, 0); word(pack, 0x14, 1); word(pack, 0x18, 0x28);
        word(pack, 0x1c, 0xe0u << 4); word(pack, 0x20, 0x108); word(pack, 0x24, 0);
        control(pack, 0x28); word(pack, 0x28 + 8, 17); pack[0x28 + 0x56] = std::byte{5};
        word(pack, 0x28 + 0x30, 0); word(pack, 0x28 + 0xd4, 0);
        records.clear(); e.scan(pack, "pack");
        check(records.size() == 1 && records[0].kind == sn::ContainerKind::runtime &&
              records[0].control.size() == 0xe0 && records[0].code.size() == 0x100 &&
              records[0].control[0x56] == std::byte{5} && records[0].control_offset == 0x28);
        // The same pack in an NDVE prefix, ahead of one network program.
        constexpr std::size_t pack_size = 0x208, section_size = 0x1f8;
        sn::Buffer wrapped(0x28 + pack_size + section_size);
        text(wrapped, 0, "NDVE", 4); word(wrapped, 0x0c, pack_size); word(wrapped, 0x10, section_size);
        word(wrapped, 0x14, 2);
        std::copy(pack.begin(), pack.end(), wrapped.begin() + 0x28);
        const auto section = 0x28 + pack_size;
        word(wrapped, section, 1); word(wrapped, section + 8, 0x18);
        word(wrapped, section + 0x0c, 0xe0); word(wrapped, section + 0x10, 0x100);
        control(wrapped, section + 0x18);
        records.clear(); e.scan(wrapped, "sequence-pack");
        check(records.size() == 2 && records[0].kind == sn::ContainerKind::runtime &&
              records[0].control[0x56] == std::byte{5} && records[1].control_offset == section + 0x18);
        // One model record puts the pack at +0x20.
        sn::Buffer single(0x20 + pack_size + section_size);
        text(single, 0, "NDVE", 4); word(single, 0x0c, pack_size); word(single, 0x10, section_size);
        word(single, 0x14, 1);
        std::copy(pack.begin(), pack.end(), single.begin() + 0x20);
        const auto single_section = 0x20 + pack_size;
        word(single, single_section, 1); word(single, single_section + 8, 0x18);
        word(single, single_section + 0x0c, 0xe0); word(single, single_section + 0x10, 0x100);
        control(single, single_section + 0x18);
        records.clear(); e.scan(single, "sequence-single");
        check(records.size() == 2 && records[0].kind == sn::ContainerKind::runtime &&
              records[0].control_offset == 0x20 + 0x28 && records[1].control_offset == single_section + 0x18);
        auto broken = video; word(broken, 0x40, 0x10); rejects([&]{ e.scan(broken, "bad"); });
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
