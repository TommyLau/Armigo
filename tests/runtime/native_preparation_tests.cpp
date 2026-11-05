// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../../src/runtime/native_preparation.h"
#include "preparation_fixture.h"
#include <armigo/runtime/launch_plan.h>
#include <armigo/execution/embedded_patch.h>

#include <armigo/aarch64/instruction.h>
#include <armigo/execution/gate_code.h>
#include <armigo/execution/indirect_x18_patch.h>
#include <armigo/linker/dynamic_module.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using armigo::runtime::native::classify_word;
using armigo::runtime::native::ModuleCodePreparation;

int failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)

bool no_role(std::uint32_t word) {
    const auto role = classify_word(word);
    return !role.uses_x18 && !role.system && !role.unknown_control && !role.exclusive_load &&
           !role.exclusive_store && !role.clear_exclusive;
}

// Host-sensitive system instructions go through a gate; the guest's own flag
// and floating-point control registers, hints and barriers run natively.
void test_system_roles() {
    CHECK(classify_word(0xd53bd068u).system);  // mrs x8, tpidrro_el0
    CHECK(classify_word(0xd51bd048u).system);  // msr tpidr_el0, x8
    CHECK(classify_word(0xd4000021u).system);  // svc #0x1
    CHECK(classify_word(0xd50b7420u).system);  // dc zva, x0
    CHECK(classify_word(0xd50343dfu).system);  // msr daifset, #3
    CHECK(no_role(0xd53b4200u));               // mrs x0, nzcv
    CHECK(no_role(0xd53b4400u));               // mrs x0, fpcr
    CHECK(no_role(0xd51b4420u));               // msr fpsr, x0
    CHECK(no_role(0xd503201fu));               // nop
    CHECK(no_role(0xd5033bbfu));               // dmb ish
}

// Any read or write of x18 needs the virtual register, including branches.
void test_x18_roles() {
    CHECK(classify_word(0x91000640u).uses_x18);   // add x0, x18, #1
    CHECK(classify_word(0xaa0003f2u).uses_x18);   // mov x18, x0
    CHECK(classify_word(0xd63f0240u).uses_x18);   // blr x18
    CHECK(classify_word(0xb4000012u).uses_x18);   // cbz x18, .
    CHECK(no_role(0xf9400020u));                  // ldr x0, [x1]
    CHECK(no_role(0xd63f0100u));                  // blr x8
    CHECK(no_role(0xd65f03c0u));                  // ret
}

void test_exclusive_roles() {
    CHECK(classify_word(0x885ffc20u).exclusive_load);   // ldaxr w0, [x1]
    CHECK(classify_word(0xc85f7c20u).exclusive_load);   // ldxr x0, [x1]
    CHECK(classify_word(0x8802fc20u).exclusive_store);  // stlxr w2, w0, [x1]
    CHECK(classify_word(0xd503305fu).clear_exclusive);  // clrex
}

// Branch and exception words the decoder does not identify must stop.
void test_unknown_control() {
    CHECK(classify_word(0xd71f0801u).unknown_control);  // braa x0, x1
    CHECK(classify_word(0xd69f03e0u).unknown_control);  // eret
    CHECK(!classify_word(0xd4200000u).unknown_control); // brk #0
}

// A module's text with every role in it, dense enough that function starts fall
// inside and outside exclusive windows, and long enough to split across threads.
std::vector<std::byte> role_text(std::size_t words) {
    static constexpr std::uint32_t special[] = {
        0xd53bd068u,  // mrs x8, tpidrro_el0
        0xd4000021u,  // svc #0x1
        0x91000640u,  // add x0, x18, #1
        0xb4000012u,  // cbz x18, .
        0x885ffc20u,  // ldaxr w0, [x1]
        0x8802fc20u,  // stlxr w2, w0, [x1]
        0xd503305fu,  // clrex
        0xd69f03e0u,  // eret
        0xaa0003f2u,  // mov x18, x0
        0xd51bd048u,  // msr tpidr_el0, x8
    };
    std::vector<std::byte> text(words * 4);
    std::uint32_t state = 12345;
    for (std::size_t i = 0; i < words; ++i) {
        state = state * 1664525u + 1013904223u;
        const auto pick = (state >> 16) % 64;
        const std::uint32_t word = pick < std::size(special) ? special[pick] : 0xd503201fu;  // nop
        std::memcpy(text.data() + i * 4, &word, 4);
    }
    return text;
}

struct Prepared {
    std::vector<std::byte> image, appended;
    ModuleCodePreparation module;
};

// `text` is the module image; its first `text_bytes` bytes (all, when zero)
// are its text and any rest is data.
Prepared prepare_text(const std::vector<std::byte>& text, bool scan, unsigned threads,
                      bool embedded = false, std::uint64_t base = 0x100000000ull, std::size_t text_bytes = 0,
                      std::vector<std::uint64_t> launch_words = {}) {
    Prepared result;
    result.module.slot = 0x10;
    result.module.embedded = embedded;
    result.module.launch_words = launch_words;
    result.module.gate = armigo::execution::make_local_gate_code();
    result.module.text_begin = 0;
    result.module.text_size = text_bytes ? text_bytes : text.size();
    const auto code = std::span(text).first(result.module.text_size);
    const auto bound = (scan ? result.module.scan_text(code, threads, text) : text.size() * (embedded ? 16 : 12) + 16384) +
                       launch_words.size() * 16;
    result.image = text;
    result.appended.assign((bound + 15) & ~std::size_t{15}, std::byte{0});
    // Function starts every 1021 words, so some fall between a load and its store.
    armigo::linker::DynamicModule metadata;
    for (std::size_t word = 0; word < result.module.text_size / 4; word += 1021) {
        armigo::linker::Symbol symbol;
        symbol.name = "f" + std::to_string(word);
        symbol.value = word * 4;
        symbol.info = 0x12;
        symbol.section = 1;
        metadata.symbols.push_back(std::move(symbol));
    }
    const std::uintptr_t bases[] = {base};
    std::vector<std::byte> key(embedded ? 16384 : 0);
    const auto key_offset = embedded ? (text.size() + 16383) & ~std::size_t{16383} : 0;
    const auto code_offset = embedded ? key_offset + key.size() : text.size();
    const armigo::runtime::ModulePreparationView view{
        0, "synthetic", base, result.image, result.appended, code_offset, metadata, bases,
        key, key_offset};
    result.module.prepare(view);
    return result;
}

// Scanning the text first, on any number of threads, must prepare the module
// exactly as classifying every word during preparation does.
void test_scanned_preparation_matches() {
    const auto text = role_text(300000);
    const auto reference = prepare_text(text, false, 1);
    CHECK(!reference.module.sites.empty() && !reference.module.stops.empty() &&
          reference.module.scalar_sites > 0 && reference.module.thread_register_sites > 0);
    for (const unsigned threads : {1u, 3u, 8u}) {
        const auto scanned = prepare_text(text, true, threads);
        CHECK(scanned.image == reference.image);
        CHECK(std::equal(scanned.appended.begin(), scanned.appended.begin() + reference.module.code_bytes,
                         reference.appended.begin()));
        CHECK(scanned.module.code_bytes == reference.module.code_bytes);
        CHECK(scanned.module.scalar_sites == reference.module.scalar_sites);
        CHECK(scanned.module.thread_register_sites == reference.module.thread_register_sites);
        CHECK(scanned.module.prepared_entries == reference.module.prepared_entries);
        CHECK(scanned.module.sites.size() == reference.module.sites.size());
        for (std::size_t i = 0; i < reference.module.sites.size() && i < scanned.module.sites.size(); ++i)
            CHECK(scanned.module.sites[i].return_address == reference.module.sites[i].return_address &&
                  scanned.module.sites[i].instruction == reference.module.sites[i].instruction);
        CHECK(scanned.module.stops.size() == reference.module.stops.size());
        for (std::size_t i = 0; i < reference.module.stops.size() && i < scanned.module.stops.size(); ++i)
            CHECK(scanned.module.stops[i].address == reference.module.stops[i].address &&
                  scanned.module.stops[i].word == reference.module.stops[i].word &&
                  scanned.module.stops[i].reason == reference.module.stops[i].reason);
    }
}

void test_embedded_preparation_rebases() {
    const auto text = role_text(4096);
    const auto ordinary = prepare_text(text, true, 1);
    const auto reference = prepare_text(text, false, 1, true);
    CHECK(reference.module.completed);
    CHECK(reference.module.scalar_sites == ordinary.module.scalar_sites);
    CHECK(reference.module.thread_register_sites == ordinary.module.thread_register_sites);
    CHECK(reference.module.stops.size() == ordinary.module.stops.size());
    CHECK(reference.module.sites.size() == ordinary.module.sites.size());
    for (unsigned threads : {1u, 3u}) {
        const auto prepared = prepare_text(text, true, threads, true, 0x200000000ull);
        CHECK(prepared.module.completed);
        CHECK(prepared.image == reference.image);
        CHECK(prepared.module.code_bytes == reference.module.code_bytes);
        CHECK(std::equal(prepared.appended.begin(), prepared.appended.begin() + prepared.module.code_bytes,
                         reference.appended.begin()));
        CHECK(prepared.module.prepared_entries.size() == reference.module.prepared_entries.size());
        for (std::size_t i = 0; i < prepared.module.sites.size(); ++i)
            CHECK(prepared.module.sites[i].return_address - 0x100000000ull ==
                  reference.module.sites[i].return_address);
    }
}

// Launch-time words become gate sites in text order: each site keeps its
// original MOVZ/MOVK for the launch to replace, and its text word branches to
// the trampoline the gate returns into. Scanned and unscanned preparation agree.
void test_launch_words() {
    std::vector<std::byte> text(64 * 4);
    const auto put = [&](std::size_t index, std::uint32_t word) { std::memcpy(text.data() + index * 4, &word, 4); };
    for (std::size_t i = 0; i < 64; ++i) put(i, 0xd503201fu);  // nop
    put(4, 0xd2824683u);   // movz x3, #0x1234
    put(10, 0x72a00204u);  // movk w4, #0x10, lsl #16
    put(20, 0x885ffc20u);  // ldaxr w0, [x1]
    put(21, 0xd28000a5u);  // movz x5, #0x5
    put(22, 0x8802fc20u);  // stlxr w2, w0, [x1]
    put(30, 0xd2800032u);  // movz x18, #0x1
    const std::uint64_t base = 0x100000000ull;
    for (const bool scan : {false, true}) {
        const auto prepared = prepare_text(text, scan, 1, true, base, 0, {40, 16});
        CHECK(prepared.module.completed && prepared.module.sites.size() == 2);
        if (prepared.module.sites.size() != 2) continue;
        CHECK(prepared.module.sites[0].instruction == 0xd2824683u && prepared.module.sites[1].instruction == 0x72a00204u);
        for (const auto& [offset, site] : {std::pair{16u, 0u}, std::pair{40u, 1u}}) {
            std::uint32_t branch;
            std::memcpy(&branch, prepared.image.data() + offset, 4);
            const auto delta = std::int64_t{static_cast<std::int32_t>(branch << 6) >> 6} * 4;
            CHECK((branch & 0xfc000000u) == 0x14000000u);
            CHECK(base + offset + static_cast<std::uint64_t>(delta) + 8 == prepared.module.sites[site].return_address);
        }
    }
    const auto refused = [&](std::vector<std::uint64_t> words, bool embedded) {
        try { prepare_text(text, true, 1, embedded, base, 0, std::move(words)); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    CHECK(refused({84}, true));      // inside the exclusive window
    CHECK(refused({0}, true));       // not a MOVZ/MOVK
    CHECK(refused({120}, true));     // writes x18
    CHECK(refused({16, 16}, true));  // listed twice
    CHECK(refused({256}, true));     // outside text
    CHECK(refused({16}, false));     // ordinary preparation patches words directly
}

// A switch through virtual x18 on a byte index (ADR X18, table; LDRB; ADD X18,
// X18, Xn, LSL #2; at most one word that leaves x18 alone; BR X18) becomes a
// bounded table trampoline. A BR X18 without that shape stops.
void test_byte_indexed_x18_switch() {
    // The 256-word table from word 16 lies inside the text, before the appendix.
    std::vector<std::uint32_t> words(512, 0xd503201fu);  // nop
    const std::uint32_t code[] = {
        0x10000212u,  // adr x18, word 16
        0x39400009u,  // ldrb w9, [x0]
        0x8b090a52u,  // add x18, x18, x9, lsl #2
        0xd61f0240u,  // br x18
        0x10000192u,  // adr x18, word 16
        0x38696809u,  // ldrb w9, [x0, x9]
        0x8b090a52u,  // add x18, x18, x9, lsl #2
        0xaa0203e1u,  // mov x1, x2
        0xd61f0240u,  // br x18
        0xaa0003f2u,  // mov x18, x0
        0xd61f0240u,  // br x18: no table
    };
    std::copy(std::begin(code), std::end(code), words.begin());
    std::vector<std::byte> text(words.size() * 4);
    std::memcpy(text.data(), words.data(), text.size());
    constexpr std::uint64_t base = 0x100000000ull;
    for (const bool scan : {false, true}) {
        const auto prepared = prepare_text(text, scan, 1);
        CHECK(prepared.module.stops.size() == 1);
        if (prepared.module.stops.size() == 1)
            CHECK(prepared.module.stops[0].address == base + 40 &&
                  std::string_view(prepared.module.stops[0].reason) ==
                      "Indirect branch on virtual x18 without a recognised table.");
        for (const std::size_t site : {3u, 8u}) {
            std::uint32_t replaced = 0;
            std::memcpy(&replaced, prepared.image.data() + site * 4, 4);
            const auto branch = armigo::aarch64::decode(replaced);
            CHECK(branch.opcode == armigo::aarch64::Opcode::b);
            if (branch.opcode != armigo::aarch64::Opcode::b) continue;
            const auto target = base + site * 4 + static_cast<std::uint64_t>(branch.displacement);
            const auto at = static_cast<std::size_t>(target - base - text.size());
            CHECK(at + 4 * armigo::execution::indirect_x18_table_words(256) <= prepared.appended.size());
            std::array<std::uint32_t, 8> head{};
            std::memcpy(head.data(), prepared.appended.data() + at, sizeof head);
            // MOVZ/MOVK X1 build the table base: the module base plus 64.
            std::uint64_t table = 0;
            for (unsigned half = 0; half != 4; ++half)
                table |= std::uint64_t{(head[4 + half] >> 5) & 0xffffu} << (16 * half);
            CHECK(table == base + 64);
            // The bound check shifts by log2 of the 1024-byte table: LSR X1, X0, #10.
            std::uint32_t bound = 0;
            std::memcpy(&bound, prepared.appended.data() + at + 4 * 9, 4);
            CHECK(bound == 0xd34afc01u);
            // The first entry restores the scratch pair and branches to the table's first word.
            std::uint32_t entry = 0;
            std::memcpy(&entry, prepared.appended.data() + at + 4 * 17, 4);
            const auto first = armigo::aarch64::decode(entry);
            CHECK(first.opcode == armigo::aarch64::Opcode::b &&
                  target + 4 * 17 + static_cast<std::uint64_t>(first.displacement) == base + 64);
        }
    }
}

// A switch through virtual x18 on a halfword index read from a bounded table
// (CMP; B.HI; ADRP/ADD of the table; ADR X18; LDRH; ADD X18, X18, Xn, LSL #2;
// BR X18) reaches the words up to its largest entry, rounded up to a power of
// two; the scan reserves that trampoline. The same shape without the bound
// check stops.
void test_halfword_indexed_x18_switch() {
    // Text: words 0-511; the table of three halfwords at 0x800, in data after the text.
    std::vector<std::uint32_t> words(2048, 0xd503201fu);  // nop
    const std::uint32_t code[] = {
        0x71000a3fu,  // cmp w17, #2
        0x540004e8u,  // b.hi word 40
        0x90000001u,  // adrp x1, page 0
        0x91200021u,  // add x1, x1, #0x800
        0x10000192u,  // adr x18, word 16
        0x78717820u,  // ldrh w0, [x1, x17, lsl #1]
        0x8b000a52u,  // add x18, x18, x0, lsl #2
        0xd61f0240u,  // br x18
    };
    std::copy(std::begin(code), std::end(code), words.begin());
    // Words 100-107: the same shape without the CMP/B.HI bound (two NOPs instead).
    std::copy(std::begin(code), std::end(code), words.begin() + 100);
    words[100] = words[101] = 0xd503201fu;
    std::vector<std::byte> image(words.size() * 4);
    std::memcpy(image.data(), words.data(), image.size());
    const std::uint16_t table[] = {0, 300, 7};
    std::memcpy(image.data() + 0x800, table, sizeof table);
    constexpr std::uint64_t base = 0x100000000ull;
    constexpr std::size_t text_bytes = 512 * 4;
    for (const bool scan : {false, true}) {
        const auto prepared = prepare_text(image, scan, 1, false, base, text_bytes);
        CHECK(prepared.module.stops.size() == 1);
        if (prepared.module.stops.size() == 1)
            CHECK(prepared.module.stops[0].address == base + 107 * 4 &&
                  std::string_view(prepared.module.stops[0].reason) ==
                      "Indirect branch on virtual x18 without a recognised table.");
        std::uint32_t replaced = 0;
        std::memcpy(&replaced, prepared.image.data() + 7 * 4, 4);
        const auto branch = armigo::aarch64::decode(replaced);
        CHECK(branch.opcode == armigo::aarch64::Opcode::b);
        if (branch.opcode != armigo::aarch64::Opcode::b) continue;
        const auto target = base + 7 * 4 + static_cast<std::uint64_t>(branch.displacement);
        const auto at = static_cast<std::size_t>(target - base - image.size());
        // The largest entry, 300, needs 301 words: a 512-entry trampoline.
        CHECK(at + 4 * armigo::execution::indirect_x18_table_words(512) <= prepared.appended.size());
        std::array<std::uint32_t, 10> head{};
        std::memcpy(head.data(), prepared.appended.data() + at, sizeof head);
        std::uint64_t table_base = 0;
        for (unsigned half = 0; half != 4; ++half)
            table_base |= std::uint64_t{(head[4 + half] >> 5) & 0xffffu} << (16 * half);
        CHECK(table_base == base + 64);
        // The bound check shifts by log2 of the 2048-byte range: LSR X1, X0, #11.
        CHECK(head[9] == 0xd34bfc01u);
        // The entry for word 300 branches to base + 64 + 1200.
        std::uint32_t entry = 0;
        const auto entry_at = at + 4 * (16 + 2 * 300 + 1);
        std::memcpy(&entry, prepared.appended.data() + entry_at, 4);
        const auto decoded = armigo::aarch64::decode(entry);
        CHECK(decoded.opcode == armigo::aarch64::Opcode::b &&
              target + (entry_at - at) + static_cast<std::uint64_t>(decoded.displacement) == base + 64 + 1200);
    }
    // The scan sizes the trampoline from the table: a table whose largest entry
    // is 10 needs no more than the 256 entries every table is given, one whose
    // largest is 300 needs 512.
    {
        auto small = image;
        const std::uint16_t small_table[] = {0, 10, 7};
        std::memcpy(small.data() + 0x800, small_table, sizeof small_table);
        const auto text_of = [&](const std::vector<std::byte>& bytes) { return std::span(bytes).first(text_bytes); };
        ModuleCodePreparation large_scan, small_scan;
        large_scan.text_size = small_scan.text_size = text_bytes;
        const auto large_bound = large_scan.scan_text(text_of(image), 1, image);
        const auto small_bound = small_scan.scan_text(text_of(small), 1, small);
        CHECK(large_bound - small_bound == 4 * (armigo::execution::indirect_x18_table_words(512) -
                                                 armigo::execution::indirect_x18_table_words(256)));
    }
    // B.HS bounds n entries, CMP may compare the 64-bit register, and LDRH may
    // extend a 32-bit index (UXTW #1).
    {
        auto variant = image;
        std::vector<std::uint32_t> shape = {
            0xf1000e3fu,  // cmp x17, #3
            0x540004e2u,  // b.hs word 40
            0x90000001u,  // adrp x1, page 0
            0x91204021u,  // add x1, x1, #0x810
            0x10000192u,  // adr x18, word 16
            0x78715820u,  // ldrh w0, [x1, w17, uxtw #1]
            0x8b000a52u,  // add x18, x18, x0, lsl #2
            0xd61f0240u,  // br x18
        };
        std::memcpy(variant.data(), shape.data(), shape.size() * 4);
        const std::uint16_t table[] = {0, 2, 600, 1};  // three entries bound: 0, 2, 600 (the fourth is past the bound)
        std::memcpy(variant.data() + 0x810, table, sizeof table);
        const auto prepared = prepare_text(variant, true, 1, false, base, text_bytes);
        CHECK(prepared.module.stops.size() == 1);  // only the unbounded shape at word 107
        std::uint32_t replaced = 0;
        std::memcpy(&replaced, prepared.image.data() + 7 * 4, 4);
        const auto branch = armigo::aarch64::decode(replaced);
        CHECK(branch.opcode == armigo::aarch64::Opcode::b);
        if (branch.opcode == armigo::aarch64::Opcode::b) {
            const auto target = base + 7 * 4 + static_cast<std::uint64_t>(branch.displacement);
            const auto at = static_cast<std::size_t>(target - base - variant.size());
            std::uint32_t bound = 0;
            std::memcpy(&bound, prepared.appended.data() + at + 4 * 9, 4);
            // Largest bound entry 600: 1024 entries, a 4096-byte range: LSR X1, X0, #12.
            CHECK(bound == 0xd34cfc01u);
        }
    }
}

// The scan's appendix bound covers what preparation emits, and is the same on
// any number of threads.
void test_scan_bound() {
    const auto text = role_text(300000);
    ModuleCodePreparation one, many;
    one.gate = many.gate = armigo::execution::make_local_gate_code();
    const auto serial = one.scan_text(text, 1, text);
    CHECK(serial == many.scan_text(text, 8, text));
    CHECK(serial >= prepare_text(text, false, 1).module.code_bytes);
}

// A scan describes the text it saw; preparation refuses a different one.
void test_changed_text_is_refused() {
    auto text = role_text(1000);
    ModuleCodePreparation module;
    module.slot = 0x10;
    module.gate = armigo::execution::make_local_gate_code();
    module.text_size = text.size();
    const auto bound = module.scan_text(text, 2, text);
    std::vector<std::byte> image = text, appended((bound + 15) & ~std::size_t{15});
    // Turn a word with a role into a different one.
    for (std::size_t i = 0; i < text.size(); i += 4) {
        std::uint32_t word;
        std::memcpy(&word, image.data() + i, 4);
        if (word == 0xd4000021u) {
            const std::uint32_t other = 0xd4000041u;
            std::memcpy(image.data() + i, &other, 4);
            break;
        }
    }
    armigo::linker::DynamicModule metadata;
    const std::uintptr_t bases[] = {0x100000000ull};
    const armigo::runtime::ModulePreparationView view{
        0, "synthetic", 0x100000000ull, image, appended, text.size(), metadata, bases};
    bool refused = false;
    try { module.prepare(view); } catch (const std::runtime_error&) { refused = true; }
    CHECK(refused);
}

// A linked function slot left zero for an undefined weak import is no root;
// a zero slot for any other symbol is refused.
void test_weak_null_import() {
    std::vector<std::byte> image(0x200), appended;
    armigo::linker::DynamicModule metadata;
    metadata.symbols.resize(3);
    metadata.symbols[1].name = "optional";
    metadata.symbols[1].info = 0x22;  // weak function, undefined
    metadata.symbols[2].name = "required";
    metadata.symbols[2].info = 0x12;  // global function, undefined
    metadata.relocations.push_back({0x100, armigo::linker::RelocationKind::jump_slot, 1, 0});
    const std::uintptr_t bases[] = {0x100000000ull};
    const armigo::runtime::ModulePreparationView view{
        0, "synthetic", 0x100000000ull, image, appended, image.size(), metadata, bases};
    bool accepted = true;
    armigo::runtime::native::ModuleRoots roots;
    try { roots = armigo::runtime::native::module_roots(view, 0, 0x100, std::nullopt); }
    catch (const std::runtime_error&) { accepted = false; }
    CHECK(accepted && roots.required_foreign_entries.empty() && roots.local.empty());
    metadata.relocations.front().symbol = 2;
    bool refused = false;
    try { static_cast<void>(armigo::runtime::native::module_roots(view, 0, 0x100, std::nullopt)); }
    catch (const std::runtime_error&) { refused = true; }
    CHECK(refused);
}

void test_embedded_native_launch() {
    using namespace armigo;
    using namespace armigo::runtime;
    test::ModuleSpec main;
    main.name = "main";
    const std::array names{"_init_libc0", "nnosInitialize", "_init_libc1", "nninitInitializeSdkModule",
        "nninitStartup", "_init_libc2", "_ZN2nn2os11SetHostArgcEi", "_ZN2nn2os11SetHostArgvEPPc", "nnMain"};
    for (const auto name : names)
        main.symbols.push_back({name, std::string_view(name) == "nnMain" ? 0x200u : 0x100u, 4, true});
    main.relocations.push_back({0x8000, 9, 257, 0});
    main.symbols.push_back({"nvnBootstrapLoader", 0, 0, false});
    main.relocations.push_back({0x8010, 10, 257, 0});
    test::put_word(main, 0x100, aarch64::encode_ret().word);
    test::put_word(main, 0x200, 0xaa0003f2); // MOV X18,X0
    test::put_word(main, 0x204, 0x91001e52); // ADD X18,X18,#7
    test::put_word(main, 0x208, 0xaa1203e0); // MOV X0,X18
    test::put_word(main, 0x20c, aarch64::encode_ret().word);
    test::put_word(main, 0x300, 0xd2800020); // MOV X0,#1
    const auto call = aarch64::encode_bl(0x400, 0x500).word;
    test::put_word(main, 0x400, call);
    test::put_word(main, 0x500, aarch64::encode_ret().word);
    // Resolution sites at the console's 1600x900; the function at 0x600
    // returns the swapchain size, width | height << 32.
    main.symbols.push_back({"resolution_probe", 0x600, 0x14, true});
    for (const auto& [offset, word] : std::initializer_list<std::pair<std::uint64_t, std::uint32_t>>{
             {0x600, 0xd280c800}, {0x604, 0xd503201f}, {0x608, 0xd503201f}, {0x60c, 0xf2c07080}, {0x610, 0xd65f03c0},
             {0x620, 0x52a89901}, {0x624, 0xf2e89902}, {0x628, 0xd280c803}, {0x62c, 0xf2c07083}, {0x630, 0x5280c804},
             {0x634, 0x52807085}, {0x638, 0x529ae146}, {0x63c, 0x72a7e466}, {0x640, 0x52a14007}, {0x644, 0x52a04008},
             {0x648, 0x52b80009}, {0x64c, 0x52b7e00a}, {0x650, 0x52a8600b}, {0x654, 0xd65f03c0}})
        test::put_word(main, offset, word);
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", test::build_module(std::move(main))});
    test::ModuleSpec rtld;
    rtld.name = "rtld";
    input.modules.push_back({"rtld", test::build_module(std::move(rtld))});
    test::ModuleSpec sdk;
    sdk.name = "sdk";
    sdk.symbols = {{"nvnBootstrapLoader", 0x100, 20, true}};
    for (std::size_t at = 0x100; at < 0x110; at += 4) test::put_word(sdk, at, 0xd503201f);
    test::put_word(sdk, 0x110, aarch64::encode_ret().word);
    constexpr std::string_view version = "NintendoSDK_NVN-20_5_22-Release";
    std::memcpy(sdk.rodata.data() + 0x2000, version.data(), version.size());
    input.modules.push_back({"sdk", test::build_module(std::move(sdk))});
    const auto source = input.modules[0].bytes;
    const auto decoded = loader::read_nso_module(source);
    CHECK(decoded.ok());
    if (!decoded.ok()) return;
    compat::PatchDescription patch;
    patch.identity = {1, decoded.image->build_id};
    patch.module = "frame-rate";
    patch.nvn_driver = true;
    patch.original_callee = 0x500;
    patch.calls.push_back({0x400, call});
    patch.layout.framework_pointer = 0x8008;
    patch.words.push_back({0x300, 0xd2800020, 0xd2800040, false});
    compat::ResolutionSites resolution;
    resolution.tv_size = {{0x600, 0x60c}};
    resolution.screen_widths = {0x620, 0x624};
    resolution.display_size = {0x628, 0x62c};
    resolution.window_crop = {0x630, 0x634};
    resolution.floor = {0x638, 0x63c};
    resolution.render_target_pool = 0x640;
    resolution.graphics_heap = 0x644;
    resolution.process_heap = 0x648;
    resolution.root_heap = 0x64c;
    resolution.load_scale = 0x650;
    resolution.swapchain_textures = 3;
    resolution.pool_width = 1600;
    resolution.pool_height = 900;
    resolution.minimum_render_width = 1024;
    patch.resolution = resolution;
    // Width chosen at launch: packed code takes any width (the sites are gate
    // sites holding the console's words), ordinary preparation writes it.
    const auto set_width = [](const char* width) {
        if (width) setenv("ARMIGO_COMPAT_RESOLUTION_WIDTH", width, 1);
        else unsetenv("ARMIGO_COMPAT_RESOLUTION_WIDTH");
    };
    const auto swapchain = [](std::uint64_t width, std::uint64_t height) { return width | height << 32; };
    LaunchProfile profile;
    profile.title_id = 1;
    std::vector<PreparedLaunch> retained;
    std::vector<EmbeddedNativeArtifacts> captures;
    EmbeddedNativeArtifacts untouched;
    untouched.title_id = 99;
    const auto refused = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
        std::span(&patch, 1), {false, &untouched});
    CHECK(std::holds_alternative<LaunchFailure>(refused) && untouched.title_id == 99);
    std::vector<std::byte> embedded_text, embedded_appendix;
    unsigned iteration = 0;
    for (bool embedded : {false, true, true}) {
        // Ordinary at 1280, then embedded captures at the console's size and at 1280.
        const bool console = iteration++ == 1;
        set_width(console ? nullptr : "1280");
        EmbeddedNativeArtifacts captured;
        auto result = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
                                         std::span(&patch, 1), {embedded, embedded ? &captured : nullptr});
        if (const auto* error = std::get_if<LaunchFailure>(&result)) {
            std::fprintf(stderr, "Synthetic native launch: %s\n", error->detail.c_str());
            CHECK(false);
            return;
        }
        retained.push_back(std::get<PreparedLaunch>(std::move(result)));
        const auto& launch = retained.back();
        CHECK(launch.nvn_driver && launch.modules->module_name(1) == "main");
        const auto& mapping = launch.modules->mapping(1);
        const auto base = mapping.base_address();
        CHECK(launch.startup_calls.size() == names.size());
        for (std::size_t i = 0; i < launch.startup_calls.size(); ++i)
            CHECK(launch.startup_calls[i].phase == names[i]);
        CHECK(launch.call_hooks && launch.call_hooks->sites().size() == 1);
        if (launch.call_hooks) {
            CHECK(launch.call_hooks->sites()[0].call_return_address == base + 0x404);
            CHECK(launch.call_hooks->sites()[0].target_address == base + 0x500);
        }
        const auto data = mapping.readable_bytes(0x8000, 8);
        std::uint64_t pointer = 0;
        if (data) std::memcpy(&pointer, data->data(), 8);
        CHECK(pointer == base + 0x200);
        const auto text = mapping.readable_bytes(0, 0x2000);
        CHECK(text.has_value());
        std::uint32_t patched_word = 0;
        if (text) std::memcpy(&patched_word, text->data() + 0x300, 4);
        CHECK(patched_word == 0xd2800040);
        if (embedded) {
            const auto appendix = *launch.modules->appended_code(1);
            const auto code = *mapping.readable_bytes(appendix.offset, appendix.size);
            if (embedded_text.empty()) {
                embedded_text.assign(text->begin(), text->end());
                embedded_appendix.assign(code.begin(), code.end());
            } else {
                CHECK(std::equal(text->begin(), text->end(), embedded_text.begin()));
                CHECK(std::equal(code.begin(), code.end(), embedded_appendix.begin()));
            }
            const auto key = *mapping.readable_bytes(appendix.offset - execution::embedded_key_page_size, 24);
            std::uint64_t slot = 0, dispatcher = 0;
            std::memcpy(&slot, key.data(), 8);
            std::memcpy(&dispatcher, key.data() + 8, 8);
            CHECK(slot == launch.thread_slot->native_access()->tsd_byte_offset);
            CHECK(dispatcher == execution::gate_dispatcher_address());
            const auto sdk_code = *launch.modules->appended_code(2);
            const auto& sdk_mapping = launch.modules->mapping(2);
            const auto nvn_key = *sdk_mapping.readable_bytes(
                sdk_code.offset - execution::embedded_key_page_size + execution::embedded_nvn_bootstrap_offset, 8);
            std::uint64_t target = 0;
            std::memcpy(&target, nvn_key.data(), 8);
            CHECK(target != 0);
            std::uint32_t redirect = 0;
            const auto entry = *sdk_mapping.readable_bytes(0x100, 4);
            std::memcpy(&redirect, entry.data(), 4);
            CHECK(aarch64::decode(redirect).opcode == aarch64::Opcode::b);
            CHECK(captured.modules.size() == 3 && captured.title_id == profile.title_id &&
                  captured.preparation_identity == installed_embedded_preparation_identity());
            CHECK(!captured.modules[0].host_executes && captured.modules[1].host_executes);
            const auto& saved = captured.modules[1];
            CHECK(saved.hooks.size() == 1 && saved.hooks[0].target.module_index == 1 &&
                  saved.hooks[0].target.image_offset == 0x500);
            CHECK(saved.foreign_entries.size() == 1 && saved.foreign_entries[0].module_index == 2 &&
                  saved.foreign_entries[0].image_offset == 0x100);
            for (std::size_t i = 0; i < captured.modules.size(); ++i) {
                const auto& artifact = captured.modules[i];
                CHECK(std::all_of(artifact.image.begin() + artifact.key_page_offset,
                                  artifact.image.begin() + artifact.appended_code_offset,
                                  [](std::byte value) { return value == std::byte{0}; }));
                if (!captures.empty()) CHECK(artifact.image == captures.front().modules[i].image);
            }
            std::uint64_t persisted_pointer = 1;
            std::memcpy(&persisted_pointer, saved.image.data() + 0x8000, 8);
            CHECK(persisted_pointer == 0);
            std::memcpy(&persisted_pointer, saved.image.data() + 0x8010, 8);
            CHECK(persisted_pointer == 0);
            if (!captures.empty()) CHECK(captured.code_settings_sha256 == captures.front().code_settings_sha256);
            captures.push_back(std::move(captured));
        }
        execution::GuestThreadContext thread;
        execution::GuestDispatchScope scope(launch.sites, thread);
        platform::ThreadSlotBinding binding(*launch.thread_slot, thread.bound_register_cell());
        CHECK(scope.setup_error() == execution::GuestDispatchFailureKind::none && binding.native_access());
        CHECK(reinterpret_cast<std::uint64_t(*)(std::uint64_t)>(base + 0x200)(35) == 42);
        CHECK(thread.guest_x18() == 42);
        CHECK(reinterpret_cast<std::uint64_t(*)()>(base + 0x600)() ==
              (console ? swapchain(1600, 900) : swapchain(1280, 720)));
        std::uint32_t site_word = 0;
        std::memcpy(&site_word, text->data() + 0x600, 4);
        CHECK(embedded ? (site_word & 0xfc000000u) == 0x14000000u : site_word == 0xd280a000u);
        CHECK(input.modules[0].bytes == source);
    }
    auto artifacts = captures.front();
    std::vector<platform::HostModule> backing_modules;
    std::vector<std::vector<platform::EmbeddedModulePiece>> pieces(artifacts.modules.size());
    std::vector<EmbeddedNativeBacking> backings(artifacts.modules.size());
    for (std::size_t i = 0; i < artifacts.modules.size(); ++i) {
        const auto& artifact = artifacts.modules[i];
        auto mapped = platform::HostModule::create(artifact.image, retained[1].modules->mapping(i).guest_regions(),
            {.max_image_bytes = artifact.image.size(), .host_executes = artifact.host_executes});
        CHECK(mapped.ok() && mapped.module->seal().ok());
        if (!mapped.ok()) return;
        for (const auto& region : mapped.module->host_regions()) {
            pieces[i].push_back({region.offset,
                {reinterpret_cast<const std::byte*>(mapped.module->base_address() + region.offset), region.size},
                memory::has_permission(region.permissions, memory::Permission::execute)});
        }
        backings[i].pieces = pieces[i];
        backing_modules.push_back(std::move(*mapped.module));
    }
    const auto load = [&] {
        return load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
            std::span(&patch, 1), {true, nullptr, &artifacts, backings});
    };
    // Restored code captured at the console's size runs at another width.
    set_width("1920");
    auto loaded = load();
    if (const auto* error = std::get_if<LaunchFailure>(&loaded)) {
        std::fprintf(stderr, "Synthetic embedded load: %s\n", error->detail.c_str());
        CHECK(false);
        return;
    }
    const auto& launch = std::get<PreparedLaunch>(loaded);
    CHECK(launch.startup_calls.size() == names.size());
    const auto& mapping = launch.modules->mapping(1);
    const auto base = mapping.base_address();
    execution::GuestThreadContext thread;
    execution::GuestDispatchScope scope(launch.sites, thread);
    platform::ThreadSlotBinding binding(*launch.thread_slot, thread.bound_register_cell());
    CHECK(scope.setup_error() == execution::GuestDispatchFailureKind::none && binding.native_access());
    CHECK(reinterpret_cast<std::uint64_t(*)(std::uint64_t)>(base + 0x200)(35) == 42);
    CHECK(thread.guest_x18() == 42);
    CHECK(reinterpret_cast<std::uint64_t(*)()>(base + 0x600)() == swapchain(1920, 1080));
    set_width(nullptr);
    for (std::size_t i = 0; i < artifacts.modules.size(); ++i) {
        const auto& artifact = artifacts.modules[i];
        const auto& mapped = launch.modules->mapping(i);
        CHECK(std::memcmp(reinterpret_cast<const void*>(mapped.base_address()), artifact.image.data(), artifact.text_size) == 0);
        CHECK(std::memcmp(reinterpret_cast<const void*>(mapped.base_address() + artifact.appended_code_offset),
            artifact.image.data() + artifact.appended_code_offset, artifact.image.size() - artifact.appended_code_offset) == 0);
    }
    const auto rejects = [&] {
        const auto refused = load();
        CHECK(std::holds_alternative<LaunchFailure>(refused));
        if (const auto* failure = std::get_if<LaunchFailure>(&refused)) CHECK(failure->code == LaunchError::artifact_mismatch);
    };
    artifacts.modules[1].source_sha256[0] ^= std::byte{1}; rejects();
    artifacts.modules[1].source_sha256[0] ^= std::byte{1};
    artifacts.code_settings_sha256[0] ^= std::byte{1}; rejects();
    artifacts.code_settings_sha256[0] ^= std::byte{1};
    const auto foreign = artifacts.modules[1].foreign_entries[0];
    artifacts.modules[1].foreign_entries[0].module_index = 99; rejects();
    artifacts.modules[1].foreign_entries[0] = foreign;
    artifacts.modules[1].image[artifacts.modules[1].key_page_offset] = std::byte{1}; rejects();
    artifacts.modules[1].image[artifacts.modules[1].key_page_offset] = std::byte{0};
    const auto hooks = artifacts.modules[1].hooks;
    artifacts.modules[1].hooks.clear(); rejects();
    artifacts.modules[1].hooks = hooks;
    backing_modules.clear();
    CHECK(reinterpret_cast<std::uint64_t(*)(std::uint64_t)>(base + 0x200)(1) == 8);

    auto pattern = (std::filesystem::temp_directory_path() / "armigo-embedded-XXXXXX").string();
    const auto directory = ::mkdtemp(pattern.data());
    CHECK(directory != nullptr);
    if (!directory) return;
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{directory};
    // Rate-specific words select at ordinary launch. Embedded capture always
    // records the profile's default, and a different fixed-code mode is refused.
    auto rate_patch = patch;
    rate_patch.frame_rate = compat::FrameRateSettings{30, 60, {30, 60, 120}};
    rate_patch.words.front().frame_rate = 60;
    auto private_word = rate_patch.words.front();
    private_word.frame_rate = 120;
    private_word.replacement_instruction = 0xd2800060u;  // MOV X0, #3.
    rate_patch.words.push_back(private_word);
    const auto word_at = [](const PreparedLaunch& selected) {
        std::uint32_t word = 0;
        const auto bytes = selected.modules->mapping(1).readable_bytes(0x300, 4);
        CHECK(bytes.has_value());
        if (bytes) std::memcpy(&word, bytes->data(), 4);
        return word;
    };
    for (const char* rate : {"0", "30", "60", "120"}) {
        setenv("ARMIGO_COMPAT_FRAME_RATE", rate, 1);
        auto selected = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
            std::span(&rate_patch, 1));
        CHECK(std::holds_alternative<PreparedLaunch>(selected));
        if (const auto* prepared = std::get_if<PreparedLaunch>(&selected))
            CHECK(word_at(*prepared) == (std::string_view(rate) == "120" ? 0xd2800060u :
                std::string_view(rate) == "60" ? 0xd2800040u : 0xd2800020u));
        CHECK(input.modules[0].bytes == source);
    }
    auto high_only = rate_patch;
    high_only.words.erase(high_only.words.begin());
    high_only.frame_rate->supported_rates = {30, 120};
    setenv("ARMIGO_COMPAT_FRAME_RATE", "60", 1);
    CHECK(std::holds_alternative<LaunchFailure>(load_native_launch(input, profile,
        std::make_shared<platform::ThreadSlotKey>(), std::span(&high_only, 1))));
    high_only.words.front().nvn_driver_only = true;
    setenv("ARMIGO_COMPAT_FRAME_RATE", "120", 1);
    setenv("ARMIGO_NVN", "stream", 1);
    CHECK(std::holds_alternative<LaunchFailure>(load_native_launch(input, profile,
        std::make_shared<platform::ThreadSlotKey>(), std::span(&high_only, 1))));
    setenv("ARMIGO_COMPAT_FRAME_RATE", "30", 1);
    CHECK(std::holds_alternative<PreparedLaunch>(load_native_launch(input, profile,
        std::make_shared<platform::ThreadSlotKey>(), std::span(&high_only, 1))));
    unsetenv("ARMIGO_NVN");
    setenv("ARMIGO_COMPAT_FRAME_RATE", "invalid", 1);
    CHECK(std::holds_alternative<LaunchFailure>(load_native_launch(input, profile,
        std::make_shared<platform::ThreadSlotKey>(), std::span(&rate_patch, 1))));
    setenv("ARMIGO_COMPAT_FRAME_RATE", "60", 1);
    EmbeddedNativeArtifacts rate_capture;
    auto selected = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
        std::span(&rate_patch, 1), {true, &rate_capture});
    CHECK(std::holds_alternative<PreparedLaunch>(selected));
    if (const auto* prepared = std::get_if<PreparedLaunch>(&selected)) CHECK(word_at(*prepared) == 0xd2800020u);
    const auto rate_directory = cleanup.path / "rate";
    CHECK(std::holds_alternative<std::monostate>(write_embedded_modules(rate_directory, rate_capture)));
    auto rate_files = open_embedded_modules(rate_directory);
    CHECK(std::holds_alternative<EmbeddedNativeFiles>(rate_files));
    if (const auto* files = std::get_if<EmbeddedNativeFiles>(&rate_files)) {
        for (const char* rate : {"60", "120"}) {
            setenv("ARMIGO_COMPAT_FRAME_RATE", rate, 1);
            auto mismatch = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
                std::span(&rate_patch, 1), {true, nullptr, &files->artifacts(), files->backings()});
            CHECK(std::holds_alternative<LaunchFailure>(mismatch));
        }
        unsetenv("ARMIGO_COMPAT_FRAME_RATE");
        auto restored = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
            std::span(&rate_patch, 1), {true, nullptr, &files->artifacts(), files->backings()});
        CHECK(std::holds_alternative<PreparedLaunch>(restored));
        if (const auto* prepared = std::get_if<PreparedLaunch>(&restored)) CHECK(word_at(*prepared) == 0xd2800020u);
    }
    unsetenv("ARMIGO_COMPAT_FRAME_RATE");
    CHECK(std::holds_alternative<std::monostate>(write_embedded_modules(cleanup.path, artifacts)));
    {
        auto files = open_embedded_modules(cleanup.path);
        if (const auto* error = std::get_if<LaunchFailure>(&files)) {
            std::fprintf(stderr, "Synthetic embedded files: %s\n", error->detail.c_str());
            CHECK(false);
            return;
        }
        const auto& opened = std::get<EmbeddedNativeFiles>(files);
        CHECK(opened.artifacts().modules.size() == artifacts.modules.size());
        for (std::size_t i = 0; i < artifacts.modules.size(); ++i)
            CHECK(opened.artifacts().modules[i].image == artifacts.modules[i].image);
        auto file_launch = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(),
            std::span(&patch, 1), {true, nullptr, &opened.artifacts(), opened.backings()});
        if (const auto* error = std::get_if<LaunchFailure>(&file_launch)) {
            std::fprintf(stderr, "Synthetic file-backed native launch: %s\n", error->detail.c_str());
            CHECK(false);
            return;
        }
        const auto& launched = std::get<PreparedLaunch>(file_launch);
        execution::GuestThreadContext file_thread;
        execution::GuestDispatchScope file_scope(launched.sites, file_thread);
        platform::ThreadSlotBinding file_binding(*launched.thread_slot, file_thread.bound_register_cell());
        CHECK(file_binding.native_access().has_value());
        CHECK(reinterpret_cast<std::uint64_t(*)(std::uint64_t)>(launched.modules->mapping(1).base_address() + 0x200)(35) == 42);
    }
    const auto manifest = cleanup.path / "modules.armigo-embedded";
    std::ifstream stream(manifest, std::ios::binary | std::ios::ate);
    std::vector<std::byte> metadata(static_cast<std::size_t>(stream.tellg()));
    stream.seekg(0); stream.read(reinterpret_cast<char*>(metadata.data()), metadata.size()); stream.close();
    const auto original_metadata = metadata;
    const auto write_metadata = [&] {
        std::ofstream output(manifest, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(metadata.data()), metadata.size());
    };
    metadata[0] ^= std::byte{1}; write_metadata();
    CHECK(std::holds_alternative<LaunchFailure>(open_embedded_modules(cleanup.path)));
    metadata = original_metadata;
    const auto module_count = 8 + 4 + 8 + 4 + artifacts.preparation_identity.size() + 32;
    std::fill_n(metadata.begin() + module_count, 4, std::byte{0xff});
    const auto digest = loader::detail::sha256(std::span(metadata).first(metadata.size() - 32));
    CHECK(digest.has_value());
    std::copy(digest->begin(), digest->end(), metadata.end() - 32); write_metadata();
    CHECK(std::holds_alternative<LaunchFailure>(open_embedded_modules(cleanup.path)));
    metadata = original_metadata; write_metadata();
    for (const auto& entry : std::filesystem::directory_iterator(cleanup.path)) {
        if (entry.path().extension() != ".armigo-image") continue;
        std::fstream image(entry.path(), std::ios::binary | std::ios::in | std::ios::out);
        const char invalid = 1; image.write(&invalid, 1); image.close();
        CHECK(std::holds_alternative<LaunchFailure>(open_embedded_modules(cleanup.path)));
        break;
    }
}

} // namespace

int main() {
    test_system_roles();
    test_x18_roles();
    test_byte_indexed_x18_switch();
    test_halfword_indexed_x18_switch();
    test_exclusive_roles();
    test_unknown_control();
    test_scanned_preparation_matches();
    test_embedded_preparation_rebases();
    test_launch_words();
    test_embedded_native_launch();
    test_scan_bound();
    test_changed_text_is_refused();
    test_weak_null_import();
    if (failures) return 1;
    std::puts("native preparation tests passed");
    return 0;
}
