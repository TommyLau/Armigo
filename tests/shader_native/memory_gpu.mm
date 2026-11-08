// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed native global memory: register and uniform address forms,
// descriptor-form addresses (the uniform register is not an address term),
// narrow signed loads, wide loads and stores, atomics, reductions,
// compare-exchange and 32-bit INC/DEC through one guest-memory pool that crosses a 4 GiB
// boundary, and the miss record of an address outside it. Fragment position
// rows, which a negated guest Y origin counts from the render target's other
// edge, use the same per-draw record as the memory tests' host.

#include "metal_harness.h"
#include "nvn_guest_memory.h"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <string>
#include <tuple>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
using Words = std::vector<std::uint32_t>;
using Pixel = std::array<std::uint32_t, 4>;
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
sn::Instruction load(unsigned d, unsigned index) { return op(0xa02, {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}}); }
sn::Instruction uniform_load(unsigned d, unsigned index) { return op(0xab9, {{16, 6, d}, {73, 3, 4}, {54, 5, 3}, {38, 16, index * 4}}); }

// The window: 64 words at guest [base, base + 256), across 0x7e00000000.
constexpr std::uint64_t base = 0x7dffffff80ull;
constexpr unsigned window_words = 64;
// The window as sixteen 16-byte pools from the host table builder: eight
// below the 2 MiB chunk boundary at 0x7e00000000 and eight above. Each
// chunk's partial hint names its first pool, so accesses to the first four
// pools of a chunk are found by the hinted scan and later ones by the search.
std::vector<std::uint32_t> pooled_table(std::uint64_t device) {
    namespace nd = armigo::platform::metal::nvn_driver;
    std::vector<nd::GuestMemoryWindow> pools;
    for (std::uint64_t at = base; at < base + window_words * 4; at += 16) pools.push_back({at, at + 16, device + (at - base)});
    auto table = nd::build_guest_memory_table(pools);
    check(table && table->entries == 16 && table->words[8 + ((base >> 21) & 8191) * 8 + 7] != 0, "pooled guest table");
    return table->words;
}
std::uint32_t initial(unsigned i) { return 0x9e3779b9u * (i + 1); }
Words memory() { Words w(window_words); for (unsigned i = 0; i < window_words; ++i) w[i] = initial(i); return w; }
unsigned index_of(std::uint64_t address) { return static_cast<unsigned>((address - base) / 4); }

// Common fields: type 73-75, order 77-80, eviction 84-86 (normal).
sn::Instruction memory_op(unsigned opcode, unsigned type, unsigned order, std::initializer_list<Field> fields) {
    auto i = op(opcode, fields);
    field(i, 73, 3, type); field(i, 77, 4, order); field(i, 84, 3, 1);
    return i;
}
// LDG.E Rd, [Ra.64 + displacement]
sn::Instruction ldg(unsigned d, unsigned a, std::int32_t displacement, unsigned type) {
    return memory_op(0x381, type, 5, {{16, 8, d}, {24, 8, a}, {40, 24, std::uint32_t(displacement) & 0xffffff}, {72, 1, 1}, {81, 3, 7}});
}
// LDG.E Rd, [Ra(.64) + UR.64 + displacement]
sn::Instruction uniform_ldg(unsigned d, unsigned a, bool wide, unsigned ur, std::int32_t displacement, unsigned type, unsigned order = 4) {
    return memory_op(0x981, type, order, {{16, 8, d}, {24, 8, a}, {32, 6, ur}, {40, 24, std::uint32_t(displacement) & 0xffffff},
        {72, 1, 1}, {81, 3, 7}, {90, 1, wide}, {91, 1, 1}});
}
// Descriptor form: URb is a memory descriptor and is not added to Ra.
sn::Instruction descriptor_ldg(unsigned d, unsigned a, unsigned ur, std::int32_t displacement, unsigned type) {
    return memory_op(0x981, type, 5, {{16, 8, d}, {24, 8, a}, {32, 6, ur}, {40, 24, std::uint32_t(displacement) & 0xffffff},
        {72, 1, 1}, {76, 1, 1}, {81, 3, 7}, {90, 1, 1}, {91, 1, 1}});
}
sn::Instruction descriptor_stg(unsigned a, unsigned data, unsigned ur, std::int32_t displacement, unsigned type) {
    return memory_op(0x986, type, 5, {{24, 8, a}, {32, 8, data}, {40, 24, std::uint32_t(displacement) & 0xffffff},
        {64, 6, ur}, {72, 1, 1}, {76, 1, 1}, {90, 1, 1}, {91, 1, 1}});
}
std::uint32_t wrap_inc(std::uint32_t cur, std::uint32_t bound) { return cur >= bound ? 0u : cur + 1u; }
std::uint32_t wrap_dec(std::uint32_t cur, std::uint32_t bound) { return cur == 0u || cur > bound ? bound : cur - 1u; }

struct Case {
    std::string name;
    std::vector<sn::Instruction> body;
    std::vector<Words> inputs;
    // Expected pixel and memory, given the input and the memory before.
    std::function<std::pair<Pixel, Words>(const Words&, Words)> expected;
};
Fixture program(const Case& c) {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    for (unsigned i = 0; i < 8; ++i) code.push_back(load(8 + i, i));
    code.insert(code.end(), c.body.begin(), c.body.end());
    code.push_back(exit_instruction());
    Fixture f(code); put(f.code, 0x48, 0xf);
    return f;
}
Words address(std::uint64_t a, std::initializer_list<std::uint32_t> more = {}) {
    Words w{std::uint32_t(a), std::uint32_t(a >> 32)}; w.insert(w.end(), more); w.resize(8); return w;
}
std::int32_t sign8(std::uint32_t word, unsigned byte) { return static_cast<std::int8_t>(word >> (8 * byte)); }
std::int32_t sign16(std::uint32_t word, unsigned half) { return static_cast<std::int16_t>(word >> (16 * half)); }

std::vector<Case> cases() {
    std::vector<Case> c;
    std::vector<Words> words;
    for (unsigned i = 7; i < window_words - 1; i += 7) words.push_back(address(base + 4 * i));
    // Register pair address with a positive and a negative displacement.
    c.push_back({"LDG.E.32 register pair", {ldg(0, 8, 0, 4), ldg(1, 8, -4, 4)},
        words,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            return std::pair{Pixel{m[at], m[at - 1], 0, 0}, m};
        }});
    std::vector<Words> quads;
    for (unsigned i = 0; i + 4 <= window_words; i += 12) quads.push_back(address(base + 4 * i));
    c.push_back({"LDG.E.128 into a register quad", {ldg(0, 8, 0, 6)}, quads,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            return std::pair{Pixel{m[at], m[at + 1], m[at + 2], m[at + 3]}, m};
        }});
    // UR4:UR5 + zero-extended R10, crossing into the high word.
    std::vector<Words> offsets;
    for (std::uint32_t o : {0x00u, 0x78u, 0x80u, 0x88u, 0xf0u}) offsets.push_back(address(base, {o}));
    c.push_back({"LDG.E.64 uniform base plus a register word", {uniform_load(4, 0), uniform_load(5, 1), uniform_ldg(0, 10, false, 4, 8, 5)}, offsets,
        [](const Words& in, Words m) {
            const auto at = index_of(base + in[2] + 8);
            return std::pair{Pixel{m[at], m[at + 1], 0, 0}, m};
        }});
    // URZ + R8.64 and UR4.64 + RZ.64: the uniform pair or the register pair alone.
    c.push_back({"LDG.E uniform and register pairs alone", {uniform_load(4, 0), uniform_load(5, 1),
        uniform_ldg(0, 8, true, 63, 0, 4, 5), uniform_ldg(1, 255, true, 4, 4, 4, 7)}, words,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            return std::pair{Pixel{m[at], m[at + 1], 0, 0}, m};
        }});
    // Narrow loads: U8, S8, U16, S16 at every byte and half of a word.
    std::vector<Words> bytes;
    for (unsigned i = 0; i < 8; ++i) bytes.push_back(address(base + 4 * (3 + i)));
    c.push_back({"narrow loads", {uniform_load(4, 0), uniform_load(5, 1),
        uniform_ldg(0, 255, true, 4, 1, 0), uniform_ldg(1, 255, true, 4, 3, 1), uniform_ldg(2, 255, true, 4, 2, 2), uniform_ldg(3, 255, true, 4, 2, 3)}, bytes,
        [](const Words& in, Words m) {
            const auto w = m[index_of(in[0] | std::uint64_t(in[1]) << 32)];
            return std::pair{Pixel{(w >> 8) & 0xff, std::uint32_t(sign8(w, 3)), w >> 16, std::uint32_t(sign16(w, 1))}, m};
        }});
    // STG.E.64 (uniform form) and STG.E.32 (register form), then a load back.
    std::vector<Words> stores;
    for (unsigned i = 0; i + 2 <= window_words; i += 10) stores.push_back(address(base + 4 * i, {0x11111111u * (i + 1), 0xcafe0000u + i}));
    c.push_back({"STG.E.64 and STG.E.32", {uniform_load(4, 0), uniform_load(5, 1),
        memory_op(0x986, 5, 5, {{24, 8, 255}, {32, 8, 10}, {64, 6, 4}, {72, 1, 1}, {90, 1, 1}, {91, 1, 1}}),
        memory_op(0x386, 4, 5, {{24, 8, 8}, {32, 8, 11}, {40, 24, 4}, {72, 1, 1}}),
        ldg(0, 8, 0, 6 - 1)}, stores,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            m[at] = in[2]; m[at + 1] = in[3];
            return std::pair{Pixel{m[at], m[at + 1], 0, 0}, m};
        }});
    // A word load and a word store in each order: weak (0), constant (4,
    // loads only), strong at SM scope (5) and strong at GPU scope (7).
    std::vector<Words> ordered;
    for (unsigned i = 3; i + 3 <= window_words; i += 11) ordered.push_back(address(base + 4 * i, {0x0badf00du + i}));
    for (const unsigned order : {0u, 4u, 5u, 7u}) {
        c.push_back({"word load and store, order " + std::to_string(order), {
            memory_op(0x381, 4, order, {{16, 8, 0}, {24, 8, 8}, {40, 24, 0}, {72, 1, 1}, {81, 3, 7}}),
            memory_op(0x386, 4, order == 4 ? 0 : order, {{24, 8, 8}, {32, 8, 10}, {40, 24, 8}, {72, 1, 1}})}, ordered,
            [](const Words& in, Words m) {
                const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
                const auto loaded = m[at];
                m[at + 2] = in[2];
                return std::pair{Pixel{loaded, 0, 0, 0}, m};
            }});
    }
    // Atomics: ADD (register form), ADD (uniform form, Ra.64 bit 70), RED MAX,
    // RED MIN signed, and compare-exchange.
    std::vector<Words> atomics;
    for (unsigned i = 0; i < window_words - 1; i += 9) atomics.push_back(address(base + 4 * i, {0x1000u + i, 0x80000005u, 0x7ffffff0u}));
    c.push_back({"ATOMG.ADD, RED.MAX and RED.MIN.S32", {uniform_load(4, 0), uniform_load(5, 1),
        memory_op(0x3a8, 0, 7, {{16, 8, 0}, {24, 8, 8}, {32, 8, 10}, {72, 1, 1}, {81, 3, 7}}),
        memory_op(0x9a8, 0, 7, {{16, 8, 1}, {24, 8, 255}, {32, 8, 10}, {64, 6, 4}, {70, 1, 1}, {72, 1, 1}, {81, 3, 7}, {91, 1, 1}}),
        memory_op(0x98e, 0, 7, {{24, 8, 255}, {32, 8, 11}, {40, 24, 4}, {64, 6, 4}, {72, 1, 1}, {87, 3, 2}, {90, 1, 1}, {91, 1, 1}}),
        memory_op(0x98e, 1, 7, {{24, 8, 8}, {32, 8, 12}, {40, 24, 4}, {72, 1, 1}, {87, 3, 1}}),
        ldg(2, 8, 0, 4), ldg(3, 8, 4, 4)}, atomics,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto first = m[at]; m[at] += in[2];
            const auto second = m[at]; m[at] += in[2];
            m[at + 1] = std::max(m[at + 1], in[3]);
            m[at + 1] = std::uint32_t(std::min(std::int32_t(m[at + 1]), std::int32_t(in[4])));
            return std::pair{Pixel{first, second, m[at], m[at + 1]}, m};
        }});
    std::vector<Words> exchanges;
    for (unsigned i = 0; i < window_words; i += 13) {
        exchanges.push_back(address(base + 4 * i, {initial(i), 0xabc00000u + i}));
        exchanges.push_back(address(base + 4 * i, {initial(i) ^ 1, 0xdef00000u + i}));
    }
    c.push_back({"ATOMG.CAS", {memory_op(0x3a9, 0, 7, {{16, 8, 0}, {24, 8, 8}, {32, 8, 10}, {64, 8, 11}, {72, 1, 1}, {81, 3, 7}}),
        ldg(1, 8, 0, 4)}, exchanges,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto old = m[at]; if (old == in[2]) m[at] = in[3];
            return std::pair{Pixel{old, m[at], 0, 0}, m};
        }});
    // Descriptor form: UR4:UR5 holds a nonzero value that must not move the
    // address. The load reads Ra, and the store writes the next word.
    std::vector<Words> described;
    for (unsigned i = 0; i + 2 <= window_words; i += 8)
        described.push_back(address(base + 4 * i, {0x01000000u, 0x2u, 0xa5a50000u + i}));
    c.push_back({"descriptor-form LDG and STG ignore the uniform register", {
        uniform_load(4, 2), uniform_load(5, 3), descriptor_ldg(0, 8, 4, 0, 4), descriptor_stg(8, 12, 4, 4, 4), ldg(1, 8, 4, 4)}, described,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto loaded = m[at];
            m[at + 1] = in[4];
            return std::pair{Pixel{loaded, m[at + 1], 0, 0}, m};
        }});
    // Descriptor-form ATOMG.INC and RED.DEC. UR5:UR6 holds a nonzero sentinel
    // at an odd index with bit 72 clear; the address is Ra above 4 GiB.
    auto described_atomic = [](std::initializer_list<std::pair<std::uint32_t, std::uint32_t>> samples) {
        std::vector<Words> inputs;
        unsigned slot = 0;
        for (auto [bound, cur] : samples) {
            inputs.push_back(address(base + 4 * slot, {0x01010101u, 0x02020202u, bound, cur}));
            slot += 4;
        }
        return inputs;
    };
    const auto described_bounds = described_atomic({{5u, 5u}, {5u, 3u}, {5u, 0u}, {0u, 0u}});
    c.push_back({"descriptor-form ATOMG.INC ignores the uniform register", {
        uniform_load(5, 2), uniform_load(6, 3),
        memory_op(0x386, 4, 5, {{24, 8, 8}, {32, 8, 13}, {72, 1, 1}}),
        memory_op(0x9a8, 0, 7, {{16, 8, 0}, {24, 8, 8}, {32, 8, 12}, {64, 6, 5}, {70, 1, 1}, {71, 1, 1}, {72, 1, 0}, {81, 3, 7}, {87, 4, 3}, {91, 1, 1}}),
        ldg(1, 8, 0, 4)}, described_bounds,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto next = wrap_inc(in[5], in[4]);
            m[at] = next;
            return std::pair{Pixel{in[5], next, 0, 0}, m};
        }});
    c.push_back({"descriptor-form RED.DEC ignores the uniform register", {
        uniform_load(5, 2), uniform_load(6, 3),
        memory_op(0x386, 4, 5, {{24, 8, 8}, {32, 8, 13}, {72, 1, 1}}),
        memory_op(0x98e, 0, 7, {{24, 8, 8}, {32, 8, 12}, {64, 6, 5}, {71, 1, 1}, {72, 1, 0}, {87, 3, 4}, {90, 1, 1}, {91, 1, 1}}),
        ldg(0, 8, 0, 4)}, described_atomic({{5u, 0u}, {5u, 9u}, {5u, 3u}, {0u, 0u}}),
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto next = wrap_dec(in[5], in[4]);
            m[at] = next;
            return std::pair{Pixel{next, 0, 0, 0}, m};
        }});
    // R10 is the bound, R11 the word stored first. INC and DEC return that
    // word; the memory word is the wrapped result.
    auto atomic_inputs = [](std::initializer_list<std::pair<std::uint32_t, std::uint32_t>> samples) {
        std::vector<Words> inputs;
        unsigned slot = 0;
        for (auto [bound, cur] : samples) {
            inputs.push_back(address(base + 4 * slot, {bound, cur}));
            slot += 4;
        }
        return inputs;
    };
    const auto bounds = atomic_inputs({{5u, 5u}, {5u, 9u}, {5u, 3u}, {0u, 0u}, {1u, 0u}, {0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xfffffffeu}});
    auto stored_atomic = [](unsigned opcode, unsigned operation, bool reduce) {
        std::vector<sn::Instruction> body{memory_op(0x386, 4, 5, {{24, 8, 8}, {32, 8, 11}, {72, 1, 1}})};
        if (reduce) body.push_back(memory_op(opcode, 0, 7, {{24, 8, 8}, {32, 8, 10}, {72, 1, 1}, {87, 3, operation}}));
        else body.push_back(memory_op(opcode, 0, 7, {{16, 8, 0}, {24, 8, 8}, {32, 8, 10}, {72, 1, 1}, {81, 3, 7}, {87, 4, operation}}));
        body.push_back(ldg(reduce ? 0 : 1, 8, 0, 4));
        return body;
    };
    c.push_back({"ATOMG.INC returns the old word and wraps", stored_atomic(0x3a8, 3, false), bounds,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto next = wrap_inc(in[3], in[2]);
            m[at] = next;
            return std::pair{Pixel{in[3], next, 0, 0}, m};
        }});
    c.push_back({"ATOMG.DEC returns the old word and wraps", stored_atomic(0x3a8, 4, false),
        atomic_inputs({{5u, 0u}, {5u, 9u}, {5u, 5u}, {5u, 1u}, {0u, 0u}, {7u, 8u}}),
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto next = wrap_dec(in[3], in[2]);
            m[at] = next;
            return std::pair{Pixel{in[3], next, 0, 0}, m};
        }});
    c.push_back({"RED.INC writes the wrapped word", stored_atomic(0x98e, 3, true), bounds,
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto next = wrap_inc(in[3], in[2]);
            m[at] = next;
            return std::pair{Pixel{next, 0, 0, 0}, m};
        }});
    c.push_back({"RED.DEC writes the wrapped word", stored_atomic(0x98e, 4, true),
        atomic_inputs({{5u, 0u}, {5u, 9u}, {5u, 5u}, {5u, 1u}, {0u, 0u}, {7u, 8u}}),
        [](const Words& in, Words m) {
            const auto at = index_of(in[0] | std::uint64_t(in[1]) << 32);
            const auto next = wrap_dec(in[3], in[2]);
            m[at] = next;
            return std::pair{Pixel{next, 0, 0, 0}, m};
        }});
    return c;
}
}
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    unsigned checked = 0;
    for (const auto& test : cases()) {
        auto compiled = gpu.compile(program(test));
        // Each input runs with the pool behind its chunk hint, behind the
        // partial hints (where only the hinted scan finds it), and as sixteen
        // pools from the host builder.
        for (const unsigned layout : {0u, 1u, 2u}) for (const auto& input : test.inputs) {
            MetalHarness::GuestMemory guest{base, memory(), {}, layout == 1};
            if (layout == 2) guest.build_table = pooled_table;
            const auto before = guest.words;
            const auto got = gpu.run(compiled, input, &guest);
            const auto [want, after] = test.expected(input, before);
            // The write log marks the one pool (entry 1, bit 1 of word 0) for
            // a stage that writes guest memory, and stays clear otherwise.
            // Sixteen pools: a writing stage marks only pools 1-16.
            const auto log = guest.table[ir::guest_memory_write_log_byte / 4];
            const bool writes = compiled.artifact.reflection.guest_memory_writes;
            if (layout == 2 ? (writes ? log == 0 || (log & ~0x1fffeu) != 0 : log != 0) : log != (writes ? 2u : 0u)) {
                std::fprintf(stderr, "%s: write log %08x\n", test.name.c_str(), log);
                throw std::runtime_error("native guest-memory write log");
            }
            if (got != want || guest.words != after || guest.table[1] != 0) {
                std::fprintf(stderr, "%s: input %08x %08x %08x: got %08x %08x %08x %08x, expected %08x %08x %08x %08x; misses %u\n",
                    test.name.c_str(), input[0], input[1], input[2], got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3], guest.table[1]);
                throw std::runtime_error("native global memory GPU result");
            }
            ++checked;
        }
    }
    // Strong orders become relaxed device atomics in the Metal source; weak
    // and constant accesses stay plain pointer accesses.
    for (const auto& test : cases()) {
        if (test.name.rfind("word load and store, order ", 0)) continue;
        const auto source = gpu.compile(program(test)).artifact.source;
        const bool strong = test.name.back() == '5' || test.name.back() == '7';
        check((source.find("atomic_load_explicit(reinterpret_cast<device atomic_uint*>(guest_device") != std::string::npos) == strong &&
              (source.find("atomic_store_explicit(reinterpret_cast<device atomic_uint*>(guest_device") != std::string::npos) == strong,
              "strong guest loads and stores are relaxed device atomics, weak and constant ones plain");
    }
    // The reflection names the stages that write guest memory.
    for (const auto& test : cases()) {
        const bool writes = gpu.compile(program(test)).artifact.reflection.guest_memory_writes;
        if (test.name == "word load and store, order 0") check(writes, "a store makes the stage a guest-memory writer");
        if (test.name.rfind("LDG", 0) == 0 && test.name.find("ST") == std::string::npos && test.name.find("ATOM") == std::string::npos)
            check(!writes, "a load-only stage reads guest memory");
    }
    // A load and a store outside the window: zero, memory unchanged, and the
    // first miss recorded with the program's tag and address.
    Case outside{"miss", {ldg(0, 8, 0, 4), memory_op(0x386, 4, 5, {{24, 8, 8}, {32, 8, 10}, {40, 24, 8}, {72, 1, 1}})}, {}, {}};
    const auto fixture = program(outside);
    auto compiled = gpu.compile(fixture);
    const std::uint64_t far = base + 4 * window_words + 16;
    MetalHarness::GuestMemory guest{base, memory(), {}};
    const auto got = gpu.run(compiled, address(far, {0x5555u}), &guest);
    const auto tag = static_cast<std::uint32_t>(std::stoul(fixture.program().stage_identity.substr(0, 8), nullptr, 16));
    check(compiled.artifact.reflection.guest_memory_tag == tag, "the guest-memory tag is the stage identity's first four bytes");
    if (got != Pixel{0, 0, 0, 0} || guest.words != memory() || guest.table[1] != 2 || guest.table[2] != tag ||
        guest.table[ir::guest_memory_write_log_byte / 4] != 0 ||
        guest.table[3] != std::uint32_t(far) || guest.table[4] != std::uint32_t(far >> 32)) {
        std::fprintf(stderr, "miss: %08x, count %u tag %08x address %08x%08x\n", got[0], guest.table[1], guest.table[2], guest.table[4], guest.table[3]);
        throw std::runtime_error("native global memory miss record");
    }
    // IPA.PASS of PositionX and PositionY at the single pixel's centre (0.5,
    // 0.5): Y is height - 0.5 when the guest Y origin is negated.
    auto ipa = [](unsigned d, unsigned attribute) {
        return op(0x326, {{16, 8, d}, {32, 8, 255}, {64, 8, attribute / 4}, {81, 3, 7}});
    };
    Fixture position({move(0, 0), move(1, 0), move(2, 0), move(3, 0), ipa(0, 0x70), ipa(1, 0x74), exit_instruction()});
    put(position.code, 0x48, 0xf); position.code[0x17] = std::byte{0x30};
    for (const bool negate : {false, true}) {
        auto rows = gpu.compile(position, negate);
        check(rows.artifact.reflection.reads_render_target_height && rows.artifact.reflection.draw_support_buffer_index,
              "fragment Y reads the render-target height from the draw record");
        for (const std::uint32_t height : {1u, 8u, 720u, 2160u}) {
            const auto got = gpu.run(rows, {}, nullptr, height);
            const float y = negate ? float(height) - 0.5f : 0.5f;
            if (got != Pixel{std::bit_cast<std::uint32_t>(0.5f), std::bit_cast<std::uint32_t>(y), 0, 0}) {
                std::fprintf(stderr, "position negate %d height %u: %08x %08x\n", negate, height, got[0], got[1]);
                throw std::runtime_error("native fragment position rows");
            }
            ++checked;
        }
    }
    // A vertex stage with no outputs (an empty program, as titles bind for
    // draws that discard their primitives) returns void and builds a
    // rasterization-disabled pipeline.
    {
        Fixture empty({exit_instruction()}, sn::Stage::vertex);
        empty.code[0x35] = std::byte{};
        sn::FrontendOptions options; options.vertex_depth = ir::DepthConvention::negative_one_to_one;
        const auto artifact = gpu.discard_pipeline(empty, options);
        check(artifact.reflection.requires_rasterizer_discard && artifact.source.find("vertex void ") != std::string::npos,
              "an output-less vertex stage requires rasterizer discard");
        ++checked;
    }
    std::printf("Native global memory and fragment position Metal execution passed (%u vectors, miss record)\n", checked + 1);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
