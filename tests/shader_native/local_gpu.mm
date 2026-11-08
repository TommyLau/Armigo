// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed native local memory. Each program stores, then loads, and
// exports R0-R3. Expected words come from an independent little-endian model
// of one 256-byte thread array: the program header contributes 64 bytes, 128
// bytes and a 2048-byte warp stack (64 bytes once divided by 32 threads).

#include "metal_harness.h"
#include <array>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sn = armigo::shader_native;
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

// STL. type selects the access width. A uniform address uses opcode 0x987
// unless a caller names 0x387, and adds UR (63 adds nothing).
sn::Instruction stl(unsigned data, unsigned ra, std::int32_t offset, unsigned type, bool uniform = false, unsigned ur = 63, unsigned opcode = 0) {
    if (opcode == 0) opcode = uniform ? 0x987u : 0x387u;
    auto i = op(opcode, {{24, 8, ra}, {32, 8, data}, {40, 24, std::uint32_t(offset) & 0xffffffu}, {73, 3, type}, {84, 3, 1}});
    if (uniform) { field(i, 64, 6, ur); field(i, 91, 1, 1); }
    return i;
}
// LDL. cache 1 is the ordinary hint; cache 3 marks the load last-use.
sn::Instruction ldl(unsigned dest, unsigned ra, std::int32_t offset, unsigned type, unsigned cache = 1, bool uniform = false, unsigned ur = 63) {
    auto i = op(0x983, {{16, 8, dest}, {24, 8, ra}, {40, 24, std::uint32_t(offset) & 0xffffffu}, {73, 3, type}, {84, 3, cache}});
    if (uniform) { field(i, 32, 6, ur); field(i, 91, 1, 1); }
    return i;
}

void expect_encoding(const sn::Instruction& instruction, std::uint32_t w0, std::uint32_t w1, std::uint32_t w2, const char* name) {
    if (instruction.words[0] != w0 || instruction.words[1] != w1 || instruction.words[2] != w2 || instruction.words[3] != 0) {
        std::fprintf(stderr, "%s: %08x %08x %08x %08x\n", name, instruction.words[0], instruction.words[1], instruction.words[2], instruction.words[3]);
        throw std::runtime_error("local memory corpus encoding");
    }
}
void check_corpus() {
    expect_encoding(stl(0, 255, 4, 4), 0xff007387u, 0x00000400u, 0x00100800u, "STL R0 [RZ+4]");
    expect_encoding(stl(4, 255, 0, 4), 0xff007387u, 0x00000004u, 0x00100800u, "STL R4 [RZ]");
    expect_encoding(stl(9, 255, 0, 4), 0xff007387u, 0x00000009u, 0x00100800u, "STL R9 [RZ]");
    expect_encoding(ldl(27, 255, 0, 4, 3), 0xff1b7983u, 0x00000000u, 0x00300800u, "LDL.LU R27 [RZ]");
    expect_encoding(ldl(0, 255, 4, 4, 3), 0xff007983u, 0x00000400u, 0x00300800u, "LDL.LU R0 [RZ+4]");
    expect_encoding(ldl(11, 255, 8, 4, 1), 0xff0b7983u, 0x00000800u, 0x00100800u, "LDL R11 [RZ+8]");
    expect_encoding(ldl(6, 255, 0, 5, 1), 0xff067983u, 0x00000000u, 0x00100a00u, "LDL.64 R6 [RZ]");
    expect_encoding(ldl(36, 36, 0x50, 6, 1), 0x24247983u, 0x00005000u, 0x00100c00u, "LDL.128 R36 [R36+0x50]");
}

// Independent model of the thread-private array. Width follows the access
// type: one byte, one halfword, or one to four words, little-endian.
struct Local {
    std::array<std::uint8_t, 256> bytes{};
    static unsigned width(unsigned type) { return type < 2 ? 1u : type < 4 ? 2u : 4u; }
    static unsigned count(unsigned type) { return type >= 6 ? 4u : type == 5 ? 2u : 1u; }
    static bool sign(unsigned type) { return type == 1 || type == 3; }
    void store(std::uint32_t address, unsigned type, const std::uint32_t* words) {
        const auto step = width(type);
        for (unsigned i = 0; i < count(type); ++i)
            for (unsigned b = 0; b < step; ++b) bytes.at(address + i * step + b) = static_cast<std::uint8_t>(words[i] >> (8 * b));
    }
    std::uint32_t load_one(std::uint32_t address, unsigned nbytes, bool is_signed) const {
        std::uint32_t value = 0;
        for (unsigned b = 0; b < nbytes; ++b) value |= std::uint32_t{bytes.at(address + b)} << (8 * b);
        if (is_signed && nbytes < 4) {
            const unsigned shift = 32u - 8u * nbytes;
            value = static_cast<std::uint32_t>(static_cast<std::int32_t>(value << shift) >> static_cast<int>(shift));
        }
        return value;
    }
    void load(std::uint32_t address, unsigned type, std::uint32_t* words) const {
        const auto step = width(type);
        for (unsigned i = 0; i < count(type); ++i) words[i] = load_one(address + i * step, step, sign(type));
    }
};
std::uint32_t address_of(std::uint32_t base, std::int32_t offset) { return base + static_cast<std::uint32_t>(offset); }

struct Case {
    std::string name;
    std::vector<sn::Instruction> body;
    std::vector<Words> inputs;
    std::function<Pixel(const Words&)> expected;
};
// Bank 3 supplies R8-R15. The three header words are the whole local allocation.
Fixture program(const Case& c) {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    for (unsigned i = 0; i < 8; ++i) code.push_back(load(8 + i, i));
    code.insert(code.end(), c.body.begin(), c.body.end());
    code.push_back(exit_instruction());
    Fixture f(code);
    put(f.code, 4, 64);
    put(f.code, 8, 128);
    put(f.code, 12, 64 * 32);
    put(f.code, 0x48, 0xf);
    return f;
}
Words pad(Words w) { w.resize(8); return w; }
std::vector<Words> patterns() {
    static constexpr std::uint32_t samples[] = {
        0x8001807fu, 0x7fff00ffu, 0xff00ff00u, 0x00018000u, 0xffffffffu, 0x01020304u, 0xfedcba98u, 0x7fff8000u,
        0x0080007fu, 0xa5a5a5a5u, 0x5a5a5a5au, 0x80000001u, 0xffff0000u, 0x12345678u, 0x9abcdef0u, 0x00000080u};
    std::vector<Words> out;
    for (unsigned start = 0; start < 16; start += 4) {
        Words w(8);
        for (unsigned i = 0; i < 8; ++i) w[i] = samples[(start + i) % 16];
        out.push_back(w);
    }
    return out;
}

std::vector<Case> cases() {
    std::vector<Case> c;
    // Offsets 0 and 4 use the uniform opcode and the plain opcode with URZ.
    // Offset 252 is the last word of the 256-byte allocation.
    c.push_back({"static 32-bit round trip", {
        stl(8, 255, 0, 4, true, 63), stl(9, 255, 4, 4, true, 63, 0x387), stl(10, 255, 16, 4), stl(11, 255, 252, 4),
        ldl(0, 255, 0, 4, 1, true, 63), ldl(1, 255, 4, 4, 3), ldl(2, 255, 16, 4), ldl(3, 255, 252, 4, 3)},
        patterns(), [](const Words& in) { return Pixel{in[0], in[1], in[2], in[3]}; }});
    c.push_back({"narrow loads of one word", {
        stl(8, 255, 0, 4), ldl(0, 255, 0, 0, 3), ldl(1, 255, 1, 1, 3), ldl(2, 255, 2, 2, 3), ldl(3, 255, 0, 3)},
        patterns(), [](const Words& in) {
            Local local; local.store(0, 4, &in[0]);
            return Pixel{local.load_one(0, 1, false), local.load_one(1, 1, true), local.load_one(2, 2, false), local.load_one(0, 2, true)};
        }});
    c.push_back({"narrow stores", {
        stl(8, 255, 3, 0), stl(9, 255, 6, 2), stl(10, 255, 1, 1),
        ldl(0, 255, 3, 0, 3), ldl(1, 255, 1, 1, 3), ldl(2, 255, 6, 2), ldl(3, 255, 0, 4)},
        patterns(), [](const Words& in) {
            Local local; local.store(3, 0, &in[0]); local.store(6, 2, &in[1]); local.store(1, 1, &in[2]);
            std::uint32_t word = 0; local.load(0, 4, &word);
            return Pixel{local.load_one(3, 1, false), local.load_one(1, 1, true), local.load_one(6, 2, false), word};
        }});
    // A zero register overwrites a previous store.
    c.push_back({"store from the zero register", {stl(8, 255, 8, 4), stl(255, 255, 8, 4), ldl(0, 255, 8, 4, 3)},
        patterns(), [](const Words&) { return Pixel{0, 0, 0, 0}; }});
    c.push_back({"128-bit store and 64-bit loads", {stl(8, 255, 240, 6), ldl(0, 255, 240, 5), ldl(2, 255, 248, 5, 3)},
        patterns(), [](const Words& in) { return Pixel{in[0], in[1], in[2], in[3]}; }});
    c.push_back({"64-bit store and narrow loads", {
        stl(8, 255, 248, 5), ldl(0, 255, 248, 5, 3), ldl(2, 255, 248, 0, 3), ldl(3, 255, 249, 1)},
        patterns(), [](const Words& in) {
            Local local; local.store(248, 5, &in[0]);
            std::uint32_t wide[2]{}; local.load(248, 5, wide);
            return Pixel{wide[0], wide[1], local.load_one(248, 1, false), local.load_one(249, 1, true)};
        }});
    // R8 is the address. Displacements 0, +4 and -4, including an odd base,
    // where the immediate stays aligned and the sum does not.
    std::vector<Words> dynamic;
    for (const std::uint32_t base : {4u, 5u, 8u, 16u, 64u, 100u, 200u, 248u})
        for (const auto& sample : patterns()) dynamic.push_back(pad({base, sample[0], sample[1], sample[2]}));
    c.push_back({"dynamic register address", {
        stl(9, 8, 0, 4), stl(10, 8, 4, 4), stl(11, 8, -4, 4),
        ldl(0, 8, 0, 4), ldl(1, 8, -4, 4, 3), ldl(2, 8, 1, 1, 3), ldl(3, 8, 4, 2)},
        dynamic, [](const Words& in) {
            Local local;
            local.store(address_of(in[0], 0), 4, &in[1]);
            local.store(address_of(in[0], 4), 4, &in[2]);
            local.store(address_of(in[0], -4), 4, &in[3]);
            return Pixel{local.load_one(address_of(in[0], 0), 4, false), local.load_one(address_of(in[0], -4), 4, false),
                local.load_one(address_of(in[0], 1), 1, true), local.load_one(address_of(in[0], 4), 2, false)};
        }});
    // UR4 is input 0. R9 is an extra displacement. Both store opcodes appear.
    std::vector<Words> uniform;
    for (const auto& [base, extra] : {std::pair{0u, 0u}, {4u, 8u}, {5u, 1u}, {16u, 32u}, {100u, 7u}, {200u, 40u}, {248u, 0u}, {1u, 3u}})
        for (const auto& sample : patterns()) uniform.push_back(pad({base, extra, sample[0], sample[1]}));
    c.push_back({"uniform address", {uniform_load(4, 0),
        stl(10, 255, 0, 4, true, 4), stl(11, 9, 4, 4, true, 4, 0x387),
        ldl(0, 255, 0, 4, 1, true, 4), ldl(1, 9, 4, 4, 1, true, 4),
        ldl(2, 255, 2, 3, 3, true, 4), ldl(3, 9, 5, 0, 1, true, 4)},
        uniform, [](const Words& in) {
            Local local;
            const auto base = in[0], extra = in[1];
            local.store(base, 4, &in[2]);
            local.store(address_of(base + extra, 4), 4, &in[3]);
            return Pixel{local.load_one(base, 4, false), local.load_one(address_of(base + extra, 4), 4, false),
                local.load_one(address_of(base, 2), 2, true), local.load_one(address_of(base + extra, 5), 1, false)};
        }});
    return c;
}
}
int main() { @autoreleasepool { try {
    check_corpus();
    MetalHarness gpu;
    unsigned checked = 0;
    for (const auto& test : cases()) {
        auto compiled = gpu.compile(program(test));
        for (const auto& input : test.inputs) {
            const auto got = gpu.run(compiled, input);
            const auto want = test.expected(input);
            if (got != want) {
                std::fprintf(stderr, "%s:", test.name.c_str());
                for (const auto word : input) std::fprintf(stderr, " %08x", word);
                std::fprintf(stderr, "\n got %08x %08x %08x %08x\n expected %08x %08x %08x %08x\n",
                    got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                throw std::runtime_error("native local memory GPU result");
            }
            ++checked;
        }
    }
    std::printf("Native local Metal execution passed (%u vectors)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
