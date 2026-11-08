// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed compute: shared loads and stores of every admitted width,
// a scaled register address, a negative displacement, a uniform-register
// address, a cross-thread value published through a workgroup barrier,
// uniform group ids and the uniform wide multiply-add. Expected words come
// from a byte-array model of the workgroup memory and from 64-bit arithmetic.

#include "metal_harness.h"
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <tuple>
#include <vector>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
using Words = std::vector<std::uint32_t>;
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
constexpr unsigned PT = 7, NOT_PT = 15, RZ = 255, URZ = 63;
constexpr std::uint64_t guest_base = 0x7dfff00000ull;
constexpr unsigned kSharedBytes = 4096;
// The uniform address register holds this byte offset and the load adds 4,
// which lands on the positive-displacement store.
constexpr std::uint32_t kUniformOffset = 252;
constexpr std::uint32_t kPositive = 256;
constexpr std::uint32_t kSentinel = 0xC0FFEE01u;

sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
sn::Instruction load(unsigned d, unsigned index) { return op(0xa02, {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}}); }
sn::Instruction s2r(unsigned d, unsigned selector) { return op(0x919, {{16, 8, d}, {72, 8, selector}}); }
sn::Instruction s2ur(unsigned d, unsigned selector) { return op(0x9c3, {{16, 6, d}, {72, 8, selector}}); }
sn::Instruction imad(unsigned opcode, unsigned d, unsigned a, unsigned b, unsigned c) {
    return op(opcode, {{16, 8, d}, {24, 8, a}, {32, 8, b}, {64, 8, c}, {81, 3, PT}, {87, 4, NOT_PT}});
}
sn::Instruction imad_imm(unsigned d, unsigned a, std::uint32_t imm, unsigned c) {
    return op(0x824, {{16, 8, d}, {24, 8, a}, {32, 32, imm}, {64, 8, c}, {81, 3, PT}, {87, 4, NOT_PT}});
}
sn::Instruction stg128(unsigned a, unsigned data, unsigned displacement) {
    auto i = op(0x386, {{24, 8, a}, {32, 8, data}, {40, 24, displacement}, {72, 1, 1}});
    field(i, 73, 3, 6); field(i, 77, 4, 5); field(i, 84, 3, 1);
    return i;
}
// STS/LDS. type is U8, S8, U16, S16, 32, 64, 128. stride 0..3 scales Ra by
// 1, 4, 8, 16. A uniform load (bit 91) adds URd at bits 32-37.
sn::Instruction sts(unsigned data, unsigned ra, std::uint32_t displacement, unsigned type, unsigned stride = 0) {
    return op(0x388, {{24, 8, ra}, {32, 8, data}, {40, 24, displacement}, {73, 3, type}, {78, 2, stride}});
}
sn::Instruction lds(unsigned dest, unsigned ra, std::uint32_t displacement, unsigned type, unsigned stride = 0, unsigned ur = 0, bool uniform = false) {
    auto i = op(0x984, {{16, 8, dest}, {24, 8, ra}, {40, 24, displacement}, {73, 3, type}, {78, 2, stride}});
    if (uniform) { field(i, 32, 8, ur); field(i, 91, 1, 1); }
    return i;
}
sn::Instruction bar() { return raw(0xb1d); }
sn::Instruction uldc(unsigned ur, unsigned byte_offset, unsigned words) {
    return op(0xab9, {{16, 6, ur}, {73, 3, words == 2 ? 5u : 4u}, {54, 5, 3}, {38, 16, byte_offset}});
}
sn::Instruction mov_ur(unsigned d, unsigned ur) { return op(0xc02, {{16, 8, d}, {32, 32, ur}, {64, 32, 0x08000f00}}); }
sn::Instruction uimad_imm(unsigned d, unsigned a, std::uint32_t imm, unsigned addend, bool is_signed) {
    return op(0x8a5, {{16, 6, d}, {24, 6, a}, {32, 32, imm}, {64, 6, addend}, {73, 1, is_signed}, {81, 3, PT}, {87, 4, NOT_PT}, {91, 1, 1}});
}
sn::Instruction uimad_reg(unsigned d, unsigned a, unsigned b, unsigned addend, bool is_signed) {
    return op(0x2a5, {{16, 6, d}, {24, 6, a}, {32, 6, b}, {64, 6, addend}, {73, 1, is_signed}, {81, 3, PT}, {87, 4, NOT_PT}, {91, 1, 1}});
}
// Equal-to-zero into P0. The predicate accumulator is PT, so the result is the comparison.
sn::Instruction equal_zero(unsigned a) {
    return op(0x20c, {{24, 8, a}, {32, 8, RZ}, {68, 4, PT}, {76, 3, 2}, {81, 3, 0}, {84, 3, PT}, {87, 4, PT}});
}
// P0 selects the true source.
sn::Instruction select(unsigned d, unsigned when_true, unsigned when_false) {
    return op(0x207, {{16, 8, d}, {24, 8, when_true}, {32, 8, when_false}, {87, 3, 0}});
}

// Guest record of `words` registers at group * threads + linear thread.
void publish(std::vector<sn::Instruction>& code, unsigned words) {
    code.insert(code.end(), {
        load(48, 0), load(49, 1), load(50, 2), load(51, 3), load(52, 4), load(53, 5),
        s2r(54, 33), s2r(55, 34), s2r(56, 35), s2r(47, 37),
        imad(0x224, 57, 55, 51, 54), imad(0x224, 57, 56, 53, 57), imad(0x224, 57, 47, 50, 57), imad(0x225, 58, 57, 52, 48)});
    for (unsigned i = 0; i < words; i += 4) code.push_back(stg128(58, i, i * 4));
    code.push_back(exit_instruction());
}
Words inputs(const std::array<std::uint32_t, 3>& size, unsigned record_bytes, std::initializer_list<std::uint32_t> tail = {}) {
    const unsigned threads = size[0] * size[1] * size[2];
    Words w{std::uint32_t(guest_base), std::uint32_t(guest_base >> 32), threads, size[0], record_bytes, size[0] * size[1]};
    w.insert(w.end(), tail.begin(), tail.end());
    return w;
}

// Independent model of the admitted shared accesses: little-endian elements,
// sign extension on the signed element types.
struct Shared {
    std::vector<std::uint8_t> bytes;
    explicit Shared(unsigned n) : bytes(n) {}
    void put(std::uint32_t addr, unsigned type, std::initializer_list<std::uint32_t> words) {
        const unsigned element = type < 2 ? 1u : type < 4 ? 2u : 4u;
        unsigned lane = 0;
        for (auto word : words) {
            for (unsigned b = 0; b < element; ++b) bytes.at(addr + lane * element + b) = std::uint8_t(word >> (8 * b));
            ++lane;
        }
    }
    std::uint32_t get(std::uint32_t addr, unsigned type) const {
        const unsigned element = type < 2 ? 1u : type < 4 ? 2u : 4u;
        std::uint32_t word = 0;
        for (unsigned b = 0; b < element; ++b) word |= std::uint32_t(bytes.at(addr + b)) << (8 * b);
        if (type == 1) return std::uint32_t(std::int32_t(std::int8_t(word)));
        if (type == 3) return std::uint32_t(std::int32_t(std::int16_t(word)));
        return word;
    }
};
Words shared_words(unsigned x) {
    Shared mem(kSharedBytes);
    mem.put(0, 0, {0xA5});
    mem.put(1, 1, {0x80});
    mem.put(2, 1, {0x7F});
    mem.put(4, 2, {0xBEEF});
    mem.put(6, 3, {0x8000});
    mem.put(8, 3, {0x0001});
    mem.put(12, 4, {0x89ABCDEF});
    mem.put(kPositive, 4, {0x10203040});
    mem.put(16, 5, {0x01234567, 0x89ABCDEF});
    mem.put(32, 6, {0x11111111, 0x22222222, 0x33333333, 0x44444444});
    mem.put(kPositive - 4, 4, {0x0E6A7100});
    mem.put(48, 4, {0x51000000});
    // Stride 1, 2 and 3 shift the thread id by 2, 3 and 4.
    mem.put(512 + (x << 2), 4, {x});
    mem.put(768 + (x << 3), 4, {x});
    mem.put(1280 + (x << 4), 4, {x});
    Words out(20);
    out[0] = mem.get(0, 0); out[1] = mem.get(1, 1); out[2] = mem.get(2, 1); out[3] = mem.get(4, 2);
    out[4] = mem.get(6, 3); out[5] = mem.get(8, 3); out[6] = mem.get(12, 4); out[7] = mem.get(kPositive, 4);
    out[8] = mem.get(16, 4); out[9] = mem.get(20, 4);
    out[10] = mem.get(kPositive - 4, 4); out[11] = mem.get(48, 4);
    out[12] = mem.get(32, 4); out[13] = mem.get(36, 4); out[14] = mem.get(40, 4); out[15] = mem.get(44, 4);
    out[16] = mem.get(512 + (x << 2), 4); out[17] = mem.get(768 + (x << 3), 4); out[18] = mem.get(1280 + (x << 4), 4);
    out[19] = mem.get(kUniformOffset + 4, 4);
    return out;
}
std::uint64_t multiply_add(std::uint32_t a, std::uint32_t b, std::uint32_t lo, std::uint32_t hi, bool is_signed) {
    const std::uint64_t product = is_signed
        ? std::uint64_t(std::int64_t(std::int32_t(a)) * std::int64_t(std::int32_t(b)))
        : std::uint64_t(a) * b;
    return product + (std::uint64_t(hi) << 32 | lo);
}

template <typename Expect>
void dispatch_case(MetalHarness& gpu, Fixture fixture, std::array<std::uint32_t, 3> size, unsigned groups, unsigned shared_bytes,
                   const Words& input, unsigned record_words, Expect expect, unsigned& checked, const char* name) {
    fixture.launch(size, shared_bytes);
    auto compiled = gpu.compile_compute(fixture);
    const unsigned threads = size[0] * size[1] * size[2];
    MetalHarness::GuestMemory guest{guest_base, Words(threads * groups * record_words, 0xdeadbeefu), {}};
    gpu.dispatch(compiled, fixture, input, guest, groups);
    for (unsigned group = 0; group < groups; ++group)
        for (unsigned z = 0; z < size[2]; ++z) for (unsigned y = 0; y < size[1]; ++y) for (unsigned x = 0; x < size[0]; ++x) {
            const unsigned linear = (z * size[1] + y) * size[0] + x;
            const unsigned record = group * threads + linear;
            const Words want = expect(x, y, z, group);
            const Words got(guest.words.begin() + record * record_words, guest.words.begin() + (record + 1) * record_words);
            if (got != want) {
                std::fprintf(stderr, "%s size %u,%u,%u thread %u,%u,%u group %u\n", name, size[0], size[1], size[2], x, y, z, group);
                for (unsigned i = 0; i < got.size(); ++i)
                    std::fprintf(stderr, "  [%u] got %08x want %08x\n", i, got[i], want[i]);
                throw std::runtime_error("native shared GPU result");
            }
            ++checked;
        }
    check(guest.table[1] == 0, "compute stores stay inside guest memory");
}
} // namespace
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    unsigned checked = 0;
    // Widths and signs, both offset signs, a scaled thread id and a uniform address.
    std::vector<sn::Instruction> shared{
        move(20, 0xA5), move(21, 0x80), move(22, 0x7F), move(23, 0xBEEF), move(24, 0x8000), move(25, 1),
        move(26, 0x89ABCDEF), move(27, 0x10203040), move(28, 0x01234567), move(29, 0x89ABCDEF),
        move(32, 0x11111111), move(33, 0x22222222), move(34, 0x33333333), move(35, 0x44444444),
        move(36, 0x0E6A7100), move(39, kPositive), move(40, 48), move(41, 0x51000000), s2r(46, 33),
        sts(20, RZ, 0, 0), sts(21, RZ, 1, 1), sts(22, RZ, 2, 1), sts(23, RZ, 4, 2), sts(24, RZ, 6, 3), sts(25, RZ, 8, 3),
        sts(26, RZ, 12, 4), sts(27, RZ, kPositive, 4), sts(28, RZ, 16, 5), sts(32, RZ, 32, 6),
        sts(36, 39, 0xFFFFFC, 4), sts(41, 40, 0, 4),
        sts(46, 46, 512, 4, 1), sts(46, 46, 768, 4, 2), sts(46, 46, 1280, 4, 3),
        uldc(4, 24, 1),
        lds(0, RZ, 0, 0), lds(1, RZ, 1, 1), lds(2, RZ, 2, 1), lds(3, RZ, 4, 2), lds(4, RZ, 6, 3), lds(5, RZ, 8, 3),
        lds(6, RZ, 12, 4), lds(7, RZ, kPositive, 4), lds(8, RZ, 16, 5), lds(10, 39, 0xFFFFFC, 4), lds(11, 40, 0, 4),
        lds(12, RZ, 32, 6), lds(16, 46, 512, 4, 1), lds(17, 46, 768, 4, 2), lds(18, 46, 1280, 4, 3),
        lds(19, RZ, 4, 4, 0, 4, true)};
    publish(shared, 20);
    const auto shared_expect = [](unsigned x, unsigned, unsigned, unsigned) { return shared_words(x); };
    for (const auto& [size, groups] : {std::pair{std::array<std::uint32_t, 3>{32, 1, 1}, 2u}, {{8, 4, 2}, 1u}, {{64, 1, 1}, 1u}})
        dispatch_case(gpu, Fixture(shared, sn::Stage::compute), size, groups, kSharedBytes, inputs(size, 80, {kUniformOffset}), 20,
                      shared_expect, checked, "shared widths");

    // Thread (0,0,0) publishes the sentinel at byte 0. Thread (1,0,0) publishes
    // its packed id at byte 80. The barrier makes both stores visible to every
    // other thread. The address select keeps the barrier on a uniform path.
    std::vector<sn::Instruction> crossed{
        move(4, kSentinel), move(61, 64), move(2, 0), move(3, 0), s2r(60, 32), equal_zero(60),
        imad_imm(62, 60, 16, 61), select(63, RZ, 62), select(64, 4, 60), sts(64, 63, 0, 4), bar(),
        lds(0, RZ, 0, 4), lds(1, RZ, 80, 4)};
    publish(crossed, 4);
    const Words crossed_want{kSentinel, 1u, 0u, 0u};
    for (const auto& [size, groups] : {std::pair{std::array<std::uint32_t, 3>{32, 1, 1}, 2u}, {{8, 4, 2}, 1u}, {{16, 2, 1}, 1u}})
        dispatch_case(gpu, Fixture(crossed, sn::Stage::compute), size, groups, kSharedBytes, inputs(size, 16), 4,
                      [&](unsigned, unsigned, unsigned, unsigned) { return crossed_want; }, checked, "shared barrier");

    // Group ids are the values that are constant across a warp. The dispatch
    // places every group on the X axis, so Y and Z stay zero.
    std::vector<sn::Instruction> groups_id{s2ur(0, 37), s2ur(1, 38), s2ur(2, 39), mov_ur(0, 0), mov_ur(1, 1), mov_ur(2, 2), move(3, 0)};
    publish(groups_id, 4);
    for (const auto& [size, groups] : {std::pair{std::array<std::uint32_t, 3>{8, 4, 2}, 3u}, {{32, 1, 1}, 4u}, {{16, 2, 2}, 1u}})
        dispatch_case(gpu, Fixture(groups_id, sn::Stage::compute), size, groups, 0, inputs(size, 16), 4,
                      [](unsigned, unsigned, unsigned, unsigned group) { return Words{group, 0u, 0u, 0u}; }, checked, "uniform group id");

    // Uniform wide products: two immediate multiplies with a zero addend, an
    // immediate and a register multiply with a uniform pair, and one signed product.
    std::vector<sn::Instruction> wide{
        uldc(0, 24, 2), uldc(2, 32, 2),
        uimad_imm(4, 0, 0x88888889u, URZ, false), uimad_imm(6, 0, 0xAAAAAAABu, URZ, false),
        uimad_imm(8, 0, 7, 2, false), uimad_reg(10, 0, 1, 2, false), uimad_imm(12, 0, 0x88888889u, 2, true),
        mov_ur(0, 4), mov_ur(1, 5), mov_ur(2, 6), mov_ur(3, 7), mov_ur(4, 8), mov_ur(5, 9),
        mov_ur(6, 10), mov_ur(7, 11), mov_ur(8, 12), mov_ur(9, 13), move(10, 0), move(11, 0)};
    publish(wide, 12);
    const std::array<std::array<std::uint32_t, 4>, 6> operands{{
        {3u, 5u, 0u, 0u}, {3u, 1u, 0xffffffffu, 4u}, {0xffffffffu, 0xffffffffu, 1u, 0u},
        {0x80000000u, 2u, 0xffffffffu, 0u}, {0u, 0x12345678u, 0x10u, 0x20u}, {0x12345678u, 0x9abcdef0u, 0x11111111u, 0x222u}}};
    for (const auto& operand : operands) {
        const auto expect = [&](unsigned, unsigned, unsigned, unsigned) {
            Words out(12);
            const auto a = operand[0], b = operand[1], lo = operand[2], hi = operand[3];
            auto store = [&](unsigned at, std::uint64_t value) { out[at] = std::uint32_t(value); out[at + 1] = std::uint32_t(value >> 32); };
            store(0, multiply_add(a, 0x88888889u, 0, 0, false));
            store(2, multiply_add(a, 0xAAAAAAABu, 0, 0, false));
            store(4, multiply_add(a, 7, lo, hi, false));
            store(6, multiply_add(a, b, lo, hi, false));
            store(8, multiply_add(a, 0x88888889u, lo, hi, true));
            return out;
        };
        for (const auto& [size, groups] : {std::pair{std::array<std::uint32_t, 3>{32, 1, 1}, 1u}, {{8, 2, 1}, 2u}}) {
            auto in = inputs(size, 48);
            in.insert(in.end(), operand.begin(), operand.end());
            dispatch_case(gpu, Fixture(wide, sn::Stage::compute), size, groups, 0, in, 12, expect, checked, "uniform wide multiply-add");
        }
    }
    std::printf("Native shared memory Metal execution passed (%u vectors)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
