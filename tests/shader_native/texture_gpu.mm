// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed native texture forms: bound, register-handle and
// uniform-handle sampling, the packed operand split, destination pairs,
// gather and dimension queries over textures whose texels encode their own
// coordinates.

#include "metal_harness.h"
#include <algorithm>
#include <cmath>
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
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
float center(unsigned i) { return (float(i) + 0.5f) / 4.0f; }
// Texture with packed operands: d0 16-23, A 24-31, B 32-39, d1 64-71, mask
// 72-75, dimension 61-63, LOD mode 87-89, handle word 3:0x40 when bound.
sn::Instruction sample(unsigned opcode, unsigned d0, unsigned a, unsigned b, unsigned d1, unsigned mask, unsigned dimension,
                       unsigned lod = 0, std::initializer_list<Field> extra = {}) {
    auto i = op(opcode, {{16, 8, d0}, {24, 8, a}, {32, 8, b}, {60, 1, 1}, {61, 3, dimension}, {64, 8, d1}, {72, 4, mask}, {81, 3, 7}, {87, 3, lod}});
    if ((opcode >> 8) == 0xb) { field(i, 40, 14, 0x10); field(i, 54, 5, 3); }
    for (auto [first, count, value] : extra) field(i, first, count, value);
    return i;
}
struct Case {
    std::string name;
    std::vector<sn::Instruction> body;
    std::vector<Words> inputs;
    std::function<Pixel(const Words&)> expected;
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
std::uint32_t alpha(unsigned x, unsigned y, unsigned z) { return bits(MetalHarness::texel(x, y, z)[3]); }
std::vector<Case> cases() {
    std::vector<Case> c;
    std::vector<Words> grid;
    for (unsigned y = 0; y < 4; y += 3) for (unsigned x = 0; x < 4; ++x)
        grid.push_back({bits(center(x)), bits(center(y)), bits(center((x + y) % 4)), 0, 0, 0, 0, 0});
    auto x_of = [](const Words& w) { return unsigned(std::bit_cast<float>(w[0]) * 4); };
    auto y_of = [](const Words& w) { return unsigned(std::bit_cast<float>(w[1]) * 4); };
    auto z_of = [](const Words& w) { return unsigned(std::bit_cast<float>(w[2]) * 4); };
    c.push_back({"2D implicit, two destination pairs", {sample(0xb60, 0, 8, 9, 2, 0xf, 1)}, grid,
        [&](const Words& w) { return Pixel{bits(float(x_of(w))), bits(float(y_of(w))), 0, alpha(x_of(w), y_of(w), 0)}; }});
    // .F16: two results to a register as binary16 halves; the texels here
    // are small integers, which binary16 holds exactly.
    auto half = [](float v) {
        if (v == 0.0f) return 0u;
        int exponent; std::frexp(v, &exponent);
        const auto mantissa = std::uint32_t(std::ldexp(v, 11 - exponent));
        return std::uint32_t(exponent - 1 + 15) << 10 | (mantissa & 0x3ff);
    };
    c.push_back({"2D implicit, halves to two destinations", {sample(0xb60, 0, 8, 9, 2, 0xf, 1, 0, {{79, 1, 1}})}, grid,
        [&](const Words& w) {
            const auto t = MetalHarness::texel(x_of(w), y_of(w), 0);
            return Pixel{half(t[0]) | half(t[1]) << 16, 0, half(t[2]) | half(t[3]) << 16, 0};
        }});
    c.push_back({"2D implicit, three halves to consecutive registers", {sample(0xb60, 0, 8, 9, 255, 0x7, 1, 0, {{79, 1, 1}})}, grid,
        [&](const Words& w) {
            const auto t = MetalHarness::texel(x_of(w), y_of(w), 0);
            return Pixel{half(t[0]) | half(t[1]) << 16, half(t[2]), 0, 0};
        }});
    // RZ as the first destination discards only the first packed pair.
    c.push_back({"halves with the first destination discarded", {sample(0xb60, 255, 8, 9, 2, 0xf, 1, 0, {{79, 1, 1}})}, grid,
        [&](const Words& w) {
            const auto t = MetalHarness::texel(x_of(w), y_of(w), 0);
            return Pixel{0, 0, half(t[2]) | half(t[3]) << 16, 0};
        }});
    c.push_back({"2D explicit LOD in the B range", {sample(0xb60, 0, 8, 10, 2, 0x7, 1, 3)},
        [&] { auto v = grid; for (auto& w : v) w[2] = 0; return v; }(),
        [&](const Words& w) { return Pixel{bits(float(x_of(w))), bits(float(y_of(w))), 0, 0}; }});
    c.push_back({"3D coordinates split two and one", {sample(0xb60, 0, 8, 10, 2, 0xd, 2)}, grid,
        [&](const Words& w) { return Pixel{bits(float(x_of(w))), bits(float(z_of(w))), alpha(x_of(w), y_of(w), z_of(w)), 0}; }});
    // Texel offsets (bit 76): one register after the level, a signed
    // four-bit field an axis from bit 0, moving the addressed texel before
    // the edge clamp. Metal samples only at constant offsets, so each
    // program moves its offsets in as an immediate.
    auto shifted = [](unsigned at, int by) { return unsigned(std::clamp(int(at) + by, 0, 3)); };
    for (auto [dx, dy] : std::initializer_list<std::pair<int, int>>{{1, -1}, {-8, 7}}) {
        const auto packed = (std::uint32_t(dx) & 0xfu) | (std::uint32_t(dy) & 0xfu) << 4;
        c.push_back({"2D texel offsets " + std::to_string(dx) + ", " + std::to_string(dy),
            {move(12, packed), sample(0xb60, 0, 8, 12, 2, 0xf, 1, 0, {{76, 1, 1}})}, grid,
            [=](const Words& w) {
                const auto x = shifted(x_of(w), dx), y = shifted(y_of(w), dy);
                return Pixel{bits(float(x)), bits(float(y)), 0, alpha(x, y, 0)};
            }});
    }
    // Unpacked (bit 60 clear): the coordinates fill A, and the level and the
    // offsets (1, -1, 1) follow in B.
    c.push_back({"3D unpacked, explicit level and offsets",
        {move(12, 0), move(13, 0x1f1), sample(0xb60, 0, 8, 12, 2, 0xf, 2, 3, {{60, 1, 0}, {76, 1, 1}})}, grid,
        [=](const Words& w) {
            const auto x = shifted(x_of(w), 1), y = shifted(y_of(w), -1), z = shifted(z_of(w), 1);
            return Pixel{bits(float(x)), bits(float(y)), bits(float(z)), alpha(x, y, z)};
        }});
    c.push_back({"2D array layer leads the list", {sample(0xb60, 0, 13, 10, 255, 0x5, 5)},
        [&] { auto v = grid; for (unsigned i = 0; i < v.size(); ++i) { v[i][5] = i % 4; v[i][6] = v[i][0]; v[i][2] = v[i][1]; } return v; }(),
        [&](const Words& w) { return Pixel{bits(float(x_of(Words{w[6]}))), bits(float(w[5])), 0, 0}; }});
    c.push_back({"register handle follows the coordinates", {sample(0x361, 0, 8, 12, 255, 0x3, 1, 0, {{59, 1, 1}})}, grid,
        [&](const Words& w) { return Pixel{bits(float(x_of(w))), bits(float(y_of(w))), 0, 0}; }});
    c.push_back({"uniform handle", {op(0xab9, {{16, 6, 4}, {73, 3, 4}, {54, 5, 3}, {38, 16, 20}}),
                                    sample(0xf60, 0, 8, 9, 255, 0x9, 1, 0, {{40, 6, 4}, {91, 1, 1}})}, grid,
        [&](const Words& w) { return Pixel{bits(float(x_of(w))), alpha(x_of(w), y_of(w), 0), 0, 0}; }});
    std::vector<Words> corners;
    for (unsigned y = 0; y < 3; ++y) for (unsigned x = 0; x < 3; ++x) corners.push_back({bits(float(x + 1) / 4), bits(float(y + 1) / 4), 0, 0, 0, 0, 0, 0});
    // Gather order: (x, y+1), (x+1, y+1), (x+1, y), (x, y) for the corner texels.
    c.push_back({"gather alpha", {sample(0xb63, 0, 8, 9, 2, 0xf, 1, 0, {{87, 3, 3}})}, corners,
        [](const Words& w) {
            const unsigned x = unsigned(std::bit_cast<float>(w[0]) * 4) - 1, y = unsigned(std::bit_cast<float>(w[1]) * 4) - 1;
            return Pixel{alpha(x, y + 1, 0), alpha(x + 1, y + 1, 0), alpha(x + 1, y, 0), alpha(x, y, 0)};
        }});
    c.push_back({"gather red", {sample(0xb63, 0, 8, 9, 2, 0xf, 1, 0, {{87, 3, 0}})}, corners,
        [](const Words& w) {
            const unsigned x = unsigned(std::bit_cast<float>(w[0]) * 4) - 1;
            return Pixel{bits(float(x)), bits(float(x + 1)), bits(float(x + 1)), bits(float(x))};
        }});
    // Gather offsets, moved in as immediates: one offset (1, -1), signed 6-bit
    // fields at bits 0 and 8, moves the footprint; per-texel offsets (two
    // registers, a byte a coordinate) give texel k the footprint corner at
    // offset k: (1, 0), (0, 1), (-1, -1) and (2, 1). Coordinates clamp to the
    // edge.
    auto clamped = [](int at) { return unsigned(std::clamp(at, 0, 3)); };
    // .F16 gather packs the four components into two registers, the same pairs a sample uses.
    auto packed_gather = [&](unsigned component, auto lane) {
        return [=](const Words& w) {
            const unsigned x = unsigned(std::bit_cast<float>(w[0]) * 4) - 1, y = unsigned(std::bit_cast<float>(w[1]) * 4) - 1;
            const auto a = lane(x, y + 1), b = lane(x + 1, y + 1), c = lane(x + 1, y), d = lane(x, y);
            (void)component;
            return Pixel{a | (b << 16), 0, c | (d << 16), 0};
        };
    };
    auto red_half = [&](unsigned x, unsigned) { return half(float(x)); };
    auto green_half = [&](unsigned, unsigned y) { return half(float(y)); };
    auto blue_half = [&](unsigned, unsigned) { return half(0.0f); };
    c.push_back({"gather red halves", {sample(0xb63, 0, 8, 9, 2, 0xf, 1, 0, {{79, 1, 1}, {87, 3, 0}})}, corners, packed_gather(0, red_half)});
    c.push_back({"gather green halves", {sample(0xb63, 0, 8, 9, 2, 0xf, 1, 0, {{79, 1, 1}, {87, 3, 1}})}, corners, packed_gather(1, green_half)});
    c.push_back({"gather blue halves", {sample(0xb63, 0, 8, 9, 2, 0xf, 1, 0, {{79, 1, 1}, {87, 3, 2}})}, corners, packed_gather(2, blue_half)});
    c.push_back({"gather red halves with per-texel offsets",
        {move(12, 0x01000001u), move(13, 0x01023f3fu), sample(0xb63, 0, 8, 12, 2, 0xf, 1, 0, {{76, 2, 2}, {79, 1, 1}, {87, 3, 0}})}, corners,
        [=](const Words& w) {
            const int x = int(std::bit_cast<float>(w[0]) * 4) - 1, y = int(std::bit_cast<float>(w[1]) * 4) - 1;
            const auto lane = [&](int ox, int oy) { return half(float(clamped(ox))); };
            // Offsets (1, 0), (0, 1), (-1, -1), (2, 1) select the red of each moved corner.
            return Pixel{lane(x + 1, y) | (lane(x, y + 1) << 16), 0, lane(x - 1, y - 1) | (lane(x + 2, y + 1) << 16), 0};
        }});
    c.push_back({"gather alpha with an offset", {move(12, 0x3f01), sample(0xb63, 0, 8, 12, 2, 0xf, 1, 0, {{76, 2, 1}, {87, 3, 3}})}, corners,
        [=](const Words& w) {
            const int x = int(std::bit_cast<float>(w[0]) * 4) - 1 + 1, y = int(std::bit_cast<float>(w[1]) * 4) - 1 - 1;
            return Pixel{alpha(clamped(x), clamped(y + 1), 0), alpha(clamped(x + 1), clamped(y + 1), 0),
                         alpha(clamped(x + 1), clamped(y), 0), alpha(clamped(x), clamped(y), 0)};
        }});
    c.push_back({"gather alpha with per-texel offsets",
        {move(12, 0x01000001u), move(13, 0x01023f3fu), sample(0xb63, 0, 8, 12, 2, 0xf, 1, 0, {{76, 2, 2}, {87, 3, 3}})}, corners,
        [=](const Words& w) {
            const int x = int(std::bit_cast<float>(w[0]) * 4) - 1, y = int(std::bit_cast<float>(w[1]) * 4) - 1;
            return Pixel{alpha(clamped(x + 1), clamped(y), 0), alpha(clamped(x), clamped(y + 1), 0),
                         alpha(clamped(x - 1), clamped(y - 1), 0), alpha(clamped(x + 2), clamped(y + 1), 0)};
        }});
    // The same four offsets, loaded from a constant-bank word per pixel
    // rather than moved in as an immediate. Each texel is the footprint
    // corner at its own signed 6-bit byte pair.
    auto sext6 = [](std::uint32_t word, unsigned byte) {
        const int field = int((word >> (8 * byte)) & 0x3fu);
        return field & 0x20 ? field - 64 : field;
    };
    c.push_back({"gather alpha with register per-texel offsets",
        {sample(0xb63, 0, 8, 12, 2, 0xf, 1, 0, {{76, 2, 2}, {87, 3, 3}})},
        [&] {
            std::vector<Words> v;
            const std::uint32_t patterns[][2] = {{0x01000001u, 0x01023f3fu}, {0x033e0000u, 0x3f013807u}};
            for (unsigned y = 0; y < 3; ++y) for (unsigned x = 0; x < 3; ++x)
                for (const auto& pattern : patterns)
                    v.push_back({bits(float(x + 1) / 4.0f), bits(float(y + 1) / 4.0f), 0, 0, pattern[0], pattern[1], 0, 0});
            return v;
        }(),
        [=](const Words& w) {
            const int x = int(std::bit_cast<float>(w[0]) * 4) - 1, y = int(std::bit_cast<float>(w[1]) * 4) - 1;
            Pixel out{};
            for (unsigned texel = 0; texel < 4; ++texel) {
                const auto word = w[4 + texel / 2];
                const unsigned base = (texel % 2) * 2;
                out[texel] = alpha(clamped(x + sext6(word, base)), clamped(y + sext6(word, base + 1)), 0);
            }
            return out;
        }});
    // TMML (no fault predicate field): constant coordinates give zero
    // derivatives, so the clamped LOD of a single-level texture is zero.
    c.push_back({"LOD query", {op(0xb69, {{16, 8, 0}, {24, 8, 8}, {32, 8, 255}, {40, 14, 0x10}, {54, 5, 3}, {61, 3, 1}, {64, 8, 255}, {72, 4, 2}}),
                               sample(0xb60, 1, 8, 9, 255, 0x1, 1)}, grid,
        [&](const Words& w) { return Pixel{0, bits(float(x_of(w))), 0, 0}; }});
    // TMML encoded as 2D on a handle the sample proves a 2D array: the LOD
    // does not depend on the layer, so the query takes the array shape.
    c.push_back({"LOD query of an array texture", {op(0xb69, {{16, 8, 0}, {24, 8, 13}, {32, 8, 255}, {40, 14, 0x10}, {54, 5, 3}, {61, 3, 1}, {64, 8, 255}, {72, 4, 2}}),
                                                  sample(0xb60, 1, 13, 10, 255, 0x1, 5)},
        [&] { auto v = grid; for (unsigned i = 0; i < v.size(); ++i) { v[i][5] = i % 4; v[i][6] = v[i][0]; v[i][2] = v[i][1]; } return v; }(),
        [&](const Words& w) { return Pixel{0, bits(float(x_of(Words{w[6]}))), 0, 0}; }});
    // TXQ width, height and level count; the sample proves the handle's shape.
    c.push_back({"dimension query", {op(0xb6f, {{16, 8, 0}, {24, 8, 13}, {40, 14, 0x10}, {54, 5, 3}, {64, 8, 2}, {72, 4, 0xb}}),
                                     sample(0xb60, 3, 8, 9, 255, 0x1, 1)}, grid,
        [&](const Words& w) { return Pixel{4, 4, 1, bits(float(x_of(w)))}; }});
    // TLD: integer texel coordinates at level zero (LZ, mode 1) or an
    // explicit level (LL, mode 3) after them; the array layer leads.
    std::vector<Words> texels;
    for (unsigned y = 0; y < 4; y += 3) for (unsigned x = 0; x < 4; ++x) texels.push_back({x, y, (x + y) % 4, 0, 0, 0, 0, 0});
    c.push_back({"TLD 2D at level zero", {sample(0xb66, 0, 8, 9, 2, 0xf, 1, 1)}, texels,
        [](const Words& w) { return Pixel{bits(float(w[0])), bits(float(w[1])), 0, alpha(w[0], w[1], 0)}; }});
    // Texel offsets after the coordinates (bit 76), a signed four-bit field
    // an axis, moved in as an immediate (1, -2): each texel reads an
    // in-range neighbour.
    c.push_back({"TLD 2D at level zero with offsets", {move(12, 0xe1), sample(0xb66, 0, 8, 12, 2, 0xf, 1, 1, {{76, 1, 1}})},
        [] {
            std::vector<Words> v;
            for (std::uint32_t y = 2; y < 4; ++y) for (std::uint32_t x = 0; x < 3; ++x) v.push_back({x, y, 0, 0, 0, 0, 0, 0});
            return v;
        }(),
        [](const Words& w) { return Pixel{bits(float(w[0] + 1)), bits(float(w[1] - 2)), 0, alpha(w[0] + 1, w[1] - 2, 0)}; }});
    // The offset word is loaded per pixel. Its low nibbles are signed
    // four-bit x and y. A coordinate that leaves the image reads zero.
    c.push_back({"TLD 2D with a register offset", {sample(0xb66, 0, 8, 12, 2, 0xf, 1, 1, {{76, 1, 1}})},
        {Words{1, 2, 0, 0, 0xf1, 0, 0, 0}, Words{0, 1, 0, 0, 0xf, 0, 0, 0},
         Words{3, 0, 0, 0, 0x1, 0, 0, 0}, Words{2, 2, 0, 0, 0, 0, 0, 0}},
        [](const Words& w) {
            auto nibble = [](std::uint32_t word, unsigned shift) {
                const int field = int((word >> shift) & 0xfu);
                return field >= 8 ? field - 16 : field;
            };
            const int x = int(w[0]) + nibble(w[4], 0), y = int(w[1]) + nibble(w[4], 4);
            if (x < 0 || y < 0 || x > 3 || y > 3) return Pixel{0, 0, 0, 0};
            return Pixel{bits(float(x)), bits(float(y)), 0, alpha(unsigned(x), unsigned(y), 0)};
        }});
    c.push_back({"TLD 2D at an explicit level", {sample(0xb66, 0, 8, 10, 255, 0x3, 1, 3)},
        [&] { auto v = texels; for (auto& w : v) w[2] = 0; return v; }(),
        [](const Words& w) { return Pixel{bits(float(w[0])), bits(float(w[1])), 0, 0}; }});
    c.push_back({"TLD 2D array, the layer leading", {sample(0xb66, 0, 13, 10, 255, 0x9, 5, 1)},
        [&] { auto v = texels; for (auto& w : v) { w[5] = w[2]; w[6] = w[0]; w[2] = w[1]; } return v; }(),
        [](const Words& w) { return Pixel{bits(float(w[6])), alpha(w[6], w[2], w[5]), 0, 0}; }});
    c.push_back({"TLD 3D", {sample(0xb66, 0, 8, 10, 2, 0xd, 2, 1)}, texels,
        [](const Words& w) { return Pixel{bits(float(w[0])), bits(float(w[2])), alpha(w[0], w[1], w[2]), 0}; }});
    c.push_back({"TLD register handle after the coordinates", {sample(0x367, 0, 8, 9, 255, 0x3, 1, 1, {{59, 1, 1}})}, texels,
        [](const Words& w) { return Pixel{bits(float(w[0])), bits(float(w[1])), 0, 0}; }});
    return c;
}
}
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    unsigned checked = 0;
    for (const auto& test : cases()) {
        auto compiled = gpu.compile(program(test));
        for (const auto& input : test.inputs) {
            const auto got = gpu.run(compiled, input), want = test.expected(input);
            if (got != want) {
                std::fprintf(stderr, "%s: input %08x %08x %08x: got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n", test.name.c_str(),
                    input[0], input[1], input[2], got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                throw std::runtime_error("native texture GPU result");
            }
            ++checked;
        }
    }
    // A handle read from guest memory at a constant-buffer address (LDG at
    // c3 words 0-1 plus 0x20) is the host's (a handle program): the harness
    // evaluates it and binds the pattern only for texture 5 and sampler 1,
    // so another texture, another sampler, swapped halves or a word read at
    // another address sample zeros.
    {
        auto ldg = op(0x381, {{16, 8, 12}, {24, 8, 14}, {40, 24, 0x20}, {72, 1, 1}, {73, 3, 4}, {77, 4, 5}, {81, 3, 7}, {84, 3, 1}});
        Fixture pooled({move(0, 0), move(1, 0), move(2, 0), move(3, 0), load(14, 0), load(15, 1), ldg, load(8, 2), load(9, 3),
                        sample(0x361, 0, 8, 12, 255, 0x3, 1, 0, {{59, 1, 1}}), exit_instruction()});
        put(pooled.code, 0x48, 0xf);
        const auto compiled = gpu.compile(pooled);
        check(compiled.artifact.reflection.handle_programs.size() == 1, "the handle is a handle program");
        constexpr std::uint64_t guest_base = 0x7d00001000ull;
        gpu.handle_texture = 5; gpu.handle_sampler = 1;
        for (const auto [word, named] : std::initializer_list<std::pair<std::uint32_t, bool>>{
                 {0x00100005u, true}, {0x00100006u, false}, {0x00200005u, false}, {0x00500001u, false}}) {
            MetalHarness::GuestMemory guest{guest_base, Words(64, 0), {}};
            guest.words[8] = word;          // the handle at +0x20
            // Beside it, the opposite outcome: a word read at another address fails.
            guest.words[7] = guest.words[9] = named ? 0u : 0x00100005u;
            for (unsigned y = 0; y < 4; y += 3) for (unsigned x = 0; x < 4; ++x) {
                const auto got = gpu.run(compiled, {std::uint32_t(guest_base), std::uint32_t(guest_base >> 32), bits(center(x)), bits(center(y))}, &guest);
                const Pixel want = named ? Pixel{bits(float(x)), bits(float(y)), 0, 0} : Pixel{0, 0, 0, 0};
                if (got != want) {
                    std::fprintf(stderr, "handle program texture %08x: texel %u,%u got %08x %08x\n", word, x, y, got[0], got[1]);
                    throw std::runtime_error("native handle program texture GPU result");
                }
                ++checked;
            }
        }
        gpu.handle_texture = gpu.handle_sampler = 0;
    }
    // IPA of the front-facing attribute: zero for the harness triangle under
    // clockwise front faces, a mask of all ones under counter-clockwise ones.
    {
        Fixture facing({move(0, 5), move(1, 0), move(2, 0), move(3, 0),
                        op(0x326, {{16, 8, 0}, {32, 8, 255}, {64, 8, 0xff}, {78, 1, 1}, {81, 3, 7}}), exit_instruction()});
        put(facing.code, 0x48, 0xf);
        const auto compiled = gpu.compile(facing);
        for (const auto winding : {MTLWindingClockwise, MTLWindingCounterClockwise}) {
            gpu.front_winding = winding;
            const auto got = gpu.run(compiled, {});
            if (got[0] != (winding == MTLWindingCounterClockwise ? 0xffffffffu : 0u)) throw std::runtime_error("front-facing mask");
            ++checked;
        }
        gpu.front_winding = MTLWindingClockwise;
    }
    std::printf("Native texture Metal execution passed (%u vectors)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
