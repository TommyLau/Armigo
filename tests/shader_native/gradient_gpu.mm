// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed gradient samples. The harness texture is one level with a
// nearest sampler, so an explicit derivative selects level zero and the texel
// is the one that contains the coordinate. Zero and non-zero derivatives
// therefore address the same texel. The model below recovers that texel from
// the coordinate alone. The derivative-order cases sample a two-level texture
// whose level the derivative grouping decides.

#include "metal_harness.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
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
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
float center(unsigned i) { return (float(i) + 0.5f) / 4.0f; }
// Bound handle at bank 3 word 0x10. The register form sets the handle selector
// and leaves that constant field clear. Eviction priority is the ordinary value.
sn::Instruction gradient(unsigned opcode, unsigned d0, unsigned a, unsigned b, unsigned d1, unsigned mask, unsigned dimension) {
    auto i = op(opcode, {{16, 8, d0}, {24, 8, a}, {32, 8, b}, {61, 3, dimension}, {64, 8, d1}, {72, 4, mask}, {81, 3, 7}, {84, 3, 1}});
    if ((opcode >> 8) == 0xb) { field(i, 40, 14, 0x10); field(i, 54, 5, 3); }
    else field(i, 59, 1, 1);
    return i;
}
// Nearest texel of a 4-wide axis. Coordinates in the vectors are texel centers,
// where this is exact; values outside the texture clamp to the edge texel.
unsigned nearest(std::uint32_t encoded) {
    const auto scaled = static_cast<int>(std::bit_cast<float>(encoded) * 4.0f);
    if (scaled < 0) return 0;
    if (scaled > 3) return 3;
    return static_cast<unsigned>(scaled);
}
Pixel texel_bits(unsigned x, unsigned y, unsigned z, unsigned level = 0) {
    const auto t = MetalHarness::texel(x, y, z, level);
    return {bits(t[0]), bits(t[1]), bits(t[2]), bits(t[3])};
}
// Enabled components are packed into R0, R1 and then R2, R3. Disabled lanes stay zero.
Pixel masked(const Pixel& rgba, unsigned mask) {
    Pixel out{};
    unsigned lane = 0;
    for (unsigned component = 0; component < 4; ++component) if (mask & (1u << component)) out[lane++] = rgba[component];
    return out;
}
struct Case {
    std::string name;
    std::vector<sn::Instruction> body;
    std::vector<Words> inputs;
    std::function<Pixel(const Words&)> expected;
    unsigned levels = 1;
};
Fixture program(const Case& c) {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    for (unsigned i = 0; i < 12; ++i) code.push_back(load(8 + i, i));
    code.insert(code.end(), c.body.begin(), c.body.end());
    code.push_back(exit_instruction());
    Fixture f(code);
    put(f.code, 0x48, 0xf);
    return f;
}
Words words_of(std::initializer_list<std::uint32_t> values) {
    Words out(17, 0);
    unsigned i = 0;
    for (auto value : values) out[i++] = value;
    return out;
}
// input[0..] is R8.. . A gradient of 1 is large enough that using it as a
// coordinate would address the edge texel, which these centers do not.
std::vector<Words> with_gradients(std::vector<Words> bases, unsigned first, unsigned count) {
    std::vector<Words> out;
    for (auto base : bases) {
        out.push_back(base);
        for (unsigned i = 0; i < count; ++i) base[first + i] = bits(1.0f);
        out.push_back(base);
    }
    return out;
}
std::vector<Case> cases() {
    std::vector<Case> c;
    std::vector<Words> grid;
    for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 3; ++x)
        grid.push_back(words_of({bits(center(x)), bits(center(y))}));
    // 2D: R8,R9 coordinates, R10..R13 the four derivative pairs.
    c.push_back({"2D explicit derivatives", {gradient(0xb6c, 0, 8, 10, 2, 0xf, 1)},
        with_gradients(grid, 2, 4),
        [](const Words& w) { return texel_bits(nearest(w[0]), nearest(w[1]), 0); }});
    // Three components: red, green and the array layer, so the second destination holds one value.
    std::vector<Words> layers;
    for (unsigned layer = 0; layer < 2; ++layer) for (const auto& base : grid) {
        auto row = base;
        row[2] = layer;
        layers.push_back(row);
    }
    c.push_back({"2D array, layer in the parameter word", {gradient(0xb6c, 0, 8, 11, 2, 0x7, 5)},
        with_gradients(layers, 3, 4),
        [](const Words& w) { return masked(texel_bits(nearest(w[0]), nearest(w[1]), w[2]), 0x7); }});
    std::vector<Words> volume;
    for (unsigned z = 0; z < 2; ++z) for (const auto& base : grid) {
        auto row = base;
        row[2] = bits(center(z));
        volume.push_back(row);
    }
    c.push_back({"3D explicit derivatives", {gradient(0xb6c, 0, 8, 11, 2, 0xf, 2)},
        with_gradients(volume, 3, 6),
        [](const Words& w) { return texel_bits(nearest(w[0]), nearest(w[1]), nearest(w[2])); }});
    // Register handle, then the two coordinates, in R8..R10. Derivatives follow at R11.
    std::vector<Words> handled;
    for (const auto& base : grid) {
        auto row = words_of({});
        row[1] = base[0];
        row[2] = base[1];
        handled.push_back(row);
    }
    c.push_back({"register handle before the coordinates", {gradient(0x36d, 0, 8, 11, 2, 0xf, 1)},
        with_gradients(handled, 3, 4),
        [](const Words& w) { return texel_bits(nearest(w[1]), nearest(w[2]), 0); }});
    // Texel offsets follow the coordinates in one parameter word, four signed
    // bits per axis from bit 16, and move the addressed texel before the
    // edge clamp. Metal takes offsets as constants, so each program moves its
    // parameter word in as an immediate.
    for (auto [dx, dy] : std::initializer_list<std::pair<int, int>>{{1, 0}, {-1, 2}, {2, -1}, {-8, 7}}) {
        auto offset_sample = gradient(0xb6c, 0, 8, 11, 2, 0xf, 1);
        field(offset_sample, 76, 1, 1);
        const auto parameter = (std::uint32_t(dx) & 0xfu) << 16 | (std::uint32_t(dy) & 0xfu) << 20;
        c.push_back({"texel offsets " + std::to_string(dx) + ", " + std::to_string(dy), {move(10, parameter), offset_sample},
            with_gradients(grid, 3, 4),
            [dx, dy](const Words& w) {
                const auto axis = [](unsigned at, int by) { return unsigned(std::clamp(int(at) + by, 0, 3)); };
                return texel_bits(axis(nearest(w[0]), dx), axis(nearest(w[1]), dy), 0);
            }});
    }
    // The block holds d/dx then d/dy of each coordinate in turn. With the
    // four words (t, t, 0, 0) the screen-axis vectors are (t, 0) and (t, 0),
    // a footprint of 4t = 2^0.25 texels and level 0; (t, 0, t, 0) makes
    // (t, t) and (0, 0), 2^0.75 texels and level 1. Grouping the words by
    // axis instead would swap the two levels.
    const float t = std::exp2(0.25f) / 4.0f;
    for (bool spread : {false, true}) {
        std::vector<Words> rows;
        for (auto row : grid) {
            row[2] = bits(t);
            row[spread ? 4 : 3] = bits(t);
            rows.push_back(row);
        }
        c.push_back({spread ? "derivatives of both coordinates along x" : "derivatives of u along x and y",
            {gradient(0xb6c, 0, 8, 10, 2, 0xf, 1)}, rows,
            [spread](const Words& w) {
                return spread ? texel_bits(nearest(w[0]) / 2, nearest(w[1]) / 2, 0, 1) : texel_bits(nearest(w[0]), nearest(w[1]), 0);
            }, 2});
    }
    return c;
}
} // namespace
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    unsigned checked = 0;
    for (const auto& test : cases()) {
        gpu.mip_levels = test.levels;
        auto compiled = gpu.compile(program(test));
        for (const auto& input : test.inputs) {
            const auto got = gpu.run(compiled, input), want = test.expected(input);
            if (got != want) {
                std::fprintf(stderr, "%s: got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n", test.name.c_str(),
                    got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                throw std::runtime_error("native gradient GPU result");
            }
            ++checked;
        }
    }
    std::printf("Native gradient Metal execution passed (%u vectors)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
