// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed native special-register, vote, R2UR and P2R forms.
// Expected words come from the instruction contracts: a per-invocation
// uniform copy, a predicate-byte gather, lane masks derived from the lane
// id, and subgroup all/any/equal plus the ballot bit of the invoking lane.

#include "metal_harness.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
using Words = std::vector<std::uint32_t>;
using Pixel = std::array<std::uint32_t, 4>;
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
constexpr unsigned PT = 7, NOT_PT = 15;
constexpr unsigned kPredicates = 0x55;

sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
sn::Instruction load(unsigned d, unsigned index) { return op(0xa02, {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}}); }
sn::Instruction isetp(unsigned a, unsigned b, unsigned relation, unsigned first) {
    return op(0x20c, {{24, 8, a}, {32, 8, b}, {68, 4, PT}, {76, 3, relation}, {81, 3, first}, {84, 3, PT}, {87, 4, PT}});
}
sn::Instruction r2ur(unsigned uniform, unsigned ra) { return op(0x3c2, {{16, 6, uniform}, {24, 8, ra}, {81, 3, 7}}); }
sn::Instruction p2r(unsigned d, unsigned ra, std::uint32_t mask, unsigned byte) {
    return op(0x803, {{16, 8, d}, {24, 8, ra}, {32, 32, mask}, {76, 2, byte}});
}
sn::Instruction p2r_reg(unsigned d, unsigned ra, unsigned rb, unsigned byte) {
    return op(0x203, {{16, 8, d}, {24, 8, ra}, {32, 8, rb}, {76, 2, byte}});
}
sn::Instruction s2r(unsigned d, unsigned selector) { return op(0x919, {{16, 8, d}, {72, 8, selector}}); }
sn::Instruction vote(unsigned opcode, unsigned ballot, unsigned mode, unsigned pred, unsigned src) {
    return op(opcode, {{16, opcode == 0x886 ? 6u : 8u, ballot}, {72, 2, mode}, {81, 3, pred}, {87, 4, src}});
}
sn::Instruction umov(unsigned uniform, std::uint32_t imm) {
    auto i = raw(0x882); field(i, 16, 6, uniform); i.words[1] = imm; return i;
}
sn::Instruction mov_ur(unsigned dest, unsigned uniform) {
    auto i = raw(0xc02); field(i, 16, 8, dest); i.words[1] = uniform; i.words[2] = 0x08000f00; return i;
}
sn::Instruction mov_reg(unsigned dest, unsigned source) {
    auto i = raw(0x202); field(i, 16, 8, dest); i.words[1] = source; i.words[2] = 0xf00; return i;
}
sn::Instruction usel(unsigned dest, unsigned pred, unsigned a, unsigned b) {
    return op(0x287, {{16, 6, dest}, {24, 6, a}, {32, 6, b}, {87, 3, pred}, {91, 1, 1}});
}
sn::Instruction uadd3(unsigned dest, unsigned a, unsigned b, unsigned c) {
    return op(0x290, {{16, 6, dest}, {24, 6, a}, {32, 6, b}, {64, 6, c}, {91, 1, 1}, {77, 4, NOT_PT}, {81, 3, PT}, {84, 3, PT}, {87, 4, NOT_PT}});
}
// Nonzero result of Ra & 1, written to a predicate. The data result is discarded.
sn::Instruction lane_bit(unsigned pred, unsigned ra) {
    return op(0x812, {{16, 8, 255}, {24, 8, ra}, {32, 32, 1}, {64, 8, 255}, {72, 8, 0xc0}, {81, 3, pred}, {87, 4, NOT_PT}});
}

std::uint32_t gather(std::uint32_t ra, std::uint32_t mask, unsigned byte, unsigned predicates) {
    const std::uint32_t selected = (mask & 0xffu) << (8 * byte);
    std::uint32_t insert = 0;
    for (unsigned bit = 0; bit < 7; ++bit) if (predicates & (1u << bit)) insert |= 1u << (bit + 8 * byte);
    return (ra & ~selected) | (insert & selected);
}
std::vector<sn::Instruction> predicate_bits(unsigned mask) {
    std::vector<sn::Instruction> code;
    for (unsigned bit = 0; bit < 7; ++bit) code.push_back(isetp(255, 255, (mask & (1u << bit)) ? 7 : 0, bit));
    return code;
}
void append(std::vector<sn::Instruction>& code, std::vector<sn::Instruction> more) {
    code.insert(code.end(), more.begin(), more.end());
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
    Fixture f(code, sn::Stage::fragment); put(f.code, 0x48, 0xf);
    return f;
}
void expect(const char* name, Pixel got, Pixel want) {
    if (got == want) return;
    std::fprintf(stderr, "%s: got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n", name,
        got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
    throw std::runtime_error(name);
}
void words(const char* name, sn::Instruction i, std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    if (i.words[0] == a && i.words[1] == b && i.words[2] == c && i.words[3] == 0) return;
    std::fprintf(stderr, "%s encodes %08x %08x %08x %08x\n", name, i.words[0], i.words[1], i.words[2], i.words[3]);
    throw std::runtime_error(name);
}
void refused(const char* name, Fixture f, std::string_view text) {
    auto result = sn::translate_native(f.program());
    const auto* error = std::get_if<ir::Diagnostic>(&result);
    if (!error || error->message.find(text) == std::string::npos) {
        std::fprintf(stderr, "%s: %s\n", name, error ? error->message.c_str() : "translated");
        throw std::runtime_error(name);
    }
}
Fixture fragment_of(std::vector<sn::Instruction> code) {
    code.push_back(exit_instruction());
    return Fixture(code, sn::Stage::fragment);
}

// Low two lane bits follow the 2x2 fragment quad: horizontal neighbours differ
// by lane xor 1 and vertical neighbours by lane xor 2.
void check_quad_lanes(const char* name, const std::vector<Pixel>& pixels, unsigned component) {
    auto lane = [&](unsigned x, unsigned y) { return pixels[y * 4 + x][component]; };
    for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
        const auto id = lane(x, y), horizontal = lane(x ^ 1, y), vertical = lane(x, y ^ 1);
        if (id > 31 || (id ^ horizontal) != 1u || (id ^ vertical) != 2u) {
            std::fprintf(stderr, "%s lane at (%u, %u): %u, horizontal %u, vertical %u\n", name, x, y, id, horizontal, vertical);
            throw std::runtime_error(name);
        }
    }
}
std::uint32_t lane_bit(unsigned lane) { return 1u << lane; }
} // namespace

int main() {
try {
    MetalHarness gpu;
    unsigned checked = 0;
    words("r2ur corpus", r2ur(4, 38), 0x260473c2u, 0, 0x000e0000u);
    words("p2r corpus", p2r(54, 255, 0x40, 0), 0xff367803u, 0x40u, 0);
    words("voteu any p0", vote(0x886, 63, 1, 0, 0), 0x003f7886u, 0, 0x00000100u);
    words("voteu all ur4", vote(0x886, 4, 0, 7, 7), 0x00047886u, 0, 0x038e0000u);
    words("voteu any ur6", vote(0x886, 6, 1, 7, 7), 0x00067886u, 0, 0x038e0100u);
    words("voteu any p1", vote(0x886, 63, 1, 0, 1), 0x003f7886u, 0, 0x00800100u);
    words("voteu any p3", vote(0x886, 63, 1, 0, 3), 0x003f7886u, 0, 0x01800100u);
    checked += 7;

    refused("r2ur extension", fragment_of({op(0x3c2, {{16, 6, 4}, {22, 2, 1}, {24, 8, 8}, {81, 3, 7}})}), "uniform destination field extension");
    refused("r2ur predicate", fragment_of({op(0x3c2, {{16, 6, 4}, {24, 8, 8}, {81, 3, 1}})}), "predicate result is not admitted");
    refused("r2ur reserved", fragment_of({op(0x3c2, {{16, 6, 4}, {24, 8, 8}, {80, 1, 1}, {81, 3, 7}})}), "reserved bits are not admitted");
    refused("p2r reserved", fragment_of({op(0x803, {{16, 8, 0}, {24, 8, 255}, {32, 32, 1}, {78, 1, 1}})}), "reserved bits are not admitted");
    refused("s2r lane in vertex", Fixture({s2r(0, 0), exit_instruction()}, sn::Stage::vertex), "lane id requires a fragment or workgroup stage");
    refused("s2r mask in vertex", Fixture({s2r(0, 56), exit_instruction()}, sn::Stage::vertex), "lane mask requires a fragment or workgroup stage");
    refused("s2r thread id", fragment_of({s2r(0, 0x21)}), "thread ids require a workgroup stage");
    refused("s2r workgroup id", fragment_of({s2r(0, 0x25)}), "group ids require a compute stage");
    refused("s2r other register", Fixture({s2r(0, 0x24), exit_instruction()}, sn::Stage::compute), "special register is not admitted");
    refused("s2r packed thread id", fragment_of({s2r(0, 32)}), "thread ids require a workgroup stage");
    refused("s2r w scale", fragment_of({s2r(0, 30)}), "special register is not admitted");
    refused("s2r variable rate", fragment_of({s2r(0, 0x84)}), "shading rate requires proven disabled pixel-rate state");
    refused("s2r reserved", fragment_of({op(0x919, {{16, 8, 0}, {72, 8, 0}, {80, 1, 1}})}), "reserved bits are not admitted");
    refused("vote mode", fragment_of({vote(0x806, 255, 3, 7, 7)}), "vote mode is not admitted");
    refused("vote in vertex", Fixture({vote(0x806, 0, 0, 7, 7), exit_instruction()}, sn::Stage::vertex), "vote requires a fragment or workgroup stage");
    refused("voteu extension", fragment_of({op(0x886, {{16, 6, 4}, {22, 2, 1}, {72, 2, 1}, {81, 3, 7}, {87, 4, 7}})}), "uniform destination field extension");
    checked += 15;

    const Words edge{0, 1, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x12345678u, 0xfedcba98u, 0x40u};
    std::vector<Words> inputs;
    for (std::size_t r = 0; r < 4; ++r) {
        Words w(8);
        for (unsigned i = 0; i < 8; ++i) w[i] = edge[(i + r * 3) % edge.size()] ^ std::uint32_t(r * 0x01000100u);
        inputs.push_back(w);
    }
    std::vector<Case> cases;
    cases.push_back({"r2ur copy guard and zero",
        {mov_reg(38, 8), r2ur(4, 38), mov_ur(0, 4), r2ur(5, 255), mov_ur(1, 5), umov(1, 0xa5a5a5a5u), isetp(255, 255, 0, 0),
         [&] { auto blocked = r2ur(1, 8); field(blocked, 12, 4, 0); return blocked; }(), mov_ur(2, 1),
         isetp(255, 255, 7, 1), [&] { auto taken = r2ur(2, 9); field(taken, 12, 4, 1); return taken; }(), mov_ur(3, 2)},
        inputs, [](const Words& w) { return Pixel{w[0], 0u, 0xa5a5a5a5u, w[1]}; }});
    auto with_predicates = [](std::vector<sn::Instruction> body) {
        auto code = predicate_bits(kPredicates);
        append(code, std::move(body));
        return code;
    };
    cases.push_back({"p2r byte mask and bit 7", with_predicates({
            p2r(0, 8, 0x7f, 0), p2r(1, 8, 0xff, 0), p2r(2, 8, 0x80, 0), p2r(3, 8, 0x1ff, 1)}),
        inputs, [](const Words& w) { return Pixel{gather(w[0], 0x7f, 0, kPredicates), gather(w[0], 0xff, 0, kPredicates),
                                                  gather(w[0], 0x80, 0, kPredicates), gather(w[0], 0x1ff, 1, kPredicates)}; }});
    cases.push_back({"p2r corpus masks", with_predicates({
            p2r(0, 8, 1, 0), p2r(1, 8, 2, 0), p2r(2, 8, 4, 0), p2r(3, 8, 0x20, 0)}),
        inputs, [](const Words& w) { return Pixel{gather(w[0], 1, 0, kPredicates), gather(w[0], 2, 0, kPredicates),
                                                  gather(w[0], 4, 0, kPredicates), gather(w[0], 0x20, 0, kPredicates)}; }});
    cases.push_back({"p2r register form alias and zero source", with_predicates({
            p2r(54, 255, 0x40, 0), mov_reg(0, 54), p2r(1, 8, 0x40, 2), p2r_reg(2, 8, 9, 0),
            p2r(8, 8, 0x0f, 0), mov_reg(3, 8)}),
        inputs, [](const Words& w) { return Pixel{gather(0, 0x40, 0, kPredicates), gather(w[0], 0x40, 2, kPredicates),
                                                  gather(w[0], w[1], 0, kPredicates), gather(w[0], 0x0f, 0, kPredicates)}; }});
    cases.push_back({"p2r execution predicate", with_predicates({
            move(0, 0x11111111u), [&] { auto blocked = p2r(0, 8, 0xff, 0); field(blocked, 12, 4, 8); return blocked; }(),
            [&] { auto taken = p2r(1, 8, 0xff, 0); field(taken, 12, 4, 0); return taken; }()}),
        inputs, [](const Words& w) { return Pixel{0x11111111u, gather(w[0], 0xff, 0, kPredicates), 0u, 0u}; }});

    for (const auto& test : cases) {
        auto compiled = gpu.compile(program(test));
        for (const auto& input : test.inputs) {
            expect(test.name.c_str(), gpu.run(compiled, input), test.expected(input));
            ++checked;
        }
    }

    auto grid = [&](std::vector<sn::Instruction> code) {
        code.push_back(exit_instruction());
        Fixture f(code, sn::Stage::fragment); put(f.code, 0x48, 0xf);
        return gpu.grid(gpu.compile(f), {}, 4, 4);
    };
    {
        const auto pixels = grid({move(0, 0), move(1, 0), move(2, 0), move(3, 0), s2r(0, 0), s2r(1, 56), s2r(2, 57), s2r(3, 59)});
        check_quad_lanes("lane masks", pixels, 0);
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const auto lane = pixels[y * 4 + x][0];
            const auto bit = lane_bit(lane), less = bit - 1u;
            expect("lane eq lt gt", pixels[y * 4 + x], Pixel{lane, bit, less, ~(less | bit)});
            ++checked;
        }
    }
    {
        const auto pixels = grid({move(0, 0), move(1, 0), move(2, 0), move(3, 0), s2r(0, 0), s2r(1, 58), s2r(2, 60), s2r(3, 56)});
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const auto lane = pixels[y * 4 + x][0];
            const auto bit = lane_bit(lane), less = bit - 1u;
            expect("lane le ge eq", pixels[y * 4 + x], Pixel{lane, less | bit, ~less, bit});
            ++checked;
        }
    }
    {
        Fixture y({move(0, 0), move(1, 0), move(2, 0), move(3, 0), s2r(0, 18), exit_instruction()});
        put(y.code, 0x48, 0xf);
        expect("y direction", gpu.run(gpu.compile(y, false), {}), Pixel{0x3f800000u, 0, 0, 0});
        expect("y direction negated", gpu.run(gpu.compile(y, true), {}), Pixel{0xbf800000u, 0, 0, 0});
        checked += 2;
        auto position = raw(0x322); field(position, 24, 8, 255); field(position, 40, 10, 0x70); field(position, 64, 8, 255); field(position, 74, 2, 3);
        auto varying = raw(0x322); field(varying, 24, 8, 255); field(varying, 32, 8, 4); field(varying, 40, 10, 0x80); field(varying, 64, 8, 255);
        Fixture vertex({s2r(4, 18), move(0, 0), move(1, 0), move(2, 0), move(3, 0x3f800000u), position, varying, exit_instruction()}, sn::Stage::vertex);
        vertex.code[0x36] = std::byte{1};
        sn::FrontendOptions options; options.vertex_depth = ir::DepthConvention::negative_one_to_one;
        auto translated = sn::translate_native(vertex.program(), options);
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        const auto& source = std::get<ir::MslArtifact>(translated).source;
        if (source.find("armigo_y_negate") == std::string::npos || source.find("0x3f800000u") == std::string::npos ||
            source.find("0xbf800000u") == std::string::npos)
            throw std::runtime_error("vertex y direction word");
        ++checked;
    }

    auto vote_program = [](bool uniform, unsigned src, bool from_lane) {
        std::vector<sn::Instruction> code{move(0, 0), move(1, 0), move(2, 0), move(3, 0), s2r(1, 0)};
        if (from_lane) { code.push_back(lane_bit(6, 1)); src = 6; }
        if (uniform) {
            code.push_back(vote(0x886, 7, 0, 0, src));
            code.push_back(vote(0x886, 63, 1, 1, src));
            code.push_back(vote(0x886, 63, 2, 2, src));
            code.push_back(umov(1, 1)); code.push_back(umov(2, 2)); code.push_back(umov(4, 4));
            code.push_back(usel(10, 0, 1, 63)); code.push_back(usel(11, 1, 2, 63)); code.push_back(usel(12, 2, 4, 63));
            code.push_back(uadd3(13, 10, 11, 12));
            code.push_back(mov_ur(2, 13)); code.push_back(mov_ur(0, 7));
        } else {
            code.push_back(vote(0x806, 0, 0, 0, src));
            code.push_back(vote(0x806, 255, 1, 1, src));
            code.push_back(vote(0x806, 255, 2, 2, src));
            code.push_back(p2r(2, 255, 7, 0));
        }
        return code;
    };
    for (const bool uniform : {false, true}) for (unsigned kind = 0; kind < 3; ++kind) {
        const char* name = uniform ? "uniform vote" : "vote";
        const auto pixels = grid(vote_program(uniform, kind == 0 ? PT : NOT_PT, kind == 2));
        check_quad_lanes(name, pixels, 1);
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const auto& got = pixels[y * 4 + x];
            const auto lane = got[1];
            const unsigned packed = kind == 0 ? 7u : kind == 1 ? 4u : 2u;
            const unsigned bit = kind == 0 ? 1u : kind == 1 ? 0u : (lane & 1u);
            const auto& origin = pixels[(y & ~1u) * 4 + (x & ~1u)];
            if (got[2] != packed || ((got[0] >> lane) & 1u) != bit || got[0] != origin[0] || got[2] != origin[2] || got[3] != 0) {
                std::fprintf(stderr, "%s kind %u at (%u, %u): ballot %08x lane %u packed %08x\n", name, kind, x, y, got[0], lane, got[2]);
                throw std::runtime_error(name);
            }
            ++checked;
        }
    }

    // Depth and sample-mask exports (header word 0x4c: bit 0 the sample mask,
    // bit 1 depth) follow the color registers: the sample mask in the next
    // register, depth in the one after it.
    {
        gpu.depth_target = true;
        Fixture depth_only({move(1, 0x3e800000), exit_instruction()}, sn::Stage::fragment);
        put(depth_only.code, 0x48, 0); put(depth_only.code, 0x4c, 2);
        gpu.run(gpu.compile(depth_only), {});
        if (gpu.depths.size() != 1 || gpu.depths[0] != 0.25f) throw std::runtime_error("depth-only export reads R1");
        Fixture color_depth({move(0, 1), move(1, 2), move(2, 3), move(3, 4), move(4, 0x3f800000), move(5, 0x3f400000),
                             exit_instruction()}, sn::Stage::fragment);
        put(color_depth.code, 0x48, 0xf); put(color_depth.code, 0x4c, 2);
        expect("color with depth", gpu.run(gpu.compile(color_depth), {}), Pixel{1, 2, 3, 4});
        if (gpu.depths.size() != 1 || gpu.depths[0] != 0.75f) throw std::runtime_error("depth after a color target reads R5");
        // A target exporting only x and y still takes four registers: depth
        // is R5.
        Fixture sparse({move(0, 1), move(1, 2), move(3, 0x3f400000), move(5, 0x3f000000), exit_instruction()}, sn::Stage::fragment);
        put(sparse.code, 0x48, 0x3); put(sparse.code, 0x4c, 2);
        const auto sparse_pixel = gpu.run(gpu.compile(sparse), {});
        if (sparse_pixel[0] != 1 || sparse_pixel[1] != 2 || gpu.depths.size() != 1 || gpu.depths[0] != 0.5f)
            throw std::runtime_error("sparse color with depth reads depth from R5");
        gpu.depth_target = false;
        for (const std::uint32_t mask : {0u, 1u}) {
            Fixture masked({move(0, 5), move(1, 6), move(2, 7), move(3, 8), move(4, mask), exit_instruction()}, sn::Stage::fragment);
            put(masked.code, 0x48, 0xf); put(masked.code, 0x4c, 1);
            const auto c = MetalHarness::cleared;
            expect(mask ? "sample mask 1" : "sample mask 0", gpu.run(gpu.compile(masked), {}), mask ? Pixel{5, 6, 7, 8} : Pixel{c, c, c, c});
        }
        Fixture unknown({move(0, 1), exit_instruction()}, sn::Stage::fragment);
        put(unknown.code, 0x4c, 4);
        refused("unknown export bits", unknown, "other than the sample mask and depth");
        checked += 6;
    }

    std::printf("Native system Metal execution passed (%u vectors)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
