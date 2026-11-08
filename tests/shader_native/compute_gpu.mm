// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed native compute stages: the control's thread group size
// launches the kernel, and every invocation stores its thread id, group id,
// lane id, packed thread id and lane mask to guest memory at its own linear
// index. The expected words follow from the dispatch alone.

#include "metal_harness.h"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <tuple>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
using Words = std::vector<std::uint32_t>;
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
constexpr unsigned PT = 7, NOT_PT = 15;
constexpr std::uint64_t base = 0x7dfff00000ull;
sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
sn::Instruction load(unsigned d, unsigned index) { return op(0xa02, {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}}); }
sn::Instruction s2r(unsigned d, unsigned selector) { return op(0x919, {{16, 8, d}, {72, 8, selector}}); }
// IMAD Rd = Ra * Rb + Rc, and the unsigned IMAD.WIDE Rd:Rd+1 = Ra * Rb + Rc:Rc+1.
sn::Instruction imad(unsigned opcode, unsigned d, unsigned a, unsigned b, unsigned c) {
    return op(opcode, {{16, 8, d}, {24, 8, a}, {32, 8, b}, {64, 8, c}, {81, 3, PT}, {87, 4, NOT_PT}});
}
// STG.E.128 [Ra.64 + displacement], Rb..Rb+3
sn::Instruction stg128(unsigned a, unsigned data, unsigned displacement) {
    auto i = op(0x386, {{24, 8, a}, {32, 8, data}, {40, 24, displacement}, {72, 1, 1}});
    field(i, 73, 3, 6); field(i, 77, 4, 5); field(i, 84, 3, 1);
    return i;
}
// Input words (bank 3): guest base low and high, threads per group, group
// width, record bytes, group width times height. Each invocation stores
// eight words at its linear index in the dispatch.
Fixture program() {
    return Fixture({load(8, 0), load(9, 1), load(13, 2), load(15, 3), load(16, 4), load(17, 5),
                    s2r(0, 33), s2r(1, 34), s2r(2, 37), s2r(3, 0), s2r(4, 32), s2r(5, 35), s2r(6, 57), s2r(7, 39),
                    imad(0x224, 14, 1, 15, 0), imad(0x224, 14, 5, 17, 14), imad(0x224, 12, 2, 13, 14), imad(0x225, 10, 12, 16, 8),
                    stg128(10, 0, 0), stg128(10, 4, 16), exit_instruction()},
                   sn::Stage::compute);
}
} // namespace
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    unsigned checked = 0;
    for (const auto& [size, groups] : {std::pair{std::array<std::uint32_t, 3>{32, 1, 1}, 2u}, {{8, 4, 1}, 3u}, {{16, 2, 2}, 1u}, {{64, 1, 1}, 2u}}) {
        auto fixture = program();
        fixture.launch(size, 0);
        const auto threads = size[0] * size[1] * size[2];
        auto compiled = gpu.compile_compute(fixture);
        check(compiled.artifact.reflection.compute && compiled.artifact.reflection.compute->local_size.dimensions == size,
              "the kernel takes the control's thread group size");
        MetalHarness::GuestMemory guest{base, Words(threads * groups * 8, 0xdeadbeefu), {}};
        gpu.dispatch(compiled, fixture, {std::uint32_t(base), std::uint32_t(base >> 32), threads, size[0], 32, size[0] * size[1]}, guest, groups);
        for (unsigned group = 0; group < groups; ++group)
            for (unsigned z = 0; z < size[2]; ++z) for (unsigned y = 0; y < size[1]; ++y) for (unsigned x = 0; x < size[0]; ++x) {
                // SIMD-groups take consecutive linear thread indices, x fastest.
                const unsigned linear = (z * size[1] + y) * size[0] + x;
                const unsigned lane = linear % 32;
                const unsigned record = group * threads + linear;
                const Words want{x, y, group, lane, x | y << 16 | z << 26, z, (1u << lane) - 1, 0};
                const Words got(guest.words.begin() + record * 8, guest.words.begin() + record * 8 + 8);
                if (got != want) {
                    std::fprintf(stderr, "group size %ux%ux%u, group %u, thread %u,%u,%u: got %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        size[0], size[1], size[2], group, x, y, z, got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7]);
                    throw std::runtime_error("native compute GPU result");
                }
                ++checked;
            }
        check(guest.table[1] == 0, "compute stores stay inside guest memory");
    }
    // SUST.P.2D: each thread of a 4x4 group stores its texel through the
    // bound handle; unpicked components store zero.
    for (const unsigned mask : {0xfu, 0x5u, 0x8u}) {
        const auto sust = op(0xb9b, {{24, 8, 0}, {32, 8, 4}, {40, 14, 0x1e}, {54, 5, 0}, {61, 3, 3}, {72, 4, mask}, {84, 3, 1}});
        Fixture image({s2r(0, 33), s2r(1, 34), load(4, 0), load(5, 1), load(6, 2), load(7, 3), sust, exit_instruction()}, sn::Stage::compute);
        image.launch({4, 4, 1}, 0);
        auto compiled = gpu.compile_compute(image);
        MetalHarness::GuestMemory unused{base, Words(4, 0), {}};
        const Words data{0x3f800000u, 0x40000000u, 0xbf000000u, 0x42c80000u};  // 1, 2, -0.5, 100
        gpu.dispatch(compiled, image, data, unused);
        const auto images = gpu.images();
        check(images.size() == 1, "a surface store binds one written image");
        for (unsigned texel = 0; texel < 16; ++texel) {
            unsigned picked = 0;
            for (unsigned component = 0; component < 4; ++component) {
                const float want = (mask & (1u << component)) ? std::bit_cast<float>(data[picked++]) : 0.0f;
                if (images[0][texel * 4 + component] != want) {
                    std::fprintf(stderr, "surface store mask %x texel %u component %u: got %g, expected %g\n", mask, texel, component,
                                 images[0][texel * 4 + component], want);
                    throw std::runtime_error("native surface store GPU result");
                }
            }
            ++checked;
        }
    }
    // SURED.2D: each thread ORs, adds and XORs its own R32Uint texel. The
    // seed is the harness pattern; the expected word is that sequence.
    {
        const auto reduction = [](unsigned data, unsigned kind) {
            return op(0x59f, {{24, 8, 0}, {32, 8, data}, {40, 14, 0x1e}, {54, 5, 0}, {61, 3, 3}, {77, 3, 7}, {84, 1, 1}, {87, 3, kind}});
        };
        Fixture image({s2r(0, 33), s2r(1, 34), load(4, 0), load(5, 1), load(6, 2),
                       reduction(4, 6), reduction(5, 0), reduction(6, 7), exit_instruction()},
                      sn::Stage::compute);
        image.launch({4, 4, 1}, 0);
        auto compiled = gpu.compile_compute(image);
        MetalHarness::GuestMemory unused{base, Words(4, 0), {}};
        constexpr std::uint32_t or_operand = 0x00f0000fu, add_operand = 3u, xor_operand = 0x0000ff00u;
        gpu.dispatch(compiled, image, {or_operand, add_operand, xor_operand}, unused);
        const auto images = gpu.atomic_images();
        check(images.size() == 1 && gpu.images().empty(), "a surface reduction binds one atomic image");
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const auto seed = MetalHarness::atomic_texel(x, y);
            const auto want = ((seed | or_operand) + add_operand) ^ xor_operand;
            if (images[0][y * 4 + x] != want) {
                std::fprintf(stderr, "surface reduction at %u,%u: got %08x, expected %08x\n", x, y, images[0][y * 4 + x], want);
                throw std::runtime_error("native surface reduction GPU result");
            }
            ++checked;
        }
    }
    // SULD.P.2D: each thread of a 4x4 group loads its texel of the bound
    // pattern image, all four components into R4-R7 and green alone into R12,
    // and stores them at its record (eight words).
    {
        const auto suld = [](unsigned d, unsigned mask) {
            return op(0xb97, {{16, 8, d}, {24, 8, 0}, {40, 14, 0x1e}, {54, 5, 0}, {61, 3, 3}, {72, 4, mask}, {81, 4, 15}});
        };
        Fixture image({load(8, 0), load(9, 1), load(15, 2), load(16, 3), s2r(0, 33), s2r(1, 34), suld(4, 0xf), suld(12, 0x2),
                       imad(0x224, 14, 1, 15, 0), imad(0x225, 10, 14, 16, 8), stg128(10, 4, 0), stg128(10, 12, 16), exit_instruction()},
                      sn::Stage::compute);
        image.launch({4, 4, 1}, 0);
        auto compiled = gpu.compile_compute(image);
        MetalHarness::GuestMemory guest{base, Words(16 * 8, 0), {}};
        gpu.dispatch(compiled, image, {std::uint32_t(base), std::uint32_t(base >> 32), 4, 32}, guest);
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const auto want = MetalHarness::texel(x, y, 0);
            const auto* got = &guest.words[(y * 4 + x) * 8];
            for (unsigned c = 0; c < 4; ++c)
                if (got[c] != std::bit_cast<std::uint32_t>(want[c]))
                    throw std::runtime_error("SULD.P component " + std::to_string(c) + " at " + std::to_string(x) + "," + std::to_string(y));
            if (got[4] != std::bit_cast<std::uint32_t>(want[1])) throw std::runtime_error("SULD.P green alone");
            ++checked;
        }
    }
    // SULD.P outside the image reads zero in every component: R and A of an
    // R32Float image (alpha 1 inside it) at x = 2 * tid.x - 2, so x -2 and 4
    // lie outside the 4-texel row and 0 and 2 inside it. A load to RZ reads
    // no image at all.
    {
        const auto suld_at = [](unsigned d, unsigned a, unsigned mask) {
            return op(0xb97, {{16, 8, d}, {24, 8, a}, {40, 14, 0x1e}, {54, 5, 0}, {61, 3, 3}, {72, 4, mask}, {81, 4, 15}});
        };
        gpu.red_pattern = true;
        Fixture outside({load(8, 0), load(9, 1), load(15, 2), load(16, 3), load(17, 4), load(18, 5), s2r(0, 33), s2r(1, 34), s2r(3, 34),
                         imad(0x224, 2, 0, 17, 18), suld_at(4, 2, 0x9), imad(0x224, 14, 1, 15, 0), imad(0x225, 10, 14, 16, 8),
                         stg128(10, 4, 0), exit_instruction()},
                        sn::Stage::compute);
        outside.launch({4, 4, 1}, 0);
        auto compiled = gpu.compile_compute(outside);
        MetalHarness::GuestMemory guest{base, Words(16 * 8, 0xdeadbeefu), {}};
        gpu.dispatch(compiled, outside, {std::uint32_t(base), std::uint32_t(base >> 32), 4, 32, 2, 0xfffffffeu}, guest);
        gpu.red_pattern = false;
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            const int column = 2 * int(x) - 2;
            const bool inside = column >= 0 && column < 4;
            const auto red = inside ? std::bit_cast<std::uint32_t>(MetalHarness::texel(unsigned(column), y, 0)[3]) : 0u;
            const auto alpha = inside ? 0x3f800000u : 0u;
            const auto* got = &guest.words[(y * 4 + x) * 8];
            if (got[0] != red || got[1] != alpha) {
                std::fprintf(stderr, "SULD.P x %d y %u: red %08x alpha %08x, expected %08x %08x\n", column, y, got[0], got[1], red, alpha);
                throw std::runtime_error("native surface load outside the image");
            }
            ++checked;
        }
        Fixture discarded({s2r(0, 33), s2r(1, 34), suld_at(255, 0, 0xf), exit_instruction()}, sn::Stage::compute);
        discarded.launch({4, 4, 1}, 0);
        auto translated = sn::translate_native(discarded.program());
        check(std::holds_alternative<ir::MslArtifact>(translated), "SULD.P to RZ translates");
        const auto& artifact = std::get<ir::MslArtifact>(translated);
        check(std::none_of(artifact.reflection.resources.begin(), artifact.reflection.resources.end(),
                           [](const auto& resource) { return std::holds_alternative<ir::TextureUse>(resource.use); }) &&
              artifact.source.find(".read(") == std::string::npos,
              "SULD.P to RZ reads no image");
    }
    // SUST.P.3D: one thread stores the input RGBA texel at the input
    // coordinate (x, y, z). Every other texel of the 4x4x4 volume stays zero.
    {
        const auto sust = op(0xb9b, {{24, 8, 0}, {32, 8, 4}, {40, 14, 0x1e}, {54, 5, 0}, {61, 3, 5}, {72, 4, 0xf}, {84, 3, 1}});
        Fixture image({load(0, 0), load(1, 1), load(2, 2), load(4, 3), load(5, 4), load(6, 5), load(7, 6), sust, exit_instruction()},
                      sn::Stage::compute);
        image.launch({1, 1, 1}, 0);
        auto compiled = gpu.compile_compute(image);
        MetalHarness::GuestMemory unused{base, Words(4, 0), {}};
        const unsigned x = 1, y = 2, z = 3;
        const Words data{0x3f800000u, 0x40000000u, 0xbf000000u, 0x42c80000u};  // 1, 2, -0.5, 100
        gpu.dispatch(compiled, image, {x, y, z, data[0], data[1], data[2], data[3]}, unused);
        const auto images = gpu.images3d();
        check(images.size() == 1 && gpu.images().empty(), "a 3D surface store binds one written volume");
        for (unsigned tz = 0; tz < 4; ++tz) for (unsigned ty = 0; ty < 4; ++ty) for (unsigned tx = 0; tx < 4; ++tx) {
            const unsigned texel = ((tz * 4 + ty) * 4 + tx) * 4;
            for (unsigned component = 0; component < 4; ++component) {
                const float want = (tx == x && ty == y && tz == z) ? std::bit_cast<float>(data[component]) : 0.0f;
                if (images[0][texel + component] != want) {
                    std::fprintf(stderr, "3D surface store at %u,%u,%u component %u: got %g, expected %g\n",
                                 tx, ty, tz, component, images[0][texel + component], want);
                    throw std::runtime_error("native 3D surface store GPU result");
                }
            }
            ++checked;
        }
    }
    // SULD.P.3D: one thread loads the pattern texel at the input (x, y, z)
    // and the same x, y with a depth outside the volume, which reads zero.
    {
        const auto suld = [](unsigned d, unsigned a) {
            return op(0xb97, {{16, 8, d}, {24, 8, a}, {40, 14, 0x1e}, {54, 5, 0}, {61, 3, 5}, {72, 4, 0xf}, {81, 4, 15}});
        };
        Fixture image({load(8, 0), load(9, 1), load(0, 2), load(1, 3), load(2, 4), load(4, 5), load(5, 6), load(6, 7),
                       suld(12, 0), suld(16, 4), stg128(8, 12, 0), stg128(8, 16, 16), exit_instruction()},
                      sn::Stage::compute);
        image.launch({1, 1, 1}, 0);
        auto compiled = gpu.compile_compute(image);
        MetalHarness::GuestMemory guest{base, Words(8, 0xdeadbeefu), {}};
        const unsigned x = 1, y = 2, z = 3;
        gpu.dispatch(compiled, image, {std::uint32_t(base), std::uint32_t(base >> 32), x, y, z, x, y, 4u}, guest);
        const auto want = MetalHarness::texel(x, y, z);
        for (unsigned c = 0; c < 4; ++c)
            if (guest.words[c] != std::bit_cast<std::uint32_t>(want[c])) {
                std::fprintf(stderr, "SULD.P.3D component %u: got %08x, expected %08x\n", c, guest.words[c],
                             std::bit_cast<std::uint32_t>(want[c]));
                throw std::runtime_error("native 3D surface load GPU result");
            }
        for (unsigned c = 0; c < 4; ++c)
            if (guest.words[4 + c] != 0) {
                std::fprintf(stderr, "SULD.P.3D out-of-range depth component %u: got %08x\n", c, guest.words[4 + c]);
                throw std::runtime_error("native 3D surface load outside the volume");
            }
        checked += 2;
    }
    // REDUX over the 32 lanes of a group: R1 = lane * 0x10000001 (its sum
    // wraps), R2 = lane - 16 (signed and unsigned extrema differ). Each lane
    // stores the seven uniform results it reads back.
    {
        const auto redux = [](unsigned uniform, unsigned source, unsigned operation, bool signed_order) {
            return op(0x3c4, {{16, 6, uniform}, {24, 8, source}, {73, 1, signed_order}, {78, 3, operation}});
        };
        const auto mov_ur = [](unsigned dest, unsigned uniform) {
            auto i = raw(0xc02); field(i, 16, 8, dest); i.words[1] = uniform; i.words[2] = 0x08000f00; return i;
        };
        Fixture reduce({load(8, 0), load(9, 1), load(16, 2), load(17, 3), load(18, 4), load(19, 5), s2r(0, 0),
                        imad(0x224, 1, 0, 17, 255), imad(0x224, 2, 0, 18, 19),
                        redux(1, 1, 3, false), redux(2, 2, 4, false), redux(3, 2, 5, false), redux(4, 2, 4, true),
                        redux(5, 2, 5, true), redux(6, 1, 1, false), redux(7, 1, 2, false),
                        mov_ur(4, 1), mov_ur(5, 2), mov_ur(6, 3), mov_ur(7, 4), mov_ur(12, 5), mov_ur(13, 6), mov_ur(14, 7),
                        imad(0x225, 10, 0, 16, 8), stg128(10, 4, 0), stg128(10, 12, 16), exit_instruction()},
                       sn::Stage::compute);
        reduce.launch({32, 1, 1}, 0);
        auto compiled = gpu.compile_compute(reduce);
        MetalHarness::GuestMemory guest{base, Words(32 * 8, 0), {}};
        gpu.dispatch(compiled, reduce, {std::uint32_t(base), std::uint32_t(base >> 32), 32, 0x10000001u, 1, 0xfffffff0u}, guest);
        std::uint32_t sum = 0, any = 0, parity = 0;
        for (std::uint32_t lane = 0; lane < 32; ++lane) {
            const std::uint32_t v = lane * 0x10000001u;
            sum += v; any |= v; parity ^= v;
        }
        const std::array<std::uint32_t, 7> want{sum, 0u, 0xffffffffu, 0xfffffff0u, 15u, any, parity};
        for (unsigned lane = 0; lane < 32; ++lane) {
            const auto* got = &guest.words[lane * 8];
            const std::array<std::uint32_t, 7> results{got[0], got[1], got[2], got[3], got[4], got[5], got[6]};
            if (results != want) {
                std::fprintf(stderr, "REDUX lane %u: %08x %08x %08x %08x %08x %08x %08x\n", lane,
                             got[0], got[1], got[2], got[3], got[4], got[5], got[6]);
                throw std::runtime_error("native warp reduction GPU result");
            }
            ++checked;
        }
        const auto refused = [](sn::Instruction instruction, const char* name) {
            Fixture f({instruction, exit_instruction()}, sn::Stage::compute);
            check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(f.program())), name);
        };
        refused(redux(1, 1, 0, false), "REDUX.AND is not admitted");
        refused(redux(1, 1, 3, true), "a signed REDUX.SUM is not admitted");
        auto guarded = redux(1, 1, 3, false); field(guarded, 12, 4, 0);
        refused(guarded, "a guarded REDUX is not admitted");
    }
    // RED.F32 and ATOMG.F32 (type 3, flushed, nearest): 64 threads add 1.0
    // and 0.5 to words 0 and 1; each ATOMG adds 1.0 to word 2 and RED adds
    // the old value it returned to word 3, 0 + 1 + ... + 63 when every
    // thread saw its own.
    {
        auto red_f32 = [](unsigned data, unsigned displacement) {
            return op(0x98e, {{24, 8, 8}, {32, 8, data}, {40, 24, displacement}, {72, 1, 1}, {73, 3, 3}, {77, 4, 7}});
        };
        auto atom_f32 = [](unsigned d, unsigned data, unsigned displacement) {
            return op(0x3a8, {{16, 8, d}, {24, 8, 8}, {32, 8, data}, {40, 24, displacement}, {72, 1, 1}, {73, 3, 3}, {77, 4, 7}, {81, 3, 7}});
        };
        Fixture adds({load(8, 0), load(9, 1), move(4, 0x3f800000u), move(5, 0x3f000000u), red_f32(4, 0), red_f32(5, 4),
                      atom_f32(6, 4, 8), red_f32(6, 12), exit_instruction()}, sn::Stage::compute);
        adds.launch({64, 1, 1}, 0);
        auto compiled = gpu.compile_compute(adds);
        MetalHarness::GuestMemory guest{base, Words(4, 0), {}};
        gpu.dispatch(compiled, adds, {std::uint32_t(base), std::uint32_t(base >> 32)}, guest);
        const Words want{std::bit_cast<std::uint32_t>(64.0f), std::bit_cast<std::uint32_t>(32.0f),
                         std::bit_cast<std::uint32_t>(64.0f), std::bit_cast<std::uint32_t>(2016.0f)};
        if (guest.words != want) {
            std::fprintf(stderr, "float atomics: got %08x %08x %08x %08x\n", guest.words[0], guest.words[1], guest.words[2], guest.words[3]);
            throw std::runtime_error("native float atomic GPU result");
        }
        check(guest.table[1] == 0, "float atomics stay inside guest memory");
        ++checked;
    }
    // 64 threads, one word, bound 63. ATOMG.INC from 0 finishes at 0 and the
    // returned old values are {0..63} once each. ATOMG.DEC from 63 returns
    // that set and finishes at 63. RED.INC from 0 finishes at 0. Each batch
    // is dispatched four times so a dropped retry is likely to show.
    {
        auto atomic = [](unsigned opcode, unsigned operation, bool reduce) {
            if (reduce)
                return op(opcode, {{24, 8, 8}, {32, 8, 4}, {72, 1, 1}, {73, 3, 0}, {77, 4, 7}, {84, 3, 1}, {87, 3, operation}});
            return op(opcode, {{16, 8, 6}, {24, 8, 8}, {32, 8, 4}, {72, 1, 1}, {73, 3, 0}, {77, 4, 7}, {81, 3, 7}, {84, 3, 1}, {87, 4, operation}});
        };
        const auto store_old = op(0x386, {{24, 8, 10}, {32, 8, 6}, {40, 24, 4}, {72, 1, 1}, {73, 3, 4}, {77, 4, 5}, {84, 3, 1}});
        auto body = [&](sn::Instruction update, bool keep) {
            std::vector<sn::Instruction> code{load(8, 0), load(9, 1), s2r(0, 33), move(4, 63), move(12, 4), update};
            if (keep) { code.push_back(imad(0x225, 10, 0, 12, 8)); code.push_back(store_old); }
            code.push_back(exit_instruction());
            Fixture fixture(std::move(code), sn::Stage::compute);
            fixture.launch({64, 1, 1}, 0);
            return fixture;
        };
        const auto inc = body(atomic(0x3a8, 3, false), true);
        const auto dec = body(atomic(0x3a8, 4, false), true);
        const auto red = body(atomic(0x98e, 3, true), false);
        const auto inc_kernel = gpu.compile_compute(inc);
        const auto dec_kernel = gpu.compile_compute(dec);
        const auto red_kernel = gpu.compile_compute(red);
        const Words input{std::uint32_t(base), std::uint32_t(base >> 32)};
        auto olds = [](const Words& words, std::uint32_t final_word, const char* name) {
            if (words[0] != final_word) {
                std::fprintf(stderr, "%s: final word %08x, expected %08x\n", name, words[0], final_word);
                throw std::runtime_error("native contended INC/DEC final word");
            }
            unsigned seen[64] = {};
            for (unsigned thread = 0; thread < 64; ++thread) {
                const auto old = words[thread + 1];
                if (old >= 64 || seen[old]++) {
                    std::fprintf(stderr, "%s: thread %u returned %08x\n", name, thread, old);
                    throw std::runtime_error("native contended INC/DEC old values");
                }
            }
        };
        for (unsigned pass = 0; pass < 4; ++pass) {
            MetalHarness::GuestMemory guest{base, Words(65, 0xffffffffu), {}};
            guest.words[0] = 0;
            gpu.dispatch(inc_kernel, inc, input, guest);
            olds(guest.words, 0, "ATOMG.INC");
            check(guest.table[1] == 0, "contended INC stays inside guest memory");
            guest.words.assign(65, 0xffffffffu);
            guest.words[0] = 63;
            gpu.dispatch(dec_kernel, dec, input, guest);
            olds(guest.words, 63, "ATOMG.DEC");
            check(guest.table[1] == 0, "contended DEC stays inside guest memory");
            MetalHarness::GuestMemory reduced{base, Words{0}, {}};
            gpu.dispatch(red_kernel, red, input, reduced);
            if (reduced.words[0] != 0 || reduced.table[1] != 0) {
                std::fprintf(stderr, "RED.INC: word %08x misses %u\n", reduced.words[0], reduced.table[1]);
                throw std::runtime_error("native contended RED.INC");
            }
            checked += 3;
        }
    }
    // Local memory sized by the control: +0x68 per thread, +0x50 the warp's,
    // 32 times it. Each thread stores its x to local +12, loads it back and
    // writes it to guest memory at its index. Sizes that disagree, or one too
    // small for +12, are refused.
    {
        auto stl = op(0x387, {{24, 8, 255}, {32, 8, 0}, {40, 24, 12}, {73, 3, 4}, {84, 3, 1}});
        auto ldl = op(0x983, {{16, 8, 4}, {24, 8, 255}, {40, 24, 12}, {73, 3, 4}, {84, 3, 1}});
        auto stg = op(0x386, {{24, 8, 10}, {32, 8, 4}, {72, 1, 1}, {73, 3, 4}, {77, 4, 5}, {84, 3, 1}});
        Fixture local({load(8, 0), load(9, 1), s2r(0, 33), move(12, 4), stl, ldl, imad(0x225, 10, 0, 12, 8), stg, exit_instruction()},
                      sn::Stage::compute);
        local.launch({32, 1, 1}, 0);
        put(local.control, 0x68, 16); put(local.control, 0x50, 512);
        auto compiled = gpu.compile_compute(local);
        MetalHarness::GuestMemory guest{base, Words(32, 0xdeadbeefu), {}};
        gpu.dispatch(compiled, local, {std::uint32_t(base), std::uint32_t(base >> 32)}, guest);
        for (std::uint32_t x = 0; x < 32; ++x)
            if (guest.words[x] != x) {
                std::fprintf(stderr, "compute local memory: thread %u got %08x\n", x, guest.words[x]);
                throw std::runtime_error("native compute local memory GPU result");
            }
        checked += 32;
        auto disagreeing = local; put(disagreeing.control, 0x50, 256);
        check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(disagreeing.program())), "disagreeing local memory sizes are refused");
        auto small = local; put(small.control, 0x68, 8); put(small.control, 0x50, 256);
        check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(small.program())), "a local access past the size is refused");
    }
    std::printf("Native compute Metal execution passed (%u invocations)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
