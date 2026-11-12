// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed warp matrix instructions in one 32-lane compute group. Each
// lane loads its A, B and C fragments from guest memory, runs HMMA.16816 or
// MOVM, and stores its result. The CPU model builds whole matrices, hands
// each lane its fragment in the documented layout (lane L: g = L / 4,
// t = L % 4), and computes D as an in-order binary32 fused multiply-add
// chain over each eight-product block, rounded to binary16 at both block
// boundaries. Random products and binary32 partial sums remain exact; the
// binary16 boundary conversions are modeled explicitly.

#include "metal_harness.h"
#include <bit>
#include <cmath>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <random>
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
constexpr unsigned record_words = 16;  // A 0-3, B 4-5, C (binary16) 6-7, C (binary32) 8-11, D 12-15
sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
sn::Instruction load(unsigned d, unsigned index) { return op(0xa02, {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}}); }
sn::Instruction memory_op(unsigned opcode, unsigned type, std::initializer_list<Field> fields) {
    auto i = op(opcode, fields);
    field(i, 73, 3, type); field(i, 77, 4, 5); field(i, 84, 3, 1);
    return i;
}
// LDG.E / STG.E [Ra.64 + displacement]; type 5 is 64 bits, 6 is 128 bits.
sn::Instruction ldg(unsigned d, unsigned a, unsigned displacement, unsigned type) {
    return memory_op(0x381, type, {{16, 8, d}, {24, 8, a}, {40, 24, displacement}, {72, 1, 1}, {81, 3, 7}});
}
sn::Instruction stg(unsigned a, unsigned data, unsigned displacement, unsigned type) {
    return memory_op(0x386, type, {{24, 8, a}, {32, 8, data}, {40, 24, displacement}, {72, 1, 1}});
}
// The lane's record address: R10:R11 = lane * 64 + base.
std::vector<sn::Instruction> prologue() {
    return {load(8, 0), load(9, 1), load(21, 2), op(0x919, {{16, 8, 20}, {72, 8, 0}}),
            op(0x225, {{16, 8, 10}, {24, 8, 20}, {32, 8, 21}, {64, 8, 8}, {81, 3, PT}, {87, 4, NOT_PT}}),
            ldg(0, 10, 0, 6), ldg(4, 10, 16, 5), ldg(6, 10, 24, 5), ldg(12, 10, 32, 6)};
}
Fixture program(std::vector<sn::Instruction> body) {
    auto code = prologue();
    code.insert(code.end(), body.begin(), body.end());
    code.push_back(exit_instruction());
    return Fixture(code, sn::Stage::compute);
}

// binary16 <-> binary32, rounding to nearest even.
float from_half(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h >> 15) << 31, exponent = (h >> 10) & 31, fraction = h & 0x3ff;
    if (exponent == 31) return std::bit_cast<float>(sign | 0x7f800000u | fraction << 13);
    if (exponent == 0) return std::bit_cast<float>(sign) + (sign ? -1.0f : 1.0f) * std::ldexp(float(fraction), -24);
    return std::bit_cast<float>(sign | (exponent + 112) << 23 | fraction << 13);
}
std::uint16_t to_half(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value), sign = (bits >> 16) & 0x8000;
    const float magnitude = std::fabs(value);
    if (std::isnan(value)) return std::uint16_t(sign | 0x7e00);
    if (magnitude >= 65520.0f) return std::uint16_t(sign | 0x7c00);
    if (magnitude < std::ldexp(1.0f, -14)) {  // subnormal or zero: units of 2^-24
        const float units = std::nearbyint(magnitude * std::ldexp(1.0f, 24));
        return std::uint16_t(sign | std::uint32_t(units));
    }
    int exponent; std::frexp(magnitude, &exponent);  // magnitude in [2^(e-1), 2^e)
    const float scaled = std::nearbyint(std::ldexp(magnitude, 11 - exponent));  // 11 significant bits
    std::uint32_t mantissa = std::uint32_t(scaled), biased = std::uint32_t(exponent - 1 + 15);
    if (mantissa == 2048) { mantissa = 1024; ++biased; }
    if (biased >= 31) return std::uint16_t(sign | 0x7c00);
    return std::uint16_t(sign | biased << 10 | (mantissa & 0x3ff));
}
std::uint32_t pack(std::uint16_t low, std::uint16_t high) { return low | std::uint32_t(high) << 16; }

struct Matrices { std::uint16_t a[16][16], b[16][8], c[16][8]; float c32[16][8]; };
Matrices random_matrices(std::mt19937& rng) {
    // Halves with few significant bits and small exponents keep every product
    // and partial sum exact or well inside binary32, as a network's weights do.
    std::uniform_int_distribution<int> magnitude(-24, 24), scale(-3, 1);
    auto value = [&] { return to_half(std::ldexp(float(magnitude(rng)), scale(rng))); };
    Matrices m;
    for (auto& row : m.a) for (auto& x : row) x = value();
    for (auto& row : m.b) for (auto& x : row) x = value();
    for (unsigned i = 0; i < 16; ++i) for (unsigned j = 0; j < 8; ++j) { m.c[i][j] = value(); m.c32[i][j] = from_half(value()) * 3.0f; }
    return m;
}
// Lane L's fragments of the matrices: A (row g / g + 8, k 2t, 2t + 1, +8),
// B (k 2t, 2t + 1, +8; column g), C (row g / g + 8, columns 2t, 2t + 1).
Words fragments(const Matrices& m) {
    Words w(32 * record_words, 0);
    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned g = lane / 4, t = lane % 4;
        auto* r = &w[lane * record_words];
        for (unsigned reg = 0; reg < 4; ++reg) {
            const unsigned row = g + 8 * (reg & 1), k = 2 * t + 8 * (reg >> 1);
            r[reg] = pack(m.a[row][k], m.a[row][k + 1]);
        }
        for (unsigned reg = 0; reg < 2; ++reg) r[4 + reg] = pack(m.b[2 * t + 8 * reg][g], m.b[2 * t + 8 * reg + 1][g]);
        for (unsigned reg = 0; reg < 2; ++reg) r[6 + reg] = pack(m.c[g + 8 * reg][2 * t], m.c[g + 8 * reg][2 * t + 1]);
        for (unsigned i = 0; i < 4; ++i) r[8 + i] = std::bit_cast<std::uint32_t>(m.c32[g + 8 * (i >> 1)][2 * t + (i & 1)]);
    }
    return w;
}
float product(const Matrices& m, unsigned row, unsigned column, float accumulator) {
    for (unsigned first = 0; first < 16; first += 8) {
        for (unsigned k = first; k < first + 8; ++k)
            accumulator = std::fma(from_half(m.a[row][k]), from_half(m.b[k][column]), accumulator);
        accumulator = from_half(to_half(accumulator));
    }
    return accumulator;
}

unsigned run(MetalHarness& gpu, const Fixture& fixture, Words memory, const std::function<Words(unsigned, const Words&)>& expected,
             const std::string& name) {
    auto compiled = gpu.compile_compute(fixture);
    MetalHarness::GuestMemory guest{base, memory, {}};
    gpu.dispatch(compiled, fixture, {std::uint32_t(base), std::uint32_t(base >> 32), record_words * 4}, guest);
    check(guest.table[1] == 0, "matrix test memory stays inside guest memory");
    for (unsigned lane = 0; lane < 32; ++lane) {
        const auto want = expected(lane, memory);
        const Words got(guest.words.begin() + lane * record_words + 12, guest.words.begin() + lane * record_words + 12 + want.size());
        if (got != want) {
            std::fprintf(stderr, "%s lane %u:", name.c_str(), lane);
            for (unsigned i = 0; i < want.size(); ++i) std::fprintf(stderr, " got %08x want %08x;", got[i], want[i]);
            std::fprintf(stderr, "\n");
            throw std::runtime_error("native warp matrix GPU result");
        }
    }
    return 32;
}
} // namespace
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    unsigned checked = 0;
    std::mt19937 rng(20261108);
    auto hmma = [](unsigned d, unsigned a, unsigned b, unsigned c, bool f32) {
        return op(0x23c, {{16, 8, d}, {24, 8, a}, {32, 8, b}, {64, 8, c}, {75, 1, 1}, {76, 1, f32}});
    };
    // Exact binary32 sums isolate the two binary16 block boundaries from
    // within-block accumulation order. Expected words are independent constants.
    unsigned rounding_failures = 0;
    for (unsigned probe = 0; probe < 5; ++probe) {
        Matrices m{};
        const float initial = probe == 2 ? -1.0f : probe == 3 ? 1.0f + std::ldexp(1.0f, -10)
                              : probe == 4 ? 65504.0f : 1.0f;
        const std::uint16_t expected[] = {0x0000, 0x3c02, 0x0000, 0x1400, 0x7c00};
        for (auto& row : m.c) for (auto& value : row) value = to_half(initial);
        for (unsigned row = 0; row < 16; ++row) {
            if (probe == 1) {
                for (unsigned k = 8; k < 16; ++k) m.a[row][k] = to_half(std::ldexp(1.0f, -12));
            } else {
                m.a[row][0] = to_half(probe == 4 ? 32.0f : std::ldexp(probe == 2 ? -1.0f : 1.0f, -11));
                m.a[row][8] = to_half(-initial);
                if (probe == 4) m.a[row][8] = to_half(-32.0f);
            }
        }
        for (auto& row : m.b) for (auto& value : row) value = to_half(1.0f);
        try {
            checked += run(gpu, program({hmma(16, 0, 4, 6, false), stg(10, 16, 48, 5)}), fragments(m),
                [&](unsigned, const Words&) { return Words(2, pack(expected[probe], expected[probe])); },
                "HMMA block rounding probe " + std::to_string(probe));
        } catch (const std::exception&) { ++rounding_failures; }
    }
    if (rounding_failures) throw std::runtime_error("native HMMA eight-product block rounding failed: " + std::to_string(rounding_failures));
    for (unsigned trial = 0; trial < 4; ++trial) {
        const auto m = random_matrices(rng);
        const auto memory = fragments(m);
        // binary16 accumulator from C in R6, R7, and from RZ (C = 0).
        for (const bool zero : {false, true}) {
            checked += run(gpu, program({hmma(16, 0, 4, zero ? 255 : 6, false), stg(10, 16, 48, 5)}), memory,
                [&](unsigned lane, const Words&) {
                    const unsigned g = lane / 4, t = lane % 4;
                    Words out;
                    for (unsigned half_row = 0; half_row < 2; ++half_row) {
                        const unsigned row = g + 8 * half_row;
                        auto element = [&](unsigned column) {
                            return to_half(product(m, row, column, zero ? 0.0f : from_half(m.c[row][column])));
                        };
                        out.push_back(pack(element(2 * t), element(2 * t + 1)));
                    }
                    return out;
                }, zero ? "HMMA.16816.F16 RZ" : "HMMA.16816.F16");
        }
    }
    // One element whose binary32 partial sums can land on a binary16 tie:
    // C = 1, A[0][0] B[0][0] = 2^-11 + 2^-21, A[0][1] B[1][0] = -2^-21 +
    // 2^-24. The exact sum 1 + 2^-11 + 2^-24 rounds once to 0x3c01; an order
    // that rounds 1 + 2^-11 + 2^-24 to binary32 first lands on the tie and
    // gives 0x3c00. Either is admitted; every other element is zero or C.
    {
        Matrices m{};
        m.a[0][0] = 0x3c01; m.b[0][0] = 0x1000; m.a[0][1] = 0x0008; m.b[1][0] = 0xbb00; m.c[0][0] = 0x3c00;
        auto fixture = program({hmma(16, 0, 4, 6, false), stg(10, 16, 48, 5)});
        auto compiled = gpu.compile_compute(fixture);
        MetalHarness::GuestMemory guest{base, fragments(m), {}};
        gpu.dispatch(compiled, fixture, {std::uint32_t(base), std::uint32_t(base >> 32), record_words * 4}, guest);
        for (unsigned lane = 0; lane < 32; ++lane) for (unsigned w = 0; w < 2; ++w) {
            const auto got = guest.words[lane * record_words + 12 + w];
            const bool tie = lane == 0 && w == 0;
            const bool ok = tie ? ((got & 0xffffu) == 0x3c00u || (got & 0xffffu) == 0x3c01u) && (got >> 16) == 0 : got == 0;
            if (!ok) {
                std::fprintf(stderr, "HMMA tie case lane %u word %u: got %08x\n", lane, w, got);
                throw std::runtime_error("native warp matrix tie case");
            }
        }
        checked += 32;
    }
    // The binary32 accumulator is refused.
    check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(program({hmma(16, 0, 4, 12, true), exit_instruction()}).program())),
          "HMMA with a binary32 accumulator is refused");
    // MOVM.16.MT88: lane L holds row L / 4, columns 2 (L % 4), + 1 of an 8x8
    // block of halves; afterwards it holds the transposed block's.
    Words block(32 * record_words, 0);
    std::uint16_t element[8][8];
    for (unsigned r = 0; r < 8; ++r) for (unsigned c = 0; c < 8; ++c) element[r][c] = std::uint16_t(0x1000 + r * 16 + c);
    for (unsigned lane = 0; lane < 32; ++lane) block[lane * record_words] = pack(element[lane / 4][2 * (lane % 4)], element[lane / 4][2 * (lane % 4) + 1]);
    checked += run(gpu, program({op(0x23a, {{16, 8, 16}, {24, 8, 0}}), stg(10, 16, 48, 4)}), block,
        [&](unsigned lane, const Words&) {
            const unsigned r = lane / 4, t = lane % 4;
            return Words{pack(element[2 * t][r], element[2 * t + 1][r])};
        }, "MOVM.16.MT88");
    // LDSM.16.M88.4 and .MT88.4: each lane stores its record's four words
    // (A) to shared memory at lane * 16 with STS.128, so lane L holds row
    // L % 8 of block L / 8, and gives that row's address to LDSM.
    for (unsigned transposed = 0; transposed < 2; ++transposed) {
        const auto m = random_matrices(rng);
        const auto memory = fragments(m);
        auto sts = op(0x388, {{24, 8, 22}, {32, 8, 0}, {40, 24, 0}, {73, 3, 6}});
        auto ldsm = op(0x83b, {{16, 8, 24}, {24, 8, 22}, {72, 2, 2}, {78, 2, transposed}});
        auto fixture = program({load(23, 3), op(0x224, {{16, 8, 22}, {24, 8, 20}, {32, 8, 23}, {64, 8, 255}, {81, 3, PT}, {87, 4, NOT_PT}}),
                                sts, ldsm, stg(10, 24, 48, 6)});
        fixture.launch({32, 1, 1}, 512);
        auto compiled = gpu.compile_compute(fixture);
        MetalHarness::GuestMemory guest{base, memory, {}};
        gpu.dispatch(compiled, fixture, {std::uint32_t(base), std::uint32_t(base >> 32), record_words * 4, 16}, guest);
        auto row_word = [&](unsigned row_lane, unsigned w) { return memory[row_lane * record_words + w]; };
        for (unsigned lane = 0; lane < 32; ++lane) {
            const unsigned g = lane / 4, t = lane % 4;
            for (unsigned block = 0; block < 4; ++block) {
                std::uint32_t want;
                if (!transposed) want = row_word(8 * block + g, t);
                else {
                    auto half_at = [&](unsigned row) { return (row_word(8 * block + row, g / 2) >> (16 * (g % 2))) & 0xffffu; };
                    want = half_at(2 * t) | half_at(2 * t + 1) << 16;
                }
                const auto got = guest.words[lane * record_words + 12 + block];
                if (got != want) {
                    std::fprintf(stderr, "LDSM%s lane %u block %u: got %08x want %08x\n", transposed ? ".MT88" : "", lane, block, got, want);
                    throw std::runtime_error("native warp matrix load GPU result");
                }
            }
            ++checked;
        }
    }
    // Three HMMAs on one A and B: D1 = A B + C, D2 = A B + 0, and D3 = A B +
    // D1, which nothing else reads, so D1 stays in the SIMD-group's matrices
    // across the unchained D2. A is the lanes' registers or an LDSM's four
    // blocks (plain or .MT88), and B the registers or the same rows' first
    // two blocks; the rows lie at lane * 16 (one stride: the matrix hardware
    // reads them in place) or lane * lane * 16 (no stride: the lanes copy
    // them). Block q is A's quadrant q and B's half q, as an LDSM's words are.
    struct Chain { const char* name; bool ldsm; unsigned transposed; bool strided; bool b_ldsm; };
    for (const auto& chain : {Chain{"registers", false, 0, true, false}, Chain{"LDSM.M88", true, 0, true, false},
                              Chain{"LDSM.MT88", true, 1, true, false}, Chain{"LDSM.M88 unstrided", true, 0, false, false},
                              Chain{"LDSM.MT88 unstrided", true, 1, false, false}, Chain{"LDSM.M88 A and B", true, 0, true, true},
                              Chain{"LDSM.MT88 A and B", true, 1, true, true}, Chain{"LDSM.M88 A and B unstrided", true, 0, false, true},
                              Chain{"LDSM.MT88 A and B unstrided", true, 1, false, true}}) {
        const auto m = random_matrices(rng);
        const auto memory = fragments(m);
        std::vector<sn::Instruction> body;
        unsigned a = 0;
        if (chain.ldsm) {
            auto imad = [](unsigned d, unsigned x, unsigned y) {
                return op(0x224, {{16, 8, d}, {24, 8, x}, {32, 8, y}, {64, 8, 255}, {81, 3, PT}, {87, 4, NOT_PT}});
            };
            body = {load(23, 3)};
            if (chain.strided) body.push_back(imad(22, 20, 23));
            else { body.push_back(imad(22, 20, 20)); body.push_back(imad(22, 22, 23)); }
            body.push_back(op(0x388, {{24, 8, 22}, {32, 8, 0}, {40, 24, 0}, {73, 3, 6}}));
            body.push_back(op(0x83b, {{16, 8, 24}, {24, 8, 22}, {72, 2, 2}, {78, 2, chain.transposed}}));
            if (chain.b_ldsm) body.push_back(op(0x83b, {{16, 8, 4}, {24, 8, 22}, {72, 2, 1}, {78, 2, chain.transposed}}));
            a = 24;
        }
        body.insert(body.end(), {hmma(16, a, 4, 6, false), hmma(18, a, 4, 255, false), hmma(28, a, 4, 16, false),
                                 stg(10, 28, 48, 5), stg(10, 18, 56, 5)});
        auto fixture = program(body);
        fixture.launch({32, 1, 1}, chain.strided ? 512 : 31 * 31 * 16 + 16);
        auto compiled = gpu.compile_compute(fixture);
        // Call sites: A placed once (or loaded as tiles, with no lane words),
        // B placed once, D read back only for D2 and D3.
        auto calls = [&](const std::string& call) {
            unsigned n = 0;
            for (auto at = compiled.artifact.source.find(call); at != std::string::npos; at = compiled.artifact.source.find(call, at + 1)) ++n;
            return n;
        };
        check(calls("armigo_mma_put_a(armigo_mma_block") == (chain.ldsm ? 0u : 1u) &&
              calls("armigo_ldsm_tiles(armigo_t") == (chain.ldsm ? 1u : 0u) && calls("armigo_ldsm_tiles(armigo_u") == (chain.b_ldsm ? 1u : 0u) &&
              calls("armigo_ldsm(armigo_shared_words") == 0 && calls("armigo_mma_put_b(armigo_mma_block") == (chain.b_ldsm ? 0u : 1u) &&
              calls("= armigo_mma_lanes(armigo_mma_block") == 2,
              "HMMA operands stay where the matrix plan keeps them");
        // Element (r, c) of block q: half c of the record of lane 8q + r.
        auto block_element = [&](unsigned q, unsigned r, unsigned c) {
            return std::uint16_t(memory[(8 * q + r) * record_words + c / 2] >> (16 * (c % 2)));
        };
        Matrices product_of = m;
        if (chain.ldsm)
            for (unsigned row = 0; row < 16; ++row) for (unsigned k = 0; k < 16; ++k) {
                const unsigned q = row / 8 + 2 * (k / 8);
                product_of.a[row][k] = chain.transposed ? block_element(q, k % 8, row % 8) : block_element(q, row % 8, k % 8);
            }
        if (chain.b_ldsm)
            for (unsigned k = 0; k < 16; ++k) for (unsigned n = 0; n < 8; ++n)
                product_of.b[k][n] = chain.transposed ? block_element(k / 8, k % 8, n) : block_element(k / 8, n, k % 8);
        MetalHarness::GuestMemory guest{base, memory, {}};
        gpu.dispatch(compiled, fixture, {std::uint32_t(base), std::uint32_t(base >> 32), record_words * 4, 16}, guest);
        for (unsigned lane = 0; lane < 32; ++lane) {
            const unsigned g = lane / 4, t = lane % 4;
            for (unsigned w = 0; w < 4; ++w) {
                const unsigned row = g + 8 * (w & 1);
                auto element = [&](unsigned column) {
                    const float c = w < 2 ? from_half(to_half(product(product_of, row, column, from_half(m.c[row][column])))) : 0.0f;
                    return to_half(product(product_of, row, column, c));
                };
                const auto want = pack(element(2 * t), element(2 * t + 1)), got = guest.words[lane * record_words + 12 + w];
                if (got != want) {
                    std::fprintf(stderr, "HMMA chain (%s) lane %u word %u: got %08x want %08x\n", chain.name, lane, w, got, want);
                    throw std::runtime_error("native warp matrix chain GPU result");
                }
            }
            ++checked;
        }
    }
    // D as A: D1 = A B + C and D2 = A B form A' = [D1 | D2] (k 0-7 from D1,
    // 8-15 from D2) of D3 = A' B, as an attention network's second product
    // takes its first. D1 is read only as matrices, D2 also by the lanes.
    // Small integers keep every sum exact.
    {
        std::uniform_int_distribution<int> small(-3, 3);
        Matrices m{};
        for (auto& row : m.a) for (auto& x : row) x = to_half(float(small(rng)));
        for (auto& row : m.b) for (auto& x : row) x = to_half(float(small(rng)));
        for (auto& row : m.c) for (auto& x : row) x = to_half(float(small(rng)));
        auto fixture = program({hmma(16, 0, 4, 6, false), hmma(18, 0, 4, 255, false), hmma(28, 16, 4, 255, false),
                                stg(10, 28, 48, 5), stg(10, 18, 56, 5)});
        auto compiled = gpu.compile_compute(fixture);
        auto calls = [&](const std::string& call) {
            unsigned n = 0;
            for (auto at = compiled.artifact.source.find(call); at != std::string::npos; at = compiled.artifact.source.find(call, at + 1)) ++n;
            return n;
        };
        check(calls("armigo_mma_put_a(armigo_mma_block") == 1 && calls("= armigo_mma_lanes(armigo_mma_block") == 2,
              "an HMMA takes earlier D matrices as its A");
        Matrices second = m;
        for (unsigned row = 0; row < 16; ++row) for (unsigned k = 0; k < 16; ++k)
            second.a[row][k] = k < 8 ? to_half(product(m, row, k, from_half(m.c[row][k]))) : to_half(product(m, row, k - 8, 0.0f));
        MetalHarness::GuestMemory guest{base, fragments(m), {}};
        gpu.dispatch(compiled, fixture, {std::uint32_t(base), std::uint32_t(base >> 32), record_words * 4}, guest);
        for (unsigned lane = 0; lane < 32; ++lane) {
            const unsigned g = lane / 4, t = lane % 4;
            for (unsigned w = 0; w < 4; ++w) {
                const unsigned row = g + 8 * (w & 1);
                auto element = [&](unsigned column) { return to_half(product(w < 2 ? second : m, row, column, 0.0f)); };
                const auto want = pack(element(2 * t), element(2 * t + 1)), got = guest.words[lane * record_words + 12 + w];
                if (got != want) {
                    std::fprintf(stderr, "HMMA D as A lane %u word %u: got %08x want %08x\n", lane, w, got, want);
                    throw std::runtime_error("native warp matrix D-as-A GPU result");
                }
            }
            ++checked;
        }
    }
    std::printf("Native warp matrix Metal execution passed (%u lanes)\n", checked);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
