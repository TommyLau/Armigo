// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal execution of native texture pools (the project documentation).
// Each fragment reads one handle from constant bank 3 at a byte offset of
// four times its pixel column, so the handle is neither a constant-buffer
// key nor a handle program. texture_pools routes that read to the pool
// tables the harness binds.

#include "metal_harness.h"
#include <bit>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <tuple>
#include <vector>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
using Pixel = std::array<std::uint32_t, 4>;
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
constexpr unsigned kWidth = 8, kRows = 2, kPad = 16;
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
// Packed 2D sample at an explicit level of zero (level mode 1). The handle
// register follows the coordinates. Mask 0x1 keeps the red channel.
sn::Instruction sample(unsigned dest, unsigned coords, unsigned handle) {
    return op(0x361, {{16, 8, dest}, {24, 8, coords}, {32, 8, handle}, {59, 1, 1}, {60, 1, 1},
                      {61, 3, 1}, {64, 8, 255}, {72, 4, 0x1}, {81, 3, 7}, {87, 3, 1}});
}
sn::FrontendOptions pool_options() {
    sn::FrontendOptions options;
    options.texture_pools = true;
    return options;
}
// PositionX, truncated to the column, shifted to a byte offset, then loaded
// from c[3]. A one-quad helper just past the target still lands in the
// copied prefix when the input is kPad words.
Fixture indexed_stage(bool kill) {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    code.push_back(op(0x326, {{16, 8, 4}, {32, 8, 255}, {64, 8, 0x70 / 4}, {81, 3, 7}}));
    code.push_back(op(0x305, {{16, 8, 4}, {32, 8, 4}, {75, 2, 2}, {77, 1, 1}, {78, 2, 3}, {80, 1, 1}, {84, 2, 2}}));
    code.push_back(op(0x211, {{16, 8, 5}, {24, 8, 4}, {32, 8, 255}, {64, 8, 255}, {75, 5, 2}, {81, 3, 7}, {87, 4, 15}}));
    code.push_back(op(0xb82, {{16, 8, 12}, {24, 8, 5}, {54, 5, 3}, {73, 3, 4}}));
    code.push_back(move(8, 0x3f000000u));
    code.push_back(move(9, 0x3f000000u));
    code.push_back(sample(0, 8, 12));
    if (kill) code.push_back(op(0x95b, {{87, 4, 7}}));
    code.push_back(exit_instruction());
    Fixture fixture(code);
    put(fixture.code, 0x48, 0xf);
    fixture.code[0x17] = std::byte{0x10};
    return fixture;
}
// Two samples of one texture through two sampler IDs. The handles are
// immediates, which are not constant-buffer keys, so both pools are read.
Fixture sampler_stage() {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    code.push_back(move(8, bits(1.1f)));
    code.push_back(move(9, bits(0.1f)));
    code.push_back(move(12, 1u | (1u << 20)));
    code.push_back(move(13, 1u | (2u << 20)));
    code.push_back(sample(0, 8, 12));
    code.push_back(sample(1, 8, 13));
    code.push_back(exit_instruction());
    Fixture fixture(code);
    put(fixture.code, 0x48, 0xf);
    return fixture;
}
void require_text(const std::string& source, const char* text, const char* what) {
    if (source.find(text) == std::string::npos) {
        std::fprintf(stderr, "pool source missing %s\n", what);
        throw std::runtime_error(what);
    }
}
void require_absent(const std::string& source, const char* text, const char* what) {
    if (source.find(text) != std::string::npos) {
        std::fprintf(stderr, "pool source unexpectedly contains %s\n", what);
        throw std::runtime_error(what);
    }
}
void require_pools(const ir::MslArtifact& artifact) {
    unsigned textures = 0, samplers = 0;
    for (const auto& resource : artifact.reflection.resources) {
        const auto* range = std::get_if<ir::MetalRange>(&resource.location);
        const bool buffer = range && range->space == ir::MetalNamespace::buffer && range->count == 1;
        if (const auto* texture = std::get_if<ir::TextureUse>(&resource.use); texture && texture->pool)
            textures += buffer && texture->key == ir::TextureKey{ir::texture_pool_slot, 0};
        if (const auto* sampler = std::get_if<ir::SamplerUse>(&resource.use); sampler && sampler->pool)
            samplers += buffer && sampler->key == ir::TextureKey{ir::texture_pool_slot, 0};
    }
    check(textures == 1 && samplers == 1, "one texture pool and one sampler pool");
}
void expect_pixel(const char* name, unsigned x, unsigned y, const Pixel& got, const Pixel& want) {
    if (got == want) return;
    std::fprintf(stderr, "%s pixel %u,%u: got %08x %08x %08x %08x, expected %08x %08x %08x %08x\n",
                 name, x, y, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
    throw std::runtime_error(name);
}
void expect_bitmap(const char* name, const std::vector<std::uint32_t>& got, std::initializer_list<std::uint32_t> want) {
    const std::vector<std::uint32_t> expected(want);
    if (got == expected) return;
    std::fprintf(stderr, "%s bitmap:", name);
    for (auto word : got) std::fprintf(stderr, " %08x", word);
    std::fprintf(stderr, ", expected");
    for (auto word : expected) std::fprintf(stderr, " %08x", word);
    std::fprintf(stderr, "\n");
    throw std::runtime_error(name);
}
MetalHarness::TexturePool color_pool(std::uint32_t n, float at_zero, float at_n) {
    MetalHarness::TexturePool pool;
    pool.n = n;
    pool.entries.resize(n + 1);
    for (std::uint32_t id = 0; id <= n; ++id) {
        float red = float(id);
        if (id == 0) red = at_zero;
        if (id == n) red = at_n;
        pool.entries[id].color = {red, 0.f, 0.f, 1.f};
    }
    return pool;
}
MetalHarness::SamplerPool nearest_clamp(std::uint32_t n) {
    MetalHarness::SamplerPool pool;
    pool.n = n;
    pool.entries.resize(n + 1);
    return pool;
}
void bind_pools(MetalHarness& gpu, MetalHarness::TexturePool textures, MetalHarness::SamplerPool samplers) {
    gpu.depth_target = false;
    gpu.occlude_columns = 0;
    gpu.texture_pool = std::move(textures);
    gpu.sampler_pool = std::move(samplers);
}
// n = 34. Columns read IDs 0, 1, 33, 34, 40, 32, 3 and 31. ID 0 is the
// dummy stored at entry 0; IDs at and past n read the other dummy and mark
// nothing. Padding words repeat ID 0 so a helper just past column 7 sets a
// bit this draw already expects.
void ids_and_bitmap(MetalHarness& gpu, const MetalHarness::Compiled& compiled) {
    bind_pools(gpu, color_pool(34, 1000.f, 2000.f), nearest_clamp(1));
    const std::uint32_t handles[kWidth] = {0, 1, 33, 34, 40, 32, 3, 31};
    std::vector<std::uint32_t> input(kPad, 0);
    for (unsigned x = 0; x < kWidth; ++x) input[x] = handles[x];
    const auto pixels = gpu.grid(compiled, input, kWidth, kRows);
    check(pixels.size() == kWidth * kRows, "id grid size");
    for (unsigned y = 0; y < kRows; ++y) for (unsigned x = 0; x < kWidth; ++x) {
        const auto id = handles[x];
        const float red = id == 0 ? 1000.f : id >= 34 ? 2000.f : float(id);
        expect_pixel("texture ids", x, y, pixels[y * kWidth + x], Pixel{bits(red), 0, 0, 0});
    }
    // Bits of the in-range IDs: 0, 1, 3, 31 in word 0 and 32, 33 in word 1.
    expect_bitmap("texture ids", gpu.pool_bitmap, {0x8000000bu, 0x3u});
}
// The same draw with a zero bitmap address: the pixels still resolve, and
// the allocated bitmap stays zero.
void zero_bitmap(MetalHarness& gpu, const MetalHarness::Compiled& compiled) {
    auto textures = color_pool(34, 1000.f, 2000.f);
    textures.mark = false;
    bind_pools(gpu, std::move(textures), nearest_clamp(1));
    const std::uint32_t handles[kWidth] = {0, 1, 33, 34, 40, 32, 3, 31};
    std::vector<std::uint32_t> input(kPad, 0);
    for (unsigned x = 0; x < kWidth; ++x) input[x] = handles[x];
    const auto pixels = gpu.grid(compiled, input, kWidth, kRows);
    for (unsigned y = 0; y < kRows; ++y) for (unsigned x = 0; x < kWidth; ++x) {
        const auto id = handles[x];
        const float red = id == 0 ? 1000.f : id >= 34 ? 2000.f : float(id);
        expect_pixel("zero bitmap address", x, y, pixels[y * kWidth + x], Pixel{bits(red), 0, 0, 0});
    }
    expect_bitmap("zero bitmap address", gpu.pool_bitmap, {0u, 0u});
}
// Sampler IDs 1 and 2 of one texture. Nearest clamp of u = 1.1 stays on the
// last column (red 13); nearest repeat wraps 1.1 to 0.1 and reads column 0
// (red 10). v = 0.1 is the first row, clear of a texel boundary.
void sampler_pools(MetalHarness& gpu) {
    MetalHarness::TexturePool textures;
    textures.n = 2;
    textures.entries.resize(3);
    textures.entries[1].width = 4;
    textures.entries[1].height = 4;
    textures.entries[1].column_red = true;
    textures.entries[1].column_base = 10.f;
    auto samplers = nearest_clamp(3);
    samplers.entries[2].address = MTLSamplerAddressModeRepeat;
    bind_pools(gpu, std::move(textures), std::move(samplers));
    const auto compiled = gpu.compile(sampler_stage(), pool_options());
    require_pools(compiled.artifact);
    require_text(compiled.artifact.source, " >> 20u, ", "sampler id shift");
    const auto pixels = gpu.grid(compiled, {}, 2, 2);
    const Pixel want{bits(13.f), bits(10.f), 0, 0};
    for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x)
        expect_pixel("sampler pool", x, y, pixels[y * 2 + x], want);
    expect_bitmap("sampler pool", gpu.pool_bitmap, {0x2u});
}
// An occluder writes depth over the first four columns. The marking stage
// is behind it and tests less-than, so those columns do not run and their
// IDs stay clear. Padding handles are n, which marks nothing if a helper
// past the target runs.
void early_tests(MetalHarness& gpu, const MetalHarness::Compiled& compiled) {
    bind_pools(gpu, color_pool(8, 0.f, 2000.f), nearest_clamp(1));
    gpu.depth_target = true;
    gpu.occlude_columns = 4;
    std::vector<std::uint32_t> input(kPad, 8);
    for (unsigned x = 0; x < kWidth; ++x) input[x] = x;
    const auto pixels = gpu.grid(compiled, input, kWidth, kRows);
    check(gpu.depths.size() == kWidth * kRows, "early depth size");
    for (unsigned y = 0; y < kRows; ++y) for (unsigned x = 0; x < kWidth; ++x) {
        const bool hidden = x < 4;
        const Pixel want = hidden ? Pixel{0, 0, 0, 0} : Pixel{bits(float(x)), 0, 0, 0};
        expect_pixel("early fragment tests", x, y, pixels[y * kWidth + x], want);
        const float depth = gpu.depths[y * kWidth + x];
        if (depth != (hidden ? 0.f : 1.f)) {
            std::fprintf(stderr, "early fragment tests depth %u,%u: %g\n", x, y, depth);
            throw std::runtime_error("early fragment tests depth");
        }
    }
    expect_bitmap("early fragment tests", gpu.pool_bitmap, {0xf0u});
    gpu.depth_target = false;
    gpu.occlude_columns = 0;
}
void kill_has_no_early_tests(MetalHarness& gpu) {
    const auto compiled = gpu.compile(indexed_stage(true), pool_options());
    require_pools(compiled.artifact);
    require_text(compiled.artifact.source, "armigo_pool_read(", "discarding stage pool read");
    require_absent(compiled.artifact.source, "early_fragment_tests", "early tests on a discarding stage");
}
// Unpacked 2D-array sample at an explicit level of zero. A holds the layer
// and both coordinates; B holds the handle.
sn::Instruction array_sample(unsigned dest, unsigned coords, unsigned handle) {
    return op(0x361, {{16, 8, dest}, {24, 8, coords}, {32, 8, handle}, {59, 1, 1}, {61, 3, 5},
                      {64, 8, 255}, {72, 4, 0x1}, {81, 3, 7}, {87, 3, 1}});
}
// One stage reads a 2D texture and a 2D-array texture. Each element type has
// its own pool; both sampler IDs read the stage's one sampler pool.
Fixture two_types_stage() {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    code.push_back(move(8, bits(0.5f)));
    code.push_back(move(9, bits(0.5f)));
    code.push_back(move(12, 1u | (1u << 20)));
    code.push_back(move(16, 0));
    code.push_back(move(17, bits(0.5f)));
    code.push_back(move(18, bits(0.5f)));
    code.push_back(sample(0, 8, 12));
    code.push_back(array_sample(1, 16, 12));
    code.push_back(exit_instruction());
    Fixture fixture(code);
    put(fixture.code, 0x48, 0xf);
    return fixture;
}
MetalHarness::TexturePool solid_pool(float red, bool array) {
    MetalHarness::TexturePool pool;
    pool.n = 2;
    pool.array = array;
    pool.entries.resize(3);
    pool.entries[1].color = {red, 0.f, 0.f, 1.f};
    pool.entries[0].color = {1.f, 0.f, 0.f, 1.f};
    pool.entries[2].color = {2.f, 0.f, 0.f, 1.f};
    return pool;
}
const MetalHarness::PoolBitmap* bitmap_for(const MetalHarness& gpu, bool array) {
    const MetalHarness::PoolBitmap* found = nullptr;
    for (const auto& bitmap : gpu.pool_bitmaps) {
        if (bitmap.dimension != ir::TextureDimension::d2 || bitmap.comparison || bitmap.array != array) continue;
        check(found == nullptr, "one bitmap for the element type");
        found = &bitmap;
    }
    check(found != nullptr, "element type bitmap");
    return found;
}
// ID 1 of each pool. The 2D texel is red 4 and the array texel is red 9, so
// the channels name which table was read. Each bitmap marks ID 1.
void two_element_types(MetalHarness& gpu) {
    gpu.texture_pool.reset();
    gpu.texture_pools = {solid_pool(4.f, false), solid_pool(9.f, true)};
    gpu.sampler_pool = nearest_clamp(2);
    const auto compiled = gpu.compile(two_types_stage(), pool_options());
    unsigned textures = 0, samplers = 0;
    for (const auto& resource : compiled.artifact.reflection.resources) {
        if (const auto* texture = std::get_if<ir::TextureUse>(&resource.use); texture && texture->pool) ++textures;
        if (const auto* sampler = std::get_if<ir::SamplerUse>(&resource.use); sampler && sampler->pool) {
            ++samplers;
            check(!sampler->comparison && sampler->normalized_coordinates, "one untyped sampler pool");
        }
    }
    check(textures == 2 && samplers == 1, "two texture pools and one sampler pool");
    require_text(compiled.artifact.source, "texture2d<float>", "2D pool declaration");
    require_text(compiled.artifact.source, "texture2d_array<float>", "2D-array pool declaration");
    const auto pixels = gpu.grid(compiled, {}, 1, 1);
    expect_pixel("two element types", 0, 0, pixels.at(0), Pixel{bits(4.f), bits(9.f), 0, 0});
    const auto* flat = bitmap_for(gpu, false);
    const auto* layered = bitmap_for(gpu, true);
    if (flat) expect_bitmap("2D element type", flat->words, {0x2u});
    if (layered) expect_bitmap("2D-array element type", layered->words, {0x2u});
    gpu.texture_pools.clear();
    gpu.sampler_pool.reset();
}
// The indexed stage also writes depth 0.25. Behind an occluder, with a
// depth state that writes, the occluded columns still run: their IDs are
// marked, and the late test keeps the occluder's depth.
Fixture depth_write_stage() {
    std::vector<sn::Instruction> code;
    for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
    code.push_back(op(0x326, {{16, 8, 4}, {32, 8, 255}, {64, 8, 0x70 / 4}, {81, 3, 7}}));
    code.push_back(op(0x305, {{16, 8, 4}, {32, 8, 4}, {75, 2, 2}, {77, 1, 1}, {78, 2, 3}, {80, 1, 1}, {84, 2, 2}}));
    code.push_back(op(0x211, {{16, 8, 5}, {24, 8, 4}, {32, 8, 255}, {64, 8, 255}, {75, 5, 2}, {81, 3, 7}, {87, 4, 15}}));
    code.push_back(op(0xb82, {{16, 8, 12}, {24, 8, 5}, {54, 5, 3}, {73, 3, 4}}));
    code.push_back(move(8, 0x3f000000u));
    code.push_back(move(9, 0x3f000000u));
    code.push_back(sample(0, 8, 12));
    code.push_back(move(5, bits(0.25f)));
    code.push_back(exit_instruction());
    Fixture fixture(code);
    put(fixture.code, 0x48, 0xf);
    fixture.code[0x4c] = std::byte{0x2};
    fixture.code[0x17] = std::byte{0x10};
    return fixture;
}
void depth_write_runs_when_occluded(MetalHarness& gpu) {
    const auto compiled = gpu.compile(depth_write_stage(), pool_options());
    require_pools(compiled.artifact);
    require_absent(compiled.artifact.source, "early_fragment_tests", "early tests on a depth-writing stage");
    require_text(compiled.artifact.source, "depth(any)", "the stage exports depth");
    bind_pools(gpu, color_pool(8, 0.f, 2000.f), nearest_clamp(1));
    gpu.depth_target = true;
    gpu.occlude_columns = 4;
    gpu.stage_writes_depth = true;
    std::vector<std::uint32_t> input(kPad, 8);
    for (unsigned x = 0; x < kWidth; ++x) input[x] = x;
    const auto pixels = gpu.grid(compiled, input, kWidth, kRows);
    check(gpu.depths.size() == kWidth * kRows, "depth-write size");
    for (unsigned y = 0; y < kRows; ++y) for (unsigned x = 0; x < kWidth; ++x) {
        const bool hidden = x < 4;
        const Pixel want = hidden ? Pixel{0, 0, 0, 0} : Pixel{bits(float(x)), 0, 0, 0};
        expect_pixel("depth-writing pool stage", x, y, pixels[y * kWidth + x], want);
        const float depth = gpu.depths[y * kWidth + x];
        const float expected = hidden ? 0.f : 0.25f;
        if (depth != expected) {
            std::fprintf(stderr, "depth-writing pool stage depth %u,%u: %g\n", x, y, depth);
            throw std::runtime_error("depth-writing pool stage depth");
        }
    }
    expect_bitmap("depth-writing pool stage", gpu.pool_bitmap, {0xffu});
    gpu.depth_target = false;
    gpu.occlude_columns = 0;
    gpu.stage_writes_depth = false;
}
}
int main() { @autoreleasepool { try {
    MetalHarness gpu;
    const auto compiled = gpu.compile(indexed_stage(false), pool_options());
    require_pools(compiled.artifact);
    require_text(compiled.artifact.source, "armigo_pool_read(", "pool read");
    require_text(compiled.artifact.source, " & 0xfffffu)", "texture id mask");
    require_text(compiled.artifact.source, "[[early_fragment_tests]] fragment ", "early tests on a marking stage");
    ids_and_bitmap(gpu, compiled);
    zero_bitmap(gpu, compiled);
    sampler_pools(gpu);
    early_tests(gpu, compiled);
    kill_has_no_early_tests(gpu);
    two_element_types(gpu);
    depth_write_runs_when_occluded(gpu);
    std::printf("Native texture pool Metal execution passed\n");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
