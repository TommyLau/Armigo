// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Metal-executed task and mesh stages. A mesh stage writes four vertices of
// a full-target quad (position and generic 0.x), the primitive count and the
// index bytes to its stage buffer; the drawn pixels show its workgroup's
// number. Under a task stage, the task writes two mesh launches and a payload
// record; each mesh workgroup draws one triangle of the quad by its number
// among the launches, so the quad covers the target only when both ran with
// their own numbers, and the pixels show the record.

#include "metal_harness.h"
#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <tuple>

namespace sn = armigo::shader_native;
using namespace sn::test;
namespace {
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
constexpr unsigned PT = 7, NOT_PT = 15;
sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
    auto i = raw(opcode);
    for (auto [first, count, value] : fields) field(i, first, count, value);
    return i;
}
// ISBERD (0x923) Rd or ISBEWR (0x927) data, [Ra + offset], mode.
sn::Instruction isbe(unsigned opcode, unsigned vector, unsigned ra, std::uint32_t offset, unsigned mode) {
    return op(opcode, {{opcode == 0x927 ? 32u : 16u, 8, vector}, {24, 8, ra}, {40, 24, offset}, {72, 8, mode}});
}
sn::Instruction attribute(unsigned data, unsigned row, unsigned vertex) { return isbe(0x927, data, 255, row * 128 + vertex * 4, 0xf8); }
sn::Instruction index_byte(unsigned data, unsigned at) { return isbe(0x927, data, 255, 4 + at, 0x80); }
// ISETP Pfirst = Ra relation Rb (unsigned; relation 2 is EQ).
sn::Instruction isetp(unsigned a, unsigned b, unsigned relation, unsigned first) {
    return op(0x20c, {{24, 8, a}, {32, 8, b}, {68, 4, PT}, {76, 3, relation}, {81, 3, first}, {84, 3, PT}, {87, 4, PT}});
}
sn::Instruction guarded(sn::Instruction i, unsigned predicate) { field(i, 12, 4, predicate); return i; }
// IMAD Rd = Ra * Rb + Rc
sn::Instruction imad(unsigned d, unsigned a, unsigned b, unsigned c) {
    return op(0x224, {{16, 8, d}, {24, 8, a}, {32, 8, b}, {64, 8, c}, {81, 3, PT}, {87, 4, NOT_PT}});
}
// Vertex v of the quad: x -1 or 1 by bit 0, y by bit 1, z 0, w 1, generic 0.x
// the word in register value. R1 holds -1.0 and R2 1.0.
void quad(std::vector<sn::Instruction>& code, unsigned value) {
    for (unsigned v = 0; v < 4; ++v) {
        code.push_back(attribute(v & 1 ? 2 : 1, 0, v));
        code.push_back(attribute(v & 2 ? 2 : 1, 1, v));
        code.push_back(attribute(255, 2, v));
        code.push_back(attribute(2, 3, v));
        code.push_back(attribute(value, 4, v));
    }
}
// 32 threads; a mesh stage outputs its position and generic 0.x.
Fixture workgroup(std::vector<sn::Instruction> code, sn::Stage stage, std::uint32_t payload, bool task = false,
                  std::uint32_t max_vertices = 4, std::uint32_t max_primitives = 2) {
    code.push_back(exit_instruction());
    Fixture f(std::move(code), stage);
    put(f.control, 0x58, 32 | (payload << 16));
    if (stage == sn::Stage::mesh) {
        f.code[0x36] = std::byte{1};
        put(f.control, 0x5c, task ? 0x03010000u : 0x03000000u);
        put(f.control, 0x60, max_vertices | (max_primitives << 16));
    }
    return f;
}
unsigned covered(const std::vector<std::array<std::uint32_t, 4>>& pixels) {
    return static_cast<unsigned>(std::count_if(pixels.begin(), pixels.end(), [](const auto& p) { return p[1] == 1; }));
}
// A task launching count mesh workgroups (payload word 1) with the record
// 0x1230 + its workgroup number, over a mesh that draws the full quad only
// in its workgroup last.
std::pair<Fixture, Fixture> launches(std::uint32_t count, std::uint32_t last) {
    auto task = workgroup({move(1, count), move(2, 0x1230), move(5, 1), isbe(0x923, 3, 255, 0, 0x78), imad(4, 3, 5, 2),
                           isbe(0x927, 1, 255, 4, 0xb8), isbe(0x927, 4, 255, 0x10, 0xb8)},
                          sn::Stage::task, 0x20);
    std::vector<sn::Instruction> code{move(1, 0xbf800000u), move(2, 0x3f800000u), move(4, 2), move(6, 1), move(7, 2), move(8, 3),
                                      move(14, last), isbe(0x923, 10, 255, 0, 0x38), isbe(0x923, 11, 255, 0x10, 0x38), isetp(10, 14, 2, 0)};
    quad(code, 11);
    for (const auto& i : {guarded(isbe(0x927, 4, 255, 3, 0x88), 0), index_byte(255, 0), index_byte(6, 1), index_byte(7, 2),
                          index_byte(7, 3), index_byte(6, 4), index_byte(8, 5)})
        code.push_back(i);
    return {std::move(task), workgroup(code, sn::Stage::mesh, 0x20, true)};
}
void expect(const std::vector<std::array<std::uint32_t, 4>>& pixels, std::uint32_t value, const char* what) {
    for (std::size_t i = 0; i < pixels.size(); ++i)
        if (pixels[i] != std::array<std::uint32_t, 4>{value, 1, 0, 0}) {
            std::fprintf(stderr, "%s: pixel %zu is (%x, %u, %u, %u), expected (%x, 1, 0, 0)\n", what, i, pixels[i][0], pixels[i][1],
                         pixels[i][2], pixels[i][3], value);
            throw std::runtime_error("native mesh GPU result");
        }
}
} // namespace

int main() { @autoreleasepool { try {
    MetalHarness gpu;
    // Mesh only: the workgroup's number is the draw's first workgroup.
    std::vector<sn::Instruction> mesh_code{move(1, 0xbf800000u), move(2, 0x3f800000u), move(4, 2), move(6, 1), move(7, 2), move(8, 3),
                                           isbe(0x923, 9, 255, 0, 0x78)};
    quad(mesh_code, 9);
    for (const auto& i : {isbe(0x927, 255, 255, 3, 0x88), isbe(0x927, 4, 255, 3, 0x88), index_byte(255, 0), index_byte(6, 1),
                          index_byte(7, 2), index_byte(7, 3), index_byte(6, 4), index_byte(8, 5)})
        mesh_code.push_back(i);
    const auto mesh = workgroup(mesh_code, sn::Stage::mesh, 0);
    expect(gpu.mesh_grid(nullptr, mesh, 7, 1, 4, 4), 7, "mesh stage quad");

    // Paired with a fragment stage that reads generic 0.x, 2.y and 2.w, the
    // mesh outputs generic 0.x and the defaults 0.0 and 1.0 for the
    // components no output map enables; a written component the fragment
    // does not read (generic 1.x) is not output.
    {
        check(sn::mesh_generic_outputs(mesh.program())[0] == 1, "the mesh output maps enable generic 0.x");
        std::vector<sn::Instruction> code{move(1, 0xbf800000u), move(2, 0x3f800000u), move(4, 2), move(6, 1), move(7, 2), move(8, 3),
                                          isbe(0x923, 9, 255, 0, 0x78), move(10, 0x55)};
        quad(code, 9);
        for (unsigned v = 0; v < 4; ++v) code.push_back(attribute(10, 5, v));
        for (const auto& i : {isbe(0x927, 4, 255, 3, 0x88), index_byte(255, 0), index_byte(6, 1), index_byte(7, 2), index_byte(7, 3),
                              index_byte(6, 4), index_byte(8, 5)})
            code.push_back(i);
        auto two = workgroup(code, sn::Stage::mesh, 0);
        two.code[0x36] = std::byte{0x11};  // generic 0.x and 1.x
        check(sn::mesh_generic_outputs(two.program())[1] == 1, "the mesh output maps enable generic 1.x");
        std::array<std::uint8_t, 32> reads{};
        reads[0] = 1; reads[2] = 0xa;
        gpu.mesh_fragment_reads = reads;
        gpu.mesh_fragment_source = R"(#include <metal_stdlib>
using namespace metal;
struct MeshTestInput { float4 position [[position]]; float generic0_0 [[user(locn0_0), flat]];
    float generic2_1 [[user(locn2_1), flat]]; float generic2_3 [[user(locn2_3), flat]]; };
fragment uint4 mesh_test_fragment(MeshTestInput in [[stage_in]]) {
    return uint4(as_type<uint>(in.generic0_0), as_type<uint>(in.generic2_1), as_type<uint>(in.generic2_3), 1u);
})";
        const auto pixels = gpu.mesh_grid(nullptr, two, 7, 1, 4, 4);
        for (const auto& p : pixels)
            check(p == std::array<std::uint32_t, 4>{7, 0, 0x3f800000u, 1}, "a paired mesh outputs the fragment's reads with their defaults");
        check(gpu.mesh_source.find("locn1_0") == std::string::npos, "a component the fragment does not read is not output");
        gpu.mesh_fragment_reads.reset(); gpu.mesh_fragment_source.clear();
        gpu.mesh_grid(nullptr, two, 7, 1, 4, 4);
        check(gpu.mesh_source.find("locn1_0") != std::string::npos, "without a partner every enabled component is output");
    }

    // Task and mesh: two launches of one triangle each, the record 0x1230
    // plus the task's workgroup number.
    const auto task = workgroup({move(1, 2), move(2, 0x1230), move(5, 1), isbe(0x923, 3, 255, 0, 0x78), imad(4, 3, 5, 2),
                                 isbe(0x927, 1, 255, 4, 0xb8), isbe(0x927, 4, 255, 0x10, 0xb8)},
                                sn::Stage::task, 0x20);
    std::vector<sn::Instruction> launched{move(1, 0xbf800000u), move(2, 0x3f800000u), move(4, 1), move(6, 1), move(7, 2),
                                          isbe(0x923, 10, 255, 0, 0x38), isbe(0x923, 11, 255, 0x10, 0x38)};
    quad(launched, 11);
    // Workgroup n draws triangle (2n, 1, n + 2): (0, 1, 2), then (2, 1, 3).
    for (const auto& i : {isbe(0x927, 4, 255, 3, 0x88), imad(12, 10, 7, 255), imad(13, 10, 6, 7),
                          index_byte(12, 0), index_byte(6, 1), index_byte(13, 2)})
        launched.push_back(i);
    const auto under_task = workgroup(launched, sn::Stage::mesh, 0x20, true, 4, 1);
    expect(gpu.mesh_grid(&task, under_task, 3, 1, 4, 4), 0x1233, "task and mesh stages");
    check(gpu.mesh_status == (std::array<std::uint32_t, 2>{0, 0}), "a task within Metal's mesh grid records no status");

    // Metal launches at most 1,024 mesh workgroups from one object
    // threadgroup: 1,024 reach the last one; 1,025 launch none and record the
    // request and the task program.
    {
        auto [task, mesh] = launches(1024, 1023);
        expect(gpu.mesh_grid(&task, mesh, 0, 1, 4, 4), 0x1230, "1,024 mesh workgroups");
        check(gpu.mesh_status == (std::array<std::uint32_t, 2>{0, 0}), "1,024 mesh workgroups record no status");
    }
    {
        auto [task, mesh] = launches(1025, 0);
        const auto pixels = gpu.mesh_grid(&task, mesh, 0, 1, 4, 4);
        check(covered(pixels) == 0, "a task asking for 1,025 mesh workgroups launches none");
        const auto tag = static_cast<std::uint32_t>(std::stoul(task.program().stage_identity.substr(0, 8), nullptr, 16));
        check(gpu.mesh_status == (std::array<std::uint32_t, 2>{1025, tag}), "the refused request and its program are recorded");
    }
    // A payload shorter than 8 bytes holds no launch count: it launches
    // none, even when the stage stores a count in word 1.
    for (const std::uint32_t payload : {4u, 5u, 7u, 8u}) {
        const auto task = workgroup({move(1, 1), isbe(0x927, 1, 255, 4, 0xb8)}, sn::Stage::task, payload);
        auto [unused, mesh] = launches(1, 0);
        check(covered(gpu.mesh_grid(&task, mesh, 0, 1, 4, 4)) == (payload >= 8 ? 16u : 0u), "short payloads launch no mesh workgroup");
    }
    // Point meshes: one point at a pixel centre covers one pixel without a
    // point size in the output map (1.0), four with a mapped size of 2.0.
    for (const bool sized : {false, true}) {
        const std::uint32_t at = sized ? 0xbf000000u : 0xbf400000u;  // -0.5 or -0.75
        const unsigned rows = sized ? 1 : 0;
        std::vector<sn::Instruction> code{move(1, at), move(2, 0x3f800000u), move(3, 0x40000000u), move(4, 1)};
        if (sized) code.push_back(attribute(3, 0, 0));
        for (const auto& i : {attribute(1, rows, 0), attribute(1, rows + 1, 0), attribute(255, rows + 2, 0), attribute(2, rows + 3, 0),
                              attribute(4, rows + 4, 0), isbe(0x927, 4, 255, 3, 0x88), index_byte(255, 0)})
            code.push_back(i);
        auto points = workgroup(code, sn::Stage::mesh, 0, false, 1, 1);
        points.control[0x5f] = std::byte{1};
        if (sized) points.code[0x35] |= std::byte{8};
        check(covered(gpu.mesh_grid(nullptr, points, 0, 1, 4, 4)) == (sized ? 4u : 1u), "point mesh size");
    }
    std::printf("Native task and mesh Metal execution passed\n");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
