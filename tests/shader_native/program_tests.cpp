// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/program.h>
#include <iostream>

namespace sn = armigo::shader_native;
namespace {
void check(bool value) { if (!value) throw std::runtime_error("Synthetic program assertion failed"); }
void word(sn::Buffer& b, std::size_t p, std::uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) b.at(p + i) = std::byte(v >> (8 * i));
}
struct Fixture {
    sn::Buffer control = sn::Buffer(0x100), code = sn::Buffer(0x110);
    sn::Record record;
    Fixture(unsigned stage = 0) {
        word(control, 0, 0x98761234); word(control, 4, 0xc9); word(control, 8, 20);
        word(control, 0x30, control.size()); word(control, 0x38, 0xa0);
        word(control, 0x3c, 0x10); word(control, 0x40, 0x100); word(control, 0x44, code.size());
        control[0x56] = std::byte(stage);
        word(control, 0x58, 8); word(control, 0x5c, 2); word(control, 0x60, 1);
        code[0x100] = std::byte{0x42};
        record.control = control; record.code = code; record.kind = sn::ContainerKind::bnsh;
        record.slot = stage == 0 ? 0 : stage == 1 ? 4 : stage == 5 ? 5 : stage + 1;
    }
    sn::Program parse() { auto p = sn::parse_program(record); check(std::holds_alternative<sn::Program>(p)); return std::get<sn::Program>(p); }
    void rejects(sn::ProgramError error) { auto p = sn::parse_program(record); check(std::holds_alternative<sn::ProgramFailure>(p)); check(std::get<sn::ProgramFailure>(p).code == error); }
};
}
int main() {
    try {
        Fixture vertex; auto p = vertex.parse();
        check(p.header.size() == 128 && p.instructions.size() == 32 && p.padding.size() == 96 && p.constants.size() == 16);
        check(p.stage == sn::Stage::vertex && !p.launch);
        auto image = p.image_digest, identity = p.stage_identity;
        vertex.code[0x100] = std::byte{0x43}; p = vertex.parse();
        check(p.image_digest == image && p.stage_identity != identity); identity = p.stage_identity;
        // Bytes beyond the reader span and the container extents are file-only.
        vertex.control[0xf0] = std::byte{0x79}; p = vertex.parse();
        check(p.image_digest == image && p.stage_identity == identity && p.control_digest != p.canonical_control_digest);
        vertex.control[0xd0] = std::byte{0x79}; p = vertex.parse();
        check(p.stage_identity != identity); identity = p.stage_identity;
        Fixture tail; word(tail.control, 0x3c, 4);
        check(tail.parse().trailing_padding.size() == 12 && tail.parse().constants.size() == 4);
        tail.code.back() = std::byte{1}; tail.rejects(sn::ProgramError::nonzero_padding);
        Fixture fragment(1); fragment.control[0x57] = std::byte{0xab};
        check(fragment.parse().stage == sn::Stage::fragment);
        Fixture compute(5); p = compute.parse(); check(p.header.empty() && p.instructions.size() == 0xa0);
        check(p.launch && p.launch->local_size == (std::array<std::uint32_t, 3>{8, 2, 1}) && p.launch->shared_memory_bytes == 0);
        word(compute.control, 0x64, 0x1380); check(compute.parse().launch->shared_memory_bytes == 0x1380);
        // A thread group must hold 1 to 1,024 threads.
        word(compute.control, 0x5c, 0); compute.rejects(sn::ProgramError::program_bounds);
        word(compute.control, 0x5c, 129); compute.rejects(sn::ProgramError::program_bounds);
        for (auto stage : {2u, 3u, 4u, 255u}) { Fixture f(stage); f.rejects(sn::ProgramError::unsupported_stage); }
        // Task (6, BNSH slot 7) and mesh (7, slot 8) stages: a 128-byte
        // header and the launch facts their controls hold.
        {
            Fixture task(6);
            word(task.control, 0x58, 0x0c800020);  // 32 threads, 0xc80 payload bytes
            p = task.parse();
            check(p.stage == sn::Stage::task && p.header.size() == 128 && !p.launch && p.mesh_launch);
            check(p.mesh_launch->threads == 32 && p.mesh_launch->payload_bytes == 0xc80);
            Fixture mesh(7);
            word(mesh.control, 0x58, 0x0c800020); word(mesh.control, 0x5c, 0x03010000); word(mesh.control, 0x60, 0x00200060);
            p = mesh.parse();
            check(p.stage == sn::Stage::mesh && p.mesh_launch && p.mesh_launch->task && p.mesh_launch->primitive_vertices == 3 &&
                  p.mesh_launch->max_vertices == 0x60 && p.mesh_launch->max_primitives == 0x20 && !p.mesh_launch->primitive_header);
            // Topology 0 reads as triangles; above 3 is refused.
            word(mesh.control, 0x5c, 0x00000000); check(mesh.parse().mesh_launch->primitive_vertices == 3);
            mesh.control[0x5f] = std::byte{4}; mesh.rejects(sn::ProgramError::program_bounds);
            mesh.control[0x5f] = std::byte{1}; check(mesh.parse().mesh_launch->primitive_vertices == 1);
            // The per-primitive header (+0x64 when +0x68 bit 7) lies inside the program.
            word(mesh.control, 0x38, 0x100); mesh.control[0x68] = std::byte{0x80}; word(mesh.control, 0x64, 0x80);
            check(mesh.parse().mesh_launch->primitive_header == 0x80u);
            word(mesh.control, 0x64, 0x90); mesh.rejects(sn::ProgramError::program_bounds);
            word(mesh.control, 0x58, 0); mesh.rejects(sn::ProgramError::program_bounds);
            Fixture slotted(7); slotted.record.slot = 7; slotted.rejects(sn::ProgramError::stage_slot);
        }
        Fixture f; f.record.control = sn::Bytes(f.control).first(0xd7); f.rejects(sn::ProgramError::truncated_control);
        f.record.control = f.control; word(f.control, 8, 19); f.rejects(sn::ProgramError::control_family);
        // Version 17 reads the same fields; its stage identity differs.
        {
            Fixture v17; const auto v20_identity = v17.parse().stage_identity;
            word(v17.control, 8, 17); const auto older = v17.parse();
            check(older.instructions.size() == 32 && older.stage_identity != v20_identity);
        }
        word(f.control, 8, 20); word(f.control, 0, 0x34127698); f.rejects(sn::ProgramError::control_family);
        word(f.control, 0, 0x98761234); f.record.slot = 5; f.rejects(sn::ProgramError::stage_slot);
        f.record.slot = 0; word(f.control, 0x44, 0x100); f.rejects(sn::ProgramError::payload_size);
        word(f.control, 0x44, f.code.size()); word(f.control, 0x38, 0x70); f.rejects(sn::ProgramError::program_bounds);
        word(f.control, 0x38, 0x91); f.rejects(sn::ProgramError::instruction_alignment);
        word(f.control, 0x38, 0xa0); f.code[0xa0] = std::byte{1}; f.rejects(sn::ProgramError::nonzero_padding);
        f.code[0xa0] = std::byte{}; word(f.control, 0x40, ~0u); f.rejects(sn::ProgramError::program_bounds);
        word(f.control, 0x40, 0x100); f.record.kind = sn::ContainerKind::bahs; f.record.slot.reset();
        word(f.control, 0x30, 0); word(f.control, 0xd4, f.control.size()); check(f.parse().header.size() == 128);
        word(f.control, 0xd4, 0); f.rejects(sn::ProgramError::payload_size);
        // Driver-supplied runtime controls: exactly the 0xe0-byte reader span,
        // zero container extents, code bounded by +0x44.
        Fixture runtime(1);
        runtime.control.resize(sn::runtime_control_bytes); word(runtime.control, 0x30, 0); word(runtime.control, 0xd4, 0);
        runtime.record.control = runtime.control; runtime.record.kind = sn::ContainerKind::runtime; runtime.record.slot.reset();
        p = runtime.parse(); check(p.stage == sn::Stage::fragment && p.instructions.size() == 32 && p.control.size() == 0xe0);
        // Version 17 pack controls are the same reader span, with no container extent.
        {
            Fixture older(5);
            older.control.resize(sn::runtime_control_bytes);
            word(older.control, 8, 17); word(older.control, 0x30, 0); word(older.control, 0xd4, 0);
            older.record.control = older.control; older.record.kind = sn::ContainerKind::runtime; older.record.slot.reset();
            check(older.parse().stage == sn::Stage::compute && older.parse().control.size() == sn::runtime_control_bytes);
        }
        const auto runtime_identity = p.stage_identity;
        runtime.control[0xdc] = std::byte{1}; check(runtime.parse().stage_identity != runtime_identity);
        runtime.record.slot = 4; runtime.rejects(sn::ProgramError::stage_slot); runtime.record.slot.reset();
        word(runtime.control, 0x44, 0x100); runtime.rejects(sn::ProgramError::payload_size);
        word(runtime.control, 0x44, runtime.code.size());
        runtime.record.control = sn::Bytes(runtime.control).first(0xd8); runtime.rejects(sn::ProgramError::truncated_control);
        sn::Buffer longer(runtime.control); longer.resize(0x100); runtime.record.control = longer;
        runtime.rejects(sn::ProgramError::truncated_control);
        runtime.record.control = runtime.control; word(runtime.control, 8, 19); runtime.rejects(sn::ProgramError::control_family);
        Fixture file_zero; word(file_zero.control, 0x30, 0); file_zero.rejects(sn::ProgramError::payload_size);
        // A file record and its runtime re-presentation share one identity.
        Fixture file(1), driver(1);
        file.control[0xe8] = std::byte{0x5a};
        driver.control.resize(sn::runtime_control_bytes); word(driver.control, 0x30, 0);
        driver.record.control = driver.control; driver.record.kind = sn::ContainerKind::runtime; driver.record.slot.reset();
        const auto from_file = file.parse(), from_driver = driver.parse();
        check(from_file.stage_identity == from_driver.stage_identity && from_file.control_digest != from_driver.control_digest);
        driver.code[0x10] = std::byte{0x11}; check(driver.parse().stage_identity != from_file.stage_identity);
        std::cout << "Native program synthetic contracts passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
