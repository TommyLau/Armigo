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
        record.slot = stage == 0 ? 0 : stage == 1 ? 4 : 5;
    }
    sn::Program parse() { auto p = sn::parse_program(record); check(std::holds_alternative<sn::Program>(p)); return std::get<sn::Program>(p); }
    void rejects(sn::ProgramError error) { auto p = sn::parse_program(record); check(std::holds_alternative<sn::ProgramFailure>(p)); check(std::get<sn::ProgramFailure>(p).code == error); }
};
}
int main() {
    try {
        Fixture vertex; auto p = vertex.parse();
        check(p.header.size() == 128 && p.instructions.size() == 32 && p.padding.size() == 96 && p.constants.size() == 16);
        check(p.stage == sn::Stage::vertex && !p.compute_metadata && !p.launch);
        auto image = p.image_digest, identity = p.stage_identity;
        vertex.code[0x100] = std::byte{0x43}; p = vertex.parse();
        check(p.image_digest == image && p.stage_identity != identity); identity = p.stage_identity;
        vertex.control[0xf0] = std::byte{0x79}; p = vertex.parse();
        check(p.image_digest == image && p.stage_identity != identity);
        Fixture tail; word(tail.control, 0x3c, 4);
        check(tail.parse().trailing_padding.size() == 12 && tail.parse().constants.size() == 4);
        tail.code.back() = std::byte{1}; tail.rejects(sn::ProgramError::nonzero_padding);
        Fixture fragment(1); fragment.control[0x57] = std::byte{0xab};
        check(fragment.parse().stage == sn::Stage::fragment);
        Fixture compute(5); p = compute.parse(); check(p.header.empty() && p.instructions.size() == 0xa0);
        check(p.compute_metadata->stage_words == (std::array<std::uint32_t, 3>{8, 2, 1}) && !p.launch);
        for (auto stage : {2u, 3u, 4u, 6u, 7u, 255u}) { Fixture f(stage); f.rejects(sn::ProgramError::unsupported_stage); }
        Fixture f; f.record.control = sn::Bytes(f.control).first(0xd7); f.rejects(sn::ProgramError::truncated_control);
        f.record.control = f.control; word(f.control, 8, 19); f.rejects(sn::ProgramError::control_family);
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
        std::cout << "Native program synthetic contracts passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
