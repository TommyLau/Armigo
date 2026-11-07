// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <armigo/shader_native/frontend.h>
#include <armigo/shader_native/instruction.h>
#include <stdexcept>

namespace armigo::shader_native::test {
namespace sn = armigo::shader_native;
inline void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
inline void field(sn::Instruction& i, unsigned first, unsigned count, std::uint64_t v) {
    for (unsigned n = 0; n < count; ++n) {
        auto& w = i.words[(first + n) / 32]; auto mask = 1u << ((first + n) % 32);
        w = (w & ~mask) | ((v & (1ull << n)) ? mask : 0);
    }
}
inline sn::Instruction raw(unsigned opcode) { sn::Instruction i; field(i, 0, 12, opcode); field(i, 12, 4, 7); return i; }
inline sn::Instruction move(unsigned dest, std::uint32_t bits, unsigned guard = 7) {
    auto i = raw(0x802); field(i, 16, 8, dest); field(i, 12, 4, guard); i.words[1] = bits; i.words[2] = 0xf00; return i;
}
inline sn::Instruction exit_instruction() { auto i = raw(0x94d); field(i, 87, 4, 7); return i; }
inline void put(sn::Buffer& b, unsigned at, std::uint32_t value) { for (unsigned n = 0; n < 4; ++n) b.at(at + n) = std::byte(value >> (8 * n)); }
struct Fixture {
    sn::Buffer control = sn::Buffer(0xe0), code;
    sn::Stage stage;
    Fixture(std::vector<sn::Instruction> instructions, sn::Stage s = sn::Stage::fragment) : code(128 + instructions.size() * 16), stage(s) {
        put(control, 0, 0x98761234); put(control, 4, 0xc9); put(control, 8, 20); put(control, 0x30, control.size());
        put(control, 0x38, code.size()); put(control, 0x40, code.size()); put(control, 0x44, code.size()); control[0x56] = std::byte(unsigned(stage));
        put(code, 0, (4 << 5) | (stage == sn::Stage::vertex ? 1 | (1 << 10) : 2 | (5 << 10) | (1 << 14)));
        if (stage == sn::Stage::fragment) put(code, 0x48, 1);
        else code[0x35] = std::byte{0xf0};
        for (unsigned n = 0; n < instructions.size(); ++n) for (unsigned w = 0; w < 4; ++w) put(code, 128 + n * 16 + w * 4, instructions[n].words[w]);
    }
    sn::Program program() const {
        sn::Record r; r.control = control; r.code = code; r.kind = sn::ContainerKind::bnsh;
        auto p = sn::parse_program(r); check(std::holds_alternative<sn::Program>(p), "fixture metadata"); return std::get<sn::Program>(p);
    }
};
} // namespace armigo::shader_native::test
