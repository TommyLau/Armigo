// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/instruction.h>
#include <armigo/shader_native/opcode_inventory.h>
#include <iostream>

namespace sn = armigo::shader_native;
namespace {
void check(bool value) { if (!value) throw std::runtime_error("Synthetic instruction assertion failed"); }
void field(sn::Instruction& i, unsigned first, unsigned count, std::uint64_t value) {
    for (unsigned bit = 0; bit < count; ++bit) {
        auto& word = i.words[(first + bit) / 32]; const auto mask = 1u << ((first + bit) % 32);
        word = (word & ~mask) | ((value & (1ull << bit)) ? mask : 0);
    }
}
sn::Instruction instruction(unsigned opcode) {
    sn::Instruction i; field(i, 0, 12, opcode); field(i, 12, 4, 7); return i;
}
sn::DecodedInstruction decode(sn::Instruction i, std::uint64_t pc = 0, std::uint64_t extent = 64) {
    auto result = sn::decode_instruction(i, pc, extent); check(std::holds_alternative<sn::DecodedInstruction>(result));
    return std::get<sn::DecodedInstruction>(result);
}
void refuses(sn::Instruction i, sn::DecodeError code, std::uint64_t pc = 0, std::uint64_t extent = 64) {
    auto result = sn::decode_instruction(i, pc, extent); check(std::holds_alternative<sn::DecodeFailure>(result));
    check(std::get<sn::DecodeFailure>(result).code == code);
}
}
int main() {
    try {
        sn::Buffer bytes(16);
        for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = std::byte(i * 13 + 7);
        const auto raw = sn::Instruction::read(bytes);
        check(raw.words[0] == 0x2e211407u);
        // Cross every word boundary, including 64-bit ranges, against byte-wise extraction.
        for (unsigned first = 0; first < 128; ++first) for (unsigned count = 1; count <= 64 && first + count <= 128; ++count) {
            std::uint64_t expected = 0;
            for (unsigned bit = 0; bit < count; ++bit)
                expected |= std::uint64_t((std::to_integer<unsigned>(bytes[(first + bit) / 8]) >> ((first + bit) % 8)) & 1) << bit;
            check(raw.bits(first, count) == expected);
        }
        for (auto [first, count] : {std::pair{0u, 0u}, {0u, 65u}, {127u, 2u}, {128u, 1u}}) {
            bool threw = false; try { raw.bits(first, count); } catch (const sn::InputError&) { threw = true; } check(threw);
        }
        bool short_read = false; try { sn::Instruction::read(sn::Bytes(bytes).first(15)); } catch (const sn::InputError&) { short_read = true; } check(short_read);
        auto mov = instruction(0x802); field(mov, 16, 8, 237); mov.words[1] = 0xa5b4c3d2; mov.words[2] = 0xf00;
        field(mov, 105, 21, 0x155555); field(mov, 12, 4, 9);
        auto decoded = decode(mov); check(decoded.destination->index == 237);
        check(std::get<sn::Immediate>(*decoded.source).bits == 0xa5b4c3d2 && decoded.scheduling == 0x155555);
        check(decoded.execution.index == 1 && decoded.execution.inverted && decoded.execution.file == sn::PredicateFile::general);
        field(mov, 0, 12, 0x202); mov.words[1] = 255;
        check(std::get<sn::Register>(*decode(mov).source).is_zero());
        mov.words[1] = 256; refuses(mov, sn::DecodeError::unvalidated_mode);
        field(mov, 0, 12, 0xc02); mov.words[1] = 63; mov.words[2] = 0x08000f00;
        check(std::get<sn::Register>(*decode(mov).source).file == sn::RegisterFile::uniform);
        mov.words[1] = 64; refuses(mov, sn::DecodeError::unvalidated_mode);
        mov = instruction(0xa02); mov.words[2] = 0xf00; field(mov, 38, 16, 0x1234); field(mov, 54, 5, 24);
        const auto constant = std::get<sn::Constant>(*decode(mov).source); check(constant.bank == 24 && constant.byte_offset == 0x1234);
        field(mov, 38, 16, 3); refuses(mov, sn::DecodeError::unvalidated_mode);
        auto uniform = instruction(0x882); field(uniform, 16, 8, 59); field(uniform, 12, 4, 10); uniform.words[1] = 0x12345678;
        decoded = decode(uniform); check(decoded.destination->file == sn::RegisterFile::uniform && decoded.execution.file == sn::PredicateFile::uniform);
        check(decoded.execution.index == 2 && decoded.execution.inverted);
        field(uniform, 16, 8, 64); refuses(uniform, sn::DecodeError::unvalidated_mode);
        field(uniform, 16, 8, 63); check(decode(uniform).destination->is_zero());
        field(uniform, 0, 12, 0xc82); uniform.words[1] = 61; uniform.words[2] = 0x08000000;
        check(std::get<sn::Register>(*decode(uniform).source).index == 61);
        auto nop = instruction(0x918); check(decode(nop).operation == sn::Operation::nop);
        field(nop, 96, 1, 1); refuses(nop, sn::DecodeError::unvalidated_mode);
        auto exit = instruction(0x94d); field(exit, 87, 4, 13);
        check(decode(exit).condition->index == 5 && decode(exit).condition->inverted);
        field(exit, 84, 1, 1); refuses(exit, sn::DecodeError::unvalidated_mode);
        auto branch = instruction(0x947); field(branch, 87, 4, 7);
        field(branch, 32, 50, 16); check(decode(branch).branch_target == 32);
        field(branch, 32, 50, (1ull << 50) - 32); check(decode(branch, 16).branch_target == 0);
        field(branch, 32, 50, (1ull << 50) - 16); check(decode(branch).branch_target == 0);
        field(branch, 32, 50, (1ull << 50) - 32); refuses(branch, sn::DecodeError::branch_target);
        field(branch, 32, 50, 48); refuses(branch, sn::DecodeError::branch_target);
        field(branch, 32, 50, 4); refuses(branch, sn::DecodeError::branch_target);
        for (unsigned mode = 1; mode < 4; ++mode) { field(branch, 32, 50, mode); refuses(branch, sn::DecodeError::unvalidated_mode); }
        for (auto op : {0x3a9u, 0x905u, 0xc08u}) { auto unknown = instruction(op); field(unknown, 12, 4, 15); refuses(unknown, sn::DecodeError::unknown_opcode); }
        refuses(instruction(0x220), sn::DecodeError::unvalidated_opcode);
        refuses(instruction(0xfff), sn::DecodeError::unvalidated_opcode);
        refuses(instruction(0x918), sn::DecodeError::invalid_pc, 1);
        refuses(instruction(0x918), sn::DecodeError::invalid_pc, 64);
        refuses(instruction(0x918), sn::DecodeError::invalid_pc, 0, 63);
        sn::Buffer control(0xe0), code(48);
        auto put = [](sn::Buffer& b, std::size_t at, std::uint32_t value) {
            for (unsigned i = 0; i < 4; ++i) b.at(at + i) = std::byte(value >> (8 * i));
        };
        put(control, 0, 0x98761234); put(control, 4, 0xc9); put(control, 8, 20);
        put(control, 0x30, control.size()); put(control, 0x38, code.size());
        put(control, 0x40, code.size()); put(control, 0x44, code.size()); control[0x56] = std::byte{5};
        put(control, 0x58, 32); put(control, 0x5c, 1); put(control, 0x60, 1);
        std::array<sn::Instruction, 3> stream{instruction(0x94d), instruction(0x3a9), instruction(0x918)};
        field(stream[0], 87, 4, 7);
        for (unsigned n = 0; n < stream.size(); ++n) for (unsigned w = 0; w < 4; ++w) put(code, n * 16 + w * 4, stream[n].words[w]);
        sn::Record record; record.kind = sn::ContainerKind::bnsh; record.control = control; record.code = code;
        auto parsed = sn::parse_program(record); check(std::holds_alternative<sn::Program>(parsed));
        sn::OpcodeInventory inventory;
        const auto& audit = inventory.inspect(std::get<sn::Program>(parsed));
        check(audit.slots == 3 && audit.decoded == 2 && audit.refused == 1 && audit.first_failure->pc == 16);
        check(inventory.counts()[0x918].decoded == 1 && inventory.counts()[0x3a9].refused == 1);
        control[0xd0] = std::byte{1}; auto variant = sn::parse_program(record);
        inventory.inspect(std::get<sn::Program>(variant));
        check(inventory.programs() == 1 && inventory.counts()[0x918].slots == 1);
        check(inventory.failures()[unsigned(sn::DecodeError::unknown_opcode)] == 1);
        std::cout << "Native instruction synthetic contracts passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
