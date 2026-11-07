// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/instruction.h>
#include <limits>

namespace armigo::shader_native {
std::uint64_t Instruction::bits(unsigned first, unsigned count) const {
    if (!count || count > 64 || first >= 128 || count > 128 - first)
        throw InputError("Native instruction bit range is invalid", first);
    std::uint64_t value = 0;
    for (unsigned bit = 0; bit < count; ++bit)
        value |= std::uint64_t((words[(first + bit) / 32] >> ((first + bit) % 32)) & 1) << bit;
    return value;
}
Instruction Instruction::read(Bytes bytes) {
    if (bytes.size() != 16) throw InputError("Native instruction must contain exactly 16 bytes");
    Instruction instruction;
    for (unsigned i = 0; i < 4; ++i) instruction.words[i] = static_cast<std::uint32_t>(little_endian(bytes, i * 4, 4));
    return instruction;
}
DecodeResult decode_instruction(Instruction raw, std::uint64_t pc, std::uint64_t extent) {
    const auto opcode = static_cast<std::uint16_t>(raw.bits(0, 12));
    auto fail = [&](DecodeError code, const char* message) -> DecodeResult { return DecodeFailure{code, opcode, pc, message}; };
    if (extent % 16 || pc % 16 || pc >= extent || extent > std::uint64_t(std::numeric_limits<std::int64_t>::max()) - (1ull << 50))
        return fail(DecodeError::invalid_pc, "Native PC or instruction span is invalid");
    switch (opcode) {
        case 0x3a9: case 0x905: case 0xc08:
            return fail(DecodeError::unknown_opcode, "Native opcode has no established decode contract");
        case 0x202: case 0x802: case 0xa02: case 0xc02: case 0x882: case 0xc82:
        case 0x918: case 0x94d: case 0x947: break;
        default: return fail(DecodeError::unvalidated_opcode, "Native opcode is not admitted by this decoder");
    }
    if (raw.words[3] & 0xc00001ffu)
        return fail(DecodeError::unvalidated_mode, "Native reserved or extension bits are not admitted");
    const bool uniform = opcode == 0x882 || opcode == 0xc82;
    auto predicate = [&](unsigned first, PredicateFile file) {
        return Predicate{file, static_cast<std::uint8_t>(raw.bits(first, 3)), raw.bits(first + 3, 1) != 0};
    };
    DecodedInstruction result;
    result.raw = raw; result.pc = pc; result.opcode = opcode;
    result.execution = predicate(12, uniform ? PredicateFile::uniform : PredicateFile::general);
    result.scheduling = static_cast<std::uint32_t>(raw.bits(105, 21));
    const auto destination = static_cast<std::uint8_t>(raw.bits(16, 8));
    if (opcode == 0x918 || opcode == 0x94d || opcode == 0x947) {
        if (raw.bits(16, 16)) return fail(DecodeError::unvalidated_mode, "Native flow register mode is not admitted");
        if (opcode == 0x918) {
            if (raw.words[1] || raw.words[2]) return fail(DecodeError::unvalidated_mode, "Native NOP mode is not admitted");
            result.operation = Operation::nop;
        } else if (opcode == 0x94d) {
            if (raw.words[1] || (raw.words[2] & ~0x07800000u))
                return fail(DecodeError::unvalidated_mode, "Native EXIT mode is not admitted");
            result.operation = Operation::exit; result.condition = predicate(87, PredicateFile::general);
        } else {
            if (raw.bits(32, 2) || (raw.words[2] & ~0x0783ffffu))
                return fail(DecodeError::unvalidated_mode, "Native convergence branch mode is not admitted");
            const auto encoded = raw.bits(32, 50);
            const auto relative = encoded & (1ull << 49) ? static_cast<std::int64_t>(encoded) - (1ll << 50) : static_cast<std::int64_t>(encoded);
            const auto target = static_cast<std::int64_t>(pc) + 16 + relative;
            if (target < 0 || target % 16 || static_cast<std::uint64_t>(target) >= extent)
                return fail(DecodeError::branch_target, "Native branch target is not an instruction boundary inside the program");
            result.operation = Operation::branch; result.branch_target = target;
            result.condition = predicate(87, PredicateFile::general);
        }
        return result;
    }
    if (raw.bits(24, 8) || (uniform && destination > 63))
        return fail(DecodeError::unvalidated_mode, "Native move register mode is not admitted");
    result.destination = Register{uniform ? RegisterFile::uniform : RegisterFile::general, destination};
    result.operation = uniform ? Operation::uniform_move : Operation::move;
    const std::uint32_t expected = opcode == 0x882 ? 0 : opcode == 0xc82 ? 0x08000000 : opcode == 0xc02 ? 0x08000f00 : 0xf00;
    if (raw.words[2] != expected) return fail(DecodeError::unvalidated_mode, "Native move modifiers are not admitted");
    switch (opcode) {
        case 0x802: case 0x882: result.source = Immediate{raw.words[1]}; break;
        case 0x202:
            if (raw.words[1] > 255) return fail(DecodeError::unvalidated_mode, "Native general register source is invalid");
            result.source = Register{RegisterFile::general, static_cast<std::uint8_t>(raw.words[1])}; break;
        case 0xc02: case 0xc82:
            if (raw.words[1] > 63) return fail(DecodeError::unvalidated_mode, "Native uniform register source is invalid");
            result.source = Register{RegisterFile::uniform, static_cast<std::uint8_t>(raw.words[1])}; break;
        case 0xa02:
            if (raw.words[1] & ~0x07ffff00u) return fail(DecodeError::unvalidated_mode, "Native constant address mode or alignment is invalid");
            result.source = Constant{static_cast<std::uint8_t>(raw.bits(54, 5)), static_cast<std::uint16_t>(raw.bits(38, 16))}; break;
        default: break;
    }
    return result;
}
} // namespace armigo::shader_native
