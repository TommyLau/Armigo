// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/instruction.h>

#include "displacement.h"

namespace armigo::aarch64 {
namespace {

constexpr Encoded failure(EncodeError error) noexcept {
    return Encoded{.word = 0, .error = error};
}

constexpr Encoded success(std::uint32_t word) noexcept {
    return Encoded{.word = word, .error = EncodeError::none};
}

// Word displacement in [-(2^(bits-1)), 2^(bits-1)) from pc to target, already
// checked for alignment and range.
struct WordDisplacement {
    std::uint32_t field = 0;
    EncodeError error = EncodeError::none;
};

WordDisplacement word_displacement(std::uint64_t pc, std::uint64_t target, unsigned bits) noexcept {
    if (pc % 4 != 0 || target % 4 != 0) {
        return {0, EncodeError::misaligned_address};
    }
    const auto delta = detail::distance(pc, target);
    const std::int64_t limit = std::int64_t{1} << (bits + 1);  // bytes
    if (!delta || !detail::fits(*delta, -limit, limit - 4)) {
        return {0, EncodeError::out_of_range};
    }
    const auto words = static_cast<std::uint64_t>(*delta / 4);
    return {static_cast<std::uint32_t>(words & ((std::uint64_t{1} << bits) - 1)), EncodeError::none};
}

Encoded immediate_branch(std::uint32_t base, std::uint64_t pc, std::uint64_t target) noexcept {
    const auto displacement = word_displacement(pc, target, 26);
    if (displacement.error != EncodeError::none) {
        return failure(displacement.error);
    }
    return success(base | displacement.field);
}

Encoded register_branch(std::uint32_t base, unsigned rn) noexcept {
    if (rn > 31) {
        return failure(EncodeError::invalid_register);
    }
    return success(base | (rn << 5));
}

Encoded system_move(std::uint32_t base, unsigned rt, std::uint16_t system_register) noexcept {
    if (rt > 31) {
        return failure(EncodeError::invalid_register);
    }
    if ((system_register & 0x8000u) == 0) {
        return failure(EncodeError::invalid_operand);
    }
    return success(base | (std::uint32_t{system_register & 0x7FFFu} << 5) | rt);
}

// Load (literal) opc/V selection and the matching unsigned-offset encoding.
struct LoadClass {
    std::uint32_t literal_bits = 0;
    std::uint32_t unsigned_offset_base = 0;
    std::uint32_t width = 0;
};

std::optional<LoadClass> load_class(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::ldr_literal_w:
        return LoadClass{0x18000000u, 0xB9400000u, 4};
    case Opcode::ldr_literal_x:
        return LoadClass{0x58000000u, 0xF9400000u, 8};
    case Opcode::ldrsw_literal:
        return LoadClass{0x98000000u, 0xB9800000u, 4};
    case Opcode::prfm_literal:
        return LoadClass{0xD8000000u, 0xF9800000u, 8};
    case Opcode::ldr_literal_s:
        return LoadClass{0x1C000000u, 0xBD400000u, 4};
    case Opcode::ldr_literal_d:
        return LoadClass{0x5C000000u, 0xFD400000u, 8};
    case Opcode::ldr_literal_q:
        return LoadClass{0x9C000000u, 0x3DC00000u, 16};
    default:
        return std::nullopt;
    }
}

} // namespace

Encoded encode_b(std::uint64_t pc, std::uint64_t target) noexcept {
    return immediate_branch(0x14000000u, pc, target);
}

Encoded encode_bl(std::uint64_t pc, std::uint64_t target) noexcept {
    return immediate_branch(0x94000000u, pc, target);
}

Encoded encode_b_cond(Condition condition, std::uint64_t pc, std::uint64_t target,
                      bool consistent) noexcept {
    const auto cond = static_cast<std::uint32_t>(condition);
    if (cond > 15) {
        return failure(EncodeError::invalid_operand);
    }
    const auto displacement = word_displacement(pc, target, 19);
    if (displacement.error != EncodeError::none) {
        return failure(displacement.error);
    }
    return success(0x54000000u | (displacement.field << 5) | (consistent ? 0x10u : 0u) | cond);
}

Encoded encode_cbz(bool nonzero, bool wide, unsigned rt, std::uint64_t pc,
                   std::uint64_t target) noexcept {
    if (rt > 31) {
        return failure(EncodeError::invalid_register);
    }
    const auto displacement = word_displacement(pc, target, 19);
    if (displacement.error != EncodeError::none) {
        return failure(displacement.error);
    }
    return success((wide ? 0x80000000u : 0u) | 0x34000000u | (nonzero ? 0x01000000u : 0u) |
                   (displacement.field << 5) | rt);
}

Encoded encode_tbz(bool nonzero, unsigned rt, unsigned bit, std::uint64_t pc,
                   std::uint64_t target) noexcept {
    if (rt > 31) {
        return failure(EncodeError::invalid_register);
    }
    if (bit > 63) {
        return failure(EncodeError::invalid_operand);
    }
    const auto displacement = word_displacement(pc, target, 14);
    if (displacement.error != EncodeError::none) {
        return failure(displacement.error);
    }
    return success(((bit >> 5) << 31) | 0x36000000u | (nonzero ? 0x01000000u : 0u) |
                   ((bit & 31u) << 19) | (displacement.field << 5) | rt);
}

Encoded encode_adr(unsigned rd, std::uint64_t pc, std::uint64_t target) noexcept {
    if (rd > 31) {
        return failure(EncodeError::invalid_register);
    }
    if (pc % 4 != 0) {
        return failure(EncodeError::misaligned_address);
    }
    const auto delta = detail::distance(pc, target);
    if (!delta || !detail::fits(*delta, -(std::int64_t{1} << 20), (std::int64_t{1} << 20) - 1)) {
        return failure(EncodeError::out_of_range);
    }
    const auto immediate = static_cast<std::uint32_t>(static_cast<std::uint64_t>(*delta) & 0x1FFFFFu);
    return success(0x10000000u | ((immediate & 3u) << 29) | ((immediate >> 2) << 5) | rd);
}

Encoded encode_adrp(unsigned rd, std::uint64_t pc, std::uint64_t target) noexcept {
    if (rd > 31) {
        return failure(EncodeError::invalid_register);
    }
    if (pc % 4 != 0) {
        return failure(EncodeError::misaligned_address);
    }
    constexpr std::uint64_t page_mask = ~std::uint64_t{0xFFF};
    const auto delta = detail::distance(pc & page_mask, target & page_mask);
    if (!delta || !detail::fits(*delta, -(std::int64_t{1} << 32), (std::int64_t{1} << 32) - 4096)) {
        return failure(EncodeError::out_of_range);
    }
    const auto pages = static_cast<std::uint32_t>((static_cast<std::uint64_t>(*delta) >> 12) & 0x1FFFFFu);
    return success(0x90000000u | ((pages & 3u) << 29) | ((pages >> 2) << 5) | rd);
}

Encoded encode_load_literal(Opcode opcode, unsigned rt, std::uint64_t pc,
                            std::uint64_t target) noexcept {
    const auto selected = load_class(opcode);
    if (!selected) {
        return failure(EncodeError::invalid_operand);
    }
    if (rt > 31) {
        return failure(EncodeError::invalid_register);
    }
    const auto displacement = word_displacement(pc, target, 19);
    if (displacement.error != EncodeError::none) {
        return failure(displacement.error);
    }
    return success(selected->literal_bits | (displacement.field << 5) | rt);
}

Encoded encode_br(unsigned rn) noexcept {
    return register_branch(0xD61F0000u, rn);
}

Encoded encode_blr(unsigned rn) noexcept {
    return register_branch(0xD63F0000u, rn);
}

Encoded encode_ret(unsigned rn) noexcept {
    return register_branch(0xD65F0000u, rn);
}

Encoded encode_svc(std::uint16_t immediate) noexcept {
    return success(0xD4000001u | (std::uint32_t{immediate} << 5));
}

Encoded encode_brk(std::uint16_t immediate) noexcept {
    return success(0xD4200000u | (std::uint32_t{immediate} << 5));
}

Encoded encode_mrs(unsigned rt, std::uint16_t system_register) noexcept {
    return system_move(0xD5300000u, rt, system_register);
}

Encoded encode_msr(std::uint16_t system_register, unsigned rt) noexcept {
    return system_move(0xD5100000u, rt, system_register);
}

Encoded encode_add_immediate(bool wide, unsigned rd, unsigned rn, std::uint32_t imm12,
                             bool shift12) noexcept {
    if (rd > 31 || rn > 31) {
        return failure(EncodeError::invalid_register);
    }
    if (imm12 > 0xFFFu) {
        return failure(EncodeError::out_of_range);
    }
    return success((wide ? 0x91000000u : 0x11000000u) | (shift12 ? 0x00400000u : 0u) |
                   (imm12 << 10) | (rn << 5) | rd);
}

Encoded encode_load_unsigned_offset(Opcode literal_class, unsigned rt, unsigned rn,
                                    std::uint32_t byte_offset) noexcept {
    const auto selected = load_class(literal_class);
    if (!selected) {
        return failure(EncodeError::invalid_operand);
    }
    if (rt > 31 || rn > 31) {
        return failure(EncodeError::invalid_register);
    }
    if (byte_offset % selected->width != 0) {
        return failure(EncodeError::misaligned_address);
    }
    const std::uint32_t scaled = byte_offset / selected->width;
    if (scaled > 0xFFFu) {
        return failure(EncodeError::out_of_range);
    }
    return success(selected->unsigned_offset_base | (scaled << 10) | (rn << 5) | rt);
}

} // namespace armigo::aarch64
