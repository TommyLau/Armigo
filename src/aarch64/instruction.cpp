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

constexpr std::uint32_t field(std::uint32_t word, unsigned low, unsigned bits) noexcept {
    return (word >> low) & ((1u << bits) - 1u);
}

constexpr std::int64_t sign_extend(std::uint32_t value, unsigned bits) noexcept {
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    return static_cast<std::int64_t>((std::uint64_t{value} ^ sign) - sign);
}

struct SystemRegisterName {
    SystemRegister reg;
    std::uint16_t encoding;
};

struct CacheName {
    CacheOperation operation;
    std::uint16_t encoding;
};

constexpr CacheName cache_operations[] = {
    {CacheOperation::clean_invalidate_data, cache_operation_encoding(3, 7, 14, 1)},
    {CacheOperation::invalidate_instruction, cache_operation_encoding(3, 7, 5, 1)},
};

constexpr SystemRegisterName system_registers[] = {
    {SystemRegister::nzcv, system_register_encoding(3, 3, 4, 2, 0)},
    {SystemRegister::fpcr, system_register_encoding(3, 3, 4, 4, 0)},
    {SystemRegister::fpsr, system_register_encoding(3, 3, 4, 4, 1)},
    {SystemRegister::ctr_el0, system_register_encoding(3, 3, 0, 0, 1)},
    {SystemRegister::dczid_el0, system_register_encoding(3, 3, 0, 0, 7)},
    {SystemRegister::tpidr_el0, system_register_encoding(3, 3, 13, 0, 2)},
    {SystemRegister::tpidrro_el0, system_register_encoding(3, 3, 13, 0, 3)},
    {SystemRegister::cntfrq_el0, system_register_encoding(3, 3, 14, 0, 0)},
    {SystemRegister::cntpct_el0, system_register_encoding(3, 3, 14, 0, 1)},
    {SystemRegister::cntvct_el0, system_register_encoding(3, 3, 14, 0, 2)},
    {SystemRegister::cntpctss_el0, system_register_encoding(3, 3, 14, 0, 5)},
    {SystemRegister::cntvctss_el0, system_register_encoding(3, 3, 14, 0, 6)},
};

Instruction decode_immediate_branch(std::uint32_t word, Opcode opcode) {
    Instruction result{.word = word, .opcode = opcode};
    result.displacement = sign_extend(field(word, 0, 26), 26) * 4;
    if (opcode == Opcode::bl) {
        result.writes = register_bit(link_register);
    }
    return result;
}

Instruction decode_conditional_branch(std::uint32_t word) {
    Instruction result{.word = word,
                       .opcode = field(word, 4, 1) ? Opcode::bc_cond : Opcode::b_cond};
    result.condition = static_cast<Condition>(field(word, 0, 4));
    result.displacement = sign_extend(field(word, 5, 19), 19) * 4;
    return result;
}

Instruction decode_compare_branch(std::uint32_t word) {
    Instruction result{.word = word, .opcode = field(word, 24, 1) ? Opcode::cbnz : Opcode::cbz};
    result.reg = static_cast<std::uint8_t>(field(word, 0, 5));
    result.wide = field(word, 31, 1) != 0;
    result.displacement = sign_extend(field(word, 5, 19), 19) * 4;
    result.reads = register_bit(result.reg);
    return result;
}

Instruction decode_test_branch(std::uint32_t word) {
    Instruction result{.word = word, .opcode = field(word, 24, 1) ? Opcode::tbnz : Opcode::tbz};
    result.reg = static_cast<std::uint8_t>(field(word, 0, 5));
    result.wide = field(word, 31, 1) != 0;
    result.test_bit = static_cast<std::uint8_t>((field(word, 31, 1) << 5) | field(word, 19, 5));
    result.displacement = sign_extend(field(word, 5, 14), 14) * 4;
    result.reads = register_bit(result.reg);
    return result;
}

Instruction decode_address_generation(std::uint32_t word) {
    const bool page = field(word, 31, 1) != 0;
    Instruction result{.word = word, .opcode = page ? Opcode::adrp : Opcode::adr};
    result.reg = static_cast<std::uint8_t>(field(word, 0, 5));
    result.wide = true;
    const std::uint32_t immediate = (field(word, 5, 19) << 2) | field(word, 29, 2);
    result.displacement = sign_extend(immediate, 21) * (page ? 4096 : 1);
    result.writes = register_bit(result.reg);
    return result;
}

Instruction decode_load_literal(std::uint32_t word) {
    const std::uint32_t opc = field(word, 30, 2);
    const bool vector = field(word, 26, 1) != 0;
    static constexpr Opcode general[] = {Opcode::ldr_literal_w, Opcode::ldr_literal_x,
                                         Opcode::ldrsw_literal, Opcode::prfm_literal};
    static constexpr Opcode simd[] = {Opcode::ldr_literal_s, Opcode::ldr_literal_d,
                                      Opcode::ldr_literal_q, Opcode::unclassified};
    const Opcode opcode = vector ? simd[opc] : general[opc];
    if (opcode == Opcode::unclassified) {
        return Instruction{.word = word};  // opc=11 V=1 is unallocated.
    }
    Instruction result{.word = word, .opcode = opcode};
    result.displacement = sign_extend(field(word, 5, 19), 19) * 4;
    const auto rt = static_cast<std::uint8_t>(field(word, 0, 5));
    if (opcode == Opcode::prfm_literal) {
        result.prefetch_operation = rt;
    } else {
        result.reg = rt;
    }
    if (!vector && opcode != Opcode::prfm_literal) {
        result.wide = opcode != Opcode::ldr_literal_w;
        result.writes = register_bit(rt);
    }
    return result;
}

Instruction decode_register_branch(std::uint32_t word) {
    static constexpr Opcode opcodes[] = {Opcode::br, Opcode::blr, Opcode::ret};
    const std::uint32_t opc = field(word, 21, 4);
    if (opc > 2) {
        return Instruction{.word = word};
    }
    Instruction result{.word = word, .opcode = opcodes[opc]};
    result.reg = static_cast<std::uint8_t>(field(word, 5, 5));
    result.wide = true;
    result.reads = register_bit(result.reg);
    if (result.opcode == Opcode::blr) {
        result.writes = register_bit(link_register);
    }
    return result;
}

Instruction decode_exception(std::uint32_t word) {
    const std::uint32_t opc = field(word, 21, 3);
    const std::uint32_t ll = field(word, 0, 2);
    Opcode opcode = Opcode::unclassified;
    if (opc == 0 && ll == 1) {
        opcode = Opcode::svc;
    } else if (opc == 0 && ll == 2) {
        opcode = Opcode::hvc;
    } else if (opc == 0 && ll == 3) {
        opcode = Opcode::smc;
    } else if (opc == 1 && ll == 0) {
        opcode = Opcode::brk;
    } else if (opc == 2 && ll == 0) {
        opcode = Opcode::hlt;
    }
    if (opcode == Opcode::unclassified) {
        return Instruction{.word = word};
    }
    Instruction result{.word = word, .opcode = opcode};
    result.immediate = static_cast<std::uint16_t>(field(word, 5, 16));
    return result;
}

Instruction decode_system_move(std::uint32_t word) {
    const bool read = field(word, 21, 1) != 0;
    Instruction result{.word = word, .opcode = read ? Opcode::mrs : Opcode::msr_register};
    result.reg = static_cast<std::uint8_t>(field(word, 0, 5));
    result.wide = true;
    result.system_register = static_cast<std::uint16_t>(0x8000u | field(word, 5, 15));
    if (read) {
        result.writes = register_bit(result.reg);
    } else {
        result.reads = register_bit(result.reg);
    }
    return result;
}

Instruction decode_system_instruction(std::uint32_t word) {
    // op1:CRn:CRm:op2 selects the operation; Rt names the address operand.
    const auto encoding = cache_operation_encoding(
        (word >> 16) & 7u, (word >> 12) & 0xfu, (word >> 8) & 0xfu, (word >> 5) & 7u);
    const auto operation = identify_cache_operation(encoding);
    if (operation == CacheOperation::other) return Instruction{.word = word};
    Instruction result{.word = word, .opcode = Opcode::cache_maintenance};
    result.reg = static_cast<std::uint8_t>(field(word, 0, 5));
    result.system_register = encoding;
    result.cache_operation = operation;
    result.reads = register_bit(result.reg);
    return result;
}

} // namespace

PointerAuthentication pointer_authentication(std::uint32_t word) noexcept {
    using P = PointerAuthentication;
    switch (word) {
    case 0xd65f0fff: return P::retab;
    case 0xd503237f: return P::pacibsp;
    case 0xd50323ff: return P::autibsp;
    case 0xd50323bf: return P::autiasp;
    case 0xd50320ff: return P::xpaclri;
    case 0xd503219f: return P::autia1716;
    case 0xd50321df: return P::autib1716;
    case 0xd503211f: case 0xd503215f: case 0xd503231f: case 0xd503233f:
    case 0xd503235f: case 0xd503239f: case 0xd50323df: return P::unsupported;
    default: break;
    }
    switch (word & 0xfffffc00u) {
    case 0xdac10000: return P::pacia;
    case 0xdac10400: return P::pacib;
    case 0xdac11400: return P::autib;
    default: break;
    }
    if ((word & 0xffffffe0u) == 0xdac143e0u) return P::xpaci;
    if ((word & 0xffe0fc00u) == 0x9ac03000u) return P::pacga;
    // Remaining one-source PAUTH, authenticated register branches/returns,
    // and authenticated loads retain an explicit unsupported boundary.
    if ((word & 0xffff0000u) == 0xdac10000u ||
        (word & 0xfe1ff800u) == 0xd61f0800u ||
        (word & 0xff200400u) == 0xf8200400u) return P::unsupported;
    return P::none;
}

CacheOperation identify_cache_operation(std::uint16_t encoding) noexcept {
    for (const auto& entry : cache_operations) {
        if (entry.encoding == encoding) return entry.operation;
    }
    return CacheOperation::other;
}

SystemRegister identify_system_register(std::uint16_t encoding) noexcept {
    for (const auto& entry : system_registers) {
        if (entry.encoding == encoding) {
            return entry.reg;
        }
    }
    return SystemRegister::other;
}

std::optional<std::uint16_t> encoding_of(SystemRegister reg) noexcept {
    for (const auto& entry : system_registers) {
        if (entry.reg == reg) {
            return entry.encoding;
        }
    }
    return std::nullopt;
}

Instruction decode(std::uint32_t word) noexcept {
    if (word == 0xd65f0fffu)
        return {.word = word, .opcode = Opcode::retab, .reg = link_register,
                .wide = true, .reads = register_bit(link_register)};

    if ((word & 0x7C000000u) == 0x14000000u) {
        return decode_immediate_branch(word, (word >> 31) ? Opcode::bl : Opcode::b);
    }
    if ((word & 0xFF000000u) == 0x54000000u) {
        return decode_conditional_branch(word);
    }
    if ((word & 0x7E000000u) == 0x34000000u) {
        return decode_compare_branch(word);
    }
    if ((word & 0x7E000000u) == 0x36000000u) {
        return decode_test_branch(word);
    }
    if ((word & 0x1F000000u) == 0x10000000u) {
        return decode_address_generation(word);
    }
    if ((word & 0x3B000000u) == 0x18000000u) {
        return decode_load_literal(word);
    }
    if ((word & 0xFE1FFC1Fu) == 0xD61F0000u) {
        return decode_register_branch(word);
    }
    if ((word & 0xFF00001Cu) == 0xD4000000u) {
        return decode_exception(word);
    }
    if ((word & 0xFFD00000u) == 0xD5100000u) {
        return decode_system_move(word);
    }
    // System instructions carry op1:CRn:CRm:op2 with L=0 and o0=1.
    if ((word & 0xFFF00000u) == 0xD5000000u) {
        return decode_system_instruction(word);
    }
    return Instruction{.word = word};
}

bool is_pc_relative(Opcode opcode) noexcept {
    return displacement_field_mask(opcode) != 0;
}

bool writes_link_address(Opcode opcode) noexcept {
    return opcode == Opcode::bl || opcode == Opcode::blr;
}

std::uint32_t displacement_field_mask(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::b:
    case Opcode::bl:
        return 0x03FFFFFFu;
    case Opcode::b_cond:
    case Opcode::bc_cond:
    case Opcode::cbz:
    case Opcode::cbnz:
    case Opcode::ldr_literal_w:
    case Opcode::ldr_literal_x:
    case Opcode::ldrsw_literal:
    case Opcode::prfm_literal:
    case Opcode::ldr_literal_s:
    case Opcode::ldr_literal_d:
    case Opcode::ldr_literal_q:
        return 0x00FFFFE0u;
    case Opcode::tbz:
    case Opcode::tbnz:
        return 0x0007FFE0u;
    case Opcode::adr:
    case Opcode::adrp:
        return 0x60FFFFE0u;
    default:
        return 0;
    }
}

std::optional<DisplacementRange> displacement_range(Opcode opcode) noexcept {
    switch (displacement_field_mask(opcode)) {
    case 0x03FFFFFFu:
        return DisplacementRange{-(std::int64_t{1} << 27), (std::int64_t{1} << 27) - 4, 4};
    case 0x00FFFFE0u:
        return DisplacementRange{-(std::int64_t{1} << 20), (std::int64_t{1} << 20) - 4, 4};
    case 0x0007FFE0u:
        return DisplacementRange{-(std::int64_t{1} << 15), (std::int64_t{1} << 15) - 4, 4};
    case 0x60FFFFE0u:
        if (opcode == Opcode::adrp) {
            return DisplacementRange{-(std::int64_t{1} << 32), (std::int64_t{1} << 32) - 4096,
                                     4096};
        }
        return DisplacementRange{-(std::int64_t{1} << 20), (std::int64_t{1} << 20) - 1, 1};
    default:
        return std::nullopt;
    }
}

std::optional<std::uint64_t> referenced_address(const Instruction& instruction,
                                                std::uint64_t pc) noexcept {
    if (!is_pc_relative(instruction.opcode) || pc % 4 != 0) {
        return std::nullopt;
    }
    const std::uint64_t base = instruction.opcode == Opcode::adrp ? (pc & ~std::uint64_t{0xFFF}) : pc;
    return detail::offset_address(base, instruction.displacement);
}

} // namespace armigo::aarch64
