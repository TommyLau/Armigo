// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Bounded A64 decoding and encoding for offline instruction preparation.
//
// This is not a general A64 decoder. It classifies exactly the encodings an
// offline preparer must locate or move: PC-relative branches, ADR/ADRP and
// literal loads, register branches, exception generation, and MRS/MSR system
// register accesses. Every other word decodes as Opcode::unclassified, which
// means "not examined", never "position independent" or "host safe".

#pragma once

#include <cstdint>
#include <optional>

namespace armigo::aarch64 {

// Bit n (0..30) stands for general-purpose register n (Xn or Wn). Field value
// 31 is the zero register or the stack pointer depending on the operand; it is
// never represented in a mask.
using RegisterMask = std::uint32_t;

inline constexpr std::uint8_t register_field_zero_or_sp = 31;
inline constexpr std::uint8_t platform_register = 18;
inline constexpr std::uint8_t link_register = 30;

constexpr RegisterMask register_bit(unsigned reg) noexcept {
    return reg < 31 ? RegisterMask{1} << reg : 0;
}

enum class Opcode : std::uint8_t {
    unclassified,

    // Immediate branches.
    b,              // 000101 imm26
    bl,             // 100101 imm26
    b_cond,         // 01010100 imm19 0 cond
    bc_cond,        // 01010100 imm19 1 cond (FEAT_HBC)
    cbz,            // sf 0110100 imm19 Rt
    cbnz,           // sf 0110101 imm19 Rt
    tbz,            // b5 0110110 b40 imm14 Rt
    tbnz,           // b5 0110111 b40 imm14 Rt

    // PC-relative address generation.
    adr,            // 0 immlo 10000 immhi Rd
    adrp,           // 1 immlo 10000 immhi Rd

    // Load register (literal): opc 011 V 00 imm19 Rt.
    ldr_literal_w,  // opc=00 V=0
    ldr_literal_x,  // opc=01 V=0
    ldrsw_literal,  // opc=10 V=0
    prfm_literal,   // opc=11 V=0; Rt is the prefetch operation
    ldr_literal_s,  // opc=00 V=1
    ldr_literal_d,  // opc=01 V=1
    ldr_literal_q,  // opc=10 V=1

    // Unconditional branch (register), base forms only.
    br,             // 1101011 0000 11111 000000 Rn 00000
    blr,            // 1101011 0001 11111 000000 Rn 00000
    ret,            // 1101011 0010 11111 000000 Rn 00000

    // Exception generation: 11010100 opc imm16 op2 LL.
    svc,            // opc=000 op2=000 LL=01
    hvc,            // opc=000 op2=000 LL=10
    smc,            // opc=000 op2=000 LL=11
    brk,            // opc=001 op2=000 LL=00
    hlt,            // opc=010 op2=000 LL=00

    // System register move: 1101010100 L 1 o0 op1 CRn CRm op2 Rt.
    msr_register,   // L=0
    mrs,            // L=1

    // System instruction with a single general-register operand:
    // 1101010100 0 0 1 op1 CRn CRm op2 Rt.
    cache_maintenance,  // DC/IC operations that name an address
};

// Cache-maintenance operations identified by their CRn:CRm:op2 encoding. These
// name a virtual address in Rt and have no effect on the host's own caches, so
// the runtime decides what each one means for the guest.
enum class CacheOperation : std::uint8_t {
    other,
    clean_invalidate_data,  // DC CIVAC: clean and invalidate the data cache line
    invalidate_instruction, // IC IVAU: invalidate the instruction cache line
};

// System instructions are identified by their 16-bit op1:CRn:CRm:op2 value.
constexpr std::uint16_t cache_operation_encoding(unsigned op1, unsigned crn, unsigned crm,
                                                 unsigned op2) noexcept {
    return static_cast<std::uint16_t>(((op1 & 7u) << 11) | ((crn & 0xfu) << 7) |
                                      ((crm & 0xfu) << 3) | (op2 & 7u));
}

CacheOperation identify_cache_operation(std::uint16_t encoding) noexcept;

enum class Condition : std::uint8_t {
    eq, ne, hs, lo, mi, pl, vs, vc, hi, ls, ge, lt, gt, le, al, nv,
};

// Inverts a condition. AL and NV are both "always" in A64 branches and have
// no inverse; callers must handle them before inverting.
constexpr Condition invert(Condition condition) noexcept {
    return static_cast<Condition>(static_cast<std::uint8_t>(condition) ^ 1u);
}

// System registers are identified by the 16-bit op0:op1:CRn:CRm:op2 value
// (op0 in bits 15..14, op1 in 13..11, CRn in 10..7, CRm in 6..3, op2 in 2..0).
constexpr std::uint16_t system_register_encoding(unsigned op0, unsigned op1, unsigned crn,
                                                 unsigned crm, unsigned op2) noexcept {
    return static_cast<std::uint16_t>(((op0 & 3u) << 14) | ((op1 & 7u) << 11) |
                                      ((crn & 15u) << 7) | ((crm & 15u) << 3) | (op2 & 7u));
}

enum class SystemRegister : std::uint8_t {
    other,
    nzcv,          // 3,3,4,2,0
    fpcr,          // 3,3,4,4,0
    fpsr,          // 3,3,4,4,1
    ctr_el0,       // 3,3,0,0,1
    dczid_el0,     // 3,3,0,0,7
    tpidr_el0,     // 3,3,13,0,2
    tpidrro_el0,   // 3,3,13,0,3
    cntfrq_el0,    // 3,3,14,0,0
    cntpct_el0,    // 3,3,14,0,1
    cntvct_el0,    // 3,3,14,0,2
    cntpctss_el0,  // 3,3,14,0,5 (FEAT_ECV)
    cntvctss_el0,  // 3,3,14,0,6 (FEAT_ECV)
};

SystemRegister identify_system_register(std::uint16_t encoding) noexcept;
// Returns nullopt for SystemRegister::other.
std::optional<std::uint16_t> encoding_of(SystemRegister reg) noexcept;

// Decoded fields. Only the fields documented for the decoded opcode are
// meaningful; all others stay zero.
struct Instruction {
    std::uint32_t word = 0;
    Opcode opcode = Opcode::unclassified;

    // Rt/Rd (bits 4..0) for branches, address generation, literal loads, MRS
    // and MSR; Rn (bits 9..5) for BR/BLR/RET. Unused for PRFM and exceptions.
    std::uint8_t reg = 0;
    // 64-bit general register operand: CBZ/CBNZ sf, TBZ/TBNZ b5, ADR/ADRP,
    // LDR (literal) X and LDRSW, MRS/MSR.
    bool wide = false;
    Condition condition = Condition::eq;  // B.cond and BC.cond
    std::uint8_t test_bit = 0;            // TBZ/TBNZ: b5:b40, 0..63
    std::uint8_t prefetch_operation = 0;  // PRFM (literal): bits 4..0
    std::uint16_t immediate = 0;          // exception generation imm16
    std::uint16_t system_register = 0;    // MRS/MSR, see system_register_encoding
    CacheOperation cache_operation = CacheOperation::other;  // cache_maintenance

    // Sign-extended byte displacement from the instruction address. For ADRP
    // this is the page displacement (imm21 << 12) applied to the 4 KiB page of
    // the instruction address.
    std::int64_t displacement = 0;

    // Architectural general-register accesses of the decoded instruction.
    // Exception generation lists none: a supervisor ABI's register usage is not
    // architectural and is not modeled here.
    RegisterMask reads = 0;
    RegisterMask writes = 0;

    bool classified() const noexcept { return opcode != Opcode::unclassified; }
};

Instruction decode(std::uint32_t word) noexcept;

// True for opcodes whose behavior depends on their own address.
bool is_pc_relative(Opcode opcode) noexcept;
// True for opcodes that write the link register with the address after them.
bool writes_link_address(Opcode opcode) noexcept;
// Bits that encode the displacement; relocation changes no other bit.
// Zero for opcodes that are not PC-relative.
std::uint32_t displacement_field_mask(Opcode opcode) noexcept;

// Smallest and largest displacements the opcode encodes, in bytes, and the
// required displacement granularity. ADRP limits apply to page differences.
struct DisplacementRange {
    std::int64_t minimum = 0;
    std::int64_t maximum = 0;
    std::uint32_t granularity = 1;
};
std::optional<DisplacementRange> displacement_range(Opcode opcode) noexcept;

// Address the instruction refers to when it is located at pc: the branch
// target, the generated address (ADRP: a page address), or the literal
// address. Returns nullopt when the opcode is not PC-relative, pc is not
// 4-byte aligned, or the computation leaves the 64-bit address space.
std::optional<std::uint64_t> referenced_address(const Instruction& instruction,
                                                std::uint64_t pc) noexcept;

// ---------------------------------------------------------------------------
// Checked encoders. They never truncate: an operand outside the encodable
// range is an error and no word is produced.

enum class EncodeError : std::uint8_t {
    none,
    misaligned_address,  // pc or target violates the opcode's alignment
    out_of_range,        // the displacement does not fit its field
    invalid_register,    // a register number above 31 or not allowed here
    invalid_operand,     // any other unencodable operand
};

struct Encoded {
    std::uint32_t word = 0;
    EncodeError error = EncodeError::none;
    bool ok() const noexcept { return error == EncodeError::none; }
};

Encoded encode_b(std::uint64_t pc, std::uint64_t target) noexcept;
Encoded encode_bl(std::uint64_t pc, std::uint64_t target) noexcept;
// consistent selects BC.cond (FEAT_HBC) instead of B.cond.
Encoded encode_b_cond(Condition condition, std::uint64_t pc, std::uint64_t target,
                      bool consistent = false) noexcept;
Encoded encode_cbz(bool nonzero, bool wide, unsigned rt, std::uint64_t pc,
                   std::uint64_t target) noexcept;
// Bits 0..31 use the W form (b5 = 0) and bits 32..63 the X form, as the
// encoding requires.
Encoded encode_tbz(bool nonzero, unsigned rt, unsigned bit, std::uint64_t pc,
                   std::uint64_t target) noexcept;
Encoded encode_adr(unsigned rd, std::uint64_t pc, std::uint64_t target) noexcept;
// Encodes the page of target relative to the page of pc; target's low 12 bits
// are ignored.
Encoded encode_adrp(unsigned rd, std::uint64_t pc, std::uint64_t target) noexcept;
// opcode must be one of the literal-load opcodes. For PRFM, rt is the
// prefetch operation.
Encoded encode_load_literal(Opcode opcode, unsigned rt, std::uint64_t pc,
                            std::uint64_t target) noexcept;
Encoded encode_br(unsigned rn) noexcept;
Encoded encode_blr(unsigned rn) noexcept;
Encoded encode_ret(unsigned rn = link_register) noexcept;
Encoded encode_svc(std::uint16_t immediate) noexcept;
Encoded encode_brk(std::uint16_t immediate) noexcept;
// The system register's op0 must be 2 or 3; MRS/MSR cannot encode others.
Encoded encode_mrs(unsigned rt, std::uint16_t system_register) noexcept;
Encoded encode_msr(std::uint16_t system_register, unsigned rt) noexcept;

// ADD (immediate), 64- or 32-bit. Register 31 is SP for both rd and rn.
// shift12 selects the LSL #12 form.
Encoded encode_add_immediate(bool wide, unsigned rd, unsigned rn, std::uint32_t imm12,
                             bool shift12 = false) noexcept;

// Loads with an unsigned scaled immediate offset (LDR/LDRSW/PRFM
// "unsigned offset" forms). The access width is taken from the literal-load
// opcode of the same register class: ldr_literal_w (4), ldr_literal_x (8),
// ldrsw_literal (4), prfm_literal (8), ldr_literal_s (4), ldr_literal_d (8),
// ldr_literal_q (16). byte_offset must be a multiple of the width below
// 4096 * width. Register 31 as rn is SP.
Encoded encode_load_unsigned_offset(Opcode literal_class, unsigned rt, unsigned rn,
                                    std::uint32_t byte_offset) noexcept;

} // namespace armigo::aarch64
