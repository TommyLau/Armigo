// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// General-purpose register operands and effects of A64 instructions.
//
// decode_register_effects() describes, for a bounded set of encodings, every
// general register the instruction names: which 5-bit field holds it, whether
// field value 31 means the zero register or the stack pointer for that operand,
// the 32- or 64-bit view, whether it is read, written or both, and whether the
// access depends on a condition. Loads and stores also describe the address
// computation, base writeback and the memory access itself.
//
// Coverage is deliberately bounded to the named integer, branch, load/store
// and AdvSIMD-transfer classes.
// Every word outside it is EffectStatus::unknown, which carries no register
// claim at all. Only EffectStatus::decoded effects are complete, and only
// complete effects can prove that an instruction does not use a register.
//
// This header describes instructions; it does not rewrite them, choose scratch
// registers, or reason about liveness across instructions.

#pragma once

#include <armigo/aarch64/instruction.h>

#include <array>
#include <cstdint>
#include <optional>

namespace armigo::aarch64 {

enum class EffectStatus : std::uint8_t {
    // Not decoded by this library. Nothing is known about the instruction's
    // register use; it may be allocated, unallocated, or an extension.
    unknown,
    // Inside a decoded class but unallocated or reserved by the field rules
    // this library applies (for example a 32-bit form with sf = 0 and N = 1).
    // Executing it is UNDEFINED; it has no register effects to describe.
    reserved,
    // Allocated, but these operand values make the behavior CONSTRAINED
    // UNPREDICTABLE (see EffectNote). Operands are filled in for inspection,
    // but the effects are not a reliable description of execution.
    unpredictable,
    // Allocated with fully described general register effects.
    decoded,
};

// Canonical instruction names. Preferred disassembly aliases (MOV, CMP, TST,
// LSL, UBFX, MUL, CSET, ...) decode to the instruction they alias; aliasing
// never changes effects. Loads and stores are named by data form; the
// addressing mode distinguishes, for example, LDR, LDUR and LDTR.
enum class Operation : std::uint8_t {
    none,        // unknown or reserved
    classified,  // see RegisterEffects::opcode (instruction.h classifier)

    // Data processing (immediate).
    add_immediate, adds_immediate, sub_immediate, subs_immediate,
    and_immediate, orr_immediate, eor_immediate, ands_immediate,
    movn, movz, movk,
    sbfm, bfm, ubfm,
    extr,

    // Data processing (register).
    and_shifted, bic_shifted, orr_shifted, orn_shifted,
    eor_shifted, eon_shifted, ands_shifted, bics_shifted,
    add_shifted, adds_shifted, sub_shifted, subs_shifted,
    add_extended, adds_extended, sub_extended, subs_extended,
    adc, adcs, sbc, sbcs,
    ccmn_register, ccmp_register, ccmn_immediate, ccmp_immediate,
    csel, csinc, csinv, csneg,
    udiv, sdiv, lslv, lsrv, asrv, rorv,
    crc32b, crc32h, crc32w, crc32x, crc32cb, crc32ch, crc32cw, crc32cx,
    rbit, rev16, rev32, rev, clz, cls,
    madd, msub, smaddl, smsubl, smulh, umaddl, umsubl, umulh,
    // Scalar moves between a general register and a SIMD&FP register.
    fmov_general,
    // Scalar floating-point to integer conversion (FCVTNS/NU/AS/AU/PS/PU/MS/MU/ZS/ZU).
    fcvt_general,
    // AdvSIMD broadcast of a general register into every vector element (DUP).
    dup_general, ins_general, smov, umov,
    // General-register integer/fixed-point input to scalar floating-point.
    scvtf_general, ucvtf_general,

    // Hints and barriers without register operands.
    nop, clrex, dsb, dmb, isb,

    // Single-register loads and stores. The vector forms transfer a SIMD&FP
    // register (B, H, S, D or Q) through a general-register address.
    strb, ldrb, ldrsb, strh, ldrh, ldrsh, str, ldr, ldrsw, prfm,
    str_vector, ldr_vector,
    // AdvSIMD structure transfers, including single-lane and replicate loads.
    st1_multiple, st2_multiple, st3_multiple, st4_multiple,
    ld1_multiple, ld2_multiple, ld3_multiple, ld4_multiple,
    st1_single, st2_single, st3_single, st4_single,
    ld1_single, ld2_single, ld3_single, ld4_single,
    ld1r, ld2r, ld3r, ld4r,

    // Register pairs.
    stp, ldp, ldpsw, stnp, ldnp,
    stp_vector, ldp_vector, stnp_vector, ldnp_vector,

    // Exclusive and ordered accesses.
    ldxrb, ldxrh, ldxr, ldaxrb, ldaxrh, ldaxr,
    stxrb, stxrh, stxr, stlxrb, stlxrh, stlxr,
    ldxp, ldaxp, stxp, stlxp,
    ldarb, ldarh, ldar, stlrb, stlrh, stlr,

    // Data Processing - Scalar Floating-Point and Advanced SIMD (op0 = x111)
    // groups with no general-register operand. Vector operands carry the
    // arrangement or scalar view; nothing here reads or writes NZCV except
    // the scalar floating-point compare groups.
    //
    // Advanced SIMD vector and scalar integer, bitwise, shift, permute,
    // extract, copy, modified-immediate, indexed-element, table-lookup and
    // crypto operations. Accumulation is visible per operand (read_write
    // destinations), not in the operation.
    simd_data,
    // The scalar forms of the same groups (scalar three same, two-register
    // misc, pairwise, shift by immediate, indexed element, copy element).
    simd_scalar_data,
    // Scalar floating-point arithmetic and precision change: one-source
    // (FMOV register, FABS, FNEG, FSQRT, FCVT), two-source (FADD, FMUL,
    // FDIV, FMAX, FRECPS, ...), three-source (FMADD family) and FCVT.
    fp_data,
    // Scalar floating-point comparison (FCMP, FCMPE): writes NZCV.
    fp_compare,
    // Scalar floating-point conditional comparison (FCCMP, FCCMPE): reads
    // and writes NZCV.
    fp_conditional_compare,
    // Scalar floating-point conditional select (FCSEL): reads NZCV.
    fp_conditional_select,
    // Scalar floating-point immediate (FMOV immediate).
    fp_immediate,

    // Observed pointer authentication; hints retain their implicit operands.
    pacibsp, autibsp, autiasp, xpaclri, pacia, pacib, autib,
    autia1716, autib1716, xpaci, pacga,
};

enum class RegisterFile : std::uint8_t {
    general,  // X0..X30, with field value 31 as the zero register or SP
    vector,   // SIMD&FP V0..V31; never part of a general register mask
};

// Meaning of field value 31 for one general-register operand. It depends on
// the operand position of the specific opcode, never on the field alone.
enum class Register31 : std::uint8_t {
    not_applicable,  // vector operand
    zero_register,   // XZR/WZR: reads as zero, writes are discarded
    stack_pointer,   // SP/WSP
};

enum class Access : std::uint8_t {
    read = 1,
    write = 2,
    read_write = 3,
};

constexpr bool includes_read(Access access) noexcept {
    return (static_cast<unsigned>(access) & 1u) != 0;
}
constexpr bool includes_write(Access access) noexcept {
    return (static_cast<unsigned>(access) & 2u) != 0;
}

enum class OperandRole : std::uint8_t {
    destination,  // computed result (MOVK and BFM also read it: Access::read_write)
    source,       // data input of a computation, compare, test or system move
    accumulator,  // Ra of multiply-accumulate
    base,         // address base; Access::read_write when written back
    index,        // register offset of a load or store
    transfer,     // register loaded or stored
    status,       // store-exclusive result (0 success, 1 failure)
    branch_target,
    link,         // implicit X30 written by BL and BLR
};

// Bit position of a 5-bit register field. Fields are 0 (Rd/Rt), 5 (Rn),
// 10 (Ra/Rt2) and 16 (Rm/Rs); implicit operands have no field.
inline constexpr std::uint8_t implicit_operand = 0xFF;

struct RegisterOperand {
    std::uint8_t field = implicit_operand;
    std::uint8_t number = 0;  // field value 0..31, or the implicit register
    RegisterFile file = RegisterFile::general;
    Register31 at_31 = Register31::not_applicable;
    // Register view in bits. General registers: 32 (W) or 64 (X); a 32-bit
    // write zeroes bits 63..32, so it still replaces the whole register.
    // Vector registers: 8, 16, 32, 64 or 128.
    std::uint8_t bits = 64;
    Access access = Access::read;
    OperandRole role = OperandRole::source;
    // The access happens only for one outcome of a condition (conditional
    // select sources, conditional compare operands). Masks include it.
    bool conditional = false;

    bool is_general() const noexcept { return file == RegisterFile::general; }
    // A general register X0..X30 (not the zero register or SP).
    bool names_general_register(unsigned reg) const noexcept {
        return is_general() && number == reg && reg < 31;
    }
    bool names_stack_pointer() const noexcept {
        return is_general() && number == 31 && at_31 == Register31::stack_pointer;
    }
    bool names_zero_register() const noexcept {
        return is_general() && number == 31 && at_31 == Register31::zero_register;
    }
    // Word bits holding this operand's field; zero for implicit operands.
    std::uint32_t field_mask() const noexcept {
        return field == implicit_operand ? 0u : std::uint32_t{0x1F} << field;
    }
};

inline constexpr std::size_t max_register_operands = 6;

enum class MemoryOperation : std::uint8_t { none, load, store, prefetch };

enum class AddressingMode : std::uint8_t {
    none,
    literal,          // PC-relative; see instruction.h
    unsigned_offset,  // [Xn|SP, #imm], scaled unsigned imm12
    unscaled_offset,  // [Xn|SP, #imm], signed imm9 (LDUR, LDTR)
    signed_offset,    // [Xn|SP, #imm], scaled signed imm7 (pairs)
    pre_index,        // [Xn|SP, #imm]!
    post_index,       // [Xn|SP], #imm or Xm (structure transfers)
    register_offset,  // [Xn|SP, Rm{, extend {#amount}}]
    base_register,    // [Xn|SP] (exclusive and ordered accesses)
};

enum class IndexExtend : std::uint8_t { none, uxtw, lsl, sxtw, sxtx };

enum class MemoryAttribute : std::uint8_t {
    non_temporal = 1u << 0,  // LDNP/STNP hint
    unprivileged = 1u << 1,  // LDTR/STTR family
    sign_extend = 1u << 2,   // loaded value is sign-extended
    // Exclusive access: a load-exclusive marks the address in the exclusive
    // monitor and a store-exclusive succeeds only while it stays marked.
    // Instructions inserted between them can clear the monitor.
    exclusive = 1u << 3,
    acquire = 1u << 4,       // load-acquire ordering
    release = 1u << 5,       // store-release ordering
};
using MemoryAttributes = std::uint8_t;

constexpr bool has(MemoryAttributes attributes, MemoryAttribute attribute) noexcept {
    return (attributes & static_cast<MemoryAttributes>(attribute)) != 0;
}

struct MemoryEffect {
    MemoryOperation operation = MemoryOperation::none;
    AddressingMode addressing = AddressingMode::none;
    std::uint8_t element_bytes = 0;  // memory bytes per transferred register or lane
    std::uint8_t elements = 0;       // transferred registers/lanes (1..4); 0 for prefetch
    IndexExtend extend = IndexExtend::none;  // register offset or register post-index only
    std::uint8_t index_shift = 0;           // register offset or register post-index only
    // Byte offset added to the base: the address offset for unsigned,
    // unscaled, signed and pre-index forms, and the writeback amount for
    // immediate post-index forms (whose access uses the unmodified base).
    // Register post-index has offset zero and an explicit index operand.
    std::int64_t offset = 0;
    bool writeback = false;  // the base register is updated
    MemoryAttributes attributes = 0;
};

enum class ControlFlow : std::uint8_t {
    none,               // continues with the next instruction
    branch,             // B
    conditional_branch, // B.cond, BC.cond, CBZ, CBNZ, TBZ, TBNZ
    call,               // BL
    indirect_branch,    // BR
    indirect_call,      // BLR
    return_,            // RET
    exception,          // SVC, HVC, SMC, BRK, HLT
};

// Facts that qualify or limit the effects.
enum class EffectNote : std::uint16_t {
    // Load or store with base writeback where Rt (or Rt2) equals Rn and
    // Rn != 31: CONSTRAINED UNPREDICTABLE (status unpredictable).
    writeback_overlaps_transfer = 1u << 0,
    // Pair load with Rt == Rt2: CONSTRAINED UNPREDICTABLE.
    pair_load_same_register = 1u << 1,
    // A should-be-one field (SMULH/UMULH Ra, unused Rs and Rt2 fields of
    // exclusive and ordered accesses) is not all ones: CONSTRAINED
    // UNPREDICTABLE.
    should_be_one_field = 1u << 2,
    // Exception generation: registers observed or changed by the handler
    // (for SVC, the guest supervisor ABI) are not architectural and not
    // described. Effects are never complete for these instructions.
    handler_defined_registers = 1u << 3,
    // The base register is SP; an SP alignment check may apply.
    stack_pointer_base = 1u << 4,
    // Store-exclusive whose status register Rs equals Rt or Rt2, or equals
    // Rn with Rn != 31: CONSTRAINED UNPREDICTABLE.
    status_overlaps_operand = 1u << 5,
};
using EffectNotes = std::uint16_t;

constexpr bool has(EffectNotes notes, EffectNote note) noexcept {
    return (notes & static_cast<EffectNotes>(note)) != 0;
}

struct RegisterEffects {
    std::uint32_t word = 0;
    EffectStatus status = EffectStatus::unknown;
    Operation operation = Operation::none;
    // The instruction.h opcode for Operation::classified; otherwise
    // Opcode::unclassified.
    Opcode opcode = Opcode::unclassified;

    std::array<RegisterOperand, max_register_operands> operands{};
    std::uint8_t operand_count = 0;

    // General registers X0..X30 read or written, including conditional and
    // implicit accesses. Field value 31 is never in a mask; see operands.
    RegisterMask reads = 0;
    RegisterMask writes = 0;

    bool reads_flags = false;   // NZCV is an input
    bool writes_flags = false;  // NZCV is an output

    ControlFlow control_flow = ControlFlow::none;
    MemoryEffect memory;
    EffectNotes notes = 0;

    bool decoded() const noexcept { return status == EffectStatus::decoded; }
};

RegisterEffects decode_register_effects(std::uint32_t word) noexcept;

// True only for decoded effects whose notes do not leave registers undescribed
// and none of whose operands, explicit or implicit, is general register reg
// (0..30). Unknown, reserved and unpredictable words are never free.
bool proven_free_of(const RegisterEffects& effects, unsigned reg) noexcept;

// Word bits of every explicit operand field that names general register reg
// (0..30). Returns nullopt unless the effects are decoded and every use of
// reg is an explicit field, so the result is exactly the set of fields that
// would name another register if their values changed. An empty mask means
// the instruction does not use reg.
std::optional<std::uint32_t> register_fields(const RegisterEffects& effects,
                                             unsigned reg) noexcept;

bool reads_stack_pointer(const RegisterEffects& effects) noexcept;
bool writes_stack_pointer(const RegisterEffects& effects) noexcept;

} // namespace armigo::aarch64
