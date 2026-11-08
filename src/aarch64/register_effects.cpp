// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Field layouts follow the A64 encoding index of the Arm Architecture
// Reference Manual. Forms this file does not positively recognize stay
// EffectStatus::unknown; "reserved" is used only for field combinations the
// encoding tables mark unallocated or reserved.

#include <armigo/aarch64/register_effects.h>

namespace armigo::aarch64 {
namespace {

constexpr std::uint32_t field(std::uint32_t word, unsigned low, unsigned bits) noexcept {
    return (word >> low) & ((1u << bits) - 1u);
}

constexpr bool bit(std::uint32_t word, unsigned position) noexcept {
    return ((word >> position) & 1u) != 0;
}

constexpr std::int64_t sign_extend(std::uint32_t value, unsigned bits) noexcept {
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    return static_cast<std::int64_t>((std::uint64_t{value} ^ sign) - sign);
}

constexpr std::uint8_t rd_field = 0;
constexpr std::uint8_t rn_field = 5;
constexpr std::uint8_t ra_field = 10;
constexpr std::uint8_t rm_field = 16;

class Builder {
public:
    explicit Builder(std::uint32_t word) noexcept { effects_.word = word; }

    Builder& operation(Operation operation) noexcept {
        effects_.operation = operation;
        return *this;
    }

    // An explicit general-register operand read from the word's field.
    Builder& general(std::uint8_t field_lsb, Register31 at_31, unsigned bits, Access access,
                     OperandRole role, bool conditional = false) noexcept {
        add(RegisterOperand{.field = field_lsb,
                            .number = static_cast<std::uint8_t>(field(effects_.word, field_lsb, 5)),
                            .file = RegisterFile::general,
                            .at_31 = at_31,
                            .bits = static_cast<std::uint8_t>(bits),
                            .access = access,
                            .role = role,
                            .conditional = conditional});
        return *this;
    }

    Builder& vector(std::uint8_t field_lsb, unsigned bits, Access access,
                    OperandRole role) noexcept {
        add(RegisterOperand{.field = field_lsb,
                            .number = static_cast<std::uint8_t>(field(effects_.word, field_lsb, 5)),
                            .file = RegisterFile::vector,
                            .at_31 = Register31::not_applicable,
                            .bits = static_cast<std::uint8_t>(bits),
                            .access = access,
                            .role = role});
        return *this;
    }

    Builder& implicit_general(unsigned number, Register31 at_31, Access access,
                              OperandRole role, unsigned bits = 64) noexcept {
        add(RegisterOperand{.field = implicit_operand, .number = static_cast<std::uint8_t>(number),
            .file = RegisterFile::general, .at_31 = at_31, .bits = static_cast<std::uint8_t>(bits),
            .access = access, .role = role});
        return *this;
    }

    Builder& implicit_link() noexcept {
        add(RegisterOperand{.field = implicit_operand,
                            .number = link_register,
                            .file = RegisterFile::general,
                            .at_31 = Register31::zero_register,
                            .bits = 64,
                            .access = Access::write,
                            .role = OperandRole::link});
        return *this;
    }

    Builder& flags(bool read, bool write) noexcept {
        effects_.reads_flags = effects_.reads_flags || read;
        effects_.writes_flags = effects_.writes_flags || write;
        return *this;
    }

    Builder& control(ControlFlow control_flow) noexcept {
        effects_.control_flow = control_flow;
        return *this;
    }

    Builder& memory(const MemoryEffect& memory) noexcept {
        effects_.memory = memory;
        return *this;
    }

    Builder& note(EffectNote note) noexcept {
        effects_.notes = static_cast<EffectNotes>(effects_.notes | static_cast<EffectNotes>(note));
        return *this;
    }

    RegisterEffects decoded() noexcept { return finish(EffectStatus::decoded); }
    RegisterEffects unpredictable() noexcept { return finish(EffectStatus::unpredictable); }
    RegisterEffects finish(EffectStatus status) noexcept {
        effects_.status = status;
        for (std::size_t index = 0; index < effects_.operand_count; ++index) {
            const auto& operand = effects_.operands[index];
            if (!operand.is_general()) {
                continue;
            }
            if (includes_read(operand.access)) {
                effects_.reads |= register_bit(operand.number);
            }
            if (includes_write(operand.access)) {
                effects_.writes |= register_bit(operand.number);
            }
        }
        return effects_;
    }

    RegisterEffects& raw() noexcept { return effects_; }

private:
    void add(const RegisterOperand& operand) noexcept {
        // Every encoding below lists at most max_register_operands operands.
        effects_.operands[effects_.operand_count++] = operand;
    }

    RegisterEffects effects_;
};


// SIMD and scalar floating-point handlers, defined after the classified
// groups; declared here for the top-level dispatcher.
RegisterEffects simd_vector_data(std::uint32_t word) noexcept;
RegisterEffects simd_vector_indexed(std::uint32_t word) noexcept;
RegisterEffects simd_scalar_shift(std::uint32_t word) noexcept;
RegisterEffects fp_three_source(std::uint32_t word) noexcept;
RegisterEffects scalar_p1(std::uint32_t word) noexcept;
RegisterEffects fp_scalar_data(std::uint32_t word) noexcept;

RegisterEffects unknown(std::uint32_t word) noexcept {
    return RegisterEffects{.word = word};
}

RegisterEffects reserved(std::uint32_t word) noexcept {
    return RegisterEffects{.word = word, .status = EffectStatus::reserved};
}

constexpr unsigned width(std::uint32_t word) noexcept {
    return bit(word, 31) ? 64u : 32u;
}

constexpr auto zr = Register31::zero_register;
constexpr auto sp = Register31::stack_pointer;

// ---------------------------------------------------------------------------
// Classified instructions (instruction.h).

RegisterEffects classified(const Instruction& instruction) noexcept {
    const std::uint32_t word = instruction.word;
    Builder builder(word);
    builder.operation(Operation::classified);
    builder.raw().opcode = instruction.opcode;
    const unsigned bits = instruction.wide ? 64u : 32u;

    switch (instruction.opcode) {
    case Opcode::b:
        builder.control(ControlFlow::branch);
        break;
    case Opcode::bl:
        builder.implicit_link().control(ControlFlow::call);
        break;
    case Opcode::b_cond:
    case Opcode::bc_cond:
        builder.flags(true, false).control(ControlFlow::conditional_branch);
        break;
    case Opcode::cbz:
    case Opcode::cbnz:
    case Opcode::tbz:
    case Opcode::tbnz:
        builder.general(rd_field, zr, bits, Access::read, OperandRole::source)
            .control(ControlFlow::conditional_branch);
        break;
    case Opcode::adr:
    case Opcode::adrp:
        builder.general(rd_field, zr, 64, Access::write, OperandRole::destination);
        break;
    case Opcode::ldr_literal_w:
    case Opcode::ldr_literal_x:
    case Opcode::ldrsw_literal: {
        const bool signed_word = instruction.opcode == Opcode::ldrsw_literal;
        builder.general(rd_field, zr, bits, Access::write, OperandRole::transfer)
            .memory(MemoryEffect{
                .operation = MemoryOperation::load,
                .addressing = AddressingMode::literal,
                .element_bytes = static_cast<std::uint8_t>(
                    instruction.opcode == Opcode::ldr_literal_x ? 8 : 4),
                .elements = 1,
                .offset = instruction.displacement,
                .attributes = signed_word
                                  ? static_cast<MemoryAttributes>(MemoryAttribute::sign_extend)
                                  : MemoryAttributes{0}});
        break;
    }
    case Opcode::prfm_literal:
        builder.memory(MemoryEffect{.operation = MemoryOperation::prefetch,
                                    .addressing = AddressingMode::literal,
                                    .offset = instruction.displacement});
        break;
    case Opcode::ldr_literal_s:
    case Opcode::ldr_literal_d:
    case Opcode::ldr_literal_q: {
        const unsigned bytes = instruction.opcode == Opcode::ldr_literal_s   ? 4u
                               : instruction.opcode == Opcode::ldr_literal_d ? 8u
                                                                             : 16u;
        builder.vector(rd_field, bytes * 8, Access::write, OperandRole::transfer)
            .memory(MemoryEffect{.operation = MemoryOperation::load,
                                 .addressing = AddressingMode::literal,
                                 .element_bytes = static_cast<std::uint8_t>(bytes),
                                 .elements = 1,
                                 .offset = instruction.displacement});
        break;
    }
    case Opcode::br:
        builder.general(rn_field, zr, 64, Access::read, OperandRole::branch_target)
            .control(ControlFlow::indirect_branch);
        break;
    case Opcode::blr:
        builder.general(rn_field, zr, 64, Access::read, OperandRole::branch_target)
            .implicit_link()
            .control(ControlFlow::indirect_call);
        break;
    case Opcode::retab:
        builder.implicit_general(30, zr, Access::read, OperandRole::branch_target)
            .implicit_general(31, Register31::stack_pointer, Access::read, OperandRole::source)
            .control(ControlFlow::return_);
        break;
    case Opcode::ret:
        builder.general(rn_field, zr, 64, Access::read, OperandRole::branch_target)
            .control(ControlFlow::return_);
        break;
    case Opcode::svc:
    case Opcode::hvc:
    case Opcode::smc:
    case Opcode::brk:
    case Opcode::hlt:
        builder.control(ControlFlow::exception).note(EffectNote::handler_defined_registers);
        break;
    case Opcode::mrs:
        builder.general(rd_field, zr, 64, Access::write, OperandRole::destination);
        builder.flags(identify_system_register(instruction.system_register) ==
                          SystemRegister::nzcv,
                      false);
        break;
    case Opcode::msr_register:
        builder.general(rd_field, zr, 64, Access::read, OperandRole::source);
        builder.flags(false, identify_system_register(instruction.system_register) ==
                                 SystemRegister::nzcv);
        break;
    case Opcode::cache_maintenance:
        // The operation names an address in Rt and leaves it unchanged.
        builder.general(rd_field, zr, 64, Access::read, OperandRole::source);
        break;
    case Opcode::unclassified:
        return unknown(word);
    }
    return builder.decoded();
}

// ---------------------------------------------------------------------------
// Pointer-authentication effects, followed by operand-free hints and barriers.

RegisterEffects authentication_effects(std::uint32_t word, PointerAuthentication kind) noexcept {
    using P = PointerAuthentication;
    constexpr auto zr = Register31::zero_register, sp = Register31::stack_pointer;
    Builder builder(word);
    switch (kind) {
    case P::pacibsp: case P::autibsp: case P::autiasp: case P::xpaclri:
        builder.operation(kind == P::pacibsp ? Operation::pacibsp : kind == P::autibsp ? Operation::autibsp :
            kind == P::autiasp ? Operation::autiasp : Operation::xpaclri)
            .implicit_general(30, zr, Access::read_write, OperandRole::destination);
        if (kind != P::xpaclri) builder.implicit_general(31, sp, Access::read, OperandRole::source);
        return builder.decoded();
    case P::autia1716: case P::autib1716:
        return builder.operation(kind == P::autia1716 ? Operation::autia1716 : Operation::autib1716)
            .implicit_general(17, zr, Access::read_write, OperandRole::destination)
            .implicit_general(16, zr, Access::read, OperandRole::source).decoded();
    case P::pacia: case P::pacib: case P::autib:
        return builder.operation(kind == P::pacia ? Operation::pacia : kind == P::pacib ? Operation::pacib : Operation::autib)
            .general(rd_field, zr, 64, Access::read_write, OperandRole::destination)
            .general(rn_field, sp, 64, Access::read, OperandRole::source).decoded();
    case P::xpaci:
        return builder.operation(Operation::xpaci)
            .general(rd_field, zr, 64, Access::read_write, OperandRole::destination).decoded();
    case P::pacga:
        return builder.operation(Operation::pacga)
            .general(rd_field, zr, 64, Access::write, OperandRole::destination)
            .general(rn_field, zr, 64, Access::read, OperandRole::source)
            .general(rm_field, sp, 64, Access::read, OperandRole::source).decoded();
    default: return unknown(word);
    }
}

RegisterEffects hint_or_barrier(std::uint32_t word) noexcept {
    if (word == 0xD503201Fu) {
        return Builder(word).operation(Operation::nop).decoded();
    }
    // 1101010100 0 00 011 0011 CRm op2 11111, CRm is an option or immediate.
    if ((word & 0xFFFFF01Fu) == 0xD503301Fu) {
        switch (field(word, 5, 3)) {
        case 2:
            return Builder(word).operation(Operation::clrex).decoded();
        case 4:
            return Builder(word).operation(Operation::dsb).decoded();
        case 5:
            return Builder(word).operation(Operation::dmb).decoded();
        case 6:
            return Builder(word).operation(Operation::isb).decoded();
        default:
            break;
        }
    }
    return unknown(word);
}

// ---------------------------------------------------------------------------
// Data processing (immediate): bits 28..26 = 100.

// DecodeBitMasks validity for logical immediates: the element size must be
// at least two bits and the run of ones must not fill the element.
bool valid_logical_immediate(bool wide, std::uint32_t n, std::uint32_t imms) noexcept {
    if (!wide && n != 0) {
        return false;
    }
    const std::uint32_t combined = (n << 6) | (~imms & 0x3Fu);
    if (combined < 2) {
        return false;  // element length below two bits
    }
    unsigned length = 6;
    while (((combined >> length) & 1u) == 0) {
        --length;
    }
    const std::uint32_t levels = (1u << length) - 1u;
    return (imms & levels) != levels;
}

RegisterEffects add_sub_immediate(std::uint32_t word) noexcept {
    static constexpr Operation operations[] = {Operation::add_immediate, Operation::adds_immediate,
                                               Operation::sub_immediate, Operation::subs_immediate};
    const bool set_flags = bit(word, 29);
    const unsigned bits = width(word);
    return Builder(word)
        .operation(operations[field(word, 29, 2)])
        .general(rd_field, set_flags ? zr : sp, bits, Access::write, OperandRole::destination)
        .general(rn_field, sp, bits, Access::read, OperandRole::source)
        .flags(false, set_flags)
        .decoded();
}

RegisterEffects logical_immediate(std::uint32_t word) noexcept {
    const bool wide = bit(word, 31);
    if (!valid_logical_immediate(wide, field(word, 22, 1), field(word, 10, 6))) {
        return reserved(word);
    }
    static constexpr Operation operations[] = {Operation::and_immediate, Operation::orr_immediate,
                                               Operation::eor_immediate, Operation::ands_immediate};
    const std::uint32_t opc = field(word, 29, 2);
    const bool set_flags = opc == 3;
    const unsigned bits = width(word);
    return Builder(word)
        .operation(operations[opc])
        .general(rd_field, set_flags ? zr : sp, bits, Access::write, OperandRole::destination)
        .general(rn_field, zr, bits, Access::read, OperandRole::source)
        .flags(false, set_flags)
        .decoded();
}

RegisterEffects move_wide(std::uint32_t word) noexcept {
    const std::uint32_t opc = field(word, 29, 2);
    if (opc == 1 || (!bit(word, 31) && bit(word, 22))) {
        return reserved(word);
    }
    const Operation operation = opc == 0 ? Operation::movn
                                : opc == 2 ? Operation::movz
                                           : Operation::movk;
    // MOVK keeps the other bits of Rd, so it reads Rd.
    const Access access = operation == Operation::movk ? Access::read_write : Access::write;
    return Builder(word)
        .operation(operation)
        .general(rd_field, zr, width(word), access, OperandRole::destination)
        .decoded();
}

RegisterEffects bitfield(std::uint32_t word) noexcept {
    const std::uint32_t opc = field(word, 29, 2);
    const bool wide = bit(word, 31);
    if (opc == 3 || wide != bit(word, 22) ||
        (!wide && (bit(word, 21) || bit(word, 15)))) {
        return reserved(word);
    }
    static constexpr Operation operations[] = {Operation::sbfm, Operation::bfm, Operation::ubfm};
    // BFM inserts into Rd and keeps its other bits.
    const Access access = opc == 1 ? Access::read_write : Access::write;
    const unsigned bits = width(word);
    return Builder(word)
        .operation(operations[opc])
        .general(rd_field, zr, bits, access, OperandRole::destination)
        .general(rn_field, zr, bits, Access::read, OperandRole::source)
        .decoded();
}

RegisterEffects extract(std::uint32_t word) noexcept {
    if (field(word, 29, 2) != 0 || bit(word, 21)) {
        return unknown(word);  // op21 or o0 nonzero: not EXTR
    }
    const bool wide = bit(word, 31);
    if (wide != bit(word, 22) || (!wide && bit(word, 15))) {
        return reserved(word);
    }
    const unsigned bits = width(word);
    return Builder(word)
        .operation(Operation::extr)
        .general(rd_field, zr, bits, Access::write, OperandRole::destination)
        .general(rn_field, zr, bits, Access::read, OperandRole::source)
        .general(rm_field, zr, bits, Access::read, OperandRole::source)
        .decoded();
}

RegisterEffects data_processing_immediate(std::uint32_t word) noexcept {
    switch (field(word, 23, 3)) {
    case 2:
        return add_sub_immediate(word);
    case 4:
        return logical_immediate(word);
    case 5:
        return move_wide(word);
    case 6:
        return bitfield(word);
    case 7:
        return extract(word);
    default:
        // 00x is ADR/ADRP (classified before reaching here); 011 holds
        // extension classes (tagged add/subtract, min/max immediate).
        return unknown(word);
    }
}

// ---------------------------------------------------------------------------
// Data processing (register): bits 27..25 = 101.

// Three ordinary operands Rd = f(Rn, Rm) with field value 31 as the zero
// register.
Builder three_register(std::uint32_t word, Operation operation, unsigned bits) noexcept {
    Builder builder(word);
    builder.operation(operation)
        .general(rd_field, zr, bits, Access::write, OperandRole::destination)
        .general(rn_field, zr, bits, Access::read, OperandRole::source)
        .general(rm_field, zr, bits, Access::read, OperandRole::source);
    return builder;
}

RegisterEffects logical_shifted(std::uint32_t word) noexcept {
    if (!bit(word, 31) && bit(word, 15)) {
        return reserved(word);  // 32-bit shift amount above 31
    }
    static constexpr Operation operations[] = {
        Operation::and_shifted, Operation::bic_shifted, Operation::orr_shifted,
        Operation::orn_shifted, Operation::eor_shifted, Operation::eon_shifted,
        Operation::ands_shifted, Operation::bics_shifted,
    };
    const std::uint32_t index = (field(word, 29, 2) << 1) | field(word, 21, 1);
    return three_register(word, operations[index], width(word))
        .flags(false, field(word, 29, 2) == 3)
        .decoded();
}

RegisterEffects add_sub_shifted(std::uint32_t word) noexcept {
    if (field(word, 22, 2) == 3 || (!bit(word, 31) && bit(word, 15))) {
        return reserved(word);  // ROR shift or 32-bit shift amount above 31
    }
    static constexpr Operation operations[] = {Operation::add_shifted, Operation::adds_shifted,
                                               Operation::sub_shifted, Operation::subs_shifted};
    return three_register(word, operations[field(word, 29, 2)], width(word))
        .flags(false, bit(word, 29))
        .decoded();
}

RegisterEffects add_sub_extended(std::uint32_t word) noexcept {
    if (field(word, 22, 2) != 0) {
        return unknown(word);
    }
    if (field(word, 10, 3) > 4) {
        return reserved(word);  // left shift above 4
    }
    static constexpr Operation operations[] = {Operation::add_extended, Operation::adds_extended,
                                               Operation::sub_extended, Operation::subs_extended};
    const bool set_flags = bit(word, 29);
    const unsigned bits = width(word);
    // UXTX and SXTX (option<1:0> = 11) name an X register in 64-bit forms;
    // every other extension names a W register.
    const unsigned index_bits = bits == 64 && field(word, 13, 2) == 3 ? 64u : 32u;
    return Builder(word)
        .operation(operations[field(word, 29, 2)])
        .general(rd_field, set_flags ? zr : sp, bits, Access::write, OperandRole::destination)
        .general(rn_field, sp, bits, Access::read, OperandRole::source)
        .general(rm_field, zr, index_bits, Access::read, OperandRole::source)
        .flags(false, set_flags)
        .decoded();
}

RegisterEffects add_sub_carry(std::uint32_t word) noexcept {
    if (field(word, 10, 6) != 0) {
        return unknown(word);  // flag manipulation and checked-pointer classes
    }
    static constexpr Operation operations[] = {Operation::adc, Operation::adcs, Operation::sbc,
                                               Operation::sbcs};
    return three_register(word, operations[field(word, 29, 2)], width(word))
        .flags(true, bit(word, 29))
        .decoded();
}

RegisterEffects conditional_compare(std::uint32_t word) noexcept {
    if (!bit(word, 29) || bit(word, 10) || bit(word, 4)) {
        return reserved(word);  // S = 0, o2 = 1 or o3 = 1
    }
    const bool immediate = bit(word, 11);
    const bool compare = bit(word, 30);
    const Operation operation =
        immediate ? (compare ? Operation::ccmp_immediate : Operation::ccmn_immediate)
                  : (compare ? Operation::ccmp_register : Operation::ccmn_register);
    const unsigned bits = width(word);
    // The comparison and its operand reads happen only when the condition
    // holds; otherwise NZCV takes the immediate nzcv value.
    Builder builder(word);
    builder.operation(operation).general(rn_field, zr, bits, Access::read, OperandRole::source,
                                         true);
    if (!immediate) {
        builder.general(rm_field, zr, bits, Access::read, OperandRole::source, true);
    }
    return builder.flags(true, true).decoded();
}

RegisterEffects conditional_select(std::uint32_t word) noexcept {
    if (bit(word, 29) || bit(word, 11)) {
        return reserved(word);  // S = 1 or op2 = 1x
    }
    static constexpr Operation operations[] = {Operation::csel, Operation::csinc, Operation::csinv,
                                               Operation::csneg};
    const unsigned bits = width(word);
    // Rn is used when the condition holds and Rm otherwise.
    return Builder(word)
        .operation(operations[(field(word, 30, 1) << 1) | field(word, 10, 1)])
        .general(rd_field, zr, bits, Access::write, OperandRole::destination)
        .general(rn_field, zr, bits, Access::read, OperandRole::source, true)
        .general(rm_field, zr, bits, Access::read, OperandRole::source, true)
        .flags(true, false)
        .decoded();
}

RegisterEffects data_processing_two_source(std::uint32_t word) noexcept {
    if (bit(word, 29)) {
        return unknown(word);  // S = 1: extension forms (SUBPS)
    }
    const bool wide = bit(word, 31);
    const std::uint32_t opcode = field(word, 10, 6);
    switch (opcode) {
    case 0x02:
        return three_register(word, Operation::udiv, width(word)).decoded();
    case 0x03:
        return three_register(word, Operation::sdiv, width(word)).decoded();
    case 0x08:
        return three_register(word, Operation::lslv, width(word)).decoded();
    case 0x09:
        return three_register(word, Operation::lsrv, width(word)).decoded();
    case 0x0A:
        return three_register(word, Operation::asrv, width(word)).decoded();
    case 0x0B:
        return three_register(word, Operation::rorv, width(word)).decoded();
    default:
        break;
    }
    if ((opcode & 0x38) == 0x10) {
        // CRC32{C}{B,H,W,X}: sz = opcode<1:0>; only CRC32X/CRC32CX are 64-bit.
        const std::uint32_t size = opcode & 3;
        if (wide != (size == 3)) {
            return reserved(word);
        }
        static constexpr Operation operations[] = {
            Operation::crc32b, Operation::crc32h, Operation::crc32w, Operation::crc32x,
            Operation::crc32cb, Operation::crc32ch, Operation::crc32cw, Operation::crc32cx,
        };
        return Builder(word)
            .operation(operations[opcode & 7])
            .general(rd_field, zr, 32, Access::write, OperandRole::destination)
            .general(rn_field, zr, 32, Access::read, OperandRole::source)
            .general(rm_field, zr, size == 3 ? 64 : 32, Access::read, OperandRole::source)
            .decoded();
    }
    return unknown(word);
}

RegisterEffects data_processing_one_source(std::uint32_t word) noexcept {
    if (bit(word, 29) || field(word, 16, 5) != 0) {
        return unknown(word);  // pointer authentication and other extensions
    }
    const bool wide = bit(word, 31);
    Operation operation = Operation::none;
    switch (field(word, 10, 6)) {
    case 0:
        operation = Operation::rbit;
        break;
    case 1:
        operation = Operation::rev16;
        break;
    case 2:
        operation = wide ? Operation::rev32 : Operation::rev;
        break;
    case 3:
        if (!wide) {
            return reserved(word);
        }
        operation = Operation::rev;
        break;
    case 4:
        operation = Operation::clz;
        break;
    case 5:
        operation = Operation::cls;
        break;
    default:
        return unknown(word);
    }
    const unsigned bits = width(word);
    return Builder(word)
        .operation(operation)
        .general(rd_field, zr, bits, Access::write, OperandRole::destination)
        .general(rn_field, zr, bits, Access::read, OperandRole::source)
        .decoded();
}

RegisterEffects data_processing_three_source(std::uint32_t word) noexcept {
    if (field(word, 29, 2) != 0) {
        return unknown(word);
    }
    const bool wide = bit(word, 31);
    const std::uint32_t op31 = field(word, 21, 3);
    const bool o0 = bit(word, 15);
    if (!wide && op31 != 0) {
        return reserved(word);
    }
    Operation operation = Operation::none;
    unsigned source_bits = 64;
    bool high_half = false;
    switch (op31) {
    case 0:
        operation = o0 ? Operation::msub : Operation::madd;
        source_bits = width(word);
        break;
    case 1:
        operation = o0 ? Operation::smsubl : Operation::smaddl;
        source_bits = 32;
        break;
    case 5:
        operation = o0 ? Operation::umsubl : Operation::umaddl;
        source_bits = 32;
        break;
    case 2:
    case 6:
        if (o0) {
            return unknown(word);
        }
        operation = op31 == 2 ? Operation::smulh : Operation::umulh;
        high_half = true;
        break;
    default:
        return unknown(word);
    }
    const unsigned bits = width(word);
    Builder builder(word);
    builder.operation(operation)
        .general(rd_field, zr, bits, Access::write, OperandRole::destination)
        .general(rn_field, zr, source_bits, Access::read, OperandRole::source)
        .general(rm_field, zr, source_bits, Access::read, OperandRole::source);
    if (!high_half) {
        builder.general(ra_field, zr, bits, Access::read, OperandRole::accumulator);
        return builder.decoded();
    }
    // SMULH/UMULH Ra is a should-be-one field, not an operand.
    if (field(word, 10, 5) != 31) {
        return builder.note(EffectNote::should_be_one_field).unpredictable();
    }
    return builder.decoded();
}

RegisterEffects data_processing_register(std::uint32_t word) noexcept {
    const std::uint32_t op2 = field(word, 21, 4);
    if (!bit(word, 28)) {
        if ((op2 & 8) == 0) {
            return logical_shifted(word);
        }
        return (op2 & 1) == 0 ? add_sub_shifted(word) : add_sub_extended(word);
    }
    if (op2 & 8) {
        return data_processing_three_source(word);
    }
    switch (op2) {
    case 0:
        return add_sub_carry(word);
    case 2:
        return conditional_compare(word);
    case 4:
        return conditional_select(word);
    case 6:
        return bit(word, 30) ? data_processing_one_source(word) : data_processing_two_source(word);
    default:
        return unknown(word);
    }
}

// ---------------------------------------------------------------------------
// Loads and stores.

struct DataForm {
    Operation operation = Operation::none;
    MemoryOperation memory = MemoryOperation::none;
    std::uint8_t bytes = 0;       // memory element size, or scale for prefetch
    std::uint8_t register_bits = 0;
    bool sign_extend = false;
    bool vector = false;
};

enum class SingleClass : std::uint8_t {
    unsigned_offset, unscaled, pre_index, post_index, unprivileged, register_offset,
};

// size:V:opc of the single-register classes. A result with no operation is
// unallocated in the given class.
DataForm single_form(std::uint32_t size, bool vector, std::uint32_t opc,
                     SingleClass kind) noexcept {
    if (vector) {
        if (kind == SingleClass::unprivileged) {
            return {};
        }
        std::uint8_t bytes = static_cast<std::uint8_t>(1u << size);
        if (opc & 2) {
            if (size != 0) {
                return {};
            }
            bytes = 16;
        }
        const bool load = (opc & 1) != 0;
        return {.operation = load ? Operation::ldr_vector : Operation::str_vector,
                .memory = load ? MemoryOperation::load : MemoryOperation::store,
                .bytes = bytes,
                .register_bits = static_cast<std::uint8_t>(bytes * 8),
                .vector = true};
    }
    const auto bytes = static_cast<std::uint8_t>(1u << size);
    switch (opc) {
    case 0: {
        static constexpr Operation stores[] = {Operation::strb, Operation::strh, Operation::str,
                                               Operation::str};
        return {.operation = stores[size],
                .memory = MemoryOperation::store,
                .bytes = bytes,
                .register_bits = static_cast<std::uint8_t>(size == 3 ? 64 : 32)};
    }
    case 1: {
        static constexpr Operation loads[] = {Operation::ldrb, Operation::ldrh, Operation::ldr,
                                              Operation::ldr};
        return {.operation = loads[size],
                .memory = MemoryOperation::load,
                .bytes = bytes,
                .register_bits = static_cast<std::uint8_t>(size == 3 ? 64 : 32)};
    }
    case 2: {
        if (size == 3) {
            // PRFM/PRFUM exist only with an offset, never with writeback or
            // as an unprivileged access.
            if (kind == SingleClass::pre_index || kind == SingleClass::post_index ||
                kind == SingleClass::unprivileged) {
                return {};
            }
            return {.operation = Operation::prfm, .memory = MemoryOperation::prefetch, .bytes = 8};
        }
        static constexpr Operation loads[] = {Operation::ldrsb, Operation::ldrsh, Operation::ldrsw};
        return {.operation = loads[size],
                .memory = MemoryOperation::load,
                .bytes = bytes,
                .register_bits = 64,
                .sign_extend = true};
    }
    default:
        if (size >= 2) {
            return {};
        }
        return {.operation = size == 0 ? Operation::ldrsb : Operation::ldrsh,
                .memory = MemoryOperation::load,
                .bytes = bytes,
                .register_bits = 32,
                .sign_extend = true};
    }
}

constexpr unsigned log2_bytes(unsigned bytes) noexcept {
    unsigned shift = 0;
    while ((1u << shift) < bytes) {
        ++shift;
    }
    return shift;
}

void add_transfer(Builder& builder, const DataForm& form, std::uint8_t field_lsb) noexcept {
    if (form.memory == MemoryOperation::prefetch) {
        return;  // Rt is the prefetch operation, not a register
    }
    const Access access = form.memory == MemoryOperation::load ? Access::write : Access::read;
    if (form.vector) {
        builder.vector(field_lsb, form.register_bits, access, OperandRole::transfer);
    } else {
        builder.general(field_lsb, zr, form.register_bits, access, OperandRole::transfer);
    }
}

MemoryAttributes attributes_of(const DataForm& form) noexcept {
    return form.sign_extend ? static_cast<MemoryAttributes>(MemoryAttribute::sign_extend)
                            : MemoryAttributes{0};
}

// Atomic memory operations (bits 29..27 = 111, V = 0, bits 25..24 = 00,
// bit 21 = 1, bits 11..10 = 00): size V A R 1 Rs o3 opc 00 Rn Rt. A orders
// the load as an acquire, R the store as a release.
RegisterEffects atomic_memory(std::uint32_t word) noexcept {
    const std::uint32_t size = field(word, 30, 2);
    const bool acquire = bit(word, 23);
    const bool release = bit(word, 22);
    const std::uint32_t rs = field(word, 16, 5);
    const bool o3 = bit(word, 15);
    const std::uint32_t opc = field(word, 12, 3);
    const unsigned register_bits = size == 3 ? 64 : 32;

    Operation operation = Operation::none;
    bool load_only = false;
    if (!o3) {
        static constexpr Operation operations[] = {Operation::ldadd, Operation::ldclr, Operation::ldeor,
                                                   Operation::ldset, Operation::ldsmax, Operation::ldsmin,
                                                   Operation::ldumax, Operation::ldumin};
        operation = operations[opc];
    } else if (opc == 0) {
        operation = Operation::swp;
    } else if (opc == 4 && acquire && !release && rs == 31) {
        operation = Operation::ldapr;
        load_only = true;
    } else {
        return unknown(word);  // LD64B/ST64B family and unallocated forms
    }

    MemoryAttributes attributes = 0;
    if (acquire) attributes |= static_cast<MemoryAttributes>(MemoryAttribute::acquire);
    if (release) attributes |= static_cast<MemoryAttributes>(MemoryAttribute::release);
    Builder builder(word);
    builder.operation(operation);
    if (!load_only) builder.general(rm_field, zr, register_bits, Access::read, OperandRole::source);
    builder.general(rd_field, zr, register_bits, Access::write, OperandRole::transfer)
        .general(rn_field, sp, 64, Access::read, OperandRole::base)
        .memory(MemoryEffect{.operation = load_only ? MemoryOperation::load : MemoryOperation::atomic,
                             .addressing = AddressingMode::base_register,
                             .element_bytes = static_cast<std::uint8_t>(1u << size),
                             .elements = 1,
                             .attributes = attributes});
    if (field(word, 5, 5) == 31) builder.note(EffectNote::stack_pointer_base);
    return builder.decoded();
}

// Compare and swap (bits 29..23 = 0010001, bit 21 = 1): size 0010001 L 1 Rs
// o0 11111 Rn Rt. L orders the load as an acquire, o0 the store as a release.
RegisterEffects compare_and_swap(std::uint32_t word) noexcept {
    const std::uint32_t size = field(word, 30, 2);
    const unsigned register_bits = size == 3 ? 64 : 32;
    MemoryAttributes attributes = 0;
    if (bit(word, 22)) attributes |= static_cast<MemoryAttributes>(MemoryAttribute::acquire);
    if (bit(word, 15)) attributes |= static_cast<MemoryAttributes>(MemoryAttribute::release);
    Builder builder(word);
    builder.operation(Operation::cas)
        .general(rm_field, zr, register_bits, Access::read_write, OperandRole::transfer)
        .general(rd_field, zr, register_bits, Access::read, OperandRole::transfer)
        .general(rn_field, sp, 64, Access::read, OperandRole::base)
        .memory(MemoryEffect{.operation = MemoryOperation::atomic,
                             .addressing = AddressingMode::base_register,
                             .element_bytes = static_cast<std::uint8_t>(1u << size),
                             .elements = 1,
                             .attributes = attributes});
    if (field(word, 5, 5) == 31) builder.note(EffectNote::stack_pointer_base);
    if (field(word, 10, 5) != 31) {
        builder.note(EffectNote::should_be_one_field);
        return builder.unpredictable();
    }
    return builder.decoded();
}

// Compare and swap pair (bit 31 = 0, bits 29..23 = 0010000, bit 21 = 1): 0 sz
// 0010000 L 1 Rs o0 11111 Rn Rt. Rs and Rt name even registers; the pairs are
// Rs, Rs+1 and Rt, Rt+1, and an odd Rs or Rt is UNDEFINED.
RegisterEffects compare_and_swap_pair(std::uint32_t word) noexcept {
    const std::uint32_t rs = field(word, 16, 5);
    const std::uint32_t rt = field(word, 0, 5);
    if ((rs & 1) || (rt & 1)) return reserved(word);
    const unsigned register_bits = bit(word, 30) ? 64 : 32;
    MemoryAttributes attributes = 0;
    if (bit(word, 22)) attributes |= static_cast<MemoryAttributes>(MemoryAttribute::acquire);
    if (bit(word, 15)) attributes |= static_cast<MemoryAttributes>(MemoryAttribute::release);
    Builder builder(word);
    builder.operation(Operation::casp)
        .general(rm_field, zr, register_bits, Access::read_write, OperandRole::transfer)
        .implicit_general(rs + 1, zr, Access::read_write, OperandRole::transfer, register_bits)
        .general(rd_field, zr, register_bits, Access::read, OperandRole::transfer)
        .implicit_general(rt + 1, zr, Access::read, OperandRole::transfer, register_bits)
        .general(rn_field, sp, 64, Access::read, OperandRole::base)
        .memory(MemoryEffect{.operation = MemoryOperation::atomic,
                             .addressing = AddressingMode::base_register,
                             .element_bytes = static_cast<std::uint8_t>(register_bits / 8),
                             .elements = 2,
                             .attributes = attributes});
    if (field(word, 5, 5) == 31) builder.note(EffectNote::stack_pointer_base);
    if (field(word, 10, 5) != 31) {
        builder.note(EffectNote::should_be_one_field);
        return builder.unpredictable();
    }
    return builder.decoded();
}

RegisterEffects load_store_single(std::uint32_t word) noexcept {
    const std::uint32_t size = field(word, 30, 2);
    const bool vector = bit(word, 26);
    const std::uint32_t opc = field(word, 22, 2);

    SingleClass kind = SingleClass::unsigned_offset;
    if (!bit(word, 24)) {
        if (!bit(word, 21)) {
            static constexpr SingleClass kinds[] = {SingleClass::unscaled, SingleClass::post_index,
                                                    SingleClass::unprivileged,
                                                    SingleClass::pre_index};
            kind = kinds[field(word, 10, 2)];
        } else if (field(word, 10, 2) == 2) {
            kind = SingleClass::register_offset;
        } else if (field(word, 10, 2) == 0 && !vector) {
            return atomic_memory(word);
        } else {
            return unknown(word);  // pointer-authenticated loads, unallocated forms
        }
    }

    const DataForm form = single_form(size, vector, opc, kind);
    if (form.operation == Operation::none) {
        return reserved(word);
    }

    MemoryEffect memory{.operation = form.memory,
                        .element_bytes = form.memory == MemoryOperation::prefetch ? std::uint8_t{0}
                                                                                  : form.bytes,
                        .elements = static_cast<std::uint8_t>(
                            form.memory == MemoryOperation::prefetch ? 0 : 1),
                        .attributes = attributes_of(form)};
    const std::int64_t imm9 = sign_extend(field(word, 12, 9), 9);
    switch (kind) {
    case SingleClass::unsigned_offset:
        memory.addressing = AddressingMode::unsigned_offset;
        memory.offset = static_cast<std::int64_t>(field(word, 10, 12)) * form.bytes;
        break;
    case SingleClass::unscaled:
        memory.addressing = AddressingMode::unscaled_offset;
        memory.offset = imm9;
        break;
    case SingleClass::unprivileged:
        memory.addressing = AddressingMode::unscaled_offset;
        memory.offset = imm9;
        memory.attributes = static_cast<MemoryAttributes>(
            memory.attributes | static_cast<MemoryAttributes>(MemoryAttribute::unprivileged));
        break;
    case SingleClass::pre_index:
        memory.addressing = AddressingMode::pre_index;
        memory.offset = imm9;
        memory.writeback = true;
        break;
    case SingleClass::post_index:
        memory.addressing = AddressingMode::post_index;
        memory.offset = imm9;
        memory.writeback = true;
        break;
    case SingleClass::register_offset: {
        const std::uint32_t option = field(word, 13, 3);
        if ((option & 2) == 0) {
            return reserved(word);  // option<1> = 0 is reserved
        }
        if (form.memory == MemoryOperation::prefetch && (word & 0x18u) == 0x18u) {
            // FEAT_RPRFM uses Rm as 64-bit range metadata, not a PRFM index.
            return unknown(word);
        }
        static constexpr IndexExtend extends[] = {IndexExtend::uxtw, IndexExtend::lsl,
                                                  IndexExtend::sxtw, IndexExtend::sxtx};
        memory.addressing = AddressingMode::register_offset;
        memory.extend = extends[((option >> 2) << 1) | (option & 1)];
        memory.index_shift =
            static_cast<std::uint8_t>(bit(word, 12) ? log2_bytes(form.bytes) : 0);
        break;
    }
    }

    Builder builder(word);
    builder.operation(form.operation);
    add_transfer(builder, form, rd_field);
    builder.general(rn_field, sp, 64, memory.writeback ? Access::read_write : Access::read,
                    OperandRole::base);
    if (kind == SingleClass::register_offset) {
        const unsigned index_bits = (field(word, 13, 3) & 1) ? 64u : 32u;
        builder.general(rm_field, zr, index_bits, Access::read, OperandRole::index);
    }
    builder.memory(memory);

    const std::uint32_t rt = field(word, 0, 5);
    const std::uint32_t rn = field(word, 5, 5);
    if (rn == 31) {
        builder.note(EffectNote::stack_pointer_base);
    }
    if (memory.writeback && !form.vector && form.memory != MemoryOperation::prefetch && rt == rn &&
        rn != 31) {
        return builder.note(EffectNote::writeback_overlaps_transfer).unpredictable();
    }
    return builder.decoded();
}

RegisterEffects load_store_pair(std::uint32_t word) noexcept {
    const std::uint32_t opc = field(word, 30, 2);
    const bool vector = bit(word, 26);
    const std::uint32_t mode = field(word, 23, 2);
    const bool load = bit(word, 22);
    const bool non_temporal = mode == 0;

    DataForm form;
    form.vector = vector;
    form.memory = load ? MemoryOperation::load : MemoryOperation::store;
    if (vector) {
        if (opc == 3) {
            return unknown(word);
        }
        form.bytes = static_cast<std::uint8_t>(4u << opc);
        form.register_bits = static_cast<std::uint8_t>(form.bytes * 8);
        form.operation = non_temporal ? (load ? Operation::ldnp_vector : Operation::stnp_vector)
                                      : (load ? Operation::ldp_vector : Operation::stp_vector);
    } else if (opc == 0 || opc == 2) {
        form.bytes = opc == 0 ? 4 : 8;
        form.register_bits = static_cast<std::uint8_t>(form.bytes * 8);
        form.operation = non_temporal ? (load ? Operation::ldnp : Operation::stnp)
                                      : (load ? Operation::ldp : Operation::stp);
    } else if (opc == 1 && load && !non_temporal) {
        form.bytes = 4;
        form.register_bits = 64;
        form.sign_extend = true;
        form.operation = Operation::ldpsw;
    } else {
        return unknown(word);  // STGP, other extension and unallocated forms
    }

    static constexpr AddressingMode modes[] = {AddressingMode::signed_offset,
                                               AddressingMode::post_index,
                                               AddressingMode::signed_offset,
                                               AddressingMode::pre_index};
    MemoryEffect memory{.operation = form.memory,
                        .addressing = modes[mode],
                        .element_bytes = form.bytes,
                        .elements = 2,
                        .offset = sign_extend(field(word, 15, 7), 7) * form.bytes,
                        .writeback = mode == 1 || mode == 3,
                        .attributes = attributes_of(form)};
    if (non_temporal) {
        memory.attributes = static_cast<MemoryAttributes>(
            memory.attributes | static_cast<MemoryAttributes>(MemoryAttribute::non_temporal));
    }

    Builder builder(word);
    builder.operation(form.operation);
    add_transfer(builder, form, rd_field);
    add_transfer(builder, form, ra_field);
    builder.general(rn_field, sp, 64, memory.writeback ? Access::read_write : Access::read,
                    OperandRole::base);
    builder.memory(memory);

    const std::uint32_t rt = field(word, 0, 5);
    const std::uint32_t rt2 = field(word, 10, 5);
    const std::uint32_t rn = field(word, 5, 5);
    if (rn == 31) {
        builder.note(EffectNote::stack_pointer_base);
    }
    bool unpredictable = false;
    if (load && rt == rt2) {
        builder.note(EffectNote::pair_load_same_register);
        unpredictable = true;
    }
    if (memory.writeback && !vector && rn != 31 && (rt == rn || rt2 == rn)) {
        builder.note(EffectNote::writeback_overlaps_transfer);
        unpredictable = true;
    }
    return unpredictable ? builder.unpredictable() : builder.decoded();
}

// Load/store exclusive register and pair, and load-acquire/store-release
// (bits 29..24 = 001000): size o2 L o1 Rs o0 Rt2 Rn Rt.
RegisterEffects load_store_exclusive(std::uint32_t word) noexcept {
    const std::uint32_t size = field(word, 30, 2);
    const bool ordered = bit(word, 23);  // o2
    const bool load = bit(word, 22);
    const bool o1 = bit(word, 21);
    const bool o0 = bit(word, 15);
    const std::uint32_t rs = field(word, 16, 5);
    const std::uint32_t rt2 = field(word, 10, 5);
    const std::uint32_t rn = field(word, 5, 5);
    const std::uint32_t rt = field(word, 0, 5);

    bool pair = false;
    if (ordered) {
        if (o1) return compare_and_swap(word);
        if (!o0) return unknown(word);  // LDLAR/STLLR
    } else if (o1) {
        if (!bit(word, 31)) return compare_and_swap_pair(word);
        pair = true;
    }

    unsigned bytes = 1u << size;
    unsigned register_bits = size == 3 ? 64 : 32;
    if (pair) {
        bytes = bit(word, 30) ? 8 : 4;
        register_bits = bytes * 8;
    }

    Operation operation = Operation::none;
    if (ordered) {
        static constexpr Operation loads[] = {Operation::ldarb, Operation::ldarh, Operation::ldar,
                                              Operation::ldar};
        static constexpr Operation stores[] = {Operation::stlrb, Operation::stlrh, Operation::stlr,
                                               Operation::stlr};
        operation = load ? loads[size] : stores[size];
    } else if (pair) {
        operation = load ? (o0 ? Operation::ldaxp : Operation::ldxp)
                         : (o0 ? Operation::stlxp : Operation::stxp);
    } else {
        static constexpr Operation loads[2][4] = {
            {Operation::ldxrb, Operation::ldxrh, Operation::ldxr, Operation::ldxr},
            {Operation::ldaxrb, Operation::ldaxrh, Operation::ldaxr, Operation::ldaxr},
        };
        static constexpr Operation stores[2][4] = {
            {Operation::stxrb, Operation::stxrh, Operation::stxr, Operation::stxr},
            {Operation::stlxrb, Operation::stlxrh, Operation::stlxr, Operation::stlxr},
        };
        operation = load ? loads[o0][size] : stores[o0][size];
    }

    MemoryAttributes attributes = 0;
    if (!ordered) {
        attributes |= static_cast<MemoryAttributes>(MemoryAttribute::exclusive);
    }
    if (o0) {
        attributes |= static_cast<MemoryAttributes>(load ? MemoryAttribute::acquire
                                                         : MemoryAttribute::release);
    }

    const Access transfer_access = load ? Access::write : Access::read;
    Builder builder(word);
    builder.operation(operation)
        .general(rd_field, zr, register_bits, transfer_access, OperandRole::transfer);
    if (pair) {
        builder.general(ra_field, zr, register_bits, transfer_access, OperandRole::transfer);
    }
    builder.general(rn_field, sp, 64, Access::read, OperandRole::base);
    const bool has_status = !ordered && !load;
    if (has_status) {
        builder.general(rm_field, zr, 32, Access::write, OperandRole::status);
    }
    builder.memory(MemoryEffect{.operation = load ? MemoryOperation::load : MemoryOperation::store,
                                .addressing = AddressingMode::base_register,
                                .element_bytes = static_cast<std::uint8_t>(bytes),
                                .elements = static_cast<std::uint8_t>(pair ? 2 : 1),
                                .attributes = attributes});
    if (rn == 31) {
        builder.note(EffectNote::stack_pointer_base);
    }

    bool unpredictable = false;
    // Fields the form does not use must be all ones.
    if ((!has_status && rs != 31) || (!pair && rt2 != 31)) {
        builder.note(EffectNote::should_be_one_field);
        unpredictable = true;
    }
    if (has_status && (rs == rt || (pair && rs == rt2) || (rs == rn && rn != 31))) {
        builder.note(EffectNote::status_overlaps_operand);
        unpredictable = true;
    }
    if (load && pair && rt == rt2) {
        builder.note(EffectNote::pair_load_same_register);
        unpredictable = true;
    }
    return unpredictable ? builder.unpredictable() : builder.decoded();
}

// Structure lists encode only their first vector register. Later registers
// wrap modulo 32 and have no independently rewritable instruction field.
RegisterEffects structure_transfer(std::uint32_t word) noexcept {
    const bool single = bit(word, 24);
    const bool post = bit(word, 23);
    const bool load = bit(word, 22);
    const bool q = bit(word, 30);
    const unsigned rm = field(word, 16, 5);
    const unsigned size = field(word, 10, 2);
    if ((!single && bit(word, 21)) || (!post && rm != 0)) {
        // Includes extension encodings such as LDAP1/STL1, not baseline forms.
        return unknown(word);
    }
    unsigned count = 0;
    unsigned bytes = 0;
    unsigned vector_bits = q ? 128u : 64u;
    bool replicate = false;
    Operation operation;
    if (single) {
        const unsigned opcode = field(word, 13, 3);
        unsigned scale = opcode >> 1;
        count = ((opcode & 1u) << 1) + unsigned(bit(word, 21)) + 1;
        if (scale == 3) {
            if (!load || bit(word, 12)) return reserved(word);
            replicate = true;
            scale = size;
        } else {
            if ((scale == 1 && (size & 1u) != 0) ||
                (scale == 2 && ((size & 2u) != 0 || (size == 1 && bit(word, 12))))) {
                return reserved(word);
            }
            if (scale == 2 && size == 1) scale = 3;
            vector_bits = 128;  // A lane load preserves all other lanes.
        }
        bytes = 1u << scale;
        static constexpr Operation stores[] = {Operation::st1_single, Operation::st2_single,
                                               Operation::st3_single, Operation::st4_single};
        static constexpr Operation loads[] = {Operation::ld1_single, Operation::ld2_single,
                                              Operation::ld3_single, Operation::ld4_single};
        static constexpr Operation replicates[] = {Operation::ld1r, Operation::ld2r,
                                                   Operation::ld3r, Operation::ld4r};
        operation = replicate ? replicates[count - 1] : load ? loads[count - 1] : stores[count - 1];
    } else {
        const unsigned opcode = field(word, 12, 4);
        unsigned structures;
        switch (opcode) {
        case 0: count = 4; structures = 4; break;
        case 2: count = 4; structures = 1; break;
        case 4: count = 3; structures = 3; break;
        case 6: count = 3; structures = 1; break;
        case 7: count = 1; structures = 1; break;
        case 8: count = 2; structures = 2; break;
        case 10: count = 2; structures = 1; break;
        default: return reserved(word);
        }
        if (!q && size == 3 && structures != 1) return reserved(word);
        bytes = vector_bits / 8;
        static constexpr Operation stores[] = {Operation::st1_multiple, Operation::st2_multiple,
                                               Operation::st3_multiple, Operation::st4_multiple};
        static constexpr Operation loads[] = {Operation::ld1_multiple, Operation::ld2_multiple,
                                              Operation::ld3_multiple, Operation::ld4_multiple};
        operation = load ? loads[structures - 1] : stores[structures - 1];
    }
    const Access vector_access = !load ? Access::read
        : single && !replicate ? Access::read_write : Access::write;
    Builder builder(word);
    builder.operation(operation).vector(rd_field, vector_bits, vector_access, OperandRole::transfer);
    for (unsigned index = 1; index < count; ++index) {
        auto operand = builder.raw().operands[0];
        operand.field = implicit_operand;
        operand.number = static_cast<std::uint8_t>((operand.number + index) & 31u);
        builder.raw().operands[builder.raw().operand_count++] = operand;
    }
    builder.general(rn_field, sp, 64, post ? Access::read_write : Access::read, OperandRole::base);
    const bool register_post = post && rm != 31;
    if (register_post) builder.general(rm_field, zr, 64, Access::read, OperandRole::index);
    builder.memory(MemoryEffect{
        .operation = load ? MemoryOperation::load : MemoryOperation::store,
        .addressing = post ? AddressingMode::post_index : AddressingMode::base_register,
        .element_bytes = static_cast<std::uint8_t>(bytes),
        .elements = static_cast<std::uint8_t>(count),
        .extend = register_post ? IndexExtend::lsl : IndexExtend::none,
        .offset = post && !register_post ? static_cast<std::int64_t>(bytes * count) : 0,
        .writeback = post});
    if (field(word, 5, 5) == 31) builder.note(EffectNote::stack_pointer_base);
    return builder.decoded();
}

// ---------------------------------------------------------------------------
// Data Processing - Scalar Floating-Point and Advanced SIMD (op0 = x111).
//
// Field layouts and allocation follow the A64 encoding tables (Armv8.0-A
// through Armv8.2-A FP16, plus the dot-product extension), checked against
// words encoded by the host assembler. Encodings allocated only by extensions
// outside that set (FRINT32/64, FJCVTZS, FCMA, RDMA, SHA3/SM3/SM4/SHA512) stay
// EffectStatus::unknown; unallocated field values inside a recognized frame
// return reserved.

constexpr bool q_bit(std::uint32_t word) noexcept { return bit(word, 30); }
constexpr bool u_bit(std::uint32_t word) noexcept { return bit(word, 29); }
constexpr std::uint32_t size_field(std::uint32_t word) noexcept { return field(word, 22, 2); }
constexpr std::uint32_t rm_value(std::uint32_t word) noexcept { return field(word, 16, 5); }
constexpr std::uint32_t opcode6(std::uint32_t word) noexcept { return field(word, 10, 6); }
constexpr std::uint32_t opcode5(std::uint32_t word) noexcept { return field(word, 11, 5); }
constexpr unsigned register_bits(bool q) noexcept { return q ? 128u : 64u; }

// A decoded scalar SIMD&FP instruction: every operand names the low
// element. No general register is named and NZCV is untouched.
RegisterEffects scalar_effects(std::uint32_t word, Operation operation,
                               std::initializer_list<RegisterOperand> operands) noexcept {
    Builder builder(word);
    builder.operation(operation);
    for (const auto& operand : operands) {
        builder.raw().operands[builder.raw().operand_count++] = operand;
    }
    return builder.decoded();
}

RegisterOperand scalar_vector(std::uint32_t word, std::uint8_t field_lsb, unsigned bits,
                              Access access, OperandRole role) noexcept {
    return RegisterOperand{.field = field_lsb,
                           .number = static_cast<std::uint8_t>(field(word, field_lsb, 5)),
                           .file = RegisterFile::vector,
                           .at_31 = Register31::not_applicable,
                           .bits = static_cast<std::uint8_t>(bits),
                           .access = access,
                           .role = role};
}

// A decoded vector instruction: destination first, then sources in field
// order. No general register is ever named; flags change only in the scalar
// compare groups handled separately.
RegisterEffects vector_effects(std::uint32_t word, Operation operation,
                               unsigned destination_bits, Access destination_access,
                               std::initializer_list<unsigned> source_bits) noexcept {
    Builder builder(word);
    builder.operation(operation)
        .vector(rd_field, destination_bits, destination_access, OperandRole::destination);
    constexpr std::uint8_t source_fields[] = {rn_field, rm_field, ra_field};
    unsigned index = 0;
    for (unsigned bits : source_bits) {
        builder.vector(source_fields[index++], bits, Access::read, OperandRole::source);
    }
    return builder.decoded();
}

// TBL writes zeros for out-of-range indices; TBX keeps the destination
// lanes, so it reads Vd. Table registers after the first are implicit.
RegisterEffects table_lookup(std::uint32_t word, bool keep) noexcept {
    const unsigned registers = 1u + field(word, 13, 2);
    Builder builder(word);
    builder.operation(Operation::simd_data)
        .vector(rd_field, register_bits(q_bit(word)),
                keep ? Access::read_write : Access::write, OperandRole::destination)
        .vector(rn_field, register_bits(true), Access::read, OperandRole::source);
    for (unsigned index = 1; index < registers; ++index) {
        auto operand = builder.raw().operands[1];
        operand.field = implicit_operand;
        operand.number = static_cast<std::uint8_t>((operand.number + index) & 31u);
        builder.raw().operands[builder.raw().operand_count++] = operand;
    }
    builder.vector(rm_field, register_bits(true), Access::read, OperandRole::source);
    return builder.decoded();
}

// ---------------------------------------------------------------------------
// Vector x indexed element (bit24 = 1, bit10 = 0): the size field selects
// the arrangement, bits 21:20 and 11 hold the lane index, and the low
// should-be-zero index bits constrain the allocation.

// Cryptographic AES, two-register SHA and three-register SHA (A64 crypto groups, size = 00).
// They name only SIMD&FP registers; the destination is also an input except for AESMC/AESIMC
// and SHA1H.
RegisterEffects simd_crypto(std::uint32_t word) noexcept {
    const std::uint32_t size = field(word, 22, 2);
    if ((word & 0xFF3E0C00u) == 0x4E280800u) {  // AES
        const std::uint32_t opcode = field(word, 12, 5);
        if (size != 0 || opcode < 4 || opcode > 7) return reserved(word);
        const bool rounds = opcode == 4 || opcode == 5;  // AESE, AESD
        return vector_effects(word, Operation::simd_data, 128u,
                              rounds ? Access::read_write : Access::write, {128u});
    }
    if ((word & 0xFF3E0C00u) == 0x5E280800u) {  // two-register SHA
        const std::uint32_t opcode = field(word, 12, 5);
        if (size != 0 || opcode > 2) return reserved(word);
        if (opcode == 0) {  // SHA1H Sd, Sn
            return vector_effects(word, Operation::simd_data, 32u, Access::write, {32u});
        }
        return vector_effects(word, Operation::simd_data, 128u, Access::read_write, {128u});
    }
    if ((word & 0xFF208C00u) == 0x5E000000u) {  // three-register SHA
        const std::uint32_t opcode = field(word, 12, 3);
        if (size != 0 || opcode == 7) return reserved(word);
        // SHA1C/P/M take Sn as the second operand; the others read full vectors.
        const unsigned rn_bits = opcode <= 2 ? 32u : 128u;
        return vector_effects(word, Operation::simd_data, 128u, Access::read_write,
                              {rn_bits, 128u});
    }
    return unknown(word);
}

RegisterEffects simd_vector_indexed(std::uint32_t word) noexcept {
    // Advanced SIMD vector x indexed element: 0 Q U 01111 size L M Rm opcode H 0 Rn Rd.
    // size (bits 23:22) selects the element; L, M and H only form the lane index and never
    // change allocation or the registers named. Rows follow the Arm A64 encoding table.
    const unsigned u = u_bit(word) ? 1u : 0u;
    const std::uint32_t size = field(word, 22, 2);
    const std::uint32_t opcode = field(word, 12, 4);
    const bool q = q_bit(word);
    enum class Shape { plain, accumulate, long_plain, long_accumulate };
    // Element masks: bit n set means size n is allocated.
    constexpr unsigned integer_hs = (1u << 1) | (1u << 2);  // H, S
    constexpr unsigned float_hsd = (1u << 0) | (1u << 2) | (1u << 3);  // FP16, single, double
    struct Row { unsigned u; std::uint32_t opcode; unsigned sizes; Shape shape; bool floating; };
    static constexpr Row rows[] = {
        {0, 0x1, float_hsd, Shape::accumulate, true},       // FMLA
        {0, 0x2, integer_hs, Shape::long_accumulate, false}, // SMLAL{2}
        {0, 0x3, integer_hs, Shape::long_accumulate, false}, // SQDMLAL{2}
        {0, 0x5, float_hsd, Shape::accumulate, true},       // FMLS
        {0, 0x6, integer_hs, Shape::long_accumulate, false}, // SMLSL{2}
        {0, 0x7, integer_hs, Shape::long_accumulate, false}, // SQDMLSL{2}
        {0, 0x8, integer_hs, Shape::plain, false},           // MUL
        {0, 0x9, float_hsd, Shape::plain, true},             // FMUL
        {0, 0xA, integer_hs, Shape::long_plain, false},      // SMULL{2}
        {0, 0xB, integer_hs, Shape::long_plain, false},      // SQDMULL{2}
        {0, 0xC, integer_hs, Shape::plain, false},           // SQDMULH
        {0, 0xD, integer_hs, Shape::plain, false},           // SQRDMULH
        {0, 0xE, 1u << 2, Shape::accumulate, false},         // SDOT
        {1, 0x0, integer_hs, Shape::accumulate, false},      // MLA
        {1, 0x2, integer_hs, Shape::long_accumulate, false}, // UMLAL{2}
        {1, 0x4, integer_hs, Shape::accumulate, false},      // MLS
        {1, 0x6, integer_hs, Shape::long_accumulate, false}, // UMLSL{2}
        {1, 0x9, float_hsd, Shape::plain, true},             // FMULX
        {1, 0xA, integer_hs, Shape::long_plain, false},      // UMULL{2}
        {1, 0xD, integer_hs, Shape::accumulate, false},      // SQRDMLAH
        {1, 0xE, 1u << 2, Shape::accumulate, false},         // UDOT
        {1, 0xF, integer_hs, Shape::accumulate, false},      // SQRDMLSH
    };
    const Row* matched = nullptr;
    for (const auto& row : rows) {
        if (row.u == u && row.opcode == opcode) { matched = &row; break; }
    }
    if (matched == nullptr) {
        return unknown(word);  // FMLAL, FCMLA, BFDOT and other extension rows
    }
    if ((matched->sizes & (1u << size)) == 0) {
        return reserved(word);
    }
    // Double-precision FP by element needs Q = 1 and L = 0.
    if (matched->floating && size == 3 && (!q || bit(word, 21))) {
        return reserved(word);
    }
    const bool long_result = matched->shape == Shape::long_plain ||
                             matched->shape == Shape::long_accumulate;
    const bool accumulates = matched->shape != Shape::plain;
    return vector_effects(word, Operation::simd_data,
                          long_result ? 128u : register_bits(q),
                          accumulates ? Access::read_write : Access::write,
                          {register_bits(q), 128u});
}

// ---------------------------------------------------------------------------
// Vector shift by immediate (bit24 = 1 frame, bit23 = 0, nonzero immh).

RegisterEffects simd_shift_immediate(std::uint32_t word) noexcept {
    const unsigned immh = field(word, 19, 4);
    if (immh == 0) {
        return reserved(word);
    }
    const std::uint32_t opcode = opcode6(word);
    if (!bit(word, 10)) {
        return reserved(word);
    }
    enum class Shape { plain, accumulate, narrow, widening, fp_convert };
    Shape shape = Shape::plain;
    bool allocated = false;
    const unsigned u = u_bit(word) ? 1u : 0u;
    switch (opcode | (u << 8)) {
    case 0x01: case 0x09: case 0x15:             // SSHR, SRSHR, SHL
    case 0x05:                                   // SSRA
    case 0x0D:                                   // SRSRA
    case 0x1D:                                   // SQSHL
    case 0x101: case 0x109: case 0x10D:          // USHR, URSHR, URSRA
    case 0x105: case 0x111: case 0x115:          // USRA, SRI, SLI
    case 0x11D:                                  // SQSHLU
        shape = opcode == 0x05 || opcode == 0x0D || (u && (opcode == 0x11 || opcode == 0x15))
                    ? Shape::accumulate
                    : Shape::plain;
        allocated = true;
        break;
    case 0x21: case 0x23: case 0x25: case 0x27:  // SHRN, RSHRN, SQSHRN, SQRSHRN
    case 0x121: case 0x123: case 0x125: case 0x127:  // SQSHRUN, SQRSHRUN, UQSHRN, UQRSHRN
        shape = Shape::narrow;
        allocated = true;
        break;
    case 0x29: case 0x129:                       // SSHLL/USHLL
        shape = Shape::widening;
        allocated = true;
        break;
    case 0x39: case 0x139:                       // SCVTF/UCVTF fixed point
    case 0x3F: case 0x13F:                       // FCVTZS/FCVTZU fixed point
        shape = Shape::fp_convert;
        allocated = true;
        break;
    default:
        break;
    }
    if (!allocated) {
        return reserved(word);
    }
    // Widening (SHLL) has no doubleword-element source row.
    if (shape == Shape::widening && immh > 7) {
        return reserved(word);
    }
    // immh selects the largest element: 0001 byte, 001x half, 01xx word,
    // 1xxx doubleword. Narrowing reads that element and writes the next
    // smaller one; widening reads it and writes the next larger one.
    unsigned element_log2 = 0;
    for (unsigned mask = immh; mask > 1u; mask >>= 1u) {
        ++element_log2;
    }
    unsigned destination_bits = register_bits(q_bit(word));
    unsigned source_bits = destination_bits;
    Access destination_access = Access::write;
    if (shape == Shape::narrow) {
        source_bits = 128u;
    } else if (shape == Shape::widening) {
        destination_bits = 128u;
        source_bits = register_bits(q_bit(word));
    } else if (shape == Shape::accumulate) {
        destination_access = Access::read_write;
    }
    return vector_effects(word, Operation::simd_data, destination_bits, destination_access,
                          {source_bits});
}

// ---------------------------------------------------------------------------
// Vector modified immediate (bit24 = 1 frame, bits 23..18 = 000000).

RegisterEffects simd_modified_immediate(std::uint32_t word) noexcept {
    // Advanced SIMD modified immediate: 0 Q op 0111100000 a b c cmode o2 1 defgh Rd.
    // Rows follow the A64 table: op/cmode select MOVI/MVNI/ORR/BIC/FMOV; o2 = 1 only for
    // FMOV (half precision, cmode 1111, op 0).
    if (field(word, 19, 4) != 0 || !bit(word, 10)) {
        return reserved(word);
    }
    const bool q = q_bit(word);
    const bool op = u_bit(word);
    const std::uint32_t cmode = field(word, 12, 4);
    const bool o2 = bit(word, 11);
    if (o2) {
        if (op || cmode != 0xF) {
            return reserved(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q), Access::write, {});
    }
    // ORR/BIC (immediate) read the destination: cmode 0xx1 and 10x1.
    const bool logical = (cmode & 1u) != 0 && cmode < 0xC;
    if (op && cmode == 0xF && !q) {
        return reserved(word);  // FMOV double needs Q = 1
    }
    // MOVI 64-bit (op 1, cmode 1110) with Q = 0 writes the scalar D register.
    const unsigned bits = (op && cmode == 0xE && !q) ? 64u : register_bits(q);
    return vector_effects(word, Operation::simd_data, bits,
                          logical ? Access::read_write : Access::write, {});
}

// ---------------------------------------------------------------------------
// Copy element, table lookup, EXT, permute and dot product (bit21 = 0), plus
// the FP16 three-same rows in the half-precision sizes.

RegisterEffects simd_copy_table_extract(std::uint32_t word) noexcept {
    const std::uint32_t size = size_field(word);
    const bool u = u_bit(word);
    const std::uint32_t opcode = opcode6(word);

    // SDOT/UDOT (vector): size 10 only; the dot product accumulates into Vd.
    if (size == 2 && opcode == 0x25 && bit(word, 10)) {
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              Access::read_write, {register_bits(q_bit(word)),
                                                   register_bits(q_bit(word))});
    }
    if ((size == 1 || size == 3) && bit(word, 10)) {
        // FP16 three same: size selects an opcode column; only the rows
        // verified against the encoding tables decode.
        const std::uint32_t row_opcode = opcode5(word);
        bool accumulate = false;
        bool allocated = false;
        if (!u && size == 1) {
            allocated = row_opcode == 0x01 || row_opcode == 0x02 || row_opcode == 0x04 ||
                        row_opcode == 0x07;
            accumulate = row_opcode == 0x01;  // FMLA
        } else if (u && size == 1) {
            allocated = row_opcode == 0x02 || row_opcode == 0x07;  // FADDP, FDIV
        } else if (!u && size == 3) {
            allocated = row_opcode == 0x01 || row_opcode == 0x07;  // FMLS, FRSQRTS
            accumulate = row_opcode == 0x01;
        } else {
            allocated = row_opcode == 0x02;  // FABD
        }
        if (!allocated) {
            return unknown(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              accumulate ? Access::read_write : Access::write,
                              {register_bits(q_bit(word)), register_bits(q_bit(word))});
    }

    if (bit(word, 10)) {
        // DUP element (U = 0) and INS element (U = 1, Q = 1). Bits 14:11
        // are the second lane index for INS and zero for DUP. The source
        // lane lies in the upper half exactly when index + 1 exceeds the
        // low doubleword.
        if (bit(word, 15)) {
            return reserved(word);
        }
        // imm5 = (index << (size code + 1)) | (1 << size code): the lowest
        // set bit selects the element size and the bits above it the lane.
        // The lane lies in the upper half exactly when it reaches past the
        // low doubleword.
        const std::uint32_t imm5 = field(word, 16, 5);
        if (imm5 == 0) {
            return reserved(word);
        }
        const unsigned size_code = static_cast<unsigned>(__builtin_ctz(imm5));
        const unsigned element_bits = 8u << size_code;
        const auto lane_bits = [](unsigned index, unsigned bits) {
            return (index + 1u) * bits > 64u ? 128u : 64u;
        };
        if (!u) {
            if (field(word, 11, 4) != 0) {
                return reserved(word);
            }
            const unsigned destination_index = imm5 >> (size_code + 1u);
            return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                                  Access::write, {lane_bits(destination_index, element_bits)});
        }
        if (!q_bit(word)) {
            return reserved(word);
        }
        // The source index is imm4 shifted by the size code; the low bits
        // are should-be-zero.
        const std::uint32_t source_imm4 = field(word, 11, 4);
        if ((source_imm4 & ((1u << size_code) - 1u)) != 0) {
            return reserved(word);
        }
        const unsigned source_index = source_imm4 >> size_code;
        return vector_effects(word, Operation::simd_data, 128, Access::read_write,
                              {lane_bits(source_index, element_bits)});
    }
    if (bit(word, 15)) {
        return reserved(word);
    }
    if (u) {
        // EXT: the immediate spans bits 14:11; .8B requires imm4<3> = 0.
        if (bit(word, 15)) {
            return reserved(word);
        }
        if (!q_bit(word) && bit(word, 14)) {
            return reserved(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              Access::write, {register_bits(q_bit(word)),
                                              register_bits(q_bit(word))});
    }
    if (bit(word, 15)) {
        return reserved(word);
    }
    if (!bit(word, 12) && !bit(word, 11) && size == 0) {
        // TBL: the register count is at bits 14:13 (rows 0, 2, 4, 6).
        return table_lookup(word, false);
    }
    if (bit(word, 12) && !bit(word, 14) && !bit(word, 13) && !bit(word, 11) && size == 0) {
        return table_lookup(word, true);  // TBX
    }
    switch (opcode5(word)) {
    case 0x03: case 0x05: case 0x07: case 0x0B: case 0x0D: case 0x0F:
        // UZP1/TRN1/ZIP1 and UZP2/TRN2/ZIP2; .1D needs Q = 1.
        if (size_field(word) == 3 && !q_bit(word)) {
            return reserved(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              Access::write, {register_bits(q_bit(word)),
                                              register_bits(q_bit(word))});
    default:
        return reserved(word);
    }
}

// ---------------------------------------------------------------------------
// Three-register rows with bit21 = 1: integer three same and floating-point
// three same (bit10 = 1), the rm-tagged tables (bit10 = 0), and three
// different (bit10 = 0 with even opcodes).

RegisterEffects simd_three_same(std::uint32_t word) noexcept {
    const bool u = u_bit(word);
    const std::uint32_t size = size_field(word);
    const std::uint32_t opcode = opcode5(word);
    if (size == 3 && !q_bit(word)) {
        return reserved(word);  // .1D arrangements do not exist
    }
    if (opcode >= 0x18) {
        // Floating-point three same: bit22 selects single/double and
        // bit23 is an opcode extension bit.
        if (bit(word, 23) == 0 && size > 1) {
            return reserved(word);  // bit23 = 0 integer rows without doubleword
        }
        if (bit(word, 23) == 1 && size == 3 && !q_bit(word)) {
            return reserved(word);
        }
        const std::uint32_t key =
            (static_cast<std::uint32_t>(u) << 6) | (static_cast<std::uint32_t>(bit(word, 23)) << 5) |
            opcode;
        bool accumulate = false;
        bool allocated = false;
        switch (key) {
        case 0x19: allocated = true; accumulate = true; break;  // FMLA
        case 0x39: allocated = true; accumulate = true; break;  // FMLS
        case 0x1A: allocated = true; break;                     // FADD
        case 0x3A: allocated = true; break;                     // FSUB
        case 0x7A: allocated = true; break;                     // FABD
        case 0x5A: allocated = true; break;                     // FADDP
        case 0x1B: allocated = true; break;                     // FMULX
        case 0x5B: allocated = true; break;                     // FMUL
        case 0x5F: allocated = true; break;                     // FDIV
        case 0x1F: allocated = true; break;                     // FRECPS
        case 0x3F: allocated = true; break;                     // FRSQRTS
        case 0x1C: allocated = true; break;                     // FCMEQ
        case 0x5C: allocated = true; break;                     // FCMGE
        case 0x7C: allocated = true; break;                     // FCMGT
        case 0x5D: allocated = true; break;                     // FACGE
        case 0x7D: allocated = true; break;                     // FACGT
        case 0x1E: allocated = true; break;                     // FMAX
        case 0x3E: allocated = true; break;                     // FMIN
        case 0x5E: allocated = true; break;                     // FMAXP
        case 0x7E: allocated = true; break;                     // FMINP
        case 0x18: allocated = true; break;                     // FMAXNM
        case 0x38: allocated = true; break;                     // FMINNM
        case 0x58: allocated = true; break;                     // FMAXNMP
        case 0x78: allocated = true; break;                     // FMINNMP
        default:
            break;
        }
        if (!allocated) {
            return reserved(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              accumulate ? Access::read_write : Access::write,
                              {register_bits(q_bit(word)), register_bits(q_bit(word))});
    }
    if (opcode == 0x03) {
        // Bitwise rows: the size bits select AND/BIC/ORR/ORN (U = 0) and
        // EOR/BSL/BIT/BIF (U = 1); BSL, BIT and BIF read Vd.
        const bool keep = u && size != 0;
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              keep ? Access::read_write : Access::write,
                              {register_bits(q_bit(word)), register_bits(q_bit(word))});
    }
    // Integer three same. The size field selects byte, halfword, word and
    // (where allowed) doubleword arrangements.
    struct Row {
        std::uint32_t opcode;
        unsigned sizes;   // bitmask of allocated size fields
        unsigned u_mask;  // bit 0: U = 0 allocated, bit 1: U = 1 allocated
        bool accumulate;
    };
    static constexpr Row rows[] = {
        {0x00, 0x7, 0x3, false},  // SHADD/UHADD
        {0x01, 0xF, 0x3, false},  // SQADD/UQADD
        {0x02, 0x7, 0x3, false},  // SRHADD/URHADD
        {0x04, 0x7, 0x3, false},  // SHSUB/UHSUB
        {0x05, 0xF, 0x3, false},  // SQSUB/UQSUB
        {0x06, 0xF, 0x3, false},  // CMGT/CMHI
        {0x07, 0xF, 0x3, false},  // CMGE/CMHS
        {0x08, 0xF, 0x3, false},  // SSHL/USHL
        {0x09, 0xF, 0x3, false},  // SQSHL/UQSHL (register)
        {0x0A, 0xF, 0x3, false},  // SRSHL/URSHL (register)
        {0x0B, 0xF, 0x3, false},  // SQRSHL/URQSHL (register)
        {0x0C, 0x7, 0x3, false},  // SMAX/UMAX
        {0x0D, 0x7, 0x3, false},  // SMIN/UMIN
        {0x0E, 0x7, 0x3, false},  // SABD/UABD
        {0x0F, 0x7, 0x3, true},   // SABA/UABA
        {0x10, 0xF, 0x3, false},  // ADD/SUB
        {0x11, 0xF, 0x3, false},  // CMTST (U = 0) / CMEQ (U = 1)
        {0x12, 0x7, 0x3, true},   // MLA/MLS
        {0x13, 0x7, 0x1, false},  // MUL (U = 0)
        {0x13, 0x1, 0x2, false},  // PMUL (U = 1, bytes only)
        {0x14, 0x7, 0x3, false},  // SMAXP/UMAXP
        {0x15, 0x7, 0x3, false},  // SMINP/UMINP
        {0x16, 0x6, 0x3, false},  // SQDMULH/SQRDMULH
        {0x17, 0xF, 0x1, false},  // ADDP (U = 0)
    };
    for (const auto& row : rows) {
        if (row.opcode != opcode || (row.u_mask & (u + 1u)) == 0) {
            continue;
        }
        if ((row.sizes & (1u << size)) == 0) {
            break;
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              row.accumulate ? Access::read_write : Access::write,
                              {register_bits(q_bit(word)), register_bits(q_bit(word))});
    }
    return reserved(word);
}

// Three different: long, wide, narrowing and accumulate-long operations.
// Wide sources mix views (Vn full width, Vm half width), so this group
// builds its operands directly.
RegisterEffects simd_three_different(std::uint32_t word) noexcept {
    const bool u = u_bit(word);
    const std::uint32_t size = size_field(word);
    const std::uint32_t opcode = opcode5(word);
    if (size == 3) {
        if (opcode != 0x1C || u || q_bit(word)) {
            return reserved(word);  // PMULL with 64-bit elements is crypto-only
        }
        return vector_effects(word, Operation::simd_data, 128, Access::write, {64, 64});
    }
    enum class Shape { long_, narrow, long_accumulate, wide };
    Shape shape = Shape::long_;
    bool allocated = false;
    switch (opcode) {
    case 0x00: shape = Shape::long_; allocated = true; break;            // SADDL/UADDL
    case 0x02: shape = Shape::wide; allocated = true; break;             // SADDW/UADDW
    case 0x04: shape = Shape::long_; allocated = true; break;            // SSUBL/USUBL
    case 0x06: shape = Shape::wide; allocated = true; break;             // SSUBW/USUBW
    case 0x08: shape = Shape::narrow; allocated = true; break;           // ADDHN/RADDHN
    case 0x0A: shape = Shape::long_accumulate; allocated = true; break;  // SABAL/UABAL
    case 0x0C: shape = Shape::narrow; allocated = true; break;           // SUBHN/RSUBHN
    case 0x0E: shape = Shape::long_; allocated = true; break;            // SABDL/UABDL
    case 0x10: shape = Shape::long_accumulate; allocated = true; break;  // SMLAL/UMLAL
    case 0x14: shape = Shape::long_accumulate; allocated = true; break;  // SMLSL/UMLSL
    case 0x18: shape = Shape::long_; allocated = true; break;            // SMULL/UMULL
    case 0x1A: shape = Shape::long_; allocated = true; break;            // SQDMULL (U = 0)
    case 0x1C: shape = Shape::long_; allocated = true; break;            // PMULL (bytes)
    default:
        break;
    }
    if (!allocated) {
        return reserved(word);
    }
    if (opcode == 0x1A && u) {
        return reserved(word);
    }
    unsigned destination_bits = register_bits(q_bit(word));
    unsigned source_bits = destination_bits;
    Access destination_access = Access::write;
    switch (shape) {
    case Shape::long_:
        destination_bits = 128;
        source_bits = 64;
        break;
    case Shape::long_accumulate:
        destination_bits = 128;
        source_bits = 64;
        destination_access = Access::read_write;
        break;
    case Shape::narrow:
        destination_bits = 64;
        source_bits = 128;
        break;
    case Shape::wide:
        destination_bits = 128;
        source_bits = 128;
        break;
    }
    Builder builder(word);
    builder.operation(Operation::simd_data)
        .vector(rd_field, destination_bits, destination_access, OperandRole::destination);
    const unsigned second_source = shape == Shape::wide ? 64u : source_bits;
    builder.vector(rn_field, source_bits, Access::read, OperandRole::source)
        .vector(rm_field, second_source, Access::read, OperandRole::source);
    return builder.decoded();
}

// One- and two-register rows with bit21 = 1 and bit10 = 0, keyed by the
// bits-20:16 tag: 00000 misc and zero compares, 00001 narrow/long and FP
// conversions, 01000 AES, 10000/10001 across lanes, 11000/11001 FP16 misc.
RegisterEffects simd_tagged_register(std::uint32_t word) noexcept {
    const bool u = u_bit(word);
    const std::uint32_t tag = rm_value(word);
    const std::uint32_t opcode = opcode6(word);
    const std::uint32_t size = size_field(word);

    // Two-register misc and zero compares (tag 00000).
    if (tag == 0x00) {
        switch (opcode) {
        case 0x02:  // REV64 (U = 0) / REV32 (U = 1)
            break;
        case 0x06:  // REV16 (U = 0)
            if (u) return reserved(word);
            break;
        case 0x0A:  // SADDLP/UADDLP
            break;
        case 0x1A:  // SADALP/UADALP accumulate
            return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                                  Access::read_write, {register_bits(q_bit(word))});
        case 0x0E:  // SUQADD/USQADD accumulate into Vd
            return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                                  Access::read_write, {register_bits(q_bit(word))});
        case 0x12:  // CLS/CLZ
            break;
        case 0x16:  // CNT (U = 0), NOT and RBIT (U = 1), all byte elements
            if (size != 0) return reserved(word);
            break;
        case 0x1E:  // SQABS/SQNEG
            break;
        case 0x22:  // CMGT/CMGE (zero)
        case 0x26:  // CMEQ/CMLE (zero)
            break;
        case 0x2A:  // CMLT (zero, U = 0)
            if (u) return reserved(word);
            break;
        case 0x2E:  // ABS (U = 0) / NEG (U = 1)
            break;
        case 0x32:  // FCMGT/FCMGE (zero), floating point
        case 0x36:  // FCMEQ/FCMLE (zero)
        case 0x3A:  // FCMLT (zero, U = 0)
        case 0x3E:  // FABS/FNEG
            if (bit(word, 23) == 0) return reserved(word);
            break;
        default:
            return reserved(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              Access::write, {register_bits(q_bit(word))});
    }

    // Narrow, long and floating-point conversions (tag 00001).
    if (tag == 0x01) {
        switch (opcode) {
        case 0x0A:  // XTN (U = 0) / SQXTUN (U = 1)
            return vector_effects(word, Operation::simd_data, 64, Access::write, {128});
        case 0x0E:  // SHLL/SHLL2 (U = 1)
            if (!u) return reserved(word);
            return vector_effects(word, Operation::simd_data, 128, Access::write, {64});
        case 0x12:  // SQXTN (U = 0) / UQXTN (U = 1)
            return vector_effects(word, Operation::simd_data, 64, Access::write, {128});
        case 0x1A:  // FCVTN/FCVTXN: narrow a doubleword vector
            return vector_effects(word, Operation::simd_data, 64, Access::write, {128});
        case 0x1E:  // FCVTL: widen
            return vector_effects(word, Operation::simd_data, 128, Access::write, {64});
        case 0x22:  // FRINTN/FRINTP/FRINTA
        case 0x26:  // FRINTI/FRINTM/FRINTX/FRINTZ
        case 0x2A:  // FCVTNS/NU/PS/PU
        case 0x2E:  // FCVTMS/MU/ZS/ZU
        case 0x36:  // SCVTF/UCVTF (bit23 = 0) or FRECPE/FRSQRTE (bit23 = 1)
            break;
        case 0x32:  // FCVTAS/FCVTAU (bit23 = 0) or extension rows (bit23 = 1)
            if (bit(word, 23)) return unknown(word);
            break;
        case 0x3E:  // FSQRT (U = 1, floating point)
            if (!u || bit(word, 23) == 0) return reserved(word);
            break;
        default:
            return reserved(word);
        }
        // Floating-point conversion rows use bit23 as an opcode bit: 0 for
        // the whole register set (any size), 1 for FRECPE/FRSQRTE/FSQRT
        // (half/word/double elements only).
        if (bit(word, 23) && (opcode == 0x36 || opcode == 0x3E) && size < 2) {
            return reserved(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              Access::write, {register_bits(q_bit(word))});
    }

    // AES rounds (tag 01000, U = 0, Q = 1, byte arrangement).
    if (tag == 0x08) {
        if (u || !q_bit(word) || size != 0) {
            return reserved(word);
        }
        switch (opcode) {
        case 0x12: case 0x16: case 0x1A: case 0x1E:
            return vector_effects(word, Operation::simd_data, 128, Access::write, {128});
        case 0x02: case 0x06: case 0x0A:
            // SHA1H (000010), SHA1SU1 (000110), SHA256SU0 (001010).
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, 32, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, 32, Access::read,
                                                 OperandRole::source)});
        default:
            return reserved(word);
        }
    }

    // Across-lane reductions (tags 10000/10001).
    if (tag == 0x10 || tag == 0x11) {
        bool allocated = false;
        bool floating = false;
        if (tag == 0x10) {
            switch (opcode) {
            case 0x0E: allocated = true; break;  // SADDLV/UADDLV
            case 0x2A: allocated = true; break;  // SMAXV/UMAXV
            case 0x32:                            // FMAXNMV/FMINNMV (U = 0)
                allocated = !u;
                floating = true;
                break;
            case 0x3E:                            // FMAXV/FMINV (U = 1)
                allocated = u;
                floating = true;
                break;
            default:
                break;
            }
        } else {
            switch (opcode) {
            case 0x2A: allocated = true; break;  // SMINV/UMINV
            case 0x2E:                           // ADDV (U = 0)
                allocated = !u;
                break;
            default:
                break;
            }
        }
        if (!allocated) {
            return reserved(word);
        }
        if (floating) {
            // The result is single for Q = 1 and half for Q = 0.
            return vector_effects(word, Operation::simd_data, q_bit(word) ? 32u : 16u,
                                  Access::write, {register_bits(q_bit(word))});
        }
        if (!q_bit(word) && size > 1) {
            return reserved(word);
        }
        if (q_bit(word) && size > 2) {
            return reserved(word);
        }
        return vector_effects(word, Operation::simd_data, 8u << size, Access::write,
                              {register_bits(q_bit(word))});
    }

    // FP16 two-register misc and zero compares (tags 11000/11001). Only the
    // rows verified against the FP16 encoding tables decode; the sparse
    // remainder of the frame stays unknown.
    if (tag == 0x18 || tag == 0x19) {
        if (size != 1 && size != 3) {
            return unknown(word);
        }
        bool allocated = false;
        bool accumulate = false;
        switch (opcode | (static_cast<std::uint32_t>(tag) << 8)) {
        case 0x18 << 8 | 0x32:  // FCMGT (U = 0) / FCMGE (U = 1) zero, .8H only
        case 0x18 << 8 | 0x36:  // FCMEQ (U = 0) / FCMLE (U = 1) zero
        case 0x18 << 8 | 0x3A:  // FCMLT (zero, U = 0)
        case 0x18 << 8 | 0x3E:  // FABS (U = 0) / FNEG (U = 1)
            allocated = size == 3;
            break;
        case 0x18 << 8 | 0x22:  // rows not present in the FP16 tables
        case 0x18 << 8 | 0x26:
        case 0x18 << 8 | 0x2A:
        case 0x18 << 8 | 0x2E:
            return unknown(word);
        case 0x19 << 8 | 0x22:  // FRINTN/FRINTP/FRINTA
        case 0x19 << 8 | 0x26:  // FRINTI/FRINTM/FRINTX/FRINTZ
        case 0x19 << 8 | 0x2A:  // FCVTNS/NU/PS/PU
        case 0x19 << 8 | 0x2E:  // FCVTMS/MU/ZS/ZU
        case 0x19 << 8 | 0x32:  // FCVTAS/AU
        case 0x19 << 8 | 0x36:  // SCVTF/UCVTF (bit23 = 0), FRECPE/FRSQRTE (bit23 = 1)
            allocated = true;
            break;
        case 0x19 << 8 | 0x3E:  // FSQRT (U = 1, .8H only)
            allocated = u && size == 3;
            break;
        default:
            return unknown(word);
        }
        if (bit(word, 23) && opcode == 0x36 && size < 2) {
            return reserved(word);  // FRECPE/FRSQRTE have no half-element rows
        }
        if (!allocated) {
            return unknown(word);
        }
        return vector_effects(word, Operation::simd_data, register_bits(q_bit(word)),
                              accumulate ? Access::read_write : Access::write,
                              {register_bits(q_bit(word))});
    }
    return reserved(word);
}

// Only the general-register variants of the AdvSIMD copy class are covered.
RegisterEffects simd_general_copy(std::uint32_t word) noexcept {
    const unsigned opcode = field(word, 11, 4);
    if (opcode != 1 && opcode != 3 && opcode != 5 && opcode != 7) return unknown(word);
    const bool q = bit(word, 30);
    const unsigned imm5 = field(word, 16, 5);
    if ((imm5 & 15u) == 0) return reserved(word);
    unsigned size = 0;
    while ((imm5 & (1u << size)) == 0) ++size;
    const unsigned element_bits = 8u << size;
    Builder builder(word);
    if (opcode == 1 || opcode == 3) {
        if ((!q && size == 3) || (opcode == 3 && !q)) return reserved(word);
        builder.operation(opcode == 1 ? Operation::dup_general : Operation::ins_general)
            .general(rn_field, zr, size == 3 ? 64 : 32, Access::read, OperandRole::source)
            .vector(rd_field, q ? 128 : 64, opcode == 3 ? Access::read_write : Access::write,
                    OperandRole::destination);
    } else {
        const unsigned bits = q ? 64u : 32u;
        if (opcode == 5 ? element_bits >= bits : (q ? size != 3 : size == 3)) {
            return reserved(word);
        }
        builder.operation(opcode == 5 ? Operation::smov : Operation::umov)
            .vector(rn_field, bit(word, 20) ? 128 : 64, Access::read, OperandRole::source)
            .general(rd_field, zr, bits, Access::write, OperandRole::destination);
    }
    return builder.decoded();
}

// Scalar conversions with an integer register, including fixed-point scale,
// and FMOV between the register files. Other FP and extension forms stay unknown.
RegisterEffects scalar_fp_general(std::uint32_t word) noexcept {
    const bool fixed = !bit(word, 21);
    const unsigned op = field(word, 16, 5);
    const unsigned fp_type = field(word, 22, 2);
    const unsigned bits = width(word);
    const bool to_fp = op == 2 || op == 3;
    const bool move = !fixed && (op == 6 || op == 7 || op == 14 || op == 15);
    if (fixed) {
        if (!to_fp && op != 24 && op != 25) return unknown(word);
        if (bits == 32 && field(word, 10, 6) < 32) return reserved(word);
    } else {
        if (field(word, 10, 6) != 0) return unknown(word);
        if (!move && !to_fp && op != 0 && op != 1 && op != 4 && op != 5 &&
            op != 8 && op != 9 && op != 16 && op != 17 && op != 24 && op != 25) {
            return unknown(word);
        }
    }
    unsigned fp_bits = fp_type == 0 ? 32u : fp_type == 1 ? 64u : fp_type == 3 ? 16u : 0u;
    bool top_half = false;
    if (move) {
        top_half = op == 14 || op == 15;
        if (top_half) {
            if (bits != 64 || fp_type != 2) return reserved(word);
            fp_bits = 128;
        } else if (fp_bits == 0 || (fp_bits != 16 && fp_bits != bits)) {
            return reserved(word);
        }
    } else if (fp_bits == 0) {
        return reserved(word);
    }
    const bool input_general = move ? (op & 1u) != 0 : to_fp;
    Builder builder(word);
    builder.operation(move ? Operation::fmov_general : to_fp
        ? (op == 2 ? Operation::scvtf_general : Operation::ucvtf_general) : Operation::fcvt_general);
    if (input_general) {
        builder.general(rn_field, zr, bits, Access::read, OperandRole::source)
            .vector(rd_field, fp_bits, top_half ? Access::read_write : Access::write,
                    OperandRole::destination);
    } else {
        builder.vector(rn_field, fp_bits, Access::read, OperandRole::source)
            .general(rd_field, zr, bits, Access::write, OperandRole::destination);
    }
    return builder.decoded();
}

// ---------------------------------------------------------------------------
// Scalar floating-point three-source (0x1F/0x9F) and scalar shift by
// immediate (0x3F/0x5F/0x7F with bit30 = 1) share the bit24 = 1 frame.

RegisterEffects fp_three_source(std::uint32_t word) noexcept {
    const std::uint32_t type = size_field(word);
    // The type field holds 00 (single), 01 (double) and 11 (half); 10 is
    // unallocated.
    const unsigned bits = type == 0 ? 32u : type == 1 ? 64u : type == 3 ? 16u : 0u;
    if (bits == 0) {
        return reserved(word);
    }
    Builder builder(word);
    builder.operation(Operation::fp_data)
        .vector(rd_field, bits, Access::write, OperandRole::destination)
        .vector(rn_field, bits, Access::read, OperandRole::source)
        .vector(rm_field, bits, Access::read, OperandRole::source)
        .vector(ra_field, bits, Access::read, OperandRole::accumulator);
    return builder.decoded();
}

RegisterEffects simd_scalar_shift(std::uint32_t word) noexcept {
    const unsigned immh = field(word, 19, 4);
    if (bit(word, 31) || immh == 0) {
        return reserved(word);
    }
    const std::uint32_t opcode = opcode6(word);
    if (!bit(word, 10)) {
        return reserved(word);
    }
    const bool u = u_bit(word);
    bool allocated = false;
    bool accumulate = false;
    bool narrow = false;
    switch (opcode) {
    case 0x01: allocated = true; break;                   // SSHR/USHR
    case 0x05: allocated = true; accumulate = true; break;  // SSRA/USRA
    case 0x09: allocated = true; break;                   // SRSHR/URSHR
    case 0x0D: allocated = true; accumulate = true; break;  // SRSRA/URSRA
    case 0x11:                                            // SRI (U = 1)
        allocated = u;
        accumulate = true;
        break;
    case 0x15:                                            // SLI (U = 1)
        allocated = u;
        accumulate = true;
        break;
    case 0x1D: allocated = true; break;                   // SQSHL/SQSHLU
    case 0x21: case 0x23:                                 // SQSHRUN/SQRSHRUN (U = 1)
        allocated = u;
        narrow = true;
        break;
    case 0x25: case 0x27:                                 // SQSHRN..UQRSHRN
        allocated = true;
        narrow = true;
        break;
    case 0x39: allocated = true; break;                   // SCVTF/UCVTF fixed
    case 0x3F: allocated = true; break;                   // FCVTZS/FCVTZU fixed
    default:
        break;
    }
    if (!allocated) {
        return reserved(word);
    }
    if (narrow && immh < 2) {
        return reserved(word);
    }
    if ((opcode == 0x39 || opcode == 0x3F) && immh < 2) {
        return reserved(word);
    }
    unsigned element = 8u;
    for (unsigned mask = immh; mask > 1u; mask >>= 1u) {
        element <<= 1u;
    }
    return scalar_effects(word, Operation::simd_scalar_data,
                          {scalar_vector(word, rd_field, element,
                                         accumulate ? Access::read_write : Access::write,
                                         OperandRole::destination),
                           scalar_vector(word, rn_field, element, Access::read, OperandRole::source)});
}

// ---------------------------------------------------------------------------
// The bit30 = 1 scalar rows (0x5E/0x7E): scalar integer three same, scalar
// two-register misc, scalar narrow/long, scalar pairwise add, scalar FP
// zero compares, and the reciprocal estimate family.

// Advanced SIMD scalar copy: DUP (element) to a scalar, 01 0 11110000 imm5 0 0000 1 Rn Rd.
RegisterEffects simd_scalar_copy(std::uint32_t word) noexcept {
    if (field(word, 11, 4) != 0) {
        return reserved(word);
    }
    const unsigned imm5 = field(word, 16, 5);
    if ((imm5 & 0xFu) == 0) {
        return reserved(word);
    }
    unsigned size = 0;
    while ((imm5 & (1u << size)) == 0) ++size;
    return scalar_effects(word, Operation::simd_scalar_data,
                          {scalar_vector(word, rd_field, 8u << size, Access::write,
                                         OperandRole::destination),
                           scalar_vector(word, rn_field, 128, Access::read, OperandRole::source)});
}

// Advanced SIMD scalar x indexed element: 01 U 11111 size L M Rm opcode H 0 Rn Rd.
RegisterEffects simd_scalar_indexed(std::uint32_t word) noexcept {
    const unsigned u = u_bit(word) ? 1u : 0u;
    const std::uint32_t size = field(word, 22, 2);
    const std::uint32_t opcode = field(word, 12, 4);
    enum class Kind { fp, fp_accumulate, long_int, long_accumulate, integer, int_accumulate };
    struct Row { unsigned u; std::uint32_t opcode; Kind kind; };
    static constexpr Row rows[] = {
        {0, 0x1, Kind::fp_accumulate},   // FMLA
        {0, 0x3, Kind::long_accumulate}, // SQDMLAL
        {0, 0x5, Kind::fp_accumulate},   // FMLS
        {0, 0x7, Kind::long_accumulate}, // SQDMLSL
        {0, 0x9, Kind::fp},              // FMUL
        {0, 0xB, Kind::long_int},        // SQDMULL
        {0, 0xC, Kind::integer},         // SQDMULH
        {0, 0xD, Kind::integer},         // SQRDMULH
        {1, 0x9, Kind::fp},              // FMULX
        {1, 0xD, Kind::int_accumulate},  // SQRDMLAH
        {1, 0xF, Kind::int_accumulate},  // SQRDMLSH
    };
    const Row* matched = nullptr;
    for (const auto& row : rows) {
        if (row.u == u && row.opcode == opcode) { matched = &row; break; }
    }
    if (matched == nullptr) {
        return unknown(word);
    }
    const bool floating = matched->kind == Kind::fp || matched->kind == Kind::fp_accumulate;
    unsigned element = 0;
    if (floating) {
        if (size == 1 || (size == 3 && bit(word, 21))) return reserved(word);
        element = size == 0 ? 16u : size == 2 ? 32u : 64u;
    } else {
        if (size != 1 && size != 2) return reserved(word);
        element = size == 1 ? 16u : 32u;
    }
    const bool long_result = matched->kind == Kind::long_int || matched->kind == Kind::long_accumulate;
    const bool accumulates = matched->kind == Kind::fp_accumulate ||
                             matched->kind == Kind::long_accumulate ||
                             matched->kind == Kind::int_accumulate;
    return scalar_effects(word, Operation::simd_scalar_data,
                          {scalar_vector(word, rd_field, long_result ? element * 2 : element,
                                         accumulates ? Access::read_write : Access::write,
                                         OperandRole::destination),
                           scalar_vector(word, rn_field, element, Access::read, OperandRole::source),
                           scalar_vector(word, rm_field, 128, Access::read, OperandRole::source)});
}

// Scalar two-register misc FP rows (bits 21:17 = 10000, bits 11:10 = 10): conversions and
// reciprocal estimates. Returns unknown for rows handled elsewhere.
RegisterEffects simd_scalar_fp_misc(std::uint32_t word) noexcept {
    const unsigned u = u_bit(word) ? 1u : 0u;
    const bool high = bit(word, 23);
    const std::uint32_t opcode = field(word, 12, 5);
    bool allocated = false;
    if (!high) {
        allocated = opcode == 0x1A || opcode == 0x1B || opcode == 0x1C ||  // FCVTN/M/A S|U
                    opcode == 0x1D;                                        // SCVTF / UCVTF
    } else {
        allocated = opcode == 0x1A || opcode == 0x1B ||                    // FCVTP/Z S|U
                    opcode == 0x1D ||                                      // FRECPE / FRSQRTE
                    (opcode == 0x1F && u == 0);                            // FRECPX
    }
    if (!allocated) {
        return unknown(word);
    }
    const unsigned element = bit(word, 22) ? 64u : 32u;
    return scalar_effects(word, Operation::simd_scalar_data,
                          {scalar_vector(word, rd_field, element, Access::write,
                                         OperandRole::destination),
                           scalar_vector(word, rn_field, element, Access::read, OperandRole::source)});
}

RegisterEffects scalar_p1(std::uint32_t word) noexcept {
    if (bit(word, 31)) {
        return reserved(word);
    }
    const bool u = u_bit(word);
    const std::uint32_t size = size_field(word);
    const std::uint32_t tag = rm_value(word);
    const std::uint32_t opcode = opcode6(word);

    if ((word & 0xDFE08400u) == 0x5E000400u) {
        return simd_scalar_copy(word);
    }
    if (field(word, 17, 5) == 0x10 && field(word, 10, 2) == 2) {
        const auto fp_misc = simd_scalar_fp_misc(word);
        if (fp_misc.status != EffectStatus::unknown) {
            return fp_misc;
        }
    }

    // Scalar integer three same and FABD/FMULX stay separate: only the
    // verified integer rows decode here.
    if (bit(word, 10)) {
        if (!bit(word, 21)) {
            return reserved(word);
        }
        const std::uint32_t row_opcode = opcode5(word);
        if (row_opcode >= 0x18) {
            // Scalar FP three same (single/double; sz = bit 22): U and size<1> (bit 23)
            // select the row, per the A64 "Advanced SIMD scalar three same" table. FP16 forms
            // use a separate encoding (bit 21 = 0) and do not reach this path.
            const unsigned high = bit(word, 23) ? 1u : 0u;
            static constexpr struct { unsigned u, high; std::uint32_t opcode; } fp_rows[] = {
                {0, 0, 0x1B}, {0, 0, 0x1C}, {0, 0, 0x1F},  // FMULX, FCMEQ, FRECPS
                {0, 1, 0x1F},                              // FRSQRTS
                {1, 0, 0x1C}, {1, 0, 0x1D},                // FCMGE, FACGE
                {1, 1, 0x1A}, {1, 1, 0x1C}, {1, 1, 0x1D},  // FABD, FCMGT, FACGT
            };
            bool allocated = false;
            for (const auto& row : fp_rows) {
                allocated = allocated || (row.u == (u ? 1u : 0u) && row.high == high &&
                                          row.opcode == row_opcode);
            }
            if (!allocated) {
                return reserved(word);
            }
            const unsigned fp_bits = bit(word, 22) ? 64u : 32u;
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, fp_bits, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, fp_bits, Access::read,
                                                 OperandRole::source),
                                   scalar_vector(word, rm_field, fp_bits, Access::read,
                                                 OperandRole::source)});
        }
        static constexpr struct {
            std::uint32_t opcode;
            unsigned u;      // required U value
            unsigned sizes;  // allocated size bitmask
        } rows[] = {
            {0x01, 0, 0xF}, {0x01, 1, 0xF},  // SQADD/UQADD
            {0x03, 0, 0xF}, {0x03, 1, 0xF},  // SQSUB/UQSUB
            {0x06, 0, 0xF}, {0x06, 1, 0xF},  // CMGT/CMHI
            {0x07, 0, 0xF}, {0x07, 1, 0xF},  // CMGE/CMHS
            {0x08, 0, 0xF}, {0x08, 1, 0xF},  // SSHL/USHL
            {0x10, 0, 0xF}, {0x10, 1, 0xF},  // ADD/SUB
            {0x11, 0, 0xF},                  // CMEQ
            {0x16, 0, 0x6}, {0x16, 1, 0x6},  // SQDMULH/SQRDMULH
        };
        for (const auto& row : rows) {
            if (row.opcode == row_opcode && row.u == (u ? 1u : 0u)) {
                if ((row.sizes & (1u << size)) == 0) {
                    break;
                }
                return scalar_effects(word, Operation::simd_scalar_data,
                                      {scalar_vector(word, rd_field, 8u << size, Access::write,
                                                     OperandRole::destination),
                                       scalar_vector(word, rn_field, 8u << size, Access::read,
                                                     OperandRole::source),
                                       scalar_vector(word, rm_field, 8u << size, Access::read,
                                                     OperandRole::source)});
            }
        }
        return reserved(word);
    }

    if (!bit(word, 21)) {
        // Scalar floating-point zero compares (tag 0, bit23 = 1, opcodes
        // 110010/110110/111010) write an element-width compare result.
        if (tag == 0x00 && bit(word, 23) &&
            (opcode == 0x32 || opcode == 0x36 || opcode == 0x3A)) {
            if (size < 2 || (opcode == 0x3A && u)) {
                return reserved(word);
            }
            const unsigned bits = size == 2 ? 32u : 16u;
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, bits, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, bits, Access::read,
                                                 OperandRole::source)});
        }
        // Three-register cryptographic rows: SHA1C/SHA1P/SHA1M/SHA1SU0 and
        // SHA256H/SHA256H2/SHA256SU1 (U = 0, Q = 1, byte size, free Rm).
        if (u_bit(word) || !q_bit(word) || size != 0) {
            return reserved(word);
        }
        switch (opcode) {
        case 0x00: case 0x04: case 0x08: case 0x0C:
        case 0x10: case 0x14: case 0x18: {
            Builder builder(word);
            builder.operation(Operation::simd_data)
                .vector(rd_field, 128, Access::read_write, OperandRole::destination)
                .vector(rn_field, 32, Access::read, OperandRole::source)
                .vector(rm_field, 128, Access::read, OperandRole::source);
            return builder.decoded();
        }
        default:
            return reserved(word);
        }
    }

    switch (tag) {
    case 0x00: {
        // Scalar two-register misc: integer rows (bit23 = 0) and the scalar
        // FP zero compares (bit23 = 1).
        if (bit(word, 23)) {
            switch (opcode) {
            case 0x32:  // FCMGT (U = 0) / FCMGE (U = 1)
            case 0x36:  // FCMEQ (U = 0) / FCMLE (U = 1)
            case 0x3A:  // FCMLT (U = 0)
                break;
            default:
                return reserved(word);
            }
            // size 1x with sz (bit 22) selecting single or double.
            const unsigned bits = size == 2 ? 32u : size == 3 ? 64u : 0u;
            if (bits == 0 || (opcode == 0x3A && u)) {
                return reserved(word);
            }
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, bits, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, bits, Access::read,
                                                 OperandRole::source)});
        }
        switch (opcode) {
        case 0x0E:  // SUQADD (U = 0) / USQADD (U = 1) accumulate
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, 8u << size, Access::read_write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, 8u << size, Access::read,
                                                 OperandRole::source)});
        case 0x1E:  // SQABS (U = 0) / SQNEG (U = 1)
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, 8u << size, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, 8u << size, Access::read,
                                                 OperandRole::source)});
        case 0x22:  // CMGT (U = 0) / CMGE (U = 1) zero
        case 0x26:  // CMEQ (U = 0) / CMLE (U = 1) zero
        case 0x2A:  // CMLT (zero, U = 0)
        case 0x2E:  // ABS (U = 0) / NEG (U = 1)
        {
            if (opcode == 0x2A && u) {
                return reserved(word);
            }
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, 8u << size, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, 8u << size, Access::read,
                                                 OperandRole::source)});
        }
        default:
            return reserved(word);
        }
    }
    case 0x01: {
        // Scalar narrow (SQXTN/UQXTN/SQXTUN) and the reciprocal estimates.
        const unsigned source_bits = (8u << size) * 2u;
        switch (opcode) {
        case 0x0A:  // SQXTUN (U = 1)
        case 0x12:  // SQXTN (U = 0) / UQXTN (U = 1)
            if (size == 3 || (opcode == 0x0A && !u)) {
                return reserved(word);
            }
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, 8u << size, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, source_bits, Access::read,
                                                 OperandRole::source)});
        case 0x36:  // FRECPE (U = 0) / FRSQRTE (U = 1)
        {
            if (size < 2) {
                return reserved(word);
            }
            const unsigned bits = size == 2 ? 32u : 16u;
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, bits, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, bits, Access::read,
                                                 OperandRole::source)});
        }
        case 0x3E:  // FRECPX (U = 0)
        {
            if (u || size < 2) {
                return reserved(word);
            }
            const unsigned bits = size == 3 ? 64u : 16u;
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, bits, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, bits, Access::read,
                                                 OperandRole::source)});
        }
        default:
            return reserved(word);
        }
    }
    case 0x08: {
        // SHA1H, SHA1SU1 and SHA256SU0 (U = 0, halfword-size marker).
        if (u || size != 1) {
            return reserved(word);
        }
        switch (opcode) {
        case 0x02: case 0x06: case 0x0A:
            return scalar_effects(word, Operation::simd_data,
                                  {scalar_vector(word, rd_field, 32, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, 32, Access::read,
                                                 OperandRole::source)});
        default:
            return reserved(word);
        }
    }
    case 0x10: case 0x11: {
        // Scalar pairwise (bits 21:17 = 11000, bits 11:10 = 10): ADDP D, FP16 and FP
        // FADDP/FMAXP/FMINP/FMAXNMP/FMINNMP. bit 16 is the top opcode bit.
        if (field(word, 10, 2) != 2) {
            return reserved(word);
        }
        const std::uint32_t pair_opcode = field(word, 12, 5);
        if (!u && pair_opcode == 0x1B) {
            if (size != 3) {
                return reserved(word);
            }
            return scalar_effects(word, Operation::simd_scalar_data,
                                  {scalar_vector(word, rd_field, 64, Access::write,
                                                 OperandRole::destination),
                                   scalar_vector(word, rn_field, 128, Access::read,
                                                 OperandRole::source)});
        }
        const bool high = bit(word, 23);
        const bool allocated = (!high && (pair_opcode == 0x0C || pair_opcode == 0x0D ||
                                          pair_opcode == 0x0F)) ||
                               (high && (pair_opcode == 0x0C || pair_opcode == 0x0F));
        if (!allocated) {
            return reserved(word);
        }
        // U = 0 selects the FP16 forms (bit 22 must be 0); U = 1 the single/double forms.
        if (!u && bit(word, 22)) {
            return reserved(word);
        }
        const unsigned element = !u ? 16u : bit(word, 22) ? 64u : 32u;
        return scalar_effects(word, Operation::simd_scalar_data,
                              {scalar_vector(word, rd_field, element, Access::write,
                                             OperandRole::destination),
                               scalar_vector(word, rn_field, element * 2, Access::read,
                                             OperandRole::source)});
    }
    case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07:
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0F:
    case 0x12: case 0x13: case 0x14: case 0x15: case 0x16: case 0x17: case 0x18:
        return reserved(word);  // reserved tags in the scalar frame
    default: {
        // FRECPS (U = 0) and FRSQRTS (U = 0, doubleword type) with a free
        // Rm field: two-source rows at opcode 111111. No extension allocates
        // the remaining tags and opcodes in this frame.
        if (opcode != 0x3F) {
            return reserved(word);
        }
        const unsigned bits = size == 0 ? 32u : size == 1 ? 16u : size == 3 ? 64u : 0u;
        if (bits == 0) {
            return reserved(word);
        }
        return scalar_effects(word, Operation::simd_scalar_data,
                              {scalar_vector(word, rd_field, bits, Access::write,
                                             OperandRole::destination),
                               scalar_vector(word, rn_field, bits, Access::read, OperandRole::source),
                               scalar_vector(word, rm_field, bits, Access::read, OperandRole::source)});
    }
    }
}

// ---------------------------------------------------------------------------
// Scalar floating-point data processing with bit30 = 0 (0x1E/0x9E), after
// the general-register moves and conversions are handled.

RegisterEffects fp_scalar_data(std::uint32_t word) noexcept {
    if (bit(word, 31) || u_bit(word)) {
        return reserved(word);  // scalar FP arithmetic has sf = 0 and U = 0
    }
    const std::uint32_t type = size_field(word);
    const unsigned bits = type == 0 ? 32u : type == 1 ? 64u : type == 3 ? 16u : 0u;
    const std::uint32_t opcode = opcode6(word);
    const std::uint32_t tag = rm_value(word);
    // FMOV immediate: bits 12:5 = 10000000. Checked first: the immediate
    // bits overlap the compare and one-source opcode fields.
    if (bit(word, 21) && field(word, 5, 8) == 0x80u) {
        if (bits == 0) {
            return reserved(word);
        }
        return scalar_effects(word, Operation::fp_immediate,
                              {scalar_vector(word, rd_field, bits, Access::write,
                                             OperandRole::destination)});
    }
    if (!bit(word, 21)) {
        return unknown(word);  // fixed-point conversions not recognized above
    }
    if (bit(word, 10)) {
        // Bits 11:10 = x1: conditional compare (01) and conditional
        // select (11).
        switch (opcode & 3u) {
        case 1: {
            // FCCMP (bit4 = 0) / FCCMPE (bit4 = 1): condition in bits
            // 15:12, NZCV in bits 3:0, no destination register.
            if (bits == 0) {
                return reserved(word);
            }
            Builder builder(word);
            builder.operation(Operation::fp_conditional_compare)
                .vector(rn_field, bits, Access::read, OperandRole::source)
                .vector(rm_field, bits, Access::read, OperandRole::source);
            builder.raw().operands[0].conditional = true;
            builder.raw().operands[1].conditional = true;
            return builder.flags(true, true).decoded();
        }
        case 3: {
            // FCSEL: condition in bits 15:12.
            if (bits == 0) {
                return reserved(word);
            }
            Builder builder(word);
            builder.operation(Operation::fp_conditional_select)
                .vector(rd_field, bits, Access::write, OperandRole::destination)
                .vector(rn_field, bits, Access::read, OperandRole::source)
                .vector(rm_field, bits, Access::read, OperandRole::source);
            builder.raw().operands[1].conditional = true;
            builder.raw().operands[2].conditional = true;
            return builder.flags(true, false).decoded();
        }
        default:
            return reserved(word);
        }
    }
    if ((opcode & 3u) == 2) {
        // Two-source rows: FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FMAXNM,
        // FMINNM and FNMUL (opcode 1000).
        static constexpr std::uint32_t two_source[] = {0x02, 0x06, 0x0A, 0x0E,
                                                       0x12, 0x16, 0x1A, 0x1E, 0x22};
        for (const std::uint32_t candidate : two_source) {
            if (candidate == opcode) {
                if (bits == 0) {
                    return reserved(word);
                }
                return scalar_effects(word, Operation::fp_data,
                                      {scalar_vector(word, rd_field, bits, Access::write,
                                                     OperandRole::destination),
                                       scalar_vector(word, rn_field, bits, Access::read,
                                                     OperandRole::source),
                                       scalar_vector(word, rm_field, bits, Access::read,
                                                     OperandRole::source)});
            }
        }
        return reserved(word);
    }
    // Bits 11:10 = 00: compare, one-source rows, and the JSCVT slot.
    if (opcode == 0x00 && tag == 0x1E) {
        return unknown(word);  // FJCVTZS (JSCVT extension)
    }
    if (opcode == 0x08) {
        // FCMP/FCMPE: bit3 selects register or zero source, bit4 = E.
        if (bits == 0) {
            return reserved(word);
        }
        Builder builder(word);
        builder.operation(Operation::fp_compare)
            .vector(rn_field, bits, Access::read, OperandRole::source);
        if (!bit(word, 3)) {
            builder.vector(rm_field, bits, Access::read, OperandRole::source);
        } else if (tag != 0) {
            return reserved(word);
        }
        return builder.flags(false, true).decoded();
    }
    if (opcode == 0x10 || opcode == 0x30) {
        // One-source rows keyed by the low tag bits. Ftype 10 is
        // unallocated in this frame.
        if (type == 2) {
            return reserved(word);
        }
        unsigned source_bits = bits;
        unsigned destination_bits = bits;
        bool allocated = false;
        if (opcode == 0x10) {
            switch (tag) {
            case 0x00: allocated = true; break;  // FMOV
            case 0x01: allocated = true; break;  // FNEG
            case 0x02:                           // FCVT -> S
                allocated = true;
                destination_bits = 32;
                break;
            case 0x04: case 0x05: case 0x06: case 0x07:  // FRINTN/M/A/X
                allocated = true;
                break;
            default:
                break;
            }
        } else {
            switch (tag) {
            case 0x00: allocated = true; break;  // FABS
            case 0x01: allocated = true; break;  // FSQRT
            case 0x02:                           // FCVT -> D
                allocated = type == 0 || type == 3;
                destination_bits = 64;
                break;
            case 0x03:                           // FCVT -> H
                allocated = type == 0 || type == 1;
                destination_bits = 16;
                break;
            case 0x04: case 0x05: case 0x07:     // FRINTP/Z/I
                allocated = true;
                break;
            default:
                break;
            }
        }
        if (!allocated) {
            return unknown(word);
        }
        return scalar_effects(word, Operation::fp_data,
                              {scalar_vector(word, rd_field, destination_bits, Access::write,
                                             OperandRole::destination),
                               scalar_vector(word, rn_field, source_bits, Access::read,
                                             OperandRole::source)});
    }
    return reserved(word);
}

// ---------------------------------------------------------------------------
// Vector space dispatcher (bit28 = 0): shifts and modified immediates in the
// bit24 = 1 frame, then the copy/permute/table, three-register and tagged
// one/two-register groups.

RegisterEffects simd_vector_data(std::uint32_t word) noexcept {
    if (bit(word, 31)) {
        // Only the SHA3/SM3/SM4/SHA512 extensions allocate bit31 = 1 here.
        return bit(word, 30) && !u_bit(word) ? unknown(word) : reserved(word);
    }
    if (bit(word, 24)) {
        if (!bit(word, 10)) {
            // ADDP (scalar): tag 10001, opcode 101110, bits 23:21 = 111.
            if (bit(word, 23) && bit(word, 22) && bit(word, 21) &&
                field(word, 20, 5) == 0x11 && opcode6(word) == 0x2E) {
                if (u_bit(word)) {
                    return reserved(word);
                }
                return scalar_effects(word, Operation::simd_scalar_data,
                                      {scalar_vector(word, rd_field, 32, Access::write,
                                                     OperandRole::destination),
                                       scalar_vector(word, rn_field, 64, Access::read,
                                                     OperandRole::source)});
            }
            return simd_vector_indexed(word);
        }
        if (!bit(word, 23) && field(word, 19, 4) != 0) {
            return simd_shift_immediate(word);
        }
        if (!bit(word, 23) && field(word, 19, 4) == 0) {
            return simd_modified_immediate(word);
        }
        return reserved(word);
    }
    if (!bit(word, 21)) {
        return simd_copy_table_extract(word);
    }
    if (bit(word, 10)) {
        return simd_three_same(word);
    }
    if ((opcode5(word) & 1u) == 0) {
        return simd_three_different(word);
    }
    return simd_tagged_register(word);
}

} // namespace

RegisterEffects decode_register_effects(std::uint32_t word) noexcept {
    const auto authentication = pointer_authentication(word);
    if (authentication == PointerAuthentication::unsupported) return unknown(word);
    if (authentication != PointerAuthentication::none && authentication != PointerAuthentication::retab)
        return authentication_effects(word, authentication);

    const Instruction instruction = decode(word);
    if (instruction.classified()) {
        return classified(instruction);
    }
    if ((word & 0xFFFFF000u) == 0xD5032000u || (word & 0xFFFFF000u) == 0xD5033000u) {
        return hint_or_barrier(word);
    }
    if ((word & 0x1C000000u) == 0x10000000u) {
        return data_processing_immediate(word);
    }
    if ((word & 0x0E000000u) == 0x0A000000u) {
        return data_processing_register(word);
    }
    if ((word & 0x3A000000u) == 0x38000000u) {
        return load_store_single(word);
    }
    if ((word & 0x3A000000u) == 0x28000000u) {
        return load_store_pair(word);
    }
    if ((word & 0x3F000000u) == 0x08000000u) {
        return load_store_exclusive(word);
    }

    if ((word & 0xBE000000u) == 0x0C000000u) return structure_transfer(word);
    if ((word & 0x0E000000u) == 0x0E000000u) {
        if ((word & 0xFF3E0C00u) == 0x4E280800u || (word & 0xFF3E0C00u) == 0x5E280800u ||
            (word & 0xFF208C00u) == 0x5E000000u) {
            return simd_crypto(word);
        }
        // Data Processing - Scalar Floating-Point and Advanced SIMD.
        RegisterEffects general_copy{.word = word};
        if ((word & 0xBFE08400u) == 0x0E000400u) {
            general_copy = simd_general_copy(word);
            if (general_copy.status != EffectStatus::unknown) {
                return general_copy;
            }
        }
        if (!bit(word, 28)) {
            return simd_vector_data(word);
        }
        if (bit(word, 24)) {
            // Scalar floating-point three-source (bit30 = 0) and scalar
            // shift by immediate (bit30 = 1).
            if (bit(word, 30) && !bit(word, 10)) {
                return simd_scalar_indexed(word);
            }
            return bit(word, 30) ? simd_scalar_shift(word) : fp_three_source(word);
        }
        if (bit(word, 30)) {
            return scalar_p1(word);
        }
        if (u_bit(word)) {
            return reserved(word);  // scalar floating-point rows have U = 0
        }
        if (bit(word, 21) && field(word, 5, 8) == 0x80u) {
            return fp_scalar_data(word);  // FMOV immediate frame
        }
        RegisterEffects general = scalar_fp_general(word);
        if (general.status != EffectStatus::unknown) {
            return general;
        }
        return fp_scalar_data(word);
    }
    return unknown(word);
}

bool proven_free_of(const RegisterEffects& effects, unsigned reg) noexcept {
    return register_fields(effects, reg) == std::optional<std::uint32_t>{0u};
}

std::optional<std::uint32_t> register_fields(const RegisterEffects& effects,
                                             unsigned reg) noexcept {
    if (reg >= 31 || !effects.decoded() ||
        has(effects.notes, EffectNote::handler_defined_registers)) {
        return std::nullopt;
    }
    std::uint32_t mask = 0;
    for (std::size_t index = 0; index < effects.operand_count; ++index) {
        const auto& operand = effects.operands[index];
        if (!operand.names_general_register(reg)) {
            continue;
        }
        if (operand.field == implicit_operand) {
            return std::nullopt;
        }
        // A CASP pair field also names the pair's implicit second register,
        // so renaming it alone is not a single-register substitution.
        if (effects.operation == Operation::casp && operand.role == OperandRole::transfer) {
            return std::nullopt;
        }
        mask |= operand.field_mask();
    }
    return mask;
}

bool reads_stack_pointer(const RegisterEffects& effects) noexcept {
    for (std::size_t index = 0; index < effects.operand_count; ++index) {
        const auto& operand = effects.operands[index];
        if (operand.names_stack_pointer() && includes_read(operand.access)) {
            return true;
        }
    }
    return false;
}

bool writes_stack_pointer(const RegisterEffects& effects) noexcept {
    for (std::size_t index = 0; index < effects.operand_count; ++index) {
        const auto& operand = effects.operands[index];
        if (operand.names_stack_pointer() && includes_write(operand.access)) {
            return true;
        }
    }
    return false;
}

} // namespace armigo::aarch64
