// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/relocation.h>

#include "displacement.h"

namespace armigo::aarch64 {
namespace {

RelocationError from_encode_error(EncodeError error) noexcept {
    switch (error) {
    case EncodeError::none:
        return RelocationError::none;
    case EncodeError::out_of_range:
        return RelocationError::out_of_range;
    case EncodeError::misaligned_address:
        return RelocationError::misaligned_address;
    case EncodeError::invalid_register:
    case EncodeError::invalid_operand:
        break;
    }
    return RelocationError::unsupported_operand;
}

// Re-encodes a PC-relative instruction at pc so it refers to target.
Encoded encode_at(const Instruction& i, std::uint64_t pc, std::uint64_t target) noexcept {
    switch (i.opcode) {
    case Opcode::b:
        return encode_b(pc, target);
    case Opcode::bl:
        return encode_bl(pc, target);
    case Opcode::b_cond:
    case Opcode::bc_cond:
        return encode_b_cond(i.condition, pc, target, i.opcode == Opcode::bc_cond);
    case Opcode::cbz:
    case Opcode::cbnz:
        return encode_cbz(i.opcode == Opcode::cbnz, i.wide, i.reg, pc, target);
    case Opcode::tbz:
    case Opcode::tbnz:
        return encode_tbz(i.opcode == Opcode::tbnz, i.reg, i.test_bit, pc, target);
    case Opcode::adr:
        return encode_adr(i.reg, pc, target);
    case Opcode::adrp:
        return encode_adrp(i.reg, pc, target);
    case Opcode::prfm_literal:
        return encode_load_literal(i.opcode, i.prefetch_operation, pc, target);
    case Opcode::ldr_literal_w:
    case Opcode::ldr_literal_x:
    case Opcode::ldrsw_literal:
    case Opcode::ldr_literal_s:
    case Opcode::ldr_literal_d:
    case Opcode::ldr_literal_q:
        return encode_load_literal(i.opcode, i.reg, pc, target);
    default:
        return Encoded{.word = 0, .error = EncodeError::invalid_operand};
    }
}

std::uint32_t load_width(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::ldr_literal_x:
    case Opcode::prfm_literal:
    case Opcode::ldr_literal_d:
        return 8;
    case Opcode::ldr_literal_q:
        return 16;
    default:
        return 4;
    }
}

// Accumulates sequence words; the first failure is kept.
class SequenceBuilder {
public:
    explicit SequenceBuilder(std::uint64_t start) noexcept : start_(start) {}

    // Address of the next word, or nullopt if it leaves the address space.
    std::optional<std::uint64_t> next_address() const noexcept {
        return detail::offset_address(start_, static_cast<std::int64_t>(4 * result_.count));
    }

    void add(Encoded encoded) noexcept {
        if (!result_.ok()) {
            return;
        }
        if (!encoded.ok()) {
            result_.error = from_encode_error(encoded.error);
            return;
        }
        if (result_.count == max_sequence_words) {
            result_.error = RelocationError::unsupported_operand;
            return;
        }
        if (!next_address()) {
            result_.error = RelocationError::address_overflow;
            return;
        }
        result_.words[result_.count++] = encoded.word;
    }

    void fail(RelocationError error) noexcept {
        if (result_.ok()) {
            result_.error = error;
        }
    }

    RelocatedSequence finish() noexcept {
        if (!result_.ok()) {
            result_.count = 0;
            result_.words = {};
        }
        return result_;
    }

private:
    std::uint64_t start_;
    RelocatedSequence result_{};
};

// adrp base ; then either one access at base + lo12 or an explicit add.
// Emits the page materialization and returns the low 12 bits.
std::uint32_t emit_page(SequenceBuilder& builder, unsigned base, std::uint64_t target) noexcept {
    const auto pc = builder.next_address();
    if (!pc) {
        builder.fail(RelocationError::address_overflow);
        return 0;
    }
    builder.add(encode_adrp(base, *pc, target));
    return static_cast<std::uint32_t>(target & 0xFFFu);
}

void emit_branch_to(SequenceBuilder& builder, std::uint64_t target) noexcept {
    const auto pc = builder.next_address();
    if (!pc) {
        builder.fail(RelocationError::address_overflow);
        return;
    }
    builder.add(encode_b(*pc, target));
}

// Inverted test that skips the following unconditional branch.
void emit_skip_and_branch(SequenceBuilder& builder, const Instruction& i, std::uint64_t to,
                          std::uint64_t target) noexcept {
    const auto skip = detail::offset_address(to, 8);
    if (!skip) {
        builder.fail(RelocationError::address_overflow);
        return;
    }
    switch (i.opcode) {
    case Opcode::b_cond:
    case Opcode::bc_cond:
        builder.add(encode_b_cond(invert(i.condition), to, *skip, i.opcode == Opcode::bc_cond));
        break;
    case Opcode::cbz:
    case Opcode::cbnz:
        builder.add(encode_cbz(i.opcode == Opcode::cbz, i.wide, i.reg, to, *skip));
        break;
    default:
        builder.add(encode_tbz(i.opcode == Opcode::tbz, i.reg, i.test_bit, to, *skip));
        break;
    }
    emit_branch_to(builder, target);
}

void emit_far_load(SequenceBuilder& builder, const Instruction& i, unsigned base,
                   std::uint64_t target) noexcept {
    const unsigned destination =
        i.opcode == Opcode::prfm_literal ? i.prefetch_operation : i.reg;
    const std::uint32_t low = emit_page(builder, base, target);
    if (low % load_width(i.opcode) == 0) {
        builder.add(encode_load_unsigned_offset(i.opcode, destination, base, low));
        return;
    }
    builder.add(encode_add_immediate(true, base, base, low));
    builder.add(encode_load_unsigned_offset(i.opcode, destination, base, 0));
}

void emit_preserved_link_call(SequenceBuilder& builder, const Instruction& i, std::uint64_t from,
                              std::uint64_t target) noexcept {
    if (i.opcode == Opcode::blr && i.reg == link_register) {
        // x30 would be overwritten before the branch reads it.
        builder.fail(RelocationError::unsupported_operand);
        return;
    }
    const auto link = detail::offset_address(from, 4);
    if (!link) {
        builder.fail(RelocationError::address_overflow);
        return;
    }
    const std::uint32_t low = emit_page(builder, link_register, *link);
    if (low != 0) {
        builder.add(encode_add_immediate(true, link_register, link_register, low));
    }
    if (i.opcode == Opcode::bl) {
        emit_branch_to(builder, target);
    } else {
        builder.add(encode_br(i.reg));
    }
}

bool valid_scratch(unsigned reg) noexcept {
    return reg <= 30 && reg != platform_register;
}

} // namespace

RelocatedWord relocate_instruction(std::uint32_t word, std::uint64_t from,
                                   std::uint64_t to) noexcept {
    if (from % 4 != 0 || to % 4 != 0) {
        return RelocatedWord{.error = RelocationError::misaligned_address};
    }
    const Instruction i = decode(word);
    if (!i.classified()) {
        return RelocatedWord{.error = RelocationError::unclassified_instruction};
    }
    const bool link_changed = writes_link_address(i.opcode) && from != to;
    if (!is_pc_relative(i.opcode)) {
        return RelocatedWord{.word = word, .link_value_changed = link_changed};
    }
    const auto target = referenced_address(i, from);
    if (!target) {
        return RelocatedWord{.error = RelocationError::address_overflow};
    }
    const Encoded encoded = encode_at(i, to, *target);
    if (!encoded.ok()) {
        return RelocatedWord{.error = from_encode_error(encoded.error)};
    }
    return RelocatedWord{.word = encoded.word, .link_value_changed = link_changed};
}

RelocatedSequence relocate_instruction_sequence(std::uint32_t word, std::uint64_t from,
                                                std::uint64_t to,
                                                const SequenceOptions& options) noexcept {
    const Instruction i = decode(word);
    const bool preserve_link =
        options.preserve_link_address && writes_link_address(i.opcode) && from != to;

    const RelocatedWord single = relocate_instruction(word, from, to);
    if (single.ok() && !preserve_link) {
        RelocatedSequence result;
        result.words[0] = single.word;
        result.count = 1;
        result.link_value_changed = single.link_value_changed;
        return result;
    }
    if (!single.ok() && single.error != RelocationError::out_of_range) {
        return RelocatedSequence{.error = single.error};
    }

    SequenceBuilder builder(to);
    if (preserve_link) {
        std::uint64_t target = 0;
        if (i.opcode == Opcode::bl) {
            target = *referenced_address(i, from);  // checked by relocate_instruction
        }
        emit_preserved_link_call(builder, i, from, target);
        return builder.finish();
    }

    const std::uint64_t target = *referenced_address(i, from);
    switch (i.opcode) {
    case Opcode::b_cond:
    case Opcode::bc_cond:
        if (i.condition == Condition::al || i.condition == Condition::nv) {
            emit_branch_to(builder, target);  // both encodings branch unconditionally
        } else {
            emit_skip_and_branch(builder, i, to, target);
        }
        break;
    case Opcode::cbz:
    case Opcode::cbnz:
    case Opcode::tbz:
    case Opcode::tbnz:
        emit_skip_and_branch(builder, i, to, target);
        break;
    case Opcode::adr:
        if (i.reg == register_field_zero_or_sp) {
            // ADD would address SP for register 31.
            builder.fail(RelocationError::unsupported_operand);
            break;
        }
        if (const std::uint32_t low = emit_page(builder, i.reg, target); low != 0) {
            builder.add(encode_add_immediate(true, i.reg, i.reg, low));
        }
        break;
    case Opcode::ldr_literal_w:
    case Opcode::ldr_literal_x:
    case Opcode::ldrsw_literal:
        if (i.reg == register_field_zero_or_sp) {
            // A base register of 31 would address SP, not XZR.
            builder.fail(RelocationError::unsupported_operand);
            break;
        }
        emit_far_load(builder, i, i.reg, target);
        break;
    case Opcode::prfm_literal:
    case Opcode::ldr_literal_s:
    case Opcode::ldr_literal_d:
    case Opcode::ldr_literal_q:
        if (options.scratch_register == no_scratch_register) {
            builder.fail(RelocationError::scratch_register_required);
        } else if (!valid_scratch(options.scratch_register)) {
            builder.fail(RelocationError::invalid_scratch_register);
        } else {
            emit_far_load(builder, i, options.scratch_register, target);
        }
        break;
    default:
        // B, BL without link preservation, and ADRP have no register-free
        // expansion; their single-word failure stands.
        builder.fail(RelocationError::out_of_range);
        break;
    }
    return builder.finish();
}

} // namespace armigo::aarch64
