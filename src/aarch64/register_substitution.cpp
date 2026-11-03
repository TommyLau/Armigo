// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The rewrite uses only operand fields from decode_register_effects(). Every
// result is decoded again and compared member by member with the original, so
// a decoder or field-layout error surfaces as a refusal rather than as a
// silently different instruction.

#include <armigo/aarch64/register_substitution.h>

namespace armigo::aarch64 {
namespace {

SubstitutedWord failure(SubstitutionError error) noexcept {
    return SubstitutedWord{.error = error};
}

// Validated input: either a refusal, or the field mask the rewrite must change
// (zero when the word stays unchanged).
struct Plan {
    SubstitutionError error = SubstitutionError::none;
    std::uint32_t fields = 0;
};

Plan plan(const RegisterEffects& effects, unsigned from, unsigned to,
          UnchangedWord unchanged) noexcept {
    if (from > 30 || to > 30) {
        return {SubstitutionError::invalid_register};
    }
    switch (effects.status) {
    case EffectStatus::unknown:
        return {SubstitutionError::unknown_encoding};
    case EffectStatus::reserved:
        return {SubstitutionError::reserved_encoding};
    case EffectStatus::unpredictable:
        return {SubstitutionError::unpredictable_encoding};
    case EffectStatus::decoded:
        break;
    }
    if (has(effects.notes, EffectNote::handler_defined_registers)) {
        return {SubstitutionError::handler_defined_registers};
    }
    const std::optional<std::uint32_t> fields = register_fields(effects, from);
    if (!fields) {
        return {SubstitutionError::implicit_operand};
    }
    if (from == to || *fields == 0) {
        if (unchanged == UnchangedWord::accept) {
            return {};
        }
        return {from == to ? SubstitutionError::same_register
                           : SubstitutionError::register_absent};
    }
    for (std::size_t index = 0; index < effects.operand_count; ++index) {
        if (effects.operands[index].names_general_register(to)) {
            return {SubstitutionError::register_collision};
        }
    }
    return {SubstitutionError::none, *fields};
}

constexpr RegisterMask rename(RegisterMask mask, unsigned from, unsigned to) noexcept {
    if ((mask & register_bit(from)) == 0) {
        return mask;
    }
    return (mask & ~register_bit(from)) | register_bit(to);
}

bool same_operand_shape(const RegisterOperand& a, const RegisterOperand& b) noexcept {
    return a.field == b.field && a.file == b.file && a.at_31 == b.at_31 && a.bits == b.bits &&
           a.access == b.access && a.role == b.role && a.conditional == b.conditional;
}

bool same_memory(const MemoryEffect& a, const MemoryEffect& b) noexcept {
    return a.operation == b.operation && a.addressing == b.addressing &&
           a.element_bytes == b.element_bytes && a.elements == b.elements &&
           a.extend == b.extend && a.index_shift == b.index_shift && a.offset == b.offset &&
           a.writeback == b.writeback && a.attributes == b.attributes;
}

// The rewritten effects must be the original effects with every general
// operand that named from naming to instead, and nothing else changed.
bool renamed_effects(const RegisterEffects& original, const RegisterEffects& rewritten,
                     unsigned from, unsigned to) noexcept {
    if (rewritten.operation != original.operation || rewritten.opcode != original.opcode ||
        rewritten.operand_count != original.operand_count ||
        rewritten.reads != rename(original.reads, from, to) ||
        rewritten.writes != rename(original.writes, from, to) ||
        rewritten.reads_flags != original.reads_flags ||
        rewritten.writes_flags != original.writes_flags ||
        rewritten.control_flow != original.control_flow ||
        !same_memory(rewritten.memory, original.memory) || rewritten.notes != original.notes) {
        return false;
    }
    for (std::size_t index = 0; index < original.operand_count; ++index) {
        const RegisterOperand& before = original.operands[index];
        const RegisterOperand& after = rewritten.operands[index];
        const unsigned expected = before.names_general_register(from) ? to : before.number;
        if (!same_operand_shape(before, after) || after.number != expected) {
            return false;
        }
    }
    return true;
}

// Classifier fields beyond the register effects (displacement, condition,
// test bit, system register, immediates) must be unchanged.
bool renamed_classification(const Instruction& original, const Instruction& rewritten,
                            unsigned from, unsigned to) noexcept {
    const unsigned expected_reg =
        original.classified() && original.reg == from ? to : original.reg;
    return rewritten.opcode == original.opcode && rewritten.reg == expected_reg &&
           rewritten.wide == original.wide && rewritten.condition == original.condition &&
           rewritten.test_bit == original.test_bit &&
           rewritten.prefetch_operation == original.prefetch_operation &&
           rewritten.immediate == original.immediate &&
           rewritten.system_register == original.system_register &&
           rewritten.displacement == original.displacement &&
           rewritten.reads == rename(original.reads, from, to) &&
           rewritten.writes == rename(original.writes, from, to);
}

SubstitutedWord check(std::uint32_t original, const RegisterEffects& original_effects,
                      std::uint32_t rewritten, unsigned from, unsigned to,
                      const Plan& validated) noexcept {
    if (((original ^ rewritten) & ~validated.fields) != 0) {
        return failure(SubstitutionError::changed_outside_fields);
    }
    if (validated.fields == 0) {
        return SubstitutedWord{.word = original, .effects = original_effects};
    }
    const RegisterEffects effects = decode_register_effects(rewritten);
    if (!effects.decoded()) {
        return failure(SubstitutionError::output_not_decoded);
    }
    if (!renamed_effects(original_effects, effects, from, to) ||
        !renamed_classification(decode(original), decode(rewritten), from, to)) {
        return failure(SubstitutionError::effects_mismatch);
    }
    return SubstitutedWord{.word = rewritten, .fields = validated.fields, .effects = effects};
}

} // namespace

SubstitutedWord substitute_register(std::uint32_t word, unsigned from, unsigned to,
                                    UnchangedWord unchanged) noexcept {
    const RegisterEffects effects = decode_register_effects(word);
    const Plan validated = plan(effects, from, to, unchanged);
    if (validated.error != SubstitutionError::none) {
        return failure(validated.error);
    }
    std::uint32_t rewritten = word;
    if (validated.fields != 0) {
        for (std::size_t index = 0; index < effects.operand_count; ++index) {
            const RegisterOperand& operand = effects.operands[index];
            if (operand.names_general_register(from)) {
                rewritten = (rewritten & ~operand.field_mask()) |
                            (std::uint32_t{to} << operand.field);
            }
        }
    }
    return check(word, effects, rewritten, from, to, validated);
}

SubstitutedWord verify_register_substitution(std::uint32_t original, std::uint32_t rewritten,
                                             unsigned from, unsigned to,
                                             UnchangedWord unchanged) noexcept {
    const RegisterEffects effects = decode_register_effects(original);
    const Plan validated = plan(effects, from, to, unchanged);
    if (validated.error != SubstitutionError::none) {
        return failure(validated.error);
    }
    return check(original, effects, rewritten, from, to, validated);
}

} // namespace armigo::aarch64
