// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/scalar_x18_patch.h>

#include <armigo/aarch64/register_substitution.h>
#include <armigo/execution/guest_register_cell.h>

#include <limits>
#include <optional>
#include <utility>

namespace armigo::execution {
namespace {

using namespace aarch64;

constexpr std::uint32_t stack_pointer = 31;

// The trampoline reserves 16 bytes below guest SP. The spill occupies
// [SP - 16, SP); every prepared access must stay outside those bytes.
constexpr std::int64_t spill_bytes = 16;

bool covers_spill(std::int64_t address_offset, std::int64_t transfer_bytes) noexcept {
    return address_offset < 0 && address_offset + transfer_bytes > -spill_bytes;
}

// The access moves 16 bytes up because SP itself moved down by the spill.
// Returns the word to execute at the trampoline access slot (two words when a
// pair is split) and the SP displacement the epilogue must apply from the
// spilled frame.
struct AdjustedAccess {
    std::uint32_t word = 0;
    std::uint32_t second = 0;
    std::int64_t restore_delta = spill_bytes;
};

// A general-register pair (STP, LDP, LDPSW, STNP, LDNP) whose moved offset no
// longer fits its imm7, as two single-register transfers with scaled unsigned
// offsets: Rt at `moved` and Rt2 one element above, the same bytes the pair
// transfers. Each register is loaded or stored as by the pair (LDPSW as
// LDRSW); the two halves are separate accesses, so a 16-byte pair's
// single-copy atomicity as one access is not kept. STGP, SIMD&FP pairs and a
// load into one register twice are not split.
std::optional<std::pair<std::uint32_t, std::uint32_t>> split_pair(std::uint32_t instruction, std::int64_t moved,
                                                                  std::int64_t bytes) noexcept {
    if ((instruction >> 26) & 1u) return {};
    const auto opc = instruction >> 30;
    const bool load = (instruction >> 22) & 1u;
    const auto mode = (instruction >> 23) & 3u;
    const auto rt = instruction & 31u, rt2 = (instruction >> 10) & 31u, rn = (instruction >> 5) & 31u;
    std::uint32_t base = 0;
    if (opc == 2) base = load ? 0xf9400000u : 0xf9000000u;       // LDR/STR X
    else if (opc == 0) base = load ? 0xb9400000u : 0xb9000000u;  // LDR/STR W
    else if (opc == 1 && load && mode != 0) base = 0xb9800000u;   // LDRSW X
    else return {};
    if (load && rt == rt2) return {};
    if (moved < 0 || moved % bytes != 0 || moved / bytes + 1 > 4095) return {};
    const auto first = static_cast<std::uint32_t>(moved / bytes);
    return std::pair{base | (first << 10) | (rn << 5) | rt, base | ((first + 1) << 10) | (rn << 5) | rt2};
}

enum class AdjustStatus : std::uint8_t { unsupported, out_of_range, ok };

struct StackAdjustment {
    AdjustStatus status = AdjustStatus::unsupported;
    AdjustedAccess access{};
};

StackAdjustment adjust_stack_access(std::uint32_t instruction,
                                    const RegisterEffects& effects) noexcept {
    // These field adjustments belong only to scalar/vector single-register
    // and pair encodings. Structure transfers and ordered/exclusive accesses
    // have different layouts even when their memory summaries look similar.
    const auto memory_class = instruction & 0x3A000000u;
    if (memory_class != 0x38000000u && memory_class != 0x28000000u) return {};
    const MemoryEffect& memory = effects.memory;
    const std::int64_t bytes = memory.element_bytes;
    if (bytes <= 0 || (bytes & (bytes - 1)) != 0 || memory.elements == 0) return {};
    const std::int64_t transfer = bytes * memory.elements;
    const bool writeback_form = memory.addressing == AddressingMode::pre_index ||
                                memory.addressing == AddressingMode::post_index;
    const std::int64_t access_offset =
        memory.addressing == AddressingMode::post_index ? 0 : memory.offset;
    if (covers_spill(access_offset, transfer)) return {};
    const std::int64_t moved = access_offset + spill_bytes;
    AdjustedAccess result{.word = instruction,
                          .restore_delta = writeback_form ? memory.offset + spill_bytes
                                                          : spill_bytes};
    if (memory.elements == 2) {
        const std::uint32_t mode = (instruction >> 23) & 3u;
        if (mode == 1 || mode == 3) {
            // Pre-index and post-index become one plain signed-offset access;
            // the epilogue performs the guest's SP update.
            result.word = (instruction & ~(3u << 23)) | (2u << 23);
        } else if (mode != 0 && mode != 2) {
            return {};
        }
        if (moved % bytes != 0) return {AdjustStatus::out_of_range, {}};
        const std::int64_t immediate = moved / bytes;
        if (immediate > 63) {
            // Only the upper end overflows: the spill raises every offset by
            // 16, and imm7's lowest offset stays in range after that.
            const auto split = split_pair(instruction, moved, bytes);
            if (!split) return {AdjustStatus::out_of_range, {}};
            result.word = split->first;
            result.second = split->second;
            return {AdjustStatus::ok, result};
        }
        if (immediate < -64) return {AdjustStatus::out_of_range, {}};
        result.word = (result.word & ~(0x7Fu << 15)) |
                      (static_cast<std::uint32_t>(immediate) & 0x7Fu) << 15;
        return {AdjustStatus::ok, result};
    }
    if ((instruction >> 24) & 1u) {
        // Unsigned scaled immediate offset.
        const std::int64_t immediate = moved / bytes;
        if (moved % bytes != 0 || immediate < 0 || immediate > 4095) {
            return {AdjustStatus::out_of_range, {}};
        }
        result.word = (instruction & ~(4095u << 10)) |
                      (static_cast<std::uint32_t>(immediate) << 10);
        return {AdjustStatus::ok, result};
    }
    if ((instruction >> 21) & 1u) return {};  // register offset
    const std::uint32_t mode = (instruction >> 10) & 3u;
    if (mode == 1 || mode == 3) {
        // Pre-index and post-index become one unscaled access; the epilogue
        // performs the guest's SP update.
        result.word = instruction & ~(3u << 10);
    }
    if (moved < -256 || moved > 255) return {AdjustStatus::out_of_range, {}};
    result.word = (result.word & ~(0x1FFu << 12)) |
                  (static_cast<std::uint32_t>(moved) & 0x1FFu) << 12;
    return {AdjustStatus::ok, result};
}

// The two halves of a split pair transfer the pair's registers, one each, at
// `moved` and one element above, as the pair would: same direction, element
// width and attributes (apart from the non-temporal hint), no writeback.
bool split_matches(const RegisterEffects& pair, std::uint32_t first, std::uint32_t second,
                   std::int64_t moved) noexcept {
    const auto attributes = static_cast<MemoryAttributes>(
        pair.memory.attributes & ~static_cast<MemoryAttributes>(MemoryAttribute::non_temporal));
    std::uint64_t reads = 0, writes = 0;
    std::int64_t offset = moved;
    for (const auto word : {first, second}) {
        const auto half = decode_register_effects(word);
        if (!half.decoded() || half.memory.operation != pair.memory.operation ||
            half.memory.element_bytes != pair.memory.element_bytes || half.memory.elements != 1 ||
            half.memory.addressing != AddressingMode::unsigned_offset || half.memory.offset != offset ||
            half.memory.writeback || half.memory.attributes != attributes || half.reads_flags || half.writes_flags)
            return false;
        reads |= half.reads;
        writes |= half.writes;
        offset += pair.memory.element_bytes;
    }
    return reads == pair.reads && writes == pair.writes;
}

// ADR cannot be relocated into an appended trampoline: its byte displacement
// is limited to +-1 MiB, which a whole-module appendix usually exceeds. The
// target is known when the module is prepared, so materialize that absolute
// address instead. MOVZ/MOVK never change NZCV.
std::size_t emit_address_value(std::uint64_t target, unsigned reg,
                               std::uint32_t* out) noexcept {
    for (unsigned half = 0; half != 4; ++half) {
        const auto part = static_cast<std::uint32_t>((target >> (16 * half)) & 0xFFFFu);
        out[half] = (half == 0 ? 0xd2800000u : 0xf2800000u) | (half << 21) | (part << 5) | reg;
    }
    return 4;
}

// The adjusted access must be the original transfer with only its address
// arithmetic changed.
bool adjustment_matches(const RegisterEffects& before, const RegisterEffects& after,
                        std::int64_t moved, bool writeback_form) noexcept {
    if (!after.decoded() || after.operation != before.operation ||
        after.memory.operation != before.memory.operation ||
        after.memory.element_bytes != before.memory.element_bytes ||
        after.memory.elements != before.memory.elements ||
        after.memory.attributes != before.memory.attributes || after.memory.writeback ||
        after.memory.offset != moved || after.reads != before.reads ||
        after.writes != before.writes || after.notes != before.notes) {
        return false;
    }
    const AddressingMode expected = before.memory.elements == 2 ? AddressingMode::signed_offset
                                                                : AddressingMode::unscaled_offset;
    if (writeback_form) {
        return after.memory.addressing == expected;
    }
    return after.memory.addressing == before.memory.addressing;
}

// CASP whose Rs or Rt pair starts at x18: the pair's second register is the
// guest's own x19, so x18 cannot be renamed alone. The pair is rebuilt in an
// even/odd scratch pair: E takes guest x18, E+1 a copy of x19, and a third
// scratch C holds the register-cell address. The spill therefore reserves 32
// bytes below SP (E and C at [SP], E+1 at [SP, #16]); SP-based accesses are
// refused before this point. When Rs names x18 the instruction writes the
// pair, and E and E+1 go back to the cell and to x19.
ScalarX18PatchResult make_pair_patch(std::uint32_t instruction, std::uint64_t address,
                                     std::uint64_t trampoline, std::uint32_t slot,
                                     const RegisterEffects& effects) noexcept {
    unsigned even = 32, cell = 32;
    for (unsigned reg = 0; reg + 1 < 16 && even == 32; reg += 2)
        if (proven_free_of(effects, reg) && proven_free_of(effects, reg + 1)) even = reg;
    for (unsigned reg = 0; reg < 16 && cell == 32; ++reg)
        if (reg != even && reg != even + 1 && proven_free_of(effects, reg)) cell = reg;
    if (even == 32 || cell == 32) return {.error = ScalarX18PatchError::substitution_failed};
    std::uint32_t rewritten = instruction;
    for (const unsigned field : {0u, 5u, 16u})
        if (((instruction >> field) & 31u) == platform_register)
            rewritten = (rewritten & ~(31u << field)) | (even << field);
    const auto after = decode_register_effects(rewritten);
    if (!after.decoded() || after.operation != Operation::casp ||
        after.memory.operation != effects.memory.operation || after.memory.elements != effects.memory.elements ||
        after.memory.element_bytes != effects.memory.element_bytes ||
        after.memory.attributes != effects.memory.attributes ||
        ((after.reads | after.writes) & (register_bit(platform_register) | register_bit(cell)))) {
        return {.error = ScalarX18PatchError::substitution_failed};
    }
    const bool writes_pair = ((instruction >> 16) & 31u) == platform_register;
    const std::uint32_t word_count = writes_pair ? 12 : 10;
    const auto enter = encode_b(address, trampoline);
    const auto resume = encode_b(trampoline + 4 * (word_count - 1), address + 4);
    if (!enter.ok() || !resume.ok()) return {.error = ScalarX18PatchError::unreachable_branch};

    static_assert(kGuestRegisterCellX18Offset == 0);
    const unsigned odd = even + 1;
    constexpr std::uint32_t guest_x19 = 19;
    ScalarX18Patch patch{.instruction_address = address,
                         .trampoline_address = trampoline,
                         .expected_instruction = instruction,
                         .replacement_instruction = enter.word,
                         .value_register = static_cast<std::uint8_t>(even),
                         .cell_register = static_cast<std::uint8_t>(cell)};
    std::size_t next = 0;
    auto& words = patch.trampoline;
    words[next++] = 0xa9be0000u | (cell << 10) | (stack_pointer << 5) | even;          // STP E, C, [SP, #-32]!
    words[next++] = encode_mrs(cell, system_register_encoding(3, 3, 13, 0, 3)).word;  // MRS C, TPIDRRO_EL0
    words[next++] = 0xf9400000u | ((slot / 8) << 10) | (cell << 5) | cell;            // LDR C, [C, #slot]
    words[next++] = 0xf9000800u | (stack_pointer << 5) | odd;                         // STR E+1, [SP, #16]
    words[next++] = 0xf9400000u | (cell << 5) | even;                                 // LDR E, [C]
    words[next++] = 0xaa0003e0u | (guest_x19 << 16) | odd;                            // MOV E+1, X19
    words[next++] = rewritten;
    if (writes_pair) {
        words[next++] = 0xf9000000u | (cell << 5) | even;                             // STR E, [C]
        words[next++] = 0xaa0003e0u | (odd << 16) | guest_x19;                        // MOV X19, E+1
    }
    words[next++] = 0xf9400800u | (stack_pointer << 5) | odd;                         // LDR E+1, [SP, #16]
    words[next++] = 0xa8c20000u | (cell << 10) | (stack_pointer << 5) | even;         // LDP E, C, [SP], #32
    words[next++] = resume.word;
    patch.trampoline_word_count = static_cast<std::uint8_t>(next);
    return {.patch = patch};
}

} // namespace

ScalarX18PatchResult make_scalar_x18_patch(
    std::uint32_t instruction, std::uint64_t address, std::uint64_t trampoline,
    std::uint32_t slot, ScalarX18Policy policy) noexcept {
    using namespace aarch64;
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (address % 4 != 0 || trampoline % 4 != 0 || address > maximum - 7 ||
        trampoline > maximum - 47) {
        return {.error = ScalarX18PatchError::invalid_layout};
    }
    if (address <= trampoline + 47 && trampoline <= address + 7) {
        return {.error = ScalarX18PatchError::overlapping_regions};
    }
    if (slot % 8 != 0 || slot > 4095 * 8) {
        return {.error = ScalarX18PatchError::invalid_thread_slot};
    }
    if (!policy.exclusive_monitor_clear || !policy.stack_scratch_available ||
        !policy.runtime_storage_disjoint) {
        return {.error = ScalarX18PatchError::missing_execution_contract};
    }
    const auto effects = decode_register_effects(instruction);
    const bool immediate_address = effects.opcode == Opcode::adr || effects.opcode == Opcode::adrp;
    if (!effects.decoded() || (effects.operation == Operation::classified && !immediate_address) ||
        effects.operation == Operation::none || effects.control_flow != ControlFlow::none ||
        has(effects.notes, EffectNote::handler_defined_registers)) {
        return {.error = ScalarX18PatchError::unsupported_instruction};
    }
    // Any SP operand changes meaning inside the trampoline, because the spill
    // has already moved the hardware stack pointer down by 16 bytes. Immediate
    // arithmetic can use a reconstructed guest SP in the value scratch.
    const bool stack_arithmetic = (instruction & 31u) == platform_register &&
        ((instruction >> 5) & 31u) == stack_pointer &&
        (effects.operation == Operation::add_immediate ||
         effects.operation == Operation::adds_immediate ||
         effects.operation == Operation::sub_immediate ||
         effects.operation == Operation::subs_immediate);
    const bool stack_base = reads_stack_pointer(effects) &&
                            effects.memory.operation != MemoryOperation::none;
    if (!stack_base && !stack_arithmetic &&
        (reads_stack_pointer(effects) || writes_stack_pointer(effects))) {
        return {.error = ScalarX18PatchError::stack_pointer_operand};
    }
    std::optional<AdjustedAccess> adjusted;
    if (stack_base) {
        const auto adjustment = adjust_stack_access(instruction, effects);
        if (adjustment.status == AdjustStatus::unsupported) {
            return {.error = ScalarX18PatchError::stack_pointer_operand};
        }
        if (adjustment.status == AdjustStatus::out_of_range) {
            return {.error = ScalarX18PatchError::substitution_failed};
        }
        adjusted = adjustment.access;
    }
    if (has(effects.memory.attributes, MemoryAttribute::exclusive)) {
        return {.error = ScalarX18PatchError::exclusive_instruction};
    }
    if (((effects.reads | effects.writes) & register_bit(platform_register)) == 0) {
        return {.error = ScalarX18PatchError::register_not_used};
    }
    if (effects.operation == Operation::casp &&
        (((instruction >> 16) & 31u) == platform_register || (instruction & 31u) == platform_register)) {
        if (!policy.pair_stack_scratch_available) return {.error = ScalarX18PatchError::missing_execution_contract};
        return make_pair_patch(instruction, address, trampoline, slot, effects);
    }
    std::array<unsigned, 2> scratch{};
    unsigned count = 0;
    for (unsigned reg = 0; reg < 16 && count < scratch.size(); ++reg) {
        if (proven_free_of(effects, reg)) scratch[count++] = reg;
    }
    if (count != scratch.size()) {
        return {.error = ScalarX18PatchError::substitution_failed};
    }
    const auto value = scratch[0];
    const auto cell = scratch[1];
    std::array<std::uint32_t, 4> address_words{};
    std::size_t address_word_count = 1;
    if (immediate_address) {
        const auto target = referenced_address(decode(instruction), address);
        if (!target) return {.error = ScalarX18PatchError::substitution_failed};
        if (effects.opcode == Opcode::adr) {
            address_word_count = emit_address_value(*target, value, address_words.data());
        } else {
            // ADRP keeps the original 4 KiB page target; relocate the page
            // calculation to the substituted word at trampoline instruction 4.
            const auto rewritten = encode_adrp(value, trampoline + 16, *target);
            if (!rewritten.ok()) return {.error = ScalarX18PatchError::substitution_failed};
            address_words[0] = rewritten.word;
        }
        if (address_word_count > 4) return {.error = ScalarX18PatchError::substitution_failed};
    } else if (adjusted && adjusted->second) {
        const auto moved = effects.memory.addressing == AddressingMode::post_index ? spill_bytes
                                                                                   : effects.memory.offset + spill_bytes;
        if (!split_matches(effects, adjusted->word, adjusted->second, moved)) {
            return {.error = ScalarX18PatchError::substitution_failed};
        }
        // Only the half that names x18 moves to the value register.
        address_word_count = 2;
        for (const auto [index, word] : {std::pair{0, adjusted->word}, std::pair{1, adjusted->second}}) {
            const auto half = decode_register_effects(word);
            if (((half.reads | half.writes) & register_bit(platform_register)) == 0) {
                address_words[index] = word;
                continue;
            }
            const auto rewritten = substitute_register(word, platform_register, value, UnchangedWord::reject);
            if (!rewritten.ok()) return {.error = ScalarX18PatchError::substitution_failed};
            address_words[index] = rewritten.word;
        }
    } else {
        const std::uint32_t access = adjusted ? adjusted->word : instruction;
        if (adjusted && !adjustment_matches(effects, decode_register_effects(access),
                                            effects.memory.addressing == AddressingMode::post_index
                                                ? spill_bytes
                                                : effects.memory.offset + spill_bytes,
                                            effects.memory.addressing == AddressingMode::pre_index ||
                                                effects.memory.addressing == AddressingMode::post_index)) {
            return {.error = ScalarX18PatchError::substitution_failed};
        }
        const auto rewritten = substitute_register(access, platform_register, value,
                                                   UnchangedWord::reject);
        if (!rewritten.ok()) return {.error = ScalarX18PatchError::substitution_failed};
        address_words[0] = rewritten.word;
        if (stack_arithmetic) {
            // Checked substitution handles the destination. SP is not a GPR
            // that API may rename, so only this validated immediate class has
            // its source replaced by the reconstructed guest SP.
            address_words[0] = (address_words[0] & ~(31u << 5)) | (value << 5);
            const auto after = decode_register_effects(address_words[0]);
            if (!after.decoded() || after.operation != effects.operation ||
                after.reads != register_bit(value) || after.writes != register_bit(value) ||
                after.reads_flags != effects.reads_flags ||
                after.writes_flags != effects.writes_flags ||
                reads_stack_pointer(after) || writes_stack_pointer(after)) {
                return {.error = ScalarX18PatchError::substitution_failed};
            }
        }
    }
    const bool long_epilogue = adjusted && adjusted->restore_delta != spill_bytes;
    const std::uint32_t word_count = static_cast<std::uint32_t>(
        4 + address_word_count + 2 + (long_epilogue ? 1 : 0) + 1);
    const auto enter = encode_b(address, trampoline);
    const auto resume = encode_b(trampoline + 4 * (word_count - 1), address + 4);
    if (!enter.ok() || !resume.ok()) {
        return {.error = ScalarX18PatchError::unreachable_branch};
    }

    static_assert(kGuestRegisterCellX18Offset == 0);
    const auto read_tsd = encode_mrs(cell, system_register_encoding(3, 3, 13, 0, 3));
    // All injected memory operations use ordinary 64-bit words and leave
    // NZCV untouched. The original instruction alone determines flag changes.
    ScalarX18Patch patch{.instruction_address = address,
                         .trampoline_address = trampoline,
                         .expected_instruction = instruction,
                         .replacement_instruction = enter.word,
                         .value_register = static_cast<std::uint8_t>(value),
                         .cell_register = static_cast<std::uint8_t>(cell)};
    std::array<std::uint32_t, 12> words{};
    std::size_t next = 0;
    words[next++] = 0xa9bf0000u | (cell << 10) | (stack_pointer << 5) | value; // STP [SP, #-16]!
    words[next++] = read_tsd.word;
    words[next++] = 0xf9400000u | ((slot / 8) << 10) | (cell << 5) | cell;
    words[next++] = stack_arithmetic
        ? 0x91000000u | (16u << 10) | (stack_pointer << 5) | value // ADD Xvalue, SP, #16
        : 0xf9400000u | (cell << 5) | value;
    for (std::size_t index = 0; index < address_word_count; ++index) {
        words[next++] = address_words[index];
    }
    words[next++] = 0xf9000000u | (cell << 5) | value;
    if (!long_epilogue) {
        // LDP [SP], #16 both reloads the scratch registers and restores SP.
        words[next++] = 0xa8c10000u | (cell << 10) | (stack_pointer << 5) | value;
    } else {
        words[next++] = 0xa9400000u | (cell << 10) | (stack_pointer << 5) | value;
        const std::int64_t delta = adjusted->restore_delta;
        const auto magnitude = static_cast<std::uint32_t>(delta < 0 ? -delta : delta);
        const std::uint32_t base = delta < 0 ? 0xd1000000u : 0x91000000u;  // SUB/ADD SP, SP, #imm
        words[next++] = base | (magnitude << 10) | (stack_pointer << 5) | stack_pointer;
    }
    words[next++] = resume.word;
    for (std::size_t index = 0; index < next; ++index) patch.trampoline[index] = words[index];
    patch.trampoline_word_count = static_cast<std::uint8_t>(next);
    return {.patch = patch};
}

} // namespace armigo::execution
