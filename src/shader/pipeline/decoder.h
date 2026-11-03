// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once
#include "facts.h"
#include "ir.h"

#include <array>
#include <bit>
#include <cstddef>
#include <vector>

namespace armigo::shader::pipeline {
// Owned input copied from validated shader_input spans. bytes contains only the
// instruction stream, including scheduling words. Entry is normally byte 8.
// Compute has no SPH; its zero-filled sph is unused and launch state is supplied as facts.
// The adapter checks SPH/stage, strips wrappers/C1, and rejects >32-bit sizes.
struct StageImage {
    Stage stage{Stage::vertex};
    std::vector<std::byte> bytes;
    std::array<std::byte, 0x50> sph{};
    Pc entry_pc{8};
};
struct RawInstruction {
    std::uint64_t word{};
    template <unsigned First, unsigned Width>
    [[nodiscard]] constexpr std::uint64_t bits() const {
        static_assert(Width > 0 && Width <= 64 && First <= 64 - Width);
        if constexpr (Width == 64) return word;
        else return (word >> First) & ((std::uint64_t{1} << Width) - 1);
    }
    template <unsigned First, unsigned Width>
    [[nodiscard]] constexpr std::int64_t signed_bits() const {
        const auto value = bits<First, Width>();
        if constexpr (Width == 64) return std::bit_cast<std::int64_t>(value);
        else {
            const auto sign = std::uint64_t{1} << (Width - 1);
            return static_cast<std::int64_t>(value ^ sign) - static_cast<std::int64_t>(sign);
        }
    }
};
enum class Encoding : std::uint8_t {
    none, registers, immediate, immediate32, constant_buffer,
    register_constant, constant_register, texture_bound, texture_bindless
};
struct InstructionPredicate {
    std::uint8_t index{7};
    bool inverted{};
    bool enabled{}; // False for NoPred encodings, independent of bits in the word.
};
enum class RegisterRole : std::uint8_t {
    destination, second_destination, source_a, source_b, source_c,
    predicate_destination, second_predicate_destination, predicate_source,
    condition_code
};
struct RegisterAccess {
    RegisterRole role{RegisterRole::source_a};
    std::uint8_t lane{}; // Ascending word/component within a register pair/vector.
    Register reg;
    Access access{Access::read};
};
enum class SourceRole : std::uint8_t { a, b, c, predicate };
enum class Subword : std::uint8_t { full, byte0, byte1, byte2, byte3, half_low, half_high };
struct SourceModifiers {
    bool absolute{}, negate{};
    Subword subword{Subword::full};
    ScalarType scalar{ScalarType::u32};
};
struct DecodedSource {
    SourceRole role{SourceRole::a};
    std::variant<Register, LiteralBits, ConstantWord> value;
    SourceModifiers modifiers;
};
// Family selectors preserve the encoded selector bits, not guessed semantics.
// Shared keys avoid field-name drift; each family documents accepted selector values.
enum class DecodedField : std::uint8_t {
    condition, boolean_operation, write_cc, extended, signedness,
    source_type, destination_type, half_mode, result_scale,
    shift_mode, shift_wrap, bitfield_mode, permutation_mode, function,
    branch_displacement, branch_absolute, target_register,
    texture_dimension, texture_mode, texture_component_mask, gather_component,
    texture_offset, texture_handle_byte_offset, texture_cbuf_slot,
    memory_element_bytes, memory_elements, memory_byte_offset, memory_space,
    memory_order, memory_scope, atomic_operation,
    io_byte_offset, interpolation_mode, interpolation_location,
    barrier_kind, barrier_id,
    // Texture selectors retain ISA values: target is TEXS/TLDS Target; query is
    // TXQ TexQuery. compare/multisample/lod_clamp/ndv/clamp are explicit booleans.
    // texture_mode is TEX Lod (0,1,2,3,6,7), or TLD Lod (0,1).
    // texture_dimension is TexDim (dimension * 2 + array); texture_offset is
    // TLD4 TexOffset (0,1,2), otherwise a boolean. Compact component masks keep
    // their encoded 3-bit selector; other masks retain their four component bits.
    texture_target, texture_query, texture_compare, texture_multisample,
    texture_lod_clamp, texture_ndv, texture_clamp,
    // ALU selectors added for decode/emit alignment: integral_round keeps the
    // 3-bit F2F integral-rounding selector (0..3 pass the value through,
    // 4 round, 5 floor, 6 ceiling, 7 truncate); integer_saturation is the I2I
    // destination saturation flag. shuffle_mode keeps the 2-bit SHFL mode
    // (index=0, up=1, down=2, butterfly=3).
    integral_round, integer_saturation, shuffle_mode
};
struct FieldValue { DecodedField field; std::uint64_t bits{}; };
struct DecodedInstruction {
    Pc pc{};
    RawInstruction raw;
    InstructionName name{InstructionName::invalid};
    Encoding encoding{Encoding::none};
    InstructionPredicate predicate;
    // Exact physical reads/writes, including second results and individual CC flags.
    std::vector<RegisterAccess> registers;
    std::vector<DecodedSource> sources; // At most one entry for each SourceRole.
    std::vector<FieldValue> fields; // At most one entry for each DecodedField.
    // Decoders normalize operands/byte offsets once. Family callbacks use these
    // shared fields; raw remains available for new format accessors and diagnostics.
    // Missing required fields are errors, never zero-valued defaults.
    std::optional<NumericMode> numeric;
};
[[nodiscard]] inline Result<std::uint64_t> decoded_field(const DecodedInstruction& instruction, DecodedField field) {
    std::optional<std::uint64_t> found;
    for (const auto& item : instruction.fields) if (item.field == field) {
        if (found) return diagnose(Phase::decode, DiagnosticCode::malformed_input,
            {instruction.pc, instruction.name}, "Duplicate decoded field");
        found = item.bits;
    }
    if (found) return *found;
    return diagnose(Phase::emit_ir, DiagnosticCode::unsupported_modifier,
        {instruction.pc, instruction.name}, "Required decoded field is absent");
}
[[nodiscard]] inline Result<DecodedSource> decoded_source(const DecodedInstruction& instruction, SourceRole role) {
    std::optional<DecodedSource> found;
    for (const auto& source : instruction.sources) if (source.role == role) {
        if (found) return diagnose(Phase::decode, DiagnosticCode::malformed_input,
            {instruction.pc, instruction.name}, "Duplicate decoded source role");
        found = source;
    }
    if (found) return *found;
    return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input,
        {instruction.pc, instruction.name}, "Required decoded source role is absent");
}
struct SchedulingWord { Pc pc{}; std::uint64_t raw{}; };
enum class TokenKind : std::uint8_t { sync, break_target, continue_target };
struct ReconvergenceToken {
    TokenKind kind{TokenKind::sync};
    Pc push_pc{};
    BlockId target;
    std::vector<BlockId> consumers;
};
struct TokenStackState {
    std::vector<TokenId> stack; // Bottom to top; each element retains its kind.
};
struct TokenTransition {
    StackStateId incoming, outgoing;
    BlockId successor;
    std::optional<TokenId> consumed;
};
struct ReconvergenceTarget {
    Pc pop_pc{};
    TokenId token;
    BlockId target;
    StackStateId incoming;
    std::uint32_t selector{}; // Stable per-consumer selector for IR dispatch.
};
struct DecodedBlock {
    FunctionId function;
    Pc begin_pc{}, end_pc{}; // Half-open interval, including intervening scheduling words.
    std::vector<DecodedInstruction> instructions;
    std::vector<BlockId> predecessors, successors;
    std::vector<TokenId> pushes;
    std::vector<StackStateId> incoming_stacks;
    std::vector<TokenTransition> transitions;
    std::vector<ReconvergenceTarget> reconvergence;
};
struct DecodedFunction {
    Pc entry_pc{};
    BlockId entry;
    std::vector<BlockId> blocks;
    std::vector<FunctionId> callers;
};
struct DecodedProgram {
    Stage stage{Stage::vertex};
    Pc code_bytes{};
    FunctionId entry;
    Limits limits;
    std::vector<DecodedFunction> functions;
    std::vector<DecodedBlock> blocks;
    std::vector<ReconvergenceToken> tokens;
    std::vector<TokenStackState> stacks;
    std::vector<SchedulingWord> scheduling;
    std::vector<Dependency> dependencies;
    // Original stage header retains the fragment export register map.
    std::array<std::byte, 0x50> sph{};
};
// Returns the decoded fallthrough, or an invalid ID when execution reaches the
// stream end. Scheduling words have no executable block of their own.
[[nodiscard]] inline Result<BlockId> decoded_fallthrough(const DecodedProgram& program, BlockId block) {
    if (!block.valid() || block.value >= program.blocks.size()) {
        return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input, {}, "Unknown decoded block");
    }
    const auto& source = program.blocks[block.value];
    auto next_pc = source.end_pc;
    if (next_pc % 32 == 0) next_pc += 8;
    BlockId result;
    for (const auto successor : source.successors) {
        if (!successor.valid() || successor.value >= program.blocks.size() ||
            program.blocks[successor.value].function != source.function) {
            return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input, {}, "Invalid decoded successor");
        }
        if (program.blocks[successor.value].begin_pc != next_pc) continue;
        if (result.valid()) {
            return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input, {}, "Ambiguous decoded fallthrough");
        }
        result = successor;
    }
    if (!result.valid() && next_pc < program.code_bytes) {
        return diagnose(Phase::emit_ir, DiagnosticCode::invalid_ir, {}, "Missing decoded fallthrough successor");
    }
    return result;
}
// Lanes 1/2 supply disjoint decoder tables; lane 3 owns decode and CFG discovery.
using DecodeInstruction = Result<DecodedInstruction> (*)(Pc, RawInstruction);
struct DecodeRule {
    std::uint64_t mask{}, value{};
    DecodeInstruction decode{};
};
[[nodiscard]] std::span<const DecodeRule> alu_decode_rules();
[[nodiscard]] std::span<const DecodeRule> resource_decode_rules();
// The flow lane merges these spans, rejects overlapping rules and unknown encodings.
// Scheduling words (pc % 32 == 0) must never be passed to a DecodeRule.
[[nodiscard]] Result<DecodedProgram> decode(const StageImage&, FactReader&, Limits);
static_assert(RawInstruction{0x80}.signed_bits<0, 8>() == -128);
static_assert(RawInstruction{0xffff}.bits<4, 4>() == 15);
static_assert(RawInstruction{~std::uint64_t{0}}.signed_bits<0, 64>() == -1);
} // namespace armigo::shader::pipeline
