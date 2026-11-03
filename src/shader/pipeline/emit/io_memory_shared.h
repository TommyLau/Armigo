// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Emitter-lane conventions shared by the io_memory_* callbacks:
//
// 1. Registers are consumed through DecodedInstruction::registers by role, and
//    register sources through DecodedInstruction::sources; the decoder lanes own
//    that population. Family selector bits that the frozen DecodedField
//    vocabulary cannot express (saturation, IPA multiply mode, indexed and
//    output-space flags, vector size selectors) are read here from
//    DecodedInstruction::raw at the positions below. Emitters never reinterpret
//    a shared field.
// 2. Store instructions encode their stored value in the guest D slot. The
//    decoder publishes it as a source-B register access. Legacy synthetic
//    contexts may also describe the read using the destination role.
// 3. Memory and IO addresses are bytes. MemoryInfo::byte_offset and
//    IoInfo::location hold the static part; the register address operand, when
//    present, is passed as the dynamic_byte_offset input and holds the full
//    dynamic byte displacement (for physical attribute access, relative to
//    generic attribute zero). Memory register addresses and signed immediates
//    are added at guest word width before downstream bounds are proved.
// 4. Vector accesses emit one result word per guest register in ascending
//    address order.

#pragma once
#include "pipeline/decoder.h"
#include "pipeline/dispatch.h"

#include <optional>
#include <string>

namespace armigo::shader::pipeline::io_memory {

// --- guest selector layouts (bit positions from the instruction encodings) ---

// ALD/AST: attribute load/store (InstAld / InstAst).
[[nodiscard]] inline std::uint8_t attribute_dest_index(const RawInstruction& raw) {
    return static_cast<std::uint8_t>(raw.bits<0, 8>());
}
[[nodiscard]] inline std::uint8_t attribute_address_index(const RawInstruction& raw) {
    return static_cast<std::uint8_t>(raw.bits<8, 8>());
}
[[nodiscard]] inline std::uint32_t attribute_byte_offset(const RawInstruction& raw) {
    return static_cast<std::uint32_t>(raw.bits<20, 11>());
}
[[nodiscard]] inline bool attribute_per_patch(const RawInstruction& raw) {
    return raw.bits<31, 1>() != 0;
}
[[nodiscard]] inline bool attribute_output_space(const RawInstruction& raw) {
    return raw.bits<32, 1>() != 0;
}
[[nodiscard]] inline unsigned attribute_words(const RawInstruction& raw) {
    return static_cast<unsigned>(raw.bits<47, 2>()) + 1;
}
// Physical addressing: the address register supplies the byte offset and the
// immediate is zero.
[[nodiscard]] inline bool attribute_physical(const RawInstruction& raw) {
    return !attribute_per_patch(raw) && attribute_byte_offset(raw) == 0 &&
           attribute_address_index(raw) != 255;
}

// IPA: interpolator (InstIpa).
enum class IpaOperation : std::uint8_t { pass, multiply, constant, sc };
[[nodiscard]] inline IpaOperation ipa_operation(const RawInstruction& raw) {
    return static_cast<IpaOperation>(raw.bits<54, 2>());
}
[[nodiscard]] inline bool ipa_saturate(const RawInstruction& raw) {
    return raw.bits<51, 1>() != 0;
}
[[nodiscard]] inline bool ipa_indexed(const RawInstruction& raw) {
    return raw.bits<38, 1>() != 0;
}
[[nodiscard]] inline std::uint32_t ipa_byte_offset(const RawInstruction& raw) {
    return static_cast<std::uint32_t>(raw.bits<28, 10>());
}

// LDC: constant buffer load (InstLdc). LsSize2 selector values: 0 u8, 1 s8,
// 2 u16, 3 s16, 4 b32, 5 b64, 6 b128 (invalid for LDC).
[[nodiscard]] inline std::uint8_t constant_slot(const RawInstruction& raw) {
    return static_cast<std::uint8_t>(raw.bits<36, 5>());
}
[[nodiscard]] inline std::uint8_t load_store_size(const RawInstruction& raw) {
    return static_cast<std::uint8_t>(raw.bits<48, 3>());
}
[[nodiscard]] inline std::uint8_t constant_address_mode(const RawInstruction& raw) {
    return static_cast<std::uint8_t>(raw.bits<44, 2>());
}
// Address modes 2 (Is) and 3 (Isl) index the slot with the high half of SrcA
// and mask the address to its low half.
[[nodiscard]] inline bool constant_slot_indexed(std::uint8_t mode) {
    return mode == 2 || mode == 3;
}

// LDL/STL/LDS/STS/LDG/STG: LsSize selectors (0 u8, 1 s8, 2 u16, 3 s16, 4 b32,
// 5 b64, 6 b128, 7 ub128). Signed 24-bit byte displacements.
[[nodiscard]] inline std::uint32_t memory_byte_displacement(const RawInstruction& raw) {
    return static_cast<std::uint32_t>(raw.bits<20, 24>());
}
// LDG/STG extended (E) addressing forms a 40-bit address with the next
// register; bit 45 on both encodings.
[[nodiscard]] inline bool global_extended(const RawInstruction& raw) {
    return raw.bits<45, 1>() != 0;
}

// Size selector interpretation shared by the memory loads/stores.
struct SizeShape {
    ScalarType scalar{ScalarType::u32};
    std::uint8_t element_bytes{4};
    std::uint8_t elements{1};
};
[[nodiscard]] constexpr SizeShape size_shape(std::uint8_t size) {
    switch (size) {
    case 0: return {ScalarType::u8, 1, 1};
    case 1: return {ScalarType::s8, 1, 1};
    case 2: return {ScalarType::u16, 2, 1};
    case 3: return {ScalarType::s16, 2, 1};
    case 5: return {ScalarType::u32, 4, 2};
    case 6: return {ScalarType::u32, 4, 4};
    case 7: return {ScalarType::u32, 4, 4};
    default: return {ScalarType::u32, 4, 1};
    }
}

// BAR/MEMBAR selectors (InstBar / InstMembar). BAR operation 0 is sync.
[[nodiscard]] inline std::uint8_t barrier_operation(const RawInstruction& raw) {
    return static_cast<std::uint8_t>(raw.bits<32, 3>());
}
[[nodiscard]] inline std::uint8_t memory_barrier_kind(const RawInstruction& raw) {
    return static_cast<std::uint8_t>(raw.bits<8, 2>());
}

// --- operand helpers ---

// Non-failing field probe for optional selectors; decoded_field() treats an
// absent field as an error, which is wrong for genuinely optional bits.
[[nodiscard]] inline std::optional<std::uint64_t> optional_field(
    const DecodedInstruction& instruction, DecodedField field) {
    for (const auto& item : instruction.fields) {
        if (item.field == field) return item.bits;
    }
    return std::nullopt;
}

[[nodiscard]] inline const RegisterAccess* find_register(
    const DecodedInstruction& instruction, RegisterRole role) {
    for (const auto& access : instruction.registers) {
        if (access.role == role) return &access;
    }
    return nullptr;
}

// Register sources must be plain GPR reads; immediates here are malformed input.
[[nodiscard]] inline Result<Register> source_register(
    const DecodedInstruction& instruction, SourceRole role) {
    auto source = decoded_source(instruction, role);
    if (const auto* error = std::get_if<Diagnostic>(&source)) return *error;
    const auto& value = std::get<DecodedSource>(source).value;
    if (const auto* reg = std::get_if<Register>(&value)) return *reg;
    return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input,
        {instruction.pc, instruction.name}, "Memory source operand is not a register");
}

// Stored values travel in the guest D slot; see the header conventions.
[[nodiscard]] inline Result<Register> store_source_register(const DecodedInstruction& instruction) {
    if (const auto* access = find_register(instruction, RegisterRole::source_b)) {
        if (access->reg.file != RegisterFile::gpr || access->access != Access::read) {
            return diagnose(Phase::emit_ir, DiagnosticCode::malformed_input,
                {instruction.pc, instruction.name}, "Store data must be a GPR read");
        }
        return access->reg;
    }
    if (const auto* access = find_register(instruction, RegisterRole::destination)) {
        return access->reg;
    }
    return source_register(instruction, SourceRole::b);
}

[[nodiscard]] inline bool is_zero_literal(const Operand& operand) {
    const auto* literal = std::get_if<LiteralBits>(&operand);
    return literal != nullptr && literal->bits == 0;
}

// Context register read for a GPR index; the context resolves RZ to zero.
[[nodiscard]] inline Result<Operand> read_gpr(EmitterContext& context, std::uint16_t index) {
    return context.read(Register{RegisterFile::gpr, index});
}

// Address arithmetic uses raw 32-bit bits; the MSL lane applies the space base.
constexpr NumericMode address_mode{Domain::bits, ScalarType::u32};
// Guest floating arithmetic is IEEE round-to-nearest-even; FTZ is a modifier
// lanes 6/7 decode, and no io_memory instruction carries one.
constexpr NumericMode float_multiply{Domain::floating, ScalarType::f32, Rounding::nearest_even};

[[nodiscard]] constexpr NumericMode interpretation(ScalarType scalar) {
    return NumericMode{scalar_domain(scalar), scalar,
        scalar_domain(scalar) == Domain::floating ? Rounding::nearest_even : Rounding::not_applicable, false,
        false, false, false};
}

// Literal helpers.
[[nodiscard]] inline Operand word_literal(std::uint32_t bits) {
    return LiteralBits{bits, ValueType::word32};
}
// A static RZ address needs no operation. A live base must include the signed
// displacement before widening to the host buffer's byte-address arithmetic.
[[nodiscard]] inline Status memory_address(EmitterContext& context, MemoryInfo& info,
                                           Operand base, std::vector<Operand>& inputs) {
    if (is_zero_literal(base)) return std::monostate{};
    if (info.byte_offset != 0) {
        auto sum = context.binary(Opcode::add, base, word_literal(info.byte_offset),
            NumericMode{Domain::unsigned_integer, ScalarType::u32}, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&sum)) return *error;
        base = std::get<Operand>(sum);
    }
    info.byte_offset = 0;
    info.dynamic_byte_offset = InputIndex{static_cast<std::uint16_t>(inputs.size())};
    inputs.push_back(base);
    return std::monostate{};
}
[[nodiscard]] inline Operand float_literal(float value) {
    return LiteralBits{std::bit_cast<std::uint32_t>(value), ValueType::word32};
}

[[nodiscard]] inline Diagnostic fail(EmitterContext& context, DiagnosticCode code,
                                     std::string message) {
    return context.error(code, std::move(message));
}

// Generic attribute zero is at guest byte offset 0x80; there are 32 vec4 slots.
inline constexpr std::uint32_t user_attribute_base = 0x80;
inline constexpr std::uint32_t user_attribute_end = user_attribute_base + 32 * 16;

} // namespace armigo::shader::pipeline::io_memory
