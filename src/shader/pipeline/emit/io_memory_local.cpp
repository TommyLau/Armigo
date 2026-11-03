// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Local and shared memory loads/stores (LDL/STL, LDS/STS) plus the generic
// LD/ST diagnostic. Narrow accesses use byte-addressed elements with the
// sign/zero interpretation in the numeric mode.

#include "io_memory_shared.h"
#include "pipeline/decode/resource_fields.h"

#include <vector>

namespace armigo::shader::pipeline {
namespace {

using namespace io_memory;

[[nodiscard]] MemoryInfo memory_info(AddressSpace space, Access access, std::uint32_t byte_offset,
                                     const SizeShape& shape) {
    MemoryInfo info;
    info.space = space;
    info.access = access;
    info.byte_offset = byte_offset;
    info.element_bytes = shape.element_bytes;
    info.elements = shape.elements;
    info.alignment = shape.element_bytes;
    return info;
}

// Shared load shape for LDL/LDS: SrcA holds the dynamic byte displacement.
[[nodiscard]] Status emit_sized_load(EmitterContext& context, AddressSpace space) {
    const DecodedInstruction& inst = context.instruction();
    const std::uint8_t size = load_store_size(inst.raw);
    // B128 (6) is the widest encoded size; selector 7 is invalid here.
    if (size > 6) {
        return fail(context, DiagnosticCode::unsupported_modifier, "Invalid load size selector");
    }
    const SizeShape shape = size_shape(size);
    const auto byte_offset = static_cast<std::uint32_t>(inst.raw.signed_bits<20, 24>());
    const auto* destination = find_register(inst, RegisterRole::destination);
    if (destination == nullptr || destination->reg.file != RegisterFile::gpr) {
        return fail(context, DiagnosticCode::malformed_input,
            "Memory load is missing its GPR destination register access");
    }
    if (space == AddressSpace::shared && unsigned(destination->reg.index) + shape.elements > 256)
        return fail(context, DiagnosticCode::unsupported_modifier, "Shared load vector crosses the guest register file");
    auto address = read_gpr(context, attribute_address_index(inst.raw));
    if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
    const Operand& source = std::get<Operand>(address);

    MemoryInfo info = memory_info(space, Access::read, byte_offset, shape);
    std::vector<Operand> inputs;
    if (auto status = memory_address(context, info, source, inputs);
        !std::holds_alternative<std::monostate>(status)) return status;
    const std::vector<ValueType> result_types(shape.elements, ValueType::word32);
    auto loaded = context.load(info, inputs, interpretation(shape.scalar), result_types);
    if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
    const auto& words = std::get<std::vector<ValueId>>(loaded);
    for (unsigned index = 0; index < words.size(); ++index) {
        const auto dest = static_cast<std::uint16_t>(destination->reg.index + index);
        if (dest >= 255) break;
        if (auto status = context.write(Register{RegisterFile::gpr, dest},
                Operand{words[index]}); !std::holds_alternative<std::monostate>(status)) {
            return status;
        }
    }
    return std::monostate{};
}

// Shared store shape for STL/STS: the stored values travel in consecutive D
// slot registers (see store_source_register).
[[nodiscard]] Status emit_sized_store(EmitterContext& context, AddressSpace space) {
    const DecodedInstruction& inst = context.instruction();
    const std::uint8_t size = load_store_size(inst.raw);
    if (size > 6) {
        return fail(context, DiagnosticCode::unsupported_modifier, "Invalid store size selector");
    }
    const SizeShape shape = size_shape(size);
    const auto byte_offset = static_cast<std::uint32_t>(inst.raw.signed_bits<20, 24>());
    auto value_register = store_source_register(inst);
    if (const auto* error = std::get_if<Diagnostic>(&value_register)) return *error;
    const auto value_base = std::get<Register>(value_register).index;
    if (space == AddressSpace::shared && unsigned(value_base) + shape.elements > 256)
        return fail(context, DiagnosticCode::unsupported_modifier, "Shared store vector crosses the guest register file");
    auto address = read_gpr(context, attribute_address_index(inst.raw));
    if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
    const Operand& source = std::get<Operand>(address);

    MemoryInfo info = memory_info(space, Access::write, byte_offset, shape);
    std::vector<Operand> inputs;
    if (auto status = memory_address(context, info, source, inputs);
        !std::holds_alternative<std::monostate>(status)) return status;
    for (unsigned index = 0; index < shape.elements; ++index) {
        const auto source_index = static_cast<std::uint16_t>(
            value_base + index < 255 ? value_base + index : 255);
        auto value = context.read(Register{RegisterFile::gpr, source_index});
        if (const auto* error = std::get_if<Diagnostic>(&value)) return *error;
        info.data.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())});
        inputs.push_back(std::get<Operand>(value));
    }
    return context.store(info, inputs, interpretation(shape.scalar));
}

} // namespace

// LDL: local memory load.
Status emit_load_local(EmitterContext& context) {
    return emit_sized_load(context, AddressSpace::local);
}

// STL: local memory store.
Status emit_store_local(EmitterContext& context) {
    return emit_sized_store(context, AddressSpace::local);
}

// Shared memory belongs to the compute workgroup, with the same byte layout as local memory.
// The LDS encoding carries a U bit at position 44, but the emitter never reads
// it, so the access is the ordinary shared load either way instead of rejecting
// the modifier.
Status emit_load_shared(EmitterContext& context) {
    if (context.decoded_program().stage != Stage::compute)
        return fail(context, DiagnosticCode::unsupported_instruction, "LDS requires a compute stage");
    return emit_sized_load(context, AddressSpace::shared);
}
Status emit_store_shared(EmitterContext& context) {
    if (context.decoded_program().stage != Stage::compute)
        return fail(context, DiagnosticCode::unsupported_instruction, "STS requires a compute stage");
    return emit_sized_store(context, AddressSpace::shared);
}

// ATOMS uses an aligned word address plus a signed immediate measured in words.
Status emit_shared_atomic(EmitterContext& context) {
    if (context.decoded_program().stage != Stage::compute)
        return fail(context, DiagnosticCode::unsupported_capability, "ATOMS requires a compute stage");
    const auto& inst = context.instruction();
    const auto size = resource_decode::atoms_size_selector(inst.raw);
    auto operation_field = decoded_field(inst, DecodedField::atomic_operation);
    if (const auto* error = std::get_if<Diagnostic>(&operation_field)) return *error;
    const auto operation = std::get<std::uint64_t>(operation_field);
    if (size > 1 || operation > 8 || operation == 3 || operation == 4)
        return fail(context, DiagnosticCode::unsupported_modifier, "ATOMS requires a supported 32-bit integer operation");
    constexpr AtomicKind kinds[] = {AtomicKind::add, AtomicKind::minimum, AtomicKind::maximum,
        AtomicKind::increment, AtomicKind::decrement, AtomicKind::bit_and, AtomicKind::bit_or,
        AtomicKind::bit_xor, AtomicKind::exchange};
    auto address = read_gpr(context, attribute_address_index(inst.raw));
    if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
    auto aligned = context.binary(Opcode::bit_and, std::get<Operand>(address), word_literal(0xfffffffcu),
                                  address_mode, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&aligned)) return *error;
    MemoryInfo info;
    info.space = AddressSpace::shared; info.access = Access::read_write;
    auto displacement_field = decoded_field(inst, DecodedField::memory_byte_offset);
    if (const auto* error = std::get_if<Diagnostic>(&displacement_field)) return *error;
    const auto displacement = std::bit_cast<std::int64_t>(std::get<std::uint64_t>(displacement_field));
    if (displacement < 0 || displacement > UINT32_MAX || displacement % 4 != 0)
        return fail(context, DiagnosticCode::unsupported_modifier,
            "ATOMS negative word displacement needs a signed word-index address contract");
    // Keep the word displacement separate: wrapping a byte sum at 32 bits
    // could turn an invalid shared word index into a valid low address.
    info.byte_offset = static_cast<std::uint32_t>(displacement);
    info.alignment = 4; info.atomic = kinds[operation]; info.scope = MemoryScope::workgroup;
    info.dynamic_byte_offset = InputIndex{0};
    std::vector<Operand> inputs{std::get<Operand>(aligned)};
    auto data = read_gpr(context, static_cast<std::uint16_t>(inst.raw.bits<20, 8>()));
    if (const auto* error = std::get_if<Diagnostic>(&data)) return *error;
    info.data.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())});
    inputs.push_back(std::get<Operand>(data));
    const ValueType result_type = ValueType::word32;
    auto result = context.emit(Opcode::atomic, interpretation(size == 1 ? ScalarType::s32 : ScalarType::u32),
                               inputs, std::span{&result_type, 1}, info, Effects::none);
    if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
    return context.write(Register{RegisterFile::gpr, static_cast<std::uint16_t>(inst.raw.bits<0, 8>())},
                         Operand{std::get<std::vector<ValueId>>(result).front()});
}

// LD/ST generic-space forms are unimplemented and return a structured
// diagnostic.
Status emit_generic_memory_unimplemented(EmitterContext& context) {
    return fail(context, DiagnosticCode::unsupported_instruction,
        "Generic-space LD/ST has no implemented emitter");
}

} // namespace armigo::shader::pipeline
