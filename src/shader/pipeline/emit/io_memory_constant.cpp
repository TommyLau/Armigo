// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// LDC: constant buffer load with direct and slot-indexed addressing. Indexed
// slots select among shader-addressable buffers independently of draw bindings.
// Each candidate is guarded by the runtime constant-buffer read-window policy.

#include "io_memory_shared.h"

#include <vector>

namespace armigo::shader::pipeline {
namespace {

using namespace io_memory;

[[nodiscard]] MemoryInfo constant_load_info(std::uint8_t slot, std::uint32_t byte_offset,
                                            const SizeShape& shape) {
    MemoryInfo info;
    info.space = AddressSpace::constant_buffer;
    info.access = Access::read;
    info.constant_buffer_slot = slot;
    info.byte_offset = byte_offset;
    info.element_bytes = shape.element_bytes;
    info.elements = shape.elements;
    info.alignment = shape.element_bytes;
    return info;
}

} // namespace

// LDC: constant buffer load.
Status emit_constant_load(EmitterContext& context) {
    const DecodedInstruction& inst = context.instruction();
    const std::uint8_t size = load_store_size(inst.raw);
    // B128 and beyond is invalid for LDC; only sizes through B64 are admitted.
    if (size > 5) {
        return fail(context, DiagnosticCode::unsupported_modifier, "Invalid LDC size selector");
    }
    const std::uint8_t slot = constant_slot(inst.raw);
    // Fact keys describe constant buffer slots 0..17; larger encoded slots are
    // outside the pipeline slot range.
    if (slot > 17) {
        return fail(context, DiagnosticCode::unsupported_capability,
            "LDC slot exceeds the pipeline constant buffer slot range");
    }
    // The 16-bit immediate is a signed byte displacement.
    const auto byte_offset = static_cast<std::uint32_t>(
        inst.raw.signed_bits<20, 16>());
    const SizeShape shape = size_shape(size);

    auto address = read_gpr(context, attribute_address_index(inst.raw));
    if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
    const Operand& source = std::get<Operand>(address);
    const auto* destination = find_register(inst, RegisterRole::destination);
    if (destination == nullptr || destination->reg.file != RegisterFile::gpr) {
        return fail(context, DiagnosticCode::malformed_input,
            "LDC is missing its GPR destination register access");
    }

    const std::vector<ValueType> result_types(shape.elements, ValueType::word32);

    if (constant_slot_indexed(constant_address_mode(inst.raw))) {
        // Slot-indexed access: the high half of SrcA selects the buffer, the
        // low half addresses inside it.
        auto high = context.binary(Opcode::shift_right, source, word_literal(16),
            address_mode, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&high)) return *error;
        auto slot_value = context.binary(Opcode::add, word_literal(slot),
            std::get<Operand>(high), interpretation(ScalarType::u32), ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&slot_value)) return *error;
        auto masked = context.binary(Opcode::bit_and, source, word_literal(0xffff),
            address_mode, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&masked)) return *error;

        // Every addressable candidate has a guarded load and valid backing,
        // including unbound draw slots. The selected value is zero for a
        // selector outside the supported slot range.
        std::vector<Operand> selected(shape.elements, word_literal(0));
        for (std::uint8_t candidate = 0; candidate < 18; ++candidate) {
            MemoryInfo info = constant_load_info(candidate, byte_offset, shape);
            std::vector<Operand> probe;
            if (auto status = memory_address(context, info, std::get<Operand>(masked), probe);
                !std::holds_alternative<std::monostate>(status)) return status;
            auto loaded = context.load(info, probe,
                interpretation(shape.scalar), result_types);
            if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
            const auto& words = std::get<std::vector<ValueId>>(loaded);
            const Operand& slot_operand = std::get<Operand>(slot_value);
            auto equals = context.compare(slot_operand, word_literal(candidate),
                ComparisonInfo{Compare::equal, false}, interpretation(ScalarType::u32));
            if (const auto* error = std::get_if<Diagnostic>(&equals)) return *error;
            for (std::size_t lane = 0; lane < words.size(); ++lane) {
                auto chosen = context.select(std::get<Operand>(equals), words[lane],
                    selected[lane], ValueType::word32);
                if (const auto* error = std::get_if<Diagnostic>(&chosen)) return *error;
                selected[lane] = std::get<Operand>(chosen);
            }
        }
        for (unsigned index = 0; index < shape.elements; ++index) {
            const auto dest = static_cast<std::uint16_t>(destination->reg.index + index);
            if (dest >= 255) break;
            if (auto status = context.write(Register{RegisterFile::gpr, dest}, selected[index]);
                !std::holds_alternative<std::monostate>(status)) return status;
        }
        return std::monostate{};
    }

    MemoryInfo info = constant_load_info(slot, byte_offset, shape);
    std::vector<Operand> inputs;
    if (auto status = memory_address(context, info, source, inputs);
        !std::holds_alternative<std::monostate>(status)) return status;
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

} // namespace armigo::shader::pipeline
