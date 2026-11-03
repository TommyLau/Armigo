// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Global memory loads/stores retain their address pair until descriptor
// provenance is resolved into a storage binding.

#include "io_memory_shared.h"

#include <vector>

namespace armigo::shader::pipeline {
namespace io_memory {
Status global_address(EmitterContext& context, MemoryInfo& info, Operand low,
                      std::vector<Operand>& inputs) {
    if (!global_extended(context.instruction().raw))
        return memory_address(context, info, low, inputs);
    const auto base = attribute_address_index(context.instruction().raw);
    if (base >= 254)
        return fail(context, DiagnosticCode::unresolved_resource,
            "Extended global address requires two nonzero GPRs with descriptor provenance");
    auto high = read_gpr(context, static_cast<std::uint16_t>(base + 1));
    if (const auto* error = std::get_if<Diagnostic>(&high)) return *error;
    info.dynamic_byte_offset = InputIndex{static_cast<std::uint16_t>(inputs.size())};
    inputs.push_back(low);
    info.global_address_high = InputIndex{static_cast<std::uint16_t>(inputs.size())};
    inputs.push_back(std::get<Operand>(high));
    return std::monostate{};
}

// LDG: global memory load.
Status emit_load_global_impl(EmitterContext& context) {
    const DecodedInstruction& inst = context.instruction();
    const SizeShape shape = size_shape(load_store_size(inst.raw));
    const auto byte_offset = static_cast<std::uint32_t>(inst.raw.signed_bits<20, 24>());
    const auto* destination = find_register(inst, RegisterRole::destination);
    if (destination == nullptr || destination->reg.file != RegisterFile::gpr) {
        return fail(context, DiagnosticCode::malformed_input,
            "LDG is missing its GPR destination register access");
    }
    auto address = read_gpr(context, attribute_address_index(inst.raw));
    if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
    const Operand& source = std::get<Operand>(address);

    MemoryInfo info;
    info.space = AddressSpace::storage;
    info.access = Access::read;
    info.byte_offset = byte_offset;
    info.element_bytes = shape.element_bytes;
    info.elements = shape.elements;
    info.alignment = shape.element_bytes;
    std::vector<Operand> inputs;
    if (auto status = global_address(context, info, source, inputs);
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

// STG: global memory store.
Status emit_store_global_impl(EmitterContext& context) {
    const DecodedInstruction& inst = context.instruction();
    const SizeShape shape = size_shape(load_store_size(inst.raw));
    const auto byte_offset = static_cast<std::uint32_t>(inst.raw.signed_bits<20, 24>());
    auto value_register = store_source_register(inst);
    if (const auto* error = std::get_if<Diagnostic>(&value_register)) return *error;
    const auto value_base = std::get<Register>(value_register).index;
    auto address = read_gpr(context, attribute_address_index(inst.raw));
    if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
    const Operand& source = std::get<Operand>(address);

    MemoryInfo info;
    info.space = AddressSpace::storage;
    info.access = Access::write;
    info.byte_offset = byte_offset;
    info.element_bytes = shape.element_bytes;
    info.elements = shape.elements;
    info.alignment = shape.element_bytes;
    std::vector<Operand> inputs;
    if (auto status = global_address(context, info, source, inputs);
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
} // namespace io_memory

Status emit_load_global(EmitterContext& context) {
    return io_memory::emit_load_global_impl(context);
}

Status emit_store_global(EmitterContext& context) {
    return io_memory::emit_store_global_impl(context);
}

} // namespace armigo::shader::pipeline
