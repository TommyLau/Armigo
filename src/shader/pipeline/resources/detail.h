// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The shared-memory alignment proof is independently implemented for Armigo IR.

#pragma once
#include "../phases.h"
#include <algorithm>
#include <bit>

namespace armigo::shader::pipeline::resources {
[[nodiscard]] Status check_profile(const MetalProfile&, Phase);
[[nodiscard]] Status check_requirements(const Program&, const ResourceRequirements&,
                                        const MetalProfile&, Phase);
// Derives the float/uint/sint class of every directly addressed generic vertex
// input from IR uses (resources/vertex_class.cpp), rewriting each load to its
// class interpretation. Runs inside legalize after the cleanup passes.
[[nodiscard]] Status classify_vertex_inputs(Program&);
[[nodiscard]] bool valid_descriptor(const TextureDescriptor&);
[[nodiscard]] bool needs_sampler(const Operation&);
[[nodiscard]] bool is_image(const Operation&);
[[nodiscard]] Status check_texture(const Operation&, const TextureDescriptor&, const MetalProfile&, Stage, Phase);
[[nodiscard]] inline bool covers(Access available, Access required) {
    return available == Access::read_write || available == required;
}
[[nodiscard]] inline Access combine(Access first, Access second) {
    return first == second ? first : Access::read_write;
}
// Descriptor inputs are explicit loads introduced by legalization. Recheck
// their identities at later phase boundaries instead of trusting size metadata.
inline bool storage_address_inputs(const Program& p, const Operation& op,
                                   const StorageAddress& a, const StorageBufferSource& source,
                                   std::uint64_t& work) {
    const auto& m = std::get<MemoryInfo>(op.info);
    if (a.descriptor != source.descriptor ||
        !m.dynamic_byte_offset || m.dynamic_byte_offset_upper_bound || m.array_index) return false;
    const auto key = source.descriptor;
    if (key.cbuf_slot >= constant_buffer_slot_count || key.handle_byte_offset % 8 != 0 ||
        key.handle_byte_offset > constant_buffer_window_bytes - 12) return false;
    const InputIndex roles[]{a.base_low, a.base_high, a.byte_size};
    for (unsigned i = 0; i < 3; ++i) {
        if (roles[i].value >= op.inputs.size()) return false;
        const auto* value = std::get_if<ValueId>(&op.inputs[roles[i].value]);
        if (!value) return false;
        bool found = false;
        for (const auto& definition : p.operations) {
            if (++work > p.limits.analysis_steps) return false;
            for (const auto& output : definition.outputs) {
                const auto* id = std::get_if<ValueId>(&output);
                if (!id || *id != *value) continue;
                const auto* load = std::get_if<MemoryInfo>(&definition.info);
                if (definition.opcode != Opcode::load || !load ||
                    load->space != AddressSpace::constant_buffer || load->constant_buffer_slot != key.cbuf_slot ||
                    load->byte_offset != key.handle_byte_offset + i * 4 ||
                    load->dynamic_byte_offset || load->element_bytes != 4 || load->elements != 1 ||
                    load->is_volatile || definition.outputs.size() != 1) return false;
                found = true;
            }
        }
        if (!found) return false;
    }
    return true;
}
inline bool storage_descriptor_inputs(const Program& p, const Operation& op,
                                      const StorageBufferSource& source, std::uint64_t& work) {
    const auto& m = std::get<MemoryInfo>(op.info);
    return m.storage_address && storage_address_inputs(p, op, *m.storage_address, source, work);
}
// Only naturally aligned accesses are admitted until unaligned shared-memory
// behavior has an established contract. Check every reaching mutable definition.
inline bool shared_address_aligned(const Program& p, const Operand& value, unsigned alignment,
                                   std::uint64_t& work, unsigned depth = 0) {
    if (++work > p.limits.analysis_steps || depth > std::min(p.limits.nesting_depth, 256U)) return false;
    if (alignment == 1) return true;
    if (const auto* literal = std::get_if<LiteralBits>(&value))
        return literal->type == ValueType::word32 && literal->bits % alignment == 0;
    const auto* id = std::get_if<ValueId>(&value);
    if (!id || id->value >= p.values.size()) return false;
    auto aligned = [&](const Operand& operand, unsigned required) {
        return shared_address_aligned(p, operand, required, work, depth + 1);
    };
    bool found = false;
    for (const auto& op : p.operations) {
        if (++work > p.limits.analysis_steps) return false;
        if (std::none_of(op.outputs.begin(), op.outputs.end(), [&](const auto& output) {
            const auto* result = std::get_if<ValueId>(&output); return result && *result == *id;
        })) continue;
        found = true;
        if (op.numeric.scalar != ScalarType::u32 ||
            (op.numeric.domain != Domain::bits && op.numeric.domain != Domain::unsigned_integer)) return false;
        if (op.opcode == Opcode::copy && op.inputs.size() == 1) {
            if (!aligned(op.inputs[0], alignment)) return false;
        } else if (op.opcode == Opcode::select && op.inputs.size() == 3) {
            if (!aligned(op.inputs[1], alignment) || !aligned(op.inputs[2], alignment)) return false;
        } else if (op.inputs.size() == 2) {
            if (op.opcode == Opcode::bit_and) {
                if (!aligned(op.inputs[0], alignment) && !aligned(op.inputs[1], alignment)) return false;
            } else if (op.opcode == Opcode::shift_left) {
                const auto* count = std::get_if<LiteralBits>(&op.inputs[1]);
                if (!count) return false;
                const auto shift = unsigned(count->bits & 31);
                const auto factor = shift >= 2 ? 4u : 1u << shift;
                if (factor < alignment && !aligned(op.inputs[0], alignment / factor)) return false;
            } else if (op.opcode == Opcode::multiply) {
                bool product = false;
                for (unsigned side = 0; side < 2; ++side) {
                    if (const auto* factor = std::get_if<LiteralBits>(&op.inputs[side])) {
                        unsigned remaining = alignment;
                        while (remaining > 1 && factor->bits % (alignment / remaining * 2) == 0) remaining /= 2;
                        if (aligned(op.inputs[1 - side], remaining)) product = true;
                    }
                }
                if (!product) return false;
            } else if (op.opcode == Opcode::add || op.opcode == Opcode::subtract ||
                       op.opcode == Opcode::bit_or || op.opcode == Opcode::bit_xor) {
                if (!aligned(op.inputs[0], alignment) || !aligned(op.inputs[1], alignment)) return false;
            } else return false;
        } else return false;
    }
    for (const auto& block : p.blocks) for (const auto& phi : block.phis) {
        if (++work > p.limits.analysis_steps) return false;
        if (phi.output != *id) continue;
        found = true;
        for (const auto& input : phi.inputs) if (!aligned(input.value, alignment)) return false;
    }
    return found;
}
} // namespace armigo::shader::pipeline::resources
