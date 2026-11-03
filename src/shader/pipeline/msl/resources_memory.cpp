// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "stage_internal.h"
#include "../dispatch.h"

namespace armigo::shader::pipeline {
using namespace msl;
Status generate_stage_io(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id); const auto* info = std::get_if<IoInfo>(&op.info);
        require(info, "IO operation has no IO metadata"); const auto& io = *info;
        plain_numeric(op);
        require(!io.dynamic_byte_offset && !io.vertex_index && !io.sample_index && !io.offset_x && !io.offset_y,
                "Dynamic/pull-model IO requires explicit lowering", DiagnosticCode::unsupported_capability);
        const bool store = op.opcode == Opcode::store_io;
        require(!io.output_storage || op.opcode == Opcode::load_io || store,
                "Output storage requires an IO load or store");
        require(!io.interpolation_location_override ||
                    ((*io.interpolation_location_override == InterpolationLocation::center ||
                      *io.interpolation_location_override == InterpolationLocation::centroid) &&
                     io.interpolation.location == *io.interpolation_location_override),
                "IO interpolation location override is invalid or unresolved",
                DiagnosticCode::unsupported_modifier);
        const bool output_storage = store || io.output_storage;
        if (op.opcode == Opcode::interpolate) {
            require(op.numeric.domain == Domain::floating && op.numeric.scalar == ScalarType::f32 && op.numeric.rounding == Rounding::nearest_even,
                    "Interpolation arithmetic requires explicit binary32 nearest-even semantics", DiagnosticCode::unsupported_modifier);
            require(c.program().ir.stage == Stage::fragment && io.semantic == IoSemantic::generic && io.type.scalar == ScalarType::f32,
                    "Interpolation requires a fragment floating user attribute", DiagnosticCode::unsupported_capability);
            require(io.reciprocal_w.has_value() == (io.interpolation.mode == InterpolationMode::perspective),
                    "Perspective interpolation requires its explicit PositionW factor; linear/flat interpolation must not apply it");
        }
        require(!io.reciprocal_w || (op.opcode == Opcode::interpolate && io.type.scalar == ScalarType::f32),
                "Interpolation factor is invalid for this IO operation", DiagnosticCode::unsupported_modifier);
        require(io.component_mask && io.component_mask < 16 && !(io.component_mask >> io.type.components), "Invalid IO component mask");
        require((store ? io.data.size() : op.outputs.size()) == static_cast<std::size_t>(std::popcount(io.component_mask)), "IO component count mismatch");
        std::string factor;
        if (io.reciprocal_w) {
            factor = "interpolation_factor" + number(id.value);
            take(c.statement("const uint " + factor + " = " + input(c, op, *io.reciprocal_w, ScalarType::u32) + ";"));
        }
        std::size_t result = 0;
        // Snapshot store data before writing any destination, preserving parallel
        // component semantics when an input aliases another exported component.
        for (unsigned component = 0; component < 4; ++component) if (io.component_mask & (1u << component)) {
            const IoUse* use = nullptr;
            for (const auto& candidate : c.resources().io)
                if (candidate.semantic == io.semantic && candidate.location == io.location &&
                    candidate.access == (output_storage ? Access::write : Access::read) && (candidate.mask & (1u << component))) {
                    require(!use, "Ambiguous IO requirement"); use = &candidate;
                }
            require(use && use->type == io.type && (store || use->interpolation == io.interpolation), "IO operation disagrees with planned interface");
            const bool parameter = !output_storage && (io.semantic == IoSemantic::vertex_id || io.semantic == IoSemantic::instance_id ||
                io.semantic == IoSemantic::thread_id || io.semantic == IoSemantic::workgroup_id || io.semantic == IoSemantic::subgroup_lane_id ||
                io.semantic == IoSemantic::quad_lane_id ||
                // The y-direction word names the stage-level function-constant
                // selection, like the entry-parameter builtins above.
                io.semantic == IoSemantic::y_direction);
            const bool adaptive_color = output_storage && io.semantic == IoSemantic::color &&
                c.fragment_output_types_from_render_targets();
            std::string expression = adaptive_color
                ? "armigo_" + io_name(io.semantic, io.location) + "_raw"
                : std::string(parameter ? "" : output_storage ? "out." : "in.") + io_name(io.semantic, io.location);
            if (io.semantic == IoSemantic::subgroup_lane_id) expression = "(" + expression + " & 31u)";
            const bool flipped_point_coord = !output_storage && io.semantic == IoSemantic::point_coord &&
                                             component == 1;
            if (io.semantic == IoSemantic::generic && !(c.program().ir.stage == Stage::vertex && !output_storage)) expression += "_" + number(component);
            else if (io.type.components > 1) expression += "." + std::string(1, "xyzw"[component]);
            if (flipped_point_coord)
                expression = "(armigo_point_coord_lower_left ? 1.0f - " + expression + " : " + expression + ")";
            if (store) take(c.statement(expression + " = " + input(c, op, io.data[result], io.type.scalar) + ";"));
            else {
                if (io.reciprocal_w) {
                    take(c.require_helper("armigo_multiply_rne"));
                    expression = "as_type<float>(armigo_multiply_rne(as_type<uint>(" + expression + "), " +
                        factor + "))";
                }
                take(c.assign(output(op, result), expression, io.type.scalar));
            }
            ++result;
        }
    });
}
namespace {
void generate_shared_memory(MslContext& c, OperationId id, const Operation& op, const MemoryInfo& m) {
    require(c.program().ir.stage == Stage::compute && c.resources().compute.has_value(), "Shared memory requires compute launch facts");
    require(!m.resource && !m.constant_buffer_slot && !m.array_index && !m.is_volatile &&
            m.order == MemoryOrder::relaxed && (m.scope == MemoryScope::invocation || m.scope == MemoryScope::workgroup),
            "Shared memory modifiers are unsupported", DiagnosticCode::unsupported_modifier);
    require(op.numeric.domain != Domain::floating && scalar_bits(op.numeric.scalar) <= 32,
            "Shared memory requires an integer/raw-word interpretation", DiagnosticCode::unsupported_modifier);
    require(m.element_bytes == 1 || m.element_bytes == 2 || m.element_bytes == 4, "Unsupported shared element size", DiagnosticCode::unsupported_capability);
    const bool store = op.opcode == Opcode::store;
    const bool atomic = op.opcode == Opcode::atomic;
    require(op.opcode == Opcode::load || store || atomic, "Invalid shared operation");
    require(m.elements >= 1 && m.elements <= 4 && (store ? m.data.size() : op.outputs.size()) == m.elements,
            "Invalid shared vector shape");
    require(!atomic || (m.atomic && m.element_bytes == 4 && m.elements == 1 && m.data.size() == 1 &&
                       m.access == Access::read_write), "Invalid shared atomic shape");
    require(atomic || !m.atomic, "Atomic metadata on ordinary shared access");
    require(!atomic || (op.numeric.scalar == ScalarType::u32 || op.numeric.scalar == ScalarType::s32),
            "Shared atomics require 32-bit integers", DiagnosticCode::unsupported_modifier);
    const auto extent = c.resources().compute->shared_memory_bytes;
    std::uint64_t maximum = 0;
    // armigo-shared-guard-v1: a nonliteral offset without a legalized bound
    // stays runtime-checked. The guard performs the whole access only
    // when it ends inside the launch allocation, an out-of-range load reads
    // zero and an out-of-range store is dropped (the runtime_zero_words
    // membership rule). Atomics keep strict proofs: a dropped RMW is not a
    // defined guest outcome.
    bool guarded = false;
    if (m.dynamic_byte_offset) {
        if (const auto* literal = std::get_if<LiteralBits>(&op.inputs[m.dynamic_byte_offset->value])) maximum = literal->bits;
        else if (m.dynamic_byte_offset_upper_bound) maximum = *m.dynamic_byte_offset_upper_bound;
        else guarded = true;
    }
    require(!guarded || !atomic, "Guarded shared atomics require a proven offset bound",
        DiagnosticCode::unsupported_capability);
    if (!guarded)
        require(std::uint64_t(m.byte_offset) + maximum + std::uint64_t(m.element_bytes) * m.elements <= extent,
                "Shared access exceeds launch allocation");
    auto base = number(m.byte_offset) + "ul";
    if (m.dynamic_byte_offset) base = "(" + base + " + ulong(" + input(c, op, *m.dynamic_byte_offset, ScalarType::u32) + "))";
    const auto address = "shared_address" + number(id.value);
    take(c.statement("const ulong " + address + " = " + base + ";"));
    std::string inside;
    if (guarded) {
        inside = "shared_inside" + number(id.value);
        take(c.statement("const bool " + inside + " = " + address + " + " +
            number(std::uint64_t(m.element_bytes) * m.elements) + "ul <= " + number(extent) + "ul;"));
    }
    // One word-slot CAS preserves adjacent lanes when different invocations
    // update one backing word; guarded byte stores repeat it per byte.
    const auto cas_byte_store = [&](const std::string& byte_address, const std::string& data) {
        const auto slot = "&armigo_shared_words[" + byte_address + " / 4ul]";
        const auto shift = "uint((" + byte_address + " % 4ul) * 8ul)";
        const auto old = "shared_old" + number(id.value), next = "shared_next" + number(id.value);
        // Each byte runs in its own scope: one guarded store repeats these names.
        take(c.statement("{"));
        take(c.statement("uint " + old + " = atomic_load_explicit(" + slot + ", memory_order_relaxed);"));
        take(c.statement("do {"));
        take(c.statement("const uint " + next + " = (" + old + " & ~(255u << " + shift + ")) | (((" + data +
            " & 255u) << " + shift + "));"));
        take(c.statement("if (atomic_compare_exchange_weak_explicit(" + slot + ", &" + old + ", " + next +
            ", memory_order_relaxed, memory_order_relaxed)) break;"));
        take(c.statement("} while (true);"));
        take(c.statement("}"));
    };
    for (unsigned i = 0; i < m.elements; ++i) {
        const auto offset = "(" + address + " + " + number(i * m.element_bytes) + "ul)";
        const auto pointer = "&armigo_shared_words[" + offset + " / 4ul]";
        const auto byte_pointer = [&](const std::string& byte_address) {
            return "&armigo_shared_words[" + byte_address + " / 4ul]";
        };
        if (atomic) {
            std::string function;
            switch (*m.atomic) {
            case AtomicKind::add: function = "atomic_fetch_add_explicit"; break;
            case AtomicKind::minimum: function = "atomic_fetch_min_explicit"; break;
            case AtomicKind::maximum: function = "atomic_fetch_max_explicit"; break;
            case AtomicKind::bit_and: function = "atomic_fetch_and_explicit"; break;
            case AtomicKind::bit_or: function = "atomic_fetch_or_explicit"; break;
            case AtomicKind::bit_xor: function = "atomic_fetch_xor_explicit"; break;
            case AtomicKind::exchange: function = "atomic_exchange_explicit"; break;
            default: fail("Shared atomic operation is unsupported", DiagnosticCode::unsupported_capability);
            }
            const bool signed_compare = op.numeric.scalar == ScalarType::s32 &&
                (*m.atomic == AtomicKind::minimum || *m.atomic == AtomicKind::maximum);
            const auto type = signed_compare ? ScalarType::s32 : ScalarType::u32;
            auto ptr = signed_compare ? "reinterpret_cast<threadgroup atomic_int*>(" + pointer + ")" : pointer;
            take(c.assign(output(op, 0), function + "(" + ptr + ", " + input(c, op, m.data[0], type) + ", memory_order_relaxed)", type));
        } else if (store) {
            auto data = input(c, op, m.data[i], ScalarType::u32);
            if (guarded) {
                if (m.element_bytes == 4) {
                    take(c.statement("if (" + inside + ") {"));
                    for (unsigned byte = 0; byte < 4; ++byte)
                        cas_byte_store("(" + offset + " + " + number(byte) + "ul)",
                            "(" + data + " >> " + number(byte * 8) + "u)");
                    take(c.statement("}"));
                } else {
                    take(c.statement("if (" + inside + ") {"));
                    for (unsigned byte = 0; byte < m.element_bytes; ++byte)
                        cas_byte_store("(" + offset + " + " + number(byte) + "ul)",
                            "(" + data + " >> " + number(byte * 8) + "u)");
                    take(c.statement("}"));
                }
            } else if (m.element_bytes == 4) take(c.statement("atomic_store_explicit(" + pointer + ", " + data + ", memory_order_relaxed);"));
            else {
                // CAS preserves adjacent lanes when different invocations update one backing word.
                const auto suffix = number(id.value) + "_" + number(i);
                const auto shift = "shared_shift" + suffix, mask = "shared_mask" + suffix;
                const auto old = "shared_old" + suffix, next = "shared_next" + suffix;
                take(c.statement("const uint " + shift + " = uint((" + offset + " & " + (m.element_bytes == 1 ? "3ul" : "2ul") + ") * 8ul);"));
                take(c.statement("const uint " + mask + " = " + (m.element_bytes == 1 ? "255u" : "65535u") + " << " + shift + ";"));
                take(c.statement("uint " + old + " = atomic_load_explicit(" + pointer + ", memory_order_relaxed);"));
                take(c.statement("do {"));
                take(c.statement("const uint " + next + " = (" + old + " & ~" + mask + ") | ((" + data + " << " + shift + ") & " + mask + ");"));
                take(c.statement("if (atomic_compare_exchange_weak_explicit(" + pointer + ", &" + old + ", " + next + ", memory_order_relaxed, memory_order_relaxed)) break;"));
                take(c.statement("} while (true);"));
            }
        } else {
            auto expression = "atomic_load_explicit(" + pointer + ", memory_order_relaxed)";
            if (guarded) {
                std::string assembled;
                for (unsigned byte = 0; byte < m.element_bytes; ++byte) {
                    const auto byte_address = "(" + offset + " + " + number(byte) + "ul)";
                    auto part = "((atomic_load_explicit(" + byte_pointer(byte_address) +
                        ", memory_order_relaxed) >> uint((" + byte_address + " % 4ul) * 8ul)) & 255u)";
                    if (byte) part = "(" + part + " << " + number(byte * 8) + "u)";
                    if (!assembled.empty()) assembled += " | ";
                    assembled += part;
                }
                expression = "(" + assembled + ")";
                if (m.element_bytes < 4 && op.numeric.domain == Domain::signed_integer)
                    expression = "as_type<uint>(int(as_type<" + std::string(m.element_bytes == 1 ? "char>(uchar(" : "short>(ushort(") + expression + "))))";
            } else if (m.element_bytes < 4) {
                expression = "((" + expression + " >> uint((" + offset + " & " + (m.element_bytes == 1 ? "3ul" : "2ul") + ") * 8ul)) & " +
                    (m.element_bytes == 1 ? "255u)" : "65535u)");
                if (op.numeric.domain == Domain::signed_integer)
                    expression = "as_type<uint>(int(as_type<" + std::string(m.element_bytes == 1 ? "char>(uchar(" : "short>(ushort(") + expression + "))))";
            }
            if (guarded) expression = "(" + inside + " ? " + expression + " : 0u)";
            take(c.assign(output(op, i), expression, ScalarType::u32));
        }
    }
}
}
namespace {
// The recorded allocation fact sizes both the prologue declaration and this
// static recheck; emission has no FactReader, so the dependency is the source.
std::uint64_t local_allocation(MslContext& c) {
    std::uint64_t allocation = 0;
    bool found = false;
    for (const auto& dependency : c.resources().dependencies)
        if (dependency.key.kind == FactKind::local_memory_bytes) {
            require(!found && std::holds_alternative<std::uint32_t>(dependency.value),
                    "Local memory allocation dependency is duplicated or invalid", DiagnosticCode::invalid_fact);
            found = true;
            allocation = std::get<std::uint32_t>(dependency.value);
        }
    require(found && allocation > 0, "Local memory access has no recorded allocation", DiagnosticCode::missing_fact);
    return allocation;
}

// Per-thread LDL/STL lowering onto the prologue's word array: byte elements
// always extract from or
// insert into their containing word, while wider elements require a static
// word-aligned address; everything else lowers as byte-wise little-endian
// accesses, because the pipeline proves no alignment for guest register values.
void generate_local_memory(MslContext& c, OperationId id, const Operation& op, const MemoryInfo& m) {
    const bool store = op.opcode == Opcode::store;
    require(!m.resource && !m.constant_buffer_slot && !m.array_index && !m.is_volatile && !m.atomic &&
            m.order == MemoryOrder::relaxed && m.scope == MemoryScope::invocation,
            "Local memory modifiers are unsupported", DiagnosticCode::unsupported_modifier);
    require(op.numeric.domain != Domain::floating && scalar_bits(op.numeric.scalar) <= 32,
            "Local memory requires an integer/raw-word interpretation", DiagnosticCode::unsupported_modifier);
    require(m.element_bytes == 1 || m.element_bytes == 2 || m.element_bytes == 4,
            "Unsupported local element size", DiagnosticCode::unsupported_capability);
    require(m.elements >= 1 && m.elements <= 4 && (store ? m.data.size() : op.outputs.size()) == m.elements,
            "Invalid local vector shape");
    require(!m.dynamic_byte_offset || m.dynamic_byte_offset->value < op.inputs.size(), "Dynamic offset role is out of range");
    require(m.dynamic_byte_offset || !m.dynamic_byte_offset_upper_bound, "Offset bound without a dynamic offset");
    const auto allocation = local_allocation(c);
    std::uint64_t maximum = 0;
    // armigo-local-guard-v1: an unproven register offset keeps the access
    // runtime-checked; a literal or legalized bound keeps the static check.
    bool guarded = false;
    if (m.dynamic_byte_offset) {
        if (const auto* literal = std::get_if<LiteralBits>(&op.inputs[m.dynamic_byte_offset->value])) maximum = literal->bits;
        else if (m.dynamic_byte_offset_upper_bound) maximum = *m.dynamic_byte_offset_upper_bound;
        else guarded = true;
    }
    if (!guarded)
        require(std::uint64_t(m.byte_offset) + maximum + std::uint64_t(m.element_bytes) * m.elements <= allocation,
                "Local memory access exceeds the declared allocation");
    auto base = number(m.byte_offset) + "ul";
    if (m.dynamic_byte_offset) base = "(" + base + " + ulong(" + input(c, op, *m.dynamic_byte_offset, ScalarType::u32) + "))";
    const auto suffix = number(id.value);
    const auto address = "local_address" + suffix;
    take(c.statement("const ulong " + address + " = " + base + ";"));
    std::string inside;
    if (guarded) {
        inside = "local_inside" + suffix;
        take(c.statement("const bool " + inside + " = " + address + " + " +
            number(std::uint64_t(m.element_bytes) * m.elements) + "ul <= sizeof(armigo_local_words);"));
    }
    const auto load_word = [&](const std::string& offset) {
        return "armigo_local_words[" + offset + " / 4ul]";
    };
    for (unsigned i = 0; i < m.elements; ++i) {
        const auto element_offset = std::uint64_t(m.byte_offset) + std::uint64_t(i) * m.element_bytes;
        const auto offset = "(" + address + " + " + number(std::uint64_t(i) * m.element_bytes) + "ul)";
        const bool static_aligned = !m.dynamic_byte_offset && (element_offset & 3u) + m.element_bytes <= 4;
        if (store) {
            if (guarded) take(c.statement("if (" + inside + ") {"));
            const auto data = input(c, op, m.data[i], ScalarType::u32);
            if (static_aligned) {
                if (m.element_bytes == 4) {
                    take(c.statement(load_word(offset) + " = " + data + ";"));
                } else {
                    const auto shift = number((element_offset & 3u) * 8u) + "u";
                    const auto mask = m.element_bytes == 1 ? "255u" : "65535u";
                    const auto slot = load_word(offset);
                    take(c.statement(slot + " = (" + slot + " & ~(" + mask + " << " + shift + ")) | ((" +
                        data + " << " + shift + ") & (" + mask + " << " + shift + "));"));
                }
            } else {
                for (unsigned byte = 0; byte < m.element_bytes; ++byte) {
                    const auto byte_address = "(" + offset + " + " + number(byte) + "ul)";
                    const auto slot = load_word(byte_address);
                    const auto shift = "uint((" + byte_address + " % 4ul) * 8ul)";
                    take(c.statement(slot + " = (" + slot + " & ~(255u << " + shift + ")) | (((" + data +
                        " >> " + number(byte * 8) + "u) & 255u) << " + shift + ");"));
                }
            }
            if (guarded) take(c.statement("}"));
        } else {
            std::string expression;
            if (static_aligned) {
                expression = load_word(offset);
                if (m.element_bytes < 4)
                    expression = "((" + expression + " >> uint((" + offset + " & " + (m.element_bytes == 1 ? "3ul" : "2ul") +
                        ") * 8ul)) & " + (m.element_bytes == 1 ? "255u)" : "65535u)");
            } else {
                for (unsigned byte = 0; byte < m.element_bytes; ++byte) {
                    const auto byte_address = "(" + offset + " + " + number(byte) + "ul)";
                    auto part = "((" + load_word(byte_address) + " >> uint((" + byte_address + " % 4ul) * 8ul)) & 255u)";
                    if (byte) part = "(" + part + " << " + number(byte * 8) + "u)";
                    if (!expression.empty()) expression += " | ";
                    expression += part;
                }
                expression = "(" + expression + ")";
            }
            if (m.element_bytes < 4 && op.numeric.domain == Domain::signed_integer)
                expression = "as_type<uint>(int(as_type<" + std::string(m.element_bytes == 1 ? "char>(uchar(" : "short>(ushort(") + expression + "))))";
            if (guarded) expression = "(" + inside + " ? " + expression + " : 0u)";
            if (guarded) expression = "(" + inside + " ? " + expression + " : 0u)";
            take(c.assign(output(op, i), expression, ScalarType::u32));
        }
    }
}
}
Status generate_compute_barrier(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        const auto* m = std::get_if<MemoryInfo>(&op.info);
        require(c.program().ir.stage == Stage::compute && m && op.inputs.empty() && op.outputs.empty() &&
                op.numeric == NumericMode{} && m->space == AddressSpace::shared && m->access == Access::read_write &&
                !m->resource && !m->constant_buffer_slot && !m->dynamic_byte_offset && !m->array_index &&
                !m->dynamic_byte_offset_upper_bound && m->data.empty() && !m->atomic && !m->is_volatile &&
                m->byte_offset == 0 && m->element_bytes == 4 && m->elements == 1 && m->alignment == 1 &&
                m->order == MemoryOrder::acquire_release &&
                (m->scope == MemoryScope::workgroup || m->scope == MemoryScope::device),
                "Unsupported compute barrier metadata", DiagnosticCode::unsupported_modifier);
        if (op.opcode == Opcode::barrier) {
            require(m->scope == MemoryScope::workgroup, "Execution barrier must have workgroup scope");
            take(c.statement("threadgroup_barrier(mem_flags::mem_threadgroup);"));
        } else {
            const auto scope = m->scope == MemoryScope::device ? "thread_scope_device" : "thread_scope_threadgroup";
            take(c.statement("atomic_thread_fence(mem_flags::mem_threadgroup | mem_flags::mem_device | mem_flags::mem_texture, memory_order_seq_cst, " + std::string(scope) + ");"));
        }
    });
}
// One storage element at a descriptor-relative byte address. A 4-byte
// element whose address legalization proved word aligned is one host word;
// everything else extracts its bytes from the containing words.
static std::string storage_load(const std::string& name, const std::string& address, const MemoryInfo& m,
                                const Operation& op, const std::string& unit = "ul") {
    if (m.element_bytes == 4 && m.storage_word_aligned) return name + "[" + address + " / 4" + unit + "]";
    std::string expression;
    for (unsigned byte = 0; byte < m.element_bytes; ++byte) {
        const auto position = "(" + address + " + " + number(byte) + unit + ")";
        auto part = "((" + name + "[" + position + " / 4" + unit + "] >> uint((" + position +
            " % 4" + unit + ") * 8" + unit + ")) & 255u)";
        if (byte) part = "(" + part + " << " + number(byte * 8) + "u)";
        if (!expression.empty()) expression += " | ";
        expression += part;
    }
    expression = "(" + expression + ")";
    if (m.element_bytes < 4 && op.numeric.domain == Domain::signed_integer)
        expression = "as_type<uint>(int(as_type<" +
            std::string(m.element_bytes == 1 ? "char>(uchar(" : "short>(ushort(") + expression + "))))";
    return expression;
}

// The descriptor base and size accompany the storage pointer as one binding
// contract. Full-width subtraction checks the actual high word, including any
// carry supplied by the guest, before a host pointer can be dereferenced.
void generate_multi_target_storage(MslContext& c, OperationId id, const Operation& op,
                                   const MemoryInfo& m);

void generate_global_storage(MslContext& c, OperationId id, const Operation& op,
                             const MemoryInfo& m, const BufferUse& buffer) {
    require(m.storage_address && m.dynamic_byte_offset &&
            buffer.size.kind == SizeKind::storage_descriptor && buffer.size.bytes == 4,
            "Global memory has no runtime storage descriptor");
    if (!m.storage_alternatives.empty()) { generate_multi_target_storage(c, id, op, m); return; }
    const auto& a = *m.storage_address;
    const auto word = [&](InputIndex index) { return input(c, op, index, ScalarType::u32); };
    const auto suffix = number(id.value);
    const auto delta = "storage_offset" + suffix;
    const auto size = "storage_size" + suffix;
    take(c.statement("const uint " + size + " = " + word(a.byte_size) + ";"));
    // The GPU's integer units are 32 bits wide, so the descriptor-relative
    // offset and its checks use 32-bit halves with explicit carries.
    std::string membership;
    if (m.global_address_high) {
        // The 64-bit guest address plus its signed displacement, minus the
        // 64-bit descriptor base: inside the descriptor only when the
        // difference's high half is zero and the address is not below the
        // base (the high half alone wraps for a base whose high word is
        // all ones).
        const auto displacement = static_cast<std::int32_t>(m.byte_offset);
        const auto low = "global_low" + suffix, high = "global_high" + suffix, inside = "storage_inside" + suffix;
        const auto guest_low = word(*m.dynamic_byte_offset);
        const auto base_low = "storage_base_low" + suffix, base_high = "storage_base_high" + suffix;
        take(c.statement("const uint " + base_low + " = " + word(a.base_low) + ";"));
        take(c.statement("const uint " + base_high + " = " + word(a.base_high) + ";"));
        take(c.statement("const uint " + low + " = " + guest_low + " + " +
            number(static_cast<std::uint32_t>(displacement)) + "u;"));
        take(c.statement("const uint " + high + " = " + word(*m.global_address_high) +
            (displacement < 0 ? " + 4294967295u" : "") + " + uint(" + low + " < " + guest_low + ");"));
        take(c.statement("const uint " + delta + " = " + low + " - " + base_low + ";"));
        take(c.statement("const bool " + inside + " = " + high + " - " + base_high + " - uint(" + low +
            " < " + base_low + ") == 0u && " + high + " >= " + base_high + ";"));
        membership = inside + " && ";
    } else {
        // The nonextended form addresses the descriptor-relative low-word
        // window. Its arithmetic deliberately remains modulo 2^32.
        const auto base_low = "storage_base_low" + suffix;
        take(c.statement("const uint " + base_low + " = " + word(a.base_low) + ";"));
        take(c.statement("const uint " + delta + " = " + word(*m.dynamic_byte_offset) +
            " + " + number(m.byte_offset) + "u - " + base_low + ";"));
    }
    const auto name = take(c.resource(*m.resource));
    const bool store = op.opcode == Opcode::store;
    if (store) require(m.element_bytes == 4 && m.alignment >= 4,
        "Subword global storage writes need explicit lowering", DiagnosticCode::unsupported_capability);
    for (unsigned i = 0; i < m.elements; ++i) {
        const auto component = number(i * m.element_bytes) + "u";
        const auto address = "(" + delta + " + " + component + ")";
        // Ordered subtraction avoids overflow.
        const auto valid = "(" + membership + delta + " <= " + size + " && " + component +
            " <= " + size + " - " + delta + " && " + number(m.element_bytes) + "u <= " +
            size + " - " + delta + " - " + component + ")";
        if (store) {
            take(c.statement("if " + valid + " { " + name + "[" + address + " / 4u] = " +
                word(m.data[i]) + "; }"));
        } else {
            take(c.assign(output(op, i), valid + " ? " + storage_load(name, address, m, op, "u") + " : 0u",
                          ScalarType::u32));
        }
    }
}

// An address with several descriptor origins
// is served by the first candidate whose [base, base + size) range contains
// it; with no match a load reads zero and a store is dropped. Each component
// keeps the single-target descriptor-size guard.
void generate_multi_target_storage(MslContext& c, OperationId id, const Operation& op,
                                   const MemoryInfo& m) {
    struct Target { std::string name; const StorageAddress* address; };
    std::vector<Target> targets{{take(c.resource(*m.resource)), &*m.storage_address}};
    for (const auto& alternative : m.storage_alternatives) {
        const auto& planned_use = planned(c.resources(), alternative.resource).use;
        const auto* buffer = std::get_if<BufferUse>(&planned_use);
        require(buffer && std::holds_alternative<StorageBufferSource>(buffer->source) &&
                buffer->size.kind == SizeKind::storage_descriptor && buffer->size.bytes == 4,
                "Storage alternative has no runtime storage descriptor");
        targets.push_back({take(c.resource(alternative.resource)), &alternative.address});
    }
    const auto word = [&](InputIndex index) { return input(c, op, index, ScalarType::u32); };
    const auto suffix = number(id.value);
    const auto guest = "global_address" + suffix;
    const auto chosen = "storage_target" + suffix;
    if (m.global_address_high) {
        const auto displacement = static_cast<std::int32_t>(m.byte_offset);
        take(c.statement("const ulong " + guest + " = (ulong(" + word(*m.dynamic_byte_offset) +
            ") | (ulong(" + word(*m.global_address_high) + ") << 32u)) + ulong(" +
            std::to_string(displacement) + "l);"));
    }
    std::string selection = number(targets.size()) + "u";
    for (std::size_t t = targets.size(); t-- > 0;) {
        const auto& a = *targets[t].address;
        const auto tag = suffix + "_" + number(t);
        const auto base = "storage_base" + tag, size = "storage_size" + tag, delta = "storage_offset" + tag;
        take(c.statement("const ulong " + base + " = ulong(" + word(a.base_low) +
            ") | (ulong(" + word(a.base_high) + ") << 32u);"));
        take(c.statement("const ulong " + size + " = ulong(" + word(a.byte_size) + ");"));
        std::string membership;
        if (m.global_address_high) {
            take(c.statement("const ulong " + delta + " = " + guest + " - " + base + ";"));
            membership = guest + " >= " + base + " && ";
        } else {
            take(c.statement("const ulong " + delta + " = ulong(uint(" + word(*m.dynamic_byte_offset) +
                " + " + number(m.byte_offset) + "u - " + word(a.base_low) + "));"));
        }
        selection = "(" + membership + delta + " < " + size + ") ? " + number(t) + "u : " + selection;
    }
    take(c.statement("const uint " + chosen + " = " + selection + ";"));
    const bool store = op.opcode == Opcode::store;
    if (store) require(m.element_bytes == 4 && m.alignment >= 4,
        "Subword global storage writes need explicit lowering", DiagnosticCode::unsupported_capability);
    for (unsigned i = 0; i < m.elements; ++i) {
        const auto component = number(i * m.element_bytes) + "ul";
        std::string loaded = "0u";
        for (std::size_t t = targets.size(); t-- > 0;) {
            const auto tag = suffix + "_" + number(t);
            const auto size = "storage_size" + tag, delta = "storage_offset" + tag;
            const auto& name = targets[t].name;
            const auto address = "(" + delta + " + " + component + ")";
            const auto valid = "(" + chosen + " == " + number(t) + "u && " + component + " <= " + size +
                " - " + delta + " && " + number(m.element_bytes) + "ul <= " + size + " - " + delta +
                " - " + component + ")";
            if (store) {
                take(c.statement("if " + valid + " { " + name + "[" + address + " / 4ul] = " +
                    word(m.data[i]) + "; }"));
                continue;
            }
            loaded = valid + " ? " + storage_load(name, address, m, op) + " : " + loaded;
        }
        if (!store) take(c.assign(output(op, i), loaded, ScalarType::u32));
    }
}

Status generate_stage_memory(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id); const auto* info = std::get_if<MemoryInfo>(&op.info);
        require(info, "Memory operation has no memory metadata"); const auto& m = *info;
        plain_numeric(op);
        if (m.space == AddressSpace::shared) { generate_shared_memory(c, id, op, m); return; }
        if (m.space == AddressSpace::local) { generate_local_memory(c, id, op, m); return; }
        require(op.opcode == Opcode::load || op.opcode == Opcode::store,
                "Buffer atomics require a proven Metal address adapter", DiagnosticCode::unsupported_capability);
        require(m.space == AddressSpace::constant_buffer || m.space == AddressSpace::storage,
                "Memory address space has no implemented allocation", DiagnosticCode::unsupported_capability);
        require(m.resource.has_value(), "Memory resource is unresolved", DiagnosticCode::unresolved_resource);
        require(!m.array_index && !m.atomic && !m.is_volatile && m.order == MemoryOrder::relaxed && m.scope == MemoryScope::invocation,
                "Memory modifiers require explicit lowering", DiagnosticCode::unsupported_modifier);
        require(m.elements >= 1 && m.elements <= 4 && std::has_single_bit(m.element_bytes) && m.element_bytes <= 8, "Invalid memory shape");
        const auto& resource = planned(c.resources(), *m.resource);
        const auto* buffer = std::get_if<BufferUse>(&resource.use);
        require(buffer, "Memory resource is not a buffer");
        if (m.space == AddressSpace::constant_buffer) {
            const auto* source = std::get_if<ConstantBufferSource>(&buffer->source);
            require(source && m.constant_buffer_slot == source->slot && op.opcode == Opcode::load && m.access == Access::read,
                    "Constant-buffer binding/source mismatch");
        } else require(std::holds_alternative<StorageBufferSource>(buffer->source), "Storage buffer source mismatch");
        const bool store = op.opcode == Opcode::store;
        require(store ? buffer->access != Access::read && m.access != Access::read : buffer->access != Access::write && m.access == Access::read,
                "Buffer access direction mismatch");
        require((store ? m.data.size() : op.outputs.size()) == m.elements, "Memory vector count mismatch");
        if (m.storage_address) {
            require(m.space == AddressSpace::storage && m.element_bytes <= 4,
                "Runtime storage descriptor has an invalid memory shape");
            generate_global_storage(c, id, op, m, *buffer);
            return;
        }
        require(!m.global_address_high && buffer->size.kind != SizeKind::storage_descriptor,
            "Unresolved extended global address");
        const bool runtime_window = m.space == AddressSpace::constant_buffer;
        const auto bytes = std::uint64_t(m.element_bytes) * m.elements;
        if (!runtime_window)
            require(std::uint64_t(m.byte_offset) + bytes <= buffer->size.bytes, "Memory access exceeds proven extent");
        require(m.dynamic_byte_offset || !m.dynamic_byte_offset_upper_bound, "Offset bound without a dynamic offset");
        std::optional<std::uint64_t> maximum_offset;
        if (m.dynamic_byte_offset) {
            require(m.dynamic_byte_offset->value < op.inputs.size(), "Dynamic offset role is out of range");
            const auto* literal = std::get_if<LiteralBits>(&op.inputs[m.dynamic_byte_offset->value]);
            const auto bound = literal ? std::optional<std::uint64_t>{literal->bits} :
                m.dynamic_byte_offset_upper_bound ? std::optional<std::uint64_t>{*m.dynamic_byte_offset_upper_bound} : std::nullopt;
            maximum_offset = bound;
            if (!runtime_window) {
                require(bound.has_value(), "Dynamic memory offset has no proven upper bound", DiagnosticCode::missing_fact);
                require(std::uint64_t(m.byte_offset) + *bound + bytes <= buffer->size.bytes, "Dynamic memory access exceeds its binding extent");
            }
            if (literal && m.dynamic_byte_offset_upper_bound) require(literal->bits <= *m.dynamic_byte_offset_upper_bound, "Literal offset contradicts its bound");
            if (store && literal) require(literal->bits % 4 == 0, "Storage write offset is not word aligned");
        }
        if (runtime_window) {
            std::uint64_t extent = constant_buffer_window_bytes;
            if (!m.dynamic_byte_offset || maximum_offset) {
                const auto last_component = std::uint64_t(m.byte_offset) + maximum_offset.value_or(0) +
                    std::uint64_t(m.element_bytes) * (m.elements - 1);
                const auto last_word = (last_component + (m.element_bytes == 8 ? 4u : 0u)) & ~std::uint64_t{3};
                extent = std::min<std::uint64_t>(last_word + 4, constant_buffer_window_bytes);
            }
            require(extent <= buffer->size.bytes,
                    "Constant-buffer read window does not cover potentially valid accesses");
        }
        // Guest-width arithmetic has already been emitted into the offset IR.
        // Widen before adding host component offsets so a large address cannot
        // wrap around into valid backing. Runtime visibility is separate from
        // the reflected read window and from any optional analysis bound.
        const auto name = take(c.resource(*m.resource));
        std::string base = number(m.byte_offset) + "ul";
        if (m.dynamic_byte_offset) base = "(" + base + " + ulong(" + input(c, op, *m.dynamic_byte_offset, ScalarType::u32) + "))";
        const auto address_name = "memory_address" + number(id.value);
        take(c.statement("const ulong " + address_name + " = " + base + ";"));
        base = address_name;
        require(buffer->size.bytes % 4 == 0, "Raw-word buffer binding requires a padded word extent", DiagnosticCode::missing_fact);
        if (store) require(m.element_bytes == 4 && m.byte_offset % 4 == 0 && m.alignment >= 4,
                "Unaligned/subword storage writes need explicit lowering", DiagnosticCode::unsupported_capability);
        std::string runtime_size;
        if (runtime_window) {
            require(buffer->size.kind == SizeKind::constant_buffer_window &&
                    buffer->size.constant_buffer_policy == ConstantBufferPolicy::runtime_zero_words_v1,
                    "Constant-buffer load requires the runtime word-zero policy");
            const PlannedResource* support = nullptr;
            for (const auto& candidate : c.resources().resources) {
                const auto* use = std::get_if<BufferUse>(&candidate.use);
                const auto* source = use ? std::get_if<SupportBufferSource>(&use->source) : nullptr;
                if (source && source->schema == constant_buffer_support_schema &&
                    source->version == constant_buffer_support_version) {
                    require(!support, "Duplicate constant-buffer support table");
                    support = &candidate;
                }
            }
            require(support, "Constant-buffer support table is absent", DiagnosticCode::unresolved_resource);
            runtime_size = take(c.resource(support->resource)) + "[" + number(*m.constant_buffer_slot) + "u]";
            take(c.require_helper("armigo_cbuf_word"));
        }
        for (unsigned i = 0; i < m.elements; ++i) {
            auto offset = "(" + base + " + " + number(i * m.element_bytes) + "ul)";
            if (store) {
                take(c.statement(name + "[" + offset + " / 4ul] = " + input(c, op, m.data[i], ScalarType::u32) + ";"));
            } else {
                std::string expression;
                const auto type = m.element_bytes == 8 ? ScalarType::u64 : ScalarType::u32;
                if (runtime_window) {
                    auto word = [&](const std::string& address) {
                        return "armigo_cbuf_word(" + name + ", " + address + ", " + runtime_size +
                            ", " + number(buffer->size.bytes) + "u)";
                    };
                    expression = word(offset);
                    if (m.element_bytes == 8) {
                        expression = "(ulong(" + expression + ") | (ulong(" + word("(" + offset + " + 4ul)") + ") << 32u))";
                    } else if (m.element_bytes < 4) {
                        // Halfword addresses align down to two bytes; all
                        // narrow reads require a complete visible backing word.
                        const auto mask = m.element_bytes == 1 ? "3ul" : "2ul";
                        expression = "((" + expression + " >> uint((" + offset + " & " + mask + ") * 8ul)) & " +
                            (m.element_bytes == 1 ? "255u)" : "65535u)");
                    }
                } else {
                    for (unsigned byte = 0; byte < m.element_bytes; ++byte) {
                        auto address = "(" + offset + " + " + number(byte) + "ul)";
                        auto part = "((" + name + "[" + address + " / 4ul] >> uint((" + address + " % 4ul) * 8ul)) & 255u)";
                        if (type == ScalarType::u64) part = "ulong(" + part + ")";
                        if (byte) part = "(" + part + " << " + number(byte * 8) + "u)";
                        if (!expression.empty()) expression += " | ";
                        expression += part;
                    }
                    expression = "(" + expression + ")";
                }
                if (m.element_bytes < 4 && op.numeric.domain == Domain::signed_integer)
                    expression = "as_type<uint>(int(as_type<" + std::string(m.element_bytes == 1 ? "char>(uchar(" : "short>(ushort(") + expression + "))))";
                take(c.assign(output(op, i), expression, type));
            }
        }
    });
}
Status generate_stage_discard(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(c.program().ir.stage == Stage::fragment && op.inputs.empty() && op.outputs.empty(), "Discard is only valid in a fragment stage");
        require(op.numeric == NumericMode{}, "Discard has unsupported modifiers", DiagnosticCode::unsupported_modifier);
        // Discard is a distinct effect. It does not become a return or an assumed
        // helper demotion, and following operations remain in their original order.
        take(c.statement("discard_fragment();"));
    });
}
} // namespace armigo::shader::pipeline
