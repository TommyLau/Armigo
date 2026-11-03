// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Barrier emitters: BAR, MEMBAR and DEPBAR. Only BAR sync is defined guest
// behavior; MEMBAR keeps its memory-fence meaning (no execution barrier), and
// DEPBAR coordinates instruction scoreboards only, so it has no semantic
// effect on results.

#include "io_memory_shared.h"
#include "pipeline/decode/resource_fields.h"

namespace armigo::shader::pipeline {
namespace {
using namespace io_memory;

[[nodiscard]] Status emit_zero_arg(Opcode opcode, EmitterContext& context, MemoryScope scope) {
    // Barrier opcodes carry memory info; the scope is the only guest-visible
    // state and space stays the workgroup association.
    MemoryInfo info;
    info.space = AddressSpace::shared;
    info.access = Access::read_write;
    info.order = MemoryOrder::acquire_release;
    info.scope = scope;
    auto emitted = context.emit(opcode, NumericMode{}, {}, {}, info, Effects::none);
    if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
    return std::monostate{};
}

} // namespace

// BAR: execution barrier. Sync is the only supported mode.
Status emit_execution_barrier(EmitterContext& context) {
    if (barrier_operation(context.instruction().raw) != 0) {
        return fail(context, DiagnosticCode::unsupported_modifier,
            "BAR supports the sync operation only");
    }
    if (auto id = optional_field(context.instruction(), DecodedField::barrier_id);
        id && *id != 0) {
        return fail(context, DiagnosticCode::unsupported_modifier,
            "BAR with a nonzero barrier id is not supported");
    }
    if (context.decoded_program().stage == Stage::compute) {
        const auto& inst = context.instruction();
        if (!resource_decode::bar_a_fix(inst.raw) || !resource_decode::bar_b_fix(inst.raw) ||
            inst.raw.bits<8, 8>() != 0 || inst.raw.bits<35, 2>() != 0 ||
            inst.raw.bits<39, 3>() != 7 || inst.raw.bits<42, 1>() != 0 || (inst.predicate.enabled && (inst.predicate.index != 7 || inst.predicate.inverted)))
            return fail(context, DiagnosticCode::unsupported_modifier,
                "BAR requires fixed barrier zero, the full workgroup, and an unconditional instruction predicate");
    }
    return emit_zero_arg(Opcode::barrier, context, MemoryScope::workgroup);
}

// MEMBAR keeps memory-fence semantics without an execution rendezvous.
// Compute admits CTA/device scopes; system/viewport and cache operations fail.
Status emit_memory_fence(EmitterContext& context) {
    const auto kind = memory_barrier_kind(context.instruction().raw);
    if (context.decoded_program().stage == Stage::compute &&
        (kind > 1 || resource_decode::membar_invalidate(context.instruction().raw) != 0))
        return fail(context, DiagnosticCode::unsupported_modifier,
            "MEMBAR system/viewport scopes and cache invalidation require a separate host adapter");
    const auto scope = kind == 0 ? MemoryScope::workgroup : MemoryScope::device;
    return emit_zero_arg(Opcode::memory_barrier, context, scope);
}

// DEPBAR is a scheduling-only dependency-barrier wait with no
// result-producing or memory-visible effect, so no operation is emitted.
Status emit_dependency_barrier(EmitterContext& context) {
    (void)context;
    return std::monostate{};
}

// NOP does nothing.
Status emit_no_operation(EmitterContext& context) {
    (void)context;
    return std::monostate{};
}

// VOTE.VTG votes across the warp's vertex, tessellation or geometry threads
// and writes no register, predicate or memory, so no operation is emitted.
Status emit_vertex_vote(EmitterContext& context) {
    (void)context;
    return std::monostate{};
}

} // namespace armigo::shader::pipeline
