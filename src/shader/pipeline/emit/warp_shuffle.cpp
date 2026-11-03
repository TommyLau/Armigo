// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo emits the frozen IR shuffle operation with its SubgroupInfo selector;
// the Maxwell clamp/segment arithmetic itself belongs to the MSL generator,
// which the MSL generator lowers to inline code.

#include "pipeline/emit/integer_emit.h"

namespace armigo::shader::pipeline {
namespace {
using namespace integer_emit;

// The shuffle lowering needs the SIMD-group lane id, and the frozen shuffle
// schema has no lane input, so the MSL generator reads the declared builtin
// entry parameter. Emitting the matching load_io here records the read in the
// resource plan, which declares the parameter in the entry signature; the
// legalizer restricts this builtin to compute stages today.
Status emit_lane_id_read(EmitterContext& context) {
    IoInfo lane;
    lane.semantic = IoSemantic::subgroup_lane_id;
    lane.type = {ScalarType::u32, 1};
    lane.component_mask = 1;
    const ValueType word = ValueType::word32;
    auto read = context.load_io(lane, {}, NumericMode{Domain::unsigned_integer, ScalarType::u32}, std::span{&word, 1});
    if (const auto* error = std::get_if<Diagnostic>(&read)) return *error;
    return std::monostate{};
}
} // namespace

// SHFL: warp shuffle. Source A is the transferred word, source B is the lane
// index, backward/forward delta or xor mask, and source C carries the 5-bit
// clamp in its low byte with the 5-bit segment mask in its second byte. The
// results are the transferred word and the in-range predicate, written to the
// guest destination register and the LPd predicate.
Status emit_shfl(EmitterContext& context) {
    auto selector = field(context.instruction(), DecodedField::shuffle_mode);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    auto transferred = source_value(context, SourceRole::a);
    if (const auto* error = std::get_if<Diagnostic>(&transferred)) return *error;
    auto lane = source_value(context, SourceRole::b);
    if (const auto* error = std::get_if<Diagnostic>(&lane)) return *error;
    auto bounds = source_value(context, SourceRole::c);
    if (const auto* error = std::get_if<Diagnostic>(&bounds)) return *error;
    if (auto status = emit_lane_id_read(context); failed(status)) return status;
    SubgroupInfo subgroup;
    switch (value(selector) & 3) {
    case 0: subgroup.shuffle = ShuffleKind::index; break;
    case 1: subgroup.shuffle = ShuffleKind::up; break;
    case 2: subgroup.shuffle = ShuffleKind::down; break;
    case 3: subgroup.shuffle = ShuffleKind::butterfly; break;
    default: return context.error(DiagnosticCode::unsupported_modifier, "Unknown SHFL mode selector");
    }
    const std::array<Operand, 3> inputs{value(transferred), value(lane), value(bounds)};
    const std::array<ValueType, 2> outputs{ValueType::word32, ValueType::predicate};
    auto results = context.emit(Opcode::shuffle, bits32, inputs, outputs, subgroup, Effects::none);
    if (const auto* error = std::get_if<Diagnostic>(&results)) return *error;
    auto destination = destination_register(context);
    if (const auto* error = std::get_if<Diagnostic>(&destination)) return *error;
    if (auto status = context.write(value(destination), std::get<std::vector<ValueId>>(results).front());
        failed(status)) return status;
    auto predicate = role_register(context, RegisterRole::predicate_destination);
    if (const auto* error = std::get_if<Diagnostic>(&predicate)) return *error;
    return context.write(value(predicate), std::get<std::vector<ValueId>>(results).back());
}
} // namespace armigo::shader::pipeline
