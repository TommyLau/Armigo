// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include "../msl_context.h"
#include <algorithm>
#include <bit>
#include <set>
#include <stdexcept>

namespace armigo::shader::pipeline::msl {
struct Failure { Diagnostic diagnostic; };
[[noreturn]] inline void fail(std::string message, DiagnosticCode code = DiagnosticCode::invalid_ir) {
    throw Failure{diagnose(Phase::emit_msl, code, {}, std::move(message))};
}
inline void require(bool condition, std::string message, DiagnosticCode code = DiagnosticCode::invalid_ir) {
    if (!condition) fail(std::move(message), code);
}
template<class T> T take(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) throw Failure{std::move(*error)};
    return std::get<T>(std::move(result));
}
inline std::string number(std::uint64_t n) { return std::to_string(n); }
inline std::string scalar(ScalarType type) {
    switch (type) {
    case ScalarType::u8: return "uchar"; case ScalarType::s8: return "char";
    case ScalarType::u16: return "ushort"; case ScalarType::s16: return "short";
    case ScalarType::u32: return "uint"; case ScalarType::s32: return "int";
    case ScalarType::u64: return "ulong"; case ScalarType::s64: return "long";
    case ScalarType::f16: return "half"; case ScalarType::f32: return "float";
    case ScalarType::boolean: return "bool";
    case ScalarType::f64: fail("Metal has no native FP64 representation", DiagnosticCode::unsupported_capability);
    }
    fail("Invalid scalar type");
}
inline std::string raw_type(ValueType type) {
    switch(type) { case ValueType::word32: return "uint"; case ValueType::word64: return "ulong";
    case ValueType::predicate: return "bool"; } fail("Invalid storage type");
}
inline std::string interpret(std::string value, ScalarType type) {
    const auto name = scalar(type);
    if (type == ScalarType::u32 || type == ScalarType::u64 || type == ScalarType::boolean) return value;
    if (type == ScalarType::u8 || type == ScalarType::u16) return name + "(" + value + ")";
    if (scalar_bits(type) < 32) value = std::string(scalar_bits(type) == 8 ? "uchar(" : "ushort(") + value + ")";
    return "as_type<" + name + ">(" + value + ")";
}
inline std::string raw(std::string value, ScalarType type) {
    const auto name = scalar(type);
    if (type == ScalarType::u32 || type == ScalarType::u64 || type == ScalarType::boolean) return value;
    if (type == ScalarType::s8 || type == ScalarType::s16) return "as_type<uint>(int(" + value + "))";
    if (type == ScalarType::u8 || type == ScalarType::u16) return "uint(" + value + ")";
    if (type == ScalarType::f16) return "uint(as_type<ushort>(" + value + "))";
    return "as_type<" + raw_type(storage_type(type)) + ">(" + value + ")";
}
inline std::string io_name(IoSemantic semantic, std::uint16_t location) {
    switch (semantic) {
    case IoSemantic::generic: return "generic" + number(location);
    case IoSemantic::position: case IoSemantic::fragment_coord: return "position";
    case IoSemantic::color: return "color" + number(location);
    default: return "builtin" + number(static_cast<unsigned>(semantic));
    }
}
inline const PlannedResource& planned(const ResourcePlan& plan, ResourceId id) {
    auto it = std::find_if(plan.resources.begin(), plan.resources.end(), [&](const auto& r) { return r.resource == id; });
    require(it != plan.resources.end(), "Resource ID is absent from the binding plan", DiagnosticCode::unresolved_resource);
    return *it;
}
inline std::string input(MslContext& c, const Operation& op, InputIndex i, ScalarType type) {
    require(i.value < op.inputs.size(), "Input role is out of range");
    return take(c.operand(op.inputs[i.value], type));
}
inline ValueId output(const Operation& op, std::size_t i) {
    require(i < op.outputs.size() && std::holds_alternative<ValueId>(op.outputs[i]), "Output must be a value ID");
    return std::get<ValueId>(op.outputs[i]);
}
template<class F> Status callback(MslContext& c, OperationId id, F function) {
    try { function(); return std::monostate{}; }
    catch (Failure& f) {
        auto error = c.error(id, f.diagnostic.code, f.diagnostic.message, f.diagnostic.modifier);
        error.missing_fact = f.diagnostic.missing_fact;
        return error;
    }
}
inline void plain_numeric(const Operation& op) {
    require(!op.numeric.flush_inputs && !op.numeric.flush_output && !op.numeric.saturate &&
            (op.numeric.rounding == Rounding::not_applicable || op.numeric.rounding == Rounding::nearest_even),
            "Resource operation requires explicit lowering of numeric modifiers", DiagnosticCode::unsupported_modifier);
}
std::string resource_argument(const PlannedResource&);
std::string texture_type(const TextureUse&);
std::string pool_declarations(const ResourcePlan&);
std::string io_declarations(const ResourcePlan&, bool output,
                            bool fragment_output_types_from_render_targets,
                            bool emulate_depth_bias, bool fixed_point_size, bool invariant_position = false);
void validate_plan(const StructuredProgram&, const ResourcePlan&, const EmitOptions&);
ResourcePlan plan_custom_border(const StructuredProgram&, ResourcePlan, const EmitOptions&);
} // namespace armigo::shader::pipeline::msl
