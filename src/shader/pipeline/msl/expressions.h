// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo emits MSL directly from the owned IR: operands are bit-level words,
// and every floating modifier stays an explicit, checked transformation.

#pragma once
#include "stage_internal.h"

#include <string>
#include <vector>

namespace armigo::shader::pipeline::msl {
// Lane-internal machinery shared by expressions.cpp and numeric.cpp: checked
// operand/output access, storage-to-interpretation mapping, and named bit
// temporaries. Generated results are assigned as raw words so the tested
// integer-assisted helpers keep their exact contracts.

// Bounds-checked positional operand text. Numeric-schema inputs are
// positional; metadata-role inputs use input() from stage_internal.h instead.
inline std::string operand_at(MslContext& c, const Operation& op, std::size_t index, ScalarType type) {
    require(index < op.inputs.size(), "Operation input is missing");
    return take(c.operand(op.inputs[index], type));
}

inline ValueId output_at(const Operation& op, std::size_t index) {
    require(index < op.outputs.size() && std::holds_alternative<ValueId>(op.outputs[index]),
            "Operation result must be a value ID");
    return std::get<ValueId>(op.outputs[index]);
}

// The storage scalar of an operation result: an interpretation never changes
// the declared value storage.
inline ScalarType storage_scalar(const StructuredProgram& program, const Operation& op, std::size_t index = 0) {
    const auto id = output_at(op, index);
    require(id.value < program.ir.values.size(), "Result value is outside the value arena");
    switch (program.ir.values[id.value].type) {
    case ValueType::word32: return ScalarType::u32;
    case ValueType::word64: return ScalarType::u64;
    case ValueType::predicate: return ScalarType::boolean;
    }
    fail("Invalid result storage type");
}

// Interpreted scalar of a numeric-mode operand. Binary32 is the only floating
// interpretation with proven exact generators: FP64 has no Metal
// representation and scalar FP16 arithmetic decomposes through f32 with
// explicit pack/unpack operations.
inline ScalarType interpreted_scalar(const Operation& op) {
    switch (op.numeric.domain) {
    case Domain::floating:
        require(op.numeric.scalar == ScalarType::f32,
                "Binary16/binary64 scalar arithmetic has no proven MSL generator",
                DiagnosticCode::unsupported_capability);
        return ScalarType::f32;
    case Domain::signed_integer: return op.numeric.scalar == ScalarType::s64 ? ScalarType::s64 : ScalarType::s32;
    case Domain::unsigned_integer: return op.numeric.scalar == ScalarType::u64 ? ScalarType::u64 : ScalarType::u32;
    case Domain::bits:
        require(op.numeric.scalar == ScalarType::u32 || op.numeric.scalar == ScalarType::u64,
                "Raw-bit operations use word32/word64 storage");
        return op.numeric.scalar;
    case Domain::boolean: fail("Boolean operands require predicate logic, not numeric arithmetic");
    }
    fail("Invalid numeric domain");
}

// Whether the interpreted mode occupies a 64-bit word.
inline bool wide_mode(const Operation& op) {
    return op.numeric.scalar == ScalarType::u64 || op.numeric.scalar == ScalarType::s64;
}

// Emits a named word/long-word constant holding intermediate bits and returns
// its name. Every operation emits once, so operation-unique names cannot
// collide with value names or helper parameters.
inline std::string bits_constant(MslContext& c, OperationId id, unsigned slot, bool wide,
                                 const std::string& initializer) {
    const auto name = "armigo_w" + number(id.value) + "_" + number(slot);
    take(c.statement("const " + std::string(wide ? "ulong" : "uint") + " " + name + " = " + initializer + ";"));
    return name;
}

// Binary32 sign flush at the word level: subnormals become signed zeros.
inline std::string flush_word32(const std::string& bits) {
    return "((" + bits + " & 0x7f800000u) == 0u ? (" + bits + " & 0x80000000u) : (" + bits + "))";
}

// Binary16 sign flush at the half-word level, applied to a uint expression
// holding f16 bits in its low half.
inline std::string flush_half16(const std::string& bits) {
    return "((" + bits + " & 0x7c00u) == 0u ? (" + bits + " & 0x8000u) : (" + bits + "))";
}

// Canonical quiet binary32 NaN for arithmetic results, at the word level.
inline std::string canonical_word32(const std::string& bits) {
    return "((" + bits + " & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : (" + bits + "))";
}

inline std::string as_float(const std::string& bits) { return "as_type<float>(" + bits + ")"; }

// Assigns one operation result from a bit-level expression of the storage's
// raw scalar, after checking the result exists exactly once.
inline Status assign_bits(MslContext& c, const Operation& op, std::size_t index, const std::string& bits) {
    return c.assign(output_at(op, index), bits, ScalarType::u32);
}

// Floating-domain generators (numeric.cpp). Each rejects modes without a
// proven exact contract instead of approximating.
namespace numeric {
Status arithmetic(MslContext&, OperationId, Opcode opcode);
Status extrema(MslContext&, OperationId, Opcode opcode); // minimum or maximum
Status clamp(MslContext&, OperationId);
Status sign(MslContext&, OperationId, bool absolute);
Status compare(MslContext&, OperationId);
Status conversion(MslContext&, OperationId);
Status fused_multiply_add(MslContext&, OperationId);
Status reciprocal(MslContext&, OperationId);
Status reciprocal_sqrt(MslContext&, OperationId);
Status square_root(MslContext&, OperationId);
Status transcendental(MslContext&, OperationId, Opcode opcode); // sine..log2
Status range_reduce(MslContext&, OperationId);
Status integral_round(MslContext&, OperationId, Opcode opcode); // ceiling..truncate
Status is_nan(MslContext&, OperationId);
Status pack_half2(MslContext&, OperationId);
Status unpack_half2(MslContext&, OperationId);
} // namespace numeric
} // namespace armigo::shader::pipeline::msl
