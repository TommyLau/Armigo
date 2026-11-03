// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Floating generation emits native operations under a host policy that pins
// fp math_mode(relaxed) with fast contraction, the native Metal arithmetic:
// add/subtract/divide/multiply are the plain operators, the fused product is
// fma, integral rounding is ceil/floor/rint/trunc, and float-to-integer
// conversions are the native int()/uint() casts. The compiler may reassociate,
// contract and ignore signed zeros. The extrema, clamp and saturation use the
// precise namespace forms explicitly: math mode does not select intrinsics,
// and the unqualified names compile to fast variants. The GPU flushes binary32
// subnormals at arithmetic inputs and results itself, so guest flush
// modifiers stay explicit word-level selects only where a subnormal would pass
// through (sine, cosine, a same-type move) or the compiler folds an operation
// on literals alone. The extrema and the clamp are the precise native
// operations alone, with the departures documented at the extrema, and
// integer-to-float with a directed rounding mode keeps the exact
// integer-assisted helper because the native cast rounds nearest-even only.
// Every explicit NaN test (the unordered comparison flag, ordered NE and the
// NaN test) is an integer test on the word: relaxed math may fold isnan() and
// the NaN-sensitive relations, but not integer arithmetic.

#include "expressions.h"
#include "../dispatch.h"

#include <algorithm>

namespace armigo::shader::pipeline::msl {
namespace numeric {
namespace {

// Whether a float value is NaN, tested on its bits so no math mode folds it.
std::string nan_bits(const std::string& value) {
    return "((as_type<uint>(" + value + ") & 0x7fffffffu) > 0x7f800000u)";
}

void proven_rounding(const Operation& op) {
    require(op.numeric.rounding == Rounding::nearest_even || op.numeric.rounding == Rounding::not_applicable,
            "The floating operation has no proven exact generator for this rounding mode",
            DiagnosticCode::unsupported_modifier);
}

void no_saturation(const Operation& op) {
    require(!op.numeric.saturate, "Guest saturation is an explicit unit-interval operation, not a default",
            DiagnosticCode::unsupported_modifier);
}

void selection_rounding(const Operation& op) {
    // Extrema select an existing word without rounding. Keep the producer's
    // mode, including directed F2I rounding already applied before its clamp.
    switch (op.numeric.rounding) {
    case Rounding::not_applicable:
    case Rounding::nearest_even:
    case Rounding::toward_zero:
    case Rounding::toward_positive:
    case Rounding::toward_negative: return;
    default:
        fail("Floating selection carries an unsupported rounding mode", DiagnosticCode::unsupported_modifier);
    }
}

// Whether every operand is a literal word. The compiler folds such an
// operation at compile time without the GPU's subnormal flush, so its flush
// stays explicit there (and folds away as well).
bool literal_only(const Operation& op) {
    return std::all_of(op.inputs.begin(), op.inputs.end(),
                       [](const auto& input) { return std::holds_alternative<LiteralBits>(input); });
}

// Positional binary32 inputs as host floats. The GPU itself treats a binary32
// subnormal as a signed zero at the inputs and the result of add, subtract,
// multiply, fma, divide, reciprocal, rsqrt, sqrt, exp2, log2, the integral
// roundings, saturation and the relations (device probe), so a guest flush
// modifier on those needs no code. Sine and cosine, which pass a subnormal
// through, flush their inputs explicitly (the extrema and the clamp do not;
// see extrema below), as does any operation on literals alone, which the
// compiler folds without the device: each word moves into a named constant so
// the flush expression never duplicates operand text.
std::vector<std::string> float_inputs(MslContext& c, const Operation& op, OperationId id,
                                      bool passes_subnormals = false) {
    interpreted_scalar(op);
    const bool flush = op.numeric.flush_inputs && (passes_subnormals || literal_only(op));
    std::vector<std::string> values;
    for (std::size_t i = 0; i < op.inputs.size(); ++i) {
        const auto word = operand_at(c, op, i, ScalarType::u32);
        if (flush) {
            values.push_back(as_float(bits_constant(c, id, static_cast<unsigned>(i), false,
                                                    flush_word32(word))));
        } else {
            values.push_back(as_float(word));
        }
    }
    return values;
}

// Emits one word result, then applies the precise unit-interval saturation.
// The GPU flushes an arithmetic result itself, so the guest output flush needs
// code only when the compiler folds the operation (snapshotted so the flush
// expression never duplicates the operation text). The precise clamp maps NaN
// and negatives to plus zero like the guest modifier.
Status emit_word_result(MslContext& c, const Operation& op, OperationId id, const std::string& bits,
                        bool saturate) {
    std::string result = bits;
    if (op.numeric.flush_output && literal_only(op))
        result = flush_word32(bits_constant(c, id, 7u, false, result));
    if (saturate) result = "as_type<uint>(precise::clamp(as_type<float>(" + result + "), 0.0f, 1.0f))";
    return assign_bits(c, op, 0, result);
}

} // namespace

// Native binary32 addition, subtraction, multiplication and division use plain
// operators; the GPU flushes their subnormal inputs and results.
Status arithmetic(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Binary arithmetic takes two inputs and one result");
        if (opcode == Opcode::remainder) {
            fail("A floating remainder has no guest producer and no proven generator",
                 DiagnosticCode::unsupported_capability);
        }
        if (opcode == Opcode::multiply && op.numeric.rounding == Rounding::toward_zero) {
            // Metal has no directed-rounding product. FMUL.RZ (texture
            // coordinates rebuilt from 1/w, and sine and cosine range
            // reduction) is emitted as the native product, rounded to nearest:
            // a deliberate departure of at most one ulp, matching the
            // reference driver's arithmetic for these instructions. The exact
            // toward-zero emulation cost several percent of the frame at
            // native resolution. Beyond the one ulp, a product reaching
            // 2^128 - 2^103 becomes infinity instead of the largest finite
            // value, NaN results are not canonical, and, like every native
            // product under relaxed math, it may fuse into a following add.
            // Subnormals flush at both boundaries, as the GPU's native product
            // does, so the mode keeps requiring both flush flags.
            require(op.numeric.flush_inputs && op.numeric.flush_output,
                    "The toward-zero product flushes subnormals at both boundaries by contract",
                    DiagnosticCode::unsupported_modifier);
            // The inputs go through the same explicit flush as every native
            // product, so a product of literals still flushes subnormals.
            const auto values = float_inputs(c, op, id);
            take(emit_word_result(c, op, id, "as_type<uint>(" + values[0] + " * " + values[1] + ")",
                                  op.numeric.saturate));
            return;
        }
        proven_rounding(op);
        const auto values = float_inputs(c, op, id);
        const auto symbol = opcode == Opcode::add ? " + "
                            : opcode == Opcode::subtract ? " - "
                            : opcode == Opcode::multiply ? " * "
                            : " / ";
        take(emit_word_result(c, op, id,
                              "as_type<uint>(" + values[0] + symbol + values[1] + ")",
                              op.numeric.saturate));
    });
}

// Floating extrema use native min/max over interpreted values in the precise
// namespace so the intrinsic itself is not a fast variant: on the device
// exactly one NaN operand yields the other and two NaNs yield a NaN. That is
// the whole contract of the emitted form. Deliberate departures from the
// portable CPU contract, shared with the reference driver's native min/max:
// a tie between operands the device compares as zero (zeros and subnormals)
// of opposite sign returns the device's choice instead of ordering -0 below
// +0, a NaN result keeps the device's payload instead of the canonical quiet
// NaN, and subnormal operands or results are not flushed. A kept subnormal is
// seen only where it is stored or read as an integer word, or through such a
// tie (min(0x00000001, -0) may be +tiny, whose reciprocal is +inf instead of
// -inf); arithmetic consumers flush it. Operands that are all literals keep
// the explicit flush because the compiler folds those without the device.
Status extrema(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Extremum takes two inputs and one result");
        selection_rounding(op);
        no_saturation(op);
        const bool minimum = opcode == Opcode::minimum;
        const auto values = float_inputs(c, op, id);
        const std::string function = minimum ? "precise::fmin" : "precise::fmax";
        std::string result = "as_type<uint>(" + function + "(" + values[0] + ", " + values[1] + "))";
        if (op.numeric.flush_output && literal_only(op)) result = flush_word32(bits_constant(c, id, 4u, false, result));
        take(assign_bits(c, op, 0, result));
    });
}

// The guest clamp composes min(max(value, low), high)
// in the precise namespace so neither intrinsic is a fast variant: a NaN
// value yields the low bound, and the bounds pass through unchanged. The
// subnormal flush follows the extrema rule above.
Status clamp(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 3 && op.outputs.size() == 1, "Clamp takes three inputs and one result");
        selection_rounding(op);
        no_saturation(op);
        const auto values = float_inputs(c, op, id);
        std::string result = "as_type<uint>(precise::fmin(precise::fmax(" + values[0] + ", " +
                             values[1] + "), " + values[2] + "))";
        if (op.numeric.flush_output && literal_only(op)) result = flush_word32(bits_constant(c, id, 7u, false, result));
        take(assign_bits(c, op, 0, result));
    });
}

// Exact sign-bit operations: neither performs arithmetic, so neither flushes
// nor saturates, and no NaN is canonicalized.
Status sign(MslContext& c, OperationId id, bool absolute) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Sign operations take one input and one result");
        require(!op.numeric.flush_inputs && !op.numeric.flush_output,
                "A sign operation performs no arithmetic and cannot flush",
                DiagnosticCode::unsupported_modifier);
        proven_rounding(op);
        no_saturation(op);
        interpreted_scalar(op);
        const auto word = operand_at(c, op, 0, ScalarType::u32);
        take(assign_bits(c, op, 0, absolute ? "(" + word + " & 0x7fffffffu)" : "(" + word + " ^ 0x80000000u)"));
    });
}

// Floating comparisons use ordered operators on interpreted values, and the
// unordered flag adds explicit NaN terms.
// The C-style inequality is an unordered test, and relaxed math rewrites the
// ordered pair (a < b) || (a > b) into it, so the ordered not-equal relation
// excludes NaN operands explicitly: false whenever either operand is NaN,
// exactly like the guest FSETP.OF-family encodings.
Status compare(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Comparison takes two inputs and one predicate");
        require(std::holds_alternative<ComparisonInfo>(op.info), "Comparison carries no comparison metadata");
        const auto& comparison = std::get<ComparisonInfo>(op.info);
        proven_rounding(op);
        const auto values = float_inputs(c, op, id);
        std::string expression;
        switch (comparison.relation) {
        case Compare::never: expression = "false"; break;
        case Compare::always: expression = "true"; break;
        case Compare::not_equal:
            if (comparison.unordered) {
                expression = "(" + values[0] + " != " + values[1] + " || " + nan_bits(values[0]) +
                             " || " + nan_bits(values[1]) + ")";
            } else {
                expression = "(" + values[0] + " != " + values[1] + " && !" + nan_bits(values[0]) +
                             " && !" + nan_bits(values[1]) + ")";
            }
            break;
        default: {
            const char* symbol = "<invalid>";
            switch (comparison.relation) {
            case Compare::less: symbol = "<"; break;
            case Compare::equal: symbol = "=="; break;
            case Compare::less_equal: symbol = "<="; break;
            case Compare::greater: symbol = ">"; break;
            case Compare::greater_equal: symbol = ">="; break;
            default: fail("Unknown comparison relation");
            }
            expression = "(" + values[0] + " " + symbol + " " + values[1] + ")";
            if (comparison.unordered)
                expression = "(" + expression + " || " + nan_bits(values[0]) + " || " + nan_bits(values[1]) + ")";
            break;
        }
        }
        take(c.assign(output_at(op, 0), expression, ScalarType::boolean));
    });
}

// The fused product uses native single-rounding fma.
Status fused_multiply_add(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 3 && op.outputs.size() == 1, "Fused multiply-add takes three inputs and one result");
        proven_rounding(op);
        const auto values = float_inputs(c, op, id);
        take(emit_word_result(c, op, id,
                              "as_type<uint>(fma(" + values[0] + ", " + values[1] + ", " + values[2] + "))",
                              op.numeric.saturate));
    });
}

// MUFU.RCP uses native division of one by the source. The guest multifunction
// unit is an approximation, so Metal's own
// quotient defines the rounding and the NaN, infinity and zero results. The
// exact reciprocal helper stays reserved for non-multifunction users.
Status reciprocal(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Reciprocal takes one input and one result");
        proven_rounding(op);
        const auto values = float_inputs(c, op, id);
        take(emit_word_result(c, op, id, "as_type<uint>(1.0f / " + values[0] + ")",
                              op.numeric.saturate));
    });
}

// Native square root; MUFU.RSQ is emitted separately below.
Status square_root(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Square root takes one input and one result");
        proven_rounding(op);
        const auto values = float_inputs(c, op, id);
        take(emit_word_result(c, op, id, "as_type<uint>(sqrt(" + values[0] + "))",
                              op.numeric.saturate));
    });
}

// MUFU.RSQ uses native approximate inverse square root. The guest unit is
// itself approximate, so Metal's rsqrt defines
// the result, including its NaN, infinity and zero handling.
Status reciprocal_sqrt(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1,
                "Reciprocal square root takes one input and one result");
        proven_rounding(op);
        const auto values = float_inputs(c, op, id);
        take(emit_word_result(c, op, id, "as_type<uint>(rsqrt(" + values[0] + "))",
                              op.numeric.saturate));
    });
}

// Elementary functions use native MSL functions; their accuracy contracts
// need device probes before any stronger
// claim.
Status transcendental(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Elementary functions take one input and one result");
        proven_rounding(op);
        const auto function = opcode == Opcode::sine   ? "sin"
                              : opcode == Opcode::cosine ? "cos"
                              : opcode == Opcode::exp2   ? "exp2"
                              : "log2";
        // Native sine and cosine pass a subnormal through, so a flush request
        // (which the guest never makes) stays explicit at their input; a
        // flushed input cannot produce a subnormal result.
        const auto values = float_inputs(c, op, id, opcode == Opcode::sine || opcode == Opcode::cosine);
        take(emit_word_result(c, op, id,
                              "as_type<uint>(" + std::string(function) + "(" + values[0] + "))",
                              op.numeric.saturate));
    });
}

// The range-reduction modifier lowers to the source unchanged in the
// established composition: the paired function consumes the value
// directly, with no arithmetic and no NaN canonicalization.
Status range_reduce(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Range reduction takes one input and one result");
        require(!op.numeric.flush_inputs && !op.numeric.flush_output && !op.numeric.saturate,
                "Range reduction is a bit-exact pass-through and cannot flush or saturate",
                DiagnosticCode::unsupported_modifier);
        proven_rounding(op);
        take(assign_bits(c, op, 0, operand_at(c, op, 0, ScalarType::u32)));
    });
}

// Integral rounding by the native roundToIntegral functions: exact on every
// finite input, infinities pass through, and a NaN propagates without
// canonicalization, matching the same-size rounding contract.
Status integral_round(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Integral rounding takes one input and one result");
        const auto rounding = opcode == Opcode::ceiling ? Rounding::toward_positive
                              : opcode == Opcode::floor ? Rounding::toward_negative
                              : opcode == Opcode::round ? Rounding::nearest_even
                              : Rounding::toward_zero;
        // The opcode fixes the integral rounding direction. Admit the matching
        // decoded mode as well as the existing neutral/nearest-even IR form.
        require(op.numeric.rounding == rounding || op.numeric.rounding == Rounding::nearest_even ||
                    op.numeric.rounding == Rounding::not_applicable,
                "Integral rounding direction disagrees with its opcode",
                DiagnosticCode::unsupported_modifier);
        const auto function = opcode == Opcode::ceiling ? "ceil"
                              : opcode == Opcode::floor ? "floor"
                              : opcode == Opcode::round ? "rint"
                              : "trunc";
        const auto values = float_inputs(c, op, id);
        take(emit_word_result(c, op, id,
                              "as_type<uint>(" + std::string(function) + "(" + values[0] + "))",
                              op.numeric.saturate));
    });
}

Status is_nan(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "The NaN test takes one input and one predicate");
        require(!op.numeric.flush_inputs, "A NaN test is unaffected by flushing and demands no flush",
                DiagnosticCode::unsupported_modifier);
        proven_rounding(op);
        interpreted_scalar(op);
        const auto word = operand_at(c, op, 0, ScalarType::u32);
        take(c.assign(output_at(op, 0), "((" + word + " & 0x7fffffffu) > 0x7f800000u)", ScalarType::boolean));
    });
}

// Two binary32 words pack into one binary16 pair with a nearest-even
// conversion per component, as one native packed conversion. A binary32
// subnormal input rounds to signed zero at the binary16 boundary on its own;
// only the demanded output flush at the binary16 boundary stays explicit.
Status pack_half2(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Half packing takes two words and one packed result");
        proven_rounding(op);
        no_saturation(op);
        const auto first = as_float(operand_at(c, op, 0, ScalarType::u32));
        const auto second = as_float(operand_at(c, op, 1, ScalarType::u32));
        if (!op.numeric.flush_output) {
            take(assign_bits(c, op, 0, "as_type<uint>(half2(" + first + ", " + second + "))"));
            return;
        }
        // Snapshot both half results: an output may alias an input word, and
        // each flushed half feeds the joined word twice.
        const auto low = bits_constant(c, id, 0, false,
            "uint(as_type<ushort>(half(" + first + ")))");
        const auto high = bits_constant(c, id, 1, false,
            "uint(as_type<ushort>(half(" + second + ")))");
        const auto flushed_low = flush_half16(low);
        const auto flushed_high = flush_half16(high);
        take(assign_bits(c, op, 0, "(" + flushed_low + " | (" + flushed_high + " << 16u))"));
    });
}

// One packed binary16 pair widens exactly into two binary32 words as the two
// lanes of one native half2; only the demanded binary16 input flush changes
// bits.
Status unpack_half2(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 2, "Half unpacking takes one word and low/high results");
        proven_rounding(op);
        no_saturation(op);
        require(!op.numeric.flush_output, "A binary16 source is never a binary32 subnormal",
                DiagnosticCode::unsupported_modifier);
        // Snapshot the packed word: a result may alias its input.
        const auto word = bits_constant(c, id, 0, false, operand_at(c, op, 0, ScalarType::u32));
        if (!op.numeric.flush_inputs) {
            // The f32 assignment converts the widened lane to its word.
            take(c.assign(output_at(op, 0), "float(as_type<half2>(" + word + ").x)", ScalarType::f32));
            take(c.assign(output_at(op, 1), "float(as_type<half2>(" + word + ").y)", ScalarType::f32));
            return;
        }
        auto low = "(" + word + " & 0xffffu)";
        auto high = "(" + word + " >> 16u)";
        low = bits_constant(c, id, 1, false, flush_half16(low));
        high = bits_constant(c, id, 2, false, flush_half16(high));
        take(c.assign(output_at(op, 0), "float(as_type<half>(ushort(" + low + ")))", ScalarType::f32));
        take(c.assign(output_at(op, 1), "float(as_type<half>(ushort(" + high + ")))", ScalarType::f32));
    });
}

Status conversion(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Conversion takes one input and one result");
        require(std::holds_alternative<ConversionInfo>(op.info), "Conversion carries no conversion metadata");
        const auto& info = std::get<ConversionInfo>(op.info);
        require(info.source != ScalarType::f64 && info.destination != ScalarType::f64,
                "Metal has no native FP64 representation", DiagnosticCode::unsupported_capability);
        const bool source_floating = scalar_domain(info.source) == Domain::floating;
        if (source_floating && scalar_domain(info.destination) == Domain::floating) {
            if (info.source == info.destination) {
                // A same-type conversion does not round, but still owns its
                // input/output flush boundaries. Integral rounding uses the
                // dedicated integral-round opcodes.
                require(op.numeric.rounding == Rounding::not_applicable ||
                            op.numeric.rounding == Rounding::nearest_even,
                        "Same-type conversion preserves value; integral rounding uses dedicated opcodes",
                        DiagnosticCode::unsupported_modifier);
                require(info.source == ScalarType::f32 ||
                            (!op.numeric.flush_inputs && !op.numeric.flush_output && !op.numeric.saturate),
                        "Modified same-type half conversions require explicit unpack/pack boundaries",
                        DiagnosticCode::unsupported_modifier);
                // With no arithmetic between the boundaries, either flush
                // request has the same signed-zero transformation; the flush
                // stays explicit because a pass-through move flushes nothing.
                std::string result = operand_at(c, op, 0, ScalarType::u32);
                if (op.numeric.flush_inputs || op.numeric.flush_output)
                    result = flush_word32(result);
                take(emit_word_result(c, op, id, result, op.numeric.saturate));
                return;
            }
            if (info.source == ScalarType::f16) {
                // Binary16 widening is the native half-to-float conversion and
                // is exact: only the demanded input flush at the binary16
                // boundary changes bits.
                require(op.numeric.rounding == Rounding::not_applicable ||
                            op.numeric.rounding == Rounding::nearest_even,
                        "Binary16 widening rounds nothing", DiagnosticCode::unsupported_modifier);
                no_saturation(op);
                require(!op.numeric.flush_output, "A binary16 source is never a binary32 subnormal",
                        DiagnosticCode::unsupported_modifier);
                const auto word = operand_at(c, op, 0, ScalarType::u32);
                std::string low_half;
                if (op.numeric.flush_inputs) {
                    low_half = bits_constant(c, id, 0, false, flush_half16("(" + word + " & 0xffffu)"));
                } else {
                    low_half = "(" + word + " & 0xffffu)";
                }
                take(c.assign(output_at(op, 0), "float(as_type<half>(ushort(" + low_half + ")))", ScalarType::f32));
                return;
            }
            // Binary32 narrowing is the native float-to-half conversion and
            // rounds to nearest even in the target; other modes are unproven.
            require(op.numeric.rounding == Rounding::not_applicable ||
                        op.numeric.rounding == Rounding::nearest_even,
                    "Binary32 narrowing rounds to nearest even; other modes are unproven",
                    DiagnosticCode::unsupported_modifier);
            no_saturation(op);
            const auto word = operand_at(c, op, 0, ScalarType::u32);
            const auto half_bits = "uint(as_type<ushort>(half(as_type<float>(" + word + "))))";
            if (!op.numeric.flush_output) {
                take(assign_bits(c, op, 0, half_bits));
                return;
            }
            const auto bits = bits_constant(c, id, 0, false, half_bits);
            take(assign_bits(c, op, 0, flush_half16(bits)));
            return;
        }
        if (scalar_domain(info.destination) == Domain::floating) {
            require(!op.numeric.flush_inputs && !op.numeric.flush_output,
                    "An integer boundary has no subnormal inputs or outputs",
                    DiagnosticCode::unsupported_modifier);
            no_saturation(op);
            require(scalar_bits(info.source) <= 32,
                    "A 64-bit integer source has no proven exact conversion",
                    DiagnosticCode::unsupported_capability);
            const bool signed_source = scalar_domain(info.source) == Domain::signed_integer;
            const auto bits = scalar_bits(info.source);
            if (op.numeric.rounding == Rounding::toward_zero) {
                // A directed rounding has no native form: the native cast
                // rounds nearest-even only. The exact integer-assisted helper
                // assembles the bits with the guest truncation, so the halfway
                // result differs from the host cast exactly as the guest
                // demands.
                std::string word;
                if (bits == 8) {
                    // Eight-bit words normalize into the 32-bit source the
                    // exact conversion accepts, preserving the interpreted
                    // sign.
                    const auto text = operand_at(c, op, 0, info.source);
                    word = bits_constant(c, id, 0u, false,
                        signed_source ? "uint(int(" + text + "))" : "uint(" + text + ")");
                } else {
                    // Sixteen-bit sources are masked and sign-extended inside
                    // the exact conversion, so the raw word is its input.
                    word = operand_at(c, op, 0, ScalarType::u32);
                }
                take(c.require_helper("armigo_i2f"));
                take(assign_bits(c, op, 0, "armigo_i2f(" + word + ", " +
                                             (signed_source ? "true" : "false") + ", " +
                                             (bits == 16 ? "16u" : "32u") + ", true, false)"));
                return;
            }
            require(op.numeric.rounding == Rounding::nearest_even ||
                        op.numeric.rounding == Rounding::not_applicable,
                    "Integer-to-float conversion admits nearest-even natively; other directed modes need the exact helper",
                    DiagnosticCode::unsupported_modifier);
            // The typed operand text carries the interpreted signedness.
            const auto source = operand_at(c, op, 0, info.source);
            take(assign_bits(c, op, 0, "as_type<uint>(float(" + source + "))"));
            return;
        }
        // Binary32 to integer uses native int()/uint() casts. The
        // emitter applies the guest integral rounding and its clamps as
        // explicit operations before this cast, and a subnormal source
        // truncates to signed zero on its own. Out-of-range results follow the
        // host cast on this platform.
        require(info.source == ScalarType::f32,
                "Binary16-to-integer conversion decomposes through f32",
                DiagnosticCode::unsupported_capability);
        require(op.numeric.rounding == Rounding::toward_zero ||
                    op.numeric.rounding == Rounding::nearest_even ||
                    op.numeric.rounding == Rounding::not_applicable,
                "The conversion cast truncates; the guest rounding is a dedicated rounding operation",
                DiagnosticCode::unsupported_modifier);
        require(!op.numeric.saturate,
                "Destination-bound clamping belongs to the conversion contract, not a saturation modifier",
                DiagnosticCode::unsupported_modifier);
        const bool signed_destination = scalar_domain(info.destination) == Domain::signed_integer;
        require(scalar_bits(info.destination) == 32,
                "Narrow integer conversions decompose through a 32-bit carrier with explicit clamps",
                DiagnosticCode::unsupported_capability);
        const auto value = as_float(operand_at(c, op, 0, ScalarType::u32));
        take(assign_bits(c, op, 0, signed_destination ? "as_type<uint>(int(" + value + "))"
                                                      : "uint(" + value + ")"));
    });
}
} // namespace numeric
} // namespace armigo::shader::pipeline::msl
