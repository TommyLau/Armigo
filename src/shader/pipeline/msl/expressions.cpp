// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Per-opcode MSL generators operate on owned words: modular integer arithmetic
// stays at the raw
// word level, signed division guards the most-negative quotient, and find
// first/last bit return -1 (all bits set) for an empty result.

#include "expressions.h"
#include "../dispatch.h"

namespace armigo::shader::pipeline {
using namespace msl;
namespace {

const ConversionInfo& conversion_info(const Operation& op) {
    require(std::holds_alternative<ConversionInfo>(op.info), "Conversion operation carries no conversion metadata");
    return std::get<ConversionInfo>(op.info);
}

const ComparisonInfo& comparison_info(const Operation& op) {
    require(std::holds_alternative<ComparisonInfo>(op.info), "Comparison carries no comparison metadata");
    return std::get<ComparisonInfo>(op.info);
}

// A widening integral conversion is the destination construction of the
// source text: unsigned destinations take the source modulo the width and
// wider signed destinations are value-preserving, exactly as conversions
// behave in the source language.
std::string widened(const std::string& source, ScalarType destination) {
    return std::string(scalar(destination)) + "(" + source + ")";
}

// A narrowing integral conversion masks the raw word at the destination width
// and reinterprets, so no signed-narrowing rule of the target language is
// consulted.
std::string narrowed(const std::string& source, ScalarType source_type, ScalarType destination) {
    const auto word = raw(source, source_type);
    switch (destination) {
    case ScalarType::u8: return "uchar(" + word + " & 0xffu)";
    case ScalarType::s8: return "as_type<char>(uchar(" + word + " & 0xffu))";
    case ScalarType::u16: return "ushort(" + word + " & 0xffffu)";
    case ScalarType::s16: return "as_type<short>(ushort(" + word + " & 0xffffu))";
    case ScalarType::u32:
        return scalar_bits(source_type) == 64 ? "uint(" + word + " & 0xfffffffful)" : word;
    case ScalarType::s32:
        return scalar_bits(source_type) == 64
            ? "as_type<int>(uint(" + word + " & 0xfffffffful))"
            : "as_type<int>(" + word + ")";
    default: fail("Narrowing into a 64-bit destination is a widening, not a narrowing");
    }
}

Status integer_conversion(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        const auto& conversion = conversion_info(op);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Conversion takes exactly one input and result");
        const auto source = operand_at(c, op, 0, conversion.source);
        const auto source_bits = scalar_bits(conversion.source);
        const auto destination_bits = scalar_bits(conversion.destination);
        std::string expression;
        if (source_bits < destination_bits) expression = widened(source, conversion.destination);
        else if (source_bits > destination_bits) expression = narrowed(source, conversion.source, conversion.destination);
        else expression = "as_type<" + std::string(scalar(conversion.destination)) + ">(" + source + ")";
        take(c.assign(output_at(op, 0), expression, conversion.destination));
    });
}

// Modular integer arithmetic at the raw word: addition, subtraction and
// multiplication are interpretation-independent. Division and remainder use
// the interpreted domain; the signed forms guard the most-negative quotient
// that would otherwise be undefined in the target language.
Status integer_arithmetic(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Binary arithmetic takes two inputs and one result");
        const bool wide = wide_mode(op);
        const bool signed_domain = op.numeric.domain == Domain::signed_integer;
        const auto raw_scalar = wide ? ScalarType::u64 : ScalarType::u32;
        const auto typed_scalar = signed_domain ? (wide ? ScalarType::s64 : ScalarType::s32) : raw_scalar;
        const auto modular_zero = wide ? "0ul" : "0u";
        const auto typed_zero = signed_domain ? (wide ? "0l" : "0") : modular_zero;
        const bool modular = opcode == Opcode::add || opcode == Opcode::subtract || opcode == Opcode::multiply;
        const auto left = operand_at(c, op, 0, modular ? raw_scalar : typed_scalar);
        const auto right = operand_at(c, op, 1, modular ? raw_scalar : typed_scalar);
        std::string expression;
        switch (opcode) {
        case Opcode::add: expression = "(" + left + " + " + right + ")"; break;
        case Opcode::subtract: expression = "(" + left + " - " + right + ")"; break;
        case Opcode::multiply: expression = "(" + left + " * " + right + ")"; break;
        case Opcode::divide:
            expression = signed_domain
                ? "((" + right + " == -1) ? as_type<" + std::string(scalar(typed_scalar)) + ">(" + modular_zero +
                      " - " + (wide ? "as_type<ulong>(" : "as_type<uint>(") + left + ")) : (" + left + " / " + right + "))"
                : "(" + left + " / " + right + ")";
            break;
        case Opcode::remainder:
            expression = signed_domain
                ? "((" + right + " == -1) ? " + typed_zero + " : (" + left + " % " + right + "))"
                : "(" + left + " % " + right + ")";
            break;
        default: fail("Unknown arithmetic opcode");
        }
        // Modular results are assigned at the raw word; quotient and remainder
        // keep their interpreted signedness so negative values re-encode
        // exactly.
        take(c.assign(output_at(op, 0), expression,
                      opcode == Opcode::divide || opcode == Opcode::remainder ? typed_scalar : raw_scalar));
    });
}

Status integer_extremum(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Extremum takes two inputs and one result");
        const auto type = interpreted_scalar(op);
        const auto left = operand_at(c, op, 0, type);
        const auto right = operand_at(c, op, 1, type);
        const std::string function = opcode == Opcode::minimum ? "min" : "max";
        take(c.assign(output_at(op, 0), function + "(" + left + ", " + right + ")", type));
    });
}

Status integer_clamp(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 3 && op.outputs.size() == 1, "Clamp takes three inputs and one result");
        const auto type = interpreted_scalar(op);
        // Snapshot every operand: the composed minimum/maximum references each
        // one twice, and a destination may alias an input. Keep the interpreted
        // type: unsigned snapshots would turn negative signed bounds positive.
        const auto snapshot = [&](unsigned slot) {
            const auto name = "armigo_w" + number(id.value) + "_" + number(slot);
            take(c.statement("const " + std::string(scalar(type)) + " " + name + " = " +
                             operand_at(c, op, slot, type) + ";"));
            return name;
        };
        const auto value = snapshot(0);
        const auto low = snapshot(1);
        const auto high = snapshot(2);
        take(c.assign(output_at(op, 0), "min(max(" + value + ", " + low + "), " + high + ")", type));
    });
}

Status integer_sign(MslContext& c, OperationId id, bool absolute) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Sign operations take one input and one result");
        require(!absolute || op.numeric.domain == Domain::signed_integer,
                "Absolute of an unsigned interpretation is not a distinct operation",
                DiagnosticCode::unsupported_modifier);
        const bool wide = wide_mode(op);
        const auto raw_scalar = wide ? ScalarType::u64 : ScalarType::u32;
        const auto bits = operand_at(c, op, 0, raw_scalar);
        const auto sign_mask = wide ? "0x8000000000000000ul" : "0x80000000u";
        const auto zero = wide ? std::string("0ul") : std::string("0u");
        const auto expression = absolute
            ? "(" + bits + " < " + sign_mask + " ? " + bits + " : " + zero + " - " + bits + ")"
            : "(" + zero + " - " + bits + ")";
        take(assign_bits(c, op, 0, expression));
    });
}

Status multiply_high(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Multiply-high takes two inputs and one result");
        require(!wide_mode(op), "A 128-bit product has no proven MSL decomposition",
                DiagnosticCode::unsupported_capability);
        const auto left = operand_at(c, op, 0, ScalarType::u32);
        const auto right = operand_at(c, op, 1, ScalarType::u32);
        const auto wide_product = op.numeric.domain == Domain::signed_integer
            ? "as_type<ulong>(long(as_type<int>(" + left + ")) * long(as_type<int>(" + right + ")))"
            : "(ulong(" + left + ") * ulong(" + right + "))";
        take(assign_bits(c, op, 0, "uint(" + wide_product + " >> 32u)"));
    });
}

Status carry_operation(MslContext& c, OperationId id, bool addition) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 3 && op.outputs.size() == 2, "Carry arithmetic takes left/right/carry and result/carry");
        const auto left = operand_at(c, op, 0, ScalarType::u32);
        const auto right = operand_at(c, op, 1, ScalarType::u32);
        const auto carry_in = operand_at(c, op, 2, ScalarType::boolean);
        const auto incoming = carry_in + " ? 1ul : 0ul";
        const auto wide = bits_constant(c, id, 0, true, addition
            ? "(ulong(" + left + ") + ulong(" + right + ") + " + incoming + ")"
            : "(ulong(" + left + ") - ulong(" + right + ") - " + incoming + ")");
        take(assign_bits(c, op, 0, "uint(" + wide + ")"));
        take(c.assign(output_at(op, 1), "(" + wide + " > 0xfffffffful)", ScalarType::boolean));
    });
}

Status predicate_logic(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.outputs.size() == 1, "Predicate logic takes one result");
        std::string expression;
        switch (opcode) {
        case Opcode::logical_and:
            require(op.inputs.size() == 2, "Logical conjunction takes two inputs");
            expression = "(" + operand_at(c, op, 0, ScalarType::boolean) + " && " +
                         operand_at(c, op, 1, ScalarType::boolean) + ")";
            break;
        case Opcode::logical_or:
            require(op.inputs.size() == 2, "Logical disjunction takes two inputs");
            expression = "(" + operand_at(c, op, 0, ScalarType::boolean) + " || " +
                         operand_at(c, op, 1, ScalarType::boolean) + ")";
            break;
        case Opcode::logical_xor:
            require(op.inputs.size() == 2, "Logical exclusive-or takes two inputs");
            expression = "(" + operand_at(c, op, 0, ScalarType::boolean) + " != " +
                         operand_at(c, op, 1, ScalarType::boolean) + ")";
            break;
        case Opcode::logical_not:
            require(op.inputs.size() == 1, "Logical negation takes one input");
            expression = "(!" + operand_at(c, op, 0, ScalarType::boolean) + ")";
            break;
        default: fail("Unknown predicate opcode");
        }
        take(c.assign(output_at(op, 0), expression, ScalarType::boolean));
    });
}

Status bit_binary(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Bitwise arithmetic takes two inputs and one result");
        const bool wide = wide_mode(op);
        const auto raw_scalar = wide ? ScalarType::u64 : ScalarType::u32;
        const auto left = operand_at(c, op, 0, raw_scalar);
        const auto right = operand_at(c, op, 1, raw_scalar);
        const auto function = opcode == Opcode::bit_and ? "&" : opcode == Opcode::bit_or ? "|" : "^";
        take(assign_bits(c, op, 0, "(" + left + " " + function + " " + right + ")"));
    });
}

Status shift(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 2 && op.outputs.size() == 1, "Shift takes a value, a count and one result");
        const bool wide = wide_mode(op);
        const bool arithmetic = op.numeric.domain == Domain::signed_integer;
        const auto count = operand_at(c, op, 1, ScalarType::u32);
        const auto mask = wide ? "63u" : "31u";
        std::string value;
        std::string expression;
        if (opcode == Opcode::shift_left) {
            // A left shift is a raw word movement in both interpretations.
            value = operand_at(c, op, 0, wide ? ScalarType::u64 : ScalarType::u32);
            expression = "(" + value + " << (" + count + " & " + mask + "))";
        } else {
            value = operand_at(c, op, 0, arithmetic ? (wide ? ScalarType::s64 : ScalarType::s32)
                                                    : (wide ? ScalarType::u64 : ScalarType::u32));
            // The target language defines a right shift of a negative value as
            // an arithmetic shift, matching the interpreted domain.
            expression = "(" + value + " >> (" + count + " & " + mask + "))";
        }
        take(assign_bits(c, op, 0, expression));
    });
}

Status funnel_shift(MslContext& c, OperationId id, bool left) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 3 && op.outputs.size() == 1, "Funnel shift takes low/high/count and one result");
        require(!wide_mode(op), "The funnel shift is a 32-bit operation", DiagnosticCode::unsupported_capability);
        const auto low = operand_at(c, op, 0, ScalarType::u32);
        const auto high = operand_at(c, op, 1, ScalarType::u32);
        const auto count = operand_at(c, op, 2, ScalarType::u32);
        const auto masked = "(" + count + " & 31u)";
        const auto joined = "((ulong(" + high + ") << 32u) | ulong(" + low + "))";
        // Direction-selected word of the joined pair with the count masked to
        // the word width: shifting left by zero selects the high word and
        // shifting right by zero selects the low word.
        take(assign_bits(c, op, 0, left ? "uint((" + joined + " << " + masked + ") >> 32u)"
                                        : "uint(" + joined + " >> " + masked + ")"));
    });
}

Status bit_field(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(!wide_mode(op), "Bit-field extraction and insertion are 32-bit operations",
                DiagnosticCode::unsupported_capability);
        // Metal has no GLSL-style bitfield intrinsics, so extraction and
        // insertion are composed from masks. Retain a full-word count of 32;
        // field offsets/counts are bounded, unlike masked shift counts. Empty
        // and out-of-word fields extract zero and leave an insertion unchanged.
        if (opcode == Opcode::bit_extract) {
            require(op.inputs.size() == 3 && op.outputs.size() == 1, "Extraction takes value/start/count and one result");
            const auto signed_domain = op.numeric.domain == Domain::signed_integer;
            // The field is taken from the raw word and sign-extended only
            // afterwards, so the interpreted sign never shifts arithmetic.
            const auto value = bits_constant(c, id, 0, false, operand_at(c, op, 0, ScalarType::u32));
            const auto offset_text = bits_constant(c, id, 1, false,
                "min(" + operand_at(c, op, 1, ScalarType::u32) + ", 32u)");
            const auto count = bits_constant(c, id, 2, false,
                "min(" + operand_at(c, op, 2, ScalarType::u32) + ", 32u - " + offset_text + ")");
            const auto field = "((" + value + " >> (" + offset_text + " & 31u)) & (" + count +
                " == 0u ? 0u : (0xffffffffu >> ((32u - " + count + ") & 31u))))";
            const auto expression = signed_domain
                ? "((" + count + " == 0u) ? 0u : as_type<uint>(as_type<int>((" + field + ") << ((32u - " + count +
                      ") & 31u)) >> ((32u - " + count + ") & 31u)))"
                : field;
            take(assign_bits(c, op, 0, expression));
        } else {
            require(op.inputs.size() == 4 && op.outputs.size() == 1, "Insertion takes base/data/start/count and one result");
            const auto base = operand_at(c, op, 0, ScalarType::u32);
            const auto data = operand_at(c, op, 1, ScalarType::u32);
            const auto offset_text = bits_constant(c, id, 0, false,
                "min(" + operand_at(c, op, 2, ScalarType::u32) + ", 32u)");
            const auto count = bits_constant(c, id, 1, false,
                "min(" + operand_at(c, op, 3, ScalarType::u32) + ", 32u - " + offset_text + ")");
            const auto mask = "(" + count + " == 0u ? 0u : ((0xffffffffu >> ((32u - " + count + ") & 31u)) << (" +
                              offset_text + " & 31u)))";
            take(assign_bits(c, op, 0,
                             "((" + base + " & ~" + mask + ") | ((" + data + " << (" + offset_text + " & 31u)) & " + mask + "))"));
        }
    });
}

Status word_query(MslContext& c, OperationId id, Opcode opcode) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Word queries take one input and one result");
        const bool wide = wide_mode(op);
        const bool signed_domain = op.numeric.domain == Domain::signed_integer;
        require(!(wide && opcode == Opcode::bit_reverse), "A 64-bit bit reversal has no proven MSL decomposition",
                DiagnosticCode::unsupported_capability);
        const auto raw_scalar = wide ? ScalarType::u64 : ScalarType::u32;
        const auto value = bits_constant(c, id, 0, wide, operand_at(c, op, 0, raw_scalar));
        const auto zero = wide ? "0ul" : "0u";
        const auto all_bits = "0xffffffffu";
        const auto width = wide ? "63u" : "31u";
        std::string expression;
        switch (opcode) {
        case Opcode::bit_reverse: {
            // A 32-bit bit reversal by successive mask-and-swap rounds.
            const auto first = "((" + value + " & 0x55555555u) << 1u) | ((" + value + " >> 1u) & 0x55555555u)";
            const auto second = bits_constant(c, id, 1, false, first);
            const auto third = bits_constant(c, id, 2, false,
                "((" + second + " & 0x33333333u) << 2u) | ((" + second + " >> 2u) & 0x33333333u)");
            const auto fourth = bits_constant(c, id, 3, false,
                "((" + third + " & 0x0f0f0f0fu) << 4u) | ((" + third + " >> 4u) & 0x0f0f0f0fu)");
            const auto fifth = bits_constant(c, id, 4, false,
                "((" + fourth + " & 0x00ff00ffu) << 8u) | ((" + fourth + " >> 8u) & 0x00ff00ffu)");
            expression = "((" + fifth + " << 16u) | (" + fifth + " >> 16u))";
            break;
        }
        case Opcode::bit_count: expression = wide ? "uint(popcount(" + value + "))" : "popcount(" + value + ")"; break;
        case Opcode::find_lsb:
            expression = "(" + value + " == " + zero + " ? " + all_bits + " : uint(ctz(" + value + ")))";
            break;
        case Opcode::find_msb:
            if (signed_domain) {
                // The signed most-significant bit excludes the sign: a
                // negative value reports its highest zero bit, and zero or
                // negative one have no such bit.
                const auto typed = bits_constant(c, id, 1, wide,
                    "as_type<" + std::string(scalar(raw_scalar == ScalarType::u64 ? ScalarType::s64 : ScalarType::s32)) + ">(" + value + ")");
                const auto inverted = wide ? "~as_type<ulong>(" + typed + ")" : "~as_type<uint>(" + typed + ")";
                const auto direct = wide ? "as_type<ulong>(" + typed + ")" : "as_type<uint>(" + typed + ")";
                const auto typed_zero = wide ? "0l" : "0";
                expression = "((" + typed + " == " + typed_zero + " || " + typed + " == -1) ? " + all_bits +
                    " : (" + typed + " < 0 ? " + width + " - uint(clz(" + inverted + ")) : " + width + " - uint(clz(" + direct + "))))";
            } else {
                expression = "(" + value + " == " + zero + " ? " + all_bits + " : " + width + " - uint(clz(" + value + ")))";
            }
            break;
        default: fail("Unknown word query opcode");
        }
        take(assign_bits(c, op, 0, expression));
    });
}

Status word64_pack(MslContext& c, OperationId id, bool pack) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.numeric.domain == Domain::bits, "Word64 packing is a raw-bit operation",
                DiagnosticCode::unsupported_modifier);
        if (pack) {
            require(op.inputs.size() == 2 && op.outputs.size() == 1, "Word64 packing takes low/high words and one result");
            const auto low = operand_at(c, op, 0, ScalarType::u32);
            const auto high = operand_at(c, op, 1, ScalarType::u32);
            take(c.assign(output_at(op, 0), "(ulong(" + high + ") << 32u | ulong(" + low + "))", ScalarType::u64));
        } else {
            require(op.inputs.size() == 1 && op.outputs.size() == 2, "Word64 unpacking takes one input and low/high results");
            const auto value = bits_constant(c, id, 0, true, operand_at(c, op, 0, ScalarType::u64));
            take(assign_bits(c, op, 0, "uint(" + value + " & 0xfffffffful)"));
            take(assign_bits(c, op, 1, "uint(" + value + " >> 32u)"));
        }
    });
}

Status integer_compare(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        const auto& comparison = comparison_info(op);
        require(!comparison.unordered, "An unordered flag without floating inputs has no meaning",
                DiagnosticCode::unsupported_modifier);
        const auto type = interpreted_scalar(op);
        const auto left = operand_at(c, op, 0, type);
        const auto right = operand_at(c, op, 1, type);
        const char* symbol = "<invalid>";
        switch (comparison.relation) {
        case Compare::never: take(c.assign(output_at(op, 0), "false", ScalarType::boolean)); return;
        case Compare::always: take(c.assign(output_at(op, 0), "true", ScalarType::boolean)); return;
        case Compare::less: symbol = "<"; break;
        case Compare::equal: symbol = "=="; break;
        case Compare::less_equal: symbol = "<="; break;
        case Compare::greater: symbol = ">"; break;
        case Compare::not_equal: symbol = "!="; break;
        case Compare::greater_equal: symbol = ">="; break;
        }
        take(c.assign(output_at(op, 0), "(" + left + " " + symbol + " " + right + ")", ScalarType::boolean));
    });
}

} // namespace

// Mixed-domain opcode generators: floating modes delegate to numeric.cpp and
// integer modes stay here.
Status generate_copy(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Copy takes one input and one result");
        const auto type = storage_scalar(c.program(), op);
        take(c.assign(output_at(op, 0), operand_at(c, op, 0, type), type));
    });
}

Status generate_absolute(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::sign(c, id, true) : integer_sign(c, id, true);
}

Status generate_negate(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::sign(c, id, false) : integer_sign(c, id, false);
}

Status generate_add(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::arithmetic(c, id, Opcode::add)
                                                 : integer_arithmetic(c, id, Opcode::add);
}

Status generate_subtract(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::arithmetic(c, id, Opcode::subtract)
                                                 : integer_arithmetic(c, id, Opcode::subtract);
}

Status generate_multiply(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::arithmetic(c, id, Opcode::multiply)
                                                 : integer_arithmetic(c, id, Opcode::multiply);
}

Status generate_divide(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::arithmetic(c, id, Opcode::divide)
                                                 : integer_arithmetic(c, id, Opcode::divide);
}

Status generate_remainder(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::arithmetic(c, id, Opcode::remainder)
                                                 : integer_arithmetic(c, id, Opcode::remainder);
}

Status generate_multiply_high(MslContext& c, OperationId id) { return multiply_high(c, id); }

Status generate_add_carry(MslContext& c, OperationId id) { return carry_operation(c, id, true); }

Status generate_subtract_borrow(MslContext& c, OperationId id) { return carry_operation(c, id, false); }

Status generate_minimum(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::extrema(c, id, Opcode::minimum)
                                                 : integer_extremum(c, id, Opcode::minimum);
}

Status generate_maximum(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::extrema(c, id, Opcode::maximum)
                                                 : integer_extremum(c, id, Opcode::maximum);
}

Status generate_clamp(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::clamp(c, id) : integer_clamp(c, id);
}

Status generate_fused_multiply_add(MslContext& c, OperationId id) { return numeric::fused_multiply_add(c, id); }

Status generate_reciprocal(MslContext& c, OperationId id) { return numeric::reciprocal(c, id); }

Status generate_reciprocal_sqrt(MslContext& c, OperationId id) { return numeric::reciprocal_sqrt(c, id); }

Status generate_square_root(MslContext& c, OperationId id) { return numeric::square_root(c, id); }

Status generate_sine(MslContext& c, OperationId id) { return numeric::transcendental(c, id, Opcode::sine); }

Status generate_cosine(MslContext& c, OperationId id) { return numeric::transcendental(c, id, Opcode::cosine); }

Status generate_exp2(MslContext& c, OperationId id) { return numeric::transcendental(c, id, Opcode::exp2); }

Status generate_log2(MslContext& c, OperationId id) { return numeric::transcendental(c, id, Opcode::log2); }

Status generate_range_reduce_sincos(MslContext& c, OperationId id) { return numeric::range_reduce(c, id); }

Status generate_range_reduce_exp2(MslContext& c, OperationId id) { return numeric::range_reduce(c, id); }

Status generate_ceiling(MslContext& c, OperationId id) { return numeric::integral_round(c, id, Opcode::ceiling); }

Status generate_floor(MslContext& c, OperationId id) { return numeric::integral_round(c, id, Opcode::floor); }

Status generate_round(MslContext& c, OperationId id) { return numeric::integral_round(c, id, Opcode::round); }

Status generate_truncate(MslContext& c, OperationId id) { return numeric::integral_round(c, id, Opcode::truncate); }

Status generate_is_nan(MslContext& c, OperationId id) { return numeric::is_nan(c, id); }

Status generate_compare(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    return op.numeric.domain == Domain::floating ? numeric::compare(c, id) : integer_compare(c, id);
}

Status generate_select(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 3 && op.outputs.size() == 1, "Select takes a condition and two alternatives");
        const auto type = storage_scalar(c.program(), op);
        const auto condition = operand_at(c, op, 0, ScalarType::boolean);
        const auto when_true = operand_at(c, op, 1, type);
        const auto when_false = operand_at(c, op, 2, type);
        take(c.assign(output_at(op, 0), "(" + condition + " ? " + when_true + " : " + when_false + ")", type));
    });
}

Status generate_convert(MslContext& c, OperationId id) {
    const auto& op = c.operation(id);
    const auto& conversion = conversion_info(op);
    const bool floating_side = scalar_domain(conversion.source) == Domain::floating ||
                               scalar_domain(conversion.destination) == Domain::floating;
    return floating_side ? numeric::conversion(c, id) : integer_conversion(c, id);
}

Status generate_bitcast(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        const auto& conversion = conversion_info(op);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Bitcast takes one input and one result");
        require(op.numeric.domain == Domain::bits, "A bitcast is a raw reinterpretation",
                DiagnosticCode::unsupported_modifier);
        require(scalar_bits(conversion.source) == scalar_bits(conversion.destination),
                "A bitcast preserves its width");
        require(conversion.source != ScalarType::f64 && conversion.destination != ScalarType::f64,
                "Metal has no native FP64 representation", DiagnosticCode::unsupported_capability);
        const auto source = operand_at(c, op, 0, conversion.source);
        take(c.assign(output_at(op, 0),
                     "as_type<" + std::string(scalar(conversion.destination)) + ">(" + source + ")",
                     conversion.destination));
    });
}

Status generate_predicate_to_mask(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Predicate-to-mask takes one input and one result");
        const auto predicate = operand_at(c, op, 0, ScalarType::boolean);
        take(assign_bits(c, op, 0, "(" + predicate + " ? 0xffffffffu : 0u)"));
    });
}

Status generate_mask_to_predicate(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Mask-to-predicate takes one input and one result");
        const auto mask = operand_at(c, op, 0, ScalarType::u32);
        take(c.assign(output_at(op, 0), "(" + mask + " != 0u)", ScalarType::boolean));
    });
}

Status generate_bit_and(MslContext& c, OperationId id) { return bit_binary(c, id, Opcode::bit_and); }

Status generate_bit_or(MslContext& c, OperationId id) { return bit_binary(c, id, Opcode::bit_or); }

Status generate_bit_xor(MslContext& c, OperationId id) { return bit_binary(c, id, Opcode::bit_xor); }

Status generate_bit_not(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 1 && op.outputs.size() == 1, "Bitwise negation takes one input and one result");
        take(assign_bits(c, op, 0, "(~" + operand_at(c, op, 0, wide_mode(op) ? ScalarType::u64 : ScalarType::u32) + ")"));
    });
}

Status generate_logical_and(MslContext& c, OperationId id) { return predicate_logic(c, id, Opcode::logical_and); }

Status generate_logical_or(MslContext& c, OperationId id) { return predicate_logic(c, id, Opcode::logical_or); }

Status generate_logical_xor(MslContext& c, OperationId id) { return predicate_logic(c, id, Opcode::logical_xor); }

Status generate_logical_not(MslContext& c, OperationId id) { return predicate_logic(c, id, Opcode::logical_not); }

Status generate_shift_left(MslContext& c, OperationId id) { return shift(c, id, Opcode::shift_left); }

Status generate_shift_right(MslContext& c, OperationId id) { return shift(c, id, Opcode::shift_right); }

Status generate_funnel_shift_left(MslContext& c, OperationId id) { return funnel_shift(c, id, true); }

Status generate_funnel_shift_right(MslContext& c, OperationId id) { return funnel_shift(c, id, false); }

Status generate_bit_extract(MslContext& c, OperationId id) { return bit_field(c, id, Opcode::bit_extract); }

Status generate_bit_insert(MslContext& c, OperationId id) { return bit_field(c, id, Opcode::bit_insert); }

Status generate_bit_reverse(MslContext& c, OperationId id) { return word_query(c, id, Opcode::bit_reverse); }

Status generate_bit_count(MslContext& c, OperationId id) { return word_query(c, id, Opcode::bit_count); }

Status generate_find_lsb(MslContext& c, OperationId id) { return word_query(c, id, Opcode::find_lsb); }

Status generate_find_msb(MslContext& c, OperationId id) { return word_query(c, id, Opcode::find_msb); }

Status generate_pack_half2(MslContext& c, OperationId id) { return numeric::pack_half2(c, id); }

Status generate_unpack_half2(MslContext& c, OperationId id) { return numeric::unpack_half2(c, id); }

Status generate_pack_word64(MslContext& c, OperationId id) { return word64_pack(c, id, true); }

Status generate_unpack_word64(MslContext& c, OperationId id) { return word64_pack(c, id, false); }

// Warp shuffle follows the Maxwell clamp/segment contract. The entry's
// subgroup-lane builtin parameter and one unconditional simd_shuffle replace
// the generated helper:
// the inline expressions reproduce the segment base, the clamp ceiling, the
// per-mode source lane and the in-range comparison, and the transfer runs on
// every lane so the group-wide execution rule of the native function holds.
// Apple SIMD-groups are 32 lanes wide like a Maxwell warp; the emitted index
// arithmetic keeps that width and a wider host group would violate it.
Status generate_shuffle(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id);
        require(op.inputs.size() == 3 && op.outputs.size() == 2,
                "Shuffle takes value, lane and mask inputs and returns the word and its validity");
        require(std::holds_alternative<SubgroupInfo>(op.info), "Shuffle carries no subgroup metadata");
        const auto& subgroup = std::get<SubgroupInfo>(op.info);
        require(subgroup.width == 32, "Metal SIMD-groups are 32 lanes wide");
        require(c.resources().profile.supports_simd_shuffle,
                "Metal profile does not expose SIMD-group shuffle", DiagnosticCode::unsupported_capability);
        // The lane id rides the entry signature parameter that the resource
        // plan declares from the shuffle's recorded lane read; MSL has no
        // lane-query function in this dialect.
        const IoUse* lane_use = nullptr;
        for (const auto& io : c.resources().io) {
            if (io.semantic != IoSemantic::subgroup_lane_id || io.access != Access::read) continue;
            require(!lane_use, "Ambiguous subgroup lane requirement");
            lane_use = &io;
        }
        require(lane_use && lane_use->type == IoType{ScalarType::u32, 1},
                "Shuffle requires the declared subgroup lane builtin", DiagnosticCode::unsupported_capability);
        const auto lane = "(" + io_name(IoSemantic::subgroup_lane_id, 0) + " & 31u)";
        const auto value = operand_at(c, op, 0, ScalarType::u32);
        const auto lane_index = operand_at(c, op, 1, ScalarType::u32);
        const auto bounds = operand_at(c, op, 2, ScalarType::u32);
        // Snapshot every operand: the lowering references each one more than
        // once and a resource-backed operand would otherwise reload per use.
        const auto mask = bits_constant(c, id, 1, false, bounds);
        const auto data = bits_constant(c, id, 2, false, value);
        const auto index = bits_constant(c, id, 3, false, lane_index);
        const auto segment = bits_constant(c, id, 4, false, "(" + mask + " >> 8) & 0x1fu");
        const auto segment_base = bits_constant(c, id, 5, false, "(" + lane + " & " + segment + ")");
        std::string source_lane;
        switch (subgroup.shuffle) {
        case ShuffleKind::index:
            source_lane = "((" + index + " & ~" + segment + ") | " + segment_base + ")";
            break;
        case ShuffleKind::up: source_lane = "(" + lane + " - " + index + ")"; break;
        case ShuffleKind::down: source_lane = "(" + lane + " + " + index + ")"; break;
        case ShuffleKind::butterfly: source_lane = "(" + lane + " ^ " + index + ")"; break;
        }
        std::string in_range;
        if (subgroup.shuffle == ShuffleKind::up) {
            // Compare the wrapped source lane as a signed word, so an underflow
            // below the segment base is out of range.
            in_range = "(as_type<int>(" + source_lane + ") >= as_type<int>(" + segment_base + "))";
        } else {
            const auto clamp = bits_constant(c, id, 6, false, "(" + mask + " & 0x1fu)");
            const auto segment_top =
                bits_constant(c, id, 7, false, "((" + clamp + " & ~" + segment + ") | " + segment_base + ")");
            in_range = "(" + source_lane + " <= " + segment_top + ")";
        }
        const auto picked = bits_constant(c, id, 8, false, "simd_shuffle(" + data + ", " + source_lane + ")");
        take(assign_bits(c, op, 0, "(" + in_range + " ? " + picked + " : " + data + ")"));
        take(c.assign(output_at(op, 1), in_range, ScalarType::boolean));
    });
}
} // namespace armigo::shader::pipeline
