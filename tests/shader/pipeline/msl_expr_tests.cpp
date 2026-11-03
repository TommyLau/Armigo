// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic, independently authored programs: the fixtures exercise the pure
// MSL opcode generators' emitted text and their rejection diagnostics. The
// Each fixture names the emitter contract under test.
#include "pipeline/phases.h"

#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

int failures{};
std::filesystem::path fixture_directory;
bool compile_metal{};

void check(bool value, const std::string& message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
Operand predicate(bool value) { return LiteralBits{value ? 1u : 0u, ValueType::predicate}; }
NumericMode floating(Rounding rounding = Rounding::nearest_even) {
    return {Domain::floating, ScalarType::f32, rounding};
}
NumericMode integer_domain(bool signed_domain) {
    return {signed_domain ? Domain::signed_integer : Domain::unsigned_integer,
            signed_domain ? ScalarType::s32 : ScalarType::u32};
}

// A single-block fragment program with no IO: every pure generator is
// observable through the module text and through emission diagnostics.
struct Fixture {
    StructuredProgram p;
    ResourcePlan plan;
    EmitOptions options;
    Fixture() {
        p.ir.stage = Stage::fragment;
        p.ir.entry = FunctionId{0};
        p.ir.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
        p.ir.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
        p.regions = {BlockRegion{BlockId{0}}};
        p.functions = {{FunctionId{0}, RegionId{0}}};
        p.profile = {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        plan.stage = Stage::fragment;
        plan.profile = p.profile;
    }
    ValueId value(ValueType type = ValueType::word32) {
        ValueId id{static_cast<std::uint32_t>(p.ir.values.size())};
        p.ir.values.push_back({type, ValueKind::temporary});
        return id;
    }
    ValueId word_value() { return value(ValueType::word32); }
    // A defined word input: register-form IR forbids undefined reads, so
    // fixture inputs are copy definitions of a literal.
    Operand defined_word(std::uint32_t bits = 0) {
        const auto id = word_value();
        Operation copy;
        copy.opcode = Opcode::copy;
        copy.inputs.push_back(word(bits));
        copy.outputs.push_back(id);
        operation(std::move(copy));
        return Operand{id};
    }
    Operand defined_wide() {
        const auto id = value(ValueType::word64);
        Operation copy;
        copy.opcode = Opcode::copy;
        copy.inputs.push_back(LiteralBits{0, ValueType::word64});
        copy.outputs.push_back(id);
        operation(std::move(copy));
        return Operand{id};
    }
    OperationId operation(Operation op) {
        op.effects = required_effects(op);
        op.source = {40 + static_cast<Pc>(p.ir.operations.size() * 8), InstructionName::mov};
        OperationId id{static_cast<std::uint32_t>(p.ir.operations.size())};
        p.ir.operations.push_back(std::move(op));
        p.ir.blocks[0].operations.push_back(id);
        return id;
    }
    // One pure operation: inputs are already-created values or literals, and
    // outputs are fresh values of the declared storage.
    void pure(Opcode opcode, std::vector<Operand> inputs, unsigned outputs, NumericMode mode = {},
              OperationInfo info = {}, ValueType output_type = ValueType::word32) {
        Operation op;
        op.opcode = opcode;
        op.numeric = mode;
        op.inputs = std::move(inputs);
        for (unsigned i = 0; i < outputs; ++i) op.outputs.push_back(value(output_type));
        op.info = std::move(info);
        operation(std::move(op));
    }
    Result<MslArtifact> emit() { return emit_msl(p, plan, options); }
};

std::string quote(const std::string& value) {
    std::string result = "'";
    for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
    return result + "'";
}

std::optional<std::string> accepted(Fixture f, const std::string& label) {
    auto result = f.emit();
    if (const auto* d = std::get_if<Diagnostic>(&result)) {
        check(false, label + ": " + d->message);
        return std::nullopt;
    }
    auto artifact = std::get<MslArtifact>(result);
    check(artifact.reflection.dependencies == f.plan.dependencies, label + " reflection");
    if (!fixture_directory.empty()) {
        const auto path = fixture_directory / (label + ".metal");
        std::ofstream(path) << artifact.source;
        if (compile_metal) {
            const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
                                 quote((fixture_directory / "module-cache").string()) + " -c " + quote(path.string()) +
                                 " -o " + quote((fixture_directory / (label + ".air")).string());
            check(std::system(command.c_str()) == 0, label + " Metal compilation");
        }
    }
    return artifact.source;
}

void rejected(Fixture f, const std::string& label, std::optional<DiagnosticCode> code = {}) {
    auto result = f.emit();
    const auto* d = std::get_if<Diagnostic>(&result);
    check(d && d->phase == Phase::emit_msl && (!code || d->code == *code), label);
    if (d && code && d->code != *code) std::cerr << "  got: " << d->message << '\n';
}

void contains(const std::optional<std::string>& source, const std::string& needle, const std::string& label) {
    check(source && source->find(needle) != std::string::npos, label + " emits " + needle);
}

void lacks(const std::optional<std::string>& source, const std::string& needle, const std::string& label) {
    check(source && source->find(needle) == std::string::npos, label + " omits " + needle);
}

// Bit temporaries are named after the operation emitting them.
std::string const_name(Fixture& f, unsigned slot) {
    return "armigo_w" + std::to_string(f.p.ir.operations.size()) + "_" + std::to_string(slot);
}

NumericMode flushed_float() {
    auto mode = floating();
    mode.flush_inputs = true;
    return mode;
}

Fixture unary_fixture(Opcode opcode, NumericMode mode, OperationInfo info = {}) {
    Fixture f;
    f.pure(opcode, {f.defined_word()}, 1, mode, std::move(info));
    return f;
}

Fixture binary_fixture(Opcode opcode, NumericMode mode, OperationInfo info = {}) {
    Fixture f;
    f.pure(opcode, {f.defined_word(), f.defined_word()}, 1, mode, std::move(info));
    return f;
}

// Output storage must match the opcode schema, so pure() takes the type.
// (pure() gains an out_type parameter in Fixture below.)

void opcode_text_tests() {
    { // copy moves the raw word through the storage interpretation.
        Fixture f;
        f.pure(Opcode::copy, {word(7)}, 1);
        contains(accepted(f, "copy"), "value0 = 7u;", "copy");
    }
    { // floating sign operations flip mask bits without arithmetic.
        contains(accepted(unary_fixture(Opcode::absolute, floating()), "absolute_f32"), "& 0x7fffffffu",
                 "absolute f32");
        contains(accepted(unary_fixture(Opcode::negate, floating()), "negate_f32"), "^ 0x80000000u", "negate f32");
    }
    { // integer sign operations negate on the modular word.
        contains(accepted(unary_fixture(Opcode::negate, integer_domain(true)), "negate_s32"), "(0u - value0)",
                 "negate s32");
        contains(accepted(unary_fixture(Opcode::absolute, integer_domain(true)), "absolute_s32"),
                 "(value0 < 0x80000000u ? value0 : 0u - value0)", "absolute s32");
    }
    { // Native sums, differences and quotients implement the floating operators.
        contains(accepted(binary_fixture(Opcode::add, floating()), "add_f32"),
                 "as_type<uint>(as_type<float>(value0) + as_type<float>(value1))", "add f32");
        contains(accepted(binary_fixture(Opcode::subtract, floating()), "subtract_f32"),
                 "as_type<uint>(as_type<float>(value0) - as_type<float>(value1))", "subtract f32");
        contains(accepted(binary_fixture(Opcode::divide, floating()), "divide_f32"),
                 "as_type<uint>(as_type<float>(value0) / as_type<float>(value1))", "divide f32");
    }
    { // the GPU flushes arithmetic inputs and results itself, so a flushed
      // add is the bare native operator with no word-level selects.
        Fixture f;
        auto mode = floating();
        mode.flush_inputs = mode.flush_output = true;
        const auto a = f.defined_word();
        const auto b = f.defined_word();
        f.pure(Opcode::add, {a, b}, 1, mode);
        const auto source = accepted(f, "add_f32_ftz");
        contains(source, "as_type<uint>(as_type<float>(value0) + as_type<float>(value1))",
                 "flushed add is the native operator");
        check(source && source->find("0x7f800000u) == 0u") == std::string::npos,
              "flushed add emits no explicit flush");
    }
    { // an operation on literals alone folds at compile time without the
      // GPU's flush, so its flush stays explicit there.
        Fixture f;
        auto mode = floating();
        mode.flush_inputs = mode.flush_output = true;
        f.pure(Opcode::add, {word(0x00400000u), word(0x3f800000u)}, 1, mode);
        contains(accepted(f, "add_literals_ftz"), "& 0x7f800000u) == 0u", "literal flushed add keeps its flush");
    }
    { // FMUL.RZ on literals alone, emitted as the native product, keeps the
      // explicit flush as well.
        Fixture f;
        auto mode = floating(Rounding::toward_zero);
        mode.flush_inputs = mode.flush_output = true;
        f.pure(Opcode::multiply, {word(0x00400000u), word(0x40800000u)}, 1, mode);
        contains(accepted(f, "multiply_rz_literals_ftz"), "(4194304u & 0x7f800000u) == 0u",
                 "literal FMUL.RZ flushes its subnormal input");
    }
    { // the product is the native operator for the nearest-even mode.
        contains(accepted(binary_fixture(Opcode::multiply, floating()), "multiply_rne"),
                 "as_type<uint>(as_type<float>(value0) * as_type<float>(value1))", "multiply native");
    }
    { // integer arithmetic stays modular at the raw word.
        contains(accepted(binary_fixture(Opcode::add, integer_domain(true)), "add_s32"), "(value0 + value1)",
                 "add s32");
        contains(accepted(binary_fixture(Opcode::multiply, integer_domain(false)), "multiply_u32"),
                 "(value0 * value1)", "multiply u32");
        contains(accepted(binary_fixture(Opcode::divide, integer_domain(false)), "divide_u32"), "(value0 / value1)",
                 "divide u32");
        contains(accepted(binary_fixture(Opcode::divide, integer_domain(true)), "divide_s32"), "== -1",
                 "signed divide guard");
        contains(accepted(binary_fixture(Opcode::remainder, integer_domain(true)), "remainder_s32"),
                 "% as_type<int>", "remainder s32");
    }
    { // multiply-high forms the 64-bit product explicitly.
        contains(accepted(binary_fixture(Opcode::multiply_high, integer_domain(false)), "multiply_high_u32"),
                 "(ulong(value0) * ulong(value1)) >> 32u", "multiply high u32");
        contains(accepted(binary_fixture(Opcode::multiply_high, integer_domain(true)), "multiply_high_s32"),
                 "long(as_type<int>(value0))", "multiply high s32");
    }
    { // carry and borrow widen to a 64-bit accumulation with a predicate flag.
        Fixture f;
        f.pure(Opcode::add_carry, {f.defined_word(), f.defined_word(), predicate(true)}, 2, integer_domain(false));
        f.p.ir.values[std::get<ValueId>(f.p.ir.operations[2].outputs[1]).value].type = ValueType::predicate;
        const auto source = accepted(f, "add_carry");
        contains(source, "1ul : 0ul", "carry incoming");
        contains(source, "> 0xfffffffful", "carry out");
        Fixture b;
        const auto x = b.defined_word();
        const auto y = b.defined_word();
        b.pure(Opcode::subtract_borrow, {x, y, predicate(false)}, 2, integer_domain(false));
        b.p.ir.values[std::get<ValueId>(b.p.ir.operations[2].outputs[1]).value].type = ValueType::predicate;
        contains(accepted(b, "subtract_borrow"), "ulong(value0) - ulong(value1)", "borrow");
    }
    { // floating extrema are the precise native min/max over interpreted
      // values alone: no canonical-NaN or signed-zero select and no operand
      // flush around the core (deliberate departures documented in the
      // emitter); every selection rounding/output-flush combination stays
      // accepted.
        contains(accepted(binary_fixture(Opcode::minimum, floating()), "minimum_f32"),
                 "as_type<uint>(precise::fmin(as_type<float>(value0), as_type<float>(value1)))",
                 "minimum f32");
        contains(accepted(binary_fixture(Opcode::maximum, floating()), "maximum_f32"),
                 "as_type<uint>(precise::fmax(as_type<float>(value0), as_type<float>(value1)))",
                 "maximum f32");
        lacks(accepted(binary_fixture(Opcode::minimum, floating()), "minimum_f32"),
              "? 0x7fc00000u :", "minimum has no canonical-NaN select");
        lacks(accepted(binary_fixture(Opcode::maximum, floating()), "maximum_f32"),
              "== 0x80000000u) ? 0u :", "maximum has no zero-tie select");
        {
            Fixture f;
            const auto a = f.defined_word();
            const auto b = f.defined_word();
            f.pure(Opcode::minimum, {a, b}, 1, flushed_float());
            const auto source = accepted(f, "minimum_flushed");
            contains(source,
                     "as_type<uint>(precise::fmin(as_type<float>(value0), as_type<float>(value1)))",
                     "minimum leaves its inputs to the device flush");
            lacks(source, "0x7f800000u) == 0u", "flushed minimum on values emits no explicit flush");
        }
        { // on literals alone the compiler folds the extremum without the
          // device, so the guest flush stays explicit there.
            Fixture f;
            f.pure(Opcode::minimum, {word(0x00400000u), word(0x3f800000u)}, 1, flushed_float());
            contains(accepted(f, "minimum_literals_ftz"), "(4194304u & 0x7f800000u) == 0u",
                     "literal flushed minimum flushes its subnormal input");
        }
        for (const auto rounding : {Rounding::nearest_even, Rounding::toward_zero,
                                     Rounding::toward_negative, Rounding::toward_positive}) {
            for (const bool flush_output : {false, true}) {
                auto mode = floating();
                mode.rounding = rounding;
                mode.flush_output = flush_output;
                mode.precise = true;
                const auto suffix = std::to_string(static_cast<unsigned>(rounding)) +
                                    (flush_output ? "_ftz" : "_input_ftz");
                contains(accepted(binary_fixture(Opcode::minimum, mode), "minimum_" + suffix),
                         "as_type<uint>(precise::fmin(as_type<float>(value0), as_type<float>(value1)))",
                         "minimum preserves selection modes");
                contains(accepted(binary_fixture(Opcode::maximum, mode), "maximum_" + suffix),
                         "as_type<uint>(precise::fmax(as_type<float>(value0), as_type<float>(value1)))",
                         "maximum preserves selection modes");
                Fixture clamp;
                clamp.pure(Opcode::clamp,
                           {clamp.defined_word(), clamp.defined_word(), clamp.defined_word()}, 1, mode);
                contains(accepted(clamp, "clamp_" + suffix),
                         "as_type<uint>(precise::fmin(precise::fmax(as_type<float>(value0), as_type<float>(value1)), as_type<float>(value2)))",
                         "clamp preserves selection modes");
            }
        }
    }
    { // integer extrema use the typed native functions.
        contains(accepted(binary_fixture(Opcode::minimum, integer_domain(true)), "minimum_s32"),
                 "min(as_type<int>(value0), as_type<int>(value1))", "minimum s32");
    }
    { // Integer clamps compose the required extremum order.
        Fixture c;
        const auto x = c.defined_word();
        const auto lo = c.defined_word();
        const auto hi = c.defined_word();
        const auto value_name = const_name(c, 0);
        const auto low_name = const_name(c, 1);
        const auto high_name = const_name(c, 2);
        c.pure(Opcode::clamp, {x, lo, hi}, 1, integer_domain(false));
        contains(accepted(c, "clamp_u32"), "min(max(" + value_name + ", " + low_name + "), " + high_name + ")",
                 "clamp u32");
        Fixture s;
        const auto signed_input = s.defined_word(1);
        const auto signed_name = const_name(s, 0);
        const auto signed_low = const_name(s, 1);
        s.pure(Opcode::clamp, {signed_input, word(0xffff8000u), word(0x7fffu)}, 1, integer_domain(true));
        const auto signed_source = accepted(s, "clamp_s32");
        contains(signed_source, "const int " + signed_name + " = as_type<int>(value0);", "signed clamp input");
        contains(signed_source, "const int " + signed_low + " = as_type<int>(4294934528u);", "signed clamp bound");
    }
    { // The floating clamp uses the precise min/max composition on the
      // interpreted operands without an operand flush.
        Fixture g;
        const auto x = g.defined_word();
        const auto lo = g.defined_word();
        const auto hi = g.defined_word();
        g.pure(Opcode::clamp, {x, lo, hi}, 1, flushed_float());
        const auto source = accepted(g, "clamp_f32");
        contains(source,
                 "as_type<uint>(precise::fmin(precise::fmax(as_type<float>(value0), as_type<float>(value1)), as_type<float>(value2)))",
                 "clamp f32");
        lacks(source, "0x7f800000u) == 0u", "flushed clamp on values emits no explicit flush");
    }
    { // a clamp on literals alone keeps the explicit flush.
        Fixture g;
        g.pure(Opcode::clamp, {word(0x00400000u), word(0u), word(0x3f800000u)}, 1, flushed_float());
        contains(accepted(g, "clamp_literals_ftz"), "(4194304u & 0x7f800000u) == 0u",
                 "literal flushed clamp flushes its subnormal input");
    }
    { // the fused product is the native single-rounding fma.
        Fixture f;
        auto mode = floating();
        mode.flush_inputs = mode.flush_output = true;
        const auto a = f.defined_word();
        const auto b = f.defined_word();
        const auto c = f.defined_word();
        f.pure(Opcode::fused_multiply_add, {a, b, c}, 1, mode);
        contains(accepted(f, "fma"),
                 "as_type<uint>(fma(as_type<float>(value0), as_type<float>(value1), as_type<float>(value2)))",
                 "fma native");
        Fixture unsaturated_flush;
        unsaturated_flush.pure(Opcode::fused_multiply_add,
                               {unsaturated_flush.defined_word(), unsaturated_flush.defined_word(),
                                unsaturated_flush.defined_word()}, 1, floating());
        contains(accepted(std::move(unsaturated_flush), "fma_unflushed"),
                 "as_type<uint>(fma(as_type<float>(value0), as_type<float>(value1), as_type<float>(value2)))",
                 "fma needs no flush flags");
        Fixture saturated;
        auto saturating = floating();
        saturating.saturate = true;
        saturated.pure(Opcode::multiply, {saturated.defined_word(), saturated.defined_word()}, 1, saturating);
        contains(accepted(std::move(saturated), "multiply_saturate"),
                 "as_type<uint>(precise::clamp(as_type<float>(as_type<uint>(as_type<float>(value0) * "
                 "as_type<float>(value1))), 0.0f, 1.0f))", "saturation is the native unit clamp");
    }
    { // The multifunction unit maps to native MSL operations.
        contains(accepted(unary_fixture(Opcode::reciprocal, floating()), "reciprocal"),
                 "as_type<uint>(1.0f / as_type<float>(value0))", "reciprocal native divide");
        contains(accepted(unary_fixture(Opcode::square_root, floating()), "square_root"),
                 "sqrt(as_type<float>(value0))", "square root");
        contains(accepted(unary_fixture(Opcode::reciprocal_sqrt, floating()), "reciprocal_sqrt"),
                 "as_type<uint>(rsqrt(as_type<float>(value0)))", "reciprocal square root native");
    }
    { // elementary functions use the native MSL functions.
        for (const auto& [opcode, name] : {std::pair{Opcode::sine, std::string{"sin"}},
                                           std::pair{Opcode::cosine, std::string{"cos"}},
                                           std::pair{Opcode::exp2, std::string{"exp2"}},
                                           std::pair{Opcode::log2, std::string{"log2"}}}) {
            contains(accepted(unary_fixture(opcode, floating()), "function_" + name),
                     name + "(as_type<float>(value0))", "function " + name);
        }
    }
    { // range reduction is a bit-exact pass-through.
        contains(accepted(unary_fixture(Opcode::range_reduce_sincos, floating()), "range_reduce_sincos"),
                 "value1 = value0;", "range reduce identity");
    }
    { // integral rounding maps to the roundToIntegral functions.
        for (const auto& [opcode, name] : {std::pair{Opcode::ceiling, std::string{"ceil"}},
                                           std::pair{Opcode::floor, std::string{"floor"}},
                                           std::pair{Opcode::round, std::string{"rint"}},
                                           std::pair{Opcode::truncate, std::string{"trunc"}}}) {
            contains(accepted(unary_fixture(opcode, floating()), "round_" + name),
                     name + "(as_type<float>(value0))", "rounding " + name);
        }
    }
    { // the NaN test is an integer test on the word, which no math mode folds.
        Fixture n;
        n.pure(Opcode::is_nan, {n.defined_word()}, 1, floating(), {}, ValueType::predicate);
        contains(accepted(n, "is_nan"), "((value0 & 0x7fffffffu) > 0x7f800000u)", "is nan");
    }
    { // floating comparison decomposes into the ordered operator and explicit
      // NaN terms set the unordered flag.
        Fixture f;
        ComparisonInfo unordered;
        unordered.relation = Compare::less;
        unordered.unordered = true;
        f.pure(Opcode::compare, {f.defined_word(), f.defined_word()}, 1, floating(), unordered,
               ValueType::predicate);
        contains(accepted(f, "compare_f32_unordered"),
                 "(as_type<float>(value0) < as_type<float>(value1)) || ((as_type<uint>(as_type<float>(value0)) & 0x7fffffffu) > 0x7f800000u) || ((as_type<uint>(as_type<float>(value1)) & 0x7fffffffu) > 0x7f800000u)",
                 "compare f32 unordered");
        Fixture ordered;
        ComparisonInfo strict;
        strict.relation = Compare::less_equal;
        ordered.pure(Opcode::compare, {ordered.defined_word(), ordered.defined_word()}, 1, floating(), strict,
                     ValueType::predicate);
        contains(accepted(std::move(ordered), "compare_f32_ordered"),
                 "(as_type<float>(value0) <= as_type<float>(value1))", "compare f32 ordered");
        Fixture inequality;
        ComparisonInfo ordered_unequal;
        ordered_unequal.relation = Compare::not_equal;
        inequality.pure(Opcode::compare, {inequality.defined_word(), inequality.defined_word()}, 1,
                        floating(), ordered_unequal, ValueType::predicate);
        contains(accepted(std::move(inequality), "compare_f32_ordered_ne"),
                 "(as_type<float>(value0) != as_type<float>(value1) && !((as_type<uint>(as_type<float>(value0)) & 0x7fffffffu) > 0x7f800000u) && !((as_type<uint>(as_type<float>(value1)) & 0x7fffffffu) > 0x7f800000u))",
                 "ordered NE excludes NaN operands");
        Fixture unordered_inequality;
        ComparisonInfo unordered_not_equal;
        unordered_not_equal.relation = Compare::not_equal;
        unordered_not_equal.unordered = true;
        unordered_inequality.pure(Opcode::compare,
                                  {unordered_inequality.defined_word(), unordered_inequality.defined_word()},
                                  1, floating(), unordered_not_equal, ValueType::predicate);
        contains(accepted(std::move(unordered_inequality), "compare_f32_unordered_ne"),
                 "(as_type<float>(value0) != as_type<float>(value1) || ((as_type<uint>(as_type<float>(value0)) & 0x7fffffffu) > 0x7f800000u) || ((as_type<uint>(as_type<float>(value1)) & 0x7fffffffu) > 0x7f800000u))",
                 "unordered NE keeps the NaN terms");
        Fixture constant;
        ComparisonInfo truth;
        truth.relation = Compare::always;
        constant.pure(Opcode::compare, {constant.defined_word(), constant.defined_word()}, 1, floating(), truth,
                      ValueType::predicate);
        contains(accepted(std::move(constant), "compare_f32_true"), "true", "compare always");
        Fixture g;
        ComparisonInfo integer_relation;
        integer_relation.relation = Compare::greater_equal;
        g.pure(Opcode::compare, {g.defined_word(), g.defined_word()}, 1, integer_domain(true), integer_relation,
               ValueType::predicate);
        contains(accepted(g, "compare_s32"), ">= ", "compare s32");
    }
    { // select is a raw conditional move between same-storage alternatives.
        Fixture f;
        f.pure(Opcode::select, {predicate(true), f.defined_word(), f.defined_word()}, 1);
        contains(accepted(f, "select"), "true ? value0 : value1", "select");
    }
    { // conversions use the native casts and the exact integer movements.
        Fixture f;
        auto truncating = floating(Rounding::toward_zero);
        truncating.flush_inputs = true;
        f.pure(Opcode::convert, {f.defined_word()}, 1, truncating,
               ConversionInfo{ScalarType::f32, ScalarType::s32});
        contains(accepted(f, "convert_f32_s32_trunc"), "as_type<uint>(int(as_type<float>(value0)))",
                 "convert f32 to s32 trunc");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(), ConversionInfo{ScalarType::u32, ScalarType::f32});
        contains(accepted(f, "convert_u32_f32"), "as_type<uint>(float(value0))",
                 "convert u32 to f32");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(), ConversionInfo{ScalarType::s32, ScalarType::f32});
        contains(accepted(f, "convert_s32_f32"), "as_type<uint>(float(as_type<int>(value0)))",
                 "convert s32 to f32");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, integer_domain(true),
               ConversionInfo{ScalarType::s16, ScalarType::s32});
        contains(accepted(f, "convert_s16_s32"), "int(as_type<short>(ushort(value0)))", "convert s16 to s32");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, {Domain::unsigned_integer, ScalarType::u16},
               ConversionInfo{ScalarType::s32, ScalarType::u16});
        contains(accepted(f, "convert_s32_u16"), "ushort(as_type<uint>(as_type<int>(value0)) & 0xffffu)",
                 "convert s32 to u16");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(), ConversionInfo{ScalarType::f32, ScalarType::f16});
        contains(accepted(f, "convert_f32_f16"), "half(as_type<float>(value0))", "convert f32 to f16");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, {Domain::floating, ScalarType::f16, Rounding::nearest_even},
               ConversionInfo{ScalarType::f16, ScalarType::f32});
        contains(accepted(f, "convert_f16_f32"), "float(as_type<half>(ushort((value0 & 0xffffu))))",
                 "convert f16 to f32");
    }
    { // bitcasts reinterpret the same width without value conversion.
        Fixture f;
        f.pure(Opcode::bitcast, {f.defined_word()}, 1, {}, ConversionInfo{ScalarType::u32, ScalarType::f32});
        contains(accepted(f, "bitcast_u32_f32"), "as_type<float>(value0)", "bitcast u32 to f32");
    }
    { // predicate and mask conversions follow the all-bits contract.
        Fixture f;
        f.pure(Opcode::predicate_to_mask, {predicate(false)}, 1);
        contains(accepted(f, "predicate_to_mask"), "? 0xffffffffu : 0u", "predicate to mask");
        f = Fixture{};
        f.pure(Opcode::mask_to_predicate, {f.defined_word()}, 1, {}, {}, ValueType::predicate);
        contains(accepted(f, "mask_to_predicate"), "!= 0u", "mask to predicate");
    }
    { // bitwise and logical operations map to their native operators.
        contains(accepted(binary_fixture(Opcode::bit_and, {}), "bit_and"), "value0 & value1", "bit and");
        contains(accepted(binary_fixture(Opcode::bit_or, {}), "bit_or"), "value0 | value1", "bit or");
        contains(accepted(binary_fixture(Opcode::bit_xor, {}), "bit_xor"), "value0 ^ value1", "bit xor");
        contains(accepted(unary_fixture(Opcode::bit_not, {}), "bit_not"), "~value0", "bit not");
        Fixture f;
        f.pure(Opcode::logical_and, {predicate(true), predicate(false)}, 1, {Domain::boolean, ScalarType::boolean},
               {}, ValueType::predicate);
        contains(accepted(f, "logical_and"), "&&", "logical and");
        f = Fixture{};
        f.pure(Opcode::logical_not, {predicate(true)}, 1, {Domain::boolean, ScalarType::boolean}, {},
               ValueType::predicate);
        contains(accepted(f, "logical_not"), "!", "logical not");
    }
    { // shifts mask the count to the scalar width minus one.
        contains(accepted(binary_fixture(Opcode::shift_left, integer_domain(false)), "shift_left"),
                 "<< (value1 & 31u)", "shift left");
        contains(accepted(binary_fixture(Opcode::shift_right, integer_domain(true)), "shift_right_s32"),
                 ">> (value1 & 31u)", "shift right signed");
    }
    { // funnel shifts join the low/high pair into a 64-bit word.
        Fixture f;
        f.pure(Opcode::funnel_shift_left, {f.defined_word(), f.defined_word(), f.defined_word()}, 1, integer_domain(false));
        contains(accepted(f, "funnel_left"), "<< (value2 & 31u)) >> 32u", "funnel left");
        Fixture g;
        g.pure(Opcode::funnel_shift_right, {g.defined_word(), g.defined_word(), g.defined_word()}, 1, integer_domain(false));
        contains(accepted(g, "funnel_right"), ">> (value2 & 31u)", "funnel right");
    }
    { // bit-field operations use the native extraction and insertion.
        Fixture f;
        f.pure(Opcode::bit_extract, {f.defined_word(), f.defined_word(), f.defined_word()}, 1, integer_domain(true));
        contains(accepted(f, "bit_extract_s32"), "as_type<int>((", "extract signed sign extension");
        contains(accepted(f, "bit_extract_s32"), "const uint armigo_w3_2 = min(value2, 32u - armigo_w3_1);",
                 "extract count is bounded without wrapping");
        contains(accepted(f, "bit_extract_s32"), "== 0u ? 0u : (0xffffffffu >> ((32u -", "extract signed mask");
        Fixture g;
        g.pure(Opcode::bit_insert, {g.defined_word(), g.defined_word(), g.defined_word(), g.defined_word()}, 1,
               integer_domain(false));
        contains(accepted(g, "bit_insert"), "value0 & ~", "insert keeps base outside the field");
        contains(accepted(g, "bit_insert"), "const uint armigo_w4_1 = min(value3, 32u - armigo_w4_0);",
                 "insert count is bounded without wrapping");
        contains(accepted(g, "bit_insert"), "(value1 << (armigo_w4_0 & 31u)) &", "insert places data inside the field");
    }
    { // Word queries return -1 (all bits set) for an empty result.
        const auto reversed = accepted(unary_fixture(Opcode::bit_reverse, {}), "bit_reverse");
        contains(reversed, "0x55555555u) << 1u", "bit reverse first swap");
        contains(reversed, "<< 16u) | (armigo_w1_4 >> 16u)", "bit reverse final swap");
        contains(accepted(unary_fixture(Opcode::bit_count, {}), "bit_count"), "popcount(armigo_w1_0)",
                 "bit count");
        contains(accepted(unary_fixture(Opcode::find_lsb, {}), "find_lsb"),
                 "? 0xffffffffu : uint(ctz(armigo_w1_0))", "find lsb");
        contains(accepted(unary_fixture(Opcode::find_msb, integer_domain(true)), "find_msb_s32"),
                 "clz(~as_type<uint>", "find msb signed");
    }
    { // half packing is one native packed conversion; the output flush keeps
      // its explicit boundary, and unpacking reads the half2 lanes.
        Fixture f;
        f.pure(Opcode::pack_half2, {f.defined_word(), f.defined_word()}, 1, floating(),
               ConversionInfo{ScalarType::f32, ScalarType::f16});
        contains(accepted(f, "pack_half2"), "as_type<uint>(half2(as_type<float>(value0), as_type<float>(value1)))",
                 "pack native half2");
        Fixture flushed;
        auto flushed_out = floating();
        flushed_out.flush_output = true;
        flushed.pure(Opcode::pack_half2, {flushed.defined_word(), flushed.defined_word()}, 1, flushed_out,
                     ConversionInfo{ScalarType::f32, ScalarType::f16});
        contains(accepted(std::move(flushed), "pack_half2_flushed"), "0x7c00u) == 0u",
                 "pack output flush stays explicit");
        f = Fixture{};
        const auto source = f.defined_word();
        const auto word_name = const_name(f, 0);
        f.pure(Opcode::unpack_half2, {source}, 2, floating(),
               ConversionInfo{ScalarType::f16, ScalarType::f32});
        const auto unpacked = accepted(f, "unpack_half2");
        contains(unpacked, "float(as_type<half2>(" + word_name + ").x)", "unpack low");
        contains(unpacked, "float(as_type<half2>(" + word_name + ").y)", "unpack high");
        Fixture flushed_in;
        auto flushed_mode = floating();
        flushed_mode.flush_inputs = true;
        flushed_in.pure(Opcode::unpack_half2, {flushed_in.defined_word()}, 2, flushed_mode,
                        ConversionInfo{ScalarType::f16, ScalarType::f32});
        contains(accepted(std::move(flushed_in), "unpack_half2_flushed"), "0x7c00u) == 0u",
                 "unpack input flush stays explicit");
    }
    { // Packed half comparisons (HSET2/HSETP2) compare the unpacked binary32
      // lanes: the back end never sees binary16 arithmetic, and the comparison
      // text is the native ordered operator with explicit NaN terms.
        Fixture f;
        f.pure(Opcode::unpack_half2, {f.defined_word()}, 2, floating(),
               ConversionInfo{ScalarType::f16, ScalarType::f32});
        f.pure(Opcode::unpack_half2, {f.defined_word()}, 2, floating(),
               ConversionInfo{ScalarType::f16, ScalarType::f32});
        // Each statement appends its input copy before the unpack, so the
        // unpacks are operations 1 and 3 with value pairs 1/2 and 4/5.
        const auto lane = [&](std::size_t unpack, std::size_t half) {
            return std::get<ValueId>(f.p.ir.operations[1 + 2 * unpack].outputs[half]);
        };
        ComparisonInfo ordered_less;
        ordered_less.relation = Compare::less;
        f.pure(Opcode::compare, {Operand{lane(0, 0)}, Operand{lane(1, 0)}}, 1, floating(),
               ordered_less, ValueType::predicate);
        ComparisonInfo unordered_less;
        unordered_less.relation = Compare::less;
        unordered_less.unordered = true;
        f.pure(Opcode::compare, {Operand{lane(0, 1)}, Operand{lane(1, 1)}}, 1, floating(),
               unordered_less, ValueType::predicate);
        f.pure(Opcode::logical_and, {Operand{ValueId{6}}, Operand{ValueId{7}}}, 1,
               {Domain::boolean, ScalarType::boolean}, {}, ValueType::predicate);
        const auto source = accepted(f, "half_compare_predicates");
        contains(source, "(as_type<float>(value1) < as_type<float>(value4))",
                 "low lanes compare in binary32");
        contains(source, "(as_type<float>(value2) < as_type<float>(value5)) || ((as_type<uint>(as_type<float>(value2)) & 0x7fffffffu) > 0x7f800000u) || ((as_type<uint>(as_type<float>(value5)) & 0x7fffffffu) > 0x7f800000u)",
                 "high lanes keep the unordered NaN terms");
        contains(source, "(value6 && value7)", "lane predicates combine logically");
        Fixture direct;
        ComparisonInfo half_mode;
        half_mode.relation = Compare::less;
        direct.pure(Opcode::compare, {direct.defined_word(), direct.defined_word()}, 1,
                    {Domain::floating, ScalarType::f16, Rounding::nearest_even}, half_mode,
                    ValueType::predicate);
        rejected(std::move(direct), "binary16 comparison stays rejected",
                 DiagnosticCode::unsupported_capability);
    }
    { // word64 packing joins and splits the raw halves.
        Fixture f;
        f.pure(Opcode::pack_word64, {f.defined_word(), f.defined_word()}, 1, {}, {}, ValueType::word64);
        const auto source = accepted(f, "pack_word64");
        contains(source, "value2 = (ulong(value1) << 32u | ulong(value0));", "pack word64");
        check(f.p.ir.values[2].type == ValueType::word64, "pack result is word64");
        Fixture g;
        const auto wide_input = g.defined_wide();
        const auto word_name = const_name(g, 0);
        g.pure(Opcode::unpack_word64, {wide_input}, 2);
        const auto unpacked = accepted(g, "unpack_word64");
        contains(unpacked, "value1 = uint(" + word_name + " & 0xfffffffful);", "unpack low word");
        contains(unpacked, "value2 = uint(" + word_name + " >> 32u);", "unpack high word");
    }
    { // a multi-output operation stays correct when a result aliases its input.
        Fixture f;
        const auto shared = f.word_value();
        f.p.ir.values[shared.value].kind = ValueKind::mutable_local;
        f.pure(Opcode::copy, {word(0x12345678)}, 1);
        f.p.ir.operations.back().outputs[0] = shared;
        const auto word_name = const_name(f, 0);
        f.pure(Opcode::unpack_half2, {Operand{shared}}, 2, floating(),
               ConversionInfo{ScalarType::f16, ScalarType::f32});
        f.p.ir.operations[1].outputs[0] = shared;
        const auto aliased = accepted(f, "unpack_alias");
        contains(aliased, "const uint " + word_name + " = value0;", "aliased unpack snapshots input");
        contains(aliased, "value0 = as_type<uint>(float(as_type<half2>(" + word_name + ").x));",
                 "aliased unpack writes after the snapshot");
    }
}

// Warp shuffle: one inline lowering per mode with the Maxwell clamp/segment
// arithmetic and a single unconditional simd_shuffle. The lane id rides the
// entry builtin parameter declared from the plan's recorded lane read, so the
// fixtures are compute stages, the only stage whose IO contract admits that
// builtin today.
void shuffle_text_tests() {
    const ComputeConfiguration compute{{32, 1, 1}, 0};
    const EvidenceOrigin origin{OriginKind::explicit_option, "synthetic-compute"};
    auto lane_use = [] {
        IoUse io;
        io.semantic = IoSemantic::subgroup_lane_id;
        io.mask = 1;
        io.type = {ScalarType::u32, 1};
        io.access = Access::read;
        return io;
    };
    auto shuffle_fixture = [&](ShuffleKind kind, std::uint8_t width = 32, bool with_lane = true) {
        Fixture f;
        f.p.ir.stage = Stage::compute;
        f.plan.stage = Stage::compute;
        f.p.profile.supports_simd_shuffle = true;
        f.plan.profile = f.p.profile;
        f.p.requirements.compute = compute;
        f.plan.compute = compute;
        f.p.ir.dependencies = {{FactKey{FactKind::compute_local_size}, FactValue{ComputeLocalSize{{32, 1, 1}}}, origin},
                               {FactKey{FactKind::compute_shared_memory_bytes}, FactValue{std::uint32_t{0}}, origin}};
        f.plan.dependencies = f.p.ir.dependencies;
        if (with_lane) {
            f.p.requirements.io.push_back(lane_use());
            f.plan.io.push_back(lane_use());
        }
        Operation op;
        op.opcode = Opcode::shuffle;
        op.numeric = NumericMode{Domain::bits, ScalarType::u32};
        op.inputs = {f.defined_word(), f.defined_word(), f.defined_word()};
        op.outputs.push_back(f.value(ValueType::word32));
        op.outputs.push_back(f.value(ValueType::predicate));
        op.info = SubgroupInfo{kind, width};
        f.operation(std::move(op));
        return f;
    };
    { // Index mode: segment base, clamp ceiling, absolute source lane.
        const auto text = accepted(shuffle_fixture(ShuffleKind::index), "shuffle_index");
        contains(text, "builtin" + std::to_string(unsigned(IoSemantic::subgroup_lane_id)) + " & 31u",
                 "shuffle entry lane parameter");
        contains(text, ">> 8) & 0x1fu", "shuffle segment mask");
        contains(text, "& 0x1fu", "shuffle clamp");
        contains(text, "simd_shuffle(", "shuffle native transfer");
        contains(text, ") <= ", "shuffle in-range comparison");
    }
    { // Up mode compares the wrapped source lane as a signed word.
        contains(accepted(shuffle_fixture(ShuffleKind::up), "shuffle_up"), "as_type<int>(", "shuffle up signed");
    }
    contains(accepted(shuffle_fixture(ShuffleKind::down), "shuffle_down"), " + ", "shuffle down source lane");
    contains(accepted(shuffle_fixture(ShuffleKind::butterfly), "shuffle_butterfly"), " ^ ", "shuffle xor source lane");
    { // A missing lane read, a wider or narrower group and a non-SIMD profile
        // are explicit rejections.
        rejected(shuffle_fixture(ShuffleKind::index, 32, false), "Shuffle without a lane read",
                 DiagnosticCode::unsupported_capability);
        rejected(shuffle_fixture(ShuffleKind::index, 16), "Non-32-lane shuffle width",
                 DiagnosticCode::invalid_ir);
        Fixture f = shuffle_fixture(ShuffleKind::index);
        f.p.profile.supports_simd_shuffle = false;
        f.plan.profile = f.p.profile;
        rejected(std::move(f), "Profile without SIMD shuffle", DiagnosticCode::unsupported_capability);
    }
}

void modifier_rejection_tests() {
    { // modes without a native equivalent are named, never approximated.
        rejected(binary_fixture(Opcode::add, floating(Rounding::toward_zero)),
                 "Rounding mode without a generator", DiagnosticCode::unsupported_modifier);
        rejected(binary_fixture(Opcode::multiply, floating(Rounding::toward_zero)),
                 "Toward-zero product without both flush boundaries", DiagnosticCode::unsupported_modifier);
        auto truncating_ftz = floating(Rounding::toward_zero);
        truncating_ftz.flush_inputs = truncating_ftz.flush_output = true;
        {
            // FMUL.RZ is the native product (a deliberate departure of at
            // most one ulp, matching the reference driver), not the helper.
            const auto source = accepted(binary_fixture(Opcode::multiply, truncating_ftz), "multiply_rz_ftz");
            contains(source, ") * as_type<float>(", "FMUL.RZ emits the native product");
            check(source && source->find("armigo_mul_rtz_ftz(") == std::string::npos,
                  "FMUL.RZ no longer calls the toward-zero helper");
        }
        rejected(unary_fixture(Opcode::sine, floating(Rounding::toward_negative)), "Function rounding",
                 DiagnosticCode::unsupported_modifier);
        auto unsupported_rounding = floating();
        unsupported_rounding.rounding = Rounding::nearest_away;
        rejected(binary_fixture(Opcode::minimum, unsupported_rounding), "Unproven extremum rounding",
                 DiagnosticCode::unsupported_modifier);
        auto saturating_extremum = floating();
        saturating_extremum.saturate = true;
        rejected(binary_fixture(Opcode::maximum, saturating_extremum), "Extremum saturation",
                 DiagnosticCode::unsupported_modifier);
    }
    { // sign operations perform no arithmetic.
        Fixture f;
        auto flushed = floating();
        flushed.flush_inputs = true;
        f.pure(Opcode::negate, {f.defined_word()}, 1, flushed);
        rejected(std::move(f), "Sign operations cannot flush", DiagnosticCode::unsupported_modifier);
    }
    { // the NaN test is unaffected by flushing.
        Fixture f;
        auto flushed = floating();
        flushed.flush_inputs = true;
        f.pure(Opcode::is_nan, {f.defined_word()}, 1, flushed, {}, ValueType::predicate);
        rejected(std::move(f), "NaN test cannot flush", DiagnosticCode::unsupported_modifier);
    }
    { // conversions admit the native casts and their emitter contracts.
        Fixture f;
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(), ConversionInfo{ScalarType::f32, ScalarType::s32});
        contains(accepted(f, "convert_f32_s32_rne"), "as_type<uint>(int(as_type<float>(value0)))",
                 "the cast truncates whatever the guest rounding produced");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(Rounding::toward_zero),
               ConversionInfo{ScalarType::u32, ScalarType::f32});
        contains(accepted(f, "convert_u32_f32_rz"), "armigo_i2f(value0, false, 32u, true, false)",
                 "directed integer-to-float keeps the exact helper");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(Rounding::toward_positive),
               ConversionInfo{ScalarType::s32, ScalarType::f32});
        rejected(std::move(f), "Signed integer-to-float toward-positive has no generator",
                 DiagnosticCode::unsupported_modifier);
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(Rounding::toward_negative),
               ConversionInfo{ScalarType::s16, ScalarType::f32});
        rejected(std::move(f), "Narrow integer-to-float toward-negative has no generator",
                 DiagnosticCode::unsupported_modifier);
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(Rounding::toward_zero),
               ConversionInfo{ScalarType::s16, ScalarType::f32});
        contains(accepted(f, "convert_s16_f32_rz"), "armigo_i2f(value0, true, 16u, true, false)",
                 "narrow directed integer-to-float keeps the exact helper");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(Rounding::toward_zero),
               ConversionInfo{ScalarType::f32, ScalarType::s32});
        contains(accepted(f, "convert_f32_s32_trunc"), "as_type<uint>(int(as_type<float>(value0)))",
                 "explicit truncation stays accepted");
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_wide()}, 1, floating(),
               ConversionInfo{ScalarType::s64, ScalarType::f32});
        rejected(std::move(f), "64-bit integer source", DiagnosticCode::unsupported_capability);
        f = Fixture{};
        f.pure(Opcode::convert, {f.defined_word()}, 1, floating(Rounding::nearest_even),
               ConversionInfo{ScalarType::f32, ScalarType::f32});
        contains(accepted(f, "same_type_convert"), "value1 = value0;", "Exact same-type conversion");
        f = Fixture{};
        auto pass = floating();
        pass.flush_inputs = pass.flush_output = true;
        f.pure(Opcode::convert, {f.defined_word(1)}, 1, pass, ConversionInfo{ScalarType::f32, ScalarType::f32});
        contains(accepted(f, "same_type_convert_ftz"),
                 "value1 = ((value0 & 0x7f800000u) == 0u ? (value0 & 0x80000000u) : (value0));",
                 "Same-type conversion honors its FTZ boundary");
        f = Fixture{};
        auto saturating = floating();
        saturating.saturate = true;
        f.pure(Opcode::convert, {f.defined_word()}, 1, saturating, ConversionInfo{ScalarType::f32, ScalarType::f32});
        contains(accepted(f, "same_type_convert_saturate"),
                 "as_type<uint>(precise::clamp(as_type<float>(value0), 0.0f, 1.0f))",
                 "Same-type saturation is the native unit clamp");
        f = Fixture{};
        auto flushing_saturation = floating();
        flushing_saturation.saturate = true;
        flushing_saturation.flush_inputs = flushing_saturation.flush_output = true;
        f.pure(Opcode::convert, {f.defined_word()}, 1, flushing_saturation,
               ConversionInfo{ScalarType::f32, ScalarType::f32});
        contains(accepted(f, "same_type_convert_ftz_saturate"),
                 "as_type<uint>(precise::clamp(as_type<float>(((value0 & 0x7f800000u) == 0u ? (value0 & 0x80000000u) : (value0))), 0.0f, 1.0f))",
                 "Same-type flush precedes the unit clamp");
        f = Fixture{};
        auto flushed = floating();
        flushed.flush_inputs = true;
        f.pure(Opcode::convert, {f.defined_word()}, 1, flushed, ConversionInfo{ScalarType::f32, ScalarType::s16});
        rejected(std::move(f), "Float-to-half-word destination", DiagnosticCode::unsupported_capability);
    }
    { // binary64 and binary16 arithmetic have no proven generators.
        Fixture f;
        f.pure(Opcode::add, {f.defined_wide(), f.defined_wide()}, 1,
               {Domain::floating, ScalarType::f64, Rounding::nearest_even}, {}, ValueType::word64);
        rejected(std::move(f), "Binary64 arithmetic", DiagnosticCode::unsupported_capability);
        Fixture g;
        g.pure(Opcode::multiply, {g.defined_word(), g.defined_word()}, 1,
               {Domain::floating, ScalarType::f16, Rounding::nearest_even});
        rejected(std::move(g), "Binary16 scalar arithmetic", DiagnosticCode::unsupported_capability);
    }
    { // unsigned absolute and wide multiply-high are not distinct operations.
        rejected(unary_fixture(Opcode::absolute, integer_domain(false)), "Unsigned absolute",
                 DiagnosticCode::unsupported_modifier);
        Fixture f;
        f.pure(Opcode::multiply_high, {f.defined_wide(), f.defined_wide()}, 1,
               {Domain::signed_integer, ScalarType::s64}, {}, ValueType::word64);
        rejected(std::move(f), "Wide multiply-high", DiagnosticCode::unsupported_capability);
    }
    { // a floating remainder has no producer and no generator.
        rejected(binary_fixture(Opcode::remainder, floating()), "Floating remainder",
                 DiagnosticCode::unsupported_capability);
    }
    { // raw-bit schemas refuse an interpreted numeric mode.
        Fixture f;
        f.pure(Opcode::pack_word64, {f.defined_word(), f.defined_word()}, 1, floating(), {}, ValueType::word64);
        rejected(std::move(f), "Word64 packing needs raw bits", DiagnosticCode::unsupported_modifier);
    }
}
} // namespace

#if defined(ARMIGO_MSL_EXPR_GPU)
namespace armigo::shader::test::msl_expr {
int run_gpu_cases();
}
#endif

int main(int argc, char** argv) {
    bool run_gpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--metal") compile_metal = true;
        else if (argument == "--gpu") run_gpu = true;
        else if (!argument.empty() && argument[0] != '-') {
            fixture_directory = argument;
            std::filesystem::create_directories(fixture_directory);
        }
    }
    opcode_text_tests();
    shuffle_text_tests();
    modifier_rejection_tests();
#if defined(ARMIGO_MSL_EXPR_GPU)
    if (run_gpu) failures += armigo::shader::test::msl_expr::run_gpu_cases();
#endif
    if (failures) {
        std::cerr << failures << " MSL expression tests failed\n";
        return 1;
    }
    std::cout << "MSL expression tests passed\n";
    return 0;
}
