// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "fixtures.h"
#include "test_support.h"

#include <armigo/shader/validation.h>

#include <limits>
#include <vector>

namespace armigo::shader::test {
namespace {

constexpr std::uint32_t kNone = Diagnostic::kNoInstruction;

std::vector<DiagnosticCode> codes(const std::vector<Diagnostic>& diagnostics) {
    std::vector<DiagnosticCode> result;
    for (const Diagnostic& diagnostic : diagnostics) {
        result.push_back(diagnostic.code);
    }
    return result;
}

// Asserts that `program` produces exactly one diagnostic with the given code
// and instruction index.
void expect_single(const Program& program, DiagnosticCode code,
                   std::uint32_t instruction, const char* file, int line) {
    const std::vector<Diagnostic> diagnostics = validate(program);
    if (diagnostics.size() != 1 || diagnostics[0].code != code ||
        diagnostics[0].instruction != instruction) {
        record_failure(file, line,
                       std::string("expected exactly ") + to_string(code) +
                           " at " + std::to_string(instruction) + ", got:\n" +
                           describe_diagnostics(diagnostics));
    } else if (diagnostics[0].message.empty()) {
        record_failure(file, line, "diagnostic message is empty");
    }
}

#define EXPECT_SINGLE(program, code, instruction) \
    expect_single(program, DiagnosticCode::code, instruction, __FILE__, __LINE__)

// One f32 input copied to one f32 output: the smallest valid program.
Program passthrough() {
    ProgramBuilder b("passthrough");
    b.add_input(ScalarType::f32);
    b.add_output(ScalarType::f32);
    b.store_output(ScalarType::f32, 0, b.load_input(ScalarType::f32, 0));
    return b.take();
}

Instruction raw(Opcode opcode, ScalarType type, ValueId a, ValueId b,
                std::uint32_t immediate) {
    return Instruction{opcode, type, a, b, immediate};
}

ValueId v(std::uint32_t index) { return ValueId{index}; }

}  // namespace

ARMIGO_TEST(validation, fixtures_are_valid) {
    for (const NamedProgram& fixture : msl_fixtures()) {
        const std::vector<Diagnostic> diagnostics = validate(fixture.program);
        if (!diagnostics.empty()) {
            record_failure(__FILE__, __LINE__,
                           fixture.name + " is invalid:\n" +
                               describe_diagnostics(diagnostics));
        }
    }
    ARMIGO_CHECK(validate(passthrough()).empty());
}

ARMIGO_TEST(validation, entry_point_names) {
    ARMIGO_CHECK(is_valid_entry_point("kernel_main"));
    ARMIGO_CHECK(!is_valid_entry_point("_private"));
    ARMIGO_CHECK(is_valid_entry_point("Blend2"));
    ARMIGO_CHECK(is_valid_entry_point("float5"));
    ARMIGO_CHECK(is_valid_entry_point(std::string(kMaxEntryPointLength, 'a')));

    ARMIGO_CHECK(!is_valid_entry_point(""));
    ARMIGO_CHECK(!is_valid_entry_point("2fast"));
    ARMIGO_CHECK(!is_valid_entry_point("has-dash"));
    ARMIGO_CHECK(!is_valid_entry_point("has space"));
    ARMIGO_CHECK(!is_valid_entry_point("caf\xc3\xa9"));
    ARMIGO_CHECK(!is_valid_entry_point(std::string(kMaxEntryPointLength + 1, 'a')));
    ARMIGO_CHECK(!is_valid_entry_point("_Upper"));
    ARMIGO_CHECK(!is_valid_entry_point("__x"));
    ARMIGO_CHECK(!is_valid_entry_point("a__b"));
    ARMIGO_CHECK(!is_valid_entry_point("armigo_kernel"));
    for (const char* reserved :
         {"kernel", "float", "uint", "main", "metal", "device", "constant",
          "return", "as_type", "threadgroup", "half", "bfloat"}) {
        if (is_valid_entry_point(reserved)) {
            record_failure(__FILE__, __LINE__,
                           std::string("accepted reserved word ") + reserved);
        }
    }
    for (const char* type_name :
         {"float4", "half3", "uint2", "bool4", "float4x4", "half2x3"}) {
        if (is_valid_entry_point(type_name)) {
            record_failure(__FILE__, __LINE__,
                           std::string("accepted type name ") + type_name);
        }
    }

    Program program = passthrough();
    program.entry_point = "kernel";
    EXPECT_SINGLE(program, invalid_entry_point, kNone);
}

ARMIGO_TEST(validation, declaration_limits) {
    Program program = passthrough();
    program.outputs.clear();
    program.body.pop_back();
    EXPECT_SINGLE(program, no_outputs, kNone);

    program = passthrough();
    program.parameters.assign(kMaxParameters + 1, ScalarType::f32);
    EXPECT_SINGLE(program, too_many_parameters, kNone);

    program = passthrough();
    program.parameters.assign(kMaxParameters, ScalarType::f32);
    ARMIGO_CHECK(validate(program).empty());

    program = passthrough();
    program.inputs.assign(kMaxIoBindings, ScalarType::f32);
    EXPECT_SINGLE(program, too_many_bindings, kNone);

    program = passthrough();
    program.inputs.assign(kMaxIoBindings - 1, ScalarType::f32);
    ARMIGO_CHECK(validate(program).empty());

    program = passthrough();
    program.body.resize(kMaxInstructions + 1,
                        raw(Opcode::constant, ScalarType::u32, ValueId::none(),
                            ValueId::none(), 0));
    EXPECT_SINGLE(program, too_many_instructions, kNone);
}

ARMIGO_TEST(validation, unknown_types_and_opcodes) {
    const auto bad_type = static_cast<ScalarType>(7);

    Program program = passthrough();
    program.parameters.push_back(bad_type);
    EXPECT_SINGLE(program, unknown_type, kNone);

    program = passthrough();
    program.body.insert(program.body.begin(),
                        raw(static_cast<Opcode>(42), ScalarType::f32,
                            ValueId::none(), ValueId::none(), 0));
    // The load becomes instruction 1 and the store refers to it.
    program.body[2].operand0 = v(1);
    EXPECT_SINGLE(program, unknown_opcode, 0);

    program = passthrough();
    program.body.insert(program.body.begin(),
                        raw(Opcode::constant, bad_type, ValueId::none(),
                            ValueId::none(), 0));
    program.body[2].operand0 = v(1);
    EXPECT_SINGLE(program, unknown_type, 0);

    // A store with an unknown type does not also report its output as missing.
    program = passthrough();
    program.body[1].type = bad_type;
    EXPECT_SINGLE(program, unknown_type, 1);
}

ARMIGO_TEST(validation, unused_fields_must_be_empty) {
    Program program = passthrough();
    program.body[0].operand0 = v(0);
    EXPECT_SINGLE(program, unexpected_field, 0);

    program = passthrough();
    program.body[1].operand1 = v(0);
    EXPECT_SINGLE(program, unexpected_field, 1);

    ProgramBuilder b("immediate_on_add");
    b.add_output(ScalarType::f32);
    const ValueId one = b.constant_f32(1.0f);
    const ValueId sum = b.fadd(one, one);
    b.store_output(ScalarType::f32, 0, sum);
    program = b.take();
    program.body[1].immediate = 3;
    EXPECT_SINGLE(program, unexpected_field, 1);
}

ARMIGO_TEST(validation, invalid_value_references) {
    ProgramBuilder b("references");
    b.add_output(ScalarType::f32);
    const ValueId one = b.constant_f32(1.0f);
    const ValueId sum = b.fadd(one, one);
    b.store_output(ScalarType::f32, 0, sum);
    const Program base = b.take();

    Program program = base;
    program.body[1].operand1 = ValueId::none();
    EXPECT_SINGLE(program, operand_missing, 1);

    program = base;
    program.body[1].operand0 = v(99);
    EXPECT_SINGLE(program, operand_out_of_range, 1);

    // Self reference.
    program = base;
    program.body[1].operand1 = v(1);
    EXPECT_SINGLE(program, operand_not_dominating, 1);

    // Forward reference.
    program = base;
    program.body.insert(program.body.begin() + 1,
                        raw(Opcode::fmul, ScalarType::f32, v(0), v(2), 0));
    program.body[3].operand0 = v(2);
    EXPECT_SINGLE(program, operand_not_dominating, 1);

    // Reference to a store, which defines no value.
    ProgramBuilder c("store_reference");
    c.add_output(ScalarType::f32);
    c.add_output(ScalarType::f32);
    const ValueId k = c.constant_f32(2.0f);
    c.store_output(ScalarType::f32, 0, k);
    c.store_output(ScalarType::f32, 1, ValueId{1});
    EXPECT_SINGLE(c.program(), operand_has_no_value, 2);
}

ARMIGO_TEST(validation, type_rules) {
    // fadd and fmul require f32 operands.
    ProgramBuilder b("typed");
    b.add_output(ScalarType::f32);
    const ValueId count = b.constant_u32(3);
    const ValueId one = b.constant_f32(1.0f);
    b.store_output(ScalarType::f32, 0, b.fmul(one, count));
    EXPECT_SINGLE(b.program(), operand_type_mismatch, 2);

    // fadd results are f32.
    Program program = passthrough();
    program.body.insert(program.body.begin() + 1,
                        raw(Opcode::fadd, ScalarType::u32, v(0), v(0), 0));
    EXPECT_SINGLE(program, result_type_mismatch, 1);

    // Loads must match the declaration.
    program = passthrough();
    program.body[0].type = ScalarType::u32;
    program.body[1].type = ScalarType::u32;
    const std::vector<Diagnostic> diagnostics = validate(program);
    ARMIGO_CHECK_EQ(codes(diagnostics).size(), 2u);
    if (diagnostics.size() == 2) {
        ARMIGO_CHECK_EQ(diagnostics[0].code,
                        DiagnosticCode::result_type_mismatch);
        ARMIGO_CHECK_EQ(diagnostics[0].instruction, 0u);
        ARMIGO_CHECK_EQ(diagnostics[1].code,
                        DiagnosticCode::output_type_mismatch);
        ARMIGO_CHECK_EQ(diagnostics[1].instruction, 1u);
    }

    // Store type must match the stored value.
    program = passthrough();
    program.outputs[0] = ScalarType::u32;
    program.body[1].type = ScalarType::u32;
    EXPECT_SINGLE(program, operand_type_mismatch, 1);

    // Store type must match the output declaration.
    ProgramBuilder c("store_typed");
    c.add_output(ScalarType::f32);
    c.store_output(ScalarType::u32, 0, c.constant_u32(1));
    EXPECT_SINGLE(c.program(), output_type_mismatch, 1);

    ProgramBuilder d("parameter_typed");
    d.add_parameter(ScalarType::u32);
    d.add_output(ScalarType::f32);
    d.store_output(ScalarType::f32, 0, d.load_parameter(ScalarType::f32, 0));
    EXPECT_SINGLE(d.program(), result_type_mismatch, 0);
}

ARMIGO_TEST(validation, nonfinite_constants) {
    const std::uint32_t rejected[] = {
        0x7f800000u,  // +infinity
        0xff800000u,  // -infinity
        0x7fc00000u,  // quiet NaN
        0x7f800001u,  // signaling NaN
        0xffffffffu,  // negative NaN with full payload
    };
    for (std::uint32_t bits : rejected) {
        Program program = passthrough();
        program.body.insert(program.body.begin(),
                            raw(Opcode::constant, ScalarType::f32,
                                ValueId::none(), ValueId::none(), bits));
        program.body[2].operand0 = v(1);
        EXPECT_SINGLE(program, nonfinite_constant, 0);
    }

    const std::uint32_t accepted[] = {
        0x00000000u,  // +0
        0x80000000u,  // -0
        0x00000001u,  // smallest subnormal
        0x7f7fffffu,  // largest finite
        0xff7fffffu,  // most negative finite
    };
    for (std::uint32_t bits : accepted) {
        ProgramBuilder b("finite");
        b.add_output(ScalarType::f32);
        b.store_output(ScalarType::f32, 0, b.constant_f32(bits_f32(bits)));
        ARMIGO_CHECK(validate(b.program()).empty());
    }

    // The same bit patterns are ordinary u32 constants.
    ProgramBuilder c("u32_bits");
    c.add_output(ScalarType::u32);
    c.store_output(ScalarType::u32, 0, c.constant_u32(0x7fc00000u));
    ARMIGO_CHECK(validate(c.program()).empty());
}

ARMIGO_TEST(validation, binding_bounds) {
    Program program = passthrough();
    program.body[0].immediate = 1;
    EXPECT_SINGLE(program, input_out_of_range, 0);

    program = passthrough();
    program.body[1].immediate = 1;
    const std::vector<Diagnostic> diagnostics = validate(program);
    ARMIGO_CHECK_EQ(diagnostics.size(), 2u);
    if (diagnostics.size() == 2) {
        ARMIGO_CHECK_EQ(diagnostics[0].code,
                        DiagnosticCode::output_out_of_range);
        ARMIGO_CHECK_EQ(diagnostics[0].instruction, 1u);
        ARMIGO_CHECK_EQ(diagnostics[1].code,
                        DiagnosticCode::output_never_stored);
        ARMIGO_CHECK_EQ(diagnostics[1].instruction, kNone);
    }

    program = passthrough();
    program.body[1].immediate = std::numeric_limits<std::uint32_t>::max();
    ARMIGO_CHECK_EQ(validate(program).size(), 2u);

    ProgramBuilder b("parameter_bounds");
    b.add_parameter(ScalarType::f32);
    b.add_output(ScalarType::f32);
    b.store_output(ScalarType::f32, 0, b.load_parameter(ScalarType::f32, 1));
    EXPECT_SINGLE(b.program(), parameter_out_of_range, 0);
}

ARMIGO_TEST(validation, output_store_counts) {
    Program program = passthrough();
    program.body.push_back(program.body[1]);
    EXPECT_SINGLE(program, output_stored_twice, 2);

    program = passthrough();
    program.outputs.push_back(ScalarType::f32);
    EXPECT_SINGLE(program, output_never_stored, kNone);
}

ARMIGO_TEST(validation, invalid_operands_do_not_cascade) {
    // Instruction 1 has a bad operand. Instructions 2 and 3 use its result
    // and must not report further diagnostics.
    ProgramBuilder b("cascade");
    b.add_output(ScalarType::f32);
    const ValueId one = b.constant_f32(1.0f);
    const ValueId bad = b.fadd(one, ValueId{50});
    const ValueId product = b.fmul(bad, one);
    b.store_output(ScalarType::f32, 0, product);
    EXPECT_SINGLE(b.program(), operand_out_of_range, 1);
}

ARMIGO_TEST(validation, reports_all_problems_in_order) {
    Program program;
    program.entry_point = "1bad";
    program.inputs = {ScalarType::f32};
    program.outputs = {ScalarType::f32, ScalarType::f32};
    program.body = {
        raw(Opcode::constant, ScalarType::f32, ValueId::none(), ValueId::none(),
            0x7f800000u),
        raw(Opcode::load_input, ScalarType::f32, ValueId::none(),
            ValueId::none(), 4),
        raw(Opcode::store_output, ScalarType::f32, v(9), ValueId::none(), 0),
    };
    const std::vector<DiagnosticCode> expected = {
        DiagnosticCode::invalid_entry_point,
        DiagnosticCode::nonfinite_constant,
        DiagnosticCode::input_out_of_range,
        DiagnosticCode::operand_out_of_range,
        DiagnosticCode::output_never_stored,
    };
    const std::vector<Diagnostic> diagnostics = validate(program);
    ARMIGO_CHECK(codes(diagnostics) == expected);
    if (codes(diagnostics) != expected) {
        record_failure(__FILE__, __LINE__, describe_diagnostics(diagnostics));
    }
    // Validation is deterministic.
    const std::vector<Diagnostic> again = validate(program);
    ARMIGO_CHECK_EQ(again.size(), diagnostics.size());
    for (std::size_t i = 0; i < again.size() && i < diagnostics.size(); ++i) {
        ARMIGO_CHECK_EQ(format(again[i]), format(diagnostics[i]));
    }
}

}  // namespace armigo::shader::test
