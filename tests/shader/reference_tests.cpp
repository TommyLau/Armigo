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

#include <armigo/shader/reference_evaluator.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace armigo::shader::test {
namespace {

using Words = std::vector<std::uint32_t>;

Words f32_words(std::initializer_list<float> values) {
    Words words;
    for (float value : values) {
        words.push_back(f32_bits(value));
    }
    return words;
}

ReferenceResult run(const Program& program, const Words& parameters,
                    const std::vector<Words>& inputs, std::uint32_t count,
                    SubnormalMode subnormals = SubnormalMode::preserve) {
    return evaluate_reference(program, parameters, inputs, count,
                              ReferenceOptions{subnormals});
}

bool is_nan_bits(std::uint32_t bits) {
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
}

}  // namespace

ARMIGO_TEST(reference, scale_bias_values) {
    // Values chosen so every intermediate result is exactly representable.
    const ReferenceResult result =
        run(scale_bias_program(), f32_words({2.0f, -0.5f}),
            {f32_words({0.0f, 1.0f, -3.25f, 1024.0f})}, 4);
    ARMIGO_CHECK(result.ok());
    ARMIGO_CHECK_EQ(result.outputs.size(), 1u);
    ARMIGO_CHECK(result.outputs[0] ==
                 f32_words({-0.5f, 1.5f, -7.0f, 2047.5f}));
}

ARMIGO_TEST(reference, addition_is_not_reassociated) {
    // (1e8 + -1e8) + 1 == 1, but 1e8 + (-1e8 + 1) == 0 because -1e8 + 1
    // rounds back to -1e8 in binary32.
    const std::vector<Words> inputs = {f32_words({1.0e8f}),
                                       f32_words({-1.0e8f}),
                                       f32_words({1.0f})};
    const ReferenceResult left = run(sum_left_program(), {}, inputs, 1);
    const ReferenceResult right = run(sum_right_program(), {}, inputs, 1);
    ARMIGO_CHECK(left.ok() && right.ok());
    ARMIGO_CHECK_EQ(left.outputs[0][0], f32_bits(1.0f));
    ARMIGO_CHECK_EQ(right.outputs[0][0], f32_bits(0.0f));
}

ARMIGO_TEST(reference, multiply_add_rounds_twice) {
    // a = b = 1 + 2^-12, so a * b = 1 + 2^-11 + 2^-24. The product rounds
    // (ties to even) to 1 + 2^-11, and adding -(1 + 2^-11) gives +0. A fused
    // multiply-add would instead return 2^-24.
    const float a = std::ldexp(1.0f, 0) + std::ldexp(1.0f, -12);
    const float c = -(1.0f + std::ldexp(1.0f, -11));
    const ReferenceResult result =
        run(mul_add_program(), {},
            {f32_words({a}), f32_words({a}), f32_words({c})}, 1);
    ARMIGO_CHECK(result.ok());
    ARMIGO_CHECK_EQ(result.outputs[0][0], 0x00000000u);
    ARMIGO_CHECK(result.outputs[0][0] != f32_bits(std::ldexp(1.0f, -24)));
}

ARMIGO_TEST(reference, signed_zero_and_special_values) {
    const float inf = INFINITY;
    const ReferenceResult result =
        run(zero_identity_program(), {},
            {f32_words({-0.0f, 0.0f, -2.0f, inf, NAN})}, 5);
    ARMIGO_CHECK(result.ok());
    const Words& product = result.outputs[0];
    const Words& sum = result.outputs[1];
    // x * +0
    ARMIGO_CHECK_EQ(product[0], 0x80000000u);
    ARMIGO_CHECK_EQ(product[1], 0x00000000u);
    ARMIGO_CHECK_EQ(product[2], 0x80000000u);
    ARMIGO_CHECK(is_nan_bits(product[3]));
    ARMIGO_CHECK(is_nan_bits(product[4]));
    // x + +0: -0 + +0 is +0 in round-to-nearest.
    ARMIGO_CHECK_EQ(sum[0], 0x00000000u);
    ARMIGO_CHECK_EQ(sum[1], 0x00000000u);
    ARMIGO_CHECK_EQ(sum[2], f32_bits(-2.0f));
    ARMIGO_CHECK_EQ(sum[3], f32_bits(inf));
    ARMIGO_CHECK(is_nan_bits(sum[4]));
}

ARMIGO_TEST(reference, overflow_and_subnormals) {
    ProgramBuilder b("edges");
    b.add_input(ScalarType::f32);
    b.add_output(ScalarType::f32);
    b.add_output(ScalarType::f32);
    const ValueId x = b.load_input(ScalarType::f32, 0);
    const ValueId big = b.constant_f32(bits_f32(0x7f7fffffu));
    const ValueId quarter = b.constant_f32(0.25f);
    b.store_output(ScalarType::f32, 0, b.fmul(x, big));
    b.store_output(ScalarType::f32, 1, b.fmul(x, quarter));
    // 2^-126 * 0.25 is the subnormal 2^-128; 4 * FLT_MAX overflows.
    const ReferenceResult result =
        run(b.program(), {}, {f32_words({std::ldexp(1.0f, -126), 4.0f})}, 2);
    ARMIGO_CHECK(result.ok());
    ARMIGO_CHECK_EQ(result.outputs[1][0], 0x00200000u);
    ARMIGO_CHECK_EQ(result.outputs[0][1], f32_bits(INFINITY));
    ARMIGO_CHECK_EQ(result.outputs[1][1], f32_bits(1.0f));
}

ARMIGO_TEST(reference, subnormal_modes) {
    // output0 = x * 2^20, output1 = x + 2^-126, output2 = x * 0.25,
    // output3 = x (copy).
    ProgramBuilder b("subnormals");
    b.add_input(ScalarType::f32);
    for (int i = 0; i < 4; ++i) {
        b.add_output(ScalarType::f32);
    }
    const ValueId x = b.load_input(ScalarType::f32, 0);
    const ValueId scale = b.constant_f32(std::ldexp(1.0f, 20));
    const ValueId min_normal = b.constant_f32(std::ldexp(1.0f, -126));
    const ValueId quarter = b.constant_f32(0.25f);
    b.store_output(ScalarType::f32, 0, b.fmul(x, scale));
    b.store_output(ScalarType::f32, 1, b.fadd(x, min_normal));
    b.store_output(ScalarType::f32, 2, b.fmul(x, quarter));
    b.store_output(ScalarType::f32, 3, x);
    const Program program = b.take();
    // 2^-140 and the negative smallest subnormal are subnormal operands;
    // 2^-126 * 0.25 has a subnormal result.
    const std::vector<Words> inputs = {
        Words{f32_bits(std::ldexp(1.0f, -140)), 0x80000001u,
              f32_bits(std::ldexp(1.0f, -126))}};

    const ReferenceResult preserved = run(program, {}, inputs, 3);
    ARMIGO_CHECK(preserved.ok());
    ARMIGO_CHECK(preserved.outputs[0] ==
                 (Words{0x03800000u, 0x80100000u, 0x0a800000u}));
    ARMIGO_CHECK(preserved.outputs[1] ==
                 (Words{0x00800200u, 0x007fffffu, 0x01000000u}));
    ARMIGO_CHECK(preserved.outputs[2] ==
                 (Words{0x00000080u, 0x80000000u, 0x00200000u}));
    ARMIGO_CHECK(preserved.outputs[3] == inputs[0]);

    const ReferenceResult flushed =
        run(program, {}, inputs, 3, SubnormalMode::flush_to_zero);
    ARMIGO_CHECK(flushed.ok());
    ARMIGO_CHECK(flushed.outputs[0] ==
                 (Words{0x00000000u, 0x80000000u, 0x0a800000u}));
    ARMIGO_CHECK(flushed.outputs[1] ==
                 (Words{0x00800000u, 0x00800000u, 0x01000000u}));
    ARMIGO_CHECK(flushed.outputs[2] ==
                 (Words{0x00000000u, 0x80000000u, 0x00000000u}));
    // Copies keep subnormal bit patterns in both modes.
    ARMIGO_CHECK(flushed.outputs[3] == inputs[0]);
}

ARMIGO_TEST(reference, mixed_types_pass_raw_words) {
    const ReferenceResult result = run(
        mixed_types_program(), {f32_bits(4.0f), 0xdeadbeefu},
        {f32_words({1.0f, -0.125f, 3.0f}), Words{7u, 0xffffffffu, 0u}}, 3);
    ARMIGO_CHECK(result.ok());
    ARMIGO_CHECK_EQ(result.outputs.size(), 4u);
    ARMIGO_CHECK(result.outputs[0] == f32_words({4.5f, 0.0f, 12.5f}));
    ARMIGO_CHECK(result.outputs[1] == (Words{7u, 0xffffffffu, 0u}));
    ARMIGO_CHECK(result.outputs[2] == (Words{0xdeadbeefu, 0xdeadbeefu,
                                             0xdeadbeefu}));
    ARMIGO_CHECK(result.outputs[3] == (Words{0xa5a5a5a5u, 0xa5a5a5a5u,
                                             0xa5a5a5a5u}));
}

ARMIGO_TEST(reference, extra_input_elements_are_ignored) {
    const ReferenceResult result =
        run(scale_bias_program(), f32_words({1.0f, 1.0f}),
            {f32_words({1.0f, 2.0f, 3.0f})}, 2);
    ARMIGO_CHECK(result.ok());
    ARMIGO_CHECK(result.outputs[0] == f32_words({2.0f, 3.0f}));

    const ReferenceResult empty =
        run(scale_bias_program(), f32_words({1.0f, 1.0f}), {Words{}}, 0);
    ARMIGO_CHECK(empty.ok());
    ARMIGO_CHECK_EQ(empty.outputs.size(), 1u);
    ARMIGO_CHECK(empty.outputs[0].empty());
}

ARMIGO_TEST(reference, rejects_invalid_programs) {
    Program program = scale_bias_program();
    program.body[4].operand0 = ValueId{4};
    const ReferenceResult result =
        run(program, f32_words({1.0f, 1.0f}), {f32_words({1.0f})}, 1);
    ARMIGO_CHECK(!result.ok());
    ARMIGO_CHECK(result.outputs.empty());
    ARMIGO_CHECK_EQ(result.diagnostics.size(), 1u);
    if (!result.diagnostics.empty()) {
        ARMIGO_CHECK_EQ(result.diagnostics[0].code,
                        DiagnosticCode::operand_not_dominating);
    }
}

ARMIGO_TEST(reference, rejects_malformed_arguments) {
    const Program program = scale_bias_program();

    ReferenceResult result =
        run(program, f32_words({1.0f}), {f32_words({1.0f})}, 1);
    ARMIGO_CHECK_EQ(result.diagnostics.size(), 1u);
    if (!result.diagnostics.empty()) {
        ARMIGO_CHECK_EQ(result.diagnostics[0].code,
                        DiagnosticCode::parameter_count_mismatch);
    }
    ARMIGO_CHECK(result.outputs.empty());

    result = run(program, f32_words({1.0f, 1.0f}), {}, 1);
    ARMIGO_CHECK_EQ(result.diagnostics.size(), 1u);
    if (!result.diagnostics.empty()) {
        ARMIGO_CHECK_EQ(result.diagnostics[0].code,
                        DiagnosticCode::input_count_mismatch);
    }

    result = run(program, f32_words({1.0f, 1.0f}), {f32_words({1.0f})}, 2);
    ARMIGO_CHECK_EQ(result.diagnostics.size(), 1u);
    if (!result.diagnostics.empty()) {
        ARMIGO_CHECK_EQ(result.diagnostics[0].code,
                        DiagnosticCode::input_too_short);
    }
    ARMIGO_CHECK(result.outputs.empty());
}

ARMIGO_TEST(reference, rejects_unknown_subnormal_mode) {
    const ReferenceResult result = evaluate_reference(
        scale_bias_program(), f32_words({1.0f, 1.0f}),
        std::vector<Words>{f32_words({1.0f})}, 1,
        ReferenceOptions{static_cast<SubnormalMode>(255)});
    ARMIGO_CHECK(!result.ok());
    ARMIGO_CHECK(result.outputs.empty());
    ARMIGO_CHECK_EQ(result.diagnostics.size(), 1u);
    if (!result.diagnostics.empty()) {
        ARMIGO_CHECK_EQ(result.diagnostics[0].code,
                        DiagnosticCode::invalid_subnormal_mode);
    }
}

}  // namespace armigo::shader::test
