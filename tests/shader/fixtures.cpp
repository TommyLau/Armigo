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

#include <bit>

namespace armigo::shader::test {

std::uint32_t f32_bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

float bits_f32(std::uint32_t bits) { return std::bit_cast<float>(bits); }

Program scale_bias_program() {
    ProgramBuilder b("scale_bias");
    const std::uint32_t scale = b.add_parameter(ScalarType::f32);
    const std::uint32_t bias = b.add_parameter(ScalarType::f32);
    const std::uint32_t in = b.add_input(ScalarType::f32);
    const std::uint32_t out = b.add_output(ScalarType::f32);
    const ValueId x = b.load_input(ScalarType::f32, in);
    const ValueId s = b.load_parameter(ScalarType::f32, scale);
    const ValueId product = b.fmul(x, s);
    const ValueId c = b.load_parameter(ScalarType::f32, bias);
    const ValueId sum = b.fadd(product, c);
    b.store_output(ScalarType::f32, out, sum);
    return b.take();
}

namespace {

Program sum_program(const char* name, bool left) {
    ProgramBuilder b(name);
    for (int i = 0; i < 3; ++i) {
        b.add_input(ScalarType::f32);
    }
    const std::uint32_t out = b.add_output(ScalarType::f32);
    const ValueId x = b.load_input(ScalarType::f32, 0);
    const ValueId y = b.load_input(ScalarType::f32, 1);
    const ValueId z = b.load_input(ScalarType::f32, 2);
    ValueId result;
    if (left) {
        result = b.fadd(b.fadd(x, y), z);
    } else {
        const ValueId yz = b.fadd(y, z);
        result = b.fadd(x, yz);
    }
    b.store_output(ScalarType::f32, out, result);
    return b.take();
}

}  // namespace

Program sum_left_program() { return sum_program("sum_left", true); }

Program sum_right_program() { return sum_program("sum_right", false); }

Program mul_add_program() {
    ProgramBuilder b("mul_add");
    for (int i = 0; i < 3; ++i) {
        b.add_input(ScalarType::f32);
    }
    const std::uint32_t out = b.add_output(ScalarType::f32);
    const ValueId x = b.load_input(ScalarType::f32, 0);
    const ValueId y = b.load_input(ScalarType::f32, 1);
    const ValueId z = b.load_input(ScalarType::f32, 2);
    const ValueId product = b.fmul(x, y);
    b.store_output(ScalarType::f32, out, b.fadd(product, z));
    return b.take();
}

Program zero_identity_program() {
    ProgramBuilder b("zero_identity");
    const std::uint32_t in = b.add_input(ScalarType::f32);
    const std::uint32_t product_out = b.add_output(ScalarType::f32);
    const std::uint32_t sum_out = b.add_output(ScalarType::f32);
    const ValueId x = b.load_input(ScalarType::f32, in);
    const ValueId zero = b.constant_f32(0.0f);
    b.store_output(ScalarType::f32, product_out, b.fmul(x, zero));
    b.store_output(ScalarType::f32, sum_out, b.fadd(x, zero));
    return b.take();
}

Program mixed_types_program() {
    ProgramBuilder b("mixed_types");
    const std::uint32_t gain = b.add_parameter(ScalarType::f32);
    const std::uint32_t tag = b.add_parameter(ScalarType::u32);
    const std::uint32_t samples = b.add_input(ScalarType::f32);
    const std::uint32_t ids = b.add_input(ScalarType::u32);
    const std::uint32_t scaled_out = b.add_output(ScalarType::f32);
    const std::uint32_t id_out = b.add_output(ScalarType::u32);
    const std::uint32_t tag_out = b.add_output(ScalarType::u32);
    const std::uint32_t marker_out = b.add_output(ScalarType::u32);

    const ValueId sample = b.load_input(ScalarType::f32, samples);
    const ValueId g = b.load_parameter(ScalarType::f32, gain);
    const ValueId half = b.constant_f32(0.5f);
    const ValueId scaled = b.fadd(b.fmul(sample, g), half);
    b.store_output(ScalarType::f32, scaled_out, scaled);
    b.store_output(ScalarType::u32, id_out, b.load_input(ScalarType::u32, ids));
    b.store_output(ScalarType::u32, tag_out,
                   b.load_parameter(ScalarType::u32, tag));
    b.store_output(ScalarType::u32, marker_out, b.constant_u32(0xa5a5a5a5u));
    return b.take();
}

Program dead_values_program() {
    ProgramBuilder b("dead_values");
    b.add_parameter(ScalarType::u32);
    b.add_input(ScalarType::f32);
    b.add_output(ScalarType::f32);
    const ValueId x = b.load_input(ScalarType::f32, 0);
    b.load_parameter(ScalarType::u32, 0);
    b.fmul(x, x);
    b.store_output(ScalarType::f32, 0, x);
    return b.take();
}

std::vector<NamedProgram> msl_fixtures() {
    std::vector<NamedProgram> fixtures = {
        {"scale_bias", scale_bias_program()},
        {"sum_left", sum_left_program()},
        {"sum_right", sum_right_program()},
        {"mul_add", mul_add_program()},
        {"zero_identity", zero_identity_program()},
        {"mixed_types", mixed_types_program()},
        {"dead_values", dead_values_program()},
        // The entry point repeats a generated parameter name, which is legal
        // because function parameters shadow the kernel name.
        {"input0", [] {
             Program program = scale_bias_program();
             program.entry_point = "input0";
             return program;
         }()},
    };
    for (const char* name : {"NAN", "INFINITY", "FLT_MAX", "assert",
                             "packed_float3", "complex"}) {
        Program program = scale_bias_program();
        program.entry_point = name;
        fixtures.push_back({name, std::move(program)});
    }
    return fixtures;
}

}  // namespace armigo::shader::test
