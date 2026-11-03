// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic shader IR programs shared by the tests.
#pragma once

#include <armigo/shader/ir.h>

#include <cstdint>
#include <string>
#include <vector>

namespace armigo::shader::test {

struct NamedProgram {
    std::string name;
    Program program;
};

std::uint32_t f32_bits(float value);
float bits_f32(std::uint32_t bits);

// output0 = input0 * parameter0 + parameter1
Program scale_bias_program();
// output0 = (input0 + input1) + input2
Program sum_left_program();
// output0 = input0 + (input1 + input2)
Program sum_right_program();
// output0 = input0 * input1 + input2, evaluated as two rounded operations.
Program mul_add_program();
// output0 = input0 * 0.0; output1 = input0 + 0.0. Both are identities only
// under unsafe algebra, so they expose signed-zero and NaN folding.
Program zero_identity_program();
// Mixes f32 and u32 parameters, inputs, constants, and outputs.
Program mixed_types_program();

// Defines values that no instruction uses.
Program dead_values_program();

// Every valid fixture, in a fixed order, for emission and compilation checks.
std::vector<NamedProgram> msl_fixtures();

}  // namespace armigo::shader::test
