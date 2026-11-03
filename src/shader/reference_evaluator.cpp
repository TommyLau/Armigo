// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/reference_evaluator.h>

#include <armigo/shader/validation.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Each IR operation is evaluated as one separately rounded binary32 operation.
// The build also passes -ffp-contract=off for this file; the pragma keeps the
// guarantee when the file is compiled by other means.
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace armigo::shader {
namespace {

static_assert(std::numeric_limits<float>::is_iec559,
              "the reference evaluator requires IEEE-754 binary32 floats");
static_assert(sizeof(float) == sizeof(std::uint32_t));

float as_f32(std::uint32_t bits) { return std::bit_cast<float>(bits); }
std::uint32_t as_bits(float value) { return std::bit_cast<std::uint32_t>(value); }

// Replaces a subnormal bit pattern with a zero of the same sign. Bit-level
// handling keeps the result independent of the host floating-point mode.
std::uint32_t flush_subnormal(std::uint32_t bits) {
    const bool subnormal =
        (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
    return subnormal ? (bits & 0x80000000u) : bits;
}

// Evaluates one binary f32 operation under the selected subnormal mode.
template <typename Operation>
std::uint32_t binary_f32(std::uint32_t lhs, std::uint32_t rhs,
                         SubnormalMode mode, Operation operation) {
    if (mode == SubnormalMode::flush_to_zero) {
        lhs = flush_subnormal(lhs);
        rhs = flush_subnormal(rhs);
    }
    const std::uint32_t result = as_bits(operation(as_f32(lhs), as_f32(rhs)));
    return mode == SubnormalMode::flush_to_zero ? flush_subnormal(result)
                                                : result;
}

std::vector<Diagnostic> check_arguments(
    const Program& program, std::span<const std::uint32_t> parameters,
    std::span<const std::vector<std::uint32_t>> inputs,
    std::uint32_t element_count) {
    std::vector<Diagnostic> diagnostics;
    if (parameters.size() != program.parameters.size()) {
        diagnostics.push_back(
            {DiagnosticCode::parameter_count_mismatch,
             Diagnostic::kNoInstruction,
             "expected " + std::to_string(program.parameters.size()) +
                 " parameter words, got " + std::to_string(parameters.size())});
    }
    if (inputs.size() != program.inputs.size()) {
        diagnostics.push_back(
            {DiagnosticCode::input_count_mismatch, Diagnostic::kNoInstruction,
             "expected " + std::to_string(program.inputs.size()) +
                 " input arrays, got " + std::to_string(inputs.size())});
        return diagnostics;
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() < element_count) {
            diagnostics.push_back(
                {DiagnosticCode::input_too_short, Diagnostic::kNoInstruction,
                 "input " + std::to_string(i) + " has " +
                     std::to_string(inputs[i].size()) +
                     " elements, fewer than the element count " +
                     std::to_string(element_count)});
        }
    }
    return diagnostics;
}

}  // namespace

ReferenceResult evaluate_reference(
    const Program& program, std::span<const std::uint32_t> parameters,
    std::span<const std::vector<std::uint32_t>> inputs,
    std::uint32_t element_count, const ReferenceOptions& options) {
    ReferenceResult result;
    result.diagnostics = validate(program);
    if (!result.diagnostics.empty()) {
        return result;
    }
    result.diagnostics =
        check_arguments(program, parameters, inputs, element_count);
    if (options.subnormals != SubnormalMode::preserve &&
        options.subnormals != SubnormalMode::flush_to_zero) {
        result.diagnostics.push_back(
            {DiagnosticCode::invalid_subnormal_mode, Diagnostic::kNoInstruction,
             "subnormal mode is not a supported enumerator"});
    }
    if (!result.diagnostics.empty()) {
        return result;
    }

    result.outputs.assign(program.outputs.size(),
                          std::vector<std::uint32_t>(element_count, 0));
    // Raw 32-bit word of every value; store_output slots are left unused.
    std::vector<std::uint32_t> values(program.body.size(), 0);

    for (std::uint32_t invocation = 0; invocation < element_count;
         ++invocation) {
        for (std::size_t i = 0; i < program.body.size(); ++i) {
            const Instruction& inst = program.body[i];
            switch (inst.opcode) {
            case Opcode::constant:
                values[i] = inst.immediate;
                break;
            case Opcode::load_parameter:
                values[i] = parameters[inst.immediate];
                break;
            case Opcode::load_input:
                values[i] = inputs[inst.immediate][invocation];
                break;
            case Opcode::fadd:
                values[i] = binary_f32(
                    values[inst.operand0.index], values[inst.operand1.index],
                    options.subnormals, [](float lhs, float rhs) {
                        const float sum = lhs + rhs;
                        return sum;
                    });
                break;
            case Opcode::fmul:
                values[i] = binary_f32(
                    values[inst.operand0.index], values[inst.operand1.index],
                    options.subnormals, [](float lhs, float rhs) {
                        const float product = lhs * rhs;
                        return product;
                    });
                break;
            case Opcode::store_output:
                result.outputs[inst.immediate][invocation] =
                    values[inst.operand0.index];
                break;
            }
        }
    }
    return result;
}

}  // namespace armigo::shader
