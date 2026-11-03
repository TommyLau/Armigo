// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Registered families on the guarded forward path. Forward metadata admission
// runs each registered lowerer in validation-only mode, so a family needs no
// separate metadata case: these tests lower forged forward programs whose
// guarded instruction is a registered family under PT, a runtime predicate in
// both outcomes, and never-taken !PT, and check that an unsupported modifier is
// still rejected under a false guard. Synthetic programs only.
#include <armigo/shader/graphics_forward.h>
#include "test_support.h"

#include <armigo/shader/families/ffma_register.h>
#include <armigo/shader/families/float_immediate32.h>
#include <armigo/shader/families/mov_constant.h>
#include <armigo/shader/families/range_reduction.h>
#include <armigo/shader/maxwell_forward_program.h>

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace gx = armigo::shader::graphics;
namespace mw = armigo::shader::maxwell;
namespace si = armigo::shader_input;
using Family = mw::OpcodeFamily;

constexpr std::uint32_t kDefault = 0x40e00000u;  // 7.0f, written before the guard

mw::Predicate pt() { return mw::Predicate{7, false}; }
mw::Predicate never() { return mw::Predicate{7, true}; }
mw::Predicate p0() { return mw::Predicate{0, false}; }
mw::RegisterOperand r(unsigned index) {
    return mw::RegisterOperand{static_cast<std::uint8_t>(index)};
}

std::array<std::byte, 80> fragment_sph() {
    std::array<std::byte, 80> sph{};
    const std::uint32_t word0 = si::sph_stage_type_for(si::Stage::Fragment)
                                << si::format::sph_stage_shift;
    for (unsigned i = 0; i < 4; ++i) sph[i] = std::byte(word0 >> (i * 8));
    sph[0x48] = std::byte{1};  // color target 0, lane x
    return sph;
}

mw::Instruction at(std::uint32_t pc, Family family, mw::Predicate guard,
                   mw::InstructionPayload payload) {
    return mw::Instruction{pc, 0, family, guard, std::move(payload)};
}

// R1 = 2.0, R2 = 3.0, R3 = 0.5, R4 = c0[0] (runtime), R5 = 0,
// P0 = (R4 != R5) through ISETP, R0 = 7.0, then the guarded instruction
// writes R0, then EXIT. R0 is the exported color.
mw::ForwardProgram program(mw::Instruction guarded) {
    std::vector<mw::ForwardInstruction> nodes{
        at(8, Family::Mov32Immediate, pt(), mw::Mov32Immediate{r(1), 15, 0x40000000u}),
        at(16, Family::Mov32Immediate, pt(), mw::Mov32Immediate{r(2), 15, 0x40400000u}),
        at(24, Family::Mov32Immediate, pt(), mw::Mov32Immediate{r(3), 15, 0x3f000000u}),
        at(40, Family::MovConstant, pt(), mw::MovConstant{r(4), {0, 0}, 15}),
        at(48, Family::Mov32Immediate, pt(), mw::Mov32Immediate{r(5), 15, 0}),
        at(56, Family::IntegerSetPredicateRegister, pt(),
           mw::IntegerSetPredicate{0, 7, r(4), r(5), pt(), false}),
        at(72, Family::Mov32Immediate, pt(), mw::Mov32Immediate{r(0), 15, kDefault}),
    };
    guarded.pc = 80;
    nodes.push_back(guarded);
    nodes.push_back(at(88, Family::Exit, pt(), mw::Exit{15}));
    mw::ForwardProgram result;
    result.stage = si::Stage::Fragment;
    result.sph = fragment_sph();
    result.program_size = 96;
    result.instructions = std::move(nodes);
    return result;
}

std::uint32_t color(const gx::LowerResult& lowered, std::uint32_t condition) {
    const auto* lowered_program = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(lowered_program != nullptr);
    if (lowered_program == nullptr) return 0xdeadbeefu;
    const std::array<std::uint32_t, 1> words{condition};
    gx::Inputs inputs;
    inputs.constant_buffers[0] = words;
    const auto evaluated = gx::evaluate(*lowered_program, inputs);
    const auto* outputs = std::get_if<gx::Evaluation>(&evaluated);
    ARMIGO_CHECK(outputs != nullptr && !outputs->outputs.empty());
    if (outputs == nullptr || outputs->outputs.empty()) return 0xdeadbeefu;
    return outputs->outputs.front().bits;
}

// Lowers the instruction under PT, P0 (both runtime outcomes) and !PT.
void check_guards(mw::Instruction instruction, std::uint32_t expected) {
    instruction.predicate = pt();
    const auto always = gx::lower_forward(program(instruction));
    ARMIGO_CHECK_EQ(color(always, 0), expected);

    instruction.predicate = p0();
    const auto guarded = gx::lower_forward(program(instruction));
    ARMIGO_CHECK_EQ(color(guarded, 1), expected);
    ARMIGO_CHECK_EQ(color(guarded, 0), kDefault);

    instruction.predicate = never();
    const auto skipped = gx::lower_forward(program(instruction));
    ARMIGO_CHECK_EQ(color(skipped, 1), kDefault);
}

void check_rejected_under_false_guard(mw::Instruction instruction) {
    instruction.predicate = never();
    const auto lowered = gx::lower_forward(program(instruction));
    const auto* error = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(error != nullptr);
    if (error != nullptr) {
        ARMIGO_CHECK(error->code == gx::Error::Code::UnsupportedModifier);
        ARMIGO_CHECK_EQ(error->pc, 80u);
    }
}

}  // namespace

ARMIGO_TEST(forward_registered, ffma_register_on_every_guard) {
    const mw::FloatFmaRegister value{r(0), r(1), r(2), r(3)};
    check_guards(at(0, Family::FloatFmaRegister, pt(), value), 0x40d00000u);  // 6.5
    auto rounding = value;
    rounding.rounding = mw::RoundingMode::Zero;
    check_rejected_under_false_guard(at(0, Family::FloatFmaRegister, pt(), rounding));
    auto negate_c = value;
    negate_c.negate_c = true;
    check_guards(at(0, Family::FloatFmaRegister, pt(), negate_c), 0x40b00000u);  // 5.5
}

ARMIGO_TEST(forward_registered, immediate32_on_every_guard) {
    const mw::FloatMultiplyImmediate32 multiply{r(0), r(1), 0x40800000u,
                                                mw::FlushMode::FlushToZero, false};
    check_guards(at(0, Family::FloatMultiplyImmediate32, pt(), multiply), 0x41000000u);
    const mw::FloatAddImmediate32 add{r(0), r(2), 0x3f800000u, false, false, false, false};
    check_guards(at(0, Family::FloatAddImmediate32, pt(), add), 0x40800000u);
    auto flush = multiply;
    flush.flush = mw::FlushMode::None;
    check_rejected_under_false_guard(at(0, Family::FloatMultiplyImmediate32, pt(), flush));
}

ARMIGO_TEST(forward_registered, range_reduction_and_barrier_on_every_guard) {
    const mw::RangeReduction copy{r(0), r(2), true, false, false};
    check_guards(at(0, Family::RangeReductionRegister, pt(), copy), 0x40400000u);
    const mw::RangeReduction constant{r(0), mw::ConstantBufferOperand{0, 0}, false,
                                      false, false};
    // The guarded copy reads c0[0]; with P0 true that word is the condition 1.
    auto constant_instruction = at(0, Family::RangeReductionConstant, pt(), constant);
    constant_instruction.predicate = p0();
    ARMIGO_CHECK_EQ(color(gx::lower_forward(program(constant_instruction)), 1), 1u);
    auto negated = copy;
    negated.negate = true;
    check_guards(at(0, Family::RangeReductionRegister, pt(), negated), 0xc0400000u);
    // DEPBAR changes nothing on any guard.
    check_guards(at(0, Family::DependencyBarrier, pt(), mw::DependencyBarrier{0}), kDefault);
}

}  // namespace armigo::shader::test
