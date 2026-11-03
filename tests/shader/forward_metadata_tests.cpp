// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Contract tests for pure forward-instruction metadata admission: every
// check re-derives the admission profile of the existing Builder and its
// registered family lowerers over a real initialized Builder, including the
// malformed false-guard forms the lowering path could otherwise skip. All
// headers and payloads are synthetic; no forward lowering runs here.

#include "forward_metadata.h"
#include "graphics_builder.h"
#include "test_support.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

namespace armigo::shader::test {
namespace {
namespace gx = graphics;
namespace mx = maxwell;
using Stage = shader_input::Stage;
using Family = mx::OpcodeFamily;
using Code = gx::Error::Code;

void put_word(std::array<std::byte, 80>& data, unsigned offset,
              std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        data[offset + i] = std::byte(static_cast<unsigned char>(value >> (i * 8)));
    }
}

// Minimal synthetic SPH: only the stage type field, everything else zero.
std::array<std::byte, 80> sph(Stage stage) {
    std::array<std::byte, 80> data{};
    put_word(data, 0, static_cast<std::uint32_t>(
                          shader_input::sph_stage_type_for(stage)) << 10);
    return data;
}

// Vertex SPH declaring generic input vector 0 (components 32..35), the
// position input x, the position output x, and generic output vector 0.
std::array<std::byte, 80> vertex_sph() {
    auto data = sph(Stage::Vertex);
    data[0x17] |= std::byte{1u << 4};
    data[0x18] |= std::byte{15};
    data[0x35] |= std::byte{1u << 4};
    data[0x36] |= std::byte{15};
    return data;
}

// Fragment SPH declaring the position w input (component 31) and generic
// input vector 0 with the perspective interpolation mode on lane 0.
std::array<std::byte, 80> fragment_sph_perspective() {
    auto data = sph(Stage::Fragment);
    data[0x17] |= std::byte{1u << 7};
    data[0x18] |= std::byte{2};
    return data;
}

// The same perspective vector without the position w input, so loading w
// reports component 31 as undeclared.
std::array<std::byte, 80> fragment_sph_perspective_without_w() {
    auto data = sph(Stage::Fragment);
    data[0x18] |= std::byte{2};
    return data;
}

// Fragment SPH declaring generic input vector 0 with screen-linear mode.
std::array<std::byte, 80> fragment_sph_linear() {
    auto data = sph(Stage::Fragment);
    data[0x18] |= std::byte{3};
    return data;
}

gx::Builder vertex_builder() {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Vertex, vertex_sph());
    return builder;
}

gx::Builder fragment_builder(const std::array<std::byte, 80>& header) {
    gx::Builder builder(gx::Limits{});
    builder.begin_graph(Stage::Fragment, header);
    return builder;
}

mx::Instruction make(Family family, mx::InstructionPayload payload,
                     mx::Predicate guard = {7, false},
                     std::uint32_t pc = 8) {
    return mx::Instruction{pc, 0, family, guard, std::move(payload)};
}

std::optional<gx::Error> validate(const gx::Builder& builder,
                                  const mx::Instruction& instruction) {
    return gx::detail::validate_forward_instruction_metadata(instruction,
                                                             builder);
}

void admits(const gx::Builder& builder, const mx::Instruction& instruction) {
    const auto error = validate(builder, instruction);
    if (error.has_value()) {
        record_failure(__FILE__, __LINE__,
                       "instruction unexpectedly rejected with code " +
                           std::to_string(static_cast<int>(error->code)) +
                           " detail " + std::to_string(error->detail));
    }
}

void rejects(const gx::Builder& builder, const mx::Instruction& instruction,
             Code code, std::uint32_t detail = 0, bool check_detail = false) {
    const auto error = validate(builder, instruction);
    if (!error.has_value()) {
        record_failure(__FILE__, __LINE__, "expected rejection was not reported");
        return;
    }
    if (error->code != code) {
        record_failure(__FILE__, __LINE__,
                       "unexpected rejection code " +
                           std::to_string(static_cast<int>(error->code)));
    }
    if (error->pc != instruction.pc) {
        record_failure(__FILE__, __LINE__, "rejection pc differs from instruction pc");
    }
    if (check_detail && error->detail != detail) {
        record_failure(__FILE__, __LINE__, "rejection detail differs");
    }
}

constexpr mx::Predicate pt{7, false};
constexpr mx::Predicate not_pt{7, true};
constexpr mx::Predicate p0{0, false};
constexpr mx::RegisterOperand rz{255};
constexpr mx::RegisterOperand r1{1};
constexpr mx::RegisterOperand r2{2};
constexpr mx::RegisterOperand r5{5};
constexpr mx::RegisterOperand d0{0};
constexpr mx::RegisterOperand d4{4};

// A representative register-move word in every family-independent shape.
mx::Instruction mov(mx::RegisterOperand destination, mx::RegisterOperand source,
                    std::uint8_t mask = 15, mx::Predicate guard = pt) {
    return make(Family::MovRegister,
                mx::MovRegister{destination, source, mask}, guard);
}
}  // namespace

// The motivating gap: a guard-false instruction with malformed metadata must
// still be rejected, because the lowering path skips it before the Builder
// ever validates the payload.
ARMIGO_TEST(forward_metadata, false_guard_mov32_immediate_mask_zero_rejected) {
    const auto builder = vertex_builder();
    rejects(builder,
            make(Family::Mov32Immediate,
                 mx::Mov32Immediate{r1, 0, 0x40000000}, not_pt),
            Code::UnsupportedModifier);
    // The identical shape under an inverted PT guard is exactly the skipped
    // form; the well-formed mask stays admissible under the same guard.
    admits(builder,
           make(Family::Mov32Immediate,
                mx::Mov32Immediate{r1, 15, 0x40000000}, not_pt));
}

ARMIGO_TEST(forward_metadata, guard_profiles_admit_and_forged_index_rejected) {
    const auto builder = vertex_builder();
    admits(builder, mov(r1, r2, 15, pt));
    admits(builder, mov(r1, r2, 15, not_pt));
    admits(builder, mov(r1, r2, 15, p0));
    mx::Instruction forged = mov(r1, r2, 15, pt);
    forged.predicate.index = 9;
    rejects(builder, forged, Code::InvalidInstruction, 9, true);
}

ARMIGO_TEST(forward_metadata, payload_type_mismatch_rejected) {
    const auto builder = vertex_builder();
    rejects(builder,
            make(Family::Mov32Immediate, mx::MovRegister{r1, r2, 15}),
            Code::InvalidInstruction);
    rejects(builder,
            make(Family::FloatAddRegister, mx::FloatMultiply{r1, r2, rz}),
            Code::InvalidInstruction);
    rejects(builder,
            make(Family::Exit, mx::MovRegister{r1, r2, 15}),
            Code::InvalidInstruction);
}

ARMIGO_TEST(forward_metadata, unknown_and_branch_families_rejected_even_false_guard) {
    const auto builder = vertex_builder();
    rejects(builder, make(Family::Unknown, mx::MovRegister{r1, r2, 15}, not_pt),
            Code::InvalidInstruction);
    rejects(builder, make(Family::Branch, mx::MovRegister{r1, r2, 15}, not_pt),
            Code::InvalidInstruction);
}

ARMIGO_TEST(forward_metadata, predicate_set_predicate_is_unsupported) {
    const auto builder = vertex_builder();
    rejects(builder,
            make(Family::PredicateSetPredicate,
                 mx::PredicateSet{1, 2, {3, false}, {4, false}, {5, false},
                                  mx::PredicateBooleanOperation::And,
                                  mx::PredicateBooleanOperation::Or}),
            Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, isetp_register_profile) {
    const auto builder = vertex_builder();
    const mx::IntegerSetPredicate valid{1, 2, r1, r2, {7, false}, true};
    admits(builder, make(Family::IntegerSetPredicateRegister, valid));
    rejects(builder,
            make(Family::IntegerSetPredicateRegister,
                 mx::IntegerSetPredicate{8, 2, r1, r2, {7, false}, false}),
            Code::InvalidInstruction);
    rejects(builder,
            make(Family::IntegerSetPredicateRegister,
                 mx::IntegerSetPredicate{1, 2, r1, r2, {3, false}, false}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::IntegerSetPredicateRegister,
                 mx::IntegerSetPredicate{1, 2, r1, r2, {7, true}, false}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::IntegerSetPredicateRegister, mx::MovRegister{r1, r2, 15}),
            Code::InvalidInstruction);
}

ARMIGO_TEST(forward_metadata, mov_register_masks_alias_and_undefined_source) {
    const auto builder = vertex_builder();
    admits(builder, mov(r1, r2, 15));
    admits(builder, mov(r1, r2, 1));
    rejects(builder, mov(r1, r2, 4), Code::UnsupportedModifier);
    // RZ sources alias the zero literal; reads of undefined registers are a
    // definedness concern of the executed path, not metadata admission.
    admits(builder, mov(r1, rz, 15));
    admits(builder, mov(r1, r5, 15));
    rejects(builder, make(Family::MovRegister, mx::WordAndImmediate{r1, r2, 15}),
            Code::InvalidInstruction);
}

ARMIGO_TEST(forward_metadata, mov_constant_masks_and_buffer_bounds) {
    const auto builder = vertex_builder();
    const mx::ConstantBufferOperand edge{17, 65532};
    admits(builder, make(Family::MovConstant, mx::MovConstant{r1, edge, 1}));
    rejects(builder,
            make(Family::MovConstant,
                 mx::MovConstant{r1, mx::ConstantBufferOperand{18, 0}, 15}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::MovConstant,
                 mx::MovConstant{r1, mx::ConstantBufferOperand{1, 2}, 15}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::MovConstant,
                 mx::MovConstant{r1, mx::ConstantBufferOperand{1, 65536}, 15}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::MovConstant, mx::MovConstant{r1, edge, 2}),
            Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, float_add_profiles) {
    const auto builder = vertex_builder();
    admits(builder,
           make(Family::FloatAddRegister,
                mx::FloatAdd{r1, r2, r5, mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero, false}));
    rejects(builder,
            make(Family::FloatAddRegister,
                 mx::FloatAdd{r1, r2, r5, mx::RoundingMode::Zero,
                              mx::FlushMode::FlushToZero, false}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::FloatAddConstant,
                 mx::FloatAdd{r1, r2, mx::ConstantBufferOperand{0, 0},
                              mx::RoundingMode::NearestEven,
                              mx::FlushMode::FlushToZero, true}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::FloatAddConstant,
                 mx::FloatAdd{r1, r2, r5, mx::RoundingMode::NearestEven,
                              mx::FlushMode::FlushToZero, false}),
            Code::InvalidInstruction);
    rejects(builder,
            make(Family::FloatAddRegister,
                 mx::FloatAdd{r1, r2, mx::ConstantBufferOperand{0, 0},
                              mx::RoundingMode::NearestEven,
                              mx::FlushMode::FlushToZero, false}),
            Code::InvalidInstruction);
    rejects(builder,
            make(Family::FloatAddRegister,
                 mx::FloatAdd{r1, r2, r5, mx::RoundingMode::NearestEven,
                              mx::FlushMode::None, false}),
            Code::UnsupportedModifier);
    admits(builder,
           make(Family::FloatAddImmediate,
                mx::FloatAdd{r1, r2, mx::ShortFloatImmediate{0x3f800000},
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero, false}));
    rejects(builder,
            make(Family::FloatAddImmediate,
                 mx::FloatAdd{r1, r2, mx::ShortFloatImmediate{4095},
                              mx::RoundingMode::NearestEven,
                              mx::FlushMode::FlushToZero, false}),
            Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, float_multiply_profiles) {
    const auto builder = vertex_builder();
    admits(builder,
           make(Family::FloatMultiplyRegister,
                mx::FloatMultiply{r1, r2, r5, mx::RoundingMode::NearestEven,
                                  mx::FlushMode::FlushToZero}));
    admits(builder,
           make(Family::FloatMultiplyRegister,
                mx::FloatMultiply{r1, r2, r5, mx::RoundingMode::Zero,
                                  mx::FlushMode::FlushToZero}));
    rejects(builder,
           make(Family::FloatMultiplyRegister,
                mx::FloatMultiply{r1, r2, r5, mx::RoundingMode::PositiveInfinity,
                                  mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
    rejects(builder,
           make(Family::FloatMultiplyConstant,
                mx::FloatMultiply{r1, r2, r5, mx::RoundingMode::NearestEven,
                                  mx::FlushMode::FlushToZero}),
           Code::InvalidInstruction);
    rejects(builder,
           make(Family::FloatMultiplyConstant,
                mx::FloatMultiply{r1, r2, mx::ConstantBufferOperand{18, 0},
                                  mx::RoundingMode::NearestEven,
                                  mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
    rejects(builder,
           make(Family::FloatMultiplyImmediate,
                mx::FloatMultiply{r1, r2, mx::ShortFloatImmediate{1},
                                  mx::RoundingMode::NearestEven,
                                  mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, saturate_register_syntax) {
    const auto builder = vertex_builder();
    admits(builder,
           make(Family::FloatMultiplySaturateRegister,
                mx::FloatMultiply{r1, r2, r5, mx::RoundingMode::NearestEven,
                                  mx::FlushMode::FlushToZero}));
    rejects(builder,
           make(Family::FloatMultiplySaturateRegister,
                mx::FloatMultiply{r1, r2, r5, mx::RoundingMode::Zero,
                                  mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
    rejects(builder,
           make(Family::FloatMultiplySaturateRegister,
                mx::FloatMultiply{r1, r2, mx::ConstantBufferOperand{0, 0},
                                  mx::RoundingMode::NearestEven,
                                  mx::FlushMode::FlushToZero}),
           Code::InvalidInstruction);
}

ARMIGO_TEST(forward_metadata, float_fma_profiles) {
    const auto builder = vertex_builder();
    admits(builder,
           make(Family::FloatFmaImmediate,
                mx::FloatFma{r1, r2, mx::ShortFloatImmediate{0x3f800000}, r5,
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero}));
    rejects(builder,
           make(Family::FloatFmaImmediate,
                mx::FloatFma{r1, r2, mx::ShortFloatImmediate{1}, r5,
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
    admits(builder,
           make(Family::FloatFmaConstantRegister,
                mx::FloatFma{r1, r2, mx::ConstantBufferOperand{0, 4}, r5,
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero}));
    rejects(builder,
           make(Family::FloatFmaConstantRegister,
                mx::FloatFma{r1, r2, mx::ShortFloatImmediate{0}, r5,
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero}),
           Code::InvalidInstruction);
    rejects(builder,
           make(Family::FloatFmaConstantRegister,
                mx::FloatFma{r1, r2, mx::ConstantBufferOperand{0, 0}, r5,
                             mx::RoundingMode::NegativeInfinity,
                             mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, fma_registered_families) {
    const auto builder = vertex_builder();
    admits(builder,
           make(Family::FloatFmaRegisterConstant,
                mx::FloatFmaRegisterConstant{r1, r2, r5,
                                             mx::ConstantBufferOperand{0, 4},
                                             mx::RoundingMode::NearestEven,
                                             mx::FlushMode::FlushToZero, true}));
    rejects(builder,
           make(Family::FloatFmaRegisterConstant,
                mx::FloatFmaRegisterConstant{
                    r1, r2, r5, mx::ConstantBufferOperand{0, 0},
                    mx::RoundingMode::Zero, mx::FlushMode::FlushToZero, false}),
           Code::UnsupportedModifier);
    rejects(builder,
           make(Family::FloatFmaRegisterConstant,
                mx::FloatFmaRegisterConstant{
                    r1, r2, r5, mx::ConstantBufferOperand{18, 0},
                    mx::RoundingMode::NearestEven,
                    mx::FlushMode::FlushToZero, false}),
           Code::UnsupportedModifier);
    admits(builder,
           make(Family::FloatFmaConstantNegated,
                mx::FloatFmaConstantNegated{r1, r2, mx::ConstantBufferOperand{1, 8},
                                            r5, mx::RoundingMode::NearestEven,
                                            mx::FlushMode::FlushToZero}));
    rejects(builder,
           make(Family::FloatFmaConstantNegated,
                mx::FloatFmaConstantNegated{
                    r1, r2, mx::ConstantBufferOperand{1, 8}, r5,
                    mx::RoundingMode::NearestEven, mx::FlushMode::None}),
           Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, float_set_register_profile) {
    const auto builder = vertex_builder();
    admits(builder,
           make(Family::FloatSetRegister,
                mx::FloatSetRegister{r1, r2, r5, {7, false},
                                     mx::OrderedComparison::GreaterThan, false,
                                     true, mx::FlushMode::FlushToZero}));
    rejects(builder,
           make(Family::FloatSetRegister,
                mx::FloatSetRegister{r1, r2, r5, {7, true},
                                     mx::OrderedComparison::LessThan, false,
                                     true, mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
    rejects(builder,
           make(Family::FloatSetRegister,
                mx::FloatSetRegister{r1, r2, r5, {7, false},
                                     mx::OrderedComparison::LessThan, false,
                                     false, mx::FlushMode::FlushToZero}),
           Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, word_and_and_reciprocal_are_admitted) {
    const auto builder = vertex_builder();
    admits(builder,
           make(Family::WordAndImmediate, mx::WordAndImmediate{r1, r2, 0xff}));
    admits(builder,
           make(Family::MultiFunction, mx::Reciprocal{r1, r2}));
    rejects(builder,
           make(Family::MultiFunction, mx::WordAndImmediate{r1, r2, 0xff}),
           Code::InvalidInstruction);
}

ARMIGO_TEST(forward_metadata, vertex_attribute_load_and_store) {
    const auto builder = vertex_builder();
    // Generic input vector 0 spans components 32..35 (byte offsets 128..140);
    // the declared store targets the matching generic output vector.
    admits(builder,
           make(Family::AttributeLoad,
                mx::AttributeAccess{r1, rz, rz, 128, 0, false}));
    rejects(builder,
            make(Family::AttributeLoad,
                 mx::AttributeAccess{r1, rz, rz, 64, 0, false}),
            Code::UndeclaredAttribute, 16, true);
    admits(builder,
           make(Family::AttributeStore,
                mx::AttributeAccess{r2, rz, rz, 128, 0, false}));
    rejects(builder,
            make(Family::AttributeStore,
                 mx::AttributeAccess{r2, rz, rz, 256, 0, false}),
            Code::UndeclaredAttribute, 64, true);
    rejects(builder,
            make(Family::AttributeLoad,
                 mx::AttributeAccess{r1, rz, rz, 0, 4, false}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::AttributeLoad,
                 mx::AttributeAccess{r1, r5, rz, 0, 0, false}),
            Code::UnsupportedModifier);
    rejects(builder,
            make(Family::AttributeLoad,
                 mx::AttributeAccess{r1, rz, rz, 0, 0, true}),
            Code::UnsupportedModifier);
    const auto fragment = fragment_builder(fragment_sph_linear());
    rejects(fragment,
            make(Family::AttributeLoad,
                 mx::AttributeAccess{r1, rz, rz, 0, 0, false}),
            Code::UnsupportedStage);
}

ARMIGO_TEST(forward_metadata, fragment_interpolation_profile) {
    const auto perspective = fragment_builder(fragment_sph_perspective());
    admits(perspective,
           make(Family::InterpolateAttribute,
                mx::InterpolateAttribute{r1, rz, rz, 32, 0x7ff, false}));
    // A perspective vector reads position.w, so component 31 must be
    // declared; without it the perspective header rejects at component 31.
    const auto perspective_without_w =
        fragment_builder(fragment_sph_perspective_without_w());
    rejects(perspective_without_w,
            make(Family::InterpolateAttribute,
                 mx::InterpolateAttribute{r1, rz, rz, 32, 0x7ff, false}),
            Code::UndeclaredAttribute, 31, true);
    // A screen-linear vector never reads position.w, so the same
    // undeclared w input stays admissible.
    const auto linear = fragment_builder(fragment_sph_linear());
    admits(linear,
           make(Family::InterpolateAttribute,
                mx::InterpolateAttribute{r1, rz, rz, 32, 0x7ff, false}));
    rejects(perspective,
            make(Family::InterpolateAttribute,
                 mx::InterpolateAttribute{r1, rz, rz, 32, 0x700, false}),
            Code::UnsupportedModifier);
    rejects(perspective,
            make(Family::InterpolateAttribute,
                 mx::InterpolateAttribute{r1, r5, r5, 32, 0x7ff, false}),
            Code::UnsupportedModifier);
    rejects(perspective,
            make(Family::InterpolateAttribute,
                 mx::InterpolateAttribute{r1, rz, rz, 64, 0x7ff, false}),
            Code::UndeclaredAttribute, 64, true);
    const auto vertex = vertex_builder();
    rejects(vertex,
            make(Family::InterpolateAttribute,
                 mx::InterpolateAttribute{r1, rz, rz, 32, 0x7ff, false}),
            Code::UnsupportedStage);
}

ARMIGO_TEST(forward_metadata, texture_sample_profile) {
    const auto fragment = fragment_builder(fragment_sph_linear());
    admits(fragment,
           make(Family::TextureSample,
                mx::TextureSample2D{d0, d4, r1, r2, 8, false}));
    rejects(fragment,
            make(Family::TextureSample,
                 mx::TextureSample2D{mx::RegisterOperand{1}, d4, r1, r2, 0,
                                     false}),
            Code::UnsupportedModifier);
    rejects(fragment,
            make(Family::TextureSample,
                 mx::TextureSample2D{mx::RegisterOperand{254}, d4, r1, r2, 0,
                                     false}),
            Code::UnsupportedModifier);
    rejects(fragment,
            make(Family::TextureSample,
                 mx::TextureSample2D{d0, d4, r1, r2, 32768, false}),
            Code::UnsupportedModifier);
    rejects(fragment,
            make(Family::TextureSample,
                 mx::TextureSample2D{d0, d4, r1, r2, 2, false}),
            Code::UnsupportedModifier);
    const auto vertex = vertex_builder();
    rejects(vertex,
            make(Family::TextureSample, mx::TextureSample2D{d0, d4, r1, r2, 0,
                                                            false}),
            Code::UnsupportedStage);
}

ARMIGO_TEST(forward_metadata, exit_flow_test) {
    const auto builder = vertex_builder();
    admits(builder, make(Family::Exit, mx::Exit{15}));
    rejects(builder, make(Family::Exit, mx::Exit{14}),
            Code::UnsupportedModifier);
}

ARMIGO_TEST(forward_metadata, validator_leaves_builder_untouched) {
    auto builder = vertex_builder();
    const auto nodes_before = builder.program.nodes().size();
    const auto exports_before = builder.program.exports().size();
    // MOV R1, R5 reads a register no instruction defined; admission is
    // metadata-only, so the undefined source stays unobserved and no node,
    // register, or stored value may appear.
    admits(builder, mov(r1, r5, 15));
    ARMIGO_CHECK_EQ(builder.program.nodes().size(), nodes_before);
    ARMIGO_CHECK_EQ(builder.program.exports().size(), exports_before);
    for (const auto& entry : builder.registers) {
        ARMIGO_CHECK(!entry.has_value());
    }
    for (const auto& entry : builder.stored) {
        ARMIGO_CHECK(!entry.has_value());
    }
    rejects(builder, mov(r1, r2, 4), Code::UnsupportedModifier);
    ARMIGO_CHECK_EQ(builder.program.nodes().size(), nodes_before);
    for (const auto& entry : builder.registers) {
        ARMIGO_CHECK(!entry.has_value());
    }
}
}  // namespace armigo::shader::test
