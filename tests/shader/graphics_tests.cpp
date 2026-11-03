// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_msl.h>
#include "family_registry_validation.h"
#include "graphics_families.h"
#include "test_support.h"

#include <bit>
#include <cfenv>
#include <cstddef>
#include <stdexcept>

namespace armigo::shader::test {
namespace {
namespace gx = graphics;
namespace mx = maxwell;
using Stage = shader_input::Stage;
using Family = mx::OpcodeFamily;

std::uint32_t fbits(float value) { return std::bit_cast<std::uint32_t>(value); }
void put(mx::DecodedProgram& program, unsigned at, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) program.sph[at + i] = std::byte(value >> (i * 8));
}
mx::DecodedProgram base(Stage stage) {
    mx::DecodedProgram result;
    result.stage = stage;
    put(result, 0, shader_input::sph_stage_type_for(stage) << 10);
    return result;
}
void attribute(mx::DecodedProgram& program, unsigned component, bool output = false,
               unsigned interpolation = 2) {
    unsigned at = 0, bit = 0, value = 1;
    if (component >= 28 && component < 32) {
        at = output ? 0x35 : 0x17;
        bit = component - 28 + 4;
    } else if (program.stage == Stage::Vertex) {
        at = (output ? 0x36 : 0x18) + (component - 32) / 8;
        bit = (component - 32) % 8;
    } else {
        at = 0x18 + (component - 32) / 4;
        bit = ((component - 32) % 4) * 2;
        value = interpolation;
    }
    program.sph[at] |= std::byte(value << bit);
}
void append(mx::DecodedProgram& program, Family family, mx::InstructionPayload payload) {
    const auto i = program.instructions.size();
    program.instructions.push_back({static_cast<std::uint32_t>((i / 3) * 32 + (i % 3 + 1) * 8),
        0, family, {7, false}, std::move(payload)});
}
void mov(mx::DecodedProgram& program, std::uint8_t reg, std::uint32_t bits) {
    append(program, Family::Mov32Immediate, mx::Mov32Immediate{{reg}, 15, bits});
}
void reg_move(mx::DecodedProgram& program, std::uint8_t destination,
              std::uint8_t source, std::uint8_t write_mask = 15) {
    append(program, Family::MovRegister,
           mx::MovRegister{{destination}, {source}, write_mask});
}
void load(mx::DecodedProgram& program, std::uint8_t reg, unsigned component) {
    append(program, Family::AttributeLoad, mx::AttributeAccess{{reg}, {255}, {255},
        static_cast<std::uint16_t>(component * 4), 0, false});
}
void store(mx::DecodedProgram& program, std::uint8_t reg, unsigned component) {
    append(program, Family::AttributeStore, mx::AttributeAccess{{reg}, {255}, {255},
        static_cast<std::uint16_t>(component * 4), 0, false});
}
void exit(mx::DecodedProgram& program) { append(program, Family::Exit, mx::Exit{15}); }
void ipa(mx::DecodedProgram& program, std::uint8_t reg, std::uint8_t component) {
    append(program, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{reg}, {255}, {255}, component, 0x7ff});
}
gx::EvaluateResult run(const mx::DecodedProgram& program, const gx::Inputs& inputs = {}) {
    const auto lowered = gx::lower(program);
    if (const auto* error = std::get_if<gx::Error>(&lowered)) return *error;
    return gx::evaluate(std::get<gx::Program>(lowered), inputs);
}
void output(const gx::EvaluateResult& result, gx::OutputKind kind, unsigned component,
            std::uint32_t expected) {
    const auto* evaluated = std::get_if<gx::Evaluation>(&result);
    if (!evaluated) {
        record_failure(__FILE__, __LINE__, std::get<gx::Error>(result).message());
        return;
    }
    for (const auto& value : evaluated->outputs) {
        if (value.kind == kind && value.component == component) {
            ARMIGO_CHECK_EQ(value.bits, expected);
            return;
        }
    }
    record_failure(__FILE__, __LINE__, "Expected graphics output was not exported");
}
template<class T> void rejected(const std::variant<T, gx::Error>& result, gx::Error::Code code) {
    const auto* error = std::get_if<gx::Error>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error) ARMIGO_CHECK(error->code == code);
}
mx::DecodedProgram one_output() {
    auto program = base(Stage::Vertex);
    attribute(program, 28, true);
    return program;
}
}

ARMIGO_TEST(graphics, register_versions_zero_and_last_attribute_store) {
    auto program = one_output();
    attribute(program, 32);
    attribute(program, 32, true);
    load(program, 2, 32);
    mov(program, 255, fbits(42.0f));
    append(program, Family::FloatAddRegister, mx::FloatAdd{{2}, {2}, mx::RegisterOperand{255},
        mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero, true});
    store(program, 2, 28);
    mov(program, 2, fbits(7.0f));
    store(program, 2, 28);
    store(program, 255, 32);
    exit(program);
    const std::array attrs = {gx::AttributeInput{32, fbits(-3.0f)}};
    gx::Inputs inputs;
    inputs.attributes = attrs;
    const auto result = run(program, inputs);
    output(result, gx::OutputKind::Position, 0, fbits(7.0f));
    output(result, gx::OutputKind::Generic, 0, 0);
    const auto lowered = gx::lower(program);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (const auto* ir = std::get_if<gx::Program>(&lowered)) {
        const auto& add = std::get<gx::FloatArithmetic>(ir->nodes()[3].operation);
        ARMIGO_CHECK(add.lhs.index < 3 && add.rhs.index < 3);
        ARMIGO_CHECK_EQ(ir->exports().size(), 2u);
    }
}

ARMIGO_TEST(graphics, ordered_float_booleans_with_signed_ftz) {
    const std::array<std::pair<std::uint32_t, bool>, 7> cases = {{
        {fbits(-2.0f), true}, {fbits(3.0f), false}, {0u, false}, {0x80000000u, false},
        {0x80000001u, false}, {1u, false}, {0x7fc01234u, false},
    }};
    for (const auto& [input, negative] : cases) {
        auto program = one_output();
        mov(program, 3, input);
        append(program, Family::FloatSetRegister, mx::FloatSetRegister{{4}, {3}, {255}, {7, false},
            mx::OrderedComparison::LessThan, false, true, mx::FlushMode::FlushToZero});
        store(program, 4, 28);
        attribute(program, 29, true);
        append(program, Family::FloatSetRegister, mx::FloatSetRegister{{4}, {3}, {255}, {7, false},
            mx::OrderedComparison::GreaterThan, true, true, mx::FlushMode::FlushToZero});
        store(program, 4, 29);
        exit(program);
        const auto result = run(program);
        output(result, gx::OutputKind::Position, 0, negative ? fbits(1.0f) : 0u);
        output(result, gx::OutputKind::Position, 1, negative ? fbits(1.0f) : 0u);
    }
}

ARMIGO_TEST(graphics, nearest_and_zero_multiplication_are_distinct) {
    auto program = one_output();
    attribute(program, 29, true);
    mov(program, 1, 0x3f800800u);
    mov(program, 2, 0x3f801800u);
    append(program, Family::FloatMultiplyRegister, mx::FloatMultiply{{3}, {1}, mx::RegisterOperand{2},
        mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero});
    append(program, Family::FloatMultiplyRegister, mx::FloatMultiply{{1}, {1}, mx::RegisterOperand{2},
        mx::RoundingMode::Zero, mx::FlushMode::FlushToZero});
    store(program, 3, 28);
    store(program, 1, 29);
    exit(program);
    const auto result = run(program);
    output(result, gx::OutputKind::Position, 0, 0x3f802002u);
    output(result, gx::OutputKind::Position, 1, 0x3f802001u);
}

ARMIGO_TEST(graphics, fused_cancellation_and_constant_buffer_reads) {
    auto program = one_output();
    attribute(program, 29, true);
    attribute(program, 30, true);
    mov(program, 1, 0x3f800800u);
    mov(program, 2, 0xbf801000u);
    append(program, Family::FloatFmaConstantRegister, mx::FloatFma{{3}, {1},
        mx::ConstantBufferOperand{6, 8}, {2}, mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero});
    store(program, 3, 28);
    append(program, Family::FloatMultiplyConstant, mx::FloatMultiply{{4}, {1},
        mx::ConstantBufferOperand{6, 8}, mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero});
    append(program, Family::FloatAddRegister, mx::FloatAdd{{4}, {4}, mx::RegisterOperand{2},
        mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero, false});
    store(program, 4, 29);
    mov(program, 5, fbits(1.5f));
    mov(program, 6, fbits(-2.0f));
    append(program, Family::FloatFmaImmediate, mx::FloatFma{{5}, {5},
        mx::ShortFloatImmediate{fbits(2.0f)}, {6}, mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero});
    append(program, Family::FloatAddConstant, mx::FloatAdd{{5}, {5},
        mx::ConstantBufferOperand{6, 0}, mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero, false});
    store(program, 5, 30);
    exit(program);
    const std::array<std::uint32_t, 3> buffer = {fbits(4.0f), 0u, 0x3f800800u};
    gx::Inputs inputs;
    inputs.constant_buffers[6] = buffer;
    const auto result = run(program, inputs);
    output(result, gx::OutputKind::Position, 0, 0x33800000u);
    output(result, gx::OutputKind::Position, 1, 0u);
    output(result, gx::OutputKind::Position, 2, fbits(5.0f));
    inputs.constant_buffers[6] = std::span(buffer).first(2);
    rejected(run(program, inputs), gx::Error::Code::ConstantBounds);
}

ARMIGO_TEST(graphics, immediate_float_add_matches_existing_add_semantics) {
    auto program = one_output();
    attribute(program, 29, true);
    attribute(program, 30, true);
    attribute(program, 31, true);
    attribute(program, 32, true);
    mov(program, 1, fbits(2.5f));
    append(program, Family::FloatAddImmediate, mx::FloatAdd{{2}, {1},
        mx::ShortFloatImmediate{fbits(1.0f)}, mx::RoundingMode::NearestEven,
        mx::FlushMode::FlushToZero, false});
    store(program, 2, 28);
    append(program, Family::FloatAddImmediate, mx::FloatAdd{{2}, {1},
        mx::ShortFloatImmediate{fbits(-1.0f)}, mx::RoundingMode::NearestEven,
        mx::FlushMode::FlushToZero, false});
    store(program, 2, 29);
    append(program, Family::FloatAddImmediate, mx::FloatAdd{{2}, {1},
        mx::ShortFloatImmediate{0x80000000u}, mx::RoundingMode::NearestEven,
        mx::FlushMode::FlushToZero, false});
    store(program, 2, 30);
    append(program, Family::FloatAddImmediate, mx::FloatAdd{{2}, {1},
        mx::ShortFloatImmediate{fbits(1.0f)}, mx::RoundingMode::NearestEven,
        mx::FlushMode::FlushToZero, true});
    store(program, 2, 31);
    mov(program, 3, 0x80000000u);
    append(program, Family::FloatAddImmediate, mx::FloatAdd{{4}, {3},
        mx::ShortFloatImmediate{0u}, mx::RoundingMode::NearestEven,
        mx::FlushMode::FlushToZero, false});
    store(program, 4, 32);
    exit(program);
    const auto result = run(program);
    output(result, gx::OutputKind::Position, 0, fbits(3.5f));
    output(result, gx::OutputKind::Position, 1, fbits(1.5f));
    output(result, gx::OutputKind::Position, 2, fbits(2.5f));
    output(result, gx::OutputKind::Position, 3, fbits(-1.5f));
    output(result, gx::OutputKind::Generic, 0, 0u);
}

ARMIGO_TEST(graphics, forged_immediate_float_add_operands_are_rejected) {
    const auto lower_with = [](mx::FloatSource source,
                               mx::RoundingMode rounding, mx::FlushMode flush) {
        auto program = base(Stage::Vertex);
        append(program, Family::FloatAddImmediate,
               mx::FloatAdd{{1}, {255}, source, rounding, flush, false});
        exit(program);
        return gx::lower(program);
    };
    rejected(lower_with(mx::ShortFloatImmediate{0x3f800001u},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero),
             gx::Error::Code::UnsupportedModifier);
    rejected(lower_with(mx::RegisterOperand{255}, mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero),
             gx::Error::Code::InvalidInstruction);
    rejected(lower_with(mx::ShortFloatImmediate{0x3f800000u},
                        mx::RoundingMode::Zero, mx::FlushMode::FlushToZero),
             gx::Error::Code::UnsupportedModifier);
    rejected(lower_with(mx::ShortFloatImmediate{0x3f800000u},
                        mx::RoundingMode::NearestEven, mx::FlushMode::None),
             gx::Error::Code::UnsupportedModifier);
}

ARMIGO_TEST(graphics, forged_multiply_immediate_source_is_rejected) {
    // The short immediate alternative belongs only to the immediate form.
    for (const auto family :
         {Family::FloatMultiplyRegister, Family::FloatMultiplyConstant}) {
        auto program = base(Stage::Vertex);
        append(program, family,
               mx::FloatMultiply{{1}, {255}, mx::ShortFloatImmediate{0x3f800000u},
                                 mx::RoundingMode::NearestEven,
                                 mx::FlushMode::FlushToZero});
        exit(program);
        rejected(gx::lower(program), gx::Error::Code::InvalidInstruction);
    }
}

ARMIGO_TEST(graphics, immediate_float_multiply_matches_existing_multiply_semantics) {
    auto program = one_output();
    for (unsigned component = 29; component < 32; ++component) {
        attribute(program, component, true);
    }
    for (unsigned component = 32; component < 37; ++component) {
        attribute(program, component, true);
    }
    mov(program, 1, 0x3f800c00u);
    mov(program, 2, 0x80000001u);
    mov(program, 3, 0x7fc12345u);
    mov(program, 4, 0x80000000u);
    mov(program, 5, fbits(3.0f));
    mov(program, 6, fbits(2.0f));
    const auto multiply = [&](std::uint8_t destination, std::uint8_t source_a,
                              std::uint32_t literal_bits,
                              mx::RoundingMode rounding) {
        append(program, Family::FloatMultiplyImmediate,
               mx::FloatMultiply{{destination}, {source_a},
                                 mx::ShortFloatImmediate{literal_bits}, rounding,
                                 mx::FlushMode::FlushToZero});
    };
    multiply(7, 1, 0x3f801000u, mx::RoundingMode::NearestEven);
    store(program, 7, 28);
    multiply(7, 1, 0x3f801000u, mx::RoundingMode::Zero);
    store(program, 7, 29);
    multiply(7, 2, 0x40000000u, mx::RoundingMode::NearestEven);
    store(program, 7, 30);
    multiply(7, 3, 0x3f800000u, mx::RoundingMode::NearestEven);
    store(program, 7, 31);
    multiply(7, 4, 0x00000000u, mx::RoundingMode::NearestEven);
    store(program, 7, 32);
    multiply(7, 5, 0x00001000u, mx::RoundingMode::NearestEven);
    store(program, 7, 33);
    multiply(7, 6, 0x7f800000u, mx::RoundingMode::NearestEven);
    store(program, 7, 34);
    multiply(7, 255, 0x7f800000u, mx::RoundingMode::NearestEven);
    store(program, 7, 35);
    multiply(255, 6, 0x40000000u, mx::RoundingMode::NearestEven);
    multiply(8, 255, 0x40000000u, mx::RoundingMode::NearestEven);
    store(program, 8, 36);
    exit(program);
    const auto result = run(program);
    output(result, gx::OutputKind::Position, 0, 0x3f801c02u);
    output(result, gx::OutputKind::Position, 1, 0x3f801c01u);
    output(result, gx::OutputKind::Position, 2, 0x80000000u);
    output(result, gx::OutputKind::Position, 3, 0x7fc00000u);
    output(result, gx::OutputKind::Generic, 0, 0x80000000u);
    output(result, gx::OutputKind::Generic, 1, 0u);
    output(result, gx::OutputKind::Generic, 2, 0x7f800000u);
    output(result, gx::OutputKind::Generic, 3, 0x7fc00000u);
    output(result, gx::OutputKind::Generic, 4, 0u);
    const auto lowered = gx::lower(program);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (const auto* ir = std::get_if<gx::Program>(&lowered)) {
        unsigned nearest = 0;
        unsigned zero = 0;
        for (const auto& node : ir->nodes()) {
            const auto* arithmetic = std::get_if<gx::FloatArithmetic>(&node.operation);
            if (arithmetic == nullptr) continue;
            if (arithmetic->operation == gx::Arithmetic::MultiplyNearest) {
                ++nearest;
            } else if (arithmetic->operation == gx::Arithmetic::MultiplyZero) {
                ++zero;
            } else {
                continue;
            }
            const auto* literal = std::get_if<gx::Literal>(
                &ir->nodes()[arithmetic->rhs.index].operation);
            ARMIGO_CHECK(literal != nullptr);
            if (literal != nullptr) ARMIGO_CHECK_EQ(literal->bits & 4095u, 0u);
        }
        ARMIGO_CHECK_EQ(nearest, 9u);
        ARMIGO_CHECK_EQ(zero, 1u);
    }
}

ARMIGO_TEST(graphics, forged_immediate_float_multiply_operands_are_rejected) {
    const auto lower_with = [](mx::FloatSource source, mx::RoundingMode rounding,
                               mx::FlushMode flush,
                               mx::RegisterOperand destination) {
        auto program = base(Stage::Vertex);
        append(program, Family::FloatMultiplyImmediate,
               mx::FloatMultiply{destination, {255}, source, rounding, flush});
        exit(program);
        return gx::lower(program);
    };
    // The immediate encoding carries twelve implicit low zero bits.
    rejected(lower_with(mx::ShortFloatImmediate{0x3f800001u},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero, {1}),
             gx::Error::Code::UnsupportedModifier);
    // Register and constant-buffer alternatives belong to the other forms.
    rejected(lower_with(mx::RegisterOperand{255}, mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero, {1}),
             gx::Error::Code::InvalidInstruction);
    rejected(lower_with(mx::ConstantBufferOperand{0, 0},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero, {1}),
             gx::Error::Code::InvalidInstruction);
    for (const auto rounding : {mx::RoundingMode::NegativeInfinity,
                                mx::RoundingMode::PositiveInfinity}) {
        rejected(lower_with(mx::ShortFloatImmediate{0x3f800000u}, rounding,
                            mx::FlushMode::FlushToZero, {1}),
                 gx::Error::Code::UnsupportedModifier);
    }
    for (const auto flush :
         {mx::FlushMode::None, mx::FlushMode::FlushMultiplyZero}) {
        rejected(lower_with(mx::ShortFloatImmediate{0x3f800000u},
                            mx::RoundingMode::NearestEven, flush, {1}),
                 gx::Error::Code::UnsupportedModifier);
    }
    // Accepted controls: both rounding modes, an exact literal, and RZ as the
    // destination register.
    for (const auto rounding :
         {mx::RoundingMode::NearestEven, mx::RoundingMode::Zero}) {
        ARMIGO_CHECK(std::holds_alternative<gx::Program>(lower_with(
            mx::ShortFloatImmediate{0x3f800000u}, rounding,
            mx::FlushMode::FlushToZero, {1})));
        ARMIGO_CHECK(std::holds_alternative<gx::Program>(lower_with(
            mx::ShortFloatImmediate{0x3f800000u}, rounding,
            mx::FlushMode::FlushToZero, {255})));
    }
}

ARMIGO_TEST(graphics, arithmetic_flushes_signed_subnormals_and_canonicalizes_nan) {
    auto program = one_output();
    attribute(program, 29, true);
    mov(program, 1, 0x80000001u);
    mov(program, 2, fbits(2.0f));
    append(program, Family::FloatMultiplyRegister, mx::FloatMultiply{{1}, {1}, mx::RegisterOperand{2},
        mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero});
    store(program, 1, 28);
    mov(program, 1, 0x7fc12345u);
    append(program, Family::FloatAddRegister, mx::FloatAdd{{1}, {1}, mx::RegisterOperand{255},
        mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero, false});
    store(program, 1, 29);
    exit(program);
    const auto result = run(program);
    output(result, gx::OutputKind::Position, 0, 0x80000000u);
    output(result, gx::OutputKind::Position, 1, 0x7fc00000u);
}

ARMIGO_TEST(graphics, reciprocal_preserves_gradual_underflow) {
    auto program = one_output();
    mov(program, 1, 0x7f000000u);
    append(program, Family::MultiFunction, mx::Reciprocal{{1}, {1}});
    store(program, 1, 28);
    exit(program);
    output(run(program), gx::OutputKind::Position, 0, 0x00400000u);
}

ARMIGO_TEST(graphics, perspective_ipa_then_explicit_reciprocal_and_zero_multiply) {
    auto program = base(Stage::Fragment);
    attribute(program, 31);
    attribute(program, 32, false, 2);
    put(program, 0x48, 1);
    ipa(program, 1, 32);
    ipa(program, 2, 31);
    append(program, Family::MultiFunction, mx::Reciprocal{{3}, {2}});
    append(program, Family::FloatMultiplyRegister, mx::FloatMultiply{{0}, {1}, mx::RegisterOperand{3},
        mx::RoundingMode::Zero, mx::FlushMode::FlushToZero});
    exit(program);
    const std::array attrs = {gx::AttributeInput{31, fbits(0.5f)}, gx::AttributeInput{32, fbits(0.75f)}};
    gx::Inputs inputs;
    inputs.attributes = attrs;
    output(run(program, inputs), gx::OutputKind::Color, 0, fbits(0.75f));
    inputs.attributes = std::span(attrs).last(1);
    rejected(run(program, inputs), gx::Error::Code::MissingAttribute);
    const auto lowered = gx::lower(program);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (const auto* ir = std::get_if<gx::Program>(&lowered)) {
        const auto& interpolation = std::get<gx::Interpolate>(ir->nodes()[3].operation);
        ARMIGO_CHECK(interpolation.mode == gx::Interpolation::Perspective);
        ARMIGO_CHECK(interpolation.position_w.has_value());
        ARMIGO_CHECK_EQ(interpolation.encoding_control, 0x7ffu);
    }
}

ARMIGO_TEST(graphics, constant_and_screen_linear_ipa_preserve_input_bits) {
    for (const auto mode : {1u, 3u}) {
        auto program = base(Stage::Fragment);
        attribute(program, 37, false, mode);
        put(program, 0x48, 1);
        ipa(program, 0, 37);
        exit(program);
        const std::array attrs = {gx::AttributeInput{37, 0x80000001u}};
        gx::Inputs inputs;
        inputs.attributes = attrs;
        output(run(program, inputs), gx::OutputKind::Color, 0, 0x80000001u);
    }
}

ARMIGO_TEST(graphics, texture_tuple_reads_coordinates_before_overlapping_writes) {
    auto program = base(Stage::Fragment);
    put(program, 0x48, 15);
    mov(program, 0, fbits(0.25f));
    mov(program, 1, fbits(0.75f));
    append(program, Family::TextureSample, mx::TextureSample2D{{0}, {2}, {0}, {1}, 0x34, true});
    exit(program);
    unsigned calls = 0;
    gx::Inputs inputs;
    inputs.sample_2d_implicit_lod = [&](const gx::TextureRequest& request,
                                       std::array<std::uint32_t, 4>& result) {
        ++calls;
        ARMIGO_CHECK_EQ(request.handle_byte_offset, 0x34u);
        ARMIGO_CHECK_EQ(request.x_bits, fbits(0.25f));
        ARMIGO_CHECK_EQ(request.y_bits, fbits(0.75f));
        ARMIGO_CHECK(request.no_dependency);
        result = {fbits(2.0f), fbits(3.0f), fbits(4.0f), fbits(5.0f)};
        return true;
    };
    const auto result = run(program, inputs);
    ARMIGO_CHECK_EQ(calls, 1u);
    for (unsigned lane = 0; lane < 4; ++lane) {
        output(result, gx::OutputKind::Color, lane, fbits(static_cast<float>(lane + 2)));
    }
    rejected(run(program), gx::Error::Code::MissingSampler);
    inputs.sample_2d_implicit_lod = [](const auto&, auto&) { return false; };
    rejected(run(program, inputs), gx::Error::Code::SamplingFailed);
    inputs.sample_2d_implicit_lod = [](const auto&, auto&) -> bool { throw std::runtime_error("Synthetic failure"); };
    rejected(run(program, inputs), gx::Error::Code::SamplingFailed);
}

ARMIGO_TEST(graphics, fragment_export_masks_preserve_disabled_lane_register_slots) {
    auto program = base(Stage::Fragment);
    put(program, 0x48, 0x25);
    mov(program, 0, 101);
    mov(program, 2, 102);
    mov(program, 5, 103);
    exit(program);
    const auto result = run(program);
    output(result, gx::OutputKind::Color, 0, 101);
    output(result, gx::OutputKind::Color, 2, 102);
    output(result, gx::OutputKind::Color, 5, 103);
    if (const auto* evaluated = std::get_if<gx::Evaluation>(&result)) {
        ARMIGO_CHECK_EQ(evaluated->outputs.size(), 3u);
    }
}

ARMIGO_TEST(graphics, malformed_metadata_and_missing_values_fail_explicitly) {
    auto program = one_output();
    store(program, 3, 28);
    exit(program);
    rejected(gx::lower(program), gx::Error::Code::UninitializedRegister);
    program = one_output();
    exit(program);
    rejected(gx::lower(program), gx::Error::Code::MissingOutput);
    program = base(Stage::Vertex);
    mov(program, 1, 0);
    rejected(gx::lower(program), gx::Error::Code::MissingExit);
    exit(program);
    rejected(gx::lower(program, {1}), gx::Error::Code::Limit);
    auto invalid = program;
    invalid.instructions[0].predicate.inverted = true;
    rejected(gx::lower(invalid), gx::Error::Code::UnsupportedModifier);
    invalid = program;
    invalid.instructions[0].family = Family::FloatAddRegister;
    rejected(gx::lower(invalid), gx::Error::Code::InvalidInstruction);
    invalid = program;
    invalid.instructions[0].pc = 24;
    rejected(gx::lower(invalid), gx::Error::Code::InvalidInstruction);
    invalid = program;
    mov(invalid, 2, 0);
    rejected(gx::lower(invalid), gx::Error::Code::InvalidInstruction);
    invalid = program;
    put(invalid, 0, 5u << 10);
    rejected(gx::lower(invalid), gx::Error::Code::HeaderMismatch);
    invalid = program;
    put(invalid, 4, 4);
    rejected(gx::lower(invalid), gx::Error::Code::UnsupportedInterface);
    invalid = base(Stage::Fragment);
    put(invalid, 0x4c, 1);
    exit(invalid);
    rejected(gx::lower(invalid), gx::Error::Code::UnsupportedInterface);
}

ARMIGO_TEST(graphics, forged_modifiers_and_resource_operands_are_rejected) {
    for (const auto mode : {mx::RoundingMode::NegativeInfinity, mx::RoundingMode::PositiveInfinity}) {
        auto program = base(Stage::Vertex);
        append(program, Family::FloatMultiplyRegister, mx::FloatMultiply{{1}, {255}, mx::RegisterOperand{255},
            mode, mx::FlushMode::FlushToZero});
        exit(program);
        rejected(gx::lower(program), gx::Error::Code::UnsupportedModifier);
    }
    for (const auto operand : {mx::ConstantBufferOperand{18, 0}, mx::ConstantBufferOperand{2, 3}}) {
        auto program = base(Stage::Vertex);
        append(program, Family::FloatAddConstant, mx::FloatAdd{{1}, {255}, operand,
            mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero, false});
        exit(program);
        rejected(gx::lower(program), gx::Error::Code::UnsupportedModifier);
    }
    auto program = base(Stage::Fragment);
    attribute(program, 32);
    ipa(program, 0, 32);
    std::get<mx::InterpolateAttribute>(program.instructions[0].payload).encoding_control = 3;
    exit(program);
    rejected(gx::lower(program), gx::Error::Code::UnsupportedModifier);
    program = base(Stage::Fragment);
    append(program, Family::TextureSample, mx::TextureSample2D{{254}, {2}, {255}, {255}, 0, false});
    exit(program);
    rejected(gx::lower(program), gx::Error::Code::UnsupportedModifier);
}

ARMIGO_TEST(graphics, attribute_masks_bounds_and_input_uniqueness) {
    auto program = one_output();
    load(program, 1, 33);
    store(program, 1, 28);
    exit(program);
    rejected(gx::lower(program), gx::Error::Code::UndeclaredAttribute);
    attribute(program, 33);
    const std::array duplicate = {gx::AttributeInput{33, 4}, gx::AttributeInput{33, 5}};
    gx::Inputs inputs;
    inputs.attributes = duplicate;
    rejected(run(program, inputs), gx::Error::Code::DuplicateAttribute);
    inputs.attributes = {};
    rejected(run(program, inputs), gx::Error::Code::MissingAttribute);
    attribute(program, 28);
    std::get<mx::AttributeAccess>(program.instructions[0].payload).byte_offset = 28 * 4;
    const std::array position_input = {gx::AttributeInput{28, fbits(8.0f)}};
    inputs.attributes = position_input;
    output(run(program, inputs), gx::OutputKind::Position, 0, fbits(8.0f));
}

ARMIGO_TEST(graphics, floating_environment_is_restored_on_success_and_failure) {
    std::fenv_t original;
    ARMIGO_CHECK_EQ(std::fegetenv(&original), 0);
    ARMIGO_CHECK_EQ(std::fesetround(FE_DOWNWARD), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID);
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    auto program = one_output();
    mov(program, 1, fbits(3.0f));
    append(program, Family::MultiFunction, mx::Reciprocal{{1}, {1}});
    store(program, 1, 28);
    exit(program);
    output(run(program), gx::OutputKind::Position, 0, 0x3eaaaaabu);
    ARMIGO_CHECK_EQ(std::fegetround(), FE_DOWNWARD);
    ARMIGO_CHECK_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    program = one_output();
    attribute(program, 32);
    load(program, 1, 32);
    store(program, 1, 28);
    exit(program);
    rejected(run(program), gx::Error::Code::MissingAttribute);
    ARMIGO_CHECK_EQ(std::fegetround(), FE_DOWNWARD);
    ARMIGO_CHECK_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    ARMIGO_CHECK_EQ(std::fesetenv(&original), 0);
}

ARMIGO_TEST(graphics, register_move_copies_every_bit_pattern) {
    const std::array cases = {
        0x3f800000u, 0x80000000u, 0x7fc12345u, 0x7f800000u, 0xff800000u,
        0xffffffffu, 0xdeadbeefu, 0x00000001u, 0x7f7fffffu,
    };
    for (const std::uint32_t bits : cases) {
        auto program = one_output();
        mov(program, 1, bits);
        reg_move(program, 2, 1);
        store(program, 2, 28);
        exit(program);
        output(run(program), gx::OutputKind::Position, 0, bits);
    }
}

ARMIGO_TEST(graphics, register_move_zero_and_overlap_are_safe) {
    auto zero_source = one_output();
    reg_move(zero_source, 2, 255);
    store(zero_source, 2, 28);
    exit(zero_source);
    output(run(zero_source), gx::OutputKind::Position, 0, 0u);

    auto zero_destination = one_output();
    mov(zero_destination, 1, 0x7fc12345u);
    reg_move(zero_destination, 255, 1);
    store(zero_destination, 1, 28);
    exit(zero_destination);
    output(run(zero_destination), gx::OutputKind::Position, 0, 0x7fc12345u);

    auto self_copy = one_output();
    mov(self_copy, 1, 0x80000001u);
    reg_move(self_copy, 1, 1);
    store(self_copy, 1, 28);
    exit(self_copy);
    output(run(self_copy), gx::OutputKind::Position, 0, 0x80000001u);

    auto aliased = one_output();
    mov(aliased, 1, 0x3f800000u);
    mov(aliased, 2, 0xbf800000u);
    reg_move(aliased, 2, 1);
    reg_move(aliased, 1, 2);
    store(aliased, 1, 28);
    exit(aliased);
    output(run(aliased), gx::OutputKind::Position, 0, 0x3f800000u);
}

ARMIGO_TEST(graphics, register_move_forged_forms_and_undefined_source_fail) {
    auto undefined = base(Stage::Vertex);
    reg_move(undefined, 1, 2);
    exit(undefined);
    rejected(gx::lower(undefined), gx::Error::Code::UninitializedRegister);

    for (const unsigned write_mask : {0u, 2u, 3u, 7u, 8u, 14u}) {
        auto program = base(Stage::Vertex);
        mov(program, 2, 0x3f800000u);
        reg_move(program, 1, 2, static_cast<std::uint8_t>(write_mask));
        exit(program);
        rejected(gx::lower(program), gx::Error::Code::UnsupportedModifier);
    }

    auto forged = base(Stage::Vertex);
    append(forged, Family::MovRegister, mx::Mov32Immediate{{1}, 15, 0});
    exit(forged);
    rejected(gx::lower(forged), gx::Error::Code::InvalidInstruction);

    auto single_lane = one_output();
    mov(single_lane, 1, 0x7fc12345u);
    reg_move(single_lane, 2, 1, 1);
    store(single_lane, 2, 28);
    exit(single_lane);
    output(run(single_lane), gx::OutputKind::Position, 0, 0x7fc12345u);

    auto aliased = one_output();
    mov(aliased, 1, 0x7fc12345u);
    reg_move(aliased, 2, 1);
    store(aliased, 2, 28);
    exit(aliased);
    const auto lowered = gx::lower(aliased);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (const auto* ir = std::get_if<gx::Program>(&lowered)) {
        ARMIGO_CHECK_EQ(ir->nodes().size(), 2u);
        ARMIGO_CHECK_EQ(ir->exports().size(), 1u);
        ARMIGO_CHECK_EQ(ir->exports()[0].value.index, 1u);
    }
}

ARMIGO_TEST(graphics, lowerer_registry_admits_only_move_families_today) {
    const std::span<const gx::detail::LowerEntry> lowerers =
        gx::detail::registered_lowerers();
    ARMIGO_CHECK(!lowerers.empty());
    const gx::detail::LowerEntry* move_entry = nullptr;
    for (const gx::detail::LowerEntry& registered : lowerers) {
        if (registered.family == Family::MovRegister) {
            move_entry = &registered;
        }
    }
    ARMIGO_CHECK(move_entry != nullptr);
    if (move_entry == nullptr) {
        return;
    }
    ARMIGO_CHECK(move_entry->lower != nullptr);
    for (std::size_t i = 0; i < lowerers.size(); ++i) {
        for (std::size_t j = i + 1; j < lowerers.size(); ++j) {
            ARMIGO_CHECK(lowerers[i].family != lowerers[j].family);
        }
    }
    // Families with legacy lowering cases are never registered: the registry
    // only extends dispatch, it cannot shadow the legacy switch.
    ARMIGO_CHECK(gx::detail::find_registered_lowerer(Family::Mov32Immediate) ==
                 nullptr);
    ARMIGO_CHECK(gx::detail::find_registered_lowerer(Family::Exit) == nullptr);
    ARMIGO_CHECK(gx::detail::find_registered_lowerer(
                     Family::FloatMultiplyImmediate) == nullptr);
    // The register-move pilot is reachable in the lowerer registry.
    const auto* entry = gx::detail::find_registered_lowerer(Family::MovRegister);
    ARMIGO_CHECK(entry != nullptr);
}

ARMIGO_TEST(graphics, lowerer_registry_validator_rejects_bad_tables) {
    using LowerEntry = gx::detail::LowerEntry;
    const auto valid = [](std::span<const LowerEntry> table) {
        return shader::detail::family_registry_valid(
            table, [](const LowerEntry& entry) {
                return entry.lower != nullptr &&
                       entry.family != Family::Unknown;
            });
    };
    // The production table satisfies the shared validator.
    ARMIGO_CHECK(valid(gx::detail::registered_lowerers()));

    const std::array<LowerEntry, 2> distinct{{
        {Family::MovRegister, gx::detail::lower_mov_register},
        {Family::FloatAddRegister, gx::detail::lower_mov_register},
    }};
    ARMIGO_CHECK(valid(distinct));

    const std::array<LowerEntry, 2> duplicate_family{{
        {Family::MovRegister, gx::detail::lower_mov_register},
        {Family::MovRegister, gx::detail::lower_mov_register},
    }};
    ARMIGO_CHECK(!valid(duplicate_family));

    const std::array<LowerEntry, 2> null_lowerer{{
        {Family::MovRegister, nullptr},
        {Family::FloatAddRegister, gx::detail::lower_mov_register},
    }};
    ARMIGO_CHECK(!valid(null_lowerer));
}


ARMIGO_TEST(graphics, ipa_register_multiply_reads_old_destination_and_preserves_subnormals) {
    for (const auto mode : {1u, 3u}) {
        auto program = base(Stage::Fragment);
        attribute(program, 32, false, mode);
        put(program, 0x48, 1);
        mov(program, 0, fbits(0.5f));
        append(program, Family::InterpolateAttribute,
               mx::InterpolateAttribute{{0}, {255}, {0}, 32, 0x7ff, true});
        exit(program);
        const std::array attrs = {gx::AttributeInput{32, 2u}};
        gx::Inputs inputs; inputs.attributes = attrs;
        output(run(program, inputs), gx::OutputKind::Color, 0, 1u);
    }
}

ARMIGO_TEST(graphics, ipa_multiply_rounds_perspective_before_register_product) {
    auto program = base(Stage::Fragment);
    attribute(program, 31); attribute(program, 32, false, 2);
    put(program, 0x48, 15);
    for (unsigned reg = 1; reg <= 3; ++reg) mov(program, reg, 0);
    mov(program, 8, fbits(2.0f));
    append(program, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{0}, {255}, {8}, 32, 0x7ff, true});
    exit(program);
    const std::array attrs = {gx::AttributeInput{31, fbits(0.5f)}, gx::AttributeInput{32, 1u}};
    gx::Inputs inputs; inputs.attributes = attrs;
    output(run(program, inputs), gx::OutputKind::Color, 0, 0u);
    auto lowered = gx::lower(program);
    const auto* ir = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(ir != nullptr);
    if (!ir) return;
    graphics_msl::EmitOptions options;
    options.fragment_position = graphics_msl::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    const auto emitted = graphics_msl::emit(*ir, options);
    const auto* stage = std::get_if<graphics_msl::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    if (stage) ARMIGO_CHECK(stage->source.find("armigo_multiply_rne(armigo_multiply_rne(") != std::string::npos);
}

ARMIGO_TEST(graphics, ipa_multiply_zero_register_and_undefined_operand) {
    auto program = base(Stage::Fragment);
    attribute(program, 32, false, 3); put(program, 0x48, 1);
    append(program, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{0}, {255}, {255}, 32, 0x7ff, true});
    exit(program);
    const std::array attrs = {gx::AttributeInput{32, fbits(-2.0f)}};
    gx::Inputs inputs; inputs.attributes = attrs;
    output(run(program, inputs), gx::OutputKind::Color, 0, 0x80000000u);
    auto& value = std::get<mx::InterpolateAttribute>(program.instructions[0].payload);
    value.multiplier = {11};
    rejected(gx::lower(program), gx::Error::Code::UninitializedRegister);
    value.multiply = false;
    rejected(gx::lower(program), gx::Error::Code::UnsupportedModifier);
}

ARMIGO_TEST(graphics, ipa_multiply_rejects_forged_future_dependency) {
    auto program = base(Stage::Fragment);
    attribute(program, 32, false, 3); put(program, 0x48, 1);
    append(program, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{0}, {255}, {255}, 32, 0x7ff, true});
    exit(program);
    auto lowered = gx::lower(program);
    auto* ir = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(ir != nullptr); if (!ir) return;
    auto* nodes = const_cast<gx::Node*>(ir->nodes().data());
    for (std::size_t i = 0; i < ir->nodes().size(); ++i) {
        if (auto* value = std::get_if<gx::Interpolate>(&nodes[i].operation))
            value->multiplier = gx::ScalarId{0xffffffffu};
    }
    const std::array attrs = {gx::AttributeInput{32, fbits(1.0f)}};
    gx::Inputs inputs; inputs.attributes = attrs;
    rejected(gx::evaluate(*ir, inputs), gx::Error::Code::InvalidInstruction);
    ARMIGO_CHECK(std::holds_alternative<graphics_msl::EmitError>(graphics_msl::emit(*ir)));
}

}  // namespace armigo::shader::test
