// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic lowering tests for the saturating register-FMUL family. Every
// expected product and saturation word below is authored from
// nearest-even flush-to-zero multiplication, then a distinct saturation
// step; no oracle recomputes any result word.

#include "test_support.h"

#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/maxwell.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {

namespace mx = maxwell;
namespace gx = graphics;
namespace gm = graphics_msl;
using Family = mx::OpcodeFamily;
using Stage = shader_input::Stage;
using Value = mx::FloatMultiply;

// One synthetic saturating register-FMUL word in the admitted profile
// (mask 0xfffc, value 0x5c6c): nearest-even rounding, FTZ flush, no scale,
// condition code, or negation, PT instruction predicate, saturation set.
std::uint64_t word(unsigned destination = 3, unsigned source_a = 1,
                   unsigned source_b = 2) {
    return (std::uint64_t{0x5c6c} << 48) | (std::uint64_t{7} << 16) |
           destination | (std::uint64_t{source_a} << 8) |
           (std::uint64_t{source_b} << 20) | (std::uint64_t{1} << 44);
}

void put(std::vector<std::uint8_t>& bytes, unsigned at, std::uint64_t value,
         unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        bytes[at + i] = std::uint8_t(value >> (8 * i));
    }
}

// Full container decode of one instruction at pc 8. The admitted profile is
// exhausted by the decoder tests; this keeps the decode-lower identity.
mx::DecodeResult decode(std::uint64_t instruction) {
    std::vector<std::uint8_t> control(0x718), code(0xa0);
    put(control, 0, 0x98761234, 4);
    put(control, 4, 1, 4);
    put(control, 8, 14, 4);
    put(control, 0x20, control.size(), 4);
    put(control, 0x700, code.size(), 4);
    put(control, 0x704, code.size(), 4);
    put(control, 0x708, 0x30, 4);
    put(control, 0x714, 0, 4);
    put(code, 0, 0x12345678, 4);
    put(code, 0x30,
        shader_input::sph_stage_type_for(Stage::Vertex) << 10, 4);
    put(code, 0x88, instruction, 8);
    put(code, 0x90,
        (std::uint64_t{0xe300} << 48) | (std::uint64_t{7} << 16) | 15, 8);
    return mx::decode({control.data(), control.size()},
                      {code.data(), code.size()});
}

// The payload exactly as the admitted decode profile produces it.
Value payload(unsigned destination = 3, unsigned source_a = 1,
              unsigned source_b = 2) {
    return Value{{std::uint8_t(destination)}, {std::uint8_t(source_a)},
                 mx::RegisterOperand{std::uint8_t(source_b)},
                 mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero};
}

// Straight-line program: MOV32I both multiplicands, the saturating multiply,
// then an unconditional EXIT. Vertex programs store the destination to the
// generic attribute lanes; fragment programs expose register zero as the
// color output. An optional second saturating multiply over the declared
// color registers extends the fragment program for node-budget cases.
mx::DecodedProgram program(Value value, std::uint32_t a, std::uint32_t b,
                           Stage stage = Stage::Vertex, bool init_b = true,
                           std::optional<Value> second_multiply = std::nullopt) {
    const auto destination = value.destination;
    const auto* multiplier_operand =
        std::get_if<mx::RegisterOperand>(&value.source_b);
    const auto multiplier =
        multiplier_operand != nullptr ? multiplier_operand->index : 0u;
    mx::DecodedProgram p;
    p.stage = stage;
    const auto header = shader_input::sph_stage_type_for(stage) << 10;
    for (unsigned i = 0; i < 4; ++i) {
        p.sph[i] = std::byte(header >> (i * 8));
    }
    if (stage == Stage::Vertex) {
        p.sph[0x35] = std::byte{0xf0};
    } else {
        p.sph[0x48] = std::byte{0xf};
    }
    const auto append = [&p](Family family, mx::InstructionPayload payload) {
        const auto i = p.instructions.size();
        p.instructions.push_back(
            {std::uint32_t((i / 3) * 32 + (i % 3 + 1) * 8), 0, family,
             {7, false}, std::move(payload)});
    };
    const auto mov = [&append](unsigned reg, std::uint32_t bits) {
        append(Family::Mov32Immediate,
               mx::Mov32Immediate{{std::uint8_t(reg)}, 15, bits});
    };
    mov(value.source_a.index, a);
    if (init_b) {
        mov(multiplier, b);
    }
    if (stage == Stage::Fragment) {
        // Declared color lanes read every output register.
        mov(0, 0);
        mov(3, 0x3f800000u);
    }
    append(Family::FloatMultiplySaturateRegister, std::move(value));
    if (second_multiply) {
        append(Family::FloatMultiplySaturateRegister,
               std::move(*second_multiply));
    }
    if (stage == Stage::Vertex) {
        for (unsigned c = 28; c < 32; ++c) {
            append(Family::AttributeStore,
                   mx::AttributeAccess{
                       {std::uint8_t(destination.is_zero()
                                         ? 1 : destination.index)},
                       {255}, {255}, std::uint16_t(c * 4), 0, false});
        }
    }
    append(Family::Exit, mx::Exit{15});
    return p;
}

std::uint32_t evaluate(const mx::DecodedProgram& p) {
    const auto lowered = gx::lower(p);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (!std::holds_alternative<gx::Program>(lowered)) {
        const auto& failure = std::get<gx::Error>(lowered);
        record_failure(__FILE__, __LINE__,
                       std::string("lower failed: ") + failure.message() +
                           " pc=" + std::to_string(failure.pc) +
                           " detail=" + std::to_string(failure.detail));
        return 0;
    }
    const auto result = gx::evaluate(std::get<gx::Program>(lowered), {});
    const auto* evaluated = std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(evaluated != nullptr);
    if (evaluated == nullptr) {
        return 0;
    }
    for (const auto& output : evaluated->outputs) {
        if (output.component == 0) {
            return output.bits;
        }
    }
    ARMIGO_CHECK(false);
    return 0;
}

void error(const mx::DecodedProgram& p, gx::Error::Code expected,
           gx::Limits limits = {}) {
    const auto lowered = gx::lower(p, limits);
    const auto* e = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(e != nullptr);
    if (e != nullptr) {
        ARMIGO_CHECK(e->code == expected);
    }
}

}  // namespace

// Decode, lower, and evaluate in both stages: the product saturates only
// after the multiply, so 2*3 lands on one, not six.
ARMIGO_TEST(fmul_saturate_register_lower, lowers_both_stages_to_one_saturate) {
    for (const auto stage : {Stage::Vertex, Stage::Fragment}) {
        const unsigned destination = stage == Stage::Vertex ? 3u : 0u;
        const auto decoded = decode(word(destination, 1, 2));
        const auto* p = std::get_if<mx::DecodedProgram>(&decoded);
        ARMIGO_CHECK(p != nullptr);
        if (p == nullptr) {
            continue;
        }
        const auto& instruction = p->instructions.front();
        ARMIGO_CHECK(instruction.family ==
                     Family::FloatMultiplySaturateRegister);
        const auto* value = std::get_if<Value>(&instruction.payload);
        ARMIGO_CHECK(value != nullptr);
        if (value == nullptr) {
            continue;
        }
        ARMIGO_CHECK_EQ(
            evaluate(program(*value, 0x40000000u, 0x40400000u, stage)),
            0x3f800000u);
    }
}

// Authored bit edges: products inside the unit interval keep their bits,
// values on both sides of one map onto the interval, and signed zeros,
// infinities, NaN products, and the FTZ underflow window all normalize per
// the documented bounded policy.
ARMIGO_TEST(fmul_saturate_register_lower, applies_authored_product_edges) {
    struct Case {
        std::uint32_t a, b, expected;
    };
    const std::array<Case, 14> cases{{
        {0x3f000000u, 0x3e800000u, 0x3e000000u},  // 0.5*0.25 stays 0.125
        {0x40000000u, 0x3e800000u, 0x3f000000u},  // 2*0.25 stays 0.5, so saturation follows the product: clamping the operands first would give 1*0.25 = 0.25
        {0x3f000000u, 0x3f800000u, 0x3f000000u},  // 0.5*1 stays 0.5
        {0x3f800000u, 0x3f800000u, 0x3f800000u},  // exactly one stays one
        {0x3f800001u, 0x3f800000u, 0x3f800000u},  // smallest word above one
        {0x40000000u, 0x40000000u, 0x3f800000u},  // 4 clamps to one
        {0x40800000u, 0x40000000u, 0x3f800000u},  // 8 clamps to one
        {0xbf800000u, 0x40000000u, 0x00000000u},  // negative product to +0
        {0x00000000u, 0x40000000u, 0x00000000u},  // +0 product stays +0
        {0x80000000u, 0x3f800000u, 0x00000000u},  // -0 product maps to +0
        {0x7f800000u, 0x40000000u, 0x3f800000u},  // +infinity maps to one
        {0xff800000u, 0x3f800000u, 0x00000000u},  // -infinity maps to +0
        {0x7fc00000u, 0x00000000u, 0x00000000u},  // NaN*0 to +0
        {0x00800000u, 0x3f000000u, 0x00000000u},  // FTZ subnormal to +0
    }};
    for (const auto& c : cases) {
        ARMIGO_CHECK_EQ(evaluate(program(payload(), c.a, c.b)), c.expected);
    }
}

// The multiplicand reads happen before the aliased destination write, and a
// register-zero destination discards the result without clobbering reads.
ARMIGO_TEST(fmul_saturate_register_lower, aliases_destination_and_discards_rz) {
    ARMIGO_CHECK_EQ(evaluate(program(payload(1, 1, 2), 0x40000000u,
                                     0x40400000u)),
                    0x3f800000u);  // destination == source A
    ARMIGO_CHECK_EQ(evaluate(program(payload(2, 1, 2), 0x40000000u,
                                     0x40400000u)),
                    0x3f800000u);  // destination == source B
    ARMIGO_CHECK_EQ(evaluate(program(payload(1, 1, 1), 0x40000000u,
                                     0x40000000u)),
                    0x3f800000u);  // fully aliased multiplicands
    ARMIGO_CHECK_EQ(evaluate(program(payload(255, 1, 2), 0x40000000u,
                                     0x40400000u)),
                    0x40000000u);  // RZ discard leaves source A intact
}

// The lowered node order is MultiplyNearest, then the saturate of that
// product, with no addend and no negation on the multiply.
ARMIGO_TEST(fmul_saturate_register_lower, multiply_node_precedes_saturate) {
    const auto lowered = gx::lower(program(payload(), 0x40000000u,
                                          0x40400000u));
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (!std::holds_alternative<gx::Program>(lowered)) {
        return;
    }
    const auto nodes = std::get<gx::Program>(lowered).nodes();
    std::optional<std::size_t> multiply, saturate;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (const auto* arithmetic =
                std::get_if<gx::FloatArithmetic>(&nodes[i].operation)) {
            if (arithmetic->operation == gx::Arithmetic::MultiplyNearest) {
                ARMIGO_CHECK(!arithmetic->addend.has_value());
                ARMIGO_CHECK(!arithmetic->negate_lhs);
                multiply = i;
            }
        } else if (std::holds_alternative<gx::Saturate>(nodes[i].operation)) {
            saturate = i;
        }
    }
    ARMIGO_CHECK(multiply.has_value() && saturate.has_value());
    if (!multiply.has_value() || !saturate.has_value()) {
        return;
    }
    ARMIGO_CHECK(*multiply < *saturate);
    const auto& saturation =
        std::get<gx::Saturate>(nodes[*saturate].operation);
    ARMIGO_CHECK_EQ(saturation.operand.index,
                    static_cast<std::uint32_t>(*multiply));
}

// Forged payloads and modifiers keep the established structured errors, and
// the family consumes no node-budget headroom beyond the existing limit.
ARMIGO_TEST(fmul_saturate_register_lower, forged_payloads_and_limits_fail) {
    auto forged = program(payload(), 0x40000000u, 0x40400000u);
    for (auto& instruction : forged.instructions) {
        if (instruction.family == Family::FloatMultiplySaturateRegister) {
            instruction.payload = mx::MovConstant{{3}, {5, 0}, 15};
        }
    }
    error(forged, gx::Error::Code::InvalidInstruction);

    auto constant_multiplier = payload();
    constant_multiplier.source_b = mx::ConstantBufferOperand{5, 0};
    error(program(constant_multiplier, 0x40000000u, 0x40400000u),
          gx::Error::Code::InvalidInstruction);

    auto rounding = payload();
    rounding.rounding = mx::RoundingMode::Zero;
    error(program(rounding, 0x40000000u, 0x40400000u),
          gx::Error::Code::UnsupportedModifier);

    auto flush = payload();
    flush.flush = mx::FlushMode::None;
    error(program(flush, 0x40000000u, 0x40400000u),
          gx::Error::Code::UnsupportedModifier);

    error(program(payload(), 0x40000000u, 0x40400000u, Stage::Vertex, false),
          gx::Error::Code::UninitializedRegister);
    error(program(payload(), 0x40000000u, 0x40400000u),
          gx::Error::Code::Limit, gx::Limits{2});
}

// The node cap, not the instruction-count gate, must bound the extra
// saturate node. Two saturating multiplies over the four declared color
// registers lower seven real instructions into nine nodes: the zero literal,
// four move literals, and two nodes per multiply. An eight-node limit passes
// the instruction gate and fails exactly when the second multiply appends
// its saturate node, reporting the limit at that multiply's pc.
ARMIGO_TEST(fmul_saturate_register_lower, node_cap_fails_saturate_at_multiply) {
    const auto p = program(payload(0, 1, 2), 0x40000000u, 0x3e800000u,
                           Stage::Fragment, true, payload(3, 0, 1));
    ARMIGO_CHECK_EQ(p.instructions.size(), 7u);
    std::uint32_t multiply_pc = 0;
    unsigned multiplies = 0;
    for (const auto& instruction : p.instructions) {
        if (instruction.family == Family::FloatMultiplySaturateRegister) {
            multiply_pc = instruction.pc;
            ++multiplies;
        }
    }
    ARMIGO_CHECK_EQ(multiplies, 2u);
    const auto lowered = gx::lower(p, gx::Limits{8});
    const auto* e = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(e != nullptr);
    if (e == nullptr) {
        return;
    }
    ARMIGO_CHECK(e->code == gx::Error::Code::Limit);
    ARMIGO_CHECK_EQ(e->pc, multiply_pc);
}

// Emission renders the shared saturation helper around the multiply result
// and reflection adds no resources to the stage: the emitted body must call
// the helper on the multiply node's product variable, not merely define it.
ARMIGO_TEST(fmul_saturate_register_lower, emits_helper_and_adds_no_resources) {
    const auto lowered = gx::lower(program(payload(), 0x40000000u,
                                          0x40400000u));
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if (!std::holds_alternative<gx::Program>(lowered)) {
        return;
    }
    const auto nodes = std::get<gx::Program>(lowered).nodes();
    std::optional<std::size_t> multiply_node, saturate_node;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (const auto* arithmetic =
                std::get_if<gx::FloatArithmetic>(&nodes[i].operation)) {
            if (arithmetic->operation == gx::Arithmetic::MultiplyNearest) {
                multiply_node = i;
            }
        } else if (std::holds_alternative<gx::Saturate>(
                       nodes[i].operation)) {
            saturate_node = i;
        }
    }
    ARMIGO_CHECK(multiply_node.has_value() && saturate_node.has_value());
    if (!multiply_node.has_value() || !saturate_node.has_value()) {
        return;
    }
    const auto& saturation =
        std::get<gx::Saturate>(nodes[*saturate_node].operation);
    ARMIGO_CHECK_EQ(saturation.operand.index,
                    static_cast<std::uint32_t>(*multiply_node));
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const auto emitted = gm::emit(std::get<gx::Program>(lowered), options);
    const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage != nullptr);
    if (stage == nullptr) {
        record_failure(__FILE__, __LINE__,
                       std::string("emit failed: ") +
                           std::get<gm::EmitError>(emitted).message());
        return;
    }
    ARMIGO_CHECK(stage->source.find(
                     "inline uint armigo_saturate_float32(uint bits)") !=
                 std::string::npos);
    // The emitted multiply feeds the emitted saturate call: the call argument
    // names the multiply node's scalar and the product line holds the nearest
    // multiply.
    ARMIGO_CHECK(stage->source.find("armigo_saturate_float32(v" +
                                    std::to_string(*multiply_node) + ")") !=
                 std::string::npos);
    ARMIGO_CHECK(stage->source.find("= armigo_mul_rn_ftz(") !=
                 std::string::npos);
    ARMIGO_CHECK(stage->constant_buffers.empty());
}

}  // namespace armigo::shader::test
