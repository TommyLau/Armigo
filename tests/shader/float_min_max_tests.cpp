// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "test_support.h"

#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_msl.h>
#include "graphics_operations.h"

#include <array>
#include <bit>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace gx = graphics;
namespace mx = maxwell;
namespace gm = graphics_msl;
using Stage = shader_input::Stage;
using Kind = gx::PureKind;
using Family = mx::OpcodeFamily;

struct Case { std::uint32_t a, b, minimum, maximum; };
constexpr std::array cases{
    Case{0x3f800000, 0x40000000, 0x3f800000, 0x40000000},
    Case{0xc0000000, 0xbf800000, 0xc0000000, 0xbf800000},
    Case{0xbf800000, 0x3f800000, 0xbf800000, 0x3f800000},
    Case{0, 0x80000000, 0x80000000, 0},
    Case{0x80000000, 0x80000000, 0x80000000, 0x80000000},
    Case{0, 0, 0, 0},
    Case{1, 0x807fffff, 0x80000000, 0},
    Case{0x007fffff, 0x00800000, 0, 0x00800000},
    Case{0x807fffff, 0x80800000, 0x80800000, 0x80000000},
    Case{0x7f800000, 0xff800000, 0xff800000, 0x7f800000},
    Case{0x7f7fffff, 0x7f800000, 0x7f7fffff, 0x7f800000},
    Case{0xff7fffff, 0xff800000, 0xff800000, 0xff7fffff},
    Case{0x3f800000, 0x3f800001, 0x3f800000, 0x3f800001},
    Case{0xbf800000, 0xbf800001, 0xbf800001, 0xbf800000},
    Case{0x7fc12345, 0x3f800000, 0x3f800000, 0x3f800000},
    Case{0xff812345, 0xff800000, 0xff800000, 0xff800000},
    Case{0x7f800001, 0x80000001, 0x80000000, 0x80000000},
    Case{0xffc12345, 0x7f812345, 0x7fc00000, 0x7fc00000},
    Case{0x7fc12345, 0x7fc12345, 0x7fc00000, 0x7fc00000},
};

gx::GraphSpec graph(std::uint32_t a, std::uint32_t b) {
    gx::GraphSpec spec;
    spec.stage = Stage::Vertex;
    const auto header = shader_input::sph_stage_type_for(spec.stage) << 10;
    for (unsigned i = 0; i < 4; ++i) spec.sph[i] = std::byte(header >> (8 * i));
    spec.sph[0x35] = std::byte{0xf0};
    spec.nodes = {{8, gx::Literal{a}}, {16, gx::Literal{b}},
                  {24, gx::Pure{Kind::FloatMinimum, {0, 1, 0}, 0}},
                  {40, gx::Pure{Kind::FloatMaximum, {0, 1, 0}, 0}},
                  {48, gx::Pure{Kind::FloatMinimum, {1, 0, 0}, 0}},
                  {56, gx::Pure{Kind::FloatMaximum, {1, 0, 0}, 0}}};
    for (std::uint16_t i = 0; i < 4; ++i)
        spec.exports.push_back({gx::OutputKind::Position, i, gx::ScalarId{2u + i}});
    spec.blocks = {{0, 6, gx::Return{}}};
    return spec;
}

void put(std::vector<std::uint8_t>& bytes, unsigned offset, std::uint64_t value,
         unsigned size) {
    for (unsigned i = 0; i < size; ++i) bytes[offset + i] = value >> (8 * i);
}

// Independently authored two-instruction container: tested instruction, EXIT.
mx::DecodeResult decode(std::uint64_t instruction, Stage stage = Stage::Vertex) {
    std::vector<std::uint8_t> control(0x718), code(0xa0);
    put(control, 0, shader_input::format::control_magic, 4);
    put(control, 4, 1, 4);
    put(control, 8, 14, 4);
    put(control, 0x20, control.size(), 4);
    put(control, 0x700, code.size(), 4);
    put(control, 0x704, code.size(), 4);
    put(control, 0x708, 0x30, 4);
    put(control, 0x714, static_cast<unsigned>(stage), 4);
    put(code, 0, shader_input::format::code_wrapper_magic, 4);
    put(code, 0x30, shader_input::sph_stage_type_for(stage) << 10, 4);
    put(code, 0x88, instruction, 8);
    put(code, 0x90, (std::uint64_t{0xe300} << 48) | (7u << 16) | 15u, 8);
    return mx::decode({control.data(), control.size()}, {code.data(), code.size()});
}

std::uint64_t word(unsigned window = 0x5c60) {
    return (std::uint64_t{window} << 48) | (std::uint64_t{1} << 44) |
           (std::uint64_t{7} << 39) | (7u << 16) | (1u << 8) | 3u;
}

std::uint64_t modifiers(unsigned mask) {
    return (std::uint64_t{mask & 1u} << 46) |
           (std::uint64_t{(mask >> 1) & 1u} << 48) |
           (std::uint64_t{(mask >> 2) & 1u} << 49) |
           (std::uint64_t{(mask >> 3) & 1u} << 45);
}

void rejection(std::uint64_t encoded, mx::DecodeField field, std::uint64_t value,
               std::uint64_t expected = 0) {
    const auto result = decode(encoded);
    const auto* error = std::get_if<mx::DecodeError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (!error) return;
    ARMIGO_CHECK(error->code == mx::DecodeError::Code::UnsupportedModifier);
    ARMIGO_CHECK(error->field == field);
    ARMIGO_CHECK_EQ(error->pc, 8u);
    ARMIGO_CHECK_EQ(error->value, value);
    ARMIGO_CHECK_EQ(error->expected, expected);
}

void append(mx::DecodedProgram& program, Family family, mx::InstructionPayload payload) {
    const auto index = program.instructions.size();
    program.instructions.push_back({static_cast<std::uint32_t>(32 * (index / 3) +
        8 * (index % 3 + 1)), 0, family, {7, false}, std::move(payload)});
}

mx::DecodedProgram program(mx::FloatMinMax value, Family family,
                           Stage stage = Stage::Vertex, bool initialize_b = true) {
    mx::DecodedProgram result;
    result.stage = stage;
    const auto header = shader_input::sph_stage_type_for(stage) << 10;
    for (unsigned i = 0; i < 4; ++i) result.sph[i] = std::byte(header >> (8 * i));
    result.sph[stage == Stage::Vertex ? 0x35 : 0x48] =
        stage == Stage::Vertex ? std::byte{0xf0} : std::byte{0x0f};
    append(result, Family::Mov32Immediate, mx::Mov32Immediate{{0}, 15, 0});
    append(result, Family::Mov32Immediate, mx::Mov32Immediate{{1}, 15, 0xc0000000});
    if (initialize_b)
        append(result, Family::Mov32Immediate, mx::Mov32Immediate{{2}, 15, 0x3f800000});
    append(result, Family::Mov32Immediate, mx::Mov32Immediate{{3}, 15, 0x3f800000});
    append(result, family, value);
    if (stage == Stage::Vertex) {
        const auto source = value.destination.is_zero() ? mx::RegisterOperand{1} : value.destination;
        for (unsigned lane = 0; lane < 4; ++lane)
            append(result, Family::AttributeStore, mx::AttributeAccess{source, {255}, {255},
                static_cast<std::uint16_t>((28 + lane) * 4), 0, false});
    }
    append(result, Family::Exit, mx::Exit{15});
    return result;
}

std::uint32_t evaluate_first(const gx::Program& program, const gx::Inputs& inputs = {}) {
    const auto result = gx::evaluate(program, inputs);
    const auto* evaluation = std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(evaluation != nullptr);
    if (!evaluation) return 0;
    for (const auto& output : evaluation->outputs)
        if (output.component == 0) return output.bits;
    ARMIGO_CHECK(false);
    return 0;
}

void lower_error(const mx::DecodedProgram& program, gx::Error::Code code) {
    const auto lowered = gx::lower(program);
    const auto* error = std::get_if<gx::Error>(&lowered);
    ARMIGO_CHECK(error != nullptr);
    if (error) ARMIGO_CHECK(error->code == code);
}
}  // namespace

ARMIGO_TEST(float_min_max, cpu_graph_edges_and_operand_order) {
    for (const auto& item : cases) {
        const auto built = gx::make_graph(graph(item.a, item.b));
        const auto* p = std::get_if<gx::Program>(&built);
        ARMIGO_CHECK(p != nullptr);
        if (!p) continue;
        const auto result = gx::evaluate(*p, {});
        const auto* evaluation = std::get_if<gx::Evaluation>(&result);
        ARMIGO_CHECK(evaluation != nullptr);
        if (!evaluation) continue;
        ARMIGO_CHECK_EQ(evaluation->outputs.size(), 4u);
        for (const auto& output : evaluation->outputs)
            ARMIGO_CHECK_EQ(output.bits, output.component % 2 ? item.maximum : item.minimum);
    }
}

ARMIGO_TEST(float_min_max, finite_order_matches_independent_float_comparison) {
    const auto* minimum = gx::detail::find_operation(Kind::FloatMinimum);
    const auto* maximum = gx::detail::find_operation(Kind::FloatMaximum);
    ARMIGO_CHECK(minimum && maximum);
    if (!minimum || !maximum) return;
    std::uint32_t state = 0x819abcde;
    const auto next = [&] { state = state * 1664525u + 1013904223u; return state; };
    for (unsigned i = 0; i < 4096; ++i) {
        const auto a = next(), b = next();
        const auto normal = [](std::uint32_t v) {
            const auto exponent = v & 0x7f800000u;
            return exponent != 0 && exponent != 0x7f800000u;
        };
        if (!normal(a) || !normal(b)) continue;
        const bool less = std::bit_cast<float>(a) < std::bit_cast<float>(b);
        ARMIGO_CHECK_EQ(minimum->evaluate({a, b, 0}, 0), less ? a : b);
        ARMIGO_CHECK_EQ(maximum->evaluate({a, b, 0}, 0), less ? b : a);
    }
}

ARMIGO_TEST(float_min_max, decodes_three_sources_and_six_modifier_profiles) {
    for (const auto stage : {Stage::Vertex, Stage::Fragment})
    for (const unsigned window : {0x5c60u, 0x4c60u, 0x3860u})
    for (unsigned mask = 0; mask < 16; ++mask)
    for (const bool maximum : {false, true}) {
        const bool admitted = mask == 0 || mask == 1 || mask == 2 ||
                              mask == 4 || mask == 5 || mask == 8;
        auto encoded = word(window) | modifiers(mask) | (std::uint64_t{maximum} << 42);
        if (window == 0x5c60) encoded |= std::uint64_t{254} << 20;
        if (window == 0x4c60) encoded |= (std::uint64_t{17} << 34) | (std::uint64_t{0x3fff} << 20);
        if (window == 0x3860) encoded |= (std::uint64_t{1} << 56) | (std::uint64_t{0x3f800} << 20);
        const auto result = decode(encoded, stage);
        if (!admitted) {
            const auto* error = std::get_if<mx::DecodeError>(&result);
            ARMIGO_CHECK(error != nullptr);
            if (error) {
                ARMIGO_CHECK(error->code == mx::DecodeError::Code::UnsupportedModifier);
                const auto field = (mask & 1) ? mx::DecodeField::AbsoluteA :
                                   (mask & 4) ? mx::DecodeField::AbsoluteB : mx::DecodeField::NegateB;
                ARMIGO_CHECK(error->field == field);
            }
            continue;
        }
        const auto* p = std::get_if<mx::DecodedProgram>(&result);
        ARMIGO_CHECK(p != nullptr);
        if (!p) continue;
        const auto& instruction = p->instructions.front();
        const auto& v = std::get<mx::FloatMinMax>(instruction.payload);
        ARMIGO_CHECK_EQ(instruction.original_word, encoded);
        ARMIGO_CHECK_EQ(instruction.pc, 8u);
        ARMIGO_CHECK(v.destination.index == 3 && v.source_a.index == 1);
        ARMIGO_CHECK(v.absolute_a == bool(mask & 1) && v.negate_a == bool(mask & 2));
        ARMIGO_CHECK(v.absolute_b == bool(mask & 4) && v.negate_b == bool(mask & 8));
        ARMIGO_CHECK(v.negate_predicate == maximum && v.flush == mx::FlushMode::FlushToZero);
        if (window == 0x5c60) {
            ARMIGO_CHECK(instruction.family == Family::FloatMinMaxRegister);
            ARMIGO_CHECK_EQ(std::get<mx::RegisterOperand>(v.source_b).index, 254u);
        } else if (window == 0x4c60) {
            ARMIGO_CHECK(instruction.family == Family::FloatMinMaxConstant);
            const auto b = std::get<mx::ConstantBufferOperand>(v.source_b);
            ARMIGO_CHECK(b.binding == 17 && b.byte_offset == 65532);
        } else {
            ARMIGO_CHECK(instruction.family == Family::FloatMinMaxImmediate);
            ARMIGO_CHECK_EQ(std::get<mx::ShortFloatImmediate>(v.source_b).raw_bits, 0xbf800000u);
        }
    }
}

ARMIGO_TEST(float_min_max, rejects_select_predicates_modes_and_stray_bits) {
    for (const unsigned window : {0x5c60u, 0x4c60u, 0x3860u}) {
        for (unsigned p = 0; p < 7; ++p)
            rejection((word(window) & ~(std::uint64_t{7} << 39)) | (std::uint64_t{p} << 39),
                      mx::DecodeField::SelectPredicate, p, 7);
        rejection(word(window) ^ (std::uint64_t{1} << 44), mx::DecodeField::FlushMode, 0, 1);
        rejection(word(window) | (std::uint64_t{1} << 47), mx::DecodeField::ConditionCode, 1);
        for (unsigned bit : {43u, 50u})
            rejection(word(window) | (std::uint64_t{1} << bit),
                      mx::DecodeField::ReservedBits, std::uint64_t{1} << bit);
        for (unsigned p = 0; p < 16; ++p) {
            if (p == 7) continue;
            const auto result = decode((word(window) & ~(std::uint64_t{15} << 16)) | (p << 16));
            const auto* error = std::get_if<mx::DecodeError>(&result);
            ARMIGO_CHECK(error && error->code == mx::DecodeError::Code::UnsupportedPredicate);
        }
    }
    for (unsigned bit = 28; bit < 39; ++bit)
        rejection(word() | (std::uint64_t{1} << bit), mx::DecodeField::ReservedBits,
                  std::uint64_t{1} << bit);
    for (unsigned binding = 18; binding < 32; ++binding)
        rejection(word(0x4c60) | (std::uint64_t{binding} << 34),
                  mx::DecodeField::ConstantBinding, binding, 17);
}

ARMIGO_TEST(float_min_max, lowers_sources_aliases_zero_register_and_modifiers) {
    for (const auto family : {Family::FloatMinMaxRegister, Family::FloatMinMaxConstant,
                              Family::FloatMinMaxImmediate})
    for (unsigned mask : {0u, 1u, 2u, 4u, 5u, 8u})
    for (const bool maximum : {false, true})
    for (unsigned destination : {1u, 2u, 3u, 255u}) {
        mx::FloatMinMax v{{static_cast<std::uint8_t>(destination)}, {1}, mx::RegisterOperand{2}};
        if (family == Family::FloatMinMaxConstant) v.source_b = mx::ConstantBufferOperand{5, 8};
        if (family == Family::FloatMinMaxImmediate) v.source_b = mx::ShortFloatImmediate{0x3f800000};
        v.absolute_a = mask & 1; v.negate_a = mask & 2;
        v.absolute_b = mask & 4; v.negate_b = mask & 8;
        v.negate_predicate = maximum;
        const auto lowered = gx::lower(program(v, family));
        const auto* p = std::get_if<gx::Program>(&lowered);
        ARMIGO_CHECK(p != nullptr);
        if (!p) continue;
        const std::array data{0u, 0u, 0x3f800000u};
        gx::Inputs inputs; inputs.constant_buffers[5] = data;
        const bool positive_a = mask == 1 || mask == 2 || mask == 5;
        const auto expected = maximum ? (positive_a ? 0x40000000u :
            (mask == 8 ? 0xbf800000u : 0x3f800000u)) :
            (positive_a ? 0x3f800000u : 0xc0000000u);
        ARMIGO_CHECK_EQ(evaluate_first(*p, inputs), destination == 255 ? 0xc0000000u : expected);
        std::vector<Kind> kinds;
        for (const auto& node : p->nodes())
            if (const auto* pure = std::get_if<gx::Pure>(&node.operation)) kinds.push_back(pure->kind);
        std::vector<Kind> expected_kinds;
        if (v.absolute_a) expected_kinds.push_back(Kind::FloatAbsolute);
        if (v.negate_a) expected_kinds.push_back(Kind::FloatNegate);
        if (v.absolute_b) expected_kinds.push_back(Kind::FloatAbsolute);
        if (v.negate_b) expected_kinds.push_back(Kind::FloatNegate);
        expected_kinds.push_back(maximum ? Kind::FloatMaximum : Kind::FloatMinimum);
        ARMIGO_CHECK(kinds == expected_kinds);
    }
    mx::FloatMinMax v{{0}, {255}, mx::RegisterOperand{255}};
    const auto lowered = gx::lower(program(v, Family::FloatMinMaxRegister, Stage::Fragment));
    const auto* p = std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(p != nullptr);
    if (p) ARMIGO_CHECK_EQ(evaluate_first(*p), 0u);
}

ARMIGO_TEST(float_min_max, malformed_lowering_fails_explicitly) {
    mx::FloatMinMax base{{3}, {1}, mx::RegisterOperand{2}};
    auto v = base; v.flush = mx::FlushMode::None;
    lower_error(program(v, Family::FloatMinMaxRegister), gx::Error::Code::UnsupportedModifier);
    v = base; v.absolute_a = true; v.negate_b = true;
    lower_error(program(v, Family::FloatMinMaxRegister), gx::Error::Code::UnsupportedModifier);
    lower_error(program(base, Family::FloatMinMaxConstant), gx::Error::Code::InvalidInstruction);
    lower_error(program(base, Family::FloatMinMaxRegister, Stage::Vertex, false),
                gx::Error::Code::UninitializedRegister);
    v = base; v.source_a = {4};
    lower_error(program(v, Family::FloatMinMaxRegister), gx::Error::Code::UninitializedRegister);
    for (const auto operand : {mx::ConstantBufferOperand{18, 0}, {5, 2}, {5, 65536}}) {
        v = base; v.source_b = operand;
        lower_error(program(v, Family::FloatMinMaxConstant), gx::Error::Code::UnsupportedModifier);
    }
    auto malformed = program(base, Family::FloatMinMaxRegister);
    for (auto& instruction : malformed.instructions)
        if (instruction.family == Family::FloatMinMaxRegister) instruction.payload = mx::Exit{15};
    lower_error(malformed, gx::Error::Code::InvalidInstruction);
}

ARMIGO_TEST(float_min_max, registry_signatures_and_graph_validation) {
    for (const auto kind : {Kind::FloatMinimum, Kind::FloatMaximum}) {
        const auto* entry = gx::detail::find_operation(kind);
        ARMIGO_CHECK(entry && entry->arity == 2 && entry->result == gx::ValueType::Raw32);
        if (entry) {
            ARMIGO_CHECK(entry->operands[0] == gx::ValueType::Raw32);
            ARMIGO_CHECK(entry->operands[1] == gx::ValueType::Raw32);
            ARMIGO_CHECK(entry->admits(0) && !entry->admits(1));
        }
        for (unsigned fault = 0; fault < 3; ++fault) {
            auto spec = graph(0, 0);
            auto& operation = std::get<gx::Pure>(spec.nodes[2].operation);
            operation.kind = kind;
            if (fault == 0) operation.immediate = 1;
            if (fault == 1) operation.operands[2] = 1;
            if (fault == 2) spec.nodes[1].operation = gx::PredicateLiteral{true};
            const auto result = gx::make_graph(spec);
            const auto* error = std::get_if<gx::Error>(&result);
            ARMIGO_CHECK(error != nullptr);
            if (error) ARMIGO_CHECK(error->code == (fault == 2 ? gx::Error::Code::GraphTypes :
                                                                 gx::Error::Code::GraphFormat));
        }
    }
}

ARMIGO_TEST(float_min_max, msl_helpers_and_constant_reflection) {
    const auto check = [](const gx::Program& p, bool both) {
        gm::EmitOptions options;
        options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
        const auto emitted = gm::emit(p, options);
        const auto* stage = std::get_if<gm::EmittedStage>(&emitted);
        ARMIGO_CHECK(stage != nullptr);
        if (!stage) return;
        for (const auto name : {"armigo_float_min", "armigo_float_max"}) {
            const auto declaration = std::string("inline uint ") + name;
            const auto offset = stage->source.find(declaration);
            if (both || std::string(name) == "armigo_float_min") {
                ARMIGO_CHECK(offset != std::string::npos);
                ARMIGO_CHECK(stage->source.find(declaration, offset + 1) == std::string::npos);
            } else ARMIGO_CHECK(offset == std::string::npos);
        }
        ARMIGO_CHECK(stage->source.find("0x7fc00000u") != std::string::npos);
        ARMIGO_CHECK(stage->source.find("fmin(") == std::string::npos);
        ARMIGO_CHECK(stage->source.find("fmax(") == std::string::npos);
        if (!both) {
            ARMIGO_CHECK_EQ(stage->constant_buffers.size(), 1u);
            if (!stage->constant_buffers.empty()) {
                ARMIGO_CHECK_EQ(stage->constant_buffers[0].guest_binding, 5u);
                ARMIGO_CHECK_EQ(stage->constant_buffers[0].minimum_size, 12u);
            }
        }
    };
    const auto built = gx::make_graph(graph(0, 0));
    if (const auto* p = std::get_if<gx::Program>(&built)) check(*p, true);
    else ARMIGO_CHECK(false);
    mx::FloatMinMax v{{3}, {1}, mx::ConstantBufferOperand{5, 8}};
    const auto lowered = gx::lower(program(v, Family::FloatMinMaxConstant));
    if (const auto* p = std::get_if<gx::Program>(&lowered)) {
        check(*p, false);
        const auto evaluated = gx::evaluate(*p, {});
        const auto* error = std::get_if<gx::Error>(&evaluated);
        ARMIGO_CHECK(error && error->code == gx::Error::Code::ConstantBounds);
    } else ARMIGO_CHECK(false);
}
}  // namespace armigo::shader::test
