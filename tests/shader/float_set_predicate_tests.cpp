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
#include "graphics_builder.h"
#include "graphics_families.h"
#include "graphics_operations.h"

#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/graphics_forward.h>
#include <armigo/shader/graphics_msl.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader::test {
namespace {
namespace mx = maxwell;
namespace gx = graphics;
namespace gm = graphics_msl;
using Family = mx::OpcodeFamily;
using gx::PureKind;
using Stage = shader_input::Stage;

std::array<std::byte, 80> header(Stage stage, unsigned color_mask = 0) {
    std::array<std::byte, 80> result{};
    const auto word = shader_input::sph_stage_type_for(stage) << 10;
    for (unsigned i = 0; i < 4; ++i) result[i] = std::byte(word >> (8 * i));
    result[0x48] = std::byte(color_mask);
    if (stage == Stage::Vertex) result[0x35] = std::byte{0xf0};
    return result;
}

gx::Node pure(PureKind kind, std::array<std::uint32_t, 3> inputs,
              std::uint32_t immediate = 0) {
    return {8, gx::Pure{kind, inputs, immediate}};
}

gx::GraphSpec compare_graph(std::uint32_t a, std::uint32_t b,
                            unsigned first_code, bool ftz) {
    gx::GraphSpec spec;
    spec.stage = Stage::Fragment;
    spec.sph = header(Stage::Fragment, 15);
    spec.nodes = {{8, gx::Literal{a}}, {8, gx::Literal{b}},
                  {8, gx::Literal{1}}, {8, gx::Literal{0}}};
    for (unsigned lane = 0; lane < 4; ++lane) {
        const auto index = static_cast<std::uint32_t>(spec.nodes.size());
        spec.nodes.push_back(pure(PureKind::FloatComparePredicate, {0, 1, 0},
                                 first_code + lane + (ftz ? 16 : 0)));
        spec.nodes.push_back(pure(PureKind::Select, {index, 2, 3}));
        spec.exports.push_back({gx::OutputKind::Color, static_cast<std::uint16_t>(lane),
                                gx::ScalarId{index + 1}});
    }
    spec.blocks = {{0, static_cast<std::uint32_t>(spec.nodes.size()), gx::Return{}}};
    return spec;
}

std::array<std::uint32_t, 4> evaluate(const gx::Program& program,
                                     std::span<const std::uint32_t> constants = {}) {
    gx::Inputs inputs;
    inputs.constant_buffers[0] = constants;
    const auto result = gx::evaluate(program, inputs);
    const auto* values = std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(values != nullptr);
    std::array<std::uint32_t, 4> bits{};
    if (values) for (const auto& value : values->outputs) bits[value.component] = value.bits;
    return bits;
}

std::string emit(const gx::Program& program) {
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const auto result = gm::emit(program, options);
    const auto* stage = std::get_if<gm::EmittedStage>(&result);
    ARMIGO_CHECK(stage != nullptr);
    return stage ? stage->source : std::string{};
}

constexpr std::uint64_t mask(unsigned width, unsigned shift) {
    return ((std::uint64_t{1} << width) - 1) << shift;
}
std::uint64_t field(std::uint64_t word, unsigned shift, unsigned width,
                    std::uint64_t value) {
    return (word & ~mask(width, shift)) | (value << shift);
}
std::uint64_t encoded(std::uint16_t opcode = 0x5bb0) {
    return (std::uint64_t{opcode} << 48) | (1ull << 48) | (1ull << 47) |
           (7ull << 39) | (9ull << 20) | (7ull << 16) | (4ull << 8) | (3ull << 3) | 6;
}

// Independently authored container with one instruction and an unconditional EXIT.
mx::DecodeResult decode(std::uint64_t word) {
    std::vector<std::uint8_t> control(0x718), code(0xa0);
    const auto put = [](auto& bytes, unsigned at, std::uint64_t value, unsigned count) {
        for (unsigned i = 0; i < count; ++i) bytes[at + i] = std::uint8_t(value >> (8 * i));
    };
    put(control, 0, shader_input::format::control_magic, 4);
    put(control, 4, shader_input::format::control_tag, 4);
    put(control, 8, 14, 4);
    put(control, 0x20, control.size(), 4);
    put(control, 0x700, code.size(), 4);
    put(control, 0x704, code.size(), 4);
    put(control, 0x708, 0x30, 4);
    put(code, 0, 0x12345678, 4);
    put(code, 0x30, shader_input::sph_stage_type_for(Stage::Vertex) << 10, 4);
    put(code, 0x88, word, 8);
    put(code, 0x90, (0xe300ull << 48) | (7ull << 16) | 15, 8);
    return mx::decode({control.data(), control.size()}, {code.data(), code.size()});
}

std::optional<mx::FloatSetPredicate> payload(std::uint64_t word, Family family) {
    const auto result = decode(word);
    const auto* program = std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (!program) return {};
    ARMIGO_CHECK_EQ(program->instructions.size(), 2u);
    const auto& instruction = program->instructions.front();
    ARMIGO_CHECK(instruction.family == family);
    ARMIGO_CHECK_EQ(instruction.original_word, word);
    const auto* value = std::get_if<mx::FloatSetPredicate>(&instruction.payload);
    ARMIGO_CHECK(value != nullptr);
    return value ? std::optional{*value} : std::nullopt;
}

mx::Instruction instruction(Family family, auto value, mx::Predicate guard = {7, false}) {
    return {0, 0, family, guard, value};
}
mx::Instruction mov(std::uint8_t destination, std::uint32_t bits,
                    mx::Predicate guard = {7, false}) {
    return instruction(Family::Mov32Immediate,
                       mx::Mov32Immediate{{destination}, 15, bits}, guard);
}
mx::FloatSetPredicate comparison() {
    mx::FloatSetPredicate value;
    value.destination_a = 0;
    value.destination_b = 1;
    value.source_a = {10};
    value.source_b = mx::RegisterOperand{11};
    value.boolean_predicate = {7, false};
    value.comparison = mx::FloatCompareCode::LessThan;
    return value;
}
void append(mx::ForwardProgram& program, mx::Instruction value) {
    const auto index = static_cast<std::uint32_t>(program.instructions.size());
    value.pc = (index / 3) * 32 + 8 + (index % 3) * 8;
    program.instructions.push_back(value);
    program.program_size = (index / 3 + 1) * 32;
}

// Observe the two predicates independently, even when both become true or false.
mx::ForwardProgram forward(mx::FloatSetPredicate value, std::uint32_t a,
                           std::uint32_t b, Family family = Family::FloatSetPredicateRegister,
                           bool seed = true, mx::Predicate guard = {7, false}) {
    mx::ForwardProgram program;
    program.stage = Stage::Fragment;
    program.sph = header(Stage::Fragment, 3);
    append(program, mov(0, 0));
    append(program, mov(1, 0));
    append(program, mov(10, a));
    append(program, mov(11, b));
    if (seed) {
        auto initial = comparison();
        initial.comparison = mx::FloatCompareCode::True;
        initial.destination_a = 0;
        initial.destination_b = 1;
        append(program, instruction(Family::FloatSetPredicateRegister, initial));
    }
    append(program, instruction(family, value, guard));
    append(program, mov(0, 1, {0, false}));
    append(program, mov(1, 1, {1, false}));
    append(program, instruction(Family::Exit, mx::Exit{15}));
    return program;
}
std::array<std::uint32_t, 4> run(const mx::ForwardProgram& source,
                                std::span<const std::uint32_t> constants = {}) {
    const auto result = gx::lower_forward(source);
    const auto* program = std::get_if<gx::Program>(&result);
    if (!program) {
        const auto& error = std::get<gx::Error>(result);
        record_failure(__FILE__, __LINE__, std::string(error.message()) +
                       " pc=" + std::to_string(error.pc));
        return {};
    }
    return evaluate(*program, constants);
}
void check_pair(const std::array<std::uint32_t, 4>& actual, bool a, bool b) {
    ARMIGO_CHECK_EQ(actual[0], static_cast<std::uint32_t>(a));
    ARMIGO_CHECK_EQ(actual[1], static_cast<std::uint32_t>(b));
}
}  // namespace

ARMIGO_TEST(float_set_predicate, all_codes_and_binary32_boundaries) {
    // Explicit truth masks in encoding order: less, equal, greater, unordered.
    constexpr std::uint16_t lt = 0xaaaa, eq = 0xcccc, gt = 0xf0f0, nan = 0xff00;
    const struct { std::uint32_t a, b; std::uint16_t preserve, flush; } cases[] = {
        {0, 0x80000000, eq, eq}, {0x80000000, 0, eq, eq},
        {1, 0, gt, eq}, {0, 1, lt, eq}, {0x80000001, 0, lt, eq},
        {0, 0x807fffff, gt, eq}, {1, 2, lt, eq}, {0x80000001, 0x80000002, gt, eq},
        {0x007fffff, 0x00800000, lt, lt}, {0x80800000, 0x807fffff, lt, lt},
        {0x3f800000, 0x40000000, lt, lt}, {0xc0000000, 0xbf800000, lt, lt},
        {0xbf800000, 0x3f800000, lt, lt}, {0x3f800000, 0x3f800001, lt, lt},
        {0x7f800000, 0x7f7fffff, gt, gt}, {0xff800000, 0xff7fffff, lt, lt},
        {0x7f800000, 0x7f800000, eq, eq}, {0xff800000, 0xff800000, eq, eq},
        {0x7fc12345, 0x3f800000, nan, nan}, {0, 0xff800001, nan, nan},
        {0x7f800001, 0x7f800000, nan, nan}, {0xffc54321, 0x7fc12345, nan, nan},
    };
    for (const auto& item : cases) for (const bool ftz : {false, true}) {
        for (unsigned first = 0; first < 16; first += 4) {
            const auto graph = gx::make_graph(compare_graph(item.a, item.b, first, ftz));
            const auto* program = std::get_if<gx::Program>(&graph);
            ARMIGO_CHECK(program != nullptr);
            if (!program) continue;
            const auto actual = evaluate(*program);
            const auto truth = ftz ? item.flush : item.preserve;
            for (unsigned lane = 0; lane < 4; ++lane) {
                ARMIGO_CHECK_EQ(actual[lane], (truth >> (first + lane)) & 1u);
            }
        }
    }
}

ARMIGO_TEST(float_set_predicate, registry_rejects_invalid_immediates_and_types) {
    const auto* entry = gx::detail::find_operation(PureKind::FloatComparePredicate);
    ARMIGO_CHECK(entry != nullptr);
    if (!entry) return;
    ARMIGO_CHECK(entry->arity == 2 && entry->result == gx::ValueType::Predicate);
    for (unsigned i = 0; i < 32; ++i) ARMIGO_CHECK(entry->admits(i));
    for (const auto invalid : {32u, 0x80000000u, 0xffffffffu}) {
        auto spec = compare_graph(0, 0, 0, true);
        std::get<gx::Pure>(spec.nodes[4].operation).immediate = invalid;
        const auto result = gx::make_graph(spec);
        const auto* error = std::get_if<gx::Error>(&result);
        ARMIGO_CHECK(error && error->code == gx::Error::Code::GraphFormat);
    }
    auto spec = compare_graph(0, 0, 0, true);
    spec.nodes[0].operation = gx::PredicateLiteral{true};
    const auto result = gx::make_graph(spec);
    const auto* error = std::get_if<gx::Error>(&result);
    ARMIGO_CHECK(error && error->code == gx::Error::Code::GraphTypes);
}

ARMIGO_TEST(float_set_predicate, decoder_all_forms_and_fields) {
    for (const auto opcode : {0x5bb0, 0x4bb0, 0x36b0}) {
        const auto family = opcode == 0x5bb0 ? Family::FloatSetPredicateRegister :
            opcode == 0x4bb0 ? Family::FloatSetPredicateConstant : Family::FloatSetPredicateImmediate;
        for (unsigned code = 0; code < 16; ++code) for (unsigned bop = 0; bop < 3; ++bop) {
            auto word = field(field(encoded(opcode), 48, 4, code), 45, 2, bop);
            word |= (1ull << 6) | (1ull << 7) | (1ull << 42) | (1ull << 43) | (1ull << 44);
            const auto value = payload(word, family);
            if (!value) continue;
            ARMIGO_CHECK(value->destination_a == 3 && value->destination_b == 6);
            ARMIGO_CHECK(value->source_a.index == 4);
            ARMIGO_CHECK(value->boolean_predicate.index == 7 && value->boolean_predicate.inverted);
            ARMIGO_CHECK(value->absolute_a && value->negate_a && value->absolute_b && value->negate_b);
            ARMIGO_CHECK_EQ(static_cast<unsigned>(value->comparison), code);
            ARMIGO_CHECK_EQ(static_cast<unsigned>(value->boolean_operation), bop);
            ARMIGO_CHECK(value->flush_to_zero);
        }
        const auto value = payload(field(field(encoded(opcode), 47, 1, 0), 39, 3, 0), family);
        ARMIGO_CHECK(value && !value->flush_to_zero && value->boolean_predicate.index == 0);
    }
    const auto reg = payload(field(field(field(encoded(), 0, 6, 63), 8, 8, 255), 20, 8, 255),
                             Family::FloatSetPredicateRegister);
    ARMIGO_CHECK(reg && reg->destination_a == 7 && reg->destination_b == 7 && reg->source_a.is_zero());
    if (reg) ARMIGO_CHECK(std::get<mx::RegisterOperand>(reg->source_b).is_zero());
    const auto cb = payload(field(encoded(0x4bb0), 20, 19, (17u << 14) | 0x3fff),
                            Family::FloatSetPredicateConstant);
    if (cb) {
        const auto source = std::get<mx::ConstantBufferOperand>(cb->source_b);
        ARMIGO_CHECK(source.binding == 17 && source.byte_offset == 0xfffc);
    }
    for (unsigned sign = 0; sign < 2; ++sign) {
        const auto imm = payload(field(field(encoded(0x36b0), 20, 19, 0x40000), 56, 1, sign),
                                 Family::FloatSetPredicateImmediate);
        if (imm) ARMIGO_CHECK_EQ(std::get<mx::ShortFloatImmediate>(imm->source_b).raw_bits,
                                 0x40000000u | (sign << 31));
    }
}

ARMIGO_TEST(float_set_predicate, decoder_rejections_are_structured) {
    const auto reject = [](std::uint64_t word, mx::DecodeField field, std::uint64_t value,
                           std::uint64_t expected, Family family) {
        const auto result = decode(word);
        const auto* error = std::get_if<mx::DecodeError>(&result);
        ARMIGO_CHECK(error != nullptr);
        if (!error) return;
        ARMIGO_CHECK(error->code == mx::DecodeError::Code::UnsupportedModifier);
        ARMIGO_CHECK(error->field == field && error->family == family);
        ARMIGO_CHECK_EQ(error->pc, 8u);
        ARMIGO_CHECK_EQ(error->value, value);
        ARMIGO_CHECK_EQ(error->expected, expected);
    };
    for (const auto opcode : {0x5bb0, 0x4bb0, 0x36b0}) {
        const auto family = opcode == 0x5bb0 ? Family::FloatSetPredicateRegister :
            opcode == 0x4bb0 ? Family::FloatSetPredicateConstant : Family::FloatSetPredicateImmediate;
        reject(field(encoded(opcode), 45, 2, 3), mx::DecodeField::BooleanOperation, 3, 2, family);
    }
    for (unsigned bit = 28; bit <= 38; ++bit) {
        reject(encoded() | (1ull << bit), mx::DecodeField::ReservedBits, 1ull << bit, 0,
               Family::FloatSetPredicateRegister);
    }
    for (unsigned binding = 18; binding < 32; ++binding) {
        reject(field(encoded(0x4bb0), 34, 5, binding), mx::DecodeField::ConstantBinding,
               binding, 17, Family::FloatSetPredicateConstant);
    }
    const auto guarded = decode(field(encoded(), 16, 4, 3));
    const auto* error = std::get_if<mx::DecodeError>(&guarded);
    ARMIGO_CHECK(error && error->code == mx::DecodeError::Code::UnsupportedPredicate);
}

ARMIGO_TEST(float_set_predicate, lowering_boolean_truth_tables_and_aliases) {
    for (unsigned bop = 0; bop < 3; ++bop) for (bool compare : {false, true}) {
        for (unsigned gate_index : {0u, 1u, 7u}) for (bool inverted : {false, true}) {
            auto value = comparison();
            value.boolean_operation = static_cast<mx::PredicateBooleanOperation>(bop);
            value.boolean_predicate = {static_cast<std::uint8_t>(gate_index), inverted};
            const bool gate = (gate_index != 1) != inverted;
            const auto combine = [=](bool p) {
                return bop == 0 ? p && gate : bop == 1 ? p || gate : p != gate;
            };
            check_pair(run(forward(value, compare ? 0x3f800000 : 0x40400000, 0x40000000)),
                       combine(compare), combine(!compare));
        }
    }
    auto value = comparison();
    value.destination_b = 0;
    check_pair(run(forward(value, 0x3f800000, 0x40000000)), false, false);
    value.destination_a = value.destination_b = 7;
    check_pair(run(forward(value, 0x3f800000, 0x40000000)), true, false);
}

ARMIGO_TEST(float_set_predicate, lowering_sources_modifiers_and_guards) {
    auto value = comparison();
    value.source_b = mx::ConstantBufferOperand{0, 0};
    const std::array constants{0x40000000u};
    check_pair(run(forward(value, 0x3f800000, 0, Family::FloatSetPredicateConstant), constants), true, false);
    value.source_b = mx::ShortFloatImmediate{0x40000000};
    check_pair(run(forward(value, 0x40400000, 0, Family::FloatSetPredicateImmediate)), false, true);
    value = comparison();
    value.absolute_a = value.negate_a = value.absolute_b = value.negate_b = true;
    // abs then negate: -2 < -1, rather than +2 < +1.
    check_pair(run(forward(value, 0xc0000000, 0xbf800000)), true, false);
    value = comparison();
    value.comparison = mx::FloatCompareCode::Equal;
    check_pair(run(forward(value, 1, 0)), true, false);
    value.flush_to_zero = false;
    check_pair(run(forward(value, 1, 0)), false, true);
    // A false guard preserves the seeded predicate values.
    check_pair(run(forward(value, 1, 0, Family::FloatSetPredicateRegister, true, {1, false})), true, false);
    check_pair(run(forward(value, 1, 0, Family::FloatSetPredicateRegister, true, {0, false})), false, true);
}

ARMIGO_TEST(float_set_predicate, lowering_rejects_bad_payloads_and_undefined_sources) {
    const auto reject = [](mx::ForwardProgram source, gx::Error::Code code) {
        const auto result = gx::lower_forward(source);
        const auto* error = std::get_if<gx::Error>(&result);
        ARMIGO_CHECK(error && error->code == code);
    };
    auto value = comparison();
    value.boolean_predicate = {2, false};
    reject(forward(value, 0, 0), gx::Error::Code::UninitializedRegister);
    value = comparison();
    value.source_a = {12};
    reject(forward(value, 0, 0), gx::Error::Code::UninitializedRegister);
    value = comparison();
    value.comparison = static_cast<mx::FloatCompareCode>(16);
    reject(forward(value, 0, 0), gx::Error::Code::UnsupportedModifier);
    value = comparison();
    value.boolean_operation = static_cast<mx::PredicateBooleanOperation>(3);
    reject(forward(value, 0, 0), gx::Error::Code::UnsupportedModifier);
    value = comparison();
    value.source_b = mx::ShortFloatImmediate{0};
    reject(forward(value, 0, 0), gx::Error::Code::InvalidInstruction);
    value.source_b = mx::ShortFloatImmediate{1};
    reject(forward(value, 0, 0, Family::FloatSetPredicateImmediate), gx::Error::Code::InvalidInstruction);
    value = comparison();
    value.destination_b = 8;
    reject(forward(value, 0, 0), gx::Error::Code::InvalidInstruction);
}

ARMIGO_TEST(float_set_predicate, straight_line_and_graph_msl) {
    auto value = comparison();
    value.source_a = {255};
    value.source_b = mx::RegisterOperand{255};
    mx::DecodedProgram source;
    source.stage = Stage::Fragment;
    source.sph = header(Stage::Fragment, 15);
    source.instructions = {mov(0, 0x3f800000), mov(1, 0), mov(2, 0), mov(3, 0),
                           instruction(Family::FloatSetPredicateRegister, value),
                           instruction(Family::Exit, mx::Exit{15})};
    for (unsigned i = 0; i < source.instructions.size(); ++i) {
        source.instructions[i].pc = (i / 3) * 32 + 8 + (i % 3) * 8;
    }
    const auto result = gx::lower(source);
    const auto* program = std::get_if<gx::Program>(&result);
    ARMIGO_CHECK(program != nullptr);
    if (program) {
        const auto text = emit(*program);
        ARMIGO_CHECK(text.find("armigo_float_compare(1u, true,") != std::string::npos);
        ARMIGO_CHECK(text.find("const bool") != std::string::npos);
        const auto actual = evaluate(*program);
        ARMIGO_CHECK_EQ(actual[0], 0x3f800000u);
        unsigned comparisons = 0, combiners = 0;
        for (const auto& node : program->nodes()) if (const auto* op = std::get_if<gx::Pure>(&node.operation)) {
            if (op->kind == PureKind::FloatComparePredicate) ++comparisons;
            if (op->kind == PureKind::PredicateAnd) ++combiners;
        }
        ARMIGO_CHECK(comparisons == 1 && combiners == 2);
    }
    const auto graph = gx::make_graph(compare_graph(1, 0, 0, false));
    const auto* graph_program = std::get_if<gx::Program>(&graph);
    ARMIGO_CHECK(graph_program != nullptr);
    if (graph_program) {
        const auto text = emit(*graph_program);
        const auto helper = text.find("inline bool armigo_float_compare");
        ARMIGO_CHECK(helper != std::string::npos);
        ARMIGO_CHECK(text.find("inline bool armigo_float_compare", helper + 1) == std::string::npos);
        ARMIGO_CHECK(text.find("armigo_float_compare(1u, false,") != std::string::npos);
        ARMIGO_CHECK_EQ(text, emit(*graph_program));
    }
    source.instructions.erase(source.instructions.begin() + 4);
    for (unsigned i = 0; i < source.instructions.size(); ++i) {
        source.instructions[i].pc = (i / 3) * 32 + 8 + (i % 3) * 8;
    }
    const auto plain = gx::lower(source);
    const auto* plain_program = std::get_if<gx::Program>(&plain);
    ARMIGO_CHECK(plain_program != nullptr);
    if (plain_program) ARMIGO_CHECK(emit(*plain_program).find("armigo_float_compare") == std::string::npos);
}

}  // namespace armigo::shader::test
