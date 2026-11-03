// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================


#include "pipeline/dispatch.h"

#include <algorithm>
#include <iostream>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
bool valid(const Program& program, IrForm form = IrForm::ssa) {
    auto result = validate(program, form, Phase::construct_ssa);
    if (const auto* error = std::get_if<Diagnostic>(&result)) {
        std::cerr << "Unexpected validation error: " << error->message << '\n';
        return false;
    }
    return true;
}
void rejected(const Program& program, std::string_view message, DiagnosticCode code = DiagnosticCode::invalid_ir) {
    auto result = validate(program, IrForm::ssa, Phase::construct_ssa);
    check(std::holds_alternative<Diagnostic>(result), message);
    if (const auto* error = std::get_if<Diagnostic>(&result)) {
        check(error->phase == Phase::construct_ssa, "Validator retains phase");
        check(error->code == code, message);
    }
}
Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
Program loop_program() {
    Program p;
    p.entry = FunctionId{0};
    p.stage = Stage::fragment;
    p.functions.push_back({8, BlockId{2}, {BlockId{2}, BlockId{0}, BlockId{3}, BlockId{1}}, {}, {}});
    p.values = {{ValueType::word32}, {ValueType::word32}, {ValueType::word32}, {ValueType::predicate}};
    const NumericMode integer{Domain::unsigned_integer, ScalarType::u32};
    p.operations.push_back({{8, InstructionName::mov}, Opcode::copy, Effects::none, {}, {word(0)}, {ValueId{0}}, {}});
    p.operations.push_back({{40, InstructionName::isetp}, Opcode::compare, Effects::none, integer,
        {ValueId{1}, word(4)}, {ValueId{3}}, ComparisonInfo{Compare::less, false}});
    p.operations.push_back({{72, InstructionName::iadd}, Opcode::add, Effects::none, integer,
        {ValueId{1}, word(1)}, {ValueId{2}}, {}});
    IoInfo output;
    output.semantic = IoSemantic::color;
    output.type = {ScalarType::u32, 1};
    output.data = {InputIndex{0}};
    p.operations.push_back({{104, InstructionName::ast}, Opcode::store_io, Effects::write_io, integer,
        {ValueId{1}}, {}, output});
    p.blocks.resize(4);
    for (auto& b : p.blocks) b.function = FunctionId{0};
    p.blocks[2].operations = {OperationId{0}};
    p.blocks[2].terminator = Jump{BlockId{0}};
    p.blocks[0].predecessors = {BlockId{3}, BlockId{2}};
    p.blocks[0].phis = {{{40, InstructionName::iadd}, ValueId{1}, {{BlockId{3}, ValueId{2}}, {BlockId{2}, ValueId{0}}}}};
    p.blocks[0].operations = {OperationId{1}};
    p.blocks[0].terminator = Branch{ValueId{3}, BlockId{3}, BlockId{1}};
    p.blocks[3].predecessors = {BlockId{0}};
    p.blocks[3].operations = {OperationId{2}};
    p.blocks[3].terminator = Jump{BlockId{0}};
    p.blocks[1].predecessors = {BlockId{0}};
    p.blocks[1].operations = {OperationId{3}};
    p.blocks[1].terminator = Return{};
    return p;
}
class SyntheticFacts final : public FactSource {
public:
    mutable bool change = false;
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::constant_buffer_word) {
            return Known<FactValue>{std::uint32_t(change ? 1 : 0), {OriginKind::explicit_option, "synthetic-word"}};
        }
        if (key.kind == FactKind::constant_buffer_size) {
            return Known<FactValue>{true, {OriginKind::explicit_option, "wrong-type"}};
        }
        return Unknown{"Synthetic fact was not supplied"};
    }
};
Status synthetic_emitter(EmitterContext&) { return std::monostate{}; }
Status synthetic_generator(MslContext&, OperationId) { return std::monostate{}; }
static_assert(valid_bindings(std::array{EmitterBinding{InstructionName::mov, synthetic_emitter}}));
static_assert(!valid_bindings(std::array{EmitterBinding{InstructionName::mov, synthetic_emitter},
                                        EmitterBinding{InstructionName::mov, synthetic_emitter}}));
static_assert(!valid_bindings(std::array{MslBinding{Opcode::copy, synthetic_generator},
                                        MslBinding{Opcode::copy, synthetic_generator}}));
static_assert(!valid_bindings(std::array{EmitterBinding{InstructionName::invalid, synthetic_emitter}}));

void test_ir() {
    const auto original = loop_program();
    check(valid(original), "Loop with predecessor-labelled backedge phi validates");
    check(valid(original, IrForm::legal), "Explicit IO store is valid legal IR");
    auto p = original;
    std::swap(p.operations[0], p.operations[2]);
    p.blocks[2].operations = {OperationId{2}};
    p.blocks[3].operations = {OperationId{0}};
    check(valid(p), "Operation arena index has no execution ordering meaning");
    p = original;
    p.blocks[0].phis[0].inputs.pop_back();
    rejected(p, "Phi missing an incoming edge is rejected");
    p = original;
    p.blocks[0].phis[0].inputs[0].predecessor = BlockId{2};
    rejected(p, "Duplicate phi predecessor is rejected");
    p = original;
    p.operations[3].inputs[0] = ValueId{2};
    rejected(p, "Loop-body value does not dominate the exit");
    p = original;
    p.operations[2].inputs[0] = ValueId{2};
    rejected(p, "Self read before definition is rejected");
    p = original;
    p.operations[3].inputs[0] = Undefined{};
    rejected(p, "Live undefined reads fail", DiagnosticCode::undefined_read);
    p = original;
    p.operations[3].effects = Effects::none;
    rejected(p, "IO stores cannot lose their effect");
    p = original;
    std::get<IoInfo>(p.operations[3].info).data = {InputIndex{8}};
    rejected(p, "Metadata input references are checked");
    p = original;
    p.operations[2].outputs[0] = ValueId{0};
    rejected(p, "Duplicate SSA definition is rejected");
    p = original;
    p.blocks[1].operations.push_back(OperationId{2});
    rejected(p, "Operations cannot have two owners");
    p = original;
    p.blocks[0].terminator = Jump{BlockId{99}};
    rejected(p, "Out-of-range block target is rejected");
    p = original;
    p.limits.analysis_steps = 1;
    rejected(p, "Analysis work is bounded", DiagnosticCode::limit_exceeded);
    p = original;
    p.operations[2].numeric = {Domain::floating, ScalarType::f32};
    rejected(p, "Floating mode requires explicit rounding");
    p.operations[2].numeric.rounding = Rounding::nearest_even;
    p.operations[2].numeric.flush_inputs = true;
    p.operations[2].numeric.flush_output = true;
    check(valid(p), "Explicit numerical modifiers survive validation");
    p = original;
    p.values.push_back({ValueType::predicate});
    p.operations[2].opcode = Opcode::add_carry;
    p.operations[2].inputs.push_back(LiteralBits{0, ValueType::predicate});
    p.operations[2].outputs.push_back(ValueId{4});
    check(valid(p), "Multiple result carry operation is representable");
    p.values[4].type = ValueType::word32;
    rejected(p, "Carry output must be a predicate");
    p = original;
    p.operations[2].inputs[0] = LiteralBits{1, ValueType::predicate};
    rejected(p, "Arithmetic cannot consume predicate storage");
    p = original;
    p.operations[3].inputs.push_back(word(9));
    rejected(p, "Inputs without metadata roles are rejected");
    p = original;
    p.operations[2].opcode = Opcode::bitcast;
    p.operations[2].inputs.resize(1);
    p.operations[2].info = ConversionInfo{ScalarType::u32, ScalarType::u16};
    p.operations[2].numeric = {};
    rejected(p, "Bitcast cannot narrow semantic width within a word container");
    p = original;
    p.operations[2].numeric = {Domain::bits, ScalarType::f32};
    rejected(p, "Raw-bits domain cannot disguise floating arithmetic");
    p = original;
    p.operations[2].numeric = {Domain::floating, ScalarType::u32, Rounding::nearest_even};
    rejected(p, "Floating domain cannot carry an integer scalar");
    p = original;
    p.operations[2].numeric.rounding = Rounding::toward_zero;
    rejected(p, "Nonfloating arithmetic cannot carry rounding policy");
    p = original;
    p.operations[2].opcode = Opcode::bit_and;
    p.operations[2].numeric = {Domain::floating, ScalarType::f32, Rounding::nearest_even};
    rejected(p, "Bitwise operations cannot carry floating interpretation");
    p = original;
    p.values.push_back({ValueType::predicate});
    p.operations[2] = {{72, InstructionName::iadd}, Opcode::add, Effects::none,
        {Domain::boolean, ScalarType::boolean}, {ValueId{3}, ValueId{3}}, {ValueId{4}}, {}};
    rejected(p, "Boolean arithmetic is distinct from predicate logic");
    p = original;
    p.operations[2].opcode = Opcode::pack_half2;
    p.operations[2].numeric = {Domain::floating, ScalarType::f32, Rounding::nearest_even};
    p.operations[2].info = ConversionInfo{};
    rejected(p, "Packed half conversion cannot use default integer metadata");
    p.operations[2].info = ConversionInfo{ScalarType::f32, ScalarType::f16};
    check(valid(p), "Explicit half packing contract validates");
    p = original;
    p.operations[2] = {{72, InstructionName::ldl}, Opcode::load, Effects::read_memory, {}, {}, {ValueId{2}}, MemoryInfo{}};
    auto& local = std::get<MemoryInfo>(p.operations[2].info);
    local.space = AddressSpace::local;
    check(valid(p), "Explicit local word load is structurally valid");
    local.resource = ResourceId{0};
    rejected(p, "Local memory cannot pretend to be an external buffer resource");
    local.resource.reset();
    p.values[2].type = ValueType::word64;
    rejected(p, "Memory result width must agree with its element size");
    p = original;
    p.dependencies = {
        {{FactKind::constant_buffer_word, 1, 4}, std::uint32_t{0}, {OriginKind::explicit_option, "synthetic-one"}},
        {{FactKind::constant_buffer_word, 0, 4}, std::uint32_t{0}, {OriginKind::explicit_option, "synthetic-zero"}}};
    rejected(p, "Program dependency snapshot must already be canonical");
    std::reverse(p.dependencies.begin(), p.dependencies.end());
    check(valid(p), "Canonical dependency snapshot validates");
    p = original;
    p.values[0].kind = ValueKind::mutable_local;
    rejected(p, "Mutable post-phi locals cannot masquerade as SSA");
    p = original;
    p.values.push_back({ValueType::word32});
    TextureInfo sample;
    sample.coordinates = {InputIndex{0}, InputIndex{1}};
    const NumericMode floating{Domain::floating, ScalarType::f32, Rounding::nearest_even};
    p.operations.push_back({{88, InstructionName::tex}, Opcode::texture_sample,
        Effects::texture | Effects::derivative, floating, {word(0), word(0)}, {ValueId{4}}, sample});
    p.blocks[3].operations.insert(p.blocks[3].operations.begin(), OperationId{4});
    check(valid(p), "Implicit sample preserves derivative sensitivity");
    std::get<TextureInfo>(p.operations[4].info).data = {InputIndex{0}};
    rejected(p, "Sample operations cannot carry image-store data");
    std::get<TextureInfo>(p.operations[4].info).data.clear();
    p.operations[4].effects = Effects::texture;
    rejected(p, "Implicit samples cannot drop derivative effect");

}
void test_facts_and_decode() {
    SyntheticFacts source;
    FactReader facts(source);
    auto known = facts.require(FactQuery<FactKind::constant_buffer_word>{1, 16}, Phase::decode,
                               {8, InstructionName::brx});
    check(std::get_if<std::uint32_t>(&known) && std::get<std::uint32_t>(known) == 0, "Known zero stays a real fact");
    check(facts.dependencies().size() == 1, "Consumed fact records dependency");
    (void)facts.require(FactQuery<FactKind::constant_buffer_word>{1, 16}, Phase::decode);
    check(facts.dependencies().size() == 1, "Repeated consumption is deduplicated");
    (void)facts.require(FactQuery<FactKind::constant_buffer_word>{0, 4}, Phase::decode);
    check(facts.dependencies()[0].key.index == 0, "Dependencies have canonical key ordering");
    auto unknown = facts.require(FactQuery<FactKind::texture_descriptor>{1, 32}, Phase::legalize,
                                 {72, InstructionName::tex});
    check(std::holds_alternative<Diagnostic>(unknown), "Unknown fact is a diagnostic");
    if (auto* d = std::get_if<Diagnostic>(&unknown)) {
        check(d->code == DiagnosticCode::missing_fact && d->phase == Phase::legalize && d->pc == 72 &&
              d->instruction == InstructionName::tex && d->missing_fact.has_value(), "Missing-fact diagnostic retains context");
    }
    check(facts.dependencies().size() == 2, "Unknown has no invented dependency value");
    auto wrong = facts.require(FactQuery<FactKind::constant_buffer_size>{1}, Phase::legalize);
    check(std::holds_alternative<Diagnostic>(wrong), "Wrong typed fact is rejected");
    check(facts.dependencies().size() == 2, "Invalid type is not recorded");
    auto merged = merge_dependencies(facts.dependencies(), facts.dependencies(), {}, Phase::legalize);
    check(std::get<std::vector<Dependency>>(merged).size() == 2, "Phase dependency union is canonical and deduplicated");
    std::vector<Dependency> changed(facts.dependencies().begin(), facts.dependencies().end());
    changed[0].value = std::uint32_t{999};
    auto bad_union = merge_dependencies(facts.dependencies(), changed, {}, Phase::legalize);
    check(std::get<Diagnostic>(bad_union).code == DiagnosticCode::conflicting_fact, "Phase handoff rejects a conflicting fact");
    auto invalid_key = facts.require(FactQuery<FactKind::constant_buffer_word>{18, 16}, Phase::decode);
    check(std::get<Diagnostic>(invalid_key).code == DiagnosticCode::invalid_fact, "Fact key validates CB slot bounds");
    auto interpolation = fact_query::interpolation(3, 2);
    check(interpolation.key().index == 3 && interpolation.key().component == 2, "Interpolation keys preserve components");
    source.change = true;
    auto conflict = facts.require(FactQuery<FactKind::constant_buffer_word>{1, 16}, Phase::decode);
    check(std::get<Diagnostic>(conflict).code == DiagnosticCode::conflicting_fact, "Fact mutation is rejected");
    DecodedInstruction instruction;
    instruction.pc = 8;
    instruction.name = InstructionName::fadd;
    check(std::holds_alternative<Diagnostic>(decoded_field(instruction, DecodedField::result_scale)), "Missing decoded field cannot silently default");
    check(std::holds_alternative<Diagnostic>(decoded_source(instruction, SourceRole::a)), "Missing decoded source role fails");
    instruction.sources.push_back({SourceRole::a, Register{RegisterFile::gpr, 3}, {}});
    check(std::holds_alternative<DecodedSource>(decoded_source(instruction, SourceRole::a)), "Decoded source role is checked and retrievable");
    instruction.sources.push_back(instruction.sources[0]);
    check(std::holds_alternative<Diagnostic>(decoded_source(instruction, SourceRole::a)), "Duplicate decoded source role fails");
    instruction.fields.push_back({DecodedField::result_scale, 0});
    check(std::get<std::uint64_t>(decoded_field(instruction, DecodedField::result_scale)) == 0, "Decoded zero field is distinguishable from absence");
}
}
int main() {
    test_ir();
    test_facts_and_decode();
    if (failures) return 1;
    std::cout << "Shader pipeline contracts passed\n";
    return 0;
}
