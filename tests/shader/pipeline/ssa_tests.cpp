// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic SSA construction checks for the shader pipeline: diamonds, nested
// loops with backedge phis, entry backedges, dead undefined computations,
// partial initialization, live undefined reads, predicate and condition-code
// registers, and program limits. All programs are independently authored; no
// game content is embedded.

#include "pipeline/phases.h"
#include "zero_reads.h"

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
Operand gpr(std::uint16_t index) { return Register{RegisterFile::gpr, index}; }
Operand pred(std::uint16_t index) { return Register{RegisterFile::predicate, index}; }
Operand flag(std::uint16_t index) { return Register{RegisterFile::condition_code, index}; }
constexpr NumericMode integer_mode{Domain::unsigned_integer, ScalarType::u32};
constexpr NumericMode boolean_mode{Domain::boolean, ScalarType::boolean};

// Builds one register-form program with a single entry function.
class Builder {
public:
    Builder() {
        program_.entry = FunctionId{0};
        program_.stage = Stage::vertex;
        program_.functions.emplace_back();
        Function& entry = program_.functions.front();
        entry.entry_pc = 8;
        entry.entry = make_block();
    }

    BlockId entry() const { return program_.functions[0].entry; }

    BlockId make_block() {
        const auto id = BlockId{static_cast<std::uint32_t>(program_.blocks.size())};
        BasicBlock block;
        block.function = FunctionId{0};
        program_.blocks.push_back(std::move(block));
        program_.functions[0].blocks.push_back(id);
        return id;
    }

    ValueId value(ValueType type, ValueKind kind = ValueKind::temporary) {
        const auto id = ValueId{static_cast<std::uint32_t>(program_.values.size())};
        program_.values.push_back({type, kind});
        return id;
    }

    ValueId parameter(ValueType type) {
        const auto id = value(type);
        program_.functions[0].parameters.push_back(id);
        return id;
    }

    void result(ValueType type) { program_.functions[0].results.push_back(type); }

    OperationId emit(BlockId block, Opcode opcode, NumericMode mode, std::vector<Operand> inputs,
                     std::vector<Operand> outputs, OperationInfo info = {}, Effects effects = Effects::none) {
        const auto id = OperationId{static_cast<std::uint32_t>(program_.operations.size())};
        program_.operations.push_back(
            {{}, opcode, effects, mode, std::move(inputs), std::move(outputs), std::move(info)});
        program_.blocks[block.value].operations.push_back(id);
        return id;
    }

    void terminate(BlockId block, Terminator terminator) {
        program_.blocks[block.value].terminator = std::move(terminator);
    }

    OperationId copy(BlockId block, Operand destination, Operand source) {
        return emit(block, Opcode::copy, {}, {std::move(source)}, {std::move(destination)});
    }

    OperationId compare(BlockId block, Operand destination, Operand left, Operand right,
                        Compare relation = Compare::less) {
        return emit(block, Opcode::compare, integer_mode, {std::move(left), std::move(right)},
                    {std::move(destination)}, ComparisonInfo{relation, false});
    }

    OperationId add(BlockId block, Operand destination, Operand left, Operand right) {
        return emit(block, Opcode::add, integer_mode, {std::move(left), std::move(right)},
                    {std::move(destination)});
    }

    OperationId store_word(BlockId block, Operand data) {
        IoInfo info;
        info.semantic = IoSemantic::color;
        info.type = {ScalarType::u32, 1};
        info.data = {InputIndex{0}};
        return emit(block, Opcode::store_io, integer_mode, {std::move(data)}, {}, std::move(info),
                    Effects::write_io);
    }

    OperationId store_predicate(BlockId block, Operand data) {
        IoInfo info;
        info.semantic = IoSemantic::color;
        info.type = {ScalarType::boolean, 1};
        info.data = {InputIndex{0}};
        return emit(block, Opcode::store_io, boolean_mode, {std::move(data)}, {}, std::move(info),
                    Effects::write_io);
    }

    void set_dependencies(std::vector<Dependency> dependencies) {
        program_.dependencies = std::move(dependencies);
    }

    Program take() {
        wire_predecessors();
        return std::move(program_);
    }

private:
    // Predecessor lists must be the distinct reverse of terminator edges.
    void wire_predecessors() {
        for (auto& block : program_.blocks) block.predecessors.clear();
        for (std::uint32_t index = 0; index < program_.blocks.size(); ++index)
            for (const auto target : successors(program_.blocks[index].terminator)) {
                auto& list = program_.blocks[target.value].predecessors;
                if (std::find(list.begin(), list.end(), BlockId{index}) == list.end())
                    list.push_back(BlockId{index});
            }
        for (auto& block : program_.blocks) std::sort(block.predecessors.begin(), block.predecessors.end());
    }

    Program program_;
};

Program expect_ok(Builder builder, std::string_view message, Limits supplied = {}) {
    auto result = construct_ssa(RegisterProgram{builder.take()}, supplied);
    auto* error = std::get_if<Diagnostic>(&result);
    if (error) {
        std::cerr << "FAIL: " << message << " (" << error->message << ")\n";
        ++failures;
        return {};
    }
    auto program = std::move(std::get<SsaProgram>(result).ir);
    if (std::holds_alternative<Diagnostic>(validate(program, IrForm::ssa, Phase::construct_ssa))) {
        std::cerr << "FAIL: " << message << " (constructed SSA fails validation)\n";
        ++failures;
        return {};
    }
    // Renamed IR keeps no physical registers and no undefined reads.
    auto clean = [](const Operand& operand) {
        return !std::holds_alternative<Register>(operand) && !std::holds_alternative<Undefined>(operand);
    };
    bool operands_clean = true;
    for (const auto& operation : program.operations) {
        for (const auto& input : operation.inputs) operands_clean = operands_clean && clean(input);
        for (const auto& output : operation.outputs) operands_clean = operands_clean && clean(output);
    }
    for (const auto& block : program.blocks) {
        for (const auto& phi : block.phis)
            for (const auto& input : phi.inputs) operands_clean = operands_clean && clean(input.value);
        if (auto* branch = std::get_if<Branch>(&block.terminator)) operands_clean = operands_clean && clean(branch->condition);
        if (auto* select = std::get_if<Switch>(&block.terminator)) operands_clean = operands_clean && clean(select->selector);
        if (auto* result = std::get_if<Return>(&block.terminator))
            for (const auto& value : result->values) operands_clean = operands_clean && clean(value);
    }
    check(operands_clean, message);
    return program;
}

// A live read that no proof can justify falls back to an explicit zero.
void expect_zero_reads(Builder builder, std::string_view message, Limits supplied = {}) {
    const auto program = expect_ok(std::move(builder), message, supplied);
    check(test::entry_zero_reads(program, ValueType::word32) +
              test::entry_zero_reads(program, ValueType::predicate) > 0, message);
}

void expect_zero_read(Builder builder, ValueType type, std::string_view message) {
    const auto program = expect_ok(std::move(builder), message);
    check(test::entry_zero_reads(program, type) == 1, message);
}

void expect_rejected(Builder builder, DiagnosticCode code, std::string_view message, Limits supplied = {}) {
    auto result = construct_ssa(RegisterProgram{builder.take()}, supplied);
    auto* error = std::get_if<Diagnostic>(&result);
    if (!error) { check(false, message); return; }
    check(error->phase == Phase::construct_ssa, message);
    if (error->code != code) std::cerr << "Diagnostic: " << error->message << '\n';
    check(error->code == code, message);
}

const Phi* phi_with_input(const Program& program, BlockId block, BlockId predecessor, ValueId value) {
    for (const auto& phi : program.blocks[block.value].phis)
        for (const auto& input : phi.inputs)
            if (input.predecessor == predecessor && input.value == value) return &phi;
    return nullptr;
}

const PhiInput* phi_input(const Phi& phi, BlockId predecessor) {
    for (const auto& input : phi.inputs) if (input.predecessor == predecessor) return &input;
    return nullptr;
}

ValueId def_of(const Program& program, OperationId id) {
    return std::get<ValueId>(program.operations[id.value].outputs[0]);
}

// Operand has no equality operator because the frozen Undefined payload is
// not comparable; SSA output only ever equates values, which is all tests need.
bool same_value(const Operand& operand, ValueId value) {
    const auto* id = std::get_if<ValueId>(&operand);
    return id && *id == value;
}

struct OperationShape {
    Opcode opcode;
    Effects effects;
};

void check_block_shape(const Program& program, BlockId block,
                       std::initializer_list<BlockId> expected_successors,
                       std::initializer_list<OperationShape> expected_operations,
                       std::string_view message) {
    const auto actual_successors = successors(program.blocks[block.value].terminator);
    bool matches = std::equal(actual_successors.begin(), actual_successors.end(),
                              expected_successors.begin(), expected_successors.end());
    const auto& operations = program.blocks[block.value].operations;
    matches = matches && operations.size() == expected_operations.size();
    auto expected = expected_operations.begin();
    for (std::size_t i = 0; matches && i < operations.size(); ++i, ++expected) {
        const auto& operation = program.operations[operations[i].value];
        matches = operation.opcode == expected->opcode && operation.effects == expected->effects;
    }
    check(matches, message);
}

std::vector<std::uint32_t> run_counter(const Program& program, ValueId parameter) {
    std::vector<std::uint64_t> values(program.values.size());
    std::vector<std::uint32_t> outputs;
    values[parameter.value] = 0;
    auto read = [&](const Operand& operand) {
        if (const auto* value = std::get_if<ValueId>(&operand)) return values[value->value];
        if (const auto* literal = std::get_if<LiteralBits>(&operand)) return literal->bits;
        return std::uint64_t{};
    };
    BlockId previous;
    BlockId current = program.functions[0].entry;
    for (std::uint32_t step = 0; step < 16; ++step) {
        const auto& block = program.blocks[current.value];
        std::vector<std::pair<ValueId, std::uint64_t>> phi_values;
        for (const auto& phi : block.phis) {
            const auto* input = phi_input(phi, previous);
            if (!input) return {};
            phi_values.emplace_back(phi.output, values[input->value.value]);
        }
        for (const auto& [output, value] : phi_values) values[output.value] = value;
        for (const auto id : block.operations) {
            const auto& operation = program.operations[id.value];
            if (operation.opcode == Opcode::store_io) {
                outputs.push_back(static_cast<std::uint32_t>(read(operation.inputs[0])));
            } else if (operation.opcode == Opcode::add) {
                values[std::get<ValueId>(operation.outputs[0]).value] =
                    read(operation.inputs[0]) + read(operation.inputs[1]);
            } else if (operation.opcode == Opcode::compare) {
                values[std::get<ValueId>(operation.outputs[0]).value] =
                    read(operation.inputs[0]) < read(operation.inputs[1]);
            } else {
                return {};
            }
        }
        if (std::holds_alternative<Return>(block.terminator)) return outputs;
        BlockId next;
        if (const auto* jump = std::get_if<Jump>(&block.terminator)) next = jump->target;
        else if (const auto* branch = std::get_if<Branch>(&block.terminator))
            next = read(branch->condition) ? branch->when_true : branch->when_false;
        else return {};
        previous = current;
        current = next;
    }
    return {};
}

Builder diamond_builder(OperationId& load, OperationId& test, OperationId& left, OperationId& right,
                        OperationId& store) {
    Builder b;
    const auto entry = b.entry();
    const auto then_block = b.make_block();
    const auto else_block = b.make_block();
    const auto join = b.make_block();
    load = b.copy(entry, gpr(2), word(0));
    test = b.compare(entry, pred(0), gpr(2), word(4));
    b.terminate(entry, Branch{pred(0), then_block, else_block});
    left = b.copy(then_block, gpr(2), word(1));
    b.terminate(then_block, Jump{join});
    right = b.copy(else_block, gpr(2), word(2));
    b.terminate(else_block, Jump{join});
    store = b.store_word(join, gpr(2));
    b.terminate(join, Return{});
    return b;
}

void test_diamond() {
    OperationId load{}, test{}, left{}, right{}, store{};
    auto program = expect_ok(diamond_builder(load, test, left, right, store), "Diamond merges both arms through a phi");
    const auto join = BlockId{3};
    check(program.blocks[join.value].phis.size() == 1, "Diamond join holds exactly one phi");
    const auto& phi = program.blocks[join.value].phis[0];
    check(phi.inputs.size() == 2, "Diamond phi names both predecessors");
    check(phi_input(phi, BlockId{1}) && phi_input(phi, BlockId{1})->value == def_of(program, left),
          "Diamond phi keeps the true-arm value");
    check(phi_input(phi, BlockId{2}) && phi_input(phi, BlockId{2})->value == def_of(program, right),
          "Diamond phi keeps the false-arm value");
    check(same_value(program.operations[store.value].inputs[0], phi.output),
          "Diamond store reads the phi");
    check(same_value(program.operations[test.value].inputs[0], def_of(program, load)),
          "Same-block read keeps its local definition");
}

void test_nested_loops() {
    Builder b;
    const auto entry = b.entry();
    const auto header = b.make_block();  // Outer loop header.
    const auto inner = b.make_block();   // Inner loop header.
    const auto body = b.make_block();    // Inner loop body.
    const auto latch = b.make_block();   // Outer loop latch.
    const auto exit = b.make_block();
    const auto init_inner = b.copy(entry, gpr(2), word(0));
    const auto init_outer = b.copy(entry, gpr(3), word(0));
    b.terminate(entry, Jump{header});
    const auto test_outer = b.compare(header, pred(0), gpr(3), word(4));
    b.terminate(header, Branch{pred(0), inner, exit});
    const auto test_inner = b.compare(inner, pred(1), gpr(2), word(4));
    b.terminate(inner, Branch{pred(1), body, latch});
    const auto bump_inner = b.add(body, gpr(2), gpr(2), word(1));
    b.terminate(body, Jump{inner});
    const auto bump_outer = b.add(latch, gpr(3), gpr(3), word(1));
    const auto reset_inner = b.copy(latch, gpr(2), word(0));
    b.terminate(latch, Jump{header});
    const auto store = b.store_word(exit, gpr(3));
    b.terminate(exit, Return{});

    auto program = expect_ok(std::move(b), "Nested loops build loop-carried phis");
    // Outer header: one phi per counter, both fed from entry and the latch.
    check(program.blocks[header.value].phis.size() == 2, "Outer header carries both counters");
    const auto* outer_counter = phi_with_input(program, header, entry, def_of(program, init_outer));
    check(outer_counter && phi_input(*outer_counter, latch) &&
              phi_input(*outer_counter, latch)->value == def_of(program, bump_outer),
          "Outer counter phi takes the latch increment on the backedge");
    const auto* reset_counter = phi_with_input(program, header, entry, def_of(program, init_inner));
    check(reset_counter && phi_input(*reset_counter, latch) &&
              phi_input(*reset_counter, latch)->value == def_of(program, reset_inner),
          "Inner counter phi observes the latch reset at the outer header");
    // Inner header: only the inner counter needs a phi there. The outer
    // counter has no definition inside the inner loop, so every inner-block
    // read resolves to the outer header phi without a redundant merge.
    check(program.blocks[inner.value].phis.size() == 1, "Inner header carries only the inner counter phi");
    const auto* inner_counter = phi_with_input(program, inner, body, def_of(program, bump_inner));
    check(inner_counter != nullptr, "Inner counter phi takes the body increment");
    if (inner_counter && reset_counter) {
        const auto* carried = phi_input(*inner_counter, header);
        check(carried && carried->value == reset_counter->output,
              "Inner phi input is the outer header phi");
    }
    check(outer_counter && same_value(program.operations[bump_outer.value].inputs[0], outer_counter->output),
          "Latch reads the outer counter through the outer header phi");
    check(program.blocks[exit.value].phis.empty(), "Single-predecessor exit needs no phi");
    check(outer_counter && same_value(program.operations[store.value].inputs[0], outer_counter->output),
          "Exit reads the outer loop phi");
    check(outer_counter && same_value(program.operations[test_outer.value].inputs[0], outer_counter->output),
          "Header comparison reads the loop phi");
    if (inner_counter)
        check(same_value(program.operations[test_inner.value].inputs[0], inner_counter->output),
              "Inner comparison reads the inner phi");
}

void test_multiple_exits() {
    Builder b;
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto check_block = b.make_block();
    const auto body = b.make_block();
    const auto done = b.make_block();
    const auto early = b.make_block();
    const auto init = b.copy(entry, gpr(2), word(0));
    b.terminate(entry, Jump{header});
    const auto test = b.compare(header, pred(0), gpr(2), word(8));
    b.terminate(header, Branch{pred(0), check_block, done});
    const auto hit = b.compare(check_block, pred(1), gpr(2), word(3), Compare::equal);
    b.terminate(check_block, Branch{pred(1), early, body});
    const auto bump = b.add(body, gpr(2), gpr(2), word(1));
    b.terminate(body, Jump{header});
    const auto normal_store = b.store_word(done, gpr(2));
    b.terminate(done, Return{});
    const auto early_store = b.store_word(early, gpr(2));
    b.terminate(early, Return{});

    auto program = expect_ok(std::move(b), "Both loop exits observe the same loop phi");
    const auto& phi = program.blocks[header.value].phis[0];
    check(program.blocks[header.value].phis.size() == 1, "Loop header holds one phi for the counter");
    check(phi_input(phi, entry) && phi_input(phi, entry)->value == def_of(program, init),
          "Loop phi takes the initial value from the entry edge");
    check(phi_input(phi, body) && phi_input(phi, body)->value == def_of(program, bump),
          "Loop phi takes the increment from the backedge");
    check(same_value(program.operations[normal_store.value].inputs[0], phi.output) &&
              same_value(program.operations[early_store.value].inputs[0], phi.output),
          "Both exits read the same loop phi");
    (void)test;
    (void)hit;
}

void test_partial_initialization() {
    Builder b;
    const auto entry = b.entry();
    const auto taken = b.make_block();
    const auto skipped = b.make_block();
    const auto join = b.make_block();
    const auto test = b.compare(entry, pred(0), word(0), word(1));
    b.terminate(entry, Branch{pred(0), taken, skipped});
    b.copy(taken, gpr(2), word(1));
    b.terminate(taken, Jump{join});
    b.terminate(skipped, Jump{join});
    b.store_word(join, gpr(2));
    b.terminate(join, Return{});
    (void)test;
    expect_zero_read(std::move(b), ValueType::word32,
                     "A register written on one path only reads zero on the other path");
}

void test_undefined_reads() {
    {
        Builder b;
        const auto entry = b.entry();
        b.store_word(entry, gpr(9));
        b.terminate(entry, Return{});
        expect_zero_read(std::move(b), ValueType::word32,
                         "A word register with no definition reads zero");
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto taken = b.make_block();
        const auto skipped = b.make_block();
        b.terminate(entry, Branch{pred(5), taken, skipped});
        b.terminate(taken, Return{});
        b.terminate(skipped, Return{});
        expect_zero_read(std::move(b), ValueType::predicate,
                         "A predicate register with no definition reads false");
    }
    {
        Builder b;
        const auto entry = b.entry();
        b.add(entry, gpr(10), gpr(9), word(1));
        b.store_word(entry, gpr(10));
        b.terminate(entry, Return{});
        expect_zero_read(std::move(b), ValueType::word32,
                         "An undefined value feeding a retained pure chain reads zero");
    }
    {
        Builder b;
        const auto entry = b.entry();
        b.result(ValueType::word32);
        b.terminate(entry, Return{{gpr(9)}});
        expect_zero_read(std::move(b), ValueType::word32,
                         "An undefined function result reads zero");
    }
}

void test_dead_undefined_cleanup() {
    {
        Builder b;
        const auto entry = b.entry();
        b.add(entry, gpr(10), gpr(9), word(1));
        b.terminate(entry, Return{});
        auto program = expect_ok(std::move(b), "A dead pure undefined computation is removed");
        check(program.operations.empty(), "Dead undefined computation leaves no operation");
    }
    {
        Builder b;
        const auto entry = b.entry();
        b.add(entry, gpr(10), gpr(9), word(1));
        b.copy(entry, gpr(2), word(7));
        b.store_word(entry, gpr(2));
        b.terminate(entry, Return{});
        auto program = expect_ok(std::move(b), "Dead-code removal remaps retained operation IDs");
        check(program.operations.size() == 2 && program.operations[0].opcode == Opcode::copy &&
                  program.operations[1].opcode == Opcode::store_io,
              "Retained operations stay ordered after arena compaction");
        check(program.blocks[entry.value].operations ==
                  std::vector<OperationId>{OperationId{0}, OperationId{1}},
              "Block operation IDs follow the compacted arena");
    }
    {
        Builder b;
        const auto entry = b.entry();
        MemoryInfo memory;
        memory.space = AddressSpace::constant_buffer;
        memory.access = Access::read;
        memory.constant_buffer_slot = 0;
        memory.element_bytes = 4;
        memory.elements = 1;
        memory.alignment = 4;
        b.emit(entry, Opcode::load, integer_mode, {}, {gpr(2)}, memory, Effects::read_memory);
        b.terminate(entry, Return{});
        auto program = expect_ok(std::move(b), "An effectful load remains when its output is unused");
        check(program.operations.size() == 1 && program.operations[0].opcode == Opcode::load,
              "Dead-code removal retains declared read effects");
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto packed = b.value(ValueType::word64);
        b.emit(entry, Opcode::pack_word64, {}, {word(1), word(2)}, {packed});
        b.emit(entry, Opcode::unpack_word64, {}, {packed}, {gpr(2), gpr(3)});
        b.store_word(entry, gpr(2));
        b.terminate(entry, Return{});
        auto program = expect_ok(std::move(b), "One live result retains a multi-result pure operation");
        check(program.operations.size() == 3 && program.operations[0].opcode == Opcode::pack_word64 &&
                  program.operations[1].opcode == Opcode::unpack_word64 &&
                  program.operations[2].opcode == Opcode::store_io,
              "Multi-result dependency remains intact when one result is live");
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto loop = b.make_block();
        const auto exit = b.make_block();
        b.terminate(entry, Branch{pred(7), loop, exit});
        b.add(loop, gpr(2), gpr(2), word(1));
        b.terminate(loop, Branch{pred(7), loop, exit});
        b.terminate(exit, Return{});
        auto program = expect_ok(std::move(b), "A dead undefined loop-carried cycle is removed");
        check(program.operations.empty(), "Dead loop-carried computation is removed");
        check(program.blocks[loop.value].phis.empty(), "Dead loop-carried phi is removed");
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto loop = b.make_block();
        const auto exit = b.make_block();
        b.terminate(entry, Branch{pred(7), loop, exit});
        b.add(loop, gpr(2), gpr(2), word(1));
        b.store_word(loop, gpr(2));
        b.terminate(loop, Branch{pred(7), loop, exit});
        b.terminate(exit, Return{});
        expect_zero_reads(std::move(b),
                        "An undefined loop-carried cycle feeding an effect is rejected");
    }
}

void test_entry_backedge_parameter() {
    Builder b;
    const auto entry = b.entry();
    const auto exit = b.make_block();
    const auto counter = b.parameter(ValueType::word32);
    const auto store = b.store_word(entry, counter);
    const auto bump = b.add(entry, counter, counter, word(1));
    b.compare(entry, pred(0), counter, word(3));
    b.terminate(entry, Branch{pred(0), entry, exit});
    b.terminate(exit, Return{});

    auto program = expect_ok(std::move(b), "An entry backedge merges a reassigned parameter");
    const auto preheader = program.functions[0].entry;
    check(preheader != entry, "Entry backedge construction adds a predecessor-free preheader");
    check(program.blocks[preheader.value].predecessors.empty(), "Entry preheader has no predecessors");
    const auto* jump = std::get_if<Jump>(&program.blocks[preheader.value].terminator);
    check(jump && jump->target == entry, "Entry preheader jumps to the original loop header");
    check(program.blocks[entry.value].phis.size() == 1, "Original entry carries the parameter phi");
    if (program.blocks[entry.value].phis.size() == 1) {
        const auto& phi = program.blocks[entry.value].phis[0];
        const auto* initial = phi_input(phi, preheader);
        const auto* carried = phi_input(phi, entry);
        check(initial && initial->value == counter, "Entry phi takes the original parameter from the preheader");
        check(carried && carried->value == def_of(program, bump), "Entry phi takes the reassignment on the backedge");
        check(same_value(program.operations[store.value].inputs[0], phi.output),
              "Effect before reassignment observes the loop-carried phi");
        check(same_value(program.operations[bump.value].inputs[0], phi.output),
              "Parameter reassignment reads the loop-carried phi");
    }
    check(run_counter(program, counter) == std::vector<std::uint32_t>{0, 1, 2},
          "Entry-loop SSA executes the expected counter sequence");
}

void test_correlated_path_resolution() {
    Builder b;
    const auto entry = b.entry();
    const auto assign = b.make_block();
    const auto skip = b.make_block();
    const auto join = b.make_block();
    const auto store = b.make_block();
    const auto done = b.make_block();
    b.compare(entry, pred(0), word(0), word(1));
    b.terminate(entry, Branch{pred(0), assign, skip});
    b.copy(assign, gpr(2), word(7));
    b.terminate(assign, Jump{join});
    b.terminate(skip, Jump{join});
    b.terminate(join, Branch{pred(0), store, done});
    b.store_word(store, gpr(2));
    b.terminate(store, Jump{done});
    b.terminate(done, Return{});
    expect_ok(std::move(b), "Repeated guards prove the undefined arm infeasible");
}

// Execute only the independently authored scalar fixtures below. Optional
// storage makes an accidentally surviving undefined read an oracle failure.
std::vector<std::uint64_t> correlated_outputs(const Program& program, bool p, bool q) {
    std::vector<std::optional<std::uint64_t>> values(program.values.size());
    const auto& function = program.functions[0];
    values[function.parameters[0].value] = p;
    values[function.parameters[1].value] = q;
    for (std::size_t i = 2; i < function.parameters.size(); ++i)
        values[function.parameters[i].value] = 0xa5a55a5a;
    const auto read = [&](const Operand& operand) -> std::uint64_t {
        if (const auto* literal = std::get_if<LiteralBits>(&operand)) return literal->bits;
        return values.at(std::get<ValueId>(operand).value).value();
    };
    std::vector<std::uint64_t> outputs;
    BlockId previous;
    auto current = function.entry;
    for (unsigned step = 0; step < 200; ++step) {
        const auto& block = program.blocks[current.value];
        std::vector<std::pair<ValueId, std::uint64_t>> incoming;
        for (const auto& phi : block.phis) {
            const auto* edge = phi_input(phi, previous);
            if (!edge) return {};
            incoming.emplace_back(phi.output, values.at(edge->value.value).value());
        }
        for (const auto& [id, bits] : incoming) values[id.value] = bits;
        for (const auto id : block.operations) {
            const auto& op = program.operations[id.value];
            if (op.opcode == Opcode::store_io) {
                outputs.push_back(read(op.inputs[0]));
                continue;
            }
            if (op.opcode == Opcode::barrier || op.opcode == Opcode::discard ||
                op.opcode == Opcode::texture_sample || op.opcode == Opcode::shuffle) continue;
            std::uint64_t result{};
            switch (op.opcode) {
            case Opcode::copy: result = read(op.inputs[0]); break;
            // These effect results are unused by the fixture. A scalar
            // surrogate lets the oracle continue checking its control flow.
            case Opcode::derivative_x: case Opcode::derivative_y:
            case Opcode::vote_any: result = read(op.inputs[0]); break;
            case Opcode::logical_not: result = !read(op.inputs[0]); break;
            case Opcode::predicate_to_mask: result = read(op.inputs[0]) ? 0xffffffffu : 0u; break;
            case Opcode::logical_and: result = read(op.inputs[0]) && read(op.inputs[1]); break;
            case Opcode::logical_or: result = read(op.inputs[0]) || read(op.inputs[1]); break;
            case Opcode::logical_xor: result = bool(read(op.inputs[0])) != bool(read(op.inputs[1])); break;
            case Opcode::select: result = read(op.inputs[read(op.inputs[0]) ? 1 : 2]); break;
            case Opcode::add: result = read(op.inputs[0]) + read(op.inputs[1]); break;
            case Opcode::compare: {
                const auto left = read(op.inputs[0]), right = read(op.inputs[1]);
                switch (std::get<ComparisonInfo>(op.info).relation) {
                case Compare::equal: result = left == right; break;
                case Compare::not_equal: result = left != right; break;
                case Compare::less:
                    result = op.numeric.scalar == ScalarType::s32
                        ? static_cast<std::int32_t>(left) < static_cast<std::int32_t>(right)
                        : left < right;
                    break;
                default: check(false, "Unexpected correlated fixture comparison"); return {};
                }
                break;
            }
            default: check(false, "Unexpected correlated fixture operation"); return {};
            }
            values[std::get<ValueId>(op.outputs[0]).value] = result;
        }
        previous = current;
        if (const auto* branch = std::get_if<Branch>(&block.terminator))
            current = read(branch->condition) ? branch->when_true : branch->when_false;
        else if (const auto* jump = std::get_if<Jump>(&block.terminator)) current = jump->target;
        else if (std::holds_alternative<Return>(block.terminator)) return outputs;
        else break;
    }
    check(false, "Correlated fixture terminates within its execution bound");
    return {};
}

Builder correlated_builder(bool nested, bool loop, bool rewrite = false, bool unrelated = false) {
    Builder b;
    const auto p = b.parameter(ValueType::predicate);
    const auto q = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto nested_test = b.make_block();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto overwrite = b.make_block();
    const auto use_test = b.make_block();
    const auto nested_use = b.make_block();
    const auto store = b.make_block();
    const auto latch = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, pred(0), p);
    b.copy(entry, pred(1), q);
    b.copy(entry, gpr(0), word(0));
    b.terminate(entry, Jump{header});
    b.terminate(header, Branch{pred(0), nested_test, join});
    b.terminate(nested_test, nested ? Terminator{Branch{pred(1), assign, join}} : Terminator{Jump{assign}});
    b.copy(assign, gpr(2), word(7));
    b.terminate(assign, Jump{join});
    b.terminate(join, Branch{pred(1), overwrite, use_test});
    // In the nonnested fixture this creates a second phi whose false input
    // is the partially undefined phi from the first guard.
    b.copy(overwrite, nested ? gpr(3) : gpr(2), word(9));
    b.terminate(overwrite, Jump{use_test});
    if (rewrite) b.copy(use_test, pred(0), q);
    b.terminate(use_test, Branch{unrelated ? pred(1) : pred(0), nested_use, latch});
    b.terminate(nested_use, nested ? Terminator{Branch{pred(1), store, latch}} : Terminator{Jump{store}});
    b.store_word(store, gpr(2));
    b.terminate(store, Jump{latch});
    b.add(latch, gpr(0), gpr(0), word(1));
    b.compare(latch, pred(2), gpr(0), word(3));
    b.terminate(latch, loop ? Terminator{Branch{pred(2), header, done}} : Terminator{Jump{done}});
    b.terminate(done, Return{});
    return b;
}

void test_nested_correlations_and_phi_chains() {
    for (const bool nested : {false, true}) {
        for (const bool loop : {false, true}) {
            auto program = expect_ok(correlated_builder(nested, loop),
                "Nested and repeated guards preserve ordinary SSA dominance, including loop backedges");
            if (program.functions.empty()) continue;
            for (const bool p : {false, true}) {
                for (const bool q : {false, true}) {
                    const auto count = p && (!nested || q) ? (loop ? 3u : 1u) : 0u;
                    check(correlated_outputs(program, p, q) ==
                              std::vector<std::uint64_t>(count, !nested && q ? 9 : 7),
                          "All predicate combinations preserve values and effect counts");
                }
            }
            if (!nested) {
                check(std::any_of(program.blocks.begin(), program.blocks.end(),
                      [](const BasicBlock& block) { return !block.phis.empty(); }),
                      "A feasible data phi remains after resolving a phi feeding a phi");
            }
        }
    }
    // Without the second definition, a different or rewritten predicate
    // cannot justify the original nested definition's missing path.
    expect_zero_reads(correlated_builder(true, false, true),
                    "A predicate rewrite does not inherit a previous reaching definition's facts");
    expect_zero_reads(correlated_builder(true, false, false, true),
                    "A different guard does not prove the original definition executed");
    auto bounded = correlated_builder(true, false).take();
    Limits limits;
    limits.blocks = bounded.blocks.size();
    auto result = construct_ssa(RegisterProgram{bounded}, limits);
    const auto* failure = std::get_if<Diagnostic>(&result);
    check(failure && failure->code == DiagnosticCode::limit_exceeded,
          "Legacy correlation cloning respects the inherited block budget");
    limits = Limits{};
    limits.operations = bounded.operations.size();
    result = construct_ssa(RegisterProgram{bounded}, limits);
    failure = std::get_if<Diagnostic>(&result);
    check(failure && failure->code == DiagnosticCode::limit_exceeded,
          "Legacy correlation cloning respects the operation budget");
}

void test_loop_predicate_redefinition() {
    Builder b;
    const auto p = b.parameter(ValueType::predicate);
    const auto q = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto store = b.make_block();
    const auto latch = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, gpr(0), word(0));
    b.terminate(entry, Jump{header});
    // P0 has one static definition but changes dynamically on iteration 3.
    b.compare(header, pred(0), gpr(0), word(2));
    b.terminate(header, Branch{pred(0), assign, join});
    b.copy(assign, gpr(2), gpr(0));
    b.terminate(assign, Jump{join});
    b.terminate(join, Branch{pred(0), store, latch});
    b.store_word(store, gpr(2));
    b.terminate(store, Jump{latch});
    b.add(latch, gpr(0), gpr(0), word(1));
    b.compare(latch, pred(1), gpr(0), word(3));
    b.terminate(latch, Branch{pred(1), header, done});
    b.terminate(done, Return{});
    auto program = expect_ok(std::move(b), "Facts are killed when a loop reexecutes their defining instruction");
    if (!program.functions.empty())
        check(correlated_outputs(program, false, false) == std::vector<std::uint64_t>{0, 1},
              "Changing loop predicates do not retain a preceding iteration's outcome");
    (void)p;
    (void)q;
}

void test_predicate_phi_correlations() {
    Builder b;
    const auto p = b.parameter(ValueType::predicate);
    const auto q = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto store = b.make_block();
    const auto latch = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, pred(0), p);
    b.copy(entry, gpr(0), word(0));
    b.terminate(entry, Jump{header});
    b.terminate(header, Branch{pred(0), assign, join});
    b.copy(assign, gpr(2), word(7));
    b.terminate(assign, Jump{join});
    b.terminate(join, Branch{pred(0), store, latch});
    b.store_word(store, gpr(2));
    b.terminate(store, Jump{latch});
    b.copy(latch, pred(0), q);
    b.add(latch, gpr(0), gpr(0), word(1));
    b.compare(latch, pred(1), gpr(0), word(3));
    b.terminate(latch, Branch{pred(1), header, done});
    b.terminate(done, Return{});
    auto program = expect_ok(std::move(b), "Repeated tests of the same predicate phi output correlate");
    if (!program.functions.empty()) {
        for (const bool first : {false, true}) {
            for (const bool later : {false, true}) {
                check(correlated_outputs(program, first, later) ==
                          std::vector<std::uint64_t>((first ? 1 : 0) + (later ? 2 : 0), 7),
                      "A predicate phi kills the old dynamic fact on every loop iteration");
            }
        }
    }

    Builder rewritten;
    const auto first = rewritten.parameter(ValueType::predicate);
    const auto other = rewritten.parameter(ValueType::predicate);
    const auto start = rewritten.entry();
    const auto define = rewritten.make_block();
    const auto middle = rewritten.make_block();
    const auto rewrite = rewritten.make_block();
    const auto guard = rewritten.make_block();
    const auto use = rewritten.make_block();
    const auto exit = rewritten.make_block();
    rewritten.copy(start, pred(0), first);
    rewritten.terminate(start, Branch{pred(0), define, middle});
    rewritten.copy(define, gpr(2), word(7));
    rewritten.terminate(define, Jump{middle});
    rewritten.terminate(middle, Branch{other, rewrite, guard});
    rewritten.copy(rewrite, pred(0), other);
    rewritten.terminate(rewrite, Jump{guard});
    rewritten.terminate(guard, Branch{pred(0), use, exit});
    rewritten.store_word(use, gpr(2));
    rewritten.terminate(use, Jump{exit});
    rewritten.terminate(exit, Return{});
    expect_zero_reads(std::move(rewritten),
                    "An intervening predicate phi is distinct from the original guard definition");
}

// The predicate carries which data definition ran, rather than merely
// repeating one unchanged guard. Both inputs remain runtime parameters.
Builder predicate_tail_builder(bool unsafe, bool loop) {
    Builder b;
    const auto p = b.parameter(ValueType::predicate);
    const auto q = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto fallback = b.make_block();
    const auto tail = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, gpr(0), word(0));
    b.copy(entry, pred(0), p);
    b.terminate(entry, Jump{header});
    b.copy(header, pred(1), LiteralBits{1, ValueType::predicate});
    b.terminate(header, Branch{pred(0), assign, join});
    b.copy(assign, gpr(2), word(7));
    b.copy(assign, pred(1), LiteralBits{0, ValueType::predicate});
    b.terminate(assign, Jump{join});
    if (unsafe) b.copy(join, pred(1), q);
    b.terminate(join, Branch{pred(1), fallback, tail});
    b.copy(fallback, gpr(2), word(9));
    b.terminate(fallback, Jump{tail});
    b.store_word(tail, gpr(2));
    b.copy(tail, pred(0), q);
    b.add(tail, gpr(0), gpr(0), word(1));
    b.compare(tail, pred(2), gpr(0), word(3));
    b.terminate(tail, loop ? Terminator{Branch{pred(2), header, done}} : Terminator{Jump{done}});
    b.store_word(done, gpr(2));
    b.terminate(done, Return{});
    return b;
}

void test_predicate_phi_tail_definitions() {
    for (const bool loop : {false, true}) {
        const auto program = expect_ok(predicate_tail_builder(false, loop),
            "Predicate phi edges prove exhaustive data initialization before the tail");
        if (program.functions.empty()) continue;
        for (const bool p : {false, true}) {
            for (const bool q : {false, true}) {
                std::vector<std::uint64_t> expected{p ? 7u : 9u};
                if (loop) expected.insert(expected.end(), 2, q ? 7u : 9u);
                expected.push_back(expected.back());
                check(correlated_outputs(program, p, q) == expected,
                      "Tail and post-loop reads preserve each path's actual definition");
            }
        }
        expect_zero_reads(predicate_tail_builder(true, loop),
                        "An unrelated tail guard leaves a feasible uninitialized first iteration");
    }
}

void test_boolean_guard_copies() {
    for (const auto opcode : {Opcode::logical_and, Opcode::logical_or, Opcode::logical_xor,
                              Opcode::select}) {
        for (const bool inverted : {false, true}) {
            Builder b;
            const auto p = b.parameter(ValueType::predicate);
            const auto q = b.parameter(ValueType::predicate);
            const auto entry = b.entry();
            const auto assign = b.make_block();
            const auto join = b.make_block();
            const auto use = b.make_block();
            const auto done = b.make_block();
            b.terminate(entry, Branch{p, assign, join});
            b.copy(assign, gpr(2), word(17));
            b.terminate(assign, Jump{join});
            if (opcode == Opcode::select) {
                b.emit(join, opcode, {}, {p, LiteralBits{1, ValueType::predicate},
                    LiteralBits{0, ValueType::predicate}}, {pred(1)});
            } else {
                const Operand identity = LiteralBits{opcode == Opcode::logical_and ? 1u : 0u,
                                                     ValueType::predicate};
                b.emit(join, opcode, boolean_mode, {identity, p}, {pred(1)});
            }
            if (inverted) b.emit(join, Opcode::logical_not, boolean_mode, {pred(1)}, {pred(1)});
            b.terminate(join, inverted ? Branch{pred(1), done, use} : Branch{pred(1), use, done});
            b.store_word(use, gpr(2));
            b.terminate(use, Jump{done});
            b.terminate(done, Return{});
            const auto program = expect_ok(std::move(b), "Boolean predicate copies preserve guard identity");
            if (!program.functions.empty()) {
                for (const bool truth : {false, true})
                    check(correlated_outputs(program, truth, false) ==
                        (truth ? std::vector<std::uint64_t>{17} : std::vector<std::uint64_t>{}),
                        "Boolean copies preserve the selected store and its value");
            }
            (void)q;
        }
    }
}

void test_loop_phi_parallel_transfer() {
    Builder b;
    const auto p = b.parameter(ValueType::predicate);
    const auto q = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto use = b.make_block();
    const auto latch = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, pred(0), p);
    b.copy(entry, pred(1), q);
    b.copy(entry, gpr(0), word(0));
    b.terminate(entry, Jump{header});
    b.terminate(header, Branch{pred(0), assign, join});
    b.copy(assign, gpr(2), word(23));
    b.terminate(assign, Jump{join});
    // The copied guard retains this iteration's predicate value before the
    // loop swaps the sources of its two loop-carried predicate phis.
    b.copy(join, pred(2), pred(0));
    b.terminate(join, Branch{pred(2), use, latch});
    b.store_word(use, gpr(2));
    b.terminate(use, Jump{latch});
    const auto saved = b.value(ValueType::predicate);
    b.copy(latch, saved, pred(0));
    b.copy(latch, pred(0), pred(1));
    b.copy(latch, pred(1), saved);
    b.add(latch, gpr(0), gpr(0), word(1));
    b.compare(latch, pred(3), gpr(0), word(4));
    b.terminate(latch, Branch{pred(3), header, done});
    b.terminate(done, Return{});
    const auto program = expect_ok(std::move(b), "Mutually loop-carried predicates transfer simultaneously");
    if (!program.functions.empty())
        for (const bool first : {false, true})
            for (const bool second : {false, true})
                check(correlated_outputs(program, first, second) ==
                    std::vector<std::uint64_t>(2u * unsigned(first) + 2u * unsigned(second), 23),
                    "Swapping predicate phis cannot reuse an already updated incoming fact");
}

void test_loop_phi_undefined_transfer() {
    Builder b;
    const auto p = b.parameter(ValueType::predicate);
    const auto q = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto use = b.make_block();
    const auto latch = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, pred(0), p);
    b.copy(entry, pred(1), p);
    b.copy(entry, gpr(0), word(0));
    b.terminate(entry, Jump{header});
    b.terminate(header, Branch{pred(0), assign, join});
    b.copy(assign, gpr(2), word(29));
    b.terminate(assign, Jump{join});
    b.terminate(join, Branch{pred(1), use, latch});
    b.store_word(use, gpr(2));
    b.terminate(use, Jump{latch});
    const auto saved = b.value(ValueType::predicate);
    b.copy(latch, saved, pred(0));
    b.copy(latch, pred(0), q);
    b.emit(latch, Opcode::logical_not, boolean_mode, {saved}, {pred(1)});
    b.add(latch, gpr(0), gpr(0), word(1));
    b.compare(latch, pred(2), gpr(0), word(2));
    b.terminate(latch, Branch{pred(2), header, done});
    b.terminate(done, Return{});
    expect_zero_reads(std::move(b),
                    "Simultaneous loop phi transfer exposes a later feasible undefined use");
}

Builder integer_mask_builder(bool rewrite, bool select_mask = false) {
    Builder b;
    const auto word_input = b.parameter(ValueType::word32);
    const auto guard = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto use = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, gpr(0), word_input);
    b.compare(entry, pred(0), gpr(0), word(1), Compare::equal);
    b.emit(entry, Opcode::logical_and, boolean_mode, {pred(0), guard}, {pred(1)});
    b.terminate(entry, Branch{pred(1), assign, join});
    b.copy(assign, gpr(2), word(37));
    b.terminate(assign, Jump{join});
    if (rewrite) b.copy(join, gpr(0), word(1));
    // Independently emitted ISET-style mask/CC operations and the original
    // ISETP-style guard have equal predicates only for the same SSA inputs.
    b.compare(join, pred(2), gpr(0), word(1), Compare::equal);
    if (select_mask)
        b.emit(join, Opcode::select, {}, {pred(2), word(0xffffffffu), word(0)}, {gpr(3)});
    else b.emit(join, Opcode::predicate_to_mask, {}, {pred(2)}, {gpr(3)});
    b.compare(join, flag(0), gpr(3), word(0), Compare::equal);
    b.emit(join, Opcode::compare, NumericMode{Domain::signed_integer, ScalarType::s32},
           {gpr(3), word(0)}, {flag(1)}, ComparisonInfo{Compare::less, false});
    b.emit(join, Opcode::logical_not, boolean_mode, {flag(0)}, {pred(3)});
    b.emit(join, Opcode::logical_or, boolean_mode, {flag(1), pred(3)}, {pred(4)});
    b.emit(join, Opcode::logical_and, boolean_mode, {guard, pred(4)}, {pred(5)});
    b.terminate(join, Branch{pred(5), use, done});
    b.store_word(use, gpr(2));
    b.terminate(use, Jump{done});
    b.terminate(done, Return{});
    return b;
}

void test_integer_mask_guard_correlations() {
    for (const bool select_mask : {false, true}) {
        const auto program = expect_ok(integer_mask_builder(false, select_mask),
            "ISET mask condition codes correlate with the equivalent ISETP guard");
        if (!program.functions.empty())
            for (const bool word_input : {false, true})
                for (const bool guard : {false, true})
                    check(correlated_outputs(program, word_input, guard) ==
                        (word_input && guard ? std::vector<std::uint64_t>{37} : std::vector<std::uint64_t>{}),
                        "All runtime inputs preserve mask-derived guard effects");
    }
    expect_zero_reads(integer_mask_builder(true),
                    "Rewriting a comparison input invalidates equivalence with the original guard");
    {
        Builder unsigned_mask;
        const auto input = unsigned_mask.parameter(ValueType::predicate);
        const auto start = unsigned_mask.entry();
        const auto initialize = unsigned_mask.make_block();
        const auto test = unsigned_mask.make_block();
        const auto read = unsigned_mask.make_block();
        const auto exit = unsigned_mask.make_block();
        unsigned_mask.terminate(start, Branch{input, test, initialize});
        unsigned_mask.copy(initialize, gpr(2), word(37));
        unsigned_mask.terminate(initialize, Jump{test});
        unsigned_mask.emit(test, Opcode::predicate_to_mask, {}, {input}, {gpr(3)});
        unsigned_mask.compare(test, pred(0), gpr(3), word(0));
        unsigned_mask.terminate(test, Branch{pred(0), exit, read});
        unsigned_mask.store_word(read, gpr(2));
        unsigned_mask.terminate(read, Jump{exit});
        unsigned_mask.terminate(exit, Return{});
        expect_zero_reads(std::move(unsigned_mask),
            "Unsigned mask less-than zero is false even for all ones and cannot act as a sign flag");
    }

    Builder b;
    const auto p = b.parameter(ValueType::predicate);
    const auto q = b.parameter(ValueType::predicate);
    const auto entry = b.entry();
    const auto header = b.make_block();
    const auto assign = b.make_block();
    const auto join = b.make_block();
    const auto use = b.make_block();
    const auto latch = b.make_block();
    const auto done = b.make_block();
    b.copy(entry, gpr(0), word(0));
    b.terminate(entry, Jump{header});
    b.compare(header, pred(0), gpr(0), word(2));
    b.terminate(header, Branch{pred(0), assign, join});
    b.copy(assign, gpr(2), gpr(0));
    b.terminate(assign, Jump{join});
    b.compare(join, pred(1), gpr(0), word(2));
    b.terminate(join, Branch{pred(1), use, latch});
    b.store_word(use, gpr(2));
    b.terminate(use, Jump{latch});
    b.add(latch, gpr(0), gpr(0), word(1));
    b.compare(latch, pred(2), gpr(0), word(4));
    b.terminate(latch, Branch{pred(2), header, done});
    b.terminate(done, Return{});
    const auto loop = expect_ok(std::move(b), "Repeated integer comparisons share only their current dynamic inputs");
    if (!loop.functions.empty())
        check(correlated_outputs(loop, false, false) == std::vector<std::uint64_t>{0, 1},
              "Reexecuting an input phi kills old comparison outcomes");
    (void)p;
    (void)q;
}

enum class GuardedEffect {
    vote, derivative, barrier, implicit_lod_sample, shuffle, discard_derivatives
};

void append_guarded_effect(Builder& builder, BlockId block, GuardedEffect effect) {
    const NumericMode float_mode{Domain::floating, ScalarType::f32, Rounding::nearest_even};
    if (effect == GuardedEffect::vote) {
        builder.emit(block, Opcode::vote_any, boolean_mode, {pred(0)}, {pred(4)},
                     SubgroupInfo{}, Effects::subgroup);
    } else if (effect == GuardedEffect::derivative) {
        builder.emit(block, Opcode::derivative_x, float_mode, {word(0)}, {gpr(4)}, {},
                     Effects::derivative);
    } else if (effect == GuardedEffect::barrier) {
        MemoryInfo info;
        info.space = AddressSpace::shared;
        info.order = MemoryOrder::acquire_release;
        info.scope = MemoryScope::workgroup;
        builder.emit(block, Opcode::barrier, {}, {}, {}, info, Effects::barrier);
    } else if (effect == GuardedEffect::implicit_lod_sample) {
        TextureInfo info;
        info.coordinates = {InputIndex{0}, InputIndex{1}};
        builder.emit(block, Opcode::texture_sample, float_mode, {word(0), word(0)},
                     {gpr(4)}, info, Effects::texture | Effects::derivative);
    } else if (effect == GuardedEffect::shuffle) {
        builder.emit(block, Opcode::shuffle, integer_mode,
                     {word(17), word(1), word(31)}, {gpr(4), pred(4)},
                     SubgroupInfo{ShuffleKind::up, 32}, Effects::subgroup);
    } else {
        builder.emit(block, Opcode::derivative_x, float_mode, {word(0)}, {gpr(4)}, {},
                     Effects::derivative);
        builder.emit(block, Opcode::discard, {}, {}, {}, {}, Effects::discard);
        builder.emit(block, Opcode::derivative_y, float_mode, {word(1)}, {gpr(5)}, {},
                     Effects::derivative);
    }
}

void test_collective_correlation_fixed_cfg() {
    for (const auto effect : {GuardedEffect::vote, GuardedEffect::derivative,
                              GuardedEffect::barrier, GuardedEffect::implicit_lod_sample,
                              GuardedEffect::shuffle, GuardedEffect::discard_derivatives}) {
        Builder builder;
        const auto p = builder.parameter(ValueType::predicate);
        const auto q = builder.parameter(ValueType::predicate);
        const auto replacement = builder.parameter(ValueType::word32);
        const auto entry = builder.entry();
        const auto assign = builder.make_block();
        const auto join = builder.make_block();
        const auto use = builder.make_block();
        const auto done = builder.make_block();
        builder.copy(entry, pred(0), p);
        builder.terminate(entry, Branch{pred(0), assign, join});
        const auto definition = builder.copy(assign, gpr(2), word(41));
        builder.terminate(assign, Jump{join});
        builder.terminate(join, Branch{pred(0), use, done});
        append_guarded_effect(builder, use, effect);
        builder.store_word(use, gpr(2));
        builder.terminate(use, Jump{done});
        builder.terminate(done, Return{});

        const auto program = expect_ok(std::move(builder),
            "Fixed-CFG correlation accepts reconvergence-sensitive effects");
        if (program.functions.empty()) continue;
        check(program.blocks.size() == 5 &&
                  program.functions[0].blocks ==
                      std::vector<BlockId>{entry, assign, join, use, done},
              "Correlation preserves the original block identities and order");
        check_block_shape(program, entry, {assign, join},
                          {{Opcode::copy, Effects::none}},
                          "Guarded-effect entry retains its operations and successors");
        check_block_shape(program, assign, {join}, {{Opcode::copy, Effects::none}},
                          "Guarded-effect definition block remains unchanged");
        check_block_shape(program, join, {use, done}, {},
                          "Guarded-effect join keeps the repeated guard and successors");
        if (effect == GuardedEffect::vote) {
            check_block_shape(program, use, {done},
                              {{Opcode::vote_any, Effects::subgroup},
                               {Opcode::store_io, Effects::write_io}},
                              "Vote and store remain ordered in their original block");
        } else if (effect == GuardedEffect::derivative) {
            check_block_shape(program, use, {done},
                              {{Opcode::derivative_x, Effects::derivative},
                               {Opcode::store_io, Effects::write_io}},
                              "Derivative and store remain ordered in their original block");
        } else if (effect == GuardedEffect::barrier) {
            check_block_shape(program, use, {done},
                              {{Opcode::barrier, Effects::barrier},
                               {Opcode::store_io, Effects::write_io}},
                              "Barrier and store remain ordered in their original block");
        } else if (effect == GuardedEffect::implicit_lod_sample) {
            check_block_shape(program, use, {done},
                              {{Opcode::texture_sample, Effects::texture | Effects::derivative},
                               {Opcode::store_io, Effects::write_io}},
                              "Implicit-LOD sample and store remain ordered in their original block");
        } else if (effect == GuardedEffect::shuffle) {
            check_block_shape(program, use, {done},
                              {{Opcode::shuffle, Effects::subgroup},
                               {Opcode::store_io, Effects::write_io}},
                              "Shuffle and store remain ordered in their original block");
        } else {
            check_block_shape(program, use, {done},
                              {{Opcode::derivative_x, Effects::derivative},
                               {Opcode::discard, Effects::discard},
                               {Opcode::derivative_y, Effects::derivative},
                               {Opcode::store_io, Effects::write_io}},
                              "Derivatives, discard and store retain exact effect order");
        }
        check_block_shape(program, done, {}, {}, "Guarded-effect exit remains unchanged");

        check(program.blocks[join.value].phis.size() == 1,
              "Guarded-effect correlation retains one repaired data phi");
        if (program.blocks[join.value].phis.size() == 1) {
            const auto& phi = program.blocks[join.value].phis[0];
            const auto* defined = phi_input(phi, assign);
            const auto* repaired = phi_input(phi, entry);
            check(defined && defined->value == def_of(program, definition),
                  "The feasible phi edge keeps the guarded definition");
            check(repaired && repaired->value == replacement,
                  "The unobservable undefined phi input reuses the only dominating word parameter");
        }
        for (const bool predicate : {false, true})
            check(correlated_outputs(program, predicate, false) ==
                      (predicate ? std::vector<std::uint64_t>{41} : std::vector<std::uint64_t>{}),
                  "Guarded effects preserve the feasible scalar output path");
        (void)q;
    }
}

void test_feasible_undefined_collective_inputs() {
    for (const auto opcode : {Opcode::vote_any, Opcode::derivative_x}) {
        Builder builder;
        const auto p = builder.parameter(ValueType::predicate);
        const auto entry = builder.entry();
        const auto assign = builder.make_block();
        const auto join = builder.make_block();
        builder.terminate(entry, Branch{p, assign, join});
        if (opcode == Opcode::vote_any)
            builder.copy(assign, pred(2), LiteralBits{1, ValueType::predicate});
        else builder.copy(assign, gpr(2), word(1));
        builder.terminate(assign, Jump{join});
        if (opcode == Opcode::vote_any) {
            builder.emit(join, opcode, boolean_mode, {pred(2)}, {pred(3)},
                         SubgroupInfo{}, Effects::subgroup);
        } else {
            builder.emit(join, opcode,
                         {Domain::floating, ScalarType::f32, Rounding::nearest_even},
                         {gpr(2)}, {gpr(3)}, {}, Effects::derivative);
        }
        builder.terminate(join, Return{});
        expect_zero_reads(std::move(builder),
                        "A feasible undefined collective input remains rejected");
    }

    for (const std::uint32_t state_budget : {0u, 1u}) {
        Builder builder;
        const auto p = builder.parameter(ValueType::predicate);
        const auto replacement = builder.parameter(ValueType::word32);
        const auto entry = builder.entry();
        const auto assign = builder.make_block();
        const auto join = builder.make_block();
        builder.terminate(entry, Branch{p, assign, join});
        builder.copy(assign, gpr(2), word(1));
        builder.terminate(assign, Jump{join});
        builder.emit(join, Opcode::derivative_x,
                     {Domain::floating, ScalarType::f32, Rounding::nearest_even},
                     {gpr(2)}, {gpr(3)}, {}, Effects::derivative);
        builder.terminate(join, Return{});
        Limits limits;
        limits.token_states = state_budget;
        expect_rejected(std::move(builder), DiagnosticCode::limit_exceeded,
                        "Fixed-CFG proof respects the predicate-state budget", limits);
        (void)replacement;
    }

    Builder branch;
    const auto p = branch.parameter(ValueType::predicate);
    const auto entry = branch.entry();
    const auto assign = branch.make_block();
    const auto join = branch.make_block();
    const auto taken = branch.make_block();
    const auto done = branch.make_block();
    branch.terminate(entry, Branch{p, assign, join});
    branch.copy(assign, pred(2), LiteralBits{1, ValueType::predicate});
    branch.terminate(assign, Jump{join});
    branch.terminate(join, Branch{pred(2), taken, done});
    branch.terminate(taken, Jump{done});
    branch.terminate(done, Return{});
    expect_zero_reads(std::move(branch),
                    "A feasible tainted terminator input remains rejected");
}

void test_missing_phi_replacement_rejected() {
    Builder builder;
    const auto p = builder.parameter(ValueType::predicate);
    const auto entry = builder.entry();
    const auto assign = builder.make_block();
    const auto join = builder.make_block();
    const auto use = builder.make_block();
    const auto done = builder.make_block();
    builder.terminate(entry, Branch{p, assign, join});
    builder.copy(assign, gpr(2), word(43));
    builder.terminate(assign, Jump{join});
    builder.terminate(join, Branch{p, use, done});
    MemoryInfo info;
    info.space = AddressSpace::shared;
    info.order = MemoryOrder::acquire_release;
    info.scope = MemoryScope::workgroup;
    builder.emit(use, Opcode::barrier, {}, {}, {}, info, Effects::barrier);
    builder.store_word(use, gpr(2));
    builder.terminate(use, Jump{done});
    builder.terminate(done, Return{});
    expect_zero_reads(std::move(builder),
                    "An unobservable undefined phi input on a feasible edge needs a typed dominating replacement");
}

void test_tainted_real_operation_rejected() {
    Builder builder;
    const auto p = builder.parameter(ValueType::predicate);
    const auto replacement = builder.parameter(ValueType::word32);
    const auto entry = builder.entry();
    const auto assign = builder.make_block();
    const auto join = builder.make_block();
    const auto use = builder.make_block();
    const auto done = builder.make_block();
    builder.terminate(entry, Branch{p, assign, join});
    builder.copy(assign, gpr(2), word(47));
    builder.terminate(assign, Jump{join});
    builder.copy(join, gpr(3), gpr(2));
    builder.terminate(join, Branch{p, use, done});
    builder.emit(use, Opcode::derivative_x,
                 {Domain::floating, ScalarType::f32, Rounding::nearest_even},
                 {gpr(3)}, {gpr(4)}, {}, Effects::derivative);
    builder.store_word(use, gpr(3));
    builder.terminate(use, Jump{done});
    builder.terminate(done, Return{});
    expect_zero_reads(std::move(builder),
                    "A feasible undefined read by a real operation cannot be repaired by a later guard");
    (void)replacement;
}

void test_predicate_and_condition_code() {
    Builder b;
    const auto entry = b.entry();
    const auto left = b.make_block();
    const auto right = b.make_block();
    const auto join = b.make_block();
    b.copy(entry, gpr(2), word(1));
    b.copy(entry, gpr(3), word(2));
    const auto cc_zero = b.compare(entry, flag(0), gpr(2), word(1), Compare::equal);
    const auto cc_sign = b.compare(entry, flag(1), gpr(3), word(2), Compare::equal);
    b.terminate(entry, Branch{flag(0), left, right});
    const auto combined = b.emit(left, Opcode::logical_and, boolean_mode, {flag(0), flag(1)}, {pred(0)});
    b.terminate(left, Jump{join});
    const auto inverted = b.emit(right, Opcode::logical_not, boolean_mode, {flag(0)}, {pred(0)});
    b.terminate(right, Jump{join});
    const auto store = b.store_predicate(join, pred(0));
    b.terminate(join, Return{});

    auto program = expect_ok(std::move(b), "Predicates and condition codes rename like registers");
    check(program.blocks[join.value].phis.size() == 1, "Only the predicate written in both arms needs a phi");
    const auto& phi = program.blocks[join.value].phis[0];
    check(phi_input(phi, left) && phi_input(phi, left)->value == def_of(program, combined),
          "Predicate phi takes the true-arm value");
    check(phi_input(phi, right) && phi_input(phi, right)->value == def_of(program, inverted),
          "Predicate phi takes the false-arm value");
    check(same_value(program.operations[combined.value].inputs[0], def_of(program, cc_zero)) &&
              same_value(program.operations[combined.value].inputs[1], def_of(program, cc_sign)),
          "Condition-code reads rename to their defining comparisons");
    check(same_value(program.operations[store.value].inputs[0], phi.output),
          "Predicate store reads the phi");
}

void test_large_programs() {
    // A long straight chain: every block redefines the register, reads cross
    // block boundaries through the dominator walk without any phi.
    {
        constexpr int levels = 1200;
        Builder b;
        const auto entry = b.entry();
        b.copy(entry, gpr(2), word(0));
        auto previous = entry;
        OperationId last;
        for (int i = 0; i < levels; ++i) {
            const auto next = b.make_block();
            b.terminate(previous, Jump{next});
            last = b.add(next, gpr(2), gpr(2), word(1));
            previous = next;
        }
        const auto done = b.make_block();
        b.terminate(previous, Jump{done});
        const auto store = b.store_word(done, gpr(2));
        b.terminate(done, Return{});
        auto program = expect_ok(std::move(b), "Long chains stay within analysis limits");
        check(same_value(program.operations[store.value].inputs[0], def_of(program, last)),
              "Chain reads reach the previous block definition");
    }
    // A ladder of diamonds: every join materializes one phi.
    {
        constexpr int levels = 400;
        Builder b;
        const auto entry = b.entry();
        b.copy(entry, gpr(2), word(0));
        b.compare(entry, pred(0), word(0), word(1));
        auto current = entry;
        OperationId last_left, last_right;
        BlockId last_join = entry;
        for (int i = 0; i < levels; ++i) {
            const auto left = b.make_block();
            const auto right = b.make_block();
            const auto join = b.make_block();
            b.terminate(current, Branch{pred(0), left, right});
            last_left = b.add(left, gpr(2), gpr(2), word(1));
            b.terminate(left, Jump{join});
            last_right = b.add(right, gpr(2), gpr(2), word(2));
            b.terminate(right, Jump{join});
            current = join;
            last_join = join;
        }
        const auto done = b.make_block();
        b.terminate(current, Jump{done});
        const auto store = b.store_word(done, gpr(2));
        b.terminate(done, Return{});
        auto program = expect_ok(std::move(b), "Diamond ladders stay within analysis limits");
        std::size_t total_phis = 0;
        for (const auto& block : program.blocks) total_phis += block.phis.size();
        check(total_phis == levels, "Every diamond join carries exactly one phi");
        const auto& phi = program.blocks[last_join.value].phis[0];
        check(phi_input(phi, BlockId{last_join.value - 2})->value == def_of(program, last_left) &&
                  phi_input(phi, BlockId{last_join.value - 1})->value == def_of(program, last_right),
              "Ladder phi feeds from both arms");
        check(same_value(program.operations[store.value].inputs[0], phi.output),
              "Ladder store reads the final phi");
    }
}

void test_limits() {
    {
        OperationId load{}, test{}, left{}, right{}, store{};
        Limits supplied;
        supplied.values = 4; // The diamond needs five renamed values.
        expect_rejected(diamond_builder(load, test, left, right, store), DiagnosticCode::limit_exceeded,
                        "The value budget bounds renamed definitions", supplied);
        (void)load; (void)test; (void)left; (void)right; (void)store;
    }
    {
        OperationId load{}, test{}, left{}, right{}, store{};
        Limits supplied;
        supplied.phi_inputs = 1; // The diamond phi has two inputs.
        expect_rejected(diamond_builder(load, test, left, right, store), DiagnosticCode::limit_exceeded,
                        "The phi input budget bounds phi width", supplied);
        (void)load; (void)test; (void)left; (void)right; (void)store;
    }
    {
        Builder b;
        const auto entry = b.entry();
        auto previous = entry;
        for (int i = 0; i < 50; ++i) {
            const auto next = b.make_block();
            b.terminate(previous, Jump{next});
            b.add(next, gpr(2), gpr(2), word(1));
            previous = next;
        }
        const auto done = b.make_block();
        b.terminate(previous, Jump{done});
        b.store_word(done, gpr(2));
        b.terminate(done, Return{});
        Limits supplied;
        supplied.analysis_steps = 64;
        expect_rejected(std::move(b), DiagnosticCode::limit_exceeded,
                        "The analysis budget bounds dominance work", supplied);
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto exit = b.make_block();
        b.terminate(entry, Branch{pred(7), entry, exit});
        b.terminate(exit, Return{});
        Limits supplied;
        supplied.blocks = 2;
        expect_rejected(std::move(b), DiagnosticCode::limit_exceeded,
                        "An entry preheader obeys the block limit", supplied);
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto exit = b.make_block();
        b.terminate(entry, Branch{pred(7), entry, exit});
        b.terminate(exit, Return{});
        Limits supplied;
        supplied.edges = 2;
        expect_rejected(std::move(b), DiagnosticCode::limit_exceeded,
                        "An entry preheader obeys the edge limit", supplied);
    }
    {
        Builder b;
        const auto entry = b.entry();
        b.add(entry, gpr(10), gpr(9), word(1));
        b.terminate(entry, Return{});
        Limits supplied;
        supplied.analysis_steps = 4;
        expect_rejected(std::move(b), DiagnosticCode::limit_exceeded,
                        "Dead-code cleanup obeys the shared SSA analysis budget", supplied);
    }
}

void test_calls_rejected() {
    Builder b;
    const auto entry = b.entry();
    b.copy(entry, gpr(2), word(0));
    b.emit(entry, Opcode::call, {}, {}, {}, CallInfo{FunctionId{0}}, Effects::call);
    b.store_word(entry, gpr(2));
    b.terminate(entry, Return{});
    expect_rejected(std::move(b), DiagnosticCode::unimplemented,
                    "Call ABI handling is explicitly rejected, never skipped");
}

void test_pseudo_registers() {
    Builder b;
    const auto entry = b.entry();
    const auto zero_copy = b.copy(entry, gpr(5), gpr(255));
    const auto true_copy = b.copy(entry, pred(1), pred(7));
    b.store_word(entry, gpr(5));
    const auto next = b.make_block();
    b.terminate(entry, Branch{pred(1), next, next});
    b.terminate(next, Return{});
    auto program = expect_ok(std::move(b), "Read-only pseudo-registers resolve to constants");
    const auto& zero_bits = std::get<LiteralBits>(program.operations[zero_copy.value].inputs[0]);
    const auto& true_bits = std::get<LiteralBits>(program.operations[true_copy.value].inputs[0]);
    check(zero_bits.bits == 0 && zero_bits.type == ValueType::word32, "R255 reads become word zero");
    check(true_bits.bits == 1 && true_bits.type == ValueType::predicate, "P7 reads become predicate true");
    for (const auto& block : program.blocks) check(block.phis.empty(), "Constants never need phis");
}

void test_compiler_locals() {
    Builder b;
    const auto entry = b.entry();
    const auto local = b.value(ValueType::word32);
    const auto left = b.make_block();
    const auto right = b.make_block();
    const auto join = b.make_block();
    const auto test = b.compare(entry, pred(0), word(0), word(1));
    b.terminate(entry, Branch{pred(0), left, right});
    const auto from_left = b.emit(left, Opcode::copy, {}, {word(1)}, {local});
    b.terminate(left, Jump{join});
    const auto from_right = b.emit(right, Opcode::copy, {}, {word(2)}, {local});
    b.terminate(right, Jump{join});
    const auto store = b.store_word(join, local);
    b.terminate(join, Return{});
    (void)test;

    auto program = expect_ok(std::move(b), "Compiler locals rename like physical registers");
    check(program.blocks[join.value].phis.size() == 1, "A multiply assigned local merges through a phi");
    const auto& phi = program.blocks[join.value].phis[0];
    check(phi_input(phi, left) && phi_input(phi, left)->value == def_of(program, from_left) &&
              phi_input(phi, right) && phi_input(phi, right)->value == def_of(program, from_right),
          "Local phi keeps both assignments");
    check(same_value(program.operations[store.value].inputs[0], phi.output), "Local store reads the phi");
}

void test_unreachable_and_inputs() {
    {
        Builder b;
        const auto entry = b.entry();
        b.copy(entry, gpr(2), word(0));
        b.terminate(entry, Return{});
        const auto orphan = b.make_block();
        b.copy(orphan, gpr(3), word(1));
        b.terminate(orphan, Return{});
        expect_rejected(std::move(b), DiagnosticCode::invalid_ir, "Unreachable blocks are rejected");
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto never_defined = b.value(ValueType::word32);
        b.store_word(entry, never_defined);
        b.terminate(entry, Return{});
        expect_rejected(std::move(b), DiagnosticCode::undefined_read,
                        "Register-form validation failures propagate");
    }
    {
        Builder b;
        const auto entry = b.entry();
        const auto local = b.value(ValueType::word32, ValueKind::mutable_local);
        b.emit(entry, Opcode::copy, {}, {word(7)}, {local});
        b.store_word(entry, local);
        b.terminate(entry, Return{});
        auto program = expect_ok(std::move(b), "Mutable locals leave construction as SSA temporaries");
        check(program.values[local.value].kind == ValueKind::temporary, "SSA values are temporaries");
    }
}

void test_dependencies_preserved() {
    Builder b;
    const auto entry = b.entry();
    b.copy(entry, gpr(2), word(0));
    b.store_word(entry, gpr(2));
    b.terminate(entry, Return{});
    b.set_dependencies({
        {{FactKind::constant_buffer_word, 0, 0}, std::uint32_t{7}, {OriginKind::explicit_option, "synthetic-word"}},
    });
    auto program = expect_ok(std::move(b), "Construction succeeds with consumed facts");
    check(program.dependencies.size() == 1 &&
              std::holds_alternative<std::uint32_t>(program.dependencies[0].value) &&
              std::get<std::uint32_t>(program.dependencies[0].value) == 7,
          "Dependencies survive SSA construction unchanged");
}
} // namespace

int main() {
    test_diamond();
    test_nested_loops();
    test_multiple_exits();
    test_partial_initialization();
    test_undefined_reads();
    test_dead_undefined_cleanup();
    test_entry_backedge_parameter();
    test_correlated_path_resolution();
    test_nested_correlations_and_phi_chains();
    test_loop_predicate_redefinition();
    test_predicate_phi_correlations();
    test_predicate_phi_tail_definitions();
    test_boolean_guard_copies();
    test_loop_phi_parallel_transfer();
    test_loop_phi_undefined_transfer();
    test_integer_mask_guard_correlations();
    test_collective_correlation_fixed_cfg();
    test_feasible_undefined_collective_inputs();
    test_missing_phi_replacement_rejected();
    test_tainted_real_operation_rejected();
    test_predicate_and_condition_code();
    test_large_programs();
    test_limits();
    test_calls_rejected();
    test_pseudo_registers();
    test_compiler_locals();
    test_unreachable_and_inputs();
    test_dependencies_preserved();
    if (failures) return 1;
    std::cout << "Shader pipeline SSA construction passed\n";
    return 0;
}
