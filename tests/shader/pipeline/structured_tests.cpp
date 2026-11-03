// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Structured-control-flow tests and interpreters are independently authored.

#include "pipeline/phases.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
Operand word(std::uint32_t value) { return LiteralBits{value, ValueType::word32}; }
const NumericMode integer{Domain::unsigned_integer, ScalarType::u32};
MetalProfile profile() { return {"synthetic", 1, 16, 16, 16, true, true, false, false, false, false}; }

struct Builder {
    LegalProgram legal;
    explicit Builder(std::size_t blocks) {
        legal.profile = profile();
        auto& p = legal.ir;
        p.entry = FunctionId{0};
        p.functions.push_back({0, BlockId{0}, {}, {}, {}});
        p.blocks.resize(blocks);
        for (std::uint32_t b = 0; b < blocks; ++b) {
            p.blocks[b].function = FunctionId{0};
            p.functions[0].blocks.push_back(BlockId{b});
        }
    }
    ValueId value(ValueType type = ValueType::word32) {
        ValueId id{static_cast<std::uint32_t>(legal.ir.values.size())};
        legal.ir.values.push_back({type});
        return id;
    }
    ValueId parameter(ValueType type = ValueType::word32) {
        auto id = value(type);
        legal.ir.functions[0].parameters.push_back(id);
        return id;
    }
    void op(std::uint32_t block, Operation operation) {
        OperationId id{static_cast<std::uint32_t>(legal.ir.operations.size())};
        legal.ir.operations.push_back(std::move(operation));
        legal.ir.blocks[block].operations.push_back(id);
    }
    void trace(std::uint32_t block) {
        IoInfo io;
        io.semantic = IoSemantic::color;
        io.type = {ScalarType::u32, 1};
        io.data = {InputIndex{0}};
        op(block, {{}, Opcode::store_io, Effects::write_io, integer, {word(block)}, {}, io});
    }
    void fragment_io() {
        legal.ir.stage = Stage::fragment;
        IoUse input;
        input.semantic = IoSemantic::generic;
        input.mask = 1;
        input.type = {ScalarType::u32, 1};
        input.interpolation.mode = InterpolationMode::flat;
        input.access = Access::read;
        legal.requirements.io.push_back(input);
        IoUse output;
        output.semantic = IoSemantic::color;
        output.mask = 1;
        output.type = {ScalarType::u32, 1};
        output.access = Access::write;
        legal.requirements.io.push_back(output);
    }
    void color(std::uint32_t block, std::uint32_t value) {
        IoInfo io;
        io.semantic = IoSemantic::color;
        io.type = {ScalarType::u32, 1};
        io.data = {InputIndex{0}};
        op(block, {{}, Opcode::store_io, Effects::write_io, integer, {word(value)}, {}, io});
    }
    void jump(std::uint32_t from, std::uint32_t to) { legal.ir.blocks[from].terminator = Jump{BlockId{to}}; }
    void branch(std::uint32_t from, std::uint32_t yes, std::uint32_t no) {
        const auto input = value();
        IoInfo io;
        io.type = {ScalarType::u32, 1};
        io.interpolation.mode = InterpolationMode::flat;
        op(from, {{}, Opcode::load_io, Effects::read_io, integer, {}, {input}, io});
        const auto condition = value(ValueType::predicate);
        op(from, {{}, Opcode::compare, Effects::none, integer, {input, word(0)}, {condition}, ComparisonInfo{Compare::not_equal, false}});
        legal.ir.blocks[from].terminator = Branch{condition, BlockId{yes}, BlockId{no}};
    }
    LegalProgram finish() {
        for (auto& block : legal.ir.blocks) block.predecessors.clear();
        for (std::uint32_t b = 0; b < legal.ir.blocks.size(); ++b)
            for (auto successor : successors(legal.ir.blocks[b].terminator)) legal.ir.blocks[successor.value].predecessors.push_back(BlockId{b});
        return std::move(legal);
    }
};

struct Execution {
    std::vector<std::uint64_t> returned, trace;
    std::size_t reads{};
    bool operator==(const Execution&) const = default;
};
class Machine {
public:
    Machine(const Program& program, const std::vector<std::uint64_t>& arguments,
            const std::vector<std::uint64_t>& inputs)
        : p(program), values(p.values.size()), initialized(p.values.size()), inputs(inputs) {
        const auto& parameters = p.functions[p.entry.value].parameters;
        if (arguments.size() != parameters.size()) throw std::runtime_error("Wrong synthetic argument count");
        for (std::size_t i = 0; i < arguments.size(); ++i) set(parameters[i], arguments[i]);
    }
    Execution cfg();
    Execution ast(const StructuredProgram&);
private:
    const Program& p;
    std::vector<std::uint64_t> values;
    std::vector<bool> initialized;
    const std::vector<std::uint64_t>& inputs;
    Execution result;
    std::size_t steps{};
    void step() { if (++steps > 100000) throw std::runtime_error("Synthetic execution step limit"); }
    std::uint64_t get(const Operand& input) {
        if (const auto* literal = std::get_if<LiteralBits>(&input)) return literal->bits;
        const auto id = std::get<ValueId>(input);
        if (!initialized.at(id.value)) throw std::runtime_error("Synthetic interpreter read before definition");
        return values.at(id.value);
    }
    void set(ValueId id, std::uint64_t value) { values.at(id.value) = value; initialized.at(id.value) = true; }
    void operations(BlockId);
    enum class Control { next, returned, leave, repeat };
    struct Signal { Control control{Control::next}; RegionId loop; };
    Signal region(const StructuredProgram&, RegionId);
};

void Machine::operations(BlockId block) {
    for (auto id : p.blocks[block.value].operations) {
        step();
        const auto& op = p.operations[id.value];
        std::vector<std::uint64_t> arguments;
        for (const auto& input : op.inputs) arguments.push_back(get(input));
        std::uint64_t output = 0;
        switch (op.opcode) {
        case Opcode::copy: output = arguments[0]; break;
        case Opcode::add: output = static_cast<std::uint32_t>(arguments[0] + arguments[1]); break;
        case Opcode::logical_not: output = !arguments[0]; break;
        case Opcode::logical_or: output = arguments[0] || arguments[1]; break;
        case Opcode::compare:
            switch (std::get<ComparisonInfo>(op.info).relation) {
            case Compare::less: output = arguments[0] < arguments[1]; break;
            case Compare::equal: output = arguments[0] == arguments[1]; break;
            case Compare::not_equal: output = arguments[0] != arguments[1]; break;
            default: throw std::runtime_error("Unsupported synthetic comparison");
            }
            break;
        case Opcode::load_io:
            if (result.reads >= inputs.size()) throw std::runtime_error("Synthetic input stream exhausted");
            output = inputs[result.reads++];
            break;
        case Opcode::store_io: result.trace.push_back(arguments[0]); break;
        default: throw std::runtime_error("Unsupported synthetic operation");
        }
        if (!op.outputs.empty()) set(std::get<ValueId>(op.outputs[0]), output);
    }
}

Execution Machine::cfg() {
    BlockId current = p.functions[p.entry.value].entry, predecessor;
    for (;;) {
        step();
        const auto& block = p.blocks[current.value];
        std::vector<std::pair<ValueId, std::uint64_t>> copies;
        for (const auto& phi : block.phis) {
            auto input = std::find_if(phi.inputs.begin(), phi.inputs.end(), [&](auto input) { return input.predecessor == predecessor; });
            if (input == phi.inputs.end()) throw std::runtime_error("Synthetic phi edge missing");
            copies.emplace_back(phi.output, get(input->value));
        }
        for (auto [destination, value] : copies) set(destination, value);
        operations(current);
        BlockId next;
        if (const auto* jump = std::get_if<Jump>(&block.terminator)) next = jump->target;
        else if (const auto* branch = std::get_if<Branch>(&block.terminator)) next = get(branch->condition) ? branch->when_true : branch->when_false;
        else if (const auto* select = std::get_if<Switch>(&block.terminator)) {
            const auto selector = get(select->selector);
            next = select->fallback;
            for (const auto& item : select->cases) if (item.bits == selector) next = item.target;
        } else if (const auto* returned = std::get_if<Return>(&block.terminator)) {
            for (const auto& input : returned->values) result.returned.push_back(get(input));
            return result;
        } else throw std::runtime_error("Synthetic unterminated block");
        predecessor = current;
        current = next;
    }
}

Machine::Signal Machine::region(const StructuredProgram& s, RegionId id) {
    step();
    const auto& node = s.regions.at(id.value);
    if (const auto* sequence = std::get_if<SequenceRegion>(&node)) {
        for (auto child : sequence->children) {
            auto signal = region(s, child);
            if (signal.control != Control::next) return signal;
        }
    } else if (const auto* block = std::get_if<BlockRegion>(&node)) operations(block->block);
    else if (const auto* branch = std::get_if<IfRegion>(&node)) return region(s, get(branch->condition) ? branch->when_true : branch->when_false);
    else if (const auto* loop = std::get_if<LoopRegion>(&node)) {
        for (;;) {
            auto signal = region(s, loop->body);
            if (signal.control == Control::leave && signal.loop == id) break;
            if (signal.control == Control::returned ||
                (signal.control != Control::next && signal.loop != id)) return signal;
            signal = region(s, loop->continuing);
            if (signal.control == Control::leave && signal.loop == id) break;
            if (signal.control == Control::returned ||
                (signal.control != Control::next && signal.loop != id)) return signal;
        }
    } else if (const auto* leave = std::get_if<BreakRegion>(&node)) return {Control::leave, leave->loop};
    else if (const auto* repeat = std::get_if<ContinueRegion>(&node)) return {Control::repeat, repeat->loop};
    else if (const auto* returned = std::get_if<ReturnRegion>(&node)) {
        for (const auto& input : returned->values) result.returned.push_back(get(input));
        return {Control::returned, {}};
    }
    return {};
}
Execution Machine::ast(const StructuredProgram& s) {
    const auto function = std::find_if(s.functions.begin(), s.functions.end(), [&](auto function) { return function.function == p.entry; });
    if (function == s.functions.end() || region(s, function->root).control != Control::returned)
        throw std::runtime_error("Synthetic AST fell through without returning");
    return result;
}

StructuredProgram structured(const LegalProgram& input, Limits limits = {}) {
    auto result = structure(input, limits);
    if (const auto* error = std::get_if<Diagnostic>(&result)) throw std::runtime_error(error->message);
    return std::get<StructuredProgram>(std::move(result));
}
void equivalent(const LegalProgram& input, const StructuredProgram& output,
                const std::vector<std::uint64_t>& arguments = {}, const std::vector<std::uint64_t>& inputs = {}) {
    const auto expected = Machine(input.ir, arguments, inputs).cfg();
    const auto actual = Machine(output.ir, arguments, inputs).ast(output);
    check(actual == expected, "CFG and structured execution agree on returns, effects and input consumption");
}
void rejected(Status status, std::string_view message, DiagnosticCode code = DiagnosticCode::invalid_ir) {
    const auto* error = std::get_if<Diagnostic>(&status);
    check(error && error->phase == Phase::structure && error->code == code, message);
}

void emits_msl(const StructuredProgram& program, std::string_view label) {
    auto planned = plan_resources(program, program.profile);
    if (const auto* error = std::get_if<Diagnostic>(&planned)) {
        check(false, std::string(label) + " resource planning succeeds: " + error->message);
        return;
    }
    auto artifact = emit_msl(program, std::get<ResourcePlan>(std::move(planned)), {});
    if (const auto* error = std::get_if<Diagnostic>(&artifact)) {
        check(false, std::string(label) + " MSL emission succeeds: " + error->message);
        return;
    }
    check(!std::get<MslArtifact>(artifact).source.empty(),
          std::string(label) + " emits nonempty MSL");
}

void rejects_missing_output(const StructuredProgram& program, std::string_view label) {
    auto planned = plan_resources(program, program.profile);
    if (const auto* error = std::get_if<Diagnostic>(&planned)) {
        check(false, std::string(label) + " resource planning succeeds: " + error->message);
        return;
    }
    const auto artifact = emit_msl(program, std::get<ResourcePlan>(std::move(planned)), {});
    const auto* error = std::get_if<Diagnostic>(&artifact);
    check(error && error->phase == Phase::emit_msl && error->code == DiagnosticCode::undefined_read,
          std::string(label) + " rejects a genuinely unwritten output path");
}

StructuredProgram reducible(LegalProgram input, std::string_view label) {
    const auto value_count = input.ir.values.size();
    auto output = structured(input);
    check(output.ir.values.size() == value_count,
          std::string(label) + " does not synthesize dispatch values");
    emits_msl(output, label);
    return output;
}

template <class T>
std::size_t region_count(const StructuredProgram& program) {
    return std::count_if(program.regions.begin(), program.regions.end(),
                         [](const Region& region) { return std::holds_alternative<T>(region); });
}

LegalProgram cycle_program(std::size_t length, ValueType type = ValueType::word32, bool critical = false) {
    Builder b(4);
    std::vector<ValueId> initial, carried;
    for (std::size_t i = 0; i < length; ++i) initial.push_back(b.parameter(type));
    const auto limit = b.parameter();
    for (std::size_t i = 0; i < length; ++i) carried.push_back(b.value(type));
    const auto zero = b.value(), count = b.value(), next = b.value();
    const auto condition = b.value(ValueType::predicate);
    b.op(0, {{}, Opcode::copy, Effects::none, {}, {word(0)}, {zero}, {}});
    b.jump(0, 1);
    for (std::size_t i = 0; i < length; ++i)
        b.legal.ir.blocks[1].phis.push_back({{}, carried[i], {{BlockId{0}, initial[i]}, {BlockId{critical ? 1U : 2U}, carried[(i + 1) % length]}}});
    b.legal.ir.blocks[1].phis.push_back({{}, count, {{BlockId{0}, zero}, {BlockId{critical ? 1U : 2U}, next}}});
    if (critical) b.op(1, {{}, Opcode::add, Effects::none, integer, {count, word(1)}, {next}, {}});
    b.op(1, {{}, Opcode::compare, Effects::none, integer, {critical ? Operand{next} : Operand{count}, limit}, {condition}, ComparisonInfo{Compare::less, false}});
    b.legal.ir.blocks[1].terminator = Branch{condition, BlockId{critical ? 1U : 2U}, BlockId{critical ? 2U : 3U}};
    if (!critical) {
        b.op(2, {{}, Opcode::add, Effects::none, integer, {count, word(1)}, {next}, {}});
        b.jump(2, 1);
    } else b.jump(2, 3);
    std::vector<Operand> outputs;
    for (auto id : carried) outputs.push_back(id);
    b.legal.ir.functions[0].results.assign(length, type);
    b.legal.ir.blocks[3].terminator = Return{outputs};
    return b.finish();
}

void test_parallel_copies() {
    for (auto type : {ValueType::word32, ValueType::word64, ValueType::predicate})
        for (std::size_t length : {1U, 2U, 3U}) {
            const auto input = cycle_program(length, type);
            const auto output = structured(input);
            for (std::uint64_t rounds = 0; rounds < 7; ++rounds) {
                std::vector<std::uint64_t> arguments;
                for (std::size_t i = 0; i < length; ++i)
                    arguments.push_back(type == ValueType::predicate ? i % 2 : (type == ValueType::word64 ? (1ULL << 40) : 0) + 11 + i * 7);
                arguments.push_back(rounds);
                equivalent(input, output, arguments);
                auto expected = Machine(input.ir, arguments, {}).cfg().returned;
                for (std::size_t i = 0; i < length; ++i) check(expected[i] == arguments[(i + rounds) % length], "Phi rotates parallel values without corruption");
            }
            for (const auto& block : output.ir.blocks) check(block.phis.empty(), "Phi destruction removes every phi");
        }
    const auto critical = cycle_program(2, ValueType::word32, true);
    const auto output = structured(critical);
    for (std::uint64_t rounds = 1; rounds < 6; ++rounds) equivalent(critical, output, {13, 29, rounds});
    check(Machine(output.ir, {13, 29, 1}, {}).ast(output).returned == std::vector<std::uint64_t>({13, 29}),
          "Untaken critical backedge cannot overwrite live phi values");
    auto fanout = cycle_program(3);
    fanout.ir.blocks[1].phis[2].inputs[1].value = fanout.ir.blocks[1].phis[0].output;
    equivalent(fanout, structured(fanout), {10, 20, 30, 5});
    auto chain = cycle_program(3);
    chain.ir.blocks[1].phis[2].inputs[1].value = chain.ir.blocks[1].phis[2].output;
    equivalent(chain, structured(chain), {10, 20, 30, 1});
}

void test_nested_loops() {
    Builder b(9);
    for (std::uint32_t i = 0; i < 9; ++i) b.trace(i);
    b.jump(0, 1);
    b.branch(1, 2, 8);
    b.branch(2, 3, 6);
    b.branch(3, 4, 5);
    b.branch(4, 8, 1);
    b.jump(5, 2);
    b.jump(6, 7);
    b.jump(7, 1);
    b.legal.ir.blocks[8].terminator = Return{};
    auto input = b.finish();
    const auto value_count = input.ir.values.size();
    auto output = structured(input);
    // The inner loop has two normal exits and uses flag fallback. SCC order and
    // its single pending target make the final return singleton exhaustive.
    check(std::count_if(output.regions.begin(), output.regions.end(), [](const auto& r) { return std::holds_alternative<LoopRegion>(r); }) == 2,
          "Multi-exit nested graph preserves its two CFG loops");
    check(output.ir.values.size() > value_count,
          "Multi-exit nested graph uses flag dispatch fallback");
    equivalent(input, output, {}, {1, 1, 0, 0, 0});
    equivalent(input, output, {}, {1, 1, 1, 0, 0});
    equivalent(input, output, {}, {1, 1, 1, 1});
    equivalent(input, output, {}, {1, 1, 0, 1, 0, 0, 1, 0, 0});
    std::reverse(input.ir.functions[0].blocks.begin(), input.ir.functions[0].blocks.end());
    equivalent(input, structured(input), {}, {1, 1, 1, 0, 0});
    // Reverse the actual arena IDs as well, preserving the graph through remaps.
    const auto count = static_cast<std::uint32_t>(input.ir.blocks.size());
    auto remap = [&](BlockId id) { return BlockId{count - 1 - id.value}; };
    std::reverse(input.ir.blocks.begin(), input.ir.blocks.end());
    for (auto& function : input.ir.functions) {
        function.entry = remap(function.entry);
        for (auto& block : function.blocks) block = remap(block);
    }
    for (auto& block : input.ir.blocks) {
        for (auto& predecessor : block.predecessors) predecessor = remap(predecessor);
        if (auto* jump = std::get_if<Jump>(&block.terminator)) jump->target = remap(jump->target);
        if (auto* branch = std::get_if<Branch>(&block.terminator)) {
            branch->when_true = remap(branch->when_true);
            branch->when_false = remap(branch->when_false);
        }
    }
    equivalent(input, structured(input), {}, {1, 1, 1, 0, 0});
}

void test_direct_reducible_regions() {
    Builder chain(2);
    chain.fragment_io();
    chain.legal.ir.functions[0].entry = BlockId{1};
    chain.color(1, 11);
    chain.jump(1, 0);
    chain.color(0, 12);
    chain.legal.ir.blocks[0].terminator = Return{};
    const auto chain_input = chain.finish();
    const auto chain_output = reducible(chain_input, "reversed-ID straight chain");
    equivalent(chain_input, chain_output);
    check(region_count<IfRegion>(chain_output) == 0 && region_count<LoopRegion>(chain_output) == 0,
          "Reversed-ID straight chain emits a direct sequence");

    Builder diamond(4);
    diamond.fragment_io();
    diamond.branch(0, 1, 2);
    diamond.color(1, 21);
    diamond.jump(1, 3);
    diamond.color(2, 22);
    diamond.jump(2, 3);
    diamond.color(3, 23);
    diamond.legal.ir.blocks[3].terminator = Return{};
    const auto diamond_input = diamond.finish();
    const auto diamond_output = reducible(diamond_input, "if/else diamond");
    equivalent(diamond_input, diamond_output, {}, {0});
    equivalent(diamond_input, diamond_output, {}, {1});
    check(region_count<IfRegion>(diamond_output) == 1,
          "If/else diamond emits one conditional region");

    Builder loops(7);
    loops.fragment_io();
    loops.color(0, 30);
    loops.jump(0, 1);
    loops.branch(1, 2, 6);
    loops.branch(2, 3, 4);
    loops.branch(3, 4, 2);
    loops.branch(4, 5, 6);
    loops.jump(5, 1);
    loops.legal.ir.blocks[6].terminator = Return{};
    const auto loops_input = loops.finish();
    const auto loops_output = reducible(loops_input, "nested loops with normal exits");
    check(region_count<LoopRegion>(loops_output) == 2,
          "Canonical nested loops emit two loop regions");
    equivalent(loops_input, loops_output, {}, {0});
    equivalent(loops_input, loops_output, {}, {1, 0, 0});
    equivalent(loops_input, loops_output, {}, {1, 1, 1, 0});
    equivalent(loops_input, loops_output, {}, {1, 1, 0, 1, 1, 0});
    equivalent(loops_input, loops_output, {}, {1, 0, 1, 0});

    Builder early(3);
    early.fragment_io();
    early.branch(0, 1, 2);
    early.color(1, 41);
    early.legal.ir.blocks[1].terminator = Return{};
    early.color(2, 42);
    early.legal.ir.blocks[2].terminator = Return{};
    const auto early_input = early.finish();
    const auto early_output = reducible(early_input, "early returns inside if/else");
    equivalent(early_input, early_output, {}, {0});
    equivalent(early_input, early_output, {}, {1});
    check(region_count<IfRegion>(early_output) == 1 && region_count<ReturnRegion>(early_output) == 2,
          "Both early-return paths remain inside one conditional");

    Builder guarded(7);
    guarded.fragment_io();
    guarded.branch(0, 1, 2);
    guarded.color(1, 51);
    guarded.jump(1, 3);
    guarded.color(2, 52);
    guarded.jump(2, 3);
    guarded.branch(3, 4, 5);
    guarded.color(4, 54);
    guarded.jump(4, 6);
    guarded.color(5, 55);
    guarded.jump(5, 6);
    guarded.legal.ir.blocks[6].terminator = Return{};
    const auto guarded_input = guarded.finish();
    const auto guarded_output = reducible(guarded_input, "guarded one-store blocks");
    for (const auto& choices : {std::vector<std::uint64_t>{0, 0},
                               std::vector<std::uint64_t>{0, 1},
                               std::vector<std::uint64_t>{1, 0},
                               std::vector<std::uint64_t>{1, 1}})
        equivalent(guarded_input, guarded_output, {}, choices);
    check(region_count<IfRegion>(guarded_output) == 2,
          "Guarded store diamonds emit one conditional each");

    Builder sequential(5);
    sequential.fragment_io();
    sequential.color(0, 110);
    sequential.branch(0, 1, 2);
    sequential.color(1, 111);
    sequential.jump(1, 2);
    sequential.branch(2, 3, 4);
    sequential.color(3, 113);
    sequential.jump(3, 4);
    sequential.legal.ir.blocks[4].terminator = Return{};
    const auto sequential_input = sequential.finish();
    const auto sequential_output = reducible(
        sequential_input, "sequential one-sided guarded stores");
    for (const auto& choices : {std::vector<std::uint64_t>{0, 0},
                               std::vector<std::uint64_t>{0, 1},
                               std::vector<std::uint64_t>{1, 0},
                               std::vector<std::uint64_t>{1, 1}})
        equivalent(sequential_input, sequential_output, {}, choices);
    check(region_count<IfRegion>(sequential_output) == 2,
          "Sequential guarded stores emit one one-sided conditional each");

    Builder switched(4);
    switched.fragment_io();
    const auto selector = switched.value();
    IoInfo input;
    input.type = {ScalarType::u32, 1};
    input.interpolation.mode = InterpolationMode::flat;
    switched.op(0, {{}, Opcode::load_io, Effects::read_io, integer, {}, {selector}, input});
    switched.legal.ir.blocks[0].terminator = Switch{
        selector, {{1, BlockId{1}}, {2, BlockId{1}}, {3, BlockId{2}}}, BlockId{2}};
    switched.color(1, 101);
    switched.jump(1, 3);
    switched.color(2, 102);
    switched.jump(2, 3);
    switched.legal.ir.blocks[3].terminator = Return{};
    const auto switched_input = switched.finish();
    const auto input_mutable_predicates = std::count_if(
        switched_input.ir.values.begin(), switched_input.ir.values.end(),
        [](const Value& value) {
            return value.type == ValueType::predicate && value.kind == ValueKind::mutable_local;
        });
    const auto switched_output = structured(switched_input);
    const auto output_mutable_predicates = std::count_if(
        switched_output.ir.values.begin(), switched_output.ir.values.end(),
        [](const Value& value) {
            return value.type == ValueType::predicate && value.kind == ValueKind::mutable_local;
        });
    check(output_mutable_predicates == input_mutable_predicates,
          "Reducible switch normalization does not synthesize dispatch flags");
    for (const auto choice : {0U, 1U, 2U, 3U, 4U})
        equivalent(switched_input, switched_output, {}, {choice});
    emits_msl(switched_output, "reducible switch with duplicate targets");
}

void test_literal_branch_folding() {
    Builder literal(3);
    literal.fragment_io();
    literal.legal.ir.blocks[0].terminator = Branch{LiteralBits{1, ValueType::predicate}, BlockId{1}, BlockId{2}};
    literal.color(1, 61);
    literal.legal.ir.blocks[1].terminator = Return{};
    literal.color(2, 62);
    literal.legal.ir.blocks[2].terminator = Return{};
    const auto literal_input = literal.finish();
    const auto literal_output = reducible(literal_input, "literal branch");
    equivalent(literal_input, literal_output);
    check(literal_output.ir.blocks.size() == 2 && literal_output.ir.operations.size() == 1 &&
              region_count<IfRegion>(literal_output) == 0,
          "Literal branch prunes its unreachable arm");

    Builder copied(3);
    copied.fragment_io();
    const auto condition = copied.value(ValueType::predicate);
    copied.op(0, {{}, Opcode::copy, Effects::none, {},
                  {LiteralBits{0, ValueType::predicate}}, {condition}, {}});
    copied.legal.ir.blocks[0].terminator = Branch{condition, BlockId{1}, BlockId{2}};
    copied.color(1, 71);
    copied.legal.ir.blocks[1].terminator = Return{};
    copied.color(2, 72);
    copied.legal.ir.blocks[2].terminator = Return{};
    const auto copied_input = copied.finish();
    const auto copied_output = reducible(copied_input, "copy-literal branch");
    equivalent(copied_input, copied_output);
    check(copied_output.ir.blocks.size() == 2 && copied_output.ir.operations.size() == 2 &&
              region_count<IfRegion>(copied_output) == 0,
          "Copy-literal branch prunes its unreachable arm");

    Builder phi_edge(5);
    phi_edge.fragment_io();
    const auto false_value = phi_edge.value(ValueType::predicate);
    const auto true_value = phi_edge.value(ValueType::predicate);
    phi_edge.op(0, {{}, Opcode::copy, Effects::none, {},
                    {LiteralBits{0, ValueType::predicate}}, {false_value}, {}});
    phi_edge.op(0, {{}, Opcode::copy, Effects::none, {},
                    {LiteralBits{1, ValueType::predicate}}, {true_value}, {}});
    phi_edge.legal.ir.blocks[0].terminator = Branch{
        LiteralBits{1, ValueType::predicate}, BlockId{1}, BlockId{2}};
    phi_edge.jump(1, 2);
    const auto merged = phi_edge.value(ValueType::predicate);
    phi_edge.legal.ir.blocks[2].phis = {
        {{}, merged, {{BlockId{0}, false_value}, {BlockId{1}, true_value}}}};
    phi_edge.legal.ir.blocks[2].terminator = Branch{merged, BlockId{3}, BlockId{4}};
    phi_edge.color(3, 73);
    phi_edge.legal.ir.blocks[3].terminator = Return{};
    phi_edge.color(4, 74);
    phi_edge.legal.ir.blocks[4].terminator = Return{};
    const auto phi_edge_input = phi_edge.finish();
    const auto phi_edge_output = structured(phi_edge_input);
    equivalent(phi_edge_input, phi_edge_output);
    // Four live CFG blocks remain, plus the edge block for the surviving phi copy.
    check(phi_edge_output.ir.blocks.size() == 5 && region_count<IfRegion>(phi_edge_output) == 0 &&
              region_count<ReturnRegion>(phi_edge_output) == 1,
          "Folding removes a live predecessor edge and its now-false return arm");
    check(std::all_of(phi_edge_output.ir.blocks.begin(), phi_edge_output.ir.blocks.end(),
                      [](const BasicBlock& block) { return block.phis.empty(); }),
          "Phi with a pruned predecessor lowers from its surviving input");
    emits_msl(phi_edge_output, "folded predecessor phi");
}

void test_msl_output_flow_regressions() {
    Builder missing(4);
    missing.fragment_io();
    missing.branch(0, 1, 2);
    missing.color(1, 81);
    missing.jump(1, 3);
    missing.jump(2, 3);
    missing.legal.ir.blocks[3].terminator = Return{};
    const auto missing_input = missing.finish();
    const auto missing_value_count = missing_input.ir.values.size();
    const auto missing_output = structured(missing_input);
    check(missing_output.ir.values.size() == missing_value_count,
          "Missing-output diamond does not synthesize dispatch values");
    equivalent(missing_input, missing_output, {}, {0});
    equivalent(missing_input, missing_output, {}, {1});
    rejects_missing_output(missing_output, "missing-output diamond");

    Builder fallback(9);
    fallback.fragment_io();
    fallback.jump(0, 1);
    fallback.branch(1, 2, 8);
    fallback.branch(2, 3, 6);
    fallback.branch(3, 4, 5);
    fallback.branch(4, 8, 1);
    fallback.jump(5, 2);
    fallback.jump(6, 7);
    fallback.jump(7, 1);
    fallback.color(8, 91);
    fallback.legal.ir.blocks[8].terminator = Return{};
    const auto fallback_input = fallback.finish();
    const auto fallback_value_count = fallback_input.ir.values.size();
    const auto fallback_output = structured(fallback_input);
    check(fallback_output.ir.values.size() > fallback_value_count,
          "Multi-exit nested loop exercises dispatch fallback");
    equivalent(fallback_input, fallback_output, {}, {0});
    equivalent(fallback_input, fallback_output, {}, {1, 0, 0});
    equivalent(fallback_input, fallback_output, {}, {1, 1, 1, 0, 0});
    equivalent(fallback_input, fallback_output, {}, {1, 1, 0, 1, 0, 0, 1, 0, 0});
    emits_msl(fallback_output, "returning dispatch fallback");
}

void test_exits_switches_and_returns() {
    Builder b(6);
    for (std::uint32_t i = 0; i < 6; ++i) b.trace(i);
    const auto selector = b.parameter();
    const auto switch_value = b.value();
    b.jump(0, 1);
    b.legal.ir.blocks[1].terminator = Switch{selector, {{4, BlockId{2}}, {7, BlockId{3}}, {8, BlockId{2}}}, BlockId{3}};
    b.legal.ir.blocks[3].phis = {{{}, switch_value, {{BlockId{1}, selector}}}};
    IoInfo io;
    io.semantic = IoSemantic::color;
    io.type = {ScalarType::u32, 1};
    io.data = {InputIndex{0}};
    b.op(3, {{}, Opcode::store_io, Effects::write_io, integer, {switch_value}, {}, io});
    b.jump(2, 4); b.jump(3, 4); b.jump(4, 5);
    b.legal.ir.blocks[5].terminator = Return{};
    const auto input = b.finish();
    const auto output = structured(input);
    for (auto selector : {4U, 7U, 8U, 99U}) equivalent(input, output, {selector});
    Builder early(3);
    early.branch(0, 1, 2);
    early.legal.ir.blocks[1].terminator = Return{};
    early.legal.ir.blocks[2].terminator = Return{};
    const auto returns = early.finish();
    equivalent(returns, structured(returns), {}, {0});
    equivalent(returns, structured(returns), {}, {1});
    Builder endless(1);
    endless.jump(0, 0);
    check(std::holds_alternative<StructuredProgram>(structure(endless.finish(), {})), "Exitless entry loop is representable");
    Builder duplicate(3);
    const auto incoming = duplicate.parameter();
    const auto merged = duplicate.value();
    duplicate.branch(0, 1, 1);
    duplicate.legal.ir.blocks[1].phis = {{{}, merged, {{BlockId{0}, incoming}}}};
    duplicate.jump(1, 2);
    duplicate.legal.ir.blocks[2].terminator = Return{{merged}};
    duplicate.legal.ir.functions[0].results = {ValueType::word32};
    const auto same = duplicate.finish();
    const auto same_output = structured(same);
    equivalent(same, same_output, {31}, {0});
    equivalent(same, same_output, {31}, {1});
}

void test_graph_sweep() {
    std::minstd_rand random(123);
    unsigned accepted = 0, irreducible = 0;
    for (unsigned trial = 0; trial < 100; ++trial) {
        Builder b(8);
        for (std::uint32_t block = 0; block < 8; ++block) {
            b.trace(block);
            if (block == 7) b.legal.ir.blocks[block].terminator = Return{};
            else b.branch(block, static_cast<std::uint32_t>(random() % 8), block + 1);
        }
        const auto input = b.finish();
        const auto result = structure(input, {});
        if (const auto* error = std::get_if<Diagnostic>(&result)) {
            check(error->code == DiagnosticCode::unsupported_capability, "Graph sweep only rejects irreducible graphs");
            ++irreducible;
            continue;
        }
        ++accepted;
        std::vector<std::uint64_t> choices(28);
        for (std::size_t i = 0; i < 20; ++i) choices[i] = random() % 2;
        equivalent(input, std::get<StructuredProgram>(result), {}, choices);
    }
    check(accepted > 10 && irreducible > 10, "Graph sweep exercises accepted and irreducible control flow");

    Builder functions(2);
    const auto first = functions.parameter(), second = functions.value();
    functions.legal.ir.functions[0].blocks = {BlockId{0}};
    functions.legal.ir.functions[0].results = {ValueType::word32};
    functions.legal.ir.functions.push_back({0, BlockId{1}, {BlockId{1}}, {second}, {ValueType::word32}});
    functions.legal.ir.blocks[1].function = FunctionId{1};
    functions.legal.ir.blocks[0].terminator = Return{{first}};
    functions.legal.ir.blocks[1].terminator = Return{{second}};
    functions.legal.ir.entry = FunctionId{1};
    const auto input = functions.finish();
    const auto output = structured(input);
    check(output.functions.size() == 2, "Every function receives its own structured root");
    equivalent(input, output, {53});
}

void test_validation() {
    const auto input = cycle_program(2);
    const auto original = structured(input);
    auto mutated = original;
    auto& reordered = std::get<SequenceRegion>(mutated.regions[mutated.functions[0].root.value]);
    std::swap(reordered.children[0], reordered.children[1]);
    rejected(validate(mutated), "Moving initialization after use is rejected", DiagnosticCode::undefined_read);
    mutated = original;
    for (auto& region : mutated.regions) if (std::holds_alternative<ReturnRegion>(region)) { region = SequenceRegion{}; break; }
    rejected(validate(mutated), "Dropping a return is rejected");
    mutated = original;
    mutated.regions.push_back(SequenceRegion{});
    rejected(validate(mutated), "Orphan region rejected");
    mutated = original;
    auto& root = std::get<SequenceRegion>(mutated.regions[mutated.functions[0].root.value]);
    root.children.push_back(root.children.front());
    rejected(validate(mutated), "Shared region rejected");
    mutated = original;
    std::get<SequenceRegion>(mutated.regions[mutated.functions[0].root.value]).children.push_back(mutated.functions[0].root);
    rejected(validate(mutated), "Region cycle rejected");
    mutated = original;
    for (auto& region : mutated.regions) if (auto* branch = std::get_if<IfRegion>(&region)) { branch->condition = word(1); break; }
    rejected(validate(mutated), "AST-only nonpredicate condition rejected");
    mutated = original;
    for (auto& region : mutated.regions) if (auto* returned = std::get_if<ReturnRegion>(&region)) { returned->values.clear(); break; }
    rejected(validate(mutated), "AST-only return signature rejected");
    mutated = original;
    for (auto& region : mutated.regions) if (auto* repeat = std::get_if<ContinueRegion>(&region)) { repeat->loop = mutated.functions[0].root; break; }
    rejected(validate(mutated), "Nonloop continue target rejected");
    mutated = original;
    mutated.ir.blocks[1].phis = input.ir.blocks[1].phis;
    rejected(validate(mutated), "Residual phi rejected");
    mutated = original;
    mutated.ir.values[input.ir.blocks[1].phis[0].output.value].kind = ValueKind::temporary;
    rejected(validate(mutated), "Phi destination must be mutable");
    mutated = original;
    mutated.ir.operations.push_back(mutated.ir.operations.front());
    rejected(validate(mutated), "Orphan operation rejected");
    mutated = original;
    mutated.ir.limits.nesting_depth = 1;
    rejected(validate(mutated), "Region depth limit enforced", DiagnosticCode::limit_exceeded);
    auto malformed = input;
    malformed.ir.blocks[1].phis[0].inputs.pop_back();
    check(std::holds_alternative<Diagnostic>(structure(malformed, {})), "Malformed input phi rejected before lowering");
    Limits limits;
    limits.values = static_cast<std::uint32_t>(input.ir.values.size());
    auto limited = structure(input, limits);
    check(std::holds_alternative<Diagnostic>(limited) && std::get<Diagnostic>(limited).code == DiagnosticCode::limit_exceeded,
          "New temporaries obey supplied limits");
    malformed = input;
    malformed.ir.limits.regions = 1;
    check(std::holds_alternative<Diagnostic>(structure(malformed, {})), "Inherited limits cannot be widened");
    malformed = input;
    malformed.ir.limits.analysis_steps = 1;
    check(std::holds_alternative<Diagnostic>(structure(malformed, {})), "Analysis limit enforced");
    Builder irreducible(4);
    irreducible.branch(0, 1, 2);
    irreducible.jump(1, 2);
    irreducible.branch(2, 1, 3);
    irreducible.legal.ir.blocks[3].terminator = Return{};
    auto unsupported = structure(irreducible.finish(), {});
    check(std::holds_alternative<Diagnostic>(unsupported) && std::get<Diagnostic>(unsupported).code == DiagnosticCode::unsupported_capability,
          "Irreducible loop rejected explicitly");
    Builder inner(5);
    inner.branch(0, 1, 2);
    inner.jump(1, 2);
    inner.branch(2, 1, 3);
    inner.branch(3, 0, 4);
    inner.legal.ir.blocks[4].terminator = Return{};
    unsupported = structure(inner.finish(), {});
    check(std::holds_alternative<Diagnostic>(unsupported) && std::get<Diagnostic>(unsupported).code == DiagnosticCode::unsupported_capability,
          "Irreducible inner cycle inside a single-entry outer SCC is rejected");
    Builder entry_phi(2);
    const auto initial = entry_phi.parameter();
    const auto uninitialized = entry_phi.value();
    entry_phi.legal.ir.blocks[0].phis = {{{}, uninitialized, {{BlockId{0}, initial}}}};
    entry_phi.branch(0, 0, 1);
    entry_phi.legal.ir.blocks[1].terminator = Return{{uninitialized}};
    entry_phi.legal.ir.functions[0].results = {ValueType::word32};
    unsupported = structure(entry_phi.finish(), {});
    check(std::holds_alternative<Diagnostic>(unsupported) && std::get<Diagnostic>(unsupported).code == DiagnosticCode::undefined_read,
          "Function-entry phi cannot obtain a value from an unexecuted backedge");
}

void test_preservation() {
    Builder b(1);
    const auto input = b.parameter(), output = b.value();
    NumericMode mode{Domain::floating, ScalarType::f32, Rounding::toward_negative, true, true, true, true};
    b.op(0, {{20, InstructionName::fadd}, Opcode::add, Effects::none, mode, {input, word(0)}, {output}, {}});
    b.legal.ir.blocks[0].terminator = Return{};
    b.legal.ir.dependencies.push_back({{FactKind::constant_buffer_word, 0, 0}, std::uint32_t{17}, {OriginKind::explicit_option, "synthetic"}});
    const auto legal = b.finish();
    const auto result = structured(legal);
    check(result.ir.operations[0].numeric == mode, "All numerical modifiers survive unchanged");
    check(result.ir.operations[0].source.pc == 20 && result.ir.operations[0].source.instruction == InstructionName::fadd, "Source location survives");
    check(result.ir.dependencies == legal.ir.dependencies && result.profile == legal.profile, "Dependencies and profile survive");
    Builder wide(1);
    const auto left = wide.parameter(ValueType::word64), right = wide.value(ValueType::word64);
    mode.scalar = ScalarType::f64;
    wide.op(0, {{}, Opcode::add, Effects::none, mode, {left, LiteralBits{0, ValueType::word64}}, {right}, {}});
    wide.legal.ir.blocks[0].terminator = Return{};
    const auto fp64 = structured(wide.finish());
    check(fp64.ir.operations[0].numeric == mode && fp64.ir.values[right.value].type == ValueType::word64,
          "FP64 operations and modifiers are preserved without narrowing");
    Builder buffer(1);
    const auto loaded = buffer.value();
    MemoryInfo memory;
    memory.constant_buffer_slot = 0;
    memory.resource = ResourceId{0};
    buffer.op(0, {{}, Opcode::load, Effects::read_memory, {}, {}, {loaded}, memory});
    buffer.legal.requirements.uses.push_back(BufferUse{ConstantBufferSource{0}, Access::read, {SizeKind::accessed_extent, 4, 4}});
    buffer.legal.ir.blocks[0].terminator = Return{};
    const auto resolved = structured(buffer.finish());
    check(std::get<MemoryInfo>(resolved.ir.operations[0].info).resource == ResourceId{0} && resolved.requirements.uses.size() == 1,
          "Resolved buffer metadata and requirements survive structural validation");
    auto missing = resolved;
    missing.requirements.uses.clear();
    rejected(validate(missing), "Missing resource requirement is rejected", DiagnosticCode::unresolved_resource);
}
} // namespace

int main() {
    try {
        test_parallel_copies();
        test_nested_loops();
        test_direct_reducible_regions();
        test_literal_branch_folding();
        test_msl_output_flow_regressions();
        test_exits_switches_and_returns();
        test_graph_sweep();
        test_validation();
        test_preservation();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "Unexpected test exception: " << error.what() << '\n';
    }
    if (failures) std::cerr << failures << " structured pipeline checks failed\n";
    return failures ? 1 : 0;
}
