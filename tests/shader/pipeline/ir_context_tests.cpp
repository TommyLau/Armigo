// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic tests for the concrete EmitterContext and the emit_ir phase.
// All programs are hand-built from independently chosen encodings; no game
// bytes, captured words or hashes appear here.

#include "pipeline/ir/emit_ir.h"
#include "pipeline/phases.h"
#include "zero_reads.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

// Extracts the success payload or reports a failure once and returns a
// fallback, keeping each test linear in the happy path. Returns by value so
// callers never borrow from a temporary variant.
template <class T, class Variant>
T expect(Variant&& result, std::string_view message) {
    if (auto* value = std::get_if<T>(&result)) return std::move(*value);
    if (const auto* failure = std::get_if<Diagnostic>(&result)) {
        std::cerr << "Unexpected diagnostic (" << failure->message << "): " << message << '\n';
    } else {
        std::cerr << "Unexpected variant state: " << message << '\n';
    }
    ++failures;
    return T{};
}

bool failed_with(const Result<RegisterProgram>& result, DiagnosticCode code) {
    const auto* failure = std::get_if<Diagnostic>(&result);
    return failure && failure->code == code && failure->phase == Phase::emit_ir;
}

Register gpr(std::uint16_t index) { return Register{RegisterFile::gpr, index}; }
Register pred(std::uint16_t index) { return Register{RegisterFile::predicate, index}; }
constexpr NumericMode f32_mode{Domain::floating, ScalarType::f32, Rounding::nearest_even};
constexpr NumericMode u32_mode{Domain::unsigned_integer, ScalarType::u32};

DecodedInstruction make_instruction(Pc pc, InstructionName name, InstructionPredicate predicate = {}) {
    DecodedInstruction decoded;
    decoded.pc = pc;
    decoded.name = name;
    decoded.predicate = predicate;
    return decoded;
}

// Head block: FADD then BRA to byte 40; exit block: a Return terminator.
DecodedProgram branch_program() {
    DecodedProgram decoded;
    decoded.stage = Stage::vertex;
    decoded.code_bytes = 64;
    decoded.entry = FunctionId{0};
    decoded.limits = Limits{};
    DecodedFunction function;
    function.entry_pc = 8;
    function.entry = BlockId{0};
    function.blocks = {BlockId{0}, BlockId{1}};
    decoded.functions.push_back(function);
    DecodedBlock head;
    head.function = FunctionId{0};
    head.begin_pc = 8;
    head.end_pc = 32;
    head.instructions.push_back(make_instruction(8, InstructionName::fadd));
    auto branch = make_instruction(16, InstructionName::bra);
    branch.fields.push_back({DecodedField::branch_absolute, 40});
    head.instructions.push_back(branch);
    DecodedBlock exit_block;
    exit_block.function = FunctionId{0};
    exit_block.begin_pc = 40;
    exit_block.end_pc = 48;
    exit_block.instructions.push_back(make_instruction(40, InstructionName::exit));
    decoded.blocks.push_back(std::move(head));
    decoded.blocks.push_back(std::move(exit_block));
    return decoded;
}

// Reads R4/R5, adds them in f32, writes R0.
Status emit_test_fadd(EmitterContext& context) {
    const auto& left = expect<Operand>(context.read(gpr(4)), "R4 is readable");
    const auto& right = expect<Operand>(context.read(gpr(5)), "R5 is readable");
    const auto& sum = expect<Operand>(
        context.binary(Opcode::add, left, right, f32_mode, ValueType::word32), "binary add emits");
    return context.write(gpr(0), sum);
}

// Jumps to the absolute byte target carried in DecodedField::branch_absolute.
Status emit_test_bra(EmitterContext& context) {
    const auto& target = expect<std::uint64_t>(
        decoded_field(context.instruction(), DecodedField::branch_absolute), "branch target field exists");
    const auto& block = expect<BlockId>(context.label(static_cast<Pc>(target)), "branch label is allocated");
    return context.terminate(Jump{block});
}

// Ends the program with an explicit return terminator.
Status emit_test_exit(EmitterContext& context) { return context.terminate(Return{}); }

constexpr std::array<EmitterBinding, 3> branch_bindings{{
    {InstructionName::fadd, emit_test_fadd},
    {InstructionName::bra, emit_test_bra},
    {InstructionName::exit, emit_test_exit},
}};

class EmptyFacts final : public FactSource {
    Knowledge<FactValue> lookup(FactKey) const override { return Unknown{"Test supplies no facts"}; }
};

class WordFacts final : public FactSource {
public:
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::constant_buffer_word) {
            return Known<FactValue>{std::uint32_t{7}, {OriginKind::explicit_option, "synthetic-word"}};
        }
        return Unknown{"Test fact was not supplied"};
    }
};

const Operation* find_operation(const Program& program, Opcode opcode) {
    for (const auto& operation : program.operations) {
        if (operation.opcode == opcode) return &operation;
    }
    return nullptr;
}

const Operation* find_copy_to(const Program& program, Register destination) {
    for (const auto& operation : program.operations) {
        if (operation.opcode != Opcode::copy || operation.outputs.size() != 1) continue;
        if (const auto* reg = std::get_if<Register>(&operation.outputs[0]); reg && *reg == destination) {
            return &operation;
        }
    }
    return nullptr;
}

void test_expected_operations() {
    EmptyFacts source;
    FactReader facts(source);
    auto result = emit_ir_with(branch_program(), facts, Limits{}, branch_bindings);
    const auto* program = std::get_if<RegisterProgram>(&result);
    check(program != nullptr, "Branch program emits a register program");
    if (!program) return;
    check(program->ir.functions.size() == 1 && program->ir.blocks.size() == 2,
          "Decoded functions and blocks map one to one");
    check(program->ir.entry == FunctionId{0}, "Program entry matches the decoded entry");
    if (const auto* jump = std::get_if<Jump>(&program->ir.blocks[0].terminator)) {
        check(jump->target == BlockId{1}, "Branch emitter terminates the head block");
    } else {
        check(false, "Branch emitter terminates the head block");
    }
    check(std::holds_alternative<Return>(program->ir.blocks[1].terminator),
          "Exit emitter terminates the exit block");
    check(program->ir.blocks[1].predecessors == std::vector<BlockId>{BlockId{0}},
          "Predecessors follow the installed terminator edges");
    const auto* add = find_operation(program->ir, Opcode::add);
    check(add != nullptr && add->inputs.size() == 2, "Test emitter produced a two-input add");
    check(add && add->source.pc == Pc{8} && add->source.instruction == InstructionName::fadd,
          "Operations carry the guest instruction as source location");
    check(find_copy_to(program->ir, gpr(0)) != nullptr,
          "Register writes append an explicit copy to the register");
    check(program->ir.blocks[0].operations.size() == 2, "Head block owns the add and the copy in order");
    check(std::holds_alternative<std::monostate>(validate(program->ir, IrForm::registers, Phase::emit_ir)),
          "Emitted branch program validates");
}

void test_pseudo_register_writes() {
    DecodedProgram decoded;
    decoded.entry = FunctionId{0};
    decoded.limits = Limits{};
    DecodedFunction function;
    function.entry = BlockId{0};
    function.blocks = {BlockId{0}};
    decoded.functions.push_back(function);
    DecodedBlock block;
    block.function = FunctionId{0};
    block.begin_pc = 8;
    block.end_pc = 16;
    block.instructions.push_back(make_instruction(8, InstructionName::mov));
    block.instructions.push_back(make_instruction(12, InstructionName::exit));
    decoded.blocks.push_back(std::move(block));

    struct Binding {
        static Status emit(EmitterContext& context) {
            const auto& zero = expect<Operand>(context.read(zero_register), "RZ is readable");
            const auto* zero_bits = std::get_if<LiteralBits>(&zero);
            check(zero_bits && zero_bits->bits == 0 && zero_bits->type == ValueType::word32,
                  "RZ reads resolve to a zero literal");
            const auto& truth = expect<Operand>(context.read(true_predicate), "PT is readable");
            const auto* truth_bits = std::get_if<LiteralBits>(&truth);
            check(truth_bits && truth_bits->bits == 1 && truth_bits->type == ValueType::predicate,
                  "PT reads resolve to a true literal");
            if (auto failure = context.write(zero_register, truth); std::holds_alternative<Diagnostic>(failure)) {
                return failure;
            }
            if (auto failure = context.write(true_predicate, zero); std::holds_alternative<Diagnostic>(failure)) {
                return failure;
            }
            return context.write(gpr(5), zero);
        }
    };
    const std::array<EmitterBinding, 2> bindings{{
        {InstructionName::mov, &Binding::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    EmptyFacts source;
    FactReader facts(source);
    auto result = emit_ir_with(decoded, facts, Limits{}, bindings);
    const auto* program = std::get_if<RegisterProgram>(&result);
    check(program != nullptr, "Pseudo-register program emits");
    if (!program) return;
    check(program->ir.operations.size() == 1, "RZ/PT writes are discarded, real writes remain");    const auto& copy = program->ir.operations.front();
    const auto* destination = std::get_if<Register>(&copy.outputs[0]);
    check(copy.opcode == Opcode::copy && destination && *destination == gpr(5),
          "The surviving write targets R5");
}

void test_multi_result_and_cc() {
    DecodedProgram decoded;
    decoded.entry = FunctionId{0};
    decoded.limits = Limits{};
    DecodedFunction function;
    function.entry = BlockId{0};
    function.blocks = {BlockId{0}};
    decoded.functions.push_back(function);
    DecodedBlock block;
    block.function = FunctionId{0};
    block.begin_pc = 8;
    block.end_pc = 16;
    block.instructions.push_back(make_instruction(8, InstructionName::iadd));
    block.instructions.push_back(make_instruction(12, InstructionName::isetp));
    block.instructions.push_back(make_instruction(16, InstructionName::exit));
    decoded.blocks.push_back(std::move(block));

    struct Carry {
        static Status emit(EmitterContext& context) {
            const std::array<Operand, 3> inputs{Operand{gpr(4)}, Operand{gpr(6)},
                                                Operand{LiteralBits{0, ValueType::predicate}}};
            const std::array<ValueType, 2> outputs{ValueType::word32, ValueType::predicate};
            auto result = context.emit(Opcode::add_carry, u32_mode, inputs, outputs, {}, Effects::none);
            const auto& values = expect<std::vector<ValueId>>(std::move(result), "add_carry emits");
            check(values.size() == 2, "add_carry emits two results");
            check(context.limits().values > 0, "Context exposes the phase limits");
            if (auto failure = context.write(gpr(8), values[0]); std::holds_alternative<Diagnostic>(failure)) {
                return failure;
            }
            return context.write(Register{RegisterFile::condition_code, 0}, values[1]);
        }
    };
    struct SetPredicate {
        static Status emit(EmitterContext& context) {
            const auto& left = expect<Operand>(context.read(gpr(4)), "R4 is readable");
            const auto& relation = expect<Operand>(
                context.compare(left, Operand{LiteralBits{4, ValueType::word32}},
                                ComparisonInfo{Compare::less, false}, u32_mode),
                "compare emits a predicate");
            return context.write(pred(0), relation);
        }
    };
    const std::array<EmitterBinding, 3> bindings{{
        {InstructionName::iadd, &Carry::emit},
        {InstructionName::isetp, &SetPredicate::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    EmptyFacts source;
    FactReader facts(source);
    auto result = emit_ir_with(decoded, facts, Limits{}, bindings);
    const auto* program = std::get_if<RegisterProgram>(&result);
    check(program != nullptr, "Multi-result program emits");
    if (!program) return;
    const auto* carry = find_operation(program->ir, Opcode::add_carry);
    check(carry != nullptr && carry->outputs.size() == 2 &&
              program->ir.values[std::get<ValueId>(carry->outputs[0]).value].type == ValueType::word32 &&
              program->ir.values[std::get<ValueId>(carry->outputs[1]).value].type == ValueType::predicate,
          "Carry results are typed word32 and predicate");
    check(find_copy_to(program->ir, Register{RegisterFile::condition_code, 0}) != nullptr,
          "CC writes are explicit copies to the indexed flag");
    check(find_copy_to(program->ir, pred(0)) != nullptr, "Predicate writes reach P0");
    check(program->ir.blocks[0].operations.size() == 5,
          "Both instructions expanded into the same block in order");
}

void test_predication() {
    DecodedProgram decoded;
    decoded.entry = FunctionId{0};
    decoded.limits = Limits{};
    DecodedFunction function;
    function.entry = BlockId{0};
    function.blocks = {BlockId{0}};
    decoded.functions.push_back(function);
    DecodedBlock block;
    block.function = FunctionId{0};
    block.begin_pc = 8;
    block.end_pc = 32;
    InstructionPredicate guarded;
    guarded.index = 2;
    guarded.enabled = true;
    block.instructions.push_back(make_instruction(8, InstructionName::fadd, guarded));
    InstructionPredicate inverted = guarded;
    inverted.inverted = true;
    block.instructions.push_back(make_instruction(16, InstructionName::mov, inverted));
    InstructionPredicate unconditional;
    unconditional.index = 7;
    unconditional.enabled = true;
    block.instructions.push_back(make_instruction(24, InstructionName::exit, unconditional));
    decoded.blocks.push_back(std::move(block));

    struct Mov {
        static Status emit(EmitterContext& context) {
            const auto& value = expect<Operand>(context.read(gpr(4)), "R4 is readable");
            return context.write(gpr(5), value);
        }
    };
    const std::array<EmitterBinding, 3> bindings{{
        {InstructionName::fadd, emit_test_fadd},
        {InstructionName::mov, &Mov::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    EmptyFacts source;
    FactReader facts(source);
    auto result = emit_ir_with(decoded, facts, Limits{}, bindings);
    const auto* program = std::get_if<RegisterProgram>(&result);
    check(program != nullptr, "Predicated program emits");
    if (!program) return;
    check(program->ir.blocks.size() == 4, "Complementary instructions share one if/else region");
    const auto& entry = program->ir.blocks[0];
    const auto* branch = std::get_if<Branch>(&entry.terminator);
    check(branch != nullptr, "The guard branches on the instruction predicate");
    check(branch && std::get<Register>(branch->condition) == pred(2), "The guard reads P2 once");
    check(entry.operations.empty(), "No destination write happens before the guard branch");
    check(find_operation(program->ir, Opcode::logical_not) == nullptr,
          "The complementary arm uses the false edge without reevaluating the predicate");
    check(std::holds_alternative<Jump>(program->ir.blocks[1].terminator),
          "The first guarded body jumps to its continuation");
    check(std::holds_alternative<Jump>(program->ir.blocks[3].terminator),
          "The second guarded body jumps to its continuation");
    check(branch && branch->when_true == BlockId{1} && branch->when_false == BlockId{3},
          "The two expansions execute on mutually exclusive branch edges");
    check(std::get<Jump>(program->ir.blocks[1].terminator).target == BlockId{2} &&
              std::get<Jump>(program->ir.blocks[3].terminator).target == BlockId{2},
          "Both arms join at the same continuation");
    check(std::holds_alternative<Return>(program->ir.blocks[2].terminator),
          "An always-true P7 predicate does not create a guard");
}

void test_missing_emitter() {
    auto decoded = branch_program();
    // FP64 arithmetic is intentionally unregistered, locally and globally.
    decoded.blocks[0].instructions[0].name = InstructionName::dadd;
    EmptyFacts source;
    FactReader facts(source);
    auto result = emit_ir_with(decoded, facts, Limits{}, branch_bindings);
    check(failed_with(result, DiagnosticCode::unsupported_instruction),
          "An unregistered instruction fails the phase");
    const auto& failure = std::get<Diagnostic>(result);
    check(failure.pc == Pc{8} && failure.instruction == InstructionName::dadd,
          "Missing-emitter diagnostics retain the guest instruction");
    // The global table has no row for it either.
    auto global = emit_ir(decoded, facts, Limits{});
    check(failed_with(global, DiagnosticCode::unsupported_instruction),
          "emit_ir rejects instructions that no registry row covers");
    // Malformed local tables are rejected up front.
    const std::array<EmitterBinding, 2> duplicated{{
        {InstructionName::fadd, emit_test_fadd},
        {InstructionName::fadd, emit_test_fadd},
    }};
    auto bad_table = emit_ir_with(branch_program(), facts, Limits{}, duplicated);
    check(failed_with(bad_table, DiagnosticCode::duplicate_dispatch), "Duplicate local bindings are rejected");
}

void test_limits() {
    EmptyFacts source;
    FactReader facts(source);
    for (const bool inherited : {false, true}) {
        for (const bool bytes : {false, true}) {
            auto decoded = branch_program();
            Limits supplied;
            auto& limits = inherited ? decoded.limits : supplied;
            if (bytes) limits.code_bytes = 32;
            else limits.instructions = 2;
            check(failed_with(emit_ir_with(decoded, facts, supplied, branch_bindings),
                              DiagnosticCode::limit_exceeded),
                  "Supplied and inherited byte/instruction ceilings apply before dispatch");
            // An unsupported first instruction must not mask the preflight error.
            decoded.blocks[0].instructions[0].name = InstructionName::dadd;
            check(failed_with(emit_ir_with(decoded, facts, supplied, branch_bindings),
                              DiagnosticCode::limit_exceeded),
                  "Limit preflight runs before any instruction callback");
        }
    }
    Limits exact;
    exact.code_bytes = 64;
    exact.instructions = 3;
    check(std::holds_alternative<RegisterProgram>(emit_ir_with(branch_program(), facts, exact, branch_bindings)),
          "Exact byte and instruction ceilings are inclusive");
    Limits small_operations;
    small_operations.operations = 1; // FADD expands to add + copy.
    auto operation_limit = emit_ir_with(branch_program(), facts, small_operations, branch_bindings);
    check(failed_with(operation_limit, DiagnosticCode::limit_exceeded), "Operation limits bound the expansion");

    DecodedProgram decoded;
    decoded.entry = FunctionId{0};
    decoded.limits = Limits{};
    DecodedFunction function;
    function.entry = BlockId{0};
    function.blocks = {BlockId{0}};
    decoded.functions.push_back(function);
    DecodedBlock block;
    block.function = FunctionId{0};
    block.begin_pc = 8;
    block.end_pc = 16;
    InstructionPredicate guarded;
    guarded.index = 1;
    guarded.enabled = true;
    block.instructions.push_back(make_instruction(8, InstructionName::mov, guarded));
    block.instructions.push_back(make_instruction(12, InstructionName::exit));
    decoded.blocks.push_back(std::move(block));
    struct Mov {
        static Status emit(EmitterContext& context) {
            const auto& value = expect<Operand>(context.read(gpr(4)), "R4 is readable");
            return context.write(gpr(5), value);
        }
    };
    const std::array<EmitterBinding, 2> bindings{{
        {InstructionName::mov, &Mov::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    Limits small_blocks;
    small_blocks.blocks = 2; // A guard needs entry + body + continuation.
    auto block_limit = emit_ir_with(decoded, facts, small_blocks, bindings);
    check(failed_with(block_limit, DiagnosticCode::limit_exceeded), "Block limits bound guard creation");

    // Inherited limits also apply: the decoded program can lower the ceiling.
    auto tightened = decoded;
    tightened.limits.blocks = 1;
    auto inherited = emit_ir_with(tightened, facts, Limits{}, bindings);
    check(failed_with(inherited, DiagnosticCode::limit_exceeded),
          "Phase limits are the component-wise minimum with the decoded limits");

    auto complementary = decoded;
    guarded.inverted = true;
    complementary.blocks[0].instructions.insert(complementary.blocks[0].instructions.begin() + 1,
        make_instruction(10, InstructionName::mov, guarded));
    Limits pair_limits;
    pair_limits.blocks = 4;
    pair_limits.edges = 4;
    check(std::holds_alternative<RegisterProgram>(emit_ir_with(complementary, facts, pair_limits, bindings)),
          "A complementary pair fits exactly four blocks and four edges");
    pair_limits.blocks = 3;
    check(failed_with(emit_ir_with(complementary, facts, pair_limits, bindings), DiagnosticCode::limit_exceeded),
          "The alternative body obeys the block limit");
    pair_limits.blocks = 4;
    pair_limits.edges = 3;
    check(failed_with(emit_ir_with(complementary, facts, pair_limits, bindings), DiagnosticCode::limit_exceeded),
          "The alternative arm's join obeys the edge limit");
}

void test_unterminated_block() {
    auto decoded = branch_program();
    decoded.blocks[0].instructions.pop_back(); // Drop the BRA; nothing terminates the head.
    EmptyFacts source;
    FactReader facts(source);
    const std::array<EmitterBinding, 2> bindings{{
        {InstructionName::fadd, emit_test_fadd},
        {InstructionName::exit, emit_test_exit},
    }};
    auto result = emit_ir_with(decoded, facts, Limits{}, bindings);
    check(failed_with(result, DiagnosticCode::invalid_ir),
          "A block without an executable terminator fails the phase");
}

void test_fallthrough_and_entry() {
    EmptyFacts source;
    FactReader facts(source);
    for (const bool guarded : {false, true}) {
        auto decoded = branch_program();
        decoded.blocks[0].instructions.pop_back();
        decoded.blocks[0].successors = {BlockId{1}};
        decoded.blocks[1].predecessors = {BlockId{0}};
        if (guarded) decoded.blocks[0].instructions[0].predicate = {0, false, true};
        auto result = emit_ir_with(decoded, facts, Limits{}, branch_bindings);
        const auto* program = std::get_if<RegisterProgram>(&result);
        check(program != nullptr, "Ordinary and guarded final instructions materialize fallthrough");
        if (!program) continue;
        const auto entry = program->ir.functions[0].entry;
        auto tail = entry;
        if (guarded) {
            const auto* branch = std::get_if<Branch>(&program->ir.blocks[entry.value].terminator);
            check(branch != nullptr, "Guarded fallthrough retains its branch");
            if (!branch) continue;
            tail = branch->when_false;
        }
        const auto* jump = std::get_if<Jump>(&program->ir.blocks[tail.value].terminator);
        check(jump && std::holds_alternative<Return>(program->ir.blocks[jump->target.value].terminator),
              "The open continuation jumps to the decoded successor");
        if (jump) check(program->ir.blocks[jump->target.value].predecessors == std::vector<BlockId>{tail},
                        "Fallthrough predecessor names the current continuation");

        decoded.blocks[0].successors = {BlockId{99}};
        check(failed_with(emit_ir_with(decoded, facts, Limits{}, branch_bindings), DiagnosticCode::malformed_input),
              "An out-of-range fallthrough is rejected");
        decoded.blocks[0].successors = {BlockId{1}, BlockId{1}};
        check(failed_with(emit_ir_with(decoded, facts, Limits{}, branch_bindings), DiagnosticCode::malformed_input),
              "Ambiguous fallthrough successors are rejected");
        decoded.blocks[0].successors = {BlockId{1}};
        decoded.blocks[1].begin_pc = 48;
        check(failed_with(emit_ir_with(decoded, facts, Limits{}, branch_bindings), DiagnosticCode::invalid_ir),
              "A nonadjacent successor cannot stand in for fallthrough");
    }

    for (const auto name : {InstructionName::exit, InstructionName::kil}) {
        auto decoded = branch_program();
        decoded.stage = Stage::fragment;
        auto flow = make_instruction(24, name, {0, false, true});
        flow.fields.push_back({DecodedField::condition, 15});
        decoded.blocks[0].instructions = {flow};
        decoded.blocks[0].successors = {BlockId{1}};
        decoded.blocks[1].predecessors = {BlockId{0}};
        // Use the real flow emitters: they leave the current block at the
        // already-connected decoded continuation, unlike the generic guard.
        auto result = emit_ir(decoded, facts, Limits{});
        const auto* program = std::get_if<RegisterProgram>(&result);
        check(program != nullptr, "Predicated flow retains an already-entered fallthrough successor");
        if (!program) continue;
        const auto entry = program->ir.functions[0].entry;
        const auto* branch = std::get_if<Branch>(&program->ir.blocks[entry.value].terminator);
        check(branch && std::holds_alternative<Return>(program->ir.blocks[branch->when_false.value].terminator),
              "Flow fallthrough stays executable without an artificial self-loop");
        check(program->ir.blocks.size() == 3,
              "Flow-owned predication is not wrapped in a second guard");
        if (name == InstructionName::kil) {
            const auto* discard = find_operation(program->ir, Opcode::discard);
            check(discard && discard->source.pc == Pc{24}, "Guarded discard remains an explicit effect");
        }
    }

    auto loop = branch_program();
    loop.blocks[0].instructions.erase(loop.blocks[0].instructions.begin());
    loop.blocks[0].instructions[0].fields[0].bits = 8;
    loop.blocks.resize(1);
    loop.functions[0].blocks = {BlockId{0}};
    auto result = emit_ir_with(loop, facts, Limits{}, branch_bindings);
    const auto* program = std::get_if<RegisterProgram>(&result);
    check(program != nullptr, "A decoded entry backedge emits with a preheader");
    if (program) {
        const auto entry = program->ir.functions[0].entry;
        check(program->ir.blocks.size() == 2 && program->ir.blocks[entry.value].predecessors.empty(),
              "Function entry is predecessor-free even when the decoded entry loops");
        const auto* jump = std::get_if<Jump>(&program->ir.blocks[entry.value].terminator);
        check(jump && jump->target != entry && program->ir.blocks[jump->target.value].predecessors.size() == 2,
              "The loop header merges its backedge with the ordinary preheader edge");
        check(std::holds_alternative<SsaProgram>(construct_ssa(*program, Limits{})),
              "The emitted entry loop passes SSA construction");
    }
    Limits one_block;
    one_block.blocks = 1;
    check(failed_with(emit_ir_with(loop, facts, one_block, branch_bindings), DiagnosticCode::limit_exceeded),
          "Entry preheaders obey the block limit");
    Limits one_edge;
    one_edge.edges = 1;
    check(failed_with(emit_ir_with(loop, facts, one_edge, branch_bindings), DiagnosticCode::limit_exceeded),
          "Entry preheaders obey the edge limit");
}

Status emit_guard_fixture(EmitterContext& context) {
    switch (context.instruction().name) {
    case InstructionName::isetp: {
        // Independent runtime words keep both guards feasible; constants
        // would let SSA prove the deliberately undefined arms unreachable.
        for (std::uint16_t index = 0; index < 2; ++index) {
            auto condition = context.compare(ConstantWord{0, std::uint32_t(index) * 4},
                LiteralBits{0, ValueType::word32}, ComparisonInfo{Compare::not_equal, false}, u32_mode);
            if (const auto* failure = std::get_if<Diagnostic>(&condition)) return *failure;
            if (auto status = context.write(pred(index), std::get<Operand>(condition));
                std::holds_alternative<Diagnostic>(status)) return status;
        }
        return std::monostate{};
    }
    case InstructionName::psetp:
    case InstructionName::r2p: return context.write(pred(0), LiteralBits{0, ValueType::predicate});
    case InstructionName::mov: return context.write(gpr(2), LiteralBits{7, ValueType::word32});
    case InstructionName::ast: {
        IoInfo io;
        io.semantic = IoSemantic::color;
        io.type = {ScalarType::u32, 1};
        io.data = {InputIndex{0}};
        const std::array<Operand, 1> inputs{gpr(2)};
        return context.store_io(io, inputs, u32_mode);
    }
    case InstructionName::bar: {
        MemoryInfo info;
        info.space = AddressSpace::shared;
        info.access = Access::read_write;
        info.order = MemoryOrder::acquire_release;
        info.scope = MemoryScope::workgroup;
        auto result = context.emit(Opcode::barrier, {}, {}, {}, info, Effects::barrier);
        if (const auto* failure = std::get_if<Diagnostic>(&result)) return *failure;
        return std::monostate{};
    }
    case InstructionName::nop: return std::monostate{};
    default: return context.error(DiagnosticCode::unsupported_instruction, "Unsupported synthetic guard instruction");
    }
}

constexpr std::array<EmitterBinding, 9> guard_bindings{{
    {InstructionName::isetp, emit_guard_fixture}, {InstructionName::psetp, emit_guard_fixture},
    {InstructionName::mov, emit_guard_fixture}, {InstructionName::ast, emit_guard_fixture},
    {InstructionName::bar, emit_guard_fixture}, {InstructionName::nop, emit_guard_fixture},
    {InstructionName::r2p, emit_guard_fixture},
    {InstructionName::bra, emit_test_bra}, {InstructionName::exit, emit_test_exit},
}};

DecodedProgram guard_program(bool inverted = false) {
    auto decoded = branch_program();
    decoded.blocks.resize(1);
    decoded.functions[0].blocks = {BlockId{0}};
    auto& block = decoded.blocks[0];
    block.end_pc = 40;
    block.instructions = {
        make_instruction(8, InstructionName::isetp),
        make_instruction(16, InstructionName::mov, {0, inverted, true}),
        make_instruction(24, InstructionName::ast, {0, inverted, true}),
        make_instruction(32, InstructionName::exit),
    };
    return decoded;
}

std::size_t branch_count(const Program& program) {
    return std::count_if(program.blocks.begin(), program.blocks.end(),
        [](const BasicBlock& block) { return std::holds_alternative<Branch>(block.terminator); });
}

void test_guard_runs() {
    EmptyFacts source;
    FactReader facts(source);
    for (const bool inverted : {false, true}) {
        auto result = emit_ir_with(guard_program(inverted), facts, Limits{}, guard_bindings);
        const auto* program = std::get_if<RegisterProgram>(&result);
        check(program != nullptr, "Identically guarded write/use sequence emits");
        if (!program) continue;
        check(branch_count(program->ir) == 1, "Adjacent identical guards share one branch");
        const auto entry = program->ir.functions[0].entry;
        const auto& branch = std::get<Branch>(program->ir.blocks[entry.value].terminator);
        const auto& body = program->ir.blocks[branch.when_true.value];
        check(body.operations.size() == 2 &&
                  program->ir.operations[body.operations[0].value].source.pc == Pc{16} &&
                  program->ir.operations[body.operations[1].value].source.pc == Pc{24},
              "The shared body preserves instruction order and source locations");
        check(program->ir.blocks[branch.when_false.value].operations.empty(),
              "The guard's false edge skips every write and effect");
        check(std::holds_alternative<SsaProgram>(construct_ssa(*program, Limits{})),
              "A guarded definition dominates the adjacent guarded effect through SSA");
    }

    for (const bool change_polarity : {false, true}) {
        auto decoded = guard_program();
        if (change_polarity) decoded.blocks[0].instructions[2].predicate.inverted = true;
        else decoded.blocks[0].instructions[2].predicate.index = 1;
        auto result = emit_ir_with(decoded, facts, Limits{}, guard_bindings);
        const auto* program = std::get_if<RegisterProgram>(&result);
        check(program && branch_count(program->ir) == (change_polarity ? 1u : 2u),
              "Only opposite polarities of the same predicate share an if/else");
        if (program) {
            auto ssa = construct_ssa(*program, Limits{});
            const auto* built = std::get_if<SsaProgram>(&ssa);
            check(built && test::entry_zero_reads(built->ir, ValueType::word32) > 0,
                  "Pairing does not supply a definition from the other arm");
        }
    }
    for (const auto access : {Access::write, Access::read_write}) {
        for (const bool complementary : {false, true}) {
            auto decoded = guard_program();
            decoded.blocks[0].instructions[2].predicate.inverted = complementary;
            auto writer = make_instruction(20, InstructionName::psetp, {0, false, true});
            writer.registers.push_back({RegisterRole::predicate_destination, 0, pred(0), access});
            decoded.blocks[0].instructions.insert(decoded.blocks[0].instructions.begin() + 2, writer);
            auto result = emit_ir_with(decoded, facts, Limits{}, guard_bindings);
            const auto* program = std::get_if<RegisterProgram>(&result);
            check(program && branch_count(program->ir) == 3,
                  "Guard-register writes split both sides of a run and force predicate reevaluation");
        }
    }
    auto predicate_mask = guard_program();
    predicate_mask.blocks[0].instructions.insert(predicate_mask.blocks[0].instructions.begin() + 2,
        make_instruction(20, InstructionName::r2p, {0, false, true}));
    auto mask_result = emit_ir_with(predicate_mask, facts, Limits{}, guard_bindings);
    const auto* mask_program = std::get_if<RegisterProgram>(&mask_result);
    check(mask_program && branch_count(mask_program->ir) == 3,
          "Dynamic predicate-mask writes cannot merge even with incomplete decoded write metadata");
    auto barrier = guard_program();
    barrier.blocks[0].instructions.insert(barrier.blocks[0].instructions.begin() + 2,
        make_instruction(20, InstructionName::bar, {0, false, true}));
    auto barrier_result = emit_ir_with(barrier, facts, Limits{}, guard_bindings);
    const auto* barrier_program = std::get_if<RegisterProgram>(&barrier_result);
    check(barrier_program && branch_count(barrier_program->ir) == 3,
          "Synchronization effects are not absorbed into guard runs");

    // SSA proves these correlated paths even though emission cannot merge
    // across an unguarded instruction or a decoded block boundary.
    for (const bool split_block : {false, true}) {
        auto decoded = guard_program();
        if (split_block) {
            auto tail = decoded.blocks[0];
            tail.begin_pc = 24;
            tail.instructions.erase(tail.instructions.begin(), tail.instructions.begin() + 2);
            tail.predecessors = {BlockId{0}};
            decoded.blocks[0].end_pc = 24;
            decoded.blocks[0].instructions.resize(2);
            decoded.blocks[0].successors = {BlockId{1}};
            decoded.blocks.push_back(std::move(tail));
            decoded.functions[0].blocks.push_back(BlockId{1});
        } else {
            decoded.blocks[0].instructions.insert(decoded.blocks[0].instructions.begin() + 2,
                make_instruction(20, InstructionName::nop));
        }
        auto result = emit_ir_with(decoded, facts, Limits{}, guard_bindings);
        const auto* program = std::get_if<RegisterProgram>(&result);
        check(program && branch_count(program->ir) == 2, "Guard merging respects instruction and block boundaries");
        if (!program) continue;
        auto ssa = construct_ssa(*program, Limits{});
        check(std::holds_alternative<SsaProgram>(ssa),
              "SSA proves correlated paths across instruction and block boundaries");
    }
}

void test_facts_and_assign() {
    WordFacts source;
    FactReader facts(source);
    DecodedProgram decoded;
    decoded.entry = FunctionId{0};
    decoded.limits = Limits{};
    DecodedFunction function;
    function.entry = BlockId{0};
    function.blocks = {BlockId{0}};
    decoded.functions.push_back(function);
    DecodedBlock block;
    block.function = FunctionId{0};
    block.begin_pc = 8;
    block.end_pc = 16;
    block.instructions.push_back(make_instruction(8, InstructionName::ldc));
    block.instructions.push_back(make_instruction(12, InstructionName::exit));
    decoded.blocks.push_back(std::move(block));

    struct LoadConstant {
        static Status emit(EmitterContext& context) {
            auto word = context.facts().require<FactKind::constant_buffer_word>(
                fact_query::constant_word(1, 4), Phase::emit_ir,
                SourceLocation{context.instruction().pc, context.instruction().name});
            const auto& bits = expect<std::uint32_t>(std::move(word), "constant word fact is known");
            // A compiler local may be assigned repeatedly before SSA renames it.
            const auto& id = expect<ValueId>(context.temporary(ValueType::word32), "local is allocated");
            if (auto failure = context.assign(id, Operand{LiteralBits{bits + 1, ValueType::word32}});
                std::holds_alternative<Diagnostic>(failure)) return failure;
            if (auto failure = context.assign(id, Operand{LiteralBits{bits, ValueType::word32}});
                std::holds_alternative<Diagnostic>(failure)) return failure;
            return context.write(gpr(6), Operand{id});
        }
    };
    const std::array<EmitterBinding, 2> bindings{{
        {InstructionName::ldc, &LoadConstant::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    auto result = emit_ir_with(decoded, facts, Limits{}, bindings);
    const auto* program = std::get_if<RegisterProgram>(&result);
    check(program != nullptr, "Fact-consuming program emits");
    if (!program) return;
    check(program->ir.dependencies.size() == 1 &&
              program->ir.dependencies[0].key.kind == FactKind::constant_buffer_word,
          "Consumed facts become program dependencies");
    check(std::get<std::uint32_t>(program->ir.dependencies[0].value) == 7,
          "Dependency values keep the consumed fact value");
    check(!program->ir.operations.empty() && program->ir.operations[0].opcode == Opcode::copy &&
              std::holds_alternative<ValueId>(program->ir.operations[0].outputs[0]),
          "Assignments append copies into the compiler local");
}

void test_signature_diagnostics() {
    DecodedProgram decoded;
    decoded.entry = FunctionId{0};
    decoded.limits = Limits{};
    DecodedFunction function;
    function.entry = BlockId{0};
    function.blocks = {BlockId{0}};
    decoded.functions.push_back(function);
    DecodedBlock block;
    block.function = FunctionId{0};
    block.begin_pc = 8;
    block.end_pc = 16;
    block.instructions.push_back(make_instruction(8, InstructionName::fmul));
    block.instructions.push_back(make_instruction(12, InstructionName::exit));
    decoded.blocks.push_back(std::move(block));

    struct WrongArity {
        static Status emit(EmitterContext& context) {
            const std::array<Operand, 1> inputs{Operand{gpr(4)}};
            const std::array<ValueType, 1> outputs{ValueType::word32};
            auto emitted = context.emit(Opcode::add, f32_mode, inputs, outputs, {}, Effects::none);
            if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
            return std::monostate{};
        }
    };
    struct BadStorage {
        static Status emit(EmitterContext& context) {
            return context.write(gpr(4), Operand{LiteralBits{1, ValueType::predicate}});
        }
    };
    struct UnknownOpcode {
        static Status emit(EmitterContext& context) {
            const std::array<Operand, 1> inputs{Operand{gpr(4)}};
            const std::array<ValueType, 1> outputs{ValueType::word32};
            auto emitted = context.emit(static_cast<Opcode>(static_cast<std::uint16_t>(Opcode::count) + 1),
                                        f32_mode, inputs, outputs, {}, Effects::none);
            if (const auto* failure = std::get_if<Diagnostic>(&emitted)) return *failure;
            return std::monostate{};
        }
    };
    struct DoubleTerminate {
        static Status emit(EmitterContext& context) {
            const auto& block = expect<BlockId>(context.create_block(), "spare block is created");
            if (auto failure = context.terminate(Jump{block}); std::holds_alternative<Diagnostic>(failure)) {
                return failure;
            }
            return context.terminate(Jump{block});
        }
    };
    EmptyFacts source;
    FactReader facts(source);
    const std::array<EmitterBinding, 2> arity_bindings{{
        {InstructionName::fmul, &WrongArity::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    check(failed_with(emit_ir_with(decoded, facts, Limits{}, arity_bindings), DiagnosticCode::invalid_ir),
          "Emit rejects operands outside the opcode arity");
    const std::array<EmitterBinding, 2> storage_bindings{{
        {InstructionName::fmul, &BadStorage::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    check(failed_with(emit_ir_with(decoded, facts, Limits{}, storage_bindings), DiagnosticCode::invalid_ir),
          "Writes reject operands with the wrong storage type");
    const std::array<EmitterBinding, 2> opcode_bindings{{
        {InstructionName::fmul, &UnknownOpcode::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    check(failed_with(emit_ir_with(decoded, facts, Limits{}, opcode_bindings),
                      DiagnosticCode::unsupported_instruction),
          "Unknown IR opcodes are rejected");
    const std::array<EmitterBinding, 2> terminate_bindings{{
        {InstructionName::fmul, &DoubleTerminate::emit},
        {InstructionName::exit, emit_test_exit},
    }};
    check(failed_with(emit_ir_with(decoded, facts, Limits{}, terminate_bindings), DiagnosticCode::invalid_ir),
          "A block accepts only one terminator");
}
} // namespace

int main() {
    test_expected_operations();
    test_pseudo_register_writes();
    test_multi_result_and_cc();
    test_predication();
    test_missing_emitter();
    test_limits();
    test_unterminated_block();
    test_fallthrough_and_entry();
    test_guard_runs();
    test_facts_and_assign();
    test_signature_diagnostics();
    if (failures) return 1;
    std::cout << "Shader pipeline IR context passed\n";
    return 0;
}
