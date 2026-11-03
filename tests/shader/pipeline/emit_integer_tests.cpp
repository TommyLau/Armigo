// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Tests for the integer/move/predicate/condition-code emitter families
// (pipeline lane 6), independently authored synthetic cases. Each test builds
// a decoded instruction or decodes a concrete instruction word through the
// published ALU rules, dispatches it through the emitter registry into a
// recording EmitterContext, and checks the emitted IR operations, the flag
// writes, and the structured diagnostics for unsupported modifiers.

#include "pipeline/decoder.h"
#include "pipeline/dispatch.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
bool is_literal(const Operand& operand, std::uint32_t bits) {
    const auto* literal = std::get_if<LiteralBits>(&operand);
    return literal && literal->type == ValueType::word32 && literal->bits == bits;
}
Register gpr(std::uint16_t index) { return Register{RegisterFile::gpr, index}; }
Register pred(std::uint16_t index) { return Register{RegisterFile::predicate, index}; }
Register cc(ConditionCode code) {
    return Register{RegisterFile::condition_code, static_cast<std::uint16_t>(code)};
}

// Mirrors a decode-table row: pattern index 0 is instruction bit 63.
constexpr std::uint64_t opcode_pattern(std::string_view pattern) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < pattern.size(); ++index) {
        if (pattern[index] == '1') value |= std::uint64_t{1} << (63 - index);
    }
    return value;
}

constexpr std::uint64_t set_bits(std::uint64_t word, unsigned first, unsigned width,
                                 std::uint64_t value) {
    const auto clear = ~(((std::uint64_t{1} << width) - 1) << first);
    return (word & clear) | (value << first);
}

DecodedInstruction decode_word(std::uint64_t word, std::string_view message) {
    const DecodeRule* rule = nullptr;
    for (const auto& candidate : alu_decode_rules()) {
        if ((word & candidate.mask) == candidate.value) {
            rule = &candidate;
            break;
        }
    }
    check(rule != nullptr, message);
    if (rule == nullptr) return {};
    auto result = rule->decode(40, RawInstruction{word});
    check(std::holds_alternative<DecodedInstruction>(result), message);
    if (const auto* error = std::get_if<Diagnostic>(&result)) {
        std::cerr << "  decode error: " << error->message << '\n';
        return {};
    }
    return std::get<DecodedInstruction>(std::move(result));
}

SourceModifiers modifiers(bool invert) {
    return SourceModifiers{false, invert, Subword::full, ScalarType::u32};
}

// Hand-built decoded instruction with the lane's documented field values.
struct Instr {
    DecodedInstruction inst;
    explicit Instr(InstructionName name, Pc pc = 8) { inst.pc = pc; inst.name = name; }
    Instr& field(DecodedField key, std::uint64_t bits) {
        inst.fields.push_back({key, bits});
        return *this;
    }
    Instr& word_source(SourceRole role, std::uint16_t index, bool invert = false) {
        inst.sources.push_back({role, gpr(index), modifiers(invert)});
        return *this;
    }
    Instr& literal_source(SourceRole role, std::uint32_t bits) {
        inst.sources.push_back({role, LiteralBits{bits, ValueType::word32}, modifiers(false)});
        return *this;
    }
    Instr& cbuf_source(SourceRole role, std::uint8_t slot, std::uint32_t byte_offset) {
        inst.sources.push_back({role, ConstantWord{slot, byte_offset}, modifiers(false)});
        return *this;
    }
    Instr& register_source(SourceRole role, Register reg, bool invert = false) {
        inst.sources.push_back({role, reg, modifiers(invert)});
        return *this;
    }
    Instr& predicate_source(std::uint16_t index, bool invert = false) {
        inst.sources.push_back({SourceRole::predicate, pred(index), modifiers(invert)});
        return *this;
    }
    Instr& destination(std::uint16_t index) {
        inst.registers.push_back({RegisterRole::destination, 0, gpr(index), Access::write});
        return *this;
    }
    Instr& predicate_destination(std::uint16_t index) {
        inst.registers.push_back({RegisterRole::predicate_destination, 0, pred(index), Access::write});
        return *this;
    }
    Instr& second_predicate_destination(std::uint16_t index) {
        inst.registers.push_back({RegisterRole::second_predicate_destination, 0, pred(index), Access::write});
        return *this;
    }
};

class NullFacts final : public FactSource {
public:
    Knowledge<FactValue> lookup(FactKey) const override { return Unknown{"No facts in emitter tests"}; }
};

// Records every emitted operation, register write, and diagnostic. Reads
// resolve RZ/PT to literals exactly as the contract requires and materialize
// other registers as memoized typed values.
class RecordingContext final : public EmitterContext {
public:
    DecodedProgram program{};
    DecodedInstruction current{};
    std::vector<Operation> operations;
    std::vector<Value> values;
    std::vector<std::pair<Register, Operand>> writes;
    std::map<std::pair<std::uint8_t, std::uint16_t>, ValueId> reads;
    mutable int error_count = 0;
    Limits limit_set{};

    const DecodedProgram& decoded_program() const override { return program; }
    const DecodedInstruction& instruction() const override { return current; }
    FunctionId function() const override { return FunctionId{0}; }
    BlockId block() const override { return BlockId{0}; }
    FactReader& facts() override { return reader; }
    const Limits& limits() const override { return limit_set; }

    [[nodiscard]] Result<Operand> read(Register reg) override {
        if (reg.file == RegisterFile::gpr && reg.index == 255) return Operand{word(0)};
        if (reg.file == RegisterFile::predicate && reg.index == 7) {
            return Operand{LiteralBits{1, ValueType::predicate}};
        }
        const auto key = std::make_pair(static_cast<std::uint8_t>(reg.file), reg.index);
        const auto found = reads.find(key);
        if (found != reads.end()) return Operand{found->second};
        const auto id = fresh(reg.file == RegisterFile::gpr ? ValueType::word32 : ValueType::predicate);
        reads.emplace(key, id);
        return Operand{id};
    }
    [[nodiscard]] Status write(Register reg, Operand value) override {
        // RZ/PT writes are discarded before IR validation.
        if (reg.file == RegisterFile::gpr && reg.index == 255) return std::monostate{};
        if (reg.file == RegisterFile::predicate && reg.index == 7) return std::monostate{};
        writes.emplace_back(reg, std::move(value));
        Operation op;
        op.source = location();
        op.opcode = Opcode::copy;
        op.numeric = NumericMode{};
        op.inputs = {std::move(value)};
        op.outputs = {fresh(reg.file == RegisterFile::gpr ? ValueType::word32 : ValueType::predicate)};
        operations.push_back(std::move(op));
        return std::monostate{};
    }
    [[nodiscard]] Result<ValueId> temporary(ValueType type) override { return fresh(type); }
    [[nodiscard]] Status assign(ValueId, Operand) override { return std::monostate{}; }
    [[nodiscard]] Result<OperationId> append(Operation operation) override {
        operations.push_back(std::move(operation));
        return OperationId{static_cast<std::uint32_t>(operations.size() - 1)};
    }
    [[nodiscard]] Result<std::vector<ValueId>> emit(Opcode opcode, NumericMode mode,
                                                    std::span<const Operand> inputs,
                                                    std::span<const ValueType> outputs,
                                                    OperationInfo info, Effects effects) override {
        std::vector<ValueId> ids;
        for (const auto type : outputs) ids.push_back(fresh(type));
        Operation operation_record;
        operation_record.source = location();
        operation_record.opcode = opcode;
        operation_record.effects = effects;
        operation_record.numeric = mode;
        operation_record.inputs.assign(inputs.begin(), inputs.end());
        operation_record.outputs.assign(ids.begin(), ids.end());
        operation_record.info = std::move(info);
        operations.push_back(std::move(operation_record));
        return ids;
    }
    [[nodiscard]] Result<Operand> unary(Opcode opcode, Operand input, NumericMode mode, ValueType type) override {
        auto ids = emit(opcode, mode, {&input, 1}, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&ids)) return *error;
        return Operand{std::get<std::vector<ValueId>>(ids).front()};
    }
    [[nodiscard]] Result<Operand> binary(Opcode opcode, Operand left, Operand right, NumericMode mode,
                                         ValueType type) override {
        Operand inputs[] = {std::move(left), std::move(right)};
        auto ids = emit(opcode, mode, inputs, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&ids)) return *error;
        return Operand{std::get<std::vector<ValueId>>(ids).front()};
    }
    [[nodiscard]] Result<Operand> ternary(Opcode opcode, Operand first, Operand second, Operand third,
                                          NumericMode mode, ValueType type) override {
        Operand inputs[] = {std::move(first), std::move(second), std::move(third)};
        auto ids = emit(opcode, mode, inputs, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&ids)) return *error;
        return Operand{std::get<std::vector<ValueId>>(ids).front()};
    }
    [[nodiscard]] Result<Operand> convert(Operand input, ConversionInfo info, NumericMode mode,
                                          ValueType type) override {
        auto ids = emit(Opcode::convert, mode, {&input, 1}, {&type, 1}, info, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&ids)) return *error;
        return Operand{std::get<std::vector<ValueId>>(ids).front()};
    }
    [[nodiscard]] Result<Operand> compare(Operand left, Operand right, ComparisonInfo info,
                                          NumericMode mode) override {
        Operand inputs[] = {std::move(left), std::move(right)};
        const ValueType type = ValueType::predicate;
        auto ids = emit(Opcode::compare, mode, inputs, {&type, 1}, info, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&ids)) return *error;
        return Operand{std::get<std::vector<ValueId>>(ids).front()};
    }
    [[nodiscard]] Result<Operand> select(Operand condition, Operand when_true, Operand when_false,
                                         ValueType type) override {
        Operand inputs[] = {std::move(condition), std::move(when_true), std::move(when_false)};
        const NumericMode mode{};
        auto ids = emit(Opcode::select, mode, inputs, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&ids)) return *error;
        return Operand{std::get<std::vector<ValueId>>(ids).front()};
    }
    [[nodiscard]] Result<std::vector<ValueId>> load(MemoryInfo, std::span<const Operand>, NumericMode,
                                                    std::span<const ValueType>) override {
        return error(DiagnosticCode::unimplemented, "Memory loads are not exercised by these tests");
    }
    [[nodiscard]] Status store(MemoryInfo, std::span<const Operand>, NumericMode) override {
        return error(DiagnosticCode::unimplemented, "Memory stores are not exercised by these tests");
    }
    [[nodiscard]] Result<std::vector<ValueId>> texture(Opcode, TextureInfo, std::span<const Operand>,
                                                       NumericMode, std::span<const ValueType>) override {
        return error(DiagnosticCode::unimplemented, "Texture operations are not exercised by these tests");
    }
    [[nodiscard]] Result<std::vector<ValueId>> load_io(IoInfo info, std::span<const Operand> inputs,
                                                       NumericMode numeric,
                                                       std::span<const ValueType> outputs) override {
        Operation op;
        op.source = location();
        op.opcode = Opcode::load_io;
        op.numeric = numeric;
        // The concrete context attaches required effects; mirror it here.
        op.effects = signature(Opcode::load_io)->required_effects;
        op.inputs.assign(inputs.begin(), inputs.end());
        op.info = std::move(info);
        std::vector<ValueId> results;
        for (auto type : outputs) results.push_back(fresh(type));
        op.outputs.assign(results.begin(), results.end());
        operations.push_back(std::move(op));
        return results;
    }
    [[nodiscard]] Status store_io(IoInfo, std::span<const Operand>, NumericMode) override {
        return error(DiagnosticCode::unimplemented, "IO stores are not exercised by these tests");
    }
    [[nodiscard]] Status discard() override {
        return error(DiagnosticCode::unimplemented, "Discard is not exercised by these tests");
    }
    [[nodiscard]] Result<BlockId> label(Pc) override {
        return error(DiagnosticCode::unimplemented, "Flow labels are not exercised by these tests");
    }
    [[nodiscard]] Result<BlockId> create_block() override {
        return error(DiagnosticCode::unimplemented, "Block creation is not exercised by these tests");
    }
    [[nodiscard]] Status enter_block(BlockId) override {
        return error(DiagnosticCode::unimplemented, "Block entry is not exercised by these tests");
    }
    [[nodiscard]] Status terminate(Terminator) override {
        return error(DiagnosticCode::unimplemented, "Terminators are not exercised by these tests");
    }
    [[nodiscard]] Result<ValueId> reconvergence_selector(BlockId, Pc) override {
        return error(DiagnosticCode::unimplemented, "Selectors are not exercised by these tests");
    }
    [[nodiscard]] Diagnostic error(DiagnosticCode code, std::string message,
                                   std::string modifier = {}) const override {
        ++error_count;
        Diagnostic diagnostic;
        diagnostic.phase = Phase::emit_ir;
        diagnostic.code = code;
        diagnostic.pc = current.pc;
        diagnostic.instruction = current.name;
        diagnostic.modifier = std::move(modifier);
        diagnostic.message = std::move(message);
        return diagnostic;
    }

    void reset(DecodedInstruction instruction) {
        current = std::move(instruction);
        operations.clear();
        values.clear();
        writes.clear();
        reads.clear();
        error_count = 0;
    }

private:
    NullFacts source;
    FactReader reader{source, Limits{}};
    [[nodiscard]] ValueId fresh(ValueType type) {
        values.push_back(Value{type, ValueKind::temporary});
        return ValueId{static_cast<std::uint32_t>(values.size() - 1)};
    }
    [[nodiscard]] SourceLocation location() const { return SourceLocation{current.pc, current.name}; }
};

[[nodiscard]] Status run(RecordingContext& context, Instr builder) {
    context.reset(std::move(builder.inst));
    return dispatch_instruction(context);
}
[[nodiscard]] Status run(RecordingContext& context, DecodedInstruction instruction) {
    context.reset(std::move(instruction));
    return dispatch_instruction(context);
}

std::size_t count_ops(const RecordingContext& context, Opcode opcode) {
    std::size_t total = 0;
    for (const auto& operation : context.operations) total += operation.opcode == opcode ? 1 : 0;
    return total;
}
const Operation* first_op(const RecordingContext& context, Opcode opcode) {
    for (const auto& operation : context.operations) {
        if (operation.opcode == opcode) return &operation;
    }
    return nullptr;
}
bool wrote_to(const RecordingContext& context, Register reg) {
    for (const auto& [target, value] : context.writes) {
        if (target == reg) return true;
        (void)value;
    }
    return false;
}
const Operand* write_operand(const RecordingContext& context, Register reg) {
    for (auto iterator = context.writes.rbegin(); iterator != context.writes.rend(); ++iterator) {
        if (iterator->first == reg) return &iterator->second;
    }
    return nullptr;
}
bool failed(const Status& status) { return std::holds_alternative<Diagnostic>(status); }
const Operation* compare_with_domain(const RecordingContext& context, Domain domain) {
    for (const auto& operation : context.operations) {
        if (operation.opcode == Opcode::compare && operation.numeric.domain == domain) return &operation;
    }
    return nullptr;
}

// Assembles the recorded operations into a single-block register-form program
// and runs the frozen contract validator over it.
bool valid_ir(const RecordingContext& context) {
    Program program;
    program.stage = Stage::vertex;
    program.entry = FunctionId{0};
    program.limits = Limits{};
    std::vector<ValueId> parameters;
    for (const auto& entry : context.reads) parameters.push_back(entry.second);
    program.functions.push_back({Pc{0}, BlockId{0}, {BlockId{0}}, parameters, {}});
    program.values = context.values;
    program.operations = context.operations;
    BasicBlock block;
    block.function = FunctionId{0};
    for (std::uint32_t index = 0; index < context.operations.size(); ++index) {
        block.operations.push_back(OperationId{index});
    }
    block.terminator = Return{};
    program.blocks.push_back(std::move(block));
    auto result = validate(program, IrForm::registers, Phase::emit_ir);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&result)) {
        std::cerr << "  validator: " << diagnostic->message << '\n';
        return false;
    }
    return true;
}

void expect_emitted(RecordingContext& context, Status status, std::string_view message) {
    if (const auto* d = std::get_if<Diagnostic>(&status)) std::cerr << "  status: " << d->message << '\n';
    check(!failed(status) && context.error_count == 0, message);
    if (!failed(status) && context.error_count == 0) check(valid_ir(context), message);
}
void expect_rejected(RecordingContext& context, Status status, std::string_view message) {
    check(failed(status) || context.error_count > 0, message);
}

void test_iadd() {
    RecordingContext context;
    // Plain register add lands in the destination with unsigned word mode.
    expect_emitted(context, run(context, Instr(InstructionName::iadd).word_source(SourceRole::a, 1)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::extended, 0).field(DecodedField::write_cc, 0)
                                  .field(DecodedField::function, 0)),
                   "IADD emits one addition");
    check(count_ops(context, Opcode::add) == 1, "IADD emits exactly one add");
    const auto* add = first_op(context, Opcode::add);
    check(add && add->numeric.domain == Domain::unsigned_integer &&
          add->numeric.scalar == ScalarType::u32, "IADD uses unsigned word32 arithmetic");
    check(std::holds_alternative<ValueId>(add->inputs[0]) && std::holds_alternative<ValueId>(add->inputs[1]),
          "IADD consumes both register sources");
    check(wrote_to(context, gpr(0)), "IADD writes the destination");

    // AvgMode negA negates the left operand first.
    expect_emitted(context, run(context, Instr(InstructionName::iadd).word_source(SourceRole::a, 1)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::extended, 0).field(DecodedField::write_cc, 0)
                                  .field(DecodedField::function, 2)),
                   "IADD negA emits");
    check(count_ops(context, Opcode::negate) == 1 && count_ops(context, Opcode::add) == 1,
          "IADD negA negates exactly one source");
    check(first_op(context, Opcode::negate)->numeric.scalar == ScalarType::s32,
          "Integer negation records a signed interpretation");

    // Extended adds the carry flag word.
    expect_emitted(context, run(context, Instr(InstructionName::iadd).word_source(SourceRole::a, 1)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::extended, 1).field(DecodedField::write_cc, 0)
                                  .field(DecodedField::function, 0)),
                   "IADD.X emits");
    check(count_ops(context, Opcode::convert) == 1, "IADD.X converts the carry flag to a word");
    check(count_ops(context, Opcode::add) == 2, "IADD.X adds the carry into the sum");

    // WriteCC computes C/V/Z/N against the negated operands.
    expect_emitted(context, run(context, Instr(InstructionName::iadd).word_source(SourceRole::a, 1)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::extended, 0).field(DecodedField::write_cc, 1)
                                  .field(DecodedField::function, 0)),
                   "IADD CC emits");
    check(count_ops(context, Opcode::compare) == 4, "IADD CC writes four flag comparisons");
    for (const auto code : {ConditionCode::zero, ConditionCode::sign, ConditionCode::carry, ConditionCode::overflow}) {
        check(wrote_to(context, Register{RegisterFile::condition_code, static_cast<std::uint16_t>(code)}),
              "IADD CC writes every condition-code flag");
    }
    check(valid_ir(context), "IADD CC program validates");

    // RZ reads resolve to a zero literal.
    expect_emitted(context, run(context, Instr(InstructionName::iadd).word_source(SourceRole::a, 255)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::extended, 0).field(DecodedField::write_cc, 0)
                                  .field(DecodedField::function, 0)),
                   "IADD from RZ emits");
    add = first_op(context, Opcode::add);
    check(add && is_literal(add->inputs[0], 0), "RZ resolves to the zero literal");

    // Saturation and the plus-one AvgMode are unsupported modifiers.
    expect_rejected(context, run(context, Instr(InstructionName::iadd).word_source(SourceRole::a, 1)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::extended, 0).field(DecodedField::write_cc, 0)
                                  .field(DecodedField::function, 1 << 2)),
                    "IADD saturation is rejected");
    expect_rejected(context, run(context, Instr(InstructionName::iadd).word_source(SourceRole::a, 1)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::extended, 0).field(DecodedField::write_cc, 0)
                                  .field(DecodedField::function, 3)),
                    "IADD plus-one mode is rejected");
}

void test_iadd3() {
    RecordingContext context;
    // Half selects with a left post-shift: A=H1, B=H0, C=B32, Lrs=left.
    expect_emitted(context, run(context, Instr(InstructionName::iadd3)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::half_mode, 2 | (1 << 2) | (2 << 6))
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "IADD3 emits");
    check(count_ops(context, Opcode::shift_right) == 1, "IADD3 H1 shifts the high half down");
    check(count_ops(context, Opcode::bit_and) == 1, "IADD3 H0 masks the low half");
    check(count_ops(context, Opcode::shift_left) == 1, "IADD3 Lrs=left shifts the partial sum");
    check(count_ops(context, Opcode::add) == 2, "IADD3 adds twice");

    // Per-source negation uses arithmetic negation on the extended halves.
    expect_emitted(context, run(context, Instr(InstructionName::iadd3)
                                  .word_source(SourceRole::a, 1, true).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3, true).destination(0)
                                  .field(DecodedField::half_mode, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "IADD3 negation emits");
    check(count_ops(context, Opcode::negate) == 2, "IADD3 negates both flagged sources");

    // CC and X are unimplemented; setting either is rejected.
    expect_rejected(context, run(context, Instr(InstructionName::iadd3)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::half_mode, 0)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 0)),
                    "IADD3 condition codes are rejected");
}

void test_imad_imul() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::imad)
                                  .word_source(SourceRole::a, 1).literal_source(SourceRole::b, 5)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::signedness, 0).field(DecodedField::half_mode, 0)
                                  .field(DecodedField::function, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "IMAD emits");
    check(count_ops(context, Opcode::multiply) == 1 && count_ops(context, Opcode::add) == 1,
          "IMAD multiplies then accumulates");

    // IMUL drops the literal zero addend entirely.
    expect_emitted(context, run(context, Instr(InstructionName::imul)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(0)
                                  .field(DecodedField::signedness, 0).field(DecodedField::half_mode, 0)),
                   "IMUL emits");
    check(count_ops(context, Opcode::multiply) == 1 && count_ops(context, Opcode::add) == 0,
          "IMUL omits the zero addend");

    // High signed×signed uses the signed high multiplier.
    expect_emitted(context, run(context, Instr(InstructionName::imad)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .literal_source(SourceRole::c, 0).destination(0)
                                  .field(DecodedField::signedness, 3).field(DecodedField::half_mode, 1)
                                  .field(DecodedField::function, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "IMAD.HI signed emits");
    const auto* high = first_op(context, Opcode::multiply_high);
    check(high && high->numeric.domain == Domain::signed_integer,
          "IMAD.HI signed uses the signed high multiplier");

    // One signed operand corrects the unsigned high product.
    expect_emitted(context, run(context, Instr(InstructionName::imad)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .literal_source(SourceRole::c, 0).destination(0)
                                  .field(DecodedField::signedness, 1).field(DecodedField::half_mode, 1)
                                  .field(DecodedField::function, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "IMAD.HI mixed emits");
    check(first_op(context, Opcode::multiply_high) &&
          first_op(context, Opcode::multiply_high)->numeric.domain == Domain::unsigned_integer,
          "IMAD.HI mixed starts from the unsigned high product");
    const auto* fill = first_op(context, Opcode::shift_right);
    check(fill && fill->numeric.domain == Domain::signed_integer,
          "IMAD.HI mixed fills sign bits arithmetically");
    check(count_ops(context, Opcode::multiply) == 1, "IMAD.HI mixed multiplies the sign contribution");

    // IMAD condition codes remain unimplemented upstream and are rejected.
    expect_rejected(context, run(context, Instr(InstructionName::imad)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .literal_source(SourceRole::c, 0).destination(0)
                                  .field(DecodedField::signedness, 0).field(DecodedField::half_mode, 0)
                                  .field(DecodedField::function, 0)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 0)),
                    "IMAD condition codes are rejected");
}

void test_iscadd_lea() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::iscadd)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(0)
                                  .field(DecodedField::shift_mode, 4)
                                  .field(DecodedField::function, 0).field(DecodedField::write_cc, 0)),
                   "ISCADD emits");
    const auto* scaled = first_op(context, Opcode::shift_left);
    check(scaled && is_literal(scaled->inputs[1], 4), "ISCADD scales by the immediate shift");

    // ISCADD32I has no negation modifier even with the field unset.
    expect_emitted(context, run(context, Instr(InstructionName::iscadd32i)
                                  .word_source(SourceRole::a, 1).literal_source(SourceRole::b, 7)
                                  .destination(0)
                                  .field(DecodedField::shift_mode, 0).field(DecodedField::write_cc, 0)),
                   "ISCADD32I emits without AvgMode");

    expect_emitted(context, run(context, Instr(InstructionName::lea)
                                  .word_source(SourceRole::a, 1, true).word_source(SourceRole::b, 2)
                                  .destination(0)
                                  .field(DecodedField::shift_mode, 2)),
                   "LEA emits");
    check(count_ops(context, Opcode::shift_left) == 1 && count_ops(context, Opcode::add) == 1 &&
          count_ops(context, Opcode::negate) == 1, "LEA shifts, negates, and adds");

    // LEA.HI negates the 64-bit pair through a borrow subtraction.
    expect_emitted(context, run(context, Instr(InstructionName::lea_hi)
                                  .word_source(SourceRole::a, 1, true).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::shift_mode, 4)),
                   "LEA.HI emits");
    check(count_ops(context, Opcode::bit_not) == 2, "LEA.HI negation inverts both words");
    check(count_ops(context, Opcode::convert) == 1 && count_ops(context, Opcode::subtract) == 1,
          "LEA.HI subtracts the borrow from the high word");
}

void test_xmad() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::xmad)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::function, 0).field(DecodedField::half_mode, 0)
                                  .field(DecodedField::signedness, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "XMAD Cfull emits");
    check(count_ops(context, Opcode::bit_and) == 2, "XMAD zero-extends both 16-bit halves");
    check(count_ops(context, Opcode::multiply) == 1 && count_ops(context, Opcode::add) == 1,
          "XMAD multiplies and accumulates");

    // Csfu subtracts the sign adjustments; Mrg merges into the high half.
    expect_emitted(context, run(context, Instr(InstructionName::xmad)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::function, 3).field(DecodedField::half_mode, 8)
                                  .field(DecodedField::signedness, 3)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 0)),
                   "XMAD Csfu Mrg emits");
    check(count_ops(context, Opcode::subtract) == 1, "XMAD Csfu subtracts the sign adjustment");
    check(count_ops(context, Opcode::shift_left) == 3, "XMAD Csfu shifts both fills and the merge");
    check(count_ops(context, Opcode::compare) == 4, "XMAD write-CC computes the four flags");

    // Extended form adds the carry flag word.
    expect_emitted(context, run(context, Instr(InstructionName::xmad)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::function, 0).field(DecodedField::half_mode, 0)
                                  .field(DecodedField::signedness, 0)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 1)),
                   "XMAD.X emits");
    check(count_ops(context, Opcode::convert) == 1, "XMAD.X consumes the carry flag");

    expect_rejected(context, run(context, Instr(InstructionName::xmad)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::function, 5).field(DecodedField::half_mode, 0)
                                  .field(DecodedField::signedness, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                    "XMAD rejects an out-of-range mode");
}

void test_icmp_iset() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::icmp)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::condition, 1).field(DecodedField::signedness, 0)),
                   "ICMP emits");
    const auto* comparison = first_op(context, Opcode::compare);
    check(comparison && comparison->numeric.domain == Domain::unsigned_integer,
          "ICMP compares unsigned by default");
    check(comparison && std::get<ComparisonInfo>(comparison->info).relation == Compare::less,
          "ICMP preserves the guest relation");
    check(count_ops(context, Opcode::select) == 1, "ICMP selects between the two words");

    expect_emitted(context, run(context, Instr(InstructionName::icmp)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::condition, 1).field(DecodedField::signedness, 1)),
                   "ICMP signed emits");
    check(first_op(context, Opcode::compare) &&
          first_op(context, Opcode::compare)->numeric.domain == Domain::signed_integer,
          "ICMP signedness reaches the comparison");

    // ISET with BoolOp AND and CC writes.
    expect_emitted(context, run(context, Instr(InstructionName::iset)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::condition, 2).field(DecodedField::boolean_operation, 0)
                                  .field(DecodedField::signedness, 0).field(DecodedField::destination_type, 0)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 0)),
                   "ISET emits");
    check(count_ops(context, Opcode::logical_and) == 1, "ISET combines the comparison with the predicate");
    check(count_ops(context, Opcode::compare) == 3, "ISET CC adds the zero and sign tests");

    // BVal writes the floating boolean encoding and flags with float compares.
    expect_emitted(context, run(context, Instr(InstructionName::iset)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::condition, 2).field(DecodedField::boolean_operation, 0)
                                  .field(DecodedField::signedness, 0).field(DecodedField::destination_type, 1)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 0)),
                   "ISET BVal emits");
    const auto* selection = first_op(context, Opcode::select);
    check(selection && is_literal(selection->inputs[1], 0x3f800000) && is_literal(selection->inputs[2], 0),
          "ISET BVal selects between 1.0f and 0.0f encodings");
    const auto* float_compare = compare_with_domain(context, Domain::floating);
    check(float_compare && float_compare->numeric.rounding != Rounding::not_applicable,
          "ISET BVal flag compares carry an explicit rounding mode");
    check(valid_ir(context), "ISET BVal program validates");

    // The extended ISET comparison reads condition codes instead.
    expect_emitted(context, run(context, Instr(InstructionName::iset)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::condition, 1).field(DecodedField::boolean_operation, 3)
                                  .field(DecodedField::signedness, 0).field(DecodedField::destination_type, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 1)),
                   "ISET.X emits");
    check(count_ops(context, Opcode::logical_or) == 1 && count_ops(context, Opcode::logical_and) == 1,
          "ISET.X combines the high comparison with the long-word flags");

    // ISETP writes both predicates, the second inverted.
    expect_emitted(context, run(context, Instr(InstructionName::isetp)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .predicate_source(3, true)
                                  .predicate_destination(0).second_predicate_destination(1)
                                  .field(DecodedField::condition, 5).field(DecodedField::boolean_operation, 1)
                                  .field(DecodedField::signedness, 0).field(DecodedField::extended, 0)),
                   "ISETP emits");
    check(wrote_to(context, pred(0)) && wrote_to(context, pred(1)), "ISETP writes both destinations");
    check(count_ops(context, Opcode::logical_not) == 2, "ISETP inverts for the second predicate");
    check(count_ops(context, Opcode::logical_or) == 2, "ISETP applies BoolOp or to both results");
}

void test_imnmx() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::imnmx)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::signedness, 0).field(DecodedField::write_cc, 1)),
                   "IMNMX emits");
    const auto* smallest = first_op(context, Opcode::minimum);
    check(smallest && smallest->numeric.domain == Domain::unsigned_integer,
          "IMNMX unsigned uses unsigned minimum");
    check(count_ops(context, Opcode::maximum) == 1 && count_ops(context, Opcode::select) == 1,
          "IMNMX selects between minimum and maximum");
    check(count_ops(context, Opcode::compare) == 2, "IMNMX writes the zero and sign flags");

    expect_emitted(context, run(context, Instr(InstructionName::imnmx)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::signedness, 1).field(DecodedField::write_cc, 0)),
                   "IMNMX signed emits");
    check(first_op(context, Opcode::minimum)->numeric.domain == Domain::signed_integer,
          "IMNMX signed uses signed minimum");
}

void test_lop() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::lop)
                                  .word_source(SourceRole::a, 1, true).word_source(SourceRole::b, 2)
                                  .predicate_destination(4).destination(0)
                                  .field(DecodedField::boolean_operation, 0).field(DecodedField::condition, 3)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 0)),
                   "LOP emits");
    check(count_ops(context, Opcode::bit_not) == 1, "LOP inverts only the flagged source");
    check(count_ops(context, Opcode::bit_and) == 1, "LOP performs the word AND");
    const auto* zero_test = first_op(context, Opcode::compare);
    check(zero_test && std::get<ComparisonInfo>(zero_test->info).relation == Compare::not_equal,
          "LOP predicate write tests nonzero (Nz)");
    check(wrote_to(context, pred(4)), "LOP writes the destination predicate");
    check(count_ops(context, Opcode::compare) == 3, "LOP CC adds the zero and sign tests");

    // Pass-B keeps the right operand.
    expect_emitted(context, run(context, Instr(InstructionName::lop)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(0)
                                  .field(DecodedField::boolean_operation, 3).field(DecodedField::condition, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "LOP pass-B emits");
    check(count_ops(context, Opcode::bit_and) == 0 && count_ops(context, Opcode::bit_or) == 0 &&
          count_ops(context, Opcode::bit_xor) == 0, "LOP pass-B emits no combining operation");

    // LOP32I fixes the predicate destination to PT: no predicate write.
    expect_emitted(context, run(context, Instr(InstructionName::lop32i)
                                  .word_source(SourceRole::a, 1).literal_source(SourceRole::b, 0xff)
                                  .destination(0)
                                  .field(DecodedField::boolean_operation, 2)
                                  .field(DecodedField::write_cc, 1).field(DecodedField::extended, 1)),
                   "LOP32I emits");
    check(count_ops(context, Opcode::bit_xor) == 1, "LOP32I performs the word XOR");
    check(count_ops(context, Opcode::logical_and) == 1,
          "LOP32I extended zero flag ANDs the incoming ZF");
    check(!wrote_to(context, pred(0)), "LOP32I leaves the predicate file untouched");
}

// Evaluates recorded pure word operations so LOP3 truth tables can be checked
// against the guest table directly. Word sources enter through the read cache.
bool evaluate(const Operand& operand,
              std::map<ValueId, std::uint32_t>& environment, std::uint32_t& out) {
    if (const auto* literal = std::get_if<LiteralBits>(&operand)) {
        out = static_cast<std::uint32_t>(literal->bits);
        return literal->type == ValueType::word32 || literal->type == ValueType::predicate;
    }
    if (const auto* id = std::get_if<ValueId>(&operand)) {
        const auto found = environment.find(*id);
        if (found == environment.end()) return false;
        out = found->second;
        return true;
    }
    return false;
}
bool evaluate_operations(const RecordingContext& context, const Operand& result,
                         std::map<ValueId, std::uint32_t> environment, std::uint32_t& out) {
    for (const auto& operation : context.operations) {
        std::uint32_t lhs = 0, rhs = 0, third = 0;
        auto output = [&](std::uint32_t value) {
            environment[std::get<ValueId>(operation.outputs[0])] = value;
        };
        switch (operation.opcode) {
        case Opcode::add:
        case Opcode::subtract:
        case Opcode::multiply:
        case Opcode::bit_and:
        case Opcode::bit_or:
        case Opcode::bit_xor:
        case Opcode::logical_and:
        case Opcode::logical_or:
        case Opcode::logical_xor:
            if (!evaluate(operation.inputs[0], environment, lhs) ||
                !evaluate(operation.inputs[1], environment, rhs)) return false;
            switch (operation.opcode) {
            case Opcode::add: output(lhs + rhs); break;
            case Opcode::subtract: output(lhs - rhs); break;
            case Opcode::multiply: output(lhs * rhs); break;
            case Opcode::bit_and: output(lhs & rhs); break;
            case Opcode::bit_or: output(lhs | rhs); break;
            case Opcode::bit_xor: output(lhs ^ rhs); break;
            case Opcode::logical_and: output(lhs != 0 && rhs != 0); break;
            case Opcode::logical_or: output(lhs != 0 || rhs != 0); break;
            default: output((lhs != 0) != (rhs != 0)); break;
            }
            break;
        case Opcode::bit_not:
        case Opcode::logical_not:
        case Opcode::negate:
        case Opcode::predicate_to_mask:
        case Opcode::convert:
            if (!evaluate(operation.inputs[0], environment, lhs)) return false;
            if (operation.opcode == Opcode::bit_not) output(~lhs);
            else if (operation.opcode == Opcode::logical_not) output(lhs == 0);
            else if (operation.opcode == Opcode::negate) output(0u - lhs);
            else if (operation.opcode == Opcode::predicate_to_mask) output(lhs ? 0xffffffffu : 0u);
            else output(lhs != 0);
            break;
        case Opcode::multiply_high:
            if (!evaluate(operation.inputs[0], environment, lhs) ||
                !evaluate(operation.inputs[1], environment, rhs)) return false;
            if (operation.numeric.domain == Domain::signed_integer) {
                const auto product = static_cast<std::int64_t>(static_cast<std::int32_t>(lhs)) *
                                     static_cast<std::int64_t>(static_cast<std::int32_t>(rhs));
                output(static_cast<std::uint32_t>(product >> 32));
            } else {
                output(static_cast<std::uint32_t>((static_cast<std::uint64_t>(lhs) * rhs) >> 32));
            }
            break;
        case Opcode::shift_left:
        case Opcode::shift_right:
            if (!evaluate(operation.inputs[0], environment, lhs) ||
                !evaluate(operation.inputs[1], environment, rhs)) return false;
            if (operation.opcode == Opcode::shift_left) output(lhs << (rhs & 31));
            else if (operation.numeric.domain == Domain::signed_integer) {
                output(static_cast<std::uint32_t>(static_cast<std::int32_t>(lhs) >> (rhs & 31)));
            } else output(lhs >> (rhs & 31));
            break;
        case Opcode::funnel_shift_left:
        case Opcode::funnel_shift_right:
            if (!evaluate(operation.inputs[0], environment, lhs) ||
                !evaluate(operation.inputs[1], environment, rhs) ||
                !evaluate(operation.inputs[2], environment, third)) return false;
            if (operation.opcode == Opcode::funnel_shift_left) {
                const auto joined = (static_cast<std::uint64_t>(rhs) << 32) | lhs;
                output(static_cast<std::uint32_t>((joined << (third & 31)) >> 32));
            } else {
                const auto joined = (static_cast<std::uint64_t>(rhs) << 32) | lhs;
                output(static_cast<std::uint32_t>(joined >> (third & 31)));
            }
            break;
        case Opcode::bit_extract:
            if (!evaluate(operation.inputs[0], environment, lhs) ||
                !evaluate(operation.inputs[1], environment, rhs) ||
                !evaluate(operation.inputs[2], environment, third) || third == 0 || third > 32) return false;
            if (third == 32) output(lhs);
            else {
                auto extracted = (lhs >> rhs) & ((std::uint32_t{1} << third) - 1);
                if (operation.numeric.domain == Domain::signed_integer &&
                    (extracted & (std::uint32_t{1} << (third - 1)))) {
                    extracted |= ~((std::uint32_t{1} << third) - 1);
                }
                output(extracted);
            }
            break;
        case Opcode::compare: {
            if (!evaluate(operation.inputs[0], environment, lhs) ||
                !evaluate(operation.inputs[1], environment, rhs)) return false;
            const auto relation = std::get<ComparisonInfo>(operation.info).relation;
            const bool signed_compare = operation.numeric.domain == Domain::signed_integer;
            const auto slhs = static_cast<std::int32_t>(lhs);
            const auto srhs = static_cast<std::int32_t>(rhs);
            bool compared = false;
            switch (relation) {
            case Compare::never: break;
            case Compare::less: compared = signed_compare ? slhs < srhs : lhs < rhs; break;
            case Compare::equal: compared = lhs == rhs; break;
            case Compare::less_equal: compared = signed_compare ? slhs <= srhs : lhs <= rhs; break;
            case Compare::greater: compared = signed_compare ? slhs > srhs : lhs > rhs; break;
            case Compare::not_equal: compared = lhs != rhs; break;
            case Compare::greater_equal: compared = signed_compare ? slhs >= srhs : lhs >= rhs; break;
            case Compare::always: compared = true; break;
            }
            output(compared);
            break;
        }
        case Opcode::select:
            if (!evaluate(operation.inputs[0], environment, lhs) ||
                !evaluate(operation.inputs[1], environment, rhs) ||
                !evaluate(operation.inputs[2], environment, third)) return false;
            output(lhs ? rhs : third);
            break;
        case Opcode::copy:
            if (!evaluate(operation.inputs[0], environment, lhs)) return false;
            output(lhs);
            break;
        default:
            if (!operation.inputs.empty() || !operation.outputs.empty()) return false;
            break;
        }
    }
    return evaluate(result, environment, out);
}

void test_lop3() {
    RecordingContext context;
    // Inputs enumerate every (a, b, c) bit combination four times.
    const std::uint32_t input_a = 0xaaaaaaaa;
    const std::uint32_t input_b = 0xcccccccc;
    const std::uint32_t input_c = 0xf0f0f0f0;
    for (const std::uint32_t table : {std::uint32_t{0x00}, std::uint32_t{0xff}, std::uint32_t{0xca},
                                      std::uint32_t{0x96}, std::uint32_t{0xe8}, std::uint32_t{0x16},
                                      std::uint32_t{0x59}, std::uint32_t{0x81}, std::uint32_t{0x7e},
                                      std::uint32_t{0xac}}) {
        const auto status = run(context, Instr(InstructionName::lop3)
                                    .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                    .word_source(SourceRole::c, 3).destination(0)
                                    .field(DecodedField::permutation_mode, table)
                                    .field(DecodedField::condition, 0)
                                    .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0));
        if (failed(status)) {
            check(false, "LOP3 table emits");
            continue;
        }
        std::uint32_t expected = 0;
        for (unsigned bit = 0; bit < 32; ++bit) {
            const unsigned row = (((input_a >> bit) & 1) << 2) | (((input_b >> bit) & 1) << 1) |
                                 ((input_c >> bit) & 1);
            expected |= ((table >> row) & 1) << bit;
        }
        const auto* destination_write = write_operand(context, gpr(0));
        std::uint32_t produced = 0;
        std::map<ValueId, std::uint32_t> environment;
        environment[context.reads.at({static_cast<std::uint8_t>(RegisterFile::gpr), 1})] = input_a;
        environment[context.reads.at({static_cast<std::uint8_t>(RegisterFile::gpr), 2})] = input_b;
        environment[context.reads.at({static_cast<std::uint8_t>(RegisterFile::gpr), 3})] = input_c;
        check(destination_write &&
              evaluate_operations(context, *destination_write, environment, produced) &&
              produced == expected,
              "LOP3 truth table matches the guest function");
    }
    check(valid_ir(context), "LOP3 program validates");

    // Constant tables collapse to literals without reading the inputs.
    expect_emitted(context, run(context, Instr(InstructionName::lop3)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::permutation_mode, 0)
                                  .field(DecodedField::condition, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "LOP3 false table emits");
    check(count_ops(context, Opcode::bit_and) == 0 && count_ops(context, Opcode::bit_or) == 0,
          "LOP3 false table collapses to a literal");
    check(write_operand(context, gpr(0)) && is_literal(*write_operand(context, gpr(0)), 0),
          "LOP3 false table writes zero");
}

void test_decoded_regressions() {
    RecordingContext context;
    auto emit_word = [&](std::uint64_t word, std::string_view message) {
        auto decoded = decode_word(word, message);
        expect_emitted(context, run(context, std::move(decoded)), message);
    };
    auto inputs = [&](std::initializer_list<std::pair<Register, std::uint32_t>> values) {
        std::map<ValueId, std::uint32_t> environment;
        for (const auto& [reg, bits] : values) {
            const auto key = std::make_pair(static_cast<std::uint8_t>(reg.file), reg.index);
            const auto found = context.reads.find(key);
            if (found != context.reads.end()) environment[found->second] = bits;
        }
        return environment;
    };
    auto result_is = [&](Register destination,
                         std::map<ValueId, std::uint32_t> environment,
                         std::uint32_t expected, std::string_view message) {
        const auto* result = write_operand(context, destination);
        std::uint32_t produced = 0;
        const bool evaluated = result != nullptr &&
                               evaluate_operations(context, *result, std::move(environment), produced);
        if (!evaluated || produced != expected) {
            std::cerr << "  expected 0x" << std::hex << expected << ", produced 0x" << produced
                      << std::dec << '\n';
        }
        check(evaluated && produced == expected, message);
    };

    // LOP3 table 0xf0 selects A because the guest row index is A:B:C.
    auto encoded = opcode_pattern("0101101111100");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 8, 7);
    encoded = set_bits(encoded, 28, 8, 0xf0);
    encoded = set_bits(encoded, 39, 8, 8);
    emit_word(encoded, "Decoded LOP3 selector emits");
    result_is(gpr(5), inputs({{gpr(6), 0x12345678}, {gpr(7), 0x0f0f0f0f},
                              {gpr(8), 0xa5a5a5a5}}),
              0x12345678, "Decoded LOP3 0xf0 selects source A");

    // Integer ISET true is an all-ones word, while carry conversion remains 0/1.
    encoded = opcode_pattern("010110110101");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 8, 6);
    encoded = set_bits(encoded, 45, 2, 3);
    encoded = set_bits(encoded, 47, 1, 1);
    encoded = set_bits(encoded, 49, 3, 2);
    emit_word(encoded, "Decoded ISET mask emits");
    check(count_ops(context, Opcode::predicate_to_mask) == 1,
          "Decoded integer ISET uses predicate-to-mask");
    const auto iset_inputs = inputs({{gpr(6), 11}});
    result_is(gpr(5), iset_inputs, 0xffffffff,
              "Decoded true integer ISET produces all ones");
    result_is(cc(ConditionCode::sign), iset_inputs, 1,
              "Decoded true integer ISET sets the sign flag from its mask");

    // PSETP applies the second Boolean operator to both Q and !Q before C.
    encoded = opcode_pattern("0101000010010");
    encoded = set_bits(encoded, 0, 3, 5);
    encoded = set_bits(encoded, 3, 3, 4);
    encoded = set_bits(encoded, 12, 3, 2);
    encoded = set_bits(encoded, 29, 3, 1);
    encoded = set_bits(encoded, 39, 3, 3);
    emit_word(encoded, "Decoded PSETP emits");
    const auto predicate_inputs = inputs({{pred(1), 0}, {pred(2), 0}, {pred(3), 0}});
    result_is(pred(4), predicate_inputs, 0, "Decoded PSETP writes Q AND C");
    result_is(pred(5), predicate_inputs, 0, "Decoded PSETP writes not-Q AND C");

    // SHF uses the funnel opcode so a dynamic effective count of zero selects
    // exactly one input word instead of ORing both words.
    encoded = opcode_pattern("0101101111111");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 8, 7);
    encoded = set_bits(encoded, 39, 8, 8);
    emit_word(encoded, "Decoded SHF left emits");
    check(count_ops(context, Opcode::funnel_shift_left) == 1,
          "Decoded SHF left uses the funnel opcode");
    result_is(gpr(5), inputs({{gpr(6), 1}, {gpr(7), 0}, {gpr(8), 2}}), 2,
              "Decoded SHF left count zero selects the high word");

    encoded = opcode_pattern("0101110011111");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 8, 7);
    encoded = set_bits(encoded, 39, 8, 8);
    emit_word(encoded, "Decoded SHF right emits");
    check(count_ops(context, Opcode::funnel_shift_right) == 1,
          "Decoded SHF right uses the funnel opcode");
    result_is(gpr(5), inputs({{gpr(6), 1}, {gpr(7), 0}, {gpr(8), 2}}), 1,
              "Decoded SHF right count zero selects the low word");

    // Unsigned IMAD.HI must remain a plain unsigned high multiplication.
    encoded = opcode_pattern("010110100");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 8, 7);
    encoded = set_bits(encoded, 39, 8, 255);
    encoded = set_bits(encoded, 54, 1, 1);
    emit_word(encoded, "Decoded unsigned IMAD.HI emits");
    check(count_ops(context, Opcode::multiply) == 0 && count_ops(context, Opcode::shift_right) == 0,
          "Decoded unsigned IMAD.HI has no signed correction");
    result_is(gpr(5), inputs({{gpr(6), 1}, {gpr(7), 0x80000000}}), 0,
              "Decoded unsigned IMAD.HI returns the unsigned high word");

    // IADD3.RS preserves bit 32 of A+B as bit 16 of the shifted partial sum.
    encoded = opcode_pattern("010111001100");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 8, 7);
    encoded = set_bits(encoded, 37, 2, 1);
    encoded = set_bits(encoded, 39, 8, 8);
    emit_word(encoded, "Decoded IADD3.RS emits");
    result_is(gpr(5), inputs({{gpr(6), 0xffffffff}, {gpr(7), 1}, {gpr(8), 0}}),
              0x00010000, "Decoded IADD3.RS preserves the intermediate carry");

    // XMAD.PSL computes carry from the shifted product used by accumulation.
    encoded = opcode_pattern("0101101100");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 8, 7);
    encoded = set_bits(encoded, 36, 1, 1);
    encoded = set_bits(encoded, 39, 8, 8);
    encoded = set_bits(encoded, 47, 1, 1);
    emit_word(encoded, "Decoded XMAD.PSL.CC emits");
    const auto xmad_inputs = inputs({{gpr(6), 1}, {gpr(7), 1}, {gpr(8), 0xffff0001}});
    result_is(gpr(5), xmad_inputs, 1, "Decoded XMAD.PSL writes its accumulated result");
    result_is(cc(ConditionCode::carry), xmad_inputs, 1,
              "Decoded XMAD.PSL derives carry from the shifted product");

    // Immediate XMAD.X uses bit 38 and therefore omits C in favor of carry-in.
    encoded = opcode_pattern("0011011x00");
    encoded = set_bits(encoded, 0, 8, 5);
    encoded = set_bits(encoded, 8, 8, 6);
    encoded = set_bits(encoded, 20, 16, 3);
    encoded = set_bits(encoded, 38, 1, 1);
    encoded = set_bits(encoded, 39, 8, 8);
    emit_word(encoded, "Decoded immediate XMAD.X emits");
    result_is(gpr(5), inputs({{gpr(6), 2}, {gpr(8), 5}, {cc(ConditionCode::carry), 0}}), 6,
              "Decoded immediate XMAD.X consumes carry instead of C");
}

void test_shifts() {
    RecordingContext context;
    // SHF right funnel with clamped counts.
    expect_emitted(context, run(context, Instr(InstructionName::shf)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::shift_mode, 4 | 0).field(DecodedField::shift_wrap, 0)
                                  .field(DecodedField::write_cc, 0)),
                   "SHF right emits");
    check(count_ops(context, Opcode::funnel_shift_right) == 1,
          "SHF right combines both halves with a funnel shift");
    check(count_ops(context, Opcode::select) == 1, "SHF unwrapped clamps via select");

    // Wrapped long shift selects the spill half for counts past 31.
    expect_emitted(context, run(context, Instr(InstructionName::shf)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::shift_mode, 2 | 4).field(DecodedField::shift_wrap, 1)
                                  .field(DecodedField::write_cc, 0)),
                   "SHF long right emits");
    const auto* mask = first_op(context, Opcode::bit_and);
    check(mask && is_literal(mask->inputs[1], 63), "SHF long wrap masks the count to 63");
    check(count_ops(context, Opcode::select) == 1, "SHF long selects the spill branch");
    check(count_ops(context, Opcode::subtract) == 1, "SHF forms the b-32 spill count");

    // S64 fills the spilled upper word arithmetically.
    expect_emitted(context, run(context, Instr(InstructionName::shf)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)
                                  .field(DecodedField::shift_mode, 3 | 4).field(DecodedField::shift_wrap, 1)
                                  .field(DecodedField::write_cc, 0)),
                   "SHF s64 emits");
    bool signed_spill = false;
    for (const auto& operation : context.operations) {
        if (operation.opcode == Opcode::shift_right &&
            operation.numeric.domain == Domain::signed_integer) signed_spill = true;
    }
    check(signed_spill, "SHF s64 spills with an arithmetic shift");

    expect_emitted(context, run(context, Instr(InstructionName::shl)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(0)
                                  .field(DecodedField::shift_wrap, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                   "SHL emits");
    check(count_ops(context, Opcode::shift_left) == 1 && count_ops(context, Opcode::select) == 1,
          "SHL clamps counts of 32 or more");

    expect_emitted(context, run(context, Instr(InstructionName::shr)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(0)
                                  .field(DecodedField::shift_wrap, 0).field(DecodedField::signedness, 1)
                                  .field(DecodedField::bitfield_mode, 1)
                                  .field(DecodedField::write_cc, 0)),
                   "SHR signed reversed emits");
    check(count_ops(context, Opcode::bit_reverse) == 1, "SHR Brev reverses the source");
    check(first_op(context, Opcode::shift_right) &&
          first_op(context, Opcode::shift_right)->numeric.domain == Domain::signed_integer,
          "SHR signed shifts arithmetically");
    check(count_ops(context, Opcode::select) == 1, "SHR unwrapped clamps via select");
}

void test_bitfield() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::bfe)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(0)
                                  .field(DecodedField::signedness, 1).field(DecodedField::bitfield_mode, 0)),
                   "BFE signed emits");
    bool signed_extraction = false;
    for (const auto& operation : context.operations) {
        if (operation.opcode == Opcode::bit_extract && operation.inputs.size() == 3 &&
            operation.numeric.domain == Domain::signed_integer) signed_extraction = true;
    }
    check(signed_extraction, "BFE extracts with the guest signedness");
    const auto* position = first_op(context, Opcode::bit_and);
    check(position && is_literal(position->inputs[1], 0xff), "BFE masks the position byte");

    expect_emitted(context, run(context, Instr(InstructionName::bfi)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .word_source(SourceRole::c, 3).destination(0)),
                   "BFI emits");
    const auto* insertion = first_op(context, Opcode::bit_insert);
    check(insertion && insertion->inputs.size() == 4, "BFI inserts with base, value, position, and size");

    expect_emitted(context, run(context, Instr(InstructionName::flo)
                                  .word_source(SourceRole::b, 2, true).destination(0)
                                  .field(DecodedField::signedness, 0).field(DecodedField::bitfield_mode, 1)),
                   "FLO sh emits");
    check(count_ops(context, Opcode::bit_not) == 1 && count_ops(context, Opcode::bit_reverse) == 1,
          "FLO sh inverts then reverses");
    check(first_op(context, Opcode::find_lsb) != nullptr, "FLO sh finds the least significant bit");

    expect_emitted(context, run(context, Instr(InstructionName::flo)
                                  .word_source(SourceRole::b, 2).destination(0)
                                  .field(DecodedField::signedness, 1).field(DecodedField::bitfield_mode, 0)),
                   "FLO signed msb emits");
    const auto* msb = first_op(context, Opcode::find_msb);
    check(msb && msb->numeric.domain == Domain::signed_integer, "FLO signed uses the signed msb");

    expect_emitted(context, run(context, Instr(InstructionName::popc)
                                  .word_source(SourceRole::b, 2).destination(0)),
                   "POPC emits");
    check(first_op(context, Opcode::bit_count) != nullptr, "POPC counts bits");
}

void test_moves_and_predicates() {
    RecordingContext context;
    expect_emitted(context, run(context, Instr(InstructionName::mov)
                                  .word_source(SourceRole::a, 1).destination(0)),
                   "MOV emits");
    check(context.operations.size() == 1 && context.operations.front().opcode == Opcode::copy,
          "MOV emits a single copy");

    // Constant-buffer words flow through as deferred operands.
    expect_emitted(context, run(context, Instr(InstructionName::mov)
                                  .cbuf_source(SourceRole::a, 2, 0x40).destination(3)),
                   "MOV from cbuf emits");
    const auto* copy = first_op(context, Opcode::copy);
    check(copy && std::holds_alternative<ConstantWord>(copy->inputs[0]),
          "MOV preserves the constant-buffer word operand");

    expect_emitted(context, run(context, Instr(InstructionName::sel)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .predicate_source(3, true).destination(0)),
                   "SEL emits");
    check(count_ops(context, Opcode::select) == 1, "SEL selects on the predicate");
    check(first_op(context, Opcode::logical_not) != nullptr, "SEL inversion precedes the select");

    // R2P scatters into the predicate file; PT writes are discarded.
    expect_emitted(context, run(context, Instr(InstructionName::r2p)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .field(DecodedField::bitfield_mode, 0)),
                   "R2P emits");
    int predicate_writes = 0;
    for (const auto& [target, value] : context.writes) {
        (void)value;
        predicate_writes += target.file == RegisterFile::predicate ? 1 : 0;
    }
    check(predicate_writes == 7, "R2P writes the seven writable predicates");

    // CCPR targets the four condition-code flags instead.
    expect_emitted(context, run(context, Instr(InstructionName::r2p)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .field(DecodedField::bitfield_mode, 4 | 1)),
                   "R2P CCPR emits");
    int flag_writes = 0;
    for (const auto& [target, value] : context.writes) {
        (void)value;
        flag_writes += target.file == RegisterFile::condition_code ? 1 : 0;
    }
    check(flag_writes == 4, "R2P CCPR writes the four flags");

    expect_emitted(context, run(context, Instr(InstructionName::p2r)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(5)
                                  .field(DecodedField::bitfield_mode, 4)),
                   "P2R CCPR emits");
    check(wrote_to(context, gpr(5)), "P2R writes the destination");
    check(count_ops(context, Opcode::select) == 4, "P2R gathers the four flags");
    check(count_ops(context, Opcode::bit_or) == 5, "P2R combines gathered bits and preserves unmasked bits");

    // CSET Eq with BoolOp OR against P3.
    expect_emitted(context, run(context, Instr(InstructionName::cset)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::condition, 2).field(DecodedField::boolean_operation, 1)
                                  .field(DecodedField::destination_type, 0)),
                   "CSET emits");
    check(count_ops(context, Opcode::logical_or) == 1, "CSET applies the BoolOp");
    check(write_operand(context, gpr(0)) && std::holds_alternative<ValueId>(*write_operand(context, gpr(0))),
          "CSET writes the boolean word");

    // BVal selects the float encoding.
    expect_emitted(context, run(context, Instr(InstructionName::cset)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::condition, 15).field(DecodedField::boolean_operation, 3)
                                  .field(DecodedField::destination_type, 1)),
                   "CSET BVal emits");
    const auto* selection = first_op(context, Opcode::select);
    check(selection && is_literal(selection->inputs[1], 0x3f800000),
          "CSET BVal selects the 1.0f encoding");

    // Reserved camera conditions are rejected instead of silently true.
    expect_rejected(context, run(context, Instr(InstructionName::cset)
                                  .predicate_source(3).destination(0)
                                  .field(DecodedField::condition, 24).field(DecodedField::boolean_operation, 0)
                                  .field(DecodedField::destination_type, 0)),
                    "CSET reserved condition is rejected");

    expect_emitted(context, run(context, Instr(InstructionName::csetp)
                                  .predicate_source(3)
                                  .predicate_destination(0).second_predicate_destination(1)
                                  .field(DecodedField::condition, 5).field(DecodedField::boolean_operation, 0)),
                   "CSETP emits");
    check(wrote_to(context, pred(0)) && wrote_to(context, pred(1)), "CSETP writes both predicates");

    // PSET inverts each input per its modifier and applies both BoolOps.
    expect_emitted(context, run(context, Instr(InstructionName::pset)
                                  .register_source(SourceRole::a, pred(1))
                                  .register_source(SourceRole::b, pred(2), true)
                                  .register_source(SourceRole::c, pred(3)).destination(0)
                                  .field(DecodedField::boolean_operation, 2 | (1 << 2))
                                  .field(DecodedField::destination_type, 0)),
                   "PSET emits");
    check(count_ops(context, Opcode::logical_xor) == 1 && count_ops(context, Opcode::logical_or) == 1,
          "PSET applies BoolOpAB then BoolOpC");
    // The word sources here stand in for predicate registers in this synthetic
    // case; the reads materialize as word values.
    check(valid_ir(context), "PSET program validates");

    expect_emitted(context, run(context, Instr(InstructionName::psetp)
                                  .register_source(SourceRole::a, pred(1))
                                  .register_source(SourceRole::b, pred(2), true)
                                  .register_source(SourceRole::c, pred(3))
                                  .predicate_destination(0).second_predicate_destination(1)
                                  .field(DecodedField::boolean_operation, 0)),
                   "PSETP emits");
    check(wrote_to(context, pred(0)) && wrote_to(context, pred(1)), "PSETP writes both predicates");

    // S2R has no IO contract representation yet and fails explicitly.
    expect_rejected(context, run(context, Instr(InstructionName::s2r)
                                  .destination(0).field(DecodedField::function, 32)),
                    "S2R fails with a structured diagnostic");

    // SR_YDIRECTION (18) reads the guest Y sign as a graphics draw-state
    // interface input, selected at draw time by a function constant.
    context.program.stage = Stage::fragment;
    expect_emitted(context, run(context, Instr(InstructionName::s2r)
                                  .destination(0).field(DecodedField::function, 18)),
                   "S2R SR_YDIRECTION emits in a fragment stage");
    const auto* direction = first_op(context, Opcode::load_io);
    check(direction && std::holds_alternative<IoInfo>(direction->info) &&
              std::get<IoInfo>(direction->info).semantic == IoSemantic::y_direction &&
              std::get<IoInfo>(direction->info).type == IoType{ScalarType::u32, 1} &&
              std::get<IoInfo>(direction->info).component_mask == 1,
          "S2R SR_YDIRECTION loads the y_direction interface word");
    check(wrote_to(context, gpr(0)), "S2R SR_YDIRECTION writes its destination");
    // Compute has no rasterization Y direction and keeps failing explicitly.
    context.program.stage = Stage::compute;
    expect_rejected(context, run(context, Instr(InstructionName::s2r)
                                  .destination(0).field(DecodedField::function, 18)),
                    "S2R SR_YDIRECTION rejects in a compute stage");
    context.program.stage = Stage::vertex;
    expect_emitted(context, run(context, Instr(InstructionName::s2r)
                                  .destination(0).field(DecodedField::function, 18)),
                   "S2R SR_YDIRECTION emits in a vertex stage");
}

void test_diagnostics_and_registry() {
    RecordingContext context;
    // A missing destination register is a structured error, not a drop.
    expect_rejected(context, run(context, Instr(InstructionName::mov).word_source(SourceRole::a, 1)),
                    "Missing destination is rejected");
    // A missing decoded source role is a structured error.
    expect_rejected(context, run(context, Instr(InstructionName::mov).destination(0)),
                    "Missing source role is rejected");
    // A required decoded field that the decoder did not publish fails.
    expect_rejected(context, run(context, Instr(InstructionName::lop)
                                  .word_source(SourceRole::a, 1).word_source(SourceRole::b, 2)
                                  .destination(0).field(DecodedField::condition, 0)
                                  .field(DecodedField::write_cc, 0).field(DecodedField::extended, 0)),
                    "Missing LogicOp field is rejected");

    // The registry aliases dispatch through the same callbacks.
    expect_emitted(context, run(context, Instr(InstructionName::mov32i)
                                  .literal_source(SourceRole::a, 7).destination(2)),
                   "MOV32I shares the MOV callback");
    expect_emitted(context, run(context, Instr(InstructionName::iadd32i)
                                  .word_source(SourceRole::a, 1).literal_source(SourceRole::b, 9)
                                  .destination(0)
                                  .field(DecodedField::extended, 0).field(DecodedField::write_cc, 0)
                                  .field(DecodedField::function, 0)),
                   "IADD32I shares the IADD callback");
    check(valid_ir(context), "Aliased dispatch programs validate");
}
} // namespace

int main() {
    test_iadd();
    test_iadd3();
    test_imad_imul();
    test_iscadd_lea();
    test_xmad();
    test_icmp_iset();
    test_imnmx();
    test_lop();
    test_lop3();
    test_decoded_regressions();
    test_shifts();
    test_bitfield();
    test_moves_and_predicates();
    test_diagnostics_and_registry();
    if (failures) {
        std::cerr << failures << " emitter integer checks failed\n";
        return 1;
    }
    std::cout << "Shader pipeline integer emitters passed\n";
    return 0;
}
