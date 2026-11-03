// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Decode/emit alignment tests for the integer, move, predicate, floating,
// conversion and multifunction emitter families. Each case builds a synthetic
// guest word, decodes it through the published ALU rules, and dispatches it
// through the emitter registry into a recording EmitterContext, checking the
// emitted IR against guest instruction semantics. The synthetic words are
// independently authored.

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

// Mirrors the table-row encoding: index 0 is bit 63.
constexpr std::uint64_t opcode_pattern(std::string_view pattern) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < pattern.size(); ++index) {
        if (pattern[index] == '1') value |= std::uint64_t{1} << (63 - index);
    }
    return value;
}

constexpr std::uint64_t set(std::uint64_t word, unsigned first, unsigned width, std::uint64_t value) {
    const auto clear = ~(((std::uint64_t{1} << width) - 1) << first);
    return (word & clear) | (value << first);
}

const DecodeRule* find_rule(std::uint64_t word) {
    for (const auto& entry : alu_decode_rules()) {
        if ((word & entry.mask) == entry.value) return &entry;
    }
    return nullptr;
}

// Decodes a synthetic word through the published lane-1 rules.
bool decode_word(std::uint64_t word, std::string_view message, DecodedInstruction& decoded) {
    const auto* entry = find_rule(word);
    check(entry != nullptr, message);
    if (entry == nullptr) return false;
    auto result = entry->decode(40, RawInstruction{word});
    if (std::get_if<Diagnostic>(&result) != nullptr) {
        std::cerr << "  decode error: " << std::get<Diagnostic>(result).message << '\n';
        check(false, message);
        return false;
    }
    decoded = std::get<DecodedInstruction>(std::move(result));
    return true;
}

Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
Register gpr(std::uint16_t index) { return Register{RegisterFile::gpr, index}; }
Register pred(std::uint16_t index) { return Register{RegisterFile::predicate, index}; }

class NullFacts final : public FactSource {
public:
    Knowledge<FactValue> lookup(FactKey) const override { return Unknown{"No facts in alignment tests"}; }
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
        auto ids = emit(Opcode::select, NumericMode{}, inputs, {&type, 1}, {}, Effects::none);
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
                                                       NumericMode mode,
                                                       std::span<const ValueType> outputs) override {
        std::vector<ValueId> ids;
        for (const auto type : outputs) ids.push_back(fresh(type));
        Operation operation_record;
        operation_record.source = location();
        operation_record.opcode = Opcode::load_io;
        operation_record.numeric = mode;
        operation_record.inputs.assign(inputs.begin(), inputs.end());
        operation_record.outputs.assign(ids.begin(), ids.end());
        operation_record.info = std::move(info);
        operations.push_back(std::move(operation_record));
        return ids;
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

RecordingContext context;

// Decodes and dispatches one synthetic word; returns true when the emitter
// accepted it. The decoded instruction stays available for inspection.
bool emits(std::uint64_t word, std::string_view message) {
    DecodedInstruction decoded;
    if (!decode_word(word, message, decoded)) return false;
    context.reset(std::move(decoded));
    Status status = dispatch_instruction(context);
    const bool ok = std::holds_alternative<std::monostate>(status) && context.error_count == 0;
    if (!ok && std::holds_alternative<Diagnostic>(status)) {
        std::cerr << "  dispatch error: " << std::get<Diagnostic>(status).message << '\n';
    }
    check(ok, message);
    return ok;
}

Diagnostic rejected(std::uint64_t word, std::string_view message) {
    DecodedInstruction decoded;
    if (decode_word(word, message, decoded)) {
        context.reset(std::move(decoded));
        Status status = dispatch_instruction(context);
        check(std::holds_alternative<Diagnostic>(status) || context.error_count > 0, message);
        if (std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    }
    return {};
}

std::size_t count_ops(Opcode opcode) {
    std::size_t total = 0;
    for (const auto& operation : context.operations) total += operation.opcode == opcode ? 1 : 0;
    return total;
}
const Operation* first_op(Opcode opcode) {
    for (const auto& operation : context.operations) {
        if (operation.opcode == opcode) return &operation;
    }
    return nullptr;
}
bool wrote_to(Register reg) {
    for (const auto& [target, value] : context.writes) {
        if (target == reg) return true;
        (void)value;
    }
    return false;
}

const DecodedSource* current_source(SourceRole role) {
    for (const auto& source : context.current.sources) {
        if (source.role == role) return &source;
    }
    return nullptr;
}

bool current_field(DecodedField field, std::uint64_t bits) {
    for (const auto& value : context.current.fields) {
        if (value.field == field && value.bits == bits) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Lane 6: integer arithmetic, comparison, logical, shifts, bit-field, moves
// and predicate instructions.
// ---------------------------------------------------------------------------

void test_integer_arithmetic_words() {
    // IADD R5, R6, R7: one unsigned add into the destination.
    emits(opcode_pattern("0101110000010") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7),
          "IADD register word dispatches");
    check(count_ops(Opcode::add) == 1 && wrote_to(gpr(5)), "IADD emits one add into Rd");
    check(first_op(Opcode::add) != nullptr &&
              first_op(Opcode::add)->numeric.domain == Domain::unsigned_integer,
          "IADD adds in the unsigned domain");

    // IADD32I R5, R6, 0x1234: the 32-bit immediate rides source B.
    emits(opcode_pattern("0001110") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x1234),
          "IADD32I word dispatches");
    check(count_ops(Opcode::add) == 1, "IADD32I emits one add");

    // IADD3 R5, R6, R7, R8 with H0/H1 half selects on A and B.
    emits(opcode_pattern("010111001100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) |
              set(std::uint64_t{0}, 35, 2, 1) | set(std::uint64_t{0}, 33, 2, 2),
          "IADD3 register word dispatches");
    check(count_ops(Opcode::add) >= 2, "IADD3 sums the three operands");
    check(first_op(Opcode::bit_and) != nullptr && first_op(Opcode::shift_right) != nullptr,
          "IADD3 half selects mask and shift the low words");

    // IMAD R5, R6, R7, R8: multiply plus accumulate.
    emits(opcode_pattern("010110100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8),
          "IMAD register word dispatches");
    check(count_ops(Opcode::multiply) == 1 && count_ops(Opcode::add) == 1, "IMAD multiplies and accumulates");

    // IMAD Rc multiplies by the register in SrcC's physical field and adds
    // the constant-buffer operand.
    emits(opcode_pattern("010100100") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 39, 8, 8) |
              set(std::uint64_t{0}, 34, 5, 1) | set(std::uint64_t{0}, 20, 14, 4),
          "IMAD register-constant word dispatches");
    const auto* source_b = current_source(SourceRole::b);
    const auto* source_c = current_source(SourceRole::c);
    check(source_b != nullptr && std::holds_alternative<Register>(source_b->value) &&
              std::get<Register>(source_b->value) == gpr(8),
          "IMAD Rc preserves the register multiplier");
    check(source_c != nullptr && std::holds_alternative<ConstantWord>(source_c->value),
          "IMAD Rc preserves the constant addend");

    // IMAD32I R5, R6, 0x40 accumulates the old value of R5.
    emits(opcode_pattern("000100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x40),
          "IMAD32I word dispatches");
    check(count_ops(Opcode::multiply) == 1, "IMAD32I multiplies");
    source_c = current_source(SourceRole::c);
    check(source_c != nullptr && std::holds_alternative<Register>(source_c->value) &&
              std::get<Register>(source_c->value) == gpr(5),
          "IMAD32I reads its addend from the old destination");

    emits(0x1000000000270103, "Reviewed IMAD32I destination-addend word dispatches");
    source_c = current_source(SourceRole::c);
    check(source_c != nullptr && std::holds_alternative<Register>(source_c->value) &&
              std::get<Register>(source_c->value) == gpr(3) && wrote_to(gpr(3)),
          "Reviewed IMAD32I word reads and writes R3");

    // IMUL keeps only the product (no zero addend emitted).
    emits(opcode_pattern("0101110000111") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7),
          "IMUL register word dispatches");
    check(count_ops(Opcode::multiply) == 1 && count_ops(Opcode::add) == 0, "IMUL emits only the product");

    emits(opcode_pattern("00011111") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 7),
          "IMUL32I word dispatches");
    check(count_ops(Opcode::multiply) == 1, "IMUL32I emits only the product");

    // ISCADD R5, R6, R7, scale 3: shifted left operand then one add.
    emits(opcode_pattern("0101110000011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 5, 3),
          "ISCADD register word dispatches");
    check(count_ops(Opcode::shift_left) == 1 && count_ops(Opcode::add) == 1, "ISCADD shifts and adds");

    emits(opcode_pattern("000101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x10) | set(std::uint64_t{0}, 53, 5, 2),
          "ISCADD32I word dispatches");
    check(count_ops(Opcode::shift_left) == 1 && count_ops(Opcode::add) == 1, "ISCADD32I shifts and adds");

    // LEA R5, R6, R7, shift 4.
    emits(opcode_pattern("0101101111010") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 5, 4),
          "LEA register word dispatches");
    check(count_ops(Opcode::shift_left) == 1 && count_ops(Opcode::add) == 1, "LEA shifts A and adds B");

    // LEAHI R5, R6, R7, R8: folds the shifted-out high bits into C.
    emits(opcode_pattern("0101101111011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 28, 5, 4),
          "LEAHI register word dispatches");
    check(count_ops(Opcode::add) == 1 && count_ops(Opcode::bit_or) == 1,
          "LEAHI merges the high bits and adds both addends");

    // XMAD R5, R6, R7, R8 with Ccop Cfull and the merge flag.
    emits(opcode_pattern("0101101100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 37, 1, 1),
          "XMAD register word dispatches");
    check(count_ops(Opcode::multiply) == 1, "XMAD multiplies the extended halves");
    check(first_op(Opcode::bit_or) != nullptr, "XMAD merge inserts the product into the high half");

    emits(opcode_pattern("010100010") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 39, 8, 8) |
              set(std::uint64_t{0}, 34, 5, 1) | set(std::uint64_t{0}, 20, 14, 4),
          "XMAD register-constant word dispatches");
    source_b = current_source(SourceRole::b);
    source_c = current_source(SourceRole::c);
    check(source_b != nullptr && std::holds_alternative<Register>(source_b->value) &&
              std::get<Register>(source_b->value) == gpr(8),
          "XMAD Rc preserves the register multiplier");
    check(source_c != nullptr && std::holds_alternative<ConstantWord>(source_c->value),
          "XMAD Rc preserves the constant addend");

    emits(0x3600014000370100, "XMAD immediate extended word dispatches");
    check(current_field(DecodedField::extended, 1), "XMAD immediate reads X from bit 38");

    // IADD saturation stays an explicit rejection.
    rejected(opcode_pattern("0101110000010") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
                 set(std::uint64_t{0}, 20, 8, 7) | (std::uint64_t{1} << 50),
             "IADD saturation is an unsupported modifier");
}

void test_integer_comparison_words() {
    // ICMP selects between A and B on the comparison of C with zero.
    emits(opcode_pattern("010110110100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 49, 3, 1),
          "ICMP register word dispatches");
    check(count_ops(Opcode::compare) == 1 && count_ops(Opcode::select) == 1,
          "ICMP compares C against zero and selects");

    // ISET writes the integer Boolean word even when predicate P2 sets bit 40.
    emits(opcode_pattern("010110110101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 3, 2) | set(std::uint64_t{0}, 49, 3, 2) |
              set(std::uint64_t{0}, 45, 2, 1),
          "ISET register word dispatches");
    check(wrote_to(gpr(5)) && count_ops(Opcode::predicate_to_mask) == 1,
          "ISET expands the predicate result to an all-ones mask");
    check(current_field(DecodedField::destination_type, 0), "ISET reads BVal from bit 44");

    emits(0x5b54038000270100, "Reviewed ISET integer-result word dispatches");
    check(current_field(DecodedField::destination_type, 0) && wrote_to(gpr(0)) &&
              count_ops(Opcode::predicate_to_mask) == 1,
          "Reviewed ISET word emits an integer all-ones mask result");

    // ISETP writes the paired predicate destinations.
    emits(opcode_pattern("010110110110") | set(std::uint64_t{0}, 3, 3, 4) | set(std::uint64_t{0}, 0, 3, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 3, 2) |
              set(std::uint64_t{0}, 49, 3, 5),
          "ISETP register word dispatches");
    check(wrote_to(pred(4)) && wrote_to(pred(5)), "ISETP writes both destination predicates");

    // IMNMX selects between the minimum and maximum by the guest predicate.
    emits(opcode_pattern("0101110000100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 3, 1),
          "IMNMX register word dispatches");
    check(count_ops(Opcode::minimum) == 1 && count_ops(Opcode::maximum) == 1 &&
              count_ops(Opcode::select) == 1,
          "IMNMX emits min, max and the predicate select");
}

void test_logical_and_shift_words() {
    // LOP R5, R6, R7 with XOR and a zero-test predicate write.
    emits(opcode_pattern("0101110001000") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 41, 2, 2) | set(std::uint64_t{0}, 44, 2, 3) |
              set(std::uint64_t{0}, 48, 3, 4),
          "LOP register word dispatches");
    check(count_ops(Opcode::bit_xor) == 1 && wrote_to(pred(4)) && wrote_to(gpr(5)),
          "LOP writes the word result and the zero-test predicate");

    // LOP3 R5, R6, R7, R8 with truth table 0xCA.
    emits(opcode_pattern("0101101111100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 28, 8, 0xCA),
          "LOP3 register word dispatches");
    check(wrote_to(gpr(5)), "LOP3 writes the expanded truth table");

    for (const auto route : {std::string_view{"001111"}, std::string_view{"0000001x"}}) {
        const bool immediate = route == "001111";
        const auto instruction = opcode_pattern(route) | 5ull | (6ull << 8) |
            (3ull << 39) | (0x96ull << 48) | (1ull << 47) |
            (immediate ? (1ull << 56) | (0xffull << 20) : 0);
        emits(instruction, "LOP3 immediate/constant word dispatches with CC");
        check(current_field(DecodedField::condition, 0) && current_field(DecodedField::extended, 0),
              "LOP3 nonregister routes explicitly have false predicate test and no extended-Z input");
        const auto* a = current_source(SourceRole::a);
        const auto* b = current_source(SourceRole::b);
        check(a && b && !a->modifiers.negate && !b->modifiers.negate,
              "LOP3 source-C bits never become A/B inversion flags");
    }

    // LOP32I R5, R6, 0xFF with AND.
    emits(opcode_pattern("000001") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0xFF) | set(std::uint64_t{0}, 53, 2, 0),
          "LOP32I word dispatches");
    check(count_ops(Opcode::bit_and) == 1, "LOP32I ands the immediate");

    // SHF right with the u64 MaxShift and wrap; A is the low word, C the high.
    emits(opcode_pattern("0101110011111") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 37, 2, 2) |
              (std::uint64_t{1} << 50),
          "SHF right register word dispatches");
    check(wrote_to(gpr(5)) && count_ops(Opcode::funnel_shift_right) == 1, "SHF builds a funnel shift");

    // SHL wraps the count with the M flag.
    emits(opcode_pattern("0101110001001") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | (std::uint64_t{1} << 39),
          "SHL register word dispatches");
    check(count_ops(Opcode::bit_and) == 1 && count_ops(Opcode::shift_left) == 1,
          "SHL masks the count under wrap");

    // SHR signs its shift with the signed selector.
    emits(opcode_pattern("0101110000101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | (std::uint64_t{1} << 48),
          "SHR register word dispatches");
    check(first_op(Opcode::shift_right) != nullptr &&
              first_op(Opcode::shift_right)->numeric.domain == Domain::signed_integer,
          "SHR shifts arithmetically when signed");
    // SHL condition codes are unimplemented and are rejected.
    rejected(opcode_pattern("0101110001001") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
                 set(std::uint64_t{0}, 20, 8, 7) | (std::uint64_t{1} << 47),
             "SHL condition-code writes are rejected");
}

void test_bitfield_move_words() {
    // BFE R5, R6, R7.
    emits(opcode_pattern("0101110000000") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7),
          "BFE register word dispatches");
    check(count_ops(Opcode::bit_extract) >= 1, "BFE extracts position and size from B");

    // BFI R5, R6, R7, R8.
    emits(opcode_pattern("0101101111110") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8),
          "BFI register word dispatches");
    check(count_ops(Opcode::bit_insert) == 1, "BFI inserts A into the C base");

    // FLO R5, R7 finds the most significant set bit.
    emits(opcode_pattern("0101110000110") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 7),
          "FLO register word dispatches");
    check(count_ops(Opcode::find_msb) == 1, "FLO finds the most significant bit");

    // POPC R5, R7.
    emits(opcode_pattern("0101110000001") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 7),
          "POPC register word dispatches");
    check(count_ops(Opcode::bit_count) == 1, "POPC counts bits");

    // MOV R5, R6 is a pure write.
    emits(opcode_pattern("0101110010011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6),
          "MOV register word dispatches");
    check(count_ops(Opcode::copy) == 1 && wrote_to(gpr(5)), "MOV copies the source");

    emits(opcode_pattern("000000010000") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 32, 0xBEEF),
          "MOV32I word dispatches");
    check(wrote_to(gpr(5)), "MOV32I writes the immediate");

    // SEL R5, R6, R7, P1.
    emits(opcode_pattern("0101110010100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 3, 1),
          "SEL register word dispatches");
    check(count_ops(Opcode::select) == 1, "SEL selects on the predicate");
}

void test_predicate_words() {
    // R2P scatters the low byte of R6 into P0..P6.
    emits(opcode_pattern("0101110011110") | set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 8, 7) |
              set(std::uint64_t{0}, 41, 2, 0),
          "R2P register word dispatches");
    check(wrote_to(pred(0)) && wrote_to(pred(6)), "R2P writes the predicate file");

    // P2R gathers P0..P6 of the R6 byte into R5.
    emits(opcode_pattern("0101110011101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7),
          "P2R register word dispatches");
    check(wrote_to(gpr(5)), "P2R writes the gathered word");

    // S2R is a documented structured rejection: no IO contract representation.
    const Diagnostic s2r = rejected(opcode_pattern("1111000011001") | set(std::uint64_t{0}, 0, 8, 5) |
                                        set(std::uint64_t{0}, 20, 8, 33),
                                    "S2R rejects with a structured diagnostic");
    check(s2r.code == DiagnosticCode::unimplemented, "S2R diagnostic names the missing IO contract");

    // The viewport W-scale factors (SR 30/31) read zero in a graphics stage.
    for (const std::uint64_t sreg : {30u, 31u}) {
        emits(opcode_pattern("1111000011001") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, sreg),
              "S2R viewport W-scale factor dispatches");
        bool zero = false;
        for (const auto& [target, value] : context.writes) {
            const auto* literal = std::get_if<LiteralBits>(&value);
            zero = zero || (target == gpr(5) && literal != nullptr && literal->bits == 0);
        }
        check(zero && context.writes.size() == 1, "S2R W-scale factor writes a zero word");
    }

    // CSET R5 = (LT(P2) || sign), integer Boolean word output.
    emits(opcode_pattern("0101000010011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 5, 1) |
              set(std::uint64_t{0}, 39, 3, 2) | set(std::uint64_t{0}, 45, 2, 1),
          "CSET word dispatches");
    check(wrote_to(gpr(5)) && count_ops(Opcode::compare) == 0, "CSET writes the condition result");
    check(current_field(DecodedField::destination_type, 0), "CSET reads BVal from bit 44");

    // CSETP P4/P5 from the carry flag.
    emits(opcode_pattern("0101000010100") | set(std::uint64_t{0}, 3, 3, 4) | set(std::uint64_t{0}, 0, 3, 5) |
              set(std::uint64_t{0}, 8, 5, 17) | set(std::uint64_t{0}, 39, 3, 1),
          "CSETP word dispatches");
    check(wrote_to(pred(4)) && wrote_to(pred(5)), "CSETP writes both predicates");

    // PSET R5 = (P1 | P2) & P3 (BoolOpAB=or, BoolOpC=and).
    emits(opcode_pattern("0101000010001") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 12, 3, 1) |
              set(std::uint64_t{0}, 29, 3, 2) | set(std::uint64_t{0}, 39, 3, 3) |
              set(std::uint64_t{0}, 24, 2, 1) | set(std::uint64_t{0}, 45, 2, 0),
          "PSET word dispatches");
    check(count_ops(Opcode::logical_or) == 1 && count_ops(Opcode::logical_and) == 1,
          "PSET combines the three predicates in order");
    check(current_field(DecodedField::destination_type, 0), "PSET reads BVal from bit 44");

    // PSETP P4/P5 from the same combination.
    emits(opcode_pattern("0101000010010") | set(std::uint64_t{0}, 3, 3, 4) | set(std::uint64_t{0}, 0, 3, 5) |
              set(std::uint64_t{0}, 12, 3, 1) | set(std::uint64_t{0}, 29, 3, 2) | set(std::uint64_t{0}, 39, 3, 3) |
              set(std::uint64_t{0}, 24, 2, 2),
          "PSETP word dispatches");
    check(wrote_to(pred(4)) && wrote_to(pred(5)) && count_ops(Opcode::logical_xor) == 1,
          "PSETP writes the result and its inverse");
}

// ---------------------------------------------------------------------------
// Warp shuffle: SHFL dispatches into the subgroup shuffle operation.
// ---------------------------------------------------------------------------

void test_shuffle_words() {
    // SHFL R4, R5, R6, R7, down mode; the validity predicate rides P1.
    emits(opcode_pattern("1110111100010") | set(std::uint64_t{0}, 0, 8, 4) | set(std::uint64_t{0}, 8, 8, 5) |
              set(std::uint64_t{0}, 20, 8, 6) | set(std::uint64_t{0}, 39, 8, 7) | (7ull << 16) |
              set(std::uint64_t{0}, 30, 2, 2) | set(std::uint64_t{0}, 48, 3, 1),
          "SHFL register word dispatches");
    const auto* shuffle = first_op(Opcode::shuffle);
    check(shuffle != nullptr, "SHFL emits the shuffle operation");
    if (shuffle != nullptr) {
        check(shuffle->inputs.size() == 3 && shuffle->outputs.size() == 2,
              "SHFL shuffle carries three inputs and two results");
        const auto& subgroup = std::get<SubgroupInfo>(shuffle->info);
        check(subgroup.shuffle == ShuffleKind::down && subgroup.width == 32,
              "SHFL keeps the mode selector and the 32-lane width");
        check(std::all_of(shuffle->inputs.begin(), shuffle->inputs.end(), [](const Operand& operand) {
                  return std::holds_alternative<ValueId>(operand);
              }),
              "SHFL shuffle inputs are register reads");
    }
    check(wrote_to(gpr(4)) && wrote_to(pred(1)), "SHFL writes the destination word and predicate");

    // Immediate lane (up) and immediate mask (butterfly) forms keep their kind.
    emits(opcode_pattern("1110111100010") | set(std::uint64_t{0}, 0, 8, 4) | set(std::uint64_t{0}, 8, 8, 5) |
              set(std::uint64_t{0}, 20, 5, 6) | std::uint64_t{1} << 28 | (7ull << 16) |
              set(std::uint64_t{0}, 30, 2, 1) | set(std::uint64_t{0}, 48, 3, 2),
          "SHFL immediate lane word dispatches");
    shuffle = first_op(Opcode::shuffle);
    check(shuffle != nullptr && std::get<SubgroupInfo>(shuffle->info).shuffle == ShuffleKind::up,
          "SHFL immediate lane keeps the up mode");
    check(shuffle != nullptr && std::holds_alternative<LiteralBits>(shuffle->inputs[1]),
          "SHFL immediate lane is a literal shuffle input");

    emits(opcode_pattern("1110111100010") | set(std::uint64_t{0}, 0, 8, 4) | set(std::uint64_t{0}, 8, 8, 5) |
              (0x101full << 34) | std::uint64_t{1} << 29 | (7ull << 16) |
              set(std::uint64_t{0}, 30, 2, 3) | set(std::uint64_t{0}, 48, 3, 2),
          "SHFL immediate mask word dispatches");
    shuffle = first_op(Opcode::shuffle);
    check(shuffle != nullptr && std::get<SubgroupInfo>(shuffle->info).shuffle == ShuffleKind::butterfly,
          "SHFL immediate mask keeps the butterfly mode");
    check(shuffle != nullptr && std::holds_alternative<LiteralBits>(shuffle->inputs[2]),
          "SHFL immediate mask is a literal shuffle input");
    check(wrote_to(gpr(4)) && wrote_to(pred(2)), "SHFL II writes its results");
}

// ---------------------------------------------------------------------------
// Lane 7: floating arithmetic, comparison, conversion, half and multifunction
// instructions.
// ---------------------------------------------------------------------------

void test_float_arithmetic_words() {
    // FADD R5, R6, R7 with FTZ.
    emits(opcode_pattern("0101110001011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | (std::uint64_t{1} << 44),
          "FADD register word dispatches");
    const auto* add = first_op(Opcode::add);
    check(add != nullptr && add->numeric.domain == Domain::floating &&
              add->numeric.scalar == ScalarType::f32 && add->numeric.flush_inputs,
          "FADD adds in the flushed f32 domain");

    emits(0x3858103f80070100, "FADD compact 1.0 word dispatches");
    const auto* source_b = current_source(SourceRole::b);
    check(source_b != nullptr && std::holds_alternative<LiteralBits>(source_b->value) &&
              std::get<LiteralBits>(source_b->value).bits == 0x3f800000,
          "FADD expands the compact floating immediate");

    emits(0x0863f80000070100, "FADD32I modifier word dispatches");
    check(count_ops(Opcode::add) == 1, "FADD32I adds the immediate");
    const auto* source_a = current_source(SourceRole::a);
    source_b = current_source(SourceRole::b);
    check(source_a != nullptr && source_a->modifiers.absolute,
          "FADD32I applies AbsA before emission");
    check(source_b != nullptr && source_b->modifiers.negate,
          "FADD32I applies NegB before emission");

    // FMUL R5, R6, R7 with a divide-by-two scale on A.
    emits(opcode_pattern("0101110001101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 41, 3, 1),
          "FMUL register word dispatches");
    check(count_ops(Opcode::multiply) == 2, "FMUL scales A then multiplies");

    emits(opcode_pattern("00011110") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x40000000),
          "FMUL32I word dispatches");
    check(count_ops(Opcode::multiply) == 1, "FMUL32I multiplies the immediate");

    emits(opcode_pattern("0101110001101") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 8, 7) |
              set(std::uint64_t{0}, 44, 2, 2),
          "FMUL FMZ word dispatches");
    check(current_field(DecodedField::function, 2), "FMUL preserves the FMZ selector for emission");

    // FFMA R5, R6, R7, R8 stays fused.
    emits(opcode_pattern("010110011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8),
          "FFMA register word dispatches");
    check(count_ops(Opcode::fused_multiply_add) == 1, "FFMA emits one fused operation");

    emits(opcode_pattern("000011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x3F800000),
          "FFMA32I word dispatches");
    check(count_ops(Opcode::fused_multiply_add) == 1, "FFMA32I stays fused");
    const auto* source_c = current_source(SourceRole::c);
    check(source_c != nullptr && std::holds_alternative<Register>(source_c->value) &&
              std::get<Register>(source_c->value) == gpr(5),
          "FFMA32I reads its addend from the old destination");

    emits(opcode_pattern("010100011") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 39, 8, 8) |
              set(std::uint64_t{0}, 34, 5, 1) | set(std::uint64_t{0}, 20, 14, 4),
          "FFMA register-constant word dispatches");
    source_b = current_source(SourceRole::b);
    source_c = current_source(SourceRole::c);
    check(source_b != nullptr && std::holds_alternative<Register>(source_b->value) &&
              std::get<Register>(source_b->value) == gpr(8),
          "FFMA Rc preserves the register multiplier");
    check(source_c != nullptr && std::holds_alternative<ConstantWord>(source_c->value),
          "FFMA Rc preserves the constant addend");

    emits(opcode_pattern("010110011") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 8, 7) |
              set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 53, 2, 2),
          "FFMA FMZ word dispatches");
    check(current_field(DecodedField::function, 2), "FFMA preserves the FMZ selector for emission");

    // FMNMX with the predicate select and FTZ.
    emits(opcode_pattern("0101110001100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 3, 2) | (std::uint64_t{1} << 44),
          "FMNMX register word dispatches");
    check(count_ops(Opcode::minimum) == 1 && count_ops(Opcode::maximum) == 1 &&
              count_ops(Opcode::select) == 1,
          "FMNMX emits the min/max select");
}

// FSWZADD: the per-lane swizzle add. One synthetic word per mask value and
// the modifier combinations, checked against the defined FSWZADD expression and
// Helper_SwizzleAdd semantics.
void test_swizzle_add_words() {
    const auto base = opcode_pattern("0101000011111");
    const auto encode = [&](std::uint32_t pn, std::uint64_t modifiers = 0) {
        return base | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
               set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 28, 8, pn) | modifiers;
    };
    emits(encode(0x1b), "FSWZADD register word dispatches");
    const auto* lane = first_op(Opcode::load_io);
    check(lane != nullptr && std::holds_alternative<IoInfo>(lane->info) &&
              std::get<IoInfo>(lane->info).semantic == IoSemantic::quad_lane_id &&
              std::get<IoInfo>(lane->info).type == IoType{ScalarType::u32, 1},
          "FSWZADD reads the fragment quad position builtin");
    check(count_ops(Opcode::multiply) == 2 && count_ops(Opcode::add) == 1 &&
              count_ops(Opcode::select) == 6,
          "FSWZADD keeps the sign-table product form");
    check(wrote_to(gpr(5)), "FSWZADD writes Rd");
    // Every PN word decodes and dispatches; the mask rides the code-selection
    // shift as an exact literal.
    for (std::uint32_t pn = 0; pn <= 255; ++pn) {
        if (!emits(encode(pn), "FSWZADD pn word dispatches")) break;
        const auto* shifted = first_op(Opcode::shift_right);
        check(shifted != nullptr && shifted->inputs.size() == 2 &&
                  std::holds_alternative<LiteralBits>(shifted->inputs[0]) &&
                  std::get<LiteralBits>(shifted->inputs[0]).bits == pn,
              "FSWZADD selects the code with the PN word");
    }
    // FTZ and the rounding selector ride the decoded mode onto every
    // arithmetic step unchanged. Guest rounding selector 3 maps to
    // toward-zero.
    emits(encode(0, std::uint64_t{1} << 44 | set(std::uint64_t{0}, 39, 2, 3)),
          "FSWZADD ftz word dispatches");
    const auto* product = first_op(Opcode::multiply);
    check(product != nullptr && product->numeric.flush_inputs && product->numeric.flush_output &&
              product->numeric.rounding == Rounding::toward_zero,
          "FSWZADD keeps FTZ and the rounding selector in the mode");
    // The write-CC bit adds the floating zero/sign flag writes.
    emits(encode(0, std::uint64_t{1} << 47), "FSWZADD cc word dispatches");
    const Register cc_zero{RegisterFile::condition_code, static_cast<std::uint16_t>(ConditionCode::zero)};
    const Register cc_sign{RegisterFile::condition_code, static_cast<std::uint16_t>(ConditionCode::sign)};
    check(count_ops(Opcode::compare) >= 5 && wrote_to(cc_zero) && wrote_to(cc_sign),
          "FSWZADD writes the floating CC flags");
    // The NDV modifier stays decoded beside the PN word and, as in the
    // defined semantics, does not change the emitted expression.
    emits(encode(0x80, std::uint64_t{1} << 38), "FSWZADD ndv word dispatches");
    check(current_field(DecodedField::permutation_mode, 0x180),
          "FSWZADD preserves the NDV bit beside the PN word");
    check(count_ops(Opcode::multiply) == 2 && count_ops(Opcode::add) == 1,
          "FSWZADD NDV leaves the expression unchanged");
}

void test_float_comparison_words() {
    // FSET R5, R6, R7, lt with the pass-through boolean op.
    emits(opcode_pattern("01011000") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 3, 1) | set(std::uint64_t{0}, 48, 4, 1) |
              set(std::uint64_t{0}, 45, 2, 3),
          "FSET register word dispatches");
    const auto* comparison = first_op(Opcode::compare);
    check(comparison != nullptr, "FSET compares");
    if (comparison) {
        const auto* info = std::get_if<ComparisonInfo>(&comparison->info);
        check(info != nullptr && info->relation == Compare::less && !info->unordered,
              "FSET relation follows the FComp selector");
    }
    check(count_ops(Opcode::predicate_to_mask) == 1, "FSET expands the Boolean result to an all-ones mask");

    emits(opcode_pattern("0011000x") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 19, 0x3f800) |
              set(std::uint64_t{0}, 48, 4, 2) | set(std::uint64_t{0}, 45, 2, 3),
          "FSET compact-immediate word dispatches");
    const auto* compact = current_source(SourceRole::b);
    check(compact != nullptr && std::holds_alternative<LiteralBits>(compact->value) &&
              std::get<LiteralBits>(compact->value).bits == 0x3f800000,
          "FSET expands its compact floating immediate");

    // FSETP P4/P5, R6, R7, lt.
    const auto fsetp_word = opcode_pattern("010110111011") | set(std::uint64_t{0}, 3, 3, 4) |
                            set(std::uint64_t{0}, 0, 3, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 48, 4, 2) |
                            set(std::uint64_t{0}, 45, 2, 1);
    emits(fsetp_word, "FSETP register word dispatches");
    check(wrote_to(pred(4)) && wrote_to(pred(5)), "FSETP writes both predicates");
    const auto* source_a = current_source(SourceRole::a);
    const auto* source_b = current_source(SourceRole::b);
    check(source_a != nullptr && !source_a->modifiers.absolute &&
              source_b != nullptr && !source_b->modifiers.negate,
          "FSETP does not consume FSET modifier bits");
    emits(fsetp_word | (std::uint64_t{1} << 6) | (std::uint64_t{1} << 7),
          "FSETP modifier word dispatches");
    source_a = current_source(SourceRole::a);
    source_b = current_source(SourceRole::b);
    check(source_a != nullptr && source_a->modifiers.absolute &&
              source_b != nullptr && source_b->modifiers.negate,
          "FSETP consumes AbsA and NegB from bits 7 and 6");

    emits(0x5bb1038000270107, "Reviewed FSETP modifier-position word dispatches");
    source_b = current_source(SourceRole::b);
    check(source_b != nullptr && !source_b->modifiers.negate &&
              count_ops(Opcode::compare) == 1 && wrote_to(pred(0)),
          "Reviewed FSETP word compares without spurious B negation");

    // FCMP selects between A and B on comparing C against zero.
    emits(opcode_pattern("010110111010") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 48, 4, 2),
          "FCMP register word dispatches");
    check(count_ops(Opcode::select) == 1, "FCMP selects between A and B");

    emits(opcode_pattern("0011011x1010") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 19, 0x3f800) |
              set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 48, 4, 2),
          "FCMP compact-immediate word dispatches");
    compact = current_source(SourceRole::b);
    check(compact != nullptr && std::holds_alternative<LiteralBits>(compact->value) &&
              std::get<LiteralBits>(compact->value).bits == 0x3f800000,
          "FCMP expands its compact floating immediate");
}

void test_conversion_words() {
    // F2F R5, R6 with pass rounding: no integral operation.
    emits(opcode_pattern("0101110010101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2),
          "F2F pass word dispatches");
    check(count_ops(Opcode::round) == 0 && count_ops(Opcode::convert) == 0,
          "F2F pass-through applies no integral rounding");

    emits(opcode_pattern("0011100x10101") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2) |
              set(std::uint64_t{0}, 20, 19, 0x3f800),
          "F2F compact-immediate word dispatches");
    const auto* compact = current_source(SourceRole::b);
    check(compact != nullptr && std::holds_alternative<LiteralBits>(compact->value) &&
              std::get<LiteralBits>(compact->value).bits == 0x3f800000,
          "F2F expands its compact floating immediate");

    // Same-type F2F with IntegerRound Round rounds to an integer: the round
    // operation must appear even though the numeric rounding is nearest-even.
    emits(opcode_pattern("0101110010101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2) | (std::uint64_t{1} << 42),
          "F2F round word dispatches");
    check(count_ops(Opcode::round) == 1, "Same-type F2F with Round emits the round operation");
    // Floor and truncate follow the low selector bits.
    emits(opcode_pattern("0101110010101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2) | set(std::uint64_t{0}, 39, 2, 1) |
              (std::uint64_t{1} << 42),
          "F2F floor word dispatches");
    check(count_ops(Opcode::floor) == 1, "F2F Floor emits the floor operation");

    // Cross-type F2F keeps a conversion and no integral rounding.
    emits(opcode_pattern("0101110010101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 8, 2, 3) | set(std::uint64_t{0}, 10, 2, 2) | (std::uint64_t{1} << 42),
          "F2F widen word dispatches");
    check(count_ops(Opcode::convert) == 1 && count_ops(Opcode::round) == 0,
          "Cross-type F2F converts without rounding to an integer");

    emits(opcode_pattern("0101110010101") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 20, 8, 6) | set(std::uint64_t{0}, 8, 2, 2) |
              set(std::uint64_t{0}, 10, 2, 1) | (std::uint64_t{1} << 41),
          "F2F high-half word dispatches");
    const auto* source_b = current_source(SourceRole::b);
    check(source_b != nullptr && source_b->modifiers.subword == Subword::half_high &&
              count_ops(Opcode::unpack_half2) == 1,
          "F2F emits from the selected high half");

    emits(0x5ca8020000270600, "Reviewed F2F high-half word dispatches");
    source_b = current_source(SourceRole::b);
    check(source_b != nullptr && source_b->modifiers.subword == Subword::half_high &&
              count_ops(Opcode::unpack_half2) == 1 && wrote_to(gpr(0)),
          "Reviewed F2F word emits from the high half into R0");

    emits(opcode_pattern("0101110010101") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 20, 8, 6) | set(std::uint64_t{0}, 8, 2, 1) |
              set(std::uint64_t{0}, 10, 2, 2) | set(std::uint64_t{0}, 39, 2, 3),
          "F2F narrowing round-toward-zero word dispatches");
    check(context.current.numeric && context.current.numeric->rounding == Rounding::toward_zero,
          "Cross-format F2F keeps conversion rounding through dispatch");

    // F2I R5, R6 to S32: explicit round and convert.
    emits(opcode_pattern("0101110010110") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 12, 1, 1) | set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2),
          "F2I word dispatches");
    check(count_ops(Opcode::round) == 1 && count_ops(Opcode::convert) == 1,
          "F2I rounds and converts explicitly");

    emits(opcode_pattern("0101110010110") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 20, 8, 6) | set(std::uint64_t{0}, 8, 2, 2) |
              set(std::uint64_t{0}, 10, 2, 1) | (std::uint64_t{1} << 41),
          "F2I high-half word dispatches");
    source_b = current_source(SourceRole::b);
    check(source_b != nullptr && source_b->modifiers.subword == Subword::half_high &&
              count_ops(Opcode::unpack_half2) == 1,
          "F2I emits from the selected high half");

    // I2F R5, R6 from U32.
    emits(opcode_pattern("0101110010111") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2),
          "I2F word dispatches");
    const auto* conversion = first_op(Opcode::convert);
    check(conversion != nullptr, "I2F converts");
    if (conversion) {
        const auto* info = std::get_if<ConversionInfo>(&conversion->info);
        check(info != nullptr && info->source == ScalarType::u32 && info->destination == ScalarType::f32,
              "I2F conversion names both types");
    }

    emits(opcode_pattern("0101110010111") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 20, 8, 6) | set(std::uint64_t{0}, 8, 2, 2) |
              (std::uint64_t{1} << 13),
          "I2F signed-byte word dispatches");
    conversion = first_op(Opcode::convert);
    check(current_field(DecodedField::source_type, 4), "I2F reads source signedness from bit 13");
    if (conversion) {
        const auto* info = std::get_if<ConversionInfo>(&conversion->info);
        check(info != nullptr && info->source == ScalarType::s32 && info->destination == ScalarType::f32,
              "I2F sign-extends the byte before floating conversion");
    }

    emits(0x5cb8000000272200, "Reviewed I2F signed-byte word dispatches");
    conversion = first_op(Opcode::convert);
    check(current_field(DecodedField::source_type, 4) && wrote_to(gpr(0)),
          "Reviewed I2F word keeps signed-byte source metadata");
    if (conversion) {
        const auto* info = std::get_if<ConversionInfo>(&conversion->info);
        check(info != nullptr && info->source == ScalarType::s32 &&
                  info->destination == ScalarType::f32,
              "Reviewed I2F word sign-extends before conversion");
    }

    // I2I R5, R6 U32 to U32 without saturation is a pure move.
    emits(opcode_pattern("0101110011100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2),
          "I2I plain word dispatches");
    check(count_ops(Opcode::clamp) == 0, "Unsaturated I2I clamps nothing");

    emits(opcode_pattern("0101110011100") | set(std::uint64_t{0}, 0, 8, 5) |
              set(std::uint64_t{0}, 20, 8, 6) | set(std::uint64_t{0}, 8, 2, 2) |
              (std::uint64_t{1} << 13),
          "I2I signed-byte word dispatches");
    check(current_field(DecodedField::source_type, 4) &&
              current_field(DecodedField::destination_type, 2),
          "I2I separates source and destination signedness");

    // I2I with the saturation bit clamps to the destination range.
    emits(opcode_pattern("0101110011100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6) |
              set(std::uint64_t{0}, 8, 2, 2) | set(std::uint64_t{0}, 10, 2, 2) | (std::uint64_t{1} << 50),
          "I2I saturated word dispatches");
    const auto* clamp = first_op(Opcode::clamp);
    check(clamp != nullptr, "Saturated I2I emits the clamp");
}

void test_half_float_words() {
    // HADD2 R5, R6, R7 (F16 swizzles, F16 output).
    emits(opcode_pattern("0101110100010") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7),
          "HADD2 register word dispatches");
    check(count_ops(Opcode::unpack_half2) >= 2 && count_ops(Opcode::add) == 2 &&
              count_ops(Opcode::pack_half2) == 1,
          "HADD2 computes both half lanes and packs");

    emits(opcode_pattern("0010110") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x38003800),
          "HADD232I word dispatches");
    check(count_ops(Opcode::add) == 2 && count_ops(Opcode::pack_half2) == 1, "HADD232I packs both lanes");

    // HFMA2 R5, R6, R7, R8.
    emits(opcode_pattern("0101110100000") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8),
          "HFMA2 register word dispatches");
    check(count_ops(Opcode::fused_multiply_add) == 2 && count_ops(Opcode::pack_half2) == 1,
          "HFMA2 fuses both lanes and packs");

    // HFMA232I: named distinctly and dispatched to the same fused path.
    emits(opcode_pattern("0010100") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x38003C00),
          "HFMA232I word dispatches");
    check(count_ops(Opcode::fused_multiply_add) == 2 && count_ops(Opcode::pack_half2) == 1,
          "HFMA232I reuses the fused half path");
    const auto* accumulator = current_source(SourceRole::c);
    check(accumulator && std::get_if<Register>(&accumulator->value) &&
              std::get<Register>(accumulator->value) == gpr(5) &&
              accumulator->modifiers.scalar == ScalarType::f16,
          "HFMA232I accumulates both packed halves from the old destination");

    // HMUL2 R5, R6, R7.
    emits(opcode_pattern("0101110100001") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7),
          "HMUL2 register word dispatches");
    check(count_ops(Opcode::multiply) == 2 && count_ops(Opcode::pack_half2) == 1, "HMUL2 multiplies both lanes");

    emits(opcode_pattern("0010101") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x38003800),
          "HMUL232I word dispatches");
    check(count_ops(Opcode::multiply) == 2, "HMUL232I multiplies both lanes");

    // Every reachable packed-half multiply/FMA route keeps FTZ admitted. FMZ
    // rejection is covered in the decode tests because rejected words do not
    // reach emitter dispatch.
    emits(opcode_pattern("0101110100000") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 8, 8) |
              set(std::uint64_t{0}, 37, 2, 1),
          "HFMA2 register FTZ word dispatches");
    emits(opcode_pattern("01110xxx0") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 19, 0x18001) | set(std::uint64_t{0}, 39, 8, 8) |
              set(std::uint64_t{0}, 57, 2, 1),
          "HFMA2 immediate FTZ word dispatches");
    emits(opcode_pattern("01110xxx1") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 14, 4) | set(std::uint64_t{0}, 34, 5, 1) |
              set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 57, 2, 1),
          "HFMA2 constant FTZ word dispatches");
    emits(opcode_pattern("01100xxx1") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 14, 4) | set(std::uint64_t{0}, 34, 5, 1) |
              set(std::uint64_t{0}, 39, 8, 8) | set(std::uint64_t{0}, 57, 2, 1),
          "HFMA2 register-constant FTZ word dispatches");
    emits(opcode_pattern("0101110100001") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 39, 2, 1),
          "HMUL2 register FTZ word dispatches");
    emits(opcode_pattern("0111100x0") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 10, 1) | set(std::uint64_t{0}, 30, 9, 1) |
              set(std::uint64_t{0}, 39, 2, 1),
          "HMUL2 immediate FTZ word dispatches");
    emits(opcode_pattern("0111100x1") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 14, 4) | set(std::uint64_t{0}, 34, 5, 1) |
              set(std::uint64_t{0}, 39, 2, 1),
          "HMUL2 constant FTZ word dispatches");
    emits(opcode_pattern("0010101") | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 32, 0x38003800) | set(std::uint64_t{0}, 55, 2, 1),
          "HMUL232I FTZ word dispatches");

    // HSET2 R5, R6, R7, lt with the integer boolean output.
    emits(opcode_pattern("0101110100011") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 35, 4, 1) | set(std::uint64_t{0}, 45, 2, 0),
          "HSET2 register word dispatches");
    check(wrote_to(gpr(5)) && count_ops(Opcode::compare) == 2, "HSET2 compares both lanes");

    // HSETP2 P4/P5, R6, R7, lt.
    emits(opcode_pattern("0101110100100") | set(std::uint64_t{0}, 3, 3, 4) | set(std::uint64_t{0}, 0, 3, 5) |
              set(std::uint64_t{0}, 8, 8, 6) | set(std::uint64_t{0}, 20, 8, 7) | set(std::uint64_t{0}, 35, 4, 1),
          "HSETP2 register word dispatches");
    check(wrote_to(pred(4)) && wrote_to(pred(5)), "HSETP2 writes both predicates");
}

void test_multifunction_words() {
    // MUFU R5, R6, rcp.
    emits(opcode_pattern("0101000010000") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 8, 8, 6) |
              set(std::uint64_t{0}, 20, 4, 4),
          "MUFU word dispatches");
    check(count_ops(Opcode::reciprocal) == 1, "MUFU selects the requested function");

    // RRO R5, R6 lowers to a modifier-preserving move.
    emits(opcode_pattern("0101110010010") | set(std::uint64_t{0}, 0, 8, 5) | set(std::uint64_t{0}, 20, 8, 6),
          "RRO word dispatches");
    check(wrote_to(gpr(5)), "RRO writes the reduced source");
}
} // namespace

int main() {
    test_integer_arithmetic_words();
    test_integer_comparison_words();
    test_logical_and_shift_words();
    test_bitfield_move_words();
    test_predicate_words();
    test_shuffle_words();
    test_float_arithmetic_words();
    test_swizzle_add_words();
    test_float_comparison_words();
    test_conversion_words();
    test_half_float_words();
    test_multifunction_words();
    if (failures) {
        std::cerr << failures << " decode/emit alignment assertion(s) failed\n";
        return 1;
    }
    std::cout << "Shader pipeline ALU decode/emit alignment passed\n";
    return 0;
}
