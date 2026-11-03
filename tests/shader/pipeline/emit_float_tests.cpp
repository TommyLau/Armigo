// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic tests for the lane 7 floating, half,
// conversion and multifunction instruction emitters. A recording
// EmitterContext captures every emitted operation and numeric mode so the
// guest modifier combinations can be checked exactly. No game bytes, captured
// all instruction words are synthetic.

#include "pipeline/decoder.h"
#include "pipeline/dispatch.h"

#include <array>
#include <iostream>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Mirrors a decode-table pattern: index zero names instruction bit 63.
constexpr std::uint64_t opcode_pattern(std::string_view pattern) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < pattern.size(); ++index) {
        if (pattern[index] == '1') value |= std::uint64_t{1} << (63 - index);
    }
    return value;
}

constexpr std::uint64_t set_bits(std::uint64_t word, unsigned first, unsigned width,
                                 std::uint64_t value) {
    const auto mask = ((std::uint64_t{1} << width) - 1) << first;
    return (word & ~mask) | (value << first);
}

constexpr NumericMode f32_mode(Rounding rounding, bool flush_inputs = false,
                               bool flush_output = false, bool saturate = false) {
    return {Domain::floating, ScalarType::f32, rounding, flush_inputs, flush_output, saturate};
}

Register gpr(std::uint16_t index) { return Register{RegisterFile::gpr, index}; }
Register pred_reg(std::uint16_t index) { return Register{RegisterFile::predicate, index}; }

struct RecordedOperation {
    Opcode opcode;
    NumericMode numeric;
    std::vector<Operand> inputs;
    std::vector<Operand> outputs;
    OperationInfo info;
};

class NullFacts final : public FactSource {
public:
    Knowledge<FactValue> lookup(FactKey) const override { return Unknown{"Not supplied"}; }
};

// Records every operation and register write a family emitter performs.
class RecordContext final : public EmitterContext {
public:
    RecordContext() : facts_(std::make_unique<FactReader>(facts_source_)) {}
    DecodedInstruction current;
    std::vector<RecordedOperation> operations;
    std::vector<std::pair<Register, Operand>> writes;
    mutable std::vector<Diagnostic> failures_recorded;
    std::uint32_t next_value = 0;

    [[nodiscard]] const DecodedProgram& decoded_program() const override { return program_; }
    [[nodiscard]] const DecodedInstruction& instruction() const override { return current; }
    [[nodiscard]] FunctionId function() const override { return FunctionId{0}; }
    [[nodiscard]] BlockId block() const override { return BlockId{0}; }
    [[nodiscard]] FactReader& facts() override { return *facts_; }
    [[nodiscard]] const Limits& limits() const override { return limits_; }
    [[nodiscard]] Result<Operand> read(Register reg) override {
        if (reg == zero_register) return Operand{LiteralBits{0, ValueType::word32}};
        if (reg == true_predicate) return Operand{LiteralBits{1, ValueType::predicate}};
        return Operand{reg};
    }
    [[nodiscard]] Status write(Register reg, Operand value) override {
        if (reg == zero_register || reg == true_predicate) return std::monostate{};
        writes.emplace_back(reg, value);
        return std::monostate{};
    }
    [[nodiscard]] Result<ValueId> temporary(ValueType) override { return ValueId{next_value++}; }
    [[nodiscard]] Status assign(ValueId, Operand) override { return std::monostate{}; }
    [[nodiscard]] Result<OperationId> append(Operation operation) override {
        operations.push_back({operation.opcode, operation.numeric, operation.inputs,
                              operation.outputs, operation.info});
        return OperationId{next_value++};
    }
    [[nodiscard]] Result<std::vector<ValueId>> emit(Opcode opcode, NumericMode numeric,
                                                    std::span<const Operand> inputs,
                                                    std::span<const ValueType> outputs,
                                                    OperationInfo info, Effects) override {
        std::vector<ValueId> results;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            results.push_back(ValueId{next_value++});
        }
        operations.push_back({opcode, numeric, {inputs.begin(), inputs.end()},
                              {results.begin(), results.end()}, std::move(info)});
        return results;
    }
    [[nodiscard]] Result<Operand> unary(Opcode opcode, Operand input, NumericMode numeric,
                                        ValueType output) override {
        return single(opcode, numeric, {input}, output, std::monostate{});
    }
    [[nodiscard]] Result<Operand> binary(Opcode opcode, Operand left, Operand right,
                                         NumericMode numeric, ValueType output) override {
        return single(opcode, numeric, {left, right}, output, std::monostate{});
    }
    [[nodiscard]] Result<Operand> ternary(Opcode opcode, Operand first, Operand second,
                                          Operand third, NumericMode numeric,
                                          ValueType output) override {
        return single(opcode, numeric, {first, second, third}, output, std::monostate{});
    }
    [[nodiscard]] Result<Operand> convert(Operand input, ConversionInfo conversion,
                                          NumericMode numeric, ValueType output) override {
        return single(Opcode::convert, numeric, {input}, output, conversion);
    }
    [[nodiscard]] Result<Operand> compare(Operand left, Operand right, ComparisonInfo comparison,
                                          NumericMode numeric) override {
        return single(Opcode::compare, numeric, {left, right}, ValueType::predicate, comparison);
    }
    [[nodiscard]] Result<Operand> select(Operand condition, Operand when_true, Operand when_false,
                                         ValueType) override {
        return single(Opcode::select, {}, {condition, when_true, when_false}, ValueType::word32,
                      std::monostate{});
    }
    [[nodiscard]] Result<std::vector<ValueId>> load(MemoryInfo, std::span<const Operand>,
                                                    NumericMode,
                                                    std::span<const ValueType>) override {
        return error(DiagnosticCode::unimplemented, "Loads are outside the lane 7 tests");
    }
    [[nodiscard]] Status store(MemoryInfo, std::span<const Operand>, NumericMode) override {
        return error(DiagnosticCode::unimplemented, "Stores are outside the lane 7 tests");
    }
    [[nodiscard]] Result<std::vector<ValueId>> texture(Opcode, TextureInfo,
                                                       std::span<const Operand>, NumericMode,
                                                       std::span<const ValueType>) override {
        return error(DiagnosticCode::unimplemented, "Textures are outside the lane 7 tests");
    }
    [[nodiscard]] Result<std::vector<ValueId>> load_io(IoInfo, std::span<const Operand>,
                                                       NumericMode,
                                                       std::span<const ValueType>) override {
        return error(DiagnosticCode::unimplemented, "IO loads are outside the lane 7 tests");
    }
    [[nodiscard]] Status store_io(IoInfo, std::span<const Operand>, NumericMode) override {
        return error(DiagnosticCode::unimplemented, "IO stores are outside the lane 7 tests");
    }
    [[nodiscard]] Status discard() override { return std::monostate{}; }
    [[nodiscard]] Result<BlockId> label(Pc) override {
        return error(DiagnosticCode::unimplemented, "Labels are outside the lane 7 tests");
    }
    [[nodiscard]] Result<BlockId> create_block() override {
        return error(DiagnosticCode::unimplemented, "Blocks are outside the lane 7 tests");
    }
    [[nodiscard]] Status enter_block(BlockId) override { return std::monostate{}; }
    [[nodiscard]] Status terminate(Terminator) override { return std::monostate{}; }
    [[nodiscard]] Result<ValueId> reconvergence_selector(BlockId, Pc) override {
        return error(DiagnosticCode::unimplemented, "Selectors are outside the lane 7 tests");
    }
    [[nodiscard]] Diagnostic error(DiagnosticCode code, std::string message,
                                   std::string modifier = {}) const override {
        Diagnostic diagnostic =
            diagnose(Phase::emit_ir, code, {current.pc, current.name}, std::move(message));
        diagnostic.modifier = std::move(modifier);
        failures_recorded.push_back(diagnostic);
        return diagnostic;
    }

    [[nodiscard]] bool failed() const { return !failures_recorded.empty(); }
    void reset() {
        operations.clear();
        writes.clear();
        failures_recorded.clear();
        next_value = 0;
    }

private:
    [[nodiscard]] Result<Operand> single(Opcode opcode, NumericMode numeric,
                                         std::vector<Operand>&& inputs, ValueType output,
                                         OperationInfo info) {
        const std::array<ValueType, 1> outputs{output};
        auto results = emit(opcode, numeric, inputs, outputs, info, Effects::none);
        if (results.index() == 1) return std::move(std::get<1>(results));
        return Operand{std::get<std::vector<ValueId>>(results)[0]};
    }
    DecodedProgram program_;
    NullFacts facts_source_;
    std::unique_ptr<FactReader> facts_;
    static constexpr Limits limits_{};
};

bool decode_and_emit(std::uint64_t word, std::string_view message, RecordContext& context) {
    const DecodeRule* matched = nullptr;
    for (const auto& rule : alu_decode_rules()) {
        if ((word & rule.mask) == rule.value) {
            matched = &rule;
            break;
        }
    }
    check(matched != nullptr, message);
    if (matched == nullptr) return false;
    auto decoded = matched->decode(40, RawInstruction{word});
    if (const auto* diagnostic = std::get_if<Diagnostic>(&decoded)) {
        std::cerr << "  decode error: " << diagnostic->message << '\n';
        check(false, message);
        return false;
    }
    context.current = std::get<DecodedInstruction>(std::move(decoded));
    const Status emitted = dispatch_instruction(context);
    if (const auto* diagnostic = std::get_if<Diagnostic>(&emitted)) {
        std::cerr << "  emit error: " << diagnostic->message << '\n';
        check(false, message);
        return false;
    }
    return true;
}

[[nodiscard]] const RecordedOperation* find(const RecordContext& context, Opcode opcode) {
    for (const auto& operation : context.operations) {
        if (operation.opcode == opcode) return &operation;
    }
    return nullptr;
}
[[nodiscard]] std::size_t count(const RecordContext& context, Opcode opcode) {
    std::size_t total = 0;
    for (const auto& operation : context.operations) {
        if (operation.opcode == opcode) ++total;
    }
    return total;
}
[[nodiscard]] bool writes_to(const RecordContext& context, Register reg) {
    for (const auto& written : context.writes) {
        if (written.first == reg) return true;
    }
    return false;
}
[[nodiscard]] bool holds_register(const Operand& operand, Register reg) {
    const auto* candidate = std::get_if<Register>(&operand);
    return candidate != nullptr && *candidate == reg;
}
[[nodiscard]] bool holds_literal(const Operand& operand, std::uint32_t bits,
                                 ValueType type = ValueType::word32) {
    const auto* literal = std::get_if<LiteralBits>(&operand);
    return literal != nullptr && literal->bits == bits && literal->type == type;
}

void add_source(DecodedInstruction& instruction, SourceRole role,
                std::variant<Register, LiteralBits, ConstantWord> value,
                SourceModifiers modifiers = {}) {
    instruction.sources.push_back({role, std::move(value), modifiers});
}
void add_field(DecodedInstruction& instruction, DecodedField field, std::uint64_t bits) {
    instruction.fields.push_back({field, bits});
}

// A canonical float arithmetic instruction: sources R1/R2, destination R3.
RecordContext arithmetic_context(InstructionName name, NumericMode numeric, bool write_cc = false) {
    RecordContext context;
    context.current = DecodedInstruction{};
    context.current.pc = 8;
    context.current.name = name;
    context.current.predicate.enabled = true;
    context.current.numeric = numeric;
    add_source(context.current, SourceRole::a, gpr(1), {});
    add_source(context.current, SourceRole::b, gpr(2), {});
    context.current.registers.push_back({RegisterRole::destination, 0, gpr(3), Access::write});
    add_field(context.current, DecodedField::write_cc, write_cc ? 1 : 0);
    return context;
}

std::uint64_t float_selector(ScalarType type) {
    switch (type) {
    case ScalarType::f16: return 1;
    case ScalarType::f64: return 3;
    default: return 2;
    }
}

// The guest F2F IntegerRound selector: nearest-even on a same-type conversion
// means pass unless the caller asks for round-to-integer explicitly.
std::uint64_t integral_round_selector(Rounding rounding) {
    switch (rounding) {
    case Rounding::toward_negative: return 5;
    case Rounding::toward_positive: return 6;
    case Rounding::toward_zero: return 7;
    default: return 1;
    }
}

void test_float_arithmetic() {
    // Every rounding, flush and saturation combination lands in the add op
    // exactly as decoded; unmodified sources pass straight through.
    for (const Rounding rounding : {Rounding::nearest_even, Rounding::toward_zero,
                                    Rounding::toward_positive, Rounding::toward_negative}) {
        for (const bool saturate : {false, true}) {
            for (const bool flush_inputs : {false, true}) {
                for (const bool flush_output : {false, true}) {
                    const NumericMode mode = f32_mode(rounding, flush_inputs, flush_output,
                                                      saturate);
                    RecordContext context = arithmetic_context(InstructionName::fadd, mode);
                    const Status result = dispatch_instruction(context);
                    check(std::holds_alternative<std::monostate>(result), "FADD emits");
                    check(!context.failed(), "FADD reports no failure");
                    const RecordedOperation* add = find(context, Opcode::add);
                    check(add != nullptr, "FADD emits an add");
                    if (add) {
                        check(add->numeric == mode,
                              "FADD preserves every numeric modifier");
                        check(add->inputs.size() == 2, "FADD adds two inputs");
                        check(holds_register(add->inputs[0], gpr(1)) &&
                                  holds_register(add->inputs[1], gpr(2)),
                              "FADD consumes the decoded sources");
                        check(writes_to(context, gpr(3)), "FADD writes its destination");
                    }
                    check(count(context, Opcode::compare) == 0,
                          "FADD without CC writes no flags");
                }
            }
        }
    }
    // Abs/negate expand to explicit unary operations ahead of the add.
    RecordContext modified = arithmetic_context(InstructionName::fadd,
                                                f32_mode(Rounding::nearest_even));
    modified.current.sources[0].modifiers = {true, true};
    modified.current.sources[1].modifiers = {false, true};
    const Status modified_result = dispatch_instruction(modified);
    check(std::holds_alternative<std::monostate>(modified_result), "Modified FADD emits");
    check(count(modified, Opcode::absolute) == 1 && count(modified, Opcode::negate) == 2,
          "FADD source modifiers become explicit unaries");
    // Condition-code flags compare the result against zero.
    RecordContext flags = arithmetic_context(InstructionName::fadd,
                                             f32_mode(Rounding::nearest_even), true);
    const Status flags_result = dispatch_instruction(flags);
    check(std::holds_alternative<std::monostate>(flags_result), "FADD with CC emits");
    check(count(flags, Opcode::compare) == 2, "FADD CC writes two flag comparisons");
    bool zero_flag = false;
    bool sign_flag = false;
    for (const auto& written : flags.writes) {
        if (written.first.file != RegisterFile::condition_code) continue;
        if (written.first.index == static_cast<std::uint16_t>(ConditionCode::zero)) {
            zero_flag = true;
        }
        if (written.first.index == static_cast<std::uint16_t>(ConditionCode::sign)) {
            sign_flag = true;
        }
    }
    check(zero_flag && sign_flag, "FADD CC targets the zero and sign flags");
    const RecordedOperation* flag_compare = find(flags, Opcode::compare);
    check(flag_compare != nullptr && !flag_compare->numeric.saturate &&
              !flag_compare->numeric.flush_output,
          "Flag comparisons drop output modifiers");
    // A missing numeric mode must fail loudly instead of assuming one.
    RecordContext modeless = arithmetic_context(InstructionName::fadd,
                                                f32_mode(Rounding::nearest_even));
    modeless.current.numeric.reset();
    const Status modeless_result = dispatch_instruction(modeless);
    check(std::holds_alternative<Diagnostic>(modeless_result),
          "FADD without a decoded numeric mode fails");
}

void test_float_multiply_and_fma() {
    // FMUL scale selectors multiply by exact power-of-two words first. The
    // selector order uses divide selectors 1-3 to scale
    // by 0.5/0.25/0.125, and multiply selectors 4-6 scale by 8/4/2.
    const std::array<std::uint32_t, 7> scale_words{0, 0x3F000000U, 0x3E800000U, 0x3E000000U,
                                                   0x41000000U, 0x40800000U, 0x40000000U};
    for (std::uint64_t scale = 0; scale < scale_words.size(); ++scale) {
        RecordContext context = arithmetic_context(InstructionName::fmul,
                                                   f32_mode(Rounding::nearest_even));
        add_field(context.current, DecodedField::result_scale, scale);
        add_field(context.current, DecodedField::function, 0);
        const Status result = dispatch_instruction(context);
        check(std::holds_alternative<std::monostate>(result), "FMUL emits");
        if (scale == 0) {
            check(count(context, Opcode::multiply) == 1,
                  "FMUL without scale multiplies once");
            continue;
        }
        check(count(context, Opcode::multiply) == 2,
              "FMUL scale multiplies before the result");
        const RecordedOperation* first = nullptr;
        for (const auto& operation : context.operations) {
            if (operation.opcode == Opcode::multiply) {
                first = &operation;
                break;
            }
        }
        check(first != nullptr && first->inputs.size() == 2 &&
                  holds_literal(first->inputs[1], scale_words[scale]),
              "FMUL scale constant is the exact power of two");
        check(first != nullptr && !first->numeric.saturate,
              "FMUL scale multiply never saturates");
    }
    RecordContext invalid_scale = arithmetic_context(InstructionName::fmul,
                                                     f32_mode(Rounding::nearest_even));
    add_field(invalid_scale.current, DecodedField::result_scale, 7);
    add_field(invalid_scale.current, DecodedField::function, 0);
    const Status invalid_result = dispatch_instruction(invalid_scale);
    check(std::holds_alternative<Diagnostic>(invalid_result),
          "Unknown FMUL scale fails instead of assuming no scale");

    RecordContext fused = arithmetic_context(InstructionName::ffma,
                                             f32_mode(Rounding::toward_negative, false, false,
                                                      true));
    add_source(fused.current, SourceRole::c, gpr(4), {false, true});
    add_field(fused.current, DecodedField::function, 0);
    const Status fused_result = dispatch_instruction(fused);
    check(std::holds_alternative<std::monostate>(fused_result), "FFMA emits");
    const RecordedOperation* fma = find(fused, Opcode::fused_multiply_add);
    check(fma != nullptr, "FFMA emits one fused operation");
    check(fma != nullptr &&
              fma->numeric == f32_mode(Rounding::toward_negative, false, false, true),
          "FFMA keeps rounding and saturation");
    check(fma != nullptr && fma->inputs.size() == 3, "FFMA consumes three inputs");
    check(count(fused, Opcode::negate) >= 1, "FFMA negation stays explicit");
}

void test_float_min_max() {
    RecordContext context = arithmetic_context(InstructionName::fmnmx,
                                               f32_mode(Rounding::nearest_even));
    add_source(context.current, SourceRole::predicate, pred_reg(2), {false, true});
    const Status result = dispatch_instruction(context);
    check(std::holds_alternative<std::monostate>(result), "FMNMX emits");
    const RecordedOperation* minimum = find(context, Opcode::minimum);
    const RecordedOperation* maximum = find(context, Opcode::maximum);
    check(minimum != nullptr && maximum != nullptr,
          "FMNMX computes both directions for the predicate select");
    check(minimum != nullptr && !minimum->numeric.saturate,
          "FMNMX min/max carry no saturation");
    const RecordedOperation* select = find(context, Opcode::select);
    check(select != nullptr, "FMNMX selects by the source predicate");
    check(select != nullptr && std::holds_alternative<ValueId>(select->inputs[0]) &&
              count(context, Opcode::logical_not) == 1,
          "FMNMX selection condition inverts the negated predicate");
    RecordContext flushed = arithmetic_context(InstructionName::fmnmx,
                                               f32_mode(Rounding::nearest_even, true, true));
    flushed.current.numeric->precise = true;
    add_source(flushed.current, SourceRole::predicate, pred_reg(2), {});
    check(std::holds_alternative<std::monostate>(dispatch_instruction(flushed)), "FMNMX.FTZ emits");
    for (const auto opcode : {Opcode::minimum, Opcode::maximum}) {
        const auto* operation = find(flushed, opcode);
        check(operation != nullptr && operation->numeric == *flushed.current.numeric,
              "FMNMX keeps both flush boundaries and precision on each extremum");
    }
}

void test_float_set_and_set_predicate() {
    // Comparison selector vocabulary: LT..GE with and without the unordered
    // flag; relation and unordered flag come straight from the selector bits.
    struct SelectorExpectation {
        std::uint64_t selector;
        Compare relation;
        bool unordered;
    };
    const std::array<SelectorExpectation, 12> expectations{{
        {1, Compare::less, false},           {2, Compare::equal, false},
        {3, Compare::less_equal, false},     {4, Compare::greater, false},
        {5, Compare::not_equal, false},      {6, Compare::greater_equal, false},
        {9, Compare::less, true},            {10, Compare::equal, true},
        {11, Compare::less_equal, true},     {12, Compare::greater, true},
        {13, Compare::not_equal, true},      {14, Compare::greater_equal, true},
    }};
    for (const auto& expectation : expectations) {
        RecordContext context = arithmetic_context(InstructionName::fset,
                                                   f32_mode(Rounding::nearest_even));
        add_field(context.current, DecodedField::condition, expectation.selector);
        add_field(context.current, DecodedField::boolean_operation, 3);
        add_field(context.current, DecodedField::half_mode, 0);
        add_source(context.current, SourceRole::predicate, pred_reg(1), {});
        const Status result = dispatch_instruction(context);
        check(std::holds_alternative<std::monostate>(result), "FSET emits");
        const RecordedOperation* comparison = find(context, Opcode::compare);
        check(comparison != nullptr, "FSET emits a comparison");
        if (comparison) {
            const auto* info = std::get_if<ComparisonInfo>(&comparison->info);
            check(info != nullptr && info->relation == expectation.relation &&
                      info->unordered == expectation.unordered,
                  "FSET comparison relation matches the selector");
        }
    }
    // Never/always and NaN forms expand without a comparison instruction.
    {
        RecordContext never = arithmetic_context(InstructionName::fset,
                                                 f32_mode(Rounding::nearest_even));
        add_field(never.current, DecodedField::condition, 0);
        add_field(never.current, DecodedField::boolean_operation, 3);
        add_field(never.current, DecodedField::half_mode, 0);
        add_source(never.current, SourceRole::predicate, pred_reg(1), {});
        (void)dispatch_instruction(never);
        check(find(never, Opcode::compare) == nullptr && count(never, Opcode::is_nan) == 0,
              "FSET F is the false constant without a comparison");
        RecordContext nan_case = arithmetic_context(InstructionName::fset,
                                                    f32_mode(Rounding::nearest_even));
        add_field(nan_case.current, DecodedField::condition, 8);
        add_field(nan_case.current, DecodedField::boolean_operation, 3);
        add_field(nan_case.current, DecodedField::half_mode, 0);
        add_source(nan_case.current, SourceRole::predicate, pred_reg(1), {});
        (void)dispatch_instruction(nan_case);
        check(count(nan_case, Opcode::is_nan) == 2 &&
                  count(nan_case, Opcode::logical_or) == 1 &&
                  count(nan_case, Opcode::logical_not) == 0,
              "FSET NAN is the OR of both NaN tests");
        RecordContext number = arithmetic_context(InstructionName::fset,
                                                  f32_mode(Rounding::nearest_even));
        add_field(number.current, DecodedField::condition, 7);
        add_field(number.current, DecodedField::boolean_operation, 3);
        add_field(number.current, DecodedField::half_mode, 0);
        add_source(number.current, SourceRole::predicate, pred_reg(1), {});
        (void)dispatch_instruction(number);
        check(count(number, Opcode::logical_not) == 1, "FSET NUM inverts the NaN OR");
    }
    // Boolean output selector: float 1.0/0.0 or integer word, with matching flags.
    {
        RecordContext boolean_float = arithmetic_context(InstructionName::fset,
                                                         f32_mode(Rounding::nearest_even), true);
        add_field(boolean_float.current, DecodedField::condition, 1);
        add_field(boolean_float.current, DecodedField::boolean_operation, 0);
        add_field(boolean_float.current, DecodedField::half_mode, 1);
        add_source(boolean_float.current, SourceRole::predicate, pred_reg(1), {});
        (void)dispatch_instruction(boolean_float);
        const RecordedOperation* select = find(boolean_float, Opcode::select);
        check(select != nullptr && holds_literal(select->inputs[1], 0x3F800000U) &&
                  holds_literal(select->inputs[2], 0),
              "FSET boolean float selects 1.0/0.0 words");
        check(count(boolean_float, Opcode::convert) == 0,
              "FSET boolean float needs no word widening");
        RecordContext integer_boolean = arithmetic_context(InstructionName::fset,
                                                           f32_mode(Rounding::nearest_even),
                                                           true);
        add_field(integer_boolean.current, DecodedField::condition, 1);
        add_field(integer_boolean.current, DecodedField::boolean_operation, 0);
        add_field(integer_boolean.current, DecodedField::half_mode, 0);
        add_source(integer_boolean.current, SourceRole::predicate, pred_reg(1), {});
        (void)dispatch_instruction(integer_boolean);
        check(count(integer_boolean, Opcode::predicate_to_mask) == 1,
              "FSET integer boolean expands the predicate to an all-ones mask");
        check(count(integer_boolean, Opcode::compare) == 3,
              "FSET integer boolean writes integer flags beside the comparison");
    }
    // FSETP writes both destinations, the second carrying the inverted result.
    {
        RecordContext context = arithmetic_context(InstructionName::fsetp,
                                                   f32_mode(Rounding::nearest_even));
        add_field(context.current, DecodedField::condition, 1);
        add_field(context.current, DecodedField::boolean_operation, 0);
        add_source(context.current, SourceRole::predicate, pred_reg(0), {});
        context.current.registers.push_back(
            {RegisterRole::predicate_destination, 0, pred_reg(4), Access::write});
        context.current.registers.push_back(
            {RegisterRole::second_predicate_destination, 0, pred_reg(5), Access::write});
        const Status result = dispatch_instruction(context);
        check(std::holds_alternative<std::monostate>(result), "FSETP emits");
        check(writes_to(context, pred_reg(4)) && writes_to(context, pred_reg(5)),
              "FSETP writes both predicate destinations");
        check(count(context, Opcode::logical_not) >= 1 &&
                  count(context, Opcode::logical_and) == 2,
              "FSETP combines both results with the boolean operation");
    }
}

void test_float_compare_select() {
    RecordContext context = arithmetic_context(InstructionName::fcmp,
                                               f32_mode(Rounding::nearest_even));
    add_source(context.current, SourceRole::c, gpr(4), {});
    add_field(context.current, DecodedField::condition, 2);
    const Status result = dispatch_instruction(context);
    check(std::holds_alternative<std::monostate>(result), "FCMP emits");
    const RecordedOperation* comparison = find(context, Opcode::compare);
    check(comparison != nullptr && holds_literal(comparison->inputs[1], 0),
          "FCMP compares source C against a zero word");
    const RecordedOperation* select = find(context, Opcode::select);
    check(select != nullptr && holds_register(select->inputs[1], gpr(1)) &&
              holds_register(select->inputs[2], gpr(2)),
          "FCMP selects between A and B");
}

void test_f2f() {
    const auto build = [](ScalarType source, ScalarType destination, Rounding rounding,
                          bool saturate = false, SourceModifiers modifiers = {}) {
        RecordContext context;
        context.current = DecodedInstruction{};
        context.current.pc = 8;
        context.current.name = InstructionName::f2f;
        context.current.predicate.enabled = true;
        context.current.numeric = f32_mode(rounding, false, false, saturate);
        add_field(context.current, DecodedField::source_type, float_selector(source));
        add_field(context.current, DecodedField::destination_type, float_selector(destination));
        add_field(context.current, DecodedField::integral_round, integral_round_selector(rounding));
        add_source(context.current, SourceRole::b, gpr(2), modifiers);
        context.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        context.current.registers.push_back(
            {RegisterRole::second_destination, 0, gpr(4), Access::write});
        return context;
    };
    {
        RecordContext passthrough =
            build(ScalarType::f32, ScalarType::f32, Rounding::nearest_even);
        (void)dispatch_instruction(passthrough);
        check(passthrough.operations.empty() && writes_to(passthrough, gpr(3)),
              "Pass-through F2F is a pure destination write");
    }
    {
        RecordContext truncated =
            build(ScalarType::f32, ScalarType::f32, Rounding::toward_zero);
        (void)dispatch_instruction(truncated);
        const RecordedOperation* truncate = find(truncated, Opcode::truncate);
        check(truncate != nullptr && truncate->numeric.rounding == Rounding::toward_zero,
              "F2F truncation carries the guest rounding");
        check(find(truncated, Opcode::floor) == nullptr &&
                  find(truncated, Opcode::ceiling) == nullptr,
              "F2F emits only the requested integral direction");
        RecordContext floored =
            build(ScalarType::f32, ScalarType::f32, Rounding::toward_negative);
        (void)dispatch_instruction(floored);
        check(find(floored, Opcode::floor) != nullptr, "F2F floor follows the selector");
        RecordContext ceiled =
            build(ScalarType::f32, ScalarType::f32, Rounding::toward_positive);
        (void)dispatch_instruction(ceiled);
        check(find(ceiled, Opcode::ceiling) != nullptr, "F2F ceiling follows the selector");
    }
    {
        RecordContext saturated =
            build(ScalarType::f32, ScalarType::f32, Rounding::nearest_even, true);
        (void)dispatch_instruction(saturated);
        const RecordedOperation* conversion = find(saturated, Opcode::convert);
        check(conversion != nullptr && conversion->numeric.saturate,
              "Saturating pass-through F2F carries saturation");
    }
    {
        // A rounding mode together with saturation: the rounding becomes the
        // dedicated integral operation, and the saturation boundary follows
        // it as a rounding-free conversion after rounding and before
        // saturate). The rounding op itself never saturates; the boundary
        // conversion carries nearest-even because it rounds nothing.
        for (const auto [rounding, opcode] :
             {std::pair{Rounding::toward_negative, Opcode::floor},
              std::pair{Rounding::toward_positive, Opcode::ceiling},
              std::pair{Rounding::toward_zero, Opcode::truncate}}) {
            RecordContext combined = build(ScalarType::f32, ScalarType::f32, rounding, true);
            (void)dispatch_instruction(combined);
            const RecordedOperation* integral = find(combined, opcode);
            check(integral != nullptr,
                  "Rounding-with-saturation F2F keeps its dedicated rounding operation");
            check(integral == nullptr || !integral->numeric.saturate,
                  "The dedicated rounding operation carries no saturation");
            const RecordedOperation* conversion = find(combined, Opcode::convert);
            check(conversion != nullptr && conversion->numeric.saturate &&
                      conversion->numeric.rounding == Rounding::nearest_even,
                  "The saturation boundary conversion rounds nothing explicitly");
        }
    }
    {
        RecordContext widened = build(ScalarType::f32, ScalarType::f64, Rounding::nearest_even);
        (void)dispatch_instruction(widened);
        const RecordedOperation* conversion = find(widened, Opcode::convert);
        check(conversion != nullptr,
              "F32 to F64 conversion stays a conversion (no FP32 substitution)");
        if (conversion) {
            const auto* info = std::get_if<ConversionInfo>(&conversion->info);
            check(info != nullptr && info->source == ScalarType::f32 &&
                      info->destination == ScalarType::f64,
                  "F32 to F64 conversion metadata names both types");
            check(conversion->numeric.scalar == ScalarType::f32,
                  "Conversion numeric mode describes the floating source");
        }
        check(find(widened, Opcode::unpack_word64) != nullptr && writes_to(widened, gpr(3)) &&
                  writes_to(widened, gpr(4)),
              "F64 results unpack into the low and high destinations");
    }
    {
        RecordContext narrowed =
            build(ScalarType::f64, ScalarType::f32, Rounding::nearest_even);
        (void)dispatch_instruction(narrowed);
        check(find(narrowed, Opcode::pack_word64) != nullptr,
              "F64 sources pack the register pair explicitly");
        check(find(narrowed, Opcode::convert) != nullptr,
              "F64 to F32 keeps an explicit conversion");
    }
    {
        SourceModifiers high_half{};
        high_half.subword = Subword::half_high;
        high_half.scalar = ScalarType::f16;
        RecordContext half_source =
            build(ScalarType::f16, ScalarType::f32, Rounding::nearest_even, false, high_half);
        (void)dispatch_instruction(half_source);
        check(find(half_source, Opcode::unpack_half2) != nullptr,
              "F16 sources unpack the packed word");
    }
    {
        RecordContext half_destination =
            build(ScalarType::f32, ScalarType::f16, Rounding::nearest_even);
        (void)dispatch_instruction(half_destination);
        const RecordedOperation* pack = find(half_destination, Opcode::pack_half2);
        check(pack != nullptr && holds_literal(pack->inputs[1], 0),
              "F16 destinations pack the value with a zero high half");
    }
    {
        RecordContext invalid = build(ScalarType::f16, ScalarType::f64, Rounding::nearest_even);
        const Status result = dispatch_instruction(invalid);
        check(std::holds_alternative<Diagnostic>(result),
              "F16/F64 conversion combinations fail explicitly");
    }
}

void test_f2i() {
    const auto build = [](std::uint64_t destination, Rounding rounding) {
        RecordContext context;
        context.current = DecodedInstruction{};
        context.current.pc = 8;
        context.current.name = InstructionName::f2i;
        context.current.predicate.enabled = true;
        context.current.numeric = f32_mode(rounding);
        add_field(context.current, DecodedField::source_type, 2);
        add_field(context.current, DecodedField::destination_type, destination);
        add_source(context.current, SourceRole::b, gpr(2), {});
        context.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        return context;
    };
    for (const auto rounding : {Rounding::nearest_even, Rounding::toward_negative,
                                 Rounding::toward_positive, Rounding::toward_zero}) {
        RecordContext flushed = build(2, rounding);
        flushed.current.numeric = f32_mode(rounding, true, true);
        flushed.current.numeric->precise = true;
        check(std::holds_alternative<std::monostate>(dispatch_instruction(flushed)), "F2I.FTZ emits");
        const auto* clamp = find(flushed, Opcode::maximum);
        check(clamp != nullptr && clamp->numeric == *flushed.current.numeric,
              "F2I unsigned extremum preserves rounding, both flush boundaries and precision");
        const auto round_opcode = rounding == Rounding::nearest_even ? Opcode::round
            : rounding == Rounding::toward_negative ? Opcode::floor
            : rounding == Rounding::toward_positive ? Opcode::ceiling : Opcode::truncate;
        const auto* rounded = find(flushed, round_opcode);
        check(rounded != nullptr && rounded->numeric == *flushed.current.numeric,
              "F2I applies input flushing at the integral rounding boundary");
    }
    {
        RecordContext rounded = build(6, Rounding::nearest_even); // S32.
        (void)dispatch_instruction(rounded);
        check(find(rounded, Opcode::round) != nullptr,
              "F2I round-to-nearest emits the round operation");
        check(find(rounded, Opcode::maximum) == nullptr,
              "Signed F2I has no unsigned lower clamp");
        check(find(rounded, Opcode::clamp) == nullptr,
              "32-bit F2I has no small-destination clamp");
        const RecordedOperation* conversion = find(rounded, Opcode::convert);
        check(conversion != nullptr, "F2I converts explicitly");
        if (conversion) {
            const auto* info = std::get_if<ConversionInfo>(&conversion->info);
            check(info != nullptr && info->source == ScalarType::f32 &&
                      info->destination == ScalarType::s32,
                  "F2I conversion metadata names the integer destination");
        }
    }
    {
        RecordContext truncated = build(2, Rounding::toward_zero); // U32.
        (void)dispatch_instruction(truncated);
        check(find(truncated, Opcode::truncate) != nullptr,
              "F2I truncation follows the selector");
        check(find(truncated, Opcode::maximum) != nullptr,
              "Unsigned F2I clamps negatives before conversion");
    }
    {
        RecordContext small = build(5, Rounding::nearest_even); // S16.
        (void)dispatch_instruction(small);
        const RecordedOperation* clamp = find(small, Opcode::clamp);
        check(clamp != nullptr && holds_literal(clamp->inputs[1], 0xFFFF8000U) &&
                  holds_literal(clamp->inputs[2], 0x7FFFU),
              "S16 F2I clamps to the exact signed 16-bit range");
    }
    {
        RecordContext small_unsigned = build(1, Rounding::nearest_even); // U16.
        (void)dispatch_instruction(small_unsigned);
        const RecordedOperation* clamp = find(small_unsigned, Opcode::clamp);
        check(clamp != nullptr && holds_literal(clamp->inputs[1], 0) &&
                  holds_literal(clamp->inputs[2], 0xFFFFU),
              "U16 F2I clamps to the exact unsigned 16-bit range");
    }
    {
        RecordContext wide = build(3, Rounding::nearest_even); // U64.
        const Status result = dispatch_instruction(wide);
        check(std::holds_alternative<Diagnostic>(result) &&
                  std::get<Diagnostic>(result).code == DiagnosticCode::unsupported_capability,
              "64-bit F2I fails instead of dropping precision silently");
    }
}

void test_i2f() {
    const auto build = [](std::uint64_t source, std::uint64_t destination, Subword position) {
        RecordContext context;
        context.current = DecodedInstruction{};
        context.current.pc = 8;
        context.current.name = InstructionName::i2f;
        context.current.predicate.enabled = true;
        context.current.numeric = f32_mode(Rounding::nearest_even);
        add_field(context.current, DecodedField::source_type, source);
        add_field(context.current, DecodedField::destination_type, destination);
        SourceModifiers modifiers{};
        modifiers.subword = position;
        modifiers.scalar = ScalarType::u32;
        add_source(context.current, SourceRole::b, gpr(2), modifiers);
        context.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        context.current.registers.push_back(
            {RegisterRole::second_destination, 0, gpr(4), Access::write});
        return context;
    };
    {
        RecordContext word = build(2, 2, Subword::full); // U32 to F32.
        (void)dispatch_instruction(word);
        const RecordedOperation* conversion = find(word, Opcode::convert);
        check(conversion != nullptr, "I2F converts explicitly");
        if (conversion) {
            const auto* info = std::get_if<ConversionInfo>(&conversion->info);
            check(info != nullptr && info->source == ScalarType::u32 &&
                      info->destination == ScalarType::f32,
                  "Unsigned I2F conversion metadata names the unsigned source");
            check(conversion->numeric.rounding == Rounding::nearest_even,
                  "I2F conversion rounds to nearest even");
        }
        check(count(word, Opcode::bit_extract) == 0, "Word I2F does not extract");
    }
    {
        RecordContext byte = build(4, 2, Subword::byte1); // S8 at byte 1.
        (void)dispatch_instruction(byte);
        const RecordedOperation* extract = find(byte, Opcode::bit_extract);
        check(extract != nullptr && holds_literal(extract->inputs[1], 8) &&
                  holds_literal(extract->inputs[2], 8),
              "Signed byte I2F extracts eight bits at the selected byte");
        check(extract != nullptr && extract->numeric.domain == Domain::signed_integer,
              "Signed extraction keeps the signed numeric domain");
    }
    {
        RecordContext half = build(1, 3, Subword::half_high); // U16 to F64.
        (void)dispatch_instruction(half);
        check(find(half, Opcode::bit_extract) != nullptr &&
                  find(half, Opcode::unpack_word64) != nullptr,
              "U16 to F64 extracts and unpacks into the destination pair");
        const RecordedOperation* conversion = find(half, Opcode::convert);
        check(conversion != nullptr &&
                  std::get_if<ConversionInfo>(&conversion->info) != nullptr &&
                  std::get<ConversionInfo>(conversion->info).destination == ScalarType::f64,
              "I2F to F64 keeps FP64 semantics");
    }
    {
        RecordContext wide = build(3, 2, Subword::full); // U64 source.
        const Status result = dispatch_instruction(wide);
        check(std::holds_alternative<Diagnostic>(result),
              "64-bit I2F sources fail explicitly");
        RecordContext signed_wide = build(7, 2, Subword::full); // S64 source.
        const Status signed_result = dispatch_instruction(signed_wide);
        check(std::holds_alternative<Diagnostic>(signed_result),
              "Signed 64-bit I2F sources fail explicitly");
    }
}

void test_i2i() {
    const auto build = [](std::uint64_t source, std::uint64_t destination, bool saturate,
                          bool write_cc) {
        RecordContext context;
        context.current = DecodedInstruction{};
        context.current.pc = 8;
        context.current.name = InstructionName::i2i;
        context.current.predicate.enabled = true;
        context.current.numeric = {Domain::unsigned_integer, ScalarType::u32};
        add_field(context.current, DecodedField::source_type, source);
        add_field(context.current, DecodedField::destination_type, destination);
        add_field(context.current, DecodedField::write_cc, write_cc ? 1 : 0);
        add_field(context.current, DecodedField::integer_saturation, saturate ? 1 : 0);
        add_source(context.current, SourceRole::b, gpr(2), {});
        context.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        return context;
    };
    {
        RecordContext word = build(2, 2, false, false); // U32 to U32.
        (void)dispatch_instruction(word);
        check(word.operations.empty() && writes_to(word, gpr(3)),
              "Same-format I2I with no modifiers is a pure move");
    }
    {
        RecordContext narrow = build(1, 2, true, true); // U16 to U32 with saturation.
        (void)dispatch_instruction(narrow);
        const RecordedOperation* extract = find(narrow, Opcode::bit_extract);
        check(extract != nullptr && extract->numeric.domain == Domain::unsigned_integer &&
                  holds_literal(extract->inputs[2], 16),
              "U16 I2I extracts sixteen bits unsigned");
        const RecordedOperation* clamp = find(narrow, Opcode::clamp);
        check(clamp != nullptr && holds_literal(clamp->inputs[1], 0) &&
                  holds_literal(clamp->inputs[2], 0xFFFFFFFFU),
              "Saturated U16-to-U32 I2I uses the complete destination range");
        check(count(narrow, Opcode::compare) == 2,
              "I2I condition codes write zero and sign flags");
    }
    {
        RecordContext negative = build(5, 6, false, false); // S16 to S32.
        (void)dispatch_instruction(negative);
        const RecordedOperation* extract = find(negative, Opcode::bit_extract);
        check(extract != nullptr && extract->numeric.domain == Domain::signed_integer,
              "Signed I2I extraction uses the signed domain");
    }
    {
        RecordContext invalid = build(3, 2, false, false); // U64 selector is invalid.
        const Status result = dispatch_instruction(invalid);
        check(std::holds_alternative<Diagnostic>(result),
              "Invalid I2I selectors fail instead of dropping the instruction");
    }
}

void test_mufu_and_range_reduction() {
    const auto build = [](std::uint64_t function, bool saturate = false) {
        RecordContext context;
        context.current = DecodedInstruction{};
        context.current.pc = 8;
        context.current.name = InstructionName::mufu;
        context.current.predicate.enabled = true;
        context.current.numeric = f32_mode(Rounding::nearest_even, false, false, saturate);
        add_field(context.current, DecodedField::function, function);
        add_source(context.current, SourceRole::a, gpr(1), {true, false});
        context.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        return context;
    };
    struct Mapping {
        std::uint64_t selector;
        Opcode opcode;
    };
    const std::array<Mapping, 7> direct{{
        {0, Opcode::cosine},
        {1, Opcode::sine},
        {2, Opcode::exp2},
        {3, Opcode::log2},
        {4, Opcode::reciprocal},
        {5, Opcode::reciprocal_sqrt},
        {8, Opcode::square_root},
    }};
    for (const auto& mapping : direct) {
        RecordContext context = build(mapping.selector);
        const Status result = dispatch_instruction(context);
        check(std::holds_alternative<std::monostate>(result), "MUFU emits");
        const RecordedOperation* operation = find(context, mapping.opcode);
        check(operation != nullptr, "MUFU maps the selector to its function");
        check(operation != nullptr && count(context, Opcode::absolute) == 1,
              "MUFU applies the absolute modifier");
        check(operation != nullptr && std::holds_alternative<ValueId>(operation->inputs[0]) &&
                  count(context, Opcode::absolute) == 1 &&
                  holds_register(find(context, Opcode::absolute)->inputs[0], gpr(1)),
              "MUFU consumes the absolute-modified source");
    }
    {
        RecordContext saturated = build(4, true);
        (void)dispatch_instruction(saturated);
        const RecordedOperation* reciprocal = find(saturated, Opcode::reciprocal);
        check(reciprocal != nullptr && reciprocal->numeric.saturate,
              "MUFU saturation rides on the emitted function");
    }
    for (const std::uint64_t selector : {6, 7}) {
        RecordContext context = build(selector);
        (void)dispatch_instruction(context);
        check(find(context, Opcode::pack_word64) != nullptr &&
                  find(context, Opcode::unpack_word64) != nullptr,
              "MUFU 64h helpers pack and unpack the double explicitly");
        const RecordedOperation* wide =
            find(context, selector == 6 ? Opcode::reciprocal : Opcode::reciprocal_sqrt);
        check(wide != nullptr && wide->numeric.scalar == ScalarType::f64,
              "MUFU 64h helpers keep FP64 numeric semantics");
    }
    {
        RecordContext unknown = build(9);
        const Status result = dispatch_instruction(unknown);
        check(std::holds_alternative<Diagnostic>(result),
              "Unknown MUFU functions fail instead of passing through");
    }
    {
        RecordContext reduction;
        reduction.current = DecodedInstruction{};
        reduction.current.pc = 8;
        reduction.current.name = InstructionName::rro;
        reduction.current.predicate.enabled = true;
        reduction.current.numeric = f32_mode(Rounding::nearest_even);
        add_source(reduction.current, SourceRole::b, gpr(2), {false, true});
        reduction.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        const Status result = dispatch_instruction(reduction);
        check(std::holds_alternative<std::monostate>(result), "RRO emits");
        check(count(reduction, Opcode::negate) == 1 && reduction.operations.size() == 1 &&
                  writes_to(reduction, gpr(3)),
              "RRO is a modifier-preserving move");
    }
}

void test_half_arithmetic() {
    const auto build = [](InstructionName name, std::uint64_t output, SourceModifiers a_swizzle,
                          bool saturate = false) {
        RecordContext context;
        context.current = DecodedInstruction{};
        context.current.pc = 8;
        context.current.name = name;
        context.current.predicate.enabled = true;
        context.current.numeric = f32_mode(Rounding::nearest_even, false, false, saturate);
        add_field(context.current, DecodedField::half_mode, output);
        SourceModifiers half_pair{};
        half_pair.scalar = ScalarType::f16;
        half_pair.subword = Subword::full;
        add_source(context.current, SourceRole::a, gpr(1), a_swizzle);
        add_source(context.current, SourceRole::b, gpr(2), half_pair);
        context.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        return context;
    };
    const SourceModifiers f32_broadcast{false, false, Subword::full, ScalarType::f32};
    {
        RecordContext packed = build(InstructionName::hadd2, 0, f32_broadcast);
        const Status result = dispatch_instruction(packed);
        check(std::holds_alternative<std::monostate>(result), "HADD2 emits");
        check(count(packed, Opcode::unpack_half2) == 1,
              "HADD2 unpacks only the packed f16 source");
        check(count(packed, Opcode::add) == 2, "HADD2 adds both lanes in f32");
        check(find(packed, Opcode::pack_half2) != nullptr,
              "HADD2 packs the result halves");
        const RecordedOperation* add = find(packed, Opcode::add);
        check(add != nullptr && !add->numeric.saturate,
              "Unsaturated HADD2 adds carry no saturation");
    }
    {
        RecordContext saturated = build(InstructionName::hadd2, 0, f32_broadcast, true);
        (void)dispatch_instruction(saturated);
        const RecordedOperation* add = find(saturated, Opcode::add);
        check(add != nullptr && add->numeric.saturate,
              "Saturated HADD2 carries saturation per lane");
    }
    {
        RecordContext single = build(InstructionName::hadd2, 1, f32_broadcast); // F32 output.
        (void)dispatch_instruction(single);
        check(find(single, Opcode::pack_half2) == nullptr, "F32 half output skips the pack");
    }
    {
        RecordContext merged = build(InstructionName::hadd2, 2, f32_broadcast); // MrgH0.
        (void)dispatch_instruction(merged);
        check(count(merged, Opcode::unpack_half2) == 1 && count(merged, Opcode::bit_and) == 1 &&
                  count(merged, Opcode::bit_or) == 1,
              "MrgH0 preserves destination bits without converting the kept half");
        check(find(merged, Opcode::pack_half2) != nullptr,
              "MrgH0 packs only the computed half");
    }
    {
        RecordContext broadcast = build(InstructionName::hadd2, 0, f32_broadcast);
        broadcast.current.sources.clear();
        SourceModifiers h0h0{false, false, Subword::half_low, ScalarType::f16};
        add_source(broadcast.current, SourceRole::a, gpr(1), h0h0);
        SourceModifiers half_pair{};
        half_pair.scalar = ScalarType::f16;
        add_source(broadcast.current, SourceRole::b, gpr(2), half_pair);
        (void)dispatch_instruction(broadcast);
        check(count(broadcast, Opcode::unpack_half2) == 2,
              "H0H0 still decomposes the packed word");
    }
    {
        RecordContext fused = build(InstructionName::hfma2, 0, f32_broadcast);
        add_source(fused.current, SourceRole::c, gpr(4), {});
        const Status result = dispatch_instruction(fused);
        check(std::holds_alternative<std::monostate>(result), "HFMA2 emits");
        check(count(fused, Opcode::fused_multiply_add) == 2, "HFMA2 fuses both lanes");
    }
    {
        RecordContext multiplied = build(InstructionName::hmul2, 0, f32_broadcast);
        (void)dispatch_instruction(multiplied);
        check(count(multiplied, Opcode::multiply) == 2, "HMUL2 multiplies both lanes");
    }
}

void test_half_set() {
    // The decoder produces the packed-half numeric mode (scalar f16); the
    // comparisons must still run on the unpacked binary32 lanes.
    const NumericMode packed_mode{Domain::floating, ScalarType::f16, Rounding::nearest_even};
    const auto build = [&](std::uint64_t boolean_output) {
        RecordContext context;
        context.current = DecodedInstruction{};
        context.current.pc = 8;
        context.current.name = InstructionName::hset2;
        context.current.predicate.enabled = true;
        context.current.numeric = packed_mode;
        add_field(context.current, DecodedField::condition, 1);
        add_field(context.current, DecodedField::boolean_operation, 0);
        add_field(context.current, DecodedField::half_mode, boolean_output);
        SourceModifiers half_pair{};
        half_pair.scalar = ScalarType::f16;
        add_source(context.current, SourceRole::a, gpr(1), half_pair);
        add_source(context.current, SourceRole::b, gpr(2), half_pair);
        add_source(context.current, SourceRole::predicate, pred_reg(1), {});
        context.current.registers.push_back(
            {RegisterRole::destination, 0, gpr(3), Access::write});
        return context;
    };
    {
        RecordContext words = build(0);
        const Status result = dispatch_instruction(words);
        check(std::holds_alternative<std::monostate>(result), "HSET2 emits");
        check(count(words, Opcode::compare) == 2, "HSET2 compares both lanes");
        for (const auto& operation : words.operations) {
            if (operation.opcode != Opcode::compare) continue;
            check(operation.numeric.scalar == ScalarType::f32,
                  "HSET2 compares unpacked lanes in binary32");
        }
        const RecordedOperation* low_mask = find(words, Opcode::select);
        check(low_mask != nullptr && holds_literal(low_mask->inputs[1], 0xFFFFU) &&
                  holds_literal(low_mask->inputs[2], 0),
              "HSET2 integer lanes select all-ones half masks");
        check(find(words, Opcode::bit_and) != nullptr &&
                  find(words, Opcode::shift_left) != nullptr &&
                  find(words, Opcode::bit_or) != nullptr,
              "HSET2 packs integer booleans into both halves");
    }
    {
        RecordContext floats = build(1);
        (void)dispatch_instruction(floats);
        check(count(floats, Opcode::select) == 2 && find(floats, Opcode::pack_half2) != nullptr,
              "HSET2 boolean floats pack 1.0/0.0 halves");
    }
    {
        RecordContext predicates;
        predicates.current = DecodedInstruction{};
        predicates.current.pc = 8;
        predicates.current.name = InstructionName::hsetp2;
        predicates.current.predicate.enabled = true;
        predicates.current.numeric = packed_mode;
        add_field(predicates.current, DecodedField::condition, 1);
        add_field(predicates.current, DecodedField::boolean_operation, 3);
        add_field(predicates.current, DecodedField::half_mode, 1); // Lane AND.
        SourceModifiers half_pair{};
        half_pair.scalar = ScalarType::f16;
        add_source(predicates.current, SourceRole::a, gpr(1), half_pair);
        add_source(predicates.current, SourceRole::b, gpr(2), half_pair);
        add_source(predicates.current, SourceRole::predicate, pred_reg(0), {});
        predicates.current.registers.push_back(
            {RegisterRole::predicate_destination, 0, pred_reg(4), Access::write});
        predicates.current.registers.push_back(
            {RegisterRole::second_predicate_destination, 0, pred_reg(5), Access::write});
        const Status result = dispatch_instruction(predicates);
        check(std::holds_alternative<std::monostate>(result), "HSETP2 emits");
        check(count(predicates, Opcode::logical_and) == 1 &&
                  count(predicates, Opcode::logical_not) == 1,
              "HSETP2 lane-AND combines the lanes and inverts for the second destination");
        for (const auto& operation : predicates.operations) {
            if (operation.opcode != Opcode::compare) continue;
            check(operation.numeric.scalar == ScalarType::f32,
                  "HSETP2 compares unpacked lanes in binary32");
        }
        check(writes_to(predicates, pred_reg(4)) && writes_to(predicates, pred_reg(5)),
              "HSETP2 writes both predicate destinations");
    }
}

// HSETP2 through the production decoder: every comparison selector, the half
// swizzles, source abs/neg, the predicate combination and the lane-AND mode,
// each compared on the unpacked binary32 lanes with the decoded modifiers
// preserved. The guest fields sit at: complement predicate 0..2, primary
// predicate 3..5, sources 8..15/20..27, B swizzle 28..29, absB 30, negB 31,
// FComp 35..38, predicate input 39..41, negA 43, absA 44, Bop 45..46,
// A swizzle 47..48, HAnd 49, flush 6.
void test_half_set_predicate() {
    const auto all_of = [](const RecordContext& context, Opcode opcode) {
        std::vector<const RecordedOperation*> found;
        for (const auto& operation : context.operations) {
            if (operation.opcode == opcode) found.push_back(&operation);
        }
        return found;
    };
    const auto lane_value = [](const Operand& operand) -> const ValueId* {
        return std::get_if<ValueId>(&operand);
    };
    const auto build = [&](const char* label, std::uint64_t condition, std::uint64_t a_swizzle,
                           std::uint64_t modifiers, std::uint64_t lane_and,
                           std::uint64_t boolean_operation, RecordContext& context) {
        auto word = opcode_pattern("0101110100100");
        word = set_bits(word, 3, 3, 4);  // Primary destination P4.
        word = set_bits(word, 0, 3, 2);  // Complement destination P2.
        word = set_bits(word, 8, 8, 1);  // Source A = R1.
        word = set_bits(word, 20, 8, 2); // Source B = R2.
        word = set_bits(word, 35, 4, condition);
        word = set_bits(word, 39, 3, 7); // P7 (true) predicate input.
        word = set_bits(word, 45, 2, boolean_operation);
        word = set_bits(word, 47, 2, a_swizzle);
        word = set_bits(word, 49, 1, lane_and);
        word |= modifiers;
        return decode_and_emit(word, label, context);
    };
    // FComp selectors 0..15: F, LT, EQ, LE, GT, NE, GE, NUM, NAN, and the
    // unordered forms with the NaN flag set.
    constexpr std::array<Compare, 7> relations{{
        Compare::never, Compare::less, Compare::equal, Compare::less_equal,
        Compare::greater, Compare::not_equal, Compare::greater_equal,
    }};
    for (std::uint64_t selector = 0; selector != 16; ++selector) {
        RecordContext context;
        const auto label = "Encoded HSETP2 FComp " + std::to_string(selector) + " decodes and emits";
        if (!build(label.c_str(), selector, 0, 0, 0, 3, context)) continue;
        const auto compares = all_of(context, Opcode::compare);
        const auto nan_tests = all_of(context, Opcode::is_nan);
        for (const auto* operation : compares) {
            check(operation->numeric.domain == Domain::floating &&
                      operation->numeric.scalar == ScalarType::f32,
                  label + " compares unpacked binary32 lanes");
        }
        for (const auto* operation : nan_tests) {
            check(operation->numeric.scalar == ScalarType::f32,
                  label + " NaN tests read unpacked binary32 lanes");
        }
        if (selector == 0 || selector == 15) {
            check(compares.empty() && nan_tests.empty(),
                  label + " constant truth needs no comparison");
            continue;
        }
        if (selector == 7 || selector == 8) {
            // Both lanes expand through is_nan tests: four in total, one
            // logical OR joining each lane's pair, and NUM inverts the result.
            const auto ors = all_of(context, Opcode::logical_or);
            const auto nots = all_of(context, Opcode::logical_not);
            check(compares.empty() && nan_tests.size() == 4 && ors.size() == 2 &&
                      nots.size() == (selector == 7 ? 2u : 0u),
                  label + " expands through per-lane NaN tests");
            continue;
        }
        check(compares.size() == 2 && nan_tests.empty(),
              label + " compares both lanes natively");
        const auto* comparison = std::get_if<ComparisonInfo>(&compares[0]->info);
        check(comparison != nullptr && comparison->relation == relations[selector & 7] &&
                  comparison->unordered == ((selector & 8) != 0),
              label + " carries the decoded relation and unordered flag");
    }
    // Half swizzle selectors as decoded: 0 F16 packed pair, 1 F32 broadcast,
    // 2 H0H0, 3 H1H1. The packed forms unpack their word; the broadcast feeds
    // the register word to both lanes without a conversion.
    const auto unpack_lanes = [&](RecordContext& context, std::size_t unpack_index) {
        const auto unpacks = all_of(context, Opcode::unpack_half2);
        return unpacks.size() > unpack_index ? &unpacks[unpack_index]->outputs : nullptr;
    };
    {
        RecordContext context;
        if (build("Encoded HSETP2 F16 pair unpacks", 2, 0, 0, 0, 3, context)) {
            check(count(context, Opcode::unpack_half2) == 2,
                  "F16 swizzle unpacks both packed sources");
            const auto* unpack = unpack_lanes(context, 0);
            check(unpack != nullptr && unpack->size() == 2,
                  "F16 unpack yields both half lanes");
        }
    }
    {
        RecordContext context;
        if (build("Encoded HSETP2 F32 broadcast", 2, 1, 0, 0, 3, context)) {
            check(count(context, Opcode::unpack_half2) == 1,
                  "F32 swizzle broadcasts the word without unpacking");
            const auto compares = all_of(context, Opcode::compare);
            check(compares.size() == 2 && holds_register(compares[0]->inputs[0], gpr(1)) &&
                      holds_register(compares[1]->inputs[0], gpr(1)),
                  "F32 swizzle feeds both lanes from the full word");
        }
    }
    {
        RecordContext context;
        if (build("Encoded HSETP2 H0H0", 2, 2, 0, 0, 3, context)) {
            const auto* lanes = unpack_lanes(context, 0);
            const auto compares = all_of(context, Opcode::compare);
            check(lanes != nullptr && compares.size() == 2 && lane_value(compares[0]->inputs[0]) &&
                      lane_value(compares[1]->inputs[0]) &&
                      *lane_value(compares[0]->inputs[0]) == *lane_value(compares[1]->inputs[0]) &&
                      lanes->size() == 2 &&
                      *lane_value(compares[0]->inputs[0]) == *lane_value((*lanes)[0]),
                  "H0H0 compares the low half in both lanes");
        }
    }
    {
        RecordContext context;
        if (build("Encoded HSETP2 H1H1", 2, 3, 0, 0, 3, context)) {
            const auto* lanes = unpack_lanes(context, 0);
            const auto compares = all_of(context, Opcode::compare);
            check(lanes != nullptr && compares.size() == 2 && lane_value(compares[0]->inputs[0]) &&
                      *lane_value(compares[0]->inputs[0]) == *lane_value((*lanes)[1]) &&
                      *lane_value(compares[1]->inputs[0]) == *lane_value((*lanes)[1]),
                  "H1H1 compares the high half in both lanes");
        }
    }
    // Source abs/neg applies to each unpacked lane in binary32.
    {
        RecordContext context;
        if (build("Encoded HSETP2 abs neg sources", 2, 0, (1ull << 44) | (1ull << 43), 0, 3,
                  context)) {
            check(count(context, Opcode::absolute) == 2 && count(context, Opcode::negate) == 2,
                  "HSETP2 applies abs and negate per lane");
            for (const auto& operation : context.operations) {
                if (operation.opcode != Opcode::absolute && operation.opcode != Opcode::negate)
                    continue;
                check(operation.numeric.scalar == ScalarType::f32,
                      "Half sign modifiers operate on binary32 lanes");
            }
        }
    }
    // The decoded input flush stays on the comparison boundary.
    {
        RecordContext context;
        if (build("Encoded HSETP2 flush", 2, 0, 1ull << 6, 0, 3, context)) {
            const auto compares = all_of(context, Opcode::compare);
            check(!compares.empty() && compares[0]->numeric.flush_inputs,
                  "HSETP2 preserves the decoded input flush");
        }
    }
    // Both predicate outputs: pass-through Bop writes the raw comparisons;
    // AND/OR/XOR combine with the predicate input; HAnd ANDs the lanes first
    // and writes the inverted combined result to the complement destination.
    {
        RecordContext context;
        if (build("Encoded HSETP2 pass writes", 2, 0, 0, 0, 3, context)) {
            const auto compares = all_of(context, Opcode::compare);
            check(writes_to(context, pred_reg(4)) && writes_to(context, pred_reg(2)) &&
                      count(context, Opcode::logical_and) == 0 &&
                      count(context, Opcode::logical_not) == 0,
                  "Pass-through Bop writes both raw comparisons");
            check(!context.writes.empty() && lane_value(context.writes[0].second) &&
                      lane_value(compares[0]->outputs[0]) &&
                      *lane_value(context.writes[0].second) ==
                          *lane_value(compares[0]->outputs[0]),
                  "Primary destination receives the low comparison");
        }
    }
    for (const auto& [operation_bit, opcode, name] :
         std::array<std::tuple<std::uint64_t, Opcode, const char*>, 3>{
             {{0, Opcode::logical_and, "AND"}, {1, Opcode::logical_or, "OR"},
              {2, Opcode::logical_xor, "XOR"}}}) {
        RecordContext context;
        const auto label = std::string("Encoded HSETP2 Bop ") + name + " combines";
        if (build(label.c_str(), 2, 0, 0, 0, operation_bit, context)) {
            // Both predicate outputs combine with the predicate input, and P7
            // resolves to the constant-true predicate operand.
            const auto combined = all_of(context, opcode);
            check(combined.size() == 2 &&
                      holds_literal(combined[0]->inputs[1], 1, ValueType::predicate) &&
                      holds_literal(combined[1]->inputs[1], 1, ValueType::predicate),
                  label + " with the predicate input");
        }
    }
    {
        RecordContext context;
        if (build("Encoded HSETP2 lane AND", 2, 0, 0, 1, 3, context)) {
            const auto combined = all_of(context, Opcode::logical_and);
            const auto inverted = all_of(context, Opcode::logical_not);
            check(combined.size() == 1 && inverted.size() == 1,
                  "HAnd ANDs the lane comparisons and inverts the combined result");
            check(writes_to(context, pred_reg(4)) && writes_to(context, pred_reg(2)),
                  "HAnd still writes both predicate destinations");
        }
    }
    // The immediate and constant-buffer routes run the same unpacked-lane
    // comparison with their own field positions.
    {
        auto word = opcode_pattern("0111111x0");
        word = set_bits(word, 3, 3, 4);
        word = set_bits(word, 0, 3, 2);
        word = set_bits(word, 49, 4, 2);
        word = set_bits(word, 39, 3, 7);
        word = set_bits(word, 45, 2, 3);
        RecordContext context;
        if (decode_and_emit(word, "Encoded HSETP2 immediate route decodes and emits", context)) {
            const auto compares = all_of(context, Opcode::compare);
            check(compares.size() == 2 && compares[0]->numeric.scalar == ScalarType::f32,
                  "Immediate route compares unpacked binary32 lanes");
        }
    }
    {
        auto word = opcode_pattern("0111111x1");
        word = set_bits(word, 3, 3, 4);
        word = set_bits(word, 0, 3, 2);
        word = set_bits(word, 49, 4, 2);
        word = set_bits(word, 39, 3, 7);
        word = set_bits(word, 45, 2, 3);
        RecordContext context;
        if (decode_and_emit(word, "Encoded HSETP2 constant route decodes and emits", context)) {
            const auto compares = all_of(context, Opcode::compare);
            check(compares.size() == 2 && compares[0]->numeric.scalar == ScalarType::f32,
                  "Constant route compares unpacked binary32 lanes");
        }
    }
}

void test_decoded_float_regressions() {
    // Integer FSET and HSET2 results use full-width true masks.
    {
        auto word = opcode_pattern("01011000");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 8, 1);
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 39, 3, 7);
        word = set_bits(word, 45, 2, 3);
        word = set_bits(word, 48, 4, 2);
        RecordContext context;
        if (decode_and_emit(word, "Encoded integer FSET decodes and emits", context)) {
            check(find(context, Opcode::predicate_to_mask) != nullptr,
                  "Encoded integer FSET emits a full predicate mask");
        }
    }
    {
        auto word = opcode_pattern("0101110100011");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 8, 1);
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 35, 4, 2);
        word = set_bits(word, 39, 3, 7);
        word = set_bits(word, 45, 2, 3);
        RecordContext context;
        if (decode_and_emit(word, "Encoded integer HSET2 decodes and emits", context)) {
            const RecordedOperation* lane_mask = find(context, Opcode::select);
            check(lane_mask != nullptr && holds_literal(lane_mask->inputs[1], 0xFFFFU),
                  "Encoded integer HSET2 emits all-ones half masks");
        }
    }

    // Saturating I2I clamps against the complete decoded destination type.
    struct IntegerBounds {
        std::uint64_t destination;
        std::uint32_t minimum;
        std::uint32_t maximum;
    };
    for (const auto [destination, minimum, maximum] :
         std::array<IntegerBounds, 6>{{
             {0, 0, 0xFFU}, {1, 0, 0xFFFFU}, {2, 0, 0xFFFFFFFFU},
             {4, 0xFFFFFF80U, 0x7FU}, {5, 0xFFFF8000U, 0x7FFFU},
             {6, 0x80000000U, 0x7FFFFFFFU}}}) {
        auto word = opcode_pattern("0101110011100");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 2, destination & 3U);
        word = set_bits(word, 12, 1, destination >> 2);
        word = set_bits(word, 10, 2, 2);
        word = set_bits(word, 20, 8, 2);
        word |= std::uint64_t{1} << 50;
        RecordContext context;
        if (decode_and_emit(word, "Encoded saturating I2I decodes and emits", context)) {
            const RecordedOperation* clamp = find(context, Opcode::clamp);
            check(clamp != nullptr && holds_literal(clamp->inputs[1], minimum) &&
                      holds_literal(clamp->inputs[2], maximum),
                  "Encoded saturating I2I uses its complete destination bounds");
        }
    }

    // I2F carries directed rounding to the real conversion boundary.
    {
        auto word = opcode_pattern("0101110010111");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 2, 2);
        word = set_bits(word, 10, 2, 2);
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 39, 2, 3);
        RecordContext context;
        if (decode_and_emit(word, "Encoded RZ I2F decodes and emits", context)) {
            const RecordedOperation* conversion = find(context, Opcode::convert);
            check(conversion != nullptr &&
                      conversion->numeric.rounding == Rounding::toward_zero,
                  "Encoded RZ I2F preserves rounding on the conversion");
        }
    }
    {
        auto word = opcode_pattern("0101110010101");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 2, 1);
        word = set_bits(word, 10, 2, 2);
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 39, 2, 3);
        RecordContext context;
        if (decode_and_emit(word, "Encoded RZ F32-to-F16 decodes and emits", context)) {
            const RecordedOperation* pack = find(context, Opcode::pack_half2);
            const auto* info = pack == nullptr ? nullptr :
                std::get_if<ConversionInfo>(&pack->info);
            check(pack != nullptr && info != nullptr &&
                      info->source == ScalarType::f32 &&
                      info->destination == ScalarType::f16 &&
                      pack->numeric.rounding == Rounding::toward_zero,
                  "Encoded RZ F32-to-F16 preserves rounding on the half pack");
        }
    }

    // FMZ retains its zero-product selection for multiply and fused multiply-add.
    {
        auto word = opcode_pattern("0101110001101");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 8, 1);
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 39, 2, 3);
        word = set_bits(word, 44, 2, 2);
        RecordContext context;
        if (decode_and_emit(word, "Encoded FMUL.FMZ decodes and emits", context)) {
            const RecordedOperation* selected = find(context, Opcode::select);
            const RecordedOperation* comparison = find(context, Opcode::compare);
            const RecordedOperation* multiply = find(context, Opcode::multiply);
            check(count(context, Opcode::compare) == 2 &&
                      find(context, Opcode::logical_or) != nullptr && selected != nullptr &&
                      holds_literal(selected->inputs[1], 0),
                  "Encoded FMUL.FMZ selects zero when either multiplicand is zero");
            check(comparison != nullptr && multiply != nullptr &&
                      comparison->numeric.rounding == Rounding::nearest_even &&
                      multiply->numeric.rounding == Rounding::toward_zero,
                  "FMZ zero tests do not inherit the multiply rounding direction");
        }
    }
    {
        auto word = opcode_pattern("010110011");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 8, 1);
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 39, 8, 4);
        word = set_bits(word, 51, 2, 3);
        word = set_bits(word, 53, 2, 2);
        RecordContext context;
        if (decode_and_emit(word, "Encoded FFMA.FMZ decodes and emits", context)) {
            const RecordedOperation* selected = find(context, Opcode::select);
            const RecordedOperation* boundary = find(context, Opcode::convert);
            const auto* info = boundary == nullptr ? nullptr :
                std::get_if<ConversionInfo>(&boundary->info);
            check(boundary != nullptr && info != nullptr &&
                      info->source == ScalarType::f32 && info->destination == ScalarType::f32 &&
                      boundary->numeric.flush_inputs && boundary->numeric.flush_output,
                  "Encoded FFMA.FMZ flushes the selected addend at the operation boundary");
            check(selected != nullptr &&
                      std::holds_alternative<ValueId>(selected->inputs[1]),
                  "Encoded FFMA.FMZ selects the flushed addend for a zero product");
        }
    }
    {
        auto word = opcode_pattern("0101110001101");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 8, 1);
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 44, 2, 2);
        word |= std::uint64_t{1} << 50;
        RecordContext context;
        if (decode_and_emit(word, "Encoded saturated FMUL.FMZ decodes and emits", context)) {
            const RecordedOperation* multiply = find(context, Opcode::multiply);
            check(multiply != nullptr && multiply->numeric.saturate &&
                      find(context, Opcode::select) == nullptr,
                  "Saturated FMZ remains on the saturation path");
        }
    }

    // Same-size pass-through F2F still materializes the FTZ boundary.
    {
        auto word = opcode_pattern("0101110010101");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 2, 2);
        word = set_bits(word, 10, 2, 2);
        word = set_bits(word, 20, 8, 2);
        word |= std::uint64_t{1} << 44;
        RecordContext context;
        if (decode_and_emit(word, "Encoded FTZ F32 pass conversion decodes and emits", context)) {
            const RecordedOperation* conversion = find(context, Opcode::convert);
            const auto* info = conversion == nullptr ? nullptr :
                std::get_if<ConversionInfo>(&conversion->info);
            check(conversion != nullptr && info != nullptr &&
                      info->source == ScalarType::f32 && info->destination == ScalarType::f32 &&
                      conversion->numeric.rounding == Rounding::nearest_even &&
                      conversion->numeric.flush_inputs && conversion->numeric.flush_output,
                  "Encoded FTZ F32 pass conversion preserves its flush boundary");
        }
    }
}

// Shipped-stage regression for the same-type F2F integral-rounding selectors.
// Every decoded selector 4..7 becomes exactly one directed integral operation
// carrying the guest rounding (never saturation), followed by the saturation
// boundary when asked; no emitted floating operation may carry the neutral
// no-rounding tag that the frozen validator rejects.
void test_f2f_integral_decoded() {
    const auto encoded_f2f = [](std::uint64_t selector, bool flush, bool saturate,
                                bool absolute, bool negate) {
        auto word = opcode_pattern("0101110010101");
        word = set_bits(word, 0, 8, 3);   // Rd = R3.
        word = set_bits(word, 20, 8, 2);  // Rb = R2.
        word = set_bits(word, 8, 2, 2);   // Destination F32.
        word = set_bits(word, 10, 2, 2);  // Source F32.
        word = set_bits(word, 39, 2, selector & 3);
        word = set_bits(word, 42, 1, selector >> 2);
        word = set_bits(word, 44, 1, flush ? 1 : 0);
        word = set_bits(word, 50, 1, saturate ? 1 : 0);
        word = set_bits(word, 49, 1, absolute ? 1 : 0);
        word = set_bits(word, 45, 1, negate ? 1 : 0);
        return word;
    };
    for (const std::uint64_t selector : {std::uint64_t{4}, std::uint64_t{5},
                                         std::uint64_t{6}, std::uint64_t{7}}) {
        const Rounding rounding = selector == 4   ? Rounding::nearest_even
                                  : selector == 5 ? Rounding::toward_negative
                                  : selector == 6 ? Rounding::toward_positive
                                                  : Rounding::toward_zero;
        const Opcode integral_opcode =
            rounding == Rounding::nearest_even     ? Opcode::round
            : rounding == Rounding::toward_negative ? Opcode::floor
            : rounding == Rounding::toward_positive ? Opcode::ceiling
                                                    : Opcode::truncate;
        for (const bool saturate : {false, true}) {
            for (const bool flush : {false, true}) {
                for (const bool absolute : {false, true}) {
                    for (const bool negate : {false, true}) {
                        const std::string label =
                            "selector " + std::to_string(selector) +
                            (saturate ? " sat" : "") + (flush ? " ftz" : "") +
                            (absolute ? " abs" : "") + (negate ? " neg" : "");
                        RecordContext context;
                        if (!decode_and_emit(encoded_f2f(selector, flush, saturate,
                                                         absolute, negate),
                                             "Encoded F2F " + label + " decodes and emits",
                                             context)) {
                            continue;
                        }
                        for (const auto& operation : context.operations) {
                            check(!(operation.numeric.domain == Domain::floating &&
                                    operation.numeric.rounding == Rounding::not_applicable),
                                  "F2F " + label + " keeps an explicit rounding everywhere");
                        }
                        check(count(context, Opcode::round) + count(context, Opcode::floor) +
                                  count(context, Opcode::ceiling) +
                                  count(context, Opcode::truncate) == 1,
                              "F2F " + label + " emits exactly one integral rounding");
                        const RecordedOperation* integral = find(context, integral_opcode);
                        check(integral != nullptr && integral->numeric.rounding == rounding,
                              "F2F " + label + " follows the decoded direction");
                        check(integral == nullptr || !integral->numeric.saturate,
                              "F2F " + label + " rounding op carries no saturation");
                        check(integral == nullptr ||
                                  (integral->numeric.flush_inputs == flush &&
                                   integral->numeric.flush_output == flush),
                              "F2F " + label + " keeps its decoded flush boundary");
                        const RecordedOperation* conversion = find(context, Opcode::convert);
                        check((conversion != nullptr) == saturate,
                              "F2F " + label + " saturation boundary is exact");
                        if (saturate && conversion != nullptr) {
                            check(conversion->numeric.saturate &&
                                      conversion->numeric.rounding == Rounding::nearest_even,
                                  "F2F " + label + " boundary rounds nothing explicitly");
                        }
                        std::size_t integral_index = context.operations.size();
                        for (std::size_t index = 0; index < context.operations.size(); ++index) {
                            if (&context.operations[index] == integral) integral_index = index;
                        }
                        if (absolute) {
                            std::size_t index = 0;
                            while (index < context.operations.size() &&
                                   context.operations[index].opcode != Opcode::absolute) ++index;
                            check(index < integral_index,
                                  "F2F " + label + " applies absolute before rounding");
                        }
                        if (negate) {
                            std::size_t index = 0;
                            while (index < context.operations.size() &&
                                   context.operations[index].opcode != Opcode::negate) ++index;
                            check(index < integral_index,
                                  "F2F " + label + " applies negation before rounding");
                        }
                    }
                }
            }
        }
    }
    // A saturating pass-through conversion keeps only the boundary, which the
    // validator accepts because it carries an explicit rounding tag.
    RecordContext passthrough;
    if (decode_and_emit(encoded_f2f(1, false, true, false, false),
                        "Encoded saturating pass-through F2F decodes and emits", passthrough)) {
        check(find(passthrough, Opcode::round) == nullptr &&
                  find(passthrough, Opcode::floor) == nullptr &&
                  find(passthrough, Opcode::ceiling) == nullptr &&
                  find(passthrough, Opcode::truncate) == nullptr,
              "Saturating pass-through F2F performs no integral rounding");
        const RecordedOperation* conversion = find(passthrough, Opcode::convert);
        check(conversion != nullptr && conversion->numeric.saturate &&
                  conversion->numeric.rounding == Rounding::nearest_even,
              "Saturating pass-through F2F boundary is a nearest-even operation");
    }
    // A same-type binary16 conversion spends the selector on the integral
    // operation; its binary16 pack is its own nearest-even boundary instead of
    // inheriting the spent direction (which no pack generator admits).
    {
        auto word = opcode_pattern("0101110010101");
        word = set_bits(word, 0, 8, 3);
        word = set_bits(word, 8, 2, 1);   // Destination F16.
        word = set_bits(word, 10, 2, 1);  // Source F16.
        word = set_bits(word, 20, 8, 2);
        word = set_bits(word, 39, 2, 1);
        word = set_bits(word, 42, 1, 1);
        RecordContext half;
        if (decode_and_emit(word, "Encoded F16 floor F2F decodes and emits", half)) {
            check(find(half, Opcode::floor) != nullptr,
                  "F16 F2F floor keeps the dedicated rounding operation");
            const RecordedOperation* pack = find(half, Opcode::pack_half2);
            check(pack != nullptr && pack->numeric.rounding == Rounding::nearest_even,
                  "F16 F2F pack rounds nearest-even after the spent selector");
        }
    }
}
} // namespace

int main() {
    test_float_arithmetic();
    test_float_multiply_and_fma();
    test_float_min_max();
    test_float_set_and_set_predicate();
    test_float_compare_select();
    test_f2f();
    test_f2f_integral_decoded();
    test_f2i();
    test_i2f();
    test_i2i();
    test_mufu_and_range_reduction();
    test_half_arithmetic();
    test_half_set();
    test_half_set_predicate();
    test_decoded_float_regressions();
    if (failures) {
        std::cerr << failures << " lane 7 float emitter checks failed\n";
        return 1;
    }
    std::cout << "Shader pipeline float emitters passed\n";
    return 0;
}
