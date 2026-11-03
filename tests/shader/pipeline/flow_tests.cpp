// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic flow-lane tests: rule aggregation, decode entry, CFG and typed
// token stack construction, and flow-control emission. All instruction words
// are independently authored; the encoding patterns mirror the published
// guest field layout.

#include "pipeline/dispatch.h"
#include "pipeline/flow/flow.h"
#include "pipeline/phases.h"
#include "pipeline/ir/emit_ir.h"

#include <array>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, std::string_view message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
bool passed(Status status) { return std::holds_alternative<std::monostate>(status); }

// ---------------------------------------------------------------------------
// Local decode rules. Lanes 1-2 have not published their tables yet, so these
// tests bind a small synthetic rule set whose mask/value pairs use the
// guest encoding field layout (predicate bits 16..18, invert bit 19,
// condition selector bits 0..4, signed 24-bit displacement bits 20..43).
constexpr std::pair<std::uint64_t, std::uint64_t> pattern(std::string_view bits) {
    std::uint64_t mask = 0;
    std::uint64_t value = 0;
    for (const auto character : bits) {
        mask <<= 1;
        value <<= 1;
        if (character == '0' || character == '1') {
            mask |= 1;
            value |= character == '1' ? std::uint64_t{1} : std::uint64_t{0};
        }
    }
    const auto shift = 64 - bits.size();
    return {mask << shift, value << shift};
}

template <InstructionName Name, bool NoPred, bool Displacement>
Result<DecodedInstruction> decode_flow_rule(Pc pc, RawInstruction raw) {
    DecodedInstruction instruction;
    instruction.pc = pc;
    instruction.raw = raw;
    instruction.name = Name;
    instruction.encoding = Encoding::none;
    instruction.predicate = InstructionPredicate{
        static_cast<std::uint8_t>(raw.bits<16, 3>()), raw.bits<19, 1>() != 0, !NoPred};
    if constexpr (Displacement) {
        const auto displacement = raw.signed_bits<20, 24>() + 8;
        instruction.fields.push_back(
            FieldValue{DecodedField::branch_displacement, static_cast<std::uint64_t>(displacement)});
    }
    instruction.fields.push_back(FieldValue{DecodedField::condition, raw.bits<0, 5>()});
    return instruction;
}

constexpr DecodeRule rule(std::string_view bits, DecodeInstruction decode) {
    return DecodeRule{pattern(bits).first, pattern(bits).second, decode};
}

constexpr std::array<DecodeRule, 12> local_rules{{
    rule("111000100100", &decode_flow_rule<InstructionName::bra, false, true>),
    rule("111000110100", &decode_flow_rule<InstructionName::brk, false, false>),
    rule("111000110101", &decode_flow_rule<InstructionName::cont, false, false>),
    rule("1111000011111", &decode_flow_rule<InstructionName::sync, false, false>),
    rule("111000110000", &decode_flow_rule<InstructionName::exit, false, false>),
    rule("111000110011", &decode_flow_rule<InstructionName::kil, false, false>),
    rule("111000110010", &decode_flow_rule<InstructionName::ret, false, false>),
    rule("111000100110", &decode_flow_rule<InstructionName::cal, true, true>),
    rule("111000101001", &decode_flow_rule<InstructionName::ssy, true, true>),
    rule("111000101010", &decode_flow_rule<InstructionName::pbk, true, true>),
    rule("111000101011", &decode_flow_rule<InstructionName::pcnt, true, true>),
    rule("0101000010110", &decode_flow_rule<InstructionName::nop, false, false>),
}};

// ---------------------------------------------------------------------------
// Instruction stream encoder. Byte 0 always holds the leading scheduling word.
// The immediate is relative to the next instruction; the +8 bias
// turns it into a displacement from the branch's own PC.
constexpr std::uint64_t displacement(std::uint64_t from, std::uint64_t to) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(to) - static_cast<std::int64_t>(from) - 8) &
           0xFFFFFF;
}
constexpr std::uint64_t flow_word(std::string_view bits, std::uint64_t from = 0, std::uint64_t to = 0,
                                  std::uint64_t predicate = 7, bool inverted = false, std::uint64_t condition = 15) {
    return pattern(bits).second | (displacement(from, to) << 20) | (predicate << 16) |
           (inverted ? (std::uint64_t{1} << 19) : std::uint64_t{0}) | condition;
}

struct Stream {
    std::vector<std::byte> bytes;
    Pc cursor{};

    void word(std::uint64_t value) {
        for (unsigned index = 0; index < 8; ++index) {
            bytes.push_back(std::byte{static_cast<std::uint8_t>((value >> (8 * index)) & 0xFF)});
        }
        cursor += 8;
    }
    void schedule(std::uint64_t value = 0) { word(value); }
    void pad() { word(0); }
    void bra(Pc from, Pc to, std::uint64_t predicate = 7, bool inverted = false, std::uint64_t condition = 15) {
        word(flow_word("111000100100", from, to, predicate, inverted, condition));
    }
    void pop(std::string_view bits, std::uint64_t predicate = 7, bool inverted = false, std::uint64_t condition = 15) {
        word(flow_word(bits, 0, 0, predicate, inverted, condition));
    }
    void push(std::string_view bits, Pc from, Pc to) { word(flow_word(bits, from, to)); }
    void exit() { pop("111000110000"); }
    void kil(std::uint64_t predicate = 7, bool inverted = false, std::uint64_t condition = 15) {
        pop("111000110011", predicate, inverted, condition);
    }
    void ret() { pop("111000110010"); }
    void cal(Pc from, Pc to) { word(flow_word("111000100110", from, to)); }
    void ssy(Pc from, Pc to) { push("111000101001", from, to); }
    void pbk(Pc from, Pc to) { push("111000101010", from, to); }
    void sync(std::uint64_t predicate = 7, bool inverted = false, std::uint64_t condition = 15) {
        pop("1111000011111", predicate, inverted, condition);
    }
    void brk(std::uint64_t predicate = 7, bool inverted = false, std::uint64_t condition = 15) {
        pop("111000110100", predicate, inverted, condition);
    }
    void nop() { word(flow_word("0101000010110")); }

    [[nodiscard]] StageImage image(Stage stage = Stage::vertex) const {
        StageImage staged;
        staged.stage = stage;
        staged.bytes = bytes;
        staged.entry_pc = 8;
        return staged;
    }
};

// ---------------------------------------------------------------------------
class NullFacts final : public FactSource {
public:
    Knowledge<FactValue> lookup(FactKey) const override { return Unknown{"No facts are provided"}; }
};

// Minimal EmitterContext recording terminators, selector assignments and
// comparisons; enough to drive the flow callbacks end to end.
class FlowTestContext final : public EmitterContext {
public:
    const DecodedProgram* program{};
    const DecodedInstruction* current{};
    FunctionId current_function{};
    BlockId current_block{};
    Limits limits_{};
    std::vector<std::optional<Terminator>> ir_terminators;
    std::vector<std::pair<ValueId, std::uint32_t>> selector_assigns;
    std::vector<std::pair<ValueId, std::uint32_t>> selector_compares;
    int discards = 0;
    std::vector<Register> register_reads;
    std::uint32_t next_value{};

    void bind(const DecodedProgram& decoded) {
        program = &decoded;
        ir_terminators.clear();
        ir_terminators.resize(decoded.blocks.size());
        selector_assigns.clear();
        selector_compares.clear();
        discards = 0;
        register_reads.clear();
        next_value = 0;
    }

    [[nodiscard]] const Terminator& terminator(BlockId block) const { return *ir_terminators[block.value]; }

    [[nodiscard]] const DecodedProgram& decoded_program() const override { return *program; }
    [[nodiscard]] const DecodedInstruction& instruction() const override { return *current; }
    [[nodiscard]] FunctionId function() const override { return current_function; }
    [[nodiscard]] BlockId block() const override { return current_block; }
    [[nodiscard]] FactReader& facts() override { return reader; }
    [[nodiscard]] const Limits& limits() const override { return limits_; }

    [[nodiscard]] Result<Operand> read(Register reg) override {
        register_reads.push_back(reg);
        if (reg == zero_register) return Operand(LiteralBits{0, ValueType::word32});
        if (reg == true_predicate) return Operand(LiteralBits{1, ValueType::predicate});
        return Operand(ValueId{next_value++});
    }
    Status write(Register, Operand) override { return std::monostate{}; }
    [[nodiscard]] Result<ValueId> temporary(ValueType) override { return ValueId{next_value++}; }
    Status assign(ValueId local, Operand operand) override {
        const auto constant = std::get_if<LiteralBits>(&operand);
        selector_assigns.emplace_back(local, constant ? static_cast<std::uint32_t>(constant->bits) : 0xFFFFFFFFu);
        return std::monostate{};
    }
    [[nodiscard]] Result<OperationId> append(Operation) override { return OperationId{next_value++}; }
    [[nodiscard]] Result<std::vector<ValueId>> emit(Opcode, NumericMode, std::span<const Operand>,
        std::span<const ValueType> outputs, OperationInfo, Effects) override {
        std::vector<ValueId> results;
        for (std::size_t index = 0; index < outputs.size(); ++index) results.push_back(ValueId{next_value++});
        return results;
    }
    [[nodiscard]] Result<Operand> unary(Opcode, Operand, NumericMode, ValueType) override {
        return Operand(ValueId{next_value++});
    }
    [[nodiscard]] Result<Operand> binary(Opcode, Operand, Operand, NumericMode, ValueType) override {
        return Operand(ValueId{next_value++});
    }
    [[nodiscard]] Result<Operand> ternary(Opcode, Operand, Operand, Operand, NumericMode, ValueType) override {
        return Operand(ValueId{next_value++});
    }
    [[nodiscard]] Result<Operand> convert(Operand, ConversionInfo, NumericMode, ValueType) override {
        return Operand(ValueId{next_value++});
    }
    [[nodiscard]] Result<Operand> compare(Operand local, Operand reference, ComparisonInfo, NumericMode) override {
        std::uint32_t constant = 0xFFFFFFFFu;
        if (const auto* literal = std::get_if<LiteralBits>(&reference)) {
            constant = static_cast<std::uint32_t>(literal->bits);
        }
        if (const auto* value = std::get_if<ValueId>(&local)) {
            selector_compares.emplace_back(*value, constant);
        }
        return Operand(ValueId{next_value++});
    }
    [[nodiscard]] Result<Operand> select(Operand, Operand, Operand, ValueType) override {
        return Operand(ValueId{next_value++});
    }
    [[nodiscard]] Result<std::vector<ValueId>> load(MemoryInfo, std::span<const Operand>, NumericMode,
        std::span<const ValueType>) override {
        return diagnose(Phase::emit_ir, DiagnosticCode::unsupported_instruction, {}, "Not used by flow tests");
    }
    Status store(MemoryInfo, std::span<const Operand>, NumericMode) override {
        return diagnose(Phase::emit_ir, DiagnosticCode::unsupported_instruction, {}, "Not used by flow tests");
    }
    [[nodiscard]] Result<std::vector<ValueId>> texture(Opcode, TextureInfo, std::span<const Operand>, NumericMode,
        std::span<const ValueType>) override {
        return diagnose(Phase::emit_ir, DiagnosticCode::unsupported_instruction, {}, "Not used by flow tests");
    }
    [[nodiscard]] Result<std::vector<ValueId>> load_io(IoInfo, std::span<const Operand>, NumericMode,
        std::span<const ValueType>) override {
        return diagnose(Phase::emit_ir, DiagnosticCode::unsupported_instruction, {}, "Not used by flow tests");
    }
    Status store_io(IoInfo, std::span<const Operand>, NumericMode) override {
        return diagnose(Phase::emit_ir, DiagnosticCode::unsupported_instruction, {}, "Not used by flow tests");
    }
    Status discard() override {
        ++discards;
        return std::monostate{};
    }
    [[nodiscard]] Result<BlockId> label(Pc pc) override {
        for (std::size_t index = 0; index < program->blocks.size(); ++index) {
            if (program->blocks[index].begin_pc == pc) return BlockId{static_cast<std::uint32_t>(index)};
        }
        return diagnose(Phase::emit_ir, DiagnosticCode::invalid_ir, {pc, std::nullopt}, "No decoded block begins at this PC");
    }
    [[nodiscard]] Result<BlockId> create_block() override {
        ir_terminators.emplace_back();
        return BlockId{static_cast<std::uint32_t>(ir_terminators.size() - 1)};
    }
    Status enter_block(BlockId block) override {
        current_block = block;
        return std::monostate{};
    }
    Status terminate(Terminator terminator) override {
        ir_terminators[current_block.value] = std::move(terminator);
        return std::monostate{};
    }
    [[nodiscard]] Result<ValueId> reconvergence_selector(BlockId consumer, Pc pop_pc) override {
        const auto key = std::pair{consumer.value, pop_pc};
        const auto known = selectors.find(key);
        if (known != selectors.end()) return known->second;
        const auto id = ValueId{next_value++};
        selectors.emplace(key, id);
        return id;
    }
    [[nodiscard]] Diagnostic error(DiagnosticCode code, std::string message, std::string modifier = {}) const override {
        auto diagnostic = diagnose(Phase::emit_ir, code,
            current ? SourceLocation{current->pc, current->name} : SourceLocation{}, std::move(message));
        diagnostic.modifier = std::move(modifier);
        return diagnostic;
    }

private:
    NullFacts source;
    FactReader reader{source};
    std::map<std::pair<std::uint32_t, std::uint32_t>, ValueId> selectors;
};

// ---------------------------------------------------------------------------
Result<DecodedProgram> decode(const Stream& stream, Limits limits = {}) {
    NullFacts source;
    FactReader facts(source);
    auto result = flow::decode_stream(stream.image(), facts, limits, local_rules);
    if (const auto* error = std::get_if<Diagnostic>(&result)) {
        std::cerr << "decode failed: " << error->message
                  << " at pc=" << (error->pc ? std::to_string(*error->pc) : std::string("-")) << '\n';
    }
    return result;
}

BlockId block_id_at(const DecodedProgram& program, Pc begin) {
    for (std::size_t index = 0; index < program.blocks.size(); ++index) {
        if (program.blocks[index].begin_pc == begin) return BlockId{static_cast<std::uint32_t>(index)};
    }
    return BlockId{};
}

const DecodedBlock& program_blocks(const DecodedProgram& program, Pc begin) {
    const auto id = block_id_at(program, begin);
    return id.valid() ? program.blocks[id.value] : program.blocks.front();
}

std::optional<TokenId> token_at(const DecodedProgram& program, Pc push_pc) {
    for (std::size_t index = 0; index < program.tokens.size(); ++index) {
        if (program.tokens[index].push_pc == push_pc) return TokenId{static_cast<std::uint32_t>(index)};
    }
    return std::nullopt;
}

bool stack_equals(const DecodedProgram& program, StackStateId state, const std::vector<TokenId>& content) {
    return state.value < program.stacks.size() && program.stacks[state.value].stack == content;
}

std::optional<Diagnostic> emit_one(
    FlowTestContext& context, const DecodedProgram& program, BlockId block, const DecodedInstruction& instruction) {
    context.program = &program;
    context.current_function = program.blocks[block.value].function;
    context.current_block = block;
    context.current = &instruction;
    const auto status = dispatch_instruction(context);
    if (const auto* error = std::get_if<Diagnostic>(&status)) return *error;
    return std::nullopt;
}

std::optional<Diagnostic> emit_block(FlowTestContext& context, const DecodedProgram& program, BlockId block) {
    context.bind(program);
    for (const auto& instruction : program.blocks[block.value].instructions) {
        if (auto error = emit_one(context, program, block, instruction)) return error;
    }
    return std::nullopt;
}

// The shared multi-alternative program: one SYNC is reached with the stacks
// [t_a] and [t_a, t_b], so it dispatches between two reconvergence targets.
Result<DecodedProgram> decode_alternatives() {
    auto stream = Stream{};
    stream.schedule();
    stream.ssy(8, 56);      // t_a, reconverges at 56
    stream.bra(16, 40, 0);  // into the inner region
    stream.bra(24, 48);     // unconditional jump over the inner push
    stream.schedule();
    stream.ssy(40, 72);     // t_b, reconverges at 72
    stream.sync();          // reached with both alternative stacks
    stream.exit();
    stream.schedule();
    stream.exit();
    return decode(stream);
}

// ---------------------------------------------------------------------------
void test_rule_aggregation() {
    check(std::holds_alternative<Diagnostic>(flow::merge_decode_rules(
              std::array<DecodeRule, 1>{DecodeRule{1, 0, nullptr}}, {})),
        "Decode rule without a callback is rejected");
    check(std::holds_alternative<Diagnostic>(flow::merge_decode_rules(
              std::array<DecodeRule, 1>{DecodeRule{0, 0, &decode_flow_rule<InstructionName::nop, false, false>}}, {})),
        "Decode rule matching every word is rejected");
    const std::array identical{rule("111000100100", &decode_flow_rule<InstructionName::bra, false, true>),
        rule("111000100100", &decode_flow_rule<InstructionName::brk, false, false>)};
    check(std::holds_alternative<Diagnostic>(flow::merge_decode_rules(identical, {})),
        "Identical decode rules overlap");
    const std::array nested{rule("111000100100", &decode_flow_rule<InstructionName::bra, false, true>),
        rule("1110001001", &decode_flow_rule<InstructionName::cal, true, true>)};
    check(std::holds_alternative<Diagnostic>(flow::merge_decode_rules(nested, {})),
        "A rule nested inside another rule's mask overlaps");
    const std::array first{rule("111000100100", &decode_flow_rule<InstructionName::bra, false, true>)};
    const std::array second{rule("111000101001", &decode_flow_rule<InstructionName::ssy, true, true>)};
    auto merged = flow::merge_decode_rules(first, second);
    check(std::holds_alternative<std::vector<DecodeRule>>(merged) &&
              std::get<std::vector<DecodeRule>>(merged).size() == 2,
        "Disjoint decode rules merge");
}

void test_decode_entry_limits() {
    check(std::holds_alternative<Diagnostic>(decode(Stream{})),
        "Entry beyond the stream is rejected");
    auto stream = Stream{};
    stream.schedule();
    stream.exit();
    auto too_large = decode(stream, [] {
        Limits limits;
        limits.code_bytes = 8;
        return limits;
    }());
    check(std::holds_alternative<Diagnostic>(too_large) &&
              std::get<Diagnostic>(too_large).code == DiagnosticCode::limit_exceeded,
        "Streams beyond the code byte limit are rejected");
    // The public decode() binds the published lane 1 and lane 2 tables, which
    // decode the EXIT-only stream into a single block.
    NullFacts source;
    FactReader facts(source);
    auto published = decode(stream.image(), facts, Limits{});
    const auto* decoded = std::get_if<DecodedProgram>(&published);
    check(decoded != nullptr && decoded->blocks.size() == 1 &&
              decoded->blocks[0].instructions.size() == 1 &&
              decoded->blocks[0].instructions[0].name == InstructionName::exit,
        "Published rule tables decode an EXIT-only stream");
}

void test_linear_and_branches() {
    auto stream = Stream{};
    stream.schedule();
    stream.exit();
    auto decoded = decode(stream);
    check(std::holds_alternative<DecodedProgram>(decoded), "A lone EXIT decodes");
    if (!std::holds_alternative<DecodedProgram>(decoded)) return;
    const auto& program = std::get<DecodedProgram>(decoded);
    check(program.functions.size() == 1 && program.blocks.size() == 1, "One function with one block");
    check(program.blocks[0].successors.empty(), "Unconditional EXIT has no successors");
    check(program.scheduling.empty(),
        "Scheduling words outside decoded blocks are never walked");
    check(program.dependencies.empty(), "Decode consumes no facts and carries no dependencies");

    // A scheduling word inside a block is skipped but stays in the interval.
    auto scheduled = Stream{};
    scheduled.schedule();
    scheduled.nop();
    scheduled.nop();
    scheduled.nop();
    scheduled.schedule(0x1234);
    scheduled.exit();
    auto scheduled_program = decode(scheduled);
    check(std::holds_alternative<DecodedProgram>(scheduled_program), "A block spanning a scheduling word decodes");
    if (std::holds_alternative<DecodedProgram>(scheduled_program)) {
        const auto& spanned = std::get<DecodedProgram>(scheduled_program);
        const auto& block = program_blocks(spanned, 8);
        check(block.instructions.size() == 4 && block.end_pc == 48,
            "Block interval includes the intervening scheduling word");
        check(spanned.scheduling.size() == 1 && spanned.scheduling[0].pc == 32 &&
                  spanned.scheduling[0].raw == 0x1234,
            "Interior scheduling word is retained with its raw value");
    }

    // A branch to a scheduling word resumes at the instruction after it.
    auto to_schedule = Stream{};
    to_schedule.schedule();
    to_schedule.bra(8, 32);
    to_schedule.nop();
    to_schedule.nop();
    to_schedule.schedule();
    to_schedule.exit();
    auto to_schedule_program = decode(to_schedule);
    check(std::holds_alternative<DecodedProgram>(to_schedule_program),
        "A branch targeting a scheduling word decodes");
    if (std::holds_alternative<DecodedProgram>(to_schedule_program)) {
        const auto& resumed = std::get<DecodedProgram>(to_schedule_program);
        const auto& head = program_blocks(resumed, 8);
        check(head.successors.size() == 1 && resumed.blocks[head.successors[0].value].begin_pc == 40,
            "The branch successor begins after the scheduling word");
    }

    // A conditional branch ending a group falls through past the next
    // scheduling word.
    auto group_end = Stream{};
    group_end.schedule();
    group_end.nop();
    group_end.nop();
    group_end.bra(24, 48, 0);
    group_end.schedule();
    group_end.nop();
    group_end.exit();
    auto group_end_program = decode(group_end);
    check(std::holds_alternative<DecodedProgram>(group_end_program),
        "A branch ending a scheduling group decodes");
    if (std::holds_alternative<DecodedProgram>(group_end_program)) {
        const auto& ended = std::get<DecodedProgram>(group_end_program);
        const auto& head = program_blocks(ended, 8);
        check(head.successors.size() == 2 && ended.blocks[head.successors[0].value].begin_pc == 40,
            "The fallthrough successor begins after the scheduling word");
    }

    // An SSY directly before a branch target ends its block and falls through.
    auto push_end = Stream{};
    push_end.schedule();
    push_end.ssy(8, 48);
    push_end.nop();          // 16: target of the backward branch below.
    push_end.bra(24, 16, 0); // 24: @P0 BRA 16.
    push_end.schedule();
    push_end.sync();         // 40: reconverges at 48.
    push_end.exit();
    auto push_end_program = decode(push_end);
    check(std::holds_alternative<DecodedProgram>(push_end_program),
        "A push that ends its block decodes");
    if (std::holds_alternative<DecodedProgram>(push_end_program)) {
        const auto& pushed = std::get<DecodedProgram>(push_end_program);
        const auto& head = program_blocks(pushed, 8);
        check(head.instructions.size() == 1 && head.successors.size() == 1 &&
                  pushed.blocks[head.successors[0].value].begin_pc == 16,
            "The push block falls through to the branch target");
    }

    // Conditional branch: fallthrough successor first, then the taken target.
    auto branch = Stream{};
    branch.schedule();
    branch.bra(8, 40, 0);
    branch.exit();
    branch.pad();
    branch.schedule();
    branch.exit();
    auto branch_program = decode(branch);
    check(std::holds_alternative<DecodedProgram>(branch_program), "A predicated branch decodes");
    if (std::holds_alternative<DecodedProgram>(branch_program)) {
        const auto& branched = std::get<DecodedProgram>(branch_program);
        const auto& head = program_blocks(branched, 8);
        check(head.successors.size() == 2, "Conditional branch has two successors");
        check(head.successors[0] == block_id_at(branched, 16) && head.successors[1] == block_id_at(branched, 40),
            "Fallthrough precedes the taken target in the successor list");
    }

    // A branch to the very next instruction collapses into one edge.
    auto next = Stream{};
    next.schedule();
    next.bra(8, 16, 0);
    next.exit();
    auto next_program = decode(next);
    check(std::holds_alternative<DecodedProgram>(next_program), "A branch to the next instruction decodes");
    if (std::holds_alternative<DecodedProgram>(next_program)) {
        const auto& degenerate = std::get<DecodedProgram>(next_program);
        check(program_blocks(degenerate, 8).successors.size() == 1,
            "Branch and fallthrough to the same block produce one edge");
    }
}

void test_token_stacks() {
    // Nested SSY/SYNC: the innermost token is consumed first.
    auto nested = Stream{};
    nested.schedule();
    nested.ssy(8, 56);
    nested.ssy(16, 48);
    nested.sync();
    nested.schedule();
    nested.sync();
    nested.sync();
    nested.exit();
    auto nested_program = decode(nested);
    check(std::holds_alternative<DecodedProgram>(nested_program), "Nested SSY/SYNC decodes");
    if (!std::holds_alternative<DecodedProgram>(nested_program)) return;
    const auto& program = std::get<DecodedProgram>(nested_program);
    check(program.tokens.size() == 2 && program.tokens[0].kind == TokenKind::sync &&
              program.tokens[1].kind == TokenKind::sync,
        "Two typed sync tokens are interned");
    const auto outer = token_at(program, 8);
    const auto inner = token_at(program, 16);
    check(outer && inner && outer->value != inner->value, "Each push instruction owns one stable token");
    const auto& head = program_blocks(program, 8);
    check(head.pushes.size() == 2, "The block records its pushes in order");
    check(head.transitions.size() == 1 && head.transitions[0].consumed == inner,
        "The first SYNC consumes the topmost (innermost) token");
    check(stack_equals(program, head.transitions[0].outgoing, {*outer}),
        "After the inner SYNC the outer token remains on the stack");
    const auto& tail = program_blocks(program, 48);
    check(tail.reconvergence.size() == 1 && tail.reconvergence[0].token == outer,
        "The second SYNC consumes the outer token");
    check(stack_equals(program, tail.transitions[0].outgoing, {}),
        "Both tokens are consumed at the reconvergence point");
    check(passed(flow::validate_decoded_cfg(program)), "Nested token program passes the CFG self-check");

    // A conditional SYNC keeps its stack on the fallthrough path.
    auto conditional = Stream{};
    conditional.schedule();
    conditional.ssy(8, 48);
    conditional.sync(0);
    conditional.sync();
    conditional.schedule();
    conditional.pad();
    conditional.exit();
    auto conditional_program = decode(conditional);
    check(std::holds_alternative<DecodedProgram>(conditional_program), "Conditional SYNC decodes");
    if (!std::holds_alternative<DecodedProgram>(conditional_program)) return;
    const auto& kept = std::get<DecodedProgram>(conditional_program);
    const auto token = token_at(kept, 8);
    const auto& pop_block = program_blocks(kept, 8);
    check(pop_block.transitions.size() == 2, "Conditional pop has a consumed and a fallthrough transition");
    check(pop_block.transitions[0].consumed == token &&
              pop_block.transitions[0].successor == block_id_at(kept, 48),
        "The consumed transition targets the push target");
    check(!pop_block.transitions[1].consumed &&
              stack_equals(kept, pop_block.transitions[1].outgoing, {*token}),
        "The fallthrough transition keeps the unconsumed stack");
    const auto& second_pop = program_blocks(kept, 24);
    check(second_pop.transitions.size() == 1 && second_pop.transitions[0].consumed == token,
        "The later SYNC consumes the token the conditional SYNC left behind");
    check(kept.tokens[token->value].consumers.size() == 2, "Both pop blocks consume the same token");

    // BRK unwinds past an SSY pushed inside the loop body.
    auto breaking = Stream{};
    breaking.schedule();
    breaking.ssy(8, 48);
    breaking.pbk(16, 48);
    breaking.ssy(24, 48);
    breaking.schedule();
    breaking.brk();
    breaking.exit();
    auto break_program = decode(breaking);
    check(std::holds_alternative<DecodedProgram>(break_program), "PBK/BRK with an interior SSY decodes");
    if (!std::holds_alternative<DecodedProgram>(break_program)) return;
    const auto& body = std::get<DecodedProgram>(break_program);
    const auto outer_token = token_at(body, 8);
    const auto break_token = token_at(body, 16);
    const auto inner_token = token_at(body, 24);
    const auto& loop_body = program_blocks(body, 8);
    check(loop_body.transitions.size() == 1 && loop_body.transitions[0].consumed == break_token,
        "BRK consumes the topmost break token");
    check(stack_equals(body, loop_body.transitions[0].outgoing, {*outer_token}),
        "BRK unwinds past the interior SSY, keeping only the outer stack");
    check(body.tokens[inner_token->value].consumers.empty(),
        "The unwound SSY token has no consumers");
    check(body.tokens[break_token->value].target == block_id_at(body, 48),
        "The break token target is the loop exit block");

    // A loop with a backedge.
    auto loop = Stream{};
    loop.schedule();
    loop.nop();
    loop.bra(16, 8, 0);
    loop.exit();
    auto loop_program = decode(loop);
    check(std::holds_alternative<DecodedProgram>(loop_program), "A loop with a backedge decodes");
    if (std::holds_alternative<DecodedProgram>(loop_program)) {
        const auto& looped = std::get<DecodedProgram>(loop_program);
        const auto& header = program_blocks(looped, 8);
        check(header.predecessors.size() == 1 &&
                  std::find(header.predecessors.begin(), header.predecessors.end(), BlockId{0}) !=
                      header.predecessors.end(),
            "The backedge makes the header its own predecessor");
        check(header.successors.size() == 2 &&
                  header.successors[1] == BlockId{0},
            "The backedge appears after the fallthrough edge");
        check(passed(flow::validate_decoded_cfg(looped)), "Loop CFG passes the self-check");
    }

    // One SYNC reached with two alternative stacks: two selector targets.
    auto alternatives_program = decode_alternatives();
    check(std::holds_alternative<DecodedProgram>(alternatives_program), "A multi-alternative SYNC decodes");
    if (!std::holds_alternative<DecodedProgram>(alternatives_program)) return;
    const auto& multi = std::get<DecodedProgram>(alternatives_program);
    const auto& join = program_blocks(multi, 48);
    check(join.incoming_stacks.size() == 2, "The join block keeps both alternative stack states");
    check(join.reconvergence.size() == 2, "The pop records one target per alternative");
    check(join.reconvergence[0].selector == 0 && join.reconvergence[1].selector == 1,
        "Selector values are unique and stable per (consumer, pop PC)");
    check(join.reconvergence[0].target != join.reconvergence[1].target,
        "Multi-target SYNC keeps distinct targets");
    check(join.successors.size() == 2, "Both alternatives carry CFG edges");
}

void test_rejections() {
    auto unmatched = Stream{};
    unmatched.schedule();
    unmatched.sync();
    auto unmatched_program = decode(unmatched);
    check(std::holds_alternative<Diagnostic>(unmatched_program) &&
              std::get<Diagnostic>(unmatched_program).code == DiagnosticCode::malformed_input,
        "SYNC without a matching SSY is rejected");

    auto runaway = Stream{};
    runaway.schedule();
    runaway.ssy(8, 40);
    runaway.bra(16, 8, 0);
    runaway.sync();
    runaway.schedule();
    runaway.exit();
    Limits small;
    small.token_depth = 4;
    auto runaway_program = decode(runaway, small);
    check(std::holds_alternative<Diagnostic>(runaway_program) &&
              std::get<Diagnostic>(runaway_program).code == DiagnosticCode::limit_exceeded,
        "A loop that grows the token stack hits the token depth limit");

    auto unknown = Stream{};
    unknown.schedule();
    unknown.word(0xFFFFFFFFFFFFFFFFull);
    check(std::holds_alternative<Diagnostic>(decode(unknown)), "An unknown encoding is rejected");

    // Trailing self-branch after an unconditional EXIT stays unreachable.
    auto junk = Stream{};
    junk.schedule();
    junk.exit();
    junk.bra(16, 16);
    auto junk_program = decode(junk);
    check(std::holds_alternative<DecodedProgram>(junk_program), "A stream with unreachable trailing junk decodes");
    if (std::holds_alternative<DecodedProgram>(junk_program)) {
        const auto& pruned = std::get<DecodedProgram>(junk_program);
        check(pruned.blocks.size() == 1 && pruned.blocks[0].instructions.size() == 1,
            "Unreachable trailing self-branch is not decoded");
    }

    // Ambiguous pops: hand-mutated alternatives must be rejected by the
    // structural self-check.
    auto alternatives_program = decode_alternatives();
    if (!std::holds_alternative<DecodedProgram>(alternatives_program)) {
        check(false, "Multi-alternative stream decodes for the ambiguity tests");
        return;
    }
    const auto& program = std::get<DecodedProgram>(alternatives_program);
    const auto join_index = block_id_at(program, 48).value;
    auto duplicated = program;
    duplicated.blocks[join_index].reconvergence.push_back(
        duplicated.blocks[join_index].reconvergence[1]);
    check(std::holds_alternative<Diagnostic>(flow::validate_decoded_cfg(duplicated)),
        "One token consumed twice by the same pop is ambiguous and rejected");
    auto colliding = program;
    colliding.blocks[join_index].reconvergence[1].selector = 0;
    check(std::holds_alternative<Diagnostic>(flow::validate_decoded_cfg(colliding)),
        "Two alternatives sharing one selector are ambiguous and rejected");
    auto severed = program;
    severed.blocks[0].successors.clear();
    check(std::holds_alternative<Diagnostic>(flow::validate_decoded_cfg(severed)),
        "Edges without transitions are rejected");
}

void test_calls_and_kil() {
    auto stream = Stream{};
    stream.schedule();
    stream.cal(8, 48);
    stream.exit();
    stream.pad();
    stream.schedule();
    stream.pad();
    stream.ret();
    auto decoded = decode(stream);
    check(std::holds_alternative<DecodedProgram>(decoded), "CAL with a callee decodes");
    if (!std::holds_alternative<DecodedProgram>(decoded)) return;
    const auto& program = std::get<DecodedProgram>(decoded);
    check(program.functions.size() == 2, "CAL discovers the callee function");
    check(program.functions[1].entry_pc == 48 && program.functions[1].callers.size() == 1 &&
              program.functions[1].callers[0] == FunctionId{0},
        "The callee records its caller");
    const auto& call_block = program_blocks(program, 8);
    check(call_block.successors.size() == 1 && call_block.successors[0] == block_id_at(program, 16),
        "The call continuation stays reachable in the caller CFG");
    const auto& callee_block = program_blocks(program, 48);
    check(callee_block.function == FunctionId{1} && callee_block.successors.empty(),
        "RET terminates the callee with no successors");
    check(passed(flow::validate_decoded_cfg(program)), "Call CFG passes the self-check");

    // KIL keeps its fallthrough: it is an effect, not a control transfer.
    auto kill = Stream{};
    kill.schedule();
    kill.kil();
    kill.exit();
    auto kill_program = decode(kill);
    check(std::holds_alternative<DecodedProgram>(kill_program), "KIL decodes");
    if (std::holds_alternative<DecodedProgram>(kill_program)) {
        const auto& killed = std::get<DecodedProgram>(kill_program);
        check(program_blocks(killed, 8).successors.size() == 1 &&
                  program_blocks(killed, 8).successors[0] == block_id_at(killed, 16),
            "Unconditional KIL still falls through to the next block");
    }
}

void test_emitters() {
    FlowTestContext context;

    auto branch = Stream{};
    branch.schedule();
    branch.bra(8, 40, 0);
    branch.exit();
    branch.pad();
    branch.schedule();
    branch.exit();
    auto branch_decoded = decode(branch);
    if (!std::holds_alternative<DecodedProgram>(branch_decoded)) {
        check(false, "Branch stream decodes for emitter tests");
        return;
    }
    const auto& branch_program = std::get<DecodedProgram>(branch_decoded);
    check(!emit_block(context, branch_program, BlockId{0}), "Predicated BRA emits");
    const auto* branch_terminator = std::get_if<Branch>(&context.terminator(BlockId{0}));
    check(branch_terminator != nullptr, "Predicated BRA becomes a Branch terminator");
    if (branch_terminator) {
        check(branch_terminator->when_true == block_id_at(branch_program, 40) &&
                  branch_terminator->when_false == block_id_at(branch_program, 16),
            "Branch targets keep taken and fallthrough blocks");
    }
    check(!emit_block(context, branch_program, block_id_at(branch_program, 16)),
        "Unconditional EXIT emits");
    check(std::holds_alternative<Return>(context.terminator(block_id_at(branch_program, 16))),
        "EXIT becomes Return");

    // SSY assigns selector constants; a single-target SYNC is a plain branch.
    auto single = Stream{};
    single.schedule();
    single.ssy(8, 40);
    single.sync();
    single.pad();
    single.schedule();
    single.exit();
    auto single_decoded = decode(single);
    if (!std::holds_alternative<DecodedProgram>(single_decoded)) {
        check(false, "Single-target stream decodes for emitter tests");
        return;
    }
    const auto& simple = std::get<DecodedProgram>(single_decoded);
    check(!emit_block(context, simple, BlockId{0}), "SSY and SYNC emit");
    check(context.selector_assigns.size() == 1 && context.selector_assigns[0].second == 0,
        "SSY assigns its selector constant into the consumer's selector local");
    check(std::holds_alternative<Jump>(context.terminator(BlockId{0})), "Single-target SYNC becomes a Jump");

    // Multi-target SYNC dispatches on the selector shared with both pushes.
    auto alternatives_decoded = decode_alternatives();
    if (!std::holds_alternative<DecodedProgram>(alternatives_decoded)) {
        check(false, "Multi-alternative stream decodes for emitter tests");
        return;
    }
    const auto& multi = std::get<DecodedProgram>(alternatives_decoded);
    const auto head = block_id_at(multi, 8);
    const auto inner = block_id_at(multi, 40);
    const auto join = block_id_at(multi, 48);
    context.bind(multi);
    check(!emit_one(context, multi, head, multi.blocks[head.value].instructions[0]),
        "Outer SSY emits");
    check(!emit_one(context, multi, inner, multi.blocks[inner.value].instructions[0]),
        "Inner SSY emits");
    check(context.selector_assigns.size() == 2 && context.selector_assigns[0].second == 1 &&
              context.selector_assigns[1].second == 0,
        "Both pushes assign their own selector constant");
    check(!emit_one(context, multi, join, multi.blocks[join.value].instructions.back()),
        "Multi-target SYNC emits");
    check(context.selector_assigns.size() == 2 && context.selector_compares.size() == 1,
        "Dispatch compares the selector once per alternative");
    check(context.selector_assigns[0].first == context.selector_compares[0].first,
        "Push and pop use the same (consumer, pop PC) selector key");
    check(context.selector_compares[0].second == 0, "The first dispatch alternative uses selector zero");
    const auto* dispatch = std::get_if<Branch>(&context.terminator(join));
    check(dispatch != nullptr, "Multi-target SYNC dispatches through a Branch");
    if (dispatch) {
        check(dispatch->when_true == block_id_at(multi, 72),
            "The first alternative targets the inner reconvergence point");
        check(std::holds_alternative<Jump>(context.terminator(dispatch->when_false)) &&
                  std::get<Jump>(context.terminator(dispatch->when_false)).target == block_id_at(multi, 56),
            "The last alternative runs unconditionally");
    }

    // KIL guards its discard with control flow.
    auto kill = Stream{};
    kill.schedule();
    kill.kil(0);
    kill.exit();
    auto kill_decoded = decode(kill);
    if (!std::holds_alternative<DecodedProgram>(kill_decoded)) {
        check(false, "KIL stream decodes for emitter tests");
        return;
    }
    const auto& discard_program = std::get<DecodedProgram>(kill_decoded);
    check(!emit_block(context, discard_program, BlockId{0}), "Predicated KIL emits");
    check(context.discards == 1, "KIL discards exactly once");
    const auto* guard = std::get_if<Branch>(&context.terminator(BlockId{0}));
    check(guard != nullptr, "KIL branches around the discard");
    if (guard) {
        check(guard->when_false == block_id_at(discard_program, 16),
            "The untaken path rejoins the fallthrough");
        check(std::holds_alternative<Jump>(context.terminator(guard->when_true)),
            "The discard block rejoins the fallthrough");
    }

    // CAL and RET report their unimplemented or invalid forms explicitly.
    auto calls = Stream{};
    calls.schedule();
    calls.cal(8, 48);
    calls.exit();
    calls.pad();
    calls.schedule();
    calls.pad();
    calls.ret();
    auto calls_decoded = decode(calls);
    if (!std::holds_alternative<DecodedProgram>(calls_decoded)) {
        check(false, "Call stream decodes for emitter tests");
        return;
    }
    const auto& call_program = std::get<DecodedProgram>(calls_decoded);
    auto call_error = emit_block(context, call_program, BlockId{0});
    check(call_error && call_error->code == DiagnosticCode::unimplemented, "CAL rejects the unimplemented call ABI");
    const auto callee = block_id_at(call_program, 48);
    check(!emit_block(context, call_program, callee), "RET in a callee emits a Return");
    check(std::holds_alternative<Return>(context.terminator(callee)), "RET becomes Return");
    context.bind(call_program);
    context.current_function = FunctionId{0};
    context.current_block = callee;
    context.current = &call_program.blocks[callee.value].instructions.back();
    check(std::holds_alternative<Diagnostic>(emit_flow_ret(context)),
        "RET in the entry function is rejected");
}

// These regressions use the published opcode tables and concrete register IR
// emitter, so decoded IDs cannot accidentally alias a mock emitted block arena.
Result<RegisterProgram> emit_real_stream(const Stream& stream) {
    NullFacts source;
    FactReader facts(source);
    auto decoded = armigo::shader::pipeline::decode(stream.image(), facts, {});
    if (const auto* error = std::get_if<Diagnostic>(&decoded)) return *error;
    const std::array bindings{EmitterBinding{InstructionName::nop,
        +[](EmitterContext&) -> Status { return std::monostate{}; }}};
    return emit_ir_with(std::get<DecodedProgram>(decoded), facts, {}, bindings);
}

void test_real_flow_emission() {
    Stream single;
    single.schedule();
    single.ssy(8, 40);
    single.sync();
    single.pad();
    single.schedule();
    single.exit();
    check(std::holds_alternative<RegisterProgram>(emit_real_stream(single)),
          "Real SSY/SYNC resolves forward decoded targets to emitted labels");

    Stream guarded;
    guarded.schedule();
    guarded.ssy(8, 40);
    guarded.word(flow_word("0101000010110", 0, 0, 0));
    guarded.sync();
    guarded.schedule();
    guarded.exit();
    check(std::holds_alternative<RegisterProgram>(emit_real_stream(guarded)),
          "Predicated instruction expansion preserves decoded pop identity");

    Stream multiple;
    multiple.schedule();
    multiple.ssy(8, 56);
    multiple.bra(16, 40, 0);
    multiple.bra(24, 48);
    multiple.schedule();
    multiple.ssy(40, 72);
    multiple.sync();
    multiple.exit();
    multiple.schedule();
    multiple.exit();
    check(std::holds_alternative<RegisterProgram>(emit_real_stream(multiple)),
          "Real multi-target reconvergence resolves both emitted labels");

    for (const auto name : {InstructionName::bra, InstructionName::exit, InstructionName::kil}) {
        Stream boundary;
        boundary.schedule();
        boundary.nop();
        boundary.nop();
        if (name == InstructionName::bra) boundary.bra(24, 48, 0);
        else if (name == InstructionName::exit) boundary.pop("111000110000", 0);
        else boundary.kil(0);
        boundary.schedule();
        boundary.nop();
        boundary.exit();
        auto emitted = emit_real_stream(boundary);
        check(std::holds_alternative<RegisterProgram>(emitted),
              "Conditional flow skips the scheduling word at fallthrough");
        if (const auto* program = std::get_if<RegisterProgram>(&emitted)) {
            const auto& entry = program->ir.blocks[program->ir.functions[0].entry.value];
            const auto* branch = std::get_if<Branch>(&entry.terminator);
            check(branch != nullptr, "Conditional boundary flow keeps a branch");
            if (branch) {
                const auto& continuation = program->ir.blocks[branch->when_false.value];
                const auto* jump = std::get_if<Jump>(&continuation.terminator);
                check(std::holds_alternative<Return>(continuation.terminator) ||
                          (jump && std::holds_alternative<Return>(program->ir.blocks[jump->target.value].terminator)),
                      "Untaken flow reaches the executable continuation and EXIT");
            }
        } else {
            std::cerr << std::get<Diagnostic>(emitted).message << '\n';
        }
    }

    Stream calls;
    calls.schedule();
    calls.cal(8, 40);
    calls.exit();
    calls.pad();
    calls.schedule();
    calls.pop("111000110010", 0, false, 10);
    calls.nop();
    calls.ret();
    NullFacts source;
    FactReader facts(source);
    auto decoded = armigo::shader::pipeline::decode(calls.image(), facts, {});
    check(std::holds_alternative<DecodedProgram>(decoded), "Real conditional RET stream decodes");
    if (const auto* program = std::get_if<DecodedProgram>(&decoded)) {
        FlowTestContext context;
        const auto callee = block_id_at(*program, 40);
        check(!emit_block(context, *program, callee), "Conditional RET emits in a decoded callee");
        check(std::find(context.register_reads.begin(), context.register_reads.end(),
                        Register{RegisterFile::predicate, 0}) != context.register_reads.end() &&
                  std::find(context.register_reads.begin(), context.register_reads.end(),
                        Register{RegisterFile::condition_code, static_cast<std::uint16_t>(ConditionCode::zero)}) !=
                        context.register_reads.end(),
              "RET evaluates both its predicate and Ccc zero flag");
        const auto* branch = std::get_if<Branch>(&context.terminator(callee));
        check(branch != nullptr, "RET predicate and condition guard return");
        if (branch) {
            check(std::holds_alternative<Return>(context.terminator(branch->when_true)),
                  "Taken RET returns from the callee");
            check(branch->when_false == block_id_at(*program, 48),
                  "Untaken RET continues in the callee");
        }
        auto fragment_callee = *program;
        fragment_callee.stage = Stage::fragment;
        fragment_callee.sph[0x48] = std::byte{15};
        check(!emit_block(context, fragment_callee, callee),
              "RET never performs fragment exports even when the stage output map is enabled");
        auto emitted = emit_ir(*program, facts, {});
        check(std::holds_alternative<Diagnostic>(emitted) &&
                  std::get<Diagnostic>(emitted).code == DiagnosticCode::unimplemented,
              "CAL remains explicitly unsupported");
    }
}


class FragmentFacts final : public FactSource {
public:
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::fragment_output_type) {
            return Known<FactValue>{IoType{key.index == 2 ? ScalarType::u32 : ScalarType::f32, 4},
                                    EvidenceOrigin{OriginKind::explicit_option, "synthetic-fragment-output"}};
        }
        return Unknown{"No synthetic fact is provided"};
    }
};

void set_sph_word(StageImage& image, std::size_t offset, std::uint32_t value) {
    for (unsigned byte = 0; byte < 4; ++byte) {
        image.sph[offset + byte] = std::byte{static_cast<std::uint8_t>(value >> (8 * byte))};
    }
}

void test_fragment_exports() {
    FragmentFacts source;
    Stream stream;
    stream.schedule();
    stream.exit();
    auto image = stream.image(Stage::fragment);
    // RT0 and RT2 are active; the skipped RT1 does not consume registers.
    set_sph_word(image, 0x48, 0xf0f);
    set_sph_word(image, 0x4c, 3);
    FactReader facts(source);
    auto decoded = armigo::shader::pipeline::decode(image, facts, {});
    check(std::holds_alternative<DecodedProgram>(decoded), "Fragment export fixture decodes");
    if (!std::holds_alternative<DecodedProgram>(decoded)) return;
    const auto& program = std::get<DecodedProgram>(decoded);
    check(std::get<DecodedProgram>(decoded).sph == image.sph, "Decoder preserves the fragment SPH output map");
    auto emitted = emit_ir(program, facts, {});
    check(std::holds_alternative<RegisterProgram>(emitted), "Fragment EXIT emits explicit outputs");
    if (const auto* result = std::get_if<RegisterProgram>(&emitted)) {
        std::vector<const Operation*> stores;
        for (const auto& operation : result->ir.operations) {
            if (operation.opcode == Opcode::store_io) stores.push_back(&operation);
        }
        check(stores.size() == 10, "EXIT exports eight color lanes, sample mask and depth");
        for (std::size_t index = 0; index < stores.size(); ++index) {
            const auto& operation = *stores[index];
            const auto& io = std::get<IoInfo>(operation.info);
            check(io.output_storage, "Fragment exports preserve output storage identity");
            const auto* reg = std::get_if<Register>(&operation.inputs[0]);
            check(reg && reg->index == index, "Fragment output export keeps the mapped register");
            if (index < 8) {
                check(io.semantic == IoSemantic::color && io.location == (index < 4 ? 0 : 2) &&
                          io.component_mask == (1u << (index % 4)) && io.type.components == 4,
                      "Color exports retain target, lane and full interface type");
                check(io.type.scalar == (index < 4 ? ScalarType::f32 : ScalarType::u32),
                      "Color exports retain the required render target scalar type");
            } else {
                check(io.semantic == (index == 8 ? IoSemantic::sample_mask : IoSemantic::fragment_depth),
                      "Fragment sample mask and depth occupy their explicit semantics");
            }
        }
    } else {
        std::cerr << std::get<Diagnostic>(emitted).message << '\n';
    }

    Stream conditional;
    conditional.schedule();
    conditional.pop("111000110000", 0);
    conditional.exit();
    auto conditional_image = conditional.image(Stage::fragment);
    set_sph_word(conditional_image, 0x48, 15);
    FactReader conditional_facts(source);
    auto conditional_decoded = armigo::shader::pipeline::decode(conditional_image, conditional_facts, {});
    check(std::holds_alternative<DecodedProgram>(conditional_decoded), "Conditional fragment EXIT decodes");
    if (const auto* decoded_program = std::get_if<DecodedProgram>(&conditional_decoded)) {
        auto conditional_ir = emit_ir(*decoded_program, conditional_facts, {});
        check(std::holds_alternative<RegisterProgram>(conditional_ir), "Conditional fragment EXIT emits");
        if (const auto* result = std::get_if<RegisterProgram>(&conditional_ir)) {
            const auto& entry = result->ir.blocks[result->ir.functions[0].entry.value];
            check(entry.operations.empty(), "Conditional EXIT exports no values before testing its guard");
            const auto* branch = std::get_if<Branch>(&entry.terminator);
            check(branch != nullptr, "Conditional fragment EXIT branches");
            if (branch) {
                for (const auto target : {branch->when_true, branch->when_false}) {
                    const auto& block = result->ir.blocks[target.value];
                    check(block.operations.size() == 4 && std::holds_alternative<Return>(block.terminator),
                          "Each executed EXIT exports its own color registers before returning");
                }
            }
        }
    }

    auto sparse = image;
    set_sph_word(sparse, 0x48, 5);
    FactReader sparse_facts(source);
    auto sparse_decoded = armigo::shader::pipeline::decode(sparse, sparse_facts, {});
    if (const auto* decoded_program = std::get_if<DecodedProgram>(&sparse_decoded)) {
        auto unsupported = emit_ir(*decoded_program, sparse_facts, {});
        check(std::holds_alternative<Diagnostic>(unsupported) &&
                  std::get<Diagnostic>(unsupported).code == DiagnosticCode::unsupported_capability,
              "Sparse color map with depth/sample mask explicitly rejects unresolved register layout");
    } else check(false, "Sparse fragment export fixture decodes");

    auto sparse_colors = program;
    sparse_colors.sph[0x48] = std::byte{5};
    sparse_colors.sph[0x49] = std::byte{0};
    sparse_colors.sph[0x4c] = std::byte{0};
    FactReader sparse_color_facts(source);
    auto sparse_color_ir = emit_ir(sparse_colors, sparse_color_facts, {});
    check(std::holds_alternative<RegisterProgram>(sparse_color_ir),
          "Sparse color map without depth/sample exports is supported");
    if (const auto* result = std::get_if<RegisterProgram>(&sparse_color_ir)) {
        check(result->ir.operations.size() == 2 &&
                  std::get<Register>(result->ir.operations[0].inputs[0]).index == 0 &&
                  std::get<Register>(result->ir.operations[1].inputs[0]).index == 2,
              "Disabled color components retain holes in the register group");
    }
    NullFacts missing_source;
    FactReader missing_facts(missing_source);
    auto missing = emit_ir(program, missing_facts, {});
    check(std::holds_alternative<Diagnostic>(missing) &&
              std::get<Diagnostic>(missing).code == DiagnosticCode::missing_fact,
          "Fragment export requires an established render target scalar type");

    image.stage = Stage::vertex;
    FactReader vertex_facts(source);
    auto vertex_decoded = armigo::shader::pipeline::decode(image, vertex_facts, {});
    if (const auto* decoded_program = std::get_if<DecodedProgram>(&vertex_decoded)) {
        auto vertex_ir = emit_ir(*decoded_program, vertex_facts, {});
        check(std::holds_alternative<RegisterProgram>(vertex_ir) &&
                  std::get<RegisterProgram>(vertex_ir).ir.operations.empty(),
              "Vertex EXIT does not interpret fragment output map fields");
    } else check(false, "Vertex EXIT fixture decodes");
}

void test_fragment_export_backend() {
    FragmentFacts source;
    FactReader facts(source);
    Stream stream;
    for (unsigned reg = 0; reg < 10; ++reg) {
        if (stream.cursor % 32 == 0) stream.schedule();
        stream.word(pattern("000000010000").second | reg | (7ull << 16) |
            (std::uint64_t{reg < 4 || reg == 9 ? 0x3f800000u : 1u} << 20));
    }
    if (stream.cursor % 32 == 0) stream.schedule();
    stream.exit();
    auto image = stream.image(Stage::fragment);
    set_sph_word(image, 0x48, 0xf0f);
    set_sph_word(image, 0x4c, 3);
    auto decoded = armigo::shader::pipeline::decode(image, facts, {});
    check(std::holds_alternative<DecodedProgram>(decoded), "Initialized fragment exports decode");
    if (!std::holds_alternative<DecodedProgram>(decoded)) return;
    check(std::get<DecodedProgram>(decoded).sph == image.sph, "Decoder preserves the fragment SPH output map");
    auto registers = emit_ir(std::get<DecodedProgram>(decoded), facts, {});
    check(std::holds_alternative<RegisterProgram>(registers), "Initialized fragment exports emit");
    if (!std::holds_alternative<RegisterProgram>(registers)) return;
    auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(registers)), {});
    check(std::holds_alternative<SsaProgram>(ssa), "EXIT-generated exports reach SSA");
    if (!std::holds_alternative<SsaProgram>(ssa)) return;
    const MetalProfile profile{"synthetic-metal", 1, 31, 32, 16, true, false, true, false, false, true};
    auto legal = legalize(std::get<SsaProgram>(std::move(ssa)), facts, profile);
    check(std::holds_alternative<LegalProgram>(legal), "EXIT-generated exports legalize");
    if (!std::holds_alternative<LegalProgram>(legal)) return;
    const auto& value = std::get<LegalProgram>(legal);
    StructuredProgram direct{value.ir, value.requirements, value.profile,
        {BlockRegion{BlockId{0}}}, {{FunctionId{0}, RegionId{0}}}};
    auto plan = plan_resources(direct, profile);
    check(std::holds_alternative<ResourcePlan>(plan), "EXIT-generated exports plan");
    if (!std::holds_alternative<ResourcePlan>(plan)) return;
    auto artifact = emit_msl(direct, std::get<ResourcePlan>(plan), {});
    check(std::holds_alternative<MslArtifact>(artifact), "EXIT-generated color/depth/sample exports emit MSL");
    if (const auto* output = std::get_if<MslArtifact>(&artifact)) {
        check(output->source.find("[[color(0)]]") != std::string::npos &&
                  output->source.find("[[color(2)]]") != std::string::npos &&
                  output->source.find("[[sample_mask]]") != std::string::npos &&
                  output->source.find("[[depth(any)]]") != std::string::npos,
              "EXIT export reflection reaches all requested Metal output attributes");
    }
    // Also exercise general structurization; its state-copy generator is owned
    // by the independent numeric lane and is absent from this baseline.
    auto structured = structure(std::get<LegalProgram>(std::move(legal)), {});
    check(std::holds_alternative<StructuredProgram>(structured), "EXIT-generated exports structurize");
    if (const auto* program = std::get_if<StructuredProgram>(&structured))
        check(std::holds_alternative<ResourcePlan>(plan_resources(*program, profile)),
              "Structurized EXIT exports retain their resource plan");
}

} // namespace

int main() {
    test_rule_aggregation();
    test_decode_entry_limits();
    test_linear_and_branches();
    test_token_stacks();
    test_rejections();
    test_calls_and_kil();
    test_emitters();
    test_real_flow_emission();
    test_fragment_exports();
    test_fragment_export_backend();
    if (failures) return 1;
    std::cout << "Shader pipeline flow passed\n";
    return 0;
}
