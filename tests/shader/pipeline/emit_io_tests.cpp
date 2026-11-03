// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic tests for the lane 8 emitters.
// A recording emitter context checks opcode signatures, numeric modes and
// memory/IO metadata roles, then tests assert the emitted operations.

#include "pipeline/dispatch.h"

#include <iostream>
#include <map>
#include <optional>
#include <string_view>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, std::string_view label) {
    if (!condition) { std::cerr << "FAIL: " << label << '\n'; ++failures; }
}

// --- synthetic facts ---------------------------------------------------------

class TestFacts final : public FactSource {
public:
    std::map<FactKey, FactValue> values;
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (const auto found = values.find(key); found != values.end()) {
            return Known<FactValue>{found->second, {OriginKind::explicit_option, "emit-io-test"}};
        }
        return Unknown{"Test fact was not supplied"};
    }
};

Interpolation perspective{InterpolationMode::perspective, InterpolationLocation::center};
Interpolation flat_mode{InterpolationMode::flat, InterpolationLocation::center};
Interpolation linear_mode{InterpolationMode::linear, InterpolationLocation::center};

// --- recording emitter context ----------------------------------------------

struct RecordedOp {
    Opcode opcode{Opcode::copy};
    NumericMode numeric;
    std::vector<Operand> inputs;
    std::vector<ValueId> outputs;
    OperationInfo info;
};

class RecordingContext final : public EmitterContext {
public:
    RecordingContext(Stage stage, const TestFacts& facts) :
        decoded_{}, facts_reader_(facts) {
        decoded_.stage = stage;
        decoded_.limits = Limits{};
        decoded_.entry = FunctionId{0};
    }

    std::vector<RecordedOp> ops;
    std::vector<std::pair<Register, Operand>> writes;
    std::uint32_t next_value{};
    DecodedProgram decoded_;
    const DecodedInstruction* current{};
    FactReader facts_reader_;

    [[nodiscard]] const DecodedProgram& decoded_program() const override { return decoded_; }
    [[nodiscard]] const DecodedInstruction& instruction() const override { return *current; }
    [[nodiscard]] FunctionId function() const override { return FunctionId{0}; }
    [[nodiscard]] BlockId block() const override { return BlockId{0}; }
    [[nodiscard]] FactReader& facts() override { return facts_reader_; }
    [[nodiscard]] const Limits& limits() const override { return decoded_.limits; }

    [[nodiscard]] Result<Operand> read(Register reg) override {
        if (reg.file == RegisterFile::gpr && reg.index == 255) {
            return Operand{LiteralBits{0, ValueType::word32}};
        }
        if (reg.file == RegisterFile::predicate && reg.index == 7) {
            return Operand{LiteralBits{1, ValueType::predicate}};
        }
        return Operand{mint(storage_type(ScalarType::u32))};
    }
    [[nodiscard]] Status write(Register reg, Operand operand) override {
        if (reg.file == RegisterFile::gpr && reg.index == 255) return std::monostate{};
        if (reg.file == RegisterFile::predicate && reg.index == 7) return std::monostate{};
        writes.push_back({reg, std::move(operand)});
        return std::monostate{};
    }
    [[nodiscard]] Result<ValueId> temporary(ValueType type) override { return mint(type); }
    [[nodiscard]] Status assign(ValueId, Operand) override { return std::monostate{}; }
    [[nodiscard]] Result<OperationId> append(Operation) override {
        return diagnose(Phase::emit_ir, DiagnosticCode::unimplemented, {}, "append is unused here");
    }

    [[nodiscard]] Result<std::vector<ValueId>> emit(Opcode opcode, NumericMode numeric,
        std::span<const Operand> inputs, std::span<const ValueType> outputs,
        OperationInfo info, Effects) override {
        if (auto error = validate(opcode, numeric, inputs, outputs, info)) return *error;
        std::vector<ValueId> results;
        for (auto type : outputs) results.push_back(mint(type));
        ops.push_back({opcode, numeric, {inputs.begin(), inputs.end()}, results, std::move(info)});
        return results;
    }
    [[nodiscard]] Result<Operand> unary(Opcode opcode, Operand input, NumericMode numeric,
                                        ValueType type) override {
        auto emitted = emit(opcode, numeric, {&input, 1}, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return Operand{std::get<std::vector<ValueId>>(emitted).front()};
    }
    [[nodiscard]] Result<Operand> binary(Opcode opcode, Operand left, Operand right,
                                         NumericMode numeric, ValueType type) override {
        Operand inputs[] = {std::move(left), std::move(right)};
        auto emitted = emit(opcode, numeric, inputs, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return Operand{std::get<std::vector<ValueId>>(emitted).front()};
    }
    [[nodiscard]] Result<Operand> ternary(Opcode opcode, Operand first, Operand second,
                                          Operand third, NumericMode numeric,
                                          ValueType type) override {
        Operand inputs[] = {std::move(first), std::move(second), std::move(third)};
        auto emitted = emit(opcode, numeric, inputs, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return Operand{std::get<std::vector<ValueId>>(emitted).front()};
    }
    [[nodiscard]] Result<Operand> convert(Operand input, ConversionInfo conversion,
                                          NumericMode numeric, ValueType type) override {
        auto emitted = emit(Opcode::convert, numeric, {&input, 1}, {&type, 1}, conversion, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return Operand{std::get<std::vector<ValueId>>(emitted).front()};
    }
    [[nodiscard]] Result<Operand> compare(Operand left, Operand right, ComparisonInfo comparison,
                                          NumericMode numeric) override {
        Operand inputs[] = {std::move(left), std::move(right)};
        const ValueType predicate = ValueType::predicate;
        auto emitted = emit(Opcode::compare, numeric, inputs, {&predicate, 1}, comparison, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return Operand{std::get<std::vector<ValueId>>(emitted).front()};
    }
    [[nodiscard]] Result<Operand> select(Operand condition, Operand when_true, Operand when_false,
                                         ValueType type) override {
        Operand inputs[] = {std::move(condition), std::move(when_true), std::move(when_false)};
        auto emitted = emit(Opcode::select, NumericMode{}, inputs, {&type, 1}, {}, Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return Operand{std::get<std::vector<ValueId>>(emitted).front()};
    }
    [[nodiscard]] Result<std::vector<ValueId>> load(MemoryInfo info,
        std::span<const Operand> inputs, NumericMode numeric,
        std::span<const ValueType> outputs) override {
        return emit(Opcode::load, numeric, inputs, outputs, std::move(info), Effects::none);
    }
    [[nodiscard]] Status store(MemoryInfo info, std::span<const Operand> inputs,
                               NumericMode numeric) override {
        auto emitted = emit(Opcode::store, numeric, inputs, {}, std::move(info), Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return std::monostate{};
    }
    [[nodiscard]] Result<std::vector<ValueId>> load_io(IoInfo info,
        std::span<const Operand> inputs, NumericMode numeric,
        std::span<const ValueType> outputs) override {
        return emit(Opcode::load_io, numeric, inputs, outputs, std::move(info), Effects::none);
    }
    [[nodiscard]] Status store_io(IoInfo info, std::span<const Operand> inputs,
                                  NumericMode numeric) override {
        auto emitted = emit(Opcode::store_io, numeric, inputs, {}, std::move(info), Effects::none);
        if (const auto* error = std::get_if<Diagnostic>(&emitted)) return *error;
        return std::monostate{};
    }
    [[nodiscard]] Result<std::vector<ValueId>> texture(Opcode, TextureInfo,
        std::span<const Operand>, NumericMode, std::span<const ValueType>) override {
        return diagnose(Phase::emit_ir, DiagnosticCode::unimplemented, {}, "Texture is not used here");
    }
    [[nodiscard]] Status discard() override { return std::monostate{}; }

    [[nodiscard]] Result<BlockId> label(Pc) override { return BlockId{0}; }
    [[nodiscard]] Result<BlockId> create_block() override { return BlockId{0}; }
    [[nodiscard]] Status enter_block(BlockId) override { return std::monostate{}; }
    [[nodiscard]] Status terminate(Terminator) override { return std::monostate{}; }
    [[nodiscard]] Result<ValueId> reconvergence_selector(BlockId, Pc) override {
        return mint(ValueType::predicate);
    }
    [[nodiscard]] Diagnostic error(DiagnosticCode code, std::string message,
                                   std::string modifier = {}) const override {
        Diagnostic diagnostic{Phase::emit_ir, code, {}, {}, std::move(modifier), {},
                              std::move(message)};
        if (current != nullptr) {
            diagnostic.pc = current->pc;
            diagnostic.instruction = current->name;
        }
        return diagnostic;
    }

    void run(InstructionName name, const DecodedInstruction& instruction) {
        current = &instruction;
        ops.clear();
        writes.clear();
        last = std::monostate{};
        const auto emitter = instruction_emitters[static_cast<std::size_t>(name)];
        last = emitter(*this);
    }
    Status last{std::monostate{}};
    [[nodiscard]] bool ok() const { return std::holds_alternative<std::monostate>(last); }
    [[nodiscard]] const Diagnostic* error_status() const {
        return std::get_if<Diagnostic>(&last);
    }

private:
    [[nodiscard]] ValueId mint(ValueType type) {
        values_.emplace(next_value, type);
        return ValueId{next_value++};
    }
    [[nodiscard]] std::optional<Diagnostic> validate(Opcode opcode, NumericMode numeric,
        std::span<const Operand> inputs, std::span<const ValueType> outputs,
        const OperationInfo& info) {
        const auto* shape = signature(opcode);
        if (shape == nullptr) return unexpected("Unknown opcode");
        if (!shape->inputs.accepts(inputs.size()) || !shape->outputs.accepts(outputs.size())) {
            return unexpected("Arity mismatch");
        }
        if (!valid_numeric_mode(numeric)) return unexpected("Invalid numeric mode");
        switch (operand_schema(opcode)) {
        case OperandSchema::memory:
            if (auto error = validate_memory(std::get_if<MemoryInfo>(&info), inputs, opcode);
                error.has_value()) return error;
            break;
        case OperandSchema::io:
            if (auto error = validate_io(std::get_if<IoInfo>(&info), inputs, opcode);
                error.has_value()) return error;
            break;
        case OperandSchema::comparison:
            if (!std::holds_alternative<ComparisonInfo>(info)) return unexpected("Missing comparison info");
            break;
        case OperandSchema::conversion: case OperandSchema::bitcast:
        case OperandSchema::pack_half: case OperandSchema::unpack_half:
            if (!std::holds_alternative<ConversionInfo>(info)) return unexpected("Missing conversion info");
            break;
        case OperandSchema::numeric: case OperandSchema::copy: case OperandSchema::select:
        case OperandSchema::carry: case OperandSchema::shift: case OperandSchema::funnel:
        case OperandSchema::extract: case OperandSchema::insert:
        case OperandSchema::pack_word64: case OperandSchema::unpack_word64:
        case OperandSchema::predicate_logic: case OperandSchema::predicate_to_mask:
        case OperandSchema::mask_to_predicate: case OperandSchema::is_nan:
            if (info.index() != 0) return unexpected("Unexpected operation info");
            break;
        case OperandSchema::texture: case OperandSchema::call: case OperandSchema::shuffle:
        case OperandSchema::ballot: case OperandSchema::vote: case OperandSchema::none:
            break;
        }
        return std::nullopt;
    }
    // Every metadata role must reference an existing input and every input
    // must be referenced (contracts, memory/IO schema rules).
    [[nodiscard]] std::optional<Diagnostic> validate_roles(const std::vector<InputIndex>& roles,
        std::size_t count, std::string_view what) {
        std::vector<bool> used(count, false);
        for (const auto& role : roles) {
            if (role.value >= count) return unexpected(std::string(what) + " references a missing input");
            used[role.value] = true;
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (!used[index]) return unexpected(std::string(what) + " leaves an input unreferenced");
        }
        return std::nullopt;
    }
    [[nodiscard]] std::optional<Diagnostic> validate_memory(const MemoryInfo* info,
        std::span<const Operand> inputs, Opcode opcode) {
        if (info == nullptr) return unexpected("Missing memory info");
        if ((info->space == AddressSpace::constant_buffer) != info->constant_buffer_slot.has_value()) {
            return unexpected("Constant buffer slot presence does not match the space");
        }
        if (info->space != AddressSpace::constant_buffer && info->resource.has_value()) {
            return unexpected("Non-buffer spaces carry no resource before legalization");
        }
        if (opcode == Opcode::load && !info->data.empty()) return unexpected("Load carries store data");
        if (opcode == Opcode::store && info->data.empty()) return unexpected("Store carries no data");
        if (info->elements == 0 || info->element_bytes == 0) return unexpected("Empty memory access");
        std::vector<InputIndex> roles;
        if (info->dynamic_byte_offset) roles.push_back(*info->dynamic_byte_offset);
        if (info->array_index) roles.push_back(*info->array_index);
        if (info->global_address_high) roles.push_back(*info->global_address_high);
        roles.insert(roles.end(), info->data.begin(), info->data.end());
        return validate_roles(roles, inputs.size(), "Memory info");
    }
    [[nodiscard]] std::optional<Diagnostic> validate_io(const IoInfo* info,
        std::span<const Operand> inputs, Opcode opcode) {
        if (info == nullptr) return unexpected("Missing IO info");
        if (opcode == Opcode::load_io && !info->data.empty()) return unexpected("load_io carries store data");
        if (opcode == Opcode::store_io && info->data.empty()) return unexpected("store_io carries no data");
        std::vector<InputIndex> roles;
        if (info->dynamic_byte_offset) roles.push_back(*info->dynamic_byte_offset);
        if (info->vertex_index) roles.push_back(*info->vertex_index);
        if (info->sample_index) roles.push_back(*info->sample_index);
        if (info->reciprocal_w) roles.push_back(*info->reciprocal_w);
        if (info->offset_x) roles.push_back(*info->offset_x);
        if (info->offset_y) roles.push_back(*info->offset_y);
        roles.insert(roles.end(), info->data.begin(), info->data.end());
        return validate_roles(roles, inputs.size(), "IO info");
    }
    [[nodiscard]] Diagnostic unexpected(std::string message) const {
        return diagnose(Phase::emit_ir, DiagnosticCode::invalid_ir, {}, std::move(message));
    }
    std::map<std::uint32_t, ValueType> values_;
};

// --- instruction builders ----------------------------------------------------

constexpr Pc test_pc = 0x40;

[[nodiscard]] DecodedInstruction base_instruction(InstructionName name, std::uint64_t word) {
    DecodedInstruction instruction;
    instruction.pc = test_pc;
    instruction.raw = RawInstruction{word};
    instruction.name = name;
    return instruction;
}

[[nodiscard]] RegisterAccess access(RegisterRole role, std::uint16_t index, Access access_kind) {
    return RegisterAccess{role, 0, Register{RegisterFile::gpr, index}, access_kind};
}

// ALD dest, srcA, srcB, imm11, per-patch, output, size(words-1).
[[nodiscard]] DecodedInstruction ald(std::uint16_t dest, std::uint8_t src_a,
    std::uint8_t src_b, std::uint32_t offset, bool per_patch, bool output, unsigned words) {
    std::uint64_t word = dest | static_cast<std::uint64_t>(src_a) << 8 |
        static_cast<std::uint64_t>(offset) << 20 |
        (per_patch ? std::uint64_t{1} << 31 : 0) | (output ? std::uint64_t{1} << 32 : 0) |
        static_cast<std::uint64_t>(src_b) << 39 | static_cast<std::uint64_t>(words - 1) << 47;
    auto instruction = base_instruction(InstructionName::ald, word);
    instruction.registers.push_back(access(RegisterRole::destination, dest, Access::write));
    if (src_a != 255) {
        instruction.registers.push_back(access(RegisterRole::source_a, src_a, Access::read));
    }
    return instruction;
}

// AST value(D slot), srcA, imm11, per-patch, size(words-1).
[[nodiscard]] DecodedInstruction ast(std::uint16_t value, std::uint8_t src_a,
    std::uint32_t offset, bool per_patch, unsigned words) {
    std::uint64_t word = value | static_cast<std::uint64_t>(src_a) << 8 |
        static_cast<std::uint64_t>(offset) << 20 |
        (per_patch ? std::uint64_t{1} << 31 : 0) |
        static_cast<std::uint64_t>(words - 1) << 47;
    auto instruction = base_instruction(InstructionName::ast, word);
    instruction.registers.push_back(access(RegisterRole::destination, value, Access::read));
    if (src_a != 255) {
        instruction.registers.push_back(access(RegisterRole::source_a, src_a, Access::read));
    }
    return instruction;
}

// IPA dest, srcA, srcB, imm10, indexed, saturate and operation selector.
// This builder dispatches through the published resource decoder so emitter
// regressions exercise the actual decoded selector and register contracts.
[[nodiscard]] DecodedInstruction ipa(std::uint16_t dest, std::uint8_t src_a, std::uint8_t src_b,
    std::uint32_t offset, bool indexed, bool saturate, unsigned operation,
    unsigned location = 0) {
    std::uint64_t word = std::uint64_t{0xe0} << 56 | dest |
        static_cast<std::uint64_t>(src_a) << 8 |
        static_cast<std::uint64_t>(src_b) << 20 | static_cast<std::uint64_t>(offset) << 28 |
        (indexed ? std::uint64_t{1} << 38 : 0) | (saturate ? std::uint64_t{1} << 51 : 0) |
        static_cast<std::uint64_t>(location) << 52 |
        static_cast<std::uint64_t>(operation) << 54 | std::uint64_t{7} << 47;
    for (const DecodeRule& candidate : resource_decode_rules()) {
        if ((word & candidate.mask) != candidate.value) continue;
        auto decoded = candidate.decode(test_pc, RawInstruction{word});
        if (const auto* instruction = std::get_if<DecodedInstruction>(&decoded)) {
            return *instruction;
        }
        check(false, "Synthetic IPA word decodes successfully");
        break;
    }
    check(false, "Synthetic IPA word matches a resource decode rule");
    return base_instruction(InstructionName::ipa, word);
}

// LDC dest, srcA, slot, byte offset, mode, size selector.
[[nodiscard]] DecodedInstruction ldc(std::uint16_t dest, std::uint8_t src_a, std::uint8_t slot,
    std::uint32_t byte_offset, std::uint8_t mode, std::uint8_t size) {
    std::uint64_t word = dest | static_cast<std::uint64_t>(src_a) << 8 |
        static_cast<std::uint64_t>(byte_offset) << 20 | static_cast<std::uint64_t>(slot) << 36 |
        static_cast<std::uint64_t>(mode) << 44 | static_cast<std::uint64_t>(size) << 48;
    auto instruction = base_instruction(InstructionName::ldc, word);
    instruction.registers.push_back(access(RegisterRole::destination, dest, Access::write));
    return instruction;
}

// LDL/STL/LDG/STG with the shared layout; stores place the value in the D slot.
[[nodiscard]] DecodedInstruction sized_access(InstructionName name, std::uint16_t reg0,
    std::uint8_t src_a, std::uint32_t byte_offset, bool extended, std::uint8_t size) {
    std::uint64_t word = reg0 | static_cast<std::uint64_t>(src_a) << 8 |
        static_cast<std::uint64_t>(byte_offset) << 20 |
        (extended ? std::uint64_t{1} << 45 : 0) | static_cast<std::uint64_t>(size) << 48;
    auto instruction = base_instruction(name, word);
    const Access direction = name == InstructionName::stl || name == InstructionName::stg
        ? Access::read : Access::write;
    instruction.registers.push_back(access(RegisterRole::destination, reg0, direction));
    return instruction;
}

[[nodiscard]] DecodedInstruction bar(std::uint8_t operation) {
    return base_instruction(InstructionName::bar, static_cast<std::uint64_t>(operation) << 32);
}
[[nodiscard]] DecodedInstruction membar(std::uint8_t kind) {
    return base_instruction(InstructionName::membar, static_cast<std::uint64_t>(kind) << 8);
}
[[nodiscard]] DecodedInstruction depbar() {
    return base_instruction(InstructionName::depbar, 0);
}

// --- helpers -----------------------------------------------------------------

bool success(RecordingContext& context, std::string_view label) {
    if (!context.ok()) {
        check(false, label);
        if (const auto* error = context.error_status()) std::cerr << "  " << error->message << '\n';
        return false;
    }
    return true;
}

bool rejected(RecordingContext& context, DiagnosticCode code, std::string_view label) {
    const auto* error = context.error_status();
    check(error != nullptr && error->code == code, label);
    if (error != nullptr && error->code != code) std::cerr << "  " << error->message << '\n';
    return error != nullptr && error->code == code;
}

std::size_t count_op(const RecordingContext& context, Opcode opcode) {
    std::size_t count = 0;
    for (const auto& op : context.ops) {
        if (op.opcode == opcode) ++count;
    }
    return count;
}

const RecordedOp* find_op(RecordingContext& context, Opcode opcode, std::string_view label) {
    if (!success(context, label)) return nullptr;
    for (const auto& candidate : context.ops) {
        if (candidate.opcode == opcode) return &candidate;
    }
    const RecordedOp* found = nullptr;
    check(found != nullptr, label);
    return found;
}

std::uint64_t instruction_pattern(std::string_view bits) {
    std::uint64_t result{};
    for (std::size_t index = 0; index < bits.size(); ++index) {
        if (bits[index] == '1') result |= std::uint64_t{1} << (63 - index);
    }
    return result;
}

std::uint64_t encoded_ast(unsigned source, unsigned offset, unsigned words = 1) {
    return instruction_pattern("1110111111110x") | source | (255ull << 8) |
        (7ull << 16) | (std::uint64_t{offset} << 20) |
        (std::uint64_t{words - 1} << 47);
}

std::uint64_t encoded_ald(unsigned destination, unsigned offset, bool output) {
    return instruction_pattern("1110111111011x") | destination | (255ull << 8) |
        (7ull << 16) | (std::uint64_t{offset} << 20) |
        (output ? std::uint64_t{1} << 32 : 0);
}

std::uint64_t encoded_ipa(unsigned destination, unsigned offset, unsigned location = 0,
                          unsigned operation = 0) {
    return instruction_pattern("11100000xxxxxx") | destination | (255ull << 8) | (7ull << 16) |
        (255ull << 20) | (std::uint64_t{offset} << 28) | (7ull << 47) |
        (std::uint64_t{location} << 52) | (std::uint64_t{operation} << 54);
}

StageImage stage_image(Stage stage, std::initializer_list<std::uint64_t> instructions) {
    StageImage image;
    image.stage = stage;
    auto append = [&](std::uint64_t word) {
        for (unsigned byte = 0; byte < 8; ++byte) {
            image.bytes.push_back(std::byte((word >> (byte * 8)) & 0xff));
        }
    };
    for (auto instruction : instructions) {
        if (image.bytes.size() % 32 == 0) append(0);
        append(instruction);
    }
    if (image.bytes.size() % 32 == 0) append(0);
    append(instruction_pattern("111000110000") | (7ull << 16) | 15);
    return image;
}

std::optional<RegisterProgram> decoded_registers(const StageImage& image,
                                                 TestFacts& facts) {
    FactReader reader(facts);
    auto decoded = decode(image, reader, {});
    if (const auto* error = std::get_if<Diagnostic>(&decoded)) {
        check(false, error->message);
        return std::nullopt;
    }
    auto emitted = emit_ir(std::get<DecodedProgram>(std::move(decoded)), reader, {});
    if (const auto* error = std::get_if<Diagnostic>(&emitted)) {
        check(false, error->message);
        return std::nullopt;
    }
    return std::get<RegisterProgram>(std::move(emitted));
}

const Operation* program_op(const Program& program, Opcode opcode,
                            unsigned ordinal = 0) {
    for (const auto& operation : program.operations) {
        if (operation.opcode == opcode && ordinal-- == 0) return &operation;
    }
    return nullptr;
}

// --- tests -------------------------------------------------------------------

void test_registry() {
    for (auto name : {InstructionName::ald, InstructionName::ast, InstructionName::ipa,
             InstructionName::out, InstructionName::ldc, InstructionName::ldl,
             InstructionName::stl, InstructionName::lds, InstructionName::sts,
             InstructionName::ld, InstructionName::st, InstructionName::ldg,
             InstructionName::stg, InstructionName::bar, InstructionName::membar,
             InstructionName::depbar}) {
        check(instruction_emitters[static_cast<std::size_t>(name)] != nullptr,
            "Lane 8 instruction has a registered emitter");
    }
    check(instruction_emitters[static_cast<std::size_t>(InstructionName::dadd)] == nullptr,
        "Unregistered instructions stay unassigned");
}

void test_ald_direct(Stage stage) {
    TestFacts facts;
    if (stage == Stage::vertex) {
        facts.values[{FactKind::vertex_attribute_format, 0, 0, 0}] =
            AttributeFormat{VertexFormat::f32, 4};
    } else {
        for (std::uint8_t component = 0; component < 4; ++component) {
            facts.values[{FactKind::input_interpolation, 0, 0, component}] = perspective;
        }
    }
    RecordingContext context(stage, facts);
    // Generic attribute zero, one word: destination R4, byte offset 0x80.
    auto instruction = ald(4, 255, 255, 0x80, false, false, 1);
    context.run(InstructionName::ald, instruction);
    if (!success(context, "ALD generic load succeeds")) return;
    check(context.ops.size() == 1, "ALD emits one load per word");
    const auto& op = context.ops.front();
    check(op.opcode == Opcode::load_io, "ALD emits load_io");
    const auto& info = std::get<IoInfo>(op.info);
    check(info.semantic == IoSemantic::generic && info.location == 0, "ALD targets generic slot zero");
    check(info.component_mask == 1, "ALD first word reads component zero");
    check(info.type.components == 4, "ALD preserves the full generic interface width");
    check(context.writes.size() == 1 && context.writes.front().first.index == 4,
        "ALD writes the destination register");
    check(context.facts_reader_.dependencies().size() == 1,
        "ALD consumes one per-location fact");
    if (stage == Stage::vertex) {
        check(context.facts_reader_.dependencies().front().key.kind ==
                FactKind::vertex_attribute_format,
            "Vertex ALD consumes the attribute format fact");
    } else {
        check(context.facts_reader_.dependencies().front().key.kind ==
                FactKind::input_interpolation,
            "Fragment ALD consumes the interpolation fact");
    }
    // Fourth component of the same attribute, byte offset 0x8c.
    auto fourth = ald(5, 255, 255, 0x8c, false, false, 1);
    context.run(InstructionName::ald, fourth);
    if (success(context, "ALD fourth component succeeds")) {
        const auto& mask = std::get<IoInfo>(context.ops.front().info).component_mask;
        check(mask == (1u << 3), "ALD byte offset selects the component");
    }
}

void test_ald_raw_word() {
    // No vertex_attribute_format fact: the attribute register word stays raw
    // and legalize derives the declared class from the IR uses.
    TestFacts facts;
    RecordingContext context(Stage::vertex, facts);
    context.run(InstructionName::ald, ald(4, 255, 255, 0x80, false, false, 1));
    if (!success(context, "Fact-less ALD generic load succeeds")) return;
    const auto& op = context.ops.front();
    check(op.opcode == Opcode::load_io && op.numeric == NumericMode{Domain::bits, ScalarType::u32},
        "Fact-less ALD loads a raw uninterpreted word");
    check(std::get<IoInfo>(op.info).type == IoType{ScalarType::u32, 4},
        "Fact-less ALD keeps the full raw interface width");
    check(context.facts_reader_.dependencies().empty(),
        "Fact-less ALD consumes no format fact");
    // A supplied fact still selects the old interpretation path.
    facts.values[{FactKind::vertex_attribute_format, 0, 0, 0}] = AttributeFormat{VertexFormat::u32, 4};
    RecordingContext with_fact(Stage::vertex, facts);
    with_fact.run(InstructionName::ald, ald(4, 255, 255, 0x80, false, false, 1));
    if (success(with_fact, "Formatted ALD generic load succeeds")) {
        check(with_fact.ops.front().numeric == NumericMode{Domain::unsigned_integer, ScalarType::u32},
            "Supplied u32 format keeps the unsigned interpretation");
    }
}

void test_ald_ranges() {
    TestFacts facts;
    facts.values[{FactKind::vertex_attribute_format, 0, 0, 0}] = AttributeFormat{VertexFormat::f32, 4};
    RecordingContext vertex(Stage::vertex, facts);

    // Position output: four consecutive words at 0x70..0x7c.
    auto position = ald(8, 255, 255, 0x70, false, true, 4);
    vertex.run(InstructionName::ald, position);
    if (success(vertex, "ALD position output succeeds")) {
        check(vertex.ops.size() == 4, "Position vector loads four words");
        bool semantics = true;
        for (unsigned index = 0; index < 4; ++index) {
            const auto& info = std::get<IoInfo>(vertex.ops[index].info);
            semantics = semantics && info.semantic == IoSemantic::position &&
                        info.component_mask == (1u << index) && info.type.components == 4 &&
                        info.output_storage;
        }
        check(semantics, "Position words map to components in order");
        check(vertex.writes.size() == 4, "Position results reach four registers");
    }

    // Built-ins: clip distance output, vertex id, instance id.
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x2c4, false, true, 1));
    if (success(vertex, "ALD clip distance output succeeds")) {
        const auto& info = std::get<IoInfo>(vertex.ops.front().info);
        check(info.semantic == IoSemantic::clip_distance && info.location == 1,
            "Clip distance byte offset selects the clip index");
    }
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x2fc, false, false, 1));
    if (success(vertex, "ALD vertex id succeeds")) {
        const auto& operation = vertex.ops.front();
        const auto& info = std::get<IoInfo>(operation.info);
        check(info.semantic == IoSemantic::vertex_id && info.type == IoType{ScalarType::u32, 1} &&
              operation.numeric == NumericMode{Domain::unsigned_integer, ScalarType::u32},
            "Vertex id uses an unsigned word in IO and numeric metadata");
    }
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x2f8, false, false, 1));
    if (success(vertex, "ALD instance id succeeds")) {
        const auto& operation = vertex.ops.front();
        const auto& info = std::get<IoInfo>(operation.info);
        check(info.semantic == IoSemantic::instance_id && info.type == IoType{ScalarType::u32, 1} &&
              operation.numeric == NumericMode{Domain::unsigned_integer, ScalarType::u32},
            "Instance id uses an unsigned word in IO and numeric metadata");
    }

    // Invalid addresses and stage mismatches.
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x280, false, false, 1));
    check(rejected(vertex, DiagnosticCode::unsupported_capability,
        "Fixed-function color offsets are rejected"), "");
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x400, false, false, 1));
    check(rejected(vertex, DiagnosticCode::unsupported_capability,
        "Offsets beyond the attribute map are rejected"), "");
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x80, true, false, 1));
    check(rejected(vertex, DiagnosticCode::unsupported_capability,
        "Per-patch addressing is rejected outside tessellation"), "");
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x2fc, false, true, 1));
    check(rejected(vertex, DiagnosticCode::unsupported_capability,
        "Vertex id is not an output"), "");
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x070, false, false, 1));
    check(rejected(vertex, DiagnosticCode::unsupported_capability,
        "Position is not a vertex input"), "");
    vertex.run(InstructionName::ald, ald(2, 255, 255, 0x80, false, false, 4));
    // Words 1..3 leave attribute zero; they stay generic slot zero components 1..3.
    if (success(vertex, "ALD vector into attribute zero succeeds")) {
        check(vertex.ops.size() == 4, "Vector ALD emits one load per word");
    }
    // RZ destination: nothing is emitted, and the access still succeeds.
    vertex.run(InstructionName::ald, ald(255, 255, 255, 0x80, false, false, 1));
    check(vertex.ok() && vertex.writes.empty(), "RZ destination discards the load");
    // Missing format fact: the register word stays raw and legalize derives
    // the declared class from the IR uses (see vertex_class_tests).
    TestFacts empty;
    RecordingContext undeclared(Stage::vertex, empty);
    undeclared.run(InstructionName::ald, ald(2, 255, 255, 0x80, false, false, 1));
    if (success(undeclared, "Fact-less vertex generic load succeeds")) {
        const auto& raw = std::get<IoInfo>(undeclared.ops.front().info);
        check(undeclared.ops.front().numeric == NumericMode{Domain::bits, ScalarType::u32} &&
              raw.type == IoType{ScalarType::u32, 4},
            "Fact-less vertex load keeps the raw word interpretation");
    }

    // Fragment built-ins.
    TestFacts fragment_facts;
    for (std::uint8_t component = 0; component < 4; ++component) {
        fragment_facts.values[{FactKind::input_interpolation, 0, 0, component}] = perspective;
    }
    RecordingContext fragment(Stage::fragment, fragment_facts);
    fragment.run(InstructionName::ald, ald(2, 255, 255, 0x070, false, false, 4));
    if (success(fragment, "ALD fragment position succeeds")) {
        check(std::get<IoInfo>(fragment.ops.front().info).semantic == IoSemantic::fragment_coord,
            "Fragment position reads fragment coord");
    }
    fragment.run(InstructionName::ald, ald(2, 255, 255, 0x3fc, false, false, 1));
    if (success(fragment, "ALD front facing succeeds")) {
        const auto& info = std::get<IoInfo>(fragment.ops.front().info);
        check(info.semantic == IoSemantic::front_facing &&
              info.type == IoType{ScalarType::boolean, 1},
            "Front facing loads through the Boolean interface type");
        check(fragment.ops.size() == 2 &&
              fragment.ops.back().opcode == Opcode::predicate_to_mask,
            "Front facing converts its predicate to the guest word mask");
    }
    fragment.run(InstructionName::ald, ald(2, 255, 255, 0x80, false, true, 1));
    check(rejected(fragment, DiagnosticCode::unsupported_capability,
        "Generic outputs are not readable in the fragment stage"), "");
    fragment.run(InstructionName::ald, ald(2, 255, 255, 0x2fc, false, false, 1));
    check(rejected(fragment, DiagnosticCode::unsupported_capability,
        "Vertex id is not available in the fragment stage"), "");
}

void test_ald_physical() {
    TestFacts facts;
    RecordingContext context(Stage::vertex, facts);
    // Physical addressing: imm11 zero with a live address register (R2).
    auto instruction = ald(6, 2, 255, 0, false, false, 2);
    context.run(InstructionName::ald, instruction);
    if (!success(context, "Physical ALD succeeds")) return;
    check(count_op(context, Opcode::load_io) == 1, "Physical ALD emits one vector load");
    const auto* op = find_op(context, Opcode::load_io, "Physical ALD emits one vector load");
    if (op == nullptr) return;
    const auto& info = std::get<IoInfo>(op->info);
    check(info.dynamic_byte_offset.has_value(), "Physical ALD carries a dynamic byte offset");
    check(op->inputs.size() == 1, "Physical ALD references its offset input");
    check(info.type.components == 4 && info.component_mask == 3,
        "Physical ALD preserves interface width while selecting two words");
    check(context.writes.size() == 2, "Physical ALD fills both destination registers");
}

void test_ast() {
    TestFacts facts;
    facts.values[{FactKind::fragment_output_type, 0, 0, 0}] = IoType{ScalarType::f32, 4};
    RecordingContext vertex(Stage::vertex, facts);
    vertex.run(InstructionName::ast, ast(10, 255, 0x70, false, 4));
    if (success(vertex, "AST position store succeeds")) {
        check(vertex.ops.size() == 4, "Position store emits one store per word");
        const auto& info = std::get<IoInfo>(vertex.ops.front().info);
        check(info.semantic == IoSemantic::position && info.data.size() == 1,
            "Position store targets the position semantic with one data role");
    }
    vertex.run(InstructionName::ast, ast(10, 255, 0x80, false, 1));
    if (success(vertex, "AST generic store succeeds")) {
        check(std::get<IoInfo>(vertex.ops.front().info).semantic == IoSemantic::generic,
            "Generic store targets the generic semantic");
    }
    vertex.run(InstructionName::ast, ast(10, 4, 0, false, 1)); // physical: imm 0, srcA live
    if (const auto* op = find_op(vertex, Opcode::store_io, "AST physical store succeeds")) {
        const auto& info = std::get<IoInfo>(op->info);
        check(info.dynamic_byte_offset.has_value() && info.data.size() == 1,
            "Physical store keeps its dynamic byte offset");
    }
    RecordingContext fragment(Stage::fragment, facts);
    fragment.run(InstructionName::ast, ast(10, 255, 0x80, false, 1));
    if (const auto* op = find_op(fragment, Opcode::store_io, "Fragment color store succeeds")) {
        const auto& info = std::get<IoInfo>(op->info);
        check(info.semantic == IoSemantic::color,
            "Fragment generic output stores target the color semantic");
        check(facts.values.count({FactKind::fragment_output_type, 0, 0, 0}) == 1,
            "Color store consumes the fragment output type fact");
        check(fragment.facts_reader_.dependencies().front().key.kind ==
                FactKind::fragment_output_type,
            "Color store records the output type dependency");
    }
    fragment.run(InstructionName::ast, ast(10, 255, 0x70, false, 1));
    check(rejected(fragment, DiagnosticCode::unsupported_capability,
        "Position is not a fragment output"), "");
}

void test_ipa() {
    TestFacts facts;
    facts.values[{FactKind::input_interpolation, 0, 0, 0}] = flat_mode;
    RecordingContext context(Stage::fragment, facts);
    auto instruction = ipa(4, 255, 6, 0x80, false, false, 0);

    context.run(InstructionName::ipa, ipa(4, 255, 6, 0x80, false, false, 2));
    if (success(context, "Decoded IPA.CONSTANT reads a proven flat attribute")) {
        check(context.ops.size() == 1 && context.ops.front().opcode == Opcode::load_io &&
                  std::get<IoInfo>(context.ops.front().info).interpolation.mode == InterpolationMode::flat,
              "IPA.CONSTANT has no perspective correction or source multiplier");
    }
    context.run(InstructionName::ipa, ipa(4, 255, 6, 0x3fc, false, false, 2));
    if (success(context, "Decoded IPA.CONSTANT reads front-facing")) {
        check(context.ops.size() == 2 && context.ops.front().opcode == Opcode::load_io &&
                  context.ops.back().opcode == Opcode::predicate_to_mask,
              "IPA.CONSTANT preserves the fixed front-facing word conversion");
    }
    context.run(InstructionName::ipa, ipa(4, 255, 6, 0x80, false, false, 3));
    check(rejected(context, DiagnosticCode::unsupported_modifier, "Decoded IPA.SC is explicit"), "");
    context.run(InstructionName::ipa, ipa(4, 2, 6, 0, true, false, 2));
    check(rejected(context, DiagnosticCode::unsupported_modifier, "Indexed IPA.CONSTANT is explicit"), "");
    TestFacts nonflat;
    nonflat.values[{FactKind::input_interpolation, 0, 0, 0}] = perspective;
    RecordingContext no_override(Stage::fragment, nonflat);
    no_override.run(InstructionName::ipa, ipa(4, 255, 6, 0x80, false, false, 2));
    check(rejected(no_override, DiagnosticCode::unsupported_modifier,
                   "IPA.CONSTANT cannot override a nonflat fact under frozen contracts"), "");

    // Flat component: a plain load, no W multiply.
    context.run(InstructionName::ipa, instruction);
    if (success(context, "IPA flat load succeeds")) {
        check(context.ops.size() == 1 && context.ops.front().opcode == Opcode::load_io,
            "Flat IPA emits only the load");
        const auto& info = std::get<IoInfo>(context.ops.front().info);
        check(info.type.components == 4 &&
              info.interpolation_location_override == InterpolationLocation::center,
            "IPA preserves interface width and its center selector");
    }
    facts.values[{FactKind::input_interpolation, 0, 0, 0}] = linear_mode;
    RecordingContext centroid_context(Stage::fragment, facts);
    centroid_context.run(InstructionName::ipa,
        ipa(4, 255, 6, 0x80, false, false, 0, 1));
    if (success(centroid_context, "IPA centroid load succeeds")) {
        const auto& info = std::get<IoInfo>(centroid_context.ops.front().info);
        check(info.interpolation == Interpolation{InterpolationMode::linear,
                  InterpolationLocation::centroid} &&
              info.interpolation_location_override == InterpolationLocation::centroid,
            "IPA combines the fact mode with the decoded centroid location");
    }
    facts.values[{FactKind::input_interpolation, 0, 0, 0}] = flat_mode;
    for (unsigned location : {2u, 3u}) {
        context.run(InstructionName::ipa,
            ipa(4, 255, 6, 0x80, false, false, 0, location));
        check(rejected(context, DiagnosticCode::unsupported_modifier,
            "Unsupported IPA location is rejected"), "");
    }
    // Multiplier: an extra multiply by SrcB.
    auto multiplied = ipa(4, 255, 6, 0x80, false, false, 1);
    context.run(InstructionName::ipa, multiplied);
    if (success(context, "IPA multiplier succeeds")) {
        check(context.ops.size() == 2 && context.ops.back().opcode == Opcode::multiply,
            "IPA multiply mode multiplies by the source register");
    }
    // Saturation: the final multiply carries the guest saturate flag.
    auto saturated = ipa(4, 255, 6, 0x80, false, true, 1);
    context.run(InstructionName::ipa, saturated);
    if (success(context, "IPA saturate succeeds")) {
        check(context.ops.back().numeric.saturate, "Saturated IPA keeps the saturate modifier");
    }
    // Built-in position read without negation.
    auto position = ipa(4, 255, 255, 0x70, false, false, 0);
    context.run(InstructionName::ipa, position);
    if (success(context, "IPA position read succeeds")) {
        const auto& info = std::get<IoInfo>(context.ops.front().info);
        check(info.semantic == IoSemantic::fragment_coord && info.type.components == 4,
            "IPA position reads the full fragment coord interface");
    }
    auto front_facing = ipa(4, 255, 255, 0x3fc, false, false, 0);
    context.run(InstructionName::ipa, front_facing);
    if (success(context, "IPA front-facing read succeeds")) {
        check(context.ops.size() == 2 && context.ops.front().opcode == Opcode::load_io &&
              std::get<IoInfo>(context.ops.front().info).type.scalar == ScalarType::boolean &&
              context.ops.back().opcode == Opcode::predicate_to_mask,
            "IPA front-facing maps Boolean input to a guest word mask");
    }
    // Negated Y needs the support buffer, so it is rejected.
    facts.values[{FactKind::negate_y, 0, 0, 0}] = true;
    auto position_y = ipa(4, 255, 255, 0x74, false, false, 0);
    context.run(InstructionName::ipa, position_y);
    check(rejected(context, DiagnosticCode::unsupported_capability,
        "Negated Y position reads are rejected"), "");
    // Indexed addressing: dynamic offset load plus the explicit W correction;
    // indexed accesses consume no per-location fact.
    auto indexed = ipa(4, 2, 255, 0, true, false, 0);
    context.run(InstructionName::ipa, indexed);
    if (success(context, "IPA indexed load succeeds")) {
        const auto* load = find_op(context, Opcode::load_io, "IPA indexed load succeeds");
        if (load != nullptr) {
            const auto& info = std::get<IoInfo>(load->info);
            check(info.dynamic_byte_offset.has_value(),
                "Indexed IPA carries a dynamic byte offset");
            check(count_op(context, Opcode::subtract) == 1 &&
                  count_op(context, Opcode::multiply) == 1,
                "Indexed IPA subtracts the attribute base and multiplies by W");
        }
    }
    // Vertex-stage IPA reads attributes without interpolation facts.
    RecordingContext vertex(Stage::vertex, facts);
    vertex.run(InstructionName::ipa, instruction);
    check(success(vertex, "Vertex IPA reads attributes without interpolation facts"), "");

    // Perspective components multiply by the explicit fragment W.
    facts.values[{FactKind::input_interpolation, 0, 0, 0}] = perspective;
    RecordingContext perspective_context(Stage::fragment, facts);
    perspective_context.run(InstructionName::ipa, instruction);
    if (success(perspective_context, "IPA perspective load succeeds")) {
        check(perspective_context.ops.size() == 3,
            "Perspective IPA emits load, W load and multiply");
        check(perspective_context.ops[1].opcode == Opcode::load_io &&
              std::get<IoInfo>(perspective_context.ops[1].info).semantic ==
                  IoSemantic::fragment_coord &&
              std::get<IoInfo>(perspective_context.ops[1].info).component_mask == (1u << 3),
            "Perspective IPA loads fragment W");
        check(perspective_context.ops[2].opcode == Opcode::multiply,
            "Perspective IPA multiplies by W explicitly");
    }
    // Linear components load without correction.
    facts.values[{FactKind::input_interpolation, 0, 0, 0}] = linear_mode;
    RecordingContext linear_context(Stage::fragment, facts);
    linear_context.run(InstructionName::ipa, instruction);
    if (success(linear_context, "IPA linear load succeeds")) {
        check(linear_context.ops.size() == 1, "Linear IPA multiplies nothing");
    }
    // Undeclared component: missing fact, not a silent default.
    facts.values.erase({FactKind::input_interpolation, 0, 0, 0});
    RecordingContext undeclared(Stage::fragment, facts);
    undeclared.run(InstructionName::ipa, instruction);
    check(rejected(undeclared, DiagnosticCode::missing_fact,
        "Undeclared interpolation is a missing fact"), "");
}

void test_ldc() {
    TestFacts facts;
    RecordingContext context(Stage::vertex, facts);

    // Direct word load from slot 2 at byte offset 0x10.
    context.run(InstructionName::ldc, ldc(4, 255, 2, 0x10, 0, 4));
    if (const auto* op = find_op(context, Opcode::load, "LDC direct succeeds")) {
        const auto& info = std::get<MemoryInfo>(op->info);
        check(info.space == AddressSpace::constant_buffer &&
              info.constant_buffer_slot == 2 && info.byte_offset == 0x10,
            "LDC direct keeps slot and byte offset");
        check(!info.dynamic_byte_offset.has_value() && op->inputs.empty(),
            "LDC with RZ address has no dynamic input");
        check(op->numeric.domain == Domain::unsigned_integer &&
              op->numeric.scalar == ScalarType::u32,
            "Word loads carry the unsigned word interpretation");
        check(context.writes.size() == 1 && context.writes.front().first.index == 4,
            "LDC writes its destination");
    }
    // Addressed load: SrcA holds the byte displacement.
    context.run(InstructionName::ldc, ldc(4, 3, 1, 0x8, 0, 4));
    if (const auto* op = find_op(context, Opcode::load, "LDC addressed succeeds")) {
        check(op->inputs.size() == 1 &&
              std::get<MemoryInfo>(op->info).dynamic_byte_offset.has_value(),
            "LDC passes the register displacement explicitly");
    }
    // 64-bit load: two elements and two registers.
    context.run(InstructionName::ldc, ldc(4, 255, 1, 0x0, 0, 5));
    if (const auto* op = find_op(context, Opcode::load, "LDC 64-bit succeeds")) {
        check(std::get<MemoryInfo>(op->info).elements == 2 && op->outputs.size() == 2 &&
              context.writes.size() == 2,
            "64-bit LDC produces two ascending words");
    }
    // Narrow loads carry their interpretation in the numeric mode.
    context.run(InstructionName::ldc, ldc(4, 255, 1, 0x2, 0, 3));
    if (const auto* op = find_op(context, Opcode::load, "LDC narrow succeeds")) {
        const auto& info = std::get<MemoryInfo>(op->info);
        check(info.element_bytes == 2 && op->numeric.scalar == ScalarType::s16,
            "Signed half loads keep element width and sign mode");
    }
    // Indexed slots translate without captured binding-mask or size facts.
    context.run(InstructionName::ldc, ldc(4, 3, 1, 0x0, 2, 4));
    if (success(context, "LDC indexed succeeds")) {
        bool has_compare = false, has_select = false;
        for (const auto& op : context.ops) {
            has_compare = has_compare || op.opcode == Opcode::compare;
            has_select = has_select || op.opcode == Opcode::select;
        }
        check(has_compare && has_select, "Indexed LDC selects between supported slots");
        check(context.facts_reader_.dependencies().empty(),
            "Indexed LDC consumes no binding facts");
        std::uint32_t candidate_mask = 0;
        for (const auto& op : context.ops) {
            if (op.opcode != Opcode::load) continue;
            const auto& info = std::get<MemoryInfo>(op.info);
            candidate_mask |= 1u << info.constant_buffer_slot.value();
        }
        check(candidate_mask == 0x3ffffu, "Indexed LDC preserves all supported candidate slots");
    }
    facts.values[{FactKind::constant_buffer_bound_mask, 0, 0, 0}] = std::uint32_t{0};
    RecordingContext unbound(Stage::vertex, facts);
    unbound.run(InstructionName::ldc, ldc(4, 3, 1, 0x0, 2, 4));
    if (success(unbound, "Captured empty mask does not change indexed translation")) {
        check(count_op(unbound, Opcode::load) == 18 && unbound.facts_reader_.dependencies().empty(),
            "Empty captured mask is not consumed or used to prune candidate loads");
    }
    // Invalid encodings.
    context.run(InstructionName::ldc, ldc(4, 255, 1, 0x0, 0, 6));
    check(rejected(context, DiagnosticCode::unsupported_modifier, "B128 LDC is rejected"), "");
    context.run(InstructionName::ldc, ldc(4, 255, 20, 0x0, 0, 4));
    check(rejected(context, DiagnosticCode::unsupported_capability,
        "Slots beyond the pipeline range are rejected"), "");
}

void test_local_memory() {
    TestFacts facts;
    RecordingContext context(Stage::vertex, facts);
    // Word load from local memory.
    context.run(InstructionName::ldl, sized_access(InstructionName::ldl, 4, 3, 0x10, false, 4));
    if (const auto* op = find_op(context, Opcode::load, "LDL succeeds")) {
        const auto& info = std::get<MemoryInfo>(op->info);
        check(info.space == AddressSpace::local && info.byte_offset == 0 &&
              info.dynamic_byte_offset.has_value() && count_op(context, Opcode::add) == 1,
            "LDL composes its signed displacement into the dynamic address");
    }
    // Narrow store inserts at byte granularity.
    context.run(InstructionName::stl, sized_access(InstructionName::stl, 7, 3, 0x4, false, 1));
    if (const auto* op = find_op(context, Opcode::store, "STL narrow succeeds")) {
        check(count_op(context, Opcode::store) == 1, "STL emits one store");
        const auto& info = std::get<MemoryInfo>(op->info);
        check(info.element_bytes == 1 && info.data.size() == 1,
            "Byte stores keep one data role and byte elements");
    }
    // 128-bit load: four words.
    context.run(InstructionName::ldl, sized_access(InstructionName::ldl, 4, 3, 0x0, false, 6));
    if (const auto* op = find_op(context, Opcode::load, "LDL 128-bit succeeds")) {
        check(op->outputs.size() == 4 && std::get<MemoryInfo>(op->info).elements == 4,
            "128-bit loads produce four words");
    }
    context.run(InstructionName::ldl, sized_access(InstructionName::ldl, 4, 3, 0x0, false, 7));
    check(rejected(context, DiagnosticCode::unsupported_modifier, "Invalid LDL size is rejected"), "");
    // Shared memory is compute-only in this contract.
    context.run(InstructionName::lds, sized_access(InstructionName::lds, 4, 3, 0x0, false, 4));
    check(rejected(context, DiagnosticCode::unsupported_instruction, "LDS is rejected"), "");
    context.run(InstructionName::sts, sized_access(InstructionName::sts, 7, 3, 0x0, false, 4));
    check(rejected(context, DiagnosticCode::unsupported_instruction, "STS is rejected"), "");
    // Generic LD/ST stay explicit diagnostics.
    context.run(InstructionName::ld, sized_access(InstructionName::ld, 4, 3, 0x0, false, 4));
    check(rejected(context, DiagnosticCode::unsupported_instruction, "Generic LD is rejected"), "");
    context.run(InstructionName::st, sized_access(InstructionName::st, 7, 3, 0x0, false, 4));
    check(rejected(context, DiagnosticCode::unsupported_instruction, "Generic ST is rejected"), "");
}

void test_global_memory() {
    TestFacts facts;
    RecordingContext context(Stage::vertex, facts);
    context.run(InstructionName::ldg, sized_access(InstructionName::ldg, 4, 3, 0x8, false, 4));
    if (const auto* op = find_op(context, Opcode::load, "LDG succeeds")) {
        check(std::get<MemoryInfo>(op->info).space == AddressSpace::storage,
            "LDG addresses storage space");
    }
    context.run(InstructionName::stg, sized_access(InstructionName::stg, 7, 3, 0x8, false, 4));
    if (const auto* op = find_op(context, Opcode::store, "STG succeeds")) {
        const auto& info = std::get<MemoryInfo>(op->info);
        check(info.space == AddressSpace::storage && info.data.size() == 1,
            "STG stores one word into storage");
    }
    context.run(InstructionName::ldg, sized_access(InstructionName::ldg, 4, 3, 0x8, true, 4));
    if (const auto* op = find_op(context, Opcode::load, "Extended addressing retains its register pair")) {
        const auto& info = std::get<MemoryInfo>(op->info);
        check(info.dynamic_byte_offset && info.global_address_high && info.byte_offset == 8,
            "Extended global address preserves both register words and signed displacement");
    }
}

void test_barriers() {
    TestFacts facts;
    RecordingContext context(Stage::fragment, facts);
    context.run(InstructionName::bar, bar(0));
    if (const auto* op = find_op(context, Opcode::barrier, "BAR sync succeeds")) {
        check(std::get<MemoryInfo>(op->info).scope == MemoryScope::workgroup,
            "BAR sync keeps workgroup scope");
    }
    context.run(InstructionName::bar, bar(1));
    check(rejected(context, DiagnosticCode::unsupported_modifier, "Non-sync BAR is rejected"), "");
    DecodedInstruction with_id = bar(0);
    with_id.fields.push_back({DecodedField::barrier_id, 3});
    context.run(InstructionName::bar, with_id);
    check(rejected(context, DiagnosticCode::unsupported_modifier, "Nonzero barrier id is rejected"), "");
    context.run(InstructionName::membar, membar(0));
    if (const auto* op = find_op(context, Opcode::memory_barrier, "MEMBAR CTA succeeds")) {
        check(std::get<MemoryInfo>(op->info).scope == MemoryScope::workgroup,
            "CTA fences keep workgroup scope");
    }
    context.run(InstructionName::membar, membar(1));
    if (const auto* op = find_op(context, Opcode::memory_barrier, "MEMBAR GPU succeeds")) {
        check(std::get<MemoryInfo>(op->info).scope == MemoryScope::device,
            "GPU fences widen to device scope");
    }
    context.run(InstructionName::depbar, depbar());
    check(success(context, "DEPBAR is a scheduling no-op") && context.ops.empty(),
        "DEPBAR emits nothing");
}

void test_dispatch_routing() {
    TestFacts facts;
    RecordingContext context(Stage::vertex, facts);
    // The registry feeds checked dispatch: OUT has a row, and reaching it in a
    // vertex/fragment stage is a structured diagnostic rather than a drop.
    context.run(InstructionName::out, base_instruction(InstructionName::out, 0));
    check(rejected(context, DiagnosticCode::unsupported_instruction,
        "OUT is a structured diagnostic"), "");
}

void test_decoded_interpolation_msl() {
    const MetalProfile profile{"synthetic-metal", 1, 31, 32, 16, true,
        false, true, false, false, true};
    for (unsigned component = 0; component < 4; ++component) {
        for (unsigned location = 0; location < 2; ++location) {
            for (unsigned operation : {0u, 2u}) {
                TestFacts facts;
                facts.values[{FactKind::input_interpolation, 0, 0, static_cast<std::uint8_t>(component)}] =
                    operation == 2 ? flat_mode : linear_mode;
                facts.values[{FactKind::fragment_output_type, 0, 0, 0}] = IoType{ScalarType::f32, 4};
                auto registers = decoded_registers(stage_image(Stage::fragment,
                    {encoded_ipa(4, 0x80 + 4 * component, location, operation), encoded_ast(4, 0x80)}), facts);
                if (!registers) continue;
                if (operation == 2) {
                    check(program_op(registers->ir, Opcode::load_io) != nullptr &&
                              program_op(registers->ir, Opcode::multiply) == nullptr,
                          "Decoded IPA.CONSTANT and export retain no perspective arithmetic");
                }
                auto ssa = construct_ssa(std::move(*registers), {});
                check(std::holds_alternative<SsaProgram>(ssa), "Decoded component IPA reaches SSA");
                if (!std::holds_alternative<SsaProgram>(ssa)) continue;
                FactReader reader(facts);
                auto legal = legalize(std::get<SsaProgram>(std::move(ssa)), reader, profile);
                check(std::holds_alternative<LegalProgram>(legal), "IPA reconciles instruction location with fact mode");
                if (!std::holds_alternative<LegalProgram>(legal)) continue;
                auto value = std::get<LegalProgram>(std::move(legal));
                // The straight-line fixture has no phis. Avoid unrelated numeric
                // state-machine copies while testing the complete IO backend.
                StructuredProgram structured{std::move(value.ir), std::move(value.requirements), value.profile,
                    {BlockRegion{BlockId{0}}}, {{FunctionId{0}, RegionId{0}}}};
                auto plan = plan_resources(structured, profile);
                check(std::holds_alternative<ResourcePlan>(plan), "Decoded IPA resource plan succeeds");
                if (!std::holds_alternative<ResourcePlan>(plan)) continue;
                auto artifact = emit_msl(structured, std::get<ResourcePlan>(plan), {});
                if (operation == 2 && location != 0) {
                    const auto* error = std::get_if<Diagnostic>(&artifact);
                    check(error && error->code == DiagnosticCode::unsupported_modifier,
                          "Flat centroid IPA.CONSTANT keeps the backend capability diagnostic");
                    continue;
                }
                check(std::holds_alternative<MslArtifact>(artifact), "Decoded vector component IPA emits MSL");
                if (const auto* output = std::get_if<MslArtifact>(&artifact)) {
                    check(output->source.find(operation == 2 ? ", flat]]" :
                          location == 0 ? "center_no_perspective" : "centroid_no_perspective") != std::string::npos,
                        "MSL combines interpolation mode with the encoded sampling location");
                    check(output->source.find("in.generic0_" + std::to_string(component)) != std::string::npos,
                        "MSL reads the encoded vector lane");
                }
            }
        }
    }
}

void test_decode_to_pipeline_io() {
    TestFacts facts;
    facts.values[{FactKind::input_interpolation, 0, 0, 0}] = perspective;

    auto perspective_ir = decoded_registers(
        stage_image(Stage::fragment, {encoded_ipa(4, 0x80)}), facts);
    if (perspective_ir) {
        const auto* varying = program_op(perspective_ir->ir, Opcode::load_io, 0);
        const auto* position_w = program_op(perspective_ir->ir, Opcode::load_io, 1);
        const auto* multiply = program_op(perspective_ir->ir, Opcode::multiply);
        check(varying != nullptr && position_w != nullptr && multiply != nullptr,
            "Decoded perspective IPA emits both IO loads and explicit correction");
        if (varying != nullptr && position_w != nullptr) {
            const auto& varying_io = std::get<IoInfo>(varying->info);
            const auto& position_io = std::get<IoInfo>(position_w->info);
            check(varying_io.type == IoType{ScalarType::f32, 4} &&
                  varying_io.interpolation == perspective &&
                  position_io.semantic == IoSemantic::fragment_coord &&
                  position_io.type == IoType{ScalarType::f32, 4} &&
                  position_io.component_mask == 8,
                "Decoded perspective IPA preserves full interface widths");
        }
    }

    auto facing_ir = decoded_registers(
        stage_image(Stage::fragment, {encoded_ipa(4, 0x3fc)}), facts);
    if (facing_ir) {
        const auto* load = program_op(facing_ir->ir, Opcode::load_io);
        const auto* conversion = program_op(facing_ir->ir, Opcode::predicate_to_mask);
        check(load != nullptr && conversion != nullptr,
            "Decoded front-facing IPA emits a Boolean load and explicit mask conversion");
        if (load != nullptr && conversion != nullptr) {
            const auto& io = std::get<IoInfo>(load->info);
            check(io.semantic == IoSemantic::front_facing &&
                  io.type == IoType{ScalarType::boolean, 1} &&
                  conversion->outputs.size() == 1,
                "Front-facing conversion produces the guest word mask operation");
        }
    }

    facts.values[{FactKind::depth_convention, 0, 0, 0}] =
        DepthConvention::zero_to_one;
    FactReader reader(facts);
    auto decoded = decode(stage_image(Stage::vertex, {
        encoded_ast(255, 0x70, 4), encoded_ast(255, 0x80),
        encoded_ald(4, 0x80, true), encoded_ast(4, 0x84)}), reader, {});
    check(std::holds_alternative<DecodedProgram>(decoded),
        "Output-space IO fixture decodes");
    if (!std::holds_alternative<DecodedProgram>(decoded)) return;
    auto depth = reader.require(FactQuery<FactKind::depth_convention>{},
        Phase::emit_ir);
    check(std::holds_alternative<DepthConvention>(depth),
        "Output-space IO fixture records its depth convention");
    auto emitted = emit_ir(std::get<DecodedProgram>(std::move(decoded)), reader, {});
    check(std::holds_alternative<RegisterProgram>(emitted),
        "Decoded output-space ALD reaches register IR");
    if (!std::holds_alternative<RegisterProgram>(emitted)) return;
    const auto* output_load = program_op(std::get<RegisterProgram>(emitted).ir,
        Opcode::load_io);
    check(output_load != nullptr && std::get<IoInfo>(output_load->info).output_storage,
        "Decoded ALD.O retains output storage identity");

    auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(emitted)), {});
    check(std::holds_alternative<SsaProgram>(ssa), "ALD.O reaches SSA");
    if (!std::holds_alternative<SsaProgram>(ssa)) return;
    const MetalProfile profile{"synthetic-metal", 1, 31, 32, 16, true,
        false, true, false, false, true};
    auto legal = legalize(std::get<SsaProgram>(std::move(ssa)), reader, profile);
    check(std::holds_alternative<LegalProgram>(legal), "ALD.O legalizes");
    if (!std::holds_alternative<LegalProgram>(legal)) return;
    const auto& linear = std::get<LegalProgram>(legal);
    StructuredProgram direct{linear.ir, linear.requirements, linear.profile,
        {BlockRegion{BlockId{0}}}, {{FunctionId{0}, RegionId{0}}}};
    auto direct_plan = plan_resources(direct, profile);
    check(std::holds_alternative<ResourcePlan>(direct_plan), "Direct ALD.O plan succeeds");
    if (const auto* planned = std::get_if<ResourcePlan>(&direct_plan)) {
        auto direct_msl = emit_msl(direct, *planned, {});
        check(std::holds_alternative<MslArtifact>(direct_msl), "Decoded ALD.O emits output storage MSL");
        if (const auto* output = std::get_if<MslArtifact>(&direct_msl))
            check(output->source.find("as_type<uint>(out.generic0_0)") != std::string::npos,
                "ALD.O reads the written output rather than the input interface");
    }
    auto structured = structure(std::get<LegalProgram>(std::move(legal)), {});
    check(std::holds_alternative<StructuredProgram>(structured), "ALD.O structures");
    if (!std::holds_alternative<StructuredProgram>(structured)) return;
    auto plan = plan_resources(std::get<StructuredProgram>(structured), profile);
    check(std::holds_alternative<ResourcePlan>(plan), "ALD.O resources plan");
    if (!std::holds_alternative<ResourcePlan>(plan)) return;
    auto artifact = emit_msl(std::get<StructuredProgram>(std::move(structured)),
        std::get<ResourcePlan>(std::move(plan)), {});
    check(std::holds_alternative<MslArtifact>(artifact),
        "Decoded ALD.O emits through the structured pipeline");
    if (const auto* generated = std::get_if<MslArtifact>(&artifact))
        check(generated->source.find("as_type<uint>(out.generic0_0)") != std::string::npos,
            "Decoded ALD.O reaches an output-space MSL read");
}

} // namespace

int main() {
    test_registry();
    test_ald_direct(Stage::vertex);
    test_ald_direct(Stage::fragment);
    test_ald_raw_word();
    test_ald_ranges();
    test_ald_physical();
    test_ast();
    test_ipa();
    test_ldc();
    test_local_memory();
    test_global_memory();
    test_barriers();
    test_dispatch_routing();
    test_decode_to_pipeline_io();
    test_decoded_interpolation_msl();
    if (failures != 0) {
        std::cerr << failures << " emit IO checks failed\n";
        return 1;
    }
    std::cout << "shader.pipeline_emit_io: all checks passed\n";
    return 0;
}
