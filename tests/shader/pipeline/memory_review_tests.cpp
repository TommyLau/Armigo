// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pipeline/phases.h"
#include "pipeline/dispatch.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
void check(bool value, std::string_view label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
template<class T> T take(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) {
        std::cerr << error->message << '\n';
        throw 1;
    }
    return std::get<T>(std::move(result));
}
struct Facts final : FactSource {
    std::map<FactKey, FactValue> values;
    Facts() {
        values[{FactKind::fragment_output_type, 0}] = IoType{ScalarType::f32, 4};
        values[{FactKind::local_memory_bytes}] = std::uint32_t{64};
    }
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (auto it = values.find(key); it != values.end())
            return Known<FactValue>{it->second, {OriginKind::explicit_option, "synthetic-memory-review"}};
        return Unknown{"Synthetic fact unavailable"};
    }
};
MetalProfile profile() { return {"synthetic-metal", 1, 31, 32, 16, true, false, true, false, false, true}; }
std::uint64_t pattern(std::string_view bits) {
    std::uint64_t result{};
    for (std::size_t i = 0; i < bits.size(); ++i) if (bits[i] == '1') result |= std::uint64_t{1} << (63 - i);
    return result;
}
std::uint64_t mov(unsigned reg, std::uint32_t value) {
    return pattern("000000010000") | reg | (7ull << 16) | (std::uint64_t{value} << 20);
}
std::uint64_t ldc(unsigned base, int offset, unsigned size = 4, unsigned mode = 0) {
    return pattern("1110111110010x") | 4 | (std::uint64_t{base} << 8) | (7ull << 16) |
        (std::uint64_t{static_cast<std::uint16_t>(offset)} << 20) | (std::uint64_t{mode} << 44) |
        (std::uint64_t{size} << 48);
}
std::uint64_t ast(unsigned source = 4) {
    return pattern("1110111111110x") | source | (255ull << 8) | (7ull << 16) | (0x80ull << 20);
}
std::uint64_t memory(std::string_view opcode, unsigned base, int offset) {
    return pattern(opcode) | 4 | (std::uint64_t{base} << 8) | (7ull << 16) |
        ((std::uint64_t{static_cast<std::uint32_t>(offset)} & 0xffffff) << 20) | (4ull << 48);
}
StageImage image(std::initializer_list<std::uint64_t> words) {
    StageImage result; result.stage = Stage::fragment;
    auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) result.bytes.push_back(std::byte((word >> (i * 8)) & 255));
    };
    for (auto word : words) {
        if (result.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    if (result.bytes.size() % 32 == 0) append(0);
    append(pattern("111000110000") | (7ull << 16) | 15);
    return result;
}
RegisterProgram registers(const StageImage& input, Facts& facts) {
    FactReader reader(facts);
    return take(emit_ir(take(decode(input, reader, {})), reader, {}));
}
LegalProgram legalized(const StageImage& input, Facts& facts) {
    FactReader reader(facts);
    return take(legalize(take(construct_ssa(registers(input, facts), {})), reader, profile()));
}
MslArtifact msl(LegalProgram input) {
    // These straight-line shaders need no control temporaries from the general
    // structurizer, whose copy generator belongs to the separate numeric lane.
    check(input.ir.blocks.size() == 1 && input.ir.blocks[0].phis.empty(), "Straight-line MSL fixture");
    StructuredProgram structured{std::move(input.ir), std::move(input.requirements), input.profile,
        {BlockRegion{BlockId{0}}}, {{FunctionId{0}, RegionId{0}}}};
    auto plan = take(plan_resources(structured, profile()));
    return take(emit_msl(structured, plan, {}));
}
// Exercise the production memory generator against the complete decoded,
// legalized and planned dynamic load. Numeric expression generators are owned
// by another lane and are not registered in this baseline.
struct MemoryContext final : MslContext {
    StructuredProgram p;
    ResourcePlan plan;
    std::string text;
    explicit MemoryContext(LegalProgram input) : p(take(structure(std::move(input), {}))),
        plan(take(plan_resources(p, profile()))) {}
    const StructuredProgram& program() const override { return p; }
    const ResourcePlan& resources() const override { return plan; }
    const Operation& operation(OperationId id) const override { return p.ir.operations[id.value]; }
    Result<std::string> operand(const Operand& value, ScalarType) override {
        if (const auto* literal = std::get_if<LiteralBits>(&value)) return std::to_string(literal->bits) + "u";
        return "value" + std::to_string(std::get<ValueId>(value).value);
    }
    Result<std::string> resource(ResourceId id) override { return "resource" + std::to_string(id.value); }
    Status assign(ValueId, std::string_view expression, ScalarType) override { text += expression; return std::monostate{}; }
    Status statement(std::string_view statement) override { text += statement; return std::monostate{}; }
    Status require_helper(std::string_view) override { return std::monostate{}; }
    Diagnostic error(OperationId id, DiagnosticCode code, std::string message, std::string modifier) const override {
        auto result = diagnose(Phase::emit_msl, code, operation(id).source, std::move(message));
        result.modifier = std::move(modifier);
        return result;
    }
};
void dynamic_msl(LegalProgram input) {
    MemoryContext context(std::move(input));
    for (std::uint32_t i = 0; i < context.p.ir.operations.size(); ++i) {
        auto& op = context.p.ir.operations[i];
        if (op.opcode != Opcode::load) continue;
        take(generate_stage_memory(context, OperationId{i}));
        check(!context.text.empty(), "Dynamic LDC reaches production MSL memory generator");
        auto& info = std::get<MemoryInfo>(op.info);
        const bool had_bound = info.dynamic_byte_offset_upper_bound.has_value();
        info.dynamic_byte_offset_upper_bound.reset();
        const auto regenerated = generate_stage_memory(context, OperationId{i});
        check(had_bound ? std::holds_alternative<Diagnostic>(regenerated) :
                          std::holds_alternative<std::monostate>(regenerated),
            had_bound ? "Narrow dynamic window rejects lost proof metadata" :
                        "Full dynamic window needs no invented proof metadata");
    }
}
const Operation& operation(const Program& p, Opcode code) {
    for (const auto& op : p.operations) if (op.opcode == code) return op;
    throw 1;
}
void indexed_zero() {
    Facts facts;
    // Every shader-addressable candidate is independent of the current draw binding.
    auto input = image({mov(3, 0x10000), ldc(3, 0, 4, 2), ast()});
    auto ir = registers(input, facts);
    const auto& select = operation(ir.ir, Opcode::select);
    check(std::get<LiteralBits>(select.inputs[2]).bits == 0, "Indexed LDC fallback is zero");
    check(operation(ir.ir, Opcode::compare).opcode == Opcode::compare, "First candidate is compared");
    auto legal = legalized(input, facts);
    check(legal.requirements.uses.size() == constant_buffer_slot_count + 1,
        "Indexed LDC plans all CB windows and one size table");
    check(std::none_of(legal.ir.dependencies.begin(), legal.ir.dependencies.end(), [](const auto& dependency) {
        return dependency.key.kind == FactKind::constant_buffer_bound_mask ||
               dependency.key.kind == FactKind::constant_buffer_size;
    }), "Indexed LDC consumes no draw-binding facts");
    check(!msl(std::move(legal)).source.empty(), "Indexed LDC reaches MSL");
    facts.values[{FactKind::constant_buffer_bound_mask}] = std::uint32_t{0};
    facts.values.erase({FactKind::constant_buffer_size, 0});
    legal = legalized(input, facts);
    check(legal.requirements.uses.size() == constant_buffer_slot_count + 1,
        "Deprecated mask values do not change indexed CB resources");
    check(!msl(std::move(legal)).source.empty(), "Deprecated mask values do not change indexed MSL");
}
void stores_and_signed_offsets() {
    Facts facts;
    const auto attribute = registers(image({mov(4, 7), ast()}), facts);
    check(operation(attribute.ir, Opcode::store_io).inputs.size() == 1, "Decoded AST consumes source-B data");
    for (auto code : {"1110111101010x", "1110111011011x"}) {
        auto ir = registers(image({mov(3, 4), mov(4, 7), memory(code, 3, -4)}), facts);
        const auto& store = operation(ir.ir, Opcode::store);
        const auto& info = std::get<MemoryInfo>(store.info);
        check(info.byte_offset == 0 && info.dynamic_byte_offset.has_value(), "Store complete address is explicit");
        check(std::get<Register>(store.inputs[info.data[0].value]).index == 4, "Decoded store consumes data register R4");
        const auto& add = operation(ir.ir, Opcode::add);
        check(add.numeric.scalar == ScalarType::u32 && std::get<LiteralBits>(add.inputs[1]).bits == 0xfffffffc,
            "Signed store displacement is added at guest width");
    }
    for (auto code : {"1110111101000x", "1110111011010x"}) {
        auto ir = registers(image({mov(3, 4), memory(code, 3, -4), ast()}), facts);
        check(std::get<MemoryInfo>(operation(ir.ir, Opcode::load).info).byte_offset == 0,
            "Local/global load uses complete guest address");
        check(operation(ir.ir, Opcode::add).numeric.scalar == ScalarType::u32, "Load address addition stays word sized");
    }
    for (auto [base, displacement] : {std::pair{4u, -4}, std::pair{0xfffffffcu, 4}}) {
        auto legal = legalized(image({mov(3, base), ldc(3, displacement), ast()}), facts);
        const auto& load = operation(legal.ir, Opcode::load);
        const auto& info = std::get<MemoryInfo>(load.info);
        check(info.byte_offset == 0 && std::get<LiteralBits>(load.inputs[info.dynamic_byte_offset->value]).bits == 0,
            "Guest address wraps before widened bounds arithmetic");
        check(!msl(std::move(legal)).source.empty(), "Signed/wrapped LDC reaches MSL");
    }
}
LegalProgram dynamic_legal(unsigned mask, unsigned size, Facts& facts) {
    const auto ald = pattern("1110111111011x") | 3 | (255ull << 8) | (7ull << 16) | (0x70ull << 20);
    auto program = registers(image({ald, ldc(3, 0, size), ast()}), facts);
    // The LDC and address producer use the real decoder. Insert an independently
    // authored IR range proof: the unrelated LOP32I decoder/emitter field
    // mismatch is outside this lane's edit ownership.
    auto& p = program.ir;
    for (auto& block : p.blocks) {
        for (std::size_t index = 0; index < block.operations.size(); ++index) {
            const auto id = block.operations[index];
            if (p.operations[id.value].opcode != Opcode::load) continue;
            auto& load = p.operations[id.value];
            const auto role = std::get<MemoryInfo>(load.info).dynamic_byte_offset->value;
            const auto base = load.inputs[role];
            ValueId bounded{static_cast<std::uint32_t>(p.values.size())};
            p.values.push_back({ValueType::word32});
            load.inputs[role] = bounded;
            OperationId mask_id{static_cast<std::uint32_t>(p.operations.size())};
            p.operations.push_back({load.source, Opcode::bit_and, Effects::none, {},
                {base, LiteralBits{mask, ValueType::word32}}, {bounded}, {}});
            block.operations.insert(block.operations.begin() + index, mask_id);
            break;
        }
    }
    FactReader reader(facts);
    return take(legalize(take(construct_ssa(std::move(program), {})), reader, profile()));
}
void dynamic_bound() {
    Facts facts;
    auto legal = dynamic_legal(60, 4, facts);
    const auto& info = std::get<MemoryInfo>(operation(legal.ir, Opcode::load).info);
    check(info.dynamic_byte_offset_upper_bound == 60, "Legalization persists masked byte-offset bound");
    dynamic_msl(std::move(legal));
    legal = dynamic_legal(3, 0, facts);
    check(std::get<BufferUse>(legal.requirements.uses.front()).size.bytes == 4,
        "Dynamic byte loads reflect only complete proven backing words");
    dynamic_msl(std::move(legal));

    const auto ald = pattern("1110111111011x") | 3 | (255ull << 8) | (7ull << 16) | (0x70ull << 20);
    legal = legalized(image({ald, ldc(3, 0, 4), ast()}), facts);
    const auto& unknown = std::get<MemoryInfo>(operation(legal.ir, Opcode::load).info);
    check(!unknown.dynamic_byte_offset_upper_bound &&
        std::get<BufferUse>(legal.requirements.uses.front()).size.bytes == constant_buffer_window_bytes,
        "Unknown dynamic LDC uses the full runtime window without an invented bound");
    dynamic_msl(std::move(legal));
}

void narrow_backing() {
    Facts facts;
    for (unsigned size = 0; size < 4; ++size) {
        for (int offset : {0, 3, 62}) {
            auto legal = legalized(image({ldc(255, offset, size), ast()}), facts);
            const auto& buffer = std::get<BufferUse>(legal.requirements.uses.front());
            const auto width = size < 2 ? 1u : 2u;
            const auto guest_address = width == 2 ? (unsigned(offset) & ~1u) : unsigned(offset);
            check(buffer.size.bytes == ((guest_address & ~3u) + 4u) && buffer.size.alignment == 4,
                "Narrow LDC reflects its normalized complete-word backing range");
            const auto source = msl(std::move(legal)).source;
            check(source.find(width == 1 ? "& 255u" : "& 65535u") != std::string::npos,
                "Narrow load extracts the normalized subword");
            if (size & 1) check(source.find(size == 1 ? "as_type<char>" : "as_type<short>") != std::string::npos,
                "Signed narrow load explicitly sign extends");
        }
    }
    FactReader reader(facts);
    auto result = legalize(take(construct_ssa(registers(image({ldc(255, 0, 0), ast()}), facts), {})), reader, profile());
    check(std::holds_alternative<LegalProgram>(result),
        "Ordinary byte load does not specialize on a short draw binding");
}
}
int main() {
    for (auto test : {indexed_zero, stores_and_signed_offsets, dynamic_bound, narrow_backing}) {
        try { test(); } catch (...) { ++failures; std::cerr << "Test could not complete\n"; }
    }
    return failures ? 1 : 0;
}
