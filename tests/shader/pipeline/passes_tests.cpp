// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic fixtures exercise the optimization passes.

#include "pipeline/passes/passes.h"
#include "pipeline/phases.h"
#include <algorithm>
#include <iostream>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
constexpr NumericMode u32{Domain::unsigned_integer, ScalarType::u32};
constexpr NumericMode s32{Domain::signed_integer, ScalarType::s32};
constexpr NumericMode u64{Domain::unsigned_integer, ScalarType::u64};
constexpr NumericMode s64{Domain::signed_integer, ScalarType::s64};
constexpr NumericMode boolean{Domain::boolean, ScalarType::boolean};
constexpr NumericMode floating{Domain::floating, ScalarType::f32, Rounding::nearest_even};
void check(bool condition, std::string_view message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
bool success(Status status, std::string_view message) {
    if (const auto* error = std::get_if<Diagnostic>(&status)) {
        std::cerr << "FAIL: " << message << ": " << error->message << '\n'; ++failures; return false;
    }
    return true;
}
void rejected(Status status, DiagnosticCode code, std::string_view message) {
    const auto* error = std::get_if<Diagnostic>(&status);
    check(error && error->phase == Phase::legalize && error->code == code, message);
}
LiteralBits word(std::uint64_t value) { return {value, ValueType::word32}; }
LiteralBits wide(std::uint64_t value) { return {value, ValueType::word64}; }
LiteralBits pred(bool value) { return {static_cast<std::uint64_t>(value), ValueType::predicate}; }
Program empty() {
    Program p; p.stage = Stage::fragment; p.entry = FunctionId{0};
    p.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
    p.blocks.push_back({FunctionId{0}, {8, {}}, {}, {}, {}, Return{}});
    return p;
}
ValueId value(Program& p, ValueType type = ValueType::word32) {
    ValueId id{static_cast<std::uint32_t>(p.values.size())}; p.values.push_back({type}); return id;
}
ValueId parameter(Program& p, ValueType type = ValueType::word32) {
    auto id = value(p, type); p.functions[0].parameters.push_back(id); return id;
}
ValueId emit(Program& p, Opcode opcode, std::vector<Operand> inputs, NumericMode mode = u32,
             ValueType type = ValueType::word32, OperationInfo info = {}, Effects effects = Effects::none,
             BlockId block = BlockId{0}) {
    auto id = value(p, type);
    p.blocks[block.value].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back({{40, InstructionName::iadd}, opcode, effects, mode, std::move(inputs), {id}, std::move(info)});
    return id;
}
void returns(Program& p, std::vector<Operand> values, std::vector<ValueType> types, BlockId block = BlockId{0}) {
    p.functions[0].results = std::move(types); p.blocks[block.value].terminator = Return{std::move(values)};
}
const LiteralBits* returned_literal(const Program& p, std::size_t index = 0) {
    for (const auto& block : p.blocks) if (const auto* result = std::get_if<Return>(&block.terminator)) {
        if (result->values.size() > index) return std::get_if<LiteralBits>(&result->values[index]);
    }
    return nullptr;
}
void folded(Opcode opcode, std::vector<Operand> inputs, std::uint64_t expected,
            NumericMode mode = u32, ValueType type = ValueType::word32, OperationInfo info = {}) {
    auto p = empty(); auto result = emit(p, opcode, std::move(inputs), mode, type, std::move(info));
    returns(p, {result}, {type});
    if (!success(passes::cleanup(p), "Constant expression cleanup")) return;
    const auto* bits = returned_literal(p);
    check(bits && bits->bits == expected && bits->type == type, "Folded value has exact expected bits and type");
    check(p.operations.empty(), "Folded expression and copies are removed");
}
void integer_folding() {
    folded(Opcode::add, {word(UINT32_MAX), word(2)}, 1);
    folded(Opcode::add, {wide(UINT64_MAX), wide(2)}, 1, u64, ValueType::word64);
    folded(Opcode::multiply, {word(0x80000001), word(3)}, 0x80000003, s32);
    folded(Opcode::multiply, {wide(UINT64_MAX), wide(UINT64_MAX)}, 1, s64, ValueType::word64);
    folded(Opcode::subtract, {word(0), word(1)}, UINT32_MAX);
    folded(Opcode::negate, {word(0x80000000)}, 0x80000000, s32);
    folded(Opcode::absolute, {wide(0x8000000000000000ULL)}, 0x8000000000000000ULL, s64, ValueType::word64);
    folded(Opcode::divide, {word(0xfffffff9), word(2)}, 0xfffffffd, s32);
    folded(Opcode::remainder, {word(0xfffffff9), word(2)}, UINT32_MAX, s32);
    folded(Opcode::divide, {wide(0xfffffffffffffff9ULL), wide(2)}, 0xfffffffffffffffdULL, s64, ValueType::word64);
    folded(Opcode::divide, {word(UINT32_MAX), word(2)}, 0x7fffffff);
    folded(Opcode::shift_left, {word(3), word(33)}, 6);
    folded(Opcode::shift_right, {word(0x80000000), word(33)}, 0xc0000000, s32);
    folded(Opcode::shift_right, {word(0x80000000), word(32)}, 0x80000000, s32);
    folded(Opcode::shift_right, {wide(0x8000000000000000ULL), word(63)}, UINT64_MAX, s64, ValueType::word64);
    folded(Opcode::minimum, {word(0xffffffff), word(1)}, UINT32_MAX, s32);
    folded(Opcode::maximum, {word(0xffffffff), word(1)}, UINT32_MAX);
    folded(Opcode::clamp, {word(0xfffffff0), word(0xfffffffc), word(2)}, 0xfffffffc, s32);
    folded(Opcode::compare, {word(0x80000000), word(0x7fffffff)}, 1, s32, ValueType::predicate, ComparisonInfo{Compare::less, false});
    folded(Opcode::compare, {wide(UINT64_MAX), wide(1)}, 1, u64, ValueType::predicate, ComparisonInfo{Compare::greater, false});
    folded(Opcode::bit_not, {word(0x12345678)}, 0xedcba987);
    folded(Opcode::bit_reverse, {word(1)}, 0x80000000);
    folded(Opcode::bit_count, {wide(UINT64_MAX)}, 64, u64, ValueType::word64);
    folded(Opcode::bit_extract, {word(0x1234f000), word(12), word(4)}, UINT32_MAX, s32);
    folded(Opcode::bit_extract, {wide(UINT64_MAX), word(0), word(64)}, UINT64_MAX, u64, ValueType::word64);
    folded(Opcode::bit_insert, {word(0xffff0000), word(5), word(4), word(4)}, 0xffff0050);
    folded(Opcode::funnel_shift_left, {word(0x80000000), word(2), word(1)}, 5);
    folded(Opcode::funnel_shift_right, {word(2), word(1), word(1)}, 0x80000001);
    folded(Opcode::funnel_shift_left, {word(7), word(9), word(32)}, 9);
    folded(Opcode::pack_word64, {word(0x12345678), word(0x87654321)}, 0x8765432112345678ULL, u64, ValueType::word64);
    folded(Opcode::logical_xor, {pred(true), pred(false)}, 1, boolean, ValueType::predicate);
    folded(Opcode::predicate_to_mask, {pred(true)}, UINT32_MAX);
    folded(Opcode::mask_to_predicate, {word(0x80000000)}, 1, boolean, ValueType::predicate);
}
void copies_and_modes() {
    auto p = empty(); auto input = parameter(p);
    auto a = emit(p, Opcode::copy, {input});
    auto b = emit(p, Opcode::add, {a, word(0)});
    auto c = emit(p, Opcode::multiply, {b, word(1)});
    auto d = emit(p, Opcode::bit_and, {c, word(UINT32_MAX)});
    auto e = emit(p, Opcode::select, {pred(true), d, word(9)});
    returns(p, {e}, {ValueType::word32});
    std::reverse(p.operations.begin(), p.operations.end());
    for (auto& id : p.blocks[0].operations) id.value = static_cast<std::uint32_t>(p.operations.size() - 1 - id.value);
    success(passes::cleanup(p), "Copies simplify independently of operation arena order");
    check(p.operations.empty() && std::get<ValueId>(std::get<Return>(p.blocks[0].terminator).values[0]) == input, "Integer identities preserve original parameter");
    for (auto bits : {0x80000000U, 0x7fc12345U, 0x7f812345U, 1U}) {
        p = empty(); auto mode = floating; mode.precise = true; mode.flush_inputs = true; mode.flush_output = true; mode.saturate = true;
        auto result = emit(p, Opcode::add, {word(bits), word(0)}, mode);
        returns(p, {result}, {ValueType::word32});
        success(passes::cleanup(p), "Floating boundary cleanup");
        check(p.operations.size() == 1 && p.operations[0].opcode == Opcode::add && p.operations[0].numeric == mode,
              "Signed zero, NaNs and FTZ retain arithmetic and every numeric mode");
        check(std::get<LiteralBits>(p.operations[0].inputs[0]).bits == bits, "Floating payload preserved exactly");
        p.operations[0].numeric = floating;
        success(passes::cleanup(p), "Ordinary floating constants are not host-folded");
        check(p.operations[0].opcode == Opcode::add, "Floating add-zero is never simplified");
    }
    p = empty(); auto mode = u32; mode.precise = true;
    a = emit(p, Opcode::add, {word(1), word(2)}, mode); returns(p, {a}, {ValueType::word32});
    success(passes::cleanup(p), "Precise integer metadata survives");
    check(p.operations.size() == 1 && p.operations[0].numeric.precise, "Precise flag is never discarded by folding");
    p = empty(); a = emit(p, Opcode::pack_half2, {word(0x3f800001), word(0x80000000)}, floating,
        ValueType::word32, ConversionInfo{ScalarType::f32, ScalarType::f16});
    auto low = value(p), high = value(p);
    p.blocks[0].operations.push_back(OperationId{1});
    p.operations.push_back({{}, Opcode::unpack_half2, Effects::none, floating, {a}, {low, high}, ConversionInfo{ScalarType::f16, ScalarType::f32}});
    returns(p, {low, high}, {ValueType::word32, ValueType::word32});
    success(passes::cleanup(p), "Half pack/unpack is retained");
    check(p.operations.size() == 2, "Half rounding boundary is not cancelled");
}
Program loop(bool increment = false) {
    auto p = empty(); p.blocks.resize(4);
    for (auto& block : p.blocks) block.function = FunctionId{0};
    p.functions[0].entry = BlockId{2}; p.functions[0].blocks = {BlockId{2}, BlockId{0}, BlockId{3}, BlockId{1}};
    auto initial = parameter(p), condition = parameter(p, ValueType::predicate), phi = value(p);
    p.blocks[2].terminator = Jump{BlockId{0}};
    p.blocks[0].predecessors = {BlockId{3}, BlockId{2}};
    p.blocks[0].terminator = Branch{condition, BlockId{3}, BlockId{1}};
    p.blocks[3].predecessors = {BlockId{0}}; p.blocks[3].terminator = Jump{BlockId{0}};
    p.blocks[1].predecessors = {BlockId{0}};
    auto back = emit(p, increment ? Opcode::add : Opcode::copy,
        increment ? std::vector<Operand>{phi, word(1)} : std::vector<Operand>{phi}, u32, ValueType::word32, {}, Effects::none, BlockId{3});
    p.blocks[0].phis = {{{}, phi, {{BlockId{3}, back}, {BlockId{2}, initial}}}};
    returns(p, {phi}, {ValueType::word32}, BlockId{1});
    return p;
}
void phis_and_dead_code() {
    auto p = loop(); success(passes::cleanup(p), "Self phi through backedge copy collapses");
    check(p.operations.empty() && p.blocks[0].phis.empty(), "Trivial loop phi and backedge copy removed");
    check(std::get<ValueId>(std::get<Return>(p.blocks[1].terminator).values[0]) == ValueId{0}, "Phi returns initial parameter");
    p = loop(true); success(passes::cleanup(p), "Induction phi survives");
    check(p.operations.size() == 1 && p.blocks[0].phis.size() == 1, "Live changing backedge is not collapsed");
    returns(p, {}, {}, BlockId{1}); success(passes::cleanup(p), "Unused induction cycle is removed");
    check(p.operations.empty() && p.blocks[0].phis.empty(), "Mark/sweep deletes unused phi/arithmetic cycle");
    p = loop(); auto initial_b = parameter(p), phi_b = value(p);
    p.operations[0].inputs = {phi_b};
    p.blocks[0].phis.push_back({{}, phi_b, {{BlockId{2}, initial_b}, {BlockId{3}, ValueId{2}}}});
    returns(p, {ValueId{2}, phi_b}, {ValueType::word32, ValueType::word32}, BlockId{1});
    success(passes::cleanup(p), "Parallel loop phi swap survives");
    check(p.blocks[0].phis.size() == 2 && p.operations.empty(), "Phi swap keeps distinct initial values and only removes the copy");
    p = empty();
    emit(p, Opcode::add, {word(1), word(2)});
    emit(p, Opcode::derivative_x, {word(1)}, floating, ValueType::word32, {}, Effects::derivative);
    MemoryInfo memory; memory.constant_buffer_slot = 0; memory.alignment = 4; memory.is_volatile = true;
    emit(p, Opcode::load, {}, {}, ValueType::word32, memory, Effects::read_memory);
    p.blocks[0].operations.push_back(OperationId{3});
    p.operations.push_back({{}, Opcode::discard, Effects::discard, {}, {}, {}, {}});
    success(passes::cleanup(p), "Effects survive unused-output cleanup");
    check(p.operations.size() == 3 && p.operations[0].opcode == Opcode::derivative_x &&
        p.operations[1].opcode == Opcode::load && p.operations[2].opcode == Opcode::discard,
        "Derivatives, volatile reads and discard retain execution order");
}
Program diamond() {
    auto p = empty(); p.blocks.resize(4);
    for (auto& block : p.blocks) block.function = FunctionId{0};
    p.functions[0].entry = BlockId{3}; p.functions[0].blocks = {BlockId{3}, BlockId{2}, BlockId{0}, BlockId{1}};
    p.blocks[3].terminator = Branch{pred(true), BlockId{0}, BlockId{2}};
    p.blocks[0].predecessors = {BlockId{3}}; p.blocks[0].terminator = Jump{BlockId{1}};
    p.blocks[2].predecessors = {BlockId{3}}; p.blocks[2].terminator = Jump{BlockId{1}};
    p.blocks[1].predecessors = {BlockId{0}, BlockId{2}};
    auto a = emit(p, Opcode::copy, {word(17)}, u32, ValueType::word32, {}, Effects::none, BlockId{0});
    auto b = emit(p, Opcode::copy, {word(29)}, u32, ValueType::word32, {}, Effects::none, BlockId{2});
    auto phi = value(p); p.blocks[1].phis.push_back({{}, phi, {{BlockId{2}, b}, {BlockId{0}, a}}});
    returns(p, {phi}, {ValueType::word32}, BlockId{1}); return p;
}
void branches() {
    auto constant_phi = diamond();
    const auto phi_condition = parameter(constant_phi, ValueType::predicate);
    std::get<Branch>(constant_phi.blocks[3].terminator).condition = phi_condition;
    constant_phi.operations[1].inputs[0] = word(17);
    success(passes::cleanup(constant_phi), "Equivalent literal phi arms collapse without folding the branch");
    check(constant_phi.blocks.size() == 4 && constant_phi.operations.empty() && returned_literal(constant_phi) &&
        returned_literal(constant_phi)->bits == 17, "Constant phi becomes exact bits while both CFG arms remain");
    auto p = diamond(); success(passes::cleanup(p), "Constant branch removes unreachable arm");
    check(p.blocks.size() == 3 && p.operations.empty(), "Unreachable block and orphan operations compacted");
    check(returned_literal(p) && returned_literal(p)->bits == 17, "Phi loses only removed edge");
    check(p.functions[0].entry == BlockId{2}, "Function entry remapped without relying on arena order");
    p = diamond(); p.blocks[3].terminator = Switch{word(9), {{9, BlockId{2}}, {10, BlockId{0}}}, BlockId{0}};
    success(passes::cleanup(p), "Constant switch selects case");
    check(returned_literal(p) && returned_literal(p)->bits == 29, "Switch uses matching case bits");
    p = empty(); auto condition = parameter(p, ValueType::predicate);
    p.blocks.push_back({FunctionId{0}, {}, {BlockId{0}}, {}, {}, Return{}});
    p.functions[0].blocks.push_back(BlockId{1});
    p.blocks[0].terminator = Branch{condition, BlockId{1}, BlockId{1}};
    success(passes::cleanup(p), "Duplicate edge branch collapses");
    check(std::holds_alternative<Jump>(p.blocks[0].terminator) && p.blocks[1].predecessors.size() == 1, "Distinct predecessor invariant retained");
}
class Facts final : public FactSource {
public:
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::negate_y) return Known<FactValue>{false, {OriginKind::explicit_option, "synthetic-pass-fact"}};
        return Unknown{"Synthetic fact unavailable"};
    }
};
MetalProfile profile() { return {"synthetic-passes", 1, 31, 32, 16, true, false, true, false, false, true}; }
void legalization() {
    Facts source; FactReader reader(source); auto metal = profile(); auto p = empty();
    auto result = emit(p, Opcode::add, {word(0x80000000), word(1)}, floating);
    returns(p, {result}, {ValueType::word32});
    p.operations[0].numeric.flush_inputs = true; p.operations[0].numeric.precise = true;
    const auto mode = p.operations[0].numeric;
    success(passes::legalize_nontexture(p, reader, metal), "Explicit floating modes survive legalization");
    check(p.operations[0].numeric == mode && p.dependencies.empty(), "No precision substitutions or invented facts");
    auto fact = reader.require(FactQuery<FactKind::negate_y>{}, Phase::legalize);
    check(std::holds_alternative<bool>(fact), "Synthetic fact consumed");
    success(passes::legalize_nontexture(p, reader, metal), "Consumed dependencies merge");
    check(p.dependencies.size() == 1, "Reader dependency is retained by pass");
    p = empty(); result = emit(p, Opcode::add, {wide(0), wide(1)}, {Domain::floating, ScalarType::f64, Rounding::nearest_even}, ValueType::word64);
    returns(p, {result}, {ValueType::word64});
    rejected(passes::legalize_nontexture(p, reader, metal), DiagnosticCode::unsupported_capability, "FP64 never substitutes FP32");
    check(p.operations[0].numeric.scalar == ScalarType::f64, "Rejected FP64 remains intact");
    p = empty(); result = emit(p, Opcode::pack_half2, {word(0), word(0)}, floating, ValueType::word32, ConversionInfo{ScalarType::f32, ScalarType::f16});
    returns(p, {result}, {ValueType::word32}); metal.supports_fp16 = false;
    rejected(passes::legalize_nontexture(p, reader, metal), DiagnosticCode::unsupported_capability, "Half conversion capability checked");
    metal = profile(); p = empty(); p.stage = Stage::vertex;
    emit(p, Opcode::derivative_x, {word(1)}, floating, ValueType::word32, {}, Effects::derivative);
    rejected(passes::legalize_nontexture(p, reader, metal), DiagnosticCode::unsupported_capability, "Vertex derivative rejected");
    p = empty(); p.blocks[0].operations.push_back(OperationId{0});
    MemoryInfo barrier; barrier.space = AddressSpace::local;
    p.operations.push_back({{}, Opcode::memory_barrier, Effects::barrier, {}, {}, {}, barrier});
    rejected(passes::legalize_nontexture(p, reader, metal), DiagnosticCode::unsupported_capability, "Unproven memory scope rejected");
    p = empty(); auto shuffled = value(p), valid = value(p, ValueType::predicate);
    p.blocks[0].operations.push_back(OperationId{0});
    p.operations.push_back({{}, Opcode::shuffle, Effects::subgroup, {}, {word(17), word(1), word(31)}, {shuffled, valid}, SubgroupInfo{ShuffleKind::up, 32}});
    success(passes::legalize_nontexture(p, reader, metal), "Explicit subgroup operation preserved");
    check(p.operations[0].outputs.size() == 2 && p.operations[0].inputs.size() == 3, "Shuffle mask and validity result retained");
    metal.supports_simd_shuffle = false;
    rejected(passes::legalize_nontexture(p, reader, metal), DiagnosticCode::unsupported_capability, "Missing SIMD capability rejected");
    metal = profile(); std::get<SubgroupInfo>(p.operations[0].info).width = 16;
    rejected(passes::legalize_nontexture(p, reader, metal), DiagnosticCode::unsupported_modifier, "Unsupported subgroup width rejected");
    for (auto numerator : {word(7), word(0x80000000)}) {
        p = empty(); result = emit(p, Opcode::divide, {numerator, word(numerator.bits == 7 ? 0 : UINT32_MAX)}, s32);
        returns(p, {result}, {ValueType::word32});
        success(passes::cleanup(p), "Exceptional divide is not folded to a fallback");
        check(p.operations.size() == 1 && p.operations[0].opcode == Opcode::divide, "Exceptional integer division preserved for diagnosis");
        rejected(passes::legalize_nontexture(p, reader, metal), DiagnosticCode::unsupported_modifier, "Undefined integer divide diagnosed");
    }
}
void production_bridge() {
    Facts source; FactReader reader(source);
    auto p = empty();
    auto result = emit(p, Opcode::add, {word(UINT32_MAX), word(3)});
    returns(p, {result}, {ValueType::word32});
    auto legal = legalize({p}, reader, profile());
    const auto* output = std::get_if<LegalProgram>(&legal);
    check(output && output->ir.operations.empty() && returned_literal(output->ir) &&
        returned_literal(output->ir)->bits == 2, "Lane 10 invokes the production cleanup pass");
    p = empty(); result = emit(p, Opcode::copy, {ConstantWord{0, 0}});
    returns(p, {result}, {ValueType::word32});
    auto unknown = legalize({p}, reader, profile());
    const auto* runtime = std::get_if<LegalProgram>(&unknown);
    check(runtime && runtime->ir.dependencies.empty() && runtime->requirements.uses.size() == 2 &&
        std::any_of(runtime->ir.operations.begin(), runtime->ir.operations.end(), [](const auto& operation) {
            return operation.opcode == Opcode::load &&
                std::get<MemoryInfo>(operation.info).space == AddressSpace::constant_buffer;
        }), "Cleanup preserves demanded CB words as guarded runtime loads without binding facts");
}
void invalid_and_limits() {
    auto invalid_compare = empty();
    auto comparison = emit(invalid_compare, Opcode::compare, {word(1), word(2)}, u32,
        ValueType::predicate, ComparisonInfo{static_cast<Compare>(255), false});
    returns(invalid_compare, {comparison}, {ValueType::predicate});
    rejected(passes::cleanup(invalid_compare), DiagnosticCode::unsupported_modifier, "Unknown comparison is not folded to false");
    check(invalid_compare.operations[0].opcode == Opcode::compare, "Invalid comparison diagnosis preserves original IR");
    for (const auto opcode : {Opcode::convert, Opcode::bitcast}) {
        auto invalid_conversion = empty();
        auto converted = emit(invalid_conversion, opcode, {word(1)}, opcode == Opcode::bitcast ? NumericMode{} : u32,
            ValueType::word32, ConversionInfo{static_cast<ScalarType>(255), ScalarType::u32});
        returns(invalid_conversion, {converted}, {ValueType::word32});
        rejected(passes::cleanup(invalid_conversion), DiagnosticCode::unsupported_modifier, "Unknown conversion scalar is rejected");
        Facts source; FactReader reader(source);
        rejected(passes::legalize_nontexture(invalid_conversion, reader, profile()), DiagnosticCode::unsupported_modifier,
            "Unknown conversion scalar cannot acquire a default width");
    }
    auto p = loop(); p.blocks[0].phis[0].inputs.pop_back();
    rejected(passes::cleanup(p), DiagnosticCode::invalid_ir, "Malformed phi rejected before mutation");
    check(p.operations.size() == 1 && p.blocks[0].phis[0].inputs.size() == 1, "Failed cleanup preserves caller program");
    p = empty(); auto result = emit(p, Opcode::copy, {Undefined{}}); returns(p, {result}, {ValueType::word32});
    rejected(passes::cleanup(p), DiagnosticCode::undefined_read, "Undefined is not replaced by zero");
    p = loop(); p.limits.analysis_steps = 1;
    rejected(passes::cleanup(p), DiagnosticCode::limit_exceeded, "Analysis budget enforced");
    p = empty(); auto input = parameter(p);
    for (unsigned i = 0; i < 30; ++i) input = emit(p, Opcode::copy, {input});
    returns(p, {input}, {ValueType::word32});
    bool saw_pass_limit = false;
    for (std::uint64_t limit = 100; limit <= 1000 && !saw_pass_limit; limit += 50) {
        p.limits.analysis_steps = limit;
        if (std::holds_alternative<Diagnostic>(validate(p, IrForm::ssa, Phase::legalize))) continue;
        auto attempt = p; auto status = passes::cleanup(attempt);
        if (const auto* error = std::get_if<Diagnostic>(&status)) {
            saw_pass_limit = error->code == DiagnosticCode::limit_exceeded;
            check(attempt.operations.size() == p.operations.size(), "Budget rejection is transactional");
        }
    }
    check(saw_pass_limit, "Pass work is bounded separately from structural validation");
}
} // namespace
int main() {
    integer_folding(); copies_and_modes(); phis_and_dead_code(); branches(); legalization(); production_bridge(); invalid_and_limits();
    if (failures) std::cerr << failures << " pipeline pass checks failed\n";
    else std::cout << "All pipeline pass checks passed\n";
    return failures ? 1 : 0;
}
