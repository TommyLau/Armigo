// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/frontend.h>
#include <armigo/shader_native/instruction.h>
#include "arithmetic.h"
#include <armigo/shader/vertex_fetch.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <map>
#include <set>

namespace armigo::shader_native {
namespace {
struct Failure { ir::Diagnostic diagnostic; };
std::string opcode_name(unsigned opcode) {
    constexpr char digits[] = "0123456789abcdef";
    std::string name = "0x";
    for (int shift = 8; shift >= 0; shift -= 4) name += digits[(opcode >> shift) & 15];
    return name;
}
class Builder {
public:
    Builder(const Program& source, ir::Limits limits, PixelRateState rate = PixelRateState::unknown) : source_(source), rate_(rate) {
        p_.stage = source.stage == Stage::vertex ? ir::Stage::vertex : ir::Stage::fragment;
        p_.limits = limits; p_.entry = ir::FunctionId{0};
        p_.functions.push_back({0, {}, {}, {}, {}});
    }
    ir::RegisterProgram run();
    bool requires_disabled_rate() const { return requires_disabled_rate_; }
private:
    const Program& source_;
    PixelRateState rate_;
    bool requires_disabled_rate_{};
    ir::Program p_;
    std::map<std::uint32_t, ir::BlockId> blocks_;
    std::deque<std::uint32_t> pending_;
    ir::BlockId current_;
    std::uint32_t pc_{};
    unsigned opcode_{};
    [[noreturn]] void fail(std::string message, ir::DiagnosticCode code = ir::DiagnosticCode::unsupported_instruction) const {
        throw Failure{ir::diagnose(ir::Phase::emit_ir, code, {pc_, {}},
            "Native " + opcode_name(opcode_) + ": " + std::move(message))};
    }
    ir::BlockId extra();
    ir::BlockId block(std::uint32_t pc);
    ir::Operand reg(unsigned index) const {
        return index == 255 ? ir::Operand{ir::LiteralBits{0, ir::ValueType::word32}} : ir::Operand{ir::Register{ir::RegisterFile::gpr, static_cast<std::uint16_t>(index)}};
    }
    ir::Operand temporary(ir::Opcode opcode, std::vector<ir::Operand> inputs,
                          ir::NumericMode mode = {}, ir::OperationInfo info = {}, ir::ValueType type = ir::ValueType::word32);
    void operation(ir::Operation op);
    void write(unsigned destination, ir::Operand value);
    ir::Operand predicate(Predicate guard);
    ir::Operand condition(const DecodedInstruction& instruction);
    void guard(ir::Operand condition, ir::BlockId yes, ir::BlockId no);
    ir::Operand operand(const Operand& input);
    void exports();
    void attribute(Instruction raw);
    void arithmetic(Instruction raw);
    void constant_load(Instruction raw);
    void compare_predicates(Instruction raw);
    void interpolate(Instruction raw);
    void instruction(Instruction raw);
    void check_definitions();
};
ir::BlockId Builder::extra() {
    if (p_.blocks.size() >= p_.limits.blocks) fail("block limit exceeded", ir::DiagnosticCode::limit_exceeded);
    ir::BlockId id{static_cast<std::uint32_t>(p_.blocks.size())};
    p_.blocks.push_back({ir::FunctionId{0}, {pc_, {}}, {}, {}, {}, ir::Unreachable{}});
    p_.functions[0].blocks.push_back(id); return id;
}
ir::BlockId Builder::block(std::uint32_t pc) {
    if (pc >= source_.instructions.size() || pc % 16) fail("control flow leaves the instruction image");
    if (auto found = blocks_.find(pc); found != blocks_.end()) return found->second;
    auto id = extra(); blocks_.emplace(pc, id); pending_.push_back(pc); return id;
}
void Builder::operation(ir::Operation op) {
    if (op.numeric.domain == ir::Domain::floating) {
        const bool arithmetic = op.opcode == ir::Opcode::add || op.opcode == ir::Opcode::multiply ||
            op.opcode == ir::Opcode::fused_multiply_add || op.opcode == ir::Opcode::reciprocal ||
            op.opcode == ir::Opcode::reciprocal_sqrt || op.opcode == ir::Opcode::square_root;
        if ((arithmetic && (!op.numeric.flush_inputs || !op.numeric.flush_output)) ||
            (op.opcode == ir::Opcode::compare && !op.numeric.flush_inputs))
            fail("non-FTZ floating semantics require a subnormal-preserving backend", ir::DiagnosticCode::unsupported_capability);
    }
    if (p_.operations.size() >= p_.limits.operations) fail("operation limit exceeded", ir::DiagnosticCode::limit_exceeded);
    op.source = {pc_, {}}; op.effects = ir::required_effects(op);
    p_.blocks[current_.value].operations.push_back(ir::OperationId{static_cast<std::uint32_t>(p_.operations.size())});
    p_.operations.push_back(std::move(op));
}
ir::Operand Builder::temporary(ir::Opcode opcode, std::vector<ir::Operand> inputs, ir::NumericMode mode, ir::OperationInfo info, ir::ValueType type) {
    if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
    ir::ValueId id{static_cast<std::uint32_t>(p_.values.size())}; p_.values.push_back({type});
    operation({{}, opcode, {}, mode, std::move(inputs), {id}, std::move(info)}); return id;
}
void Builder::write(unsigned destination, ir::Operand value) {
    if (destination == 255) return;
    operation({{}, ir::Opcode::copy, {}, {}, {std::move(value)}, {ir::Register{ir::RegisterFile::gpr, static_cast<std::uint16_t>(destination)}}, {}});
}
ir::Operand Builder::predicate(Predicate g) {
    if (g.file != PredicateFile::general) fail("uniform predicate semantics are not admitted");
    if (g.index == 7) return ir::LiteralBits{g.inverted ? 0u : 1u, ir::ValueType::predicate};
    ir::Operand value = ir::Register{ir::RegisterFile::predicate, g.index};
    if (g.inverted) value = temporary(ir::Opcode::logical_not, {value}, {ir::Domain::boolean, ir::ScalarType::boolean}, {}, ir::ValueType::predicate);
    return value;
}
ir::Operand Builder::condition(const DecodedInstruction& instruction) {
    auto a = predicate(instruction.execution);
    if (!instruction.condition) return a;
    auto b = predicate(*instruction.condition);
    if (auto literal = std::get_if<ir::LiteralBits>(&a)) return literal->bits ? b : a;
    if (auto literal = std::get_if<ir::LiteralBits>(&b)) return literal->bits ? a : b;
    return temporary(ir::Opcode::logical_and, {a, b}, {ir::Domain::boolean, ir::ScalarType::boolean}, {}, ir::ValueType::predicate);
}
void Builder::guard(ir::Operand condition, ir::BlockId yes, ir::BlockId no) {
    if (const auto* literal = std::get_if<ir::LiteralBits>(&condition); literal && literal->bits)
        p_.blocks[current_.value].terminator = ir::Jump{yes};
    else p_.blocks[current_.value].terminator = ir::Branch{std::move(condition), yes, no};
}
ir::Operand Builder::operand(const Operand& input) {
    if (auto immediate = std::get_if<Immediate>(&input)) return ir::LiteralBits{immediate->bits, ir::ValueType::word32};
    if (auto r = std::get_if<Register>(&input)) {
        if (r->file != RegisterFile::general) fail("uniform register semantics are not admitted");
        return reg(r->index);
    }
    auto c = std::get<Constant>(input);
    if (c.bank >= 18) fail("constant bank exceeds the established IR binding contract");
    return ir::ConstantWord{c.bank, c.byte_offset};
}
void Builder::exports() {
    if (source_.stage != Stage::fragment) return;
    const auto mask = little_endian(source_.header, 0x48, 4);
    if (little_endian(source_.header, 0x4c, 4)) fail("fragment depth/sample-mask export layout is not admitted");
    if (!(little_endian(source_.header, 0, 4) & (1u << 14)) && mask) fail("non-MRT color broadcast requires draw state");
    unsigned base = 0;
    for (unsigned target = 0; target < 8; ++target) {
        const unsigned components = (mask >> (target * 4)) & 15;
        if (!components) continue;
        for (unsigned c = 0; c < 4; ++c) if (components & (1u << c)) {
            ir::IoInfo info; info.semantic = ir::IoSemantic::color; info.location = target;
            info.component_mask = 1u << c; info.type = {ir::ScalarType::u32, 4};
            info.output_storage = true; info.data = {ir::InputIndex{0}};
            operation({{}, ir::Opcode::store_io, {}, {ir::Domain::unsigned_integer, ir::ScalarType::u32}, {reg(base + c)}, {}, info});
        }
        base += 4;
    }
}
void Builder::attribute(Instruction raw) {
    if (source_.stage != Stage::vertex) fail("attribute load/store requires the established vertex layout");
    const bool store = opcode_ == 0x322;
    const unsigned address = raw.bits(40, 10), count = raw.bits(74, 2) + 1;
    if (raw.bits(24, 8) != 255 || raw.bits(50, 14) || raw.bits(72, 2) || raw.bits(76, 20) || address % 4)
        fail("indexed, patch or modified attribute mode is not admitted");
    const unsigned start = store ? raw.bits(32, 8) : raw.bits(16, 8);
    const unsigned vertex = store ? raw.bits(64, 8) : raw.bits(32, 8);
    if (vertex != 255 || (!store && raw.bits(64, 8)) || (store && raw.bits(16, 8))) fail("attribute vertex selection is not admitted");
    if (start != 255 && start + count > 255) fail("attribute register span crosses RZ");
    if (store && address == 0x5c && count == 1) {
        if (rate_ != PixelRateState::disabled)
            fail("VPRS table-index output requires proven disabled pixel-rate state",
                 rate_ == PixelRateState::unknown ? ir::DiagnosticCode::missing_fact : ir::DiagnosticCode::unsupported_capability);
        requires_disabled_rate_ = true;
        return;
    }
    for (unsigned c = 0; c < count; ++c) {
        const unsigned at = address + 4 * c;
        ir::IoInfo info; info.type = {ir::ScalarType::u32, 4};
        if (at >= 0x70 && at < 0x80 && store) {
            info.semantic = ir::IoSemantic::position; info.type.scalar = ir::ScalarType::f32;
            info.component_mask = 1u << ((at - 0x70) / 4);
            if (!(std::to_integer<unsigned>(source_.header[0x35]) & (info.component_mask << 4))) fail("position store is absent from SPH output map");
        } else if (at >= 0x80 && at < 0x280) {
            info.semantic = ir::IoSemantic::generic; if (store) info.type.scalar = ir::ScalarType::f32; info.location = (at - 0x80) / 16; info.component_mask = 1u << ((at / 4) % 4);
            const auto map = std::to_integer<unsigned>(source_.header[(store ? 0x36 : 0x18) + info.location / 2]);
            if (!((map >> (4 * (info.location % 2))) & info.component_mask)) fail("attribute is absent from SPH map");
        } else if (store && at == 0x6c) {
            info.semantic = ir::IoSemantic::point_size; info.type = {ir::ScalarType::f32, 1}; info.component_mask = 1;
            if (!(std::to_integer<unsigned>(source_.header[0x35]) & 8)) fail("point size is absent from SPH output map");
        } else if (!store && (at == 0x2f8 || at == 0x2fc)) {
            info.semantic = at == 0x2fc ? ir::IoSemantic::vertex_id : ir::IoSemantic::instance_id;
            info.type = {ir::ScalarType::u32, 1}; info.component_mask = 1;
            const unsigned bit = at == 0x2fc ? 15 : 14;
            if (!(little_endian(source_.header, 0x2a, 2) & (1u << bit))) fail("system value is absent from SPH input map");
        } else fail("attribute semantic is not admitted");
        info.output_storage = store;
        if (store) {
            info.data = {ir::InputIndex{0}};
            const auto mode = info.type.scalar == ir::ScalarType::f32 ? ir::NumericMode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even} : ir::NumericMode{ir::Domain::unsigned_integer, ir::ScalarType::u32};
            operation({{}, ir::Opcode::store_io, {}, mode, {reg(start == 255 ? 255 : start + c)}, {}, info});
        } else {
            auto value = temporary(ir::Opcode::load_io, {}, {ir::Domain::unsigned_integer, ir::ScalarType::u32}, info);
            write(start == 255 ? 255 : start + c, value);
        }
    }
}
void Builder::arithmetic(Instruction raw) {
    auto decoded = decode_arithmetic(raw);
    if (auto error = std::get_if<std::string>(&decoded)) fail(*error);
    const auto& a = std::get<Arithmetic>(decoded);
    std::vector<ir::Operand> inputs;
    for (const auto& source : a.sources) {
        auto value = operand(source.value);
        const ir::NumericMode sign_mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
        if (source.absolute) value = temporary(ir::Opcode::absolute, {value}, sign_mode);
        if (source.negate) value = temporary(ir::Opcode::negate, {value}, sign_mode);
        inputs.push_back(value);
    }
    ir::Operand result;
    if (a.integer_multiply_add) {
        auto product = temporary(ir::Opcode::multiply, {inputs[0], inputs[1]}, a.numeric);
        result = temporary(ir::Opcode::add, {product, inputs[2]}, a.numeric);
    } else result = temporary(a.operation, std::move(inputs), a.numeric);
    write(a.destination, result);
}
void Builder::constant_load(Instruction raw) {
    const auto width = raw.bits(73, 3);
    if ((width != 4 && width != 5) || raw.words[2] != (width << 9) || (raw.words[1] & ~0x07ffffc0u))
        fail("indexed constant load width or address mode is not admitted");
    const auto bank = raw.bits(54, 5), index = raw.bits(24, 8), destination = raw.bits(16, 8);
    const unsigned elements = width == 5 ? 2 : 1;
    if (elements == 2 && ((destination & 1) || destination >= 254)) fail("paired constant load register range is not admitted");
    if (bank >= 18) fail("constant bank exceeds the established IR binding contract");
    const auto encoded = static_cast<std::uint32_t>(raw.bits(38, 16));
    const auto displacement = encoded & 0x8000 ? encoded | 0xffff0000u : encoded;
    const ir::NumericMode mode{ir::Domain::unsigned_integer, ir::ScalarType::u32};
    ir::MemoryInfo info; info.space = ir::AddressSpace::constant_buffer; info.access = ir::Access::read;
    info.constant_buffer_slot = bank; info.element_bytes = 4; info.elements = elements; info.alignment = 4;
    std::vector<ir::Operand> inputs;
    if (index == 255) info.byte_offset = displacement;
    else {
        auto address = reg(index);
        if (displacement) address = temporary(ir::Opcode::add, {address, ir::LiteralBits{displacement, ir::ValueType::word32}}, mode);
        info.dynamic_byte_offset = ir::InputIndex{0}; inputs.push_back(address);
    }
    // One vector load snapshots the address before either destination is
    // written, and the backend widens before adding component offsets.
    std::vector<ir::Operand> outputs;
    for (unsigned component = 0; component < elements; ++component) {
        if (p_.values.size() >= p_.limits.values) fail("constant load value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        outputs.push_back(ir::ValueId{static_cast<std::uint32_t>(p_.values.size())}); p_.values.push_back({ir::ValueType::word32});
    }
    operation({{}, ir::Opcode::load, {}, mode, std::move(inputs), outputs, info});
    for (unsigned component = 0; component < elements; ++component) write(destination + component, outputs[component]);
}
void Builder::compare_predicates(Instruction raw) {
    if (raw.bits(16, 8) || (raw.words[1] & ~0xc00000ffu) || (raw.words[2] & ~0x07ffff00u))
        fail("floating predicate source or extension mode is not admitted");
    const unsigned combine = raw.bits(74, 2), function = raw.bits(76, 4);
    const unsigned first = raw.bits(81, 3), second = raw.bits(84, 3);
    if (combine == 3 || (first == second && first != 7)) fail("floating predicate destination or combination mode is not admitted");
    const ir::NumericMode sign{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
    auto source = [&](unsigned at, unsigned absolute, unsigned negate) {
        auto value = reg(raw.bits(at, 8));
        if (raw.bits(absolute, 1)) value = temporary(ir::Opcode::absolute, {value}, sign);
        if (raw.bits(negate, 1)) value = temporary(ir::Opcode::negate, {value}, sign);
        return value;
    };
    const auto a = source(24, 73, 72), b = source(32, 62, 63);
    auto mode = sign; mode.flush_inputs = raw.bits(80, 1);
    const ir::NumericMode boolean{ir::Domain::boolean, ir::ScalarType::boolean};
    ir::Operand comparison;
    if (function == 0 || function == 15) comparison = ir::LiteralBits{function == 15 ? 1u : 0u, ir::ValueType::predicate};
    else if (function == 7 || function == 8) {
        auto nan_mode = mode; nan_mode.flush_inputs = false;
        auto left_nan = temporary(ir::Opcode::is_nan, {a}, nan_mode, {}, ir::ValueType::predicate);
        auto right_nan = temporary(ir::Opcode::is_nan, {b}, nan_mode, {}, ir::ValueType::predicate);
        comparison = temporary(ir::Opcode::logical_or, {left_nan, right_nan}, boolean, {}, ir::ValueType::predicate);
        if (function == 7) comparison = temporary(ir::Opcode::logical_not, {comparison}, boolean, {}, ir::ValueType::predicate);
    } else {
        constexpr ir::Compare comparisons[]{ir::Compare::never, ir::Compare::less, ir::Compare::equal,
            ir::Compare::less_equal, ir::Compare::greater, ir::Compare::not_equal, ir::Compare::greater_equal};
        comparison = temporary(ir::Opcode::compare, {a, b}, mode,
            ir::ComparisonInfo{comparisons[function & 7], function >= 9}, ir::ValueType::predicate);
    }
    auto complement = temporary(ir::Opcode::logical_not, {comparison}, boolean, {}, ir::ValueType::predicate);
    auto accumulator = predicate({PredicateFile::general, static_cast<std::uint8_t>(raw.bits(87, 3)), bool(raw.bits(90, 1))});
    const auto operation = combine == 0 ? ir::Opcode::logical_and : combine == 1 ? ir::Opcode::logical_or : ir::Opcode::logical_xor;
    // Snapshot both results before either architectural predicate is written.
    // The accumulator may alias either destination.
    auto primary = temporary(operation, {comparison, accumulator}, boolean, {}, ir::ValueType::predicate);
    auto secondary = temporary(operation, {complement, accumulator}, boolean, {}, ir::ValueType::predicate);
    if (first != 7) this->operation({{}, ir::Opcode::copy, {}, {}, {primary}, {ir::Register{ir::RegisterFile::predicate, static_cast<std::uint16_t>(first)}}, {}});
    if (second != 7) this->operation({{}, ir::Opcode::copy, {}, {}, {secondary}, {ir::Register{ir::RegisterFile::predicate, static_cast<std::uint16_t>(second)}}, {}});
}
void Builder::interpolate(Instruction raw) {
    if (source_.stage != Stage::fragment) fail("interpolation requires a fragment stage");
    const auto sample = raw.bits(76, 2);
    const bool constant = raw.bits(78, 1);
    if (raw.bits(24, 8) || raw.words[1] != 255 || sample > 1 ||
        (raw.words[2] & ~0x70ffu) != 0xe0000u)
        fail("indexed or modified interpolation mode is not admitted");
    const auto address = raw.bits(64, 8) * 4;
    const ir::NumericMode floating{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
    ir::IoInfo info;
    info.type = {ir::ScalarType::f32, 4};
    info.interpolation = {ir::InterpolationMode::flat, ir::InterpolationLocation::center};
    if (address >= 0x80 && address < 0x280) {
        info.semantic = ir::IoSemantic::generic; info.location = (address - 0x80) / 16;
        const auto component = (address / 4) % 4;
        info.component_mask = 1u << component;
        const auto mode = (std::to_integer<unsigned>(source_.header[0x18 + info.location]) >> (component * 2)) & 3;
        if (!mode || (constant && mode != 1)) fail("interpolation disagrees with the SPH component mode");
        info.interpolation.mode = mode == 1 ? ir::InterpolationMode::flat : mode == 2 ? ir::InterpolationMode::perspective : ir::InterpolationMode::linear;
        info.interpolation.location = sample ? ir::InterpolationLocation::centroid : ir::InterpolationLocation::center;
        info.interpolation_location_override = info.interpolation.location;
        if (mode == 2 && sample) fail("centroid perspective interpolation requires centroid-W evidence");
    } else if (address == 0x7c && !sample && !constant) {
        if (!(std::to_integer<unsigned>(source_.header[0x17]) & 0x80)) fail("PositionW is absent from the SPH input map");
        info.semantic = ir::IoSemantic::fragment_coord; info.component_mask = 8;
    } else fail("fragment coordinate or interpolation attribute is not admitted");
    auto value = temporary(ir::Opcode::load_io, {}, floating, info);
    if (info.semantic == ir::IoSemantic::generic && info.interpolation.mode == ir::InterpolationMode::perspective) {
        // Native PASS returns the plane value before the shader's explicit
        // reciprocal-W correction. Metal's perspective input is corrected.
        ir::IoInfo w; w.semantic = ir::IoSemantic::fragment_coord; w.component_mask = 8;
        w.type = {ir::ScalarType::f32, 4}; w.interpolation = {ir::InterpolationMode::flat, ir::InterpolationLocation::center};
        auto reciprocal_w = temporary(ir::Opcode::load_io, {}, floating, w);
        value = temporary(ir::Opcode::multiply, {value, reciprocal_w}, floating);
    }
    write(raw.bits(16, 8), value);
}
void Builder::instruction(Instruction raw) {
    opcode_ = raw.bits(0, 12);
    if (raw.words[3] & 0xc00001ffu) fail("reserved or extension instruction bits are not admitted");
    const auto guard_bits = raw.bits(12, 4);
    if (opcode_ == 0x321 || opcode_ == 0x322 || opcode_ == 0x424 || opcode_ == 0x624 || opcode_ == 0xb82 || opcode_ == 0x20b || opcode_ == 0x326 || arithmetic_opcode(opcode_)) {
        auto next = block(pc_ + 16);
        auto action = extra(); auto guard_value = predicate({PredicateFile::general, static_cast<std::uint8_t>(guard_bits & 7), bool(guard_bits & 8)});
        guard(guard_value, action, next); current_ = action;
        if (opcode_ == 0x326) interpolate(raw);
        else if (opcode_ == 0x20b) compare_predicates(raw);
        else if (opcode_ == 0xb82) constant_load(raw);
        else if (arithmetic_opcode(opcode_)) arithmetic(raw);
        else if (opcode_ == 0x321 || opcode_ == 0x322) attribute(raw);
        else {
            if (raw.bits(24, 8) != 255 || raw.words[2] != 0x078e00ff) fail("integer move is not the established zero-product form");
            Operand source = Immediate{raw.words[1]};
            if (opcode_ == 0x624) {
                if (raw.words[1] & ~0x07ffff00u) fail("integer move constant address mode is not admitted");
                source = Constant{static_cast<std::uint8_t>(raw.bits(54, 5)), static_cast<std::uint16_t>(raw.bits(38, 16))};
            }
            write(raw.bits(16, 8), operand(source));
        }
        p_.blocks[current_.value].terminator = ir::Jump{next}; return;
    }
    auto decoded = decode_instruction(raw, pc_, source_.instructions.size());
    if (auto failure = std::get_if<DecodeFailure>(&decoded)) fail(failure->message);
    const auto& value = std::get<DecodedInstruction>(decoded);
    if (value.operation == Operation::uniform_move) fail("uniform moves require a uniform-state lowering contract");
    if (value.operation == Operation::exit) {
        auto test = condition(value);
        if (auto literal = std::get_if<ir::LiteralBits>(&test)) {
            if (literal->bits) { exports(); p_.blocks[current_.value].terminator = ir::Return{}; }
            else p_.blocks[current_.value].terminator = ir::Jump{block(pc_ + 16)};
        } else {
            auto yes = extra(); auto no = block(pc_ + 16); guard(test, yes, no); current_ = yes;
            exports(); p_.blocks[current_.value].terminator = ir::Return{};
        }
    } else if (value.operation == Operation::branch) {
        auto test = condition(value);
        if (auto literal = std::get_if<ir::LiteralBits>(&test))
            p_.blocks[current_.value].terminator = ir::Jump{block(literal->bits ? *value.branch_target : pc_ + 16)};
        else { auto yes = block(*value.branch_target); auto no = block(pc_ + 16); guard(test, yes, no); }
    } else {
        auto next = block(pc_ + 16);
        if (value.operation == Operation::move) {
            auto action = extra(); auto test = predicate(value.execution); guard(test, action, next); current_ = action;
            write(value.destination->index, operand(*value.source));
        }
        p_.blocks[current_.value].terminator = ir::Jump{next};
    }
}
void Builder::check_definitions() {
    // Native entry register contents are not invented by the shared SSA phase.
    using Set = std::bitset<263>;
    auto index = [](ir::Register r) -> unsigned { return r.file == ir::RegisterFile::gpr ? r.index : 256 + r.index; };
    std::vector<Set> in(p_.blocks.size()), out(p_.blocks.size());
    for (auto& s : out) s.set();
    bool changed = true; std::uint64_t steps = 0;
    while (changed) {
        changed = false;
        for (unsigned b = 0; b < p_.blocks.size(); ++b) {
            if (++steps > p_.limits.analysis_steps) fail("definedness analysis limit exceeded", ir::DiagnosticCode::limit_exceeded);
            Set incoming;
            if (b != p_.functions[0].entry.value) {
                incoming.set(); for (auto pred : p_.blocks[b].predecessors) incoming &= out[pred.value];
            }
            in[b] = incoming;
            for (auto op : p_.blocks[b].operations) for (auto& output : p_.operations[op.value].outputs)
                if (auto r = std::get_if<ir::Register>(&output)) incoming.set(index(*r));
            if (out[b] != incoming) { out[b] = incoming; changed = true; }
        }
    }
    for (unsigned b = 0; b < p_.blocks.size(); ++b) {
        auto defined = in[b];
        auto check = [&](const ir::Operand& input) {
            if (auto r = std::get_if<ir::Register>(&input); r && !defined[index(*r)])
                fail("entry or path-dependent register value is not established", ir::DiagnosticCode::undefined_read);
        };
        for (auto id : p_.blocks[b].operations) {
            pc_ = p_.operations[id.value].source.pc.value_or(0);
            for (auto& input : p_.operations[id.value].inputs) check(input);
            for (auto& output : p_.operations[id.value].outputs) if (auto r = std::get_if<ir::Register>(&output)) defined.set(index(*r));
        }
        if (auto branch = std::get_if<ir::Branch>(&p_.blocks[b].terminator)) check(branch->condition);
    }
}
ir::RegisterProgram Builder::run() {
    if (source_.stage == Stage::compute) fail("compute launch facts are not established");
    if (source_.stage != Stage::vertex && source_.stage != Stage::fragment) fail("stage semantics are not admitted");
    if (source_.header.size() != 128 || source_.instructions.empty() || source_.instructions.size() % 16 || source_.instructions.size() > p_.limits.code_bytes || source_.instructions.size() / 16 > p_.limits.instructions)
        fail("invalid graphics image", ir::DiagnosticCode::malformed_input);
    const auto header = little_endian(source_.header, 0, 4);
    if ((header & 31) != (source_.stage == Stage::vertex ? 1u : 2u) || ((header >> 10) & 15) != (source_.stage == Stage::vertex ? 1u : 5u) || ((header >> 5) & 31) != 4)
        fail("SPH family, version or stage is not admitted");
    p_.functions[0].entry = block(0);
    while (!pending_.empty()) {
        pc_ = pending_.front(); pending_.pop_front(); current_ = blocks_.at(pc_);
        instruction(Instruction::read(source_.instructions.subspan(pc_, 16)));
    }
    for (std::uint32_t i = 0; i < p_.blocks.size(); ++i) for (auto next : ir::successors(p_.blocks[i].terminator))
        p_.blocks[next.value].predecessors.push_back(ir::BlockId{i});
    check_definitions();
    if (auto checked = ir::validate(p_, ir::IrForm::registers, ir::Phase::emit_ir); auto error = std::get_if<ir::Diagnostic>(&checked)) throw Failure{*error};
    return {std::move(p_)};
}
class NativeFacts final : public ir::FactSource {
public:
    NativeFacts(const Program& p, FrontendOptions options) : p_(p), options_(options) {}
    ir::Knowledge<ir::FactValue> lookup(ir::FactKey key) const override {
        if (key.kind == ir::FactKind::depth_convention && options_.vertex_depth)
            return ir::Known<ir::FactValue>{*options_.vertex_depth, {ir::OriginKind::explicit_option, "native-depth-option-v1"}};
        if (key.kind == ir::FactKind::fragment_output_type && p_.stage == Stage::fragment)
            return ir::Known<ir::FactValue>{ir::IoType{ir::ScalarType::u32, 4}, {ir::OriginKind::derived, "native-raw-export-adaptive-target-v1"}};
        if (key.kind == ir::FactKind::input_interpolation && p_.stage == Stage::fragment && key.index < 32 && key.component < 4) {
            const auto mode = (std::to_integer<unsigned>(p_.header[0x18 + key.index]) >> (key.component * 2)) & 3;
            if (mode) return ir::Known<ir::FactValue>{ir::Interpolation{
                mode == 1 ? ir::InterpolationMode::flat : mode == 2 ? ir::InterpolationMode::perspective : ir::InterpolationMode::linear,
                ir::InterpolationLocation::center}, {ir::OriginKind::validated_header, "native-sph-v4-interpolation"}};
        }
        return ir::Unknown{"Native binding or specialization fact is not established"};
    }
private: const Program& p_; FrontendOptions options_;
};
template<class T> T checked(ir::Result<T> value) {
    if (auto error = std::get_if<ir::Diagnostic>(&value)) throw Failure{*error};
    return std::get<T>(std::move(value));
}
}
ir::Result<ir::RegisterProgram> emit_native_ir(const Program& program, ir::Limits limits) {
    try { return Builder{program, limits}.run(); } catch (const Failure& failure) { return failure.diagnostic; }
}
ir::Result<ir::MslArtifact> translate_native(const Program& program, const FrontendOptions& options) {
    try {
        ir::Limits limits;
        NativeFacts facts{program, options}; ir::FactReader reader{facts, limits};
        Builder builder{program, limits, options.pixel_rate};
        auto registers = builder.run();
        auto ssa = checked(ir::construct_ssa(std::move(registers), limits));
        ir::MetalProfile profile{builder.requires_disabled_rate() ? std::string(disabled_pixel_rate_profile) : "armigo-native-metal4", 2, 31, 128, 16, true, false, true, false, false, false};
        auto legal = checked(ir::legalize(std::move(ssa), reader, profile));
        auto structured = checked(ir::structure(std::move(legal), limits));
        auto plan = checked(ir::plan_resources(structured, profile));
        if (options.vertex_fetch && program.stage == Stage::vertex && std::any_of(plan.io.begin(), plan.io.end(), [](const auto& io) { return io.semantic == ir::IoSemantic::generic && io.access == ir::Access::read; })) {
            for (const auto& io : plan.io)
                if (io.semantic == ir::IoSemantic::generic && io.access == ir::Access::read && io.location >= armigo::shader::vertex_fetch_attributes)
                    throw Failure{ir::diagnose(ir::Phase::plan_resources, ir::DiagnosticCode::unsupported_capability, {}, "Native attribute exceeds the host vertex fetch table")};
            std::array<bool, 31> occupied{};
            for (auto& resource : plan.resources) if (auto range = std::get_if<ir::MetalRange>(&resource.location); range && range->space == ir::MetalNamespace::buffer)
                for (unsigned i = 0; i < range->count; ++i) occupied.at(range->first + i) = true;
            if (plan.argument_buffer) occupied.at(plan.argument_buffer->buffer_index) = true;
            if (plan.draw_support_buffer_index) occupied.at(*plan.draw_support_buffer_index) = true;
            auto free = std::find(occupied.begin(), occupied.end(), false);
            if (free == occupied.end()) throw Failure{ir::diagnose(ir::Phase::plan_resources, ir::DiagnosticCode::limit_exceeded, {}, "Native vertex fetch has no free buffer slot")};
            plan.vertex_fetch_buffer_index = free - occupied.begin();
        }
        ir::EmitOptions emit;
        emit.fragment_output_types_from_render_targets = program.stage == Stage::fragment;
        emit.fragment_position = options.fragment_position;
        auto artifact = checked(ir::emit_msl(structured, plan, emit));
        // Keep the shared Maxwell emitter unchanged. Native instruction
        // boundaries require separate rounding even for admitted FTZ forms.
        const auto replace_policy = [&](std::string_view old_policy, std::string_view new_policy) {
            const auto at = artifact.source.find(old_policy);
            if (at == std::string::npos || artifact.source.find(old_policy, at + old_policy.size()) != std::string::npos)
                throw Failure{ir::diagnose(ir::Phase::emit_msl, ir::DiagnosticCode::unsupported_capability, {}, "Native numeric prelude boundary changed")};
            artifact.source.replace(at, old_policy.size(), new_policy);
        };
        replace_policy("#pragma METAL fp math_mode(relaxed)", "#pragma METAL fp math_mode(safe)");
        replace_policy("#pragma METAL fp contract(fast)", "#pragma METAL fp contract(off)");
        return artifact;
    } catch (const Failure& failure) { return failure.diagnostic; }
}
} // namespace armigo::shader_native
