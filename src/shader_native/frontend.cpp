// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "builder.h"
#include "mesh_stage.h"
#include "arithmetic.h"
#include <armigo/shader/vertex_fetch.h>
#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <set>

namespace armigo::shader_native {
namespace detail {
std::string opcode_name(unsigned opcode) {
    constexpr char digits[] = "0123456789abcdef";
    std::string name = "0x";
    for (int shift = 8; shift >= 0; shift -= 4) name += digits[(opcode >> shift) & 15];
    return name;
}
Fields::Fields(Instruction raw) : raw_(raw) { take(0, 16); take(105, 21); }
std::uint64_t Fields::take(unsigned first, unsigned count) {
    const auto value = raw_.bits(first, count);
    for (unsigned bit = first; bit < first + count; ++bit) read_[bit / 32] |= 1u << (bit % 32);
    return value;
}
bool Fields::unread() const {
    for (unsigned i = 0; i < 4; ++i) if (raw_.words[i] & ~read_[i]) return true;
    return false;
}
Builder::Builder(const Program& source, ir::Limits limits, PixelRateState rate, bool raw_vertex_inputs)
    : source_(source), rate_(rate), raw_vertex_inputs_(raw_vertex_inputs) {
    // Task and mesh stages translate as compute: their body runs in a Metal
    // object or mesh function as a workgroup does (mesh_stage.h).
    p_.stage = source.stage == Stage::vertex ? ir::Stage::vertex : source.stage == Stage::fragment ? ir::Stage::fragment : ir::Stage::compute;
    p_.limits = limits; p_.entry = ir::FunctionId{0};
    p_.functions.push_back({0, {}, {}, {}, {}});
}
void Builder::fail(std::string message, ir::DiagnosticCode code) const {
    throw Failure{ir::diagnose(ir::Phase::emit_ir, code, {pc_, {}},
        "Native " + opcode_name(opcode_) + ": " + std::move(message))};
}
ir::Operand Builder::reg(unsigned index) const {
    return index == 255 ? ir::Operand{ir::LiteralBits{0, ir::ValueType::word32}} : ir::Operand{ir::Register{ir::RegisterFile::gpr, static_cast<std::uint16_t>(index)}};
}
ir::BlockId Builder::extra() {
    if (p_.blocks.size() >= p_.limits.blocks) fail("block limit exceeded", ir::DiagnosticCode::limit_exceeded);
    ir::BlockId id{static_cast<std::uint32_t>(p_.blocks.size())};
    p_.blocks.push_back({ir::FunctionId{0}, {pc_, {}}, {}, {}, {}, ir::Unreachable{}});
    p_.functions[0].blocks.push_back(id); return id;
}
ir::BlockId Builder::block(std::uint32_t pc) { return block_in(pc, context_); }
ir::BlockId Builder::block_in(std::uint32_t pc, std::uint32_t context) {
    if (pc >= source_.instructions.size() || pc % 16) fail("control flow leaves the instruction image");
    if (auto found = blocks_.find({pc, context}); found != blocks_.end()) return found->second;
    auto id = extra(); blocks_.emplace(std::pair{pc, context}, id); pending_.push_back({pc, context}); return id;
}
// The context a call from the current one opens: its callee and the
// instruction its RET returns to. Recursion and deep chains are refused.
std::uint32_t Builder::call_context(std::uint32_t callee, std::uint32_t return_pc) {
    unsigned depth = 0;
    for (auto at = context_; at; at = contexts_[at].parent, ++depth)
        if (contexts_[at].callee == callee) fail("recursive native call is not admitted");
    if (depth >= 8) fail("native call chain deeper than eight is not admitted");
    const auto key = std::tuple{context_, callee, return_pc};
    if (auto found = context_ids_.find(key); found != context_ids_.end()) return found->second;
    contexts_.push_back({context_, callee, return_pc});
    const auto id = static_cast<std::uint32_t>(contexts_.size() - 1);
    context_ids_.emplace(key, id);
    return id;
}
void Builder::operation(ir::Operation op) {
    if (op.numeric.domain == ir::Domain::floating) {
        const bool arithmetic = op.opcode == ir::Opcode::add || op.opcode == ir::Opcode::multiply ||
            op.opcode == ir::Opcode::fused_multiply_add || op.opcode == ir::Opcode::reciprocal ||
            op.opcode == ir::Opcode::reciprocal_sqrt || op.opcode == ir::Opcode::square_root ||
            op.opcode == ir::Opcode::exp2 || op.opcode == ir::Opcode::log2;
        if (!metal_flush_ && ((arithmetic && (!op.numeric.flush_inputs || !op.numeric.flush_output)) ||
            (op.opcode == ir::Opcode::compare && !op.numeric.flush_inputs)))
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
    if (g.file != PredicateFile::general) return uniform_predicate(g.index, g.inverted);
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
    if (auto r = std::get_if<Register>(&input)) return r->file == RegisterFile::general ? reg(r->index) : uniform(r->index);
    auto c = std::get<Constant>(input);
    return constant(c.bank, c.byte_offset);
}
ir::Operand Builder::constant(unsigned bank, unsigned byte_offset) {
    if (bank >= 18) fail("constant bank exceeds the established IR binding contract");
    if (byte_offset % 4 || byte_offset > 0xfffc) fail("constant word address is not admitted");
    return ir::ConstantWord{static_cast<std::uint8_t>(bank), static_cast<std::uint16_t>(byte_offset)};
}
ir::ValueId Builder::local(std::optional<ir::ValueId>& slot, ir::ValueType type) {
    if (!slot) {
        if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        slot = ir::ValueId{static_cast<std::uint32_t>(p_.values.size())};
        p_.values.push_back({type, ir::ValueKind::mutable_local});
    }
    return *slot;
}
ir::Operand Builder::uniform(unsigned index) {
    if (index == 63) return ir::LiteralBits{0, ir::ValueType::word32};
    if (index > 63) fail("uniform register index is not admitted");
    return local(uniform_registers_[index], ir::ValueType::word32);
}
void Builder::write_uniform(unsigned index, ir::Operand value) {
    if (index == 63) return;
    if (index > 63) fail("uniform register index is not admitted");
    operation({{}, ir::Opcode::copy, {}, {}, {std::move(value)}, {local(uniform_registers_[index], ir::ValueType::word32)}, {}});
}
ir::Operand Builder::uniform_predicate(unsigned index, bool inverted) {
    if (index == 7) return ir::LiteralBits{inverted ? 0u : 1u, ir::ValueType::predicate};
    ir::Operand value = local(uniform_predicates_[index], ir::ValueType::predicate);
    if (inverted) value = temporary(ir::Opcode::logical_not, {value}, {ir::Domain::boolean, ir::ScalarType::boolean}, {}, ir::ValueType::predicate);
    return value;
}
void Builder::write_uniform_predicate(unsigned index, ir::Operand value) {
    if (index == 7) return;
    operation({{}, ir::Opcode::copy, {}, {}, {std::move(value)}, {local(uniform_predicates_[index], ir::ValueType::predicate)}, {}});
}
void Builder::write_predicate(unsigned index, ir::Operand value) {
    if (index == 7) return;
    operation({{}, ir::Opcode::copy, {}, {}, {std::move(value)}, {ir::Register{ir::RegisterFile::predicate, static_cast<std::uint16_t>(index)}}, {}});
}
ir::Operand Builder::predicate_source(Fields& fields, unsigned first, bool uniform) {
    const auto index = static_cast<std::uint8_t>(fields.take(first, 3));
    const bool inverted = fields.flag(first + 3);
    return uniform ? uniform_predicate(index, inverted) : predicate({PredicateFile::general, index, inverted});
}
void Builder::enter(const Fields& fields, bool uniform_guard) {
    const auto bits = fields.raw().bits(12, 4);
    const Predicate guard_predicate{uniform_guard ? PredicateFile::uniform : PredicateFile::general,
                                    static_cast<std::uint8_t>(bits & 7), (bits & 8) != 0};
    next_ = block(pc_ + 16);
    auto test = predicate(guard_predicate);
    if (const auto* literal = std::get_if<ir::LiteralBits>(&test); literal && literal->bits) return;
    auto action = extra(); guard(test, action, *next_); current_ = action;
}
void Builder::enter_transfer(const Fields& fields) {
    if (fields.raw().bits(12, 4) != 7) enter(fields, false);
}
void Builder::leave(const Fields& fields) {
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
    p_.blocks[current_.value].terminator = ir::Jump{*next_};
}
void Builder::exports() {
    if (source_.stage != Stage::fragment) return;
    const auto mask = little_endian(source_.header, 0x48, 4);
    // Word 0x4c: the sample mask (bit 0) and depth (bit 1) follow the color
    // registers, the sample mask in the next register and depth in the one
    // after it, as the Maxwell header lays them out. Each target with any
    // component enabled takes four registers, however few it enables, so
    // depth is always register 4 * targets + 1.
    const auto extra = little_endian(source_.header, 0x4c, 4);
    if (extra & ~3u) fail("fragment export word holds bits other than the sample mask and depth");
    const bool sample_mask = extra & 1u, depth = extra & 2u;
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
    if (sample_mask) {
        ir::IoInfo info; info.semantic = ir::IoSemantic::sample_mask; info.type = {ir::ScalarType::u32, 1};
        info.output_storage = true; info.data = {ir::InputIndex{0}};
        operation({{}, ir::Opcode::store_io, {}, {ir::Domain::unsigned_integer, ir::ScalarType::u32}, {reg(base)}, {}, info});
    }
    if (depth) {
        ir::IoInfo info; info.semantic = ir::IoSemantic::fragment_depth; info.type = {ir::ScalarType::f32, 1};
        info.output_storage = true; info.data = {ir::InputIndex{0}};
        operation({{}, ir::Opcode::store_io, {}, {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even},
                   {reg(base + 1)}, {}, info});
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
            // Shader-side vertex fetch rebuilds the guest attribute register
            // word for every format, so a generic input is raw bits whatever
            // its uses read it as; the bitwise pass-through (x | x, which the
            // shared folding keeps) makes it a raw word class for the shared
            // vertex classifier.
            if (raw_vertex_inputs_ && info.semantic == ir::IoSemantic::generic)
                value = temporary(ir::Opcode::bit_or, {value, value}, {ir::Domain::bits, ir::ScalarType::u32});
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
    if (a.zero_sum_toward_minus) {
        // RZ + b in a directed mode, in exact bit operations (the host would
        // fold an addition of -0 away, and its flush with it): FTZ flushes a
        // subnormal to its signed zero, a zero sum is +0 unless rounding
        // toward minus, and NaN becomes the canonical 0x7fffffff.
        const ir::NumericMode word_mode{ir::Domain::unsigned_integer, ir::ScalarType::u32};
        auto equal = [&](ir::Operand left, std::uint32_t right) {
            return temporary(ir::Opcode::compare, {std::move(left), ir::LiteralBits{right, ir::ValueType::word32}}, word_mode,
                             ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
        };
        auto word = [](std::uint32_t value) { return ir::Operand{ir::LiteralBits{value, ir::ValueType::word32}}; };
        auto value = inputs[0];
        auto magnitude = bits(ir::Opcode::bit_and, {value, word(0x7fffffffu)});
        auto sign = bits(ir::Opcode::bit_and, {value, word(0x80000000u)});
        if (a.numeric.flush_inputs)
            value = bits(ir::Opcode::select, {equal(bits(ir::Opcode::bit_and, {value, word(0x7f800000u)}), 0), sign, value});
        if (!*a.zero_sum_toward_minus) value = bits(ir::Opcode::select, {equal(value, 0x80000000u), word(0), value});
        auto nan = temporary(ir::Opcode::compare, {magnitude, word(0x7f800000u)}, word_mode,
                             ir::ComparisonInfo{ir::Compare::greater, false}, ir::ValueType::predicate);
        write(a.destination, bits(ir::Opcode::select, {nan, word(0x7fffffffu), value}));
        return;
    }
    if (a.operation == ir::Opcode::sine || a.operation == ir::Opcode::cosine) {
        // MUFU.SIN and MUFU.COS of y are sin(2 pi y) and cos(2 pi y): the
        // compiler multiplies a radian angle by 1/(2 pi) first. Whole turns
        // drop exactly (y - rint(y)), so only the remaining fraction is scaled
        // back to radians, keeping the error near the unit's own for large y.
        auto turns = temporary(ir::Opcode::subtract, {inputs[0], temporary(ir::Opcode::round, {inputs[0]}, a.numeric)}, a.numeric);
        inputs[0] = temporary(ir::Opcode::multiply, {turns, ir::LiteralBits{0x40c90fdbu, ir::ValueType::word32}}, a.numeric);
    }
    if (a.scale) {
        // The scale multiplies source A before the product and never saturates.
        auto mode = a.numeric; mode.saturate = false;
        inputs[0] = temporary(ir::Opcode::multiply, {inputs[0], ir::LiteralBits{*a.scale, ir::ValueType::word32}}, mode);
    }
    // FFMA without FTZ is admitted with Metal's flushing (a documented
    // difference for subnormal operands and results, as on Switch 1).
    metal_flush_ = a.operation == ir::Opcode::fused_multiply_add;
    write(a.destination, temporary(a.operation, std::move(inputs), a.numeric));
    metal_flush_ = false;
}
void Builder::constant_load(Instruction raw) {
    // Widths 0-3 read U8, S8, U16 and S16 into one register (zero- or
    // sign-extended, as the Maxwell sized loads); 4 and 5 read one or two words.
    const auto width = raw.bits(73, 3);
    if (width > 5 || raw.words[2] != (width << 9) || (raw.words[1] & ~0x07ffffc0u))
        fail("indexed constant load width or address mode is not admitted");
    const auto bank = raw.bits(54, 5), index = raw.bits(24, 8), destination = raw.bits(16, 8);
    const unsigned elements = width == 5 ? 2 : 1;
    if (elements == 2 && ((destination & 1) || destination >= 254)) fail("paired constant load register range is not admitted");
    if (bank >= 18) fail("constant bank exceeds the established IR binding contract");
    const auto encoded = static_cast<std::uint32_t>(raw.bits(38, 16));
    const auto displacement = encoded & 0x8000 ? encoded | 0xffff0000u : encoded;
    const ir::NumericMode mode{ir::Domain::unsigned_integer, ir::ScalarType::u32};
    constexpr ir::ScalarType narrow[]{ir::ScalarType::u8, ir::ScalarType::s8, ir::ScalarType::u16, ir::ScalarType::s16};
    const ir::NumericMode load_mode = width < 4
        ? ir::NumericMode{width % 2 ? ir::Domain::signed_integer : ir::Domain::unsigned_integer, narrow[width]} : mode;
    const std::uint8_t bytes = width < 2 ? 1 : width < 4 ? 2 : 4;
    ir::MemoryInfo info; info.space = ir::AddressSpace::constant_buffer; info.access = ir::Access::read;
    info.constant_buffer_slot = bank; info.element_bytes = bytes; info.elements = elements; info.alignment = bytes;
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
    operation({{}, ir::Opcode::load, {}, load_mode, std::move(inputs), outputs, info});
    for (unsigned component = 0; component < elements; ++component) write(destination + component, outputs[component]);
}
void Builder::interpolate(Instruction raw) {
    if (source_.stage != Stage::fragment) fail("interpolation requires a fragment stage");
    const auto sample = static_cast<unsigned>(raw.bits(76, 2));
    const bool constant = raw.bits(78, 1);
    const bool indexed = opcode_ == 0x526;
    if (raw.words[1] != 255 || sample > 1 || (raw.words[2] & ~0x70ffu) != (indexed ? 0x080e0000u : 0xe0000u) ||
        (!indexed && raw.bits(24, 8)) || (indexed && raw.bits(30, 2)))
        fail("indexed or modified interpolation mode is not admitted");
    const auto attribute = static_cast<unsigned>(raw.bits(64, 8)) * 4;
    const auto index = static_cast<unsigned>(raw.bits(24, 6));
    if (!indexed || index == 63) { write(raw.bits(16, 8), interpolated(attribute, sample, constant)); return; }
    // The attribute indexed by a uniform register (byte address URa plus the
    // attribute's): the generic input component the SPH enables at that
    // address, selected among all of them, or zero for any other address. A
    // constant index folds to the one component.
    if (constant) fail("indexed constant interpolation is not admitted");
    const ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
    auto address = temporary(ir::Opcode::add, {uniform(index), ir::LiteralBits{attribute, ir::ValueType::word32}}, unsigned_word);
    ir::Operand value = ir::LiteralBits{0, ir::ValueType::word32};
    for (unsigned location = 0; location < 32; ++location)
        for (unsigned component = 0; component < 4; ++component) {
            if (!((std::to_integer<unsigned>(source_.header[0x18 + location]) >> (component * 2)) & 3)) continue;
            const auto candidate = 0x80 + location * 16 + component * 4;
            auto selected = temporary(ir::Opcode::compare, {address, ir::LiteralBits{candidate, ir::ValueType::word32}}, unsigned_word,
                                      ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
            value = temporary(ir::Opcode::select, {selected, interpolated(candidate, sample, false), value}, {ir::Domain::bits, ir::ScalarType::u32});
        }
    write(raw.bits(16, 8), value);
}
ir::Operand Builder::interpolated(unsigned address, unsigned sample, bool constant) {
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
        // A perspective input at the centroid is rebuilt from Metal's
        // centroid-corrected value and the pixel-centre W below, the same W a
        // centroid read of PositionW returns, so the shader's own division by
        // that W gives back the centroid-corrected value.
    } else if ((address == 0x2e0 || address == 0x2e4) && !sample && !constant) {
        // The point coordinate, as the Maxwell attribute map defines it.
        info.semantic = ir::IoSemantic::point_coord; info.component_mask = 1u << ((address - 0x2e0) / 4);
        info.type = {ir::ScalarType::f32, 2};
    } else if (address >= 0x70 && address < 0x80 && !constant && (!sample || address == 0x7c)) {
        // PositionX-W. A centroid read of PositionW returns the pixel-centre
        // W (Metal exposes no centroid position), which differs from the
        // true centroid W only on partially covered pixels.
        const unsigned component = (address - 0x70) / 4;
        if (!(std::to_integer<unsigned>(source_.header[0x17]) & (0x10u << component)))
            fail("fragment position component is absent from the SPH input map");
        info.semantic = ir::IoSemantic::fragment_coord; info.component_mask = 1u << component;
    } else if (address == 0x3fc && !sample) {
        // Front facing, as the Maxwell attribute: a mask of all ones when the
        // primitive faces the viewer and zero otherwise.
        info.semantic = ir::IoSemantic::front_facing; info.component_mask = 1;
        info.type = {ir::ScalarType::boolean, 1};
        auto facing = temporary(ir::Opcode::load_io, {}, {ir::Domain::boolean, ir::ScalarType::boolean}, info, ir::ValueType::predicate);
        return temporary(ir::Opcode::predicate_to_mask, {facing}, {}, {}, ir::ValueType::word32);
    } else fail("fragment coordinate or interpolation attribute is not admitted");
    auto value = temporary(ir::Opcode::load_io, {}, floating, info);
    if (address == 0x74) {
        // Metal counts fragment rows from the target's first memory row. A
        // negated guest Y origin (the y_direction word is -1.0f) counts them
        // from the other edge: y = height - y, with the render-target height
        // read per draw. Pixel centres keep the half offset either way.
        ir::IoInfo direction; direction.semantic = ir::IoSemantic::y_direction; direction.type = {ir::ScalarType::u32, 1}; direction.component_mask = 1;
        ir::IoInfo height; height.semantic = ir::IoSemantic::render_target_height; height.type = {ir::ScalarType::u32, 1}; height.component_mask = 1;
        const ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
        auto negated = temporary(ir::Opcode::compare, {temporary(ir::Opcode::load_io, {}, unsigned_word, direction), ir::LiteralBits{0xbf800000u, ir::ValueType::word32}},
                                 unsigned_word, ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
        auto rows = temporary(ir::Opcode::convert, {temporary(ir::Opcode::load_io, {}, unsigned_word, height)}, floating,
                              ir::ConversionInfo{ir::ScalarType::u32, ir::ScalarType::f32});
        auto flushing = floating; flushing.flush_inputs = flushing.flush_output = true;
        auto flipped = temporary(ir::Opcode::subtract, {rows, value}, flushing);
        value = temporary(ir::Opcode::select, {negated, flipped, value}, {ir::Domain::bits, ir::ScalarType::u32});
    }
    if (info.semantic == ir::IoSemantic::generic && info.interpolation.mode == ir::InterpolationMode::perspective) {
        // Native PASS returns the plane value before the shader's explicit
        // reciprocal-W correction. Metal's perspective input is corrected.
        // The reconstruction flushes like the guest FTZ arithmetic that
        // consumes it.
        ir::IoInfo w; w.semantic = ir::IoSemantic::fragment_coord; w.component_mask = 8;
        w.type = {ir::ScalarType::f32, 4}; w.interpolation = {ir::InterpolationMode::flat, ir::InterpolationLocation::center};
        auto reciprocal_w = temporary(ir::Opcode::load_io, {}, floating, w);
        auto flushing = floating; flushing.flush_inputs = flushing.flush_output = true;
        value = temporary(ir::Opcode::multiply, {value, reciprocal_w}, flushing);
    }
    return value;
}
void Builder::instruction(Instruction raw) {
    opcode_ = raw.bits(0, 12);
    if (raw.words[3] & 0xc00001ffu) fail("reserved or extension instruction bits are not admitted");
    const auto guard_bits = raw.bits(12, 4);
    // An instruction that never executes falls through without splitting the
    // block around it.
    if (never_executes(raw)) { p_.blocks[current_.value].terminator = ir::Jump{block(pc_ + 16)}; return; }
    if (opcode_ == 0x321 || opcode_ == 0x322 || opcode_ == 0xb82 || opcode_ == 0x326 || opcode_ == 0x526 || arithmetic_opcode(opcode_)) {
        auto next = block(pc_ + 16);
        auto action = extra(); auto guard_value = predicate({PredicateFile::general, static_cast<std::uint8_t>(guard_bits & 7), bool(guard_bits & 8)});
        guard(guard_value, action, next); current_ = action;
        if (opcode_ == 0x326 || opcode_ == 0x526) interpolate(raw);
        else if (opcode_ == 0xb82) constant_load(raw);
        else if (arithmetic_opcode(opcode_)) arithmetic(raw);
        else attribute(raw);
        p_.blocks[current_.value].terminator = ir::Jump{next}; return;
    }
    Fields fields(raw);
    if (integer(fields) || uniform_operation(fields) || half_precision(fields) || conversion(fields) || flow(fields) || texture(fields) ||
            global_memory(fields) || system_value(fields) || local_memory(fields) || texture_gradient(fields) || shared_memory(fields) || tensor(fields) || surface(fields) || stage_buffer(fields)) return;
    auto decoded = decode_instruction(raw, pc_, source_.instructions.size());
    if (auto failure = std::get_if<DecodeFailure>(&decoded)) fail(failure->message);
    const auto& value = std::get<DecodedInstruction>(decoded);
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
        if (value.operation == Operation::move || value.operation == Operation::uniform_move) {
            // UMOV executes under its uniform predicate guard.
            auto action = extra(); auto test = predicate(value.execution); guard(test, action, next); current_ = action;
            if (value.operation == Operation::move) write(value.destination->index, operand(*value.source));
            else write_uniform(value.destination->index, operand(*value.source));
        }
        p_.blocks[current_.value].terminator = ir::Jump{next};
    }
}
// Every instruction opens a block of its own; a block that jumps to one only
// it reaches absorbs it, so straight-line code forms one block and later
// structuring sees only the real joins and branches.
void Builder::merge_chains() {
    const auto count = p_.blocks.size();
    std::vector<std::uint32_t> predecessors(count);
    for (const auto& block : p_.blocks) for (auto next : ir::successors(block.terminator)) ++predecessors[next.value];
    const auto entry = p_.functions[0].entry;
    std::vector<bool> alive(count, true);
    for (std::uint32_t b = 0; b < count; ++b) {
        if (!alive[b]) continue;
        while (const auto* jump = std::get_if<ir::Jump>(&p_.blocks[b].terminator)) {
            const auto target = jump->target.value;
            if (target == b || target == entry.value || predecessors[target] != 1 || !alive[target]) break;
            auto& absorbed = p_.blocks[target];
            auto& block = p_.blocks[b];
            block.operations.insert(block.operations.end(), absorbed.operations.begin(), absorbed.operations.end());
            block.terminator = std::move(absorbed.terminator);
            absorbed.operations.clear(); absorbed.terminator = ir::Unreachable{};
            alive[target] = false;
        }
    }
    std::vector<std::uint32_t> renumbered(count);
    std::vector<ir::BasicBlock> kept;
    for (std::uint32_t b = 0; b < count; ++b) if (alive[b]) { renumbered[b] = static_cast<std::uint32_t>(kept.size()); kept.push_back(std::move(p_.blocks[b])); }
    auto remap = [&](ir::BlockId& id) { id = ir::BlockId{renumbered[id.value]}; };
    for (auto& block : kept) std::visit([&](auto& t) {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T, ir::Jump>) remap(t.target);
        else if constexpr (std::is_same_v<T, ir::Branch>) { remap(t.when_true); remap(t.when_false); }
        else if constexpr (std::is_same_v<T, ir::Switch>) { for (auto& item : t.cases) remap(item.target); remap(t.fallback); }
    }, block.terminator);
    p_.blocks = std::move(kept);
    auto& function = p_.functions[0];
    remap(function.entry);
    function.blocks.clear();
    for (std::uint32_t b = 0; b < p_.blocks.size(); ++b) function.blocks.push_back(ir::BlockId{b});
}
ir::RegisterProgram Builder::run() {
    if (source_.stage != Stage::vertex && source_.stage != Stage::fragment && !workgroup_stage())
        fail("stage semantics are not admitted");
    // Compute code has no program header; its launch facts come from the
    // control (see NativeFacts).
    const bool compute = source_.stage == Stage::compute;
    if (source_.header.size() != (compute ? 0u : 128u) || source_.instructions.empty() || source_.instructions.size() % 16 ||
        source_.instructions.size() > p_.limits.code_bytes || source_.instructions.size() / 16 > p_.limits.instructions)
        fail(compute ? "invalid compute image" : "invalid graphics image", ir::DiagnosticCode::malformed_input);
    if (compute && !source_.launch) fail("compute launch facts are absent from the control");
    if (!compute) {
        const auto header = little_endian(source_.header, 0, 4);
        const bool fragment = source_.stage == Stage::fragment;
        // A task or mesh stage's header is of the vertex or domain kind.
        const auto kind = (header >> 10) & 15;
        const bool kind_admitted = fragment ? kind == 5 : source_.stage == Stage::vertex ? kind == 1 : kind == 1 || kind == 3;
        if ((header & 31) != (fragment ? 2u : 1u) || !kind_admitted || ((header >> 5) & 31) != 4)
            fail("SPH family, version or stage is not admitted");
    }
    if (source_.stage == Stage::task || source_.stage == Stage::mesh) {
        if (!source_.mesh_launch) fail("task or mesh launch facts are absent from the control");
        auto memory = stage_memory(source_);
        if (const auto* error = std::get_if<std::string>(&memory)) fail(*error, ir::DiagnosticCode::limit_exceeded);
        stage_memory_ = std::get<StageMemory>(std::move(memory));
    }
    // Jump tables start where a BRX that can execute finds its table.
    for (std::uint32_t pc = 0; pc < source_.instructions.size(); pc += 16) {
        const auto raw = Instruction::read(source_.instructions.subspan(pc, 16));
        if ((raw.bits(0, 12) == 0x949 || raw.bits(0, 12) == 0x958) && !never_executes(raw))
            if (const auto offset = table_offset(pc)) table_starts_.insert(*offset);
    }
    p_.functions[0].entry = block(0);
    while (!pending_.empty()) {
        const auto next = pending_.front(); pending_.pop_front();
        pc_ = next.first; context_ = next.second; current_ = blocks_.at(next);
        instruction(Instruction::read(source_.instructions.subspan(pc_, 16)));
    }
    merge_chains();
    for (std::uint32_t i = 0; i < p_.blocks.size(); ++i) for (auto next : ir::successors(p_.blocks[i].terminator))
        p_.blocks[next.value].predecessors.push_back(ir::BlockId{i});
    // A uniform register or predicate no instruction writes reads zero
    // (false), the owner's undefined-read decision; registers and partially
    // defined locals reach the same rule in the shared SSA pass.
    std::set<std::uint32_t> written;
    for (const auto& op : p_.operations) for (const auto& output : op.outputs)
        if (const auto* value = std::get_if<ir::ValueId>(&output)) written.insert(value->value);
    std::vector<ir::OperationId> zeros;
    auto zero = [&](const std::optional<ir::ValueId>& local, ir::ValueType type) {
        if (!local || written.count(local->value)) return;
        zeros.push_back(ir::OperationId{static_cast<std::uint32_t>(p_.operations.size())});
        ir::Operation op{{0, {}}, ir::Opcode::copy, {}, {}, {ir::LiteralBits{0, type}}, {*local}, {}};
        op.effects = ir::required_effects(op);
        p_.operations.push_back(std::move(op));
    };
    for (const auto& local : uniform_registers_) zero(local, ir::ValueType::word32);
    for (const auto& local : uniform_predicates_) zero(local, ir::ValueType::predicate);
    auto& entry = p_.blocks[p_.functions[0].entry.value].operations;
    entry.insert(entry.begin(), zeros.begin(), zeros.end());
    if (auto checked = ir::validate(p_, ir::IrForm::registers, ir::Phase::emit_ir); auto error = std::get_if<ir::Diagnostic>(&checked)) throw Failure{*error};
    return {std::move(p_)};
}
} // namespace detail
namespace {
using detail::Failure;
class NativeFacts final : public ir::FactSource {
public:
    NativeFacts(const Program& p, FrontendOptions options, const detail::TextureEvidence& textures, const std::set<ir::TextureKey>& assumed)
        : p_(p), options_(options), textures_(textures), assumed_(assumed) {}
    ir::Knowledge<ir::FactValue> lookup(ir::FactKey key) const override {
        if (key.kind == ir::FactKind::texture_descriptor) {
            const auto found = textures_.find(ir::TextureKey{static_cast<std::uint8_t>(key.index), key.byte_offset});
            if (found == textures_.end() || !found->second) return ir::Unknown{"Native texture instructions prove no single descriptor for this handle"};
            return ir::Known<ir::FactValue>{*found->second, {ir::OriginKind::derived, assumed_.count(found->first)
                ? "code:assumed-query-only-2d" : "code:native-texture-instruction-shape"}};
        }
        // A compute stage's thread group size (control +0x58) and shared
        // memory bytes (+0x64) are static in its control.
        if (key.kind == ir::FactKind::compute_local_size && p_.launch)
            return ir::Known<ir::FactValue>{ir::ComputeLocalSize{p_.launch->local_size}, {ir::OriginKind::validated_header, "native-control-launch-v1"}};
        if (key.kind == ir::FactKind::compute_shared_memory_bytes && p_.launch)
            return ir::Known<ir::FactValue>{p_.launch->shared_memory_bytes, {ir::OriginKind::validated_header, "native-control-launch-v1"}};
        // A task or mesh stage runs one-dimensional workgroups of the
        // control's threads (+0x58), with its stage buffer as their memory.
        if (key.kind == ir::FactKind::compute_local_size && p_.mesh_launch)
            return ir::Known<ir::FactValue>{ir::ComputeLocalSize{{p_.mesh_launch->threads, 1, 1}},
                                            {ir::OriginKind::validated_header, "native-control-mesh-launch-v1"}};
        if (key.kind == ir::FactKind::compute_shared_memory_bytes && p_.mesh_launch) {
            const auto memory = stage_memory(p_);
            if (const auto* layout = std::get_if<StageMemory>(&memory))
                return ir::Known<ir::FactValue>{layout->bytes, {ir::OriginKind::validated_header, "native-stage-buffer-v1"}};
        }
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
        // The 128-byte graphics header keeps the shared program header: the
        // low 24 bits of common words 1, 2 and 3 are the per-thread local
        // memory, the high part of that allocation, and the warp-wide
        // call-stack bytes. Divide the stack by the 32-thread warp and add it.
        if (key.kind == ir::FactKind::local_memory_bytes) {
            // A compute stage has no header: its control gives the per-thread
            // local memory at +0x68 and the warp's at +0x50, 32 times it.
            if (p_.stage == Stage::compute && p_.control.size() >= 0x6c) {
                const auto thread = static_cast<std::uint32_t>(little_endian(p_.control, 0x68, 4));
                if (little_endian(p_.control, 0x50, 4) != std::uint64_t{thread} * 32)
                    return ir::Unknown{"Native compute local memory sizes disagree"};
                return ir::Known<ir::FactValue>{thread, {ir::OriginKind::validated_header, "native-compute-local-memory"}};
            }
            if (p_.header.size() < 16) return ir::Unknown{"Native local memory size is absent from the program header"};
            constexpr std::uint32_t field = (std::uint32_t{1} << 24) - 1;
            const auto word = [&](std::size_t offset) {
                return static_cast<std::uint32_t>(little_endian(p_.header, offset, 4)) & field;
            };
            return ir::Known<ir::FactValue>{word(4) + word(8) + word(12) / 32,
                {ir::OriginKind::validated_header, "native-sph-v4-local-memory"}};
        }
        return ir::Unknown{"Native binding or specialization fact is not established"};
    }
private: const Program& p_; FrontendOptions options_; const detail::TextureEvidence& textures_; const std::set<ir::TextureKey>& assumed_;
};
template<class T> T checked(ir::Result<T> value) {
    if (auto error = std::get_if<ir::Diagnostic>(&value)) throw Failure{*error};
    return std::get<T>(std::move(value));
}
}
ir::Result<ir::RegisterProgram> emit_native_ir(const Program& program, ir::Limits limits) {
    try { return detail::Builder{program, limits}.run(); } catch (const Failure& failure) { return failure.diagnostic; }
}
ir::Result<ir::MslArtifact> translate_native(const Program& program, const FrontendOptions& options) {
    try {
        ir::Limits limits;
        detail::TextureEvidence textures;
        std::set<ir::TextureKey> assumed_textures;
        NativeFacts facts{program, options, textures, assumed_textures}; ir::FactReader reader{facts, limits};
        detail::Builder builder{program, limits, options.pixel_rate, options.vertex_fetch};
        auto registers = builder.run();
        auto ssa = checked(ir::construct_ssa(std::move(registers), limits));
        detail::verify_jump_tables(ssa.ir, builder.jump_tables());
        auto handle_programs = detail::build_handle_programs(ssa.ir);
        detail::collect_texture_evidence(ssa.ir, textures, assumed_textures);
        detail::fit_lod_queries(ssa.ir, textures);
        ir::MetalProfile profile{builder.requires_disabled_rate() ? std::string(disabled_pixel_rate_profile) : "armigo-native-metal4", 3, 31, 128, 16, true, false, true, false, false, true};
        profile.texture_pools = options.texture_pools;
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
        // The guest computes identical position math identically in every
        // program; a game may draw geometry twice with different programs
        // and test depth for equality.
        emit.invariant_position = true;
        // A guest-memory miss names its program by the stage identity's first
        // four bytes.
        if (program.stage_identity.size() >= 8)
            emit.guest_memory_tag = static_cast<std::uint32_t>(std::stoul(program.stage_identity.substr(0, 8), nullptr, 16));
        auto artifact = checked(ir::emit_msl(structured, plan, emit));
        artifact.reflection.handle_programs = std::move(handle_programs);
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
        if (program.stage == Stage::task || program.stage == Stage::mesh) {
            const auto memory = stage_memory(program);
            const auto* layout = std::get_if<StageMemory>(&memory);
            if (!layout) throw Failure{ir::diagnose(ir::Phase::emit_msl, ir::DiagnosticCode::limit_exceeded, {}, "Native " + std::get<std::string>(memory))};
            if (const auto error = wrap_workgroup_stage(artifact, program, *layout, options.vertex_depth, options.mesh_fragment_reads))
                throw Failure{ir::diagnose(ir::Phase::emit_msl, ir::DiagnosticCode::unsupported_capability, {}, "Native " + *error)};
        }
        return artifact;
    } catch (const Failure& failure) { return failure.diagnostic; }
}
} // namespace armigo::shader_native
