// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Native global memory: LDG, STG, RED and ATOMG at 64-bit guest addresses,
// lowered to guest-memory accesses the host serves through its windows.
//
// The register forms take the address from the pair Ra:Ra+1 (the .E bit, 72,
// is required). The uniform forms (bit 91; RED selects its form by that bit
// alone) add a 64-bit uniform pair (its
// width bit is 72) to Ra, which is a pair when its width bit is set and a
// zero-extended word otherwise. Every form adds the signed 24-bit
// displacement at bits 40-63. A set descriptor bit (76 for LDG and STG, 71
// for ATOMG and RED) makes that uniform register a memory descriptor, a
// cache policy, and not an address term: the address is Ra plus the
// displacement. The field is still consumed. Its width bit and index are
// not checked, and the register is not read.
// Predicated loads stay refused. The access type is bits 73-75 (U8, S8, U16,
// S16, 32, 64, 128 bits; atomics U32 or S32, and type 3, an F32 add with
// flushed operands rounded to nearest, for RED and ATOMG's add), the memory
// order bits 77-80
// and the eviction priority, a cache hint, bits 84-86. Weak and constant
// loads and stores are plain device accesses of the invocation; strong ones
// keep their scope, and the back end makes each word a relaxed device atomic.
// Atomics are strong at GPU scope, which relaxed device atomics provide.
// INC and DEC of a 32-bit word (type 0 or 1) are a compare-and-swap loop.
// 64-bit INC and DEC, and every other size, stay refused.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode signed_word{ir::Domain::signed_integer, ir::ScalarType::s32};
} // namespace

bool Builder::global_memory(Fields& fields) {
    const bool load = opcode_ == 0x381 || opcode_ == 0x981, store = opcode_ == 0x386 || opcode_ == 0x986;
    const bool reduce = opcode_ == 0x98e, exchange = opcode_ == 0x3a9;
    const bool atomic = opcode_ == 0x3a8 || opcode_ == 0x9a8 || exchange;
    if (!load && !store && !reduce && !atomic) return false;
    enter(fields, false);
    // RED has one opcode for both forms; the others name the form twice.
    const bool uniform_form = reduce ? fields.flag(91) : (opcode_ >> 8) == 9;
    if (!reduce && fields.flag(91) != uniform_form) fail("global memory operand form is not admitted");
    auto pair = [&](unsigned index) -> std::array<ir::Operand, 2> {
        if (index == 255) return {word(0), word(0)};
        if (index % 2 || index > 253) fail("global address register pair is not admitted");
        return {reg(index), reg(index + 1)};
    };

    // The 64-bit address before the displacement.
    const auto ra = static_cast<unsigned>(fields.take(24, 8));
    std::array<ir::Operand, 2> address;
    if (!uniform_form) {
        if (!fields.flag(72)) fail("32-bit global address is not admitted");
        address = pair(ra);
    } else {
        const unsigned at = load ? 32 : 64;
        // Bit 76 (LDG, STG) or 71 (ATOMG, RED), read before the uniform
        // address checks. Set, the uniform register is a memory descriptor,
        // a cache policy, and is not an address term: the address is Ra
        // plus the displacement. The field bits are consumed either way.
        // Only the other uniform form checks the width bit and the pair,
        // and only that form reads the register.
        const bool descriptor = fields.flag(load || store ? 76 : 71);
        const auto ur = static_cast<unsigned>(fields.take(at, 6));
        if (load && fields.take(38, 2)) fail("uniform register field extension is not admitted");
        const bool uniform_pair = fields.flag(72);
        if (!descriptor) {
            if (!uniform_pair) fail("32-bit uniform global address is not admitted");
            if (ur != 63 && (ur % 2 || ur > 61)) fail("uniform global address pair is not admitted");
        }
        const bool wide = fields.flag(opcode_ == 0x9a8 ? 70 : 90);
        const std::array<ir::Operand, 2> offset = wide ? pair(ra) : std::array<ir::Operand, 2>{ra == 255 ? word(0) : reg(ra), word(0)};
        if (descriptor) {
            address = offset;
        } else {
            const std::array<ir::Operand, 2> base{uniform(ur), ur == 63 ? word(0) : uniform(ur + 1)};
            auto [low, carry] = add_with_carry(base[0], offset[0], ir::LiteralBits{0, ir::ValueType::predicate});
            auto [high, ignored] = add_with_carry(base[1], offset[1], carry);
            address = {low, high};
        }
    }
    const auto displacement = static_cast<std::uint32_t>(fields.take(40, 24));

    const auto type = static_cast<unsigned>(fields.take(73, 3));
    const auto order = fields.take(77, 4);
    if (fields.take(84, 3) > 5) fail("global memory eviction priority is not admitted");
    if ((load || atomic) && fields.take(81, 3) != 7) fail("global memory fault predicate is not admitted");
    if (opcode_ == 0x981 && (fields.take(64, 3) != 0 || fields.flag(67)))
        fail("predicated global load is not admitted");

    ir::MemoryInfo info;
    info.space = ir::AddressSpace::guest;
    info.byte_offset = (displacement ^ 0x800000u) - 0x800000u;
    info.dynamic_byte_offset = ir::InputIndex{0};
    info.global_address_high = ir::InputIndex{1};
    std::vector<ir::Operand> inputs{address[0], address[1]};
    auto data = [&](ir::Operand value) {
        info.data.push_back(ir::InputIndex{static_cast<std::uint16_t>(inputs.size())});
        inputs.push_back(std::move(value));
    };

    if (load || store) {
        // Weak, constant, strong at SM scope and strong at GPU scope.
        if (order != 0 && order != 4 && order != 5 && order != 7) fail("global memory order is not admitted");
        // Strong accesses keep their scope; the back end makes each word a
        // relaxed device atomic.
        if (order == 5) info.scope = ir::MemoryScope::workgroup;
        if (order == 7) info.scope = ir::MemoryScope::device;
        static constexpr std::array<ir::ScalarType, 4> narrow{ir::ScalarType::u8, ir::ScalarType::s8, ir::ScalarType::u16, ir::ScalarType::s16};
        if (type > 6 || (store && type < 4)) fail("global memory access type is not admitted");
        const unsigned words = type < 5 ? 1u : type == 5 ? 2u : 4u;
        const auto scalar = type < 4 ? narrow[type] : ir::ScalarType::u32;
        info.element_bytes = static_cast<std::uint8_t>(type < 2 ? 1 : type < 4 ? 2 : 4);
        info.elements = static_cast<std::uint8_t>(words);
        info.alignment = static_cast<std::uint8_t>(info.element_bytes * words);
        const ir::NumericMode mode{type == 1 || type == 3 ? ir::Domain::signed_integer : ir::Domain::unsigned_integer, scalar};
        const auto r = static_cast<unsigned>(fields.take(load ? 16 : 32, 8));
        if (r != 255 && (r % words || r + words > 255)) fail("global memory register vector is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        if (load) {
            info.access = ir::Access::read;
            std::vector<ir::Operand> outputs;
            for (unsigned i = 0; i < words; ++i) {
                ir::ValueId id{static_cast<std::uint32_t>(p_.values.size())}; p_.values.push_back({ir::ValueType::word32});
                outputs.push_back(id);
            }
            operation({{}, ir::Opcode::load, {}, mode, std::move(inputs), outputs, info});
            if (r != 255) for (unsigned i = 0; i < words; ++i) write(r + i, outputs[i]);
        } else {
            info.access = ir::Access::write;
            for (unsigned i = 0; i < words; ++i) data(r == 255 ? word(0) : reg(r + i));
            operation({{}, ir::Opcode::store, {}, mode, std::move(inputs), {}, info});
        }
        leave(fields); return true;
    }

    // Atomics and reductions: one 32-bit word, strong at GPU scope.
    if (order != 7) fail("global atomic memory order is not admitted");
    const bool float_add = type == 3;
    if (type > 1 && !float_add) fail("global atomic type is not admitted");
    const auto operation_kind = static_cast<unsigned>(exchange ? 0 : fields.take(87, reduce ? 3 : 4));
    if (float_add) {
        // F32.FTZ.RN adds a float to the word. No Metal device atomic adds
        // floats in that form, so a compare-and-swap loop does: from a
        // strong load of the word, the flushed sum replaces the word while
        // it still holds what the sum was made from, else the sum is made
        // again from the word now there. The old word is ATOMG's result.
        if (exchange || operation_kind != 0) fail("floating global atomic operation is not admitted");
        const auto d = reduce ? 255u : static_cast<unsigned>(fields.take(16, 8));
        const auto r = static_cast<unsigned>(fields.take(32, 8));
        const auto value = r == 255 ? word(0) : reg(r);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::MemoryInfo read = info;
        read.access = ir::Access::read; read.scope = ir::MemoryScope::device;
        read.element_bytes = 4; read.elements = 1; read.alignment = 4;
        if (p_.values.size() + 2 >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const ir::ValueId loaded{static_cast<std::uint32_t>(p_.values.size())}; p_.values.push_back({ir::ValueType::word32});
        operation({{}, ir::Opcode::load, {}, unsigned_word, inputs, {loaded}, read});
        const ir::ValueId expected{static_cast<std::uint32_t>(p_.values.size())};
        p_.values.push_back({ir::ValueType::word32, ir::ValueKind::mutable_local});
        operation({{}, ir::Opcode::copy, {}, {}, {loaded}, {expected}, {}});
        const auto loop = extra();
        p_.blocks[current_.value].terminator = ir::Jump{loop};
        current_ = loop;
        const ir::NumericMode flushed{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even, true, true};
        const auto sum = temporary(ir::Opcode::add, {expected, value}, flushed);
        info.access = ir::Access::read_write;
        info.scope = ir::MemoryScope::device;
        info.atomic = ir::AtomicKind::compare_exchange;
        info.alignment = 4;
        data(expected);
        data(sum);
        const auto previous = temporary(ir::Opcode::atomic, std::move(inputs), unsigned_word, info);
        const auto stored = temporary(ir::Opcode::compare, {previous, expected}, unsigned_word,
                                      ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
        operation({{}, ir::Opcode::copy, {}, {}, {previous}, {expected}, {}});
        const auto done = extra();
        guard(stored, done, loop);
        current_ = done;
        write(d, expected);
        leave(fields); return true;
    }
    // 32-bit INC (3) and DEC (4). The word in memory is cur. INC stores 0
    // when cur >= bound and cur + 1 otherwise. DEC stores the bound when cur
    // is 0 or cur > bound, and cur - 1 otherwise. Comparisons are unsigned.
    // A compare-and-swap retries until the word still holds cur, and cur is
    // the result. RED names no destination. 64-bit forms never reach here.
    if (operation_kind == 3 || operation_kind == 4) {
        const bool increment = operation_kind == 3;
        const auto d = reduce ? 255u : static_cast<unsigned>(fields.take(16, 8));
        const auto r = static_cast<unsigned>(fields.take(32, 8));
        const auto bound = r == 255 ? word(0) : reg(r);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::MemoryInfo read = info;
        read.access = ir::Access::read; read.scope = ir::MemoryScope::device;
        read.element_bytes = 4; read.elements = 1; read.alignment = 4;
        if (p_.values.size() + 2 >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const ir::ValueId loaded{static_cast<std::uint32_t>(p_.values.size())}; p_.values.push_back({ir::ValueType::word32});
        operation({{}, ir::Opcode::load, {}, unsigned_word, inputs, {loaded}, read});
        const ir::ValueId expected{static_cast<std::uint32_t>(p_.values.size())};
        p_.values.push_back({ir::ValueType::word32, ir::ValueKind::mutable_local});
        operation({{}, ir::Opcode::copy, {}, {}, {loaded}, {expected}, {}});
        const auto loop = extra();
        p_.blocks[current_.value].terminator = ir::Jump{loop};
        current_ = loop;
        ir::Operand next;
        if (increment) {
            const auto wrapped = temporary(ir::Opcode::compare, {expected, bound}, unsigned_word,
                ir::ComparisonInfo{ir::Compare::greater_equal, false}, ir::ValueType::predicate);
            const auto increased = bits(ir::Opcode::add, {expected, word(1)}, unsigned_word);
            next = bits(ir::Opcode::select, {wrapped, word(0), increased}, unsigned_word);
        } else {
            const auto zero = temporary(ir::Opcode::compare, {expected, word(0)}, unsigned_word,
                ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
            const auto above = temporary(ir::Opcode::compare, {expected, bound}, unsigned_word,
                ir::ComparisonInfo{ir::Compare::greater, false}, ir::ValueType::predicate);
            const auto wrapped = logic(ir::Opcode::logical_or, {zero, above});
            const auto decreased = bits(ir::Opcode::add, {expected, word(0xffffffffu)}, unsigned_word);
            next = bits(ir::Opcode::select, {wrapped, bound, decreased}, unsigned_word);
        }
        info.access = ir::Access::read_write;
        info.scope = ir::MemoryScope::device;
        info.atomic = ir::AtomicKind::compare_exchange;
        info.alignment = 4;
        data(expected);
        data(next);
        const auto previous = temporary(ir::Opcode::atomic, std::move(inputs), unsigned_word, info);
        const auto stored = temporary(ir::Opcode::compare, {previous, expected}, unsigned_word,
            ir::ComparisonInfo{ir::Compare::equal, false}, ir::ValueType::predicate);
        operation({{}, ir::Opcode::copy, {}, {}, {previous}, {expected}, {}});
        const auto done = extra();
        guard(stored, done, loop);
        current_ = done;
        write(d, expected);
        leave(fields); return true;
    }
    // Indices 3 and 4 are the loop above, so this table is not how they lower.
    static constexpr std::array<ir::AtomicKind, 9> kinds{ir::AtomicKind::add, ir::AtomicKind::minimum, ir::AtomicKind::maximum,
        ir::AtomicKind::increment, ir::AtomicKind::decrement, ir::AtomicKind::bit_and, ir::AtomicKind::bit_or,
        ir::AtomicKind::bit_xor, ir::AtomicKind::exchange};
    if (operation_kind >= kinds.size())
        fail("global atomic operation is not admitted");
    const auto kind = exchange ? ir::AtomicKind::compare_exchange : kinds[operation_kind];
    info.access = ir::Access::read_write;
    info.scope = ir::MemoryScope::device;
    info.atomic = kind;
    info.alignment = 4;
    const auto d = reduce ? 255u : static_cast<unsigned>(fields.take(16, 8));
    const auto source = [&](unsigned at) { const auto r = static_cast<unsigned>(fields.take(at, 8)); return r == 255 ? word(0) : reg(r); };
    data(source(32));
    if (exchange) data(source(64));
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
    const auto old = temporary(ir::Opcode::atomic, std::move(inputs), type == 1 ? signed_word : unsigned_word, info);
    write(d, old);
    leave(fields); return true;
}
} // namespace armigo::shader_native::detail
