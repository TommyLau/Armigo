// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Per-workgroup shared memory (LDS 0x984, STS 0x388) and the workgroup
// barrier (BAR 0xb1d). Compute stages only: other stages have no workgroup
// allocation.
//
// Ra is bits 24-31 (255 reads zero). A store's data starts at bits 32-39; a
// load's destination is bits 16-23. Bit 91 on a load adds a uniform register
// at bits 32-37 (63 adds zero). The same bit on a store is a different
// encoding and is refused. The signed byte displacement is bits 40-63. The
// access type is bits 73-75: U8, S8, U16, S16, 32, 64 or 128 bits, naturally
// aligned to that element. Stride bits 78-79 scale Ra by 1, 4, 8 or 16 (a
// left shift of 0, 2, 3 or 4); 255 is not scaled. A memory order, a cache
// hint or any other set modifier is refused. The access itself stays relaxed;
// cross-lane ordering is the barrier.
//
// BAR with a true guard and mode bits 77-80 clear waits for every thread in
// the group. A predicated barrier or any other mode is refused.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};

bool zero_word(const ir::Operand& operand) {
    const auto* literal = std::get_if<ir::LiteralBits>(&operand);
    return literal != nullptr && literal->bits == 0 && literal->type == ir::ValueType::word32;
}

ir::NumericMode access_mode(unsigned type) {
    static constexpr std::array<ir::ScalarType, 4> narrow{
        ir::ScalarType::u8, ir::ScalarType::s8, ir::ScalarType::u16, ir::ScalarType::s16};
    if (type < 4) return {type == 1 || type == 3 ? ir::Domain::signed_integer : ir::Domain::unsigned_integer, narrow[type]};
    return unsigned_word;
}
} // namespace

bool Builder::shared_memory(Fields& fields) {
    const bool load = opcode_ == 0x984;
    const bool store = opcode_ == 0x388;
    const bool barrier = opcode_ == 0xb1d;
    const bool atomic = opcode_ == 0x38c || opcode_ == 0xf8c;
    if (!load && !store && !barrier && !atomic) return false;
    if (source_.stage != Stage::compute) fail("shared memory requires a compute stage");

    if (atomic) {
        enter(fields, false);
        const bool one = opcode_ == 0xf8c;
        const auto destination = static_cast<unsigned>(fields.take(16, 8));
        const auto address_register = static_cast<unsigned>(fields.take(24, 8));
        const auto data_register = static_cast<unsigned>(fields.take(32, 8));
        const auto displacement = static_cast<std::uint32_t>(fields.take(40, 24));
        const auto byte_offset = (displacement ^ 0x800000u) - 0x800000u;
        const auto type = static_cast<unsigned>(fields.take(73, 3));
        const auto stride = static_cast<unsigned>(fields.take(78, 2));
        const auto kind = static_cast<unsigned>(fields.take(87, 4));
        const bool uniform_address = fields.flag(91);
        if (type != 0) fail("shared atomic size is not admitted");
        if (one) {
            if (kind != 10 || !uniform_address || destination != 255)
                fail("implicit shared increment encoding is not admitted");
            if (fields.take(64, 6) != 63)
                fail("uniform implicit shared increment address is not admitted");
        } else if (kind != 0 || uniform_address) {
            fail("shared atomic operation or uniform address is not admitted");
        }
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::MemoryInfo info;
        info.space = ir::AddressSpace::shared;
        info.access = ir::Access::read_write;
        info.scope = ir::MemoryScope::workgroup;
        info.atomic = ir::AtomicKind::add;
        info.native_shared_atomic_policy = ir::NativeSharedAtomicPolicy::word_address_v1;
        info.alignment = info.element_bytes = 4;
        std::vector<ir::Operand> inputs;
        if (address_register != 255) {
            ir::Operand address = reg(address_register);
            if (stride) address = bits(ir::Opcode::shift_left, {address, word(stride + 1)});
            if (byte_offset) address = bits(ir::Opcode::add, {address, word(byte_offset)}, unsigned_word);
            // Shared atomics select a word after wrapping the byte address.
            address = bits(ir::Opcode::bit_and, {address, word(0xfffffffcu)});
            info.dynamic_byte_offset = ir::InputIndex{0};
            inputs.push_back(std::move(address));
        } else {
            info.byte_offset = byte_offset & ~3u;
        }
        info.data.push_back(ir::InputIndex{static_cast<std::uint16_t>(inputs.size())});
        // POPC.INC adds one for each executing lane; Rb is not an operand.
        inputs.push_back(one ? word(1) : reg(data_register));
        const auto old = temporary(ir::Opcode::atomic, std::move(inputs), unsigned_word, info);
        write(destination, old);
        leave(fields); return true;
    }

    if (barrier) {
        // A barrier some lanes skip deadlocks the group, so the guard is PT.
        if (fields.raw().bits(12, 4) != 7) fail("workgroup barrier must be unconditional");
        enter(fields, false);
        if (fields.take(77, 4) != 0) fail("workgroup barrier mode is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        // Defaults keep the one-word relaxed geometry the barrier opcode
        // requires: alignment 1, element bytes 4, no dynamic address.
        ir::MemoryInfo info;
        info.space = ir::AddressSpace::shared;
        info.access = ir::Access::read_write;
        info.order = ir::MemoryOrder::acquire_release;
        info.scope = ir::MemoryScope::workgroup;
        operation({{}, ir::Opcode::barrier, {}, {}, {}, {}, info});
        leave(fields);
        return true;
    }

    enter(fields, false);
    const auto ra = static_cast<unsigned>(fields.take(24, 8));
    const auto at_32 = static_cast<unsigned>(fields.take(32, 8));
    const auto displacement = static_cast<std::uint32_t>(fields.take(40, 24));
    // Sign-extend the 24-bit byte displacement. Flipping its high bit keeps the
    // two's-complement result defined for every encoding of that field.
    const auto byte_offset = (displacement ^ 0x800000u) - 0x800000u;
    const bool uniform_base = fields.flag(91);
    if (store && uniform_base) fail("uniform shared store form is not admitted");

    unsigned ur = 63;
    unsigned data = 255;
    unsigned destination = 255;
    if (load) {
        destination = static_cast<unsigned>(fields.take(16, 8));
        if (uniform_base) {
            if (at_32 > 63) fail("uniform register field extension is not admitted");
            ur = at_32;
        } else if (at_32 != 0) {
            fail("shared load uniform address is not admitted");
        }
    } else {
        data = at_32;
    }

    const auto type = static_cast<unsigned>(fields.take(73, 3));
    if (type > 6) fail("shared memory access type is not admitted");
    const auto stride = static_cast<unsigned>(fields.take(78, 2));
    const unsigned words = type >= 6 ? 4u : type == 5 ? 2u : 1u;
    const auto element_bytes = static_cast<std::uint8_t>(type < 2 ? 1 : type < 4 ? 2 : 4);
    const auto vector = load ? destination : data;
    if (vector != 255 && (vector % words != 0 || vector + words > 255))
        fail("shared memory register vector is not admitted");
    if (byte_offset & (element_bytes - 1u)) fail("shared memory displacement is not naturally aligned");
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");

    ir::Operand base = word(0);
    auto add_base = [&](ir::Operand term) {
        if (zero_word(base)) base = std::move(term);
        else base = temporary(ir::Opcode::add, {std::move(base), std::move(term)}, unsigned_word);
    };
    if (ra != 255) {
        ir::Operand term = reg(ra);
        if (stride) term = bits(ir::Opcode::shift_left, {std::move(term), word(stride + 1)});
        add_base(std::move(term));
    }
    if (uniform_base && ur != 63) add_base(uniform(ur));
    const bool dynamic = !zero_word(base);
    if (!dynamic && byte_offset >= 0x80000000u)
        fail("negative shared displacement needs an address register");
    if (dynamic && byte_offset != 0)
        base = temporary(ir::Opcode::add, {std::move(base), word(byte_offset)}, unsigned_word);

    ir::MemoryInfo info;
    info.space = ir::AddressSpace::shared;
    info.byte_offset = dynamic ? 0u : byte_offset;
    info.element_bytes = element_bytes;
    info.elements = static_cast<std::uint8_t>(words);
    info.alignment = element_bytes;
    std::vector<ir::Operand> inputs;
    if (dynamic) {
        info.dynamic_byte_offset = ir::InputIndex{0};
        inputs.push_back(std::move(base));
    }
    const auto mode = access_mode(type);
    if (load) {
        info.access = ir::Access::read;
        std::vector<ir::Operand> outputs;
        for (unsigned i = 0; i < words; ++i) {
            if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
            ir::ValueId id{static_cast<std::uint32_t>(p_.values.size())};
            p_.values.push_back({ir::ValueType::word32});
            outputs.push_back(id);
        }
        operation({{}, ir::Opcode::load, {}, mode, std::move(inputs), outputs, info});
        if (destination != 255)
            for (unsigned i = 0; i < words; ++i) write(destination + i, outputs[i]);
    } else {
        info.access = ir::Access::write;
        for (unsigned i = 0; i < words; ++i) {
            info.data.push_back(ir::InputIndex{static_cast<std::uint16_t>(inputs.size())});
            const auto index = data == 255 || data + i >= 255 ? 255u : data + i;
            inputs.push_back(index == 255 ? word(0) : reg(index));
        }
        operation({{}, ir::Opcode::store, {}, mode, std::move(inputs), {}, info});
    }
    leave(fields);
    return true;
}
} // namespace armigo::shader_native::detail
