// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Per-invocation local memory. STL (0x387, and 0x987 when the address adds a
// uniform register) and LDL (0x983) address one thread-private array.
//
// Ra is bits 24-31 (255 reads zero). A store's data starts at bits 32-39; a
// load's destination is bits 16-23. Bit 91 adds a uniform register, bits
// 32-37 on a load and bits 64-69 on a store (63 adds zero). The signed
// byte displacement is bits 40-63. The access type is bits 73-75: U8, S8,
// U16, S16, 32, 64 or 128 bits, naturally aligned to that element. Bits
// 84-86 are a cache hint (normal, or last-use on a load) and do not change
// the access. A scaled address, a memory order, a predicate result or any
// other set modifier is refused.
//
// The byte address and the sized access match the graphics local-memory
// operations: one word per register, in ascending address order, with the
// sign of a narrow load in the numeric mode. The array length is the program
// header's local-memory total, required when the access is legalized.

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

bool Builder::local_memory(Fields& fields) {
    const bool load = opcode_ == 0x983;
    const bool store = opcode_ == 0x387 || opcode_ == 0x987;
    if (!load && !store) return false;
    enter(fields, false);

    const auto ra = static_cast<unsigned>(fields.take(24, 8));
    const auto at_32 = static_cast<unsigned>(fields.take(32, 8));
    const auto displacement = static_cast<std::uint32_t>(fields.take(40, 24));
    // Sign-extend the 24-bit byte displacement. Flipping its high bit keeps the
    // two's-complement result defined for every encoding of that field.
    const auto byte_offset = (displacement ^ 0x800000u) - 0x800000u;
    const bool uniform_base = fields.flag(91);
    if (opcode_ == 0x987 && !uniform_base) fail("uniform local store form is not admitted");

    unsigned ur = 63;
    unsigned data = 255;
    unsigned destination = 255;
    if (load) {
        destination = static_cast<unsigned>(fields.take(16, 8));
        if (uniform_base) {
            if (at_32 > 63) fail("uniform register field extension is not admitted");
            ur = at_32;
        } else if (at_32 != 0) {
            fail("local load uniform address is not admitted");
        }
    } else {
        data = at_32;
        if (uniform_base) ur = static_cast<unsigned>(fields.take(64, 6));
    }

    const auto type = static_cast<unsigned>(fields.take(73, 3));
    if (type > 6) fail("local memory access type is not admitted");
    const auto cache = fields.take(84, 3);
    // Normal is the ordinary hint. A load may also be marked last-use.
    if (cache != 1 && !(load && cache == 3)) fail("local memory cache hint is not admitted");

    const unsigned words = type >= 6 ? 4u : type == 5 ? 2u : 1u;
    const auto element_bytes = static_cast<std::uint8_t>(type < 2 ? 1 : type < 4 ? 2 : 4);
    const auto vector = load ? destination : data;
    if (vector != 255 && (vector % words != 0 || vector + words > 255))
        fail("local memory register vector is not admitted");
    if (byte_offset & (element_bytes - 1u)) fail("local memory displacement is not naturally aligned");
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");

    ir::Operand base = word(0);
    auto add_base = [&](ir::Operand term) {
        if (zero_word(base)) base = std::move(term);
        else base = temporary(ir::Opcode::add, {std::move(base), std::move(term)}, unsigned_word);
    };
    if (ra != 255) add_base(reg(ra));
    if (uniform_base && ur != 63) add_base(uniform(ur));
    const bool dynamic = !zero_word(base);
    if (!dynamic && byte_offset >= 0x80000000u)
        fail("negative local displacement needs an address register");
    if (dynamic && byte_offset != 0)
        base = temporary(ir::Opcode::add, {std::move(base), word(byte_offset)}, unsigned_word);

    ir::MemoryInfo info;
    info.space = ir::AddressSpace::local;
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
