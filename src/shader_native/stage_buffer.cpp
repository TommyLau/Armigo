// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The stage buffer of task and mesh stages (ISBERD 0x923, ISBEWR 0x927), kept
// in workgroup memory (stage_memory.h).
//
// Ra is bits 24-31 (255 reads zero) and the signed byte displacement bits
// 40-63; ISBERD's destination is bits 16-23 and ISBEWR's data bits 32-39.
// Bits 72-79 select the access; the address is Ra plus the displacement in
// its region, with no lane skew:
//
//   ISBERD 0x78   the workgroup's number (Ra RZ, displacement 0)
//   ISBERD 0x38   a word of a mesh stage's input
//   ISBERD 0xb8   a word of a task stage's own payload
//   ISBEWR 0xb8   a word of a task stage's payload
//   ISBEWR 0xf8   a word of a mesh stage's attributes
//   ISBEWR 0x88   the primitive count byte; a store of RZ is dropped: the
//                 buffer starts at zero, and a mesh workgroup of 32 threads
//                 runs as one warp on the hardware, so that zero never
//                 overtakes the count another thread stores later. In a
//                 larger workgroup it is refused.
//   ISBEWR 0x80   a primitive index byte
//
// Other accesses are refused. A word access ignores its address's low two
// bits. Bytes go through the workgroup memory's byte store, since
// neighbouring threads write neighbouring bytes of one word.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
} // namespace

bool Builder::stage_buffer(Fields& fields) {
    if (opcode_ != 0x923 && opcode_ != 0x927) return false;
    if (!stage_memory_) fail("stage buffer access requires a task or mesh stage");
    const auto& memory = *stage_memory_;
    const bool store = opcode_ == 0x927;
    const bool mesh = source_.stage == Stage::mesh;

    enter(fields, false);
    const auto ra = static_cast<unsigned>(fields.take(24, 8));
    const auto displacement = static_cast<std::uint32_t>(fields.take(40, 24));
    const auto byte_offset = (displacement ^ 0x800000u) - 0x800000u;
    const auto mode = static_cast<unsigned>(fields.take(72, 8));
    const auto vector = static_cast<unsigned>(store ? fields.take(32, 8) : fields.take(16, 8));
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");

    std::uint32_t region = 0;
    std::uint8_t element_bytes = 4;
    if (!store && mode == 0x78) {
        if (ra != 255 || byte_offset != 0) fail("stage buffer workgroup read with an address is not admitted");
        region = memory.workgroup;
    } else if (!store && mode == 0x38 && mesh) {
        region = memory.input;
    } else if ((mode == 0xb8 && !mesh) || (store && mode == 0xf8 && mesh)) {
        region = memory.output;
    } else if (store && (mode == 0x88 || mode == 0x80) && mesh) {
        region = memory.primitives;
        element_bytes = 1;
    } else {
        fail("stage buffer access mode is not admitted");
    }
    if (store && mode == 0x88 && vector == 255) {
        // The drop rests on one warp: in a larger workgroup another warp's
        // count may come first, and that zero must then win.
        if (source_.mesh_launch->threads > 32) fail("primitive count store of RZ in a mesh workgroup of more than 32 threads is not admitted");
        leave(fields);
        return true;
    }

    // A word access ignores the address's low two bits, as a buffer of words
    // indexed by byte address does.
    ir::Operand address = word(element_bytes == 4 ? (region + byte_offset) & ~3u : region + byte_offset);
    if (ra != 255) {
        address = temporary(ir::Opcode::add, {reg(ra), word(region + byte_offset)}, unsigned_word);
        // Shifting the low bits out keeps the register offset unbounded (a
        // runtime-guarded access) and proves the word alignment.
        if (element_bytes == 4)
            address = bits(ir::Opcode::shift_left, {bits(ir::Opcode::shift_right, {std::move(address), word(2)}), word(2)});
    }
    ir::MemoryInfo info;
    info.space = ir::AddressSpace::shared;
    info.element_bytes = element_bytes;
    info.elements = 1;
    info.alignment = element_bytes;
    info.dynamic_byte_offset = ir::InputIndex{0};
    std::vector<ir::Operand> inputs{std::move(address)};
    const ir::NumericMode mode_bits = element_bytes == 1
        ? ir::NumericMode{ir::Domain::unsigned_integer, ir::ScalarType::u8} : unsigned_word;
    if (store) {
        info.access = ir::Access::write;
        info.data.push_back(ir::InputIndex{1});
        inputs.push_back(vector == 255 ? word(0) : reg(vector));
        operation({{}, ir::Opcode::store, {}, mode_bits, std::move(inputs), {}, info});
    } else {
        info.access = ir::Access::read;
        if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const ir::ValueId id{static_cast<std::uint32_t>(p_.values.size())};
        p_.values.push_back({ir::ValueType::word32});
        operation({{}, ir::Opcode::load, {}, mode_bits, std::move(inputs), {id}, info});
        if (vector != 255) write(vector, id);
    }
    leave(fields);
    return true;
}
} // namespace armigo::shader_native::detail
