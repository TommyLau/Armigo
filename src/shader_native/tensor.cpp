// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Warp-cooperative matrix instructions, computed per lane. A fragment of an
// 8x8 block of halves is spread over the warp's 32 lanes: lane L holds row
// L / 4 at columns 2 (L % 4) and 2 (L % 4) + 1, the lower column in the low
// half of its register. Each lane gathers the elements it needs from the
// lanes that hold them with SIMD-group shuffles (Apple SIMD-groups are 32
// lanes wide, as the warp) and computes its own part. The instruction set
// requires every lane of the warp to execute these instructions together, so
// a guarded one is refused; the translation keeps each lane's control flow,
// so lanes that reach one together on the guest reach the SIMD-group
// shuffles and barriers together on Metal. Reaching one from divergent
// control flow is undefined on the guest and is not detected.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode word_bits{ir::Domain::bits, ir::ScalarType::u32};
constexpr ir::NumericMode single{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
} // namespace

bool Builder::tensor(Fields& fields) {
    if (opcode_ != 0x23c && opcode_ != 0x23a && opcode_ != 0x83b) return false;
    if (opcode_ == 0x83b && source_.stage != Stage::compute) fail("matrix load from shared memory requires a compute stage");
    if (source_.stage != Stage::compute && source_.stage != Stage::fragment)
        fail("warp matrix instructions outside a fragment or compute stage are not admitted");
    if (fields.raw().bits(12, 4) != 7) fail("a guarded warp matrix instruction is not admitted");
    enter(fields, false);
    ir::IoInfo lane_io; lane_io.semantic = ir::IoSemantic::subgroup_lane_id; lane_io.type = {ir::ScalarType::u32, 1}; lane_io.component_mask = 1;
    const auto lane = temporary(ir::Opcode::load_io, {}, unsigned_word, lane_io);
    const auto group = bits(ir::Opcode::shift_right, {lane, word(2)}, unsigned_word);
    const auto column_pair = bits(ir::Opcode::bit_and, {lane, word(3)});
    // The word a given lane holds in register r: an index shuffle over the
    // whole SIMD-group (clamp 31, no segment), which every lane executes.
    auto from_lane_value = [&](ir::Operand value, ir::Operand source_lane) -> ir::Operand {
        if (p_.values.size() + 2 > p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const ir::ValueId result{static_cast<std::uint32_t>(p_.values.size())}, valid{result.value + 1};
        p_.values.push_back({ir::ValueType::word32}); p_.values.push_back({ir::ValueType::predicate});
        operation({{}, ir::Opcode::shuffle, {}, word_bits, {std::move(value), std::move(source_lane), word(0x1f)}, {result, valid},
                   ir::SubgroupInfo{ir::ShuffleKind::index, 32}});
        return result;
    };
    auto from_lane = [&](unsigned r, ir::Operand source_lane) { return from_lane_value(reg(r), std::move(source_lane)); };
    auto plus = [&](ir::Operand a, std::uint32_t b) { return b ? bits(ir::Opcode::add, {std::move(a), word(b)}, unsigned_word) : a; };

    if (opcode_ == 0x83b) { // LDSM.16.M88.n / .MT88.n Rd, [Ra + UR + offset]
        // n (bits 72-73: 1, 2 or 4) 8x8 blocks of halves from shared memory.
        // Lanes 8i to 8i + 7 hold the byte addresses of block i's rows; lane
        // L receives, of each block, row L / 4 at columns 2 (L % 4), + 1, or,
        // transposed (bits 78-79 = 1), column L / 4 at rows 2 (L % 4), + 1.
        // The uniform register (bits 32-37) is added when bit 91 is set;
        // the signed byte offset is bits 40-63.
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        const auto ra = static_cast<unsigned>(fields.take(24, 8));
        const auto ur = static_cast<unsigned>(fields.take(32, 6));
        const auto offset = static_cast<std::uint32_t>(fields.take(40, 24));
        const bool uniform_address = fields.flag(91);
        const auto count_field = static_cast<unsigned>(fields.take(72, 2));
        const auto mode = static_cast<unsigned>(fields.take(78, 2));
        if (!uniform_address && ur) fail("matrix load uniform address field without its selector is not admitted");
        if (count_field > 2 || mode > 1) fail("matrix load count or mode is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        const unsigned count = 1u << count_field;
        if (d != 255 && d + count > 255) fail("matrix load register vector crosses RZ");
        ir::Operand address = ra == 255 ? word(0) : reg(ra);
        if (uniform_address && ur != 63) address = bits(ir::Opcode::add, {address, uniform(ur)}, unsigned_word);
        if (const auto displacement = (offset ^ 0x800000u) - 0x800000u)
            address = bits(ir::Opcode::add, {address, word(displacement)}, unsigned_word);
        // One operation per instruction: the back end trades the row
        // addresses with SIMD-group shuffles and reads each lane's words
        // between SIMD-group barriers. Another lane's store just before the
        // load may not have reached shared memory on Metal without the first,
        // and no lane's next store may reach the rows before every lane has
        // read them; the guest warp keeps its shared memory accesses in order.
        // A row whose word lies outside the allocation reads zero.
        if (p_.values.size() + count > p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        std::vector<ir::ValueId> results;
        for (unsigned m = 0; m < count; ++m) {
            results.push_back(ir::ValueId{static_cast<std::uint32_t>(p_.values.size())});
            p_.values.push_back({ir::ValueType::word32});
        }
        operation({{}, ir::Opcode::matrix_load, {}, unsigned_word, {lane, address, word(mode)}, {results.begin(), results.end()},
                   ir::SubgroupInfo{ir::ShuffleKind::index, 32}});
        if (d != 255) for (unsigned m = 0; m < count; ++m) write(d + m, results[m]);
        leave(fields); return true;
    }

    if (opcode_ == 0x23a) { // MOVM.16.MT88 Rd, Ra: the warp's 8x8 block of halves, transposed into the same layout.
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        const auto a = static_cast<unsigned>(fields.take(24, 8));
        if (fields.take(78, 2)) fail("matrix move other than the 8x8 transpose is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        // Output (row, column 2t + j) is input (column 2t + j, row): lane
        // 4 (2t + j) + row / 2 holds it, in half row % 2.
        const auto half_shift = bits(ir::Opcode::shift_left, {bits(ir::Opcode::bit_and, {group, word(1)}), word(4)});
        std::array<ir::Operand, 2> h;
        for (unsigned j = 0; j < 2; ++j) {
            const auto column = plus(bits(ir::Opcode::shift_left, {column_pair, word(1)}), j);
            const auto source = bits(ir::Opcode::add, {bits(ir::Opcode::shift_left, {column, word(2)}), bits(ir::Opcode::shift_right, {group, word(1)}, unsigned_word)}, unsigned_word);
            h[j] = bits(ir::Opcode::bit_extract, {from_lane(a, source), half_shift, word(16)}, unsigned_word);
        }
        write(d, bits(ir::Opcode::bit_or, {h[0], bits(ir::Opcode::shift_left, {h[1], word(16)})}));
        leave(fields); return true;
    }

    // HMMA.16816 Rd, Ra, Rb, Rc: D (16x8) = A (16x16) x B (16x8) + C with
    // binary16 sources and accumulator. Lane L (g = L / 4, t = L % 4) holds
    // A rows g and g + 8 at k = 2t, 2t + 1 (registers Ra, Ra + 1) and k + 8
    // (Ra + 2, Ra + 3); B at k = 2t, 2t + 1 (Rb) and k + 8 (Rb + 1) of column
    // g; C and D at rows g, g + 8 and columns 2t, 2t + 1 as two packed
    // registers. The SIMD-group matrix hardware multiplies the binary16
    // values widened to binary32 (every product exact) and accumulates in
    // binary32 in its own order; D rounds to binary16 at the end. Metal
    // flushes binary32 subnormals, but every nonzero partial sum of binary16
    // products and a binary16 C is at least 2^-48, so the flush changes no
    // result. Compute stages only: the matrices pass through threadgroup
    // memory. The binary32 accumulator (bit 76) is refused: Metal would flush
    // its subnormal C and D words.
    const auto d = static_cast<unsigned>(fields.take(16, 8));
    const auto a = static_cast<unsigned>(fields.take(24, 8));
    const auto b = static_cast<unsigned>(fields.take(32, 8));
    const auto c = static_cast<unsigned>(fields.take(64, 8));
    const auto size = static_cast<unsigned>(fields.take(75, 1) | (fields.take(78, 1) << 1));
    const bool f32 = fields.flag(76);
    if (size != 1) fail("matrix multiply-add shape other than 16x8x16 is not admitted");
    if (f32) fail("matrix multiply-add with a binary32 accumulator is not admitted");
    if (source_.stage != Stage::compute) fail("matrix multiply-add requires a compute stage");
    if (fields.take(82, 2)) fail("matrix multiply-add source type other than binary16 is not admitted");
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
    const unsigned c_words = 2, d_words = 2;
    if ((a != 255 && (a % 4 || a + 4 > 255)) || (b != 255 && (b % 2 || b + 2 > 255)) ||
        (c != 255 && (c % c_words || c + c_words > 255)) || (d != 255 && (d % d_words || d + d_words > 255)))
        fail("matrix multiply-add register vector is not admitted");
    auto source_register = [](unsigned base, unsigned r) { return base == 255 ? 255u : base + r; };
    // One operation per instruction: the back end lowers it to the
    // SIMD-group matrix hardware.
    std::vector<ir::Operand> inputs{lane};
    for (unsigned r = 0; r < 4; ++r) inputs.push_back(reg(source_register(a, r)));
    for (unsigned r = 0; r < 2; ++r) inputs.push_back(reg(source_register(b, r)));
    for (unsigned r = 0; r < c_words; ++r) inputs.push_back(reg(source_register(c, r)));
    if (p_.values.size() + d_words > p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
    std::vector<ir::ValueId> results;
    for (unsigned i = 0; i < d_words; ++i) {
        results.push_back(ir::ValueId{static_cast<std::uint32_t>(p_.values.size())});
        p_.values.push_back({ir::ValueType::word32});
    }
    operation({{}, ir::Opcode::matrix_multiply_add, {}, single, std::move(inputs), {results.begin(), results.end()},
               ir::SubgroupInfo{ir::ShuffleKind::index, 32}});
    if (d != 255) for (unsigned i = 0; i < d_words; ++i) write(d + i, results[i]);
    leave(fields); return true;
}
} // namespace armigo::shader_native::detail
