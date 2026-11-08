// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Native special registers, votes and predicate/uniform moves.
// R2UR copies one invocation's vector register into its uniform local.
// P2R gathers P0-P6 into one byte of the source register. S2R admits the
// fragment lane id, the lane masks derived from it, and the Y-direction
// word. S2UR writes a uniform register and admits only the compute group
// id, which has one value in every lane. VOTE and VOTEU reduce a per-lane
// predicate across the fragment SIMD group, REDUX a word into a uniform
// register.

#include "builder.h"

namespace armigo::shader_native::detail {
namespace {
constexpr ir::NumericMode word_bits{ir::Domain::bits, ir::ScalarType::u32};
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode boolean{ir::Domain::boolean, ir::ScalarType::boolean};
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
constexpr ir::SubgroupInfo subgroup{ir::ShuffleKind::index, 32};
} // namespace

bool Builder::system_value(Fields& fields) {
    switch (opcode_) {
    case 0x3c2: { // R2UR URd, Ra
        // Each invocation writes its own vector register. Bits 22-23 extend
        // the uniform index and are zero; bits 81-83 are the absent predicate
        // result and hold the true-predicate index.
        enter(fields, false);
        const auto destination = static_cast<unsigned>(fields.take(16, 6));
        if (fields.take(22, 2)) fail("uniform destination field extension is not admitted");
        const auto source = reg(static_cast<unsigned>(fields.take(24, 8)));
        if (fields.take(81, 3) != 7) fail("register-to-uniform predicate result is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write_uniform(destination, source);
        leave(fields); return true;
    }
    case 0x803: case 0x203: { // P2R Rd, Ra, mask, byte
        // The mask selects bits of one byte of Ra (the byte is bits 76-77).
        // Each selected bit 0-6 is replaced by predicate P0-P6. Bit 7 of that
        // byte is not a predicate, so a mask bit there clears it. Mask bits
        // above bit 7 are ignored. The mask is an immediate (0x803, bits
        // 32-63) or a register (0x203, bits 32-39).
        enter(fields, false);
        const auto destination = static_cast<unsigned>(fields.take(16, 8));
        const auto source = reg(static_cast<unsigned>(fields.take(24, 8)));
        const bool immediate = opcode_ == 0x803;
        const auto mask_bits = immediate ? static_cast<std::uint32_t>(fields.take(32, 32)) : 0;
        const auto mask_register = immediate ? 0u : static_cast<unsigned>(fields.take(32, 8));
        const auto byte = static_cast<unsigned>(fields.take(76, 2));
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::Operand mask = immediate ? word((mask_bits & 0xffu) << (byte * 8))
                                     : bits(ir::Opcode::bit_and, {reg(mask_register), word(0xff)});
        if (!immediate && byte) mask = bits(ir::Opcode::shift_left, {mask, word(byte * 8u)});
        ir::Operand insert = word(0);
        for (unsigned bit = 0; bit < 7; ++bit) {
            auto placed = bits(ir::Opcode::select, {predicate({PredicateFile::general, static_cast<std::uint8_t>(bit), false}),
                                                     word(1u << (bit + byte * 8)), word(0)}, unsigned_word);
            insert = bits(ir::Opcode::bit_or, {insert, placed});
        }
        auto kept = bits(ir::Opcode::bit_and, {source, bits(ir::Opcode::bit_not, {mask})});
        write(destination, bits(ir::Opcode::bit_or, {kept, bits(ir::Opcode::bit_and, {insert, mask})}));
        leave(fields); return true;
    }
    case 0x883: { // UP2UR URd, UPR, URZ, mask
        // P2R over the uniform predicates: each mask bit 0-6 (bits 32-63)
        // places UP0-UP6 at that bit of the uniform destination. Only the
        // established form is admitted: the source URZ (bits 24-31 = 63),
        // byte 0 (bits 76-77) and the uniform datapath (bit 91).
        enter(fields, true);
        const auto destination = static_cast<unsigned>(fields.take(16, 6));
        if (fields.take(22, 2)) fail("uniform destination field extension is not admitted");
        if (fields.take(24, 8) != 63) fail("uniform predicate move with a register source is not admitted");
        const auto mask = static_cast<std::uint32_t>(fields.take(32, 32));
        if (!fields.flag(91) || fields.take(76, 2)) fail("uniform predicate move mode is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::Operand result = word(0);
        for (unsigned bit = 0; bit < 7; ++bit) if ((mask >> bit) & 1)
            result = bits(ir::Opcode::bit_or, {result, bits(ir::Opcode::select,
                {uniform_predicate(bit, false), word(1u << bit), word(0)}, unsigned_word)});
        write_uniform(destination, result);
        leave(fields); return true;
    }
    case 0x919: { // S2R Rd, SR
        // Selector bits 72-79 use the graphics numbering. Fragment and compute
        // stages read the lane id and the five lane masks (the lane's own
        // bit, the bits below it, and their complements). Vertex and fragment
        // stages read the Y-direction word, the bit pattern of +1 or -1.
        // Compute stages read the thread id in the group (33-35), the group
        // id (37-39) and the packed thread id (32: x in bits 0-15, y from bit
        // 16, z from bit 26). A fragment stage reads its shading rate (132):
        // with the pixel rate proven disabled every fragment is 1x1, width
        // and height in bytes 1 and 2 (0x00010100). Every other selector is
        // refused.
        enter(fields, false);
        const auto destination = static_cast<unsigned>(fields.take(16, 8));
        const auto selector = static_cast<unsigned>(fields.take(72, 8));
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        auto load_word = [&](ir::IoSemantic semantic, unsigned component = 0) {
            ir::IoInfo io; io.semantic = semantic; io.component_mask = static_cast<std::uint8_t>(1u << component);
            io.type = {ir::ScalarType::u32, static_cast<std::uint8_t>(semantic == ir::IoSemantic::thread_id ||
                                                                     semantic == ir::IoSemantic::workgroup_id ? 3 : 1)};
            return temporary(ir::Opcode::load_io, {}, unsigned_word, io);
        };
        const bool compute = source_.stage == Stage::compute;
        const bool lane = selector == 0 || (selector >= 56 && selector <= 60);
        const bool thread = selector == 32 || (selector >= 33 && selector <= 35);
        const bool group = selector >= 37 && selector <= 39;
        if (lane && source_.stage != Stage::fragment && !workgroup_stage())
            fail(selector == 0 ? "lane id requires a fragment or workgroup stage" : "lane mask requires a fragment or workgroup stage");
        if (selector == 18 && source_.stage != Stage::vertex && source_.stage != Stage::fragment)
            fail("Y direction requires a graphics stage");
        if (thread && !workgroup_stage()) fail("thread ids require a workgroup stage");
        // A task or mesh stage reads its workgroup's number from the stage
        // buffer; the group id register of those stages is not established.
        if (group && !compute) fail("group ids require a compute stage");
        if (selector == 132) {
            if (source_.stage != Stage::fragment) fail("shading rate requires a fragment stage");
            if (rate_ != PixelRateState::disabled)
                fail("shading rate requires proven disabled pixel-rate state",
                     rate_ == PixelRateState::unknown ? ir::DiagnosticCode::missing_fact : ir::DiagnosticCode::unsupported_capability);
            requires_disabled_rate_ = true;
            write(destination, word(0x00010100u));
        } else if (selector == 18) write(destination, load_word(ir::IoSemantic::y_direction));
        else if (selector >= 33 && selector <= 35) write(destination, load_word(ir::IoSemantic::thread_id, selector - 33));
        else if (selector >= 37 && selector <= 39) write(destination, load_word(ir::IoSemantic::workgroup_id, selector - 37));
        else if (selector == 32) {
            auto x = load_word(ir::IoSemantic::thread_id, 0), y = load_word(ir::IoSemantic::thread_id, 1);
            auto z = load_word(ir::IoSemantic::thread_id, 2);
            auto yz = bits(ir::Opcode::bit_or, {bits(ir::Opcode::shift_left, {y, word(16)}), bits(ir::Opcode::shift_left, {z, word(26)})});
            write(destination, bits(ir::Opcode::bit_or, {x, yz}));
        } else if (lane) {
            auto id = load_word(ir::IoSemantic::subgroup_lane_id);
            if (selector == 0) write(destination, id);
            else {
                auto bit = bits(ir::Opcode::shift_left, {word(1), id});
                auto less = bits(ir::Opcode::subtract, {bit, word(1)}, unsigned_word);
                ir::Operand result = bit;
                if (selector == 57) result = less;
                else if (selector == 58) result = bits(ir::Opcode::bit_or, {less, bit});
                else if (selector == 59) result = bits(ir::Opcode::bit_not, {bits(ir::Opcode::bit_or, {less, bit})});
                else if (selector == 60) result = bits(ir::Opcode::bit_not, {less});
                write(destination, result);
            }
        } else fail("special register is not admitted");
        leave(fields); return true;
    }
    case 0x9c3: { // S2UR URd, SR
        // Selector bits 72-79 use the same numbering as S2R. The destination
        // is one uniform register, so the value has to be the same in every
        // lane. Compute group ids are. A thread id, packed id, lane id or
        // lane mask is lane-dependent and is refused, as is any other selector.
        if (source_.stage != Stage::compute) fail("uniform special register requires a compute stage");
        enter(fields, true);
        const auto destination = static_cast<unsigned>(fields.take(16, 6));
        if (fields.take(22, 2)) fail("uniform destination field extension is not admitted");
        const auto selector = static_cast<unsigned>(fields.take(72, 8));
        if (selector < 37 || selector > 39) fail("uniform special register is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::IoInfo io; io.semantic = ir::IoSemantic::workgroup_id;
        io.component_mask = static_cast<std::uint8_t>(1u << (selector - 37));
        io.type = {ir::ScalarType::u32, 3};
        write_uniform(destination, temporary(ir::Opcode::load_io, {}, unsigned_word, io));
        leave(fields); return true;
    }
    case 0x3c4: { // REDUX URd, Ra
        // Reduces Ra over the active lanes into a uniform register: the
        // operation is bits 78-80 (1 OR, 2 XOR, 3 SUM modulo 2^32, 4 MIN,
        // 5 MAX; the extrema signed with bit 73), the destination bits 16-21.
        // Inactive lanes take no part. Unguarded only; fragment and compute
        // stages, as Metal offers SIMD-group functions to those.
        if (source_.stage != Stage::fragment && !workgroup_stage()) fail("warp reduction requires a fragment or workgroup stage");
        if (fields.raw().bits(12, 4) != 7) fail("guarded warp reduction is not admitted");
        enter(fields, false);
        const auto destination = static_cast<unsigned>(fields.take(16, 6));
        if (fields.take(22, 2)) fail("uniform destination field extension is not admitted");
        const auto source = reg(static_cast<unsigned>(fields.take(24, 8)));
        const bool signed_order = fields.flag(73);
        const auto operation_kind = static_cast<unsigned>(fields.take(78, 3));
        static constexpr std::array<std::optional<ir::AtomicKind>, 6> kinds{std::nullopt, ir::AtomicKind::bit_or,
            ir::AtomicKind::bit_xor, ir::AtomicKind::add, ir::AtomicKind::minimum, ir::AtomicKind::maximum};
        if (operation_kind >= kinds.size() || !kinds[operation_kind]) fail("warp reduction operation is not admitted");
        if (signed_order && operation_kind < 4) fail("signed warp reduction other than an extremum is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::SubgroupInfo reduction = subgroup;
        reduction.reduction = kinds[operation_kind];
        const ir::NumericMode mode = signed_order ? ir::NumericMode{ir::Domain::signed_integer, ir::ScalarType::s32} : unsigned_word;
        auto value = temporary(ir::Opcode::subgroup_reduce, {source}, mode, reduction);
        write_uniform(destination, value);
        leave(fields); return true;
    }
    case 0x806: case 0x886: { // VOTE Rd, Pd, P / VOTEU URd, UPd, P
        // Mode bits 72-73 select all, any or equal. The predicate source
        // (bits 87-90) and the guard are per-lane for both forms. The ballot
        // is a register (VOTE, bits 16-23) or a uniform register (VOTEU,
        // bits 16-21). The vote predicate is bits 81-83, a uniform predicate
        // for VOTEU. A true-predicate or zero-register destination skips that
        // result. Fragment and compute stages only, as Metal offers SIMD-group
        // functions to those.
        if (source_.stage != Stage::fragment && !workgroup_stage()) fail("vote requires a fragment or workgroup stage");
        const bool uniform = opcode_ == 0x886;
        enter(fields, false);
        const auto ballot = static_cast<unsigned>(uniform ? fields.take(16, 6) : fields.take(16, 8));
        if (uniform && fields.take(22, 2)) fail("uniform destination field extension is not admitted");
        const auto mode = static_cast<unsigned>(fields.take(72, 2));
        const auto vote_destination = static_cast<unsigned>(fields.take(81, 3));
        const auto source = predicate_source(fields, 87, false);
        if (mode > 2) fail("vote mode is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        if (vote_destination != 7) {
            constexpr ir::Opcode opcodes[]{ir::Opcode::vote_all, ir::Opcode::vote_any, ir::Opcode::vote_equal};
            auto vote = temporary(opcodes[mode], {source}, boolean, subgroup, ir::ValueType::predicate);
            if (uniform) write_uniform_predicate(vote_destination, vote); else write_predicate(vote_destination, vote);
        }
        if (ballot != (uniform ? 63u : 255u)) {
            auto mask = temporary(ir::Opcode::ballot, {source}, word_bits, subgroup);
            if (uniform) write_uniform(ballot, mask); else write(ballot, mask);
        }
        leave(fields); return true;
    }
    default: return false;
    }
}
} // namespace armigo::shader_native::detail
