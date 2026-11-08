// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once
#include "types.h"
#include <array>
#include <cstddef>
#include <string_view>

namespace armigo::shader::pipeline {
enum class Effects : std::uint32_t {
    none = 0, read_memory = 1U << 0, write_memory = 1U << 1,
    read_io = 1U << 2, write_io = 1U << 3, texture = 1U << 4,
    derivative = 1U << 5, discard = 1U << 6, barrier = 1U << 7,
    atomic = 1U << 8, subgroup = 1U << 9, call = 1U << 10
};
constexpr Effects operator|(Effects a, Effects b) {
    return static_cast<Effects>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
constexpr Effects operator&(Effects a, Effects b) {
    return static_cast<Effects>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}
constexpr bool includes(Effects set, Effects required) { return (set & required) == required; }
enum class InfoKind : std::uint8_t { none, conversion, comparison, memory, texture, io, call, subgroup };
struct Arity {
    std::uint16_t minimum, maximum;
    constexpr bool accepts(std::size_t count) const { return count >= minimum && count <= maximum; }
};
// This is the authoritative scalar opcode vocabulary and signature table.
// Domain/width come from NumericMode. Terminators and phis have separate types.
#define ARMIGO_PIPELINE_OPCODES(X) \
    X(copy,                 1,1, 1,1, none,       none) \
    X(absolute,             1,1, 1,1, none,       none) \
    X(negate,               1,1, 1,1, none,       none) \
    X(add,                  2,2, 1,1, none,       none) \
    X(subtract,             2,2, 1,1, none,       none) \
    X(multiply,             2,2, 1,1, none,       none) \
    X(divide,               2,2, 1,1, none,       none) \
    X(remainder,            2,2, 1,1, none,       none) \
    X(multiply_high,        2,2, 1,1, none,       none) \
    X(add_carry,            3,3, 2,2, none,       none) \
    X(subtract_borrow,      3,3, 2,2, none,       none) \
    X(minimum,              2,2, 1,1, none,       none) \
    X(maximum,              2,2, 1,1, none,       none) \
    X(clamp,                3,3, 1,1, none,       none) \
    X(fused_multiply_add,   3,3, 1,1, none,       none) \
    X(reciprocal,           1,1, 1,1, none,       none) \
    X(reciprocal_sqrt,      1,1, 1,1, none,       none) \
    X(square_root,          1,1, 1,1, none,       none) \
    X(sine,                 1,1, 1,1, none,       none) \
    X(cosine,               1,1, 1,1, none,       none) \
    X(exp2,                 1,1, 1,1, none,       none) \
    X(log2,                 1,1, 1,1, none,       none) \
    X(range_reduce_sincos,  1,1, 1,1, none,       none) \
    X(range_reduce_exp2,    1,1, 1,1, none,       none) \
    X(ceiling,              1,1, 1,1, none,       none) \
    X(floor,                1,1, 1,1, none,       none) \
    X(round,                1,1, 1,1, none,       none) \
    X(truncate,             1,1, 1,1, none,       none) \
    X(is_nan,               1,1, 1,1, none,       none) \
    X(compare,              2,2, 1,1, comparison, none) \
    X(select,               3,3, 1,1, none,       none) \
    X(convert,              1,1, 1,1, conversion, none) \
    X(bitcast,              1,1, 1,1, conversion, none) \
    X(predicate_to_mask,    1,1, 1,1, none,       none) \
    X(mask_to_predicate,    1,1, 1,1, none,       none) \
    X(bit_and,              2,2, 1,1, none,       none) \
    X(bit_or,               2,2, 1,1, none,       none) \
    X(bit_xor,              2,2, 1,1, none,       none) \
    X(bit_not,              1,1, 1,1, none,       none) \
    X(logical_and,          2,2, 1,1, none,       none) \
    X(logical_or,           2,2, 1,1, none,       none) \
    X(logical_xor,          2,2, 1,1, none,       none) \
    X(logical_not,          1,1, 1,1, none,       none) \
    X(shift_left,           2,2, 1,1, none,       none) \
    X(shift_right,          2,2, 1,1, none,       none) \
    X(funnel_shift_left,    3,3, 1,1, none,       none) \
    X(funnel_shift_right,   3,3, 1,1, none,       none) \
    X(bit_extract,          3,3, 1,1, none,       none) \
    X(bit_insert,           4,4, 1,1, none,       none) \
    X(bit_reverse,          1,1, 1,1, none,       none) \
    X(bit_count,            1,1, 1,1, none,       none) \
    X(find_lsb,             1,1, 1,1, none,       none) \
    X(find_msb,             1,1, 1,1, none,       none) \
    X(pack_half2,           2,2, 1,1, conversion, none) \
    X(unpack_half2,         1,1, 2,2, conversion, none) \
    X(pack_word64,          2,2, 1,1, none,       none) \
    X(unpack_word64,        1,1, 2,2, none,       none) \
    X(load,                 0,26, 1,4, memory,    read_memory) \
    X(store,                1,30, 0,0, memory,    write_memory) \
    X(atomic,               1,5, 1,1, memory,     atomic) \
    X(load_io,              0,2, 1,4, io,         read_io) \
    X(store_io,             1,6, 0,0, io,         write_io) \
    X(interpolate,          0,4, 1,4, io,         read_io) \
    X(texture_sample,       1,32,1,4, texture,    texture) \
    X(texture_query,        0,8, 1,4, texture,    texture) \
    X(image_load,           1,8, 1,4, texture,    read_memory) \
    X(image_store,          2,12,0,0, texture,    write_memory) \
    X(image_atomic,         2,12,1,1, texture,    atomic) \
    X(derivative_x,         1,1, 1,1, none,       derivative) \
    X(derivative_y,         1,1, 1,1, none,       derivative) \
    X(discard,              0,0, 0,0, none,       discard) \
    X(memory_barrier,       0,0, 0,0, memory,     barrier) \
    X(barrier,              0,0, 0,0, memory,     barrier) \
    X(shuffle,              3,3, 2,2, subgroup,   subgroup) \
    X(ballot,               1,1, 1,1, subgroup,   subgroup) \
    X(vote_all,             1,1, 1,1, subgroup,   subgroup) \
    X(vote_any,             1,1, 1,1, subgroup,   subgroup) \
    X(vote_equal,           1,1, 1,1, subgroup,   subgroup) \
    X(subgroup_reduce,      1,1, 1,1, subgroup,   subgroup) \
    X(call,                 0,256,0,256,call,     call) \
    X(matrix_multiply_add,  9,9, 2,2, subgroup,   subgroup) \
    X(matrix_load,          3,3, 1,4, subgroup,   barrier)

enum class Opcode : std::uint16_t {
#define ARMIGO_ENUM(name, ...) name,
    ARMIGO_PIPELINE_OPCODES(ARMIGO_ENUM)
#undef ARMIGO_ENUM
    count
};
struct OpcodeSignature {
    std::string_view name;
    Arity inputs, outputs;
    InfoKind info;
    Effects required_effects;
};
inline constexpr auto opcode_signatures = std::to_array<OpcodeSignature>({
#define ARMIGO_SIGNATURE(name, imin, imax, omin, omax, info, effects) \
    {#name, {imin, imax}, {omin, omax}, InfoKind::info, Effects::effects},
    ARMIGO_PIPELINE_OPCODES(ARMIGO_SIGNATURE)
#undef ARMIGO_SIGNATURE
});
#undef ARMIGO_PIPELINE_OPCODES
[[nodiscard]] constexpr const OpcodeSignature* signature(Opcode opcode) {
    const auto index = static_cast<std::size_t>(opcode);
    return index < opcode_signatures.size() ? &opcode_signatures[index] : nullptr;
}
// Fixed operand roles and type rules are authoritative alongside arity.
// "numeric" uses NumericMode::scalar storage for every input and result.
// Variable signatures use explicitly indexed roles in OperationInfo.
enum class OperandSchema : std::uint8_t {
    numeric, copy, comparison, select, conversion, bitcast, carry, shift, funnel,
    extract, insert, pack_half, unpack_half, pack_word64, unpack_word64,
    predicate_logic, predicate_to_mask, mask_to_predicate, is_nan,
    shuffle, ballot, vote, memory, texture, io, call, none,
    // Warp matrix multiply-add 16x8x16 with binary16 sources and
    // accumulator: the lane id, A (4 words), B (2) and C (2 packed words)
    // in, D (2 packed words) out. Warp matrix load: the lane id, the lane's
    // shared-memory row address and the transposed flag (0 or 1) in, one
    // word per 8x8 block out; it orders shared memory as a barrier does.
    matrix,
    // A word reduced across the active subgroup lanes: one word in, the
    // reduction (SubgroupInfo::reduction, signedness from the numeric mode)
    // out.
    reduction
};
[[nodiscard]] constexpr OperandSchema operand_schema(Opcode opcode) {
    switch (opcode) {
    case Opcode::copy: return OperandSchema::copy;
    case Opcode::compare: return OperandSchema::comparison;
    case Opcode::select: return OperandSchema::select;
    case Opcode::convert: return OperandSchema::conversion;
    case Opcode::bitcast: return OperandSchema::bitcast;
    case Opcode::add_carry: case Opcode::subtract_borrow: return OperandSchema::carry;
    case Opcode::shift_left: case Opcode::shift_right: return OperandSchema::shift;
    case Opcode::funnel_shift_left: case Opcode::funnel_shift_right: return OperandSchema::funnel;
    case Opcode::bit_extract: return OperandSchema::extract;
    case Opcode::bit_insert: return OperandSchema::insert;
    case Opcode::pack_half2: return OperandSchema::pack_half;
    case Opcode::unpack_half2: return OperandSchema::unpack_half;
    case Opcode::pack_word64: return OperandSchema::pack_word64;
    case Opcode::unpack_word64: return OperandSchema::unpack_word64;
    case Opcode::logical_and: case Opcode::logical_or: case Opcode::logical_xor:
    case Opcode::logical_not: return OperandSchema::predicate_logic;
    case Opcode::predicate_to_mask: return OperandSchema::predicate_to_mask;
    case Opcode::mask_to_predicate: return OperandSchema::mask_to_predicate;
    case Opcode::is_nan: return OperandSchema::is_nan;
    case Opcode::shuffle: return OperandSchema::shuffle;
    case Opcode::ballot: return OperandSchema::ballot;
    case Opcode::vote_all: case Opcode::vote_any: case Opcode::vote_equal: return OperandSchema::vote;
    case Opcode::subgroup_reduce: return OperandSchema::reduction;
    case Opcode::load: case Opcode::store: case Opcode::atomic:
    case Opcode::memory_barrier: case Opcode::barrier: return OperandSchema::memory;
    case Opcode::texture_sample: case Opcode::texture_query: case Opcode::image_load:
    case Opcode::image_store: case Opcode::image_atomic: return OperandSchema::texture;
    case Opcode::load_io: case Opcode::store_io: case Opcode::interpolate: return OperandSchema::io;
    case Opcode::call: return OperandSchema::call;
    case Opcode::matrix_multiply_add: case Opcode::matrix_load: return OperandSchema::matrix;
    case Opcode::discard: case Opcode::count: return OperandSchema::none;
    default: return OperandSchema::numeric;
    }
}
static_assert(operand_schema(Opcode::add_carry) == OperandSchema::carry);
static_assert(opcode_signatures.size() == static_cast<std::size_t>(Opcode::count));
static_assert(signature(Opcode::fused_multiply_add)->inputs.accepts(3));
static_assert(signature(Opcode::shuffle)->outputs.accepts(2));
static_assert(signature(Opcode::count) == nullptr);
} // namespace armigo::shader::pipeline
