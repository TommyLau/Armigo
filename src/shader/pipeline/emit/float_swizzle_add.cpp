// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The frozen opcode vocabulary has no aggregate swizzle-add operation, so the
// straight-line expression stays explicit here: the per-lane
// sign table selects word constants, and two multiplies and an add with the
// decoded numeric mode reproduce the defined result bit for bit, including
// the multiply-by-zero term of the move code (a NaN or infinity first source
// therefore propagates through code 3, and a
// negative-zero second source gains the first source's sign). FTZ rides the
// mode of every arithmetic step; a directed rounding mode reaches the MSL
// generator, which rejects it with a structured diagnostic. The NDV modifier
// is decoded and preserved but does not
// alter the emitted expression; its hardware effect is unverified.

#include "pipeline/dispatch.h"
#include "pipeline/emit/float_helpers.h"
#include "pipeline/emit/integer_emit.h"

namespace armigo::shader::pipeline {
namespace {
using namespace integer_emit;

// The PN word selects a 2-bit code per group-of-four lane: 0 adds, 1 reverses
// the subtraction, 2 subtracts, and 3 moves the second source. The sign
// constants implement the per-lane sign tables.
Operand swizzle_sign(bool first_source, std::uint32_t code) {
    static constexpr std::uint32_t x_table[4] = {float_emit::float_one_bits, 0xbf800000U,
                                                 float_emit::float_one_bits, float_emit::float_zero_bits};
    static constexpr std::uint32_t y_table[4] = {float_emit::float_one_bits, float_emit::float_one_bits,
                                                 0xbf800000U, float_emit::float_one_bits};
    return float_emit::word(first_source ? x_table[code] : y_table[code]);
}
} // namespace

// FSWZADD: per-lane signed combination of two registers with the operation
// code taken from the PN word by the thread's position within its group of
// four lanes. The lane read is the fragment quad position; the result keeps
// the decoded rounding, flush and precision mode on every arithmetic step,
// and the CC zero/sign flags follow floating-point zero/sign semantics.
Status emit_fswzadd(EmitterContext& context) {
    ARMIGO_FLOAT_TRY(mode, float_emit::float_mode(context));
    const auto& instruction = context.instruction();
    ARMIGO_FLOAT_TRY(packed, decoded_field(instruction, DecodedField::permutation_mode));
    const auto pn_word = static_cast<std::uint32_t>(packed & 0xff);
    ARMIGO_FLOAT_TRY(x, float_emit::source_operand(context, instruction, SourceRole::a));
    ARMIGO_FLOAT_TRY(y, float_emit::source_operand(context, instruction, SourceRole::b));
    ARMIGO_FLOAT_TRY(cc, float_emit::write_cc_enabled(context));

    // The frozen shuffle schema has no lane input, so the MSL generator reads
    // the declared builtin entry parameter. Recording the load here puts the
    // read in the resource plan, which declares the entry signature.
    IoInfo lane;
    lane.semantic = IoSemantic::quad_lane_id;
    lane.type = {ScalarType::u32, 1};
    lane.component_mask = 1;
    const ValueType word = ValueType::word32;
    ARMIGO_FLOAT_TRY(lane_read, context.load_io(lane, {}, NumericMode{Domain::unsigned_integer, ScalarType::u32},
                                                std::span{&word, 1}));

    // lut_index = (pn_word >> ((lane & 3) * 2)) & 3 selects the lane's 2-bit
    // code using integer-word arithmetic.
    auto lane_pair = context.binary(Opcode::bit_and, lane_read.front(),
                                    float_emit::word(3), bits32, ValueType::word32);
    if (lane_pair.index() == 1) return std::move(std::get<1>(lane_pair));
    auto doubled = context.binary(Opcode::shift_left, std::get<0>(lane_pair),
                                  float_emit::word(1), bits32, ValueType::word32);
    if (doubled.index() == 1) return std::move(std::get<1>(doubled));
    auto shifted = context.binary(Opcode::shift_right, float_emit::word(pn_word),
                                  std::get<0>(doubled), bits32, ValueType::word32);
    if (shifted.index() == 1) return std::move(std::get<1>(shifted));
    auto index = context.binary(Opcode::bit_and, std::get<0>(shifted),
                                float_emit::word(3), bits32, ValueType::word32);
    if (index.index() == 1) return std::move(std::get<1>(index));

    // One comparison per table code except the last, which is the select
    // fallback; all four codes stay reachable through the PN word. The index
    // compares as an unsigned word, the interpreted domain for integer
    // comparisons.
    Operand codes[3];
    for (unsigned code = 0; code < 3; ++code) {
        auto match = context.compare(std::get<0>(index), float_emit::word(code),
                                     ComparisonInfo{Compare::equal, false}, word32_unsigned);
        if (match.index() == 1) return std::move(std::get<1>(match));
        codes[code] = std::get<0>(match);
    }
    auto sign_select = [&](bool first_source) -> Result<Operand> {
        Operand signs[4];
        for (unsigned code = 0; code < 4; ++code) signs[code] = swizzle_sign(first_source, code);
        auto third = context.select(codes[2], signs[2], signs[3], ValueType::word32);
        if (third.index() == 1) return std::move(std::get<1>(third));
        auto second = context.select(codes[1], signs[1], std::get<0>(third), ValueType::word32);
        if (second.index() == 1) return std::move(std::get<1>(second));
        return context.select(codes[0], signs[0], std::get<0>(second), ValueType::word32);
    };
    ARMIGO_FLOAT_TRY(x_sign, sign_select(true));
    ARMIGO_FLOAT_TRY(y_sign, sign_select(false));

    auto scaled_x = context.binary(Opcode::multiply, x, x_sign, mode, storage_type(mode.scalar));
    if (scaled_x.index() == 1) return std::move(std::get<1>(scaled_x));
    auto scaled_y = context.binary(Opcode::multiply, y, y_sign, mode, storage_type(mode.scalar));
    if (scaled_y.index() == 1) return std::move(std::get<1>(scaled_y));
    auto result = context.binary(Opcode::add, std::get<0>(scaled_x), std::get<0>(scaled_y),
                                 mode, storage_type(mode.scalar));
    if (result.index() == 1) return std::move(std::get<1>(result));
    const Status written = float_emit::write_destination(context, instruction, std::get<0>(result));
    if (written.index() == 1) return written;
    return float_emit::write_float_flags(context, std::get<0>(result), mode, cc);
}
} // namespace armigo::shader::pipeline
