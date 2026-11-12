// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include "fragment_depth_fixture.h"

namespace armigo::shader::test::early_fragment {
using namespace pipeline;
enum class Behaviour { read_only, store, atomic, discard, depth, sample_mask };
inline Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
inline Program program(Behaviour behaviour = Behaviour::read_only) {
    Program p; p.stage = Stage::fragment; p.entry = FunctionId{0};
    p.functions.push_back({0, BlockId{0}, {BlockId{0}}, {}, {}});
    p.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
    const auto emit = [&](Opcode opcode, std::vector<Operand> inputs, NumericMode mode,
                          OperationInfo info, std::vector<ValueType> types) {
        std::vector<Operand> outputs;
        for (auto type : types) { outputs.push_back(ValueId{static_cast<std::uint32_t>(p.values.size())}); p.values.push_back({type}); }
        Operation op{{}, opcode, {}, mode, std::move(inputs), outputs, std::move(info)};
        op.effects = required_effects(op);
        p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
        p.operations.push_back(std::move(op)); return outputs;
    };
    const NumericMode fp{Domain::floating, ScalarType::f32, Rounding::nearest_even};
    const NumericMode u32{Domain::unsigned_integer, ScalarType::u32};
    IoInfo position; position.semantic = IoSemantic::fragment_coord; position.type = {ScalarType::f32, 4};
    position.component_mask = 15; position.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
    auto xyzw = emit(Opcode::load_io, {}, {}, position, std::vector<ValueType>(4, ValueType::word32));
    auto sum = emit(Opcode::add, {xyzw[0], xyzw[1]}, fp, {}, {ValueType::word32})[0];
    sum = emit(Opcode::add, {sum, xyzw[2]}, fp, {}, {ValueType::word32})[0];
    sum = emit(Opcode::add, {sum, xyzw[3]}, fp, {}, {ValueType::word32})[0];
    auto index = emit(Opcode::convert, {sum}, {Domain::floating, ScalarType::f32, Rounding::toward_zero},
        ConversionInfo{ScalarType::f32, ScalarType::u32}, {ValueType::word32})[0];
    index = emit(Opcode::bit_and, {index, word(3)}, u32, {}, {ValueType::word32})[0];
    const auto low = emit(Opcode::shift_left, {index, word(4)}, u32, {}, {ValueType::word32})[0];
    MemoryInfo memory; memory.space = AddressSpace::guest; memory.access = Access::read;
    memory.element_bytes = 4; memory.elements = 4; memory.alignment = 16;
    memory.dynamic_byte_offset = InputIndex{0}; memory.global_address_high = InputIndex{1};
    const auto color = emit(Opcode::load, {low, word(1)}, u32, memory, std::vector<ValueType>(4, ValueType::word32));
    if (behaviour == Behaviour::store || behaviour == Behaviour::atomic) {
        memory.elements = 1; memory.alignment = 4; memory.access = Access::read_write; memory.data = {InputIndex{2}};
        if (behaviour == Behaviour::store) {
            memory.access = Access::write; emit(Opcode::store, {low, word(1), word(0)}, u32, memory, {});
        } else {
            memory.atomic = AtomicKind::add; memory.scope = MemoryScope::device;
            emit(Opcode::atomic, {low, word(1), word(1)}, u32, memory, {ValueType::word32});
        }
    }
    if (behaviour == Behaviour::discard) emit(Opcode::discard, {}, {}, {}, {});
    if (behaviour == Behaviour::depth || behaviour == Behaviour::sample_mask) {
        IoInfo io; io.semantic = behaviour == Behaviour::depth ? IoSemantic::fragment_depth : IoSemantic::sample_mask;
        io.type = {behaviour == Behaviour::depth ? ScalarType::f32 : ScalarType::u32, 1};
        io.component_mask = 1; io.output_storage = true; io.data = {InputIndex{0}};
        emit(Opcode::store_io, {word(behaviour == Behaviour::depth ? 0x3f000000u : 0xffffffffu)}, {}, io, {});
    }
    for (unsigned c = 0; c < 4; ++c) {
        IoInfo io; io.semantic = IoSemantic::color; io.type = {ScalarType::f32, 4};
        io.component_mask = std::uint8_t(1u << c); io.output_storage = true; io.data = {InputIndex{0}};
        emit(Opcode::store_io, {color[c]}, {}, io, {});
    }
    return p;
}
inline MslArtifact translate(Behaviour behaviour = Behaviour::read_only) {
    fragment_depth::Facts source; FactReader facts(source);
    const MetalProfile profile{"synthetic-early-fragment", 1, 31, 128, 16, true, false, true, false, false, false};
    auto ssa = fragment_depth::take(construct_ssa(RegisterProgram{program(behaviour)}, {}));
    auto legal = fragment_depth::take(legalize(std::move(ssa), facts, profile));
    auto structured = fragment_depth::take(structure(std::move(legal), {}));
    auto plan = fragment_depth::take(plan_resources(structured, profile));
    EmitOptions options;
    options.fragment_position = FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    return fragment_depth::take(emit_msl(structured, plan, options));
}
} // namespace armigo::shader::test::early_fragment
