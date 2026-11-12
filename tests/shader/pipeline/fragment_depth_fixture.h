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
#include "pipeline/phases.h"
#include <bit>
#include <stdexcept>

namespace armigo::shader::test::fragment_depth {
using namespace pipeline;
inline constexpr TextureKey depth_key{2, 16};
struct Facts final : FactSource {
    Knowledge<FactValue> lookup(FactKey key) const override {
        const EvidenceOrigin origin{OriginKind::explicit_option, "synthetic-stored-depth"};
        if (key == fact_query::texture(depth_key).key())
            return Known<FactValue>{TextureDescriptor{{TextureDimension::d2, false, false}, ScalarType::f32,
                ComponentFormat::depth, true, false, 1, "synthetic-depth32"}, origin};
        if (key.kind == FactKind::fragment_output_type)
            return Known<FactValue>{IoType{ScalarType::f32, 4}, origin};
        return Unknown{"No synthetic draw-state fact"};
    }
};
template<class T> T take(Result<T> value) {
    if (auto* error = std::get_if<Diagnostic>(&value)) throw std::runtime_error(error->message);
    return std::get<T>(std::move(value));
}
inline Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
inline Program program(bool texture_first = true, bool texture_comparison = true, bool scaled = false) {
    Program p; p.stage = Stage::fragment; p.entry = FunctionId{0};
    p.functions.push_back({0, BlockId{0}, {BlockId{0}}, {}, {}});
    p.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
    const auto emit = [&](Operation op, std::vector<ValueType> types) {
        std::vector<Operand> outputs;
        for (auto type : types) { outputs.push_back(ValueId{static_cast<std::uint32_t>(p.values.size())}); p.values.push_back({type}); }
        op.outputs = outputs; op.effects = required_effects(op);
        p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
        p.operations.push_back(std::move(op)); return outputs;
    };
    const auto position = [&](std::uint8_t mask, unsigned count) {
        IoInfo io; io.semantic = IoSemantic::fragment_coord; io.type = {ScalarType::f32, 4}; io.component_mask = mask;
        io.interpolation = {InterpolationMode::flat, InterpolationLocation::center};
        return emit({{}, Opcode::load_io, {}, {}, {}, {}, io}, std::vector<ValueType>(count, ValueType::word32));
    };
    const auto xy = position(3, 2);
    const auto integer = [&](Operand input) {
        return emit({{}, Opcode::convert, {}, {Domain::floating, ScalarType::f32, Rounding::toward_zero}, {input}, {},
            ConversionInfo{ScalarType::f32, ScalarType::u32}}, {ValueType::word32})[0];
    };
    const auto x = integer(xy[0]), y = integer(xy[1]);
    const auto depth = [&] {
        TextureInfo texture; texture.texture = depth_key; texture.mode = TextureMode::fetch;
        texture.coordinates = {InputIndex{0}, InputIndex{1}}; texture.level_or_bias = InputIndex{2}; texture.component_mask = 1;
        return emit({{}, Opcode::texture_sample, {}, {Domain::floating, ScalarType::f32, Rounding::nearest_even},
            {x, y, word(0)}, {}, texture}, {ValueType::word32})[0];
    };
    Operand d, z;
    if (texture_first) { d = depth(); z = position(4, 1)[0]; }
    else { z = position(4, 1)[0]; d = depth(); }
    if (!texture_comparison) d = word(0x3f000000u);
    if (scaled) {
        const NumericMode fp{Domain::floating, ScalarType::f32, Rounding::nearest_even};
        z = emit({{}, Opcode::multiply, {}, fp, {z, word(0x40000000u)}, {}, {}}, {ValueType::word32})[0];
        d = emit({{}, Opcode::multiply, {}, fp, {d, word(0x40000000u)}, {}, {}}, {ValueType::word32})[0];
    }
    const auto predicate = [&](Compare relation) {
        return emit({{}, Opcode::compare, {}, {Domain::floating, ScalarType::f32, Rounding::nearest_even}, {z, d}, {},
            ComparisonInfo{relation, false}}, {ValueType::predicate})[0];
    };
    const auto floating = [&](Operand condition) {
        return emit({{}, Opcode::select, {}, {}, {condition, word(0x3f800000u), word(0)}, {}, {}}, {ValueType::word32})[0];
    };
    const auto above = floating(predicate(Compare::greater)), below = floating(predicate(Compare::less));
    const Operand colors[]{word(0x3f800000u), above, below, word(0x3f800000u)};
    for (unsigned component = 0; component < 4; ++component) {
        IoInfo io; io.semantic = IoSemantic::color; io.location = 0; io.component_mask = std::uint8_t(1u << component);
        io.type = {ScalarType::f32, 4}; io.output_storage = true; io.data = {InputIndex{0}};
        emit({{}, Opcode::store_io, {}, {}, {colors[component]}, {}, io}, {});
    }
    return p;
}
inline MslArtifact translate(Program p) {
    Facts source; FactReader facts(source);
    const MetalProfile profile{"synthetic-stored-depth", 1, 31, 128, 16, true, false, true, false, false, false};
    auto ssa = take(construct_ssa(RegisterProgram{std::move(p)}, {}));
    auto legal = take(legalize(std::move(ssa), facts, profile));
    auto structured = take(structure(std::move(legal), {}));
    auto plan = take(plan_resources(structured, profile));
    return take(emit_msl(structured, plan, {}));
}
} // namespace armigo::shader::test::fragment_depth
