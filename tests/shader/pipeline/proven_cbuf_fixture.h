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
#include <array>
#include <algorithm>
#include <stdexcept>

namespace armigo::shader::test::proven_cbuf {
using namespace pipeline;
enum class Offset { literal, mask, minimum, wrap, unknown, phi, loop };
struct Case {
    const char* name;
    Offset offset;
    std::uint32_t address;
    std::uint8_t bytes, elements;
    ConstantBufferPolicy policy;
    std::uint32_t window;
};
inline const std::array cases{
    Case{"literal", Offset::literal, 60, 4, 1, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"mask", Offset::mask, 60, 4, 1, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"minimum", Offset::minimum, 60, 4, 1, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"uint32_wrap", Offset::wrap, 0, 4, 1, ConstantBufferPolicy::proven_zero_words_v2, 4},
    Case{"negative", Offset::literal, 0xfffffffcu, 4, 1, ConstantBufferPolicy::padded_zero_words_v2, 65536},
    Case{"unknown", Offset::unknown, 0, 4, 1, ConstantBufferPolicy::padded_zero_words_v2, 65536},
    Case{"unaligned_u8", Offset::literal, 63, 1, 1, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"unaligned_u16", Offset::literal, 63, 2, 1, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"unaligned_u32", Offset::literal, 63, 4, 1, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"vector", Offset::literal, 48, 4, 4, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"wide", Offset::literal, 56, 8, 1, ConstantBufferPolicy::proven_zero_words_v2, 64},
    Case{"last_64k", Offset::literal, 65532, 4, 1, ConstantBufferPolicy::proven_zero_words_v2, 65536},
    Case{"cross_wide", Offset::literal, 65532, 8, 1, ConstantBufferPolicy::padded_zero_words_v2, 65536},
    Case{"cross_vector", Offset::literal, 65524, 4, 4, ConstantBufferPolicy::padded_zero_words_v2, 65536},
    Case{"phi", Offset::phi, 60, 4, 1, ConstantBufferPolicy::padded_zero_words_v2, 65536},
    Case{"loop", Offset::loop, 60, 4, 1, ConstantBufferPolicy::padded_zero_words_v2, 65536},
};
inline Operand word(std::uint32_t value) { return LiteralBits{value, ValueType::word32}; }
struct Facts final : FactSource {
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::fragment_output_type)
            return Known<FactValue>{IoType{ScalarType::u32, 4}, {OriginKind::explicit_option, "synthetic-proven-window"}};
        return Unknown{"No synthetic draw binding fact"};
    }
};
template<class T> T take(Result<T> result) {
    if (const auto* error = std::get_if<Diagnostic>(&result)) throw std::runtime_error(error->message);
    return std::get<T>(std::move(result));
}
inline MslArtifact translate(const Case& item) {
    Program p; p.stage = Stage::fragment; p.entry = FunctionId{0};
    p.functions.push_back({0, BlockId{0}, {BlockId{0}}, {}, {}});
    p.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
    unsigned block = 0;
    const auto value = [&](ValueType type) {
        const ValueId id{static_cast<std::uint32_t>(p.values.size())}; p.values.push_back({type}); return id;
    };
    const auto emit = [&](Opcode opcode, std::vector<Operand> inputs, OperationInfo info,
                          std::vector<ValueType> types, NumericMode mode = NumericMode{}) {
        std::vector<Operand> outputs; for (auto type : types) outputs.push_back(value(type));
        Operation op{{}, opcode, {}, mode, std::move(inputs), outputs, std::move(info)};
        op.effects = required_effects(op);
        p.blocks[block].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
        p.operations.push_back(std::move(op)); return outputs;
    };
    const NumericMode u32{Domain::unsigned_integer, ScalarType::u32};
    const auto binary = [&](Opcode op, Operand a, Operand b) { return emit(op, {a,b}, {}, {ValueType::word32}, u32)[0]; };
    const auto load = [&](unsigned slot, unsigned offset, std::optional<Operand> dynamic, unsigned bytes, unsigned elements) {
        MemoryInfo m; m.space = AddressSpace::constant_buffer; m.access = Access::read;
        m.constant_buffer_slot = slot; m.byte_offset = offset; m.element_bytes = bytes; m.elements = elements;
        m.alignment = std::min(4u, bytes); std::vector<Operand> inputs;
        if (dynamic) { m.dynamic_byte_offset = InputIndex{0}; inputs.push_back(*dynamic); }
        return emit(Opcode::load, inputs, m,
            std::vector<ValueType>(elements, bytes == 8 ? ValueType::word64 : ValueType::word32));
    };
    std::optional<Operand> offset;
    std::optional<ValueId> induction;
    if (item.offset == Offset::wrap) offset = binary(Opcode::add, word(UINT32_MAX), word(1));
    else if (item.offset != Offset::literal) {
        auto input = load(4, 0, {}, 4, 1)[0];
        if (item.offset == Offset::mask) offset = binary(Opcode::bit_and, input, word(item.address));
        else if (item.offset == Offset::minimum) offset = binary(Opcode::minimum, input, word(item.address));
        else if (item.offset == Offset::unknown) offset = input;
        else {
            const auto initial = emit(Opcode::copy, {word(0)}, {}, {ValueType::word32})[0];
            if (item.offset == Offset::phi) {
                auto condition = emit(Opcode::compare, {input,word(0)}, ComparisonInfo{Compare::not_equal,false}, {ValueType::predicate}, u32)[0];
                p.blocks[0].terminator = Branch{condition, BlockId{1}, BlockId{2}};
                for (unsigned n = 1; n < 4; ++n) p.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Jump{BlockId{3}}});
                block = 1; auto left = emit(Opcode::copy, {initial}, {}, {ValueType::word32})[0];
                block = 2; auto right = emit(Opcode::copy, {word(item.address)}, {}, {ValueType::word32})[0];
                block = 3; const auto joined = value(ValueType::word32);
                p.blocks[3].phis.push_back({{}, joined, {{BlockId{1},std::get<ValueId>(left)}, {BlockId{2},std::get<ValueId>(right)}}});
                offset = joined;
            } else {
                p.blocks[0].terminator = Jump{BlockId{1}};
                p.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
                p.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
                block = 1; induction = value(ValueType::word32); offset = *induction;
                p.blocks[1].phis.push_back({{}, *induction, {{BlockId{0},std::get<ValueId>(initial)}}});
            }
            p.functions[0].blocks.clear();
            for (unsigned n = 0; n < p.blocks.size(); ++n) p.functions[0].blocks.push_back(BlockId{n});
        }
    }
    auto color = load(3, offset ? 0 : item.address, offset, item.bytes, item.elements);
    if (item.bytes == 8) color = emit(Opcode::unpack_word64, color, {}, {ValueType::word32,ValueType::word32});
    if (induction) {
        auto next = binary(Opcode::add, *induction, word(4));
        p.blocks[1].phis[0].inputs.push_back({BlockId{1},std::get<ValueId>(next)});
        auto condition = emit(Opcode::compare, {next,word(item.address)}, ComparisonInfo{Compare::less_equal,false}, {ValueType::predicate}, u32)[0];
        p.blocks[1].terminator = Branch{condition,BlockId{1},BlockId{2}}; block = 2;
    }
    const auto repeated = color; while (color.size() < 4) color.push_back(repeated[color.size() % repeated.size()]);
    IoInfo io; io.semantic = IoSemantic::color; io.type = {ScalarType::u32,4}; io.component_mask = 15; io.output_storage = true;
    io.data = {InputIndex{0},InputIndex{1},InputIndex{2},InputIndex{3}};
    emit(Opcode::store_io, color, io, {}); p.blocks[block].terminator = Return{};
    Facts source; FactReader facts(source);
    const MetalProfile profile{"synthetic-proven-window",1,31,128,16,true,false,true,false,false,false};
    for (unsigned n = 0; n < p.blocks.size(); ++n) {
        const auto edge = [&](BlockId target) {
            auto& predecessors = p.blocks[target.value].predecessors;
            if (std::find(predecessors.begin(), predecessors.end(), BlockId{n}) == predecessors.end())
                predecessors.push_back(BlockId{n});
        };
        if (const auto* jump = std::get_if<Jump>(&p.blocks[n].terminator)) edge(jump->target);
        else if (const auto* branch = std::get_if<Branch>(&p.blocks[n].terminator)) {
            edge(branch->when_true); edge(branch->when_false);
        }
    }
    auto legal = take(legalize(SsaProgram{std::move(p)}, facts, profile));
    auto structured = take(structure(std::move(legal), {})); auto plan = take(plan_resources(structured,profile));
    return take(emit_msl(structured,plan,{}));
}
} // namespace armigo::shader::test::proven_cbuf
