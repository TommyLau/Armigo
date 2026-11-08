// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "matrix_plan.h"
#include <map>
#include <optional>

namespace armigo::shader::pipeline {
namespace {
// The values one operation writes: temporaries, and mutable locals that
// structuring declared for the whole function but that keep their single
// assignment. The program comes from SSA, so that operation's result is what
// every read of the value sees.
class Single {
public:
    Single(const Program& ir, const std::function<void()>& step) : definitions_(ir.values.size()) {
        for (const auto& op : ir.operations) {
            step();
            for (const auto& output : op.outputs)
                if (const auto* value = std::get_if<ValueId>(&output); value && value->value < definitions_.size())
                    ++definitions_[value->value];
        }
    }
    std::optional<std::uint32_t> operator()(const Operand& operand) const {
        const auto* value = std::get_if<ValueId>(&operand);
        if (!value || value->value >= definitions_.size() || definitions_[value->value] != 1) return {};
        return value->value;
    }
    template<std::size_t N>
    std::optional<std::array<std::uint32_t, N>> words(const Operation& op, std::size_t first) const {
        std::array<std::uint32_t, N> words{};
        for (std::size_t i = 0; i < N; ++i) {
            const auto value = (*this)(op.inputs[first + i]);
            if (!value) return {};
            words[i] = *value;
        }
        return words;
    }
private:
    std::vector<std::uint32_t> definitions_;
};

bool literal_zero(const Operand& operand) {
    const auto* literal = std::get_if<LiteralBits>(&operand);
    return literal && literal->bits == 0;
}
} // namespace

MatrixPlan plan_matrices(const StructuredProgram& p, const std::function<void()>& step) {
    const auto& ir = p.ir;
    std::vector<std::uint32_t> uses(ir.values.size());
    const auto use = [&](const Operand& operand) {
        if (const auto* value = std::get_if<ValueId>(&operand); value && value->value < uses.size()) ++uses[value->value];
    };
    for (const auto& op : ir.operations) {
        step();
        for (const auto& input : op.inputs) use(input);
    }
    for (const auto& region : p.regions) {
        step();
        if (const auto* branch = std::get_if<IfRegion>(&region)) use(branch->condition);
        else if (const auto* exit = std::get_if<ReturnRegion>(&region)) for (const auto& value : exit->values) use(value);
    }

    MatrixPlan plan;
    const Single single(ir, step);
    // The single-assignment words that matrix loads and multiplies define,
    // which their tiles and D matrices stand for anywhere in the function.
    using Source = MatrixPlan::Tile::Source;
    std::map<std::uint32_t, MatrixPlan::Tile> defined;
    for (std::uint32_t index = 0; index < ir.operations.size(); ++index) {
        step();
        const auto& op = ir.operations[index];
        const bool load = op.opcode == Opcode::matrix_load;
        if (!load && (op.opcode != Opcode::matrix_multiply_add || op.outputs.size() != 2)) continue;
        if (load) plan.loads[index] = {};
        for (std::size_t i = 0; i < op.outputs.size() && i < 4; ++i)
            if (const auto word = single(op.outputs[i]))
                defined[*word] = {load ? Source::load : Source::product, {index}, static_cast<std::uint8_t>(i)};
    }
    // Reads of each defined word as part of a matrix.
    std::vector<std::uint32_t> matrix_uses(ir.values.size());
    // An operand's tiles: the defined matrices when every word is one (a D
    // only as an A quadrant), else an earlier multiply's of the block with the
    // same words, else this multiply's own.
    const auto tiles = [&]<std::size_t N>(const Operation& op, std::size_t first, OperationId id,
                                          std::map<std::array<std::uint32_t, N>, std::array<MatrixPlan::Tile, N>>& earlier) {
        std::array<MatrixPlan::Tile, N> result;
        for (std::uint8_t i = 0; i < N; ++i) result[i] = {Source::own, id, i};
        const auto words = single.template words<N>(op, first);
        if (!words) return result;
        if (std::all_of(words->begin(), words->end(), [&](auto word) {
                const auto found = defined.find(word);
                return found != defined.end() && (N == 4 || found->second.source == Source::load);
            })) {
            for (std::size_t i = 0; i < N; ++i) result[i] = defined.at((*words)[i]);
            for (const auto word : *words) ++matrix_uses[word];
        } else if (const auto found = earlier.find(*words); found != earlier.end()) {
            result = found->second;
        }
        earlier.try_emplace(*words, result);
        return result;
    };
    for (const auto& block : ir.blocks) {
        std::map<std::array<std::uint32_t, 4>, std::array<MatrixPlan::Tile, 4>> a_matrices;
        std::map<std::array<std::uint32_t, 2>, std::array<MatrixPlan::Tile, 2>> b_matrices;
        for (const auto id : block.operations) {
            step();
            if (id.value >= ir.operations.size()) continue;
            const auto& op = ir.operations[id.value];
            if (op.opcode != Opcode::matrix_multiply_add || op.inputs.size() != 9 || op.outputs.size() != 2) continue;
            auto& multiply = plan.multiplies[id.value];
            multiply.a = tiles(op, 1, id, a_matrices);
            multiply.b = tiles(op, 5, id, b_matrices);
            if (literal_zero(op.inputs[7]) && literal_zero(op.inputs[8])) {
                multiply.c = MatrixPlan::Multiply::Accumulator::zero;
            } else if (const auto words = single.words<2>(op, 7)) {
                const auto low = defined.find((*words)[0]), high = defined.find((*words)[1]);
                if (low != defined.end() && high != defined.end() && low->second.source == Source::product &&
                    high->second.source == Source::product && low->second.op == high->second.op && low->second.index == 0 &&
                    high->second.index == 1) {
                    multiply.c = MatrixPlan::Multiply::Accumulator::chain;
                    multiply.c_from = low->second.op;
                    for (const auto word : *words) ++matrix_uses[word];
                }
            }
        }
    }
    for (const auto& [id, multiply] : plan.multiplies) {
        step();
        const auto note = [&](const MatrixPlan::Tile& tile, bool b) {
            if (tile.source == Source::load) {
                auto& load = plan.loads.at(tile.op.value);
                (b ? load.b_tiles : load.a_tiles) |= static_cast<std::uint8_t>(1u << tile.index);
            } else if (tile.source == Source::product) {
                plan.multiplies.at(tile.op.value).kept = true;
            }
        };
        for (const auto& tile : multiply.a) note(tile, false);
        for (const auto& tile : multiply.b) note(tile, true);
        if (multiply.c == MatrixPlan::Multiply::Accumulator::chain) plan.multiplies.at(multiply.c_from.value).kept = true;
    }
    // Words are read back into the lanes only when something reads them
    // other than as part of a matrix.
    const auto lanes = [&](const Operation& op) {
        return std::any_of(op.outputs.begin(), op.outputs.end(), [&](const Operand& output) {
            const auto word = single(output);
            return !word || uses[*word] != matrix_uses[*word];
        });
    };
    for (auto& [id, load] : plan.loads) {
        step();
        load.lanes = lanes(ir.operations[id]);
    }
    for (auto& [id, multiply] : plan.multiplies) {
        step();
        multiply.lanes = lanes(ir.operations[id]);
    }
    return plan;
}
} // namespace armigo::shader::pipeline
