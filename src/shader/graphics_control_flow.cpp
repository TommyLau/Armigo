// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "graphics_operations.h"
#include "graphics_validation.h"
#include <armigo/shader/graphics_control_flow.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace armigo::shader::graphics {
namespace {

using Stage = shader_input::Stage;

struct Failure { Error error; };
[[noreturn]] void fail(Error::Code code, std::uint32_t pc = 0,
                       std::uint32_t detail = 0) { throw Failure{{code, pc, detail}}; }
void require(bool condition, Error::Code code, std::uint32_t pc = 0,
             std::uint32_t detail = 0) {
    if (!condition) fail(code, pc, detail);
}

// Validator state for one graph specification. Blocks are always indexed
// before any data-flow question is asked, so every lookup below is in range.
struct Validation {
    const GraphSpec& spec;
    std::size_t block_count = 0;
    std::vector<std::uint32_t> node_block;
    std::vector<std::vector<std::uint32_t>> predecessors;
    std::vector<std::uint64_t> dominators;
    std::size_t words = 0;
    std::uint32_t return_block = 0;

    explicit Validation(const GraphSpec& description)
        : spec(description), block_count(description.blocks.size()),
          node_block(description.nodes.size(), 0) {}

    bool dominates(std::uint32_t upper, std::uint32_t lower) const {
        const std::size_t offset = static_cast<std::size_t>(lower) * words;
        return (dominators[offset + upper / 64] >> (upper % 64)) & 1;
    }
    // Up to two forward targets; a return has none.
    std::pair<std::array<std::uint32_t, 2>, std::size_t> targets(
        std::uint32_t block) const {
        std::array<std::uint32_t, 2> found{};
        std::size_t count = 0;
        std::visit([&](const auto& terminator) {
            using T = std::decay_t<decltype(terminator)>;
            if constexpr (std::is_same_v<T, Jump>) {
                found[count++] = terminator.target.index;
            } else if constexpr (std::is_same_v<T, BranchIf>) {
                found[count++] = terminator.true_target.index;
                found[count++] = terminator.false_target.index;
            }
        }, spec.blocks[block].terminator);
        return {found, count};
    }
};

void check_structure(Validation& state) {
    const auto& spec = state.spec;
    require(!spec.blocks.empty(), Error::Code::GraphFormat);
    require(spec.blocks.size() <= max_blocks, Error::Code::Limit,
            0, static_cast<std::uint32_t>(spec.blocks.size()));
    require(spec.nodes.size() <= std::numeric_limits<std::uint32_t>::max(),
            Error::Code::Limit);
    // Block ranges must tile the node list exactly once.
    std::uint32_t cursor = 0;
    for (std::size_t index = 0; index < state.block_count; ++index) {
        const auto& block = spec.blocks[index];
        require(block.begin <= block.end, Error::Code::GraphFormat,
                0, static_cast<std::uint32_t>(index));
        require(block.begin == cursor, Error::Code::GraphFormat,
                0, static_cast<std::uint32_t>(index));
        cursor = block.end;
    }
    require(cursor == spec.nodes.size(), Error::Code::GraphFormat, 0, cursor);
    for (std::size_t index = 0; index < state.block_count; ++index) {
        for (std::uint32_t node = spec.blocks[index].begin;
             node < spec.blocks[index].end; ++node) {
            state.node_block[node] = static_cast<std::uint32_t>(index);
        }
    }
    // Terminators: targets inside the graph and strictly forward; exactly one
    // return, and it must be the last block so every forward walk terminates.
    std::uint32_t returns = 0;
    for (std::size_t index = 0; index < state.block_count; ++index) {
        const auto self = static_cast<std::uint32_t>(index);
        std::visit([&](const auto& terminator) {
            using T = std::decay_t<decltype(terminator)>;
            if constexpr (std::is_same_v<T, Jump>) {
                require(terminator.target.index < state.block_count,
                        Error::Code::GraphFormat, 0, terminator.target.index);
                require(terminator.target.index > self, Error::Code::GraphShape,
                        0, terminator.target.index);
            } else if constexpr (std::is_same_v<T, BranchIf>) {
                require(terminator.true_target.index < state.block_count,
                        Error::Code::GraphFormat, 0, terminator.true_target.index);
                require(terminator.true_target.index > self, Error::Code::GraphShape,
                        0, terminator.true_target.index);
                require(terminator.false_target.index < state.block_count,
                        Error::Code::GraphFormat, 0, terminator.false_target.index);
                require(terminator.false_target.index > self, Error::Code::GraphShape,
                        0, terminator.false_target.index);
            } else {
                ++returns;
                state.return_block = self;
            }
        }, spec.blocks[index].terminator);
    }
    require(returns == 1, Error::Code::GraphShape, 0, returns);
    require(state.return_block == state.block_count - 1, Error::Code::GraphShape,
            0, state.return_block);
    // Entry is block 0 and every block must be reachable from it.
    std::vector<char> reached(state.block_count, 0);
    std::vector<std::uint32_t> pending{0};
    reached[0] = 1;
    while (!pending.empty()) {
        const auto block = pending.back();
        pending.pop_back();
        const auto [succ, count] = state.targets(block);
        for (std::size_t index = 0; index < count; ++index) {
            if (!reached[succ[index]]) {
                reached[succ[index]] = 1;
                pending.push_back(succ[index]);
            }
        }
    }
    for (std::size_t index = 0; index < state.block_count; ++index) {
        require(reached[index] != 0, Error::Code::GraphShape,
                0, static_cast<std::uint32_t>(index));
    }
    // Predecessor lists, deduplicated and in increasing block order.
    state.predecessors.assign(state.block_count, {});
    for (std::size_t index = 0; index < state.block_count; ++index) {
        const auto [succ, count] = state.targets(static_cast<std::uint32_t>(index));
        for (std::size_t entry = 0; entry < count; ++entry) {
            if (state.predecessors[succ[entry]].empty() ||
                state.predecessors[succ[entry]].back() != index) {
                state.predecessors[succ[entry]].push_back(
                    static_cast<std::uint32_t>(index));
            }
        }
    }
}

// Iterative dominator fixpoint over reverse postorder. All blocks are
// reachable here, so every predecessor row is initialized.
void compute_dominators(Validation& state) {
    const std::size_t count = state.block_count;
    state.words = (count + 63) / 64;
    std::vector<std::uint32_t> postorder;
    std::vector<char> done(count, 0);
    std::vector<std::pair<std::uint32_t, std::size_t>> frames;
    frames.push_back({0, 0});
    done[0] = 1;
    while (!frames.empty()) {
        const std::uint32_t block = frames.back().first;
        auto& next = frames.back().second;
        const auto [succ, outgoing] = state.targets(block);
        bool descended = false;
        for (; next < outgoing; ++next) {
            if (!done[succ[next]]) {
                done[succ[next]] = 1;
                frames.push_back({succ[next], 0});
                descended = true;
                break;
            }
        }
        if (!descended) {
            postorder.push_back(block);
            frames.pop_back();
        }
    }
    std::reverse(postorder.begin(), postorder.end());
    const auto all = ~std::uint64_t{0};
    state.dominators.assign(count * state.words, all);
    auto set = [&](std::uint32_t block, std::size_t word, std::uint64_t bits) {
        state.dominators[block * state.words + word] = bits;
    };
    for (std::size_t word = 0; word < state.words; ++word) {
        set(0, word, 0);
    }
    set(0, 0, 1);
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto block : postorder) {
            if (block == 0) continue;
            std::vector<std::uint64_t> merged(state.words, all);
            for (const auto before : state.predecessors[block]) {
                const std::size_t offset = before * state.words;
                for (std::size_t word = 0; word < state.words; ++word) {
                    merged[word] &= state.dominators[offset + word];
                }
            }
            merged[block / 64] |= std::uint64_t{1} << (block % 64);
            const std::size_t offset = block * state.words;
            for (std::size_t word = 0; word < state.words && !changed; ++word) {
                changed = merged[word] != state.dominators[offset + word];
            }
            for (std::size_t word = 0; word < state.words; ++word) {
                set(block, word, merged[word]);
            }
        }
    }
}

}  // namespace

GraphResult make_graph(const GraphSpec& spec, Limits limits) {
    try {
        require(spec.stage == Stage::Vertex || spec.stage == Stage::Fragment,
                Error::Code::UnsupportedStage);
        require(spec.nodes.size() <= limits.max_nodes, Error::Code::Limit);
        Validation state{spec};
        check_structure(state);
        compute_dominators(state);

        // Typed operand checks shared by plain operations and phi arms.
        const auto& nodes = spec.nodes;
        const auto check_word = [&](std::uint32_t id, std::uint32_t pc) {
            require(id < nodes.size(), Error::Code::GraphFormat, pc, id);
            require(nodes[id].type() == ValueType::Raw32,
                    Error::Code::GraphTypes, pc, id);
        };
        const auto check_predicate = [&](std::uint32_t id, std::uint32_t pc) {
            require(id < nodes.size(), Error::Code::GraphFormat, pc, id);
            require(nodes[id].type() == ValueType::Predicate,
                    Error::Code::GraphTypes, pc, id);
        };
        const auto check_sample = [&](std::uint32_t id, std::uint32_t pc) {
            require(id < nodes.size(), Error::Code::GraphFormat, pc, id);
            require(nodes[id].type() == ValueType::Sample4,
                    Error::Code::GraphTypes, pc, id);
        };
        // A non-phi definition is readable from its own block when it comes
        // earlier, or from any block its definition block dominates.
        const auto readable = [&](std::uint32_t definition, std::uint32_t use,
                                  std::uint32_t pc) {
            const auto definition_block = state.node_block[definition];
            const auto use_block = state.node_block[use];
            const bool available = definition_block == use_block
                ? definition < use
                : state.dominates(definition_block, use_block);
            require(available, Error::Code::GraphDataFlow, pc, definition);
        };
        // At a block boundary (terminator, phi arm, exit) only dominating
        // definitions of the predecessor's own block are readable.
        const auto readable_at_end = [&](std::uint32_t definition,
                                         std::uint32_t block, std::uint32_t pc) {
            const auto definition_block = state.node_block[definition];
            require(definition_block == block ||
                        state.dominates(definition_block, block),
                    Error::Code::GraphDataFlow, pc, definition);
        };

        for (std::size_t index = 0; index < state.block_count; ++index) {
            const auto block = static_cast<std::uint32_t>(index);
            for (std::uint32_t node = spec.blocks[index].begin;
                 node < spec.blocks[index].end; ++node) {
                const auto pc = nodes[node].pc;
                std::visit([&](const auto& operation) {
                    using T = std::decay_t<decltype(operation)>;
                    if constexpr (std::is_same_v<T, Literal> ||
                                  std::is_same_v<T, AttributeLoad> ||
                                  std::is_same_v<T, ConstantLoad> ||
                                  std::is_same_v<T, PredicateLiteral>) {
                        return;
                    } else if constexpr (std::is_same_v<T, FloatCompare>) {
                        check_word(operation.lhs.index, pc);
                        check_word(operation.rhs.index, pc);
                        readable(operation.lhs.index, node, pc);
                        readable(operation.rhs.index, node, pc);
                    } else if constexpr (std::is_same_v<T, FloatArithmetic>) {
                        check_word(operation.lhs.index, pc);
                        check_word(operation.rhs.index, pc);
                        readable(operation.lhs.index, node, pc);
                        readable(operation.rhs.index, node, pc);
                        if (operation.addend) {
                            check_word(operation.addend->index, pc);
                            readable(operation.addend->index, node, pc);
                        }
                    } else if constexpr (std::is_same_v<T, Reciprocal>) {
                        check_word(operation.operand.index, pc);
                        readable(operation.operand.index, node, pc);
                    } else if constexpr (std::is_same_v<T, Saturate>) {
                        check_word(operation.operand.index, pc);
                        readable(operation.operand.index, node, pc);
                    } else if constexpr (std::is_same_v<T, Interpolate>) {
                        check_word(operation.attribute.index, pc);
                        readable(operation.attribute.index, node, pc);
                        if (operation.position_w) {
                            check_word(operation.position_w->index, pc);
                            readable(operation.position_w->index, node, pc);
                        }
                        if (operation.multiplier) {
                            check_word(operation.multiplier->index, pc);
                            readable(operation.multiplier->index, node, pc);
                        }
                    } else if constexpr (std::is_same_v<T, TextureSample>) {
                        check_word(operation.x.index, pc);
                        check_word(operation.y.index, pc);
                        readable(operation.x.index, node, pc);
                        readable(operation.y.index, node, pc);
                    } else if constexpr (std::is_same_v<T, SampleComponent>) {
                        check_sample(operation.sample.index, pc);
                        readable(operation.sample.index, node, pc);
                    } else if constexpr (std::is_same_v<T, WordAnd>) {
                        check_word(operation.operand.index, pc);
                        readable(operation.operand.index, node, pc);
                    } else if constexpr (std::is_same_v<T, WordNotEqual>) {
                        check_word(operation.lhs.index, pc);
                        check_word(operation.rhs.index, pc);
                        readable(operation.lhs.index, node, pc);
                        readable(operation.rhs.index, node, pc);
                    } else if constexpr (std::is_same_v<T, PredicateNot>) {
                        check_predicate(operation.operand.index, pc);
                        readable(operation.operand.index, node, pc);
                    } else if constexpr (std::is_same_v<T, Pure>) {
                        // The registry entry is the operation's whole contract:
                        // an unregistered kind, a rejected immediate, or a
                        // nonzero unused operand slot is malformed.
                        const auto* entry = detail::find_operation(operation.kind);
                        require(entry != nullptr && entry->admits(operation.immediate),
                                Error::Code::GraphFormat, pc, node);
                        for (std::size_t slot = 0; slot < operation.operands.size();
                             ++slot) {
                            const auto operand = operation.operands[slot];
                            if (slot >= entry->arity) {
                                require(operand == 0, Error::Code::GraphFormat, pc,
                                        operand);
                                continue;
                            }
                            if (entry->operands[slot] == ValueType::Predicate) {
                                check_predicate(operand, pc);
                            } else {
                                check_word(operand, pc);
                            }
                            readable(operand, node, pc);
                        }
                    } else {
                        // Phi inputs are labelled with actual predecessors and
                        // each value must hold on the full labelled path.
                        require(block != 0, Error::Code::GraphDataFlow, pc, block);
                        const auto arms = operation.arms;
                        using Arm = typename std::decay_t<decltype(arms)>::value_type;
                        const auto& expected = state.predecessors[block];
                        require(arms.size() == expected.size(),
                                Error::Code::GraphDataFlow, pc,
                                static_cast<std::uint32_t>(arms.size()));
                        std::vector<std::uint32_t> labels;
                        labels.reserve(arms.size());
                        for (const auto& arm : arms) {
                            require(arm.predecessor.index < state.block_count,
                                    Error::Code::GraphFormat, pc,
                                    arm.predecessor.index);
                            labels.push_back(arm.predecessor.index);
                            if constexpr (std::is_same_v<Arm, ScalarPhiArm>) {
                                check_word(arm.value.index, pc);
                            } else {
                                check_predicate(arm.value.index, pc);
                            }
                            readable_at_end(arm.value.index,
                                            arm.predecessor.index, pc);
                        }
                        std::sort(labels.begin(), labels.end());
                        require(std::adjacent_find(labels.begin(), labels.end()) ==
                                    labels.end(),
                                Error::Code::GraphDataFlow, pc, block);
                        require(labels ==
                                    std::vector<std::uint32_t>(expected.begin(),
                                                               expected.end()),
                                Error::Code::GraphDataFlow, pc, block);
                    }
                }, nodes[node].operation);
            }
            // The terminator condition is evaluated at the block end.
            std::visit([&](const auto& terminator) {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<T, BranchIf>) {
                    const auto& source = spec.blocks[index];
                    const auto pc = source.begin == source.end ? 0u : nodes[source.end - 1].pc;
                    check_predicate(terminator.condition.index, pc);
                    readable_at_end(terminator.condition.index, block, pc);
                }
            }, spec.blocks[index].terminator);
        }

        for (const auto& output : spec.exports) {
            check_word(output.value.index, 0);
            readable_at_end(output.value.index, state.return_block, 0);
        }

        if (const auto error = detail::validate_graph_interface(spec)) return *error;
        Program program;
        program.stage_ = spec.stage;
        program.sph_ = spec.sph;
        program.nodes_ = spec.nodes;
        program.exports_ = spec.exports;
        program.blocks_ = spec.blocks;
        return program;
    } catch (const Failure& failure) { return failure.error; }
    catch (const std::bad_alloc&) { return Error{Error::Code::Allocation}; }
    catch (const std::length_error&) { return Error{Error::Code::Allocation}; }
}

}  // namespace armigo::shader::graphics
