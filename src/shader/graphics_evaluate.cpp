// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/graphics.h>
#include <armigo/shader/reciprocal.h>
#include <armigo/shader/rtz_multiply.h>
#include <armigo/shader/rne_multiply.h>
#include <armigo/shader/saturate.h>

#include "graphics_operations.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

#pragma STDC FP_CONTRACT OFF

namespace armigo::shader::graphics {
namespace {
namespace mw = maxwell;
struct Failure { Error error; };
[[noreturn]] void fail(Error::Code code, std::uint32_t pc = 0,
                      std::uint32_t detail = 0) { throw Failure{{code, pc, detail}}; }
void require(bool condition, Error::Code code, std::uint32_t pc = 0,
             std::uint32_t detail = 0) {
    if (!condition) fail(code, pc, detail);
}
bool generic(unsigned component) { return component >= 32 && component < 160; }
bool position(unsigned component) { return component >= 28 && component < 32; }
std::uint32_t flush(std::uint32_t bits) {
    return (bits & 0x7f800000u) == 0 ? bits & 0x80000000u : bits;
}
std::uint32_t canonical(std::uint32_t bits) {
    return (bits & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : bits;
}
float floating(std::uint32_t bits) { return std::bit_cast<float>(bits); }
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

struct FloatingScope {
    std::fenv_t saved{};
    bool active = false;
    FloatingScope() {
        require(std::feholdexcept(&saved) == 0, Error::Code::FloatingEnvironment);
        active = true;
    }
    ~FloatingScope() { if (active) std::fesetenv(&saved); }
    void restore() {
        require(std::fesetenv(&saved) == 0, Error::Code::FloatingEnvironment);
        active = false;
    }
    void prepare() {
        require(std::fesetround(FE_TONEAREST) == 0, Error::Code::FloatingEnvironment);
        volatile float small = floating(1u);
        volatile float one = 1.0f;
        volatile float minimum = std::numeric_limits<float>::min();
        volatile float half = 0.5f;
        require(bits(small * one) == 1u && bits(minimum * half) == 0x00400000u,
                Error::Code::FloatingEnvironment);
    }
};

// One value per node. Raw words, sample tuples, and predicates are distinct
// alternatives, and the storage is optional per node: a node whose block never
// executed has no value, so an invalid read fails instead of finding zero.
using Raw32 = std::uint32_t;
using Sample4 = std::array<std::uint32_t, 4>;
using Value = std::variant<Raw32, Sample4, bool>;

// Shared single-node execution for straight-line and graph programs. `from`
// carries the block control actually arrived from, so a phi reads only that
// predecessor's arm and never touches the other arms.
struct Execution {
    const Program& program;
    const Inputs& inputs;
    std::array<std::optional<std::uint32_t>, 160> attributes{};
    std::vector<std::optional<Value>> values;
    std::optional<std::uint32_t> from;

    Execution(const Program& owner, const Inputs& supplied)
        : program{owner}, inputs{supplied}, values(owner.nodes().size()) {
        for (const auto& input : inputs.attributes) {
            require(position(input.component) || generic(input.component),
                    Error::Code::UndeclaredAttribute, 0, input.component);
            require(!attributes[input.component].has_value(), Error::Code::DuplicateAttribute,
                    0, input.component);
            attributes[input.component] = input.bits;
        }
    }

    const Value& defined(std::uint32_t index, std::uint32_t pc) {
        require(index < values.size(), Error::Code::InvalidInstruction, pc, index);
        const auto& slot = values[index];
        if (!slot) fail(Error::Code::UninitializedRegister, pc, index);
        return *slot;
    }
    std::uint32_t word(ScalarId id, std::uint32_t pc) {
        const auto* value = std::get_if<Raw32>(&defined(id.index, pc));
        require(value != nullptr, Error::Code::InvalidInstruction, pc, id.index);
        return *value;
    }
    bool flag(PredicateId id, std::uint32_t pc) {
        const auto* value = std::get_if<bool>(&defined(id.index, pc));
        require(value != nullptr, Error::Code::InvalidInstruction, pc, id.index);
        return *value;
    }
    const Sample4& tuple(SampleId id, std::uint32_t pc) {
        const auto* value = std::get_if<Sample4>(&defined(id.index, pc));
        require(value != nullptr, Error::Code::InvalidInstruction, pc, id.index);
        return *value;
    }
    // A phi value is selected by the edge it was reached on; arms belonging to
    // other predecessors are never read.
    template <class Arms, class Read>
    auto joined(const Arms& arms, std::uint32_t pc, Read read) {
        require(from.has_value(), Error::Code::GraphDataFlow, pc);
        for (const auto& arm : arms) {
            if (arm.predecessor.index == *from) return read(arm.value);
        }
        fail(Error::Code::GraphDataFlow, pc, *from);
    }

    Value execute(const Node& node) {
        return std::visit([&](const auto& operation) -> Value {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, Literal>) {
                return Value{operation.bits};
            } else if constexpr (std::is_same_v<T, AttributeLoad>) {
                require(attributes[operation.component].has_value(), Error::Code::MissingAttribute,
                        node.pc, operation.component);
                return Value{*attributes[operation.component]};
            } else if constexpr (std::is_same_v<T, ConstantLoad>) {
                const auto& buffer = inputs.constant_buffers[operation.binding];
                require(operation.byte_offset / 4 < buffer.size(), Error::Code::ConstantBounds,
                        node.pc, operation.binding);
                return Value{buffer[operation.byte_offset / 4]};
            } else if constexpr (std::is_same_v<T, WordAnd>) {
                return Value{word(operation.operand, node.pc) & operation.mask};
            } else if constexpr (std::is_same_v<T, FloatCompare>) {
                auto lhs = flush(word(operation.lhs, node.pc));
                const auto rhs = flush(word(operation.rhs, node.pc));
                if (operation.negate_lhs) lhs ^= 0x80000000u;
                const bool result = operation.comparison == mw::OrderedComparison::LessThan
                    ? floating(lhs) < floating(rhs) : floating(lhs) > floating(rhs);
                return Value{result ? 0x3f800000u : 0u};
            } else if constexpr (std::is_same_v<T, FloatArithmetic>) {
                auto lhs = flush(word(operation.lhs, node.pc));
                const auto rhs = flush(word(operation.rhs, node.pc));
                if (operation.negate_lhs) lhs ^= 0x80000000u;
                if (operation.operation == Arithmetic::MultiplyZero) {
                    return Value{multiply_rtz_ftz(lhs, rhs)};
                }
                volatile float a = floating(lhs), b = floating(rhs);
                float result = 0;
                switch (operation.operation) {
                    case Arithmetic::Add: result = a + b; break;
                    case Arithmetic::MultiplyNearest: result = a * b; break;
                    case Arithmetic::FusedMultiplyAdd:
                        result = std::fma(a, b,
                            floating(flush(word(*operation.addend, node.pc)))); break;
                    case Arithmetic::MultiplyZero: break;
                }
                return Value{canonical(flush(bits(result)))};
            } else if constexpr (std::is_same_v<T, Reciprocal>) {
                return Value{reciprocal_rne(word(operation.operand, node.pc))};
            } else if constexpr (std::is_same_v<T, Saturate>) {
                // The source arithmetic has already stored its word; the
                // saturation step reads it once and applies the bounded
                // integer policy, independent of host rounding or FTZ.
                return Value{saturate_float32(word(operation.operand, node.pc))};
            } else if constexpr (std::is_same_v<T, Interpolate>) {
                auto result = word(operation.attribute, node.pc);
                if (operation.position_w) {
                    volatile float a = floating(result);
                    volatile float b = floating(word(*operation.position_w, node.pc));
                    result = canonical(bits(a * b));
                }
                if (operation.multiplier) {
                    result = multiply_rne(result, word(*operation.multiplier, node.pc));
                }
                return Value{result};
            } else if constexpr (std::is_same_v<T, TextureSample>) {
                require(static_cast<bool>(inputs.sample_2d_implicit_lod),
                        Error::Code::MissingSampler, node.pc);
                Sample4 result{};
                bool success = false;
                try {
                    success = inputs.sample_2d_implicit_lod({operation.handle_byte_offset,
                        word(operation.x, node.pc), word(operation.y, node.pc),
                        operation.no_dependency}, result);
                } catch (...) { fail(Error::Code::SamplingFailed, node.pc); }
                require(success, Error::Code::SamplingFailed, node.pc);
                require(std::fegetround() == FE_TONEAREST,
                        Error::Code::FloatingEnvironment, node.pc);
                return Value{result};
            } else if constexpr (std::is_same_v<T, SampleComponent>) {
                return Value{tuple(operation.sample, node.pc)[operation.component]};
            } else if constexpr (std::is_same_v<T, PredicateLiteral>) {
                return Value{operation.value};
            } else if constexpr (std::is_same_v<T, WordNotEqual>) {
                return Value{word(operation.lhs, node.pc) != word(operation.rhs, node.pc)};
            } else if constexpr (std::is_same_v<T, PredicateNot>) {
                return Value{!flag(operation.operand, node.pc)};
            } else if constexpr (std::is_same_v<T, ScalarPhi>) {
                return Value{joined(operation.arms, node.pc,
                                    [&](ScalarId id) { return word(id, node.pc); })};
            } else if constexpr (std::is_same_v<T, Pure>) {
                const auto* entry = detail::find_operation(operation.kind);
                require(entry != nullptr && entry->admits(operation.immediate),
                        Error::Code::InvalidInstruction, node.pc);
                detail::OperationWords words{};
                for (std::size_t slot = entry->arity; slot < operation.operands.size(); ++slot) {
                    require(operation.operands[slot] == 0, Error::Code::InvalidInstruction,
                            node.pc, operation.operands[slot]);
                }
                for (std::size_t slot = 0; slot < entry->arity; ++slot) {
                    const auto index = operation.operands[slot];
                    words[slot] = entry->operands[slot] == ValueType::Predicate
                        ? (flag(PredicateId{index}, node.pc) ? 1u : 0u)
                        : word(ScalarId{index}, node.pc);
                }
                const std::uint32_t result = entry->evaluate(words, operation.immediate);
                require(std::fegetround() == FE_TONEAREST,
                        Error::Code::FloatingEnvironment, node.pc);
                if (entry->result == ValueType::Predicate) return Value{result != 0};
                return Value{Raw32{result}};
            } else {
                return Value{joined(operation.arms, node.pc,
                                    [&](PredicateId id) { return flag(id, node.pc); })};
            }
        }, node.operation);
    }

    void store(std::uint32_t node) {
        values[node] = execute(program.nodes()[node]);
    }

    Evaluation exports() {
        Evaluation result;
        for (const auto& output : program.exports()) {
            result.outputs.push_back(
                {output.kind, output.component, word(output.value, 0)});
        }
        return result;
    }

    // Straight-line programs execute every node once, in list order.
    Evaluation sequential() {
        for (std::uint32_t node = 0; node < program.nodes().size(); ++node) store(node);
        return exports();
    }

    std::optional<std::uint32_t> successor(std::uint32_t candidate,
                                           std::uint32_t current) {
        require(candidate < program.blocks().size(), Error::Code::GraphFormat,
                0, candidate);
        // Targets are strictly forward, as validated, so the walk terminates
        // at the unique return through structure alone; no iteration budget
        // may stand in for that guarantee.
        require(candidate > current, Error::Code::GraphShape, 0, candidate);
        return candidate;
    }

    // Graph programs execute only the blocks the terminators choose. Loads,
    // samples, and callbacks in skipped arms never run.
    Evaluation graph() {
        const auto entry = program.entry();
        require(entry.has_value(), Error::Code::GraphFormat);
        std::uint32_t current = entry->index;
        from.reset();
        for (;;) {
            require(current < program.blocks().size(), Error::Code::GraphFormat,
                    0, current);
            const auto& block = program.blocks()[current];
            for (std::uint32_t node = block.begin; node < block.end; ++node) store(node);
            const auto pc = block.begin == block.end
                ? 0u : program.nodes()[block.end - 1].pc;
            const auto next = std::visit([&](const auto& terminator)
                    -> std::optional<std::uint32_t> {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<T, Jump>) {
                    return successor(terminator.target.index, current);
                } else if constexpr (std::is_same_v<T, BranchIf>) {
                    const auto taken_arm = successor(terminator.true_target.index, current);
                    const auto skipped_arm = successor(terminator.false_target.index, current);
                    return flag(terminator.condition, pc) ? taken_arm : skipped_arm;
                } else {
                    return std::nullopt;
                }
            }, block.terminator);
            if (!next) return exports();
            from = current;
            current = *next;
        }
    }
};

}  // namespace

static EvaluateResult evaluate_body(const Program& program, const Inputs& inputs) {
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
    try {
        Execution execution{program, inputs};
        return program.entry().has_value() ? execution.graph()
                                           : execution.sequential();
    } catch (const Failure& failure) { return failure.error; }
    catch (const std::bad_alloc&) { return Error{Error::Code::Allocation}; }
    catch (const std::length_error&) { return Error{Error::Code::Allocation}; }
}

EvaluateResult evaluate(const Program& program, const Inputs& inputs) {
    try {
        // Graph programs walk their validated blocks and execute only the
        // chosen ones; straight-line programs keep the whole-list walk. Both
        // share one node evaluator and the same floating controls.
        FloatingScope environment;
        environment.prepare();
        auto result = evaluate_body(program, inputs);
        environment.restore();
        return result;
    } catch (const Failure& failure) {
        return failure.error;
    }
}

}  // namespace armigo::shader::graphics
