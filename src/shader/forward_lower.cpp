// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Guarded forward-program lowering into the typed SSA graph. The unit owns
// block-ordered state merging over the decoded ForwardProgram view and
// the integer-set-predicate graph operations; every numeric instruction is
// lowered by the existing Builder methods, never by a duplicated numeric
// lowerer. See graphics_forward.h for the admission and error contract.
#include <armigo/shader/graphics_forward.h>

#include "forward_metadata.h"
#include "forward_state.h"
#include "graphics_builder.h"
#include "graphics_families.h"

#include <armigo/shader/families/fma_constant_negated.h>
#include <armigo/shader/families/fma_register_constant.h>
#include <armigo/shader/families/mov_constant.h>
#include <armigo/shader/families/predicate_logic.h>
#include <armigo/shader/families/word_and_immediate.h>
#include <armigo/shader/graphics_control_flow.h>
#include <armigo/shader/maxwell_flow.h>
#include <armigo/shader/maxwell_predicates.h>

#include <algorithm>
#include <array>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::graphics {
namespace {

namespace mw = maxwell;
using Failure = BuilderFailure;

constexpr std::uint32_t kEntryPc = 8;
constexpr std::uint32_t kPendingBlock = 0x7ffffffu;

[[noreturn]] void fail(Error::Code code, std::uint32_t pc,
                       std::uint32_t detail = 0) {
    throw Failure{{code, pc, detail}};
}

bool is_pt(mw::Predicate predicate) {
    return predicate.index == 7 && !predicate.inverted;
}

bool is_inverted_pt(mw::Predicate predicate) {
    return predicate.index == 7 && predicate.inverted;
}

// Next instruction-slot pc after a fallthrough, skipping the scheduling word
// that starts each 32-byte group; false when the path leaves the region.
bool fallthrough_pc(std::uint32_t pc, std::uint32_t program_size,
                    std::uint32_t& next) {
    std::uint64_t wide = static_cast<std::uint64_t>(pc) +
                         shader_input::format::instruction_word_size;
    if (wide % shader_input::format::scheduling_group_size == 0) {
        wide += shader_input::format::instruction_word_size;
    }
    if (wide + shader_input::format::instruction_word_size > program_size) {
        return false;
    }
    next = static_cast<std::uint32_t>(wide);
    return true;
}

// Input slot pc of one node view: branches are keyed by their source pc.
std::uint32_t node_pc(const mw::ForwardInstruction& node) {
    if (const auto* branch = std::get_if<mw::ForwardBranch>(&node)) {
        return branch->source_pc;
    }
    return std::get<mw::Instruction>(node).pc;
}

bool builder_implemented(mw::OpcodeFamily family) {
    using F = mw::OpcodeFamily;
    switch (family) {
        case F::Mov32Immediate:
        case F::AttributeLoad:
        case F::AttributeStore:
        case F::FloatSetRegister:
        case F::FloatAddRegister:
        case F::FloatAddConstant:
        case F::FloatAddImmediate:
        case F::FloatMultiplyRegister:
        case F::FloatMultiplyConstant:
        case F::FloatMultiplyImmediate:
        case F::FloatFmaImmediate:
        case F::FloatFmaConstantRegister:
        case F::MultiFunction:
        case F::InterpolateAttribute:
        case F::TextureSample:
        case F::Exit: return true;
        default: return false;
    }
}

// Admission reflects what the Builder and the registered family lowerers
// actually implement, plus the graph-only integer-set-predicate profile;
// it never accepts a family merely because the decoder has an enum for it.
bool family_admitted(mw::OpcodeFamily family) {
    return family == mw::OpcodeFamily::IntegerSetPredicateRegister ||
           builder_implemented(family) ||
           detail::find_registered_lowerer(family) != nullptr;
}

using detail::ForwardCell;
using detail::ForwardState;
using detail::InitExpr;
using detail::init_false;
using detail::init_true;

struct Guard {
    enum class Value { true_, false_, unknown };
    Value value = Value::unknown;
    PredicateId id{0};
    bool inverted = false;
};

// Resolves one guard against the block's known facts. A PredicateNot alias
// destination (ISETP's B write) canonicalizes to its operand and orientation.
// The comparison and its complement consequently share an immutable identity
// for both branch facts and initialization proofs across joins.
Guard resolve_guard(const Builder& builder, const ForwardState& state,
                    mw::Predicate predicate, std::uint32_t pc) {
    if (predicate.index == 7) {
        Guard resolved;
        resolved.value = predicate.inverted ? Guard::Value::false_
                                            : Guard::Value::true_;
        return resolved;
    }
    // Only proved-initialized cells are exposed in the Builder view.
    const auto& definition = builder.predicates[predicate.index];
    if (!definition) {
        fail(Error::Code::UninitializedRegister, pc, predicate.index);
    }
    PredicateId id = *definition;
    bool inverted = predicate.inverted;
    const std::span<const Node> nodes = builder.program.nodes();
    while (const auto* alias =
               std::get_if<PredicateNot>(&nodes[id.index].operation)) {
        id = alias->operand;
        inverted = !inverted;
    }
    if (const auto fact = state.facts.find(id.index);
        fact != state.facts.end()) {
        Guard resolved;
        resolved.value =
            fact->second != inverted ? Guard::Value::true_
                                     : Guard::Value::false_;
        return resolved;
    }
    Guard resolved;
    resolved.value = Guard::Value::unknown;
    resolved.id = id;
    resolved.inverted = inverted;
    return resolved;
}

struct ExitSnapshot {
    std::uint32_t block = kPendingBlock;
    std::vector<Export> exports;
};

class ForwardLowerer {
public:
    ForwardLowerer(const mw::ForwardProgram& program, Builder& builder)
        : program_(program), builder_(builder) {
        leaders_.insert(kEntryPc);
        for (std::size_t index = 0; index < program_.instructions.size(); ++index) {
            const auto& node = program_.instructions[index];
            const auto pc = node_pc(node);
            index_of_.emplace(pc, index);
            bool boundary = false;
            if (const auto* branch = std::get_if<mw::ForwardBranch>(&node)) {
                leaders_.insert(branch->target_pc);
                boundary = true;
            } else {
                const auto& instruction = std::get<mw::Instruction>(node);
                boundary = instruction.predicate.index != 7 ||
                           instruction.family == mw::OpcodeFamily::Exit;
                if (instruction.predicate.index != 7) leaders_.insert(pc);
            }
            std::uint32_t next = 0;
            if (boundary && fallthrough_pc(pc, program_.program_size, next)) {
                leaders_.insert(next);
            }
        }
        literals_.emplace(builder_.zero.index, 0);
    }

    GraphResult run() {
        // Concrete invalid carriers dominate every block but are never
        // exposed as guest definitions without a successful validity proof.
        const PredicateId false_carrier{builder_.append(PredicateLiteral{false})};
        for (auto& cell : state_.predicates) cell.carrier = false_carrier;
        incoming_[kEntryPc].push_back(Incoming{kPendingBlock, false, state_});
        while (!incoming_.empty()) {
            auto item = incoming_.extract(incoming_.begin());
            const auto pc = item.key();
            builder_.pc = expressions_.pc = pc;
            const auto block = open_block();
            for (const auto& edge : item.mapped()) patch_parent(edge, block);
            state_ = merge(item.mapped());
            process_block(index_of(pc), block);
        }
        return finish();
    }

private:
    struct Incoming {
        std::uint32_t predecessor = kPendingBlock;
        bool true_slot = false;
        ForwardState state;
    };

    std::size_t index_of(std::uint32_t pc) const {
        const auto found = index_of_.find(pc);
        if (found == index_of_.end()) fail(Error::Code::InvalidInstruction, pc, pc);
        return found->second;
    }

    std::uint32_t next_pc(std::uint32_t pc) const {
        std::uint32_t next = 0;
        if (!fallthrough_pc(pc, program_.program_size, next)) {
            fail(Error::Code::MissingExit, pc);
        }
        return next;
    }

    std::uint32_t node_count() const {
        return static_cast<std::uint32_t>(builder_.program.nodes().size());
    }

    std::uint32_t open_block() {
        if (blocks_.size() >= max_blocks) {
            fail(Error::Code::Limit, builder_.pc, static_cast<std::uint32_t>(max_blocks));
        }
        BasicBlock block;
        block.begin = blocks_.empty() ? 0 : node_count();
        blocks_.push_back(block);
        return static_cast<std::uint32_t>(blocks_.size() - 1);
    }

    void patch_parent(const Incoming& edge, std::uint32_t block) {
        if (edge.predecessor == kPendingBlock) return;
        auto& terminator = blocks_[edge.predecessor].terminator;
        if (auto* jump = std::get_if<Jump>(&terminator)) {
            jump->target = BlockId{block};
        } else {
            auto& branch = std::get<BranchIf>(terminator);
            (edge.true_slot ? branch.true_target : branch.false_target) = BlockId{block};
        }
    }

    // All predecessors are already lowered: admitted guest edges are
    // forward, and the only synthetic block is an immediate guarded body.
    template<class Id, std::size_t N>
    void merge_cells(std::array<ForwardCell<Id>, N>& result,
                     const std::vector<Incoming>& incoming,
                     std::array<ForwardCell<Id>, N> ForwardState::* member) {
        for (std::size_t slot = 0; slot < N; ++slot) {
            const auto& first = (incoming.front().state.*member)[slot];
            bool same_carrier = true, same_validity = true;
            InitExpr validity = init_false;
            for (const auto& edge : incoming) {
                const auto& cell = (edge.state.*member)[slot];
                same_carrier &= cell.carrier.index == first.carrier.index;
                same_validity &= cell.validity == first.validity;
                // The edge scope is the actual predecessor selection. In
                // particular, true on one edge cannot initialize another.
                validity = expressions_.disjunction(validity,
                    expressions_.conjunction(edge.state.scope, cell.validity));
            }
            result[slot].validity = same_validity ? first.validity : validity;
            if (same_carrier) {
                result[slot].carrier = first.carrier;
                continue;
            }
            using Phi = std::conditional_t<std::is_same_v<Id, ScalarId>,
                                           ScalarPhi, PredicatePhi>;
            Phi phi;
            phi.arms.reserve(incoming.size());
            for (const auto& edge : incoming) {
                phi.arms.push_back({BlockId{edge.predecessor},
                                   (edge.state.*member)[slot].carrier});
            }
            result[slot].carrier = Id{builder_.append(std::move(phi))};
        }
    }

    ForwardState merge(const std::vector<Incoming>& incoming) {
        if (incoming.size() == 1) return incoming.front().state;
        ForwardState result;
        result.scope = init_false;
        result.facts = incoming.front().state.facts;
        for (const auto& edge : incoming) {
            result.scope = expressions_.disjunction(result.scope, edge.state.scope);
            for (auto fact = result.facts.begin(); fact != result.facts.end();) {
                const auto other = edge.state.facts.find(fact->first);
                if (other == edge.state.facts.end() || other->second != fact->second) {
                    fact = result.facts.erase(fact);
                } else {
                    ++fact;
                }
            }
        }
        merge_cells(result.registers, incoming, &ForwardState::registers);
        merge_cells(result.predicates, incoming, &ForwardState::predicates);
        merge_cells(result.stored, incoming, &ForwardState::stored);
        return result;
    }

    // Builder's existing checked read/export primitives see only cells
    // proved valid under this block's scope and immutable predicate facts.
    // Invalid carriers remain in state_ for phi construction, never as an
    // optional definition that a numeric family could accidentally read.
    void prepare_reads() {
        std::map<InitExpr, bool> proofs;
        const auto expose = [&](const auto& cells, auto& definitions) {
            for (std::size_t slot = 0; slot < cells.size(); ++slot) {
                const auto& cell = cells[slot];
                auto [proof, inserted] = proofs.emplace(cell.validity, false);
                if (inserted) {
                    proof->second = expressions_.proves(state_.scope, cell.validity,
                                                         state_.facts);
                }
                if (proof->second) definitions[slot] = cell.carrier;
                else definitions[slot].reset();
            }
        };
        expose(state_.registers, builder_.registers);
        expose(state_.predicates, builder_.predicates);
        expose(state_.stored, builder_.stored);
    }

    void publish_writes() {
        const auto publish = [](auto& cells, const auto& definitions) {
            for (std::size_t slot = 0; slot < cells.size(); ++slot) {
                if (definitions[slot]) cells[slot] = {*definitions[slot], init_true};
            }
        };
        publish(state_.registers, builder_.registers);
        publish(state_.predicates, builder_.predicates);
        publish(state_.stored, builder_.stored);
    }

    Guard guard_for(mw::Predicate predicate) {
        prepare_reads();
        auto guard = resolve_guard(builder_, state_, predicate, builder_.pc);
        if (guard.value == Guard::Value::unknown) {
            const auto positive = expressions_.literal(guard.id, !guard.inverted);
            const auto negative = expressions_.literal(guard.id, guard.inverted);
            if (expressions_.proves(state_.scope, positive, state_.facts)) {
                guard.value = Guard::Value::true_;
            } else if (expressions_.proves(state_.scope, negative, state_.facts)) {
                guard.value = Guard::Value::false_;
            }
        }
        return guard;
    }

    ForwardState on_edge(const Guard& guard, bool taken) {
        ForwardState result = state_;
        const bool fact = taken != guard.inverted;
        result.facts[guard.id.index] = fact;
        result.scope = expressions_.conjunction(result.scope,
                                                expressions_.literal(guard.id, fact));
        return result;
    }

    void enqueue(std::uint32_t pc, std::uint32_t block, bool true_slot,
                 ForwardState state) {
        incoming_[pc].push_back(Incoming{block, true_slot, std::move(state)});
    }

    void close_with(std::uint32_t block, Terminator terminator) {
        blocks_[block].end = node_count();
        blocks_[block].terminator = std::move(terminator);
    }

    void jump_to(std::uint32_t block, std::uint32_t pc) {
        close_with(block, Jump{BlockId{kPendingBlock}});
        enqueue(pc, block, true, std::move(state_));
    }

    void branch_on(std::uint32_t block, const Guard& guard) {
        PredicateId condition = guard.id;
        if (guard.inverted) condition = PredicateId{builder_.append(PredicateNot{guard.id})};
        close_with(block, BranchIf{condition, BlockId{kPendingBlock}, BlockId{kPendingBlock}});
    }

    void terminate_exit(std::uint32_t block) {
        prepare_reads();
        std::vector<Export> snapshot = builder_.collect_exports();
        close_with(block, Jump{BlockId{kPendingBlock}});
        exits_.push_back(ExitSnapshot{block, std::move(snapshot)});
    }

    void process_block(std::size_t index, std::uint32_t block) {
        const auto start = index;
        while (true) {
            const auto& node = program_.instructions[index];
            const auto pc = node_pc(node);
            builder_.pc = expressions_.pc = pc;
            if (index != start && leaders_.contains(pc)) {
                jump_to(block, pc);
                return;
            }
            if (const auto* branch = std::get_if<mw::ForwardBranch>(&node)) {
                const auto guard = guard_for(branch->predicate);
                if (guard.value == Guard::Value::true_) {
                    jump_to(block, branch->target_pc);
                } else if (guard.value == Guard::Value::false_) {
                    jump_to(block, next_pc(pc));
                } else if (branch->target_pc == next_pc(pc)) {
                    // A branch with identical successors has one predecessor
                    // edge and establishes no predicate fact at the join.
                    jump_to(block, branch->target_pc);
                } else {
                    branch_on(block, guard);
                    enqueue(branch->target_pc, block, true, on_edge(guard, true));
                    enqueue(next_pc(pc), block, false, on_edge(guard, false));
                }
                return;
            }
            const auto& instruction = std::get<mw::Instruction>(node);
            const auto guard = guard_for(instruction.predicate);
            if (guard.value == Guard::Value::unknown) {
                branch_on(block, guard);
                enqueue(next_pc(pc), block, false, on_edge(guard, false));
                state_ = on_edge(guard, true);
                const auto body = open_block();
                patch_parent(Incoming{block, true, {}}, body);
                if (instruction.family == mw::OpcodeFamily::Exit) {
                    terminate_exit(body);
                } else {
                    prepare_reads();
                    execute_instruction(instruction);
                    publish_writes();
                    jump_to(body, next_pc(pc));
                }
                return;
            }
            if (guard.value == Guard::Value::true_) {
                if (instruction.family == mw::OpcodeFamily::Exit) {
                    terminate_exit(block);
                    return;
                }
                execute_instruction(instruction);
                publish_writes();
            }
            index = index_of(next_pc(pc));
        }
    }

    void execute_instruction(const mw::Instruction& instruction) {
        if (instruction.family ==
            mw::OpcodeFamily::IntegerSetPredicateRegister) {
            execute_isetp(
                std::get<mw::IntegerSetPredicate>(instruction.payload));
            return;
        }
        mw::Instruction unguarded = instruction;
        // The guard decision has been made; the existing Builder method is
        // strict PT, so the decided-true guard is passed through verbatim.
        unguarded.predicate = mw::Predicate{7, false};
        builder_.instruction(unguarded);
        if (instruction.family == mw::OpcodeFamily::Mov32Immediate) {
            const auto& mov =
                std::get<mw::Mov32Immediate>(instruction.payload);
            if (!mov.destination.is_zero()) {
                if (const auto& written =
                        builder_.registers[mov.destination.index]) {
                    literals_[written->index] = mov.immediate_bits;
                }
            }
        }
    }

    std::optional<std::uint32_t> literal(ScalarId value) const {
        const auto found = literals_.find(value.index);
        if (found == literals_.end()) return std::nullopt;
        return found->second;
    }

    // ISETP NE/AND/PT-source profile: A = (lhs != rhs) as a raw word
    // comparison (the signed field never affects not-equal), B = NOT(A);
    // sources are captured before either destination is written,
    // destinations are written in A then B order, and PT writes discard.
    void execute_isetp(const mw::IntegerSetPredicate& value) {
        const ScalarId lhs = builder_.read(value.lhs);
        const ScalarId rhs = builder_.read(value.rhs);
        const PredicateId comparison{builder_.append(WordNotEqual{lhs, rhs})};
        if (const auto lhs_bits = literal(lhs)) {
            if (const auto rhs_bits = literal(rhs)) {
                state_.facts[comparison.index] = *lhs_bits != *rhs_bits;
            }
        }
        if (value.destination_a != 7) {
            builder_.predicates[value.destination_a] = comparison;
        }
        if (value.destination_b != 7) {
            const PredicateId inverted{
                builder_.append(PredicateNot{comparison})};
            if (const auto fact = state_.facts.find(comparison.index);
                fact != state_.facts.end()) {
                state_.facts[inverted.index] = !fact->second;
            }
            builder_.predicates[value.destination_b] = inverted;
        }
    }

    GraphResult finish() {
        if (exits_.empty()) {
            fail(Error::Code::MissingExit, kEntryPc);
        }
        if (blocks_.size() >= max_blocks) {
            fail(Error::Code::Limit, builder_.pc, static_cast<std::uint32_t>(max_blocks));
        }
        const std::uint32_t return_block =
            static_cast<std::uint32_t>(blocks_.size());
        BasicBlock block;
        block.begin = node_count();
        for (auto& exit : exits_) {
            blocks_[exit.block].terminator = Jump{BlockId{return_block}};
        }
        // Every reachable exit must define the same declared interface;
        // collect_exports already rejected an incomplete snapshot.
        const std::vector<Export>& shape = exits_.front().exports;
        for (std::size_t index = 1; index < exits_.size(); ++index) {
            const std::vector<Export>& other = exits_[index].exports;
            const bool same = other.size() == shape.size() &&
                std::equal(other.begin(), other.end(), shape.begin(),
                           [](const Export& left, const Export& right) {
                               return left.kind == right.kind &&
                                      left.component == right.component;
                           });
            if (!same) {
                fail(Error::Code::MissingOutput, builder_.pc);
            }
        }
        GraphSpec spec;
        spec.stage = program_.stage;
        spec.sph = program_.sph;
        spec.exports.reserve(shape.size());
        for (std::size_t component = 0; component < shape.size();
             ++component) {
            ScalarPhi phi;
            phi.arms.reserve(exits_.size());
            for (const ExitSnapshot& exit : exits_) {
                phi.arms.push_back(ScalarPhiArm{
                    BlockId{exit.block}, exit.exports[component].value});
            }
            Export merged = shape[component];
            const bool identical = std::all_of(phi.arms.begin(), phi.arms.end(),
                [&](const auto& arm) { return arm.value.index == phi.arms.front().value.index; });
            merged.value = identical ? phi.arms.front().value
                                     : ScalarId{builder_.append(std::move(phi))};
            spec.exports.push_back(merged);
        }
        // Node collection happens after the join phis are appended so the
        // return block range covers every node exactly once.
        spec.nodes.assign(builder_.program.nodes().begin(),
                          builder_.program.nodes().end());
        block.end = node_count();
        block.terminator = Return{};
        blocks_.push_back(block);
        spec.blocks = std::move(blocks_);
        return make_graph(spec, builder_.limits);
    }

    const mw::ForwardProgram& program_;
    Builder& builder_;
    ForwardState state_;
    detail::InitExpressions expressions_;
    std::vector<BasicBlock> blocks_;
    std::vector<ExitSnapshot> exits_;
    std::map<std::uint32_t, std::vector<Incoming>> incoming_;
    std::set<std::uint32_t> leaders_;
    std::map<std::uint32_t, std::uint32_t> literals_;
    std::map<std::uint32_t, std::size_t> index_of_;
};

// Metadata admission for every decoded instruction, run before the
// structural walk touches any typed payload and before any guard decision:
// forged payloads, unsupported modifiers, and unimplemented profiles are
// rejected with the Builder's own error code, pc, and detail even under a
// statically false guard, so a skipped instruction can never act as an
// unvalidated no-op. The context is the begin_graph-initialized Builder;
// branch shape is the structural validator's own contract.
void validate_instruction_metadata(const mw::ForwardProgram& program,
                                   const Builder& context) {
    for (const mw::ForwardInstruction& node : program.instructions) {
        if (std::holds_alternative<mw::ForwardBranch>(node)) continue;
        const mw::Instruction& instruction = std::get<mw::Instruction>(node);
        if (const std::optional<Error> error =
                detail::validate_forward_instruction_metadata(instruction,
                                                              context)) {
            fail(error->code, error->pc, error->detail);
        }
    }
}

// Independent validation of the public mutable ForwardProgram layout: pc
// shape, registry availability, branch targets, and the structural
// successor closure computed with the same guard rules the decoder used,
// before any fact-driven pruning runs. Typed payload consistency and
// modifier profiles are already covered by the metadata pass above, so no
// check here touches the payload variant.
void validate_forward_program(const mw::ForwardProgram& program) {
    using F = mw::OpcodeFamily;
    if (program.stage != shader_input::Stage::Vertex &&
        program.stage != shader_input::Stage::Fragment) {
        fail(Error::Code::UnsupportedStage, 0);
    }
    if (program.instructions.empty()) {
        fail(Error::Code::MissingExit, kEntryPc);
    }
    std::map<std::uint32_t, std::size_t> positions;
    std::uint32_t previous = 0;
    for (std::size_t slot = 0; slot < program.instructions.size(); ++slot) {
        const mw::ForwardInstruction& node = program.instructions[slot];
        const std::uint32_t pc = node_pc(node);
        if (slot != 0 && pc <= previous) {
            fail(Error::Code::InvalidInstruction, pc, pc);
        }
        previous = pc;
        if (pc < kEntryPc || pc % 8 != 0 || pc % 32 == 0 ||
            static_cast<std::uint64_t>(pc) + 8 > program.program_size) {
            fail(Error::Code::InvalidInstruction, pc, pc);
        }
        mw::Predicate guard{7, false};
        if (const auto* branch = std::get_if<mw::ForwardBranch>(&node)) {
            guard = branch->predicate;
            if (branch->target_pc % 8 != 0 || branch->target_pc % 32 == 0 ||
                static_cast<std::uint64_t>(branch->target_pc) + 8 >
                    program.program_size ||
                branch->target_pc <= branch->source_pc) {
                fail(Error::Code::InvalidInstruction, branch->source_pc,
                     branch->target_pc);
            }
        } else {
            const mw::Instruction& instruction =
                std::get<mw::Instruction>(node);
            guard = instruction.predicate;
            if (instruction.pc != pc) {
                fail(Error::Code::InvalidInstruction, pc, instruction.pc);
            }
            if (!family_admitted(instruction.family)) {
                fail(Error::Code::UnsupportedModifier, pc,
                     static_cast<std::uint32_t>(instruction.family));
            }
        }
        if (guard.index > 7) {
            fail(Error::Code::InvalidInstruction, pc, guard.index);
        }
        if (!positions.emplace(pc, slot).second) {
            fail(Error::Code::InvalidInstruction, pc, pc);
        }
    }
    const auto entry = positions.find(kEntryPc);
    if (entry == positions.end()) {
        fail(Error::Code::MissingExit, kEntryPc);
    }

    // Structural reachability with PT / inverted-PT / guarded edges, before
    // any fact pruning: correlated predicates may prune structurally
    // reachable paths later, but genuinely extra input nodes outside this
    // closure are rejected.
    std::vector<bool> visited(program.instructions.size(), false);
    std::vector<std::size_t> worklist{entry->second};
    visited[entry->second] = true;
    while (!worklist.empty()) {
        const std::size_t slot = worklist.back();
        worklist.pop_back();
        const mw::ForwardInstruction& node = program.instructions[slot];
        const std::uint32_t pc = node_pc(node);
        const auto enqueue = [&](std::uint32_t successor) {
            const auto found = positions.find(successor);
            if (found == positions.end()) {
                fail(Error::Code::InvalidInstruction, pc, successor);
            }
            if (!visited[found->second]) {
                visited[found->second] = true;
                worklist.push_back(found->second);
            }
        };
        mw::Predicate guard{7, false};
        bool path_done = false;
        if (const auto* branch = std::get_if<mw::ForwardBranch>(&node)) {
            guard = branch->predicate;
            if (!is_inverted_pt(guard)) {
                enqueue(branch->target_pc);
            }
            // An always-PT branch transfers control; its fallthrough is not
            // structurally reachable.
            path_done = is_pt(guard);
        } else {
            const mw::Instruction& instruction =
                std::get<mw::Instruction>(node);
            // A plain PT-guarded instruction still falls through; only an
            // unconditional PT EXIT ends the structural path.
            path_done = instruction.family == F::Exit && is_pt(instruction.predicate);
        }
        if (path_done) {
            continue;
        }
        std::uint32_t next = 0;
        if (!fallthrough_pc(pc, program.program_size, next)) {
            fail(Error::Code::MissingExit, pc);
        }
        enqueue(next);
    }
    for (std::size_t slot = 0; slot < visited.size(); ++slot) {
        if (!visited[slot]) {
            const std::uint32_t pc = node_pc(program.instructions[slot]);
            fail(Error::Code::InvalidInstruction, pc, pc);
        }
    }
}

}  // namespace

LowerResult lower_forward(const mw::ForwardProgram& program, Limits limits) {
    try {
        // begin_graph validates the stage, adopts the SPH, and creates the
        // entry zero literal; the metadata pass then queries that real
        // context for the stage-, SPH-, and interface-dependent profiles.
        Builder builder{limits};
        builder.begin_graph(program.stage, program.sph);
        validate_instruction_metadata(program, builder);
        validate_forward_program(program);
        ForwardLowerer lowerer{program, builder};
        return lowerer.run();
    } catch (const Failure& failure) {
        return failure.error;
    } catch (const std::bad_alloc&) {
        return Error{Error::Code::Allocation};
    } catch (const std::length_error&) {
        return Error{Error::Code::Allocation};
    }
}

}  // namespace armigo::shader::graphics
