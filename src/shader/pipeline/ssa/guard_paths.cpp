// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include "pipeline/ssa/guard_paths.h"
#include "pipeline/ssa/dominance.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace armigo::shader::pipeline {
namespace {
// A signed atom identifies an expression over reaching SSA definitions,
// never a register name. Boolean identities preserve that identity; literal
// predicates are exact constants. Phis acquire facts only on their selected edge.
struct Atom {
    ValueId value;
    bool inverted{};
    std::optional<bool> constant;
};
using Facts = std::map<std::uint32_t, bool>;
struct State {
    BlockId block;
    Facts facts;
    auto operator<=>(const State&) const = default;
};

class GuardPaths {
public:
    GuardPaths(const Program& registers, const Program& provisional, std::uint64_t& steps)
        : registers_(registers), ssa_(provisional), steps_(steps), output_(registers) {}

    Result<std::optional<Program>> run(bool preserve_cfg = false) {
        definitions_.resize(ssa_.values.size());
        producers_.resize(ssa_.values.size());
        for (std::uint32_t i = 0; i < ssa_.blocks.size(); ++i) {
            for (const auto& phi : ssa_.blocks[i].phis) {
                if (!charge()) return *error_;
                definitions_[phi.output.value] = BlockId{i};
            }
            for (const auto id : ssa_.blocks[i].operations) {
                if (!charge()) return *error_;
                const auto& op = ssa_.operations[id.value];
                for (const auto& operand : op.outputs) {
                    const auto value = std::get<ValueId>(operand);
                    definitions_[value.value] = BlockId{i};
                    producers_[value.value] = id;
                }
            }
        }
        kills_.resize(ssa_.values.size());
        for (std::uint32_t i = 0; i < definitions_.size(); ++i)
            if (definitions_[i].valid()) kills_[i].insert(definitions_[i]);
        normalized_.resize(ssa_.values.size());
        phi_definitions_.resize(ssa_.values.size());
        for (const auto& block : ssa_.blocks)
            for (const auto& phi : block.phis)
                phi_definitions_[phi.output.value] = &phi;
        atoms_.resize(ssa_.blocks.size());
        std::map<std::uint32_t, std::vector<BlockId>> tests;
        for (std::uint32_t i = 0; i < ssa_.blocks.size(); ++i) {
            if (!charge()) return *error_;
            const auto* branch = std::get_if<Branch>(&ssa_.blocks[i].terminator);
            if (!branch) continue;
            atoms_[i] = atom(branch->condition);
            if (error_) return *error_;
            if (atoms_[i].value.valid()) tests[atoms_[i].value.value].push_back(BlockId{i});
        }
        // Changing reconvergence can change collective/derivative behavior.
        // Such functions remain ineligible for specialization. Fixed-CFG
        // proof does not change reconvergence and can analyze them normally.
        std::vector<bool> eligible(ssa_.functions.size(), true);
        constexpr auto collective = Effects::subgroup | Effects::derivative | Effects::barrier;
        for (const auto& block : ssa_.blocks) {
            for (const auto id : block.operations) {
                if (!charge()) return *error_;
                if (!preserve_cfg && (ssa_.operations[id.value].effects & collective) != Effects::none)
                    eligible[block.function.value] = false;
            }
        }

        // A predicate phi is evaluated on its selected predecessor edge,
        // simultaneously with the other phis. Keep precisely the input facts
        // needed for those transfers, including loop-carried predicate copies.
        std::vector<std::uint32_t> pending_phis;
        for (const auto& [value, blocks] : tests) {
            (void)blocks;
            pending_phis.push_back(value);
        }
        std::set<std::uint32_t> visited;
        transfers_.resize(ssa_.blocks.size());
        while (!pending_phis.empty()) {
            if (!charge()) return *error_;
            const auto value = pending_phis.back();
            pending_phis.pop_back();
            if (!visited.insert(value).second || !phi_definitions_[value]) continue;
            const auto& phi = *phi_definitions_[value];
            transfers_[definitions_[value].value].push_back(&phi);
            for (const auto& input : phi.inputs) {
                const auto incoming = atom(Operand{input.value});
                if (error_) return *error_;
                if (!incoming.value.valid()) continue;
                tests[incoming.value.value].push_back(input.predecessor);
                pending_phis.push_back(incoming.value.value);
            }
        }

        // Facts expire when a contributing SSA definition runs again. A phi's
        // new incoming fact is retained at its definition block, but the old
        // fact never propagates backwards across that definition.
        live_.resize(ssa_.blocks.size());
        bool candidates = false;
        for (const auto& [value, blocks] : tests) {
            if (!eligible[ssa_.blocks[blocks.front().value].function.value]) continue;
            candidates |= blocks.size() >= 2 || phi_definitions_[value] != nullptr;
            std::vector<BlockId> pending = blocks;
            while (!pending.empty()) {
                if (!charge()) return *error_;
                const auto block = pending.back();
                pending.pop_back();
                if (kills_[value].contains(block)) {
                    if (phi_definitions_[value]) live_[block.value].insert(value);
                    continue;
                }
                if (!live_[block.value].insert(value).second) continue;
                for (const auto predecessor : ssa_.blocks[block.value].predecessors)
                    pending.push_back(predecessor);
            }
        }
        for (std::size_t i = 0; i < atoms_.size(); ++i)
            candidates |= atoms_[i].constant.has_value() && eligible[ssa_.blocks[i].function.value];
        eligible_ = std::move(eligible);

        if (preserve_cfg) return resolve_phi_inputs();
        if (!candidates) return std::optional<Program>{};
        // Invariant 8: no undefined operand is replaced. A clone represents
        // exactly the original paths with its recorded predicate outcomes.
        // Re-testing the same predicate over the same dynamic inputs selects only that edge.
        // Keeping those histories separate makes feasibility ordinary CFG
        // dominance when SSA is rebuilt. Every operation on an executed path
        // still executes once, in its original order.
        output_.blocks.clear();
        output_.operations.clear();
        for (auto& function : output_.functions) function.blocks.clear();
        for (std::size_t i = 0; i < registers_.functions.size(); ++i) {
            const auto entry = intern(registers_.functions[i].entry, {});
            if (error_) return *error_;
            output_.functions[i].entry = entry;
        }
        for (std::size_t next = 0; next < states_.size(); ++next) {
            if (!charge()) return *error_;
            // intern() can grow both vectors, so retain no references to them.
            const State state = states_[next];
            const auto& source = registers_.blocks[state.block.value];
            std::vector<OperationId> operations;
            for (const auto id : source.operations) {
                if (!charge()) return *error_;
                if (output_.operations.size() >= registers_.limits.operations)
                    return exceeded("operation");
                operations.push_back(OperationId{static_cast<std::uint32_t>(output_.operations.size())});
                output_.operations.push_back(registers_.operations[id.value]);
            }
            Terminator terminator = source.terminator;
            if (auto* branch = std::get_if<Branch>(&terminator)) {
                const Atom condition = atoms_[state.block.value];
                const auto known = evaluate(condition, state.facts);
                if (eligible_[source.function.value] && known) {
                    const auto target = *known
                        ? branch->when_true : branch->when_false;
                    terminator = Jump{intern(target, state.facts, state.block)};
                    ++folded_;
                } else {
                    auto on_true = state.facts;
                    auto on_false = state.facts;
                    if (condition.value.valid()) {
                        on_true[condition.value.value] = !condition.inverted;
                        on_false[condition.value.value] = condition.inverted;
                    }
                    branch->when_true = intern(branch->when_true, std::move(on_true), state.block);
                    branch->when_false = intern(branch->when_false, std::move(on_false), state.block);
                }
            } else if (auto* jump = std::get_if<Jump>(&terminator)) {
                jump->target = intern(jump->target, state.facts, state.block);
            } else if (auto* select = std::get_if<Switch>(&terminator)) {
                for (auto& item : select->cases) item.target = intern(item.target, state.facts, state.block);
                select->fallback = intern(select->fallback, state.facts, state.block);
            }
            if (error_) return *error_;
            edges_ += successors(terminator).size();
            if (edges_ > registers_.limits.edges) return exceeded("edge");
            output_.blocks[next].operations = std::move(operations);
            output_.blocks[next].terminator = std::move(terminator);
        }
        if (folded_ == 0) return std::optional<Program>{};
        for (std::uint32_t i = 0; i < output_.blocks.size(); ++i) {
            for (const auto target : successors(output_.blocks[i].terminator)) {
                if (!charge()) return *error_;
                auto& predecessors = output_.blocks[target.value].predecessors;
                if (std::find(predecessors.begin(), predecessors.end(), BlockId{i}) == predecessors.end())
                    predecessors.push_back(BlockId{i});
            }
        }
        return std::optional<Program>{std::move(output_)};
    }

private:
    bool charge() {
        if (++steps_ <= registers_.limits.analysis_steps) return true;
        error_ = exceeded("analysis work");
        return false;
    }
    Diagnostic exceeded(const char* resource) const {
        return diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded, {},
                        std::string("Correlated guard analysis exceeded its ") + resource + " limit");
    }
    static Atom negate(Atom value) {
        if (value.constant) value.constant = !*value.constant;
        else value.inverted = !value.inverted;
        return value;
    }
    static bool same(Atom left, Atom right) {
        return left.value.valid() && left.value == right.value && left.inverted == right.inverted;
    }
    std::optional<bool> evaluate(Atom value, const Facts& facts) const {
        if (value.constant) return value.constant;
        const auto found = facts.find(value.value.value);
        if (found == facts.end()) return std::nullopt;
        return found->second != value.inverted;
    }
    Atom cached(const Operand& operand) const {
        if (const auto* literal = std::get_if<LiteralBits>(&operand))
            return {{}, false, literal->bits != 0};
        if (const auto* value = std::get_if<ValueId>(&operand))
            return normalized_[value->value].value_or(Atom{*value, false, {}});
        return {};
    }
    bool transparent(const Operation& op) const {
        if (op.effects != Effects::none) return false;
        switch (op.opcode) {
        case Opcode::copy: case Opcode::logical_not:
        case Opcode::logical_and: case Opcode::logical_or: case Opcode::logical_xor:
        case Opcode::select: break;
        default: return false;
        }
        return std::all_of(op.inputs.begin(), op.inputs.end(), [&](const Operand& input) {
            if (const auto* value = std::get_if<ValueId>(&input))
                return ssa_.values[value->value].type == ValueType::predicate;
            const auto* literal = std::get_if<LiteralBits>(&input);
            return literal && literal->type == ValueType::predicate;
        });
    }
    Atom simplify(ValueId value, const Operation& op) {
        const auto a = cached(op.inputs[0]);
        if (op.opcode == Opcode::copy) return a;
        if (op.opcode == Opcode::logical_not) return negate(a);
        const auto b = cached(op.inputs[1]);
        if (op.opcode == Opcode::select) {
            const auto c = cached(op.inputs[2]);
            if (a.constant) return *a.constant ? b : c;
            if (same(b, c)) return b;
            if (b.constant && c.constant && *b.constant != *c.constant)
                return *b.constant ? a : negate(a);
        } else if (op.opcode == Opcode::logical_and) {
            if (a.constant) return *a.constant ? b : a;
            if (b.constant) return *b.constant ? a : b;
            if (same(a, b)) return a;
        } else if (op.opcode == Opcode::logical_or) {
            if (a.constant) return *a.constant ? a : b;
            if (b.constant) return *b.constant ? b : a;
            if (same(a, b)) return a;
        } else if (op.opcode == Opcode::logical_xor) {
            if (a.constant) return *a.constant ? negate(b) : b;
            if (b.constant) return *b.constant ? negate(a) : a;
        }
        if (op.opcode != Opcode::select && a.value.valid() && b.value.valid()) {
            auto left = std::pair{a.value, a.inverted};
            auto right = std::pair{b.value, b.inverted};
            if (right < left) std::swap(left, right);
            const auto key = std::tuple{op.opcode, left, right};
            const auto [found, inserted] = boolean_expressions_.emplace(key, value);
            if (inserted) {
                kills_[value.value].clear();
                for (const auto input : {a.value, b.value}) {
                    for (const auto block : kills_[input.value]) {
                        if (!charge()) return {};
                        kills_[value.value].insert(block);
                    }
                }
            }
            return {found->second, false, {}};
        }
        return {value, false, {}};
    }

    using Word = std::variant<ValueId, LiteralBits>;
    std::optional<Atom> integer_comparison(ValueId value, const Operation& op) {
        if (op.opcode != Opcode::compare || op.effects != Effects::none ||
            (op.numeric.scalar != ScalarType::s32 && op.numeric.scalar != ScalarType::u32) ||
            std::get<ComparisonInfo>(op.info).unordered) return std::nullopt;
        std::array<Word, 2> inputs;
        bool has_value = false;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (const auto* input = std::get_if<ValueId>(&op.inputs[i])) {
                if (ssa_.values[input->value].type != ValueType::word32) return std::nullopt;
                inputs[i] = *input;
                has_value = true;
            } else if (const auto* literal = std::get_if<LiteralBits>(&op.inputs[i])) inputs[i] = *literal;
            else return std::nullopt;
        }
        // A shared SSA input ties the expression to one function. Memory
        // operands are excluded: only already evaluated words are identical.
        if (!has_value) return std::nullopt;
        const auto key = std::tuple{op.numeric, std::get<ComparisonInfo>(op.info).relation, inputs};
        const auto [found, inserted] = comparisons_.emplace(key, value);
        if (inserted) {
            kills_[value.value].clear();
            for (const auto& input : inputs)
                if (const auto* id = std::get_if<ValueId>(&input); id && definitions_[id->value].valid())
                    kills_[value.value].insert(definitions_[id->value]);
        }
        return Atom{found->second, false, {}};
    }

    struct MaskPredicate { Operand condition; bool inverted; };
    std::optional<MaskPredicate> mask_predicate(const Operation& op) {
        if (op.opcode != Opcode::compare || op.effects != Effects::none ||
            (op.numeric.scalar != ScalarType::s32 && op.numeric.scalar != ScalarType::u32)) return std::nullopt;
        const auto& comparison = std::get<ComparisonInfo>(op.info);
        const auto* zero = std::get_if<LiteralBits>(&op.inputs[1]);
        if (!zero || zero->type != ValueType::word32 || zero->bits != 0 || comparison.unordered)
            return std::nullopt;
        const bool equal = comparison.relation == Compare::equal;
        if (!equal && comparison.relation != Compare::not_equal &&
            !(comparison.relation == Compare::less && op.numeric.scalar == ScalarType::s32))
            return std::nullopt;
        Operand source = op.inputs[0];
        for (std::size_t i = 0; i < ssa_.values.size(); ++i) {
            if (!charge()) return std::nullopt;
            const auto* value = std::get_if<ValueId>(&source);
            if (!value || !producers_[value->value].valid()) return std::nullopt;
            const auto& producer = ssa_.operations[producers_[value->value].value];
            if (producer.effects != Effects::none) return std::nullopt;
            if (producer.opcode == Opcode::copy) { source = producer.inputs[0]; continue; }
            if (producer.opcode == Opcode::predicate_to_mask)
                return MaskPredicate{producer.inputs[0], equal};
            if (producer.opcode != Opcode::select) return std::nullopt;
            const auto* yes = std::get_if<LiteralBits>(&producer.inputs[1]);
            const auto* no = std::get_if<LiteralBits>(&producer.inputs[2]);
            if (!yes || !no || yes->type != ValueType::word32 || no->type != ValueType::word32)
                return std::nullopt;
            // ISET's integer result is exactly 0 or all ones. Z and S tests
            // of that mask recover the original predicate or its complement.
            if (yes->bits == 0xffffffffu && no->bits == 0)
                return MaskPredicate{producer.inputs[0], equal};
            if (yes->bits == 0 && no->bits == 0xffffffffu)
                return MaskPredicate{producer.inputs[0], !equal};
            return std::nullopt;
        }
        return std::nullopt;
    }

    Atom atom(const Operand& operand) {
        const auto* root = std::get_if<ValueId>(&operand);
        if (!root) return cached(operand);
        // Explicit postorder avoids host-stack growth on long predicate-copy
        // chains. Phis stay opaque here, so dependencies cannot form cycles.
        std::vector<std::pair<ValueId, bool>> pending{{*root, false}};
        while (!pending.empty()) {
            if (!charge()) return {};
            const auto [value, ready] = pending.back();
            pending.pop_back();
            if (normalized_[value.value]) continue;
            const auto producer = producers_[value.value];
            if (!producer.valid()) {
                normalized_[value.value] = Atom{value, false, {}};
                continue;
            }
            const auto& op = ssa_.operations[producer.value];
            if (const auto mask = mask_predicate(op)) {
                if (ready) {
                    auto result = cached(mask->condition);
                    normalized_[value.value] = mask->inverted ? negate(result) : result;
                } else {
                    pending.emplace_back(value, true);
                    if (const auto* source = std::get_if<ValueId>(&mask->condition))
                        pending.emplace_back(*source, false);
                }
                continue;
            }
            if (error_) return {};
            if (const auto comparison = integer_comparison(value, op)) {
                normalized_[value.value] = *comparison;
                continue;
            }
            if (!transparent(op)) {
                normalized_[value.value] = Atom{value, false, {}};
                continue;
            }
            if (ready) {
                normalized_[value.value] = simplify(value, op);
                continue;
            }
            pending.emplace_back(value, true);
            for (const auto& input : op.inputs)
                if (const auto* source = std::get_if<ValueId>(&input)) pending.emplace_back(*source, false);
        }
        return cached(operand);
    }
    Facts transfer_facts(BlockId block, Facts facts, BlockId predecessor) {
        // Read all phi inputs from the predecessor's facts before assigning
        // any output, including mutually dependent loop-carried phis.
        std::vector<std::pair<std::uint32_t, std::optional<bool>>> incoming;
        if (predecessor.valid() && eligible_[registers_.blocks[block.value].function.value]) {
            for (const auto* phi : transfers_[block.value]) {
                for (const auto& input : phi->inputs) {
                    if (!charge()) return {};
                    if (input.predecessor == predecessor)
                        incoming.emplace_back(phi->output.value, evaluate(cached(Operand{input.value}), facts));
                }
            }
        }
        for (const auto& [value, truth] : incoming) {
            if (truth) facts[value] = *truth;
            else facts.erase(value);
        }
        for (auto it = facts.begin(); it != facts.end();) {
            if (!charge()) return {};
            if (!live_[block.value].contains(it->first)) it = facts.erase(it);
            else ++it;
        }
        return facts;
    }

    BlockId intern(BlockId block, Facts facts, BlockId predecessor = {}) {
        facts = transfer_facts(block, std::move(facts), predecessor);
        if (error_) return {};
        State state{block, std::move(facts)};
        if (const auto found = ids_.find(state); found != ids_.end()) return found->second;
        if (output_.blocks.size() >= registers_.limits.blocks) {
            error_ = exceeded("block");
            return {};
        }
        const BlockId id{static_cast<std::uint32_t>(output_.blocks.size())};
        ids_.emplace(state, id);
        states_.push_back(std::move(state));
        BasicBlock clone;
        clone.function = registers_.blocks[block.value].function;
        clone.source = registers_.blocks[block.value].source;
        output_.blocks.push_back(std::move(clone));
        output_.functions[registers_.blocks[block.value].function.value].blocks.push_back(id);
        return id;
    }

    // Fixed-CFG invariant: undefinedness may travel through phi copies, but
    // never reaches an executed operation or terminator on any feasible path.
    // Only after proving this for every missing definition do we replace its
    // phi inputs with existing, type-matching definitions that dominate the
    // incoming edge. Their bits are unobservable. This is not register
    // initialization: a feasible instruction-level undefined read still fails.
    // No operation, block, edge, terminator, or effect is added or changed, so
    // reconvergence, discard ordering, and derivative/collective lane sets
    // remain exactly those of the input CFG. Unsupported proofs fail closed.
    Result<std::optional<Program>> resolve_phi_inputs() {
        std::vector<bool> missing(ssa_.values.size());
        for (std::size_t i = 0; i < missing.size(); ++i)
            missing[i] = !definitions_[i].valid();
        for (const auto& function : ssa_.functions)
            for (const auto value : function.parameters) missing[value.value] = false;
        bool has_missing_phi = false;
        for (const auto& block : ssa_.blocks)
            for (const auto& phi : block.phis)
                for (const auto& input : phi.inputs) {
                    if (!charge()) return *error_;
                    has_missing_phi |= missing[input.value.value];
                }
        if (!has_missing_phi) return std::optional<Program>{};

        // Forget taint only after its last possible use, including phi reads
        // on outgoing edges. Otherwise independent dead histories can multiply
        // states without adding any information to the undefined-read proof.
        std::vector<std::set<std::uint32_t>> taint_live(ssa_.blocks.size());
        std::vector<std::pair<BlockId, ValueId>> uses;
        const auto use = [&](BlockId block, const Operand& operand) {
            if (const auto* value = std::get_if<ValueId>(&operand);
                value && phi_definitions_[value->value]) uses.emplace_back(block, *value);
        };
        for (std::uint32_t i = 0; i < ssa_.blocks.size(); ++i) {
            const auto& block = ssa_.blocks[i];
            for (const auto& phi : block.phis)
                for (const auto& input : phi.inputs) {
                    if (!charge()) return *error_;
                    use(input.predecessor, Operand{input.value});
                }
            for (const auto id : block.operations)
                for (const auto& input : ssa_.operations[id.value].inputs) {
                    if (!charge()) return *error_;
                    use(BlockId{i}, input);
                }
            if (const auto* branch = std::get_if<Branch>(&block.terminator)) use(BlockId{i}, branch->condition);
            else if (const auto* select = std::get_if<Switch>(&block.terminator)) use(BlockId{i}, select->selector);
            else if (const auto* result = std::get_if<Return>(&block.terminator))
                for (const auto& input : result->values) use(BlockId{i}, input);
        }
        while (!uses.empty()) {
            if (!charge()) return *error_;
            const auto [block, value] = uses.back();
            uses.pop_back();
            if (!taint_live[block.value].insert(value.value).second || block == definitions_[value.value])
                continue;
            for (const auto predecessor : ssa_.blocks[block.value].predecessors)
                uses.emplace_back(predecessor, value);
        }

        struct ProofState {
            BlockId block;
            Facts facts;
            std::set<std::uint32_t> tainted;
            auto operator<=>(const ProofState&) const = default;
        };
        std::set<ProofState> seen;
        std::vector<ProofState> pending;
        auto enqueue = [&](ProofState state) {
            if (!charge()) return;
            for (auto it = state.tainted.begin(); it != state.tainted.end();) {
                if (!charge()) return;
                if (!taint_live[state.block.value].contains(*it)) it = state.tainted.erase(it);
                else ++it;
            }
            if (seen.contains(state)) return;
            if (seen.size() >= ssa_.limits.token_states) {
                error_ = exceeded("predicate state");
                return;
            }
            seen.insert(state);
            pending.push_back(std::move(state));
        };
        for (const auto& function : ssa_.functions)
            enqueue({function.entry, {}, {}});
        if (error_) return *error_;
        while (!pending.empty()) {
            if (!charge()) return *error_;
            auto state = std::move(pending.back());
            pending.pop_back();
            const auto& block = ssa_.blocks[state.block.value];
            const auto tainted = [&](ValueId value) {
                return missing[value.value] || state.tainted.contains(value.value);
            };
            const auto safe = [&](const Operand& operand) {
                const auto* value = std::get_if<ValueId>(&operand);
                return !value || !tainted(*value);
            };
            for (const auto id : block.operations) {
                if (!charge()) return *error_;
                for (const auto& input : ssa_.operations[id.value].inputs) {
                    if (!charge()) return *error_;
                    if (!safe(input)) return std::optional<Program>{};
                }
            }
            const auto follow = [&](BlockId target, Facts facts) {
                ProofState next{target, transfer_facts(target, std::move(facts), state.block),
                                state.tainted};
                // All sources are read from the old state, even on self edges
                // and mutually dependent loop-carried phi transfers.
                for (const auto& phi : ssa_.blocks[target.value].phis) {
                    for (const auto& input : phi.inputs) {
                        if (!charge()) return;
                        if (input.predecessor != state.block) continue;
                        if (tainted(input.value)) next.tainted.insert(phi.output.value);
                        else next.tainted.erase(phi.output.value);
                    }
                }
                if (!error_) enqueue(std::move(next));
            };
            if (const auto* branch = std::get_if<Branch>(&block.terminator)) {
                if (!safe(branch->condition)) return std::optional<Program>{};
                const auto condition = atoms_[state.block.value];
                if (const auto known = evaluate(condition, state.facts)) {
                    follow(*known ? branch->when_true : branch->when_false, state.facts);
                } else {
                    auto yes = state.facts;
                    auto no = state.facts;
                    if (condition.value.valid()) {
                        yes[condition.value.value] = !condition.inverted;
                        no[condition.value.value] = condition.inverted;
                    }
                    follow(branch->when_true, std::move(yes));
                    follow(branch->when_false, std::move(no));
                }
            } else if (const auto* select = std::get_if<Switch>(&block.terminator)) {
                if (!safe(select->selector)) return std::optional<Program>{};
                for (const auto target : successors(block.terminator)) follow(target, state.facts);
            } else if (const auto* jump = std::get_if<Jump>(&block.terminator)) {
                follow(jump->target, state.facts);
            } else if (const auto* result = std::get_if<Return>(&block.terminator)) {
                for (const auto& value : result->values) {
                    if (!charge()) return *error_;
                    if (!safe(value)) return std::optional<Program>{};
                }
            }
            if (error_) return *error_;
        }

        // Proof used the original missing roots exclusively. A replacement is
        // never fed back into predicate reasoning or used to justify itself.
        Program repaired = ssa_;
        for (const auto& function : ssa_.functions) {
            auto result = find_dominators(ssa_, function, ssa_.limits, steps_);
            if (const auto* failure = std::get_if<Diagnostic>(&result)) return *failure;
            const auto& dominators = std::get<Dominators>(result);
            for (const auto block_id : function.blocks) {
                for (auto& phi : repaired.blocks[block_id.value].phis) {
                    for (auto& input : phi.inputs) {
                        if (!charge()) return *error_;
                        if (!missing[input.value.value]) continue;
                        const auto type = ssa_.values[input.value.value].type;
                        ValueId replacement;
                        auto cursor = input.predecessor;
                        while (!replacement.valid()) {
                            if (!charge()) return *error_;
                            for (const auto id : ssa_.blocks[cursor.value].operations) {
                                for (const auto& output : ssa_.operations[id.value].outputs) {
                                    if (!charge()) return *error_;
                                    const auto value = std::get<ValueId>(output);
                                    if (ssa_.values[value.value].type == type) {
                                        replacement = value;
                                        break;
                                    }
                                }
                                if (replacement.valid()) break;
                            }
                            if (cursor == function.entry) break;
                            cursor = dominators.immediate_dominator[cursor.value];
                        }
                        if (!replacement.valid()) {
                            for (const auto value : function.parameters) {
                                if (!charge()) return *error_;
                                if (ssa_.values[value.value].type == type) {
                                    replacement = value;
                                    break;
                                }
                            }
                        }
                        if (!replacement.valid()) return std::optional<Program>{};
                        input.value = replacement;
                    }
                }
            }
        }
        return std::optional<Program>{std::move(repaired)};
    }

    const Program& registers_;
    const Program& ssa_;
    std::uint64_t& steps_;
    Program output_;
    std::vector<BlockId> definitions_;
    std::vector<OperationId> producers_;
    std::vector<Atom> atoms_;
    std::vector<std::optional<Atom>> normalized_;
    std::vector<std::set<BlockId>> kills_;
    std::map<std::tuple<Opcode, std::pair<ValueId, bool>, std::pair<ValueId, bool>>, ValueId> boolean_expressions_;
    std::map<std::tuple<NumericMode, Compare, std::array<Word, 2>>, ValueId> comparisons_;
    std::vector<const Phi*> phi_definitions_;
    std::vector<std::vector<const Phi*>> transfers_;
    std::vector<bool> eligible_;
    std::vector<std::set<std::uint32_t>> live_;
    std::map<State, BlockId> ids_;
    std::vector<State> states_;
    std::uint64_t folded_{}, edges_{};
    std::optional<Diagnostic> error_;
};
} // namespace

Result<std::optional<Program>> specialize_guard_paths(
    const Program& registers, const Program& provisional, std::uint64_t& steps) {
    return GuardPaths(registers, provisional, steps).run();
}
Result<std::optional<Program>> resolve_guarded_phi_inputs(
    const Program& provisional, std::uint64_t& steps) {
    return GuardPaths(provisional, provisional, steps).run(true);
}
} // namespace armigo::shader::pipeline
