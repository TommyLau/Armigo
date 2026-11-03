// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
//
// Two-pass SSA construction over register-form IR: pass one renames each
// block's local definitions and flags phi sites along iterated dominance
// frontiers; pass two resolves remaining reads by walking immediate
// dominators and lazily materializing predecessor-labelled phis. Missing
// definitions remain provisional until bounded, effect-aware dead-code
// removal proves whether they are live. A live missing definition first
// goes through the guarded-path repairs below; one that survives them reads
// zero (false for a predicate), by owner decision (2026-10-24). Hardware
// initial register contents are not modelled.
// If a live undefined edge survives, first prove that its phi-carried value
// cannot reach an executed instruction under correlated predicates. Only those
// unobservable phi inputs can receive an existing dominating definition, with
// the entire CFG and every operation preserved. Otherwise, bounded
// specialization of repeated predicate SSA identities and selected predicate
// phi inputs makes proven
// infeasible edges explicit in the CFG.
// A second ordinary SSA pass then enforces dominance and applies the same
// zero reading to undefined reads; specialization never substitutes phi inputs. See guard_paths.cpp for both proofs.
// Call operations are rejected because interprocedural register arguments are
// not implemented.

#include "pipeline/phases.h"
#include "pipeline/ssa/dominance.h"
#include "pipeline/ssa/guard_paths.h"

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace armigo::shader::pipeline {
namespace {
constexpr std::uint32_t gpr_slots = 256;
constexpr std::uint32_t predicate_slots = 8;
constexpr std::uint32_t condition_code_slots = 4;

// One key per renameable variable: compiler locals keep their ValueId index,
// physical registers follow behind them. Keys are dense, so per-block maps
// and per-block lookup arrays stay small.
struct BlockDefs {
    std::unordered_map<std::uint32_t, ValueId> definitions; // Reaching definition at block end.
    std::unordered_map<std::uint32_t, ValueId> phi_values;  // Materialized phis, one per key.
    std::unordered_set<std::uint32_t> phi_flags;            // Flagged phi sites awaiting materialization.
};

struct UnresolvedDefinition {
    SourceLocation source;
    std::string description;
};

Limits clamped_limits(const Limits& inherited, const Limits& supplied) {
    Limits effective;
    effective.code_bytes = std::min(inherited.code_bytes, supplied.code_bytes);
    effective.instructions = std::min(inherited.instructions, supplied.instructions);
    effective.functions = std::min(inherited.functions, supplied.functions);
    effective.blocks = std::min(inherited.blocks, supplied.blocks);
    effective.edges = std::min(inherited.edges, supplied.edges);
    effective.operations = std::min(inherited.operations, supplied.operations);
    effective.values = std::min(inherited.values, supplied.values);
    effective.phi_inputs = std::min(inherited.phi_inputs, supplied.phi_inputs);
    effective.operands_per_operation = std::min(inherited.operands_per_operation, supplied.operands_per_operation);
    effective.token_depth = std::min(inherited.token_depth, supplied.token_depth);
    effective.token_states = std::min(inherited.token_states, supplied.token_states);
    effective.regions = std::min(inherited.regions, supplied.regions);
    effective.nesting_depth = std::min(inherited.nesting_depth, supplied.nesting_depth);
    effective.resources = std::min(inherited.resources, supplied.resources);
    effective.dependencies = std::min(inherited.dependencies, supplied.dependencies);
    effective.source_bytes = std::min(inherited.source_bytes, supplied.source_bytes);
    effective.analysis_steps = std::min(inherited.analysis_steps, supplied.analysis_steps);
    return effective;
}

class SsaBuilder {
public:
    SsaBuilder(Program& program, Limits limits, std::uint64_t& steps)
        : p_(program), limits_(clamped_limits(program.limits, limits)), steps_(steps) {
        p_.limits = limits_;
    }

    Status run(bool correlate = true) {
        if (const auto checked = validate(p_, IrForm::registers, Phase::construct_ssa);
            std::holds_alternative<Diagnostic>(checked)) return std::get<Diagnostic>(checked);
        if (const auto rejected = reject_calls(); std::holds_alternative<Diagnostic>(rejected))
            return std::get<Diagnostic>(rejected);
        add_entry_preheaders();
        if (error_) return *error_;
        const std::optional<Program> registers = correlate ? std::optional<Program>{p_} : std::nullopt;
        // Register form leaves the kind unconstrained; SSA output may not
        // carry mutable locals, so every value becomes a plain temporary.
        for (auto& value : p_.values) value.kind = ValueKind::temporary;
        local_count_ = static_cast<std::uint32_t>(p_.values.size());
        total_keys_ = local_count_ + gpr_slots + predicate_slots + condition_code_slots;
        block_defs_.resize(p_.blocks.size());
        local_defs_.assign(total_keys_, ValueId{});
        memo_.assign(total_keys_, ValueId{});
        for (const auto& function : p_.functions) {
            rename_function(function);
            if (error_) break;
        }
        if (error_) return *error_;
        remove_dead_code();
        if (error_ && error_->code == DiagnosticCode::undefined_read && registers) {
            auto repaired = resolve_guarded_phi_inputs(p_, steps_);
            if (const auto* failure = std::get_if<Diagnostic>(&repaired)) return *failure;
            if (auto& fixed_cfg = std::get<std::optional<Program>>(repaired); fixed_cfg) {
                p_ = std::move(*fixed_cfg);
                error_.reset();
            } else {
                auto specialized = specialize_guard_paths(*registers, p_, steps_);
                if (const auto* failure = std::get_if<Diagnostic>(&specialized)) return *failure;
                auto& replacement = std::get<std::optional<Program>>(specialized);
                if (replacement) {
                    p_ = std::move(*replacement);
                    return SsaBuilder(p_, limits_, steps_).run(false);
                }
            }
        }
        if (error_ && error_->code == DiagnosticCode::undefined_read) {
            // No repair proved the read unobservable: it reads zero.
            error_.reset();
            remove_dead_code(true);
        }
        if (error_) return *error_;
        if (const auto checked = validate(p_, IrForm::ssa, Phase::construct_ssa);
            std::holds_alternative<Diagnostic>(checked)) return std::get<Diagnostic>(checked);
        return std::monostate{};
    }

private:
    Status reject_calls() const {
        for (const auto& operation : p_.operations) if (operation.opcode == Opcode::call)
            return diagnose(Phase::construct_ssa, DiagnosticCode::unimplemented, operation.source,
                            "Call ABI handling is not implemented in SSA construction");
        return std::monostate{};
    }

    void fail(Diagnostic diagnostic) {
        if (!error_) error_ = std::move(diagnostic);
    }

    bool charge(SourceLocation where = {}) {
        if (++steps_ <= limits_.analysis_steps) return true;
        fail(diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded, where,
                      "SSA construction exceeded its analysis work limit"));
        return false;
    }

    void add_entry_preheaders() {
        std::uint64_t edge_count = 0;
        for (const auto& block : p_.blocks) {
            if (!charge(block.source)) return;
            edge_count += successors(block.terminator).size();
        }
        for (std::uint32_t index = 0; index < p_.functions.size(); ++index) {
            auto& function = p_.functions[index];
            const BlockId old_entry = function.entry;
            if (p_.blocks[old_entry.value].predecessors.empty()) continue;
            if (p_.blocks.size() >= limits_.blocks) {
                fail(diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded,
                              p_.blocks[old_entry.value].source,
                              "SSA entry preheader exceeds the block limit"));
                return;
            }
            if (edge_count >= limits_.edges) {
                fail(diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded,
                              p_.blocks[old_entry.value].source,
                              "SSA entry preheader exceeds the CFG edge limit"));
                return;
            }
            const BlockId preheader{static_cast<std::uint32_t>(p_.blocks.size())};
            BasicBlock block;
            block.function = FunctionId{index};
            block.source = p_.blocks[old_entry.value].source;
            block.terminator = Jump{old_entry};
            p_.blocks.push_back(std::move(block));
            function.entry = preheader;
            function.blocks.push_back(preheader);
            auto& predecessors = p_.blocks[old_entry.value].predecessors;
            predecessors.push_back(preheader);
            std::sort(predecessors.begin(), predecessors.end());
            ++edge_count;
        }
    }

    std::uint32_t key_of(const Register& reg) const {
        switch (reg.file) {
        case RegisterFile::gpr: return local_count_ + reg.index;
        case RegisterFile::predicate: return local_count_ + gpr_slots + reg.index;
        case RegisterFile::condition_code: return local_count_ + gpr_slots + predicate_slots + reg.index;
        }
        return local_count_;
    }

    ValueType type_of_key(std::uint32_t key) const {
        if (key < local_count_) return p_.values[key].type;
        if (key < local_count_ + gpr_slots) return ValueType::word32;
        return ValueType::predicate;
    }

    std::string describe_key(std::uint32_t key) const {
        if (key < local_count_) return "Compiler local value " + std::to_string(key);
        if (key < local_count_ + gpr_slots) return "Guest register R" + std::to_string(key - local_count_);
        if (key < local_count_ + gpr_slots + predicate_slots)
            return "Predicate register P" + std::to_string(key - local_count_ - gpr_slots);
        return "Condition code flag " + std::to_string(key - local_count_ - gpr_slots - predicate_slots);
    }

    // R255 and P7 are read-only zero/true pseudo-registers; reads resolve to
    // constants and writes cannot occur after register-form validation.
    std::optional<Operand> pseudo_read(const Register& reg) const {
        if (reg.file == RegisterFile::gpr && reg.index == zero_register.index)
            return Operand{LiteralBits{0, ValueType::word32}};
        if (reg.file == RegisterFile::predicate && reg.index == true_predicate.index)
            return Operand{LiteralBits{1, ValueType::predicate}};
        return std::nullopt;
    }

    ValueId new_value(ValueType type, SourceLocation where) {
        if (p_.values.size() >= limits_.values) {
            fail(diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded, where, "SSA value limit exceeded"));
            return ValueId{};
        }
        const auto id = static_cast<std::uint32_t>(p_.values.size());
        p_.values.push_back({type, ValueKind::temporary});
        return ValueId{id};
    }

    void rename_function(const Function& function) {
        auto dominators = find_dominators(p_, function, limits_, steps_);
        if (const auto* failure = std::get_if<Diagnostic>(&dominators)) { fail(*failure); return; }
        auto frontiers = find_dominance_frontiers(p_, function, std::get<Dominators>(dominators), limits_, steps_);
        if (const auto* failure = std::get_if<Diagnostic>(&frontiers)) { fail(*failure); return; }
        dominators_ = std::move(std::get<Dominators>(dominators));
        frontiers_ = std::move(std::get<std::vector<std::vector<BlockId>>>(frontiers));
        entry_ = function.entry;
        parameters_.clear();
        for (const auto parameter : function.parameters) parameters_.insert(parameter.value);
        for (const auto block : function.blocks) block_defs_[block.value] = {};

        for (const auto block : function.blocks) {
            rename_block_locals(block);
            if (error_) return;
        }
        for (const auto block : function.blocks) {
            rename_block_global(block);
            if (error_) return;
        }
    }

    // ---- First pass: block-local renaming, global definition and phi flags.

    Operand rename_local(const Operand& operand) {
        if (const auto* reg = std::get_if<Register>(&operand)) {
            if (auto literal = pseudo_read(*reg)) return *literal;
            const auto key = key_of(*reg);
            if (local_defs_[key].valid()) return Operand{local_defs_[key]};
            return operand;
        }
        if (const auto* id = std::get_if<ValueId>(&operand)) {
            if (id->value >= local_count_) return operand; // Already a renamed SSA value.
            if (local_defs_[id->value].valid()) return Operand{local_defs_[id->value]};
        }
        return operand;
    }

    std::optional<std::uint32_t> output_key(const Operand& operand) const {
        if (const auto* id = std::get_if<ValueId>(&operand)) {
            if (id->value < local_count_) return id->value;
            return std::nullopt;
        }
        if (const auto* reg = std::get_if<Register>(&operand)) return key_of(*reg);
        return std::nullopt;
    }

    void define_local(std::uint32_t key, ValueId value) {
        if (!local_defs_[key].valid()) block_touched_.push_back(key);
        local_defs_[key] = value;
    }

    void clear_local_defs() {
        for (const auto key : block_touched_) local_defs_[key] = ValueId{};
        block_touched_.clear();
    }

    void rename_terminator_locals(BasicBlock& block) {
        if (auto* branch = std::get_if<Branch>(&block.terminator)) branch->condition = rename_local(branch->condition);
        else if (auto* select = std::get_if<Switch>(&block.terminator)) select->selector = rename_local(select->selector);
        else if (auto* result = std::get_if<Return>(&block.terminator))
            for (auto& value : result->values) value = rename_local(value);
    }

    void rename_block_locals(BlockId block_id) {
        auto& block = p_.blocks[block_id.value];
        block_touched_.clear();
        for (const auto operation_id : block.operations) {
            auto& operation = p_.operations[operation_id.value];
            for (auto& input : operation.inputs) input = rename_local(input);
            for (auto& output : operation.outputs) {
                const auto key = output_key(output);
                if (!key) continue;
                const ValueId fresh = new_value(type_of_key(*key), operation.source);
                if (error_) return;
                define_local(*key, fresh);
                output = Operand{fresh};
            }
        }
        rename_terminator_locals(block);
        if (error_) return;

        // Record the final reaching definition of every written key and flag
        // phis along the iterated dominance frontiers of that definition.
        for (const auto key : block_touched_) {
            if (!charge()) return;
            block_defs_[block_id.value].definitions.emplace(key, local_defs_[key]);
            queue_.clear();
            queue_.push_back(block_id);
            while (!queue_.empty()) {
                const auto current = queue_.back();
                queue_.pop_back();
                for (const auto frontier : frontiers_[current.value]) {
                    if (!charge()) return;
                    if (block_defs_[frontier.value].phi_flags.insert(key).second) queue_.push_back(frontier);
                }
            }
        }
        clear_local_defs();
    }

    // ---- Second pass: cross-block reads, lazy phi materialization.

    void clear_memo() {
        for (const auto key : memo_touched_) memo_[key] = ValueId{};
        memo_touched_.clear();
    }

    // Plain definition or materialized phi inside one block, if any.
    std::optional<ValueId> probe_block(std::uint32_t key, BlockId block) const {
        const auto& defs = block_defs_[block.value];
        if (const auto found = defs.definitions.find(key); found != defs.definitions.end()) return found->second;
        if (const auto phi = defs.phi_values.find(key); phi != defs.phi_values.end()) return phi->second;
        return std::nullopt;
    }

    ValueId parameter_or_undefined(std::uint32_t key, SourceLocation where) {
        if (key < local_count_ && parameters_.count(key)) return ValueId{key};
        const ValueId value = new_value(type_of_key(key), where);
        if (!error_)
            unresolved_.emplace(value.value, UnresolvedDefinition{
                where, describe_key(key) + " is read before any definition reaches this point"});
        return value;
    }

    // Definition visible at the end of a block: plain definitions win over
    // phis, because execution passes the whole block before an edge is taken.
    ValueId definition_at_end(std::uint32_t key, BlockId start, SourceLocation where) {
        auto cursor = start;
        while (true) {
            if (!charge(where)) return ValueId{};
            if (auto found = probe_block(key, cursor)) return *found;
            if (block_defs_[cursor.value].phi_flags.count(key)) return materialize_phi(cursor, key, where);
            if (cursor == entry_) break;
            cursor = dominators_.immediate_dominator[cursor.value];
        }
        return parameter_or_undefined(key, where);
    }

    // Definition visible at a use inside a block. Block-local definitions do
    // not apply (pass one already substituted the reads that follow them), so
    // only the block's phis and its immediate dominator chain are consulted.
    ValueId definition_for_use(std::uint32_t key, BlockId block, SourceLocation where) {
        const auto& defs = block_defs_[block.value];
        if (const auto phi = defs.phi_values.find(key); phi != defs.phi_values.end()) return phi->second;
        if (defs.phi_flags.count(key)) return materialize_phi(block, key, where);
        if (block != entry_) return definition_at_end(key, dominators_.immediate_dominator[block.value], where);
        return parameter_or_undefined(key, where);
    }

    ValueId materialize_phi(BlockId block, std::uint32_t key, SourceLocation where) {
        const ValueId output = new_value(type_of_key(key), p_.blocks[block.value].source);
        if (error_) return ValueId{};
        // Register the phi before resolving inputs so backedge searches and
        // repeated lookups observe it instead of materializing duplicates.
        block_defs_[block.value].phi_values.emplace(key, output);
        auto& phis = p_.blocks[block.value].phis;
        phis.push_back(Phi{p_.blocks[block.value].source, output, {}});
        auto& phi = phis.back();
        for (const auto predecessor : p_.blocks[block.value].predecessors) {
            if (!charge(where)) return ValueId{};
            const ValueId incoming = definition_at_end(key, predecessor, where);
            if (error_) return ValueId{};
            if (phi_inputs_ >= limits_.phi_inputs) {
                fail(diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded, where, "Phi input limit exceeded"));
                return ValueId{};
            }
            ++phi_inputs_;
            phi.inputs.push_back({predecessor, incoming});
        }
        return output;
    }

    Operand rename_global(const Operand& operand, BlockId block, SourceLocation where) {
        std::uint32_t key;
        if (const auto* reg = std::get_if<Register>(&operand)) {
            if (auto literal = pseudo_read(*reg)) return *literal;
            key = key_of(*reg);
        } else if (const auto* id = std::get_if<ValueId>(&operand)) {
            if (id->value >= local_count_) return operand; // Already a renamed SSA value.
            key = id->value;
        } else {
            return operand; // Literals and constant-buffer words need no renaming.
        }
        if (!memo_[key].valid()) {
            memo_[key] = definition_for_use(key, block, where);
            if (error_) return operand;
            memo_touched_.push_back(key);
        }
        return Operand{memo_[key]};
    }

    void rename_terminator_global(BasicBlock& block, BlockId block_id) {
        if (auto* branch = std::get_if<Branch>(&block.terminator))
            branch->condition = rename_global(branch->condition, block_id, block.source);
        else if (auto* select = std::get_if<Switch>(&block.terminator))
            select->selector = rename_global(select->selector, block_id, block.source);
        else if (auto* result = std::get_if<Return>(&block.terminator))
            for (auto& value : result->values) value = rename_global(value, block_id, block.source);
    }

    void rename_block_global(BlockId block_id) {
        clear_memo();
        const auto& block = p_.blocks[block_id.value];
        for (const auto operation_id : block.operations) {
            auto& operation = p_.operations[operation_id.value];
            for (auto& input : operation.inputs) {
                input = rename_global(input, block_id, operation.source);
                if (error_) return;
            }
        }
        rename_terminator_global(p_.blocks[block_id.value], block_id);
    }

    // Retain every explicitly effectful operation and every value used by a
    // terminator. Reverse liveness then retains their pure dependencies. Pure
    // cycles with no retained consumer disappear, including cycles containing
    // provisional missing definitions. Finish liveness and pruning before
    // reporting missing definitions so fixed-CFG proof sees only retained uses.
    void remove_dead_code(bool zero_undefined = false) {
        struct Definition {
            OperationId operation;
            const Phi* phi{};
        };
        std::vector<Definition> definitions(p_.values.size());
        std::vector<bool> parameters(p_.values.size());
        std::vector<bool> live_values(p_.values.size());
        std::vector<bool> live_operations(p_.operations.size());
        std::vector<ValueId> pending;
        std::optional<Diagnostic> undefined;
        struct Zeroed { ValueId value; SourceLocation source; };
        std::vector<Zeroed> zeroed;

        auto mark_value = [&](ValueId value, SourceLocation where) {
            if (!charge(where)) return;
            if (!live_values[value.value]) {
                live_values[value.value] = true;
                pending.push_back(value);
            }
        };
        auto mark_operand = [&](const Operand& operand, SourceLocation where) {
            if (const auto* value = std::get_if<ValueId>(&operand)) mark_value(*value, where);
        };
        auto mark_operation = [&](OperationId id) {
            if (live_operations[id.value]) return;
            live_operations[id.value] = true;
            const auto& operation = p_.operations[id.value];
            for (const auto& input : operation.inputs) mark_operand(input, operation.source);
        };

        for (const auto& function : p_.functions)
            for (const auto parameter : function.parameters) parameters[parameter.value] = true;
        for (const auto& block : p_.blocks) {
            for (const auto& phi : block.phis) {
                if (!charge(phi.source)) return;
                definitions[phi.output.value].phi = &phi;
            }
            for (const auto id : block.operations) {
                if (!charge(p_.operations[id.value].source)) return;
                const auto& operation = p_.operations[id.value];
                for (const auto& output : operation.outputs)
                    definitions[std::get<ValueId>(output).value].operation = id;
                if (operation.effects != Effects::none) mark_operation(id);
            }
            if (const auto* branch = std::get_if<Branch>(&block.terminator))
                mark_operand(branch->condition, block.source);
            else if (const auto* select = std::get_if<Switch>(&block.terminator))
                mark_operand(select->selector, block.source);
            else if (const auto* result = std::get_if<Return>(&block.terminator))
                for (const auto& value : result->values) mark_operand(value, block.source);
            if (error_) return;
        }

        while (!pending.empty()) {
            const ValueId value = pending.back();
            pending.pop_back();
            if (!charge()) return;
            if (const auto unresolved = unresolved_.find(value.value); unresolved != unresolved_.end()) {
                if (zero_undefined) zeroed.push_back({value, unresolved->second.source});
                else if (!undefined) undefined = diagnose(Phase::construct_ssa, DiagnosticCode::undefined_read,
                              unresolved->second.source, unresolved->second.description);
                continue;
            }
            const auto& definition = definitions[value.value];
            if (definition.operation.valid()) {
                mark_operation(definition.operation);
            } else if (definition.phi) {
                for (const auto& input : definition.phi->inputs) mark_value(input.value, definition.phi->source);
            } else if (!parameters[value.value]) {
                if (!undefined) undefined = diagnose(Phase::construct_ssa, DiagnosticCode::undefined_read, {},
                              "Live SSA value has no reaching definition");
            }
            if (error_) return;
        }

        std::vector<OperationId> remap(p_.operations.size());
        std::vector<Operation> operations;
        operations.reserve(p_.operations.size());
        for (std::uint32_t index = 0; index < p_.operations.size(); ++index) {
            if (!charge(p_.operations[index].source)) return;
            if (!live_operations[index]) continue;
            remap[index] = OperationId{static_cast<std::uint32_t>(operations.size())};
            operations.push_back(std::move(p_.operations[index]));
        }
        for (auto& block : p_.blocks) {
            std::erase_if(block.phis, [&](const Phi& phi) {
                charge(phi.source);
                return !live_values[phi.output.value];
            });
            std::erase_if(block.operations, [&](OperationId id) {
                charge(p_.operations[id.value].source);
                return !live_operations[id.value];
            });
            if (error_) return;
            for (auto& id : block.operations) id = remap[id.value];
        }
        p_.operations = std::move(operations);
        if (undefined) fail(std::move(*undefined));
        if (error_ || zeroed.empty()) return;
        // Define each surviving undefined read as zero at the function entry,
        // which dominates every use. Calls are rejected, so one function exists.
        if (p_.functions.size() != 1) {
            fail(diagnose(Phase::construct_ssa, DiagnosticCode::undefined_read, zeroed.front().source,
                          "Undefined read outside a single-function program"));
            return;
        }
        std::vector<OperationId> definitions_at_entry;
        for (const auto& item : zeroed) {
            if (!charge(item.source)) return;
            if (p_.operations.size() >= limits_.operations) {
                fail(diagnose(Phase::construct_ssa, DiagnosticCode::limit_exceeded, item.source,
                              "Zero definitions for undefined reads exceed the operation limit"));
                return;
            }
            Operation zero;
            zero.source = item.source;
            zero.opcode = Opcode::copy;
            zero.inputs.push_back(LiteralBits{0, p_.values[item.value.value].type});
            zero.outputs.push_back(item.value);
            definitions_at_entry.push_back(OperationId{static_cast<std::uint32_t>(p_.operations.size())});
            p_.operations.push_back(std::move(zero));
            unresolved_.erase(item.value.value);
        }
        auto& entry_operations = p_.blocks[p_.functions.front().entry.value].operations;
        entry_operations.insert(entry_operations.begin(), definitions_at_entry.begin(),
                                definitions_at_entry.end());
    }

    Program& p_;
    Limits limits_;
    std::uint64_t& steps_;
    std::uint32_t local_count_{}, total_keys_{};
    std::uint64_t phi_inputs_{};
    BlockId entry_{};
    std::unordered_set<std::uint32_t> parameters_;
    std::vector<BlockDefs> block_defs_;
    std::vector<ValueId> local_defs_, memo_;
    std::vector<std::uint32_t> block_touched_, memo_touched_;
    std::vector<BlockId> queue_;
    Dominators dominators_;
    std::vector<std::vector<BlockId>> frontiers_;
    std::unordered_map<std::uint32_t, UnresolvedDefinition> unresolved_;
    std::optional<Diagnostic> error_;
};
} // namespace

Result<SsaProgram> construct_ssa(RegisterProgram program, Limits limits) {
    Program& ir = program.ir;
    std::uint64_t steps{};
    SsaBuilder builder(ir, std::move(limits), steps);
    if (auto status = builder.run(); std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    return SsaProgram{std::move(ir)};
}
} // namespace armigo::shader::pipeline
