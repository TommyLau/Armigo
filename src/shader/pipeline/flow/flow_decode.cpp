// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values, interned token stack states and checked
// contracts rather than managed block graphs and side tables.

#include "flow.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace armigo::shader::pipeline {
namespace flow {
namespace {

constexpr Limits tightened(Limits supplied) {
    const Limits defaults{};
    supplied.code_bytes = std::min(supplied.code_bytes, defaults.code_bytes);
    supplied.instructions = std::min(supplied.instructions, defaults.instructions);
    supplied.functions = std::min(supplied.functions, defaults.functions);
    supplied.blocks = std::min(supplied.blocks, defaults.blocks);
    supplied.edges = std::min(supplied.edges, defaults.edges);
    supplied.operations = std::min(supplied.operations, defaults.operations);
    supplied.values = std::min(supplied.values, defaults.values);
    supplied.phi_inputs = std::min(supplied.phi_inputs, defaults.phi_inputs);
    supplied.operands_per_operation = std::min(supplied.operands_per_operation, defaults.operands_per_operation);
    supplied.token_depth = std::min(supplied.token_depth, defaults.token_depth);
    supplied.token_states = std::min(supplied.token_states, defaults.token_states);
    supplied.regions = std::min(supplied.regions, defaults.regions);
    supplied.nesting_depth = std::min(supplied.nesting_depth, defaults.nesting_depth);
    supplied.resources = std::min(supplied.resources, defaults.resources);
    supplied.dependencies = std::min(supplied.dependencies, defaults.dependencies);
    supplied.source_bytes = std::min(supplied.source_bytes, defaults.source_bytes);
    supplied.analysis_steps = std::min(supplied.analysis_steps, defaults.analysis_steps);
    return supplied;
}

// A push or branch target discovered while filling a block. Targets are only
// turned into blocks after the filling block is complete, because a target may
// land inside it and require a split.
struct DeferredTarget {
    TokenId token{}; // Valid for pushes, invalid for branch targets.
    Pc instruction_pc{};
    Pc target_pc{};
};

// Per-function block discovery state. Functions decode independently, so the
// same address may host distinct blocks in distinct functions.
struct FunctionWork {
    std::map<Pc, BlockId> starts;
    std::vector<Pc> sorted;          // Ascending block start addresses.
    std::vector<BlockId> pending;    // Blocks awaiting fill/termination, FIFO.
    std::set<std::uint32_t> filled;  // Blocks whose instructions are complete.
    std::set<std::uint32_t> terminated;
    std::vector<DeferredTarget> deferred;
    std::size_t deferred_cursor{};
    std::map<Pc, BlockId> resolved_branch;  // Branch instruction PC -> target.
    std::size_t cursor{};
};

struct Decoder {
    const StageImage& image;
    Limits limits;
    std::span<const DecodeRule> rules;
    std::vector<DecodedFunction> functions;
    std::vector<FunctionWork> work;
    std::vector<DecodedBlock> blocks;
    std::vector<ReconvergenceToken> tokens;
    std::vector<TokenStackState> stacks;
    std::vector<SchedulingWord> scheduling;
    std::set<Pc> scheduled;
    std::uint32_t instructions{};
    std::uint64_t steps{};
    std::uint32_t edges{};
    std::size_t function_cursor{};

    [[nodiscard]] std::uint64_t load_word(Pc pc) const {
        std::uint64_t word = 0;
        for (unsigned index = 0; index < 8; ++index) {
            word |= std::uint64_t(std::to_integer<std::uint8_t>(image.bytes[pc + index])) << (8 * index);
        }
        return word;
    }

    [[nodiscard]] const DecodeRule* find_rule(std::uint64_t word) const {
        for (const auto& candidate : rules) {
            if ((word & candidate.mask) == candidate.value) return &candidate;
        }
        return nullptr;
    }

    [[nodiscard]] Result<FunctionId> enqueue_function(Pc pc) {
        for (std::size_t index = 0; index < functions.size(); ++index) {
            if (functions[index].entry_pc == pc) return FunctionId{static_cast<std::uint32_t>(index)};
        }
        if (functions.size() >= limits.functions) {
            return diagnose(Phase::decode, DiagnosticCode::limit_exceeded, {}, "Function count exceeds limit");
        }
        functions.push_back(DecodedFunction{pc, BlockId{}, {}, {}});
        work.push_back(FunctionWork{});
        return FunctionId{static_cast<std::uint32_t>(functions.size() - 1)};
    }

    [[nodiscard]] Pc fill_limit(FunctionId function, Pc begin) const {
        Pc limit = static_cast<Pc>(image.bytes.size());
        const auto& state = work[function.value];
        const auto position = std::upper_bound(state.sorted.begin(), state.sorted.end(), begin);
        if (position != state.sorted.end()) limit = std::min(limit, *position);
        return limit;
    }

    [[nodiscard]] Result<BlockId> create_block(FunctionId function, Pc pc) {
        if (blocks.size() >= limits.blocks) {
            return diagnose(Phase::decode, DiagnosticCode::limit_exceeded, {}, "Block count exceeds limit");
        }
        auto& state = work[function.value];
        blocks.push_back(DecodedBlock{function, pc, pc, {}, {}, {}, {}, {}, {}, {}});
        const auto id = BlockId{static_cast<std::uint32_t>(blocks.size() - 1)};
        functions[function.value].blocks.push_back(id);
        state.starts.emplace(pc, id);
        state.sorted.insert(std::upper_bound(state.sorted.begin(), state.sorted.end(), pc), pc);
        state.pending.push_back(id);
        return id;
    }

    // Splits an already filled block at pc, moving the tail instructions,
    // pushes and outgoing edges into a new block that begins at pc. The right
    // half needs no refill; it still needs termination if it was never
    // terminated, which the fill queue handles through the filled set.
    [[nodiscard]] Result<BlockId> split_block(FunctionId function, BlockId owner, Pc pc) {
        auto position = std::find_if(blocks[owner.value].instructions.begin(),
            blocks[owner.value].instructions.end(),
            [&](const DecodedInstruction& instruction) { return instruction.pc == pc; });
        if (position == blocks[owner.value].instructions.end()) {
            return diagnose(Phase::decode, DiagnosticCode::malformed_input, {pc, std::nullopt},
                "Branch target does not land on an instruction of the containing block");
        }
        const auto index = static_cast<std::size_t>(position - blocks[owner.value].instructions.begin());
        if (index == 0) {
            return diagnose(Phase::decode, DiagnosticCode::malformed_input, {pc, std::nullopt}, "Invalid block split position");
        }
        DecodedBlock right;
        right.function = function;
        right.begin_pc = pc;
        right.end_pc = blocks[owner.value].end_pc;
        auto& left_instructions = blocks[owner.value].instructions;
        right.instructions.assign(std::make_move_iterator(left_instructions.begin() +
                                                              static_cast<std::ptrdiff_t>(index)),
            std::make_move_iterator(left_instructions.end()));
        left_instructions.resize(index);
        right.successors = std::move(blocks[owner.value].successors);
        blocks[owner.value].successors.clear();
        blocks[owner.value].end_pc = pc;
        auto& owner_pushes = blocks[owner.value].pushes;
        const auto push_split = std::find_if(owner_pushes.begin(), owner_pushes.end(),
            [&](TokenId token) { return tokens[token.value].push_pc >= pc; });
        right.pushes.assign(push_split, owner_pushes.end());
        owner_pushes.erase(push_split, owner_pushes.end());
        if (blocks.size() >= limits.blocks) {
            return diagnose(Phase::decode, DiagnosticCode::limit_exceeded, {pc, std::nullopt}, "Block count exceeds limit");
        }
        blocks.push_back(std::move(right));
        const auto right_id = BlockId{static_cast<std::uint32_t>(blocks.size() - 1)};
        functions[function.value].blocks.push_back(right_id);
        auto& state = work[function.value];
        state.starts.emplace(pc, right_id);
        state.sorted.insert(std::upper_bound(state.sorted.begin(), state.sorted.end(), pc), pc);
        state.filled.insert(right_id.value);
        // A block split before its own termination has no successors to hand
        // over yet; queue the right half so its terminator still gets edges.
        if (!state.terminated.count(owner.value)) state.pending.push_back(right_id);
        // The right half inherits the filled instructions; only the
        // fallthrough edge to it is new.
        blocks[owner.value].successors.push_back(right_id);
        blocks[right_id.value].predecessors.push_back(owner);
        for (const auto successor : blocks[right_id.value].successors) {
            auto& predecessors = blocks[successor.value].predecessors;
            std::replace(predecessors.begin(), predecessors.end(), owner, right_id);
        }
        return right_id;
    }

    [[nodiscard]] Result<BlockId> get_block(FunctionId function, Pc pc) {
        // Fallthrough after a group's last instruction lands on the next
        // scheduling word; the block starts at the instruction after it.
        if (pc % 32 == 0) pc += 8;
        if (pc < 8 || pc % 8 != 0 ||
            pc >= image.bytes.size() || pc + 8 > image.bytes.size()) {
            return diagnose(Phase::decode, DiagnosticCode::malformed_input, {pc, std::nullopt},
                "Branch or push target is not a decodable instruction address");
        }
        auto& state = work[function.value];
        const auto known = state.starts.find(pc);
        if (known != state.starts.end()) return known->second;
        const auto position = std::upper_bound(state.sorted.begin(), state.sorted.end(), pc);
        if (position != state.sorted.begin()) {
            const auto start = *std::prev(position);
            const auto owner = state.starts.find(start);
            if (owner != state.starts.end() && pc < blocks[owner->second.value].end_pc) {
                return split_block(function, owner->second, pc);
            }
        }
        return create_block(function, pc);
    }

    Status attach_edge(BlockId from, BlockId to) {
        auto& successors = blocks[from.value].successors;
        if (std::find(successors.begin(), successors.end(), to) == successors.end()) {
            if (edges >= limits.edges) {
                auto error = diagnose(Phase::decode, DiagnosticCode::limit_exceeded,
                    SourceLocation{}, "CFG edge count exceeds limit");
                error.pc = blocks[from.value].begin_pc;
                return error;
            }
            ++edges;
            successors.push_back(to);
        }
        auto& predecessors = blocks[to.value].predecessors;
        if (std::find(predecessors.begin(), predecessors.end(), from) == predecessors.end()) {
            predecessors.push_back(from);
        }
        return std::monostate{};
    }

    // Decodes instructions until a terminator or the next block. Push and
    // branch targets are recorded as deferred work, never resolved inline.
    [[nodiscard]] Status fill(FunctionId function, BlockId block) {
        const auto limit = fill_limit(function, blocks[block.value].begin_pc);
        auto pc = blocks[block.value].begin_pc;
        while (pc + 8 <= limit) {
            if (pc % 32 == 0) {
                if (scheduled.insert(pc).second) {
                    scheduling.push_back(SchedulingWord{pc, load_word(pc)});
                }
                pc += 8;
                continue;
            }
            const auto word = load_word(pc);
            const auto* rule = find_rule(word);
            if (rule == nullptr) {
                return diagnose(Phase::decode, DiagnosticCode::unsupported_instruction,
                    {pc, std::nullopt}, "Instruction encoding matches no decode rule");
            }
            if (instructions >= limits.instructions) {
                return diagnose(Phase::decode, DiagnosticCode::limit_exceeded,
                    {pc, std::nullopt}, "Instruction count exceeds limit");
            }
            ++instructions;
            auto decoded = rule->decode(pc, RawInstruction{word});
            if (auto* error = std::get_if<Diagnostic>(&decoded)) return *error;
            auto& instruction = std::get<DecodedInstruction>(decoded);
            if (instruction.pc != pc || instruction.name == InstructionName::invalid) {
                return diagnose(Phase::decode, DiagnosticCode::malformed_input,
                    {pc, std::nullopt}, "Decode rule produced a mismatched instruction");
            }
            const auto flow = classify_flow(instruction.name);
            if (flow == FlowClass::unsupported) {
                return diagnose(Phase::decode, DiagnosticCode::unimplemented,
                    {pc, instruction.name}, "Dynamic or extended flow control is not implemented");
            }
            if (flow == FlowClass::push) {
                auto target = flow_target(instruction);
                if (auto* error = std::get_if<Diagnostic>(&target)) return *error;
                const auto token = TokenId{static_cast<std::uint32_t>(tokens.size())};
                tokens.push_back(ReconvergenceToken{push_token_kind(instruction.name), pc, BlockId{}, {}});
                blocks[block.value].pushes.push_back(token);
                work[function.value].deferred.push_back(DeferredTarget{token, pc, std::get<Pc>(target)});
            } else if (flow == FlowClass::branch) {
                auto target = flow_target(instruction);
                if (auto* error = std::get_if<Diagnostic>(&target)) return *error;
                work[function.value].deferred.push_back(
                    DeferredTarget{TokenId{}, pc, std::get<Pc>(target)});
            }
            blocks[block.value].instructions.push_back(std::move(instruction));
            pc += 8;
            if (flow == FlowClass::pop || flow == FlowClass::branch || flow == FlowClass::call ||
                flow == FlowClass::exit_like) {
                break;
            }
        }
        blocks[block.value].end_pc = pc;
        return std::monostate{};
    }

    // Builds the block's successor edges from its terminating instruction.
    // Runs after target resolution, so it only creates fallthrough blocks,
    // which can never land inside an existing block.
    [[nodiscard]] Status terminate_block(FunctionId function, BlockId block) {
        if (blocks[block.value].instructions.empty()) {
            return diagnose(Phase::decode, DiagnosticCode::malformed_input,
                {blocks[block.value].begin_pc, std::nullopt}, "Decoded block has no instructions");
        }
        const auto last = blocks[block.value].instructions.back();
        const auto flow = classify_flow(last.name);
        const auto end = blocks[block.value].end_pc;
        const auto code_end = static_cast<Pc>(image.bytes.size());
        // An invalid block id means the stream ends here with no fallthrough.
        const auto fallthrough = [&](void) -> Result<BlockId> {
            if (end >= code_end) return BlockId{};
            return get_block(function, end);
        };
        switch (flow) {
        case FlowClass::call: {
            auto target = flow_target(last);
            if (auto* error = std::get_if<Diagnostic>(&target)) return *error;
            auto callee = enqueue_function(std::get<Pc>(target));
            if (auto* error = std::get_if<Diagnostic>(&callee)) return *error;
            const auto callee_id = std::get<FunctionId>(callee);
            auto& callers = functions[callee_id.value].callers;
            if (std::find(callers.begin(), callers.end(), function) == callers.end()) {
                callers.push_back(function);
            }
            // Calls return to the next instruction, so the continuation stays
            // reachable in the caller's CFG.
            auto continuation = fallthrough();
            if (auto* error = std::get_if<Diagnostic>(&continuation)) return *error;
            if (const auto target = std::get<BlockId>(continuation); target.valid()) {
                const auto outcome = attach_edge(block, target);
                if (const auto* error = std::get_if<Diagnostic>(&outcome)) return *error;
            }
            break;
        }
        case FlowClass::pop: {
            if (!flow_unconditional(last)) {
                auto continuation = fallthrough();
                if (auto* error = std::get_if<Diagnostic>(&continuation)) return *error;
                if (const auto target = std::get<BlockId>(continuation); target.valid()) {
                    const auto outcome = attach_edge(block, target);
                    if (const auto* error = std::get_if<Diagnostic>(&outcome)) return *error;
                }
            }
            break;
        }
        case FlowClass::branch: {
            const auto& state = work[function.value];
            const auto resolved = state.resolved_branch.find(last.pc);
            if (resolved == state.resolved_branch.end()) {
                return diagnose(Phase::decode, DiagnosticCode::invalid_ir,
                    {last.pc, last.name}, "Branch target was not resolved");
            }
            if (!flow_unconditional(last)) {
                auto continuation = fallthrough();
                if (auto* error = std::get_if<Diagnostic>(&continuation)) return *error;
                if (const auto target = std::get<BlockId>(continuation); target.valid()) {
                    const auto outcome = attach_edge(block, target);
                    if (const auto* error = std::get_if<Diagnostic>(&outcome)) return *error;
                }
            }
            if (const auto outcome = attach_edge(block, resolved->second); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
            break;
        }
        case FlowClass::exit_like: {
            // KIL is an effect rather than a control transfer, so execution
            // always continues; EXIT/RET fall through only when predicated.
            if (last.name == InstructionName::kil || !flow_unconditional(last)) {
                auto continuation = fallthrough();
                if (auto* error = std::get_if<Diagnostic>(&continuation)) return *error;
                if (const auto target = std::get<BlockId>(continuation); target.valid()) {
                    const auto outcome = attach_edge(block, target);
                    if (const auto* error = std::get_if<Diagnostic>(&outcome)) return *error;
                }
            }
            break;
        }
        // A push ends a block when the next instruction is a branch target; it
        // falls through like any other non-transferring instruction.
        case FlowClass::plain:
        case FlowClass::push: {
            auto continuation = fallthrough();
            if (auto* error = std::get_if<Diagnostic>(&continuation)) return *error;
            if (const auto target = std::get<BlockId>(continuation); target.valid()) {
                const auto outcome = attach_edge(block, target);
                if (const auto* error = std::get_if<Diagnostic>(&outcome)) return *error;
            }
            break;
        }
        case FlowClass::unsupported:
            return diagnose(Phase::decode, DiagnosticCode::malformed_input,
                {last.pc, last.name}, "Instruction cannot terminate a decoded block");
        }
        return std::monostate{};
    }

    [[nodiscard]] Status decode_function(FunctionId function) {
        auto entry = get_block(function, functions[function.value].entry_pc);
        if (auto* error = std::get_if<Diagnostic>(&entry)) return *error;
        functions[function.value].entry = std::get<BlockId>(entry);
        while (true) {
            BlockId block{};
            {
                auto& state = work[function.value];
                if (state.cursor >= state.pending.size()) break;
                block = state.pending[state.cursor++];
            }
            if (!work[function.value].filled.count(block.value)) {
                if (const auto outcome = fill(function, block); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
                work[function.value].filled.insert(block.value);
            }
            // Resolve deferred targets; this may split filled blocks.
            {
                auto& state = work[function.value];
                while (state.deferred_cursor < state.deferred.size()) {
                    const auto item = state.deferred[state.deferred_cursor++];
                    auto target = get_block(function, item.target_pc);
                    if (auto* error = std::get_if<Diagnostic>(&target)) return *error;
                    if (item.token.valid()) {
                        tokens[item.token.value].target = std::get<BlockId>(target);
                    } else {
                        state.resolved_branch[item.instruction_pc] = std::get<BlockId>(target);
                    }
                }
            }
            if (const auto outcome = terminate_block(function, block); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
            work[function.value].terminated.insert(block.value);
        }
        return std::monostate{};
    }

    [[nodiscard]] Result<StackStateId> intern_stack(std::vector<TokenId> stack,
                                                   std::map<std::vector<TokenId>, StackStateId>& interned) {
        const auto known = interned.find(stack);
        if (known != interned.end()) return known->second;
        if (stacks.size() >= limits.token_states) {
            return diagnose(Phase::decode, DiagnosticCode::limit_exceeded, {}, "Token stack state count exceeds limit");
        }
        const auto id = StackStateId{static_cast<std::uint32_t>(stacks.size())};
        stacks.push_back(TokenStackState{stack});
        interned.emplace(std::move(stack), id);
        return id;
    }

    // Path-sensitive token propagation: every (block, incoming stack) pair is
    // processed once; pushes extend the stack and pops consume the topmost
    // token of their kind, unwinding everything above it.
    [[nodiscard]] Status propagate(FunctionId function) {
        std::map<std::vector<TokenId>, StackStateId> interned;
        // Discovery-time successors of pop blocks are exactly their
        // conditional fallthroughs; consumed targets join the CFG later.
        std::vector<std::vector<BlockId>> pop_fallthroughs(blocks.size());
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            const auto& candidate = blocks[index];
            if (!candidate.instructions.empty() &&
                classify_flow(candidate.instructions.back().name) == FlowClass::pop) {
                pop_fallthroughs[index] = candidate.successors;
            }
        }
        auto empty = intern_stack({}, interned);
        if (auto* error = std::get_if<Diagnostic>(&empty)) return *error;
        std::vector<std::pair<BlockId, StackStateId>> pending;
        pending.emplace_back(functions[function.value].entry, std::get<StackStateId>(empty));
        std::set<std::pair<std::uint32_t, std::uint32_t>> visited;
        while (!pending.empty()) {
            if (++steps > limits.analysis_steps) {
                return diagnose(Phase::decode, DiagnosticCode::limit_exceeded, {},
                    "Decoding analysis work exceeds limit");
            }
            const auto [block_id, state_id] = pending.back();
            pending.pop_back();
            if (!visited.emplace(block_id.value, state_id.value).second) continue;
            auto& block = blocks[block_id.value];
            block.incoming_stacks.push_back(state_id);
            auto stack = stacks[state_id.value].stack;
            for (const auto token : block.pushes) {
                stack.push_back(token);
                if (stack.size() > limits.token_depth) {
                    return diagnose(Phase::decode, DiagnosticCode::limit_exceeded,
                        {block.begin_pc, std::nullopt}, "Token stack depth exceeds limit");
                }
            }
            auto final_state = intern_stack(stack, interned);
            if (auto* error = std::get_if<Diagnostic>(&final_state)) return *error;
            if (block.instructions.empty()) {
                return diagnose(Phase::decode, DiagnosticCode::malformed_input,
                    {block.begin_pc, std::nullopt}, "Decoded block has no instructions");
            }
            const auto& last = block.instructions.back();
            if (classify_flow(last.name) != FlowClass::pop) {
                for (const auto successor : block.successors) {
                    block.transitions.push_back(
                        TokenTransition{std::get<StackStateId>(final_state), std::get<StackStateId>(final_state),
                                        successor, std::nullopt});
                    if (edges >= limits.edges) {
                        return diagnose(Phase::decode, DiagnosticCode::limit_exceeded,
                            {block.begin_pc, std::nullopt}, "CFG edge count exceeds limit");
                    }
                    ++edges;
                    pending.emplace_back(successor, std::get<StackStateId>(final_state));
                }
                continue;
            }
            const auto kind = pop_token_kind(last.name);
            bool found = false;
            auto matched = TokenId{};
            std::size_t matched_index = 0;
            for (auto index = stack.size(); index-- > 0;) {
                if (tokens[stack[index].value].kind == kind) {
                    found = true;
                    matched = stack[index];
                    matched_index = index;
                    break;
                }
            }
            if (!found) {
                return diagnose(Phase::decode, DiagnosticCode::malformed_input,
                    {last.pc, last.name}, "Pop instruction has no matching push token on the stack");
            }
            auto& token = tokens[matched.value];
            if (std::find(token.consumers.begin(), token.consumers.end(), block_id) == token.consumers.end()) {
                token.consumers.push_back(block_id);
            }
            stack.resize(matched_index);
            auto outgoing = intern_stack(stack, interned);
            if (auto* error = std::get_if<Diagnostic>(&outgoing)) return *error;
            for (const auto& existing : block.reconvergence) {
                if (existing.pop_pc == last.pc && existing.token == matched) {
                    return diagnose(Phase::decode, DiagnosticCode::malformed_input,
                        {last.pc, last.name}, "Pop instruction matches one token through ambiguous stacks");
                }
            }
            const auto selector = static_cast<std::uint32_t>(std::count_if(
                block.reconvergence.begin(), block.reconvergence.end(),
                [&](const ReconvergenceTarget& target) { return target.pop_pc == last.pc; }));
            block.reconvergence.push_back(
                ReconvergenceTarget{last.pc, matched, token.target, std::get<StackStateId>(final_state), selector});
            block.transitions.push_back(
                TokenTransition{std::get<StackStateId>(final_state), std::get<StackStateId>(outgoing),
                                token.target, matched});
            if (edges >= limits.edges) {
                return diagnose(Phase::decode, DiagnosticCode::limit_exceeded,
                    {block.begin_pc, std::nullopt}, "CFG edge count exceeds limit");
            }
            ++edges;
            if (const auto outcome = attach_edge(block_id, token.target); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
            pending.emplace_back(token.target, std::get<StackStateId>(outgoing));
            // Conditional pops keep the unconsumed stack on their fallthrough.
            for (const auto successor : pop_fallthroughs[block_id.value]) {
                block.transitions.push_back(
                    TokenTransition{std::get<StackStateId>(final_state), std::get<StackStateId>(final_state),
                                    successor, std::nullopt});
                if (edges >= limits.edges) {
                    return diagnose(Phase::decode, DiagnosticCode::limit_exceeded,
                        {block.begin_pc, std::nullopt}, "CFG edge count exceeds limit");
                }
                ++edges;
                pending.emplace_back(successor, std::get<StackStateId>(final_state));
            }
        }
        return std::monostate{};
    }
};
} // namespace

FlowClass classify_flow(InstructionName name) {
    switch (name) {
    case InstructionName::ssy:
    case InstructionName::pbk:
    case InstructionName::pcnt:
        return FlowClass::push;
    case InstructionName::sync:
    case InstructionName::brk:
    case InstructionName::cont:
        return FlowClass::pop;
    case InstructionName::bra:
        return FlowClass::branch;
    case InstructionName::cal:
        return FlowClass::call;
    case InstructionName::exit:
    case InstructionName::kil:
    case InstructionName::ret:
        return FlowClass::exit_like;
    case InstructionName::brx:
    case InstructionName::jmp:
    case InstructionName::jcal:
    case InstructionName::longjmp:
    case InstructionName::plongjmp:
    case InstructionName::pret:
    case InstructionName::pexit:
        return FlowClass::unsupported;
    default:
        return FlowClass::plain;
    }
}

TokenKind push_token_kind(InstructionName name) {
    switch (name) {
    case InstructionName::pbk: return TokenKind::break_target;
    case InstructionName::pcnt: return TokenKind::continue_target;
    default: return TokenKind::sync;
    }
}

TokenKind pop_token_kind(InstructionName name) {
    switch (name) {
    case InstructionName::brk: return TokenKind::break_target;
    case InstructionName::cont: return TokenKind::continue_target;
    default: return TokenKind::sync;
    }
}

bool flow_unconditional(const DecodedInstruction& instruction) {
    if (!instruction.predicate.enabled) return true;
    if (instruction.predicate.index != 7 || instruction.predicate.inverted) return false;
    for (const auto& field : instruction.fields) {
        if (field.field == DecodedField::condition) return field.bits == 15; // Ccc T.
    }
    return true;
}

Result<Pc> flow_target(const DecodedInstruction& instruction) {
    auto bits = decoded_field(instruction, DecodedField::branch_displacement);
    if (auto* error = std::get_if<Diagnostic>(&bits)) return *error;
    const auto displacement = static_cast<std::int64_t>(std::get<std::uint64_t>(bits));
    auto target = static_cast<std::int64_t>(instruction.pc) + displacement;
    // A target on a scheduling word starts executing at the instruction that
    // follows it, so the block decoder skips the scheduling word.
    if (target % 32 == 0) target += 8;
    if (target < 8 || target % 8 != 0 ||
        target > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        return diagnose(Phase::decode, DiagnosticCode::malformed_input,
            {instruction.pc, instruction.name}, "Branch target is not a valid instruction address");
    }
    return static_cast<Pc>(target);
}

Result<std::vector<DecodeRule>> merge_decode_rules(
    std::span<const DecodeRule> first, std::span<const DecodeRule> second) {
    std::vector<DecodeRule> merged;
    merged.reserve(first.size() + second.size());
    for (const auto table : {first, second}) {
        for (const auto& rule : table) {
            if (rule.decode == nullptr) {
                return diagnose(Phase::decode, DiagnosticCode::malformed_input, {},
                    "Decode rule has no decode callback");
            }
            if (rule.mask == 0) {
                return diagnose(Phase::decode, DiagnosticCode::malformed_input, {},
                    "Decode rule matches every instruction encoding");
            }
            for (const auto& other : merged) {
                if (((rule.value ^ other.value) & rule.mask & other.mask) == 0) {
                    return diagnose(Phase::decode, DiagnosticCode::malformed_input, {},
                        "Decode rules overlap and match the same encoding");
                }
            }
            merged.push_back(rule);
        }
    }
    return merged;
}

Status validate_decoded_cfg(const DecodedProgram& program) {
    const auto invalid = [&](std::string message, SourceLocation location = {}) {
        return diagnose(Phase::decode, DiagnosticCode::invalid_ir, location, std::move(message));
    };
    if (program.entry.value >= program.functions.size()) return invalid("Entry function is missing");
    for (std::size_t index = 0; index < program.functions.size(); ++index) {
        const auto& function = program.functions[index];
        if (function.entry.value >= program.blocks.size()) return invalid("Function entry block is missing");
        const auto& owned = program.functions[index].blocks;
        if (std::find(owned.begin(), owned.end(), function.entry) == owned.end()) {
            return invalid("Function entry is not one of its blocks");
        }
        for (const auto block : owned) {
            if (block.value >= program.blocks.size() ||
                program.blocks[block.value].function.value != index) {
                return invalid("Function block list disagrees with block ownership");
            }
        }
    }
    for (std::size_t index = 0; index < program.blocks.size(); ++index) {
        const auto& block = program.blocks[index];
        if (block.function.value >= program.functions.size()) return invalid("Block function is missing");
        if (block.instructions.empty()) {
            return invalid("Decoded block has no instructions", {block.begin_pc, std::nullopt});
        }
        if (block.begin_pc > block.instructions.front().pc ||
            block.end_pc <= block.instructions.back().pc) {
            return invalid("Block range does not contain its instructions", {block.begin_pc, std::nullopt});
        }
        for (std::size_t position = 1; position < block.instructions.size(); ++position) {
            if (block.instructions[position - 1].pc >= block.instructions[position].pc) {
                return invalid("Block instructions are not ordered by PC", {block.begin_pc, std::nullopt});
            }
        }
        for (const auto predecessor : block.predecessors) {
            if (predecessor.value >= program.blocks.size()) return invalid("Predecessor block is missing");
            const auto& successors = program.blocks[predecessor.value].successors;
            const auto self = BlockId{static_cast<std::uint32_t>(index)};
            if (std::find(successors.begin(), successors.end(), self) == successors.end()) {
                return invalid("Predecessor edge has no matching successor edge", {block.begin_pc, std::nullopt});
            }
        }
        if (std::adjacent_find(block.predecessors.begin(), block.predecessors.end()) != block.predecessors.end() ||
            std::adjacent_find(block.successors.begin(), block.successors.end()) != block.successors.end()) {
            return invalid("Block edge lists contain duplicates", {block.begin_pc, std::nullopt});
        }
        for (const auto successor : block.successors) {
            if (successor.value >= program.blocks.size()) return invalid("Successor block is missing");
            const auto& predecessors = program.blocks[successor.value].predecessors;
            const auto self = BlockId{static_cast<std::uint32_t>(index)};
            if (std::find(predecessors.begin(), predecessors.end(), self) == predecessors.end()) {
                return invalid("Successor edge has no matching predecessor edge", {block.begin_pc, std::nullopt});
            }
        }
        for (const auto state : block.incoming_stacks) {
            if (state.value >= program.stacks.size()) return invalid("Incoming stack state is missing");
        }
        if (std::adjacent_find(block.incoming_stacks.begin(), block.incoming_stacks.end()) !=
            block.incoming_stacks.end()) {
            return invalid("Block lists one incoming stack state twice", {block.begin_pc, std::nullopt});
        }
        for (const auto token : block.pushes) {
            if (token.value >= program.tokens.size()) return invalid("Pushed token is missing");
            const auto& pushed = program.tokens[token.value];
            if (pushed.target.value >= program.blocks.size()) return invalid("Token target block is missing");
            const auto instruction = std::find_if(block.instructions.begin(), block.instructions.end(),
                [&](const DecodedInstruction& item) { return item.pc == pushed.push_pc; });
            if (instruction == block.instructions.end() || classify_flow(instruction->name) != FlowClass::push) {
                return invalid("Pushed token has no push instruction in the block", {pushed.push_pc, std::nullopt});
            }
        }
        for (const auto& transition : block.transitions) {
            if (transition.incoming.value >= program.stacks.size() ||
                transition.outgoing.value >= program.stacks.size()) {
                return invalid("Transition references an unknown stack state", {block.begin_pc, std::nullopt});
            }
            if (transition.successor.value >= program.blocks.size()) return invalid("Transition target is missing");
            if (std::find(block.successors.begin(), block.successors.end(), transition.successor) ==
                block.successors.end()) {
                return invalid("Transition has no matching successor edge", {block.begin_pc, std::nullopt});
            }
            if (transition.consumed && transition.consumed->value >= program.tokens.size()) {
                return invalid("Transition consumes an unknown token", {block.begin_pc, std::nullopt});
            }
        }
        for (const auto successor : block.successors) {
            const auto covered = std::any_of(block.transitions.begin(), block.transitions.end(),
                [&](const TokenTransition& transition) { return transition.successor == successor; });
            if (!covered) return invalid("Successor edge has no outgoing transition", {block.begin_pc, std::nullopt});
        }
        if (!block.reconvergence.empty()) {
            const auto& last = block.instructions.back();
            if (classify_flow(last.name) != FlowClass::pop) {
                return invalid("Reconvergence targets exist without a pop instruction", {block.begin_pc, std::nullopt});
            }
            const auto expected_kind = pop_token_kind(last.name);
            for (const auto& target : block.reconvergence) {
                if (target.token.value >= program.tokens.size()) return invalid("Reconvergence token is missing");
                if (program.tokens[target.token.value].kind != expected_kind) {
                    return invalid("Pop instruction consumed a token of the wrong kind", {target.pop_pc, std::nullopt});
                }
                if (target.target.value >= program.blocks.size()) return invalid("Reconvergence target is missing");
                if (target.incoming.value >= program.stacks.size()) {
                    return invalid("Reconvergence incoming state is missing");
                }
                if (target.pop_pc != last.pc) {
                    return invalid("Reconvergence target does not match the pop instruction", {target.pop_pc, std::nullopt});
                }
                const auto consumed = std::any_of(block.transitions.begin(), block.transitions.end(),
                    [&](const TokenTransition& transition) {
                        return transition.consumed == target.token && transition.successor == target.target;
                    });
                if (!consumed) return invalid("Reconvergence target has no consumed transition", {target.pop_pc, std::nullopt});
            }
            for (std::size_t a = 0; a < block.reconvergence.size(); ++a) {
                for (std::size_t b = a + 1; b < block.reconvergence.size(); ++b) {
                    const auto& first = block.reconvergence[a];
                    const auto& second = block.reconvergence[b];
                    if (first.token == second.token) {
                        return invalid("Ambiguous pop: one token is consumed twice", {last.pc, last.name});
                    }
                    if (first.selector == second.selector) {
                        return invalid("Ambiguous pop: two alternatives share one selector", {last.pc, last.name});
                    }
                }
            }
        }
    }
    for (std::size_t index = 0; index < program.tokens.size(); ++index) {
        const auto& token = program.tokens[index];
        for (const auto consumer : token.consumers) {
            if (consumer.value >= program.blocks.size()) return invalid("Token consumer block is missing");
            const auto recorded = std::any_of(program.blocks[consumer.value].reconvergence.begin(),
                program.blocks[consumer.value].reconvergence.end(),
                [&](const ReconvergenceTarget& target) {
                    return target.token == TokenId{static_cast<std::uint32_t>(index)};
                });
            if (!recorded) return invalid("Token consumer has no reconvergence record", {token.push_pc, std::nullopt});
        }
    }
    for (const auto& state : program.stacks) {
        for (const auto token : state.stack) {
            if (token.value >= program.tokens.size()) return invalid("Stack state holds an unknown token");
        }
    }
    return std::monostate{};
}

Result<DecodedProgram> decode_stream(
    const StageImage& image, FactReader& facts, Limits limits, std::span<const DecodeRule> rules) {
    const auto effective = tightened(limits);
    if (image.bytes.size() > effective.code_bytes) {
        return diagnose(Phase::decode, DiagnosticCode::limit_exceeded, {},
            "Instruction stream exceeds the code byte limit");
    }
    if (image.entry_pc % 8 != 0 || image.entry_pc % 32 == 0 ||
        image.entry_pc >= image.bytes.size() || image.entry_pc + 8 > image.bytes.size()) {
        return diagnose(Phase::decode, DiagnosticCode::malformed_input, {},
            "Entry PC is not a decodable instruction address");
    }
    Decoder decoder{.image = image, .limits = effective, .rules = rules};
    auto entry_function = decoder.enqueue_function(image.entry_pc);
    if (auto* error = std::get_if<Diagnostic>(&entry_function)) return *error;
    while (decoder.function_cursor < decoder.functions.size()) {
        const auto function = FunctionId{static_cast<std::uint32_t>(decoder.function_cursor++)};
        if (const auto outcome = decoder.decode_function(function); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    }
    for (std::size_t index = 0; index < decoder.functions.size(); ++index) {
        const auto outcome =
            decoder.propagate(FunctionId{static_cast<std::uint32_t>(index)});
        if (const auto* error = std::get_if<Diagnostic>(&outcome)) return *error;
    }
    for (std::size_t index = 0; index < decoder.blocks.size(); ++index) {
        if (decoder.blocks[index].incoming_stacks.empty()) {
            return diagnose(Phase::decode, DiagnosticCode::malformed_input,
                {decoder.blocks[index].begin_pc, std::nullopt},
                "Decoded block is unreachable from its function entry");
        }
    }
    DecodedProgram program;
    program.stage = image.stage;
    program.sph = image.sph;
    program.code_bytes = static_cast<Pc>(image.bytes.size());
    program.entry = FunctionId{0};
    program.limits = effective;
    program.functions = std::move(decoder.functions);
    program.blocks = std::move(decoder.blocks);
    program.tokens = std::move(decoder.tokens);
    program.stacks = std::move(decoder.stacks);
    program.scheduling = std::move(decoder.scheduling);
    if (const auto outcome = validate_decoded_cfg(program); std::get_if<Diagnostic>(&outcome)) return std::get<Diagnostic>(outcome);
    auto dependencies = merge_dependencies(facts.dependencies(), {}, effective, Phase::decode);
    if (auto* error = std::get_if<Diagnostic>(&dependencies)) return *error;
    program.dependencies = std::move(std::get<std::vector<Dependency>>(dependencies));
    return program;
}
} // namespace flow

Result<DecodedProgram> decode(const StageImage& image, FactReader& facts, Limits limits) {
    auto merged = flow::merge_decode_rules(alu_decode_rules(), resource_decode_rules());
    if (auto* error = std::get_if<Diagnostic>(&merged)) return *error;
    return flow::decode_stream(image, facts, limits, std::get<std::vector<DecodeRule>>(merged));
}
} // namespace armigo::shader::pipeline
