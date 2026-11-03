// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "stage_internal.h"
#include "custom_border.h"
#include "../dispatch.h"
#include "../../graphics_operations.h"
#include <armigo/shader/rne_multiply.h>
#include <armigo/shader/rtz_multiply.h>
#include <armigo/shader/fma.h>
#include <armigo/shader/reciprocal.h>
#include <armigo/shader/saturate.h>
#include <armigo/shader/vertex_fetch.h>
#include <cctype>
#include <iterator>
#include <map>
#include <tuple>

namespace armigo::shader::pipeline {
namespace {
using namespace msl;
using Definitions = std::set<std::uint32_t>;
using OutputComponents = std::set<std::uint64_t>;
std::uint64_t output_key(IoSemantic semantic, std::uint16_t location, unsigned component) {
    return (std::uint64_t(semantic) << 32) | (std::uint64_t(location) << 8) | component;
}
struct Flow {
    bool reachable{true};
    Definitions defined;
    OutputComponents outputs;
    // Only paths that have not lost predicate correlations may witness a write.
    bool exact{true};
};
bool reserved_entry(const std::string& name) {
    static const std::set<std::string> keywords{
        "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept",
        "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
        "class", "compl", "concept", "const", "consteval", "constexpr", "constinit", "const_cast", "continue",
        "co_await", "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
        "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if",
        "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
        "operator", "or", "or_eq", "private", "protected", "public", "register", "reinterpret_cast",
        "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
        "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
        "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
        "vertex", "fragment", "kernel", "device", "constant", "thread", "threadgroup", "threadgroup_imageblock",
        "ray_data", "object_data", "metal", "sampler", "array", "as_type", "ArmigoInput", "ArmigoOutput", "ArmigoStageResources"
    };
    if (keywords.contains(name) || name.starts_with('_') || name.find("__") != std::string::npos ||
        name.starts_with("texture") || name.starts_with("depth") || name.starts_with("atomic_")) return true;
    for (std::string_view base : {"half", "float", "int", "uint", "short", "ushort", "char", "uchar", "long", "ulong", "bool"}) {
        if (name == base) return true;
        if (name.starts_with(base)) {
            auto suffix = std::string_view(name).substr(base.size());
            if (suffix.size() == 1 && suffix[0] >= '2' && suffix[0] <= '4') return true;
            if (suffix.size() == 3 && suffix[0] >= '2' && suffix[0] <= '4' && suffix[1] == 'x' && suffix[2] >= '2' && suffix[2] <= '4') return true;
        }
    }
    return false;
}

Flow merge(const std::vector<Flow>& flows) {
    Flow result{false, {}, {}};
    for (const auto& flow : flows) if (flow.reachable) {
        if (!result.reachable) result = flow;
        else {
            Definitions common;
            std::set_intersection(result.defined.begin(), result.defined.end(), flow.defined.begin(), flow.defined.end(), std::inserter(common, common.end()));
            result.defined = std::move(common);
            OutputComponents common_outputs;
            std::set_intersection(result.outputs.begin(), result.outputs.end(), flow.outputs.begin(), flow.outputs.end(), std::inserter(common_outputs, common_outputs.end()));
            result.outputs = std::move(common_outputs);
        }
    }
    return result;
}

// Dispatch guards introduced by structurization are correlated. Analyze the
// region graph, not the retained CFG. Coalesce equivalent must-facts exactly
// when possible, then widen after a fixed number of predicate partitions.
// Loop backedges participate in the same fixed point as entry edges; an initial
// flag value cannot hide a later read.
class Definedness {
    struct Boolean {
        // Constants use no atom. Other values equal an atom or its complement.
        std::uint64_t atom{UINT64_MAX};
        bool inverted{};
        auto operator<=>(const Boolean&) const = default;
    };
    using Environment = std::vector<std::pair<std::uint32_t, Boolean>>;
    struct Node {
        OperationId operation;
        std::optional<Operand> condition;
        std::vector<std::size_t> next, predecessors;
        Definitions live, predicates;
        std::map<Environment, Flow> incoming;
        std::size_t precise_updates{};
        bool queued{}, widened{};
    };
    const StructuredProgram& p_;
    const ResourcePlan& plan_;
    std::uint64_t& steps_;
    std::vector<Node> nodes_;
    std::vector<std::size_t> region_nodes_, operation_nodes_, rank_;
    std::map<RegionId, std::pair<std::size_t, std::size_t>> loops_;
    std::set<std::uint32_t> blocks_;
    OutputComponents written_;
    static constexpr std::size_t absent = SIZE_MAX;
    // Bound both storage and partition changes, including loop iterations.
    // Counting only simultaneous environments would allow repeated coalescing
    // and re-splitting to evade the bound.
    static constexpr std::size_t environment_limit = 32;

    void step(std::uint64_t count = 1) {
        require(count <= p_.ir.limits.analysis_steps - steps_, "MSL analysis budget exceeded", DiagnosticCode::limit_exceeded);
        steps_ += count;
    }
    std::size_t node() { step(); nodes_.emplace_back(); return nodes_.size() - 1; }
    static bool boolean_expression(Opcode opcode) {
        return opcode == Opcode::copy || opcode == Opcode::logical_not || opcode == Opcode::logical_and ||
            opcode == Opcode::logical_or || opcode == Opcode::logical_xor || opcode == Opcode::select;
    }
    static std::optional<Boolean> boolean(const Operand& value, const Environment& env) {
        if (const auto* literal = std::get_if<LiteralBits>(&value)) {
            if (literal->type == ValueType::predicate) return Boolean{UINT64_MAX, literal->bits != 0};
        } else if (const auto* id = std::get_if<ValueId>(&value)) {
            const auto found = std::lower_bound(env.begin(), env.end(), id->value,
                [](const auto& item, auto key) { return item.first < key; });
            if (found != env.end() && found->first == id->value) return found->second;
        }
        return {};
    }
    static void set(Environment& env, std::uint32_t id, Boolean value) {
        auto found = std::lower_bound(env.begin(), env.end(), id,
            [](const auto& item, auto key) { return item.first < key; });
        if (found != env.end() && found->first == id) found->second = value;
        else env.insert(found, {id, value});
    }
    static void canonicalize(Environment& env) {
        // Names depend on live aliases rather than the number of loop trips.
        std::map<std::uint64_t, std::pair<std::uint32_t, bool>> atoms;
        for (auto& [id, value] : env) if (value.atom != UINT64_MAX) {
            const auto representative = atoms.try_emplace(value.atom, id, value.inverted).first->second;
            value.atom = representative.first;
            value.inverted ^= representative.second;
        }
    }
    Environment join(const Environment& a, const Environment& b) {
        step(a.size() + b.size() + 1);
        Environment result;
        // Two values remain aliases only if both environments agree on their
        // relative parity. The pair of atom names and parity is a class key.
        std::map<std::tuple<std::uint64_t, std::uint64_t, bool>, std::pair<std::uint32_t, bool>> classes;
        auto left = a.begin(), right = b.begin();
        while (left != a.end() && right != b.end()) {
            if (left->first < right->first) { ++left; continue; }
            if (right->first < left->first) { ++right; continue; }
            const auto [id, x] = *left++;
            const auto y = (right++)->second;
            if (x.atom == UINT64_MAX && y.atom == UINT64_MAX && x.inverted == y.inverted) {
                result.emplace_back(id, x);
                continue;
            }
            const auto key = std::tuple{x.atom, y.atom, bool(x.inverted ^ y.inverted)};
            const auto representative = classes.try_emplace(key, id, x.inverted).first->second;
            result.emplace_back(id, Boolean{representative.first, bool(x.inverted ^ representative.second)});
        }
        return result;
    }
    std::optional<Environment> coalesce(const Environment& a, const Environment& b) {
        step(a.size() + b.size() + 1);
        if (a.size() != b.size()) return {};
        auto joined = a;
        std::optional<std::pair<std::uint32_t, bool>> representative;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i].first != b[i].first) return {};
            if (a[i].second == b[i].second) continue;
            const auto x = a[i].second, y = b[i].second;
            if (x.atom != UINT64_MAX || y.atom != UINT64_MAX) return {};
            if (!representative) representative = {a[i].first, x.inverted};
            joined[i].second = {(1ull << 32) + representative->first,
                                bool(x.inverted ^ representative->second)};
        }
        canonicalize(joined);
        return joined;
    }
    template<class T> bool intersect(std::set<T>& destination, const std::set<T>& source) {
        step(destination.size());
        const auto size = destination.size();
        std::erase_if(destination, [&](auto id) { return !source.contains(id); });
        return size != destination.size();
    }
    std::size_t build(RegionId id, std::size_t next, unsigned depth) {
        step();
        require(depth < p_.ir.limits.nesting_depth, "Structured nesting limit exceeded", DiagnosticCode::limit_exceeded);
        require(id.value < p_.regions.size() && region_nodes_[id.value] == absent,
                "Invalid, cyclic or shared structured region");
        const auto index = node();
        region_nodes_[id.value] = index;
        auto entry = next;
        std::visit([&](const auto& r) {
            using T = std::decay_t<decltype(r)>;
            if constexpr (std::is_same_v<T, SequenceRegion>) {
                for (auto it = r.children.rbegin(); it != r.children.rend(); ++it)
                    entry = build(*it, entry, depth + 1);
            } else if constexpr (std::is_same_v<T, BlockRegion>) {
                require(r.block.value < p_.ir.blocks.size() && blocks_.insert(r.block.value).second,
                        "Invalid or repeated structured block");
                const auto& operations = p_.ir.blocks[r.block.value].operations;
                for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
                    require(it->value < operation_nodes_.size() && operation_nodes_[it->value] == absent,
                            "Invalid or repeated operation");
                    for (const auto& output : p_.ir.operations[it->value].outputs)
                        require(std::holds_alternative<ValueId>(output), "Output must be a value ID");
                    const auto operation = node();
                    operation_nodes_[it->value] = operation;
                    nodes_[operation].operation = *it;
                    nodes_[operation].next = {entry};
                    entry = operation;
                }
            } else if constexpr (std::is_same_v<T, IfRegion>) {
                const auto yes = build(r.when_true, next, depth + 1);
                const auto no = r.when_false.valid() ? build(r.when_false, next, depth + 1) : next;
                nodes_[index].condition = r.condition;
                nodes_[index].next = {yes, no};
            } else if constexpr (std::is_same_v<T, LoopRegion>) {
                const auto continuing = node();
                loops_[id] = {next, index};
                const auto tail = r.continuing.valid() ? build(r.continuing, index, depth + 1) : index;
                nodes_[continuing].next = {tail};
                loops_[id].second = continuing;
                entry = build(r.body, continuing, depth + 1);
                loops_.erase(id);
            } else if constexpr (std::is_same_v<T, BreakRegion> || std::is_same_v<T, ContinueRegion>) {
                const auto loop = loops_.find(r.loop);
                require(loop != loops_.end(), "Loop transfer target is not an active enclosing loop");
                entry = std::is_same_v<T, BreakRegion> ? loop->second.first : loop->second.second;
            } else if constexpr (std::is_same_v<T, ReturnRegion>) {
                entry = absent;
            }
        }, p_.regions[id.value]);
        if (!nodes_[index].condition && entry != absent) nodes_[index].next = {entry};
        return index;
    }
    void liveness() {
        // Visit successors first. Arena order interleaves region anchors with
        // reverse-built operation chains and otherwise causes quadratic waves
        // of liveness updates even in an acyclic sequence of blocks.
        rank_.assign(nodes_.size(), absent);
        std::vector<bool> seen(nodes_.size());
        std::size_t finished = 0;
        for (std::size_t root = 0; root < nodes_.size(); ++root) if (!seen[root]) {
            std::vector<std::pair<std::size_t, std::size_t>> stack{{root, 0}};
            seen[root] = true;
            while (!stack.empty()) {
                step();
                auto& [index, edge] = stack.back();
                if (edge == nodes_[index].next.size()) {
                    rank_[index] = finished++; stack.pop_back();
                } else {
                    const auto next = nodes_[index].next[edge++];
                    if (!seen[next]) { seen[next] = true; stack.emplace_back(next, 0); }
                }
            }
        }
        std::set<std::pair<std::size_t, std::size_t>> queue;
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            for (auto next : nodes_[i].next) nodes_[next].predecessors.push_back(i);
            queue.emplace(rank_[i], i); nodes_[i].queued = true;
        }
        while (!queue.empty()) {
            step();
            const auto index = queue.begin()->second; queue.erase(queue.begin());
            auto& n = nodes_[index]; n.queued = false;
            Definitions live, predicates;
            for (auto next : n.next) {
                step(nodes_[next].live.size() + nodes_[next].predicates.size());
                live.insert(nodes_[next].live.begin(), nodes_[next].live.end());
                predicates.insert(nodes_[next].predicates.begin(), nodes_[next].predicates.end());
            }
            auto use = [&](const Operand& operand) {
                step();
                if (const auto* id = std::get_if<ValueId>(&operand)) live.insert(id->value);
            };
            if (n.operation.valid()) {
                const auto& op = p_.ir.operations[n.operation.value];
                bool tracked = false;
                for (const auto& output : op.outputs) {
                    const auto id = std::get<ValueId>(output).value;
                    live.erase(id); tracked |= predicates.erase(id) != 0;
                }
                for (const auto& input : op.inputs) {
                    use(input);
                    if (tracked && boolean_expression(op.opcode))
                        if (const auto* id = std::get_if<ValueId>(&input); id && p_.ir.values[id->value].type == ValueType::predicate)
                            predicates.insert(id->value);
                }
            }
            if (n.condition) {
                use(*n.condition);
                if (const auto* id = std::get_if<ValueId>(&*n.condition)) predicates.insert(id->value);
            }
            if (live != n.live || predicates != n.predicates) {
                n.live = std::move(live);
                n.predicates = std::move(predicates);
                for (auto pred : n.predecessors) if (!nodes_[pred].queued) {
                    queue.emplace(rank_[pred], pred); nodes_[pred].queued = true;
                }
            }
        }
    }
    std::optional<Boolean> evaluate(const Operation& op, const Environment& env) {
        auto input = [&](unsigned i) { return boolean(op.inputs[i], env); };
        if (op.opcode == Opcode::copy) return input(0);
        if (op.opcode == Opcode::logical_not) {
            if (auto value = input(0)) { value->inverted = !value->inverted; return value; }
        } else if (op.opcode == Opcode::logical_and || op.opcode == Opcode::logical_or || op.opcode == Opcode::logical_xor) {
            auto a = input(0), b = input(1);
            if (a && b) {
                if (a->atom == b->atom) {
                    if (op.opcode == Opcode::logical_xor) return Boolean{UINT64_MAX, a->inverted != b->inverted};
                    if (a->inverted == b->inverted) return a;
                    return Boolean{UINT64_MAX, op.opcode == Opcode::logical_or};
                }
                if (b->atom == UINT64_MAX) std::swap(a, b);
                if (a->atom == UINT64_MAX) {
                    if (op.opcode == Opcode::logical_and) return a->inverted ? b : a;
                    if (op.opcode == Opcode::logical_or) return a->inverted ? a : b;
                    b->inverted ^= a->inverted; return b;
                }
            }
        } else if (op.opcode == Opcode::select) {
            if (auto condition = input(0); condition && condition->atom == UINT64_MAX)
                return input(condition->inverted ? 1 : 2);
        }
        return {};
    }
    void run(std::size_t entry) {
        std::set<std::pair<std::size_t, std::size_t>> queue;
        auto propagate = [&](std::size_t target, Environment env, Flow flow) {
            step(env.size() + flow.defined.size() + flow.outputs.size() + 1);
            auto& n = nodes_[target];
            std::erase_if(env, [&](const auto& item) { return !n.predicates.contains(item.first); });
            canonicalize(env);
            std::erase_if(flow.defined, [&](auto id) { return !n.live.contains(id); });
            // When both arms establish identical must-facts, their differing
            // Boolean constants can be represented by one fresh atom (and its
            // aliases/complements). This is their exact union, so future guards
            // retain correlations without enumerating independent choices.
            if (!n.widened) {
                for (auto it = n.incoming.begin(); it != n.incoming.end();) {
                    step(flow.defined.size() + flow.outputs.size() + 1);
                    if (it->first != env && it->second.defined == flow.defined && it->second.outputs == flow.outputs) {
                        if (auto joined = coalesce(env, it->first)) {
                            env = std::move(*joined);
                            flow.exact &= it->second.exact;
                            n.incoming.erase(it);
                            it = n.incoming.begin();
                            continue;
                        }
                    }
                    ++it;
                }
            }
            // After the fixed bound, retain one monotone summary. Constants
            // and alias relations survive only when all paths agree; must-facts
            // intersect. Lost correlations can reject valid shaders, never make
            // an undefined read safe. The summary cannot witness entry writes.
            if (n.widened || (n.precise_updates == environment_limit && !n.incoming.contains(env))) {
                for (const auto& [old_env, old_flow] : n.incoming) {
                    env = join(env, old_env);
                    intersect(flow.defined, old_flow.defined);
                    intersect(flow.outputs, old_flow.outputs);
                }
                flow.exact = false;
                if (n.widened && n.incoming.begin()->first == env &&
                    n.incoming.begin()->second.defined == flow.defined &&
                    n.incoming.begin()->second.outputs == flow.outputs) return;
                n.incoming.clear();
                n.widened = true;
            }
            auto [found, inserted] = n.incoming.try_emplace(std::move(env), flow);
            if (inserted && !n.widened) ++n.precise_updates;
            bool changed = inserted;
            if (!inserted) {
                changed |= intersect(found->second.defined, flow.defined);
                changed |= intersect(found->second.outputs, flow.outputs);
                if (flow.exact && !found->second.exact) {
                    found->second.exact = true;
                    changed = true;
                }
            }
            if (changed && !n.queued) { queue.emplace(nodes_.size() - rank_[target], target); n.queued = true; }
        };
        propagate(entry, {}, {});
        while (!queue.empty()) {
            step();
            const auto index = queue.begin()->second; queue.erase(queue.begin());
            auto& n = nodes_[index]; n.queued = false;
            // Charge the snapshot before copying it, including output facts.
            for (const auto& [env, flow] : n.incoming)
                step(env.size() + flow.defined.size() + flow.outputs.size() + 1);
            const auto incoming = n.incoming;
            for (const auto& [environment, incoming_flow] : incoming) {
                step(environment.size() + incoming_flow.defined.size() + incoming_flow.outputs.size() + 1);
                if (n.condition) {
                    const auto truth = boolean(*n.condition, environment);
                    for (unsigned arm = 0; arm < 2; ++arm)
                        if (!truth || truth->atom != UINT64_MAX || truth->inverted == (arm == 0)) {
                        auto env = environment;
                        if (truth && truth->atom != UINT64_MAX) {
                            for (auto& [id, value] : env) if (value.atom == truth->atom)
                                value = {UINT64_MAX, bool((arm == 0) ^ truth->inverted ^ value.inverted)};
                        }
                        propagate(n.next[arm], std::move(env), incoming_flow);
                    }
                    continue;
                }
                auto flow = incoming_flow;
                auto env = environment;
                if (n.operation.valid()) {
                    const auto& op = p_.ir.operations[n.operation.value];
                    // Evaluate before any destination changes, including aliases.
                    const auto truth = op.outputs.size() == 1 && p_.ir.values[std::get<ValueId>(op.outputs[0]).value].type == ValueType::predicate
                        ? evaluate(op, environment) : std::optional<Boolean>{};
                    for (const auto& output : op.outputs) {
                        const auto id = std::get<ValueId>(output).value;
                        flow.defined.insert(id);
                        if (p_.ir.values[id].type != ValueType::predicate ||
                            n.next.empty() || !nodes_[n.next[0]].predicates.contains(id)) continue;
                        step(env.size() + 1);
                        // Fresh atoms cannot collide with canonical live names,
                        // which are always ValueIds below 2^32.
                        set(env, id, truth.value_or(Boolean{(1ull << 32) + id, false}));
                    }
                    // A widened path is only a possible execution, not a
                    // witness for entry initialization. Must-write facts still
                    // propagate normally and can prove unconditional exports.
                    if (op.opcode == Opcode::store_io) {
                        const auto& io = std::get<IoInfo>(op.info);
                        for (unsigned component = 0; component < 4; ++component) if (io.component_mask & (1u << component)) {
                            const auto key = output_key(io.semantic, io.location, component);
                            flow.outputs.insert(key);
                            if (flow.exact) written_.insert(key);
                        }
                    } else if (op.opcode == Opcode::discard) {
                        for (const auto& io : plan_.io) if (io.access == Access::write)
                            for (unsigned component = 0; component < 4; ++component) if (io.mask & (1u << component)) {
                                const auto key = output_key(io.semantic, io.location, component);
                                flow.outputs.insert(key);
                                if (flow.exact) written_.insert(key);
                            }
                    }
                }
                for (auto next : n.next) propagate(next, env, flow);
            }
        }
    }
    Flow proof(std::size_t index) {
        Flow result{false, {}, {}};
        for (const auto& [env, flow] : nodes_[index].incoming) {
            step(flow.defined.size() + flow.outputs.size() + 1);
            if (!result.reachable) result = flow;
            else {
                intersect(result.defined, flow.defined);
                intersect(result.outputs, flow.outputs);
            }
        }
        return result;
    }
public:
    Definedness(const StructuredProgram& p, const ResourcePlan& plan, std::uint64_t& steps)
        : p_(p), plan_(plan), steps_(steps), region_nodes_(p.regions.size(), absent), operation_nodes_(p.ir.operations.size(), absent) {
        node(); // Implicit return when the root falls through.
        const auto entry = build(p.functions[0].root, 0, 0);
        liveness(); run(entry);
    }
    Flow region(RegionId id) { return proof(region_nodes_.at(id.value)); }
    Flow operation(OperationId id) { return proof(operation_nodes_.at(id.value)); }
    Flow fallthrough() { return proof(0); }
    // Output keys witnessed before losing predicate correlations. Widening
    // must not enable entry initialization from a spurious reachable write.
    const OutputComponents& written() const { return written_; }
};
class StageContext final : public MslContext {
    const StructuredProgram& p_;
    const ResourcePlan& plan_;
    const EmitOptions& options_;
    std::string text_, helpers_;
    std::set<std::string> helper_names_;
    std::set<std::uint32_t> regions_, blocks_, operations_, assignments_;
    Flow flow_;
    std::optional<Definedness> definedness_;
    OutputComponents initialized_outputs_;
    bool live_{true};
    struct Loop { RegionId id; bool continuing{}; std::vector<Flow> breaks, continues; };
    std::vector<Loop> loops_;
    unsigned indent_{1};
    std::uint64_t steps_{};
    OperationId current_;
    std::optional<DepthConvention> depth_;
    bool fixed_point_size_ = false;

    void append(std::string_view text) {
        require(text.size() <= p_.ir.limits.source_bytes && text_.size() + helpers_.size() <= p_.ir.limits.source_bytes - text.size(),
                "MSL source limit exceeded", DiagnosticCode::limit_exceeded);
        text_ += text;
    }
    void line(std::string_view text) { append(std::string(indent_ * 4, ' ') + std::string(text) + "\n"); }
    void step() { require(++steps_ <= p_.ir.limits.analysis_steps, "MSL analysis budget exceeded", DiagnosticCode::limit_exceeded); }
    std::string loop_transfers(std::size_t count) const {
        std::string condition;
        for (std::size_t i = 0; i < count; ++i) {
            if (!condition.empty()) condition += " || ";
            const auto id = number(loops_[i].id.value);
            condition += "armigo_break_" + id + " || armigo_continue_" + id;
        }
        return condition;
    }
    void emit_return() {
        for (const auto& io : plan_.io) if (io.access == Access::write)
            for (unsigned component = 0; component < 4; ++component) if (io.mask & (1u << component))
                require(!live_ || initialized_outputs_.contains(output_key(io.semantic, io.location, component)) ||
                            flow_.outputs.contains(output_key(io.semantic, io.location, component)),
                        "Stage output is not written on every live return path (semantic=" +
                        number(static_cast<unsigned>(io.semantic)) + ", location=" + number(io.location) +
                        ", component=" + number(component) + ")", DiagnosticCode::undefined_read);
        if (depth_ == DepthConvention::negative_one_to_one) {
            line("const float armigo_depth_sum = out.position.z + out.position.w;");
            line("out.position.z = armigo_depth_sum * 0.5f;");
        }
        if (options_.fragment_output_types_from_render_targets) {
            std::set<std::uint16_t> locations;
            for (const auto& io : plan_.io) {
                if (io.access != Access::write || io.semantic != IoSemantic::color ||
                    !locations.insert(io.location).second) continue;
                const auto name = io_name(io.semantic, io.location);
                const auto raw = "armigo_" + name + "_raw";
                line("if (armigo_" + name + "_is_f32) out." + name + "_f32 = as_type<float4>(" + raw + ");");
                line("if (armigo_" + name + "_is_u32) out." + name + "_u32 = " + raw + ";");
                line("if (armigo_" + name + "_is_s32) out." + name + "_s32 = as_type<int4>(" + raw + ");");
            }
        }
        line(p_.ir.stage == Stage::compute ? "return;" : "return out;");
    }
    void region(RegionId id, unsigned nesting = 0) {
        step();
        require(nesting < p_.ir.limits.nesting_depth, "Structured nesting limit exceeded", DiagnosticCode::limit_exceeded);
        require(id.value < p_.regions.size() && regions_.insert(id.value).second, "Invalid, cyclic or shared structured region");
        std::visit([&](const auto& r) {
            using T = std::decay_t<decltype(r)>;
            if constexpr (std::is_same_v<T, SequenceRegion>) {
                for (auto child : r.children) region(child, nesting + 1);
            } else if constexpr (std::is_same_v<T, BlockRegion>) {
                require(r.block.value < p_.ir.blocks.size() && blocks_.insert(r.block.value).second, "Invalid or repeated structured block");
                const auto& block = p_.ir.blocks[r.block.value];
                require(block.function == p_.ir.entry && block.phis.empty(), "Structured block ownership or phi destruction is invalid");
                for (auto op_id : block.operations) {
                    step();
                    require(flow_.reachable, "Unreachable operation in structured sequence");
                    require(op_id.value < p_.ir.operations.size() && operations_.insert(op_id.value).second, "Invalid or repeated operation");
                    current_ = op_id;
                    const auto proof = definedness_->operation(op_id);
                    live_ = proof.reachable; flow_.defined = proof.defined; flow_.outputs = proof.outputs;
                    const auto& op = operation(op_id);
                    // Check all raw inputs before callbacks can assign aliased destinations.
                    for (const auto& value : op.inputs) {
                        if (const auto* v = std::get_if<ValueId>(&value)) {
                            require(v->value < p_.ir.values.size(), "Input value is outside arena");
                            take(operand(value, p_.ir.values[v->value].type == ValueType::predicate ? ScalarType::boolean :
                                p_.ir.values[v->value].type == ValueType::word64 ? ScalarType::u64 : ScalarType::u32));
                        } else if (const auto* literal = std::get_if<LiteralBits>(&value)) {
                            take(operand(value, literal->type == ValueType::predicate ? ScalarType::boolean :
                                literal->type == ValueType::word64 ? ScalarType::u64 : ScalarType::u32));
                        } else fail("Unlowered or undefined operand remains", DiagnosticCode::undefined_read);
                    }
                    if (options_.source_locations && op.source.pc) line("// Guest instruction byte offset " + number(*op.source.pc));
                    if (op.opcode == Opcode::load_io) {
                        const auto& io = std::get<IoInfo>(op.info);
                        if (io.output_storage) {
                            for (unsigned component = 0; component < 4; ++component) if (io.component_mask & (1u << component))
                                require(!live_ || flow_.outputs.contains(output_key(io.semantic, io.location, component)),
                                        "Stage output is read before it is written", DiagnosticCode::undefined_read);
                        }
                    }
                    take(dispatch_msl(*this, op_id));
                    if (op.opcode == Opcode::store_io) {
                        const auto& io = std::get<IoInfo>(op.info);
                        for (unsigned component = 0; component < 4; ++component) if (io.component_mask & (1u << component))
                            flow_.outputs.insert(output_key(io.semantic, io.location, component));
                    } else if (op.opcode == Opcode::discard) {
                        // Killed invocations have no observable stage outputs. They
                        // still execute subsequent IR and retain value-read checks.
                        for (const auto& io : plan_.io) if (io.access == Access::write)
                            for (unsigned component = 0; component < 4; ++component) if (io.mask & (1u << component))
                                flow_.outputs.insert(output_key(io.semantic, io.location, component));
                    }
                }
            } else if constexpr (std::is_same_v<T, IfRegion>) {
                require(flow_.reachable, "Unreachable conditional region");
                const auto proof = definedness_->region(id);
                live_ = proof.reachable; flow_.defined = proof.defined; flow_.outputs = proof.outputs;
                auto condition = take(operand(r.condition, ScalarType::boolean));
                const auto incoming = flow_;
                line("if (" + condition + ") {"); ++indent_;
                region(r.when_true, nesting + 1); --indent_;
                const auto first = flow_; flow_ = incoming;
                if (r.when_false.valid()) {
                    line("} else {"); ++indent_; region(r.when_false, nesting + 1); --indent_;
                }
                line("}"); flow_ = merge({first, flow_});
            } else if constexpr (std::is_same_v<T, LoopRegion>) {
                require(flow_.reachable, "Unreachable loop region");
                loops_.push_back({id, false, {}, {}});
                line("while (true) {"); ++indent_;
                line("bool armigo_break_" + number(id.value) + " = false;");
                line("bool armigo_continue_" + number(id.value) + " = false;");
                line("do {"); ++indent_; region(r.body, nesting + 1); --indent_; line("} while (false);");
                loops_.back().continues.push_back(flow_);
                flow_ = merge(loops_.back().continues);
                if (loops_.size() > 1) line("if (" + loop_transfers(loops_.size() - 1) + ") break;");
                line("if (armigo_break_" + number(id.value) + ") break;");
                line("if (armigo_continue_" + number(id.value) + ") armigo_continue_" + number(id.value) + " = false;");
                loops_.back().continuing = true;
                if (r.continuing.valid()) region(r.continuing, nesting + 1);
                flow_ = merge(loops_.back().breaks);
                loops_.pop_back(); --indent_; line("}");
                if (!loops_.empty()) {
                    if (loops_.back().continuing) line("if (armigo_continue_" + number(loops_.back().id.value) + ") continue;");
                    line("if (" + loop_transfers(loops_.size()) + ") break;");
                }
            } else if constexpr (std::is_same_v<T, BreakRegion> || std::is_same_v<T, ContinueRegion>) {
                auto target = std::find_if(loops_.begin(), loops_.end(), [&](const auto& loop) { return loop.id == r.loop; });
                require(flow_.reachable && target != loops_.end(), "Loop transfer target is not an active enclosing loop");
                const bool direct_continuing = target == std::prev(loops_.end()) && target->continuing;
                if constexpr (std::is_same_v<T, BreakRegion>) {
                    target->breaks.push_back(flow_);
                    if (!direct_continuing) line("armigo_break_" + number(target->id.value) + " = true;");
                    line("break;");
                } else {
                    if (!target->continuing) target->continues.push_back(flow_);
                    if (!direct_continuing) line("armigo_continue_" + number(target->id.value) + " = true;");
                    line(direct_continuing ? "continue;" : "break;");
                }
                flow_.reachable = false;
            } else if constexpr (std::is_same_v<T, ReturnRegion>) {
                require(r.values.empty(), "Entry point cannot return IR values", DiagnosticCode::unsupported_capability);
                require(flow_.reachable, "Unreachable return region");
                const auto proof = definedness_->region(id);
                live_ = proof.reachable; flow_.defined = proof.defined; flow_.outputs = proof.outputs;
                emit_return(); flow_.reachable = false;
            }
        }, p_.regions[id.value]);
    }
public:
    StageContext(const StructuredProgram& p, const ResourcePlan& plan, const EmitOptions& o) : p_(p), plan_(plan), options_(o) {}
    const StructuredProgram& program() const override { return p_; }
    const ResourcePlan& resources() const override { return plan_; }
    bool fragment_output_types_from_render_targets() const override {
        return options_.fragment_output_types_from_render_targets;
    }
    const Operation& operation(OperationId id) const override { return p_.ir.operations.at(id.value); }
    Diagnostic error(OperationId id, DiagnosticCode code, std::string message, std::string modifier = {}) const override {
        auto error = diagnose(Phase::emit_msl, code, id.value < p_.ir.operations.size() ? operation(id).source : SourceLocation{}, std::move(message));
        error.modifier = std::move(modifier); return error;
    }
    Result<std::string> operand(const Operand& value, ScalarType type) override {
        try {
            require(type != ScalarType::f16 || plan_.profile.supports_fp16, "FP16 is not supported by profile", DiagnosticCode::unsupported_capability);
            std::string expression;
            if (const auto* v = std::get_if<ValueId>(&value)) {
                require(v->value < p_.ir.values.size() && p_.ir.values[v->value].type == storage_type(type), "Operand interpretation changes storage width");
                expression = "value" + number(v->value);
            } else if (const auto* literal = std::get_if<LiteralBits>(&value)) {
                require(literal->type == storage_type(type), "Literal interpretation changes storage width");
                require(literal->type != ValueType::word32 || literal->bits <= UINT32_MAX, "Literal exceeds word width");
                require(literal->type != ValueType::predicate || literal->bits <= 1, "Predicate literal is not Boolean");
                expression = literal->type == ValueType::predicate ? (literal->bits ? "true" : "false") : number(literal->bits) + (literal->type == ValueType::word64 ? "ul" : "u");
            } else fail("Unlowered or undefined MSL operand", DiagnosticCode::undefined_read);
            return interpret(expression, type);
        } catch (const Failure& f) { return error(current_, f.diagnostic.code, f.diagnostic.message); }
    }
    Result<std::string> resource(ResourceId id) override {
        try {
            const auto& binding = planned(plan_, id);
            return std::string(std::holds_alternative<ArgumentBufferMember>(binding.location) ? "resources.resource" : "resource") + number(id.value);
        }
        catch (const Failure& f) { return error(current_, f.diagnostic.code, f.diagnostic.message); }
    }
    Status assign(ValueId id, std::string_view expression, ScalarType type) override {
        return callback(*this, current_, [&] {
            require(id.value < p_.ir.values.size() && p_.ir.values[id.value].type == storage_type(type), "Assignment changes storage width");
            require(p_.ir.values[id.value].kind == ValueKind::mutable_local || assignments_.insert(id.value).second, "Temporary assigned more than once");
            line("value" + number(id.value) + " = " + raw(std::string(expression), type) + ";");
            flow_.defined.insert(id.value);
        });
    }
    Status statement(std::string_view statement) override {
        return callback(*this, current_, [&] { line(statement); });
    }
    Status require_helper(std::string_view name) override {
        return callback(*this, current_, [&] {
            if (helper_names_.contains(std::string(name))) return;
            std::string source;
            if (name == "armigo_cbuf_word") source = R"(
inline uint armigo_cbuf_word(constant uint* cb, ulong address, uint size_bytes, uint window_bytes) {
    // Every bound window holds window_bytes: guest words below the bound size
    // and zero words past it, so only the window limits the read.
    const ulong word_address = address & ~3ul;
    return word_address <= ulong(window_bytes - 4u) ? cb[word_address / 4ul] : 0u;
}
)";
            else if (name == "armigo_multiply_rne") source = armigo::shader::emit_rne_multiply_msl();
            else if (name == "armigo_reciprocal_rne") source = armigo::shader::emit_reciprocal_msl();
            else if (name == "armigo_mul_rtz_ftz") source = armigo::shader::emit_rtz_ftz_multiply_msl();
            else if (name == "armigo_i2f")
                source = std::string(armigo::shader::graphics::detail::integer_to_float_msl_helper());
            // armigo_cbuf_word, armigo_multiply_rne: requested by the memory
            // generators (runtime window guard, perspective-correction
            // product). armigo_reciprocal_rne stays registered as the reserved
            // exact reciprocal for non-multifunction users. armigo_i2f carries
            // the exact integer-to-float truncation, which the native cast
            // cannot express; armigo_mul_rtz_ftz the exact toward-zero product, no longer requested here (FMUL.RZ is the native product). Every other floating operation generator is
            // native Metal.
            else fail("Unknown MSL helper: " + std::string(name), DiagnosticCode::unsupported_capability);
            require(text_.size() + helpers_.size() + source.size() <= p_.ir.limits.source_bytes, "MSL helper source limit exceeded", DiagnosticCode::limit_exceeded);
            helpers_ += source; helper_names_.insert(std::string(name));
        });
    }
    MslArtifact emit() {
        // A vertex stage writes the fixed point size at entry, and a program
        // exporting its own size through attribute 0x6c overrides it. BotW
        // draws points only at NVN's fixed size 1.0 or with a program size, so
        // the fixed write is the constant 1.0 and draw
        // time refuses any other fixed size explicitly; a function constant would
        // force every vertex function to be specialized. Pipelines leave the
        // topology class unspecified so the export is valid with any primitive;
        // layered stages (layer or viewport index exports) need the class and
        // therefore get no synthesized size.
        fixed_point_size_ = plan_.stage == Stage::vertex &&
            std::none_of(plan_.io.begin(), plan_.io.end(), [](const auto& io) {
                return io.access == Access::write &&
                       (io.semantic == IoSemantic::point_size || io.semantic == IoSemantic::layer ||
                        io.semantic == IoSemantic::viewport_index);
            });
        validate_plan(p_, plan_, options_);
        const auto& ir = p_.ir;
        require(p_.regions.size() <= ir.limits.regions && ir.values.size() <= ir.limits.values && ir.operations.size() <= ir.limits.operations &&
                ir.blocks.size() <= ir.limits.blocks && ir.functions.size() <= ir.limits.functions, "Structured program exceeds limits", DiagnosticCode::limit_exceeded);
        require(p_.functions.size() == 1 && ir.functions.size() == 1 && ir.entry.value == 0 && p_.functions.front().function == ir.entry,
                "MSL function-call ABI is not implemented", DiagnosticCode::unsupported_capability);
        require(ir.functions[0].parameters.empty() && ir.functions[0].results.empty(), "Entry function cannot have IR arguments/results");
        // The shared validator predates phi destruction and enforces SSA for legal
        // IR. Its register-form ownership/schema checks remain useful after making
        // a private metadata view; structured definite assignment is checked above.
        Program structural = ir;
        for (auto& op : structural.operations) if (auto* memory = std::get_if<MemoryInfo>(&op.info))
            if (memory->space == AddressSpace::constant_buffer) memory->resource.reset();
        take(validate(structural, IrForm::registers, Phase::emit_msl));
        definedness_.emplace(p_, plan_, steps_);
        if (ir.stage == Stage::vertex) {
            // armigo-output-init-v1: an output with a write on some live path is
            // initialized at entry as a whole.
            const auto& written = definedness_->written();
            for (const auto& io : plan_.io) if (io.access == Access::write) {
                bool written_somewhere = false;
                for (unsigned component = 0; component < 4; ++component) if (io.mask & (1u << component))
                    written_somewhere |= written.contains(output_key(io.semantic, io.location, component));
                if (written_somewhere)
                    for (unsigned component = 0; component < 4; ++component) if (io.mask & (1u << component))
                        initialized_outputs_.insert(output_key(io.semantic, io.location, component));
            }
            bool position = false;
            for (const auto& io : plan_.io) if (io.semantic == IoSemantic::position && io.access == Access::write && io.mask == 15) position = true;
            require(position, "Vertex stage must declare a complete position output");
            for (const auto& d : plan_.dependencies) if (d.key.kind == FactKind::depth_convention) {
                require(!depth_ && std::holds_alternative<DepthConvention>(d.value), "Conflicting/invalid vertex depth dependency");
                depth_ = std::get<DepthConvention>(d.value);
            }
            if (!depth_) {
                auto d = diagnose(Phase::emit_msl, DiagnosticCode::missing_fact, {}, "Vertex stage requires a recorded clip-depth convention");
                d.missing_fact = FactKey{FactKind::depth_convention}; throw Failure{d};
            }
            require(*depth_ <= DepthConvention::negative_one_to_one, "Unknown vertex depth convention");
        }
        // Barriers stay where the guest placed them: the guest program already
        // keeps a barrier workgroup-uniform on the hardware, so control flow
        // around it needs no proof.
        std::string entry = options_.entry_point.empty() ? (ir.stage == Stage::vertex ? "armigo_vertex_main" : ir.stage == Stage::compute ? "armigo_compute_main" : "armigo_fragment_main") : options_.entry_point;
        require(!entry.empty() && (std::isalpha(static_cast<unsigned char>(entry[0])) || entry[0] == '_') &&
                std::all_of(entry.begin(), entry.end(), [](unsigned char c) { return std::isalnum(c) || c == '_'; }), "Entry point is not an identifier");
        require(!reserved_entry(entry), "Entry point uses a reserved MSL identifier");
        if (plan_.argument_buffer) {
            std::vector<const PlannedResource*> members;
            for (const auto& resource : plan_.resources)
                if (std::holds_alternative<ArgumentBufferMember>(resource.location)) members.push_back(&resource);
            std::sort(members.begin(), members.end(), [](const auto* a, const auto* b) {
                return std::get<ArgumentBufferMember>(a->location).byte_offset <
                       std::get<ArgumentBufferMember>(b->location).byte_offset;
            });
            append("struct ArmigoStageResources {\n");
            for (const auto* member : members) append("    " + resource_argument(*member) + ";\n");
            append("};\n");
        }
        if (plan_.custom_border) append(std::string(custom_border_msl));
        append(io_declarations(plan_, false, options_.fragment_output_types_from_render_targets, options_.emulate_depth_bias, fixed_point_size_));
        append(io_declarations(plan_, true, options_.fragment_output_types_from_render_targets, options_.emulate_depth_bias, fixed_point_size_));
        const bool vertex_fetch = plan_.vertex_fetch_buffer_index.has_value();
        if (vertex_fetch) append(std::string(vertex_fetch_msl()));
        // The S2R SR_YDIRECTION sign is draw state, so the stage declares the
        // armigo_y_negate function constant and materializes the word once;
        // the specialized value is ±1.0f's bit pattern, mirroring the guest's
        // flipped fragment-Y view when the YControl negate bit is set. Index 9
        // follows the render-target classes (0..7) and depth-bias fallback (8).
        if (std::any_of(plan_.io.begin(), plan_.io.end(), [](const auto& io) {
                return io.access == Access::read && io.semantic == IoSemantic::y_direction; }))
            append(std::string("constant bool armigo_y_negate [[function_constant(9)]];\n"
                "constant uint ") + io_name(IoSemantic::y_direction, 0) +
                " = armigo_y_negate ? 0xbf800000u : 0x3f800000u;\n");
        // Metal's [[point_coord]] has an upper-left origin. Draw time sets index
        // 11 for point draws whose guest state selects a lower-left origin
        // when PointCoordReplace bit 2 is clear. Point-coordinate reads flip y
        // under that setting.
        if (plan_.stage == Stage::fragment &&
            std::any_of(plan_.io.begin(), plan_.io.end(), [](const auto& io) {
                return io.access == Access::read && io.semantic == IoSemantic::point_coord;
            }))
            append(std::string("constant bool armigo_point_coord_lower_left [[function_constant(11)]];\n"));
        // Custom-border records are read only when draw time leaves index 12
        // unset or sets it: a pipeline whose records are all disabled
        // specializes the record checks away. Only fragments with render-target
        // output types declare it: every pipeline already specializes those,
        // and Metal refuses an unspecialized function that declares a constant.
        if (plan_.custom_border && options_.fragment_output_types_from_render_targets)
            append("constant bool armigo_border_records_ [[function_constant(12)]];\n"
                   "constant bool armigo_border_records = is_function_constant_defined("
                   "armigo_border_records_) ? armigo_border_records_ : true;\n");
        append((ir.stage == Stage::compute ? "kernel void " : ir.stage == Stage::vertex ? "vertex ArmigoOutput " : "fragment ArmigoOutput ") + entry + "(");
        bool argument = false;
        // A fetching stage reads its generic inputs itself: no stage_in.
        const auto stage_input = [&](const IoUse& io) {
            return io.access == Access::read && io.semantic != IoSemantic::vertex_id &&
                   io.semantic != IoSemantic::instance_id && !(vertex_fetch && io.semantic == IoSemantic::generic);
        };
        if (ir.stage != Stage::compute && std::any_of(plan_.io.begin(), plan_.io.end(), stage_input)) {
            append("ArmigoInput in [[stage_in]]"); argument = true;
        }
        if (vertex_fetch) {
            // The fetch indexes streams by the vertex and instance IDs (both
            // include their draw's base) and the base instance; the guest's
            // own ID reads alias them in the prologue.
            if (argument) append(",\n    ");
            append("uint armigo_fetch_vertex_id [[vertex_id]],\n    uint armigo_fetch_instance_id [[instance_id]],\n"
                   "    uint armigo_fetch_base_instance [[base_instance]],\n"
                   "    constant ArmigoVertexFetch& armigo_vertex_fetch [[buffer(" +
                   number(*plan_.vertex_fetch_buffer_index) + ")]]");
            argument = true;
        }
        for (const auto& io : plan_.io) if (!vertex_fetch && io.access == Access::read &&
                (io.semantic == IoSemantic::vertex_id || io.semantic == IoSemantic::instance_id)) {
            if (argument) append(",\n    ");
            if (io.semantic == IoSemantic::vertex_id) {
                // Maxwell and Metal both include first/base vertex in this ID.
                append("uint " + io_name(io.semantic, io.location) + " [[vertex_id]]");
            } else {
                // Keep the original guest base separate from Metal's per-call
                // base, which includes the index of a split continuation.
                append("uint armigo_instance_index [[instance_id]]");
            }
            argument = true;
        }
        for (const auto& io : plan_.io) if (io.access == Access::read &&
                (io.semantic == IoSemantic::thread_id || io.semantic == IoSemantic::workgroup_id ||
                 io.semantic == IoSemantic::subgroup_lane_id || io.semantic == IoSemantic::quad_lane_id)) {
            if (argument) append(",\n    ");
            const auto attribute = io.semantic == IoSemantic::thread_id ? "thread_position_in_threadgroup" :
                io.semantic == IoSemantic::workgroup_id ? "threadgroup_position_in_grid" :
                // The fragment quad position: the group-of-four lane index the
                // guest quad swizzle arithmetic selects with.
                io.semantic == IoSemantic::quad_lane_id ? "thread_index_in_quadgroup" :
                "thread_index_in_simdgroup";
            const auto vector3 = io.semantic == IoSemantic::thread_id || io.semantic == IoSemantic::workgroup_id;
            append(std::string(vector3 ? "uint3 " : "uint ") +
                io_name(io.semantic, io.location) + " [[" + attribute + "]]");
            argument = true;
        }
        if (plan_.argument_buffer) {
            if (argument) append(",\n    ");
            append("const device ArmigoStageResources& resources [[buffer(" + number(plan_.argument_buffer->buffer_index) + ")]]");
            argument = true;
        }
        for (const auto& resource : plan_.resources) {
            if (std::holds_alternative<ArgumentBufferMember>(resource.location)) continue;
            if (argument) append(",\n    ");
            append(resource_argument(resource)); argument = true;
        }
        if (plan_.custom_border) {
            if (argument) append(",\n    ");
            append("const device ArmigoCustomBorder* armigo_border [[buffer(" +
                number(plan_.custom_border->buffer_index) + ")]]");
            argument = true;
        }
        if (plan_.draw_support_buffer_index) {
            if (argument) append(",\n    ");
            append("constant uint& armigo_guest_first_instance [[buffer(" +
                number(*plan_.draw_support_buffer_index) + ")]]");
            argument = true;
        }
        const bool input_position = std::any_of(plan_.io.begin(), plan_.io.end(), [](const auto& io) {
            return io.access == Access::read &&
                (io.semantic == IoSemantic::position || io.semantic == IoSemantic::fragment_coord);
        });
        if (options_.emulate_depth_bias) {
            if (argument) append(",\n    ");
            append("constant float4& armigo_depth_bias [[buffer(30)]]");
            if (!input_position) append(",\n    float4 armigo_position [[position]]");
        }
        append(") {\n");
        if (ir.stage != Stage::compute) line("ArmigoOutput out;");
        if (vertex_fetch) {
            if (std::any_of(plan_.io.begin(), plan_.io.end(), [](const auto& io) {
                    return io.access == Access::read && io.semantic == IoSemantic::generic; }))
                line("ArmigoInput in;");
            for (const auto& io : plan_.io) {
                if (io.access != Access::read) continue;
                if (io.semantic == IoSemantic::vertex_id)
                    line("const uint " + io_name(io.semantic, io.location) + " = armigo_fetch_vertex_id;");
                else if (io.semantic == IoSemantic::instance_id)
                    line("const uint armigo_instance_index = armigo_fetch_instance_id;");
                else if (io.semantic == IoSemantic::generic) {
                    require(io.location < vertex_fetch_attributes && io.type.components >= 1 &&
                            io.type.components <= 4, "Vertex fetch input is outside the fetch table");
                    const auto fetch = io.type.scalar == ScalarType::u32 ? "armigo_vertex_fetch_uint" :
                        io.type.scalar == ScalarType::s32 ? "armigo_vertex_fetch_int" :
                        io.type.scalar == ScalarType::f32 ? "armigo_vertex_fetch_float" : nullptr;
                    require(fetch != nullptr, "Vertex fetch input has an unsupported component type",
                            DiagnosticCode::unsupported_capability);
                    static constexpr const char* swizzles[]{"", ".x", ".xy", ".xyz", ""};
                    line("in." + io_name(io.semantic, io.location) + " = " + fetch + "(armigo_vertex_fetch, " +
                         number(io.location) + "u, armigo_fetch_vertex_id, armigo_fetch_instance_id, "
                         "armigo_fetch_base_instance)" + swizzles[io.type.components] + ";");
                }
            }
        }
        if (options_.emulate_depth_bias) {
            // Derivatives precede guest control flow and discard. Keep the
            // combined signed clamp in window-depth units, independent of the
            // native attachment's floating-point bias exponent.
            line("if (armigo_has_depth) {");
            ++indent_;
            line(std::string("const float armigo_z = ") + (input_position ? "in.position.z;" : "armigo_position.z;"));
            line("float armigo_depth_offset = armigo_depth_bias.x;");
            line("if (armigo_depth_bias.y != 0.0f) {");
            ++indent_;
            line("const float armigo_depth_slope = max(abs(dfdx(armigo_z)), abs(dfdy(armigo_z)));");
            line("armigo_depth_offset += armigo_depth_bias.y * armigo_depth_slope;");
            --indent_;
            line("}");
            line("if (armigo_depth_bias.z > 0.0f) armigo_depth_offset = min(armigo_depth_offset, armigo_depth_bias.z);");
            line("else if (armigo_depth_bias.z < 0.0f) armigo_depth_offset = max(armigo_depth_offset, armigo_depth_bias.z);");
            line("out.armigo_depth = armigo_depth_bias.w != 0.0f ? clamp(armigo_z + armigo_depth_offset, 0.0f, 1.0f) : armigo_z;");
            if (input_position) line("in.position.z = out.armigo_depth;");
            --indent_;
            line("}");
        }
        if (options_.fragment_output_types_from_render_targets) {
            std::set<std::uint16_t> locations;
            for (const auto& io : plan_.io) {
                if (io.access == Access::write && io.semantic == IoSemantic::color &&
                    locations.insert(io.location).second)
                    line("uint4 armigo_" + io_name(io.semantic, io.location) + "_raw = uint4(0);");
            }
        }
        if (plan_.compute && plan_.compute->shared_memory_bytes)
            line("threadgroup atomic_uint armigo_shared_words[" + number(plan_.compute->shared_memory_bytes / 4) + "];");
        // armigo-local-zero-init-v1: the allocation fact sizes the per-thread
        // array indexed by LDL/STL; the zero fill keeps a load before any store
        // defined (Maxwell leaves guest local memory uninitialized).
        for (const auto& dependency : plan_.dependencies)
            if (dependency.key.kind == FactKind::local_memory_bytes) {
                require(std::holds_alternative<std::uint32_t>(dependency.value),
                        "Local memory allocation dependency has an invalid value", DiagnosticCode::invalid_fact);
                if (const auto words = (std::get<std::uint32_t>(dependency.value) + 3u) / 4u)
                    line("thread uint armigo_local_words[" + number(words) + "] = {};");
            }
        // armigo-output-init-v1: entry values for every covered output component,
        // so an early exit observes position (0, 0, 0, 1) and zero elsewhere.
        if (ir.stage == Stage::vertex)
            for (const auto& io : plan_.io) if (io.access == Access::write)
                for (unsigned component = 0; component < 4; ++component) {
                    if (!(io.mask & (1u << component)) ||
                        !initialized_outputs_.contains(output_key(io.semantic, io.location, component))) continue;
                    const auto target = "out." + io_name(io.semantic, io.location) +
                        (io.semantic == IoSemantic::generic ? "_" + number(component) :
                         io.type.components > 1 ? std::string(".") + std::string(1, "xyzw"[component]) : std::string());
                    // Position and generic vectors start as (0, 0, 0, 1); a
                    // non-float generic keeps the bits of 1.0f.
                    const bool vector = io.semantic == IoSemantic::position || io.semantic == IoSemantic::generic;
                    const auto value = !vector || component != 3 ? scalar(io.type.scalar) + "(0)" :
                        io.type.scalar == ScalarType::f32 ? std::string("1.0f") :
                        "as_type<" + scalar(io.type.scalar) + ">(1.0f)";
                    line(target + " = " + value + ";");
                }
        // The synthesized size is one whole-output constant write at entry,
        // which covers every return path like the initialized outputs above.
        if (ir.stage == Stage::vertex && fixed_point_size_) {
            line("out." + io_name(IoSemantic::point_size, 0) + " = 1.0f;");
            initialized_outputs_.insert(output_key(IoSemantic::point_size, 0, 0));
        }
        for (const auto& io : plan_.io) if (io.access == Access::read && io.semantic == IoSemantic::instance_id)
            line("const uint " + io_name(io.semantic, io.location) + " = armigo_instance_index - armigo_guest_first_instance;");
        // Owner decision: every value starts at zero/false, so a read whose
        // definition does not reach every structured path observes zero instead
        // of an uninitialized MSL local.
        for (std::size_t i = 0; i < ir.values.size(); ++i) line(raw_type(ir.values[i].type) + " value" + number(i) + " = {};");
        region(p_.functions[0].root);
        require(regions_.size() == p_.regions.size() && blocks_.size() == ir.blocks.size() && operations_.size() == ir.operations.size(),
                "Structured traversal does not cover every region/block/operation");
        if (flow_.reachable) {
            const auto proof = definedness_->fallthrough();
            live_ = proof.reachable; flow_.defined = proof.defined; flow_.outputs = proof.outputs;
            emit_return();
        }
        append("}\n");
        // Relaxed math with fast contraction is the native Metal arithmetic.
        // The guest rules it would lose stay explicit in the emitted code:
        // NaN tests on the bits, precise extrema and saturation, and the
        // flush steps of sine, cosine, same-type moves and operations on
        // literals alone (the extrema's departures are documented there).
        std::string prelude = "#include <metal_stdlib>\nusing namespace metal;\n#pragma METAL fp math_mode(relaxed)\n#pragma METAL fp contract(fast)\n";
        require(prelude.size() + helpers_.size() + text_.size() <= ir.limits.source_bytes, "MSL source limit exceeded", DiagnosticCode::limit_exceeded);
        require(helpers_.find(" " + entry + "(") == std::string::npos, "Entry point collides with an emitted helper");
        auto reflection = plan_;
        reflection.fragment_position = options_.fragment_position;
        reflection.emulate_depth_bias = options_.emulate_depth_bias;
        reflection.fixed_point_size = fixed_point_size_;
        reflection.reads_point_coord = std::any_of(plan_.io.begin(), plan_.io.end(),
            [](const auto& io) { return io.access == Access::read && io.semantic == IoSemantic::point_coord; });
        return {prelude + helpers_ + text_, entry, std::move(reflection)};
    }
};
}
Result<MslArtifact> emit_msl(const StructuredProgram& program, const ResourcePlan& plan, const EmitOptions& options) {
    try {
        const auto sampling_plan = msl::plan_custom_border(program, plan, options);
        return StageContext(program, sampling_plan, options).emit();
    }
    catch (const msl::Failure& failure) { return failure.diagnostic; }
    catch (const std::bad_alloc&) { return diagnose(Phase::emit_msl, DiagnosticCode::limit_exceeded, {}, "MSL allocation failed"); }
}
} // namespace armigo::shader::pipeline
