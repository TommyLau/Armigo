// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Exclusive and ordered instruction families have distinct handling. No
// injected traffic is allowed between an exclusive load and its closing store.
// Armigo follows the reached CFG instead of a linear segment. Whole-window
// register substitution retains native monitor behavior; no emulated monitor
// or CAS lowering is used.

#include <armigo/execution/prepared_image.h>

#include <armigo/aarch64/instruction.h>
#include <armigo/aarch64/register_effects.h>
#include <armigo/aarch64/register_substitution.h>
#include <armigo/aarch64/relocation.h>
#include <armigo/execution/conditional_x18_patch.h>
#include <armigo/execution/gate_code.h>
#include <armigo/execution/gate_patch.h>
#include <armigo/execution/indirect_gate_patch.h>
#include <armigo/execution/indirect_x18_patch.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <utility>

namespace armigo::execution {
namespace {

using aarch64::EffectStatus;
using aarch64::Opcode;
using aarch64::Operation;

constexpr std::uint64_t kPlanningBase = 0x1'0000'0000ull;
constexpr std::uint64_t kAppendixAlignment = 16 * 1024;
constexpr std::uint64_t kMaximum = std::numeric_limits<std::uint64_t>::max();

bool add_overflows(std::uint64_t left, std::uint64_t right) noexcept {
    return right > kMaximum - left;
}

bool location_equal(PreparedLocation left, PreparedLocation right) noexcept {
    return left.module_index == right.module_index && left.image_offset == right.image_offset;
}

bool valid_location(PreparedLocation location, const PreparedImageOptions& options,
                    std::uint64_t local_size) noexcept {
    return location.module_index < options.module_count && location.image_offset % 4 == 0 &&
           (location.module_index != options.module_index || location.image_offset < local_size);
}

bool is_native_system_register(const aarch64::Instruction& decoded) noexcept {
    if ((decoded.opcode != Opcode::mrs && decoded.opcode != Opcode::msr_register) ||
        decoded.reg == aarch64::platform_register)
        return false;
    const auto system = aarch64::identify_system_register(decoded.system_register);
    return system == aarch64::SystemRegister::nzcv || system == aarch64::SystemRegister::fpcr ||
           system == aarch64::SystemRegister::fpsr;
}

bool is_gate_system(const aarch64::Instruction& decoded) noexcept {
    return decoded.opcode == Opcode::svc || decoded.opcode == Opcode::cache_maintenance ||
           ((decoded.opcode == Opcode::mrs || decoded.opcode == Opcode::msr_register) &&
            !is_native_system_register(decoded));
}

bool is_conditional_x18(const aarch64::Instruction& decoded) noexcept {
    return decoded.reg == aarch64::platform_register &&
           (decoded.opcode == Opcode::cbz || decoded.opcode == Opcode::cbnz ||
            decoded.opcode == Opcode::tbz || decoded.opcode == Opcode::tbnz);
}

bool needs_gate(const CoveredInstruction& covered) noexcept {
    const auto decoded = aarch64::decode(covered.word);
    return is_gate_system(decoded) ||
           ((decoded.opcode == Opcode::br || decoded.opcode == Opcode::blr) &&
            decoded.reg != aarch64::platform_register);
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

template <typename WordRange>
std::uint64_t append_words(PreparedImage& image, const WordRange& words, std::size_t count) {
    const auto offset = image.appendix_offset + image.appendix.size();
    const auto bytes = count * sizeof(std::uint32_t);
    const auto old_size = image.appendix.size();
    image.appendix.resize(old_size + bytes);
    std::memcpy(image.appendix.data() + old_size, words.data(), bytes);
    return offset;
}

std::uint64_t append_bytes(PreparedImage& image, std::span<const std::byte> bytes) {
    const auto offset = image.appendix_offset + image.appendix.size();
    image.appendix.insert(image.appendix.end(), bytes.begin(), bytes.end());
    return offset;
}

void add_patch(PreparedImage& image, const CoveredInstruction& covered, std::uint32_t replacement) {
    image.replacements.push_back({covered.image_offset, covered.word, replacement});
}

PreparedImageResult fail(PreparedImageError error, std::size_t index = 0) {
    return {.error = error, .instruction_index = index};
}

bool reads_or_writes_x18(const aarch64::RegisterEffects& effects) noexcept {
    return ((effects.reads | effects.writes) & aarch64::register_bit(aarch64::platform_register)) !=
           0;
}

bool fixed_width_mov_sequence(std::span<const std::byte> bytes, std::uint64_t offset) noexcept {
    if (offset % 4 != 0 || offset > bytes.size() || bytes.size() - offset < 16)
        return false;
    std::uint32_t words[4];
    std::memcpy(words, bytes.data() + offset, sizeof(words));
    const unsigned reg = words[0] & 31u;
    for (unsigned half = 0; half != 4; ++half) {
        const std::uint32_t opcode = half == 0 ? 0xd2800000u : 0xf2800000u;
        const std::uint32_t mask = 0xffe0001fu;
        if ((words[half] & mask) != (opcode | (half << 21) | reg))
            return false;
    }
    return true;
}

bool thread_slot_load(std::uint32_t word) noexcept {
    // LDR Xt,[Xn,#imm12*8], with the Rt/Rn fields intentionally unrestricted.
    return (word & 0xffc00000u) == 0xf9400000u;
}

bool branch26(std::uint32_t word) noexcept { return (word & 0x7c000000u) == 0x14000000u; }

std::optional<std::uint32_t> planned_word(const PreparedImage& prepared,
                                          std::uint64_t offset) noexcept {
    if (offset >= prepared.appendix_offset) {
        const auto relative = offset - prepared.appendix_offset;
        if (relative > prepared.appendix.size() || prepared.appendix.size() - relative < 4) {
            return std::nullopt;
        }
        std::uint32_t word;
        std::memcpy(&word, prepared.appendix.data() + relative, sizeof(word));
        return word;
    }
    const auto it = std::lower_bound(
        prepared.replacements.begin(), prepared.replacements.end(), offset,
        [](const PreparedWordPatch& patch, std::uint64_t at) { return patch.image_offset < at; });
    if (it == prepared.replacements.end() || it->image_offset != offset)
        return std::nullopt;
    return it->replacement_word;
}

// Boundary instructions execute with the original SP and register state.
// Complete non-PC-relative effects suffice for copying scalar instructions
// that the smaller control-flow classifier intentionally leaves unclassified.
std::optional<std::uint32_t> move_boundary_word(std::uint32_t word, std::uint64_t from,
                                               std::uint64_t to) {
    const auto instruction = aarch64::decode(word);
    if (!aarch64::is_pc_relative(instruction.opcode)) return word;
    const auto moved = aarch64::relocate_instruction(word, from, to);
    if (!moved.ok() || moved.link_value_changed) return std::nullopt;
    return moved.word;
}

std::optional<std::uint32_t> redirect_branch(std::uint32_t word, std::uint64_t from,
                                           std::uint64_t target) {
    const auto instruction = aarch64::decode(word);
    aarch64::Encoded encoded;
    switch (instruction.opcode) {
    case Opcode::b: encoded = aarch64::encode_b(from, target); break;
    case Opcode::b_cond: case Opcode::bc_cond:
        encoded = aarch64::encode_b_cond(instruction.condition, from, target,
                                         instruction.opcode == Opcode::bc_cond); break;
    case Opcode::cbz: case Opcode::cbnz:
        encoded = aarch64::encode_cbz(instruction.opcode == Opcode::cbnz, instruction.wide,
                                      instruction.reg, from, target); break;
    case Opcode::tbz: case Opcode::tbnz:
        encoded = aarch64::encode_tbz(instruction.opcode == Opcode::tbnz, instruction.reg,
                                      instruction.test_bit, from, target); break;
    default: return std::nullopt;
    }
    if (!encoded.ok()) return std::nullopt;
    return encoded.word;
}

PreparedImageApplyResult apply_fail(PreparedImageError error, std::size_t index = 0) noexcept {
    return {.error = error, .record_index = index};
}

} // namespace

const char* exclusive_window_error_name(ExclusiveWindowError error) noexcept {
    switch (error) {
    case ExclusiveWindowError::none: return "none";
    case ExclusiveWindowError::invalid_coverage: return "invalid-coverage";
    case ExclusiveWindowError::x18_operand: return "x18-exclusive-operand";
    case ExclusiveWindowError::prepared_rewrite: return "prepared-rewrite";
    case ExclusiveWindowError::unclosed_exit: return "unclosed-exit";
    case ExclusiveWindowError::function_boundary: return "function-boundary";
    case ExclusiveWindowError::uncovered_successor: return "uncovered-successor";
    case ExclusiveWindowError::unsupported_instruction: return "unsupported-instruction";
    case ExclusiveWindowError::window_limit: return "window-limit";
    case ExclusiveWindowError::word_limit: return "word-limit";
    case ExclusiveWindowError::step_limit: return "step-limit";
    case ExclusiveWindowError::no_scratch_register: return "no-scratch-register";
    case ExclusiveWindowError::entry_transfer: return "entry-transfer";
    case ExclusiveWindowError::exit_transfer: return "exit-transfer";
    case ExclusiveWindowError::substitution_failed: return "substitution-failed";
    }
    return "unknown";
}

ExclusiveWindowProof prove_exclusive_windows(
    std::span<const CoveredInstruction> covered, const ExclusiveWindowLimits& limits,
    std::span<const std::uint64_t> function_entries, std::uint32_t module_index,
    std::span<const std::uint64_t> external_entries) {
    ExclusiveWindowProof proof;
    proof.monitor_clear.resize(covered.size(), true);
    struct Decoded {
        aarch64::Instruction instruction;
        aarch64::RegisterEffects effects;
        bool exclusive;
        bool load;
    };
    std::vector<Decoded> decoded;
    decoded.reserve(covered.size());
    std::vector<std::size_t> loads;
    std::set<std::uint64_t> entries(function_entries.begin(), function_entries.end());
    for (std::size_t index = 0; index < covered.size(); ++index) {
        const auto& site = covered[index];
        if (site.image_offset % 4 || site.image_offset > kMaximum - 4 ||
            (index && covered[index - 1].image_offset >= site.image_offset)) {
            proof.diagnostics.push_back({ExclusiveWindowError::invalid_coverage, {}, site.image_offset});
            return proof;
        }
        const auto instruction = aarch64::decode(site.word);
        const auto effects = aarch64::decode_register_effects(site.word);
        if (!effects.decoded()) {
            proof.windows_found = loads.size();
            proof.diagnostics.push_back({ExclusiveWindowError::unsupported_instruction, {}, site.image_offset});
            return proof;
        }
        const bool exclusive = aarch64::has(effects.memory.attributes, aarch64::MemoryAttribute::exclusive);
        const bool load = exclusive && effects.memory.operation == aarch64::MemoryOperation::load;
        decoded.push_back({instruction, effects, exclusive, load});
        if (load) loads.push_back(index);
        if (instruction.opcode == Opcode::bl && !site.cross_module_target) {
            if (const auto target = aarch64::referenced_address(instruction, site.image_offset))
                entries.insert(*target);
        }
        for (const auto& target : site.indirect_targets)
            if ((instruction.opcode == Opcode::blr || instruction.opcode == Opcode::br) &&
                target.prepared.module_index == module_index)
                entries.insert(target.prepared.image_offset);
    }
    proof.windows_found = loads.size();
    if (loads.size() > limits.max_windows) {
        const auto at = covered[loads[limits.max_windows]].image_offset;
        proof.diagnostics.push_back({ExclusiveWindowError::window_limit, at, at});
        return proof;
    }

    // Generation marks bound memory to the coverage size even with many
    // overlapping/nested windows. A backedge is visited once per opening load.
    std::vector<std::size_t> visited(covered.size(), 0);
    std::vector<std::vector<std::size_t>> windows(loads.size());
    for (std::size_t window = 0; window < loads.size(); ++window) {
        const auto start = covered[loads[window]].image_offset;
        const auto generation = window + 1;
        std::vector<std::size_t> pending{loads[window]};
        visited[loads[window]] = generation;
        std::size_t words = 0;
        bool refused = false;
        const auto refuse = [&](ExclusiveWindowError reason, std::uint64_t at) {
            if (!refused) proof.diagnostics.push_back({reason, start, at});
            refused = true;
        };
        const auto enqueue = [&](std::uint64_t offset) {
            if (offset != start && entries.contains(offset)) {
                refuse(ExclusiveWindowError::function_boundary, offset);
                return;
            }
            const auto found = std::lower_bound(covered.begin(), covered.end(), offset,
                [](const CoveredInstruction& site, std::uint64_t value) { return site.image_offset < value; });
            if (found == covered.end() || found->image_offset != offset) {
                refuse(ExclusiveWindowError::uncovered_successor, offset);
                return;
            }
            const auto index = static_cast<std::size_t>(found - covered.begin());
            if (visited[index] != generation) {
                visited[index] = generation;
                pending.push_back(index);
            }
        };
        while (!pending.empty() && !refused) {
            const auto index = pending.back();
            pending.pop_back();
            const auto& site = covered[index];
            const auto& node = decoded[index];
            if (words == limits.max_words_per_window) {
                refuse(ExclusiveWindowError::word_limit, site.image_offset);
                break;
            }
            if (proof.steps == limits.max_total_steps) {
                refuse(ExclusiveWindowError::step_limit, site.image_offset);
                return proof;
            }
            ++words;
            ++proof.steps;
            proof.monitor_clear[index] = false;
            windows[window].push_back(index);
            if (!node.effects.decoded()) {
                refuse(ExclusiveWindowError::unsupported_instruction, site.image_offset);
                break;
            }
            // Register-only substitution can cover x18 uses, but runtime gates
            // cannot execute while a reservation may be live.
            if (needs_gate(site)) {
                refuse(ExclusiveWindowError::prepared_rewrite, site.image_offset);
                break;
            }
            if (node.effects.operation == Operation::clrex ||
                (node.exclusive && !node.load)) continue;
            switch (node.instruction.opcode) {
            case Opcode::bl: case Opcode::blr: case Opcode::ret: case Opcode::retab:
                refuse(ExclusiveWindowError::unclosed_exit, site.image_offset);
                break;
            case Opcode::b: case Opcode::b_cond: case Opcode::bc_cond:
            case Opcode::cbz: case Opcode::cbnz: case Opcode::tbz: case Opcode::tbnz: {
                const auto target = aarch64::referenced_address(node.instruction, site.image_offset);
                if (site.cross_module_target || !target) {
                    refuse(ExclusiveWindowError::function_boundary, site.image_offset);
                    break;
                }
                enqueue(*target);
                if (node.instruction.opcode != Opcode::b &&
                    !((node.instruction.opcode == Opcode::b_cond || node.instruction.opcode == Opcode::bc_cond) &&
                      (node.instruction.condition == aarch64::Condition::al ||
                       node.instruction.condition == aarch64::Condition::nv)))
                    enqueue(site.image_offset + 4);
                break;
            }
            default:
                if (node.effects.control_flow != aarch64::ControlFlow::none)
                    refuse(ExclusiveWindowError::unclosed_exit, site.image_offset);
                else
                    enqueue(site.image_offset + 4);
                break;
            }
        }
        if (!refused) ++proof.windows_proved;
    }
    // A store-exclusive can be reached without a preceding load (and fail
    // natively). Its register operands still must not expose physical x18.
    for (std::size_t index = 0; index < covered.size(); ++index) {
        if (decoded[index].exclusive && !decoded[index].load && proof.monitor_clear[index] &&
            reads_or_writes_x18(decoded[index].effects))
            proof.diagnostics.push_back({ExclusiveWindowError::x18_operand, {}, covered[index].image_offset});
    }
    if (!proof.ok() || loads.empty()) return proof;

    // Merge overlapping proofs, including nested loads and shared closures.
    // A component must use one physical register on every incoming path.
    const auto absent = covered.size();
    std::vector<std::size_t> parent(loads.size());
    for (std::size_t i = 0; i < parent.size(); ++i) parent[i] = i;
    const auto root = [&](std::size_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    std::vector<std::size_t> owner(covered.size(), loads.size());
    for (std::size_t w = 0; w < windows.size(); ++w) {
        for (const auto i : windows[w]) {
            if (owner[i] != loads.size()) parent[root(w)] = root(owner[i]);
            owner[i] = w;
        }
    }
    std::vector<std::vector<std::size_t>> components(loads.size());
    for (std::size_t i = 0; i < covered.size(); ++i)
        if (owner[i] != loads.size()) components[root(owner[i])].push_back(i);
    if (std::none_of(components.begin(), components.end(), [&](const auto& component) {
            return std::any_of(component.begin(), component.end(), [&](auto i) {
                return reads_or_writes_x18(decoded[i].effects);
            });
        })) return proof;

    // External tail branches are ingress barriers for register substitution,
    // without changing the admission of windows that remain entirely native.
    entries.insert(external_entries.begin(), external_entries.end());
    const auto find_index = [&](std::uint64_t offset) {
        const auto it = std::lower_bound(covered.begin(), covered.end(), offset,
            [](const auto& site, auto at) { return site.image_offset < at; });
        return it != covered.end() && it->image_offset == offset
            ? static_cast<std::size_t>(it - covered.begin()) : absent;
    };
    std::vector<std::vector<std::size_t>> successors(covered.size()), predecessors(covered.size());
    std::vector<bool> opaque(covered.size());
    for (std::size_t i = 0; i < covered.size(); ++i) {
        const auto& instruction = decoded[i].instruction;
        const auto edge = [&](std::uint64_t offset) {
            const auto target = find_index(offset);
            if (target == absent) opaque[i] = true;
            else {
                successors[i].push_back(target);
                predecessors[target].push_back(i);
            }
        };
        switch (instruction.opcode) {
        case Opcode::b: case Opcode::b_cond: case Opcode::bc_cond:
        case Opcode::cbz: case Opcode::cbnz: case Opcode::tbz: case Opcode::tbnz: {
            const auto target = aarch64::referenced_address(instruction, covered[i].image_offset);
            if (!target || covered[i].cross_module_target) opaque[i] = true;
            else edge(*target);
            if (instruction.opcode != Opcode::b &&
                !((instruction.opcode == Opcode::b_cond || instruction.opcode == Opcode::bc_cond) &&
                  (instruction.condition == aarch64::Condition::al ||
                   instruction.condition == aarch64::Condition::nv)))
                edge(covered[i].image_offset + 4);
            break;
        }
        default:
            if (decoded[i].effects.control_flow != aarch64::ControlFlow::none || needs_gate(covered[i])) {
                opaque[i] = true;
                // Calls and system gates may resume at their continuation.
                // They remain opaque to liveness, but their incoming edge must
                // not disappear when another path already reaches that word.
                if (instruction.opcode == Opcode::bl || instruction.opcode == Opcode::blr ||
                    is_gate_system(instruction)) edge(covered[i].image_offset + 4);
            } else edge(covered[i].image_offset + 4);
            break;
        }
    }
    std::vector<bool> boundary_used(covered.size());
    for (const auto& component : components) {
        if (component.empty() || std::none_of(component.begin(), component.end(), [&](auto i) {
                return reads_or_writes_x18(decoded[i].effects);
            })) continue;
        const auto origin = covered[loads[root(owner[component.front()])]].image_offset;
        const auto refuse = [&](ExclusiveWindowError reason, std::size_t i) {
            proof.diagnostics.push_back({reason, origin, covered[i].image_offset});
        };
        ExclusiveX18Region region;
        region.instructions = component;
        const auto inside = [&](std::size_t i) {
            return owner[i] != loads.size() && root(owner[i]) == root(owner[component.front()]);
        };
        for (const auto i : component) {
            if (entries.contains(covered[i].image_offset)) {
                refuse(ExclusiveWindowError::entry_transfer, i);
                return proof;
            }
            bool incoming = predecessors[i].empty();
            for (const auto predecessor : predecessors[i]) incoming |= !inside(predecessor);
            if (incoming) {
                // Fallthrough uses the preceding native word as a patch slot.
                // Direct incoming branches bypass that word and enter the
                // transfer itself; internal backedges remain untouched.
                if (!decoded[i].load || i == 0 || covered[i - 1].image_offset + 4 != covered[i].image_offset ||
                    !proof.monitor_clear[i - 1] || boundary_used[i - 1] ||
                    decoded[i - 1].effects.control_flow != aarch64::ControlFlow::none ||
                    needs_gate(covered[i - 1]) || reads_or_writes_x18(decoded[i - 1].effects) ||
                    decoded[i - 1].exclusive) {
                    refuse(ExclusiveWindowError::entry_transfer, i);
                    return proof;
                }
                for (const auto predecessor : predecessors[i]) {
                    if (!inside(predecessor) && predecessor != i - 1 &&
                        ((reads_or_writes_x18(decoded[predecessor].effects) &&
                          !is_conditional_x18(decoded[predecessor].instruction)) || needs_gate(covered[predecessor]))) {
                        refuse(ExclusiveWindowError::entry_transfer, predecessor);
                        return proof;
                    }
                    if (!inside(predecessor) && predecessor != i - 1)
                        region.entry_branches.push_back({predecessor, i});
                }
                region.entries.push_back(i);
            }
            if (decoded[i].effects.operation == Operation::clrex ||
                (decoded[i].exclusive && !decoded[i].load)) {
                const auto exit = find_index(covered[i].image_offset + 4);
                if (exit == absent || !proof.monitor_clear[exit] || boundary_used[exit] ||
                    entries.contains(covered[exit].image_offset)) {
                    refuse(ExclusiveWindowError::exit_transfer, i);
                    return proof;
                }
                // A shared continuation could otherwise export an unrelated
                // scratch value to the guest register cell.
                for (const auto predecessor : predecessors[exit]) {
                    if (!inside(predecessor)) {
                        refuse(ExclusiveWindowError::exit_transfer, exit);
                        return proof;
                    }
                }
                region.exits.push_back(exit);
            }
        }
        if (region.entries.empty()) {
            refuse(ExclusiveWindowError::entry_transfer, component.front());
            return proof;
        }
        bool found_scratch = false;
        for (unsigned reg = 0; reg < 31 && !found_scratch; ++reg) {
            if (reg == aarch64::platform_register || std::any_of(component.begin(), component.end(), [&](auto i) {
                    return !aarch64::proven_free_of(decoded[i].effects, reg);
                })) continue;
            const auto bit = aarch64::register_bit(reg);
            std::vector<bool> seen(covered.size());
            std::vector<std::size_t> pending(region.exits);
            bool dead = true;
            while (!pending.empty() && dead) {
                const auto i = pending.back();
                pending.pop_back();
                if (seen[i]) continue;
                seen[i] = true;
                if (proof.steps == limits.max_total_steps) {
                    refuse(ExclusiveWindowError::step_limit, i);
                    return proof;
                }
                ++proof.steps;
                const auto& effects = decoded[i].effects;
                if ((effects.reads & bit) || aarch64::has(effects.notes, aarch64::EffectNote::handler_defined_registers) ||
                    effects.control_flow == aarch64::ControlFlow::call ||
                    effects.control_flow == aarch64::ControlFlow::indirect_call ||
                    effects.control_flow == aarch64::ControlFlow::return_ || needs_gate(covered[i]) ||
                    entries.contains(covered[i].image_offset)) {
                    dead = false;
                    break;
                }
                bool killed = false;
                for (std::size_t operand = 0; operand < effects.operand_count; ++operand) {
                    const auto& field = effects.operands[operand];
                    killed |= field.names_general_register(reg) && aarch64::includes_write(field.access) &&
                              !field.conditional;
                }
                if (killed) continue;
                if (opaque[i]) {
                    dead = false;
                    break;
                }
                pending.insert(pending.end(), successors[i].begin(), successors[i].end());
            }
            if (dead) {
                region.scratch_register = static_cast<std::uint8_t>(reg);
                found_scratch = true;
            }
        }
        if (!found_scratch) {
            refuse(ExclusiveWindowError::no_scratch_register, component.front());
            return proof;
        }
        for (const auto i : component) {
            if (!aarch64::substitute_register(covered[i].word, 18, region.scratch_register,
                                             aarch64::UnchangedWord::accept).ok()) {
                refuse(ExclusiveWindowError::substitution_failed, i);
                return proof;
            }
        }
        for (const auto i : region.entries) boundary_used[i - 1] = true;
        for (const auto i : region.exits) {
            if (boundary_used[i]) {
                refuse(ExclusiveWindowError::exit_transfer, i);
                return proof;
            }
            boundary_used[i] = true;
        }
        proof.x18_regions.push_back(std::move(region));
    }
    return proof;
}

PreparedImageResult prepare_image(std::uint64_t original_size,
                                  std::span<const CoveredInstruction> covered,
                                  const PreparedImageOptions& options) {
    static_assert(std::endian::native == std::endian::little);
    if (options.module_count == 0 || options.module_index >= options.module_count) {
        return fail(PreparedImageError::invalid_options);
    }
    if (!options.scalar_x18_policy.stack_scratch_available ||
        !options.scalar_x18_policy.runtime_storage_disjoint) {
        return fail(PreparedImageError::missing_execution_contract);
    }
    if (original_size == 0 || original_size > kMaximum - (kAppendixAlignment - 1) ||
        add_overflows(kPlanningBase, original_size)) {
        return fail(PreparedImageError::invalid_image_size);
    }

    PreparedImage image;
    image.module_index = options.module_index;
    image.original_image_size = original_size;
    image.appendix_offset = align_up(original_size, kAppendixAlignment);
    try {
        image.coverage.reserve(covered.size());
        image.replacements.reserve(covered.size());
        std::uint64_t previous = 0;
        for (std::size_t index = 0; index < covered.size(); ++index) {
            const auto& site = covered[index];
            if (site.image_offset % 4 != 0 || site.image_offset > original_size ||
                original_size - site.image_offset < 4 ||
                (index != 0 && site.image_offset <= previous)) {
                return fail(PreparedImageError::invalid_coverage, index);
            }
            previous = site.image_offset;
            image.coverage.push_back({site.image_offset, site.word, site.word});
        }

        auto proof = prove_exclusive_windows(covered, options.exclusive_limits,
                                             options.function_entries, options.module_index, options.external_entries);
        if (!proof.ok()) {
            const auto& diagnostic = proof.diagnostics.front();
            const auto found = std::lower_bound(covered.begin(), covered.end(), diagnostic.instruction_offset,
                [](const CoveredInstruction& site, std::uint64_t at) { return site.image_offset < at; });
            const auto error = diagnostic.reason == ExclusiveWindowError::x18_operand
                ? PreparedImageError::exclusive_instruction
                : diagnostic.reason == ExclusiveWindowError::unsupported_instruction
                    ? PreparedImageError::unsupported_instruction : PreparedImageError::exclusive_window;
            auto result = fail(error, static_cast<std::size_t>(found - covered.begin()));
            result.exclusive_proof = std::move(proof);
            return result;
        }

        std::vector<unsigned> window_register(covered.size(), 31);
        std::vector<std::optional<std::uint64_t>> conditional_taken_site(covered.size());
        for (const auto& region : proof.x18_regions)
            for (const auto i : region.instructions) window_register[i] = region.scratch_register;

        std::uint64_t gate_address = 0;
        std::uint64_t gate_size = 0;
        if (std::any_of(covered.begin(), covered.end(), needs_gate)) {
            auto gate = make_local_gate_template();
            gate_address = append_bytes(image, gate.code);
            gate_size = gate.code.size();
            image.fixups.push_back({PreparedFixupKind::gate_dispatcher_address,
                                    gate_address + gate.dispatcher_offset,
                                    {}});
        }

        for (std::size_t index = 0; index < covered.size(); ++index) {
            const auto& site = covered[index];
            const auto decoded = aarch64::decode(site.word);
            const auto effects = aarch64::decode_register_effects(site.word);
            const auto address = kPlanningBase + site.image_offset;
            auto site_policy = options.scalar_x18_policy;
            site_policy.exclusive_monitor_clear = proof.monitor_clear[index];
            if (aarch64::pointer_authentication(site.word) == aarch64::PointerAuthentication::unsupported)
                return fail(PreparedImageError::unsupported_instruction, index);
            if (decoded.opcode == Opcode::brk || decoded.opcode == Opcode::hlt ||
                decoded.opcode == Opcode::hvc || decoded.opcode == Opcode::smc) {
                return fail(PreparedImageError::unsupported_instruction, index);
            }
            if (site.cross_module_target) {
                if ((decoded.opcode != Opcode::b && decoded.opcode != Opcode::bl) ||
                    !site.indirect_targets.empty() ||
                    !valid_location(*site.cross_module_target, options, original_size) ||
                    site.cross_module_target->module_index == options.module_index) {
                    return fail(PreparedImageError::invalid_target, index);
                }
                const auto placeholder = decoded.opcode == Opcode::bl
                                             ? aarch64::encode_bl(address, address)
                                             : aarch64::encode_b(address, address);
                if (!placeholder.ok())
                    return fail(PreparedImageError::patch_failed, index);
                add_patch(image, site, placeholder.word);
                image.fixups.push_back({PreparedFixupKind::branch26_module_target,
                                        site.image_offset, *site.cross_module_target});
                continue;
            }

            if (decoded.opcode == Opcode::br || decoded.opcode == Opcode::blr) {
                if (site.indirect_targets.empty()) {
                    return fail(PreparedImageError::missing_targets, index);
                }
                if (site.indirect_targets.size() > (kMaximum - kPlanningBase) / 4 - 1) {
                    return fail(PreparedImageError::layout_overflow, index);
                }
                std::vector<PreparedLocation> guest_targets;
                guest_targets.reserve(site.indirect_targets.size());
                for (const auto& target : site.indirect_targets) {
                    if (!valid_location(target.guest, options, original_size) ||
                        !valid_location(target.prepared, options, original_size)) {
                        return fail(PreparedImageError::invalid_target, index);
                    }
                    guest_targets.push_back(target.guest);
                }
                std::sort(guest_targets.begin(), guest_targets.end(),
                          [](PreparedLocation left, PreparedLocation right) {
                              if (left.module_index != right.module_index)
                                  return left.module_index < right.module_index;
                              return left.image_offset < right.image_offset;
                          });
                if (std::adjacent_find(guest_targets.begin(), guest_targets.end(),
                                       location_equal) != guest_targets.end()) {
                    return fail(PreparedImageError::invalid_target, index);
                }
                if (decoded.reg == aarch64::platform_register) {
                    std::vector<IndirectX18Target> targets;
                    targets.reserve(site.indirect_targets.size());
                    for (std::size_t target_index = 0; target_index < site.indirect_targets.size();
                         ++target_index) {
                        const auto& target = site.indirect_targets[target_index];
                        const auto guest_placeholder = kPlanningBase + 4 * (target_index + 1);
                        const auto prepared_placeholder =
                            target.prepared.module_index == options.module_index
                                ? kPlanningBase + target.prepared.image_offset
                                : address;
                        targets.push_back({guest_placeholder, prepared_placeholder});
                    }
                    const auto trampoline = image.appendix_offset + image.appendix.size();
                    const auto plan =
                        make_indirect_x18_patch(site.word, address, kPlanningBase + trampoline, 0,
                                                targets, site_policy);
                    if (!plan.ok())
                        return fail(PreparedImageError::patch_failed, index);
                    const auto at =
                        append_words(image, plan.patch->trampoline, plan.patch->trampoline.size());
                    add_patch(image, site, plan.patch->replacement_instruction);
                    image.fixups.push_back({PreparedFixupKind::thread_slot_load, at + 8, {}});
                    for (std::size_t target_index = 0; target_index < site.indirect_targets.size();
                         ++target_index) {
                        const auto& target = site.indirect_targets[target_index];
                        image.fixups.push_back({PreparedFixupKind::module_address_mov_wide,
                                                at + 4 * (4 + 8 * target_index), target.guest});
                        if (target.prepared.module_index != options.module_index) {
                            image.fixups.push_back({PreparedFixupKind::branch26_module_target,
                                                    at + 4 * (11 + 8 * target_index),
                                                    target.prepared});
                        }
                    }
                } else {
                    std::vector<std::uint64_t> targets;
                    targets.reserve(site.indirect_targets.size());
                    for (std::size_t target_index = 0; target_index < site.indirect_targets.size();
                         ++target_index) {
                        const auto& target = site.indirect_targets[target_index];
                        if (!location_equal(target.guest, target.prepared)) {
                            return fail(PreparedImageError::invalid_target, index);
                        }
                        targets.push_back(kPlanningBase + 4 * (target_index + 1));
                    }
                    const auto trampoline = image.appendix_offset + image.appendix.size();
                    const auto plan = make_indirect_gate_site_patch(
                        site.word, address, kPlanningBase + trampoline,
                        {kPlanningBase + gate_address, gate_size}, targets);
                    if (!plan.ok())
                        return fail(PreparedImageError::patch_failed, index);
                    const auto at =
                        append_words(image, plan.patch->trampoline, plan.patch->trampoline.size());
                    add_patch(image, site, plan.patch->replacement_instruction);
                    PreparedGateSite gate_site{at + 8, site.word, {}};
                    for (const auto& target : site.indirect_targets) {
                        gate_site.allowed_indirect_targets.push_back(target.prepared);
                    }
                    image.gate_sites.push_back(std::move(gate_site));
                }
                continue;
            }

            if (!site.indirect_targets.empty()) {
                return fail(PreparedImageError::unexpected_targets, index);
            }
            if (window_register[index] != 31) {
                const auto renamed = aarch64::substitute_register(site.word, 18, window_register[index],
                                                                  aarch64::UnchangedWord::accept);
                if (!renamed.ok()) return fail(PreparedImageError::patch_failed, index);
                if (renamed.changed()) add_patch(image, site, renamed.word);
                continue;
            }
            if (is_gate_system(decoded)) {
                const auto trampoline = image.appendix_offset + image.appendix.size();
                const auto plan = make_gate_site_patch(
                    site.word, address, kPlanningBase + trampoline,
                    {kPlanningBase + gate_address, gate_size}, options.gate_options);
                if (!plan.ok())
                    return fail(PreparedImageError::unsupported_instruction, index);
                const auto at =
                    append_words(image, plan.patch->trampoline, plan.patch->trampoline.size());
                add_patch(image, site, plan.patch->replacement_instruction);
                image.gate_sites.push_back({at + 8, site.word, {}});
                continue;
            }
            if (is_conditional_x18(decoded)) {
                const auto trampoline = image.appendix_offset + image.appendix.size();
                const auto plan = make_conditional_x18_patch(
                    site.word, address, kPlanningBase + trampoline, 0, site_policy);
                if (!plan.ok())
                    return fail(PreparedImageError::patch_failed, index);
                const auto at =
                    append_words(image, plan.patch->trampoline, plan.patch->trampoline.size());
                add_patch(image, site, plan.patch->replacement_instruction);
                image.fixups.push_back({PreparedFixupKind::thread_slot_load, at + 8, {}});
                conditional_taken_site[index] = at + (plan.patch->trampoline.size() - 1) * 4;
                continue;
            }
            if (reads_or_writes_x18(effects)) {
                const auto trampoline = image.appendix_offset + image.appendix.size();
                const auto plan = make_scalar_x18_patch(
                    site.word, address, kPlanningBase + trampoline, 0, site_policy);
                if (!plan.ok())
                    return fail(PreparedImageError::patch_failed, index);
                const auto at =
                    append_words(image, plan.patch->trampoline, plan.patch->trampoline_word_count);
                add_patch(image, site, plan.patch->replacement_instruction);
                image.fixups.push_back({PreparedFixupKind::thread_slot_load, at + 8, {}});
                if (decoded.opcode == Opcode::adr) {
                    const auto target = aarch64::referenced_address(decoded, address);
                    if (!target || *target < kPlanningBase ||
                        *target - kPlanningBase >= original_size) {
                        return fail(PreparedImageError::invalid_target, index);
                    }
                    image.fixups.push_back({PreparedFixupKind::module_address_mov_wide,
                                            at + 16,
                                            {options.module_index, *target - kPlanningBase}});
                }
                continue;
            }
            if (effects.status != EffectStatus::decoded) {
                return fail(PreparedImageError::unsupported_instruction, index);
            }
        }

        const auto boundary_failure = [&](ExclusiveWindowError reason, const ExclusiveX18Region& region,
                                           std::size_t i) {
            proof.diagnostics.push_back({reason, covered[region.entries.front()].image_offset,
                                         covered[i].image_offset});
            auto result = fail(PreparedImageError::exclusive_window, i);
            result.exclusive_proof = std::move(proof);
            return result;
        };
        const auto replace = [&](std::size_t i, std::uint32_t word) {
            const auto existing = std::find_if(image.replacements.begin(), image.replacements.end(),
                [&](const auto& patch) { return patch.image_offset == covered[i].image_offset; });
            if (existing == image.replacements.end()) add_patch(image, covered[i], word);
            else existing->replacement_word = word;
        };
        const auto transfer = [&](std::size_t i, unsigned reg, bool entering) {
            // MOV Xscratch,X18 / MOV X18,Xscratch reuses the scalar builder's
            // private spill, thread-cell access, flag preservation and fixup.
            const auto word = 0xaa0003e0u | ((entering ? 18u : reg) << 16) | (entering ? reg : 18u);
            return make_scalar_x18_patch(word, kPlanningBase + covered[i].image_offset,
                kPlanningBase + image.appendix_offset + image.appendix.size(), 0,
                {true, options.scalar_x18_policy.stack_scratch_available,
                 options.scalar_x18_policy.runtime_storage_disjoint});
        };
        for (const auto& region : proof.x18_regions) {
            for (const auto entry : region.entries) {
                const auto prior = entry - 1;
                const auto wrapper = image.appendix_offset + image.appendix.size();
                const auto original = move_boundary_word(covered[prior].word,
                    kPlanningBase + covered[prior].image_offset, kPlanningBase + wrapper);
                if (!original) return boundary_failure(ExclusiveWindowError::entry_transfer, region, prior);
                append_words(image, std::array{*original}, 1);
                const auto plan = transfer(prior, region.scratch_register, true);
                if (!plan.ok()) return boundary_failure(ExclusiveWindowError::entry_transfer, region, prior);
                const auto at = append_words(image, plan.patch->trampoline, plan.patch->trampoline_word_count);
                image.fixups.push_back({PreparedFixupKind::thread_slot_load, at + 8, {}});
                const auto enter = aarch64::encode_b(kPlanningBase + covered[prior].image_offset,
                                                     kPlanningBase + wrapper);
                if (!enter.ok()) return boundary_failure(ExclusiveWindowError::entry_transfer, region, prior);
                replace(prior, enter.word);
                // Only outside edges enter the transfer. A native backedge
                // within the merged region must retain its live reservation.
                const auto first_edge = std::lower_bound(region.entry_branches.begin(), region.entry_branches.end(), entry,
                    [](const auto& edge, auto index) { return edge.entry_index < index; });
                for (auto edge = first_edge; edge != region.entry_branches.end() && edge->entry_index == entry; ++edge) {
                    const auto i = edge->instruction_index;
                    if (conditional_taken_site[i]) {
                        const auto taken_at = *conditional_taken_site[i];
                        const auto redirected = aarch64::encode_b(kPlanningBase + taken_at, kPlanningBase + at);
                        if (!redirected.ok())
                            return boundary_failure(ExclusiveWindowError::entry_transfer, region, i);
                        std::memcpy(image.appendix.data() + taken_at - image.appendix_offset,
                                    &redirected.word, sizeof(redirected.word));
                        continue;
                    }
                    const auto redirected = redirect_branch(covered[i].word,
                        kPlanningBase + covered[i].image_offset, kPlanningBase + at);
                    if (redirected) {
                        replace(i, *redirected);
                        continue;
                    }
                    // Short conditional branches may not reach an appendix.
                    // This register-free stub runs only outside live windows;
                    // both outcomes retain their original register/flag state.
                    const auto stub = image.appendix_offset + image.appendix.size();
                    const auto enter_stub = aarch64::encode_b(kPlanningBase + covered[i].image_offset,
                                                               kPlanningBase + stub);
                    const auto condition = redirect_branch(covered[i].word, kPlanningBase + stub,
                                                            kPlanningBase + stub + 8);
                    const auto fallthrough = aarch64::encode_b(kPlanningBase + stub + 4,
                        kPlanningBase + covered[i].image_offset + 4);
                    const auto taken = aarch64::encode_b(kPlanningBase + stub + 8, kPlanningBase + at);
                    if (!enter_stub.ok() || !condition || !fallthrough.ok() || !taken.ok())
                        return boundary_failure(ExclusiveWindowError::entry_transfer, region, i);
                    append_words(image, std::array{*condition, fallthrough.word, taken.word}, 3);
                    replace(i, enter_stub.word);
                }
            }
        }
        for (const auto& region : proof.x18_regions) {
            for (const auto exit : region.exits) {
                if (std::any_of(image.fixups.begin(), image.fixups.end(), [&](const auto& fixup) {
                        return fixup.image_offset == covered[exit].image_offset;
                    })) return boundary_failure(ExclusiveWindowError::exit_transfer, region, exit);
                const auto wrapper = image.appendix_offset + image.appendix.size();
                auto plan = transfer(exit, region.scratch_register, false);
                if (!plan.ok()) return boundary_failure(ExclusiveWindowError::exit_transfer, region, exit);
                // Continue into the displaced boundary instruction (or its
                // already prepared trampoline) only after publishing x18.
                const auto count = plan.patch->trampoline_word_count;
                const auto body = wrapper + count * 4;
                const auto resume = aarch64::encode_b(kPlanningBase + wrapper + (count - 1) * 4,
                                                      kPlanningBase + body);
                if (!resume.ok()) return boundary_failure(ExclusiveWindowError::exit_transfer, region, exit);
                plan.patch->trampoline[count - 1] = resume.word;
                append_words(image, plan.patch->trampoline, count);
                image.fixups.push_back({PreparedFixupKind::thread_slot_load, wrapper + 8, {}});
                const auto existing = std::find_if(image.replacements.begin(), image.replacements.end(),
                    [&](const auto& patch) { return patch.image_offset == covered[exit].image_offset; });
                const auto original = move_boundary_word(existing == image.replacements.end()
                    ? covered[exit].word : existing->replacement_word,
                    kPlanningBase + covered[exit].image_offset, kPlanningBase + body);
                if (!original) return boundary_failure(ExclusiveWindowError::exit_transfer, region, exit);
                append_words(image, std::array{*original}, 1);
                const auto continuation = aarch64::encode_b(kPlanningBase + body + 4,
                    kPlanningBase + covered[exit].image_offset + 4);
                if (!continuation.ok()) return boundary_failure(ExclusiveWindowError::exit_transfer, region, exit);
                append_words(image, std::array{continuation.word}, 1);
                replace(exit, plan.patch->replacement_instruction);
            }
        }
        std::sort(image.replacements.begin(), image.replacements.end(),
                  [](const auto& left, const auto& right) { return left.image_offset < right.image_offset; });

        if (add_overflows(image.appendix_offset, image.appendix.size())) {
            return fail(PreparedImageError::layout_overflow);
        }
        std::sort(image.fixups.begin(), image.fixups.end(),
                  [](const PreparedFixup& left, const PreparedFixup& right) {
                      return left.image_offset < right.image_offset;
                  });
        return {.image = std::move(image), .exclusive_proof = std::move(proof)};
    } catch (const std::bad_alloc&) {
        return fail(PreparedImageError::allocation_failure);
    } catch (const std::length_error&) {
        return fail(PreparedImageError::layout_overflow);
    } catch (const std::logic_error&) {
        return fail(PreparedImageError::patch_failed);
    }
}

PreparedImageApplyResult apply_prepared_image(const PreparedImage& prepared,
                                              std::span<std::byte> image,
                                              const PreparedImageBindings& bindings) noexcept {
    static_assert(std::endian::native == std::endian::little);
    if (prepared.original_image_size == 0 ||
        prepared.original_image_size > kMaximum - (kAppendixAlignment - 1) ||
        prepared.appendix_offset != align_up(prepared.original_image_size, kAppendixAlignment) ||
        add_overflows(prepared.appendix_offset, prepared.appendix.size()) ||
        prepared.image_size() != image.size()) {
        return apply_fail(PreparedImageError::invalid_image_size);
    }
    const bool has_gate_binding =
        std::any_of(prepared.fixups.begin(), prepared.fixups.end(), [](const PreparedFixup& fixup) {
            return fixup.kind == PreparedFixupKind::gate_dispatcher_address;
        });
    const bool has_thread_slot =
        std::any_of(prepared.fixups.begin(), prepared.fixups.end(), [](const PreparedFixup& fixup) {
            return fixup.kind == PreparedFixupKind::thread_slot_load;
        });
    if (prepared.module_index >= bindings.module_bases.size() ||
        bindings.module_bases[prepared.module_index] == 0 ||
        bindings.module_bases[prepared.module_index] % kAppendixAlignment != 0 ||
        add_overflows(bindings.module_bases[prepared.module_index], prepared.image_size()) ||
        (has_thread_slot && (bindings.thread_slot_byte_offset % 8 != 0 ||
                             bindings.thread_slot_byte_offset > 4095 * 8)) ||
        (has_gate_binding && bindings.gate_dispatcher_address == 0)) {
        return apply_fail(PreparedImageError::invalid_binding);
    }

    std::uint64_t previous = 0;
    for (std::size_t index = 0; index < prepared.coverage.size(); ++index) {
        const auto& record = prepared.coverage[index];
        if (record.image_offset % 4 != 0 || record.image_offset > prepared.original_image_size ||
            prepared.original_image_size - record.image_offset < 4 ||
            record.replacement_word != record.expected_word ||
            (index != 0 && record.image_offset <= previous)) {
            return apply_fail(PreparedImageError::invalid_coverage, index);
        }
        previous = record.image_offset;
        std::uint32_t actual;
        std::memcpy(&actual, image.data() + record.image_offset, sizeof(actual));
        if (actual != record.expected_word) {
            return apply_fail(PreparedImageError::preimage_mismatch, index);
        }
    }

    previous = 0;
    for (std::size_t index = 0; index < prepared.replacements.size(); ++index) {
        const auto& patch = prepared.replacements[index];
        if (patch.image_offset % 4 != 0 || patch.image_offset > prepared.original_image_size ||
            prepared.original_image_size - patch.image_offset < 4 ||
            (index != 0 && patch.image_offset <= previous)) {
            return apply_fail(PreparedImageError::overlapping_regions, index);
        }
        previous = patch.image_offset;
        const auto coverage =
            std::lower_bound(prepared.coverage.begin(), prepared.coverage.end(), patch.image_offset,
                             [](const PreparedWordPatch& record, std::uint64_t at) {
                                 return record.image_offset < at;
                             });
        if (coverage == prepared.coverage.end() || coverage->image_offset != patch.image_offset ||
            coverage->expected_word != patch.expected_word) {
            return apply_fail(PreparedImageError::invalid_coverage, index);
        }
    }

    std::uint64_t prior_fixup_end = 0;
    for (std::size_t index = 0; index < prepared.fixups.size(); ++index) {
        const auto& fixup = prepared.fixups[index];
        std::size_t width = 0;
        switch (fixup.kind) {
        case PreparedFixupKind::thread_slot_load:
            width = 4;
            break;
        case PreparedFixupKind::gate_dispatcher_address:
            width = sizeof(std::uintptr_t);
            break;
        case PreparedFixupKind::module_address_mov_wide:
            width = 16;
            break;
        case PreparedFixupKind::branch26_module_target:
            width = 4;
            break;
        default:
            return apply_fail(PreparedImageError::invalid_fixup, index);
        }
        if (fixup.image_offset % (width == 8 ? 8 : 4) != 0 ||
            fixup.image_offset > prepared.image_size() ||
            prepared.image_size() - fixup.image_offset < width ||
            (index != 0 && fixup.image_offset < prior_fixup_end)) {
            return apply_fail(PreparedImageError::invalid_fixup, index);
        }
        prior_fixup_end = fixup.image_offset + width;
        if (fixup.kind == PreparedFixupKind::module_address_mov_wide ||
            fixup.kind == PreparedFixupKind::branch26_module_target) {
            if (fixup.target.module_index >= bindings.module_bases.size() ||
                bindings.module_bases[fixup.target.module_index] == 0 ||
                add_overflows(bindings.module_bases[fixup.target.module_index],
                              fixup.target.image_offset)) {
                return apply_fail(PreparedImageError::invalid_binding, index);
            }
            if (fixup.kind == PreparedFixupKind::branch26_module_target &&
                fixup.target.image_offset % 4 != 0) {
                return apply_fail(PreparedImageError::invalid_fixup, index);
            }
        }
        const auto word = planned_word(prepared, fixup.image_offset);
        if (fixup.kind == PreparedFixupKind::thread_slot_load &&
            (!word || !thread_slot_load(*word))) {
            return apply_fail(PreparedImageError::invalid_fixup, index);
        }
        if (fixup.kind == PreparedFixupKind::branch26_module_target &&
            (!word || !branch26(*word))) {
            return apply_fail(PreparedImageError::invalid_fixup, index);
        }
        if (fixup.kind == PreparedFixupKind::module_address_mov_wide) {
            if (fixup.image_offset < prepared.appendix_offset ||
                !fixed_width_mov_sequence(prepared.appendix,
                                          fixup.image_offset - prepared.appendix_offset)) {
                return apply_fail(PreparedImageError::invalid_fixup, index);
            }
        }
        if (fixup.kind == PreparedFixupKind::gate_dispatcher_address) {
            if (fixup.image_offset < prepared.appendix_offset) {
                return apply_fail(PreparedImageError::invalid_fixup, index);
            }
            std::uintptr_t zero = 1;
            std::memcpy(&zero,
                        prepared.appendix.data() + (fixup.image_offset - prepared.appendix_offset),
                        sizeof(zero));
            if (zero != 0)
                return apply_fail(PreparedImageError::invalid_fixup, index);
        }
        if (fixup.kind == PreparedFixupKind::branch26_module_target) {
            const auto pc = bindings.module_bases[prepared.module_index] + fixup.image_offset;
            const auto target =
                bindings.module_bases[fixup.target.module_index] + fixup.target.image_offset;
            const auto encoded = (*word & 0x80000000u) != 0 ? aarch64::encode_bl(pc, target)
                                                            : aarch64::encode_b(pc, target);
            if (!encoded.ok())
                return apply_fail(PreparedImageError::unreachable_target, index);
        }
    }

    std::uint64_t prior_return_address = 0;
    for (std::size_t index = 0; index < prepared.gate_sites.size(); ++index) {
        const auto& site = prepared.gate_sites[index];
        if (!has_gate_binding || site.return_address_offset % 4 != 0 ||
            site.return_address_offset < prepared.appendix_offset + 8 ||
            site.return_address_offset > prepared.image_size() ||
            (index != 0 && site.return_address_offset <= prior_return_address)) {
            return apply_fail(PreparedImageError::invalid_fixup, index);
        }
        prior_return_address = site.return_address_offset;
        for (const auto target : site.allowed_indirect_targets) {
            if (target.module_index >= bindings.module_bases.size() ||
                bindings.module_bases[target.module_index] == 0 || target.image_offset % 4 != 0 ||
                add_overflows(bindings.module_bases[target.module_index], target.image_offset)) {
                return apply_fail(PreparedImageError::invalid_binding, index);
            }
        }
    }

    if (!prepared.appendix.empty()) {
        std::memcpy(image.data() + prepared.appendix_offset, prepared.appendix.data(),
                    prepared.appendix.size());
    }
    for (const auto& patch : prepared.replacements) {
        std::memcpy(image.data() + patch.image_offset, &patch.replacement_word,
                    sizeof(patch.replacement_word));
    }
    for (const auto& fixup : prepared.fixups) {
        auto* destination = image.data() + fixup.image_offset;
        switch (fixup.kind) {
        case PreparedFixupKind::thread_slot_load: {
            std::uint32_t word;
            std::memcpy(&word, destination, sizeof(word));
            word = (word & ~(4095u << 10)) | ((bindings.thread_slot_byte_offset / 8) << 10);
            std::memcpy(destination, &word, sizeof(word));
            break;
        }
        case PreparedFixupKind::gate_dispatcher_address:
            std::memcpy(destination, &bindings.gate_dispatcher_address,
                        sizeof(bindings.gate_dispatcher_address));
            break;
        case PreparedFixupKind::module_address_mov_wide: {
            const auto value =
                bindings.module_bases[fixup.target.module_index] + fixup.target.image_offset;
            std::uint32_t words[4];
            std::memcpy(words, destination, sizeof(words));
            for (unsigned half = 0; half != 4; ++half) {
                words[half] = (words[half] & ~(0xffffu << 5)) |
                              (static_cast<std::uint32_t>(value >> (16 * half)) & 0xffffu) << 5;
            }
            std::memcpy(destination, words, sizeof(words));
            break;
        }
        case PreparedFixupKind::branch26_module_target: {
            std::uint32_t word;
            std::memcpy(&word, destination, sizeof(word));
            const auto pc = bindings.module_bases[prepared.module_index] + fixup.image_offset;
            const auto target =
                bindings.module_bases[fixup.target.module_index] + fixup.target.image_offset;
            const auto encoded = (word & 0x80000000u) != 0 ? aarch64::encode_bl(pc, target)
                                                           : aarch64::encode_b(pc, target);
            std::memcpy(destination, &encoded.word, sizeof(encoded.word));
            break;
        }
        default:
            return apply_fail(PreparedImageError::invalid_fixup);
        }
    }
    return {};
}

} // namespace armigo::execution
