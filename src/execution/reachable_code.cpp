// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/reachable_code.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace armigo::execution {
namespace {

using aarch64::Condition;
using aarch64::EffectStatus;
using aarch64::Instruction;
using aarch64::Opcode;
using aarch64::Operation;
using aarch64::OperandRole;
using aarch64::RegisterEffects;

constexpr std::uint64_t max_address = std::numeric_limits<std::uint64_t>::max();

enum : std::uint8_t { unvisited = 0, visited = 1, thunk_interior = 2 };

// UDF #imm16: permanently undefined.
constexpr bool is_udf(std::uint32_t word) noexcept { return (word >> 16) == 0; }
// HINT space: allocated hints and unallocated ones (executed as NOP) alike
// continue with the next instruction.
constexpr bool is_hint(std::uint32_t word) noexcept {
    return (word & 0xFFFFF01Fu) == 0xD503201Fu;
}
// Branches, exception generation and system instructions (op0 = x101).
constexpr bool in_branch_group(std::uint32_t word) noexcept {
    return (word & 0x1C000000u) == 0x14000000u;
}
// System instructions (hints, barriers, PSTATE, SYS, MRS/MSR) do not write PC.
constexpr bool in_system_subgroup(std::uint32_t word) noexcept {
    return (word & 0xFFC00000u) == 0xD5000000u;
}

struct ExecutableRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint32_t image = 0;
    std::vector<std::uint8_t> state;  // one entry per word

    std::uint8_t& at(std::uint64_t address) { return state[(address - begin) / 4]; }
};

struct ThunkShape {
    std::uint64_t slot_offset = 0;
    std::uint8_t address_register = 0;
    std::uint8_t target_register = 0;
    bool has_add = false;
};

const aarch64::RegisterOperand* operand_with_role(const RegisterEffects& effects,
                                                  OperandRole role) noexcept {
    for (std::size_t i = 0; i < effects.operand_count; ++i) {
        if (effects.operands[i].role == role) return &effects.operands[i];
    }
    return nullptr;
}

class Collector {
public:
    Collector(std::span<const ReachableImage> images, std::span<const ExternalLeaf> leaves,
              const ReachableOptions& options)
        : images_(images), options_(options), leaves_(leaves.begin(), leaves.end()) {}

    std::optional<ReachableFailure> validate(std::span<const std::uint64_t> entries) {
        image_order_.resize(images_.size());
        for (std::uint32_t i = 0; i < images_.size(); ++i) {
            const ReachableImage& image = images_[i];
            if (image.bytes.empty() || image.bytes.size() > max_address - image.base) {
                return setup(ReachableError::invalid_image, image.base);
            }
            image_order_[i] = i;
            for (const AddressRange& range : image.executable) {
                if (range.begin >= range.end || range.begin % 4 != 0 || range.end % 4 != 0 ||
                    range.begin < image.base || range.end > image_end(image)) {
                    return setup(ReachableError::invalid_executable_range, range.begin);
                }
                ranges_.push_back({range.begin, range.end, i, {}});
            }
        }
        std::sort(image_order_.begin(), image_order_.end(), [&](auto a, auto b) {
            return images_[a].base < images_[b].base;
        });
        for (std::size_t i = 1; i < image_order_.size(); ++i) {
            if (images_[image_order_[i]].base < image_end(images_[image_order_[i - 1]])) {
                return setup(ReachableError::overlapping_images, images_[image_order_[i]].base);
            }
        }
        std::sort(ranges_.begin(), ranges_.end(),
                  [](const auto& a, const auto& b) { return a.begin < b.begin; });
        for (std::size_t i = 1; i < ranges_.size(); ++i) {
            if (ranges_[i].begin < ranges_[i - 1].end) {
                return setup(ReachableError::invalid_executable_range, ranges_[i].begin);
            }
        }

        std::sort(leaves_.begin(), leaves_.end(),
                  [](const auto& a, const auto& b) { return a.address < b.address; });
        for (std::size_t i = 0; i < leaves_.size(); ++i) {
            if (leaves_[i].address % 4 != 0 ||
                (i > 0 && leaves_[i].address == leaves_[i - 1].address &&
                 leaves_[i].returns != leaves_[i - 1].returns)) {
                return setup(ReachableError::invalid_leaf, leaves_[i].address);
            }
        }
        leaves_.erase(std::unique(leaves_.begin(), leaves_.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.address == b.address;
                                  }),
                      leaves_.end());

        for (const std::uint64_t entry : entries) {
            if (entry % 4 != 0 || find_range(entry) == nullptr || find_leaf(entry) != nullptr) {
                return setup(ReachableError::invalid_entry, entry);
            }
        }
        for (ExecutableRange& range : ranges_) {
            range.state.assign((range.end - range.begin) / 4, unvisited);
        }
        return std::nullopt;
    }

    ReachableCode run(std::span<const std::uint64_t> entries) {
        // Entries are explored in the supplied order; discovered targets LIFO.
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            worklist_.emplace_back(*it, *it);
        }
        while (!worklist_.empty() && !stopped_) {
            const auto [address, source] = worklist_.back();
            worklist_.pop_back();
            walk(address, source);
        }
        return finish();
    }

private:
    static std::uint64_t image_end(const ReachableImage& image) noexcept {
        return image.base + image.bytes.size();
    }

    static ReachableFailure setup(ReachableError error, std::uint64_t address) noexcept {
        return ReachableFailure{.error = error, .address = address};
    }

    ExecutableRange* find_range(std::uint64_t address) {
        auto it = std::upper_bound(ranges_.begin(), ranges_.end(), address,
                                   [](std::uint64_t a, const auto& r) { return a < r.begin; });
        if (it == ranges_.begin()) return nullptr;
        --it;
        return address < it->end ? &*it : nullptr;
    }

    const ExternalLeaf* find_leaf(std::uint64_t address) const {
        auto it = std::lower_bound(leaves_.begin(), leaves_.end(), address,
                                   [](const auto& l, std::uint64_t a) { return l.address < a; });
        return it != leaves_.end() && it->address == address ? &*it : nullptr;
    }

    // Reads size bytes at address if they lie inside one supplied image.
    std::optional<std::uint64_t> read(std::uint64_t address, unsigned size) const {
        auto it = std::upper_bound(image_order_.begin(), image_order_.end(), address,
                                   [&](std::uint64_t a, std::uint32_t i) {
                                       return a < images_[i].base;
                                   });
        if (it == image_order_.begin()) return std::nullopt;
        const ReachableImage& image = images_[*(it - 1)];
        const std::uint64_t offset = address - image.base;
        if (offset >= image.bytes.size() || image.bytes.size() - offset < size) {
            return std::nullopt;
        }
        std::uint64_t value = 0;
        for (unsigned i = 0; i < size; ++i) {
            value |= static_cast<std::uint64_t>(image.bytes[offset + i]) << (8 * i);
        }
        return value;
    }

    std::uint32_t word_at(std::uint64_t address) const {
        return static_cast<std::uint32_t>(*read(address, 4));
    }

    void fail(ReachableError error, std::uint64_t address, std::uint32_t word,
              std::uint64_t source, std::uint64_t target = 0, std::uint64_t slot = 0) {
        if (failures_.size() >= options_.max_failures) {
            truncated_ = true;
            stopped_ = true;
            return;
        }
        failures_.push_back({error, address, word, source, target, slot});
    }

    ReachableSite& add_site(std::uint64_t address, std::uint32_t word, const ExecutableRange& range,
                            SiteKind kind) {
        ReachableSite site;
        site.address = address;
        site.word = word;
        site.image = range.image;
        site.kind = kind;
        site.instruction = aarch64::decode(word);
        site.effects = aarch64::decode_register_effects(word);
        sites_.push_back(site);
        return sites_.back();
    }

    // Records an edge and queues an internal target. Returns the leaf when the
    // target is a declared leaf; *ok reports whether the target was accepted.
    const ExternalLeaf* follow(std::uint64_t target, std::uint64_t source_address,
                               std::uint32_t word, std::uint64_t source_origin, EdgeKind kind,
                               bool* ok) {
        *ok = false;
        if (const ExternalLeaf* leaf = find_leaf(target)) {
            edges_.push_back({source_address, target, kind, true});
            *ok = true;
            return leaf;
        }
        if (target % 4 != 0) {
            fail(ReachableError::misaligned_target, source_address, word, source_origin, target);
            return nullptr;
        }
        if (find_range(target) == nullptr) {
            fail(ReachableError::target_not_executable, source_address, word, source_origin,
                 target);
            return nullptr;
        }
        edges_.push_back({source_address, target, kind, false});
        worklist_.emplace_back(target, source_address);
        *ok = true;
        return nullptr;
    }

    std::optional<ThunkShape> match_thunk(ExecutableRange& range, std::uint64_t pc,
                                          const Instruction& adrp) {
        if (adrp.reg >= 31 || range.end - pc < 12) return std::nullopt;
        const RegisterEffects load = aarch64::decode_register_effects(word_at(pc + 4));
        const auto* transfer = operand_with_role(load, OperandRole::transfer);
        const auto* base = operand_with_role(load, OperandRole::base);
        if (load.status != EffectStatus::decoded || load.operation != Operation::ldr ||
            load.memory.addressing != aarch64::AddressingMode::unsigned_offset ||
            load.memory.element_bytes != 8 || load.memory.writeback || transfer == nullptr ||
            base == nullptr || !transfer->is_general() || transfer->bits != 64 ||
            transfer->number >= 31 || base->number != adrp.reg) {
            return std::nullopt;
        }
        ThunkShape shape{static_cast<std::uint64_t>(load.memory.offset), adrp.reg,
                         transfer->number, false};
        const auto is_branch_through = [&](std::uint64_t address) {
            const Instruction branch = aarch64::decode(word_at(address));
            return branch.opcode == Opcode::br && branch.reg == shape.target_register;
        };
        if (is_branch_through(pc + 8)) return shape;

        // ADD Xa, Xa, #slot_offset must not clobber the loaded target.
        if (range.end - pc < 16 || shape.target_register == shape.address_register) {
            return std::nullopt;
        }
        const std::uint32_t add_word = word_at(pc + 8);
        const RegisterEffects add = aarch64::decode_register_effects(add_word);
        const auto* destination = operand_with_role(add, OperandRole::destination);
        const auto* source = operand_with_role(add, OperandRole::source);
        const std::uint64_t immediate = static_cast<std::uint64_t>((add_word >> 10) & 0xFFFu)
                                        << (((add_word >> 22) & 1u) ? 12 : 0);
        if (add.status != EffectStatus::decoded || add.operation != Operation::add_immediate ||
            (add_word >> 31) == 0 || destination == nullptr || source == nullptr ||
            destination->number != shape.address_register ||
            source->number != shape.address_register || immediate != shape.slot_offset ||
            !is_branch_through(pc + 12)) {
            return std::nullopt;
        }
        shape.has_add = true;
        return shape;
    }

    bool calls_terminal_thunk(std::uint64_t entry) {
        auto* range = find_range(entry);
        if (range == nullptr) return false;
        const auto adrp = aarch64::decode(word_at(entry));
        if (adrp.opcode != Opcode::adrp) return false;
        const auto shape = match_thunk(*range, entry, adrp);
        if (!shape) return false;
        const auto page = aarch64::referenced_address(adrp, entry);
        if (!page || shape->slot_offset > max_address - *page) return false;
        const auto slot = *page + shape->slot_offset;
        if (slot % 8) return false;
        const auto target = read(slot, 8);
        if (!target) return false;
        const auto* leaf = find_leaf(*target);
        return leaf != nullptr && !leaf->returns;
    }

    void resolve_thunk(ExecutableRange& range, std::uint64_t pc, const Instruction& adrp, const ThunkShape& shape) {
        const std::uint64_t branch = pc + (shape.has_add ? 12 : 8);
        static constexpr SiteKind interior_kinds[] = {SiteKind::thunk_load, SiteKind::thunk_add,
                                                      SiteKind::thunk_branch};
        std::size_t branch_site = sites_.size();  // not yet created
        bool branch_created = false;
        for (std::uint64_t address = pc + 4; address <= branch; address += 4) {
            const SiteKind kind = address == branch ? SiteKind::thunk_branch
                                                    : interior_kinds[(address - pc) / 4 - 1];
            std::uint8_t& state = range.at(address);
            if (state != unvisited) continue;  // an earlier path already failed here
            if (sites_.size() >= options_.max_sites) {
                fail(ReachableError::resource_limit, address, word_at(address), pc);
                stopped_ = true;
                return;
            }
            state = thunk_interior;
            if (address == branch) {
                branch_site = sites_.size();
                branch_created = true;
            }
            add_site(address, word_at(address), range, kind);
        }
        const std::uint32_t branch_word = word_at(branch);
        const auto thunk_failed = [&](ReachableError error, std::uint64_t target,
                                      std::uint64_t slot) {
            fail(error, branch, branch_word, pc, target, slot);
            if (branch_created) sites_[branch_site].kind = SiteKind::failed;
        };

        const auto page = aarch64::referenced_address(adrp, pc);
        if (!page || shape.slot_offset > max_address - *page) {
            thunk_failed(ReachableError::target_address_overflow, 0, 0);
            return;
        }
        const std::uint64_t slot = *page + shape.slot_offset;
        if (slot % 8 != 0) {
            thunk_failed(ReachableError::thunk_slot_misaligned, 0, slot);
            return;
        }
        const std::optional<std::uint64_t> target = read(slot, 8);
        if (!target) {
            thunk_failed(ReachableError::thunk_slot_unreadable, 0, slot);
            return;
        }
        ResolvedThunk thunk{pc, branch, slot, *target, false, shape.has_add,
                            shape.address_register, shape.target_register};
        if (find_leaf(*target) != nullptr) {
            thunk.external_leaf = true;
            edges_.push_back({pc, *target, EdgeKind::thunk, true});
        } else if (*target % 4 != 0) {
            thunk_failed(ReachableError::misaligned_target, *target, slot);
            return;
        } else if (find_range(*target) == nullptr) {
            thunk_failed(ReachableError::target_not_executable, *target, slot);
            return;
        } else {
            edges_.push_back({pc, *target, EdgeKind::thunk, false});
            worklist_.emplace_back(*target, pc);
        }
        thunks_.push_back(thunk);
    }

    // Walks one straight-line path from address.
    void walk(std::uint64_t pc, std::uint64_t source) {
        while (!stopped_) {
            ExecutableRange& range = *find_range(pc);
            std::uint8_t& state = range.at(pc);
            const std::uint32_t word = word_at(pc);
            if (state == thunk_interior) {
                fail(ReachableError::thunk_interior_entry, pc, word, source);
                return;
            }
            if (state == visited) return;
            if (sites_.size() >= options_.max_sites) {
                fail(ReachableError::resource_limit, pc, word, source);
                stopped_ = true;
                return;
            }
            state = visited;
            const ReachableSite& site = add_site(pc, word, range, SiteKind::sequential);
            const std::size_t index = sites_.size() - 1;
            const Instruction instruction = site.instruction;
            const EffectStatus status = site.effects.status;
            const auto failed_here = [&](ReachableError error) {
                fail(error, pc, word, source);
                sites_[index].kind = SiteKind::failed;
            };
            const auto set_kind = [&](SiteKind kind) { sites_[index].kind = kind; };
            const auto target = [&]() -> std::optional<std::uint64_t> {
                const auto address = aarch64::referenced_address(instruction, pc);
                if (!address) {
                    fail(ReachableError::target_address_overflow, pc, word, source);
                }
                return address;
            };
            bool ok = false;
            bool continues = true;

            if (is_udf(word)) {
                if (options_.traps == TrapPolicy::fail) {
                    failed_here(ReachableError::trap);
                } else {
                    set_kind(SiteKind::trap);
                    ++statistics_.traps;
                }
                return;
            }
            switch (instruction.opcode) {
            case Opcode::b:
                set_kind(SiteKind::branch);
                if (const auto t = target()) follow(*t, pc, word, source, EdgeKind::branch, &ok);
                return;
            case Opcode::bl: {
                set_kind(SiteKind::call);
                const auto t = target();
                if (!t) return;
                const ExternalLeaf* leaf = follow(*t, pc, word, source, EdgeKind::call, &ok);
                if (leaf != nullptr && !leaf->returns) return;
                // A validated tail thunk preserves the final leaf's nonreturn
                // contract. Its instructions are still queued and collected.
                if (ok && leaf == nullptr && calls_terminal_thunk(*t)) return;
                break;
            }
            case Opcode::b_cond:
            case Opcode::bc_cond:
            case Opcode::cbz:
            case Opcode::cbnz:
            case Opcode::tbz:
            case Opcode::tbnz: {
                const bool always = (instruction.opcode == Opcode::b_cond ||
                                     instruction.opcode == Opcode::bc_cond) &&
                                    (instruction.condition == Condition::al ||
                                     instruction.condition == Condition::nv);
                set_kind(always ? SiteKind::branch : SiteKind::conditional_branch);
                const auto t = target();
                if (t) {
                    follow(*t, pc, word, source,
                           always ? EdgeKind::branch : EdgeKind::conditional_branch, &ok);
                }
                if (always || !t) return;
                break;
            }
            case Opcode::adrp:
                if (const auto shape = match_thunk(range, pc, instruction)) {
                    set_kind(SiteKind::thunk_address);
                    resolve_thunk(range, pc, instruction, *shape);
                    return;
                }
                break;
            case Opcode::br:
                failed_here(ReachableError::indirect_branch);
                return;
            case Opcode::blr:
                failed_here(ReachableError::indirect_call);
                continues = options_.follow_indirect_call_return;
                break;
            case Opcode::ret: case Opcode::retab:
                set_kind(SiteKind::return_);
                return;
            case Opcode::svc:
                if (options_.supervisor_calls == SupervisorCallPolicy::fail) {
                    failed_here(ReachableError::supervisor_call);
                    return;
                }
                set_kind(SiteKind::supervisor_call);
                ++statistics_.supervisor_calls;
                break;
            case Opcode::hvc:
            case Opcode::smc:
            case Opcode::brk:
            case Opcode::hlt:
                if (options_.traps == TrapPolicy::fail) {
                    failed_here(ReachableError::trap);
                } else {
                    set_kind(SiteKind::trap);
                    ++statistics_.traps;
                }
                return;
            case Opcode::unclassified:
                if (status == EffectStatus::reserved) {
                    failed_here(ReachableError::reserved_instruction);
                    return;
                }
                if (status == EffectStatus::unpredictable) {
                    failed_here(ReachableError::unpredictable_instruction);
                    return;
                }
                if (status == EffectStatus::unknown && !is_hint(word)) {
                    if (in_branch_group(word) && !in_system_subgroup(word)) {
                        failed_here(ReachableError::unsupported_control_flow);
                        return;
                    }
                    if (options_.unknown_instructions == UnknownInstructionPolicy::fail) {
                        failed_here(ReachableError::unknown_instruction);
                        return;
                    }
                    set_kind(SiteKind::assumed_sequential);
                    ++statistics_.assumed_sequential;
                }
                break;
            default:  // ADR, literal loads, MRS, MSR
                break;
            }
            if (!continues) return;

            if (pc == max_address - 3) {
                fail(ReachableError::fallthrough_not_executable, pc, word, source, 0);
                return;
            }
            const std::uint64_t next = pc + 4;
            if (find_leaf(next) != nullptr) return;
            if (find_range(next) == nullptr) {
                fail(ReachableError::fallthrough_not_executable, pc, word, source, next);
                return;
            }
            source = pc;
            pc = next;
        }
    }

    ReachableCode finish() {
        ReachableCode result;
        result.status = failures_.empty() && !truncated_ ?
                            CollectionStatus::complete : CollectionStatus::partial;
        std::sort(sites_.begin(), sites_.end(),
                  [](const auto& a, const auto& b) { return a.address < b.address; });
        const auto edge_key = [](const ReachableEdge& e) {
            return std::tuple(e.source, e.target, e.kind, e.external_leaf);
        };
        std::sort(edges_.begin(), edges_.end(),
                  [&](const auto& a, const auto& b) { return edge_key(a) < edge_key(b); });
        edges_.erase(std::unique(edges_.begin(), edges_.end(),
                                 [&](const auto& a, const auto& b) {
                                     return edge_key(a) == edge_key(b);
                                 }),
                     edges_.end());
        std::sort(thunks_.begin(), thunks_.end(),
                  [](const auto& a, const auto& b) { return a.entry < b.entry; });
        const auto failure_key = [](const ReachableFailure& f) {
            return std::tuple(f.address, f.error, f.source, f.target, f.slot, f.word);
        };
        std::sort(failures_.begin(), failures_.end(),
                  [&](const auto& a, const auto& b) { return failure_key(a) < failure_key(b); });
        failures_.erase(std::unique(failures_.begin(), failures_.end(),
                                    [&](const auto& a, const auto& b) {
                                        return failure_key(a) == failure_key(b);
                                    }),
                        failures_.end());
        statistics_.external_leaf_edges = static_cast<std::uint64_t>(
            std::count_if(edges_.begin(), edges_.end(),
                          [](const auto& e) { return e.external_leaf; }));
        result.sites = std::move(sites_);
        result.edges = std::move(edges_);
        result.thunks = std::move(thunks_);
        result.failures = std::move(failures_);
        result.failures_truncated = truncated_;
        result.statistics = statistics_;
        return result;
    }

    std::span<const ReachableImage> images_;
    const ReachableOptions& options_;
    std::vector<ExternalLeaf> leaves_;
    std::vector<std::uint32_t> image_order_;
    std::vector<ExecutableRange> ranges_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> worklist_;  // address, source
    std::vector<ReachableSite> sites_;
    std::vector<ReachableEdge> edges_;
    std::vector<ResolvedThunk> thunks_;
    std::vector<ReachableFailure> failures_;
    ReachableStatistics statistics_;
    bool truncated_ = false;
    bool stopped_ = false;
};

} // namespace

const char* describe(ReachableError error) noexcept {
    switch (error) {
    case ReachableError::none: return "none";
    case ReachableError::invalid_image: return "invalid image view";
    case ReachableError::overlapping_images: return "overlapping image views";
    case ReachableError::invalid_executable_range: return "invalid executable range";
    case ReachableError::invalid_leaf: return "invalid external leaf";
    case ReachableError::invalid_entry: return "invalid entry address";
    case ReachableError::unknown_instruction: return "instruction outside the decoder";
    case ReachableError::reserved_instruction: return "reserved (undefined) instruction";
    case ReachableError::unpredictable_instruction: return "constrained unpredictable instruction";
    case ReachableError::unsupported_control_flow: return "unsupported control-flow instruction";
    case ReachableError::indirect_branch: return "unresolved indirect branch";
    case ReachableError::indirect_call: return "unresolved indirect call";
    case ReachableError::supervisor_call: return "supervisor call refused by policy";
    case ReachableError::trap: return "trapping instruction refused by policy";
    case ReachableError::target_address_overflow: return "target leaves the address space";
    case ReachableError::misaligned_target: return "misaligned branch target";
    case ReachableError::target_not_executable: return "target outside executable code and leaves";
    case ReachableError::fallthrough_not_executable: return "fallthrough leaves executable code";
    case ReachableError::thunk_slot_misaligned: return "misaligned thunk slot";
    case ReachableError::thunk_slot_unreadable: return "thunk slot outside supplied images";
    case ReachableError::thunk_interior_entry: return "control enters a thunk after its start";
    case ReachableError::resource_limit: return "site limit exceeded";
    }
    return "unrecognized error";
}

ReachableCode collect_reachable_code(std::span<const ReachableImage> images,
                                     std::span<const std::uint64_t> entries,
                                     std::span<const ExternalLeaf> leaves,
                                     const ReachableOptions& options) {
    Collector collector(images, leaves, options);
    if (const auto error = collector.validate(entries)) {
        ReachableCode result;
        result.status = CollectionStatus::invalid_input;
        result.setup_error = *error;
        return result;
    }
    return collector.run(entries);
}

} // namespace armigo::execution
