// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "preparation_analysis.h"

#include <armigo/aarch64/instruction.h>
#include <armigo/aarch64/register_effects.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace armigo::runtime::preparation {
namespace {
namespace a64 = aarch64;
using Opcode = a64::Opcode;

bool within(std::uint64_t offset, std::uint64_t size, std::uint64_t extent) {
    return offset <= extent && size <= extent - offset;
}

bool contains(const loader::NsoSegment& segment, std::uint64_t offset,
              std::uint64_t size) {
    return offset >= segment.image_offset &&
           within(offset - segment.image_offset, size, segment.content_size);
}

bool permission(const loader::NsoSegment& segment, loader::SegmentPermission bit) {
    return (static_cast<unsigned>(segment.permission) & static_cast<unsigned>(bit)) != 0;
}

std::uint64_t read(std::span<const std::byte> bytes, std::uint64_t offset, unsigned size) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < size; ++i)
        value |= std::to_integer<std::uint64_t>(bytes[static_cast<std::size_t>(offset + i)]) << (i * 8);
    return value;
}

std::optional<std::uint64_t> relative_sdata4(std::uint64_t base, std::uint64_t encoded) {
    const std::int64_t displacement = (encoded & 0x80000000u)
        ? static_cast<std::int64_t>(encoded) - 0x1'0000'0000ll
        : static_cast<std::int64_t>(encoded);
    if (displacement < 0) {
        const auto magnitude = static_cast<std::uint64_t>(-displacement);
        if (magnitude > base) return {};
        return base - magnitude;
    }
    const auto magnitude = static_cast<std::uint64_t>(displacement);
    if (magnitude > std::numeric_limits<std::uint64_t>::max() - base) return {};
    return base + magnitude;
}

struct Node {
    Target location;
    a64::Instruction instruction;
    a64::RegisterEffects effects;
};

struct DataRange {
    std::uint64_t begin;
    std::uint64_t end;
};

class Analysis {
public:
    Analysis(std::span<const AnalysisModule> modules, const AnalysisLimits& limits)
        : modules_(modules), limits_(limits) {}

    AnalysisResult failure_result(const char* message) {
        evidence_.error = message;
        evidence_.error += " Exclusive windows: found=" + std::to_string(evidence_.exclusive_windows_found) +
            ", proved=" + std::to_string(evidence_.exclusive_windows_proved) +
            ", proof=" + (evidence_.exclusive_proof_complete ? "complete" : "incomplete") + ".";
        std::map<execution::ExclusiveWindowError, std::size_t> reasons;
        for (const auto& diagnostic : evidence_.exclusive_diagnostics) ++reasons[diagnostic.window.reason];
        for (const auto& [reason, count] : reasons)
            evidence_.error += " " + std::string(execution::exclusive_window_error_name(reason)) + "=" +
                std::to_string(count) + ".";
        return std::move(evidence_);
    }

    AnalysisResult run() {
        validate();
        collect_roots();
        if (roots_.empty()) fail("No supported executable metadata roots.");
        drain();
        // New indirect edges may introduce leaders inside an earlier proof.
        // Resolve discovery first, then revalidate every proof with all leaders.
        std::size_t cursor = 0;
        while (cursor < indirect_.size()) {
            const auto site = indirect_[cursor++];
            const auto target = resolve(site);
            resolved_.emplace(site, target);
            enqueue(target, true);
            drain();
        }
        for (const auto& [site, target] : resolved_)
            if (resolve(site) != target) fail("An indirect target proof changed after discovery.");
        check_return_provenance();
        const auto exclusive_proofs = check_exclusive_windows();
        check_address_escapes();
        check_data_overlap();

        AnalysisResult result = std::move(evidence_);
        result.modules.resize(modules_.size());
        for (const auto address : roots_) result.roots.push_back(location(address, true));
        for (const auto& [address, node] : nodes_) {
            auto& module = result.modules[node.location.module_index];
            module.code.push_back({node.location.image_offset, node.instruction.word});
            if (const auto found = resolved_.find(address); found != resolved_.end())
                module.indirect_sites.push_back({node.location.image_offset,
                                                {location(found->second, true)}});
        }
        for (std::size_t module = 0; module < result.modules.size(); ++module) {
            result.modules[module].function_entries = std::move(function_entries_[module]);
            result.modules[module].external_entries = std::move(external_entries_[module]);
            for (std::size_t index = 0; index < result.modules[module].code.size(); ++index)
                result.modules[module].code[index].exclusive_monitor_clear =
                    exclusive_proofs[module].monitor_clear[index];
        }
        // Manifest order is independent of synthetic address order.
        std::sort(result.roots.begin(), result.roots.end(), [](const Target& a, const Target& b) {
            return std::pair{a.module_index, a.image_offset} < std::pair{b.module_index, b.image_offset};
        });
        return result;
    }

private:
    [[noreturn]] static void fail(const char* message) { throw std::runtime_error(message); }

    std::vector<execution::ExclusiveWindowProof> check_exclusive_windows() {
        std::vector<std::vector<execution::CoveredInstruction>> words(modules_.size());
        function_entries_.resize(modules_.size());
        external_entries_.resize(modules_.size());
        auto& entries = function_entries_;
        for (const auto address : roots_) {
            const auto owner = location(address, true);
            entries[owner.module_index].push_back(owner.image_offset);
        }
        for (const auto& [address, node] : nodes_) {
            execution::CoveredInstruction covered{node.location.image_offset, node.instruction.word, {}, {}};
            // Every discovered direct transfer must retain its target ownership,
            // including conditional ingress into another module's window.
            if (node.effects.control_flow == a64::ControlFlow::branch ||
                node.effects.control_flow == a64::ControlFlow::conditional_branch ||
                node.effects.control_flow == a64::ControlFlow::call) {
                const auto target = location(referenced(node, address), true);
                if (target.module_index != node.location.module_index) {
                    covered.cross_module_target = {target.module_index, target.image_offset};
                    external_entries_[target.module_index].push_back(target.image_offset);
                }
                if (node.instruction.opcode == Opcode::bl)
                    entries[target.module_index].push_back(target.image_offset);
            }
            if (const auto target = resolved_.find(address); target != resolved_.end()) {
                const auto owner = location(target->second, true);
                const execution::PreparedLocation destination{owner.module_index, owner.image_offset};
                covered.indirect_targets.push_back({destination, destination});
                if (owner.module_index != node.location.module_index)
                    external_entries_[owner.module_index].push_back(owner.image_offset);
                if (node.instruction.opcode == Opcode::blr)
                    entries[owner.module_index].push_back(owner.image_offset);
            }
            words[node.location.module_index].push_back(std::move(covered));
        }
        std::vector<execution::ExclusiveWindowProof> proofs;
        auto remaining = limits_.exclusive_windows;
        for (std::size_t module = 0; module < modules_.size(); ++module) {
            auto proof = execution::prove_exclusive_windows(words[module], remaining, entries[module],
                                                            static_cast<std::uint32_t>(module), external_entries_[module]);
            remaining.max_windows -= std::min(remaining.max_windows, proof.windows_found);
            remaining.max_total_steps -= proof.steps;
            evidence_.exclusive_windows_proved += proof.windows_proved;
            for (const auto& diagnostic : proof.diagnostics)
                evidence_.exclusive_diagnostics.push_back({static_cast<std::uint32_t>(module), diagnostic});
            proofs.push_back(std::move(proof));
        }
        evidence_.exclusive_proof_complete = evidence_.exclusive_diagnostics.empty();
        if (!evidence_.exclusive_diagnostics.empty()) {
            const auto& diagnostic = evidence_.exclusive_diagnostics.front();
            std::string message = "Exclusive window refusal: module=" + std::to_string(diagnostic.module_index);
            if (diagnostic.window.window_offset)
                message += " window=" + std::to_string(*diagnostic.window.window_offset);
            message += " site=" + std::to_string(diagnostic.window.instruction_offset) + " reason=" +
                execution::exclusive_window_error_name(diagnostic.window.reason) +
                " (windows=" + std::to_string(evidence_.exclusive_windows_found) +
                ", proved=" + std::to_string(evidence_.exclusive_windows_proved) + ").";
            throw std::runtime_error(message);
        }
        return proofs;
    }

    void validate() const {
        if (modules_.empty() || modules_.size() > limits_.max_modules ||
            modules_.size() > std::numeric_limits<std::uint32_t>::max())
            fail("Invalid preparation analysis module count.");
        std::size_t pointer_proofs = 0;
        for (std::size_t i = 0; i < modules_.size(); ++i) {
            const auto& module = modules_[i];
            if (!module.image || !module.metadata || module.linked_image.empty() ||
                module.linked_image.size() != module.image->bytes.size() ||
                module.synthetic_base == 0 || module.synthetic_base % 0x1000 != 0 ||
                module.linked_image.size() > std::numeric_limits<std::uint64_t>::max() - module.synthetic_base)
                fail("Invalid preparation analysis image.");
            if (module.relocated_pointers.size() > limits_.max_code_words - pointer_proofs)
                fail("Preparation analysis relocated-pointer limit exceeded.");
            pointer_proofs += module.relocated_pointers.size();
            std::uint64_t previous_slot = 0;
            bool first_slot = true;
            for (const auto& pointer : module.relocated_pointers) {
                if (pointer.image_offset % 8 ||
                    !within(pointer.image_offset, 8, module.linked_image.size()) ||
                    (!first_slot && pointer.image_offset <= previous_slot) ||
                    pointer.target.module_index >= modules_.size() ||
                    pointer.target.image_offset >= modules_[pointer.target.module_index].linked_image.size())
                    fail("Relocated pointer proofs must be sorted unique module-bounded locations.");
                previous_slot = pointer.image_offset;
                first_slot = false;
            }
            for (std::size_t segment_index = 0; segment_index < module.image->segments.size(); ++segment_index) {
                const auto& segment = module.image->segments[segment_index];
                if (segment.kind != static_cast<loader::NsoSegmentKind>(segment_index) ||
                    segment.content_size > segment.mapped_size ||
                    !within(segment.image_offset, segment.mapped_size, module.linked_image.size()))
                    fail("A preparation analysis segment is out of bounds.");
                for (std::size_t other_index = 0; other_index < segment_index; ++other_index) {
                    const auto& other = module.image->segments[other_index];
                    if (segment.mapped_size && other.mapped_size &&
                        segment.image_offset < other.image_offset + other.mapped_size &&
                        other.image_offset < segment.image_offset + segment.mapped_size)
                        fail("Preparation analysis segment mappings overlap.");
                }
            }
            const auto& text = module.image->segment(loader::NsoSegmentKind::text);
            if (text.image_offset % 4 || text.content_size % 4 ||
                !permission(text, loader::SegmentPermission::execute) ||
                permission(text, loader::SegmentPermission::write))
                fail("Preparation requires aligned immutable executable text.");
            for (std::size_t j = 0; j < i; ++j) {
                const auto& other = modules_[j];
                if (module.synthetic_base < other.synthetic_base + other.linked_image.size() &&
                    other.synthetic_base < module.synthetic_base + module.linked_image.size())
                    fail("Preparation analysis images overlap.");
            }
        }
    }

    Target location(std::uint64_t address, bool executable) const {
        for (std::size_t i = 0; i < modules_.size(); ++i) {
            const auto& module = modules_[i];
            if (address < module.synthetic_base ||
                !within(address - module.synthetic_base, 1, module.linked_image.size())) continue;
            const auto offset = address - module.synthetic_base;
            if (executable && (address % 4 || !contains(
                module.image->segment(loader::NsoSegmentKind::text), offset, 4)))
                fail("A control-flow target is outside executable text content.");
            return {static_cast<std::uint32_t>(i), offset};
        }
        fail("A preparation address has no module owner.");
    }

    void enqueue(std::uint64_t address, bool leader) {
        (void)location(address, true);
        if (leader) leaders_.insert(address);
        if (!nodes_.contains(address)) {
            pending_.insert(address);
            if (pending_.size() + nodes_.size() > limits_.max_code_words)
                fail("Preparation analysis code-word limit exceeded.");
        }
    }

    void root(std::uint64_t address) {
        enqueue(address, true);
        roots_.insert(address);
        if (roots_.size() > limits_.max_roots) fail("Preparation analysis root limit exceeded.");
    }

    void collect_roots() {
        std::size_t metadata_entries = 0;
        for (const auto& module : modules_) {
            const auto& metadata = *module.metadata;
            if (module.entry_roots.size() > limits_.max_roots - metadata_entries)
                fail("Preparation analysis explicit-root limit exceeded.");
            metadata_entries += module.entry_roots.size();
            for (const auto entry : module.entry_roots)
                root(module.synthetic_base + entry);
            if (metadata.symbols.size() > limits_.max_roots - metadata_entries)
                fail("Preparation analysis metadata limit exceeded.");
            metadata_entries += metadata.symbols.size();
            for (const auto& symbol : metadata.symbols) {
                if (!symbol.defined() || (symbol.info & 15u) != 2u) continue;
                if (symbol.section >= 0xff00 ||
                    !contains(module.image->segment(loader::NsoSegmentKind::text), symbol.value,
                              std::max<std::uint64_t>(symbol.size, 4)))
                    fail("A function symbol has an unsupported executable extent.");
                root(module.synthetic_base + symbol.value);
            }
            for (const auto entry : {metadata.init, metadata.fini}) {
                if (!entry) continue;
                if (!within(entry, 4, module.linked_image.size()))
                    fail("An initializer entry is outside its module.");
                root(module.synthetic_base + entry);
            }
            for (const auto [offset, size] : {
                    std::pair{metadata.init_array, metadata.init_array_size},
                    std::pair{metadata.fini_array, metadata.fini_array_size}}) {
                if (offset % 8 || size % 8 || !within(offset, size, module.linked_image.size()))
                    fail("An initializer array has invalid bounds or alignment.");
                if (size / 8 > limits_.max_roots - metadata_entries)
                    fail("Preparation analysis metadata limit exceeded.");
                metadata_entries += static_cast<std::size_t>(size / 8);
                for (std::uint64_t at = offset; at < offset + size; at += 8) {
                    const auto address = read(module.linked_image, at, 8);
                    if (address == 0 || address == std::numeric_limits<std::uint64_t>::max()) continue;
                    if (!relocated_value(module, at))
                        fail("An initializer array entry lacks provider-relative relocation provenance.");
                    root(address);
                }
            }
            collect_unwind_roots(module, metadata_entries);
        }
    }

    void collect_unwind_roots(const AnalysisModule& module, std::size_t& metadata_entries) {
        if (!module.image->module_header) return;
        const auto& header = *module.image->module_header;
        const auto begin = header.eh_frame_hdr_start_offset;
        const auto end = header.eh_frame_hdr_end_offset;
        if (begin == end) return;
        const auto& rodata = module.image->segment(loader::NsoSegmentKind::rodata);
        if (begin > end || begin % 4 || end - begin < 12 ||
            !contains(rodata, begin, end - begin) ||
            !permission(rodata, loader::SegmentPermission::read) ||
            permission(rodata, loader::SegmentPermission::write) ||
            permission(rodata, loader::SegmentPermission::execute))
            fail("Invalid unwind metadata header bounds or permissions.");
        // LSB .eh_frame_hdr format: the frame pointer uses its field address;
        // both signed search-table fields use the start of the header instead.
        // https://refspecs.linuxfoundation.org/LSB_2.1.0/LSB-Core-generic/LSB-Core-generic/ehframehdr.html
        if (read(module.linked_image, begin, 4) != 0x3b031b01u)
            fail("Unsupported unwind metadata header version or encodings.");
        const auto count = read(module.linked_image, begin + 8, 4);
        if (count > limits_.max_roots - metadata_entries)
            fail("Preparation analysis unwind-root limit exceeded.");
        if (count != (end - begin - 12) / 8 || (end - begin - 12) % 8)
            fail("Unwind metadata table extent does not match its entry count.");
        metadata_entries += static_cast<std::size_t>(count);
        const auto frame = relative_sdata4(begin + 4, read(module.linked_image, begin + 4, 4));
        if (!frame || *frame % 4 || !contains(rodata, *frame, 4) ||
            (*frame < end && begin < *frame + 4))
            fail("Unwind metadata frame pointer is outside independent read-only storage.");

        std::set<std::uint64_t> referenced_fdes;
        std::vector<std::uint64_t> entries;
        entries.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto at = begin + 12 + i * 8;
            const auto pc = relative_sdata4(begin, read(module.linked_image, at, 4));
            const auto fde = relative_sdata4(begin, read(module.linked_image, at + 4, 4));
            if (!pc || *pc % 4 || !contains(
                    module.image->segment(loader::NsoSegmentKind::text), *pc, 4) ||
                (!entries.empty() && *pc <= entries.back()))
                fail("Unwind metadata roots must be strictly sorted executable locations.");
            if (!fde || *fde % 4 || *fde < *frame || !contains(rodata, *fde, 8) ||
                !referenced_fdes.insert(*fde).second)
                fail("Unwind metadata has an invalid or duplicate FDE pointer.");
            entries.push_back(*pc);
        }

        // Walk framing only, through the last indexed FDE. This establishes
        // actual record boundaries rather than accepting pointers into payloads.
        // A DWARF32 .eh_frame FDE's nonzero CIE pointer is a backward distance
        // from that field. CFI instructions and augmentation data stay opaque.
        // https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.98.pdf (section 4.7)
        std::set<std::uint64_t> cies;
        std::uint64_t cursor = *frame;
        while (!referenced_fdes.empty()) {
            if (unwind_records_ >= limits_.max_unwind_records)
                fail("Preparation analysis unwind-record limit exceeded.");
            ++unwind_records_;
            if (!contains(rodata, cursor, 8))
                fail("Unwind metadata record header is outside read-only storage.");
            const auto length = read(module.linked_image, cursor, 4);
            if (length == 0 || length == 0xffffffffu || length < 4 ||
                (length + 4) % 4 || !contains(rodata, cursor, length + 4))
                fail("Unwind metadata record has unsupported or invalid framing.");
            const auto record_end = cursor + length + 4;
            if (cursor < end && begin < record_end)
                fail("Unwind metadata records overlap the search header.");
            const auto cie_distance = read(module.linked_image, cursor + 4, 4);
            if (cie_distance == 0) {
                if (length < 9 || referenced_fdes.contains(cursor))
                    fail("Unwind metadata indexes a CIE or a truncated CIE record.");
                cies.insert(cursor);
            } else {
                if (length < 12 || cie_distance > cursor + 4 ||
                    !cies.contains(cursor + 4 - cie_distance))
                    fail("Unwind metadata FDE has an invalid backward CIE reference.");
                referenced_fdes.erase(cursor);
            }
            if (!referenced_fdes.empty() && *referenced_fdes.begin() < record_end)
                fail("Unwind metadata FDE pointer does not name a record boundary.");
            cursor = record_end;
        }
        for (const auto offset : entries) root(module.synthetic_base + offset);
    }

    std::uint64_t referenced(const Node& node, std::uint64_t address) const {
        const auto target = a64::referenced_address(node.instruction, address);
        if (!target) fail("A PC-relative address is invalid or overflows.");
        return *target;
    }

    void record_literal(const Node& node, std::uint64_t address) {
        if (node.effects.memory.addressing != a64::AddressingMode::literal ||
            node.effects.memory.operation != a64::MemoryOperation::load) return;
        const auto start = referenced(node, address);
        const auto owner = location(start, false);
        const auto size = node.effects.memory.element_bytes * node.effects.memory.elements;
        if (!size || !within(owner.image_offset, size, modules_[owner.module_index].linked_image.size()))
            fail("A literal reference is outside its module.");
        data_.push_back({start, start + size});
    }

    void drain() {
        while (!pending_.empty()) {
            const auto address = *pending_.begin();
            pending_.erase(pending_.begin());
            if (nodes_.contains(address)) continue;
            const auto owner = location(address, true);
            const auto word = static_cast<std::uint32_t>(read(
                modules_[owner.module_index].linked_image, owner.image_offset, 4));
            if (a64::pointer_authentication(word) == a64::PointerAuthentication::unsupported)
                fail("Pointer authentication form is unsupported by preparation.");
            const Node node{owner, a64::decode(word), a64::decode_register_effects(word)};
            if (!node.effects.decoded()) fail("A reached instruction has unsupported register effects.");
            if (a64::has(node.effects.memory.attributes, a64::MemoryAttribute::exclusive)) {
                const bool load = node.effects.memory.operation == a64::MemoryOperation::load;
                if (load) ++evidence_.exclusive_windows_found;
            }
            nodes_.emplace(address, node);
            record_literal(node, address);
            const auto next = [&] (bool leader = false) {
                if (address > std::numeric_limits<std::uint64_t>::max() - 4)
                    fail("Instruction fallthrough overflows.");
                const auto following = location(address + 4, true);
                if (following.module_index != owner.module_index)
                    fail("Instruction fallthrough crosses module ownership.");
                enqueue(address + 4, leader);
            };
            switch (node.instruction.opcode) {
            case Opcode::b:
                enqueue(referenced(node, address), true); break;
            case Opcode::bl:
                enqueue(referenced(node, address), true); next(true); break;
            case Opcode::b_cond: case Opcode::bc_cond:
                enqueue(referenced(node, address), true);
                if (node.instruction.condition != a64::Condition::al &&
                    node.instruction.condition != a64::Condition::nv) next(true);
                break;
            case Opcode::cbz: case Opcode::cbnz: case Opcode::tbz: case Opcode::tbnz:
                enqueue(referenced(node, address), true); next(true); break;
            case Opcode::br: case Opcode::blr:
                if (indirect_.size() >= limits_.max_indirect_sites)
                    fail("Preparation analysis indirect-site limit exceeded.");
                indirect_.push_back(address);
                if (node.instruction.opcode == Opcode::blr) next(true);
                break;
            case Opcode::ret: case Opcode::retab:
                if (node.instruction.reg != 30) refuse_return(node, ReturnError::unsupported_return_register);
                break;
            case Opcode::brk: case Opcode::hvc: case Opcode::smc: case Opcode::hlt:
                fail("A reached source trap is unsupported by preparation.");
            case Opcode::svc:
                next(true); break;
            default:
                if (node.effects.control_flow != a64::ControlFlow::none)
                    fail("A reached control-flow instruction is unsupported by preparation.");
                next(); break;
            }
        }
    }

    [[noreturn]] void refuse_return(const Node& node, ReturnError reason) {
        evidence_.return_diagnostics.push_back(
            {node.location.module_index, node.location.image_offset, reason});
        throw std::runtime_error("Instruction lacks a proven RET X30 target/stack contract: module=" +
            std::to_string(node.location.module_index) + " site=" +
            std::to_string(node.location.image_offset) + " reason=" +
            std::to_string(static_cast<unsigned>(reason)) + ".");
    }

    // Follow actual proven return values, not an assumed callee-preserved LR.
    // Roots have independent symbolic entry links. Calls execute their bodies
    // with a concrete continuation; no ABI register-preservation summary is used.
    void check_return_provenance() {
        // If calls are the only LR writers, every possible return is already
        // an entry link or a discovered continuation, without any saved state.
        if (std::none_of(nodes_.begin(), nodes_.end(), [](const auto& entry) {
                const auto& node = entry.second;
                return (node.effects.writes & a64::register_bit(30)) &&
                    node.instruction.opcode != Opcode::bl && node.instruction.opcode != Opcode::blr;
            })) return;
        enum class Kind { unknown, entry_link, call_link, stack, maybe_stack };
        struct Value {
            Kind kind = Kind::unknown;
            std::uint64_t link = 0;
            std::int64_t offset = 0;
            bool operator==(const Value&) const = default;
            bool is_stack() const { return kind == Kind::stack || kind == Kind::maybe_stack; }
            bool is_link() const { return kind == Kind::entry_link || kind == Kind::call_link; }
        };
        struct State {
            std::array<Value, 32> registers{};
            // Only whole eight-byte value provenance is retained. Unknown
            // values here are disjoint external data under the entry contract;
            // unproven stack loads instead carry maybe_stack provenance.
            // Every overlapping write, including narrow/vector/exclusive
            // writes, kills the previous slot proof.
            std::map<std::int64_t, Value> slots;
        };
        const auto join = [](Value left, Value right) {
            if (left == right) return left;
            return Value{left.is_stack() || right.is_stack() ? Kind::maybe_stack : Kind::unknown};
        };
        constexpr std::int64_t stack_bound = 16 * 1024 * 1024;
        const auto shifted = [](Value value, std::int64_t delta) {
            if (!value.is_stack()) return Value{};
            if (value.kind != Kind::stack || delta < -stack_bound || delta > stack_bound ||
                value.offset + delta < -stack_bound || value.offset + delta > stack_bound)
                return Value{Kind::maybe_stack};
            value.offset += delta;
            return value;
        };
        std::size_t steps = 0;
        for (const auto entry : roots_) {
            std::map<std::uint64_t, State> incoming;
            std::set<std::uint64_t> pending;
            if (limits_.max_return_states == 0)
                refuse_return(nodes_.at(entry), ReturnError::state_limit);
            State initial;
            initial.registers[30].kind = Kind::entry_link;
            initial.registers[31].kind = Kind::stack;
            incoming.emplace(entry, initial);
            pending.insert(entry);
            const auto merge = [&](std::uint64_t address, const State& state) {
                const auto found = incoming.find(address);
                if (found == incoming.end()) {
                    if (incoming.size() >= limits_.max_return_states)
                        refuse_return(nodes_.at(address), ReturnError::state_limit);
                    incoming.emplace(address, state);
                    pending.insert(address);
                    return;
                }
                auto& before = found->second;
                bool changed = false;
                for (unsigned reg = 0; reg < 32; ++reg) {
                    const auto value = join(before.registers[reg], state.registers[reg]);
                    if (value != before.registers[reg]) {
                        before.registers[reg] = value;
                        changed = true;
                    }
                }
                for (auto it = before.slots.begin(); it != before.slots.end();) {
                    const auto other = state.slots.find(it->first);
                    if (other == state.slots.end() || other->second != it->second) {
                        it = before.slots.erase(it);
                        changed = true;
                    } else ++it;
                }
                if (changed) pending.insert(address);
            };
            while (!pending.empty()) {
                const auto address = *pending.begin();
                pending.erase(pending.begin());
                const auto& node = nodes_.at(address);
                if (steps++ >= limits_.max_return_steps)
                    refuse_return(node, ReturnError::work_limit);
                const auto before = incoming.at(address);
                auto after = before;
                const auto& effects = node.effects;
                const auto& memory = effects.memory;
                const auto word = node.instruction.word;
                for (unsigned reg = 0; reg < 31; ++reg)
                    if (effects.writes & a64::register_bit(reg)) after.registers[reg] = {};
                if (a64::writes_stack_pointer(effects)) after.registers[31] = {};

                // Exact 64-bit MOV register and ADD/SUB immediate on stack
                // addresses only. Arithmetic on a link never preserves it.
                if ((word & 0xffe0ffe0u) == 0xaa0003e0u) {
                    const auto source = (word >> 16) & 31u;
                    if ((word & 31u) != 31)
                        after.registers[word & 31u] = source == 31 ? Value{} : before.registers[source];
                } else if ((word & 0xbf800000u) == 0x91000000u) {
                    const auto source = before.registers[(word >> 5) & 31u];
                    const auto immediate = std::int64_t((word >> 10) & 0xfffu) <<
                        ((word & (1u << 22)) ? 12 : 0);
                    if (source.is_stack())
                        after.registers[word & 31u] = shifted(source, (word & (1u << 30)) ? -immediate : immediate);
                } else if (memory.operation == a64::MemoryOperation::none &&
                           effects.control_flow == a64::ControlFlow::none) {
                    bool stack_input = false;
                    for (unsigned i = 0; i < effects.operand_count; ++i) {
                        const auto& operand = effects.operands[i];
                        if (a64::includes_read(operand.access) && operand.is_general() &&
                            !operand.names_zero_register() && before.registers[operand.number].is_stack())
                            stack_input = true;
                    }
                    if (stack_input) {
                        if (!effects.writes && !effects.writes_flags)
                            refuse_return(node, ReturnError::stack_pointer_escape);
                        for (unsigned i = 0; i < effects.operand_count; ++i) {
                            const auto& operand = effects.operands[i];
                            if (!a64::includes_write(operand.access)) continue;
                            if (!operand.is_general()) refuse_return(node, ReturnError::stack_pointer_escape);
                            if (!operand.names_zero_register())
                                after.registers[operand.number] = {Kind::maybe_stack};
                        }
                    }
                }

                if (memory.operation == a64::MemoryOperation::load ||
                    memory.operation == a64::MemoryOperation::store) {
                    Value base;
                    unsigned base_register = 32;
                    bool index = false;
                    for (unsigned i = 0; i < effects.operand_count; ++i) {
                        const auto& operand = effects.operands[i];
                        if (operand.role == a64::OperandRole::base) {
                            base_register = operand.number;
                            base = before.registers[base_register];
                        }
                        if (operand.role == a64::OperandRole::index) {
                            index = true;
                            if (before.registers[operand.number].is_stack()) base = {Kind::maybe_stack};
                        }
                    }
                    const auto access = index ? Value{base.is_stack() ? Kind::maybe_stack : Kind::unknown} : shifted(base,
                        memory.addressing == a64::AddressingMode::post_index ? 0 : memory.offset);
                    if (memory.writeback && base_register < 32)
                        after.registers[base_register] = index
                            ? Value{base.is_stack() ? Kind::maybe_stack : Kind::unknown} : shifted(base, memory.offset);
                    const auto sp = memory.addressing == a64::AddressingMode::pre_index && base_register == 31
                        ? after.registers[31] : before.registers[31];
                    const auto bytes = std::int64_t(memory.element_bytes) * memory.elements;
                    const bool private_slot = base.is_stack() && access.kind == Kind::stack &&
                        sp.kind == Kind::stack && access.offset >= sp.offset &&
                        bytes > 0 && access.offset <= -bytes;
                    if (memory.operation == a64::MemoryOperation::store) {
                        if (base.is_stack() && !private_slot) after.slots.clear();
                        if (private_slot) {
                            for (auto it = after.slots.begin(); it != after.slots.end();) {
                                if (it->first < access.offset + bytes && access.offset < it->first + 8)
                                    it = after.slots.erase(it);
                                else ++it;
                            }
                        }
                    }
                    unsigned element = 0;
                    for (unsigned i = 0; i < effects.operand_count; ++i) {
                        const auto& operand = effects.operands[i];
                        if (operand.role != a64::OperandRole::transfer) continue;
                        const auto slot = access.offset + element++ * memory.element_bytes;
                        const bool exact = private_slot && operand.is_general() && operand.bits == 64 &&
                            memory.element_bytes == 8 &&
                            !a64::has(memory.attributes, a64::MemoryAttribute::exclusive);
                        if (memory.operation == a64::MemoryOperation::store) {
                            const auto value = operand.is_general() && !operand.names_zero_register()
                                ? before.registers[operand.number] : Value{};
                            // Stack aliases may be copied in registers, but
                            // spilling them would require memory alias provenance
                            // beyond this bounded saved-link proof.
                            if (value.is_stack()) refuse_return(node, ReturnError::stack_pointer_escape);
                            if (exact) after.slots[slot] = value;
                        } else if (base.is_stack()) {
                            // Unproven stack contents can themselves be an alias
                            // into the frame. Do not turn such a load into an
                            // assumed-disjoint external pointer.
                            if (!operand.is_general())
                                refuse_return(node, ReturnError::stack_pointer_escape);
                            if (!operand.names_zero_register()) {
                                after.registers[operand.number] = {Kind::maybe_stack};
                                const auto saved = before.slots.find(slot);
                                if (exact && saved != before.slots.end())
                                    after.registers[operand.number] = saved->second;
                            }
                        }
                    }
                }
                if (after.registers[31].kind != Kind::stack || after.registers[31].offset > 0)
                    refuse_return(node, ReturnError::unproven_stack_pointer);
                // Deallocated storage cannot later recreate a saved-link proof.
                for (auto it = after.slots.begin(); it != after.slots.end();) {
                    if (it->first < after.registers[31].offset) it = after.slots.erase(it);
                    else ++it;
                }
                if (after.slots.size() > limits_.max_return_stack_slots)
                    refuse_return(node, ReturnError::stack_slot_limit);
                const auto opcode = node.instruction.opcode;
                const bool call = opcode == Opcode::bl || opcode == Opcode::blr;
                if (call) after.registers[30] = {Kind::call_link, address + 4};
                if (!call && (effects.writes & a64::register_bit(30)) && !after.registers[30].is_link())
                    refuse_return(node, ReturnError::unproven_link_write);
                switch (opcode) {
                case Opcode::bl: case Opcode::b:
                    merge(referenced(node, address), after); break;
                case Opcode::blr: case Opcode::br:
                    merge(resolved_.at(address), after); break;
                case Opcode::b_cond: case Opcode::bc_cond:
                    merge(referenced(node, address), after);
                    if (node.instruction.condition != a64::Condition::al &&
                        node.instruction.condition != a64::Condition::nv) merge(address + 4, after);
                    break;
                case Opcode::cbz: case Opcode::cbnz: case Opcode::tbz: case Opcode::tbnz:
                    merge(referenced(node, address), after); merge(address + 4, after); break;
                case Opcode::ret: case Opcode::retab:
                    if (!after.registers[30].is_link()) refuse_return(node, ReturnError::unproven_return);
                    if (after.registers[30].kind == Kind::call_link) merge(after.registers[30].link, after);
                    break;
                case Opcode::svc:
                    // Trusted SVC handlers preserve the private frame and LR.
                    // The callback API can change any general register. Retain
                    // only possible stack aliasing, never an exact saved link
                    // or stack offset, across this opaque host operation.
                    for (unsigned reg = 0; reg < 30; ++reg) {
                        const bool stack_alias = after.registers[reg].is_stack();
                        if (reg < 18 && stack_alias) refuse_return(node, ReturnError::stack_pointer_escape);
                        after.registers[reg] = {stack_alias ? Kind::maybe_stack : Kind::unknown};
                    }
                    merge(address + 4, after); break;
                default:
                    merge(address + 4, after); break;
                }
            }
        }
    }

    std::optional<std::uint64_t> relocated_value(const AnalysisModule& module,
                                                std::uint64_t offset) const {
        const auto found = std::lower_bound(module.relocated_pointers.begin(), module.relocated_pointers.end(),
            offset, [](const RelocatedPointer& pointer, std::uint64_t slot) { return pointer.image_offset < slot; });
        if (found == module.relocated_pointers.end() || found->image_offset != offset) return {};
        const auto& target = modules_[found->target.module_index];
        const auto expected = target.synthetic_base + found->target.image_offset;
        const auto actual = read(module.linked_image, offset, 8);
        return actual == expected ? std::optional<std::uint64_t>{actual} : std::nullopt;
    }

    std::optional<std::uint64_t> immutable_pointer(std::uint64_t address) const {
        for (const auto& module : modules_) {
            if (address < module.synthetic_base) continue;
            const auto offset = address - module.synthetic_base;
            const auto& rodata = module.image->segment(loader::NsoSegmentKind::rodata);
            if (!contains(rodata, offset, 8)) continue;
            if (address % 8 || !permission(rodata, loader::SegmentPermission::read) ||
                permission(rodata, loader::SegmentPermission::write) ||
                permission(rodata, loader::SegmentPermission::execute)) return {};
            return relocated_value(module, offset);
        }
        return {};
    }

    std::optional<std::uint64_t> value_before(std::uint64_t address, unsigned reg,
                                             unsigned remaining) const {
        if (!remaining || reg >= 31 || address < 4 || leaders_.contains(address)) return {};
        const auto found = nodes_.find(address - 4);
        if (found == nodes_.end()) return {};
        const auto& node = found->second;
        if (node.location.module_index != nodes_.at(address).location.module_index ||
            node.effects.control_flow != a64::ControlFlow::none) return {};
        const auto& instruction = node.instruction;
        const auto word = instruction.word;
        if ((instruction.opcode == Opcode::adr || instruction.opcode == Opcode::adrp) &&
            instruction.reg == reg) return referenced(node, address - 4);
        if (instruction.opcode == Opcode::ldr_literal_x && instruction.reg == reg)
            return immutable_pointer(referenced(node, address - 4));
        // The exact masks admit only 64-bit ADD immediate without flags and
        // 64-bit LDR unsigned offset. SP and zero-register producers are refused.
        if ((word & 0xff800000u) == 0x91000000u && (word & 31u) == reg) {
            const auto source = value_before(address - 4, (word >> 5) & 31u, remaining - 1);
            const auto immediate = std::uint64_t((word >> 10) & 0xfffu) << ((word & (1u << 22)) ? 12 : 0);
            if (!source || immediate > std::numeric_limits<std::uint64_t>::max() - *source) return {};
            return *source + immediate;
        }
        if ((word & 0xffc00000u) == 0xf9400000u && (word & 31u) == reg) {
            const auto source = value_before(address - 4, (word >> 5) & 31u, remaining - 1);
            const auto immediate = std::uint64_t((word >> 10) & 0xfffu) * 8;
            if (!source || immediate > std::numeric_limits<std::uint64_t>::max() - *source) return {};
            return immutable_pointer(*source + immediate);
        }
        return {};
    }

    std::uint64_t resolve(std::uint64_t address) const {
        const auto& node = nodes_.at(address);
        const auto target = value_before(address, node.instruction.reg, 3);
        if (!target) fail("An indirect transfer lacks a local immutable singleton target proof.");
        (void)location(*target, true);
        return *target;
    }

    std::optional<std::uint64_t> bootstrap_value_before(std::uint64_t address,
        unsigned reg, unsigned remaining = 8) const {
        if (!remaining || reg >= 31 || address < 4 || leaders_.contains(address)) return {};
        if (const auto value = value_before(address, reg, 3)) return value;
        const auto previous = nodes_.find(address - 4);
        if (previous == nodes_.end() ||
            previous->second.location.module_index != nodes_.at(address).location.module_index ||
            previous->second.effects.control_flow != a64::ControlFlow::none ||
            (previous->second.effects.writes & (a64::RegisterMask{1} << reg))) return {};
        return bootstrap_value_before(address - 4, reg, remaining - 1);
    }

    bool overlaps_relocated_pointer(std::uint64_t address, std::uint64_t size) const {
        const auto owner = location(address, false);
        const auto& module = modules_[owner.module_index];
        if (!within(owner.image_offset, size, module.linked_image.size()))
            fail("A pointer-bearing data reference is outside its module.");
        const auto begin = owner.image_offset > 7 ? owner.image_offset - 7 : 0;
        const auto found = std::lower_bound(module.relocated_pointers.begin(),
            module.relocated_pointers.end(), begin,
            [](const RelocatedPointer& pointer, std::uint64_t offset) {
                return pointer.image_offset < offset;
            });
        return found != module.relocated_pointers.end() &&
               found->image_offset < owner.image_offset + size;
    }

    // Track module-address provenance across the discovered graph. Unlike an
    // indirect-target proof, losing an exact value must not erase provenance:
    // an unproven data use is refused rather than assumed to be outside code.
    void check_address_escapes() {
        using Mask = a64::RegisterMask;
        std::map<std::uint64_t, Mask> incoming;
        std::set<std::uint64_t> pending;
        std::vector<std::uint64_t> continuations;
        for (const auto& [address, node] : nodes_) {
            incoming.emplace(address, 0);
            pending.insert(address);
            if (node.instruction.opcode == Opcode::bl || node.instruction.opcode == Opcode::blr)
                continuations.push_back(address + 4);
        }
        Mask returned = 0;
        std::size_t steps = 0;
        const auto merge = [&](std::uint64_t address, Mask mask) {
            auto& state = incoming.at(address);
            if ((state | mask) != state) {
                state |= mask;
                pending.insert(address);
            }
        };
        while (!pending.empty()) {
            if (steps++ >= limits_.max_address_steps)
                fail("Preparation analysis address-provenance work limit exceeded.");
            const auto address = *pending.begin();
            pending.erase(pending.begin());
            const auto& node = nodes_.at(address);
            const auto& effects = node.effects;
            const auto& memory = effects.memory;
            const auto before = incoming.at(address);
            Mask after = before & ~effects.writes;
            std::optional<std::uint64_t> data_address;
            std::optional<unsigned> stored_address_register;
            Mask address_registers = 0;
            for (unsigned i = 0; i < effects.operand_count; ++i) {
                const auto& operand = effects.operands[i];
                if (!operand.is_general() || operand.number >= 31) continue;
                const auto bit = Mask{1} << operand.number;
                if (operand.role == a64::OperandRole::base ||
                    operand.role == a64::OperandRole::index) address_registers |= bit;
                if (memory.operation == a64::MemoryOperation::store &&
                    operand.role == a64::OperandRole::transfer && (before & bit)) {
                    if (stored_address_register)
                        fail("A module address escapes through a stored data reference.");
                    stored_address_register = operand.number;
                }
                if (operand.role != a64::OperandRole::base) continue;
                if (memory.addressing == a64::AddressingMode::register_offset) continue;
                const auto base = modules_[node.location.module_index].guest_relocations
                    ? bootstrap_value_before(address, operand.number)
                    : value_before(address, operand.number, 3);
                if (!base) continue;
                const auto offset = memory.addressing == a64::AddressingMode::post_index
                    ? 0 : memory.offset;
                if (offset >= 0 && std::uint64_t(offset) <=
                        std::numeric_limits<std::uint64_t>::max() - *base)
                    data_address = *base + std::uint64_t(offset);
                else if (offset < 0 && std::uint64_t(-(offset + 1)) + 1 <= *base)
                    data_address = *base - (std::uint64_t(-(offset + 1)) + 1);
            }
            if (memory.addressing == a64::AddressingMode::literal)
                data_address = referenced(node, address);
            if ((before & address_registers) &&
                (memory.operation == a64::MemoryOperation::load ||
                 memory.operation == a64::MemoryOperation::store)) {
                if (!data_address)
                    fail("A module address has an unproven register-addressed data reference.");
                const auto owner = location(*data_address, false);
                const auto size = memory.element_bytes * memory.elements;
                if (!size || !within(owner.image_offset, size,
                                     modules_[owner.module_index].linked_image.size()))
                    fail("A register-addressed data reference is outside its module.");
                data_.push_back({*data_address, *data_address + size});
            }
            if (stored_address_register) {
                if (!modules_[node.location.module_index].guest_relocations ||
                    !data_address || memory.element_bytes != 8 || memory.elements != 1)
                    fail("A module address escapes through a stored data reference.");
                const auto slot = location(*data_address, false);
                const auto& owner = modules_[slot.module_index];
                const auto& data = owner.image->segment(loader::NsoSegmentKind::data);
                const auto expected = relocated_value(owner, slot.image_offset);
                const auto written = bootstrap_value_before(address, *stored_address_register);
                if (!owner.guest_relocations || !contains(data, slot.image_offset, 8) ||
                    !permission(data, loader::SegmentPermission::write) ||
                    !expected || !written || *expected != *written)
                    fail("A module address escapes through a stored data reference.");
            }
            if (node.instruction.opcode == Opcode::adr || node.instruction.opcode == Opcode::adrp) {
                after |= effects.writes;
            } else if (memory.operation == a64::MemoryOperation::load) {
                // Relocations seed provenance even in writable storage and
                // through partial reads. These are may-address facts, not the
                // immutable singleton values required for indirect transfers.
                const auto size = memory.element_bytes * memory.elements;
                if (data_address && overlaps_relocated_pointer(*data_address, size)) {
                    for (unsigned i = 0; i < effects.operand_count; ++i) {
                        const auto& operand = effects.operands[i];
                        if (operand.role != a64::OperandRole::transfer) continue;
                        if (!operand.is_general())
                            fail("A relocated pointer escapes through a vector data reference.");
                        if (operand.number < 31) after |= Mask{1} << operand.number;
                    }
                }
                // Writeback preserves the base's provenance even though a
                // transferred value normally replaces its destination.
                if (memory.writeback) after |= before & address_registers;
            } else if (memory.operation == a64::MemoryOperation::store) {
                if (memory.writeback) after |= before & address_registers;
            } else if (effects.control_flow == a64::ControlFlow::none && (effects.reads & before)) {
                if (!effects.writes)
                    fail("A module address escapes through an unsupported data reference use.");
                after |= effects.writes;
                for (unsigned i = 0; i < effects.operand_count; ++i) {
                    const auto& operand = effects.operands[i];
                    if (a64::includes_write(operand.access) &&
                        (!operand.is_general() || operand.names_stack_pointer()))
                        fail("A module address escapes the general-register data reference proof.");
                }
            }
            switch (node.instruction.opcode) {
            case Opcode::b:
                merge(referenced(node, address), after); break;
            case Opcode::bl:
                merge(referenced(node, address), after);
                merge(address + 4, after | returned); break;
            case Opcode::b_cond: case Opcode::bc_cond:
                merge(referenced(node, address), after);
                if (node.instruction.condition != a64::Condition::al &&
                    node.instruction.condition != a64::Condition::nv) merge(address + 4, after);
                break;
            case Opcode::cbz: case Opcode::cbnz: case Opcode::tbz: case Opcode::tbnz:
                merge(referenced(node, address), after); merge(address + 4, after); break;
            case Opcode::br: case Opcode::blr:
                merge(resolved_.at(address), after);
                if (node.instruction.opcode == Opcode::blr) merge(address + 4, after | returned);
                break;
            case Opcode::ret: case Opcode::retab:
                if (before & (Mask{1} << 30))
                    fail("RET X30 uses a constructed module address instead of the callable return contract.");
                if ((returned | after) != returned) {
                    returned |= after;
                    for (const auto continuation : continuations) merge(continuation, returned);
                }
                break;
            default:
                merge(address + 4, after); break;
            }
        }
    }

    void check_data_overlap() const {
        for (const auto& range : data_) {
            auto code = nodes_.lower_bound(range.begin & ~std::uint64_t{3});
            if (code != nodes_.end() && code->first < range.end)
                fail("Reached code overlaps a literal or register-addressed data reference.");
        }
    }

    std::span<const AnalysisModule> modules_;
    const AnalysisLimits& limits_;
    std::set<std::uint64_t> pending_, leaders_, roots_;
    std::map<std::uint64_t, Node> nodes_;
    std::vector<std::uint64_t> indirect_;
    std::map<std::uint64_t, std::uint64_t> resolved_;
    std::vector<DataRange> data_;
    std::vector<std::vector<std::uint64_t>> function_entries_, external_entries_;
    std::size_t unwind_records_ = 0;
    AnalysisResult evidence_;
};
} // namespace

AnalysisResult analyze_modules(std::span<const AnalysisModule> modules, const AnalysisLimits& limits) {
    Analysis analysis(modules, limits);
    try {
        return analysis.run();
    } catch (const std::exception& exception) {
        return analysis.failure_result(exception.what());
    }
}

} // namespace armigo::runtime::preparation
