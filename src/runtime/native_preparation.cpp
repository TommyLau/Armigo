// ============================================================================
// Armigo Runtime - Native Module Preparation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "native_preparation.h"

#include <armigo/aarch64/instruction.h>
#include <armigo/aarch64/register_effects.h>
#include <armigo/execution/conditional_x18_patch.h>
#include <armigo/execution/embedded_patch.h>
#include <armigo/execution/indirect_x18_patch.h>
#include <armigo/execution/gate_patch.h>
#include <armigo/linker/dynamic_module.h>

#include <algorithm>
#include <array>
#include <optional>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace armigo::runtime::native {
namespace {

using aarch64::Opcode;
using aarch64::Operation;

bool exclusive_load(Operation operation) noexcept {
    switch (operation) {
    case Operation::ldxrb: case Operation::ldxrh: case Operation::ldxr:
    case Operation::ldaxrb: case Operation::ldaxrh: case Operation::ldaxr:
    case Operation::ldxp: case Operation::ldaxp:
        return true;
    default:
        return false;
    }
}

bool exclusive_store(Operation operation) noexcept {
    switch (operation) {
    case Operation::stxrb: case Operation::stxrh: case Operation::stxr:
    case Operation::stlxrb: case Operation::stlxrh: case Operation::stlxr:
    case Operation::stxp: case Operation::stlxp:
        return true;
    default:
        return false;
    }
}

// The system instruction class (1101 0101 00 L op0 op1 CRn CRm op2 Rt)
// excluding hints and barriers, which run natively.
bool system_class(std::uint32_t word) noexcept {
    if ((word & 0xffc00000u) != 0xd5000000u) return false;
    const auto field = (word >> 12) & 0x3ffu;  // L:op0:op1:CRn
    return field != 0x032u && field != 0x033u;
}

struct UnwindValue {
    std::uint64_t bits;
    bool signed_value;
};

UnwindValue unwind_value(std::span<const std::byte> bytes, std::uint64_t& cursor,
                         std::uint64_t limit, unsigned encoding) {
    const auto format = encoding & 15u;
    const unsigned width = format == 3 || format == 11 ? 4 :
                           format == 0 || format == 4 || format == 12 ? 8 : 0;
    if (!width || cursor > limit || width > limit - cursor || limit > bytes.size())
        throw std::runtime_error("Unsupported or truncated unwind-header value.");
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + cursor, width);
    cursor += width;
    const bool signed_value = format == 11 || format == 12;
    if (signed_value && width == 4)
        value = static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(value)));
    return {value, signed_value};
}

std::uint64_t unwind_pointer(const ModulePreparationView& view, std::uint64_t& cursor,
                             std::uint64_t limit, unsigned encoding, std::uint64_t header) {
    const auto at = cursor;
    if (encoding & 0x80u) throw std::runtime_error("Indirect unwind-header pointers are unsupported.");
    const auto value = unwind_value(view.image, cursor, limit, encoding);
    const auto application = encoding & 0x70u;
    if (!application) return value.bits;
    if (application != 0x10u && application != 0x30u)
        throw std::runtime_error("Unsupported unwind-header pointer base.");
    const auto base = view.base_address + (application == 0x10u ? at : header);
    if (value.signed_value && static_cast<std::int64_t>(value.bits) < 0) {
        const auto magnitude = std::uint64_t{0} - value.bits;
        if (magnitude > base) throw std::runtime_error("Unwind pointer underflows its address space.");
        return base - magnitude;
    }
    if (value.bits > std::numeric_limits<std::uint64_t>::max() - base)
        throw std::runtime_error("Unwind pointer overflows its address space.");
    return base + value.bits;
}

template<class T>
void sort_unique(std::vector<T>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

} // namespace

WordRole classify_word(std::uint32_t word) noexcept {
    WordRole role;
    const auto instruction = aarch64::decode(word);
    const auto effects = aarch64::decode_register_effects(word);
    const bool known = effects.status == aarch64::EffectStatus::decoded ||
                       effects.status == aarch64::EffectStatus::unpredictable;
    role.uses_x18 = known && ((effects.reads | effects.writes) & (1u << 18));
    switch (instruction.opcode) {
    case Opcode::svc: case Opcode::hvc: case Opcode::smc: case Opcode::hlt:
    case Opcode::cache_maintenance:
        role.system = true;
        break;
    case Opcode::mrs: case Opcode::msr_register: {
        // Flags and floating-point control registers are the guest's own.
        const auto reg = aarch64::identify_system_register(instruction.system_register);
        role.system = reg != aarch64::SystemRegister::nzcv && reg != aarch64::SystemRegister::fpcr &&
                      reg != aarch64::SystemRegister::fpsr;
        break;
    }
    case Opcode::unclassified:
        role.system = system_class(word);
        // Unconditional branch (register) and exception generation classes.
        role.unknown_control = (word & 0xfe000000u) == 0xd6000000u ||
                               (word & 0xff000000u) == 0xd4000000u;
        break;
    default:
        break;
    }
    role.exclusive_load = exclusive_load(effects.operation);
    role.exclusive_store = exclusive_store(effects.operation);
    role.clear_exclusive = effects.operation == Operation::clrex;
    return role;
}

ModuleRoots module_roots(const ModulePreparationView& view, std::uint64_t text_begin,
                         std::uint64_t text_size, const std::optional<loader::ModuleHeaderInfo>& header,
                         std::size_t maximum_functions) {
    if (text_begin > view.image.size() || text_size > view.image.size() - text_begin ||
        text_begin % 4 || text_size % 4 ||
        view.base_address > std::numeric_limits<std::uint64_t>::max() - view.image.size())
        throw std::runtime_error("Preparation executable bounds are invalid.");
    ModuleRoots result;
    const auto first = view.base_address + text_begin;
    const auto end = first + text_size;
    const auto owns = [&](std::uint64_t address) { return address >= first && address < end; };
    const auto local = [&](std::uint64_t address) {
        if (!owns(address) || address % 4)
            throw std::runtime_error("Declared function root is outside original executable text.");
        result.local.push_back(address);
    };
    for (const auto& symbol : view.metadata.symbols) {
        if (!symbol.defined() || symbol.section >= 0xff00 || (symbol.info & 15u) != 2u) continue;
        if (symbol.value >= view.image.size())
            throw std::runtime_error("Defined function offset is outside its module.");
        local(view.base_address + symbol.value);
    }
    if (view.metadata.init) local(view.base_address + view.metadata.init);
    // Constructor pointers are read after relocation.
    const auto array = view.metadata.init_array;
    const auto size = view.metadata.init_array_size;
    if (size % 8 || array > view.image.size() || size > view.image.size() - array)
        throw std::runtime_error("Relocated constructor array is outside its module.");
    for (std::uint64_t offset = 0; offset < size; offset += 8) {
        std::uint64_t target;
        std::memcpy(&target, view.image.data() + array + offset, sizeof(target));
        if (target == 0 || target % 4)
            throw std::runtime_error("Constructor pointer is null or misaligned.");
        if (owns(target)) result.local.push_back(target);
        else result.required_foreign_entries.push_back(target);
    }
    if (header && header->eh_frame_hdr_end_offset > header->eh_frame_hdr_start_offset) {
        const auto start = header->eh_frame_hdr_start_offset;
        const auto limit = header->eh_frame_hdr_end_offset;
        if (start > view.image.size() || limit > view.image.size() || limit - start < 4 ||
            std::to_integer<unsigned>(view.image[start]) != 1)
            throw std::runtime_error("Unsupported or invalid unwind-header extent/version.");
        const unsigned pointer_encoding = std::to_integer<unsigned>(view.image[start + 1]);
        const unsigned count_encoding = std::to_integer<unsigned>(view.image[start + 2]);
        const unsigned table_encoding = std::to_integer<unsigned>(view.image[start + 3]);
        auto cursor = start + 4;
        if (pointer_encoding != 0xff)
            static_cast<void>(unwind_pointer(view, cursor, limit, pointer_encoding, start));
        if ((count_encoding & 0xf0u) != 0 || table_encoding == 0xff)
            throw std::runtime_error("Unsupported unwind-header index encoding.");
        const auto count = unwind_value(view.image, cursor, limit, count_encoding);
        if ((count.signed_value && static_cast<std::int64_t>(count.bits) < 0) ||
            count.bits > maximum_functions)
            throw std::runtime_error("Unwind-header index exceeds its function limit.");
        for (std::uint64_t i = 0; i != count.bits; ++i) {
            const auto address = unwind_pointer(view, cursor, limit, table_encoding, start);
            static_cast<void>(unwind_pointer(view, cursor, limit, table_encoding, start));
            local(address);
            ++result.unwind_roots;
        }
    }
    for (const auto& relocation : view.metadata.relocations) {
        if (relocation.kind != linker::RelocationKind::jump_slot &&
            relocation.kind != linker::RelocationKind::global_data &&
            relocation.kind != linker::RelocationKind::absolute64) continue;
        if (relocation.symbol >= view.metadata.symbols.size())
            throw std::runtime_error("Linked function relocation has no symbol.");
        const auto& symbol = view.metadata.symbols[relocation.symbol];
        if (relocation.kind != linker::RelocationKind::jump_slot && (symbol.info & 15u) != 2u) continue;
        if (relocation.target > view.image.size() || view.image.size() - relocation.target < 8)
            throw std::runtime_error("Linked function relocation is outside its module.");
        std::uint64_t target;
        std::memcpy(&target, view.image.data() + relocation.target, 8);
        // An undefined weak import stays zero, as the system loader leaves it;
        // callers test it before calling, so it is no root.
        if (!target && !symbol.defined() && (symbol.info >> 4) == 2u && relocation.addend == 0) continue;
        if (!target || target % 4)
            throw std::runtime_error("Linked function target is null or misaligned.");
        if (owns(target)) local(target);
        else {
            result.external_leaves.push_back({target, true});
            result.required_foreign_entries.push_back(target);
        }
    }
    sort_unique(result.local);
    sort_unique(result.required_foreign_entries);
    std::sort(result.external_leaves.begin(), result.external_leaves.end(),
        [](const auto& a, const auto& b) { return a.address < b.address; });
    result.external_leaves.erase(std::unique(result.external_leaves.begin(), result.external_leaves.end(),
        [](const auto& a, const auto& b) { return a.address == b.address; }), result.external_leaves.end());
    if (result.local.size() > maximum_functions)
        throw std::runtime_error("Combined module function roots exceed the configured limit.");
    return result;
}

namespace {

// BR X18, and the table a byte index can reach from its ADR base.
constexpr std::uint32_t br_x18 = 0xd61f0240u;
constexpr std::uint32_t jump_table_entries = 256;

// The words a switch through virtual x18 can reach: an ADR X18 base and an
// entry count (a power of two, as the table trampoline requires).
struct JumpTable {
    std::uint64_t base_offset = 0;
    std::uint32_t entries = 0;
};

std::int64_t sign_extend(std::uint64_t value, unsigned bits) noexcept {
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    return static_cast<std::int64_t>((value ^ sign) - sign);
}

// Finds the table of the BR X18 at module `offset`. `word_at` returns the
// original text word at a module offset; `image` is the whole module image
// (empty: tables read from it are not recognised).
//
// A byte index: ADR X18, base; LDRB Wn; ADD X18, X18, Xn, LSL #2; at most one
// word that leaves x18 alone; BR X18. The target is one of 256 words from base.
//
// A halfword index read from a bounded table, as compilers lower a switch:
// CMP Wi, #n; B.HI or B.HS; ADRP Xt; ADD Xt, Xt, #lo12; ADR X18, base;
// LDRH Wn, [Xt, Xi, LSL #1] (or UXTW #1); ADD X18, X18, Xn, LSL #2; at most
// one word that leaves x18 alone; BR X18. The entries are read from the table;
// the target is one of the words up to the largest, rounded up to a power of
// two of at most 4096.
template <class WordAt>
std::optional<JumpTable> find_jump_table(std::uint64_t offset, std::uint64_t text_begin, const WordAt& word_at,
                                         std::span<const std::byte> image) {
    for (const std::uint64_t gap : {4u, 8u}) {
        if (offset < text_begin + gap + 8) continue;
        if (gap == 8 && classify_word(word_at(offset - 4)).uses_x18) continue;
        const auto add = word_at(offset - gap);
        if ((add & 0xffe0fc00u) != 0x8b000800u || ((add >> 5) & 31u) != 18 || (add & 31u) != 18) continue;
        const auto index = (add >> 16) & 31u;
        if (index == 18 || index == 31) continue;
        const auto load = word_at(offset - gap - 4);
        const bool byte_load = (load & 0xffe00c00u) == 0x38600800u || (load & 0xffc00000u) == 0x39400000u ||
                               (load & 0xffe00400u) == 0x38400400u;
        // LDRH (register), scaled: option LSL/UXTX (011) or UXTW (010), S = 1.
        const auto option = (load >> 13) & 7u;
        const bool halfword_load = (load & 0xffe00c00u) == 0x78600800u && ((load >> 12) & 1u) &&
                                   (option == 3u || option == 2u);
        if ((!byte_load && !halfword_load) || (load & 31u) != index) continue;
        const auto adr_offset = offset - gap - 8;
        const auto adr = word_at(adr_offset);
        if ((adr & 0x9f00001fu) != 0x10000012u) continue;
        const auto adr_immediate = sign_extend(((adr >> 29) & 3u) | (((adr >> 5) & 0x7ffffu) << 2), 21);
        const auto base = adr_offset + static_cast<std::uint64_t>(adr_immediate);
        if (byte_load) return JumpTable{base, jump_table_entries};

        const auto table_register = (load >> 5) & 31u, index_register = (load >> 16) & 31u;
        if (image.empty() || offset < text_begin + gap + 24 || table_register == 18 || table_register == 31 ||
            index_register == 18 || index_register == 31 || table_register == index_register)
            continue;
        const auto add_table = word_at(adr_offset - 4);
        const auto adrp_offset = adr_offset - 8;
        const auto adrp = word_at(adrp_offset);
        const auto branch = word_at(adr_offset - 12);
        const auto compare = word_at(adr_offset - 16);
        if ((add_table & 0xffc00000u) != 0x91000000u || (add_table & 31u) != table_register ||
            ((add_table >> 5) & 31u) != table_register)
            continue;
        if ((adrp & 0x9f00001fu) != (0x90000000u | table_register)) continue;
        const bool higher = (branch & 0xff00001fu) == 0x54000008u, higher_or_same = (branch & 0xff00001fu) == 0x54000002u;
        if (!higher && !higher_or_same) continue;
        if (((compare & 0x7fc0001fu) != 0x7100001fu) || ((compare >> 5) & 31u) != index_register) continue;
        const auto bound = (compare >> 10) & 0xfffu;
        const std::uint64_t count = higher ? std::uint64_t{bound} + 1 : bound;
        if (count == 0) continue;
        // Module images start on a page boundary, so ADRP's page is the
        // module-relative page of the instruction plus its displacement.
        const auto page_displacement = sign_extend(((adrp >> 29) & 3u) | (((adrp >> 5) & 0x7ffffu) << 2), 21);
        const auto table = (adrp_offset & ~std::uint64_t{0xfff}) + static_cast<std::uint64_t>(page_displacement << 12) +
                           ((add_table >> 10) & 0xfffu);
        if (table > image.size() || 2 * count > image.size() - table) continue;
        std::uint32_t largest = 0;
        for (std::uint64_t i = 0; i != count; ++i) {
            const auto low = std::to_integer<std::uint32_t>(image[table + 2 * i]);
            const auto high = std::to_integer<std::uint32_t>(image[table + 2 * i + 1]);
            largest = std::max(largest, low | (high << 8));
        }
        std::uint32_t entries = 1;
        while (entries <= largest) entries <<= 1;
        if (entries > 4096) continue;
        return JumpTable{base, entries};
    }
    return std::nullopt;
}

enum RoleBit : std::uint8_t {
    role_x18 = 1, role_system = 2, role_unknown_control = 4, role_exclusive_load = 8,
    role_exclusive_store = 16, role_clear_exclusive = 32,
};

std::uint8_t role_flags(const WordRole& role) noexcept {
    return static_cast<std::uint8_t>((role.uses_x18 ? role_x18 : 0) | (role.system ? role_system : 0) |
        (role.unknown_control ? role_unknown_control : 0) | (role.exclusive_load ? role_exclusive_load : 0) |
        (role.exclusive_store ? role_exclusive_store : 0) | (role.clear_exclusive ? role_clear_exclusive : 0));
}

WordRole role_from_flags(std::uint8_t flags) noexcept {
    WordRole role;
    role.uses_x18 = flags & role_x18;
    role.system = flags & role_system;
    role.unknown_control = flags & role_unknown_control;
    role.exclusive_load = flags & role_exclusive_load;
    role.exclusive_store = flags & role_exclusive_store;
    role.clear_exclusive = flags & role_clear_exclusive;
    return role;
}

} // namespace

std::size_t ModuleCodePreparation::scan_text(std::span<const std::byte> text, unsigned threads,
                                              std::span<const std::byte> image) {
    constexpr std::size_t serial_words = 1u << 16;
    const std::size_t words = text.size() / 4;
    if (words > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Module text is too large to scan.");
    const std::size_t parts = std::clamp<std::size_t>(words / serial_words, 1, std::max(threads, 1u));
    std::vector<std::vector<RoleWord>> found(parts);
    const auto scan_part = [&](std::size_t part) {
        const std::size_t begin = words * part / parts, end = words * (part + 1) / parts;
        auto& out = found[part];
        for (std::size_t i = begin; i < end; ++i) {
            std::uint32_t word;
            std::memcpy(&word, text.data() + i * 4, 4);
            const auto flags = role_flags(classify_word(word));
            if (flags) out.push_back({static_cast<std::uint32_t>(i), word, flags});
        }
    };
    std::vector<std::thread> workers;
    std::exception_ptr failure;
    try {
        workers.reserve(parts - 1);
        for (std::size_t part = 1; part < parts; ++part) workers.emplace_back(scan_part, part);
    } catch (const std::system_error&) {
        // A thread that could not be created leaves its parts to this one.
    } catch (...) {
        failure = std::current_exception();
    }
    // A part whose thread did not start runs here.
    try {
        scan_part(0);
        for (std::size_t part = workers.size() + 1; part < parts; ++part) scan_part(part);
    } catch (...) {
        if (!failure) failure = std::current_exception();
    }
    for (auto& worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
    role_words_.clear();
    if (embedded) {
        const auto prototype = execution::make_embedded_gate_code(0x8000, 0x4008);
        if (!prototype) throw std::runtime_error("Embedded gate template is invalid.");
        gate = *prototype;
    }
    std::size_t bytes = (gate.size() + 15) & ~std::size_t{15};
    const auto text_word = [&](std::uint64_t offset) -> std::uint32_t {
        std::uint32_t word = 0;
        if (offset >= text_begin && offset - text_begin + 4 <= text.size())
            std::memcpy(&word, text.data() + (offset - text_begin), 4);
        return word;
    };
    for (auto& part : found) {
        for (const auto& entry : part)
            if (entry.word == br_x18) {
                const auto table = find_jump_table(text_begin + 4 * std::uint64_t{entry.index}, text_begin, text_word, image);
                const auto entries = table ? std::max(table->entries, jump_table_entries) : jump_table_entries;
                bytes += 4 * (execution::indirect_x18_table_words(entries) + (embedded ? 2 : 0));
            }
            else if (entry.flags & (role_x18 | role_system)) bytes += embedded ? 64 : 48;
        role_words_.insert(role_words_.end(), part.begin(), part.end());
    }
    scanned_text_size_ = text.size();
    scanned_ = true;
    return bytes;
}

void ModuleCodePreparation::prepare(const ModulePreparationView& view) {
    using namespace execution;
    completed = false;
    sites.clear();
    stops.clear();
    scalar_sites = 0;
    thread_register_sites = 0;
    if (text_begin > view.image.size() || text_size > view.image.size() - text_begin || text_size % 4)
        throw std::runtime_error("Module text is outside its linked image.");
    const auto roots = module_roots(view, text_begin, text_size, module_header);
    prepared_entries = roots.local;
    required_foreign_entries = roots.required_foreign_entries;
    prepared_begin = view.base_address + text_begin;
    prepared_end = prepared_begin + text_size;
    std::uint64_t key = 0;
    if (embedded) {
        if (view.appended_data.size() < embedded_key_page_size ||
            view.appended_data_offset % embedded_key_page_size != 0 ||
            view.appended_data_offset < view.image.size() ||
            view.appended_data_offset > view.appended_code_offset ||
            view.appended_data.size() > view.appended_code_offset - view.appended_data_offset)
            throw std::runtime_error("Embedded preparation requires a separate writable key page.");
        key = view.base_address + view.appended_data_offset;
        const auto prepared_gate = make_embedded_gate_code(view.base_address + view.appended_code_offset,
                                                          key + embedded_gate_dispatcher_offset);
        if (!prepared_gate) throw std::runtime_error("Embedded gate cannot address its key page.");
        gate = *prepared_gate;
    }
    if (gate.empty() || gate.size() % 8 || gate.size() > view.appended_code.size())
        throw std::runtime_error("Local gate appendix is missing or too small.");
    std::memcpy(view.appended_code.data(), gate.data(), gate.size());
    std::size_t cursor = (gate.size() + 15) & ~std::size_t{15};
    const GateCodeRegion local_gate{view.base_address + view.appended_code_offset, gate.size()};
    const auto terminal = aarch64::encode_brk(0xae21).word;
    bool exclusive = false;
    std::size_t next_function = 0;
    // Original words of the last role words visited: earlier sites are
    // already rewritten in the image when a later word looks back at them.
    std::array<std::pair<std::uint64_t, std::uint32_t>, 8> recent{};
    std::size_t recent_next = 0;
    const auto original_at = [&](std::uint64_t offset) -> std::uint32_t {
        for (const auto& [at, word] : recent)
            if (at == offset && word) return word;
        std::uint32_t word;
        std::memcpy(&word, view.image.data() + offset, 4);
        return word;
    };
    // A switch through virtual x18 (find_jump_table): its base address and
    // entry count.
    const auto jump_table = [&](std::uint64_t offset) -> std::optional<JumpTable> {
        auto table = find_jump_table(offset, text_begin, original_at, view.image);
        if (table) table->base_offset += view.base_address;
        return table;
    };
    // Handles one text word with its role. Words with no role that matters
    // change nothing, so the scanned path skips them.
    const auto visit = [&](std::size_t i, std::uint32_t word, const WordRole& role) {
        const auto offset = text_begin + i * 4;
        const auto address = view.base_address + offset;
        if (role.exclusive_load) exclusive = true;
        const bool in_window = exclusive;
        if (role.exclusive_store || role.clear_exclusive) exclusive = false;
        if (!role.uses_x18 && !role.system && !role.unknown_control) return;
        recent[recent_next++ % recent.size()] = {offset, word};
        const auto stop = [&](const char* reason) {
            std::memcpy(view.image.data() + offset, &terminal, 4);
            stops.push_back({address, word, reason});
        };
        if (role.unknown_control) {
            stop("Control instruction unsupported by native preparation.");
            return;
        }
        if (in_window || !stack_contract.exclusive_monitor_clear || !stack_contract.stack_scratch_available ||
            !stack_contract.runtime_storage_disjoint) {
            stop("Host-sensitive instruction inside an exclusive-access window.");
            return;
        }
        const auto reserve = [&](std::size_t length) {
            if (cursor > view.appended_code.size() || length > view.appended_code.size() - cursor)
                throw std::runtime_error("Native preparation appendix is full.");
        };
        const auto trampoline = view.base_address + view.appended_code_offset + cursor;
        const auto instruction = aarch64::decode(word);
        if (embedded) {
            std::optional<EmbeddedPatch> plan;
            if (role.system) {
                plan = make_embedded_thread_register_patch(word, address, trampoline, key, stack_contract);
            } else if (word == br_x18) {
                const auto table = jump_table(offset);
                if (table) plan = make_embedded_indirect_x18_table_patch(
                    word, address, trampoline, key, table->base_offset, table->entries, stack_contract);
            } else if (instruction.opcode == Opcode::cbz || instruction.opcode == Opcode::cbnz ||
                       instruction.opcode == Opcode::tbz || instruction.opcode == Opcode::tbnz) {
                plan = make_embedded_conditional_x18_patch(word, address, trampoline, key, stack_contract);
            } else {
                plan = make_embedded_scalar_x18_patch(word, address, trampoline, key, stack_contract);
            }
            if (plan) {
                const auto bytes = plan->trampoline.size() * 4;
                reserve(bytes);
                std::memcpy(view.image.data() + offset, &plan->replacement_instruction, 4);
                std::memcpy(view.appended_code.data() + cursor, plan->trampoline.data(), bytes);
                cursor += bytes;
                if (role.system) ++thread_register_sites;
                else ++scalar_sites;
                return;
            }
            if (!role.system) {
                stop("Unsupported embedded virtual-x18 instruction.");
                return;
            }
        }
        if (role.system) {
            reserve(16);
            // Thread-pointer reads need no host: they load the cell word directly.
            if (const auto read = make_thread_register_read_patch(word, address, trampoline, slot);
                !embedded && read.ok()) {
                std::memcpy(view.image.data() + offset, &read.patch->replacement_instruction, 4);
                std::memcpy(view.appended_code.data() + cursor, read.patch->trampoline.data(), 16);
                cursor += 16;
                ++thread_register_sites;
                return;
            }
            const auto plan = make_gate_site_patch(word, address, trampoline, local_gate);
            if (!plan.ok()) { stop("Unsupported system-register or exception encoding."); return; }
            std::memcpy(view.image.data() + offset, &plan.patch->replacement_instruction, 4);
            std::memcpy(view.appended_code.data() + cursor, plan.patch->trampoline.data(), 16);
            sites.push_back(plan.patch->dispatch_site);
            cursor += 16;
        } else if (word == br_x18) {
            const auto table = jump_table(offset);
            if (!table) { stop("Indirect branch on virtual x18 without a recognised table."); return; }
            const auto words = indirect_x18_table_words(table->entries);
            reserve(words * 4);
            const auto plan = make_indirect_x18_table_patch(word, address, trampoline, slot, table->base_offset,
                                                            table->entries, stack_contract);
            if (!plan.ok()) { stop("Unsupported table branch on virtual x18."); return; }
            std::memcpy(view.image.data() + offset, &plan.patch->replacement_instruction, 4);
            std::memcpy(view.appended_code.data() + cursor, plan.patch->trampoline.data(), words * 4);
            cursor += words * 4;
            ++scalar_sites;
        } else if (instruction.opcode == Opcode::cbz || instruction.opcode == Opcode::cbnz ||
                   instruction.opcode == Opcode::tbz || instruction.opcode == Opcode::tbnz) {
            reserve(36);
            const auto plan = make_conditional_x18_patch(word, address, trampoline, slot, stack_contract);
            if (!plan.ok()) { stop("Unsupported conditional virtual-x18 instruction."); return; }
            std::memcpy(view.image.data() + offset, &plan.patch->replacement_instruction, 4);
            std::memcpy(view.appended_code.data() + cursor, plan.patch->trampoline.data(), 36);
            cursor += 36;
            ++scalar_sites;
        } else {
            reserve(48);
            const auto plan = make_scalar_x18_patch(word, address, trampoline, slot, stack_contract);
            if (!plan.ok()) { stop("Unsupported scalar virtual-x18 instruction."); return; }
            std::memcpy(view.image.data() + offset, &plan.patch->replacement_instruction, 4);
            const std::size_t bytes = plan.patch->trampoline_word_count * 4;
            std::memcpy(view.appended_code.data() + cursor, plan.patch->trampoline.data(), bytes);
            cursor += bytes;
            ++scalar_sites;
        }
    };
    // A function start ends any exclusive window left open before it. Every
    // root is a text word, so a root at or before a visited word has been passed.
    const auto pass_functions = [&](std::uint64_t address) {
        while (next_function < roots.local.size() && roots.local[next_function] <= address) {
            exclusive = false;
            ++next_function;
        }
    };
    // Launch-time words, in text order between the role words: each becomes a
    // gate site whose table entry the launch supplies.
    std::vector<std::uint64_t> launch(launch_words.begin(), launch_words.end());
    std::sort(launch.begin(), launch.end());
    if (!launch.empty() && !embedded) throw std::runtime_error("Launch-time words require embedded preparation.");
    if (std::adjacent_find(launch.begin(), launch.end()) != launch.end())
        throw std::runtime_error("A launch-time word is listed twice.");
    std::size_t next_launch = 0;
    const auto launch_before = [&](std::uint64_t end) {
        for (; next_launch < launch.size() && launch[next_launch] < end; ++next_launch) {
            const auto offset = launch[next_launch];
            if (offset < text_begin || offset - text_begin >= text_size || offset % 4)
                throw std::runtime_error("A launch-time word is outside module text.");
            const auto address = view.base_address + offset;
            pass_functions(address);
            if (exclusive || !stack_contract.exclusive_monitor_clear || !stack_contract.stack_scratch_available)
                throw std::runtime_error("A launch-time word is inside an exclusive-access window.");
            if (cursor > view.appended_code.size() || 16 > view.appended_code.size() - cursor)
                throw std::runtime_error("Native preparation appendix is full.");
            std::uint32_t word;
            std::memcpy(&word, view.image.data() + offset, 4);
            const auto trampoline = view.base_address + view.appended_code_offset + cursor;
            // MOVZ or MOVK, not of x18 (a virtual-register site of its own).
            if ((word & 0x1f800000u) != 0x12800000u || (word >> 29 & 3) < 2 || (word & 31) == 18)
                throw std::runtime_error("A launch-time word is not a MOVZ/MOVK the gate can serve.");
            const auto plan = make_gate_site_patch(word, address, trampoline, local_gate, {.move_wide = true});
            if (!plan.ok()) throw std::runtime_error("A launch-time word cannot become a gate site.");
            std::memcpy(view.image.data() + offset, &plan.patch->replacement_instruction, 4);
            std::memcpy(view.appended_code.data() + cursor, plan.patch->trampoline.data(), 16);
            sites.push_back(plan.patch->dispatch_site);
            cursor += 16;
        }
    };
    if (scanned_ && scanned_text_size_ == text_size) {
        for (const auto& entry : role_words_) {
            const auto offset = text_begin + std::uint64_t{entry.index} * 4;
            std::uint32_t word;
            std::memcpy(&word, view.image.data() + offset, 4);
            if (word != entry.word) throw std::runtime_error("Module text changed after it was scanned.");
            launch_before(offset);
            pass_functions(view.base_address + offset);
            visit(entry.index, word, role_from_flags(entry.flags));
        }
    } else {
        for (std::size_t i = 0; i < text_size / 4; ++i) {
            const auto offset = text_begin + i * 4;
            launch_before(offset);
            pass_functions(view.base_address + offset);
            std::uint32_t word;
            std::memcpy(&word, view.image.data() + offset, 4);
            visit(i, word, classify_word(word));
        }
    }
    launch_before(std::numeric_limits<std::uint64_t>::max());
    code_bytes = cursor;
    completed = true;
}

bool foreign_entries_prepared(std::span<const ModuleCodePreparation> modules) noexcept {
    for (const auto& module : modules) {
        if (!module.completed) return false;
        for (const auto target : module.required_foreign_entries) {
            const bool owned = std::any_of(modules.begin(), modules.end(), [&](const auto& owner) {
                return owner.completed && target >= owner.prepared_begin && target < owner.prepared_end;
            });
            if (!owned) return false;
        }
    }
    return true;
}

} // namespace armigo::runtime::native
