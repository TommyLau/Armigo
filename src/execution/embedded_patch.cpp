// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/embedded_patch.h>
#include <armigo/execution/conditional_x18_patch.h>
#include <armigo/execution/indirect_x18_patch.h>
#include <armigo/execution/thread_register_patch.h>

#include <cstring>
#include <limits>
#include <span>

extern "C" {
extern const std::byte armigo_execution_embedded_gate_begin[];
extern const std::byte armigo_execution_embedded_gate_load[];
extern const std::byte armigo_execution_embedded_gate_end[];
}

namespace armigo::execution {
namespace {
using namespace aarch64;
constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t nop = 0xd503201f;

bool disjoint(std::uint64_t first, std::uint64_t length,
              std::uint64_t second, std::uint64_t other_length) {
    return length != 0 && other_length != 0 && first <= maximum - (length - 1) &&
        second <= maximum - (other_length - 1) &&
        (first + length - 1 < second || second + other_length - 1 < first);
}

bool valid_layout(std::uint64_t address, std::uint64_t trampoline,
                  std::size_t words, std::uint64_t key) {
    return address % 4 == 0 && trampoline % 4 == 0 && key % 8 == 0 &&
        disjoint(address, 8, trampoline, words * 4) &&
        disjoint(address, 8, key, 8) && disjoint(trampoline, words * 4, key, 8);
}

bool key_load(std::uint32_t* out, unsigned reg, std::uint64_t pc, std::uint64_t key) {
    const auto page = encode_adrp(reg, pc, key);
    if (!page.ok()) return false;
    out[0] = page.word;
    out[1] = 0xf9400000u | (static_cast<std::uint32_t>((key & 4095) / 8) << 10) |
        (reg << 5) | reg;
    return true;
}

bool address_value(std::uint32_t* out, unsigned reg, std::uint64_t pc,
                   std::uint64_t target) {
    const auto page = encode_adrp(reg, pc, target);
    if (!page.ok()) return false;
    out[0] = page.word;
    out[1] = 0x91000000u | (static_cast<std::uint32_t>(target & 4095) << 10) |
        (reg << 5) | reg;
    out[2] = nop;
    out[3] = nop;
    return true;
}

// Expand only the known ordinary-builder prologue. Every PC-relative word is
// re-encoded; internal labels move with the inserted words, external labels
// retain their original destination. ADRP always names an external page.
std::optional<EmbeddedPatch> expand(std::uint32_t replacement,
    std::span<const std::uint32_t> source, std::uint64_t address,
    std::uint64_t trampoline, std::uint64_t key, unsigned value, unsigned cell) {
    if (source.size() < 4 || !valid_layout(address, trampoline, source.size() + 2, key) ||
        source[1] != encode_mrs(cell, system_register_encoding(3, 3, 13, 0, 3)).word ||
        source[2] != (0xf9400000u | (cell << 5) | cell)) return {};
    EmbeddedPatch result{replacement, {}};
    result.trampoline.reserve(source.size() + 2);
    const auto translated = [&](std::uint64_t target) {
        if (target >= trampoline + 12 && target < trampoline + source.size() * 4)
            return target + 8;
        return target;
    };
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (i == 2) {
            std::uint32_t load[2];
            if (!key_load(load, value, trampoline + 8, key)) return {};
            result.trampoline.insert(result.trampoline.end(), {load[0], load[1],
                0xf8606800u | (value << 16) | (cell << 5) | cell});
            continue;
        }
        const auto instruction = decode(source[i]);
        const auto old_pc = trampoline + i * 4;
        const auto pc = trampoline + result.trampoline.size() * 4;
        const auto target = referenced_address(instruction, old_pc);
        Encoded encoded{source[i]};
        if (target) {
            const auto moved = translated(*target);
            switch (instruction.opcode) {
            case Opcode::b: encoded = encode_b(pc, moved); break;
            case Opcode::bl: encoded = encode_bl(pc, moved); break;
            case Opcode::cbz: case Opcode::cbnz:
                encoded = encode_cbz(instruction.opcode == Opcode::cbnz,
                    instruction.wide, instruction.reg, pc, moved); break;
            case Opcode::tbz: case Opcode::tbnz:
                encoded = encode_tbz(instruction.opcode == Opcode::tbnz,
                    instruction.reg, instruction.test_bit, pc, moved); break;
            case Opcode::adr: encoded = encode_adr(instruction.reg, pc, moved); break;
            case Opcode::adrp: encoded = encode_adrp(instruction.reg, pc, *target); break;
            default: return {};
            }
        }
        if (!encoded.ok()) return {};
        result.trampoline.push_back(encoded.word);
    }
    return result;
}
} // namespace

std::optional<EmbeddedPatch> make_embedded_scalar_x18_patch(
    std::uint32_t word, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key, ScalarX18Policy policy) {
    const auto ordinary = make_scalar_x18_patch(word, address, trampoline, 0, policy);
    if (!ordinary.ok()) return {};
    const auto& patch = *ordinary.patch;
    auto result = expand(patch.replacement_instruction,
        std::span(patch.trampoline).first(patch.trampoline_word_count), address,
        trampoline, key, patch.value_register, patch.cell_register);
    if (!result) return {};
    if (decode(word).opcode == Opcode::adr) {
        const auto target = referenced_address(decode(word), address);
        if (!target || !address_value(result->trampoline.data() + 6,
                                     patch.value_register, trampoline + 24, *target)) return {};
    }
    return result;
}

std::optional<EmbeddedPatch> make_embedded_conditional_x18_patch(
    std::uint32_t word, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key, ScalarX18Policy policy) {
    const auto ordinary = make_conditional_x18_patch(word, address, trampoline, 0, policy);
    if (!ordinary.ok()) return {};
    const auto& patch = *ordinary.patch;
    auto result = expand(patch.replacement_instruction, patch.trampoline, address,
                         trampoline, key, patch.value_register, patch.cell_register);
    const auto target = referenced_address(decode(word), address);
    if (!result || !target || !disjoint(*target, 4, trampoline, result->trampoline.size() * 4)) return {};
    return result;
}

std::optional<EmbeddedPatch> make_embedded_indirect_x18_table_patch(
    std::uint32_t word, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key, std::uint64_t table, std::uint32_t entries, ScalarX18Policy policy) {
    const auto ordinary = make_indirect_x18_table_patch(word, address, trampoline, 0,
                                                       table, entries, policy);
    if (!ordinary.ok()) return {};
    auto result = expand(ordinary.patch->replacement_instruction, ordinary.patch->trampoline,
                         address, trampoline, key, 0, 1);
    if (!result || !disjoint(table, entries * 4ull, trampoline, result->trampoline.size() * 4) ||
        !disjoint(table, entries * 4ull, key, 8) ||
        !address_value(result->trampoline.data() + 6, 1, trampoline + 24, table)) return {};
    return result;
}

std::optional<EmbeddedPatch> make_embedded_thread_register_patch(
    std::uint32_t word, std::uint64_t address, std::uint64_t trampoline,
    std::uint64_t key, ScalarX18Policy policy) {
    if (!policy.exclusive_monitor_clear || !policy.stack_scratch_available ||
        !policy.runtime_storage_disjoint || !valid_layout(address, trampoline, 8, key)) return {};
    const auto ordinary = make_thread_register_read_patch(word, address, trampoline, 0);
    if (!ordinary.ok()) return {};
    const unsigned rt = ordinary.patch->destination_register;
    const unsigned scratch = rt == 0 ? 1 : 0;
    const auto resume = encode_b(trampoline + 28, address + 4);
    if (!resume.ok()) return {};
    EmbeddedPatch result{ordinary.patch->replacement_instruction, std::vector<std::uint32_t>(8)};
    auto& words = result.trampoline;
    words[0] = 0xf81f0fe0u | scratch; // STR scratch,[SP,#-16]!
    if (!key_load(words.data() + 1, rt, trampoline + 4, key)) return {};
    words[3] = encode_mrs(scratch, system_register_encoding(3, 3, 13, 0, 3)).word;
    words[4] = 0xf8606800u | (rt << 16) | (scratch << 5) | rt;
    words[5] = ordinary.patch->trampoline[2];
    words[6] = 0xf84107e0u | scratch; // LDR scratch,[SP],#16
    words[7] = resume.word;
    return result;
}

std::optional<std::vector<std::byte>> make_embedded_gate_code(
    std::uint64_t address, std::uint64_t key) {
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_execution_embedded_gate_begin);
    const auto load = reinterpret_cast<std::uintptr_t>(armigo_execution_embedded_gate_load);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_execution_embedded_gate_end);
    if (end <= begin || end - begin > 65536 || load < begin || load > end ||
        end - load < 8 || (load - begin) % 4 || address % 4 || key % 8 ||
        !disjoint(address, end - begin, key, 8)) return {};
    std::uint32_t words[2];
    if (!key_load(words, 16, address + (load - begin), key)) return {};
    std::vector<std::byte> code(end - begin);
    std::memcpy(code.data(), armigo_execution_embedded_gate_begin, code.size());
    std::memcpy(code.data() + (load - begin), words, sizeof(words));
    return code;
}

std::optional<EmbeddedPatch> make_embedded_host_branch(
    std::uint64_t address, std::uint64_t trampoline, std::uint64_t key) {
    if (!valid_layout(address, trampoline, 3, key)) return {};
    const auto enter = encode_b(address, trampoline);
    if (!enter.ok()) return {};
    EmbeddedPatch result{enter.word, std::vector<std::uint32_t>(3)};
    if (!key_load(result.trampoline.data(), 16, trampoline, key)) return {};
    result.trampoline[2] = 0xd61f0200u; // BR X16
    return result;
}

} // namespace armigo::execution
