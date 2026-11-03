// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/shader/maxwell_forward_program.h>

#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace armigo::shader::maxwell::detail {

enum class TokenKind { Ssy, Pbk };
using StackId = std::size_t;

// Interned persistent stacks share their tails. Identity compares the entire
// ordered stack, without copying a growing stack for every reachable slot.
class ReconvergenceStacks {
public:
    [[nodiscard]] StackId push(StackId parent, TokenKind kind, std::uint32_t target);
    struct Pop {
        StackId remaining;
        std::uint32_t target_pc;
    };
    [[nodiscard]] std::optional<Pop> pop(StackId stack, TokenKind kind) const;

private:
    struct Token {
        StackId parent;
        TokenKind kind;
        std::uint32_t target_pc;
    };
    std::vector<Token> tokens_;
    std::map<std::tuple<StackId, TokenKind, std::uint32_t>, StackId> interned_;
};

enum class ReconvergenceOpcode { None, Ssy, Sync, UnsupportedBreak };
[[nodiscard]] ReconvergenceOpcode reconvergence_opcode(std::uint64_t word);
[[nodiscard]] std::variant<Instruction, DecodeError> decode_ssy(
    std::uint64_t word, std::uint32_t pc, std::uint32_t program_size);
[[nodiscard]] std::optional<DecodeError> validate_sync(std::uint64_t word,
                                                     std::uint32_t pc);
[[nodiscard]] DecodeError reconvergence_target_error(
    const ForwardBranchError& error, OpcodeFamily family);

}  // namespace armigo::shader::maxwell::detail
