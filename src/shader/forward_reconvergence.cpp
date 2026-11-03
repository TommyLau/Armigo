// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "forward_reconvergence.h"

#include "maxwell_decode_helpers.h"

namespace armigo::shader::maxwell::detail {

StackId ReconvergenceStacks::push(StackId parent, TokenKind kind,
                                 std::uint32_t target) {
    const auto key = std::tuple{parent, kind, target};
    if (const auto found = interned_.find(key); found != interned_.end()) {
        return found->second;
    }
    const StackId id = tokens_.size() + 1;
    tokens_.push_back(Token{parent, kind, target});
    interned_.emplace(key, id);
    return id;
}

std::optional<ReconvergenceStacks::Pop> ReconvergenceStacks::pop(
    StackId stack, TokenKind kind) const {
    while (stack != 0) {
        const Token& token = tokens_[stack - 1];
        if (token.kind == kind) return Pop{token.parent, token.target_pc};
        stack = token.parent;
    }
    return std::nullopt;
}

ReconvergenceOpcode reconvergence_opcode(std::uint64_t word) {
    if (high_matches(word, 0xfff0, 0xe290)) return ReconvergenceOpcode::Ssy;
    if (high_matches(word, 0xfff8, 0xf0f8)) return ReconvergenceOpcode::Sync;
    if (high_matches(word, 0xfff0, 0xe2a0) ||
        high_matches(word, 0xfff0, 0xe340)) {
        return ReconvergenceOpcode::UnsupportedBreak;
    }
    return ReconvergenceOpcode::None;
}

DecodeError reconvergence_target_error(const ForwardBranchError& error,
                                       OpcodeFamily family) {
    return failure(DecodeError::Code::UnsupportedControlFlow, error.pc, family,
                   DecodeField::ReconvergenceTarget, error.detail,
                   static_cast<std::uint64_t>(error.code));
}

std::variant<Instruction, DecodeError> decode_ssy(
    std::uint64_t word, std::uint32_t pc, std::uint32_t program_size) {
    constexpr auto family = OpcodeFamily::ReconvergenceSetup;
    if (field(word, 5, 1) != 0) {
        return failure(DecodeError::Code::UnsupportedControlFlow, pc, family,
                       DecodeField::AddressMode, 1, 0);
    }
    // SSY is processed before generic predication. Admit
    // only zero non-address operand bits, rather than assigning guard or
    // absolute-address semantics to other encodings.
    constexpr auto allowed = high_mask(0xfff0) | bit_mask(24, 20);
    if ((word & ~allowed) != 0) {
        return failure(DecodeError::Code::UnsupportedControlFlow, pc, family,
                       DecodeField::ReservedBits, word & ~allowed, 0);
    }
    const auto raw_displacement = static_cast<std::int64_t>(field(word, 20, 24));
    const auto displacement = raw_displacement >= 0x800000
                                  ? raw_displacement - 0x1000000
                                  : raw_displacement;
    auto target = resolve_forward_target(static_cast<std::int64_t>(pc) + 8 + displacement,
                                         pc, program_size, Predicate{7, false});
    if (const auto* error = std::get_if<ForwardBranchError>(&target)) {
        return reconvergence_target_error(*error, family);
    }
    return Instruction{pc, word, family, Predicate{7, false},
                       ReconvergenceSetup{std::get<ForwardBranch>(target).target_pc}};
}

std::optional<DecodeError> validate_sync(std::uint64_t word, std::uint32_t pc) {
    constexpr auto family = OpcodeFamily::ReconvergenceSync;
    if (field(word, 0, 5) != 15) {
        return failure(DecodeError::Code::UnsupportedControlFlow, pc, family,
                       DecodeField::FlowTest, field(word, 0, 5), 15);
    }
    if (field(word, 5, 1) != 0) {
        return failure(DecodeError::Code::UnsupportedControlFlow, pc, family,
                       DecodeField::AddressMode, 1, 0);
    }
    constexpr auto allowed = high_mask(0xfff8) | bit_mask(4, 16) | bit_mask(5, 0);
    if ((word & ~allowed) != 0) {
        return failure(DecodeError::Code::UnsupportedControlFlow, pc, family,
                       DecodeField::ReservedBits, word & ~allowed, 0);
    }
    return std::nullopt;
}

}  // namespace armigo::shader::maxwell::detail
