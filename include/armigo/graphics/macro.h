// ============================================================================
// Armigo Graphics - Maxwell Macro Interpreter
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace armigo::graphics {

enum class MacroResultCode : std::uint8_t {
    Success,
    InvalidEntry,
    MissingArgument,
    SurplusArguments,
    ProgramCounterOutOfRange,
    InvalidOperation,
    InvalidAluOperation,
    InvalidBranchTarget,
    BranchInDelaySlot,
    UnsupportedVariableShift,
    UnsupportedMethodAddressOverflow,
    InstructionBudgetExceeded,
    ReadCallbackUnavailable,
    EmitCallbackUnavailable,
    ReadFailed,
    EmitFailed,
    CallbackStopped,
};

struct MacroResult {
    MacroResultCode code = MacroResultCode::Success;
    std::string_view detail{};
    // Word index in the supplied code span. Callback results are stamped with
    // the instruction that invoked the callback.
    std::uint32_t pc = 0;

    [[nodiscard]] constexpr bool ok() const noexcept { return code == MacroResultCode::Success; }

    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

using MacroReadRegister = MacroResult (*)(void*, std::uint32_t, std::uint32_t&) noexcept;
using MacroEmitMethod = MacroResult (*)(void*, std::uint32_t, std::uint32_t) noexcept;

struct MacroCallbacks {
    MacroReadRegister read_register = nullptr;
    MacroEmitMethod emit_method = nullptr;
    void* context = nullptr;
};

inline constexpr std::uint32_t default_macro_instruction_budget = 1U << 20;

// Executes one macro invocation synchronously. arguments[0] initializes r1;
// later arguments are consumed by fetch assignment operations. A successful
// invocation must consume every supplied argument. No interpreter state is
// retained between calls.
[[nodiscard]] MacroResult
execute_macro(std::span<const std::uint32_t> code,
              std::uint32_t entry_word,
              std::span<const std::uint32_t> arguments,
              MacroCallbacks callbacks,
              std::uint32_t instruction_budget = default_macro_instruction_budget) noexcept;

}  // namespace armigo::graphics
