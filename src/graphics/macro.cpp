// ============================================================================
// Armigo Graphics - Maxwell Macro Interpreter
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/macro.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace armigo::graphics {
namespace {

constexpr MacroResult success(std::uint32_t pc = 0) noexcept
{
    return {MacroResultCode::Success, {}, pc};
}

constexpr MacroResult
failure(MacroResultCode code, std::string_view detail, std::uint32_t pc) noexcept
{
    return {code, detail, pc};
}

constexpr std::uint32_t field(std::uint32_t word, unsigned first, std::uint32_t mask) noexcept
{
    return (word >> first) & mask;
}

constexpr std::int32_t signed_immediate(std::uint32_t word) noexcept
{
    constexpr unsigned bits = 18;
    constexpr std::uint32_t mask = (1U << bits) - 1;
    constexpr std::uint32_t sign = 1U << (bits - 1);
    const std::uint32_t value = (word >> 14) & mask;
    if ((value & sign) != 0) {
        return static_cast<std::int32_t>(value) - static_cast<std::int32_t>(1U << bits);
    }
    return static_cast<std::int32_t>(value);
}

constexpr std::uint32_t bit_mask(std::uint32_t width) noexcept
{
    return width == 0 ? 0 : (std::uint32_t{1} << width) - 1;
}

class Interpreter {
  public:
    Interpreter(std::span<const std::uint32_t> code,
                std::span<const std::uint32_t> arguments,
                MacroCallbacks callbacks,
                std::uint32_t budget) noexcept
        : code_(code), arguments_(arguments), callbacks_(callbacks), remaining_budget_(budget)
    {
    }

    MacroResult run(std::uint32_t entry_word) noexcept
    {
        if (entry_word >= code_.size()) {
            return failure(
                MacroResultCode::InvalidEntry, "macro entry is outside uploaded code", entry_word);
        }
        if (arguments_.empty()) {
            return failure(MacroResultCode::MissingArgument,
                           "macro invocation has no first argument",
                           entry_word);
        }

        registers_[1] = arguments_[0];
        next_argument_ = 1;
        pc_ = entry_word;

        while (true) {
            const StepResult step_result = step(false);
            if (!step_result.result) {
                return step_result.result;
            }
            if (step_result.exited) {
                if (next_argument_ != arguments_.size()) {
                    return failure(MacroResultCode::SurplusArguments,
                                   "macro invocation left arguments unconsumed",
                                   current_pc_);
                }
                return success(current_pc_);
            }
        }
    }

  private:
    struct StepResult {
        MacroResult result{};
        bool exited = false;
    };

    StepResult step(bool delay_slot) noexcept
    {
        if (remaining_budget_ == 0) {
            return {failure(MacroResultCode::InstructionBudgetExceeded,
                            "macro instruction budget was exhausted",
                            pc_),
                    false};
        }
        if (pc_ >= code_.size()) {
            return {failure(MacroResultCode::ProgramCounterOutOfRange,
                            "macro program counter is outside uploaded code",
                            pc_),
                    false};
        }

        --remaining_budget_;
        current_pc_ = pc_;
        const std::uint32_t word = code_[pc_++];
        const std::uint32_t operation = field(word, 0, 0x7);

        if (operation == 7) {
            if (delay_slot) {
                return {failure(MacroResultCode::BranchInDelaySlot,
                                "macro branch in a delay slot is unsupported",
                                current_pc_),
                        false};
            }
            const bool branch_on_nonzero = field(word, 4, 0x1) != 0;
            const bool condition = read_register(field(word, 11, 0x7)) != 0;
            if (condition == branch_on_nonzero) {
                const std::int64_t target =
                    static_cast<std::int64_t>(current_pc_) + signed_immediate(word);
                if (target < 0 || static_cast<std::uint64_t>(target) >= code_.size()) {
                    return {failure(MacroResultCode::InvalidBranchTarget,
                                    "macro branch target is outside uploaded code",
                                    current_pc_),
                            false};
                }
                if (field(word, 5, 0x1) != 0) {
                    pc_ = static_cast<std::uint32_t>(target);
                    return {success(current_pc_), false};
                }

                const StepResult delayed = step(true);
                if (!delayed.result) {
                    return delayed;
                }
                pc_ = static_cast<std::uint32_t>(target);
                return {success(current_pc_), false};
            }
        } else {
            const MacroResult operation_result = execute_value_operation(word, operation);
            if (!operation_result) {
                return {operation_result, false};
            }
        }

        if (field(word, 7, 0x1) != 0 && !delay_slot) {
            const StepResult delayed = step(true);
            if (!delayed.result) {
                return delayed;
            }
            return {success(current_pc_), true};
        }
        return {success(current_pc_), false};
    }

    MacroResult execute_value_operation(std::uint32_t word, std::uint32_t operation) noexcept
    {
        std::uint32_t result = 0;
        const std::uint32_t source_a = read_register(field(word, 11, 0x7));
        const std::uint32_t source_b = read_register(field(word, 14, 0x7));

        switch (operation) {
        case 0: {
            const MacroResult alu_result =
                execute_alu(field(word, 17, 0x1f), source_a, source_b, result);
            if (!alu_result) {
                return alu_result;
            }
            break;
        }
        case 1:
            result = source_a + static_cast<std::uint32_t>(signed_immediate(word));
            break;
        case 2: {
            const std::uint32_t source_bit = field(word, 17, 0x1f);
            const std::uint32_t width = field(word, 22, 0x1f);
            const std::uint32_t destination_bit = field(word, 27, 0x1f);
            const std::uint32_t mask = bit_mask(width);
            const std::uint32_t inserted = (source_b >> source_bit) & mask;
            result = (source_a & ~(mask << destination_bit)) | (inserted << destination_bit);
            break;
        }
        case 3: {
            if (source_a >= 32) {
                return failure(MacroResultCode::UnsupportedVariableShift,
                               "macro variable shift is at least 32",
                               current_pc_);
            }
            const std::uint32_t width = field(word, 22, 0x1f);
            const std::uint32_t destination_bit = field(word, 27, 0x1f);
            result = ((source_b >> source_a) & bit_mask(width)) << destination_bit;
            break;
        }
        case 4: {
            if (source_a >= 32) {
                return failure(MacroResultCode::UnsupportedVariableShift,
                               "macro variable shift is at least 32",
                               current_pc_);
            }
            const std::uint32_t source_bit = field(word, 17, 0x1f);
            const std::uint32_t width = field(word, 22, 0x1f);
            result = ((source_b >> source_bit) & bit_mask(width)) << source_a;
            break;
        }
        case 5: {
            if (callbacks_.read_register == nullptr) {
                return failure(MacroResultCode::ReadCallbackUnavailable,
                               "macro register reader is unavailable",
                               current_pc_);
            }
            const std::uint32_t method =
                source_a + static_cast<std::uint32_t>(signed_immediate(word));
            MacroResult callback = callbacks_.read_register(callbacks_.context, method, result);
            if (!callback) {
                callback.pc = current_pc_;
                return callback;
            }
            break;
        }
        case 6:
            return failure(
                MacroResultCode::InvalidOperation, "macro operation 6 is invalid", current_pc_);
        default:
            return failure(
                MacroResultCode::InvalidOperation, "macro operation is invalid", current_pc_);
        }

        return assign_result(word, result);
    }

    MacroResult execute_alu(std::uint32_t operation,
                            std::uint32_t left,
                            std::uint32_t right,
                            std::uint32_t& result) noexcept
    {
        switch (operation) {
        case 0: {
            const std::uint64_t wide = static_cast<std::uint64_t>(left) + right;
            result = static_cast<std::uint32_t>(wide);
            carry_ = wide > std::numeric_limits<std::uint32_t>::max();
            return success(current_pc_);
        }
        case 1: {
            const std::uint64_t wide =
                static_cast<std::uint64_t>(left) + right + (carry_ ? 1U : 0U);
            result = static_cast<std::uint32_t>(wide);
            carry_ = wide > std::numeric_limits<std::uint32_t>::max();
            return success(current_pc_);
        }
        case 2:
            result = left - right;
            carry_ = left >= right;
            return success(current_pc_);
        case 3: {
            const std::uint64_t subtrahend = static_cast<std::uint64_t>(right) + (carry_ ? 0U : 1U);
            result = left - static_cast<std::uint32_t>(subtrahend);
            carry_ = static_cast<std::uint64_t>(left) >= subtrahend;
            return success(current_pc_);
        }
        case 8:
            result = left ^ right;
            return success(current_pc_);
        case 9:
            result = left | right;
            return success(current_pc_);
        case 10:
            result = left & right;
            return success(current_pc_);
        case 11:
            result = left & ~right;
            return success(current_pc_);
        case 12:
            result = ~(left & right);
            return success(current_pc_);
        default:
            return failure(MacroResultCode::InvalidAluOperation,
                           "macro ALU operation is invalid",
                           current_pc_);
        }
    }

    MacroResult assign_result(std::uint32_t word, std::uint32_t result) noexcept
    {
        const std::uint32_t assignment = field(word, 4, 0x7);
        const std::uint32_t destination = field(word, 8, 0x7);
        switch (assignment) {
        case 0: {
            std::uint32_t argument = 0;
            if (const MacroResult fetched = fetch_argument(argument); !fetched) {
                return fetched;
            }
            write_register(destination, argument);
            return success(current_pc_);
        }
        case 1:
            write_register(destination, result);
            return success(current_pc_);
        case 2:
            write_register(destination, result);
            set_method(result);
            return success(current_pc_);
        case 3: {
            std::uint32_t argument = 0;
            if (const MacroResult fetched = fetch_argument(argument); !fetched) {
                return fetched;
            }
            write_register(destination, argument);
            return emit(result);
        }
        case 4:
            write_register(destination, result);
            return emit(result);
        case 5: {
            std::uint32_t argument = 0;
            if (const MacroResult fetched = fetch_argument(argument); !fetched) {
                return fetched;
            }
            write_register(destination, argument);
            set_method(result);
            return success(current_pc_);
        }
        case 6: {
            write_register(destination, result);
            set_method(result);
            std::uint32_t argument = 0;
            if (const MacroResult fetched = fetch_argument(argument); !fetched) {
                return fetched;
            }
            return emit(argument);
        }
        case 7:
            write_register(destination, result);
            set_method(result);
            return emit((result >> 12) & 0x3f);
        default:
            return failure(MacroResultCode::InvalidOperation,
                           "macro assignment operation is invalid",
                           current_pc_);
        }
    }

    MacroResult fetch_argument(std::uint32_t& value) noexcept
    {
        if (next_argument_ >= arguments_.size()) {
            return failure(MacroResultCode::MissingArgument,
                           "macro attempted to fetch a missing argument",
                           current_pc_);
        }
        value = arguments_[next_argument_++];
        return success(current_pc_);
    }

    MacroResult emit(std::uint32_t value) noexcept
    {
        if (callbacks_.emit_method == nullptr) {
            return failure(MacroResultCode::EmitCallbackUnavailable,
                           "macro method sink is unavailable",
                           current_pc_);
        }
        if (method_increment_ > 0xfffU - method_address_) {
            return failure(MacroResultCode::UnsupportedMethodAddressOverflow,
                           "macro method address increment would overflow 12 bits",
                           current_pc_);
        }
        MacroResult callback = callbacks_.emit_method(callbacks_.context, method_address_, value);
        if (!callback) {
            callback.pc = current_pc_;
            return callback;
        }
        method_address_ += method_increment_;
        return success(current_pc_);
    }

    void set_method(std::uint32_t packed) noexcept
    {
        method_address_ = packed & 0xfff;
        method_increment_ = (packed >> 12) & 0x3f;
    }

    [[nodiscard]] std::uint32_t read_register(std::uint32_t index) const noexcept
    {
        return index == 0 ? 0 : registers_[index];
    }

    void write_register(std::uint32_t index, std::uint32_t value) noexcept
    {
        if (index != 0) {
            registers_[index] = value;
        }
    }

    std::span<const std::uint32_t> code_;
    std::span<const std::uint32_t> arguments_;
    MacroCallbacks callbacks_{};
    std::array<std::uint32_t, 8> registers_{};
    std::size_t next_argument_ = 0;
    std::uint32_t pc_ = 0;
    std::uint32_t current_pc_ = 0;
    std::uint32_t method_address_ = 0;
    std::uint32_t method_increment_ = 0;
    std::uint32_t remaining_budget_ = 0;
    bool carry_ = false;
};

}  // namespace

MacroResult execute_macro(std::span<const std::uint32_t> code,
                          std::uint32_t entry_word,
                          std::span<const std::uint32_t> arguments,
                          MacroCallbacks callbacks,
                          std::uint32_t instruction_budget) noexcept
{
    return Interpreter(code, arguments, callbacks, instruction_budget).run(entry_word);
}

}  // namespace armigo::graphics
