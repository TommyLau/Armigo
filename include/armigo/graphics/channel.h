// ============================================================================
// Armigo Graphics - Guest GPU Command Channel
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/memory.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace armigo::graphics {

enum class CommandStatus : std::uint8_t {
    Complete,
    InvalidEntry,
    InvalidHeader,
    MemoryFault,
    UnboundEngine,
    Unsupported,
    Stopped,
    BudgetExceeded,
};

struct CommandResult {
    CommandStatus status = CommandStatus::Complete;
    std::string_view detail{};
    std::uint64_t address = 0;
    std::size_t entry_index = 0;
    std::uint32_t subchannel = 0;
    std::uint32_t method = 0;
    std::uint32_t argument = 0;
    std::uint32_t engine = 0;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return status == CommandStatus::Complete;
    }
};

struct MethodCall {
    std::uint32_t method = 0;
    std::uint32_t subchannel = 0;
    std::uint32_t argument = 0;
    std::uint64_t address = 0;
    bool last = false;
};

// A non-complete result halts the channel. Its detail must remain valid for the
// channel lifetime because the sticky result is returned by later submissions.
using MethodSink = CommandResult (*)(void*, const MethodCall&,
                                     std::uint32_t engine) noexcept;

namespace engine_class {
inline constexpr std::uint32_t twod = 0x902d;
inline constexpr std::uint32_t threed = 0xb197;
inline constexpr std::uint32_t compute = 0xb1c0;
inline constexpr std::uint32_t inline_to_memory = 0xa140;
inline constexpr std::uint32_t dma_copy = 0xb0b5;
}  // namespace engine_class

class GpfifoChannel {
public:
    static constexpr std::uint64_t default_word_budget = 1u << 24;
    static constexpr std::uint32_t max_entry_words = 1u << 20;
    static constexpr std::size_t max_submission_entries = 1u << 20;

    [[nodiscard]] static constexpr std::uint64_t entry_address(std::uint64_t entry) noexcept {
        return entry & ((std::uint64_t{1} << 40) - 4);
    }
    [[nodiscard]] static constexpr std::uint32_t entry_words(std::uint64_t entry) noexcept {
        return static_cast<std::uint32_t>((entry >> 42) & ((1u << 21) - 1));
    }

    // Captured bytes concatenate the segments in descriptor order. When present,
    // decoding never reads command memory; engine data still uses its live view.
    [[nodiscard]] CommandResult process(
        std::span<const std::uint64_t> entries,
        const GpuAddressSpace& memory, MethodSink sink, void* context,
        std::uint64_t word_budget = default_word_budget,
        std::optional<std::span<const std::byte>> captured_commands = std::nullopt);

    [[nodiscard]] std::span<const std::uint32_t, 8> bindings() const noexcept {
        return bindings_;
    }

    [[nodiscard]] bool has_pending() const noexcept {
        return remaining_ != 0;
    }

private:
    [[nodiscard]] CommandResult dispatch(std::uint32_t argument,
                                         std::uint64_t address,
                                         std::size_t entry_index,
                                         MethodSink sink,
                                         void* context) noexcept;
    [[nodiscard]] CommandResult halt(CommandResult result) noexcept;

    std::array<std::uint32_t, 8> bindings_{};
    std::uint32_t method_ = 0;
    std::uint32_t subchannel_ = 0;
    std::uint32_t remaining_ = 0;
    bool non_incrementing_ = false;
    bool increment_once_ = false;
    bool halted_ = false;
    CommandResult halt_result_{};
};

}  // namespace armigo::graphics
