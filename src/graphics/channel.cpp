// ============================================================================
// Armigo Graphics - Guest GPU Command Channel
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/channel.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace armigo::graphics {
namespace {

constexpr std::uint64_t entry_conditional_bit = std::uint64_t{1} << 0;
constexpr std::uint64_t entry_reserved_bit = std::uint64_t{1} << 1;
constexpr std::uint64_t entry_privileged_bit = std::uint64_t{1} << 40;

constexpr std::uint32_t header_method_mask = 0xfff;
constexpr std::uint32_t header_method_extension = 1u << 12;
constexpr std::uint32_t header_subchannel_shift = 13;
constexpr std::uint32_t header_count_shift = 16;
constexpr std::uint32_t header_count_mask = 0x1fff;
constexpr std::uint32_t header_mode_shift = 29;
constexpr std::uint32_t puller_method_limit = 0x40;
constexpr std::uint32_t bind_selector_mask = 0x1f0000;
constexpr std::uint32_t bind_class_mask = 0xffff;
constexpr std::uint32_t bind_invalid_mask =
    ~(bind_selector_mask | bind_class_mask);

enum class SubmissionMode : std::uint32_t {
    LegacyIncreasing = 0,
    Increasing = 1,
    LegacyNonIncreasing = 2,
    NonIncreasing = 3,
    Inline = 4,
    IncrementOnce = 5,
    Reserved = 6,
    EndSegment = 7,
};

constexpr bool supported_engine(std::uint32_t engine) noexcept {
    switch (engine) {
        case engine_class::twod:
        case engine_class::threed:
        case engine_class::compute:
        case engine_class::inline_to_memory:
        case engine_class::dma_copy:
            return true;
        default:
            return false;
    }
}

constexpr CommandResult result(CommandStatus status, std::string_view detail,
                               std::uint64_t address = 0,
                               std::size_t entry_index = 0,
                               std::uint32_t subchannel = 0,
                               std::uint32_t method = 0,
                               std::uint32_t argument = 0,
                               std::uint32_t engine = 0) noexcept {
    return {status, detail, address, entry_index, subchannel, method, argument,
            engine};
}

std::uint32_t read_word(std::span<const std::byte> bytes,
                        std::size_t word_index) noexcept {
    const std::size_t offset = word_index * 4;
    return std::uint32_t{std::to_integer<unsigned char>(bytes[offset])} |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[offset + 1])}
            << 8) |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[offset + 2])}
            << 16) |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[offset + 3])}
            << 24);
}

}  // namespace

CommandResult GpfifoChannel::halt(CommandResult command_result) noexcept {
    halted_ = true;
    halt_result_ = command_result;
    return halt_result_;
}

CommandResult GpfifoChannel::dispatch(std::uint32_t argument,
                                      std::uint64_t address,
                                      std::size_t entry_index,
                                      MethodSink sink,
                                      void* context) noexcept {
    const bool last = remaining_ == 1;
    std::uint32_t engine = 0;

    if (method_ == 0) {
        const std::uint32_t class_id = argument & bind_class_mask;
        if ((argument & bind_invalid_mask) != 0) {
            return halt(result(CommandStatus::Unsupported,
                               "GPU bind argument has unsupported high bits",
                               address, entry_index, subchannel_, method_,
                               argument, class_id));
        }
        if (!supported_engine(class_id)) {
            return halt(result(CommandStatus::Unsupported,
                               "GPU bind names an unsupported engine class",
                               address, entry_index, subchannel_, method_,
                               argument, class_id));
        }
        bindings_[subchannel_] = class_id;
    } else {
        if (method_ >= puller_method_limit) {
            engine = bindings_[subchannel_];
            if (engine == 0) {
                return halt(result(CommandStatus::UnboundEngine,
                                   "GPU method targets an unbound subchannel",
                                   address, entry_index, subchannel_, method_,
                                   argument));
            }
        }
        if (sink == nullptr) {
            return halt(result(CommandStatus::Unsupported,
                               "GPU method sink is unavailable", address,
                               entry_index, subchannel_, method_, argument,
                               engine));
        }

        const MethodCall call{method_, subchannel_, argument, address, last};
        CommandResult sink_result = sink(context, call, engine);
        if (!sink_result.ok()) {
            sink_result.address = address;
            sink_result.entry_index = entry_index;
            sink_result.subchannel = subchannel_;
            sink_result.method = method_;
            sink_result.argument = argument;
            sink_result.engine = engine;
            if (sink_result.detail.empty()) {
                sink_result.detail = "GPU method sink stopped command processing";
            }
            return halt(sink_result);
        }
    }

    --remaining_;
    if (!non_incrementing_ && remaining_ != 0) {
        ++method_;
    }
    if (increment_once_) {
        non_incrementing_ = true;
        increment_once_ = false;
    }
    return {};
}

CommandResult GpfifoChannel::process(
    std::span<const std::uint64_t> entries, const GpuAddressSpace& memory,
    MethodSink sink, void* context, std::uint64_t word_budget,
    std::optional<std::span<const std::byte>> captured_commands) {
    if (halted_) {
        return halt_result_;
    }
    if (entries.size() > max_submission_entries) {
        return halt(result(CommandStatus::BudgetExceeded,
                           "GPFIFO submission exceeds the entry budget"));
    }

    std::vector<std::uint64_t> owned_entries;
    try {
        owned_entries.assign(entries.begin(), entries.end());
    } catch (const std::bad_alloc&) {
        return halt(result(CommandStatus::BudgetExceeded,
                           "GPFIFO entry snapshot allocation failed"));
    }

    std::uint64_t consumed_words = 0;
    for (std::size_t entry_index = 0; entry_index < owned_entries.size();
         ++entry_index) {
        const std::uint64_t raw_entry = owned_entries[entry_index];
        if (raw_entry == 0) {
            continue;
        }

        const std::uint64_t address = entry_address(raw_entry);
        const std::uint32_t length = entry_words(raw_entry);
        if ((raw_entry & (entry_conditional_bit | entry_reserved_bit)) != 0) {
            return halt(result(CommandStatus::InvalidEntry,
                               "GPFIFO entry uses unsupported low control bits",
                               address, entry_index));
        }
        if ((raw_entry & entry_privileged_bit) != 0) {
            return halt(result(CommandStatus::InvalidEntry,
                               "GPFIFO entry requests privileged execution",
                               address, entry_index));
        }
        if (length == 0) {
            return halt(result(CommandStatus::InvalidEntry,
                               "Nonzero GPFIFO entry has zero length", address,
                               entry_index));
        }
        if (length > max_entry_words ||
            consumed_words > word_budget ||
            length > word_budget - consumed_words) {
            return halt(result(CommandStatus::BudgetExceeded,
                               "GPFIFO command words exceed the processing budget",
                               address, entry_index));
        }
        consumed_words += length;

        std::vector<std::byte> fetched;
        std::span<const std::byte> bytes;
        const auto byte_count = static_cast<std::size_t>(length) * 4;
        if (captured_commands) {
            if (captured_commands->size() < byte_count) {
                return halt(result(CommandStatus::MemoryFault,
                                   "GPFIFO captured command segment is truncated",
                                   address, entry_index));
            }
            bytes = captured_commands->first(byte_count);
            *captured_commands = captured_commands->subspan(byte_count);
        } else {
            try {
                fetched.resize(byte_count);
            } catch (const std::bad_alloc&) {
                return halt(result(CommandStatus::BudgetExceeded,
                                   "GPFIFO command snapshot allocation failed",
                                   address, entry_index));
            }
            const MemoryResult memory_result = memory.read(address, fetched);
            if (!memory_result) {
                return halt(result(CommandStatus::MemoryFault,
                                   memory_result.detail.empty()
                                       ? "GPFIFO command memory read failed"
                                       : memory_result.detail,
                                   address, entry_index));
            }
            bytes = fetched;
        }

        for (std::size_t word_index = 0; word_index < length; ++word_index) {
            const std::uint32_t word = read_word(bytes, word_index);
            const std::uint64_t word_address =
                address + static_cast<std::uint64_t>(word_index) * 4;
            if (remaining_ != 0) {
                if (const CommandResult call_result =
                        dispatch(word, word_address, entry_index, sink, context);
                    !call_result.ok()) {
                    return call_result;
                }
                continue;
            }

            // Zero count and mode 0 clear every bit from the count field up,
            // forming an empty legacy header. Its low method and subchannel
            // bits may be nonzero without naming a modern method, so the word
            // is consumed whole, with no dispatch and no channel state change.
            if ((word >> header_count_shift) == 0) {
                continue;
            }
            if ((word & header_method_extension) != 0) {
                return halt(result(CommandStatus::InvalidHeader,
                                   "GPFIFO header uses disputed method bit 12",
                                   word_address, entry_index));
            }

            const auto mode =
                static_cast<SubmissionMode>(word >> header_mode_shift);
            const std::uint32_t method = word & header_method_mask;
            const std::uint32_t subchannel =
                (word >> header_subchannel_shift) & 7;
            const std::uint32_t count =
                (word >> header_count_shift) & header_count_mask;

            if (mode == SubmissionMode::Inline) {
                method_ = method;
                subchannel_ = subchannel;
                remaining_ = 1;
                non_incrementing_ = true;
                increment_once_ = false;
                if (const CommandResult call_result =
                        dispatch(count, word_address, entry_index, sink, context);
                    !call_result.ok()) {
                    return call_result;
                }
                continue;
            }

            if (mode != SubmissionMode::Increasing &&
                mode != SubmissionMode::NonIncreasing &&
                mode != SubmissionMode::IncrementOnce) {
                // The decoded method and subchannel cannot identify an
                // unrecognized mode, so this failure reports the raw
                // header word in the argument field. The submitted header is
                // preserved without interpreting it as a method argument.
                return halt(result(CommandStatus::Unsupported,
                                   "GPFIFO header mode is unsupported",
                                   word_address, entry_index, subchannel, method,
                                   word));
            }
            if (count == 0) {
                return halt(result(CommandStatus::InvalidHeader,
                                   "GPFIFO packet has zero arguments",
                                   word_address, entry_index, subchannel, method));
            }
            if ((mode == SubmissionMode::Increasing &&
                 count - 1 > header_method_mask - method) ||
                (mode == SubmissionMode::IncrementOnce && count > 1 &&
                 method == header_method_mask)) {
                return halt(result(CommandStatus::InvalidHeader,
                                   "GPFIFO packet increments past method 0xfff",
                                   word_address, entry_index, subchannel, method,
                                   word));
            }

            method_ = method;
            subchannel_ = subchannel;
            remaining_ = count;
            non_incrementing_ = mode == SubmissionMode::NonIncreasing;
            increment_once_ = mode == SubmissionMode::IncrementOnce;
        }
    }

    return {};
}

}  // namespace armigo::graphics
