// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/channel.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace {

using armigo::graphics::AddressSpaceResult;
using armigo::graphics::CommandResult;
using armigo::graphics::CommandStatus;
using armigo::graphics::GpuAddressSpace;
using armigo::graphics::GpfifoChannel;
using armigo::graphics::Mapping;
using armigo::graphics::MemoryCallbacks;
using armigo::graphics::MethodCall;
using armigo::graphics::engine_class::dma_copy;
using armigo::graphics::engine_class::threed;
using armigo::graphics::make_address_space;

unsigned failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

constexpr std::uint64_t gpu_base = 0x1000;
constexpr std::uint64_t guest_base = 0x8000;

struct GuestMemory {
    std::array<std::byte, 0x2000> bytes{};
    std::vector<std::pair<std::uint64_t, std::size_t>> reads;
    std::uint64_t fail_address = 0;

    static bool read(void* raw, std::uint64_t address,
                     std::span<std::byte> destination) noexcept {
        auto& self = *static_cast<GuestMemory*>(raw);
        self.reads.emplace_back(address, destination.size());
        if (address == self.fail_address || address < guest_base) {
            return false;
        }
        const std::uint64_t offset = address - guest_base;
        if (offset > self.bytes.size() ||
            destination.size() > self.bytes.size() - offset) {
            return false;
        }
        std::memcpy(destination.data(), self.bytes.data() + offset,
                    destination.size());
        return true;
    }

    void set_words(std::uint64_t gpu_address,
                   std::span<const std::uint32_t> words) noexcept {
        const std::size_t offset = static_cast<std::size_t>(gpu_address - gpu_base);
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t word = words[index];
            for (unsigned byte = 0; byte < 4; ++byte) {
                bytes[offset + index * 4 + byte] =
                    static_cast<std::byte>((word >> (byte * 8)) & 0xff);
            }
        }
    }

    GpuAddressSpace address_space() {
        const Mapping mapping{gpu_base, bytes.size(), 0, guest_base,
                              bytes.size(), 0x1000};
        AddressSpaceResult result = make_address_space(
            {&mapping, 1}, MemoryCallbacks{read, nullptr, this});
        CHECK(std::holds_alternative<GpuAddressSpace>(result));
        return std::get<GpuAddressSpace>(std::move(result));
    }
};

constexpr std::uint32_t header(std::uint32_t mode, std::uint32_t method,
                               std::uint32_t subchannel,
                               std::uint32_t count_or_argument) noexcept {
    return (mode << 29) | (count_or_argument << 16) | (subchannel << 13) |
           method;
}

constexpr std::uint64_t entry(std::uint64_t address, std::uint32_t length,
                              bool level = false, bool sync = false) noexcept {
    return address | (std::uint64_t{level} << 41) |
           (std::uint64_t{length} << 42) | (std::uint64_t{sync} << 63);
}

struct RecordedCall {
    MethodCall call{};
    std::uint32_t engine = 0;
};

struct Recorder {
    std::vector<RecordedCall> calls;
    CommandStatus stop_status = CommandStatus::Complete;
    unsigned stop_at = std::numeric_limits<unsigned>::max();
    unsigned completed = 0;

    static CommandResult record(void* raw, const MethodCall& call,
                                std::uint32_t engine) noexcept {
        auto& self = *static_cast<Recorder*>(raw);
        self.calls.push_back({call, engine});
        if (self.calls.size() - 1 == self.stop_at) {
            return {self.stop_status, "Synthetic sink stop"};
        }
        ++self.completed;
        return {};
    }
};

void bind_words(std::vector<std::uint32_t>& words, std::uint32_t subchannel,
                std::uint32_t engine) {
    words.push_back(header(3, 0, subchannel, 1));
    words.push_back(engine);
}

void test_header_modes_and_golden_fields() {
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    std::vector<std::uint32_t> words;
    bind_words(words, 5, threed);
    words.insert(words.end(), {header(1, 0x100, 5, 3), 0x11, 0x12, 0x13,
                               header(3, 0x200, 5, 2), 0x21, 0x22,
                               header(5, 0x300, 5, 3), 0x31, 0x32, 0x33,
                               header(4, 0x20, 6, 0x456)});
    guest.set_words(0x1000, words);

    GpfifoChannel channel;
    Recorder recorder;
    const std::uint64_t descriptor = entry(0x1000, words.size(), true, true);
    CHECK(channel.process({&descriptor, 1}, memory, Recorder::record, &recorder)
              .ok());
    CHECK(recorder.calls.size() == 9);

    CHECK(recorder.calls[0].call.method == 0x100);
    CHECK(recorder.calls[1].call.method == 0x101);
    CHECK(recorder.calls[2].call.method == 0x102);
    CHECK(!recorder.calls[0].call.last && recorder.calls[2].call.last);
    CHECK(recorder.calls[0].call.subchannel == 5);
    CHECK(recorder.calls[0].engine == threed);
    CHECK(recorder.calls[0].call.address == 0x100c);

    CHECK(recorder.calls[3].call.method == 0x200);
    CHECK(recorder.calls[4].call.method == 0x200);
    CHECK(!recorder.calls[3].call.last && recorder.calls[4].call.last);
    CHECK(recorder.calls[5].call.method == 0x300);
    CHECK(recorder.calls[6].call.method == 0x301);
    CHECK(recorder.calls[7].call.method == 0x301);
    CHECK(recorder.calls[7].call.last);
    CHECK(recorder.calls[8].call.method == 0x20);
    CHECK(recorder.calls[8].call.subchannel == 6);
    CHECK(recorder.calls[8].call.argument == 0x456);
    CHECK(recorder.calls[8].engine == 0);
    CHECK(recorder.calls[8].call.last);
}

void test_continuation_across_entries_and_submissions() {
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array first_words{header(3, 0, 2, 1), threed,
                                 header(1, 0x120, 2, 3), 0xa1u};
    const std::array second_words{0xa2u, 0xa3u};
    guest.set_words(0x1000, first_words);
    guest.set_words(0x1100, second_words);

    GpfifoChannel channel;
    Recorder recorder;
    const std::uint64_t first = entry(0x1000, first_words.size());
    CHECK(channel.process({&first, 1}, memory, Recorder::record, &recorder).ok());
    CHECK(channel.has_pending());
    CHECK(recorder.calls.size() == 1);
    CHECK(!recorder.calls[0].call.last);

    const std::uint64_t second = entry(0x1100, second_words.size());
    CHECK(channel.process({&second, 1}, memory, Recorder::record, &recorder).ok());
    CHECK(!channel.has_pending());
    CHECK(recorder.calls.size() == 3);
    CHECK(recorder.calls[1].call.method == 0x121);
    CHECK(recorder.calls[2].call.method == 0x122);
    CHECK(recorder.calls[2].call.last);
}

void test_dynamic_binding_and_rebinding() {
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array words{header(3, 0, 7, 1), threed | 0x1f0000,
                           header(3, 0x100, 7, 1), 1u,
                           header(3, 0, 7, 1), dma_copy,
                           header(3, 0x100, 7, 1), 2u};
    guest.set_words(0x1000, words);

    GpfifoChannel channel;
    Recorder recorder;
    const std::uint64_t descriptor = entry(0x1000, words.size());
    CHECK(channel.process({&descriptor, 1}, memory, Recorder::record, &recorder)
              .ok());
    CHECK(recorder.calls.size() == 2);
    CHECK(recorder.calls[0].engine == threed);
    CHECK(recorder.calls[1].engine == dma_copy);
    CHECK(channel.bindings()[7] == dma_copy);

    const auto rejected_bind = [&](std::uint32_t argument) {
        GuestMemory local_guest;
        GpuAddressSpace local_memory = local_guest.address_space();
        const std::array local_words{header(3, 0, 1, 1), argument};
        local_guest.set_words(0x1000, local_words);
        GpfifoChannel local_channel;
        Recorder local_recorder;
        const std::uint64_t local_entry = entry(0x1000, local_words.size());
        return local_channel.process({&local_entry, 1}, local_memory,
                                     Recorder::record, &local_recorder);
    };
    CHECK(rejected_bind(0x200000 | threed).status ==
          CommandStatus::Unsupported);
    CHECK(rejected_bind(0x9999).status == CommandStatus::Unsupported);
}

void test_entry_controls_and_zero_forms() {
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array<std::uint32_t, 1> nop{0};
    guest.set_words(0x1000, nop);
    guest.set_words(0x1100, nop);
    const std::array accepted{std::uint64_t{0}, entry(0x1000, 1),
                              entry(0x1100, 1, true, true)};
    GpfifoChannel channel;
    Recorder recorder;
    CHECK(channel.process(accepted, memory, Recorder::record, &recorder).ok());
    CHECK(recorder.calls.empty());
    CHECK(guest.reads.size() == 2);

    const auto rejected = [&](std::uint64_t descriptor) {
        GpfifoChannel local_channel;
        return local_channel.process({&descriptor, 1}, memory,
                                     Recorder::record, &recorder);
    };
    CHECK(rejected(entry(0x1000, 1) | 1).status ==
          CommandStatus::InvalidEntry);
    CHECK(rejected(entry(0x1000, 1) | 2).status ==
          CommandStatus::InvalidEntry);
    CHECK(rejected(entry(0x1000, 1) | (std::uint64_t{1} << 40)).status ==
          CommandStatus::InvalidEntry);
    CHECK(rejected(std::uint64_t{1} << 41).status ==
          CommandStatus::InvalidEntry);
}

struct MutationContext {
    Recorder recorder;
    GuestMemory* guest = nullptr;
    std::uint64_t* entries = nullptr;
    std::uint64_t replacement_entry = 0;
    std::uint64_t mutation_address = 0;
    std::uint32_t mutation_word = 0;

    static CommandResult mutate(void* raw, const MethodCall& call,
                                std::uint32_t engine) noexcept {
        auto& self = *static_cast<MutationContext*>(raw);
        const CommandResult recorded = Recorder::record(&self.recorder, call, engine);
        if (self.recorder.calls.size() == 1) {
            if (self.entries != nullptr) {
                self.entries[1] = self.replacement_entry;
            }
            const std::array word{self.mutation_word};
            self.guest->set_words(self.mutation_address, word);
        }
        return recorded;
    }
};

void test_descriptor_and_command_capture_points() {
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array first_words{header(4, 2, 0, 1)};
    const std::array second_words{header(4, 2, 0, 2)};
    guest.set_words(0x1000, first_words);
    guest.set_words(0x1100, second_words);
    std::array descriptors{entry(0x1000, 1), entry(0x1100, 1)};

    MutationContext later;
    later.guest = &guest;
    later.entries = descriptors.data();
    later.replacement_entry = entry(0x1f00, 1);
    later.mutation_address = 0x1100;
    later.mutation_word = header(4, 2, 0, 7);
    GpfifoChannel channel;
    CHECK(channel.process(descriptors, memory, MutationContext::mutate, &later)
              .ok());
    CHECK(later.recorder.calls.size() == 2);
    CHECK(later.recorder.calls[1].call.argument == 7);
    CHECK(guest.reads.size() == 2);
    CHECK(guest.reads[1].first == guest_base + 0x100);

    GuestMemory current_guest;
    GpuAddressSpace current_memory = current_guest.address_space();
    const std::array current_words{header(4, 2, 0, 3),
                                   header(4, 2, 0, 4)};
    current_guest.set_words(0x1200, current_words);
    MutationContext current;
    current.guest = &current_guest;
    current.mutation_address = 0x1204;
    current.mutation_word = header(4, 2, 0, 9);
    const std::uint64_t current_entry = entry(0x1200, 2);
    GpfifoChannel current_channel;
    CHECK(current_channel
              .process({&current_entry, 1}, current_memory,
                       MutationContext::mutate, &current)
              .ok());
    CHECK(current.recorder.calls.size() == 2);
    CHECK(current.recorder.calls[1].call.argument == 4);
}

void test_faults_and_stops_are_sticky() {
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array words{header(4, 2, 0, 1)};
    guest.set_words(0x1000, words);
    guest.set_words(0x1100, words);
    std::array descriptors{entry(0x1000, 1), entry(0x1100, 1)};

    guest.fail_address = guest_base;
    GpfifoChannel faulted;
    Recorder fault_recorder;
    const CommandResult fault = faulted.process(
        descriptors, memory, Recorder::record, &fault_recorder);
    CHECK(fault.status == CommandStatus::MemoryFault);
    CHECK(guest.reads.size() == 1);
    CHECK(fault_recorder.calls.empty());
    CHECK(faulted.process({descriptors.data() + 1, 1}, memory,
                          Recorder::record, &fault_recorder)
              .status == CommandStatus::MemoryFault);
    CHECK(guest.reads.size() == 1);

    GuestMemory stop_guest;
    GpuAddressSpace stop_memory = stop_guest.address_space();
    stop_guest.set_words(0x1000, words);
    stop_guest.set_words(0x1100, words);
    GpfifoChannel stopped;
    Recorder stop_recorder;
    stop_recorder.stop_status = CommandStatus::Stopped;
    stop_recorder.stop_at = 0;
    const CommandResult stop = stopped.process(
        descriptors, stop_memory, Recorder::record, &stop_recorder);
    CHECK(stop.status == CommandStatus::Stopped);
    CHECK(stop_guest.reads.size() == 1);
    CHECK(stop_recorder.calls.size() == 1);
    CHECK(stop_recorder.completed == 0);
    CHECK(stopped.process({descriptors.data() + 1, 1}, stop_memory,
                          Recorder::record, &stop_recorder)
              .status == CommandStatus::Stopped);
    CHECK(stop_guest.reads.size() == 1);
    CHECK(stop_recorder.calls.size() == 1);
}

CommandResult process_one_word(std::uint32_t word, Recorder& recorder) {
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array words{word};
    guest.set_words(0x1000, words);
    GpfifoChannel channel;
    const std::uint64_t descriptor = entry(0x1000, 1);
    return channel.process({&descriptor, 1}, memory, Recorder::record, &recorder);
}

void test_empty_legacy_headers_are_noops() {
    Recorder recorder;
    constexpr std::uint32_t empty_plain = header(0, 5, 0, 0);
    constexpr std::uint32_t empty_fields = header(0, 0xfff, 7, 0);
    constexpr std::uint32_t empty_extension = 1u << 12;
    constexpr std::uint32_t empty_extension_fields =
        (1u << 12) | header(0, 0x37b, 5, 0);
    static_assert(empty_plain == 5u);
    for (const std::uint32_t word :
         {empty_plain, empty_fields, empty_extension, empty_extension_fields}) {
        CHECK(process_one_word(word, recorder).ok());
    }
    CHECK(recorder.calls.empty());

    // A word that decodes as an empty header stays a real argument while a
    // packet is active, and empty headers around it never disturb bindings
    // or the following packets, across entries and submissions.
    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array nop_entry{empty_fields};
    const std::array bound_words{header(3, 0, 2, 1), threed,
                                 header(1, 0x40, 2, 3), empty_plain, 0x11u};
    const std::array tail_words{0x22u, 0x33u, empty_fields,
                                header(4, 0x50, 2, 0x33)};
    guest.set_words(0x1000, nop_entry);
    guest.set_words(0x1050, bound_words);
    guest.set_words(0x1100, tail_words);
    const std::array first_entries{entry(0x1000, nop_entry.size()),
                                   entry(0x1050, bound_words.size())};

    GpfifoChannel channel;
    CHECK(channel.process(first_entries, memory, Recorder::record, &recorder)
              .ok());
    CHECK(channel.has_pending());
    CHECK(recorder.calls.size() == 2);
    CHECK(recorder.calls[0].call.method == 0x40);
    CHECK(recorder.calls[0].call.argument == empty_plain);
    CHECK(recorder.calls[0].engine == threed);
    CHECK(!recorder.calls[1].call.last);
    CHECK(recorder.calls[1].call.method == 0x41);
    CHECK(channel.bindings()[2] == threed);

    const std::uint64_t tail = entry(0x1100, tail_words.size());
    CHECK(channel.process({&tail, 1}, memory, Recorder::record, &recorder)
              .ok());
    CHECK(!channel.has_pending());
    CHECK(recorder.calls.size() == 4);
    CHECK(recorder.calls[2].call.method == 0x42);
    CHECK(recorder.calls[2].call.argument == 0x22);
    CHECK(recorder.calls[2].call.last);
    CHECK(recorder.calls[3].call.method == 0x50);
    CHECK(recorder.calls[3].call.argument == 0x33);
    CHECK(recorder.calls[3].call.last);
    CHECK(recorder.calls[3].engine == threed);

    // Nonempty legacy packets and standard zero-count packets keep their
    // explicit failures.
    constexpr std::uint32_t nonempty_legacy = header(0, 5, 0, 1);
    const CommandResult stopped =
        process_one_word(nonempty_legacy, recorder);
    CHECK(stopped.status == CommandStatus::Unsupported);
    CHECK(stopped.method == 5);
    CHECK(stopped.subchannel == 0);
    CHECK(stopped.argument == nonempty_legacy);
    CHECK(process_one_word(header(3, 0x10, 0, 0), recorder).status ==
          CommandStatus::InvalidHeader);
    CHECK(process_one_word(header(1, 0x10, 0, 0), recorder).status ==
          CommandStatus::InvalidHeader);
    CHECK(recorder.calls.size() == 4);
}

void test_invalid_headers_unbound_engines_and_budgets() {
    Recorder recorder;
    CHECK(process_one_word((1u << 12) | header(1, 0xabc, 2, 3), recorder)
              .status == CommandStatus::InvalidHeader);
    CHECK(process_one_word((1u << 12) | header(4, 2, 0, 1), recorder).status ==
          CommandStatus::InvalidHeader);
    CHECK(process_one_word(header(0, 1, 0, 1), recorder).status ==
          CommandStatus::Unsupported);
    CHECK(process_one_word(header(2, 1, 0, 1), recorder).status ==
          CommandStatus::Unsupported);
    CHECK(process_one_word(header(6, 1, 0, 1), recorder).status ==
          CommandStatus::Unsupported);
    CHECK(process_one_word(header(7, 1, 0, 1), recorder).status ==
          CommandStatus::Unsupported);
    CHECK(process_one_word(header(1, 1, 0, 0), recorder).status ==
          CommandStatus::InvalidHeader);
    for (const unsigned mode : {1u, 5u}) {
        const auto raw = header(mode, 0xfff, 3, 2);
        const auto failure = process_one_word(raw, recorder);
        CHECK(failure.status == CommandStatus::InvalidHeader);
        CHECK(failure.method == 0xfff);
        CHECK(failure.subchannel == 3);
        CHECK(failure.argument == raw);
    }
    CHECK(process_one_word(header(4, 0x100, 3, 1), recorder).status ==
          CommandStatus::UnboundEngine);

    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    GpfifoChannel channel;
    const std::uint64_t too_large =
        entry(0x1000, GpfifoChannel::max_entry_words + 1);
    CHECK(channel.process({&too_large, 1}, memory, Recorder::record, &recorder)
              .status == CommandStatus::BudgetExceeded);
    CHECK(guest.reads.empty());

    const std::array nop{0u};
    guest.set_words(0x1000, nop);
    GpfifoChannel zero_budget;
    const std::uint64_t one_word = entry(0x1000, 1);
    CHECK(zero_budget.process({&one_word, 1}, memory, Recorder::record,
                              &recorder, 0)
              .status == CommandStatus::BudgetExceeded);
    CHECK(guest.reads.empty());
}

void test_unsupported_header_mode_reports_raw_header() {
    constexpr std::uint32_t raw_header = header(7, 0x37b, 5, 0x1a5);
    static_assert(raw_header == 0xe1a5a37bu);

    GuestMemory guest;
    GpuAddressSpace memory = guest.address_space();
    const std::array words{raw_header};
    guest.set_words(0x1000, words);
    const std::uint64_t descriptor = entry(0x1000, 1);

    GpfifoChannel channel;
    Recorder recorder;
    const CommandResult stopped =
        channel.process({&descriptor, 1}, memory, Recorder::record, &recorder);
    CHECK(stopped.status == CommandStatus::Unsupported);
    CHECK(stopped.detail == "GPFIFO header mode is unsupported");
    CHECK(stopped.address == 0x1000);
    CHECK(stopped.entry_index == 0);
    CHECK(stopped.subchannel == 5);
    CHECK(stopped.method == 0x37b);
    CHECK(stopped.argument == raw_header);
    CHECK(stopped.engine == 0);
    CHECK(recorder.calls.empty());

    const CommandResult repeated =
        channel.process({&descriptor, 1}, memory, Recorder::record, &recorder);
    CHECK(repeated.status == stopped.status);
    CHECK(repeated.detail == stopped.detail);
    CHECK(repeated.address == stopped.address);
    CHECK(repeated.entry_index == stopped.entry_index);
    CHECK(repeated.subchannel == stopped.subchannel);
    CHECK(repeated.method == stopped.method);
    CHECK(repeated.argument == raw_header);
    CHECK(repeated.engine == 0);
    CHECK(guest.reads.size() == 1);
}

void test_captured_commands_preserve_segments_and_diagnostics() {
    GuestMemory guest;
    auto memory = guest.address_space();
    // The second segment overlaps the first GPU address, but its captured
    // argument completes the first segment's packet. No live read is allowed.
    const std::array words{header(1, 0, 0, 1), threed,
                           header(1, 0x100, 0, 1), 0x12345678u};
    guest.set_words(gpu_base, words);
    const std::vector<std::byte> snapshot(guest.bytes.begin(), guest.bytes.begin() + sizeof(words));
    guest.bytes.fill(std::byte{0xff});
    guest.fail_address = guest_base;
    const std::array descriptors{entry(gpu_base, 3), std::uint64_t{0}, entry(gpu_base, 1)};
    GpfifoChannel channel;
    Recorder recorder;
    CHECK(channel.process(descriptors, memory, Recorder::record, &recorder,
                          GpfifoChannel::default_word_budget, snapshot).ok());
    CHECK(!channel.has_pending());
    CHECK(recorder.calls.size() == 1);
    if (recorder.calls.size() == 1) {
        CHECK(recorder.calls[0].call.argument == 0x12345678u);
        CHECK(recorder.calls[0].call.address == gpu_base);
        CHECK(recorder.calls[0].call.method == 0x100);
    }
    CHECK(guest.reads.empty());

    // A truncated capture is sticky and reports the original segment identity;
    // it must never recover missing bytes from reused guest command memory.
    GpfifoChannel truncated;
    const auto failure = truncated.process(descriptors, memory, Recorder::record, &recorder,
        GpfifoChannel::default_word_budget, std::span(snapshot).first(12));
    CHECK(failure.status == CommandStatus::MemoryFault);
    CHECK(failure.entry_index == 2 && failure.address == gpu_base);
    CHECK(truncated.process(descriptors, memory, Recorder::record, &recorder,
                            GpfifoChannel::default_word_budget, snapshot).status == failure.status);
    CHECK(guest.reads.empty());
}

}  // namespace

int main() {
    test_header_modes_and_golden_fields();
    test_continuation_across_entries_and_submissions();
    test_dynamic_binding_and_rebinding();
    test_entry_controls_and_zero_forms();
    test_descriptor_and_command_capture_points();
    test_faults_and_stops_are_sticky();
    test_empty_legacy_headers_are_noops();
    test_invalid_headers_unbound_engines_and_budgets();
    test_unsupported_header_mode_reports_raw_header();
    test_captured_commands_preserve_segments_and_diagnostics();
    return failures == 0 ? 0 : 1;
}
