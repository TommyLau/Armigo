// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/horizon/handle_table.h>
#include <armigo/horizon/ipc.h>
#include <armigo/horizon/startup_queries.h>
#include <armigo/services/hid.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {
using armigo::horizon::AddressRange;
using armigo::horizon::Handle;
using armigo::horizon::HandleTable;
using armigo::horizon::IpcCallContext;
using armigo::horizon::IpcDispatcher;
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::horizon::QueryRegisters;
using armigo::horizon::TransferMemoryRecord;
using armigo::services::hid::HidSession;
using armigo::services::hid::SevenSixAxisSamplingState;

constexpr std::uint64_t test_pid = 0x1'0000'0047;
constexpr std::uint64_t ring_address = 0x10000;
constexpr std::uint64_t scratch_address = 0x20000;
constexpr std::size_t ring_size = 0x1000;
constexpr std::size_t scratch_size = 0x7f000;
constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint32_t hid_tag = 0x480dd4ea;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

void put32(std::span<std::byte> bytes, std::size_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void put64(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::uint32_t get32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::uint64_t get64(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

struct WireRequest {
    IpcMessage message{};
    std::size_t cmif = 0;
};

WireRequest pid_request(std::uint32_t type, std::uint32_t command, std::span<const std::byte> input,
                        std::span<const Handle> copied_handles = {}) {
    WireRequest request;
    put32(request.message, 0, type);
    const auto raw_words = static_cast<std::uint32_t>((16 + input.size() + 16) / 4);
    put32(request.message, 4, 0x80000000u | raw_words);
    put32(request.message, 8, 1u | (static_cast<std::uint32_t>(copied_handles.size()) << 1));
    put64(request.message, 12, 0xabcdef1234567890);
    for (std::size_t index = 0; index < copied_handles.size(); ++index)
        put32(request.message, 20 + index * 4, copied_handles[index]);
    const auto raw_start = 20 + copied_handles.size() * 4;
    request.cmif = (raw_start + 15) & ~std::size_t{15};
    put32(request.message, request.cmif, sfci);
    put32(request.message, request.cmif + 4, 1);
    put32(request.message, request.cmif + 8, command);
    std::copy(input.begin(), input.end(), request.message.begin() + request.cmif + 16);
    return request;
}

WireRequest aruid_request(std::uint32_t type, std::uint32_t command, std::uint64_t aruid) {
    std::array<std::byte, 8> input{};
    put64(input, 0, aruid);
    return pid_request(type, command, input);
}

WireRequest initialize_request(std::uint32_t type, std::uint64_t aruid,
                               std::span<const Handle> copied_handles,
                               std::uint64_t first_size = ring_size,
                               std::uint64_t second_size = scratch_size) {
    std::array<std::byte, 24> input{};
    put64(input, 0, aruid);
    put64(input, 8, first_size);
    put64(input, 16, second_size);
    return pid_request(type, 306, input, copied_handles);
}

struct MemoryView {
    explicit MemoryView(std::vector<std::byte> *storage = nullptr) : bytes(storage) {}

    std::vector<std::byte> *bytes = nullptr;
    std::uint64_t fail_read_begin = 0;
    std::uint64_t fail_read_end = 0;
    std::uint64_t fail_write_begin = 0;
    std::uint64_t fail_write_end = 0;
    std::size_t reads = 0;
    std::size_t writes = 0;

    static bool overlaps(std::uint64_t address, std::size_t size, std::uint64_t first,
                         std::uint64_t last) noexcept {
        return first < last && size != 0 && address < last &&
               first < address + static_cast<std::uint64_t>(size);
    }

    static bool read(void *context, std::uint64_t address, std::span<std::byte> output) noexcept {
        auto &self = *static_cast<MemoryView *>(context);
        ++self.reads;
        if (!self.bytes || address > self.bytes->size() ||
            output.size() > self.bytes->size() - static_cast<std::size_t>(address) ||
            overlaps(address, output.size(), self.fail_read_begin, self.fail_read_end))
            return false;
        std::copy_n(self.bytes->begin() + static_cast<std::size_t>(address), output.size(),
                    output.begin());
        return true;
    }

    static bool write(void *context, std::uint64_t address,
                      std::span<const std::byte> input) noexcept {
        auto &self = *static_cast<MemoryView *>(context);
        ++self.writes;
        if (!self.bytes || address > self.bytes->size() ||
            input.size() > self.bytes->size() - static_cast<std::size_t>(address) ||
            overlaps(address, input.size(), self.fail_write_begin, self.fail_write_end))
            return false;
        std::copy(input.begin(), input.end(),
                  self.bytes->begin() + static_cast<std::size_t>(address));
        return true;
    }
};

struct CallResult {
    IpcDisposition disposition = IpcDisposition::unsupported;
    QueryRegisters registers{};
    IpcMessage message{};
};

struct TrackedRecord {
    Handle handle = 0;
    std::weak_ptr<TransferMemoryRecord> object;
};

struct Fixture {
    explicit Fixture(SevenSixAxisSamplingState initial_state = {})
        : handles(16), session(std::make_shared<HidSession>(test_pid, initial_state)),
          memory(0xa0000, std::byte{0xcc}), default_view{&memory} {
        session_handle = handles.insert(
            std::make_shared<armigo::horizon::ForwardingSession>(session)).value();
        std::fill_n(memory.begin() + ring_address, ring_size, std::byte{0x6b});
        std::fill_n(memory.begin() + scratch_address, scratch_size, std::byte{0x7d});
    }

    CallResult call(WireRequest request, MemoryView *view = nullptr) {
        CallResult result;
        result.registers.fill(0x123456789abcdef0);
        result.registers[0] = session_handle;
        result.message = request.message;
        auto &selected = view ? *view : default_view;
        IpcCallContext caller;
        caller.process_id = test_pid;
        caller.message = result.message;
        caller.read_memory = MemoryView::read;
        caller.write_memory = MemoryView::write;
        caller.memory_context = &selected;
        result.disposition = dispatcher.dispatch(0x21, result.registers, caller);
        return result;
    }

    void register_resource(std::uint32_t type) {
        const auto result = call(aruid_request(type, 0, test_pid));
        CHECK(result.disposition == IpcDisposition::handled);
        CHECK(result.registers[0] == 0);
        CHECK(get32(result.message, 32) == sfco);
        CHECK(get32(result.message, 40) == 0);
        CHECK(get32(result.message, 44) == hid_tag);
    }

    TrackedRecord record(std::uint64_t address, std::uint64_t size, std::uint32_t permission,
                         unsigned identity) {
        auto log = destruction_order;
        auto object = std::shared_ptr<TransferMemoryRecord>(
            new TransferMemoryRecord(AddressRange{address, size}, permission),
            [log, identity](TransferMemoryRecord *record) {
                log->push_back(identity);
                delete record;
            });
        TrackedRecord result;
        result.object = object;
        result.handle = handles.insert(std::move(object)).value();
        return result;
    }

    HandleTable handles;
    std::shared_ptr<HidSession> session;
    Handle session_handle = 0;
    IpcDispatcher dispatcher{handles, nullptr};
    std::vector<std::byte> memory;
    MemoryView default_view;
    std::shared_ptr<std::vector<unsigned>> destruction_order =
        std::make_shared<std::vector<unsigned>>();
};

void check_reply(const CallResult &result, std::uint32_t expected_result = 0) {
    CHECK(result.disposition == IpcDisposition::handled);
    CHECK(result.registers[0] == 0);
    CHECK(get32(result.message, 4) == 8);
    CHECK(get32(result.message, 16) == sfco);
    CHECK(get32(result.message, 24) == expected_result);
    CHECK(get32(result.message, 28) == hid_tag);
}

std::vector<std::byte> range(const Fixture &fixture, std::uint64_t address, std::size_t size) {
    return {fixture.memory.begin() + static_cast<std::size_t>(address),
            fixture.memory.begin() + static_cast<std::size_t>(address) + size};
}

std::vector<std::byte> expected_initialized_ring(std::vector<std::byte> bytes) {
    std::fill_n(bytes.begin(), 0x20, std::byte{});
    bytes[8] = std::byte{33};
    for (std::size_t index = 0; index < 33; ++index) {
        const auto entry = bytes.begin() + 0x20 + index * 0x50;
        std::fill_n(entry, 8, std::byte{});
        std::fill_n(entry + 0x10, 0x40, std::byte{});
    }
    return bytes;
}

std::vector<std::byte> expected_initialized_scratch(std::vector<std::byte> bytes) {
    put64(bytes, 0, 0x7efe0);
    put32(bytes, 8, 0);
    put64(bytes, 0x7eff0, 0);
    put32(bytes, 0x7eff8, 2);
    return bytes;
}

void test_lifecycle(std::uint32_t type) {
    Fixture fixture;
    fixture.register_resource(type);
    const auto original_ring = range(fixture, ring_address, ring_size);
    const auto original_scratch = range(fixture, scratch_address, scratch_size);

    check_reply(fixture.call(aruid_request(type, 303, 0x2'0000'0047)), 0x60a);
    auto state = fixture.session->seven_six_axis_sampling_state();
    CHECK(!state.resource_running && !state.fallback_running);
    check_reply(fixture.call(aruid_request(type, 303, test_pid)));
    state = fixture.session->seven_six_axis_sampling_state();
    CHECK(!state.resource_running && !state.fallback_running);
    check_reply(fixture.call(aruid_request(type, 304, test_pid)));
    check_reply(fixture.call(aruid_request(type, 304, test_pid)));
    state = fixture.session->seven_six_axis_sampling_state();
    CHECK(state.resource_running && !state.fallback_running);
    check_reply(fixture.call(aruid_request(type, 310, test_pid)));
    CHECK(range(fixture, ring_address, ring_size) == original_ring);
    CHECK(range(fixture, scratch_address, scratch_size) == original_scratch);

    const auto first = fixture.record(ring_address, ring_size, 1, 1);
    const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
    const std::array handles{first.handle, second.handle};
    check_reply(fixture.call(initialize_request(type, test_pid, handles)));
    const auto initialized_ring = expected_initialized_ring(original_ring);
    const auto initialized_scratch = expected_initialized_scratch(original_scratch);
    CHECK(range(fixture, ring_address, ring_size) == initialized_ring);
    CHECK(range(fixture, scratch_address, scratch_size) == initialized_scratch);
    CHECK(get64(fixture.memory, ring_address) == 0);
    CHECK(get64(fixture.memory, ring_address + 8) == 33);
    CHECK(get64(fixture.memory, ring_address + 16) == 0);
    CHECK(get64(fixture.memory, ring_address + 24) == 0);
    CHECK(get64(fixture.memory, scratch_address) == 0x7efe0);
    CHECK(get32(fixture.memory, scratch_address + 8) == 0);
    CHECK(get64(fixture.memory, scratch_address + 0x7eff0) == 0);
    CHECK(get32(fixture.memory, scratch_address + 0x7eff8) == 2);
    CHECK(first.object.use_count() == 2 && second.object.use_count() == 2);

    const auto before_duplicate = fixture.memory;
    const auto duplicate = fixture.call(initialize_request(type, test_pid, handles));
    CHECK(duplicate.disposition == IpcDisposition::unsupported);
    CHECK(fixture.memory == before_duplicate);
    CHECK(first.object.use_count() == 2 && second.object.use_count() == 2);

    CHECK(fixture.handles.close(first.handle));
    CHECK(fixture.handles.close(second.handle));
    CHECK(first.object.use_count() == 1 && second.object.use_count() == 1);
    CHECK(fixture.destruction_order->empty());

    MemoryView finalize_view{&fixture.memory};
    const auto initialization_writes = fixture.default_view.writes;
    check_reply(fixture.call(aruid_request(type, 310, test_pid), &finalize_view));
    CHECK(fixture.default_view.writes == initialization_writes);
    CHECK(finalize_view.writes == 0);
    CHECK(fixture.memory == before_duplicate);
    check_reply(fixture.call(aruid_request(type, 307, test_pid), &finalize_view));
    state = fixture.session->seven_six_axis_sampling_state();
    CHECK(!state.resource_running && !state.fallback_running);
    CHECK(first.object.expired() && second.object.expired());
    CHECK(*fixture.destruction_order == std::vector<unsigned>({2, 1}));
    CHECK(finalize_view.writes == 1);
    CHECK(range(fixture, ring_address, ring_size) == initialized_ring);

    auto finalized_scratch = initialized_scratch;
    std::fill(finalized_scratch.begin() + 0x10, finalized_scratch.begin() + 0x7eff0, std::byte{});
    CHECK(range(fixture, scratch_address, scratch_size) == finalized_scratch);
    const auto finalized_memory = fixture.memory;
    check_reply(fixture.call(aruid_request(type, 307, test_pid), &finalize_view));
    CHECK(fixture.memory == finalized_memory);
}

void test_fallback_selection() {
    Fixture fixture{{true, true}};
    check_reply(fixture.call(aruid_request(6, 303, 0)));
    auto state = fixture.session->seven_six_axis_sampling_state();
    CHECK(state.resource_running && state.fallback_running);
    check_reply(fixture.call(aruid_request(6, 304, 0)));
    check_reply(fixture.call(aruid_request(6, 310, 0)));
    state = fixture.session->seven_six_axis_sampling_state();
    CHECK(state.resource_running && state.fallback_running);
    check_reply(fixture.call(aruid_request(6, 307, 0)));
    state = fixture.session->seven_six_axis_sampling_state();
    CHECK(state.resource_running && !state.fallback_running);
    check_reply(fixture.call(aruid_request(6, 307, 0)));
}

void test_descriptor_validation() {
    for (const auto invalid : std::array{std::pair{std::uint64_t{0x2000}, std::uint32_t{1}},
                                         std::pair{std::uint64_t{ring_size}, std::uint32_t{0}}}) {
        Fixture fixture;
        fixture.register_resource(6);
        const auto before = fixture.memory;
        const auto first = fixture.record(ring_address, invalid.first, invalid.second, 1);
        const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
        const std::array handles{first.handle, second.handle};
        check_reply(fixture.call(initialize_request(6, test_pid, handles)), 0x878ca);
        CHECK(fixture.memory == before);
        CHECK(first.object.use_count() == 1 && second.object.use_count() == 1);
    }

    Fixture unsupported_size;
    unsupported_size.register_resource(6);
    const auto first = unsupported_size.record(ring_address, ring_size, 1, 1);
    const auto second = unsupported_size.record(scratch_address, scratch_size, 0, 2);
    const std::array handles{first.handle, second.handle};
    const auto before = unsupported_size.memory;
    const auto result =
        unsupported_size.call(initialize_request(6, test_pid, handles, 0x2000, scratch_size));
    CHECK(result.disposition == IpcDisposition::unsupported);
    CHECK(unsupported_size.memory == before);
    CHECK(first.object.use_count() == 1 && second.object.use_count() == 1);
}

void test_registration_and_object_validation() {
    {
        Fixture fixture;
        const auto first = fixture.record(ring_address, ring_size, 1, 1);
        const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
        const std::array handles{first.handle, second.handle};
        const auto before = fixture.memory;
        const auto result = fixture.call(initialize_request(6, test_pid, handles));
        CHECK(result.disposition == IpcDisposition::unsupported);
        CHECK(fixture.memory == before);
        CHECK(first.object.use_count() == 1 && second.object.use_count() == 1);
    }

    {
        Fixture fixture;
        fixture.register_resource(6);
        const auto first = fixture.record(ring_address, ring_size, 1, 1);
        const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
        const std::array handles{first.handle, second.handle};
        const auto before = fixture.memory;
        check_reply(fixture.call(initialize_request(6, 0x2'0000'0047, handles)), 0x60a);
        CHECK(fixture.memory == before);
        CHECK(first.object.use_count() == 1 && second.object.use_count() == 1);
    }

    {
        Fixture fixture;
        fixture.register_resource(6);
        const auto wrong =
            fixture.handles.insert(std::make_shared<armigo::horizon::Event>()).value();
        const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
        const std::array handles{wrong, second.handle};
        const auto before = fixture.memory;
        check_reply(fixture.call(initialize_request(6, test_pid, handles)), 0x878ca);
        CHECK(fixture.memory == before);
        CHECK(second.object.use_count() == 1);
    }
}

void test_memory_failures() {
    for (const bool fail_read : {true, false}) {
        Fixture fixture;
        fixture.register_resource(4);
        const auto first = fixture.record(ring_address, ring_size, 1, 1);
        const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
        const std::array handles{first.handle, second.handle};
        const auto before = fixture.memory;
        MemoryView failing{&fixture.memory};
        if (fail_read) {
            failing.fail_read_begin = ring_address;
            failing.fail_read_end = ring_address + ring_size;
        } else {
            failing.fail_write_begin = ring_address;
            failing.fail_write_end = ring_address + ring_size;
        }
        check_reply(fixture.call(initialize_request(4, test_pid, handles), &failing), 0x878ca);
        CHECK(fixture.memory == before);
        CHECK(first.object.use_count() == 1 && second.object.use_count() == 1);
    }

    Fixture partial;
    partial.register_resource(4);
    const auto original_ring = range(partial, ring_address, ring_size);
    const auto original_scratch = range(partial, scratch_address, scratch_size);
    const auto first = partial.record(ring_address, ring_size, 1, 1);
    const auto second = partial.record(scratch_address, scratch_size, 0, 2);
    const std::array handles{first.handle, second.handle};
    MemoryView failing_second{&partial.memory};
    failing_second.fail_write_begin = scratch_address;
    failing_second.fail_write_end = scratch_address + scratch_size;
    check_reply(partial.call(initialize_request(4, test_pid, handles), &failing_second), 0x878ca);
    const auto initialized_ring = expected_initialized_ring(original_ring);
    CHECK(range(partial, ring_address, ring_size) == initialized_ring);
    CHECK(range(partial, scratch_address, scratch_size) == original_scratch);
    CHECK(first.object.use_count() == 2 && second.object.use_count() == 1);
    CHECK(partial.handles.close(first.handle));
    CHECK(partial.handles.close(second.handle));
    CHECK(first.object.use_count() == 1 && second.object.expired());
    CHECK(*partial.destruction_order == std::vector<unsigned>({2}));
    MemoryView finalizer{&partial.memory};
    check_reply(partial.call(aruid_request(4, 307, test_pid), &finalizer));
    CHECK(first.object.expired());
    CHECK(*partial.destruction_order == std::vector<unsigned>({2, 1}));
    CHECK(finalizer.writes == 0);
    CHECK(range(partial, ring_address, ring_size) == initialized_ring);
    CHECK(range(partial, scratch_address, scratch_size) == original_scratch);
}

void test_invalid_copied_handle() {
    Fixture fixture;
    fixture.register_resource(6);
    const auto first = fixture.record(ring_address, ring_size, 1, 1);
    const std::array handles{first.handle, Handle{0xffffffff}};
    const auto request = initialize_request(6, test_pid, handles);
    const auto before = fixture.memory;
    const auto result = fixture.call(request);
    CHECK(result.disposition == IpcDisposition::handled);
    CHECK(result.registers[0] == 0xe401);
    CHECK(result.message == request.message);
    CHECK(fixture.memory == before);
    CHECK(first.object.use_count() == 1);
    const auto state = fixture.session->seven_six_axis_sampling_state();
    CHECK(!state.resource_running && !state.fallback_running);
}

void test_pseudo_handles_are_not_transfer_memory() {
    for (const auto pseudo : {armigo::horizon::current_process_handle,
                              armigo::horizon::current_thread_handle}) {
        Fixture fixture;
        fixture.register_resource(6);
        const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
        const std::array handles{pseudo, second.handle};
        const auto before = fixture.memory;
        check_reply(fixture.call(initialize_request(6, test_pid, handles)), 0x878ca);
        CHECK(fixture.memory == before);
        CHECK(second.object.use_count() == 1);
    }
}

void test_copied_handles_after_pid() {
    Fixture fixture;
    fixture.register_resource(6);
    const auto first = fixture.record(ring_address, ring_size, 1, 1);
    const auto second = fixture.record(scratch_address, scratch_size, 0, 2);
    const std::array handles{first.handle, second.handle};
    auto request = initialize_request(6, test_pid, handles);
    // PID storage can still contain a previous reply's magic. It is neither
    // a copied handle nor the trusted identity supplied by the dispatcher.
    put64(request.message, 12, std::uint64_t{sfco} << 32);
    CHECK(get32(request.message, 0) == 6);
    CHECK(get32(request.message, 4) == 0x8000000e);
    CHECK(get32(request.message, 8) == 5);
    CHECK(get32(request.message, 20) == first.handle);
    CHECK(get32(request.message, 24) == second.handle);
    CHECK(request.cmif == 32);
    const auto original_ring = range(fixture, ring_address, ring_size);
    const auto original_scratch = range(fixture, scratch_address, scratch_size);
    check_reply(fixture.call(request));
    CHECK(range(fixture, ring_address, ring_size) == expected_initialized_ring(original_ring));
    CHECK(range(fixture, scratch_address, scratch_size) ==
          expected_initialized_scratch(original_scratch));
    CHECK(first.object.use_count() == 2 && second.object.use_count() == 2);
    CHECK(fixture.handles.lookup(first.handle) == first.object.lock());
    CHECK(fixture.handles.lookup(second.handle) == second.object.lock());
}
} // namespace

int main() {
    test_copied_handles_after_pid();
    test_lifecycle(4);
    test_lifecycle(6);
    test_fallback_selection();
    test_descriptor_validation();
    test_registration_and_object_validation();
    test_memory_failures();
    test_invalid_copied_handle();
    test_pseudo_handles_are_not_transfer_memory();
    if (failures != 0)
        std::fprintf(stderr, "%u failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
