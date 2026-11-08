// ============================================================================
// Armigo Tests - Human Interface Device Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/hid.h>

#include <armigo/horizon/event.h>
#include <armigo/horizon/handle_table.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::services::hid::AppletResourceSession;
using armigo::services::hid::HidSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

void put64(IpcMessage& message, std::size_t offset, std::uint64_t value) {
    std::memcpy(message.data() + offset, &value, 8);
}

// One readable input buffer for the supported-npad-id list.
struct Buffer {
    std::array<std::byte, 32> bytes{};
    static bool read(void* context, std::uint64_t address, std::span<std::byte> value) noexcept {
        auto& self = *static_cast<Buffer*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        std::memcpy(value.data(), self.bytes.data() + address, value.size());
        return true;
    }
    static bool write(void*, std::uint64_t, std::span<const std::byte>) noexcept { return false; }
};

IpcMessage stop_request(std::uint32_t type, std::uint64_t aruid) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, type);
    put(message, 4, 0x8000000a);
    put(message, 8, 1);
    put64(message, 12, 0xabcdef1234567890); // Untrusted transport PID placeholder.
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 305);
    put(message, 44, 0x12345678);
    put64(message, 48, aruid);
    return message;
}

void test_seven_six_axis_stop() {
    using namespace armigo::horizon;
    constexpr std::uint64_t pid = 0x100000047;
    for (const auto type : {4u, 6u}) {
        HandleTable handles{8};
        auto session = std::make_shared<HidSession>(
            pid, armigo::services::hid::SevenSixAxisSamplingState{true, true});
        const auto handle = handles.insert(session).value();
        IpcDispatcher dispatcher{handles, nullptr};
        IpcMessage registration{};
        put(registration, 0, type);
        put(registration, 4, 0x8000000a);
        put(registration, 8, 1);
        put64(registration, 12, 0xabcdef1234567890);
        put(registration, 32, 0x49434653);
        put(registration, 36, 1);
        put(registration, 40, 0);
        put64(registration, 48, pid);
        const auto acquired = session->request(registration, pid);
        CHECK(acquired.disposition == IpcDisposition::handled);
        auto resource = std::dynamic_pointer_cast<AppletResourceSession>(acquired.transferred[0]);
        CHECK(resource != nullptr);
        if (!resource) continue;
        const auto shared = resource->shared_memory();
        std::fill(shared->bytes().begin(), shared->bytes().end(), std::byte{0x6b});
        const auto shared_references = shared.use_count();
        const auto handle_count = handles.size();

        const auto state_is = [&](bool registered, bool fallback) {
            const auto state = session->seven_six_axis_sampling_state();
            CHECK(state.resource_running == registered);
            CHECK(state.fallback_running == fallback);
        };
        const auto send = [&](IpcMessage request, std::uint64_t trusted_pid,
                              IpcDisposition expected_disposition, std::uint32_t result = 0) {
            const auto before = request;
            QueryRegisters registers;
            registers.fill(0x123456789abcdef0);
            registers[0] = handle;
            const auto saved_registers = registers;
            IpcCallContext caller{trusted_pid, request, nullptr, nullptr};
            CHECK(dispatcher.dispatch(0x21, registers, caller) == expected_disposition);
            if (expected_disposition == IpcDisposition::handled) {
                IpcMessage expected{};
                put(expected, 4, 8);
                put(expected, 16, 0x4f434653);
                put(expected, 24, result);
                put(expected, 28, 0x480dd4ea);
                CHECK(std::equal(expected.begin(), expected.begin() + 40, request.begin()));
                CHECK(std::equal(before.begin() + 40, before.end(), request.begin() + 40));
                CHECK(registers[0] == 0);
                CHECK(session->diagnostic() == nullptr);
            } else {
                CHECK(request == before && registers == saved_registers);
                CHECK(session->diagnostic() != nullptr);
            }
            CHECK(handles.size() == handle_count);
            CHECK(session->domain_object_count() == 0);
            CHECK(shared.use_count() == shared_references);
            CHECK(std::all_of(shared->bytes().begin(), shared->bytes().end(),
                              [](std::byte value) { return value == std::byte{0x6b}; }));
            CHECK(session->supported_styles() == 0 && !session->activated());
        };

        send(stop_request(type, pid), pid + 1, IpcDisposition::unsupported);
        state_is(true, true);
        // Only the low word matches: the full inline identity must be compared.
        send(stop_request(type, static_cast<std::uint32_t>(pid)), pid,
             IpcDisposition::handled, 0x60a);
        state_is(true, true);
        send(stop_request(type, pid + 1), pid, IpcDisposition::handled, 0x60a);
        state_is(true, true);
        for (unsigned malformed = 0; malformed != 9; ++malformed) {
            auto request = stop_request(type, pid);
            switch (malformed) {
            case 0: put(request, 8, 0); break; // Missing transport PID.
            case 1: put(request, 4, 0x80000009); break; // Missing alignment allowance.
            case 2: put(request, 4, 0x80000008); break; // Truncated inline u64.
            case 3: put(request, 4, 0x800003ff); break; // Raw extent exceeds message.
            case 4: put(request, 0, type | 0x10000); break; // Pointer descriptor.
            case 5: put(request, 4, 0x8000040a); break; // Receive-list descriptor.
            case 6: put(request, 8, 3); break; // Copied handle.
            case 7: put(request, 8, 0x21); break; // Moved handle.
            case 8: put(request, 4, 10); break; // Missing special header.
            }
            if (malformed == 6) {
                // Invalid copied handles are now rejected by the dispatcher
                // before service dispatch. Check this malformed service shape
                // directly; transport handle failures have separate coverage.
                CHECK(session->request(request, pid).disposition == IpcDisposition::unsupported);
                state_is(true, true);
                continue;
            }
            send(request, pid, IpcDisposition::unsupported);
            state_is(true, true);
        }
        Buffer backing;
        GuestBuffers buffers{Buffer::read, Buffer::write, &backing};
        buffers.add({0, 4, true, false});
        CHECK(session->request_with_buffers(stop_request(type, pid), pid, buffers).disposition ==
              IpcDisposition::unsupported);
        state_is(true, true);

        send(stop_request(type, pid), pid, IpcDisposition::handled);
        state_is(false, true);
        send(stop_request(type, pid), pid, IpcDisposition::handled);
        state_is(false, true);
        send(stop_request(type, 0), pid, IpcDisposition::handled);
        state_is(false, false);
        send(stop_request(type, 0), pid, IpcDisposition::handled);
        state_is(false, false);

        HidSession unregistered{pid, {true, true}};
        const auto stopped = unregistered.request(stop_request(type, pid), pid);
        CHECK(stopped.disposition == IpcDisposition::handled && stopped.size == 40);
        CHECK(unregistered.seven_six_axis_sampling_state().resource_running);
        CHECK(!unregistered.seven_six_axis_sampling_state().fallback_running);
        HidSession zero_registered{pid, {true, true}};
        put64(registration, 48, 0);
        CHECK(zero_registered.request(registration, pid).disposition == IpcDisposition::handled);
        CHECK(zero_registered.request(stop_request(type, 0), pid).disposition == IpcDisposition::handled);
        CHECK(!zero_registered.seven_six_axis_sampling_state().resource_running);
        CHECK(zero_registered.seven_six_axis_sampling_state().fallback_running);
        HidSession inactive{pid};
        CHECK(inactive.request(stop_request(type, pid), pid).disposition == IpcDisposition::handled);
        CHECK(!inactive.seven_six_axis_sampling_state().resource_running);
        CHECK(!inactive.seven_six_axis_sampling_state().fallback_running);
        auto start = stop_request(type, pid);
        put(start, 40, 304);
        CHECK(inactive.request(start, pid).disposition == IpcDisposition::handled);
        CHECK(inactive.seven_six_axis_sampling_state().fallback_running);
    }
}
}  // namespace

namespace {
constexpr std::uint64_t test_pid = 17;
constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint32_t hid_tag = 0x480dd4ea;

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

void put32(std::span<std::byte> bytes, std::size_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void put_u64(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void put_float(std::span<std::byte> bytes, std::size_t offset, float value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

struct PointerInput {
    std::uint64_t address;
    std::uint32_t size;
};

struct WireRequest {
    IpcMessage message{};
    std::size_t cmif = 0;
};

WireRequest request(std::uint32_t command, bool send_pid, std::span<const std::byte> input = {},
                    std::span<const PointerInput> pointers = {}) {
    WireRequest result;
    const auto pointer_count = static_cast<std::uint32_t>(pointers.size());
    put(result.message, 0, 6u | (pointer_count << 16));
    const auto raw_words = static_cast<std::uint32_t>((32 + input.size() + 3) / 4);
    put(result.message, 4, raw_words | (send_pid ? 0x80000000u : 0u));
    std::size_t cursor = 8;
    if (send_pid) {
        put(result.message, cursor, 1);
        put64(result.message, cursor + 4, 0xabcdef1234567890);
        cursor += 12;
    }
    for (const auto& pointer : pointers) {
        const auto upper = pointer.address >> 32;
        const auto packed = (pointer.size << 16) |
                            (static_cast<std::uint32_t>((upper >> 4) & 7) << 6) |
                            (static_cast<std::uint32_t>(upper & 0xf) << 12);
        put(result.message, cursor, packed);
        put(result.message, cursor + 4, static_cast<std::uint32_t>(pointer.address));
        cursor += 8;
    }
    result.cmif = (cursor + 15) & ~std::size_t{15};
    put(result.message, result.cmif, sfci);
    put(result.message, result.cmif + 4, 1);
    put(result.message, result.cmif + 8, command);
    if (!input.empty())
        std::memcpy(result.message.data() + result.cmif + 16, input.data(), input.size());
    return result;
}

template <std::size_t Size>
WireRequest request(std::uint32_t command, bool send_pid, const std::array<std::byte, Size>& input,
                    std::span<const PointerInput> pointers = {}) {
    return request(command, send_pid, std::span<const std::byte>(input), pointers);
}

std::array<std::byte, 8> one_u64(std::uint64_t value) {
    std::array<std::byte, 8> input{};
    put_u64(input, 0, value);
    return input;
}

std::array<std::byte, 16> id_aruid(std::uint32_t id, std::uint64_t aruid = test_pid) {
    std::array<std::byte, 16> input{};
    put32(input, 0, id);
    put_u64(input, 8, aruid);
    return input;
}

std::array<std::byte, 24> id_aruid_extra(std::uint32_t id, std::uint64_t extra = 0) {
    std::array<std::byte, 24> input{};
    put32(input, 0, id);
    put_u64(input, 8, test_pid);
    put_u64(input, 16, extra);
    return input;
}

void check_reply(const armigo::horizon::IpcReply& reply, std::size_t size, std::uint32_t result = 0,
                 std::uint32_t tag = hid_tag) {
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.size == size);
    CHECK(get32(reply.bytes, 4) == (size - 8) / 4);
    CHECK(get32(reply.bytes, 16) == sfco);
    CHECK(get32(reply.bytes, 24) == result);
    CHECK(get32(reply.bytes, 28) == tag);
}

struct WireMemory {
    std::array<std::byte, 512> bytes{};

    static bool read(void* context, std::uint64_t address, std::span<std::byte> output) noexcept {
        auto& self = *static_cast<WireMemory*>(context);
        if (address > self.bytes.size() || output.size() > self.bytes.size() - address)
            return false;
        std::memcpy(output.data(), self.bytes.data() + address, output.size());
        return true;
    }

    static bool write(void* context, std::uint64_t address,
                      std::span<const std::byte> input) noexcept {
        auto& self = *static_cast<WireMemory*>(context);
        if (address > self.bytes.size() || input.size() > self.bytes.size() - address) return false;
        std::memcpy(self.bytes.data() + address, input.data(), input.size());
        return true;
    }
};

struct RootFixture {
    std::shared_ptr<HidSession> session = std::make_shared<HidSession>(test_pid);
    std::shared_ptr<AppletResourceSession> resource;

    RootFixture() {
        const auto created =
            session->request(request(0, true, one_u64(test_pid)).message, test_pid);
        check_reply(created, 40);
        CHECK(created.transferred_count == 1 && !created.transferred_as_copy);
        resource = std::dynamic_pointer_cast<AppletResourceSession>(created.transferred[0]);
        CHECK(resource != nullptr);
    }

    armigo::horizon::IpcReply call(std::uint32_t command, bool send_pid,
                                   std::span<const std::byte> input = {}) {
        return session->request(request(command, send_pid, input).message, test_pid);
    }
};

void test_resource_and_reply_protocol() {
    RootFixture fixture;
    const auto repeated =
        fixture.session->request(request(0, true, one_u64(test_pid)).message, test_pid);
    check_reply(repeated, 40);
    CHECK(repeated.transferred_count == 1 && repeated.transferred[0] == fixture.resource);

    auto shared_reply = fixture.resource->request(request(0, false).message, test_pid);
    check_reply(shared_reply, 40, 0, 0x1f3e5d57);
    CHECK(shared_reply.transferred_count == 1 && shared_reply.transferred_as_copy);
    CHECK(shared_reply.transferred[0] == fixture.resource->shared_memory());
    CHECK(fixture.resource->shared_memory()->size() == 0x40000);

    auto wrong = request(100, true, id_aruid(1, test_pid + 1));
    const auto mismatch = fixture.session->request(wrong.message, test_pid);
    check_reply(mismatch, 40, 0x60a);
    CHECK(fixture.session->supported_styles() == 0);

    const auto missing_resource =
        fixture.session->request(request(100, true, id_aruid(1, 0)).message, test_pid);
    check_reply(missing_resource, 40, 0x58cca);
    CHECK(fixture.session->supported_styles() == 0);

    auto malformed = request(100, true, id_aruid(1));
    put(malformed.message, 4, 0x8000000bu);
    const auto shared_bytes = fixture.resource->shared_memory()->bytes();
    const std::vector<std::byte> before(shared_bytes.begin(), shared_bytes.end());
    const auto rejected = fixture.session->request(malformed.message, test_pid);
    CHECK(rejected.disposition == IpcDisposition::unsupported);
    CHECK(fixture.session->diagnostic() != nullptr);
    CHECK(fixture.session->supported_styles() == 0);
    CHECK(std::equal(before.begin(), before.end(),
                     fixture.resource->shared_memory()->bytes().begin()));

    const auto unknown = fixture.call(999, true, one_u64(test_pid));
    CHECK(unknown.disposition == IpcDisposition::unsupported);
    CHECK(fixture.session->diagnostic() != nullptr);

    // Initialization must not report success without receiving and mapping its
    // two transfer-memory objects. This remains an explicit capability gap.
    CHECK(fixture.call(306, false).disposition == IpcDisposition::unsupported);
    CHECK(fixture.session->diagnostic() != nullptr);
}

void check_neutral_full_key(const armigo::horizon::SharedMemory& memory, bool connected) {
    constexpr std::size_t npad = 0x9a00;
    constexpr std::size_t ring = npad + 0x28;
    const auto bytes = memory.bytes();
    CHECK(get32(bytes, npad) == (connected ? 1u : 0u));
    CHECK(get64(bytes, ring + 8) == 17);
    const auto latest = get64(bytes, ring + 16);
    const auto count = get64(bytes, ring + 24);
    CHECK(latest < 17 && count > 0 && count <= 16);
    const auto sample = ring + 0x20 + static_cast<std::size_t>(latest) * 0x30;
    CHECK(get64(bytes, sample) != 0);
    CHECK(get64(bytes, sample + 8) != 0);
    CHECK(get64(bytes, sample) == 2 * get64(bytes, sample + 8));
    CHECK(get64(bytes, sample + 16) == 0);
    CHECK(get32(bytes, sample + 24) == 0 && get32(bytes, sample + 28) == 0);
    CHECK(get32(bytes, sample + 32) == 0 && get32(bytes, sample + 36) == 0);
    CHECK(get64(bytes, sample + 40) == (connected ? 1u : 0u));
}

void test_npad_configuration_and_shared_memory() {
    RootFixture fixture;
    auto query = fixture.call(101, true, one_u64(test_pid));
    check_reply(query, 44);
    CHECK(get32(query.bytes, 32) == 0);

    auto style = id_aruid(1);
    check_reply(fixture.call(100, true, style), 40);
    CHECK(fixture.session->supported_styles() == 1);
    query = fixture.call(101, true, one_u64(test_pid));
    check_reply(query, 44);
    CHECK(get32(query.bytes, 32) == 1);

    WireMemory memory;
    put32(memory.bytes, 0x40, 0);
    const std::array supported_descriptor{PointerInput{0x40, 40}};
    auto ids_request = request(102, true, one_u64(test_pid), supported_descriptor);
    armigo::horizon::GuestBuffers ids(WireMemory::read, WireMemory::write, &memory);
    ids.add({0x40, 40, true, false});
    check_reply(fixture.session->request_with_buffers(ids_request.message, test_pid, ids), 40);

    check_reply(fixture.call(103, true, one_u64(test_pid)), 40);
    CHECK(fixture.session->activated());
    const auto& bytes = fixture.resource->shared_memory()->bytes();
    constexpr std::size_t first_sample = 0x9a00 + 0x28 + 0x20 + 0x30;
    CHECK(get64(bytes, 0x9a00 + 0x28 + 8) == 17);
    CHECK(get64(bytes, 0x9a00 + 0x28 + 16) == 1);
    CHECK(get64(bytes, 0x9a00 + 0x28 + 24) == 1);
    CHECK(get64(bytes, first_sample) == 2);
    CHECK(get64(bytes, first_sample + 8) == 1);
    check_neutral_full_key(*fixture.resource->shared_memory(), true);

    const auto current_shared = fixture.resource->shared_memory()->bytes();
    const std::vector<std::byte> before_oversize(current_shared.begin(), current_shared.end());
    const std::array oversize_descriptor{PointerInput{0x80, 44}};
    auto oversize_request = request(102, true, one_u64(test_pid), oversize_descriptor);
    armigo::horizon::GuestBuffers oversize(WireMemory::read, WireMemory::write, &memory);
    oversize.add({0x80, 44, true, false});
    check_reply(fixture.session->request_with_buffers(oversize_request.message, test_pid, oversize),
                40, 0x596ca);
    CHECK(std::equal(before_oversize.begin(), before_oversize.end(),
                     fixture.resource->shared_memory()->bytes().begin()));

    auto missing_buffer = request(102, true, one_u64(test_pid), supported_descriptor);
    const auto before_missing = fixture.resource->shared_memory()->bytes();
    const std::vector<std::byte> missing_snapshot(before_missing.begin(), before_missing.end());
    const auto missing_reply = fixture.session->request(missing_buffer.message, test_pid);
    CHECK(missing_reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.session->diagnostic() != nullptr);
    CHECK(std::equal(missing_snapshot.begin(), missing_snapshot.end(),
                     fixture.resource->shared_memory()->bytes().begin()));

    const std::array empty_descriptor{PointerInput{0x20, 0}};
    auto empty_request = request(102, true, one_u64(test_pid), empty_descriptor);
    armigo::horizon::GuestBuffers empty(WireMemory::read, WireMemory::write, &memory);
    empty.add({0x20, 0, true, false});
    check_reply(fixture.session->request_with_buffers(empty_request.message, test_pid, empty), 40);

    check_reply(fixture.call(104, true, one_u64(test_pid)), 40);
    CHECK(!fixture.session->activated());
    check_neutral_full_key(*fixture.resource->shared_memory(), false);

    CHECK(!fixture.session->touch_activated());
    check_reply(fixture.call(11, true, one_u64(test_pid)), 40);
    CHECK(fixture.session->touch_activated());
    check_reply(fixture.session->request(request(11, true, one_u64(test_pid + 1)).message, test_pid), 40, 0x60a);

    CHECK(!fixture.session->mouse_activated());
    check_reply(fixture.call(21, true, one_u64(test_pid)), 40);
    CHECK(fixture.session->mouse_activated());
    check_reply(fixture.session->request(request(21, true, one_u64(test_pid + 1)).message, test_pid), 40, 0x60a);
    CHECK(!fixture.session->keyboard_activated());
    check_reply(fixture.call(31, true, one_u64(test_pid)), 40);
    CHECK(fixture.session->keyboard_activated());
    check_reply(fixture.session->request(request(31, true, one_u64(test_pid + 1)).message, test_pid), 40, 0x60a);
    std::array<std::byte, 16> mouse_version{};
    put32(mouse_version, 0, 1);
    put_u64(mouse_version, 8, test_pid);
    check_reply(fixture.call(3150, true, mouse_version), 40);
    CHECK(fixture.session->mouse_library_version() == 1);

    std::array<std::byte, 16> revision{};
    put32(revision, 0, 5);
    put_u64(revision, 8, test_pid);
    check_reply(fixture.call(109, true, revision), 40);
    CHECK(fixture.session->activated() && fixture.session->revision() == 5);

    check_reply(fixture.session->request_with_buffers(ids_request.message, test_pid, ids), 40);
    for (unsigned index = 0; index < 40; ++index)
        check_reply(fixture.call(100, true, style), 40);
    CHECK(get64(bytes, 0x9a00 + 0x28 + 24) == 16);
    check_neutral_full_key(*fixture.resource->shared_memory(), true);
    for (unsigned index = 1; index < 10; ++index)
        CHECK(get32(bytes, 0x9a00 + 0x5000 * index) == 0);
}

void test_style_events_and_assignments() {
    RootFixture fixture;
    auto event_reply = fixture.call(106, true, id_aruid_extra(0));
    check_reply(event_reply, 40);
    CHECK(event_reply.transferred_count == 1 && event_reply.transferred_as_copy);
    auto event = std::dynamic_pointer_cast<armigo::horizon::Event>(event_reply.transferred[0]);
    CHECK(event != nullptr && event->is_signaled());
    CHECK(event != nullptr && event->reset());

    const auto repeated = fixture.call(106, true, id_aruid_extra(0));
    check_reply(repeated, 40);
    CHECK(repeated.transferred[0] == event && event->is_signaled());
    CHECK(event->reset());

    WireMemory memory;
    put32(memory.bytes, 0x40, 0);
    const std::array descriptor{PointerInput{0x40, 4}};
    const auto ids_request = request(102, true, one_u64(test_pid), descriptor);
    armigo::horizon::GuestBuffers ids(WireMemory::read, WireMemory::write, &memory);
    ids.add({0x40, 4, true, false});
    check_reply(fixture.session->request_with_buffers(ids_request.message, test_pid, ids), 40);
    check_reply(fixture.call(103, true, one_u64(test_pid)), 40);
    CHECK(!event->is_signaled());
    check_reply(fixture.call(100, true, id_aruid(1)), 40);
    CHECK(event->is_signaled());

    std::array<std::byte, 16> hold{};
    put_u64(hold, 0, test_pid);
    put_u64(hold, 8, 1);
    check_reply(fixture.call(120, true, hold), 40);
    CHECK(fixture.session->npad_joy_hold_type() == 1);
    auto value = fixture.call(121, true, one_u64(test_pid));
    check_reply(value, 48);
    CHECK(get64(value.bytes, 32) == 1);

    check_reply(fixture.call(122, true, id_aruid(0)), 40);
    check_reply(fixture.call(124, true, id_aruid(0)), 40);
    std::array<std::byte, 24> single{};
    put32(single, 0, 0);
    put_u64(single, 8, test_pid);
    put_u64(single, 16, 1);
    check_reply(fixture.call(123, true, single), 40);

    std::array<std::byte, 16> mode{};
    put_u64(mode, 0, test_pid);
    put_u64(mode, 8, 2);
    check_reply(fixture.call(128, true, mode), 40);
    CHECK(fixture.session->npad_handheld_activation_mode() == 2);
    value = fixture.call(129, true, one_u64(test_pid));
    check_reply(value, 48);
    CHECK(get64(value.bytes, 32) == 2);
}

void test_vibration_and_six_axis() {
    RootFixture fixture;
    std::array<std::byte, 1> permit{};
    check_reply(fixture.call(204, false, permit), 40);
    auto permitted = fixture.call(205, false);
    check_reply(permitted, 44);
    CHECK(get32(permitted.bytes, 32) == 0);
    permit[0] = std::byte{1};
    check_reply(fixture.call(204, false, permit), 40);
    permitted = fixture.call(205, false);
    check_reply(permitted, 44);
    CHECK(get32(permitted.bytes, 32) == 1);
    const auto vibration_list = fixture.call(203, false);
    check_reply(vibration_list, 40);
    CHECK(vibration_list.transferred_count == 1 && !vibration_list.transferred_as_copy);
    auto active =
        std::dynamic_pointer_cast<armigo::horizon::ClientSession>(vibration_list.transferred[0]);
    CHECK(active != nullptr);
    if (active != nullptr) {
        std::array<std::byte, 4> handle{};
        put32(handle, 0, 3);
        auto opened = active->request(request(0, false, handle).message, test_pid);
        CHECK(opened.disposition == IpcDisposition::handled && opened.size == 40);
        CHECK(get32(opened.bytes, 4) == 8 && get32(opened.bytes, 16) == sfco);
        CHECK(get32(opened.bytes, 24) == 0 && get32(opened.bytes, 28) == 0x679703ea);
        opened = active->request(request(0, false, handle).message, test_pid);
        CHECK(opened.disposition == IpcDisposition::handled);
        put32(handle, 0, 0x00010003);
        opened = active->request(request(0, false, handle).message, test_pid);
        CHECK(opened.disposition == IpcDisposition::handled);
    }

    std::array<std::byte, 4> handle{};
    put32(handle, 0, 3);
    auto info = fixture.call(200, false, handle);
    check_reply(info, 48);
    CHECK(get32(info.bytes, 32) == 1 && get32(info.bytes, 36) == 1);
    put32(handle, 0, 0x00010003);
    info = fixture.call(200, false, handle);
    check_reply(info, 48);
    CHECK(get32(info.bytes, 32) == 1 && get32(info.bytes, 36) == 2);

    std::array<std::byte, 32> vibration{};
    put32(vibration, 0, 3);
    put_float(vibration, 4, 1.0f);
    put_float(vibration, 8, 80.0f);
    put_float(vibration, 12, 0.5f);
    put_float(vibration, 16, 160.0f);
    put_u64(vibration, 24, test_pid);
    check_reply(fixture.call(201, true, vibration), 40);
    // The virtual Pro Controller's motors are mounted with or without a host
    // actuator: the actual value is the one last applied.
    auto actual = fixture.call(202, true, id_aruid(3));
    check_reply(actual, 56);
    CHECK(get32(actual.bytes, 32) == 0x3f800000);
    CHECK(get32(actual.bytes, 36) == 0x42a00000);
    CHECK(get32(actual.bytes, 40) == 0x3f000000);
    CHECK(get32(actual.bytes, 44) == 0x43200000);
    // The other motor has not been sent anything.
    actual = fixture.call(202, true, id_aruid(0x00010003));
    check_reply(actual, 56);
    CHECK(get32(actual.bytes, 32) == 0 && get32(actual.bytes, 36) == 0x43200000);

    WireMemory memory;
    put32(memory.bytes, 0x40, 3);
    put32(memory.bytes, 0x44, 0x00010003);
    const std::array matched_descriptors{PointerInput{0x40, 8}, PointerInput{0x80, 32}};
    auto batch = request(206, false, one_u64(test_pid), matched_descriptors);
    armigo::horizon::GuestBuffers matched(WireMemory::read, WireMemory::write, &memory);
    matched.add({0x40, 8, true, false});
    matched.add({0x80, 32, true, false});
    check_reply(fixture.session->request_with_buffers(batch.message, test_pid, matched), 40);

    const std::array mismatched_descriptors{PointerInput{0x40, 8}, PointerInput{0x80, 16}};
    batch = request(206, false, one_u64(test_pid), mismatched_descriptors);
    armigo::horizon::GuestBuffers mismatched(WireMemory::read, WireMemory::write, &memory);
    mismatched.add({0x40, 8, true, false});
    mismatched.add({0x80, 16, true, false});
    check_reply(fixture.session->request_with_buffers(batch.message, test_pid, mismatched), 40,
                0x106ca);

    // DisconnectNpad: player two's controller is not presented, so it is
    // already disconnected; the presented controller and invalid ids are refused.
    std::array<std::byte, 16> disconnect{};
    put32(disconnect, 0, 1);
    put_u64(disconnect, 8, test_pid);
    check_reply(fixture.call(107, true, disconnect), 40);
    put32(disconnect, 0, 0);
    CHECK(fixture.call(107, true, disconnect).disposition == IpcDisposition::unsupported);
    put32(disconnect, 0, 9);
    CHECK(fixture.call(107, true, disconnect).disposition == IpcDisposition::unsupported);

    // IsVibrationDeviceMounted: an unknown style is refused; a FullKey motor
    // answers one byte.
    std::array<std::byte, 16> mount{};
    put32(mount, 0, 1);
    put_u64(mount, 8, test_pid);
    check_reply(fixture.call(211, true, mount), 40, 0xf4ca);
    put32(mount, 0, 3);
    const auto mounted = fixture.call(211, true, mount);
    check_reply(mounted, 44);
    CHECK(get32(mounted.bytes, 32) <= 1);

    // Switch 2 vibration (214): a u16 and the aruid, then the handle and value
    // arrays. It sends the values and never fails; Switch 1 has no 214.
    std::array<std::byte, 16> switch2_args{};
    put_u64(switch2_args, 8, test_pid);
    put32(memory.bytes, 0x40, 3);
    put_float(memory.bytes, 0x80, 0.25f);
    put_float(memory.bytes, 0x84, 100.0f);
    put_float(memory.bytes, 0x88, 0.75f);
    put_float(memory.bytes, 0x8c, 200.0f);
    const std::array switch2_descriptors{PointerInput{0x40, 4}, PointerInput{0x80, 16}};
    batch = request(214, false, switch2_args, switch2_descriptors);
    armigo::horizon::GuestBuffers switch2_buffers(WireMemory::read, WireMemory::write, &memory);
    switch2_buffers.add({0x40, 4, true, false});
    switch2_buffers.add({0x80, 16, true, false});
    CHECK(fixture.session->request_with_buffers(batch.message, test_pid, switch2_buffers).disposition ==
          IpcDisposition::unsupported);
    HidSession switch2(test_pid, {}, true);
    check_reply(switch2.request_with_buffers(batch.message, test_pid, switch2_buffers), 40);

    // FullKey's ordinary SDK handle has device index 2, unlike its vibration handles.
    check_reply(fixture.call(66, true, id_aruid(0x20003)), 40);
    std::array<std::byte, 16> drift{};
    put32(drift, 0, 0x20003);
    put32(drift, 4, 2);
    put_u64(drift, 8, test_pid);
    check_reply(fixture.call(79, true, drift), 40);
    check_reply(fixture.call(67, true, id_aruid(0x20003)), 40);
    // Sensors of controllers that are not presented start and stop too:
    // handheld (style 4, player ID 0x20) and dual Joy-Con (style 5) devices.
    check_reply(fixture.call(66, true, id_aruid(0x22004)), 40);
    check_reply(fixture.call(66, true, id_aruid(0x10005)), 40);
    check_reply(fixture.call(67, true, id_aruid(0x10005)), 40);
    // An invalid player ID or device index is not a controller device.
    CHECK(fixture.call(66, true, id_aruid(0x0903)).disposition == IpcDisposition::unsupported);
    CHECK(fixture.call(66, true, id_aruid(0x30003)).disposition == IpcDisposition::unsupported);
}

void test_dispatcher_transfers() {
    using namespace armigo::horizon;
    HandleTable handles{16};
    auto session = std::make_shared<HidSession>(test_pid);
    const auto root = handles.insert(session).value();
    IpcDispatcher dispatcher{handles, nullptr};
    QueryRegisters registers{};
    registers[0] = root;
    auto created = request(0, true, one_u64(test_pid)).message;
    IpcCallContext caller{test_pid, created, nullptr, nullptr};
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    const auto resource_handle = static_cast<Handle>(get32(created, 20));
    CHECK(get32(created, 8) == 0x21);
    CHECK(handles.lookup_as<AppletResourceSession>(resource_handle) != nullptr);

    // Exercise the transport's real pointer decoding, including its readable /
    // writable capability flags and omission of zero-length descriptors.
    WireMemory memory;
    put32(memory.bytes, 0x40, 0);
    caller.read_memory = WireMemory::read;
    caller.write_memory = WireMemory::write;
    caller.memory_context = &memory;
    const std::array id_descriptor{PointerInput{0x40, 4}};
    auto ids = request(102, true, one_u64(test_pid), id_descriptor).message;
    caller.message = ids;
    registers[0] = root;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    CHECK(get32(ids, 24) == 0);
    const std::array empty_descriptor{PointerInput{0, 0}};
    ids = request(102, true, one_u64(test_pid), empty_descriptor).message;
    caller.message = ids;
    registers[0] = root;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    CHECK(get32(ids, 24) == 0);

    put32(memory.bytes, 0x40, 3);
    const std::array vibration_descriptors{PointerInput{0x40, 4}, PointerInput{0x80, 16}};
    auto batch = request(206, false, one_u64(test_pid), vibration_descriptors).message;
    caller.message = batch;
    registers[0] = root;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    CHECK(get32(batch, 24) == 0);
    const std::array empty_vibration_descriptors{PointerInput{0, 0}, PointerInput{0, 0}};
    batch = request(206, false, one_u64(test_pid), empty_vibration_descriptors).message;
    caller.message = batch;
    registers[0] = root;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    CHECK(get32(batch, 24) == 0);

    registers[0] = root;
    auto acquire = request(106, true, id_aruid_extra(0)).message;
    caller.message = acquire;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    const auto event_handle = static_cast<Handle>(get32(acquire, 20));
    CHECK(get32(acquire, 8) == 3);
    CHECK(handles.lookup_as<Event>(event_handle) != nullptr);

    registers[0] = root;
    auto vibration = request(203, false).message;
    caller.message = vibration;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    const auto vibration_handle = static_cast<Handle>(get32(vibration, 20));
    CHECK(get32(vibration, 8) == 0x21);
    CHECK(handles.lookup_as<ClientSession>(vibration_handle) != nullptr);
}
}  // namespace

// Host input reaches player one's FullKey ring through the 5 ms sampler once
// the application's controller is connected.
void test_host_input_samples() {
    RootFixture fixture;
    check_reply(fixture.call(100, true, id_aruid(1)), 40);
    WireMemory memory;
    put32(memory.bytes, 0x40, 0);
    const std::array supported_descriptor{PointerInput{0x40, 40}};
    auto ids_request = request(102, true, one_u64(test_pid), supported_descriptor);
    armigo::horizon::GuestBuffers ids(WireMemory::read, WireMemory::write, &memory);
    ids.add({0x40, 40, true, false});
    check_reply(fixture.session->request_with_buffers(ids_request.message, test_pid, ids), 40);
    check_reply(fixture.call(103, true, one_u64(test_pid)), 40);
    armigo::services::hid::set_host_full_key_input({0x3, 1000, -2000, 3000, -4000});
    constexpr std::size_t ring = 0x9a00 + 0x28;
    const auto& bytes = fixture.resource->shared_memory()->bytes();
    const auto latest_sample = [&] {
        return ring + 0x20 + static_cast<std::size_t>(get64(bytes, ring + 16)) * 0x30;
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (get64(bytes, latest_sample() + 16) != 0x3 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto sample = latest_sample();
    CHECK(get64(bytes, sample + 16) == 0x3);
    CHECK(static_cast<std::int32_t>(get32(bytes, sample + 24)) == 1000);
    CHECK(static_cast<std::int32_t>(get32(bytes, sample + 28)) == -2000);
    CHECK(static_cast<std::int32_t>(get32(bytes, sample + 32)) == 3000);
    CHECK(static_cast<std::int32_t>(get32(bytes, sample + 36)) == -4000);
    CHECK(get64(bytes, sample + 40) == 1);
    CHECK(get64(bytes, sample) == 2 * get64(bytes, sample + 8));
}


// An immediate publication is the latest sample when it returns, without
// waiting for the 5 ms sampler (guest-present input scripts rely on it).
void test_host_input_published_now() {
    RootFixture fixture;
    check_reply(fixture.call(100, true, id_aruid(1)), 40);
    WireMemory memory;
    put32(memory.bytes, 0x40, 0);
    const std::array supported_descriptor{PointerInput{0x40, 40}};
    auto ids_request = request(102, true, one_u64(test_pid), supported_descriptor);
    armigo::horizon::GuestBuffers ids(WireMemory::read, WireMemory::write, &memory);
    ids.add({0x40, 40, true, false});
    check_reply(fixture.session->request_with_buffers(ids_request.message, test_pid, ids), 40);
    check_reply(fixture.call(103, true, one_u64(test_pid)), 40);
    constexpr std::size_t ring = 0x9a00 + 0x28;
    const auto& bytes = fixture.resource->shared_memory()->bytes();
    for (const std::uint64_t buttons : {0x5u, 0x9u, 0x0u}) {
        const auto x = static_cast<std::int16_t>(buttons * 100);
        armigo::services::hid::publish_host_full_key_input_now({buttons, x, -x, x, -x});
        const auto sample = ring + 0x20 + static_cast<std::size_t>(get64(bytes, ring + 16)) * 0x30;
        CHECK(get64(bytes, sample + 16) == buttons);
        CHECK(static_cast<std::int32_t>(get32(bytes, sample + 24)) == x);
        CHECK(static_cast<std::int32_t>(get32(bytes, sample + 28)) == -x);
    }
}


// Connects player one's FullKey controller (style FullKey, supported id 0,
// activated) on a fresh fixture.
void connect_full_key(RootFixture& fixture) {
    check_reply(fixture.call(100, true, id_aruid(1)), 40);
    WireMemory memory;
    put32(memory.bytes, 0x40, 0);
    const std::array supported_descriptor{PointerInput{0x40, 40}};
    auto ids_request = request(102, true, one_u64(test_pid), supported_descriptor);
    armigo::horizon::GuestBuffers ids(WireMemory::read, WireMemory::write, &memory);
    ids.add({0x40, 40, true, false});
    check_reply(fixture.session->request_with_buffers(ids_request.message, test_pid, ids), 40);
    check_reply(fixture.call(103, true, one_u64(test_pid)), 40);
}

float get_float(std::span<const std::byte> bytes, std::size_t offset) {
    return std::bit_cast<float>(get32(bytes, offset));
}

// Player one's FullKey six-axis ring: nothing before StartSixAxisSensor; then
// one sample per 5 ms with the host's motion, consecutive sampling numbers
// and even latches; a controller at rest without host motion; no new
// samples after StopSixAxisSensor.
void test_six_axis_samples() {
    RootFixture fixture;
    connect_full_key(fixture);
    constexpr std::size_t ring = 0x9a00 + 0x1758;
    const auto& bytes = fixture.resource->shared_memory()->bytes();
    // The GameCube trigger ring is initialized at 0x4228; 0x41a8 is the applet
    // footer, not a ring.
    CHECK(get64(bytes, 0x9a00 + 0x4228 + 8) == 17);
    CHECK(get64(bytes, 0x9a00 + 0x41b0) == 0);

    armigo::services::hid::HostFullKeyInput input;
    input.has_motion = true;
    input.acceleration = {0.0f, 0.0f, -1.0f};
    input.angular_velocity = {0.0f, 0.0f, 0.25f};
    armigo::services::hid::set_host_full_key_input(input);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(get64(bytes, ring + 0x18) == 0);  // not started: no samples

    check_reply(fixture.call(66, true, id_aruid(0x20003)), 40);
    const auto wait_for_count = [&](std::uint64_t count) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (get64(bytes, ring + 0x18) < count && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return get64(bytes, ring + 0x18) >= count;
    };
    CHECK(wait_for_count(16));
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    const auto newest = static_cast<std::size_t>(get64(bytes, ring + 0x10));
    const auto entry = [&](std::size_t index) { return ring + 0x20 + index * 0x68; };
    const auto latest = entry(newest), older = entry((newest + 16) % 17);
    const auto number = get64(bytes, latest + 0x10);
    CHECK(get64(bytes, latest) == number << 1);
    CHECK(number - get64(bytes, older + 0x10) == 1);  // neighbours are consecutive
    CHECK(get64(bytes, latest + 8) == 5'000'000);     // delta time in ns
    CHECK(get_float(bytes, latest + 0x18) == 0.0f && get_float(bytes, latest + 0x1c) == 0.0f &&
          get_float(bytes, latest + 0x20) == -1.0f);  // acceleration in G
    CHECK(get_float(bytes, latest + 0x2c) == 0.25f);  // angular velocity z, rev/s
    CHECK(get_float(bytes, latest + 0x38) > 0.0f);    // accumulated angle z
    CHECK(get32(bytes, latest + 0x60) == 1);          // IsConnected
    // The orientation's rows stay unit length, and the z row stays up while
    // the controller turns about z.
    for (std::size_t row = 0; row < 3; ++row) {
        const auto base = latest + 0x3c + row * 12;
        const float x = get_float(bytes, base), y = get_float(bytes, base + 4), z = get_float(bytes, base + 8);
        CHECK(std::fabs(x * x + y * y + z * z - 1.0f) < 1e-3f);
    }
    CHECK(get_float(bytes, latest + 0x3c + 24 + 8) > 0.999f);

    // Without host motion the sensor reports a controller at rest, face up.
    armigo::services::hid::set_host_full_key_input({});
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    const auto rest = entry(static_cast<std::size_t>(get64(bytes, ring + 0x10)));
    CHECK(get_float(bytes, rest + 0x20) == -1.0f);
    CHECK(get_float(bytes, rest + 0x24) == 0.0f && get_float(bytes, rest + 0x28) == 0.0f &&
          get_float(bytes, rest + 0x2c) == 0.0f);

    check_reply(fixture.call(67, true, id_aruid(0x20003)), 40);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto stopped = get64(bytes, entry(static_cast<std::size_t>(get64(bytes, ring + 0x10))) + 0x10);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    CHECK(get64(bytes, entry(static_cast<std::size_t>(get64(bytes, ring + 0x10))) + 0x10) == stopped);
}

// Vibration values reach the host sink per motor; the actual value is the
// last one applied while a sink is mounted; a refused permission stops both
// motors.
void test_vibration_sink() {
    struct Sent {
        unsigned device;
        armigo::services::hid::VibrationValue value;
    };
    std::vector<Sent> sent;
    std::mutex mutex;
    armigo::services::hid::set_host_vibration_sink(
        [&](unsigned device, const armigo::services::hid::VibrationValue& value) {
            const std::lock_guard lock(mutex);
            sent.push_back({device, value});
        });
    RootFixture fixture;
    std::array<std::byte, 1> permit{std::byte{1}};
    check_reply(fixture.call(204, false, permit), 40);

    const auto send = [&](std::uint32_t handle, float low, float high) {
        std::array<std::byte, 32> vibration{};
        put32(vibration, 0, handle);
        put_float(vibration, 4, low);
        put_float(vibration, 8, 80.0f);
        put_float(vibration, 12, high);
        put_float(vibration, 16, 300.0f);
        put_u64(vibration, 24, test_pid);
        check_reply(fixture.call(201, true, vibration), 40);
    };
    send(3, 1.0f, 0.5f);
    send(0x00010003, 0.25f, 0.0f);
    {
        const std::lock_guard lock(mutex);
        CHECK(sent.size() == 2);
        if (sent.size() == 2) {
            CHECK(sent[0].device == 0 && sent[0].value.amplitude_low == 1.0f &&
                  sent[0].value.frequency_low == 80.0f && sent[0].value.amplitude_high == 0.5f &&
                  sent[0].value.frequency_high == 300.0f);
            CHECK(sent[1].device == 1 && sent[1].value.amplitude_low == 0.25f);
        }
    }
    auto actual = fixture.call(202, true, id_aruid(3));
    check_reply(actual, 56);
    CHECK(get_float(actual.bytes, 32) == 1.0f && get_float(actual.bytes, 36) == 80.0f &&
          get_float(actual.bytes, 40) == 0.5f && get_float(actual.bytes, 44) == 300.0f);

    // A batch reaches the sink pair by pair.
    WireMemory memory;
    put32(memory.bytes, 0x40, 3);
    put32(memory.bytes, 0x44, 0x00010003);
    put_float(memory.bytes, 0x80, 0.75f);
    put_float(memory.bytes, 0x84, 160.0f);
    put_float(memory.bytes, 0x90, 0.125f);
    put_float(memory.bytes, 0x94, 160.0f);
    const std::array descriptors{PointerInput{0x40, 8}, PointerInput{0x80, 32}};
    auto batch = request(206, false, one_u64(test_pid), descriptors);
    armigo::horizon::GuestBuffers buffers(WireMemory::read, WireMemory::write, &memory);
    buffers.add({0x40, 8, true, false});
    buffers.add({0x80, 32, true, false});
    check_reply(fixture.session->request_with_buffers(batch.message, test_pid, buffers), 40);
    {
        const std::lock_guard lock(mutex);
        CHECK(sent.size() == 4);
        if (sent.size() == 4)
            CHECK(sent[2].device == 0 && sent[2].value.amplitude_low == 0.75f && sent[3].device == 1 &&
                  sent[3].value.amplitude_low == 0.125f);
    }

    // Refusing permission stops both motors and later values stay neutral.
    permit[0] = std::byte{0};
    check_reply(fixture.call(204, false, permit), 40);
    send(3, 1.0f, 1.0f);
    {
        const std::lock_guard lock(mutex);
        CHECK(sent.size() == 7);
        for (std::size_t i = 4; i < sent.size(); ++i)
            CHECK(sent[i].value.amplitude_low == 0.0f && sent[i].value.amplitude_high == 0.0f);
    }
    // The last applied value is the neutral one sent while not permitted.
    armigo::services::hid::set_host_vibration_sink({});
    actual = fixture.call(202, true, id_aruid(3));
    check_reply(actual, 56);
    CHECK(get32(actual.bytes, 32) == 0 && get32(actual.bytes, 36) == 0x43200000);
}

// A sensor started before player one's controller connects publishes nothing
// until it connects.
void test_six_axis_waits_for_connection() {
    RootFixture fixture;
    check_reply(fixture.call(66, true, id_aruid(0x20003)), 40);
    constexpr std::size_t ring = 0x9a00 + 0x1758;
    const auto& bytes = fixture.resource->shared_memory()->bytes();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(get64(bytes, ring + 0x18) == 0);
    connect_full_key(fixture);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (get64(bytes, ring + 0x18) == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(get64(bytes, ring + 0x18) > 0);
}

int main() {
    test_seven_six_axis_stop();
    test_resource_and_reply_protocol();
    test_npad_configuration_and_shared_memory();
    test_style_events_and_assignments();
    test_vibration_and_six_axis();
    test_dispatcher_transfers();
    // Last: the host input source stays set for the rest of the process.
    test_host_input_samples();
    test_host_input_published_now();
    test_six_axis_samples();
    test_vibration_sink();
    test_six_axis_waits_for_connection();
    if (failures == 0) std::fprintf(stderr, "HID service failures: 0\n");
    return failures == 0 ? 0 : 1;
}
