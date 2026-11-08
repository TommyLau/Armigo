// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/shared_clock.h>
#include <armigo/services/time_session.h>

#include <algorithm>
#include <cstdio>

namespace {
using namespace armigo::horizon;
using armigo::services::time::TimeServiceSession;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)

void put(IpcMessage& message, std::size_t at, std::uint32_t value, unsigned width = 4) {
    for (unsigned i = 0; i < width; ++i)
        message[at + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

std::uint32_t get(const IpcMessage& message, std::size_t at, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned i = width; i != 0; --i)
        value = value * 256 + std::to_integer<unsigned char>(message[at + i - 1]);
    return value;
}

IpcMessage control(unsigned type, unsigned command) {
    IpcMessage message{};
    put(message, 0, type); put(message, 4, 8);
    put(message, 16, 0x49434653); put(message, 20, 1);
    put(message, 24, command); put(message, 28, 0x12345678);
    return message;
}

IpcMessage domain(unsigned type, unsigned command, unsigned object) {
    IpcMessage message{};
    put(message, 0, type); put(message, 4, 12);
    put(message, 16, 0x00100001); put(message, 20, object);
    put(message, 28, 0xdeadbeef);
    put(message, 32, 0x49434653); put(message, 36, 1);
    put(message, 40, command); put(message, 44, 0xaabbccdd);
    return message;
}

std::array<std::byte, 4096> clock_page() {
    alignas(8) std::array<std::byte, 4096> bytes{};
    armigo::services::shared_clock::Configuration config;
    config.counter_nanoseconds = 8'000'000'000;
    config.steady_offset_nanoseconds = -1'000'000'000;
    config.steady_source_id[0] = std::byte{0x71};
    config.steady_source_id[15] = std::byte{0xb9};
    config.fallback_wall_seconds = 5000;
    CHECK(armigo::services::shared_clock::initialize(bytes, config) ==
          armigo::services::shared_clock::InitializeResult::initialized);
    return bytes;
}

struct Fixture {
    HandleTable handles;
    const std::array<std::byte, 4096> expected_page = clock_page();
    std::shared_ptr<ReadOnlySharedMemory> memory = std::make_shared<ReadOnlySharedMemory>(expected_page);
    std::shared_ptr<TimeServiceSession> session;
    IpcDispatcher dispatcher;
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{701, message, nullptr, nullptr};
    Handle handle;
    std::uint32_t root = 0;

    explicit Fixture(std::size_t capacity = 16)
        : handles(capacity), session(std::make_shared<TimeServiceSession>(701, handles, memory)),
          dispatcher(handles, nullptr), handle(handles.insert(session).value()) {}

    IpcDisposition send() {
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    }

    void convert(unsigned type = 7) {
        message = control(type, 0);
        CHECK(send() == IpcDisposition::handled && registers[0] == 0);
        root = get(message, 32);
        CHECK(root != 0 && session->domain_object_count() == 1);
    }

    void unsupported(IpcMessage input) {
        message = input;
        registers.fill(0x76543210);
        registers[0] = handle;
        const auto before = registers;
        const auto count = session->domain_object_count();
        const auto handle_count = handles.size();
        CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::unsupported);
        CHECK(message == input && registers == before);
        CHECK(session->domain_object_count() == count && handles.size() == handle_count);
        CHECK(session->diagnostic() != nullptr);
    }
};

// GetClockSnapshot (400): the user and network contexts from the page, the
// user time on the steady clock, a zero network time without a network clock,
// UTC calendars, the steady time point and the requested type.
void test_clock_snapshot() {
    HandleTable handles(16);
    const auto page = clock_page();
    auto memory = std::make_shared<ReadOnlySharedMemory>(page);
    TimeServiceSession session(701, handles, memory, [] { return std::optional<std::int64_t>{10'000'000'000}; });
    CHECK(session.request(control(7, 0), 701).disposition == IpcDisposition::handled);
    auto request = domain(6, 400, 1);
    put(request, 4, 12 | (2u << 10));
    put(request, 48, 1, 1);
    struct Output {
        std::array<std::byte, 0xd0> bytes{};
        static bool read(void*, std::uint64_t, std::span<std::byte>) noexcept { return false; }
        static bool write(void* context, std::uint64_t address, std::span<const std::byte> in) noexcept {
            auto& self = *static_cast<Output*>(context);
            if (address + in.size() > self.bytes.size()) return false;
            std::copy(in.begin(), in.end(), self.bytes.begin() + static_cast<std::ptrdiff_t>(address));
            return true;
        }
    } output;
    armigo::horizon::GuestBuffers buffers(Output::read, Output::write, &output);
    buffers.add({0, 0xd0, false, true});
    const auto reply = session.request_with_buffers(request, 701, buffers);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 48 && get(reply.bytes, 40) == 0);
    const auto at64 = [&](std::size_t offset) {
        std::int64_t value = 0;
        std::memcpy(&value, output.bytes.data() + offset, 8);
        return value;
    };
    CHECK(std::equal(page.begin() + 0x40, page.begin() + 0x60, output.bytes.begin()));
    CHECK(at64(0x40) == 5002 && at64(0x48) == 0);
    CHECK(output.bytes[0x50] == std::byte{0xb2} && output.bytes[0x51] == std::byte{0x07});  // 1970
    CHECK(output.bytes[0x52] == std::byte{1} && output.bytes[0x53] == std::byte{1} &&
          output.bytes[0x54] == std::byte{1} && output.bytes[0x55] == std::byte{23} &&
          output.bytes[0x56] == std::byte{22});
    CHECK(output.bytes[0x5c] == std::byte{0} && output.bytes[0x60] == std::byte{4});  // Thursday
    CHECK(at64(0x90) == 9 && std::equal(page.begin() + 0x10, page.begin() + 0x20, output.bytes.begin() + 0x98));
    CHECK(output.bytes[0xa8] == std::byte{'U'} && output.bytes[0xcd] == std::byte{1});
    // A snapshot needs its output buffer and the guest counter.
    CHECK(session.request(request, 701).disposition == IpcDisposition::unsupported);
    TimeServiceSession no_counter(701, handles, memory);
    CHECK(no_counter.request(control(7, 0), 701).disposition == IpcDisposition::handled);
    CHECK(no_counter.request_with_buffers(request, 701, buffers).disposition == IpcDisposition::unsupported);
}

void test_control_and_child_reply() {
    Fixture f;
    for (const unsigned type : {5, 7}) {
        const auto reply = f.session->request(control(type, 3), 701);
        IpcMessage expected{};
        put(expected, 4, 9); put(expected, 16, 0x4f434653);
        put(expected, 28, 0xec6be3ff); put(expected, 32, 0x1000, 2);
        CHECK(reply.disposition == IpcDisposition::handled && reply.transport_result == 0);
        CHECK(reply.size == 44 && reply.bytes == expected);
    }
    f.convert();
    IpcMessage expected_conversion{};
    put(expected_conversion, 4, 9); put(expected_conversion, 16, 0x4f434653);
    put(expected_conversion, 28, 0xec6be3ff); put(expected_conversion, 32, f.root);
    CHECK(std::equal(expected_conversion.begin(), expected_conversion.begin() + 44, f.message.begin()));
    std::uint32_t previous = 0;
    for (const unsigned type : {4, 6}) {
        const auto reply = f.session->request(domain(type, 3, f.root), 701);
        const auto child = get(reply.bytes, 48);
        CHECK(child != 0 && child != f.root && child != previous);
        CHECK(f.session->is_time_zone_object(child) && !f.session->is_time_zone_object(f.root));
        IpcMessage expected{};
        put(expected, 4, 13); put(expected, 16, 1);
        put(expected, 32, 0x4f434653); put(expected, 44, 0x0920b7a6); put(expected, 48, child);
        CHECK(reply.disposition == IpcDisposition::handled && reply.transport_result == 0);
        CHECK(reply.size == 60 && reply.bytes == expected && f.handles.size() == 1);
        f.unsupported(domain(type, 20, child));
        // The child reports the fixed UTC location name.
        auto location = domain(type, 0, child);
        f.message = location;
        f.registers.fill(0);
        f.registers[0] = f.handle;
        CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::handled);
        CHECK(get(f.message, 40) == 0);
        CHECK(f.message[48] == std::byte{'U'} && f.message[49] == std::byte{'T'} &&
              f.message[50] == std::byte{'C'} && f.message[51] == std::byte{0});
        // ToCalendarTimeWithMyRule converts the input POSIX time in UTC.
        auto calendar = domain(type, 101, child);
        put(calendar, 48, 0, 8);
        f.message = calendar;
        f.registers.fill(0);
        f.registers[0] = f.handle;
        CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::handled);
        CHECK(get(f.message, 40) == 0);
        CHECK(get(f.message, 48, 2) == 1970 && f.message[50] == std::byte{1} &&
              f.message[51] == std::byte{1});
        CHECK(f.message[52] == std::byte{0} && f.message[53] == std::byte{0} &&
              f.message[54] == std::byte{0});
        CHECK(get(f.message, 56) == 4 && get(f.message, 60) == 0);
        previous = child;
    }
    CHECK(f.session->domain_object_count() == 3);
    Fixture other;
    other.convert();
    other.unsupported(domain(6, 20, previous));
    f.message = control(7, 3);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 32, 2) == 0x1000);
}

void test_copy_handle_and_lifetime() {
    Fixture f;
    f.convert(5);
    std::weak_ptr<ReadOnlySharedMemory> weak_memory = f.memory;
    std::weak_ptr<TimeServiceSession> weak_session = f.session;
    f.memory.reset();
    std::array<Handle, 2> copies{};
    for (unsigned i = 0; i < copies.size(); ++i) {
        f.message = domain(i == 0 ? 4 : 6, 20, f.root);
        f.message[63] = std::byte{0x63};
        f.message[64] = std::byte{0x64};
        f.message[100] = std::byte{0x7c};
        CHECK(f.send() == IpcDisposition::handled && f.registers[0] == 0);
        copies[i] = get(f.message, 12);
        CHECK(copies[i] != 0 && copies[i] != f.handle);
        IpcMessage expected{};
        put(expected, 4, 0x8000000c); put(expected, 8, 2); put(expected, 12, copies[i]);
        put(expected, 32, 0x4f434653); put(expected, 44, 0x0920b7a6);
        CHECK(std::equal(expected.begin(), expected.begin() + 64, f.message.begin()));
        CHECK(f.message[64] == std::byte{0x64} && f.message[100] == std::byte{0x7c});
        const auto memory = f.handles.lookup_as<ReadOnlySharedMemory>(copies[i]);
        CHECK(memory != nullptr && memory == weak_memory.lock());
        CHECK(std::equal(f.expected_page.begin(), f.expected_page.end(), memory->bytes().begin()));
    }
    CHECK(copies[0] != copies[1] && f.handles.size() == 3);
    f.session.reset();
    CHECK(f.handles.close(f.handle) && weak_session.expired());
    CHECK(!weak_memory.expired());
    CHECK(f.handles.close(copies[0]) && !weak_memory.expired());
    CHECK(f.handles.close(copies[1]) && weak_memory.expired());
}

void test_unsupported_preserves_state() {
    Fixture f;
    f.unsupported(domain(4, 3, 1));
    f.caller.process_id = 702;
    f.unsupported(control(7, 0));
    f.caller.process_id = 701;
    f.unsupported(control(7, 4));
    auto input = control(7, 3); put(input, 4, 7); f.unsupported(input);
    input = control(5, 3); put(input, 16, 0); f.unsupported(input);
    input = control(7, 3); put(input, 4, 0x80000008); put(input, 8, 1); f.unsupported(input);
    f.convert();
    f.unsupported(control(5, 0));
    f.unsupported(domain(6, 0, f.root));
    f.unsupported(domain(6, 20, 0));
    f.unsupported(domain(6, 20, 999));
    input = domain(6, 20, f.root); put(input, 4, 11); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 4, 63); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 16, 0x00100002); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 16, 0x00100101); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 18, 15, 2); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 18, 25, 2); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 32, 0); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 0, 0x10006); f.unsupported(input);
    input = domain(6, 20, f.root); put(input, 4, 0x8000000c); put(input, 8, 1); f.unsupported(input);
    Fixture full(1);
    full.convert();
    full.unsupported(domain(6, 20, full.root));
    CHECK(full.handles.size() == 1);
}

void test_immutable_payload_and_constructor_validation() {
    auto page = clock_page();
    const auto original = page;
    ReadOnlySharedMemory memory(page);
    page.fill(std::byte{0xff});
    CHECK(std::equal(original.begin(), original.end(), memory.bytes().begin()));
    bool rejected = false;
    try { ReadOnlySharedMemory empty({}); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { ReadOnlySharedMemory partial(std::span<const std::byte>(page).first(4095)); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    HandleTable handles(1);
    rejected = false;
    try { TimeServiceSession null_page(701, handles, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    std::array<std::byte, 8192> two_pages{};
    auto generic = std::make_shared<ReadOnlySharedMemory>(two_pages);
    CHECK(generic->bytes().size() == 8192);
    rejected = false;
    try { TimeServiceSession wrong_size(701, handles, generic); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}
}  // namespace

int main() {
    test_control_and_child_reply();
    test_clock_snapshot();
    test_copy_handle_and_lifetime();
    test_unsupported_preserves_state();
    test_immutable_payload_and_constructor_validation();
    if (failures) return 1;
    std::puts("time-session checks passed");
    return 0;
}
