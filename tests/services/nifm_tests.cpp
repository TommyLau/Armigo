// ============================================================================
// Armigo Tests - Network Interface Management Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/nifm.h>
#include <armigo/horizon/event.h>
#include <armigo/horizon/handle_table.h>

#include <array>
#include <cstdio>
#include <span>

namespace {
using namespace armigo::horizon;
using namespace armigo::services::nifm;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)
constexpr std::uint64_t pid = 41;

void put(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
std::uint32_t word(const IpcMessage& bytes, unsigned offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}
// A non-domain request with `inputs` raw input words after the CMIF header.
IpcMessage command(std::uint32_t id, std::span<const std::uint32_t> inputs = {}) {
    IpcMessage bytes{};
    put(bytes, 0, 4);
    put(bytes, 4, 8 + static_cast<std::uint32_t>(inputs.size()));
    put(bytes, 16, 0x49434653);
    put(bytes, 20, 1);
    put(bytes, 24, id);
    for (std::size_t i = 0; i < inputs.size(); ++i) put(bytes, 32 + 4 * static_cast<unsigned>(i), inputs[i]);
    return bytes;
}
// A domain request to `object`: domain header at 16, CMIF at 32.
IpcMessage domain_command(std::uint32_t object, std::uint32_t id, std::span<const std::uint32_t> inputs = {}) {
    IpcMessage bytes{};
    put(bytes, 0, 4);
    put(bytes, 4, 12 + static_cast<std::uint32_t>(inputs.size()));
    put(bytes, 16, 1 | (16 + 4 * static_cast<std::uint32_t>(inputs.size())) << 16);
    put(bytes, 20, object);
    put(bytes, 32, 0x49434653);
    put(bytes, 36, 1);
    put(bytes, 40, id);
    for (std::size_t i = 0; i < inputs.size(); ++i) put(bytes, 48 + 4 * static_cast<unsigned>(i), inputs[i]);
    return bytes;
}
IpcMessage control(std::uint32_t id) {
    IpcMessage bytes{};
    put(bytes, 0, 5);
    put(bytes, 4, 8);
    put(bytes, 16, 0x49434653);
    put(bytes, 24, id);
    return bytes;
}
struct Buffer {
    std::array<std::byte, 16> bytes{};
    static bool read(void*, std::uint64_t, std::span<std::byte>) noexcept { return false; }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> value) noexcept {
        auto& self = *static_cast<Buffer*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        for (std::size_t i = 0; i < value.size(); ++i) self.bytes[address + i] = value[i];
        return true;
    }
};
const std::array<std::uint32_t, 2> reserved{0, 0};
const std::array<std::uint32_t, 1> preset{2};

void test_unconverted_sessions() {
    Session root(pid);
    auto size = root.request(control(3), pid);
    CHECK(size.disposition == IpcDisposition::handled && word(size.bytes, 32) == 0x1000);
    // CreateGeneralService (5) and CreateGeneralServiceOld (4) each hand out a
    // general service session with its own client id.
    auto created = root.request(command(5, reserved), pid);
    CHECK(created.disposition == IpcDisposition::handled && created.transferred_count == 1 &&
          !created.transferred_as_copy);
    auto general = std::dynamic_pointer_cast<Session>(created.transferred[0]);
    auto old = std::dynamic_pointer_cast<Session>(root.request(command(4), pid).transferred[0]);
    CHECK(general != nullptr && old != nullptr);
    if (!general || !old) return;
    CHECK(general->object()->kind == Object::Kind::general_service &&
          general->object()->client_id != old->object()->client_id);
    CHECK(root.request(command(5, reserved), pid + 1).disposition == IpcDisposition::unsupported);
    CHECK(root.request(command(6), pid).disposition == IpcDisposition::unsupported);

    // GetClientId writes the id to its output buffer.
    Buffer buffer;
    GuestBuffers out(Buffer::read, Buffer::write, &buffer);
    out.add({0, 4, false, true});
    CHECK(general->request_with_buffers(command(1), pid, out).disposition == IpcDisposition::handled);
    CHECK(std::to_integer<std::uint32_t>(buffer.bytes[0]) == general->object()->client_id);
    CHECK(general->request(command(1), pid).disposition == IpcDisposition::unsupported);

    // CreateRequest(preset) hands out a free request with two events.
    auto requested = general->request(command(4, preset), pid);
    CHECK(requested.disposition == IpcDisposition::handled && requested.transferred_count == 1);
    auto request = std::dynamic_pointer_cast<Session>(requested.transferred[0]);
    CHECK(request != nullptr && request->object()->requirement_preset == 2);
    if (!request) return;
    auto state = request->request(command(0), pid);
    CHECK(state.disposition == IpcDisposition::handled && word(state.bytes, 32) == 1);
    auto events = request->request(command(2), pid);
    CHECK(events.disposition == IpcDisposition::handled && events.transferred_count == 2 &&
          events.transferred_as_copy && events.transferred[0] != events.transferred[1]);
    const auto state_event = std::dynamic_pointer_cast<Event>(events.transferred[0]);
    const auto other_event = std::dynamic_pointer_cast<Event>(events.transferred[1]);
    CHECK(state_event && other_event && !state_event->is_signaled());
    // No result and no applet before a submission.
    CHECK(request->request(command(1), pid).disposition == IpcDisposition::unsupported);
    const std::array<std::uint32_t, 1> theme{0x12345678};
    CHECK(request->request(command(21, theme), pid).disposition == IpcDisposition::unsupported);
    // Submit fails at once with no candidate: free again, the first event signaled.
    auto submitted = request->request(command(4), pid);
    CHECK(submitted.disposition == IpcDisposition::handled && word(submitted.bytes, 24) == 0);
    CHECK(state_event && state_event->is_signaled() && other_event && !other_event->is_signaled());
    state = request->request(command(0), pid);
    CHECK(word(state.bytes, 32) == 1);
    auto result = request->request(command(1), pid);
    CHECK(result.disposition == IpcDisposition::handled && word(result.bytes, 16) == 0x4f434653 &&
          word(result.bytes, 24) == 0x8986e);
    // GetAppletInfo names the network-connection applet, mode 0, and its records.
    struct Records {
        std::array<std::byte, 0x100> bytes{};
        static bool read(void*, std::uint64_t, std::span<std::byte>) noexcept { return false; }
        static bool write(void* context, std::uint64_t address, std::span<const std::byte> value) noexcept {
            auto& self = *static_cast<Records*>(context);
            if (address + value.size() > self.bytes.size()) return false;
            for (std::size_t i = 0; i < value.size(); ++i) self.bytes[address + i] = value[i];
            return true;
        }
    } records;
    GuestBuffers info_buffer(Records::read, Records::write, &records);
    info_buffer.add({0, 0x100, false, true});
    auto info = request->request_with_buffers(command(21, theme), pid, info_buffer);
    CHECK(info.disposition == IpcDisposition::handled && word(info.bytes, 24) == 0);
    CHECK(word(info.bytes, 32) == 15 && word(info.bytes, 36) == 0 && word(info.bytes, 40) == 0x30);
    const auto record = [&](unsigned offset) {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) value |= std::to_integer<std::uint32_t>(records.bytes[offset + i]) << (8 * i);
        return value;
    };
    CHECK(record(0) == 0x20 && record(4) == 1 && record(8) == 0x20 && record(16) == 0x12345678);
    CHECK(record(36) == 4 && record(40) == 3 && record(44) == 0xffffffff);
}

void test_domain() {
    Session root(pid);
    auto converted = root.request(control(0), pid);
    CHECK(converted.disposition == IpcDisposition::handled && word(converted.bytes, 32) == 1);
    CHECK(root.request(control(0), pid).disposition == IpcDisposition::unsupported);
    // Objects come back as domain object ids after the CMIF header.
    auto general = root.request(domain_command(1, 5, reserved), pid);
    CHECK(general.disposition == IpcDisposition::handled && general.transferred_count == 0 &&
          word(general.bytes, 16) == 1 && word(general.bytes, 32) == 0x4f434653);
    const auto general_id = word(general.bytes, 48);
    CHECK(root.object(general_id) && root.object(general_id)->kind == Object::Kind::general_service);
    auto request = root.request(domain_command(general_id, 4, preset), pid);
    CHECK(request.disposition == IpcDisposition::handled && word(request.bytes, 16) == 1);
    const auto request_id = word(request.bytes, 48);
    CHECK(root.object(request_id) && root.object(request_id)->requirement_preset == 2);
    auto state = root.request(domain_command(request_id, 0), pid);
    CHECK(state.disposition == IpcDisposition::handled && word(state.bytes, 48) == 1);
    auto events = root.request(domain_command(request_id, 2), pid);
    CHECK(events.disposition == IpcDisposition::handled && events.transferred_count == 2 &&
          events.transferred_as_copy && word(events.bytes, 32) == 0x4f434653);
    CHECK(root.request(domain_command(99, 0), pid).disposition == IpcDisposition::unsupported);
    // A domain close releases a handed-out object, never the port object.
    IpcMessage close{};
    put(close, 0, 4);
    put(close, 4, 8);
    put(close, 16, 2);
    put(close, 20, request_id);
    CHECK(root.request(close, pid).disposition == IpcDisposition::handled && root.domain_object_count() == 2);
    CHECK(root.request(close, pid).disposition == IpcDisposition::unsupported);
    put(close, 20, 1);
    CHECK(root.request(close, pid).disposition == IpcDisposition::unsupported);
}

// The dispatcher hands both readable events to the caller in the copy-handle array.
void test_two_copied_events() {
    HandleTable handles{16};
    auto root = std::make_shared<Session>(pid);
    const auto handle = handles.insert(root).value();
    IpcDispatcher dispatcher{handles, nullptr};
    QueryRegisters registers{};
    auto request_message = [&](IpcMessage message) {
        registers[0] = handle;
        IpcCallContext caller{pid, message, nullptr, nullptr};
        CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled && registers[0] == 0);
        return message;
    };
    const auto general = request_message(command(5, reserved));
    const auto general_handle = static_cast<Handle>(word(general, 20));
    CHECK(handles.lookup_as<Session>(general_handle) != nullptr);
    auto message = command(4, preset);
    registers[0] = general_handle;
    IpcCallContext caller{pid, message, nullptr, nullptr};
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    const auto request_handle = static_cast<Handle>(word(message, 20));
    message = command(2);
    registers[0] = request_handle;
    caller.message = message;
    IpcCallContext events_caller{pid, message, nullptr, nullptr};
    CHECK(dispatcher.dispatch(0x21, registers, events_caller) == IpcDisposition::handled && registers[0] == 0);
    CHECK((word(message, 4) >> 31) == 1);
    CHECK(((word(message, 8) >> 1) & 0xf) == 2 && ((word(message, 8) >> 5) & 0xf) == 0);
    // The reply's special header precedes a zero PID word pair; handles follow at byte 20.
    const auto first = handles.lookup_as<Event>(static_cast<Handle>(word(message, 20)));
    const auto second = handles.lookup_as<Event>(static_cast<Handle>(word(message, 24)));
    CHECK(first != nullptr && second != nullptr && first != second);
    CHECK(word(message, 32) == 0x4f434653 && word(message, 40) == 0);
}
} // namespace

int main() {
    test_unconverted_sessions();
    test_domain();
    test_two_copied_events();
    if (failures) return 1;
    std::puts("Network interface service checks passed");
    return 0;
}
