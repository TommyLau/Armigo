// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/parental_control.h>

#include <algorithm>
#include <cstdio>

namespace {
using namespace armigo::horizon;
using armigo::services::pctl::PctlSession;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (false)
void put(IpcMessage& bytes, unsigned offset, std::uint64_t value, unsigned width = 4) {
    for (unsigned i = 0; i != width; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
std::uint32_t get(const IpcMessage& bytes, unsigned offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
    return value;
}
IpcMessage control(unsigned type, unsigned command) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, type);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    put(message, 28, 0x12345678);
    return message;
}
IpcMessage call(unsigned type, std::uint32_t id, bool pid) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, type);
    put(message, 4, pid ? 0x8000000e : 12);
    if (pid) {
        put(message, 8, 1);
        put(message, 12, 999, 8); // Untrusted transport placeholder.
    }
    const unsigned domain = pid ? 32 : 16;
    put(message, domain, pid ? 0x00180001 : 0x00100001);
    put(message, domain + 4, id);
    put(message, domain + 8, 0);
    put(message, domain + 12, 0x87654321);
    put(message, domain + 16, 0x49434653);
    put(message, domain + 20, 1);
    put(message, domain + 24, 1);
    put(message, domain + 28, 0x11223344);
    if (pid) put(message, domain + 32, 123456, 8);
    return message;
}
void expected_reply(const IpcMessage& actual, unsigned size, std::uint32_t words,
                    unsigned cmif, std::uint32_t tag, std::uint32_t objects,
                    std::uint32_t value = 0) {
    IpcMessage expected{};
    put(expected, 4, words);
    if (cmif == 32) put(expected, 16, objects);
    put(expected, cmif, 0x4f434653);
    put(expected, cmif + 12, tag);
    if (cmif == 16 || objects) put(expected, cmif + 16, value);
    CHECK(std::equal(expected.begin(), expected.begin() + size, actual.begin()));
    if (!objects) CHECK(std::all_of(actual.begin() + size, actual.end(),
                      [](std::byte value) { return value == std::byte{0xa5}; }));
}
void test_startup(unsigned control_type, unsigned request_type) {
    HandleTable handles{4};
    auto session = std::make_shared<PctlSession>(71);
    const auto handle = handles.insert(session).value();
    IpcDispatcher dispatcher{handles, nullptr};
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};
    const auto send = [&] {
        registers.fill(0x123456789abcdef0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    };
    const auto reject = [&] {
        const auto before = message;
        registers.fill(0x123456789abcdef0);
        registers[0] = handle;
        const auto saved = registers;
        CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::unsupported);
        CHECK(message == before && registers == saved && session->diagnostic() != nullptr);
    };
    message = call(request_type, 1, true);
    reject();
    message = control(control_type, 3);
    CHECK(send() == IpcDisposition::handled);
    expected_reply(message, 44, 9, 16, 0xec6be3ff, 0);
    message = control(control_type, 0);
    CHECK(send() == IpcDisposition::handled);
    const auto root = get(message, 32);
    CHECK(root != 0);
    expected_reply(message, 44, 9, 16, 0xec6be3ff, 0, root);
    message = control(control_type, 0);
    reject();
    message = call(request_type, root, true);
    const auto factory_before = message;
    CHECK(send() == IpcDisposition::handled);
    const auto child = get(message, 48);
    CHECK(child != 0 && child != root);
    CHECK(std::equal(factory_before.begin() + 60, factory_before.end(), message.begin() + 60));
    expected_reply(message, 60, 13, 32, 0x64f05bf6, 1, child);
    CHECK(handles.size() == 1);
    for (unsigned repeat = 0; repeat != 2; ++repeat) {
        message = call(request_type, child, false);
        CHECK(send() == IpcDisposition::handled);
        expected_reply(message, 56, 12, 32, 0x81a6fff1, 0);
        CHECK(std::all_of(registers.begin(), registers.end(), [](auto value) { return value == 0; }));
    }
    message = call(request_type, root, false);
    reject();
    message = call(request_type, child, true);
    reject();
    message = call(request_type, child, false);
    put(message, 40, 1001); // Initialization does not grant communication permission.
    reject();
    message = call(request_type, root, true);
    put(message, 8, 3); // Unexpected copied handle.
    reject();
    message = call(request_type, root, true);
    put(message, 34, 23, 2);
    reject();
    message = call(request_type, root, true);
    caller.process_id = 72;
    reject();
    caller.process_id = 71;
    message = call(request_type, root, true);
    CHECK(send() == IpcDisposition::handled);
    CHECK(get(message, 48) != child && handles.size() == 1);
    std::weak_ptr<PctlSession> lifetime = session;
    session.reset();
    CHECK(!lifetime.expired());
    CHECK(handles.close(handle));
    CHECK(lifetime.expired());
}
} // namespace

int main() {
    test_startup(5, 4);
    test_startup(7, 6);
    std::printf("Parental-control startup failures: %u\n", failures);
    return failures ? 1 : 0;
}
