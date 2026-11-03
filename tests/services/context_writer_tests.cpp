// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/context_writer.h>

#include <algorithm>
#include <cstdio>

namespace {
using namespace armigo::horizon;
using armigo::services::err::ContextWriterSession;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (false)
void put(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
IpcMessage control(unsigned type, unsigned command) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, type);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    put(message, 28, 0x98765432);
    return message;
}
void verify_reply(const IpcMessage& message, unsigned value) {
    IpcMessage expected{};
    put(expected, 4, 9);
    put(expected, 16, 0x4f434653);
    put(expected, 28, 0xec6be3ff);
    put(expected, 32, value);
    CHECK(std::equal(expected.begin(), expected.begin() + 44, message.begin()));
    CHECK(std::all_of(message.begin() + 44, message.end(),
                      [](std::byte value) { return value == std::byte{0xa5}; }));
}
void test_session(unsigned type) {
    HandleTable handles{4};
    auto session = std::make_shared<ContextWriterSession>(71);
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
    message = control(type, 3);
    CHECK(send() == IpcDisposition::handled);
    verify_reply(message, 0);
    CHECK(std::all_of(registers.begin(), registers.end(), [](auto value) { return value == 0; }));
    message = control(type, 0);
    CHECK(send() == IpcDisposition::handled);
    verify_reply(message, 1);
    message = control(type, 0);
    reject();
    message = control(type, 3);
    CHECK(send() == IpcDisposition::handled);
    verify_reply(message, 0);
    CHECK(session->diagnostic() == nullptr && handles.size() == 1);
    for (auto unsupported_type : {4u, 6u}) {
        message = control(unsupported_type, 0);
        reject(); // Context registration is not implemented by a control endpoint.
    }
    message = control(type, 1);
    reject();
    message = control(type, 3);
    put(message, 4, 7);
    reject();
    message = control(type, 3);
    put(message, 4, 0x80000008);
    reject();
    message = control(type, 3);
    put(message, 0, type | 0x10000);
    reject();
    message = control(type, 3);
    caller.process_id = 72;
    reject();
    std::weak_ptr<ContextWriterSession> lifetime = session;
    session.reset();
    CHECK(!lifetime.expired());
    CHECK(handles.close(handle));
    CHECK(lifetime.expired());
}
} // namespace

int main() {
    test_session(5);
    test_session(7);
    std::printf("Error-context control failures: %u\n", failures);
    return failures ? 1 : 0;
}
