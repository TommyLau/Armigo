// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/log_manager.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <thread>

namespace {
using namespace armigo::horizon;
using armigo::services::lm::LogManagerSession;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (false)

void put(IpcMessage& message, std::size_t offset, std::uint64_t value, unsigned size = 4) {
    for (unsigned i = 0; i < size; ++i)
        message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
std::uint64_t get(const IpcMessage& message, std::size_t offset, unsigned size = 4) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < size; ++i)
        value |= std::uint64_t{std::to_integer<unsigned char>(message[offset + i])} << (8 * i);
    return value;
}
IpcMessage control(std::uint32_t command) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, 5);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 20, 0);
    put(message, 24, command);
    put(message, 28, 0);
    return message;
}
IpcMessage open_logger(std::uint32_t root) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, 4);
    put(message, 4, 0x8000000e);
    put(message, 8, 1);
    put(message, 12, 999, 8); // Forged HIPC placeholder; transport PID is 71.
    put(message, 32, 0x00180001); // Domain type1, zero objects, data size24.
    put(message, 36, root);
    put(message, 40, 0, 8);
    put(message, 48, 0x49434653);
    put(message, 52, 0);
    put(message, 56, 0);
    put(message, 60, 0);
    put(message, 64, 123456, 8); // Independent forged inline PID placeholder.
    return message;
}

IpcMessage log_request(std::uint32_t logger, std::uint32_t data_size = 16) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, 6);              // HIPC type6 request, length halfword 0.
    put(message, 2, 0x11, 2);        // The wire length halfword is not the type.
    put(message, 4, 12);             // Twelve raw words, no descriptors.
    put(message, 8, 0x00bc1000);     // Input buffer extent and address.
    put(message, 12, 0x08b625f8);
    put(message, 32, 1 | (data_size << 16)); // Domain type1, zero objects.
    put(message, 36, logger);
    put(message, 40, 0, 8);
    put(message, 48, 0x49434653);
    put(message, 52, 1);
    put(message, 56, 0);
    put(message, 60, 0);
    return message;
}

struct Fixture {
    HandleTable handles{8};
    std::shared_ptr<LogManagerSession> session = std::make_shared<LogManagerSession>(71);
    IpcDispatcher dispatcher{handles, nullptr};
    Handle handle = handles.insert(session).value();
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};

    IpcDisposition send() {
        registers.fill(0x123456789abcdef0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    }
    void expect_unsupported() {
        const auto before = message;
        registers.fill(0x123456789abcdef0);
        registers[0] = handle;
        const auto saved_registers = registers;
        const auto count = session->domain_object_count();
        CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::unsupported);
        CHECK(registers == saved_registers && message == before);
        CHECK(session->domain_object_count() == count && session->diagnostic() != nullptr);
    }
    std::uint32_t convert() {
        message = control(0);
        CHECK(send() == IpcDisposition::handled);
        const auto id = static_cast<std::uint32_t>(get(message, 32));
        CHECK(id != 0 && session->domain_object_count() == 1);
        return id;
    }
};

void check_control_reply(const IpcMessage& actual, std::uint32_t output, unsigned width) {
    IpcMessage expected{};
    put(expected, 4, 9);
    put(expected, 16, 0x4f434653);
    put(expected, 28, 0xec6be3ff);
    put(expected, 32, output, width);
    CHECK(std::equal(expected.begin(), expected.begin() + 44, actual.begin()));
    CHECK(std::all_of(actual.begin() + 44, actual.end(), [](std::byte b) { return b == std::byte{0xa5}; }));
}

void test_startup() {
    Fixture f;
    CHECK(f.handles.lookup_as<LogManagerSession>(f.handle) == f.session);
    f.message = control(3);
    put(f.message, 28, 0x11223344); // Firmware response is not a token echo.
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(std::all_of(f.registers.begin(), f.registers.end(), [](auto v) { return v == 0; }));
    check_control_reply(f.message, 0x400, 2);
    CHECK(f.session->domain_object_count() == 0);
    const auto root = f.convert();
    check_control_reply(f.message, root, 4);
    CHECK(!f.session->logger_process_id(root));
    f.message = open_logger(root);
    CHECK(f.send() == IpcDisposition::handled);
    const auto logger = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(logger != 0 && logger != root);
    CHECK(f.session->logger_process_id(logger) == 71);
    CHECK(f.session->domain_object_count() == 2 && f.handles.size() == 1);
    IpcMessage expected{};
    put(expected, 4, 13);
    put(expected, 16, 1);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0xe1d7f748);
    put(expected, 48, logger);
    CHECK(std::equal(expected.begin(), expected.begin() + 60, f.message.begin()));
    // The dispatcher copies only the reply's admitted wire extent.
    CHECK(get(f.message, 64, 8) == 123456);
    CHECK(f.session->diagnostic() == nullptr);
    f.message = open_logger(root);
    CHECK(f.send() == IpcDisposition::handled);
    const auto second = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(second != logger && second != root && f.session->logger_process_id(second) == 71);
    CHECK(f.session->domain_object_count() == 3 && f.handles.size() == 1);
    f.message = control(3);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0x400, 2);
    CHECK(f.session->domain_object_count() == 3);

    std::weak_ptr<LogManagerSession> lifetime = f.session;
    f.session.reset();
    CHECK(!lifetime.expired());
    CHECK(f.handles.close(f.handle));
    CHECK(lifetime.expired());
}

void check_log_reply(const IpcMessage& actual) {
    IpcMessage expected{};
    put(expected, 4, 10);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0xe1d7f748);
    CHECK(std::equal(expected.begin(), expected.begin() + 48, actual.begin()));
    // The request still holds SFCI at byte48; only the admitted reply extent
    // is overwritten, so the untouched tail begins after the request payload.
    CHECK(std::all_of(actual.begin() + 64, actual.end(), [](std::byte b) { return b == std::byte{0xa5}; }));
}

void test_logger_log_command() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_logger(root);
    CHECK(f.send() == IpcDisposition::handled);
    const auto logger = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(f.session->logger_process_id(logger) == 71);
    f.message = log_request(logger);
    CHECK(f.send() == IpcDisposition::handled);
    check_log_reply(f.message);
    CHECK(std::all_of(f.registers.begin(), f.registers.end(), [](auto v) { return v == 0; }));
    CHECK(f.session->domain_object_count() == 2 && f.session->diagnostic() == nullptr);
    // Validation still applies: no oversized payload, foreign sender, input
    // objects, unknown command or missing magic reaches a success reply.
    f.message = log_request(logger, 24);
    f.expect_unsupported();
    f.message = log_request(logger);
    f.caller.process_id = 72;
    f.expect_unsupported();
    f.caller.process_id = 71;
    f.message = log_request(logger);
    put(f.message, 33, 1);
    f.expect_unsupported();
    f.message = log_request(logger);
    put(f.message, 56, 1);
    f.expect_unsupported();
    f.message = log_request(logger);
    put(f.message, 48, 0);
    f.expect_unsupported();
    f.message = log_request(logger);
    put(f.message, 32, 2, 1); // Domain request type2 is a reply, not a request.
    f.expect_unsupported();
    f.message = log_request(0xffffffffu);
    f.expect_unsupported();
    f.message = log_request(root);
    f.expect_unsupported();
    f.message = log_request(logger);
    put(f.message, 4, 12 | 0x00000400u); // A handle descriptor is refused.
    f.expect_unsupported();
}

void test_context_wire_types() {
    Fixture f;
    f.message = control(3);
    put(f.message, 0, 7);
    put(f.message, 20, 1); // Request version1; response remains version0.
    put(f.message, 28, 0x12345678); // Context is not echoed in the reply.
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0x400, 2);
    CHECK(f.session->domain_object_count() == 0);

    f.message = control(0);
    put(f.message, 0, 7);
    put(f.message, 20, 1);
    put(f.message, 28, 0x98765432);
    CHECK(f.send() == IpcDisposition::handled);
    const auto root = static_cast<std::uint32_t>(get(f.message, 32));
    CHECK(root != 0 && f.session->domain_object_count() == 1);
    check_control_reply(f.message, root, 4);

    f.message = open_logger(root);
    put(f.message, 0, 6);
    put(f.message, 44, 0xabcdef01); // Domain request context.
    put(f.message, 52, 1); // Nested SFCI version1.
    put(f.message, 60, 0x11223344); // Distinct nested header word.
    CHECK(f.send() == IpcDisposition::handled);
    const auto logger = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(logger != 0 && logger != root && f.session->logger_process_id(logger) == 71);
    CHECK(f.session->domain_object_count() == 2 && f.handles.size() == 1);
    IpcMessage expected{};
    put(expected, 4, 13);
    put(expected, 16, 1);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0xe1d7f748);
    put(expected, 48, logger);
    CHECK(std::equal(expected.begin(), expected.begin() + 60, f.message.begin()));
    CHECK(std::all_of(f.registers.begin(), f.registers.end(), [](auto v) { return v == 0; }));
    // Wire type, not the input version word, chooses the route.
    f.message = control(3);
    put(f.message, 20, 1);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0x400, 2);
    f.message = control(3);
    put(f.message, 0, 7);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0x400, 2);
}

void test_unsupported_is_not_a_reply() {
    Fixture f;
    f.message = open_logger(1); // Conversion has not happened.
    f.expect_unsupported();
    f.message = control(3);
    f.caller.process_id = 72;
    f.expect_unsupported();
    f.caller.process_id = 71;
    f.message = control(3);
    put(f.message, 16, 0);
    f.expect_unsupported();
    f.message = control(3);
    put(f.message, 4, 7); // Too little HIPC raw data.
    f.expect_unsupported();
    f.message = control(4); // Not QueryPointerBufferSize.
    f.expect_unsupported();
    const auto root = f.convert();
    f.message = control(0);
    f.expect_unsupported();
    for (const auto id : {0u, 0xffffffffu}) {
        f.message = open_logger(id);
        f.expect_unsupported();
    }
    f.message = open_logger(root);
    put(f.message, 8, 0); // Missing PID descriptor.
    f.expect_unsupported();
    f.message = open_logger(root);
    put(f.message, 8, 3); // A copy handle was added.
    f.expect_unsupported();
    f.message = open_logger(root);
    put(f.message, 0, 0x10004); // A pointer descriptor was added.
    f.expect_unsupported();
    f.message = open_logger(root);
    put(f.message, 34, 23, 2);
    f.expect_unsupported();
    f.message = open_logger(root);
    put(f.message, 34, 0xffff, 2);
    f.expect_unsupported();
    f.message = open_logger(root);
    put(f.message, 32, 2, 1); // Domain close reply framing is not implemented.
    f.expect_unsupported();
    f.message = open_logger(root);
    put(f.message, 56, 1);
    f.expect_unsupported();
    f.message = open_logger(root);
    CHECK(f.send() == IpcDisposition::handled);
    f.message = open_logger(static_cast<std::uint32_t>(get(f.message, 48)));
    f.expect_unsupported(); // OpenLogger framing never reaches a logger object.
}

void test_serialized_object_creation() {
    Fixture f;
    const auto root = f.convert();
    std::array<std::uint32_t, 2> ids{};
    std::array<IpcDisposition, 2> dispositions{};
    const auto run = [&](unsigned index) {
        auto message = open_logger(root);
        QueryRegisters registers{};
        registers[0] = f.handle;
        const IpcCallContext caller{71, message, nullptr, nullptr};
        dispositions[index] = f.dispatcher.dispatch(0x21, registers, caller);
        ids[index] = static_cast<std::uint32_t>(get(message, 48));
    };
    std::thread first(run, 0), second(run, 1);
    first.join(); second.join();
    CHECK(dispositions[0] == IpcDisposition::handled && dispositions[1] == IpcDisposition::handled);
    CHECK(ids[0] != ids[1] && ids[0] != root && ids[1] != root);
    CHECK(f.session->logger_process_id(ids[0]) == 71 && f.session->logger_process_id(ids[1]) == 71);
    CHECK(f.session->domain_object_count() == 3 && f.handles.size() == 1);
}
} // namespace

int main() {
    test_startup();
    test_logger_log_command();
    test_context_wire_types();
    test_unsupported_is_not_a_reply();
    test_serialized_object_creation();
    std::printf("Log-manager failures: %u\n", failures);
    return failures ? 1 : 0;
}
