// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/performance.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <thread>

namespace {
using namespace armigo::horizon;
using armigo::services::apm::PerformanceSession;
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
IpcMessage plain_open_session() {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, 6);          // HIPC type6, length halfword 0.
    put(message, 4, 8);          // Eight raw words, no descriptors.
    put(message, 8, 2);          // Alignment padding the SDK leaves stale.
    put(message, 12, 0x88010);
    put(message, 16, 0x49434653);
    put(message, 20, 1);         // Input version1.
    put(message, 24, 0);         // Command0 OpenSession.
    put(message, 28, 0);
    return message;
}
IpcMessage open_session(std::uint32_t root) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, 4);
    put(message, 4, 12);
    put(message, 16, 0x00100001); // Domain type1, zero objects, data size16.
    put(message, 20, root);
    put(message, 24, 0, 8);
    put(message, 32, 0x49434653);
    put(message, 36, 0);
    put(message, 40, 0);
    put(message, 44, 0);
    return message;
}

struct Fixture {
    HandleTable handles{8};
    std::shared_ptr<PerformanceSession> session;
    IpcDispatcher dispatcher{handles, nullptr};
    Handle handle;
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};

    explicit Fixture(std::uint16_t capacity = 0,
                     armigo::services::OperationMode mode = armigo::services::OperationMode::Console)
        : session(std::make_shared<PerformanceSession>(71, capacity, PerformanceSession::Role::root, mode)),
          handle(handles.insert(session).value()) {}

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
    CHECK(f.handles.lookup_as<PerformanceSession>(f.handle) == f.session);
    f.message = control(3);
    put(f.message, 28, 0x11223344); // Firmware response is not a token echo.
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(std::all_of(f.registers.begin(), f.registers.end(), [](auto v) { return v == 0; }));
    check_control_reply(f.message, 0, 2);
    CHECK(f.session->domain_object_count() == 0);
    const auto root = f.convert();
    check_control_reply(f.message, root, 4);
    f.message = open_session(root);
    CHECK(f.send() == IpcDisposition::handled);
    const auto child = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(child != 0 && child != root);
    CHECK(f.session->domain_object_count() == 2 && f.handles.size() == 1);
    IpcMessage expected{};
    put(expected, 4, 13);
    put(expected, 16, 1);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0xbc2665da);
    put(expected, 48, child);
    CHECK(std::equal(expected.begin(), expected.begin() + 60, f.message.begin()));
    // The dispatcher copies only the reply's admitted wire extent.
    CHECK(std::all_of(f.message.begin() + 60, f.message.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
    CHECK(f.session->diagnostic() == nullptr);
    f.message = open_session(root);
    CHECK(f.send() == IpcDisposition::handled);
    const auto second = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(second != child && second != root);
    CHECK(f.session->domain_object_count() == 3 && f.handles.size() == 1);
    f.message = control(3);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0, 2);
    CHECK(f.session->domain_object_count() == 3);

    std::weak_ptr<PerformanceSession> lifetime = f.session;
    f.session.reset();
    CHECK(!lifetime.expired());
    CHECK(f.handles.close(f.handle));
    CHECK(lifetime.expired());
}

void test_plain_open_session() {
    Fixture f;
    f.message = plain_open_session();
    CHECK(f.send() == IpcDisposition::handled);
    // One moved handle: special header at byte8, handle at byte20, CMIF at 32.
    const auto child_handle = static_cast<std::uint32_t>(get(f.message, 20));
    CHECK(child_handle != 0 && child_handle != f.handle);
    CHECK(get(f.message, 0) == 0 && get(f.message, 4) == 0x80000004u);
    CHECK(get(f.message, 8) == 0x21u && get(f.message, 12) == 0);
    CHECK(get(f.message, 32) == 0x4f434653u && get(f.message, 36) == 0 &&
          get(f.message, 40) == 0 && get(f.message, 44) == 0xbc2665da);
    CHECK(f.handles.lookup_as<PerformanceSession>(child_handle) != nullptr);
    CHECK(f.handles.lookup_as<PerformanceSession>(child_handle) != f.session);
    CHECK(f.session->domain_object_count() == 0 && f.session->diagnostic() == nullptr);
    CHECK(f.handles.size() == 2);
    // The opened session accepts command0 SetPerformanceConfiguration and
    // records the requested mode and configuration.
    IpcMessage set;
    set.fill(std::byte{0xa5});
    put(set, 0, 6);
    put(set, 4, 10);
    put(set, 16, 0x49434653);
    put(set, 20, 1);
    put(set, 24, 0);
    put(set, 28, 0);
    put(set, 32, 1);
    put(set, 36, 0x10000);
    f.message = set;
    f.registers.fill(0);
    f.registers[0] = child_handle;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::handled);
    CHECK(get(f.message, 16) == 0x4f434653u && get(f.message, 24) == 0);
    {
        const auto child = f.handles.lookup_as<PerformanceSession>(child_handle);
        CHECK(child != nullptr);
        if (child) {
            const auto requested = child->requested_configuration();
            CHECK(requested.set && requested.mode == 1 && requested.configuration == 0x10000);
        }
    }
    // Any other session command stays explicitly unsupported.
    IpcMessage other = set;
    put(other, 24, 1);
    f.message = other;
    f.registers.fill(0);
    f.registers[0] = child_handle;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::unsupported);
    // The opened session reports its own unsupported commands and does not
    // reopen itself.
    f.message = plain_open_session();
    f.registers.fill(0);
    f.registers[0] = child_handle;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::unsupported);
    CHECK(f.handles.close(child_handle));
    // The plain path validates its exact shape.
    f.message = plain_open_session();
    put(f.message, 20, 2);
    f.expect_unsupported();
    f.message = plain_open_session();
    put(f.message, 0, 0x00010006); // A pointer descriptor is not part of this variant.
    f.expect_unsupported();
    f.message = plain_open_session();
    put(f.message, 16, 0);
    f.expect_unsupported();
    f.message = plain_open_session();
    put(f.message, 24, 99);
    f.expect_unsupported();
}

void test_provisioned_capacity() {
    Fixture f{0x80};
    f.message = control(3);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0x80, 2);
}

void test_context_wire_types() {
    Fixture f;
    f.message = control(3);
    put(f.message, 0, 7);
    put(f.message, 20, 1); // Request version1; response remains version0.
    put(f.message, 28, 0x12345678); // Context is not echoed in the reply.
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0, 2);
    CHECK(f.session->domain_object_count() == 0);

    f.message = control(0);
    put(f.message, 0, 7);
    put(f.message, 20, 1);
    put(f.message, 28, 0x98765432);
    CHECK(f.send() == IpcDisposition::handled);
    const auto root = static_cast<std::uint32_t>(get(f.message, 32));
    CHECK(root != 0 && f.session->domain_object_count() == 1);
    check_control_reply(f.message, root, 4);

    f.message = open_session(root);
    put(f.message, 0, 6);
    put(f.message, 28, 0xabcdef01); // Domain request context.
    put(f.message, 36, 1); // Nested SFCI version1.
    put(f.message, 44, 0x11223344); // Distinct nested header word.
    CHECK(f.send() == IpcDisposition::handled);
    const auto child = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(child != 0 && child != root);
    CHECK(f.session->domain_object_count() == 2 && f.handles.size() == 1);
    IpcMessage expected{};
    put(expected, 4, 13);
    put(expected, 16, 1);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0xbc2665da);
    put(expected, 48, child);
    CHECK(std::equal(expected.begin(), expected.begin() + 60, f.message.begin()));
    CHECK(std::all_of(f.registers.begin(), f.registers.end(), [](auto v) { return v == 0; }));
    // Wire type, not the input version word, chooses the route.
    f.message = control(3);
    put(f.message, 20, 1);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0, 2);
    f.message = control(3);
    put(f.message, 0, 7);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0, 2);
}

void test_unsupported_is_not_a_reply() {
    Fixture f;
    f.message = open_session(1); // Conversion has not happened.
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
        f.message = open_session(id);
        f.expect_unsupported();
    }
    f.message = open_session(root);
    put(f.message, 4, 0x8000000c); // Unexpected PID or handle descriptor.
    f.expect_unsupported();
    f.message = open_session(root);
    put(f.message, 4, 11); // Insufficient declared raw data.
    f.expect_unsupported();
    f.message = open_session(root);
    put(f.message, 0, 0x10004); // A pointer descriptor was added.
    f.expect_unsupported();
    f.message = open_session(root);
    put(f.message, 18, 15, 2);
    f.expect_unsupported();
    f.message = open_session(root);
    put(f.message, 18, 0xffff, 2);
    f.expect_unsupported();
    f.message = open_session(root);
    put(f.message, 16, 2, 1); // Domain close reply framing is not implemented.
    f.expect_unsupported();
    f.message = open_session(root);
    put(f.message, 40, 99);
    f.expect_unsupported();
    f.message = open_session(root);
    CHECK(f.send() == IpcDisposition::handled);
    f.message = open_session(static_cast<std::uint32_t>(get(f.message, 48)));
    f.expect_unsupported(); // Child methods are distinct from OpenSession.
}

void test_performance_modes() {
    using armigo::services::OperationMode;
    for (const auto mode : {OperationMode::Handheld, OperationMode::Console}) {
        for (const auto type : {4u, 6u}) {
            Fixture f{0, mode};
            auto open = plain_open_session();
            put(open, 0, type);
            const auto opened_reply = f.session->request(open, 71);
            auto opened = std::dynamic_pointer_cast<PerformanceSession>(opened_reply.transferred[0]);
            CHECK(opened != nullptr);
            for (const auto requested_mode : {0u, 1u}) {
                auto configuration = plain_open_session();
                put(configuration, 0, type);
                put(configuration, 4, 10);
                put(configuration, 32, requested_mode);
                put(configuration, 36, 0x10000u + requested_mode);
                CHECK(opened && opened->request(configuration, 71).disposition == IpcDisposition::handled);
                if (opened) {
                    const auto recorded = opened->requested_configuration();
                    CHECK(recorded.set && recorded.mode == requested_mode &&
                          recorded.configuration == 0x10000u + requested_mode);
                }
            }
            // Configuring either performance mode does not switch the launch profile.
            for (bool domain : {false, true}) {
                const auto root = domain ? f.convert() : 0;
                f.message = domain ? open_session(root) : plain_open_session();
                put(f.message, 0, type);
                put(f.message, domain ? 40 : 24, 1);
                CHECK(f.send() == IpcDisposition::handled);
                IpcMessage expected{};
                put(expected, 4, domain ? 13 : 9);
                const auto cmif = domain ? 32 : 16;
                put(expected, cmif, 0x4f434653);
                put(expected, cmif + 12, 0xbc2665da);
                put(expected, cmif + 16, mode == OperationMode::Console ? 1 : 0);
                const auto size = domain ? 60 : 44;
                CHECK(std::equal(expected.begin(), expected.begin() + size, f.message.begin()));
                CHECK(std::all_of(f.message.begin() + size, f.message.end(),
                                  [](std::byte b) { return b == std::byte{0xa5}; }));
                f.message = domain ? open_session(root) : plain_open_session();
                put(f.message, domain ? 40 : 24, 1);
                f.caller.process_id = 72;
                f.expect_unsupported();
                f.caller.process_id = 71;
            }
        }
    }
}

void test_serialized_object_creation() {
    Fixture f;
    const auto root = f.convert();
    std::array<std::uint32_t, 2> ids{};
    std::array<IpcDisposition, 2> dispositions{};
    const auto run = [&](unsigned index) {
        auto message = open_session(root);
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
    CHECK(f.session->domain_object_count() == 3 && f.handles.size() == 1);
}
} // namespace

int main() {
    test_performance_modes();
    test_startup();
    test_plain_open_session();
    test_provisioned_capacity();
    test_context_wire_types();
    test_unsupported_is_not_a_reply();
    test_serialized_object_creation();
    std::printf("Performance-service failures: %u\n", failures);
    return failures ? 1 : 0;
}
