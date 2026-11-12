// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/applet.h>
#include <armigo/services/settings.h>
#include <armigo/horizon/startup_queries.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <thread>

namespace {
using namespace armigo::horizon;
using armigo::services::am::AppletServiceSession;
using armigo::services::am::ApplicationRunningState;
using armigo::services::am::GameplayRecordingPolicy;
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
std::shared_ptr<ProcessRecord> make_process() {
    ProcessConfiguration configuration;
    configuration.heap_address = 0x1000000000;
    configuration.alias_address = 0x2000000000;
    configuration.stack_address = 0x6000000000;
    configuration.allowed_core_mask = 1;
    auto built = ProcessRecord::create(configuration, {});
    CHECK(built.ok());
    return built.process;
}

IpcMessage open_proxy(std::uint32_t root, Handle copied, std::uint32_t type = 6) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, type);
    put(message, 4, 0x8000000e);
    put(message, 8, 3); // Send PID and one copied handle.
    put(message, 12, 999, 8); // Untrusted PID placeholder.
    put(message, 20, copied);
    put(message, 32, 0x00180001); // Domain send, zero input objects, CMIF size24.
    put(message, 36, root);
    put(message, 40, 0);
    put(message, 44, 0x12345678); // Domain request context.
    put(message, 48, 0x49434653);
    put(message, 52, 1);
    put(message, 56, 0);
    put(message, 60, 0x98765432);
    put(message, 64, 123456, 8); // Independent inline PID placeholder.
    return message;
}

struct Fixture {
    HandleTable handles{8};
    std::shared_ptr<ProcessRecord> process = make_process();
    std::shared_ptr<AppletServiceSession> session;
    IpcDispatcher dispatcher{handles, nullptr};
    Handle handle;
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};

    explicit Fixture(std::uint64_t pid = 71,
                     GameplayRecordingPolicy recording_policy = GameplayRecordingPolicy::Unavailable,
                     std::shared_ptr<ApplicationRunningState> running_state = {},
                     armigo::services::OperationMode mode = armigo::services::OperationMode::Console,
                     armigo::services::am::AppletContract contract = armigo::services::am::AppletContract::switch1)
        : session(std::make_shared<AppletServiceSession>(pid, process, handles, recording_policy,
                                                         std::move(running_state), mode, contract)),
          handle(handles.insert(session).value()) { caller.process_id = pid; }

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
        CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::unsupported);
        CHECK(registers == saved_registers && message == before);
        CHECK(session->diagnostic() != nullptr);
    }
    std::uint32_t convert() {
        message = control(0);
        CHECK(send() == IpcDisposition::handled);
        const auto id = static_cast<std::uint32_t>(get(message, 32));
        CHECK(id != 0);
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

void test_controls() {
    for (auto type : {5u, 7u}) {
        Fixture f;
        f.message = control(3);
        put(f.message, 0, type);
        put(f.message, 20, 1);
        put(f.message, 28, 0x98765432);
        CHECK(f.send() == IpcDisposition::handled);
        check_control_reply(f.message, 0, 2);
        CHECK(std::all_of(f.registers.begin(), f.registers.end(), [](auto v) { return v == 0; }));
        f.message = control(0);
        put(f.message, 0, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto root = static_cast<std::uint32_t>(get(f.message, 32));
        CHECK(root != 0);
        check_control_reply(f.message, root, 4);
        f.message = control(0);
        f.expect_unsupported();
        f.message = control(3);
        CHECK(f.send() == IpcDisposition::handled);
        check_control_reply(f.message, 0, 2);
        std::weak_ptr<AppletServiceSession> lifetime = f.session;
        f.session.reset();
        CHECK(!lifetime.expired());
        CHECK(f.handles.close(f.handle));
        CHECK(lifetime.expired());
    }
}

void test_proxy_factory() {
    for (auto type : {4u, 6u}) {
        Fixture f;
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        CHECK(proxy != 0 && proxy != root && f.session->domain_object_count() == 2);
        CHECK(f.handles.size() == 1);
        IpcMessage expected{};
        put(expected, 4, 13);
        put(expected, 16, 1);
        put(expected, 32, 0x4f434653);
        put(expected, 44, 0xbb83bce3);
        put(expected, 48, proxy);
        CHECK(std::equal(expected.begin(), expected.begin() + 60, f.message.begin()));
        CHECK(get(f.message, 64, 8) == 123456);
        CHECK(f.session->diagnostic() == nullptr);
        CHECK(std::all_of(f.registers.begin(), f.registers.end(), [](auto v) { return v == 0; }));
        f.message = open_proxy(proxy, current_process_handle, type);
        f.expect_unsupported(); // Proxy child interfaces are not implemented.
    }

    Fixture f;
    const auto root = f.convert();
    struct CopiedObject final : KernelObject {};
    auto object = std::make_shared<CopiedObject>();
    std::weak_ptr<KernelObject> lifetime = object;
    const auto copied = f.handles.insert(object).value();
    f.message = open_proxy(root, copied);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(f.handles.lookup(copied) == object); // Copy does not consume sender handle.
    object.reset();
    CHECK(f.handles.close(copied));
    CHECK(!lifetime.expired()); // The admitted proxy owns a pin.
    f.session.reset();
    CHECK(f.handles.close(f.handle));
    CHECK(lifetime.expired());
}

IpcMessage proxy_getter(std::uint32_t proxy, std::uint32_t command, std::uint32_t type = 6) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, type);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, proxy);
    put(message, 24, 0);
    put(message, 28, 0x12345678);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, command);
    put(message, 44, 0x98765432);
    return message;
}

void test_switch2_startup_admission() {
    Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
        armigo::services::OperationMode::Console,
        armigo::services::am::AppletContract::switch2);
    f.message = control(3);
    CHECK(f.send() == IpcDisposition::handled);
    check_control_reply(f.message, 0, 2);
    f.message = control(0); put(f.message, 0, 5 | (1u << 16));
    f.expect_unsupported();
    CHECK(f.session->domain_object_count() == 0);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    put(f.message, 56, 10);
    f.expect_unsupported(); // The extended factory requires its additional word.
    CHECK(!f.session->application_proxy_configuration());
    put(f.message, 4, 0x80000010);
    put(f.message, 32, 0x00200001);
    put(f.message, 72, 0x10203040);
    f.expect_unsupported();
    CHECK(!f.session->application_proxy_configuration());
    put(f.message, 72, 0);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 44) == 0xbb83bce3);
    CHECK(f.session->application_proxy_configuration() == 0);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 0);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 44) == 0x7d59c154);
    const auto child = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 60);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 44) == 0x7d59c154);
    const auto graphics = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(graphics != child && graphics != proxy);
    f.message = proxy_getter(graphics, 10); // Provisional GetCurrentGraphicsScale.
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 15);
    CHECK(get(f.message, 48) == 10 && get(f.message, 52) == 0 && get(f.message, 56) == 0);
    f.message = proxy_getter(graphics, 1050);
    put(f.message, 4, 13);
    put(f.message, 16, 0x00110001);
    put(f.message, 48, 1);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 12 && get(f.message, 40) == 0);
    f.message = proxy_getter(graphics, 11);
    f.expect_unsupported();
    f.message = proxy_getter(proxy, 40);
    CHECK(f.send() == IpcDisposition::handled);
    const auto status = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(status != graphics && status != child && status != proxy);
    f.message = proxy_getter(status, 0);
    f.expect_unsupported();
    f.message = proxy_getter(proxy, 2);
    CHECK(f.send() == IpcDisposition::handled);
    const auto window = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(window, 1);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 44) == 0x0279422d);
    CHECK(get(f.message, 4) == 14 && get(f.message, 48, 8) == 71);
    f.message = proxy_getter(window, 10);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 12);
    CHECK(get(f.message, 44) == 0x0279422d);
    f.message = proxy_getter(child, 999);
    f.expect_unsupported();
    CHECK(!f.session->exit_to_library_on_next_program_enabled());
    f.message = proxy_getter(child, 900);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 12);
    CHECK(get(f.message, 44) == 0x32f27d77 && f.session->exit_to_library_on_next_program_enabled());
    f.message = proxy_getter(child, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto event = f.handles.lookup_as<Event>(static_cast<Handle>(get(f.message, 12)));
    CHECK(event && !event->is_signaled());
    f.message = proxy_getter(child, 1);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 40) == 0x680);
    f.message = proxy_getter(child, 9);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 48) == 2);
    f.session->grant_foreground();
    CHECK(event && event->is_signaled());
    f.message = proxy_getter(child, 9);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 48) == 1);
    f.message = proxy_getter(child, 1);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 13);
    CHECK(get(f.message, 48) == 15 && event && !event->is_signaled());
    f.message = proxy_getter(child, 1);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 40) == 0x680);
    f.message = {}; put(f.message, 0, 2);
    const auto close = f.message;
    CHECK(f.send() == IpcDisposition::handled && f.registers[0] == 0xf601);
    CHECK(f.message == close && f.handles.lookup(f.handle) == f.session);
    CHECK(f.session->domain_object_count() == 0);
    f.message = proxy_getter(proxy, 0);
    const auto request = f.message;
    CHECK(f.send() == IpcDisposition::handled && f.registers[0] == 0xf601);
    CHECK(f.message == request);
    CHECK(f.handles.close(f.handle));
}

void test_switch2_provisional_handlers() {
    using armigo::services::am::AppletContract;
    for (const auto contract : {AppletContract::switch1, AppletContract::switch2}) {
        Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
            armigo::services::OperationMode::Console, contract);
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 20);
        CHECK(f.send() == IpcDisposition::handled);
        const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
        const auto objects = f.session->domain_object_count();
        // Repeating a command replies the same way after its warning was logged.
        for (const auto command : {320u, 321u, 330u, 320u}) {
            f.message = proxy_getter(functions, command);
            if (contract == AppletContract::switch1) {
                f.expect_unsupported();
                continue;
            }
            CHECK(f.send() == IpcDisposition::handled);
            CHECK(get(f.message, 4) == 13 && get(f.message, 32) == 0x4f434653);
            CHECK(get(f.message, 40) == 0 && get(f.message, 44) == 0x21f9610e && get(f.message, 48) == 0);
            CHECK(f.session->diagnostic() == nullptr);
        }
        f.message = proxy_getter(proxy, 1);
        CHECK(f.send() == IpcDisposition::handled);
        const auto self = static_cast<std::uint32_t>(get(f.message, 48));
        // Command2400 is a boolean setter without output on Switch 2.
        f.message = proxy_getter(self, 2400);
        put(f.message, 4, 13);
        put(f.message, 16, 0x00110001);
        put(f.message, 48, 1);
        if (contract == AppletContract::switch1) {
            f.expect_unsupported();
            continue;
        }
        CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 12);
        CHECK(get(f.message, 40) == 0 && get(f.message, 44) == 0x56efeaa1);
    }
}

void test_startup_proxy_getters() {
    for (auto type : {4u, 6u}) {
        Fixture f;
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        std::vector<std::uint32_t> children;
        for (auto command : {20u, 0u, 1u, 2u, 3u, 4u, 11u, 1000u}) {
            f.message = proxy_getter(proxy, command, type);
            CHECK(f.send() == IpcDisposition::handled);
            const auto child = static_cast<std::uint32_t>(get(f.message, 48));
            CHECK(child != 0 && child != root && child != proxy);
            CHECK(std::find(children.begin(), children.end(), child) == children.end());
            children.push_back(child);
            IpcMessage expected{};
            put(expected, 4, 13);
            put(expected, 16, 1);
            put(expected, 32, 0x4f434653);
            put(expected, 44, 0x7d59c154);
            put(expected, 48, child);
            CHECK(std::equal(expected.begin(), expected.begin() + 60, f.message.begin()));
            CHECK(std::all_of(f.message.begin() + 60, f.message.end(),
                              [](std::byte b) { return b == std::byte{0xa5}; }));
            CHECK(f.handles.size() == 1);
            f.message = proxy_getter(child, 1000, type);
            f.expect_unsupported(); // A child is not another ApplicationProxy.
        }
        CHECK(f.session->domain_object_count() == 10);
        f.message = proxy_getter(root, 20, type);
        f.expect_unsupported();
        f.message = proxy_getter(proxy, 999, type);
        f.expect_unsupported();
        f.message = proxy_getter(proxy, 20, type);
        put(f.message, 4, 11);
        f.expect_unsupported();
        f.message = proxy_getter(proxy, 20, type);
        put(f.message, 4, 0x8000000c);
        f.expect_unsupported();
        CHECK(f.session->domain_object_count() == 10);
    }
}

void test_window_resource_identity() {
    constexpr std::uint64_t instance_id = 0x1234567800000047;
    for (auto type : {4u, 6u}) {
        Fixture f{instance_id};
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 2, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto window = static_cast<std::uint32_t>(get(f.message, 48));
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            f.message = proxy_getter(window, 1, type);
            CHECK(f.send() == IpcDisposition::handled);
            IpcMessage expected{};
            put(expected, 4, 14);
            put(expected, 32, 0x4f434653);
            put(expected, 44, 0x0279422d);
            put(expected, 48, instance_id, 8);
            CHECK(std::equal(expected.begin(), expected.begin() + 64, f.message.begin()));
            CHECK(std::all_of(f.message.begin() + 64, f.message.end(),
                              [](std::byte b) { return b == std::byte{0xa5}; }));
            CHECK(f.session->domain_object_count() == 3 && f.handles.size() == 1);
        }
        f.message = proxy_getter(window, 1, type);
        put(f.message, 16, 0x000f0001); // Truncated CMIF header.
        f.expect_unsupported();
        f.message = proxy_getter(window, 1, type);
        f.caller.process_id = 72;
        f.expect_unsupported();
    }
}

void test_next_program_exit_preference() {
    Fixture f;
    CHECK(!f.session->exit_to_library_on_next_program_enabled());
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto common = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(common, 900);
    f.caller.process_id = 72;
    f.expect_unsupported();
    CHECK(!f.session->exit_to_library_on_next_program_enabled());
    f.caller.process_id = 71;
    for (auto type : {4u, 6u}) {
        f.message = proxy_getter(common, 900, type);
        CHECK(f.send() == IpcDisposition::handled);
        CHECK(f.session->exit_to_library_on_next_program_enabled());
        IpcMessage expected{};
        put(expected, 4, 12);
        put(expected, 32, 0x4f434653);
        put(expected, 44, 0x32f27d77);
        CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
        CHECK(std::all_of(f.message.begin() + 56, f.message.end(),
                          [](std::byte b) { return b == std::byte{0xa5}; }));
        CHECK(f.session->domain_object_count() == 3);
    }
}

Handle request_event(Fixture& f, std::uint32_t object, std::uint32_t command,
                     std::uint32_t tag, std::uint32_t type = 6) {
    f.message = proxy_getter(object, command, type);
    CHECK(f.send() == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(get(f.message, 12));
    IpcMessage expected{};
    put(expected, 4, 0x8000000c);
    put(expected, 8, 2); // One copied handle, no PID or moved handles.
    put(expected, 12, handle);
    put(expected, 32, 0x4f434653);
    put(expected, 44, tag);
    CHECK(std::equal(expected.begin(), expected.begin() + 64, f.message.begin()));
    CHECK(std::all_of(f.message.begin() + 64, f.message.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
    CHECK(handle != 0 && f.handles.lookup_as<Event>(handle) != nullptr);
    return handle;
}

// SelfController command44 creates the display's managed separable layer and
// answers with the display and the recording layer id. The application keys its
// managed layer on the display id, so both must be non-zero.
void test_managed_display_layer() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 1);
    CHECK(f.send() == IpcDisposition::handled);
    const auto self = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(self, 44);
    CHECK(f.send() == IpcDisposition::handled);
    IpcMessage expected{};
    put(expected, 4, 14);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0x56efeaa1);
    CHECK(std::equal(expected.begin(), expected.begin() + 48, f.message.begin()));
    const auto display_layer = get(f.message, 48);
    const auto recording_layer = get(f.message, 56);
    CHECK(display_layer != 0 && recording_layer != 0 && display_layer != recording_layer);
    CHECK(get(f.message, 52) == 0 && get(f.message, 60) == 0);
    CHECK(std::all_of(f.message.begin() + 64, f.message.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
    // A second call hands out fresh ids instead of repeating the first pair.
    f.message = proxy_getter(self, 44);
    CHECK(f.send() == IpcDisposition::handled);
    const auto next_display = get(f.message, 48);
    const auto next_recording = get(f.message, 56);
    CHECK(next_display != 0 && next_recording != 0);
    CHECK(next_display != display_layer && next_recording != recording_layer);
    CHECK(next_display != next_recording);
}

void test_suspended_tick_event(armigo::services::am::AppletContract contract =
        armigo::services::am::AppletContract::switch1) {
    Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
        armigo::services::OperationMode::Console, contract);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 1);
    CHECK(f.send() == IpcDisposition::handled);
    const auto self = static_cast<std::uint32_t>(get(f.message, 48));
    const auto first = request_event(f, self, 91, 0x56efeaa1, 4);
    auto event = f.handles.lookup_as<Event>(first);
    CHECK(event != nullptr && !event->is_signaled());
    const auto second = request_event(f, self, 91, 0x56efeaa1);
    CHECK(first != second && f.handles.lookup(second) == event);
    CHECK(f.session->domain_object_count() == 3);
    CHECK(f.handles.close(first));
    CHECK(f.handles.lookup(second) == event);
    event->signal(); // Host-side service notification reaches all readable copies.
    const auto third = request_event(f, self, 91, 0x56efeaa1);
    CHECK(f.handles.lookup_as<Event>(third)->is_signaled());
    CHECK(event->reset() && !event->is_signaled());
    CHECK(f.handles.close(second) && f.handles.close(third));
    std::weak_ptr<Event> lifetime = event;
    event.reset();
    CHECK(!lifetime.expired()); // The application still owns its event.
    f.session.reset();
    CHECK(f.handles.close(f.handle));
    CHECK(lifetime.expired());
}

void test_ensure_save_data() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    // Command20 takes the user's UID and returns the space still required.
    f.message = proxy_getter(functions, 20);
    put(f.message, 16, 0x00200001); // Domain send, zero objects, CMIF size32.
    put(f.message, 4, 16);
    put(f.message, 48, 0x01234567);
    put(f.message, 60, 0x89abcdef);
    CHECK(f.send() == IpcDisposition::handled);
    IpcMessage expected{};
    put(expected, 4, 12);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0x21f9610e);
    CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
    CHECK(f.session->diagnostic() == nullptr);
    // The UID is required. A zero UID skips only the user save, so it also
    // succeeds with no space required.
    f.message = proxy_getter(functions, 20);
    put(f.message, 16, 0x00100001);
    f.expect_unsupported();
    f.message = proxy_getter(functions, 20);
    put(f.message, 16, 0x00200001);
    put(f.message, 4, 16);
    for (const std::size_t offset : {48, 52, 56, 60}) put(f.message, offset, 0);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
    CHECK(f.session->diagnostic() == nullptr);
    f.session.reset();
    CHECK(f.handles.close(f.handle));
}

void test_terminate_result() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    // Command22 carries one inline u32 and returns the plain domain header.
    f.message = proxy_getter(functions, 22);
    put(f.message, 16, 0x00140001); // Domain send, zero objects, CMIF size20.
    put(f.message, 4, 13);
    put(f.message, 48, 0x80040005u);
    put(f.message, 52, 0);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(f.session->terminate_result() == 0x80040005u);
    IpcMessage expected{};
    put(expected, 4, 10);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0x21f9610e);
    CHECK(std::equal(expected.begin(), expected.begin() + 48, f.message.begin()));
    // Only the admitted reply extent is overwritten; the request's own inline
    // payload sits at byte48, so the untouched tail starts after it.
    CHECK(std::all_of(f.message.begin() + 56, f.message.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
    CHECK(f.session->diagnostic() == nullptr);
    // A second call replaces the recorded value and is still idempotent.
    f.message = proxy_getter(functions, 22);
    put(f.message, 16, 0x00140001);
    put(f.message, 4, 13);
    put(f.message, 48, 0x0002e401u);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(f.session->terminate_result() == 0x0002e401u);
    // The inline payload is required and the call needs 13 raw words.
    f.message = proxy_getter(functions, 22);
    put(f.message, 16, 0x00100001);
    f.expect_unsupported();
    f.message = proxy_getter(functions, 22);
    put(f.message, 16, 0x00140001);
    put(f.message, 4, 12);
    f.expect_unsupported();
    f.message = proxy_getter(functions, 23);
    f.expect_unsupported();
    f.session.reset();
    CHECK(f.handles.close(f.handle));
}

void test_pop_launch_parameter() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    // Command1 finds no queued launch parameter: the empty-channel result is the CMIF
    // result word, not the transport result, and the client's no-parameter path handles it.
    f.message = proxy_getter(functions, 1);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(f.registers[0] == 0);
    CHECK(get(f.message, 32) == 0x4f434653);
    CHECK(get(f.message, 36) == 1);
    CHECK(get(f.message, 40) == 0x480);
    CHECK(f.session->diagnostic() == nullptr);
}

struct StorageMemory {
    std::array<std::byte, 0x100> bytes{};
    static bool read(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& memory = static_cast<StorageMemory*>(context)->bytes;
        if (address > memory.size() || out.size() > memory.size() - address) return false;
        std::copy_n(memory.begin() + static_cast<std::ptrdiff_t>(address), out.size(), out.begin());
        return true;
    }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& memory = static_cast<StorageMemory*>(context)->bytes;
        if (address > memory.size() || in.size() > memory.size() - address) return false;
        std::copy(in.begin(), in.end(), memory.begin() + static_cast<std::ptrdiff_t>(address));
        return true;
    }
};

// IStorageAccessor command11 with its B descriptor ahead of the raw data, so
// the domain header sits at byte 32.
IpcMessage storage_read(std::uint32_t accessor, std::uint64_t offset) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, 4 | (1u << 24));
    put(message, 4, 0xc00 | 15); // One receive-list entry follows the raw data.
    put(message, 32, 0x00180001); // Domain send, zero objects, CMIF size24.
    put(message, 36, accessor);
    put(message, 40, 0, 8);
    put(message, 48, 0x49434653);
    put(message, 52, 1);
    put(message, 56, 11);
    put(message, 60, 0);
    put(message, 64, offset, 8);
    return message;
}

void test_preselected_user_launch_parameter() {
    Fixture f;
    std::array<std::byte, 16> uid{};
    for (std::size_t index = 0; index < uid.size(); ++index) uid[index] = static_cast<std::byte>(index + 1);
    f.session->preselect_user(uid);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    const auto pop = [&](std::uint32_t kind) {
        f.message = proxy_getter(functions, 1);
        put(f.message, 16, 0x00140001);
        put(f.message, 4, 13);
        put(f.message, 48, kind);
        put(f.message, 52, 0);
        return f.send();
    };
    // Other kinds stay empty.
    CHECK(pop(1) == IpcDisposition::handled && get(f.message, 40) == 0x480);
    // The preselected user is a storage object, handed out once.
    CHECK(pop(2) == IpcDisposition::handled && get(f.message, 40) == 0);
    CHECK(get(f.message, 16) == 1);
    const auto storage = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(pop(2) == IpcDisposition::handled && get(f.message, 40) == 0x480);

    f.message = proxy_getter(storage, 0);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 16) == 1);
    CHECK(get(f.message, 44) == 0x8e211f3d);   // IStorage's interface word.
    const auto accessor = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(accessor, 0);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 48, 8) == 0x88);
    CHECK(get(f.message, 44) == 0x919f469a);   // IStorageAccessor's interface word.

    StorageMemory memory;
    GuestBuffers buffers(StorageMemory::read, StorageMemory::write, &memory);
    buffers.add({0x10, 0x88, false, true});
    auto read = f.session->request_with_buffers(storage_read(accessor, 0), 71, buffers);
    CHECK(read.disposition == IpcDisposition::handled && read.size == 48);
    CHECK(get(read.bytes, 44) == 0x919f469a);
    CHECK(get(memory.bytes, 0x10) == 0xc79497ca && memory.bytes[0x14] == std::byte{1});
    CHECK(std::equal(uid.begin(), uid.end(), memory.bytes.begin() + 0x18));
    CHECK(memory.bytes[0x10 + 0x87] == std::byte{0});
    GuestBuffers tail(StorageMemory::read, StorageMemory::write, &memory);
    tail.add({0x10, 0x10, false, true});
    read = f.session->request_with_buffers(storage_read(accessor, 0x78), 71, tail);
    CHECK(read.disposition == IpcDisposition::handled);
    read = f.session->request_with_buffers(storage_read(accessor, 0x80), 71, tail);
    CHECK(read.disposition == IpcDisposition::unsupported);

    // Closing the storage releases it; its id no longer answers.
    f.message.fill(std::byte{0xa5});
    put(f.message, 0, 4);
    put(f.message, 4, 8);
    put(f.message, 16, 2);
    put(f.message, 20, storage);
    CHECK(f.send() == IpcDisposition::handled);
    f.message = proxy_getter(storage, 0);
    f.expect_unsupported();
}

// IStorageAccessor command10 with its A descriptor ahead of the raw data.
IpcMessage storage_write(std::uint32_t accessor, std::uint64_t offset) {
    auto message = storage_read(accessor, offset);
    put(message, 0, 4 | (1u << 20));
    put(message, 4, 14);
    put(message, 56, 10);
    return message;
}

// The network-connection applet stand-in: storages are created and written,
// the applet takes its input, finishes when started and leaves one output.
void test_network_connection_applet() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 11);
    CHECK(f.send() == IpcDisposition::handled);
    const auto creator = static_cast<std::uint32_t>(get(f.message, 48));
    // The foreground application may launch library applets: the launchable
    // event is signaled.
    f.session->grant_foreground();
    f.message = proxy_getter(proxy, 1);
    CHECK(f.send() == IpcDisposition::handled);
    const auto self = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(self, 9);
    CHECK(f.send() == IpcDisposition::handled && ((get(f.message, 8) >> 1) & 0xf) == 1);
    const auto launchable = f.handles.lookup_as<Event>(static_cast<Handle>(get(f.message, 12)));
    CHECK(launchable != nullptr && launchable->is_signaled());
    const auto with_u64 = [&](std::uint32_t object, std::uint32_t command, std::uint64_t value) {
        f.message = proxy_getter(object, command);
        put(f.message, 16, 0x00180001);
        put(f.message, 4, 14);
        put(f.message, 48, value, 8);
        return f.send();
    };
    // CreateStorage(16) and a write through its accessor.
    CHECK(with_u64(creator, 10, 16) == IpcDisposition::handled && get(f.message, 16) == 1);
    const auto storage = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(storage, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto accessor = static_cast<std::uint32_t>(get(f.message, 48));
    StorageMemory memory;
    for (unsigned i = 0; i < 8; ++i) memory.bytes[0x20 + i] = static_cast<std::byte>(0x40 + i);
    GuestBuffers input(StorageMemory::read, StorageMemory::write, &memory);
    input.add({0x20, 8, true, false});
    const auto written = f.session->request_with_buffers(storage_write(accessor, 4), 71, input);
    CHECK(written.disposition == IpcDisposition::handled && get(written.bytes, 44) == 0x919f469a);
    CHECK(f.session->request_with_buffers(storage_write(accessor, 12), 71, input).disposition ==
          IpcDisposition::unsupported);
    GuestBuffers back(StorageMemory::read, StorageMemory::write, &memory);
    back.add({0xa0, 16, false, true});
    CHECK(f.session->request_with_buffers(storage_read(accessor, 0), 71, back).disposition ==
          IpcDisposition::handled);
    CHECK(get(memory.bytes, 0xa0) == 0 && memory.bytes[0xa4] == std::byte{0x40} &&
          memory.bytes[0xab] == std::byte{0x47} && get(memory.bytes, 0xac) == 0);
    CHECK(with_u64(creator, 10, 0) == IpcDisposition::unsupported);

    // Only the network-connection (15) and software keyboard (17) applets exist.
    f.message = proxy_getter(creator, 0);
    put(f.message, 16, 0x00180001);
    put(f.message, 4, 14);
    put(f.message, 48, 19);
    put(f.message, 52, 0);
    CHECK(f.send() == IpcDisposition::unsupported);
    f.message = proxy_getter(creator, 0);
    put(f.message, 16, 0x00180001);
    put(f.message, 4, 14);
    put(f.message, 48, 15);
    put(f.message, 52, 0);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 16) == 1);
    const auto applet = static_cast<std::uint32_t>(get(f.message, 48));

    f.message = proxy_getter(applet, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto event = f.handles.lookup_as<Event>(static_cast<Handle>(get(f.message, 12)));
    CHECK(event != nullptr && !event->is_signaled());
    // PushInData passes the storage as an input object after the raw data.
    f.message = proxy_getter(applet, 100);
    put(f.message, 16, 0x00100101);
    put(f.message, 4, 13);
    put(f.message, 48, storage);
    CHECK(f.send() == IpcDisposition::handled);
    f.message = proxy_getter(applet, 30);
    f.expect_unsupported();
    f.message = proxy_getter(applet, 10);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 40) == 0);
    CHECK(event && event->is_signaled());
    f.message = proxy_getter(applet, 1);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 48) == 1);
    f.message = proxy_getter(applet, 30);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 40) == 0);
    // One output storage holding u32 1, then no data in the channel.
    f.message = proxy_getter(applet, 101);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 16) == 1);
    const auto output = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(applet, 101);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 40) == 0x480);
    f.message = proxy_getter(output, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto output_accessor = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(output_accessor, 0);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 48, 8) == 4);
    GuestBuffers out(StorageMemory::read, StorageMemory::write, &memory);
    out.add({0x80, 4, false, true});
    CHECK(f.session->request_with_buffers(storage_read(output_accessor, 0), 71, out).disposition ==
          IpcDisposition::handled);
    CHECK(get(memory.bytes, 0x80) == 1);
    // A second start is refused.
    f.message = proxy_getter(applet, 10);
    f.expect_unsupported();
}

// Guest memory for the software keyboard tests: storages are written from it,
// read back into it, and the work buffer's transfer memory lives in it.
struct KeyboardMemory {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x4000);
    static bool read(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& memory = static_cast<KeyboardMemory*>(context)->bytes;
        if (address > memory.size() || out.size() > memory.size() - address) return false;
        std::copy_n(memory.begin() + static_cast<std::ptrdiff_t>(address), out.size(), out.begin());
        return true;
    }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& memory = static_cast<KeyboardMemory*>(context)->bytes;
        if (address > memory.size() || in.size() > memory.size() - address) return false;
        std::copy(in.begin(), in.end(), memory.begin() + static_cast<std::ptrdiff_t>(address));
        return true;
    }
    void put_text(std::size_t address, std::u16string_view text) {
        for (const auto unit : text) {
            bytes[address++] = static_cast<std::byte>(unit & 0xff);
            bytes[address++] = static_cast<std::byte>(unit >> 8);
        }
    }
};

// Common arguments and a 0x4c8-byte configuration as the 8.0.0+ keyboard
// library writes them.
std::vector<std::byte> keyboard_common_arguments(std::uint32_t library_version = 0x8000d) {
    std::vector<std::byte> bytes(0x20);
    const std::uint32_t words[] = {1, 0x20, library_version};
    for (std::size_t index = 0; index < 3; ++index)
        for (unsigned i = 0; i < 4; ++i)
            bytes[index * 4 + i] = static_cast<std::byte>((words[index] >> (8 * i)) & 255);
    return bytes;
}
std::vector<std::byte> keyboard_config(std::uint32_t max_length, std::uint32_t min_length, bool utf8,
                                       std::uint32_t initial_length, bool text_check = false) {
    std::vector<std::byte> bytes(0x4c8);
    const auto set = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    for (std::size_t index = 0; index < 4; ++index) bytes[0x24 + 2 * index] = static_cast<std::byte>("Name"[index]);
    set(0x3ac, max_length);
    set(0x3b0, min_length);
    bytes[0x3bd] = std::byte{utf8 ? std::uint8_t{1} : std::uint8_t{0}};
    set(0x3c0, initial_length ? 0x14 : 0);
    set(0x3c4, initial_length);
    bytes[0x3d0] = std::byte{text_check ? std::uint8_t{1} : std::uint8_t{0}};
    return bytes;
}

// ILibraryAppletCreator command11 CreateTransferMemoryStorage: the special
// header carries one copied handle, and the raw data still starts at byte 16.
IpcMessage transfer_storage_request(std::uint32_t creator, Handle memory, std::uint64_t size) {
    IpcMessage message;
    message.fill(std::byte{0xa5});
    put(message, 0, 4);
    put(message, 4, 0x80000000u | 16);
    put(message, 8, 2); // One copied handle, no process id.
    put(message, 12, memory);
    put(message, 16, 0x00200001); // Domain send, zero objects, CMIF size32.
    put(message, 20, creator);
    put(message, 24, 0, 8);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 11);
    put(message, 44, 0);
    put(message, 48, 1, 8); // Writable.
    put(message, 56, size, 8);
    return message;
}

struct KeyboardRun {
    IpcDisposition start = IpcDisposition::unsupported;
    std::vector<std::byte> output;
    bool state_signaled = false;
    bool interactive_signaled = true;
    std::size_t output_size = 0;
};

// Drives the applet the way the keyboard library does: create applet 17,
// push common arguments, configuration and the work buffer (transfer
// memory), start, then pop and read the output storage.
KeyboardRun run_keyboard_applet(Fixture& f, const std::vector<std::byte>& common,
                                const std::vector<std::byte>& config, std::u16string_view initial,
                                std::uint32_t mode = 0) {
    KeyboardRun run;
    KeyboardMemory memory;
    GuestBuffers guest(KeyboardMemory::read, KeyboardMemory::write, &memory);
    const auto call = [&](const IpcMessage& message, const GuestBuffers& buffers) {
        auto reply = f.session->request_with_buffers(message, 71, buffers);
        std::copy_n(reply.bytes.begin(), reply.bytes.size(), f.message.begin());
        return reply.disposition;
    };
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 11);
    CHECK(f.send() == IpcDisposition::handled);
    const auto creator = static_cast<std::uint32_t>(get(f.message, 48));
    const auto with_words = [&](std::uint32_t object, std::uint32_t command, std::uint64_t value) {
        auto message = proxy_getter(object, command);
        put(message, 16, 0x00180001);
        put(message, 4, 14);
        put(message, 48, value, 8);
        return call(message, guest);
    };
    if (with_words(creator, 0, std::uint64_t{mode} << 32 | 17) != IpcDisposition::handled) return run;
    const auto applet = static_cast<std::uint32_t>(get(f.message, 48));
    // A written storage, pushed to the applet's input channel.
    const auto push_bytes = [&](const std::vector<std::byte>& bytes) {
        CHECK(with_words(creator, 10, bytes.size()) == IpcDisposition::handled);
        const auto storage = static_cast<std::uint32_t>(get(f.message, 48));
        CHECK(call(proxy_getter(storage, 0), guest) == IpcDisposition::handled);
        const auto accessor = static_cast<std::uint32_t>(get(f.message, 48));
        std::copy(bytes.begin(), bytes.end(), memory.bytes.begin() + 0x1000);
        GuestBuffers input(KeyboardMemory::read, KeyboardMemory::write, &memory);
        input.add({0x1000, static_cast<std::uint32_t>(bytes.size()), true, false});
        CHECK(call(storage_write(accessor, 0), input) == IpcDisposition::handled);
        auto push = proxy_getter(applet, 100);
        put(push, 16, 0x00100101);
        put(push, 4, 13);
        put(push, 48, storage);
        CHECK(call(push, guest) == IpcDisposition::handled);
    };
    CHECK(call(proxy_getter(applet, 0), guest) == IpcDisposition::handled);
    const auto state = f.handles.lookup_as<Event>(static_cast<Handle>(get(f.message, 12)));
    push_bytes(common);
    push_bytes(config);
    // The work buffer: initial text at offset 0x14, lent as transfer memory.
    constexpr std::uint64_t work = 0x2000, work_size = 0x1000;
    memory.put_text(work + 0x14, initial);
    const auto lent = f.handles.insert(std::make_shared<TransferMemoryRecord>(AddressRange{work, work_size}, 0));
    CHECK(lent.has_value());
    CHECK(call(transfer_storage_request(creator, *lent, work_size + 1), guest) == IpcDisposition::unsupported);
    CHECK(call(transfer_storage_request(creator, *lent, work_size), guest) == IpcDisposition::handled);
    const auto work_storage = static_cast<std::uint32_t>(get(f.message, 48));
    // A transfer storage is not opened by the application.
    CHECK(call(proxy_getter(work_storage, 0), guest) == IpcDisposition::unsupported);
    auto push = proxy_getter(applet, 100);
    put(push, 16, 0x00100101);
    put(push, 4, 13);
    put(push, 48, work_storage);
    CHECK(call(push, guest) == IpcDisposition::handled);
    CHECK(call(proxy_getter(applet, 106), guest) == IpcDisposition::handled);
    const auto interactive = f.handles.lookup_as<Event>(static_cast<Handle>(get(f.message, 12)));
    run.start = call(proxy_getter(applet, 10), guest);
    run.state_signaled = state && state->is_signaled();
    run.interactive_signaled = !interactive || interactive->is_signaled();
    if (run.start != IpcDisposition::handled) return run;
    CHECK(call(proxy_getter(applet, 30), guest) == IpcDisposition::handled && get(f.message, 40) == 0);
    CHECK(call(proxy_getter(applet, 101), guest) == IpcDisposition::handled && get(f.message, 16) == 1);
    const auto output = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(call(proxy_getter(output, 0), guest) == IpcDisposition::handled);
    const auto accessor = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(call(proxy_getter(accessor, 0), guest) == IpcDisposition::handled);
    run.output_size = get(f.message, 48, 8);
    // The library reads the close result, then the text area from offset 4.
    GuestBuffers back(KeyboardMemory::read, KeyboardMemory::write, &memory);
    back.add({0x3000, static_cast<std::uint32_t>(run.output_size), false, true});
    CHECK(call(storage_read(accessor, 0), back) == IpcDisposition::handled);
    run.output.assign(memory.bytes.begin() + 0x3000,
                      memory.bytes.begin() + 0x3000 + static_cast<std::ptrdiff_t>(run.output_size));
    return run;
}

std::u16string output_utf16(const std::vector<std::byte>& output) {
    std::u16string text;
    for (std::size_t index = 4; index + 1 < output.size(); index += 2) {
        const auto unit = static_cast<char16_t>(std::to_integer<unsigned>(output[index]) |
                                                std::to_integer<unsigned>(output[index + 1]) << 8);
        if (!unit) break;
        text.push_back(unit);
    }
    return text;
}

void test_software_keyboard_applet() {
    using armigo::services::am::KeyboardRequest;
    using armigo::services::am::KeyboardResponse;
    {
        // Headless: OK with the initial text, UTF-16, in a 0x7d8-byte storage.
        Fixture f;
        const auto run = run_keyboard_applet(f, keyboard_common_arguments(), keyboard_config(8, 1, false, 3), u"Ren");
        CHECK(run.start == IpcDisposition::handled);
        CHECK(run.state_signaled && !run.interactive_signaled);
        CHECK(run.output_size == 0x7d8);
        CHECK(run.output.size() >= 4 && run.output[0] == std::byte{0} && output_utf16(run.output) == u"Ren");
        CHECK(f.session->diagnostic() == nullptr);
    }
    {
        // No initial text: the fixed default, fitted to the limits, as UTF-8.
        Fixture f;
        const auto run = run_keyboard_applet(f, keyboard_common_arguments(), keyboard_config(4, 0, true, 0), u"");
        CHECK(run.start == IpcDisposition::handled);
        CHECK(run.output.size() == 0x7d8 && std::string(reinterpret_cast<const char*>(run.output.data() + 4)) == "Play");
    }
    {
        // A host input decides the text; UTF-8 output keeps multi-byte characters.
        Fixture f;
        f.session->set_keyboard_input([](const KeyboardRequest& request) {
            CHECK(request.header == u"Name" && request.initial == u"Ren" && request.max_length == 10);
            return KeyboardResponse{true, u"\u96e8\u5bae"};
        });
        const auto run = run_keyboard_applet(f, keyboard_common_arguments(), keyboard_config(10, 1, true, 3), u"Ren");
        CHECK(run.output.size() == 0x7d8 &&
              std::string(reinterpret_cast<const char*>(run.output.data() + 4)) == "\xe9\x9b\xa8\xe5\xae\xae");
    }
    {
        // Cancel: close result 1 and no text.
        Fixture f;
        f.session->set_keyboard_input([](const KeyboardRequest&) { return KeyboardResponse{}; });
        const auto run = run_keyboard_applet(f, keyboard_common_arguments(), keyboard_config(10, 0, false, 3), u"Ren");
        CHECK(run.output.size() == 0x7d8 && run.output[0] == std::byte{1} &&
              output_utf16(run.output).empty());
    }
    {
        // Text checks through interactive storages are refused at start.
        Fixture f;
        const auto run = run_keyboard_applet(f, keyboard_common_arguments(), keyboard_config(10, 0, false, 3, true), u"Ren");
        CHECK(run.start == IpcDisposition::unsupported && !run.state_signaled);
        CHECK(f.session->diagnostic() != nullptr);
    }
    {
        // Another library version or configuration size is refused.
        Fixture f;
        const auto run = run_keyboard_applet(f, keyboard_common_arguments(0x50009), keyboard_config(10, 0, false, 3), u"Ren");
        CHECK(run.start == IpcDisposition::unsupported);
    }
    {
        // Only the full-screen mode (0) exists.
        Fixture f;
        const auto run = run_keyboard_applet(f, keyboard_common_arguments(), keyboard_config(10, 0, false, 3), u"Ren", 1);
        CHECK(run.start == IpcDisposition::unsupported && run.output.empty());
        CHECK(f.session->diagnostic() != nullptr);
    }
    {
        // Length fitting repeats short text and cuts long text.
        KeyboardRequest request;
        request.min_length = 7;
        request.max_length = 9;
        CHECK(armigo::services::am::fit_keyboard_text(request, u"Ren") == u"Ren Ren");
        request.min_length = 8;
        CHECK(armigo::services::am::fit_keyboard_text(request, u"Ren") == u"Ren Ren R");
        request.min_length = 0;
        request.max_length = 0;
        CHECK(armigo::services::am::fit_keyboard_text(request, std::u16string(600, u'a')).size() == 500);
    }
}

void test_desired_language() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    // Command21 answers the eight-byte language code at the first output word.
    f.message = proxy_getter(functions, 21);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(f.registers[0] == 0);
    CHECK(get(f.message, 4) == 12);
    CHECK(get(f.message, 32) == 0x4f434653);
    CHECK(get(f.message, 44) == 0x21f9610e);
    CHECK(get(f.message, 48, 8) == 0x00000053552d6e65ull);
    CHECK(f.session->diagnostic() == nullptr);
    CHECK(armigo::services::set::configure_console_language("zh-CN"));
    armigo::services::set::configure_supported_languages(1u << 14);
    f.message = proxy_getter(functions, 21);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(get(f.message, 4) == 10);
    CHECK(get(f.message, 40) == 0x25810);
    CHECK(f.session->diagnostic() != nullptr);
    armigo::services::set::configure_supported_languages(0xffff);
    CHECK(armigo::services::set::configure_console_language("en-US"));
}

void test_common_state_command66() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto common = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(common, 66);
    put(f.message, 16, 0x00140001);
    put(f.message, 4, 13);
    put(f.message, 36, 1);
    put(f.message, 48, 1);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(f.session->common_state_command66() == 1u);
    CHECK(f.session->common_state_boost_active());
    IpcMessage expected{};
    put(expected, 4, 12);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0x32f27d77);
    CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
    CHECK(f.session->diagnostic() == nullptr);
    // Only the low byte is the mode: 0x100 is mode 0.
    f.message = proxy_getter(common, 66);
    put(f.message, 16, 0x00140001);
    put(f.message, 4, 13);
    put(f.message, 48, 0x100);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 40) == 0);
    CHECK(f.session->common_state_command66() == 0u);
    // A mode above one is Result 0x3f480 on Switch 1 and changes nothing.
    f.message = proxy_getter(common, 66);
    put(f.message, 16, 0x00140001);
    put(f.message, 4, 13);
    put(f.message, 48, 2);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 40) == 0x3f480);
    CHECK(f.session->common_state_command66() == 0u);
    CHECK(f.session->common_state_boost_active());
    f.session.reset();
    CHECK(f.handles.close(f.handle));
}

void test_switch2_common_state_command66() {
    Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
        armigo::services::OperationMode::Console, armigo::services::am::AppletContract::switch2);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto common = static_cast<std::uint32_t>(get(f.message, 48));
    const auto set_mode = [&](std::uint32_t mode, std::uint32_t result) {
        f.message = proxy_getter(common, 66);
        put(f.message, 16, 0x00140001);
        put(f.message, 4, 13);
        put(f.message, 48, mode);
        CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 12 && get(f.message, 40) == result);
    };
    // Switch 2 adds the GPU-only mode 2.
    set_mode(2, 0);
    CHECK(f.session->common_state_command66() == 2u && f.session->common_state_boost_active());
    f.message = proxy_getter(common, 67);
    CHECK(f.send() == IpcDisposition::handled && !f.session->common_state_boost_active());
    // A larger mode is answered with 0x3f480 and changes nothing.
    set_mode(3, 0x3f480);
    CHECK(f.session->common_state_command66() == 2u && !f.session->common_state_boost_active());
    // Only the low byte is read.
    set_mode(0x100, 0);
    CHECK(f.session->common_state_command66() == 0u && f.session->common_state_boost_active());
}

// ApplicationFunctions 23 (GetDisplayVersion): 16 bytes, the launch's version
// NUL-terminated; unsupported when the launch supplies none.
void test_switch2_display_version() {
    Fixture f(72, GameplayRecordingPolicy::Unavailable, {},
        armigo::services::OperationMode::Console, armigo::services::am::AppletContract::switch2);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(functions, 23);
    CHECK(f.send() == IpcDisposition::unsupported && f.session->diagnostic());
    f.session->set_display_version("2.0.0");
    f.message = proxy_getter(functions, 23);
    CHECK(f.send() == IpcDisposition::handled && get(f.message, 4) == 14 && get(f.message, 32) == 0x4f434653 &&
          get(f.message, 40) == 0);
    CHECK(std::memcmp(f.message.data() + 48, "2.0.0", 6) == 0);
    CHECK(std::all_of(f.message.begin() + 54, f.message.begin() + 64, [](std::byte b) { return b == std::byte{0}; }));
}

// The Switch 2 SDK releases a child with the CMIF domain close (type 2 and
// the object id), as on Switch 1.
void test_switch2_domain_close() {
    Fixture f(74, GameplayRecordingPolicy::Unavailable, {},
        armigo::services::OperationMode::Console, armigo::services::am::AppletContract::switch2);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    // Ids stay assigned; the closed object takes no more requests or closes.
    const auto close = [&] {
        f.message = {};
        put(f.message, 0, 6);
        put(f.message, 4, 8);
        put(f.message, 16, 2);
        put(f.message, 20, functions);
        return f.send();
    };
    CHECK(close() == IpcDisposition::handled);
    f.message = proxy_getter(functions, 23);
    CHECK(f.send() != IpcDisposition::handled);
    CHECK(close() != IpcDisposition::handled);
}

void test_common_state_command67() {
    for (const auto type : {4u, 6u}) {
        Fixture f;
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 0, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto common = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto self = static_cast<std::uint32_t>(get(f.message, 48));
        const auto event_handle = request_event(f, common, 0, 0x32f27d77, type);
        const auto event = f.handles.lookup_as<Event>(event_handle);
        CHECK(event != nullptr && !event->is_signaled());

        const auto check_cancel_reply = [&] {
            IpcMessage expected{};
            put(expected, 4, 12);
            put(expected, 32, 0x4f434653);
            put(expected, 44, 0x32f27d77);
            CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
            CHECK(std::all_of(f.message.begin() + 56, f.message.end(),
                              [](std::byte b) { return b == std::byte{0xa5}; }));
            CHECK(f.registers[0] == 0);
            CHECK(f.session->diagnostic() == nullptr);
        };
        const auto cancel = [&] {
            f.message = proxy_getter(common, 67, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_cancel_reply();
            CHECK(!f.session->common_state_boost_active());
        };
        const auto activate = [&](std::uint32_t mode) {
            f.message = proxy_getter(common, 66, type);
            put(f.message, 16, 0x00140001);
            put(f.message, 4, 13);
            put(f.message, 48, mode);
            CHECK(f.send() == IpcDisposition::handled);
            CHECK(f.session->common_state_command66() == mode);
            CHECK(f.session->common_state_boost_active());
        };

        // Cancellation succeeds before activation and leaves the mode unset.
        CHECK(!f.session->common_state_command66().has_value());
        CHECK(!f.session->common_state_boost_active());
        cancel();
        CHECK(!f.session->common_state_command66().has_value());
        CHECK(event != nullptr && !event->is_signaled());

        // Cancellation preserves mode zero and remains idempotent.
        activate(0);
        cancel();
        CHECK(f.session->common_state_command66() == 0u);
        cancel();
        CHECK(f.session->common_state_command66() == 0u);

        // A later valid command66 reactivates the logical state.
        activate(1);
        const auto objects = f.session->domain_object_count();
        const auto handles = f.handles.size();

        f.message = proxy_getter(common, 67, type);
        put(f.message, 16, 0x000f0001);
        f.expect_unsupported();
        CHECK(f.session->common_state_command66() == 1u);
        CHECK(f.session->common_state_boost_active());

        f.message = proxy_getter(common, 67, type);
        f.caller.process_id = 72;
        f.expect_unsupported();
        f.caller.process_id = 71;
        CHECK(f.session->common_state_command66() == 1u);
        CHECK(f.session->common_state_boost_active());

        f.message = proxy_getter(self, 67, type);
        f.expect_unsupported();
        CHECK(f.session->common_state_command66() == 1u);
        CHECK(f.session->common_state_boost_active());
        CHECK(f.session->domain_object_count() == objects);
        CHECK(f.handles.size() == handles);
        CHECK(event != nullptr && !event->is_signaled());

        // Cancellation does not enqueue, consume or reset focus notification state.
        f.session->grant_foreground();
        CHECK(event != nullptr && event->is_signaled());
        cancel();
        CHECK(f.session->common_state_command66() == 1u);
        CHECK(event != nullptr && event->is_signaled());
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        IpcMessage notification{};
        put(notification, 4, 13);
        put(notification, 32, 0x4f434653);
        put(notification, 44, 0x32f27d77);
        put(notification, 48, 15);
        CHECK(std::equal(notification.begin(), notification.begin() + 60, f.message.begin()));
        CHECK(event != nullptr && !event->is_signaled());
        CHECK(f.handles.close(event_handle));
    }
}

void test_application_recording_state() {
    for (const auto type : {4u, 6u}) {
        Fixture f{71, GameplayRecordingPolicy::Unavailable};
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 20, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 0, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto common = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto self = static_cast<std::uint32_t>(get(f.message, 48));
        const auto event_handle = request_event(f, common, 0, 0x32f27d77, type);
        const auto event = f.handles.lookup_as<Event>(event_handle);
        CHECK(event != nullptr && !event->is_signaled());
        const auto objects = f.session->domain_object_count();
        const auto handles = f.handles.size();
        const auto request = [&](std::uint32_t state) {
            f.message = proxy_getter(functions, 67, type);
            put(f.message, 4, 13);
            put(f.message, 16, 0x00140001);
            put(f.message, 48, state);
        };
        const auto check_reply = [&](std::uint32_t result) {
            CHECK(f.send() == IpcDisposition::handled);
            IpcMessage expected{};
            put(expected, 4, 12);
            put(expected, 32, 0x4f434653);
            put(expected, 40, result);
            put(expected, 44, 0x21f9610e);
            CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
            CHECK(std::all_of(f.message.begin() + 56, f.message.end(),
                              [](std::byte b) { return b == std::byte{0xa5}; }));
            CHECK(f.registers[0] == 0);
            CHECK(f.session->diagnostic() == nullptr);
            CHECK(f.session->domain_object_count() == objects);
            CHECK(f.handles.size() == handles);
        };

        request(1);
        check_reply(0);
        CHECK(event != nullptr && !event->is_signaled());

        // The two command67 interfaces must not share state or reply framing.
        f.message = proxy_getter(common, 66, type);
        put(f.message, 4, 13);
        put(f.message, 16, 0x00140001);
        put(f.message, 48, 1);
        CHECK(f.send() == IpcDisposition::handled);
        f.session->grant_foreground();
        for (unsigned repetition = 0; repetition != 2; ++repetition) {
            for (const auto state : {0u, 1u, 0x100u, 0x101u, 2u, 0x102u, 0xffffffffu}) {
                request(state);
                check_reply((state & 0xff) <= 1 ? 0 : 0x3f480);
                CHECK(f.session->common_state_boost_active());
                CHECK(f.session->common_state_command66() == 1u);
                CHECK(event != nullptr && event->is_signaled());
            }
        }
        request(1);
        put(f.message, 16, 0x00130001);
        f.expect_unsupported();
        request(1);
        put(f.message, 4, 12);
        f.expect_unsupported();
        request(1);
        f.caller.process_id = 72;
        f.expect_unsupported();
        f.caller.process_id = 71;
        request(1);
        put(f.message, 20, self);
        f.expect_unsupported();
        f.message = proxy_getter(functions, 66, type);
        f.expect_unsupported();
        CHECK(f.session->domain_object_count() == objects);
        CHECK(f.handles.size() == handles);
        CHECK(f.session->common_state_boost_active());
        CHECK(event != nullptr && event->is_signaled());

        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        CHECK(get(f.message, 40) == 0 && get(f.message, 48) == 15);
        CHECK(event != nullptr && !event->is_signaled());
        request(1);
        check_reply(0);
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        CHECK(get(f.message, 40) == 0x680);
        CHECK(event != nullptr && !event->is_signaled());
        CHECK(f.handles.close(event_handle));

        Fixture unknown{71, static_cast<GameplayRecordingPolicy>(0xff)};
        const auto unknown_root = unknown.convert();
        unknown.message = open_proxy(unknown_root, current_process_handle, type);
        CHECK(unknown.send() == IpcDisposition::handled);
        const auto unknown_proxy = static_cast<std::uint32_t>(get(unknown.message, 48));
        unknown.message = proxy_getter(unknown_proxy, 20, type);
        CHECK(unknown.send() == IpcDisposition::handled);
        const auto unknown_functions = static_cast<std::uint32_t>(get(unknown.message, 48));
        for (const auto state : {1u, 2u}) {
            unknown.message = proxy_getter(unknown_functions, 67, type);
            put(unknown.message, 4, 13);
            put(unknown.message, 16, 0x00140001);
            put(unknown.message, 48, state);
            if (state == 1) {
                unknown.expect_unsupported();
            } else {
                // Input validation precedes the unsupported policy check.
                CHECK(unknown.send() == IpcDisposition::handled);
                CHECK(get(unknown.message, 40) == 0x3f480);
                CHECK(get(unknown.message, 44) == 0x21f9610e);
            }
        }
    }
}

// ApplicationFunctions command40 NotifyRunning answers the launch's running
// notification state. The state is host-owned per launch, shared across every
// session and domain object serving that launch, and never selected from
// application metadata.
std::uint32_t acquire_application_functions(Fixture& f, std::uint32_t type = 6) {
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle, type);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20, type);
    CHECK(f.send() == IpcDisposition::handled);
    return static_cast<std::uint32_t>(get(f.message, 48));
}

void check_notify_running_reply(const IpcMessage& actual, bool running) {
    IpcMessage expected{};
    put(expected, 4, 13);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0x21f9610e);
    put(expected, 48, running ? 1u : 0u);
    CHECK(std::equal(expected.begin(), expected.begin() + 60, actual.begin()));
    CHECK(std::all_of(actual.begin() + 60, actual.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
}

void test_notify_running_disabled_by_default() {
    for (auto type : {4u, 6u}) {
        // No supplied state: the session installs the explicit host default,
        // which keeps running notifications disabled and mutates nothing.
        Fixture f;
        const auto functions = acquire_application_functions(f, type);
        f.message = proxy_getter(functions, 40, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_notify_running_reply(f.message, false);
        CHECK(f.registers[0] == 0);
        CHECK(f.session->diagnostic() == nullptr);

        // An explicitly supplied default state answers the same and stays
        // unnotified.
        auto state = std::make_shared<ApplicationRunningState>();
        CHECK(!state->enabled() && !state->delivery_ready() && !state->notification_sent());
        Fixture supplied{71, GameplayRecordingPolicy::Unavailable, state};
        const auto supplied_functions = acquire_application_functions(supplied, type);
        supplied.message = proxy_getter(supplied_functions, 40, type);
        CHECK(supplied.send() == IpcDisposition::handled);
        check_notify_running_reply(supplied.message, false);
        CHECK(!state->notification_sent());
    }
}

void test_notify_running_enabled_not_ready() {
    std::atomic<unsigned> deliveries{0};
    auto state = std::make_shared<ApplicationRunningState>(true, false, [&] { ++deliveries; });
    CHECK(state->enabled() && !state->delivery_ready());
    Fixture f{71, GameplayRecordingPolicy::Unavailable, state};
    const auto functions = acquire_application_functions(f);
    for (unsigned repeat = 0; repeat != 2; ++repeat) {
        f.message = proxy_getter(functions, 40);
        CHECK(f.send() == IpcDisposition::handled);
        // The enabled result is reported on every call, but a launch whose
        // recipient is not ready never records or delivers a notification.
        check_notify_running_reply(f.message, true);
        CHECK(f.registers[0] == 0);
        CHECK(f.session->diagnostic() == nullptr);
        CHECK(!state->notification_sent());
    }
    CHECK(deliveries.load() == 0);
}

void test_notify_running_ready_delivers_once() {
    std::atomic<unsigned> deliveries{0};
    auto state = std::make_shared<ApplicationRunningState>(true, true, [&] { ++deliveries; });
    Fixture f{71, GameplayRecordingPolicy::Unavailable, state};
    const auto functions = acquire_application_functions(f);
    f.message = proxy_getter(functions, 40);
    CHECK(f.send() == IpcDisposition::handled);
    check_notify_running_reply(f.message, true);
    CHECK(state->notification_sent());
    CHECK(deliveries.load() == 1);
    // Repeated requests keep answering the enabled result without redelivering.
    f.message = proxy_getter(functions, 40);
    CHECK(f.send() == IpcDisposition::handled);
    check_notify_running_reply(f.message, true);
    CHECK(deliveries.load() == 1);
    CHECK(state->notification_sent());
}

void test_notify_running_ready_without_observer_marks_sent() {
    auto state = std::make_shared<ApplicationRunningState>(true, true);
    Fixture f{71, GameplayRecordingPolicy::Unavailable, state};
    const auto functions = acquire_application_functions(f);
    f.message = proxy_getter(functions, 40);
    CHECK(f.send() == IpcDisposition::handled);
    check_notify_running_reply(f.message, true);
    // The null-recipient branch still records the notification.
    CHECK(state->notification_sent());
}

void test_notify_running_shared_state_across_sessions() {
    std::atomic<unsigned> deliveries{0};
    auto state = std::make_shared<ApplicationRunningState>(true, true, [&] { ++deliveries; });
    Fixture first{71, GameplayRecordingPolicy::Unavailable, state};
    Fixture second{71, GameplayRecordingPolicy::Unavailable, state};
    const auto first_functions = acquire_application_functions(first);
    const auto second_functions = acquire_application_functions(second);
    first.message = proxy_getter(first_functions, 40);
    second.message = proxy_getter(second_functions, 40);
    IpcDisposition first_disposition = IpcDisposition::unsupported;
    IpcDisposition second_disposition = IpcDisposition::unsupported;
    std::thread one([&] { first_disposition = first.send(); });
    std::thread two([&] { second_disposition = second.send(); });
    one.join();
    two.join();
    CHECK(first_disposition == IpcDisposition::handled);
    CHECK(second_disposition == IpcDisposition::handled);
    check_notify_running_reply(first.message, true);
    check_notify_running_reply(second.message, true);
    // Independent sessions racing on one launch state deliver exactly once.
    CHECK(deliveries.load() == 1);
    CHECK(state->notification_sent());
}

void test_notify_running_domains_share_owner_state() {
    std::atomic<unsigned> deliveries{0};
    auto state = std::make_shared<ApplicationRunningState>(true, true, [&] { ++deliveries; });
    Fixture f{71, GameplayRecordingPolicy::Unavailable, state};
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto first = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto second = static_cast<std::uint32_t>(get(f.message, 48));
    CHECK(first != second);
    f.message = proxy_getter(first, 40);
    CHECK(f.send() == IpcDisposition::handled);
    check_notify_running_reply(f.message, true);
    f.message = proxy_getter(second, 40);
    CHECK(f.send() == IpcDisposition::handled);
    check_notify_running_reply(f.message, true);
    CHECK(deliveries.load() == 1);
    CHECK(state->notification_sent());
}

void test_notify_running_rejections_preserve_state() {
    std::atomic<unsigned> deliveries{0};
    auto state = std::make_shared<ApplicationRunningState>(true, true, [&] { ++deliveries; });
    Fixture f{71, GameplayRecordingPolicy::Unavailable, state};
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 1);
    CHECK(f.send() == IpcDisposition::handled);
    const auto self = static_cast<std::uint32_t>(get(f.message, 48));

    // A request from another process never reaches the launch state.
    f.message = proxy_getter(functions, 40);
    f.caller.process_id = 72;
    f.expect_unsupported();
    f.caller.process_id = 71;
    // Another child interface has no command 40.
    f.message = proxy_getter(self, 40);
    f.expect_unsupported();
    // The declared CMIF extent and raw word minimum still bound the request.
    f.message = proxy_getter(functions, 40);
    put(f.message, 16, 0x000f0001); // Truncated CMIF header.
    f.expect_unsupported();
    f.message = proxy_getter(functions, 40);
    put(f.message, 4, 11); // Below the domain raw word minimum.
    f.expect_unsupported();
    CHECK(deliveries.load() == 0);
    CHECK(!state->notification_sent());
    CHECK(state->enabled() && state->delivery_ready());

    // Stale payload bytes beyond the declared CMIF extent are ignored.
    f.message = proxy_getter(functions, 40);
    put(f.message, 48, 0xdeadbeef);
    put(f.message, 52, 0xcafef00d);
    CHECK(f.send() == IpcDisposition::handled);
    check_notify_running_reply(f.message, true);
    CHECK(deliveries.load() == 1);
    CHECK(state->notification_sent());
}

void test_notify_running_observer_failure_is_explicit() {
    std::atomic<unsigned> deliveries{0};
    auto state = std::make_shared<ApplicationRunningState>(true, true, [&] {
        ++deliveries;
        throw std::runtime_error("synthetic observer failure");
    });
    Fixture f{71, GameplayRecordingPolicy::Unavailable, state};
    const auto functions = acquire_application_functions(f);
    // The failing delivery fails its own request explicitly instead of
    // reporting success; the sent mark is retained.
    f.message = proxy_getter(functions, 40);
    f.expect_unsupported();
    CHECK(deliveries.load() == 1);
    CHECK(state->notification_sent());
    // The next request is not a retry: delivery is already complete, so it
    // succeeds without invoking the observer again.
    f.message = proxy_getter(functions, 40);
    CHECK(f.send() == IpcDisposition::handled);
    check_notify_running_reply(f.message, true);
    CHECK(deliveries.load() == 1);
    CHECK(f.session->diagnostic() == nullptr);
}

void test_gpu_error_event(armigo::services::am::AppletContract contract =
        armigo::services::am::AppletContract::switch1) {
    Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
        armigo::services::OperationMode::Console, contract);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 1);
    CHECK(f.send() == IpcDisposition::handled);
    const auto self = static_cast<std::uint32_t>(get(f.message, 48));
    const auto suspended_handle = request_event(f, self, 91, 0x56efeaa1);
    const auto first = request_event(f, functions, 130, 0x21f9610e, 4);
    auto gpu_event = f.handles.lookup_as<Event>(first);
    const auto suspended_event = f.handles.lookup_as<Event>(suspended_handle);
    CHECK(gpu_event != nullptr && gpu_event != suspended_event);
    CHECK(!gpu_event->is_signaled() && !suspended_event->is_signaled());
    const auto second = request_event(f, functions, 130, 0x21f9610e);
    CHECK(first != second && f.handles.lookup(second) == gpu_event);
    suspended_event->signal();
    CHECK(!gpu_event->is_signaled());
    gpu_event->signal();
    CHECK(f.handles.lookup_as<Event>(second)->is_signaled());
    CHECK(f.handles.close(first) && f.handles.close(second));
    CHECK(f.handles.close(suspended_handle));
    std::weak_ptr<Event> lifetime = gpu_event;
    gpu_event.reset();
    CHECK(!lifetime.expired());
    CHECK(f.session->domain_object_count() == 4);
    const auto count = f.handles.size();
    f.message = proxy_getter(self, 130);
    f.expect_unsupported();
    f.message = proxy_getter(functions, 130);
    f.caller.process_id = 72;
    f.expect_unsupported();
    CHECK(f.handles.size() == count);
    f.session.reset();
    CHECK(f.handles.close(f.handle));
    CHECK(lifetime.expired());
}

void test_common_state_end_vr_mode() {
    Fixture f;
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto common = static_cast<std::uint32_t>(get(f.message, 48));
    // EndVrModeEx succeeds with no output payload; the runtime never enters VR.
    f.message = proxy_getter(common, 54);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(f.registers[0] == 0);
    IpcMessage expected{};
    put(expected, 4, 12);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0x32f27d77);
    CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
    CHECK(f.session->diagnostic() == nullptr);
}

void test_common_state_event(armigo::services::am::AppletContract contract =
        armigo::services::am::AppletContract::switch1) {
    Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
        armigo::services::OperationMode::Console, contract);
    const auto root = f.convert();
    f.message = open_proxy(root, current_process_handle);
    CHECK(f.send() == IpcDisposition::handled);
    const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
    f.message = proxy_getter(proxy, 0);
    CHECK(f.send() == IpcDisposition::handled);
    const auto common = static_cast<std::uint32_t>(get(f.message, 48));
    const auto first = request_event(f, common, 0, 0x32f27d77, 4);
    auto event = f.handles.lookup_as<Event>(first);
    CHECK(event != nullptr && !event->is_signaled());
    event->signal();
    const auto second = request_event(f, common, 0, 0x32f27d77);
    CHECK(second != first && f.handles.lookup(second) == event);
    CHECK(f.handles.lookup_as<Event>(second)->is_signaled());
    CHECK(event->reset() && !f.handles.lookup_as<Event>(second)->is_signaled());
    f.message = proxy_getter(proxy, 1);
    CHECK(f.send() == IpcDisposition::handled);
    const auto self = static_cast<std::uint32_t>(get(f.message, 48));
    const auto suspended = request_event(f, self, 91, 0x56efeaa1);
    CHECK(f.handles.lookup(suspended) != event);
    f.message = proxy_getter(proxy, 20);
    CHECK(f.send() == IpcDisposition::handled);
    const auto functions = static_cast<std::uint32_t>(get(f.message, 48));
    const auto gpu = request_event(f, functions, 130, 0x21f9610e);
    CHECK(f.handles.lookup(gpu) != event);
    CHECK(f.handles.close(first) && f.handles.close(second));
    CHECK(f.handles.close(suspended) && f.handles.close(gpu));
    std::weak_ptr<Event> lifetime = event;
    event.reset();
    CHECK(!lifetime.expired());
    const auto count = f.handles.size();
    f.message = proxy_getter(common, 0);
    f.caller.process_id = 72;
    f.expect_unsupported();
    CHECK(f.handles.size() == count);
    f.session.reset();
    CHECK(f.handles.close(f.handle));
    CHECK(lifetime.expired());
}

void check_common_value_reply(const IpcMessage& actual, std::uint32_t value,
                              unsigned width, bool empty = false) {
    IpcMessage expected{};
    put(expected, 4, empty ? 12 : 13);
    put(expected, 32, 0x4f434653);
    put(expected, 40, empty ? 0x680 : 0);
    put(expected, 44, 0x32f27d77);
    if (!empty) put(expected, 48, value, width);
    const auto size = empty ? 56 : 60;
    CHECK(std::equal(expected.begin(), expected.begin() + size, actual.begin()));
    CHECK(std::all_of(actual.begin() + size, actual.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
}

void test_common_state_mode_queries(armigo::services::OperationMode mode) {
    using armigo::services::OperationMode;
    for (const auto type : {4u, 6u}) {
        Fixture f{71, GameplayRecordingPolicy::Unavailable, {}, mode};
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 0, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto common = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 2, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto window = static_cast<std::uint32_t>(get(f.message, 48));
        const auto event_handle = request_event(f, common, 0, 0x32f27d77, type);
        const auto event = f.handles.lookup_as<Event>(event_handle);
        CHECK(event != nullptr && !event->is_signaled());
        const auto resolution_handle = request_event(f, common, 61, 0x32f27d77, type);
        const auto resolution_event = f.handles.lookup_as<Event>(resolution_handle);
        CHECK(resolution_event && resolution_event != event && !resolution_event->is_signaled());
        const auto second_resolution_handle = request_event(f, common, 61, 0x32f27d77, type);
        CHECK(f.handles.lookup_as<Event>(second_resolution_handle) == resolution_event);
        CHECK(f.handles.close(second_resolution_handle));
        const auto object_count = f.session->domain_object_count();
        const auto handle_count = f.handles.size();

        const auto query_modes = [&](bool signaled) {
            f.message = proxy_getter(common, 60, type);
            CHECK(f.send() == IpcDisposition::handled);
            IpcMessage resolution_reply{};
            put(resolution_reply, 4, 14);
            put(resolution_reply, 32, 0x4f434653);
            put(resolution_reply, 44, 0x32f27d77);
            put(resolution_reply, 48, mode == OperationMode::Console ? 1920 : 1280);
            put(resolution_reply, 52, mode == OperationMode::Console ? 1080 : 720);
            CHECK(std::equal(resolution_reply.begin(), resolution_reply.begin() + 64, f.message.begin()));
            CHECK(std::all_of(f.message.begin() + 64, f.message.end(),
                              [](std::byte b) { return b == std::byte{0xa5}; }));
            CHECK(!resolution_event->is_signaled());
            for (const auto command : {5u, 6u, 50u}) {
                for (const auto stale : {0u, 1u, 0xffffffffu}) {
                    for (unsigned repeat = 0; repeat != 2; ++repeat) {
                        f.message = proxy_getter(common, command, type);
                        // No inline input is declared; these bytes must not
                        // select a mode or change notification preferences.
                        put(f.message, 48, stale);
                        CHECK(f.send() == IpcDisposition::handled);
                        check_common_value_reply(f.message, command == 50 ? 0 :
                            (mode == OperationMode::Console ? 1 : 0), command == 6 ? 4 : 1);
                        CHECK(f.session->diagnostic() == nullptr);
                        CHECK(std::all_of(f.registers.begin(), f.registers.end(),
                                          [](auto value) { return value == 0; }));
                        CHECK(event != nullptr && event->is_signaled() == signaled);
                        CHECK(f.session->domain_object_count() == object_count);
                        CHECK(f.handles.size() == handle_count);
                    }
                }
            }
        };

        query_modes(false);
        f.message = proxy_getter(common, 9, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 2, 1);
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 0, 4, true);

        // Foreground ownership changes focus, but not the fixed launch modes.
        f.session->grant_foreground();
        query_modes(true);
        f.message = proxy_getter(common, 54, type);
        CHECK(f.send() == IpcDisposition::handled);
        query_modes(true);
        f.message = proxy_getter(common, 9, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 1, 1);

        for (const auto command : {5u, 6u, 50u, 60u, 61u}) {
            f.message = proxy_getter(window, command, type);
            f.expect_unsupported();
            f.message = proxy_getter(common, command, type);
            f.caller.process_id = 72;
            f.expect_unsupported();
            f.caller.process_id = 71;

            struct InvalidField {
                unsigned offset;
                std::uint32_t value;
            };
            const InvalidField malformed[] = {
                {0, 0x10000u | type}, // Unexpected descriptor count.
                {4, 0x8000000c}, // Unexpected special header.
                {4, 11}, // Truncated raw extent.
                {4, static_cast<std::uint32_t>(f.message.size() / 4)}, // Oversized raw extent.
                {16, 0x000f0001}, // Truncated CMIF header.
                {16, 0x00190001}, // CMIF extent exceeds the raw extent.
                {16, 0x00100101}, // Unexpected input domain object.
                {16, 0x00100002}, // Wrong domain operation.
                {20, 0xffffffff}, // Unknown domain object.
                {32, 0}, // Invalid CMIF magic.
            };
            for (const auto field : malformed) {
                f.message = proxy_getter(common, command, type);
                put(f.message, field.offset, field.value);
                f.expect_unsupported();
                CHECK(event != nullptr && event->is_signaled());
            }
        }

        // Neither successful queries nor rejected requests consume the message.
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 15, 4);
        CHECK(event != nullptr && !event->is_signaled());
        query_modes(false);
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 0, 4, true);
        CHECK(f.handles.close(event_handle));
        CHECK(f.handles.close(resolution_handle));
    }
}

void test_foreground_launch_notification() {
    for (bool grant_before_event : {false, true}) {
        for (auto type : {4u, 6u}) {
            Fixture f;
            const auto root = f.convert();
            f.message = open_proxy(root, current_process_handle);
            CHECK(f.send() == IpcDisposition::handled);
            const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
            f.message = proxy_getter(proxy, 0);
            CHECK(f.send() == IpcDisposition::handled);
            const auto common = static_cast<std::uint32_t>(get(f.message, 48));
            f.message = proxy_getter(common, 1, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_common_value_reply(f.message, 0, 4, true);
            f.message = proxy_getter(common, 9, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_common_value_reply(f.message, 2, 1);
            if (grant_before_event) f.session->grant_foreground();
            const auto handle = request_event(f, common, 0, 0x32f27d77, type);
            auto event = f.handles.lookup_as<Event>(handle);
            CHECK(event->is_signaled() == grant_before_event);
            f.session->grant_foreground();
            f.session->grant_foreground();
            CHECK(event->is_signaled());
            f.message = proxy_getter(common, 9, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_common_value_reply(f.message, 1, 1);
            CHECK(event->is_signaled()); // Query does not consume the launch message.
            f.message = proxy_getter(common, 1, type);
            f.caller.process_id = 72;
            f.expect_unsupported();
            CHECK(event->is_signaled());
            f.caller.process_id = 71;
            f.message = proxy_getter(common, 1, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_common_value_reply(f.message, 15, 4);
            CHECK(!event->is_signaled());
            f.session->grant_foreground(); // Already granted: no duplicate notification.
            f.message = proxy_getter(common, 1, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_common_value_reply(f.message, 0, 4, true);
            CHECK(!event->is_signaled());
            f.message = proxy_getter(common, 9, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_common_value_reply(f.message, 1, 1);
            CHECK(f.session->domain_object_count() == 3);
        }
    }
}

void test_foreground_rights() {
    using armigo::services::am::AppletContract;
    // WindowController 10 succeeds on both platforms and does not change focus.
    for (const auto contract : {AppletContract::switch1, AppletContract::switch2}) {
        Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
                  armigo::services::OperationMode::Console, contract);
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 0);
        CHECK(f.send() == IpcDisposition::handled);
        const auto common = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(common, 9);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 2, 1);
        f.message = proxy_getter(proxy, 2);
        CHECK(f.send() == IpcDisposition::handled);
        const auto window = static_cast<std::uint32_t>(get(f.message, 48));
        for (auto type : {4u, 6u}) {
            f.message = proxy_getter(window, 10, type);
            CHECK(f.send() == IpcDisposition::handled);
            IpcMessage expected{};
            put(expected, 4, 12);
            put(expected, 32, 0x4f434653);
            put(expected, 44, 0x0279422d);
            CHECK(std::equal(expected.begin(), expected.begin() + 56, f.message.begin()));
            CHECK(std::all_of(f.message.begin() + 56, f.message.end(),
                              [](std::byte b) { return b == std::byte{0xa5}; }));
        }
        f.message = proxy_getter(common, 9);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 2, 1);
    }
}

IpcMessage self_preference(std::uint32_t object, std::uint32_t command,
                           std::span<const std::uint8_t> values, std::uint32_t type = 6) {
    auto message = proxy_getter(object, command, type);
    put(message, 4, 12 + (values.size() + 3) / 4);
    put(message, 16, 1 | ((16 + values.size()) << 16));
    for (std::size_t i = 0; i < values.size(); ++i)
        put(message, 48 + i, values[i], 1);
    return message;
}

IpcMessage self_u32_request(std::uint32_t object, std::uint32_t command,
                            std::uint32_t value, std::uint32_t type = 6) {
    auto message = proxy_getter(object, command, type);
    put(message, 4, 13);
    put(message, 16, 0x00140001);
    put(message, 48, value);
    return message;
}

void check_self_preference_reply(const IpcMessage& actual, std::uint32_t result = 0) {
    IpcMessage expected{};
    put(expected, 4, 12);
    put(expected, 32, 0x4f434653);
    put(expected, 40, result);
    put(expected, 44, 0x56efeaa1);
    CHECK(std::equal(expected.begin(), expected.begin() + 56, actual.begin()));
    CHECK(std::all_of(actual.begin() + 56, actual.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
}

void check_self_u32_reply(const IpcMessage& actual, std::uint32_t value) {
    IpcMessage expected{};
    put(expected, 4, 13);
    put(expected, 32, 0x4f434653);
    put(expected, 44, 0x56efeaa1);
    put(expected, 48, value);
    CHECK(std::equal(expected.begin(), expected.begin() + 60, actual.begin()));
    CHECK(std::all_of(actual.begin() + 60, actual.end(),
                      [](std::byte b) { return b == std::byte{0xa5}; }));
}

// SelfController LockExit and UnlockExit carry nothing and toggle the kept state.
void test_exit_lock() {
    for (const auto type : {4u, 6u}) {
        Fixture f;
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto self = static_cast<std::uint32_t>(get(f.message, 48));
        CHECK(!f.session->exit_locked());
        for (const auto permission : {1u, 2u, 0u}) {
            f.message = self_u32_request(self, 10, permission, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_self_preference_reply(f.message);
            CHECK(f.session->screenshot_permission() == permission);
        }
        f.message = self_u32_request(self, 10, 3, type);
        CHECK(f.send() == IpcDisposition::unsupported);
        CHECK(f.session->screenshot_permission() == 0);
        for (const auto command : {1u, 2u, 1u}) {
            f.message = proxy_getter(self, command, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_self_preference_reply(f.message);
            CHECK(f.session->exit_locked() == (command == 1));
        }
    }
}

void test_idle_time_detection_extension() {
    for (const auto type : {4u, 6u}) {
        Fixture f;
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 0, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto common = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto first_self = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        const auto second_self = static_cast<std::uint32_t>(get(f.message, 48));

        const auto event_handle = request_event(f, common, 0, 0x32f27d77, type);
        const auto event = f.handles.lookup_as<Event>(event_handle);
        CHECK(event != nullptr && !event->is_signaled());
        f.session->grant_foreground();
        CHECK(event != nullptr && event->is_signaled());
        const auto objects = f.session->domain_object_count();
        const auto handles = f.handles.size();

        const auto check_value = [&](std::uint32_t object, std::uint32_t expected) {
            f.message = proxy_getter(object, 63, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_self_u32_reply(f.message, expected);
            CHECK(event != nullptr && event->is_signaled());
        };

        check_value(first_self, 0);
        for (const auto value : {0u, 1u, 2u, 2u}) {
            f.message = self_u32_request(first_self, 62, value, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_self_preference_reply(f.message);
            check_value(second_self, value);
        }

        // A separately constructed session retains its own default state.
        Fixture isolated;
        const auto isolated_root = isolated.convert();
        isolated.message = open_proxy(isolated_root, current_process_handle);
        CHECK(isolated.send() == IpcDisposition::handled);
        const auto isolated_proxy = static_cast<std::uint32_t>(get(isolated.message, 48));
        isolated.message = proxy_getter(isolated_proxy, 1, type);
        CHECK(isolated.send() == IpcDisposition::handled);
        const auto isolated_self = static_cast<std::uint32_t>(get(isolated.message, 48));
        isolated.message = proxy_getter(isolated_self, 63, type);
        CHECK(isolated.send() == IpcDisposition::handled);
        check_self_u32_reply(isolated.message, 0);

        for (const auto value : {3u, 0x100u, 0x101u, 0x102u, 0x80000000u, 0xffffffffu}) {
            f.message = self_u32_request(second_self, 62, value, type);
            CHECK(f.send() == IpcDisposition::handled);
            check_self_preference_reply(f.message, 0x3f480);
            check_value(first_self, 2);
        }

        f.message = self_u32_request(first_self, 62, 1, type);
        put(f.message, 16, 0x00130001);
        f.expect_unsupported();
        check_value(second_self, 2);
        f.message = self_u32_request(first_self, 62, 1, type);
        put(f.message, 4, 12);
        f.expect_unsupported();
        check_value(second_self, 2);

        f.message = proxy_getter(second_self, 63, type);
        put(f.message, 16, 0x000f0001);
        f.expect_unsupported();
        f.message = proxy_getter(second_self, 63, type);
        put(f.message, 4, 11);
        f.expect_unsupported();

        f.message = self_u32_request(common, 62, 1, type);
        f.expect_unsupported();
        f.message = proxy_getter(common, 63, type);
        f.expect_unsupported();
        f.message = self_u32_request(first_self, 62, 1, type);
        f.caller.process_id = 72;
        f.expect_unsupported();
        f.message = proxy_getter(first_self, 63, type);
        f.expect_unsupported();
        f.caller.process_id = 71;

        check_value(second_self, 2);
        CHECK(f.session->domain_object_count() == objects);
        CHECK(f.handles.size() == handles);
        CHECK(event != nullptr && event->is_signaled());
        CHECK(f.handles.close(event_handle));
    }
}

void test_initial_focus_preferences(armigo::services::am::AppletContract contract =
        armigo::services::am::AppletContract::switch1) {
    for (auto type : {4u, 6u}) {
        Fixture f(71, GameplayRecordingPolicy::Unavailable, {},
            armigo::services::OperationMode::Console, contract);
        const auto root = f.convert();
        f.message = open_proxy(root, current_process_handle);
        CHECK(f.send() == IpcDisposition::handled);
        const auto proxy = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 0);
        CHECK(f.send() == IpcDisposition::handled);
        const auto common = static_cast<std::uint32_t>(get(f.message, 48));
        f.message = proxy_getter(proxy, 1);
        CHECK(f.send() == IpcDisposition::handled);
        const auto self = static_cast<std::uint32_t>(get(f.message, 48));
        const auto handle = request_event(f, common, 0, 0x32f27d77, type);
        auto event = f.handles.lookup_as<Event>(handle);
        f.session->grant_foreground();
        CHECK(event->is_signaled());
        const std::array<std::uint8_t, 3> initial_mode{0, 0, 1};
        const std::array<std::uint8_t, 1> initial_out_of_focus{0};
        f.message = self_preference(self, 13, initial_mode, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_self_preference_reply(f.message);
        CHECK(!event->is_signaled());
        f.message = self_preference(self, 16, initial_out_of_focus, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_self_preference_reply(f.message);
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 0, 4, true);
        const std::array<std::uint8_t, 2> short_mode{1, 1};
        f.message = self_preference(self, 13, short_mode, type);
        f.expect_unsupported();
        f.message = self_preference(self, 16, {}, type);
        f.expect_unsupported();
        CHECK(!event->is_signaled());
        const std::array<std::uint8_t, 3> masked_off{2, 3, 3};
        f.message = self_preference(self, 13, masked_off, type);
        CHECK(f.send() == IpcDisposition::handled);
        CHECK(!event->is_signaled()); // Boolean decoding uses bit0, not nonzero.
        f.message = proxy_getter(common, 9, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 1, 1);
        const std::array<std::uint8_t, 3> masked_on{3, 2, 2};
        f.message = self_preference(self, 13, masked_on, type);
        put(f.message, 4, 12);
        f.expect_unsupported();
        CHECK(!event->is_signaled());
        f.message = self_preference(self, 13, masked_on, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_self_preference_reply(f.message);
        CHECK(event->is_signaled()); // Disabling notifications preserved pending15.
        const std::array<std::uint8_t, 1> suspend_hint{1};
        f.message = self_preference(self, 16, suspend_hint, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_self_preference_reply(f.message);
        CHECK(event->is_signaled());
        f.message = proxy_getter(common, 9, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 1, 1);
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 15, 4);
        CHECK(!event->is_signaled());
        f.message = proxy_getter(common, 1, type);
        CHECK(f.send() == IpcDisposition::handled);
        check_common_value_reply(f.message, 0, 4, true);
    }
}

void test_proxy_rejections() {
    Fixture f;
    f.message = open_proxy(1, current_process_handle);
    f.expect_unsupported();
    const auto root = f.convert();
    for (auto handle : {Handle{0}, Handle{0x1234}, current_thread_handle}) {
        f.message = open_proxy(root, handle);
        f.expect_unsupported();
    }
    f.message = open_proxy(root, current_process_handle);
    f.caller.process_id = 72;
    f.expect_unsupported();
    f.caller.process_id = 71;
    for (auto offset : {8u, 32u, 36u, 48u}) {
        f.message = open_proxy(root, current_process_handle);
        put(f.message, offset, 0);
        f.expect_unsupported();
    }
    f.message = open_proxy(root, current_process_handle);
    put(f.message, 4, 0x8000000d);
    f.expect_unsupported();
    f.message = open_proxy(root, current_process_handle);
    put(f.message, 32, 0xffff0001);
    f.expect_unsupported();
    f.message = open_proxy(root, current_process_handle);
    put(f.message, 0, 0x10006);
    f.expect_unsupported();
    CHECK(f.session->domain_object_count() == 1);
}

void test_unsupported_preservation() {
    Fixture f;
    f.message = control(3);
    f.caller.process_id = 72;
    f.expect_unsupported();
    f.caller.process_id = 71;
    f.message = control(4);
    f.expect_unsupported();
    f.message = control(3);
    put(f.message, 4, 7);
    f.expect_unsupported();
    f.message = control(3);
    put(f.message, 16, 0);
    f.expect_unsupported();
    f.message = control(3);
    put(f.message, 0, 0x10005);
    f.expect_unsupported();
    f.message = control(3);
    put(f.message, 4, 0x80000008);
    f.expect_unsupported();
}
} // namespace

int main() {
    test_switch2_startup_admission();
    test_switch2_provisional_handlers();
    test_controls();
    test_proxy_factory();
    test_proxy_rejections();
    test_startup_proxy_getters();
    test_window_resource_identity();
    test_next_program_exit_preference();
    test_managed_display_layer();
    test_suspended_tick_event();
    test_suspended_tick_event(armigo::services::am::AppletContract::switch2);
    test_switch2_display_version();
    test_switch2_domain_close();
    test_ensure_save_data();
    test_preselected_user_launch_parameter();
    test_network_connection_applet();
    test_software_keyboard_applet();
    test_terminate_result();
    test_desired_language();
    test_pop_launch_parameter();
    test_common_state_command66();
    test_switch2_common_state_command66();
    test_common_state_command67();
    test_application_recording_state();
    test_notify_running_disabled_by_default();
    test_notify_running_enabled_not_ready();
    test_notify_running_ready_delivers_once();
    test_notify_running_ready_without_observer_marks_sent();
    test_notify_running_shared_state_across_sessions();
    test_notify_running_domains_share_owner_state();
    test_notify_running_rejections_preserve_state();
    test_notify_running_observer_failure_is_explicit();
    test_common_state_end_vr_mode();
    test_gpu_error_event();
    test_gpu_error_event(armigo::services::am::AppletContract::switch2);
    test_common_state_event();
    test_common_state_event(armigo::services::am::AppletContract::switch2);
    test_common_state_mode_queries(armigo::services::OperationMode::Console);
    test_common_state_mode_queries(armigo::services::OperationMode::Handheld);
    test_foreground_launch_notification();
    test_foreground_rights();
    test_idle_time_detection_extension();
    test_exit_lock();
    test_initial_focus_preferences();
    test_initial_focus_preferences(armigo::services::am::AppletContract::switch2);
    test_unsupported_preservation();
    std::printf("Applet-service failures: %u\n", failures);
    return failures ? 1 : 0;
}
