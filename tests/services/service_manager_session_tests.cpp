// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/services/service_manager_session.h"

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace {
using namespace armigo::horizon;
using namespace armigo::services::sm;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)

void put(IpcMessage& message, std::size_t offset, std::uint64_t value, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
        message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

std::uint64_t get(const IpcMessage& message, std::size_t offset, std::size_t count) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < count; ++i)
        value |= std::uint64_t{std::to_integer<unsigned char>(message[offset + i])} << (8 * i);
    return value;
}

ServiceName service_name(std::string_view value) {
    ServiceName name{};
    for (std::size_t i = 0; i < value.size() && i < name.size(); ++i)
        name[i] = static_cast<std::byte>(value[i]);
    return name;
}

IpcMessage lookup(std::string_view value) {
    IpcMessage message{};
    put(message, 0, 0x200000011ULL, 8);
    const auto name = service_name(value);
    std::copy(name.begin(), name.end(), message.begin() + 8);
    return message;
}

class SyntheticClient final : public ClientSession {
public:
    unsigned requests = 0;
    IpcReply request(const IpcMessage& message, std::uint64_t pid) override {
        ++requests;
        CHECK(pid == 71 && message[0] == std::byte{0xa5});
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 1;
        reply.bytes[0] = std::byte{0x5a};
        return reply;
    }
};

struct Fixture {
    HandleTable handles{8};
    std::shared_ptr<SyntheticClient> client = std::make_shared<SyntheticClient>();
    bool available = false;
    unsigned connections = 0;
    std::weak_ptr<int> server_lifetime;
    std::shared_ptr<ServiceManagerSession> sm;
    IpcDispatcher dispatcher;
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext context;
    Handle sm_handle = 0;

    Fixture()
        : dispatcher(handles, connect_port, this), context{71, message, read_byte, nullptr} {
        ProcessConfig process{71, true, {std::byte{3}, std::byte{'u'}, std::byte{'n'},
            std::byte{'i'}, std::byte{'t'}, std::byte{3}, std::byte{'w'}, std::byte{'a'},
            std::byte{'i'}, std::byte{'t'}}};
        auto lifetime = std::make_shared<int>(7);
        server_lifetime = lifetime;
        sm = std::make_shared<ServiceManagerSession>(std::move(process),
            [this, lifetime](std::uint64_t pid, const ServiceName& name) {
                CHECK(*lifetime == 7);
                ++connections;
                CHECK(pid == 71);
                if (name == service_name("wait") && !available)
                    return LookupResult{LookupKind::pending, 0, 0};
                const auto handle = handles.insert(client);
                CHECK(handle.has_value());
                return LookupResult{LookupKind::connected, handle.value_or(0), 0};
            });
        registers.fill(0x1234);
        registers[1] = 0x2000;
        CHECK(dispatcher.dispatch(0x1f, registers, context) == IpcDisposition::handled);
        CHECK(registers[0] == 0);
        sm_handle = static_cast<Handle>(registers[1]);
        CHECK(handles.lookup_as<ServiceManagerSession>(sm_handle) == sm);
    }

    static bool read_byte(void*, std::uint64_t address, std::byte& byte) noexcept {
        constexpr char port[] = "sm:";
        if (address < 0x2000 || address - 0x2000 >= sizeof(port)) return false;
        byte = static_cast<std::byte>(port[address - 0x2000]);
        return true;
    }

    static SessionConnection connect_port(void* opaque, std::string_view port) {
        auto& self = *static_cast<Fixture*>(opaque);
        CHECK(port == "sm:");
        return {IpcDisposition::handled, 0, self.sm};
    }

    IpcDisposition send(Handle handle) {
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, context);
    }

    void register_client() {
        message.fill(std::byte{0});
        put(message, 0, 0x8000000000000010ULL, 8);
        put(message, 8, 1, 4);
        // Deliberately forged wire PID; the trusted transport PID is 71.
        put(message, 12, 999, 8);
        CHECK(send(sm_handle) == IpcDisposition::handled && registers[0] == 0);
        CHECK(get(message, 0, 8) == 0x100000010ULL && get(message, 8, 4) == 0);
    }
};

void test_real_session_publication() {
    Fixture f;
    f.message = lookup("unit");
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled);
    CHECK(get(f.message, 16, 4) == 0x415 && f.connections == 0);
    f.register_client();
    f.message = lookup("deny");
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled);
    CHECK(get(f.message, 16, 4) == 0x1015 && get(f.message, 12, 4) == 0 && f.connections == 0);
    f.message = lookup("unit");
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled && f.registers[0] == 0);
    CHECK(get(f.message, 0, 8) == 0x8000000100000011ULL && get(f.message, 8, 4) == 0x20);
    CHECK(get(f.message, 16, 4) == 0 && f.connections == 1);
    const auto client_handle = static_cast<Handle>(get(f.message, 12, 4));
    CHECK(f.handles.lookup_as<SyntheticClient>(client_handle) == f.client);
    f.message.fill(std::byte{0});
    f.message[0] = std::byte{0xa5};
    CHECK(f.send(client_handle) == IpcDisposition::handled);
    CHECK(f.message[0] == std::byte{0x5a} && f.client->requests == 1);
}

void test_deferred_notification() {
    Fixture f;
    f.register_client();
    f.message = lookup("wait");
    f.registers.fill(0xabcdef);
    f.registers[0] = f.sm_handle;
    const auto saved_message = f.message;
    const auto saved_registers = f.registers;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.context) == IpcDisposition::pending);
    CHECK(f.message == saved_message && f.registers == saved_registers && f.connections == 1);
    const auto unrelated = f.sm->service_registered(service_name("other"));
    CHECK(unrelated.disposition == IpcDisposition::unsupported && unrelated.size == 0);
    CHECK(f.connections == 1);
    // A repeated call is not an implicit registration notification or retry.
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.context) == IpcDisposition::unsupported);
    CHECK(f.connections == 1 && f.message == saved_message && f.registers == saved_registers);
    f.available = true;
    const auto reply = f.sm->service_registered(service_name("wait"));
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 20);
    CHECK(get(reply.bytes, 16, 4) == 0 && f.connections == 2);
    CHECK(f.handles.lookup_as<SyntheticClient>(static_cast<Handle>(get(reply.bytes, 12, 4))) == f.client);
    // Notification returns a reply; only the scheduler may complete the saved call.
    CHECK(f.message == saved_message && f.registers == saved_registers);
    CHECK(f.sm->service_registered(service_name("wait")).disposition == IpcDisposition::unsupported);
    CHECK(f.connections == 2);
}

void test_explicit_unsupported() {
    Fixture f;
    f.register_client();
    f.message = lookup("unit");
    f.context.process_id = 72;
    CHECK(f.send(f.sm_handle) == IpcDisposition::unsupported && f.sm->diagnostic() != nullptr);
    CHECK(f.connections == 0);
    f.context.process_id = 71;
    f.message[4] = std::byte{3};
    const auto saved = f.message;
    CHECK(f.send(f.sm_handle) == IpcDisposition::unsupported && f.sm->diagnostic() != nullptr);
    CHECK(f.message == saved && f.connections == 0);
    f.message = lookup("unit");
    CHECK(f.send(f.sm_handle) == IpcDisposition::unsupported && f.connections == 0);
}
void test_remote_close_lifecycle() {
    Fixture f;
    f.register_client();
    f.message = lookup("unit");
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled);
    const auto client = static_cast<Handle>(get(f.message, 12, 4));
    CHECK(f.handles.lookup(client) == f.client && !f.server_lifetime.expired());
    f.message = {}; put(f.message, 0, 2, 4);
    const auto close_message = f.message;
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled && f.registers[0] == 0xf601);
    CHECK(f.message == close_message && f.handles.lookup(f.sm_handle) == f.sm);
    CHECK(f.server_lifetime.expired());
    const auto connections = f.connections;
    f.message = lookup("unit");
    const auto request = f.message;
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled && f.registers[0] == 0xf601);
    CHECK(f.message == request && f.connections == connections);
    f.message = {}; put(f.message, 0, 0x8000000000000010ULL, 8); put(f.message, 8, 1, 4);
    const auto registration = f.message;
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled && f.registers[0] == 0xf601);
    CHECK(f.message == registration);
    // Services already handed to the caller are independent of the SM session.
    f.message.fill(std::byte{0xa5});
    CHECK(f.send(client) == IpcDisposition::handled && f.registers[0] == 0);
    CHECK(f.handles.close(f.sm_handle));
    f.message = lookup("unit");
    CHECK(f.send(f.sm_handle) == IpcDisposition::handled && f.registers[0] == 0xe401);

    Fixture pending;
    pending.register_client();
    pending.message = lookup("wait");
    CHECK(pending.send(pending.sm_handle) == IpcDisposition::pending);
    pending.message = close_message;
    CHECK(pending.send(pending.sm_handle) == IpcDisposition::handled && pending.registers[0] == 0xf601);
    pending.available = true;
    const auto retry = pending.sm->service_registered(service_name("wait"));
    CHECK(retry.disposition == IpcDisposition::handled && retry.transport_result == 0xf601);
    CHECK(pending.connections == 1 && pending.server_lifetime.expired());
}
}  // namespace

int main() {
    test_real_session_publication();
    test_deferred_notification();
    test_explicit_unsupported();
    test_remote_close_lifecycle();
    if (failures) return 1;
    std::puts("service-manager session checks passed");
    return 0;
}
