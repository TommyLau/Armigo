// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/ipc.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace armigo::horizon;
namespace {
unsigned failures = 0;
#define CHECK(x) do { if (!(x)) { ++failures; \
    std::fprintf(stderr, "Line %u failed: %s\n", __LINE__, #x); } } while (false)

struct SyntheticSession : ClientSession {
    IpcDisposition disposition = IpcDisposition::handled;
    std::uint64_t pid = 0;
    IpcMessage received{};
    std::size_t size = 4;
    std::uint32_t result = 0;
    unsigned calls = 0;
    std::array<std::shared_ptr<armigo::horizon::KernelObject>, armigo::horizon::max_transferred_handles> transferred{};
    std::size_t transferred_count = 0;
    bool transferred_as_copy = false;
    std::size_t buffer_count = 0;
    std::uint64_t buffer_address = 0;
    std::uint32_t buffer_size = 0;
    std::string buffer_path;
    std::atomic<unsigned> closes{0};
    IpcReply close_server() override {
        ++closes;
        return {.disposition = IpcDisposition::handled};
    }
    IpcReply request_with_buffers(const IpcMessage& input, std::uint64_t caller,
                                  const GuestBuffers& buffers) override {
        buffer_count = buffers.size();
        if (!buffers.empty()) {
            buffer_address = buffers[0].address;
            buffer_size = buffers[0].size;
            if (const auto path = buffers.read_string(0)) buffer_path = *path;
        }
        return request(input, caller);
    }
    IpcReply request(const IpcMessage& input, std::uint64_t caller) override {
        received = input;
        pid = caller;
        ++calls;
        IpcReply reply{disposition, result};
        reply.bytes.fill(std::byte{0x39});
        reply.size = size;
        reply.transferred = transferred;
        reply.transferred_count = transferred_count;
        reply.transferred_as_copy = transferred_as_copy;
        return reply;
    }
};

struct ObservedSession : ForwardingSession {
    using ForwardingSession::ForwardingSession;
    unsigned ordinary_calls = 0;
    unsigned buffered_calls = 0;
    IpcReply request(const IpcMessage& message, std::uint64_t pid) override {
        ++ordinary_calls;
        return ForwardingSession::request(message, pid);
    }
    IpcReply request_with_buffers(const IpcMessage& message, std::uint64_t pid,
                                  const GuestBuffers& buffers) override {
        ++buffered_calls;
        return ForwardingSession::request_with_buffers(message, pid, buffers);
    }
};

struct Fixture {
    HandleTable handles{2};
    std::shared_ptr<SyntheticSession> session = std::make_shared<SyntheticSession>();
    std::array<std::byte, 12> name{};
    IpcMessage message{};
    IpcDispatcher dispatcher{handles, connect, this};
    std::uint64_t address = 0x100000;
    std::array<std::byte, 16>* guest = nullptr;
    IpcCallContext caller{0x12345, message, read, this};
    unsigned connects = 0;
    static bool read(void* raw, std::uint64_t at, std::byte& value) noexcept {
        const auto& self = *static_cast<Fixture*>(raw);
        if (at < self.address || at - self.address >= self.name.size()) return false;
        value = self.name[at - self.address];
        return true;
    }
    static SessionConnection connect(void* raw, std::string_view name) {
        auto& self = *static_cast<Fixture*>(raw);
        ++self.connects;
        if (name != "sm:") return {IpcDisposition::handled, 0xf201, {}};
        return {IpcDisposition::handled, 0, self.session};
    }
    Fixture() { std::memcpy(name.data(), "sm:", 3); }
    QueryRegisters registers() const {
        QueryRegisters result;
        result.fill(0xaaaaaaaaffffffff);
        result[1] = address;
        return result;
    }
};

void connection_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0 && f.handles.lookup_as<ClientSession>(r[1]) == f.session);
    CHECK(std::all_of(r.begin() + 2, r.end(), [](auto x) { return x == 0; }));
    CHECK(f.handles.close(r[1]));
    f.name.fill(std::byte{0x41});
    r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0xee01 && r[1] == f.address && f.connects == 1);
    r = f.registers();
    r[1] = f.address + f.name.size();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0xe601);
    f.name.fill(std::byte{});
    r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0xf201 && r[1] == f.address);
    std::memcpy(f.name.data(), "sm:", 4);
    CHECK(f.handles.insert(f.session).has_value());
    CHECK(f.handles.insert(f.session).has_value());
    r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0xd201 && f.handles.size() == 2);
    f.name.fill(std::byte{});
    r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0xd201); // The handle is reserved before the port lookup.
}

void request_tests() {
    Fixture f;
    const auto handle = *f.handles.insert(f.session);
    auto r = f.registers();
    r[0] = 0xffffffff00000000ull | handle;
    f.message.fill(std::byte{0xf5});
    // Keep a descriptor-free HIPC header while testing opaque payload/tail
    // forwarding. Random header bits would request real copied handles.
    std::fill_n(f.message.begin(), 8, std::byte{});
    f.message[0] = std::byte{6};
    const auto original = f.message;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(f.session->pid == f.caller.process_id && f.session->received == original);
    CHECK(std::all_of(r.begin(), r.end(), [](auto x) { return x == 0; }));
    CHECK(f.message[0] == std::byte{0x39} && f.message[3] == std::byte{0x39});
    CHECK(std::equal(original.begin() + 4, original.end(), f.message.begin() + 4));
    for (auto status : {IpcDisposition::pending, IpcDisposition::unsupported}) {
        f.session->disposition = status;
        f.message = original;
        r = f.registers();
        r[0] = handle;
        const auto before = r;
        CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == status);
        CHECK(r == before && f.message == original);
    }
    f.session->disposition = IpcDisposition::handled;
    f.session->size = ipc_message_capacity + 1;
    r = f.registers(); r[0] = handle;
    const auto before = r;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::invalid_backend);
    CHECK(r == before && f.message == original);
    CHECK(f.handles.close(handle));
    r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0xe401 && r[1] == 0);
    // HIPC Close (type 2) tears down the server side without reaching the service and
    // completes as session-closed. The client's own handle stays open until it closes
    // it with svcCloseHandle.
    const auto closing = *f.handles.insert(f.session);
    const auto received = f.session->received;
    f.message.fill(std::byte{});
    std::memcpy(f.message.data(), "\x02\x00\x00\x00", 4);
    r = f.registers();
    r[0] = closing;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == session_closed_result && r[1] == 0);
    CHECK(f.handles.lookup(closing) == f.session);
    CHECK(f.session->received == received);
    // A later request never reaches the closed server.
    f.message.fill(std::byte{});
    r = f.registers();
    r[0] = closing;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == session_closed_result && f.session->received == received);
    // TIPC CloseSession (method 15) closes the same way and completes as
    // session-closed.
    std::memcpy(f.message.data(), "\x0f\x00\x00\x00", 4);
    r = f.registers();
    r[0] = closing;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == session_closed_result);
    CHECK(f.handles.lookup(closing) == f.session && f.session->received == received);
    // Only the first close runs the server's close hook.
    CHECK(f.session->closes == 1);
    CHECK(f.handles.close(closing));
    r = f.registers();
    r[0] = closing;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0xe401);
    r = f.registers();
    const auto unchanged = r;
    CHECK(f.dispatcher.dispatch(0x20, r, f.caller) == IpcDisposition::unsupported);
    CHECK(r == unchanged);
}

void copied_pseudo_handle_tests() {
    Fixture f;
    const auto wrapper = std::make_shared<ObservedSession>(f.session);
    const auto session_handle = *f.handles.insert(wrapper);
    CHECK(!wrapper->receives_copied_objects());
    for (const auto pseudo : {current_process_handle, current_thread_handle}) {
        f.message.fill(std::byte{});
        const auto put = [&](std::size_t offset, std::uint32_t value) {
            std::memcpy(f.message.data() + offset, &value, sizeof(value));
        };
        put(0, 6);
        put(4, 0x8000000a);
        put(8, 3); // Send PID and one copied pseudo-handle.
        put(20, pseudo);
        put(32, 0x49434653);
        const auto original = f.message;
        auto registers = f.registers();
        registers[0] = session_handle;
        const auto calls = f.session->calls;
        CHECK(f.dispatcher.dispatch(0x21, registers, f.caller) == IpcDisposition::handled);
        CHECK(registers[0] == 0);
        CHECK(f.session->calls == calls + 1 && f.session->received == original);
        CHECK(wrapper->ordinary_calls == 0);
        CHECK(wrapper->buffered_calls == calls + 1);
    }
}

void forwarding_buffer_tests() {
    auto inner = std::make_shared<SyntheticSession>();
    ObservedSession wrapper(inner);
    IpcMessage message{};
    GuestBuffers buffers;
    buffers.add({.address = 0x12000, .size = 64, .input = true});
    const auto reply = wrapper.request_with_handles(message, 42, buffers, {});
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(wrapper.ordinary_calls == 0 && wrapper.buffered_calls == 1);
    CHECK(inner->pid == 42 && inner->received == message);
    CHECK(inner->buffer_count == 1 && inner->buffer_address == 0x12000 &&
          inner->buffer_size == 64);
}
void buffer_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    // A buffer-carrying request: PID-variant domain header at byte32, CMIF at 48,
    // one descriptor at byte8 {size, address low bits}. The runtime supplies the
    // region base whose low 32 bits are implied.
    std::array<std::byte, 16> guest{};
    std::memcpy(guest.data(), "rom:/test", 9);
    IpcMessage request;
    request.fill(std::byte{0});
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i != 4; ++i)
            request[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    put(0, 0x01000006);
    put(4, 0x10);
    put(8, static_cast<std::uint32_t>(guest.size()));
    put(12, static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(guest.data())));
    put(32, 0x00200001);
    put(36, 2);
    put(48, 0x49434653);
    put(52, 1);
    auto caller = f.caller;
    const auto region = reinterpret_cast<std::uint64_t>(guest.data()) & ~0xFFFFFFFFull;
    caller.buffer_region_base = 0;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* fixture = static_cast<Fixture*>(raw);
        const auto* base = fixture->guest->data();
        const auto offset = address - reinterpret_cast<std::uint64_t>(base);
        if (offset > 16 || out.size() > 16 - offset) return false;
        std::memcpy(out.data(), base + offset, out.size());
        return true;
    };
    caller.memory_context = &f;
    caller.buffer_region_base = region;
    f.guest = &guest;
    f.message = request;
    f.session->buffer_count = 0;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    CHECK(f.session->buffer_count == 1);
    CHECK(f.session->buffer_address == reinterpret_cast<std::uint64_t>(guest.data()));
    CHECK(f.session->buffer_size == guest.size());
    CHECK(f.session->buffer_path == "rom:/test");
}

void packed_map_alias_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    // The file-system storage read declares one map-alias receive. Its packed
    // descriptor word carries address bits 32..39; the runtime must use them
    // instead of OR-ing the low word with the arena base, because the same low
    // word means different addresses in the arena and the heap.
    std::array<std::byte, 16> guest{};
    std::memcpy(guest.data(), "rom:/test", 9);
    constexpr std::uint64_t guest_address = 0x7100000000ull;
    IpcMessage request;
    request.fill(std::byte{0});
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i != 4; ++i)
            request[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    const auto packed = static_cast<std::uint32_t>(((guest_address >> 36) & 7) << 2 |
                                                   ((guest_address >> 32) & 0xf) << 28);
    CHECK(packed == 0x1000001cu);
    put(0, 0x01000006);
    put(4, 0x10);
    put(8, static_cast<std::uint32_t>(guest.size()));
    put(12, static_cast<std::uint32_t>(guest_address & 0xFFFFFFFFu));
    put(16, packed);
    put(32, 0x00200001);
    put(36, 2);
    put(48, 0x49434653);
    put(52, 1);
    auto caller = f.caller;
    caller.buffer_region_base = 0x7000000000ull;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* fixture = static_cast<Fixture*>(raw);
        const auto base = reinterpret_cast<std::uint64_t>(fixture->guest->data());
        if (address < 0x7100000000ull || address - 0x7100000000ull > 16 ||
            out.size() > 16 - (address - 0x7100000000ull))
            return false;
        std::memcpy(out.data(), fixture->guest->data() + (address - 0x7100000000ull), out.size());
        (void)base;
        return true;
    };
    caller.memory_context = &f;
    f.guest = &guest;
    f.message = request;
    f.session->buffer_count = 0;
    f.session->buffer_path.clear();
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    CHECK(f.session->buffer_count == 1);
    CHECK(f.session->buffer_address == guest_address);
    CHECK(f.session->buffer_size == guest.size());
    CHECK(f.session->buffer_path == "rom:/test");
}

void inline_descriptor_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    std::array<std::byte, 16> guest{};
    IpcMessage request;
    request.fill(std::byte{0});
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i != 4; ++i)
            request[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    // A request whose HIPC header declares an inline descriptor (buffer-C flag in bits
    // 10..13): the two-word descriptor follows the CMIF header at byte56.
    put(0, 6);
    put(4, 0x00000c0d);
    put(16, 0x00100001);
    put(20, 1);
    put(32, 0x49434653);
    put(36, 1);
    put(56, static_cast<std::uint32_t>(guest.size()));
    put(60, static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(guest.data())));
    auto caller = f.caller;
    caller.buffer_region_base = reinterpret_cast<std::uint64_t>(guest.data()) & ~0xFFFFFFFFull;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* fixture = static_cast<Fixture*>(raw);
        const auto* base = fixture->guest->data();
        const auto offset = address - reinterpret_cast<std::uint64_t>(base);
        if (offset > 16 || out.size() > 16 - offset) return false;
        std::memcpy(out.data(), base + offset, out.size());
        return true;
    };
    caller.memory_context = &f;
    f.guest = &guest;
    f.message = request;
    f.session->buffer_count = 0;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    CHECK(f.session->buffer_count == 1);
    CHECK(f.session->buffer_address == reinterpret_cast<std::uint64_t>(guest.data()));
    CHECK(f.session->buffer_size == guest.size());
    // Without the flag the same words are command data, not a descriptor.
    request[4] = std::byte{0x0c};
    request[5] = std::byte{0};
    f.message = request;
    f.session->buffer_count = 0;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    CHECK(f.session->buffer_count == 0);
}

void stale_inline_descriptor_tests() {
    // A domain request can leave an earlier SFCI header in the first inline
    // descriptor slot; the real {size, address} pair follows. The dispatcher
    // must skip the magic-sized pair and keep the plausible buffer.
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    std::array<std::byte, 16> guest{};
    IpcMessage request;
    request.fill(std::byte{0});
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i != 4; ++i)
            request[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    put(0, 6);
    put(4, 0x00000c00);  // inline descriptor flag in bits 10..13
    put(32, 0x49434653);
    put(36, 1);
    put(40, 2);
    put(48, 0x49434653);  // stale pair: size looks like the SFCI magic
    put(52, 1);
    put(56, static_cast<std::uint32_t>(guest.size()));
    put(60, static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(guest.data())));
    auto caller = f.caller;
    caller.buffer_region_base = reinterpret_cast<std::uint64_t>(guest.data()) & ~0xFFFFFFFFull;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* fixture = static_cast<Fixture*>(raw);
        const auto base = reinterpret_cast<std::uint64_t>(fixture->guest->data());
        const auto offset = address - (base & ~0xFFFFFFFFull);
        if (offset > 16 || out.size() > 16 - offset) return false;
        std::memcpy(out.data(), fixture->guest->data() + offset, out.size());
        return true;
    };
    caller.memory_context = &f;
    f.guest = &guest;
    f.message = request;
    f.session->buffer_count = 0;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    CHECK(f.session->buffer_count == 1);
    CHECK(f.session->buffer_size == guest.size());
    CHECK(f.session->buffer_address == reinterpret_cast<std::uint64_t>(guest.data()));
}

void counted_buffer_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    // A counted HIPC request: pointer count 1, send count 1, receive count 2 and
    // eleven raw words. An empty pointer descriptor precedes one send and two
    // receive descriptors, and the CMIF header follows them at byte64.
    std::array<std::byte, 0x10> guest{};
    IpcMessage request;
    request.fill(std::byte{0});
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i != 4; ++i)
            request[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    const auto low = [&](std::size_t offset) {
        return static_cast<std::uint32_t>(
            (reinterpret_cast<std::uintptr_t>(guest.data()) + offset) & 0xFFFFFFFFu);
    };
    // The packed descriptor word carries the address bits above 32 and the send
    // and receive attributes.
    const auto packed = static_cast<std::uint32_t>(
        (((reinterpret_cast<std::uintptr_t>(guest.data()) >> 36) & 7) << 2) |
        (((reinterpret_cast<std::uintptr_t>(guest.data()) >> 32) & 0xf) << 28));
    put(0, 0x02110006);
    put(4, 0x100b);
    put(16, 0x18);
    put(20, low(0x00));
    put(24, packed);
    put(28, 0x18);
    put(32, low(0x00));
    put(36, packed);
    put(40, 0x04);
    put(44, low(0x08));
    put(48, packed);
    put(64, 0x49434653);
    put(68, 1);
    put(72, 12);
    auto caller = f.caller;
    caller.buffer_region_base = 0;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* fixture = static_cast<Fixture*>(raw);
        const auto* base = fixture->guest->data();
        const auto offset = address - reinterpret_cast<std::uint64_t>(base);
        if (offset > 0x10 || out.size() > 0x10 - offset) return false;
        std::memcpy(out.data(), base + offset, out.size());
        return true;
    };
    caller.write_memory = [](void*, std::uint64_t, std::span<const std::byte>) noexcept {
        return true;
    };
    caller.memory_context = &f;
    f.guest = &guest;
    f.message = request;
    f.session->buffer_count = 0;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    // The empty pointer descriptor is skipped and the three named buffers survive.
    CHECK(f.session->buffer_count == 3);
    CHECK(f.session->buffer_address == reinterpret_cast<std::uint64_t>(guest.data()));
}

// The file-system storage read shape: word0 declares one receive (B)
// descriptor, word1 has no special header, and the descriptor carries an
// address whose packed word sets mode NonSecure and address bits 38..36 while
// bits 35..32 stay zero. The decoded address must come from the
// packed bits alone; a nonzero region base must not be OR-ed in.
void storage_read_heap_descriptor_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    constexpr std::uint32_t size = 0x00060000;
    constexpr std::uint32_t low = 0x12345000u;
    constexpr std::uint64_t address = (std::uint64_t{5} << 36) | low;
    IpcMessage request;
    request.fill(std::byte{0});
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i != 4; ++i)
            request[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    put(0, 0x01000006);
    put(4, 0x00000010);
    put(8, size);
    put(12, low);
    put(16, 0x00000015);  // mode 1 (NonSecure), address bits 38..36 = 5, bits 35..32 = 0
    put(32, 0x00200001);  // domain header: SendMessage, 0x20 words
    put(36, 3);           // storage object id
    put(48, 0x49434653);
    put(52, 1);
    put(56, 0);           // IStorage::Read
    put(64, 0x1000);      // offset
    put(72, size);        // length
    auto caller = f.caller;
    // A region base that would produce a different address: the packed bits
    // must win even though the mid field (bits 35..32) is zero.
    caller.buffer_region_base = 0x1000000000ull;
    caller.read_memory = [](void*, std::uint64_t, std::span<std::byte>) noexcept { return true; };
    caller.write_memory = [](void*, std::uint64_t, std::span<const std::byte>) noexcept {
        return true;
    };
    f.message = request;
    f.session->buffer_count = 0;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    CHECK(f.session->buffer_count == 1);
    CHECK(f.session->buffer_address == address);
    CHECK(f.session->buffer_size == size);
}

// A counted receive-buffer request with the PID special header, the shape the vi
// client uses for OpenLayer: word0 carries one receive buffer, word1 sets the
// special header, the special word declares a PID and no handles, and the
// three-word map-alias descriptor follows.
void receive_buffer_with_pid_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    std::array<std::byte, 0x10> guest{};
    IpcMessage request;
    request.fill(std::byte{0});
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned i = 0; i != 4; ++i)
            request[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
    };
    const auto ptr = reinterpret_cast<std::uintptr_t>(guest.data());
    const auto packed = static_cast<std::uint32_t>((((ptr >> 36) & 7) << 2) |
                                                   (((ptr >> 32) & 0xf) << 28));
    put(0, 0x01000006);
    put(4, 0x8000001c);
    put(8, 1);                       // special header: PID present, no handles
    put(12, 0);
    put(16, 0);
    put(20, 0x10);                   // buffer size
    put(24, static_cast<std::uint32_t>(ptr & 0xFFFFFFFFu));
    put(28, packed);
    put(32, 0x49434653);
    put(36, 1);
    put(40, 2020);
    auto caller = f.caller;
    caller.buffer_region_base = 0;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* fixture = static_cast<Fixture*>(raw);
        const auto* base = fixture->guest->data();
        const auto offset = address - reinterpret_cast<std::uint64_t>(base);
        if (offset > 0x10 || out.size() > 0x10 - offset) return false;
        std::memcpy(out.data(), base + offset, out.size());
        return true;
    };
    caller.write_memory = [](void*, std::uint64_t, std::span<const std::byte>) noexcept {
        return true;
    };
    caller.memory_context = &f;
    f.guest = &guest;
    f.message = request;
    f.session->buffer_count = 0;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, caller) == IpcDisposition::handled);
    CHECK(f.session->buffer_count == 1);
    CHECK(f.session->buffer_address == reinterpret_cast<std::uint64_t>(guest.data()));
    CHECK(f.session->buffer_size == 0x10);
}

void transfer_tests() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    // A reply that transfers one new session keeps the HIPC descriptor array at
    // byte8, declares one move handle in header bits20..23, and leaves the
    // CMIF header at byte16.
    auto child = std::make_shared<SyntheticSession>();
    f.session->transferred[0] = child;
    f.session->transferred_count = 1;
    f.session->size = 32;
    f.session->result = 0;
    f.message.fill(std::byte{0x5a});
    const auto before = f.message;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0);
    const auto word = [&](std::size_t offset) {
        std::uint32_t value = 0;
        for (unsigned i = 0; i != 4; ++i)
            value |= std::uint32_t{std::to_integer<unsigned char>(f.message[offset + i])} << (8 * i);
        return value;
    };
    // The special header at byte8 declares one moved handle; the SDK reads the
    // handle from byte20, and the authored CMIF header moves to byte32.
    const auto transferred = static_cast<Handle>(word(20));
    CHECK(transferred != 0 && transferred != handle);
    CHECK(word(0) == 0u && word(4) == 0x80000004u);
    CHECK(word(8) == 0x21u && word(12) == 0u);
    CHECK(word(16) == 0u && word(24) == 0u && word(28) == 0u);
    CHECK(word(32) == 0x39393939u && word(44) == 0x39393939u);
    CHECK(f.handles.lookup_as<SyntheticSession>(transferred) == child);
    CHECK(std::all_of(f.message.begin() + 48, f.message.end(),
                      [](std::byte b) { return b == std::byte{0x5a}; }));
    CHECK(f.handles.close(transferred));
    // A reply that copies an object the caller keeps (a shared memory) declares one
    // copied handle instead: the special header carries the copy count at bit1 and the
    // handle still travels in the same array slot.
    f.session->transferred_as_copy = true;
    f.session->transferred[0] = std::make_shared<SyntheticSession>();
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::handled);
    CHECK(r[0] == 0);
    const auto copied = static_cast<Handle>(word(20));
    CHECK(copied != 0 && copied != handle);
    CHECK(word(8) == 0x03u);
    CHECK(f.handles.lookup(copied) != nullptr);
    CHECK(f.handles.close(copied));
    f.session->transferred_as_copy = false;
    f.session->transferred[0] = nullptr;
    // An authored reply too short to hold the header is refused, and the
    // caller's message stays untouched.
    f.message = before;
    f.session->transferred[0] = std::make_shared<SyntheticSession>();
    f.session->size = 12;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::invalid_backend);
    CHECK(f.message == before);
    f.session->transferred[0] = nullptr;
    f.session->transferred_count = 0;
}

void transfer_limits() {
    Fixture f;
    auto r = f.registers();
    CHECK(f.dispatcher.dispatch(0x1f, r, f.caller) == IpcDisposition::handled);
    const auto handle = static_cast<Handle>(r[1]);
    f.session->transferred[0] = std::make_shared<SyntheticSession>();
    f.session->transferred_count = max_transferred_handles + 1;
    f.session->size = 32;
    f.message.fill(std::byte{0x5a});
    const auto before = f.message;
    r = f.registers(); r[0] = handle;
    CHECK(f.dispatcher.dispatch(0x21, r, f.caller) == IpcDisposition::invalid_backend);
    CHECK(f.message == before);
}
} // namespace

int main() {
    forwarding_buffer_tests();
    copied_pseudo_handle_tests();
    connection_tests();
    request_tests();
    transfer_tests();
    transfer_limits();
    buffer_tests();
    packed_map_alias_tests();
    stale_inline_descriptor_tests();
    inline_descriptor_tests();
    counted_buffer_tests();
    storage_read_heap_descriptor_tests();
    receive_buffer_with_pid_tests();
    return failures ? 1 : 0;
}
