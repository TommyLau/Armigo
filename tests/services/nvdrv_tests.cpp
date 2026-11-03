// ============================================================================
// Armigo Tests - Nvidia Driver Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/presentation.h>
#include <armigo/services/nvdrv.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace {
using armigo::horizon::HandleTable;
using armigo::horizon::IpcCallContext;
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcDispatcher;
using armigo::horizon::IpcMessage;
using armigo::horizon::IpcReply;
using armigo::horizon::Handle;
using armigo::services::nvdrv::DriverSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

std::uint32_t get(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}
void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

struct Fixture {
    std::shared_ptr<DriverSession> session = std::make_shared<DriverSession>();
    HandleTable handles{4};
    IpcDispatcher dispatcher{handles, [](void*, std::string_view) {
        return armigo::horizon::SessionConnection{};
    }};
    IpcMessage message{};
    armigo::horizon::QueryRegisters registers{};
    IpcCallContext caller{};
    Handle handle = 0;
    std::vector<std::byte> path_backing = std::vector<std::byte>(64);
    static bool read_range(void*, std::uint64_t, std::span<std::byte> out) noexcept {
        std::fill(out.begin(), out.end(), std::byte{0});
        return true;
    }
    static bool write_range(void*, std::uint64_t, std::span<const std::byte>) noexcept {
        return true;
    }
    Fixture() {
        handle = *handles.insert(session);
        caller.process_id = 5;
        caller.read_memory = read_range;
        caller.write_memory = write_range;
        caller.buffer_region_base = 0;
    }
    IpcReply send() {
        registers.fill(0);
        registers[0] = handle;
        return session->request(message, 5);
    }
};
// Buffered requests need real guest memory: the argument buffer holds the sizes
// and addresses the ioctl reads, and the description buffer receives the answer.
struct BufferFixture {
    std::shared_ptr<DriverSession> session = std::make_shared<DriverSession>();
    HandleTable handles{4};
    armigo::horizon::IpcDispatcher dispatcher{handles, [](void*, std::string_view) {
        return armigo::horizon::SessionConnection{};
    }};
    IpcMessage message{};
    IpcCallContext caller{};
    Handle handle = 0;
    static constexpr std::uint64_t base = 0x7030000000ull;
    std::vector<std::byte> memory = std::vector<std::byte>(0x1000);
    bool fail_read = false;
    bool fail_write = false;
    static bool read_memory(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<BufferFixture*>(raw);
        if (self.fail_read || address < base || address - base + out.size() > self.memory.size())
            return false;
        std::memcpy(out.data(), self.memory.data() + (address - base), out.size());
        return true;
    }
    static bool write_memory(void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& self = *static_cast<BufferFixture*>(raw);
        if (self.fail_write || address < base || address - base + in.size() > self.memory.size())
            return false;
        std::memcpy(self.memory.data() + (address - base), in.data(), in.size());
        return true;
    }
    BufferFixture() {
        handle = *handles.insert(session);
        caller.process_id = 5;
        caller.message = message;
        caller.read_memory = read_memory;
        caller.write_memory = write_memory;
        caller.memory_context = this;
        caller.buffer_region_base = base;
    }
    static void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
        std::memcpy(message.data() + offset, &value, 4);
    }
    // A counted HIPC request: one pointer descriptor (empty), one send buffer and
    // two receive buffers, followed by the CMIF header at byte64.
    // The packed descriptor word carries the guest address bits above 32.
    static std::uint32_t packed(std::uint64_t address) {
        return static_cast<std::uint32_t>(((address >> 36) & 7) << 2 |
                                          ((address >> 32) & 0xf) << 28);
    }
    void build_ioctl3(std::uint32_t fd, std::uint32_t request, std::uint32_t first_size,
                      std::uint64_t first_address, std::uint32_t second_size,
                      std::uint64_t second_address) {
        message.fill(std::byte{0});
        put(message, 0, 0x02110006);
        put(message, 4, 0x100b);
        put(message, 16, first_size);
        put(message, 20, static_cast<std::uint32_t>(first_address));
        put(message, 24, packed(first_address));
        put(message, 28, first_size);
        put(message, 32, static_cast<std::uint32_t>(first_address));
        put(message, 36, packed(first_address));
        put(message, 40, second_size);
        put(message, 44, static_cast<std::uint32_t>(second_address));
        put(message, 48, packed(second_address));
        put(message, 64, 0x49434653);
        put(message, 68, 1);
        put(message, 72, 12);
        put(message, 80, fd);
        put(message, 84, request);
    }
    // Dispatch through the runtime so the request's buffer descriptors are
    // decoded exactly as the guest built them.
    bool dispatch() {
        registers.fill(0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled;
    }
    IpcReply reply() const {
        IpcReply result{};
        result.disposition = IpcDisposition::handled;
        result.size = 64;
        std::memcpy(result.bytes.data(), message.data(), result.size);
        return result;
    }
    armigo::horizon::QueryRegisters registers{};
    struct MapperState {
        std::uint64_t mapped_address = 0;
        std::uint64_t mapped_size = 0;
        std::uint64_t unmapped_address = 0;
        unsigned map_calls = 0;
        std::shared_ptr<unsigned> unmap_calls = std::make_shared<unsigned>(0);
    };
    std::shared_ptr<MapperState> mapper_state = std::make_shared<MapperState>();
    static bool map(void* raw, std::uint64_t address, std::uint64_t size) noexcept {
        auto& state = *static_cast<MapperState*>(raw);
        state.mapped_address = address;
        state.mapped_size = size;
        ++state.map_calls;
        return true;
    }
    static void unmap(void* raw, std::uint64_t address, std::uint64_t) noexcept {
        auto& state = *static_cast<MapperState*>(raw);
        state.unmapped_address = address;
        ++*state.unmap_calls;
    }
    armigo::horizon::GuestMemoryMapper mapper() {
        return armigo::horizon::GuestMemoryMapper{map, unmap, mapper_state.get(), mapper_state};
    }
};

}  // namespace


void test_gpu_channel() {
    BufferFixture b;
    const auto open = [&](const char* path) {
        b.message.fill(std::byte{0});
        BufferFixture::put(b.message, 0, 6);
        BufferFixture::put(b.message, 4, 12);
        BufferFixture::put(b.message, 16, 0x00100001);
        BufferFixture::put(b.message, 20, 1);
        BufferFixture::put(b.message, 32, 0x49434653);
        BufferFixture::put(b.message, 36, 1);
        BufferFixture::put(b.message, 40, 0);
        std::memcpy(b.message.data() + 48, path, std::strlen(path));
        b.dispatch();
        return get(b.message, 32);
    };
    const auto ctrl = open("/dev/nvhost-ctrl-gpu");
    const auto address_space = open("/dev/nvhost-as-gpu");
    const auto channel = open("/dev/nvhost-gpu");
    const auto nvmap = open("/dev/nvmap");
    CHECK(ctrl == 1 && address_space == 2 && channel == 3 && nvmap == 4);

    // Bind the GPU channel to the address space.
    BufferFixture::put(b.message, 0, 0);
    std::memcpy(b.memory.data() + 0x100, &channel, 4);
    b.build_ioctl3(address_space, 0x40044101, 4, BufferFixture::base + 0x100, 0, 0);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 0);
    CHECK(b.session->device_bound_channel(address_space) == channel);
    // Binding an unknown descriptor is a bad parameter.
    const std::uint32_t unknown = 0x40;
    std::memcpy(b.memory.data() + 0x100, &unknown, 4);
    b.build_ioctl3(address_space, 0x40044101, 4, BufferFixture::base + 0x100, 0, 0);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 3);

    // ZCull context size and information.
    std::memset(b.memory.data() + 0x120, 0, 4);
    b.build_ioctl3(ctrl, 0x80044701, 4, BufferFixture::base + 0x120, 4, BufferFixture::base + 0x120);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 0);
    CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + 0x120) == 1);
    std::memset(b.memory.data() + 0x140, 0, 0x28);
    b.build_ioctl3(ctrl, 0x80284702, 0x28, BufferFixture::base + 0x140, 0x28, BufferFixture::base + 0x140);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 0);
    CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + 0x140) == 0x20);

    // The channel binds the nvmap descriptor and allocates its GPFIFO.
    std::memcpy(b.memory.data() + 0x160, &nvmap, 4);
    b.build_ioctl3(channel, 0x40044801, 4, BufferFixture::base + 0x160, 0, 0);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 0);
    std::memset(b.memory.data() + 0x180, 0, 0x20);
    const std::uint32_t entries = 0x8000;
    const std::uint32_t jobs = 1;
    std::memcpy(b.memory.data() + 0x180, &entries, 4);
    std::memcpy(b.memory.data() + 0x184, &jobs, 4);
    b.build_ioctl3(channel, 0xc020481a, 0x20, BufferFixture::base + 0x180, 0x20, BufferFixture::base + 0x180);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 0);
    CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + 0x180 + 12) == 24);

    // A reservation wrap can make an unsigned syncpoint fence expire without
    // advancing its completed value. That state change must wake host waiters.
    const auto reserve = [&](std::uint32_t increment) {
        std::memset(b.memory.data() + 0x1c0, 0, 0x18);
        const std::uint32_t reserve_flags = 0x104;
        std::memcpy(b.memory.data() + 0x1cc, &reserve_flags, 4);
        std::memcpy(b.memory.data() + 0x1d4, &increment, 4);
        b.build_ioctl3(channel, 0xc018481b, 0x18,
                       BufferFixture::base + 0x1c0, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
    };
    reserve(1);
    CHECK(b.session->is_fence_expired(24, 1) == false);

    auto access = b.session->graphics_backing_access(nullptr);
    std::promise<void> wait_started;
    auto wait_ready = wait_started.get_future();
    std::stop_source stop;
    auto waiter = std::async(std::launch::async, [&] {
        wait_started.set_value();
        return access.wait_fence({24, 1}, std::nullopt, stop.get_token());
    });
    wait_ready.wait();
    CHECK(waiter.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);
    reserve(0xffffffffu);
    CHECK(b.session->is_fence_expired(24, 1) == true);
    const bool reservation_woke_waiter =
        waiter.wait_for(std::chrono::seconds(1)) ==
        std::future_status::ready;
    if (!reservation_woke_waiter) stop.request_stop();
    CHECK(waiter.wait_for(std::chrono::seconds(1)) ==
          std::future_status::ready);
    const auto waited = waiter.get();
    CHECK(reservation_woke_waiter &&
          std::holds_alternative<std::monostate>(waited));

    // A submission without its separate inline entries is rejected before execution.
    std::memset(b.memory.data() + 0x1c0, 0, 0x18);
    const std::uint64_t address = 0x12340000;
    const std::uint32_t submitted = 2;
    const std::uint32_t flags = 0x104;
    std::memcpy(b.memory.data() + 0x1c0, &address, 8);
    std::memcpy(b.memory.data() + 0x1c8, &submitted, 4);
    std::memcpy(b.memory.data() + 0x1cc, &flags, 4);
    b.build_ioctl3(channel, 0xc018481b, 0x18, BufferFixture::base + 0x1c0, 0, 0);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 10);
    CHECK(b.session->diagnostic() == nullptr);
}

void test_shared_client_memory() {
    BufferFixture b;
    b.message.fill(std::byte{0});
    put(b.message, 0, 6);
    put(b.message, 4, 12);
    put(b.message, 16, 0x49434653);
    put(b.message, 24, 0);
    std::memcpy(b.message.data() + 32, "/dev/nvmap", 11);
    CHECK(b.dispatch());
    const auto fd = get(b.message, 32);
    const auto parent_handle = b.handle;
    const auto allocate = [&](std::uint32_t size) {
        std::memcpy(b.memory.data() + 0x100, &size, 4);
        b.build_ioctl3(fd, 0xc0080101, 8, BufferFixture::base + 0x100, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        std::uint32_t id = 0;
        std::memcpy(&id, b.memory.data() + 0x104, 4);
        return id;
    };
    const auto first = allocate(0x2000);
    auto cloned = std::make_shared<DriverSession>(*b.session);
    const auto clone_handle = *b.handles.insert(cloned);
    b.handle = clone_handle;
    const auto second = allocate(0x3000);
    CHECK(second != first);
    const auto query_size = [&](std::uint32_t id) {
        const std::uint32_t property = 1;
        const std::uint32_t guard = 0xa5b6c7d8u;
        std::memcpy(b.memory.data() + 0x140, &id, 4);
        std::memcpy(b.memory.data() + 0x144, &property, 4);
        std::memcpy(b.memory.data() + 0x14c, &guard, 4);
        b.build_ioctl3(fd, 0xc00c0109, 12, BufferFixture::base + 0x140, 0, 0);
        CHECK(b.dispatch());
        std::uint32_t size = 0;
        std::memcpy(&size, b.memory.data() + 0x148, 4);
        std::uint32_t kept_handle = 0, kept_property = 0, kept_guard = 0;
        std::memcpy(&kept_handle, b.memory.data() + 0x140, 4);
        std::memcpy(&kept_property, b.memory.data() + 0x144, 4);
        std::memcpy(&kept_guard, b.memory.data() + 0x14c, 4);
        CHECK(kept_handle == id && kept_property == property && kept_guard == guard);
        return size;
    };
    CHECK(query_size(first) == 0x2000);
    CHECK(get(b.message, 32) == 0);
    b.handle = parent_handle;
    CHECK(query_size(second) == 0x3000);
    CHECK(get(b.message, 32) == 0);
    std::memset(b.memory.data() + 0x180, 0, 24);
    std::memcpy(b.memory.data() + 0x180, &first, 4);
    b.build_ioctl3(fd, 0xc0180105, 24, BufferFixture::base + 0x180, 0, 0);
    CHECK(b.dispatch());
    CHECK(get(b.message, 32) == 0);
    b.handle = clone_handle;
    query_size(first);
    CHECK(get(b.message, 32) != 0);
}

// Multimedia channels (NVDEC): NVMap binding, a channel syncpoint, no wait
// base, a submit timeout, and buffers pinned at 32-bit device addresses with
// one pin per MapBuffer. Submit reads the command buffers from the NVMap
// handles' memory, writes each syncpoint increment's fence threshold back and
// runs the commands on the engine, which completes the syncpoint.
void test_multimedia_channel() {
    BufferFixture b;
    b.memory.resize(0x4000);
    // The dispatcher hands every request the caller's memory mapper.
    b.caller.memory_mapper = b.mapper();
    const auto open = [&](const char* path) {
        b.message.fill(std::byte{0});
        put(b.message, 0, 6);
        put(b.message, 4, 12);
        put(b.message, 16, 0x49434653);
        put(b.message, 24, 0);
        std::memcpy(b.message.data() + 32, path, std::strlen(path) + 1);
        CHECK(b.dispatch());
        return get(b.message, 32);
    };
    const auto nvmap = open("/dev/nvmap");
    const auto nvdec = open("/dev/nvhost-nvdec");
    const auto ioctl = [&](std::uint32_t fd, std::uint32_t request, std::size_t offset, std::uint32_t size) {
        b.build_ioctl3(fd, request, size, BufferFixture::base + offset, 0, 0);
        CHECK(b.dispatch());
        return get(b.message, 32);
    };
    const auto word = [&](std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, b.memory.data() + offset, 4);
        return value;
    };
    const auto set = [&](std::size_t offset, std::uint32_t value) { std::memcpy(b.memory.data() + offset, &value, 4); };
    // An NVMap handle with guest memory.
    set(0x100, 0x2000);
    CHECK(ioctl(nvmap, 0xc0080101, 0x100, 8) == 0);
    const auto handle = word(0x104);
    std::memset(b.memory.data() + 0x200, 0, 0x20);
    set(0x200, handle);
    set(0x20c, 0x1000);
    const std::uint64_t guest = BufferFixture::base + 0x2000;
    std::memcpy(b.memory.data() + 0x218, &guest, 8);
    CHECK(ioctl(nvmap, 0xc0200104, 0x200, 0x20) == 0);

    set(0x300, nvmap);
    CHECK(ioctl(nvdec, 0x40044801, 0x300, 4) == 0);
    set(0x310, 0);
    set(0x314, 0);
    CHECK(ioctl(nvdec, 0xc0080002, 0x310, 8) == 0);
    const auto syncpoint = word(0x314);
    CHECK(syncpoint != 0);
    set(0x314, 0);
    CHECK(ioctl(nvdec, 0xc0080002, 0x310, 8) == 0 && word(0x314) == syncpoint);
    set(0x320, 0);
    set(0x324, 0xffffffffu);
    CHECK(ioctl(nvdec, 0xc0080003, 0x320, 8) == 0 && word(0x324) == 0);
    set(0x330, 5000);
    CHECK(ioctl(nvdec, 0x40040007, 0x330, 4) == 0);
    // Two pins share one device address; the third release names no pin.
    const auto map = [&](std::uint32_t command, std::uint32_t target) {
        set(0x400, 1);
        set(0x404, 0);
        set(0x408, 0);
        set(0x40c, target);
        set(0x410, 0xdeadbeef);
        return ioctl(nvdec, 0xc0140000 | command, 0x400, 0x14);
    };
    CHECK(map(0x09, handle) == 0);
    const auto address = word(0x410);
    CHECK(address != 0 && address != 0xdeadbeef && address % 0x1000 == 0);
    CHECK(map(0x09, handle) == 0 && word(0x410) == address);
    CHECK(map(0x0a, handle) == 0 && word(0x410) == 0);
    CHECK(map(0x0a, handle) == 0);
    CHECK(map(0x0a, handle) != 0);
    CHECK(map(0x09, 0x7777) != 0);
    // A freed handle's device address is reused by the next pin that fits.
    const auto backed = [&]() {
        set(0x100, 0x2000);
        CHECK(ioctl(nvmap, 0xc0080101, 0x100, 8) == 0);
        const auto created = word(0x104);
        std::memset(b.memory.data() + 0x200, 0, 0x20);
        set(0x200, created);
        set(0x20c, 0x1000);
        std::memcpy(b.memory.data() + 0x218, &guest, 8);
        CHECK(ioctl(nvmap, 0xc0200104, 0x200, 0x20) == 0);
        return created;
    };
    const auto temporary = backed();
    CHECK(map(0x09, temporary) == 0);
    const auto temporary_address = word(0x410);
    CHECK(temporary_address != address);
    std::memset(b.memory.data() + 0x180, 0, 24);
    set(0x180, temporary);
    CHECK(ioctl(nvmap, 0xc0180105, 0x180, 24) == 0);
    const auto reuser = backed();
    CHECK(map(0x09, reuser) == 0 && word(0x410) == temporary_address);
    // One command buffer: select the decoder class, then increment the
    // channel syncpoint through the host interface.
    set(0x2000, 0xf0u << 6);
    set(0x2004, 0x40000000u | syncpoint);
    std::memset(b.memory.data() + 0x500, 0, 0x40);
    set(0x500, 1);           // command buffers
    set(0x508, 1);           // syncpoint increments
    set(0x50c, 1);           // fences
    set(0x510, handle);      // command buffer: handle, offset, words
    set(0x518, 2);
    set(0x51c, syncpoint);   // increment: id, count
    set(0x520, 1);
    set(0x530, 0xdeadbeef);  // fence threshold, written back
    CHECK(ioctl(nvdec, 0xc0340001, 0x500, 0x34) == 0);
    CHECK(word(0x530) == 1 && word(0x500) == 1 && word(0x51c) == syncpoint);
    const auto ctrl = open("/dev/nvhost-ctrl");
    bool completed = false;
    for (int attempt = 0; attempt < 200 && !completed; ++attempt) {
        set(0x600, syncpoint);
        set(0x604, 0);
        CHECK(ioctl(ctrl, 0xc0080014, 0x600, 8) == 0);
        completed = word(0x604) == 1;
        if (!completed) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(completed);
    // A command buffer outside its handle changes nothing.
    set(0x518, 0x2000);
    set(0x530, 0xdeadbeef);
    CHECK(ioctl(nvdec, 0xc0340001, 0x500, 0x34) != 0);
    CHECK(word(0x530) == 0xdeadbeef);
    set(0x600, syncpoint);
    CHECK(ioctl(ctrl, 0xc008001a, 0x600, 8) == 0 && word(0x604) == 1);
}

int main() {
    test_shared_client_memory();
    test_multimedia_channel();
    Fixture f;
    // Control: report the pointer buffer size, then convert to a domain.
    f.message.fill(std::byte{0});
    put(f.message, 0, 7);
    put(f.message, 4, 8);
    put(f.message, 16, 0x49434653);
    put(f.message, 24, 3);
    auto reply = f.send();
    CHECK(reply.disposition == IpcDisposition::handled);
    put(f.message, 24, 0);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(f.session->domain_object_count() == 1);

    // Open a device with an inline path and receive a descriptor.
    const auto domain_command = [&](std::uint32_t command, std::uint32_t argument) {
        f.message.fill(std::byte{0});
        put(f.message, 0, 6);
        put(f.message, 4, 12);
        put(f.message, 16, 0x00100001);
        put(f.message, 20, 1);
        put(f.message, 32, 0x49434653);
        put(f.message, 36, 1);
        put(f.message, 40, command);
        put(f.message, 48, argument);
    };
    domain_command(0, 0);
    std::memcpy(f.message.data() + 48, "/dev/nvhost-ctrl", 17);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::handled);
    // The client reads the descriptor and its result word directly after the
    // reply header, so the SFCO word sits at byte16 and the data at byte32.
    CHECK(reply.size == 40);
    CHECK(get(reply.bytes, 4) == 8);
    CHECK(get(reply.bytes, 16) == 0x4f434653);
    CHECK(get(reply.bytes, 32) == 1);
    CHECK(get(reply.bytes, 36) == 0);
    CHECK(f.session->device_path(1) == "/dev/nvhost-ctrl");

    // A second open gets its own descriptor.
    domain_command(0, 0);
    std::memcpy(f.message.data() + 48, "/dev/nvmap", 11);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 32) == 2);

    // An ioctl the service does not implement is answered with the driver result
    // word the firmware uses, together with a diagnostic naming the device and the
    // request, so the guest decides whether the operation is optional.
    domain_command(1, 1);
    put(f.message, 52, 0x4e000000);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.size == 36);
    CHECK(get(reply.bytes, 16) == 0x4f434653);
    CHECK(get(reply.bytes, 32) == 1);
    CHECK(f.session->diagnostic() != nullptr &&
          std::string(f.session->diagnostic()).find("/dev/nvhost-ctrl") != std::string::npos);
    CHECK(f.session->diagnostic() != nullptr &&
          std::string(f.session->diagnostic()).find("0x4e000000") != std::string::npos);
    // An ioctl on an unknown descriptor is a caller error, so it stays a refusal.
    domain_command(1, 99);
    put(f.message, 52, 0x4e000000);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::unsupported);

    // A session clone continues the same client, so it sees the parent's
    // descriptors and continues the descriptor numbering: the guest keeps one
    // descriptor table per client and dispatches every request against it.
    f.message.fill(std::byte{0});
    put(f.message, 0, 7);
    put(f.message, 4, 8);
    put(f.message, 16, 0x49434653);
    put(f.message, 24, 2);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    CHECK(reply.transferred[0] != nullptr);
    auto clone = std::static_pointer_cast<DriverSession>(reply.transferred[0]);
    CHECK(clone != nullptr);
    if (clone) {
        CHECK(clone->device_path(2) == "/dev/nvmap");
        // An ioctl issued on the clone resolves the descriptor opened through the
        // parent, and the clone's own refusal names the same device.
        f.message.fill(std::byte{0});
        put(f.message, 0, 6);
        put(f.message, 4, 12);
        put(f.message, 16, 0x00100001);
        put(f.message, 20, 1);
        put(f.message, 32, 0x49434653);
        put(f.message, 36, 1);
        put(f.message, 40, 1);
        put(f.message, 48, 2);
        put(f.message, 52, 0x4e000001);
        auto cloned_reply = clone->request(f.message, 5);
        CHECK(cloned_reply.disposition == IpcDisposition::handled);
        CHECK(get(cloned_reply.bytes, 32) == 1);
        CHECK(clone->diagnostic() != nullptr &&
              std::string(clone->diagnostic()).find("/dev/nvmap") != std::string::npos);
        // A clone's open continues the shared numbering.
        f.message.fill(std::byte{0});
        put(f.message, 0, 6);
        put(f.message, 4, 12);
        put(f.message, 16, 0x00100001);
        put(f.message, 20, 1);
        put(f.message, 32, 0x49434653);
        put(f.message, 36, 1);
        put(f.message, 40, 0);
        std::memcpy(f.message.data() + 48, "/dev/nvhost-as-gpu", 19);
        cloned_reply = clone->request(f.message, 5);
        CHECK(cloned_reply.disposition == IpcDisposition::handled);
        CHECK(get(cloned_reply.bytes, 32) == 3);
        CHECK(f.session->device_path(3) == "/dev/nvhost-as-gpu");
    }

    // GPU control ioctls: the driver opens the GPU control node and asks for the
    // GPU description and the cluster mask before it creates an address space.
    {
        BufferFixture b;
        b.message.fill(std::byte{0});
        put(b.message, 0, 6);
        put(b.message, 4, 12);
        put(b.message, 16, 0x00100001);
        put(b.message, 20, 1);
        put(b.message, 32, 0x49434653);
        put(b.message, 36, 1);
        put(b.message, 40, 0);
        std::memcpy(b.message.data() + 48, "/dev/nvhost-ctrl-gpu", 20);
        CHECK(b.dispatch());
        CHECK(b.session->device_path(1) == "/dev/nvhost-ctrl-gpu");

        // GetCharacteristics: the argument buffer names an output size and a zero
        // address, and the description lands in the separate receive buffer.
        const std::size_t arguments = 0x100;
        const std::size_t description = 0x400;
        std::vector<std::byte> header(16);
        const std::uint64_t size = 0xa0;
        std::memcpy(header.data(), &size, 8);
        std::memcpy(b.memory.data() + arguments, header.data(), header.size());
        b.build_ioctl3(1, 0xc0b04705, 0xb0, BufferFixture::base + arguments, 0xa0,
                       BufferFixture::base + description);
        CHECK(b.dispatch());
        auto reply = b.reply();
        CHECK(get(reply.bytes, 16) == 0x4f434653);
        CHECK(get(reply.bytes, 32) == 0);
        const auto* words = reinterpret_cast<const std::uint32_t*>(b.memory.data() + description);
        CHECK(words[0] == 0x120);     // architecture: GM20B
        CHECK(words[1] == 0xb);       // implementation
        CHECK(words[2] == 0xa1);      // revision
        CHECK(words[3] == 1);         // one GPC
        CHECK(words[4] == 0x40000);   // L2 cache size, 64-bit
        CHECK(words[6] == 0);         // no board memory
        CHECK(words[8] == 2);         // two TPCs per GPC
        CHECK(words[10] == 0x20000);  // big page size
        CHECK(words[14] == 1);        // GPC mask
        CHECK(words[15] == 0x503);    // SM architecture version
        // The chip name is the 64-bit word at 0x90; "gm20b" reads low byte first.
        const auto* chip = reinterpret_cast<const std::uint32_t*>(b.memory.data() + description + 0x90);
        CHECK(chip[0] == 0x30326d67u);
        CHECK(chip[1] == 0x62u);
        const auto* written = reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments);
        CHECK(written[0] == 0xa0);

        // GetTpcMasks: the mask is written to the named buffer and to the argument
        // word the driver reads back.
        // The mask request names its output size and the mask buffer address.
        const auto mask_arguments = [&](std::size_t offset, std::uint32_t value) {
            std::memcpy(b.memory.data() + 0x200 + offset, &value, 4);
        };
        mask_arguments(0, 4);
        const auto mask_address = static_cast<std::uint64_t>(0x800);
        std::memcpy(b.memory.data() + 0x200 + 8, &mask_address, 8);
        b.build_ioctl3(1, 0xc0184706, 0x18, BufferFixture::base + 0x200, 4,
                       BufferFixture::base + 0x800);
        CHECK(b.dispatch());
        reply = b.reply();
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(get(reply.bytes, 32) == 0);
        const auto* mask = reinterpret_cast<const std::uint32_t*>(b.memory.data() + 0x800);
        CHECK(mask[0] == 3);
        const auto* written_arguments =
            reinterpret_cast<const std::uint32_t*>(b.memory.data() + 0x200);
        CHECK(written_arguments[4] == 3);
    }

    // The address-space node binds a range and then serves GPU virtual addresses
    // from it: the region query reports the small-page and big-page windows, and an
    // allocation answers with a bound address inside the matching window.
    {
        BufferFixture b;
        b.message.fill(std::byte{0});
        put(b.message, 0, 6);
        put(b.message, 4, 12);
        put(b.message, 16, 0x00100001);
        put(b.message, 20, 1);
        put(b.message, 32, 0x49434653);
        put(b.message, 36, 1);
        put(b.message, 40, 0);
        std::memcpy(b.message.data() + 48, "/dev/nvhost-as-gpu", 18);
        CHECK(b.dispatch());
        CHECK(b.session->device_path(1) == "/dev/nvhost-as-gpu");
        const std::uint64_t initialize = 0x300;
        const std::uint64_t regions = 0x400;
        const std::uint64_t allocation = 0x500;
        // Initialize: {flags, reserved, big page size, reserved, start, end, split}.
        const std::uint64_t big_page_size = 0x20000;
        const std::uint64_t range_start = 0x8000000;
        const std::uint64_t range_end = 1ull << 37;
        const std::uint64_t range_split = 1ull << 34;
        std::memcpy(b.memory.data() + initialize + 8, &big_page_size, 8);
        std::memcpy(b.memory.data() + initialize + 16, &range_start, 8);
        std::memcpy(b.memory.data() + initialize + 24, &range_end, 8);
        std::memcpy(b.memory.data() + initialize + 32, &range_split, 8);
        b.build_ioctl3(1, 0x40284109, 0x28, BufferFixture::base + initialize, 0x28,
                       BufferFixture::base + initialize);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);

        b.build_ioctl3(1, 0xc0404108, 0x40, BufferFixture::base + regions, 0,
                       BufferFixture::base + regions);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const auto* region = reinterpret_cast<const std::uint64_t*>(b.memory.data() + regions);
        CHECK(region[0] == 0);              // unused
        CHECK(region[1] == 0x30);           // two regions fit in the buffer
        CHECK(region[2] == range_start);    // small-page window
        CHECK(region[3] == 0x1000);         // small page size, with its padding
        CHECK(region[4] == (range_split - range_start) / 0x1000);
        CHECK(region[5] == range_split);    // big-page window
        CHECK(region[6] == big_page_size);
        CHECK(region[7] == (range_end - range_split) / big_page_size);

        // AllocSpace: {pages, page size, flags, padding, offset or alignment}.
        const auto allocate = [&](std::uint32_t pages, std::uint32_t page_size,
                                  std::uint32_t flags, std::uint64_t offset_or_alignment) {
            std::memset(b.memory.data() + allocation, 0, 0x18);
            std::memcpy(b.memory.data() + allocation + 0, &pages, 4);
            std::memcpy(b.memory.data() + allocation + 4, &page_size, 4);
            std::memcpy(b.memory.data() + allocation + 8, &flags, 4);
            std::memcpy(b.memory.data() + allocation + 16, &offset_or_alignment, 8);
            b.build_ioctl3(1, 0xc0184102, 0x18, BufferFixture::base + allocation, 0x18,
                           BufferFixture::base + allocation);
            CHECK(b.dispatch());
            return *reinterpret_cast<const std::uint64_t*>(b.memory.data() + allocation + 16);
        };
        CHECK(get(b.message, 32) == 0);
        const auto first = allocate(2, 0x1000, 0, 0x1000);
        CHECK(first == range_start);
        const auto second = allocate(1, 0x1000, 0, 0x1000);
        CHECK(second == range_start + 2 * 0x1000);
        const auto big = allocate(1, static_cast<std::uint32_t>(big_page_size), 0,
                                  big_page_size);
        CHECK(big == range_split);
        // A fixed binding keeps the address the guest selected and refuses one that
        // is already bound.
        const auto fixed = allocate(1, 0x1000, 1, range_start + 4 * 0x1000);
        CHECK(fixed == range_start + 4 * 0x1000);
        allocate(1, 0x1000, 1, range_start + 4 * 0x1000);
        CHECK(get(b.message, 32) == 9);
        // A released run is served again before the cursor grows.
        allocate(2, 0x1000, 0, 0x1000);

        // MapBufferEx binds a memory handle at a GPU address and reports it back.
        // The handle comes from the memory node, which is opened for it here.
        b.message.fill(std::byte{0});
        put(b.message, 0, 6);
        put(b.message, 4, 12);
        put(b.message, 16, 0x00100001);
        put(b.message, 20, 1);
        put(b.message, 32, 0x49434653);
        put(b.message, 36, 1);
        put(b.message, 40, 0);
        std::memcpy(b.message.data() + 48, "/dev/nvmap", 10);
        CHECK(b.dispatch());
        const std::uint64_t handle_arguments = 0x700;
        std::memset(b.memory.data() + handle_arguments, 0, 8);
        const std::uint32_t handle_size = 0x2000;
        std::memcpy(b.memory.data() + handle_arguments, &handle_size, 4);
        b.build_ioctl3(2, 0xc0080101, 0x08, BufferFixture::base + handle_arguments, 0x08,
                       BufferFixture::base + handle_arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const auto created_handle =
            *reinterpret_cast<const std::uint32_t*>(b.memory.data() + handle_arguments + 4);
        // Allocation alignment defines the physical extent available for mapping.
        const std::uint64_t alloc_arguments = 0x740;
        const auto allocation_alignment = static_cast<std::uint32_t>(big_page_size);
        const std::uint64_t guest_backing = BufferFixture::base + 0x10000;
        std::memset(b.memory.data() + alloc_arguments, 0, 0x20);
        std::memcpy(b.memory.data() + alloc_arguments, &created_handle, 4);
        std::memcpy(b.memory.data() + alloc_arguments + 12, &allocation_alignment, 4);
        std::memcpy(b.memory.data() + alloc_arguments + 24, &guest_backing, 8);
        b.caller.memory_mapper = b.mapper();
        const std::uint32_t invalid_alignment = 0x3000;
        std::memcpy(b.memory.data() + alloc_arguments + 12, &invalid_alignment, 4);
        b.build_ioctl3(2, 0xc0200104, 0x20, BufferFixture::base + alloc_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 11);
        CHECK(b.mapper_state->mapped_address == 0);
        std::memcpy(b.memory.data() + alloc_arguments + 12, &allocation_alignment, 4);
        b.build_ioctl3(2, 0xc0200104, 0x20, BufferFixture::base + alloc_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const std::uint64_t map_arguments = 0x600;
        std::memset(b.memory.data() + map_arguments, 0, 0x28);
        const std::uint32_t fixed_flag = 1;
        const std::uint32_t memory_handle = created_handle;
        const std::uint32_t small_page = 0x1000;
        const std::uint64_t mapping_size = 0x2000;
        const std::uint64_t fixed_offset = range_start + 0x10000;
        std::memcpy(b.memory.data() + map_arguments + 0, &fixed_flag, 4);
        std::memcpy(b.memory.data() + map_arguments + 8, &memory_handle, 4);
        std::memcpy(b.memory.data() + map_arguments + 12, &small_page, 4);
        std::memcpy(b.memory.data() + map_arguments + 24, &mapping_size, 8);
        std::memcpy(b.memory.data() + map_arguments + 32, &fixed_offset, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0x28,
                       BufferFixture::base + map_arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const auto* mapped =
            reinterpret_cast<const std::uint64_t*>(b.memory.data() + map_arguments + 32);
        CHECK(*mapped == fixed_offset);

        // A mapping smaller than its page size covers one whole page, which is what
        // the console reports for a big-page binding.
        const std::uint32_t big_page = static_cast<std::uint32_t>(big_page_size);
        const std::uint64_t small_size = 0x1000;
        const std::uint64_t big_fixed = range_split + 0x20000;
        std::memset(b.memory.data() + map_arguments, 0, 0x28);
        std::memcpy(b.memory.data() + map_arguments + 0, &fixed_flag, 4);
        std::memcpy(b.memory.data() + map_arguments + 8, &memory_handle, 4);
        std::memcpy(b.memory.data() + map_arguments + 12, &big_page, 4);
        std::memcpy(b.memory.data() + map_arguments + 24, &small_size, 8);
        std::memcpy(b.memory.data() + map_arguments + 32, &big_fixed, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0x28,
                       BufferFixture::base + map_arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const auto* big_mapped =
            reinterpret_cast<const std::uint64_t*>(b.memory.data() + map_arguments + 32);
        CHECK(*big_mapped == big_fixed);

        // The requested GPU extent may include the final page's backing padding.
        const std::uint64_t padded_size = big_page_size;
        const std::uint64_t padded_address = big_fixed + big_page_size;
        std::memcpy(b.memory.data() + map_arguments + 24, &padded_size, 8);
        std::memcpy(b.memory.data() + map_arguments + 32, &padded_address, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        CHECK(*reinterpret_cast<const std::uint64_t*>(b.memory.data() + map_arguments + 32) == padded_address);
        const std::uint64_t excess_size = big_page_size * 2;
        const std::uint64_t excess_address = padded_address + big_page_size;
        std::memcpy(b.memory.data() + map_arguments + 24, &excess_size, 8);
        std::memcpy(b.memory.data() + map_arguments + 32, &excess_address, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) != 0);
        // Allocation padding also applies when the GPU mapping uses small pages.
        const std::uint64_t padded_small_address = range_start + 0x50000;
        std::memcpy(b.memory.data() + map_arguments + 12, &small_page, 4);
        std::memcpy(b.memory.data() + map_arguments + 24, &padded_size, 8);
        std::memcpy(b.memory.data() + map_arguments + 32, &padded_small_address, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const std::uint64_t padded_suboffset = 0x1000;
        const std::uint64_t padded_remainder = padded_size - padded_suboffset;
        const std::uint64_t padded_subaddress = padded_small_address + padded_size;
        std::memcpy(b.memory.data() + map_arguments + 16, &padded_suboffset, 8);
        std::memcpy(b.memory.data() + map_arguments + 24, &padded_remainder, 8);
        std::memcpy(b.memory.data() + map_arguments + 32, &padded_subaddress, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        // Nonzero backing offsets are retained and must fit the handle.
        const std::uint64_t buffer_offset = 0x1000;
        const std::uint64_t offset_mapping = range_start + 0x30000;
        std::memset(b.memory.data() + map_arguments, 0, 0x28);
        std::memcpy(b.memory.data() + map_arguments, &fixed_flag, 4);
        std::memcpy(b.memory.data() + map_arguments + 8, &memory_handle, 4);
        std::memcpy(b.memory.data() + map_arguments + 12, &small_page, 4);
        std::memcpy(b.memory.data() + map_arguments + 16, &buffer_offset, 8);
        std::memcpy(b.memory.data() + map_arguments + 24, &small_size, 8);
        std::memcpy(b.memory.data() + map_arguments + 32, &offset_mapping, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const std::uint64_t invalid_offset = 0x3000;
        std::memcpy(b.memory.data() + map_arguments + 16, &invalid_offset, 8);
        b.build_ioctl3(1, 0xc0284106, 0x28, BufferFixture::base + map_arguments, 0, 0);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) != 0);

    }

    // Memory handles: the driver creates a handle for a size and frees it later. An
    // allocation reports that it needs guest-backed memory instead of inventing an
    // address the guest would then fault on.
    {
        BufferFixture b;
        b.message.fill(std::byte{0});
        put(b.message, 0, 6);
        put(b.message, 4, 12);
        put(b.message, 16, 0x00100001);
        put(b.message, 20, 1);
        put(b.message, 32, 0x49434653);
        put(b.message, 36, 1);
        put(b.message, 40, 0);
        std::memcpy(b.message.data() + 48, "/dev/nvmap", 10);
        CHECK(b.dispatch());
        CHECK(b.session->device_path(1) == "/dev/nvmap");

        const std::uint64_t arguments = 0x100;
        const auto write_size = [&](std::uint32_t size) {
            std::memset(b.memory.data() + arguments, 0, 0x20);
            std::memcpy(b.memory.data() + arguments, &size, 4);
        };
        write_size(0x1234);
        b.build_ioctl3(1, 0xc0080101, 0x08, BufferFixture::base + arguments, 0x08,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        const auto* created = reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments);
        CHECK(created[0] == 0x1234);
        CHECK(created[1] == 4);
        const auto created_handle = created[1];
        CHECK(!b.session->retain_exported_memory(created_handle));
        CHECK(!b.session->retain_exported_memory(0x400));

        // GetId exports a created object without requiring allocation. The id is
        // stable, the input handle is preserved, and a clone sees the same client.
        const auto get_id = [&](Handle dispatch_handle, std::uint32_t handle,
                                std::uint32_t size = 8) {
            std::memset(b.memory.data() + arguments, 0xa5, 8);
            std::memcpy(b.memory.data() + arguments + 4, &handle, 4);
            const auto saved_handle = b.handle;
            b.handle = dispatch_handle;
            b.build_ioctl3(1, 0xc008010e, size, BufferFixture::base + arguments, size,
                           BufferFixture::base + arguments);
            CHECK(b.dispatch());
            b.handle = saved_handle;
            return get(b.message, 32);
        };
        CHECK(get_id(b.handle, created_handle) == 0);
        CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments) ==
              created_handle);
        CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments + 4) ==
              created_handle);
        CHECK(get_id(b.handle, created_handle) == 0);
        auto clone = std::make_shared<DriverSession>(*b.session);
        const auto clone_handle = *b.handles.insert(clone);
        CHECK(get_id(clone_handle, created_handle) == 0);
        CHECK(!b.session->retain_exported_memory(created_handle));

        // Firmware distinguishes zero from an unknown nonzero client handle.
        CHECK(get_id(b.handle, 0) == 11);
        CHECK(get_id(b.handle, 0x400) == 0x30010);
        CHECK(get_id(b.handle, created_handle, 4) == 10);

        // Guest copy failures retain their established invalid-address result.
        b.fail_read = true;
        CHECK(get_id(b.handle, created_handle) == 9);
        b.fail_read = false;
        b.fail_write = true;
        CHECK(get_id(b.handle, created_handle) == 9);
        b.fail_write = false;
        CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments) ==
              0xa5a5a5a5u);
        CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments + 4) ==
              created_handle);

        // A zero size is refused.
        write_size(0);
        b.build_ioctl3(1, 0xc0080101, 0x08, BufferFixture::base + arguments, 0x08,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 10);

        // Allocation names the handle and the address the driver selected, and the
        // runtime maps the handle's memory there.
        write_size(4);
        const auto requested = static_cast<std::uint64_t>(BufferFixture::base + 0x800);
        std::memcpy(b.memory.data() + arguments + 24, &requested, 8);
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 1);
        CHECK(b.session->diagnostic() != nullptr &&
              std::string(b.session->diagnostic()).find("guest bytes at") != std::string::npos);

        // With the runtime's mapper the allocation succeeds and names the address.
        // The driver's requested alignment is kept for the parameter query.
        const std::uint32_t requested_alignment = 0x02000000;
        std::memcpy(b.memory.data() + arguments + 12, &requested_alignment, 4);
        b.caller.memory_mapper = b.mapper();
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        CHECK(b.mapper_state->mapped_address == requested);
        CHECK(b.mapper_state->mapped_size == 0x2000);
        const auto* bound = reinterpret_cast<const std::uint64_t*>(b.memory.data() + arguments + 24);
        CHECK(*bound == requested);
        // Param reports the alignment the driver asked for.
        const std::uint64_t param_arguments = 0x120;
        std::memset(b.memory.data() + param_arguments, 0, 0x0c);
        const std::uint32_t align_parameter = 2;
        std::memcpy(b.memory.data() + param_arguments, &created_handle, 4);
        std::memcpy(b.memory.data() + param_arguments + 4, &align_parameter, 4);
        b.build_ioctl3(1, 0xc00c0109, 0x0c, BufferFixture::base + param_arguments, 0x0c,
                       BufferFixture::base + param_arguments);
        CHECK(b.dispatch());
        const auto* reported_alignment =
            reinterpret_cast<const std::uint32_t*>(b.memory.data() + param_arguments + 8);
        CHECK(*reported_alignment == requested_alignment);
        CHECK(get_id(b.handle, created_handle) == 0);
        b.session->set_memory_mapper({});
        clone->set_memory_mapper({});
        auto retained = b.session->retain_exported_memory(created_handle);
        auto clone_retained = clone->retain_exported_memory(created_handle);
        CHECK(retained && clone_retained);
        if (retained && clone_retained) {
            CHECK(retained->export_id == created_handle && retained->size == 0x2000 &&
                  retained->alignment == requested_alignment &&
                  retained->guest_address == requested);
            CHECK(retained->owner == clone_retained->owner);
        }
        const auto retained_map_calls = b.mapper_state->map_calls;
        const std::uint32_t repeat_alignment = 0x1000;
        const std::uint32_t repeat_flags = 0x40;
        const std::uint64_t repeat_address = BufferFixture::base + 0x4000;
        std::memset(b.memory.data() + arguments, 0, 0x20);
        std::memcpy(b.memory.data() + arguments, &created_handle, 4);
        std::memcpy(b.memory.data() + arguments + 8, &repeat_flags, 4);
        std::memcpy(b.memory.data() + arguments + 12, &repeat_alignment, 4);
        std::memcpy(b.memory.data() + arguments + 24, &repeat_address, 8);
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0 && b.mapper_state->map_calls == retained_map_calls);
        CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments + 12) ==
              repeat_alignment);
        CHECK(*reinterpret_cast<const std::uint64_t*>(b.memory.data() + arguments + 24) ==
              repeat_address);
        CHECK(retained && retained->alignment == requested_alignment &&
              retained->guest_address == requested);
        const std::uint32_t zero_alignment = 0;
        const std::uint64_t zero_address = 0;
        std::memcpy(b.memory.data() + arguments + 8, &zero_alignment, 4);
        std::memcpy(b.memory.data() + arguments + 12, &zero_alignment, 4);
        std::memcpy(b.memory.data() + arguments + 24, &zero_address, 8);
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0 && b.mapper_state->map_calls == retained_map_calls);
        CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments + 8) == 2);
        CHECK(*reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments + 12) ==
              0x1000);
        const std::uint32_t invalid_repeat_alignment = 0x3000;
        std::memcpy(b.memory.data() + arguments + 12, &invalid_repeat_alignment, 4);
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 11 && b.mapper_state->map_calls == retained_map_calls);
        const auto unmap_calls = b.mapper_state->unmap_calls;
        const std::weak_ptr<BufferFixture::MapperState> original_mapper = b.mapper_state;
        b.caller.memory_mapper = {};

        // Free releases the handle, and a second free is a successful no-op.
        std::memset(b.memory.data() + arguments, 0, 0x18);
        std::memcpy(b.memory.data() + arguments, &created_handle, 4);
        b.build_ioctl3(1, 0xc0180105, 0x18, BufferFixture::base + arguments, 0x18,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        CHECK(*unmap_calls == 0);
        CHECK(!b.session->retain_exported_memory(created_handle));
        std::array<std::byte, 0x18> freed_arguments{};
        std::memcpy(freed_arguments.data(), b.memory.data() + arguments, freed_arguments.size());
        b.build_ioctl3(1, 0xc0180105, 0x18, BufferFixture::base + arguments, 0x18,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(get(b.message, 32) == 0);
        CHECK(std::memcmp(freed_arguments.data(), b.memory.data() + arguments,
                          freed_arguments.size()) == 0);
        CHECK(*unmap_calls == 0);
        CHECK(get_id(b.handle, created_handle) == 0x30010);

        // An ordinary unretained handle keeps the direct Free unmap behavior.
        write_size(0x1000);
        b.build_ioctl3(1, 0xc0080101, 0x08, BufferFixture::base + arguments, 0x08,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        const auto ordinary_handle =
            *reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments + 4);
        CHECK(get_id(b.handle, ordinary_handle) == 0);
        const std::uint32_t ordinary_alignment = 0x1000;
        const std::uint64_t ordinary_address = BufferFixture::base + 0x2000;
        std::memset(b.memory.data() + arguments, 0, 0x20);
        std::memcpy(b.memory.data() + arguments, &ordinary_handle, 4);
        std::memcpy(b.memory.data() + arguments + 12, &ordinary_alignment, 4);
        std::memcpy(b.memory.data() + arguments + 24, &ordinary_address, 8);
        b.caller.memory_mapper = {BufferFixture::map, BufferFixture::unmap,
                                  b.mapper_state.get(), {}};
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(!b.session->retain_exported_memory(ordinary_handle));
        const auto map_calls = b.mapper_state->map_calls;
        b.caller.memory_mapper = b.mapper();
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(b.mapper_state->map_calls == map_calls);
        CHECK(!b.session->retain_exported_memory(ordinary_handle));
        std::memset(b.memory.data() + arguments, 0, 0x18);
        std::memcpy(b.memory.data() + arguments, &ordinary_handle, 4);
        b.build_ioctl3(1, 0xc0180105, 0x18, BufferFixture::base + arguments, 0x18,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(*unmap_calls == 1 && b.mapper_state->unmapped_address == ordinary_address);

        // The handle itself owns its lease: dropping the only external pin does
        // not unmap until Free erases the descriptor.
        write_size(0x1000);
        b.build_ioctl3(1, 0xc0080101, 0x08, BufferFixture::base + arguments, 0x08,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        const auto handle_owned =
            *reinterpret_cast<const std::uint32_t*>(b.memory.data() + arguments + 4);
        CHECK(get_id(b.handle, handle_owned) == 0);
        const std::uint64_t handle_owned_address = BufferFixture::base + 0x6000;
        std::memset(b.memory.data() + arguments, 0, 0x20);
        std::memcpy(b.memory.data() + arguments, &handle_owned, 4);
        std::memcpy(b.memory.data() + arguments + 12, &ordinary_alignment, 4);
        std::memcpy(b.memory.data() + arguments + 24, &handle_owned_address, 8);
        b.caller.memory_mapper = b.mapper();
        b.build_ioctl3(1, 0xc0200104, 0x20, BufferFixture::base + arguments, 0x20,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        auto handle_pin = b.session->retain_exported_memory(handle_owned);
        CHECK(handle_pin);
        handle_pin.reset();
        CHECK(*unmap_calls == 1);
        std::memset(b.memory.data() + arguments, 0, 0x18);
        std::memcpy(b.memory.data() + arguments, &handle_owned, 4);
        b.build_ioctl3(1, 0xc0180105, 0x18, BufferFixture::base + arguments, 0x18,
                       BufferFixture::base + arguments);
        CHECK(b.dispatch());
        CHECK(*unmap_calls == 2 &&
              b.mapper_state->unmapped_address == handle_owned_address);

        b.caller.memory_mapper = {};
        b.session->set_memory_mapper({});
        b.mapper_state.reset();
        CHECK(!original_mapper.expired());
        retained.reset();
        CHECK(*unmap_calls == 2);
        clone_retained.reset();
        CHECK(*unmap_calls == 3);
        CHECK(original_mapper.expired());
    }

    test_gpu_channel();

    // Closing the descriptor succeeds once and is refused afterwards.
    domain_command(2, 1);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::handled);
    domain_command(2, 1);
    reply = f.send();
    CHECK(reply.disposition == IpcDisposition::unsupported);

    if (failures == 0) std::fprintf(stderr, "Nvidia driver service failures: 0\n");
    return failures == 0 ? 0 : 1;
}
