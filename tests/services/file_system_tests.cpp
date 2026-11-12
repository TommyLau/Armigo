// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/romfs/romfs_virtual.h>
#include <armigo/services/addon_content_catalog.h>
#include <armigo/services/file_system.h>

#include <mach/mach.h>
#include <sys/clonefile.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <string>
#include <iterator>
#include <fstream>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace {
std::atomic<int> forced_clone_error{0};
std::atomic<unsigned> clone_attempts{0};
}

// Exercise real host clones and specific OS failure returns through the service.
extern "C" int clonefile(const char* source, const char* destination, std::uint32_t flags) {
    clone_attempts.fetch_add(1);
    if (const auto error = forced_clone_error.load()) {
        errno = error;
        return -1;
    }
    using Function = int (*)(const char*, const char*, std::uint32_t);
    static const auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "clonefile"));
    if (!original) { errno = ENOSYS; return -1; }
    return original(source, destination, flags);
}

namespace {
using namespace armigo::horizon;
using armigo::services::fs::FileSystemServiceSession;
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

struct Fixture {
    HandleTable handles{4};
    std::shared_ptr<FileSystemServiceSession> session =
        std::make_shared<FileSystemServiceSession>(71, 0x400);
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
};

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

// OpenDataStorageByDataId (202) opens an installed add-on's RomFS: its own size
// and bytes, beside the application's store from command 200.
void test_addon_data_storage() {
    std::vector<std::byte> base_bytes(100), addon_bytes(50);
    for (std::size_t index = 0; index < base_bytes.size(); ++index) base_bytes[index] = static_cast<std::byte>(index + 1);
    for (std::size_t index = 0; index < addon_bytes.size(); ++index) addon_bytes[index] = static_cast<std::byte>(index + 151);
    auto image = std::make_shared<armigo::romfs::MemoryStorage>(std::move(base_bytes));
    auto catalog = std::make_shared<armigo::services::AddOnContentCatalog>();
    catalog->base_id = 0x0100000000011000ull;
    catalog->items.push_back({catalog->base_id + 1, std::make_shared<armigo::romfs::MemoryStorage>(std::move(addon_bytes))});

    HandleTable handles{4};
    auto session = std::make_shared<FileSystemServiceSession>(71, 0, image, catalog);
    IpcDispatcher dispatcher{handles, nullptr};
    const auto handle = handles.insert(session).value();
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};
    std::vector<std::byte> guest(256);
    caller.buffer_region_base = reinterpret_cast<std::uintptr_t>(guest.data()) & ~0xFFFFFFFFull;
    caller.write_memory = [](void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || in.size() > base->size() - offset) return false;
        std::memcpy(base->data() + offset, in.data(), in.size());
        return true;
    };
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || out.size() > base->size() - offset) return false;
        std::memcpy(out.data(), base->data() + offset, out.size());
        return true;
    };
    caller.memory_context = &guest;
    const auto dispatch = [&] {
        registers.fill(0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    };
    message = control(0);
    CHECK(dispatch() == IpcDisposition::handled);

    const auto open_data = [&](std::uint64_t data_id) {
        message.fill(std::byte{0});
        put(message, 0, 6);
        put(message, 4, 16);
        put(message, 16, 0x00200001);
        put(message, 20, 1);
        put(message, 32, 0x49434653);
        put(message, 36, 1);
        put(message, 40, 202);
        put(message, 48, 0);  // storage id
        put(message, 56, static_cast<std::uint32_t>(data_id));
        put(message, 60, static_cast<std::uint32_t>(data_id >> 32));
        return dispatch();
    };
    const auto storage_size = [&](std::uint32_t object) {
        message.fill(std::byte{0});
        put(message, 0, 0x01000006);
        put(message, 4, 12);
        put(message, 32, 0x00200001);
        put(message, 36, object);
        put(message, 48, 0x49434653);
        put(message, 52, 1);
        put(message, 56, 4);
        CHECK(dispatch() == IpcDisposition::handled);
        return get(message, 48);
    };
    const auto storage_byte = [&](std::uint32_t object, std::uint64_t offset) {
        std::fill(guest.begin(), guest.end(), std::byte{0});
        message.fill(std::byte{0});
        put(message, 0, 0x01000006);
        put(message, 4, 16);
        put(message, 8, 1);
        put(message, 12, static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(guest.data())));
        put(message, 32, 0x00200001);
        put(message, 36, object);
        put(message, 48, 0x49434653);
        put(message, 52, 1);
        put(message, 56, 0);
        put(message, 64, static_cast<std::uint32_t>(offset));
        put(message, 72, 1);
        CHECK(dispatch() == IpcDisposition::handled);
        return std::to_integer<unsigned>(guest[0]);
    };
    CHECK(open_data(catalog->base_id + 1) == IpcDisposition::handled);
    const auto addon = get(message, 48);
    CHECK(addon == 2);
    CHECK(storage_size(addon) == 50);
    CHECK(storage_byte(addon, 0) == 151 && storage_byte(addon, 49) == 200);
    // A data id that is not installed stays explicit.
    CHECK(open_data(catalog->base_id + 2) == IpcDisposition::unsupported);
    // The application's own store is unchanged.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, 1);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 200);
    CHECK(dispatch() == IpcDisposition::handled);
    const auto application = get(message, 48);
    CHECK(application == 3);
    CHECK(storage_size(application) == 100 && storage_byte(application, 0) == 1);
}

void test_romfs_storage_read() {
    // A synthetic store of 100 bytes; the storage read copies what is asked and
    // reports the transferred byte count.
    std::vector<std::byte> bytes(100);
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::byte>(index + 1);
    auto image = std::make_shared<armigo::romfs::MemoryStorage>(std::move(bytes));

    HandleTable handles{4};
    auto session = std::make_shared<FileSystemServiceSession>(71, 0, image);
    IpcDispatcher dispatcher{handles, nullptr};
    const auto handle = handles.insert(session).value();
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};
    std::vector<std::byte> guest(256);
    caller.buffer_region_base = reinterpret_cast<std::uintptr_t>(guest.data()) & ~0xFFFFFFFFull;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || out.size() > base->size() - offset) return false;
        std::memcpy(out.data(), base->data() + offset, out.size());
        return true;
    };
    caller.write_memory = [](void* raw, std::uint64_t address,
                             std::span<const std::byte> in) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || in.size() > base->size() - offset) return false;
        std::memcpy(base->data() + offset, in.data(), in.size());
        return true;
    };
    caller.memory_context = &guest;

    // Convert the session to a domain first, as the generated client does.
    message = control(0);
    registers.fill(0);
    registers[0] = handle;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    CHECK(get(message, 32) == 1);

    // A store object only exists after the client opens one; the first open gets
    // domain object id 2.
    const auto open_store = [&] {
        message.fill(std::byte{0});
        put(message, 0, 6);
        put(message, 4, 12);
        put(message, 16, 0x00100001);
        put(message, 20, 1);
        put(message, 32, 0x49434653);
        put(message, 36, 1);
        put(message, 40, 200);
        put(message, 44, 0);
        registers.fill(0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    };
    CHECK(open_store() == IpcDisposition::handled);
    CHECK(get(message, 48) == 2);

    const auto storage_read = [&](std::uint64_t offset, std::uint32_t size, std::uint32_t buffer_bytes) {
        message.fill(std::byte{0});
        put(message, 0, 0x01000006);
        put(message, 4, 16);
        put(message, 8, buffer_bytes);
        put(message, 12, static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(guest.data())));
        put(message, 32, 0x00200001);
        put(message, 36, 2);
        put(message, 48, 0x49434653);
        put(message, 52, 1);
        put(message, 56, 0);
        put(message, 64, offset);
        put(message, 68, offset >> 32);
        put(message, 72, size);
        registers.fill(0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    };

    std::fill(guest.begin(), guest.end(), std::byte{0});
    CHECK(storage_read(0, 80, 80) == IpcDisposition::handled);
    CHECK(get(message, 48) == 80 && get(message, 52) == 0);
    CHECK(guest[0] == std::byte{1} && guest[79] == std::byte{80});
    // A read that runs past the image transfers only the remaining bytes.
    std::fill(guest.begin(), guest.end(), std::byte{0});
    CHECK(storage_read(90, 80, 80) == IpcDisposition::handled);
    CHECK(get(message, 48) == 10 && get(message, 52) == 0);
    CHECK(guest[0] == std::byte{91} && guest[9] == std::byte{100} && guest[10] == std::byte{0});
    // A buffer smaller than the request and an offset past the image are refused.
    CHECK(storage_read(0, 80, 40) == IpcDisposition::unsupported);
    CHECK(storage_read(200, 8, 8) == IpcDisposition::unsupported);
    // The same storage answers GetSize with the whole image length, which is the
    // query RomFS initialization issues right after the superblock read.
    const auto storage_command = [&](std::uint32_t command) {
        message.fill(std::byte{0});
        put(message, 0, 0x01000006);
        put(message, 4, 12);
        put(message, 32, 0x00200001);
        put(message, 36, 2);
        put(message, 48, 0x49434653);
        put(message, 52, 1);
        put(message, 56, command);
        registers.fill(0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    };
    CHECK(storage_command(4) == IpcDisposition::handled);
    CHECK(get(message, 48) == 100 && get(message, 52) == 0);
    // Another storage command stays explicit instead of reporting success.
    CHECK(storage_command(3) == IpcDisposition::unsupported);
    // The SDK storage client parses a reply through its descriptor counts: it
    // aligns the message plus four bytes per payload word and expects SFCO sixteen
    // bytes after that, with the 64-bit result eight bytes into the header. A reply
    // without descriptors yields a payload offset of two or three words, which both
    // align to the same position, so every storage reply must carry its CMIF header
    // at byte32 and its result at byte40.
    const auto client_parses_success = [](const IpcMessage& reply) {
        const auto words = get(reply, 4) & 0x3FFu;
        if (words == 0) return false;
        for (const auto payload_words : {2u, 3u}) {
            auto base = (payload_words * 4 + 15u) & ~15u;
            if (get(reply, base + 16) != 0x4f434653) continue;
            const auto result = get(reply, base + 24) | (std::uint64_t{get(reply, base + 28)} << 32);
            return result == 0;
        }
        return false;
    };
    std::fill(guest.begin(), guest.end(), std::byte{0});
    CHECK(storage_command(4) == IpcDisposition::handled);
    CHECK(client_parses_success(message));
    std::fill(guest.begin(), guest.end(), std::byte{0});
    CHECK(storage_read(0, 80, 80) == IpcDisposition::handled);
    CHECK(client_parses_success(message));

    // Without a configured image the store reports an explicit failure.
    auto empty = std::make_shared<FileSystemServiceSession>(71, 0);
    const auto empty_handle = handles.insert(empty).value();
    message = control(0);
    registers.fill(0);
    registers[0] = empty_handle;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    message.fill(std::byte{0});
    put(message, 0, 0x01000006);
    put(message, 4, 16);
    put(message, 8, 8);
    put(message, 12, static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(guest.data())));
    put(message, 32, 0x00200001);
    put(message, 36, 2);
    put(message, 48, 0x49434653);
    put(message, 52, 1);
    put(message, 72, 8);
    registers.fill(0);
    registers[0] = empty_handle;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::unsupported);
    CHECK(handles.close(empty_handle));
}

void test_large_storage_read() {
    // The service reads the store in bounded chunks, so a multi-megabyte read
    // must still transfer every byte across the 1 MiB chunk boundary.
    constexpr std::uint32_t size = 2'500'000;
    std::vector<std::byte> bytes(size);
    for (std::uint32_t index = 0; index < size; ++index)
        bytes[index] = static_cast<std::byte>(index & 0xff);
    auto storage = std::make_shared<armigo::romfs::MemoryStorage>(std::move(bytes));

    HandleTable handles{4};
    auto session = std::make_shared<FileSystemServiceSession>(71, 0, storage);
    IpcDispatcher dispatcher{handles, nullptr};
    const auto handle = handles.insert(session).value();
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};
    std::vector<std::byte> guest(size);
    caller.buffer_region_base = reinterpret_cast<std::uintptr_t>(guest.data()) & ~0xFFFFFFFFull;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || out.size() > base->size() - offset) return false;
        std::memcpy(out.data(), base->data() + offset, out.size());
        return true;
    };
    caller.write_memory = [](void* raw, std::uint64_t address,
                             std::span<const std::byte> in) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || in.size() > base->size() - offset) return false;
        std::memcpy(base->data() + offset, in.data(), in.size());
        return true;
    };
    caller.memory_context = &guest;

    message = control(0);
    registers.fill(0);
    registers[0] = handle;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, 1);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 200);
    registers.fill(0);
    registers[0] = handle;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    CHECK(get(message, 48) == 2);

    message.fill(std::byte{0});
    put(message, 0, 0x01000006);
    put(message, 4, 16);
    put(message, 8, size);
    put(message, 12, static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(guest.data())));
    put(message, 32, 0x00200001);
    put(message, 36, 2);
    put(message, 48, 0x49434653);
    put(message, 52, 1);
    put(message, 72, size);
    registers.fill(0);
    registers[0] = handle;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled);
    CHECK(get(message, 48) == size && get(message, 52) == 0);
    CHECK(guest[0] == std::byte{0});
    CHECK(guest[(1u << 20) - 1] == static_cast<std::byte>(0xff));
    CHECK(guest[1u << 20] == std::byte{0});
    CHECK(guest[2u << 20] == std::byte{0});
    CHECK(guest[size - 1] == static_cast<std::byte>((size - 1) & 0xff));
}


struct SaveGuest {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x1000);
    std::uintptr_t base = 0;
};

// loaded: the mount found save00 committed by an earlier run.
void test_save_data(bool loaded = false) {
    HandleTable handles{4};
    auto session = std::make_shared<FileSystemServiceSession>(71, 0);
    session->set_application_program_id(0x0100000000001000);
    IpcDispatcher dispatcher{handles, nullptr};
    const auto handle = handles.insert(session).value();
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};
    SaveGuest guest;
    // HIPC pointer descriptors carry 39-bit guest addresses, independent of
    // where the host allocator (including ASan) places this synthetic backing.
    guest.base = 0x7000000000ull;
    caller.buffer_region_base = guest.base;
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<SaveGuest*>(raw);
        const auto offset = address - self.base;
        if (offset > self.bytes.size() || out.size() > self.bytes.size() - offset) return false;
        std::memcpy(out.data(), self.bytes.data() + offset, out.size());
        return true;
    };
    caller.write_memory = [](void* raw, std::uint64_t address,
                             std::span<const std::byte> in) noexcept {
        auto& self = *static_cast<SaveGuest*>(raw);
        const auto offset = address - self.base;
        if (offset > self.bytes.size() || in.size() > self.bytes.size() - offset) return false;
        std::memcpy(self.bytes.data() + offset, in.data(), in.size());
        return true;
    };
    caller.memory_context = &guest;

    const auto send = [&] {
        registers.fill(0x123456789abcdef0);
        registers[0] = handle;
        return dispatcher.dispatch(0x21, registers, caller);
    };
    const auto convert = [&] {
        message = control(0);
        return send();
    };
    const auto set_path = [&](const char* text) {
        std::fill(guest.bytes.begin(), guest.bytes.begin() + 0x300, std::byte{0});
        std::memcpy(guest.bytes.data(), text, std::strlen(text));
    };
    // A one-pointer request: pointer descriptor at byte8, domain header at byte16.
    const auto pointer_call = [&](std::uint32_t id, std::uint32_t command) {
        message.fill(std::byte{0});
        put(message, 0, 0x00010006);
        put(message, 4, 12);
        const auto address = guest.base + 0;
        const auto upper = static_cast<std::uint32_t>(address >> 32);
        const auto packed = (0x300u << 16) | (((upper >> 4) & 7) << 6) | ((upper & 0xf) << 12);
        put(message, 8, packed);
        put(message, 12, static_cast<std::uint32_t>(address));
        put(message, 16, 0x00100001);
        put(message, 20, id);
        put(message, 32, 0x49434653);
        put(message, 36, 1);
        put(message, 40, command);
        put(message, 44, 0);
    };
    // A map-alias request: descriptor at byte8, domain header at byte32, CMIF at 48.
    const auto buffer_call = [&](std::uint32_t id, std::uint32_t command, std::uint32_t size) {
        message.fill(std::byte{0});
        put(message, 0, 0x01000006);
        put(message, 4, 12);
        put(message, 8, size);
        put(message, 12, static_cast<std::uint32_t>(guest.base));
        put(message, 16, static_cast<std::uint32_t>(
            (((guest.base >> 36) & 7) << 2) | (((guest.base >> 32) & 0xf) << 28)));
        put(message, 32, 0x00100001);
        put(message, 36, id);
        put(message, 48, 0x49434653);
        put(message, 52, 1);
        put(message, 56, command);
        put(message, 60, 0);
    };

    CHECK(convert() == IpcDisposition::handled);
    // Another program's save, whether or not the process may read it, does
    // not exist on the host: FS reports target not found.
    const auto open_other = [&](std::uint64_t program) {
        message.fill(std::byte{0});
        put(message, 0, 6);
        put(message, 4, 30);
        put(message, 16, 0x00580001);
        put(message, 20, 1);
        put(message, 32, 0x49434653);
        put(message, 36, 1);
        put(message, 40, 51);
        put(message, 48, 1);
        put(message, 56, static_cast<std::uint32_t>(program));
        put(message, 60, static_cast<std::uint32_t>(program >> 32));
        return send();
    };
    CHECK(open_other(0x0100000000003000) == IpcDisposition::handled);
    CHECK(get(message, 40) == 0x7d402);
    CHECK(open_other(0x0100000000004000) == IpcDisposition::handled);
    CHECK(get(message, 40) == 0x7d402);
    // The application's own id opens its save, as program zero does below.
    CHECK(open_other(0x0100000000001000) == IpcDisposition::handled);
    CHECK(get(message, 16) == 1);
    const auto own_file_system = static_cast<std::uint32_t>(get(message, 48));
    CHECK(own_file_system >= 2);
    // OpenSaveDataFileSystem returns a save-data file system object.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, 1);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 51);
    put(message, 44, 0);
    CHECK(send() == IpcDisposition::handled);
    CHECK(get(message, 16) == 1);
    const auto file_system = static_cast<std::uint32_t>(get(message, 48));
    CHECK(file_system >= 2);

    if (loaded) {
        // The committed file is there before this run creates anything.
        set_path("save00");
        pointer_call(file_system, 7);
        CHECK(send() == IpcDisposition::handled);
        CHECK(get(message, 48) == 1);
    }
    // CreateFile("save00", 0, 0): a new empty file.
    set_path("save00");
    pointer_call(file_system, 0);
    put(message, 48, 0);
    put(message, 56, 0, 8);
    CHECK(send() == IpcDisposition::handled);
    // GetEntryType reports a file, also through the save opened by the
    // application's own id: both name the same save tree.
    pointer_call(file_system, 7);
    CHECK(send() == IpcDisposition::handled);
    CHECK(get(message, 48) == 1);
    set_path("save00");
    pointer_call(own_file_system, 7);
    CHECK(send() == IpcDisposition::handled);
    CHECK(get(message, 48) == 1);
    // OpenFile returns a file object.
    pointer_call(file_system, 8);
    put(message, 48, 0);
    CHECK(send() == IpcDisposition::handled);
    CHECK(get(message, 16) == 1);
    const auto file = static_cast<std::uint32_t>(get(message, 48));
    CHECK(file >= 3);

    // Write "AB" at offset 4 through the file object.
    guest.bytes[0] = std::byte{'A'};
    guest.bytes[1] = std::byte{'B'};
    buffer_call(file, 1, 2);
    put(message, 64, 0);
    put(message, 72, 4, 8);
    put(message, 80, 2, 8);
    CHECK(send() == IpcDisposition::handled);
    // GetSize reports six bytes.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, file);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 4);
    put(message, 44, 0);
    CHECK(send() == IpcDisposition::handled);
    CHECK(get(message, 48, 8) == 6);

    // Read the two bytes back through the file object.
    std::fill(guest.bytes.begin(), guest.bytes.end(), std::byte{0});
    buffer_call(file, 0, 2);
    put(message, 64, 0);
    put(message, 72, 4, 8);
    put(message, 80, 2, 8);
    CHECK(send() == IpcDisposition::handled);
    CHECK(get(message, 48, 8) == 2);
    CHECK(guest.bytes[0] == std::byte{'A'} && guest.bytes[1] == std::byte{'B'});

    // Commit succeeds and the mount rejects an unimplemented command explicitly.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, file_system);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 10);
    put(message, 44, 0);
    CHECK(send() == IpcDisposition::handled);
    put(message, 16, 0x00100001);
    put(message, 20, file_system);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 15);
    put(message, 44, 0);
    CHECK(send() == IpcDisposition::unsupported);
}

// Direct buffered requests exercise file behavior independently of descriptor packing.
struct SaveClient {
    FileSystemServiceSession session{71, 0};
    IpcMessage message{};
    std::array<std::byte, 2048> memory{};
    GuestBuffers buffers;
    std::uint32_t mount = 0;
    SaveClient() : buffers(
        [](void* raw, std::uint64_t at, std::span<std::byte> out) noexcept {
            auto& bytes = static_cast<SaveClient*>(raw)->memory;
            if (at > bytes.size() || out.size() > bytes.size() - at) return false;
            std::memcpy(out.data(), bytes.data() + at, out.size()); return true;
        },
        [](void* raw, std::uint64_t at, std::span<const std::byte> in) noexcept {
            auto& bytes = static_cast<SaveClient*>(raw)->memory;
            if (at > bytes.size() || in.size() > bytes.size() - at) return false;
            std::memcpy(bytes.data() + at, in.data(), in.size()); return true;
        }, this) {
        buffers.add({0, 1024, true, true});
        buffers.add({512, 512, true, true});
        CHECK(session.request(control(0), 71).disposition == IpcDisposition::handled);
        prepare(1, 51);
        auto reply = session.request(message, 71);
        CHECK(reply.disposition == IpcDisposition::handled);
        mount = static_cast<std::uint32_t>(get(reply.bytes, 48));
        CHECK(mount >= 2);
    }
    void prepare(std::uint32_t object, std::uint32_t command) {
        message.fill(std::byte{});
        put(message, 0, 6); put(message, 4, 30);
        put(message, 16, 0x00580001); put(message, 20, object);
        put(message, 32, 0x49434653); put(message, 36, 1); put(message, 40, command);
    }
    void paths(const char* first, const char* second = "") {
        memory.fill(std::byte{});
        std::memcpy(memory.data(), first, std::strlen(first));
        std::memcpy(memory.data() + 512, second, std::strlen(second));
    }
    IpcReply send() { return session.request_with_buffers(message, 71, buffers); }
    std::uint32_t open(const char* path) {
        paths(path); prepare(mount, 8); auto reply = send();
        CHECK(reply.disposition == IpcDisposition::handled);
        return static_cast<std::uint32_t>(get(reply.bytes, 48));
    }
    std::uint64_t size(std::uint32_t file) {
        prepare(file, 4); auto reply = send(); CHECK(reply.disposition == IpcDisposition::handled);
        return get(reply.bytes, 48, 8);
    }
    void read(std::uint32_t file, std::uint64_t offset, std::uint64_t count) {
        memory.fill(std::byte{}); prepare(file, 0);
        put(message, 56, offset, 8); put(message, 64, count, 8);
        CHECK(send().disposition == IpcDisposition::handled);
    }
    void commit() { prepare(mount, 10); CHECK(send().disposition == IpcDisposition::handled); }
};

std::uint64_t physical_footprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    CHECK(task_info(mach_task_self(), TASK_VM_INFO,
                   reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS);
    return info.phys_footprint;
}

// A multi-GiB sparse tree must mount without reading any unrelated payload.
// This test never commits or opens those files, so disk use stays bounded.
void test_save_mount_is_metadata_only() {
    const auto root = std::filesystem::temp_directory_path() /
        ("armigo-save-lazy-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "nested");
    for (unsigned index = 0; index < 96; ++index) {
        std::ofstream file(root / "nested" / std::to_string(index), std::ios::binary);
        file.seekp((32u << 20) - 1); file.put('Z');
    }
    armigo::services::fs::set_host_save_directory(root);
    const auto before = physical_footprint();
    {
        SaveClient client;
        const auto after = physical_footprint();
        CHECK(after <= before + (32u << 20));
    }
    armigo::services::fs::set_host_save_directory({});
    std::filesystem::remove_all(root);
}

// Untouched payloads survive commit, rename and a second commit; opened
// aliases retain their contents after rename/delete and tree replacement.
void test_lazy_save_roundtrip() {
    const auto root = std::filesystem::temp_directory_path() /
        ("armigo-save-roundtrip-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "nested");
    std::ofstream(root / "opened", std::ios::binary) << "ABCDEFGH";
    std::ofstream(root / "nested/unopened", std::ios::binary) << "retained payload";
    std::ofstream(root / "untouched", std::ios::binary) << "second commit";
    armigo::services::fs::set_host_save_directory(root);
    {
        SaveClient client;
        client.paths("nested"); client.prepare(client.mount, 9);
        const auto directory_reply = client.send();
        CHECK(directory_reply.disposition == IpcDisposition::handled);
        const auto directory = static_cast<std::uint32_t>(get(directory_reply.bytes, 48));
        client.prepare(directory, 0);
        CHECK(client.send().disposition == IpcDisposition::handled);
        std::uint64_t listed_size = 0;
        for (unsigned byte = 0; byte < 8; ++byte)
            listed_size |= std::uint64_t(std::to_integer<unsigned>(client.memory[0x308 + byte])) << (byte * 8);
        CHECK(listed_size == 16);
        const auto file = client.open("opened");
        CHECK(client.size(file) == 8);
        client.read(file, 2, 3);
        CHECK(std::memcmp(client.memory.data(), "CDE", 3) == 0);
        client.paths("opened", "renamed"); client.prepare(client.mount, 5);
        CHECK(client.send().disposition == IpcDisposition::handled);
        client.read(file, 0, 8);
        CHECK(std::memcmp(client.memory.data(), "ABCDEFGH", 8) == 0);
        client.memory[0] = std::byte{'X'}; client.prepare(file, 1);
        put(client.message, 56, 3, 8); put(client.message, 64, 1, 8);
        CHECK(client.send().disposition == IpcDisposition::handled);
        client.paths("nested", "moved"); client.prepare(client.mount, 6);
        CHECK(client.send().disposition == IpcDisposition::handled);
        client.commit();
        client.commit();
        const auto untouched = client.open("moved/unopened");
        CHECK(client.size(untouched) == 16);
        client.read(untouched, 9, 7);
        CHECK(std::memcmp(client.memory.data(), "payload", 7) == 0);
        client.prepare(file, 3); put(client.message, 48, 5, 8);
        CHECK(client.send().disposition == IpcDisposition::handled);
        client.commit();
        client.paths("renamed"); client.prepare(client.mount, 1);
        CHECK(client.send().disposition == IpcDisposition::handled);
        client.commit();
        client.read(file, 0, 5);
        CHECK(std::memcmp(client.memory.data(), "ABCXE", 5) == 0);
        CHECK(!std::filesystem::exists(root / "renamed"));
    }
    {
        SaveClient client;
        const auto file = client.open("untouched");
        client.read(file, 0, 13);
        CHECK(std::memcmp(client.memory.data(), "second commit", 13) == 0);
    }
    // Missing host backing is an explicit failure, never an empty opened file.
    {
        SaveClient client;
        std::filesystem::remove(root / "untouched");
        client.paths("untouched"); client.prepare(client.mount, 8);
        CHECK(client.send().disposition == IpcDisposition::unsupported);
        CHECK(std::string(client.session.diagnostic()).find("backing") != std::string::npos);
        client.prepare(client.mount, 10);
        CHECK(client.send().disposition == IpcDisposition::unsupported);
        CHECK(std::filesystem::is_regular_file(root / "moved/unopened"));
    }
    armigo::services::fs::set_host_save_directory({});
    std::filesystem::remove_all(root);
    std::filesystem::remove_all(std::filesystem::path(root).concat(".commit"));
}

// Actual permission, I/O, space and collision failures must not turn into
// payload copies; only unavailable or cross-filesystem clones allow copying.
void test_unopened_save_clone_failures() {
    const auto root = std::filesystem::temp_directory_path() /
        ("armigo-save-clone-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    std::ofstream(root / "untouched", std::ios::binary) << "original payload";
    armigo::services::fs::set_host_save_directory(root);
    {
        SaveClient client;
        for (const auto error : {EACCES, EIO, ENOSPC, EEXIST}) {
            forced_clone_error.store(error);
            client.prepare(client.mount, 10);
            CHECK(client.send().disposition == IpcDisposition::unsupported);
            std::ifstream source(root / "untouched", std::ios::binary);
            CHECK(std::string(std::istreambuf_iterator<char>(source), {}) == "original payload");
            CHECK(!std::filesystem::exists(std::filesystem::path(root).concat(".previous")));
        }
        for (const auto error : {ENOTSUP, EXDEV, ENOSYS}) {
            forced_clone_error.store(error);
            client.commit();
        }
        forced_clone_error.store(0);
        const auto before = clone_attempts.load();
        client.commit();
        CHECK(clone_attempts.load() > before);
        const auto file = client.open("untouched");
        client.read(file, 0, 16);
        CHECK(std::memcmp(client.memory.data(), "original payload", 16) == 0);
    }
    forced_clone_error.store(0);
    armigo::services::fs::set_host_save_directory({});
    std::filesystem::remove_all(root);
}

// With a host save directory a commit writes the tree to disk and a later
// mount reads it back.
void test_save_data_persists() {
    const auto root = std::filesystem::temp_directory_path() /
        ("armigo-save-test-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    armigo::services::fs::set_host_save_directory(root);
    test_save_data();
    std::ifstream file(root / "save00", std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    CHECK(bytes == std::string("\0\0\0\0AB", 6));
    CHECK(!std::filesystem::exists(std::filesystem::path(root).concat(".commit")));
    test_save_data(true);
    armigo::services::fs::set_host_save_directory({});
    std::filesystem::remove_all(root);
}

// A save directory given with a trailing separator commits beside the
// directory, not inside it.
void test_save_directory_trailing_separator() {
    const auto root = std::filesystem::temp_directory_path() /
        ("armigo-save-slash-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    armigo::services::fs::set_host_save_directory(root.string() + "/");
    test_save_data();
    CHECK(std::filesystem::is_regular_file(root / "save00"));
    CHECK(!std::filesystem::exists(root / ".commit") && !std::filesystem::exists(root / ".previous"));
    armigo::services::fs::set_host_save_directory({});
    std::filesystem::remove_all(root);
}

// A commit interrupted between its two renames leaves no save directory: the
// next load restores the staged tree, or the previous one without it.
void test_save_recovery_after_interrupted_commit() {
    const auto root = std::filesystem::temp_directory_path() /
        ("armigo-save-recover-" + std::to_string(::getpid()));
    const auto staged = std::filesystem::path(root).concat(".commit");
    const auto previous = std::filesystem::path(root).concat(".previous");
    for (const bool with_staged : {true, false}) {
        std::filesystem::remove_all(root);
        std::filesystem::remove_all(staged);
        std::filesystem::remove_all(previous);
        armigo::services::fs::set_host_save_directory(root);
        test_save_data();
        std::filesystem::rename(root, previous);
        if (with_staged) {
            std::filesystem::copy(previous, staged, std::filesystem::copy_options::recursive);
            std::ofstream(staged / "staged") << "new";
        }
        test_save_data(true);
        CHECK(std::filesystem::is_regular_file(root / "save00"));
        // The staged tree is the newer commit and wins over the previous one.
        CHECK(std::filesystem::exists(root / "staged") == with_staged);
        CHECK(!std::filesystem::exists(previous) && !std::filesystem::exists(staged));
    }
    armigo::services::fs::set_host_save_directory({});
    std::filesystem::remove_all(root);
}

void test_cloned_session_storage_read() {
    // Firmware resolves domain object ids across every clone of an fsp-srv
    // session: a streaming client opened its storage through one clone and then
    // issued reads through a sibling clone. The clone therefore shares the
    // object tables instead of snapshotting them. The flow mirrors the run that
    // stopped: convert, clone, clone, two opens on the second clone, reads on
    // the storage through both clones.
    std::vector<std::byte> bytes(256);
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::byte>(index + 1);
    auto image = std::make_shared<armigo::romfs::MemoryStorage>(std::move(bytes));

    HandleTable handles{8};
    auto root = std::make_shared<FileSystemServiceSession>(71, 0, image);
    IpcDispatcher dispatcher{handles, nullptr};
    const auto root_handle = handles.insert(root).value();
    IpcMessage message{};
    QueryRegisters registers{};
    IpcCallContext caller{71, message, nullptr, nullptr};
    std::vector<std::byte> guest(0x100);
    caller.read_memory = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || out.size() > base->size() - offset) return false;
        std::memcpy(out.data(), base->data() + offset, out.size());
        return true;
    };
    caller.write_memory = [](void* raw, std::uint64_t address,
                             std::span<const std::byte> in) noexcept {
        auto* base = static_cast<std::vector<std::byte>*>(raw);
        const auto offset = address - reinterpret_cast<std::uintptr_t>(base->data());
        if (offset > base->size() || in.size() > base->size() - offset) return false;
        std::memcpy(base->data() + offset, in.data(), in.size());
        return true;
    };
    caller.memory_context = &guest;

    const auto send = [&](Handle target) {
        registers.fill(0);
        registers[0] = target;
        return dispatcher.dispatch(0x21, registers, caller);
    };

    // Convert the root to a domain, then clone it twice as the SDK client does.
    message = control(0);
    CHECK(send(root_handle) == IpcDisposition::handled);
    message = control(2);
    CHECK(send(root_handle) == IpcDisposition::handled);
    const auto first = static_cast<Handle>(get(message, 20));
    CHECK(first != 0 && first != root_handle);
    message = control(2);
    CHECK(send(first) == IpcDisposition::handled);
    const auto second = static_cast<Handle>(get(message, 20));
    CHECK(second != 0 && second != first);
    auto first_session = handles.lookup_as<FileSystemServiceSession>(first);
    auto second_session = handles.lookup_as<FileSystemServiceSession>(second);
    CHECK(first_session != nullptr && second_session != nullptr &&
          first_session != second_session);

    // Two mounts on the second clone hand out domain ids 2 and 3.
    const auto open_store = [&](Handle target) {
        message.fill(std::byte{0});
        put(message, 0, 6);
        put(message, 4, 12);
        put(message, 16, 0x00100001);
        put(message, 20, 1);
        put(message, 32, 0x49434653);
        put(message, 36, 1);
        put(message, 40, 200);
        return send(target);
    };
    CHECK(open_store(second) == IpcDisposition::handled);
    CHECK(get(message, 48) == 2);
    CHECK(open_store(second) == IpcDisposition::handled);
    CHECK(get(message, 48) == 3);

    // The streaming read: one counted B descriptor whose packed word carries a
    // heap address, the domain header at byte32 with storage id 3, and SFCI at
    // byte48 with command 0 and {offset, size}.
    const auto storage_read = [&](Handle target, std::uint32_t id) {
        const auto packed = static_cast<std::uint32_t>(
            (((reinterpret_cast<std::uintptr_t>(guest.data()) >> 36) & 7) << 2) |
            (((reinterpret_cast<std::uintptr_t>(guest.data()) >> 32) & 0xf) << 28));
        message.fill(std::byte{0});
        put(message, 0, 0x01000006);
        put(message, 4, 0x00000010);
        put(message, 8, static_cast<std::uint32_t>(guest.size()));
        put(message, 12, static_cast<std::uint32_t>(
                             reinterpret_cast<std::uintptr_t>(guest.data()) & 0xFFFFFFFFu));
        put(message, 16, packed);
        put(message, 32, 0x00200001);
        put(message, 36, id);
        put(message, 48, 0x49434653);
        put(message, 52, 1);
        put(message, 56, 0);
        put(message, 64, 16);  // offset
        put(message, 72, 80);  // size
        return send(target);
    };

    std::fill(guest.begin(), guest.end(), std::byte{0});
    CHECK(storage_read(second, 3) == IpcDisposition::handled);
    CHECK(get(message, 48) == 80 && get(message, 52) == 0);
    CHECK(guest[0] == std::byte{17} && guest[79] == std::byte{96});
    // The same read through the sibling clone resolves the same storage object.
    std::fill(guest.begin(), guest.end(), std::byte{0});
    CHECK(storage_read(first, 3) == IpcDisposition::handled);
    CHECK(get(message, 48) == 80 && get(message, 52) == 0);
    CHECK(guest[0] == std::byte{17} && guest[79] == std::byte{96});
    // The id space continues across clones: the next mount on the first clone
    // must not reuse an id the second clone already handed out.
    CHECK(open_store(first) == IpcDisposition::handled);
    CHECK(get(message, 48) == 4);
    CHECK(handles.close(second));
    CHECK(handles.close(first));
}

void test_controls_and_commands() {
    Fixture f;
    f.message = control(3);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(get(f.message, 4) == 9 && get(f.message, 16) == 0x4f434653);
    CHECK(get(f.message, 28) == 0xec6be3ff && get(f.message, 32) == 0x400);
    CHECK(get(f.message, 34) == 0);
    CHECK(f.session->domain_object_count() == 0);
    f.message = control(0);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(get(f.message, 32) == 1);
    CHECK(f.session->domain_object_count() == 1);
    // A first domain command is explicitly unsupported and preserves input.
    IpcMessage request;
    request.fill(std::byte{0xa5});
    put(request, 0, 6);
    put(request, 4, 12);
    put(request, 16, 0x00100001);
    put(request, 20, 1);
    put(request, 24, 0);
    put(request, 28, 0);
    put(request, 32, 0x49434653);
    put(request, 36, 1);
    put(request, 40, 0);
    put(request, 44, 0);
    f.message = request;
    f.registers.fill(0x123456789abcdef0);
    f.registers[0] = f.handle;
    const auto before = f.message;
    const auto saved = f.registers;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::unsupported);
    CHECK(f.registers == saved && f.message == before);
    CHECK(f.session->diagnostic() != nullptr);
    // Domain commands: open a file system (200), read the access log mode
    // (1005) and mount (512) all succeed; anything else stays unsupported.
    const auto domain = [&](std::uint32_t command, unsigned size) {
        IpcMessage request;
        request.fill(std::byte{0xa5});
        put(request, 0, 6);
        put(request, 4, 12);
        put(request, 16, 1u | (static_cast<std::uint32_t>(size) << 16));
        put(request, 20, 1);
        put(request, 32, 0x49434653);
        put(request, 36, 1);
        put(request, 40, command);
        put(request, 44, 0);
        return request;
    };
    f.message = domain(200, 16);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(get(f.message, 16) == 1 && get(f.message, 48) == 2);
    // The open reply also reports the size of the store it hands out, because
    // RomFS initialization validates the superblock regions against it. This
    // session has no configured store, so the reported size is zero.
    CHECK(get(f.message, 56) == 0 && get(f.message, 60) == 0);
    CHECK(f.session->domain_object_count() == 2);
    f.message = domain(1005, 16);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(get(f.message, 48) == 0);
    f.message = domain(512, 16);
    CHECK(f.send() == IpcDisposition::handled);
    CHECK(get(f.message, 48) == 3);
    CHECK(f.session->domain_object_count() == 3);
    // Releasing domain objects is a domain message with a zero-size header block
    // and no CMIF request; it must succeed so the client can drop a store.
    {
        IpcMessage release;
        release.fill(std::byte{0});
        put(release, 0, 6);
        put(release, 4, 8);
        put(release, 16, 2);
        put(release, 20, 2);
        f.message = release;
        f.registers.fill(0);
        f.registers[0] = f.handle;
        CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::handled);
        CHECK(get(f.message, 16) == 0x4f434653);
    }
    // Malformed controls stay unsupported.
    f.message = control(3);
    put(f.message, 16, 0);
    f.registers.fill(0);
    f.registers[0] = f.handle;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::unsupported);
    // Clone hands the caller a new session handle.
    f.message = control(2);
    f.registers.fill(0);
    f.registers[0] = f.handle;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::handled);
    const auto clone = static_cast<Handle>(get(f.message, 20));
    CHECK(clone != 0 && clone != f.handle);
    CHECK(f.handles.lookup_as<FileSystemServiceSession>(clone) != nullptr);
    CHECK(f.handles.lookup_as<FileSystemServiceSession>(clone) != f.session);
    CHECK(f.handles.close(clone));
    // Another process owns no session here.
    f.message = control(3);
    f.registers.fill(0);
    f.registers[0] = f.handle;
    f.caller.process_id = 72;
    CHECK(f.dispatcher.dispatch(0x21, f.registers, f.caller) == IpcDisposition::unsupported);
}
} // namespace

int main() {
    test_controls_and_commands();
    test_romfs_storage_read();
    test_addon_data_storage();
    test_cloned_session_storage_read();
    test_save_data();
    test_save_data_persists();
    test_save_mount_is_metadata_only();
    test_lazy_save_roundtrip();
    test_unopened_save_clone_failures();
    test_save_directory_trailing_separator();
    test_save_recovery_after_interrupted_commit();
    test_large_storage_read();
    std::printf("File-system service failures: %u\n", failures);
    return failures ? 1 : 0;
}
