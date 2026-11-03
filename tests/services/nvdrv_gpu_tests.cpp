// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/runtime/presentation.h>
#include <armigo/services/nvdrv.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using armigo::horizon::Event;
using armigo::horizon::GuestBuffer;
using armigo::horizon::GuestBuffers;
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::horizon::IpcReply;
using armigo::runtime::GuestMemoryPin;
using armigo::runtime::LaunchError;
using armigo::runtime::LaunchFailure;
using armigo::runtime::LaunchStatus;
using armigo::services::nvdrv::DriverSession;

unsigned failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t ioctl_command = 1;
constexpr std::uint32_t ioctl2_command = 11;
constexpr std::uint32_t query_event_command = 4;

constexpr std::uint32_t nvmap_create = 0xc0080101;
constexpr std::uint32_t nvmap_allocate = 0xc0200104;
constexpr std::uint32_t nvmap_free = 0xc0180105;
constexpr std::uint32_t nvmap_get_id = 0xc008010e;
constexpr std::uint32_t address_space_unmap_buffer = 0xc0084105;
constexpr std::uint32_t address_space_initialize = 0x40284109;
constexpr std::uint32_t address_space_bind_channel = 0x40044101;
constexpr std::uint32_t address_space_map_buffer = 0xc0284106;
constexpr std::uint32_t channel_set_nvmap = 0x40044801;
constexpr std::uint32_t channel_allocate_gpfifo = 0xc020481a;
constexpr std::uint32_t channel_submit_gpfifo = 0xc018481b;

constexpr std::uint32_t syncpoint_read_min = 0xc0080014;
constexpr std::uint32_t syncpoint_read_max = 0xc008001a;
constexpr std::uint32_t syncpoint_wait_allocate = 0xc010001d;
constexpr std::uint32_t syncpoint_wait_registered = 0xc010001e;
constexpr std::uint32_t event_register = 0x4004001f;

constexpr std::uint32_t nv_success = 0;
constexpr std::uint32_t nv_timeout = 5;
constexpr std::uint32_t nv_invalid_size = 10;
constexpr std::uint32_t nv_invalid_value = 11;

constexpr std::uint32_t expected_syncpoint = 24;
constexpr std::uint64_t memory_base = 0x7000000000ull;
constexpr std::size_t memory_size = 0x90000;
constexpr std::uint64_t argument_address = memory_base + 0x1000;
constexpr std::uint64_t inline_address = memory_base + 0x2000;
constexpr std::uint64_t path_address = memory_base + 0x3000;
constexpr std::uint64_t backing_address = memory_base + 0x10000;
constexpr std::uint64_t backing_size = 0x4000;
constexpr std::uint64_t buffer_offset = 0x1000;
constexpr std::uint64_t gpu_base = 0x09000000;
constexpr std::uint64_t gpu_mapping_size = 0x2000;

bool succeeded(const LaunchStatus& status) {
    return std::holds_alternative<std::monostate>(status);
}

template <class T>
bool failed_with(const std::variant<LaunchFailure, T>& result, LaunchError code) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure != nullptr && failure->stage == armigo::runtime::LaunchStage::graphics &&
           failure->code == code && !failure->operation.empty() && !failure->detail.empty();
}

class RecordingPin final : public armigo::runtime::PinnedGuestMemory {
public:
    explicit RecordingPin(armigo::horizon::AddressRange range)
        : range_(range), bytes_(static_cast<std::size_t>(range.size)) {}

    armigo::horizon::AddressRange range() const noexcept override { return range_; }
    armigo::memory::Permission permissions() const noexcept override {
        return armigo::memory::Permission::read;
    }
    std::span<const std::byte> bytes() const noexcept override { return bytes_; }
    std::span<std::byte> writable_bytes() noexcept override { return {}; }

private:
    armigo::horizon::AddressRange range_{};
    std::vector<std::byte> bytes_;
};

class RecordingGuestMemory final : public armigo::runtime::GuestMemory {
public:
    std::shared_ptr<armigo::horizon::ProcessRecord> process() const noexcept override {
        return {};
    }
    armigo::runtime::LaunchResult<GuestMemoryPin> allocate(
        std::uint64_t, armigo::horizon::MemoryType,
        armigo::memory::Permission) override {
        return LaunchFailure{};
    }
    LaunchStatus release(armigo::horizon::AddressRange) override {
        return LaunchFailure{};
    }
    armigo::runtime::LaunchResult<armigo::horizon::QueryDisposition> dispatch(
        std::uint16_t, armigo::horizon::QueryRegisters&) override {
        return LaunchFailure{};
    }
    armigo::runtime::LaunchResult<GuestMemoryPin> pin(
        armigo::horizon::AddressRange range,
        armigo::memory::Permission access) override {
        ++pin_calls;
        last_range = range;
        last_access = access;
        return std::make_shared<RecordingPin>(range);
    }
    armigo::horizon::GuestCopyOut copy_out() noexcept override { return {}; }
    GuestBuffers ipc_buffers() noexcept override { return {}; }
    armigo::horizon::GuestMemoryMapper service_mapper() noexcept override { return {}; }

    unsigned pin_calls = 0;
    armigo::horizon::AddressRange last_range{};
    armigo::memory::Permission last_access = armigo::memory::Permission::none;
};

void store_word(std::span<std::byte> bytes, std::size_t offset,
                std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[offset + index] =
            static_cast<std::byte>((value >> (index * 8)) & 0xff);
    }
}

void store_qword(std::span<std::byte> bytes, std::size_t offset,
                 std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        bytes[offset + index] =
            static_cast<std::byte>((value >> (index * 8)) & 0xff);
    }
}

std::uint32_t load_word(std::span<const std::byte> bytes,
                        std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index) {
        value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + index])}
                 << (index * 8);
    }
    return value;
}

std::uint64_t load_qword(std::span<const std::byte> bytes, std::size_t offset) {
    return load_word(bytes, offset) | (std::uint64_t{load_word(bytes, offset + 4)} << 32);
}

std::uint32_t reply_word(const IpcReply& reply, std::size_t offset) {
    return load_word(reply.bytes, offset);
}

IpcMessage request(std::uint32_t command, std::uint32_t fd = 0,
                   std::uint32_t operation = 0) {
    IpcMessage message{};
    store_word(message, 0, 6);
    store_word(message, 4, 12);
    store_word(message, 16, sfci);
    store_word(message, 24, command);
    store_word(message, 32, fd);
    store_word(message, 36, operation);
    return message;
}

constexpr std::uint32_t header(std::uint32_t mode, std::uint32_t method,
                               std::uint32_t subchannel,
                               std::uint32_t count) {
    return (mode << 29) | (count << 16) | (subchannel << 13) | method;
}

constexpr std::uint64_t entry(std::uint64_t address, std::uint32_t words) {
    return address | (std::uint64_t{words} << 42);
}

std::vector<std::byte> encode_entries(std::span<const std::uint64_t> entries) {
    std::vector<std::byte> bytes(entries.size() * 8);
    for (std::size_t index = 0; index < entries.size(); ++index) {
        store_qword(bytes, index * 8, entries[index]);
    }
    return bytes;
}

struct Fixture {
    std::shared_ptr<DriverSession> session = std::make_shared<DriverSession>();
    struct MemoryState {
        std::vector<std::byte> memory = std::vector<std::byte>(memory_size);
        std::uint64_t mapped_address = 0;
        std::uint64_t mapped_size = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> unmapped;
        bool fail_write = false;
        std::uint64_t fail_read_address = 0;
        std::uint64_t fail_read_once_address = 0;
        std::shared_ptr<std::promise<void>> fail_read_once_signal;
        std::uint64_t fail_write_address = 0;
        bool record_backing_accesses = false;
        std::vector<std::uint64_t> backing_reads;
        std::vector<std::uint64_t> backing_writes;
    };
    std::shared_ptr<MemoryState> state = std::make_shared<MemoryState>();

    static bool read_memory(void* raw, std::uint64_t address,
                            std::span<std::byte> output) noexcept {
        auto& self = *static_cast<MemoryState*>(raw);
        if (self.record_backing_accesses && address >= backing_address &&
            address < backing_address + backing_size) {
            self.backing_reads.push_back(address);
        }
        if (address == self.fail_read_address || address < memory_base) {
            return false;
        }
        if (address == self.fail_read_once_address) {
            self.fail_read_once_address = 0;
            if (self.fail_read_once_signal) self.fail_read_once_signal->set_value();
            return false;
        }
        const std::uint64_t offset = address - memory_base;
        if (offset > self.memory.size() ||
            output.size() > self.memory.size() - offset) {
            return false;
        }
        std::memcpy(output.data(), self.memory.data() + offset, output.size());
        return true;
    }

    static bool write_memory(void* raw, std::uint64_t address,
                             std::span<const std::byte> input) noexcept {
        auto& self = *static_cast<MemoryState*>(raw);
        if (self.fail_write || address == self.fail_write_address || address < memory_base) {
            return false;
        }
        const std::uint64_t offset = address - memory_base;
        if (offset > self.memory.size() ||
            input.size() > self.memory.size() - offset) {
            return false;
        }
        std::memcpy(self.memory.data() + offset, input.data(), input.size());
        if (self.record_backing_accesses && address >= backing_address &&
            address < backing_address + backing_size) {
            self.backing_writes.push_back(address);
        }
        return true;
    }

    static bool map_memory(void* raw, std::uint64_t address,
                           std::uint64_t size) noexcept {
        auto& self = *static_cast<MemoryState*>(raw);
        if (address < memory_base || size > memory_size ||
            address - memory_base > memory_size - size) {
            return false;
        }
        self.mapped_address = address;
        self.mapped_size = size;
        return true;
    }

    static void unmap_memory(void* raw, std::uint64_t address,
                             std::uint64_t size) noexcept {
        static_cast<MemoryState*>(raw)->unmapped.emplace_back(address, size);
    }

    void set_mapper(bool owned = true) {
        session->set_memory_mapper({map_memory, unmap_memory, state.get(),
                                   owned ? state : std::shared_ptr<MemoryState>{}});
    }

    Fixture() { set_mapper(); }

    bool dispatch_requests = false;

    IpcReply dispatch_request(const IpcMessage& payload, const GuestBuffers& buffers = {}) {
        // Exercise the runtime boundary and a decorator, including requests with
        // no descriptors. The callbacks still authorize indirect backing access.
        IpcMessage message{};
        store_word(message, 0, 6u | (static_cast<std::uint32_t>(buffers.size()) << 28));
        store_word(message, 4, 12);
        std::size_t cursor = 8;
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            const auto& buffer = buffers[index];
            store_word(message, cursor, buffer.size);
            store_word(message, cursor + 4, static_cast<std::uint32_t>(buffer.address));
            store_word(message, cursor + 8, static_cast<std::uint32_t>(
                (((buffer.address >> 36) & 7) << 2) |
                (((buffer.address >> 32) & 0xf) << 28)));
            cursor += 12;
        }
        const auto cmif = (cursor + 15) & ~std::size_t{15};
        std::memcpy(message.data() + cmif, payload.data() + 16, 32);
        armigo::horizon::HandleTable handles{4};
        const auto handle = handles.insert(
            std::make_shared<armigo::horizon::ForwardingSession>(session));
        CHECK(handle.has_value());
        armigo::horizon::IpcDispatcher dispatcher{handles, nullptr};
        armigo::horizon::IpcCallContext caller{};
        caller.process_id = 1;
        caller.message = message;
        caller.read_memory = read_memory;
        caller.write_memory = write_memory;
        caller.memory_context = state.get();
        caller.memory_mapper = {map_memory, unmap_memory, state.get(), state};
        armigo::horizon::QueryRegisters registers{};
        registers[0] = handle.value_or(0);
        IpcReply reply{};
        reply.disposition = dispatcher.dispatch(0x21, registers, caller);
        if (reply.disposition == IpcDisposition::handled) {
            CHECK(registers[0] == 0);
            reply.bytes = message;
            reply.size = message.size();
        }
        return reply;
    }

    void copy_to(std::uint64_t address, std::span<const std::byte> bytes) {
        CHECK(write_memory(state.get(), address, bytes));
    }

    void copy_from(std::uint64_t address, std::span<std::byte> bytes) {
        CHECK(read_memory(state.get(), address, bytes));
    }

    std::uint32_t open(std::string_view path) {
        const auto path_bytes = std::as_bytes(std::span(path.data(), path.size()));
        copy_to(path_address, path_bytes);
        GuestBuffers buffers(read_memory, write_memory, state.get());
        buffers.add({path_address, static_cast<std::uint32_t>(path.size()), true,
                     false});
        const IpcReply reply =
            session->request_with_buffers(request(0), 1, buffers);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(reply, 36) == nv_success);
        return reply_word(reply, 32);
    }

    IpcReply ioctl(std::uint32_t fd, std::uint32_t operation,
                   std::span<std::byte> arguments,
                   std::uint32_t command = ioctl_command,
                   std::span<const std::byte> inline_data = {},
                   bool include_inline = false) {
        copy_to(argument_address, arguments);
        GuestBuffers buffers(read_memory, write_memory, state.get());
        buffers.add({argument_address,
                     static_cast<std::uint32_t>(arguments.size()), true, true});
        if (include_inline) {
            copy_to(inline_address, inline_data);
            buffers.add({inline_address,
                         static_cast<std::uint32_t>(inline_data.size()), true,
                         false});
        }
        const auto message = request(command, fd, operation);
        const IpcReply reply = dispatch_requests
            ? dispatch_request(message, buffers)
            : session->request_with_buffers(message, 1, buffers);
        copy_from(argument_address, arguments);
        return reply;
    }

    IpcReply query_event(std::uint32_t fd, std::uint32_t encoded) {
        return session->request(request(query_event_command, fd, encoded), 1);
    }
};

struct PreparedFixture : Fixture {
    std::uint32_t nvmap = 0;
    std::uint32_t address_space = 0;
    std::uint32_t channel = 0;
    std::uint32_t control = 0;
    std::uint32_t memory_handle = 0;

    explicit PreparedFixture(std::uint32_t big_page_size = 0, std::uint32_t kind = 0) {
        nvmap = open("/dev/nvmap");
        address_space = open("/dev/nvhost-as-gpu");
        channel = open("/dev/nvhost-gpu");
        control = open("/dev/nvhost-ctrl");

        std::array<std::byte, 0x28> initialize{};
        store_word(initialize, 8, big_page_size);
        IpcReply reply = ioctl(address_space, address_space_initialize, initialize);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(reply, 32) == nv_success);

        std::array<std::byte, 4> binding{};
        store_word(binding, 0, channel);
        reply = ioctl(address_space, address_space_bind_channel, binding);
        CHECK(reply_word(reply, 32) == nv_success);

        store_word(binding, 0, nvmap);
        reply = ioctl(channel, channel_set_nvmap, binding);
        CHECK(reply_word(reply, 32) == nv_success);

        std::array<std::byte, 8> create{};
        store_word(create, 0, backing_size);
        reply = ioctl(nvmap, nvmap_create, create);
        CHECK(reply_word(reply, 32) == nv_success);
        memory_handle = load_word(create, 4);

        std::array<std::byte, 0x20> allocate{};
        store_word(allocate, 0, memory_handle);
        store_word(allocate, 12, 0x1000);
        store_qword(allocate, 24, backing_address);
        reply = ioctl(nvmap, nvmap_allocate, allocate);
        CHECK(reply_word(reply, 32) == nv_success);
        CHECK(state->mapped_address == backing_address);
        CHECK(state->mapped_size == backing_size);

        std::array<std::byte, 0x28> mapping{};
        store_word(mapping, 0, 1);
        store_word(mapping, 4, kind);
        store_word(mapping, 8, memory_handle);
        store_word(mapping, 12, 0x1000);
        store_qword(mapping, 16, buffer_offset);
        store_qword(mapping, 24, gpu_mapping_size);
        store_qword(mapping, 32, gpu_base);
        reply = ioctl(address_space, address_space_map_buffer, mapping);
        CHECK(reply_word(reply, 32) == nv_success);
        CHECK(load_word(mapping, 32) == static_cast<std::uint32_t>(gpu_base));
    }

    std::uint32_t create_memory(std::uint32_t size) {
        std::array<std::byte, 8> arguments{};
        store_word(arguments, 0, size);
        CHECK(reply_word(ioctl(nvmap, nvmap_create, arguments), 32) == nv_success);
        return load_word(arguments, 4);
    }

    void allocate_memory(std::uint32_t handle, std::uint64_t address,
                         std::uint32_t alignment = 0x1000, std::uint32_t flags = 0) {
        std::array<std::byte, 0x20> arguments{};
        store_word(arguments, 0, handle);
        store_word(arguments, 8, flags);
        store_word(arguments, 12, alignment);
        store_qword(arguments, 24, address);
        CHECK(reply_word(ioctl(nvmap, nvmap_allocate, arguments), 32) == nv_success);
    }

    void export_memory(std::uint32_t handle) {
        std::array<std::byte, 8> arguments{};
        store_word(arguments, 4, handle);
        CHECK(reply_word(ioctl(nvmap, nvmap_get_id, arguments), 32) == nv_success);
        CHECK(load_word(arguments, 0) == handle);
    }

    std::array<std::byte, 0x18> free_memory(std::uint32_t handle) {
        std::array<std::byte, 0x18> arguments{};
        store_word(arguments, 0, handle);
        CHECK(reply_word(ioctl(nvmap, nvmap_free, arguments), 32) == nv_success);
        return arguments;
    }

    IpcReply map_memory_handle(std::uint32_t handle, std::uint64_t& address,
                               bool fixed = true) {
        std::array<std::byte, 0x28> arguments{};
        store_word(arguments, 0, fixed ? 1 : 0);
        store_word(arguments, 8, handle);
        store_word(arguments, 12, 0x1000);
        store_qword(arguments, 16, buffer_offset);
        store_qword(arguments, 24, gpu_mapping_size);
        store_qword(arguments, 32, address);
        auto reply = ioctl(address_space, address_space_map_buffer, arguments);
        address = std::uint64_t{load_word(arguments, 32)} |
            (std::uint64_t{load_word(arguments, 36)} << 32);
        return reply;
    }

    void unmap(std::uint64_t address) {
        std::array<std::byte, 8> arguments{};
        store_qword(arguments, 0, address);
        CHECK(reply_word(ioctl(address_space, address_space_unmap_buffer, arguments), 32) == nv_success);
    }

    void close_address_space() {
        const auto reply = dispatch_request(request(2, address_space));
        CHECK(reply.disposition == IpcDisposition::handled);
    }

    std::array<std::byte, 0x20> allocate_gpfifo(std::uint32_t entries,
                                                std::uint32_t jobs) {
        std::array<std::byte, 0x20> arguments{};
        store_word(arguments, 0, entries);
        store_word(arguments, 4, jobs);
        const IpcReply reply =
            ioctl(channel, channel_allocate_gpfifo, arguments);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(reply, 32) == nv_success);
        return arguments;
    }

    void write_commands(std::uint64_t gpu_address,
                        std::span<const std::uint32_t> words) {
        const std::uint64_t guest_address =
            backing_address + buffer_offset + (gpu_address - gpu_base);
        std::vector<std::byte> bytes(words.size() * 4);
        for (std::size_t index = 0; index < words.size(); ++index) {
            store_word(bytes, index * 4, words[index]);
        }
        copy_to(guest_address, bytes);
    }

    struct Submission {
        IpcReply reply;
        std::array<std::byte, 0x18> arguments{};
    };

    Submission submit_bytes(std::uint32_t count,
                            std::span<const std::byte> inline_data,
                            std::uint32_t flags, std::uint32_t fence_id,
                            std::uint32_t fence_value) {
        Submission submission;
        store_qword(submission.arguments, 0, 0x11110000);
        store_word(submission.arguments, 8, count);
        store_word(submission.arguments, 12, flags);
        store_word(submission.arguments, 16, fence_id);
        store_word(submission.arguments, 20, fence_value);
        submission.reply = ioctl(channel, channel_submit_gpfifo,
                                 submission.arguments, ioctl2_command,
                                 inline_data, true);
        return submission;
    }

    Submission submit(std::span<const std::uint64_t> entries,
                      std::uint32_t flags, std::uint32_t fence_id,
                      std::uint32_t fence_value) {
        const std::vector<std::byte> inline_data = encode_entries(entries);
        return submit_bytes(static_cast<std::uint32_t>(entries.size()),
                            inline_data, flags, fence_id, fence_value);
    }

    std::pair<IpcReply, std::uint32_t> read_syncpoint(std::uint32_t operation,
                                                      std::uint32_t id) {
        std::array<std::byte, 8> arguments{};
        store_word(arguments, 0, id);
        IpcReply reply = ioctl(control, operation, arguments);
        return {std::move(reply), load_word(arguments, 4)};
    }

    std::uint32_t completed() {
        auto [reply, value] = read_syncpoint(syncpoint_read_min,
                                             expected_syncpoint);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(reply, 32) == nv_success);
        return value;
    }

    std::uint32_t reserved() {
        auto [reply, value] = read_syncpoint(syncpoint_read_max,
                                             expected_syncpoint);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(reply, 32) == nv_success);
        return value;
    }
};

std::vector<std::uint32_t> syncpoint_increments(std::uint32_t id,
                                                unsigned count) {
    std::vector<std::uint32_t> words{header(3, 0x1c, 0, 1), 0};
    for (unsigned index = 0; index < count; ++index) {
        words.push_back(header(3, 0x1d, 0, 1));
        words.push_back((id << 8) | 1);
    }
    return words;
}

// A DMA launch with the reserved semaphore type stays an unsupported command.
std::vector<std::uint32_t> unsupported_dma_packet() {
    return {header(3, 0, 4, 1), 0xb0b5,
            header(3, 0xc0, 4, 1), 3u << 3};
}

std::vector<std::uint32_t> rendering_stop_packet() {
    return {header(3, 0, 0, 1), 0xb197,
            header(1, 0x200, 0, 6), 0, static_cast<std::uint32_t>(gpu_base + 0x800), 16, 8, 0x37, 0,
            header(1, 0x3fd, 0, 2), 0x100002, 0x80003,
            header(1, 0x380, 0, 3), 1, 0xe0004, 0x70003,
            header(1, 0x280, 0, 6), 0x41000000, 0x40800000, 0x3f800000, 0x41000000, 0x40800000, 0,
            header(3, 0x487, 0, 1), 1,
            header(3, 0x680, 0, 1), 0x1111,
            header(1, 0x582, 0, 2), 0, static_cast<std::uint32_t>(gpu_base),
            header(1, 0x810, 0, 2), 1, 0x40,
            header(1, 0x5f2, 0, 7), 0, static_cast<std::uint32_t>(gpu_base + 0x100),
            0, static_cast<std::uint32_t>(gpu_base + 0x102), 0, 0, 3,
            header(3, 0x674, 0, 1), 4,
            header(3, 0x1d, 0, 1), (expected_syncpoint << 8) | 1};
}

void test_nvmap_free_output_contract() {
    for (const auto flags : {0u, 1u, 2u, 3u, 4u, 5u, 0x80000001u}) {
        PreparedFixture fixture;
        const auto handle = fixture.create_memory(0x1001);
        const auto address = backing_address + 0x10000;
        fixture.allocate_memory(handle, address, 0x1000, flags);
        // Repeated Alloc must not change the cache policy reported by Free.
        fixture.allocate_memory(handle, address, 0x1000, flags ^ 1u);
        fixture.export_memory(handle);
        std::array<std::byte, 0x18> arguments{};
        arguments.fill(std::byte{0xa5});
        store_word(arguments, 0, handle);
        CHECK(reply_word(fixture.ioctl(fixture.nvmap, nvmap_free, arguments), 32) == nv_success);
        CHECK(load_word(arguments, 0) == handle);
        CHECK(load_word(arguments, 4) == 0xa5a5a5a5);
        CHECK(load_qword(arguments, 8) == address);
        CHECK(load_word(arguments, 16) == 0x2000);
        CHECK(load_word(arguments, 20) == (flags & 1u));
        CHECK(fixture.state->unmapped.size() == 1);
        CHECK(!fixture.session->retain_exported_memory(handle));

        // The returned tuple supplies the exact range and cache policy needed
        // to restore CPU attributes before reusing caller-provided storage.
        const bool restore_cached = load_word(arguments, 20) == 0 &&
                                    load_qword(arguments, 8) != 0;
        CHECK(restore_cached == ((flags & 1u) == 0));
        const auto previous = arguments;
        CHECK(reply_word(fixture.ioctl(fixture.nvmap, nvmap_free, arguments), 32) == nv_success);
        CHECK(arguments == previous && fixture.state->unmapped.size() == 1);
    }

    PreparedFixture fixture;
    const auto unallocated = fixture.create_memory(0x1001);
    // Internally allocated storage is not implemented. A rejected attempt must
    // not turn a created object into firmware's internal-storage state.
    std::array<std::byte, 0x20> allocation{};
    store_word(allocation, 0, unallocated);
    store_word(allocation, 8, 2);
    CHECK(reply_word(fixture.ioctl(fixture.nvmap, nvmap_allocate, allocation), 32) == 9);
    std::array<std::byte, 0x18> arguments{};
    arguments.fill(std::byte{0xa5});
    store_word(arguments, 0, unallocated);
    CHECK(reply_word(fixture.ioctl(fixture.nvmap, nvmap_free, arguments), 32) == nv_success);
    CHECK(load_qword(arguments, 8) == 0);
    CHECK(load_word(arguments, 16) == 0x2000 && load_word(arguments, 20) == 0);
    CHECK(fixture.state->unmapped.empty());
}

void test_nvmap_free_import_references() {
    for (const bool keep_import : {false, true}) {
        PreparedFixture fixture;
        fixture.unmap(gpu_base);
        fixture.export_memory(fixture.memory_handle);
        auto imported = fixture.session->retain_exported_memory(fixture.memory_handle);
        CHECK(imported);
        if (!imported) continue;
        // Let a lease expire, then recreate it. The weak identity must track
        // the current import rather than permanently marking GetId as retained.
        std::weak_ptr<void> expired = imported->owner;
        imported.reset();
        CHECK(expired.expired());
        imported = fixture.session->retain_exported_memory(fixture.memory_handle);
        auto clone = std::make_shared<DriverSession>(*fixture.session);
        auto second = clone->retain_exported_memory(fixture.memory_handle);
        CHECK(imported && second);
        if (!imported || !second) continue;
        CHECK(!imported->owner.owner_before(second->owner) &&
              !second->owner.owner_before(imported->owner));
        second.reset();
        if (!keep_import) imported.reset();
        const auto freed = fixture.free_memory(fixture.memory_handle);
        CHECK(load_qword(freed, 8) == (keep_import ? 0 : backing_address));
        CHECK(load_word(freed, 16) == backing_size && load_word(freed, 20) == 0);
        CHECK(fixture.state->unmapped.size() == (keep_import ? 0u : 1u));
        imported.reset();
        CHECK(fixture.state->unmapped.size() == 1);
    }

    PreparedFixture fixture;
    fixture.unmap(gpu_base);
    fixture.export_memory(fixture.memory_handle);
    auto memory = std::make_shared<RecordingGuestMemory>();
    auto access = fixture.session->graphics_backing_access(memory);
    auto backing = access.lookup(fixture.memory_handle);
    CHECK(backing);
    if (!backing) return;
    auto pin = access.pin(*backing, 0, 0x1000);
    CHECK(std::holds_alternative<GuestMemoryPin>(pin));
    backing.reset();
    const auto freed = fixture.free_memory(fixture.memory_handle);
    CHECK(load_qword(freed, 8) == 0);
    CHECK(fixture.state->unmapped.empty());
    if (auto* retained = std::get_if<GuestMemoryPin>(&pin)) retained->reset();
    CHECK(fixture.state->unmapped.size() == 1);
}

void test_gpu_mapping_retains_freed_handle() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const auto commands = syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(gpu_base, commands);
    std::uint64_t alias = gpu_base + 0x10000;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, alias), 32) == nv_success);
    const auto freed = fixture.free_memory(fixture.memory_handle);
    CHECK(load_qword(freed, 8) == 0);
    CHECK(load_word(freed, 16) == backing_size);
    CHECK(load_word(freed, 20) == 0);
    CHECK(fixture.state->unmapped.empty());
    std::array<std::byte, 8> get_id{};
    store_word(get_id, 4, fixture.memory_handle);
    CHECK(reply_word(fixture.ioctl(fixture.nvmap, 0xc008010e, get_id), 32) == 0x30010);

    // An unknown address leaves both aliases intact.
    fixture.unmap(gpu_base + 0x80000);
    fixture.unmap(gpu_base + gpu_mapping_size);
    auto descriptor = entry(gpu_base, commands.size());
    CHECK(fixture.submit({&descriptor, 1}, 0, 0, 0).reply.disposition == IpcDisposition::handled);
    CHECK(fixture.completed() == 1);
    // An interior address removes the entire containing mapping.
    fixture.unmap(gpu_base + 0x1000);
    CHECK(fixture.state->unmapped.empty());
    descriptor = entry(alias, commands.size());
    CHECK(fixture.submit({&descriptor, 1}, 0, 0, 0).reply.disposition == IpcDisposition::handled);
    CHECK(fixture.completed() == 2);

    // UnmapBuffer consumes its input without requiring a writable descriptor.
    std::array<std::byte, 8> arguments{};
    store_qword(arguments, 0, alias + 0x1000);
    fixture.copy_to(argument_address, arguments);
    GuestBuffers buffers(Fixture::read_memory, Fixture::write_memory, fixture.state.get());
    buffers.add({argument_address, 8, true, false});
    fixture.state->fail_write = true;
    const auto reply = fixture.session->request_with_buffers(
        request(ioctl_command, fixture.address_space, address_space_unmap_buffer), 1, buffers);
    fixture.state->fail_write = false;
    CHECK(reply_word(reply, 32) == nv_success);
    CHECK(fixture.state->unmapped.size() == 1);
    if (!fixture.state->unmapped.empty())
        CHECK(fixture.state->unmapped.front() == std::make_pair(backing_address, backing_size));
    fixture.unmap(alias);
    CHECK(fixture.state->unmapped.size() == 1);

    // Only the GPU mapping retains the callback context after its host drops ownership.
    PreparedFixture retained;
    retained.free_memory(retained.memory_handle);
    retained.session->set_memory_mapper({});
    const std::weak_ptr<Fixture::MemoryState> owner = retained.state;
    retained.state.reset();
    CHECK(!owner.expired());
    retained.close_address_space();
    CHECK(owner.expired());
}

void test_mapping_requires_owned_allocated_backing() {
    PreparedFixture fixture(0x10000);
    const auto unallocated = fixture.create_memory(0x10000);
    std::uint64_t address = gpu_base + 0x10000;
    CHECK(reply_word(fixture.map_memory_handle(unallocated, address), 32) == 0x3000f);
    fixture.set_mapper(false);
    fixture.allocate_memory(unallocated, memory_base + 0x70000, 0x10000);
    CHECK(fixture.map_memory_handle(unallocated, address).disposition == IpcDisposition::unsupported);
    CHECK(fixture.session->diagnostic() != nullptr);

    std::array<std::byte, 0x18> reservation{};
    store_word(reservation, 0, 1);
    store_word(reservation, 4, 0x10000);
    store_word(reservation, 8, 2);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0184102, reservation), 32) == nv_success);
    const auto target = std::uint64_t{load_word(reservation, 16)} |
        (std::uint64_t{load_word(reservation, 20)} << 32);
    std::array<std::byte, 20> remap{};
    store_word(remap, 0, 4);
    store_word(remap, 4, unallocated);
    store_word(remap, 12, static_cast<std::uint32_t>(target / 0x10000));
    store_word(remap, 16, 1);
    CHECK(fixture.ioctl(fixture.address_space, 0xc0144114, remap).disposition == IpcDisposition::unsupported);
    CHECK(fixture.state->unmapped.empty());
    fixture.free_memory(unallocated);
    CHECK(fixture.state->unmapped.size() == 1);

    // Rejected mappings did not occupy their requested GPU address.
    fixture.set_mapper();
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, address), 32) == nv_success);
}

void test_unmap_releases_only_automatic_reservations() {
    PreparedFixture fixture;
    std::uint64_t automatic = 0;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, automatic, false), 32) == nv_success);
    const auto original = automatic;
    fixture.unmap(automatic);

    std::array<std::byte, 0x28> failed_mapping{};
    store_word(failed_mapping, 8, fixture.memory_handle);
    store_word(failed_mapping, 12, 0x1000);
    store_qword(failed_mapping, 16, buffer_offset);
    store_qword(failed_mapping, 24, gpu_mapping_size);
    fixture.copy_to(argument_address, failed_mapping);
    GuestBuffers buffers(Fixture::read_memory, Fixture::write_memory, fixture.state.get());
    buffers.add({argument_address, static_cast<std::uint32_t>(failed_mapping.size()), true, true});
    fixture.state->fail_write = true;
    CHECK(reply_word(fixture.session->request_with_buffers(
        request(ioctl_command, fixture.address_space, address_space_map_buffer), 1, buffers), 32) == 9);
    fixture.state->fail_write = false;
    automatic = 0;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, automatic, false), 32) == nv_success);
    CHECK(automatic == original);
    fixture.unmap(automatic);

    std::array<std::byte, 0x18> reservation{};
    store_word(reservation, 0, 2);
    store_word(reservation, 4, 0x1000);
    store_word(reservation, 8, 1);
    store_qword(reservation, 16, original);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0184102, reservation), 32) == nv_success);
    auto fixed = original;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, fixed), 32) == nv_success);
    fixture.unmap(fixed);
    automatic = 0;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, automatic, false), 32) == nv_success);
    CHECK(automatic == original + gpu_mapping_size);
    fixture.free_memory(fixture.memory_handle);
    CHECK(fixture.state->unmapped.empty());
    fixture.close_address_space();
    CHECK(fixture.state->unmapped.size() == 1);
}

void test_allocation_validation_and_first_syncpoint() {
    PreparedFixture fixture;

    std::array<std::byte, 0x20> direct{};
    store_word(direct, 0, 8);
    store_word(direct, 4, 1);
    IpcReply reply =
        fixture.ioctl(fixture.channel, channel_allocate_gpfifo, direct);
    CHECK(reply_word(reply, 32) == nv_invalid_value);

    direct.fill(std::byte{0});
    store_word(direct, 0, 24);
    store_word(direct, 4, 1);
    reply = fixture.ioctl(fixture.channel, channel_allocate_gpfifo, direct);
    CHECK(reply_word(reply, 32) == nv_invalid_value);

    direct.fill(std::byte{0});
    store_word(direct, 0, 16);
    reply = fixture.ioctl(fixture.channel, channel_allocate_gpfifo, direct);
    CHECK(reply_word(reply, 32) == nv_invalid_value);

    direct.fill(std::byte{0});
    store_word(direct, 0, 16);
    store_word(direct, 4, 1);
    reply = fixture.ioctl(fixture.channel, channel_allocate_gpfifo, direct);
    CHECK(reply_word(reply, 32) == nv_success);
    CHECK(load_word(direct, 12) == expected_syncpoint);
    CHECK(load_word(direct, 16) == 0);
}

void test_reservation_and_guest_syncpoint_methods() {
    PreparedFixture fixture;
    const auto allocation = fixture.allocate_gpfifo(16, 1);
    CHECK(load_word(allocation, 12) == expected_syncpoint);

    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base, nop);
    const std::uint64_t first_entry = entry(gpu_base, nop.size());
    auto reservation = fixture.submit({&first_entry, 1}, 0x104, 77, 3);
    CHECK(reservation.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(reservation.reply, 32) == nv_success);
    CHECK(load_word(reservation.arguments, 12) == 0);
    CHECK(load_word(reservation.arguments, 16) == expected_syncpoint);
    CHECK(load_word(reservation.arguments, 20) == 3);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 3);
    CHECK(fixture.session->is_fence_expired(expected_syncpoint, 0) == true);
    CHECK(fixture.session->is_fence_expired(expected_syncpoint, 3) == false);
    CHECK(fixture.session->is_fence_expired(expected_syncpoint, 4) == true);
    CHECK(!fixture.session->is_fence_expired(192, 0));
    auto clone = std::make_shared<DriverSession>(*fixture.session);
    CHECK(clone->is_fence_expired(expected_syncpoint, 3) == false);
    CHECK(fixture.completed() == 0 && fixture.reserved() == 3);

    const std::vector<std::uint32_t> increments =
        syncpoint_increments(expected_syncpoint, 3);
    const std::uint64_t second_gpu_address = gpu_base + 0x100;
    fixture.write_commands(second_gpu_address, increments);
    const std::uint64_t second_entry =
        entry(second_gpu_address, increments.size());
    const auto completion = fixture.submit({&second_entry, 1}, 0, 0, 0);
    CHECK(completion.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(completion.reply, 32) == nv_success);
    CHECK(fixture.completed() == 3);
    CHECK(fixture.reserved() == 3);
    CHECK(fixture.session->is_fence_expired(expected_syncpoint, 3) == true);
    CHECK(clone->is_fence_expired(expected_syncpoint, 3) == true);
    CHECK(fixture.session->is_fence_expired(expected_syncpoint, 4) == true);
    CHECK(fixture.completed() == 3 && fixture.reserved() == 3);
}

void test_client_scoped_graphics_backing_access() {
    PreparedFixture fixture;
    fixture.export_memory(fixture.memory_handle);
    auto memory = std::make_shared<RecordingGuestMemory>();
    auto access = fixture.session->graphics_backing_access(memory);
    auto backing = access.lookup(fixture.memory_handle);
    CHECK(backing);
    if (!backing) return;

    CHECK(backing->export_id == fixture.memory_handle);
    CHECK(backing->guest_address == backing_address);
    CHECK(backing->size == backing_size);
    auto clone = std::make_shared<DriverSession>(*fixture.session);
    auto clone_access = clone->graphics_backing_access(memory);
    auto clone_backing = clone_access.lookup(fixture.memory_handle);
    CHECK(clone_backing && clone_backing->owner == backing->owner);

    auto pinned = access.pin(*backing, buffer_offset, gpu_mapping_size);
    CHECK(std::holds_alternative<GuestMemoryPin>(pinned));
    CHECK(memory->pin_calls == 1);
    CHECK(memory->last_range.address == backing_address + buffer_offset);
    CHECK(memory->last_range.size == gpu_mapping_size);
    CHECK(memory->last_access == armigo::memory::Permission::read);

    auto forged = *backing;
    ++forged.guest_address;
    CHECK(failed_with(access.pin(forged, 0, 1), LaunchError::permission_denied));
    forged = *backing;
    ++forged.size;
    CHECK(failed_with(access.pin(forged, 0, 1), LaunchError::permission_denied));
    forged = *backing;
    ++forged.alignment;
    CHECK(failed_with(access.pin(forged, 0, 1), LaunchError::permission_denied));
    forged = *backing;
    ++forged.export_id;
    CHECK(failed_with(access.pin(forged, 0, 1), LaunchError::permission_denied));
    CHECK(failed_with(access.pin(*backing, backing->size, 1),
                      LaunchError::invalid_input));
    CHECK(failed_with(access.pin(*backing, 0, 0), LaunchError::invalid_input));
    CHECK(memory->pin_calls == 1);

    PreparedFixture other;
    other.export_memory(other.memory_handle);
    auto other_access = other.session->graphics_backing_access(memory);
    auto other_backing = other_access.lookup(other.memory_handle);
    CHECK(other.memory_handle == fixture.memory_handle && other_backing);
    if (other_backing) {
        CHECK(other_backing->owner != backing->owner);
        CHECK(failed_with(access.pin(*other_backing, 0, 1),
                          LaunchError::permission_denied));
    }

    const auto closed = fixture.session->request(request(2, fixture.nvmap), 1);
    CHECK(closed.disposition == IpcDisposition::handled);
    fixture.nvmap = fixture.open("/dev/nvmap");
    CHECK(access.lookup(fixture.memory_handle));
    fixture.free_memory(fixture.memory_handle);
    CHECK(!access.lookup(fixture.memory_handle));

    const auto reused_handle = fixture.create_memory(backing_size);
    fixture.allocate_memory(reused_handle, backing_address);
    fixture.export_memory(reused_handle);
    auto reused = access.lookup(reused_handle);
    CHECK(reused && reused->guest_address == backing->guest_address);
    if (reused) {
        auto mismatched = *backing;
        mismatched.export_id = reused->export_id;
        CHECK(failed_with(access.pin(mismatched, 0, 1),
                          LaunchError::permission_denied));
    }
    CHECK(std::holds_alternative<GuestMemoryPin>(
        access.pin(*backing, buffer_offset, gpu_mapping_size)));

    fixture.close_address_space();
    CHECK(fixture.state->unmapped.empty());
    forged.owner.reset();
    clone_backing.reset();
    backing.reset();
    CHECK(fixture.state->unmapped.empty());
    std::weak_ptr<RecordingGuestMemory> authority = memory;
    memory.reset();
    CHECK(authority.expired());
    CHECK(std::get<GuestMemoryPin>(pinned)->bytes().size() == gpu_mapping_size);
    if (reused) CHECK(failed_with(access.pin(*reused, 0, 1), LaunchError::invalid_input));
    std::get<GuestMemoryPin>(pinned).reset();
    CHECK(fixture.state->unmapped.size() == 1);
}

void test_graphics_fence_waits() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base, nop);
    const auto nop_entry = entry(gpu_base, nop.size());
    const auto reservation = fixture.submit({&nop_entry, 1}, 0x104, 0, 1);
    CHECK(reply_word(reservation.reply, 32) == nv_success);
    CHECK(fixture.completed() == 0 && fixture.reserved() == 1);

    auto access = fixture.session->graphics_backing_access(nullptr);
    CHECK(failed_with(access.wait_fence({-1, 0}, std::nullopt, {}),
                      LaunchError::invalid_input));
    CHECK(failed_with(access.wait_fence({192, 0}, std::nullopt, {}),
                      LaunchError::invalid_input));
    CHECK(succeeded(access.wait_fence(
        {static_cast<std::int32_t>(expected_syncpoint), 0}, std::nullopt, {})));
    CHECK(failed_with(access.wait_fence(
        {static_cast<std::int32_t>(expected_syncpoint), 1},
        std::chrono::steady_clock::now(), {}), LaunchError::deadline_expired));

    std::promise<void> cancellation_started;
    auto cancellation_ready = cancellation_started.get_future();
    std::stop_source stop;
    auto cancelled = std::async(std::launch::async, [&] {
        cancellation_started.set_value();
        return access.wait_fence(
            {static_cast<std::int32_t>(expected_syncpoint), 1}, std::nullopt,
            stop.get_token());
    });
    cancellation_ready.wait();
    CHECK(cancelled.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout);
    stop.request_stop();
    CHECK(cancelled.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    CHECK(failed_with(cancelled.get(), LaunchError::cancelled));

    std::promise<void> wait_started;
    auto wait_ready = wait_started.get_future();
    auto completed = std::async(std::launch::async, [&] {
        wait_started.set_value();
        return access.wait_fence(
            {static_cast<std::int32_t>(expected_syncpoint), 1},
            std::chrono::steady_clock::now() + std::chrono::seconds(1), {});
    });
    wait_ready.wait();
    CHECK(completed.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout);
    PreparedFixture other_client;
    other_client.allocate_gpfifo(16, 1);
    const auto foreign_increment = syncpoint_increments(expected_syncpoint, 1);
    other_client.write_commands(gpu_base, foreign_increment);
    const auto foreign_entry = entry(gpu_base, foreign_increment.size());
    CHECK(reply_word(other_client.submit({&foreign_entry, 1}, 0, 0, 0).reply, 32) == nv_success);
    CHECK(completed.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout);
    const auto increments = syncpoint_increments(expected_syncpoint, 1);
    const auto increment_address = gpu_base + 0x100;
    fixture.write_commands(increment_address, increments);
    const auto increment_entry = entry(increment_address, increments.size());
    CHECK(reply_word(fixture.submit({&increment_entry, 1}, 0, 0, 0).reply, 32) ==
          nv_success);
    CHECK(completed.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    CHECK(succeeded(completed.get()));
}

void test_automatic_increment_requires_completed_work() {
    PreparedFixture successful;
    successful.allocate_gpfifo(16, 1);
    const std::array<std::uint32_t, 1> nop{0};
    successful.write_commands(gpu_base, nop);
    const std::uint64_t nop_entry = entry(gpu_base, 1);
    const auto completed = successful.submit({&nop_entry, 1}, 2, 0, 0);
    CHECK(completed.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(completed.reply, 32) == nv_success);
    CHECK(load_word(completed.arguments, 20) == 2);
    CHECK(successful.completed() == 2);
    CHECK(successful.reserved() == 2);

    PreparedFixture failed;
    failed.allocate_gpfifo(16, 1);
    const std::vector<std::uint32_t> unsupported = unsupported_dma_packet();
    failed.write_commands(gpu_base, unsupported);
    const std::uint64_t unsupported_entry =
        entry(gpu_base, unsupported.size());
    const auto stopped = failed.submit({&unsupported_entry, 1}, 2, 0, 0);
    CHECK(stopped.reply.disposition == IpcDisposition::unsupported);
    CHECK(failed.completed() == 0);
    CHECK(failed.reserved() == 2);
}

void test_invalid_submission_inputs_have_no_effects() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base, nop);
    const std::uint64_t descriptor = entry(gpu_base, 1);

    const auto invalid_flags = fixture.submit({&descriptor, 1}, 0x101, 0, 9);
    CHECK(invalid_flags.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(invalid_flags.reply, 32) == nv_invalid_value);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 0);

    std::array<std::byte, 4> short_inline{};
    const auto short_submit =
        fixture.submit_bytes(1, short_inline, 0x104, 0, 7);
    CHECK(short_submit.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(short_submit.reply, 32) == nv_invalid_size);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 0);
}

void test_incomplete_automatic_increment_is_sticky() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const std::array incomplete{header(3, 2, 0, 1)};
    fixture.write_commands(gpu_base, incomplete);
    const std::uint64_t descriptor = entry(gpu_base, incomplete.size());

    const auto stopped = fixture.submit({&descriptor, 1}, 2, 0, 0);
    CHECK(stopped.reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 2);

    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base + 0x100, nop);
    const std::uint64_t later = entry(gpu_base + 0x100, 1);
    const auto sticky = fixture.submit({&later, 1}, 0x104, 0, 5);
    CHECK(sticky.reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 2);
}

void test_incomplete_upload_does_not_complete_automatic_fence() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const std::array commands{
        header(3, 0, 0, 1), 0xb197u,
        header(1, 0x60, 0, 4), 8u, 1u, 0u, static_cast<std::uint32_t>(gpu_base + 0x800),
        header(3, 0x6c, 0, 1), 0x1011u,
        header(3, 0x6d, 0, 1), 0x11223344u,
    };
    fixture.write_commands(gpu_base, commands);
    const auto descriptor = entry(gpu_base, commands.size());
    const auto submission = fixture.submit({&descriptor, 1}, 2, 0, 0);
    CHECK(submission.reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 2);
    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base + 0x100, nop);
    const auto later = entry(gpu_base + 0x100, 1);
    CHECK(fixture.submit({&later, 1}, 0x104, 0, 3).reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.reserved() == 2);
}

void test_rendering_stop_is_pending_and_sticky() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const std::vector<std::uint32_t> commands = rendering_stop_packet();
    fixture.write_commands(gpu_base, commands);
    const std::uint64_t descriptor = entry(gpu_base, commands.size());

    const auto stopped = fixture.submit({&descriptor, 1}, 0x104, 0, 1);
    CHECK(stopped.reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.session->diagnostic() != nullptr);
    CHECK(std::string(fixture.session->diagnostic()).find(
              "first rendering operation") != std::string::npos);
    const std::string diagnostic = fixture.session->diagnostic();
    CHECK(diagnostic.find("first rendering operation clear") != std::string::npos);
    CHECK(diagnostic.find("color[0] address 0x9000800 format 0x37 layout 0x0 size 16x8") != std::string::npos);
    CHECK(diagnostic.find("program[1] address 0x9000040 offset 0x40 control 0x1 enabled 1") != std::string::npos);
    CHECK(diagnostic.find("surface clip 0x100002,0x80003") != std::string::npos);
    CHECK(diagnostic.find("scissor[0] 0x1,0xe0004,0x70003") != std::string::npos);
    CHECK(diagnostic.find("viewport[0] 0x41000000,0x40800000,0x3f800000,0x41000000,0x40800000,0x0") != std::string::npos);
    CHECK(diagnostic.find("render targets 0x1 color mask[0] 0x1111") != std::string::npos);
    CHECK(diagnostic.find("index address 0x9000100") != std::string::npos);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 1);

    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base + 0x100, nop);
    const std::uint64_t later = entry(gpu_base + 0x100, 1);
    const auto sticky = fixture.submit({&later, 1}, 0x104, 0, 4);
    CHECK(sticky.reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 1);
}

void test_faults_and_unsupported_actions_do_not_complete() {
    PreparedFixture faulted;
    faulted.allocate_gpfifo(16, 1);
    const std::uint64_t missing = entry(gpu_base + gpu_mapping_size, 1);
    const auto memory_fault = faulted.submit({&missing, 1}, 0x104, 0, 1);
    CHECK(memory_fault.reply.disposition == IpcDisposition::unsupported);
    CHECK(faulted.completed() == 0);
    CHECK(faulted.reserved() == 1);

    PreparedFixture unsupported;
    unsupported.allocate_gpfifo(16, 1);
    const std::vector<std::uint32_t> commands = unsupported_dma_packet();
    unsupported.write_commands(gpu_base, commands);
    const std::uint64_t descriptor = entry(gpu_base, commands.size());
    const auto action = unsupported.submit({&descriptor, 1}, 0x104, 0, 1);
    CHECK(action.reply.disposition == IpcDisposition::unsupported);
    CHECK(unsupported.completed() == 0);
    CHECK(unsupported.reserved() == 1);
}

struct TestRenderer {
    unsigned calls = 0;
    bool fail = false;
    static armigo::graphics::CommandResult clear(
        void* raw, const armigo::graphics::RenderStop& snapshot,
        const armigo::graphics::GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<TestRenderer*>(raw);
        ++self.calls;
        CHECK(snapshot.kind == armigo::graphics::RenderKind::Clear);
        if (self.fail)
            return {armigo::graphics::CommandStatus::Unsupported, "Synthetic render failure"};
        constexpr std::array pixel{std::byte{0x42}, std::byte{0x21},
                                   std::byte{0x84}, std::byte{0xff}};
        CHECK(memory.write(gpu_base + 0x800, pixel).ok());
        return {};
    }
};

void test_backend_completion_and_clone_sharing() {
    for (const bool fail : {false, true}) {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        TestRenderer renderer{0, fail};
        DriverSession clone(*fixture.session);
        clone.set_render_backend({TestRenderer::clear, &renderer, nullptr, {}});
        const auto commands = rendering_stop_packet();
        fixture.write_commands(gpu_base, commands);
        const auto descriptor = entry(gpu_base, commands.size());
        const auto submitted = fixture.submit({&descriptor, 1}, 0x104, 0, 1);
        const std::string backend_diagnostic = fixture.session->diagnostic()
            ? fixture.session->diagnostic() : "";
        CHECK(renderer.calls == 1);
        CHECK(fixture.reserved() == 1);
        CHECK(fixture.completed() == (fail ? 0u : 1u));
        CHECK(submitted.reply.disposition == (fail ? IpcDisposition::unsupported
                                                   : IpcDisposition::handled));
        if (fail) {
            CHECK(backend_diagnostic.find("Synthetic render failure") !=
                  std::string::npos);
            CHECK(fixture.submit({&descriptor, 1}, 0x104, 0, 1).reply.disposition ==
                  IpcDisposition::unsupported);
            CHECK(renderer.calls == 1);
        }
    }
}

constexpr std::uint64_t completed_draw_address = gpu_base + 0x800;
constexpr std::uint32_t completed_draw_value = 0x11223344;

constexpr std::uint64_t guest_backing_address(std::uint64_t gpu_address) {
    return backing_address + buffer_offset + gpu_address - gpu_base;
}

std::uint32_t backing_word(const PreparedFixture& fixture,
                           std::uint64_t gpu_address) {
    return load_word(fixture.state->memory,
                     guest_backing_address(gpu_address) - memory_base);
}

struct OrderedDrawRenderer {
    unsigned calls = 0;
    bool fail = false;
    std::uint64_t rewrite_address = 0;
    std::uint32_t rewrite_value = 0;

    static armigo::graphics::CommandResult draw(
        void* raw, const armigo::graphics::RenderStop& snapshot,
        const armigo::graphics::GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<OrderedDrawRenderer*>(raw);
        ++self.calls;
        CHECK(snapshot.kind == armigo::graphics::RenderKind::Draw);
        CHECK(snapshot.draw.has_value());
        if (snapshot.draw) {
            CHECK(!snapshot.draw->indexed);
            CHECK(snapshot.draw->topology == 4);
            CHECK(snapshot.draw->first == 0 && snapshot.draw->count == 3);
        }
        if (self.fail) {
            return {armigo::graphics::CommandStatus::Unsupported,
                    "Synthetic synchronous draw failure"};
        }
        // These writes model completed GPU work and its required writeback.
        std::array<std::byte, 4> bytes{};
        store_word(bytes, 0, completed_draw_value);
        if (!memory.write(completed_draw_address, bytes).ok()) {
            return {armigo::graphics::CommandStatus::MemoryFault,
                    "Synthetic draw writeback failure"};
        }
        if (self.rewrite_address != 0) {
            store_word(bytes, 0, self.rewrite_value);
            if (!memory.write(self.rewrite_address, bytes).ok()) {
                return {armigo::graphics::CommandStatus::MemoryFault,
                        "Synthetic command writeback failure"};
            }
        }
        return {};
    }
};

std::vector<std::uint32_t> normal_draw_packet() {
    return {header(3, 0, 0, 1), 0xb197,
            header(1, 0x35d, 0, 2), 0, 3,
            header(3, 0x586, 0, 1), 4,
            header(3, 0x585, 0, 1), 0};
}

struct AsyncEffectRenderer {
    mutable std::mutex mutex;
    std::deque<armigo::graphics::GuestEffect> pending;
    unsigned coherence_calls = 0;
    unsigned read_coherence_calls = 0;
    unsigned coherence_publications = 0;
    unsigned coherence_writes = 0;
    bool reject = false;

    static armigo::graphics::CommandResult ordered_effect(
        void* raw, const armigo::graphics::GpuAddressSpace&,
        armigo::graphics::GuestEffect effect) noexcept {
        auto& self = *static_cast<AsyncEffectRenderer*>(raw);
        try {
            if (self.reject) {
                return {armigo::graphics::CommandStatus::Unsupported,
                        "Synthetic ordered-effect rejection"};
            }
            const std::lock_guard lock(self.mutex);
            self.pending.push_back(std::move(effect));
            return {};
        } catch (...) {
            return {armigo::graphics::CommandStatus::BudgetExceeded,
                    "Synthetic ordered-effect allocation failure"};
        }
    }

    static armigo::graphics::CommandResult coherence(
        void* raw, const armigo::graphics::GpuAddressSpace& memory,
        std::uint64_t, std::uint64_t) noexcept {
        auto& self = *static_cast<AsyncEffectRenderer*>(raw);
        ++self.coherence_calls;
        try {
            while (true) {
                armigo::graphics::GuestEffect effect;
                {
                    const std::lock_guard lock(self.mutex);
                    if (self.pending.empty()) return {};
                    if (self.pending.front().size == 0 &&
                        !self.pending.front().publish) {
                        return {armigo::graphics::CommandStatus::Unsupported,
                                "Synthetic coherence received an empty effect"};
                    }
                    effect = self.pending.front();
                }
                if (effect.size != 0) {
                    const auto written = memory.write(
                        effect.address,
                        std::span(effect.payload).first(effect.size));
                    if (!written) {
                        return {armigo::graphics::CommandStatus::MemoryFault,
                                written.detail};
                    }
                    ++self.coherence_writes;
                }
                const std::optional<armigo::graphics::CommandResult> result =
                    effect.publish ? effect.publish()
                                   : armigo::graphics::CommandResult{};
                if (!result) {
                    return {armigo::graphics::CommandStatus::Stopped,
                            "Synthetic coherence publication remained busy"};
                }
                {
                    const std::lock_guard lock(self.mutex);
                    self.pending.pop_front();
                }
                if (effect.publish) ++self.coherence_publications;
                if (!result->ok()) return *result;
            }
        } catch (...) {
            return {armigo::graphics::CommandStatus::BudgetExceeded,
                    "Synthetic coherence allocation failure"};
        }
    }

    static armigo::graphics::CommandResult read_coherence(
        void* raw, const armigo::graphics::GpuAddressSpace& memory,
        std::uint64_t address, std::uint64_t size) noexcept {
        auto& self = *static_cast<AsyncEffectRenderer*>(raw);
        ++self.read_coherence_calls;
        return coherence(raw, memory, address, size);
    }

    armigo::graphics::RenderCallbacks callbacks() {
        armigo::graphics::RenderCallbacks result{};
        result.context = this;
        result.coherence = coherence;
        result.ordered_effect = ordered_effect;
        result.read_coherence = read_coherence;
        return result;
    }

    std::size_t pending_count() const {
        const std::lock_guard lock(mutex);
        return pending.size();
    }

    std::optional<armigo::graphics::CommandResult> publish_front_async() {
        armigo::graphics::GuestPublication publication;
        {
            const std::lock_guard lock(mutex);
            if (pending.empty() || !pending.front().publish) return std::nullopt;
            publication = pending.front().publish;
        }
        auto worker = std::async(std::launch::async,
            [publication = std::move(publication)]() mutable {
                return publication();
            });
        auto result = worker.get();
        if (result) {
            const std::lock_guard lock(mutex);
            pending.pop_front();
        }
        return result;
    }
};

struct OrderedWriteOnlyRenderer {
    struct Write {
        std::uint64_t address = 0;
        std::vector<std::byte> payload{};
    };

    std::deque<Write> pending{};
    unsigned ordered_write_calls = 0;
    unsigned read_coherence_calls = 0;
    unsigned applied_writes = 0;

    static armigo::graphics::CommandResult ordered_write(
        void* raw, const armigo::graphics::GpuAddressSpace&,
        std::uint64_t address, std::span<const std::byte> payload) noexcept {
        auto& self = *static_cast<OrderedWriteOnlyRenderer*>(raw);
        ++self.ordered_write_calls;
        try {
            Write write;
            write.address = address;
            write.payload.assign(payload.begin(), payload.end());
            self.pending.push_back(std::move(write));
            return {};
        } catch (...) {
            return {armigo::graphics::CommandStatus::BudgetExceeded,
                    "Synthetic ordered-write allocation failure"};
        }
    }

    static armigo::graphics::CommandResult read_coherence(
        void* raw, const armigo::graphics::GpuAddressSpace& memory,
        std::uint64_t address, std::uint64_t size) noexcept {
        auto& self = *static_cast<OrderedWriteOnlyRenderer*>(raw);
        ++self.read_coherence_calls;
        std::size_t count = 0;
        for (std::size_t index = 0; index != self.pending.size(); ++index) {
            const auto& write = self.pending[index];
            if (write.address < address + size &&
                address < write.address + write.payload.size()) {
                count = index + 1;
            }
        }
        while (count != 0) {
            --count;
            const auto& write = self.pending.front();
            const auto written = memory.write(write.address, write.payload);
            if (!written) {
                return {armigo::graphics::CommandStatus::MemoryFault,
                        written.detail};
            }
            self.pending.pop_front();
            ++self.applied_writes;
        }
        return {};
    }

    armigo::graphics::RenderCallbacks callbacks() noexcept {
        armigo::graphics::RenderCallbacks result{};
        result.context = this;
        result.read_coherence = read_coherence;
        result.ordered_write = ordered_write;
        return result;
    }
};

void test_async_increment_waits_for_idle_publication() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    AsyncEffectRenderer renderer;
    fixture.session->set_render_backend(renderer.callbacks());

    const auto commands = syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(gpu_base, commands);
    const auto descriptor = entry(gpu_base, commands.size());
    const auto submitted = fixture.submit({&descriptor, 1}, 0x100, 0, 1);
    CHECK(submitted.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(submitted.reply, 32) == nv_success);
    CHECK(renderer.pending_count() == 1);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 1);

    std::array<std::byte, 16> wait{};
    store_word(wait, 0, expected_syncpoint);
    store_word(wait, 4, 1);
    store_word(wait, 8, 1);
    const auto wait_reply = fixture.ioctl(
        fixture.control, syncpoint_wait_allocate, wait);
    CHECK(wait_reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(wait_reply, 32) == nv_timeout);
    const auto event_reply = fixture.query_event(
        fixture.control, load_word(wait, 12));
    CHECK(event_reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(event_reply, 32) == nv_success);
    const auto event = event_reply.transferred_count == 1
        ? std::dynamic_pointer_cast<Event>(event_reply.transferred[0])
        : nullptr;
    CHECK(event != nullptr);
    CHECK(event != nullptr && !event->is_signaled());

    const auto published = renderer.publish_front_async();
    CHECK(published.has_value());
    CHECK(published && published->ok());
    CHECK(renderer.pending_count() == 0);
    CHECK(fixture.completed() == 1);
    CHECK(event != nullptr && event->is_signaled());
}

void test_async_automatic_fence_preserves_order_and_count() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    AsyncEffectRenderer renderer;
    fixture.session->set_render_backend(renderer.callbacks());

    const auto commands = syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(gpu_base, commands);
    const auto descriptor = entry(gpu_base, commands.size());
    const auto submitted = fixture.submit({&descriptor, 1}, 0x102, 0, 1);
    CHECK(submitted.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(submitted.reply, 32) == nv_success);
    CHECK(load_word(submitted.arguments, 20) == 3);
    CHECK(renderer.pending_count() == 2);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 3);

    const auto explicit_increment = renderer.publish_front_async();
    CHECK(explicit_increment.has_value());
    CHECK(explicit_increment && explicit_increment->ok());
    CHECK(fixture.completed() == 1);
    CHECK(renderer.pending_count() == 1);

    const auto automatic_increment = renderer.publish_front_async();
    CHECK(automatic_increment.has_value());
    CHECK(automatic_increment && automatic_increment->ok());
    CHECK(fixture.completed() == 3);
    CHECK(renderer.pending_count() == 0);
}

void test_input_fence_flushes_pending_publication_inline() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    AsyncEffectRenderer renderer;
    fixture.session->set_render_backend(renderer.callbacks());

    // An empty submission isolates the input wait from command-fetch coherence.
    const auto first = fixture.submit({}, 2, 0, 0);
    CHECK(first.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(first.reply, 32) == nv_success);
    CHECK(renderer.pending_count() == 1);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 2);
    const auto fence_id = load_word(first.arguments, 16);
    const auto fence_value = load_word(first.arguments, 20);
    const auto second = fixture.submit({}, 1, fence_id, fence_value);
    CHECK(second.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(second.reply, 32) == nv_success);
    CHECK(renderer.coherence_calls == 1);
    CHECK(renderer.coherence_publications == 1);
    CHECK(renderer.pending_count() == 0);
    CHECK(fixture.completed() == 2);
    CHECK(fixture.reserved() == 2);

    // A reserved increment without a publication must remain incomplete.
    CHECK(fixture.submit({}, 0x100, 0, 1).reply.disposition == IpcDisposition::handled);
    const auto incomplete = fixture.submit({}, 1, fence_id, fence_value + 1);
    CHECK(incomplete.reply.disposition == IpcDisposition::unsupported);
    CHECK(std::string_view(fixture.session->diagnostic()) ==
          "GPU submission is waiting for an incomplete input fence");
    CHECK(renderer.coherence_calls == 2);
    CHECK(fixture.completed() == 2);
    CHECK(fixture.reserved() == 3);
}

void test_syncpoint_acquire_flushes_pending_publication_inline() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    AsyncEffectRenderer renderer;
    fixture.session->set_render_backend(renderer.callbacks());

    auto commands = syncpoint_increments(expected_syncpoint, 1);
    const std::array acquire{
        header(3, 0x1c, 0, 1), 1u,
        header(3, 0x1d, 0, 1), expected_syncpoint << 8};
    commands.insert(commands.end(), acquire.begin(), acquire.end());
    fixture.write_commands(gpu_base, commands);
    const auto descriptor = entry(gpu_base, commands.size());
    const auto acquired = fixture.submit({&descriptor, 1}, 0x100, 0, 1);
    CHECK(acquired.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(acquired.reply, 32) == nv_success);
    CHECK(renderer.coherence_calls >= 1);
    CHECK(renderer.coherence_publications == 1);
    CHECK(renderer.pending_count() == 0);
    CHECK(fixture.completed() == 1);
}

void test_ordered_write_precedes_next_command_fetch() {
    constexpr auto next_address = gpu_base + 0x400;
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    AsyncEffectRenderer renderer;
    fixture.session->set_render_backend(renderer.callbacks());

    const std::array release{
        header(1, 4, 0, 2), 0u,
        static_cast<std::uint32_t>(next_address + 4),
        header(3, 0x1b, 0, 1), (expected_syncpoint << 8) | 1};
    const std::array stale_increment{
        header(3, 0x1d, 0, 1), ((expected_syncpoint + 1) << 8) | 1};
    fixture.write_commands(gpu_base, release);
    fixture.write_commands(next_address, stale_increment);
    const std::array descriptors{
        entry(gpu_base, release.size()),
        entry(next_address, stale_increment.size())};
    const auto submitted = fixture.submit(descriptors, 0x100, 0, 1);
    CHECK(submitted.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(submitted.reply, 32) == nv_success);
    CHECK(renderer.coherence_writes == 1);
    CHECK(renderer.read_coherence_calls > 0);
    CHECK(renderer.read_coherence_calls == renderer.coherence_calls);
    CHECK(backing_word(fixture, next_address + 4) ==
          ((expected_syncpoint << 8) | 1));
    CHECK(renderer.pending_count() == 1);
    CHECK(fixture.completed() == 0);

    const auto published = renderer.publish_front_async();
    CHECK(published.has_value());
    CHECK(published && published->ok());
    CHECK(fixture.completed() == 1);
    CHECK(fixture.reserved() == 1);
}

void test_ordered_write_only_backend_precedes_next_command_fetch() {
    constexpr auto next_address = gpu_base + 0x400;
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    OrderedWriteOnlyRenderer renderer;
    fixture.session->set_render_backend(renderer.callbacks());

    const std::array upload{
        header(3, 0, 1, 1), 0xb197u,
        header(1, 0x8e0, 1, 4), 4u, 0u,
        static_cast<std::uint32_t>(next_address + 4), 0u,
        header(3, 0x8e4, 1, 1), (expected_syncpoint << 8) | 1};
    const std::array stale_increment{
        header(3, 0x1d, 0, 1), ((expected_syncpoint + 1) << 8) | 1};
    fixture.write_commands(gpu_base, upload);
    fixture.write_commands(next_address, stale_increment);
    const std::array descriptors{
        entry(gpu_base, upload.size()),
        entry(next_address, stale_increment.size())};

    const auto submitted = fixture.submit(descriptors, 0x100, 0, 1);
    CHECK(submitted.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(submitted.reply, 32) == nv_success);
    CHECK(renderer.ordered_write_calls == 1);
    CHECK(renderer.read_coherence_calls > 0);
    CHECK(renderer.applied_writes == 1);
    CHECK(renderer.pending.empty());
    CHECK(backing_word(fixture, next_address + 4) ==
          ((expected_syncpoint << 8) | 1));
    CHECK(fixture.completed() == 1);
    CHECK(fixture.reserved() == 1);
}

void test_async_publication_rejection_and_driver_teardown() {
    {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        AsyncEffectRenderer renderer;
        renderer.reject = true;
        fixture.session->set_render_backend(renderer.callbacks());
        const std::array<std::uint32_t, 1> nop{0};
        fixture.write_commands(gpu_base, nop);
        const auto descriptor = entry(gpu_base, nop.size());
        const auto rejected = fixture.submit({&descriptor, 1}, 2, 0, 0);
        CHECK(rejected.reply.disposition == IpcDisposition::unsupported);
        CHECK(renderer.pending_count() == 0);
        CHECK(fixture.completed() == 0);
        CHECK(fixture.reserved() == 2);
    }

    AsyncEffectRenderer renderer;
    std::weak_ptr<DriverSession> lifetime;
    {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        fixture.session->set_render_backend(renderer.callbacks());
        lifetime = fixture.session;
        const auto commands = syncpoint_increments(expected_syncpoint, 1);
        fixture.write_commands(gpu_base, commands);
        const auto descriptor = entry(gpu_base, commands.size());
        CHECK(fixture.submit({&descriptor, 1}, 0, 0, 0)
                  .reply.disposition == IpcDisposition::handled);
        CHECK(renderer.pending_count() == 1);
    }
    CHECK(lifetime.expired());
    const auto cancelled = renderer.publish_front_async();
    CHECK(cancelled.has_value());
    CHECK(cancelled && cancelled->ok());
    CHECK(renderer.pending_count() == 0);
}

void test_render_backend_owner_survives_registration_clone() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    std::weak_ptr<OrderedDrawRenderer> lifetime;
    {
        auto renderer = std::make_shared<OrderedDrawRenderer>();
        lifetime = renderer;
        DriverSession registering_clone(*fixture.session);
        registering_clone.set_render_backend(
            {nullptr, renderer.get(), OrderedDrawRenderer::draw, renderer});
    }
    CHECK(!lifetime.expired());
    if (lifetime.expired()) return;
    const auto commands = normal_draw_packet();
    fixture.write_commands(gpu_base, commands);
    const auto descriptor = entry(gpu_base, commands.size());
    CHECK(fixture.submit({&descriptor, 1}, 2, 0, 0).reply.disposition ==
          IpcDisposition::handled);
    CHECK(lifetime.lock()->calls == 1);
    CHECK(fixture.completed() == 2);
    fixture.session->set_render_backend({});
    CHECK(lifetime.expired());
}

void test_failed_render_owner_survives_backend_replacement() {
    std::weak_ptr<OrderedDrawRenderer> lifetime;
    {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        {
            auto renderer = std::make_shared<OrderedDrawRenderer>();
            renderer->fail = true;
            lifetime = renderer;
            fixture.session->set_render_backend(
                {nullptr, renderer.get(), OrderedDrawRenderer::draw, renderer});
        }
        const auto commands = normal_draw_packet();
        fixture.write_commands(gpu_base, commands);
        const auto descriptor = entry(gpu_base, commands.size());
        CHECK(fixture.submit({&descriptor, 1}, 2, 0, 0).reply.disposition ==
              IpcDisposition::unsupported);
        fixture.session->set_render_backend({});
        CHECK(!lifetime.expired());
        CHECK(fixture.submit({&descriptor, 1}, 2, 0, 0).reply.disposition ==
              IpcDisposition::unsupported);
        CHECK(fixture.completed() == 0);
    }
    CHECK(lifetime.expired());
}

void test_draw_orders_reports_semaphores_and_fences() {
    enum class Failure { None, Draw, Writeback, Acquire };
    constexpr auto short_report = gpu_base + 0x810;
    constexpr auto long_report = gpu_base + 0x820;
    constexpr auto short_semaphore = gpu_base + 0x840;
    constexpr auto long_semaphore = gpu_base + 0x850;
    constexpr auto tail_report = gpu_base + 0x870;
    constexpr std::uint32_t report_payload = 0x55667788;
    constexpr std::uint32_t semaphore_payload = 0x24681357;

    for (const auto failure : {Failure::None, Failure::Draw,
                               Failure::Writeback, Failure::Acquire}) {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        OrderedDrawRenderer renderer;
        renderer.fail = failure == Failure::Draw;
        fixture.session->set_render_backend({nullptr, &renderer, OrderedDrawRenderer::draw, {}});
        auto commands = normal_draw_packet();
        commands.insert(commands.end(), {
            header(1, 4, 0, 2), 0, static_cast<std::uint32_t>(completed_draw_address),
            header(3, 0x1a, 0, 1),
                failure == Failure::Acquire ? completed_draw_value + 1 : completed_draw_value,
            header(1, 0x6c0, 0, 4), 0, static_cast<std::uint32_t>(short_report),
                report_payload, 1u << 28,
            header(3, 0x6c1, 0, 1), static_cast<std::uint32_t>(long_report),
            header(3, 0x6c3, 0, 1), 0,
            header(1, 4, 0, 2), 0, static_cast<std::uint32_t>(short_semaphore),
            header(3, 0x1b, 0, 1), semaphore_payload,
            header(3, 0x1a, 0, 1), semaphore_payload,
            header(1, 4, 0, 4), 0, static_cast<std::uint32_t>(long_semaphore),
                semaphore_payload, 2,
            header(3, 7, 0, 1), 1,
            header(3, 0xb2, 0, 1), expected_syncpoint,
            header(1, 0x1c, 0, 2), 1, expected_syncpoint << 8,
            header(3, 0x1d, 0, 1), (expected_syncpoint << 8) | 1,
            header(1, 0x1c, 0, 2), 2, expected_syncpoint << 8,
            header(4, 9, 0, 0), header(4, 0x14, 0, 0),
            header(4, 0x1e, 0, 0), header(4, 0x44, 0, 0),
            header(3, 0x6c1, 0, 1), static_cast<std::uint32_t>(tail_report),
            header(3, 0x6c3, 0, 1), 1u << 28,
        });
        fixture.write_commands(gpu_base, commands);
        fixture.state->record_backing_accesses = true;
        if (failure == Failure::Writeback) {
            fixture.state->fail_write_address = guest_backing_address(completed_draw_address);
        }
        const auto descriptor = entry(gpu_base, commands.size());
        // Reserve two explicit increments and the two automatic tail increments.
        const auto submitted = fixture.submit({&descriptor, 1}, 0x102, 0, 2);
        const std::string diagnostic = fixture.session->diagnostic()
            ? fixture.session->diagnostic() : "";
        CHECK(renderer.calls == 1);
        CHECK(fixture.reserved() == 4);
        CHECK(fixture.completed() == (failure == Failure::None ? 4u : 0u));
        const bool wrote_draw = failure == Failure::None || failure == Failure::Acquire;
        CHECK(backing_word(fixture, completed_draw_address) ==
              (wrote_draw ? completed_draw_value : 0u));

        if (failure == Failure::None) {
            CHECK(submitted.reply.disposition == IpcDisposition::handled);
            CHECK(reply_word(submitted.reply, 32) == nv_success);
            CHECK(load_word(submitted.arguments, 20) == 4);
            const std::vector<std::uint64_t> writes{
                guest_backing_address(completed_draw_address),
                guest_backing_address(short_report), guest_backing_address(long_report),
                guest_backing_address(short_semaphore), guest_backing_address(long_semaphore),
                guest_backing_address(tail_report)};
            CHECK(fixture.state->backing_writes == writes);
            CHECK(backing_word(fixture, short_report) == report_payload);
            CHECK(backing_word(fixture, long_report) == report_payload);
            CHECK(backing_word(fixture, long_report + 4) == 0);
            CHECK((backing_word(fixture, long_report + 8) |
                   backing_word(fixture, long_report + 12)) != 0);
            CHECK(backing_word(fixture, short_semaphore) == semaphore_payload);
            CHECK(backing_word(fixture, long_semaphore) == semaphore_payload);
            CHECK(backing_word(fixture, long_semaphore + 4) == 0);
            CHECK((backing_word(fixture, long_semaphore + 8) |
                   backing_word(fixture, long_semaphore + 12)) != 0);
            CHECK(backing_word(fixture, tail_report) == report_payload);
        } else {
            CHECK(submitted.reply.disposition == IpcDisposition::unsupported);
            CHECK(fixture.state->backing_writes.size() == (wrote_draw ? 1u : 0u));
            for (const auto address : {short_report, long_report, short_semaphore,
                                      long_semaphore, tail_report}) {
                CHECK(backing_word(fixture, address) == 0);
            }
            const auto reads = fixture.state->backing_reads;
            const auto writes = fixture.state->backing_writes;
            CHECK(!diagnostic.empty());
            renderer.fail = false;
            fixture.state->fail_write_address = 0;
            const auto repeated = fixture.submit({&descriptor, 1}, 0x102, 0, 2);
            CHECK(repeated.reply.disposition == IpcDisposition::unsupported);
            CHECK(fixture.session->diagnostic() == diagnostic);
            CHECK(renderer.calls == 1);
            CHECK(fixture.reserved() == 4 && fixture.completed() == 0);
            CHECK(fixture.state->backing_reads == reads);
            CHECK(fixture.state->backing_writes == writes);
        }
    }
}

void test_draw_writeback_precedes_next_entry_fetch() {
    enum class Failure { None, Draw, Fetch };
    constexpr auto next_address = gpu_base + 0x400;
    constexpr auto report_address = gpu_base + 0x900;
    constexpr std::uint32_t rewritten_payload = 0x12345678;

    for (const bool synchronized : {false, true}) {
        for (const auto failure : {Failure::None, Failure::Draw, Failure::Fetch}) {
            PreparedFixture fixture;
            fixture.allocate_gpfifo(16, 1);
            OrderedDrawRenderer renderer;
            renderer.fail = failure == Failure::Draw;
            renderer.rewrite_address = next_address + 12;
            renderer.rewrite_value = rewritten_payload;
            fixture.session->set_render_backend({nullptr, &renderer, OrderedDrawRenderer::draw, {}});
            const auto draw = normal_draw_packet();
            const std::array later{
                header(1, 0x6c0, 0, 4), 0u, static_cast<std::uint32_t>(report_address),
                    0xdeadbeefu, 1u << 28,
                header(3, 0x1d, 0, 1), (expected_syncpoint << 8) | 1,
            };
            fixture.write_commands(gpu_base, draw);
            fixture.write_commands(next_address, later);
            fixture.state->record_backing_accesses = true;
            if (failure == Failure::Fetch) {
                fixture.state->fail_read_address = guest_backing_address(next_address);
            }
            const std::array descriptors{
                entry(gpu_base, draw.size()),
                entry(next_address, later.size()) | (std::uint64_t{synchronized} << 63)};
            const auto submitted = fixture.submit(descriptors, 0x104, 0, 1);
            CHECK(renderer.calls == 1);
            CHECK(fixture.reserved() == 1);
            CHECK(fixture.completed() == (failure == Failure::None ? 1u : 0u));
            CHECK(backing_word(fixture, report_address) ==
                  (failure == Failure::None ? rewritten_payload : 0u));
            CHECK(backing_word(fixture, next_address + 12) ==
                  (failure == Failure::Draw ? 0xdeadbeefu : rewritten_payload));
            std::vector<std::uint64_t> reads{guest_backing_address(gpu_base)};
            if (failure != Failure::Draw) reads.push_back(guest_backing_address(next_address));
            CHECK(fixture.state->backing_reads == reads);
            CHECK(submitted.reply.disposition == (failure == Failure::None
                ? IpcDisposition::handled : IpcDisposition::unsupported));
            if (failure != Failure::None) {
                renderer.fail = false;
                fixture.state->fail_read_address = 0;
                CHECK(fixture.submit(descriptors, 0x104, 0, 1).reply.disposition ==
                      IpcDisposition::unsupported);
                CHECK(renderer.calls == 1);
                CHECK(fixture.reserved() == 1 && fixture.completed() == 0);
                CHECK(fixture.state->backing_reads == reads);
            }
        }
    }
}

void test_pending_wait_event_signals_after_gpu_increment() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base, nop);
    const std::uint64_t nop_entry = entry(gpu_base, 1);
    const auto reservation = fixture.submit({&nop_entry, 1}, 0x104, 0, 1);
    CHECK(reservation.reply.disposition == IpcDisposition::handled);
    CHECK(fixture.completed() == 0);
    CHECK(fixture.reserved() == 1);

    std::array<std::byte, 16> registered{};
    store_word(registered, 0, expected_syncpoint);
    store_word(registered, 4, 1);
    store_word(registered, 8, 1);
    store_word(registered, 12, 16);
    const IpcReply registered_reply = fixture.ioctl(
        fixture.control, syncpoint_wait_registered, registered);
    CHECK(registered_reply.disposition == IpcDisposition::unsupported);

    std::array<std::byte, 16> poll{};
    store_word(poll, 0, expected_syncpoint);
    store_word(poll, 4, 1);
    IpcReply poll_reply =
        fixture.ioctl(fixture.control, syncpoint_wait_allocate, poll);
    CHECK(poll_reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(poll_reply, 32) == nv_timeout);

    auto [invalid_reply, invalid_value] =
        fixture.read_syncpoint(syncpoint_read_min, 192);
    (void)invalid_value;
    CHECK(invalid_reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(invalid_reply, 32) == nv_invalid_value);

    std::array<std::byte, 16> wait{};
    store_word(wait, 0, expected_syncpoint);
    store_word(wait, 4, 1);
    store_word(wait, 8, 1);
    IpcReply wait_reply =
        fixture.ioctl(fixture.control, syncpoint_wait_allocate, wait);
    CHECK(wait_reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(wait_reply, 32) == nv_timeout);
    const std::uint32_t encoded = load_word(wait, 12);
    CHECK(encoded == (0x10000000u | (expected_syncpoint << 16)));

    const IpcReply event_reply = fixture.query_event(fixture.control, encoded);
    CHECK(event_reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(event_reply, 32) == nv_success);
    CHECK(event_reply.transferred_count == 1);
    CHECK(event_reply.transferred_as_copy);
    const auto event =
        std::dynamic_pointer_cast<Event>(event_reply.transferred[0]);
    CHECK(event != nullptr);
    CHECK(event != nullptr && !event->is_signaled());

    const std::vector<std::uint32_t> increment =
        syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(gpu_base + 0x100, increment);
    const std::uint64_t increment_entry =
        entry(gpu_base + 0x100, increment.size());
    const auto completed = fixture.submit({&increment_entry, 1}, 0, 0, 0);
    CHECK(completed.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(completed.reply, 32) == nv_success);
    CHECK(fixture.completed() == 1);
    CHECK(fixture.reserved() == 1);
    CHECK(event != nullptr && event->is_signaled());
}

// Every /dev/nvhost-ctrl descriptor has its own event slots: registering slot
// 0 on a second descriptor leaves the first descriptor's slot 0, its pending
// wait and its event intact, and the second descriptor does not name that wait.
void test_control_event_slots_per_descriptor() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const std::array<std::uint32_t, 1> nop{0};
    fixture.write_commands(gpu_base, nop);
    const std::uint64_t nop_entry = entry(gpu_base, 1);
    CHECK(fixture.submit({&nop_entry, 1}, 0x104, 0, 1).reply.disposition == IpcDisposition::handled);
    const auto second = fixture.open("/dev/nvhost-ctrl");
    std::array<std::byte, 4> slot{};
    CHECK(reply_word(fixture.ioctl(fixture.control, event_register, slot), 32) == nv_success);
    CHECK(reply_word(fixture.ioctl(second, event_register, slot), 32) == nv_success);

    std::array<std::byte, 16> wait{};
    store_word(wait, 0, expected_syncpoint);
    store_word(wait, 4, 1);
    store_word(wait, 8, 0xffffffffu);
    store_word(wait, 12, 0);
    CHECK(reply_word(fixture.ioctl(fixture.control, syncpoint_wait_registered, wait), 32) == nv_timeout);
    const std::uint32_t encoded = load_word(wait, 12);
    const IpcReply first_reply = fixture.query_event(fixture.control, encoded);
    CHECK(reply_word(first_reply, 32) == nv_success);
    const auto first = std::dynamic_pointer_cast<Event>(first_reply.transferred[0]);
    CHECK(first != nullptr && !first->is_signaled());

    // The second descriptor registers its slot 0 again and names no such wait.
    CHECK(reply_word(fixture.ioctl(second, event_register, slot), 32) == nv_success);
    CHECK(reply_word(fixture.query_event(second, encoded), 32) == nv_invalid_value);

    const std::vector<std::uint32_t> increment = syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(gpu_base + 0x100, increment);
    const std::uint64_t increment_entry = entry(gpu_base + 0x100, increment.size());
    CHECK(reply_word(fixture.submit({&increment_entry, 1}, 0, 0, 0).reply, 32) == nv_success);
    CHECK(fixture.completed() == 1);
    CHECK(first != nullptr && first->is_signaled());
}

struct RemapReader {
    std::uint64_t address = 0;
    std::uint32_t page = 0;
    std::array<std::byte, 3> expected{};
    unsigned calls = 0;
    struct MemoryChange {
        std::size_t removed = 0;
        std::size_t released = 0;
        std::uint64_t gpu_base = 0;
        std::uint64_t mapped_size = 0;
        std::uint64_t buffer_offset = 0;
        void* removed_owner = nullptr;
        void* released_owner = nullptr;
        std::uint32_t copyout_word = 0;
        bool backing_read = false;
        bool backing_write = false;
    };
    std::array<MemoryChange, 32> memory_changes{};
    std::size_t memory_change_count = 0;
    bool fail_memory_change = false;
    static armigo::graphics::CommandResult clear(
        void* raw, const armigo::graphics::RenderStop&,
        const armigo::graphics::GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<RemapReader*>(raw);
        ++self.calls;
        for (unsigned index = 0; index < 3; ++index) {
            std::array<std::byte, 1> value{};
            CHECK(memory.read(self.address + index * self.page, value));
            CHECK(value[0] == self.expected[index]);
        }
        const std::array value{std::byte{0x6a}};
        CHECK(memory.write(self.address + 2 * self.page + 1, value));
        return {};
    }
    static armigo::graphics::CommandResult memory_changed(
        void* raw, std::span<const armigo::graphics::Mapping> removed,
        std::span<const std::shared_ptr<void>> released,
        armigo::graphics::MemoryCallbacks callbacks) noexcept {
        auto& self = *static_cast<RemapReader*>(raw);
        CHECK(self.memory_change_count < self.memory_changes.size());
        if (self.memory_change_count < self.memory_changes.size()) {
            auto& record = self.memory_changes[self.memory_change_count];
            record.removed = removed.size();
            record.released = released.size();
            if (!removed.empty()) {
                record.gpu_base = removed.front().gpu_base;
                record.mapped_size = removed.front().mapped_size;
                record.buffer_offset = removed.front().buffer_offset;
                record.removed_owner = removed.front().owner.get();
                std::array<std::byte, 1> byte{};
                record.backing_read = callbacks.read && callbacks.read(
                    callbacks.context,
                    removed.front().guest_address + removed.front().buffer_offset,
                    byte);
                record.backing_write = record.backing_read && callbacks.write &&
                    callbacks.write(
                        callbacks.context,
                        removed.front().guest_address + removed.front().buffer_offset,
                        byte);
            }
            if (!released.empty()) record.released_owner = released.front().get();
            std::array<std::byte, 4> word{};
            if (callbacks.read && callbacks.read(
                    callbacks.context, argument_address, word)) {
                record.copyout_word = load_word(word, 0);
            }
        }
        ++self.memory_change_count;
        if (self.fail_memory_change) {
            return {armigo::graphics::CommandStatus::Unsupported,
                    "Synthetic memory-change failure"};
        }
        return {};
    }
};

struct MemoryChangeRecorder {
    struct Call {
        std::size_t removed = 0;
        std::size_t released = 0;
        std::uint64_t gpu_base = 0;
        std::uint64_t mapped_size = 0;
        void* removed_owner = nullptr;
        void* released_owner = nullptr;
        bool backing_read = false;
        bool backing_write = false;
        bool backing_was_live = false;
    };
    Fixture::MemoryState* memory = nullptr;
    std::uint64_t probe_address = backing_address;
    std::array<Call, 16> calls{};
    std::size_t count = 0;
    bool fail = false;

    static armigo::graphics::CommandResult changed(
        void* raw, std::span<const armigo::graphics::Mapping> removed,
        std::span<const std::shared_ptr<void>> released,
        armigo::graphics::MemoryCallbacks callbacks) noexcept {
        auto& self = *static_cast<MemoryChangeRecorder*>(raw);
        CHECK(self.count < self.calls.size());
        if (self.count < self.calls.size()) {
            auto& call = self.calls[self.count];
            call.removed = removed.size();
            call.released = released.size();
            call.backing_was_live = self.memory && self.memory->unmapped.empty();
            if (!removed.empty()) {
                call.gpu_base = removed.front().gpu_base;
                call.mapped_size = removed.front().mapped_size;
                call.removed_owner = removed.front().owner.get();
            }
            if (!released.empty()) call.released_owner = released.front().get();
            std::array<std::byte, 1> value{};
            call.backing_read = callbacks.read && callbacks.read(
                callbacks.context, self.probe_address, value);
            call.backing_write = call.backing_read && callbacks.write &&
                callbacks.write(callbacks.context, self.probe_address, value);
        }
        ++self.count;
        if (self.fail) {
            return {armigo::graphics::CommandStatus::Unsupported,
                    "Synthetic memory-change failure"};
        }
        return {};
    }
};

void install_memory_recorder(PreparedFixture& fixture,
                             MemoryChangeRecorder& recorder) {
    recorder.memory = fixture.state.get();
    armigo::graphics::RenderCallbacks callbacks{};
    callbacks.context = &recorder;
    callbacks.memory_changed = MemoryChangeRecorder::changed;
    fixture.session->set_render_backend(std::move(callbacks));
}

void test_nvmap_free_missing_handles_are_noops() {
    PreparedFixture fixture;
    fixture.unmap(gpu_base);
    MemoryChangeRecorder recorder;
    install_memory_recorder(fixture, recorder);
    fixture.free_memory(fixture.memory_handle);
    CHECK(recorder.count == 1);
    CHECK(recorder.calls[0].released == 1);
    CHECK(fixture.state->unmapped.size() == 1);

    const auto verify_noop = [&](std::uint32_t handle) {
        std::array<std::byte, 0x18> arguments{};
        arguments.fill(std::byte{0xa5});
        store_word(arguments, 0, handle);
        const auto original = arguments;
        fixture.copy_to(argument_address, arguments);
        GuestBuffers buffers(Fixture::read_memory, Fixture::write_memory,
                             fixture.state.get());
        buffers.add({argument_address, arguments.size(), true, true});
        fixture.state->fail_write_address = argument_address;
        const auto reply = fixture.session->request_with_buffers(
            request(ioctl_command, fixture.nvmap, nvmap_free), 1, buffers);
        fixture.state->fail_write_address = 0;
        CHECK(reply_word(reply, 32) == nv_success);
        fixture.copy_from(argument_address, arguments);
        CHECK(arguments == original);
        CHECK(recorder.count == 1);
        CHECK(fixture.state->unmapped.size() == 1);
    };

    verify_noop(fixture.memory_handle);
    verify_noop(0xfffffffc);
    verify_noop(0);
}

void test_nvmap_free_failure_atomicity() {
    for (const bool mapped : {false, true}) {
        PreparedFixture fixture;
        fixture.export_memory(fixture.memory_handle);
        if (!mapped) fixture.unmap(gpu_base);
        MemoryChangeRecorder recorder;
        install_memory_recorder(fixture, recorder);
        std::array<std::byte, 0x18> arguments{};
        arguments.fill(std::byte{0xa5});
        store_word(arguments, 0, fixture.memory_handle);
        const auto original = arguments;
        fixture.copy_to(argument_address, arguments);
        GuestBuffers buffers(Fixture::read_memory, Fixture::write_memory, fixture.state.get());
        buffers.add({argument_address, arguments.size(), true, true});
        fixture.state->fail_write_address = argument_address;
        const auto rejected = fixture.session->request_with_buffers(
            request(ioctl_command, fixture.nvmap, nvmap_free), 1, buffers);
        fixture.state->fail_write_address = 0;
        CHECK(reply_word(rejected, 32) == 9);
        fixture.copy_from(argument_address, arguments);
        CHECK(arguments == original);
        CHECK(fixture.state->unmapped.empty());
        CHECK(recorder.count == (mapped ? 0u : 1u));
        if (!mapped) {
            CHECK(recorder.calls[0].released == 1);
            CHECK(recorder.calls[0].backing_was_live);
        }
        CHECK(fixture.session->retain_exported_memory(fixture.memory_handle));
        CHECK(fixture.session->device_mapping_kind(fixture.address_space, gpu_base).has_value() == mapped);

        fixture.state->fail_read_address = argument_address;
        const auto unreadable = fixture.session->request_with_buffers(
            request(ioctl_command, fixture.nvmap, nvmap_free), 1, buffers);
        fixture.state->fail_read_address = 0;
        CHECK(reply_word(unreadable, 32) == 9);
        GuestBuffers short_buffers(Fixture::read_memory, Fixture::write_memory, fixture.state.get());
        short_buffers.add({argument_address, arguments.size() - 1, true, true});
        CHECK(reply_word(fixture.session->request_with_buffers(
            request(ioctl_command, fixture.nvmap, nvmap_free), 1, short_buffers), 32) == nv_invalid_size);
        fixture.copy_from(argument_address, arguments);
        CHECK(arguments == original && fixture.state->unmapped.empty());
        const auto freed = fixture.free_memory(fixture.memory_handle);
        CHECK(load_qword(freed, 8) == (mapped ? 0 : backing_address));
        CHECK(load_word(freed, 16) == backing_size && load_word(freed, 20) == 0);
        if (mapped) fixture.unmap(gpu_base);
        CHECK(fixture.state->unmapped.size() == 1);
        CHECK(recorder.count == (mapped ? 1u : 2u));
    }

    PreparedFixture fixture;
    fixture.unmap(gpu_base);
    MemoryChangeRecorder recorder;
    install_memory_recorder(fixture, recorder);
    recorder.fail = true;
    std::array<std::byte, 0x18> arguments{};
    arguments.fill(std::byte{0xa5});
    store_word(arguments, 0, fixture.memory_handle);
    const auto original = arguments;
    CHECK(fixture.ioctl(fixture.nvmap, nvmap_free, arguments).disposition == IpcDisposition::unsupported);
    CHECK(arguments == original && fixture.state->unmapped.empty());
    recorder.fail = false;
    const auto freed = fixture.free_memory(fixture.memory_handle);
    CHECK(load_qword(freed, 8) == backing_address);
    CHECK(fixture.state->unmapped.size() == 1);
}

void test_nvmap_free_ignores_renderer_reference_counts() {
    struct RendererPins {
        std::array<std::shared_ptr<void>, 4> owners;
        static armigo::graphics::CommandResult changed(
            void* context, std::span<const armigo::graphics::Mapping> removed,
            std::span<const std::shared_ptr<void>>, armigo::graphics::MemoryCallbacks) noexcept {
            if (!removed.empty())
                static_cast<RendererPins*>(context)->owners.fill(removed.front().owner);
            return {};
        }
    } renderer;
    PreparedFixture fixture;
    armigo::graphics::RenderCallbacks callbacks{};
    callbacks.context = &renderer;
    callbacks.memory_changed = RendererPins::changed;
    fixture.session->set_render_backend(std::move(callbacks));
    fixture.unmap(gpu_base);
    CHECK(renderer.owners.front());
    const auto freed = fixture.free_memory(fixture.memory_handle);
    CHECK(load_qword(freed, 8) == backing_address);
    CHECK(load_word(freed, 16) == backing_size && load_word(freed, 20) == 0);
    CHECK(fixture.state->unmapped.empty());
    renderer.owners.fill({});
    CHECK(fixture.state->unmapped.size() == 1);
}

std::array<std::byte, 0x28> kind_remap_arguments(
    std::uint64_t address, std::uint32_t kind,
    std::uint64_t offset = 0, std::uint64_t size = gpu_mapping_size) {
    std::array<std::byte, 0x28> arguments{};
    store_word(arguments, 0, 0x101);
    store_word(arguments, 4, kind);
    store_qword(arguments, 16, offset);
    store_qword(arguments, 24, size);
    store_qword(arguments, 32, address);
    return arguments;
}

void test_whole_mapping_kind_remap() {
    PreparedFixture fixture(0, 1);
    CHECK(fixture.session->device_mapping_kind(fixture.address_space, gpu_base) == 1);
    fixture.dispatch_requests = true;
    fixture.allocate_gpfifo(16, 1);
    MemoryChangeRecorder recorder;
    install_memory_recorder(fixture, recorder);
    const auto commands = syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(gpu_base, commands);
    fixture.write_commands(gpu_base + gpu_mapping_size - commands.size() * 4, commands);

    // The mapping, rather than a live handle, owns the backing being re-kinded.
    fixture.free_memory(fixture.memory_handle);
    CHECK(fixture.state->unmapped.empty());
    auto arguments = kind_remap_arguments(gpu_base, 0x12340007);
    const auto original = arguments;
    const auto reply = fixture.ioctl(fixture.address_space, address_space_map_buffer, arguments);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(reply, 32) == nv_success);
    CHECK(arguments == original);
    CHECK(fixture.session->device_mapping_kind(fixture.address_space, gpu_base) == 0x12340007);
    DriverSession clone(*fixture.session);
    CHECK(clone.device_mapping_kind(fixture.address_space, gpu_base) == 0x12340007);
    CHECK(recorder.count == 0);

    // Both ends still resolve the original allocation and its nonzero source offset.
    for (const auto address : {gpu_base, gpu_base + gpu_mapping_size - commands.size() * 4}) {
        const auto descriptor = entry(address, commands.size());
        const auto submitted = fixture.submit({&descriptor, 1}, 0, 0, 0);
        CHECK(submitted.reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(submitted.reply, 32) == nv_success);
    }
    CHECK(fixture.completed() == 2);
    CHECK(fixture.state->unmapped.empty());

    fixture.unmap(gpu_base);
    CHECK(!fixture.session->device_mapping_kind(fixture.address_space, gpu_base));
    CHECK(recorder.count == 1);
    CHECK(recorder.calls[0].removed == 1);
    CHECK(recorder.calls[0].released == 1);
    CHECK(recorder.calls[0].removed_owner == recorder.calls[0].released_owner);
    CHECK(recorder.calls[0].backing_was_live);
    CHECK(fixture.state->unmapped.size() == 1);
    if (!fixture.state->unmapped.empty())
        CHECK(fixture.state->unmapped.front() == std::make_pair(backing_address, backing_size));
    const auto descriptor = entry(gpu_base, commands.size());
    CHECK(fixture.submit({&descriptor, 1}, 0, 0, 0).reply.disposition == IpcDisposition::unsupported);
    fixture.unmap(gpu_base + gpu_mapping_size - 1);
    CHECK(recorder.count == 1);
    CHECK(fixture.state->unmapped.size() == 1);
}

void test_kind_remap_rejection_is_atomic() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const auto commands = syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(gpu_base, commands);
    auto arguments = kind_remap_arguments(gpu_base, 1);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, address_space_map_buffer, arguments), 32) == nv_success);
    const auto unchanged = [&] {
        CHECK(fixture.session->device_mapping_kind(fixture.address_space, gpu_base) == 1);
        CHECK(fixture.state->unmapped.empty());
        const auto descriptor = entry(gpu_base, commands.size());
        const auto submitted = fixture.submit({&descriptor, 1}, 0, 0, 0);
        CHECK(submitted.reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(submitted.reply, 32) == nv_success);
    };
    const auto reject = [&](std::uint64_t address, std::uint64_t offset, std::uint64_t size) {
        auto invalid = kind_remap_arguments(address, 7, offset, size);
        const auto original = invalid;
        CHECK(reply_word(fixture.ioctl(fixture.address_space, address_space_map_buffer, invalid), 32) == 0x3000f);
        CHECK(invalid == original);
        unchanged();
    };
    reject(gpu_base + gpu_mapping_size, 0, gpu_mapping_size);
    reject(gpu_base + 0x1000, 0, 0x1000);
    reject(gpu_base, 1, 0x1000);
    reject(gpu_base, 0, gpu_mapping_size - 1);
    reject(gpu_base, gpu_mapping_size, 0x1000);
    reject(gpu_base, gpu_mapping_size, 0);
    reject(gpu_base, 0x1000, gpu_mapping_size);
    reject(gpu_base, 0x1000, std::numeric_limits<std::uint64_t>::max() - 0xfff);
    reject(gpu_base, std::numeric_limits<std::uint64_t>::max() - 0xfff, 0x1000);

    // Valid partial remaps need interval bookkeeping; refusal must not change kind.
    auto partial = kind_remap_arguments(gpu_base, 7, 0x1000, 0x1000);
    CHECK(fixture.ioctl(fixture.address_space, address_space_map_buffer, partial).disposition == IpcDisposition::unsupported);
    CHECK(std::string_view(fixture.session->diagnostic()).find("whole") != std::string_view::npos);
    unchanged();
    auto empty = kind_remap_arguments(gpu_base, 7, 0x1000, 0);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, address_space_map_buffer, empty), 32) == nv_success);
    unchanged();

    // A failed output write cannot publish the new kind.
    arguments = kind_remap_arguments(gpu_base, 7);
    fixture.copy_to(argument_address, arguments);
    GuestBuffers buffers(Fixture::read_memory, Fixture::write_memory, fixture.state.get());
    buffers.add({argument_address, static_cast<std::uint32_t>(arguments.size()), true, true});
    fixture.state->fail_write = true;
    CHECK(reply_word(fixture.session->request_with_buffers(
        request(ioctl_command, fixture.address_space, address_space_map_buffer), 1, buffers), 32) == 9);
    fixture.state->fail_write = false;
    unchanged();
}

void test_kind_remap_preserves_va_reservations() {
    PreparedFixture fixture;
    std::uint64_t automatic = 0;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, automatic, false), 32) == nv_success);
    auto arguments = kind_remap_arguments(automatic, 7);
    // Handle, page size and the fixed flag do not select backing on this path.
    store_word(arguments, 0, 0x100);
    store_word(arguments, 8, 0xfffffffc);
    store_word(arguments, 12, 3);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, address_space_map_buffer, arguments), 32) == nv_success);
    std::uint64_t next = 0;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, next, false), 32) == nv_success);
    CHECK(next == automatic + gpu_mapping_size);
    fixture.unmap(next);
    const auto original = automatic;
    fixture.unmap(automatic);
    automatic = 0;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, automatic, false), 32) == nv_success);
    CHECK(automatic == original);
    fixture.unmap(automatic);

    std::array<std::byte, 0x18> reservation{};
    store_word(reservation, 0, 2);
    store_word(reservation, 4, 0x1000);
    store_word(reservation, 8, 1);
    store_qword(reservation, 16, original);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0184102, reservation), 32) == nv_success);
    auto fixed = original;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, fixed), 32) == nv_success);
    arguments = kind_remap_arguments(fixed, 7);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, address_space_map_buffer, arguments), 32) == nv_success);
    fixture.unmap(fixed);
    automatic = 0;
    CHECK(reply_word(fixture.map_memory_handle(fixture.memory_handle, automatic, false), 32) == nv_success);
    CHECK(automatic == original + gpu_mapping_size);
}

void test_memory_change_notifications() {
    {
        PreparedFixture fixture;
        fixture.dispatch_requests = true;
        MemoryChangeRecorder recorder;
        install_memory_recorder(fixture, recorder);
        std::uint64_t alias = gpu_base + 0x10000;
        CHECK(reply_word(fixture.map_memory_handle(
            fixture.memory_handle, alias), 32) == nv_success);

        fixture.free_memory(fixture.memory_handle);
        CHECK(recorder.count == 0);
        CHECK(fixture.state->unmapped.empty());

        std::array<std::byte, 8> unmap{};
        store_qword(unmap, 0, gpu_base);
        recorder.fail = true;
        const auto rejected = fixture.ioctl(
            fixture.address_space, address_space_unmap_buffer, unmap);
        CHECK(rejected.disposition == IpcDisposition::unsupported);
        CHECK(fixture.session->diagnostic() != nullptr);
        if (fixture.session->diagnostic()) {
            CHECK(std::string_view(fixture.session->diagnostic()) ==
                  "Synthetic memory-change failure");
        }
        CHECK(recorder.count == 1);
        CHECK(recorder.calls[0].removed == 1);
        CHECK(recorder.calls[0].released == 0);
        CHECK(recorder.calls[0].gpu_base == gpu_base);
        CHECK(recorder.calls[0].mapped_size == gpu_mapping_size);
        CHECK(recorder.calls[0].backing_read);
        CHECK(recorder.calls[0].backing_write);
        CHECK(recorder.calls[0].backing_was_live);
        CHECK(fixture.state->unmapped.empty());

        recorder.fail = false;
        CHECK(reply_word(fixture.ioctl(
            fixture.address_space, address_space_unmap_buffer, unmap), 32) == nv_success);
        CHECK(recorder.count == 2);
        CHECK(recorder.calls[1].released == 0);
        CHECK(fixture.state->unmapped.empty());
        fixture.unmap(alias);
        CHECK(recorder.count == 3);
        CHECK(recorder.calls[2].removed == 1);
        CHECK(recorder.calls[2].released == 1);
        CHECK(recorder.calls[2].removed_owner == recorder.calls[2].released_owner);
        CHECK(recorder.calls[2].backing_was_live);
        CHECK(fixture.state->unmapped.size() == 1);
    }

    {
        PreparedFixture fixture;
        fixture.dispatch_requests = true;
        MemoryChangeRecorder recorder;
        install_memory_recorder(fixture, recorder);
        fixture.unmap(gpu_base);
        CHECK(recorder.count == 1);
        CHECK(recorder.calls[0].removed == 1);
        CHECK(recorder.calls[0].released == 0);
        std::array<std::byte, 0x18> free_arguments{};
        store_word(free_arguments, 0, fixture.memory_handle);
        recorder.fail = true;
        const auto rejected = fixture.ioctl(
            fixture.nvmap, nvmap_free, free_arguments);
        CHECK(rejected.disposition == IpcDisposition::unsupported);
        CHECK(recorder.count == 2);
        CHECK(recorder.calls[1].removed == 0);
        CHECK(recorder.calls[1].released == 1);
        CHECK(recorder.calls[1].backing_read);
        CHECK(recorder.calls[1].backing_write);
        CHECK(recorder.calls[1].backing_was_live);
        CHECK(fixture.state->unmapped.empty());
        recorder.fail = false;
        fixture.free_memory(fixture.memory_handle);
        CHECK(recorder.count == 3);
        CHECK(recorder.calls[2].removed == 0);
        CHECK(recorder.calls[2].released == 1);
        CHECK(recorder.calls[2].backing_read);
        CHECK(recorder.calls[2].backing_write);
        CHECK(recorder.calls[2].backing_was_live);
        CHECK(fixture.state->unmapped.size() == 1);
    }

    {
        PreparedFixture fixture;
        fixture.dispatch_requests = true;
        MemoryChangeRecorder recorder;
        install_memory_recorder(fixture, recorder);
        std::uint64_t alias = gpu_base + 0x10000;
        CHECK(reply_word(fixture.map_memory_handle(
            fixture.memory_handle, alias), 32) == nv_success);

        const auto second_address_space = fixture.open("/dev/nvhost-as-gpu");
        std::array<std::byte, 0x28> initialize{};
        CHECK(reply_word(fixture.ioctl(
            second_address_space, address_space_initialize, initialize), 32) == nv_success);
        std::array<std::byte, 0x28> mapping{};
        store_word(mapping, 0, 1);
        store_word(mapping, 8, fixture.memory_handle);
        store_word(mapping, 12, 0x1000);
        store_qword(mapping, 16, buffer_offset);
        store_qword(mapping, 24, gpu_mapping_size);
        store_qword(mapping, 32, gpu_base + 0x20000);
        CHECK(reply_word(fixture.ioctl(
            second_address_space, address_space_map_buffer, mapping), 32) == nv_success);
        fixture.free_memory(fixture.memory_handle);
        CHECK(recorder.count == 0);

        recorder.fail = true;
        const auto rejected = fixture.dispatch_request(request(2, fixture.address_space));
        CHECK(rejected.disposition == IpcDisposition::unsupported);
        CHECK(recorder.count == 1);
        CHECK(recorder.calls[0].removed == 2);
        CHECK(recorder.calls[0].backing_read);
        CHECK(recorder.calls[0].backing_write);
        CHECK(recorder.calls[0].released == 0);
        CHECK(recorder.calls[0].backing_was_live);
        CHECK(fixture.session->device_path(fixture.address_space) ==
              "/dev/nvhost-as-gpu");
        CHECK(fixture.state->unmapped.empty());

        recorder.fail = false;
        fixture.close_address_space();
        CHECK(recorder.count == 2);
        CHECK(recorder.calls[1].removed == 2);
        CHECK(recorder.calls[1].backing_read);
        CHECK(recorder.calls[1].backing_write);
        CHECK(recorder.calls[1].released == 0);
        CHECK(recorder.calls[1].backing_was_live);
        CHECK(fixture.state->unmapped.empty());

        std::array<std::byte, 8> unmap{};
        store_qword(unmap, 0, gpu_base + 0x20000);
        CHECK(reply_word(fixture.ioctl(
            second_address_space, address_space_unmap_buffer, unmap), 32) == nv_success);
        CHECK(recorder.count == 3);
        CHECK(recorder.calls[2].removed == 1);
        CHECK(recorder.calls[2].released == 1);
        CHECK(fixture.state->unmapped.size() == 1);
    }
}

void test_sparse_remap() {
    for (const auto page : {0x10000u, 0x20000u}) {
        PreparedFixture fixture(page);
        fixture.dispatch_requests = true;
        fixture.allocate_gpfifo(16, 1);
        std::array<std::byte, 8> create{};
        store_word(create, 0, 3 * page);
        CHECK(reply_word(fixture.ioctl(fixture.nvmap, nvmap_create, create), 32) == nv_success);
        const auto pool = load_word(create, 4);
        std::array<std::byte, 0x20> allocate{};
        store_word(allocate, 0, pool);
        store_word(allocate, 12, page);
        store_qword(allocate, 24, backing_address);
        CHECK(reply_word(fixture.ioctl(fixture.nvmap, nvmap_allocate, allocate), 32) == nv_success);
        const std::array values{std::byte{0xa1}, std::byte{0xb2}, std::byte{0xc3}};
        for (unsigned index = 0; index < 3; ++index)
            fixture.copy_to(backing_address + index * page, {&values[index], 1});

        std::array<std::byte, 0x18> reservation{};
        store_word(reservation, 0, 3);
        store_word(reservation, 4, page);
        store_word(reservation, 8, 2);
        store_qword(reservation, 16, page);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0184102, reservation), 32) == nv_success);
        const std::uint64_t target = std::uint64_t{load_word(reservation, 16)} |
            (std::uint64_t{load_word(reservation, 20)} << 32);
        const auto target_page = static_cast<std::uint32_t>(target / page);
        const auto encode = [&](std::span<std::byte> bytes, std::size_t offset,
                                std::uint32_t source, std::uint32_t destination,
                                std::uint32_t pages, std::uint32_t kind = 0) {
            store_word(bytes, offset, 4u | (kind << 16));
            store_word(bytes, offset + 4, pool);
            store_word(bytes, offset + 8, source);
            store_word(bytes, offset + 12, destination);
            store_word(bytes, offset + 16, pages);
        };
        std::array<std::byte, 20> remap{};
        encode(remap, 0, 0, target_page, 3);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == nv_success);
        RemapReader reader{target, page, values};
        armigo::graphics::RenderCallbacks rendering{};
        rendering.clear = RemapReader::clear;
        rendering.context = &reader;
        rendering.memory_changed = RemapReader::memory_changed;
        fixture.session->set_render_backend(std::move(rendering));
        const std::array commands{header(1, 0, 0, 1), 0xb197u,
                                  header(1, 0x674, 0, 1), 0x3cu};
        fixture.write_commands(gpu_base, commands);
        const auto descriptor = entry(gpu_base, commands.size());
        const auto verify = [&] {
            const auto prior = reader.calls;
            CHECK(fixture.submit({&descriptor, 1}, 0, 0, 0).reply.disposition == IpcDisposition::handled);
            CHECK(reader.calls == prior + 1);
            std::array<std::byte, 1> written{};
            fixture.copy_from(backing_address + 2 * page + 1, written);
            CHECK(written[0] == std::byte{0x6a});
            CHECK(fixture.completed() == 0);
        };
        verify();

        // Replacing one interior page retains both outer slices and source offsets.
        encode(remap, 0, 2, target_page + 1, 1);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == nv_success);
        CHECK(reader.memory_change_count == 1);
        CHECK(reader.memory_changes[0].removed == 1);
        CHECK(reader.memory_changes[0].released == 0);
        CHECK(reader.memory_changes[0].gpu_base == target);
        CHECK(reader.memory_changes[0].mapped_size == 3 * page);
        CHECK(reader.memory_changes[0].buffer_offset == 0);
        CHECK(reader.memory_changes[0].copyout_word == 4);
        CHECK(reader.memory_changes[0].backing_read);
        CHECK(reader.memory_changes[0].backing_write);
        reader.expected[1] = values[2];
        verify();

        // No mappings from an invalid batch become visible.
        std::array<std::byte, 40> batch{};
        encode(batch, 0, 0, target_page + 1, 1, 7);
        encode(batch, 20, 1, target_page + 1, 0);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0284114, batch), 32) == 0x3000f);
        CHECK(load_word(batch, 0) == (4u | (1u << 16)));
        encode(batch, 0, 0, target_page + 1, 1);
        verify();
        encode(batch, 20, 1, target_page + 1, 1);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0284114, batch), 32) == nv_success);
        reader.expected[1] = values[1];
        verify();

        // Compression-tag allocation is absent, so the supported kind falls back.
        encode(remap, 0, 1, target_page + 1, 1, 7);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == nv_success);
        CHECK(load_word(remap, 0) == (4u | (1u << 16)));
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == nv_success);
        verify();
        encode(remap, 0, 3, target_page, 1);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == 0x3000f);
        encode(remap, 0, 0, target_page + 3, 1);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == 0x3000f);
        encode(remap, 0, 0, target_page, 1);
        store_word(remap, 4, 0xfffffffc);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == 0x3000f);
        CHECK(fixture.ioctl(fixture.address_space, 0xc0284114, remap).disposition == IpcDisposition::unsupported);
        encode(remap, 0, 0, target_page, 1);
        store_word(remap, 4, 0);
        CHECK(fixture.ioctl(fixture.address_space, 0xc0144114, remap).disposition == IpcDisposition::unsupported);
        encode(remap, 0, 0, target_page, 1);
        store_word(remap, 0, 0x1004);
        store_word(remap, 12, static_cast<std::uint32_t>(target / 0x1000));
        CHECK(fixture.ioctl(fixture.address_space, 0xc0144114, remap).disposition == IpcDisposition::unsupported);
        verify();

        fixture.free_memory(pool);
        CHECK(fixture.state->unmapped.empty());
        verify();
        // Sparse remap records belong to their reservation, not UnmapBuffer.
        fixture.unmap(target);
        verify();

        const auto replacement = fixture.create_memory(page);
        const auto replacement_address = memory_base + 0x70000;
        fixture.allocate_memory(replacement, replacement_address, page);
        const std::array replacement_value{std::byte{0xd4}};
        fixture.copy_to(replacement_address, replacement_value);
        const auto replace = [&](std::span<std::byte> bytes, std::size_t offset,
                                 std::uint32_t destination, std::uint32_t pages) {
            encode(bytes, offset, 0, destination, pages);
            store_word(bytes, offset + 4, replacement);
        };
        replace(batch, 0, target_page + 1, 1);
        replace(batch, 20, target_page + 2, 0);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0284114, batch), 32) == 0x3000f);
        CHECK(fixture.state->unmapped.empty());
        verify();

        replace(remap, 0, target_page + 1, 1);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == nv_success);
        reader.expected[1] = replacement_value[0];
        CHECK(fixture.state->unmapped.empty());
        verify();
        replace(remap, 0, target_page, 1);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == nv_success);
        reader.expected[0] = replacement_value[0];
        CHECK(fixture.state->unmapped.empty());
        verify();
        replace(remap, 0, target_page + 2, 1);
        store_word(remap, 0, 4u | (7u << 16));
        // Failed final copyout keeps the last original slice and its lease alive.
        fixture.copy_to(argument_address, remap);
        GuestBuffers buffers(Fixture::read_memory, Fixture::write_memory, fixture.state.get());
        buffers.add({argument_address, static_cast<std::uint32_t>(remap.size()), true, true});
        fixture.state->fail_write = true;
        const auto failed_copyout = fixture.session->request_with_buffers(
            request(ioctl_command, fixture.address_space, 0xc0144114), 1, buffers);
        fixture.state->fail_write = false;
        CHECK(reply_word(failed_copyout, 32) == 9);
        const auto before_rejection = reader.memory_change_count;
        CHECK(fixture.state->unmapped.empty());
        verify();
        reader.fail_memory_change = true;
        const auto rejected = fixture.ioctl(
            fixture.address_space, 0xc0144114, remap);
        reader.fail_memory_change = false;
        CHECK(rejected.disposition == IpcDisposition::unsupported);
        CHECK(reader.memory_change_count == before_rejection + 1);
        const auto& rejected_change =
            reader.memory_changes[before_rejection];
        CHECK(rejected_change.removed == 1);
        CHECK(rejected_change.released == 1);
        CHECK(rejected_change.gpu_base == target + 2 * page);
        CHECK(rejected_change.mapped_size == page);
        CHECK(rejected_change.buffer_offset == 2 * page);
        CHECK(rejected_change.removed_owner == rejected_change.released_owner);
        CHECK(rejected_change.copyout_word == (4u | (1u << 16)));
        CHECK(rejected_change.backing_read);
        CHECK(rejected_change.backing_write);
        CHECK(fixture.state->unmapped.empty());
        verify();
        CHECK(reply_word(fixture.ioctl(fixture.address_space, 0xc0144114, remap), 32) == nv_success);
        CHECK(reader.memory_change_count == before_rejection + 2);
        const auto& accepted_change =
            reader.memory_changes[before_rejection + 1];
        CHECK(accepted_change.removed == 1);
        CHECK(accepted_change.released == 1);
        CHECK(accepted_change.copyout_word == (4u | (1u << 16)));
        reader.expected[2] = replacement_value[0];
        CHECK(fixture.state->unmapped.size() == 1);
        if (!fixture.state->unmapped.empty())
            CHECK(fixture.state->unmapped.front() == std::make_pair(backing_address, std::uint64_t{3 * page}));
        const auto calls = reader.calls;
        CHECK(fixture.submit({&descriptor, 1}, 0, 0, 0).reply.disposition == IpcDisposition::handled);
        CHECK(reader.calls == calls + 1);
        fixture.free_memory(replacement);
        CHECK(fixture.state->unmapped.size() == 1);
        fixture.close_address_space();
        CHECK(fixture.state->unmapped.size() == 2);
    }
}

// A controlled backend proves that submission and fence reservation do not wait
// for command processing, without using timing as the execution-order oracle.
struct GatedRenderer {
    std::promise<void> entered;
    std::promise<void> release;
    std::shared_future<void> released = release.get_future().share();
    unsigned calls = 0;
    unsigned read_coherence_calls = 0;
    bool fail = false;
    static armigo::graphics::CommandResult clear(void* raw,
        const armigo::graphics::RenderStop&, const armigo::graphics::GpuAddressSpace&) noexcept {
        auto& self = *static_cast<GatedRenderer*>(raw);
        if (++self.calls == 1) {
            self.entered.set_value();
            self.released.wait();
        }
        return self.fail
            ? armigo::graphics::CommandResult{armigo::graphics::CommandStatus::Stopped,
                                               "Synthetic gated rendering failure"}
            : armigo::graphics::CommandResult{};
    }
    static armigo::graphics::CommandResult read_coherence(
        void* raw, const armigo::graphics::GpuAddressSpace&,
        std::uint64_t, std::uint64_t) noexcept {
        ++static_cast<GatedRenderer*>(raw)->read_coherence_calls;
        return {};
    }
    static armigo::graphics::CommandResult ordered_write(
        void*, const armigo::graphics::GpuAddressSpace&, std::uint64_t,
        std::span<const std::byte>) noexcept {
        return {};
    }
};

void enable_worker(PreparedFixture& fixture) {
    fixture.session->enable_async_gpu(
        {Fixture::read_memory, Fixture::write_memory, fixture.state.get()}, fixture.state);
}

// Match clients that serialize every callback through one external mutex. A
// coherence callback waits for another thread to publish earlier effects while
// still holding that mutex, as a threaded renderer's synchronous barrier does.
struct SerializedThreadedRenderer {
    std::mutex callback_mutex;
    AsyncEffectRenderer backend;
    std::optional<armigo::graphics::GpuAddressSpace> memory;
    unsigned renders = 0;

    armigo::graphics::CommandResult drain() {
        if (!memory) return {};
        return std::async(std::launch::async, [&] {
            return AsyncEffectRenderer::coherence(&backend, *memory, 0, 0);
        }).get();
    }
    static armigo::graphics::CommandResult render(void* raw,
        const armigo::graphics::RenderStop&, const armigo::graphics::GpuAddressSpace&) noexcept {
        auto& self = *static_cast<SerializedThreadedRenderer*>(raw);
        const std::lock_guard lock(self.callback_mutex);
        ++self.renders;
        return {};
    }
    static armigo::graphics::CommandResult effect(void* raw,
        const armigo::graphics::GpuAddressSpace& memory,
        armigo::graphics::GuestEffect effect) noexcept {
        auto& self = *static_cast<SerializedThreadedRenderer*>(raw);
        const std::lock_guard lock(self.callback_mutex);
        self.memory = memory;
        return AsyncEffectRenderer::ordered_effect(&self.backend, memory, std::move(effect));
    }
    static armigo::graphics::CommandResult coherence(void* raw,
        const armigo::graphics::GpuAddressSpace&, std::uint64_t, std::uint64_t) noexcept {
        auto& self = *static_cast<SerializedThreadedRenderer*>(raw);
        const std::lock_guard lock(self.callback_mutex);
        return self.drain();
    }
    static armigo::graphics::CommandResult memory_changed(void* raw,
        std::span<const armigo::graphics::Mapping>, std::span<const std::shared_ptr<void>>,
        armigo::graphics::MemoryCallbacks) noexcept {
        auto& self = *static_cast<SerializedThreadedRenderer*>(raw);
        const std::lock_guard lock(self.callback_mutex);
        const auto result = self.drain();
        if (result.ok()) self.memory.reset();
        return result;
    }
    static armigo::graphics::CommandResult resolve(void* raw, std::uint64_t,
                                                   std::uint64_t) noexcept {
        auto& self = *static_cast<SerializedThreadedRenderer*>(raw);
        const std::lock_guard lock(self.callback_mutex);
        return self.drain();
    }
    static armigo::graphics::CommandResult status(void* raw) noexcept {
        auto& self = *static_cast<SerializedThreadedRenderer*>(raw);
        const std::lock_guard lock(self.callback_mutex);
        return {};
    }
    armigo::graphics::RenderCallbacks callbacks() {
        armigo::graphics::RenderCallbacks result;
        result.context = this;
        result.clear = result.draw = result.dispatch = result.blit = render;
        result.coherence = result.read_coherence = coherence;
        result.ordered_effect = effect;
        result.memory_changed = memory_changed;
        result.resolve_guest = resolve;
        result.status = status;
        return result;
    }
};

void test_serialized_threaded_callbacks_publish_during_driver_barriers() {
    for (const bool worker : {false, true}) {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        auto renderer = std::make_shared<SerializedThreadedRenderer>();
        auto callbacks = renderer->callbacks();
        callbacks.owner = renderer;
        fixture.session->set_render_backend(std::move(callbacks));
        if (worker) enable_worker(fixture);

        // Empty submits isolate the input-fence barrier from command fetching.
        const auto first = fixture.submit({}, 2, 0, 0);
        CHECK(first.reply.disposition == IpcDisposition::handled);
        CHECK(load_word(first.arguments, 20) == 2);
        const auto dependent = fixture.submit({}, 1, expected_syncpoint, 2);
        CHECK(dependent.reply.disposition == IpcDisposition::handled);
        fixture.session->drain_gpu();
        CHECK(fixture.session->is_fence_expired(expected_syncpoint, 2) == true);

        // Synchronous execution drains the explicit increment through
        // command-fetch coherence. The worker consumes captured command bytes,
        // and its later unmap barrier drains the same pending publication.
        auto commands = normal_draw_packet();
        const auto increments = syncpoint_increments(expected_syncpoint, 1);
        commands.insert(commands.end(), increments.begin(), increments.end());
        const auto draw = normal_draw_packet();
        commands.insert(commands.end(), draw.begin(), draw.end());
        auto clear = rendering_stop_packet();
        clear.resize(clear.size() - 2);
        commands.insert(commands.end(), clear.begin(), clear.end());
        fixture.write_commands(gpu_base, commands);
        const auto descriptor = entry(gpu_base, commands.size());
        const auto submitted = fixture.submit({&descriptor, 1}, 0x102, 0, 1);
        CHECK(submitted.reply.disposition == IpcDisposition::handled);
        CHECK(load_word(submitted.arguments, 20) == 5);

        // Unmap must finish the trailing automatic fence before removing its
        // memory view. It also joins the GPU producer in worker mode.
        std::array<std::byte, 8> unmap{};
        store_qword(unmap, 0, gpu_base);
        CHECK(reply_word(fixture.ioctl(fixture.address_space, address_space_unmap_buffer, unmap),
                         32) == nv_success);
        CHECK(fixture.session->is_fence_expired(expected_syncpoint, 5) == true);
        CHECK(fixture.completed() == 5 && fixture.reserved() == 5);
        CHECK(renderer->renders == 3);
        CHECK(renderer->backend.pending_count() == 0);
        CHECK(!fixture.session->gpu_failure());
        fixture.session.reset();
    }
}

void test_worker_captures_segmented_commands_and_orders_unmap() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    auto renderer = std::make_shared<GatedRenderer>();
    armigo::graphics::RenderCallbacks callbacks{
        GatedRenderer::clear, renderer.get(), nullptr, renderer};
    callbacks.read_coherence = GatedRenderer::read_coherence;
    callbacks.ordered_write = GatedRenderer::ordered_write;
    fixture.session->set_render_backend(std::move(callbacks));
    auto entered = renderer->entered.get_future();
    auto commands = rendering_stop_packet();
    commands.resize(commands.size() - 2);
    fixture.write_commands(gpu_base, commands);
    enable_worker(fixture);
    auto descriptor = entry(gpu_base, commands.size());
    const auto first = fixture.submit({&descriptor, 1}, 2, 0, 0);
    CHECK(first.reply.disposition == IpcDisposition::handled);
    CHECK(load_word(first.arguments, 20) == 2);
    CHECK(entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    CHECK(fixture.completed() == 0 && fixture.reserved() == 2);

    // The next submission splits a packet and repeats the increment through an
    // overlapping segment. All copies must remain in descriptor order after
    // guest memory is overwritten while the preceding render holds the worker.
    const auto later_address = gpu_base + 0x400;
    const auto increments = syncpoint_increments(expected_syncpoint, 1);
    fixture.write_commands(later_address, increments);
    const std::array descriptors{
        entry(later_address, 1),
        entry(later_address + 4, 3),
        entry(later_address + 8, 2),
    };
    fixture.state->record_backing_accesses = true;
    const auto second = fixture.submit(descriptors, 0x100, 0, 2);
    CHECK(second.reply.disposition == IpcDisposition::handled);
    CHECK(load_word(second.arguments, 20) == 4);
    const std::vector<std::uint64_t> submit_reads{
        guest_backing_address(later_address),
        guest_backing_address(later_address + 4),
        guest_backing_address(later_address + 8),
    };
    CHECK(fixture.state->backing_reads == submit_reads);
    fixture.write_commands(later_address,
                           std::vector<std::uint32_t>(increments.size()));
    auto unmap = std::async(std::launch::async, [&] {
        std::array<std::byte, 8> args{};
        store_qword(args, 0, gpu_base);
        return fixture.ioctl(fixture.address_space, address_space_unmap_buffer, args);
    });
    CHECK(unmap.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    renderer->release.set_value();
    CHECK(unmap.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    CHECK(unmap.get().disposition == IpcDisposition::handled);
    fixture.session->drain_gpu();
    CHECK(fixture.completed() == 4);
    CHECK(renderer->calls == 1);
    CHECK(renderer->read_coherence_calls == 0);
    CHECK(fixture.state->backing_reads == submit_reads);
    CHECK(!fixture.session->gpu_failure());
}

void test_worker_partial_capture_failure_uses_fifo_live_fallback() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    auto renderer = std::make_shared<GatedRenderer>();
    fixture.session->set_render_backend(
        {GatedRenderer::clear, renderer.get(), nullptr, renderer});
    auto entered = renderer->entered.get_future();
    auto gated_commands = rendering_stop_packet();
    gated_commands.resize(gated_commands.size() - 2);
    fixture.write_commands(gpu_base, gated_commands);
    enable_worker(fixture);
    const auto gated_descriptor = entry(gpu_base, gated_commands.size());
    CHECK(fixture.submit({&gated_descriptor, 1}, 2, 0, 0)
              .reply.disposition == IpcDisposition::handled);
    CHECK(entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);

    constexpr auto later_address = gpu_base + 0x400;
    constexpr auto tail_address = gpu_base + 0x440;
    const auto increments = syncpoint_increments(expected_syncpoint, 1);
    const std::array<std::uint32_t, 1> nop{};
    fixture.write_commands(later_address, increments);
    fixture.write_commands(tail_address, nop);
    const std::array descriptors{
        entry(later_address, increments.size()),
        entry(tail_address, nop.size()),
    };
    fixture.state->record_backing_accesses = true;
    fixture.state->fail_read_once_address = guest_backing_address(tail_address);
    fixture.state->fail_read_once_signal = std::make_shared<std::promise<void>>();
    auto capture_failed = fixture.state->fail_read_once_signal->get_future();
    auto submitted = std::async(std::launch::async, [&] {
        return fixture.submit(descriptors, 1, expected_syncpoint, 2);
    });
    CHECK(capture_failed.wait_for(std::chrono::seconds(2)) ==
          std::future_status::ready);
    CHECK(submitted.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);

    // A failed later segment invalidates the whole partial snapshot. The FIFO
    // fallback therefore observes this overwrite when it eventually runs.
    fixture.write_commands(later_address,
                           std::vector<std::uint32_t>(increments.size()));
    renderer->release.set_value();
    CHECK(submitted.wait_for(std::chrono::seconds(2)) ==
          std::future_status::ready);
    const auto result = submitted.get();
    CHECK(result.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(result.reply, 32) == nv_success);
    fixture.session->drain_gpu();
    CHECK(fixture.completed() == 2 && fixture.reserved() == 2);
    const std::vector<std::uint64_t> expected_reads{
        guest_backing_address(later_address),
        guest_backing_address(tail_address),
        guest_backing_address(later_address),
        guest_backing_address(tail_address),
    };
    CHECK(fixture.state->backing_reads == expected_reads);
    CHECK(renderer->calls == 1);
    CHECK(!fixture.session->gpu_failure());
}

void test_worker_fallback_input_wait_is_cancelled() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    enable_worker(fixture);
    constexpr auto command_address = gpu_base + 0x400;
    const std::array<std::uint32_t, 1> nop{};
    fixture.write_commands(command_address, nop);
    const auto descriptor = entry(command_address, nop.size());
    fixture.state->fail_read_once_address = guest_backing_address(command_address);
    fixture.state->fail_read_once_signal = std::make_shared<std::promise<void>>();
    auto capture_failed = fixture.state->fail_read_once_signal->get_future();
    auto submitted = std::async(std::launch::async, [&] {
        return fixture.submit({&descriptor, 1}, 3, 1, 1);
    });
    CHECK(capture_failed.wait_for(std::chrono::seconds(2)) ==
          std::future_status::ready);
    CHECK(submitted.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);
    fixture.session->stop_gpu();
    CHECK(submitted.wait_for(std::chrono::seconds(2)) ==
          std::future_status::ready);
    CHECK(submitted.get().reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.session->is_fence_expired(expected_syncpoint, 1) == false);
    CHECK(!fixture.session->gpu_failure());
}

void test_worker_uncapturable_command_errors_are_synchronous() {
    for (const bool oversized : {false, true}) {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        enable_worker(fixture);
        const auto descriptor = oversized
            ? entry(gpu_base, armigo::graphics::GpfifoChannel::max_entry_words + 1)
            : entry(gpu_base + 0x100000, 1);
        const auto submitted = fixture.submit({&descriptor, 1}, 2, 0, 0);
        CHECK(submitted.reply.disposition == IpcDisposition::unsupported);
        const auto failure = fixture.session->gpu_failure();
        CHECK(failure.has_value());
        if (failure) {
            CHECK(failure->find(oversized ? "processing budget" : "outside a mapping") != std::string::npos);
            CHECK(failure->find("completed 0 reserved 2") != std::string::npos);
        }
        CHECK(fixture.submit({}, 0, 0, 0).reply.disposition == IpcDisposition::unsupported);
    }
}

void test_worker_oversized_capture_uses_fifo_live_fallback() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(2048, 1);
    auto renderer = std::make_shared<GatedRenderer>();
    fixture.session->set_render_backend(
        {GatedRenderer::clear, renderer.get(), nullptr, renderer});
    auto entered = renderer->entered.get_future();
    auto gated_commands = rendering_stop_packet();
    gated_commands.resize(gated_commands.size() - 2);
    fixture.write_commands(gpu_base, gated_commands);
    enable_worker(fixture);
    const auto gated_descriptor = entry(gpu_base, gated_commands.size());
    CHECK(fixture.submit({&gated_descriptor, 1}, 2, 0, 0)
              .reply.disposition == IpcDisposition::handled);
    CHECK(entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);

    constexpr auto command_address = gpu_base + 0x1000;
    constexpr std::size_t segment_words = 1024;
    constexpr std::size_t segment_count = 1025;
    fixture.write_commands(command_address,
                           std::vector<std::uint32_t>(segment_words));
    std::vector<std::uint64_t> descriptors(
        segment_count, entry(command_address, segment_words));
    fixture.state->record_backing_accesses = true;
    auto submitted = std::async(std::launch::async, [&] {
        return fixture.submit(descriptors, 0, 0, 0);
    });
    CHECK(submitted.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);
    renderer->release.set_value();
    CHECK(submitted.wait_for(std::chrono::seconds(5)) ==
          std::future_status::ready);
    const auto result = submitted.get();
    CHECK(result.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(result.reply, 32) == nv_success);
    CHECK(fixture.state->backing_reads.size() == segment_count);
    CHECK(fixture.completed() == 2 && fixture.reserved() == 2);
    CHECK(renderer->calls == 1);
    CHECK(!fixture.session->gpu_failure());
}

void test_worker_snapshot_byte_budget_applies_backpressure_and_releases() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(1024, 1);
    auto renderer = std::make_shared<GatedRenderer>();
    fixture.session->set_render_backend(
        {GatedRenderer::clear, renderer.get(), nullptr, renderer});
    auto entered = renderer->entered.get_future();
    auto gated_commands = rendering_stop_packet();
    gated_commands.resize(gated_commands.size() - 2);
    fixture.write_commands(gpu_base, gated_commands);
    enable_worker(fixture);
    const auto gated_descriptor = entry(gpu_base, gated_commands.size());
    CHECK(fixture.submit({&gated_descriptor, 1}, 2, 0, 0)
              .reply.disposition == IpcDisposition::handled);
    CHECK(entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);

    constexpr std::size_t segment_words = 2048;
    constexpr std::size_t segment_count = 512;
    constexpr std::size_t admitted_submissions = 15;
    fixture.write_commands(gpu_base,
                           std::vector<std::uint32_t>(segment_words));
    const std::vector<std::uint64_t> descriptors(
        segment_count, entry(gpu_base, segment_words));
    for (std::size_t index = 0; index < admitted_submissions; ++index) {
        const auto submitted = fixture.submit(descriptors, 0, 0, 0);
        CHECK(submitted.reply.disposition == IpcDisposition::handled);
        CHECK(reply_word(submitted.reply, 32) == nv_success);
    }

    auto blocked = std::async(std::launch::async, [&] {
        return fixture.submit(descriptors, 0, 0, 0);
    });
    CHECK(blocked.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);
    renderer->release.set_value();
    CHECK(blocked.wait_for(std::chrono::seconds(10)) ==
          std::future_status::ready);
    const auto admitted = blocked.get();
    CHECK(admitted.reply.disposition == IpcDisposition::handled);
    CHECK(reply_word(admitted.reply, 32) == nv_success);
    fixture.session->drain_gpu();
    CHECK(fixture.completed() == 2 && fixture.reserved() == 2);
    CHECK(renderer->calls == 1);
    CHECK(!fixture.session->gpu_failure());
}

void test_worker_input_fence_progress_during_map_barrier() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    enable_worker(fixture);
    const auto waiting = fixture.submit({}, 3, 1, 1);
    CHECK(waiting.reply.disposition == IpcDisposition::handled);
    CHECK(load_word(waiting.arguments, 20) == 2);
    auto barrier = std::async(std::launch::async, [&] { fixture.session->drain_gpu(); });
    CHECK(barrier.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    // A clone must be able to advance a fence while structural requests drain.
    DriverSession clone(*fixture.session);
    std::array<std::byte, 4> increment{};
    store_word(increment, 0, 1);
    fixture.copy_to(argument_address, increment);
    GuestBuffers buffers(Fixture::read_memory, Fixture::write_memory, fixture.state.get());
    buffers.add({argument_address, 4, true, true});
    CHECK(clone.request_with_buffers(request(ioctl_command, fixture.control, 0x40040015),
        1, buffers).disposition == IpcDisposition::handled);
    CHECK(barrier.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    barrier.get();
    CHECK(fixture.completed() == 2);
    CHECK(!fixture.session->gpu_failure());
}

void test_submission_failure_wakes_waiters_registered_during_processing() {
    for (const bool worker : {false, true}) {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        auto renderer = std::make_shared<GatedRenderer>();
        renderer->fail = true;
        fixture.session->set_render_backend({GatedRenderer::clear, renderer.get(), nullptr, renderer});
        auto entered = renderer->entered.get_future();
        auto commands = rendering_stop_packet();
        commands.resize(commands.size() - 2);
        fixture.write_commands(gpu_base, commands);
        const auto descriptor = entry(gpu_base, commands.size());
        if (worker) enable_worker(fixture);
        auto submitted = std::async(std::launch::async, [&] {
            return fixture.submit({&descriptor, 1}, 2, 0, 0);
        });
        CHECK(entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
        // In worker mode, finish copying the submit reply before reusing the
        // fixture's IPC argument storage for event registration.
        if (worker) CHECK(submitted.get().reply.disposition == IpcDisposition::handled);
        const auto access = fixture.session->graphics_backing_access(nullptr);
        auto waiter = std::async(std::launch::async, [&] {
            return access.wait_fence({expected_syncpoint, 2},
                std::chrono::steady_clock::now() + std::chrono::seconds(2), {});
        });
        CHECK(waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
        std::array<std::byte, 16> wait{};
        store_word(wait, 0, expected_syncpoint);
        store_word(wait, 4, 2);
        store_word(wait, 8, 1);
        CHECK(reply_word(fixture.ioctl(fixture.control, syncpoint_wait_allocate, wait), 32) == nv_timeout);
        renderer->release.set_value();
        if (!worker) CHECK(submitted.get().reply.disposition == IpcDisposition::unsupported);
        CHECK(failed_with(waiter.get(), LaunchError::host_failure));
        fixture.session->drain_gpu();
        const auto reply = fixture.query_event(fixture.control, load_word(wait, 12));
        const auto event = std::dynamic_pointer_cast<Event>(reply.transferred[0]);
        CHECK(event && event->is_signaled());
        CHECK(!fixture.session->is_fence_expired(expected_syncpoint, 2).has_value());
        CHECK(fixture.session->gpu_failure()->find("Synthetic gated rendering failure") != std::string::npos);
        CHECK(fixture.ioctl(fixture.control, syncpoint_wait_allocate, wait).disposition ==
              IpcDisposition::unsupported);
    }
}

void test_worker_failure_wakes_fence_waiters() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const auto commands = unsupported_dma_packet();
    fixture.write_commands(gpu_base, commands);
    enable_worker(fixture);
    const auto descriptor = entry(gpu_base, commands.size());
    const auto submission = fixture.submit({&descriptor, 1}, 2, 0, 0);
    CHECK(submission.reply.disposition == IpcDisposition::handled);
    const auto access = fixture.session->graphics_backing_access(nullptr);
    CHECK(failed_with(access.wait_fence({expected_syncpoint, 2},
        std::chrono::steady_clock::now() + std::chrono::seconds(2), {}), LaunchError::host_failure));
    fixture.session->drain_gpu();
    CHECK(fixture.session->gpu_failure().has_value());
    CHECK(fixture.submit({}, 2, 0, 0).reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.read_syncpoint(syncpoint_read_min, expected_syncpoint).first.disposition ==
        IpcDisposition::unsupported);
}

void test_worker_shutdown_cancels_input_wait_and_retains_memory() {
    std::weak_ptr<Fixture::MemoryState> retained;
    {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        retained = fixture.state;
        enable_worker(fixture);
        CHECK(fixture.submit({}, 3, 1, 1).reply.disposition == IpcDisposition::handled);
        fixture.session->stop_gpu();
        fixture.session->drain_gpu();
        CHECK(fixture.submit({}, 0, 0, 0).reply.disposition == IpcDisposition::unsupported);
    }
    CHECK(retained.expired());
    // Destruction alone must also cancel an unsatisfiable input dependency.
    auto destruction = std::async(std::launch::async, [] {
        PreparedFixture fixture;
        fixture.allocate_gpfifo(16, 1);
        enable_worker(fixture);
        static_cast<void>(fixture.submit({}, 1, 1, 1));
    });
    CHECK(destruction.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    destruction.get();
}

struct LateBackendFailure {
    std::atomic_bool failed = false;
    static armigo::graphics::CommandResult status(void* raw) noexcept {
        return static_cast<LateBackendFailure*>(raw)->failed.load()
            ? armigo::graphics::CommandResult{armigo::graphics::CommandStatus::Stopped,
                                              "Synthetic asynchronous backend failure"}
            : armigo::graphics::CommandResult{};
    }
};

void test_worker_idle_failure_signals_registered_event() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    auto failure = std::make_shared<LateBackendFailure>();
    armigo::graphics::RenderCallbacks callbacks;
    callbacks.context = failure.get();
    callbacks.owner = failure;
    callbacks.status = LateBackendFailure::status;
    fixture.session->set_render_backend(callbacks);
    enable_worker(fixture);
    std::array<std::byte, 16> wait{};
    store_word(wait, 0, 1);
    store_word(wait, 4, 1);
    store_word(wait, 8, 1);
    CHECK(reply_word(fixture.ioctl(fixture.control, syncpoint_wait_allocate, wait), 32) == nv_timeout);
    const auto reply = fixture.query_event(fixture.control, load_word(wait, 12));
    const auto event = std::dynamic_pointer_cast<Event>(reply.transferred[0]);
    CHECK(event && !event->is_signaled());
    armigo::horizon::HandleTable handles(4);
    const auto handle = handles.insert(event);
    CHECK(handle.has_value());
    failure->failed = true;
    if (handle) {
        const std::array selected{*handle};
        const auto result = armigo::horizon::wait_event(handles, selected, 2'000'000'000);
        CHECK(result.result == 0);
    }
    CHECK(fixture.session->gpu_failure().has_value());
    CHECK(fixture.submit({}, 0, 0, 0).reply.disposition == IpcDisposition::unsupported);
    CHECK(fixture.read_syncpoint(syncpoint_read_min, 1).first.disposition == IpcDisposition::unsupported);
    CHECK(!fixture.session->is_fence_expired(1, 1).has_value());
}

void test_worker_channel_failure_is_scoped_and_dependency_fails() {
    PreparedFixture fixture;
    fixture.allocate_gpfifo(16, 1);
    const auto first_channel = fixture.channel;
    const auto commands = unsupported_dma_packet();
    fixture.write_commands(gpu_base, commands);
    enable_worker(fixture);
    const auto descriptor = entry(gpu_base, commands.size());
    CHECK(fixture.submit({&descriptor, 1}, 2, 0, 0).reply.disposition == IpcDisposition::handled);
    fixture.session->drain_gpu();
    CHECK(fixture.session->gpu_failure().has_value());

    fixture.channel = fixture.open("/dev/nvhost-gpu");
    std::array<std::byte, 4> binding{};
    store_word(binding, 0, fixture.nvmap);
    CHECK(reply_word(fixture.ioctl(fixture.channel, channel_set_nvmap, binding), 32) == nv_success);
    store_word(binding, 0, fixture.channel);
    CHECK(reply_word(fixture.ioctl(fixture.address_space, address_space_bind_channel, binding), 32) == nv_success);
    fixture.allocate_gpfifo(16, 1);
    const auto independent = fixture.submit({}, 2, 0, 0);
    CHECK(independent.reply.disposition == IpcDisposition::handled);
    const auto own_syncpoint = load_word(independent.arguments, 16);
    CHECK(own_syncpoint != expected_syncpoint);
    fixture.session->drain_gpu();
    CHECK(fixture.read_syncpoint(syncpoint_read_min, own_syncpoint).second == 2);
    CHECK(fixture.submit({}, 1, expected_syncpoint, 2).reply.disposition == IpcDisposition::handled);
    fixture.session->drain_gpu();
    CHECK(fixture.read_syncpoint(syncpoint_read_min, own_syncpoint).first.disposition == IpcDisposition::unsupported);
    fixture.channel = first_channel;
}

}  // namespace

int main() {
    test_submission_failure_wakes_waiters_registered_during_processing();
    test_serialized_threaded_callbacks_publish_during_driver_barriers();
    test_worker_idle_failure_signals_registered_event();
    test_worker_channel_failure_is_scoped_and_dependency_fails();
    test_worker_captures_segmented_commands_and_orders_unmap();
    test_worker_partial_capture_failure_uses_fifo_live_fallback();
    test_worker_fallback_input_wait_is_cancelled();
    test_worker_uncapturable_command_errors_are_synchronous();
    test_worker_oversized_capture_uses_fifo_live_fallback();
    test_worker_snapshot_byte_budget_applies_backpressure_and_releases();
    test_worker_input_fence_progress_during_map_barrier();
    test_worker_failure_wakes_fence_waiters();
    test_worker_shutdown_cancels_input_wait_and_retains_memory();
    test_nvmap_free_output_contract();
    test_nvmap_free_import_references();
    test_nvmap_free_missing_handles_are_noops();
    test_nvmap_free_failure_atomicity();
    test_nvmap_free_ignores_renderer_reference_counts();
    test_gpu_mapping_retains_freed_handle();
    test_whole_mapping_kind_remap();
    test_kind_remap_rejection_is_atomic();
    test_kind_remap_preserves_va_reservations();
    test_mapping_requires_owned_allocated_backing();
    test_unmap_releases_only_automatic_reservations();
    test_memory_change_notifications();
    test_sparse_remap();
    test_allocation_validation_and_first_syncpoint();
    test_reservation_and_guest_syncpoint_methods();
    test_client_scoped_graphics_backing_access();
    test_graphics_fence_waits();
    test_automatic_increment_requires_completed_work();
    test_invalid_submission_inputs_have_no_effects();
    test_incomplete_automatic_increment_is_sticky();
    test_incomplete_upload_does_not_complete_automatic_fence();
    test_rendering_stop_is_pending_and_sticky();
    test_faults_and_unsupported_actions_do_not_complete();
    test_pending_wait_event_signals_after_gpu_increment();
    test_control_event_slots_per_descriptor();
    test_backend_completion_and_clone_sharing();
    test_async_increment_waits_for_idle_publication();
    test_async_automatic_fence_preserves_order_and_count();
    test_input_fence_flushes_pending_publication_inline();
    test_syncpoint_acquire_flushes_pending_publication_inline();
    test_ordered_write_precedes_next_command_fetch();
    test_ordered_write_only_backend_precedes_next_command_fetch();
    test_async_publication_rejection_and_driver_teardown();
    test_render_backend_owner_survives_registration_clone();
    test_failed_render_owner_survives_backend_replacement();
    test_draw_orders_reports_semaphores_and_fences();
    test_draw_writeback_precedes_next_entry_fetch();
    return failures == 0 ? 0 : 1;
}
