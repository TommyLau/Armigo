// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "service_registry.h"
#include "../../src/nvn/backend.h"

#include <armigo/horizon/startup_queries.h>
#include <armigo/runtime/presentation_clock.h>
#include <armigo/services/applet.h>
#include <armigo/services/nvdrv.h>
#include <armigo/services/log_manager.h>
#include <armigo/services/service_manager_session.h>
#include <armigo/services/video_interface.h>
#include <armigo/services/vi_producer.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>
#include <sys/mman.h>
#include <stdexcept>

namespace {

using namespace armigo;
using namespace armigo::runtime;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

LaunchFailure test_failure(const char* operation) {
    LaunchFailure failure;
    failure.stage = LaunchStage::memory;
    failure.code = LaunchError::unsupported;
    failure.operation = operation;
    failure.detail = "the synthetic registry test does not provide guest backing";
    return failure;
}

constexpr std::uint64_t test_memory_base = 0x7000000000ULL;
constexpr std::size_t test_memory_size = 0x20000;
constexpr std::uint64_t test_path_address = test_memory_base + 0x1000;
constexpr std::uint64_t test_argument_address = test_memory_base + 0x2000;
constexpr std::uint64_t test_backing_address = test_memory_base + 0x10000;
constexpr std::uint32_t test_backing_size = 0x1000;

class TestStorage {
public:
    explicit TestStorage(bool native = false) {
        if (!native) { bytes_.resize(test_memory_size); return; }
        auto* wanted = reinterpret_cast<void*>(test_memory_base);
        mapped_ = mmap(wanted, test_memory_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
        if (mapped_ != wanted) {
            if (mapped_ != MAP_FAILED) munmap(mapped_, test_memory_size);
            throw std::runtime_error("Synthetic native GPU memory address is unavailable");
        }
    }
    ~TestStorage() { if (mapped_) munmap(mapped_, test_memory_size); }
    std::byte* data() { return mapped_ ? static_cast<std::byte*>(mapped_) : bytes_.data(); }
    std::size_t size() const { return test_memory_size; }
private:
    void* mapped_ = nullptr;
    std::vector<std::byte> bytes_;
};

class TestPin final : public PinnedGuestMemory {
public:
    TestPin(std::shared_ptr<TestStorage> storage,
            horizon::AddressRange range, std::size_t offset)
        : storage_(std::move(storage)), range_(range), offset_(offset) {}

    horizon::AddressRange range() const noexcept override { return range_; }
    memory::Permission permissions() const noexcept override {
        return memory::Permission::read | memory::Permission::write;
    }
    std::span<const std::byte> bytes() const noexcept override {
        return std::span<const std::byte>(storage_->data(), storage_->size()).subspan(offset_, range_.size);
    }
    std::span<std::byte> writable_bytes() noexcept override {
        return std::span<std::byte>(storage_->data(), storage_->size()).subspan(offset_, range_.size);
    }

private:
    std::shared_ptr<TestStorage> storage_;
    horizon::AddressRange range_{};
    std::size_t offset_ = 0;
};

class TestMemory final : public GuestMemory {
public:
    explicit TestMemory(bool native = false) : storage_(std::make_shared<TestStorage>(native)) {
        horizon::ProcessConfiguration configuration;
        configuration.heap_address = 0x1000000000;
        configuration.alias_address = 0x2000000000;
        configuration.stack_address = 0x6000000000;
        configuration.allowed_core_mask = 1;
        horizon::MemoryAccounting accounting;
        accounting.resource_limit_bytes = 1 << 20;
        auto built = horizon::ProcessRecord::create(configuration, accounting);
        CHECK(built.ok());
        process_ = std::move(built.process);
    }

    std::shared_ptr<horizon::ProcessRecord> process() const noexcept override {
        return process_;
    }
    LaunchResult<GuestMemoryPin> allocate(
        std::uint64_t, horizon::MemoryType, memory::Permission) override {
        return test_failure("test_memory.allocate");
    }
    LaunchStatus release(horizon::AddressRange) override {
        return test_failure("test_memory.release");
    }
    LaunchResult<horizon::QueryDisposition> dispatch(
        std::uint16_t, horizon::QueryRegisters&) override {
        return horizon::QueryDisposition::unsupported_operation;
    }
    LaunchResult<GuestMemoryPin> pin(
        horizon::AddressRange range, memory::Permission access) override {
        if (range.address < test_memory_base || range.size == 0 ||
            range.address - test_memory_base > storage_->size() ||
            range.size > storage_->size() - (range.address - test_memory_base) ||
            (static_cast<unsigned>(access) &
             ~static_cast<unsigned>(memory::Permission::read | memory::Permission::write)) != 0) {
            return test_failure("test_memory.pin");
        }
        return GuestMemoryPin{std::make_shared<TestPin>(
            storage_, range, static_cast<std::size_t>(range.address - test_memory_base))};
    }
    bool copy_observation(std::uint64_t address, std::span<std::byte> out) const noexcept override {
        ++observation_reads;
        return read(const_cast<TestMemory*>(this), address, out);
    }
    mutable unsigned observation_reads = 0;
    horizon::GuestCopyOut copy_out() noexcept override { return {}; }
    horizon::GuestBuffers ipc_buffers() noexcept override {
        return {&TestMemory::read, &TestMemory::write, this};
    }
    horizon::GuestMemoryMapper service_mapper() noexcept override {
        return {&TestMemory::map, &TestMemory::unmap, this, storage_};
    }

    bool copy_to(std::uint64_t address, std::span<const std::byte> input) noexcept {
        return write(this, address, input);
    }
    bool copy_from(std::uint64_t address, std::span<std::byte> output) noexcept {
        return read(this, address, output);
    }

    static bool read(void* context, std::uint64_t address,
                     std::span<std::byte> output) noexcept {
        auto& self = *static_cast<TestMemory*>(context);
        if (address < test_memory_base ||
            address - test_memory_base > self.storage_->size() ||
            output.size() > self.storage_->size() - (address - test_memory_base))
            return false;
        std::memcpy(output.data(), self.storage_->data() +
            static_cast<std::size_t>(address - test_memory_base), output.size());
        return true;
    }

    static bool write(void* context, std::uint64_t address,
                      std::span<const std::byte> input) noexcept {
        auto& self = *static_cast<TestMemory*>(context);
        if (address < test_memory_base ||
            address - test_memory_base > self.storage_->size() ||
            input.size() > self.storage_->size() - (address - test_memory_base))
            return false;
        std::memcpy(self.storage_->data() +
            static_cast<std::size_t>(address - test_memory_base), input.data(), input.size());
        return true;
    }

    static bool map(void* context, std::uint64_t address,
                    std::uint64_t size) noexcept {
        auto& self = *static_cast<TestMemory*>(context);
        return address >= test_memory_base && size != 0 &&
               address - test_memory_base <= self.storage_->size() &&
               size <= self.storage_->size() - (address - test_memory_base);
    }

    static void unmap(void* context, std::uint64_t, std::uint64_t) noexcept {
        ++static_cast<TestMemory*>(context)->unmap_calls;
    }

    unsigned unmap_calls = 0;

private:
    std::shared_ptr<horizon::ProcessRecord> process_;
    std::shared_ptr<TestStorage> storage_;
};

class TestPresenter final : public platform::Presenter {
public:
    platform::PresentationMode mode() const noexcept override {
        return platform::PresentationMode::headless;
    }
    platform::PresentResult present(const platform::PresentationFrame& frame,
        std::optional<std::chrono::steady_clock::time_point>,
        std::stop_token) override {
        std::unique_lock lock(mutex);
        ++presented;
        entered = true;
        changed.notify_all();
        changed.wait(lock, [this] { return !block; });
        if (fail) {
            return platform::PresentationFailure{
                platform::PresentationError::host_failure,
                "injected registry presentation failure"};
        }
        platform::PresentedFrame completed;
        completed.frame_number = frame.queued.frame_number;
        completed.readback = *frame.pixels;
        return completed;
    }
    platform::PresenterStatus drain(
        std::optional<std::chrono::steady_clock::time_point>) override {
        ++drains;
        return std::monostate{};
    }
    unsigned drains = 0;
    std::mutex mutex;
    std::condition_variable changed;
    unsigned presented = 0;
    bool entered = false;
    bool block = false;
    bool fail = false;

    bool wait_until_entered() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(2),
                                [this] { return entered; });
    }

    void release() {
        const std::lock_guard lock(mutex);
        block = false;
        changed.notify_all();
    }
};

void write(std::span<std::byte> bytes, std::size_t offset,
           std::uint64_t value, std::size_t width) {
    for (std::size_t index = 0; index < width; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 0xff);
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index)
        value |= std::uint32_t{std::to_integer<std::uint8_t>(bytes[offset + index])}
                 << (8 * index);
    return value;
}

std::vector<std::byte> normalized_fs_access(std::uint64_t permissions) {
    std::vector<std::byte> bytes(0x1c);
    write(bytes, 0, 1, 4);
    write(bytes, 4, permissions, 8);
    write(bytes, 0x0c, 0x1c, 4);
    write(bytes, 0x14, 0x1c, 4);
    return bytes;
}

std::vector<std::byte> service_acl(std::string_view name) {
    std::vector<std::byte> bytes;
    bytes.push_back(static_cast<std::byte>(name.size() - 1));
    for (char value : name) bytes.push_back(static_cast<std::byte>(value));
    return bytes;
}

LaunchProfile profile(std::vector<std::byte> acl =
                          {std::byte{0}, std::byte{'*'}}) {
    LaunchProfile result;
    result.origin = ProfileOrigin::synthetic_test;
    result.process.allowed_core_mask = 1;
    result.permissions.service_access_control = std::move(acl);
    result.permissions.file_system_access_control =
        normalized_fs_access(0x3fff'ffff'ffff'ffffULL);
    result.permissions.handle_table_capacity = 128;
    return result;
}

struct Fixture {
    std::shared_ptr<horizon::HandleTable> handles =
        std::make_shared<horizon::HandleTable>(128);
    std::shared_ptr<TestMemory> memory = std::make_shared<TestMemory>();
    std::shared_ptr<GuestClock> clock;
    std::shared_ptr<PresentationClock> presentation_clock;
    std::shared_ptr<TestPresenter> presenter = std::make_shared<TestPresenter>();

    Fixture() {
        auto made_clock = create_guest_clock();
        CHECK(std::holds_alternative<std::shared_ptr<GuestClock>>(made_clock));
        if (auto* value = std::get_if<std::shared_ptr<GuestClock>>(&made_clock))
            clock = *value;
        auto sample = clock->sample();
        CHECK(std::holds_alternative<ClockSample>(sample));
        if (auto* value = std::get_if<ClockSample>(&sample)) {
            auto made_presentation = create_presentation_clock(
                clock, {value->instant, std::chrono::nanoseconds{16'666'667}});
            CHECK(std::holds_alternative<std::shared_ptr<PresentationClock>>(
                made_presentation));
            if (auto* created = std::get_if<std::shared_ptr<PresentationClock>>(
                    &made_presentation))
                presentation_clock = *created;
        }
    }

    ServiceRegistryDependencies dependencies(
        LaunchProfile admitted,
        services::OperationMode operation_mode = services::OperationMode::Console) {
        ServiceRegistryDependencies result;
        result.profile = std::move(admitted);
        result.process_id = 71;
        result.operation_mode = operation_mode;
        result.handles = handles;
        result.memory = memory;
        result.clock = clock;
        result.presentation_clock = presentation_clock;
        result.presenter = presenter;
        result.acquire_graphics_access = []()
            -> LaunchResult<std::shared_ptr<void>> {
            return std::static_pointer_cast<void>(std::make_shared<int>(1));
        };
        return result;
    }

    std::shared_ptr<ServiceRegistry> create(
        LaunchProfile admitted = profile(),
        services::OperationMode operation_mode = services::OperationMode::Console) {
        auto made = create_service_registry(
            dependencies(std::move(admitted), operation_mode));
        CHECK(std::holds_alternative<std::shared_ptr<ServiceRegistry>>(made));
        if (auto* value = std::get_if<std::shared_ptr<ServiceRegistry>>(&made))
            return *value;
        return {};
    }
};

horizon::IpcMessage register_client() {
    horizon::IpcMessage request{};
    write(request, 0, 0x8000000000000010ULL, 8);
    write(request, 8, 1, 4);
    write(request, 12, 0xfeed, 8);
    return request;
}

horizon::IpcMessage get_service(std::string_view name) {
    horizon::IpcMessage request{};
    write(request, 0, 0x0000000200000011ULL, 8);
    for (std::size_t index = 0; index < name.size(); ++index)
        request[8 + index] = static_cast<std::byte>(name[index]);
    return request;
}

std::shared_ptr<services::sm::ServiceManagerSession> connect_sm(
    const std::shared_ptr<ServiceRegistry>& registry) {
    auto connection = registry->connect_named_port("sm:");
    CHECK(connection.disposition == horizon::IpcDisposition::handled);
    CHECK(connection.result == 0 && connection.session);
    auto sm = std::dynamic_pointer_cast<services::sm::ServiceManagerSession>(
        connection.session);
    CHECK(sm != nullptr);
    auto request = register_client();
    const auto reply = sm->request(request, 71);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(read_u32(reply.bytes, 8) == 0);
    return sm;
}

horizon::IpcReply lookup(
    const std::shared_ptr<services::sm::ServiceManagerSession>& sm,
    std::string_view name) {
    auto request = get_service(name);
    return sm->request(request, 71);
}

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t nv_success = 0;
constexpr std::uint32_t nvmap_create = 0xc0080101;
constexpr std::uint32_t nvmap_allocate = 0xc0200104;
constexpr std::uint32_t nvmap_get_id = 0xc008010e;

horizon::IpcMessage service_command(std::uint32_t command) {
    horizon::IpcMessage message{};
    write(message, 0, 6, 4);
    write(message, 4, 8, 4);
    write(message, 16, sfci, 4);
    write(message, 20, 1, 4);
    write(message, 24, command, 4);
    return message;
}

horizon::IpcMessage applet_control(std::uint32_t command) {
    horizon::IpcMessage message{};
    write(message, 0, 5, 4);
    write(message, 4, 8, 4);
    write(message, 16, sfci, 4);
    write(message, 24, command, 4);
    return message;
}

horizon::IpcMessage applet_open_proxy(std::uint32_t root) {
    horizon::IpcMessage message{};
    write(message, 0, 6, 4);
    write(message, 4, 0x8000000e, 4);
    write(message, 8, 3, 4);
    write(message, 12, 999, 8);
    write(message, 20, horizon::current_process_handle, 4);
    write(message, 32, 0x00180001, 4);
    write(message, 36, root, 4);
    write(message, 48, sfci, 4);
    write(message, 52, 1, 4);
    return message;
}

horizon::IpcMessage applet_domain_command(
    std::uint32_t object, std::uint32_t command) {
    horizon::IpcMessage message{};
    write(message, 0, 6, 4);
    write(message, 4, 12, 4);
    write(message, 16, 0x00100001, 4);
    write(message, 20, object, 4);
    write(message, 32, sfci, 4);
    write(message, 36, 1, 4);
    write(message, 40, command, 4);
    return message;
}

horizon::IpcMessage driver_request(std::uint32_t command,
                                   std::uint32_t fd = 0,
                                   std::uint32_t operation = 0) {
    auto message = service_command(command);
    write(message, 4, 12, 4);
    write(message, 32, fd, 4);
    write(message, 36, operation, 4);
    return message;
}

struct DriverHarness {
    std::shared_ptr<services::nvdrv::DriverSession> driver;
    std::shared_ptr<TestMemory> memory;

    void copy_to(std::uint64_t address, std::span<const std::byte> bytes) {
        CHECK(memory->copy_to(address, bytes));
    }

    void copy_from(std::uint64_t address, std::span<std::byte> bytes) {
        CHECK(memory->copy_from(address, bytes));
    }

    std::uint32_t open(std::string_view path) {
        copy_to(test_path_address, std::as_bytes(std::span(path.data(), path.size())));
        horizon::GuestBuffers buffers(&TestMemory::read, &TestMemory::write,
                                      memory.get());
        buffers.add({test_path_address, static_cast<std::uint32_t>(path.size()),
                     true, false});
        const auto reply = driver->request_with_buffers(driver_request(0), 71, buffers);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 36) == nv_success);
        return read_u32(reply.bytes, 32);
    }

    horizon::IpcReply ioctl(std::uint32_t fd, std::uint32_t operation,
                            std::span<std::byte> arguments,
                            std::span<const std::byte> inline_data = {}) {
        copy_to(test_argument_address, arguments);
        horizon::GuestBuffers buffers(&TestMemory::read, &TestMemory::write,
                                      memory.get());
        buffers.add({test_argument_address,
                     static_cast<std::uint32_t>(arguments.size()), true, true});
        if (!inline_data.empty()) {
            copy_to(test_path_address, inline_data);
            buffers.add({test_path_address,
                         static_cast<std::uint32_t>(inline_data.size()), true, false});
        }
        const auto reply = driver->request_with_buffers(
            driver_request(inline_data.empty() ? 1 : 11, fd, operation), 71, buffers);
        copy_from(test_argument_address, arguments);
        return reply;
    }

    std::uint32_t export_backing() {
        driver->set_memory_mapper(memory->service_mapper());
        const auto nvmap = open("/dev/nvmap");
        std::array<std::byte, 8> create{};
        write(create, 0, test_backing_size, 4);
        auto reply = ioctl(nvmap, nvmap_create, create);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 32) == nv_success);
        const auto handle = read_u32(create, 4);
        CHECK(handle != 0);

        std::array<std::byte, 0x20> allocate{};
        write(allocate, 0, handle, 4);
        write(allocate, 12, 0x1000, 4);
        write(allocate, 24, test_backing_address, 8);
        reply = ioctl(nvmap, nvmap_allocate, allocate);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 32) == nv_success);

        std::array<std::byte, 8> exported{};
        write(exported, 4, handle, 4);
        reply = ioctl(nvmap, nvmap_get_id, exported);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 32) == nv_success);
        CHECK(read_u32(exported, 0) == handle);
        return handle;
    }
};

struct BinderBuffers {
    std::array<std::byte, 0x200> input{};
    std::array<std::byte, 0x200> output{};
    bool fail_write = false;

    horizon::GuestBuffers buffers(std::uint32_t input_size) {
        horizon::GuestBuffers result(
            [](void* context, std::uint64_t address,
               std::span<std::byte> destination) noexcept {
                auto& self = *static_cast<BinderBuffers*>(context);
                if (address != 0x1000 || destination.size() > self.input.size())
                    return false;
                std::memcpy(destination.data(), self.input.data(), destination.size());
                return true;
            },
            [](void* context, std::uint64_t address,
               std::span<const std::byte> source) noexcept {
                auto& self = *static_cast<BinderBuffers*>(context);
                if (self.fail_write || address != 0x2000 ||
                    source.size() > self.output.size())
                    return false;
                std::memcpy(self.output.data(), source.data(), source.size());
                return true;
            }, this);
        result.add({0x1000, input_size, true, false});
        result.add({0x2000, static_cast<std::uint32_t>(output.size()), false, true});
        return result;
    }

    void set_word(std::size_t offset, std::uint32_t value) {
        std::memcpy(input.data() + offset, &value, sizeof value);
    }

    std::size_t parcel(std::span<const std::uint32_t> arguments) {
        constexpr std::string_view token = "android.gui.IGraphicBufferProducer";
        input.fill(std::byte{});
        set_word(4, 16);
        set_word(20, static_cast<std::uint32_t>(token.size()));
        for (std::size_t index = 0; index < token.size(); ++index)
            input[24 + index * 2] = static_cast<std::byte>(token[index]);
        const std::size_t at = (24 + (token.size() + 1) * 2 + 3) & ~std::size_t{3};
        for (std::size_t index = 0; index < arguments.size(); ++index)
            set_word(at + index * 4, arguments[index]);
        set_word(0, static_cast<std::uint32_t>(at + arguments.size() * 4 - 16));
        return at;
    }
};

horizon::IpcMessage binder_transaction(std::uint32_t operation) {
    auto message = service_command(3);
    write(message, 32, 1, 4);
    write(message, 36, operation, 4);
    return message;
}

std::shared_ptr<horizon::ClientSession> transferred_session(
    const std::shared_ptr<horizon::ClientSession>& session,
    horizon::IpcMessage message) {
    auto reply = session->request(message, 71);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    auto transferred = std::dynamic_pointer_cast<horizon::ClientSession>(
        reply.transferred[0]);
    CHECK(transferred != nullptr);
    return transferred;
}

std::shared_ptr<horizon::ClientSession> clone_session(
    const std::shared_ptr<horizon::ClientSession>& session) {
    horizon::IpcMessage clone{};
    write(clone, 0, 7, 4);
    write(clone, 4, 8, 4);
    write(clone, 16, sfci, 4);
    write(clone, 24, 2, 4);
    return transferred_session(session, clone);
}

std::array<std::byte, 0x16c> flattened_buffer(std::uint32_t export_id) {
    std::array<std::byte, 0x16c> flat{};
    write(flat, 0, 0x47424652, 4);
    write(flat, 4, 8, 4);
    write(flat, 8, 8, 4);
    write(flat, 0x0c, 8, 4);
    write(flat, 0x10, 1, 4);
    write(flat, 0x14, 0x800, 4);
    write(flat, 0x20, 0, 4);
    write(flat, 0x24, 81, 4);
    write(flat, 0x2c, export_id, 4);
    write(flat, 0x34, 0xdaffcaff, 4);
    write(flat, 0x40, 0x800, 4);
    write(flat, 0x44, 1, 4);
    write(flat, 0x48, 1, 4);
    write(flat, 0x4c, 8, 4);
    write(flat, 0x50, 0x200, 4);
    write(flat, 0x54, 1, 4);
    write(flat, 0x5c, 8, 4);
    write(flat, 0x60, 8, 4);
    write(flat, 0x64, 0x00532120, 4);
    write(flat, 0x68, 1, 4);
    write(flat, 0x6c, 3, 4);
    write(flat, 0x70, 64, 4);
    write(flat, 0x78, 0, 4);
    write(flat, 0x80, 0, 4);
    write(flat, 0x94, 0x200, 8);
    return flat;
}

void write_authored_surface(TestMemory& memory) {
    std::array<std::byte, 0x200> surface{};
    constexpr std::array<unsigned, 8> columns{0, 4, 8, 12, 32, 36, 40, 44};
    constexpr std::array<unsigned, 8> rows{0, 16, 64, 80, 128, 144, 192, 208};
    for (std::uint32_t y = 0; y < 8; ++y) {
        for (std::uint32_t x = 0; x < 8; ++x) {
            const auto at = rows[y] + columns[x];
            surface[at] = static_cast<std::byte>(x);
            surface[at + 1] = static_cast<std::byte>(y);
            surface[at + 2] = static_cast<std::byte>(x + y);
            surface[at + 3] = std::byte{0xff};
        }
    }
    CHECK(memory.copy_to(test_backing_address, surface));
}

horizon::IpcReply set_preallocated_buffer(
    const std::shared_ptr<horizon::ClientSession>& relay,
    std::uint32_t export_id) {
    BinderBuffers data;
    const auto at = data.parcel(std::array<std::uint32_t, 4>{0, 1, 0x16c, 0});
    const auto flat = flattened_buffer(export_id);
    std::copy(flat.begin(), flat.end(), data.input.begin() + at + 16);
    const auto size = at + 16 + flat.size();
    data.set_word(0, static_cast<std::uint32_t>(size - 16));
    auto buffers = data.buffers(static_cast<std::uint32_t>(size));
    return relay->request_with_buffers(binder_transaction(14), 71, buffers);
}

std::pair<horizon::IpcReply, std::int32_t> dequeue_buffer(
    const std::shared_ptr<horizon::ClientSession>& relay) {
    BinderBuffers data;
    const auto at = data.parcel(std::array<std::uint32_t, 5>{0, 8, 8, 1, 0});
    auto buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    auto reply = relay->request_with_buffers(binder_transaction(3), 71, buffers);
    return {std::move(reply), std::bit_cast<std::int32_t>(read_u32(data.output, 0x10))};
}

horizon::IpcReply request_buffer(
    const std::shared_ptr<horizon::ClientSession>& relay) {
    BinderBuffers data;
    const auto at = data.parcel(std::array<std::uint32_t, 1>{0});
    auto buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
    return relay->request_with_buffers(binder_transaction(1), 71, buffers);
}

horizon::IpcReply queue_buffer(
    const std::shared_ptr<horizon::ClientSession>& relay,
    bool fail_reply_copy = false) {
    std::array<std::uint32_t, 24> arguments{};
    arguments[0] = 0;
    arguments[1] = 84;
    arguments[5] = 1;
    arguments[12] = 8;
    BinderBuffers data;
    const auto at = data.parcel(arguments);
    data.fail_write = fail_reply_copy;
    auto buffers = data.buffers(
        static_cast<std::uint32_t>(at + arguments.size() * 4));
    return relay->request_with_buffers(binder_transaction(7), 71, buffers);
}

std::shared_ptr<horizon::ClientSession> open_registry_service(
    Fixture& fixture,
    const std::shared_ptr<services::sm::ServiceManagerSession>& sm,
    std::string_view name) {
    const auto reply = lookup(sm, name);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(read_u32(reply.bytes, 16) == 0);
    const auto handle = read_u32(reply.bytes, 12);
    auto session = fixture.handles->lookup_as<horizon::ClientSession>(handle);
    CHECK(session != nullptr);
    CHECK(fixture.handles->close(handle));
    return session;
}

std::uint32_t binder_query(
    const std::shared_ptr<horizon::ClientSession>& relay,
    std::uint32_t property) {
    BinderBuffers data;
    const auto at = data.parcel(std::array<std::uint32_t, 1>{property});
    auto buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
    const auto reply = relay->request_with_buffers(binder_transaction(9), 71, buffers);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    return read_u32(data.output, 0x10);
}

void test_operation_mode_reaches_registered_services() {
    using services::OperationMode;
    for (const auto mode : {OperationMode::Console, OperationMode::Handheld}) {
        Fixture fixture;
        auto registry = fixture.create(profile(), mode);
        auto sm = connect_sm(registry);
        const auto expected = services::operation_mode_profile(mode);

        auto apm = open_registry_service(fixture, sm, "apm");
        const auto apm_reply = apm->request(service_command(1), 71);
        CHECK(apm_reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(apm_reply.bytes, 32) == expected.performance_mode);

        auto applet = open_registry_service(fixture, sm, "appletOE");
        auto applet_reply = applet->request(applet_control(0), 71);
        CHECK(applet_reply.disposition == horizon::IpcDisposition::handled);
        const auto domain_root = read_u32(applet_reply.bytes, 32);
        CHECK(domain_root != 0);
        applet_reply = applet->request(applet_open_proxy(domain_root), 71);
        CHECK(applet_reply.disposition == horizon::IpcDisposition::handled);
        const auto proxy = read_u32(applet_reply.bytes, 48);
        CHECK(proxy != 0);
        applet_reply = applet->request(applet_domain_command(proxy, 0), 71);
        CHECK(applet_reply.disposition == horizon::IpcDisposition::handled);
        const auto common_state = read_u32(applet_reply.bytes, 48);
        CHECK(common_state != 0);
        applet_reply = applet->request(applet_domain_command(common_state, 5), 71);
        CHECK(applet_reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(applet_reply.bytes, 48) == static_cast<std::uint32_t>(mode));
        applet_reply = applet->request(applet_domain_command(common_state, 6), 71);
        CHECK(applet_reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(applet_reply.bytes, 48) == expected.performance_mode);
        applet_reply = applet->request(applet_domain_command(common_state, 60), 71);
        CHECK(applet_reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(applet_reply.bytes, 48) == expected.display_width);
        CHECK(read_u32(applet_reply.bytes, 52) == expected.display_height);

        auto vi = open_registry_service(fixture, sm, "vi:u");
        auto display = transferred_session(vi, service_command(1));
        auto relay = transferred_session(display, service_command(100));
        CHECK(binder_query(relay, 0) == expected.display_width);
        CHECK(binder_query(relay, 1) == expected.display_height);
        CHECK(!registry->failure());
    }
}

void test_registry_legacy_memory_release() {
    Fixture fixture;
    unsigned acquisitions = 0;
    auto dependencies = fixture.dependencies(profile(service_acl("nvdrv")));
    dependencies.acquire_graphics_access = [&]() -> LaunchResult<std::shared_ptr<void>> {
        ++acquisitions;
        return test_failure("test.graphics_access_denied");
    };
    auto made = create_service_registry(std::move(dependencies));
    CHECK(std::holds_alternative<std::shared_ptr<ServiceRegistry>>(made));
    if (!std::holds_alternative<std::shared_ptr<ServiceRegistry>>(made)) return;
    auto registry = std::get<std::shared_ptr<ServiceRegistry>>(std::move(made));
    auto sm = connect_sm(registry);
    auto driver = std::dynamic_pointer_cast<services::nvdrv::DriverSession>(
        open_registry_service(fixture, sm, "nvdrv"));
    CHECK(driver != nullptr);
    if (!driver) return;
    DriverHarness graphics{driver, fixture.memory};
    const auto handle = graphics.export_backing();
    const auto nvmap = graphics.open("/dev/nvmap");
    std::array<std::byte, 0x18> free{};
    write(free, 0, handle, 4);
    const auto reply = graphics.ioctl(nvmap, 0xc0180105, free);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(read_u32(reply.bytes, 32) == nv_success);
    CHECK(fixture.memory->unmap_calls == 1);
    CHECK(acquisitions == 0);
    CHECK(!registry->failure());
}

void test_registry_gpu_memory_release() {
    Fixture fixture;
    fixture.memory = std::make_shared<TestMemory>(true);
    unsigned acquisitions = 0;
    unsigned releases = 0;
    auto dependencies = fixture.dependencies(profile(service_acl("nvdrv")));
    dependencies.acquire_graphics_access = [&]() -> LaunchResult<std::shared_ptr<void>> {
        CHECK(acquisitions == releases);
        ++acquisitions;
        return std::shared_ptr<void>(new int(1), [&](void* value) {
            delete static_cast<int*>(value);
            ++releases;
        });
    };
    auto made = create_service_registry(std::move(dependencies));
    CHECK(std::holds_alternative<std::shared_ptr<ServiceRegistry>>(made));
    if (!std::holds_alternative<std::shared_ptr<ServiceRegistry>>(made)) return;
    auto registry = std::get<std::shared_ptr<ServiceRegistry>>(std::move(made));
    auto sm = connect_sm(registry);
    auto driver = std::dynamic_pointer_cast<services::nvdrv::DriverSession>(
        open_registry_service(fixture, sm, "nvdrv"));
    CHECK(driver != nullptr);
    if (!driver) return;
    driver->set_memory_mapper(fixture.memory->service_mapper());
    DriverHarness graphics{driver, fixture.memory};
    const auto nvmap = graphics.open("/dev/nvmap");
    const auto address_space = graphics.open("/dev/nvhost-as-gpu");
    const auto channel = graphics.open("/dev/nvhost-gpu");
    const auto check_reply = [&](const horizon::IpcReply& reply) {
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 32) == nv_success);
        if (reply.disposition != horizon::IpcDisposition::handled)
            std::fprintf(stderr, "GPU release regression: %s\n", driver->diagnostic());
    };
    std::array<std::byte, 0x28> initialize{};
    check_reply(graphics.ioctl(address_space, 0x40284109, initialize));
    std::array<std::byte, 4> binding{};
    write(binding, 0, channel, 4);
    check_reply(graphics.ioctl(address_space, 0x40044101, binding));
    write(binding, 0, nvmap, 4);
    check_reply(graphics.ioctl(channel, 0x40044801, binding));
    std::array<std::byte, 0x20> fifo{};
    write(fifo, 0, 16, 4);
    write(fifo, 4, 1, 4);
    check_reply(graphics.ioctl(channel, 0xc020481a, fifo));

    constexpr std::uint64_t gpu_base = 0x09000000;
    constexpr std::uint64_t surface_offset = 0x800;
    const auto header = [](std::uint32_t method, std::uint32_t count) {
        return (1u << 29) | (count << 16) | method;
    };
    // Reuse both virtual ranges with a new allocation identity. Each pass writes
    // a different solid color through the registry's real persistent renderer.
    for (unsigned pass = 0; pass != 2; ++pass) {
        std::array<std::byte, 8> create{};
        write(create, 0, test_backing_size, 4);
        check_reply(graphics.ioctl(nvmap, nvmap_create, create));
        const auto handle = read_u32(create, 4);
        std::array<std::byte, 0x20> allocate{};
        write(allocate, 0, handle, 4);
        write(allocate, 12, 0x1000, 4);
        write(allocate, 24, test_backing_address, 8);
        check_reply(graphics.ioctl(nvmap, nvmap_allocate, allocate));
        std::array<std::byte, 0x28> mapping{};
        write(mapping, 0, 1, 4);
        write(mapping, 8, handle, 4);
        write(mapping, 12, 0x1000, 4);
        write(mapping, 24, test_backing_size, 8);
        write(mapping, 32, gpu_base, 8);
        check_reply(graphics.ioctl(address_space, 0xc0284106, mapping));
        const std::array words{
            header(0, 1), 0xb197u,
            header(0x200, 8), 0u, std::uint32_t(gpu_base + surface_offset),
            32u, 8u, 0xd5u, 0x1000u, 1u, 0x1000u,
            header(0x3fd, 2), 8u << 16, 8u << 16,
            header(0x360, 4), pass == 0 ? 0x3f800000u : 0u,
            pass == 0 ? 0u : 0x3f800000u, 0u, 0x3f800000u,
            header(0x674, 1), 0x3cu};
        std::array<std::byte, words.size() * 4> commands{};
        for (std::size_t index = 0; index < words.size(); ++index)
            write(commands, index * 4, words[index], 4);
        graphics.copy_to(test_backing_address, commands);
        const std::array<std::byte, 4> nop{};
        graphics.copy_to(test_backing_address + 0x400, nop);
        std::array<std::byte, 8 * 8 * 4> before{};
        graphics.copy_from(test_backing_address + surface_offset, before);
        std::array<std::byte, 8> entry{};
        write(entry, 0, gpu_base | (std::uint64_t{words.size()} << 42), 8);
        std::array<std::byte, 0x18> submit{};
        write(submit, 8, 1, 4);
        check_reply(graphics.ioctl(channel, 0xc018481b, submit, entry));

        // The runtime guard must forward read-only command coherence. Fetching
        // clean commands must not flush the earlier clear's unrelated output.
        write(entry, 0, (gpu_base + 0x400) | (std::uint64_t{1} << 42), 8);
        check_reply(graphics.ioctl(channel, 0xc018481b, submit, entry));
        std::array<std::byte, 8 * 8 * 4> after_fetch{};
        graphics.copy_from(test_backing_address + surface_offset, after_fetch);
        CHECK(after_fetch == before);

        std::array<std::byte, 8> unmap{};
        write(unmap, 0, gpu_base, 8);
        check_reply(graphics.ioctl(address_space, 0xc0084105, unmap));
        std::array<std::byte, 0x18> free{};
        write(free, 0, handle, 4);
        check_reply(graphics.ioctl(nvmap, 0xc0180105, free));
        CHECK(read_u32(free, 8) == std::uint32_t(test_backing_address));
        CHECK(read_u32(free, 12) == std::uint32_t(test_backing_address >> 32));
        CHECK(read_u32(free, 16) == test_backing_size);
        // A successful Free must release the renderer's old allocation pin,
        // not merely preserve pixels while retaining the backing indefinitely.
        CHECK(fixture.memory->unmap_calls == pass + 1);
        // The worker uses retained native memory and guest fence ownership;
        // the synchronous guest-start exclusion is not worker-side exclusion.
        CHECK(acquisitions == 0 && releases == 0);
        CHECK(registry->statistics().completed_clears == pass + 1);
        CHECK(registry->statistics().completed_draws == 0);
        CHECK(!registry->failure());
        std::array<std::byte, 8 * 8 * 4> pixels{};
        graphics.copy_from(test_backing_address + surface_offset, pixels);
        for (std::size_t at = 0; at < pixels.size(); at += 4) {
            CHECK(pixels[at] == (pass == 0 ? std::byte{0xff} : std::byte{}));
            CHECK(pixels[at + 1] == (pass == 0 ? std::byte{} : std::byte{0xff}));
            CHECK(pixels[at + 2] == std::byte{});
            CHECK(pixels[at + 3] == std::byte{0xff});
        }
    }
    CHECK(std::holds_alternative<std::monostate>(registry->drain(std::nullopt)));
}

void test_registry_presentation_counts_follow_vi_release() {
    Fixture fixture;
    auto registry = fixture.create();
    auto sm = connect_sm(registry);
    auto driver = std::dynamic_pointer_cast<services::nvdrv::DriverSession>(
        open_registry_service(fixture, sm, "nvdrv"));
    auto vi = open_registry_service(fixture, sm, "vi:u");
    CHECK(driver != nullptr && vi != nullptr);
    if (!driver || !vi) return;

    DriverHarness graphics{driver, fixture.memory};
    const auto export_id = graphics.export_backing();
    write_authored_surface(*fixture.memory);

    // The registry wraps both service transfers and the later session clone.
    auto display = transferred_session(vi, service_command(1));
    auto relay = transferred_session(display, service_command(100));
    auto clone = clone_session(relay);
    CHECK(clone != nullptr);
    if (!clone) return;
    CHECK(set_preallocated_buffer(relay, export_id).disposition ==
          horizon::IpcDisposition::handled);

    auto dequeued = dequeue_buffer(clone);
    CHECK(dequeued.first.disposition == horizon::IpcDisposition::handled);
    CHECK(dequeued.second == 0);
    CHECK(request_buffer(clone).disposition == horizon::IpcDisposition::handled);
    {
        const std::lock_guard lock(fixture.presenter->mutex);
        fixture.presenter->entered = false;
        fixture.presenter->block = true;
        fixture.presenter->fail = false;
    }
    auto first = std::async(std::launch::async, [&] { return queue_buffer(clone); });
    CHECK(fixture.presenter->wait_until_entered());
    auto statistics = registry->statistics();
    CHECK(statistics.queued_frames == 1);
    CHECK(statistics.completed_frames == 0);
    fixture.presenter->release();
    CHECK(first.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    CHECK(first.get().disposition == horizon::IpcDisposition::handled);
    statistics = registry->statistics();
    CHECK(statistics.completed_frames == 1);

    // A fresh dequeue proves the successful request released the slot.
    dequeued = dequeue_buffer(clone);
    CHECK(dequeued.first.disposition == horizon::IpcDisposition::handled);
    CHECK(dequeued.second == 0);
    CHECK(request_buffer(clone).disposition == horizon::IpcDisposition::handled);
    const auto copy_failure = queue_buffer(clone, true);
    CHECK(copy_failure.disposition == horizon::IpcDisposition::unsupported);
    statistics = registry->statistics();
    CHECK(statistics.queued_frames == 2);
    CHECK(statistics.completed_frames == 2);

    // The output copy failed after release, so the same slot is available.
    dequeued = dequeue_buffer(clone);
    CHECK(dequeued.first.disposition == horizon::IpcDisposition::handled);
    CHECK(dequeued.second == 0);
    CHECK(request_buffer(clone).disposition == horizon::IpcDisposition::handled);
    {
        const std::lock_guard lock(fixture.presenter->mutex);
        fixture.presenter->entered = false;
        fixture.presenter->fail = true;
    }
    const auto presentation_failure = queue_buffer(clone);
    CHECK(presentation_failure.disposition == horizon::IpcDisposition::unsupported);
    statistics = registry->statistics();
    CHECK(statistics.queued_frames == 3);
    CHECK(statistics.completed_frames == 2);

    // Presentation failure keeps the slot ACQUIRED and therefore unavailable.
    CHECK(set_preallocated_buffer(relay, export_id).disposition ==
          horizon::IpcDisposition::unsupported);
    dequeued = dequeue_buffer(clone);
    CHECK(dequeued.first.disposition == horizon::IpcDisposition::unsupported);
    fixture.presenter->release();
    registry->request_stop();
    CHECK(std::holds_alternative<std::monostate>(registry->drain(std::nullopt)));
}

// A native-window present (the NVN driver's in-process producer) has no
// QueueBuffer request, yet the presenter consuming it completes the frame.
void test_registry_counts_native_window_presents() {
    Fixture fixture;
    std::shared_ptr<services::vi::DisplaySession> display;
    auto dependencies = fixture.dependencies(profile());
    dependencies.observe_display =
        [&](const std::shared_ptr<services::vi::DisplaySession>& created) {
            display = created;
        };
    auto made = create_service_registry(std::move(dependencies));
    CHECK(std::holds_alternative<std::shared_ptr<ServiceRegistry>>(made));
    auto* created = std::get_if<std::shared_ptr<ServiceRegistry>>(&made);
    if (!created) return;
    auto registry = *created;
    auto sm = connect_sm(registry);
    auto vi = open_registry_service(fixture, sm, "vi:u");
    CHECK(vi != nullptr && display != nullptr);
    if (!vi || !display) return;

    int native_window = 0;
    CHECK(display->bind_native_window(&native_window));
    auto producer = services::vi::find_native_window(&native_window);
    CHECK(producer != nullptr);
    if (!producer) return;
    std::array<services::vi::NativeTexture, 2> textures;
    for (std::size_t index = 0; index < textures.size(); ++index) {
        textures[index].serial = index + 1;
        textures[index].width = 4;
        textures[index].height = 4;
        textures[index].read_pixels = [](std::array<std::int32_t, 4>)
            -> graphics::Rgba8ReadbackResult {
            graphics::Rgba8Image image;
            image.width = 4;
            image.height = 4;
            image.pixels.assign(4 * 4 * 4, std::byte{0x40});
            return image;
        };
    }
    CHECK(producer->register_textures(textures));

    for (std::uint64_t frame = 1; frame <= 3; ++frame) {
        const auto slot = producer->dequeue();
        CHECK(slot.slot >= 0);
        const auto result = producer->present(
            slot.slot, 0, [](std::uint64_t) { return true; }, {}, 0);
        CHECK(result.completed);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (registry->statistics().completed_frames < frame &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto statistics = registry->statistics();
        CHECK(statistics.queued_frames == frame);
        CHECK(statistics.completed_frames == frame);
        CHECK(fixture.presenter->presented == frame);
    }
    CHECK(producer->unregister_textures());
    services::vi::unbind_native_window(&native_window);
    registry->request_stop();
    CHECK(std::holds_alternative<std::monostate>(registry->drain(std::nullopt)));
}

void test_acl_denial_precedes_registry_lookup() {
    Fixture fixture;
    auto registry = fixture.create(profile(service_acl("lm")));
    auto sm = connect_sm(registry);
    const auto denied = lookup(sm, "nvdrv");
    CHECK(denied.disposition == horizon::IpcDisposition::handled);
    CHECK(read_u32(denied.bytes, 16) == 0x1015);
    CHECK(fixture.handles->size() == 0);
    CHECK(!registry->failure().has_value());
}

void test_file_system_permissions_fail_closed() {
    Fixture fixture;
    // A partial permission mask is admitted; it is not enforced per command.
    auto partial = profile();
    partial.permissions.file_system_access_control = normalized_fs_access(1);
    auto made = create_service_registry(fixture.dependencies(std::move(partial)));
    const auto* failure = std::get_if<LaunchFailure>(&made);
    CHECK(failure == nullptr);

    auto malformed = profile();
    malformed.permissions.file_system_access_control.resize(0x18);
    made = create_service_registry(fixture.dependencies(std::move(malformed)));
    failure = std::get_if<LaunchFailure>(&made);
    CHECK(failure && failure->code == LaunchError::unsupported);

    // A process with no fsp-srv ACL never receives an FS session, so its FS
    // permission bytes are not interpreted by this registry.
    auto without_fs = profile(service_acl("lm"));
    without_fs.permissions.file_system_access_control.clear();
    CHECK(std::holds_alternative<std::shared_ptr<ServiceRegistry>>(
        create_service_registry(fixture.dependencies(std::move(without_fs)))));
}

void test_synthetic_service_acl_is_validated() {
    Fixture fixture;
    auto malformed = profile({std::byte{7}, std::byte{'n'}, std::byte{'v'}});
    auto made = create_service_registry(fixture.dependencies(std::move(malformed)));
    const auto* failure = std::get_if<LaunchFailure>(&made);
    CHECK(failure && failure->stage == LaunchStage::services);
    CHECK(failure && failure->code == LaunchError::invalid_input);
    CHECK(failure && failure->operation == "service_registry.service_access");

    auto oversized = profile(std::vector<std::byte>(0x201, std::byte{}));
    made = create_service_registry(fixture.dependencies(std::move(oversized)));
    failure = std::get_if<LaunchFailure>(&made);
    CHECK(failure && failure->code == LaunchError::invalid_input);
}

void test_shader_package_failure_is_structured() {
    Fixture fixture;
    auto dependencies = fixture.dependencies(profile(service_acl("lm")));
    dependencies.shader_package_directory =
        "service-registry-test-missing-shader-package";
    const auto made = create_service_registry(std::move(dependencies));
    const auto* failure = std::get_if<LaunchFailure>(&made);
    CHECK(failure && failure->stage == LaunchStage::graphics);
    CHECK(failure && failure->operation == "service_registry.shader_package");
    CHECK(failure && !failure->detail.empty());
}

// A package that opens while the guest image loads reports its failure the same
// way as one the registry opens itself, and no directory means nothing to wait for.
void test_background_shader_package_failure_is_structured() {
    const std::filesystem::path missing = "service-registry-test-missing-background-package";
    Fixture fixture;
    auto dependencies = fixture.dependencies(profile(service_acl("lm")));
    dependencies.shader_package_directory = missing;
    dependencies.pending_shader_package = start_shader_package_open(missing);
    const auto made = create_service_registry(std::move(dependencies));
    const auto* failure = std::get_if<LaunchFailure>(&made);
    CHECK(failure && failure->stage == LaunchStage::graphics);
    CHECK(failure && failure->operation == "service_registry.shader_package");
    CHECK(failure && !failure->detail.empty());
    CHECK(start_shader_package_open({}) == nullptr);
}

void test_supported_service_inventory_and_real_handles() {
    Fixture fixture;
    auto registry = fixture.create();
    auto sm = connect_sm(registry);
    constexpr std::array names{
        "lm", "acc:u0", "aoc:u", "ectx:aw", "pctl", "prepo:u", "time:u",
        "fsp-srv", "ro", "nvdrv", "nfp:user", "audren:u", "apm",
        "appletOE", "hid",
    };
    for (std::string_view name : names) {
        const auto reply = lookup(sm, name);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 16) == 0);
        const auto handle = read_u32(reply.bytes, 12);
        CHECK(handle != 0);
        CHECK(fixture.handles->lookup_as<horizon::ClientSession>(handle) != nullptr);
        CHECK(fixture.handles->close(handle));
    }

    std::size_t presentation_connections = 0;
    for (std::string_view name : {"vi:u", "vi:m", "vi:s"}) {
        const auto reply = lookup(sm, name);
        if (reply.disposition == horizon::IpcDisposition::handled) {
            const auto handle = read_u32(reply.bytes, 12);
            CHECK(handle != 0 && fixture.handles->close(handle));
            ++presentation_connections;
        } else {
            CHECK(reply.disposition == horizon::IpcDisposition::unsupported);
            const auto unavailable = registry->failure();
            CHECK(unavailable && unavailable->stage == LaunchStage::presentation);
        }
    }
    CHECK(presentation_connections == 0 || presentation_connections == 3);
    const auto stats = registry->statistics();
    CHECK(stats.named_port_connections == 1);
    CHECK(stats.service_connections == names.size() + presentation_connections);
}

void test_nvn_failure_without_presentation() {
    class FailedReplay final : public nvn::ReplayBackend {
    public:
        std::string error;
        std::uint64_t submit(std::span<const nvn::api::NVNcommandHandle>) override { return 0; }
        void flush() override {}
        std::uint64_t completed_fence() const override { return 0; }
        bool wait_fence(std::uint64_t, std::uint64_t) override { return false; }
        std::string fatal_error_detail() const override { return error; }
    } replay;
    Fixture fixture;
    auto dependencies = fixture.dependencies(profile());
    dependencies.nvn_driver = true;
    auto made = create_service_registry(std::move(dependencies));
    CHECK(std::holds_alternative<std::shared_ptr<ServiceRegistry>>(made));
    if (auto* registry = std::get_if<std::shared_ptr<ServiceRegistry>>(&made)) {
        const auto saved = nvn::backends();
        nvn::set_backends({nullptr, &replay});
        CHECK(!(*registry)->failure());
        replay.error = "NVN replay abandoned submission 7: synthetic unsupported attachment view";
        const auto failure = (*registry)->failure();
        CHECK(failure && failure->stage == LaunchStage::graphics);
        CHECK(failure && failure->operation == "service_registry.nvn_replay");
        CHECK(failure && failure->detail == replay.error);
        CHECK((*registry)->statistics().completed_frames == 0);
        nvn::set_backends(saved);
    }
}

void test_unknown_authorized_service_latches_failure() {
    Fixture fixture;
    auto registry = fixture.create();
    auto sm = connect_sm(registry);
    const auto reply = lookup(sm, "unknown");
    CHECK(reply.disposition == horizon::IpcDisposition::unsupported);
    const auto failure = registry->failure();
    CHECK(failure && failure->code == LaunchError::unsupported);
    CHECK(failure && failure->operation == "service_registry.connect");
}

void test_stop_drains_and_refuses_new_work() {
    Fixture fixture;
    auto registry = fixture.create(profile(service_acl("lm")));
    CHECK(std::holds_alternative<std::monostate>(registry->drain(std::nullopt)));
    CHECK(fixture.presenter->drains == 1);
    const auto connection = registry->connect_named_port("sm:");
    CHECK(connection.disposition == horizon::IpcDisposition::cancelled);
    CHECK(!registry->failure().has_value());
}

void test_presentation_observer_preserves_authority_lifetime() {
    struct Clock final : PresentationClock {
        bool enabled = false, readable = false;
        unsigned observations = 0;
        std::shared_ptr<GuestClock> clock() const noexcept override { return {}; }
        PresentationClockConfiguration configuration() const noexcept override { return {}; }
        LaunchResult<PresentationSchedule> schedule(std::optional<std::uint64_t>, std::int32_t) const override {
            return PresentationSchedule{};
        }
        LaunchResult<PresentationSchedule> next_refresh() const override { return PresentationSchedule{}; }
        bool observes_presents() const noexcept override { return enabled; }
        LaunchStatus observe_present(std::uint64_t, std::int32_t, const PresentationMemoryReader& read) override {
            ++observations;
            std::array<std::byte, 4> bytes{};
            readable = read(test_memory_base, bytes);
            return std::monostate{};
        }
    };
    auto clock = std::make_shared<Clock>();
    auto memory = std::make_shared<TestMemory>();
    std::weak_ptr<GuestMemory> memory_lifetime = memory;
    std::weak_ptr<PresentationClock> clock_lifetime = clock;
    unsigned calls = 0;
    services::GraphicsPresenter original = [&](const services::GraphicsPresentation&) {
        ++calls;
        return services::GraphicsPresentationResult{true, {}};
    };
    auto unobserved = attach_presentation_observer(original, clock, memory);
    CHECK(unobserved.target_type() == original.target_type());
    CHECK(memory.use_count() == 1 && clock.use_count() == 1);
    CHECK(unobserved({}).completed && calls == 1 && clock->observations == 0);
    clock->enabled = true;
    auto observed = attach_presentation_observer(original, clock, memory);
    CHECK(memory.use_count() == 1);
    CHECK(observed({}).completed && calls == 2 && clock->observations == 1 && clock->readable);
    CHECK(memory->observation_reads == 1);
    CHECK(memory.use_count() == 1);
    memory.reset();
    CHECK(memory_lifetime.expired());
    CHECK(observed({}).completed && calls == 3 && clock->observations == 2 && !clock->readable);
    observed = {};
    clock.reset();
    CHECK(clock_lifetime.expired());
    CHECK(unobserved({}).completed && calls == 4);
}

} // namespace

void test_switch2_service_boundary() {
    Fixture fixture;
    auto admitted = profile(service_acl("nvdrv"));
    admitted.process.platform = armigo::GuestPlatform::switch2;
    admitted.process.baseline = armigo::horizon::Baseline::horizon_23_0_1;
    auto registry = fixture.create(std::move(admitted));
    CHECK(registry != nullptr);
    if (!registry) return;
    const auto connection = registry->connect_named_port("sm:");
    CHECK(connection.disposition == horizon::IpcDisposition::handled && connection.result == 0);
    const auto sm = std::dynamic_pointer_cast<services::sm::ServiceManagerSession>(connection.session);
    CHECK(sm != nullptr);
    if (!sm) { registry->request_stop(); return; }
    const auto handles_before = fixture.handles->size();
    auto reply = lookup(sm, "nvdrv");
    CHECK(reply.disposition == horizon::IpcDisposition::handled && read_u32(reply.bytes, 16) == 0x415);
    CHECK(!registry->failure());
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        reply = sm->request(register_client(), 71);
        CHECK(reply.disposition == horizon::IpcDisposition::handled && read_u32(reply.bytes, 8) == 0);
    }
    reply = lookup(sm, "denied");
    CHECK(reply.disposition == horizon::IpcDisposition::handled && read_u32(reply.bytes, 16) == 0x1015);
    CHECK(!registry->failure());
    reply = lookup(sm, "");
    CHECK(reply.disposition == horizon::IpcDisposition::handled && read_u32(reply.bytes, 16) == 0xc15);
    CHECK(!registry->failure());
    reply = lookup(sm, "nvdrv");
    CHECK(reply.disposition == horizon::IpcDisposition::unsupported);
    CHECK(fixture.handles->size() == handles_before);
    const auto failed = registry->failure();
    CHECK(failed && failed->operation == "service_registry.switch2_service" &&
          failed->detail.find("nvdrv") != std::string::npos);
    CHECK(sm->close_server().transport_result == 0);
    const auto closed = sm->request(register_client(), 71);
    CHECK(closed.disposition == horizon::IpcDisposition::handled && closed.transport_result == 0xf601);
    registry->request_stop();
    CHECK(registry->connect_named_port("sm:").disposition == horizon::IpcDisposition::cancelled);

    Fixture other_fixture;
    auto other_profile = profile();
    other_profile.process.platform = armigo::GuestPlatform::switch2;
    other_profile.process.baseline = armigo::horizon::Baseline::horizon_23_0_1;
    const auto other = other_fixture.create(std::move(other_profile));
    CHECK(other != nullptr);
    if (other) {
        CHECK(other->connect_named_port("other").disposition == horizon::IpcDisposition::unsupported);
        const auto unknown = other->failure();
        CHECK(unknown && unknown->operation == "service_registry.switch2_named_port");
        other->request_stop();
    }
}

void test_switch2_applet_admission() {
    Fixture fixture;
    auto admitted = profile(service_acl("appletOE"));
    admitted.process.platform = armigo::GuestPlatform::switch2;
    admitted.process.baseline = horizon::Baseline::horizon_23_0_1;
    auto registry = fixture.create(std::move(admitted));
    CHECK(registry != nullptr);
    if (!registry) return;
    const auto connection = registry->connect_named_port("sm:");
    const auto sm = std::dynamic_pointer_cast<services::sm::ServiceManagerSession>(connection.session);
    CHECK(sm != nullptr);
    if (!sm) { registry->request_stop(); return; }
    CHECK(sm->request(register_client(), 71).disposition == horizon::IpcDisposition::handled);
    const auto result = lookup(sm, "appletOE");
    CHECK(result.disposition == horizon::IpcDisposition::handled && read_u32(result.bytes, 16) == 0);
    const auto applet = fixture.handles->lookup_as<services::am::AppletServiceSession>(read_u32(result.bytes, 12));
    CHECK(applet != nullptr);
    if (applet) {
        horizon::IpcMessage request{};
        write(request, 0, 7, 4); write(request, 4, 8, 4);
        write(request, 16, 0x49434653, 4); write(request, 24, 3, 4);
        const auto reply = applet->request(request, 71);
        CHECK(reply.disposition == horizon::IpcDisposition::handled && reply.size == 44);
        CHECK(read_u32(reply.bytes, 28) == 0xec6be3ff && read_u32(reply.bytes, 32) == 0);
        CHECK(applet->close_server().transport_result == 0);
        CHECK(applet->request(request, 71).transport_result == 0xf601);
    }
    CHECK(!registry->failure());
    registry->request_stop();
}

void test_switch2_log_manager_admission() {
    Fixture fixture;
    auto admitted = profile(service_acl("lm"));
    admitted.process.platform = armigo::GuestPlatform::switch2;
    admitted.process.baseline = armigo::horizon::Baseline::horizon_23_0_1;
    const auto registry = fixture.create(std::move(admitted));
    CHECK(registry != nullptr);
    if (!registry) return;
    const auto sm = connect_sm(registry);
    const auto before = fixture.handles->size();
    auto reply = lookup(sm, "lm");
    CHECK(reply.disposition == horizon::IpcDisposition::handled && read_u32(reply.bytes, 16) == 0);
    const auto lm = fixture.handles->lookup_as<services::lm::LogManagerSession>(read_u32(reply.bytes, 12));
    CHECK(lm && fixture.handles->size() == before + 1);
    if (!lm) { registry->request_stop(); return; }
    horizon::IpcMessage request{};
    write(request, 0, 5, 4); write(request, 4, 8, 4); write(request, 16, sfci, 4);
    write(request, 24, 3, 4);
    reply = lm->request(request, 71);
    CHECK(reply.disposition == horizon::IpcDisposition::handled && read_u32(reply.bytes, 32) == 0x400);
    write(request, 24, 0, 4);
    reply = lm->request(request, 71);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    const auto root = read_u32(reply.bytes, 32);
    request = {};
    write(request, 0, 6, 4); write(request, 4, 0x8000000e, 4); write(request, 8, 1, 4);
    write(request, 32, 0x00180001, 4); write(request, 36, root, 4);
    write(request, 48, sfci, 4); write(request, 52, 1, 4);
    reply = lm->request(request, 71);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    const auto logger = read_u32(reply.bytes, 48);
    CHECK(lm->logger_process_id(logger) == 71);
    request = {};
    write(request, 0, 0x00110006, 4); write(request, 4, 12, 4);
    write(request, 32, 0x00100001, 4); write(request, 36, logger, 4);
    write(request, 48, sfci, 4);
    reply = lm->request(request, 71);
    CHECK(reply.disposition == horizon::IpcDisposition::handled && reply.size == 56 &&
          read_u32(reply.bytes, 4) == 12 && read_u32(reply.bytes, 44) == 0x3e81dad2);
    CHECK(!registry->failure() && fixture.handles->size() == before + 1);
    CHECK(lm->close_server().transport_result == 0 && lm->domain_object_count() == 0);
    CHECK(lm->request(request, 71).transport_result == 0xf601);
    registry->request_stop();
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--legacy-memory-release") {
        test_registry_legacy_memory_release();
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--gpu-memory-release") {
        test_registry_gpu_memory_release();
        return failures == 0 ? 0 : 1;
    }
    test_switch2_service_boundary();
    test_switch2_applet_admission();
    test_switch2_log_manager_admission();
    test_presentation_observer_preserves_authority_lifetime();
    test_registry_presentation_counts_follow_vi_release();
    test_registry_counts_native_window_presents();
    test_operation_mode_reaches_registered_services();
    test_acl_denial_precedes_registry_lookup();
    test_file_system_permissions_fail_closed();
    test_synthetic_service_acl_is_validated();
    test_shader_package_failure_is_structured();
    test_background_shader_package_failure_is_structured();
    test_supported_service_inventory_and_real_handles();
    test_nvn_failure_without_presentation();
    test_unknown_authorized_service_latches_failure();
    test_stop_drains_and_refuses_new_work();
    if (failures != 0) return 1;
    std::puts("service registry checks passed");
    return 0;
}
