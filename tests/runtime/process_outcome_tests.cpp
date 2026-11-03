// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/instruction.h>
#include <armigo/runtime/preparation_format.h>
#include <armigo/runtime/process_runtime.h>
#include <armigo/services/nvdrv.h>
#include <armigo/services/service_manager_session.h>

#include "../../src/platform/process_memory.h"
#include "../../src/runtime/process_runtime_test_access.h"
#include "../loader/npdm_fixture.h"
#include "preparation_fixture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

extern "C" {
extern const std::byte armigo_process_outcome_begin[];
extern const std::byte armigo_process_outcome_end[];
extern const std::byte armigo_process_outcome_main[];
extern const std::byte armigo_process_outcome_main_end[];
extern const std::byte armigo_process_outcome_heap[];
}

namespace {
using namespace armigo;
using namespace armigo::runtime;
using namespace std::chrono_literals;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

template<class T>
const T* value(const LaunchResult<T>& result) {
    return std::get_if<T>(&result);
}

template<class T>
const LaunchFailure* failure(const LaunchResult<T>& result) {
    return std::get_if<LaunchFailure>(&result);
}

const LaunchFailure* failure(const RunResult& result) {
    return std::get_if<LaunchFailure>(&result.outcome);
}

bool ok(const LaunchStatus& status) {
    return std::holds_alternative<std::monostate>(status);
}

void write(std::span<std::byte> bytes, std::size_t offset,
           std::uint64_t number, std::size_t width) {
    for (std::size_t index = 0; index < width; ++index)
        bytes[offset + index] = static_cast<std::byte>((number >> (8 * index)) & 0xff);
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t result = 0;
    for (std::size_t index = 0; index < 4; ++index)
        result |= std::uint32_t{std::to_integer<std::uint8_t>(bytes[offset + index])}
                  << (8 * index);
    return result;
}

template<class Predicate>
bool wait_until(Predicate&& predicate, std::chrono::milliseconds timeout = 2s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (predicate()) return true;
        std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

struct TemporaryDirectory {
    std::filesystem::path path;

    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() /
                        "armigo-process-outcome-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        if (char* created = ::mkdtemp(writable.data())) path = created;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

bool write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return stream.good();
}

std::optional<std::uint64_t> probe_heap_address() {
    auto reserved = platform::ProcessMemoryRegion::reserve(64ull << 20);
    auto* region = std::get_if<platform::ProcessMemoryRegion>(&reserved);
    if (!region) return std::nullopt;
    return region->address() + (32ull << 20);
}

bool native_writable(std::uint64_t address) {
#if defined(__APPLE__)
    mach_vm_address_t region_address = address;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(
        mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) ::mach_port_deallocate(mach_task_self(), object);
    return result == KERN_SUCCESS && region_address <= address &&
           region_size > address - region_address &&
           (info.protection & (VM_PROT_READ | VM_PROT_WRITE)) ==
               (VM_PROT_READ | VM_PROT_WRITE);
#else
    (void)address;
    return false;
#endif
}

bool wait_for_word(std::uint64_t address, std::uint32_t expected) {
    return wait_until([=] {
        return native_writable(address) &&
               __atomic_load_n(reinterpret_cast<const std::uint32_t*>(address),
                               __ATOMIC_ACQUIRE) == expected;
    });
}

void store_word(std::uint64_t address, std::uint32_t number) {
    __atomic_store_n(reinterpret_cast<std::uint32_t*>(address), number,
                     __ATOMIC_RELEASE);
}

std::size_t fixture_offset(const std::byte* symbol) {
    return 0x100 + static_cast<std::size_t>(symbol - armigo_process_outcome_begin);
}

std::uint32_t move_wide(bool keep, unsigned destination, unsigned half,
                        std::uint16_t immediate) {
    return (keep ? 0xf2800000u : 0xd2800000u) |
           (half << 21) | (std::uint32_t{immediate} << 5) | destination;
}

loader::ExeFsDirectoryInput process_input(std::uint64_t heap) {
    runtime::test::ModuleSpec module;
    module.name = "main";
    const auto fixture_size = static_cast<std::size_t>(
        armigo_process_outcome_end - armigo_process_outcome_begin);
    module.text.resize(std::max<std::size_t>(runtime::test::rodata_offset,
                                             fixture_offset(armigo_process_outcome_end)));
    module.data.resize(0x20000 - runtime::test::data_offset);
    std::memcpy(module.text.data() + 0x100, armigo_process_outcome_begin,
                fixture_size);
    const auto entry = fixture_offset(armigo_process_outcome_main);
    const auto end = fixture_offset(armigo_process_outcome_main_end);
    module.symbols = {{"_start", entry, end - entry, true}};
    const auto heap_instruction = fixture_offset(armigo_process_outcome_heap);
    for (unsigned half = 0; half != 4; ++half) {
        runtime::test::put_word(module, heap_instruction + half * 4,
            move_wide(half != 0, 19, half,
                      static_cast<std::uint16_t>(heap >> (half * 16))));
    }
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", runtime::test::build_module(std::move(module))});
    return input;
}

LaunchProfile synthetic_profile(std::uint64_t heap) {
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.is_64_bit = true;
    profile.process.allowed_core_mask = 1;
    profile.process.heap_address = heap;
    profile.process.heap_size = 8ull << 20;
    profile.process.base_alias_size = 2ull << 20;
    profile.process.stack_size = 2ull << 20;
    profile.permissions.supervisor_calls = {0x01, 0x07, 0x1f, 0x21};
    profile.permissions.service_access_control =
        npdm_fixture::service_access_control({{"nvdrv", false}, {"vi:u", false}});
    profile.permissions.handle_table_capacity = 128;
    profile.permissions.highest_priority = 16;
    profile.permissions.lowest_priority = 63;
    profile.main_thread.priority = 44;
    profile.main_thread.ideal_core = 0;
    profile.main_thread.stack_size = 64 * 1024;
    return profile;
}

struct Fixture {
    TemporaryDirectory temporary;
    std::filesystem::path exefs = temporary.path / "exefs";
    std::filesystem::path romfs = temporary.path / "romfs";
    std::filesystem::path prepared = temporary.path / "prepared";
    loader::ExeFsDirectoryInput input;
    LaunchDescription description;

    explicit Fixture(std::uint64_t heap) : input(process_input(heap)) {
        std::filesystem::create_directories(exefs);
        std::filesystem::create_directories(romfs);
        CHECK(!temporary.path.empty());
        CHECK(write_file(exefs / "main", input.modules.front().bytes));
        const PreparationOptions options{
            .required_preparation_identity = installed_preparation_identity(),
            .max_artifact_bytes = default_prepared_artifact_limit,
            .mode = PreparationMode::synthetic_test,
        };
        const auto artifacts = prepare_modules(input, options);
        CHECK(value(artifacts) != nullptr);
        if (value(artifacts)) CHECK(ok(write_prepared_modules(prepared, *value(artifacts))));
        description.exefs_directory = exefs;
        description.romfs_directory = romfs;
        description.prepared_directory = prepared;
        description.presentation.mode = platform::PresentationMode::headless;
        description.synthetic_profile = synthetic_profile(heap);
    }
};

class TestPresenter final : public platform::Presenter {
public:
    platform::PresentationMode mode() const noexcept override {
        return platform::PresentationMode::headless;
    }
    platform::PresentResult present(const platform::PresentationFrame& frame,
        std::optional<ClockInstant>, std::stop_token) override {
        std::unique_lock lock(mutex_);
        entered_ = true;
        changed_.notify_all();
        changed_.wait(lock, [this] { return !blocked_; });
        platform::PresentedFrame completed;
        completed.frame_number = frame.queued.frame_number;
        completed.readback = *frame.pixels;
        return completed;
    }
    platform::PresenterStatus drain(std::optional<ClockInstant>) override {
        return std::monostate{};
    }
    void block() {
        const std::lock_guard lock(mutex_);
        entered_ = false;
        blocked_ = true;
    }
    bool wait_entered() {
        std::unique_lock lock(mutex_);
        return changed_.wait_for(lock, 2s, [this] { return entered_; });
    }
    void release() {
        const std::lock_guard lock(mutex_);
        blocked_ = false;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool entered_ = false;
    bool blocked_ = false;
};

class SecondSampleGateClock final : public GuestClock {
public:
    explicit SecondSampleGateClock(std::shared_ptr<GuestClock> inner)
        : inner_(std::move(inner)) {}

    void arm_for_current_thread() {
        const std::lock_guard lock(mutex_);
        target_ = std::this_thread::get_id();
        samples_ = 0;
        entered_ = false;
        released_ = false;
        armed_ = true;
    }
    bool wait_entered() {
        std::unique_lock lock(mutex_);
        return changed_.wait_for(lock, 2s, [this] { return entered_; });
    }
    void release() {
        const std::lock_guard lock(mutex_);
        released_ = true;
        changed_.notify_all();
    }
    LaunchResult<ClockSample> sample() const override {
        {
            std::unique_lock lock(mutex_);
            if (armed_ && target_ == std::this_thread::get_id() && ++samples_ == 2) {
                entered_ = true;
                changed_.notify_all();
                changed_.wait(lock, [this] { return released_; });
            }
        }
        return inner_->sample();
    }
    execution::GuestCounter counter() noexcept override { return inner_->counter(); }
    LaunchResult<std::chrono::nanoseconds> ticks_to_nanoseconds(
        std::uint64_t ticks) const override {
        return inner_->ticks_to_nanoseconds(ticks);
    }
    LaunchResult<std::uint64_t> nanoseconds_to_ticks(
        std::chrono::nanoseconds duration) const override {
        return inner_->nanoseconds_to_ticks(duration);
    }
    LaunchResult<ClockWait> wait_until(ClockInstant deadline,
                                       std::stop_token stop) override {
        return inner_->wait_until(deadline, stop);
    }

private:
    std::shared_ptr<GuestClock> inner_;
    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    mutable std::thread::id target_;
    mutable unsigned samples_ = 0;
    mutable bool armed_ = false;
    mutable bool entered_ = false;
    mutable bool released_ = false;
};

RuntimeDependencies dependencies(const std::shared_ptr<TestPresenter>& presenter,
                                 const std::shared_ptr<GuestClock>& clock) {
    RuntimeDependencies result;
    result.memory_options.resource_limit_bytes = 32ull << 20;
    result.preparation_options = {
        .required_preparation_identity = installed_preparation_identity(),
        .max_artifact_bytes = default_prepared_artifact_limit,
    };
    result.clock = clock;
    const auto sample = clock->sample();
    CHECK(value(sample) != nullptr);
    if (value(sample)) {
        auto presentation_clock = create_presentation_clock(
            clock, {value(sample)->instant, 16'666'667ns});
        CHECK(value(presentation_clock) != nullptr);
        if (value(presentation_clock)) result.presentation_clock = *value(presentation_clock);
    }
    result.presenter = presenter;
    return result;
}

struct RuntimeHarness {
    std::unique_ptr<Fixture> fixture;
    std::unique_ptr<ProcessRuntime> runtime;
    ProcessRuntimeTestState state;
    std::uint64_t heap = 0;
};

std::optional<RuntimeHarness> create_runtime(
    const std::shared_ptr<TestPresenter>& presenter,
    const std::shared_ptr<GuestClock>& clock) {
    LaunchFailure last;
    for (unsigned attempt = 0; attempt != 8; ++attempt) {
        const auto heap = probe_heap_address();
        if (!heap) continue;
        auto fixture = std::make_unique<Fixture>(*heap);
        auto created = ProcessRuntime::create(
            fixture->description, dependencies(presenter, clock));
        if (value(created)) {
            RuntimeHarness result;
            result.fixture = std::move(fixture);
            result.runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
            result.state = ProcessRuntimeTestAccess::inspect(*result.runtime);
            result.heap = *heap;
            return result;
        }
        last = *failure(created);
        if (last.operation != "create_guest_memory reservations") break;
    }
    std::fprintf(stderr, "Outcome runtime creation failed: %s: %s\n",
                 last.operation.c_str(), last.detail.c_str());
    CHECK(false);
    return std::nullopt;
}

constexpr std::uint32_t sfci = 0x49434653;
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

horizon::IpcMessage driver_request(std::uint32_t command,
                                   std::uint32_t descriptor = 0,
                                   std::uint32_t operation = 0) {
    auto message = service_command(command);
    write(message, 4, 12, 4);
    write(message, 32, descriptor, 4);
    write(message, 36, operation, 4);
    return message;
}

horizon::IpcMessage register_client() {
    horizon::IpcMessage request{};
    write(request, 0, 0x8000000000000010ULL, 8);
    write(request, 8, 1, 4);
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
    const ProcessRuntimeTestState& state) {
    auto connection = state.services->connect_named_port("sm:");
    CHECK(connection.disposition == horizon::IpcDisposition::handled);
    auto sm = std::dynamic_pointer_cast<services::sm::ServiceManagerSession>(
        connection.session);
    CHECK(sm != nullptr);
    if (!sm) return {};
    const auto reply = sm->request(register_client(), state.process_id);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(read_u32(reply.bytes, 8) == 0);
    return sm;
}

std::shared_ptr<horizon::ClientSession> open_service(
    const ProcessRuntimeTestState& state,
    const std::shared_ptr<services::sm::ServiceManagerSession>& sm,
    std::string_view name) {
    const auto reply = sm->request(get_service(name), state.process_id);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(read_u32(reply.bytes, 16) == 0);
    const auto handle = read_u32(reply.bytes, 12);
    auto session = state.handles->lookup_as<horizon::ClientSession>(handle);
    CHECK(session != nullptr);
    CHECK(state.handles->close(handle));
    return session;
}

struct DriverHarness {
    std::shared_ptr<services::nvdrv::DriverSession> driver;
    std::shared_ptr<GuestMemory> memory;
    std::uint64_t process_id = 0;
    std::uint64_t path_address = 0;
    std::uint64_t argument_address = 0;

    bool copy_to(std::uint64_t address, std::span<const std::byte> bytes) {
        return memory->ipc_buffers().write_memory(address, bytes);
    }
    bool copy_from(std::uint64_t address, std::span<std::byte> bytes) {
        return memory->ipc_buffers().read_memory(address, bytes);
    }
    std::uint32_t open(std::string_view path) {
        CHECK(copy_to(path_address, std::as_bytes(std::span(path.data(), path.size()))));
        auto buffers = memory->ipc_buffers();
        buffers.add({path_address, static_cast<std::uint32_t>(path.size()), true, false});
        const auto reply = driver->request_with_buffers(
            driver_request(0), process_id, buffers);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 36) == 0);
        return read_u32(reply.bytes, 32);
    }
    horizon::IpcReply ioctl(std::uint32_t descriptor, std::uint32_t operation,
                            std::span<std::byte> arguments) {
        CHECK(copy_to(argument_address, arguments));
        auto buffers = memory->ipc_buffers();
        buffers.add({argument_address, static_cast<std::uint32_t>(arguments.size()),
                     true, true});
        auto reply = driver->request_with_buffers(
            driver_request(1, descriptor, operation), process_id, buffers);
        CHECK(copy_from(argument_address, arguments));
        return reply;
    }
    std::uint32_t export_backing(std::uint64_t backing_address) {
        driver->set_memory_mapper(memory->service_mapper());
        const auto nvmap = open("/dev/nvmap");
        std::array<std::byte, 8> create{};
        write(create, 0, 0x1000, 4);
        auto reply = ioctl(nvmap, nvmap_create, create);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 32) == 0);
        const auto handle = read_u32(create, 4);
        CHECK(handle != 0);
        std::array<std::byte, 0x20> allocate{};
        write(allocate, 0, handle, 4);
        write(allocate, 12, 0x1000, 4);
        write(allocate, 24, backing_address, 8);
        reply = ioctl(nvmap, nvmap_allocate, allocate);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 32) == 0);
        std::array<std::byte, 8> exported{};
        write(exported, 4, handle, 4);
        reply = ioctl(nvmap, nvmap_get_id, exported);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(read_u32(reply.bytes, 32) == 0);
        return read_u32(exported, 0);
    }
};

struct BinderBuffers {
    std::array<std::byte, 0x200> input{};
    std::array<std::byte, 0x200> output{};

    horizon::GuestBuffers buffers(std::uint32_t input_size) {
        horizon::GuestBuffers result(
            [](void* raw, std::uint64_t address,
               std::span<std::byte> destination) noexcept {
                auto& self = *static_cast<BinderBuffers*>(raw);
                if (address != 0x1000 || destination.size() > self.input.size())
                    return false;
                std::memcpy(destination.data(), self.input.data(), destination.size());
                return true;
            },
            [](void* raw, std::uint64_t address,
               std::span<const std::byte> source) noexcept {
                auto& self = *static_cast<BinderBuffers*>(raw);
                if (address != 0x2000 || source.size() > self.output.size())
                    return false;
                std::memcpy(self.output.data(), source.data(), source.size());
                return true;
            }, this);
        result.add({0x1000, input_size, true, false});
        result.add({0x2000, static_cast<std::uint32_t>(output.size()), false, true});
        return result;
    }
    void set_word(std::size_t offset, std::uint32_t number) {
        std::memcpy(input.data() + offset, &number, sizeof number);
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
    horizon::IpcMessage message, std::uint64_t process_id) {
    auto reply = session->request(message, process_id);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    auto transferred = std::dynamic_pointer_cast<horizon::ClientSession>(
        reply.transferred[0]);
    CHECK(transferred != nullptr);
    return transferred;
}

std::array<std::byte, 0x16c> flattened_buffer(std::uint32_t export_id) {
    std::array<std::byte, 0x16c> flat{};
    write(flat, 0, 0x47424652, 4);
    write(flat, 4, 8, 4);
    write(flat, 8, 8, 4);
    write(flat, 0x0c, 8, 4);
    write(flat, 0x10, 1, 4);
    write(flat, 0x14, 0x800, 4);
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
    write(flat, 0x94, 0x200, 8);
    return flat;
}

void write_authored_surface(const std::shared_ptr<GuestMemory>& memory,
                            std::uint64_t address) {
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
    CHECK(memory->ipc_buffers().write_memory(address, surface));
}

horizon::IpcReply set_preallocated_buffer(
    const std::shared_ptr<horizon::ClientSession>& relay,
    std::uint64_t process_id, std::uint32_t export_id) {
    BinderBuffers data;
    const auto at = data.parcel(std::array<std::uint32_t, 4>{0, 1, 0x16c, 0});
    const auto flat = flattened_buffer(export_id);
    std::copy(flat.begin(), flat.end(), data.input.begin() + at + 16);
    const auto size = at + 16 + flat.size();
    data.set_word(0, static_cast<std::uint32_t>(size - 16));
    auto buffers = data.buffers(static_cast<std::uint32_t>(size));
    return relay->request_with_buffers(
        binder_transaction(14), process_id, buffers);
}

std::int32_t dequeue_buffer(const std::shared_ptr<horizon::ClientSession>& relay,
                            std::uint64_t process_id) {
    BinderBuffers data;
    const auto at = data.parcel(std::array<std::uint32_t, 5>{0, 8, 8, 1, 0});
    auto buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    const auto reply = relay->request_with_buffers(
        binder_transaction(3), process_id, buffers);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    return std::bit_cast<std::int32_t>(read_u32(data.output, 0x10));
}

horizon::IpcReply request_buffer(
    const std::shared_ptr<horizon::ClientSession>& relay,
    std::uint64_t process_id) {
    BinderBuffers data;
    const auto at = data.parcel(std::array<std::uint32_t, 1>{0});
    auto buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
    return relay->request_with_buffers(
        binder_transaction(1), process_id, buffers);
}

horizon::IpcReply queue_buffer(
    const std::shared_ptr<horizon::ClientSession>& relay,
    std::uint64_t process_id) {
    std::array<std::uint32_t, 24> arguments{};
    arguments[0] = 0;
    arguments[1] = 84;
    arguments[5] = 1;
    arguments[12] = 8;
    BinderBuffers data;
    const auto at = data.parcel(arguments);
    auto buffers = data.buffers(
        static_cast<std::uint32_t>(at + arguments.size() * 4));
    return relay->request_with_buffers(
        binder_transaction(7), process_id, buffers);
}

struct PreparedVi {
    std::shared_ptr<horizon::ClientSession> relay;
    horizon::Handle relay_handle = 0;
    BinderBuffers queue;
    std::uint32_t queue_size = 0;
};

std::optional<PreparedVi> prepare_vi(const ProcessRuntimeTestState& state,
                                     std::uint64_t heap) {
    auto sm = connect_sm(state);
    if (!sm) return std::nullopt;
    auto driver = std::dynamic_pointer_cast<services::nvdrv::DriverSession>(
        open_service(state, sm, "nvdrv"));
    auto vi = open_service(state, sm, "vi:u");
    CHECK(driver != nullptr && vi != nullptr);
    if (!driver || !vi) return std::nullopt;
    constexpr std::uint64_t path_offset = 0x4000;
    constexpr std::uint64_t argument_offset = 0x5000;
    constexpr std::uint64_t backing_offset = 0x10000;
    DriverHarness graphics{driver, state.memory, state.process_id,
                           heap + path_offset, heap + argument_offset};
    const auto export_id = graphics.export_backing(heap + backing_offset);
    CHECK(export_id != 0);
    write_authored_surface(state.memory, heap + backing_offset);
    auto display = transferred_session(vi, service_command(1), state.process_id);
    auto relay = transferred_session(display, service_command(100), state.process_id);
    horizon::IpcMessage clone{};
    write(clone, 0, 7, 4);
    write(clone, 4, 8, 4);
    write(clone, 16, sfci, 4);
    write(clone, 24, 2, 4);
    auto cloned = transferred_session(relay, clone, state.process_id);
    CHECK(set_preallocated_buffer(relay, state.process_id, export_id).disposition ==
          horizon::IpcDisposition::handled);
    CHECK(dequeue_buffer(cloned, state.process_id) == 0);
    CHECK(request_buffer(cloned, state.process_id).disposition ==
          horizon::IpcDisposition::handled);
    const auto handle = state.handles->insert(cloned);
    CHECK(handle.has_value());
    if (!handle) return std::nullopt;
    PreparedVi prepared;
    prepared.relay = std::move(cloned);
    prepared.relay_handle = *handle;
    std::array<std::uint32_t, 24> arguments{};
    arguments[0] = 0;
    arguments[1] = 84;
    arguments[5] = 1;
    arguments[12] = 8;
    const auto at = prepared.queue.parcel(arguments);
    prepared.queue_size = static_cast<std::uint32_t>(
        at + arguments.size() * 4);
    return prepared;
}

std::uint32_t packed_address(std::uint64_t address) {
    return static_cast<std::uint32_t>(((address >> 36) & 7) << 2) |
           static_cast<std::uint32_t>(((address >> 32) & 15) << 28);
}

bool install_guest_queue(const RuntimeHarness& harness, const PreparedVi& prepared,
                         bool valid_output) {
    constexpr std::uint64_t packet_offset = 0x1000;
    constexpr std::uint64_t input_offset = 0x2000;
    constexpr std::uint64_t output_offset = 0x3000;
    constexpr std::uint64_t invalid_output_offset = 0x300000;
    horizon::IpcMessage packet{};
    write(packet, 0, 6 | (1u << 20) | (1u << 24), 4);
    write(packet, 4, 12, 4);
    write(packet, 8, prepared.queue_size, 4);
    write(packet, 12, static_cast<std::uint32_t>(harness.heap + input_offset), 4);
    write(packet, 16, packed_address(harness.heap + input_offset), 4);
    const auto output = harness.heap +
        (valid_output ? output_offset : invalid_output_offset);
    write(packet, 20, 0x200, 4);
    write(packet, 24, static_cast<std::uint32_t>(output), 4);
    write(packet, 28, packed_address(output), 4);
    write(packet, 32, sfci, 4);
    write(packet, 36, 1, 4);
    write(packet, 40, 3, 4);
    write(packet, 48, 1, 4);
    write(packet, 52, 7, 4);
    const auto memory = harness.state.memory->ipc_buffers();
    return memory.write_memory(harness.heap + packet_offset, packet) &&
           memory.write_memory(
               harness.heap + input_offset,
               std::span<const std::byte>(prepared.queue.input).first(
                   prepared.queue_size));
}

bool main_exited(const RuntimeHarness& harness) {
    const auto snapshot = harness.state.threads->snapshot(harness.state.main_thread);
    const auto* thread = value(snapshot);
    return thread && thread->state == ThreadState::exited;
}

struct BoundaryGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    std::stop_token stop;

    void arrive_and_wait(std::stop_token token) {
        std::unique_lock lock(mutex);
        stop = token;
        entered = true;
        changed.notify_all();
        changed.wait(lock, [this] { return released; });
    }
    bool wait_entered() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 2s, [this] { return entered; });
    }
    bool stop_requested() {
        const std::lock_guard lock(mutex);
        return stop.stop_requested();
    }
    void release() {
        const std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};

void test_stop_between_dispatch_and_service_admission(bool service_lookup, bool frame_stop) {
    auto clock_result = create_guest_clock();
    CHECK(value(clock_result) != nullptr);
    if (!value(clock_result)) return;
    auto presenter = std::make_shared<TestPresenter>();
    auto made = create_runtime(presenter, *value(clock_result));
    if (!made) return;
    auto harness = std::move(*made);
    BoundaryGate boundary;
    ProcessRuntimeTestAccess::set_ipc_admission_boundary(
        *harness.runtime, [&](std::stop_token stop) { boundary.arrive_and_wait(stop); });
    RunLimits limits;
    if (frame_stop) limits.completed_frame_target = 1;
    std::optional<RunResult> result;
    std::thread worker([&] { result = harness.runtime->run(limits, {}); });
    CHECK(wait_for_word(harness.heap, 1));
    std::optional<PreparedVi> frame;
    if (frame_stop) {
        frame = prepare_vi(harness.state, harness.heap);
        CHECK(frame.has_value());
    }
    if (service_lookup) {
        auto sm = connect_sm(harness.state);
        const auto handle = harness.state.handles->insert(sm);
        CHECK(handle.has_value());
        const auto packet = get_service("vi:u");
        CHECK(harness.state.memory->ipc_buffers().write_memory(harness.heap + 0x1000, packet));
        store_word(harness.heap + 8, handle.value_or(0));
    }
    store_word(harness.heap + 4, service_lookup ? 1 : 3);
    CHECK(boundary.wait_entered());
    if (frame) {
        CHECK(queue_buffer(frame->relay, harness.state.process_id).disposition ==
              horizon::IpcDisposition::handled);
        CHECK(wait_until([&] {
            return ProcessRuntimeTestAccess::run_success_selected(*harness.runtime) &&
                   boundary.stop_requested();
        }));
    } else {
        harness.runtime->request_stop();
    }
    CHECK(boundary.stop_requested());
    boundary.release();
    worker.join();
    CHECK(result.has_value());
    if (!result) return;
    if (frame_stop) {
        CHECK(std::holds_alternative<RunSuccess>(result->outcome));
        CHECK(result->statistics.completed_frames == 1);
    } else {
        CHECK(std::holds_alternative<RunCancelled>(result->outcome));
    }
    CHECK(result->quiescent);
    CHECK(!result->teardown_failure);
}

void test_stop_before_presentation_admission_is_cancellation() {
    auto clock_result = create_guest_clock();
    CHECK(value(clock_result) != nullptr);
    if (!value(clock_result)) return;
    auto presenter = std::make_shared<TestPresenter>();
    auto made = create_runtime(presenter, *value(clock_result));
    if (!made) return;
    auto harness = std::move(*made);
    BoundaryGate boundary;
    ProcessRuntimeTestAccess::set_ipc_admission_boundary(
        *harness.runtime, [&](std::stop_token stop) { boundary.arrive_and_wait(stop); });
    std::optional<RunResult> result;
    std::thread worker([&] { result = harness.runtime->run({}, {}); });
    CHECK(wait_for_word(harness.heap, 1));
    auto prepared = prepare_vi(harness.state, harness.heap);
    CHECK(prepared.has_value());
    if (prepared) {
        CHECK(install_guest_queue(harness, *prepared, true));
        store_word(harness.heap + 8, prepared->relay_handle);
        store_word(harness.heap + 4, 1);
        CHECK(boundary.wait_entered());
    }
    harness.runtime->request_stop();
    boundary.release();
    worker.join();
    CHECK(result.has_value());
    if (!result) return;
    CHECK(std::holds_alternative<RunCancelled>(result->outcome));
    CHECK(result->statistics.completed_frames == 0);
    CHECK(result->quiescent);
    CHECK(!result->teardown_failure);
}

void test_failed_ipc_after_target_release_is_failure() {
    auto clock_result = create_guest_clock();
    CHECK(value(clock_result) != nullptr);
    if (!value(clock_result)) return;
    auto presenter = std::make_shared<TestPresenter>();
    auto made = create_runtime(presenter, *value(clock_result));
    if (!made) return;
    auto harness = std::move(*made);
    BoundaryGate boundary;
    ProcessRuntimeTestAccess::set_failed_ipc_boundary(
        *harness.runtime, [&](std::stop_token stop) { boundary.arrive_and_wait(stop); });
    RunLimits limits;
    limits.completed_frame_target = 1;
    std::optional<RunResult> result;
    std::thread worker([&] { result = harness.runtime->run(limits, {}); });
    CHECK(wait_for_word(harness.heap, 1));
    auto prepared = prepare_vi(harness.state, harness.heap);
    CHECK(prepared.has_value());
    if (prepared) {
        CHECK(install_guest_queue(harness, *prepared, false));
        store_word(harness.heap + 8, prepared->relay_handle);
        store_word(harness.heap + 4, 1);
        CHECK(boundary.wait_entered());
        CHECK(wait_until([&] {
            return ProcessRuntimeTestAccess::run_success_selected(*harness.runtime) &&
                   boundary.stop_requested();
        }));
    }
    boundary.release();
    if (!prepared) harness.runtime->request_stop();
    worker.join();
    CHECK(result.has_value());
    if (!result) return;
    const auto* stopped = failure(*result);
    CHECK(stopped != nullptr);
    if (stopped) {
        CHECK(stopped->stage == LaunchStage::services);
        CHECK(stopped->code == LaunchError::unsupported);
        CHECK(stopped->command == 0x21);
    }
    CHECK(result->statistics.completed_frames == 1);
    CHECK(result->quiescent);
}

void test_target_release_before_immediate_exit_is_success() {
    auto base_result = create_guest_clock();
    CHECK(value(base_result) != nullptr);
    if (!value(base_result)) return;
    auto clock = std::make_shared<SecondSampleGateClock>(*value(base_result));
    auto presenter = std::make_shared<TestPresenter>();
    auto made = create_runtime(presenter, clock);
    if (!made) return;
    auto harness = std::move(*made);
    RunLimits limits;
    limits.completed_frame_target = 1;
    std::optional<RunResult> result;
    std::thread worker([&] {
        clock->arm_for_current_thread();
        result = harness.runtime->run(limits, {});
    });
    CHECK(wait_for_word(harness.heap, 1));
    CHECK(clock->wait_entered());
    auto prepared = prepare_vi(harness.state, harness.heap);
    CHECK(prepared.has_value());
    if (prepared) {
        CHECK(install_guest_queue(harness, *prepared, true));
        store_word(harness.heap + 8, prepared->relay_handle);
        store_word(harness.heap + 4, 1);
        CHECK(wait_for_word(harness.heap + 12, 1));
        CHECK(wait_until([&] { return main_exited(harness); }));
    }
    if (!prepared) harness.runtime->request_stop();
    clock->release();
    worker.join();
    CHECK(result.has_value());
    if (!result) return;
    const auto* success = std::get_if<RunSuccess>(&result->outcome);
    CHECK(success != nullptr && success->frame_target == 1);
    CHECK(result->statistics.completed_frames == 1);
    CHECK(result->quiescent);
}

void test_exit_before_release_stays_guest_exit() {
    auto clock_result = create_guest_clock();
    CHECK(value(clock_result) != nullptr);
    if (!value(clock_result)) return;
    auto clock = std::make_shared<SecondSampleGateClock>(*value(clock_result));
    auto presenter = std::make_shared<TestPresenter>();
    auto made = create_runtime(presenter, clock);
    if (!made) return;
    auto harness = std::move(*made);
    RunLimits limits;
    limits.completed_frame_target = 1;
    std::optional<RunResult> result;
    std::thread worker([&] {
        clock->arm_for_current_thread();
        result = harness.runtime->run(limits, {});
    });
    CHECK(wait_for_word(harness.heap, 1));
    CHECK(clock->wait_entered());
    auto prepared = prepare_vi(harness.state, harness.heap);
    CHECK(prepared.has_value());
    std::future<horizon::IpcReply> queued;
    if (prepared) {
        presenter->block();
        queued = std::async(std::launch::async, [&] {
            return queue_buffer(prepared->relay, harness.state.process_id);
        });
        CHECK(presenter->wait_entered());
        store_word(harness.heap + 4, 2);
        CHECK(wait_until([&] { return main_exited(harness); }));
        presenter->release();
        CHECK(queued.wait_for(2s) == std::future_status::ready);
        if (queued.valid()) {
            CHECK(queued.get().disposition == horizon::IpcDisposition::handled);
        }
    } else {
        harness.runtime->request_stop();
    }
    clock->release();
    worker.join();
    CHECK(result.has_value());
    if (!result) return;
    CHECK(std::holds_alternative<GuestExit>(result->outcome));
    CHECK(result->statistics.completed_frames == 1);
    CHECK(result->quiescent);
}

} // namespace

int main() {
    test_stop_between_dispatch_and_service_admission(false, false);
    test_stop_between_dispatch_and_service_admission(true, false);
    test_stop_between_dispatch_and_service_admission(false, true);
    test_stop_between_dispatch_and_service_admission(true, true);
    test_stop_before_presentation_admission_is_cancellation();
    test_failed_ipc_after_target_release_is_failure();
    test_target_release_before_immediate_exit_is_success();
    test_exit_before_release_stays_guest_exit();
    std::printf("Process-outcome failures: %u\n", failures);
    return failures ? 1 : 0;
}
