// ============================================================================
// Armigo Tests - Video Interface Presentation Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/presentation.h>
#include <armigo/services/video_interface.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace armigo;
using namespace armigo::runtime;
using namespace std::chrono_literals;
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;
using services::vi::DisplaySession;

std::atomic<unsigned> failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, sizeof value);
}

std::uint32_t word(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof value);
    return value;
}

IpcMessage command(std::uint32_t id) {
    IpcMessage message{};
    put(message, 0, 6);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, id);
    return message;
}

IpcMessage transact(std::uint32_t operation) {
    auto message = command(3);
    put(message, 32, 1);
    put(message, 36, operation);
    return message;
}

struct BinderBuffers {
    std::array<std::byte, 0x200> input{};
    std::array<std::byte, 0x200> output{};
    bool fail_write = false;

    horizon::GuestBuffers buffers(std::uint32_t input_size) {
        horizon::GuestBuffers result(
            [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
                auto& self = *static_cast<BinderBuffers*>(raw);
                if (address != 0x1000 || out.size() > self.input.size()) return false;
                std::memcpy(out.data(), self.input.data(), out.size());
                return true;
            },
            [](void* raw, std::uint64_t address,
               std::span<const std::byte> in) noexcept {
                auto& self = *static_cast<BinderBuffers*>(raw);
                if (self.fail_write || address != 0x2000 ||
                    in.size() > self.output.size()) return false;
                std::memcpy(self.output.data(), in.data(), in.size());
                return true;
            }, this);
        result.add({.address = 0x1000, .size = input_size, .input = true});
        result.add({.address = 0x2000,
                    .size = static_cast<std::uint32_t>(output.size()),
                    .output = true});
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

struct Pin final : PinnedGuestMemory {
    std::shared_ptr<std::vector<std::byte>> storage;
    horizon::AddressRange selected;
    std::uint64_t offset = 0;

    Pin(std::shared_ptr<std::vector<std::byte>> bytes,
        horizon::AddressRange range, std::uint64_t start)
        : storage(std::move(bytes)), selected(range), offset(start) {}

    horizon::AddressRange range() const noexcept override { return selected; }
    memory::Permission permissions() const noexcept override {
        return memory::Permission::read;
    }
    std::span<const std::byte> bytes() const noexcept override {
        return std::span<const std::byte>(*storage).subspan(offset, selected.size);
    }
    std::span<std::byte> writable_bytes() noexcept override { return {}; }
};

struct BlockingPresenter final : platform::Presenter {
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> frames;
    std::thread::id worker;
    bool block = false;
    bool fail = false;

    platform::PresentationMode mode() const noexcept override {
        return platform::PresentationMode::headless;
    }

    platform::PresentResult present(const platform::PresentationFrame& frame,
                                    std::optional<ClockInstant> deadline,
                                    std::stop_token) override {
        CHECK(!deadline);
        std::unique_lock lock(mutex);
        worker = std::this_thread::get_id();
        frames.emplace_back(frame.queued.frame_number, frame.queued.slot);
        changed.notify_all();
        changed.wait(lock, [&] { return !block; });
        if (fail) {
            return platform::PresentationFailure{
                platform::PresentationError::host_failure,
                "injected presentation failure"};
        }
        platform::PresentedFrame completed;
        completed.frame_number = frame.queued.frame_number;
        completed.readback = *frame.pixels;
        return completed;
    }

    platform::PresenterStatus drain(std::optional<ClockInstant>) override {
        return std::monostate{};
    }

    bool wait_for_frames(std::size_t count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 2s, [&] { return frames.size() >= count; });
    }

    void release() {
        const std::lock_guard lock(mutex);
        block = false;
        changed.notify_all();
    }

    std::thread::id worker_thread() const {
        const std::lock_guard lock(mutex);
        return worker;
    }

    std::vector<std::pair<std::uint64_t, std::uint32_t>> observed_frames() const {
        const std::lock_guard lock(mutex);
        return frames;
    }
};

struct IntegrationFixture {
    horizon::HandleTable handles{16};
    DisplaySession root{handles};
    std::shared_ptr<DisplaySession> relay;
    std::shared_ptr<BlockingPresenter> presenter = std::make_shared<BlockingPresenter>();
    std::array<std::shared_ptr<std::vector<std::byte>>, 2> storage{
        std::make_shared<std::vector<std::byte>>(0x200),
        std::make_shared<std::vector<std::byte>>(0x200)};
    std::shared_ptr<PresentationClock> clock;
    std::shared_ptr<PresentationBinding> binding;

    std::uint32_t pitch;
    std::uint32_t surface_size;
    std::optional<graphics::Rgba8Image> observed;

    IntegrationFixture(std::uint32_t stride = 64, std::uint32_t size = 512)
        : pitch(stride), surface_size(size) {
        auto service_reply = root.request(command(1), 5);
        CHECK(service_reply.disposition == IpcDisposition::handled);
        auto service = std::dynamic_pointer_cast<DisplaySession>(service_reply.transferred[0]);
        CHECK(service != nullptr);
        if (!service) return;
        auto relay_reply = service->request(command(100), 5);
        CHECK(relay_reply.disposition == IpcDisposition::handled);
        relay = std::dynamic_pointer_cast<DisplaySession>(relay_reply.transferred[0]);
        CHECK(relay != nullptr);
        if (!relay) return;

        for (auto& bytes : storage) {
            bytes->resize(surface_size, std::byte{0xcc});
            // Fixed GOB coordinates, independent of the production decoder.
            constexpr std::array<unsigned, 8> columns{0, 4, 8, 12, 32, 36, 40, 44};
            constexpr std::array<unsigned, 8> rows{0, 16, 64, 80, 128, 144, 192, 208};
            for (std::uint32_t y = 0; y < 8; ++y) {
                for (std::uint32_t x = 0; x < 8; ++x) {
                    const auto at = rows[y] + columns[x];
                    (*bytes)[at] = static_cast<std::byte>(x);
                    (*bytes)[at + 1] = static_cast<std::byte>(y);
                    (*bytes)[at + 2] = static_cast<std::byte>(x + y);
                    (*bytes)[at + 3] = std::byte{0xff};
                }
            }
        }

        auto guest = create_guest_clock();
        CHECK(std::holds_alternative<std::shared_ptr<GuestClock>>(guest));
        if (!std::holds_alternative<std::shared_ptr<GuestClock>>(guest)) return;
        auto created_clock = create_presentation_clock(
            std::get<std::shared_ptr<GuestClock>>(guest),
            {std::chrono::steady_clock::now(), 1ms});
        CHECK(std::holds_alternative<std::shared_ptr<PresentationClock>>(created_clock));
        if (!std::holds_alternative<std::shared_ptr<PresentationClock>>(created_clock)) return;
        clock = std::get<std::shared_ptr<PresentationClock>>(std::move(created_clock));

        services::GraphicsBufferBackingLookup lookup = [this](std::uint32_t id)
            -> std::optional<services::GraphicsBufferBacking> {
            if (id < 77 || id > 78) return std::nullopt;
            const auto index = static_cast<std::size_t>(id - 77);
            return services::GraphicsBufferBacking{
                id, 0x1000, 0x10000 + index * 0x1000, surface_size, storage[index]};
        };
        relay->set_buffer_backing_lookup(lookup);

        PresentationDependencies dependencies;
        dependencies.backing.lookup = lookup;
        dependencies.backing.pin = [this](const services::GraphicsBufferBacking& backing,
                                           std::uint64_t offset,
                                           std::uint64_t size)
            -> LaunchResult<GuestMemoryPin> {
            if (backing.export_id < 77 || backing.export_id > 78 ||
                offset > backing.size || size > backing.size - offset) {
                return LaunchFailure{.stage = LaunchStage::presentation,
                                     .code = LaunchError::invalid_input,
                                     .operation = "test.pin",
                                     .detail = "invalid synthetic backing"};
            }
            const auto index = static_cast<std::size_t>(backing.export_id - 77);
            if (backing.owner != storage[index]) {
                return LaunchFailure{.stage = LaunchStage::presentation,
                                     .code = LaunchError::permission_denied,
                                     .operation = "test.pin",
                                     .detail = "synthetic owner mismatch"};
            }
            CHECK(offset == 0 && size == surface_size);
            return GuestMemoryPin{std::make_shared<Pin>(
                storage[index],
                horizon::AddressRange{backing.guest_address + offset, size}, offset)};
        };
        dependencies.backing.wait_fence = [](const services::GraphicsFence&,
                                              std::optional<ClockInstant>,
                                              std::stop_token) -> LaunchStatus {
            return std::monostate{};
        };
        dependencies.clock = clock;
        dependencies.presenter = presenter;
        dependencies.observe_frame = [this](const platform::PresentedFrame& completed) -> LaunchStatus {
            observed = completed.readback;
            return std::monostate{};
        };
        auto created = bind_presentation(std::move(dependencies));
        CHECK(std::holds_alternative<std::shared_ptr<PresentationBinding>>(created));
        if (!std::holds_alternative<std::shared_ptr<PresentationBinding>>(created)) return;
        binding = std::get<std::shared_ptr<PresentationBinding>>(std::move(created));
        relay->set_presenter(binding->callback());
    }

    ~IntegrationFixture() {
        presenter->release();
        if (relay) relay->set_presenter({});
        if (binding) {
            binding->request_stop();
            CHECK(std::holds_alternative<std::monostate>(binding->drain(std::nullopt)));
            binding.reset();
        }
    }

    std::shared_ptr<DisplaySession> clone_relay() {
        IpcMessage clone{};
        put(clone, 0, 7);
        put(clone, 4, 8);
        put(clone, 16, 0x49434653);
        put(clone, 24, 2);
        const auto reply = relay->request(clone, 5);
        CHECK(reply.disposition == IpcDisposition::handled);
        return std::dynamic_pointer_cast<DisplaySession>(reply.transferred[0]);
    }

    std::array<std::byte, 0x16c> flattened(std::uint32_t slot) {
        std::array<std::byte, 0x16c> flat{};
        const auto set = [&](std::size_t offset, std::uint32_t value) {
            std::memcpy(flat.data() + offset, &value, sizeof value);
        };
        set(0, 0x47424652); set(4, 8); set(8, 8); set(0x0c, 8);
        set(0x10, 1); set(0x14, 0x800); set(0x20, 0); set(0x24, 81);
        set(0x2c, 77 + slot); set(0x34, 0xdaffcaff); set(0x40, 0x800);
        set(0x44, 1); set(0x48, 1); set(0x4c, 8); set(0x50, surface_size);
        set(0x54, 1); set(0x5c, 8); set(0x60, 8); set(0x64, 0x00532120);
        set(0x68, 1); set(0x6c, 3); set(0x70, pitch); set(0x78, 0);
        set(0x80, 0); set(0x94, surface_size);
        return flat;
    }

    IpcReply set_slot(std::uint32_t slot, bool present = true) {
        BinderBuffers data;
        const auto at = present
            ? data.parcel(std::array<std::uint32_t, 4>{slot, 1, 0x16c, 0})
            : data.parcel(std::array<std::uint32_t, 2>{slot, 0});
        std::size_t size = at + 8;
        if (present) {
            const auto flat = flattened(slot);
            std::copy(flat.begin(), flat.end(), data.input.begin() + at + 16);
            size = at + 16 + flat.size();
            data.set_word(0, static_cast<std::uint32_t>(size - 16));
        }
        auto buffers = data.buffers(static_cast<std::uint32_t>(size));
        return relay->request_with_buffers(transact(14), 5, buffers);
    }

    std::pair<IpcReply, std::int32_t> dequeue(DisplaySession& caller) {
        BinderBuffers data;
        const auto at = data.parcel(std::array<std::uint32_t, 5>{0, 8, 8, 1, 0});
        auto buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
        auto reply = caller.request_with_buffers(transact(3), 5, buffers);
        return {std::move(reply), std::bit_cast<std::int32_t>(word(data.output, 0x10))};
    }

    IpcReply request_buffer(DisplaySession& caller, std::uint32_t slot) {
        BinderBuffers data;
        const auto at = data.parcel(std::array<std::uint32_t, 1>{slot});
        auto buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
        return caller.request_with_buffers(transact(1), 5, buffers);
    }

    IpcReply queue(DisplaySession& caller, std::uint32_t slot,
                   bool fail_reply_copy = false) {
        std::array<std::uint32_t, 24> arguments{};
        arguments[0] = slot;
        arguments[1] = 84;
        arguments[5] = 1;
        arguments[12] = 8;
        BinderBuffers data;
        const auto at = data.parcel(arguments);
        data.fail_write = fail_reply_copy;
        auto buffers = data.buffers(static_cast<std::uint32_t>(at + arguments.size() * 4));
        return caller.request_with_buffers(transact(7), 5, buffers);
    }

    std::shared_ptr<horizon::Event> free_event(DisplaySession& caller) {
        const auto reply = caller.request(command(2), 5);
        CHECK(reply.disposition == IpcDisposition::handled);
        return std::dynamic_pointer_cast<horizon::Event>(reply.transferred[0]);
    }
};

void successful_completion_releases_slots_in_frame_order() {
    IntegrationFixture fixture;
    CHECK(fixture.relay && fixture.binding);
    if (!fixture.relay || !fixture.binding) return;
    CHECK(fixture.set_slot(0).disposition == IpcDisposition::handled);
    CHECK(fixture.set_slot(1).disposition == IpcDisposition::handled);
    auto first = fixture.clone_relay();
    auto second = fixture.clone_relay();
    auto observer = fixture.clone_relay();
    CHECK(first && second && observer);
    if (!first || !second || !observer) return;
    auto event = fixture.free_event(*observer);
    CHECK(event && event->is_signaled());

    const auto dequeued_first = fixture.dequeue(*first);
    const auto dequeued_second = fixture.dequeue(*second);
    CHECK(dequeued_first.first.disposition == IpcDisposition::handled &&
          dequeued_first.second == 0);
    CHECK(dequeued_second.first.disposition == IpcDisposition::handled &&
          dequeued_second.second == 1);
    CHECK(fixture.request_buffer(*first, 0).disposition == IpcDisposition::handled);
    CHECK(fixture.request_buffer(*second, 1).disposition == IpcDisposition::handled);
    CHECK(event && !event->is_signaled());

    fixture.presenter->block = true;
    std::thread::id first_caller;
    auto queued_first = std::async(std::launch::async, [&] {
        first_caller = std::this_thread::get_id();
        return fixture.queue(*first, 0);
    });
    CHECK(fixture.presenter->wait_for_frames(1));
    CHECK(fixture.presenter->worker_thread() != first_caller);
    CHECK(event && !event->is_signaled());

    auto native = std::async(std::launch::async, [&] { return fixture.free_event(*observer); });
    const bool native_ready = native.wait_for(500ms) == std::future_status::ready;
    CHECK(native_ready);
    if (!native_ready) fixture.presenter->release();
    auto observed_event = native.get();
    CHECK(observed_event == event);
    CHECK(event && !event->is_signaled());

    std::promise<void> second_started;
    auto second_ready = second_started.get_future();
    auto queued_second = std::async(std::launch::async, [&] {
        second_started.set_value();
        return fixture.queue(*second, 1);
    });
    second_ready.wait();
    CHECK(queued_second.wait_for(50ms) == std::future_status::timeout);
    CHECK(event && !event->is_signaled());

    fixture.presenter->release();
    CHECK(queued_first.wait_for(2s) == std::future_status::ready);
    CHECK(queued_second.wait_for(2s) == std::future_status::ready);
    CHECK(queued_first.get().disposition == IpcDisposition::handled);
    CHECK(queued_second.get().disposition == IpcDisposition::handled);
    const auto frames = fixture.presenter->observed_frames();
    CHECK((frames == std::vector<std::pair<std::uint64_t, std::uint32_t>>{
        {1, 0}, {2, 1}}));
    CHECK(event && event->is_signaled());

    const auto again = fixture.dequeue(*first);
    CHECK(again.first.disposition == IpcDisposition::handled && again.second == 0);
    CHECK(fixture.request_buffer(*first, 0).disposition == IpcDisposition::handled);
    CHECK(fixture.queue(*first, 0, true).disposition == IpcDisposition::unsupported);
    CHECK(event && event->is_signaled());
    CHECK(fixture.set_slot(0).disposition == IpcDisposition::handled);
    const auto released_after_copy_failure = fixture.dequeue(*first);
    CHECK(released_after_copy_failure.first.disposition == IpcDisposition::handled);
}

void padded_surfaces_release_slots_and_allow_following_frames() {
    for (const auto pitch : {64u, 128u}) {
        IntegrationFixture fixture(pitch, 1024);
        CHECK(fixture.relay && fixture.binding);
        if (!fixture.relay || !fixture.binding) continue;
        CHECK(fixture.set_slot(0).disposition == IpcDisposition::handled);
        auto caller = fixture.clone_relay();
        CHECK(caller != nullptr);
        if (!caller) continue;
        auto event = fixture.free_event(*caller);
        for (unsigned frame = 0; frame < 2; ++frame) {
            const auto dequeued = fixture.dequeue(*caller);
            CHECK(dequeued.first.disposition == IpcDisposition::handled && dequeued.second == 0);
            if (dequeued.first.disposition != IpcDisposition::handled) break;
            CHECK(fixture.request_buffer(*caller, 0).disposition == IpcDisposition::handled);
            CHECK(fixture.queue(*caller, 0).disposition == IpcDisposition::handled);
            CHECK(!fixture.binding->failure());
            CHECK(event && event->is_signaled());
            CHECK(fixture.observed.has_value());
            if (!fixture.observed) continue;
            const auto& image = *fixture.observed;
            CHECK(image.width == 8 && image.height == 8);
            for (unsigned y = 0; y < 8; ++y) {
                for (unsigned x = 0; x < 8; ++x) {
                    const auto at = (y * 8 + x) * 4;
                    CHECK(image.pixels[at] == std::byte(x));
                    CHECK(image.pixels[at + 1] == std::byte(y));
                    CHECK(image.pixels[at + 2] == std::byte(x + y));
                    CHECK(image.pixels[at + 3] == std::byte{0xff});
                }
            }
        }
        CHECK(fixture.presenter->observed_frames().size() == 2);
    }
}

void failed_completion_retains_acquired_slot() {
    IntegrationFixture fixture;
    CHECK(fixture.relay && fixture.binding);
    if (!fixture.relay || !fixture.binding) return;
    fixture.presenter->fail = true;
    CHECK(fixture.set_slot(0).disposition == IpcDisposition::handled);
    auto caller = fixture.clone_relay();
    CHECK(caller != nullptr);
    if (!caller) return;
    auto event = fixture.free_event(*caller);
    const auto dequeued = fixture.dequeue(*caller);
    CHECK(dequeued.first.disposition == IpcDisposition::handled && dequeued.second == 0);
    CHECK(fixture.request_buffer(*caller, 0).disposition == IpcDisposition::handled);
    CHECK(event && !event->is_signaled());
    CHECK(fixture.queue(*caller, 0).disposition == IpcDisposition::unsupported);
    CHECK(event && !event->is_signaled());
    CHECK(fixture.set_slot(0).disposition == IpcDisposition::unsupported);
    CHECK(fixture.set_slot(0, false).disposition == IpcDisposition::unsupported);
    CHECK(fixture.dequeue(*caller).first.disposition == IpcDisposition::unsupported);
}

#if defined(ARMIGO_LANE_G_GPU_TESTS)
void metal_headless_completion_matches_authored_pixels() {
    IntegrationFixture fixture;
    CHECK(fixture.relay && fixture.binding);
    if (!fixture.relay || !fixture.binding) return;
    fixture.relay->set_presenter({});
    fixture.binding->request_stop();
    CHECK(std::holds_alternative<std::monostate>(
        fixture.binding->drain(std::nullopt)));
    fixture.binding.reset();

    auto created_presenter = platform::create_presenter(
        {.mode = platform::PresentationMode::headless, .width = 8, .height = 8});
    CHECK(std::holds_alternative<std::shared_ptr<platform::Presenter>>(created_presenter));
    if (!std::holds_alternative<std::shared_ptr<platform::Presenter>>(created_presenter)) return;
    auto metal = std::get<std::shared_ptr<platform::Presenter>>(
        std::move(created_presenter));
    std::optional<graphics::Rgba8Image> observed;

    PresentationDependencies dependencies;
    dependencies.backing.lookup = [&fixture](std::uint32_t id)
        -> std::optional<services::GraphicsBufferBacking> {
        if (id != 77) return std::nullopt;
        return services::GraphicsBufferBacking{
            id, 0x1000, 0x10000, 0x200, fixture.storage[0]};
    };
    dependencies.backing.pin = [&fixture](
        const services::GraphicsBufferBacking& backing, std::uint64_t offset,
        std::uint64_t size) -> LaunchResult<GuestMemoryPin> {
        if (backing.export_id != 77 || backing.owner != fixture.storage[0] ||
            offset > backing.size || size > backing.size - offset) {
            return LaunchFailure{.stage = LaunchStage::presentation,
                                 .code = LaunchError::invalid_input,
                                 .operation = "test.metal_pin",
                                 .detail = "invalid synthetic Metal backing"};
        }
        return GuestMemoryPin{std::make_shared<Pin>(
            fixture.storage[0],
            horizon::AddressRange{backing.guest_address + offset, size}, offset)};
    };
    dependencies.backing.wait_fence = [](const services::GraphicsFence&,
                                          std::optional<ClockInstant>,
                                          std::stop_token) -> LaunchStatus {
        return std::monostate{};
    };
    dependencies.clock = fixture.clock;
    dependencies.presenter = std::move(metal);
    dependencies.observe_frame = [&observed](const platform::PresentedFrame& completed)
        -> LaunchStatus {
        if (!completed.readback) {
            return LaunchFailure{.stage = LaunchStage::presentation,
                                 .code = LaunchError::host_failure,
                                 .operation = "test.observe_metal",
                                 .detail = "headless Metal completion omitted readback"};
        }
        observed = *completed.readback;
        return std::monostate{};
    };
    auto created = bind_presentation(std::move(dependencies));
    CHECK(std::holds_alternative<std::shared_ptr<PresentationBinding>>(created));
    if (!std::holds_alternative<std::shared_ptr<PresentationBinding>>(created)) return;
    fixture.binding = std::get<std::shared_ptr<PresentationBinding>>(std::move(created));
    fixture.relay->set_presenter(fixture.binding->callback());

    CHECK(fixture.set_slot(0).disposition == IpcDisposition::handled);
    auto caller = fixture.clone_relay();
    CHECK(caller != nullptr);
    if (!caller) return;
    const auto dequeued = fixture.dequeue(*caller);
    CHECK(dequeued.first.disposition == IpcDisposition::handled && dequeued.second == 0);
    CHECK(fixture.request_buffer(*caller, 0).disposition == IpcDisposition::handled);
    CHECK(fixture.queue(*caller, 0).disposition == IpcDisposition::handled);
    CHECK(observed && observed->width == 8 && observed->height == 8 &&
          observed->pixels.size() == 8 * 8 * 4);
    if (!observed || observed->pixels.size() != 8 * 8 * 4) return;
    for (std::uint32_t y = 0; y < 8; ++y) {
        for (std::uint32_t x = 0; x < 8; ++x) {
            const auto at = (y * 8 + x) * 4;
            CHECK(observed->pixels[at] == static_cast<std::byte>(x));
            CHECK(observed->pixels[at + 1] == static_cast<std::byte>(y));
            CHECK(observed->pixels[at + 2] == static_cast<std::byte>(x + y));
            CHECK(observed->pixels[at + 3] == std::byte{0xff});
        }
    }
}
#endif

} // namespace

int main() {
    successful_completion_releases_slots_in_frame_order();
    padded_surfaces_release_slots_and_allow_following_frames();
    failed_completion_retains_acquired_slot();
#if defined(ARMIGO_LANE_G_GPU_TESTS)
    metal_headless_completion_matches_authored_pixels();
#endif
    std::fprintf(stderr, "VI presentation integration failures: %u\n", failures.load());
    return failures == 0 ? 0 : 1;
}
