// ============================================================================
// Armigo Tests - Native Runtime Verification
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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace {
using namespace armigo;
using namespace armigo::runtime;
using namespace std::chrono_literals;
std::atomic<unsigned> failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (false)

LaunchFailure error(const char* operation, LaunchError code = LaunchError::host_failure) {
    LaunchFailure result;
    result.stage = LaunchStage::presentation;
    result.code = code;
    result.operation = operation;
    result.detail = operation;
    return result;
}

struct Timeline {
    std::mutex mutex;
    std::condition_variable_any changed;
    ClockInstant now{};
    std::vector<ClockInstant> deadlines;
    ClockInstant sample() {
        const std::lock_guard lock(mutex);
        return now;
    }
    LaunchResult<ClockWait> wait(ClockInstant deadline, std::stop_token stop) {
        std::unique_lock lock(mutex);
        deadlines.push_back(deadline);
        changed.notify_all();
        return changed.wait(lock, stop, [&] { return now >= deadline; })
            ? ClockWait::reached : ClockWait::cancelled;
    }
    void advance(std::int64_t ns) {
        const std::lock_guard lock(mutex);
        now = ClockInstant{std::chrono::nanoseconds{ns}};
        changed.notify_all();
    }
    bool waiting(std::size_t count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 2s, [&] { return deadlines.size() >= count; });
    }
    ClockInstant target(std::size_t index) {
        const std::lock_guard lock(mutex);
        return deadlines.at(index);
    }
};

struct Pin final : PinnedGuestMemory {
    std::shared_ptr<std::vector<std::byte>> storage;
    horizon::AddressRange selected;
    std::uint64_t offset;
    Pin(std::shared_ptr<std::vector<std::byte>> data, horizon::AddressRange range, std::uint64_t start)
        : storage(std::move(data)), selected(range), offset(start) {}
    horizon::AddressRange range() const noexcept override { return selected; }
    memory::Permission permissions() const noexcept override { return memory::Permission::read; }
    std::span<const std::byte> bytes() const noexcept override {
        return std::span<const std::byte>(*storage).subspan(offset, selected.size);
    }
    std::span<std::byte> writable_bytes() noexcept override { return {}; }
};

struct Consumer final : platform::Presenter {
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::thread::id thread;
    std::vector<platform::PresentationFrame> frames;
    bool block = false;
    bool entered = false;
    bool fail_present = false;
    bool fail_drain = false;
    bool omit_readback = false;
    bool throw_present = false;
    unsigned drains = 0;
    std::function<void()> during_present;
    platform::PresentationMode mode() const noexcept override { return platform::PresentationMode::headless; }
    platform::PresentResult present(const platform::PresentationFrame& frame,
        std::optional<ClockInstant> deadline, std::stop_token) override {
        CHECK(!deadline);
        std::unique_lock lock(mutex);
        if (thread != std::thread::id{}) CHECK(thread == std::this_thread::get_id());
        thread = std::this_thread::get_id();
        entered = true;
        frames.push_back(frame);
        changed.notify_all();
        changed.wait(lock, [&] { return !block; });
        if (during_present) during_present();
        if (throw_present) throw 42;
        if (fail_present) return platform::PresentationFailure{
            platform::PresentationError::deadline_expired, "injected consumer timeout"};
        platform::PresentedFrame result;
        result.frame_number = frame.queued.frame_number;
        // Like a real consumer, only a frame that asked for pixels gets them.
        if (!omit_readback && frame.readback && frame.pixels) result.readback = *frame.pixels;
        return result;
    }
    platform::PresenterStatus drain(std::optional<ClockInstant>) override {
        const std::lock_guard lock(mutex);
        CHECK(thread == std::thread::id{} || thread == std::this_thread::get_id());
        ++drains;
        changed.notify_all();
        if (fail_drain) return platform::PresentationFailure{
            platform::PresentationError::deadline_expired, "injected drain timeout"};
        frames.clear();
        return std::monostate{};
    }
    bool wait_entered() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 2s, [&] { return entered; });
    }
    void release() {
        const std::lock_guard lock(mutex);
        block = false;
        changed.notify_all();
    }
    std::size_t count() {
        const std::lock_guard lock(mutex);
        return frames.size();
    }
};

struct Fixture {
    std::shared_ptr<Timeline> timeline = std::make_shared<Timeline>();
    std::shared_ptr<Consumer> consumer = std::make_shared<Consumer>();
    std::shared_ptr<std::vector<std::byte>> storage =
        std::make_shared<std::vector<std::byte>>(1024, std::byte{0xcc});
    std::shared_ptr<PresentationClock> clock;
    std::weak_ptr<Pin> last_pin;
    std::vector<std::int32_t> waited_fences;
    unsigned pins = 0;
    unsigned observations = 0;
    PresentationDependencies dependencies;
    services::GraphicsPresentation frame;
    std::shared_ptr<PresentationBinding> binding;

    Fixture(std::uint32_t pitch = 64, std::uint64_t surface_size = 512) {
        auto guest = create_guest_clock({[t = timeline] { return t->sample(); },
            [t = timeline](ClockInstant at, std::stop_token stop) { return t->wait(at, stop); }});
        clock = std::get<std::shared_ptr<PresentationClock>>(create_presentation_clock(
            std::get<std::shared_ptr<GuestClock>>(guest), {ClockInstant{}, 10ns}));
        frame.backing = {1, 4096, 0x10000, storage->size(), storage};
        storage->resize(512 + surface_size, std::byte{0xcc});
        frame.backing.size = storage->size();
        frame.width = 8; frame.height = 8; frame.pitch = pitch;
        frame.surface_offset = 512; frame.surface_size = surface_size;
        frame.auto_timestamp = true;
        frame.frame_number = 1;
        // Authored positions in the first half of a 64-byte by 8-row GOB.
        constexpr std::array<unsigned, 8> columns{0, 4, 8, 12, 32, 36, 40, 44};
        constexpr std::array<unsigned, 8> rows{0, 16, 64, 80, 128, 144, 192, 208};
        for (std::uint32_t y = 0; y < 8; ++y) {
            for (std::uint32_t x = 0; x < 8; ++x) {
                const auto offset = rows[y] + columns[x];
                (*storage)[512 + offset] = std::byte(x);
                (*storage)[513 + offset] = std::byte(y);
                (*storage)[514 + offset] = std::byte(x + y);
                (*storage)[515 + offset] = std::byte{0xff};
            }
        }
        dependencies.clock = clock;
        dependencies.presenter = consumer;
        dependencies.backing.lookup = [this](std::uint32_t id) {
            return id == frame.backing.export_id ? std::optional{frame.backing} : std::nullopt;
        };
        dependencies.backing.pin = [this](const services::GraphicsBufferBacking& backing,
            std::uint64_t offset, std::uint64_t size) -> LaunchResult<GuestMemoryPin> {
            ++pins;
            CHECK(backing.owner == storage);
            CHECK(offset == frame.surface_offset && size == frame.surface_size);
            auto pin = std::make_shared<Pin>(storage,
                horizon::AddressRange{backing.guest_address + offset, size}, offset);
            last_pin = pin;
            return GuestMemoryPin{pin};
        };
        dependencies.backing.wait_fence = [this](const services::GraphicsFence& fence,
            std::optional<ClockInstant> deadline, std::stop_token) -> LaunchStatus {
            CHECK(!last_pin.expired());
            CHECK(!deadline);
            waited_fences.push_back(fence.syncpoint);
            return std::monostate{};
        };
        dependencies.observe_frame = [this](const platform::PresentedFrame&) -> LaunchStatus {
            CHECK(!last_pin.expired());
            CHECK(consumer->thread == std::this_thread::get_id());
            ++observations;
            return std::monostate{};
        };
    }
    void bind() {
        binding = std::get<std::shared_ptr<PresentationBinding>>(bind_presentation(dependencies));
    }
    ~Fixture() {
        if (binding) {
            consumer->release();
            consumer->fail_drain = false;
            binding->request_stop();
            CHECK(std::holds_alternative<std::monostate>(binding->drain(std::nullopt)));
        }
    }
};

void crop_fences_worker_and_owned_pixels() {
    Fixture f;
    f.frame.crop = {2, 1, 6, 4};
    f.frame.fence_count = 2;
    f.frame.producer_fences[0] = {3, 10};
    f.frame.producer_fences[1] = {4, 20};
    f.bind();
    const auto caller = std::this_thread::get_id();
    CHECK(f.binding->callback()(f.frame).completed);
    CHECK(f.consumer->thread != caller);
    CHECK((f.waited_fences == std::vector<std::int32_t>{3, 4}));
    CHECK(f.pins == 1 && f.observations == 1 && f.last_pin.expired());
    const auto image = f.consumer->frames[0].pixels;
    CHECK(image->width == 4 && image->height == 3 && image->pixels.size() == 48);
    for (unsigned y = 0; y < 3; ++y) {
        for (unsigned x = 0; x < 4; ++x) {
            const auto offset = (y * 4 + x) * 4;
            CHECK(image->pixels[offset] == std::byte(x + 2));
            CHECK(image->pixels[offset + 1] == std::byte(y + 1));
            CHECK(image->pixels[offset + 2] == std::byte(x + y + 3));
            CHECK(image->pixels[offset + 3] == std::byte{0xff});
        }
    }
    std::fill(f.storage->begin(), f.storage->end(), std::byte{});
    CHECK(image->pixels[0] == std::byte{2});
    CHECK(f.timeline->deadlines.empty());
}

void padded_surfaces_preserve_pitch_and_pin_extent() {
    for (const auto pitch : {64u, 128u}) {
        Fixture f(pitch, 1024);
        f.bind();
        for (unsigned index = 0; index < 2; ++index) {
            f.frame.frame_number = index + 1;
            const auto result = f.binding->callback()(f.frame);
            CHECK(result.completed);
            CHECK(!f.binding->failure());
            if (!result.completed) continue;
            const auto& image = *f.consumer->frames.back().pixels;
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
            CHECK(f.last_pin.expired());
        }
        CHECK(f.pins == 2 && f.observations == 2);
    }
}

void authored_gobs_preserve_padded_stride_and_block_height() {
    Fixture f(192, 6272);
    f.frame.width = 24;
    f.frame.height = 24;
    f.frame.block_height_log2 = 1;
    f.frame.crop = {12, 6, 22, 21};
    std::fill(f.storage->begin(), f.storage->end(), std::byte{0xcc});
    // Six authored GOB origins: two visible columns, three visible rows.
    // The third (padding) column makes the next block row begin at byte 3072.
    constexpr std::array<std::array<unsigned, 2>, 3> origins{{
        {0, 1024}, {512, 1536}, {3072, 4096}}};
    constexpr std::array<unsigned, 16> columns{
        0, 4, 8, 12, 32, 36, 40, 44, 256, 260, 264, 268, 288, 292, 296, 300};
    constexpr std::array<unsigned, 8> rows{0, 16, 64, 80, 128, 144, 192, 208};
    for (unsigned y = 0; y < 24; ++y) {
        for (unsigned x = 0; x < 24; ++x) {
            const auto at = 512 + origins[y / 8][x / 16] + rows[y % 8] + columns[x % 16];
            (*f.storage)[at] = std::byte(x);
            (*f.storage)[at + 1] = std::byte(y);
            (*f.storage)[at + 2] = std::byte(x + y);
            (*f.storage)[at + 3] = std::byte{0xff};
        }
    }
    f.bind();
    const auto result = f.binding->callback()(f.frame);
    CHECK(result.completed);
    if (!result.completed) return;
    const auto& image = *f.consumer->frames.back().pixels;
    CHECK(image.width == 10 && image.height == 15);
    for (unsigned y = 0; y < 15; ++y) {
        for (unsigned x = 0; x < 10; ++x) {
            const auto at = (y * 10 + x) * 4;
            CHECK(image.pixels[at] == std::byte(x + 12));
            CHECK(image.pixels[at + 1] == std::byte(y + 6));
            CHECK(image.pixels[at + 2] == std::byte(x + y + 18));
            CHECK(image.pixels[at + 3] == std::byte{0xff});
        }
    }
    CHECK(f.pins == 1 && f.last_pin.expired());
    CHECK((*f.storage)[512 + 6144] == std::byte{0xcc});
}

void producer_completion_precedes_decode() {
    Fixture f;
    std::mutex fence_mutex;
    std::condition_variable_any fence_changed;
    bool waiting = false;
    bool completed = false;
    f.frame.fence_count = 1;
    f.dependencies.backing.wait_fence = [&](const auto&, auto, std::stop_token stop) -> LaunchStatus {
        std::unique_lock lock(fence_mutex);
        waiting = true;
        fence_changed.notify_all();
        if (!fence_changed.wait(lock, stop, [&] { return completed; }))
            return error("producer cancelled", LaunchError::cancelled);
        return std::monostate{};
    };
    f.bind();
    auto queued = std::async(std::launch::async, [&] { return f.binding->callback()(f.frame); });
    {
        std::unique_lock lock(fence_mutex);
        CHECK(fence_changed.wait_for(lock, 2s, [&] { return waiting; }));
        CHECK(!f.last_pin.expired());
        CHECK(f.consumer->count() == 0);
        CHECK(queued.wait_for(0ms) == std::future_status::timeout);
        // A producer writes the final pixel before publishing its fence.
        (*f.storage)[512] = std::byte{0x7f};
        completed = true;
        fence_changed.notify_all();
    }
    CHECK(queued.get().completed);
    CHECK(f.consumer->frames[0].pixels->pixels[0] == std::byte{0x7f});
}

void pin_and_fence_failures_never_decode_or_present() {
    for (unsigned mode = 0; mode < 3; ++mode) {
        Fixture f;
        if (mode == 0) f.dependencies.backing.pin = [](const auto&, auto, auto) -> LaunchResult<GuestMemoryPin> {
            return error("pin failed");
        };
        else if (mode == 1) f.dependencies.backing.pin = [](const auto&, auto, auto) -> LaunchResult<GuestMemoryPin> {
            return GuestMemoryPin{};
        };
        else {
            f.frame.fence_count = 2;
            f.dependencies.backing.wait_fence = [](const auto&, auto, auto) -> LaunchStatus {
                return error("fence failed");
            };
        }
        f.bind();
        CHECK(!f.binding->callback()(f.frame).completed);
        CHECK(f.consumer->count() == 0 && f.observations == 0);
        CHECK(f.binding->failure().has_value());
        if (mode == 2) CHECK(!f.last_pin.expired());
        CHECK(std::holds_alternative<std::monostate>(f.binding->drain(std::nullopt)));
        CHECK(f.last_pin.expired());
    }
}

void failure_retains_pin_and_mapping_until_real_drain() {
    for (unsigned mode = 0; mode < 4; ++mode) {
        Fixture f;
        f.consumer->fail_present = mode == 0;
        f.consumer->omit_readback = mode == 1;
        f.consumer->throw_present = mode == 2;
        if (mode == 3) f.dependencies.observe_frame = [](const auto&) -> LaunchStatus {
            return error("observer failed");
        };
        f.bind();
        CHECK(!f.binding->callback()(f.frame).completed);
        CHECK(!f.last_pin.expired());
        const auto first = f.binding->failure();
        CHECK(first.has_value());
        f.consumer->fail_drain = true;
        CHECK(std::holds_alternative<LaunchFailure>(f.binding->drain(std::nullopt)));
        CHECK(!f.last_pin.expired());
        CHECK(f.binding->failure()->operation == first->operation);
        CHECK(!f.binding->callback()(f.frame).completed);
        CHECK(f.pins == 1);
        f.consumer->fail_drain = false;
        CHECK(std::holds_alternative<std::monostate>(f.binding->drain(std::nullopt)));
        CHECK(f.last_pin.expired());
        CHECK(f.binding->failure()->operation == first->operation);
    }
}

void pacing_and_display_events_are_distinct() {
    Fixture f;
    services::vi::VsyncSource vsync(f.clock);
    bool created = false;
    const auto event = vsync.acquire(1, created);
    CHECK(created && f.timeline->waiting(1));
    f.frame.swap_interval = 2;
    f.consumer->block = true;
    f.bind();
    auto queued = std::async(std::launch::async, [&] { return f.binding->callback()(f.frame); });
    CHECK(f.timeline->waiting(2));
    CHECK(f.timeline->target(0) == ClockInstant{10ns});
    CHECK(f.timeline->target(1) == ClockInstant{20ns});
    f.timeline->advance(10);
    CHECK(f.timeline->waiting(3));
    CHECK(event->is_signaled());
    CHECK(event->reset());
    CHECK(queued.wait_for(0ms) == std::future_status::timeout);
    CHECK(f.consumer->count() == 0);
    f.timeline->advance(20);
    CHECK(f.consumer->wait_entered());
    CHECK(f.timeline->waiting(4));
    CHECK(event->is_signaled());
    CHECK(event->reset());
    CHECK(queued.wait_for(0ms) == std::future_status::timeout);
    f.consumer->release();
    CHECK(queued.get().completed);
    CHECK(!event->is_signaled());
    // Completing a queue does not signal display vsync. The next target skips
    // the missed refreshes instead of adding a second host sleep.
    f.timeline->advance(55);
    CHECK(f.timeline->waiting(5));
    CHECK(event->reset());
    f.frame.frame_number = 2;
    auto late = std::async(std::launch::async, [&] { return f.binding->callback()(f.frame); });
    CHECK(f.timeline->waiting(6));
    CHECK(f.timeline->target(5) == ClockInstant{60ns});
    f.timeline->advance(60);
    CHECK(late.get().completed);
    CHECK(f.observations == 2);
}

void cancellation_and_bounded_drain_keep_active_work_owned() {
    {
        Fixture f;
        f.frame.swap_interval = 1;
        f.bind();
        auto queued = std::async(std::launch::async, [&] { return f.binding->callback()(f.frame); });
        CHECK(f.timeline->waiting(1));
        f.binding->request_stop();
        CHECK(!queued.get().completed);
        CHECK(f.binding->failure()->code == LaunchError::cancelled);
        CHECK(f.consumer->count() == 0);
    }
    {
        Fixture f;
        f.consumer->block = true;
        f.bind();
        auto queued = std::async(std::launch::async, [&] { return f.binding->callback()(f.frame); });
        CHECK(f.consumer->wait_entered());
        CHECK(std::holds_alternative<LaunchFailure>(f.binding->drain(std::chrono::steady_clock::now())));
        CHECK(!f.last_pin.expired());
        CHECK(queued.wait_for(0ms) == std::future_status::timeout);
        f.consumer->release();
        CHECK(queued.get().completed);
        CHECK(std::holds_alternative<std::monostate>(f.binding->drain(std::nullopt)));
    }
}

void stopped_submission_after_drain_does_not_retain_owner() {
    Fixture f;
    f.bind();
    auto delayed_callback = f.binding->callback();
    std::weak_ptr<PresentationBinding> owner = f.binding;
    f.binding->request_stop();
    CHECK(std::holds_alternative<std::monostate>(f.binding->drain(std::nullopt)));
    // An admitted VI request may reach its callback only after the first drain.
    CHECK(!delayed_callback(f.frame).completed);
    CHECK(f.consumer->count() == 0);
    delayed_callback = {};
    f.binding.reset();
    CHECK(owner.expired());
    // Keep a failing regression safe to run repeatedly without leaking a worker.
    if (auto retained = owner.lock())
        CHECK(std::holds_alternative<std::monostate>(retained->drain(std::nullopt)));
}

void late_drain_success_requires_acknowledged_retry() {
    Fixture f;
    f.consumer->block = true;
    f.consumer->fail_present = true;
    f.bind();
    auto queued = std::async(std::launch::async, [&] { return f.binding->callback()(f.frame); });
    CHECK(f.consumer->wait_entered());
    CHECK(std::holds_alternative<LaunchFailure>(f.binding->drain(std::chrono::steady_clock::now() + 1ms)));
    f.consumer->release();
    CHECK(!queued.get().completed);
    // The abandoned drain will eventually return successful consumer drainage.
    // A following failed drain is a barrier proving that job finished, while
    // neither unsuccessful caller is allowed to retire the guest pin.
    {
        std::unique_lock lock(f.consumer->mutex);
        f.consumer->changed.wait_for(lock, 2s, [&] { return f.consumer->drains != 0; });
        CHECK(f.consumer->drains != 0);
        f.consumer->fail_drain = true;
    }
    CHECK(std::holds_alternative<LaunchFailure>(f.binding->drain(std::nullopt)));
    CHECK(!f.last_pin.expired());
    f.consumer->fail_drain = false;
    CHECK(std::holds_alternative<std::monostate>(f.binding->drain(std::nullopt)));
    CHECK(f.last_pin.expired());
}

void failed_owner_survives_for_acknowledged_drain_retry() {
    Fixture f;
    f.consumer->fail_present = true;
    f.bind();
    auto callback = f.binding->callback();
    CHECK(!callback(f.frame).completed);
    std::weak_ptr<PresentationBinding> retained = f.binding;
    callback = {};
    f.binding.reset();
    CHECK(!retained.expired());
    CHECK(!f.last_pin.expired());
    auto retry = retained.lock();
    CHECK(retry != nullptr);
    f.consumer->fail_drain = true;
    CHECK(std::holds_alternative<LaunchFailure>(retry->drain(std::nullopt)));
    retry.reset();
    CHECK(!retained.expired());
    retry = retained.lock();
    f.consumer->fail_drain = false;
    CHECK(std::holds_alternative<std::monostate>(retry->drain(std::nullopt)));
    CHECK(f.last_pin.expired());
    retry.reset();
    CHECK(retained.expired());
}

void invalid_metadata_fails_before_pinning() {
    for (unsigned mode = 0; mode < 9; ++mode) {
        Fixture f;
        switch (mode) {
        case 0: f.frame.transform = 1; break;
        case 1: f.frame.swap_interval = -1; break;
        case 2: f.frame.crop = {1, 1, 1, 2}; break;
        case 3: f.frame.surface_size = 513; break;
        case 4: f.frame.pitch = 32; break;
        case 5: f.frame.fence_count = 5; break;
        case 6: f.frame.block_height_log2 = 6; break;
        case 7: f.frame.auto_timestamp = false; break;
        case 8: f.frame.sticky_transform = 1; break;
        }
        f.bind();
        CHECK(!f.binding->callback()(f.frame).completed);
        CHECK(f.pins == 0 && f.consumer->count() == 0);
    }
    CHECK(std::holds_alternative<LaunchFailure>(bind_presentation({})));
}

// A VI native frame: a producer texture with on-demand readback and no guest
// surface. Reads count every CPU readback the bridge asks the producer for.
struct NativeFrame {
    std::shared_ptr<const void> texture = std::make_shared<const int>(0);
    std::shared_ptr<std::atomic<unsigned>> reads = std::make_shared<std::atomic<unsigned>>(0);
    void apply(services::GraphicsPresentation& frame) const {
        frame = {};
        frame.native_texture = texture;
        frame.native_texture_serial = 7;
        frame.native_vsync_paced = true;
        frame.frame_number = 5;
        frame.width = 8;
        frame.height = 6;
        frame.crop = {1, 2, 5, 6};
        frame.native_read_pixels = [reads = reads](std::array<std::int32_t, 4> crop)
            -> graphics::Rgba8ReadbackResult {
            ++*reads;
            graphics::Rgba8Image image;
            image.width = static_cast<std::uint32_t>(crop[2] - crop[0]);
            image.height = static_cast<std::uint32_t>(crop[3] - crop[1]);
            image.pixels.assign(std::size_t{image.width} * image.height * 4, std::byte{0x5a});
            return image;
        };
    }
};

void native_frames_read_back_only_when_wanted() {
    // 0: the predicate declines; 1: it asks for the frame; 2: no predicate.
    for (unsigned mode = 0; mode < 3; ++mode) {
        Fixture f;
        const NativeFrame native;
        native.apply(f.frame);
        std::vector<std::uint64_t> asked;
        if (mode != 2) f.dependencies.wants_readback = [&asked, mode](std::uint64_t frame_number) {
            asked.push_back(frame_number);
            return mode == 1;
        };
        unsigned observed = 0;
        bool observed_readback = false;
        f.dependencies.observe_frame = [&](const platform::PresentedFrame& completed) -> LaunchStatus {
            ++observed;
            observed_readback = completed.readback.has_value();
            return std::monostate{};
        };
        f.bind();
        CHECK(f.binding->callback()(f.frame).completed);
        CHECK(!f.binding->failure());
        CHECK(f.pins == 0 && f.waited_fences.empty() && f.timeline->deadlines.empty());
        CHECK(asked == (mode == 2 ? std::vector<std::uint64_t>{} : std::vector<std::uint64_t>{5}));
        CHECK(observed == 1);
        CHECK(f.consumer->count() == 1);
        if (f.consumer->count() != 1) continue;
        const auto& presented = f.consumer->frames.back();
        CHECK(presented.queued.native_texture == native.texture);
        if (mode == 0) {
            // The consumer gets the texture alone and owes no pixels.
            CHECK(*native.reads == 0);
            CHECK(!presented.readback && !presented.pixels);
            CHECK(!observed_readback);
        } else {
            CHECK(*native.reads == 1);
            CHECK(presented.readback && presented.pixels);
            if (presented.pixels)
                CHECK(presented.pixels->width == 4 && presented.pixels->height == 4);
            CHECK(observed_readback);
        }
    }
    // A frame whose pixels are wanted, by the predicate or by default, still
    // requires the consumer's matching readback.
    for (const bool predicate : {true, false}) {
        Fixture f;
        const NativeFrame native;
        native.apply(f.frame);
        if (predicate) f.dependencies.wants_readback = [](std::uint64_t) { return true; };
        f.dependencies.observe_frame = [](const platform::PresentedFrame&) -> LaunchStatus {
            return std::monostate{};
        };
        f.consumer->omit_readback = true;
        f.bind();
        CHECK(!f.binding->callback()(f.frame).completed);
        CHECK(*native.reads == 1);
        CHECK(f.binding->failure() && f.binding->failure()->operation == "presenter.complete");
    }
    // A capture every third frame: only those frames are read back, on one
    // binding, and the others reach the consumer as textures alone.
    {
        Fixture f;
        const NativeFrame native;
        native.apply(f.frame);
        f.dependencies.wants_readback = [](std::uint64_t frame_number) { return frame_number % 3 == 0; };
        std::vector<std::uint64_t> read_frames;
        f.dependencies.observe_frame = [&](const platform::PresentedFrame& completed) -> LaunchStatus {
            if (completed.readback) read_frames.push_back(completed.frame_number);
            return std::monostate{};
        };
        f.bind();
        for (std::uint64_t frame_number = 1; frame_number <= 6; ++frame_number) {
            f.frame.frame_number = frame_number;
            CHECK(f.binding->callback()(f.frame).completed);
        }
        CHECK(!f.binding->failure());
        CHECK(*native.reads == 2);
        CHECK(read_frames == (std::vector<std::uint64_t>{3, 6}));
        CHECK(f.consumer->count() == 6);
        for (std::size_t index = 0; index < f.consumer->count() && index < 6; ++index) {
            const auto& presented = f.consumer->frames[index];
            const bool wanted = (index + 1) % 3 == 0;
            CHECK(bool(presented.readback) == wanted && bool(presented.pixels) == wanted);
        }
    }
    // An unread frame keeps the native metadata checks and fails before the
    // consumer sees it.
    for (unsigned mode = 0; mode < 5; ++mode) {
        Fixture f;
        const NativeFrame native;
        native.apply(f.frame);
        switch (mode) {
        case 0: f.frame.transform = 1; break;
        case 1: f.frame.fence_count = 1; break;
        case 2: f.frame.crop = {0, 0, 9, 6}; break;
        case 3: f.frame.native_texture_serial = 0; break;
        case 4: f.frame.backing = {1, 4096, 0x10000, f.storage->size(), f.storage}; break;
        }
        f.dependencies.wants_readback = [](std::uint64_t) { return false; };
        f.bind();
        CHECK(!f.binding->callback()(f.frame).completed);
        CHECK(f.consumer->count() == 0 && *native.reads == 0 && f.observations == 0);
    }
}

} // namespace

int main() {
    crop_fences_worker_and_owned_pixels();
    padded_surfaces_preserve_pitch_and_pin_extent();
    authored_gobs_preserve_padded_stride_and_block_height();
    producer_completion_precedes_decode();
    pin_and_fence_failures_never_decode_or_present();
    failure_retains_pin_and_mapping_until_real_drain();
    pacing_and_display_events_are_distinct();
    cancellation_and_bounded_drain_keep_active_work_owned();
    invalid_metadata_fails_before_pinning();
    failed_owner_survives_for_acknowledged_drain_retry();
    late_drain_success_requires_acknowledged_retry();
    stopped_submission_after_drain_does_not_retain_owner();
    native_frames_read_back_only_when_wanted();
    std::fprintf(stderr, "Presentation failures: %u\n", failures.load());
    return failures == 0 ? 0 : 1;
}
