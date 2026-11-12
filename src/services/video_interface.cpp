// ============================================================================
// Armigo Services - Initial Video Interface Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/common/diagnostics.h>
#include <armigo/services/video_interface.h>
#include <armigo/nvn/timeline.h>

#include <armigo/horizon/event.h>
#include <armigo/platform/presentation_thread.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <bit>
#include <cstring>
#include <cstdio>
#include <limits>
#include <optional>
#include <string_view>

#include <armigo/compat/patches.h>
#include <unordered_map>
#include <utility>
#include <variant>

namespace armigo::services::vi {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

using namespace horizon::cmif;


// A reply that hands the caller one handle. The client materialises its display from
// that handle, so the special header announces one copied handle and the value sits in
// the descriptor word the client reads.
// A reply that relaxes nothing but transfers the display object itself: the dispatcher
// inserts it into the caller's handle table and builds the moved-handle layout.
IpcReply display_reply(std::shared_ptr<DisplayRecord> display) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 32;
    write(reply.bytes, 4, 4);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 28, 0x5c1d2e88);
    reply.transferred[0] = std::move(display);
    reply.transferred_count = 1;
    return reply;
}

std::uint32_t parcel_word(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    if (offset <= bytes.size() && bytes.size() - offset >= 4)
        std::memcpy(&value, bytes.data() + offset, 4);
    return value;
}

std::optional<std::size_t> parcel_arguments(std::span<const std::byte> input) {
    constexpr std::string_view interface = "android.gui.IGraphicBufferProducer";
    if (input.size() < 24 || parcel_word(input, 4) != 16 || parcel_word(input, 8) != 0)
        return std::nullopt;
    const std::size_t data_size = parcel_word(input, 0);
    if (data_size > input.size() - 16) return std::nullopt;
    const std::size_t data_end = 16 + data_size;
    const std::size_t count = parcel_word(input, 20);
    if (count != interface.size() || count > (std::numeric_limits<std::size_t>::max() - 30) / 2)
        return std::nullopt;
    const std::size_t arguments = (24 + (count + 1) * 2 + 3) & ~std::size_t{3};
    if (arguments > data_end) return std::nullopt;
    for (std::size_t index = 0; index < count; ++index) {
        if (std::to_integer<unsigned char>(input[24 + index * 2]) !=
                static_cast<unsigned char>(interface[index]) || input[25 + index * 2] != std::byte{})
            return std::nullopt;
    }
    if (input[24 + count * 2] != std::byte{} || input[25 + count * 2] != std::byte{})
        return std::nullopt;
    return arguments;
}
}  // namespace

struct DisplaySession::QueueState {
    explicit QueueState(OperationMode mode)
        : default_width(operation_mode_profile(mode).display_width),
          default_height(operation_mode_profile(mode).display_height) {}
    struct Slot {
        enum class State : std::uint8_t { Free, Dequeued, Queued, Acquired } state = State::Free;
        bool request_called = false;
        bool preallocated = false;
        std::uint64_t frame_number = 0;
        std::array<std::byte, 0x16c> flattened{};
        services::GraphicsBufferBacking backing{};
        std::optional<NativeTexture> native_texture;
    };
    std::mutex mutex;
    std::mutex consumer_mutex;
    services::GraphicsBufferBackingLookup lookup;
    services::GraphicsPresenter presenter;
    std::shared_ptr<horizon::Event> free_event = std::make_shared<horizon::Event>(false);
    std::array<Slot, 64> slots{};
    std::uint32_t maximum = 0;
    std::uint32_t default_width;
    std::uint32_t default_height;
    std::uint32_t default_format = 1;
    std::uint64_t frame_counter = 0;
    std::weak_ptr<NativeWindowProducerImpl> native_producer;
    bool native_registered = false;
    bool native_failed = false;
    // Native windows may present from only their first `active` textures (NVN
    // active textures); zero means every registered one.
    std::uint32_t active = 0;
    static constexpr std::uint32_t consumer_usage = 0x800;

    std::uint32_t dequeue_limit() const { return active ? std::min(active, maximum) : maximum; }

    void update_free_event() {
        if (native_registered && native_failed) { free_event->signal(); return; }
        bool ready = false;
        for (std::size_t index = 0; index < dequeue_limit(); ++index)
            ready = ready || slots[index].state == Slot::State::Free;
        if (ready) free_event->signal(); else static_cast<void>(free_event->reset());
    }
};

class NativeWindowProducerImpl final : public NativeWindowProducer {
public:
    NativeWindowProducerImpl(std::shared_ptr<DisplaySession::QueueState> queue,
                             std::shared_ptr<VsyncSource> source)
        : queue_(std::move(queue)), source_(std::move(source)) {}

    ~NativeWindowProducerImpl() override { detach(); }

    bool register_textures(std::span<const NativeTexture> textures) override {
        if (textures.size() < 2 || textures.size() > 4) return false;
        std::vector<NativeTexture> retained(textures.begin(), textures.end());
        for (std::size_t i = 0; i < retained.size(); ++i) {
            const auto& texture = retained[i];
            if (texture.serial == 0 || texture.width == 0 || texture.height == 0 ||
                texture.width > 16384 || texture.height > 16384 || !texture.read_pixels)
                return false;
            for (std::size_t j = 0; j < i; ++j)
                if (retained[j].serial == texture.serial) return false;
        }
        const std::lock_guard consumer_lock(queue_->consumer_mutex);
        const std::lock_guard queue_lock(queue_->mutex);
        if (!attached_ || queue_->native_registered || queue_->maximum != 0 ||
            queue_->native_producer.lock().get() != this)
            return false;
        for (const auto& slot : queue_->slots) {
            if (slot.state != DisplaySession::QueueState::Slot::State::Free ||
                slot.preallocated || slot.backing.owner || slot.native_texture)
                return false;
        }
        for (std::size_t index = 0; index < textures.size(); ++index) {
            auto& slot = queue_->slots[index];
            slot = {};
            slot.native_texture = std::move(retained[index]);
        }
        queue_->maximum = static_cast<std::uint32_t>(textures.size());
        nvn::timeline_event("slots", textures.size(), 0);
        queue_->native_registered = true;
        last_slot_.reset();
        queue_->update_free_event();
        return true;
    }

    bool set_active_textures(std::size_t count) override {
        const std::lock_guard queue_lock(queue_->mutex);
        if (!attached_ || !queue_->native_registered ||
            queue_->native_producer.lock().get() != this || count < 2 || count > queue_->maximum)
            return false;
        queue_->active = static_cast<std::uint32_t>(count);
        queue_->update_free_event();
        return true;
    }

    bool unregister_textures() override {
        if (!drain_presents(std::chrono::seconds(5)))
            return false;
        const std::lock_guard consumer_lock(queue_->consumer_mutex);
        const std::lock_guard queue_lock(queue_->mutex);
        if (!attached_ || !queue_->native_registered ||
            queue_->native_producer.lock().get() != this)
            return false;
        for (std::size_t index = 0; index < queue_->maximum; ++index) {
            const auto state = queue_->slots[index].state;
            if (state == DisplaySession::QueueState::Slot::State::Acquired ||
                state == DisplaySession::QueueState::Slot::State::Queued)
                return false;
        }
        for (std::size_t index = 0; index < queue_->maximum; ++index) {
            auto& slot = queue_->slots[index];
            if (slot.state == DisplaySession::QueueState::Slot::State::Dequeued)
                slot.state = DisplaySession::QueueState::Slot::State::Free;
        }
        clear_textures_locked();
        queue_->free_event->signal();
        return true;
    }

    NativeDequeue dequeue() override {
        const std::lock_guard queue_lock(queue_->mutex);
        if (!attached_ || !queue_->native_registered ||
            queue_->native_producer.lock().get() != this || queue_->native_failed || source_->failure())
            return {-2, 0};
        std::optional<std::size_t> selected;
        for (std::size_t index = 0; index < queue_->dequeue_limit(); ++index) {
            const auto& slot = queue_->slots[index];
            if (slot.state != DisplaySession::QueueState::Slot::State::Free ||
                !slot.native_texture)
                continue;
            if (!selected || slot.frame_number < queue_->slots[*selected].frame_number)
                selected = index;
        }
        if (!selected) return {-1, 0};
        nvn::timeline_event("dequeue", queue_->slots[*selected].frame_number,
                       std::uint64_t(*selected));
        queue_->slots[*selected].state =
            DisplaySession::QueueState::Slot::State::Dequeued;
        queue_->update_free_event();
        return {static_cast<std::int32_t>(*selected), 0};
    }

    // One queued frame. The presentation thread waits for the frame's GPU work,
    // then for the refresh, hands it to the presenter and frees the slot: a
    // present returns to the guest as soon as the frame is queued, as it does
    // natively, so the guest prepares the next frame while this one executes and
    // is shown.
    struct PendingPresent {
        NativeTexture texture;
        std::uint32_t slot = 0;
        std::uint64_t frame = 0;
        std::uint64_t fence = 0;
        std::function<bool(std::uint64_t)> wait_fence;
        std::array<std::int32_t, 4> crop{};
        int interval = 0;
    };

    // Refreshes a present waits for; see the display-link note in present().
    static int present_link_interval(int requested) {
        // Throughput profiling may bypass host pacing without changing VI events.
        if (const char* value = std::getenv("ARMIGO_VI_PACE");
            value && std::string_view(value) == "0")
            return 0;
        if (const char* value = std::getenv("ARMIGO_VI_PRESENT_LINK"); value && *value)
            return std::string_view(value) != "0" ? 1 : requested;
        return armigo::compat::present_link() ? 1 : requested;
    }

    std::chrono::nanoseconds scheduling_period(int requested) const override {
        const auto interval = present_link_interval(requested);
        return interval > 0 ? source_->interval() * interval : std::chrono::nanoseconds::zero();
    }

    std::chrono::nanoseconds guest_present_period(int requested) const override {
        // Native windows accept zero (immediate) or one through five refreshes.
        // Use the explicit cadence, not wall time that can include GPU stalls.
        return requested > 0 && requested <= 5
            ? source_->interval() * requested : std::chrono::nanoseconds::zero();
    }

    GraphicsPresentationResult present(
        std::int32_t requested_slot, std::uint64_t fence,
        std::function<bool(std::uint64_t)> wait_fence,
        std::array<std::int32_t, 4> crop, int interval) override {
        notify_guest_present();
        const std::lock_guard consumer_lock(queue_->consumer_mutex);
        std::unique_lock queue_lock(queue_->mutex);
        if (!attached_ || !queue_->native_registered ||
            queue_->native_producer.lock().get() != this)
            return failure("Native window producer is not bound");
        if (queue_->native_failed) return failure("Native presentation previously failed");
        if (requested_slot < 0 ||
            static_cast<std::size_t>(requested_slot) >= queue_->maximum)
            return failure("Native presentation slot is outside the active queue");
        auto& slot = queue_->slots[static_cast<std::size_t>(requested_slot)];
        if (slot.state != DisplaySession::QueueState::Slot::State::Dequeued ||
            !slot.native_texture)
            return failure("Native presentation requires a dequeued texture slot");
        const auto& registered = *slot.native_texture;
        const bool full_crop = crop == std::array<std::int32_t, 4>{};
        if (!full_crop &&
            (crop[0] < 0 || crop[1] < 0 || crop[2] <= crop[0] || crop[3] <= crop[1] ||
             static_cast<std::uint32_t>(crop[2]) > registered.width ||
             static_cast<std::uint32_t>(crop[3]) > registered.height))
            return failure("Native presentation crop is empty or outside the texture");
        if (interval < 0)
            return failure("Native presentation interval must be nonnegative");
        if (!wait_fence)
            return failure("Native presentation requires a fence wait callback");

        const std::uint64_t frame = ++queue_->frame_counter;
        slot.frame_number = frame;
        slot.state = DisplaySession::QueueState::Slot::State::Acquired;
        queue_->update_free_event();
        NativeTexture texture = registered;
        const auto presenter = queue_->presenter;
        queue_lock.unlock();

        const auto rollback = [&] {
            std::lock_guard lock(queue_->mutex);
            auto& failed = queue_->slots[static_cast<std::size_t>(requested_slot)];
            if (failed.state == DisplaySession::QueueState::Slot::State::Acquired && failed.frame_number == frame)
                failed.state = DisplaySession::QueueState::Slot::State::Dequeued;
            queue_->update_free_event();
        };

        if (!presenter) {
            rollback();
            return failure("Native presentation has no synchronous presenter");
        }
        if (texture.capture) {
            try {
                auto captured = texture.capture(fence);
                if (!captured || captured->serial != texture.serial || captured->width != texture.width ||
                    captured->height != texture.height || !captured->read_pixels) {
                    if (captured && captured->cancel) captured->cancel();
                    rollback();
                    return failure("Native frame capture could not be queued in submission order");
                }
                texture = std::move(*captured);
            } catch (...) {
                rollback();
                return failure("Native frame capture threw an exception");
            }
        }
        PendingPresent item;
        item.texture = texture;
        item.slot = static_cast<std::uint32_t>(requested_slot);
        item.frame = frame;
        item.fence = fence;
        item.wait_fence = std::move(wait_fence);
        item.crop = crop;
        item.interval = interval;
        {
            std::unique_lock present_lock(present_mutex_);
            if (stopping_) {
                present_lock.unlock();
                if (item.texture.cancel) item.texture.cancel();
                rollback();
                return failure("Native presentation is stopping");
            }
            pending_.push_back(std::move(item));
            ++in_flight_;
        }
        start_presentation_thread();
        present_changed_.notify_all();
        return {true, {}};
    }

    void present_queued(const PendingPresent& item, std::stop_token stop) {
        struct CancelUnconsumed {
            std::function<void()> cancel;
            ~CancelUnconsumed() { if (cancel) cancel(); }
        } cancel_unconsumed{item.texture.cancel};
        const auto queue = queue_;
        const bool ordered = item.texture.ordered_gpu_dependencies;
        const auto note_frame_failure = [this, ordered](const std::string& detail) {
            if (ordered) {
                const std::lock_guard lock(queue_->mutex);
                queue_->native_failed = true;
                queue_->update_free_event();
            }
            note_failure(detail);
        };
        struct ProducerCompletion {
            std::once_flag once;
            bool succeeded = false;
        };
        const auto cached_ready = [frame = item.frame](std::function<bool()> wait) {
            auto producer = std::make_shared<ProducerCompletion>();
            return std::function<bool()>([producer, wait = std::move(wait), frame] {
                std::call_once(producer->once, [&] {
                    nvn::timeline_event("pres_fence_wait_begin", frame, 0);
                    try { producer->succeeded = wait && wait(); } catch (...) {}
                    if (producer->succeeded) nvn::timeline_event("pres_fence_ready", frame, 0);
                });
                return producer->succeeded;
            });
        };
        std::function<bool()> producer_ready;
        if (!ordered) {
            producer_ready = cached_ready([wait = item.wait_fence, fence = item.fence] { return wait && wait(fence); });
            if (!producer_ready()) {
                note_frame_failure("the frame's GPU work did not complete");
                return;
            }
        }
        NativeTexture texture = item.texture;
        if (texture.resolve) {
            try {
                auto resolved = texture.resolve();
                if (!resolved || resolved->serial != texture.serial ||
                    resolved->width != texture.width || resolved->height != texture.height ||
                    !resolved->read_pixels || resolved->ordered_gpu_dependencies != ordered) {
                    note_frame_failure("the frame's current native texture allocation is unavailable");
                    return;
                }
                texture = std::move(*resolved);
            } catch (...) {
                note_frame_failure("the native texture allocation resolver threw an exception");
                return;
            }
        }
        if (ordered && (!texture.native_submission || !texture.ordered_producer_ready)) {
            note_frame_failure("the ordered native frame has no submission context or producer result");
            return;
        }
        if (ordered) producer_ready = cached_ready(texture.ordered_producer_ready);
        // Present pacer: a frame's slot is one refresh period after the previous
        // frame's slot. A frame that is ready early waits for its slot; a frame
        // that is late by less than one period presents at once and keeps the
        // cadence; a frame late by a whole period or more restarts the cadence
        // from now instead of presenting a burst to catch up.
        if (const auto clock = source_->imposed_clock()) {
            const auto planned = clock->schedule(last_imposed_refresh_, present_link_interval(item.interval));
            if (const auto* error = std::get_if<runtime::LaunchFailure>(&planned)) {
                note_frame_failure(error->detail);
                return;
            }
            const auto target = std::get<runtime::PresentationSchedule>(planned);
            // The guest clock ends this wait at the slot.
            const auto waited = clock->clock()->wait_until(target.deadline, stop);
            if (const auto* error = std::get_if<runtime::LaunchFailure>(&waited)) {
                note_frame_failure(error->detail);
                return;
            }
            if (std::get<runtime::ClockWait>(waited) == runtime::ClockWait::cancelled) return;
            last_imposed_refresh_ = target.refresh_index;
        } else if (const int link_interval = present_link_interval(item.interval); link_interval > 0) {
            const auto period = source_->interval() *
                static_cast<std::int64_t>(link_interval);
            const auto now = std::chrono::steady_clock::now();
            if (period > std::chrono::nanoseconds::zero()) {
                // The pacing thread declares the display period so its waits
                // end at the slot instead of a coalesced moment after it.
                if (declared_period_ != period) {
                    declared_period_ = period;
                    platform::declare_periodic_thread(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                        std::chrono::nanoseconds(2'000'000));
                }
                auto slot = last_slot_ ? *last_slot_ + period : now + period;
                // Following the window's display, the slot is the nearest
                // refresh of the grid, whose phase tracks the display's
                // vblanks; the late-frame rules below are unchanged.
                if (const auto aligned = source_->display_slot(slot)) slot = *aligned;
                if (now <= slot) {
                    nvn::timeline_event("pres_wait_slot", item.frame,
                                   std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       slot - now).count()));
                    platform::wait_until(slot);
                    last_slot_ = slot;
                } else {
                    last_slot_ = now - slot < period ? slot : now;
                }
            }
        }

        services::GraphicsPresentation presentation;
        presentation.native_read_pixels = texture.read_pixels;
        presentation.native_texture_serial = texture.serial;
        presentation.native_texture = texture.native_texture;
        presentation.native_submission = texture.native_submission;
        if (ordered) presentation.native_producer_ready = producer_ready;
        presentation.native_vsync_paced = true;
        presentation.slot = item.slot;
        presentation.frame_number = item.frame;
        presentation.width = texture.width;
        presentation.height = texture.height;
        presentation.auto_timestamp = true;
        presentation.crop = item.crop;
        presentation.swap_interval = item.interval;
        nvn::timeline_event("pres_present", item.frame, std::uint64_t(item.interval));
        try {
            const auto completed = queue->presenter(presentation);
            if (completed.completed && ordered && !producer_ready()) {
                note_frame_failure("the ordered frame's producer did not complete successfully");
                return;
            }
            if (completed.completed) nvn::timeline_event("pres_complete", item.frame, std::uint64_t(item.interval));
            if (!completed.completed && !completed.diagnostic.empty())
                note_frame_failure(completed.diagnostic);
        } catch (...) {
            note_frame_failure("the presenter threw an exception");
        }
    }

    void release_present_slot(const PendingPresent& item) {
        const std::lock_guard queue_lock(queue_->mutex);
        if (item.slot >= queue_->maximum) return;
        auto& slot = queue_->slots[item.slot];
        if (slot.state != DisplaySession::QueueState::Slot::State::Acquired ||
            slot.frame_number != item.frame || !slot.native_texture ||
            slot.native_texture->serial != item.texture.serial)
            return;
        slot.state = DisplaySession::QueueState::Slot::State::Free;
        nvn::timeline_event("slot_free", item.frame, std::uint64_t(item.slot));
        queue_->update_free_event();
    }

    void start_presentation_thread() {
        const std::lock_guard thread_lock(thread_mutex_);
        if (present_thread_.joinable()) return;
        present_thread_ = std::jthread([this](std::stop_token stop) {
            if (!platform::declare_interactive_thread())
                armigo::diag::write(armigo::diag::Level::error,
                    "Armigo vi: presentation QoS declaration failed; retaining inherited policy\n");
            for (;;) {
                PendingPresent item;
                {
                    std::unique_lock present_lock(present_mutex_);
                    present_changed_.wait(present_lock, [&] {
                        return stop.stop_requested() || stopping_ || !pending_.empty();
                    });
                    if (pending_.empty()) {
                        if (stop.stop_requested() || stopping_) return;
                        continue;
                    }
                    item = std::move(pending_.front());
                    pending_.pop_front();
                }
                present_queued(item, stop);
                release_present_slot(item);
                {
                    const std::lock_guard present_lock(present_mutex_);
                    if (in_flight_) --in_flight_;
                }
                present_changed_.notify_all();
            }
        });
    }

    // Waits until no frame is queued or being presented. Returns false when
    // they do not finish in time.
    bool drain_presents(std::chrono::nanoseconds limit) {
        std::unique_lock present_lock(present_mutex_);
        return present_changed_.wait_for(present_lock, limit, [&] {
            return pending_.empty() && in_flight_ == 0;
        });
    }

    void stop_presentation_thread() {
        {
            const std::lock_guard present_lock(present_mutex_);
            stopping_ = true;
        }
        present_changed_.notify_all();
        {
            const std::lock_guard thread_lock(thread_mutex_);
            if (present_thread_.joinable()) present_thread_.request_stop();
        }
        present_changed_.notify_all();
        const std::lock_guard thread_lock(thread_mutex_);
        if (present_thread_.joinable()) present_thread_.join();
    }

    void note_failure(const std::string& detail) {
        if (detail.empty()) return;
        armigo::diag::write(armigo::diag::Level::error,  "Armigo vi: native presentation: %s\n", detail.c_str());
    }

    std::shared_ptr<horizon::Event> free_event() const override {
        return queue_->free_event;
    }

    void detach() {
        stop_presentation_thread();
        const std::lock_guard consumer_lock(queue_->consumer_mutex);
        const std::lock_guard queue_lock(queue_->mutex);
        if (!attached_) return;
        attached_ = false;
        if (queue_->native_producer.lock().get() != this) return;
        bool acquired = false;
        for (std::size_t index = 0; index < queue_->maximum; ++index) {
            auto& slot = queue_->slots[index];
            if (slot.state == DisplaySession::QueueState::Slot::State::Dequeued) {
                slot.state = DisplaySession::QueueState::Slot::State::Free;
                slot.frame_number = 0;
            }
            acquired = acquired ||
                slot.state == DisplaySession::QueueState::Slot::State::Acquired;
        }
        if (!acquired && queue_->native_registered) clear_textures_locked();
        if (queue_->native_producer.lock().get() == this)
            queue_->native_producer.reset();
        queue_->free_event->signal();
    }

private:
    static GraphicsPresentationResult failure(std::string detail) {
        return {false, std::move(detail)};
    }

    void clear_textures_locked() {
        for (std::size_t index = 0; index < queue_->maximum; ++index)
            queue_->slots[index] = {};
        queue_->maximum = 0;
        queue_->active = 0;
        queue_->native_registered = false;
        queue_->native_failed = false;
        queue_->update_free_event();
    }

    std::shared_ptr<DisplaySession::QueueState> queue_;
    std::shared_ptr<VsyncSource> source_;
    std::optional<std::chrono::steady_clock::time_point> last_slot_;
    std::optional<std::uint64_t> last_imposed_refresh_;
    // The period the presentation thread last declared to the host scheduler.
    std::chrono::steady_clock::duration declared_period_{};
    bool attached_ = true;
    std::mutex present_mutex_;
    std::condition_variable present_changed_;
    std::deque<PendingPresent> pending_;
    unsigned in_flight_ = 0;
    bool stopping_ = false;
    std::mutex thread_mutex_;
    std::jthread present_thread_;
};

namespace {
std::mutex native_windows_mutex;
std::unordered_map<void*, std::shared_ptr<NativeWindowProducerImpl>> native_windows;
}  // namespace

DisplaySession::DisplaySession(horizon::HandleTable& handles, OperationMode operation_mode)
    : handles_(&handles), operation_mode_(operation_mode),
      queue_(std::make_shared<QueueState>(operation_mode)) {}

IpcReply DisplaySession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply DisplaySession::request_with_buffers(const IpcMessage& message, std::uint64_t,
                                              const horizon::GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    buffers_ = &buffers;
    IpcReply reply = request_locked(message);
    buffers_ = nullptr;
    return reply;
}

IpcReply DisplaySession::request(const IpcMessage& message, std::uint64_t) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    return request_locked(message);
}

IpcReply DisplaySession::request_locked(const IpcMessage& message) {
    const auto type = read(message, 0);
    if (type == 5 || type == 7) {
        if (read(message, 16) != sfci) return unsupported("Display control magic is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_reply(0, 2);
        if (command == 0) {
            if (converted_) return unsupported("Repeated display domain conversion is unsupported");
            converted_ = true;
            return control_reply(1, 4);
        }
        if (command == 2) {
            IpcReply reply = control_reply(0, 4);
            auto clone = std::make_shared<DisplaySession>(*handles_, operation_mode_);
            clone->converted_ = converted_;
            clone->root_ = root_;
            clone->application_display_ = application_display_;
            clone->displays_ = displays_;
            clone->session_handle_ = session_handle_;
            clone->selector_ = selector_;
            clone->service_handle_ = service_handle_;
            clone->display_names_ = display_names_;
            clone->display_id_ = display_id_;
            clone->next_display_id_ = next_display_id_;
            clone->vsync_ = vsync_;
            clone->queue_ = queue_;
            reply.transferred[0] = std::move(clone);
            reply.transferred_count = 1;
            return reply;
        }
        return unsupported("Display control command is unsupported");
    }
    // The message type is the leading halfword; the rest of the first word carries
    // the HIPC descriptor counts, so comparing the whole word rejects those requests.
    if ((type & 0xffffu) != 4 && (type & 0xffffu) != 6)
        return unsupported("Display request shape is unsupported");
    std::size_t cmif = 0;
    if (read(message, 48) == sfci) cmif = 48;
    else if (read(message, 32) == sfci) cmif = 32;
    else if (read(message, 16) == sfci) cmif = 16;
    else return unsupported("Display request shape is unsupported");
    const auto command = read(message, cmif + 8);
    // The display service the root session transfers answers the binder commands, which are
    // numbered from zero; the root session itself never sees them.
    if (!root_ && command <= 3) {
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = static_cast<std::size_t>(cmif + 16);
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, cmif, sfco);
        write(reply.bytes, cmif + 4, 1);
        if (command == 1) return reply;  // AdjustRefcount: the count is host state.
        if (command == 2) {
            // GetNativeHandle: a readable event the producer signals when a buffer returns.
            const std::lock_guard queue_lock(queue_->mutex);
            reply.transferred[0] = queue_->free_event;
            reply.transferred_count = 1;
            return reply;
        }
        if (buffers_ == nullptr || buffers_->size() < 2)
            return unsupported("Binder transact needs the caller's two buffers");
        std::array<std::byte, 0x200> input{};
        std::array<std::byte, 0x200> output{};
        const auto input_size = std::min<std::size_t>((*buffers_)[0].size, input.size());
        if (input_size == 0 ||
            !buffers_->read(0, std::span<std::byte>(input).first(input_size)))
            return unsupported("Binder transact input buffer is empty or unreadable");
        const auto put = [&](std::size_t offset, std::uint32_t word) {
            for (unsigned index = 0; index != 4; ++index)
                output[offset + index] = static_cast<std::byte>((word >> (8 * index)) & 255);
        };
        const auto finish = [&](std::uint32_t length) {
            if (length > output.size()) return unsupported("Binder transact reply exceeds its bound");
            reply.size = 40;
            write(reply.bytes, 4, (reply.size - 8) / 4);
            write(reply.bytes, 16, sfco);
            write(reply.bytes, 20, 1);
            write(reply.bytes, 24, 0);
            write(reply.bytes, 28, 0);
            write(reply.bytes, 32, length);
            if (!buffers_->write(1, std::span<const std::byte>(output).first(length)))
                return unsupported("Binder transact could not fill the reply buffer");
            return reply;
        };
        put(0x04, 0x10);   // data offset
        put(0x08, 0x00);   // no object table entries
        if (read(message, cmif + 16) != 1)
            return unsupported("Only producer Binder object 1 is supported");
        const auto transaction = read(message, cmif + 20);
        const auto arguments = parcel_arguments(std::span<const std::byte>(input).first(input_size));
        if (!arguments) return unsupported("Binder producer parcel header or interface token is invalid");
        const std::size_t parcel_size = 16 + parcel_word(input, 0);
        switch (transaction) {
        case 10: {         // Connect: queue geometry and pending buffers, then the status.
            const std::lock_guard queue_lock(queue_->mutex);
            put(0x00, 0x14);
            put(0x0c, 0x24);
            put(0x10, queue_->default_width);
            put(0x14, queue_->default_height);
            put(0x18, 0);
            put(0x1c, 0);
            put(0x20, 0);
            return finish(0x24);
        }
        case 9: {          // Query: only values the layer description fixes are answered.
            if (*arguments > parcel_size || parcel_size - *arguments < 4)
                return unsupported("Binder producer query argument is truncated");
            std::uint32_t what = 0;
            std::memcpy(&what, input.data() + *arguments, 4);
            std::int32_t value = 0;
            bool known = true;
            const std::lock_guard queue_lock(queue_->mutex);
            switch (what) {
            case 0: value = static_cast<std::int32_t>(queue_->default_width); break;
            case 1: value = static_cast<std::int32_t>(queue_->default_height); break;
            case 2: value = static_cast<std::int32_t>(queue_->default_format); break;
            case 3: value = 2; break;     // MinUndequeedBuffers
            case 8: value = 0; break;     // TransformHint
            case 10: value = static_cast<std::int32_t>(QueueState::consumer_usage); break;
            case 12: value = 0; break;    // DefaultDataSpace
            default: known = false; break;
            }
            if (!known) return unsupported("Binder producer query is not implemented");
            put(0x00, 0x08);
            put(0x0c, 0x18);
            put(0x10, static_cast<std::uint32_t>(value));
            put(0x14, 0);
            return finish(0x18);
        }
        case 3: {          // DequeueBuffer: one slot and no fence.
            if (*arguments > parcel_size || parcel_size - *arguments < 20)
                return unsupported("DequeueBuffer arguments are truncated");
            const auto async = parcel_word(input, *arguments);
            auto width = parcel_word(input, *arguments + 4);
            auto height = parcel_word(input, *arguments + 8);
            auto format = parcel_word(input, *arguments + 12);
            auto usage = parcel_word(input, *arguments + 16) | QueueState::consumer_usage;
            const auto dequeue_reply = [&](std::int32_t slot, std::int32_t status) {
                put(0x00, 0x38); put(0x0c, 0x48);
                put(0x10, static_cast<std::uint32_t>(slot));
                put(0x14, 1); put(0x18, 0x24); put(0x1c, 0);
                put(0x44, static_cast<std::uint32_t>(status));
                return finish(0x48);
            };
            if (async != 0) return unsupported("Asynchronous dequeue is not implemented");
            if ((width == 0) != (height == 0)) return dequeue_reply(-1, -22);
            const std::lock_guard consumer_lock(queue_->consumer_mutex);
            const std::lock_guard queue_lock(queue_->mutex);
            if (queue_->native_registered)
                return unsupported("Binder dequeue cannot access native texture slots");
            if (width == 0) { width = queue_->default_width; height = queue_->default_height; }
            if (format == 0) format = queue_->default_format;
            std::optional<std::size_t> selected;
            for (std::size_t index = 0; index < queue_->maximum; ++index) {
                const auto& slot = queue_->slots[index];
                if (slot.state != QueueState::Slot::State::Free) continue;
                if (!selected || slot.frame_number < queue_->slots[*selected].frame_number)
                    selected = index;
            }
            if (!selected) return unsupported("Synchronous dequeue has no free preallocated slot");
            auto& slot = queue_->slots[*selected];
            if (!slot.preallocated || !slot.backing.owner)
                return dequeue_reply(-1, -12);
            const auto& flat = slot.flattened;
            if (parcel_word(flat, 4) != width || parcel_word(flat, 8) != height ||
                parcel_word(flat, 0x10) != format ||
                (usage & ~parcel_word(flat, 0x14)) != 0)
                return dequeue_reply(static_cast<std::int32_t>(*selected), -19);
            auto reply = dequeue_reply(static_cast<std::int32_t>(*selected), 0);
            if (reply.disposition == IpcDisposition::handled) {
                slot.state = QueueState::Slot::State::Dequeued;
                queue_->update_free_event();
            }
            return reply;
        }
        case 1: {          // RequestBuffer: describe the slot's graphic buffer.
            if (*arguments > parcel_size || parcel_size - *arguments < 4)
                return unsupported("RequestBuffer slot is truncated");
            const auto index = parcel_word(input, *arguments);
            const std::lock_guard consumer_lock(queue_->consumer_mutex);
            const std::lock_guard queue_lock(queue_->mutex);
            if (queue_->native_registered)
                return unsupported("RequestBuffer cannot access native texture slots");
            if (index >= queue_->slots.size() ||
                queue_->slots[index].state != QueueState::Slot::State::Dequeued)
                return unsupported("RequestBuffer requires a dequeued slot");
            auto& slot = queue_->slots[index];
            put(0x00, 0x17c); put(0x0c, 0x18c);
            put(0x10, 1); put(0x14, 0x16c); put(0x18, 0);
            std::copy(slot.flattened.begin(), slot.flattened.end(), output.begin() + 0x1c);
            put(0x188, 0);
            auto reply = finish(0x18c);
            if (reply.disposition == IpcDisposition::handled) slot.request_called = true;
            return reply;
        }
        case 7: {
            if (*arguments > parcel_size || parcel_size - *arguments < 96)
                return unsupported("QueueBuffer arguments are truncated");
            const auto index = parcel_word(input, *arguments);
            if (parcel_word(input, *arguments + 4) != 84 ||
                parcel_word(input, *arguments + 8) != 0)
                return unsupported("QueueBuffer flattened input is unsupported");
            const std::size_t body = *arguments + 12;
            const auto signed_word = [&](std::size_t offset) {
                return std::bit_cast<std::int32_t>(parcel_word(input, body + offset));
            };
            const auto body_u64 = [&](std::size_t offset) {
                return std::uint64_t{parcel_word(input, body + offset)} |
                       (std::uint64_t{parcel_word(input, body + offset + 4)} << 32);
            };
            const auto fence_count = parcel_word(input, body + 0x30);
            const auto scaling = parcel_word(input, body + 0x1c);
            const auto async = parcel_word(input, body + 0x28);
            if (fence_count > 4 || scaling > 4 || async != 0)
                return unsupported("QueueBuffer fence, scaling or async mode is unsupported");
            const std::array<std::int32_t, 4> crop{
                signed_word(0x0c), signed_word(0x10), signed_word(0x14), signed_word(0x18)};

            const std::lock_guard consumer_lock(queue_->consumer_mutex);
            std::unique_lock queue_lock(queue_->mutex);
            if (queue_->native_registered)
                return unsupported("QueueBuffer cannot access native texture slots");
            if (!queue_->presenter)
                return unsupported("QueueBuffer has no synchronous presenter");
            if (index >= queue_->maximum || index >= queue_->slots.size())
                return unsupported("QueueBuffer slot is outside the active queue");
            auto& slot = queue_->slots[index];
            if (slot.state != QueueState::Slot::State::Dequeued || !slot.request_called ||
                !slot.preallocated || !slot.backing.owner)
                return unsupported("QueueBuffer requires a requested dequeued preallocated slot");
            const auto width = parcel_word(slot.flattened, 4);
            const auto height = parcel_word(slot.flattened, 8);
            if (crop[0] < 0 || crop[1] < 0 || crop[2] < crop[0] || crop[3] < crop[1] ||
                static_cast<std::uint32_t>(crop[2]) > width ||
                static_cast<std::uint32_t>(crop[3]) > height)
                return unsupported("QueueBuffer crop is outside the buffer");

            const std::uint64_t frame = ++queue_->frame_counter;
            slot.frame_number = frame;
            slot.state = QueueState::Slot::State::Queued;
            queue_->update_free_event();
            services::GraphicsPresentation presentation;
            presentation.backing = slot.backing;
            presentation.slot = index;
            presentation.frame_number = frame;
            presentation.width = width;
            presentation.height = height;
            presentation.pitch = parcel_word(slot.flattened, 0x70);
            presentation.block_height_log2 = parcel_word(slot.flattened, 0x80);
            presentation.surface_offset = parcel_word(slot.flattened, 0x78);
            presentation.surface_size = std::uint64_t{parcel_word(slot.flattened, 0x94)} |
                                        (std::uint64_t{parcel_word(slot.flattened, 0x98)} << 32);
            presentation.timestamp = std::bit_cast<std::int64_t>(body_u64(0));
            presentation.auto_timestamp = parcel_word(input, body + 8) != 0;
            presentation.crop = crop;
            presentation.scaling_mode = scaling;
            presentation.transform = parcel_word(input, body + 0x20);
            presentation.sticky_transform = parcel_word(input, body + 0x24);
            presentation.swap_interval = signed_word(0x2c);
            presentation.fence_count = fence_count;
            for (std::size_t fence = 0; fence < 4; ++fence) {
                presentation.producer_fences[fence].syncpoint = signed_word(0x34 + fence * 8);
                presentation.producer_fences[fence].threshold =
                    parcel_word(input, body + 0x38 + fence * 8);
            }
            slot.state = QueueState::Slot::State::Acquired;
            const auto presenter = queue_->presenter;
            const auto expected_backing = slot.backing;
            queue_lock.unlock();

            services::GraphicsPresentationResult completed;
            try { completed = presenter(presentation); }
            catch (...) { completed = {false, "Synchronous presenter threw an exception"}; }
            if (!completed.completed) {
                if (completed.cancelled) {
                    IpcReply cancelled;
                    cancelled.disposition = IpcDisposition::cancelled;
                    return cancelled;
                }
                const auto detail = completed.diagnostic.empty()
                    ? std::string_view{"Synchronous presenter did not complete the frame"}
                    : std::string_view{completed.diagnostic};
                constexpr std::size_t capacity = 512;
                binder_diagnostic_.resize(capacity + 1);
                const int prefix = std::snprintf(
                    binder_diagnostic_.data(), binder_diagnostic_.size(),
                    "QueueBuffer presentation failed: frame_number=%llu slot=%u width=%u "
                    "height=%u crop=(%d,%d,%d,%d) scaling_mode=%u transform=%u "
                    "sticky_transform=%u swap_interval=%d; presenter: ",
                    static_cast<unsigned long long>(presentation.frame_number),
                    static_cast<unsigned>(presentation.slot),
                    static_cast<unsigned>(presentation.width),
                    static_cast<unsigned>(presentation.height),
                    static_cast<int>(presentation.crop[0]),
                    static_cast<int>(presentation.crop[1]),
                    static_cast<int>(presentation.crop[2]),
                    static_cast<int>(presentation.crop[3]),
                    static_cast<unsigned>(presentation.scaling_mode),
                    static_cast<unsigned>(presentation.transform),
                    static_cast<unsigned>(presentation.sticky_transform),
                    static_cast<int>(presentation.swap_interval));
                std::size_t used = 0;
                if (prefix > 0)
                    used = std::min<std::size_t>(static_cast<std::size_t>(prefix), capacity);
                else
                    binder_diagnostic_[0] = '\0';
                const std::size_t length = std::min(capacity - used, detail.size());
                std::memcpy(binder_diagnostic_.data() + used, detail.data(), length);
                binder_diagnostic_[used + length] = '\0';
                binder_diagnostic_.resize(used + length);
                return unsupported(binder_diagnostic_.c_str());
            }

            queue_lock.lock();
            auto& release = queue_->slots[index];
            if (release.state != QueueState::Slot::State::Acquired ||
                release.frame_number != frame ||
                release.backing.export_id != expected_backing.export_id ||
                release.backing.guest_address != expected_backing.guest_address ||
                release.backing.size != expected_backing.size ||
                release.backing.owner != expected_backing.owner)
                return unsupported("QueueBuffer slot identity changed during presentation");
            release.state = QueueState::Slot::State::Free;
            queue_->update_free_event();
            const auto default_width = queue_->default_width;
            const auto default_height = queue_->default_height;
            queue_lock.unlock();

            const bool include_frame = (presentation.sticky_transform & 8) != 0;
            const std::uint32_t hint = include_frame ? 0x20 : 0;
            const std::uint32_t total = include_frame ? 0x2c : 0x24;
            put(0, include_frame ? 0x1c : 0x14); put(0x0c, total);
            put(0x10, default_width); put(0x14, default_height);
            put(0x18, hint); put(0x1c, 1);
            if (include_frame) {
                put(0x20, static_cast<std::uint32_t>(frame));
                put(0x24, static_cast<std::uint32_t>(frame >> 32));
                put(0x28, 0);
            } else {
                put(0x20, 0);
            }
            return finish(total);
        }
        case 8: {          // CancelBuffer: a void operation with a nonnull NoFence.
            if (*arguments > parcel_size || parcel_size - *arguments < 48)
                return unsupported("CancelBuffer slot or fence is truncated");
            const auto index = parcel_word(input, *arguments);
            if (parcel_word(input, *arguments + 4) != 0x24 ||
                parcel_word(input, *arguments + 8) != 0)
                return unsupported("CancelBuffer supports only a zero-fd NoFence");
            for (std::size_t offset = *arguments + 12; offset < *arguments + 48; ++offset)
                if (input[offset] != std::byte{})
                    return unsupported("CancelBuffer supports only a zero-fd NoFence");
            const std::lock_guard consumer_lock(queue_->consumer_mutex);
            {
                const std::lock_guard queue_lock(queue_->mutex);
                if (queue_->native_registered)
                    return unsupported("CancelBuffer cannot access native texture slots");
            }
            put(0x00, 0); put(0x0c, 0x10);
            auto reply = finish(0x10);
            if (reply.disposition != IpcDisposition::handled) return reply;
            const std::lock_guard queue_lock(queue_->mutex);
            if (index < queue_->slots.size() &&
                queue_->slots[index].state == QueueState::Slot::State::Dequeued) {
                queue_->slots[index].state = QueueState::Slot::State::Free;
                queue_->slots[index].frame_number = 0;
                queue_->update_free_event();
            }
            return reply;
        }
        case 14: {
            if (*arguments > parcel_size || parcel_size - *arguments < 8)
                return unsupported("SetPreallocatedBuffer arguments are truncated");
            const auto index = parcel_word(input, *arguments);
            const auto presence = parcel_word(input, *arguments + 4);
            if (index >= 64) {
                put(0x00, 4); put(0x0c, 0x14); put(0x10, 0xffffffeau);
                return finish(0x14);
            }
            const std::lock_guard consumer_lock(queue_->consumer_mutex);
            const std::lock_guard queue_lock(queue_->mutex);
            if (queue_->native_registered)
                return unsupported("Preallocation cannot replace native texture slots");
            if (queue_->slots[index].state == QueueState::Slot::State::Acquired)
                return unsupported("Preallocation cannot replace an acquired presentation");
            std::array<std::byte, 0x16c> flattened{};
            services::GraphicsBufferBacking backing{};
            if (presence != 0) {
                if (parcel_size - *arguments < 16 + flattened.size() ||
                    parcel_word(input, *arguments + 8) != flattened.size() ||
                    parcel_word(input, *arguments + 12) != 0)
                    return unsupported("Preallocated GraphicBuffer wrapper is unsupported or truncated");
                std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(*arguments + 16),
                            flattened.size(), flattened.begin());
                const auto u64 = [&](std::size_t offset) {
                    return std::uint64_t{parcel_word(flattened, offset)} |
                           (std::uint64_t{parcel_word(flattened, offset + 4)} << 32);
                };
                const auto width = parcel_word(flattened, 4);
                const auto height = parcel_word(flattened, 8);
                const auto stride = parcel_word(flattened, 0x0c);
                const auto usage = parcel_word(flattened, 0x14);
                const auto export_id = parcel_word(flattened, 0x2c);
                const auto pitch = parcel_word(flattened, 0x70);
                const auto surface_offset = parcel_word(flattened, 0x78);
                const auto surface_size = u64(0x94);
                const auto view_size = parcel_word(flattened, 0x50);
                const auto block_height = parcel_word(flattened, 0x80);
                const std::uint64_t minimum_pitch =
                    (std::uint64_t{width} * 4 + 63) & ~std::uint64_t{63};
                const std::uint64_t row_alignment = block_height <= 5
                    ? std::uint64_t{8} << block_height : 1;
                const std::uint64_t storage_height =
                    (std::uint64_t{height} + row_alignment - 1) & ~(row_alignment - 1);
                const std::uint64_t required_storage = std::uint64_t{pitch} * storage_height;
                if (parcel_word(flattened, 0) != 0x47424652 || !width || !height || !stride ||
                    width > 16384 || height > 16384 || stride < width ||
                    parcel_word(flattened, 0x10) != 1 || !(usage & QueueState::consumer_usage) ||
                    parcel_word(flattened, 0x20) != 0 || parcel_word(flattened, 0x24) != 81 ||
                    !export_id || parcel_word(flattened, 0x34) != 0xdaffcaff ||
                    parcel_word(flattened, 0x40) != usage || parcel_word(flattened, 0x44) != 1 ||
                    parcel_word(flattened, 0x48) != 1 ||
                    parcel_word(flattened, 0x4c) != stride || parcel_word(flattened, 0x54) != 1 ||
                    parcel_word(flattened, 0x5c) != width || parcel_word(flattened, 0x60) != height ||
                    u64(0x64) != 0x100532120ull || parcel_word(flattened, 0x6c) != 3 ||
                    block_height > 5 || !pitch || (pitch & 63) != 0 || pitch < minimum_pitch ||
                    !view_size || !surface_size || surface_size != view_size || required_storage > surface_size ||
                    required_storage > view_size)
                    return unsupported("Preallocated GraphicBuffer format or layout is unsupported");
                if (!queue_->lookup) return unsupported("No graphics-buffer backing provider is installed");
                try { if (const auto found = queue_->lookup(export_id)) backing = *found; }
                catch (...) { return unsupported("Graphics-buffer backing lookup failed"); }
                if (!backing.owner || backing.export_id != export_id || !backing.alignment ||
                    (backing.alignment & (backing.alignment - 1)) != 0 || !backing.guest_address ||
                    backing.guest_address > std::numeric_limits<std::uint64_t>::max() - backing.size ||
                    surface_offset > backing.size || surface_size > backing.size - surface_offset ||
                    required_storage > backing.size - surface_offset)
                    return unsupported("Preallocated GraphicBuffer has no matching exported backing");
                std::fill(flattened.begin() + 0x164, flattened.end(), std::byte{});
            }
            put(0x00, 4); put(0x0c, 0x14); put(0x10, 0);
            auto reply = finish(0x14);
            if (reply.disposition != IpcDisposition::handled) return reply;
            auto& slot = queue_->slots[index];
            const auto frame = slot.frame_number;
            slot = {};
            slot.frame_number = frame;
            if (presence != 0) {
                slot.preallocated = true;
                slot.flattened = flattened;
                slot.backing = std::move(backing);
                queue_->default_width = parcel_word(flattened, 4);
                queue_->default_height = parcel_word(flattened, 8);
                queue_->default_format = parcel_word(flattened, 0x10);
            }
            queue_->maximum = 0;
            bool any = false;
            for (std::size_t candidate = 0; candidate < queue_->slots.size(); ++candidate) {
                if (queue_->slots[candidate].preallocated) queue_->maximum = candidate + 1;
                any = any || static_cast<bool>(queue_->slots[candidate].backing.owner);
            }
            if (!any) {
                for (auto& cleared : queue_->slots) {
                    cleared = {};
                    cleared.frame_number = std::numeric_limits<std::uint64_t>::max();
                }
                queue_->maximum = 0;
            }
            queue_->update_free_event();
            return reply;
        }
        default:
            binder_diagnostic_ = "Binder producer transaction " + std::to_string(transaction) +
                                 " is not implemented";
            return unsupported(binder_diagnostic_.c_str());
        }
    }
    switch (command) {
    case 1: {
        // vi:u root session, GetDisplayServiceWithProxyNameExchange: the client asks for
        // the display service and keeps it as a session of its own, so the reply transfers
        // one that answers the display and binder commands.
        session_handle_ = read(message, 12);
        selector_ = read(message, cmif + 16);
        auto service = std::make_shared<DisplaySession>(*handles_, operation_mode_);
        service->queue_ = queue_;
        service->root_ = false;
        service->application_display_ = true;
        service->converted_ = true;
        service->display_names_ = display_names_;
        service->set_vsync_source(vsync_);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, 4);
        write(reply.bytes, 16, sfco);
        write(reply.bytes, 28, 0);
        reply.transferred[0] = std::move(service);
        reply.transferred_count = 1;
        return reply;
    }
    case 1102: {
        if (!application_display_)
            return unsupported("GetDisplayResolution requires the application display service");
        const auto raw_words = read(message, 4);
        if ((type != 4 && type != 6) || raw_words < (cmif + 24 - 8) / 4 ||
            raw_words > (message.size() - 8) / 4)
            return unsupported("GetDisplayResolution request extent is unsupported");
        // Returns this compatibility size independently of the
        // selected display mode and does not validate the supplied display ID.
        IpcReply reply{IpcDisposition::handled};
        reply.size = cmif + 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, cmif, sfco);
        write(reply.bytes, cmif + 12, 0xadaf6cad);
        write(reply.bytes, cmif + 16, 1280);
        write(reply.bytes, cmif + 24, 720);
        return reply;
    }
    case 1010: {
        // OpenDisplay: the name travels inline and the client keeps the display as a
        // session of its own, so the reply transfers one.
        std::string name;
        for (std::size_t offset = cmif + 16; offset < message.size(); ++offset) {
            const auto value = std::to_integer<char>(message[offset]);
            if (value == 0) break;
            name.push_back(value);
        }
        if (name.empty()) return unsupported("Display open needs a display name");
        display_names_.push_back(name);
        ++next_display_id_;
        auto display = std::make_shared<DisplaySession>(*handles_, operation_mode_);
        display->queue_ = queue_;
        display->converted_ = true;
        display->display_names_ = display_names_;
        display->display_id_ = next_display_id_;
        display->set_vsync_source(vsync_);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        // The client reads a zero result and the display id from the words after the
        // interface word: {SFCO, version, result, u64 display id}.
        reply.size = static_cast<std::size_t>(cmif + 24);
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, cmif, sfco);
        write(reply.bytes, cmif + 4, 1);
        write(reply.bytes, cmif + 8, 0);
        write(reply.bytes, cmif + 12, 0);
        write(reply.bytes, cmif + 16, next_display_id_);
        write(reply.bytes, cmif + 20, 0);
        reply.transferred[0] = std::move(display);
        reply.transferred_count = 1;
        return reply;
    }
    case 5202: {
        // GetDisplayVsyncEvent(display_id): the display's readable event, signaled once
        // per refresh interval by the host source. The reply copies the event handle.
        if (cmif != 16)
            return unsupported("Display vsync needs the non-domain display session shape");
        const auto display_id = read64(message, cmif + 16);
        if (!vsync_) vsync_ = shared_vsync_source();
        bool created = false;
        auto event = vsync_->acquire(display_id, created);
        if (const auto failure = vsync_->failure()) {
            binder_diagnostic_ = "Display refresh source failed: " + failure->detail;
            return unsupported(binder_diagnostic_.c_str());
        }
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = static_cast<std::size_t>(cmif + 16);
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, cmif, sfco);
        write(reply.bytes, cmif + 4, 1);
        write(reply.bytes, cmif + 12, 0x5c1d2e88);
        reply.transferred[0] = std::move(event);
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        return reply;
    }
    case 2020: {
        // OpenLayer: displayName[0x40], u64 layer id, AppletResourceUserId. The
        // service fills the caller's buffer with the layer description.
        if (buffers_ == nullptr || buffers_->empty() || (*buffers_)[0].size < 0x3c)
            return unsupported("OpenLayer needs the caller's layer buffer");
        const auto layer_id = static_cast<std::uint64_t>(read(message, cmif + 16 + 0x40)) |
                              (static_cast<std::uint64_t>(read(message, cmif + 16 + 0x44)) << 32);
        // Parcel flat image: {u32 data size, u32 data offset, u32 object table bytes,
        // u32 object table offset}, the data region, then one four-byte table entry
        // per object. The length is exactly header, data and table: 0x3c here.
        std::array<std::byte, 0x3c> parcel{};
        const auto put = [&](std::size_t offset, std::uint32_t value) {
            for (unsigned index = 0; index != 4; ++index)
                parcel[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
        };
        put(0x00, 0x28);   // data bytes: one 40-byte binder object
        put(0x04, 0x10);   // data offset
        put(0x08, 0x04);   // object table bytes: one four-byte offset
        put(0x0c, 0x38);   // object table offset
        // The producer as a proxy binder object: type 2 (handle), flags 0x7f, the
        // 64-bit handle at +8, a zero cookie at +0x10 and the binder driver's
        // service name in the 16 bytes at +0x18. Local binder types (0, 1) are never
        // serialized here.
        put(0x10, 2);
        put(0x14, 0x7f);
        put(0x18, layer_id != 0 ? 1u : 0u);
        put(0x1c, 0);
        put(0x20, 0);
        put(0x24, 0);
        constexpr char producer_service[] = "dispdrv";
        for (std::size_t index = 0; index != sizeof producer_service; ++index)
            parcel[0x28 + index] = static_cast<std::byte>(producer_service[index]);
        put(0x30, 0);
        put(0x34, 0);
        put(0x38, 0);      // object 0 sits at data offset zero
        if (!buffers_->write(0, parcel)) return unsupported("OpenLayer could not fill the layer buffer");
        // A reply that carries a buffer keeps its interface word directly after the
        // reply header: SFCO at 16, the result at 24 and the written length at 32.
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 40;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        write(reply.bytes, 20, 1);
        write(reply.bytes, 24, 0);
        write(reply.bytes, 28, 0);
        write(reply.bytes, 32, static_cast<std::uint32_t>(parcel.size()));
        write(reply.bytes, 36, 0);
        return reply;
    }
    case 100: {
        // GetRelayService: the display session hands the application its binder driver. The
        // driver is a session of its own, so the reply transfers one that answers the binder
        // commands the client addresses to it.
        service_handle_ = read(message, 12);
        auto driver = std::make_shared<DisplaySession>(*handles_, operation_mode_);
        driver->queue_ = queue_;
        driver->root_ = false;
        driver->converted_ = true;
        driver->set_vsync_source(vsync_);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = static_cast<std::size_t>(cmif + 16);
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, cmif, sfco);
        write(reply.bytes, cmif + 4, 1);
        reply.transferred[0] = std::move(driver);
        reply.transferred_count = 1;
        return reply;
    }
    default: return unsupported("Display command is not implemented");
    }
}

std::size_t DisplaySession::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return converted_ ? 1 + display_names_.size() : 0;
}

const char* DisplaySession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

std::vector<std::string> DisplaySession::displays() const {
    const std::lock_guard lock(mutex_);
    return display_names_;
}

void DisplaySession::set_vsync_source(std::shared_ptr<VsyncSource> source) {
    const std::lock_guard lock(mutex_);
    vsync_ = std::move(source);
}

void DisplaySession::set_buffer_backing_lookup(services::GraphicsBufferBackingLookup lookup) {
    const std::lock_guard session_lock(mutex_);
    const std::lock_guard queue_lock(queue_->mutex);
    queue_->lookup = std::move(lookup);
}

void DisplaySession::set_presenter(services::GraphicsPresenter presenter) {
    const std::lock_guard session_lock(mutex_);
    const std::lock_guard consumer_lock(queue_->consumer_mutex);
    const std::lock_guard queue_lock(queue_->mutex);
    queue_->presenter = std::move(presenter);
}

bool DisplaySession::bind_native_window(void* native_window) {
    if (native_window == nullptr) return false;
    std::shared_ptr<VsyncSource> source;
    std::shared_ptr<QueueState> queue;
    std::uint64_t display_id = 0;
    {
        const std::lock_guard session_lock(mutex_);
        if (!vsync_) vsync_ = shared_vsync_source();
        source = vsync_;
        queue = queue_;
        display_id = display_id_;
    }
    bool created = false;
    static_cast<void>(source->acquire(display_id, created));
    if (source->failure()) return false;

    std::shared_ptr<NativeWindowProducerImpl> producer;
    try {
        producer = std::make_shared<NativeWindowProducerImpl>(queue, source);
    } catch (...) {
        return false;
    }
    const std::lock_guard registry_lock(native_windows_mutex);
    if (native_windows.contains(native_window)) return false;
    const std::lock_guard queue_lock(queue->mutex);
    if (!queue->native_producer.expired() || queue->native_registered) return false;
    queue->native_producer = producer;
    native_windows.emplace(native_window, std::move(producer));
    return true;
}

std::shared_ptr<NativeWindowProducer> find_native_window(void* native_window) {
    if (native_window == nullptr) return {};
    const std::lock_guard lock(native_windows_mutex);
    const auto found = native_windows.find(native_window);
    if (found == native_windows.end()) return {};
    return found->second;
}

void unbind_native_window(void* native_window) {
    std::shared_ptr<NativeWindowProducerImpl> producer;
    {
        const std::lock_guard lock(native_windows_mutex);
        const auto found = native_windows.find(native_window);
        if (found == native_windows.end()) return;
        producer = std::move(found->second);
        native_windows.erase(found);
    }
    producer->detach();
}

namespace {
// Never destroyed: a guest thread can present during process exit.
std::mutex& guest_present_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}
std::function<void()>& guest_present_observer() {
    static auto* observer = new std::function<void()>;
    return *observer;
}
}  // namespace

void set_guest_present_observer(std::function<void()> observer) {
    const std::lock_guard lock(guest_present_mutex());
    guest_present_observer() = std::move(observer);
}

void notify_guest_present() {
    std::function<void()> observer;
    {
        const std::lock_guard lock(guest_present_mutex());
        observer = guest_present_observer();
    }
    if (observer) observer();
}

}  // namespace armigo::services::vi
