// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/presentation.h>

#include <condition_variable>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

namespace armigo::runtime {
namespace {

LaunchFailure fail(LaunchError code, const char* operation, std::string detail) {
    LaunchFailure result;
    result.stage = LaunchStage::presentation;
    result.code = code;
    result.operation = operation;
    result.detail = std::move(detail);
    return result;
}

LaunchFailure presenter_failure(const platform::PresentationFailure& error,
                                const char* operation) {
    LaunchError code = LaunchError::host_failure;
    switch (error.code) {
    case platform::PresentationError::invalid_frame: code = LaunchError::invalid_input; break;
    case platform::PresentationError::unsupported: code = LaunchError::unsupported; break;
    case platform::PresentationError::cancelled: code = LaunchError::cancelled; break;
    case platform::PresentationError::deadline_expired: code = LaunchError::deadline_expired; break;
    case platform::PresentationError::drawable_unavailable:
    case platform::PresentationError::host_failure: break;
    }
    return fail(code, operation, error.detail);
}

class Binding final : public PresentationBinding, public std::enable_shared_from_this<Binding> {
    struct Job {
        enum class Kind { present, drain, retire } kind = Kind::drain;
        std::optional<services::GraphicsPresentation> queued;
        std::optional<ClockInstant> deadline;
        std::promise<LaunchStatus> result;
    };
    struct RetainedFrame {
        // Destroy the pin before dropping the allocation's retained mapping.
        platform::PresentationFrame frame;
        GuestMemoryPin pin;
    };

public:
    explicit Binding(PresentationDependencies dependencies)
        : dependencies_(std::move(dependencies)), worker_([this] { run(); }) {}

    ~Binding() override {
        // Work retains its owner until an acknowledged successful drain. The
        // destructor therefore only stops an idle, already drained worker.
        request_stop();
        {
            const std::lock_guard lock(jobs_mutex_);
            exit_ = true;
        }
        jobs_changed_.notify_one();
        worker_.join();
    }

    services::GraphicsPresenter callback() override {
        return [self = shared_from_this()](const services::GraphicsPresentation& frame) {
            auto job = std::make_shared<Job>();
            job->kind = Job::Kind::present;
            job->queued = frame;
            std::future<LaunchStatus> pending;
            {
                const std::lock_guard lock(self->admission_mutex_);
                // A VI request may arrive after stop and successful drainage.
                // Refuse it before restoring retention or enqueueing more work.
                if (self->stop_.stop_requested()) {
                    self->remember(fail(LaunchError::cancelled, "presentation.present",
                                        "Presentation was stopped"));
                    return services::GraphicsPresentationResult{false, "Presentation was stopped", true};
                }
                self->retained_owner_ = self;
                pending = self->enqueue(std::move(job));
            }
            auto result = pending.get();
            if (auto* error = std::get_if<LaunchFailure>(&result))
                return services::GraphicsPresentationResult{false, error->detail,
                    error->code == LaunchError::cancelled};
            return services::GraphicsPresentationResult{true, {}};
        };
    }

    std::optional<LaunchFailure> failure() const override {
        const std::lock_guard lock(failure_mutex_);
        return failure_;
    }

    void request_stop() noexcept override { stop_.request_stop(); }

    LaunchStatus drain(std::optional<ClockInstant> deadline) override {
        // Admission stays closed until drainage is acknowledged. A successful
        // backend drain that finishes after our timeout cannot retire G's pin.
        const auto self = shared_from_this();
        std::unique_lock lock(admission_mutex_, std::defer_lock);
        if (deadline) {
            if (!lock.try_lock_until(*deadline))
                return fail(LaunchError::deadline_expired, "presentation.drain",
                            "Presentation drainage admission timed out");
        } else {
            lock.lock();
        }
        retained_owner_ = self;
        auto job = std::make_shared<Job>();
        job->deadline = deadline;
        auto result = enqueue(std::move(job));
        if (deadline && result.wait_until(*deadline) != std::future_status::ready)
            return fail(LaunchError::deadline_expired, "presentation.drain",
                        "Presentation worker did not drain before the deadline");
        auto status = result.get();
        if (std::holds_alternative<LaunchFailure>(status)) return status;
        auto retirement = std::make_shared<Job>();
        retirement->kind = Job::Kind::retire;
        // The worker is idle, admission is closed and retirement only drops
        // completed resources. No consumer call or guest wait follows this ack.
        auto retired = enqueue(std::move(retirement)).get();
        if (std::holds_alternative<LaunchFailure>(retired)) return retired;
        retained_owner_.reset();
        return std::monostate{};
    }

private:
    std::future<LaunchStatus> enqueue(std::shared_ptr<Job> job) {
        auto result = job->result.get_future();
        {
            const std::lock_guard lock(jobs_mutex_);
            jobs_.push_back(std::move(job));
        }
        jobs_changed_.notify_one();
        return result;
    }

    LaunchStatus remember(LaunchFailure error) {
        const std::lock_guard lock(failure_mutex_);
        if (!failure_) failure_ = error;
        return error;
    }

    void run() noexcept {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(jobs_mutex_);
                jobs_changed_.wait(lock, [this] { return exit_ || !jobs_.empty(); });
                if (exit_) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            try {
                LaunchStatus result = std::monostate{};
                switch (job->kind) {
                case Job::Kind::present: result = present(*job->queued); break;
                case Job::Kind::drain: result = drain_presenter(job->deadline); break;
                case Job::Kind::retire: retained_.reset(); break;
                }
                if (auto* error = std::get_if<LaunchFailure>(&result);
                    error && job->kind == Job::Kind::present) remember(*error);
                job->result.set_value(std::move(result));
            } catch (...) {
                auto error = fail(LaunchError::host_failure,
                    job->queued ? "presentation.present" : "presentation.drain",
                    "Presentation dependency threw an exception");
                if (job->kind == Job::Kind::present) remember(error);
                job->result.set_value(std::move(error));
            }
        }
    }

    LaunchStatus drain_presenter(std::optional<ClockInstant> deadline) {
        auto result = dependencies_.presenter->drain(deadline);
        if (auto* error = std::get_if<platform::PresentationFailure>(&result))
            return presenter_failure(*error, "presenter.drain");
        return std::monostate{};
    }

    LaunchStatus present(const services::GraphicsPresentation& queued) {
        if (auto error = failure()) return *error;
        if (stop_.stop_requested())
            return fail(LaunchError::cancelled, "presentation.present", "Presentation was stopped");

        retained_.emplace();
        retained_->frame.queued = queued;
        const auto& backing = queued.backing;
        const auto& crop = queued.crop;
        const bool full = crop == std::array<std::int32_t, 4>{};
        const bool native = queued.native_pixels || queued.native_texture_serial != 0 ||
                            queued.native_vsync_paced;
        std::optional<PresentationSchedule> target;
        const bool headless = dependencies_.presenter->mode() == platform::PresentationMode::headless;
        const bool has_texture = native && queued.native_texture && !queued.native_pixels;
        // A window draws a native producer's texture directly; nothing is
        // read back.
        const bool direct = has_texture && !headless;
        // A headless consumer copies the texture on the GPU and reads it back
        // only when the frame's pixels are kept, so every frame is consumed
        // without a CPU copy of the ones nobody keeps.
        const bool unread = has_texture && headless && dependencies_.wants_readback &&
                            !dependencies_.wants_readback(queued.frame_number);
        if (direct) {
            // The texture is the frame; its producer already paced it.
        } else if (native) {
            // A native producer may hand over the frame as a texture whose
            // readback the consumer asks for; this bridge keeps pixels, so it
            // reads them here unless they are not wanted.
            std::shared_ptr<const graphics::Rgba8Image> native_pixels = queued.native_pixels;
            if (!unread && !native_pixels && queued.native_read_pixels) {
                auto result = queued.native_read_pixels(crop);
                if (const auto* error = std::get_if<graphics::ImageReadbackError>(&result))
                    return fail(LaunchError::invalid_input, "presentation.readback",
                                std::string(error->detail));
                native_pixels = std::make_shared<const graphics::Rgba8Image>(
                    std::get<graphics::Rgba8Image>(std::move(result)));
            }
            if ((!native_pixels && !unread) || queued.native_texture_serial == 0 ||
                !queued.native_vsync_paced || backing.owner || backing.export_id != 0 ||
                backing.alignment != 0 || backing.guest_address != 0 || backing.size != 0 ||
                queued.surface_offset != 0 || queued.surface_size != 0 || queued.pitch != 0 ||
                queued.block_height_log2 != 0 || queued.fence_count != 0 ||
                queued.swap_interval < 0 || queued.transform != 0 ||
                queued.scaling_mode != 0 || queued.sticky_transform != 0)
                return fail(LaunchError::invalid_input, "presentation.validate",
                            "Native presentation metadata is inconsistent");
            if (queued.width == 0 || queued.height == 0 || queued.width > 16384 ||
                queued.height > 16384 ||
                (!full && (crop[0] < 0 || crop[1] < 0 || crop[2] <= crop[0] ||
                           crop[3] <= crop[1] ||
                           static_cast<std::uint32_t>(crop[2]) > queued.width ||
                           static_cast<std::uint32_t>(crop[3]) > queued.height)))
                return fail(LaunchError::invalid_input, "presentation.validate",
                            "Native presentation extent or crop is invalid");
            if (unread) {
                // The consumer copies the admitted crop of the texture itself.
                retained_->frame.readback = false;
            } else {
                const std::uint32_t expected_width = full
                    ? queued.width : static_cast<std::uint32_t>(crop[2] - crop[0]);
                const std::uint32_t expected_height = full
                    ? queued.height : static_cast<std::uint32_t>(crop[3] - crop[1]);
                const std::uint64_t expected_bytes =
                    std::uint64_t{expected_width} * expected_height * 4;
                if (native_pixels->width != expected_width ||
                    native_pixels->height != expected_height ||
                    native_pixels->pixels.size() != expected_bytes)
                    return fail(LaunchError::invalid_input, "presentation.validate",
                                "Native presentation pixels do not match the admitted crop");
                retained_->frame.pixels = native_pixels;
            }
        } else {
            if (!backing.owner || backing.export_id == 0 || backing.guest_address == 0 ||
                backing.size == 0 || backing.guest_address > UINT64_MAX - backing.size ||
                queued.surface_offset > backing.size || queued.surface_size == 0 ||
                queued.surface_size > backing.size - queued.surface_offset ||
                queued.surface_size > graphics::max_color_clear_storage_size)
                return fail(LaunchError::invalid_input, "presentation.validate", "Invalid retained surface range");
            // This bridge admits the existing cropped-readback profile and
            // rejects transforms that cannot be represented by the decoder.
            if (queued.width == 0 || queued.height == 0 || queued.width > 16384 ||
                queued.height > 16384 || queued.block_height_log2 > 5 ||
                queued.pitch % 64 != 0 || queued.pitch < std::uint64_t{queued.width} * 4 ||
                queued.fence_count > queued.producer_fences.size() || queued.swap_interval < 0 ||
                queued.transform != 0 || queued.scaling_mode != 0 ||
                (queued.sticky_transform & ~8u) != 0 || !queued.auto_timestamp)
                return fail(LaunchError::unsupported, "presentation.validate",
                            "Surface layout, crop policy, transform or scheduling metadata is unsupported");
            const std::uint64_t alignment = std::uint64_t{8} << queued.block_height_log2;
            const auto rows = (std::uint64_t{queued.height} + alignment - 1) & ~(alignment - 1);
            if (rows * queued.pitch > queued.surface_size)
                return fail(LaunchError::invalid_input, "presentation.validate", "Surface storage is truncated");
            if (!full && (crop[0] < 0 || crop[1] < 0 || crop[2] <= crop[0] || crop[3] <= crop[1] ||
                          static_cast<std::uint32_t>(crop[2]) > queued.width ||
                          static_cast<std::uint32_t>(crop[3]) > queued.height))
                return fail(LaunchError::invalid_input, "presentation.validate", "Crop is empty or outside the surface");

            auto pinned = dependencies_.backing.pin(backing, queued.surface_offset, queued.surface_size);
            if (auto* error = std::get_if<LaunchFailure>(&pinned)) return *error;
            retained_->pin = std::get<GuestMemoryPin>(std::move(pinned));
            const auto& pin = retained_->pin;
            const auto address = backing.guest_address + queued.surface_offset;
            if (!pin || pin->range().address != address || pin->range().size != queued.surface_size ||
                (static_cast<unsigned>(pin->permissions()) & static_cast<unsigned>(memory::Permission::read)) == 0 ||
                pin->bytes().size() != queued.surface_size || pin->bytes().data() == nullptr)
                return fail(LaunchError::invalid_input, "presentation.pin", "Backing provider did not pin the whole readable surface");

            for (std::uint32_t index = 0; index < queued.fence_count; ++index) {
                auto waited = dependencies_.backing.wait_fence(queued.producer_fences[index],
                                                              std::nullopt, stop_.get_token());
                if (auto* error = std::get_if<LaunchFailure>(&waited)) return *error;
            }
            if (stop_.stop_requested())
                return fail(LaunchError::cancelled, "presentation.decode", "Presentation was stopped before decode");
            graphics::ColorClearPlan plan;
            plan.width = queued.width;
            plan.height = queued.height;
            plan.row_bytes = queued.pitch;
            // Decoder geometry covers the pitched, block-aligned rows while
            // the pin retains the complete admitted surface allocation.
            plan.storage_size = rows * queued.pitch;
            plan.layout = graphics::ColorTargetLayout::BlockLinear;
            plan.block_height_log2 = queued.block_height_log2;
            auto decoded = graphics::readback_rgba8(plan, pin->bytes(), queued.crop);
            if (auto* error = std::get_if<graphics::ImageReadbackError>(&decoded))
                return fail(LaunchError::invalid_input, "presentation.decode", std::string(error->detail));
            retained_->frame.pixels = std::make_shared<const graphics::Rgba8Image>(
                std::get<graphics::Rgba8Image>(std::move(decoded)));

            auto schedule = dependencies_.clock->schedule(last_refresh_, queued.swap_interval);
            if (auto* error = std::get_if<LaunchFailure>(&schedule)) return *error;
            target = std::get<PresentationSchedule>(schedule);
            if (!target->immediate) {
                auto waited = dependencies_.clock->clock()->wait_until(target->deadline, stop_.get_token());
                if (auto* error = std::get_if<LaunchFailure>(&waited)) return *error;
                if (std::get<ClockWait>(waited) == ClockWait::cancelled)
                    return fail(LaunchError::cancelled, "presentation.pace", "Presentation pacing was cancelled");
            }
        }
        if (stop_.stop_requested())
            return fail(LaunchError::cancelled, "presentation.present", "Presentation was stopped before consumer submission");
        // A refresh target is not a completion timeout. Passing it to H after
        // reaching it would incorrectly expire every paced frame on submission.
        auto result = dependencies_.presenter->present(retained_->frame, std::nullopt, stop_.get_token());
        if (auto* error = std::get_if<platform::PresentationFailure>(&result))
            return presenter_failure(*error, "presenter.present");
        const auto& completed = std::get<platform::PresentedFrame>(result);
        if (completed.frame_number != queued.frame_number && direct)
            return fail(LaunchError::host_failure, "presenter.complete", "Consumer completion has no matching frame");
        static const graphics::Rgba8Image no_pixels{};
        const auto& pixels = retained_->frame.pixels ? *retained_->frame.pixels : no_pixels;
        // A headless frame that asked for pixels must return them; an unread
        // one completes on the consumer's GPU work alone.
        if (completed.frame_number != queued.frame_number ||
            (headless && retained_->frame.readback &&
             (!completed.readback || completed.readback->width != pixels.width ||
              completed.readback->height != pixels.height ||
              completed.readback->pixels.size() != pixels.pixels.size())))
            return fail(LaunchError::host_failure, "presenter.complete", "Consumer completion has no matching frame or owned readback");
        if (dependencies_.observe_frame) {
            auto observed = dependencies_.observe_frame(completed);
            if (auto* error = std::get_if<LaunchFailure>(&observed)) return *error;
        }
        if (target) last_refresh_ = target->refresh_index;
        retained_.reset();
        return std::monostate{};
    }

    PresentationDependencies dependencies_;
    // Explicit self-retention makes accidental last-owner release non-destructive
    // while work is outstanding. The runtime must keep a handle for drain retry;
    // forgetting it intentionally retains resources rather than freeing live work.
    std::timed_mutex admission_mutex_;
    std::shared_ptr<Binding> retained_owner_;
    std::stop_source stop_;
    mutable std::mutex failure_mutex_;
    std::optional<LaunchFailure> failure_;
    std::mutex jobs_mutex_;
    std::condition_variable jobs_changed_;
    std::deque<std::shared_ptr<Job>> jobs_;
    bool exit_ = false;
    // Worker-owned state. No queue/client lock is held while using dependencies.
    std::optional<RetainedFrame> retained_;
    std::optional<std::uint64_t> last_refresh_;
    std::thread worker_;
};

} // namespace

LaunchResult<std::shared_ptr<PresentationBinding>> bind_presentation(PresentationDependencies dependencies) {
    if (!dependencies.backing.lookup || !dependencies.backing.pin || !dependencies.backing.wait_fence ||
        !dependencies.clock || !dependencies.clock->clock() || !dependencies.presenter)
        return fail(LaunchError::invalid_input, "bind_presentation", "Presentation requires backing, clock and presenter dependencies");
    try {
        return std::shared_ptr<PresentationBinding>{std::make_shared<Binding>(std::move(dependencies))};
    } catch (...) {
        return fail(LaunchError::host_failure, "bind_presentation", "Could not start the presentation worker");
    }
}

} // namespace armigo::runtime
