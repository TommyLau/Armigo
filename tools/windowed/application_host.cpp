// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "application_host.h"

#include <armigo/platform/controller_rumble.h>
#include <armigo/runtime/host_controller.h>
#include <armigo/services/hid.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

namespace armigo::tools {
namespace {

// Owns the platform application for the windowed shell. All methods except
// request_quit run on the application main thread; request_quit forwards to
// the thread-safe Application wake-up.
class ApplicationHost final : public WindowedHost {
public:
    platform::ApplicationCreation create(const platform::PresenterOptions& options) override {
        platform::ApplicationCallbacks callbacks;
        callbacks.close_requested = [this] {
            if (close_handler_) {
                close_handler_();
            }
        };
        // The captured mouse turns the camera; its buttons are Y and ZR.
        callbacks.mouse_moved = [](double dx, double dy) { runtime::mouse_camera().moved(dx, dy); };
        callbacks.mouse_button = [](unsigned button, bool down) {
            runtime::mouse_camera().button(button, down);
        };
        // Without focus the guest sees a released controller (no running on
        // after Cmd-Tab or minimize) and the controller's rumble stops.
        callbacks.focus_changed = [rumble = rumble_](bool focused) {
            runtime::set_host_input_focus(focused);
            rumble->set_focused(focused);
        };
        // Player one's vibration drives the controller's haptics.
        services::hid::set_host_vibration_sink(
            [rumble = rumble_](unsigned device, const services::hid::VibrationValue& value) {
                rumble->set_level(device, platform::rumble_level(value.amplitude_low, value.amplitude_high));
            });
        // Display refreshes reach whatever handler the session installs later;
        // ARMIGO_DISPLAY_REFRESH=0 runs no display link at all.
        const char* refresh = std::getenv("ARMIGO_DISPLAY_REFRESH");
        if (!refresh || std::strcmp(refresh, "0") != 0) {
            callbacks.display_refresh = [this](std::chrono::steady_clock::time_point vblank,
                                               std::chrono::nanoseconds period) {
                std::shared_ptr<const platform::DisplayRefreshCallback> handler;
                {
                    const std::lock_guard lock(refresh_mutex_);
                    handler = refresh_handler_;
                }
                if (handler && *handler) (*handler)(vblank, period);
            };
        }
        auto creation = platform::create_application(options, std::move(callbacks));
        if (auto* application =
                std::get_if<std::unique_ptr<platform::Application>>(&creation)) {
            application_ = std::move(*application);
        }
        return creation;
    }

    std::shared_ptr<platform::Presenter> presenter() const noexcept override {
        return application_ ? application_->presenter() : nullptr;
    }

    void set_close_handler(std::function<void()> close) override {
        close_handler_ = std::move(close);
    }

    void set_display_refresh_handler(platform::DisplayRefreshCallback handler) override {
        auto shared = std::make_shared<const platform::DisplayRefreshCallback>(std::move(handler));
        const std::lock_guard lock(refresh_mutex_);
        refresh_handler_ = std::move(shared);
    }

    bool set_display_refresh_rate(double rate) noexcept override {
        return application_ && application_->set_display_refresh_rate(rate);
    }

    platform::PresenterStatus pump(std::stop_token stop) override {
        if (!application_) {
            platform::PresentationFailure failure;
            failure.code = platform::PresentationError::host_failure;
            failure.detail = "The application was not created.";
            return failure;
        }
        return application_->run(std::move(stop));
    }

    void request_quit() noexcept override {
        if (application_) {
            application_->request_quit();
        }
    }

    ~ApplicationHost() override {
        services::hid::set_host_vibration_sink({});
        platform::set_controller_rumble(0.0f, 0.0f);
    }

private:
    // Motor levels of player one's controller; shared with the HID service's
    // sink, which may outlive a call in flight.
    struct Rumble {
        std::mutex mutex;  // one consistent focus and pair per update
        float level[2]{};
        bool focused = true;
        void set_level(unsigned device, float value) {
            const std::lock_guard lock(mutex);
            level[device & 1] = value;
            send_locked();
        }
        void set_focused(bool value) {
            const std::lock_guard lock(mutex);
            focused = value;
            send_locked();
        }
        void send_locked() const {
            platform::set_controller_rumble(focused ? level[0] : 0.0f, focused ? level[1] : 0.0f);
        }
    };
    std::shared_ptr<Rumble> rumble_ = std::make_shared<Rumble>();
    // Declared before the application so they outlive it: destroying the
    // application stops its display link, the refresh handler's only caller.
    std::mutex refresh_mutex_;
    std::shared_ptr<const platform::DisplayRefreshCallback> refresh_handler_;
    std::unique_ptr<platform::Application> application_;
    std::function<void()> close_handler_;
};

} // namespace

WindowedHostCreation create_application_host() {
    return WindowedHostCreation{std::unique_ptr<WindowedHost>(new ApplicationHost())};
}

} // namespace armigo::tools
