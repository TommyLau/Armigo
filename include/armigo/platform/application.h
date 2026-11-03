// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/platform/presenter.h>
#include <chrono>
#include <functional>
#include <memory>
#include <stop_token>

namespace armigo::platform {

// One refresh of the display the window is shown on: the vblank time on the
// steady clock and the display's current refresh period.
using DisplayRefreshCallback =
    std::function<void(std::chrono::steady_clock::time_point vblank, std::chrono::nanoseconds period)>;

// Lane H owns the AppKit main-thread lifecycle; lane I starts the runtime on a
// separate worker. Host UI policy is not Horizon firmware behavior. Close requests
// notify the shell to stop the guest, not to destroy inflight graphics resources.
// Unsupported window modes and host failures are structured PresentationFailure.
// The worker drains guest and graphics work before the main thread quits.
// Resize and fullscreen change the drawable size, not guest metadata.
// A frame counts only after real consumer completion, never after a clear or draw alone.
struct ApplicationCallbacks {
    // Called once on main thread; must be nonblocking and must not destroy the
    // application. Captures remain owned until the event loop and shutdown end.
    std::function<void()> close_requested;
    // Pointer input while the window holds the mouse: a click in the window
    // captures it (the cursor hides and stays put), Esc or losing focus
    // releases it. Motion is in points with y growing downward; buttons are
    // 0 left, 1 right, 2 other. Called on the main thread; optional.
    std::function<void(double dx, double dy)> mouse_moved;
    std::function<void(unsigned button, bool down)> mouse_button;
    // The window gained (true) or lost (false) key status: switching to
    // another application, minimizing or leaving its Space all take it away.
    // Called on the main thread; optional.
    std::function<void(bool focused)> focus_changed;
    // Every refresh of the display the window is on, on a thread of its own
    // (not the main thread); it follows the window to another screen.
    // Optional: without it no display link runs.
    DisplayRefreshCallback display_refresh;
};
class Application {
public:
    virtual ~Application() = default;
    virtual std::shared_ptr<Presenter> presenter() const noexcept = 0;
    // Main-thread call. Pump until request_quit/stop; all native resize/fullscreen
    // events update drawable dimensions without changing guest buffer metadata.
    virtual PresenterStatus run(std::stop_token stop = {}) = 0;
    virtual bool set_display_refresh_rate(double rate) noexcept {
        static_cast<void>(rate);
        return false;
    }
    // Thread-safe: wake the event loop after runtime shutdown and presenter drain.
    virtual void request_quit() noexcept = 0;
};

int host_thermal_state() noexcept;
using ApplicationCreation = std::variant<PresentationFailure, std::unique_ptr<Application>>;
// Main-thread create/destroy; options.mode must be windowed. Success is nonnull.
// Destruction follows guest shutdown and successful presenter drainage.
[[nodiscard]] ApplicationCreation create_application(
    const PresenterOptions& options, ApplicationCallbacks callbacks);

} // namespace armigo::platform
