// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "frame_rate_controller.h"

#include <armigo/platform/application.h>

#include <cstdio>

namespace armigo::tools {

FrameRateController::FrameRateController(std::shared_ptr<runtime::PresentationClock> clock,
    std::shared_ptr<platform::Presenter> presenter, std::function<bool(double)> set_display_rate,
    bool automatic, bool log_performance)
    : clock_(std::move(clock)), presenter_(std::move(presenter)),
      set_display_rate_(std::move(set_display_rate)), automatic_(automatic),
      log_performance_(log_performance) {}

void FrameRateController::observe(std::chrono::steady_clock::time_point now) noexcept {
    const std::uint64_t presented = presenter_ ? presenter_->presented_count() : 0;
    if (automatic_) {
        const auto changed = governor_.observe(now, presented, platform::host_thermal_state());
        if (changed) {
            const auto period = std::chrono::nanoseconds(
                static_cast<std::chrono::nanoseconds::rep>(1e9 / *changed));
            const bool clock_set = clock_ && clock_->set_refresh_period(period);
            const bool display_set = set_display_rate_ && set_display_rate_(*changed);
            armigo::diag::write(armigo::diag::Level::info,  "Armigo iOS performance: Auto selected %u fps (clock %s, display link %s)\n",
                *changed, clock_set ? "updated" : "rejected", display_set ? "updated" : "rejected");
        }
    }
    if (!log_performance_) return;
    if (!log_started_) {
        log_started_ = now;
        log_presented_ = presented;
        return;
    }
    const double elapsed = std::chrono::duration<double>(now - *log_started_).count();
    if (elapsed < 10) return;
    const std::uint64_t frames = presented >= log_presented_ ? presented - log_presented_ : 0;
    const double fps = frames / elapsed;
    const double frame_ms = frames ? elapsed * 1000.0 / frames : 0;
    armigo::diag::opt_in( "Armigo iOS performance: %.2f fps, %.2f ms mean frame interval, %llu presented frames in %.1f s\n",
        fps, frame_ms, static_cast<unsigned long long>(frames), elapsed);
    log_started_ = now;
    log_presented_ = presented;
}

} // namespace armigo::tools
