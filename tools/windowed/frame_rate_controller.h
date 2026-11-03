// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/platform/presenter.h>
#include <armigo/runtime/frame_rate_governor.h>
#include <armigo/runtime/presentation_clock.h>

#include <chrono>
#include <functional>
#include <memory>

namespace armigo::tools {

class FrameRateController {
public:
    FrameRateController(std::shared_ptr<runtime::PresentationClock> clock,
                        std::shared_ptr<platform::Presenter> presenter,
                        std::function<bool(double)> set_display_rate,
                        bool automatic, bool log_performance);
    void observe(std::chrono::steady_clock::time_point now) noexcept;

private:
    std::shared_ptr<runtime::PresentationClock> clock_;
    std::shared_ptr<platform::Presenter> presenter_;
    std::function<bool(double)> set_display_rate_;
    bool automatic_ = false;
    bool log_performance_ = false;
    runtime::AutomaticFrameRateGovernor governor_;
    std::optional<std::chrono::steady_clock::time_point> log_started_;
    std::uint64_t log_presented_ = 0;
};

} // namespace armigo::tools
