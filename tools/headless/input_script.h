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

#include <armigo/platform/host_input.h>
#include <armigo/platform/presenter.h>
#include <functional>
#include <istream>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace armigo::tools {

struct ScriptInputState {
    std::uint64_t frame = 0;
    platform::HostInputSnapshot input;
};
struct InputScript {
    enum class Clock { presents, milliseconds };
    Clock clock = Clock::presents;
    std::vector<ScriptInputState> states;
};
// Retains the scripted controller across host polls. Buttons combine, triggers
// take the greater value, and each stick takes the greater deflection. Physical
// controller motion and keyboard state survive; the normal poller applies focus.
class ScriptInputOverlay {
public:
    void publish(const platform::HostInputSnapshot& input);
    platform::HostInputSnapshot merge(platform::HostInputSnapshot host) const;

private:
    mutable std::mutex mutex_;
    platform::HostInputDeviceState controller_;
};
// Parse the complete script before launch; errors include the source line.
std::variant<InputScript, std::string> parse_input_script(std::istream& source);
// Publishes position zero immediately, then advances on successful presents only.
// Millisecond scripts start their steady clock at the first successful present.
// Uses the same host-to-HID mapping as keyboard/controller input. The optional
// publication and clock callbacks are test seams; production leaves them empty.
std::shared_ptr<platform::Presenter> scripted_presenter(
    std::shared_ptr<platform::Presenter> presenter, InputScript script,
    std::function<void(const platform::HostInputSnapshot&)> publish = {},
    std::function<std::chrono::steady_clock::time_point()> now = {});

} // namespace armigo::tools
