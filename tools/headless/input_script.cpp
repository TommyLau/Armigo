// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "input_script.h"

#include <armigo/runtime/host_controller.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace armigo::tools {
namespace {
struct Event {
    std::uint64_t frame;
    std::string control;
    float x = 0, y = 0;
};
constexpr std::pair<std::string_view, platform::HostButton> buttons[] = {
    {"a", platform::HostButton::East}, {"b", platform::HostButton::South},
    {"x", platform::HostButton::North}, {"y", platform::HostButton::West},
    {"l", platform::HostButton::LeftShoulder}, {"r", platform::HostButton::RightShoulder},
    {"ls", platform::HostButton::LeftStick}, {"rs", platform::HostButton::RightStick},
    {"plus", platform::HostButton::Menu}, {"minus", platform::HostButton::Options},
    {"up", platform::HostButton::Up}, {"down", platform::HostButton::Down},
    {"left", platform::HostButton::Left}, {"right", platform::HostButton::Right},
};
bool number(std::string_view text, std::uint64_t& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}
bool is_button(std::string_view name) {
    return name == "zl" || name == "zr" ||
        std::any_of(std::begin(buttons), std::end(buttons),
                    [name](auto button) { return button.first == name; });
}
void apply(platform::HostInputDeviceState& state, const Event& event) {
    if (event.control == "left-stick") state.left_stick = {event.x, event.y};
    else if (event.control == "right-stick") state.right_stick = {event.x, event.y};
    else if (event.control == "zl") state.left_trigger = event.x;
    else if (event.control == "zr") state.right_trigger = event.x;
    else for (auto [name, button] : buttons) {
        if (name != event.control) continue;
        const auto mask = platform::button_mask(button);
        state.buttons = event.x ? state.buttons | mask : state.buttons & ~mask;
    }
}

class ScriptedPresenter final : public platform::Presenter {
public:
    ScriptedPresenter(std::shared_ptr<platform::Presenter> presenter, InputScript script,
                      std::function<void(const platform::HostInputSnapshot&)> publish_input,
                      std::function<std::chrono::steady_clock::time_point()> now,
                      InputDeliveryObserver observe_delivery, InputDeliveryFlush flush_delivery)
        : presenter_(std::move(presenter)), script_(std::move(script)),
          publish_input_(std::move(publish_input)), now_(std::move(now)),
          observe_delivery_(std::move(observe_delivery)), flush_delivery_(std::move(flush_delivery)) { publish(); }
    platform::PresenterStatus delivery_status() const {
        if (delivery_failure_) return *delivery_failure_;
        return std::monostate{};
    }
    platform::PresentationMode mode() const noexcept override { return presenter_->mode(); }
    std::uint64_t presented_count() const noexcept override { return presenter_->presented_count(); }
    platform::PresentResult present(const platform::PresentationFrame& frame,
        std::optional<std::chrono::steady_clock::time_point> deadline,
        std::stop_token stop) override {
        if (delivery_failure_) return *delivery_failure_;
        auto result = presenter_->present(frame, deadline, stop);
        if (std::holds_alternative<platform::PresentedFrame>(result)) {
            if (script_.clock == InputScript::Clock::milliseconds) {
                const auto current = now_();
                if (!started_) started_ = current;
                completed_ = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(current - *started_).count());
            } else {
                ++completed_;
            }
            publish();
            if (delivery_failure_) return *delivery_failure_;
        }
        return result;
    }
    platform::PresenterStatus drain(
        std::optional<std::chrono::steady_clock::time_point> deadline) override {
        return presenter_->drain(deadline);
    }
private:
    void publish() {
        bool delivered = false;
        while (next_ < script_.states.size() && script_.states[next_].frame <= completed_) {
            const auto index = next_;
            publish_input_(script_.states[index].input);
            ++next_;
            delivered = true;
            if (observe_delivery_) {
                const auto observed = observe_delivery_(index, script_.states[index]);
                if (const auto* error = std::get_if<platform::PresentationFailure>(&observed)) {
                    delivery_failure_ = *error;
                    break;
                }
            }
        }
        if (delivered && flush_delivery_) {
            const auto flushed = flush_delivery_();
            if (const auto* error = std::get_if<platform::PresentationFailure>(&flushed); error && !delivery_failure_)
                delivery_failure_ = *error;
        }
    }
    std::shared_ptr<platform::Presenter> presenter_;
    InputScript script_;
    std::function<void(const platform::HostInputSnapshot&)> publish_input_;
    std::function<std::chrono::steady_clock::time_point()> now_;
    InputDeliveryObserver observe_delivery_;
    InputDeliveryFlush flush_delivery_;
    std::optional<platform::PresentationFailure> delivery_failure_;
    std::optional<std::chrono::steady_clock::time_point> started_;
    std::size_t next_ = 0;
    std::uint64_t completed_ = 0;
};
} // namespace

void ScriptInputOverlay::publish(const platform::HostInputSnapshot& input) {
    const std::lock_guard lock(mutex_);
    controller_ = input.controller;
}

platform::HostInputSnapshot ScriptInputOverlay::merge(platform::HostInputSnapshot host) const {
    const std::lock_guard lock(mutex_);
    if (!controller_.connected || !controller_.supported) return host;
    auto& pad = host.controller;
    if (!pad.connected || !pad.supported) pad = {};
    pad.connected = pad.supported = true;
    pad.buttons |= controller_.buttons;
    pad.left_trigger = std::max(pad.left_trigger, controller_.left_trigger);
    pad.right_trigger = std::max(pad.right_trigger, controller_.right_trigger);
    const auto stronger = [](auto& host_stick, const auto& script_stick) {
        if (script_stick[0] * script_stick[0] + script_stick[1] * script_stick[1] >
            host_stick[0] * host_stick[0] + host_stick[1] * host_stick[1])
            host_stick = script_stick;
    };
    stronger(pad.left_stick, controller_.left_stick);
    stronger(pad.right_stick, controller_.right_stick);
    return host;
}

std::variant<InputScript, std::string> parse_input_script(std::istream& source) {
    std::vector<Event> events;
    InputScript script;
    bool clock_seen = false;
    std::string line;
    std::uint64_t previous = 0;
    std::size_t line_number = 0;
    while (std::getline(source, line)) {
        ++line_number;
        line = line.substr(0, line.find('#'));
        std::istringstream fields(line);
        std::string frame_text, action, control, extra;
        if (!(fields >> frame_text)) continue;
        const auto error = [&](std::string detail) -> std::variant<InputScript, std::string> {
            return "Input script line " + std::to_string(line_number) + ": " + detail;
        };
        if (frame_text == "clock") {
            if (clock_seen || !events.empty() || !(fields >> action) ||
                (action != "presents" && action != "milliseconds") || (fields >> extra))
                return error("expected one clock presents|milliseconds directive before events.");
            script.clock = action == "milliseconds" ? InputScript::Clock::milliseconds
                                                   : InputScript::Clock::presents;
            clock_seen = true;
            continue;
        }
        std::uint64_t frame = 0;
        if (!number(frame_text, frame) || frame < previous)
            return error("positions must be unsigned integers in nondecreasing order.");
        previous = frame;
        if (!(fields >> action >> control)) return error("expected FRAME ACTION CONTROL.");
        Event event{frame, control};
        if (action == "stick") {
            if (control != "left" && control != "right") return error("expected left or right stick.");
            event.control += "-stick";
            if (!(fields >> event.x >> event.y) || !std::isfinite(event.x) ||
                !std::isfinite(event.y) || std::abs(event.x) > 1 || std::abs(event.y) > 1)
                return error("stick components must be finite numbers in [-1, 1].");
            events.push_back(event);
        } else {
            if (!is_button(control)) return error("unknown button.");
            if (action != "down" && action != "up" && action != "hold")
                return error("expected down, up, hold, or stick.");
            event.x = action == "up" ? 0 : 1;
            events.push_back(event);
            if (action == "hold") {
                std::string duration_text;
                std::uint64_t duration = 0;
                if (!(fields >> duration_text) || !number(duration_text, duration) || !duration ||
                    duration > std::numeric_limits<std::uint64_t>::max() - frame)
                    return error("hold duration must be a positive count in the script clock without overflow.");
                events.push_back({frame + duration, control, 0, 0});
            }
        }
        if (fields >> extra) return error("unexpected trailing field.");
    }
    if (source.bad()) return std::string("Could not read input script.");
    std::stable_sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        return a.frame < b.frame;
    });
    platform::HostInputSnapshot state;
    state.controller.connected = state.controller.supported = true;
    script.states.push_back({0, state});
    for (const auto& event : events) {
        apply(state.controller, event);
        if (script.states.back().frame == event.frame) script.states.back().input = state;
        else script.states.push_back({event.frame, state});
    }
    return script;
}

std::shared_ptr<platform::Presenter> scripted_presenter(
    std::shared_ptr<platform::Presenter> presenter, InputScript script,
    std::function<void(const platform::HostInputSnapshot&)> publish,
    std::function<std::chrono::steady_clock::time_point()> now,
    InputDeliveryObserver observe_delivery, InputDeliveryFlush flush_delivery,
    platform::PresenterStatus* initial_delivery_status) {
    if (!now) now = [] { return std::chrono::steady_clock::now(); };
    if (!publish) publish = [](const platform::HostInputSnapshot& input) {
        services::hid::set_host_full_key_input(runtime::full_key_input(input));
    };
    auto result = std::make_shared<ScriptedPresenter>(std::move(presenter), std::move(script),
        std::move(publish), std::move(now), std::move(observe_delivery), std::move(flush_delivery));
    if (initial_delivery_status) *initial_delivery_status = result->delivery_status();
    return result;
}
} // namespace armigo::tools
