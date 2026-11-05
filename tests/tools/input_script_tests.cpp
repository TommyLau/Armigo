// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "input_script.h"
#include "runner_shell.h"
#include <armigo/runtime/host_controller.h>
#include <cstdio>
#include <sstream>

namespace {
int failures = 0;
void check(bool condition, const char* detail) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", detail); ++failures; }
}
class FakePresenter final : public armigo::platform::Presenter {
public:
    bool fail = false, drained = false;
    std::uint64_t presented_count() const noexcept override { return 42; }
    armigo::platform::PresentationMode mode() const noexcept override {
        return armigo::platform::PresentationMode::headless;
    }
    armigo::platform::PresentResult present(const armigo::platform::PresentationFrame&,
        std::optional<std::chrono::steady_clock::time_point>, std::stop_token) override {
        if (fail) return armigo::platform::PresentationFailure{};
        // The script counts successful presents, not backend frame identifiers.
        return armigo::platform::PresentedFrame{99};
    }
    armigo::platform::PresenterStatus drain(
        std::optional<std::chrono::steady_clock::time_point>) override {
        drained = true;
        return std::monostate{};
    }
};
} // namespace

int main() {
    using namespace armigo;
    std::istringstream source("# synthetic events\n0 hold a 2\n1 stick left -0.5 1\n"
                              "1 down zl\n2 down b\n3 up b\n3 up zl\n3 stick left 0 0\n");
    auto result = tools::parse_input_script(source);
    check(std::holds_alternative<tools::InputScript>(result), "valid input parses");
    if (!std::holds_alternative<tools::InputScript>(result)) return 1;
    auto script = std::get<tools::InputScript>(result);
    auto inner = std::make_shared<FakePresenter>();
    services::hid::HostFullKeyInput state;
    unsigned publications = 0;
    auto presenter = tools::scripted_presenter(inner, script,
        [&](const platform::HostInputSnapshot& snapshot) {
            state = runtime::full_key_input(snapshot);
            ++publications;
        });
    check(state.buttons == 1 && publications == 1, "frame zero publishes A before first present");
    inner->fail = true;
    check(std::holds_alternative<platform::PresentationFailure>(presenter->present({}, {}, {})),
          "presentation failure passes through");
    check(publications == 1, "failed present does not consume an event");
    inner->fail = false;
    presenter->present({}, {}, {});
    check(state.left_x == -16384 && state.left_y == 32767 && (state.buttons & 257) == 257,
          "stick and trigger use ordinary HID mapping");
    presenter->present({}, {}, {});
    check((state.buttons & 3) == 2, "hold releases exactly after duration; simultaneous B preserved");
    presenter->present({}, {}, {});
    check(state.buttons == 0 && state.left_x == 0 && state.left_y == 0, "explicit neutral resets controls");
    presenter->present({}, {}, {});
    check(publications == 4, "end of script holds last state without repeated events");
    presenter->drain({});
    check(inner->drained, "drain reaches underlying presenter");
    check(presenter->presented_count() == 42, "drawable presentation count passes through");
    for (const auto* invalid : {"-1 down a", "2 down a\n1 up a", "0 down unknown",
             "0 stick left nan 0", "0 stick right 0 1.01", "0 hold a 0", "0 hold a -2",
             "18446744073709551615 hold a 1", "0 down a extra", "0 stick middle 0 0",
             "clock seconds", "clock milliseconds extra", "0 down a\nclock milliseconds",
             "clock presents\nclock milliseconds", "0 down", "0 jump a", "0 hold a", "0 stick left 0", "0x1 down a"}) {
        std::istringstream input(invalid);
        const auto parsed = tools::parse_input_script(input);
        check(std::holds_alternative<std::string>(parsed), invalid);
        if (const auto* error = std::get_if<std::string>(&parsed))
            check(error->find("line ") != std::string::npos, "parse error identifies source line");
    }
    std::istringstream equal_frames("0 hold a 2\n2 down a\n2 stick right 1 -1\n");
    auto equal = std::get<tools::InputScript>(tools::parse_input_script(equal_frames));
    const auto mapped = runtime::full_key_input(equal.states.back().input);
    check((mapped.buttons & 1) && mapped.right_x == 32767 && mapped.right_y == -32767,
          "equal-frame events preserve source order, including generated release");
    std::istringstream empty("# empty script\n");
    check(std::get<tools::InputScript>(tools::parse_input_script(empty)).states.size() == 1,
          "empty script starts neutral");
    std::istringstream timed("clock milliseconds\n0 hold a 100\n200 down b\n300 up b\n");
    auto timed_script = std::get<tools::InputScript>(tools::parse_input_script(timed));
    auto clock_now = std::chrono::steady_clock::time_point{};
    unsigned timed_publications = 0;
    auto timed_presenter = tools::scripted_presenter(inner, timed_script,
        [&](const platform::HostInputSnapshot& snapshot) {
            state = runtime::full_key_input(snapshot);
            ++timed_publications;
        }, [&] { return clock_now; });
    clock_now += std::chrono::seconds(10);
    inner->fail = true;
    timed_presenter->present({}, {}, {});
    inner->fail = false;
    timed_presenter->present({}, {}, {});
    check(state.buttons == 1, "first successful present starts elapsed clock");
    for (unsigned i = 0; i < 20; ++i) timed_presenter->present({}, {}, {});
    check(state.buttons == 1 && timed_publications == 1, "present count does not advance elapsed script");
    clock_now += std::chrono::milliseconds(99);
    timed_presenter->present({}, {}, {});
    check(state.buttons == 1, "timed hold remains active before boundary");
    clock_now += std::chrono::milliseconds(1);
    timed_presenter->present({}, {}, {});
    check(state.buttons == 0, "timed hold releases at exact boundary");
    clock_now += std::chrono::milliseconds(200);
    inner->fail = true;
    timed_presenter->present({}, {}, {});
    check(timed_publications == 2, "failed timed present leaves due events pending");
    inner->fail = false;
    timed_presenter->present({}, {}, {});
    check(state.buttons == 0 && timed_publications == 4, "late successful present publishes all due states in order");
    tools::RunnerArguments arguments;
    arguments.input_script_file = "/nonexistent-armigo-synthetic-input-script";
    const auto report = tools::run_headless_shell(arguments, {});
    check(std::holds_alternative<runtime::LaunchFailure>(report.result.outcome),
          "missing script fails before runtime creation");
    if (auto* failure = std::get_if<runtime::LaunchFailure>(&report.result.outcome))
        check(failure->operation == "input script", "missing script has explicit diagnostic");
    char option[] = "--input-script", file[] = "synthetic.txt";
    char exefs[] = "--exefs", romfs[] = "--romfs", shaders[] = "--shaders", current[] = ".";
    char* valid[] = {exefs, current, romfs, current, shaders, current, option, file};
    const auto parsed_arguments = tools::parse_runner_arguments(platform::PresentationMode::headless, valid);
    check(parsed_arguments.ok && parsed_arguments.arguments.input_script_file == "synthetic.txt",
          "headless CLI preserves the input script path");
    const auto windowed = tools::parse_runner_arguments(platform::PresentationMode::windowed, valid);
    check(windowed.ok && windowed.arguments.input_script_file == "synthetic.txt",
          "explicit windowed CLI preserves the input script path");
    const auto defaults = tools::parse_runner_arguments(platform::PresentationMode::windowed, {});
    check(!defaults.arguments.input_script_file,
          "the empty argument span used by packed launches never selects an input script");
    char* missing[] = {option};
    check(!tools::parse_runner_arguments(platform::PresentationMode::headless, missing).ok,
          "script option requires a value");
    check(!tools::parse_runner_arguments(platform::PresentationMode::windowed, missing).ok,
          "windowed script option requires a value");
    const auto windowed_failure = tools::run_windowed_shell(arguments, {}, {});
    const auto* failure = std::get_if<runtime::LaunchFailure>(&windowed_failure.result.outcome);
    check(failure && failure->operation == "input script",
          "windowed missing script fails before window creation");

    tools::ScriptInputOverlay overlay;
    platform::HostInputSnapshot physical;
    physical.controller.connected = physical.controller.supported = true;
    physical.controller.buttons = platform::button_mask(platform::HostButton::South);
    physical.controller.left_stick = {0.25f, 0};
    physical.controller.right_stick = {0, 1};
    physical.controller.left_trigger = 0.75f;
    physical.controller.has_motion = physical.controller.has_rotation_rate = true;
    physical.controller.acceleration = {1, 2, 3};
    physical.keyboard.connected = physical.keyboard.supported = true;
    physical.keyboard.buttons = platform::button_mask(platform::HostButton::North);
    platform::HostInputSnapshot scripted;
    scripted.controller.connected = scripted.controller.supported = true;
    scripted.controller.buttons = platform::button_mask(platform::HostButton::East);
    scripted.controller.left_stick = {-1, 0};
    scripted.controller.right_stick = {0.5f, 0};
    scripted.controller.right_trigger = 1;
    overlay.publish(scripted);
    auto combined = overlay.merge(physical);
    auto input = runtime::full_key_input(combined);
    check((input.buttons & 7) == 7 && input.left_x == -32767 && input.right_y == 32767,
          "script, keyboard and physical controller merge with stronger sticks");
    check((input.buttons & (256 | 512)) == (256 | 512) && input.has_motion && input.has_gyroscope &&
          input.acceleration == physical.controller.acceleration,
          "script preserves physical triggers and motion");
    runtime::FocusFilter focus;
    check(focus.apply(input, false).buttons == 0, "normal focus loss releases merged input");
    check(focus.apply(input, true).buttons == 0, "held merged controls stay released on refocus");
    overlay.publish({});
    input = runtime::full_key_input(overlay.merge(physical));
    check((input.buttons & 7) == 6 && input.left_x == 8192,
          "script release leaves physical controls intact");
    (void)focus.apply(runtime::full_key_input(overlay.merge({})), true);
    overlay.publish(scripted);
    input = runtime::full_key_input(overlay.merge({}));
    check((focus.apply(input, true).buttons & 1) != 0,
          "new scripted press after release survives focus filtering");
    physical.controller.supported = false;
    combined = overlay.merge(physical);
    check(combined.controller.buttons == scripted.controller.buttons && !combined.controller.has_motion,
          "unsupported physical devices cannot contribute stale controls");
    return failures ? 1 : 0;
}
