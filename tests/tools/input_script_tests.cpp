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
#include "input_log.h"
#include "runner_shell.h"
#include <armigo/runtime/host_controller.h>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <unistd.h>
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
void test_delivery_log_records_application_time() {
    using namespace armigo;
    using namespace std::chrono_literals;
    auto now = runtime::ClockInstant{100ms};
    runtime::GuestClockSources sources;
    sources.now = [&] { return now; };
    sources.wait = [](runtime::ClockInstant, std::stop_token) -> runtime::LaunchResult<runtime::ClockWait> {
        return runtime::ClockWait::reached;
    };
    auto clock = std::get<std::shared_ptr<runtime::GuestClock>>(runtime::create_guest_clock(sources));
    auto output = std::make_unique<std::ostringstream>();
    auto* text = output.get();
    tools::InputDeliveryLog log(clock, runtime::ClockInstant{100ms}, tools::InputScript::Clock::milliseconds,
                                std::move(output));
    std::istringstream source("clock milliseconds\n50 down a\n100 up a\n150 stick left 1 -1\n");
    auto script = std::get<tools::InputScript>(tools::parse_input_script(source));
    auto inner = std::make_shared<FakePresenter>();
    services::hid::HostFullKeyInput applied;
    unsigned publications = 0, observations = 0;
    auto wrapped = tools::scripted_presenter(inner, script,
        [&](const platform::HostInputSnapshot& snapshot) { applied = runtime::full_key_input(snapshot); ++publications; },
        [&] { return now; },
        [&](std::size_t index, const tools::ScriptInputState& state) {
            check(publications == observations + 1 && applied.buttons == runtime::full_key_input(state.input).buttons,
                  "log callback follows actual HID publication");
            ++observations;
            return log.record(index, state);
        }, [&] { return log.flush(); });
    check(observations == 1, "initial applied state is logged exactly once");
    inner->fail = true;
    (void)wrapped->present({}, {}, {});
    check(observations == 1, "failed present delivers no additional input");
    inner->fail = false;
    (void)wrapped->present({}, {}, {});
    now += 120ms;
    (void)wrapped->present({}, {}, {});
    check(observations == 3 && applied.buttons == 0, "late present delivers both due states");
    now += 30ms;
    (void)wrapped->present({}, {}, {});
    (void)wrapped->present({}, {}, {});
    check(observations == 4, "repeated presents do not repeat delivered states");
    const auto rows = text->str();
    check(rows.find("120000000\t220000000\t1\t50\tmilliseconds\t1\t") != std::string::npos,
          "pressed state logs delivery time on the frame epoch, not scheduled 50 ms");
    check(rows.find("120000000\t220000000\t2\t100\tmilliseconds\t0\t") != std::string::npos,
          "release delivered in the same present keeps its own index and actual timestamp");
    check(rows.find("150000000\t250000000\t3\t150\tmilliseconds\t") != std::string::npos &&
          rows.find("\t32767\t-32767\t0\t0\n") != std::string::npos,
          "input row contains the mapped HID stick coordinates");
    auto failing = tools::scripted_presenter(inner, script, {}, {},
        [](std::size_t, const tools::ScriptInputState&) -> platform::PresenterStatus {
            return platform::PresentationFailure{platform::PresentationError::host_failure, "Synthetic log failure"};
        });
    check(std::holds_alternative<platform::PresentationFailure>(failing->present({}, {}, {})),
          "a delivery log failure becomes a presenter failure");
}

struct DeliveryGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false;
    void block() {
        std::unique_lock lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
    }
    bool await_entry() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(2), [&] { return entered; });
    }
    void release() {
        const std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};
struct DelayedInputBuffer : std::stringbuf {
    bool armed = false;
    DeliveryGate gate;
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (armed) gate.block();
        return std::stringbuf::xsputn(data, count);
    }
};

void test_due_input_batch_precedes_slow_output() {
    using namespace armigo;
    std::atomic<std::int64_t> now_ns{0};
    const auto now = [&] { return runtime::ClockInstant{std::chrono::nanoseconds{now_ns.load()}}; };
    runtime::GuestClockSources sources;
    sources.now = now;
    sources.wait = [](runtime::ClockInstant, std::stop_token) -> runtime::LaunchResult<runtime::ClockWait> {
        return runtime::ClockWait::reached;
    };
    auto clock = std::get<std::shared_ptr<runtime::GuestClock>>(runtime::create_guest_clock(sources));
    DelayedInputBuffer output;
    tools::InputDeliveryLog log(clock, {}, tools::InputScript::Clock::milliseconds,
                                std::make_unique<std::ostream>(&output));
    std::istringstream source("clock milliseconds\n10 down a\n20 down b\n30 up a\n40 up b\n");
    auto script = std::get<tools::InputScript>(tools::parse_input_script(source));
    std::atomic_uint applied{0};
    std::atomic<std::uint64_t> buttons{0};
    auto inner = std::make_shared<FakePresenter>();
    auto wrapped = tools::scripted_presenter(inner, script,
        [&](const platform::HostInputSnapshot& input) { buttons = runtime::full_key_input(input).buttons; ++applied; },
        now, [&](std::size_t index, const tools::ScriptInputState& state) { return log.record(index, state); },
        [&] { return log.flush(); });
    (void)wrapped->present({}, {}, {});
    now_ns = 100'000'000;
    output.armed = true;
    auto presenting = std::async(std::launch::async, [&] { return wrapped->present({}, {}, {}); });
    check(output.gate.await_entry(), "test output is held during formatting");
    check(applied == 5 && buttons == 0, "all due HID setters run before any slow output");
    now_ns = 900'000'000;
    output.gate.release();
    check(std::holds_alternative<platform::PresentedFrame>(presenting.get()), "batch completes after output release");
    const auto rows = output.str();
    for (unsigned i = 1; i <= 4; ++i)
        check(rows.find("100000000\t100000000\t" + std::to_string(i) + "\t") != std::string::npos,
              "delayed storage preserves the captured application timestamp of every due state");

    // A capacity overflow is an explicit latched failure, never success with
    // silent dropped rows. Only the accepted prefix is written.
    auto accepted = std::make_unique<std::ostringstream>();
    auto* accepted_rows = accepted.get();
    tools::InputDeliveryLog bounded(clock, {}, tools::InputScript::Clock::milliseconds, std::move(accepted), 2);
    unsigned deliveries = 0;
    auto limited = tools::scripted_presenter(inner, script,
        [&](const platform::HostInputSnapshot&) { ++deliveries; }, now,
        [&](std::size_t index, const tools::ScriptInputState& state) { return bounded.record(index, state); },
        [&] { return bounded.flush(); });
    (void)limited->present({}, {}, {});
    now_ns += 100'000'000;
    const auto failed = limited->present({}, {}, {});
    check(std::holds_alternative<platform::PresentationFailure>(failed) && !bounded.ready(), "overflow latches failure");
    check(deliveries == 4, "overflow stops the remaining due setters after the unrecordable state");
    check(accepted_rows->str().find("\t2\t20\t") != std::string::npos &&
          accepted_rows->str().find("\t3\t30\t") == std::string::npos, "overflow retains the logged prefix");
    check(std::holds_alternative<platform::PresentationFailure>(limited->present({}, {}, {})) && deliveries == 4,
          "later presents cannot hide or bypass the overflow");
}

void test_initial_log_failure_prevents_runtime_launch() {
    using namespace armigo;
    struct FailingInitialRecord : std::stringbuf {
        unsigned flushes = 0;
        int sync() override { return ++flushes == 1 ? 0 : -1; }
    } output;
    auto input_path = std::filesystem::temp_directory_path() / ("armigo-initial-input-" + std::to_string(getpid()) + ".txt");
    { std::ofstream script(input_path); script << "0 down a\n"; }
    tools::RunnerArguments arguments;
    arguments.input_script_file = input_path;
    arguments.input_log_file = "synthetic-input-log";
    bool runtime_created = false;
    auto inner = std::make_shared<FakePresenter>();
    tools::ShellEnvironment environment;
    environment.create_guest_clock = [] { return runtime::create_guest_clock(); };
    environment.create_presentation_clock = [](auto clock, auto configuration) {
        return runtime::create_presentation_clock(std::move(clock), configuration);
    };
    environment.create_presenter = [inner](const auto&) -> platform::PresenterCreation {
        return std::shared_ptr<platform::Presenter>(inner);
    };
    environment.open_input_log = [&](const auto&) { return std::make_unique<std::ostream>(&output); };
    environment.create_runtime = [&](const auto&, auto&&) -> runtime::LaunchResult<std::unique_ptr<runtime::ProcessRuntime>> {
        runtime_created = true;
        runtime::LaunchFailure error;
        error.code = runtime::LaunchError::host_failure;
        return error;
    };
    const auto report = tools::run_headless_shell(arguments, environment);
    const auto* error = std::get_if<runtime::LaunchFailure>(&report.result.outcome);
    check(output.flushes == 2, "header succeeded and initial row failed before any present");
    check(error && error->operation == "input delivery log" && !runtime_created,
          "initial record failure is surfaced before runtime creation even when no frame is ever presented");
    std::filesystem::remove(input_path);
    services::hid::set_host_full_key_input({});
}

} // namespace

int main() {
    using namespace armigo;
    test_delivery_log_records_application_time();
    test_due_input_batch_precedes_slow_output();
    test_initial_log_failure_prevents_runtime_launch();
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
    char input_log[] = "--input-log", frame_log[] = "--frame-log", log_file[] = "synthetic.tsv";
    char* delivery_args[] = {exefs, current, romfs, current, shaders, current, option, file, input_log, log_file};
    const auto delivery_options = tools::parse_runner_arguments(platform::PresentationMode::headless, delivery_args);
    check(delivery_options.ok && delivery_options.arguments.input_log_file == "synthetic.tsv",
          "headless CLI preserves explicit delivery log path");
    check(!tools::parse_runner_arguments(platform::PresentationMode::windowed, delivery_args).ok,
          "windowed CLI rejects delivery logging");
    char* no_script[] = {exefs, current, romfs, current, shaders, current, input_log, log_file};
    check(!tools::parse_runner_arguments(platform::PresentationMode::headless, no_script).ok,
          "delivery logging requires a script");
    char* colliding_logs[] = {exefs, current, romfs, current, shaders, current, option, file,
                            input_log, log_file, frame_log, log_file};
    check(!tools::parse_runner_arguments(platform::PresentationMode::headless, colliding_logs).ok,
          "frame and delivery logs cannot overwrite each other");
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
    // The physical stick is shaped again: 0.25 past a 12% dead zone, rescaled.
    check((input.buttons & 7) == 6 && input.left_x >= 4830 && input.left_x <= 4850,
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
