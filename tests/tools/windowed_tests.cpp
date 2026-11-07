// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "runner_shell.h"
#include "player_settings.h"

#include <armigo/runtime/presentation.h>
#include <armigo/common/diagnostics.h>
#include <armigo/services/vsync_source.h>
#include <cstdlib>
#include <armigo/runtime/process_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

using namespace armigo;
using tc = tools::RunnerExitCode;
using runtime::LaunchError;
using runtime::LaunchFailure;
using runtime::LaunchStage;

tools::RunnerArguments valid_windowed_arguments() {
    tools::RunnerArguments arguments;
    arguments.mode = platform::PresentationMode::windowed;
    arguments.exefs_directory = "exefs";
    arguments.romfs_directory = "romfs";
    arguments.shader_package_directory = "shaders";
    return arguments;
}

LaunchFailure make_failure(LaunchStage stage, LaunchError code, std::string operation) {
    LaunchFailure failure;
    failure.stage = stage;
    failure.code = code;
    failure.operation = std::move(operation);
    failure.detail = "test failure";
    return failure;
}

class FakePresenter final : public platform::Presenter {
public:
    platform::PresentationMode mode() const noexcept override {
        return platform::PresentationMode::windowed;
    }
    platform::PresentResult present(const platform::PresentationFrame&,
                                    std::optional<std::chrono::steady_clock::time_point>,
                                    std::stop_token) override {
        platform::PresentationFailure failure;
        failure.code = platform::PresentationError::cancelled;
        failure.detail = "no presentation in shell tests";
        return failure;
    }
    platform::PresenterStatus drain(
        std::optional<std::chrono::steady_clock::time_point>) override {
        return platform::PresenterStatus{std::monostate{}};
    }
};

// Ordered event log shared between the fakes and the test; it outlives the
// host and runtime because the session owns both only inside its scope.
struct EventLog {
    std::mutex mutex;
    std::vector<std::string> events;
    std::optional<platform::PresenterOptions> created_options;

    void record(const std::string& event) {
        const std::lock_guard lock(mutex);
        events.push_back(event);
    }
    std::size_t count(const std::string& event) {
        const std::lock_guard lock(mutex);
        return std::count(events.begin(), events.end(), event);
    }
    bool before(const std::string& first, const std::string& second) {
        const std::lock_guard lock(mutex);
        std::size_t first_index = events.size();
        std::size_t second_index = events.size();
        for (std::size_t index = 0; index < events.size(); ++index) {
            if (events[index] == first && first_index == events.size()) {
                first_index = index;
            }
            if (events[index] == second && second_index == events.size()) {
                second_index = index;
            }
        }
        return first_index < second_index && second_index != events.size();
    }
};

struct RuntimeStats {
    std::mutex mutex;
    std::condition_variable stop_seen;
    int run_calls = 0;
    int shutdown_calls = 0;
    bool stop_requested = false;
    std::thread::id run_thread{};
    std::vector<runtime::StopReason> stop_reasons;
};

class FakeRuntime final : public runtime::ProcessRuntime {
public:
    FakeRuntime(std::shared_ptr<EventLog> log)
        : stats(std::make_shared<RuntimeStats>()),
          alive(std::make_shared<std::atomic<bool>>(true)),
          log_(std::move(log)) {}
    ~FakeRuntime() override { alive->store(false); }

    runtime::RunResult run(const runtime::RunLimits&, const runtime::ReportDestination&,
                           std::stop_token) override {
        {
            const std::lock_guard lock(stats->mutex);
            stats->run_thread = std::this_thread::get_id();
            ++stats->run_calls;
        }
        if (log_) {
            log_->record("run");
        }
        if (wait_for_stop) {
            std::unique_lock lock(stats->mutex);
            stats->stop_seen.wait(lock, [this] {
                return stats->stop_reasons.size() >= stop_requests_before_return;
            });
        }
        return run_result;
    }
    void request_stop(runtime::StopReason reason) noexcept override {
        const std::lock_guard lock(stats->mutex);
        stats->stop_reasons.push_back(reason);
        stats->stop_requested = true;
        stats->stop_seen.notify_all();
    }
    runtime::LaunchStatus shutdown(std::optional<runtime::ClockInstant>) override {
        const std::lock_guard lock(stats->mutex);
        ++stats->shutdown_calls;
        if (log_) {
            log_->record("shutdown");
        }
        if (shutdown_failure) {
            return runtime::LaunchStatus{*shutdown_failure};
        }
        return runtime::LaunchStatus{std::monostate{}};
    }

    runtime::RunResult run_result;
    bool wait_for_stop = false;
    std::size_t stop_requests_before_return = 1;
    std::optional<LaunchFailure> shutdown_failure;
    std::shared_ptr<RuntimeStats> stats;
    std::shared_ptr<std::atomic<bool>> alive;

private:
    std::shared_ptr<EventLog> log_;
};

class NullApplication final : public platform::Application {
public:
    std::shared_ptr<platform::Presenter> presenter() const noexcept override { return nullptr; }
    platform::PresenterStatus run(std::stop_token) override {
        return platform::PresenterStatus{std::monostate{}};
    }
    void request_quit() noexcept override {}
};

// Fake WindowedHost: records the orchestration order and can inject failures
// and a window-close event without any AppKit session.
class FakeHost final : public tools::WindowedHost {
public:
    explicit FakeHost(std::shared_ptr<EventLog> log) : log_(std::move(log)) {}
    ~FakeHost() override { log_->record("host-destroyed"); }

    platform::ApplicationCreation create(const platform::PresenterOptions& options) override {
        log_->record("create");
        const std::lock_guard lock(log_->mutex);
        log_->created_options = options;
        if (create_failure) {
            return platform::ApplicationCreation{*create_failure};
        }
        return platform::ApplicationCreation{
            std::unique_ptr<platform::Application>(new NullApplication())};
    }
    std::shared_ptr<platform::Presenter> presenter() const noexcept override {
        return window_presenter_;
    }
    platform::DisplayCapabilities display_capabilities() const noexcept override { return display; }
    platform::DisplayCapabilities display;
    std::vector<double> display_moves;
    platform::DisplayChangeCallback display_changed;
    platform::DisplayRefreshCallback display_refresh;
    void set_display_change_handler(platform::DisplayChangeCallback handler) override {
        display_changed = std::move(handler);
    }
    void set_display_refresh_handler(platform::DisplayRefreshCallback handler) override {
        display_refresh = std::move(handler);
    }
    void set_close_handler(std::function<void()> close) override {
        log_->record("close-handler-set");
        const std::lock_guard lock(mutex_);
        close_handler_ = std::move(close);
    }
    platform::PresenterStatus pump(std::stop_token) override {
        log_->record("pump");
        // Cadence changes without a screen move must not emit move warnings.
        if (display_refresh) {
            display_refresh(std::chrono::steady_clock::now(), std::chrono::nanoseconds(16'666'667));
            display_refresh(std::chrono::steady_clock::now(), std::chrono::nanoseconds(8'333'333));
        }
        for (double hz : display_moves) {
            auto moved = display;
            moved.current.refresh_hz = hz;
            if (display_changed) display_changed(moved);
        }
        {
            const std::lock_guard lock(mutex_);
            if (close_handler_ && fire_close_on_pump) {
                close_handler_();
            }
        }
        if (!pump_failure) {
            std::unique_lock lock(mutex_);
            quit_signal_.wait_for(lock, std::chrono::seconds(10),
                                  [this] { return quit_requested_; });
        }
        log_->record("pump-done");
        if (pump_failure) {
            return platform::PresenterStatus{*pump_failure};
        }
        return platform::PresenterStatus{std::monostate{}};
    }
    void request_quit() noexcept override {
        log_->record("request_quit");
        const std::lock_guard lock(mutex_);
        quit_requested_ = true;
        quit_signal_.notify_all();
    }

    void give_presenter(std::shared_ptr<platform::Presenter> presenter) {
        window_presenter_ = std::move(presenter);
    }

    std::shared_ptr<platform::Presenter> window_presenter_;
    std::mutex mutex_;
    std::condition_variable quit_signal_;
    std::function<void()> close_handler_;
    bool fire_close_on_pump = false;
    bool quit_requested_ = false;
    std::optional<platform::PresentationFailure> create_failure;
    std::optional<platform::PresentationFailure> pump_failure;

private:
    std::shared_ptr<EventLog> log_;
};

struct FakeEnvironment {
    tools::ShellEnvironment environment;
    std::shared_ptr<EventLog> log;
    FakeRuntime* runtime = nullptr;
    std::shared_ptr<std::atomic<bool>> runtime_alive;
    std::shared_ptr<RuntimeStats> runtime_stats;
    // Configuration applied to the fake runtime when the shell creates it.
    runtime::RunResult runtime_result;
    bool runtime_wait_for_stop = false;
    std::size_t runtime_stop_requests_before_return = 1;
    std::optional<LaunchFailure> runtime_shutdown_failure;
    int runtime_factory_calls = 0;
    std::optional<LaunchFailure> clock_failure;
    std::optional<LaunchFailure> runtime_factory_failure;
    std::optional<runtime::LaunchDescription> last_description;
    std::chrono::nanoseconds runtime_refresh_period{};
    std::shared_ptr<runtime::PresentationClock> runtime_clock;
    std::chrono::nanoseconds vi_period{};

    FakeEnvironment() {
        environment.create_guest_clock = [this] {
            if (clock_failure) {
                return runtime::LaunchResult<std::shared_ptr<runtime::GuestClock>>(
                    *clock_failure);
            }
            return runtime::create_guest_clock(runtime::GuestClockSources{});
        };
        environment.create_presentation_clock =
            [](std::shared_ptr<runtime::GuestClock> clock,
               runtime::PresentationClockConfiguration configuration) {
                return runtime::create_presentation_clock(std::move(clock), configuration);
            };
        environment.create_presenter = [](const platform::PresenterOptions&) {
            // The windowed shell must use the application presenter instead.
            return platform::PresenterCreation{platform::PresentationFailure{
                platform::PresentationError::host_failure,
                "the windowed shell must not create its own presenter"}};
        };
        environment.create_runtime =
            [this](const runtime::LaunchDescription& description,
                   runtime::RuntimeDependencies&& dependencies) {
                runtime_clock = dependencies.presentation_clock;
                runtime_refresh_period = runtime_clock->configuration().refresh_period;
                services::vi::VsyncSource vsync(runtime_clock);
                vi_period = vsync.interval();
                ++runtime_factory_calls;
                if (runtime_factory_failure) {
                    return runtime::LaunchResult<std::unique_ptr<runtime::ProcessRuntime>>(
                        *runtime_factory_failure);
                }
                last_description = description;
                auto instance = std::make_unique<FakeRuntime>(log);
                instance->run_result = runtime_result;
                instance->wait_for_stop = runtime_wait_for_stop;
                instance->stop_requests_before_return = runtime_stop_requests_before_return;
                instance->shutdown_failure = runtime_shutdown_failure;
                runtime = instance.get();
                runtime_alive = runtime->alive;
                runtime_stats = runtime->stats;
                return runtime::LaunchResult<std::unique_ptr<runtime::ProcessRuntime>>(
                    std::unique_ptr<runtime::ProcessRuntime>(instance.release()));
            };
    }
};

} // namespace

int main() {
    using tools::run_windowed_shell;

    // Exercise the exact production callback installed by the packed player,
    // with clocks created by the shell and screen notifications from the host.
    struct RateCase {
        int stored;
        const char* override;
        double display_hz;
        double input_hz;
        bool explicit_hz;
        int effective;
        double clock_hz;
        unsigned fallback_warnings;
        bool moves = false;
        bool high_only = false;
        std::uint32_t title_default = 30;
    };
    const RateCase rate_cases[] = {
        {0, nullptr, 60, 30, true, 0, 30, 0},
        {30, nullptr, 60, 30, true, 30, 30, 0},
        {60, nullptr, 60, 30, true, 60, 30, 0},
        {0, nullptr, 120, 72, false, 0, 72, 0},
        {30, nullptr, 120, 72, false, 30, 72, 0},
        {60, nullptr, 120, 72, false, 60, 72, 0},
        {120, nullptr, 120, 60, false, 120, 120, 0},
        {120, nullptr, 120, 60, false, 120, 120, 0, true},
        {120, nullptr, 60, 60, false, 60, 60, 1},
        {120, nullptr, 120, 30, true, 30, 30, 1},
        {120, nullptr, 120, 60, true, 60, 60, 1},
        {120, nullptr, 120, 90, true, 60, 90, 1},
        {120, nullptr, 120, 120, true, 120, 120, 0},
        {120, "0", 120, 60, false, 0, 60, 0},
        {120, "30", 120, 60, false, 30, 60, 0},
        {120, "60", 120, 60, false, 60, 60, 0},
        {120, "60", 60, 60, false, 60, 60, 0},
        {30, "120", 120, 60, false, 120, 120, 0},
        {30, "120", 60, 60, false, 120, 60, 0},
        {120, "120", 60, 30, true, 120, 30, 0},
        {30, "120", 120, 30, true, 120, 30, 0},
        {30, "120", 120, 60, true, 120, 60, 0},
        {120, "", 120, 60, false, 0, 60, 0},
        {120, "invalid", 120, 60, false, 0, 60, 0},
        {60, nullptr, 120, 60, false, 30, 60, 1, false, true},
        {120, nullptr, 60, 24, true, 30, 24, 1},
        {120, nullptr, 120, 24, true, 60, 24, 1, false, false, 60},
        {120, "120", 60, 24, true, 120, 24, 0},
        {120, nullptr, 60, 60, false, 60, 60, 1, false, false, 60},
        {120, nullptr, 120, 60, false, 120, 120, 0, false, false, 60},
        {0, nullptr, 120, 60, false, 0, 60, 0, false, false, 60},
        {30, "120", 60, 60, true, 120, 60, 0, false, false, 60},
        {120, "60", 120, 120, true, 60, 120, 0, false, false, 60},
    };
    for (const auto& test : rate_cases) {
        for (const char* key : {"ARMIGO_COMPAT_FRAME_RATE", "ARMIGO_VI_REFRESH_HZ",
                               "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_WIDTH"}) unsetenv(key);
        if (test.override) setenv("ARMIGO_COMPAT_FRAME_RATE", test.override, 1);
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        environment.runtime_result.outcome = runtime::RunSuccess{1};
        environment.runtime_result.quiescent = true;
        std::vector<std::string> tokens{"--exefs", ".", "--romfs", "."};
        if (test.explicit_hz) { tokens.push_back("--refresh-hz"); tokens.push_back(std::to_string(test.input_hz)); }
        std::vector<char*> argv;
        for (auto& token : tokens) argv.push_back(token.data());
        auto parsed = tools::parse_runner_arguments(platform::PresentationMode::windowed, argv);
        check(parsed.ok, "explicit refresh parses for production launch");
        check(parsed.arguments.refresh_hz_explicit == test.explicit_hz, "parser records explicit refresh precedence");
        if (!test.explicit_hz) parsed.arguments.refresh_hz = test.input_hz;
        const auto factory = [&]() -> tools::WindowedHostCreation {
            auto host = std::make_unique<FakeHost>(log);
            host->give_presenter(std::make_shared<FakePresenter>());
            host->display.current = {1920, 1080, test.display_hz, false};
            if (test.moves) host->display_moves = {120, 60, 60, 120, 60};
            return tools::WindowedHostCreation{std::move(host)};
        };
        tools::PlayerChoices choices;
        choices.title_frame_rate = true;
        choices.supported_frame_rates = test.high_only ? std::vector<std::uint32_t>{30, 120}
                                                      : std::vector<std::uint32_t>{30, 60, 120};
        choices.display_refresh_rate = 60;
        choices.default_frame_rate = test.title_default;
        choices.frame_rate = test.stored;
        int calls = 0;
        auto configure = tools::mac_player_launch_configuration(choices, 1280,
            [&](const tools::PlayerChoices& effective, const tools::MacFrameRateSelection& selection) {
                ++calls;
                check(log->count("create") == 1, "selection follows window creation");
                check(environment.runtime_factory_calls == 0, "selection precedes title preparation");
                check(effective.frame_rate == test.effective, "production callback selects the effective rate");
                check(selection.refresh_hz == test.clock_hz, "production callback preserves clock precedence");
            });
        std::vector<std::string> warnings;
        diag::testing_set_sink([](diag::Level level, std::string_view line, void* context) {
            if (level == diag::Level::warning)
                static_cast<std::vector<std::string>*>(context)->emplace_back(line);
        }, &warnings);
        const auto report = run_windowed_shell(parsed.arguments, environment.environment, factory, configure);
        diag::testing_set_sink(nullptr, nullptr);
        check(report.exit_code == tc::success, "production clock policy launch succeeds");
        const auto period = std::chrono::nanoseconds(static_cast<std::int64_t>(1e9 / test.clock_hz));
        check(environment.runtime_refresh_period == period, "runtime receives the selected presentation period");
        check(environment.vi_period == period, "VI receives the same period, including 8,333,333 ns at 120 Hz");
        check(environment.runtime_clock->configuration().refresh_period == period, "display moves keep the launch clock");
        check(calls == 1, "production launch policy runs exactly once");
        const auto selected = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
        check(selected && std::string_view(selected) == (test.override ? test.override : std::to_string(test.effective)),
              "environment override stays verbatim; UI fallback selects only the effective words");
        check(warnings.size() == test.fallback_warnings + (test.moves ? 1u : 0u), "exactly one fallback or display-move warning");
        if (!warnings.empty())
            check(warnings.front().find(test.moves ? "window moved" : "stored " + std::to_string(test.stored) + " fps") != std::string::npos,
                  "warning describes the actual fallback or move");
        if (test.input_hz == 24 && !test.override) {
            check(warnings.size() == 1, "no fitting rate produces one Settings warning");
            if (!warnings.empty()) {
                check(warnings.front().find("24.000 Hz") != std::string::npos, "warning names the explicit clock");
                check(warnings.front().find("every declared rate exceeds it") != std::string::npos,
                      "warning explains why no declared rate fits");
                check(warnings.front().find("title default of " + std::to_string(test.title_default) + " fps") != std::string::npos,
                      "warning names the actual title default");
                check(warnings.front().find("using 0 fps") == std::string::npos, "Auto is never described as zero fps");
            }
        }
    }
    for (const char* key : {"ARMIGO_COMPAT_FRAME_RATE", "ARMIGO_VI_REFRESH_HZ",
                           "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_WIDTH"}) unsetenv(key);

    // Success flow: create, run on a worker, shutdown, quit, destroy last.
    {
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        environment.runtime_result.outcome = runtime::RunSuccess{1};
        environment.runtime_result.quiescent = true;
        FakeHost* host = nullptr;
        auto presenter = std::make_shared<FakePresenter>();
        const auto factory = [&log, &host, presenter]() -> tools::WindowedHostCreation {
            auto instance = std::make_unique<FakeHost>(log);
            host = instance.get();
            instance->give_presenter(presenter);
            return tools::WindowedHostCreation{std::unique_ptr<tools::WindowedHost>(
                instance.release())};
        };
        const auto report = run_windowed_shell(valid_windowed_arguments(),
                                               environment.environment, factory);
        check(report.exit_code == tc::success, "windowed success exits 0");
        check(report.result.quiescent && !report.runtime_retained,
              "windowed success is quiescent and not retained");
        check(log->created_options &&
                  log->created_options->mode == platform::PresentationMode::windowed,
              "the application is created in windowed mode");
        check(environment.runtime_factory_calls == 1, "the runtime is created once");
        const auto& stats = *environment.runtime_stats;
        check(stats.run_calls == 1 && stats.shutdown_calls == 1,
              "run and shutdown are each called once");
        check(stats.run_thread != std::this_thread::get_id(),
              "the runtime runs on a worker thread while AppKit pumps");
        check(!*environment.runtime_alive,
              "the runtime is destroyed after successful shutdown");
        check(log->before("run", "shutdown"), "shutdown follows the run");
        check(log->before("shutdown", "request_quit"),
              "the event loop is woken only after shutdown");
        check(log->before("request_quit", "host-destroyed"),
              "the application is destroyed after the quit request");
        check(log->count("pump") == 1 && log->count("pump-done") == 1, "the pump ran once");
        check(log->before("close-handler-set", "pump"),
              "the close handler is installed before pumping");
    }

    // Window close requests a guest stop and produces a cancelled report.
    {
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        environment.runtime_wait_for_stop = true;
        environment.runtime_result.outcome =
            runtime::RunCancelled{runtime::StopReason::window_closed};
        environment.runtime_result.quiescent = true;
        FakeHost* host = nullptr;
        auto presenter = std::make_shared<FakePresenter>();
        const auto factory = [&log, &host, presenter]() -> tools::WindowedHostCreation {
            auto instance = std::make_unique<FakeHost>(log);
            host = instance.get();
            instance->fire_close_on_pump = true;
            instance->give_presenter(presenter);
            return tools::WindowedHostCreation{std::unique_ptr<tools::WindowedHost>(
                instance.release())};
        };
        const auto report = run_windowed_shell(valid_windowed_arguments(),
                                               environment.environment, factory);
        check(report.exit_code == tc::success, "window close exits 0");
        const auto fields = tools::report_fields(report);
        check(fields.cancel_reason && *fields.cancel_reason == "window_closed",
              "the cancel reason is window_closed");
        bool window_stop = false;
        for (const auto reason : environment.runtime_stats->stop_reasons) {
            window_stop = window_stop || reason == runtime::StopReason::window_closed;
        }
        check(window_stop, "closing the window requests a window_closed stop");
        check(log->before("shutdown", "request_quit"),
              "close teardown still shuts down before waking the loop");
        check(!*environment.runtime_alive,
              "the runtime is released after the window-close teardown");
    }

    // Host creation failure is structured and never reaches the runtime.
    {
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        const auto factory = [log]() -> tools::WindowedHostCreation {
            return tools::WindowedHostCreation{platform::PresentationFailure{
                platform::PresentationError::host_failure, "no window server"}};
        };
        const auto report =
            run_windowed_shell(valid_windowed_arguments(), environment.environment, factory);
        check(report.exit_code == tc::failure, "host creation failure exits 4");
        check(environment.runtime_factory_calls == 0,
              "no runtime is created when the window fails");
        check(log->count("pump") == 0, "the pump never runs without an application");
        const auto fields = tools::report_fields(report);
        check(fields.failure && fields.failure->stage == LaunchStage::graphics &&
                  fields.failure->detail == "no window server",
              "the host failure is reported");
    }

    // Runtime creation failure: structured, no pumping, host destroyed.
    {
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        environment.runtime_factory_failure =
            make_failure(LaunchStage::services, LaunchError::unsupported,
                         "ProcessRuntime::create");
        environment.runtime_factory_failure->detail =
            "Process runtime is not available in this build.";
        FakeHost* host = nullptr;
        auto presenter = std::make_shared<FakePresenter>();
        const auto factory = [&log, &host, presenter]() -> tools::WindowedHostCreation {
            auto instance = std::make_unique<FakeHost>(log);
            host = instance.get();
            instance->give_presenter(presenter);
            return tools::WindowedHostCreation{std::unique_ptr<tools::WindowedHost>(
                instance.release())};
        };
        const auto report = run_windowed_shell(valid_windowed_arguments(),
                                               environment.environment, factory);
        check(report.exit_code == tc::unsupported, "missing runtime build exits 3");
        check(log->count("pump") == 0 && log->count("request_quit") == 0,
              "a failed startup never pumps");
        check(log->count("host-destroyed") == 1, "the host is still destroyed");
        const auto fields = tools::report_fields(report);
        check(fields.failure &&
                  fields.failure->detail.find("not available in this build") !=
                      std::string::npos,
              "the runtime failure detail is carried");
    }

    // A missing application presenter is a structured graphics failure.
    {
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        FakeHost* host = nullptr;
        const auto factory = [&log, &host]() -> tools::WindowedHostCreation {
            auto instance = std::make_unique<FakeHost>(log);
            host = instance.get();
            return tools::WindowedHostCreation{std::unique_ptr<tools::WindowedHost>(
                instance.release())};
        };
        const auto report = run_windowed_shell(valid_windowed_arguments(),
                                               environment.environment, factory);
        check(report.exit_code == tc::failure, "missing presenter exits 4");
        check(environment.runtime_factory_calls == 0,
              "no runtime is created without a presenter");
        check(log->count("pump") == 0, "no pump without a presenter");
    }

    // Failed shutdown retains the runtime and upgrades the exit code.
    {
        const std::size_t retained_before = tools::retained_runtime_count();
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        environment.runtime_result.outcome = runtime::RunSuccess{1};
        environment.runtime_result.quiescent = true;
        environment.runtime_shutdown_failure =
            make_failure(LaunchStage::shutdown, LaunchError::deadline_expired, "shutdown");
        FakeHost* host = nullptr;
        auto presenter = std::make_shared<FakePresenter>();
        const auto factory = [&log, &host, presenter]() -> tools::WindowedHostCreation {
            auto instance = std::make_unique<FakeHost>(log);
            host = instance.get();
            instance->give_presenter(presenter);
            return tools::WindowedHostCreation{std::unique_ptr<tools::WindowedHost>(
                instance.release())};
        };
        const auto report = run_windowed_shell(valid_windowed_arguments(),
                                               environment.environment, factory);
        check(report.exit_code == tc::failure, "failed teardown upgrades success to 4");
        check(report.runtime_retained, "the runtime is retained on teardown failure");
        check(tools::retained_runtime_count() == retained_before + 1,
              "the retained runtime is registered");
        check(*environment.runtime_alive, "the retained runtime stays alive");
        check(log->count("request_quit") == 1,
              "the event loop is still woken after failed teardown");
        check(log->before("request_quit", "host-destroyed"),
              "the host is destroyed last even on teardown failure");
    }

    // A broken event loop records a structured failure and still finishes.
    {
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        environment.runtime_wait_for_stop = true;
        environment.runtime_result.outcome =
            runtime::RunCancelled{runtime::StopReason::user_request};
        environment.runtime_result.quiescent = true;
        FakeHost* host = nullptr;
        auto presenter = std::make_shared<FakePresenter>();
        const auto factory = [&log, &host, presenter]() -> tools::WindowedHostCreation {
            auto instance = std::make_unique<FakeHost>(log);
            host = instance.get();
            instance->pump_failure = platform::PresentationFailure{
                platform::PresentationError::host_failure, "event loop broke"};
            instance->give_presenter(presenter);
            return tools::WindowedHostCreation{std::unique_ptr<tools::WindowedHost>(
                instance.release())};
        };
        const auto report = run_windowed_shell(valid_windowed_arguments(),
                                               environment.environment, factory);
        check(report.exit_code == tc::failure, "a broken pump upgrades the outcome to 4");
        check(report.result.teardown_failure &&
                  report.result.teardown_failure->operation == "application pump",
              "the pump failure is recorded as the teardown failure");
        bool user_stop = false;
        for (const auto reason : environment.runtime_stats->stop_reasons) {
            user_stop = user_stop || reason == runtime::StopReason::user_request;
        }
        check(user_stop, "a broken pump requests a cooperative stop");
    }

    // A watchdog firing while an earlier pump cancellation is still being
    // processed must not relabel the pump failure as a timeout.
    {
        auto log = std::make_shared<EventLog>();
        FakeEnvironment environment;
        environment.log = log;
        environment.runtime_wait_for_stop = true;
        environment.runtime_stop_requests_before_return = 2;
        environment.runtime_result.outcome =
            runtime::RunCancelled{runtime::StopReason::user_request};
        environment.runtime_result.quiescent = true;
        auto arguments = valid_windowed_arguments();
        arguments.time_limit = std::chrono::milliseconds(100);
        auto presenter = std::make_shared<FakePresenter>();
        const auto factory = [&log, presenter]() -> tools::WindowedHostCreation {
            auto instance = std::make_unique<FakeHost>(log);
            instance->pump_failure = platform::PresentationFailure{
                platform::PresentationError::host_failure, "event loop broke"};
            instance->give_presenter(presenter);
            return tools::WindowedHostCreation{std::unique_ptr<tools::WindowedHost>(
                instance.release())};
        };
        const auto report = run_windowed_shell(arguments, environment.environment, factory);
        check(report.exit_code == tc::failure,
              "an earlier pump cancellation is not relabelled as a timeout");
        const auto* cancelled = std::get_if<runtime::RunCancelled>(&report.result.outcome);
        check(cancelled && cancelled->reason == runtime::StopReason::user_request,
              "the earlier pump cancellation keeps its reason");
        check(environment.runtime_stats->stop_reasons.size() == 2,
              "the watchdog fires after the pump stop request");
        check(environment.runtime_stats->stop_reasons.front() ==
                  runtime::StopReason::user_request,
              "the pump stop request precedes the watchdog");
        check(environment.runtime_stats->stop_reasons.back() ==
                  runtime::StopReason::watchdog_timeout,
              "the later stop request retains watchdog provenance");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    std::puts("windowed tests passed.");
    return 0;
}
