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

#include <armigo/runtime/presentation.h>
#include <armigo/runtime/process_runtime.h>

#include <atomic>
#include <chrono>
#include <array>
#include <cstdlib>
#include <optional>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>
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

static_assert(services::operation_mode_profile(
                  services::OperationMode::Console).performance_mode == 1);
static_assert(services::operation_mode_profile(
                  services::OperationMode::Console).display_width == 1920);
static_assert(services::operation_mode_profile(
                  services::OperationMode::Console).display_height == 1080);
static_assert(services::operation_mode_profile(
                  services::OperationMode::Handheld).performance_mode == 0);
static_assert(services::operation_mode_profile(
                  services::OperationMode::Handheld).display_width == 1280);
static_assert(services::operation_mode_profile(
                  services::OperationMode::Handheld).display_height == 720);

char* arg(const char* text) { return const_cast<char*>(text); }

// Parsing validates that input paths exist, so the tests create a real
// synthetic fixture tree in the temporary directory.
std::filesystem::path fixture_root() {
    auto root = std::filesystem::temp_directory_path() /
                ("armigo-runtime-launch-tests-" + std::to_string(::getpid()));
    std::filesystem::create_directories(root / "exefs");
    std::filesystem::create_directories(root / "romfs");
    std::filesystem::create_directories(root / "shaders");
    std::filesystem::create_directories(root / "prepared");
    std::ofstream npdm(root / "custom.npdm");
    npdm.put('x');
    return root;
}

std::string path_text(const std::filesystem::path& path) { return path.string(); }

std::vector<char*> base_arguments(const std::filesystem::path& root) {
    // The strings must stay alive for the whole parse call.
    static std::vector<std::string> storage;
    static std::vector<char*> arguments;
    storage = {std::string("--exefs"), path_text(root / "exefs"),
               std::string("--romfs"), path_text(root / "romfs"),
               std::string("--shaders"), path_text(root / "shaders")};
    arguments.clear();
    for (auto& text : storage) {
        arguments.push_back(text.data());
    }
    return arguments;
}

tools::RunnerArguments valid_arguments() {
    tools::RunnerArguments arguments;
    arguments.mode = platform::PresentationMode::headless;
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
    platform::PresentationMode mode() const noexcept override { return recorded_mode; }
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

    platform::PresentationMode recorded_mode = platform::PresentationMode::headless;
};

// State shared with the test so it can inspect a runtime the shell may have
// destroyed at shutdown.
struct RuntimeStats {
    std::mutex mutex;
    std::condition_variable stop_seen;
    int run_calls = 0;
    int shutdown_calls = 0;
    bool stop_requested = false;
    std::optional<runtime::RunLimits> run_limits;
    std::optional<runtime::ReportDestination> run_destination;
    std::thread::id run_thread{};
    std::vector<runtime::StopReason> stop_reasons;
};

class FakeRuntime final : public runtime::ProcessRuntime {
public:
    FakeRuntime()
        : stats(std::make_shared<RuntimeStats>()),
          alive(std::make_shared<std::atomic<bool>>(true)) {}
    ~FakeRuntime() override { alive->store(false); }

    runtime::RunResult run(const runtime::RunLimits& limits,
                           const runtime::ReportDestination& destination,
                           std::stop_token) override {
        {
            const std::lock_guard lock(stats->mutex);
            stats->run_limits = limits;
            stats->run_destination = destination;
            stats->run_thread = std::this_thread::get_id();
            ++stats->run_calls;
        }
        if (wait_for_stop) {
            std::unique_lock lock(stats->mutex);
            stats->stop_seen.wait(lock, [this] { return stats->stop_requested; });
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
        if (shutdown_failure) {
            return runtime::LaunchStatus{*shutdown_failure};
        }
        return runtime::LaunchStatus{std::monostate{}};
    }

    RuntimeStats& statistics() { return *stats; }

    runtime::RunResult run_result;
    bool wait_for_stop = false;
    std::optional<LaunchFailure> shutdown_failure;
    std::shared_ptr<RuntimeStats> stats;
    std::shared_ptr<std::atomic<bool>> alive;
};

// Real guest/presentation clocks from the runtime library; a fake presenter
// and runtime; injectable failures at every stage.
struct FakeEnvironment {
    tools::ShellEnvironment environment;
    FakeRuntime* runtime = nullptr;
    std::shared_ptr<std::atomic<bool>> runtime_alive;
    std::shared_ptr<RuntimeStats> runtime_stats;
    // Configuration applied to the fake runtime when the shell creates it.
    runtime::RunResult runtime_result;
    bool runtime_wait_for_stop = false;
    std::optional<LaunchFailure> runtime_shutdown_failure;
    FakePresenter* presenter = nullptr;
    std::shared_ptr<FakePresenter> presenter_owner;
    int runtime_factory_calls = 0;
    int presenter_factory_calls = 0;
    std::optional<LaunchFailure> clock_failure;
    std::optional<platform::PresentationFailure> presenter_failure;
    std::optional<LaunchFailure> runtime_factory_failure;
    std::optional<runtime::LaunchDescription> last_description;
    std::optional<runtime::RuntimeDependencies> last_dependencies;

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
        environment.create_presenter = [this](const platform::PresenterOptions& options) {
            ++presenter_factory_calls;
            if (presenter_failure) {
                return platform::PresenterCreation{*presenter_failure};
            }
            presenter_owner = std::make_shared<FakePresenter>();
            presenter = presenter_owner.get();
            presenter->recorded_mode = options.mode;
            return platform::PresenterCreation{presenter_owner};
        };
        environment.create_runtime =
            [this](const runtime::LaunchDescription& description,
                   runtime::RuntimeDependencies&& dependencies) {
                ++runtime_factory_calls;
                if (runtime_factory_failure) {
                    return runtime::LaunchResult<std::unique_ptr<runtime::ProcessRuntime>>(
                        *runtime_factory_failure);
                }
                last_description = description;
                last_dependencies = std::move(dependencies);
                auto instance = std::make_unique<FakeRuntime>();
                instance->run_result = runtime_result;
                instance->wait_for_stop = runtime_wait_for_stop;
                instance->shutdown_failure = runtime_shutdown_failure;
                runtime = instance.get();
                runtime_alive = runtime->alive;
                runtime_stats = runtime->stats;
                return runtime::LaunchResult<std::unique_ptr<runtime::ProcessRuntime>>(
                    std::unique_ptr<runtime::ProcessRuntime>(instance.release()));
            };
    }
};

void test_development_run_options() {
    const std::array names{"ARMIGO_DEV_INPUT_SCRIPT", "ARMIGO_DEV_SAVE_DIR", "ARMIGO_DEV_CAPTURE_DIR", "ARMIGO_DEV_FRAMES"};
    std::array<std::optional<std::string>, names.size()> previous;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (const char* value = std::getenv(names[index])) previous[index] = value;
        ::unsetenv(names[index]);
    }
    auto ordinary = valid_arguments();
    check(std::holds_alternative<std::monostate>(tools::apply_development_run_options(ordinary, false)),
          "ordinary launch is unchanged without development options");
    ::setenv(names[0], "input.txt", 1);
    ::setenv(names[1], "scratch", 1);
    ::setenv(names[2], "captures", 1);
    ::setenv(names[3], "4200", 1);
    ordinary.save_directory = "/saved-title";
    const auto forbidden = tools::apply_development_run_options(ordinary, false);
    check(std::holds_alternative<LaunchFailure>(forbidden) && !ordinary.input_script_file,
          "ordinary packs reject scripted environment controls");
    auto development = ordinary;
    const auto configured = tools::apply_development_run_options(development, true);
    check(std::holds_alternative<std::monostate>(configured) &&
              development.input_script_file == "/Diagnostics/saved-title/input.txt" &&
              development.save_directory == "/Diagnostics/saved-title/scratch" &&
              development.capture_directory == "/Diagnostics/saved-title/captures" &&
              development.completed_frame_target == 4200,
          "development controls use a separate diagnostics namespace");
    ::setenv(names[2], "/saved-title/capture", 1);
    auto collision = ordinary;
    check(std::holds_alternative<LaunchFailure>(tools::apply_development_run_options(collision, true)),
          "development capture cannot enter the active player saves");
    ::setenv(names[2], "/Diagnostics/saved-title/scratch/capture", 1);
    check(std::holds_alternative<LaunchFailure>(tools::apply_development_run_options(collision, true)),
          "development capture cannot enter scratch saves");
    ::setenv(names[2], "captures", 1);
    ::setenv(names[3], "0", 1);
    auto invalid = ordinary;
    check(std::holds_alternative<LaunchFailure>(tools::apply_development_run_options(invalid, true)) &&
              !invalid.input_script_file && invalid.save_directory == ordinary.save_directory,
          "invalid frame count rejects the whole configuration without partial updates");
    for (const char* name : names) ::unsetenv(name);
    ::setenv(names[0], "/absolute/input.txt", 1);
    auto absolute = valid_arguments();
    check(std::holds_alternative<LaunchFailure>(tools::apply_development_run_options(absolute, true)),
          "absolute development paths cannot bypass validation of the default title saves");
    absolute.save_directory = "/active-player-saves";
    check(std::holds_alternative<std::monostate>(tools::apply_development_run_options(absolute, true)) &&
              absolute.input_script_file == "/absolute/input.txt",
          "absolute development paths validate against an explicit player save root");
    ::setenv(names[0], "relative.txt", 1);
    auto relative = valid_arguments();
    check(std::holds_alternative<LaunchFailure>(tools::apply_development_run_options(relative, true)),
          "relative development paths require an explicit data root");
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (previous[index]) ::setenv(names[index], previous[index]->c_str(), 1);
        else ::unsetenv(names[index]);
    }
}

} // namespace

int main() {
    test_development_run_options();
    using tools::parse_runner_arguments;
    using tools::run_headless_shell;
    const std::filesystem::path root = fixture_root();

    // Argument parsing: required inputs, values and defaults.
    {
        const std::vector<char*> none = {};
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless, none);
        check(!parsed.ok && parsed.error.find("--exefs") != std::string::npos,
              "missing exefs is a usage error");
    }
    {
        const std::vector<char*> help = {arg("--help")};
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless, help);
        check(parsed.help && parsed.ok, "--help is recognized");
    }
    {
        const auto arguments = base_arguments(root);
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(parsed.ok &&
                  parsed.arguments.operation_mode == services::OperationMode::Console,
              "console operation mode is the default");
    }
    {
        auto arguments = base_arguments(root);
        arguments.push_back(arg("--operation-mode"));
        arguments.push_back(arg("handheld"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::windowed,
                                                   arguments);
        check(parsed.ok &&
                  parsed.arguments.operation_mode == services::OperationMode::Handheld,
              "handheld operation mode is accepted by the shared windowed parser");
    }
    {
        auto arguments = base_arguments(root);
        arguments.push_back(arg("--operation-mode"));
        arguments.push_back(arg("portable"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok && parsed.error.find("console or handheld") != std::string::npos,
              "unknown operation mode is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--frames"));
        arguments.push_back(arg("0"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok && parsed.error.find("--frames") != std::string::npos,
              "zero frame target is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--frames"));
        arguments.push_back(arg("soon"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok, "non-numeric frame target is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--time-limit"));
        arguments.push_back(arg("0"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok && parsed.error.find("--time-limit") != std::string::npos,
              "zero time limit is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--refresh-hz"));
        arguments.push_back(arg("-30"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok, "negative refresh rate is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--width"));
        arguments.push_back(arg("640"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok && parsed.error.find("--height") != std::string::npos,
              "width without height is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--bogus"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok && parsed.error.find("--bogus") != std::string::npos,
              "unknown option is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--romfs"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(!parsed.ok && parsed.error.find("requires a value") != std::string::npos,
              "missing option value is rejected");
    }
    {
        std::vector<char*> arguments = base_arguments(root);
        arguments.push_back(arg("--frames"));
        arguments.push_back(arg("5"));
        arguments.push_back(arg("--time-limit"));
        arguments.push_back(arg("2.5"));
        arguments.push_back(arg("--width"));
        arguments.push_back(arg("640"));
        arguments.push_back(arg("--height"));
        arguments.push_back(arg("480"));
        const std::string npdm_path = path_text(root / "custom.npdm");
        const std::string prepared_path = path_text(root / "prepared");
        arguments.push_back(arg("--npdm"));
        arguments.push_back(arg(npdm_path.c_str()));
        arguments.push_back(arg("--prepared"));
        arguments.push_back(arg(prepared_path.c_str()));
        arguments.push_back(arg("--fullscreen"));
        const auto parsed = parse_runner_arguments(platform::PresentationMode::headless,
                                                   arguments);
        check(parsed.ok, "complete arguments parse");
        check(parsed.arguments.completed_frame_target == 5, "frame target parsed");
        check(parsed.arguments.time_limit &&
                  *parsed.arguments.time_limit == std::chrono::nanoseconds(2500000000),
              "time limit parsed");
        check(parsed.arguments.width == 640 && parsed.arguments.height == 480 &&
                  parsed.arguments.fullscreen,
              "presentation options parsed");
        check(parsed.arguments.npdm_file &&
                  *parsed.arguments.npdm_file == root / "custom.npdm",
              "npdm override parsed");
        check(parsed.arguments.prepared_directory &&
                  *parsed.arguments.prepared_directory == root / "prepared",
              "prepared directory parsed");
        check(parsed.arguments.mode == platform::PresentationMode::headless,
              "mode comes from the tool");
        auto embedded_arguments = arguments;
        for (auto& argument : embedded_arguments) if (std::string_view(argument) == "--prepared") argument = arg("--embedded");
        const auto embedded = parse_runner_arguments(platform::PresentationMode::headless, embedded_arguments);
        check(embedded.ok && embedded.arguments.embedded_directory == root / "prepared" &&
                  !embedded.arguments.prepared_directory, "embedded directory parsed");
        embedded_arguments.push_back(arg("--prepared"));
        embedded_arguments.push_back(arg(prepared_path.c_str()));
        check(!parse_runner_arguments(platform::PresentationMode::headless, embedded_arguments).ok,
              "mixed artifact modes rejected");
    }

    // A headless run without --saves keeps saves for the run: it never asks
    // for the title's save directory (the windowed player's default).
    {
        FakeEnvironment environment;
        environment.runtime_result.outcome = runtime::GuestExit{};
        (void)run_headless_shell(valid_arguments(), environment.environment);
        check(environment.last_description.has_value(), "the description reaches the runtime factory");
        if (environment.last_description)
            check(!environment.last_description->title_saves, "headless runs keep saves for the run");
        {
            FakeEnvironment embedded;
            auto arguments = valid_arguments();
            arguments.embedded_directory = "synthetic-embedded";
            embedded.runtime_result.outcome = runtime::GuestExit{};
            (void)run_headless_shell(arguments, embedded.environment);
            check(embedded.last_description && embedded.last_description->embedded_directory == "synthetic-embedded",
                  "embedded directory reaches the runtime factory");
        }
    }

    // The public runtime boundary rejects forged enum values before touching
    // dependencies or input paths.
    {
        runtime::LaunchDescription description;
        description.operation_mode = static_cast<services::OperationMode>(2);
        auto created = runtime::ProcessRuntime::create(description, {});
        const auto* failure = std::get_if<LaunchFailure>(&created);
        check(failure && failure->stage == LaunchStage::inputs &&
                  failure->code == LaunchError::invalid_input &&
                  failure->operation == "operation-mode",
              "runtime creation rejects an invalid operation mode");
        description.operation_mode = services::OperationMode::Console;
        description.prepared_directory = "prepared";
        description.embedded_directory = "embedded";
        created = runtime::ProcessRuntime::create(description, {});
        failure = std::get_if<LaunchFailure>(&created);
        check(failure && failure->operation == "preparation-mode", "runtime rejects mixed artifact modes");
    }

    // Headless session: success, guest exit, failure, unsupported, cancelled.
    {
        FakeEnvironment environment;
        environment.runtime_factory_failure =
            make_failure(LaunchStage::services, LaunchError::unsupported,
                         "ProcessRuntime::create");
        environment.runtime_factory_failure->detail =
            "Process runtime is not available in this build.";
        const auto report = run_headless_shell(valid_arguments(), environment.environment);
        check(report.exit_code == tc::unsupported, "missing runtime build exits 3");
        const auto fields = tools::report_fields(report);
        check(fields.failure &&
                  fields.failure->detail.find("not available in this build") !=
                      std::string::npos,
              "the not-available detail is preserved");
        check(environment.presenter_factory_calls == 1,
              "presenter creation precedes runtime creation");
    }

    // Startup failures are structured and never create a runtime.
    {
        FakeEnvironment environment;
        environment.clock_failure =
            make_failure(LaunchStage::services, LaunchError::host_failure, "clock");
        const auto report = run_headless_shell(valid_arguments(), environment.environment);
        check(report.exit_code == tc::failure, "clock failure exits 4");
        check(environment.runtime_factory_calls == 0,
              "no runtime is created when clocks fail");
        check(environment.presenter_factory_calls == 0,
              "the presenter is not created when clocks fail first");
        const auto fields = tools::report_fields(report);
        check(fields.failure && fields.failure->operation == "clock", "clock failure is named");
    }
    {
        FakeEnvironment environment;
        environment.presenter_failure = platform::PresentationFailure{
            platform::PresentationError::drawable_unavailable, "no drawable"};
        const auto report = run_headless_shell(valid_arguments(), environment.environment);
        check(report.exit_code == tc::failure, "presenter failure exits 4");
        check(environment.runtime_factory_calls == 0,
              "no runtime is created when the presenter fails");
        const auto fields = tools::report_fields(report);
        check(fields.failure && fields.failure->stage == LaunchStage::graphics &&
                  fields.failure->operation == "create_presenter",
              "presenter failure names the graphics stage");
    }

    // Runtime-backed flows through the fake runtime.
    {
        FakeEnvironment environment;
        environment.runtime_result.outcome = runtime::RunSuccess{1};
        environment.runtime_result.quiescent = true;
        auto arguments = valid_arguments();
        arguments.operation_mode = services::OperationMode::Handheld;
        const auto report = run_headless_shell(arguments, environment.environment);
        check(report.exit_code == tc::success, "default fake run success exits 0");
        check(report.result.quiescent && !report.runtime_retained,
              "successful run is quiescent and not retained");
        check(environment.runtime_alive && !*environment.runtime_alive,
              "the shell destroyed the runtime after successful shutdown");
        const auto& stats = *environment.runtime_stats;
        check(stats.run_calls == 1 && stats.shutdown_calls == 1,
              "run and shutdown are each called once");
        check(stats.run_thread != std::this_thread::get_id(),
              "the runtime runs on a worker thread");
        check(environment.last_description &&
                  environment.last_description->exefs_directory == "exefs" &&
                  environment.last_description->romfs_directory == "romfs" &&
                  environment.last_description->shader_package_directory == "shaders" &&
                  environment.last_description->operation_mode ==
                      services::OperationMode::Handheld,
              "the launch description carries the shell inputs");
        check(environment.last_dependencies &&
                  environment.last_dependencies->clock &&
                  environment.last_dependencies->presentation_clock &&
                  environment.last_dependencies->presenter,
              "the runtime receives clock and presenter dependencies");
        check(environment.presenter_factory_calls == 1 && environment.presenter &&
                  environment.presenter->recorded_mode == platform::PresentationMode::headless,
              "the headless presenter is created in headless mode");
    }
    {
        FakeEnvironment environment;
        environment.runtime_result.outcome = runtime::RunSuccess{4};
        environment.runtime_result.quiescent = true;
        environment.runtime_result.statistics.completed_frames = 4;
        environment.runtime_result.statistics.completed_draws = 12;
        auto arguments = valid_arguments();
        arguments.completed_frame_target = 4;
        arguments.report_file = "out/report.json";
        arguments.capture_directory = "out/capture";
        const auto report = run_headless_shell(arguments, environment.environment);
        check(report.exit_code == tc::success, "frame target reached exits 0");
        check(report.result.statistics.completed_frames == 4 &&
                  report.result.statistics.completed_draws == 12,
              "run statistics are transported");
        const auto& stats = *environment.runtime_stats;
        check(stats.run_limits &&
                  stats.run_limits->completed_frame_target == 4,
              "the frame target reaches the runtime limits");
        check(stats.run_destination && stats.run_destination->report_file &&
                  *stats.run_destination->report_file == "out/report.json" &&
                  stats.run_destination->capture_directory &&
                  *stats.run_destination->capture_directory == "out/capture",
              "the report destination reaches the runtime");
    }
    {
        FakeEnvironment environment;
        environment.runtime_result.outcome = runtime::GuestExit{7};
        environment.runtime_result.quiescent = true;
        const auto report = run_headless_shell(valid_arguments(), environment.environment);
        check(report.exit_code == tc::guest_exit, "guest exit exits 1");
        const auto fields = tools::report_fields(report);
        check(fields.guest_exit_status && *fields.guest_exit_status == 7,
              "guest exit status is reported");
    }
    {
        FakeEnvironment environment;
        environment.runtime_result.outcome =
            make_failure(LaunchStage::graphics, LaunchError::guest_fault, "draw");
        const auto report = run_headless_shell(valid_arguments(), environment.environment);
        check(report.exit_code == tc::failure, "guest fault exits 4");
        check(environment.runtime_stats->shutdown_calls == 1,
              "the shell shuts down after a failed run");
        const auto fields = tools::report_fields(report);
        check(fields.failure && fields.failure->stage == LaunchStage::graphics &&
                  fields.failure->code == LaunchError::guest_fault,
              "the first causal failure survives");
    }
    {
        FakeEnvironment environment;
        environment.runtime_result.outcome =
            runtime::RunCancelled{runtime::StopReason::user_request};
        environment.runtime_result.quiescent = true;
        const auto report = run_headless_shell(valid_arguments(), environment.environment);
        check(report.exit_code == tc::cancelled, "user stop exits 6");
    }

    // A watchdog-issued stop retains timeout provenance when it wins the race
    // against the runtime's own limit handling.
    {
        FakeEnvironment environment;
        environment.runtime_wait_for_stop = true;
        environment.runtime_result.outcome =
            runtime::RunCancelled{runtime::StopReason::watchdog_timeout};
        auto arguments = valid_arguments();
        arguments.time_limit = std::chrono::milliseconds(10);
        const auto report = run_headless_shell(arguments, environment.environment);
        check(report.exit_code == tc::timeout, "watchdog cancellation is a timeout");
        const auto* timeout = std::get_if<runtime::RunTimeout>(&report.result.outcome);
        check(timeout && timeout->limit == *arguments.time_limit,
              "watchdog timeout retains the actual shell limit");
    }

    // Timeout requests stop, keeps the runtime alive across failed teardown,
    // and reports the structured timeout.
    {
        const std::size_t retained_before = tools::retained_runtime_count();
        FakeEnvironment environment;
        environment.runtime_wait_for_stop = true;
        environment.runtime_result.outcome =
            runtime::RunTimeout{std::chrono::milliseconds(100)};
        environment.runtime_shutdown_failure =
            make_failure(LaunchStage::shutdown, LaunchError::deadline_expired, "shutdown");
        auto arguments = valid_arguments();
        arguments.time_limit = std::chrono::milliseconds(120);
        const auto report = run_headless_shell(arguments, environment.environment);
        check(report.exit_code == tc::timeout, "timeout exits 5");
        check(!report.result.quiescent, "a timed-out nonquiescent run stays nonquiescent");
        check(report.runtime_retained, "the runtime is retained after the timeout");
        check(tools::retained_runtime_count() == retained_before + 1,
              "the retained runtime is registered");
        check(environment.runtime_alive && *environment.runtime_alive,
              "the shell never destroys a nonquiescent timed-out runtime");
        bool watchdog_stop = false;
        for (const auto reason : environment.runtime_stats->stop_reasons) {
            watchdog_stop = watchdog_stop ||
                            reason == runtime::StopReason::watchdog_timeout;
        }
        check(watchdog_stop, "the shell requested a cooperative watchdog stop at the limit");
        const auto fields = tools::report_fields(report);
        check(fields.timeout_limit_ns && *fields.timeout_limit_ns == 100000000,
              "the timeout limit is reported");
        check(fields.teardown_failure &&
                  fields.teardown_failure->stage == LaunchStage::shutdown,
              "the teardown failure is preserved beside the timeout");
    }
    {
        const std::size_t retained_before = tools::retained_runtime_count();
        FakeEnvironment environment;
        environment.runtime_wait_for_stop = true;
        environment.runtime_result.outcome =
            runtime::RunTimeout{std::chrono::milliseconds(100)};
        auto arguments = valid_arguments();
        arguments.time_limit = std::chrono::milliseconds(120);
        const auto report = run_headless_shell(arguments, environment.environment);
        check(report.exit_code == tc::timeout, "timeout with clean teardown exits 5");
        check(report.result.quiescent && !report.runtime_retained,
              "successful bounded teardown releases the runtime");
        check(tools::retained_runtime_count() == retained_before,
              "no runtime is retained when teardown succeeds");
        check(environment.runtime_alive && !*environment.runtime_alive,
              "the shell released the runtime after successful shutdown");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    std::puts("runtime launch tests passed.");
    return 0;
}
