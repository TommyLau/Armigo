// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "runner_shell.h"

#include "frame_rate_controller.h"
#include "runner_shell_internal.h"

#include <armigo/platform/apple_input.h>
#include <armigo/runtime/host_controller.h>

#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>
#include <utility>
#include <variant>

namespace armigo::tools {

ShellReport run_windowed_shell(const RunnerArguments& arguments,
                               const ShellEnvironment& environment,
                               const WindowedHostFactory& host_factory) {
    using runtime::LaunchError;
    using runtime::LaunchFailure;
    using runtime::LaunchStage;

    // The caller is the application main thread; host creation, pumping and
    // destruction never leave it.
    if (!host_factory) {
        return shell_detail::failed_report(shell_detail::failure(
            LaunchStage::graphics, LaunchError::host_failure, "create window",
            "The shell environment is missing a window host factory."));
    }
    auto host_creation = host_factory();
    if (auto* failed = std::get_if<platform::PresentationFailure>(&host_creation)) {
        return shell_detail::failed_report(shell_detail::failure(
            LaunchStage::graphics, shell_detail::presentation_error_to_launch_error(failed->code),
            "create window", failed->detail));
    }
    auto host = std::move(std::get<std::unique_ptr<WindowedHost>>(host_creation));

    auto options_result = shell_detail::shell_presenter_options(arguments);
    if (auto* failed = std::get_if<LaunchFailure>(&options_result)) {
        return shell_detail::failed_report(std::move(*failed));
    }
    auto options = std::get<platform::PresenterOptions>(options_result);
    options.mode = platform::PresentationMode::windowed;

    auto application = host->create(options);
    if (auto* failed = std::get_if<platform::PresentationFailure>(&application)) {
        return shell_detail::failed_report(shell_detail::failure(
            LaunchStage::graphics, shell_detail::presentation_error_to_launch_error(failed->code),
            "create_application", failed->detail));
    }
    auto presenter = host->presenter();
    if (!presenter) {
        return shell_detail::failed_report(shell_detail::failure(
            LaunchStage::graphics, LaunchError::host_failure, "application presenter",
            "The platform application did not supply a presenter."));
    }

    auto clock_result = shell_detail::create_shell_clocks(arguments, environment);
    if (auto* failed = std::get_if<LaunchFailure>(&clock_result)) {
        return shell_detail::failed_report(std::move(*failed));
    }
    auto presentation_clock =
        std::move(std::get<std::shared_ptr<runtime::PresentationClock>>(clock_result));
    const char* automatic = std::getenv("ARMIGO_IOS_FRAME_RATE_AUTO");
    const char* fps_log = std::getenv("ARMIGO_IOS_FPS_LOG");
    std::shared_ptr<FrameRateController> frame_rate;
    if ((automatic && std::strcmp(automatic, "1") == 0) || (fps_log && std::strcmp(fps_log, "1") == 0)) {
        frame_rate = std::make_shared<FrameRateController>(presentation_clock, presenter,
            [host = host.get()](double rate) { return host->set_display_refresh_rate(rate); },
            automatic && std::strcmp(automatic, "1") == 0, fps_log && std::strcmp(fps_log, "1") == 0);
    }
    // The refresh grid's phase follows the window's display when its refresh
    // divides the grid evenly, so each present lands on the same vblank.
    host->set_display_refresh_handler(
        [clock = presentation_clock, frame_rate](std::chrono::steady_clock::time_point vblank,
                                                std::chrono::nanoseconds period) {
            clock->observe_display_refresh(vblank, period);
            if (frame_rate) frame_rate->observe(vblank);
        });

    if (!environment.create_runtime) {
        return shell_detail::failed_report(shell_detail::failure(
            LaunchStage::services, LaunchError::host_failure, "ProcessRuntime::create",
            "The shell environment is missing a runtime factory."));
    }
    auto description = shell_detail::build_launch_description(arguments, options);
    runtime::RuntimeDependencies dependencies;
    dependencies.clock = presentation_clock->clock();
    dependencies.presentation_clock = std::move(presentation_clock);
    dependencies.presenter = presenter;
    auto runtime_result = environment.create_runtime(description, std::move(dependencies));
    if (auto* failed = std::get_if<LaunchFailure>(&runtime_result)) {
        return shell_detail::failed_report(std::move(*failed));
    }
    auto runtime = std::shared_ptr<runtime::ProcessRuntime>(
        std::move(std::get<std::unique_ptr<runtime::ProcessRuntime>>(runtime_result)));

    // Keyboard, controllers and the mouse drive player one while the window runs.
    runtime::HostControllerPoller controller(platform::poll_apple_input);

    // Window close asks the runtime to stop; it never destroys guest or
    // graphics resources directly.
    host->set_close_handler([runtime] {
        runtime->request_stop(runtime::StopReason::window_closed);
    });

    runtime::RunLimits limits;
    limits.completed_frame_target = arguments.completed_frame_target;
    limits.timeout = arguments.time_limit;
    runtime::ReportDestination destination;
    destination.capture_directory = arguments.capture_directory;
    destination.report_file = arguments.report_file;

    ShellReport report;
    std::optional<platform::PresentationFailure> pump_failure;
    // The worker owns the report until request_quit wakes the pump; the main
    // thread reads it only after joining.
    std::thread worker([&] {
        report.result =
            shell_detail::run_runtime_with_limits(runtime, arguments, limits, destination);
        shell_detail::finish_run(runtime, report);
        // Wake the event loop only after runtime shutdown and presenter drain.
        host->request_quit();
    });

    std::stop_source pump_source;
    auto pump_status = host->pump(pump_source.get_token());
    if (auto* failed = std::get_if<platform::PresentationFailure>(&pump_status)) {
        // The event loop broke before the run finished; request a cooperative
        // stop so the worker can still reach its own teardown.
        runtime->request_stop(runtime::StopReason::user_request);
        pump_failure = *failed;
    }
    worker.join();
    if (pump_failure) {
        if (!report.result.teardown_failure) {
            LaunchFailure recorded;
            recorded.stage = LaunchStage::graphics;
            recorded.code = shell_detail::presentation_error_to_launch_error(pump_failure->code);
            recorded.operation = "application pump";
            recorded.detail = pump_failure->detail;
            report.result.teardown_failure = std::move(recorded);
        }
    }
    report.exit_code = classify_run_result(report.result);
    return report;
}

} // namespace armigo::tools
