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

// Internal helpers shared by the headless and windowed shell sessions; not a
// public interface. Both session implementations live behind the seams in
// runner_shell.h and must keep identical run/stop/teardown ordering.

#include "runner_shell.h"

#include <memory>
#include <string>

namespace armigo::tools::shell_detail {

using runtime::LaunchError;
using runtime::LaunchFailure;
using runtime::LaunchStage;

[[nodiscard]] LaunchFailure failure(LaunchStage stage, LaunchError code,
                                    std::string operation, std::string detail);
[[nodiscard]] ShellReport failed_report(LaunchFailure failure_value);
[[nodiscard]] LaunchError presentation_error_to_launch_error(
    platform::PresentationError code);
// The presentation grid epoch is the first guest clock sample (host display
// policy only); the refresh period comes from explicit shell input.
[[nodiscard]] runtime::LaunchResult<std::shared_ptr<runtime::PresentationClock>>
create_shell_clocks(const RunnerArguments& arguments, const ShellEnvironment& environment);
[[nodiscard]] runtime::LaunchResult<platform::PresenterOptions> shell_presenter_options(
    const RunnerArguments& arguments);
[[nodiscard]] runtime::LaunchDescription build_launch_description(
    const RunnerArguments& arguments, const platform::PresenterOptions& presentation);
// Runs the runtime on a dedicated worker, arming a cooperative stop request at
// the time limit, and blocks the caller until the run returns.
[[nodiscard]] runtime::RunResult run_runtime_with_limits(
    const std::shared_ptr<runtime::ProcessRuntime>& runtime, const RunnerArguments& arguments,
    const runtime::RunLimits& limits, const runtime::ReportDestination& destination);
// Attempts one bounded shutdown: on success the runtime is released and the
// run is marked quiescent; on failure the first causal teardown failure is
// preserved and the runtime is retained until process exit.
void finish_run(const std::shared_ptr<runtime::ProcessRuntime>& runtime,
                ShellReport& report);

} // namespace armigo::tools::shell_detail
