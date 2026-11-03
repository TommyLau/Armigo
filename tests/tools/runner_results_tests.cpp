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

#include <armigo/runtime/process_runtime.h>

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <thread>
#include <variant>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void check_exit(const armigo::tools::ShellReport& report, armigo::tools::RunnerExitCode expected,
                const char* message) {
    check(armigo::tools::classify_run_result(report.result) == expected, message);
}

int exit_probe_fd = -1;

void write_exit_probe() { static_cast<void>(!write(exit_probe_fd, "x", 1)); }

// Runs the helper in a forked child that owns a live thread and an atexit
// handler, and reports the child's exit status and whether static-destruction
// time handlers ran.
struct ExitProbe {
    int status = -1;
    bool handlers_ran = false;
};

ExitProbe probe_process_exit(int exit_code) {
    int fds[2];
    ExitProbe probe;
    if (pipe(fds) != 0) return probe;
    const pid_t child = fork();
    if (child == 0) {
        close(fds[0]);
        exit_probe_fd = fds[1];
        std::atexit(write_exit_probe);
        std::thread([] {
            for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
        }).detach();
        armigo::tools::exit_process_if_runtime_retained(exit_code);
        std::exit(exit_code);
    }
    close(fds[1]);
    int status = 0;
    if (child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status))
        probe.status = WEXITSTATUS(status);
    char byte = 0;
    probe.handlers_ran = read(fds[0], &byte, 1) == 1;
    close(fds[0]);
    return probe;
}

class FakeRuntime final : public armigo::runtime::ProcessRuntime {
public:
    armigo::runtime::RunResult run(const armigo::runtime::RunLimits&,
                                   const armigo::runtime::ReportDestination&,
                                   std::stop_token) override {
        return result;
    }
    void request_stop(armigo::runtime::StopReason) noexcept override {}
    armigo::runtime::LaunchStatus shutdown(std::optional<armigo::runtime::ClockInstant>) override {
        return armigo::runtime::LaunchStatus{std::monostate{}};
    }

    armigo::runtime::RunResult result;
};

armigo::runtime::RunResult make_result(armigo::runtime::RunOutcome outcome, bool quiescent,
                                       std::optional<armigo::runtime::LaunchFailure> teardown) {
    armigo::runtime::RunResult result;
    result.outcome = std::move(outcome);
    result.quiescent = quiescent;
    result.teardown_failure = std::move(teardown);
    return result;
}

armigo::runtime::LaunchFailure make_failure(armigo::runtime::LaunchError code) {
    armigo::runtime::LaunchFailure failure;
    failure.stage = armigo::runtime::LaunchStage::services;
    failure.code = code;
    failure.operation = "test";
    failure.detail = "test failure";
    return failure;
}

using namespace armigo::runtime;

} // namespace

int main() {
    using namespace armigo::tools;
    using tc = RunnerExitCode;
    const auto teardown = make_failure(LaunchError::host_failure);

    // Every outcome maps to its own exit code.
    check_exit({make_result(RunSuccess{3}, true, std::nullopt), tc::success, false},
               tc::success, "frame-target success exits 0");
    check_exit({make_result(GuestExit{5}, true, std::nullopt), tc::guest_exit, false},
               tc::guest_exit, "guest exit exits 1");
    check_exit({make_result(make_failure(LaunchError::invalid_input), false, std::nullopt),
                tc::failure, false},
               tc::failure, "plain failure exits 4");
    check_exit({make_result(make_failure(LaunchError::unsupported), false, std::nullopt),
                tc::unsupported, false},
               tc::unsupported, "unsupported stop exits 3");
    check_exit({make_result(RunTimeout{std::chrono::seconds(1)}, false, std::nullopt),
                tc::timeout, false},
               tc::timeout, "timeout exits 5");
    check_exit({make_result(RunCancelled{StopReason::window_closed}, true, std::nullopt),
                tc::success, false},
               tc::success, "window close exits 0");
    check_exit({make_result(RunCancelled{StopReason::user_request}, true, std::nullopt),
                tc::cancelled, false},
               tc::cancelled, "user stop exits 6");

    // Teardown failure upgrades non-timeout outcomes to failure; timeout keeps
    // its own code because retention is the documented timeout behavior.
    check_exit({make_result(RunSuccess{1}, false, teardown), tc::failure, true},
               tc::failure, "success with failed teardown exits 4");
    check_exit({make_result(GuestExit{std::nullopt}, false, teardown), tc::guest_exit, true},
               tc::failure, "guest exit with failed teardown exits 4");
    check_exit({make_result(RunCancelled{StopReason::window_closed}, false, teardown),
                tc::cancelled, true},
               tc::failure, "cancel with failed teardown exits 4");
    check_exit({make_result(RunTimeout{std::chrono::seconds(1)}, false, teardown),
                tc::timeout, true},
               tc::timeout, "timeout with failed teardown keeps 5");
    check_exit({make_result(make_failure(LaunchError::guest_fault), false, teardown),
                tc::failure, true},
               tc::failure, "run failure keeps 4 despite teardown failure");

    // Outcome names and structured fields for the tool frontends.
    const auto unsupported = make_result(make_failure(LaunchError::unsupported), false, std::nullopt);
    check(outcome_name(unsupported) == "unsupported", "unsupported outcome name");
    check(outcome_name(make_result(RunSuccess{2}, true, std::nullopt)) == "success",
          "success outcome name");
    check(outcome_name(make_result(GuestExit{std::nullopt}, true, std::nullopt)) == "guest-exit",
          "guest-exit outcome name");
    check(outcome_name(make_result(RunTimeout{std::chrono::seconds(2)}, false, std::nullopt)) ==
              "timeout",
          "timeout outcome name");
    check(outcome_name(make_result(RunCancelled{StopReason::window_closed}, true, std::nullopt)) ==
              "cancelled",
          "cancelled outcome name");

    auto timed_out = make_result(RunTimeout{std::chrono::milliseconds(1500)}, false, teardown);
    timed_out.statistics.queued_frames = 4;
    timed_out.statistics.completed_frames = 3;
    timed_out.statistics.completed_draws = 9;
    timed_out.statistics.elapsed = std::chrono::milliseconds(1500);
    ShellReport retained_report;
    retained_report.result = std::move(timed_out);
    retained_report.exit_code = RunnerExitCode::timeout;
    retained_report.runtime_retained = true;
    const auto fields = report_fields(retained_report);
    check(fields.outcome == "timeout", "fields carry the outcome name");
    check(fields.exit_code == 5, "fields carry the exit code");
    check(!fields.quiescent && fields.runtime_retained, "fields carry retention state");
    check(fields.statistics.completed_frames == 3 && fields.statistics.completed_draws == 9 &&
              fields.statistics.queued_frames == 4,
          "fields carry run statistics");
    check(fields.timeout_limit_ns && *fields.timeout_limit_ns == 1500000000,
          "fields carry the timeout limit");
    check(fields.teardown_failure && fields.teardown_failure->code == LaunchError::host_failure,
          "fields carry the teardown failure");
    check(!fields.failure && !fields.frame_target && !fields.guest_exit_status &&
              !fields.cancel_reason,
          "fields omit alternatives that did not happen");

    auto cancelled = make_result(RunCancelled{StopReason::window_closed}, true, std::nullopt);
    ShellReport cancelled_report;
    cancelled_report.result = std::move(cancelled);
    cancelled_report.exit_code = RunnerExitCode::cancelled;
    const auto cancelled_fields = report_fields(cancelled_report);
    check(cancelled_fields.cancel_reason &&
              *cancelled_fields.cancel_reason == "window_closed",
          "fields carry the cancel reason");

    auto guest_exit = make_result(GuestExit{-3}, true, std::nullopt);
    ShellReport guest_report;
    guest_report.result = std::move(guest_exit);
    guest_report.exit_code = RunnerExitCode::guest_exit;
    const auto guest_fields = report_fields(guest_report);
    check(guest_fields.guest_exit_status && *guest_fields.guest_exit_status == -3,
          "fields carry the guest exit status");

    // Without a retained runtime the helper returns and exit() runs its handlers.
    const auto ordinary = probe_process_exit(3);
    check(ordinary.status == 3 && ordinary.handlers_ran,
          "an ordinary exit still runs exit handlers");

    // Retention keeps a runtime alive until process exit.
    auto runtime = std::make_shared<FakeRuntime>();
    std::weak_ptr<FakeRuntime> observer = runtime;
    const std::size_t before = retained_runtime_count();
    retain_runtime_until_process_exit(runtime);
    runtime.reset();
    check(retained_runtime_count() == before + 1, "retained runtime is registered");
    check(observer.lock() != nullptr, "retained runtime stays alive");

    // A retained runtime's live workers must not race static destruction.
    const auto retained_exit = probe_process_exit(7);
    check(retained_exit.status == 7, "a retained runtime exits with the report's code");
    check(!retained_exit.handlers_ran, "a retained runtime exit skips static destruction");

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    std::puts("runner results tests passed.");
    return 0;
}
