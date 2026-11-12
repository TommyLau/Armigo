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

#include <armigo/platform/application.h>
#include <armigo/platform/presenter.h>
#include <armigo/runtime/guest_clock.h>
#include <armigo/runtime/launch_description.h>
#include <armigo/runtime/presentation_clock.h>
#include <armigo/runtime/process_runtime.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <ostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace armigo::tools {

// Runner shells for the frozen runtime facade. The shells collect user paths,
// run one process, and report timeout, failure, guest exit and frame success;
// no guest fork path exists and hard termination of a
// stuck run belongs to an external spawn/exec supervisor, never to this code.
// The ShellEnvironment/WindowedHost seams let tests drive the orchestration
// with fakes; production binaries use default_shell_environment().
//
// Exit codes, shared by both shells:
//   0 frame-target success, 1 guest exit, 2 usage error,
//   3 structured unsupported stop, 4 failure, 5 timeout, 6 cancelled
// (window close or user stop). A run cancelled by closing the window still
// reports its structured result; scripts distinguish it by exit code.
enum class RunnerExitCode : int {
    success = 0,
    guest_exit = 1,
    usage = 2,
    unsupported = 3,
    failure = 4,
    timeout = 5,
    cancelled = 6,
};

// Validated command-line inputs shared by the headless and windowed tools.
struct RunnerArguments {
    platform::PresentationMode mode = platform::PresentationMode::headless;
    services::OperationMode operation_mode = services::OperationMode::Console;
    std::filesystem::path exefs_directory;
    std::filesystem::path romfs_directory;
    // Extracted add-on content directories (--addon, repeatable), each named by
    // its 16-hexadecimal-digit data id. They work with either input mode.
    std::vector<std::filesystem::path> addon_directories;
    std::optional<std::filesystem::path> npdm_file;
    // Container input (lane K): extract a user-supplied NSP/XCI with the
    // user's keys into a private cache directory before launch. Mutually
    // exclusive with --exefs/--romfs; headless only in this increment. The
    // extraction runs before launch and only succeeds with all container
    // signatures verified.
    std::optional<std::filesystem::path> nup_file;
    std::optional<std::filesystem::path> container_file;
    std::optional<std::filesystem::path> keys_file;
    // Optional update container layered over --container: the update's
    // program NCA (matched by title id) supplies ExeFS and a BKTR-patched
    // RomFS over the base image. Requires --container; headless only.
    std::optional<std::filesystem::path> update_container_file;
    // Optional rights-id title keys ("rights-id hex = key hex"), consulted
    // when a container has no matching ticket.
    std::optional<std::filesystem::path> title_keys_file;
    // Extraction output directory; a per-process directory under the system
    // temporary path when absent.
    std::optional<std::filesystem::path> cache_directory;
    // Optional offline artifacts; without either form native NPDM-backed
    // input is prepared at load time. The two formats are mutually exclusive.
    std::optional<std::filesystem::path> prepared_directory;
    std::optional<std::filesystem::path> embedded_directory;
    std::filesystem::path shader_package_directory;
    std::optional<std::uint64_t> completed_frame_target;
    std::optional<std::chrono::nanoseconds> time_limit;
    std::optional<std::filesystem::path> capture_directory;
    std::optional<std::filesystem::path> input_script_file;
    // Host directory that keeps the title's save data between runs.
    std::optional<std::filesystem::path> save_directory;
    std::optional<std::filesystem::path> report_file;
    // Host display policy inputs, not guest metadata.
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool fullscreen = false;
    // Windowed presentation choices (platform::PresenterOptions).
    std::uint32_t content_width = 0;
    bool display_sync = true;
    // Explicit host refresh period for the presentation clock; the shell
    // default is host policy and is never read as a firmware constant.
    double refresh_hz = 60.0;
    std::optional<std::filesystem::path> pacing_schedule_file;
    std::optional<std::filesystem::path> frame_log_file;
    std::optional<std::filesystem::path> input_log_file;
    std::optional<double> pacing_max_hz;
    bool refresh_hz_explicit = false; // --refresh-hz always wins over player policy
};

// Applies explicit development-pack environment overrides. Relative paths are
// resolved against the launch's original save directory. Production packs reject
// these options; validation completes before any argument is changed.
[[nodiscard]] runtime::LaunchStatus apply_development_run_options(
    RunnerArguments& arguments, bool development_pack);

struct ParsedRunArguments {
    bool ok = false;
    bool help = false;
    RunnerArguments arguments;
    std::string error;
};
// Parses everything after the program name and the "run" subcommand.
[[nodiscard]] ParsedRunArguments parse_runner_arguments(
    platform::PresentationMode mode, std::span<char* const> arguments);
[[nodiscard]] std::string runner_run_usage();

// Shared pso-archive subcommand for the headless and packed windowed players.
// --warm-only loads an existing archive without replacing it.
int run_pso_archive_command(int argc, char* argv[]);

// Required construction seams for tests. The session reports a structured
// host failure when one is missing or fails; the optional output seam is marked.
struct ShellEnvironment {
    std::function<runtime::LaunchResult<std::shared_ptr<runtime::GuestClock>>()> create_guest_clock;
    std::function<runtime::LaunchResult<std::shared_ptr<runtime::PresentationClock>>(
        std::shared_ptr<runtime::GuestClock>, runtime::PresentationClockConfiguration)>
        create_presentation_clock;
    std::function<platform::PresenterCreation(const platform::PresenterOptions&)> create_presenter;
    std::function<runtime::LaunchResult<std::unique_ptr<runtime::ProcessRuntime>>(
        const runtime::LaunchDescription&, runtime::RuntimeDependencies&&)>
        create_runtime;
    // Optional synthetic failure seam; empty opens the requested output file.
    std::function<std::unique_ptr<std::ostream>(const std::filesystem::path&)> open_input_log;
};
// Real clock and presenter creation. The process runtime factory is not linked
// in this build (lane F): it reports LaunchError::unsupported with the detail
// "Process runtime is not available in this build." and never fake success.
[[nodiscard]] ShellEnvironment default_shell_environment();

// Raises the process's soft limit on open files toward its hard limit (at most
// OPEN_MAX). A process started from Finder or a default shell gets 256, which
// a title's data files, the shader package and the GPU driver share. Call once
// at startup, before any file is opened for the run.
void raise_open_file_limit() noexcept;

// A never-destroyed holder for runtimes that outlive a bounded shutdown. A
// nonquiescent runtime must not be freed because a deadline elapsed; retention
// lasts until process exit so the destructor never runs in-process.
void retain_runtime_until_process_exit(std::shared_ptr<runtime::ProcessRuntime> runtime);
[[nodiscard]] std::size_t retained_runtime_count();

// Ends the process with exit_code, skipping static destruction, when a runtime
// was retained. A retained runtime still owns live workers (service, sampler
// and presentation threads) that lock process-wide statics; letting exit()
// destroy those statics under them aborts the process. Call it once the report
// is written; it returns normally when nothing was retained.
void exit_process_if_runtime_retained(int exit_code);

// The folder the shell extracts a --container input into when no cache
// directory is given: it is the shell's own and is removed when the run ends,
// just before a process with a retained runtime exits, at exit, and by a
// detached watcher once the process is gone, so a killed run leaves nothing.
// Empty when this process made none. A --cache directory is never removed.
[[nodiscard]] std::filesystem::path owned_container_directory();
void remove_owned_container_directory();

struct ShellReport {
    runtime::RunResult result;
    RunnerExitCode exit_code = RunnerExitCode::failure;
    // True when the shell kept the runtime alive past the run because teardown
    // could not complete within its bound.
    bool runtime_retained = false;
    // True only for container input whose NCA header, NPDM ACID and
    // NCA-to-NPDM signatures all verified. Extracted-directory input stays
    // not_verified; there is no partially verified state.
    bool metadata_signature_verified = false;
    // Container input provenance: whether an update container layered over
    // the base, and the program title id both containers carried.
    bool update_applied = false;
    std::uint64_t container_title_id = 0;
};

// Maps a finished run to its exit code. Teardown failure upgrades a
// success/guest-exit/cancelled outcome to failure; timeout keeps its own code
// because retention is the documented timeout behavior.
[[nodiscard]] RunnerExitCode classify_run_result(const runtime::RunResult& result);
[[nodiscard]] std::string outcome_name(const runtime::RunResult& result);

// Flat structured view of a report for the tool frontends (JSON in main.mm).
struct ShellReportFields {
    std::string outcome;
    int exit_code = 0;
    bool quiescent = false;
    bool runtime_retained = false;
    bool metadata_signature_verified = false;
    bool update_applied = false;
    std::uint64_t container_title_id = 0;
    runtime::RunStatistics statistics;
    std::optional<std::uint64_t> frame_target;
    std::optional<std::int64_t> guest_exit_status;
    std::optional<std::int64_t> timeout_limit_ns;
    std::optional<std::string> cancel_reason;
    std::optional<runtime::LaunchFailure> failure;
    std::optional<runtime::LaunchFailure> teardown_failure;
};
[[nodiscard]] ShellReportFields report_fields(const ShellReport& report);

// Headless orchestration: create clocks and presenter, create the runtime,
// run it on a worker thread, request stop at the time limit, attempt bounded
// shutdown, and map the outcome. Blocks the caller until the run finishes.
[[nodiscard]] ShellReport run_headless_shell(const RunnerArguments& arguments,
                                             const ShellEnvironment& environment);

// Main-thread window orchestration seam. create/pump/destroy run on the
// calling (main) thread; request_quit must be thread-safe.
class WindowedHost {
public:
    virtual ~WindowedHost() = default;
    virtual platform::ApplicationCreation create(const platform::PresenterOptions& options) = 0;
    [[nodiscard]] virtual std::shared_ptr<platform::Presenter> presenter() const noexcept = 0;
    virtual platform::DisplayCapabilities display_capabilities() const noexcept { return {}; }
    virtual void set_close_handler(std::function<void()> close) = 0;
    // Receives every refresh of the window's display, on a display thread,
    // from now on; an empty handler stops forwarding. Hosts without a display
    // link ignore it.
    virtual void set_display_refresh_handler(platform::DisplayRefreshCallback handler) {
        static_cast<void>(handler);
    }
    virtual void set_display_change_handler(platform::DisplayChangeCallback handler) {
        static_cast<void>(handler);
    }
    virtual bool set_display_refresh_rate(double rate) noexcept {
        static_cast<void>(rate);
        return false;
    }
    virtual platform::PresenterStatus pump(std::stop_token stop) = 0;
    virtual void request_quit() noexcept = 0;
};
using WindowedHostCreation =
    std::variant<platform::PresentationFailure, std::unique_ptr<WindowedHost>>;
using WindowedHostFactory = std::function<WindowedHostCreation()>;

// Called once after window creation, before clocks or title preparation. Only
// refresh_hz and launch environment should change here; window options already
// took effect. The default leaves headless-style CLI and iOS policy untouched.
using WindowedLaunchConfiguration =
    std::function<void(RunnerArguments&, const platform::DisplayCapabilities&)>;

// Windowed orchestration, called on the application main thread: create the
// host, run the runtime on a worker, pump until the worker finished shutdown,
// then request_quit and destroy the host. Window close flows through the host
// close handler into request_stop(window_closed).
[[nodiscard]] ShellReport run_windowed_shell(const RunnerArguments& arguments,
                                             const ShellEnvironment& environment,
                                             const WindowedHostFactory& host_factory,
                                             const WindowedLaunchConfiguration& configure_launch = {});

} // namespace armigo::tools
