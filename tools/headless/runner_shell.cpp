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
#include "input_script.h"

#include <armigo/compat/title_settings.h>
#include <algorithm>
#include <armigo/loader/container.h>
#include <armigo/runtime/preparation_format.h>

#include "runner_shell_internal.h"
#include "pacing_clock.h"
#include "input_log.h"

#include <armigo/services/file_system.h>

#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

#include <sys/resource.h>
#include <sys/syslimits.h>

namespace armigo::tools {
namespace {

using runtime::LaunchError;
using runtime::LaunchFailure;
using runtime::LaunchStage;

bool path_is_directory(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_directory(path, error) && !error;
}

bool path_is_regular_file(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

// FNV-1a over a path string, used to separate cache directories of
// different container inputs. It is a naming aid, not a security bound.
std::uint64_t input_path_digest(const std::filesystem::path& path) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const char c : path.string()) {
        hash ^= static_cast<std::uint8_t>(c);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string hex_digest(std::uint64_t value) {
    static const char* digits = "0123456789abcdef";
    std::string text(16, '0');
    for (std::size_t i = 0; i < 16; ++i) {
        text[15 - i] = digits[(value >> (4 * i)) & 0xF];
    }
    return text;
}

// Extracts a --container input into its cache directory with the user's
// keys. Only a fully signature-verified extraction produces input paths.
// With --update, the update's program NCA supplies ExeFS and a BKTR-patched
// RomFS over the base image; the cache name covers both inputs.
[[nodiscard]] std::variant<runtime::LaunchFailure, loader::ContainerExtraction>
extract_container_input(const RunnerArguments& arguments) {
    std::filesystem::path cache =
        arguments.cache_directory ? *arguments.cache_directory : std::filesystem::path{};
    if (cache.empty()) {
        std::error_code error;
        cache = std::filesystem::temp_directory_path(error);
        if (error) {
            return shell_detail::failure(runtime::LaunchStage::inputs,
                                         runtime::LaunchError::host_failure, "container input",
                                         "The system temporary path is unavailable.");
        }
        cache /= "armigo-container-" + std::to_string(::getpid()) + "-" +
                 hex_digest(input_path_digest(*arguments.container_file) ^
                            (arguments.update_container_file
                                 ? input_path_digest(*arguments.update_container_file)
                                 : 0));
    }
    loader::ContainerExtractionOptions options;
    auto keys = loader::read_container_keys_file(*arguments.keys_file);
    if (auto* failed = std::get_if<loader::ContainerKeysFailure>(&keys)) {
        return shell_detail::failure(
            runtime::LaunchStage::inputs, runtime::LaunchError::invalid_input, "container input",
            "Reading the keys file failed: " + failed->detail);
    }
    options.keys = std::move(std::get<loader::ContainerKeys>(keys));
    if (arguments.title_keys_file) {
        auto title_keys = loader::read_container_keys_file(*arguments.title_keys_file);
        if (auto* failed = std::get_if<loader::ContainerKeysFailure>(&title_keys)) {
            return shell_detail::failure(runtime::LaunchStage::inputs,
                                         runtime::LaunchError::invalid_input, "container input",
                                         "Reading the title keys file failed: " + failed->detail);
        }
        options.title_keys = std::move(std::get<loader::ContainerKeys>(title_keys));
    }
    if (arguments.update_container_file) {
        options.update_container_file = *arguments.update_container_file;
    }
    options.output_directory = cache;
    const auto extraction = loader::extract_program_container(*arguments.container_file, options);
    if (auto* failed = std::get_if<loader::ContainerFailure>(&extraction)) {
        return shell_detail::failure(
            runtime::LaunchStage::inputs, runtime::LaunchError::invalid_input, "container input",
            std::string(loader::to_string(failed->code)) + ": " + failed->detail);
    }
    return std::move(std::get<loader::ContainerExtraction>(extraction));
}

template <typename Integer>
bool parse_integer(const char* text, Integer& value) {
    if (!text || *text == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    if constexpr (std::is_signed_v<Integer>) {
        if (parsed < static_cast<long long>(std::numeric_limits<Integer>::min()) ||
            parsed > static_cast<long long>(std::numeric_limits<Integer>::max())) {
            return false;
        }
    } else {
        if (parsed < 0 ||
            static_cast<unsigned long long>(parsed) >
                static_cast<unsigned long long>(std::numeric_limits<Integer>::max())) {
            return false;
        }
    }
    value = static_cast<Integer>(parsed);
    return true;
}

bool parse_positive_nanoseconds(const char* text, std::chrono::nanoseconds& value) {
    if (!text || *text == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const double seconds = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(seconds) || seconds <= 0.0) {
        return false;
    }
    const double nanoseconds = seconds * 1e9;
    if (nanoseconds > 9.0e18) {
        return false;
    }
    value = std::chrono::nanoseconds(
        static_cast<std::chrono::nanoseconds::rep>(std::ceil(nanoseconds)));
    return value > std::chrono::nanoseconds::zero();
}

bool consume_value(std::span<char* const>& arguments, const char*& value) {
    if (arguments.empty()) {
        return false;
    }
    value = arguments.front();
    arguments = arguments.subspan(1);
    return value != nullptr;
}

bool parse_positive_double(const char* text, double& value) {
    if (!text || *text == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    value = std::strtod(text, &end);
    return errno == 0 && end != text && *end == '\0' && std::isfinite(value) && value > 0.0;
}

std::mutex& retained_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::shared_ptr<runtime::ProcessRuntime>>& retained_runtimes() {
    // Deliberately never destroyed: a retained runtime must not run a
    // destructor even during process exit teardown.
    static auto* runtimes = new std::vector<std::shared_ptr<runtime::ProcessRuntime>>();
    return *runtimes;
}

} // namespace

namespace shell_detail {

LaunchFailure failure(LaunchStage stage, LaunchError code, std::string operation,
                      std::string detail) {
    LaunchFailure result;
    result.stage = stage;
    result.code = code;
    result.operation = std::move(operation);
    result.detail = std::move(detail);
    return result;
}

ShellReport failed_report(LaunchFailure failure_value) {
    ShellReport report;
    report.result.outcome = std::move(failure_value);
    report.exit_code = classify_run_result(report.result);
    return report;
}

LaunchError presentation_error_to_launch_error(platform::PresentationError code) {
    switch (code) {
        case platform::PresentationError::invalid_frame:
            return LaunchError::invalid_input;
        case platform::PresentationError::unsupported:
            return LaunchError::unsupported;
        case platform::PresentationError::drawable_unavailable:
            return LaunchError::resource_limit;
        case platform::PresentationError::cancelled:
            return LaunchError::cancelled;
        case platform::PresentationError::deadline_expired:
            return LaunchError::deadline_expired;
        case platform::PresentationError::host_failure:
            return LaunchError::host_failure;
    }
    return LaunchError::host_failure;
}

runtime::LaunchResult<std::chrono::nanoseconds> refresh_period_from_hz(double refresh_hz) {
    if (!(refresh_hz > 0.0) || !std::isfinite(refresh_hz)) {
        return failure(LaunchStage::presentation, LaunchError::invalid_input,
                       "refresh period", "The refresh rate must be finite and positive.");
    }
    const double period_ns = 1e9 / refresh_hz;
    if (!std::isfinite(period_ns) || period_ns < 1.0) {
        return failure(LaunchStage::presentation, LaunchError::invalid_input,
                       "refresh period", "The refresh period is out of range.");
    }
    return std::chrono::nanoseconds(static_cast<std::chrono::nanoseconds::rep>(period_ns));
}

runtime::LaunchResult<std::shared_ptr<runtime::PresentationClock>> create_shell_clocks(
    const RunnerArguments& arguments, const ShellEnvironment& environment) {
    if (!environment.create_guest_clock || !environment.create_presentation_clock) {
        return failure(LaunchStage::services, LaunchError::host_failure, "create clocks",
                       "The shell environment is missing a clock factory.");
    }
    auto clock_result = environment.create_guest_clock();
    if (auto* failed = std::get_if<LaunchFailure>(&clock_result)) {
        return std::move(*failed);
    }
    auto clock = std::move(std::get<std::shared_ptr<runtime::GuestClock>>(clock_result));

    auto sample = clock->sample();
    if (auto* failed = std::get_if<LaunchFailure>(&sample)) {
        return std::move(*failed);
    }
    auto period = refresh_period_from_hz(arguments.refresh_hz);
    if (auto* failed = std::get_if<LaunchFailure>(&period)) {
        return std::move(*failed);
    }
    runtime::PresentationClockConfiguration configuration;
    configuration.epoch = std::get<runtime::ClockSample>(sample).instant;
    configuration.refresh_period = std::get<std::chrono::nanoseconds>(period);
    auto created = environment.create_presentation_clock(clock, configuration);
    if (std::holds_alternative<LaunchFailure>(created) ||
        (!arguments.pacing_schedule_file && !arguments.frame_log_file)) return created;
    if (arguments.mode != platform::PresentationMode::headless)
        return failure(LaunchStage::presentation, LaunchError::invalid_input, "headless.pacing",
                       "Pacing schedules and frame logs require headless mode.");
    std::optional<PacingSchedule> schedule;
    if (arguments.pacing_schedule_file) {
        std::ifstream input(*arguments.pacing_schedule_file);
        auto parsed = parse_pacing_schedule(input, arguments.pacing_max_hz.value_or(arguments.refresh_hz));
        if (const auto* error = std::get_if<std::string>(&parsed))
            return failure(LaunchStage::presentation, LaunchError::invalid_input, "headless.pacing", *error);
        schedule = std::get<PacingSchedule>(std::move(parsed));
    }
    std::unique_ptr<std::ofstream> log;
    std::vector<PacingWatch> watches;
    if (arguments.frame_log_file) {
        if (const char* list = std::getenv("ARMIGO_FRAME_WATCH")) {
            auto parsed = parse_pacing_watches(list);
            if (const auto* error = std::get_if<std::string>(&parsed))
                return failure(LaunchStage::presentation, LaunchError::invalid_input, "headless.pacing", *error);
            watches = std::get<std::vector<PacingWatch>>(std::move(parsed));
        }
        log = std::make_unique<std::ofstream>(*arguments.frame_log_file);
        if (!*log) return failure(LaunchStage::presentation, LaunchError::invalid_input,
                                "headless.pacing", "Could not open frame log.");
    }
    auto paced = std::make_shared<PacingClock>(
        std::get<std::shared_ptr<runtime::PresentationClock>>(std::move(created)),
        std::move(schedule), std::move(log), std::move(watches));
    if (auto error = paced->observation_failure()) return *error;
    return std::shared_ptr<runtime::PresentationClock>(std::move(paced));
}

runtime::LaunchResult<platform::PresenterOptions> shell_presenter_options(
    const RunnerArguments& arguments) {
    platform::PresenterOptions options;
    options.mode = arguments.mode;
    options.width = arguments.width;
    options.height = arguments.height;
    options.fullscreen = arguments.fullscreen;
    options.content_width = arguments.content_width;
    options.display_sync = arguments.display_sync;
    return options;
}

runtime::LaunchDescription build_launch_description(
    const RunnerArguments& arguments, const platform::PresenterOptions& presentation) {
    runtime::LaunchDescription description;

    description.nup_file = arguments.nup_file.value_or(std::filesystem::path{});
    description.exefs_directory = arguments.exefs_directory;
    description.romfs_directory = arguments.romfs_directory;
    description.addon_directories = arguments.addon_directories;
    description.npdm_file = arguments.npdm_file;
    if (arguments.prepared_directory) {
        description.prepared_directory = *arguments.prepared_directory;
    }
    if (arguments.embedded_directory) description.embedded_directory = *arguments.embedded_directory;
    description.shader_package_directory = arguments.shader_package_directory;
    description.operation_mode = arguments.operation_mode;
    // Owner policy: user-supplied extracted directories or raw NUP files receive
    // structural and capability validation; the NPDM signature is not
    // verified (that belongs to container packaging) and reports say so.
    description.npdm_trust = runtime::NpdmTrustPolicy::structurally_validated_extracted_input;
    description.presentation = presentation;
    // Saves are host files in the given directory; without one, the windowed
    // player keeps them in the title's save directory and a headless run
    // keeps them for the run only.
    if (arguments.save_directory) services::fs::set_host_save_directory(*arguments.save_directory);
    else description.title_saves = arguments.mode == platform::PresentationMode::windowed;
    return description;
}

runtime::RunResult run_runtime_with_limits(
    const std::shared_ptr<runtime::ProcessRuntime>& runtime, const RunnerArguments& arguments,
    const runtime::RunLimits& limits, const runtime::ReportDestination& destination) {
    runtime::RunResult run_result;
    std::mutex finished_mutex;
    std::condition_variable finished_signal;
    bool finished = false;
    bool timeout_requested = false;

    // The runtime's own RunLimits timeout is the primary bound; this watcher
    // only requests cooperative stop if the runtime never returns on its own.
    std::thread stop_watcher;
    if (arguments.time_limit) {
        const auto deadline = std::chrono::steady_clock::now() + *arguments.time_limit;
        stop_watcher = std::thread([&, deadline] {
            std::unique_lock lock(finished_mutex);
            if (!finished_signal.wait_until(lock, deadline, [&] { return finished; })) {
                timeout_requested = true;
                runtime->request_stop(runtime::StopReason::watchdog_timeout);
            }
        });
    }

    std::thread worker([&] {
        run_result = runtime->run(limits, destination);
    });
    worker.join();
    {
        const std::lock_guard lock(finished_mutex);
        finished = true;
    }
    finished_signal.notify_all();
    if (stop_watcher.joinable()) {
        stop_watcher.join();
    }
    // The shell watchdog may win the race with the runtime's own deadline.
    // Its stop has timeout provenance, not a user or window-close request.
    if (timeout_requested) {
        const auto* cancelled = std::get_if<runtime::RunCancelled>(&run_result.outcome);
        if (cancelled && cancelled->reason == runtime::StopReason::watchdog_timeout)
            run_result.outcome = runtime::RunTimeout{*arguments.time_limit};
    }
    return run_result;
}

void finish_run(const std::shared_ptr<runtime::ProcessRuntime>& runtime, ShellReport& report) {
    // Bounded teardown: a failure keeps the runtime owned and alive instead of
    // destroying a nonquiescent runtime.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto status = runtime->shutdown(deadline);
    if (auto* failed = std::get_if<LaunchFailure>(&status)) {
        if (!report.result.teardown_failure) {
            report.result.teardown_failure = *failed;
        }
        report.result.quiescent = false;
        retain_runtime_until_process_exit(runtime);
        report.runtime_retained = true;
        return;
    }
    report.result.quiescent = true;
}

} // namespace shell_detail

ParsedRunArguments parse_runner_arguments(platform::PresentationMode mode,
                                          std::span<char* const> arguments) {
    ParsedRunArguments parsed;
    RunnerArguments& options = parsed.arguments;
    options.mode = mode;

    while (!arguments.empty()) {
        const std::string_view flag = arguments.front();
        arguments = arguments.subspan(1);
        const char* value = nullptr;
        if (flag == "--help") {
            parsed.help = true;
            parsed.ok = true;
            return parsed;
        }
        bool argument_error = false;
        if (flag == "--nup") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.nup_file = value;
        } else if (flag == "--exefs") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.exefs_directory = value;
        } else if (flag == "--romfs") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.romfs_directory = value;
        } else if (flag == "--addon") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.addon_directories.emplace_back(value);
        } else if (flag == "--npdm") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.npdm_file = value;
        } else if (flag == "--prepared") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.prepared_directory = value;
        } else if (flag == "--embedded") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.embedded_directory = value;
        } else if (flag == "--shaders") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.shader_package_directory = value;
        } else if (flag == "--report") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.report_file = value;
        } else if (flag == "--container") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.container_file = value;
        } else if (flag == "--update") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.update_container_file = value;
        } else if (flag == "--title-keys") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.title_keys_file = value;
        } else if (flag == "--keys") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.keys_file = value;
        } else if (flag == "--cache") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.cache_directory = value;
        } else if (flag == "--capture") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.capture_directory = value;
        } else if (flag == "--input-script") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.input_script_file = value;
        } else if (flag == "--saves") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) options.save_directory = value;
        } else if (flag == "--operation-mode") {
            argument_error = !consume_value(arguments, value);
            if (!argument_error) {
                const std::string_view selected = value;
                if (selected == "console")
                    options.operation_mode = services::OperationMode::Console;
                else if (selected == "handheld")
                    options.operation_mode = services::OperationMode::Handheld;
                else {
                    parsed.error = "--operation-mode requires console or handheld.";
                    return parsed;
                }
            }
        } else if (flag == "--frames") {
            std::uint64_t frames = 0;
            if (!consume_value(arguments, value) ||
                !parse_integer(value, frames) || frames == 0) {
                parsed.error = "--frames requires a positive integer.";
                return parsed;
            }
            options.completed_frame_target = frames;
        } else if (flag == "--time-limit") {
            std::chrono::nanoseconds limit{};
            if (!consume_value(arguments, value) ||
                !parse_positive_nanoseconds(value, limit)) {
                parsed.error = "--time-limit requires a positive number of seconds.";
                return parsed;
            }
            options.time_limit = limit;
        } else if (flag == "--pacing-schedule" || flag == "--frame-log" || flag == "--input-log" || flag == "--pacing-max-hz") {
            if (mode != platform::PresentationMode::headless || !consume_value(arguments, value)) {
                parsed.error = "Pacing options require headless mode and a value.";
                return parsed;
            }
            if (flag == "--pacing-schedule") options.pacing_schedule_file = value;
            else if (flag == "--frame-log") options.frame_log_file = value;
            else if (flag == "--input-log") options.input_log_file = value;
            else {
                double ceiling = 0;
                if (!parse_positive_double(value, ceiling) || ceiling < 1 || ceiling > 1000) {
                    parsed.error = "--pacing-max-hz requires a rate between 1 and 1000.";
                    return parsed;
                }
                options.pacing_max_hz = ceiling;
            }
        } else if (flag == "--refresh-hz") {
            double refresh_hz = 0.0;
            if (!consume_value(arguments, value) ||
                !parse_positive_double(value, refresh_hz)) {
                parsed.error = "--refresh-hz requires a positive number.";
                return parsed;
            }
            options.refresh_hz = refresh_hz;
            options.refresh_hz_explicit = true;
        } else if (flag == "--width") {
            std::uint32_t width = 0;
            if (!consume_value(arguments, value) ||
                !parse_integer(value, width)) {
                parsed.error = "--width requires a nonnegative integer.";
                return parsed;
            }
            options.width = width;
        } else if (flag == "--height") {
            std::uint32_t height = 0;
            if (!consume_value(arguments, value) ||
                !parse_integer(value, height)) {
                parsed.error = "--height requires a nonnegative integer.";
                return parsed;
            }
            options.height = height;
        } else if (flag == "--fullscreen") {
            options.fullscreen = true;
        } else {
            parsed.error = "Unknown run option: " + std::string(flag) + ".";
            return parsed;
        }
        if (argument_error) {
            parsed.error = "Option " + std::string(flag) + " requires a value.";
            return parsed;
        }
    }

    // A packed app carries its game under Armigo/ in its resources. Given no
    // input, it runs that game with its add-on content and its shader package.
    if (!options.nup_file && !options.container_file && options.exefs_directory.empty() && options.romfs_directory.empty()) {
        if (const auto packed = compat::packed_resources(); !packed.empty()) {
            options.exefs_directory = packed / "game" / "exefs";
            options.romfs_directory = packed / "game" / "romfs";
            if (options.addon_directories.empty()) {
                std::error_code error;
                for (const auto& entry : std::filesystem::directory_iterator(packed / "game" / "addons", error))
                    if (entry.is_directory(error)) options.addon_directories.push_back(entry.path());
                std::sort(options.addon_directories.begin(), options.addon_directories.end());
            }
            if (options.shader_package_directory.empty() && path_is_directory(packed / "shaders"))
                options.shader_package_directory = packed / "shaders";
        }
    }

    if (options.nup_file) {
        if (options.mode != platform::PresentationMode::headless || options.container_file ||
            options.keys_file || options.title_keys_file || options.update_container_file ||
            options.cache_directory || !options.exefs_directory.empty() || !options.romfs_directory.empty()) {
            parsed.error = "--nup requires headless mode and excludes container, key, cache and directory inputs.";
            return parsed;
        }
        if (!path_is_regular_file(*options.nup_file)) {
            parsed.error = "--nup is not a readable file.";
            return parsed;
        }
    } else if (options.container_file) {
        if (options.mode != platform::PresentationMode::headless) {
            parsed.error = "--container is only supported by the headless shell.";
            return parsed;
        }
        if (!options.keys_file) {
            parsed.error = "--container requires --keys.";
            return parsed;
        }
        if (!options.exefs_directory.empty()) {
            parsed.error = "--container and --exefs are mutually exclusive.";
            return parsed;
        }
        if (!options.romfs_directory.empty()) {
            parsed.error = "--container and --romfs are mutually exclusive.";
            return parsed;
        }
        if (options.update_container_file) {
            if (!path_is_regular_file(*options.update_container_file)) {
                parsed.error = "--update is not a readable file: " +
                               options.update_container_file->string() + ".";
                return parsed;
            }
        }
        if (options.title_keys_file && !path_is_regular_file(*options.title_keys_file)) {
            parsed.error = "--title-keys is not a readable file: " +
                           options.title_keys_file->string() + ".";
            return parsed;
        }
        if (!path_is_regular_file(*options.container_file)) {
            parsed.error = "--container is not a readable file: " +
                           options.container_file->string() + ".";
            return parsed;
        }
        if (!path_is_regular_file(*options.keys_file)) {
            parsed.error =
                "--keys is not a readable file: " + options.keys_file->string() + ".";
            return parsed;
        }
        if (options.cache_directory && options.cache_directory->empty()) {
            parsed.error = "--cache requires a directory path.";
            return parsed;
        }
    } else {
        if (options.update_container_file) {
            parsed.error = "--update requires --container.";
            return parsed;
        }
        if (options.title_keys_file) {
            parsed.error = "--title-keys requires --container.";
            return parsed;
        }
        if (options.keys_file) {
            parsed.error = "--keys requires --container.";
            return parsed;
        }
        if (options.cache_directory) {
            parsed.error = "--cache requires --container.";
            return parsed;
        }
        if (options.exefs_directory.empty()) {
            parsed.error = "--exefs is required.";
            return parsed;
        }
        if (!path_is_directory(options.exefs_directory)) {
            parsed.error =
                "--exefs is not a readable directory: " + options.exefs_directory.string() + ".";
            return parsed;
        }
        if (options.romfs_directory.empty()) {
            parsed.error = "--romfs is required.";
            return parsed;
        }
        if (!path_is_directory(options.romfs_directory)) {
            parsed.error =
                "--romfs is not a readable directory: " + options.romfs_directory.string() + ".";
            return parsed;
        }
    }
    for (const auto& addon : options.addon_directories) {
        if (!path_is_directory(addon)) {
            parsed.error = "--addon is not a readable directory: " + addon.string() + ".";
            return parsed;
        }
    }
    // Without --shaders a title that uses the NVN driver prepares its stages at
    // run time; any other title then has no stages, and its draws report that.
    if (!options.shader_package_directory.empty() && !path_is_directory(options.shader_package_directory)) {
        parsed.error = "--shaders is not a readable directory: " +
                       options.shader_package_directory.string() + ".";
        return parsed;
    }
    if (options.npdm_file && !path_is_regular_file(*options.npdm_file)) {
        parsed.error =
            "--npdm is not a readable file: " + options.npdm_file->string() + ".";
        return parsed;
    }
    if (options.prepared_directory && options.embedded_directory) {
        parsed.error = "--prepared and --embedded are mutually exclusive.";
        return parsed;
    }
    if (options.embedded_directory && !path_is_directory(*options.embedded_directory)) {
        parsed.error = "--embedded is not a readable directory: " + options.embedded_directory->string() + ".";
        return parsed;
    }
    if (options.prepared_directory && !path_is_directory(*options.prepared_directory)) {
        parsed.error = "--prepared is not a readable directory: " +
                       options.prepared_directory->string() + ".";
        return parsed;
    }
    if ((options.width == 0) != (options.height == 0)) {
        parsed.error = "--width and --height must be given together.";
        return parsed;
    }
    if (options.input_log_file && !options.input_script_file) {
        parsed.error = "--input-log requires --input-script.";
        return parsed;
    }
    if (options.input_log_file && options.frame_log_file &&
        std::filesystem::absolute(*options.input_log_file).lexically_normal() ==
        std::filesystem::absolute(*options.frame_log_file).lexically_normal()) {
        parsed.error = "Input and frame logs require different paths.";
        return parsed;
    }
    if (options.pacing_max_hz && !options.pacing_schedule_file) {
        parsed.error = "--pacing-max-hz requires --pacing-schedule.";
        return parsed;
    }
    if (!(options.refresh_hz > 0.0) || !std::isfinite(options.refresh_hz)) {
        parsed.error = "--refresh-hz requires a positive number.";
        return parsed;
    }

    parsed.ok = true;
    return parsed;
}

std::string runner_run_usage() {
    return "Options:\n"
           "  --exefs DIR --romfs DIR\n"
           "  --nup FILE            Read a raw NUP package lazily; no keys or extraction.\n"
           "                         Launch extracted directories (default input mode).\n"
           "  --container FILE --keys FILE [--update FILE] [--title-keys FILE]\n"
           "                         [--cache DIR]\n"
           "                         Extract a user-supplied NSP/XCI with the user's\n"
           "                         keys; all container signatures must verify.\n"
           "                         --update layers an update NSP/XCI over the\n"
           "                         base by title id (titlekey + BKTR RomFS);\n"
           "                         --title-keys supplies rights-id keys when\n"
           "                         no ticket matches. Headless only.\n"
           "  --addon DIR            Install extracted add-on content: DIR is its RomFS\n"
           "                         tree, named by its data id (16 hex digits).\n"
           "                         Repeat for each item.\n"
           "  --npdm FILE            Override the collected main.npdm (bytes must match).\n"
           "  --prepared DIR         Directory of prepared module artifacts.\n"
           "  --embedded DIR         Directory of embedded native images.\n"
           "  --frames N             Stop after N completed presentations (positive).\n"
           "  --time-limit S         Request stop after S seconds (positive).\n"
           "  --report FILE          Write the structured result to FILE.\n"
           "  --input-script FILE    Controller events; presents or elapsed clock.\n"
           "  --capture DIR          Runtime capture directory.\n"
           "  --saves DIR            Keep save data in DIR between runs.\n"
           "  --operation-mode MODE  Guest operation mode: console or handheld\n"
           "                         (default console).\n"
           "  --width N --height N   Present at this host size.\n"
           "  --fullscreen           Present fullscreen.\n"
           "  --pacing-schedule FILE Headless elapsed-time rate/hitch schedule (default off).\n"
           "  --pacing-max-hz HZ     Schedule ceiling (default --refresh-hz).\n"
           "  --input-log FILE      Log actual scripted HID deliveries (headless only).\n"
           "  --frame-log FILE      Headless per-present TSV; ARMIGO_FRAME_WATCH=address:width,...\n"
           "  --refresh-hz HZ        Host presentation refresh rate (default 60).\n"
           "Exit codes: 0 frame-target success or window closed, 1 guest exit,\n"
           "  2 usage error, 3 unsupported stop, 4 failure, 5 timeout, 6 cancelled.\n"
           "A timeout never frees a nonquiescent runtime; hard termination\n"
           "requires an external spawn/exec supervisor (there is no guest fork path).";
}

void raise_open_file_limit() noexcept {
    rlimit limit{};
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) return;
    const rlim_t wanted = std::min<rlim_t>(limit.rlim_max, OPEN_MAX);
    if (limit.rlim_cur >= wanted) return;
    limit.rlim_cur = wanted;
    static_cast<void>(setrlimit(RLIMIT_NOFILE, &limit));
}

ShellEnvironment default_shell_environment() {
    ShellEnvironment environment;
    environment.create_guest_clock = [] {
        return runtime::create_guest_clock(runtime::GuestClockSources{});
    };
    environment.create_presentation_clock =
        [](std::shared_ptr<runtime::GuestClock> clock,
           runtime::PresentationClockConfiguration configuration) {
            return runtime::create_presentation_clock(std::move(clock), configuration);
        };
    environment.create_presenter = [](const platform::PresenterOptions& options) {
        return platform::create_presenter(options);
    };
    environment.create_runtime =
        [](const runtime::LaunchDescription& description, runtime::RuntimeDependencies&& dependencies) {
            dependencies.use_platform_memory_budget = true;
            dependencies.preparation_options = {
                runtime::installed_preparation_identity(), runtime::default_prepared_artifact_limit};
            return runtime::ProcessRuntime::create(description, std::move(dependencies));
        };
    return environment;
}

void retain_runtime_until_process_exit(std::shared_ptr<runtime::ProcessRuntime> runtime) {
    const std::lock_guard lock(retained_mutex());
    retained_runtimes().push_back(std::move(runtime));
}

std::size_t retained_runtime_count() {
    const std::lock_guard lock(retained_mutex());
    return retained_runtimes().size();
}

void exit_process_if_runtime_retained(int exit_code) {
    if (retained_runtime_count() == 0) return;
    std::fflush(nullptr);
    std::_Exit(exit_code);
}

RunnerExitCode classify_run_result(const runtime::RunResult& result) {
    const bool teardown_failed = result.teardown_failure.has_value();
    return std::visit(
        [&](const auto& outcome) -> RunnerExitCode {
            using Outcome = std::decay_t<decltype(outcome)>;
            if constexpr (std::is_same_v<Outcome, runtime::LaunchFailure>) {
                return outcome.code == LaunchError::unsupported ? RunnerExitCode::unsupported
                                                                : RunnerExitCode::failure;
            } else if constexpr (std::is_same_v<Outcome, runtime::RunSuccess>) {
                return teardown_failed ? RunnerExitCode::failure : RunnerExitCode::success;
            } else if constexpr (std::is_same_v<Outcome, runtime::GuestExit>) {
                return teardown_failed ? RunnerExitCode::failure : RunnerExitCode::guest_exit;
            } else if constexpr (std::is_same_v<Outcome, runtime::RunTimeout>) {
                return RunnerExitCode::timeout;
            } else {
                // Closing the window is how a player quits: a clean stop is success.
                if (teardown_failed) return RunnerExitCode::failure;
                return outcome.reason == runtime::StopReason::window_closed ? RunnerExitCode::success
                                                                             : RunnerExitCode::cancelled;
            }
        },
        result.outcome);
}

std::string outcome_name(const runtime::RunResult& result) {
    return std::visit(
        [&](const auto& outcome) -> std::string {
            using Outcome = std::decay_t<decltype(outcome)>;
            if constexpr (std::is_same_v<Outcome, runtime::LaunchFailure>) {
                return outcome.code == LaunchError::unsupported ? "unsupported" : "failure";
            } else if constexpr (std::is_same_v<Outcome, runtime::RunSuccess>) {
                return "success";
            } else if constexpr (std::is_same_v<Outcome, runtime::GuestExit>) {
                return "guest-exit";
            } else if constexpr (std::is_same_v<Outcome, runtime::RunTimeout>) {
                return "timeout";
            } else {
                return "cancelled";
            }
        },
        result.outcome);
}

ShellReportFields report_fields(const ShellReport& report) {
    ShellReportFields fields;
    fields.outcome = outcome_name(report.result);
    fields.exit_code = static_cast<int>(report.exit_code);
    fields.quiescent = report.result.quiescent;
    fields.runtime_retained = report.runtime_retained;
    fields.metadata_signature_verified = report.metadata_signature_verified;
    fields.update_applied = report.update_applied;
    fields.container_title_id = report.container_title_id;
    fields.statistics = report.result.statistics;
    if (const auto* success = std::get_if<runtime::RunSuccess>(&report.result.outcome)) {
        fields.frame_target = success->frame_target;
    }
    if (const auto* exit = std::get_if<runtime::GuestExit>(&report.result.outcome)) {
        fields.guest_exit_status = exit->status;
    }
    if (const auto* timeout = std::get_if<runtime::RunTimeout>(&report.result.outcome)) {
        fields.timeout_limit_ns = timeout->limit.count();
    }
    if (const auto* cancelled = std::get_if<runtime::RunCancelled>(&report.result.outcome)) {
        fields.cancel_reason = cancelled->reason == runtime::StopReason::window_closed
                                   ? std::string("window_closed")
                                   : std::string("user_request");
    }
    if (const auto* failed = std::get_if<runtime::LaunchFailure>(&report.result.outcome)) {
        fields.failure = *failed;
    }
    fields.teardown_failure = report.result.teardown_failure;
    return fields;
}

ShellReport run_headless_shell(const RunnerArguments& arguments,
                               const ShellEnvironment& environment) {
    InputScript input_script;
    if (arguments.input_script_file) {
        std::ifstream source(*arguments.input_script_file);
        auto parsed = source ? parse_input_script(source)
                             : std::variant<InputScript, std::string>("Could not open input script.");
        if (auto* error = std::get_if<std::string>(&parsed)) {
            return shell_detail::failed_report(shell_detail::failure(
                runtime::LaunchStage::inputs, LaunchError::invalid_input, "input script", *error));
        }
        input_script = std::move(std::get<InputScript>(parsed));
    }
    // Validate external inputs before clocks, presentation, or guest execution.
    RunnerArguments effective = arguments;
    bool metadata_signature_verified = false;
    bool update_applied = false;
    std::uint64_t container_title_id = 0;
    if (effective.container_file) {
        auto extracted = extract_container_input(arguments);
        if (auto* failed = std::get_if<runtime::LaunchFailure>(&extracted)) {
            return shell_detail::failed_report(std::move(*failed));
        }
        const auto& paths = std::get<loader::ContainerExtraction>(extracted);
        effective.exefs_directory = paths.exefs_directory;
        effective.romfs_directory = paths.romfs_directory;
        metadata_signature_verified = true;
        update_applied = paths.update_applied;
        container_title_id = paths.title_id;
    }

    auto clock_result = shell_detail::create_shell_clocks(effective, environment);
    if (auto* failed = std::get_if<runtime::LaunchFailure>(&clock_result)) {
        return shell_detail::failed_report(std::move(*failed));
    }
    auto presentation_clock = std::move(
        std::get<std::shared_ptr<runtime::PresentationClock>>(clock_result));

    auto presenter_options = shell_detail::shell_presenter_options(effective);
    if (auto* failed = std::get_if<runtime::LaunchFailure>(&presenter_options)) {
        return shell_detail::failed_report(std::move(*failed));
    }
    if (!environment.create_presenter) {
        return shell_detail::failed_report(shell_detail::failure(
            runtime::LaunchStage::graphics, LaunchError::host_failure, "create_presenter",
            "The shell environment is missing a presenter factory."));
    }
    auto presenter_result = environment.create_presenter(
        std::get<platform::PresenterOptions>(presenter_options));
    if (auto* failed = std::get_if<platform::PresentationFailure>(&presenter_result)) {
        return shell_detail::failed_report(shell_detail::failure(
            runtime::LaunchStage::graphics,
            shell_detail::presentation_error_to_launch_error(failed->code), "create_presenter",
            failed->detail));
    }
    auto presenter = std::move(std::get<std::shared_ptr<platform::Presenter>>(presenter_result));

    if (!environment.create_runtime) {
        return shell_detail::failed_report(shell_detail::failure(
            runtime::LaunchStage::services, LaunchError::host_failure,
            "ProcessRuntime::create",
            "The shell environment is missing a runtime factory."));
    }
    auto description = shell_detail::build_launch_description(
        effective, std::get<platform::PresenterOptions>(presenter_options));
    if (container_title_id != 0) description.container_title_id = container_title_id;
    runtime::RuntimeDependencies dependencies;
    dependencies.clock = presentation_clock->clock();
    InputDeliveryObserver input_observer;
    InputDeliveryFlush flush_input;
    if (arguments.input_script_file && (arguments.frame_log_file || arguments.input_log_file)) {
        const auto path = arguments.input_log_file.value_or(
            std::filesystem::path(arguments.frame_log_file.value_or(std::filesystem::path{}).string() + ".inputs.tsv"));
        auto delivery_log = std::make_shared<InputDeliveryLog>(presentation_clock->clock(),
            presentation_clock->configuration().epoch, input_script.clock,
            environment.open_input_log ? environment.open_input_log(path) : std::make_unique<std::ofstream>(path));
        if (!delivery_log->ready())
            return shell_detail::failed_report(shell_detail::failure(runtime::LaunchStage::inputs,
                LaunchError::host_failure, "input delivery log", "Could not open or initialize input delivery log."));
        input_observer = [delivery_log](std::size_t index, const ScriptInputState& state) {
            return delivery_log->record(index, state);
        };
        flush_input = [delivery_log] { return delivery_log->flush(); };
    }
    dependencies.presentation_clock = std::move(presentation_clock);
    platform::PresenterStatus initial_input_status = std::monostate{};
    dependencies.presenter = arguments.input_script_file
        ? scripted_presenter(presenter, std::move(input_script), {}, {}, std::move(input_observer),
                             std::move(flush_input), &initial_input_status) : presenter;
    if (const auto* error = std::get_if<platform::PresentationFailure>(&initial_input_status))
        return shell_detail::failed_report(shell_detail::failure(runtime::LaunchStage::inputs,
            LaunchError::host_failure, "input delivery log", error->detail));
    auto runtime_result = environment.create_runtime(description, std::move(dependencies));
    if (auto* failed = std::get_if<runtime::LaunchFailure>(&runtime_result)) {
        return shell_detail::failed_report(std::move(*failed));
    }
    auto runtime = std::shared_ptr<runtime::ProcessRuntime>(
        std::move(std::get<std::unique_ptr<runtime::ProcessRuntime>>(runtime_result)));

    runtime::RunLimits limits;
    limits.completed_frame_target = arguments.completed_frame_target;
    limits.timeout = arguments.time_limit;
    runtime::ReportDestination destination;
    destination.capture_directory = arguments.capture_directory;
    destination.report_file = arguments.report_file;

    ShellReport report;
    report.result =
        shell_detail::run_runtime_with_limits(runtime, effective, limits, destination);
    shell_detail::finish_run(runtime, report);
    report.exit_code = classify_run_result(report.result);
    report.metadata_signature_verified = metadata_signature_verified;
    report.update_applied = update_applied;
    report.container_title_id = container_title_id;
    return report;
}

} // namespace armigo::tools
