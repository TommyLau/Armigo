// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "process_runtime_fixture.h"
#include "player_settings.h"
#include "../../src/runtime/service_registry.h"
#include <future>
#include <cmath>
#include <armigo/common/diagnostics.h>
#include <armigo/compat/title_settings.h>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace {
void test_explicit_and_platform_memory_budgets() {
    for (bool platform_default : {false, true}) {
        Fixture fixture;
        auto deps = dependencies();
        const auto explicit_budget = deps.memory_options.resource_limit_bytes;
        deps.use_platform_memory_budget = platform_default;
        if (platform_default) deps.memory_options.resource_limit_bytes = 0;
        std::uint64_t observed = 0;
        deps.create_services = [&observed](ServiceRegistryDependencies setup)
            -> LaunchResult<std::shared_ptr<ServiceRegistry>> {
            observed = setup.memory->process()->total_memory();
            return create_service_registry(std::move(setup));
        };
        const auto made = ProcessRuntime::create(fixture.description, std::move(deps));
        CHECK(value(made) != nullptr);
        CHECK(observed == (platform_default ? std::uint64_t{4} << 30 : explicit_budget));
    }
}

void test_title_rate_clock_warning_for_both_runners() {
    Fixture fixture;
    const std::uint64_t title = 0xffff000000000000ull |
        (std::uint64_t(::getpid()) << 16) | (arc4random() & 0xffffu);
    fixture.description.synthetic_profile->title_id = title;
    const auto cache = compat::title_cache_directory(title);
    CHECK(!std::filesystem::exists(cache));
    if (std::filesystem::exists(cache)) return;
    const auto image = loader::read_nso_module(fixture.input.modules.front().bytes);
    CHECK(image.ok()); if (!image.ok()) return;
    const auto offset = fixture_offset(armigo_process_fixture_main) + 8;
    std::uint32_t expected = 0;
    std::memcpy(&expected, image.image->bytes.data() + offset, 4);
    std::ostringstream identity;
    identity << "{\"title_id\":\"0x" << compat::title_id_text(title) << "\",\"main_build_id\":\"";
    for (auto byte : image.image->build_id)
        identity << std::hex << std::setfill('0') << std::setw(2) << std::to_integer<unsigned>(byte);
    identity << "\"";
    std::ostringstream policy;
    policy << ",\"frame_rate\":{\"default\":30,\"display_refresh_rate\":60},"
              "\"frame_rates\":[30,120],\"words\":[{\"offset\":\"0x" << std::hex << offset
           << "\",\"expected_instruction\":\"0x" << std::setfill('0') << std::setw(8) << expected
           << "\",\"replacement_instruction\":\"0xd503201f\",\"frame_rate\":120,"
              "\"note\":\"Synthetic clock diagnostic word; guest code is not executed.\"}]";
    const auto titles = fixture.temporary.path / "titles";
    const auto folder = titles / compat::title_id_text(title);
    std::filesystem::create_directories(folder);
    const auto environment = [](const char* name) -> std::optional<std::string> {
        if (const char* value = std::getenv(name)) return std::string(value);
        return {};
    };
    const auto old_titles = environment("ARMIGO_TITLES");
    const auto old_rate = environment("ARMIGO_COMPAT_FRAME_RATE");
    const auto old_refresh = environment("ARMIGO_VI_REFRESH_HZ");
    const auto old_resolution_rate = environment("ARMIGO_COMPAT_RESOLUTION_RATE");
    const auto old_level = diag::minimum_level();
    setenv("ARMIGO_TITLES", titles.c_str(), 1);
    setenv("ARMIGO_VI_REFRESH_HZ", "120", 1);
    diag::testing_set_level(diag::Level::warning);
    struct Captured { std::mutex mutex; std::vector<std::string> warnings; } captured;
    diag::testing_set_sink([](diag::Level level, std::string_view line, void* context) {
        if (level != diag::Level::warning ||
            (line.find("admitted title frame rate") == std::string_view::npos &&
             line.find("Armigo settings: stored") == std::string_view::npos)) return;
        auto& output = *static_cast<Captured*>(context);
        const std::lock_guard lock(output.mutex);
        output.warnings.emplace_back(line);
    }, &captured);
    struct Case { const char* requested; std::chrono::nanoseconds period; bool policy; bool warning; };
    for (auto mode : {platform::PresentationMode::headless, platform::PresentationMode::windowed}) {
        for (const auto& test : {Case{"120", 16'666'667ns, true, true},
                                Case{"120", 8'333'333ns, true, false},
                                Case{"0", 16'666'667ns, true, false},
                                Case{"120", 16'666'667ns, false, false}}) {
            std::ofstream(folder / "profile.json") << "{\"version\":1,\"patches\":[" << identity.str()
                << (test.policy ? policy.str() : "") << "}]}";
            setenv("ARMIGO_COMPAT_FRAME_RATE", test.requested, 1);
            fixture.description.presentation.mode = mode;
            auto presenter = std::make_shared<TestPresenter>(mode);
            auto deps = dependencies(presenter);
            auto configuration = deps.presentation_clock->configuration();
            { const std::lock_guard lock(captured.mutex); captured.warnings.clear(); }
            configuration.refresh_period = test.period;
            if (mode == platform::PresentationMode::windowed) {
                // The real player callback must preserve the explicit override,
                // suppress UI fallback warnings and leave only the runtime's
                // admitted-rate mismatch warning for a slower clock.
                tools::PlayerChoices choices;
                choices.title_frame_rate = test.policy;
                choices.supported_frame_rates = {30, 120};
                choices.frame_rate = 120;
                choices.display_refresh_rate = 60;
                tools::RunnerArguments arguments;
                arguments.refresh_hz = 1e9 / static_cast<double>(test.period.count());
                arguments.refresh_hz_explicit = test.period != 8'333'333ns;
                platform::DisplayCapabilities display;
                display.current = {1920, 1080, arguments.refresh_hz_explicit ? 60.0 : 120.0, false};
                tools::mac_player_launch_configuration(choices, 0)(arguments, display);
                CHECK(std::string_view(std::getenv("ARMIGO_COMPAT_FRAME_RATE")) == test.requested);
                configuration.refresh_period = std::chrono::nanoseconds(std::llround(1e9 / arguments.refresh_hz));
            }
            auto selected_clock = create_presentation_clock(deps.clock, configuration);
            CHECK(value(selected_clock));
            if (!value(selected_clock)) continue;
            deps.presentation_clock = *value(selected_clock);
            auto clock = deps.presentation_clock;
            auto created = ProcessRuntime::create(fixture.description, std::move(deps));
            CHECK(value(created));
            if (const auto* error = failure(created))
                std::fprintf(stderr, "Clock diagnostic fixture: %s: %s\n", error->operation.c_str(), error->detail.c_str());
            if (value(created)) CHECK(ok((*value(created))->shutdown(std::chrono::steady_clock::now() + 1s)));
            CHECK(clock->configuration().refresh_period == test.period);
            CHECK(presenter->present_calls == 0);
            const std::lock_guard lock(captured.mutex);
            CHECK(captured.warnings.size() == (test.warning ? 1u : 0u));
            if (!captured.warnings.empty()) {
                CHECK(captured.warnings.front().find("120 fps") != std::string::npos);
                CHECK(captured.warnings.front().find("--refresh-hz") != std::string::npos);
                CHECK(captured.warnings.front().find("continuing with the configured clock") != std::string::npos);
            }
        }
    }
    diag::testing_set_sink(nullptr, nullptr);
    diag::testing_set_level(old_level);
    diag::testing_close_log();
    std::filesystem::remove_all(cache);
    if (old_titles) setenv("ARMIGO_TITLES", old_titles->c_str(), 1); else unsetenv("ARMIGO_TITLES");
    if (old_rate) setenv("ARMIGO_COMPAT_FRAME_RATE", old_rate->c_str(), 1); else unsetenv("ARMIGO_COMPAT_FRAME_RATE");
    if (old_refresh) setenv("ARMIGO_VI_REFRESH_HZ", old_refresh->c_str(), 1); else unsetenv("ARMIGO_VI_REFRESH_HZ");
    if (old_resolution_rate) setenv("ARMIGO_COMPAT_RESOLUTION_RATE", old_resolution_rate->c_str(), 1);
    else unsetenv("ARMIGO_COMPAT_RESOLUTION_RATE");
}

void test_native_ipc_romfs_worker_and_exit() {
    Fixture fixture;
    auto presenter = std::make_shared<TestPresenter>(platform::PresentationMode::headless);
    auto created = ProcessRuntime::create(fixture.description, dependencies(presenter));
    CHECK(value(created) != nullptr);
    if (!value(created)) {
        const auto& error = *failure(created);
        std::fprintf(stderr, "Process creation failed: %s: %s\n",
                     error.operation.c_str(), error.detail.c_str());
        return;
    }
    auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
    const RunResult ran = runtime->run({}, {});
    CHECK(std::holds_alternative<GuestExit>(ran.outcome));
    CHECK(ran.quiescent);
    CHECK(!ran.teardown_failure.has_value());
    CHECK(ran.statistics.completed_frames == 0);
    CHECK(presenter->present_calls == 0);
    CHECK(ok(runtime->shutdown(std::nullopt)));
    CHECK(ok(runtime->shutdown(std::nullopt)));

    const auto repeated = runtime->run({}, {});
    CHECK(failure(repeated) != nullptr);
    if (failure(repeated)) CHECK(failure(repeated)->code == LaunchError::invalid_input);
}

void test_original_process_entry_and_worker_fp_inheritance(
    FixtureEntry entry, bool detached_metadata = false) {
    Fixture fixture(entry);
    if (detached_metadata) {
        const auto detached = fixture.temporary.path / "detached.npdm";
        std::filesystem::rename(fixture.exefs / "main.npdm", detached);
        auto executable_input = fixture.input;
        executable_input.process_metadata.reset();
        const auto artifacts = prepare_modules(executable_input, {});
        CHECK(value(artifacts) != nullptr);
        if (!value(artifacts)) return;
        fixture.description.prepared_directory = fixture.temporary.path / "detached-prepared";
        CHECK(ok(write_prepared_modules(fixture.description.prepared_directory, *value(artifacts))));
        fixture.description.npdm_file = detached;
    }
    auto created = ProcessRuntime::create(fixture.description, dependencies());
    if (const auto* error = failure(created))
        std::fprintf(stderr, "Original process construction failed: %s: %s\n",
                     error->operation.c_str(), error->detail.c_str());
    CHECK(value(created) != nullptr);
    if (!value(created)) return;
    auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
    const auto ran = runtime->run({.timeout = 2s}, {});
    if (const auto* error = failure(ran))
        std::fprintf(stderr, "Original process entry failed: %s: %s\n",
                     error->operation.c_str(), error->detail.c_str());
    CHECK(std::holds_alternative<GuestExit>(ran.outcome));
    CHECK(ran.quiescent);
    CHECK(ok(runtime->shutdown(std::nullopt)));
}

void test_creation_rejects_missing_conflicting_and_ordinary_inputs() {
    Fixture fixture;
    auto missing = fixture.description;
    missing.exefs_directory = fixture.temporary.path / "missing";
    auto result = ProcessRuntime::create(missing, dependencies());
    CHECK(failure(result) != nullptr && failure(result)->stage == LaunchStage::inputs);

    auto missing_prepared = fixture.description;
    missing_prepared.prepared_directory = fixture.temporary.path / "missing-prepared";
    result = ProcessRuntime::create(missing_prepared, dependencies());
    CHECK(failure(result) != nullptr && failure(result)->stage == LaunchStage::preparation);

    auto missing_romfs = fixture.description;
    missing_romfs.romfs_directory = fixture.temporary.path / "missing-romfs";
    result = ProcessRuntime::create(missing_romfs, dependencies());
    CHECK(failure(result) != nullptr && failure(result)->stage == LaunchStage::inputs);

    auto conflicting = fixture.description;
    const auto metadata = npdm_fixture::profile_image_a().bytes;
    const auto detached = fixture.temporary.path / "detached.npdm";
    CHECK(write_file(detached, metadata));
    conflicting.npdm_file = detached;
    result = ProcessRuntime::create(conflicting, dependencies());
    CHECK(failure(result) != nullptr && failure(result)->code == LaunchError::invalid_input);

    auto ordinary = fixture.description;
    ordinary.synthetic_profile.reset();
    ordinary.npdm_file = detached;
    ordinary.npdm_trust = NpdmTrustPolicy::structurally_validated_extracted_input;
    result = ProcessRuntime::create(ordinary, dependencies());
    CHECK(failure(result) != nullptr && failure(result)->code == LaunchError::artifact_mismatch);

    auto small_stack = fixture.description;
    small_stack.synthetic_profile->main_thread.stack_size = 4 * 1024;
    result = ProcessRuntime::create(small_stack, dependencies());
    CHECK(failure(result) != nullptr && failure(result)->code == LaunchError::invalid_input);

    auto invalid_limits = fixture.description;
    invalid_limits.input_limits.max_file_bytes = 0;
    result = ProcessRuntime::create(invalid_limits, dependencies());
    CHECK(failure(result) != nullptr);
}

void test_dependency_and_acl_validation() {
    Fixture fixture;
    auto absent_clock = dependencies();
    absent_clock.clock.reset();
    CHECK(failure(ProcessRuntime::create(fixture.description, std::move(absent_clock))) != nullptr);

    auto absent_grid = dependencies();
    absent_grid.presentation_clock.reset();
    CHECK(failure(ProcessRuntime::create(fixture.description, std::move(absent_grid))) != nullptr);

    auto absent_presenter = dependencies();
    absent_presenter.presenter.reset();
    CHECK(failure(ProcessRuntime::create(fixture.description,
                                         std::move(absent_presenter))) != nullptr);

    auto wrong_mode = dependencies(std::make_shared<TestPresenter>(
        platform::PresentationMode::windowed));
    CHECK(failure(ProcessRuntime::create(fixture.description, std::move(wrong_mode))) != nullptr);

    auto denied = fixture.description;
    denied.synthetic_profile->permissions.service_access_control =
        npdm_fixture::service_access_control({{"acc:u0", false}});
    auto denied_runtime = ProcessRuntime::create(denied, dependencies());
    CHECK(value(denied_runtime) != nullptr);
    if (value(denied_runtime)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(denied_runtime));
        const auto ran = runtime->run({}, {});
        CHECK(failure(ran) != nullptr);
        if (failure(ran)) {
            CHECK(failure(ran)->code == LaunchError::permission_denied ||
                  failure(ran)->code == LaunchError::unsupported);
        }
        CHECK(ran.quiescent);
    }

    // A partial file-system permission mask is admitted, not enforced per command.
    auto fs_partial = fixture.description;
    fs_partial.synthetic_profile->permissions.file_system_access_control =
        npdm_fixture::fs_access_block(1, 1);
    auto fs_result = ProcessRuntime::create(fs_partial, dependencies());
    CHECK(failure(fs_result) == nullptr);
}

void test_resource_failure_cleans_up_and_other_runtime_still_runs() {
    Fixture limited;
    limited.description.synthetic_profile = synthetic_profile(1);
    auto failed_runtime = ProcessRuntime::create(limited.description, dependencies());
    if (value(failed_runtime)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(failed_runtime));
        const auto ran = runtime->run({}, {});
        CHECK(failure(ran) != nullptr);
        CHECK(ran.quiescent);
        CHECK(ok(runtime->shutdown(std::nullopt)));
    } else {
        CHECK(failure(failed_runtime)->code == LaunchError::resource_limit ||
              failure(failed_runtime)->code == LaunchError::invalid_input);
    }

    Fixture healthy;
    auto created = ProcessRuntime::create(healthy.description, dependencies());
    CHECK(value(created) != nullptr);
    if (value(created)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
        const auto ran = runtime->run({}, {});
        CHECK(std::holds_alternative<GuestExit>(ran.outcome) && ran.quiescent);
    }
}

void test_failed_drain_retains_runtime_for_shutdown_retry() {
    Fixture fixture;
    auto presenter = std::make_shared<TestPresenter>(platform::PresentationMode::headless);
    presenter->fail_drains = 1;
    auto created = ProcessRuntime::create(fixture.description, dependencies(presenter));
    CHECK(value(created) != nullptr);
    if (!value(created)) return;
    auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
    const auto ran = runtime->run({}, {});
    CHECK(std::holds_alternative<GuestExit>(ran.outcome));
    CHECK(!ran.quiescent);
    CHECK(ran.teardown_failure.has_value());
    if (ran.teardown_failure) {
        CHECK(ran.teardown_failure->code == LaunchError::deadline_expired);
        CHECK(!ran.teardown_failure->operation.empty());
        CHECK(!ran.teardown_failure->detail.empty());
    }
    CHECK(presenter->drain_calls == 1);
    CHECK(ok(runtime->shutdown(std::nullopt)));
    CHECK(presenter->drain_calls == 2);
    CHECK(ok(runtime->shutdown(std::nullopt)));
}

void test_pre_stop_active_cancellation_and_timeout() {
    Fixture pre_stopped(true);
    auto created = ProcessRuntime::create(pre_stopped.description, dependencies());
    CHECK(value(created) != nullptr);
    if (value(created)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
        runtime->request_stop();
        const auto ran = runtime->run({}, {});
        CHECK(std::holds_alternative<RunCancelled>(ran.outcome));
        CHECK(ran.quiescent);
    }

    Fixture active(true);
    created = ProcessRuntime::create(active.description, dependencies());
    CHECK(value(created) != nullptr);
    if (value(created)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
        std::optional<RunResult> result;
        std::thread worker([&] { result = runtime->run({}, {}); });
        std::this_thread::sleep_for(20ms);
        runtime->request_stop(StopReason::window_closed);
        worker.join();
        CHECK(result.has_value());
        if (result) {
            const auto* stopped = std::get_if<RunCancelled>(&result->outcome);
            CHECK(stopped != nullptr && stopped->reason == StopReason::window_closed);
            CHECK(result->quiescent);
        }
    }

    Fixture timed(true);
    created = ProcessRuntime::create(timed.description, dependencies());
    CHECK(value(created) != nullptr);
    if (value(created)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
        RunLimits limits;
        limits.timeout = 10ms;
        const auto ran = runtime->run(limits, {});
        const auto* timeout = std::get_if<RunTimeout>(&ran.outcome);
        CHECK(timeout != nullptr && timeout->limit == 10ms);
        CHECK(ran.quiescent);
        CHECK(ok(runtime->shutdown(std::nullopt)));
    }
}

void test_shutdown_before_and_during_run_is_cancellation() {
    Fixture before(true);
    auto created = ProcessRuntime::create(before.description, dependencies());
    CHECK(value(created) != nullptr);
    if (value(created)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
        CHECK(ok(runtime->shutdown(std::nullopt)));
        const auto ran = runtime->run({}, {});
        CHECK(failure(ran) != nullptr);
        if (failure(ran)) CHECK(failure(ran)->code == LaunchError::invalid_input);
        CHECK(ran.quiescent);
    }

    Fixture active(true);
    created = ProcessRuntime::create(active.description, dependencies());
    CHECK(value(created) != nullptr);
    if (value(created)) {
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
        std::optional<RunResult> result;
        std::thread worker([&] { result = runtime->run({}, {}); });
        std::this_thread::sleep_for(20ms);
        const auto shutdown = runtime->shutdown(
            std::chrono::steady_clock::now() + 1s);
        worker.join();
        CHECK(ok(shutdown));
        CHECK(result.has_value());
        if (result) {
            CHECK(std::holds_alternative<RunCancelled>(result->outcome));
            CHECK(result->quiescent);
            CHECK(!result->teardown_failure.has_value());
        }
    }
}

void test_guest_join_precedes_native_shutdown_and_mapping_retirement() {
    struct Trace {
        std::mutex mutex;
        std::condition_variable changed;
        bool producer_entered = false, allow_return = false, producer_returned = false;
        bool drained = false, early_finish = false, allow_finish = false;
        unsigned finish_calls = 0, pending_native_work = 0;
        std::weak_ptr<GuestMemory> memory;
        std::weak_ptr<ServiceRegistry> services;
    };
    class Registry final : public ServiceRegistry {
    public:
        Registry(std::shared_ptr<ServiceRegistry> inner, std::shared_ptr<Trace> trace)
            : inner_(std::move(inner)), trace_(std::move(trace)) {}
        horizon::SessionConnection connect_named_port(std::string_view port) override {
            {
                std::unique_lock lock(trace_->mutex);
                // A synthetic host submission retains work beyond its guest
                // caller. The real process must join that caller first.
                ++trace_->pending_native_work;
                trace_->producer_entered = true;
                trace_->changed.notify_all();
                trace_->changed.wait(lock, [&] { return trace_->allow_return; });
            }
            auto result = inner_->connect_named_port(port);
            {
                const std::lock_guard lock(trace_->mutex);
                trace_->producer_returned = true;
                trace_->changed.notify_all();
            }
            return result;
        }
        std::optional<LaunchFailure> failure() const override { return inner_->failure(); }
        void request_stop() noexcept override { inner_->request_stop(); }
        LaunchStatus drain(std::optional<ClockInstant> deadline) override {
            auto status = inner_->drain(deadline);
            const std::lock_guard lock(trace_->mutex);
            trace_->drained = true;
            trace_->changed.notify_all();
            return status;
        }
        LaunchStatus finish_guest_shutdown(std::optional<ClockInstant> deadline) override {
            const std::lock_guard lock(trace_->mutex);
            ++trace_->finish_calls;
            trace_->early_finish |= !trace_->producer_returned;
            CHECK(trace_->producer_returned);
            CHECK(!trace_->memory.expired());
            if (!trace_->allow_finish) {
                LaunchFailure error;
                error.stage = LaunchStage::shutdown;
                error.code = LaunchError::deadline_expired;
                error.operation = "synthetic.native_shutdown";
                error.detail = "Synthetic native submission remains retained.";
                return error;
            }
            CHECK(trace_->pending_native_work != 0);
            auto status = inner_->finish_guest_shutdown(deadline);
            trace_->pending_native_work = 0;
            return status;
        }
        ServiceRegistryStatistics statistics() const noexcept override { return inner_->statistics(); }
    private:
        std::shared_ptr<ServiceRegistry> inner_;
        std::shared_ptr<Trace> trace_;
    };
    Fixture fixture;
    auto trace = std::make_shared<Trace>();
    auto deps = dependencies();
    deps.create_services = [trace](ServiceRegistryDependencies dependencies)
        -> LaunchResult<std::shared_ptr<ServiceRegistry>> {
        trace->memory = dependencies.memory;
        auto made = create_service_registry(std::move(dependencies));
        if (const auto* error = std::get_if<LaunchFailure>(&made)) return *error;
        auto registry = std::make_shared<Registry>(std::get<std::shared_ptr<ServiceRegistry>>(std::move(made)), trace);
        trace->services = registry;
        return std::shared_ptr<ServiceRegistry>(std::move(registry));
    };
    auto created = ProcessRuntime::create(fixture.description, std::move(deps));
    CHECK(value(created) != nullptr);
    if (!value(created)) return;
    auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
    auto running = std::async(std::launch::async, [&] { return runtime->run({}, {}); });
    {
        std::unique_lock lock(trace->mutex);
        CHECK(trace->changed.wait_for(lock, 2s, [&] { return trace->producer_entered; }));
    }
    runtime->request_stop();
    {
        std::unique_lock lock(trace->mutex);
        CHECK(trace->changed.wait_for(lock, 2s, [&] { return trace->drained; }));
        // The actual NativeProcessRuntime is now trying to join a guest that
        // remains inside its host call. A misplaced native phase must not run.
        CHECK(!trace->changed.wait_for(lock, 50ms, [&] { return trace->finish_calls != 0; }));
        CHECK(trace->finish_calls == 0 && !trace->memory.expired());
        trace->allow_return = true;
        trace->changed.notify_all();
    }
    const auto ran = running.get();
    CHECK(!ran.quiescent && ran.teardown_failure.has_value());
    CHECK(!trace->early_finish && trace->producer_returned && trace->finish_calls == 1);
    CHECK(!trace->memory.expired() && !trace->services.expired());
    CHECK(trace->pending_native_work != 0);
    if (ran.teardown_failure) CHECK(ran.teardown_failure->operation == "synthetic.native_shutdown");
    // Failure must retain mappings and services for an explicit later retry.
    {
        const std::lock_guard lock(trace->mutex);
        trace->allow_finish = true;
    }
    CHECK(ok(runtime->shutdown(std::chrono::steady_clock::now() + 2s)));
    CHECK(trace->finish_calls == 2 && !trace->early_finish);
    CHECK(trace->memory.expired() && trace->services.expired());
    CHECK(trace->pending_native_work == 0);
    CHECK(ok(runtime->shutdown(std::nullopt)));
    CHECK(trace->finish_calls == 2);
}

void test_successful_shutdown_releases_owned_dependencies() {
    Fixture fixture;
    auto deps = dependencies();
    std::weak_ptr<GuestClock> clock = deps.clock;
    std::weak_ptr<PresentationClock> presentation_clock = deps.presentation_clock;
    std::weak_ptr<platform::Presenter> presenter = deps.presenter;
    auto created = ProcessRuntime::create(fixture.description, std::move(deps));
    CHECK(value(created) != nullptr);
    if (!value(created)) return;
    auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
    const auto ran = runtime->run({}, {});
    CHECK(std::holds_alternative<GuestExit>(ran.outcome));
    CHECK(ran.quiescent);
    CHECK(presenter.expired());
    CHECK(presentation_clock.expired());
    CHECK(!clock.expired());
    runtime.reset();
    CHECK(clock.expired());
}

void test_stop_refusal_retains_observed_mutex_owner() {
    auto presenter = std::make_shared<TestPresenter>(platform::PresentationMode::headless);
    std::unique_ptr<Fixture> fixture;
    std::unique_ptr<ProcessRuntime> runtime;
    std::uint64_t heap_address = 0;
    LaunchFailure last_creation_failure;
    for (unsigned attempt = 0; attempt != 8 && !runtime; ++attempt) {
        const auto candidate_address = probe_heap_address();
        if (!candidate_address) continue;
        auto candidate = std::make_unique<Fixture>(
            FixtureEntry::grant_stop, *candidate_address);
        auto& profile = *candidate->description.synthetic_profile;
        profile.process.heap_address = *candidate_address;
        profile.permissions.supervisor_calls = {
            0x01, 0x07, 0x08, 0x09, 0x0a, 0x18, 0x1a, 0x1f, 0x21, 0x34, 0x35,
            0x40, 0x41, 0x42, 0x43};
        auto created = ProcessRuntime::create(candidate->description, dependencies(presenter));
        if (value(created)) {
            heap_address = *candidate_address;
            runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
            fixture = std::move(candidate);
            break;
        }
        last_creation_failure = *failure(created);
        if (last_creation_failure.operation != "create_guest_memory reservations") break;
    }
    CHECK(runtime != nullptr);
    if (!runtime) {
        std::fprintf(stderr, "Grant-stop process creation failed: %s: %s\n",
                     last_creation_failure.operation.c_str(),
                     last_creation_failure.detail.c_str());
        return;
    }

    struct Cleanup {
        ProcessRuntime* runtime;
        std::uint64_t heap;
        bool heap_writable = false;
        ~Cleanup() {
            if (!runtime) return;
            runtime->request_stop();
            if (heap_writable) {
                store_native_word(heap + 8, 1);
                store_native_word(heap, 2);
            }
            const auto status = runtime->shutdown(std::nullopt);
            if (!ok(status)) {
                std::fprintf(stderr, "Grant-stop cleanup could not quiesce the runtime.\n");
                ++failures;
            }
        }
    } cleanup{runtime.get(), heap_address};

    std::optional<RunResult> result;
    std::thread worker([&] { result = runtime->run({}, {}); });
    const bool ready = wait_for_native_word(heap_address + 4, 1, 2s);
    cleanup.heap_writable = native_writable(heap_address);
    CHECK(ready);
    CHECK(cleanup.heap_writable);
    runtime->request_stop();
    if (cleanup.heap_writable) store_native_word(heap_address + 8, 1);
    worker.join();

    CHECK(result.has_value());
    if (result) {
        const auto* stopped = std::get_if<LaunchFailure>(&result->outcome);
        if (!ready && stopped) {
            std::fprintf(stderr, "Grant-stop setup failed: %s: %s (command=%u)\n",
                         stopped->operation.c_str(), stopped->detail.c_str(),
                         stopped->command.value_or(0));
        }
        CHECK(stopped != nullptr);
        if (stopped) {
            CHECK(stopped->stage == LaunchStage::threads);
            CHECK(stopped->code == LaunchError::unsupported);
            CHECK(stopped->operation == "stop-with-mutex-grant");
        }
        // The retained boundary holds only for the shutdown grace period: the
        // owner never releases the mutex, and process termination ends the
        // thread whatever grant it owns.
        CHECK(result->quiescent);
        CHECK(!result->teardown_failure.has_value());
        // A quiescent runtime has released its guest memory.
        if (result->quiescent) cleanup.heap_writable = false;
    }

    CHECK(presenter->drain_calls > 0);
    if (cleanup.heap_writable) store_native_word(heap_address, 2);
    CHECK(ok(runtime->shutdown(std::chrono::steady_clock::now() + 2s)));
    cleanup.runtime = nullptr;
}


class PausedClock final : public GuestClock {
public:
    explicit PausedClock(std::shared_ptr<GuestClock> inner) : inner_(std::move(inner)) {}
    void arm() {
        std::lock_guard lock(mutex_);
        target_ = std::this_thread::get_id();
        armed_ = true;
    }
    bool wait_entered() {
        std::unique_lock lock(mutex_);
        return changed_.wait_for(lock, 2s, [&] { return entered_; });
    }
    void release() {
        std::lock_guard lock(mutex_);
        armed_ = false;
        changed_.notify_all();
    }
    LaunchResult<ClockSample> sample() const override {
        {
            std::unique_lock lock(mutex_);
            if (armed_ && target_ == std::this_thread::get_id()) {
                entered_ = true;
                changed_.notify_all();
                changed_.wait(lock, [&] { return !armed_; });
            }
        }
        return inner_->sample();
    }
    execution::GuestCounter counter() noexcept override { return inner_->counter(); }
    LaunchResult<std::chrono::nanoseconds> ticks_to_nanoseconds(std::uint64_t ticks) const override {
        return inner_->ticks_to_nanoseconds(ticks);
    }
    LaunchResult<std::uint64_t> nanoseconds_to_ticks(std::chrono::nanoseconds time) const override {
        return inner_->nanoseconds_to_ticks(time);
    }
    LaunchResult<ClockWait> wait_until(ClockInstant deadline, std::stop_token stop) override {
        return inner_->wait_until(deadline, stop);
    }
private:
    std::shared_ptr<GuestClock> inner_;
    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    std::thread::id target_;
    bool armed_ = false;
    mutable bool entered_ = false;
};

void test_concurrent_drain_failure_preserves_cancellation() {
    Fixture fixture(true);
    auto presenter = std::make_shared<TestPresenter>(platform::PresentationMode::headless);
    presenter->fail_drains = 1;
    auto deps = dependencies(presenter);
    auto paused = std::make_shared<PausedClock>(deps.clock);
    deps.clock = paused;
    auto grid = create_presentation_clock(paused, deps.presentation_clock->configuration());
    CHECK(value(grid) != nullptr);
    if (!value(grid)) return;
    deps.presentation_clock = *value(grid);
    auto created = ProcessRuntime::create(fixture.description, std::move(deps));
    CHECK(value(created) != nullptr);
    if (!value(created)) return;
    auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
    RunResult result;
    std::thread worker([&] {
        paused->arm();
        result = runtime->run({}, {});
    });
    CHECK(paused->wait_entered());
    const auto teardown = runtime->shutdown(std::chrono::steady_clock::now() + 1s);
    CHECK(failure(teardown) != nullptr);
    paused->release();
    worker.join();
    CHECK(std::holds_alternative<RunCancelled>(result.outcome));
    CHECK(result.teardown_failure.has_value());
    if (result.teardown_failure) CHECK(result.teardown_failure->code == LaunchError::deadline_expired);
    CHECK(result.quiescent);
    CHECK(presenter->drain_calls == 2);
}

int construction_drain_failure_child() {
    Fixture fixture;
    // Services exist before stack allocation exhausts the admitted reservation.
    fixture.description.synthetic_profile->main_thread.stack_size = 4ull << 20;
    auto presenter = std::make_shared<TestPresenter>(platform::PresentationMode::headless);
    presenter->fail_drains = 1;
    auto created = ProcessRuntime::create(fixture.description, dependencies(presenter));
    const auto* original = failure(created);
    CHECK(original != nullptr);
    if (original) {
        CHECK(original->stage == LaunchStage::memory);
        CHECK(original->code == LaunchError::resource_limit);
        CHECK(original->operation == "allocate");
        CHECK(original->detail == "Stack reservation is exhausted.");
        CHECK(original->cleanup_owner != nullptr);
        if (original->cleanup_owner) {
            CHECK(presenter->drain_calls == 1);
            std::weak_ptr<ProcessRuntime> retained = original->cleanup_owner;
            // Even discarding the error cannot destroy undrained consumers.
            created = LaunchFailure{};
            auto retry = retained.lock();
            CHECK(retry != nullptr);
            if (retry) CHECK(ok(retry->shutdown(std::nullopt)));
            CHECK(presenter->drain_calls == 2);
            retry.reset();
            CHECK(retained.expired());
        }
    }
    return failures ? 1 : 0;
}

void test_construction_drain_failure_returns_without_abort(const char* executable) {
    pid_t child = 0;
    char argument[] = "--construction-drain-failure";
    char* arguments[] = {const_cast<char*>(executable), argument, nullptr};
    const auto spawned = posix_spawn(&child, executable, nullptr, nullptr, arguments, environ);
    CHECK(spawned == 0);
    if (spawned != 0) return;
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

void test_invalid_run_limits_fail_without_execution() {
    for (unsigned which = 0; which != 2; ++which) {
        Fixture fixture;
        auto created = ProcessRuntime::create(fixture.description, dependencies());
        CHECK(value(created) != nullptr);
        if (!value(created)) continue;
        auto runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(created));
        RunLimits limits;
        if (which == 0) limits.completed_frame_target = 0;
        else limits.timeout = 0ns;
        const auto ran = runtime->run(limits, {});
        CHECK(failure(ran) != nullptr && failure(ran)->code == LaunchError::invalid_input);
        CHECK(ran.quiescent);
        CHECK(ok(runtime->shutdown(std::nullopt)));
    }
}

void test_capture_policy_selects_readback_frames() {
    // The interval is read once, on the first frame a capture directory could
    // keep; no earlier test in this process names a capture directory.
    CHECK(::setenv("ARMIGO_CAPTURE_EVERY", "3", 1) == 0);
    ReportDestination output;
    for (std::uint64_t frame = 0; frame < 7; ++frame) CHECK(!captures_frame(output, frame));
    output.capture_directory = "captures";
    for (std::uint64_t frame = 0; frame < 7; ++frame)
        CHECK(captures_frame(output, frame) == (frame % 3 == 0));
    // The first read is final, so the frame writer and presentation readback
    // cannot disagree after a later change.
    CHECK(::unsetenv("ARMIGO_CAPTURE_EVERY") == 0);
    CHECK(captures_frame(output, 6) && !captures_frame(output, 7));
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--construction-drain-failure")
        return construction_drain_failure_child();
    test_explicit_and_platform_memory_budgets();
    test_title_rate_clock_warning_for_both_runners();
    test_capture_policy_selects_readback_frames();
    test_construction_drain_failure_returns_without_abort(argv[0]);
    test_concurrent_drain_failure_preserves_cancellation();
    test_original_process_entry_and_worker_fp_inheritance(FixtureEntry::original);
    test_original_process_entry_and_worker_fp_inheritance(FixtureEntry::original_rtld);
    test_original_process_entry_and_worker_fp_inheritance(FixtureEntry::original, true);
    test_original_process_entry_and_worker_fp_inheritance(FixtureEntry::original_rtld, true);
    test_native_ipc_romfs_worker_and_exit();
    test_creation_rejects_missing_conflicting_and_ordinary_inputs();
    test_dependency_and_acl_validation();
    test_resource_failure_cleans_up_and_other_runtime_still_runs();
    test_failed_drain_retains_runtime_for_shutdown_retry();
    test_pre_stop_active_cancellation_and_timeout();
    test_shutdown_before_and_during_run_is_cancellation();
    test_guest_join_precedes_native_shutdown_and_mapping_retirement();
    test_successful_shutdown_releases_owned_dependencies();
    test_stop_refusal_retains_observed_mutex_owner();
    test_invalid_run_limits_fail_without_execution();
    std::printf("Process-runtime failures: %u\n", failures);
    return failures ? 1 : 0;
}
