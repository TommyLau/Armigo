// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/launch_result.h>
#include <armigo/runtime/launch_profile.h>
#include <armigo/runtime/launch_description.h>
#include <armigo/runtime/prepared_modules.h>
#include <armigo/runtime/launch_plan.h>
#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/guest_clock.h>
#include <armigo/runtime/presentation_clock.h>
#include <armigo/runtime/thread_runtime.h>
#include <armigo/runtime/presentation.h>
#include <armigo/runtime/process_runtime.h>
#include <armigo/platform/presenter.h>
#include <armigo/platform/application.h>

#include <cstdio>
#include <type_traits>
#include <utility>
#include <variant>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
} // namespace

int main() {
    using namespace armigo;
    using namespace armigo::runtime;
    static_assert(std::is_same_v<decltype(LaunchProfile{}.process),
                                 horizon::ProcessConfiguration>);
    static_assert(std::is_same_v<decltype(ThreadLaunch{}.entry),
                                 execution::GuestThreadEntryRequest>);
    static_assert(std::is_same_v<decltype(platform::PresentationFrame{}.queued),
                                 services::GraphicsPresentation>);
    static_assert(std::has_virtual_destructor_v<GuestMemory>);
    static_assert(std::has_virtual_destructor_v<PinnedGuestMemory>);
    static_assert(std::has_virtual_destructor_v<ThreadRuntime>);
    static_assert(std::has_virtual_destructor_v<GuestClock>);
    static_assert(std::has_virtual_destructor_v<PresentationClock>);
    static_assert(std::has_virtual_destructor_v<PresentationBinding>);
    static_assert(std::has_virtual_destructor_v<platform::Presenter>);
    static_assert(std::has_virtual_destructor_v<platform::Application>);
    static_assert(std::has_virtual_destructor_v<ProcessRuntime>);

    LaunchDescription launch;
    check(launch.presentation.mode == platform::PresentationMode::headless,
          "headless is the explicit default presentation mode");
    check(launch.npdm_trust == NpdmTrustPolicy::require_authenticated,
          "metadata trust does not silently opt out of authentication");
    check(!launch.synthetic_profile && !launch.npdm_file,
          "ordinary input does not imply synthetic configuration");
    launch.presentation.mode = platform::PresentationMode::windowed;
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.process.allowed_core_mask = 2;
    profile.process.private_system_resource_bytes = horizon::guest_page_size;
    profile.main_thread = {24, 1, 64 * horizon::guest_page_size};
    profile.permissions.highest_priority = 16;
    profile.permissions.lowest_priority = 40;
    profile.permissions.supervisor_calls = {1, 2};
    launch.synthetic_profile = profile;
    auto moved = std::move(launch);
    check(moved.synthetic_profile->main_thread.ideal_core == 1 &&
          moved.synthetic_profile->process.allowed_core_mask == 2,
          "main-thread settings and process capabilities survive value transport");

    LaunchResult<GuestMemoryPin> pin;
    LaunchStatus status;
    check(std::holds_alternative<LaunchFailure>(pin) &&
          std::holds_alternative<LaunchFailure>(status),
          "unfilled operation results are failures");
    status = std::monostate{};
    check(std::holds_alternative<std::monostate>(status), "success is explicit");
    PreparedModuleArtifacts artifacts;
    artifacts.modules.push_back({"synthetic", {}, {std::byte{1}}});
    check(artifacts.format_version == 0 && artifacts.modules.front().payload.size() == 1,
          "artifact envelope owns bytes without claiming an admitted schema");
    PreparedLaunch prepared;
    check(!prepared.modules && !prepared.sites &&
          prepared.plan.relocation_owner == LaunchOwnership::unresolved,
          "default prepared launch grants no entry capability");
    ClockSample sample{ClockInstant{}, 19'200'000, std::chrono::seconds(1)};
    PresentationSchedule schedule{sample.instant, 7, true};
    check(schedule.immediate && schedule.refresh_index == 7,
          "presentation schedules distinguish immediate work from refresh waits");
    GraphicsBackingAccess backing;
    check(!backing.lookup && !backing.pin && !backing.wait_fence,
          "missing graphics capabilities stay absent");
    platform::PresentResult presented;
    check(std::holds_alternative<platform::PresentationFailure>(presented),
          "unfilled presentation does not release a queued buffer");
    RunResult result;
    check(std::holds_alternative<LaunchFailure>(result.outcome) && !result.quiescent,
          "default run neither succeeds nor claims safe teardown");
    result.outcome = RunSuccess{3};
    result.statistics.completed_frames = 3;
    result.quiescent = true;
    check(std::get<RunSuccess>(result.outcome).frame_target == 3,
          "frame-target completion carries its target");
    result.outcome = GuestExit{};
    check(!std::get<GuestExit>(result.outcome).status,
          "guest exit does not invent a POSIX status");
    result.outcome = RunTimeout{std::chrono::seconds(2)};
    check(std::get<RunTimeout>(result.outcome).limit == std::chrono::seconds(2),
          "timeout is separate from successful completion");
    result.outcome = LaunchFailure{.stage = LaunchStage::services,
        .code = LaunchError::unsupported, .operation = "synthetic-service",
        .detail = "Synthetic unsupported command", .thread_id = {}, .guest_pc = {},
        .command = 7, .guest_result = {}};
    check(std::get<LaunchFailure>(result.outcome).command == 7 &&
          !std::get<LaunchFailure>(result.outcome).guest_result,
          "unsupported stop retains structure without inventing a guest result");
    return failures == 0 ? 0 : 1;
}
