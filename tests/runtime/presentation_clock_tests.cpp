// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/presentation_clock.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace {
using namespace armigo;
using namespace armigo::runtime;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

// Deterministic injected timeline: the test moves now by hand. Planning must
// never wait, so every schedule run asserts the wait source was not called.
struct ManualClock {
    std::atomic<std::int64_t> now_ns{0};
    std::atomic<int> wait_calls{0};

    ClockInstant instant() const {
        return ClockInstant{std::chrono::nanoseconds{
            now_ns.load(std::memory_order_relaxed)}};
    }
};

GuestClockSources manual_sources(const std::shared_ptr<ManualClock>& manual) {
    GuestClockSources sources;
    sources.now = [manual] { return manual->instant(); };
    sources.wait = [manual](ClockInstant, std::stop_token)
        -> LaunchResult<ClockWait> {
        manual->wait_calls.fetch_add(1, std::memory_order_relaxed);
        return ClockWait::reached;
    };
    return sources;
}

std::shared_ptr<GuestClock> manual_clock(ManualClock& manual) {
    auto created = create_guest_clock(manual_sources(
        std::shared_ptr<ManualClock>(&manual, [](ManualClock*) {})));
    // Non-owning view: the tests keep the ManualClock alive on the stack.
    return std::get<std::shared_ptr<GuestClock>>(std::move(created));
}

std::shared_ptr<PresentationClock> one_millisecond_grid(ManualClock& manual) {
    PresentationClockConfiguration configuration;
    configuration.epoch = ClockInstant{std::chrono::nanoseconds{1'000'000}};
    configuration.refresh_period = std::chrono::nanoseconds{1'000'000};
    auto created = create_presentation_clock(manual_clock(manual),
                                             configuration);
    return std::get<std::shared_ptr<PresentationClock>>(std::move(created));
}

ClockInstant at(std::int64_t nanoseconds) {
    return ClockInstant{std::chrono::nanoseconds{nanoseconds}};
}

template <typename T>
bool is_failure(const LaunchResult<T>& result, LaunchError code,
                const char* operation) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure && failure->code == code && failure->operation == operation;
}

void expect_planned(const LaunchResult<PresentationSchedule>& result,
                    std::uint64_t index, std::int64_t deadline_ns,
                    const char* context) {
    if (auto* failure = std::get_if<LaunchFailure>(&result)) {
        std::fprintf(stderr, "Failed (%s): unexpected %s failure: %s\n",
                     context, failure->operation.c_str(),
                     failure->detail.c_str());
        ++failures;
        return;
    }
    const auto& planned = std::get<PresentationSchedule>(result);
    CHECK(planned.refresh_index == index);
    CHECK(planned.deadline == at(deadline_ns));
    CHECK(!planned.immediate);
}

void test_creation_rejects_invalid_configuration() {
    ManualClock manual;
    auto clock = manual_clock(manual);
    PresentationClockConfiguration valid;
    valid.epoch = at(1'000'000);
    valid.refresh_period = std::chrono::nanoseconds{1'000'000};

    auto without_clock = create_presentation_clock(nullptr, valid);
    CHECK(is_failure(without_clock, LaunchError::invalid_input,
                     "create_presentation_clock"));

    PresentationClockConfiguration zero = valid;
    zero.refresh_period = std::chrono::nanoseconds::zero();
    CHECK(is_failure(create_presentation_clock(clock, zero),
                     LaunchError::invalid_input, "create_presentation_clock"));

    PresentationClockConfiguration negative = valid;
    negative.refresh_period = std::chrono::nanoseconds{-1};
    CHECK(is_failure(create_presentation_clock(clock, negative),
                     LaunchError::invalid_input, "create_presentation_clock"));

    auto created = create_presentation_clock(clock, valid);
    auto grid = std::get<std::shared_ptr<PresentationClock>>(
        std::move(created));
    CHECK(grid->clock() == clock);
    CHECK(grid->configuration().refresh_period == valid.refresh_period);
    CHECK(grid->configuration().epoch == valid.epoch);
}

void test_zero_interval_is_immediate() {
    ManualClock manual;
    auto grid = one_millisecond_grid(manual);

    manual.now_ns.store(500'000);
    auto before_epoch = std::get<PresentationSchedule>(
        grid->schedule(std::nullopt, 0));
    CHECK(before_epoch.immediate && before_epoch.refresh_index == 0 &&
          before_epoch.deadline == at(500'000));

    manual.now_ns.store(3'500'000);
    auto mid_grid = std::get<PresentationSchedule>(
        grid->schedule(std::nullopt, 0));
    CHECK(mid_grid.immediate && mid_grid.refresh_index == 2 &&
          mid_grid.deadline == at(3'500'000));
}

void test_positive_intervals_stay_on_one_grid() {
    ManualClock manual;
    auto grid = one_millisecond_grid(manual);

    manual.now_ns.store(500'000);
    expect_planned(grid->schedule(std::nullopt, 1), 1, 2'000'000,
                   "first interval before the epoch");

    manual.now_ns.store(3'500'000);
    expect_planned(grid->schedule(std::nullopt, 1), 3, 4'000'000,
                   "first interval at 2.5 periods");
    expect_planned(grid->schedule(0, 1), 3, 4'000'000,
                   "interval one after stale last refresh zero");
    expect_planned(grid->schedule(2, 2), 4, 5'000'000,
                   "interval two after last refresh two");
    expect_planned(grid->schedule(5, 1), 6, 7'000'000,
                   "interval floor beyond now");

    manual.now_ns.store(4'000'000);
    expect_planned(grid->schedule(3, 1), 4, 5'000'000,
                   "interval one from a just-completed grid point");
    expect_planned(grid->schedule(std::nullopt, 1), 4, 5'000'000,
                   "interval one defaulted at an exact grid point");
    CHECK(manual.wait_calls.load() == 0);
}

void test_next_refresh_is_strictly_future() {
    ManualClock manual;
    auto grid = one_millisecond_grid(manual);

    manual.now_ns.store(500'000);
    expect_planned(grid->next_refresh(), 0, 1'000'000,
                   "epoch point before the epoch");

    manual.now_ns.store(3'000'000);
    // Point two sits at 3'000'000, so the strictly future next refresh is
    // point three, never the point under now.
    expect_planned(grid->next_refresh(), 3, 4'000'000,
                   "the next point, not the one under now");

    manual.now_ns.store(3'000'001);
    expect_planned(grid->next_refresh(), 3, 4'000'000,
                   "just after a grid point");
    CHECK(manual.wait_calls.load() == 0);
}

void test_late_frames_skip_missed_points() {
    ManualClock manual;
    auto grid = one_millisecond_grid(manual);

    manual.now_ns.store(8'500'000);
    expect_planned(grid->schedule(1, 1), 8, 9'000'000,
                   "several missed refreshes are skipped, not accumulated");

    manual.now_ns.store(8'000'000);
    expect_planned(grid->schedule(1, 2), 7, 8'000'000,
                   "interval two still targets the first surviving point");

    manual.now_ns.store(6'000'000);
    expect_planned(grid->schedule(5, 2), 7, 8'000'000,
                   "at least two refreshes after the last completed target");
}

void test_invalid_and_overflowing_requests_fail() {
    ManualClock manual;
    auto grid = one_millisecond_grid(manual);

    manual.now_ns.store(3'500'000);
    CHECK(is_failure(grid->schedule(std::nullopt, -1),
                     LaunchError::invalid_input, "presentation_clock.schedule"));
    CHECK(is_failure(grid->schedule(0, -2), LaunchError::invalid_input,
                     "presentation_clock.schedule"));

    auto limit = std::numeric_limits<std::uint64_t>::max();
    CHECK(is_failure(grid->schedule(limit, 1), LaunchError::resource_limit,
                     "presentation_clock.schedule"));

    // A grid whose next point leaves the timeline fails explicitly.
    ManualClock extreme;
    PresentationClockConfiguration late_epoch;
    late_epoch.epoch =
        ClockInstant{std::chrono::nanoseconds{
            std::numeric_limits<std::int64_t>::max()}};
    late_epoch.refresh_period = std::chrono::nanoseconds{2};
    auto sparse = std::get<std::shared_ptr<PresentationClock>>(
        create_presentation_clock(manual_clock(extreme), late_epoch));
    extreme.now_ns.store(std::numeric_limits<std::int64_t>::max());
    CHECK(is_failure(sparse->schedule(std::nullopt, 1),
                     LaunchError::resource_limit, "presentation_clock.grid"));
    CHECK(is_failure(sparse->next_refresh(), LaunchError::resource_limit,
                     "presentation_clock.grid"));
}

void test_planning_preserves_the_first_causal_failure() {
    ManualClock manual;
    manual.now_ns.store(100);
    auto clock = manual_clock(manual);
    (void)clock->sample();
    manual.now_ns.store(50);
    auto grid = std::get<std::shared_ptr<PresentationClock>>(
        create_presentation_clock(
            clock,
            PresentationClockConfiguration{at(0),
                                           std::chrono::nanoseconds{1'000'000}}));
    auto planned = grid->schedule(std::nullopt, 1);
    const auto* failure = std::get_if<LaunchFailure>(&planned);
    CHECK(failure && failure->code == LaunchError::host_failure &&
          failure->stage == LaunchStage::services &&
          failure->operation == "guest_clock.sample");
    CHECK(is_failure(grid->next_refresh(), LaunchError::host_failure,
                     "guest_clock.sample"));
}


constexpr std::int64_t sixty_hertz = 16'666'667;
constexpr std::int64_t lead = 500'000;  // grid points this long after a vblank

std::shared_ptr<PresentationClock> sixty_hertz_grid(ManualClock& manual, std::int64_t epoch = 1'000'000) {
    PresentationClockConfiguration configuration;
    configuration.epoch = at(epoch);
    configuration.refresh_period = std::chrono::nanoseconds{sixty_hertz};
    auto created = create_presentation_clock(manual_clock(manual), configuration);
    return std::get<std::shared_ptr<PresentationClock>>(std::move(created));
}

std::int64_t next_deadline(PresentationClock& clock) {
    auto next = clock.next_refresh();
    CHECK(std::holds_alternative<PresentationSchedule>(next));
    return std::holds_alternative<PresentationSchedule>(next)
        ? std::get<PresentationSchedule>(next).deadline.time_since_epoch().count() : 0;
}

// Feeds `count` refreshes of a display, advancing the clock with them; the
// reported period may differ from the true spacing by `jitter` (alternating).
void feed(PresentationClock& clock, ManualClock& manual, std::int64_t first_vblank,
          std::int64_t display, int count, std::int64_t jitter = 0) {
    for (int i = 0; i < count; ++i) {
        const std::int64_t vblank = first_vblank + i * display;
        manual.now_ns.store(vblank, std::memory_order_relaxed);
        clock.observe_display_refresh(at(vblank), std::chrono::nanoseconds{display + (i % 2 ? jitter : -jitter)});
    }
}

// Distance from a grid point to "lead after the nearest vblank".
std::int64_t phase_error(std::int64_t deadline, std::int64_t first_vblank, std::int64_t display) {
    std::int64_t error = (deadline - lead - first_vblank) % display;
    if (error >= display / 2) error -= display;
    if (error < -display / 2) error += display;
    return error < 0 ? -error : error;
}

void test_grid_follows_displays_that_divide_it() {
    for (const std::int64_t display : {std::int64_t{16'666'667}, std::int64_t{8'333'333}, std::int64_t{4'166'667}}) {
        ManualClock manual;
        auto clock = sixty_hertz_grid(manual);
        const std::int64_t first_vblank = 7'777'777;  // out of phase for every rate here
        CHECK(phase_error(next_deadline(*clock), first_vblank, display) > 100'000);
        feed(*clock, manual, first_vblank, display, 400);
        const auto deadline = next_deadline(*clock);
        // Settled: just after a vblank, within the dead band.
        CHECK(phase_error(deadline, first_vblank, display) <= 60'000);
        // The grid keeps its period.
        auto following = clock->schedule(std::get<PresentationSchedule>(clock->next_refresh()).refresh_index, 1);
        CHECK(std::holds_alternative<PresentationSchedule>(following));
        if (std::holds_alternative<PresentationSchedule>(following))
            CHECK(std::get<PresentationSchedule>(following).deadline.time_since_epoch().count() - deadline ==
                  sixty_hertz);
    }
}

// A display period that only nearly divides the grid (59.94 Hz, 119.88 Hz),
// an hour into the session: the error is measured at the grid point near the
// vblank, so it does not grow with the distance from the epoch.
void test_grid_follows_near_divisors_late_in_a_session() {
    for (const std::int64_t display : {std::int64_t{16'683'333}, std::int64_t{8'341'667}}) {
        ManualClock manual;
        auto clock = sixty_hertz_grid(manual);
        const std::int64_t first_vblank = 3'600'000'000'000 + 3'333'333;
        const int count = int(20'000'000'000 / display);  // 20 s of refreshes
        feed(*clock, manual, first_vblank, display, count);
        const std::int64_t last = first_vblank + (count - 1) * display;
        CHECK(phase_error(next_deadline(*clock), last, display) <= 150'000);
    }
}

// A reported period that varies by a nanosecond, an hour into the session,
// does not move a settled grid at any refresh (not even back and forth).
void test_period_jitter_does_not_move_a_settled_grid() {
    ManualClock manual;
    auto clock = sixty_hertz_grid(manual);
    const std::int64_t display = 8'333'333;
    const std::int64_t first_vblank = 3'600'000'000'000 + 1'234'567;
    feed(*clock, manual, first_vblank, display, 600, 1);
    const auto settled = next_deadline(*clock);
    int moved = 0;
    for (int i = 600; i < 1800; ++i) {
        const std::int64_t vblank = first_vblank + i * display;
        manual.now_ns.store(vblank, std::memory_order_relaxed);
        clock->observe_display_refresh(at(vblank), std::chrono::nanoseconds{display + (i % 2 ? 1 : -1)});
        if ((next_deadline(*clock) - settled) % sixty_hertz != 0) ++moved;
    }
    CHECK(moved == 0);
}

void test_phase_moves_in_small_steps() {
    ManualClock manual;
    auto clock = sixty_hertz_grid(manual);
    manual.now_ns.store(2'000'000, std::memory_order_relaxed);
    const auto before = next_deadline(*clock);
    // A 120 Hz vblank whose grid point should come 1 ms later: the grid moves
    // later by one step, a 0.2 ms step divided by the two refreshes per point.
    clock->observe_display_refresh(at(before + 1'000'000 - lead), std::chrono::nanoseconds{8'333'333});
    const auto after = next_deadline(*clock);
    CHECK(after - before == 100'000);
    // And earlier when the grid point should come sooner.
    ManualClock early;
    auto sooner = sixty_hertz_grid(early);
    early.now_ns.store(2'000'000, std::memory_order_relaxed);
    const auto grid = next_deadline(*sooner);
    sooner->observe_display_refresh(at(grid - 1'000'000 - lead), std::chrono::nanoseconds{16'666'667});
    CHECK(grid - next_deadline(*sooner) == 200'000);
    // A vblank already in phase leaves the grid alone.
    ManualClock settled;
    auto aligned = sixty_hertz_grid(settled);
    const auto point = next_deadline(*aligned);
    aligned->observe_display_refresh(at(point - lead), std::chrono::nanoseconds{8'333'333});
    CHECK(next_deadline(*aligned) == point);
}

void test_other_display_rates_leave_the_grid() {
    for (const std::int64_t display : {std::int64_t{6'944'444}, std::int64_t{11'111'111}, std::int64_t{10'000'000},
                                       std::int64_t{0}, std::int64_t{-8'333'333}}) {
        ManualClock manual;
        auto clock = sixty_hertz_grid(manual);
        const auto before = next_deadline(*clock);
        for (int i = 0; i < 100; ++i)
            clock->observe_display_refresh(at(3'000'000 + i * 7'000'001), std::chrono::nanoseconds{display});
        CHECK(next_deadline(*clock) == before);
        CHECK(!clock->display_refresh_near(at(5'000'000)));
    }
}

// The nearest grid point is offered only once a display carries the grid.
void test_display_refresh_near() {
    ManualClock manual;
    auto clock = sixty_hertz_grid(manual);
    CHECK(!clock->display_refresh_near(at(50'000'000)));
    feed(*clock, manual, 7'777'777, 8'333'333, 400);
    const auto point = next_deadline(*clock);
    for (const std::int64_t offset : {std::int64_t{0}, std::int64_t{5'000'000}, std::int64_t{-5'000'000},
                                      std::int64_t{8'000'000}, std::int64_t{-8'000'000}}) {
        const auto near = clock->display_refresh_near(at(point + offset));
        CHECK(near && near->time_since_epoch().count() == point);
    }
    const auto next = clock->display_refresh_near(at(point + 9'000'000));
    CHECK(next && next->time_since_epoch().count() == point + sixty_hertz);
    const auto previous = clock->display_refresh_near(at(point - 9'000'000));
    CHECK(previous && previous->time_since_epoch().count() == point - sixty_hertz);
}

void test_refresh_period_change_preserves_indices() {
    ManualClock manual;
    auto grid = one_millisecond_grid(manual);
    manual.now_ns.store(4'500'000);
    CHECK(grid->set_refresh_period(std::chrono::nanoseconds{2'000'000}));
    CHECK(grid->configuration().refresh_period == std::chrono::nanoseconds{2'000'000});
    expect_planned(grid->schedule(3, 1), 4, 6'500'000, "retained index after a rate change");
    expect_planned(grid->schedule(std::nullopt, 1), 4, 6'500'000, "new stream after a rate change");
    expect_planned(grid->next_refresh(), 4, 6'500'000, "next refresh after a rate change");
    CHECK(!grid->set_refresh_period(std::chrono::nanoseconds::zero()));
    CHECK(!grid->set_refresh_period(std::chrono::nanoseconds{-1}));
    CHECK(grid->configuration().refresh_period == std::chrono::nanoseconds{2'000'000});
}

} // namespace

int main() {
    test_creation_rejects_invalid_configuration();
    test_zero_interval_is_immediate();
    test_positive_intervals_stay_on_one_grid();
    test_next_refresh_is_strictly_future();
    test_late_frames_skip_missed_points();
    test_invalid_and_overflowing_requests_fail();
    test_planning_preserves_the_first_causal_failure();
    test_grid_follows_displays_that_divide_it();
    test_grid_follows_near_divisors_late_in_a_session();
    test_period_jitter_does_not_move_a_settled_grid();
    test_phase_moves_in_small_steps();
    test_other_display_rates_leave_the_grid();
    test_display_refresh_near();
    test_refresh_period_change_preserves_indices();
    return failures == 0 ? 0 : 1;
}
