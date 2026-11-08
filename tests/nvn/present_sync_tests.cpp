// ============================================================================
// Armigo Tests - NVN Queue and Synchronization Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/guest_timeline.h>

#include "backend.h"
#include "entry_table.h"
#include "present_internal.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <future>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace {

using namespace armigo::nvn;

unsigned failures = 0;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);             \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

class FakeReplayBackend final : public ReplayBackend {
public:
  std::uint64_t submit(std::span<const NVNcommandHandle> handles) override {
    submitted.assign(handles.begin(), handles.end());
    ++submit_calls;
    return next_fence++;
  }

  void flush() override { ++flush_calls; }
  std::uint64_t completed_fence() const override { return completed; }

  bool wait_fence(std::uint64_t fence, std::uint64_t timeout_ns) override {
    ++wait_calls;
    waited_fence = fence;
    waited_timeout = timeout_ns;
    if (wait_succeeds)
      completed = fence;
    return wait_succeeds;
  }

  std::vector<NVNcommandHandle> submitted;
  std::uint64_t next_fence = 11;
  std::uint64_t completed = 0;
  std::uint64_t waited_fence = 0;
  std::uint64_t waited_timeout = 0;
  unsigned submit_calls = 0;
  unsigned flush_calls = 0;
  unsigned wait_calls = 0;
  bool wait_succeeds = true;
};

bool wait_hook_called = false;
bool test_wait_hook(const std::function<bool()> &wait) {
  wait_hook_called = true;
  return wait();
}

template <Entry E> typename EntrySignature<E>::type load_entry() {
  return reinterpret_cast<typename EntrySignature<E>::type>(
      armigo_nvn_bootstrap_loader(entry_name(E).data()));
}

void verify_queue_and_sync(FakeReplayBackend &backend) {
  const auto builder_set_device = load_entry<Entry::nvnQueueBuilderSetDevice>();
  const auto builder_set_defaults =
      load_entry<Entry::nvnQueueBuilderSetDefaults>();
  const auto builder_set_flags = load_entry<Entry::nvnQueueBuilderSetFlags>();
  const auto builder_set_command_size =
      load_entry<Entry::nvnQueueBuilderSetCommandMemorySize>();
  const auto builder_set_compute_size =
      load_entry<Entry::nvnQueueBuilderSetComputeMemorySize>();
  const auto builder_set_control_size =
      load_entry<Entry::nvnQueueBuilderSetControlMemorySize>();
  const auto builder_memory_size =
      load_entry<Entry::nvnQueueBuilderGetQueueMemorySize>();
  const auto builder_set_memory =
      load_entry<Entry::nvnQueueBuilderSetQueueMemory>();
  const auto builder_set_threshold =
      load_entry<Entry::nvnQueueBuilderSetCommandFlushThreshold>();
  const auto initialize = load_entry<Entry::nvnQueueInitialize>();
  const auto finalize = load_entry<Entry::nvnQueueFinalize>();
  const auto submit = load_entry<Entry::nvnQueueSubmitCommands>();
  const auto flush = load_entry<Entry::nvnQueueFlush>();
  const auto finish = load_entry<Entry::nvnQueueFinish>();
  const auto get_error = load_entry<Entry::nvnQueueGetError>();
  const auto command_used =
      load_entry<Entry::nvnQueueGetTotalCommandMemoryUsed>();
  const auto control_used =
      load_entry<Entry::nvnQueueGetTotalControlMemoryUsed>();
  const auto compute_used =
      load_entry<Entry::nvnQueueGetTotalComputeMemoryUsed>();
  const auto reset_counts = load_entry<Entry::nvnQueueResetMemoryUsageCounts>();
  const auto sync_initialize = load_entry<Entry::nvnSyncInitialize>();
  const auto sync_finalize = load_entry<Entry::nvnSyncFinalize>();
  const auto fence_sync = load_entry<Entry::nvnQueueFenceSync>();
  const auto sync_wait = load_entry<Entry::nvnSyncWait>();
  const auto queue_wait_sync = load_entry<Entry::nvnQueueWaitSync>();

  CHECK(builder_set_device && builder_set_defaults && builder_memory_size &&
        initialize && submit && finish);

  alignas(8) NVNdevice device{};
  alignas(8) NVNqueueBuilder builder{};
  builder_set_device(&builder, &device);
  builder_set_defaults(&builder);
  const auto &defaults = *reinterpret_cast<const QueueBuilderState *>(&builder);
  CHECK(defaults.device == &device);
  CHECK(defaults.command_memory_size == 65'536);
  CHECK(defaults.compute_memory_size == 262'144);
  CHECK(defaults.control_memory_size == 16'384);
  CHECK(defaults.flush_threshold == 4'096);
  CHECK(builder_memory_size(&builder) == 344'064);

  builder_set_flags(&builder, NVN_QUEUE_BUILDER_FLAGS_NO_ZCULL);
  builder_set_command_size(&builder, 1);
  builder_set_compute_size(&builder, 2);
  builder_set_control_size(&builder, 3);
  builder_set_threshold(&builder, 4);
  CHECK(builder_memory_size(&builder) == 4'096);
  CHECK(reinterpret_cast<const QueueBuilderState *>(&builder)->flags ==
        NVN_QUEUE_BUILDER_FLAGS_NO_ZCULL);
  CHECK(
      reinterpret_cast<const QueueBuilderState *>(&builder)->flush_threshold ==
      4);

  std::array<std::byte, 4'096> queue_memory{};
  builder_set_memory(&builder, queue_memory.data(), queue_memory.size());
  alignas(8) NVNqueue queue{};
  CHECK(initialize(&queue, &builder) == 1);
  CHECK(queue_valid(&queue));
  CHECK(queue_last_fence(&queue) == 0);

  const std::array<NVNcommandHandle, 3> handles{4, 7, 9};
  submit(&queue, static_cast<int>(handles.size()), handles.data());
  CHECK(backend.submit_calls == 1);
  CHECK(backend.submitted ==
        std::vector<NVNcommandHandle>(handles.begin(), handles.end()));
  CHECK(queue_last_fence(&queue) == 11);
  CHECK(reinterpret_cast<const QueueState *>(&queue)->submits == 11);
  CHECK(command_used(&queue) == 0 && control_used(&queue) == 0 &&
        compute_used(&queue) == 0);
  reset_counts(&queue);

  flush(&queue);
  CHECK(backend.flush_calls == 1);

  alignas(8) NVNsync sync{};
  CHECK(sync_initialize(&sync, &device) == 1);
  CHECK(sync_wait(&sync, 1) == NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED);
  fence_sync(&queue, &sync, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
  CHECK(reinterpret_cast<const SyncState *>(&sync)->fence == 11);

  backend.wait_succeeds = false;
  CHECK(sync_wait(&sync, 17) == NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED);
  CHECK(backend.waited_fence == 11 && backend.waited_timeout == 17);
  backend.wait_succeeds = true;
  set_blocking_wait_hook(&test_wait_hook);
  wait_hook_called = false;
  CHECK(sync_wait(&sync, 99) == NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED);
  CHECK(wait_hook_called);
  CHECK(sync_wait(&sync, 0) == NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED);

  backend.completed = 0;
  wait_hook_called = false;
  CHECK(queue_wait_sync(&queue, &sync) == 1);
  CHECK(wait_hook_called && backend.completed == 11);

  backend.completed = 0;
  wait_hook_called = false;
  finish(&queue);
  CHECK(wait_hook_called && backend.completed == 11);
  CHECK(backend.flush_calls >= 3);
  set_blocking_wait_hook(nullptr);

  NVNqueueErrorInfo info{};
  CHECK(get_error(&queue, &info) == NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR);
  queue_error(&queue, "synthetic queue failure");
  CHECK(get_error(&queue, &info) ==
        NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN);

  sync_finalize(&sync);
  CHECK(sync_wait(&sync, 0) == NVN_SYNC_WAIT_RESULT_FAILED);
  finalize(&queue);
  CHECK(!queue_valid(&queue));
}

void verify_events() {
  alignas(4) std::uint32_t waited_value = 0;
  const EventState event_state{object_tag('E', 'V', 'N', 'T'), &waited_value};
  CHECK(!event_wait_value(event_state, NVN_EVENT_WAIT_MODE_EQUAL, 1, 0));
  CHECK(!event_wait_value(event_state, NVN_EVENT_WAIT_MODE_EQUAL, 1, 100'000));
  auto waiting = std::async(std::launch::async, [&] {
    return event_wait_value(event_state, NVN_EVENT_WAIT_MODE_EQUAL, 3,
                            1'000'000'000);
  });
  CHECK(waiting.wait_for(std::chrono::milliseconds(1)) ==
        std::future_status::timeout);
  CHECK(event_signal_value(event_state, NVN_EVENT_SIGNAL_MODE_WRITE, 3));
  CHECK(waiting.get());
  CHECK(event_signal_value(event_state, NVN_EVENT_SIGNAL_MODE_WRITE, 1));
  CHECK(event_wait_value(event_state, NVN_EVENT_WAIT_MODE_GEQUAL_WRAP,
                         UINT32_MAX, 0));
  CHECK(!event_wait_value(event_state, NVN_EVENT_WAIT_MODE_GEQUAL_WRAP, 2, 0));
  CHECK(
      event_signal_value(event_state, NVN_EVENT_SIGNAL_MODE_WRITE, UINT32_MAX));
  CHECK(!event_wait_value(event_state, NVN_EVENT_WAIT_MODE_GEQUAL_WRAP, 1, 0));
  std::stop_source stop;
  stop.request_stop();
  CHECK(!event_wait_value(event_state, NVN_EVENT_WAIT_MODE_EQUAL, 7, UINT64_MAX,
                          stop.get_token()));

  const auto builder_defaults = load_entry<Entry::nvnEventBuilderSetDefaults>();
  const auto builder_storage = load_entry<Entry::nvnEventBuilderSetStorage>();
  const auto initialize = load_entry<Entry::nvnEventInitialize>();
  const auto finalize = load_entry<Entry::nvnEventFinalize>();
  const auto get_value = load_entry<Entry::nvnEventGetValue>();
  const auto signal = load_entry<Entry::nvnEventSignal>();

  alignas(8) NVNeventBuilder builder{};
  alignas(8) NVNevent event{};
  builder_defaults(&builder);
  CHECK(initialize(&event, &builder) == 0);

  alignas(8) NVNmemoryPool pool{};
  alignas(4) std::array<std::uint32_t, 4> values{1, 2, 3, 4};
  auto &pool_state = *reinterpret_cast<MemoryPoolState *>(&pool);
  pool_state.tag = object_tag('P', 'O', 'O', 'L');
  pool_state.memory = reinterpret_cast<std::uint8_t *>(values.data());
  pool_state.size = sizeof(values);

  builder_storage(&builder, &pool, 4);
  CHECK(initialize(&event, &builder) == 1);
  CHECK(get_value(&event) == 2);
  signal(&event, NVN_EVENT_SIGNAL_MODE_ADD, 5);
  CHECK(get_value(&event) == 7);
  signal(&event, NVN_EVENT_SIGNAL_MODE_WRITE, 42);
  CHECK(get_value(&event) == 42 && values[1] == 42);

  builder_storage(&builder, &pool, -1);
  CHECK(initialize(&event, &builder) == 0);
  builder_storage(&builder, &pool,
                  static_cast<std::int64_t>(sizeof(values) - 3));
  CHECK(initialize(&event, &builder) == 0);
  builder_storage(&builder, &pool, 1);
  CHECK(initialize(&event, &builder) == 0);

  finalize(&event);
  CHECK(get_value(&event) == 0);
}

void verify_timestamps() {
  const auto current_timestamp =
      load_entry<Entry::nvnDeviceGetCurrentTimestampInNanoseconds>();
  const auto convert_timestamp =
      load_entry<Entry::nvnDeviceGetTimestampInNanoseconds>();

  CHECK(timestamp_ticks_from_nanoseconds(625) == 384);
  CHECK(timestamp_nanoseconds_from_ticks(384) == 625);
  CHECK(timestamp_nanoseconds_from_ticks(
            timestamp_ticks_from_nanoseconds(1'000'000)) <= 1'000'000);

  NVNcounterData counter{};
  counter.timestamp = 768;
  CHECK(convert_timestamp(nullptr, &counter) == 1'250);
  CHECK(convert_timestamp(nullptr, nullptr) == 0);

  const auto before = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  const std::uint64_t current = current_timestamp(nullptr);
  const auto after = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
  CHECK(current >= static_cast<std::uint64_t>(before - 625));
  CHECK(current <= static_cast<std::uint64_t>(after));
}

// With a published guest clock the current timestamp runs on it: the guest
// compares GPU timestamps with its own counter, which starts at that origin.
void verify_guest_clock_timestamps() {
  const auto current_timestamp =
      load_entry<Entry::nvnDeviceGetCurrentTimestampInNanoseconds>();
  struct Fake {
    std::chrono::nanoseconds elapsed{5'000'000'000};
  } fake;
  const auto read = [](void *context) noexcept
      -> std::optional<std::chrono::nanoseconds> {
    return static_cast<Fake *>(context)->elapsed;
  };
  armigo::execution::publish_guest_timeline(
      {read, &fake, std::chrono::steady_clock::now()});
  CHECK(current_timestamp(nullptr) == 5'000'000'000);
  fake.elapsed = std::chrono::nanoseconds(7'000'000'625);
  CHECK(current_timestamp(nullptr) == 7'000'000'625);
  // A clock that cannot sample keeps the timestamp on the steady clock.
  const auto broken = [](void *) noexcept
      -> std::optional<std::chrono::nanoseconds> { return std::nullopt; };
  armigo::execution::publish_guest_timeline(
      {broken, &fake, std::chrono::steady_clock::now()});
  const auto before = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  CHECK(current_timestamp(nullptr) >= static_cast<std::uint64_t>(before - 625));
  armigo::execution::withdraw_guest_timeline(&fake);
  CHECK(!armigo::execution::guest_timeline_now());
}

} // namespace

int main() {
  FakeReplayBackend backend;
  set_backends({nullptr, &backend});
  select_api_version({55, 15});
  register_queue_entries();
  register_sync_entries();

  verify_queue_and_sync(backend);
  verify_events();
  verify_timestamps();
  verify_guest_clock_timestamps();

  set_blocking_wait_hook(nullptr);
  set_backends({});
  std::printf("NVN presentation and sync failures: %u\n", failures);
  return failures ? 1 : 0;
}
