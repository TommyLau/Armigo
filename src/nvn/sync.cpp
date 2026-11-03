// ============================================================================
// Armigo NVN - Synchronization, Events, and Device Timestamps
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "backend.h"
#include "entry_table.h"
#include "present_internal.h"
#include "armigo/nvn/timeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <thread>

namespace armigo::nvn {
namespace {

constexpr std::uint32_t sync_tag = object_tag('S', 'Y', 'N', 'C');
constexpr std::uint32_t event_tag = object_tag('E', 'V', 'N', 'T');
constexpr std::uint32_t pool_tag = object_tag('P', 'O', 'O', 'L');
constexpr std::uint64_t timestamp_ticks_per_period = 384;
constexpr std::uint64_t timestamp_nanoseconds_per_period = 625;

struct EventStorage {
  EventState state{};
  const MemoryPoolState *pool = nullptr;
  std::int64_t offset = 0;
};
static_assert(sizeof(EventStorage) <= sizeof(NVNevent));

std::atomic<BlockingWaitHook> wait_hook{nullptr};

SyncState *sync_state(NVNsync *sync) noexcept {
  return reinterpret_cast<SyncState *>(sync);
}
const SyncState *sync_state(const NVNsync *sync) noexcept {
  return reinterpret_cast<const SyncState *>(sync);
}
EventBuilderState *event_builder(NVNeventBuilder *builder) noexcept {
  return reinterpret_cast<EventBuilderState *>(builder);
}
const EventBuilderState *
event_builder(const NVNeventBuilder *builder) noexcept {
  return reinterpret_cast<const EventBuilderState *>(builder);
}
EventStorage *event_storage(NVNevent *event) noexcept {
  return reinterpret_cast<EventStorage *>(event);
}
const EventStorage *event_storage(const NVNevent *event) noexcept {
  return reinterpret_cast<const EventStorage *>(event);
}

bool sync_valid(const NVNsync *sync) noexcept {
  return sync && sync_state(sync)->tag == sync_tag;
}
bool event_valid(const NVNevent *event) noexcept {
  return event && event_storage(event)->state.tag == event_tag;
}

void report_sync_failure(const char *operation, const char *message) noexcept {
  armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: %s failed: %s\n", operation, message);
}

NVNboolean sync_initialize(NVNsync *sync, NVNdevice *device) {
  if (!sync || !device)
    return 0;
  *sync_state(sync) = SyncState{sync_tag, 0, 0};
  return 1;
}

void sync_finalize(NVNsync *sync) {
  if (sync)
    *sync_state(sync) = SyncState{};
}

void sync_set_debug_label(NVNsync *, const char *) {}

void queue_fence_sync(NVNqueue *queue, NVNsync *sync,
                      NVNsyncCondition condition, int flags) {
  (void)flags;
  if (!queue_valid(queue) || !sync_valid(sync) ||
      (condition != NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE &&
       condition != NVN_SYNC_CONDITION_GRAPHICS_WORLD_SPACE_COMPLETE)) {
    queue_error(queue,
                "fence sync received an invalid queue, sync, or condition");
    return;
  }
  sync_state(sync)->fenced = 1;
  sync_state(sync)->fence = queue_last_fence(queue);
}

NVNsyncWaitResult sync_wait(const NVNsync *sync, std::uint64_t timeout) {
  if (!sync_valid(sync) || !backends().replay)
    return NVN_SYNC_WAIT_RESULT_FAILED;
  const SyncState &state = *sync_state(sync);
  if (!state.fenced)
    return NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;

  ReplayBackend *replay = backends().replay;
  try {
    if (replay->completed_fence() >= state.fence)
      return NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED;
    if (!timeout)
      return NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;
    const bool completed =
        blocking_wait([replay, fence = state.fence, timeout] {
          return replay->wait_fence(fence, timeout);
        }, "sync_wait");
    return completed ? NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED
                     : NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;
  } catch (const std::exception &exception) {
    report_sync_failure("sync wait", exception.what());
  } catch (...) {
    report_sync_failure("sync wait",
                        "replay backend threw an unknown exception");
  }
  return NVN_SYNC_WAIT_RESULT_FAILED;
}

NVNboolean queue_wait_sync(NVNqueue *queue, const NVNsync *sync) {
  if (!queue_valid(queue) || !sync_valid(sync) || !sync_state(sync)->fenced ||
      !backends().replay) {
    queue_error(queue, "wait sync received an invalid or unfenced sync");
    return 0;
  }
  ReplayBackend *replay = backends().replay;
  const std::uint64_t fence = sync_state(sync)->fence;
  try {
    if (replay->completed_fence() >= fence)
      return 1;
    replay->flush();
    if (blocking_wait(
            [replay, fence] { return replay->wait_fence(fence, UINT64_MAX); }, "queue_wait_sync"))
      return 1;
    queue_error(queue, "replay backend did not complete queue wait sync");
  } catch (const std::exception &exception) {
    queue_error(queue, exception.what());
  } catch (...) {
    queue_error(
        queue,
        "replay backend threw an unknown exception during queue wait sync");
  }
  return 0;
}

void event_builder_set_defaults(NVNeventBuilder *builder) {
  if (builder)
    *event_builder(builder) = EventBuilderState{};
}

void event_builder_set_storage(NVNeventBuilder *builder,
                               const NVNmemoryPool *pool, std::int64_t offset) {
  if (!builder)
    return;
  event_builder(builder)->pool =
      reinterpret_cast<const MemoryPoolState *>(pool);
  event_builder(builder)->offset = offset;
}

NVNboolean event_initialize(NVNevent *event, const NVNeventBuilder *builder) {
  if (!event || !builder)
    return 0;
  const EventBuilderState &source = *event_builder(builder);
  if (!source.pool || source.pool->tag != pool_tag || !source.pool->memory ||
      source.offset < 0) {
    report_sync_failure("event initialization",
                        "storage pool is missing or invalid");
    return 0;
  }
  const std::uint64_t offset = static_cast<std::uint64_t>(source.offset);
  if (offset > source.pool->size ||
      source.pool->size - offset < sizeof(std::uint32_t)) {
    report_sync_failure("event initialization",
                        "storage offset is outside the memory pool");
    return 0;
  }
  const std::uintptr_t base =
      reinterpret_cast<std::uintptr_t>(source.pool->memory);
  if (offset > std::numeric_limits<std::uintptr_t>::max() - base ||
      ((base + offset) & (alignof(std::uint32_t) - 1)) != 0) {
    report_sync_failure("event initialization",
                        "storage address is unaligned or overflows");
    return 0;
  }

  EventStorage initialized{};
  initialized.state.tag = event_tag;
  initialized.state.value =
      reinterpret_cast<volatile std::uint32_t *>(base + offset);
  initialized.pool = source.pool;
  initialized.offset = source.offset;
  *event_storage(event) = initialized;
  return 1;
}

void event_finalize(NVNevent *event) {
  if (event)
    *event_storage(event) = EventStorage{};
}

std::uint32_t event_get_value(const NVNevent *event) {
  if (!event_valid(event) || !event_storage(event)->state.value)
    return 0;
  auto *value = const_cast<std::uint32_t *>(event_storage(event)->state.value);
  return std::atomic_ref<std::uint32_t>(*value).load(std::memory_order_acquire);
}

void event_signal(NVNevent *event, NVNeventSignalMode mode,
                  std::uint32_t signal) {
  if (!event_valid(event) || !event_storage(event)->state.value) {
    report_sync_failure("event signal", "event is invalid");
    return;
  }
  if (!event_signal_value(event_storage(event)->state, mode, signal))
    report_sync_failure("event signal", "signal mode is invalid");
}

std::uint64_t device_get_current_timestamp(const NVNdevice *) {
  return timestamp_nanoseconds_from_ticks(current_timestamp_ticks());
}

std::uint64_t device_get_timestamp(const NVNdevice *,
                                   const NVNcounterData *counter) {
  return counter ? timestamp_nanoseconds_from_ticks(counter->timestamp) : 0;
}

} // namespace

void set_blocking_wait_hook(BlockingWaitHook hook) noexcept {
  wait_hook.store(hook, std::memory_order_release);
}

bool blocking_wait(const std::function<bool()> &wait, const char *tag) {
  if (!wait)
    return false;
  if (!timeline_enabled()) {
    if (const BlockingWaitHook hook = wait_hook.load(std::memory_order_acquire))
      return hook(wait);
    return wait();
  }
  // A guest thread blocked on a fence is invisible in the GPU's own clock.
  const auto thread = std::uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  static thread_local char wait_name[64];
  std::snprintf(wait_name, sizeof wait_name, "wait_begin:%s", tag ? tag : "wait");
  timeline_event(wait_name, thread, 0);
  const auto begin = timeline_now_ns();
  bool result = false;
  if (const BlockingWaitHook hook = wait_hook.load(std::memory_order_acquire))
    result = hook(wait);
  else
    result = wait();
  std::snprintf(wait_name, sizeof wait_name, "wait_end:%s", tag ? tag : "wait");
  timeline_event(wait_name, thread, timeline_now_ns() - begin);
  return result;
}

bool event_signal_value(const EventState &event, NVNeventSignalMode mode,
                        std::uint32_t signal) noexcept {
  if (event.tag != event_tag || !event.value)
    return false;
  std::atomic_ref<std::uint32_t> value(
      *const_cast<std::uint32_t *>(event.value));
  if (mode == NVN_EVENT_SIGNAL_MODE_WRITE)
    value.store(signal, std::memory_order_release);
  else if (mode == NVN_EVENT_SIGNAL_MODE_ADD)
    value.fetch_add(signal, std::memory_order_acq_rel);
  else
    return false;
  return true;
}

bool event_wait_value(const EventState &event, NVNeventWaitMode mode,
                      std::uint32_t target, std::uint64_t timeout_ns,
                      std::stop_token stop) {
  if (event.tag != event_tag || !event.value ||
      (mode != NVN_EVENT_WAIT_MODE_EQUAL &&
       mode != NVN_EVENT_WAIT_MODE_GEQUAL_WRAP))
    return false;
  std::atomic_ref<std::uint32_t> value(
      *const_cast<std::uint32_t *>(event.value));
  const auto reached = [&] {
    const auto observed = value.load(std::memory_order_acquire);
    return mode == NVN_EVENT_WAIT_MODE_EQUAL
               ? observed == target
               : std::uint32_t(observed - target) < 0x80000000u;
  };
  if (reached())
    return true;
  if (!timeout_ns || stop.stop_requested())
    return false;
  const auto begin = std::chrono::steady_clock::now();
  return blocking_wait([&] {
    while (!stop.stop_requested()) {
      if (reached())
        return true;
      const auto elapsed = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - begin)
              .count());
      if (timeout_ns != UINT64_MAX && elapsed >= timeout_ns)
        return false;
      // GPU writes do not notify a CPU condition variable. Poll coherent
      // pool memory inside one scheduler-visible blocking operation.
      const auto delay =
          timeout_ns == UINT64_MAX
              ? 50'000ull
              : std::min<std::uint64_t>(50'000, timeout_ns - elapsed);
      std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
    }
    return false;
  });
}

std::uint64_t
timestamp_ticks_from_nanoseconds(std::uint64_t nanoseconds) noexcept {
  return (nanoseconds / timestamp_nanoseconds_per_period) *
             timestamp_ticks_per_period +
         (nanoseconds % timestamp_nanoseconds_per_period) *
             timestamp_ticks_per_period / timestamp_nanoseconds_per_period;
}

std::uint64_t timestamp_nanoseconds_from_ticks(std::uint64_t ticks) noexcept {
  return (ticks / timestamp_ticks_per_period) *
             timestamp_nanoseconds_per_period +
         (ticks % timestamp_ticks_per_period) *
             timestamp_nanoseconds_per_period / timestamp_ticks_per_period;
}

std::uint64_t current_timestamp_ticks() noexcept {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  return timestamp_ticks_from_nanoseconds(
      static_cast<std::uint64_t>(nanoseconds));
}

void register_sync_entries() noexcept {
  install<Entry::nvnDeviceGetCurrentTimestampInNanoseconds>(
      &device_get_current_timestamp);
  install<Entry::nvnDeviceGetTimestampInNanoseconds>(&device_get_timestamp);
  install<Entry::nvnSyncInitialize>(&sync_initialize);
  install<Entry::nvnSyncFinalize>(&sync_finalize);
  install<Entry::nvnSyncSetDebugLabel>(&sync_set_debug_label);
  install<Entry::nvnQueueFenceSync>(&queue_fence_sync);
  install<Entry::nvnSyncWait>(&sync_wait);
  install<Entry::nvnQueueWaitSync>(&queue_wait_sync);
  install<Entry::nvnEventBuilderSetDefaults>(&event_builder_set_defaults);
  install<Entry::nvnEventBuilderSetStorage>(&event_builder_set_storage);
  install<Entry::nvnEventInitialize>(&event_initialize);
  install<Entry::nvnEventFinalize>(&event_finalize);
  install<Entry::nvnEventGetValue>(&event_get_value);
  install<Entry::nvnEventSignal>(&event_signal);
}

} // namespace armigo::nvn
