// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "commands.h"
#include "entry_table.h"
#include "nvn_replay.h"
#include "nvn_submission_segments.h"

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

namespace {

using namespace armigo;
using namespace armigo::nvn;
using namespace armigo::nvn::api;
namespace replay = armigo::platform::metal::nvn_driver;

unsigned failures = 0;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);             \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

template <Entry E> typename EntrySignature<E>::type load() {
  return reinterpret_cast<typename EntrySignature<E>::type>(
      armigo_nvn_bootstrap_loader(entry_name(E).data()));
}

// Feedback may precede worker publication or arrive out of commit order.
void test_segment_lifecycle() {
  replay::SubmissionSegments state;
  state.submit();
  state.submit();
  state.complete(false);
  CHECK(!state.retire());
  state.seal();
  CHECK(!state.retire());
  state.complete(true);
  CHECK(state.failed && state.retire());
  CHECK(!state.retire());
  replay::SubmissionSegments early;
  early.submit();
  early.complete(false);
  CHECK(!early.retire());
  early.submit();
  early.complete(false);
  early.seal();
  CHECK(early.retire());
  replay::SegmentPolicy policy;
  CHECK(!policy.admit(0, 0));
  CHECK(policy.admit(1, 15));
  CHECK(!policy.admit(32, 16));
  CHECK(policy.admit(32, 15));
  policy.draws = 16;
  CHECK(!policy.admit(15, 0));
  CHECK(policy.admit(16, 15));
  policy.draws = 0;
  CHECK(!policy.admit(100, 0));
}

class SharedResources final : public replay::ReplayResources {
public:
  static constexpr std::uint64_t base = 0x10000000;

  SharedResources(id<MTLDevice> device, std::size_t size) {
    memory = [device newBufferWithLength:size
                                 options:MTLResourceStorageModeShared];
  }

  replay::BufferSpan buffer_at(std::uint64_t address,
                               std::uint64_t size) override {
    if (on_lookup) on_lookup(address);
    if (!memory || address < base)
      return {};
    const std::uint64_t offset = address - base;
    const std::uint64_t length = memory.length;
    if (offset > length || size > length - offset)
      return {};
    return {memory, offset};
  }

  id<MTLTexture> texture(const TextureState &,
                         const TextureViewState *) override {
    return nil;
  }

  id<MTLTexture> texture_for_id(std::int32_t) override { return nil; }
  id<MTLSamplerState> sampler_for_id(std::int32_t) override { return nil; }
  void note_gpu_write(const TextureState &) override { ++texture_writes; }
  void note_gpu_write(std::uint64_t, std::uint64_t) override {
    ++buffer_writes;
  }

  id<MTLBuffer> memory = nil;
  unsigned texture_writes = 0;
  unsigned buffer_writes = 0;
  std::function<void(std::uint64_t)> on_lookup;
};

using AdapterBufferLookup =
    replay::BufferSpan (SharedResources::*)(std::uint64_t, std::uint64_t);
using AdapterTextureLookup = id<MTLTexture> (SharedResources::*)(
    const TextureState &, const TextureViewState *);
using AdapterTextureIdLookup =
    id<MTLTexture> (SharedResources::*)(std::int32_t);
using AdapterSamplerIdLookup =
    id<MTLSamplerState> (SharedResources::*)(std::int32_t);
using AdapterTextureWrite = void (SharedResources::*)(const TextureState &);
using AdapterBufferWrite = void (SharedResources::*)(std::uint64_t,
                                                     std::uint64_t);

static_assert(
    std::same_as<decltype(&SharedResources::buffer_at), AdapterBufferLookup>);
static_assert(
    std::same_as<decltype(&SharedResources::texture), AdapterTextureLookup>);
static_assert(std::same_as<decltype(&SharedResources::texture_for_id),
                           AdapterTextureIdLookup>);
static_assert(std::same_as<decltype(&SharedResources::sampler_for_id),
                           AdapterSamplerIdLookup>);
static_assert(std::same_as<decltype(static_cast<AdapterTextureWrite>(
                               &SharedResources::note_gpu_write)),
                           AdapterTextureWrite>);
static_assert(std::same_as<decltype(static_cast<AdapterBufferWrite>(
                               &SharedResources::note_gpu_write)),
                           AdapterBufferWrite>);

struct Recorder {
  NVNdevice device{};
  NVNcommandBuffer commands{};
  NVNmemoryPool pool{};
  std::array<std::byte, 64 * 1024> donated{};

  Recorder() {
    auto &pool_state = *reinterpret_cast<MemoryPoolState *>(&pool);
    pool_state.memory = reinterpret_cast<std::uint8_t *>(donated.data());
    pool_state.size = donated.size();
    CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device) == 1);
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 0,
                                                    donated.size());
    load<Entry::nvnCommandBufferAddControlMemory>()(&commands, donated.data(),
                                                    donated.size());
  }

  NVNcommandHandle finish() {
    return load<Entry::nvnCommandBufferEndRecording>()(&commands);
  }

  void release(NVNcommandHandle handle) {
    if (handle)
      load<Entry::nvnDeviceFinalizeCommandHandle>()(&device, handle);
  }

  void finalize() { load<Entry::nvnCommandBufferFinalize>()(&commands); }
};

template <typename T>
T read_at(const SharedResources &resources, std::size_t offset) {
  T value{};
  std::memcpy(&value,
              static_cast<const std::byte *>(resources.memory.contents) +
                  offset,
              sizeof value);
  return value;
}

template <class Predicate> bool eventually(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

// A blocked GPU must still allow six encodes, but never a seventh. Reports
// keep timestamp storage live in every slot and the seventh submission reuses
// the oldest slot only after completion. Read guest results after the last fence.
void test_six_inflight_submissions(id<MTLDevice> device,
                                     id<MTL4CommandQueue> queue) {
  SharedResources resources(device, 64 * 1024);
  auto backend = replay::create_replay_backend(device, queue, resources);
  id<MTLSharedEvent> gate = [device newSharedEvent];
  [queue waitForEvent:gate value:1];
  Recorder recorder;
  std::array<NVNcommandHandle, 7> handles{};
  std::uint64_t last = 0;
  for (unsigned i = 0; i != handles.size(); ++i) {
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferReportValue>()(
        &recorder.commands, i + 1, SharedResources::base + i * 64);
    load<Entry::nvnCommandBufferReportCounter>()(
        &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP,
        SharedResources::base + i * 64 + 16);
    handles[i] = recorder.finish();
    last = backend->submit({&handles[i], 1});
    CHECK(last != 0);
  }
  CHECK(eventually([&] { return backend->statistics().submissions == 6; }));
  CHECK(backend->completed_fence() == 0);
  CHECK(!backend->wait_fence(last, 1'000'000));
  CHECK(backend->statistics().upload_arena_allocations == 6);
  gate.signaledValue = 1;
  CHECK(backend->wait_fence(last, 10'000'000'000ull));
  CHECK(!backend->last_error());
  CHECK(backend->statistics().upload_arena_allocations == 6);
  CHECK(backend->statistics().upload_arena_reuses == 1);
  std::uint64_t previous_timestamp = 0;
  for (unsigned i = 0; i != handles.size(); ++i) {
    CHECK(read_at<std::uint32_t>(resources, i * 64) == i + 1);
    const auto report = read_at<NVNcounterData>(resources, i * 64 + 16);
    CHECK(report.timestamp >= previous_timestamp && report.timestamp != 0);
    previous_timestamp = report.timestamp;
    recorder.release(handles[i]);
  }
  recorder.finalize();
}

// Hold the first submission on the GPU until the second encoder reaches a
// readback split. Its synchronous segment cannot finish if the first callback
// needs a mutex held by that encoder.
void test_feedback_during_encoder_split(id<MTLDevice> device,
                                        id<MTL4CommandQueue> queue) {
  SharedResources resources(device, 64 * 1024);
  auto backend = replay::create_replay_backend(device, queue, resources);
  id<MTLSharedEvent> gate = [device newSharedEvent];
  [queue waitForEvent:gate value:1];
  std::atomic<bool> second_encoding{false};
  resources.on_lookup = [&](std::uint64_t address) {
    if (address == SharedResources::base + 64) second_encoding = true;
  };
  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferReportValue>()(&recorder.commands, 7, SharedResources::base);
  const auto first = recorder.finish();
  CHECK(backend->submit({&first, 1}) != 0);

  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferReportValue>()(&recorder.commands, 9, SharedResources::base + 64);
  // A counter report followed by a GPU copy requests a synchronous segment.
  load<Entry::nvnCommandBufferReportCounter>()(
      &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, SharedResources::base + 128);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + 128,
      SharedResources::base + 256, sizeof(NVNcounterData), NVN_COPY_FLAGS_NONE);
  const auto second = recorder.finish();
  const auto fence = backend->submit({&second, 1});
  CHECK(eventually([&] { return second_encoding.load(); }));
  gate.signaledValue = 1;
  CHECK(backend->wait_fence(fence, 10'000'000'000ull));
  CHECK(!backend->last_error());
  CHECK(backend->statistics().synchronize_gpu_waits == 1);
  CHECK(read_at<std::uint32_t>(resources, 0) == 7);
  CHECK(read_at<std::uint32_t>(resources, 64) == 9);
  CHECK(read_at<NVNcounterData>(resources, 128).timestamp ==
        read_at<NVNcounterData>(resources, 256).timestamp);
  recorder.release(first);
  recorder.release(second);
  recorder.finalize();
}

void test_report_value_precedes_gpu_copy(id<MTLDevice> device,
                                         id<MTL4CommandQueue> queue) {
  SharedResources resources(device, 64 * 1024);
  CHECK(resources.memory != nil);
  if (!resources.memory)
    return;
  auto backend = replay::create_replay_backend(device, queue, resources);
  CHECK(backend != nullptr);
  if (!backend)
    return;

  constexpr std::size_t reported_offset = 0x100;
  constexpr std::size_t copied_offset = 0x200;
  constexpr std::uint32_t expected = 0x6d5a39c7;
  std::memset(resources.memory.contents, 0, resources.memory.length);

  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferReportValue>()(
      &recorder.commands, expected, SharedResources::base + reported_offset);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + reported_offset,
      SharedResources::base + copied_offset, sizeof expected,
      NVN_COPY_FLAGS_NONE);
  const auto handle = recorder.finish();
  CHECK(handle != 0);
  const auto fence = backend->submit({&handle, 1});
  CHECK(fence != 0);
  CHECK(backend->wait_fence(fence, 10'000'000'000ull));

  const auto reported = read_at<std::uint32_t>(resources, reported_offset);
  const auto copied = read_at<std::uint32_t>(resources, copied_offset);
  CHECK(reported == expected);
  CHECK(copied == reported);
  CHECK(resources.buffer_writes >= 2);

  recorder.release(handle);
  recorder.finalize();
}

void test_event_orders_gpu_work_in_one_stream(id<MTLDevice> device,
                                              id<MTL4CommandQueue> queue) {
  SharedResources resources(device, 64 * 1024);
  CHECK(resources.memory != nil);
  if (!resources.memory)
    return;
  auto backend = replay::create_replay_backend(device, queue, resources);
  CHECK(backend != nullptr);
  if (!backend)
    return;

  constexpr std::size_t cleared_offset = 0x1000;
  constexpr std::size_t staged_offset = 0x2000;
  constexpr std::size_t copied_offset = 0x3000;
  constexpr std::size_t event_offset = 0x4000;
  constexpr std::size_t event_copy_offset = 0x5000;
  constexpr std::size_t transfer_size = 512;
  constexpr std::uint32_t clear_value = 0x4b82d19e;
  constexpr std::uint32_t signal_value = 0xa1257c3d;
  std::memset(resources.memory.contents, 0, resources.memory.length);

  auto *event_value = reinterpret_cast<volatile std::uint32_t *>(
      static_cast<std::byte *>(resources.memory.contents) + event_offset);
  NVNevent event{};
  reinterpret_cast<EventState *>(&event)->value = event_value;

  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferClearBuffer>()(
      &recorder.commands, SharedResources::base + cleared_offset, transfer_size,
      clear_value);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + cleared_offset,
      SharedResources::base + staged_offset, transfer_size,
      NVN_COPY_FLAGS_NONE);
  load<Entry::nvnCommandBufferSignalEvent>()(
      &recorder.commands, &event, NVN_EVENT_SIGNAL_MODE_WRITE,
      NVN_EVENT_SIGNAL_LOCATION_BOTTOM, 0, signal_value);
  load<Entry::nvnCommandBufferWaitEvent>()(
      &recorder.commands, &event, NVN_EVENT_WAIT_MODE_EQUAL, signal_value);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + event_offset,
      SharedResources::base + event_copy_offset, sizeof signal_value,
      NVN_COPY_FLAGS_NONE);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + staged_offset,
      SharedResources::base + copied_offset, transfer_size,
      NVN_COPY_FLAGS_NONE);
  const auto handle = recorder.finish();
  CHECK(handle != 0);
  const auto fence = backend->submit({&handle, 1});
  CHECK(fence != 0);
  CHECK(backend->wait_fence(fence, 10'000'000'000ull));

  CHECK(*event_value == signal_value);
  CHECK(read_at<std::uint32_t>(resources, event_copy_offset) == signal_value);
  const auto *bytes = static_cast<const std::byte *>(resources.memory.contents);
  CHECK(std::memcmp(bytes + staged_offset, bytes + copied_offset,
                    transfer_size) == 0);
  for (std::size_t offset = 0; offset != transfer_size;
       offset += sizeof clear_value) {
    CHECK(read_at<std::uint32_t>(resources, copied_offset + offset) ==
          clear_value);
  }

  recorder.release(handle);
  recorder.finalize();
}

void test_timestamp_reports_precede_gpu_copies(id<MTLDevice> device,
                                               id<MTL4CommandQueue> queue) {
  constexpr std::size_t scratch_size = 4 * 1024 * 1024;
  SharedResources resources(device, scratch_size + 64 * 1024);
  CHECK(resources.memory != nil);
  if (!resources.memory)
    return;
  auto backend = replay::create_replay_backend(device, queue, resources);
  CHECK(backend != nullptr);
  if (!backend)
    return;

  constexpr std::size_t first_offset = 0x100;
  constexpr std::size_t first_copy_offset = 0x200;
  constexpr std::size_t second_offset = 0x300;
  constexpr std::size_t second_copy_offset = 0x400;
  constexpr std::size_t scratch_offset = 0x10000;
  std::memset(resources.memory.contents, 0, resources.memory.length);

  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferReportCounter>()(
      &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP,
      SharedResources::base + first_offset);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + first_offset,
      SharedResources::base + first_copy_offset, sizeof(NVNcounterData),
      NVN_COPY_FLAGS_NONE);
  load<Entry::nvnCommandBufferClearBuffer>()(
      &recorder.commands, SharedResources::base + scratch_offset, scratch_size,
      0x18e73ac5);
  load<Entry::nvnCommandBufferReportCounter>()(
      &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP,
      SharedResources::base + second_offset);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + second_offset,
      SharedResources::base + second_copy_offset, sizeof(NVNcounterData),
      NVN_COPY_FLAGS_NONE);
  const auto handle = recorder.finish();
  CHECK(handle != 0);
  const auto fence = backend->submit({&handle, 1});
  CHECK(fence != 0);
  CHECK(backend->wait_fence(fence, 10'000'000'000ull));

  const auto first = read_at<NVNcounterData>(resources, first_offset);
  const auto first_copy = read_at<NVNcounterData>(resources, first_copy_offset);
  const auto second = read_at<NVNcounterData>(resources, second_offset);
  const auto second_copy =
      read_at<NVNcounterData>(resources, second_copy_offset);
  CHECK(first.timestamp != 0 && first.counter == 0);
  CHECK(second.timestamp != 0 && second.counter == 0);
  CHECK(second.timestamp > first.timestamp);
  // Reports are stamped on the clock the driver's current timestamp reads, in
  // its 384/625 ticks per nanosecond, from GPU times inside the window sampled
  // around the submission rather than from the completion time.
  {
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    const auto now_ticks = now_ns / 625 * 384 + now_ns % 625 * 384 / 625;
    constexpr std::uint64_t five_seconds = 5'000'000'000ull / 625 * 384;
    CHECK(first.timestamp <= now_ticks && now_ticks - first.timestamp < five_seconds);
    CHECK(second.timestamp <= now_ticks && now_ticks - second.timestamp < five_seconds);
    CHECK(backend->statistics().counter_timestamp_fallbacks == 0);
  }
  CHECK(std::memcmp(&first, &first_copy, sizeof first) == 0);
  CHECK(std::memcmp(&second, &second_copy, sizeof second) == 0);

  recorder.release(handle);
  recorder.finalize();
}

void test_rejected_submit_does_not_leave_fence_gap(id<MTLDevice> device,
                                                   id<MTL4CommandQueue> queue) {
  SharedResources resources(device, 64 * 1024);
  CHECK(resources.memory != nil);
  if (!resources.memory)
    return;
  auto backend = replay::create_replay_backend(device, queue, resources);
  CHECK(backend != nullptr);
  if (!backend)
    return;

  const NVNcommandHandle invalid = std::numeric_limits<NVNcommandHandle>::max();
  CHECK(backend->submit({&invalid, 1}) == 0);
  CHECK(backend->last_error().code == replay::ReplayErrorCode::invalid_command);

  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferClearBuffer>()(
      &recorder.commands, SharedResources::base + 0x100, 64, 0x91f42b6d);
  const auto handle = recorder.finish();
  CHECK(handle != 0);
  const auto fence = backend->submit({&handle, 1});
  CHECK(fence != 0);
  CHECK(backend->wait_fence(fence, 2'000'000'000ull));
  CHECK(backend->completed_fence() >= fence);
  CHECK(read_at<std::uint32_t>(resources, 0x100) == 0x91f42b6d);

  recorder.release(handle);
  recorder.finalize();
}

void test_destructor_drains_pending_callbacks(id<MTLDevice> device) {
  constexpr std::size_t copy_size = 8 * 1024 * 1024;
  constexpr std::size_t destination_offset = copy_size;
  constexpr std::size_t completion_offset = copy_size * 2;
  constexpr std::size_t completion_copy_offset = completion_offset + 0x100;
  constexpr std::uint32_t completion_value = 0x5ed84a13;
  SharedResources resources(device, completion_offset + 64 * 1024);
  CHECK(resources.memory != nil);
  if (!resources.memory)
    return;
  std::memset(resources.memory.contents, 0, resources.memory.length);
  std::memset(resources.memory.contents, 0x6b, copy_size);

  id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
  CHECK(queue != nil);
  if (!queue)
    return;

  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferReportValue>()(
      &recorder.commands, completion_value,
      SharedResources::base + completion_offset);
  load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
      &recorder.commands, SharedResources::base + completion_offset,
      SharedResources::base + completion_copy_offset, sizeof completion_value,
      NVN_COPY_FLAGS_NONE);
  for (unsigned iteration = 0; iteration != 512; ++iteration) {
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
        &recorder.commands, SharedResources::base,
        SharedResources::base + destination_offset, copy_size,
        NVN_COPY_FLAGS_NONE);
  }
  const auto handle = recorder.finish();
  CHECK(handle != 0);

  auto backend = replay::create_replay_backend(device, queue, resources);
  CHECK(backend != nullptr);
  if (backend) {
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence != 0);
    backend.reset();
    CHECK(read_at<std::uint32_t>(resources, completion_offset) ==
          completion_value);
    CHECK(read_at<std::uint32_t>(resources, completion_copy_offset) ==
          completion_value);
    const auto *bytes =
        static_cast<const std::uint8_t *>(resources.memory.contents);
    CHECK(bytes[destination_offset] == 0x6b);
    CHECK(bytes[destination_offset + copy_size - 1] == 0x6b);
  }

  recorder.release(handle);
  recorder.finalize();
}

} // namespace

int main() {
  test_segment_lifecycle();
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] ||
        !device.hasUnifiedMemory) {
      std::fprintf(stderr, "unified-memory Metal 4 device is unavailable\n");
      return 1;
    }
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    CHECK(queue != nil);
    if (queue) {
      select_api_version({55, 15});
      register_command_entries();
      test_six_inflight_submissions(device, queue);
      test_feedback_during_encoder_split(device, queue);
      test_report_value_precedes_gpu_copy(device, queue);
      test_event_orders_gpu_work_in_one_stream(device, queue);
      test_timestamp_reports_precede_gpu_copies(device, queue);
      test_rejected_submit_does_not_leave_fence_gap(device, queue);
      test_destructor_drains_pending_callbacks(device);
    }
  }
  std::printf("Metal NVN replay ordering failures: %u\n", failures);
  return failures ? 1 : 0;
}
