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
#include "nvn_primitive_expansion.h"
#include "nvn_gpu_frame_work.h"

#include <algorithm>
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
#include <vector>

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

// Quads and fans become one triangle list; restarts end a quad or fan and
// a trailing partial quad draws nothing.
void test_primitive_expansion() {
  using replay::expand_triangles;
  using replay::expanded_capacity;
  std::array<std::uint32_t, 32> out{};
  // Two quads of 16-bit indices with a restart between them and a partial
  // quad cut short by the restart.
  const std::array<std::uint16_t, 11> quads{0, 1, 2, 3, 9, 0xffff, 4, 5, 6, 7, 8};
  CHECK(expanded_capacity(false, quads.size()) == 12);
  auto used = expand_triangles(out.data(), reinterpret_cast<const std::uint8_t*>(quads.data()), 2, 0,
                               quads.size(), false, true, 0xffff);
  const std::array<std::uint32_t, 12> quad_triangles{0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
  CHECK(used == quad_triangles.size());
  CHECK(std::equal(quad_triangles.begin(), quad_triangles.end(), out.begin()));
  // Without restart 0xffff is an ordinary index.
  const std::array<std::uint16_t, 4> plain{0xffff, 1, 2, 3};
  used = expand_triangles(out.data(), reinterpret_cast<const std::uint8_t*>(plain.data()), 2, 0, plain.size(),
                          false, false, 0);
  CHECK(used == 6 && out[0] == 0xffff && out[4] == 2 && out[5] == 3);
  // A byte-index fan restarted after two triangles, then a fan of one.
  const std::array<std::uint8_t, 8> fan{10, 11, 12, 13, 0xff, 20, 21, 22};
  CHECK(expanded_capacity(true, fan.size()) == 18);
  used = expand_triangles(out.data(), fan.data(), 1, 0, fan.size(), true, true, 0xff);
  const std::array<std::uint32_t, 9> fan_triangles{10, 11, 12, 10, 12, 13, 20, 21, 22};
  CHECK(used == fan_triangles.size());
  CHECK(std::equal(fan_triangles.begin(), fan_triangles.end(), out.begin()));
  // A non-indexed quad draw numbers its vertices from `first`.
  used = expand_triangles(out.data(), nullptr, 4, 100, 9, false, false, 0);
  CHECK(used == 12 && out[0] == 100 && out[5] == 103 && out[6] == 104 && out[11] == 107);
  // Fewer than three fan vertices or four quad vertices draw nothing.
  CHECK(expanded_capacity(true, 2) == 0 && expanded_capacity(false, 3) == 0);
  // The largest fan one Metal draw can take, and the first that is refused.
  CHECK(expanded_capacity(true, 1'431'655'767) == replay::max_expanded_indices);
  CHECK(expanded_capacity(true, 1'431'655'768) > replay::max_expanded_indices);
  CHECK(expanded_capacity(false, 0x7fffffff) <= replay::max_expanded_indices);
  // Metal's own restart value cannot be drawn.
  const std::array<std::uint32_t, 4> sentinel{0, 1, 2, 0xffffffffu};
  CHECK(expand_triangles(out.data(), reinterpret_cast<const std::uint8_t*>(sentinel.data()), 4, 0,
                         sentinel.size(), false, true, 7) == replay::expansion_collision);
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

void test_completed_frame_work() {
  replay::GpuFrameWork work;
  work.frame_boundary(2);
  work.retire(1, 4);
  CHECK(!work.latest());
  work.retire(2, 6);
  CHECK(work.latest() == 10);
  // A marker can follow feedback; the next frame includes both submissions.
  work.retire(3, 7);
  work.retire(4, 11);
  work.frame_boundary(4);
  CHECK(work.latest() == 18);
  for (unsigned i = 0; i < 15; ++i) work.frame_boundary(4);
  CHECK(work.latest() == 18);
  work.frame_boundary(4);
  CHECK(!work.latest());

  work.reset();
  work.frame_boundary(1);
  work.frame_boundary(3);
  work.retire(1, 5);
  CHECK(work.latest() == 5);
  work.retire(2, 7);
  CHECK(work.latest() == 5);
  work.retire(3, 11);
  CHECK(work.latest() == 18);

  work.reset();
  work.frame_boundary(2);
  work.retire(1, 4);
  work.retire(2, std::nullopt);
  CHECK(!work.latest());
  work.frame_boundary(3);
  work.retire(3, 9);
  CHECK(work.latest() == 9);
  work.retire(4, 0);
  work.frame_boundary(4);
  CHECK(work.latest().has_value() && *work.latest() == 0);

  work.reset();
  for (unsigned i = 0; i < 65; ++i) work.frame_boundary(1);
  work.retire(1, 7);
  CHECK(!work.latest());
  work.frame_boundary(2);
  work.retire(2, 9);
  CHECK(work.latest() == 9);

  work.reset(42);
  work.frame_boundary(42);
  CHECK(!work.latest());
  work.retire(43, 8);
  work.frame_boundary(43);
  CHECK(work.latest() == 8);

  // Collection enabled after earlier work in this frame retired. Whether
  // its marker precedes or follows the remaining retirement, discard that
  // partial frame and publish only the next fully observed frame.
  for (bool marker_first : {false, true}) {
    work.reset(42);
    if (marker_first) work.frame_boundary(43);
    work.retire(43, 3);
    if (!marker_first) work.frame_boundary(43);
    CHECK(!work.latest());
    work.frame_boundary(45);
    work.retire(44, 4);
    CHECK(!work.latest());
    work.retire(45, 6);
    CHECK(work.latest() == 10);
  }

  work.reset();
  work.frame_boundary(1);
  work.frame_boundary(2);
  work.retire(3, 10); // Missing retirement prefixes cannot become a zero-cost frame.
  work.frame_boundary(3);
  CHECK(!work.latest());
  work.retire(4, 9);
  work.frame_boundary(4);
  CHECK(work.latest() == 9);
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

// A submission can report more timestamps than one heap holds. Keep every
// result distinct while six slots are in flight, then reuse the oldest slot.
void test_timestamp_reports_across_heap_pages(id<MTLDevice> device,
                                             id<MTL4CommandQueue> queue) {
  constexpr std::size_t reports = 2049;
  constexpr std::size_t submissions = 7;
  SharedResources resources(device,
                            reports * submissions * sizeof(NVNcounterData));
  std::memset(resources.memory.contents, 0, resources.memory.length);
  auto backend = replay::create_replay_backend(device, queue, resources);
  id<MTLSharedEvent> gate = [device newSharedEvent];
  [queue waitForEvent:gate value:1];
  std::array<Recorder, submissions> recorders;
  std::array<NVNcommandHandle, submissions> handles{};
  std::uint64_t last = 0;
  for (std::size_t submission = 0; submission != submissions; ++submission) {
    auto& recorder = recorders[submission];
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    for (std::size_t report = 0; report != reports; ++report) {
      const auto offset = (submission * reports + report) * sizeof(NVNcounterData);
      load<Entry::nvnCommandBufferReportCounter>()(
          &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP,
          SharedResources::base + offset);
    }
    handles[submission] = recorder.finish();
    last = backend->submit({&handles[submission], 1});
    CHECK(last != 0);
  }
  CHECK(eventually([&] { return backend->statistics().submissions == 6; }));
  CHECK(backend->completed_fence() == 0);
  CHECK(read_at<NVNcounterData>(resources, 0).timestamp == 0);
  gate.signaledValue = 1;
  CHECK(backend->wait_fence(last, 10'000'000'000ull));
  CHECK(!backend->last_error());
  std::uint64_t previous = 0;
  bool ordered = true;
  for (std::size_t index = 0; index != reports * submissions; ++index) {
    const auto value = read_at<NVNcounterData>(resources,
                                             index * sizeof(NVNcounterData));
    ordered = ordered && value.counter == 0 && value.timestamp != 0 &&
              value.timestamp >= previous;
    previous = value.timestamp;
  }
  CHECK(ordered);
  CHECK(backend->statistics().counter_timestamp_fallbacks == 0);
  for (std::size_t index = 0; index != submissions; ++index) {
    recorders[index].release(handles[index]);
    recorders[index].finalize();
  }
}

// Exceeding the bounded storage must report an allocation failure, preserve
// the submitted prefix, and leave the rejected report and following work alone.
void test_timestamp_report_limit(id<MTLDevice> device,
                                 id<MTL4CommandQueue> queue) {
  constexpr std::size_t limit = 4096;
  SharedResources resources(device, 64);
  std::memset(resources.memory.contents, 0, resources.memory.length);
  auto backend = replay::create_replay_backend(device, queue, resources);
  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferReportCounter>()(
      &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, SharedResources::base);
  const auto prefix = recorder.finish();
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferReportCounter>()(
      &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, SharedResources::base + 16);
  load<Entry::nvnCommandBufferReportValue>()(
      &recorder.commands, 77, SharedResources::base + 32);
  const auto rejected = recorder.finish();
  std::vector<NVNcommandHandle> handles(limit, prefix);
  handles.push_back(rejected);
  const auto fence = backend->submit(handles);
  CHECK(fence != 0);
  CHECK(backend->wait_fence(fence, 10'000'000'000ull));
  const auto error = backend->last_error();
  CHECK(error.code == replay::ReplayErrorCode::allocation);
  CHECK(error.detail.find("4096") != std::string::npos);
  CHECK(read_at<NVNcounterData>(resources, 0).timestamp != 0);
  CHECK(read_at<NVNcounterData>(resources, 16).timestamp == 0);
  CHECK(read_at<std::uint32_t>(resources, 32) == 0);
  recorder.release(prefix);
  recorder.release(rejected);
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
  // Put the two copied reports on opposite sides of a heap-page boundary.
  for (std::size_t index = 0; index != 1023; ++index)
    load<Entry::nvnCommandBufferReportCounter>()(
        &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, SharedResources::base);
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

void test_gpu_frame_work_feedback(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
  SharedResources resources(device, 2 * 1024 * 1024);
  auto backend = replay::create_replay_backend(device, queue, resources);
  CHECK(backend != nullptr);
  if (!backend) return;
  CHECK(backend->gpu_frame_work_ns() == 0);
  Recorder recorder;
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferClearBuffer>()(
      &recorder.commands, SharedResources::base, resources.memory.length, 0x4172a39c);
  const auto handle = recorder.finish();
  const auto fence = backend->submit({&handle, 1});
  backend->finish_frame(0);
  CHECK(fence && backend->wait_fence(fence, 2'000'000'000ull));
  // The replay worker's frame marker can follow the GPU completion callback.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  std::uint64_t work = 0;
  while (!(work = backend->gpu_frame_work_ns()) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(work > 0);
  CHECK(read_at<std::uint32_t>(resources, 0) == 0x4172a39c);
  CHECK(read_at<std::uint32_t>(resources, resources.memory.length - 4) == 0x4172a39c);
  recorder.release(handle);
  recorder.finalize();
}

} // namespace

int main() {
  test_segment_lifecycle();
  test_primitive_expansion();
  test_completed_frame_work();
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
      test_timestamp_reports_across_heap_pages(device, queue);
      test_timestamp_report_limit(device, queue);
      test_feedback_during_encoder_split(device, queue);
      test_report_value_precedes_gpu_copy(device, queue);
      test_event_orders_gpu_work_in_one_stream(device, queue);
      test_timestamp_reports_precede_gpu_copies(device, queue);
      test_rejected_submit_does_not_leave_fence_gap(device, queue);
      test_destructor_drains_pending_callbacks(device);
      test_gpu_frame_work_feedback(device, queue);
    }
  }
  std::printf("Metal NVN replay ordering failures: %u\n", failures);
  return failures ? 1 : 0;
}
