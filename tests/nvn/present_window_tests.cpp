// ============================================================================
// Armigo Tests - NVN Window Presentation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "backend.h"
#include "entry_table.h"
#include "present.h"
#include "present_internal.h"
#include <armigo/services/video_interface.h>
#include <armigo/graphics/presentation_hints.h>
#include <atomic>
#include <chrono>
#include <vector>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

using namespace armigo::nvn;
using namespace std::chrono_literals;
namespace {
unsigned failures;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);             \
      ++failures;                                                              \
    }                                                                          \
  } while (false)
template <Entry E, class... A> auto call(A... args) {
  const auto name = entry_name(E);
  return reinterpret_cast<typename EntrySignature<E>::type>(
      armigo_nvn_bootstrap_loader(name.data()))(args...);
}
struct ThrowingReader {
  ThrowingReader() = default;
  ThrowingReader(ThrowingReader &&) noexcept = default;
  ThrowingReader(const ThrowingReader &) {
    throw std::runtime_error("Synthetic reader copy failure");
  }
  armigo::graphics::Rgba8ReadbackResult
  operator()(std::array<std::int32_t, 4>) const {
    return armigo::graphics::Rgba8Image{};
  }
};
class Replay final : public ReplayBackend {
public:
  std::atomic<std::uint64_t> submitted{0}, completed{0};
  std::atomic<unsigned> waits{0};
  std::atomic<std::uint64_t> scheduling_period{0};
  std::atomic<bool> preserve_encode_spin{false};
  void set_scheduling_period(std::uint64_t period_ns, bool preserve_spin = false) override {
    scheduling_period = period_ns;
    preserve_encode_spin = preserve_spin;
  }
  std::mutex mutex;
  std::condition_variable changed;
  std::uint64_t submit(std::span<const NVNcommandHandle>) override {
    return ++submitted;
  }
  // The queue identity the latest submission named, and the latest one
  // finalized.
  std::atomic<std::uintptr_t> submitted_queue{0}, forgotten_queue{0};
  std::uint64_t submit_from(std::uintptr_t queue, std::span<const NVNcommandHandle> handles) override {
    submitted_queue = queue;
    return submit(handles);
  }
  void forget_queue(std::uintptr_t queue) override { forgotten_queue = queue; }
  // A queued wait takes the next fence and leaves it incomplete.
  std::optional<std::uint64_t> queue_wait_fence(std::uintptr_t, std::uint64_t) override { return ++submitted; }
  bool snapshot_enabled=false;
  std::shared_ptr<std::string> native_result;
  std::optional<NativeProducerFence> snapshot_native_producer(std::uintptr_t) override {
    if(!snapshot_enabled)return std::nullopt;
    auto result=native_result;
    return NativeProducerFence{submitted.load(),[result]{return *result;}};
  }
  void flush() override {}
  std::uint64_t completed_fence() const override { return completed; }
  bool wait_fence(std::uint64_t fence, std::uint64_t timeout) override {
    ++waits;
    std::unique_lock lock(mutex);
    if (timeout == UINT64_MAX) {
      changed.wait(lock, [&] { return completed >= fence; });
      return true;
    }
    return changed.wait_for(lock, std::chrono::nanoseconds(timeout),
                            [&] { return completed >= fence; });
  }
  void complete() {
    {
      const std::lock_guard lock(mutex);
      completed = submitted.load();
    }
    changed.notify_all();
  }
};
} // namespace
namespace {
// Presents complete on the presentation thread: the guest is told when the
// frame is queued, so a test waits for the frame to be shown.
template <class Predicate>
bool wait_for_condition(Predicate predicate,
                        std::chrono::milliseconds limit = std::chrono::seconds(2)) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

void test_live_refresh_period_queries() {
  using namespace armigo;
  auto guest_result = runtime::create_guest_clock();
  CHECK(std::holds_alternative<std::shared_ptr<runtime::GuestClock>>(guest_result));
  if (!std::holds_alternative<std::shared_ptr<runtime::GuestClock>>(guest_result)) return;
  auto guest = std::get<std::shared_ptr<runtime::GuestClock>>(guest_result);
  runtime::PresentationClockConfiguration configuration;
  configuration.refresh_period = 16'666'666ns;
  auto clock_result = runtime::create_presentation_clock(guest, configuration);
  CHECK(std::holds_alternative<std::shared_ptr<runtime::PresentationClock>>(clock_result));
  if (!std::holds_alternative<std::shared_ptr<runtime::PresentationClock>>(clock_result)) return;
  auto clock = std::get<std::shared_ptr<runtime::PresentationClock>>(clock_result);
  auto source = std::make_shared<services::vi::VsyncSource>(clock);
  horizon::HandleTable handles(16);
  services::vi::DisplaySession display(handles);
  display.set_vsync_source(source);
  int token;
  CHECK(display.bind_native_window(&token));
  auto producer = services::vi::find_native_window(&token);
  CHECK(producer);
  if (!producer) return;
  CHECK(producer->scheduling_period(2) == 33'333'332ns);
  CHECK(producer->guest_present_period(2) == 33'333'332ns);
  CHECK(clock->set_refresh_period(33'333'333ns));
  CHECK(source->interval() == 33'333'333ns);
  CHECK(producer->scheduling_period(1) == 33'333'333ns);
  CHECK(producer->scheduling_period(2) == 66'666'666ns);
  setenv("ARMIGO_VI_PACE", "0", 1);
  CHECK(producer->scheduling_period(2) == 0ns);
  CHECK(producer->guest_present_period(1) == 33'333'333ns);
  CHECK(producer->guest_present_period(2) == 66'666'666ns);
  CHECK(clock->set_refresh_period(16'666'666ns));
  CHECK(producer->guest_present_period(2) == 33'333'332ns);
  unsetenv("ARMIGO_VI_PACE");
  services::vi::unbind_native_window(&token);
}
} // namespace

void test_window_snapshot_capture_gap(bool failed) {
  select_api_version({55,15});register_present_entries();
  Replay replay;replay.snapshot_enabled=true;replay.submitted=1;replay.completed=1;
  const std::string expected=failed?"Exact producer failure":"";
  replay.native_result=std::make_shared<std::string>(expected);
  set_backends({nullptr,&replay});
  armigo::horizon::HandleTable handles(16);
  armigo::services::vi::DisplaySession display(handles);
  display.set_vsync_source(std::make_shared<armigo::services::vi::VsyncSource>(2ms));
  int token;CHECK(display.bind_native_window(&token));
  std::atomic<bool> pin_live{false},captured{false};
  std::atomic<std::uint64_t> selected{0};
  display.set_presenter([](const auto&){return armigo::services::GraphicsPresentationResult{true,{}};});
  alignas(8) NVNdevice device{};alignas(8) NVNqueueBuilder qb{};alignas(8) NVNqueue queue{};
  call<Entry::nvnQueueBuilderSetDefaults>(&qb);call<Entry::nvnQueueBuilderSetDevice>(&qb,&device);
  CHECK(call<Entry::nvnQueueInitialize>(&queue,&qb));queue_set_last_fence(&queue,1);
  set_texture_exporter([&](const TextureState& state)->std::optional<armigo::services::vi::NativeTexture> {
    armigo::services::vi::NativeTexture out;out.serial=state.serial;out.width=out.height=4;
    out.read_pixels=[](auto)->armigo::graphics::Rgba8ReadbackResult{return armigo::graphics::Rgba8Image{};};
    out.capture=[&,base=out](std::uint64_t fence)->std::optional<armigo::services::vi::NativeTexture> {
      selected=fence;std::weak_ptr<std::string> old=replay.native_result;
      replay.native_result=std::make_shared<std::string>("Successor result");replay.submitted=2;replay.completed=2;queue_set_last_fence(&queue,2);
      auto pinned=old.lock();pin_live=pinned && *pinned==expected;captured=true;
      return base;
    };return out;
  });
  alignas(8) NVNtexture textures[2]{};NVNtexture* pointers[2]{&textures[0],&textures[1]};
  for(unsigned i=0;i!=2;++i){auto& t=*reinterpret_cast<TextureState*>(pointers[i]);t.tag=object_tag('T','E','X','R');t.serial=901+i;t.description.width=t.description.height=4;}
  alignas(8) NVNwindowBuilder builder{};alignas(8) NVNwindow window{};
  call<Entry::nvnWindowBuilderSetDefaults>(&builder);call<Entry::nvnWindowBuilderSetDevice>(&builder,&device);
  call<Entry::nvnWindowBuilderSetNativeWindow>(&builder,&token);call<Entry::nvnWindowBuilderSetTextures>(&builder,2,pointers);call<Entry::nvnWindowBuilderSetPresentInterval>(&builder,0);
  CHECK(call<Entry::nvnWindowInitialize>(&window,&builder));int slot=-1;
  CHECK(call<Entry::nvnWindowAcquireTexture>(&window,static_cast<NVNsync*>(nullptr),&slot)==NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS);
  call<Entry::nvnQueuePresentTexture>(&queue,&window,slot);
  call<Entry::nvnWindowFinalize>(&window);
  std::fprintf(stderr,"window_snapshot_gap selected=%llu captured=%u pin_live=%u\n",static_cast<unsigned long long>(selected.load()),unsigned(captured.load()),unsigned(pin_live.load()));
  CHECK(captured && selected==1 && pin_live);
  call<Entry::nvnQueueFinalize>(&queue);armigo::services::vi::unbind_native_window(&token);set_texture_exporter({});set_backends({});
}

int main() {
  test_window_snapshot_capture_gap(false);
  test_window_snapshot_capture_gap(true);
  test_live_refresh_period_queries();
  select_api_version({55, 15});
  register_present_entries();
  Replay replay;
  set_backends({nullptr, &replay});
  armigo::horizon::HandleTable handles(16);
  armigo::services::vi::DisplaySession display(handles);
  auto vsync = std::make_shared<armigo::services::vi::VsyncSource>(2ms);
  display.set_vsync_source(vsync);
  int native_token;
  CHECK(display.bind_native_window(&native_token));
  std::atomic<unsigned> consumed{0}, reads{0};
  std::atomic<bool> consumer_error{false};
  std::mutex times_mutex;
  std::vector<std::chrono::steady_clock::time_point> presented_at;
  std::uint64_t observed_serial = 0;
  std::array<std::int32_t, 4> observed_crop{};
  display.set_presenter(
      [&](const armigo::services::GraphicsPresentation &frame) {
#if defined(__APPLE__)
        if (consumed.load() == 0) {
          qos_class_t qos = QOS_CLASS_UNSPECIFIED;
          int priority = 0;
          CHECK(pthread_get_qos_class_np(pthread_self(), &qos, &priority) == 0);
          CHECK(qos == QOS_CLASS_USER_INTERACTIVE);
        }
#endif
        // The frame arrives as a native texture with a lazy readback: an
        // ordinary present hands over the GPU frame without reading it back.
        if (replay.completed < replay.submitted ||
            (!frame.native_pixels && !frame.native_read_pixels) ||
            !frame.native_vsync_paced)
          consumer_error = true;
        observed_serial = frame.native_texture_serial;
        observed_crop = frame.crop;
        {
          const std::lock_guard lock(times_mutex);
          presented_at.push_back(std::chrono::steady_clock::now());
        }
        ++consumed;
        return armigo::services::GraphicsPresentationResult{true, {}};
      });
  set_texture_exporter(
      [&](const TextureState &texture)
          -> std::optional<armigo::services::vi::NativeTexture> {
        armigo::services::vi::NativeTexture out;
        out.serial = texture.serial;
        out.width = 4;
        out.height = 4;
        out.read_pixels = [&](std::array<std::int32_t, 4> crop)
            -> armigo::graphics::Rgba8ReadbackResult {
          ++reads;
          if (replay.completed < replay.submitted)
            consumer_error = true;
          if (crop == std::array<std::int32_t, 4>{})
            crop = {0, 0, 4, 4};
          armigo::graphics::Rgba8Image image;
          image.width = crop[2] - crop[0];
          image.height = crop[3] - crop[1];
          image.pixels.assign(image.width * image.height * 4, std::byte{0x7f});
          return image;
        };
        return out;
      });
  alignas(8) NVNdevice device{};
  alignas(8) NVNqueueBuilder qb{};
  alignas(8) NVNqueue queue{};
  call<Entry::nvnQueueBuilderSetDefaults>(&qb);
  call<Entry::nvnQueueBuilderSetDevice>(&qb, &device);
  CHECK(call<Entry::nvnQueueInitialize>(&queue, &qb));
  alignas(8) NVNtexture textures[3]{};
  NVNtexture *pointers[3];
  for (int i = 0; i < 3; ++i) {
    auto &texture = *reinterpret_cast<TextureState *>(&textures[i]);
    texture.tag = object_tag('T', 'E', 'X', 'R');
    texture.serial = i + 11;
    texture.description.width = 4;
    texture.description.height = 4;
    texture.memory = reinterpret_cast<std::uint8_t*>(0x10000 + i * 0x1000);
    texture.size = 64;
    pointers[i] = &textures[i];
  }
  alignas(8) NVNwindowBuilder builder{};
  alignas(8) NVNwindow window{};
  alignas(8) NVNsync sync{};
  CHECK(call<Entry::nvnSyncInitialize>(&sync, &device));
  call<Entry::nvnWindowBuilderSetDevice>(&builder, &device);
  call<Entry::nvnWindowBuilderSetDefaults>(&builder);
  call<Entry::nvnWindowBuilderSetNativeWindow>(&builder, &native_token);
  call<Entry::nvnWindowBuilderSetTextures>(&builder, 3, pointers);
  call<Entry::nvnWindowBuilderSetPresentInterval>(&builder, 0);
  CHECK(call<Entry::nvnWindowInitialize>(&window, &builder));
  CHECK(window_texture_overlaps(0x10000, 1));
  CHECK(window_texture_overlaps(0xffff, 2));
  CHECK(!window_texture_overlaps(0x10040, 1));
  CHECK(call<Entry::nvnWindowGetNativeWindow>(&window) == &native_token);
  CHECK(call<Entry::nvnWindowGetPresentInterval>(&window) == 0);
  armigo::graphics::set_maximum_presentation_rate(0);
  int slots[3]{-1, -1, -1};
  for (auto &slot : slots) {
    CHECK(call<Entry::nvnWindowAcquireTexture>(&window, &sync, &slot) ==
          NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS);
    if (&slot == &slots[0]) {
      CHECK(replay.scheduling_period.load() == 0);
      CHECK(!replay.preserve_encode_spin.load());
      // Bypassing host waits must retain the guest's two-refresh deadline.
      // An explicit maximum-rate hint still takes precedence over that fallback.
      call<Entry::nvnWindowSetPresentInterval>(&window, 2);
      setenv("ARMIGO_VI_PACE", "0", 1);
    } else if (&slot == &slots[1]) {
      CHECK(replay.scheduling_period.load() == 4'000'000);
      CHECK(replay.preserve_encode_spin.load());
      armigo::graphics::set_maximum_presentation_rate(1000);
    }
  }
  unsetenv("ARMIGO_VI_PACE");
  call<Entry::nvnWindowSetPresentInterval>(&window, 0);
  CHECK(replay.scheduling_period.load() == 1'000'000);
  CHECK(!replay.preserve_encode_spin.load());
  CHECK(slots[0] != slots[1] && slots[1] != slots[2] && slots[2] != slots[0]);
  CHECK(call<Entry::nvnSyncWait>(&sync, std::uint64_t{0}) ==
        NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED);
  auto producer = armigo::services::vi::find_native_window(&native_token);
  CHECK(producer && !producer->free_event()->is_signaled());
  CHECK(producer->guest_present_period(0) == 0ns);
  CHECK(producer->guest_present_period(-1) == 0ns);
  CHECK(producer->guest_present_period(6) == 0ns);
  CHECK(producer->guest_present_period(2) == 4ms);
  // A copied object locates host state by serial; guest object address is not
  // identity.
  auto copied = window;
  auto pending_acquire = std::async(std::launch::async, [&] {
    int slot = -1;
    const auto result = call<Entry::nvnWindowAcquireTexture>(
        &copied, static_cast<NVNsync *>(nullptr), &slot);
    return result == NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS ? slot : -1;
  });
  CHECK(pending_acquire.wait_for(10ms) == std::future_status::timeout);
  const NVNcommandHandle command = 1;
  call<Entry::nvnQueueSubmitCommands>(&queue, 1, &command);
  auto pending_present = std::async(std::launch::async, [&] {
    call<Entry::nvnQueuePresentTexture>(&queue, &window, slots[0]);
  });
  // The present returns to the guest as soon as the frame is queued; the frame
  // is presented on the presentation thread once its GPU work completed.
  CHECK(pending_present.wait_for(1s) == std::future_status::ready);
  CHECK(reads == 0 && consumed == 0);
  replay.complete();
  pending_present.get();
  CHECK(pending_acquire.wait_for(1s) == std::future_status::ready);
  CHECK(pending_acquire.get() == slots[0]);
  CHECK(wait_for_condition([&] { return consumed == 1; }));
  CHECK(!consumer_error && observed_serial == std::uint64_t(slots[0] + 11));
  const auto original_texture = textures[0];
  textures[0] = {};
  call<Entry::nvnWindowSetCrop>(&window, 1, 1, 2, 2);
  textures[0] = original_texture;
  NVNrectangle crop{};
  call<Entry::nvnWindowGetCrop>(&window, &crop);
  CHECK(crop.x == 1 && crop.y == 1 && crop.width == 2 && crop.height == 2);
  call<Entry::nvnWindowSetPresentInterval>(&window, 2);
  const auto before = vsync->sequence();
  auto presented = [&] { return consumed.load(); };
  const unsigned presentations = presented();
  call<Entry::nvnQueuePresentTexture>(&queue, &window, slots[1]);
  CHECK(wait_for_condition([&] { return presented() == presentations + 1; }));
  const auto first = vsync->sequence();
  call<Entry::nvnQueuePresentTexture>(&queue, &window, slots[2]);
  CHECK(wait_for_condition([&] { return presented() == presentations + 2; }));
  const auto second = vsync->sequence();
  // Without a compatibility patch the requested interval is honoured: two
  // presentations of a two-refresh interval are at least two refresh periods
  // apart (the source's period is 2 ms here).
  {
    const std::lock_guard lock(times_mutex);
    CHECK(presented_at.size() >= 3);
    const auto last = presented_at[presented_at.size() - 1];
    const auto previous = presented_at[presented_at.size() - 2];
    const auto gap = std::chrono::duration_cast<std::chrono::microseconds>(last - previous);
    const auto first_gap = std::chrono::duration_cast<std::chrono::microseconds>(
        presented_at[1] - presented_at[0]);
    const auto span = std::chrono::duration_cast<std::chrono::microseconds>(last - presented_at[0]);
    // Each frame occupies one two-refresh slot: two of them span at least the
    // two periods the requested interval asks for (the source is 2 ms here), and
    // neither frame is presented twice in the same slot.
    CHECK(span >= std::chrono::milliseconds(5));
    CHECK(gap >= std::chrono::milliseconds(1) && first_gap >= std::chrono::milliseconds(1));
    CHECK(second >= before);
  }
  CHECK((observed_crop == std::array<std::int32_t, 4>{1, 1, 3, 3}));
  CHECK(!consumer_error);
  int again = -1;
  CHECK(call<Entry::nvnQueueAcquireTexture>(&queue, &window, &again) ==
        NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_SUCCESS);
  CHECK(again == slots[1]);
  CHECK(replay.scheduling_period.load() == 4'000'000);
  // Finalize safely cancels producer-owned dequeues, but not consumer-owned
  // work.
  call<Entry::nvnWindowFinalize>(&window);
  CHECK(!window_texture_overlaps(0x10000, 1));
  CHECK(replay.scheduling_period.load() == 0);
  CHECK(armigo::graphics::presentation_scheduling_period(0ns) == 0ns);
  CHECK(call<Entry::nvnWindowAcquireTexture>(&copied, &sync, &again) ==
        NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR);
  CHECK(call<Entry::nvnWindowInitialize>(&window, &builder));
  int in_flight = -1;
  CHECK(call<Entry::nvnWindowAcquireTexture>(&window, &sync, &in_flight) ==
        NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS);
  call<Entry::nvnQueueSubmitCommands>(&queue, 1, &command);
  auto blocked_present = std::async(std::launch::async, [&] {
    call<Entry::nvnQueuePresentTexture>(&queue, &window, in_flight);
  });
  CHECK(blocked_present.wait_for(1s) == std::future_status::ready);
  auto free_acquire = std::async(std::launch::async, [&] {
    int index = -1;
    const auto result = call<Entry::nvnWindowAcquireTexture>(
        &window, static_cast<NVNsync *>(nullptr), &index);
    return result == NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS ? index : -1;
  });
  const bool free_was_ready =
      free_acquire.wait_for(100ms) == std::future_status::ready;
  // Release the fence even on regression so this test never strands a worker.
  replay.complete();
  blocked_present.get();
  CHECK(free_was_ready && free_acquire.get() != in_flight);
  call<Entry::nvnWindowFinalize>(&window);
  // A failed render fence must quarantine the slot and skip texture readback.
  CHECK(call<Entry::nvnWindowInitialize>(&window, &builder));
  auto failed_slot = producer->dequeue();
  const auto reads_before_failure = reads.load();
  // The frame is queued; its fence wait fails on the presentation thread, which
  // frees the slot without reading the frame back.
  CHECK(producer
            ->present(
                failed_slot.slot, 1, [](std::uint64_t) { return false; }, {}, 0)
            .completed);
  CHECK(producer->unregister_textures());
  CHECK(producer->dequeue().slot != -1);
  CHECK(reads == reads_before_failure);
  call<Entry::nvnWindowFinalize>(&window);
  CHECK(reinterpret_cast<WindowState *>(&window)->tag != 0);
  armigo::services::vi::unbind_native_window(&native_token);
  CHECK(!call<Entry::nvnWindowInitialize>(&window, &builder));
  // A throwing VI registration must roll back the unreachable host serial.
  armigo::services::vi::DisplaySession failure_display(handles);
  int failure_token;
  CHECK(failure_display.bind_native_window(&failure_token));
  std::weak_ptr<armigo::services::vi::NativeWindowProducer> weak_producer =
      armigo::services::vi::find_native_window(&failure_token);
  set_texture_exporter(
      [](const TextureState &texture)
          -> std::optional<armigo::services::vi::NativeTexture> {
        armigo::services::vi::NativeTexture exported;
        exported.serial = texture.serial;
        exported.width = 4;
        exported.height = 4;
        exported.read_pixels = ThrowingReader{};
        return exported;
      });
  call<Entry::nvnWindowBuilderSetNativeWindow>(&builder, &failure_token);
  CHECK(!call<Entry::nvnWindowInitialize>(&window, &builder));
  armigo::services::vi::unbind_native_window(&failure_token);
  CHECK(weak_producer.expired());
  // Replay knows a queue by an identity of each initialization: finalizing
  // names it to replay, and a queue initialized again in the same storage
  // submits under a new one.
  call<Entry::nvnQueueSubmitCommands>(&queue, 0, nullptr);
  const auto identity = replay.submitted_queue.load();
  // nvnQueueWaitSync becomes the queue's latest fence: a queue fence sync
  // after it takes the wait's fence, which is not complete yet.
  {
    alignas(8) NVNsync fenced{}, after{};
    CHECK(call<Entry::nvnSyncInitialize>(&fenced, &device) && call<Entry::nvnSyncInitialize>(&after, &device));
    auto &fenced_state = *reinterpret_cast<SyncState *>(&fenced);
    fenced_state.fenced = 1;
    fenced_state.fence = replay.submitted + 1000;
    CHECK(call<Entry::nvnQueueWaitSync>(&queue, &fenced) == 1);
    const auto wait_fence = replay.submitted.load();
    call<Entry::nvnQueueFenceSync>(&queue, &after, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
    CHECK(reinterpret_cast<const SyncState *>(&after)->fence == wait_fence);
    CHECK(call<Entry::nvnSyncWait>(&after, std::uint64_t{0}) == NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED);
    call<Entry::nvnSyncFinalize>(&fenced);
    call<Entry::nvnSyncFinalize>(&after);
  }
  CHECK(identity != 0);
  call<Entry::nvnQueueFinalize>(&queue);
  CHECK(replay.forgotten_queue == identity);
  CHECK(call<Entry::nvnQueueInitialize>(&queue, &qb));
  call<Entry::nvnQueueSubmitCommands>(&queue, 0, nullptr);
  CHECK(replay.submitted_queue != 0 && replay.submitted_queue != identity);
  call<Entry::nvnQueueFinalize>(&queue);
  call<Entry::nvnSyncFinalize>(&sync);
  set_texture_exporter({});
  set_backends({});
  std::printf("NVN window failures: %u\n", failures);
  return failures ? 1 : 0;
}
