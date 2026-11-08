// ============================================================================
// Armigo NVN - Windows and VI Presentation
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
#include "present.h"
#include "present_internal.h"
#include "armigo/nvn/timeline.h"
#include <algorithm>
#include <armigo/horizon/event.h>
#include <armigo/graphics/presentation_hints.h>
#include <atomic>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace armigo::nvn {
namespace {
constexpr auto window_tag = object_tag('W', 'I', 'N', 'D');
constexpr auto sync_tag = object_tag('S', 'Y', 'N', 'C');
struct WindowObject {
  WindowState state;
  std::uint64_t serial;
};
static_assert(sizeof(WindowObject) <= sizeof(NVNwindow));
struct WindowHost {
  struct Backing { std::uint64_t address = 0, size = 0; };
  std::array<Backing, 4> backings{};
  std::mutex mutex;
  bool active = true;
  std::uint32_t width = 0, height = 0;
  std::shared_ptr<services::vi::NativeWindowProducer> producer;
};
std::mutex registry_mutex;
std::unordered_map<std::uint64_t, std::shared_ptr<WindowHost>> windows;
std::atomic<std::uint64_t> next_serial{1};
TextureExporter texture_exporter;

template <class T, class U> T &state(U *object) {
  return *reinterpret_cast<T *>(object);
}
template <class T, class U> const T &state(const U *object) {
  return *reinterpret_cast<const T *>(object);
}
void diagnostic(const char *text) {
  armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN window: %s\n", text);
}
std::shared_ptr<WindowHost> host(const NVNwindow *window) {
  if (!window || state<WindowObject>(window).state.tag != window_tag)
    return {};
  const std::lock_guard lock(registry_mutex);
  const auto found = windows.find(state<WindowObject>(window).serial);
  return found == windows.end() ? nullptr : found->second;
}
void defaults(NVNwindowBuilder *builder) {
  auto &b = state<WindowBuilderState>(builder);
  auto *device = b.device;
  b = {};
  b.device = device;
  b.present_interval = 1;
}
void set_device(NVNwindowBuilder *builder, NVNdevice *device) {
  state<WindowBuilderState>(builder).device = device;
}
void set_native(NVNwindowBuilder *builder, NVNnativeWindow window) {
  state<WindowBuilderState>(builder).native_window = window;
}
void set_textures(NVNwindowBuilder *builder, int count,
                  NVNtexture *const *textures) {
  auto &b = state<WindowBuilderState>(builder);
  std::fill(std::begin(b.textures), std::end(b.textures), nullptr);
  b.texture_count = 0;
  if (count < 2 || count > 4 || !textures) {
    diagnostic("Invalid window texture count");
    return;
  }
  b.texture_count = count;
  std::copy_n(textures, count, b.textures);
}
void builder_interval(NVNwindowBuilder *builder, int interval) {
  state<WindowBuilderState>(builder).present_interval = interval;
}
NVNnativeWindow builder_native(const NVNwindowBuilder *builder) {
  return state<WindowBuilderState>(builder).native_window;
}
int get_builder_interval(const NVNwindowBuilder *builder) {
  return state<WindowBuilderState>(builder).present_interval;
}
NVNboolean initialize(NVNwindow *window, const NVNwindowBuilder *builder) {
  if (!window || !builder)
    return 0;
  const auto &b = state<WindowBuilderState>(builder);
  if (!b.device || b.texture_count < 2 || b.texture_count > 4 ||
      b.present_interval < 0 || b.present_interval > 5)
    return 0;
  auto producer = services::vi::find_native_window(b.native_window);
  std::uint64_t pending_serial = 0;
  try {
    TextureExporter exporter;
    {
      const std::lock_guard lock(registry_mutex);
      exporter = texture_exporter;
    }
    if (!producer || !exporter) {
      diagnostic("Native window binding or texture exporter is missing");
      return 0;
    }
    std::array<services::vi::NativeTexture, 4> exports;
    std::array<WindowHost::Backing, 4> backings{};
    for (int i = 0; i < b.texture_count; ++i) {
      if (!b.textures[i])
        return 0;
      const auto &texture = state<TextureState>(b.textures[i]);
      if (!texture.serial || !texture.description.width ||
          !texture.description.height)
        return 0;
      auto exported = exporter(texture);
      if (!exported || exported->serial != texture.serial ||
          exported->width != texture.description.width ||
          exported->height != texture.description.height)
        return 0;
      exports[i] = std::move(*exported);
      backings[i] = {reinterpret_cast<std::uint64_t>(texture.memory), texture.size};
      if (i && (exports[i].width != exports[0].width ||
                exports[i].height != exports[0].height))
        return 0;
    }
    auto instance = std::make_shared<WindowHost>();
    instance->backings = backings;
    instance->producer = producer;
    instance->width = exports[0].width;
    instance->height = exports[0].height;
    const auto serial = next_serial.fetch_add(1);
    pending_serial = serial;
    // Allocate the host entry before acquiring VI ownership so allocation
    // failure cannot leak slots.
    {
      const std::lock_guard lock(registry_mutex);
      windows.emplace(serial, instance);
    }
    if (!producer->register_textures(
            std::span(exports).first(b.texture_count))) {
      const std::lock_guard lock(registry_mutex);
      windows.erase(serial);
      return 0;
    }
    auto &out = state<WindowObject>(window);
    out = {};
    out.serial = serial;
    out.state.tag = window_tag;
    out.state.native_window = b.native_window;
    out.state.texture_count = b.texture_count;
    out.state.present_interval = b.present_interval;
    std::copy_n(b.textures, b.texture_count, out.state.textures);
    return 1;
  } catch (...) {
    {
      const std::lock_guard lock(registry_mutex);
      windows.erase(pending_serial);
    }
    diagnostic("Window registration failed");
    return 0;
  }
}
void finalize(NVNwindow *window) {
  const auto instance = host(window);
  if (!instance)
    return;
  const std::lock_guard lock(instance->mutex);
  if (!instance->producer->unregister_textures()) {
    diagnostic("Window still has consumer-owned textures");
    return;
  }
  instance->active = false;
  const std::lock_guard registry_lock(registry_mutex);
  windows.erase(state<WindowObject>(window).serial);
  if (windows.empty()) {
    // Restore ordinary scheduling after the queued frames have been processed.
    if (backends().replay) backends().replay->finish_frame(0);
    graphics::set_maximum_presentation_rate(0);
  }
  state<WindowObject>(window) = {};
}
void label(NVNwindow *, const char *) {}
NVNwindowAcquireTextureResult acquire(NVNwindow *window, NVNsync *sync,
                                      int *index) {
  const auto instance = host(window);
  if (!instance || !index || (sync && state<SyncState>(sync).tag != sync_tag))
    return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
  std::uint64_t period = 0;
  {
    const std::lock_guard lock(instance->mutex);
    if (!instance->active) return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
    const auto interval = state<WindowState>(window).present_interval;
    const auto scheduling = graphics::presentation_scheduling(
        instance->producer->scheduling_period(interval),
        instance->producer->guest_present_period(interval));
    period = static_cast<std::uint64_t>(scheduling.period.count());
    if (backends().replay)
      backends().replay->set_scheduling_period(period, scheduling.guest_cadence);
  }
  // A policy change can take a while; present and finalize must not wait on it.
  if (backends().replay) backends().replay->acquire_tick(period);
  // A concurrent present may release the slot; do not hold the window lock
  // during the wait.
  for (;;) {
    services::vi::NativeDequeue dequeued;
    {
      const std::lock_guard lock(instance->mutex);
      if (!instance->active)
        return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
      dequeued = instance->producer->dequeue();
    }
    if (dequeued.slot >= 0) {
      *index = dequeued.slot;
      if (sync) {
        auto &s = state<SyncState>(sync);
        s.fenced = 1;
        s.fence = dequeued.fence;
      }
      return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS;
    }
    if (dequeued.slot != -1)
      return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
    const auto event = instance->producer->free_event();
    if (!event)
      return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
    const bool waited = blocking_wait([&] {
      horizon::HandleTable handles(16);
      const auto handle = handles.insert(event);
      if (!handle)
        return false;
      const std::array ids{*handle};
      const auto result = horizon::wait_event(handles, ids, -1);
      return result.disposition == horizon::EventDisposition::handled &&
             result.result == 0;
    }, "window_acquire");
    if (!waited)
      return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
  }
}
NVNnativeWindow get_native(const NVNwindow *window) {
  return state<WindowState>(window).native_window;
}
int get_interval(const NVNwindow *window) {
  return state<WindowState>(window).present_interval;
}
void set_interval(NVNwindow *window, int interval) {
  const auto instance = host(window);
  if (!instance || interval < 0 || interval > 5) {
    diagnostic("Invalid present interval");
    return;
  }
  const std::lock_guard lock(instance->mutex);
  state<WindowState>(window).present_interval = interval;
}
void set_crop(NVNwindow *window, int x, int y, int width, int height) {
  const auto instance = host(window);
  if (!instance)
    return;
  if (x < 0 || y < 0 || width < 0 || height < 0 || x > INT32_MAX - width ||
      y > INT32_MAX - height) {
    diagnostic("Invalid crop");
    return;
  }
  const std::lock_guard lock(instance->mutex);
  auto &w = state<WindowState>(window);
  if (std::uint64_t(x) + width > instance->width ||
      std::uint64_t(y) + height > instance->height) {
    diagnostic("Crop exceeds window texture");
    return;
  }
  w.crop[0] = x;
  w.crop[1] = y;
  w.crop[2] = width;
  w.crop[3] = height;
}
void get_crop(const NVNwindow *window, NVNrectangle *out) {
  if (!out)
    return;
  const auto &w = state<WindowState>(window);
  *out = {w.crop[0], w.crop[1], w.crop[2], w.crop[3]};
}
void present(NVNqueue *queue, NVNwindow *window, int index) {
  const auto instance = host(window);
  if (!instance || !queue_valid(queue) || !backends().replay) {
    queue_error(queue, "Presentation dependencies are missing");
    return;
  }
  std::array<std::int32_t, 4> crop;
  int interval;
  {
    const std::lock_guard lock(instance->mutex);
    const auto &w = state<WindowState>(window);
    if (!instance->active || index < 0 || index >= w.texture_count) {
      queue_error(queue, "Present texture index is out of range");
      return;
    }
    crop = {w.crop[0], w.crop[1], w.crop[0] + w.crop[2], w.crop[1] + w.crop[3]};
    interval = w.present_interval;
  }
  auto *replay = backends().replay;
  services::GraphicsPresentationResult result;
  try {
    replay->flush();
    const auto fence = queue_last_fence(queue);
    timeline_event("present_call", fence, std::uint64_t(index));
    const auto completed = blocking_wait([&] {
      result = instance->producer->present(
          index, fence,
          [replay](std::uint64_t value) {
            return replay->wait_fence(value, UINT64_MAX);
          },
          crop, interval);
      return result.completed;
    }, "present");
    timeline_event("present_queued", fence, result.completed ? 1u : 0u);
    if (!completed) {
      queue_error(queue, result.diagnostic.empty() ? "Presentation failed"
                                                   : result.diagnostic.c_str());
      return;
    }
    const std::lock_guard lock(instance->mutex);
    if (instance->active) {
      auto& window_state = state<WindowState>(window);
      ++window_state.presents;
      const auto scheduling = graphics::presentation_scheduling(
          instance->producer->scheduling_period(window_state.present_interval),
          instance->producer->guest_present_period(window_state.present_interval));
      replay->finish_frame(static_cast<std::uint64_t>(scheduling.period.count()),
                           scheduling.guest_cadence);
    }
  } catch (...) {
    queue_error(queue, "Presentation backend raised an exception");
  }
}
NVNqueueAcquireTextureResult queue_acquire(NVNqueue *queue, NVNwindow *window,
                                           int *index) {
  if (!queue_valid(queue) || !backends().replay)
    return NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
  alignas(8) NVNsync sync{};
  state<SyncState>(&sync) = {sync_tag, 0, 0};
  if (acquire(window, &sync, index) !=
      NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS)
    return NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
  const auto fence = state<SyncState>(&sync).fence;
  if (fence && !blocking_wait([&] {
        return backends().replay->wait_fence(fence, UINT64_MAX);
      }, "acquire_sync"))
    return NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_NATIVE_ERROR;
  return NVN_QUEUE_ACQUIRE_TEXTURE_RESULT_SUCCESS;
}
} // namespace

void set_texture_exporter(TextureExporter exporter) {
  const std::lock_guard lock(registry_mutex);
  texture_exporter = std::move(exporter);
}
bool window_texture_overlaps(std::uint64_t address, std::uint64_t size) {
  if (!size || size > UINT64_MAX - address) return true;
  const auto end = address + size;
  const std::lock_guard lock(registry_mutex);
  for (const auto& [serial, window] : windows) {
    static_cast<void>(serial);
    for (const auto& backing : window->backings)
      if (backing.address && backing.size &&
          (backing.size > UINT64_MAX - backing.address ||
           (address < backing.address + backing.size && backing.address < end)))
        return true;
  }
  return false;
}
bool set_window_active_textures(NVNwindow *window, int count) {
  const auto instance = host(window);
  if (!instance || count < 2 || count > state<WindowState>(window).texture_count)
    return false;
  const std::lock_guard lock(instance->mutex);
  return instance->active &&
         instance->producer->set_active_textures(static_cast<std::size_t>(count));
}
void register_present_entries() {
  register_queue_entries();
  register_sync_entries();
  install<Entry::nvnWindowBuilderSetDefaults>(defaults);
  install<Entry::nvnWindowBuilderSetDevice>(set_device);
  install<Entry::nvnWindowBuilderSetNativeWindow>(set_native);
  install<Entry::nvnWindowBuilderSetTextures>(set_textures);
  install<Entry::nvnWindowBuilderSetPresentInterval>(builder_interval);
  install<Entry::nvnWindowBuilderGetNativeWindow>(builder_native);
  install<Entry::nvnWindowBuilderGetPresentInterval>(get_builder_interval);
  install<Entry::nvnWindowInitialize>(initialize);
  install<Entry::nvnWindowFinalize>(finalize);
  install<Entry::nvnWindowSetDebugLabel>(label);
  install<Entry::nvnWindowAcquireTexture>(acquire);
  install<Entry::nvnWindowGetNativeWindow>(get_native);
  install<Entry::nvnWindowGetPresentInterval>(get_interval);
  install<Entry::nvnWindowSetPresentInterval>(set_interval);
  install<Entry::nvnWindowSetCrop>(set_crop);
  install<Entry::nvnWindowGetCrop>(get_crop);
  install<Entry::nvnQueuePresentTexture>(present);
  install<Entry::nvnQueueAcquireTexture>(queue_acquire);
}
} // namespace armigo::nvn
