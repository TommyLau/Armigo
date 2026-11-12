// ============================================================================
// Armigo Tests - NVN Metal Presentation Readback
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../../src/platform/metal/nvn_present.h"
#include "../../src/platform/metal/nvn_resources.h"
#include "../../src/platform/metal/nvn_replay.h"
#include "../../src/nvn/resource_layout.h"
#include <armigo/memory/write_tracker.h>
#include <atomic>
#include <dlfcn.h>
#include <stdexcept>
#include <armigo/common/diagnostics.h>
#include <cstring>
#import <Foundation/Foundation.h>
#include <armigo/runtime/presentation.h>
#include <armigo/services/video_interface.h>
#include <cstdio>
#include <future>

@interface PresentAutoreleaseSentinel : NSObject {
@public
  std::shared_ptr<std::atomic<unsigned>> counter;
}
@end
@implementation PresentAutoreleaseSentinel
- (void)dealloc { if (counter) counter->fetch_add(1); }
@end

namespace armigo::platform {
id<MTLTexture> presenter_test_native_target(const Presenter& presenter);
}

bool native_pixels_match(const std::shared_ptr<const void>& native,
                         std::array<std::byte, 4> expected, bool cropped = false) {
  @autoreleasepool {
    id<MTLTexture> source = (__bridge id<MTLTexture>)native.get();
    auto queue = [source.device newCommandQueue];
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat width:3 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    auto copy = [source.device newTextureWithDescriptor:descriptor];
    auto result = [source.device newBufferWithLength:256*4 options:MTLResourceStorageModeShared];
    auto commands = [queue commandBuffer]; auto blit = [commands blitCommandEncoder];
    if (!copy || !result || !blit) return false;
    [blit copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(cropped ? 0 : 1,cropped ? 0 : 2,0)
        sourceSize:MTLSizeMake(3,4,1) toTexture:copy destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [blit copyFromTexture:copy sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(3,4,1) toBuffer:result destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:1024];
    [blit endEncoding]; [commands commit]; [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted) return false;
    auto* bytes = static_cast<const std::byte*>(result.contents);
    for (unsigned y=0;y<4;++y) for(unsigned x=0;x<3;++x) for(unsigned c=0;c<4;++c)
      if (bytes[y*256+x*4+c] != expected[c]) return false;
    return true;
  }
}

// keep: the run keeps the frame's pixels. Otherwise the headless consumer
// copies the texture on the GPU and the observer sees no readback.
bool verify_consumer(const armigo::services::vi::NativeTexture &texture,
                     bool keep, std::function<void()> fence_ready = {},
                     std::array<std::byte, 4> expected = {
                         std::byte{255}, std::byte{}, std::byte{}, std::byte{255}},
                     const void* expected_native = nullptr,
                     std::function<void()> before_consume = {}, bool inspect_native = false,
                     std::function<void()> enter_consumer = {}) {
  using namespace armigo;
  auto clock_result = runtime::create_guest_clock();
  auto *clock =
      std::get_if<std::shared_ptr<runtime::GuestClock>>(&clock_result);
  if (!clock)
    return false;
  auto schedule_result = runtime::create_presentation_clock(
      *clock, {{}, std::chrono::milliseconds(1)});
  auto *schedule = std::get_if<std::shared_ptr<runtime::PresentationClock>>(
      &schedule_result);
  if (!schedule)
    return false;
  auto consumer_result =
      platform::create_presenter({platform::PresentationMode::headless, 3, 4});
  auto *consumer =
      std::get_if<std::shared_ptr<platform::Presenter>>(&consumer_result);
  if (!consumer)
    return false;
  bool legacy_access = false, observed = false, native_matches = true;
  runtime::PresentationDependencies dependencies;
  dependencies.clock = *schedule;
  dependencies.presenter = *consumer;
  dependencies.backing.lookup =
      [&](std::uint32_t) -> std::optional<services::GraphicsBufferBacking> {
    legacy_access = true;
    return {};
  };
  dependencies.backing.pin =
      [&](const services::GraphicsBufferBacking &, std::uint64_t,
          std::uint64_t) -> runtime::LaunchResult<runtime::GuestMemoryPin> {
    legacy_access = true;
    return runtime::LaunchFailure{};
  };
  dependencies.backing.wait_fence =
      [&](const services::GraphicsFence &, std::optional<runtime::ClockInstant>,
          std::stop_token) -> runtime::LaunchStatus {
    legacy_access = true;
    return runtime::LaunchFailure{};
  };
  dependencies.wants_readback = [keep](std::uint64_t) { return keep; };
  dependencies.observe_frame =
      [&](const platform::PresentedFrame &frame) -> runtime::LaunchStatus {
    if (!keep) {
      observed = !frame.readback;
      return std::monostate{};
    }
    observed = frame.readback && frame.readback->width == 3 &&
               frame.readback->height == 4;
    if (observed) {
      for (std::size_t i = 0; i < frame.readback->pixels.size(); i += 4)
        observed = observed && frame.readback->pixels[i] == expected[0] &&
                   frame.readback->pixels[i + 1] == expected[1] &&
                   frame.readback->pixels[i + 2] == expected[2] &&
                   frame.readback->pixels[i + 3] == expected[3];
    }
    return std::monostate{};
  };
  auto binding_result = runtime::bind_presentation(std::move(dependencies));
  auto *binding = std::get_if<std::shared_ptr<runtime::PresentationBinding>>(
      &binding_result);
  if (!binding)
    return false;
  horizon::HandleTable handles(16);
  services::vi::DisplaySession display(handles);
  display.set_presenter([&, callback = (*binding)->callback()](const services::GraphicsPresentation& frame) {
    if (expected_native) native_matches = frame.native_texture.get() == expected_native;
    if (before_consume) before_consume();
    if (enter_consumer) enter_consumer();
    auto result = callback(frame);
    // Inspection on a separate queue is ordered only after the actual
    // consumer returns. Unread frames must validate the consumed target,
    // rather than a producer texture whose upload may finish later.
    if (inspect_native && result.completed) {
      if (keep) native_matches = native_matches && native_pixels_match(frame.native_texture, expected);
      else {
        id<MTLTexture> target = platform::presenter_test_native_target(**consumer);
        if (!target) native_matches = false;
        else {
          auto owned = std::shared_ptr<const void>((__bridge_retained const void*)target,
              [](const void* value) { CFRelease(value); });
          native_matches = native_matches && native_pixels_match(owned, expected, true);
        }
      }
    }
    return result;
  });
  display.set_vsync_source(
      std::make_shared<services::vi::VsyncSource>(*schedule));
  int token;
  if (!display.bind_native_window(&token))
    return false;
  auto producer = services::vi::find_native_window(&token);
  std::array textures{texture, texture};
  textures[1].serial++;
  if (texture.resolve)
    textures[1].resolve = [resolve = texture.resolve] {
      auto resolved = resolve();
      if (resolved) ++resolved->serial;
      return resolved;
    };
  bool success = producer->register_textures(textures);
  const auto slot = producer->dequeue();
  success = success && slot.slot >= 0;
  if (success)
    success =
        producer
            ->present(
                slot.slot, 3, [&](std::uint64_t fence) {
                  if (fence_ready) fence_ready();
                  return fence == 3;
                },
                {1, 2, 4, 6}, 0)
            .completed;
  success = producer->unregister_textures() && success;
  services::vi::unbind_native_window(&token);
  const auto drained = (*binding)->drain(std::nullopt);
  const bool clean = std::holds_alternative<std::monostate>(drained);
  if (!success || !observed || !native_matches || legacy_access || !clean)
    std::fprintf(stderr, "present_consumer_evidence keep=%u success=%u observed=%u native_matches=%u legacy=%u drained=%u\n",
        unsigned(keep), unsigned(success), unsigned(observed), unsigned(native_matches), unsigned(legacy_access), unsigned(clean));
  return success && observed && native_matches && !legacy_access && clean;
}

bool verify_cpu_rewrite(id<MTLDevice> device, bool keep) {
  using namespace armigo;
  platform::metal::nvn_driver::Resources resources(device);
  auto memory = [device newBufferWithLength:0x10000 options:MTLResourceStorageModeShared];
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t*>(memory.contents); pool.size = memory.length;
  pool.flags = nvn::api::NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | nvn::api::NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  if (!resources.register_pool(pool)) return false;
  nvn::TextureState state{};
  state.description.target = nvn::api::NVN_TEXTURE_TARGET_2D; state.description.format = nvn::api::NVN_FORMAT_RGBA8;
  state.description.width = state.description.height = 8; state.description.depth = state.description.levels = state.description.samples = 1;
  const auto layout = nvn::resource_layout(state.description);
  if (!layout) return false;
  state.serial = 91; state.memory = pool.memory; state.pool = &pool; state.size = layout->total_size; state.layer_size = layout->layer_stride;
  const auto fill = [&](std::uint32_t color) {
    for (std::size_t offset=0;offset<state.size;offset+=4) std::memcpy(pool.memory+offset,&color,4);
    armigo::memory::WriteTracker::note_store(pool.memory,state.size);
  };
  fill(0xff0000ffu);
  resources.register_texture(state);
  auto exporter = platform::metal::nvn_driver::make_texture_exporter(device, resources);
  auto registered = exporter(state);
  if (!registered || !resources.flush_uploads()) return false;
  auto gate = [device newSharedEvent];
  std::promise<void> entered;
  auto entering = entered.get_future();
  std::atomic<bool> cleanup_used{false};
  auto cleanup = std::async(std::launch::async, [&] {
    if (entering.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
      cleanup_used.store(true);
      gate.signaledValue = 1;
    }
  });
  const bool result = verify_consumer(*registered, keep, [&] {
      [resources.submission_owner()->queue() waitForEvent:gate value:1];
      fill(0xff00ff00u);
    }, {std::byte{},std::byte{255},std::byte{},std::byte{255}}, nullptr, {}, true, [&] {
      gate.signaledValue = 1;
      entered.set_value();
    });
  cleanup.get();
  std::fprintf(stderr, "present_rewrite_order_evidence keep=%u consumer_result=%u cleanup_used=%u\n",
      unsigned(keep), unsigned(result), unsigned(cleanup_used.load()));
  resources.flush_uploads();
  resources.release_pool(pool);
  return result && !cleanup_used.load();
}

// A UI consumer arriving after a later write was encoded still reads the
// allocation at its reserved position in the producer's submission stream.
bool verify_reserved_frame(id<MTLDevice> device, bool keep) {
  using namespace armigo;
  using namespace platform::metal::nvn_driver;
  auto queue = [device newMTL4CommandQueue];
  auto owner = std::make_shared<Metal4SubmissionOwner>(device, queue);
  Resources resources(device, owner);
  auto backend = create_replay_backend(device, queue, resources);
  if (!backend) return false;
  auto memory = [device newBufferWithLength:0x10000 options:MTLResourceStorageModeShared];
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t*>(memory.contents); pool.size = memory.length;
  pool.flags = nvn::api::NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | nvn::api::NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  if (!resources.register_pool(pool)) return false;
  nvn::TextureState state{};
  state.description.target = nvn::api::NVN_TEXTURE_TARGET_2D; state.description.format = nvn::api::NVN_FORMAT_RGBA8;
  state.description.width = state.description.height = 8;
  state.description.depth = state.description.levels = state.description.samples = 1;
  const auto layout = nvn::resource_layout(state.description);
  if (!layout) return false;
  state.serial = 107; state.memory = pool.memory; state.pool = &pool;
  state.size = layout->total_size; state.layer_size = layout->layer_stride;
  const std::uint32_t red = 0xff0000ff;
  for (std::size_t offset = 0; offset < state.size; offset += 4) std::memcpy(pool.memory + offset, &red, 4);
  resources.register_texture(state);
  auto registered = make_texture_exporter(device, resources, *backend)(state);
  if (!registered || !resources.flush_uploads()) return false;
  auto pending = registered->capture(0);
  if (!pending) return false;
  auto captured = pending->resolve();
  if (!captured) return false;
  id<MTLTexture> texture = (__bridge id<MTLTexture>)captured->native_texture.get();
  auto allocator = [device newCommandAllocator];
  auto commands = [device newCommandBuffer];
  auto green = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
  for (unsigned i = 0; i < 64; ++i) static_cast<std::uint32_t*>(green.contents)[i] = 0xff00ff00;
  NSError* error = nil;
  auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
  [residency addAllocation:texture]; [residency addAllocation:green]; [residency commit];
  [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
  auto copy = [commands computeCommandEncoder];
  [copy barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
  [copy copyFromBuffer:green sourceOffset:0 sourceBytesPerRow:32 sourceBytesPerImage:256
      sourceSize:MTLSizeMake(8,8,1) toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [copy endEncoding]; [commands endCommandBuffer];
  const auto later = owner->publish(commands, ^(id<MTL4CommitFeedback>) {
    static_cast<void>(allocator); static_cast<void>(residency); static_cast<void>(green);
  });
  const bool held = !owner->wait_gpu(later, std::chrono::milliseconds(1));
  const bool red_before_later = verify_consumer(*captured, keep);
  const bool completed = owner->wait(later, std::chrono::seconds(5));
  pending->cancel(); captured.reset(); pending.reset(); registered.reset();
  const bool stopped = backend->stop_submission_workers(std::chrono::steady_clock::now()+std::chrono::seconds(5));
  resources.release_pool(pool);
  return held && red_before_later && completed && stopped;
}

bool verify_rejected_resolutions() {
  using namespace armigo;
  for (unsigned mode=0;mode<5;++mode) {
    horizon::HandleTable handles(16);
    services::vi::DisplaySession display(handles);
    display.set_vsync_source(std::make_shared<services::vi::VsyncSource>());
    std::atomic<unsigned> presented{0}, diagnostics{0};
    display.set_presenter([&](const services::GraphicsPresentation&) { ++presented; return services::GraphicsPresentationResult{true,{}}; });
    int token;
    if (!display.bind_native_window(&token)) return false;
    auto producer = services::vi::find_native_window(&token);
    services::vi::NativeTexture texture;
    texture.serial=19; texture.width=texture.height=8;
    texture.read_pixels=[](std::array<std::int32_t,4>) -> graphics::Rgba8ReadbackResult { return graphics::Rgba8Image{}; };
    texture.resolve=[texture,mode]() -> std::optional<services::vi::NativeTexture> {
      if (mode==0) return std::nullopt;
      if (mode==4) throw std::runtime_error("synthetic resolution failure");
      auto result=texture;
      if (mode==1) ++result.serial;
      if (mode==2) ++result.width;
      if (mode==3) result.read_pixels={};
      return result;
    };
    std::array textures{texture,texture}; ++textures[1].serial;
    if (!producer->register_textures(textures)) return false;
    const auto slot=producer->dequeue(); if(slot.slot<0) return false;
    diag::testing_set_sink([](diag::Level, std::string_view line, void* context) {
      if(line.find("native texture allocation")!=std::string_view::npos) ++*static_cast<std::atomic<unsigned>*>(context);
    },&diagnostics);
    const bool queued=producer->present(slot.slot,1,[](auto){return true;},{},0).completed;
    const bool drained=producer->unregister_textures();
    diag::testing_set_sink(nullptr,nullptr);
    services::vi::unbind_native_window(&token);
    if(!queued || !drained || presented.load()!=0 || diagnostics.load()!=1) return false;
  }
  return true;
}

int main() {
  @autoreleasepool {
    using namespace armigo;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      return 77;
    unsigned failures = 0;
    if (!verify_reserved_frame(device, true) || !verify_reserved_frame(device, false)) {
      std::fprintf(stderr, "Reserved native frame did not precede the later write\n");
      ++failures;
    }
    for (const auto format :
         {MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGRA8Unorm}) {
      auto descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                             width:8
                                                            height:8
                                                         mipmapped:NO];
      descriptor.storageMode = MTLStorageModePrivate;
      descriptor.usage =
          MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
      id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = texture;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
      auto encoder = [command renderCommandEncoderWithDescriptor:pass];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (!texture || command.status != MTLCommandBufferStatusCompleted)
        return 1;
      nvn::TextureState state{};
      state.serial = 19;
      state.description.width = 8;
      state.description.height = 8;
      auto exporter = platform::metal::nvn_driver::make_texture_exporter(
          device,
          [texture](const nvn::TextureState &current) -> id<MTLTexture> {
            return current.serial == 19 ? texture : nil;
          });
      auto exported = exporter(state);
      if (!exported || exported->serial != 19)
        return 1;
      for (const bool keep : {true, false})
        if (!verify_consumer(*exported, keep))
          ++failures;
      auto result = exported->read_pixels({1, 2, 4, 6});
      auto *image = std::get_if<graphics::Rgba8Image>(&result);
      if (!image || image->width != 3 || image->height != 4 ||
          image->pixels.size() != 48)
        return 1;
      for (std::size_t i = 0; i < image->pixels.size(); i += 4) {
        if (image->pixels[i] != std::byte{255} ||
            image->pixels[i + 1] != std::byte{} ||
            image->pixels[i + 2] != std::byte{} ||
            image->pixels[i + 3] != std::byte{255})
          ++failures;
      }
      if (!std::holds_alternative<graphics::ImageReadbackError>(
              exported->read_pixels({0, 0, 9, 1})))
        ++failures;
      // Window registration is earlier than backing replacement. Resolve only
      // after the frame fence, then keep the chosen allocation for this frame.
      id<MTLTexture> replacement = [device newTextureWithDescriptor:descriptor];
      auto green_commands = [queue commandBuffer];
      pass.colorAttachments[0].texture = replacement;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 1, 0, 1);
      auto green_encoder = [green_commands renderCommandEncoderWithDescriptor:pass];
      [green_encoder endEncoding];
      [green_commands commit];
      [green_commands waitUntilCompleted];
      if (!replacement || green_commands.status != MTLCommandBufferStatusCompleted)
        return 1;
      id<MTLTexture> current = texture;
      std::atomic<unsigned> lookup_calls{0};
      auto sentinel_deaths = std::make_shared<std::atomic<unsigned>>(0);
      // A C ABI call prevents ARC's return-value optimization from replacing
      // this deliberately autoreleased lifetime with an immediate release.
      auto autorelease = reinterpret_cast<void*(*)(void*)>(dlsym(RTLD_DEFAULT, "objc_autorelease"));
      if (!autorelease) return 1;
      auto rotating_exporter = platform::metal::nvn_driver::make_texture_exporter(
          device, [&](const nvn::TextureState&) {
            if (lookup_calls.fetch_add(1) % 2) {
              auto sentinel = [PresentAutoreleaseSentinel new];
              sentinel->counter = sentinel_deaths;
              autorelease((__bridge_retained void*)sentinel);
            }
            return current;
          });
      for (const bool keep : {true, false}) {
        current = texture;
        auto registered = rotating_exporter(state);
        if (!registered || !registered->resolve)
          return 1;
        bool drained_before_consume = false;
        if (!verify_consumer(*registered, keep, [&] { current = replacement; },
              {std::byte{}, std::byte{255}, std::byte{}, std::byte{255}},
              (__bridge const void*)replacement, [&] {
                drained_before_consume = sentinel_deaths->load() == lookup_calls.load() / 2;
                current = texture;
              }, true)) {
          std::fprintf(stderr, "Rotated presentation did not use the current allocation (readback=%u)\n", unsigned(keep));
          ++failures;
        }
        if (!drained_before_consume || sentinel_deaths->load() != lookup_calls.load() / 2) {
          std::fprintf(stderr, "Deferred presentation did not drain its autorelease pool\n");
          ++failures;
        }
        auto old_pixels = registered->read_pixels({1, 2, 4, 6});
        auto* old_image = std::get_if<graphics::Rgba8Image>(&old_pixels);
        if (!old_image || old_image->pixels[0] != std::byte{255} || old_image->pixels[1] != std::byte{})
          ++failures;
      }
      state.serial = 20;
      if (exporter(state))
        ++failures;
    }
    if (!verify_rejected_resolutions()) { std::fprintf(stderr, "Invalid native allocation was not rejected\n"); ++failures; }
    for (const bool keep : {true, false}) if (!verify_cpu_rewrite(device, keep)) {
      std::fprintf(stderr, "CPU rewritten presentation was consumed before its upload (readback=%u)\n", unsigned(keep));
      ++failures;
    }
    std::printf("NVN Metal presentation failures: %u\n", failures);
    return failures ? 1 : 0;
  }
}
