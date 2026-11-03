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
#include <armigo/runtime/presentation.h>
#include <armigo/services/video_interface.h>
#include <cstdio>

// keep: the run keeps the frame's pixels. Otherwise the headless consumer
// copies the texture on the GPU and the observer sees no readback.
bool verify_consumer(const armigo::services::vi::NativeTexture &texture,
                     bool keep) {
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
  bool legacy_access = false, observed = false;
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
        observed = observed && frame.readback->pixels[i] == std::byte{255} &&
                   frame.readback->pixels[i + 1] == std::byte{} &&
                   frame.readback->pixels[i + 2] == std::byte{} &&
                   frame.readback->pixels[i + 3] == std::byte{255};
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
  display.set_presenter((*binding)->callback());
  display.set_vsync_source(
      std::make_shared<services::vi::VsyncSource>(*schedule));
  int token;
  if (!display.bind_native_window(&token))
    return false;
  auto producer = services::vi::find_native_window(&token);
  std::array textures{texture, texture};
  textures[1].serial++;
  bool success = producer->register_textures(textures);
  const auto slot = producer->dequeue();
  success = success && slot.slot >= 0;
  if (success)
    success =
        producer
            ->present(
                slot.slot, 3, [](std::uint64_t fence) { return fence == 3; },
                {1, 2, 4, 6}, 0)
            .completed;
  success = producer->unregister_textures() && success;
  services::vi::unbind_native_window(&token);
  const auto drained = (*binding)->drain(std::nullopt);
  return success && observed && !legacy_access &&
         std::holds_alternative<std::monostate>(drained);
}

int main() {
  @autoreleasepool {
    using namespace armigo;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      return 77;
    unsigned failures = 0;
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
      state.serial = 20;
      if (exporter(state))
        ++failures;
    }
    std::printf("NVN Metal presentation failures: %u\n", failures);
    return failures ? 1 : 0;
  }
}
