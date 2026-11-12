// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include "nvn/commands.h"
#include "nvn/entry_table.h"
#include "nvn/programs.h"
#include "nvn/resource_layout.h"
#include "platform/metal/nvn_image_formats.h"
#include "platform/metal/nvn_replay.h"
#include "platform/metal/nvn_resources.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace nvn = armigo::nvn;
namespace metal = armigo::platform::metal::nvn_driver;
namespace platform = armigo::platform;
using nvn::Entry;
template <nvn::Entry E> auto load() {
  return reinterpret_cast<typename nvn::EntrySignature<E>::type>(
      nvn::bootstrap_loader_for(nvn::selected_api_version().value_or(
          nvn::ApiVersion{55, 15}))(nvn::entry_name(E).data()));
}
using namespace armigo::nvn::api;
namespace {
void require(bool value, const char *what) {
  if (!value)
    throw std::runtime_error(what);
}
struct Memory {
  std::size_t size = 16 * static_cast<std::size_t>(getpagesize());
  void *data = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ~Memory() {
    if (data != MAP_FAILED)
      munmap(data, size);
  }
};
nvn::TextureState state(nvn::MemoryPoolState &pool, bool image,
                        std::uint64_t serial) {
  nvn::TextureState s{};
  s.serial = serial;
  s.pool = &pool;
  s.memory = pool.memory;
  s.description.target = NVN_TEXTURE_TARGET_2D;
  s.description.format = NVN_FORMAT_RGBA8_SRGB;
  s.description.width = s.description.height = 4;
  s.description.depth = s.description.levels = s.description.samples = 1;
  s.description.flags = image ? NVN_TEXTURE_FLAGS_IMAGE : 0;
  auto layout = nvn::resource_layout(s.description);
  require(layout.has_value(), "synthetic texture layout");
  s.size = layout->total_size;
  s.layer_size = layout->layer_stride;
  return s;
}
void fill(metal::Resources &resources, const nvn::TextureState &texture) {
  std::vector<std::byte> pixels(64);
  for (unsigned i = 0; i < 16; ++i) {
    pixels[i * 4] = std::byte{128};
    pixels[i * 4 + 1] = std::byte{64};
    pixels[i * 4 + 2] = std::byte{192};
    pixels[i * 4 + 3] = std::byte{255};
  }
  nvn::TexelTransfer t;
  t.region = {0, 0, 0, 4, 4, 1};
  t.row_stride = 16;
  t.image_stride = 64;
  require(resources.write_texels(texture, t, pixels.data()),
          "synthetic texture upload");
}
constexpr const char *source = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void inspect(texture2d<float, access::read> image [[texture(0)]],
                    texture2d<float> sampled [[texture(1)]], device float4* out [[buffer(0)]]) {
    constexpr sampler nearest(coord::pixel, filter::nearest);
    out[0] = image.read(uint2(0)); out[1] = sampled.sample(nearest, float2(0.5f));
    out[2] = image.read(uint2(1,0));
}
kernel void store(texture2d<float, access::write> image [[texture(0)]]) {
    image.write(float4(0.75f, 0.25f, 0.5f, 1.0f), uint2(0));
}
vertex float4 triangle(uint id [[vertex_id]]) {
    constexpr float2 p[]{float2(-1,-1),float2(3,-1),float2(-1,3)};
    return float4(p[id],0,1);
}
fragment float4 feedback(texture2d<float, access::read> image [[texture(0)]],
                         texture2d<float> sampled [[texture(1)]]) {
    constexpr sampler nearest(coord::pixel, filter::nearest);
    return float4(image.read(uint2(0)).r, sampled.sample(nearest,float2(0.5f)).r, 0, 1);
}
)msl";
struct GPU {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLComputePipelineState> reader, writer;
  GPU() {
    require(device && queue, "Metal device and queue");
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library = [device newLibraryWithSource:@(source)
                                                  options:options
                                                    error:&error];
    if (!library)
      throw std::runtime_error(error.localizedDescription.UTF8String);
    reader = [device
        newComputePipelineStateWithFunction:[library
                                                newFunctionWithName:@"inspect"]
                                      error:&error];
    writer = [device
        newComputePipelineStateWithFunction:[library
                                                newFunctionWithName:@"store"]
                                      error:&error];
    require(reader && writer, "synthetic compute pipelines");
  }
  std::array<float, 12> read(id<MTLTexture> raw, id<MTLTexture> sampled) {
    id<MTLBuffer> result =
        [device newBufferWithLength:48 options:MTLResourceStorageModeShared];
    auto commands = [queue commandBuffer];
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:reader];
    [encoder setTexture:raw atIndex:0];
    [encoder setTexture:sampled atIndex:1];
    [encoder setBuffer:result offset:0 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    require(commands.status == MTLCommandBufferStatusCompleted,
            "image read completion");
    std::array<float, 12> values{};
    std::memcpy(values.data(), result.contents, 48);
    return values;
  }
  void write(id<MTLTexture> raw) {
    auto commands = [queue commandBuffer];
    auto encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:writer];
    [encoder setTexture:raw atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    require(commands.status == MTLCommandBufferStatusCompleted,
            "image write completion");
  }
};
void initial_image(GPU &gpu) {
  Memory memory;
  require(memory.data != MAP_FAILED, "pool allocation");
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t *>(memory.data);
  pool.size = memory.size;
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  metal::Resources resources(gpu.device);
  require(resources.register_pool(pool), "pool registration");
  auto texture = state(pool, true, 901);
  resources.register_texture(texture);
  resources.register_texture_descriptor(7, texture, nullptr);
  fill(resources, texture);
  metal::Resources::DescriptorStamp stamp;
  id<MTLTexture> raw = resources.image_base_for_id(7, stamp);
  require(raw != nil, "sRGB image backing exists");
  auto sampled = resources.texture_for_id(7);
  require(sampled && sampled.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB,
          "sampling retains sRGB decoding");
  require(resources.flush_uploads(), "texture uploads complete");
  auto values = gpu.read(raw, sampled);
  std::printf("Image red %g, sampled red %g\n", values[0], values[4]);
  require(std::abs(values[0] - 128.f / 255.f) < 0.002f && values[4] > 0.21f &&
              values[4] < 0.22f,
          "image reads raw while sampling decodes");
  require(raw.pixelFormat == MTLPixelFormatRGBA8Unorm &&
              (raw.usage & MTLTextureUsageShaderWrite),
          "sRGB image needs legal linear shader-write backing");
  gpu.write(raw);
  values = gpu.read(raw, sampled);
  require(std::abs(values[0] - 191.f / 255.f) < 0.002f && values[4] > 0.51f &&
              values[4] < 0.53f,
          "image stores preserve encoded bytes without a second conversion");
  auto attachment = resources.texture(texture, nullptr);
  require(attachment.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB,
          "render attachment retains encoding");
  require(resources.peek_attachment(texture, nullptr).format ==
              attachment.pixelFormat,
          "lookahead agrees with attachment view");
  nvn::TextureViewState view{};
  view.sets_format = 1;
  view.format = NVN_FORMAT_RGBA8_SRGB;
  auto explicit_attachment = resources.texture(texture, &view);
  require(explicit_attachment.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB,
          "explicit sRGB attachment view");
  require(resources.peek_attachment(texture, &view).format ==
              explicit_attachment.pixelFormat,
          "explicit view lookahead");
  resources.release_pool(pool);
}
struct Recorder {
  NVNdevice device{};
  NVNcommandBuffer commands{};
  NVNmemoryPool pool{};
  std::array<std::byte, 64 * 1024> donated{};
  Recorder() {
    auto &p = *reinterpret_cast<nvn::MemoryPoolState *>(&pool);
    p.memory = reinterpret_cast<std::uint8_t *>(donated.data());
    p.size = donated.size();
    require(load<Entry::nvnCommandBufferInitialize>()(&commands, &device),
            "recorder initialization");
    load<Entry::nvnCommandBufferAddCommandMemory>()(&commands, &pool, 0,
                                                    donated.size());
    load<Entry::nvnCommandBufferAddControlMemory>()(&commands, donated.data(),
                                                    donated.size());
  }
  ~Recorder() { load<Entry::nvnCommandBufferFinalize>()(&commands); }
};
struct Programs {
  std::shared_ptr<const platform::CompiledGraphicsStage> reader, writer, vertex,
      fragment;
  Programs() {
    auto r = std::make_shared<platform::CompiledGraphicsStage>();
    r->stage = armigo::shader_input::Stage::Compute;
    r->msl_source = source;
    r->entry_point = "inspect";
    r->compute_local_size = {1, 1, 1};
    platform::CompiledShaderResource image;
    image.kind = platform::CompiledShaderResourceKind::StorageImage;
    image.array_length = image.location.count = 1;
    image.cbuf_slot = 3;
    r->resources = {image};
    auto sampled = image;
    sampled.kind = platform::CompiledShaderResourceKind::SampledTexture;
    sampled.resource_id = sampled.location.first = 1;
    sampled.handle_byte_offset = 8;
    r->resources.push_back(sampled);
    r->storage_buffers = {
        {0, 0x310, 0, platform::CompiledShaderResourceAccess::Write, 4}};
    reader = r;
    auto w = std::make_shared<platform::CompiledGraphicsStage>(*r);
    w->entry_point = "store";
    w->resources.resize(1);
    w->resources[0].access = platform::CompiledShaderResourceAccess::Write;
    w->storage_buffers.clear();
    writer = w;
    auto v = std::make_shared<platform::CompiledGraphicsStage>();
    v->stage = armigo::shader_input::Stage::Vertex;
    v->entry_point = "triangle";
    v->msl_source = source;
    vertex = v;
    auto f = std::make_shared<platform::CompiledGraphicsStage>(*r);
    f->stage = armigo::shader_input::Stage::Fragment;
    f->entry_point = "feedback";
    f->storage_buffers.clear();
    f->interface.color_output_mask = 15;
    fragment = f;
    nvn::set_program_resolver(
        [](void *context, const nvn::DeviceState *,
           std::span<const NVNshaderData> shaders,
           std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 6>
               &stages,
           std::string &) {
          auto &self = *static_cast<Programs *>(context);
          if (shaders.size() == 2) {
            stages[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            stages[NVN_SHADER_STAGE_FRAGMENT] = self.fragment;
          } else
            stages[NVN_SHADER_STAGE_COMPUTE] =
                shaders.front().data == 1 ? self.reader : self.writer;
          return true;
        },
        this);
  }
  ~Programs() { nvn::set_program_resolver(nullptr, nullptr); }
};
void replay_images(GPU &gpu, bool initial_image_flag,
                   bool pending_clear = false, bool linear_swizzle = false) {
  Memory memory;
  require(memory.data != MAP_FAILED, "replay pool allocation");
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t *>(memory.data);
  pool.size = memory.size;
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  metal::Resources resources(gpu.device);
  require(resources.register_pool(pool), "replay pool registration");
  auto texture =
      state(pool, initial_image_flag, initial_image_flag ? 903 : 904);
  resources.register_texture(texture);
  nvn::TextureViewState image_view{};
  image_view.sets_format = 1;
  image_view.format = NVN_FORMAT_RGBA8;
  image_view.sets_swizzle = 1;
  image_view.swizzle[0] = NVN_TEXTURE_SWIZZLE_B;
  image_view.swizzle[1] = NVN_TEXTURE_SWIZZLE_G;
  image_view.swizzle[2] = NVN_TEXTURE_SWIZZLE_R;
  image_view.swizzle[3] = NVN_TEXTURE_SWIZZLE_ONE;
  resources.register_texture_descriptor(7, texture,
                                        linear_swizzle ? &image_view : nullptr);
  fill(resources, texture);
  auto original = resources.texture_for_id(7);
  require(original, "original sampled view");
  require(resources.flush_uploads(), "replay initial upload");
  const std::uint64_t handles[]{7, 7};
  std::memcpy(pool.memory + 0x4000, handles, sizeof handles);
  id<MTL4CommandQueue> queue = resources.submission_owner()->queue();
  auto backend = metal::create_replay_backend(gpu.device, queue, resources);
  require(backend != nullptr, "real resource adapter");
  Programs programs;
  Recorder recorder;
  NVNprogram reader{}, writer{};
  require(load<Entry::nvnProgramInitialize>()(&reader, &recorder.device),
          "reader program");
  require(load<Entry::nvnProgramInitialize>()(&writer, &recorder.device),
          "writer program");
  NVNshaderData read_shader{1, nullptr}, write_shader{2, nullptr};
  require(load<Entry::nvnProgramSetShaders>()(&reader, 1, &read_shader),
          "reader shader");
  require(load<Entry::nvnProgramSetShaders>()(&writer, 1, &write_shader),
          "writer shader");
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  NVNtexture target{};
  auto object_state = texture;
  object_state.snapshot = &texture;
  std::memcpy(&target, &object_state, sizeof object_state);
  const NVNtexture *targets[]{&target};
  if (pending_clear) {
    load<Entry::nvnCommandBufferSetRenderTargets>()(
        &recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    const float color[]{0.125f, 0.25f, 0.5f, 1};
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, color, 15);
  }
  load<Entry::nvnCommandBufferBindUniformBuffer>()(
      &recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0,
      reinterpret_cast<std::uint64_t>(pool.memory + 0x4000), 16);
  const auto read_into = [&](unsigned offset) {
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &reader,
                                               NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(
        &recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0,
        reinterpret_cast<std::uint64_t>(pool.memory + offset), 48);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
  };
  // This reader executes from the original allocation even when the later
  // writer replaces it.
  read_into(0x5000);
  load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &writer,
                                             NVN_SHADER_STAGE_COMPUTE_BIT);
  load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
  read_into(0x5100);
  const auto handle =
      load<Entry::nvnCommandBufferEndRecording>()(&recorder.commands);
  const auto fence = backend->submit({&handle, 1});
  require(fence && backend->wait_fence(fence, 10'000'000'000ull),
          "image replay completion");
  if (backend->last_error())
    throw std::runtime_error(backend->last_error().detail);
  require(backend->statistics().dispatches == 3 &&
              backend->statistics().skipped == 0,
          "all image dispatches execute");
  for (const auto offset : {0x5000, 0x5100}) {
    std::array<float, 12> values{};
    std::memcpy(values.data(), pool.memory + offset, 48);
    std::printf("Replay %d at %x: raw %g sampled %g\n", initial_image_flag,
                offset, values[0], values[4]);
    const bool before = offset == 0x5000;
    const float original_raw = pending_clear ? 99.f : 128.f;
    const float expected_sample = linear_swizzle
                                      ? (before ? 192.f : 128.f) / 255.f
                                  : before ? (pending_clear ? 0.125f : 0.21586f)
                                           : 0.521f;
    require(std::abs(values[0] - (before ? original_raw : 191.f) / 255.f) <
                    0.004f &&
                std::abs(values[4] - expected_sample) < 0.008f,
            "real replay keeps image identity independent of sampled format "
            "and mapping");
    require(std::abs(values[8] - (pending_clear ? 99.f : 128.f) / 255.f) <
                0.004f,
            "backing replacement preserves untouched texels");
  }
  metal::Resources::DescriptorStamp stamp;
  auto backing = resources.image_base_for_id(7, stamp);
  require(backing.pixelFormat == MTLPixelFormatRGBA8Unorm &&
              (backing.usage & MTLTextureUsageShaderWrite),
          "replay installs linear writable backing");
  auto sampled = resources.texture_for_id(7);
  auto raw = resources.raw_image_view(7, sampled);
  require(raw && raw.pixelFormat == MTLPixelFormatRGBA8Unorm,
          "raw adapter view");
  auto values = gpu.read(raw, sampled);
  require(std::abs(values[4] - (linear_swizzle ? 128.f / 255.f : 0.521f)) <
              0.008f,
          "sampled format and mapping after raw store");
  require(resources.texture(texture, nullptr).pixelFormat ==
              MTLPixelFormatRGBA8Unorm_sRGB,
          "attachment after replay upgrade");
  load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
  load<Entry::nvnProgramFinalize>()(&reader);
  load<Entry::nvnProgramFinalize>()(&writer);
  backend.reset();
  resources.release_pool(pool);
}
void feedback_images(GPU &gpu, bool linear_descriptor) {
  Memory memory;
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t *>(memory.data);
  pool.size = memory.size;
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  metal::Resources resources(gpu.device);
  require(resources.register_pool(pool), "feedback pool");
  auto texture = state(pool, true, 906);
  resources.register_texture(texture);
  resources.register_texture_descriptor(7, texture, nullptr);
  nvn::TextureViewState view{};
  view.sets_format = 1;
  view.format = NVN_FORMAT_RGBA8;
  resources.register_texture_descriptor(8, texture,
                                        linear_descriptor ? &view : nullptr);
  fill(resources, texture);
  require(resources.flush_uploads(), "feedback uploads");
  const std::uint64_t handles[]{8, 7};
  std::memcpy(pool.memory + 0x4000, handles, sizeof handles);
  Programs programs;
  Recorder recorder;
  NVNprogram program{};
  require(load<Entry::nvnProgramInitialize>()(&program, &recorder.device),
          "feedback program");
  const NVNshaderData shaders[]{{3, nullptr}, {3, nullptr}};
  require(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders),
          "feedback shaders");
  NVNtexture target{};
  auto object_state = texture;
  object_state.snapshot = &texture;
  std::memcpy(&target, &object_state, sizeof object_state);
  const NVNtexture *targets[]{&target};
  auto backend = metal::create_replay_backend(
      gpu.device, resources.submission_owner()->queue(), resources);
  load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
  load<Entry::nvnCommandBufferSetRenderTargets>()(
      &recorder.commands, 1, targets, nullptr, nullptr, nullptr);
  load<Entry::nvnCommandBufferBindProgram>()(
      &recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
  load<Entry::nvnCommandBufferBindUniformBuffer>()(
      &recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
      reinterpret_cast<std::uint64_t>(pool.memory + 0x4000), 16);
  load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 4, 4);
  load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands,
                                            NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
  const auto handle =
      load<Entry::nvnCommandBufferEndRecording>()(&recorder.commands);
  const auto fence = backend->submit({&handle, 1});
  require(fence && backend->wait_fence(fence, 10'000'000'000ull),
          "feedback completion");
  if (backend->last_error())
    throw std::runtime_error(backend->last_error().detail);
  require(backend->statistics().skipped == 0 &&
              backend->statistics().draws == 1 &&
              backend->statistics().feedback_snapshots >= 1,
          "feedback read takes a snapshot");
  auto sampled = resources.texture_for_id(7);
  auto raw = resources.raw_image_view(7, sampled);
  const auto values = gpu.read(raw, sampled);
  require(std::abs(values[4] - 128.f / 255.f) < 0.005f &&
              std::abs(values[5] - 0.21586f) < 0.005f,
          "feedback preserves raw image and sRGB sampled values");
  load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
  load<Entry::nvnProgramFinalize>()(&program);
  backend.reset();
  resources.release_pool(pool);
}

void image_mapping_cases(GPU &gpu) {
  Memory memory;
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t *>(memory.data);
  pool.size = memory.size;
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  metal::Resources resources(gpu.device);
  require(resources.register_pool(pool), "mapping pool");
  auto texture = state(pool, true, 908);
  resources.register_texture(texture);
  resources.register_texture_descriptor(7, texture, nullptr);
  auto d = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:4
                                  height:4
                               mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
            MTLTextureUsagePixelFormatView;
  auto root = [gpu.device newTextureWithDescriptor:d];
  auto sampled =
      [root newTextureViewWithPixelFormat:root.pixelFormat
                              textureType:MTLTextureType2D
                                   levels:NSMakeRange(0, 1)
                                   slices:NSMakeRange(0, 1)
                                  swizzle:MTLTextureSwizzleChannelsMake(
                                              MTLTextureSwizzleBlue,
                                              MTLTextureSwizzleGreen,
                                              MTLTextureSwizzleRed,
                                              MTLTextureSwizzleOne)];
  require(sampled, "linear sampled mapping");
  auto raw = resources.raw_image_view(7, sampled);
  require(raw && raw.swizzle.red == MTLTextureSwizzleRed &&
              raw.swizzle.alpha == MTLTextureSwizzleAlpha,
          "linear image descriptor removes sampled swizzle");
  gpu.write(raw);
  auto values = gpu.read(raw, sampled);
  require(values[0] > .74f && values[0] < .76f && values[4] > .49f &&
              values[4] < .51f,
          "linear image identity executes independently of sampled mapping");
  auto buffer = [gpu.device newBufferWithLength:256
                                        options:MTLResourceStorageModeShared];
  auto b = [MTLTextureDescriptor
      textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                                       width:16
                             resourceOptions:MTLResourceStorageModeShared
                                       usage:MTLTextureUsageShaderRead |
                                             MTLTextureUsageShaderWrite];
  auto encoded = [buffer newTextureWithDescriptor:b offset:0 bytesPerRow:256];
  require(encoded, "synthetic sRGB texture buffer");
  require(!resources.raw_image_view(7, encoded),
          "required buffer image conversion is explicitly refused");
  resources.release_pool(pool);
}
void whole_attachment_views(GPU &gpu) {
  Memory memory;
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t *>(memory.data);
  pool.size = memory.size;
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  metal::Resources resources(gpu.device);
  require(resources.register_pool(pool), "whole view pool");
  auto texture = state(pool, true, 907);
  texture.description.levels = 3;
  const auto layout = nvn::resource_layout(texture.description);
  require(layout.has_value(), "whole mip layout");
  texture.size = layout->total_size;
  texture.layer_size = layout->layer_stride;
  resources.register_texture(texture);
  resources.register_texture_descriptor(7, texture, nullptr);
  metal::Resources::DescriptorStamp stamp;
  const auto backing = resources.image_base_for_id(7, stamp);
  const auto attachment = resources.texture(texture, nullptr);
  require(attachment && attachment.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB,
          "whole sRGB view");
  require(attachment.mipmapLevelCount == backing.mipmapLevelCount &&
              attachment.textureType == backing.textureType,
          "no-view attachment retains the whole backing shape");
  require(resources.flush_uploads(), "whole view uploads");
  resources.release_pool(pool);
}
void format_twins() {
  constexpr std::pair<MTLPixelFormat, MTLPixelFormat> pairs[]{
      {MTLPixelFormatRGBA8Unorm_sRGB, MTLPixelFormatRGBA8Unorm},
      {MTLPixelFormatBGRA8Unorm_sRGB, MTLPixelFormatBGRA8Unorm},
      {MTLPixelFormatBC1_RGBA_sRGB, MTLPixelFormatBC1_RGBA},
      {MTLPixelFormatBC2_RGBA_sRGB, MTLPixelFormatBC2_RGBA},
      {MTLPixelFormatBC3_RGBA_sRGB, MTLPixelFormatBC3_RGBA},
      {MTLPixelFormatBC7_RGBAUnorm_sRGB, MTLPixelFormatBC7_RGBAUnorm}};
  for (const auto &[encoded, linear] : pairs) {
    require(metal::raw_image_format(encoded) == linear &&
                metal::raw_image_format(linear) == linear &&
                metal::raw_image_format_twins(encoded, linear) &&
                metal::raw_image_format_twins(linear, encoded),
            "exact sRGB format twins");
    require(!metal::raw_image_format_twins(encoded, MTLPixelFormatR32Uint),
            "unrelated image formats retain their pixel class");
  }
  require(metal::raw_image_format(MTLPixelFormatDepth32Float) ==
              MTLPixelFormatDepth32Float,
          "depth format is unchanged");
}
void selected_views(GPU &gpu) {
  Memory memory;
  nvn::MemoryPoolState pool{};
  pool.memory = static_cast<std::uint8_t *>(memory.data);
  pool.size = memory.size;
  pool.flags =
      NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
  metal::Resources resources(gpu.device);
  require(resources.register_pool(pool), "view pool");
  auto texture = state(pool, true, 905);
  resources.register_texture(texture);
  resources.register_texture_descriptor(7, texture, nullptr);
  const auto make = [&](MTLTextureType type, unsigned layers) {
    auto d = [MTLTextureDescriptor new];
    d.pixelFormat = MTLPixelFormatRGBA8Unorm;
    d.textureType = type;
    d.width = d.height = 16;
    d.depth = type == MTLTextureType3D ? 4 : 1;
    d.mipmapLevelCount = 4;
    d.arrayLength = layers;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
              MTLTextureUsagePixelFormatView;
    return [gpu.device newTextureWithDescriptor:d];
  };
  for (auto type :
       {MTLTextureType2DArray, MTLTextureTypeCubeArray, MTLTextureType3D}) {
    auto root = make(type, type == MTLTextureType3D ? 1 : 2);
    require(root, "view root");
    const unsigned slice = type == MTLTextureType3D          ? 0
                           : type == MTLTextureTypeCubeArray ? 6
                                                             : 1;
    const unsigned count = type == MTLTextureTypeCubeArray ? 6 : 1;
    auto selected =
        [root newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                                textureType:type
                                     levels:NSMakeRange(1, 2)
                                     slices:NSMakeRange(slice, count)
                                    swizzle:MTLTextureSwizzleChannelsMake(
                                                MTLTextureSwizzleBlue,
                                                MTLTextureSwizzleGreen,
                                                MTLTextureSwizzleRed,
                                                MTLTextureSwizzleOne)];
    require(selected, "selected sRGB view");
    auto nested =
        [selected newTextureViewWithPixelFormat:selected.pixelFormat
                                    textureType:type
                                         levels:NSMakeRange(1, 1)
                                         slices:NSMakeRange(0, count)];
    require(nested, "nested descriptor view");
    auto raw = resources.raw_image_view(7, nested);
    require(raw && raw.textureType == type && raw.mipmapLevelCount == 1 &&
                raw.width == 4,
            "raw view type and mip selection");
    require(raw.parentRelativeLevel == 2 && raw.parentRelativeSlice == slice,
            "raw view absolute subresource");
    require(raw.swizzle.red == MTLTextureSwizzleRed &&
                raw.swizzle.alpha == MTLTextureSwizzleAlpha,
            "image identity channels");
    require(selected.swizzle.red == MTLTextureSwizzleBlue,
            "sampled swizzle preserved");
    require(resources.raw_image_view(7, nested) == raw,
            "raw view cached by strong source");
    resources.register_texture_descriptor(7, texture, nullptr);
    auto replacement = resources.raw_image_view(7, nested);
    require(replacement && replacement != raw,
            "registration replaces raw view cache");
    require(raw.gpuResourceID._impl != 0, "old raw view remains alive");
  }
  resources.release_pool(pool);
}

} // namespace
int main() {
  @autoreleasepool {
    try {
      GPU gpu;
      nvn::select_api_version({55, 15});
      nvn::register_command_entries();
      image_mapping_cases(gpu);
      whole_attachment_views(gpu);
      format_twins();
      initial_image(gpu);
      selected_views(gpu);
      replay_images(gpu, true);
      replay_images(gpu, false);
      replay_images(gpu, false, true);
      replay_images(gpu, false, false, true);
      feedback_images(gpu, false);
      feedback_images(gpu, true);
      std::puts("Raw sRGB image execution passed");
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
