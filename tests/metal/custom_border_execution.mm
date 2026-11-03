// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic probe for shader-side custom border
// sampling. The CPU model resolves every spatial and mip tap separately.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline/msl/custom_border.h"
#include "texture_format.h"

#include <armigo/graphics/texture.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace graphics = armigo::graphics;
namespace platform = armigo::platform;
namespace border_msl = armigo::shader::pipeline::msl;

unsigned failures = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c);                     \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

using Color = std::array<float, 4>;
using Swizzle = std::array<std::uint32_t, 4>;
constexpr Swizzle kIdentity{0, 1, 2, 3};
constexpr std::uint32_t kLevels = 3;

struct alignas(16) CustomBorderSupport {
  MTLResourceID coverage;
  std::uint64_t reserved;
  float color[4];
  std::uint32_t swizzle[4];
  std::uint32_t enabled;
  std::uint32_t padding[3];
};
static_assert(sizeof(CustomBorderSupport) == 64);
static_assert(offsetof(CustomBorderSupport, coverage) == 0);
static_assert(offsetof(CustomBorderSupport, reserved) == 8);
static_assert(offsetof(CustomBorderSupport, color) == 16);
static_assert(offsetof(CustomBorderSupport, swizzle) == 32);
static_assert(offsetof(CustomBorderSupport, enabled) == 48);
static_assert(offsetof(CustomBorderSupport, padding) == 52);

struct alignas(8) ExplicitProbe {
  float coordinates[2];
  float lod;
  std::uint32_t padding;
};
static_assert(sizeof(ExplicitProbe) == 16);

struct alignas(8) GradientProbe {
  float coordinates[2];
  float dfdx[2];
  float dfdy[2];
};
static_assert(sizeof(GradientProbe) == 24);

struct TextureSet {
  id<MTLTexture> allocation = nil;
  id<MTLTexture> source = nil;
  id<MTLTexture> coverage = nil;
  bool srgb = false;
  std::uint32_t channels = 4;
  std::array<std::vector<std::array<std::uint8_t, 4>>, kLevels> texels;
};

float srgb_to_linear(float value) {
  return value <= 0.04045f ? value / 12.92f
                           : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

Color decode_texel(const TextureSet &texture, std::uint32_t level,
                   std::uint32_t x, std::uint32_t y) {
  const std::uint32_t width = 4u >> level;
  const auto &encoded = texture.texels[level][y * width + x];
  Color result{};
  for (unsigned component = 0; component < 4; ++component) {
    if (component >= texture.channels) {
      result[component] = component == 3 ? 1.0f : 0.0f;
      continue;
    }
    result[component] = static_cast<float>(encoded[component]) / 255.0f;
    if (texture.srgb && component < 3) {
      result[component] = srgb_to_linear(result[component]);
    }
  }
  return result;
}

Color mix(Color left, const Color &right, float weight) {
  for (unsigned component = 0; component < 4; ++component) {
    left[component] += (right[component] - left[component]) * weight;
  }
  return left;
}

struct LevelSample {
  Color color{};
  float coverage = 0.0f;
};

LevelSample sample_level(const TextureSet &texture, std::uint32_t level,
                         std::array<float, 2> coordinates, bool linear,
                         const Color &outside) {
  const int width = 4 >> level;
  const int height = 4 >> level;
  const double px = static_cast<double>(coordinates[0]) * width - 0.5;
  const double py = static_cast<double>(coordinates[1]) * height - 0.5;
  const auto tap = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= width || y >= height) {
      Color formatted = outside;
      for (std::uint32_t component = texture.channels; component < 4;
           ++component) {
        formatted[component] = component == 3 ? 1.0f : 0.0f;
      }
      return LevelSample{formatted, 0.0f};
    }
    return LevelSample{decode_texel(texture, level,
                                    static_cast<std::uint32_t>(x),
                                    static_cast<std::uint32_t>(y)),
                       1.0f};
  };
  if (!linear) {
    return tap(static_cast<int>(std::floor(px + 0.5)),
               static_cast<int>(std::floor(py + 0.5)));
  }
  const int x0 = static_cast<int>(std::floor(px));
  const int y0 = static_cast<int>(std::floor(py));
  const float fx = static_cast<float>(px - x0);
  const float fy = static_cast<float>(py - y0);
  const auto top_left = tap(x0, y0);
  const auto top_right = tap(x0 + 1, y0);
  const auto bottom_left = tap(x0, y0 + 1);
  const auto bottom_right = tap(x0 + 1, y0 + 1);
  LevelSample result;
  result.color = mix(mix(top_left.color, top_right.color, fx),
                     mix(bottom_left.color, bottom_right.color, fx), fy);
  const float top_coverage =
      top_left.coverage + (top_right.coverage - top_left.coverage) * fx;
  const float bottom_coverage =
      bottom_left.coverage +
      (bottom_right.coverage - bottom_left.coverage) * fx;
  result.coverage = top_coverage + (bottom_coverage - top_coverage) * fy;
  return result;
}

LevelSample sample_mips(const TextureSet &texture,
                        std::array<float, 2> coordinates, float lod,
                        bool linear, graphics::SamplerMipFilter mip_filter,
                        const Color &outside) {
  lod = std::clamp(lod, 0.0f, static_cast<float>(kLevels - 1));
  if (mip_filter != graphics::SamplerMipFilter::Linear) {
    const auto level = mip_filter == graphics::SamplerMipFilter::Nearest
                           ? static_cast<std::uint32_t>(std::floor(lod + 0.5f))
                           : 0u;
    return sample_level(texture, level, coordinates, linear, outside);
  }
  const auto lower = static_cast<std::uint32_t>(std::floor(lod));
  const auto upper = std::min(lower + 1, kLevels - 1);
  const float fraction = lod - static_cast<float>(lower);
  const auto low = sample_level(texture, lower, coordinates, linear, outside);
  const auto high = sample_level(texture, upper, coordinates, linear, outside);
  return {mix(low.color, high.color, fraction),
          low.coverage + (high.coverage - low.coverage) * fraction};
}

Color border_color(const graphics::SamplerDescriptor &sampler,
                   bool srgb_texture) {
  Color result = sampler.border_color;
  if (sampler.srgb_border_conversion && srgb_texture) {
    for (unsigned component = 0; component < 3; ++component) {
      result[component] = srgb_to_linear(
          static_cast<float>(sampler.srgb_border_color[component]) / 255.0f);
    }
  }
  return result;
}

Color apply_swizzle(const Color &value, const Swizzle &swizzle) {
  const std::array<float, 6> channels{value[0], value[1], value[2],
                                      value[3], 0.0f,     1.0f};
  return {channels[swizzle[0]], channels[swizzle[1]], channels[swizzle[2]],
          channels[swizzle[3]]};
}

float gradient_lod(const GradientProbe &probe) {
  const float dx = std::hypot(probe.dfdx[0] * 4.0f, probe.dfdx[1] * 4.0f);
  const float dy = std::hypot(probe.dfdy[0] * 4.0f, probe.dfdy[1] * 4.0f);
  return std::log2(std::max(dx, dy));
}

Color reference(const TextureSet &texture,
                const graphics::SamplerDescriptor &sampler,
                std::array<float, 2> coordinates, float lod, bool custom,
                const Swizzle &swizzle) {
  const bool linear = sampler.minification == graphics::SamplerFilter::Linear;
  Color border = border_color(sampler, texture.srgb);
  if (custom) {
    for (std::uint32_t component = texture.channels; component < 4;
         ++component) {
      border[component] = 0.0f;
    }
  }
  const Color outside = custom ? Color{} : border;
  auto sampled = sample_mips(texture, coordinates, lod, linear,
                             sampler.mip_filter, outside);
  if (custom) {
    for (unsigned component = 0; component < 4; ++component) {
      sampled.color[component] += (1.0f - sampled.coverage) * border[component];
    }
    return apply_swizzle(sampled.color, swizzle);
  }
  return sampled.color;
}

TextureSet make_texture(id<MTLDevice> device, bool srgb,
                        std::uint32_t channels = 4) {
  TextureSet result;
  result.srgb = srgb;
  result.channels = channels;
  const MTLPixelFormat format = srgb            ? MTLPixelFormatRGBA8Unorm_sRGB
                                : channels == 1 ? MTLPixelFormatR8Unorm
                                : channels == 2 ? MTLPixelFormatRG8Unorm
                                                : MTLPixelFormatRGBA8Unorm;
  auto *descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:4
                                                        height:4
                                                     mipmapped:YES];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
  result.allocation = [device newTextureWithDescriptor:descriptor];
  result.source = [result.allocation newTextureViewWithPixelFormat:format];
  auto *coverage_descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                   width:4
                                  height:4
                               mipmapped:YES];
  coverage_descriptor.storageMode = MTLStorageModeShared;
  coverage_descriptor.usage = MTLTextureUsageShaderRead;
  result.coverage = [device newTextureWithDescriptor:coverage_descriptor];
  CHECK(result.allocation != nil && result.source != nil &&
        result.coverage != nil);
  for (std::uint32_t level = 0; level < kLevels; ++level) {
    const std::uint32_t width = 4u >> level;
    const std::uint32_t height = 4u >> level;
    auto &texels = result.texels[level];
    texels.resize(width * height);
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        texels[y * width + x] = {
            static_cast<std::uint8_t>(24 + level * 30 + x * 40 + y * 15),
            static_cast<std::uint8_t>(35 + level * 24 + x * 12 + y * 42),
            static_cast<std::uint8_t>(55 + level * 32 + x * 28 + y * 9),
            static_cast<std::uint8_t>(96 + level * 20 + x * 20 + y * 11)};
      }
    }
    std::vector<std::uint8_t> upload(width * height * channels);
    for (std::size_t texel = 0; texel < texels.size(); ++texel) {
      for (std::uint32_t component = 0; component < channels; ++component) {
        upload[texel * channels + component] = texels[texel][component];
      }
    }
    [result.allocation replaceRegion:MTLRegionMake2D(0, 0, width, height)
                         mipmapLevel:level
                           withBytes:upload.data()
                         bytesPerRow:width * channels];
    const std::vector<std::uint8_t> coverage(width * height, 255);
    [result.coverage replaceRegion:MTLRegionMake2D(0, 0, width, height)
                       mipmapLevel:level
                         withBytes:coverage.data()
                       bytesPerRow:width];
  }
  return result;
}

graphics::SamplerDescriptor sampler(graphics::SamplerFilter filter,
                                    graphics::SamplerMipFilter mip_filter,
                                    Color border) {
  graphics::SamplerDescriptor result;
  result.wrap = {graphics::SamplerWrap::ClampToBorder,
                 graphics::SamplerWrap::ClampToBorder,
                 graphics::SamplerWrap::ClampToBorder};
  result.magnification = filter;
  result.minification = filter;
  result.mip_filter = mip_filter;
  result.min_lod = 0.0f;
  result.max_lod = 2.0f;
  result.border_color = border;
  return result;
}

const std::string kSource = std::string(R"metal(
#include <metal_stdlib>
using namespace metal;
)metal") + std::string(border_msl::custom_border_msl) +
                            R"metal(
struct ExplicitProbe { float2 coordinates; float lod; uint padding; };
struct GradientProbe { float2 coordinates; float2 dfdx; float2 dfdy; };

inline float4 finish_border(float4 black, float coverage,
                            const device ArmigoCustomBorder& support) {
    return support.enabled != 0
        ? armigo_custom_border(black, coverage, support) : black;
}

kernel void explicit_border(
    texture2d<float, access::sample> source [[texture(0)]],
    sampler state [[sampler(0)]],
    const device ArmigoCustomBorder& support [[buffer(0)]],
    const device ExplicitProbe& probe [[buffer(1)]],
    device float4* output [[buffer(2)]]) {
    const float4 black = source.sample(
        state, probe.coordinates, level(probe.lod));
    const float coverage = support.coverage.sample(
        state, probe.coordinates, level(probe.lod)).x;
    output[0] = finish_border(black, coverage, support);
}

kernel void gradient_border(
    texture2d<float, access::sample> source [[texture(0)]],
    sampler state [[sampler(0)]],
    const device ArmigoCustomBorder& support [[buffer(0)]],
    const device GradientProbe& probe [[buffer(1)]],
    device float4* output [[buffer(2)]]) {
    const gradient2d gradients(probe.dfdx, probe.dfdy);
    const float4 black = source.sample(
        state, probe.coordinates, gradients);
    const float coverage = support.coverage.sample(
        state, probe.coordinates, gradients).x;
    output[0] = finish_border(black, coverage, support);
}

struct RasterOut {
    float4 position [[position]];
    float2 coordinates;
};
vertex RasterOut border_vertex(uint id [[vertex_id]],
                               constant float4& controls [[buffer(1)]]) {
    const float2 positions[3] = {float2(-1.0, -1.0),
                                 float2(3.0, -1.0),
                                 float2(-1.0, 3.0)};
    RasterOut out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.coordinates = float2(controls.x +
        (positions[id].x + 1.0) * 0.5 * controls.y, controls.z);
    return out;
}
fragment float4 border_fragment(
    RasterOut in [[stage_in]],
    texture2d<float, access::sample> source [[texture(0)]],
    sampler state [[sampler(0)]],
    const device ArmigoCustomBorder& support [[buffer(0)]]) {
    const float4 black = source.sample(state, in.coordinates);
    const float coverage = support.coverage.sample(
        state, in.coordinates).x;
    return finish_border(black, coverage, support);
}

struct FlatColorOut {
    float4 position [[position]];
    float4 color;
};

// The custom-border sample itself runs in the vertex stage at one shared
// probe coordinate, so every vertex emits the same corrected colour and the
// interpolation cannot distort it.
vertex FlatColorOut border_sample_vertex(
    uint id [[vertex_id]],
    texture2d<float, access::sample> source [[texture(0)]],
    sampler state [[sampler(0)]],
    const device ArmigoCustomBorder& support [[buffer(0)]],
    constant float4& controls [[buffer(1)]]) {
    const float2 positions[3] = {float2(-1.0, -1.0),
                                 float2(3.0, -1.0),
                                 float2(-1.0, 3.0)};
    const float2 coordinates(controls.x, controls.z);
    const float4 black = source.sample(
        state, coordinates, level(controls.w));
    const float coverage = support.coverage.sample(
        state, coordinates, level(controls.w)).x;
    FlatColorOut out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.color = finish_border(black, coverage, support);
    return out;
}

fragment float4 border_passthrough_fragment(
    FlatColorOut in [[stage_in]]) {
    return in.color;
}
)metal";

id<MTLComputePipelineState> compute_pipeline(id<MTL4Compiler> compiler,
                                             id<MTLLibrary> library,
                                             NSString *name) {
  MTL4LibraryFunctionDescriptor *function = [MTL4LibraryFunctionDescriptor new];
  function.library = library;
  function.name = name;
  MTL4ComputePipelineDescriptor *descriptor =
      [MTL4ComputePipelineDescriptor new];
  descriptor.computeFunctionDescriptor = function;
  NSError *error = nil;
  id<MTLComputePipelineState> result =
      [compiler newComputePipelineStateWithDescriptor:descriptor
                                  compilerTaskOptions:nil
                                                error:&error];
  if (!result) {
    std::fprintf(stderr, "Custom border compute pipeline failed: %s\n",
                 error.localizedDescription.UTF8String);
  }
  return result;
}

id<MTLRenderPipelineState> render_pipeline(id<MTL4Compiler> compiler,
                                           id<MTLLibrary> library,
                                           NSString *vertex_name = @"border_vertex",
                                           NSString *fragment_name = @"border_fragment") {
  MTL4LibraryFunctionDescriptor *vertex = [MTL4LibraryFunctionDescriptor new];
  vertex.library = library;
  vertex.name = vertex_name;
  MTL4LibraryFunctionDescriptor *fragment = [MTL4LibraryFunctionDescriptor new];
  fragment.library = library;
  fragment.name = fragment_name;
  MTL4RenderPipelineDescriptor *descriptor = [MTL4RenderPipelineDescriptor new];
  descriptor.vertexFunctionDescriptor = vertex;
  descriptor.fragmentFunctionDescriptor = fragment;
  descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
  descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
  NSError *error = nil;
  id<MTLRenderPipelineState> result =
      [compiler newRenderPipelineStateWithDescriptor:descriptor
                                 compilerTaskOptions:nil
                                               error:&error];
  if (!result) {
    std::fprintf(stderr, "Custom border render pipeline failed: %s\n",
                 error.localizedDescription.UTF8String);
  }
  return result;
}

struct Case {
  const char *name;
  TextureSet *texture;
  graphics::SamplerDescriptor sampler;
  bool custom;
  bool gradient;
  ExplicitProbe explicit_probe{};
  GradientProbe gradient_probe{};
  Swizzle swizzle = kIdentity;
  Color expected{};
  id<MTLSamplerState> state = nil;
  id<MTLBuffer> support = nil;
  id<MTLBuffer> probe = nil;
  id<MTLBuffer> output = nil;
};

bool prepare_case(id<MTLDevice> device, Case &item) {
  const auto mapped =
      platform::metal_sampler_properties(item.sampler, item.texture->srgb);
  const auto *properties =
      std::get_if<platform::MetalSamplerProperties>(&mapped);
  CHECK(properties != nullptr);
  if (!properties)
    return false;
  CHECK(properties->custom_border == item.custom);
  item.state =
      [device newSamplerStateWithDescriptor:platform::metal_sampler_descriptor(
                                                *properties)];
  CustomBorderSupport support{};
  support.coverage = item.texture->coverage.gpuResourceID;
  Color effective =
      platform::metal_custom_border_color(item.sampler, item.texture->srgb);
  for (std::uint32_t component = item.texture->channels; component < 4;
       ++component) {
    effective[component] = 0.0f;
  }
  std::copy(effective.begin(), effective.end(), support.color);
  std::copy(item.swizzle.begin(), item.swizzle.end(), support.swizzle);
  support.enabled = item.custom ? 1u : 0u;
  item.support = [device newBufferWithBytes:&support
                                     length:sizeof(support)
                                    options:MTLResourceStorageModeShared];
  if (item.gradient) {
    item.probe = [device newBufferWithBytes:&item.gradient_probe
                                     length:sizeof(item.gradient_probe)
                                    options:MTLResourceStorageModeShared];
    float lod = gradient_lod(item.gradient_probe) + item.sampler.lod_bias;
    lod = std::clamp(lod, item.sampler.min_lod, item.sampler.max_lod);
    item.expected = reference(*item.texture, item.sampler,
                              {item.gradient_probe.coordinates[0],
                               item.gradient_probe.coordinates[1]},
                              lod, item.custom, item.swizzle);
  } else {
    item.probe = [device newBufferWithBytes:&item.explicit_probe
                                     length:sizeof(item.explicit_probe)
                                    options:MTLResourceStorageModeShared];
    const float lod = std::clamp(item.explicit_probe.lod, item.sampler.min_lod,
                                 item.sampler.max_lod);
    item.expected = reference(*item.texture, item.sampler,
                              {item.explicit_probe.coordinates[0],
                               item.explicit_probe.coordinates[1]},
                              lod, item.custom, item.swizzle);
  }
  item.output = [device newBufferWithLength:sizeof(Color)
                                    options:MTLResourceStorageModeShared];
  CHECK(item.state != nil && item.support != nil && item.probe != nil &&
        item.output != nil);
  return item.state && item.support && item.probe && item.output;
}

bool submit(id<MTL4CommandQueue> queue, id<MTL4CommandBuffer> commands,
            NSArray *retained) {
  [commands endCommandBuffer];
  dispatch_semaphore_t completed = dispatch_semaphore_create(0);
  __block NSError *execution_error = nil;
  MTL4CommitOptions *options = [MTL4CommitOptions new];
  [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
    static_cast<void>(retained.count);
    execution_error = feedback.error;
    dispatch_semaphore_signal(completed);
  }];
  id<MTL4CommandBuffer> submitted[] = {commands};
  [queue commit:submitted count:1 options:options];
  if (dispatch_semaphore_wait(
          completed, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) !=
      0) {
    std::fputs("Custom border execution timed out.\n", stderr);
    return false;
  }
  if (execution_error) {
    std::fprintf(stderr, "Custom border execution failed: %s\n",
                 execution_error.localizedDescription.UTF8String);
    return false;
  }
  return true;
}

bool compare(const char *name, const float *actual, const Color &expected,
             float tolerance = 0.006f) {
  bool matched = true;
  for (unsigned component = 0; component < 4; ++component) {
    if (std::fabs(actual[component] - expected[component]) > tolerance) {
      std::fprintf(stderr, "%s component %u: expected %.7f, got %.7f\n", name,
                   component, expected[component], actual[component]);
      ++failures;
      matched = false;
    }
  }
  return matched;
}

bool run_compute(id<MTLDevice> device, id<MTL4Compiler> compiler,
                 id<MTL4CommandQueue> queue, id<MTLLibrary> library,
                 TextureSet &linear, TextureSet &srgb, TextureSet &red,
                 TextureSet &red_green) {
  const Color custom_color{0.17f, 0.61f, 0.83f, 0.37f};
  auto linear_sampler =
      sampler(graphics::SamplerFilter::Linear,
              graphics::SamplerMipFilter::Linear, custom_color);
  auto nearest_sampler =
      sampler(graphics::SamplerFilter::Nearest,
              graphics::SamplerMipFilter::Nearest, custom_color);
  const auto explicit_case = [](const char *name, TextureSet &texture,
                                graphics::SamplerDescriptor state, bool custom,
                                float u, float v, float lod,
                                Swizzle swizzle = kIdentity) {
    Case result{name, &texture, std::move(state), custom, false};
    result.explicit_probe = {{u, v}, lod, 0};
    result.swizzle = swizzle;
    return result;
  };
  const auto gradient_case = [](const char *name, TextureSet &texture,
                                graphics::SamplerDescriptor state, float u,
                                float v, float dx, float dy) {
    Case result{name, &texture, std::move(state), true, true};
    result.gradient_probe = {{u, v}, {dx, 0.0f}, {0.0f, dy}};
    return result;
  };
  std::vector<Case> cases;
  cases.push_back(explicit_case("linear corner partial mip 0", linear,
                                linear_sampler, true, -0.0625f, -0.0625f,
                                0.0f));
  cases.push_back(explicit_case("linear in range mip 1", linear, linear_sampler,
                                true, 0.25f, 0.75f, 1.0f));
  cases.push_back(explicit_case("linear fully outside", linear, linear_sampler,
                                true, -0.75f, 1.6f, 0.0f));
  cases.push_back(explicit_case("linear fractional mip footprint", linear,
                                linear_sampler, true, -0.0625f, -0.0625f,
                                0.35f));
  cases.push_back(explicit_case("nearest mip 1", linear, nearest_sampler, true,
                                0.74f, 0.26f, 0.75f));
  cases.push_back(explicit_case("nearest fully outside", linear,
                                nearest_sampler, true, 1.4f, -0.4f, 0.0f));

  auto biased = linear_sampler;
  biased.lod_bias = 0.75f;
  cases.push_back(gradient_case("gradient LOD bias", linear, biased, -0.0625f,
                                0.35f, std::exp2(-0.5f) / 4.0f,
                                std::exp2(-0.5f) / 4.0f));
  auto minimum_clamped = linear_sampler;
  minimum_clamped.min_lod = 1.0f;
  cases.push_back(gradient_case("gradient minimum LOD clamp", linear,
                                minimum_clamped, 0.35f, -0.125f, 0.25f, 0.25f));
  auto maximum_clamped = linear_sampler;
  maximum_clamped.max_lod = 0.5f;
  cases.push_back(gradient_case(
      "gradient maximum LOD clamp", linear, maximum_clamped, 1.0625f, 0.4f,
      std::exp2(1.4f) / 4.0f, std::exp2(1.4f) / 4.0f));

  auto srgb_converted = linear_sampler;
  srgb_converted.border_color = {0.91f, 0.19f, 0.63f, 0.42f};
  srgb_converted.srgb_border_color = {17, 129, 231};
  srgb_converted.srgb_border_conversion = true;
  cases.push_back(explicit_case("sRGB border conversion and alpha", srgb,
                                srgb_converted, true, -0.0625f, 0.35f, 0.0f));
  auto srgb_raw = srgb_converted;
  srgb_raw.srgb_border_conversion = false;
  cases.push_back(explicit_case("sRGB conversion disabled", srgb, srgb_raw,
                                true, -0.0625f, 0.35f, 0.0f));
  cases.push_back(explicit_case("sRGB flag on linear texture", linear,
                                srgb_converted, true, -0.0625f, 0.35f, 0.0f));
  cases.push_back(explicit_case("post-correction swizzle constants", linear,
                                linear_sampler, true, -0.0625f, 0.35f, 0.0f,
                                Swizzle{2, 0, 4, 5}));

  cases.push_back(
      explicit_case("native transparent black", linear,
                    sampler(graphics::SamplerFilter::Linear,
                            graphics::SamplerMipFilter::Linear, {0, 0, 0, 0}),
                    false, -0.0625f, -0.0625f, 0.0f));
  cases.push_back(
      explicit_case("native opaque black", linear,
                    sampler(graphics::SamplerFilter::Linear,
                            graphics::SamplerMipFilter::Linear, {0, 0, 0, 1}),
                    false, -0.0625f, -0.0625f, 0.0f));
  cases.push_back(
      explicit_case("native opaque white", linear,
                    sampler(graphics::SamplerFilter::Linear,
                            graphics::SamplerMipFilter::Linear, {1, 1, 1, 1}),
                    false, -0.0625f, -0.0625f, 0.0f));
  cases.push_back(
      explicit_case("native R8 partial footprint", red,
                    sampler(graphics::SamplerFilter::Linear,
                            graphics::SamplerMipFilter::Linear, {0, 0, 0, 0}),
                    false, -0.0625f, 0.375f, 0.0f));
  cases.push_back(
      explicit_case("native R8 fully outside", red,
                    sampler(graphics::SamplerFilter::Linear,
                            graphics::SamplerMipFilter::Linear, {0, 0, 0, 0}),
                    false, -0.75f, 0.375f, 0.0f));
  cases.push_back(
      explicit_case("native RG8 partial footprint", red_green,
                    sampler(graphics::SamplerFilter::Linear,
                            graphics::SamplerMipFilter::Linear, {0, 0, 0, 0}),
                    false, -0.0625f, 0.375f, 0.0f));
  cases.push_back(explicit_case("custom R8 missing channels", red,
                                linear_sampler, true, -0.0625f, 0.375f, 0.0f));
  cases.push_back(explicit_case("custom RG8 missing channels", red_green,
                                linear_sampler, true, -0.0625f, 0.375f, 0.0f));
  cases.push_back(explicit_case("custom R8 swizzle defaults", red,
                                linear_sampler, true, -0.0625f, 0.375f, 0.0f,
                                Swizzle{3, 0, 4, 5}));

  id<MTLComputePipelineState> explicit_pipeline =
      compute_pipeline(compiler, library, @"explicit_border");
  id<MTLComputePipelineState> gradient_pipeline =
      compute_pipeline(compiler, library, @"gradient_border");
  if (!explicit_pipeline || !gradient_pipeline)
    return false;
  MTL4ArgumentTableDescriptor *table_descriptor =
      [MTL4ArgumentTableDescriptor new];
  table_descriptor.maxBufferBindCount = 3;
  table_descriptor.maxTextureBindCount = 1;
  table_descriptor.maxSamplerStateBindCount = 1;
  NSError *error = nil;
  id<MTL4ArgumentTable> table =
      [device newArgumentTableWithDescriptor:table_descriptor error:&error];
  MTLResidencySetDescriptor *residency_descriptor =
      [MTLResidencySetDescriptor new];
  residency_descriptor.initialCapacity = 64;
  id<MTLResidencySet> residency =
      [device newResidencySetWithDescriptor:residency_descriptor error:&error];
  id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
  id<MTL4CommandBuffer> commands = [device newCommandBuffer];
  if (!table || !residency || !allocator || !commands)
    return false;
  NSMutableArray *retained = [NSMutableArray
      arrayWithObjects:library, compiler, explicit_pipeline, gradient_pipeline,
                       table, residency, allocator, commands, linear.allocation,
                       linear.source, linear.coverage, srgb.allocation,
                       srgb.source, srgb.coverage, red.allocation, red.source,
                       red.coverage, red_green.allocation, red_green.source,
                       red_green.coverage, nil];
  [residency addAllocation:linear.allocation];
  [residency addAllocation:linear.coverage];
  [residency addAllocation:srgb.allocation];
  [residency addAllocation:srgb.coverage];
  [residency addAllocation:red.allocation];
  [residency addAllocation:red.coverage];
  [residency addAllocation:red_green.allocation];
  [residency addAllocation:red_green.coverage];
  for (auto &item : cases) {
    if (!prepare_case(device, item))
      return false;
    for (id<MTLAllocation> allocation in
         @[ item.support, item.probe, item.output ]) {
      [residency addAllocation:allocation];
      [retained addObject:allocation];
    }
    [retained addObject:item.state];
  }
  [residency commit];
  [commands beginCommandBufferWithAllocator:allocator];
  [commands useResidencySet:residency];
  id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
  if (!encoder)
    return false;
  for (const auto &item : cases) {
    [table setTexture:item.texture->source.gpuResourceID atIndex:0];
    [table setSamplerState:item.state.gpuResourceID atIndex:0];
    [table setAddress:item.support.gpuAddress atIndex:0];
    [table setAddress:item.probe.gpuAddress atIndex:1];
    [table setAddress:item.output.gpuAddress atIndex:2];
    [encoder setComputePipelineState:item.gradient ? gradient_pipeline
                                                   : explicit_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  [encoder endEncoding];
  if (!submit(queue, commands, retained))
    return false;
  for (const auto &item : cases) {
    compare(item.name, static_cast<const float *>(item.output.contents),
            item.expected);
  }
  std::printf("Ran %zu explicit/gradient custom border GPU probes.\n",
              cases.size());
  return true;
}

bool run_fragment(id<MTLDevice> device, id<MTL4Compiler> compiler,
                  id<MTL4CommandQueue> queue, id<MTLLibrary> library,
                  TextureSet &texture, bool sample_in_vertex = false) {
  constexpr NSUInteger width = 4;
  constexpr NSUInteger height = 4;
  auto state_description =
      sampler(graphics::SamplerFilter::Linear,
              graphics::SamplerMipFilter::Linear, {0.17f, 0.61f, 0.83f, 0.37f});
  const auto mapped =
      platform::metal_sampler_properties(state_description, texture.srgb);
  const auto *properties =
      std::get_if<platform::MetalSamplerProperties>(&mapped);
  CHECK(properties && properties->custom_border);
  if (!properties)
    return false;
  id<MTLSamplerState> state =
      [device newSamplerStateWithDescriptor:platform::metal_sampler_descriptor(
                                                *properties)];
  CustomBorderSupport support{};
  support.coverage = texture.coverage.gpuResourceID;
  const Color effective =
      platform::metal_custom_border_color(state_description, texture.srgb);
  std::copy(effective.begin(), effective.end(), support.color);
  std::copy(kIdentity.begin(), kIdentity.end(), support.swizzle);
  support.enabled = 1;
  id<MTLBuffer> support_buffer =
      [device newBufferWithBytes:&support
                          length:sizeof(support)
                         options:MTLResourceStorageModeShared];
  // The vertex probe shares the buffer layout but samples a footprint that
  // straddles the texture edge at the explicit LOD in controls.w.
  const std::array<float, 4> controls =
      sample_in_vertex ? std::array<float, 4>{0.1f, 1.5f, 0.35f, 0.0f}
                       : std::array<float, 4>{-0.35f, 1.5f, 0.35f, 0.0f};
  id<MTLBuffer> control_buffer =
      [device newBufferWithBytes:controls.data()
                          length:sizeof(controls)
                         options:MTLResourceStorageModeShared];
  auto *target_descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:width
                                  height:height
                               mipmapped:NO];
  target_descriptor.storageMode = MTLStorageModePrivate;
  target_descriptor.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> target = [device newTextureWithDescriptor:target_descriptor];
  id<MTLBuffer> readback =
      [device newBufferWithLength:width * height * sizeof(Color)
                          options:MTLResourceStorageModeShared];
  id<MTLRenderPipelineState> pipeline = sample_in_vertex
      ? render_pipeline(compiler, library, @"border_sample_vertex",
                        @"border_passthrough_fragment")
      : render_pipeline(compiler, library);
  MTL4ArgumentTableDescriptor *table_descriptor =
      [MTL4ArgumentTableDescriptor new];
  table_descriptor.maxBufferBindCount = 2;
  table_descriptor.maxTextureBindCount = 1;
  table_descriptor.maxSamplerStateBindCount = 1;
  NSError *error = nil;
  id<MTL4ArgumentTable> table =
      [device newArgumentTableWithDescriptor:table_descriptor error:&error];
  MTLResidencySetDescriptor *residency_descriptor =
      [MTLResidencySetDescriptor new];
  residency_descriptor.initialCapacity = 7;
  id<MTLResidencySet> residency =
      [device newResidencySetWithDescriptor:residency_descriptor error:&error];
  id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
  id<MTL4CommandBuffer> commands = [device newCommandBuffer];
  if (!state || !support_buffer || !control_buffer || !target || !readback ||
      !pipeline || !table || !residency || !allocator || !commands) {
    return false;
  }
  [table setAddress:support_buffer.gpuAddress atIndex:0];
  [table setAddress:control_buffer.gpuAddress atIndex:1];
  [table setTexture:texture.source.gpuResourceID atIndex:0];
  [table setSamplerState:state.gpuResourceID atIndex:0];
  for (id<MTLAllocation> allocation in @[
         texture.allocation, texture.coverage, support_buffer, control_buffer,
         target, readback
       ]) {
    [residency addAllocation:allocation];
  }
  [residency commit];
  [commands beginCommandBufferWithAllocator:allocator];
  [commands useResidencySet:residency];
  MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTL4RenderCommandEncoder> render =
      [commands renderCommandEncoderWithDescriptor:pass];
  if (!render)
    return false;
  [render setRenderPipelineState:pipeline];
  [render setArgumentTable:table
                  atStages:MTLRenderStageVertex | MTLRenderStageFragment];
  [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render endEncoding];
  id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
  if (!copy)
    return false;
  [copy barrierAfterQueueStages:MTLStageFragment
                   beforeStages:MTLStageBlit
              visibilityOptions:MTL4VisibilityOptionDevice];
  [copy copyFromTexture:target
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(width, height, 1)
                      toBuffer:readback
             destinationOffset:0
        destinationBytesPerRow:width * sizeof(Color)
      destinationBytesPerImage:0];
  [copy endEncoding];
  NSArray *retained = @[
    library, compiler, state, support_buffer, control_buffer, target, readback,
    pipeline, table, residency, allocator, texture.allocation, texture.source,
    texture.coverage
  ];
  if (!submit(queue, commands, retained))
    return false;
  const auto *actual = static_cast<const Color *>(readback.contents);
  const float implicit_lod = std::log2(controls[1]);
  Color vertex_expected{};
  if (sample_in_vertex) {
    // The vertex probe samples one shared coordinate at the explicit LOD in
    // controls.w, so every pixel must equal that single corrected sample.
    vertex_expected = reference(texture, state_description,
                                {controls[0], controls[2]}, controls[3], true,
                                kIdentity);
  }
  for (NSUInteger y = 0; y < height; ++y) {
    for (NSUInteger x = 0; x < width; ++x) {
      if (sample_in_vertex) {
        char name[96]{};
        std::snprintf(name, sizeof(name), "vertex LOD pixel (%lu,%lu)", x, y);
        compare(name, actual[y * width + x].data(), vertex_expected, 0.008f);
        continue;
      }
      const float u =
          controls[0] + (static_cast<float>(x) + 0.5f) * controls[1] / width;
      const Color expected =
          reference(texture, state_description, {u, controls[2]}, implicit_lod,
                    true, kIdentity);
      char name[96]{};
      std::snprintf(name, sizeof(name), "implicit derivative pixel (%lu,%lu)",
                    x, y);
      compare(name, actual[y * width + x].data(), expected, 0.008f);
    }
  }
  std::printf(sample_in_vertex
                  ? "Ran %lu explicit-LOD custom border vertex probes.\n"
                  : "Ran %lu implicit-derivative custom border fragment probes.\n",
              width * height);
  return true;
}

bool run(id<MTLDevice> device) {
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion4_0;
  options.mathMode = MTLMathModeSafe;
  NSError *error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:kSource.c_str()]
                   options:options
                     error:&error];
  MTL4CompilerDescriptor *compiler_descriptor = [MTL4CompilerDescriptor new];
  id<MTL4Compiler> compiler =
      [device newCompilerWithDescriptor:compiler_descriptor error:&error];
  id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
  if (!library || !compiler || !queue) {
    std::fprintf(stderr, "Custom border compiler setup failed: %s\n",
                 error.localizedDescription.UTF8String);
    return false;
  }
  auto linear = make_texture(device, false);
  auto srgb = make_texture(device, true);
  auto red = make_texture(device, false, 1);
  auto red_green = make_texture(device, false, 2);
  return run_compute(device, compiler, queue, library, linear, srgb, red,
                     red_green) &&
         run_fragment(device, compiler, queue, library, linear) &&
         run_fragment(device, compiler, queue, library, linear, true);
}

} // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] ||
        !device.hasUnifiedMemory) {
      std::fputs("Custom border GPU tests require unified-memory Metal 4.\n",
                 stderr);
      return 1;
    }
    const bool completed = run(device);
    std::printf("Custom border GPU failures: %u\n", failures);
    return completed && failures == 0 ? 0 : 1;
  }
}
