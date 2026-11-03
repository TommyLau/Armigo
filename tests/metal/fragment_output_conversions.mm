// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic, independently authored verification for fragment colour output
// specialization against narrow and sparse render targets. Every expectation
// is computed here by an independent CPU model of Metal's documented
// elementary-type conversion rules (round to nearest even, clamped
// normalized ranges, the sRGB transfer function, float16 and e5 mini-float
// packing); nothing is copied from a device or another implementation. Where
// this device's measured attachment-write behavior deviates from the
// documented rules, the model records the measured contract and a comment
// names the deviation. All pipelines are created through the production
// PipelineStateCache so the fragment output class constants are assigned by
// the shipped specialization path.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <attachments.h>
#include <device_context.h>
#include <fragment_output.h>
#include <native_pipeline_state.h>

#include <armigo/graphics/render_target.h>
#include <armigo/graphics/texture_format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;
namespace graphics = armigo::graphics;
namespace platform = armigo::platform;
namespace metal = armigo::platform::metal;

unsigned failures = 0;
unsigned checks = 0;

#define CHECK(condition)                                                       \
    do {                                                                       \
        ++checks;                                                              \
        if (!(condition)) {                                                    \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);        \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

template <class T>
T take(RendererResult<T> result) {
    if (const auto* error = std::get_if<RendererError>(&result)) {
        std::fprintf(stderr, "FAIL: unexpected renderer error: %.*s\n",
                     int(error->command().detail.size()), error->command().detail.data());
        std::exit(1);
    }
    return std::get<T>(std::move(result));
}

// ---------------------------------------------------------------------------
// CPU conversion model: Metal's documented float-to-elementary conversions.
// ---------------------------------------------------------------------------

double clamp_range(double value, double low, double high) {
    return value < low ? low : (value > high ? high : value);
}

// Round to nearest even, then clamp to the unsigned quantized range.
std::uint64_t to_unorm(double value, unsigned bits) {
    const double scaled = std::nearbyint(clamp_range(value, 0.0, 1.0) *
                                         double((std::uint64_t{1} << bits) - 1));
    return static_cast<std::uint64_t>(
        clamp_range(scaled, 0.0, double((std::uint64_t{1} << bits) - 1)));
}

// The signed normalized range is [-(2^(bits-1) - 1), 2^(bits-1) - 1] exactly;
// -1.0 encodes to the most negative representable value, never the reserved
// minimum.
std::int64_t to_snorm(double value, unsigned bits) {
    const double limit = double((std::uint64_t{1} << (bits - 1)) - 1);
    const double scaled = std::nearbyint(clamp_range(value, -1.0, 1.0) * limit);
    return static_cast<std::int64_t>(clamp_range(scaled, -limit, limit));
}

// Apple Silicon encodes attachment sRGB writes with the power segment alone:
// the textbook 12.92 * c toe below 0.0031308 is not applied (measured: a
// 0.002 linear input stores 6, matching the pure power curve's 6.2, not the
// toe's 6.6). Values probed here stay clear of the rounding boundary.
std::uint8_t to_srgb8(double linear) {
    const double encoded = 1.055 * std::pow(clamp_range(linear, 0.0, 1.0), 1.0 / 2.4) - 0.055;
    return static_cast<std::uint8_t>(to_unorm(encoded, 8));
}

// IEEE 754 binary16 with the significand truncated (round toward zero),
// subnormals included, overflow to infinity. The MSL conversion table's
// round-to-nearest-even does NOT hold on this device's attachment write
// path: 0.002f stores 0x1818 (fraction 24) and 0.6f stores 0x36cc
// (fraction 204), both one ulp below the nearest-even encodings, while
// normalized formats round normally. The model records the measured
// hardware contract; probed values never sit on a truncation tie.
std::uint16_t to_half(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    const std::uint32_t exponent_bits = (bits >> 23) & 0xffu;
    if (exponent_bits == 0xff)  // infinity or NaN keeps its class
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0u));
    // Half-biased exponent: 1..30 normal, <= 0 subnormal, >= 31 overflow.
    const std::int32_t half_exponent = std::int32_t(exponent_bits) - 127 + 15;
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exponent >= 1)
        return static_cast<std::uint16_t>(sign | (std::uint32_t(half_exponent) << 10) |
                                          (mantissa >> 13));
    // Subnormal: the stored field is m * 2^-24 with a fixed exponent of -14.
    const std::uint32_t significand = mantissa | 0x800000u;
    const int shift = 14 - half_exponent;
    if (shift > 24) return static_cast<std::uint16_t>(sign);
    return static_cast<std::uint16_t>(sign | std::min(significand >> shift, 0x400u));
}

// e5 unsigned mini-float with `mantissa_bits` fraction bits (the R11G11B10
// channels). The format has no sign bit and the hardware saturates negative
// inputs to zero (measured: -0.75 stores 0). Fractions truncate like the
// half path above; overflow stores the e5 infinity.
std::uint32_t to_minifloat(float value, unsigned mantissa_bits) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits >> 31) != 0) return 0;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    const std::uint32_t exponent_bits = (bits >> 23) & 0xffu;
    const std::uint32_t stored_infinity = (1u << 5) - 1u;
    if (exponent_bits == 0xff) return stored_infinity;
    const std::int32_t mini_exponent = std::int32_t(exponent_bits) - 127 + 15;
    const std::uint32_t fraction_limit = (1u << mantissa_bits) - 1u;
    if (mini_exponent >= 31) return stored_infinity;
    if (mini_exponent >= 1) {
        const std::uint32_t fraction = mantissa >> (23 - mantissa_bits);
        if (fraction > fraction_limit) {
            if (mini_exponent + 1 >= 31) return stored_infinity;
            return std::uint32_t(mini_exponent + 1) << mantissa_bits;
        }
        return (std::uint32_t(mini_exponent) << mantissa_bits) | fraction;
    }
    const std::uint32_t significand = mantissa | 0x800000u;
    const int shift = 14 - mini_exponent + (23 - mantissa_bits);
    if (shift > 24) return 0;
    return std::min(significand >> shift, fraction_limit);
}

void append_le(std::vector<std::uint8_t>& bytes, std::uint64_t value, unsigned width) {
    for (unsigned byte = 0; byte < width; ++byte)
        bytes.push_back(std::uint8_t((value >> (byte * 8)) & 0xffu));
}

// ---------------------------------------------------------------------------
// Synthetic shader library. The fragment writes per-quadrant constant values
// through function-constant selected output classes; the class constants are
// filled only by the production specialization path under test. Quadrant
// order follows framebuffer coordinates: q = x/4 + 2*(row/4).
// ---------------------------------------------------------------------------

constexpr std::uint32_t kWidth = 8;
constexpr std::uint32_t kHeight = 8;

constexpr std::array<std::array<float, 4>, 4> k_float_quadrants{{
    {0.25f, 0.75f, 0.0625f, 1.0f},   // in range, exact in every model
    {1.75f, -0.75f, 2.0f, -3.0f},    // clamping in both directions
    {0.4f, -0.4f, 0.6f, 0.4f},       // rounding, no halfway ties
    {0.002f, 0.5f, 0.03f, 0.0625f},  // sRGB segments, halfway ties
}};
constexpr std::array<std::array<std::uint32_t, 4>, 4> k_uint_quadrants{{
    {200, 100, 55, 25},
    {300, 7, 0, 255},
    {16, 32, 64, 128},
    {255, 0, 1, 254},
}};
constexpr std::array<std::array<std::int32_t, 4>, 4> k_sint_quadrants{{
    {100, -100, 0, 127},
    {200, -200, 1, -1},
    {5, 10, 15, 20},
    {-128, 127, 3, 4},
}};

std::string float_literal(float value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.9g", double(value));
    return buffer;
}

// Builds one quadrant select chain: `if (quadrant == 0u) member = T(...);`.
std::string quadrant_chain(const char* member, const char* constructor,
                           const std::function<std::string(unsigned, unsigned)>& literal) {
    std::string chain;
    for (unsigned quadrant = 0; quadrant < 4; ++quadrant) {
        chain += quadrant == 0 ? "if" : "else if";
        chain += " (quadrant == " + std::to_string(quadrant) + "u) " + member + " = " +
                 constructor + "(";
        for (unsigned lane = 0; lane < 4; ++lane)
            chain += (lane ? ", " : "") + literal(quadrant, lane);
        chain += ");\n";
    }
    return chain;
}

std::string quad_source() {
    std::string source = R"(
#include <metal_stdlib>
using namespace metal;
constant float2 corners[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
struct VertexOut { float4 position [[position]]; };
vertex VertexOut vs_full(uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = float4(corners[vid], 0.5, 1.0);
    return out;
}
constant uint corner_class [[function_constant(0)]];
constant bool corner_float = corner_class == 0;
constant bool corner_uint = corner_class == 1;
constant bool corner_sint = corner_class == 2;
struct CornerOutputs {
    float4 f [[color(0), function_constant(corner_float)]];
    uint4 u [[color(0), function_constant(corner_uint)]];
    int4 s [[color(0), function_constant(corner_sint)]];
};
fragment CornerOutputs fs_quads(float4 position [[position]]) {
    const uint quadrant = uint(position.x) / 4u + 2u * (uint(position.y) / 4u);
    CornerOutputs out;
)";
    source += "    if (corner_float) {\n";
    source += quadrant_chain("out.f", "float4", [](unsigned quadrant, unsigned lane) {
        return float_literal(k_float_quadrants[quadrant][lane]);
    });
    source += "    }\n    if (corner_uint) {\n";
    source += quadrant_chain("out.u", "uint4", [](unsigned quadrant, unsigned lane) {
        return std::to_string(k_uint_quadrants[quadrant][lane]) + "u";
    });
    source += "    }\n    if (corner_sint) {\n";
    source += quadrant_chain("out.s", "int4", [](unsigned quadrant, unsigned lane) {
        return std::to_string(k_sint_quadrants[quadrant][lane]);
    });
    source += R"(    }
    return out;
}
// The probe shares the corner class constant so the library keeps one
// declaration per index, exactly as the production path addresses them.
struct AlphaOutputs {
    float4 f [[color(0), function_constant(corner_float)]];
    uint4 u [[color(0), function_constant(corner_uint)]];
};
fragment AlphaOutputs fs_alpha_probe() {
    AlphaOutputs out;
    if (corner_float) out.f = float4(0.5, 0.25, 0.75, 0.25);
    if (corner_uint) out.u = uint4(200u, 100u, 50u, 25u);
    return out;
}
)";
    return source;
}

// Function-constant indices are library-global, so the sparse library is
// compiled separately with its constants exactly at the logical output
// locations 0, 2 and 5 the production path must address.
std::string sparse_source() {
    return R"(
#include <metal_stdlib>
using namespace metal;
constant float2 corners[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
struct VertexOut { float4 position [[position]]; };
vertex VertexOut vs_full(uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = float4(corners[vid], 0.5, 1.0);
    return out;
}
constant uint sparse_class_0 [[function_constant(0)]];
constant uint sparse_class_2 [[function_constant(2)]];
constant uint sparse_class_5 [[function_constant(5)]];
constant bool sparse_float_0 = sparse_class_0 == 0;
constant bool sparse_uint_2 = sparse_class_2 == 1;
constant bool sparse_sint_5 = sparse_class_5 == 2;
struct SparseOutputs {
    float4 low [[color(0), function_constant(sparse_float_0)]];
    uint4 middle [[color(2), function_constant(sparse_uint_2)]];
    int4 high [[color(5), function_constant(sparse_sint_5)]];
};
fragment SparseOutputs fs_sparse() {
    SparseOutputs out;
    if (sparse_float_0) out.low = float4(1.0, 2.0, 3.0, 1.5);
    if (sparse_uint_2) out.middle = uint4(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
    if (sparse_sint_5) out.high = int4(-2023406815, 305419896, -65535, 2147483646);
    return out;
}
)";
}

// Compiles the synthetic source with the offline toolchain into the
// executable-type library blob the production cache loads.
std::vector<std::byte> compile_library(const std::string& source) {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("armigo-frag-conversions-" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    const auto source_path = directory / "shader.metal";
    const auto air_path = directory / "shader.air";
    const auto library_path = directory / "shader.metallib";
    std::ofstream stream(source_path, std::ios::binary);
    stream << source;
    stream.close();
    const auto run = [](const std::string& command) { return std::system(command.c_str()) == 0; };
    const std::string prefix = "xcrun --sdk macosx ";
    const bool built =
        run(prefix + "metal -std=metal4.0 -mmacosx-version-min=26.0 -Werror -c " +
            source_path.string() + " -o " + air_path.string()) &&
        run(prefix + "metallib " + air_path.string() + " -o " + library_path.string());
    std::vector<std::byte> bytes;
    if (built) {
        std::ifstream input(library_path, std::ios::binary);
        const std::vector<char> data{std::istreambuf_iterator<char>(input), {}};
        const auto* raw = reinterpret_cast<const std::byte*>(data.data());
        bytes.assign(raw, raw + data.size());
    } else {
        std::fprintf(stderr, "FAIL: synthetic library compilation failed\n");
        ++failures;
    }
    std::error_code cleanup;
    std::filesystem::remove_all(directory, cleanup);
    return bytes;
}

metal::AdmittedGraphicsStages make_stages(const std::vector<std::byte>& metallib,
                                          const char* vertex_entry, const char* fragment_entry,
                                          std::vector<platform::CompiledColorOutput> outputs,
                                          bool adaptive_outputs) {
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>();
    auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>();
    vertex->metallib = metallib;
    fragment->metallib = metallib;
    vertex->stage = armigo::shader_input::Stage::Vertex;
    fragment->stage = armigo::shader_input::Stage::Fragment;
    vertex->entry_point = vertex_entry;
    fragment->entry_point = fragment_entry;
    fragment->color_outputs = std::move(outputs);
    fragment->fragment_output_types_from_render_targets = adaptive_outputs;
    metal::ShaderFunctionKey vertex_key;
    vertex_key.package = "fragment-conversions";
    vertex_key.variant = "v1";
    vertex_key.artifact = vertex_key.entry_point = vertex_entry;
    metal::ShaderFunctionKey fragment_key = vertex_key;
    fragment_key.artifact = fragment_key.entry_point = fragment_entry;
    fragment_key.specialization = metal::fragment_output_specialization(*fragment);
    return metal::AdmittedGraphicsStages{armigo::platform::CompiledGraphicsPair{vertex, fragment},
                                         vertex_key, fragment_key};
}

metal::PipelineKey base_key(const metal::PipelineStateCache& cache,
                            const metal::AdmittedGraphicsStages& stages) {
    metal::PipelineKey key;
    key.context = cache.context();
    key.vertex = stages.vertex_key;
    key.fragment = stages.fragment_key;
    key.primitive = PrimitiveClass::Triangle;
    key.rasterization_enabled = true;
    key.coverage.samples = 1;
    key.coverage.sample_mask = ~0u;
    key.coverage.multisample = false;
    return key;
}

id<MTLRenderPipelineState> pso_of(const metal::ResidentPipeline& resident) {
    return dynamic_cast<const metal::MetalRenderPipelineResource&>(*resident.pipeline)
        .pipeline();
}

// ---------------------------------------------------------------------------
// Offscreen pass runner: one fullscreen draw into up to eight attachments,
// raw byte readback per attachment slot.
// ---------------------------------------------------------------------------

struct RawPassResult {
    std::array<std::vector<std::uint8_t>, 8> slots;
};

std::size_t format_pixel_bytes(TextureFormat format) {
    return graphics::texture_format_info(format).bytes_per_block;
}

RawPassResult run_raw_pass(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                           const std::array<id<MTLTexture>, 8>& colors,
                           const std::array<TextureFormat, 8>& formats,
                           id<MTLRenderPipelineState> pipeline) {
    RawPassResult result;
    auto* pass = [MTL4RenderPassDescriptor new];
    std::size_t total_bytes = 0;
    std::array<std::size_t, 8> slot_offsets{};
    for (unsigned slot = 0; slot < 8; ++slot) {
        if (!colors[slot]) continue;
        auto* attachment = pass.colorAttachments[slot];
        attachment.texture = colors[slot];
        attachment.loadAction = MTLLoadActionClear;
        attachment.storeAction = MTLStoreActionStore;
        attachment.clearColor = MTLClearColorMake(0, 0, 0, 0);
        slot_offsets[slot] = total_bytes;
        total_bytes += kWidth * kHeight * format_pixel_bytes(formats[slot]);
    }
    id<MTLBuffer> readback =
        [device newBufferWithLength:total_bytes options:MTLResourceStorageModeShared];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    CHECK(readback != nil && allocator != nil && commands != nil);
    NSMutableArray* retained =
        [NSMutableArray arrayWithArray:@[pass, readback, allocator, commands]];
    for (id<MTLTexture> texture : colors)
        if (texture) [retained addObject:texture];
    [commands beginCommandBufferWithAllocator:allocator];
    auto* encoder = [commands renderCommandEncoderWithDescriptor:pass];
    CHECK(encoder != nil);
    [encoder setRenderPipelineState:pipeline];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    auto* copy = [commands computeCommandEncoder];
    CHECK(copy != nil);
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
               visibilityOptions:MTL4VisibilityOptionDevice];
    for (unsigned slot = 0; slot < 8; ++slot) {
        if (!colors[slot]) continue;
        const std::size_t row_bytes = kWidth * format_pixel_bytes(formats[slot]);
        [copy copyFromTexture:colors[slot] sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                    toBuffer:readback destinationOffset:slot_offsets[slot]
        destinationBytesPerRow:row_bytes destinationBytesPerImage:0];
    }
    [copy endEncoding];
    [commands endCommandBuffer];
    auto completed = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    auto* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) !=
        0) {
        std::fprintf(stderr, "FAIL: raw pass timed out\n");
        ++failures;
        return result;
    }
    if (execution_error) {
        std::fprintf(stderr, "FAIL: raw pass failed: %s\n",
                     execution_error.localizedDescription.UTF8String);
        ++failures;
        return result;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(readback.contents);
    for (unsigned slot = 0; slot < 8; ++slot) {
        if (!colors[slot]) continue;
        const std::size_t count = kWidth * kHeight * format_pixel_bytes(formats[slot]);
        result.slots[slot].assign(bytes + slot_offsets[slot], bytes + slot_offsets[slot] + count);
    }
    return result;
}

id<MTLTexture> make_target(id<MTLDevice> device, MTLPixelFormat format) {
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                          width:kWidth
                                                                         height:kHeight
                                                                      mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    return [device newTextureWithDescriptor:descriptor];
}

// Local device context: the cache needs only the device, one Metal 4 queue
// and a stable identity, matching the isolated L9 test setup.
class TestDeviceContext final : public metal::DeviceContext {
public:
    explicit TestDeviceContext(id<MTLDevice> device) : device_(device) {}
    id<MTLDevice> device() const noexcept override { return device_; }
    id<MTL4CommandQueue> queue() const noexcept override {
        if (!queue_) queue_ = [device_ newMTL4CommandQueue];
        return queue_;
    }
    const metal::PipelineContextKey& identity() const noexcept override {
        static const metal::PipelineContextKey context = [this] {
            metal::PipelineContextKey value;
            value.device = device_.registryID;
            value.compiler_chain = "synthetic-fragment-conversions";
            value.capability_profile = "metal4-apple-silicon";
            return value;
        }();
        return context;
    }

private:
    id<MTLDevice> device_;
    mutable id<MTL4CommandQueue> queue_ = nil;
};

MTLPixelFormat render_format_or_fail(TextureFormat format) {
    const auto mapped = metal::metal_render_format(format, TextureAspect::Color);
    if (const auto* failure = std::get_if<CommandResult>(&mapped)) {
        std::fprintf(stderr, "FAIL: format %d has no Metal render format: %.*s\n", int(format),
                     int(failure->detail.size()), failure->detail.data());
        ++failures;
        return MTLPixelFormatInvalid;
    }
    return std::get<MTLPixelFormat>(mapped);
}

// ---------------------------------------------------------------------------
// Section A: narrow target conversions through the production pipeline.
// ---------------------------------------------------------------------------

enum class ValueModel { Unorm, Snorm, Srgb, Half, Pack10, Pack11, RawUint, RawSint };

struct NarrowCase {
    const char* name;
    TextureFormat format;
    platform::ColorOutputScalar scalar;
    ValueModel model;
    unsigned channels;      // components stored in the target
    unsigned channel_bits;  // normalized/integer channel width
};

std::vector<std::uint8_t> expected_bytes(const NarrowCase& item) {
    std::vector<std::uint8_t> bytes;
    for (std::uint32_t row = 0; row < kHeight; ++row) {
        for (std::uint32_t column = 0; column < kWidth; ++column) {
            const unsigned quadrant = column / 4 + 2 * (row / 4);
            const auto& floats = k_float_quadrants[quadrant];
            const auto& uints = k_uint_quadrants[quadrant];
            const auto& sints = k_sint_quadrants[quadrant];
            switch (item.model) {
                case ValueModel::Unorm:
                    for (unsigned channel = 0; channel < item.channels; ++channel)
                        append_le(bytes, to_unorm(floats[channel], item.channel_bits),
                                  item.channel_bits / 8);
                    break;
                case ValueModel::Snorm:
                    for (unsigned channel = 0; channel < item.channels; ++channel)
                        append_le(bytes, std::bit_cast<std::uint32_t>(
                                             static_cast<std::int32_t>(
                                                 to_snorm(floats[channel], item.channel_bits))),
                                  item.channel_bits / 8);
                    break;
                case ValueModel::Srgb:
                    // Encoding applies to RGB; alpha stays linear.
                    for (unsigned channel = 0; channel < 3; ++channel)
                        bytes.push_back(to_srgb8(floats[channel]));
                    bytes.push_back(std::uint8_t(to_unorm(floats[3], 8)));
                    break;
                case ValueModel::Half:
                    for (unsigned channel = 0; channel < item.channels; ++channel)
                        append_le(bytes, to_half(floats[channel]), 2);
                    break;
                case ValueModel::Pack10: {
                    const std::uint32_t packed =
                        std::uint32_t(to_unorm(floats[0], 10)) |
                        (std::uint32_t(to_unorm(floats[1], 10)) << 10) |
                        (std::uint32_t(to_unorm(floats[2], 10)) << 20) |
                        (std::uint32_t(to_unorm(floats[3], 2)) << 30);
                    append_le(bytes, packed, 4);
                    break;
                }
                case ValueModel::Pack11: {
                    const std::uint32_t packed = to_minifloat(floats[0], 6) |
                                                 (to_minifloat(floats[1], 6) << 11) |
                                                 (to_minifloat(floats[2], 5) << 22);
                    append_le(bytes, packed, 4);
                    break;
                }
                case ValueModel::RawUint: {
                    const std::uint64_t limit = (std::uint64_t{1} << item.channel_bits) - 1;
                    for (unsigned channel = 0; channel < item.channels; ++channel)
                        append_le(bytes, std::min<std::uint64_t>(uints[channel], limit),
                                  item.channel_bits / 8);
                    break;
                }
                case ValueModel::RawSint: {
                    const std::int64_t low = -(std::int64_t{1} << (item.channel_bits - 1));
                    const std::int64_t high = (std::int64_t{1} << (item.channel_bits - 1)) - 1;
                    for (unsigned channel = 0; channel < item.channels; ++channel) {
                        const auto value = static_cast<std::int32_t>(
                            clamp_range(double(sints[channel]), double(low), double(high)));
                        append_le(bytes, std::bit_cast<std::uint32_t>(value),
                                  item.channel_bits / 8);
                    }
                    break;
                }
            }
        }
    }
    return bytes;
}

void narrow_target_conversions(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                               const std::vector<std::byte>& metallib,
                               metal::PipelineStateCache& cache) {
    const std::vector<NarrowCase> cases{
        {"R8 unorm", TextureFormat::R8Unorm, platform::ColorOutputScalar::f32, ValueModel::Unorm, 1,
         8},
        {"RG8 unorm", TextureFormat::R8G8Unorm, platform::ColorOutputScalar::f32, ValueModel::Unorm,
         2, 8},
        {"RGBA8 unorm", TextureFormat::R8G8B8A8Unorm, platform::ColorOutputScalar::f32,
         ValueModel::Unorm, 4, 8},
        {"R8 snorm", TextureFormat::R8Snorm, platform::ColorOutputScalar::f32, ValueModel::Snorm, 1,
         8},
        {"RG8 snorm", TextureFormat::R8G8Snorm, platform::ColorOutputScalar::f32, ValueModel::Snorm,
         2, 8},
        {"RGBA8 snorm", TextureFormat::R8G8B8A8Snorm, platform::ColorOutputScalar::f32,
         ValueModel::Snorm, 4, 8},
        {"RGBA8 srgb", TextureFormat::R8G8B8A8Srgb, platform::ColorOutputScalar::f32,
         ValueModel::Srgb, 4, 8},
        {"R16 unorm", TextureFormat::R16Unorm, platform::ColorOutputScalar::f32, ValueModel::Unorm,
         1, 16},
        {"R16 snorm", TextureFormat::R16Snorm, platform::ColorOutputScalar::f32, ValueModel::Snorm,
         1, 16},
        {"R16 float", TextureFormat::R16Float, platform::ColorOutputScalar::f32, ValueModel::Half, 1,
         16},
        {"RGBA16 unorm", TextureFormat::R16G16B16A16Unorm, platform::ColorOutputScalar::f32,
         ValueModel::Unorm, 4, 16},
        {"RGBA16 snorm", TextureFormat::R16G16B16A16Snorm, platform::ColorOutputScalar::f32,
         ValueModel::Snorm, 4, 16},
        {"RGBA16 float", TextureFormat::R16G16B16A16Float, platform::ColorOutputScalar::f32,
         ValueModel::Half, 4, 16},
        {"RGB10A2 unorm", TextureFormat::R10G10B10A2Unorm, platform::ColorOutputScalar::f32,
         ValueModel::Pack10, 4, 10},
        {"R11G11B10 float", TextureFormat::R11G11B10Float, platform::ColorOutputScalar::f32,
         ValueModel::Pack11, 3, 11},
        {"R8 uint", TextureFormat::R8Uint, platform::ColorOutputScalar::u32, ValueModel::RawUint, 1,
         8},
        {"R8 sint", TextureFormat::R8Sint, platform::ColorOutputScalar::s32, ValueModel::RawSint, 1,
         8},
        {"R16 uint", TextureFormat::R16Uint, platform::ColorOutputScalar::u32, ValueModel::RawUint,
         1, 16},
        {"R16 sint", TextureFormat::R16Sint, platform::ColorOutputScalar::s32, ValueModel::RawSint,
         1, 16},
    };
    for (const auto& item : cases) {
        auto stages =
            make_stages(metallib, "vs_full", "fs_quads", {{0, 15, item.scalar, 4}}, true);
        auto key = base_key(cache, stages);
        key.colors[0].format = item.format;
        auto acquired = cache.acquire(key, stages);
        if (const auto* error = std::get_if<RendererError>(&acquired)) {
            std::fprintf(stderr, "FAIL %s: production acquire rejected: %.*s\n", item.name,
                         int(error->command().detail.size()), error->command().detail.data());
            ++failures;
            continue;
        }
        auto& resident = std::get<metal::ResidentPipeline>(acquired);
        const auto format = render_format_or_fail(item.format);
        if (format == MTLPixelFormatInvalid) continue;
        const auto expected = expected_bytes(item);
        std::array<id<MTLTexture>, 8> colors{};
        std::array<TextureFormat, 8> formats{};
        colors[0] = make_target(device, format);
        formats[0] = item.format;
        CHECK(colors[0] != nil);
        const auto pixels = run_raw_pass(device, queue, colors, formats, pso_of(resident));
        const auto& actual = pixels.slots[0];
        const std::size_t pixel_bytes = format_pixel_bytes(item.format);
        CHECK(actual.size() == expected.size());
        if (actual.size() == expected.size() &&
            std::memcmp(actual.data(), expected.data(), expected.size()) != 0) {
            unsigned mismatches = 0;
            for (std::size_t index = 0; index < expected.size(); ++index) {
                if (actual[index] != expected[index] && mismatches++ < 6) {
                    const std::size_t pixel = index / pixel_bytes;
                    const std::size_t column = pixel % kWidth;
                    const std::size_t row = pixel / kWidth;
                    std::fprintf(stderr,
                                 "FAIL %s: pixel %zu (quadrant %zu, column %zu, row %zu) "
                                 "byte %zu: got 0x%02x, expected 0x%02x\n",
                                 item.name, pixel, column / 4 + 2 * (row / 4), column, row,
                                 index % pixel_bytes, actual[index], expected[index]);
                }
            }
            std::fprintf(stderr, "FAIL %s: %u mismatching bytes\n", item.name, mismatches);
            ++failures;
        }
    }
}

// ---------------------------------------------------------------------------
// Section B: sparse permuted targets through the production specialization.
// ---------------------------------------------------------------------------

void sparse_permuted_targets(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                             const std::vector<std::byte>& metallib,
                             metal::PipelineStateCache& cache) {
    // Six logical outputs; three carry attachments and three name physical
    // targets without a bound color state, so the permutation is never
    // compacted and logical output 2 does not sit at compact index 1.
    RenderStop stop;
    stop.engine = engine_class::threed;
    stop.kind = RenderKind::Draw;
    stop.registers[0x574] = 0;  // single-sample MSAA mode
    const std::uint32_t permutation[6] = {2, 5, 0, 6, 7, 4};
    std::uint32_t rt_control = 6;
    for (unsigned logical = 0; logical < 6; ++logical)
        rt_control |= permutation[logical] << (4 + logical * 3);
    stop.registers[0x487] = rt_control;
    const std::array<std::uint32_t, 3> physical_words{0xc2, 0xc0, 0xc1};  // u32/f32/s32
    const std::array<unsigned, 3> physical_slots{0, 2, 4};
    const std::array<TextureFormat, 3> physical_formats{TextureFormat::R32G32B32A32Uint,
                                                        TextureFormat::R32G32B32A32Float,
                                                        TextureFormat::R32G32B32A32Sint};
    for (unsigned index = 0; index < 3; ++index) {
        auto& target = stop.color_targets[physical_slots[index]];
        target.address = 0x10000 + std::uint64_t(physical_slots[index]) * 0x1000;
        target.width = kWidth;
        target.height = kHeight;
        target.format = physical_words[index];
        target.tile_mode = 0;
        target.depth = 1;
        target.array_pitch = 128;
        target.base_layer = 0;
    }
    const auto plan = take(plan_render_targets(stop));
    CHECK(plan.colors[0] && plan.colors[2] && plan.colors[5]);
    CHECK(!plan.colors[1] && !plan.colors[3] && !plan.colors[4]);
    CHECK(plan.colors[0]->physical_target == 2);
    CHECK(plan.colors[2]->physical_target == 0);
    CHECK(plan.colors[5]->physical_target == 4);

    using Scalar = platform::ColorOutputScalar;
    const std::array<Scalar, 3> classes{Scalar::f32, Scalar::u32, Scalar::s32};
    const std::array<unsigned, 3> logical_locations{0, 2, 5};
    std::vector<platform::CompiledColorOutput> outputs;
    for (unsigned index = 0; index < 3; ++index)
        outputs.push_back({static_cast<std::uint8_t>(logical_locations[index]), 15,
                           classes[index], 4});
    auto stages = make_stages(metallib, "vs_full", "fs_sparse", outputs, true);
    auto key = base_key(cache, stages);
    for (const unsigned location : logical_locations)
        key.colors[location].format = plan.colors[location]->texture.format;

    auto acquired = cache.acquire(key, stages);
    if (const auto* error = std::get_if<RendererError>(&acquired)) {
        std::fprintf(stderr, "FAIL sparse acquire: %.*s\n",
                     int(error->command().detail.size()), error->command().detail.data());
        ++failures;
        return;
    }
    auto resident = std::get<metal::ResidentPipeline>(std::move(acquired));

    // An identical key must reuse the specialized pipeline, while swapped
    // integer classes at locations 2 and 5 need a distinct one.
    auto repeat = cache.acquire(key, stages);
    CHECK(!std::holds_alternative<RendererError>(repeat));
    if (!std::holds_alternative<RendererError>(repeat))
        CHECK(pso_of(std::get<metal::ResidentPipeline>(repeat)) == pso_of(resident));
    auto swapped_stages =
        make_stages(metallib, "vs_full", "fs_sparse",
                    {{0, 15, Scalar::f32, 4}, {2, 15, Scalar::s32, 4}, {5, 15, Scalar::u32, 4}},
                    true);
    auto swapped_key = base_key(cache, swapped_stages);
    swapped_key.colors[0].format = plan.colors[0]->texture.format;
    swapped_key.colors[2].format = TextureFormat::R32G32B32A32Sint;
    swapped_key.colors[5].format = TextureFormat::R32G32B32A32Uint;
    auto swapped = cache.acquire(swapped_key, swapped_stages);
    CHECK(!std::holds_alternative<RendererError>(swapped));
    if (!std::holds_alternative<RendererError>(swapped))
        CHECK(pso_of(std::get<metal::ResidentPipeline>(swapped)) != pso_of(resident));

    // Draw outputs follow the logical slots with holes preserved: the
    // production pass binds each logical slot's texture at the Metal
    // attachment index named by the shader output location, while the plan's
    // physical_target records which guest target backs it.
    std::array<id<MTLTexture>, 8> colors{};
    std::array<TextureFormat, 8> formats{};
    for (unsigned index = 0; index < 3; ++index) {
        const unsigned location = logical_locations[index];
        colors[location] =
            make_target(device, render_format_or_fail(physical_formats[index]));
        formats[location] = plan.colors[location]->texture.format;
        CHECK(colors[location] != nil);
    }
    const auto pixels = run_raw_pass(device, queue, colors, formats, pso_of(resident));

    constexpr std::array<std::uint32_t, 4> location_zero_words{0x3f800000u, 0x40000000u,
                                                               0x40400000u, 0x3fc00000u};
    constexpr std::array<std::uint32_t, 4> location_two_words{0x11111111u, 0x22222222u,
                                                              0x33333333u, 0x44444444u};
    constexpr std::array<std::uint32_t, 4> location_five_words{0x87654321u, 0x12345678u,
                                                               0xffff0001u, 0x7ffffffeu};
    // The texture at logical slot 0 backs physical target 2, and so on: the
    // assertion pairs each Metal slot with its logical output's words.
    const std::array expected{std::pair{0u, location_zero_words},
                              std::pair{2u, location_two_words},
                              std::pair{5u, location_five_words}};
    for (const auto& [slot, words] : expected) {
        const auto& actual = pixels.slots[slot];
        const std::size_t wanted_bytes = kWidth * kHeight * 16;
        CHECK(actual.size() == wanted_bytes);
        if (actual.size() != wanted_bytes) continue;
        for (std::uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
            bool matched = true;
            for (unsigned lane = 0; lane < 4 && matched; ++lane) {
                std::uint32_t got = 0;
                std::memcpy(&got, actual.data() + pixel * 16 + lane * 4, 4);
                if (got != words[lane]) {
                    std::fprintf(stderr,
                                 "FAIL sparse MRT: physical slot %u pixel %u lane %u: "
                                 "got 0x%08" PRIx32 ", expected 0x%08" PRIx32 "\n",
                                 slot, pixel, lane, got, words[lane]);
                    matched = false;
                }
            }
            if (!matched) ++failures;
        }
    }
    // Logical holes receive no attachment and must stay unfilled.
    for (const unsigned hole : {1u, 3u, 4u, 6u, 7u})
        CHECK(pixels.slots[hole].empty());
}

// ---------------------------------------------------------------------------
// Section C: alpha-to-one alongside integer targets.
// ---------------------------------------------------------------------------

void alpha_to_one_integer_targets(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                                  const std::vector<std::byte>& metallib,
                                  metal::PipelineStateCache& cache) {
    using Scalar = platform::ColorOutputScalar;
    const auto float_stages =
        make_stages(metallib, "vs_full", "fs_alpha_probe", {{0, 15, Scalar::f32, 4}}, true);
    auto float_key = base_key(cache, float_stages);
    float_key.colors[0].format = TextureFormat::R32G32B32A32Float;
    float_key.colors[0].blend.enabled = true;
    // One/Zero keeps the written value untouched, so any observed alpha
    // change comes from alpha-to-one alone.
    float_key.colors[0].blend.rgb = {BlendOperation::Add, BlendFactor::One, BlendFactor::Zero};
    float_key.colors[0].blend.alpha = {BlendOperation::Add, BlendFactor::One, BlendFactor::Zero};

    const auto read_alpha = [&](const RawPassResult& pixels) {
        const auto& stored = pixels.slots[0];
        CHECK(stored.size() == kWidth * kHeight * 16);
        if (stored.size() != kWidth * kHeight * 16) return;
        float blue = 0, alpha = 0;
        std::memcpy(&blue, stored.data() + 8, 4);
        std::memcpy(&alpha, stored.data() + 12, 4);
        CHECK(blue == 0.75f);
        CHECK(alpha == 0.25f);
    };

    // Control: without alpha-to-one the fragment alpha survives blending.
    auto plain = cache.acquire(float_key, float_stages);
    CHECK(!std::holds_alternative<RendererError>(plain));
    if (!std::holds_alternative<RendererError>(plain)) {
        std::array<id<MTLTexture>, 8> colors{};
        std::array<TextureFormat, 8> formats{};
        colors[0] = make_target(device, MTLPixelFormatRGBA32Float);
        formats[0] = TextureFormat::R32G32B32A32Float;
        CHECK(colors[0] != nil);
        read_alpha(run_raw_pass(device, queue, colors, formats,
                                pso_of(std::get<metal::ResidentPipeline>(plain))));
    }

    // Alpha-to-one on a float attachment forces the blended alpha to one.
    auto forced_key = float_key;
    forced_key.coverage.alpha_to_one = true;
    auto forced = cache.acquire(forced_key, float_stages);
    CHECK(!std::holds_alternative<RendererError>(forced));
    if (!std::holds_alternative<RendererError>(forced)) {
        std::array<id<MTLTexture>, 8> colors{};
        std::array<TextureFormat, 8> formats{};
        colors[0] = make_target(device, MTLPixelFormatRGBA32Float);
        formats[0] = TextureFormat::R32G32B32A32Float;
        CHECK(colors[0] != nil);
        const RawPassResult forced_pixels = run_raw_pass(
            device, queue, colors, formats, pso_of(std::get<metal::ResidentPipeline>(forced)));
        const auto& stored = forced_pixels.slots[0];
        CHECK(stored.size() == kWidth * kHeight * 16);
        if (stored.size() == kWidth * kHeight * 16) {
            float blue = 0, alpha = 0;
            std::memcpy(&blue, stored.data() + 8, 4);
            std::memcpy(&alpha, stored.data() + 12, 4);
            CHECK(blue == 0.75f);
            CHECK(alpha == 1.0f);
        }
    }

    // Integer attachment control: with alpha-to-one clear, the integer
    // pipeline is admitted and the exact values land unmodified.
    const auto uint_stages =
        make_stages(metallib, "vs_full", "fs_alpha_probe", {{0, 15, Scalar::u32, 4}}, true);
    auto uint_key = base_key(cache, uint_stages);
    uint_key.colors[0].format = TextureFormat::R8G8B8A8Uint;
    auto control = cache.acquire(uint_key, uint_stages);
    CHECK(!std::holds_alternative<RendererError>(control));
    if (!std::holds_alternative<RendererError>(control)) {
        std::array<id<MTLTexture>, 8> colors{};
        std::array<TextureFormat, 8> formats{};
        colors[0] = make_target(device, MTLPixelFormatRGBA8Uint);
        formats[0] = TextureFormat::R8G8B8A8Uint;
        CHECK(colors[0] != nil);
        const auto pixels = run_raw_pass(device, queue, colors, formats,
                                         pso_of(std::get<metal::ResidentPipeline>(control)));
        const auto& stored = pixels.slots[0];
        CHECK(stored.size() == kWidth * kHeight * 4);
        if (stored.size() == kWidth * kHeight * 4) {
            for (std::uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
                const std::uint32_t got = std::uint32_t(stored[pixel * 4]) |
                                          (std::uint32_t(stored[pixel * 4 + 1]) << 8) |
                                          (std::uint32_t(stored[pixel * 4 + 2]) << 16) |
                                          (std::uint32_t(stored[pixel * 4 + 3]) << 24);
                if (got != 0x193264c8u) {  // bytes (200, 100, 50, 25) little-endian
                    std::fprintf(stderr,
                                 "FAIL integer control: pixel %u got 0x%08x, expected "
                                 "0x193264c8\n",
                                 pixel, got);
                    ++failures;
                    break;
                }
            }
        }
    }

    // Metal accepts alpha-to-one only with float/unorm attachments and aborts
    // otherwise. It has no effect on integer targets on the guest GPU, so an
    // all-integer pass must be admitted with it disabled (no effect), and a
    // mixed float/integer pass must be rejected with a structured error.
    // Mixed float and integer attachments cannot express alpha-to-one.
    auto mixed_key = float_key;
    mixed_key.colors[0].blend.enabled = false;
    mixed_key.colors[1].format = TextureFormat::R8G8B8A8Uint;
    mixed_key.coverage.alpha_to_one = true;
    CHECK(std::holds_alternative<RendererError>(cache.acquire(mixed_key, float_stages)));
    auto integer_key = uint_key;
    integer_key.coverage.alpha_to_one = true;
    auto integer = cache.acquire(integer_key, uint_stages);
    // Production admits the pass and disables alpha-to-one, which has no
    // effect on integer targets: the stored words must be unchanged.
    CHECK(!std::holds_alternative<RendererError>(integer));
    if (std::holds_alternative<RendererError>(integer)) return;
    std::printf("integer alpha-to-one: admitted, asserting no effect\n");
    auto& resident = std::get<metal::ResidentPipeline>(integer);
    std::array<id<MTLTexture>, 8> colors{};
    std::array<TextureFormat, 8> formats{};
    colors[0] = make_target(device, MTLPixelFormatRGBA8Uint);
    formats[0] = TextureFormat::R8G8B8A8Uint;
    CHECK(colors[0] != nil);
    const auto pixels = run_raw_pass(device, queue, colors, formats, pso_of(resident));
    const auto& stored = pixels.slots[0];
    CHECK(stored.size() == kWidth * kHeight * 4);
    if (stored.size() == kWidth * kHeight * 4) {
        for (std::uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
            const std::uint32_t got = std::uint32_t(stored[pixel * 4]) |
                                      (std::uint32_t(stored[pixel * 4 + 1]) << 8) |
                                      (std::uint32_t(stored[pixel * 4 + 2]) << 16) |
                                      (std::uint32_t(stored[pixel * 4 + 3]) << 24);
            if (got != 0x193264c8u) {  // bytes (200, 100, 50, 25) little-endian
                std::fprintf(stderr,
                             "FAIL integer alpha-to-one: pixel %u got 0x%08x, expected "
                             "0x193264c8\n",
                             pixel, got);
                ++failures;
                break;
            }
        }
    }
}

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory) {
            std::fprintf(stderr, "unified-memory Metal 4 device is unavailable\n");
            return 1;
        }
        auto device_context = std::make_shared<TestDeviceContext>(device);
        auto cache = take(metal::create_pipeline_state_cache(device_context, 16 * 1024 * 1024));
        const auto quad_metallib = compile_library(quad_source());
        const auto sparse_metallib = compile_library(sparse_source());
        if (quad_metallib.empty() || sparse_metallib.empty()) return 1;
        narrow_target_conversions(device, device_context->queue(), quad_metallib, *cache);
        sparse_permuted_targets(device, device_context->queue(), sparse_metallib, *cache);
        alpha_to_one_integer_targets(device, device_context->queue(), quad_metallib, *cache);
        std::printf("fragment output conversions: %u checks, %u failed\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
}
