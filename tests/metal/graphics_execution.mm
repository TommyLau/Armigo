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

#include <armigo/shader/graphics_msl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace gm = armigo::shader::graphics_msl;
namespace gx = armigo::shader::graphics;
namespace mx = armigo::shader::maxwell;
using Stage = armigo::shader_input::Stage;
using Family = mx::OpcodeFamily;

constexpr NSUInteger kWidth = 8;
constexpr NSUInteger kHeight = 8;
constexpr NSUInteger kPixelBytes = 4 * sizeof(float);
constexpr NSUInteger kGuardBytes = 64;
constexpr std::byte kGuard{0xa7};
constexpr std::uint8_t kMultiplierBinding = 5;

void put(mx::DecodedProgram& program, unsigned at, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) program.sph[at + i] = std::byte(value >> (i * 8));
}

mx::DecodedProgram base(Stage stage) {
    mx::DecodedProgram result;
    result.stage = stage;
    put(result, 0, armigo::shader_input::sph_stage_type_for(stage) << 10);
    return result;
}

void attribute(mx::DecodedProgram& program, unsigned component,
               bool output = false, unsigned interpolation = 2) {
    unsigned at = 0, bit = 0, value = 1;
    if (component >= 28 && component < 32) {
        at = output ? 0x35 : 0x17;
        bit = component - 24;
    } else if (program.stage == Stage::Vertex) {
        at = (output ? 0x36 : 0x18) + (component - 32) / 8;
        bit = (component - 32) % 8;
    } else {
        at = 0x18 + (component - 32) / 4;
        bit = ((component - 32) % 4) * 2;
        value = interpolation;
    }
    program.sph[at] |= std::byte(value << bit);
}

void append(mx::DecodedProgram& program, Family family, mx::InstructionPayload payload) {
    const auto i = program.instructions.size();
    program.instructions.push_back({
        static_cast<std::uint32_t>((i / 3) * 32 + (i % 3 + 1) * 8),
        0, family, {7, false}, std::move(payload)});
}

void mov(mx::DecodedProgram& program, std::uint8_t destination, float value) {
    append(program, Family::Mov32Immediate,
           mx::Mov32Immediate{{destination}, 15, std::bit_cast<std::uint32_t>(value)});
}

void load(mx::DecodedProgram& program, std::uint8_t destination, unsigned component) {
    append(program, Family::AttributeLoad,
           mx::AttributeAccess{{destination}, {255}, {255},
                               static_cast<std::uint16_t>(component * 4), 0, false});
}

void store(mx::DecodedProgram& program, std::uint8_t source, unsigned component) {
    append(program, Family::AttributeStore,
           mx::AttributeAccess{{source}, {255}, {255},
                               static_cast<std::uint16_t>(component * 4), 0, false});
}

void exit(mx::DecodedProgram& program) {
    append(program, Family::Exit, mx::Exit{15});
}

bool lower(mx::DecodedProgram decoded, std::optional<gx::Program>& output) {
    auto result = gx::lower(decoded);
    if (const auto* error = std::get_if<gx::Error>(&result)) {
        std::fprintf(stderr, "Graphics lowering failed: %s at %u.\n",
                     error->message(), error->pc);
        return false;
    }
    output = std::get<gx::Program>(std::move(result));
    return true;
}

bool make_programs(std::optional<gx::Program>& vertex,
                   std::optional<gx::Program>& fragment) {
    auto vs = base(Stage::Vertex);
    for (unsigned component = 32; component < 35; ++component) attribute(vs, component);
    for (unsigned component = 36; component < 39; ++component) attribute(vs, component);
    for (unsigned component = 28; component < 32; ++component) attribute(vs, component, true);
    attribute(vs, 32, true);
    attribute(vs, 33, true);
    load(vs, 1, 32); load(vs, 2, 33); load(vs, 3, 34);
    load(vs, 4, 36); load(vs, 5, 37);
    // Multiplier -1.0 with this family's fixed NegateB leaves u unchanged; a
    // lowering that drops the product-sign flip or reads another binding
    // changes the sampled image.
    append(vs, Family::FloatFmaConstantNegated,
           mx::FloatFmaConstantNegated{{4}, {4}, {kMultiplierBinding, 0}, {255}});
    mov(vs, 6, 2.0f);
    store(vs, 1, 28); store(vs, 2, 29); store(vs, 3, 30); store(vs, 6, 31);
    store(vs, 4, 32); store(vs, 5, 33);
    exit(vs);

    auto fs = base(Stage::Fragment);
    attribute(fs, 31);
    attribute(fs, 32, false, 2);
    attribute(fs, 33, false, 2);
    put(fs, 0x48, 0xf);
    mov(fs, 10, 0.5f);
    append(fs, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{4}, {255}, {10}, 32, 0x7ff, true});
    append(fs, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{5}, {255}, {255}, 33, 0x7ff});
    append(fs, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{6}, {255}, {255}, 31, 0x7ff});
    append(fs, Family::MultiFunction, mx::Reciprocal{{7}, {6}});
    append(fs, Family::FloatMultiplyRegister,
           mx::FloatMultiply{{8}, {4}, mx::RegisterOperand{7},
                             mx::RoundingMode::Zero, mx::FlushMode::FlushToZero});
    append(fs, Family::FloatMultiplyRegister,
           mx::FloatMultiply{{9}, {5}, mx::RegisterOperand{7},
                             mx::RoundingMode::Zero, mx::FlushMode::FlushToZero});
    append(fs, Family::TextureSample,
           mx::TextureSample2D{{0}, {2}, {8}, {9}, 32, true});
    exit(fs);
    return lower(std::move(vs), vertex) && lower(std::move(fs), fragment);
}

struct Context {
    id<MTLDevice> device = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTL4CommandQueue> queue = nil;
};

id<MTLLibrary> library(Context& context, const std::string& source) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> result = [context.device
        newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                     options:options error:&error];
    if (!result) std::fprintf(stderr, "Metal source compilation failed: %s\n",
                              error.localizedDescription.UTF8String);
    return result;
}

id<MTLRenderPipelineState> pipeline(Context& context,
                                    id<MTLLibrary> vertex_library, NSString* vertex_name,
                                    id<MTLLibrary> fragment_library, NSString* fragment_name,
                                    MTLVertexDescriptor* vertex_descriptor) {
    MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = vertex_library;
    vertex.name = vertex_name;
    MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = fragment_library;
    fragment.name = fragment_name;
    MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vertex;
    descriptor.fragmentFunctionDescriptor = fragment;
    descriptor.vertexDescriptor = vertex_descriptor;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    NSError* error = nil;
    id<MTLRenderPipelineState> result = [context.compiler
        newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (!result) std::fprintf(stderr, "Metal render pipeline compilation failed: %s\n",
                              error.localizedDescription.UTF8String);
    return result;
}

struct Readback {
    id<MTLTexture> target = nil;
    id<MTLBuffer> buffer = nil;
};

Readback make_readback(id<MTLDevice> device) {
    auto* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:kWidth height:kHeight mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    Readback result;
    result.target = [device newTextureWithDescriptor:descriptor];
    result.buffer = [device newBufferWithLength:kGuardBytes * 2 + kWidth * kHeight * kPixelBytes
                                        options:MTLResourceStorageModeShared];
    if (result.buffer) std::memset(result.buffer.contents, std::to_integer<int>(kGuard),
                                   result.buffer.length);
    return result;
}

bool encode_readback(id<MTL4CommandBuffer> commands, Readback readback) {
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    if (!copy) return false;
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
               visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:readback.target sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                 toBuffer:readback.buffer destinationOffset:kGuardBytes
    destinationBytesPerRow:kWidth * kPixelBytes destinationBytesPerImage:0];
    [copy endEncoding];
    return true;
}

bool submit(Context& context, id<MTL4CommandBuffer> commands, NSArray* retained) {
    [commands endCommandBuffer];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [context.queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(completed,
            dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0) {
        std::fputs("Metal graphics execution timed out.\n", stderr);
        return false;
    }
    if (execution_error) {
        std::fprintf(stderr, "Metal graphics execution failed: %s\n",
                     execution_error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

bool guards_intact(const Readback& readback) {
    const auto* bytes = static_cast<const std::byte*>(readback.buffer.contents);
    for (NSUInteger i = 0; i < kGuardBytes; ++i) {
        if (bytes[i] != kGuard || bytes[readback.buffer.length - 1 - i] != kGuard) return false;
    }
    return true;
}

const float* pixels(const Readback& readback) {
    return reinterpret_cast<const float*>(
        static_cast<const std::byte*>(readback.buffer.contents) + kGuardBytes);
}

bool position_w_probe(Context& context) {
    const std::string source = R"msl(
#include <metal_stdlib>
using namespace metal;
struct ProbeOut { float4 position [[position]]; };
vertex ProbeOut probe_vertex(device const float4* vertices [[buffer(0)]],
                             uint id [[vertex_id]]) {
    ProbeOut out;
    out.position = vertices[id];
    return out;
}
fragment float4 probe_fragment(ProbeOut in [[stage_in]]) {
    return float4(in.position.w, 0.0f, 0.0f, 1.0f);
}
)msl";
    id<MTLLibrary> metal_library = library(context, source);
    id<MTLRenderPipelineState> metal_pipeline = metal_library
        ? pipeline(context, metal_library, @"probe_vertex", metal_library,
                   @"probe_fragment", nil) : nil;
    if (!metal_pipeline) return false;

    constexpr std::array clip_w{0.5f, 2.0f, 3.0f, 4.0f};
    std::array<std::array<float, 4>, 12> vertices{};
    for (std::size_t triangle = 0; triangle < clip_w.size(); ++triangle) {
        const float w = clip_w[triangle];
        vertices[triangle * 3 + 0] = {-w, -w, 0.0f, w};
        vertices[triangle * 3 + 1] = { 3.0f * w, -w, 0.0f, w};
        vertices[triangle * 3 + 2] = {-w,  3.0f * w, 0.0f, w};
    }
    id<MTLBuffer> vertex_buffer = [context.device newBufferWithBytes:vertices.data()
        length:sizeof(vertices) options:MTLResourceStorageModeShared];
    auto readback = make_readback(context.device);
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 1;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [context.device
        newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 3;
    id<MTLResidencySet> residency = [context.device
        newResidencySetWithDescriptor:residency_descriptor error:&error];
    id<MTL4CommandAllocator> allocator = [context.device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [context.device newCommandBuffer];
    if (!vertex_buffer || !readback.target || !readback.buffer || !table || !residency ||
        !allocator || !commands) return false;
    [table setAddress:vertex_buffer.gpuAddress atIndex:0];
    [residency addAllocation:vertex_buffer];
    [residency addAllocation:readback.target];
    [residency addAllocation:readback.buffer];
    [residency commit];
    NSArray* retained = @[context.device, context.compiler, context.queue, metal_library,
                          metal_pipeline, vertex_buffer, readback.target, readback.buffer,
                          table, residency, allocator, commands];

    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture = readback.target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(-99.0, -99.0, -99.0, -99.0);
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return false;
    [render setRenderPipelineState:metal_pipeline];
    [render setArgumentTable:table atStages:MTLRenderStageVertex];
    for (NSUInteger triangle = 0; triangle < clip_w.size(); ++triangle) {
        const NSUInteger x = (triangle & 1) * 4;
        const NSUInteger y = (triangle >> 1) * 4;
        [render setScissorRect:MTLScissorRect{x, y, 4, 4}];
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:triangle * 3 vertexCount:3];
    }
    [render endEncoding];
    if (!encode_readback(commands, readback) || !submit(context, commands, retained) ||
        !guards_intact(readback)) return false;

    const float* actual = pixels(readback);
    for (NSUInteger y = 0; y < kHeight; ++y) {
        for (NSUInteger x = 0; x < kWidth; ++x) {
            const NSUInteger quadrant = (x / 4) + 2 * (y / 4);
            const float expected = 1.0f / clip_w[quadrant];
            if (!std::isfinite(actual[(y * kWidth + x) * 4]) ||
                std::abs(actual[(y * kWidth + x) * 4] - expected) > 1.0e-6f) {
                std::fprintf(stderr, "PositionW mismatch at (%lu,%lu): expected %.8g, got %.8g.\n",
                             x, y, expected, actual[(y * kWidth + x) * 4]);
                return false;
            }
        }
    }
    std::printf("Verified fragment PositionW at %zu interior pixels.\n", clip_w.size() * 16);
    return true;
}

std::array<float, 4> bilinear(float u, float v) {
    const float tx = std::clamp(u * 2.0f - 0.5f, 0.0f, 1.0f);
    const float ty = std::clamp(v * 2.0f - 0.5f, 0.0f, 1.0f);
    const std::array<std::array<float, 4>, 4> texels{{
        {1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 1, 1}}};
    std::array<float, 4> result{};
    for (unsigned lane = 0; lane < 4; ++lane) {
        const float top = texels[0][lane] * (1 - tx) + texels[1][lane] * tx;
        const float bottom = texels[2][lane] * (1 - tx) + texels[3][lane] * tx;
        result[lane] = top * (1 - ty) + bottom * ty;
    }
    return result;
}

bool emitted_graphics(Context& context) {
    std::optional<gx::Program> vertex, fragment;
    if (!make_programs(vertex, fragment)) return false;
    gm::EmitOptions vertex_options;
    vertex_options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    vertex_options.vertex_attributes = {{0, 0, gm::VertexFormat::Float32x3},
                                        {1, 1, gm::VertexFormat::Float32x3}};
    gm::EmitOptions fragment_options;
    fragment_options.fragment_position =
        gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    auto vertex_result = gm::emit(*vertex, vertex_options);
    auto fragment_result = gm::emit(*fragment, fragment_options);
    const auto* emitted_vertex = std::get_if<gm::EmittedStage>(&vertex_result);
    const auto* emitted_fragment = std::get_if<gm::EmittedStage>(&fragment_result);
    if (!emitted_vertex || !emitted_fragment) {
        std::fputs("Graphics MSL emission failed.\n", stderr);
        return false;
    }
    const auto multiplier_binding = std::find_if(
        emitted_vertex->constant_buffers.begin(), emitted_vertex->constant_buffers.end(),
        [](const gm::ConstantBufferBinding& binding) {
            return binding.guest_binding == kMultiplierBinding;
        });
    if (multiplier_binding == emitted_vertex->constant_buffers.end()) {
        std::fputs("Emitted vertex shader lost the FMA multiplier binding.\n", stderr);
        return false;
    }
    id<MTLLibrary> vertex_library = library(context, emitted_vertex->source);
    id<MTLLibrary> fragment_library = library(context, emitted_fragment->source);
    MTLVertexDescriptor* vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
    // Vertex fetch moves to argument index 1: the emitted shader owns the
    // reflected constant-buffer index for the multiplier binding.
    vertex_descriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertex_descriptor.attributes[0].offset = 0;
    vertex_descriptor.attributes[0].bufferIndex = 1;
    vertex_descriptor.attributes[1].format = MTLVertexFormatFloat3;
    vertex_descriptor.attributes[1].offset = 3 * sizeof(float);
    vertex_descriptor.attributes[1].bufferIndex = 1;
    vertex_descriptor.layouts[1].stride = 6 * sizeof(float);
    id<MTLRenderPipelineState> metal_pipeline = vertex_library && fragment_library
        ? pipeline(context, vertex_library, @"armigo_vertex_main", fragment_library,
                   @"armigo_fragment_main", vertex_descriptor) : nil;
    if (!metal_pipeline) return false;

    constexpr std::array<float, 18> vertices{
        -2, -2, -2, 0.25f,  0.75f, 0,
         6, -2, -2, 1.25f,  0.75f, 0,
        -2,  6, -2, 0.25f, -0.25f, 0};
    id<MTLBuffer> vertex_buffer = [context.device newBufferWithBytes:vertices.data()
        length:sizeof(vertices) options:MTLResourceStorageModeShared];
    auto* texture_descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:2 height:2 mipmapped:NO];
    texture_descriptor.storageMode = MTLStorageModeShared;
    texture_descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [context.device newTextureWithDescriptor:texture_descriptor];
    constexpr std::array<float, 16> texture_data{
        1, 0, 0, 1,  0, 1, 0, 1,
        0, 0, 1, 1,  1, 1, 1, 1};
    [texture replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0
                 withBytes:texture_data.data() bytesPerRow:2 * kPixelBytes];
    MTLSamplerDescriptor* sampler_descriptor = [MTLSamplerDescriptor new];
    sampler_descriptor.minFilter = MTLSamplerMinMagFilterLinear;
    sampler_descriptor.magFilter = MTLSamplerMinMagFilterLinear;
    sampler_descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler_descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> sampler = [context.device newSamplerStateWithDescriptor:sampler_descriptor];
    // A runtime buffer keeps the multiplier out of the Metal compiler's reach;
    // sign handling must come from the emitted shader, not constant folding.
    constexpr std::array<float, 1> multiplier_data{-1.0f};
    const NSUInteger multiplier_bytes =
        std::max<NSUInteger>(sizeof multiplier_data, multiplier_binding->minimum_size);
    id<MTLBuffer> multiplier_buffer = [context.device
        newBufferWithBytes:multiplier_data.data() length:multiplier_bytes
                   options:MTLResourceStorageModeShared];
    auto readback = make_readback(context.device);

    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    table_descriptor.maxTextureBindCount = 1;
    table_descriptor.maxSamplerStateBindCount = 1;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [context.device
        newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 5;
    id<MTLResidencySet> residency = [context.device
        newResidencySetWithDescriptor:residency_descriptor error:&error];
    id<MTL4CommandAllocator> allocator = [context.device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [context.device newCommandBuffer];
    if (!vertex_buffer || !multiplier_buffer || !texture || !sampler || !readback.target ||
        !readback.buffer || !table || !residency || !allocator || !commands) return false;
    [table setAddress:multiplier_buffer.gpuAddress
              atIndex:multiplier_binding->metal_buffer_index];
    [table setAddress:vertex_buffer.gpuAddress atIndex:1];
    [table setTexture:texture.gpuResourceID atIndex:0];
    [table setSamplerState:sampler.gpuResourceID atIndex:0];
    [residency addAllocation:vertex_buffer];
    [residency addAllocation:multiplier_buffer];
    [residency addAllocation:texture];
    [residency addAllocation:readback.target];
    [residency addAllocation:readback.buffer];
    [residency commit];
    NSArray* retained = @[context.device, context.compiler, context.queue, vertex_library,
                          fragment_library, metal_pipeline, vertex_buffer, multiplier_buffer,
                          texture, sampler, readback.target, readback.buffer, table, residency,
                          allocator, commands];

    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture = readback.target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(-99.0, -99.0, -99.0, -99.0);
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return false;
    [render setRenderPipelineState:metal_pipeline];
    [render setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];
    if (!encode_readback(commands, readback) || !submit(context, commands, retained) ||
        !guards_intact(readback)) return false;

    std::array<gx::AttributeInput, 3> inputs{{
        {31, std::bit_cast<std::uint32_t>(0.5f)},
        {32, 0}, {33, 0}}};
    gx::Inputs cpu_inputs;
    cpu_inputs.attributes = inputs;
    cpu_inputs.sample_2d_implicit_lod = [&](const gx::TextureRequest& request,
                                            std::array<std::uint32_t, 4>& result) {
        if (request.handle_byte_offset != 32 || !request.no_dependency) return false;
        const auto color = bilinear(std::bit_cast<float>(request.x_bits),
                                    std::bit_cast<float>(request.y_bits));
        for (unsigned lane = 0; lane < 4; ++lane)
            result[lane] = std::bit_cast<std::uint32_t>(color[lane]);
        return true;
    };
    const float* actual = pixels(readback);
    for (NSUInteger pixel = 0; pixel < kWidth * kHeight; ++pixel) {
        const float u = 0.25f + 0.5f * (float(pixel % kWidth) + 0.5f) / float(kWidth);
        const float v = 0.25f + 0.5f * (float(pixel / kWidth) + 0.5f) / float(kHeight);
        inputs[1].bits = std::bit_cast<std::uint32_t>(u);
        inputs[2].bits = std::bit_cast<std::uint32_t>(v);
        const auto cpu_result = gx::evaluate(*fragment, cpu_inputs);
        const auto* evaluation = std::get_if<gx::Evaluation>(&cpu_result);
        if (!evaluation || evaluation->outputs.size() != 4) return false;
        const auto expected = bilinear(u * 0.5f, v);
        for (unsigned lane = 0; lane < 4; ++lane) {
            if (std::bit_cast<float>(evaluation->outputs[lane].bits) != expected[lane]) return false;
            if (!std::isfinite(actual[pixel * 4 + lane]) ||
                std::abs(actual[pixel * 4 + lane] - expected[lane]) > 2.0e-6f) {
                std::fprintf(stderr,
                    "Emitted graphics mismatch at pixel %lu lane %u: expected %.8g, got %.8g.\n",
                    pixel, lane, expected[lane], actual[pixel * 4 + lane]);
                return false;
            }
        }
    }
    std::printf("Verified %lu emitted graphics pixels against the CPU evaluator.\n",
                kWidth * kHeight);
    return true;
}

}  // namespace

int main() {
    @autoreleasepool {
        Context context;
        context.device = MTLCreateSystemDefaultDevice();
        if (!context.device || ![context.device supportsFamily:MTLGPUFamilyMetal4] ||
            !context.device.hasUnifiedMemory) {
            std::fputs("Graphics execution tests require unified-memory Metal 4.\n", stderr);
            return 1;
        }
        NSError* error = nil;
        context.compiler = [context.device
            newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        context.queue = [context.device newMTL4CommandQueue];
        if (!context.compiler || !context.queue) {
            std::fprintf(stderr, "Metal 4 setup failed: %s\n",
                         error.localizedDescription.UTF8String);
            return 1;
        }
        if (!position_w_probe(context)) return 1;
        if (!emitted_graphics(context)) return 1;
        return 0;
    }
}
