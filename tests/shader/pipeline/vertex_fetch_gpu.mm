// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Shader-side vertex fetch: for every vertex format the replay maps, every
// input class (float, uint, int) and, for integer storage, both the plain and
// the integer-to-float variant, a vertex function reading the attribute
// through the fetch table must store the register words computed here on the
// CPU, over every 16-bit value of the first component. Wherever Metal's
// vertex descriptor fetch has the same semantics it must agree too (NaN
// payloads aside). Per-instance and constant streams, base instances, indexed
// draws with a base vertex and attributes that end past their stream are
// compared the same way.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <armigo/shader/vertex_fetch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using armigo::shader::VertexFetchKind;
using armigo::shader::VertexFetchTable;

int failures = 0;

struct Format {
    const char* name;
    MTLVertexFormat format;
    VertexFetchKind kind;
    std::uint32_t components;
};

#define FORMAT(name, kind, components) {#name, MTLVertexFormat##name, VertexFetchKind::kind, components}
const Format formats[] = {
    FORMAT(UCharNormalized, unorm8, 1), FORMAT(UChar2Normalized, unorm8, 2),
    FORMAT(UChar3Normalized, unorm8, 3), FORMAT(UChar4Normalized, unorm8, 4),
    FORMAT(CharNormalized, snorm8, 1), FORMAT(Char2Normalized, snorm8, 2),
    FORMAT(Char3Normalized, snorm8, 3), FORMAT(Char4Normalized, snorm8, 4),
    FORMAT(UChar, uint8, 1), FORMAT(UChar2, uint8, 2), FORMAT(UChar3, uint8, 3), FORMAT(UChar4, uint8, 4),
    FORMAT(Char, sint8, 1), FORMAT(Char2, sint8, 2), FORMAT(Char3, sint8, 3), FORMAT(Char4, sint8, 4),
    FORMAT(UShortNormalized, unorm16, 1), FORMAT(UShort2Normalized, unorm16, 2),
    FORMAT(UShort3Normalized, unorm16, 3), FORMAT(UShort4Normalized, unorm16, 4),
    FORMAT(ShortNormalized, snorm16, 1), FORMAT(Short2Normalized, snorm16, 2),
    FORMAT(Short3Normalized, snorm16, 3), FORMAT(Short4Normalized, snorm16, 4),
    FORMAT(UShort, uint16, 1), FORMAT(UShort2, uint16, 2), FORMAT(UShort3, uint16, 3), FORMAT(UShort4, uint16, 4),
    FORMAT(Short, sint16, 1), FORMAT(Short2, sint16, 2), FORMAT(Short3, sint16, 3), FORMAT(Short4, sint16, 4),
    FORMAT(Half, float16, 1), FORMAT(Half2, float16, 2), FORMAT(Half3, float16, 3), FORMAT(Half4, float16, 4),
    FORMAT(Float, float32, 1), FORMAT(Float2, float32, 2), FORMAT(Float3, float32, 3), FORMAT(Float4, float32, 4),
    FORMAT(UInt, uint32, 1), FORMAT(UInt2, uint32, 2), FORMAT(UInt3, uint32, 3), FORMAT(UInt4, uint32, 4),
    FORMAT(Int, sint32, 1), FORMAT(Int2, sint32, 2), FORMAT(Int3, sint32, 3), FORMAT(Int4, sint32, 4),
    FORMAT(UInt1010102Normalized, unorm10_10_10_2, 4), FORMAT(Int1010102Normalized, snorm10_10_10_2, 4),
    FORMAT(FloatRG11B10, float11_11_10, 3), FORMAT(UChar4Normalized_BGRA, unorm8_bgra, 4),
};
#undef FORMAT

const char* class_types[] = {"float4", "uint4", "int4"};
const char* class_fetch[] = {"armigo_vertex_fetch_float", "armigo_vertex_fetch_uint", "armigo_vertex_fetch_int"};

constexpr std::uint32_t vertex_count = 65536;
constexpr std::uint32_t stride = 24;
constexpr std::uint32_t offset = 4;

bool integer_kind(VertexFetchKind kind) {
    switch (kind) {
        case VertexFetchKind::uint8: case VertexFetchKind::sint8: case VertexFetchKind::uint16:
        case VertexFetchKind::sint16: case VertexFetchKind::uint32: case VertexFetchKind::sint32: return true;
        default: return false;
    }
}

std::uint32_t component_size(VertexFetchKind kind) {
    const auto value = static_cast<std::uint32_t>(kind);
    return value <= 4 || kind == VertexFetchKind::unorm8_bgra ? 1 : value <= 9 ? 2 : 4;
}

std::uint32_t bits(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, 4);
    return word;
}

float half_value(std::uint16_t word) {
    __fp16 value;
    std::memcpy(&value, &word, 2);
    return static_cast<float>(value);
}

std::int32_t extend(std::uint32_t value, std::uint32_t width) {
    const std::uint32_t shift = 32 - width;
    return static_cast<std::int32_t>(value << shift) >> shift;
}

// The register words a format writes for the attribute bytes at `at`.
std::array<std::uint32_t, 4> expected_words(VertexFetchKind kind, std::uint32_t components, bool scaled,
                                            const std::uint8_t* at) {
    using K = VertexFetchKind;
    const auto size = component_size(kind);
    std::array<std::uint32_t, 4> raw{};
    for (std::uint32_t c = 0; c < components && kind != K::unorm10_10_10_2 && kind != K::snorm10_10_10_2 &&
                              kind != K::float11_11_10; ++c) {
        std::uint32_t word = 0;
        std::memcpy(&word, at + c * size, size);
        raw[c] = word;
    }
    std::uint32_t packed = 0;
    std::memcpy(&packed, at, 4);
    std::array<std::uint32_t, 4> out{};
    std::uint32_t one = bits(1.0f);
    switch (kind) {
        case K::unorm8: for (int c = 0; c < 4; ++c) out[c] = bits(float(raw[c]) / 255.0f); break;
        case K::snorm8:
            for (int c = 0; c < 4; ++c) out[c] = bits(std::max(float(extend(raw[c], 8)) / 127.0f, -1.0f));
            break;
        case K::unorm16: for (int c = 0; c < 4; ++c) out[c] = bits(float(raw[c]) / 65535.0f); break;
        case K::snorm16:
            for (int c = 0; c < 4; ++c) out[c] = bits(std::max(float(extend(raw[c], 16)) / 32767.0f, -1.0f));
            break;
        case K::float16: for (int c = 0; c < 4; ++c) out[c] = bits(half_value(std::uint16_t(raw[c]))); break;
        case K::float32: out = raw; break;
        case K::uint8: case K::uint16: case K::uint32:
            for (int c = 0; c < 4; ++c) out[c] = scaled ? bits(float(raw[c])) : raw[c];
            if (!scaled) one = 1;
            break;
        case K::sint8: case K::sint16: case K::sint32: {
            const std::uint32_t width = kind == K::sint8 ? 8 : kind == K::sint16 ? 16 : 32;
            for (int c = 0; c < 4; ++c) {
                const auto value = extend(raw[c], width);
                out[c] = scaled ? bits(float(value)) : static_cast<std::uint32_t>(value);
            }
            if (!scaled) one = 1;
            break;
        }
        case K::unorm10_10_10_2:
            out = {bits(float(packed & 1023) / 1023.0f), bits(float((packed >> 10) & 1023) / 1023.0f),
                   bits(float((packed >> 20) & 1023) / 1023.0f), bits(float(packed >> 30) / 3.0f)};
            break;
        case K::snorm10_10_10_2: {
            const float scale = 1.0f / 511.0f;
            out = {bits(std::max(float(extend(packed & 1023, 10)) * scale, -1.0f)),
                   bits(std::max(float(extend((packed >> 10) & 1023, 10)) * scale, -1.0f)),
                   bits(std::max(float(extend((packed >> 20) & 1023, 10)) * scale, -1.0f)),
                   bits(std::max(float(extend(packed >> 30, 2)), -1.0f))};
            break;
        }
        case K::float11_11_10:
            out = {bits(half_value(std::uint16_t((packed & 0x7ff) << 4))),
                   bits(half_value(std::uint16_t(((packed >> 11) & 0x7ff) << 4))),
                   bits(half_value(std::uint16_t(((packed >> 22) & 0x3ff) << 5))), bits(1.0f)};
            break;
        case K::unorm8_bgra:
            out = {bits(float(raw[2]) / 255.0f), bits(float(raw[1]) / 255.0f), bits(float(raw[0]) / 255.0f),
                   bits(float(raw[3]) / 255.0f)};
            break;
        default: return {};
    }
    for (std::uint32_t c = components; c < 4; ++c) out[c] = c == 3 ? one : 0;
    return out;
}

id<MTLLibrary> compile(id<MTLDevice> device, const std::string& source) {
    auto* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
    if (!library) {
        std::fprintf(stderr, "FAIL: library: %s\n", error.localizedDescription.UTF8String);
        ++failures;
    }
    return library;
}

std::string program(unsigned input_class) {
    const std::string type = class_types[input_class];
    // The translator's prelude: relaxed math with fast contraction.
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n"
                       "#pragma METAL fp math_mode(relaxed)\n#pragma METAL fp contract(fast)\n") +
        std::string(armigo::shader::vertex_fetch_msl()) +
        "struct DescriptorInput { " + type + " value [[attribute(0)]]; };\n"
        "vertex void descriptor_fetch(DescriptorInput in [[stage_in]], uint vertex_id [[vertex_id]],\n"
        "    uint instance_id [[instance_id]], uint base_instance [[base_instance]],\n"
        "    device uint4* out [[buffer(1)]], constant uint& width [[buffer(3)]]) {\n"
        "    out[(instance_id - base_instance) * width + vertex_id] = as_type<uint4>(in.value);\n"
        "}\n"
        "vertex void shader_fetch(uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],\n"
        "    uint base_instance [[base_instance]], device uint4* out [[buffer(1)]],\n"
        "    constant ArmigoVertexFetch& fetch [[buffer(2)]], constant uint& width [[buffer(3)]]) {\n"
        "    out[(instance_id - base_instance) * width + vertex_id] = as_type<uint4>(" +
        class_fetch[input_class] + "(fetch, 0u, vertex_id, instance_id, base_instance));\n"
        "}\n";
}

struct Draw {
    std::uint32_t vertices = vertex_count, instances = 1, base_instance = 0;
    MTLVertexStepFunction step = MTLVertexStepFunctionPerVertex;
    std::uint32_t step_rate = 1;
    std::vector<std::uint16_t> indices;   // an indexed draw when not empty
    std::int32_t base_vertex = 0;
};

std::vector<std::uint32_t> run(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                               const MTLVertexFormat* format, VertexFetchKind kind, std::uint32_t components,
                               bool scaled, id<MTLBuffer> vertices, const Draw& draw, bool& refused) {
    refused = false;
    const bool constant = draw.step == MTLVertexStepFunctionConstant;
    auto* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:format ? @"descriptor_fetch" : @"shader_fetch"];
    descriptor.rasterizationEnabled = NO;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
    if (format) {
        auto* layout = [MTLVertexDescriptor vertexDescriptor];
        layout.attributes[0].format = *format;
        layout.attributes[0].offset = constant ? 0 : offset;
        layout.attributes[0].bufferIndex = 0;
        layout.layouts[0].stride = stride;
        layout.layouts[0].stepFunction = draw.step;
        layout.layouts[0].stepRate = constant ? 0 : draw.step_rate;
        descriptor.vertexDescriptor = layout;
    }
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!pipeline) {
        refused = true;
        return {};
    }
    const std::size_t outputs = std::size_t(draw.vertices) * draw.instances;
    id<MTLBuffer> out = [device newBufferWithLength:outputs * 16 options:MTLResourceStorageModeShared];
    std::memset(out.contents, 0xcd, outputs * 16);
    VertexFetchTable table{};
    table.stream[0] = vertices.gpuAddress;
    // A constant stream is a stride of zero.
    table.stride[0] = constant ? 0 : stride;
    table.step[0] = draw.step == MTLVertexStepFunctionPerInstance ? draw.step_rate
                                                                  : armigo::shader::vertex_fetch_step_vertex;
    table.limit[0] = static_cast<std::uint32_t>(vertices.length);
    table.attribute[0] = armigo::shader::vertex_fetch_attribute(kind, components, 0, constant ? 0 : offset, scaled);
    const std::uint32_t width = draw.vertices;
    auto* pass = [MTLRenderPassDescriptor new];
    pass.renderTargetWidth = 1;
    pass.renderTargetHeight = 1;
    pass.defaultRasterSampleCount = 1;
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setVertexBuffer:vertices offset:0 atIndex:0];
    [encoder setVertexBuffer:out offset:0 atIndex:1];
    [encoder setVertexBytes:&table length:sizeof table atIndex:2];
    [encoder setVertexBytes:&width length:sizeof width atIndex:3];
    [encoder useResource:vertices usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    if (draw.indices.empty()) {
        [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:draw.vertices
                  instanceCount:draw.instances baseInstance:draw.base_instance];
    } else {
        id<MTLBuffer> indices = [device newBufferWithBytes:draw.indices.data()
                                                    length:draw.indices.size() * 2
                                                   options:MTLResourceStorageModeShared];
        [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:draw.indices.size()
                             indexType:MTLIndexTypeUInt16 indexBuffer:indices indexBufferOffset:0
                         instanceCount:draw.instances baseVertex:draw.base_vertex baseInstance:draw.base_instance];
    }
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.error) {
        std::fprintf(stderr, "FAIL: command buffer: %s\n", commands.error.localizedDescription.UTF8String);
        ++failures;
        return {};
    }
    const auto* words = static_cast<const std::uint32_t*>(out.contents);
    return {words, words + outputs * 4};
}

bool same_word(std::uint32_t a, std::uint32_t b, bool float_class) {
    if (a == b) return true;
    if (!float_class) return false;
    float x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return std::isnan(x) && std::isnan(y);
}

// Words that differ, with the first one reported.
std::size_t compare(const char* what, const Format& format, unsigned input_class, bool scaled,
                    const std::vector<std::uint32_t>& expected, const std::vector<std::uint32_t>& actual) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "FAIL: %s %s%s as %s: no result\n", what, format.name, scaled ? " scaled" : "",
                     class_types[input_class]);
        ++failures;
        return 1;
    }
    // NaN payloads of float words may differ.
    const bool float_bits = input_class == 0 || !integer_kind(format.kind) || scaled;
    std::size_t mismatches = 0, first = 0;
    for (std::size_t word = 0; word < expected.size(); ++word)
        if (!same_word(expected[word], actual[word], float_bits) && mismatches++ == 0) first = word;
    if (mismatches) {
        std::fprintf(stderr, "FAIL: %s %s%s as %s: %zu of %zu words differ; vertex %zu component %zu "
                             "expected %08x got %08x\n",
                     what, format.name, scaled ? " scaled" : "", class_types[input_class], mismatches,
                     expected.size(), first / 4, first % 4, expected[first], actual[first]);
        ++failures;
    }
    return mismatches;
}

// Whether Metal's descriptor fetch writes the same words as the format's
// register semantics: float storage read as float; integers read as
// integers; 8- and 16-bit integers converted to float (the scaled formats).
bool descriptor_agrees(const Format& format, unsigned input_class, bool scaled) {
    if (!integer_kind(format.kind)) return input_class == 0;
    if (input_class != 0) return !scaled;
    return scaled && format.kind != VertexFetchKind::uint32 && format.kind != VertexFetchKind::sint32;
}

// Of those, the pairs Metal refuses: an integer format read by an integer
// input of the other signedness.
bool descriptor_refuses(const Format& format, unsigned input_class) {
    using K = VertexFetchKind;
    const bool is_signed = format.kind == K::sint8 || format.kind == K::sint16 || format.kind == K::sint32;
    return integer_kind(format.kind) && input_class != 0 && is_signed != (input_class == 2);
}

void test_formats(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLBuffer> vertices) {
    const auto* bytes = static_cast<const std::uint8_t*>(vertices.contents);
    unsigned agreed = 0, refused_pairs = 0;
    for (unsigned input_class = 0; input_class < 3; ++input_class) {
        id<MTLLibrary> library = compile(device, program(input_class));
        if (!library) return;
        for (const auto& format : formats) {
            for (const bool scaled : {false, true}) {
                if (scaled && !integer_kind(format.kind)) continue;
                std::vector<std::uint32_t> expected(std::size_t(vertex_count) * 4);
                for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
                    const auto words = expected_words(format.kind, format.components, scaled,
                                                      bytes + std::size_t(vertex) * stride + offset);
                    std::copy(words.begin(), words.end(), expected.begin() + std::size_t(vertex) * 4);
                }
                bool refused = false;
                const auto actual = run(device, queue, library, nullptr, format.kind, format.components, scaled,
                                        vertices, Draw{}, refused);
                compare("fetch", format, input_class, scaled, expected, actual);
                if (!descriptor_agrees(format, input_class, scaled)) continue;
                const auto described = run(device, queue, library, &format.format, format.kind, format.components,
                                           scaled, vertices, Draw{}, refused);
                if (refused != descriptor_refuses(format, input_class)) {
                    std::fprintf(stderr, "FAIL: Metal %s %s as %s, unexpectedly\n", refused ? "refused" : "accepted",
                                 format.name, class_types[input_class]);
                    ++failures;
                }
                if (refused) {
                    ++refused_pairs;
                    continue;
                }
                compare("descriptor", format, input_class, scaled, expected, described);
                ++agreed;
            }
        }
    }
    std::printf("formats: %u pairs also checked against Metal's descriptor fetch, %u refused by Metal\n", agreed,
                refused_pairs);
    if (agreed != 68 || refused_pairs != 24) {
        std::fprintf(stderr, "FAIL: expected 68 descriptor comparisons and 24 refusals\n");
        ++failures;
    }
}

void test_draws(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLBuffer> vertices) {
    id<MTLLibrary> library = compile(device, program(1));
    if (!library) return;
    const MTLVertexFormat format = MTLVertexFormatUInt2;
    std::vector<std::uint16_t> indices{5, 0, 9, 3, 77, 3, 41, 88};
    const Draw draws[] = {
        {3, 9, 0, MTLVertexStepFunctionPerInstance, 1},
        {3, 9, 5, MTLVertexStepFunctionPerInstance, 1},
        {3, 11, 7, MTLVertexStepFunctionPerInstance, 3},
        {5, 4, 2, MTLVertexStepFunctionConstant, 0},
        {200, 1, 0, MTLVertexStepFunctionPerVertex, 1, indices, 100},
        {200, 3, 4, MTLVertexStepFunctionPerVertex, 1, indices, 17},
    };
    for (const auto& draw : draws) {
        bool refused = false;
        const auto expected = run(device, queue, library, &format, VertexFetchKind::uint32, 2, false, vertices,
                                  draw, refused);
        const auto actual = run(device, queue, library, nullptr, VertexFetchKind::uint32, 2, false, vertices,
                                draw, refused);
        if (expected.empty() || expected != actual) {
            std::fprintf(stderr, "FAIL: step %d rate %u base instance %u base vertex %d (%zu indices) differs "
                                 "from the descriptor\n",
                         int(draw.step), draw.step_rate, draw.base_instance, draw.base_vertex, draw.indices.size());
            ++failures;
        }
    }
}

// An attribute that ends past its stream's readable length reads zeros, also
// when only its last bytes are past it.
void test_bounds(id<MTLDevice> device, id<MTLCommandQueue> queue, const std::vector<std::uint8_t>& bytes) {
    id<MTLLibrary> library = compile(device, program(1));
    if (!library) return;
    struct Case { VertexFetchKind kind; std::uint32_t components, bytes; };
    const Case cases[] = {{VertexFetchKind::uint32, 4, 16}, {VertexFetchKind::uint8, 3, 3},
                          {VertexFetchKind::unorm10_10_10_2, 4, 4}, {VertexFetchKind::uint16, 3, 6}};
    for (const auto& c : cases) {
        // The last whole attribute is vertex 99's; vertex 100's ends one byte past the stream.
        id<MTLBuffer> stream = [device newBufferWithBytes:bytes.data()
                                                   length:std::size_t(100) * stride + offset + c.bytes - 1
                                                  options:MTLResourceStorageModeShared];
        bool refused = false;
        Draw draw;
        draw.vertices = 102;
        const auto words = run(device, queue, library, nullptr, c.kind, c.components, false, stream, draw, refused);
        if (words.size() != 102 * 4) { ++failures; continue; }
        bool inside = false;
        for (unsigned component = 0; component < c.components && c.kind != VertexFetchKind::unorm10_10_10_2;
             ++component)
            inside |= words[99 * 4 + component] != 0;
        if (c.kind == VertexFetchKind::unorm10_10_10_2) inside = words[99 * 4] || words[99 * 4 + 1];
        for (unsigned vertex = 100; vertex < 102; ++vertex)
            for (unsigned component = 0; component < 4; ++component)
                if (words[vertex * 4 + component]) {
                    std::fprintf(stderr, "FAIL: kind %u vertex %u past the stream limit read %08x\n",
                                 unsigned(c.kind), vertex, words[vertex * 4 + component]);
                    ++failures;
                    vertex = 102;
                    break;
                }
        if (!inside) {
            std::fprintf(stderr, "FAIL: kind %u: the last whole attribute read nothing\n", unsigned(c.kind));
            ++failures;
        }
    }
    id<MTLBuffer> stream = [device newBufferWithBytes:bytes.data() length:bytes.size()
                                              options:MTLResourceStorageModeShared];
    bool refused = false;
    auto disabled = run(device, queue, library, nullptr, VertexFetchKind::disabled, 4, false, stream, Draw{},
                        refused);
    for (const auto word : disabled)
        if (word) {
            std::fprintf(stderr, "FAIL: a disabled attribute read %08x\n", word);
            ++failures;
            break;
        }
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "SKIP: no Metal device\n");
            return 77;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        // Random bytes; the first 16 bits of each attribute count through
        // every value (so every 8-bit value, every 16-bit value and every value
        // of the first 10-bit field appear), the next 16 bits through every
        // value in another order.
        std::vector<std::uint8_t> bytes(std::size_t(vertex_count) * stride + 64);
        std::mt19937 random(20261102);
        for (auto& byte : bytes) byte = static_cast<std::uint8_t>(random());
        for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
            auto* attribute = bytes.data() + std::size_t(vertex) * stride + offset;
            const std::uint16_t first = static_cast<std::uint16_t>(vertex);
            const std::uint16_t second = static_cast<std::uint16_t>(vertex * 40503u + 12345u);
            std::memcpy(attribute, &first, 2);
            std::memcpy(attribute + 2, &second, 2);
        }
        id<MTLBuffer> vertices = [device newBufferWithBytes:bytes.data() length:bytes.size()
                                                    options:MTLResourceStorageModeShared];
        test_formats(device, queue, vertices);
        test_draws(device, queue, vertices);
        test_bounds(device, queue, bytes);
        std::printf("vertex fetch failures: %d\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
