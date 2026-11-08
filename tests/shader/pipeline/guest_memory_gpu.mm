// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Guest-memory windows, their miss record and the render-target height read
// from the draw-parameter record, executed on Metal. A register-IR fragment
// program goes through SSA, legalization, structuring, planning and emission;
// its guest accesses reach a buffer through the window table by device
// address.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline/phases.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }

struct Builder {
    Program p;
    Builder() {
        p.stage = Stage::fragment; p.entry = FunctionId{0};
        p.functions.push_back({0, BlockId{0}, {BlockId{0}}, {}, {}});
        p.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
    }
    ValueId value(ValueType type = ValueType::word32) {
        p.values.push_back({type, ValueKind::temporary});
        return ValueId{static_cast<std::uint32_t>(p.values.size() - 1)};
    }
    void add(Operation op) {
        op.source = {static_cast<Pc>(p.operations.size() * 16), {}};
        op.effects = required_effects(op);
        p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
        p.operations.push_back(std::move(op));
    }
    // A guest access at high:low plus displacement; data words for stores
    // and atomics.
    std::vector<ValueId> guest(Opcode opcode, std::uint64_t address, std::int32_t displacement, std::uint8_t bytes,
                               ScalarType scalar, std::vector<Operand> data = {}, std::optional<AtomicKind> atomic = {},
                               std::uint8_t elements = 1) {
        MemoryInfo m;
        m.space = AddressSpace::guest;
        m.access = opcode == Opcode::load ? Access::read : opcode == Opcode::store ? Access::write : Access::read_write;
        m.byte_offset = static_cast<std::uint32_t>(displacement);
        m.element_bytes = bytes; m.elements = elements; m.alignment = static_cast<std::uint8_t>(bytes * elements); m.atomic = atomic;
        m.scope = atomic ? MemoryScope::device : MemoryScope::invocation;
        Operation op;
        op.opcode = opcode;
        op.numeric = {scalar_domain(scalar), scalar};
        op.inputs = {word(static_cast<std::uint32_t>(address)), word(static_cast<std::uint32_t>(address >> 32))};
        m.dynamic_byte_offset = InputIndex{0}; m.global_address_high = InputIndex{1};
        for (auto& d : data) { m.data.push_back(InputIndex{static_cast<std::uint16_t>(op.inputs.size())}); op.inputs.push_back(d); }
        std::vector<ValueId> results;
        if (opcode != Opcode::store)
            for (unsigned i = 0; i < (atomic ? 1u : elements); ++i) { results.push_back(value()); op.outputs.push_back(results.back()); }
        op.info = m;
        add(std::move(op));
        return results;
    }
    ValueId height() {
        IoInfo io; io.semantic = IoSemantic::render_target_height; io.type = {ScalarType::u32, 1};
        Operation op; op.opcode = Opcode::load_io; op.numeric = {Domain::unsigned_integer, ScalarType::u32};
        const auto v = value(); op.outputs.push_back(v); op.info = io;
        add(std::move(op)); return v;
    }
    void export_color(const std::array<Operand, 4>& words) {
        for (unsigned c = 0; c < 4; ++c) {
            IoInfo io; io.semantic = IoSemantic::color; io.component_mask = 1u << c;
            io.type = {ScalarType::u32, 4}; io.output_storage = true; io.data = {InputIndex{0}};
            Operation op; op.opcode = Opcode::store_io; op.numeric = {Domain::unsigned_integer, ScalarType::u32};
            op.inputs = {words[c]}; op.info = io;
            add(std::move(op));
        }
    }
};

struct NoFacts final : FactSource {
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::fragment_output_type)
            return Known<FactValue>{IoType{ScalarType::u32, 4}, {OriginKind::explicit_option, "guest-memory-test"}};
        return Unknown{"No fact in this test"};
    }
};

MslArtifact translate(Program program, std::uint32_t tag = 0) {
    auto take = []<class T>(Result<T> r) { if (auto* e = std::get_if<Diagnostic>(&r)) throw std::runtime_error(e->message); return std::get<T>(std::move(r)); };
    Limits limits; NoFacts source; FactReader facts(source, limits);
    const MetalProfile profile{"guest-memory-test", 1, 31, 128, 16, true, false, true, false, false, false};
    auto ssa = take(construct_ssa(RegisterProgram{std::move(program)}, limits));
    auto legal = take(legalize(std::move(ssa), facts, profile));
    auto structured = take(structure(std::move(legal), limits));
    auto plan = take(plan_resources(structured, profile));
    EmitOptions options; options.guest_memory_tag = tag;
    return take(emit_msl(structured, plan, options));
}

struct Gpu {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTL4Compiler> compiler;
    id<MTL4CommandQueue> queue;
    id<MTLLibrary> vertex;
    Gpu() {
        check(device != nil, "Metal device");
        NSError* error = nil;
        compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        queue = [device newMTL4CommandQueue];
        vertex = library(R"(#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; };
vertex V guest_test_vertex(uint i [[vertex_id]]) {
    const float2 p[3] = {float2(-2, -2), float2(6, -2), float2(-2, 6)};
    V out; out.position = float4(p[i], 0, 1); return out;
})");
    }
    id<MTLLibrary> library(const std::string& source) {
        MTLCompileOptions* options = [MTLCompileOptions new]; options.languageVersion = MTLLanguageVersion4_0;
        NSError* error = nil;
        auto result = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
        if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
        return result;
    }
    // Draws one pixel with the window table and the draw-parameter record
    // bound where the plan placed them and returns the exported words; the
    // table's words after the draw land in windows.
    std::array<std::uint32_t, 4> run(const MslArtifact& artifact, std::vector<std::uint32_t>& windows,
                                     NSArray<id<MTLAllocation>>* resident, std::uint32_t height = 0) {
        NSError* error = nil;
        MTL4LibraryFunctionDescriptor* vs = [MTL4LibraryFunctionDescriptor new]; vs.library = vertex; vs.name = @"guest_test_vertex";
        MTL4LibraryFunctionDescriptor* fs = [MTL4LibraryFunctionDescriptor new]; fs.library = library(artifact.source); fs.name = @"armigo_fragment_main";
        MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = vs; descriptor.fragmentFunctionDescriptor = fs;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Uint;
        auto pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        auto table_buffer = [device newBufferWithBytes:windows.data() length:windows.size() * 4 options:MTLResourceStorageModeShared];
        // Version 2 of the draw-parameter record: first instance, height.
        const std::array<std::uint32_t, 2> parameters{0, height};
        auto draw_buffer = [device newBufferWithBytes:parameters.data() length:sizeof(parameters) options:MTLResourceStorageModeShared];
        auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint width:1 height:1 mipmapped:NO];
        td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
        auto target = [device newTextureWithDescriptor:td];
        auto readback = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        MTL4ArgumentTableDescriptor* at = [MTL4ArgumentTableDescriptor new]; at.maxBufferBindCount = 31;
        auto table = [device newArgumentTableWithDescriptor:at error:&error];
        if (artifact.reflection.draw_support_buffer_index)
            [table setAddress:draw_buffer.gpuAddress atIndex:*artifact.reflection.draw_support_buffer_index];
        for (const auto& resource : artifact.reflection.resources) {
            const auto& range = std::get<MetalRange>(resource.location);
            const auto& buffer = std::get<BufferUse>(resource.use);
            const auto& support = std::get<SupportBufferSource>(buffer.source);
            check(support.schema == guest_memory_schema, "only the window table is bound");
            [table setAddress:table_buffer.gpuAddress atIndex:range.first];
        }
        auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        for (id<MTLAllocation> a in @[table_buffer, draw_buffer, target, readback]) [residency addAllocation:a];
        for (id<MTLAllocation> a in resident) [residency addAllocation:a];
        [residency commit];
        auto allocator = [device newCommandAllocator]; auto commands = [device newCommandBuffer];
        [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear; pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        auto render = [commands renderCommandEncoderWithDescriptor:pass];
        [render setRenderPipelineState:pipeline]; [render setArgumentTable:table atStages:MTLRenderStageFragment];
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [render endEncoding];
        auto copy = [commands computeCommandEncoder];
        [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        [copy copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(1, 1, 1)
                     toBuffer:readback destinationOffset:0 destinationBytesPerRow:16 destinationBytesPerImage:0];
        [copy endEncoding]; [commands endCommandBuffer];
        dispatch_semaphore_t done = dispatch_semaphore_create(0); __block NSError* failure = nil;
        MTL4CommitOptions* commit = [MTL4CommitOptions new];
        [commit addFeedbackHandler:^(id<MTL4CommitFeedback> f) { failure = f.error; dispatch_semaphore_signal(done); }];
        id<MTL4CommandBuffer> submitted[] = {commands}; [queue commit:submitted count:1 options:commit];
        check(dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0, "GPU completion");
        if (failure) throw std::runtime_error(failure.localizedDescription.UTF8String);
        std::array<std::uint32_t, 4> pixel{};
        std::memcpy(pixel.data(), readback.contents, 16);
        std::memcpy(windows.data(), table_buffer.contents, windows.size() * 4);
        return pixel;
    }
};

// Two pools sorted by guest start: guest [base2, base2 + 16) backed by the
// data buffer's bytes 64..79 (entry 1, found by binary search), and guest
// [base, base + 64) by its first 64 bytes (entry 2, also named by the hint
// of base's 2 MiB chunk).
constexpr std::uint64_t base = 0x7db6000000ull, base2 = 0x7c36001000ull;
std::vector<std::uint32_t> table_for(id<MTLBuffer> data) {
    std::vector<std::uint32_t> t(guest_memory_bytes / 4);
    t[0] = 3;
    auto entry = [&](unsigned e, std::uint64_t begin, std::uint64_t size, std::uint64_t device) {
        const unsigned w = guest_memory_header_words + e * guest_memory_entry_words;
        const std::uint64_t delta = device - begin;
        t[w] = std::uint32_t(begin); t[w + 1] = std::uint32_t(begin >> 32);
        t[w + 2] = std::uint32_t(begin + size); t[w + 3] = std::uint32_t((begin + size) >> 32);
        t[w + 4] = std::uint32_t(delta); t[w + 5] = std::uint32_t(delta >> 32);
    };
    entry(1, base2, 16, data.gpuAddress + 64);
    entry(2, base, 64, data.gpuAddress);
    const std::uint64_t chunk = base >> 21;
    t[guest_memory_header_words + (chunk & 8191) * guest_memory_entry_words + 6] = std::uint32_t(chunk << 13) | 2;
    return t;
}
} // namespace

int main() { @autoreleasepool { try {
    Gpu gpu;
    std::array<std::uint32_t, 20> words{};
    for (unsigned i = 0; i < words.size(); ++i) words[i] = 0x01010101u * (i + 1);
    words[1] = 0x8001fffeu; // halfword 1 (bytes 6-7) is 0x8001: a negative s16
    words[16] = 0xcafef00du;
    auto data = [gpu.device newBufferWithBytes:words.data() length:sizeof(words) options:MTLResourceStorageModeShared];

    Builder b;
    // A displaced word that carries into the high address half, as the guest
    // composes 64-bit addresses: (base - 4) + 8 = base + 4.
    const auto first = b.guest(Opcode::load, base - 4, 8, 4, ScalarType::u32)[0];
    const auto half = b.guest(Opcode::load, base + 6, 0, 2, ScalarType::s16)[0];
    b.guest(Opcode::store, base + 8, 0, 4, ScalarType::u32, {word(0x12345678u)});
    const auto old = b.guest(Opcode::atomic, base + 12, 0, 4, ScalarType::u32, {word(5)}, AtomicKind::add)[0];
    const auto second = b.guest(Opcode::load, base2, 0, 4, ScalarType::u32)[0];
    b.export_color({first, half, old, second});
    const auto artifact = translate(std::move(b.p));
    check(artifact.source.find("armigo_guest_address") != std::string::npos, "guest translation helper");
    auto table = table_for(data);
    auto pixel = gpu.run(artifact, table, @[data]);
    const auto* after = static_cast<const std::uint32_t*>(data.contents);
    if (pixel != std::array<std::uint32_t, 4>{words[1], 0xffff8001u, words[3], 0xcafef00du} ||
        after[2] != 0x12345678u || after[3] != words[3] + 5 || after[16] != 0xcafef00du) {
        std::fprintf(stderr, "guest: %08x %08x %08x %08x stored %08x atomic %08x window2 %08x\n", pixel[0], pixel[1], pixel[2], pixel[3], after[2], after[3], after[16]);
        throw std::runtime_error("guest memory window result");
    }
    check(table[1] == 0 && table[2] == 0 && table[3] == 0 && table[4] == 0, "accesses inside the windows record no miss");

    // A 128-bit load and a 64-bit store: one translation for all words.
    Builder wide;
    const auto quad = wide.guest(Opcode::load, base + 16, 0, 4, ScalarType::u32, {}, {}, 4);
    wide.guest(Opcode::store, base + 40, 0, 4, ScalarType::u32, {word(0xa1u), word(0xb2u)}, {}, 2);
    wide.export_color({quad[0], quad[1], quad[2], quad[3]});
    table = table_for(data);
    pixel = gpu.run(translate(std::move(wide.p)), table, @[data]);
    if (pixel != std::array<std::uint32_t, 4>{words[4], words[5], words[6], words[7]} || after[10] != 0xa1u || after[11] != 0xb2u)
        throw std::runtime_error("wide guest memory access");
    check(table[1] == 0, "wide accesses inside a window record no miss");

    // A load past the first window, one straddling its end, a misaligned
    // word inside it, a dropped store and a dropped atomic: five misses, the
    // first one recorded with the tag. The count starts two short of
    // wrapping; the latch keeps the first record regardless.
    Builder miss;
    const auto absent = miss.guest(Opcode::load, base + 64, 0, 4, ScalarType::u32)[0];
    const auto straddle = miss.guest(Opcode::load, base + 62, 0, 4, ScalarType::u32)[0];
    const auto misaligned = miss.guest(Opcode::load, base + 2, 0, 4, ScalarType::u32)[0];
    miss.guest(Opcode::store, base2 + 16, 0, 4, ScalarType::u32, {word(0xbadbadu)});
    miss.guest(Opcode::atomic, base2 - 4, 0, 4, ScalarType::u32, {word(1)}, AtomicKind::add);
    miss.export_color({absent, straddle, misaligned, word(9)});
    const auto snapshot = std::vector<std::uint32_t>(after, after + words.size());
    const auto missing = translate(std::move(miss.p), 0x5eed1234u);
    table = table_for(data);
    table[1] = 0xfffffffeu;
    pixel = gpu.run(missing, table, @[data]);
    check(pixel == (std::array<std::uint32_t, 4>{0, 0, 0, 9}), "addresses outside every window or misaligned read zero");
    check(std::equal(snapshot.begin(), snapshot.end(), after), "missing stores and atomics leave memory unchanged");
    if (table[1] != 3 || table[5] != 1 || table[2] != 0x5eed1234u || table[3] != std::uint32_t(base + 64) ||
        table[4] != std::uint32_t((base + 64) >> 32)) {
        std::fprintf(stderr, "miss record: count %u latch %u tag %08x address %08x%08x\n", table[1], table[5], table[2], table[4], table[3]);
        throw std::runtime_error("guest memory miss record");
    }
    // A later miss while the latch is set counts but keeps the first record.
    Builder later;
    later.export_color({later.guest(Opcode::load, base2 + 32, 0, 4, ScalarType::u32)[0], word(1), word(2), word(3)});
    gpu.run(translate(std::move(later.p), 0x0badf00du), table, @[data]);
    check(table[1] == 4 && table[5] == 1 && table[2] == 0x5eed1234u && table[3] == std::uint32_t(base + 64),
          "the latch keeps the first miss record");

    Builder height;
    const auto h = height.height();
    height.export_color({h, word(1), word(2), word(3)});
    const auto height_artifact = translate(std::move(height.p));
    check(height_artifact.reflection.reads_render_target_height && height_artifact.reflection.draw_support_buffer_index,
          "render-target height reads the draw-parameter record");
    check(height_artifact.source.find("function_constant(13)") == std::string::npos, "height is not a function constant");
    std::vector<std::uint32_t> empty(guest_memory_bytes / 4);
    for (const std::uint32_t value : {720u, 1080u, 333u}) {
        pixel = gpu.run(height_artifact, empty, @[], value);
        check(pixel[0] == value, "render-target height from the draw-parameter record");
    }
    std::puts("Guest memory windows, miss record and render-target height Metal execution passed");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; } } }
