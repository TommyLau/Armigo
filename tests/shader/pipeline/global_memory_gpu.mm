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

#include "global_memory_fixture.h"

#include <array>
#include <cstdio>
#include <vector>

namespace {
using namespace armigo::shader::test::global_memory;

constexpr std::uint32_t sentinel = 0xdeadbeef;
constexpr std::array<std::uint32_t, 8> input_words{
    0x13579bdf, 0x2468ace0, 0x80000001, 0xfedcba98,
    0xabababab, 0xcdcdcdcd, 0xefefefef, 0x12121212};

struct Case {
    const char* name;
    Configuration config{};
    std::uint32_t input_low{0x4000};
    std::uint32_t input_high{0x1234};
    std::uint32_t output_low{0x8000};
    std::uint32_t input_size{16};
    std::uint32_t output_size{16};
    int written_word{};
    std::uint32_t expected{input_words[0]};
    // Shared-memory staging uses STS/LDS, which only exist in compute stages.
    bool compute_only{};
};

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<Case> cases() {
    std::vector<Case> result;
    result.push_back({"nonzero_descriptor_high"});
    Case item{"copied_pair"};
    item.config.copy_pair = true;
    result.push_back(item);
    item = {"positive_immediate_carries_low_word"};
    item.input_low = 0xfffffffc;
    item.config.load_displacement = 4;
    item.expected = input_words[1];
    result.push_back(item);
    item = {"negative_immediate_borrows_below_base"};
    item.input_low = 0;
    item.config.load_displacement = -4;
    item.expected = 0;
    result.push_back(item);
    item = {"negative_immediate_inside_binding"};
    item.config.low_addend = 8;
    item.config.load_displacement = -4;
    item.expected = input_words[1];
    result.push_back(item);
    item = {"negative_immediate_borrows_to_base"};
    item.input_low = 0xfffffffc;
    item.config.low_addend = 4;
    item.config.high_addend = 1;
    item.config.load_displacement = -4;
    result.push_back(item);
    item = {"uncarried_low_wrap_is_outside_binding"};
    item.input_low = 0xfffffffc;
    item.config.low_addend = 4;
    item.expected = 0;
    result.push_back(item);
    // A base whose high word is all ones: the address (0, 0) one carry above
    // it lies below the base, although the high halves differ by one borrow.
    item = {"address_below_all_ones_base_returns_zero"};
    item.input_high = 0xffffffff;
    item.input_low = 0xfffffff8;
    item.config.low_addend = 8;
    item.config.high_addend = 1;
    item.expected = 0;
    result.push_back(item);
    item = {"high_word_changes_runtime_address"};
    item.config.high_addend = 1;
    item.expected = 0;
    result.push_back(item);
    item = {"last_complete_input_word"};
    item.config.load_displacement = 12;
    item.expected = input_words[3];
    result.push_back(item);
    item = {"load_at_descriptor_extent_returns_zero"};
    item.config.load_displacement = 16;
    item.expected = 0;
    result.push_back(item);
    item = {"partial_input_word_returns_zero"};
    item.input_size = 2;
    item.expected = 0;
    result.push_back(item);
    item = {"minimum_signed_immediate_returns_zero"};
    item.config.load_displacement = -0x800000;
    item.expected = 0;
    result.push_back(item);
    item = {"maximum_aligned_signed_immediate_returns_zero"};
    item.config.load_displacement = 0x7ffffc;
    item.expected = 0;
    result.push_back(item);
    item = {"store_immediate_carries_low_word"};
    item.output_low = 0xfffffffc;
    item.config.store_displacement = 4;
    item.written_word = 1;
    result.push_back(item);
    item = {"store_immediate_borrows_below_base"};
    item.output_low = 0;
    item.config.store_displacement = -4;
    item.written_word = -1;
    result.push_back(item);
    item = {"store_at_descriptor_extent_is_skipped"};
    item.config.store_displacement = 16;
    item.written_word = -1;
    result.push_back(item);
    item = {"partial_output_word_is_not_written"};
    item.output_size = 2;
    item.written_word = -1;
    result.push_back(item);
    item = {"shared_table_descriptor_staging"};
    item.config.shared_table = true;
    item.compute_only = true;
    result.push_back(item);
    // Multi-target load: the input base selects which descriptor the load
    // reads. Below 7 it reads the output buffer's first word (the sentinel,
    // not the zero a missed lookup returns) and stores it back unchanged.
    item = {"multi_target_reads_input"};
    item.config.branch_pair = true;
    result.push_back(item);
    item = {"multi_target_reads_output"};
    item.config.branch_pair = true;
    item.input_low = 0;
    item.expected = sentinel;
    result.push_back(item);
    item = {"nonextended_low_word_wrap"};
    item.config.extended = false;
    item.config.high_source = AddressSource::literal_nonzero;
    item.config.forge_store_high = true;
    item.input_low = 0xfffffffc;
    item.config.load_displacement = 4;
    item.expected = input_words[1];
    result.push_back(item);
    return result;
}

MslArtifact graphics_artifact(Stage stage, const Configuration& config) {
    struct GraphicsFacts final : FactSource {
        Knowledge<FactValue> lookup(FactKey key) const override {
            if (key == FactKey{FactKind::depth_convention})
                return Known<FactValue>{DepthConvention::negative_one_to_one,
                    {OriginKind::explicit_option, "synthetic-storage-graphics"}};
            return Unknown{"Synthetic storage descriptors remain runtime values"};
        }
    } source;
    auto guest = image(config);
    guest.stage = stage;
    if (stage == Stage::vertex) {
        guest.bytes.resize(guest.bytes.size() - 8); // Replace EXIT with point output.
        const auto append = [&](std::uint64_t word) {
            for (unsigned byte = 0; byte != 8; ++byte)
                guest.bytes.push_back(std::byte((word >> (8 * byte)) & 255));
        };
        const auto emit = [&](std::uint64_t word) {
            if (guest.bytes.size() % 32 == 0) append(0);
            append(word);
        };
        emit(literal(0, 0));
        emit(literal(1, 0));
        emit(literal(2, 0));
        emit(literal(3, 0x3f800000));
        // Independently authored AST position and point-size exports.
        emit(0xeff000000007ff00ull | (0x70ull << 20) | (3ull << 47));
        emit(literal(6, 0x41000000));
        emit(0xeff000000007ff06ull | (0x6cull << 20));
        emit(0xe30000000007000full);
    }
    FactReader facts(source);
    auto decoded = take(decode(guest, facts, {}));
    auto registers = take(emit_ir(decoded, facts, {}));
    auto ssa = take(construct_ssa(std::move(registers), {}));
    auto legal = take(legalize(std::move(ssa), facts, profile()));
    auto structured = take(structure(std::move(legal), {}));
    return take(emit_msl(structured, take(plan_resources(structured, profile())), {}));
}

void run_case(id<MTLDevice> device, id<MTL4Compiler> compiler,
              id<MTL4CommandQueue> queue, const Case& item, Stage stage = Stage::compute) {
    const auto shader = stage == Stage::compute ? artifact(item.config)
                                               : graphics_artifact(stage, item.config);
    NSError* error = nil;
    MTLCompileOptions* compile_options = [MTLCompileOptions new];
    compile_options.languageVersion = MTLLanguageVersion4_0;
    compile_options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:shader.source.c_str()]
        options:compile_options error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = [NSString stringWithUTF8String:shader.entry_point.c_str()];
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLRenderPipelineState> render_pipeline = nil;
    id<MTLTexture> target = nil;
    id<MTLLibrary> companion = nil;
    if (stage == Stage::compute) {
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
    } else {
        companion = [device newLibraryWithSource:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "struct Position { float4 p [[position]]; };\n"
             "vertex Position storage_vertex(uint id [[vertex_id]]) {\n"
             "float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)}; return {float4(p[id],0,1)}; }\n"
             "fragment void storage_fragment() {}\n"
            options:compile_options error:&error];
        if (!companion) throw std::runtime_error(error.localizedDescription.UTF8String);
        MTL4LibraryFunctionDescriptor* other = [MTL4LibraryFunctionDescriptor new];
        other.library = companion;
        other.name = stage == Stage::vertex ? @"storage_fragment" : @"storage_vertex";
        MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = stage == Stage::vertex ? function : other;
        descriptor.fragmentFunctionDescriptor = stage == Stage::fragment ? function : other;
        descriptor.inputPrimitiveTopology = stage == Stage::vertex
            ? MTLPrimitiveTopologyClassPoint : MTLPrimitiveTopologyClassTriangle;
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        descriptor.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
        render_pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!render_pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        MTLTextureDescriptor* texture = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
        texture.storageMode = MTLStorageModePrivate;
        texture.usage = MTLTextureUsageRenderTarget;
        target = [device newTextureWithDescriptor:texture];
        check(target != nil, "Graphics storage target allocation failed");
    }

    // Guest descriptor addresses are synthetic and deliberately unrelated to
    // Metal addresses. Reflection selects each backing allocation; guest pairs
    // determine only the offset within that allocation.
    std::array<std::uint32_t, 16> constants{};
    constants[input_descriptor_offset / 4] = item.input_low;
    constants[input_descriptor_offset / 4 + 1] = item.input_high;
    constants[input_descriptor_offset / 4 + 2] = item.input_size;
    constants[output_descriptor_offset / 4] = item.output_low;
    constants[output_descriptor_offset / 4 + 1] = 0x8765;
    constants[output_descriptor_offset / 4 + 2] = item.output_size;
    std::array<std::uint32_t, constant_buffer_slot_count> sizes{};
    sizes[descriptor_slot] = sizeof(constants);
    std::array<std::uint32_t, 8> initial;
    initial.fill(sentinel);
    id<MTLBuffer> input = [device newBufferWithBytes:input_words.data()
        length:sizeof(input_words) options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithBytes:initial.data()
        length:sizeof(initial) options:MTLResourceStorageModeShared];
    id<MTLBuffer> cbuf = [device newBufferWithBytes:constants.data()
        length:sizeof(constants) options:MTLResourceStorageModeShared];
    id<MTLBuffer> support = [device newBufferWithBytes:sizes.data()
        length:sizeof(sizes) options:MTLResourceStorageModeShared];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = profile().buffer_slots;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 4;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    check(input && output && cbuf && support && table && residency && allocator && commands,
        "Metal global-memory resource allocation failed");
    for (const auto& resource : shader.reflection.resources) {
        const auto* buffer = std::get_if<BufferUse>(&resource.use);
        check(buffer != nullptr, "Synthetic copy must reflect only buffers");
        id<MTLBuffer> allocation = nil;
        if (const auto* source = std::get_if<ConstantBufferSource>(&buffer->source)) {
            check(source->slot == descriptor_slot, "Unexpected constant-buffer slot");
            allocation = cbuf;
        } else if (const auto* source = std::get_if<StorageBufferSource>(&buffer->source)) {
            if (source->descriptor == TextureKey{descriptor_slot, input_descriptor_offset}) allocation = input;
            if (source->descriptor == TextureKey{descriptor_slot, output_descriptor_offset}) allocation = output;
        } else if (const auto* source = std::get_if<SupportBufferSource>(&buffer->source)) {
            check(source->schema == constant_buffer_support_schema &&
                source->version == constant_buffer_support_version, "Unexpected support-buffer ABI");
            allocation = support;
        }
        check(allocation != nil, "Reflected resource has no synthetic backing");
        const auto* location = std::get_if<MetalRange>(&resource.location);
        check(location && location->space == MetalNamespace::buffer && location->count == 1,
            "Synthetic direct bindings need single Metal buffer ranges");
        [table setAddress:allocation.gpuAddress atIndex:location->first];
    }
    for (id<MTLAllocation> allocation in @[input, output, cbuf, support])
        [residency addAllocation:allocation];
    if (target) [residency addAllocation:target];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    if (stage == Stage::compute) {
        id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        check(encoder != nil, "Metal compute encoder creation failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:table];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
    } else {
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
        id<MTL4RenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
        check(encoder != nil, "Metal graphics encoder creation failed");
        [encoder setRenderPipelineState:render_pipeline];
        [encoder setArgumentTable:table atStages:stage == Stage::vertex ? MTLRenderStageVertex : MTLRenderStageFragment];
        // One vertex or one covered fragment performs the store, with no
        // inter-invocation write race hiding behind identical output values.
        [encoder setScissorRect:MTLScissorRect{0, 0, 1, 1}];
        [encoder drawPrimitives:stage == Stage::vertex ? MTLPrimitiveTypePoint : MTLPrimitiveTypeTriangle
                   vertexStart:0 vertexCount:stage == Stage::vertex ? 1 : 3];
        [encoder endEncoding];
    }
    [commands endCommandBuffer];
    NSMutableArray* retained = [NSMutableArray arrayWithArray:@[library, input, output, cbuf, support,
        table, residency, allocator, commands]];
    if (pipeline) [retained addObject:pipeline];
    if (render_pipeline) [retained addObjectsFromArray:@[render_pipeline, target, companion]];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* commit = [MTL4CommitOptions new];
    [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit];
    check(dispatch_semaphore_wait(completed,
        dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0, "Metal execution timed out");
    if (execution_error) throw std::runtime_error(execution_error.localizedDescription.UTF8String);
    const auto* actual = static_cast<const std::uint32_t*>(output.contents);
    for (unsigned word = 0; word < initial.size(); ++word) {
        const auto expected = static_cast<int>(word) == item.written_word ? item.expected : sentinel;
        if (actual[word] != expected) {
            std::fprintf(stderr, "%s word %u: actual 0x%08x expected 0x%08x\n",
                item.name, word, actual[word], expected);
            throw std::runtime_error("LDG.E/STG.E GPU result or output guard mismatch");
        }
    }
    const auto* input_after = static_cast<const std::uint32_t*>(input.contents);
    for (unsigned word = 0; word < input_words.size(); ++word)
        check(input_after[word] == input_words[word], "Read-only input storage was modified");
    std::printf("PASS: %s stage=%u\n", item.name, unsigned(stage));
}
} // namespace

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            check(device != nil, "A Metal device is required for global-memory execution tests");
            NSError* error = nil;
            MTL4CompilerDescriptor* descriptor = [MTL4CompilerDescriptor new];
            id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:descriptor error:&error];
            id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
            check(compiler && queue, "Metal 4 compiler and command queue are required");
            for (const auto stage : {Stage::compute, Stage::vertex, Stage::fragment})
                for (const auto& item : cases()) {
                    if (item.compute_only && stage != Stage::compute) continue;
                    run_case(device, compiler, queue, item, stage);
                }
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}
