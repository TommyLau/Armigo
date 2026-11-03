// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Compute execution admits QMD constant buffers, decodes storage address, upper
// address byte, and 17-bit size words, validates dispatch shape, and uses the
// shared texture and sampler binding path. Each execution owns its Metal command
// objects, waits synchronously, and writes completed results back to guest memory.

#include "compute_dispatch.h"
#include <armigo/graphics/texture.h>
#include "compute_qmd.h"
#include "texture_fact.h"
#include "texture_format.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace armigo::platform {
namespace {
using namespace graphics;
namespace gm = shader::graphics_msl;

// Matches draw.mm's legacy staging discipline: one bounded per-execution
// upload budget across constant buffers, storage staging and textures.
constexpr std::size_t staging_budget = 256u << 20;

// CommandResult carries a borrowed diagnostic view that must outlive the
// callback return, so thrown diagnostics live in a bounded static ring.
// Dispatch callbacks are serialized by the owning backend, which makes a
// process-wide ring the correct scope.
[[noreturn]] void fail(std::string detail,
                       CommandStatus status = CommandStatus::Unsupported) {
    static std::array<std::array<char, 512>, 64> diagnostics;
    static std::size_t next = 0;
    auto& slot = diagnostics[next];
    next = (next + 1) % diagnostics.size();
    const auto length = std::min<std::size_t>(detail.size(), slot.size() - 1);
    if (length) std::memcpy(slot.data(), detail.data(), length);
    slot[length] = '\0';
    throw CommandResult{status, std::string_view(slot.data(), length)};
}
template <class T>
T value(std::variant<T, TextureError> item) {
    if (const auto* failure = std::get_if<TextureError>(&item)) {
        fail(std::string(failure->message()),
             failure->code == TextureErrorCode::MemoryRead ? CommandStatus::MemoryFault
                                                           : CommandStatus::Unsupported);
    }
    return std::get<T>(std::move(item));
}

// Reads exactly bytes.size() bytes from guest memory or fails.
void read_guest(const GpuAddressSpace& memory, std::uint64_t address,
                std::span<std::byte> bytes) {
    if (address > std::numeric_limits<std::uint64_t>::max() - bytes.size() ||
        !memory.read(address, bytes).ok()) {
        fail("Compute dispatch guest read failed", CommandStatus::MemoryFault);
    }
}

std::uint32_t load_word(std::span<const std::byte, 4> bytes) noexcept {
    return std::uint32_t{std::to_integer<std::uint8_t>(bytes[0])} |
           (std::uint32_t{std::to_integer<std::uint8_t>(bytes[1])} << 8) |
           (std::uint32_t{std::to_integer<std::uint8_t>(bytes[2])} << 16) |
           (std::uint32_t{std::to_integer<std::uint8_t>(bytes[3])} << 24);
}

// The decoded queue-meta-data constant buffer for one reflected slot, or a
// structured refusal naming the program and slot.
const ComputeConstantBuffer& constant_buffer(const ComputeLaunch& launch, unsigned slot) {
    return launch.constant_buffers[slot];
}

// Compute pools use the same TSC/TIC registers the shared texture resolver
// reads for graphics stages; a compute RenderStop carries them in its own
// register file snapshot (register 0x557 sampler pool, 0x55d texture pool).
constexpr std::size_t sampler_pool_register = 0x557;
constexpr std::size_t texture_pool_register = 0x55d;
constexpr std::size_t texture_descriptor_size = 0x20;

std::uint64_t pool_base(const RenderStop& stop, std::size_t register_index) {
    return static_cast<std::uint64_t>(stop.registers[register_index]) |
           (std::uint64_t{stop.registers[register_index + 1]} << 32);
}

// Resolves one reflected texture handle word against the compute pools,
// mirroring the shared resolver's handle split and pool reads on the compute
// register file.
ResolvedTextureBinding resolve_compute_texture(const RenderStop& stop,
                                               const ComputeLaunch& launch,
                                               const GpuAddressSpace& memory,
                                               unsigned slot, std::uint32_t offset,
                                               bool resolve_sampler) {
    const auto& binding = constant_buffer(launch, slot);
    if (!binding.enabled || binding.size < offset ||
        binding.size - offset < 4) {
        fail("Compute texture handle is outside its queue-meta-data constant buffer: slot " +
             std::to_string(slot), CommandStatus::MemoryFault);
    }
    std::array<std::byte, 4> handle_bytes{};
    read_guest(memory, binding.address + offset, handle_bytes);
    const std::uint32_t handle = load_word(handle_bytes);
    const std::uint32_t texture_id = handle & 0xfffff;
    const std::uint32_t sampler_id = handle >> 20;
    if (texture_id > stop.registers[texture_pool_register + 2]) {
        fail("Compute texture pool index exceeds its limit: " + std::to_string(texture_id));
    }
    if (resolve_sampler && sampler_id > stop.registers[sampler_pool_register + 2]) {
        fail("Compute sampler pool index exceeds its limit: " + std::to_string(sampler_id));
    }
    const std::uint64_t texture_descriptor_address =
        pool_base(stop, texture_pool_register) + std::uint64_t{texture_id} * texture_descriptor_size;
    std::array<std::byte, texture_descriptor_size> texture_bytes{};
    read_guest(memory, texture_descriptor_address, texture_bytes);
    const auto texture_result = decode_texture_descriptor(texture_bytes);
    if (const auto* failure = std::get_if<TextureError>(&texture_result)) {
        fail(std::string(failure->message()));
    }
    SamplerDescriptor sampler{};
    if (resolve_sampler) {
        const std::uint64_t sampler_descriptor_address =
            pool_base(stop, sampler_pool_register) +
            std::uint64_t{sampler_id} * texture_descriptor_size;
        std::array<std::byte, texture_descriptor_size> sampler_bytes{};
        read_guest(memory, sampler_descriptor_address, sampler_bytes);
        sampler = value(decode_sampler_descriptor(sampler_bytes));
    }
    ResolvedTextureBinding resolved{};
    resolved.handle = handle;
    resolved.texture_id = texture_id;
    resolved.sampler_id = sampler_id;
    resolved.texture_descriptor_address = texture_descriptor_address;
    resolved.sampler_descriptor_address = 0;
    resolved.texture = std::get<TextureDescriptor>(texture_result);
    resolved.sampler = sampler;
    resolved.verified_backing_bytes = 0;
    resolved.is_null = resolved.texture.address == 0;
    return resolved;
}

// One shared MTLBuffer staging the guest bytes of a binding.
struct Upload {
    std::size_t bytes = 0;
    void charge(std::size_t amount) {
        if (amount > staging_budget - bytes) fail("Compute dispatch staging budget exceeded",
                                                  CommandStatus::BudgetExceeded);
        bytes += amount;
    }
};

// Device-owned compute pipeline cache. Draws are serialized by the callback
// contract, so an ordinary map is the correct scope (same reasoning as the
// legacy draw pipeline cache in draw.mm). Keys never contain host pointers or
// guest addresses, only the compiled artifact bytes and entry point.
std::map<std::string, id<MTLComputePipelineState>>& pipeline_cache() noexcept {
    static std::map<std::string, id<MTLComputePipelineState>> pipelines;
    return pipelines;
}

CommandResult execute(MetalComputeContext& context, const RenderStop& stop,
                      const GpuAddressSpace& memory) {
    if (!stop.has_compute_descriptor || !stop.compute || !stop.programs[0].enabled) {
        fail("Compute dispatch requires a launch with a queue meta data descriptor");
    }
    if (!context.lookup) fail("Compute dispatch has no compiled stage lookup");
    const auto lookup = context.lookup(context.lookup_context, stop, memory);
    if (const auto* refusal = std::get_if<CommandResult>(&lookup)) return *refusal;
    const CompiledGraphicsStage& stage = *std::get<std::shared_ptr<const CompiledGraphicsStage>>(
        lookup);
    if (stage.stage != shader_input::Stage::Compute) fail("Compute lookup returned a non-compute stage");

    const ComputeLaunch& launch = *stop.compute;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory) {
        fail("Compute dispatch requires a unified-memory Metal 4 device");
    }
    NSError* error = nil;
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    auto* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 16;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor
                                                                    error:&error];
    if (!queue || !compiler || !allocator || !commands || !residency) {
        fail("Compute dispatch Metal allocation failed");
    }
    NSMutableArray* retained = [NSMutableArray
        arrayWithArray:@[queue, compiler, allocator, commands, residency]];
    const auto retain_resource = [&](id resource) {
        if (!resource) fail("Compute dispatch Metal allocation failed");
        [retained addObject:resource];
        if ([resource conformsToProtocol:@protocol(MTLAllocation)])
            [residency addAllocation:resource];
    };
    Upload staged;

    // Argument table: buffer namespace 0..7 is the queue meta data slots,
    // 8..30 the reflected storage buffers; 16 texture and sampler slots each.
    auto* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 31;
    table_descriptor.maxTextureBindCount = 16;
    table_descriptor.maxSamplerStateBindCount = 16;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor
                                                                   error:&error];
    retain_resource(table);

    // Queue-meta-data constant buffers: slot i binds Metal buffer i with the
    // exact published size.
    for (unsigned slot = 0; slot < launch.constant_buffers.size(); ++slot) {
        const auto& binding = launch.constant_buffers[slot];
        if (!binding.enabled) continue;
        std::vector<std::byte> bytes(binding.size);
        read_guest(memory, binding.address, bytes);
        staged.charge(bytes.size());
        id<MTLBuffer> buffer = [device newBufferWithBytes:bytes.data() length:bytes.size()
                                                  options:MTLResourceStorageModeShared];
        retain_resource(buffer);
        [table setAddress:buffer.gpuAddress atIndex:slot];
    }

    // Storage buffers: decode each reflected descriptor from its queue
    // meta-data constant buffer, stage the backing bytes, and remember the
    // guest ranges to write back after GPU completion.
    struct StorageWriteback {
        std::uint64_t address = 0;
        std::uint32_t size = 0;
        id<MTLBuffer> buffer;
    };
    std::vector<StorageWriteback> writeback;
    for (const auto& descriptor : stage.storage_buffers) {
        const auto& binding = constant_buffer(launch, descriptor.cbuf_slot);
        if (!binding.enabled || descriptor.descriptor_byte_offset > binding.size ||
            binding.size - descriptor.descriptor_byte_offset < 12) {
            fail("Compute storage descriptor is outside its queue-meta-data constant buffer: "
                 "slot " + std::to_string(descriptor.cbuf_slot) + " offset " +
                 std::to_string(descriptor.descriptor_byte_offset), CommandStatus::MemoryFault);
        }
        std::array<std::byte, 12> words{};
        read_guest(memory, binding.address + descriptor.descriptor_byte_offset, words);
        const std::uint64_t address =
            load_word(std::span<const std::byte, 4>(words.data(), 4)) |
            (std::uint64_t{load_word(std::span<const std::byte, 4>(words.data() + 4, 4))} << 32);
        const std::uint32_t size = load_word(std::span<const std::byte, 4>(words.data() + 8, 4));
        if (!size) {
            // A null binding has no guest access; MSL still requires a valid
            // pointer, so bind a one-word zero buffer.
            const std::array<std::byte, 4> zero{};
            id<MTLBuffer> buffer = [device newBufferWithBytes:zero.data() length:zero.size()
                                                      options:MTLResourceStorageModeShared];
            retain_resource(buffer);
            [table setAddress:buffer.gpuAddress atIndex:descriptor.metal_buffer_index];
            continue;
        }
        if (address > std::numeric_limits<std::uint64_t>::max() - size) {
            fail("Compute storage descriptor range overflows", CommandStatus::MemoryFault);
        }
        std::vector<std::byte> bytes(size);
        read_guest(memory, address, bytes);
        staged.charge(bytes.size());
        id<MTLBuffer> buffer = [device newBufferWithBytes:bytes.data() length:bytes.size()
                                                  options:MTLResourceStorageModeShared];
        retain_resource(buffer);
        [table setAddress:buffer.gpuAddress atIndex:descriptor.metal_buffer_index];
        if (descriptor.access != CompiledShaderResourceAccess::Read) {
            writeback.push_back({address, size, buffer});
        }
    }

    // Textures and samplers through the decode paths shared with the draw
    // backend: TIC/TSC entries from the compute pools, uploaded the same way.
    // Each binding's constant-buffer slot is the provenance its reflected
    // fact records; facts and bindings were pushed in one order.
    if (stage.texture_facts.size() < stage.textures.size()) {
        fail("Compute reflection is missing texture provenance facts");
    }
    for (std::size_t index = 0; index < stage.textures.size(); ++index) {
        const auto& binding = stage.textures[index];
        const auto& fact = stage.texture_facts[index];
        auto resolved = resolve_compute_texture(stop, launch, memory, fact.cbuf_slot,
                                                binding.handle_byte_offset, true);
        if (resolved.texture.type != TextureType::Texture2D || resolved.texture.samples > 1) {
            fail("Metal compute samples plain 2D textures only");
        }
        id<MTLTexture> texture;
        if (resolved.is_null) {
            auto created = metal_null_texture(device, queue, resolved.texture,
                                              metal::null_texture_scalar(&fact),
                                              metal::null_texture_depth(&fact));
            if (const auto* refusal = std::get_if<CommandResult>(&created)) {
                fail(std::string(refusal->detail), refusal->status);
            }
            texture = std::get<id<MTLTexture>>(created);
            retain_resource(texture);
        } else {
            auto sampled = read_linear_texture(resolved.texture, memory);
            if (const auto* failure = std::get_if<TextureError>(&sampled)) {
                fail(std::string(failure->message()),
                     failure->code == TextureErrorCode::MemoryRead ? CommandStatus::MemoryFault
                                                                   : CommandStatus::Unsupported);
            }
            auto data = std::get<LinearTextureData>(std::move(sampled));
            const auto mapped = value(metal_sampled_format(resolved.texture));
            auto* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:mapped.pixel_format
                width:resolved.texture.width height:resolved.texture.height mipmapped:NO];
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.usage = MTLTextureUsageShaderRead;
            descriptor.swizzle = mapped.swizzle;
            texture = [device newTextureWithDescriptor:descriptor];
            retain_resource(texture);
            staged.charge(data.bytes.size());
            [texture replaceRegion:MTLRegionMake2D(0, 0, resolved.texture.width,
                                                   resolved.texture.height)
                       mipmapLevel:0 withBytes:data.bytes.data()
                       bytesPerRow:data.row_bytes];
        }
        const auto properties = value(metal_sampler_properties(resolved.sampler,
                                                               resolved.texture.srgb));
        if (properties.custom_border) {
            fail("Metal compute has no custom-border support binding");
        }
        id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:
            metal_sampler_descriptor(properties)];
        retain_resource(sampler);
        [table setTexture:texture.gpuResourceID atIndex:binding.texture_index];
        [table setSamplerState:sampler.gpuResourceID atIndex:binding.sampler_index];
    }

    // Pipeline: cached by the compiled artifact bytes and entry point, never
    // by launch state, exactly like the legacy draw pipeline cache.
    const auto fingerprint = [](const std::vector<std::byte>& bytes) {
        std::uint64_t hash = 1469598103934665603ull;
        for (const auto byte : bytes) {
            hash ^= std::to_integer<unsigned>(byte);
            hash *= 1099511628211ull;
        }
        return hash;
    };
    const std::string pipeline_key = std::to_string(fingerprint(stage.metallib)) + ':' +
                                     stage.entry_point;
    id<MTLComputePipelineState> pipeline;
    if (const auto cached = pipeline_cache().find(pipeline_key);
        cached != pipeline_cache().end()) {
        pipeline = cached->second;
    }
    if (!pipeline) {
        dispatch_data_t data = dispatch_data_create(stage.metallib.data(), stage.metallib.size(),
                                                    nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
        if (!library) fail("Compute dispatch compiled library could not be loaded");
        retain_resource(library);
        auto* function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = [NSString stringWithUTF8String:stage.entry_point.c_str()];
        auto* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor
                                               compilerTaskOptions:nil error:&error];
        if (!pipeline) fail("Compute pipeline creation failed");
        [retained addObject:pipeline];
        pipeline_cache().emplace(pipeline_key, pipeline);
    }

    // Dispatch shape: one threadgroup per grid cell with the meta-data block
    // dimensions, checked against the device limits before encoding.
    MTLSize grid{launch.grid[0], launch.grid[1], launch.grid[2]};
    MTLSize block{launch.block[0], launch.block[1], launch.block[2]};
    if (!grid.width || !grid.height || !grid.depth) {
        fail("Compute dispatch grid dimensions must be nonzero");
    }
    const auto limits = device.maxThreadsPerThreadgroup;
    if (!block.width || !block.height || !block.depth ||
        block.width > limits.width || block.height > limits.height ||
        block.depth > limits.depth) {
        fail("Compute dispatch block dimensions exceed the device threadgroup limits");
    }
    if (pipeline.maxTotalThreadsPerThreadgroup &&
        block.width * block.height * block.depth > pipeline.maxTotalThreadsPerThreadgroup) {
        fail("Compute dispatch block size exceeds the pipeline threadgroup limit");
    }

    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    if (!encoder) fail("Compute encoder creation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:block];
    [encoder endEncoding];
    [commands endCommandBuffer];

    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    auto* commit_options = [MTL4CommitOptions new];
    [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(finished);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit_options];
    if (dispatch_semaphore_wait(finished,
            dispatch_time(DISPATCH_TIME_NOW,
                          static_cast<std::int64_t>(context.completion_timeout_ns))) != 0) {
        fail("Compute dispatch GPU completion timed out", CommandStatus::Stopped);
    }
    if (execution_error) {
        fail(std::string("Compute dispatch GPU execution failed: ") +
                 execution_error.localizedDescription.UTF8String,
             CommandStatus::Stopped);
    }

    // Publish every writable storage result to guest memory; the synchronous
    // contract returns only once results are guest-visible.
    for (const auto& pending : writeback) {
        const auto* bytes = static_cast<const std::byte*>(pending.buffer.contents);
        if (!bytes || !memory.write(pending.address, {bytes, pending.size}).ok()) {
            fail("Compute dispatch storage writeback failed", CommandStatus::MemoryFault);
        }
    }
    if (context.statistics) ++context.statistics->completed_dispatches;
    return {};
}

}  // namespace

graphics::CommandResult validate_metal_compute_state(const RenderStop& snapshot) noexcept {
    try {
        if (snapshot.kind != RenderKind::Compute || !snapshot.has_compute_descriptor ||
            !snapshot.compute) {
            fail("Compute validation requires a compute stop with a queue meta data descriptor");
        }
        const ComputeLaunch& launch = *snapshot.compute;
        if (!launch.grid[0] || !launch.grid[1] || !launch.grid[2] ||
            !launch.block[0] || !launch.block[1] || !launch.block[2]) {
            fail("Compute launch dimensions must be nonzero");
        }
        return {};
    } catch (const CommandResult& error) {
        return error;
    } catch (...) {
        return {CommandStatus::Unsupported, "Metal compute state validation failed"};
    }
}

graphics::CommandResult execute_metal_compute(void* raw, const RenderStop& snapshot,
                                              const GpuAddressSpace& memory) noexcept {
    @autoreleasepool {
        try {
            if (!raw) fail("Metal compute context is missing");
            return execute(*static_cast<MetalComputeContext*>(raw), snapshot, memory);
        } catch (const CommandResult& error) {
            return error;
        } catch (const std::bad_alloc&) {
            return {CommandStatus::BudgetExceeded, "Metal compute staging allocation failed"};
        } catch (...) {
            return {CommandStatus::Unsupported, "Metal compute provider or backend failed"};
        }
    }
}

}  // namespace armigo::platform
