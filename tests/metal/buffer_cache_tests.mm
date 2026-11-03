// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic Metal 4 tests for lane L5 buffer residency:
// readback through a cached buffer, GPU-produced bytes consumed from the same
// cached allocation via real GPU work, uniform record-time versioning, native
// binding offsets, and registry-owned ring retention during delayed completion.
// All fixtures are synthetic; no game-derived bytes appear here.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "buffer_cache.h"
#include "device_context.h"

#include <armigo/graphics/resource_memory_registry.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// Construction and native-buffer seams defined in
// src/platform/metal/buffer_cache.mm.
namespace armigo::platform::metal {
graphics::RendererResult<std::shared_ptr<graphics::BufferCache>> create_buffer_cache(
    std::shared_ptr<DeviceContext> device, std::shared_ptr<graphics::ResourceMemory> memory,
    std::uint64_t budget_bytes);
id<MTLBuffer> resident_native_buffer(const graphics::RetainedResource& storage);
}

namespace {

using namespace armigo::graphics;
namespace metal = armigo::platform::metal;

unsigned failures = 0;

#define CHECK(...)                                                             \
    do {                                                                       \
        if (!(__VA_ARGS__)) {                                                  \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__);     \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

bool succeeded(const RendererStatus& result) {
    return std::holds_alternative<std::monostate>(result);
}

template <class T>
T require_value(RendererResult<T> result) {
    CHECK(std::holds_alternative<T>(result));
    if (!std::holds_alternative<T>(result)) {
        const auto command = std::get<RendererError>(result).command();
        std::fprintf(stderr, "Cannot continue: %.*s\n",
                     static_cast<int>(command.detail.size()), command.detail.data());
        std::abort();
    }
    return std::get<T>(std::move(result));
}

struct Backend final : ResourceMemoryBackend {
    struct Allocation {
        BackingId id;
        std::vector<std::byte> bytes;
    };

    Allocation& add(BackingId id, std::size_t size, std::byte fill = std::byte{0}) {
        allocations.push_back({id, std::vector<std::byte>(size, fill)});
        return allocations.back();
    }

    Allocation* find(BackingId id) {
        const auto found = std::find_if(allocations.begin(), allocations.end(),
                                        [&](const auto& entry) { return entry.id == id; });
        return found == allocations.end() ? nullptr : &*found;
    }

    RendererStatus read_cpu(BackingRange range,
                            std::span<std::byte> destination) override {
        ++reads;
        Allocation* allocation = find(range.backing);
        if (!allocation || destination.size() != range.size ||
            range.offset > allocation->bytes.size() ||
            range.size > allocation->bytes.size() - range.offset) {
            return RendererError({CommandStatus::MemoryFault, "read bounds failure"});
        }
        std::copy_n(allocation->bytes.begin() + range.offset, destination.size(),
                    destination.begin());
        return renderer_success();
    }

    RendererStatus write_cpu(BackingRange range,
                             std::span<const std::byte> source) override {
        ++writes;
        Allocation* allocation = find(range.backing);
        if (!allocation || source.size() != range.size ||
            range.offset > allocation->bytes.size() ||
            range.size > allocation->bytes.size() - range.offset) {
            return RendererError({CommandStatus::MemoryFault, "write bounds failure"});
        }
        std::copy_n(source.begin(), source.size(), allocation->bytes.begin() + range.offset);
        return renderer_success();
    }

    RendererStatus wait_gpu(CompletionToken) override { return renderer_success(); }

    std::vector<Allocation> allocations;
    unsigned reads = 0, writes = 0;
};

struct NoopActions final : CompletionActions {
    RendererStatus make_coherent() override { return renderer_success(); }
    RendererStatus prepare_publication() override { return renderer_success(); }
    void publish() noexcept override {}
    RendererStatus preserve_failed_authority() override { return renderer_success(); }
};

// A submission whose completion the test controls; a token that is submitted
// but never given feedback stays deliberately delayed.
struct TokenSource {
    explicit TokenSource(std::uint64_t sequence)
        : completion({1, sequence}, {}, std::make_shared<NoopActions>()),
          token(completion.token()) {}

    void submit() { CHECK(succeeded(completion.submitted())); }
    void succeed() { CHECK(succeeded(completion.feedback({}))); }

    SubmissionCompletion completion;
    CompletionToken token;
};

class TestDeviceContext final : public metal::DeviceContext {
public:
    TestDeviceContext(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                      std::uint64_t identity)
        : device_(device), queue_(queue) {
        key_.device = identity;
    }
    id<MTLDevice> device() const noexcept override { return device_; }
    id<MTL4CommandQueue> queue() const noexcept override { return queue_; }
    const metal::PipelineContextKey& identity() const noexcept override { return key_; }

private:
    id<MTLDevice> device_ = nil;
    id<MTL4CommandQueue> queue_ = nil;
    metal::PipelineContextKey key_;
};

constexpr BackingId backing_a{1, 1, 1};
constexpr MappingId mapping_a{7, 1, 1};

struct Fixture {
    explicit Fixture(id<MTLDevice> device, id<MTL4CommandQueue> queue)
        : backend(std::make_shared<Backend>()) {
        backend->add(backing_a, 0x100, std::byte{0x11});
        memory = require_value(make_resource_memory(
            {7, backend, CoherenceMode::SubmissionBoundary, 0x10}));
        CHECK(succeeded(memory->register_backing({backing_a, 0x100,
                                                  std::make_shared<ResourceLease>()})));
        const Mapping mapping{0x1000, 0x100, 0, 0x8000, 0x100, 0x10};
        CHECK(succeeded(memory->register_mapping({mapping_a, backing_a, mapping})));
        context = std::make_shared<TestDeviceContext>(device, queue, 0xA11CE);
        cache = require_value(metal::create_buffer_cache(context, memory, 0x80000));
    }

    void guest_write(std::uint64_t gpu_address, std::uint64_t size, std::byte fill) {
        std::vector<std::byte> bytes(size, fill);
        auto result = memory->address_space().write(gpu_address, bytes);
        CHECK(result.ok());
    }

    RetainedRange retain(std::uint64_t gpu_address, std::uint64_t size,
                         ResourceAccess access) {
        return require_value(memory->retain(gpu_address, size, access));
    }

    std::shared_ptr<Backend> backend;
    std::shared_ptr<ResourceMemoryRegistry> memory;
    std::shared_ptr<TestDeviceContext> context;
    std::shared_ptr<BufferCache> cache;
};

std::vector<std::byte> read_image(const ResidentBuffer& resident, BackingRange range) {
    auto authority = std::dynamic_pointer_cast<const ResourceMemoryStorage>(resident.storage);
    CHECK(authority != nullptr);
    std::vector<std::byte> bytes(range.size);
    CHECK(succeeded(authority->read_gpu(range, bytes)));
    return bytes;
}

std::vector<std::byte> read_native(const RetainedResource& storage, std::uint64_t offset,
                                   std::uint64_t size) {
    std::vector<std::byte> bytes(size);
    id<MTLBuffer> native = metal::resident_native_buffer(storage);
    CHECK(native != nil);
    if (!native) return bytes;
    const auto length = static_cast<std::uint64_t>(native.length);
    CHECK(offset <= length && size <= length - offset);
    if (offset > length || size > length - offset) return bytes;
    const auto contents = static_cast<const std::byte*>(native.contents);
    std::copy_n(contents + static_cast<std::ptrdiff_t>(offset), bytes.size(), bytes.begin());
    return bytes;
}

std::vector<std::byte> read_native(const ResidentBuffer& resident) {
    return read_native(resident.storage, resident.offset, resident.size);
}

bool filled(const std::vector<std::byte>& bytes, std::byte fill) {
    return std::all_of(bytes.begin(), bytes.end(),
                       [&](std::byte value) { return value == fill; });
}

BufferRequest request(const RetainedRange& range, BufferUse use, ResourceAccess access,
                      std::uint64_t alignment) {
    return BufferRequest{range, use, access, alignment};
}

bool stopped(const RendererResult<ResidentBuffer>& result) {
    const auto* failure = std::get_if<RendererError>(&result);
    return failure && failure->command().status == CommandStatus::Stopped;
}

bool depends_on(const ResidentBuffer& resident, CompletionToken producer) {
    const auto id = producer.snapshot().id;
    return std::any_of(resident.dependencies.begin(), resident.dependencies.end(),
                       [&](const CompletionToken& dependency) {
                           return dependency.snapshot().id == id;
                       });
}

// Readback through a cached buffer, plus real GPU-produced bytes consumed from
// the same cached allocation with no guest round trip.
void test_readback_and_gpu_produced_bytes(id<MTLDevice> device,
                                          id<MTL4CommandQueue> queue) {
    Fixture fixture(device, queue);
    auto view = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(filled(read_image(view, {backing_a, 0x0, 0x40}), std::byte{0x11}));
    const auto native = metal::resident_native_buffer(view.storage);
    CHECK(native != nil);
    CHECK(native.length >= 0x40);

    // A storage-buffer write acquires the view for GPU production; the write
    // reservation commits the view's own storage authority.
    auto write_view = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Write, 1)));
    CHECK(metal::resident_native_buffer(write_view.storage) == native);

    TokenSource token(1);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({write_view.storage})));

    // A direct view can retain the pending producer as a dependency because it
    // does not copy the unfinished bytes. Snapshot and growth paths must stop:
    // waiting on a dependency after either copy cannot repair stale contents.
    {
        auto direct = require_value(fixture.cache->acquire(request(
            fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
            ResourceAccess::Read, 1)));
        CHECK(metal::resident_native_buffer(direct.storage) == native);
        CHECK(depends_on(direct, token.token));
    }
    CHECK(stopped(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10))));
    CHECK(stopped(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x50, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1))));

    // Real GPU work produces bytes inside the cached buffer: a synthetic
    // kernel writes sixteen constant bytes at offset 0x20.
    NSError* error = nil;
    const std::string kernel = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void armigo_buffer_produce(device unsigned char* destination [[buffer(0)]],
                                  constant unsigned char& value [[buffer(1)]]) {
    for (unsigned i = 0; i < 16u; ++i) { destination[0x20u + i] = value; }
}
)msl";
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:kernel.c_str()]
                     options:options error:&error];
    MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    CHECK(library != nil && compiler != nil);
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"armigo_buffer_produce";
    MTL4ComputePipelineDescriptor* pipeline_descriptor =
        [MTL4ComputePipelineDescriptor new];
    pipeline_descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler
        newComputePipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil error:&error];
    CHECK(pipeline != nil);

    const std::byte produced{0x44};
    id<MTLBuffer> value_buffer = [device newBufferWithBytes:&produced length:1
                                                    options:MTLResourceStorageModeShared];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    id<MTL4ArgumentTable> table =
        [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 4;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    CHECK(value_buffer != nil && allocator != nil && commands != nil);
    CHECK(table != nil && residency != nil);
    [table setAddress:native.gpuAddress atIndex:0];
    [table setAddress:value_buffer.gpuAddress atIndex:1];
    [residency addAllocation:native];
    [residency addAllocation:value_buffer];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    CHECK(encoder != nil);
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commands endCommandBuffer];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    auto gpu_failed = std::make_shared<std::atomic_bool>(false);
    MTL4CommitOptions* commit_options = [MTL4CommitOptions new];
    [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        gpu_failed->store(feedback.error != nil, std::memory_order_relaxed);
        dispatch_semaphore_signal(done);
    }];
    id<MTLSharedEvent> gate = [device newSharedEvent];
    CHECK(gate != nil);
    gate.signaledValue = 0;
    [queue waitForEvent:gate value:1];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit_options];
    token.submit();

    // The producer is now committed to the real queue, but the unsignaled
    // shared event proves it cannot have executed while these acquisitions run.
    CHECK(dispatch_semaphore_wait(done, DISPATCH_TIME_NOW) != 0);
    {
        auto direct = require_value(fixture.cache->acquire(request(
            fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
            ResourceAccess::Read, 1)));
        CHECK(metal::resident_native_buffer(direct.storage) == native);
        CHECK(depends_on(direct, token.token));
    }
    CHECK(stopped(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10))));
    CHECK(stopped(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x50, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1))));
    CHECK(dispatch_semaphore_wait(done, DISPATCH_TIME_NOW) != 0);

    gate.signaledValue = 1;
    CHECK(dispatch_semaphore_wait(done,
            dispatch_time(DISPATCH_TIME_NOW, 10LL * NSEC_PER_SEC)) == 0);
    CHECK(!gpu_failed->load(std::memory_order_relaxed));
    token.succeed();

    // The produced bytes are first read back through the same cached
    // allocation; guest memory never carried them.
    const auto writes_before = fixture.backend->writes;
    auto again = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(metal::resident_native_buffer(again.storage) == native);
    const auto image = read_image(again, {backing_a, 0x0, 0x40});
    CHECK(filled({image.begin() + 0x20, image.begin() + 0x30}, std::byte{0x44}));
    CHECK(filled({image.begin(), image.begin() + 0x20}, std::byte{0x11}));
    CHECK(fixture.backend->writes == writes_before);

    auto uniform = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_native(uniform), std::byte{0x44}));

    CHECK(stopped(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x50, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1))));
    view = {};
    write_view = {};
    again = {};
    reservation.reset();
    CHECK(succeeded(fixture.memory->submission_boundary()));

    auto grown = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x50, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    const auto grown_bytes = read_native(grown);
    CHECK(filled({grown_bytes.begin() + 0x20, grown_bytes.begin() + 0x30},
                 std::byte{0x44}));
    CHECK(filled({grown_bytes.begin(), grown_bytes.begin() + 0x20}, std::byte{0x11}));
    CHECK(filled({grown_bytes.begin() + 0x30, grown_bytes.end()}, std::byte{0x11}));
}

// Overlapping storage bindings prepared for one submission must use the same
// native image. Growth stops while the first binding is retained; preacquiring
// the union makes both final bindings direct contained views.
void test_overlapping_storage_bindings(id<MTLDevice> device,
                                       id<MTL4CommandQueue> queue) {
    Fixture fixture(device, queue);
    auto a = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x20, ResourceAccess::Read), BufferUse::Storage,
        ResourceAccess::Read, 1)));
    id<MTLBuffer> first_native = metal::resident_native_buffer(a.storage);
    CHECK(first_native != nil);
    CHECK(a.offset == 0);

    auto blocked = fixture.cache->acquire(request(
        fixture.retain(0x1010, 0x20, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Write, 1));
    CHECK(stopped(blocked));
    CHECK(metal::resident_native_buffer(a.storage) == first_native);
    CHECK(filled(read_native(a), std::byte{0x11}));

    a = {};
    auto complete = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x30, ResourceAccess::Read), BufferUse::Storage,
        ResourceAccess::Read, 1)));
    a = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x20, ResourceAccess::Read), BufferUse::Storage,
        ResourceAccess::Read, 1)));
    auto b = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1010, 0x20, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Write, 1)));
    id<MTLBuffer> a_native = metal::resident_native_buffer(a.storage);
    id<MTLBuffer> b_native = metal::resident_native_buffer(b.storage);
    CHECK(a_native != nil && a_native == b_native);
    CHECK(metal::resident_native_buffer(complete.storage) == a_native);
    CHECK(a.offset == complete.offset);
    CHECK(b.offset == a.offset + 0x10);

    TokenSource token(20);
    const std::array accesses{
        GpuAccessRequest{fixture.retain(0x1000, 0x20, ResourceAccess::Read),
                         ResourceAccess::Read},
        GpuAccessRequest{fixture.retain(0x1010, 0x20, ResourceAccess::ReadWrite),
                         ResourceAccess::Write},
    };
    auto reservation = require_value(fixture.memory->reserve_accesses(accesses, token.token));
    CHECK(reservation->dependencies().empty());
    CHECK(succeeded(reservation->commit({a.storage, b.storage})));

    NSError* error = nil;
    const std::string kernel = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void armigo_overlapping_storage(
    device atomic_uint* source [[buffer(0)]],
    device atomic_uint* destination [[buffer(1)]],
    device uint* observed [[buffer(2)]]) {
    atomic_store_explicit(destination, 0x78563412u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device);
    observed[0] = atomic_load_explicit(source + 4u, memory_order_relaxed);
}
)msl";
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:kernel.c_str()]
                     options:options error:&error];
    if (!library) {
        std::fprintf(stderr, "Metal overlapping-storage library compilation failed: %s\n",
                     error.localizedDescription.UTF8String);
        ++failures;
        return;
    }
    MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
    id<MTL4Compiler> compiler =
        [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    if (!compiler) {
        std::fprintf(stderr, "Metal overlapping-storage compiler creation failed: %s\n",
                     error.localizedDescription.UTF8String);
        ++failures;
        return;
    }
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"armigo_overlapping_storage";
    MTL4ComputePipelineDescriptor* pipeline_descriptor =
        [MTL4ComputePipelineDescriptor new];
    pipeline_descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler
        newComputePipelineStateWithDescriptor:pipeline_descriptor
                         compilerTaskOptions:nil error:&error];
    if (!pipeline) {
        std::fprintf(stderr, "Metal overlapping-storage pipeline creation failed: %s\n",
                     error.localizedDescription.UTF8String);
        ++failures;
        return;
    }

    id<MTLBuffer> observed = [device newBufferWithLength:sizeof(std::uint32_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 3;
    id<MTL4ArgumentTable> table =
        [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 3;
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    CHECK(observed != nil && allocator != nil && commands != nil);
    CHECK(table != nil && residency != nil);
    *static_cast<std::uint32_t*>(observed.contents) = 0;
    [table setAddress:a_native.gpuAddress + a.offset atIndex:0];
    [table setAddress:b_native.gpuAddress + b.offset atIndex:1];
    [table setAddress:observed.gpuAddress atIndex:2];
    [residency addAllocation:a_native];
    [residency addAllocation:observed];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    CHECK(encoder != nil);
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commands endCommandBuffer];

    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    auto gpu_failed = std::make_shared<std::atomic_bool>(false);
    MTL4CommitOptions* commit_options = [MTL4CommitOptions new];
    [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        gpu_failed->store(feedback.error != nil, std::memory_order_relaxed);
        dispatch_semaphore_signal(done);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit_options];
    token.submit();
    CHECK(dispatch_semaphore_wait(done,
            dispatch_time(DISPATCH_TIME_NOW, 10LL * NSEC_PER_SEC)) == 0);
    CHECK(!gpu_failed->load(std::memory_order_relaxed));
    token.succeed();
    CHECK(*static_cast<const std::uint32_t*>(observed.contents) == 0x78563412u);
}

// Refresh after a boundary is limited to the requested mapped interval even
// when an older cache image spans bytes whose mapping has been removed.
void test_partial_remapping_refresh(id<MTLDevice> device,
                                    id<MTL4CommandQueue> queue) {
    Fixture fixture(device, queue);
    auto cached = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(filled(read_native(cached), std::byte{0x11}));
    cached = {};

    CHECK(succeeded(fixture.memory->unmap(mapping_a)));
    constexpr MappingId partial_mapping{7, 2, 1};
    const Mapping partial{0x1020, 0x10, 0x20, 0x8000, 0x100, 0x10};
    CHECK(succeeded(fixture.memory->register_mapping(
        {partial_mapping, backing_a, partial})));
    fixture.guest_write(0x1020, 0x10, std::byte{0x66});
    CHECK(succeeded(fixture.memory->submission_boundary()));

    auto valid = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(valid.offset == 0x20);
    CHECK(filled(read_native(valid), std::byte{0x66}));
}

// Draw A keeps its recorded uniform version after draw B updates the same
// guest range.
void test_uniform_version_retention(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    Fixture fixture(device, queue);
    auto a = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_native(a), std::byte{0x11}));

    fixture.guest_write(0x1000, 0x40, std::byte{0x22});
    auto b = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_native(b), std::byte{0x22}));
    CHECK(filled(read_native(a), std::byte{0x11}));

    // The slices are distinct immutable ring allocations.
    CHECK(metal::resident_native_buffer(a.storage) != nil);
    CHECK(a.storage != b.storage);
}

// Reusing a released ring block exposes the new slice's nonzero offset through
// the public resident used by native binders.
void test_reused_uniform_native_offset(id<MTLDevice> device,
                                       id<MTL4CommandQueue> queue) {
    Fixture fixture(device, queue);
    auto a = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_native(a), std::byte{0x11}));
    id<MTLBuffer> block = metal::resident_native_buffer(a.storage);
    a = {};

    fixture.guest_write(0x1000, 0x40, std::byte{0x22});
    auto b = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(metal::resident_native_buffer(b.storage) == block);
    CHECK(b.offset != 0);
    CHECK(filled(read_native(b), std::byte{0x22}));
}

// Snapshots taken while earlier slices are still referenced (a deferred batch's
// encoded draws) append to the same block's tail instead of allocating a new
// block per snapshot; earlier slices keep their bytes.
void test_held_slices_share_block(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    Fixture fixture(device, queue);
    std::vector<ResidentBuffer> held;
    for (unsigned i = 0; i != 64; ++i) {
        const std::vector<std::byte> bytes(0x40, std::byte(i + 1));
        held.push_back(require_value(fixture.cache->upload(bytes, 0x10)));
    }
    id<MTLBuffer> block = metal::resident_native_buffer(held.front().storage);
    for (unsigned i = 0; i != held.size(); ++i) {
        CHECK(metal::resident_native_buffer(held[i].storage) == block);
        CHECK(held[i].offset == held.front().offset + i * 0x40);
        CHECK(filled(read_native(held[i]), std::byte(i + 1)));
    }
}

// Once a read is committed, the registry alone retains A's slice until the
// delayed token becomes GPU-quiescent. Later uniform snapshots cannot recycle
// or overwrite its ring block even after every caller-owned reference is gone.
void test_registry_retains_delayed_uniform(id<MTLDevice> device,
                                           id<MTL4CommandQueue> queue) {
    Fixture fixture(device, queue);
    auto a = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_native(a), std::byte{0x11}));
    const auto a_offset = a.offset;
    const auto a_size = a.size;
    std::weak_ptr<const ResourceLease> registry_owned = a.storage;

    TokenSource draw(2);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), ResourceAccess::Read}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, draw.token));
    CHECK(succeeded(reservation->commit({a.storage})));
    draw.submit();
    reservation.reset();
    a = {};
    CHECK(!registry_owned.expired());

    fixture.guest_write(0x1040, 0x40, std::byte{0x22});
    std::vector<ResidentBuffer> later;
    for (unsigned i = 0; i < 4; ++i) {
        later.push_back(require_value(fixture.cache->acquire(request(
            fixture.retain(0x1040, 0x40, ResourceAccess::Read), BufferUse::Uniform,
            ResourceAccess::Read, 0x10))));
        CHECK(filled(read_native(later.back()), std::byte{0x22}));
    }

    CHECK(!registry_owned.expired());
    if (auto retained = registry_owned.lock()) {
        CHECK(filled(read_native(retained, a_offset, a_size), std::byte{0x11}));
    }
    draw.succeed();
}

void test_partial_storage_word(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    // The last three mapped guest bytes must be usable by subword loads even
    // though their host uint load reaches a fourth, non-guest padding byte.
    Fixture fixture(device, queue);
    auto retained = fixture.retain(0x10fd, 3, ResourceAccess::ReadWrite);
    auto resident = require_value(fixture.cache->acquire(request(
        retained, BufferUse::Storage, ResourceAccess::ReadWrite, 4)));
    CHECK(resident.size == 3);
    CHECK(resident.key.bytes == BackingRange{backing_a, 0xfd, 3});
    CHECK(resident.offset % 4 == 0);
    const auto native = metal::resident_native_buffer(resident.storage);
    CHECK(native && native.length >= resident.offset + 4);
    const auto word = read_native(resident.storage, resident.offset, 4);
    CHECK(word == std::vector<std::byte>({std::byte{0x11}, std::byte{0x11},
                                       std::byte{0x11}, std::byte{0}}));
    const auto authority = std::dynamic_pointer_cast<const ResourceMemoryStorage>(resident.storage);
    CHECK(authority && authority->coverage() == retained.bytes);
    std::array<std::byte, 4> outside{};
    CHECK(std::holds_alternative<RendererError>(
        authority->read_gpu({backing_a, 0xfd, 4}, outside)));
}

void test_construction_validation(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    auto backend = std::make_shared<Backend>();
    auto memory = require_value(make_resource_memory(
        {7, backend, CoherenceMode::SubmissionBoundary, 0x10}));
    auto context = std::make_shared<TestDeviceContext>(device, queue, 0xA11CE);
    CHECK(std::holds_alternative<RendererError>(
        metal::create_buffer_cache({}, memory, 0x100)));
    CHECK(std::holds_alternative<RendererError>(metal::create_buffer_cache(context, {}, 0x100)));
    CHECK(std::holds_alternative<RendererError>(metal::create_buffer_cache(context, memory, 0)));
}

} // namespace

int main(int argc, char** argv) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::fputs("No Metal device available\n", stderr);
        return 1;
    }
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    if (!queue) {
        std::fputs("No Metal 4 command queue available\n", stderr);
        return 1;
    }
    struct TestCase {
        std::string_view name;
        void (*run)(id<MTLDevice>, id<MTL4CommandQueue>);
    };
    const std::array tests{
        TestCase{"construction", test_construction_validation},
        TestCase{"uniform-version", test_uniform_version_retention},
        TestCase{"uniform-native-offset", test_reused_uniform_native_offset},
        TestCase{"held-slices-share-block", test_held_slices_share_block},
        TestCase{"registry-retention", test_registry_retains_delayed_uniform},
        TestCase{"gpu-producer", test_readback_and_gpu_produced_bytes},
        TestCase{"overlapping-storage", test_overlapping_storage_bindings},
        TestCase{"partial-remapping", test_partial_remapping_refresh},
        TestCase{"partial-storage-word", test_partial_storage_word},
    };
    unsigned selected = 0;
    for (const auto& test : tests) {
        const bool run = argc == 1 || std::any_of(argv + 1, argv + argc,
            [&](const char* filter) { return test.name == filter; });
        if (!run) continue;
        ++selected;
        test.run(device, queue);
    }
    if (!selected) {
        std::fputs("No matching metal.buffer_cache test filter\n", stderr);
        return 2;
    }
    if (failures) {
        std::fprintf(stderr, "%u check(s) failed\n", failures);
        return 1;
    }
    std::printf("metal.buffer_cache: all checks passed\n");
    return 0;
}
