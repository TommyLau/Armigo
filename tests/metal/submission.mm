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

#include "clear_encoder.h"
#include "device_context.h"
#include "native_resources.h"
#include "submission.h"
#include "texture_residency.h"

#include <armigo/graphics/resource_memory_registry.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace armigo::graphics;
namespace metal = armigo::platform::metal;

unsigned failures = 0;

#define CHECK(...)                                                           \
    do {                                                                     \
        if (!(__VA_ARGS__)) {                                                \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

bool succeeded(const RendererStatus& status) {
    return std::holds_alternative<std::monostate>(status);
}

template <class T>
T require_value(RendererResult<T> result) {
    if (const auto* error = std::get_if<RendererError>(&result)) {
        const auto command = error->command();
        std::fprintf(stderr, "Cannot continue: %.*s\n",
                     int(command.detail.size()), command.detail.data());
        std::abort();
    }
    return std::get<T>(std::move(result));
}

struct Backend final : ResourceMemoryBackend {
    RendererStatus read_cpu(BackingRange, std::span<std::byte>) override {
        return RendererError({CommandStatus::MemoryFault,
                              "submission test does not read guest storage"});
    }
    RendererStatus write_cpu(BackingRange, std::span<const std::byte>) override {
        return RendererError({CommandStatus::MemoryFault,
                              "submission test does not write guest storage"});
    }
    RendererStatus wait_gpu(CompletionToken token) override {
        const auto phase = token.snapshot().phase;
        if (phase == SubmissionPhase::GpuComplete ||
            phase == SubmissionPhase::Publishing ||
            phase == SubmissionPhase::Complete)
            return renderer_success();
        return RendererError({CommandStatus::Stopped,
                              "submission test GPU token is not quiescent"});
    }
};

struct ReservationDestructionProbe {
    explicit ReservationDestructionProbe(std::thread::id expected_thread)
        : expected(expected_thread) {}
    const std::thread::id expected;
    std::atomic<unsigned> destructions = 0;
    std::atomic<bool> wrong_thread = false;
};

class ProbedReservation final : public GpuAccessReservation {
public:
    ProbedReservation(std::shared_ptr<GpuAccessReservation> wrapped,
                      std::shared_ptr<ReservationDestructionProbe> probe)
        : wrapped_(std::move(wrapped)), probe_(std::move(probe)) {}
    ~ProbedReservation() override {
        wrapped_.reset();
        if (std::this_thread::get_id() != probe_->expected)
            probe_->wrong_thread.store(true, std::memory_order_relaxed);
        probe_->destructions.fetch_add(1, std::memory_order_relaxed);
    }
    std::vector<CompletionToken> dependencies() const override {
        return wrapped_->dependencies();
    }
    RendererStatus commit(std::vector<RetainedResource> storage) override {
        return wrapped_->commit(std::move(storage));
    }
private:
    std::shared_ptr<GpuAccessReservation> wrapped_;
    std::shared_ptr<ReservationDestructionProbe> probe_;
};

// Uses ResourceMemory's default acceptance path so its acceptance lease is the
// wrapped reservation rather than the registry's passive batch-state lease.
class FallbackMemory final : public ResourceMemory {
public:
    FallbackMemory(std::shared_ptr<ResourceMemory> wrapped,
                   std::shared_ptr<ReservationDestructionProbe> probe)
        : wrapped_(std::move(wrapped)), probe_(std::move(probe)) {}
    RendererStatus register_backing(BackingRegistration backing) override {
        return wrapped_->register_backing(std::move(backing));
    }
    RendererStatus register_mapping(MappingRegistration mapping) override {
        return wrapped_->register_mapping(std::move(mapping));
    }
    RendererStatus unmap(MappingId mapping) override { return wrapped_->unmap(mapping); }
    RendererStatus retire_backing(BackingId backing) override {
        return wrapped_->retire_backing(backing);
    }
    RendererResult<RetainedRange> retain(std::uint64_t address, std::uint64_t size,
                                         ResourceAccess access) override {
        return wrapped_->retain(address, size, access);
    }
    const GpuAddressSpace& address_space() const noexcept override {
        return wrapped_->address_space();
    }
    CoherenceMode coherence_mode() const noexcept override {
        return wrapped_->coherence_mode();
    }
    RendererResult<std::vector<DirtyRange>> inspect(BackingRange range) override {
        return wrapped_->inspect(range);
    }
    RendererStatus resolve(BackingRange range, ResolveReason reason) override {
        return wrapped_->resolve(range, reason);
    }
    RendererResult<std::unique_ptr<CpuReadTransaction>> prepare_cpu_read(
        BackingRange range, ResolveReason reason) override {
        return wrapped_->prepare_cpu_read(range, reason);
    }
    RendererResult<std::unique_ptr<CpuWriteTransaction>> prepare_cpu_write(
        BackingRange range) override {
        return wrapped_->prepare_cpu_write(range);
    }
    RendererResult<std::shared_ptr<GpuAccessReservation>> reserve_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer) override {
        auto reserved = wrapped_->reserve_accesses(requests, consumer);
        if (const auto* error = std::get_if<RendererError>(&reserved)) return *error;
        try {
            return std::shared_ptr<GpuAccessReservation>(std::make_shared<ProbedReservation>(
                std::get<std::shared_ptr<GpuAccessReservation>>(std::move(reserved)), probe_));
        } catch (const std::bad_alloc&) {
            return RendererError({CommandStatus::MemoryFault,
                                  "reservation probe allocation failed"});
        }
    }
private:
    std::shared_ptr<ResourceMemory> wrapped_;
    std::shared_ptr<ReservationDestructionProbe> probe_;
};

struct Actions final : CompletionActions {
    RendererStatus make_coherent() override {
        ++coherent;
        if (throw_coherent)
            throw std::runtime_error("synthetic coherence exception");
        return renderer_success();
    }
    RendererStatus prepare_publication() override {
        ++prepared;
        return renderer_success();
    }
    void publish() noexcept override {
        if (predecessor) predecessor_published = predecessor->published == 1;
        ++published;
    }
    RendererStatus preserve_failed_authority() override {
        ++preserved;
        if (preserve_failures) {
            --preserve_failures;
            return RendererError({CommandStatus::Stopped,
                                  "synthetic failed-authority handoff failure"});
        }
        return renderer_success();
    }

    bool throw_coherent = false;
    const Actions* predecessor = nullptr;
    bool predecessor_published = false;
    unsigned preserve_failures = 0;
    std::atomic<unsigned> coherent = 0;
    std::atomic<unsigned> prepared = 0;
    std::atomic<unsigned> published = 0;
    std::atomic<unsigned> preserved = 0;
};

// Blocks only the final native lease destruction, after coordinator retirement.
struct DestructionGate {
    void destroy() {
        std::unique_lock lock(mutex);
        worker = std::this_thread::get_id();
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
    }
    bool wait_until_entered() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
    std::thread::id worker_id() {
        std::lock_guard lock(mutex);
        return worker;
    }
    std::mutex mutex;
    std::condition_variable changed;
    std::thread::id worker;
    bool entered = false, released = false;
};

struct Pipeline final : metal::MetalRenderPipelineResource {
    explicit Pipeline(id<MTLRenderPipelineState> value,
                      std::shared_ptr<std::atomic<unsigned>> destructions = {})
        : value(value), destructions(std::move(destructions)) {}
    ~Pipeline() override {
        if (destruction_gate) destruction_gate->destroy();
        if (destructions) ++*destructions;
    }
    id<MTLRenderPipelineState> pipeline() const noexcept override { return value; }
    id<MTLRenderPipelineState> value = nil;
    std::shared_ptr<std::atomic<unsigned>> destructions;
    std::shared_ptr<DestructionGate> destruction_gate;
};

struct Sampler final : metal::MetalSamplerResource {
    explicit Sampler(id<MTLSamplerState> value) : value(value) {}
    id<MTLSamplerState> sampler() const noexcept override { return value; }
    id<MTLSamplerState> value = nil;
};

struct TextureStorage final : metal::TextureResource {
    TextureStorage(id<MTLTexture> value, BackingRange bytes,
                   std::shared_ptr<std::atomic<unsigned>> destructions = {})
        : value(value), bytes(bytes), destructions(std::move(destructions)) {}
    ~TextureStorage() override {
        if (destructions) ++*destructions;
    }
    BackingRange coverage() const noexcept override { return bytes; }
    id<MTLTexture> texture() const noexcept override { return value; }
    id<MTLTexture> allocation() const noexcept override {
        return backing_texture ? backing_texture : value;
    }
    RendererStatus read_gpu(BackingRange, std::span<std::byte>) const override {
        return RendererError({CommandStatus::Unsupported,
                              "submission test texture readback is explicit"});
    }
    id<MTLTexture> value = nil;
    id<MTLTexture> backing_texture = nil;
    BackingRange bytes;
    std::shared_ptr<std::atomic<unsigned>> destructions;
};

struct DelayedFeedback final : metal::SubmissionTestControl {
    CommandResult after_gpu_quiescence(CommandResult actual) override {
        std::unique_lock lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
        returned = true;
        changed.notify_all();
        if (inject_error)
            return {CommandStatus::MemoryFault,
                    "synthetic delayed Metal feedback failure"};
        return actual;
    }

    void wait_until_entered() {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return entered; });
    }
    void release(bool fail) {
        std::lock_guard lock(mutex);
        inject_error = fail;
        released = true;
        changed.notify_all();
    }
    void wait_until_returned() {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return returned; });
    }

    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, returned = false, inject_error = false;
};

struct EncodeGate final : metal::SubmissionTestControl {
    RendererStatus before_encode(SubmissionId id) override {
        std::unique_lock lock(mutex);
        if (entered.empty()) worker = std::this_thread::get_id();
        entered.push_back(id);
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
        if (inject_error)
            return RendererError({CommandStatus::MemoryFault,
                                  "synthetic asynchronous encode failure"});
        return renderer_success();
    }

    RendererStatus before_commit(SubmissionId id) override {
        std::lock_guard lock(mutex);
        committed.push_back(id);
        if (fail_commit)
            return RendererError({CommandStatus::MemoryFault,
                                  "synthetic asynchronous commit failure"});
        return renderer_success();
    }

    std::vector<SubmissionId> commits() const {
        std::lock_guard lock(mutex);
        return committed;
    }

    CommandResult after_gpu_quiescence(CommandResult actual) override {
        return actual;
    }

    bool wait_for_entries(std::size_t count,
                          std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, timeout,
                                [&] { return entered.size() >= count; });
    }

    void release(bool fail = false) {
        std::lock_guard lock(mutex);
        inject_error = fail;
        released = true;
        changed.notify_all();
    }

    std::vector<SubmissionId> entries() const {
        std::lock_guard lock(mutex);
        return entered;
    }

    std::thread::id worker_id() const {
        std::lock_guard lock(mutex);
        return worker;
    }

    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<SubmissionId> entered, committed;
    bool fail_commit = false;
    std::thread::id worker;
    bool released = false;
    bool inject_error = false;
};

struct AsyncStatus {
    void begin() {
        std::lock_guard lock(mutex);
        started = true;
        changed.notify_all();
    }

    void finish(RendererStatus value) {
        std::lock_guard lock(mutex);
        status = std::move(value);
        done = true;
        changed.notify_all();
    }

    bool wait_started(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, timeout, [&] { return started; });
    }

    bool wait_done(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, timeout, [&] { return done; });
    }

    std::mutex mutex;
    std::condition_variable changed;
    RendererStatus status = renderer_success();
    bool started = false;
    bool done = false;
};

struct TextureAsset {
    std::shared_ptr<TextureStorage> storage;
    RetainedRange retained;
    ResidentTexture resident;
    AttachmentPlan attachment;
};

struct Fixture {
    explicit Fixture(id<MTLDevice> device) {
        context = require_value(metal::create_device_context(device));
        backend = std::make_shared<Backend>();
        memory = require_value(make_resource_memory(
            {0x5100, backend, CoherenceMode::SubmissionBoundary, 0x1000}));
    }

    TextureAsset texture(std::shared_ptr<std::atomic<unsigned>> destructions = {}) {
        const std::uint64_t index = next++;
        const BackingId backing{0x5100, index, 1};
        const MappingId mapping{0x5100, index, 1};
        const std::uint64_t address = index * 0x10000;
        CHECK(succeeded(memory->register_backing({backing, 0x1000, backend})));
        CHECK(succeeded(memory->register_mapping({mapping, backing,
            {address, 0x1000, 0, address + 0x40000000, 0x1000, 0x1000}})));
        auto retained = require_value(memory->retain(address, 0x1000,
                                                     ResourceAccess::ReadWrite));
        auto* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                         width:4 height:4 mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        id<MTLTexture> native = [context->device() newTextureWithDescriptor:descriptor];
        CHECK(native != nil);
        auto storage = std::make_shared<TextureStorage>(native, retained.bytes,
                                                        std::move(destructions));
        TextureDescriptor texture;
        texture.address = address;
        texture.backing_extent = 0x1000;
        texture.row_bytes = 16;
        texture.storage_height = 4;
        texture.layer_stride = 0x1000;
        texture.width = 4;
        texture.height = 4;
        texture.depth = 1;
        texture.levels = 1;
        texture.layers = 1;
        texture.samples = 1;
        texture.format = TextureFormat::R8G8B8A8Unorm;
        texture.layout = TextureLayout::Pitch;
        TextureViewKey key;
        key.storage = {retained.bytes, texture};
        key.aspect = TextureAspect::Color;
        ResidentTexture resident{key, {}, storage, {}};
        AttachmentPlan attachment;
        attachment.texture = texture;
        attachment.aspect = TextureAspect::Color;
        attachment.load = AttachmentLoad::Load;
        attachment.store = AttachmentStore::Store;
        return {storage, retained, resident, attachment};
    }

    std::shared_ptr<metal::DeviceContext> context;
    std::shared_ptr<Backend> backend;
    std::shared_ptr<ResourceMemoryRegistry> memory;
    std::uint64_t next = 1;
};

struct Pipelines {
    std::shared_ptr<Pipeline> producer;
    std::shared_ptr<Pipeline> consumer;
    std::shared_ptr<Sampler> sampler;
    std::shared_ptr<Pipeline> consumer_high;
    std::shared_ptr<Pipeline> absent_texture;
    std::shared_ptr<Pipeline> argument_consumer;
};

Pipelines make_pipelines(id<MTLDevice> device) {
    static constexpr const char* source = R"(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; float2 uv; };
vertex VertexOut vertex_main(uint id [[vertex_id]]) {
    constexpr float2 positions[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
    constexpr float2 coordinates[3] = {float2(0,0), float2(2,0), float2(0,2)};
    return {float4(positions[id], 0, 1), coordinates[id]};
}
fragment float4 producer_main(VertexOut in [[stage_in]]) {
    return float4(0.25, 0.5, 0.75, 1.0);
}
fragment float4 consumer_main(VertexOut in [[stage_in]],
                              texture2d<float> source [[texture(0)]],
                              sampler source_sampler [[sampler(0)]]) {
    return source.sample(source_sampler, in.uv);
}
fragment float4 consumer_high(VertexOut in [[stage_in]],
                              texture2d<float> source [[texture(127)]],
                              sampler source_sampler [[sampler(15)]]) {
    return source.sample(source_sampler, in.uv);
}
fragment float4 absent_texture(texture2d<float> source [[texture(127)]]) {
    return is_null_texture(source) ? float4(0, 1, 0, 1) : float4(1, 0, 1, 1);
}
struct SampleArguments { texture2d<float> image; sampler state; };
fragment float4 argument_consumer(const device SampleArguments& arguments [[buffer(2)]]) {
    return arguments.image.sample(arguments.state, float2(-0.5));
}
)";
    auto* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:source]
                     options:options error:&error];
    if (!library) {
        std::fprintf(stderr, "Metal test library failed: %s\n",
                     error.localizedDescription.UTF8String);
        std::abort();
    }
    const auto pipeline = [&](NSString* fragment_name) {
        auto* descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [library newFunctionWithName:@"vertex_main"];
        descriptor.fragmentFunction = [library newFunctionWithName:fragment_name];
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        id<MTLRenderPipelineState> state =
            [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (!state) {
            std::fprintf(stderr, "Metal test pipeline failed: %s\n",
                         error.localizedDescription.UTF8String);
            std::abort();
        }
        return std::make_shared<Pipeline>(state);
    };
    auto* sampler_descriptor = [MTLSamplerDescriptor new];
    sampler_descriptor.supportArgumentBuffers = YES;
    sampler_descriptor.minFilter = MTLSamplerMinMagFilterNearest;
    sampler_descriptor.magFilter = MTLSamplerMinMagFilterNearest;
    sampler_descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler_descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    auto sampler = std::make_shared<Sampler>(
        [device newSamplerStateWithDescriptor:sampler_descriptor]);
    CHECK(sampler->value != nil);
    return {pipeline(@"producer_main"), pipeline(@"consumer_main"), sampler,
            pipeline(@"consumer_high"), pipeline(@"absent_texture"),
            pipeline(@"argument_consumer")};
}

metal::PreparedRender draw_packet(SubmissionId id, const TextureAsset& target,
                           std::shared_ptr<Pipeline> pipeline,
                           std::shared_ptr<Actions> actions,
                           std::shared_ptr<ResourceMemory> memory,
                           const TextureAsset* sampled = nullptr,
                           std::shared_ptr<Sampler> sampler = {}) {
    metal::PreparedRender prepared;
    auto snapshot = std::make_shared<RenderStop>();
    snapshot->kind = RenderKind::Draw;
    prepared.request = {id, snapshot, std::move(memory), {}, std::move(actions)};
    prepared.targets.colors[0] = target.attachment;
    prepared.targets.area = {0, 0, 4, 4};
    prepared.targets.samples = 1;
    prepared.attachments.colors[0] = target.resident;
    prepared.attachments.key.colors[0] = target.resident.key;
    prepared.attachments.key.area = prepared.targets.area;
    prepared.attachments.key.samples = 1;
    metal::PreparedDraw draw;
    draw.draw.topology = PrimitiveTopology::Triangles;
    draw.draw.primitive_class = PrimitiveClass::Triangle;
    draw.draw.segments.push_back({0, 3, 0, 0, 1});
    draw.state.viewports.push_back({0, 0, 4, 4, 0, 1});
    draw.state.scissors.push_back({0, 0, 4, 4});
    draw.pipeline.pipeline = std::move(pipeline);
    if (sampled) {
        draw.bindings.textures.push_back({armigo::shader_input::Stage::Fragment, 0, 0,
                                          sampled->resident, sampler});
    }
    prepared.operation = std::move(draw);
    if (sampled) {
        prepared.accesses.push_back({sampled->retained, ResourceAccess::Read});
        prepared.access_storage.push_back(sampled->storage);
    }
    prepared.accesses.push_back({target.retained, ResourceAccess::Write});
    prepared.access_storage.push_back(target.storage);
    return prepared;
}

std::uint64_t deadline_ns();
metal::PreparedRender empty_clear_packet(SubmissionId id,
                                        std::shared_ptr<ResourceMemory> memory,
                                        std::shared_ptr<Actions> actions);

void test_asynchronous_encode_fifo_and_flush(Fixture& fixture,
                                             const Pipelines& pipelines) {
    auto first_target = fixture.texture();
    auto second_target = fixture.texture();
    auto third_target = fixture.texture();
    auto first_actions = std::make_shared<Actions>();
    auto second_actions = std::make_shared<Actions>();
    auto third_actions = std::make_shared<Actions>();
    auto control = std::make_shared<EncodeGate>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 8, 2, control));
    const auto caller = std::this_thread::get_id();

    auto first = require_value(queue->append(draw_packet(
        {24, 1}, first_target, pipelines.producer, first_actions, fixture.memory)));
    auto second = require_value(queue->append(draw_packet(
        {24, 2}, second_target, pipelines.producer, second_actions, fixture.memory)));

    // All appends have returned while the first packet remains blocked before
    // native preparation on the serial encode worker.
    CHECK(control->wait_for_entries(1, std::chrono::seconds(5)));
    CHECK(control->worker_id() != caller);
    CHECK(control->entries().size() == 1);
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(first_actions->published == 0 && second_actions->published == 0 &&
          third_actions->published == 0);

    auto flush_state = std::make_shared<AsyncStatus>();
    std::thread flush_thread([queue, flush_state] {
        flush_state->begin();
        flush_state->finish(queue->flush(SubmissionBoundary::SubmissionEnd));
    });
    CHECK(flush_state->wait_started(std::chrono::seconds(5)));
    const bool asynchronous = flush_state->wait_done(std::chrono::seconds(5));
    CHECK(asynchronous);
    if (!asynchronous) control->release();
    flush_thread.join();
    CHECK(succeeded(flush_state->status));
    CHECK(control->entries().size() == 1);
    CHECK(control->commits().empty());
    // Telemetry can sample an active encoder without becoming a hidden join.
    CHECK(submission_statistics(*queue, false).command_buffers == 0);
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Submitted);

    auto poll_state = std::make_shared<AsyncStatus>();
    std::thread poll_thread([queue, poll_state] {
        poll_state->begin();
        poll_state->finish(queue->poll());
    });
    CHECK(poll_state->wait_started(std::chrono::seconds(5)));
    const bool polled = poll_state->wait_done(std::chrono::seconds(5));
    CHECK(polled);
    if (!polled) control->release();
    poll_thread.join();
    CHECK(succeeded(poll_state->status));
    CHECK(control->commits().empty());

    // The next batch may be accepted while the first still owns its encoder.
    // Its GPU sampling also verifies that the two commits retain queue order.
    CHECK(!queue->append_repeat({}));
    third_actions->predecessor = second_actions.get();
    auto third = require_value(queue->append(draw_packet(
        {24, 3}, third_target, pipelines.consumer, third_actions, fixture.memory,
        &second_target, pipelines.sampler)));
    metal::RepeatDraw repeat;
    repeat.segments.push_back({0, 3, 0, 0, 1});
    CHECK(queue->append_repeat(std::move(repeat)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(std::holds_alternative<RendererError>(queue->append(draw_packet(
        {24, 4}, third_target, pipelines.producer, std::make_shared<Actions>(), fixture.memory))));
    CHECK(!queue->append_repeat({}));

    // Pending commits count toward retirement pressure and obey deadlines
    // even before any GPU callback exists.
    CHECK(std::holds_alternative<RendererError>(queue->retire_below(2, 0)));
    CHECK(std::holds_alternative<RendererError>(queue->drain(0)));
    auto drain_state = std::make_shared<AsyncStatus>();
    std::thread drain_thread([queue, drain_state] {
        drain_state->begin();
        drain_state->finish(queue->drain(deadline_ns()));
    });
    CHECK(drain_state->wait_started(std::chrono::seconds(5)));
    CHECK(!drain_state->wait_done(std::chrono::milliseconds(100)));
    control->release();
    drain_thread.join();
    CHECK(succeeded(drain_state->status));
    const auto commits = control->commits();
    CHECK(commits.size() == 2);
    if (commits.size() == 2) {
        CHECK(commits[0].queue == 24 && commits[0].sequence == 1);
        CHECK(commits[1].queue == 24 && commits[1].sequence == 3);
    }
    CHECK(third_actions->predecessor_published);
    CHECK(metal::submission_statistics(*queue).draws == 4);
    CHECK(control->wait_for_entries(4, std::chrono::seconds(5)));
    const auto order = control->entries();
    CHECK(order.size() == 4);
    if (order.size() == 4) {
        CHECK(order[0].queue == 24 && order[0].sequence == 1);
        CHECK(order[1].queue == 24 && order[1].sequence == 2);
        CHECK(order[2].queue == 24 && order[2].sequence == 3);
        CHECK(order[3].queue == 24 && order[3].sequence == 3);
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline.time_since_epoch()).count())));
    CHECK(first_actions->published == 1 && second_actions->published == 1 &&
          third_actions->published == 1);
    std::array<std::uint8_t, 64> pixels{};
    for (const auto* target : {&first_target, &second_target, &third_target}) {
        [target->storage->value getBytes:pixels.data() bytesPerRow:16
            fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
        for (std::size_t pixel = 0; pixel < 16; ++pixel) {
            CHECK(pixels[pixel * 4] >= 63 && pixels[pixel * 4] <= 64);
            CHECK(pixels[pixel * 4 + 1] >= 127 && pixels[pixel * 4 + 1] <= 128);
            CHECK(pixels[pixel * 4 + 2] >= 191 && pixels[pixel * 4 + 2] <= 192);
            CHECK(pixels[pixel * 4 + 3] == 255);
        }
    }
}

void test_commit_hold_encodes_then_releases_in_order(Fixture& fixture,
                                                     const Pipelines& pipelines) {
    auto control = std::make_shared<EncodeGate>();
    control->release();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 2, control));
    auto first_target = fixture.texture();
    auto second_target = fixture.texture();
    CHECK(succeeded(queue->set_commit_hold(true)));
    CHECK(queue->commit_held());
    auto first = require_value(queue->append(draw_packet(
        {31, 1}, first_target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    auto second = require_value(queue->append(draw_packet(
        {31, 2}, second_target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(control->wait_for_entries(2, std::chrono::seconds(5)));
    CHECK(control->commits().empty());
    CHECK(!queue->can_accept_while_held());
    CHECK(metal::commit_hold_blocked(metal::commit_hold_wait()));
    CHECK(metal::commit_hold_blocked(queue->retire_below(1, 0)));
    CHECK(metal::commit_hold_blocked(queue->drain(0)));
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(metal::submission_statistics(*queue).accepted == 2);
    CHECK(succeeded(queue->set_commit_hold(false)));
    CHECK(succeeded(queue->drain(deadline_ns())));
    const auto commits = control->commits();
    CHECK(commits.size() == 2);
    if (commits.size() == 2) {
        CHECK(commits[0] == (SubmissionId{31, 1}));
        CHECK(commits[1] == (SubmissionId{31, 2}));
    }
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Complete);
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Complete);
}

void test_open_held_batch_wakes_encoder(Fixture& fixture,
                                        const Pipelines& pipelines) {
    auto control = std::make_shared<EncodeGate>();
    control->release();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 8, 2, control));
    auto target = fixture.texture();
    CHECK(succeeded(queue->set_commit_hold(true)));
    auto accepted = require_value(queue->append(draw_packet(
        {40, 1}, target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory)));
    CHECK(control->wait_for_entries(1, std::chrono::seconds(5)));
    CHECK(control->commits().empty());
    CHECK(succeeded(queue->set_commit_hold(false)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Complete);
}

void test_hold_preserves_earlier_commit_marker(Fixture& fixture,
                                               const Pipelines& pipelines) {
    auto control = std::make_shared<EncodeGate>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 2, control));
    auto first_target = fixture.texture();
    auto second_target = fixture.texture();
    auto first_packet = draw_packet(
        {35, 1}, first_target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory);
    first_packet.prefix_safe_completion = true;
    auto first = require_value(queue->append(std::move(first_packet)));
    CHECK(control->wait_for_entries(1, std::chrono::seconds(5)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(succeeded(queue->set_commit_hold(true)));
    auto second = require_value(queue->append(draw_packet(
        {35, 2}, second_target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->release();
    CHECK(control->wait_for_entries(2, std::chrono::seconds(5)));
    for (unsigned attempt = 0; attempt < 5000; ++attempt) {
        CHECK(succeeded(queue->poll()));
        if (first.completion.snapshot().phase == SubmissionPhase::Complete) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Complete);
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Submitted);
    const auto commits_while_held = control->commits();
    CHECK(commits_while_held.size() == 1);
    if (!commits_while_held.empty()) CHECK(commits_while_held.front() == (SubmissionId{35, 1}));
    CHECK(succeeded(queue->set_commit_hold(false)));
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(control->commits().size() == 2);
}

void test_released_generation_survives_new_hold(Fixture& fixture,
                                                 const Pipelines& pipelines) {
    auto control = std::make_shared<EncodeGate>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 2, control));
    auto first_target = fixture.texture();
    auto second_target = fixture.texture();
    CHECK(succeeded(queue->set_commit_hold(true)));
    auto first = require_value(queue->append(draw_packet(
        {36, 1}, first_target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory)));
    CHECK(control->wait_for_entries(1, std::chrono::seconds(5)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(succeeded(queue->set_commit_hold(false)));
    CHECK(succeeded(queue->set_commit_hold(true)));
    auto second = require_value(queue->append(draw_packet(
        {36, 2}, second_target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->release();
    for (unsigned attempt = 0; attempt < 5000; ++attempt) {
        CHECK(succeeded(queue->poll()));
        if (first.completion.snapshot().phase == SubmissionPhase::Complete) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Complete);
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(control->commits().size() == 1);
    CHECK(succeeded(queue->set_commit_hold(false)));
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(control->commits().size() == 2);
}

void test_abort_held_commit_prevents_native_submission(Fixture& fixture,
                                                        const Pipelines& pipelines) {
    Fixture isolated(fixture.context->device());
    auto control = std::make_shared<EncodeGate>();
    control->release();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        isolated.context, isolated.memory, 4, 1, control));
    auto target = isolated.texture();
    CHECK(succeeded(queue->set_commit_hold(true)));
    auto accepted = require_value(queue->append(draw_packet(
        {37, 1}, target, pipelines.producer, std::make_shared<Actions>(),
        isolated.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(control->wait_for_entries(1, std::chrono::seconds(5)));
    CHECK(succeeded(queue->abort_commit_hold(
        {CommandStatus::Stopped, "synthetic held prefix failure"})));
    CHECK(!queue->commit_held());
    CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns())));
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(control->commits().empty());
    CHECK(accepted.completion.snapshot().error.has_value());
}

void test_terminal_dependency_aborts_held_commit(Fixture& fixture,
                                                  const Pipelines& pipelines) {
    Fixture isolated(fixture.context->device());
    auto control = std::make_shared<EncodeGate>();
    control->release();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        isolated.context, isolated.memory, 4, 2, control));
    auto first_target = isolated.texture();
    auto second_target = isolated.texture();
    CHECK(succeeded(queue->set_commit_hold(true)));
    auto first = require_value(queue->append(draw_packet(
        {38, 1}, first_target, pipelines.producer, std::make_shared<Actions>(),
        isolated.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    SubmissionCompletion dependency({39, 1}, {}, std::make_shared<Actions>());
    auto second_packet = draw_packet(
        {38, 2}, second_target, pipelines.producer, std::make_shared<Actions>(),
        isolated.memory);
    second_packet.request.dependencies.push_back(dependency.token());
    auto second = require_value(queue->append(std::move(second_packet)));
    CHECK(succeeded(dependency.reject(
        {CommandStatus::Stopped, "synthetic dependency failure during commit hold"})));
    CHECK(std::holds_alternative<RendererError>(
        queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(!queue->commit_held());
    CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns())));
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(control->commits().empty());
    CHECK(first.completion.snapshot().error.has_value());
    CHECK(second.completion.snapshot().error.has_value());
}

void test_held_queue_destruction(Fixture& fixture) {
    Fixture isolated(fixture.context->device());
    auto queue = require_value(metal::create_submission_queue(
        isolated.context, isolated.memory, 2, 1));
    CHECK(succeeded(queue->set_commit_hold(true)));
    auto accepted = require_value(queue->append(empty_clear_packet(
        {34, 1}, isolated.memory, std::make_shared<Actions>())));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Submitted);
}

void test_open_batch_prefix_retirement(Fixture& fixture) {
    for (bool safe : {false, true}) {
        auto queue = require_value(metal::create_submission_queue(
            fixture.context, fixture.memory, 4, 2));
        auto first_actions = std::make_shared<Actions>();
        auto first_packet = empty_clear_packet(
            {safe ? 33u : 32u, 1}, fixture.memory, first_actions);
        first_packet.prefix_safe_completion = safe;
        auto first = require_value(queue->append(std::move(first_packet)));
        CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
        auto second = require_value(queue->append(empty_clear_packet(
            {safe ? 33u : 32u, 2}, fixture.memory, std::make_shared<Actions>())));
        CHECK(metal::submission_statistics(*queue).accepted == 2);
        CHECK(succeeded(queue->poll()));
        CHECK((first.completion.snapshot().phase == SubmissionPhase::Complete) == safe);
        CHECK(second.completion.snapshot().phase == SubmissionPhase::Accepted);
        CHECK(succeeded(queue->drain(deadline_ns())));
    }
}

void test_retired_destruction_worker(Fixture& fixture, const Pipelines& pipelines) {
    for (const bool fail_publication : {false, true}) {
        auto control = std::make_shared<EncodeGate>();
        control->release();
        auto gate = std::make_shared<DestructionGate>();
        auto destructions = std::make_shared<std::atomic<unsigned>>(0);
        auto pipeline = std::make_shared<Pipeline>(pipelines.producer->value, destructions);
        pipeline->destruction_gate = gate;
        auto actions = std::make_shared<Actions>();
        actions->throw_coherent = fail_publication;
        auto queue = require_value(metal::create_submission_queue_for_testing(
            fixture.context, fixture.memory, 4, 2, control));
        auto receipt = require_value(queue->append(draw_packet(
            {fail_publication ? 42u : 41u, 1}, fixture.texture(), pipeline,
            actions, fixture.memory)));
        pipeline.reset();
        CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
        static_cast<void>(metal::submission_statistics(*queue));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (receipt.completion.snapshot().phase == SubmissionPhase::Submitted &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto retired = queue->poll();
        CHECK(succeeded(retired) != fail_publication);
        if (fail_publication) CHECK(succeeded(queue->poll()));
        CHECK(gate->wait_until_entered());
        CHECK(gate->worker_id() == control->worker_id());
        CHECK(gate->worker_id() != std::this_thread::get_id());
        CHECK(*destructions == 0);
        CHECK(actions->published == (fail_publication ? 0u : 1u));
        CHECK(actions->preserved == (fail_publication ? 1u : 0u));
        CHECK(receipt.completion.snapshot().phase ==
              (fail_publication ? SubmissionPhase::Failed : SubmissionPhase::Complete));
        // Poll never joins destruction, even when the worker is busy. A drain
        // includes cleanup but must still honor its deadline.
        CHECK(succeeded(queue->poll()));
        const auto drained = queue->drain(0);
        CHECK(std::holds_alternative<RendererError>(drained));
        if (const auto* error = std::get_if<RendererError>(&drained))
            CHECK(error->command().detail == "submission cleanup deadline expired");
        if (!fail_publication) {
            gate->release();
            CHECK(succeeded(queue->drain(deadline_ns())));
        } else {
            // Shutdown must join ownership release even after publication fails.
            auto joined = std::make_shared<AsyncStatus>();
            std::thread shutdown([queue = std::move(queue), joined]() mutable {
                joined->begin();
                queue.reset();
                joined->finish(renderer_success());
            });
            CHECK(joined->wait_started(std::chrono::seconds(5)));
            CHECK(!joined->wait_done(std::chrono::milliseconds(20)));
            gate->release();
            shutdown.join();
            CHECK(joined->wait_done(std::chrono::seconds(5)));
        }
        CHECK(*destructions == 1);
    }
}

void test_fallback_reservation_retires_on_coordinator(
    Fixture& fixture, const Pipelines& pipelines) {
    for (const bool fail_publication : {false, true}) {
        const auto caller = std::this_thread::get_id();
        auto probe = std::make_shared<ReservationDestructionProbe>(caller);
        auto memory = std::make_shared<FallbackMemory>(fixture.memory, probe);
        auto gate = std::make_shared<DestructionGate>();
        auto pipeline = std::make_shared<Pipeline>(pipelines.producer->value);
        pipeline->destruction_gate = gate;
        auto actions = std::make_shared<Actions>();
        actions->throw_coherent = fail_publication;
        auto queue = require_value(metal::create_submission_queue(
            fixture.context, memory, 4, 1));
        auto receipt = require_value(queue->append(draw_packet(
            {fail_publication ? 44u : 43u, 1}, fixture.texture(), pipeline,
            actions, memory)));
        pipeline.reset();
        CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
        static_cast<void>(metal::submission_statistics(*queue));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (receipt.completion.snapshot().phase == SubmissionPhase::Submitted &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        CHECK(probe->destructions.load(std::memory_order_relaxed) == 0);
        const auto retired = queue->poll();
        CHECK(succeeded(retired) != fail_publication);
        CHECK(probe->destructions.load(std::memory_order_relaxed) == 1);
        CHECK(!probe->wrong_thread.load(std::memory_order_relaxed));
        CHECK(actions->preserved == (fail_publication ? 1u : 0u));
        if (fail_publication) CHECK(succeeded(queue->poll()));

        CHECK(gate->wait_until_entered());
        CHECK(probe->destructions.load(std::memory_order_relaxed) == 1);
        CHECK(!probe->wrong_thread.load(std::memory_order_relaxed));
        gate->release();
        CHECK(succeeded(queue->drain(deadline_ns())));
    }
}

void test_asynchronous_encode_failure_is_sticky(Fixture& fixture,
                                                const Pipelines& pipelines) {
    auto target = fixture.texture();
    auto rejected_target = fixture.texture();
    auto actions = std::make_shared<Actions>();
    auto rejected_actions = std::make_shared<Actions>();
    auto control = std::make_shared<EncodeGate>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 2, control));
    auto accepted = require_value(queue->append(draw_packet(
        {25, 1}, target, pipelines.producer, actions, fixture.memory)));
    auto same_batch_actions = std::make_shared<Actions>();
    auto same_batch = require_value(queue->append(draw_packet(
        {25, 2}, target, pipelines.producer, same_batch_actions, fixture.memory)));
    CHECK(control->wait_for_entries(1, std::chrono::seconds(5)));

    auto flush_state = std::make_shared<AsyncStatus>();
    std::thread flush_thread([queue, flush_state] {
        flush_state->begin();
        flush_state->finish(queue->flush(SubmissionBoundary::SubmissionEnd));
    });
    CHECK(flush_state->wait_started(std::chrono::seconds(5)));
    const bool asynchronous = flush_state->wait_done(std::chrono::seconds(5));
    CHECK(asynchronous);
    if (!asynchronous) control->release(true);
    flush_thread.join();
    CHECK(succeeded(flush_state->status));
    CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Submitted);
    auto later_actions = std::make_shared<Actions>();
    auto later = require_value(queue->append(draw_packet(
        {25, 3}, rejected_target, pipelines.producer, later_actions, fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->release(true);
    CHECK(metal::submission_statistics(*queue).command_buffers == 0);
    CHECK(control->commits().empty());

    auto appended = queue->append(draw_packet(
        {25, 4}, rejected_target, pipelines.producer, rejected_actions,
        fixture.memory));
    CHECK(std::holds_alternative<RendererError>(appended));
    const auto flushed = queue->flush(SubmissionBoundary::SubmissionEnd);
    CHECK(std::holds_alternative<RendererError>(flushed));
    const auto polled = queue->poll();
    CHECK(std::holds_alternative<RendererError>(polled));
    if (const auto* error = std::get_if<RendererError>(&appended))
        CHECK(error->command().detail == "synthetic asynchronous encode failure");
    if (const auto* error = std::get_if<RendererError>(&flushed))
        CHECK(error->command().detail == "synthetic asynchronous encode failure");
    if (const auto* error = std::get_if<RendererError>(&polled))
        CHECK(error->command().detail == "synthetic asynchronous encode failure");
    for (const auto* receipt : {&accepted, &same_batch, &later}) {
        const auto snapshot = receipt->completion.snapshot();
        CHECK(snapshot.phase == SubmissionPhase::Failed);
        CHECK(snapshot.error && snapshot.error->command().detail ==
              "synthetic asynchronous encode failure");
    }
    CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns())));
    CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns())));
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(actions->preserved == 1 && same_batch_actions->preserved == 1 &&
          later_actions->preserved == 1);
    CHECK(same_batch_actions->published == 0 && later_actions->published == 0);
    CHECK(actions->coherent == 0 && actions->prepared == 0 &&
          actions->published == 0);
    CHECK(rejected_actions->coherent == 0 && rejected_actions->prepared == 0 &&
          rejected_actions->published == 0);
}

void test_asynchronous_commit_failure(Fixture& fixture, const Pipelines& pipelines) {
    auto target = fixture.texture();
    auto control = std::make_shared<EncodeGate>();
    control->fail_commit = true;
    control->release();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 2, control));
    auto first_actions = std::make_shared<Actions>();
    auto second_actions = std::make_shared<Actions>();
    auto first = require_value(queue->append(draw_packet(
        {28, 1}, target, pipelines.producer, first_actions, fixture.memory)));
    auto second = require_value(queue->append(draw_packet(
        {28, 2}, target, pipelines.producer, second_actions, fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(metal::submission_statistics(*queue).command_buffers == 0);
    CHECK(control->commits().size() == 1);
    for (const auto* receipt : {&first, &second}) {
        const auto snapshot = receipt->completion.snapshot();
        CHECK(snapshot.phase == SubmissionPhase::Failed);
        CHECK(snapshot.error && snapshot.error->command().detail ==
              "synthetic asynchronous commit failure");
        CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns())));
    }
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(first_actions->published == 0 && second_actions->published == 0);
    CHECK(first_actions->preserved == 1 && second_actions->preserved == 1);
    CHECK(std::holds_alternative<RendererError>(queue->poll()));
}

void test_encode_failure_preserves_earlier_batch_and_drains(
    Fixture& fixture, const Pipelines& pipelines) {
    auto first_target = fixture.texture();
    auto failed_target = fixture.texture();
    auto first_actions = std::make_shared<Actions>();
    auto failed_actions = std::make_shared<Actions>();
    failed_actions->preserve_failures = 1;
    auto control = std::make_shared<EncodeGate>();
    control->release();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 2, control));
    const auto first = require_value(queue->append(draw_packet(
        {27, 1}, first_target, pipelines.producer, first_actions, fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(metal::submission_statistics(*queue).command_buffers == 1);
    control->release(true);
    const auto failed = require_value(queue->append(draw_packet(
        {27, 2}, failed_target, pipelines.producer, failed_actions, fixture.memory)));
    // Statistics joins encoding without observing or retiring the failure.
    CHECK(metal::submission_statistics(*queue).command_buffers == 1);
    auto rejected = queue->append(draw_packet(
        {27, 3}, failed_target, pipelines.producer,
        std::make_shared<Actions>(), fixture.memory));
    CHECK(std::holds_alternative<RendererError>(rejected));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const auto deadline_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline.time_since_epoch()).count();
    // The first failed-authority handoff is retryable. A subsequent drain
    // retires it despite the sticky admission error and reaches quiescence.
    CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns)));
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Complete);
    CHECK(first_actions->published == 1);
    CHECK(failed.completion.snapshot().phase == SubmissionPhase::Failed);
    CHECK(failed_actions->published == 0 && failed_actions->preserved == 1);
    CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns)));
    CHECK(succeeded(queue->drain(deadline_ns)));
    CHECK(failed_actions->preserved == 2);
    CHECK(std::holds_alternative<RendererError>(queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(std::holds_alternative<RendererError>(queue->poll()));
}

void test_submission_destruction_waits_for_encode(Fixture& fixture,
                                                  const Pipelines& pipelines) {
    auto target = fixture.texture();
    auto control = std::make_shared<EncodeGate>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 1, control));
    require_value(queue->append(draw_packet(
        {26, 1}, target, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory)));
    CHECK(control->wait_for_entries(1, std::chrono::seconds(5)));

    auto destruction = std::make_shared<AsyncStatus>();
    std::thread destroy_thread(
        [queue = std::move(queue), destruction]() mutable {
            destruction->begin();
            queue.reset();
            destruction->finish(renderer_success());
        });
    CHECK(destruction->wait_started(std::chrono::seconds(5)));
    CHECK(!destruction->wait_done(std::chrono::milliseconds(100)));
    control->release();
    const bool destroyed = destruction->wait_done(std::chrono::seconds(5));
    CHECK(destroyed);
    if (!destroyed) {
        destroy_thread.detach();
        return;
    }
    destroy_thread.join();
}

void test_batch_and_barrier(Fixture& fixture, const Pipelines& pipelines) {
    auto first = fixture.texture();
    auto second = fixture.texture();
    auto actions_a = std::make_shared<Actions>();
    auto actions_b = std::make_shared<Actions>();
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 8, 2));
    auto producer = queue->append(draw_packet({7, 1}, first, pipelines.producer,
                                               actions_a, fixture.memory));
    CHECK(std::holds_alternative<SubmissionReceipt>(producer));
    auto consumer = queue->append(draw_packet({7, 2}, second, pipelines.consumer,
                                               actions_b, fixture.memory,
                                               &first, pipelines.sampler));
    CHECK(std::holds_alternative<SubmissionReceipt>(consumer));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline.time_since_epoch()).count())));
    CHECK(actions_a->published == 1 && actions_b->published == 1);
    std::array<std::uint8_t, 4 * 4 * 4> bytes{};
    [second.storage->value getBytes:bytes.data() bytesPerRow:16
                         fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (std::size_t pixel = 0; pixel < 16; ++pixel) {
        CHECK(bytes[pixel * 4 + 0] >= 63 && bytes[pixel * 4 + 0] <= 64);
        CHECK(bytes[pixel * 4 + 1] >= 127 && bytes[pixel * 4 + 1] <= 128);
        CHECK(bytes[pixel * 4 + 2] >= 191 && bytes[pixel * 4 + 2] <= 192);
        CHECK(bytes[pixel * 4 + 3] == 255);
    }
    const auto statistics = metal::submission_statistics(*queue);
    CHECK(statistics.accepted == 2);
    CHECK(statistics.command_buffers == 1);
    CHECK(statistics.render_passes == 2);
    CHECK(statistics.draws == 2);
}

void test_persistent_argument_tables(Fixture& fixture, const Pipelines& pipelines) {
    auto red = fixture.texture();
    auto blue = fixture.texture();
    const auto fill = [](const TextureAsset& texture, std::array<std::uint8_t, 4> color) {
        std::array<std::uint8_t, 64> pixels{};
        for (std::size_t pixel = 0; pixel < 16; ++pixel)
            std::copy(color.begin(), color.end(), pixels.begin() + pixel * 4);
        [texture.storage->value replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
            mipmapLevel:0 withBytes:pixels.data() bytesPerRow:16];
    };
    fill(red, {255, 0, 0, 255});
    fill(blue, {0, 0, 255, 255});
    auto target = fixture.texture();
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 8, 2));
    const auto packet = [&](std::uint64_t sequence, std::uint32_t column,
                            const TextureAsset* sampled, bool high) {
        auto prepared = draw_packet({17, sequence}, target,
            sampled ? (high ? pipelines.consumer_high : pipelines.consumer)
                    : pipelines.absent_texture,
            std::make_shared<Actions>(), fixture.memory, sampled, pipelines.sampler);
        auto& draw = std::get<metal::PreparedDraw>(prepared.operation);
        draw.state.scissors = {{column, 0, 1, 4}};
        if (sampled && high) {
            draw.bindings.textures[0].texture_index = 127;
            draw.bindings.textures[0].sampler_index = 15;
        }
        return prepared;
    };
    const auto first = require_value(queue->append(packet(1, 0, &red, false)));
    require_value(queue->append(packet(2, 1, &blue, true)));

    // An indirect texture can still request a direct sampler. Its sampler
    // index must be admitted before any fixed-size shadow array is indexed.
    auto invalid_sampler = packet(3, 2, &red, true);
    auto& invalid_bindings = std::get<metal::PreparedDraw>(invalid_sampler.operation).bindings;
    invalid_bindings.argument_buffers.push_back({armigo::shader_input::Stage::Fragment,
                                                 {1, 2, 8, 8}});
    invalid_bindings.textures[0].argument_byte_offset = 0;
    invalid_bindings.textures[0].sampler_index = 16;
    CHECK(std::holds_alternative<RendererError>(queue->append(std::move(invalid_sampler))));

    // A missing native texture is rejected before acceptance or table changes.
    // Earlier queued draws keep their snapshots and the next draw keeps blue.
    auto rejected = packet(3, 2, &red, true);
    metal::TextureBinding missing;
    missing.stage = armigo::shader_input::Stage::Fragment;
    missing.sampler_index = armigo::shader::graphics_msl::TextureBinding::no_sampler;
    std::get<metal::PreparedDraw>(rejected.operation).bindings.textures.push_back(missing);
    CHECK(std::holds_alternative<RendererError>(queue->append(std::move(rejected))));
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Accepted);
    require_value(queue->append(packet(3, 2, &blue, true)));
    // A pipeline with no declared bindings must see the same nil texture that
    // a fresh initialized table supplied before table reuse was introduced.
    require_value(queue->append(packet(4, 3, nullptr, false)));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline.time_since_epoch()).count())));
    std::array<std::uint8_t, 64> bytes{};
    [target.storage->value getBytes:bytes.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    const std::array<std::array<std::uint8_t, 4>, 4> expected{{
        {255, 0, 0, 255}, {0, 0, 255, 255}, {0, 0, 255, 255}, {0, 255, 0, 255}}};
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            for (std::size_t component = 0; component < 4; ++component)
                CHECK(bytes[(row * 4 + column) * 4 + component] == expected[column][component]);
    auto statistics = metal::submission_statistics(*queue);
    CHECK(statistics.accepted == 4);
    CHECK(statistics.draws == 4);
    CHECK(statistics.render_passes == 1);
    CHECK(statistics.argument_tables == 1);

    // Reuse continues across completion and a new command buffer. Rebinding
    // low slots after the high-slot pipeline must not allocate another table.
    require_value(queue->append(packet(5, 3, &red, false)));
    const auto next_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        next_deadline.time_since_epoch()).count())));
    [target.storage->value getBytes:bytes.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (std::size_t row = 0; row < 4; ++row) {
        CHECK(bytes[(row * 4 + 3) * 4] == 255);
        CHECK(bytes[(row * 4 + 3) * 4 + 1] == 0);
        CHECK(bytes[(row * 4 + 3) * 4 + 2] == 0);
        CHECK(bytes[(row * 4 + 3) * 4 + 3] == 255);
    }
    CHECK(metal::submission_statistics(*queue).argument_tables == 1);
}

void test_argument_arenas_and_command_pool(Fixture& fixture, const Pipelines& pipelines) {
    auto red = fixture.texture();
    auto blue = fixture.texture();
    const auto fill = [](const TextureAsset& texture, std::array<std::uint8_t, 4> color) {
        std::array<std::uint8_t, 64> pixels{};
        for (std::size_t pixel = 0; pixel < 16; ++pixel)
            std::copy(color.begin(), color.end(), pixels.begin() + pixel * 4);
        [texture.storage->value replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
            mipmapLevel:0 withBytes:pixels.data() bytesPerRow:16];
    };
    fill(red, {255, 0, 0, 255});
    fill(blue, {0, 0, 255, 255});
    auto* zero_descriptor = [MTLSamplerDescriptor new];
    zero_descriptor.supportArgumentBuffers = YES;
    zero_descriptor.minFilter = MTLSamplerMinMagFilterNearest;
    zero_descriptor.magFilter = MTLSamplerMinMagFilterNearest;
    zero_descriptor.sAddressMode = MTLSamplerAddressModeClampToZero;
    zero_descriptor.tAddressMode = MTLSamplerAddressModeClampToZero;
    auto zero_sampler = std::make_shared<Sampler>(
        [fixture.context->device() newSamplerStateWithDescriptor:zero_descriptor]);
    CHECK(zero_sampler->value != nil);
    auto first_target = fixture.texture();
    auto second_target = fixture.texture();
    auto third_target = fixture.texture();
    auto control = std::make_shared<DelayedFeedback>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 16, 3, control));
    const auto packet = [&](std::uint64_t sequence, const TextureAsset& target,
                            const TextureAsset& sampled, std::uint32_t pixel,
                            std::uint32_t alignment, std::shared_ptr<Sampler> sampler) {
        auto prepared = draw_packet({18, sequence}, target, pipelines.argument_consumer,
            std::make_shared<Actions>(), fixture.memory, &sampled);
        auto& draw = std::get<metal::PreparedDraw>(prepared.operation);
        draw.state.scissors = {{pixel % 4, pixel / 4, 1, 1}};
        draw.bindings.argument_buffers.push_back({armigo::shader_input::Stage::Fragment,
                                                  {1, 2, 16, alignment}});
        draw.bindings.textures[0].argument_byte_offset = 0;
        draw.bindings.textures[0].sampler_index =
            armigo::shader::graphics_msl::TextureBinding::no_sampler;
        draw.bindings.samplers.push_back({armigo::shader_input::Stage::Fragment, 0,
                                          std::move(sampler), 8});
        return prepared;
    };
    auto first = require_value(queue->append(packet(1, first_target, red, 0, 8, pipelines.sampler)));
    require_value(queue->append(packet(2, first_target, red, 1, 8, pipelines.sampler)));
    require_value(queue->append(packet(3, first_target, blue, 2, 8, pipelines.sampler)));
    auto statistics = metal::submission_statistics(*queue);
    CHECK(statistics.argument_pages == 1 && statistics.argument_slices == 2);
    CHECK(statistics.argument_slice_reuses == 1);

    // Reused slices still consume the original logical per-packet budget.
    auto over_budget = packet(4, first_target, blue, 3, 8, pipelines.sampler);
    std::get<metal::PreparedDraw>(over_budget.operation).feedback_bytes_limit = 22;
    auto rejected_budget = queue->append(std::move(over_budget));
    CHECK(std::holds_alternative<RendererError>(rejected_budget));
    if (const auto* error = std::get_if<RendererError>(&rejected_budget))
        CHECK(error->command().status == CommandStatus::BudgetExceeded);
    CHECK(metal::submission_statistics(*queue).argument_slices == 2);

    // Invalid viewport state is rejected before allocating a changed slice.
    // Earlier accepted slices and their current blue binding remain intact.
    auto invalid = packet(4, first_target, red, 3, 8, pipelines.sampler);
    std::get<metal::PreparedDraw>(invalid.operation).state.viewports[0].width = -1;
    CHECK(std::holds_alternative<RendererError>(queue->append(std::move(invalid))));
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Accepted);
    require_value(queue->append(packet(4, first_target, blue, 3, 8, pipelines.sampler)));
    // Identical IDs under a different layout alignment require a fresh slice.
    require_value(queue->append(packet(5, first_target, blue, 4, 512, pipelines.sampler)));
    require_value(queue->append(packet(6, first_target, blue, 5, 512, pipelines.sampler)));
    // A sampler-only change must also change the indirect argument contents.
    require_value(queue->append(packet(7, first_target, blue, 6, 512, zero_sampler)));
    statistics = metal::submission_statistics(*queue);
    CHECK(statistics.argument_pages == 1 && statistics.argument_slices == 4);
    CHECK(statistics.argument_slice_reuses == 3);
    CHECK(statistics.command_bundles == 1);
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->wait_until_entered();
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Submitted);

    // Even though Metal is quiescent, the delayed feedback has not retired the
    // batch. A second batch must acquire a different arena and command bundle.
    auto second = require_value(queue->append(packet(8, second_target, blue, 0, 8,
                                                     pipelines.sampler)));
    statistics = metal::submission_statistics(*queue);
    CHECK(statistics.argument_pages == 2 && statistics.argument_page_reuses == 0);
    CHECK(statistics.command_bundles == 2 && statistics.command_bundle_reuses == 0);
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->release(false);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline.time_since_epoch()).count())));
    CHECK(first.completion.snapshot().phase == SubmissionPhase::Complete);
    CHECK(second.completion.snapshot().phase == SubmissionPhase::Complete);
    std::array<std::uint8_t, 64> pixels{};
    [first_target.storage->value getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    const std::array<std::array<std::uint8_t, 4>, 7> expected{{
        {255, 0, 0, 255}, {255, 0, 0, 255}, {0, 0, 255, 255}, {0, 0, 255, 255},
        {0, 0, 255, 255}, {0, 0, 255, 255}, {0, 0, 0, 0}}};
    for (std::size_t pixel = 0; pixel < expected.size(); ++pixel)
        for (std::size_t component = 0; component < 4; ++component)
            CHECK(pixels[pixel * 4 + component] == expected[pixel][component]);
    [second_target.storage->value getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    CHECK(pixels[0] == 0 && pixels[1] == 0 && pixels[2] == 255 && pixels[3] == 255);

    // Recycling starts on the coordinator only after whole-batch retirement.
    require_value(queue->append(packet(9, third_target, red, 0, 8, pipelines.sampler)));
    statistics = metal::submission_statistics(*queue);
    CHECK(statistics.argument_pages == 2 && statistics.argument_page_reuses == 1);
    CHECK(statistics.command_bundles == 2 && statistics.command_bundle_reuses == 1);
    const auto final_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        final_deadline.time_since_epoch()).count())));
    [third_target.storage->value getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    CHECK(pixels[0] == 255 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 255);
}

void test_native_incarnation_breaks_pass(Fixture& fixture,
                                         const Pipelines& pipelines) {
    auto first = fixture.texture();
    auto second = first;
    auto* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    second.storage = std::make_shared<TextureStorage>(
        [fixture.context->device() newTextureWithDescriptor:descriptor],
        first.retained.bytes);
    second.resident.storage = second.storage;
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 4, 1));
    CHECK(std::holds_alternative<SubmissionReceipt>(queue->append(draw_packet(
        {8, 1}, first, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory))));
    CHECK(std::holds_alternative<SubmissionReceipt>(queue->append(draw_packet(
        {8, 2}, second, pipelines.producer, std::make_shared<Actions>(),
        fixture.memory))));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline.time_since_epoch()).count())));
    const auto statistics = metal::submission_statistics(*queue);
    CHECK(statistics.command_buffers == 1);
    CHECK(statistics.render_passes == 2);
}

void test_coordinator_failure_orders_later_effects(
    Fixture& fixture, const Pipelines& pipelines) {
    auto first = fixture.texture();
    auto second = fixture.texture();
    auto third = fixture.texture();
    auto first_actions = std::make_shared<Actions>();
    auto same_batch_actions = std::make_shared<Actions>();
    auto later_batch_actions = std::make_shared<Actions>();
    first_actions->throw_coherent = true;
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 4, 2));
    auto first_receipt = require_value(queue->append(draw_packet(
        {9, 1}, first, pipelines.producer, first_actions, fixture.memory)));
    auto same_receipt = require_value(queue->append(draw_packet(
        {9, 2}, second, pipelines.producer, same_batch_actions,
        fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    auto later_receipt = require_value(queue->append(draw_packet(
        {9, 3}, third, pipelines.producer, later_batch_actions,
        fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(10);
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        static_cast<void>(queue->drain(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                deadline.time_since_epoch()).count()));
    }
    CHECK(first_receipt.completion.snapshot().phase == SubmissionPhase::Failed);
    CHECK(same_receipt.completion.snapshot().phase == SubmissionPhase::Failed);
    CHECK(later_receipt.completion.snapshot().phase == SubmissionPhase::Failed);
    CHECK(first_actions->published == 0 && same_batch_actions->published == 0 &&
          later_batch_actions->published == 0);
    CHECK(first_actions->preserved == 1 && same_batch_actions->preserved == 1 &&
          later_batch_actions->preserved == 1);
    CHECK(std::holds_alternative<RendererError>(queue->append(draw_packet(
        {9, 4}, fixture.texture(), pipelines.producer,
        std::make_shared<Actions>(), fixture.memory))));
}

void test_delayed_success_and_backpressure(Fixture& fixture,
                                           const Pipelines& pipelines) {
    auto target = fixture.texture();
    auto later_target = fixture.texture();
    auto actions = std::make_shared<Actions>();
    auto control = std::make_shared<DelayedFeedback>();
    auto destructions = std::make_shared<std::atomic<unsigned>>(0);
    auto transient_pipeline = std::make_shared<Pipeline>(
        pipelines.producer->value, destructions);
    std::weak_ptr<Pipeline> retained = transient_pipeline;
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 1, control));
    auto accepted = require_value(queue->append(draw_packet(
        {10, 1}, target, transient_pipeline, actions, fixture.memory)));
    transient_pipeline.reset();
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->wait_until_entered();
    CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(!retained.expired() && *destructions == 0);

    auto backpressured = queue->append(draw_packet(
        {10, 2}, later_target, pipelines.producer,
        std::make_shared<Actions>(), fixture.memory));
    CHECK(std::holds_alternative<RendererError>(backpressured));
    CHECK(std::get<RendererError>(backpressured).command().status ==
          CommandStatus::Stopped);
    const auto expired = std::chrono::steady_clock::now();
    CHECK(std::holds_alternative<RendererError>(queue->drain(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            expired.time_since_epoch()).count())));
    CHECK(!retained.expired() && actions->published == 0);

    control->release(false);
    control->wait_until_returned();
    for (unsigned attempt = 0; attempt < 1000 &&
         accepted.completion.snapshot().phase == SubmissionPhase::Submitted;
         ++attempt)
        std::this_thread::yield();
    CHECK(accepted.completion.snapshot().phase == SubmissionPhase::GpuComplete);
    CHECK(!retained.expired() && actions->published == 0);
    CHECK(succeeded(queue->poll()));
    CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Complete);
    // Poll completes publication immediately; ownership release is deferred.
    static_cast<void>(metal::submission_statistics(*queue));
    CHECK(retained.expired() && *destructions == 1 && actions->published == 1);

    auto recovered = queue->append(draw_packet(
        {10, 2}, later_target, pipelines.producer,
        std::make_shared<Actions>(), fixture.memory));
    CHECK(std::holds_alternative<SubmissionReceipt>(recovered));
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(10);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline.time_since_epoch()).count())));
}

void test_delayed_error_retains(Fixture& fixture, const Pipelines& pipelines) {
    auto destructions = std::make_shared<std::atomic<unsigned>>(0);
    auto target = fixture.texture(destructions);
    std::weak_ptr<TextureStorage> retained = target.storage;
    auto actions = std::make_shared<Actions>();
    actions->preserve_failures = 1;
    auto control = std::make_shared<DelayedFeedback>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 1, control));
    auto accepted = queue->append(draw_packet({7, 3}, target, pipelines.producer,
                                               actions, fixture.memory));
    CHECK(std::holds_alternative<SubmissionReceipt>(accepted));
    const auto token = std::get<SubmissionReceipt>(accepted).completion;
    target = {};
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->wait_until_entered();
    CHECK(token.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(!retained.expired() && *destructions == 0);
    const auto expired = std::chrono::steady_clock::now();
    CHECK(std::holds_alternative<RendererError>(queue->drain(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            expired.time_since_epoch()).count())));
    CHECK(!retained.expired() && actions->published == 0);
    control->release(true);
    control->wait_until_returned();
    for (unsigned attempt = 0; attempt < 1000 &&
         token.snapshot().phase == SubmissionPhase::Submitted; ++attempt)
        std::this_thread::yield();
    const auto failed_handoff = queue->poll();
    CHECK(std::holds_alternative<RendererError>(failed_handoff));
    CHECK(std::get<RendererError>(failed_handoff).command().detail ==
          "synthetic failed-authority handoff failure");
    CHECK(token.snapshot().phase == SubmissionPhase::Failed);
    CHECK(actions->coherent == 0 && actions->prepared == 0 &&
          actions->published == 0 && actions->preserved == 1);
    CHECK(!retained.expired() && *destructions == 0);
    const auto result = queue->poll();
    CHECK(std::holds_alternative<RendererError>(result));
    CHECK(actions->preserved == 2);
    // Failed writer authority moves to the registry recovery owner. Releasing
    // the queue packet must not make the allocation reusable or destroy it.
    CHECK(!retained.expired() && *destructions == 0);
    const auto future = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    CHECK(succeeded(queue->drain(std::chrono::duration_cast<std::chrono::nanoseconds>(
        future.time_since_epoch()).count())));
    auto rejected_target = fixture.texture();
    auto rejected = queue->append(draw_packet({7, 4}, rejected_target,
        pipelines.producer, std::make_shared<Actions>(), fixture.memory));
    CHECK(std::holds_alternative<RendererError>(rejected));
    CHECK(std::get<RendererError>(rejected).command().detail ==
          "synthetic delayed Metal feedback failure");
}

std::uint64_t deadline_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        (std::chrono::steady_clock::now() + std::chrono::seconds(10))
            .time_since_epoch()).count();
}

metal::PreparedRender clear_packet(SubmissionId id, const TextureAsset& target,
                                   ClearPlan clear,
                                   std::shared_ptr<ResourceMemory> memory,
                                   std::shared_ptr<Actions> actions =
                                       std::make_shared<Actions>()) {
    metal::PreparedRender packet;
    auto snapshot = std::make_shared<RenderStop>();
    snapshot->kind = RenderKind::Clear;
    packet.request = {id, snapshot, std::move(memory), {}, std::move(actions)};
    packet.targets.area = {0, 0, 4, 4};
    packet.targets.samples = 1;
    packet.attachments.key.area = packet.targets.area;
    packet.attachments.key.samples = 1;
    if (clear.color_mask) {
        packet.targets.colors[0] = target.attachment;
        packet.attachments.colors[0] = target.resident;
        packet.attachments.key.colors[0] = target.resident.key;
    }
    if (clear.clear_depth) {
        packet.targets.depth = target.attachment;
        packet.targets.depth->aspect = TextureAspect::Depth;
        packet.attachments.depth = target.resident;
        packet.attachments.key.depth = target.resident.key;
    }
    if (clear.clear_stencil) {
        packet.targets.stencil = target.attachment;
        packet.targets.stencil->aspect = TextureAspect::Stencil;
        packet.attachments.stencil = target.resident;
        packet.attachments.key.stencil = target.resident.key;
    }
    clear.target = target.attachment;
    packet.operation = metal::PreparedClear{clear};
    packet.accesses.push_back({target.retained, ResourceAccess::Write});
    packet.access_storage.push_back(target.storage);
    return packet;
}

metal::PreparedRender empty_clear_packet(SubmissionId id,
                                        std::shared_ptr<ResourceMemory> memory,
                                        std::shared_ptr<Actions> actions) {
    metal::PreparedRender packet;
    auto snapshot = std::make_shared<RenderStop>();
    snapshot->kind = RenderKind::Clear;
    snapshot->engine = engine_class::threed;
    snapshot->registers[0x674] = 1;
    snapshot->depth.enabled = true;
    packet.request = {id, snapshot, std::move(memory), {}, std::move(actions)};
    ClearPlan clear;
    clear.no_op = true;
    packet.operation = metal::PreparedClear{clear};
    return packet;
}

void test_empty_clear_batch_waits_for_prior_feedback(Fixture& fixture,
                                                    const Pipelines& pipelines) {
    for (const bool fail : {false, true}) {
        const std::uint64_t sequence = fail ? 3 : 1;
        auto target = fixture.texture();
        auto first_actions = std::make_shared<Actions>();
        auto empty_actions = std::make_shared<Actions>();
        empty_actions->predecessor = first_actions.get();
        auto control = std::make_shared<DelayedFeedback>();
        auto queue = require_value(metal::create_submission_queue_for_testing(
            fixture.context, fixture.memory, 4, 2, control));
        auto first = require_value(queue->append(draw_packet(
            {20, sequence}, target, pipelines.producer, first_actions, fixture.memory)));
        CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
        control->wait_until_entered();
        CHECK(first.completion.snapshot().phase == SubmissionPhase::Submitted);

        auto empty = require_value(queue->append(empty_clear_packet(
            {20, sequence + 1}, fixture.memory, empty_actions)));
        CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
        CHECK(empty.completion.snapshot().phase == SubmissionPhase::Submitted);
        CHECK(succeeded(queue->poll()));
        CHECK(first_actions->published == 0 && empty_actions->published == 0);
        CHECK(empty_actions->coherent == 0 && empty_actions->prepared == 0);
        CHECK(empty.completion.snapshot().phase == SubmissionPhase::Submitted);
        const auto statistics = metal::submission_statistics(*queue);
        CHECK(statistics.accepted == 2 && statistics.command_buffers == 1 &&
              statistics.render_passes == 1 && statistics.draws == 1);

        control->release(fail);
        control->wait_until_returned();
        if (fail) {
            CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns())));
            CHECK(first.completion.snapshot().phase == SubmissionPhase::Failed);
            CHECK(empty_actions->published == 0);
            CHECK(std::holds_alternative<RendererError>(queue->drain(deadline_ns())));
            CHECK(succeeded(queue->drain(deadline_ns())));
            CHECK(empty.completion.snapshot().phase == SubmissionPhase::Failed);
            CHECK(first_actions->published == 0 && empty_actions->published == 0);
            CHECK(first_actions->preserved == 1 && empty_actions->preserved == 1);
        } else {
            CHECK(succeeded(queue->drain(deadline_ns())));
            CHECK(first.completion.snapshot().guest_visible_complete());
            CHECK(empty.completion.snapshot().guest_visible_complete());
            CHECK(first_actions->published == 1 && empty_actions->published == 1);
            CHECK(empty_actions->predecessor_published);
        }
        CHECK(metal::submission_statistics(*queue).command_buffers == 1);
    }
}

void test_empty_clear_cross_queue_dependencies(Fixture& fixture,
                                               const Pipelines& pipelines) {
    for (const bool separate_batches : {false, true}) {
        const std::uint64_t sequence = separate_batches ? 4 : 1;
        auto source = fixture.texture();
        auto target = fixture.texture();
        auto control = std::make_shared<DelayedFeedback>();
        auto producer = require_value(metal::create_submission_queue_for_testing(
            fixture.context, fixture.memory, 4, 3, control));
        // A different device context owns a genuinely independent native queue.
        auto consumer_context = require_value(
            metal::create_device_context(fixture.context->device()));
        CHECK(consumer_context->queue() != fixture.context->queue());
        auto consumer = require_value(metal::create_submission_queue(
            consumer_context, fixture.memory, 4, 1));
        auto first = require_value(producer->append(draw_packet(
            {22, sequence}, source, pipelines.producer, std::make_shared<Actions>(), fixture.memory)));
        CHECK(succeeded(producer->flush(SubmissionBoundary::SubmissionEnd)));
        control->wait_until_entered();
        CHECK(first.completion.snapshot().phase == SubmissionPhase::Submitted);

        auto middle_packet = empty_clear_packet({22, sequence + 1}, fixture.memory,
                                                std::make_shared<Actions>());
        middle_packet.request.dependencies.push_back(first.completion);
        auto middle = require_value(producer->append(std::move(middle_packet)));
        if (separate_batches)
            CHECK(succeeded(producer->flush(SubmissionBoundary::SubmissionEnd)));
        auto last_packet = empty_clear_packet({22, sequence + 2}, fixture.memory,
                                              std::make_shared<Actions>());
        last_packet.request.dependencies.push_back(middle.completion);
        auto last = require_value(producer->append(std::move(last_packet)));
        CHECK(succeeded(producer->flush(SubmissionBoundary::SubmissionEnd)));
        CHECK(middle.completion.snapshot().phase == SubmissionPhase::Submitted);
        CHECK(last.completion.snapshot().phase == SubmissionPhase::Submitted);

        auto consumer_actions = std::make_shared<Actions>();
        auto consumer_packet = draw_packet({23, sequence}, target, pipelines.producer,
                                           consumer_actions, fixture.memory);
        consumer_packet.request.dependencies.push_back(last.completion);
        auto accepted = require_value(consumer->append(std::move(consumer_packet)));
        CHECK(std::holds_alternative<RendererError>(
            consumer->flush(SubmissionBoundary::Dependency)));
        CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Accepted);
        CHECK(metal::submission_statistics(*consumer).command_buffers == 0);
        CHECK(succeeded(producer->poll()));
        CHECK(consumer_actions->published == 0);

        // The delay is at the feedback boundary after actual GPU quiescence;
        // the token must still prevent another native queue from submitting.
        control->release(false);
        control->wait_until_returned();
        CHECK(succeeded(producer->drain(deadline_ns())));
        CHECK(middle.completion.snapshot().guest_visible_complete());
        CHECK(last.completion.snapshot().guest_visible_complete());
        CHECK(succeeded(consumer->drain(deadline_ns())));
        CHECK(accepted.completion.snapshot().guest_visible_complete());
        CHECK(consumer_actions->published == 1);
        CHECK(metal::submission_statistics(*producer).command_buffers == 1);
        CHECK(metal::submission_statistics(*consumer).command_buffers == 1);
    }
}

void test_empty_clear_rejects_retained_work(Fixture& fixture) {
    auto target = fixture.texture();
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 4, 1));
    for (unsigned malformed = 0; malformed < 3; ++malformed) {
        auto actions = std::make_shared<Actions>();
        auto packet = empty_clear_packet({21, 1}, fixture.memory, actions);
        if (malformed == 0) {
            packet.accesses.push_back({target.retained, ResourceAccess::Write});
            packet.access_storage.push_back(target.storage);
        } else if (malformed == 1) {
            packet.attachments.colors[0] = target.resident;
            packet.attachments.key.colors[0] = target.resident.key;
        } else {
            std::get<metal::PreparedClear>(packet.operation).clear.clear_depth = true;
        }
        CHECK(std::holds_alternative<RendererError>(queue->append(std::move(packet))));
        CHECK(actions->coherent == 0 && actions->prepared == 0 && actions->published == 0);
    }
    CHECK(metal::submission_statistics(*queue).accepted == 0);
    const auto state = require_value(fixture.memory->inspect_storage(target.retained.bytes));
    for (const auto& interval : state) CHECK(interval.dirty.version.gpu_write == 0);
    auto actions = std::make_shared<Actions>();
    auto accepted = require_value(queue->append(empty_clear_packet({21, 1}, fixture.memory, actions)));
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(accepted.completion.snapshot().guest_visible_complete());
    CHECK(actions->published == 1);
    CHECK(metal::submission_statistics(*queue).command_buffers == 0);
}

void test_overlapping_partial_clears(Fixture& fixture) {
    // Both clears load and store the same tiles, preserve different channels,
    // and overlap in the center. Run both in one commit without an intervening wait.
    for (unsigned iteration = 0; iteration < 16; ++iteration) {
        auto target = fixture.texture();
        std::array<std::uint8_t, 64> bytes;
        bytes.fill(0x33);
        [target.storage->value replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
            mipmapLevel:0 withBytes:bytes.data() bytesPerRow:16];
        auto queue = require_value(metal::create_submission_queue(
            fixture.context, fixture.memory, 4, 1));
        ClearPlan first;
        first.region = {0, 0, 3, 3};
        first.color_mask = 0x3;
        first.color_bits = {std::bit_cast<std::uint32_t>(1.0f), 0, 0, 0};
        ClearPlan second;
        second.region = {1, 1, 3, 3};
        second.color_mask = 0x6;
        second.color_bits = {0, std::bit_cast<std::uint32_t>(1.0f),
                             std::bit_cast<std::uint32_t>(1.0f), 0};
        require_value(queue->append(clear_packet({11, iteration * 2 + 1}, target, first, fixture.memory)));
        require_value(queue->append(clear_packet({11, iteration * 2 + 2}, target, second, fixture.memory)));
        CHECK(succeeded(queue->drain(deadline_ns())));
        [target.storage->value getBytes:bytes.data() bytesPerRow:16
            fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
        for (unsigned y = 0; y < 4; ++y) {
            for (unsigned x = 0; x < 4; ++x) {
                const bool a = x < 3 && y < 3;
                const bool b = x >= 1 && y >= 1;
                const auto offset = (y * 4 + x) * 4;
                CHECK(bytes[offset] == (a ? 255 : 0x33));
                CHECK(bytes[offset + 1] == (b ? 255 : a ? 0 : 0x33));
                CHECK(bytes[offset + 2] == (b ? 255 : 0x33));
                CHECK(bytes[offset + 3] == 0x33);
            }
        }
    }
}

void test_empty_outside_scissor(Fixture& fixture, const Pipelines& pipelines) {
    auto target = fixture.texture();
    std::array<std::uint8_t, 64> bytes;
    bytes.fill(0x33);
    [target.storage->value replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
        mipmapLevel:0 withBytes:bytes.data() bytesPerRow:16];
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 4, 1));
    auto actions = std::make_shared<Actions>();
    auto packet = draw_packet({12, 1}, target, pipelines.producer, actions, fixture.memory);
    std::get<metal::PreparedDraw>(packet.operation).state.scissors = {{2000, 0, 0, 100}};
    auto result = queue->append(std::move(packet));
    CHECK(std::holds_alternative<SubmissionReceipt>(result));
    if (!std::holds_alternative<SubmissionReceipt>(result)) return;
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(std::get<SubmissionReceipt>(result).completion.snapshot().phase ==
          SubmissionPhase::Complete);
    CHECK(actions->published == 1);
    [target.storage->value getBytes:bytes.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (auto byte : bytes) CHECK(byte == 0x33);
}

void test_failed_dependency_retires_accepted_batch(Fixture& fixture,
                                                  const Pipelines& pipelines) {
    auto source = fixture.texture();
    auto target = fixture.texture();
    auto control = std::make_shared<DelayedFeedback>();
    auto queue = require_value(metal::create_submission_queue_for_testing(
        fixture.context, fixture.memory, 4, 2, control));
    auto actions_a = std::make_shared<Actions>();
    auto actions_b = std::make_shared<Actions>();
    auto a = require_value(queue->append(draw_packet(
        {13, 1}, source, pipelines.producer, actions_a, fixture.memory)));
    CHECK(succeeded(queue->flush(SubmissionBoundary::SubmissionEnd)));
    control->wait_until_entered();
    auto transient = std::make_shared<Pipeline>(pipelines.consumer->value);
    std::weak_ptr<Pipeline> retained = transient;
    auto b = require_value(queue->append(draw_packet(
        {13, 2}, target, transient, actions_b, fixture.memory, &source, pipelines.sampler)));
    transient.reset();
    CHECK(b.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(!retained.expired());
    control->release(true);
    control->wait_until_returned();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (a.completion.snapshot().phase == SubmissionPhase::Submitted &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    CHECK(a.completion.snapshot().phase == SubmissionPhase::Failed);
    CHECK(std::holds_alternative<RendererError>(
        queue->flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(b.completion.snapshot().phase == SubmissionPhase::Failed);
    // Retirement reports each failure once and must eventually drain the queue.
    for (unsigned attempt = 0; attempt < 4; ++attempt)
        static_cast<void>(queue->poll());
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(actions_a->preserved == 1 && actions_b->preserved == 1);
    CHECK(actions_a->published == 0 && actions_b->published == 0);
    CHECK(retained.expired());
    // The committed writer token must no longer be Accepted and block recovery.
    const auto state = require_value(fixture.memory->inspect_storage(target.retained.bytes));
    CHECK(!state.empty());
    for (const auto& interval : state)
        CHECK(interval.dirty.last_writer.snapshot().phase == SubmissionPhase::Failed);
    CHECK(metal::submission_statistics(*queue).command_buffers == 1);
}

void test_dependency_readiness_and_terminal_stopped(Fixture& fixture,
                                                    const Pipelines& pipelines) {
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
        const bool fail = scenario & 1;
        const bool no_op = scenario & 2;
        const std::uint64_t sequence = scenario + 1;
        auto target = fixture.texture();
        auto actions = std::make_shared<Actions>();
        SubmissionCompletion dependency({17, sequence}, {},
                                         std::make_shared<Actions>());
        SubmissionCompletion pending({18, sequence}, {}, std::make_shared<Actions>());
        auto queue = require_value(metal::create_submission_queue(
            fixture.context, fixture.memory, 4, 1));
        auto packet = no_op
            ? empty_clear_packet({19, sequence}, fixture.memory, actions)
            : draw_packet({19, sequence}, target, pipelines.producer, actions, fixture.memory);
        if (fail) packet.request.dependencies.push_back(pending.token());
        packet.request.dependencies.push_back(dependency.token());
        auto accepted = require_value(queue->append(std::move(packet)));
        CHECK(std::holds_alternative<RendererError>(
            queue->flush(SubmissionBoundary::SubmissionEnd)));
        CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Accepted);
        if (fail) {
            CHECK(succeeded(dependency.reject(
                {CommandStatus::Stopped, "synthetic terminal dependency failure"})));
            CHECK(std::holds_alternative<RendererError>(
                queue->flush(SubmissionBoundary::SubmissionEnd)));
            CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Failed);
            CHECK(std::holds_alternative<RendererError>(queue->poll()));
            CHECK(succeeded(queue->drain(deadline_ns())));
            CHECK(actions->preserved == 1 && actions->published == 0);
            CHECK(metal::submission_statistics(*queue).command_buffers == 0);
        } else {
            CHECK(succeeded(dependency.submitted()));
            CHECK(std::holds_alternative<RendererError>(
                queue->flush(SubmissionBoundary::SubmissionEnd)));
            CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Accepted);
            CHECK(succeeded(dependency.feedback({})));
            CHECK(succeeded(queue->drain(deadline_ns())));
            CHECK(accepted.completion.snapshot().phase == SubmissionPhase::Complete);
            CHECK(actions->preserved == 0 && actions->published == 1);
            CHECK(metal::submission_statistics(*queue).command_buffers == (no_op ? 0 : 1));
        }
    }
}

TextureAsset packed_texture(Fixture& fixture) {
    auto target = fixture.texture();
    auto* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                     width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                       MTLTextureUsagePixelFormatView;
    target.storage->value = [fixture.context->device() newTextureWithDescriptor:descriptor];
    CHECK(target.storage->value != nil);
    target.attachment.texture.format = TextureFormat::D32FloatS8Uint;
    target.attachment.texture.row_bytes = 32;
    target.attachment.aspect = TextureAspect::DepthStencil;
    target.resident.key.storage.descriptor = target.attachment.texture;
    target.resident.key.aspect = TextureAspect::DepthStencil;
    return target;
}

TextureAsset stencil_view(const TextureAsset& packed) {
    auto stencil = packed;
    id<MTLTexture> view = [packed.storage->value
        newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    CHECK(view != nil);
    stencil.storage = std::make_shared<TextureStorage>(view, packed.retained.bytes);
    stencil.storage->backing_texture = packed.storage->value;
    stencil.resident.storage = stencil.storage;
    stencil.resident.key.aspect = TextureAspect::Stencil;
    stencil.attachment.aspect = TextureAspect::Stencil;
    return stencil;
}

void check_packed_pixels(Fixture& fixture, id<MTLTexture> texture,
                         bool combined) {
    // The packed format uses separate native planes. Read each aspect through
    // a blit after the submission queue has retired its GPU work.
    id<MTLCommandQueue> queue = [fixture.context->device() newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLBuffer> depth = [fixture.context->device()
        newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> stencil = [fixture.context->device()
        newBufferWithLength:16 options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> copy = [commands blitCommandEncoder];
    [copy copyFromTexture:texture sourceSlice:0 sourceLevel:0
        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(4, 4, 1)
        toBuffer:depth destinationOffset:0 destinationBytesPerRow:16
        destinationBytesPerImage:64 options:MTLBlitOptionDepthFromDepthStencil];
    [copy copyFromTexture:texture sourceSlice:0 sourceLevel:0
        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(4, 4, 1)
        toBuffer:stencil destinationOffset:0 destinationBytesPerRow:4
        destinationBytesPerImage:16 options:MTLBlitOptionStencilFromDepthStencil];
    [copy endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    CHECK(commands.status == MTLCommandBufferStatusCompleted);
    const auto* depths = static_cast<const float*>(depth.contents);
    const auto* stencils = static_cast<const std::uint8_t*>(stencil.contents);
    for (unsigned y = 0; y < 4; ++y) {
        for (unsigned x = 0; x < 4; ++x) {
            const bool inside = x >= 1 && x < 3 && y >= 1 && y < 3;
            CHECK(depths[y * 4 + x] == (combined && inside ? 0.75f : 0.25f));
            CHECK(stencils[y * 4 + x] == (inside ? 0xaa : 0xa7));
        }
    }
}

void test_packed_stencil_clear_preflight(Fixture& fixture, const Pipelines& pipelines) {
    auto target = packed_texture(fixture);
    auto stencil = stencil_view(target);
    auto color = fixture.texture();
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 8, 1));
    ClearPlan seed;
    seed.region = {0, 0, 4, 4};
    seed.clear_depth = seed.clear_stencil = true;
    seed.depth_value = 0.25;
    seed.stencil_value = 0xa7;
    seed.stencil_write_mask = 0xff;
    seed.fold_into_load = true;
    require_value(queue->append(clear_packet({14, 1}, target, seed, fixture.memory)));
    auto earlier_actions = std::make_shared<Actions>();
    auto earlier = require_value(queue->append(draw_packet(
        {14, 2}, color, pipelines.producer, earlier_actions, fixture.memory)));
    ClearPlan clear;
    clear.region = {1, 1, 2, 2};
    clear.clear_stencil = true;
    clear.stencil_value = 0x5a;
    clear.stencil_write_mask = 0x0f;
    auto accepted = queue->append(clear_packet({14, 3}, stencil, clear, fixture.memory));
    CHECK(std::holds_alternative<SubmissionReceipt>(accepted));
    CHECK(earlier.completion.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(earlier.completion.snapshot().phase == SubmissionPhase::Complete);
    CHECK(earlier_actions->published == 1);
    if (!std::holds_alternative<SubmissionReceipt>(accepted)) return;
    check_packed_pixels(fixture, target.storage->value, false);

    // The identical packed native texture satisfies the combined-clear contract.
    clear.clear_depth = true;
    clear.depth_value = 0.75;
    require_value(queue->append(clear_packet({14, 4}, target, clear, fixture.memory)));
    CHECK(succeeded(queue->drain(deadline_ns())));
    check_packed_pixels(fixture, target.storage->value, true);
}

void test_clear_format_rejection_preserves_batch(Fixture& fixture,
                                                 const Pipelines& pipelines) {
    auto color = fixture.texture();
    auto invalid = fixture.texture();
    invalid.attachment.texture.format = TextureFormat::D32FloatS8Uint;
    invalid.attachment.aspect = TextureAspect::Stencil;
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 8, 1));
    auto earlier = require_value(queue->append(draw_packet(
        {15, 1}, color, pipelines.producer, std::make_shared<Actions>(), fixture.memory)));
    ClearPlan clear;
    clear.region = {0, 0, 4, 4};
    clear.clear_stencil = true;
    clear.stencil_write_mask = 0x0f;
    auto rejected = queue->append(clear_packet({15, 2}, invalid, clear, fixture.memory));
    CHECK(std::holds_alternative<RendererError>(rejected));
    CHECK(earlier.completion.snapshot().phase == SubmissionPhase::Accepted);
    const auto state = require_value(fixture.memory->inspect_storage(invalid.retained.bytes));
    for (const auto& interval : state) CHECK(interval.dirty.version.gpu_write == 0);
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(earlier.completion.snapshot().phase == SubmissionPhase::Complete);
}

void test_combined_clear_invalid_subresource_preflight(Fixture& fixture,
                                                  const Pipelines& pipelines) {
    auto target = packed_texture(fixture);
    auto stencil = stencil_view(target);
    auto color = fixture.texture();
    auto queue = require_value(metal::create_submission_queue(
        fixture.context, fixture.memory, 8, 1));
    auto earlier = require_value(queue->append(draw_packet(
        {16, 1}, color, pipelines.producer, std::make_shared<Actions>(), fixture.memory)));
    ClearPlan clear;
    clear.region = {1, 1, 2, 2};
    clear.clear_depth = clear.clear_stencil = true;
    clear.stencil_write_mask = 0x0f;
    auto packet = clear_packet({16, 2}, target, clear, fixture.memory);
    packet.attachments.stencil = stencil.resident;
    // Aspect views of the same allocation are valid. A mismatched slice must
    // still fail before earlier work is flushed or any write intent is published.
    packet.attachments.stencil->key.first_layer = 1;
    packet.attachments.key.stencil = stencil.resident.key;
    CHECK(std::holds_alternative<RendererError>(queue->append(std::move(packet))));
    CHECK(earlier.completion.snapshot().phase == SubmissionPhase::Accepted);
    const auto state = require_value(fixture.memory->inspect_storage(target.retained.bytes));
    for (const auto& interval : state) CHECK(interval.dirty.version.gpu_write == 0);
    CHECK(succeeded(queue->drain(deadline_ns())));
    CHECK(earlier.completion.snapshot().phase == SubmissionPhase::Complete);
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::fprintf(stderr, "Metal 4 device is unavailable\n");
            return 1;
        }
        Fixture fixture(device);
        const auto pipelines = make_pipelines(device);
        test_asynchronous_encode_fifo_and_flush(fixture, pipelines);
        test_commit_hold_encodes_then_releases_in_order(fixture, pipelines);
        test_open_held_batch_wakes_encoder(fixture, pipelines);
        test_hold_preserves_earlier_commit_marker(fixture, pipelines);
        test_released_generation_survives_new_hold(fixture, pipelines);
        test_abort_held_commit_prevents_native_submission(fixture, pipelines);
        test_terminal_dependency_aborts_held_commit(fixture, pipelines);
        test_open_batch_prefix_retirement(fixture);
        test_retired_destruction_worker(fixture, pipelines);
        test_fallback_reservation_retires_on_coordinator(fixture, pipelines);
        test_held_queue_destruction(fixture);
        test_asynchronous_encode_failure_is_sticky(fixture, pipelines);
        test_asynchronous_commit_failure(fixture, pipelines);
        test_overlapping_partial_clears(fixture);
        test_empty_clear_batch_waits_for_prior_feedback(fixture, pipelines);
        test_empty_clear_cross_queue_dependencies(fixture, pipelines);
        test_empty_clear_rejects_retained_work(fixture);
        test_empty_outside_scissor(fixture, pipelines);
        test_failed_dependency_retires_accepted_batch(fixture, pipelines);
        test_dependency_readiness_and_terminal_stopped(fixture, pipelines);
        test_packed_stencil_clear_preflight(fixture, pipelines);
        test_clear_format_rejection_preserves_batch(fixture, pipelines);
        test_combined_clear_invalid_subresource_preflight(fixture, pipelines);
        test_batch_and_barrier(fixture, pipelines);
        test_persistent_argument_tables(fixture, pipelines);
        test_argument_arenas_and_command_pool(fixture, pipelines);
        test_native_incarnation_breaks_pass(fixture, pipelines);
        test_coordinator_failure_orders_later_effects(fixture, pipelines);
        test_delayed_success_and_backpressure(fixture, pipelines);
        test_delayed_error_retains(fixture, pipelines);
        test_encode_failure_preserves_earlier_batch_and_drains(fixture, pipelines);
        test_submission_destruction_waits_for_encode(fixture, pipelines);
    }
    if (failures) {
        std::fprintf(stderr, "%u submission checks failed\n", failures);
        return 1;
    }
    std::puts("Metal submission checks passed");
    return 0;
}
