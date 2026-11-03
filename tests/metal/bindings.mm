// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "platform/metal/bindings.h"
#include "platform/metal/device_context.h"
#include "platform/metal/texture_fact.h"
#include "platform/metal/texture_residency.h"
#include <armigo/graphics/resource_memory_registry.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace g = armigo::graphics;
namespace m = armigo::platform::metal;
namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Metal binding check failed: " << message << '\n';
        std::exit(1);
    }
}

bool error_contains(const g::RendererResult<m::Bindings>& result,
                    std::string_view text) {
    const auto* error = std::get_if<g::RendererError>(&result);
    return error && error->command().status == g::CommandStatus::Unsupported &&
           error->command().detail.find(text) != std::string_view::npos;
}

struct Lease final : g::ResourceLease {};
struct ReadTransaction final : g::CpuReadTransaction {};

class Memory final : public g::ResourceMemoryRegistry {
public:
    explicit Memory(std::size_t size = 4096) : bytes_(size) {
        const g::Mapping mapping{0x1000, bytes_.size(), 0, 0, bytes_.size(), 1};
        remap(std::span{&mapping, 1});
    }

    void remap(std::span<const g::Mapping> mappings) {
        auto made = g::make_address_space(mappings, {read, write, this});
        check(std::holds_alternative<g::GpuAddressSpace>(made),
              "synthetic address space must be valid");
        address_space_ = std::make_unique<g::GpuAddressSpace>(
            std::get<g::GpuAddressSpace>(std::move(made)));
    }

    std::span<std::byte> bytes() { return bytes_; }
    void word(std::uint64_t address, std::uint32_t value) {
        check(address >= 0x1000 && address - 0x1000 <= bytes_.size() - 4,
              "synthetic word address must be mapped");
        auto* destination = bytes_.data() + address - 0x1000;
        destination[0] = std::byte(value & 0xffu);
        destination[1] = std::byte((value >> 8) & 0xffu);
        destination[2] = std::byte((value >> 16) & 0xffu);
        destination[3] = std::byte((value >> 24) & 0xffu);
    }

    g::RendererStatus register_backing(g::BackingRegistration) override {
        return g::renderer_success();
    }
    g::RendererStatus register_mapping(g::MappingRegistration) override {
        return g::renderer_success();
    }
    g::RendererStatus unmap(g::MappingId) override { return g::renderer_success(); }
    g::RendererStatus retire_backing(g::BackingId) override {
        return g::renderer_success();
    }
    g::RendererResult<g::RetainedRange> retain(
        std::uint64_t address, std::uint64_t size, g::ResourceAccess) override {
        ++retains;
        for (const auto& mapping : address_space_->mappings()) {
            if (address < mapping.gpu_base || !size) continue;
            const auto offset = address - mapping.gpu_base;
            const auto backed = mapping.backing_size - mapping.buffer_offset;
            if (offset >= mapping.mapped_size || offset >= backed ||
                size > mapping.mapped_size - offset || size > backed - offset) continue;
            return g::RetainedRange{{1, mapping.gpu_base, 1},
                {{1, mapping.guest_address + 1, 1}, mapping.buffer_offset + offset, size}, missing_lease ? nullptr : lease_};
        }
        return g::RendererError{{g::CommandStatus::MemoryFault,
                                 "synthetic retain is out of range"}};
    }
    const g::GpuAddressSpace& address_space() const noexcept override {
        return *address_space_;
    }
    g::CoherenceMode coherence_mode() const noexcept override {
        return g::CoherenceMode::SubmissionBoundary;
    }
    g::RendererResult<std::vector<g::DirtyRange>> inspect(g::BackingRange range) override {
        ++inspections;
        return std::vector<g::DirtyRange>{{range, version, false, gpu_dirty, writer}};
    }
    g::RendererStatus invalidate_mappings(std::span<const g::MappingId>,
                                          std::span<const g::BackingId>) override {
        return g::renderer_success();
    }
    bool has_backing(g::BackingId) const noexcept override { return true; }
    g::RendererStatus validate_retained(const g::RetainedRange&, g::ResourceAccess) override {
        return g::renderer_success();
    }
    g::RendererStatus read_retained(const g::RetainedRange&, g::BackingRange,
                                    std::span<std::byte>) override {
        return g::RendererError{{g::CommandStatus::Unsupported,
                                 "synthetic retained reads are unsupported"}};
    }
    g::RendererResult<std::vector<g::ResourceStorageRange>> inspect_storage(
        g::BackingRange range) override {
        if (native_storage)
            return std::vector<g::ResourceStorageRange>{{{range, version, false, gpu_dirty, writer},
                std::dynamic_pointer_cast<const g::ResourceMemoryStorage>(storage)}};
        return g::RendererError{{g::CommandStatus::Unsupported,
                                 "synthetic storage inspection is unsupported"}};
    }
    g::RendererStatus submission_boundary() override { return g::renderer_success(); }
    std::uint64_t cpu_cache_epoch() const noexcept override { return 0; }
    // Write tracking is off unless a test enables it; stamps then advance per
    // call. Setting cpu_stored models a store at that moment: every range
    // reports it to stamps taken before it, while cpu_stored stays set.
    std::uint64_t cpu_write_stamp() const noexcept override {
        note_store();
        return cpu_tracking ? ++stamps : 0;
    }
    bool cpu_written_since(g::BackingRange range, std::uint64_t stamp) const noexcept override {
        note_store();
        return deferred_host_store(range) || !cpu_tracking || !stamp ||
            (cpu_stored && stamp <= store_mark);
    }
    bool deferred_host_store(g::BackingRange range) const noexcept override {
        return deferred && deferred->backing == range.backing &&
            deferred->offset < range.offset + range.size &&
            range.offset < deferred->offset + deferred->size;
    }
    bool has_deferred_host_store() const noexcept override { return deferred.has_value(); }
    std::optional<std::pair<std::uint64_t, std::uint64_t>> backing_host_range(
        g::BackingId backing) const noexcept override {
        if (!host_peeks || backing != (g::BackingId{1, 1, 1})) return std::nullopt;
        return std::pair{reinterpret_cast<std::uint64_t>(bytes_.data()),
                         std::uint64_t(bytes_.size())};
    }
    void note_store() const noexcept {
        if (cpu_stored && !store_seen) store_mark = stamps;
        store_seen = cpu_stored;
    }
    mutable std::uint64_t store_mark = 0;
    mutable bool store_seen = false;
    g::RendererStatus resolve(g::BackingRange range, g::ResolveReason) override {
        if (pending_pool_write && range.offset == 0x200)
            return g::RendererError{{g::CommandStatus::Stopped, "synthetic pending pool writer"}};
        return g::renderer_success();
    }
    g::RendererResult<std::unique_ptr<g::CpuReadTransaction>> prepare_cpu_read(
        g::BackingRange, g::ResolveReason) override {
        return std::unique_ptr<g::CpuReadTransaction>{
            std::make_unique<ReadTransaction>()};
    }
    g::RendererResult<std::unique_ptr<g::CpuWriteTransaction>> prepare_cpu_write(
        g::BackingRange) override {
        return g::RendererError{{g::CommandStatus::Unsupported,
                                 "synthetic writes are unsupported"}};
    }
    g::RendererResult<std::shared_ptr<g::GpuAccessReservation>> reserve_accesses(
        std::span<const g::GpuAccessRequest>, g::CompletionToken) override {
        return g::RendererError{{g::CommandStatus::Unsupported,
                                 "synthetic reservations are unsupported"}};
    }

    std::vector<std::pair<std::uint64_t, std::size_t>> reads;
    std::size_t retains = 0, inspections = 0;
    bool gpu_dirty = false, missing_lease = false;
    bool cpu_tracking = false, cpu_stored = false, pending_pool_write = false;
    bool host_peeks = false, native_storage = false;
    g::CompletionToken writer;
    g::RetainedResource storage;
    std::optional<g::BackingRange> deferred;
    mutable std::uint64_t stamps = 0;
    g::ResourceVersion version{};

private:
    static bool read(void* context, std::uint64_t address,
                     std::span<std::byte> destination) noexcept {
        auto& self = *static_cast<Memory*>(context);
        self.reads.emplace_back(address, destination.size());
        if (self.deferred_host_store({{1, 1, 1}, address, destination.size()})) return false;
        if (self.pending_pool_write && address == 0x200) return false;
        if (address > self.bytes_.size() ||
            destination.size() > self.bytes_.size() - address) return false;
        std::copy_n(self.bytes_.data() + address, destination.size(),
                    destination.data());
        return true;
    }
    static bool write(void*, std::uint64_t,
                      std::span<const std::byte>) noexcept { return false; }

    std::vector<std::byte> bytes_;
    std::unique_ptr<g::GpuAddressSpace> address_space_;
    g::RetainedResource lease_ = std::make_shared<Lease>();
};

class Buffers final : public g::BufferCache {
public:
    struct Acquisition {
        g::BackingRange range;
        g::BufferUse use;
        g::ResourceAccess access;
        std::uint64_t alignment;
        std::uint64_t native_extent = 0;
    };

    Buffers(g::ResourceMemory& memory, std::uint64_t device = 7)
        : domain_{device, &memory} {}

    g::ResidencyDomain domain() const noexcept override { return domain_; }
    g::RendererResult<g::ResidentBuffer> acquire(
        const g::BufferRequest& request) override {
        ++acquires;
        acquired_ranges.push_back(request.range.bytes);
        acquisitions.push_back({request.range.bytes, request.use,
                                request.access, request.alignment, request.native_extent});
        if (stop_uniform && request.use == g::BufferUse::Uniform)
            return g::RendererError{{g::CommandStatus::Stopped,
                                     "synthetic uniform snapshot is not ready"}};
        return g::ResidentBuffer{{request.range.bytes}, {}, lease_, 0,
                                 request.range.bytes.size, {}};
    }
    g::RendererResult<g::ResidentBuffer> upload(
        std::span<const std::byte> bytes, std::uint64_t) override {
        uploads.emplace_back(bytes.begin(), bytes.end());
        return g::ResidentBuffer{{}, {}, lease_, 0, bytes.size(), {}};
    }
    g::RendererStatus invalidate(g::BackingRange) override {
        return g::renderer_success();
    }
    g::RendererStatus resolve(g::BufferKey, g::ResolveReason) override {
        return g::renderer_success();
    }
    g::RendererStatus trim(std::uint64_t) override {
        return g::renderer_success();
    }

    std::size_t acquires = 0;
    bool stop_uniform = false;
    std::vector<g::BackingRange> acquired_ranges;
    std::vector<Acquisition> acquisitions;
    std::vector<std::vector<std::byte>> uploads;

private:
    g::ResidencyDomain domain_;
    g::RetainedResource lease_ = std::make_shared<Lease>();
};

class Textures final : public g::TextureCache {
public:
    Textures(g::ResourceMemory& memory, std::uint64_t device = 7)
        : domain_{device, &memory} {}
    g::ResidencyDomain domain() const noexcept override { return domain_; }
    g::RendererResult<g::ResidentTexture> acquire(
        const g::TextureRequest& request) override {
        ++acquires;
        if (native) return native->acquire(request);
        if (acquires <= fail_acquires)
            return g::RendererError{{g::CommandStatus::Unsupported,
                                     "synthetic texture acquisition failed"}};
        if (cached && *cached == request.key) ++hits;
        cached = request.key;
        views.push_back(request.key);
        return g::ResidentTexture{request.key, {}, lease_, {}};
    }
    g::RendererStatus invalidate(g::BackingRange) override {
        return g::renderer_success();
    }
    g::RendererStatus resolve(const g::TextureViewKey& view, g::ResolveReason reason) override {
        resolves.emplace_back(view.storage.bytes, reason);
        if (fail_resolve)
            return g::RendererError{{g::CommandStatus::MemoryFault,
                                     "synthetic texture resolve failed"}};
        return g::renderer_success();
    }
    g::RendererStatus trim(std::uint64_t) override {
        return g::renderer_success();
    }
    std::size_t acquires = 0;
    std::size_t hits = 0, fail_acquires = 0;
    std::shared_ptr<g::TextureCache> native;
    bool fail_resolve = false;
    std::vector<std::pair<g::BackingRange, g::ResolveReason>> resolves;
    std::vector<g::TextureViewKey> views;
private:
    g::ResidencyDomain domain_;
    g::RetainedResource lease_ = std::make_shared<Lease>();
    std::optional<g::TextureViewKey> cached;
};

class Pipelines final : public m::PipelineStateCache {
public:
    explicit Pipelines(std::uint64_t device = 7) {
        key_.device = device;
    }
    const m::PipelineContextKey& context() const noexcept override { return key_; }
    g::RendererResult<m::ResidentPipeline> acquire(
        const m::PipelineKey&, const m::AdmittedGraphicsStages&) override {
        return g::RendererError{{g::CommandStatus::Unsupported,
                                 "synthetic pipeline acquisition is unexpected"}};
    }
    g::RendererResult<g::RetainedResource> depth_stencil(
        const g::DepthStencilState&) override { return lease_; }
    g::RendererResult<g::RetainedResource> sampler(
        const g::SamplerDescriptor& descriptor) override {
        sampler_keys.push_back(descriptor);
        return lease_;
    }
    g::RendererStatus trim(std::uint64_t) override {
        return g::renderer_success();
    }
    std::vector<g::SamplerDescriptor> sampler_keys;

private:
    m::PipelineContextKey key_{7, "synthetic", "synthetic"};
    g::RetainedResource lease_ = std::make_shared<Lease>();
};

m::AdmittedGraphicsStages stages() {
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>();
    vertex->stage = armigo::shader_input::Stage::Vertex;
    vertex->metallib = {std::byte{1}};
    vertex->entry_point = "vertex";
    vertex->vertex_depth =
        armigo::shader::graphics_msl::VertexDepthConvention::NegativeOneToOne;
    vertex->interface.position_output_mask = 15;

    auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>();
    fragment->stage = armigo::shader_input::Stage::Fragment;
    fragment->metallib = {std::byte{1}};
    fragment->entry_point = "fragment";
    fragment->interface.color_output_mask = 15;
    fragment->color_outputs = {
        {0, 15, armigo::platform::ColorOutputScalar::f32, 4}};
    return {{vertex, fragment}, {}, {}};
}

std::shared_ptr<m::BindingPlanner> planner(
    std::shared_ptr<m::DeviceContext> device = {}) {
    const auto identity = device ? device->identity().device : 7;
    auto made = m::create_binding_planner(
        std::make_shared<Pipelines>(identity), device);
    check(std::holds_alternative<std::shared_ptr<m::BindingPlanner>>(made),
          "planner factory must accept a valid state cache");
    return std::get<std::shared_ptr<m::BindingPlanner>>(std::move(made));
}

g::RenderStop stop_for(const g::DrawPlan& draw) {
    g::RenderStop stop;
    stop.engine = g::engine_class::threed;
    stop.kind = g::RenderKind::Draw;
    stop.draw = draw.source;
    return stop;
}

void vertex_layout_is_canonical() {
    g::DrawPlan draw;
    draw.vertex.attributes = {
        {7, 2, 8, false, {g::VertexAttribFormat::R32Float,
                          g::VertexFormatClass::float_class, 1, 4}},
        {1, 0, 0, false, {g::VertexAttribFormat::R16G16Uint,
                          g::VertexFormatClass::uint_class, 2, 4}},
    };
    draw.vertex.streams = {{2, 0x1200, 0x12ff, 16, 0},
                           {0, 0x1100, 0x11ff, 8, 3}};
    draw.vertex.steps = {{2, g::VertexStep::PerVertex, 1},
                         {0, g::VertexStep::PerInstance, 3}};
    m::PipelineKey key;
    check(std::holds_alternative<std::monostate>(
              m::populate_vertex_layout(key, draw)),
          "valid vertex layout must be admitted");
    check(key.attributes.size() == 2 && key.attributes[0].location == 1 &&
              key.attributes[0].buffer_index == 18 &&
              key.attributes[1].location == 7 &&
              key.attributes[1].buffer_index == 20,
          "attributes must be canonicalized by location");
    check(key.buffers.size() == 2 && key.buffers[0].buffer_index == 18 &&
              key.buffers[0].step == g::VertexStep::PerInstance &&
              key.buffers[0].rate == 3 && key.buffers[1].buffer_index == 20,
          "stream layouts must use native stream indices");

    draw.segments = {{0, 1, 0, 7, 4}};
    check(std::holds_alternative<std::monostate>(
              m::populate_vertex_layout(key, draw)) &&
              key.buffers[0].step == g::VertexStep::PerInstance &&
              key.buffers[0].rate == 1,
          "nonzero base instance with a divisor must select expanded rate-one fetch");

    const auto original = key;
    draw.vertex.attributes[0].stream = 13;
    check(std::holds_alternative<g::RendererError>(
              m::populate_vertex_layout(key, draw)) && key == original,
          "invalid layout must not partially replace a pipeline key");
}

// A stream whose attributes are partly constant fetches the others normally;
// the constant ones read a stride-zero zero stream on a free native slot.
void mixed_constant_stream_moves_constants_to_a_free_slot() {
    g::DrawPlan draw;
    draw.vertex.attributes = {
        {0, 0, 0, false, {g::VertexAttribFormat::R32G32Float,
                          g::VertexFormatClass::float_class, 2, 8}},
        {3, 0, 8, true, {g::VertexAttribFormat::R32Float,
                         g::VertexFormatClass::float_class, 1, 4}},
        {4, 1, 0, false, {g::VertexAttribFormat::R32Float,
                          g::VertexFormatClass::float_class, 1, 4}},
    };
    draw.vertex.streams = {{0, 0x1100, 0x11ff, 16, 0}, {1, 0x1200, 0x12ff, 4, 0}};
    draw.vertex.steps = {{0, g::VertexStep::PerVertex, 1}, {1, g::VertexStep::PerVertex, 1}};
    m::PipelineKey key;
    check(std::holds_alternative<std::monostate>(m::populate_vertex_layout(key, draw)),
          "a stream mixing constant and fetched attributes must be admitted");
    check(key.attributes.size() == 3 && key.attributes[0].buffer_index == 18 &&
              key.attributes[1].location == 3 && key.attributes[1].buffer_index == 20 &&
              key.attributes[2].buffer_index == 19,
          "constant attributes must move to the first free native stream");
    check(key.buffers.size() == 3 && key.buffers[2].buffer_index == 20 &&
              key.buffers[2].step == g::VertexStep::Constant && key.buffers[2].stride == 0,
          "the zero stream must be a constant stride-zero layout");
}

void divided_instance_stream_expands_absolute_indices() {
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->vertex_class_inputs = {
        {0, g::VertexFormatClass::uint_class}};
    vertex->interface.generic_inputs = {{0, 1}};
    admitted.stages.vertex = vertex;

    g::DrawPlan draw;
    draw.source.first = 0;
    draw.source.count = 1;
    draw.source.base_instance = 7;
    draw.source.instance_count = 4;
    draw.vertex.attributes = {
        {0, 0, 0, false, {g::VertexAttribFormat::R32Uint,
                          g::VertexFormatClass::uint_class, 1, 4}}};
    draw.vertex.streams = {{0, 0x1100, 0x11ff, 4, 3}};
    draw.vertex.steps = {{0, g::VertexStep::PerInstance, 3}};
    draw.segments = {{0, 1, 0, 7, 4}};
    memory.word(0x1108, 0x22222222);
    memory.word(0x110c, 0x33333333);
    auto stop = stop_for(draw);
    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) &&
              buffers.acquires == 0 && buffers.uploads.size() == 1 &&
              buffers.uploads[0].size() == 44,
          "divided nonzero-base instance stream must use one immutable expansion");
    const auto value = [&](std::size_t instance) {
        const auto& bytes = buffers.uploads[0];
        const auto offset = instance * 4;
        return std::to_integer<std::uint32_t>(bytes[offset]) |
               (std::to_integer<std::uint32_t>(bytes[offset + 1]) << 8) |
               (std::to_integer<std::uint32_t>(bytes[offset + 2]) << 16) |
               (std::to_integer<std::uint32_t>(bytes[offset + 3]) << 24);
    };
    check(value(7) == 0x22222222 && value(8) == 0x22222222 &&
              value(9) == 0x33333333 && value(10) == 0x33333333,
          "expanded stream must implement floor((baseInstance+local)/rate)");
    m::PipelineKey key;
    check(std::holds_alternative<std::monostate>(
              m::populate_vertex_layout(key, draw)) &&
              key.buffers.size() == 1 && key.buffers[0].rate == 1 &&
              key.buffers[0].stride == 4,
          "expanded instance stream pipeline layout must match its upload");
    stop.draw->instance_index = 1;
    const auto uploads = buffers.uploads.size();
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(error_contains(result, "does not belong") && buffers.uploads.size() == uploads,
          "a plan from another continuation must stop before uploading vertex data");
}

void runtime_windows_zero_fill_and_report_sizes() {
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->constant_buffer_windows = {{0, 8, 0}};
    vertex->constant_buffer_support_index = 1;
    admitted.stages.vertex = vertex;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    stop.constant_buffers[0][0] = {0x1000, 3, true};
    memory.bytes()[0] = std::byte{1};
    memory.bytes()[1] = std::byte{2};
    memory.bytes()[2] = std::byte{3};

    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<m::Bindings>(result),
          "runtime constant-buffer windows must prepare");
    // A word only partly inside the bound size reads zero, so a 3-byte
    // binding uploads a zero window; the size table keeps the exact size.
    check(buffers.uploads.size() == 2 && buffers.uploads[0].size() == 8 &&
              std::ranges::all_of(buffers.uploads[0], [](std::byte b) { return b == std::byte{0}; }) &&
              buffers.uploads[1].size() == 72 &&
              buffers.uploads[1][0] == std::byte{3} &&
              buffers.uploads[1][1] == std::byte{0},
          "window suffix and programmed-size table must be deterministic");

    buffers.uploads.clear();
    stop.constant_buffers[0][0] = {UINT64_MAX, 64, false};
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) &&
              buffers.uploads.size() == 2 &&
              std::all_of(buffers.uploads[0].begin(), buffers.uploads[0].end(),
                          [](std::byte value) { return value == std::byte{0}; }) &&
              buffers.uploads[1][0] == std::byte{0},
          "disabled windows must ignore their address and report zero size");
}

void unchanged_windows_reuse_their_snapshot() {
    Memory memory;
    memory.cpu_tracking = true;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->constant_buffer_windows = {{0, 8, 0}};
    vertex->constant_buffer_support_index = 1;
    admitted.stages.vertex = vertex;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    stop.constant_buffers[0][0] = {0x1000, 4, true};
    memory.bytes()[0] = std::byte{1};
    auto shared = planner();
    // An unchanged window binds the slot's previous upload, so only the
    // support table uploads; its bytes are then the previous window's.
    std::byte window_byte{};
    const auto prepare = [&](m::BindingPlanner& target) {
        buffers.uploads.clear();
        memory.reads.clear();
        auto result = target.prepare(stop, admitted, draw, memory, buffers, textures);
        check(std::holds_alternative<m::Bindings>(result) &&
                  (buffers.uploads.size() == 2 || buffers.uploads.size() == 1),
              "tracked constant-buffer windows must prepare");
        if (buffers.uploads.size() == 2 && buffers.uploads[0].size() == 8) window_byte = buffers.uploads[0][0];
        return window_byte;
    };
    check(prepare(*shared) == std::byte{1} && !memory.reads.empty(),
          "the first window preparation must snapshot guest bytes");
    // The synthetic tracker reports no store, so the snapshot stays valid.
    memory.bytes()[0] = std::byte{2};
    check(prepare(*shared) == std::byte{1} && memory.reads.empty(),
          "an unchanged window must upload its previous snapshot without reading guest bytes");
    memory.cpu_stored = true;
    check(prepare(*shared) == std::byte{2} && !memory.reads.empty(),
          "a CPU store after the stamp must snapshot the window again");
    memory.cpu_stored = false;
    memory.bytes()[0] = std::byte{3};
    memory.version.cpu_write = 7;
    check(prepare(*shared) == std::byte{3} && !memory.reads.empty(),
          "a registry version change must snapshot the window again");
    memory.bytes()[0] = std::byte{4};
    memory.cpu_tracking = false;
    check(prepare(*shared) == std::byte{4} && !memory.reads.empty(),
          "untracked memory must snapshot every window");
    memory.cpu_tracking = true;
    memory.bytes()[0] = std::byte{5};
    check(prepare(*planner()) == std::byte{5} && !memory.reads.empty(),
          "a fresh planner must snapshot its windows");
}

void runtime_windows_preserve_fragments_and_keep_descriptors_strict() {
    namespace p = armigo::platform;
    Memory memory;
    const std::array mappings{
        g::Mapping{0x1000, 0x100, 0, 0, 0x40, 1},
        g::Mapping{0x1100, 0x100, 0, 0x200, 0x100, 1}};
    memory.remap(mappings);
    std::fill_n(memory.bytes().begin(), 0x40, std::byte{0x11});
    std::fill_n(memory.bytes().begin() + 0x200, 0x100, std::byte{0x22});
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<p::CompiledGraphicsStage>(*admitted.stages.vertex);
    vertex->constant_buffer_windows = {{0, 0x120, 0}};
    vertex->constant_buffer_support_index = 17;
    admitted.stages.vertex = vertex;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    stop.constant_buffers[0][0] = {0x1000, 0x120, true};
    auto result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) && buffers.uploads.size() == 2,
          "constant-buffer windows must stage discontiguous backed fragments");
    const auto& bytes = buffers.uploads[0];
    check(bytes.size() == 0x120 && buffers.uploads[1][0] == std::byte{0x20} &&
              buffers.uploads[1][1] == std::byte{1},
          "the reflected window and exact programmed support size must remain unchanged");
    for (std::size_t i = 0; i < bytes.size(); ++i)
        check(bytes[i] == (i < 0x40 ? std::byte{0x11} :
              i < 0x100 ? std::byte{0} : std::byte{0x22}),
              "backed fragments must retain their virtual offsets around zero-filled padding");

    for (const auto offset : {0x38u, 0x40u}) {
        vertex->storage_buffers = {{0, offset, 7, p::CompiledShaderResourceAccess::Read, 4}};
        buffers.uploads.clear();
        result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
        check(std::holds_alternative<g::RendererError>(result) &&
                  std::get<g::RendererError>(result).command().status == g::CommandStatus::MemoryFault &&
                  buffers.uploads.empty(),
              "a descriptor crossing or inside padding must fail before any upload");
    }

    vertex->storage_buffers = {{0, 0x10, 7, p::CompiledShaderResourceAccess::Read, 4}};
    memory.word(0x1010, 0x1100);
    memory.word(0x1014, 0);
    memory.word(0x1018, 16);
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) &&
              buffers.uploads[0][0x10] == std::byte{0} &&
              buffers.uploads[0][0x11] == std::byte{0x11} &&
              buffers.acquisitions.size() == 1 && buffers.acquisitions[0].range.size == 16,
          "a backed descriptor must decode the exact immutable uploaded words");

    buffers.acquisitions.clear();
    memory.word(0x1010, 0x103c);
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    // A range reaching into mapping padding binds its backed prefix with a
    // native tail up to the descriptor size; the tail is not guest memory.
    check(std::holds_alternative<m::Bindings>(result) && buffers.acquisitions.size() == 1 &&
              buffers.acquisitions[0].range.size == 4 &&
              buffers.acquisitions[0].native_extent == 16,
          "storage access crossing into mapping padding must bind its backed prefix");
    buffers.acquisitions.clear();
    memory.word(0x1010, 0x1040);
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(std::holds_alternative<g::RendererError>(result) && buffers.acquisitions.empty(),
          "storage access starting in mapping padding must still be rejected");
}

// A window over GPU-written constants binds the resident image with a zero
// tail to the window size; CPU-backed constants keep the immutable snapshot.
void gpu_written_window_binds_resident_image() {
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>(*admitted.stages.vertex);
    vertex->constant_buffers.clear();
    vertex->constant_buffer_windows = {{0, 0x20, 2}};
    vertex->constant_buffer_support_index = 17;
    admitted.stages.vertex = vertex;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    stop.constant_buffers[0][0] = {0x1000, 0x10, true};

    memory.gpu_dirty = true;
    auto result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    // Only the programmed-size support table is uploaded.
    check(std::holds_alternative<m::Bindings>(result) && buffers.uploads.size() == 1 &&
              buffers.uploads[0].size() == 72 && buffers.acquisitions.size() == 1 &&
              buffers.acquisitions[0].use == g::BufferUse::Uniform &&
              buffers.acquisitions[0].range.size == 0x10 &&
              buffers.acquisitions[0].native_extent == 0x20,
          "a GPU-written window must bind its resident image");

    memory.gpu_dirty = false;
    buffers.acquisitions.clear();
    buffers.uploads.clear();
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) && buffers.uploads.size() == 2 &&
              buffers.uploads[0].size() == 0x20 && buffers.acquisitions.empty(),
          "a CPU-backed window must keep its immutable snapshot");
}

void legacy_uniform_snapshot_stops_without_substitution() {
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->constant_buffers = {{0, 4, 0}};
    admitted.stages.vertex = vertex;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    stop.constant_buffers[0][0] = {0x1000, 4, true};

    buffers.stop_uniform = true;
    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<g::RendererError>(result) &&
              std::get<g::RendererError>(result).command().status ==
                  g::CommandStatus::Stopped &&
              buffers.acquires == 1 && buffers.uploads.empty(),
          "planner must propagate a refused immutable uniform snapshot");
}

void configure_texture(Memory& memory, g::RenderStop& stop, unsigned stage_slot = 4) {
    stop.registers[0x982] = 4;
    stop.registers[0x55e] = 0x1200;
    stop.registers[0x558] = 0x1300;
    stop.constant_buffers[stage_slot][4] = {0x1100, 36, true};
    memory.word(0x1120, 0);

    std::uint32_t word0 = 9;
    for (unsigned component = 0; component != 4; ++component) {
        word0 |= 2u << (7 + component * 3);
        word0 |= (component + 2u) << (19 + component * 3);
    }
    memory.word(0x1200, word0);
    memory.word(0x1204, 0x1400);
    memory.word(0x1208, 3u << 21);
    memory.word(0x1210, 1u | (1u << 23));
    memory.word(0x1214, 1u | (1u << 31));
    memory.word(0x1300, 2u | (2u << 3) | (2u << 6));
    memory.word(0x1304, 2u | (2u << 4) | (3u << 6));
}

void decoded_pool_entries_are_reused() {
    Memory memory;
    memory.cpu_tracking = true;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    configure_texture(memory, stop);
    g::TextureDescriptorCache descriptors;
    const auto resolve = [&] {
        memory.reads.clear();
        auto result = g::resolve_texture_binding_descriptors(stop, 4, 32,
            memory, true, 4, &descriptors);
        check(std::holds_alternative<g::RetainedTextureBinding>(result),
              "cached descriptor resolution must succeed");
        return result;
    };
    resolve();
    check(memory.reads.size() == 3, "first resolution must read handle, TIC and TSC");
    resolve();
    check(memory.reads.size() == 1, "clean pool entries must skip both descriptor reads");
    memory.reads.clear();
    auto sampler = g::resolve_sampler_binding(stop, 4, 4, 32, memory, descriptors);
    check(std::holds_alternative<g::SamplerDescriptor>(sampler) && memory.reads.size() == 1,
          "independent sampler resolution must share the decoded TSC cache");

    memory.cpu_stored = true;
    memory.word(0x1204, 0x1500);
    memory.word(0x1304, 1u | (2u << 4) | (3u << 6));
    auto changed = resolve();
    check(memory.reads.size() == 3 &&
              std::get<g::RetainedTextureBinding>(changed).binding.texture.address == 0x1500 &&
              std::get<g::RetainedTextureBinding>(changed).binding.sampler.magnification ==
                  g::SamplerFilter::Nearest,
          "CPU writes to descriptor pools must reread and decode the changed values");
    memory.cpu_stored = false;
    resolve();
    resolve();
    check(memory.reads.size() == 1, "clean descriptors must become reusable again");

    ++memory.version.gpu_write;
    memory.word(0x1204, 0x1600);
    changed = resolve();
    check(memory.reads.size() == 3 &&
              std::get<g::RetainedTextureBinding>(changed).binding.texture.address == 0x1600,
          "a GPU-write version change must invalidate decoded pools even without CPU stores");
    memory.gpu_dirty = true;
    resolve();
    check(memory.reads.size() == 3, "GPU-dirty descriptors must use coherent reads");
    memory.gpu_dirty = false;
    resolve();
    resolve();
    check(memory.reads.size() == 1, "resolved pool entries must become reusable again");

    const g::Mapping mapping{0x1000, memory.bytes().size(), 0, 0, memory.bytes().size(), 1};
    memory.remap(std::span{&mapping, 1});
    resolve();
    check(memory.reads.size() == 3, "a new mapping generation must reread both pools");
    memory.pending_pool_write = true;
    auto blocked = g::resolve_texture_binding_descriptors(stop, 4, 32,
        memory, true, 4, &descriptors);
    check(std::holds_alternative<g::TextureError>(blocked) &&
              std::get<g::TextureError>(blocked).code == g::TextureErrorCode::MemoryRead,
          "pending pool writers must preserve the slow read rejection on a cache hit");
    memory.pending_pool_write = false;
    memory.cpu_tracking = false;
    resolve();
    resolve();
    check(memory.reads.size() == 3, "untracked descriptor pools must never reuse decoded entries");
}

void configure_custom_sampler(
    Memory& memory, const std::array<float, 4>& color,
    bool srgb_conversion = false,
    std::array<std::uint8_t, 3> packed_srgb = {}) {
    memory.word(0x1300, 3u | (3u << 3) | (3u << 6) |
                              (std::uint32_t{srgb_conversion} << 13));
    memory.word(0x1304, 2u | (2u << 4) | (3u << 6));
    memory.word(0x1308, std::uint32_t{packed_srgb[0]} << 24);
    memory.word(0x130c, (std::uint32_t{packed_srgb[1]} << 12) |
                              (std::uint32_t{packed_srgb[2]} << 20));
    for (unsigned component = 0; component < color.size(); ++component)
        memory.word(0x1310 + component * 4,
                    std::bit_cast<std::uint32_t>(color[component]));
}

void configure_format_and_swizzle(
    Memory& memory, std::uint32_t format,
    const std::array<std::uint32_t, 4>& swizzle, bool srgb) {
    std::uint32_t word = format;
    for (unsigned component = 0; component < 4; ++component) {
        word |= 2u << (7 + component * 3);
        word |= swizzle[component] << (19 + component * 3);
    }
    memory.word(0x1200, word);
    memory.word(0x1210, 1u | (std::uint32_t{srgb} << 22) | (1u << 23));
}

std::vector<armigo::platform::CompiledShaderResource>
custom_border_resources() {
    namespace p = armigo::platform;
    p::CompiledShaderResource texture;
    texture.resource_id = 10;
    texture.array_length = 1;
    texture.cbuf_slot = 4;
    texture.handle_byte_offset = 32;
    texture.kind = p::CompiledShaderResourceKind::SampledTexture;
    texture.texture_descriptor = p::CompiledShaderTextureDescriptor{};
    texture.texture_descriptor->component_format =
        p::TextureComponentFormat::unorm;
    texture.location.resource_namespace =
        p::CompiledShaderResourceNamespace::Texture;
    texture.location.first = 3;
    texture.location.count = 1;

    p::CompiledShaderResource sampler;
    sampler.resource_id = 11;
    sampler.array_length = 1;
    sampler.cbuf_slot = 4;
    sampler.handle_byte_offset = 32;
    sampler.kind = p::CompiledShaderResourceKind::Sampler;
    sampler.location.resource_namespace =
        p::CompiledShaderResourceNamespace::Sampler;
    sampler.location.first = 5;
    sampler.location.count = 1;
    return {texture, sampler};
}

std::shared_ptr<armigo::platform::CompiledGraphicsStage>
custom_border_fragment(m::AdmittedGraphicsStages& admitted, bool support) {
    namespace p = armigo::platform;
    auto fragment = std::make_shared<p::CompiledGraphicsStage>(
        *admitted.stages.fragment);
    fragment->fragment_position =
        armigo::shader::graphics_msl::FragmentPositionConvention::
            MetalPositionWMatchesGuestPositionW;
    fragment->resources = custom_border_resources();
    if (support)
        fragment->custom_border =
            p::CompiledCustomBorderSupport{21, {{10, 11}}};
    admitted.stages.fragment = fragment;
    return fragment;
}

std::shared_ptr<armigo::platform::CompiledGraphicsStage>
custom_border_vertex(m::AdmittedGraphicsStages& admitted, bool support) {
    namespace p = armigo::platform;
    auto vertex = std::make_shared<p::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->resources = custom_border_resources();
    if (support)
        vertex->custom_border =
            p::CompiledCustomBorderSupport{21, {{10, 11}}};
    admitted.stages.vertex = vertex;
    return vertex;
}

struct alignas(16) CustomBorderRecord {
    std::uint64_t coverage = 0;
    std::uint64_t reserved = 0;
    std::array<float, 4> color{};
    std::array<std::uint32_t, 4> swizzle{};
    std::uint32_t enabled = 0;
    std::array<std::uint32_t, 3> padding{};
};
static_assert(sizeof(CustomBorderRecord) == 64);

CustomBorderRecord custom_border_record(const Buffers& buffers) {
    check(buffers.uploads.size() == 1 && buffers.uploads[0].size() == 64,
          "custom border support must upload exactly one 64-byte record");
    CustomBorderRecord record;
    std::memcpy(&record, buffers.uploads[0].data(), sizeof(record));
    return record;
}

void custom_border_binding_profiles() {
    namespace p = armigo::platform;

    // Legacy paired reflection has no coverage binding and must stop rather
    // than silently approximating an active arbitrary border.
    {
        Memory memory;
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        auto fragment = std::make_shared<p::CompiledGraphicsStage>(
            *admitted.stages.fragment);
        fragment->fragment_position =
            armigo::shader::graphics_msl::FragmentPositionConvention::
                MetalPositionWMatchesGuestPositionW;
        fragment->textures = {{32, 0, 0}};
        fragment->texture_handle_cbuf = 4;
        fragment->texture_facts = {{4, 32, true, false}};
        admitted.stages.fragment = fragment;
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        configure_custom_sampler(memory, {0.25f, 0.5f, 0.75f, 1.0f});
        const auto result = planner()->prepare(
            stop, admitted, draw, memory, buffers, textures);
        check(error_contains(result, "legacy shader sampling has no custom-border"),
              "legacy reflection must reject an active custom border");
    }

    // Independent v3 resources still require an explicit coverage-v1 pair.
    {
        Memory memory;
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        custom_border_fragment(admitted, false);
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        configure_custom_sampler(memory, {0.25f, 0.5f, 0.75f, 1.0f});
        const auto result = planner()->prepare(
            stop, admitted, draw, memory, buffers, textures);
        check(error_contains(result, "custom border requires a coverage-v1"),
              "v3 reflection without custom-border support must stop");
    }

    // A vertex stage sampling with an active custom border is held to the
    // same coverage-v1 requirement.
    {
        Memory memory;
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        custom_border_vertex(admitted, false);
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop, 0);
        configure_custom_sampler(memory, {0.25f, 0.5f, 0.75f, 1.0f});
        const auto result = planner()->prepare(
            stop, admitted, draw, memory, buffers, textures);
        check(error_contains(result, "custom border requires a coverage-v1"),
              "vertex sampling without custom-border support must stop");
    }
}

void custom_border_records_and_views() {
    namespace p = armigo::platform;
    const auto made_device = m::create_device_context();
    check(std::holds_alternative<std::shared_ptr<m::DeviceContext>>(made_device),
          "custom border binding tests require a Metal device context");
    const auto device = std::get<std::shared_ptr<m::DeviceContext>>(made_device);

    // One-component storage exercises missing-channel masking while a guest
    // swizzle proves that the acquired Metal view is identity-swizzled.
    {
        Memory memory;
        Buffers buffers(memory, device->identity().device);
        Textures textures(memory, device->identity().device);
        auto admitted = stages();
        custom_border_fragment(admitted, true);
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        configure_format_and_swizzle(memory, 0x1d, {4, 0, 2, 7}, false);
        configure_custom_sampler(memory, {0.25f, 0.5f, 0.75f, 1.0f});
        auto pipelines = std::make_shared<Pipelines>(device->identity().device);
        const auto made_planner = m::create_binding_planner(pipelines, device);
        check(std::holds_alternative<std::shared_ptr<m::BindingPlanner>>(
                  made_planner),
              "device-backed custom border planner must be created");
        auto binding_planner =
            std::get<std::shared_ptr<m::BindingPlanner>>(made_planner);
        const auto result = binding_planner->prepare(
            stop, admitted, draw, memory, buffers, textures);
        if (const auto* error = std::get_if<g::RendererError>(&result))
            std::cerr << "Custom border binding: "
                      << error->command().detail << '\n';
        check(std::holds_alternative<m::Bindings>(result),
              "valid custom-border reflection must prepare bindings");
        if (!std::holds_alternative<m::Bindings>(result)) return;
        const auto& bindings = std::get<m::Bindings>(result);
        const auto record = custom_border_record(buffers);
        check(record.coverage != 0 && record.reserved == 0 &&
                  record.enabled == 1 && record.color[0] == 0.25f &&
                  record.color[1] == 0.0f && record.color[2] == 0.0f &&
                  record.color[3] == 0.0f &&
                  record.swizzle == std::array<std::uint32_t, 4>{2, 4, 0, 5} &&
                  std::ranges::all_of(record.padding,
                                      [](std::uint32_t value) { return value == 0; }),
              "custom border record must carry coverage, channel mask and guest swizzle");
        check(textures.views.size() == 1 &&
                  textures.views[0].swizzle ==
                      std::array{g::TextureSwizzle::Red,
                                 g::TextureSwizzle::Green,
                                 g::TextureSwizzle::Blue,
                                 g::TextureSwizzle::Alpha} &&
                  textures.views[0].storage.descriptor.swizzle ==
                      std::array{g::TextureSwizzle::Blue,
                                 g::TextureSwizzle::Zero,
                                 g::TextureSwizzle::Red,
                                 g::TextureSwizzle::OneFloat},
              "custom-border sampling must retain the guest swizzle but acquire an identity view");
        check(bindings.custom_border_textures.size() == 1 &&
                  std::ranges::any_of(bindings.buffers, [](const auto& binding) {
                      return binding.index == 21 &&
                             binding.use == g::BufferUse::Uniform &&
                             binding.access == g::ResourceAccess::Read &&
                             binding.buffer.key.bytes.backing.owner == 0;
                  }) &&
                  pipelines->sampler_keys.size() == 1 &&
                  !pipelines->sampler_keys[0].srgb_border_conversion,
              "coverage and normalized native sampler state must remain retained");
    }

    // Active sRGB conversion must upload packed-byte EOTF RGB, not float RGB.
    {
        Memory memory;
        Buffers buffers(memory, device->identity().device);
        Textures textures(memory, device->identity().device);
        auto admitted = stages();
        custom_border_fragment(admitted, true);
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        configure_format_and_swizzle(memory, 0x08, {2, 3, 4, 5}, true);
        configure_custom_sampler(memory, {0.9f, 0.8f, 0.7f, 0.25f}, true,
                                 {10, 128, 255});
        auto pipelines = std::make_shared<Pipelines>(device->identity().device);
        const auto made_planner = m::create_binding_planner(pipelines, device);
        auto binding_planner =
            std::get<std::shared_ptr<m::BindingPlanner>>(made_planner);
        const auto result = binding_planner->prepare(
            stop, admitted, draw, memory, buffers, textures);
        if (const auto* error = std::get_if<g::RendererError>(&result))
            std::cerr << "sRGB custom border binding: "
                      << error->command().detail << '\n';
        check(std::holds_alternative<m::Bindings>(result),
              "sRGB custom-border reflection must prepare bindings");
        if (!std::holds_alternative<m::Bindings>(result)) return;
        const auto record = custom_border_record(buffers);
        check(record.enabled == 1 &&
                  std::abs(record.color[0] - 0.00303527f) < 0.000001f &&
                  std::abs(record.color[1] - 0.2158605f) < 0.000001f &&
                  record.color[2] == 1.0f && record.color[3] == 0.25f &&
                  pipelines->sampler_keys.size() == 1 &&
                  pipelines->sampler_keys[0].border_color == record.color &&
                  !pipelines->sampler_keys[0].srgb_border_conversion,
              "binding record and native sampler must use the effective sRGB border");
    }

    // A vertex-stage coverage pair uploads its record and binds the support
    // buffer to the vertex stage.
    {
        Memory memory;
        Buffers buffers(memory, device->identity().device);
        Textures textures(memory, device->identity().device);
        auto admitted = stages();
        custom_border_vertex(admitted, true);
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop, 0);
        configure_custom_sampler(memory, {0.25f, 0.5f, 0.75f, 1.0f});
        auto pipelines = std::make_shared<Pipelines>(device->identity().device);
        const auto made_planner = m::create_binding_planner(pipelines, device);
        auto binding_planner =
            std::get<std::shared_ptr<m::BindingPlanner>>(made_planner);
        const auto result = binding_planner->prepare(
            stop, admitted, draw, memory, buffers, textures);
        if (const auto* error = std::get_if<g::RendererError>(&result))
            std::cerr << "Vertex custom border binding: "
                      << error->command().detail << '\n';
        check(std::holds_alternative<m::Bindings>(result),
              "vertex custom-border reflection must prepare bindings");
        if (!std::holds_alternative<m::Bindings>(result)) return;
        const auto record = custom_border_record(buffers);
        check(record.coverage != 0 && record.enabled == 1 &&
                  record.color[0] == 0.25f && record.color[3] == 1.0f,
              "vertex custom border must upload its enabled coverage record");
        const auto& bindings = std::get<m::Bindings>(result);
        check(std::ranges::any_of(bindings.buffers, [](const auto& binding) {
                  return binding.stage == armigo::shader_input::Stage::Vertex &&
                         binding.index == 21 &&
                         binding.use == g::BufferUse::Uniform &&
                         binding.access == g::ResourceAccess::Read;
              }) && bindings.custom_border_textures.size() == 1,
              "vertex custom-border support must bind to the vertex stage with resident coverage");
    }
}

void custom_border_reflection_rejections() {
    namespace p = armigo::platform;
    const auto prepare = [](auto mutate, bool anisotropic = false) {
        Memory memory;
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        auto fragment = custom_border_fragment(admitted, true);
        mutate(*fragment);
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        configure_custom_sampler(memory, {0.25f, 0.5f, 0.75f, 1.0f});
        if (anisotropic)
            memory.word(0x1300, 3u | (3u << 3) | (3u << 6) | (1u << 20));
        return planner()->prepare(stop, admitted, draw, memory,
                                  buffers, textures);
    };

    auto result = prepare([](p::CompiledGraphicsStage& stage) {
        stage.constant_buffer_windows = {{0, 4, 21}};
        stage.constant_buffer_support_index = 22;
    });
    check(error_contains(result, "aliases a buffer"),
          "custom-border support buffer collisions must reject");

    result = prepare([](p::CompiledGraphicsStage& stage) {
        stage.custom_border->pairs[0].sampler_resource = 99;
    });
    check(error_contains(result, "custom border support reflection is invalid"),
          "custom-border pairs must name existing resources");

    result = prepare([](p::CompiledGraphicsStage& stage) {
        stage.resources[1].comparison = true;
    });
    check(error_contains(result, "custom border support reflection is invalid"),
          "comparison samplers must not enter custom-border support");

    result = prepare([](p::CompiledGraphicsStage& stage) {
        stage.resources[0].texture_descriptor->array = true;
    });
    check(error_contains(result, "custom border support reflection is invalid"),
          "array textures must not enter custom-border support");

    result = prepare([](p::CompiledGraphicsStage&) {}, true);
    check(error_contains(result, "sampler descriptor field is outside"),
          "anisotropic custom-border sampling must reject before allocation");
}

void descriptor_only_planner_reads_and_retry() {
    namespace gm = armigo::shader::graphics_msl;
    struct Scenario {
        bool sampler, dirty;
        std::size_t failures;
        bool resolve_failure;
        std::size_t expected_acquires, expected_resolves;
        bool success;
    };
    const Scenario scenarios[]{
        {true, false, 0, false, 1, 0, true},
        {false, false, 0, false, 1, 0, true},
        {true, true, 0, false, 1, 0, true},
        {true, true, 1, false, 2, 1, true},
        {true, true, 2, false, 2, 1, false},
        {true, false, 1, false, 1, 0, false},
        {true, true, 1, true, 1, 1, false},
    };
    for (const auto scenario : scenarios) {
        Memory memory;
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(
            *admitted.stages.fragment);
        fragment->fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        fragment->textures = {{32, 0, scenario.sampler ? std::uint8_t{0} : gm::TextureBinding::no_sampler}};
        fragment->texture_handle_cbuf = 4;
        fragment->texture_facts = {{4, 32, true, false}};
        admitted.stages.fragment = fragment;
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        memory.gpu_dirty = scenario.dirty;
        textures.fail_acquires = scenario.failures;
        textures.fail_resolve = scenario.resolve_failure;
        auto binding_planner = planner();
        auto result = binding_planner->prepare(stop, admitted, draw, memory, buffers, textures);
        check(std::holds_alternative<m::Bindings>(result) == scenario.success,
              "descriptor-only acquire/resolve outcome must match the bounded retry policy");
        check(textures.acquires == scenario.expected_acquires &&
                  textures.resolves.size() == scenario.expected_resolves,
              "only failed GPU-dirty acquisition may resolve and retry, at most once");
        std::vector<std::pair<std::uint64_t, std::size_t>> expected{{0x120, 4}, {0x200, 32}};
        if (scenario.sampler) expected.emplace_back(0x300, 32);
        check(memory.reads == expected && memory.retains == 1,
              "persistent planner must read exactly 36/68 descriptor bytes and retain once without payload reads");
        check(memory.inspections == (scenario.failures ? 1u : 0u),
              "successful resident acquisition must not inspect or resolve authority");
        if (!textures.resolves.empty())
            check(textures.resolves[0].first.offset == 0x400 &&
                      textures.resolves[0].first.size != 0 &&
                      textures.resolves[0].second == g::ResolveReason::IncompatibleAlias,
                  "retry must explicitly resolve the exact texture backing as an incompatible alias");
        if (scenario.resolve_failure)
            check(std::get<g::RendererError>(result).command().detail ==
                      "synthetic texture resolve failed",
                  "resolve failures must propagate without a retry");
        if (!scenario.failures) {
            memory.reads.clear();
            result = binding_planner->prepare(stop, admitted, draw, memory, buffers, textures);
            check(std::holds_alternative<m::Bindings>(result) && memory.reads == expected &&
                      textures.hits == 1 && textures.resolves.empty(),
                  "resident hits must reread live descriptors without sweeping payload");
            memory.missing_lease = true;
            result = binding_planner->prepare(stop, admitted, draw, memory, buffers, textures);
            check(std::holds_alternative<g::RendererError>(result) && textures.acquires == 2,
                  "missing backing lease must reject before cache acquisition");
        }
    }
}

void unchanged_stage_textures_are_reused() {
    namespace dirty = g::threed_dirty;
    Memory memory;
    memory.cpu_tracking = true;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.fragment);
    fragment->fragment_position =
        armigo::shader::graphics_msl::FragmentPositionConvention::
            MetalPositionWMatchesGuestPositionW;
    fragment->textures = {{32, 0, 0}};
    fragment->texture_handle_cbuf = 4;
    fragment->texture_facts = {{4, 32, true, false}};
    admitted.stages.fragment = fragment;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    configure_texture(memory, stop);
    auto shared = planner();
    m::BindingReuse reuse{0, 5, {}};
    const auto prepare = [&](std::size_t acquires, const char* message) {
        auto result = shared->prepare(stop, admitted, draw, memory, buffers, textures, reuse);
        check(std::holds_alternative<m::Bindings>(result) && textures.acquires == acquires &&
                  std::get<m::Bindings>(result).textures.size() == 1 &&
                  std::get<m::Bindings>(result).samplers.empty() &&
                  std::get<m::Bindings>(result).accesses.size() == 1,
              message);
        return result;
    };
    const auto first = prepare(1, "the first draw must acquire its stage textures");
    prepare(1, "a clean stage in the same epoch must reuse its texture bindings");
    memory.cpu_stored = true;
    prepare(2, "a CPU store to a read range must rebuild the stage textures");
    memory.cpu_stored = false;
    reuse.dirty = dirty::bit(dirty::ConstantBuffers0);
    prepare(2, "another stage's constant buffers must not rebuild this stage");
    reuse.dirty = dirty::bit(dirty::ConstantBuffers0 + 4);
    prepare(2, "binds of other constant-buffer slots must keep this stage's textures");
    stop.constant_buffers[4][4].size = 40;
    prepare(3, "a changed handle slot binding must rebuild the stage textures");
    // The front end marks TexturePools only when a pool register changes.
    reuse.dirty = dirty::bit(dirty::TexturePools);
    ++stop.registers[0x55f];
    memory.reads.clear();
    prepare(4, "texture pool changes must rebuild the stage textures");
    const auto pool_was_read = [&] {
        return std::ranges::any_of(memory.reads, [](const auto& read) {
            return read.second == 32 && (read.first == 0x200 || read.first == 0x300);
        });
    };
    check(!pool_was_read(), "a stage miss must reuse clean decoded pool entries");
    reuse.dirty = 0;
    reuse.epoch = 6;
    memory.reads.clear();
    prepare(5, "a new epoch must rebuild the stage textures");
    check(!pool_was_read(), "a new stage epoch must keep clean decoded pool entries");
    static_cast<void>(first);
    reuse.epoch = 0;
    prepare(6, "epoch zero must disable reuse");
    prepare(7, "epoch zero must disable reuse");
}

void resident_attachment_writes_preserve_texture_bindings() {
    const auto made_device = m::create_device_context();
    check(std::holds_alternative<std::shared_ptr<m::DeviceContext>>(made_device),
          "resident binding tests require a Metal device");
    const auto device = std::get<std::shared_ptr<m::DeviceContext>>(made_device);
    struct Actions final : g::CompletionActions {
        g::RendererStatus make_coherent() override { return g::renderer_success(); }
        g::RendererStatus prepare_publication() override { return g::renderer_success(); }
        void publish() noexcept override {}
        g::RendererStatus preserve_failed_authority() override { return g::renderer_success(); }
    };
    for (const bool per_binding : {false, true}) {
        auto memory = std::make_shared<Memory>();
        memory->cpu_tracking = memory->host_peeks = memory->native_storage = true;
        Buffers buffers(*memory, device->identity().device);
        Textures textures(*memory, device->identity().device);
        auto made_cache = m::create_texture_cache(device, memory, 1024 * 1024);
        check(std::holds_alternative<std::shared_ptr<g::TextureCache>>(made_cache),
              "resident binding tests require the native texture cache");
        textures.native = std::get<std::shared_ptr<g::TextureCache>>(made_cache);
        auto admitted = stages();
        auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(*admitted.stages.fragment);
        fragment->fragment_position = armigo::shader::graphics_msl::FragmentPositionConvention::
            MetalPositionWMatchesGuestPositionW;
        fragment->textures = {{32, 0, 0}};
        fragment->texture_handle_cbuf = 4;
        fragment->texture_facts = {{4, 32, true, false}};
        admitted.stages.fragment = fragment;
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(*memory, stop);
        auto shared = planner(device);
        m::EpochLog log;
        m::BindingReuse reuse{0, 5, {}, &log};
        const auto prepare = [&](std::size_t acquires) {
            if (per_binding) ++stop.constant_buffers[4][4].size;
            auto result = shared->prepare(stop, admitted, draw, *memory, buffers, textures, reuse);
            if (const auto* error = std::get_if<g::RendererError>(&result))
                std::cerr << error->command().detail << '\n';
            check(std::holds_alternative<m::Bindings>(result) && textures.acquires == acquires,
                  "resident replay must skip acquire in both stage and per-binding caches");
            return std::get<m::Bindings>(std::move(result));
        };
        const auto first = prepare(1);
        const auto sampled = first.textures.front().texture;
        const auto range = first.accesses.front().range;
        auto acquired = textures.native->acquire({range, sampled.key, g::ResourceAccess::ReadWrite});
        check(std::holds_alternative<g::ResidentTexture>(acquired), "attachment acquisition must succeed");
        const auto target = std::get<g::ResidentTexture>(std::move(acquired));
        const auto identity = m::resident_texture_identity(target);
        check(!identity.allocation.expired() && identity == m::resident_texture_identity(sampled) &&
                  target.storage != sampled.storage,
              "distinct native view leases must identify the same allocation");
        // Render real GPU content into the allocation. Keep the registry writer
        // logically pending below to test the planner's dependency contract.
        const auto native = std::dynamic_pointer_cast<const m::TextureResource>(target.storage);
        id<MTLCommandQueue> queue = [device->device() newCommandQueue];
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        auto* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = native->texture();
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
        [[commands renderCommandEncoderWithDescriptor:pass] endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        check(commands.status == MTLCommandBufferStatusCompleted, "synthetic attachment clear must complete");

        for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
            g::SubmissionCompletion writer({91, sequence}, {target.storage}, std::make_shared<Actions>());
            if (sequence == 2)
                check(std::holds_alternative<std::monostate>(writer.submitted()), "writer must enter Submitted");
            memory->gpu_dirty = true;
            memory->storage = target.storage;
            memory->writer = writer.token();
            ++memory->version.gpu_write;
            log.write(sampled.key.storage.descriptor.address, range.bytes.size, reuse.epoch, identity);
            ++reuse.epoch;
            const auto reused = prepare(1);
            check(reused.textures.front().texture.storage == sampled.storage &&
                      reused.accesses.size() == 1 && reused.accesses.front().range.bytes == range.bytes &&
                      reused.accesses.front().access == g::ResourceAccess::Read,
                  "resident replay must keep the original view and hazard read");
            check(std::ranges::any_of(reused.dependencies, [&](const auto& token) {
                      return token.id() == writer.token().id();
                  }), "resident replay must carry the latest Accepted or Submitted writer");
            const auto repeated = prepare(1);
            check(std::ranges::any_of(repeated.dependencies, [&](const auto& token) {
                      return token.id() == writer.token().id();
                  }), "a subsequent same-epoch hit must retain the refreshed writer dependency");
            if (sequence == 1)
                check(std::holds_alternative<std::monostate>(writer.submitted()), "writer must submit before retirement");
            check(std::holds_alternative<std::monostate>(writer.feedback({})) &&
                      std::holds_alternative<std::monostate>(writer.finish()),
                  "synthetic writer must retire after its dependency checks");
        }
        // A different format over identical bytes acquires another allocation;
        // it cannot inherit the same-resident exemption.
        auto alias_key = sampled.key;
        alias_key.storage.descriptor.format = g::TextureFormat::R32Float;
        acquired = textures.native->acquire({range, alias_key, g::ResourceAccess::ReadWrite});
        check(std::holds_alternative<g::ResidentTexture>(acquired), "alias attachment acquisition must succeed");
        const auto alias = std::get<g::ResidentTexture>(std::move(acquired));
        check(m::resident_texture_identity(alias) != identity, "format alias must have another identity");
        memory->storage = alias.storage;
        ++memory->version.gpu_write;
        log.write(sampled.key.storage.descriptor.address, range.bytes.size, reuse.epoch,
                  m::resident_texture_identity(alias));
        ++reuse.epoch;
        prepare(2);
        memory->gpu_dirty = false;
        memory->storage.reset();
        memory->cpu_stored = true;
        ++memory->version.cpu_write;
        prepare(3);

        // A later compatible write must not erase an older incompatible write,
        // including a longer range that shared the same start.
        const m::EpochLog::ResidentRead read{sampled.key.storage.descriptor.address, range.bytes.size,
                                           range.bytes, identity};
        m::EpochLog history;
        history.write(read.base, read.size, 10, identity);
        history.write(read.base, read.size, 11);
        history.write(read.base, read.size, 12, identity);
        check(!history.textures_clean_since(10, std::span{&read, 1}) &&
                  history.textures_clean_since(12, std::span{&read, 1}),
              "a compatible write must retain the intervening incompatible epoch");
        const std::pair metadata_range{read.base, read.size};
        check(!history.clean_since(12, std::span{&metadata_range, 1}),
              "descriptor and handle reads never receive the resident-write exemption");
        history.remap(read.base, read.size, 12);
        check(!history.textures_clean_since(12, std::span{&read, 1}),
              "a remap must invalidate even an identical resident allocation");
        history.write(read.base, read.size * 2, 13);
        history.write(read.base, read.size, 14, identity);
        check(!history.textures_clean_since(14, std::span{&read, 1}),
              "a widened write must not relabel an incompatible tail");
        auto partial = sampled;
        partial.key.first_layer = 1;
        check(m::resident_texture_identity(partial).allocation.expired(),
              "partial subresources must not receive a resident identity");
        partial = sampled;
        partial.key.storage.descriptor.format = g::TextureFormat::R32Float;
        check(m::resident_texture_identity(partial).allocation.expired(),
              "a differently described view of the same allocation must not qualify");
        partial = sampled;
        partial.transfers.push_back(std::make_shared<Lease>());
        check(m::resident_texture_identity(partial).allocation.expired(),
              "a copied subresource must not receive a resident identity");
    }
}

void deferred_store_blocks_cached_texture_handles() {
    for (const std::pair<std::uint64_t, std::uint64_t> guard :
         {std::pair{0x120ull, 4ull}, std::pair{0x200ull, 32ull},
          std::pair{0x380ull, 4ull}}) {
        for (const bool move_slot : {false, true}) {
            for (const bool alias : {false, true}) {
                Memory memory;
                memory.cpu_tracking = true;
                memory.host_peeks = true;
                Buffers buffers(memory);
                Textures textures(memory);
                auto admitted = stages();
                auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(
                    *admitted.stages.fragment);
                fragment->fragment_position =
                    armigo::shader::graphics_msl::FragmentPositionConvention::
                        MetalPositionWMatchesGuestPositionW;
                fragment->textures = {{32, 0, 0}};
                fragment->texture_handle_cbuf = 4;
                fragment->texture_facts = {{4, 32, true, false}};
                admitted.stages.fragment = fragment;
                g::DrawPlan draw;
                auto stop = stop_for(draw);
                configure_texture(memory, stop);
                if (alias) {
                    const std::array mappings{
                        g::Mapping{0x1000, memory.bytes().size(), 0, 0, memory.bytes().size(), 1},
                        g::Mapping{0x5000, memory.bytes().size(), 0, 0, memory.bytes().size(), 1}};
                    memory.remap(mappings);
                    stop.constant_buffers[4][4].address = 0x5100;
                }
                auto shared = planner();
                m::BindingReuse reuse{0, 5, {}};
                auto first = shared->prepare(stop, admitted, draw, memory, buffers, textures, reuse);
                check(std::holds_alternative<m::Bindings>(first) && textures.acquires == 1,
                      "baseline texture binding must prepare before deferred store");
                memory.deferred = g::BackingRange{{1, 1, 1}, guard.first, guard.second};
                if (move_slot) stop.constant_buffers[4][4].size = 40;
                auto result = shared->prepare(stop, admitted, draw, memory, buffers, textures, reuse);
                if (guard.first == 0x380)
                    check(std::holds_alternative<m::Bindings>(result) && textures.acquires == 1,
                          "a disjoint deferred store must keep texture preparation available");
                else
                    check(std::holds_alternative<g::RendererError>(result),
                          "a deferred handle or descriptor store must block cached texture reuse");
            }
        }
    }
}

void live_texture_facts_precede_cache_hits() {
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto fragment = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.fragment);
    fragment->fragment_position =
        armigo::shader::graphics_msl::FragmentPositionConvention::
            MetalPositionWMatchesGuestPositionW;
    fragment->textures = {{32, 0, 0}};
    fragment->texture_handle_cbuf = 4;
    fragment->texture_facts = {{4, 32, true, false}};
    admitted.stages.fragment = fragment;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    configure_texture(memory, stop);

    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) &&
              textures.acquires == 1 &&
              std::get<m::Bindings>(result).textures.size() == 1,
          "matching live texture facts must acquire and bind the L4 view");
    result = planner()->prepare(stop, admitted, draw, memory,
                                buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) &&
              textures.acquires == 2 && textures.hits == 1,
          "matching live facts must remain valid on a resident cache hit");

    stop.registers[0x982] = 3;
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(error_contains(result, "texture_handle_cbuf") &&
              textures.acquires == 2,
          "handle-selector mismatch must stop before a resident cache hit");
    stop.registers[0x982] = 4;

    fragment->texture_facts[0].normalized_coordinates = false;
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(error_contains(result, "normalized-coordinate") &&
              textures.acquires == 2,
          "normalized-coordinate mismatch must stop before a resident cache hit");

    fragment->texture_facts[0] = {4, 32, true, true};
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(error_contains(result, "comparison expectation") &&
              textures.acquires == 2,
          "comparison mismatch must stop before a resident cache hit");

    fragment->texture_facts[0] = {4, 32, true, false};
    const std::uint32_t integer =
        8u | (4u << 7) | (4u << 10) | (4u << 13) | (4u << 16) |
        (2u << 19) | (3u << 22) | (4u << 25) | (5u << 28);
    memory.word(0x1200, integer);
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(error_contains(result, "sampled scalar") &&
              textures.acquires == 2,
          "numeric-class mismatch must stop before a resident cache hit");
}

void persistent_texture_targets_and_types() {
    namespace p = armigo::platform;
    namespace gm = armigo::shader::graphics_msl;
    struct Target {
        unsigned encoded;
        p::TextureFactDimension dimension;
        bool array;
        bool normalized;
        g::TextureType type;
        unsigned layers;
    };
    const Target targets[]{
        {0, p::TextureFactDimension::d1, false, true, g::TextureType::Texture1D, 1},
        {4, p::TextureFactDimension::d1, true, true, g::TextureType::Texture1DArray, 2},
        {1, p::TextureFactDimension::d2, false, true, g::TextureType::Texture2D, 1},
        {5, p::TextureFactDimension::d2, true, true, g::TextureType::Texture2DArray, 2},
        {2, p::TextureFactDimension::d3, false, true, g::TextureType::Texture3D, 1},
        {3, p::TextureFactDimension::cube, false, true, g::TextureType::Cubemap, 6},
        {8, p::TextureFactDimension::cube, true, true, g::TextureType::CubemapArray, 12},
        {7, p::TextureFactDimension::d2, false, false, g::TextureType::Texture2DRect, 1},
    };
    for (const auto target : targets) {
        Memory memory(16384);
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        auto fragment = std::make_shared<p::CompiledGraphicsStage>(*admitted.stages.fragment);
        fragment->fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        fragment->textures = {{32, 0, target.normalized ? std::uint8_t{0} : gm::TextureBinding::no_sampler}};
        fragment->texture_handle_cbuf = 4;
        const p::DerivedTextureFact fact{4, 32, target.normalized, false,
            target.dimension, target.array, false, p::TextureSampledScalar::f32,
            p::TextureComponentFormat::unorm};
        fragment->texture_facts = {fact};
        admitted.stages.fragment = fragment;
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        memory.word(0x1210, 1u | (target.encoded << 23));
        memory.word(0x1214, 1u | (1u << 16) | (unsigned(target.normalized) << 31));
        const auto prepare = [&] {
            return planner()->prepare(stop, admitted, draw, memory, buffers, textures);
        };
        auto result = prepare();
        if (auto* error = std::get_if<g::RendererError>(&result))
            std::cerr << "Target " << target.encoded << ": " << error->command().detail << '\n';
        check(std::holds_alternative<m::Bindings>(result),
              "all L4 texture targets must bind with a matching descriptor contract");
        const auto& bound = std::get<m::Bindings>(result).textures.front();
        check(bound.texture.key.storage.descriptor.type == target.type &&
                  bound.texture.key.layer_count == target.layers,
              "planner must preserve the live texture target and complete layer range");
        check(bool(bound.sampler) == target.normalized,
              "samplerless rectangle fetch must not allocate a sampler resource");
        result = prepare();
        check(std::holds_alternative<m::Bindings>(result) && textures.hits == 1,
              "wide target bindings must remain valid on resident cache hits");
        const auto acquired = textures.acquires;
        for (unsigned mutation = 0; mutation != 6; ++mutation) {
            auto& changed = fragment->texture_facts.front();
            changed = fact;
            switch (mutation) {
            case 0: changed.dimension = target.dimension == p::TextureFactDimension::d3
                    ? p::TextureFactDimension::d2 : p::TextureFactDimension::d3; break;
            case 1: changed.multisample = true; break;
            case 2: changed.normalized_coordinates = !fact.normalized_coordinates; break;
            case 3: changed.comparison = true; break;
            case 4: changed.scalar = p::TextureSampledScalar::u32; break;
            case 5: changed.component_format = p::TextureComponentFormat::snorm; break;
            }
            result = prepare();
            check(error_contains(result, "texture_descriptor fact mismatch") &&
                      textures.acquires == acquired,
                  "every changed descriptor field must reject before a resident cache hit");
        }
        // A same-dimensions arrayness flip is admitted as a layer-zero view
        // in the shader's target; only 3D has no array form. The rejection
        // message names both targets for every other field difference.
        {
            auto& changed = fragment->texture_facts.front();
            changed = fact;
            changed.array = !fact.array;
            result = prepare();
            if (target.dimension == p::TextureFactDimension::d3) {
                check(error_contains(result, "texture_descriptor fact mismatch") &&
                          textures.acquires == acquired,
                      "a 3D-array fact must still reject before a resident cache hit");
            } else {
                if (auto* error = std::get_if<g::RendererError>(&result))
                    std::cerr << "Array flip " << target.encoded << ": "
                              << error->command().detail << '\n';
                check(std::holds_alternative<m::Bindings>(result) &&
                          textures.acquires == acquired + 1,
                      "a same-dimensions arrayness flip must bind a target view");
                const auto& viewed = std::get<m::Bindings>(result).textures.front();
                const auto expected_view = m::texture_fact_target(
                    target.dimension, !target.array);
                check(expected_view.has_value() &&
                          viewed.texture.key.view_target == expected_view &&
                          viewed.texture.key.first_layer == 0 &&
                          viewed.texture.key.layer_count ==
                              (target.dimension == p::TextureFactDimension::cube ? 6u : 1u),
                      "the admitted target view must cover layer zero only");
            }
            fragment->texture_facts = {fact};
            textures.acquires = acquired;
        }
        fragment->texture_facts = {fact};
        const auto wrong_target = target.dimension == p::TextureFactDimension::d3 ? 1u : 2u;
        memory.word(0x1210, 1u | (wrong_target << 23));
        result = prepare();
        check(error_contains(result, "texture_descriptor fact mismatch") &&
                  textures.acquires == acquired,
              "a live target change must stop before a resident cache hit");
        memory.word(0x1210, 1u | (target.encoded << 23));
        memory.word(0x121c, 1u << 8);
        result = prepare();
        check(error_contains(result, "texture_descriptor fact mismatch") &&
                  textures.acquires == acquired,
              "live multisample descriptors must stop before resident acquisition");
        memory.word(0x121c, 0);
        fragment->texture_facts.clear();
        if (target.type != g::TextureType::Texture2D) {
            result = prepare();
            check(error_contains(result, "legacy") && textures.acquires == acquired,
                  "legacy reflection must not admit unsupported target shapes");
        }
    }

    for (const bool signed_format : {false, true}) {
        Memory memory;
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        auto fragment = std::make_shared<p::CompiledGraphicsStage>(*admitted.stages.fragment);
        fragment->fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        fragment->textures = {{32, 0, gm::TextureBinding::no_sampler}};
        fragment->texture_handle_cbuf = 4;
        fragment->texture_facts = {{4, 32, true, false, p::TextureFactDimension::d2,
            false, false, signed_format ? p::TextureSampledScalar::s32 : p::TextureSampledScalar::u32,
            signed_format ? p::TextureComponentFormat::signed_integer : p::TextureComponentFormat::unsigned_integer}};
        admitted.stages.fragment = fragment;
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        configure_texture(memory, stop);
        const unsigned component = signed_format ? 3 : 4;
        memory.word(0x1200, 8u | (component << 7) | (component << 10) |
            (component << 13) | (component << 16) | (2u << 19) | (3u << 22) |
            (4u << 25) | (5u << 28));
        auto result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
        check(std::holds_alternative<m::Bindings>(result) && textures.acquires == 1 &&
                  !std::get<m::Bindings>(result).textures.front().sampler,
              "matching signed and unsigned fetch contracts must bind without samplers");
        fragment->texture_facts.front().scalar = signed_format
            ? p::TextureSampledScalar::u32 : p::TextureSampledScalar::s32;
        result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
        check(error_contains(result, "sampled scalar") && textures.acquires == 1,
              "signedness mismatch must stop before resident cache acquisition");
        fragment->texture_facts.clear();
        fragment->textures.front().sampler_index = 0;
        result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
        check(error_contains(result, "integer texture") && textures.acquires == 1,
              "legacy reflection must explicitly reject integer texture types");
    }
}

void admitted_target_views_pass_remaining_facts() {
    namespace p = armigo::platform;
    g::ResolvedTextureBinding live;
    live.texture.type = g::TextureType::Texture2DArray;
    live.texture.format = g::TextureFormat::R8G8B8A8Unorm;
    live.texture.samples = 1;
    live.texture.normalized_coordinates = true;
    const p::DerivedTextureFact fact{4, 32, true, false};
    check(m::texture_fact_mismatch(fact, live).empty(),
          "an admitted same-dimensions view must pass the remaining fact checks");
    live.texture.normalized_coordinates = false;
    check(m::texture_fact_mismatch(fact, live).find("normalized-coordinate") !=
              std::string::npos,
          "admitted target views must still verify coordinate facts");
}

void float_sampled_depth_formats_are_admitted() {
    namespace p = armigo::platform;
    g::ResolvedTextureBinding live;
    live.texture.type = g::TextureType::Texture2D;
    live.texture.samples = 1;
    live.texture.normalized_coordinates = true;
    const p::DerivedTextureFact fact{4, 32, true, false};
    for (const auto format : {g::TextureFormat::D32Float, g::TextureFormat::D16Unorm}) {
        live.texture.format = format;
        check(m::texture_fact_mismatch(fact, live).empty(),
              "a non-comparison float sample of a depth format reads the depth value");
    }
    live.texture.format = g::TextureFormat::D32Float;
    live.sampler.depth_compare = true;
    check(!m::texture_fact_mismatch(fact, live).empty(),
          "a float-sampled fact still rejects a comparison sampler");
}

void comparison_facts_admit_depth_storage_aliases() {
    namespace p = armigo::platform;
    g::ResolvedTextureBinding live;
    live.texture.type = g::TextureType::Texture2D;
    live.texture.samples = 1;
    live.texture.normalized_coordinates = true;
    live.sampler.depth_compare = true;
    // a compare-sampled colour-coded TIC with a depth
    // storage alias (a DEPTH16 target sampled through R16Unorm) is a
    // depth texture's sampling view and is admitted as depth.
    const p::DerivedTextureFact compare{4, 32, true, true,
        p::TextureFactDimension::d2, false, false, p::TextureSampledScalar::f32,
        p::TextureComponentFormat::depth};
    for (const auto format : {g::TextureFormat::R16Unorm, g::TextureFormat::R32Float}) {
        live.texture.format = format;
        check(m::texture_fact_mismatch(compare, live).empty(),
              "a comparison sample of a depth-alias colour TIC must be admitted");
    }
    // Colour formats without a depth equivalent keep the structured refusal
    // naming both formats, and the alias never relaxes other contracts.
    live.texture.format = g::TextureFormat::R8Unorm;
    const auto refused = m::texture_fact_mismatch(compare, live);
    check(refused.find("comparison sampling requires a depth component format") !=
              std::string::npos &&
              refused.find("R8Unorm") != std::string::npos,
          "comparison sampling of a colour format must refuse naming both formats");
    p::DerivedTextureFact unsigned_compare = compare;
    unsigned_compare.scalar = p::TextureSampledScalar::u32;
    live.texture.format = g::TextureFormat::R16Unorm;
    check(!m::texture_fact_mismatch(unsigned_compare, live).empty(),
          "the depth alias must not relax the sampled scalar contract");
}

void compare_samplers_bind_depth_storage() {
    namespace p = armigo::platform;
    namespace gm = armigo::shader::graphics_msl;
    const p::DerivedTextureFact compare{4, 32, true, true,
        p::TextureFactDimension::d2, false, false, p::TextureSampledScalar::f32,
        p::TextureComponentFormat::depth};
    Memory memory(16384);
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto fragment = std::make_shared<p::CompiledGraphicsStage>(*admitted.stages.fragment);
    fragment->fragment_position =
        gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    fragment->textures = {{32, 0, 0}};
    fragment->texture_handle_cbuf = 4;
    fragment->texture_facts = {compare};
    admitted.stages.fragment = fragment;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    configure_texture(memory, stop);
    // TSC word 0: configure_texture's wrap state plus depth compare (bit 9)
    // and the LessOrEqual function (bits 10-12).
    const std::uint32_t compare_tsc =
        2u | (2u << 3) | (2u << 6) | (1u << 9) | (3u << 10);
    const auto prepare = [&](std::uint32_t tic, std::uint32_t tsc) {
        configure_format_and_swizzle(memory, tic, {2, 3, 4, 5}, false);
        memory.word(0x1300, tsc);
        return planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    };
    // R16Unorm TIC (TIC 0x1b UNORM, a DEPTH16 sampling view) with a
    // comparison TSC resolves onto the depth storage key.
    auto result = prepare(0x1b, compare_tsc);
    if (auto* error = std::get_if<g::RendererError>(&result))
        std::cerr << "Compare alias prepare: " << error->command().detail << '\n';
    check(std::holds_alternative<m::Bindings>(result) && textures.acquires == 1,
          "a compare-sampled R16Unorm TIC must bind through the depth alias");
    const auto& view = textures.views.back();
    check(view.storage.descriptor.format == g::TextureFormat::D16Unorm &&
              view.aspect == g::TextureAspect::Depth,
          "the compare binding must resolve the colour-coded TIC onto depth storage");
    result = prepare(0x1b, compare_tsc);
    check(std::holds_alternative<m::Bindings>(result) && textures.hits == 1,
          "the rewritten depth key must stay stable across resident cache hits");
    // A non-comparison sampler keeps the colour-coded storage key.
    const p::DerivedTextureFact plain{4, 32, true, false,
        p::TextureFactDimension::d2, false, false, p::TextureSampledScalar::f32,
        p::TextureComponentFormat::floating};
    fragment->texture_facts = {plain};
    result = prepare(0x1b, 2u | (2u << 3) | (2u << 6));
    check(std::holds_alternative<m::Bindings>(result) &&
              textures.views.back().storage.descriptor.format ==
                  g::TextureFormat::R16Unorm &&
              textures.views.back().aspect == g::TextureAspect::Color,
          "a non-comparison sampler must keep the colour-coded storage key");
    // An R8Unorm TIC has no depth alias: the fact refusal names both formats
    // and stops before resident acquisition.
    fragment->texture_facts = {compare};
    const auto acquired = textures.acquires;
    result = prepare(0x1d, compare_tsc);
    check(error_contains(result, "depth component format") &&
              error_contains(result, "R8Unorm") && textures.acquires == acquired,
          "comparison sampling without a depth alias must refuse naming both formats");
}

void samplerless_bindings_ignore_unused_tsc() {
    namespace p = armigo::platform;
    namespace gm = armigo::shader::graphics_msl;
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto fragment = std::make_shared<p::CompiledGraphicsStage>(*admitted.stages.fragment);
    fragment->fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    fragment->textures = {{32, 0, gm::TextureBinding::no_sampler}};
    fragment->texture_handle_cbuf = 4;
    fragment->texture_facts = {{4, 32, true, false}};
    admitted.stages.fragment = fragment;
    g::DrawPlan draw;
    auto stop = stop_for(draw);
    for (unsigned mutation = 0; mutation != 5; ++mutation) {
        configure_texture(memory, stop);
        stop.registers[0x48d] = 0;
        switch (mutation) {
        case 0: memory.word(0x1304, 0); break;
        case 1: stop.registers[0x558] = 0x8000; break;
        case 2: memory.word(0x1300, 2u | (2u << 3) | (2u << 6) | (1u << 9)); break;
        case 3: memory.word(0x1120, 7u << 20); break;
        case 4: stop.registers[0x48d] = 1; break;
        }
        fragment->textures.front().sampler_index = gm::TextureBinding::no_sampler;
        auto result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
        check(std::holds_alternative<m::Bindings>(result) &&
                  !std::get<m::Bindings>(result).textures.front().sampler,
              "fetch/query-only bindings must ignore unused TSC bytes, address, mode, index and compare state");
        const auto acquired = textures.acquires;
        fragment->textures.front().sampler_index = 0;
        result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
        check(std::holds_alternative<g::RendererError>(result) && textures.acquires == acquired,
              "sampled bindings must reject the same invalid or mismatched TSC state before acquisition");
    }
}

void collisions_stop_and_direct_indices_bind_unread() {
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->vertex_class_inputs = {
        {0, g::VertexFormatClass::float_class}};
    vertex->interface.generic_inputs = {{0, 1}};
    vertex->constant_buffer_windows = {{0, 4, 18}};
    vertex->constant_buffer_support_index = 17;
    admitted.stages.vertex = vertex;

    g::DrawPlan draw;
    draw.source.indexed = true;
    draw.source.count = 2;
    draw.vertex.attributes = {
        {0, 0, 0, false, {g::VertexAttribFormat::R32Float,
                          g::VertexFormatClass::float_class, 1, 4}}};
    draw.vertex.streams = {{0, 0x1100, 0x1107, 4, 0}};
    draw.vertex.steps = {{0, g::VertexStep::PerVertex, 1}};
    draw.indices = g::IndexPlan{g::IndexFormat::U16, g::IndexFormat::U16,
                                0x1000, 0x1003, false, 0, {}};
    draw.segments = {{0, 2, 0, 0, 1}};
    auto stop = stop_for(draw);
    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<g::RendererError>(result) &&
              buffers.acquires == 0 && buffers.uploads.empty(),
          "vertex and shader buffer-index collision must stop before allocation");

    vertex->constant_buffer_windows.clear();
    vertex->constant_buffer_support_index.reset();
    admitted.stages.vertex = vertex;
    memory.bytes()[0] = std::byte{0};
    memory.bytes()[1] = std::byte{0};
    memory.bytes()[2] = std::byte{5};
    memory.bytes()[3] = std::byte{0};
    memory.reads.clear();
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) && memory.reads.empty(),
          "direct guest indices bind in place unread, an index past the stream limit included");

    draw.source.base_vertex = 1;
    draw.indices->restart_enabled = true;
    draw.indices->restart_index = 0xffff;
    draw.segments[0].base_vertex = 1;
    draw.vertex.streams[0].inclusive_limit = 0x110b;
    stop = stop_for(draw);
    memory.bytes()[0] = std::byte{1};
    memory.bytes()[1] = std::byte{0};
    memory.bytes()[2] = std::byte{0xff};
    memory.bytes()[3] = std::byte{0xff};
    result = planner()->prepare(stop, admitted, draw, memory, buffers, textures);
    check(std::holds_alternative<m::Bindings>(result),
          "restart must be removed before applying native base vertex reach");
}

void overlapping_direct_buffers_preflight_the_union() {
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<armigo::platform::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->vertex_class_inputs = {
        {0, g::VertexFormatClass::float_class}};
    vertex->interface.generic_inputs = {{0, 1}};
    admitted.stages.vertex = vertex;

    g::DrawPlan draw;
    draw.source.indexed = true;
    draw.source.count = 2;
    draw.vertex.attributes = {
        {0, 0, 0, false, {g::VertexAttribFormat::R32Float,
                          g::VertexFormatClass::float_class, 1, 4}}};
    draw.vertex.streams = {{0, 0x1100, 0x1103, 4, 0}};
    draw.vertex.steps = {{0, g::VertexStep::PerVertex, 1}};
    draw.indices = g::IndexPlan{g::IndexFormat::U16, g::IndexFormat::U16,
                                0x1100, 0x1103, false, 0, {}};
    draw.segments = {{0, 2, 0, 0, 1}};
    memory.word(0x1100, 0);
    auto stop = stop_for(draw);
    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<m::Bindings>(result) &&
              buffers.acquires == 3 && buffers.acquired_ranges.size() == 3 &&
              buffers.acquired_ranges[0].offset == 0x100 &&
              buffers.acquired_ranges[0].size == 4,
          "overlapping index and vertex views must acquire their union first");
}

void storage_descriptors_bind_exact_guest_ranges() {
    namespace p = armigo::platform;
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto vertex = std::make_shared<p::CompiledGraphicsStage>(
        *admitted.stages.vertex);
    vertex->storage_buffers = {
        p::CompiledStorageBufferBinding{
            2, 16, 7, p::CompiledShaderResourceAccess::Read, 4},
        p::CompiledStorageBufferBinding{
            2, 32, 8, p::CompiledShaderResourceAccess::Write, 4},
    };
    vertex->constant_buffer_windows = {{2, 64, 0}};
    vertex->constant_buffer_support_index = 17;
    admitted.stages.vertex = vertex;

    g::DrawPlan draw;
    auto stop = stop_for(draw);
    stop.constant_buffers[0][2] = {0x1000, 64, true};
    memory.word(0x1010, 0x1200);
    memory.word(0x1014, 0);
    memory.word(0x1018, 16);
    memory.word(0x1020, 0x1300);
    memory.word(0x1024, 0);
    memory.word(0x1028, 12);

    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<m::Bindings>(result),
          "valid storage descriptors must prepare");
    if (!std::holds_alternative<m::Bindings>(result)) return;
    const auto& bindings = std::get<m::Bindings>(result);
    check(buffers.acquisitions.size() == 2 &&
              buffers.acquisitions[0].range.offset == 0x200 &&
              buffers.acquisitions[0].range.size == 16 &&
              buffers.acquisitions[0].use == g::BufferUse::Storage &&
              buffers.acquisitions[0].access == g::ResourceAccess::Read &&
              buffers.acquisitions[0].alignment == 4 &&
              buffers.acquisitions[1].range.offset == 0x300 &&
              buffers.acquisitions[1].range.size == 12 &&
              buffers.acquisitions[1].use == g::BufferUse::Storage &&
              buffers.acquisitions[1].access == g::ResourceAccess::ReadWrite &&
              buffers.acquisitions[1].alignment == 4,
          "storage acquisition must preserve range and promote shader writes");
    const auto first = std::ranges::find_if(bindings.buffers, [](const auto& item) {
        return item.use == g::BufferUse::Storage && item.index == 7;
    });
    const auto second = std::ranges::find_if(bindings.buffers, [](const auto& item) {
        return item.use == g::BufferUse::Storage && item.index == 8;
    });
    check(first != bindings.buffers.end() && second != bindings.buffers.end() &&
              first->stage == armigo::shader_input::Stage::Vertex &&
              first->access == g::ResourceAccess::Read &&
              second->access == g::ResourceAccess::ReadWrite,
          "storage buffers must use their reflected Metal indices");
    check(bindings.accesses.size() == 2 &&
              bindings.accesses[0].access == g::ResourceAccess::Read &&
              bindings.accesses[1].access == g::ResourceAccess::ReadWrite &&
              bindings.accesses[0].range.bytes ==
                  buffers.acquisitions[0].range &&
              bindings.accesses[1].range.bytes ==
                  buffers.acquisitions[1].range,
          "storage ranges must participate in GPU access tracking");
}

void zero_storage_descriptor_uses_owned_placeholder() {
    namespace p = armigo::platform;
    Memory memory;
    Buffers buffers(memory);
    Textures textures(memory);
    auto admitted = stages();
    auto fragment = std::make_shared<p::CompiledGraphicsStage>(
        *admitted.stages.fragment);
    fragment->storage_buffers = {
        p::CompiledStorageBufferBinding{
            3, 8, 9, p::CompiledShaderResourceAccess::ReadWrite, 4}};
    fragment->constant_buffer_windows = {{3, 32, 0}};
    fragment->constant_buffer_support_index = 17;
    admitted.stages.fragment = fragment;

    g::DrawPlan draw;
    auto stop = stop_for(draw);
    stop.constant_buffers[4][3] = {0x1000, 32, true};
    memory.word(0x1008, 0xffffffffu);
    memory.word(0x100c, 0xffffffffu);
    memory.word(0x1010, 0);
    auto result = planner()->prepare(stop, admitted, draw, memory,
                                     buffers, textures);
    check(std::holds_alternative<m::Bindings>(result),
          "a size-zero storage descriptor must bind safely");
    if (!std::holds_alternative<m::Bindings>(result)) return;
    const auto& bindings = std::get<m::Bindings>(result);
    check(buffers.acquisitions.empty() && buffers.uploads.size() == 3 &&
              buffers.uploads.back().size() == 4 &&
              std::ranges::all_of(buffers.uploads.back(), [](std::byte byte) {
                  return byte == std::byte{0};
              }),
          "size-zero storage must use a host-owned four-byte zero placeholder");
    const auto storage = std::ranges::find_if(bindings.buffers, [](const auto& item) {
        return item.use == g::BufferUse::Storage;
    });
    check(storage != bindings.buffers.end() && storage->index == 9 &&
              storage->access == g::ResourceAccess::Read &&
              bindings.accesses.empty(),
          "the placeholder must bind without retaining or tracking guest memory");
}

void malformed_storage_descriptors_stop_structurally() {
    namespace p = armigo::platform;
    const auto prepare = [](p::CompiledStorageBufferBinding binding,
                            g::ConstantBufferDescription descriptor,
                            std::array<std::uint32_t, 3> words) {
        Memory memory;
        Buffers buffers(memory);
        Textures textures(memory);
        auto admitted = stages();
        auto vertex = std::make_shared<p::CompiledGraphicsStage>(
            *admitted.stages.vertex);
        vertex->storage_buffers = {binding};
        vertex->constant_buffer_windows = {{binding.cbuf_slot, 64, 0}};
        vertex->constant_buffer_support_index = 17;
        admitted.stages.vertex = vertex;
        g::DrawPlan draw;
        auto stop = stop_for(draw);
        stop.constant_buffers[0][binding.cbuf_slot] = descriptor;
        if (descriptor.enabled && descriptor.address >= 0x1000 &&
            descriptor.address <= 0x1ff4) {
            memory.word(descriptor.address + binding.descriptor_byte_offset,
                        words[0]);
            memory.word(descriptor.address + binding.descriptor_byte_offset + 4,
                        words[1]);
            memory.word(descriptor.address + binding.descriptor_byte_offset + 8,
                        words[2]);
        }
        return planner()->prepare(stop, admitted, draw, memory,
                                  buffers, textures);
    };
    const p::CompiledStorageBufferBinding valid{
        2, 16, 7, p::CompiledShaderResourceAccess::Read, 4};

    auto result = prepare(valid, {0x1000, 64, false}, {0x1200, 0, 4});
    check(std::holds_alternative<g::RendererError>(result) &&
              std::get<g::RendererError>(result).command().status ==
                  g::CommandStatus::MemoryFault,
          "a disabled descriptor constant buffer must return a structured error");
    result = prepare(valid, {0x1000, 24, true}, {0x1200, 0, 4});
    check(std::holds_alternative<g::RendererError>(result) &&
              std::get<g::RendererError>(result).command().status ==
                  g::CommandStatus::MemoryFault,
          "a short descriptor constant buffer must return a structured error");
    result = prepare(valid, {0x1000, 64, true}, {0x4000, 0, 4});
    check(std::holds_alternative<g::RendererError>(result) &&
              std::get<g::RendererError>(result).command().status ==
                  g::CommandStatus::MemoryFault,
          "a nonzero unmapped storage descriptor must return a memory fault");
    auto misaligned = valid;
    misaligned.alignment = 8;
    result = prepare(misaligned, {0x1000, 64, true}, {0x1200, 0, 4});
    check(std::holds_alternative<g::RendererError>(result) &&
              std::get<g::RendererError>(result).command().status ==
                  g::CommandStatus::Unsupported,
          "storage reflection with non-ABI alignment must be rejected");
}

} // namespace

int main() {
    check(std::holds_alternative<g::RendererError>(
              m::create_binding_planner(nullptr)),
          "planner factory must reject a missing state cache");
    vertex_layout_is_canonical();
    mixed_constant_stream_moves_constants_to_a_free_slot();
    divided_instance_stream_expands_absolute_indices();
    runtime_windows_zero_fill_and_report_sizes();
    unchanged_windows_reuse_their_snapshot();
    runtime_windows_preserve_fragments_and_keep_descriptors_strict();
    gpu_written_window_binds_resident_image();
    legacy_uniform_snapshot_stops_without_substitution();
    custom_border_binding_profiles();
    custom_border_records_and_views();
    custom_border_reflection_rejections();
    descriptor_only_planner_reads_and_retry();
    live_texture_facts_precede_cache_hits();
    decoded_pool_entries_are_reused();
    unchanged_stage_textures_are_reused();
    resident_attachment_writes_preserve_texture_bindings();
    deferred_store_blocks_cached_texture_handles();
    persistent_texture_targets_and_types();
    admitted_target_views_pass_remaining_facts();
    float_sampled_depth_formats_are_admitted();
    comparison_facts_admit_depth_storage_aliases();
    compare_samplers_bind_depth_storage();
    samplerless_bindings_ignore_unused_tsc();
    collisions_stop_and_direct_indices_bind_unread();
    overlapping_direct_buffers_preflight_the_union();
    storage_descriptors_bind_exact_guest_ranges();
    zero_storage_descriptor_uses_owned_placeholder();
    malformed_storage_descriptors_stop_structurally();
    std::cout << "Metal binding checks passed\n";
}
