// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic tests for lane L5 buffer residency: shared
// coherent views across vertex/index/uniform/storage uses, partial CPU writes,
// GPU-produced consumption without a CPU round trip, uniform record-time
// versioning, and completion-gated upload-ring reuse. All fixtures are
// synthetic; no game-derived bytes appear here.

#include <armigo/graphics/buffer_cache.h>
#include <armigo/graphics/resource_memory_registry.h>
#include "buffer_residency.h"

#include <algorithm>
#include <functional>
#include <string_view>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

// Portable construction seam; defined in src/graphics/buffer_cache.cpp. The
// Metal entry point lives in src/platform/metal/buffer_cache.h.
namespace armigo::graphics {
RendererResult<std::shared_ptr<BufferCache>> make_buffer_cache(
    std::shared_ptr<ResourceMemory> memory, std::uint64_t budget_bytes);
}

namespace {

using namespace armigo::graphics;

unsigned failures = 0;

#define CHECK(...)                                                             \
    do {                                                                       \
        if (!(__VA_ARGS__)) {                                                  \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__);     \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

bool stopped(const RendererResult<ResidentBuffer>& result) {
    const auto* failure = std::get_if<RendererError>(&result);
    return failure && failure->command().status == CommandStatus::Stopped;
}

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
        if (on_read) on_read(range);
        if (fail_read) return RendererError({CommandStatus::MemoryFault, "injected CPU read failure"});
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

    RendererStatus wait_gpu(CompletionToken) override {
        ++waits;
        if (on_wait) on_wait();
        return renderer_success();
    }

    std::vector<Allocation> allocations;
    unsigned reads = 0, writes = 0, waits = 0;
    std::function<void()> on_wait;
    std::function<void(BackingRange)> on_read;
    bool fail_read = false;
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

constexpr BackingId backing_a{1, 1, 1};
constexpr BackingId backing_b{1, 2, 1};
constexpr MappingId mapping_a{7, 1, 1};
constexpr MappingId mapping_b{7, 2, 1};

// Synthetic write-tracking registry: forwards every operation to a real
// registry and overrides only the optional CPU write stamp contract. touch()
// advances the store generation for the whole address space or, with a range,
// for exactly the stored suspect ranges; a stamp taken after the last touch
// proves everything clean.
struct StampRegistry final : ResourceMemoryRegistry {
    explicit StampRegistry(std::shared_ptr<ResourceMemoryRegistry> inner)
        : inner_(std::move(inner)) {}

    std::uint64_t cpu_write_stamp() const noexcept override { return stamp_; }
    bool cpu_written_since(BackingRange range, std::uint64_t stamp) const noexcept override {
        if (stamp >= whole_since_) return false;
        if (whole_dirty_) return true;
        for (const auto& dirty : dirty_) {
            if (dirty.backing == range.backing &&
                dirty.offset < range.offset + range.size &&
                range.offset < dirty.offset + dirty.size) return true;
        }
        return false;
    }
    void touch() {
        ++stamp_;
        whole_since_ = stamp_;
        whole_dirty_ = true;
    }
    void touch(BackingRange range) {
        ++stamp_;
        whole_since_ = stamp_;
        dirty_.push_back(range);
    }

    RendererStatus register_backing(BackingRegistration backing) override {
        return inner_->register_backing(backing);
    }
    RendererStatus register_mapping(MappingRegistration mapping) override {
        return inner_->register_mapping(mapping);
    }
    RendererStatus unmap(MappingId mapping) override { return inner_->unmap(mapping); }
    RendererStatus retire_backing(BackingId backing) override {
        return inner_->retire_backing(backing);
    }
    RendererResult<RetainedRange> retain(std::uint64_t gpu_address, std::uint64_t size,
                                         ResourceAccess access) override {
        return inner_->retain(gpu_address, size, access);
    }
    const GpuAddressSpace& address_space() const noexcept override {
        return inner_->address_space();
    }
    CoherenceMode coherence_mode() const noexcept override {
        return inner_->coherence_mode();
    }
    RendererResult<std::vector<DirtyRange>> inspect(BackingRange range) override {
        return inner_->inspect(range);
    }
    RendererStatus resolve(BackingRange range, ResolveReason reason) override {
        return inner_->resolve(range, reason);
    }
    RendererResult<std::unique_ptr<CpuReadTransaction>> prepare_cpu_read(
        BackingRange range, ResolveReason reason) override {
        return inner_->prepare_cpu_read(range, reason);
    }
    RendererResult<std::unique_ptr<CpuWriteTransaction>> prepare_cpu_write(
        BackingRange range) override {
        return inner_->prepare_cpu_write(range);
    }
    RendererResult<std::shared_ptr<GpuAccessReservation>> reserve_accesses(
        std::span<const GpuAccessRequest> requests, CompletionToken consumer) override {
        return inner_->reserve_accesses(requests, consumer);
    }
    RendererStatus invalidate_mappings(std::span<const MappingId> mappings,
                                       std::span<const BackingId> discarded) override {
        return inner_->invalidate_mappings(mappings, discarded);
    }
    bool has_backing(BackingId backing) const noexcept override {
        return inner_->has_backing(backing);
    }
    RendererStatus validate_retained(const RetainedRange& range,
                                     ResourceAccess access) override {
        return inner_->validate_retained(range, access);
    }
    RendererStatus read_retained(const RetainedRange& retained, BackingRange bytes,
                                 std::span<std::byte> destination) override {
        return inner_->read_retained(retained, bytes, destination);
    }
    RendererResult<std::vector<ResourceStorageRange>> inspect_storage(
        BackingRange range) override {
        return inner_->inspect_storage(range);
    }
    RendererStatus submission_boundary() override {
        return inner_->submission_boundary();
    }
    std::uint64_t cpu_cache_epoch() const noexcept override {
        return inner_->cpu_cache_epoch();
    }

private:
    std::shared_ptr<ResourceMemoryRegistry> inner_;
    std::uint64_t stamp_ = 1, whole_since_ = 0;
    bool whole_dirty_ = false;
    std::vector<BackingRange> dirty_;
};

struct Fixture {
    explicit Fixture(std::uint64_t budget)
        : Fixture(budget, {}) {}
    Fixture(std::uint64_t budget,
            std::function<std::shared_ptr<ResourceMemoryRegistry>(
                std::shared_ptr<ResourceMemoryRegistry>)> wrap)
        : backend(std::make_shared<Backend>()) {
        backend->add(backing_a, 0x100, std::byte{0x11});
        backend->add(backing_b, 0x100, std::byte{0x33});
        auto inner = require_value(make_resource_memory(
            {7, backend, CoherenceMode::SubmissionBoundary, 0x10}));
        memory = wrap ? wrap(std::move(inner)) : std::move(inner);
        CHECK(succeeded(memory->register_backing({backing_a, 0x100,
                                                  std::make_shared<ResourceLease>()})));
        CHECK(succeeded(memory->register_backing({backing_b, 0x100,
                                                  std::make_shared<ResourceLease>()})));
        const Mapping a{0x1000, 0x100, 0, 0x8000, 0x100, 0x10};
        const Mapping b{0x2000, 0x100, 0, 0x9000, 0x100, 0x10};
        CHECK(succeeded(memory->register_mapping({mapping_a, backing_a, a})));
        CHECK(succeeded(memory->register_mapping({mapping_b, backing_b, b})));
        cache = require_value(make_buffer_cache(memory, budget));
    }

    // Guest CPU store through the tracked address-space path.
    void guest_write(std::uint64_t gpu_address, std::uint64_t size, std::byte fill) {
        std::vector<std::byte> bytes(size, fill);
        auto result = memory->address_space().write(gpu_address, bytes);
        CHECK(result.ok());
    }

    RetainedRange retain(std::uint64_t gpu_address, std::uint64_t size,
                         ResourceAccess access) {
        auto result = memory->retain(gpu_address, size, access);
        if (auto* failure = std::get_if<RendererError>(&result)) {
            const auto command = failure->command();
            std::fprintf(stderr, "retain %llu+%llu failed: %.*s\n",
                         static_cast<unsigned long long>(gpu_address),
                         static_cast<unsigned long long>(size),
                         static_cast<int>(command.detail.size()), command.detail.data());
        }
        return require_value(std::move(result));
    }

    std::shared_ptr<Backend> backend;
    std::shared_ptr<ResourceMemoryRegistry> memory;
    std::shared_ptr<BufferCache> cache;
};

std::vector<std::byte> read_image(const ResidentBuffer& resident, BackingRange range) {
    auto authority = std::dynamic_pointer_cast<const ResourceMemoryStorage>(resident.storage);
    CHECK(authority != nullptr);
    std::vector<std::byte> bytes(range.size);
    CHECK(succeeded(authority->read_gpu(range, bytes)));
    return bytes;
}

std::vector<std::byte> read_slice(const ResidentBuffer& resident, std::uint64_t size) {
    auto authority = std::dynamic_pointer_cast<const ResourceMemoryStorage>(resident.storage);
    CHECK(authority != nullptr);
    std::vector<std::byte> bytes(size);
    CHECK(succeeded(authority->read_gpu({BackingId{}, 0, size}, bytes)));
    return bytes;
}

bool filled(const std::vector<std::byte>& bytes, std::byte fill) {
    return std::all_of(bytes.begin(), bytes.end(),
                       [&](std::byte value) { return value == fill; });
}

BufferRequest request(const RetainedRange& range, BufferUse use, ResourceAccess access,
                      std::uint64_t alignment) {
    return BufferRequest{range, use, access, alignment};
}

// Overlapping vertex, index, uniform and storage requests resolve to one
// growing coherent view keyed by backing-relative range.
void test_uses_share_one_coherent_view() {
    Fixture fixture(0x40000);
    auto vertex = fixture.cache->acquire(request(
        fixture.retain(0x1010, 0x20, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 4));
    auto vertex_view = require_value(std::move(vertex));
    CHECK(vertex_view.key.bytes == BackingRange{backing_a, 0x10, 0x20});
    CHECK(filled(read_image(vertex_view, {backing_a, 0x10, 0x20}), std::byte{0x11}));

    vertex_view = {};
    auto uniform = fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x20, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10));
    const auto& uniform_view = require_value(uniform);
    CHECK(uniform_view.key.bytes == BackingRange{backing_a, 0x10, 0x30});
    // Offset stays aligned to the request inside the shared view.
    CHECK(uniform_view.offset % 0x10 == 0);
    CHECK(filled(read_slice(uniform_view, 0x20), std::byte{0x11}));

    auto storage = fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x80, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Read, 4));
    const auto& storage_view = require_value(storage);
    CHECK(storage_view.key.bytes == BackingRange{backing_a, 0x0, 0x80});
    CHECK(filled(read_image(storage_view, {backing_a, 0x0, 0x80}), std::byte{0x11}));

    // An index use of the same range hits the same coherent view.
    const auto reads_before = fixture.backend->reads;
    auto index = fixture.cache->acquire(request(
        fixture.retain(0x1008, 0x10, ResourceAccess::Read), BufferUse::Index,
        ResourceAccess::Read, 4));
    const auto& index_view = require_value(index);
    CHECK(index_view.key.bytes == BackingRange{backing_a, 0x0, 0x80});
    CHECK(fixture.backend->reads == reads_before);
    CHECK(index_view.dependencies.empty());
}

// A native extent beyond the guest range is allocated (and charged) with the
// image; an image that already covers a smaller extent is reused.
void test_native_extent_pads_the_image() {
    Fixture fixture(0x1000);
    auto too_large = request(fixture.retain(0x1010, 0x10, ResourceAccess::ReadWrite),
                             BufferUse::Storage, ResourceAccess::ReadWrite, 4);
    too_large.native_extent = 0x2000;
    CHECK(std::holds_alternative<RendererError>(fixture.cache->acquire(too_large)));
    auto padded = request(fixture.retain(0x1010, 0x10, ResourceAccess::ReadWrite),
                          BufferUse::Storage, ResourceAccess::ReadWrite, 4);
    padded.native_extent = 0x100;
    const auto view = require_value(fixture.cache->acquire(padded));
    CHECK(view.key.bytes == BackingRange{backing_a, 0x10, 0x10});
    CHECK(filled(read_image(view, {backing_a, 0x10, 0x10}), std::byte{0x11}));
    const auto reads = fixture.backend->reads;
    auto smaller = request(fixture.retain(0x1010, 0x10, ResourceAccess::ReadWrite),
                           BufferUse::Storage, ResourceAccess::ReadWrite, 4);
    smaller.native_extent = 0x40;
    const auto again = require_value(fixture.cache->acquire(smaller));
    CHECK(again.storage == view.storage);
    CHECK(fixture.backend->reads == reads);
}

// A constant window's snapshot spans the window: the guest bytes, then zeros
// past the bound size, never bytes left in the upload ring.
void test_uniform_window_snapshot_zero_tail() {
    Fixture fixture(0x1000);
    auto window = request(fixture.retain(0x1010, 0x10, ResourceAccess::Read),
                          BufferUse::Uniform, ResourceAccess::Read, 4);
    window.native_extent = 0x40;
    const auto view = require_value(fixture.cache->acquire(window));
    CHECK(view.size == 0x40);
    const auto bytes = read_slice(view, 0x40);
    CHECK(filled({bytes.begin(), bytes.begin() + 0x10}, std::byte{0x11}));
    CHECK(filled({bytes.begin() + 0x10, bytes.end()}, std::byte{0}));
}

// A partial CPU store re-uploads only the changed interval; unchanged bytes
// come from the already-synced image with no additional guest reads.
void test_partial_cpu_write_updates_only_changed_interval() {
    Fixture fixture(0x40000);
    auto first = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)first;
    const auto reads_after_first = fixture.backend->reads;

    fixture.guest_write(0x1010, 0x10, std::byte{0x22});

    auto second = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(second.offset == 0);
    const auto image = read_image(second, {backing_a, 0x0, 0x40});
    CHECK(filled({image.begin(), image.begin() + 0x10}, std::byte{0x11}));
    CHECK(filled({image.begin() + 0x10, image.begin() + 0x20}, std::byte{0x22}));
    CHECK(filled({image.begin() + 0x20, image.end()}, std::byte{0x11}));
    // Exactly one interval, the written one, was re-uploaded.
    CHECK(fixture.backend->reads == reads_after_first + 1);

    // A repeated acquire with no intervening change performs zero reads.
    const auto reads_before_idle = fixture.backend->reads;
    auto third = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)third;
    CHECK(fixture.backend->reads == reads_before_idle);
}

// A GPU write committed against the view's own storage is consumed from that
// view: no guest readback and no guest write occur for the produced interval.
void test_gpu_produced_bytes_are_consumed_without_cpu_round_trip() {
    Fixture fixture(0x40000);
    auto write_view = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Write, 1)));
    CHECK(write_view.key.bytes == BackingRange{backing_a, 0x20, 0x10});

    TokenSource token(1);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({write_view.storage})));
    token.submit();

    const auto writes_before = fixture.backend->writes;
    // Direct reuse needs only the producer dependency; growing the image
    // needs completed bytes and must stop until feedback is available.
    auto direct = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(direct.storage == write_view.storage);
    CHECK(direct.dependencies.size() == 1);
    auto pending = fixture.cache->acquire(request(
        fixture.retain(0x1010, 0x20, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1));
    CHECK(std::holds_alternative<RendererError>(pending));
    token.succeed();
    // Bindings still hold the image: growth stops. The finished bytes may
    // already be written back to the guest, which is coherent.
    CHECK(stopped(fixture.cache->acquire(request(
        fixture.retain(0x1010, 0x20, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1))));
    CHECK(fixture.backend->writes <= writes_before + 1);
    write_view = {};
    direct = {};
    CHECK(succeeded(fixture.memory->resolve({backing_a, 0x20, 0x10}, ResolveReason::Capture)));
    const auto reads_before = fixture.backend->reads;
    auto read_view = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1010, 0x20, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(fixture.backend->writes == writes_before + 1);
    CHECK(fixture.backend->reads == reads_before + 1);
    CHECK(read_view.dependencies.empty());
}

// An image held only as the registry's finished GPU authority is written back
// once, and the growth then copies those bytes without a stop.
void test_growth_writes_back_finished_authority() {
    Fixture fixture(0x40000);
    auto write_view = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Write, 1)));
    TokenSource token(1);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({write_view.storage})));
    token.submit();
    token.succeed();
    reservation.reset();
    write_view = {};
    // The renderer's boundary after completion prunes the finished access
    // while the written bytes stay GPU-authoritative.
    CHECK(succeeded(fixture.memory->internal_boundary()));
    const auto writes_before = fixture.backend->writes;
    auto grown_result = fixture.cache->acquire(request(
        fixture.retain(0x1010, 0x20, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1));
    auto grown = require_value(std::move(grown_result));
    CHECK(grown.key.bytes == BackingRange{backing_a, 0x10, 0x20});
    CHECK(fixture.backend->writes == writes_before + 1);
    const auto state = require_value(fixture.memory->inspect({backing_a, 0x20, 0x10}));
    CHECK(std::ranges::none_of(state, [](const DirtyRange& range) { return range.gpu_dirty; }));
}

// A GPU write committed through a foreign authority is pulled back through a
// registry resolve, so the view still serves the produced bytes.
void test_foreign_gpu_authority_is_resolved_once() {
    Fixture fixture(0x40000);
    auto first = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)first;

    struct ForeignStorage final : ResourceMemoryStorage {
        BackingRange coverage() const noexcept override { return bytes; }
        RendererStatus read_gpu(BackingRange range,
                                std::span<std::byte> destination) const override {
            if (destination.size() != range.size ||
                !fill.has_value() ||
                range.offset < bytes.offset ||
                range.offset - bytes.offset > bytes.size ||
                range.size > bytes.size - (range.offset - bytes.offset)) {
                return RendererError({CommandStatus::MemoryFault, "bounds"});
            }
            std::fill(destination.begin(), destination.end(), *fill);
            return renderer_success();
        }
        BackingRange bytes{backing_a, 0x28, 0x8};
        std::optional<std::byte> fill;
    };
    auto foreign = std::make_shared<ForeignStorage>();
    foreign->fill = std::byte{0x44};

    TokenSource token(2);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1028, 0x8, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({foreign})));
    token.submit();
    token.succeed();

    const auto writes_before = fixture.backend->writes;
    auto second = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    const auto image = read_image(second, {backing_a, 0x0, 0x40});
    CHECK(filled({image.begin() + 0x28, image.begin() + 0x30}, std::byte{0x44}));
    CHECK(filled({image.begin(), image.begin() + 0x28}, std::byte{0x11}));
    // The resolve delivered the produced bytes to guest memory exactly once.
    CHECK(fixture.backend->writes == writes_before + 1);
    CHECK(fixture.backend->find(backing_a)->bytes[0x28] == std::byte{0x44});

    // The resolved interval is now synced: another acquire performs no reads.
    const auto reads_before = fixture.backend->reads;
    auto third = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)third;
    CHECK(fixture.backend->reads == reads_before);
}

// Draw A keeps its recorded uniform version after draw B updates the same
// guest range, even while A's consuming submission is still in flight.
void test_uniform_version_a_is_retained_for_draw_a() {
    Fixture fixture(0x40000);
    auto a = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_slice(a, 0x40), std::byte{0x11}));

    // Draw A consumes its slice inside a submission that is submitted but
    // deliberately not completed yet.
    // Draw B updates the same guest range before draw A is submitted; the
    // recorded slice of draw A must keep the bytes it was created with.
    fixture.guest_write(0x1000, 0x40, std::byte{0x22});

    auto b = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_slice(b, 0x40), std::byte{0x22}));
    CHECK(filled(read_slice(a, 0x40), std::byte{0x11}));
    CHECK(a.storage != b.storage);

    // Draw A then consumes its slice inside a submission that stays in
    // flight; the slice keeps serving version A until the work completes.
    TokenSource draw_a(3);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), ResourceAccess::Read}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, draw_a.token));
    CHECK(succeeded(reservation->commit({a.storage})));
    draw_a.submit();
    CHECK(filled(read_slice(a, 0x40), std::byte{0x11}));
    draw_a.succeed();
}

// Ring blocks are never recycled while a retained resident or an unfinished
// submission may still read them.
void test_upload_ring_waits_for_delayed_completion() {
    Fixture fixture(0x40000);
    auto a = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Uniform,
        ResourceAccess::Read, 0x10)));
    CHECK(filled(read_slice(a, 0x40), std::byte{0x11}));

    TokenSource draw(4);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), ResourceAccess::Read}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, draw.token));
    CHECK(succeeded(reservation->commit({a.storage})));
    draw.submit();

    // Push more than a ring block of immutable uploads through while the
    // consuming submission is still unfinished.
    std::vector<std::byte> payload(0x8000, std::byte{0x55});
    for (unsigned i = 0; i < 4; ++i) {
        auto slice = require_value(fixture.cache->upload(payload, 0x100));
        CHECK(slice.size == payload.size());
        CHECK(filled(read_slice(slice, payload.size()), std::byte{0x55}));
    }
    // Draw A's bytes survived every later allocation.
    CHECK(filled(read_slice(a, 0x40), std::byte{0x11}));
    draw.succeed();

    // Host-only uploads are not guest-backed and carry no key.
    auto plain = require_value(fixture.cache->upload(payload, 0x100));
    CHECK(plain.key.bytes == BackingRange{});
    CHECK(plain.size == payload.size());
    CHECK(filled(read_slice(plain, payload.size()), std::byte{0x55}));
    CHECK(filled(read_slice(a, 0x40), std::byte{0x11}));
}

void test_invalidate_resolve_trim_and_budget() {
    Fixture fixture(0x40000);
    auto view = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)view;

    // Resolving delegates current dirty bytes to L2 and drops cached lookup;
    // clean cached bytes never overwrite the guest.
    CHECK(succeeded(fixture.cache->resolve(view.key, ResolveReason::Capture)));
    view = {};
    const auto reads_before = fixture.backend->reads;
    auto again = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)again;
    CHECK(fixture.backend->reads > reads_before);
    CHECK(succeeded(fixture.cache->resolve(
        BufferKey{BackingRange{backing_a, 0x0, 0x100}}, ResolveReason::Capture)));
    again = {};

    // Invalidating a live GPU-dirty view refuses rather than losing produced
    // bytes; after the writer completes it succeeds.
    auto write_view = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Write, 1)));
    TokenSource token(5);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({write_view.storage})));
    token.submit();
    CHECK(!succeeded(fixture.cache->invalidate(BackingRange{backing_a, 0x0, 0x40})));
    token.succeed();
    CHECK(succeeded(fixture.cache->invalidate(BackingRange{backing_a, 0x0, 0x40})));
    write_view = {};
    const auto reads_after_invalidate = fixture.backend->reads;
    auto rebuilt = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)rebuilt;
    CHECK(fixture.backend->reads > reads_after_invalidate);

    // Trim evicts the least recently used guest-backed view but keeps a view
    // that still holds GPU authority.
    auto keep = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), BufferUse::Storage,
        ResourceAccess::Write, 1)));
    TokenSource writer(6);
    const std::array write_requests{GpuAccessRequest{
        fixture.retain(0x1020, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto write_reservation =
        require_value(fixture.memory->reserve_accesses(write_requests, writer.token));
    CHECK(succeeded(write_reservation->commit({keep.storage})));
    writer.submit();
    CHECK(succeeded(fixture.cache->trim(0)));
    // The authoritative view survives trim: resolving it fails on the delayed
    // writer, not because the view is gone.
    auto delayed = fixture.cache->resolve(keep.key, ResolveReason::IncompatibleAlias);
    CHECK(std::holds_alternative<RendererError>(delayed));
    CHECK(std::get<RendererError>(delayed).command().status == CommandStatus::Stopped);
    writer.succeed();
    CHECK(succeeded(fixture.cache->resolve(keep.key, ResolveReason::IncompatibleAlias)));

    // Requests that violate alignment, bounds or budget fail with structured
    // errors.
    RendererError misaligned = std::get<RendererError>(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x10, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 3)));
    CHECK(misaligned.command().status == CommandStatus::Unsupported);
    auto emptying = fixture.retain(0x1000, 0x40, ResourceAccess::Read);
    emptying.bytes.size = 0;
    RendererError empty = std::get<RendererError>(
        fixture.cache->acquire(request(emptying, BufferUse::Vertex,
                                       ResourceAccess::Read, 1)));
    CHECK(empty.command().status == CommandStatus::Unsupported);
    RendererError no_lease = std::get<RendererError>(
        fixture.cache->acquire(request(RetainedRange{}, BufferUse::Vertex,
                                       ResourceAccess::Read, 1)));
    CHECK(no_lease.command().status == CommandStatus::Unsupported);
    const std::array<std::byte, 1> lone{std::byte{0}};
    RendererError host_misaligned =
        std::get<RendererError>(fixture.cache->upload(lone, 3));
    CHECK(host_misaligned.command().status == CommandStatus::Unsupported);
}

void test_budget_is_enforced_and_domain_matches() {
    Fixture fixture(0x40);
    const auto domain = fixture.cache->domain();
    CHECK(domain.device != 0);
    CHECK(domain.memory == static_cast<const ResourceMemory*>(fixture.memory.get()));
    auto first = require_value(fixture.cache->acquire(request(
        fixture.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    (void)first;
    // A range larger than the whole budget can never fit, even after
    // evicting the first view.
    auto second = fixture.cache->acquire(request(
        fixture.retain(0x1040, 0x80, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1));
    CHECK(std::holds_alternative<RendererError>(second));
    CHECK(std::get<RendererError>(second).command().status == CommandStatus::BudgetExceeded);
    // A larger budget holds both views of the same backing.
    Fixture roomy(0x40000);
    auto a = require_value(roomy.cache->acquire(request(
        roomy.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    auto b = require_value(roomy.cache->acquire(request(
        roomy.retain(0x1040, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1)));
    CHECK(a.key.bytes != b.key.bytes);
}

void test_factory_rejects_invalid_construction() {
    auto backend = std::make_shared<Backend>();
    auto memory = require_value(make_resource_memory(
        {7, backend, CoherenceMode::SubmissionBoundary, 0x10}));
    CHECK(std::holds_alternative<RendererError>(make_buffer_cache({}, 0x100)));
    CHECK(std::holds_alternative<RendererError>(make_buffer_cache(memory, 0)));
}

void regression_retained_read_scales_with_alias_count() {
    constexpr std::uint64_t alias_count = 512;
    constexpr std::uint64_t address_space = 17;
    constexpr std::uint64_t gpu_base = 0x10000;
    constexpr std::uint64_t guest_base = 0x800000;
    auto backend = std::make_shared<Backend>();
    auto counters = std::make_shared<ResourceMemoryWorkCounters>();
    auto memory = require_value(make_resource_memory({address_space, backend,
        CoherenceMode::SubmissionBoundary, 0x10, counters}));
    for (std::uint64_t index = 0; index < alias_count; ++index) {
        const BackingId backing{3, index + 1, 1};
        const auto fill = index + 1 == alias_count ? std::byte{0x6a} : std::byte{0x19};
        backend->add(backing, 0x10, fill);
        CHECK(succeeded(memory->register_backing(
            {backing, 0x10, std::make_shared<ResourceLease>()})));
        CHECK(succeeded(memory->register_mapping({{address_space, index + 1, 1}, backing,
            {gpu_base + index * 0x10, 0x10, 0, guest_base + index * 0x10, 0x10, 0x10}})));
    }
    auto cache = require_value(make_buffer_cache(memory, 0x1000));
    const auto address = gpu_base + (alias_count - 1) * 0x10;
    auto retained = require_value(memory->retain(address, 0x10, ResourceAccess::Read));
    *counters = {};
    auto resident = require_value(cache->acquire(request(
        retained, BufferUse::Vertex, ResourceAccess::Read, 1)));
    CHECK(filled(read_image(resident, retained.bytes), std::byte{0x6a}));
    CHECK(backend->reads == 1);
    CHECK(counters->mapping_comparisons < 128);
    CHECK(counters->backing_comparisons < 256);
    CHECK(counters->interval_comparisons < 128);
    CHECK(counters->interval_visits < 128);
    CHECK(counters->boundary_backings == 0);
}


// Review regressions deliberately use backing-relative synthetic intervals.
ResidentBuffer acquire(Fixture& fixture, std::uint64_t address, std::uint64_t size,
                       BufferUse use = BufferUse::Vertex, std::uint64_t alignment = 1,
                       ResourceAccess access = ResourceAccess::Read) {
    return require_value(fixture.cache->acquire(request(
        fixture.retain(address, size, ResourceAccess::ReadWrite), use, access, alignment)));
}

void regression_1_realignment() {
    Fixture f(0x40000);
    auto old = acquire(f, 0x1000, 0x40);
    CHECK(stopped(f.cache->acquire(request(f.retain(0x1010, 0x10, ResourceAccess::Read),
        BufferUse::Vertex, ResourceAccess::Read, 0x20))));
    CHECK(filled(read_image(old, {backing_a, 0, 0x40}), std::byte{0x11}));
    old = {};
    auto aligned = acquire(f, 0x1010, 0x10, BufferUse::Vertex, 0x20);
    CHECK(aligned.offset % 0x20 == 0);
    CHECK(filled(read_image(aligned, {backing_a, 0, 0x40}), std::byte{0x11}));
}

void regression_2_ring_alignment() {
    Fixture f(0x80000);
    { auto a = acquire(f, 0x1000, 0x10, BufferUse::Uniform, 0x10); }
    f.guest_write(0x1000, 0x10, std::byte{0x22});
    auto b = acquire(f, 0x1000, 0x10, BufferUse::Uniform, 0x20000);
    CHECK(b.offset % 0x20000 == 0);
    CHECK(filled(read_slice(b, 0x10), std::byte{0x22}));
}

void regression_3_trim_overlap() {
    Fixture f(0x1000);
    { auto a = acquire(f, 0x1000, 0x40, BufferUse::Uniform); }
    auto grown = acquire(f, 0x1000, 0x80);
    CHECK(filled(read_image(grown, {backing_a, 0, 0x80}), std::byte{0x11}));
}

void regression_4_backing_identity() {
    Fixture f(0x40000);
    auto b = acquire(f, 0x2000, 0x40);
    CHECK(filled(read_image(b, {backing_b, 0, 0x40}), std::byte{0x33}));
    CHECK(succeeded(f.cache->resolve(b.key, ResolveReason::Capture)));
    CHECK(filled(f.backend->find(backing_a)->bytes, std::byte{0x11}));
}

void regression_5_interval_versions() {
    Fixture f(0x40000);
    auto a = acquire(f, 0x1000, 0x40);
    f.guest_write(0x1020, 0x10, std::byte{0x22});
    f.guest_write(0x1000, 0x10, std::byte{0x33});
    auto b = acquire(f, 0x1000, 0x40);
    CHECK(filled(read_image(b, {backing_a, 0x20, 0x10}), std::byte{0x22}));
    CHECK(filled(read_image(b, {backing_a, 0, 0x10}), std::byte{0x33}));
    // Independently stale entries must not inherit one another's watermark.
    Fixture merged(0x40000);
    { auto low = acquire(merged, 0x1000, 0x10); }
    { auto high = acquire(merged, 0x1020, 0x10); }
    merged.guest_write(0x1020, 0x10, std::byte{0x44});
    merged.guest_write(0x1000, 0x10, std::byte{0x55});
    { auto low = acquire(merged, 0x1000, 0x10); }
    auto both = acquire(merged, 0x1000, 0x30);
    CHECK(filled(read_image(both, {backing_a, 0x20, 0x10}), std::byte{0x44}));
}

void regression_6_pending_copy() {
    Fixture f(0x40000);
    auto producer = acquire(f, 0x1020, 0x10, BufferUse::Storage, 1, ResourceAccess::Write);
    TokenSource token(60);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1020, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({producer.storage})));
    for (unsigned phase = 0; phase < 2; ++phase) {
        for (auto use : {BufferUse::Uniform, BufferUse::Vertex}) {
            const auto start = use == BufferUse::Uniform ? 0x1020 : 0x1010;
            const auto size = use == BufferUse::Uniform ? 0x10 : 0x20;
            auto result = f.cache->acquire(request(f.retain(start, size, ResourceAccess::Read),
                                                 use, ResourceAccess::Read, 1));
            CHECK(std::holds_alternative<RendererError>(result));
            if (auto* error = std::get_if<RendererError>(&result))
                CHECK(error->command().status == CommandStatus::Stopped);
        }
        if (phase == 0) token.submit();
    }
    token.succeed();
    producer = {};
    CHECK(succeeded(f.memory->resolve({backing_a, 0x20, 0x10}, ResolveReason::Capture)));
    auto done = acquire(f, 0x1010, 0x20);
    CHECK(filled(read_image(done, {backing_a, 0x10, 0x20}), std::byte{0x11}));
}

struct ForeignAuthority final : ResourceMemoryStorage {
    BackingRange bytes{backing_a, 0, 0x40};
    std::byte value{0x44};
    BackingRange coverage() const noexcept override { return bytes; }
    RendererStatus read_gpu(BackingRange range, std::span<std::byte> output) const override {
        CHECK(range.backing == bytes.backing);
        CHECK(range.offset >= bytes.offset && range.size <= bytes.size - (range.offset - bytes.offset));
        CHECK(range.size == output.size());
        std::fill(output.begin(), output.end(), value);
        return renderer_success();
    }
};

void regression_7_canceled_authority() {
    Fixture f(0x40000);
    { auto abandoned = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write); }
    auto foreign = std::make_shared<ForeignAuthority>();
    TokenSource token(70);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x40, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({foreign})));
    token.submit(); token.succeed();
    auto current = acquire(f, 0x1000, 0x40);
    CHECK(filled(read_image(current, {backing_a, 0, 0x40}), std::byte{0x44}));
    Fixture clean(0x40);
    { auto abandoned = acquire(clean, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write); }
    CHECK(succeeded(clean.cache->trim(0)));
    auto next = acquire(clean, 0x1040, 0x40);
    CHECK(next.size == 0x40);
}

void regression_8_resolve_latest() {
    Fixture f(0x40000);
    auto old = acquire(f, 0x1000, 0x40);
    f.guest_write(0x1010, 0x10, std::byte{0x66});
    const auto writes = f.backend->writes;
    CHECK(succeeded(f.cache->resolve(old.key, ResolveReason::Capture)));
    CHECK(f.backend->find(backing_a)->bytes[0x10] == std::byte{0x66});
    CHECK(f.backend->writes == writes);
    old = {};
    auto stale = acquire(f, 0x1000, 0x40);
    auto foreign = std::make_shared<ForeignAuthority>();
    TokenSource token(80);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x40, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({foreign})));
    token.submit();
    f.backend->on_wait = [&] { foreign->value = std::byte{0x77}; token.succeed(); };
    CHECK(succeeded(f.cache->resolve(stale.key, ResolveReason::Capture)));
    CHECK(filled({f.backend->find(backing_a)->bytes.begin(),
                  f.backend->find(backing_a)->bytes.begin() + 0x40}, std::byte{0x77}));
}

void regression_10_registry_budget() {
    Fixture f(0x40);
    auto a = acquire(f, 0x1000, 0x40);
    std::weak_ptr<const ResourceLease> weak = a.storage;
    TokenSource token(100);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x40, ResourceAccess::Read), ResourceAccess::Read}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({a.storage})));
    token.submit();
    a = {};
    reservation.reset();
    CHECK(!weak.expired());
    CHECK(succeeded(f.cache->trim(0)));
    auto b = f.cache->acquire(request(f.retain(0x1040, 0x40, ResourceAccess::Read),
                                      BufferUse::Vertex, ResourceAccess::Read, 1));
    CHECK(std::holds_alternative<RendererError>(b));
    if (auto* error = std::get_if<RendererError>(&b))
        CHECK(error->command().status == CommandStatus::BudgetExceeded);
    token.succeed();
    CHECK(succeeded(f.memory->submission_boundary()));
    auto after = acquire(f, 0x1040, 0x40);
    CHECK(after.size == 0x40);
}

void regression_11_reclaim_admission() {
    Fixture f(0x40);
    { auto a = acquire(f, 0x1000, 0x40); }
    auto b = acquire(f, 0x1040, 0x40);
    CHECK(filled(read_image(b, {backing_a, 0x40, 0x40}), std::byte{0x11}));
}

void regression_12_partial_mapping() {
    Fixture f(0x40000);
    CHECK(succeeded(f.memory->unmap(mapping_b)));
    CHECK(succeeded(f.memory->unmap(mapping_a)));
    CHECK(succeeded(f.memory->register_mapping({{7, 3, 1}, backing_a,
        {0x3000, 0x10, 0, 0x8000, 0x100, 0x10}})));
    CHECK(succeeded(f.memory->register_mapping({{7, 4, 1}, backing_a,
        {0x4000, 0x10, 0x20, 0x8000, 0x100, 0x10}})));
    f.guest_write(0x4000, 0x10, std::byte{0x88});
    auto a = acquire(f, 0x4000, 0x10);
    CHECK(filled(read_image(a, {backing_a, 0x20, 0x10}), std::byte{0x88}));
}

void regression_13_registry_only_ring() {
    Fixture f(0x40000);
    auto a = acquire(f, 0x1000, 0x40, BufferUse::Uniform, 0x10);
    std::weak_ptr<const ResourceLease> weak = a.storage;
    TokenSource token(130);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x40, ResourceAccess::Read), ResourceAccess::Read}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({a.storage})));
    token.submit(); a = {}; reservation.reset();
    CHECK(!weak.expired());
    for (unsigned i = 0; i < 8; ++i) {
        auto b = acquire(f, 0x1040, 0x40, BufferUse::Uniform, 0x10);
        CHECK(succeeded(f.cache->trim(0)));
    }
    ResidentBuffer observed; observed.storage = weak.lock();
    CHECK(observed.storage != nullptr);
    CHECK(filled(read_slice(observed, 0x40), std::byte{0x11}));
    observed = {};
    token.succeed();
    CHECK(succeeded(f.memory->submission_boundary()));
    CHECK(weak.expired());
}

void regression_ring_capacity_threshold() {
    for (const auto budget : {0xffffU, 0x10000U, 0x10001U}) {
        Fixture f(budget);
        auto a = acquire(f, 0x1000, 0x40, BufferUse::Uniform);
        CHECK(filled(read_slice(a, 0x40), std::byte{0x11}));
    }
}

void regression_replaced_allocation_accounting() {
    Fixture f(0xc0);
    auto a = acquire(f, 0x1000, 0x40);
    TokenSource token(101);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x40, ResourceAccess::Read), ResourceAccess::Read}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
    CHECK(succeeded(reservation->commit({a.storage})));
    token.submit();
    std::weak_ptr<const ResourceLease> old = a.storage;
    a = {}; reservation.reset();
    CHECK(stopped(f.cache->acquire(request(f.retain(0x1000, 0x80, ResourceAccess::Read),
        BufferUse::Vertex, ResourceAccess::Read, 1))));
    CHECK(!old.expired());
    CHECK(succeeded(f.cache->invalidate({backing_a, 0, 0x40})));
    auto other = acquire(f, 0x1040, 0x80);
    auto rejected = f.cache->acquire(request(f.retain(0x10c0, 0x40, ResourceAccess::Read),
        BufferUse::Vertex, ResourceAccess::Read, 1));
    CHECK(std::holds_alternative<RendererError>(rejected));
    if (auto* failure = std::get_if<RendererError>(&rejected))
        CHECK(failure->command().status == CommandStatus::BudgetExceeded);
    token.succeed();
    CHECK(succeeded(f.memory->submission_boundary()));
    CHECK(old.expired());
    auto admitted = acquire(f, 0x10c0, 0x40);
    CHECK(admitted.size == 0x40);
}

void regression_superseded_authority() {
    Fixture f(0x40000);
    auto local = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x40, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    TokenSource first(71);
    auto reservation = require_value(f.memory->reserve_accesses(requests, first.token));
    CHECK(succeeded(reservation->commit({local.storage})));
    first.submit(); first.succeed();
    auto observed = acquire(f, 0x1000, 0x40);
    CHECK(observed.storage == local.storage);
    TokenSource second(72);
    auto foreign = std::make_shared<ForeignAuthority>();
    reservation = require_value(f.memory->reserve_accesses(requests, second.token));
    CHECK(succeeded(reservation->commit({foreign})));
    second.submit(); second.succeed();
    auto latest = acquire(f, 0x1000, 0x40);
    CHECK(filled(read_image(latest, {backing_a, 0, 0x40}), std::byte{0x44}));
}

// A rejected growth must leave the original binding and lookup untouched.
// Retrying prepares the union before publishing either overlapping binding.
void regression_prepared_overlap() {
    for (const bool write_first : {false, true}) {
        Fixture f(0x40000);
        const auto first_access = write_first ? ResourceAccess::Write : ResourceAccess::Read;
        const auto second_access = write_first ? ResourceAccess::Read : ResourceAccess::Write;
        auto a = acquire(f, 0x1000, 0x20, BufferUse::Storage, 1, first_access);
        const auto reads = f.backend->reads;
        const auto grow = [&] {
            return f.cache->acquire(request(f.retain(0x1010, 0x20, ResourceAccess::ReadWrite),
                BufferUse::Storage, second_access, 1));
        };
        CHECK(stopped(grow()));
        CHECK(f.backend->reads == reads);
        auto unchanged = acquire(f, 0x1000, 0x20, BufferUse::Storage, 1, first_access);
        CHECK(unchanged.storage == a.storage);
        CHECK(filled(read_image(a, {backing_a, 0, 0x20}), std::byte{0x11}));
        unchanged = {};
        // Removing lookup must not allow a retained old binding to alias a
        // new allocation, whether invalidation or resolve removed the entry.
        if (write_first) CHECK(succeeded(f.cache->resolve(a.key, ResolveReason::Capture)));
        else CHECK(succeeded(f.cache->invalidate(a.key.bytes)));
        CHECK(stopped(grow()));
        a = {};
        auto union_view = acquire(f, 0x1000, 0x30, BufferUse::Storage);
        a = acquire(f, 0x1000, 0x20, BufferUse::Storage, 1, first_access);
        auto b = require_value(grow());
        CHECK(a.storage == b.storage && a.storage == union_view.storage);
        CHECK(b.offset == a.offset + 0x10);
        TokenSource token(140);
        const std::array requests{
            GpuAccessRequest{f.retain(0x1000, 0x20, ResourceAccess::ReadWrite), first_access},
            GpuAccessRequest{f.retain(0x1010, 0x20, ResourceAccess::ReadWrite), second_access}};
        auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
        CHECK(succeeded(reservation->commit({a.storage, b.storage})));
        token.submit(); token.succeed();
    }
}

void regression_cache_domain_overlap() {
    Fixture f(0x40000);
    auto a = acquire(f, 0x1000, 0x20, BufferUse::Storage);
    auto second = require_value(make_buffer_cache(f.memory, 0x40000));
    CHECK(second->domain() == f.cache->domain());
    const auto grow = [&] {
        return second->acquire(request(f.retain(0x1010, 0x20, ResourceAccess::ReadWrite),
            BufferUse::Storage, ResourceAccess::Write, 1));
    };
    CHECK(stopped(grow()));
    a = {};
    // Another cache's lookup also prevents duplicate overlapping residency.
    CHECK(stopped(grow()));
    a = acquire(f, 0x1000, 0x20, BufferUse::Storage);
    f.cache.reset();
    second.reset();
    // Only the retained binding now owns the old domain's exclusion records.
    second = require_value(make_buffer_cache(f.memory, 0x40000));
    CHECK(stopped(grow()));
    a = {};
    auto admitted = require_value(grow());
    CHECK(admitted.size == 0x20);

    // Domain identity must not create a registry/authority ownership cycle.
    TokenSource token(150);
    {
        const std::array requests{GpuAccessRequest{
            f.retain(0x1010, 0x20, ResourceAccess::ReadWrite), ResourceAccess::Write}};
        auto reservation = require_value(f.memory->reserve_accesses(requests, token.token));
        CHECK(succeeded(reservation->commit({admitted.storage})));
    }
    token.submit(); token.succeed();
    std::weak_ptr<ResourceMemoryRegistry> weak_memory = f.memory;
    std::weak_ptr<const ResourceLease> weak_image = admitted.storage;
    admitted = {};
    second.reset();
    f.memory.reset();
    CHECK(weak_memory.expired());
    CHECK(weak_image.expired());
}

void regression_refresh_after_partial_remap() {
    Fixture f(0x40000);
    { auto old = acquire(f, 0x1000, 0x40); }
    CHECK(succeeded(f.memory->unmap(mapping_a)));
    CHECK(succeeded(f.memory->register_mapping({{7, 3, 1}, backing_a,
        {0x3000, 0x10, 0x20, 0x8000, 0x100, 0x10}})));
    // Native guest writes are discovered through the submission epoch, with
    // unchanged interval versions and no mapping covering the obsolete head.
    std::fill(f.backend->find(backing_a)->bytes.begin(),
              f.backend->find(backing_a)->bytes.end(), std::byte{0x88});
    CHECK(succeeded(f.memory->submission_boundary()));
    const auto reads = f.backend->reads;
    {
        auto partial = acquire(f, 0x3000, 0x10);
        CHECK(filled(read_image(partial, {backing_a, 0x20, 0x10}), std::byte{0x88}));
        CHECK(f.backend->reads == reads + 1);
        auto repeated = acquire(f, 0x3000, 0x10);
        CHECK(f.backend->reads == reads + 1);
    }
    // The construction path also refreshes only its valid request, despite
    // carrying forward an old image whose prefix remains unmapped.
    CHECK(succeeded(f.memory->register_mapping({{7, 4, 1}, backing_a,
        {0x4000, 0x20, 0x30, 0x8000, 0x100, 0x10}})));
    {
        auto grown = acquire(f, 0x4000, 0x20);
        CHECK(filled(read_image(grown, {backing_a, 0x30, 0x20}), std::byte{0x88}));
    }
    // Refreshing a subset must not mark the untouched old prefix current.
    CHECK(succeeded(f.memory->register_mapping({{7, 5, 1}, backing_a,
        {0x5000, 0x50, 0, 0x8000, 0x100, 0x10}})));
    auto whole = acquire(f, 0x5000, 0x50);
    CHECK(filled(read_image(whole, {backing_a, 0, 0x50}), std::byte{0x88}));
}

void regression_scoped_refresh_versions() {
    Fixture f(0x40000);
    auto whole = acquire(f, 0x1000, 0x40);
    f.guest_write(0x1010, 0x10, std::byte{0x22});
    const auto reads = f.backend->reads;
    auto middle = acquire(f, 0x1010, 0x10);
    CHECK(f.backend->reads == reads + 1);
    auto again = acquire(f, 0x1000, 0x40);
    CHECK(f.backend->reads == reads + 1);
    auto bytes = read_image(again, {backing_a, 0, 0x40});
    CHECK(filled({bytes.begin(), bytes.begin() + 0x10}, std::byte{0x11}));
    CHECK(filled({bytes.begin() + 0x10, bytes.begin() + 0x20}, std::byte{0x22}));
    CHECK(filled({bytes.begin() + 0x20, bytes.end()}, std::byte{0x11}));
}

void regression_epoch_revalidation() {
    {
        Fixture tight(0x40);
        auto first = acquire(tight, 0x1000, 0x40);
        CHECK(succeeded(tight.memory->submission_boundary()));
        const auto reads = tight.backend->reads;
        auto second = acquire(tight, 0x1000, 0x40);
        CHECK(second.storage == first.storage);
        CHECK(tight.backend->reads == reads + 1);
    }

    // Three exact-size native allocations fit. Revalidation keeps no separate
    // packed baseline, so unchanged epochs do not consume additional budget.
    Fixture f(0xc0);
    auto old = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write);
    const auto reads = f.backend->reads;

    CHECK(succeeded(f.memory->submission_boundary()));
    auto unchanged = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write);
    CHECK(f.backend->reads == reads + 1);
    CHECK(unchanged.storage == old.storage);
    auto repeated = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write);
    CHECK(f.backend->reads == reads + 1);
    CHECK(repeated.storage == old.storage);

    std::fill(f.backend->find(backing_a)->bytes.begin(),
              f.backend->find(backing_a)->bytes.begin() + 0x40, std::byte{0x22});
    CHECK(succeeded(f.memory->submission_boundary()));
    const auto changed_reads = f.backend->reads;
    auto changed = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write);
    CHECK(f.backend->reads == changed_reads + 1);
    CHECK(changed.storage != old.storage);
    CHECK(filled(read_image(old, {backing_a, 0, 0x40}), std::byte{0x11}));
    CHECK(filled(read_image(changed, {backing_a, 0, 0x40}), std::byte{0x22}));

    // Version identity, not byte comparison, handles tracked writes. A
    // same-value store therefore replaces residency while retained consumers
    // continue to observe their original allocation.
    f.guest_write(0x1000, 0x40, std::byte{0x22});
    const auto tracked_reads = f.backend->reads;
    auto tracked = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write);
    CHECK(f.backend->reads == tracked_reads + 1);
    CHECK(tracked.storage != changed.storage);

    TokenSource writer(210);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x40, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, writer.token));
    CHECK(succeeded(reservation->commit({tracked.storage})));
    writer.submit();
    writer.succeed();
    const auto gpu_reads = f.backend->reads;
    auto gpu_current = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Read);
    CHECK(gpu_current.storage == tracked.storage);
    CHECK(f.backend->reads == gpu_reads);

    CHECK(succeeded(f.memory->submission_boundary()));
    auto resolved = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Read);
    CHECK(resolved.storage == tracked.storage);
    CHECK(f.backend->reads == gpu_reads + 1);
}

void regression_partial_epoch_revalidation() {
    {
        Fixture prefix_change(0x80);
        const auto old = acquire(prefix_change, 0x1000, 0x40);
        CHECK(succeeded(prefix_change.memory->submission_boundary()));
        CHECK(acquire(prefix_change, 0x1000, 0x20).storage == old.storage);
        // Once the whole snapshot detects a change, replacement must use all
        // of it, even bytes whose copied epoch was already current.
        prefix_change.backend->find(backing_a)->bytes[0] = std::byte{0x55};
        const auto reads = prefix_change.backend->reads;
        const auto changed = acquire(prefix_change, 0x1000, 0x40);
        CHECK(changed.storage != old.storage);
        CHECK(prefix_change.backend->reads == reads + 1);
        CHECK(read_image(changed, {backing_a, 0, 0x40})[0] == std::byte{0x55});
        CHECK(read_image(old, {backing_a, 0, 0x40})[0] == std::byte{0x11});
    }
    Fixture f(0x80);
    const auto old = acquire(f, 0x1000, 0x40);
    f.backend->find(backing_a)->bytes[0x3f] = std::byte{0x77};
    CHECK(succeeded(f.memory->submission_boundary()));
    const auto reads = f.backend->reads;
    const auto prefix = acquire(f, 0x1000, 0x20);
    CHECK(prefix.storage == old.storage);
    CHECK(f.backend->reads == reads + 1);
    // Revalidating a prefix cannot make the still-uncertain suffix current.
    const auto changed = acquire(f, 0x1000, 0x40);
    CHECK(changed.storage != old.storage);
    CHECK(f.backend->reads == reads + 2);
    CHECK(filled(read_image(old, {backing_a, 0, 0x40}), std::byte{0x11}));
    const auto bytes = read_image(changed, {backing_a, 0, 0x40});
    CHECK(filled({bytes.begin(), bytes.end() - 1}, std::byte{0x11}));
    CHECK(bytes.back() == std::byte{0x77});
    CHECK(acquire(f, 0x1000, 0x40).storage == changed.storage);
    CHECK(f.backend->reads == reads + 2);

    CHECK(succeeded(f.memory->submission_boundary()));
    f.backend->fail_read = true;
    CHECK(std::holds_alternative<RendererError>(f.cache->acquire(request(
        f.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Vertex,
        ResourceAccess::Read, 1))));
    f.backend->fail_read = false;
    CHECK(acquire(f, 0x1000, 0x40).storage == changed.storage);
    CHECK(f.backend->reads == reads + 4);
}

void regression_mixed_gpu_epoch_revalidation() {
    Fixture f(0x40);
    auto whole = acquire(f, 0x1000, 0x40, BufferUse::Storage, 1, ResourceAccess::Write);
    CHECK(succeeded(f.memory->submission_boundary()));
    auto prefix = acquire(f, 0x1000, 0x20, BufferUse::Storage, 1, ResourceAccess::Write);
    TokenSource writer(211);
    const std::array requests{GpuAccessRequest{
        f.retain(0x1000, 0x20, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(f.memory->reserve_accesses(requests, writer.token));
    CHECK(succeeded(reservation->commit({prefix.storage})));
    writer.submit();
    const auto reads = f.backend->reads;
    f.backend->on_read = [&](BackingRange range) {
        CHECK(range == BackingRange{backing_a, 0x20, 0x20});
    };
    auto acquired = f.cache->acquire(request(
        f.retain(0x1000, 0x40, ResourceAccess::Read), BufferUse::Storage,
        ResourceAccess::Read, 1));
    CHECK(std::holds_alternative<ResidentBuffer>(acquired));
    if (const auto* current = std::get_if<ResidentBuffer>(&acquired)) {
        CHECK(current->storage == whole.storage);
        CHECK(current->dependencies.size() == 1);
        CHECK(f.backend->reads == reads + 1);
        CHECK(acquire(f, 0x1000, 0x40).storage == whole.storage);
        CHECK(f.backend->reads == reads + 1);
    }
    CHECK(writer.token.snapshot().phase == SubmissionPhase::Submitted);
    writer.succeed();
}

void test_mapping_invalidation_refreshes_residency_across_generations() {
    Fixture fixture(0x40000);
    const auto old_range = fixture.retain(
        0x1000, 0x40, ResourceAccess::ReadWrite);
    auto resident = require_value(fixture.cache->acquire(request(
        old_range, BufferUse::Storage, ResourceAccess::Write, 1)));
    CHECK(filled(read_image(resident, {backing_a, 0, 0x40}), std::byte{0x11}));

    TokenSource first(200);
    const std::array first_request{
        GpuAccessRequest{old_range, ResourceAccess::Write}};
    auto reservation = require_value(
        fixture.memory->reserve_accesses(first_request, first.token));
    CHECK(succeeded(reservation->commit({resident.storage})));
    first.submit();
    first.succeed();
    const auto dirty_version = resident.uploaded_version.gpu_write + 1;
    const auto writes = fixture.backend->writes;
    const std::array removed{mapping_a};
    CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    CHECK(fixture.backend->writes == writes + 1);
    CHECK(!succeeded(fixture.memory->validate_retained(
        old_range, ResourceAccess::Read)));
    CHECK(filled(read_image(resident, {backing_a, 0, 0x40}), std::byte{0x11}));

    const MappingId remapped{7, 1, 2};
    const Mapping same_bytes{0x1000, 0x100, 0, 0x8000, 0x100, 0x10};
    CHECK(succeeded(fixture.memory->register_mapping(
        {remapped, backing_a, same_bytes})));
    const auto current_range = fixture.retain(
        0x1000, 0x40, ResourceAccess::ReadWrite);
    CHECK(current_range.mapping == remapped);
    auto current = require_value(fixture.cache->acquire(request(
        current_range, BufferUse::Storage, ResourceAccess::Write, 1)));
    CHECK(current.uploaded_version.gpu_write == dirty_version);
    CHECK(filled(read_image(current, {backing_a, 0, 0x40}), std::byte{0x11}));

    TokenSource second(201);
    const std::array second_request{
        GpuAccessRequest{current_range, ResourceAccess::Write}};
    reservation = require_value(
        fixture.memory->reserve_accesses(second_request, second.token));
    CHECK(succeeded(reservation->commit({current.storage})));
    second.submit();
    second.succeed();
    const auto writes_before_discard = fixture.backend->writes;
    const std::array discarded{backing_a};
    CHECK(succeeded(fixture.memory->invalidate_mappings(
        std::span<const MappingId>{}, discarded)));
    CHECK(fixture.backend->writes == writes_before_discard);
    CHECK(!fixture.memory->has_backing(backing_a));
    CHECK(!succeeded(fixture.memory->validate_retained(
        current_range, ResourceAccess::Read)));

    const BackingId replacement_backing{1, 1, 2};
    const MappingId replacement_mapping{7, 1, 3};
    fixture.backend->add(replacement_backing, 0x100, std::byte{0x77});
    CHECK(succeeded(fixture.memory->register_backing(
        {replacement_backing, 0x100, std::make_shared<ResourceLease>()})));
    CHECK(succeeded(fixture.memory->register_mapping(
        {replacement_mapping, replacement_backing, same_bytes})));
    const auto replacement_range = fixture.retain(
        0x1000, 0x40, ResourceAccess::Read);
    CHECK(replacement_range.mapping == replacement_mapping);
    auto replacement = require_value(fixture.cache->acquire(request(
        replacement_range, BufferUse::Vertex, ResourceAccess::Read, 1)));
    CHECK(replacement.storage != resident.storage);
    CHECK(replacement.storage != current.storage);
    CHECK(replacement.uploaded_version == ResourceVersion{});
    CHECK(filled(read_image(
        replacement, {replacement_backing, 0, 0x40}), std::byte{0x77}));
    CHECK(filled(read_image(resident, {backing_a, 0, 0x40}), std::byte{0x11}));
}

// Adjacent virtual mappings can name unrelated allocations and nonzero backing
// offsets. The snapshot must preserve virtual order rather than physical order.
void snapshot_adjacent_mappings(Fixture& fixture, bool same_backing = false) {
    CHECK(succeeded(fixture.memory->unmap(mapping_a)));
    CHECK(succeeded(fixture.memory->unmap(mapping_b)));
    CHECK(succeeded(fixture.memory->register_mapping({{7, 3, 1}, backing_a,
        {0x3000, 0x40, 0x20, 0x8000, 0x100, 0x10}})));
    CHECK(succeeded(fixture.memory->register_mapping({{7, 4, 1}, same_backing ? backing_a : backing_b,
        {0x3040, 0x40, 0x60, same_backing ? 0x8000u : 0x9000u, 0x100, 0x10}})));
}

void test_snapshot_preserves_mapping_order_and_offsets() {
    for (bool same_backing : {false, true}) {
        Fixture fixture(0x40000);
        snapshot_adjacent_mappings(fixture, same_backing);
        for (std::size_t i = 0; i < 0x100; ++i) {
            fixture.backend->find(backing_a)->bytes[i] = std::byte(i);
            fixture.backend->find(backing_b)->bytes[i] = std::byte(i ^ 0x80);
        }
        const auto bytes = require_value(snapshot_buffer_range(*fixture.memory, 0x3028, 0x38));
        CHECK(bytes.size() == 0x38);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            const auto expected = i < 0x18 ? 0x48 + i :
                ((0x60 + i - 0x18) ^ (same_backing ? 0u : 0x80u));
            CHECK(bytes[i] == std::byte(expected));
        }
        CHECK(fixture.backend->reads == 2);
        CHECK(succeeded(fixture.memory->unmap({7, 3, 1})));
        CHECK(succeeded(fixture.memory->unmap({7, 4, 1})));
    }
}

void test_snapshot_preflights_holes_padding_and_bounds() {
    Fixture fixture(0x40000);
    const auto rejects_without_reads = [&](std::uint64_t address, std::uint64_t size,
                                          CommandStatus status = CommandStatus::MemoryFault) {
        const auto reads = fixture.backend->reads;
        const auto result = snapshot_buffer_range(*fixture.memory, address, size);
        CHECK(std::holds_alternative<RendererError>(result));
        if (const auto* failure = std::get_if<RendererError>(&result))
            CHECK(failure->command().status == status);
        CHECK(fixture.backend->reads == reads);
    };
    rejects_without_reads(0x10f0, 0x1020);
    rejects_without_reads(0x1000, 0);
    rejects_without_reads(UINT64_MAX - 3, 8);
    rejects_without_reads(0x1000, (256u << 20) + 1, CommandStatus::BudgetExceeded);
    CHECK(succeeded(fixture.memory->register_mapping({{7, 3, 1}, backing_a,
        {0x3000, 0x100, 0xc0, 0x8000, 0x100, 0x10}})));
    rejects_without_reads(0x3030, 0x20);
    rejects_without_reads(0x3040, 4);
    // Failed preflight must release every earlier fragment's mapping pin.
    CHECK(succeeded(fixture.memory->unmap(mapping_a)));
    CHECK(succeeded(fixture.memory->unmap({7, 3, 1})));
}

void test_snapshot_propagates_valid_fragment_read_failure() {
    Fixture fixture(0x40000);
    snapshot_adjacent_mappings(fixture);
    fixture.backend->on_read = [&](BackingRange range) {
        if (range.backing == backing_b) fixture.backend->fail_read = true;
    };
    const auto result = snapshot_buffer_range(*fixture.memory, 0x3028, 0x38);
    CHECK(std::holds_alternative<RendererError>(result));
    if (const auto* failure = std::get_if<RendererError>(&result))
        CHECK(failure->command().status == CommandStatus::MemoryFault);
    CHECK(fixture.backend->reads == 2);
    fixture.backend->fail_read = false;
    fixture.backend->on_read = {};
    CHECK(succeeded(fixture.memory->unmap({7, 3, 1})));
    CHECK(succeeded(fixture.memory->unmap({7, 4, 1})));
}

void test_snapshot_resolves_gpu_fragment_before_any_copy() {
    Fixture fixture(0x40000);
    snapshot_adjacent_mappings(fixture);
    auto authority = std::make_shared<ForeignAuthority>();
    authority->bytes = {backing_b, 0x68, 8};
    TokenSource producer(150);
    const std::array requests{GpuAccessRequest{
        fixture.retain(0x3048, 8, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, producer.token));
    CHECK(succeeded(reservation->commit({authority})));
    producer.submit();
    fixture.backend->on_wait = [&] {
        CHECK(fixture.backend->reads == 0);
        authority->value = std::byte{0x77};
        producer.succeed();
    };
    const auto bytes = require_value(snapshot_buffer_range(*fixture.memory, 0x3028, 0x38));
    CHECK(fixture.backend->waits == 1);
    CHECK(fixture.backend->writes == 1);
    CHECK(fixture.backend->reads == 2);
    for (std::size_t i = 0; i < bytes.size(); ++i)
        CHECK(bytes[i] == (i < 0x18 ? std::byte{0x11} :
            (i >= 0x20 && i < 0x28 ? std::byte{0x77} : std::byte{0x33})));
    CHECK(fixture.backend->find(backing_b)->bytes[0x67] == std::byte{0x33});
    CHECK(fixture.backend->find(backing_b)->bytes[0x70] == std::byte{0x33});
    CHECK(fixture.backend->find(backing_b)->bytes[0x68] == std::byte{0x77});
}

void test_snapshot_honors_reservations_and_mapping_generations() {
    Fixture fixture(0x40000);
    snapshot_adjacent_mappings(fixture);
    TokenSource producer(151);
    std::vector requests{GpuAccessRequest{
        fixture.retain(0x3040, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, producer.token));
    CHECK(std::holds_alternative<RendererError>(
        snapshot_buffer_range(*fixture.memory, 0x3028, 0x38)));
    CHECK(fixture.backend->reads == 0);
    reservation.reset();
    requests.clear();
    CHECK(require_value(snapshot_buffer_range(*fixture.memory, 0x3028, 0x38)).size() == 0x38);
    fixture.backend->on_read = {};
    CHECK(succeeded(fixture.memory->unmap({7, 3, 1})));
    const auto reads = fixture.backend->reads;
    CHECK(std::holds_alternative<RendererError>(
        snapshot_buffer_range(*fixture.memory, 0x3028, 0x38)));
    CHECK(fixture.backend->reads == reads);
    CHECK(succeeded(fixture.memory->register_mapping({{7, 3, 2}, backing_b,
        {0x3000, 0x40, 0x20, 0x9000, 0x100, 0x10}})));
    const auto replaced = require_value(snapshot_buffer_range(*fixture.memory, 0x3028, 0x38));
    CHECK(filled(replaced, std::byte{0x33}));
}

void test_constant_snapshot_zeroes_padding_without_shifting_later_bytes() {
    Fixture fixture(0x40000);
    CHECK(succeeded(fixture.memory->register_mapping({{7, 3, 1}, backing_a,
        {0x4000, 0x80, 0xc0, 0x8000, 0x100, 0x10}})));
    CHECK(succeeded(fixture.memory->register_mapping({{7, 4, 1}, backing_b,
        {0x4080, 0x20, 0x30, 0x9000, 0x100, 0x10}})));
    for (std::size_t i = 0; i < 0x100; ++i)
        fixture.backend->find(backing_b)->bytes[i] = std::byte(i ^ 0x80);
    std::vector<BackingRange> copied;
    fixture.backend->on_read = [&](BackingRange range) { copied.push_back(range); };
    const auto bytes = require_value(snapshot_buffer_range(*fixture.memory, 0x4018, 0x80,
        BufferSnapshotPolicy::ZeroMissingConstantBufferBytes));
    CHECK(bytes.size() == 0x80);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const auto expected = i < 0x28 ? std::byte{0x11} :
            i < 0x68 ? std::byte{0} : std::byte((0x30 + i - 0x68) ^ 0x80);
        CHECK(bytes[i] == expected);
    }
    CHECK(copied == std::vector<BackingRange>{
        {backing_a, 0xd8, 0x28}, {backing_b, 0x30, 0x18}});
    const auto reads = fixture.backend->reads;
    CHECK(std::holds_alternative<RendererError>(
        snapshot_buffer_range(*fixture.memory, 0x4018, 0x80)));
    CHECK(fixture.backend->reads == reads);

    copied.clear();
    const auto bordered = require_value(snapshot_buffer_range(*fixture.memory, 0x3ff0, 0xc0,
        BufferSnapshotPolicy::ZeroMissingConstantBufferBytes));
    CHECK(bordered.size() == 0xc0);
    for (std::size_t i = 0; i < bordered.size(); ++i) {
        const auto expected = i >= 0x10 && i < 0x50 ? std::byte{0x11} :
            i >= 0x90 && i < 0xb0 ? std::byte((0x30 + i - 0x90) ^ 0x80) : std::byte{0};
        CHECK(bordered[i] == expected);
    }
    CHECK(copied == std::vector<BackingRange>{
        {backing_a, 0xc0, 0x40}, {backing_b, 0x30, 0x20}});
    const auto reads_after_backed = fixture.backend->reads;
    const auto missing = require_value(snapshot_buffer_range(*fixture.memory, 0x5000, 0x23,
        BufferSnapshotPolicy::ZeroMissingConstantBufferBytes));
    const auto padding = require_value(snapshot_buffer_range(*fixture.memory, 0x4040, 0x40,
        BufferSnapshotPolicy::ZeroMissingConstantBufferBytes));
    CHECK(missing.size() == 0x23 && filled(missing, std::byte{0}));
    CHECK(padding.size() == 0x40 && filled(padding, std::byte{0}));
    CHECK(fixture.backend->reads == reads_after_backed);
}

void test_constant_snapshot_does_not_hide_read_or_gpu_errors() {
    constexpr auto policy = BufferSnapshotPolicy::ZeroMissingConstantBufferBytes;
    {
        Fixture fixture(0x40000);
        fixture.backend->fail_read = true;
        const auto result = snapshot_buffer_range(*fixture.memory, 0xff0, 0x130, policy);
        CHECK(std::holds_alternative<RendererError>(result));
        if (const auto* failure = std::get_if<RendererError>(&result))
            CHECK(failure->command().status == CommandStatus::MemoryFault);
        CHECK(fixture.backend->reads == 1);
    }
    {
        Fixture fixture(0x40000);
        TokenSource producer(152);
        const std::array requests{GpuAccessRequest{
            fixture.retain(0x1030, 0x10, ResourceAccess::ReadWrite), ResourceAccess::Write}};
        auto reservation = require_value(fixture.memory->reserve_accesses(requests, producer.token));
        CHECK(std::holds_alternative<RendererError>(
            snapshot_buffer_range(*fixture.memory, 0xff0, 0x130, policy)));
        CHECK(fixture.backend->reads == 0);
        auto authority = std::make_shared<ForeignAuthority>();
        authority->bytes = {backing_a, 0x30, 0x10};
        CHECK(succeeded(reservation->commit({authority})));
        // An accepted producer and submitted producer without feedback both
        // remain errors even though the surrounding constant-buffer holes zero.
        CHECK(std::holds_alternative<RendererError>(
            snapshot_buffer_range(*fixture.memory, 0xff0, 0x130, policy)));
        CHECK(fixture.backend->reads == 0);
        producer.submit();
        CHECK(std::holds_alternative<RendererError>(
            snapshot_buffer_range(*fixture.memory, 0xff0, 0x130, policy)));
        CHECK(fixture.backend->waits == 1);
        CHECK(fixture.backend->reads == 0 && fixture.backend->writes == 0);
        producer.succeed();
    }
}

// A stamp proof established at copy time skips every guest access for the
// image, and a second acquire in the same epoch revalidates nothing.
void regression_stamp_skips_revalidation() {
    std::shared_ptr<StampRegistry> tracked;
    Fixture f(0x40000, [&](auto inner) {
        tracked = std::make_shared<StampRegistry>(std::move(inner));
        return tracked;
    });
    auto first = acquire(f, 0x1000, 0x40);
    CHECK(filled(read_image(first, {backing_a, 0, 0x40}), std::byte{0x11}));
    CHECK(succeeded(f.memory->submission_boundary()));
    const auto reads = f.backend->reads;
    auto second = acquire(f, 0x1000, 0x40);
    CHECK(f.backend->reads == reads);
    CHECK(second.storage == first.storage);
    CHECK(buffer_cache_statistics(*f.cache).revalidations_skipped == 1);
    // No store happened, so the stamp proof keeps answering clean.
    acquire(f, 0x1000, 0x40);
    CHECK(f.backend->reads == reads);
    CHECK(buffer_cache_statistics(*f.cache).revalidations_skipped == 1);
    CHECK(buffer_cache_statistics(*f.cache).reuploads == 0);
}

// A suspected stamp hashes one snapshot; identical bytes re-upload nothing.
void regression_stamp_hashes_identical_bytes() {
    std::shared_ptr<StampRegistry> tracked;
    Fixture f(0x40000, [&](auto inner) {
        tracked = std::make_shared<StampRegistry>(std::move(inner));
        return tracked;
    });
    auto first = acquire(f, 0x1000, 0x40);
    tracked->touch();
    CHECK(succeeded(f.memory->submission_boundary()));
    const auto reads = f.backend->reads;
    auto second = acquire(f, 0x1000, 0x40);
    CHECK(f.backend->reads == reads + 1);
    CHECK(second.storage == first.storage);
    const auto stats = buffer_cache_statistics(*f.cache);
    CHECK(stats.revalidations_hashed == 1);
    CHECK(stats.reuploads == 0);
    CHECK(stats.revalidations_skipped == 0);
    // The refreshed certificate answers clean across the next boundary.
    CHECK(succeeded(f.memory->submission_boundary()));
    acquire(f, 0x1000, 0x40);
    CHECK(f.backend->reads == reads + 1);
    CHECK(buffer_cache_statistics(*f.cache).revalidations_skipped == 1);
}

// Only the page range whose hash differs is re-copied, in place, while no
// consumer pins the image; pages the tracker proves clean are not hashed.
void regression_stamp_uploads_only_changed_pages() {
    std::shared_ptr<StampRegistry> tracked;
    Fixture f(0x40000, [&](auto inner) {
        tracked = std::make_shared<StampRegistry>(std::move(inner));
        return tracked;
    });
    // An image spanning two revalidation pages needs a larger backing than
    // the fixture's 0x100 registrations provide.
    const BackingId big{1, 3, 1};
    const MappingId big_map{7, 3, 1};
    f.backend->add(big, 0x4000, std::byte{0x11});
    CHECK(succeeded(
        f.memory->register_backing({big, 0x4000, std::make_shared<ResourceLease>()})));
    CHECK(succeeded(f.memory->register_mapping(
        {big_map, big, {0x4000, 0x4000, 0, 0x10000, 0x4000, 0x10}})));
    {
        auto first = acquire(f, 0x4000, 0x2000);
        CHECK(filled(read_image(first, {big, 0, 0x2000}), std::byte{0x11}));
    }
    auto* bytes = f.backend->find(big);
    CHECK(bytes != nullptr);
    // Suspect both pages, but change bytes only inside the second page.
    std::fill(bytes->bytes.begin() + 0x1000, bytes->bytes.begin() + 0x1010,
              std::byte{0x77});
    tracked->touch({big, 0, 0x2000});
    CHECK(succeeded(f.memory->submission_boundary()));
    const auto reads = f.backend->reads;
    auto patched = acquire(f, 0x4000, 0x2000);
    CHECK(f.backend->reads == reads + 1);
    const auto stats = buffer_cache_statistics(*f.cache);
    CHECK(stats.reuploads == 1);
    // Only the changed page is re-copied; the clean first page keeps its
    // certificate without a byte access.
    CHECK(stats.reupload_bytes == 0x1000);
    CHECK(stats.revalidations_hashed == 0);
    CHECK(filled(read_image(patched, {big, 0, 0x1000}), std::byte{0x11}));
    CHECK(filled(read_image(patched, {big, 0x1000, 0x10}), std::byte{0x77}));
    CHECK(filled(read_image(patched, {big, 0x1010, 0xff0}), std::byte{0x11}));
}

// Without a tracker the registry answers written-since for every range:
// epoch revalidation hashes and never reports a stamp skip.
void regression_default_tracking_hashes_every_epoch() {
    Fixture f(0x40000);
    auto first = acquire(f, 0x1000, 0x40);
    CHECK(succeeded(f.memory->submission_boundary()));
    const auto reads = f.backend->reads;
    auto second = acquire(f, 0x1000, 0x40);
    CHECK(f.backend->reads == reads + 1);
    CHECK(second.storage == first.storage);
    const auto stats = buffer_cache_statistics(*f.cache);
    CHECK(stats.revalidations_hashed == 1);
    CHECK(stats.revalidations_skipped == 0);
    CHECK(stats.reuploads == 0);
}

bool run_regression(std::string_view name) {
    const std::pair<std::string_view, void(*)()> cases[] = {
        {"1", regression_1_realignment}, {"2", regression_2_ring_alignment},
        {"3", regression_3_trim_overlap}, {"4", regression_4_backing_identity},
        {"5", regression_5_interval_versions}, {"6", regression_6_pending_copy},
        {"7", regression_7_canceled_authority}, {"8", regression_8_resolve_latest},
        {"10", regression_10_registry_budget}, {"11", regression_11_reclaim_admission},
        {"12", regression_12_partial_mapping}, {"13", regression_13_registry_only_ring},
        {"ring-capacity", regression_ring_capacity_threshold},
        {"retired-budget", regression_replaced_allocation_accounting},
        {"superseded-authority", regression_superseded_authority},
        {"prepared-overlap", regression_prepared_overlap},
        {"cache-domain", regression_cache_domain_overlap},
        {"partial-remap", regression_refresh_after_partial_remap},
        {"scoped-refresh", regression_scoped_refresh_versions},
        {"buffer-scale", regression_retained_read_scales_with_alias_count},
        {"epoch-revalidation", regression_epoch_revalidation},
        {"partial-epoch-revalidation", regression_partial_epoch_revalidation},
        {"mixed-gpu-epoch-revalidation", regression_mixed_gpu_epoch_revalidation},
        {"stamp-skip", regression_stamp_skips_revalidation},
        {"stamp-hash", regression_stamp_hashes_identical_bytes},
        {"stamp-page", regression_stamp_uploads_only_changed_pages},
        {"default-tracking", regression_default_tracking_hashes_every_epoch},
        {"snapshot-order", test_snapshot_preserves_mapping_order_and_offsets},
        {"snapshot-bounds", test_snapshot_preflights_holes_padding_and_bounds},
        {"snapshot-read-failure", test_snapshot_propagates_valid_fragment_read_failure},
        {"snapshot-gpu", test_snapshot_resolves_gpu_fragment_before_any_copy},
        {"snapshot-pins", test_snapshot_honors_reservations_and_mapping_generations},
        {"snapshot-zero-padding", test_constant_snapshot_zeroes_padding_without_shifting_later_bytes},
        {"snapshot-zero-errors", test_constant_snapshot_does_not_hide_read_or_gpu_errors}};
    for (const auto& [id, test] : cases) {
        if (name == "all" || name == id) {
            std::fprintf(stderr, "Review regression %.*s\n", static_cast<int>(id.size()), id.data());
            test();
            if (name != "all") return true;
        }
    }
    return name == "all";
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) return run_regression(argv[1]) && !failures ? 0 : 1;
    run_regression("all");
    test_uses_share_one_coherent_view();
    test_native_extent_pads_the_image();
    test_uniform_window_snapshot_zero_tail();
    test_partial_cpu_write_updates_only_changed_interval();
    test_gpu_produced_bytes_are_consumed_without_cpu_round_trip();
    test_growth_writes_back_finished_authority();
    test_foreign_gpu_authority_is_resolved_once();
    test_uniform_version_a_is_retained_for_draw_a();
    test_upload_ring_waits_for_delayed_completion();
    test_invalidate_resolve_trim_and_budget();
    test_budget_is_enforced_and_domain_matches();
    test_factory_rejects_invalid_construction();
    test_mapping_invalidation_refreshes_residency_across_generations();
    if (failures) {
        std::fprintf(stderr, "%u check(s) failed\n", failures);
        return 1;
    }
    std::printf("graphics.buffer_cache: all checks passed\n");
    return 0;
}
