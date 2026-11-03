// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/resource_memory_registry.h>
#include <armigo/memory/write_tracker.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <variant>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;

#define CHECK(...)                                                             \
    do {                                                                       \
        if (!(__VA_ARGS__)) {                                                  \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__);   \
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

struct ProbeLease final : ResourceLease {
    explicit ProbeLease(std::shared_ptr<std::atomic<unsigned>> destructions = {})
        : destructions(std::move(destructions)) {}
    ~ProbeLease() override {
        if (destructions) {
            ++*destructions;
        }
    }
    std::shared_ptr<std::atomic<unsigned>> destructions;
};

RendererError synthetic_error(const char* detail) {
    return RendererError({CommandStatus::MemoryFault, detail});
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
        std::lock_guard lock(data_mutex);
        ++reads;
        if (fail_reads) {
            return synthetic_error("synthetic CPU read failure");
        }
        Allocation* allocation = find(range.backing);
        if (!allocation || destination.size() != range.size ||
            range.offset > allocation->bytes.size() ||
            range.size > allocation->bytes.size() - range.offset) {
            return synthetic_error("synthetic CPU read bounds failure");
        }
        std::copy_n(allocation->bytes.begin() + range.offset, destination.size(),
                    destination.begin());
        return renderer_success();
    }

    RendererStatus write_cpu(BackingRange range,
                             std::span<const std::byte> source) override {
        std::lock_guard lock(data_mutex);
        ++writes;
        Allocation* allocation = find(range.backing);
        if (!allocation || source.size() != range.size ||
            range.offset > allocation->bytes.size() ||
            range.size > allocation->bytes.size() - range.offset) {
            return synthetic_error("synthetic CPU write bounds failure");
        }
        const bool rejected = fail_writes || range.backing == fail_write_backing;
        const std::size_t copied = rejected
                                       ? std::min(partial_write_bytes, source.size())
                                       : source.size();
        std::copy_n(source.begin(), copied, allocation->bytes.begin() + range.offset);
        if (rejected) {
            return synthetic_error("synthetic CPU write failure");
        }
        return renderer_success();
    }

    RendererStatus wait_gpu(CompletionToken token) override {
        ++waits;
        std::unique_lock lock(wait_mutex);
        wait_entered = true;
        wait_changed.notify_all();
        wait_changed.wait(lock, [&] { return !block_wait; });
        if (fail_waits) {
            return synthetic_error("synthetic GPU wait failure");
        }
        if (!token.valid()) {
            return synthetic_error("synthetic invalid GPU token");
        }
        return renderer_success();
    }

    void await_wait_entry() {
        std::unique_lock lock(wait_mutex);
        wait_changed.wait(lock, [&] { return wait_entered; });
    }

    void release_wait() {
        std::lock_guard lock(wait_mutex);
        block_wait = false;
        wait_changed.notify_all();
    }

    std::vector<Allocation> allocations;
    std::mutex data_mutex;
    std::atomic<unsigned> reads = 0;
    std::atomic<unsigned> writes = 0;
    std::atomic<unsigned> waits = 0;
    bool fail_reads = false;
    bool fail_writes = false;
    BackingId fail_write_backing{};
    bool fail_waits = false;
    std::size_t partial_write_bytes = 0;
    std::mutex wait_mutex;
    std::condition_variable wait_changed;
    bool block_wait = false;
    bool wait_entered = false;
};

struct Storage final : ResourceMemoryStorage {
    Storage(BackingRange bytes, std::vector<std::byte> contents)
        : bytes(bytes), contents(std::move(contents)) {}

    BackingRange coverage() const noexcept override { return bytes; }

    RendererStatus read_gpu(BackingRange range,
                            std::span<std::byte> destination) const override {
        ++reads;
        if (fail_reads || range.backing != bytes.backing ||
            range.offset < bytes.offset || destination.size() != range.size ||
            range.offset - bytes.offset > contents.size() ||
            range.size > contents.size() - (range.offset - bytes.offset)) {
            return synthetic_error("synthetic GPU storage read failure");
        }
        const std::size_t offset = range.offset - bytes.offset;
        std::copy_n(contents.begin() + offset, destination.size(), destination.begin());
        return renderer_success();
    }

    bool guest_boundary_writeback() const noexcept override { return !resident; }
    bool in_place() const noexcept override { return zero_copy; }

    BackingRange bytes;
    std::vector<std::byte> contents;
    mutable std::atomic<unsigned> reads = 0;
    bool fail_reads = false;
    bool resident = false;
    bool zero_copy = false;
};

struct NoopActions final : CompletionActions {
    RendererStatus make_coherent() override { return renderer_success(); }
    RendererStatus prepare_publication() override { return renderer_success(); }
    void publish() noexcept override {}
    RendererStatus preserve_failed_authority() override { return renderer_success(); }
};

struct BoundaryActions final : CompletionActions {
    explicit BoundaryActions(std::shared_ptr<ResourceMemoryRegistry> memory)
        : memory(std::move(memory)) {}

    RendererStatus make_coherent() override {
        ++coherence_calls;
        const auto snapshot = observed.snapshot();
        CHECK(snapshot.phase == SubmissionPhase::Publishing);
        CHECK(!snapshot.guest_visible_complete());
        return memory->submission_boundary();
    }
    RendererStatus prepare_publication() override {
        ++prepare_calls;
        return renderer_success();
    }
    void publish() noexcept override { ++publish_calls; }
    RendererStatus preserve_failed_authority() override { return renderer_success(); }

    std::shared_ptr<ResourceMemoryRegistry> memory;
    CompletionToken observed;
    unsigned coherence_calls = 0;
    unsigned prepare_calls = 0;
    unsigned publish_calls = 0;
};

struct TokenSource {
    explicit TokenSource(std::uint64_t sequence)
        : completion({1, sequence}, {}, std::make_shared<NoopActions>()),
          token(completion.token()) {}

    void submit() { CHECK(succeeded(completion.submitted())); }
    void succeed() { CHECK(succeeded(completion.feedback({}))); }
    void fail() {
        CHECK(succeeded(completion.feedback(
            {CommandStatus::MemoryFault, "synthetic GPU execution failure"})));
    }

    SubmissionCompletion completion;
    CompletionToken token;
};

constexpr BackingId backing_a{1, 1, 1};
constexpr BackingId backing_b{1, 2, 1};
constexpr MappingId mapping_a{7, 1, 1};
constexpr MappingId mapping_alias{7, 2, 1};
constexpr MappingId mapping_b{7, 3, 1};

Mapping mapped(std::uint64_t gpu_base, std::uint64_t offset = 0,
               std::uint64_t size = 0x40,
               std::uint64_t guest_address = 0x8000) {
    return {gpu_base, size, offset, guest_address, 0x40, 0x10};
}

struct Fixture {
    explicit Fixture(std::shared_ptr<ResourceMemoryWorkCounters> counters = {})
        : backend(std::make_shared<Backend>()) {
        backend->add(backing_a, 0x40);
        backend->add(backing_b, 0x40);
        memory = require_value(make_resource_memory(
            {7, backend, CoherenceMode::SubmissionBoundary, 0x10, std::move(counters)}));
    }

    void register_backing(BackingId id, std::uint64_t size = 0x40,
                          RetainedResource owner = std::make_shared<ProbeLease>()) {
        CHECK(succeeded(memory->register_backing({id, size, std::move(owner)})));
    }

    void register_mapping(MappingId id, BackingId backing, Mapping value) {
        CHECK(succeeded(memory->register_mapping({id, backing, value})));
    }

    void standard_layout() {
        register_backing(backing_a);
        register_backing(backing_b);
        register_mapping(mapping_a, backing_a, mapped(0x1000));
        register_mapping(mapping_alias, backing_a, mapped(0x2000));
        register_mapping(mapping_b, backing_b, mapped(0x3000, 0, 0x40, 0x9000));
    }

    std::shared_ptr<Backend> backend;
    std::shared_ptr<ResourceMemoryRegistry> memory;
};

RetainedRange retain(ResourceMemory& memory, std::uint64_t address,
                     std::uint64_t size, ResourceAccess access) {
    return require_value(memory.retain(address, size, access));
}

std::vector<DirtyRange> inspect(ResourceMemory& memory, BackingRange range) {
    return require_value(memory.inspect(range));
}

const DirtyRange* covering(const std::vector<DirtyRange>& ranges,
                           BackingId backing, std::uint64_t offset) {
    const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const auto& range) {
        return range.bytes.backing == backing && range.bytes.offset <= offset &&
               offset - range.bytes.offset < range.bytes.size;
    });
    return found == ranges.end() ? nullptr : &*found;
}

std::shared_ptr<GpuAccessReservation> reserve(
    ResourceMemory& memory, std::span<const GpuAccessRequest> requests,
    CompletionToken token) {
    return require_value(memory.reserve_accesses(requests, token));
}

void commit_write(ResourceMemory& memory, RetainedRange range,
                  const std::shared_ptr<Storage>& storage, TokenSource& token) {
    const std::array requests{GpuAccessRequest{range, ResourceAccess::Write}};
    auto reservation = reserve(memory, requests, token.token);
    CHECK(succeeded(reservation->commit({storage})));
    token.submit();
}

void test_factory_registration_aliases_and_versions() {
    auto backend = std::make_shared<Backend>();
    CHECK(std::holds_alternative<RendererError>(make_resource_memory({0, backend})));
    CHECK(std::holds_alternative<RendererError>(make_resource_memory({7, {}})));
    CHECK(std::holds_alternative<RendererError>(
        make_resource_memory({7, backend, CoherenceMode::TrackedNativeAccess})));
    CHECK(std::holds_alternative<RendererError>(
        make_resource_memory({7, backend, CoherenceMode::SubmissionBoundary, 0})));

    Fixture fixture;
    CHECK(fixture.memory->coherence_mode() == CoherenceMode::SubmissionBoundary);
    CHECK(!succeeded(fixture.memory->register_backing({{}, 0x40,
                                                       std::make_shared<ProbeLease>()})));
    CHECK(!succeeded(fixture.memory->register_backing({backing_a, 0,
                                                       std::make_shared<ProbeLease>()})));
    CHECK(!succeeded(fixture.memory->register_backing({backing_a, 0x40, {}})));
    fixture.standard_layout();
    CHECK(!succeeded(fixture.memory->register_backing(
        {backing_a, 0x40, std::make_shared<ProbeLease>()})));

    const auto first = retain(*fixture.memory, 0x1008, 8, ResourceAccess::ReadWrite);
    const auto alias = retain(*fixture.memory, 0x2008, 8, ResourceAccess::ReadWrite);
    CHECK(first.mapping == mapping_a);
    CHECK(alias.mapping == mapping_alias);
    CHECK(first.bytes == BackingRange{backing_a, 8, 8});
    CHECK(alias.bytes == first.bytes);
    CHECK(first.lease && alias.lease);

    auto transaction = require_value(
        fixture.memory->prepare_cpu_write({backing_a, 9, 3}));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        auto* allocation = fixture.backend->find(backing_a);
        allocation->bytes[9] = std::byte{0x91};
        allocation->bytes[10] = std::byte{0x92};
        allocation->bytes[11] = std::byte{0x93};
    }
    transaction->commit();
    transaction.reset();

    const auto ranges = inspect(*fixture.memory, {backing_a, 8, 8});
    for (std::uint64_t offset = 9; offset != 12; ++offset) {
        const auto* range = covering(ranges, backing_a, offset);
        CHECK(range != nullptr);
        if (range) {
            CHECK(range->version.cpu_write == 1);
            CHECK(range->version.gpu_write == 0);
            CHECK(range->cpu_dirty);
            CHECK(!range->gpu_dirty);
        }
    }
    const auto before_read = inspect(*fixture.memory, {backing_a, 9, 3});
    auto read = require_value(fixture.memory->prepare_cpu_read(
        {backing_a, 9, 3}, ResolveReason::CpuRead));
    read.reset();
    const auto after_read = inspect(*fixture.memory, {backing_a, 9, 3});
    CHECK(after_read.size() == before_read.size());
    if (after_read.size() == before_read.size()) {
        for (std::size_t index = 0; index != before_read.size(); ++index) {
            CHECK(after_read[index].bytes == before_read[index].bytes);
            CHECK(after_read[index].version == before_read[index].version);
            CHECK(after_read[index].cpu_dirty == before_read[index].cpu_dirty);
            CHECK(after_read[index].gpu_dirty == before_read[index].gpu_dirty);
        }
    }
}

void test_backing_identity_query_tracks_lifecycle() {
    Fixture fixture;
    const BackingId replacement{1, 1, 2};

    CHECK(!fixture.memory->has_backing(backing_a));
    CHECK(!fixture.memory->has_backing(backing_b));
    CHECK(!fixture.memory->has_backing(replacement));

    fixture.register_backing(backing_a);
    CHECK(fixture.memory->has_backing(backing_a));
    CHECK(!fixture.memory->has_backing(backing_b));
    CHECK(!fixture.memory->has_backing(replacement));

    CHECK(succeeded(fixture.memory->retire_backing(backing_a)));
    CHECK(!fixture.memory->has_backing(backing_a));
    CHECK(!fixture.memory->has_backing(replacement));

    fixture.backend->add(replacement, 0x40);
    fixture.register_backing(replacement);
    CHECK(!fixture.memory->has_backing(backing_a));
    CHECK(fixture.memory->has_backing(replacement));
}

void test_cpu_transactions_exclude_aliases_and_cancel_cleanly() {
    Fixture fixture;
    fixture.standard_layout();
    auto held = require_value(fixture.memory->prepare_cpu_write({backing_a, 4, 8}));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->prepare_cpu_write({backing_a, 10, 2})));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->prepare_cpu_read(
        {backing_a, 6, 1}, ResolveReason::CpuRead)));
    TokenSource reader(1);
    const auto alias = retain(*fixture.memory, 0x2006, 2, ResourceAccess::Read);
    const std::array blocked{GpuAccessRequest{alias, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(blocked, reader.token)));
    held.reset();
    const auto cancelled = inspect(*fixture.memory, {backing_a, 4, 8});
    const auto* unchanged = covering(cancelled, backing_a, 4);
    CHECK(unchanged != nullptr);
    if (unchanged) {
        CHECK(unchanged->version == ResourceVersion{});
        CHECK(!unchanged->gpu_dirty);
    }

    auto admitted = require_value(
        fixture.memory->prepare_cpu_write({backing_a, 4, 8}));
    admitted->commit();
    admitted.reset();
    const auto dirty = inspect(*fixture.memory, {backing_a, 4, 8});
    CHECK(covering(dirty, backing_a, 4)->version.cpu_write > 0);
}

void test_overlapping_cpu_reads_and_pending_gpu_reader_coexist() {
    Fixture fixture;
    fixture.standard_layout();
    auto first = require_value(fixture.memory->prepare_cpu_read(
        {backing_a, 4, 8}, ResolveReason::CpuRead));
    auto overlapping = require_value(fixture.memory->prepare_cpu_read(
        {backing_a, 6, 2}, ResolveReason::CpuEngineRead));

    TokenSource gpu_reader(2);
    const auto alias = retain(*fixture.memory, 0x2005, 4, ResourceAccess::Read);
    const std::array requests{GpuAccessRequest{alias, ResourceAccess::Read}};
    auto reservation = reserve(*fixture.memory, requests, gpu_reader.token);
    auto alongside = require_value(fixture.memory->prepare_cpu_read(
        {backing_a, 6, 1}, ResolveReason::Capture));
    alongside.reset();
    reservation.reset();
    overlapping.reset();
    first.reset();
}

void test_remap_reallocation_generations_and_pins() {
    Fixture fixture;
    fixture.register_backing(backing_a);
    fixture.register_mapping(mapping_a, backing_a, mapped(0x1000));
    auto old = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Read);
    CHECK(!succeeded(fixture.memory->retire_backing(backing_a)));
    CHECK(!succeeded(fixture.memory->unmap(mapping_a)));
    old.lease.reset();
    CHECK(succeeded(fixture.memory->unmap(mapping_a)));
    CHECK(!succeeded(fixture.memory->register_mapping(
        {mapping_a, backing_a, mapped(0x1000)})));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x1000, 8, ResourceAccess::Read)));

    const BackingId replacement{1, 1, 2};
    const MappingId remapped{7, 1, 2};
    fixture.backend->add(replacement, 0x40, std::byte{0x66});
    CHECK(succeeded(fixture.memory->retire_backing(backing_a)));
    fixture.register_backing(replacement);
    fixture.register_mapping(remapped, replacement, mapped(0x1000));
    const auto current = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Read);
    CHECK(current.mapping == remapped);
    CHECK(current.bytes.backing == replacement);
    CHECK(!succeeded(fixture.memory->unmap(mapping_a)));

    TokenSource token(2);
    const std::array stale_request{GpuAccessRequest{old, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(stale_request, token.token)));
}

void test_partial_gpu_cpu_overlaps_preserve_newest_bytes() {
    Fixture fixture;
    fixture.standard_layout();
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        auto* bytes = &fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0; index != bytes->size(); ++index) {
            (*bytes)[index] = std::byte(index);
        }
    }

    TokenSource gpu(3);
    const auto target = retain(*fixture.memory, 0x1008, 8, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(
        target.bytes, std::vector<std::byte>{std::byte{0xa0}, std::byte{0xa1},
            std::byte{0xa2}, std::byte{0xa3}, std::byte{0xa4}, std::byte{0xa5},
            std::byte{0xa6}, std::byte{0xa7}});
    commit_write(*fixture.memory, target, storage, gpu);
    gpu.succeed();

    auto cpu = require_value(fixture.memory->prepare_cpu_write({backing_a, 12, 4}));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(bytes[8] == std::byte{0xa0});
        CHECK(bytes[11] == std::byte{0xa3});
        bytes[12] = std::byte{0xc4};
        bytes[13] = std::byte{0xc5};
        bytes[14] = std::byte{0xc6};
        bytes[15] = std::byte{0xc7};
    }
    cpu->commit();
    cpu.reset();
    CHECK(succeeded(fixture.memory->resolve(
        {backing_a, 8, 8}, ResolveReason::CpuRead)));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(bytes[8] == std::byte{0xa0});
        CHECK(bytes[11] == std::byte{0xa3});
        CHECK(bytes[12] == std::byte{0xc4});
        CHECK(bytes[15] == std::byte{0xc7});
        CHECK(bytes[7] == std::byte{0x07});
        CHECK(bytes[16] == std::byte{0x10});
    }

    TokenSource partial(4);
    const auto middle = retain(*fixture.memory, 0x100a, 3, ResourceAccess::Write);
    auto middle_storage = std::make_shared<Storage>(
        BackingRange{backing_a, 8, 8},
        std::vector<std::byte>{std::byte{0xee}, std::byte{0xee}, std::byte{0xd2},
            std::byte{0xd3}, std::byte{0xd4}, std::byte{0xee}, std::byte{0xee},
            std::byte{0xee}});
    commit_write(*fixture.memory, middle, middle_storage, partial);
    partial.succeed();
    CHECK(succeeded(fixture.memory->resolve(
        {backing_a, 8, 8}, ResolveReason::CpuRead)));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(bytes[8] == std::byte{0xa0});
        CHECK(bytes[9] == std::byte{0xa1});
        CHECK(bytes[10] == std::byte{0xd2});
        CHECK(bytes[12] == std::byte{0xd4});
        CHECK(bytes[13] == std::byte{0xc5});
    }
}

void test_overlapping_gpu_writers_keep_newest_interval_authority() {
    Fixture fixture;
    fixture.standard_layout();
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0; index != bytes.size(); ++index) {
            bytes[index] = std::byte(0x40 + index);
        }
    }

    TokenSource first_writer(30);
    const auto whole = retain(*fixture.memory, 0x1000, 16, ResourceAccess::Write);
    std::vector<std::byte> first_bytes(16);
    for (std::size_t index = 0; index != first_bytes.size(); ++index) {
        first_bytes[index] = std::byte(0x80 + index);
    }
    auto first_storage = std::make_shared<Storage>(whole.bytes, first_bytes);
    commit_write(*fixture.memory, whole, first_storage, first_writer);
    first_writer.succeed();

    TokenSource second_writer(31);
    const auto middle = retain(*fixture.memory, 0x2004, 4, ResourceAccess::Write);
    const std::vector<std::byte> middle_bytes{
        std::byte{0xd4}, std::byte{0xd5}, std::byte{0xd6}, std::byte{0xd7}};
    auto middle_storage = std::make_shared<Storage>(middle.bytes, middle_bytes);
    commit_write(*fixture.memory, middle, middle_storage, second_writer);
    second_writer.succeed();

    CHECK(succeeded(fixture.memory->resolve(
        {backing_a, 2, 6}, ResolveReason::Capture)));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(bytes[0] == std::byte{0x40});
        CHECK(bytes[1] == std::byte{0x41});
        CHECK(bytes[2] == std::byte{0x82});
        CHECK(bytes[3] == std::byte{0x83});
        CHECK(bytes[4] == std::byte{0xd4});
        CHECK(bytes[7] == std::byte{0xd7});
        CHECK(bytes[8] == std::byte{0x48});
        CHECK(bytes[15] == std::byte{0x4f});
        CHECK(bytes[16] == std::byte{0x50});
    }
    const auto partially_resolved = inspect(*fixture.memory, whole.bytes);
    CHECK(covering(partially_resolved, backing_a, 0)->gpu_dirty);
    CHECK(!covering(partially_resolved, backing_a, 2)->gpu_dirty);
    CHECK(!covering(partially_resolved, backing_a, 4)->gpu_dirty);
    CHECK(covering(partially_resolved, backing_a, 8)->gpu_dirty);

    CHECK(succeeded(fixture.memory->resolve(
        whole.bytes, ResolveReason::SubmissionBoundary)));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0; index != 4; ++index) {
            CHECK(bytes[index] == first_bytes[index]);
        }
        for (std::size_t index = 4; index != 8; ++index) {
            CHECK(bytes[index] == middle_bytes[index - 4]);
        }
        for (std::size_t index = 8; index != 16; ++index) {
            CHECK(bytes[index] == first_bytes[index]);
        }
        CHECK(bytes[16] == std::byte{0x50});
    }
    const auto resolved = inspect(*fixture.memory, whole.bytes);
    CHECK(std::none_of(resolved.begin(), resolved.end(),
                       [](const auto& range) { return range.gpu_dirty; }));
}

void test_storage_inspection_tracks_clipped_current_authority() {
    Fixture fixture;
    fixture.standard_layout();
    const BackingRange queried{backing_a, 2, 16};
    const auto initial = require_value(fixture.memory->inspect_storage(queried));
    CHECK(initial.size() == 1);
    CHECK(initial.front().dirty.bytes == queried);
    CHECK(!initial.front().dirty.gpu_dirty);
    CHECK(!initial.front().storage);

    TokenSource first_writer(1);
    const auto whole = retain(*fixture.memory, 0x1000, 16, ResourceAccess::Write);
    auto first_storage = std::make_shared<Storage>(
        whole.bytes, std::vector<std::byte>(16, std::byte{0x31}));
    const std::array request{GpuAccessRequest{whole, ResourceAccess::Write}};
    auto reservation = reserve(*fixture.memory, request, first_writer.token);
    CHECK(succeeded(reservation->commit({first_storage})));
    reservation.reset();

    // Accepted work exposes authority without implicitly waiting or submitting.
    const auto accepted = require_value(fixture.memory->inspect_storage(whole.bytes));
    CHECK(accepted.size() == 1);
    CHECK(accepted.front().storage == first_storage);
    CHECK(accepted.front().dirty.gpu_dirty);
    CHECK(accepted.front().dirty.last_writer.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(fixture.backend->waits == 0);
    CHECK(first_storage->reads == 0);
    first_writer.submit();
    first_writer.succeed();

    TokenSource second_writer(2);
    const auto middle = retain(*fixture.memory, 0x2004, 4, ResourceAccess::Write);
    auto second_storage = std::make_shared<Storage>(
        middle.bytes, std::vector<std::byte>(4, std::byte{0x72}));
    commit_write(*fixture.memory, middle, second_storage, second_writer);
    second_writer.succeed();
    const auto split = require_value(fixture.memory->inspect_storage(queried));
    const auto dirty = inspect(*fixture.memory, queried);
    CHECK(split.size() == 4);
    CHECK(split.size() == dirty.size());
    if (split.size() != 4 || split.size() != dirty.size()) return;
    constexpr std::array offsets{2U, 4U, 8U, 16U};
    constexpr std::array sizes{2U, 4U, 8U, 2U};
    const std::array<std::shared_ptr<const ResourceMemoryStorage>, 4> authorities{
        first_storage, second_storage, first_storage, nullptr};
    for (std::size_t index = 0; index != split.size(); ++index) {
        CHECK(split[index].dirty.bytes == BackingRange{backing_a, offsets[index], sizes[index]});
        CHECK(split[index].storage == authorities[index]);
        CHECK(split[index].dirty.bytes == dirty[index].bytes);
        CHECK(split[index].dirty.version == dirty[index].version);
        CHECK(split[index].dirty.cpu_dirty == dirty[index].cpu_dirty);
        CHECK(split[index].dirty.gpu_dirty == dirty[index].gpu_dirty);
        CHECK(split[index].dirty.last_writer.snapshot().id == dirty[index].last_writer.snapshot().id);
    }
    CHECK(fixture.backend->reads == 0);
    CHECK(fixture.backend->writes == 0);
    CHECK(fixture.backend->waits == 0);
    CHECK(first_storage->reads == 0);
    CHECK(second_storage->reads == 0);

    CHECK(succeeded(fixture.memory->resolve({backing_a, 3, 6}, ResolveReason::Capture)));
    const auto resolved = require_value(fixture.memory->inspect_storage(queried));
    for (const auto& entry : resolved) {
        const auto offset = entry.dirty.bytes.offset;
        if (offset >= 3 && offset < 9) {
            CHECK(!entry.dirty.gpu_dirty);
            CHECK(!entry.storage);
        } else if (offset < 16) {
            CHECK(entry.dirty.gpu_dirty);
            CHECK(entry.storage == first_storage);
        } else {
            CHECK(!entry.storage);
        }
    }
    // Previously returned snapshots retain descriptors but do not remain current.
    CHECK(split[1].storage == second_storage);
    CHECK(split[1].dirty.gpu_dirty);
    CHECK(succeeded(fixture.memory->resolve(whole.bytes, ResolveReason::Capture)));
    const auto clean = require_value(fixture.memory->inspect_storage(queried));
    CHECK(std::all_of(clean.begin(), clean.end(), [](const auto& entry) {
        return !entry.dirty.gpu_dirty && !entry.storage;
    }));
}

void test_publication_boundary_does_not_wait_for_complete() {
    Fixture fixture;
    fixture.standard_layout();
    auto actions = std::make_shared<BoundaryActions>(fixture.memory);
    SubmissionCompletion completion({1, 32}, {}, actions);
    const auto token = completion.token();
    actions->observed = token;

    const auto target = retain(*fixture.memory, 0x1020, 4, ResourceAccess::Write);
    const std::array request{GpuAccessRequest{target, ResourceAccess::Write}};
    auto reservation = reserve(*fixture.memory, request, token);
    auto storage = std::make_shared<Storage>(
        target.bytes, std::vector<std::byte>(4, std::byte{0xec}));
    CHECK(succeeded(reservation->commit({storage})));
    CHECK(succeeded(completion.submitted()));
    CHECK(succeeded(completion.feedback({})));
    CHECK(token.snapshot().phase == SubmissionPhase::GpuComplete);
    CHECK(fixture.backend->waits == 0);
    CHECK(succeeded(completion.finish()));
    CHECK(token.snapshot().phase == SubmissionPhase::Complete);
    CHECK(fixture.backend->waits == 0);
    CHECK(actions->coherence_calls == 1);
    CHECK(actions->prepare_calls == 1);
    CHECK(actions->publish_calls == 1);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(std::all_of(bytes.begin() + 0x20, bytes.begin() + 0x24,
                          [](std::byte byte) { return byte == std::byte{0xec}; }));
    }
}

void test_submission_owner_retains_registry_and_reader_storage() {
    Fixture fixture;
    fixture.standard_layout();
    auto actions = std::make_shared<BoundaryActions>(fixture.memory);
    SubmissionCompletion completion({1, 33}, {}, actions);
    actions->observed = completion.token();
    std::weak_ptr<ResourceMemoryRegistry> registry = fixture.memory;
    auto storage = std::make_shared<ProbeLease>();
    std::weak_ptr<const ResourceLease> retained_storage = storage;
    {
        auto range = retain(*fixture.memory, 0x1000, 4, ResourceAccess::Read);
        const std::array requests{GpuAccessRequest{range, ResourceAccess::Read}};
        auto reservation = reserve(*fixture.memory, requests, completion.token());
        CHECK(succeeded(reservation->commit({storage})));
    }
    CHECK(succeeded(completion.submitted()));
    storage.reset();
    fixture.memory.reset();
    actions.reset();
    CHECK(!registry.expired());
    CHECK(!retained_storage.expired());
    CHECK(succeeded(completion.feedback({})));
    CHECK(!registry.expired());
    CHECK(!retained_storage.expired());
    CHECK(succeeded(completion.finish()));
    CHECK(registry.expired());
    CHECK(retained_storage.expired());
}

void test_deterministic_sequential_byte_oracle() {
    Fixture fixture;
    fixture.standard_layout();
    std::array<std::byte, 0x40> newest{};
    std::uint32_t random = 0x6d2b79f5u;
    std::uint64_t sequence = 100;
    auto next = [&] {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };

    for (unsigned operation = 0; operation != 192; ++operation) {
        const std::size_t size = 1 + next() % 12;
        const std::size_t offset = next() % (newest.size() - size + 1);
        const unsigned kind = next() % 3;
        if (kind == 0) {
            auto transaction = require_value(fixture.memory->prepare_cpu_write(
                {backing_a, offset, size}));
            std::lock_guard lock(fixture.backend->data_mutex);
            auto& bytes = fixture.backend->find(backing_a)->bytes;
            for (std::size_t index = 0; index != size; ++index) {
                const auto value = std::byte((next() >> 8) & 0xff);
                bytes[offset + index] = value;
                newest[offset + index] = value;
            }
            transaction->commit();
        } else if (kind == 1) {
            const std::uint64_t alias_base = (next() & 1) ? 0x1000 : 0x2000;
            const auto range = retain(*fixture.memory, alias_base + offset, size,
                                      ResourceAccess::Write);
            std::vector<std::byte> contents(size);
            for (std::size_t index = 0; index != size; ++index) {
                contents[index] = std::byte((next() >> 16) & 0xff);
                newest[offset + index] = contents[index];
            }
            TokenSource writer(sequence++);
            auto storage = std::make_shared<Storage>(range.bytes, std::move(contents));
            commit_write(*fixture.memory, range, storage, writer);
            writer.succeed();
        } else {
            CHECK(succeeded(fixture.memory->resolve(
                {backing_a, offset, size}, ResolveReason::Capture)));
            std::lock_guard lock(fixture.backend->data_mutex);
            const auto& bytes = fixture.backend->find(backing_a)->bytes;
            CHECK(std::equal(newest.begin() + offset, newest.begin() + offset + size,
                             bytes.begin() + offset));
        }
    }

    CHECK(succeeded(fixture.memory->submission_boundary()));
    std::lock_guard lock(fixture.backend->data_mutex);
    const auto& bytes = fixture.backend->find(backing_a)->bytes;
    CHECK(std::equal(newest.begin(), newest.end(), bytes.begin()));
}

void test_randomized_eager_byte_model_equivalence(std::uint32_t seed, bool fused) {
    struct ByteState {
        ResourceVersion version;
        bool cpu_dirty = true;
        bool gpu_dirty = false;
        SubmissionId writer;
        std::shared_ptr<const ResourceMemoryStorage> storage;
        std::byte newest{};
        std::byte cpu_visible{};
    };

    Fixture fixture;
    fixture.standard_layout();
    constexpr std::size_t byte_count = 0x40;
    constexpr std::size_t granule = 0x10;
    std::array<ByteState, byte_count> expected{};
    std::uint32_t random = seed;
    std::uint64_t cpu_version = 0;
    std::uint64_t gpu_version = 0;
    std::uint64_t sequence = 1000;
    MappingId primary_mapping = mapping_a;
    const auto stale_lease = retain(
        *fixture.memory, 0x1008, 8, ResourceAccess::ReadWrite);

    auto next = [&] {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };
    auto report = [&](bool condition, unsigned operation, std::size_t offset,
                      const char* property) {
        if (condition) return;
        std::fprintf(stderr,
                     "FAIL randomized resource model seed=0x%08x fused=%u operation=%u "
                     "offset=%zu property=%s\n",
                     seed, unsigned(fused), operation, offset, property);
        ++failures;
    };
    auto resolve_expected = [&](std::size_t offset, std::size_t size) {
        for (std::size_t index = offset; index != offset + size; ++index) {
            auto& byte = expected[index];
            if (!byte.gpu_dirty) continue;
            byte.cpu_visible = byte.newest;
            byte.gpu_dirty = false;
            byte.storage.reset();
        }
    };
    auto preserved_range = [=](std::size_t offset, std::size_t size) {
        const std::size_t first = offset - offset % granule;
        const std::size_t end = offset + size;
        const std::size_t last = std::min(
            byte_count, (end + granule - 1) / granule * granule);
        return std::pair{first, last - first};
    };
    auto verify = [&](unsigned operation) {
        const BackingRange whole{backing_a, 0, byte_count};
        const auto dirty = inspect(*fixture.memory, whole);
        const auto storage = require_value(fixture.memory->inspect_storage(whole));
        report(!dirty.empty(), operation, 0, "inspect returned intervals");
        report(dirty.size() == storage.size(), operation, 0,
               "inspect and inspect_storage interval counts agree");

        for (std::size_t offset = 0; offset != byte_count; ++offset) {
            const auto* actual = covering(dirty, backing_a, offset);
            const auto stored = std::find_if(
                storage.begin(), storage.end(), [&](const auto& entry) {
                    return entry.dirty.bytes.backing == backing_a &&
                           entry.dirty.bytes.offset <= offset &&
                           offset - entry.dirty.bytes.offset < entry.dirty.bytes.size;
                });
            report(actual != nullptr, operation, offset,
                   "inspect covers every byte");
            report(stored != storage.end(), operation, offset,
                   "inspect_storage covers every byte");
            if (!actual || stored == storage.end()) continue;

            const auto& reference = expected[offset];
            report(actual->version == reference.version, operation, offset,
                   "inspect version matches eager model");
            report(actual->cpu_dirty == reference.cpu_dirty, operation, offset,
                   "inspect CPU dirty state matches eager model");
            report(actual->gpu_dirty == reference.gpu_dirty, operation, offset,
                   "inspect GPU dirty state matches eager model");
            report(actual->last_writer.valid() == (reference.writer.queue != 0),
                   operation, offset, "inspect writer validity matches eager model");
            if (actual->last_writer.valid() && reference.writer.queue != 0) {
                report(actual->last_writer.snapshot().id == reference.writer,
                       operation, offset,
                       "inspect writer identity matches eager model");
            }
            report(stored->dirty.version == reference.version, operation, offset,
                   "inspect_storage version matches eager model");
            report(stored->dirty.cpu_dirty == reference.cpu_dirty, operation,
                   offset, "inspect_storage CPU dirty state matches eager model");
            report(stored->dirty.gpu_dirty == reference.gpu_dirty, operation,
                   offset, "inspect_storage GPU dirty state matches eager model");
            report(stored->dirty.last_writer.valid() ==
                       (reference.writer.queue != 0),
                   operation, offset,
                   "inspect_storage writer validity matches eager model");
            if (stored->dirty.last_writer.valid() && reference.writer.queue != 0) {
                report(stored->dirty.last_writer.snapshot().id == reference.writer,
                       operation, offset,
                       "inspect_storage writer identity matches eager model");
            }
            report(stored->storage == reference.storage, operation, offset,
                   "inspect_storage authority matches eager model");
        }

        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t offset = 0; offset != byte_count; ++offset) {
            report(bytes[offset] == expected[offset].cpu_visible, operation,
                   offset, "CPU-visible contents match eager model");
        }
    };

    verify(0);
    for (unsigned operation = 1; operation <= 320; ++operation) {
        if (operation == 137) {
            const std::array removed{primary_mapping};
            CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
            resolve_expected(0, byte_count);
            CHECK(!succeeded(fixture.memory->validate_retained(
                stale_lease, ResourceAccess::Read)));
            TokenSource stale_consumer(sequence++);
            const std::array stale_request{
                GpuAccessRequest{stale_lease, ResourceAccess::Read}};
            if (fused) {
                CHECK(std::holds_alternative<RendererError>(fixture.memory->accept_accesses(
                    stale_request, stale_consumer.token, {std::make_shared<ProbeLease>()})));
            } else {
                CHECK(std::holds_alternative<RendererError>(
                    fixture.memory->reserve_accesses(stale_request, stale_consumer.token)));
            }

            primary_mapping = {7, 1, 2};
            fixture.register_mapping(primary_mapping, backing_a, mapped(0x1000));
            const auto remapped = retain(
                *fixture.memory, 0x1008, 8, ResourceAccess::ReadWrite);
            CHECK(remapped.mapping == primary_mapping);
            CHECK(succeeded(fixture.memory->validate_retained(
                remapped, ResourceAccess::ReadWrite)));
            verify(operation);
            continue;
        }

        const std::size_t size = 1 + next() % 12;
        const std::size_t offset = next() % (byte_count - size + 1);
        const std::uint64_t gpu_base = (next() & 1) ? 0x1000 : 0x2000;
        const unsigned kind = operation <= 7 ? operation - 1 : next() % 7;
        if (kind == 0) {
            auto transaction = require_value(fixture.memory->prepare_cpu_write(
                {backing_a, offset, size}));
            const auto [preserved_offset, preserved_size] =
                preserved_range(offset, size);
            resolve_expected(preserved_offset, preserved_size);
            ++cpu_version;
            {
                std::lock_guard lock(fixture.backend->data_mutex);
                auto& bytes = fixture.backend->find(backing_a)->bytes;
                for (std::size_t index = offset; index != offset + size; ++index) {
                    const auto value = std::byte((next() >> 8) & 0xff);
                    bytes[index] = value;
                    expected[index].newest = value;
                    expected[index].cpu_visible = value;
                }
            }
            transaction->commit();
            for (std::size_t index = offset; index != offset + size; ++index) {
                auto& byte = expected[index];
                byte.version.cpu_write = cpu_version;
                byte.cpu_dirty = true;
                byte.gpu_dirty = false;
                byte.writer = {};
                byte.storage.reset();
            }
        } else if (kind == 1) {
            const auto range = retain(*fixture.memory, gpu_base + offset, size,
                                      ResourceAccess::Write);
            std::vector<std::byte> contents(size);
            for (std::size_t index = 0; index != size; ++index) {
                contents[index] = std::byte((next() >> 16) & 0xff);
            }
            TokenSource writer(sequence++);
            const auto writer_id = writer.token.snapshot().id;
            auto authority = std::make_shared<Storage>(range.bytes, contents);
            if (fused) {
                const std::array request{GpuAccessRequest{range, ResourceAccess::Write}};
                auto accepted = require_value(fixture.memory->accept_accesses(
                    request, writer.token, {authority}));
                CHECK(accepted.lease != nullptr);
                writer.submit();
            } else {
                commit_write(*fixture.memory, range, authority, writer);
            }
            writer.succeed();
            ++gpu_version;
            for (std::size_t index = 0; index != size; ++index) {
                auto& byte = expected[offset + index];
                byte.version.gpu_write = gpu_version;
                byte.cpu_dirty = false;
                byte.gpu_dirty = true;
                byte.writer = writer_id;
                byte.storage = authority;
                byte.newest = contents[index];
            }
        } else if (kind == 2) {
            const auto range = retain(*fixture.memory, gpu_base + offset, size,
                                      ResourceAccess::Read);
            TokenSource reader(sequence++);
            const std::array request{
                GpuAccessRequest{range, ResourceAccess::Read}};
            if (fused) {
                auto accepted = require_value(fixture.memory->accept_accesses(
                    request, reader.token, {std::make_shared<ProbeLease>()}));
                CHECK(accepted.lease != nullptr);
            } else {
                auto reservation = reserve(*fixture.memory, request, reader.token);
                CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
            }
            reader.submit();
            reader.succeed();
            // A GPU read publishes only its intent: CPU dirtiness stays.
            ++gpu_version;
        } else if (kind == 3) {
            CHECK(succeeded(fixture.memory->resolve(
                {backing_a, offset, size}, ResolveReason::Capture)));
            resolve_expected(offset, size);
        } else if (kind == 4) {
            const auto range = retain(*fixture.memory, gpu_base + offset, size,
                                      ResourceAccess::Write);
            TokenSource cancelled(sequence++);
            const std::array request{
                GpuAccessRequest{range, ResourceAccess::Write}};
            if (fused) {
                CHECK(std::holds_alternative<RendererError>(fixture.memory->accept_accesses(
                    request, cancelled.token, {std::make_shared<ProbeLease>()})));
            } else {
                auto reservation = reserve(*fixture.memory, request, cancelled.token);
                reservation.reset();
            }
        } else if (kind == 5) {
            auto transaction = require_value(fixture.memory->prepare_cpu_write(
                {backing_a, offset, size}));
            const auto [preserved_offset, preserved_size] =
                preserved_range(offset, size);
            resolve_expected(preserved_offset, preserved_size);
            ++cpu_version;
            transaction.reset();
        } else {
            const auto first_epoch = fixture.memory->cpu_cache_epoch();
            CHECK(succeeded(fixture.memory->submission_boundary()));
            resolve_expected(0, byte_count);
            for (auto& byte : expected) byte.cpu_dirty = true;
            CHECK(fixture.memory->cpu_cache_epoch() == first_epoch + 1);
            verify(operation);

            {
                std::lock_guard lock(fixture.backend->data_mutex);
                auto& bytes = fixture.backend->find(backing_a)->bytes;
                for (std::size_t index = offset; index != offset + size; ++index) {
                    const auto value = std::byte((next() >> 24) & 0xff);
                    bytes[index] = value;
                    expected[index].newest = value;
                    expected[index].cpu_visible = value;
                }
            }
            verify(operation);
            const auto second_epoch = fixture.memory->cpu_cache_epoch();
            CHECK(succeeded(fixture.memory->submission_boundary()));
            for (auto& byte : expected) byte.cpu_dirty = true;
            CHECK(fixture.memory->cpu_cache_epoch() == second_epoch + 1);
        }
        verify(operation);
    }
}

void test_batch_journal_repeats_publish_once() {
    auto counters = std::make_shared<ResourceMemoryWorkCounters>();
    Fixture fixture(counters);
    fixture.standard_layout();
    auto batch = std::make_shared<ProbeLease>();
    const auto range = retain(*fixture.memory, 0x1000, 16, ResourceAccess::ReadWrite);
    auto storage = std::make_shared<Storage>(range.bytes, std::vector<std::byte>(16, std::byte{0x73}));
    std::vector<GpuAccessRequest> requests{{range, ResourceAccess::Write}};
    std::vector<RetainedResource> allocations{storage};
    for (unsigned i = 0; i < 19; ++i) {
        requests.push_back({retain(*fixture.memory, 0x3020 + i, 1, ResourceAccess::Read), ResourceAccess::Read});
        allocations.push_back(std::make_shared<ProbeLease>());
    }
    TokenSource first(1);
    require_value(fixture.memory->accept_batch_accesses(requests, first.token, allocations, batch));
    *counters = {};
    constexpr unsigned count = 1000;
    std::vector<std::unique_ptr<TokenSource>> tokens;
    for (unsigned i = 2; i <= count; ++i) tokens.push_back(std::make_unique<TokenSource>(i));
    const auto begin = std::chrono::steady_clock::now();
    for (const auto& token : tokens) {
        auto accepted = require_value(fixture.memory->accept_batch_accesses(
            requests, token->token, allocations, batch));
        CHECK(accepted.dependencies.empty());
    }
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    CHECK(counters->interval_comparisons == 0);
    CHECK(counters->interval_visits == 0);
    CHECK(counters->backing_comparisons == 0);
    CHECK(counters->mapping_comparisons == 0);
    CHECK(counters->journal_publications == 0);
    CHECK(succeeded(fixture.memory->advance_cpu_epoch()));
    CHECK(counters->journal_publications == 0);
    CHECK(succeeded(fixture.memory->flush_access_journal()));
    CHECK(counters->journal_publications == 1);
    CHECK(counters->journal_entries_published == 20);
    const auto dirty = inspect(*fixture.memory, range.bytes);
    CHECK(dirty.size() == 1 && dirty[0].last_writer.id() == tokens.back()->token.id());
    CHECK(succeeded(fixture.memory->flush_access_journal()));
    CHECK(counters->journal_publications == 1);
    std::fprintf(stdout, "access journal repeated twenty-access packet: %.1f ns/call\n",
        std::chrono::duration<double, std::nano>(elapsed).count() / tokens.size());
}

void test_batch_journal_cpu_read_flushes_newest_writer() {
    auto counters = std::make_shared<ResourceMemoryWorkCounters>();
    Fixture fixture(counters);
    fixture.standard_layout();
    auto batch = std::make_shared<ProbeLease>();
    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(range.bytes, std::vector<std::byte>(8, std::byte{0x68}));
    const std::array requests{GpuAccessRequest{range, ResourceAccess::Write}};
    const std::array<RetainedResource, 1> allocations{storage};
    TokenSource first(1), latest(2);
    require_value(fixture.memory->accept_batch_accesses(requests, first.token, allocations, batch));
    require_value(fixture.memory->accept_batch_accesses(requests, latest.token, allocations, batch));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->prepare_cpu_read(range.bytes, ResolveReason::CpuRead)));
    CHECK(counters->journal_publications == 1);
    CHECK(fixture.backend->waits == 0);
    CHECK(inspect(*fixture.memory, range.bytes)[0].last_writer.id() == latest.token.id());
    first.submit();
    first.succeed();
    latest.submit();
    // The old writer is finished, but the newest writer still prevents reads.
    CHECK(std::holds_alternative<RendererError>(fixture.memory->prepare_cpu_read(range.bytes, ResolveReason::CpuRead)));
    CHECK(fixture.backend->waits == 1);
    latest.succeed();
    auto transaction = require_value(fixture.memory->prepare_cpu_read(range.bytes, ResolveReason::CpuRead));
    CHECK(fixture.backend->find(backing_a)->bytes[0] == std::byte{0x68});
}

void test_batch_journal_failure_and_partial_ownership() {
    auto counters = std::make_shared<ResourceMemoryWorkCounters>();
    Fixture fixture(counters);
    fixture.standard_layout();
    auto batch = std::make_shared<ProbeLease>();
    const auto range = retain(*fixture.memory, 0x1000, 16, ResourceAccess::Write);
    const auto middle = retain(*fixture.memory, 0x2004, 4, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(range.bytes, std::vector<std::byte>(16, std::byte{0x12}));
    auto other = std::make_shared<Storage>(middle.bytes, std::vector<std::byte>(4, std::byte{0x34}));
    const std::array requests{GpuAccessRequest{range, ResourceAccess::Write}};
    const std::array partial{GpuAccessRequest{middle, ResourceAccess::Write}};
    const std::array<RetainedResource, 1> allocations{storage}, replacements{other};
    TokenSource first(1), second(2), third(3), fourth(4);
    require_value(fixture.memory->accept_batch_accesses(requests, first.token, allocations, batch));
    fixture.memory->discard_access_journal(batch);
    CHECK(succeeded(first.completion.reject({CommandStatus::Stopped, "synthetic encoding failure"})));
    CHECK(succeeded(fixture.memory->flush_access_journal()));
    CHECK(counters->journal_publications == 0);
    CHECK(!inspect(*fixture.memory, range.bytes)[0].gpu_dirty);
    const std::array<RetainedResource, 1> invalid{std::make_shared<ProbeLease>()};
    CHECK(std::holds_alternative<RendererError>(fixture.memory->accept_batch_accesses(requests, second.token, invalid, batch)));
    CHECK(!inspect(*fixture.memory, range.bytes)[0].gpu_dirty);
    require_value(fixture.memory->accept_batch_accesses(requests, second.token, allocations, batch));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->accept_batch_accesses(
        partial, third.token, invalid, batch)));
    fixture.memory->discard_access_journal(std::make_shared<ProbeLease>());
    CHECK(counters->journal_publications == 0);
    CHECK(inspect(*fixture.memory, range.bytes)[0].last_writer.id() == second.token.id());
    require_value(fixture.memory->accept_batch_accesses(partial, third.token, replacements, batch));
    require_value(fixture.memory->accept_batch_accesses(requests, fourth.token, allocations, batch));
    auto authority = require_value(fixture.memory->inspect_storage(range.bytes));
    CHECK(authority.size() == 1);
    CHECK(authority[0].storage == storage);
    CHECK(authority[0].dirty.last_writer.id() == fourth.token.id());
    TokenSource fifth(5), sixth(6);
    require_value(fixture.memory->accept_batch_accesses(requests, fifth.token, allocations, batch));
    require_value(fixture.memory->accept_batch_accesses(partial, sixth.token, replacements, batch));
    authority = require_value(fixture.memory->inspect_storage(range.bytes));
    CHECK(authority.size() == 3);
    CHECK(authority[0].storage == storage && authority[0].dirty.bytes.size == 4);
    CHECK(authority[1].storage == other && authority[1].dirty.bytes.size == 4);
    CHECK(authority[2].storage == storage && authority[2].dirty.bytes.size == 8);
    CHECK(authority[0].dirty.last_writer.id() == fifth.token.id());
    CHECK(authority[1].dirty.last_writer.id() == sixth.token.id());
}

void test_batch_journal_cross_queue_and_failed_feedback() {
    Fixture fixture;
    fixture.standard_layout();
    auto batch = std::make_shared<ProbeLease>();
    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::ReadWrite);
    auto storage = std::make_shared<Storage>(range.bytes, std::vector<std::byte>(8, std::byte{0x53}));
    const std::array reads{GpuAccessRequest{range, ResourceAccess::Read}};
    const std::array writes{GpuAccessRequest{range, ResourceAccess::Write}};
    const std::array<RetainedResource, 1> allocations{storage};
    SubmissionCompletion reader({9, 1}, {}, std::make_shared<NoopActions>());
    require_value(fixture.memory->accept_accesses(reads, reader.token(), {storage}));
    TokenSource first(1), latest(2);
    auto accepted = require_value(fixture.memory->accept_batch_accesses(writes, first.token, allocations, batch));
    CHECK(accepted.dependencies.size() == 1 && accepted.dependencies[0].id() == reader.token().id());
    accepted = require_value(fixture.memory->accept_batch_accesses(writes, latest.token, allocations, batch));
    CHECK(accepted.dependencies.size() == 1 && accepted.dependencies[0].id() == reader.token().id());
    auto other_batch = std::make_shared<ProbeLease>();
    SubmissionCompletion consumer({7, 1}, {}, std::make_shared<NoopActions>());
    accepted = require_value(fixture.memory->accept_batch_accesses(reads, consumer.token(), allocations, other_batch));
    CHECK(accepted.dependencies.size() == 1 && accepted.dependencies[0].id() == latest.token.id());
    fixture.memory->discard_access_journal(other_batch);
    // An observation already exposed this authority. Preserve its failure,
    // as eager acceptance does, rather than making stale CPU bytes authoritative.
    CHECK(succeeded(latest.completion.reject({CommandStatus::Stopped, "synthetic observed failure"})));
    CHECK(inspect(*fixture.memory, range.bytes)[0].gpu_dirty);
    CHECK(!succeeded(fixture.memory->resolve(range.bytes, ResolveReason::CpuRead)));

    Fixture failed;
    failed.standard_layout();
    const auto fresh = retain(*failed.memory, 0x1000, 8, ResourceAccess::Write);
    const std::array fresh_writes{GpuAccessRequest{fresh, ResourceAccess::Write}};
    TokenSource rejected(1);
    require_value(failed.memory->accept_batch_accesses(fresh_writes, rejected.token, allocations, batch));
    CHECK(succeeded(rejected.completion.reject({CommandStatus::Stopped, "synthetic unobserved failure"})));
    CHECK(!succeeded(failed.memory->flush_access_journal()));
    CHECK(!inspect(*failed.memory, fresh.bytes)[0].gpu_dirty);
}

void test_fused_acceptance_before_submission_and_rejection() {
    Fixture fixture;
    fixture.standard_layout();
    const auto whole = retain(*fixture.memory, 0x1000, 16, ResourceAccess::ReadWrite);
    const auto middle = retain(*fixture.memory, 0x2004, 4, ResourceAccess::Write);
    TokenSource first(1), rejected(2), next(3);
    const std::array initial{GpuAccessRequest{whole, ResourceAccess::Write}};
    auto original = std::make_shared<Storage>(whole.bytes,
        std::vector<std::byte>(16, std::byte{0x11}));
    auto accepted = require_value(fixture.memory->accept_accesses(
        initial, first.token, {original}));
    CHECK(accepted.lease != nullptr);
    CHECK(accepted.dependencies.empty());
    CHECK(first.token.snapshot().phase == SubmissionPhase::Accepted);
    auto state = require_value(fixture.memory->inspect_storage(whole.bytes));
    CHECK(state.size() == 1);
    CHECK(state.front().storage == original);
    CHECK(state.front().dirty.last_writer.id() == first.token.id());
    const auto original_version = state.front().dirty.version;

    // A valid first entry followed by invalid storage must publish neither.
    const auto tail = retain(*fixture.memory, 0x100c, 4, ResourceAccess::Write);
    const std::array invalid{GpuAccessRequest{middle, ResourceAccess::Write},
                             GpuAccessRequest{tail, ResourceAccess::Write}};
    auto replacement = std::make_shared<Storage>(middle.bytes,
        std::vector<std::byte>(4, std::byte{0x77}));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->accept_accesses(
        invalid, rejected.token, {replacement, std::make_shared<ProbeLease>()})));
    state = require_value(fixture.memory->inspect_storage(whole.bytes));
    CHECK(state.size() == 1);
    CHECK(state.front().storage == original);
    CHECK(state.front().dirty.version == original_version);

    const std::array update{GpuAccessRequest{middle, ResourceAccess::Write}};
    auto updated = require_value(fixture.memory->accept_accesses(
        update, next.token, {replacement}));
    CHECK(updated.dependencies.size() == 1);
    CHECK(updated.dependencies.front().id() == first.token.id());
    state = require_value(fixture.memory->inspect_storage(whole.bytes));
    CHECK(state.size() == 3);
    CHECK(state[0].storage == original && state[0].dirty.bytes.size == 4);
    CHECK(state[1].storage == replacement && state[1].dirty.bytes.size == 4);
    CHECK(state[1].dirty.version.gpu_write == original_version.gpu_write + 1);
    CHECK(state[2].storage == original && state[2].dirty.bytes.size == 8);
    first.submit();
    first.succeed();
    next.submit();
    next.succeed();
    CHECK(succeeded(fixture.memory->resolve(whole.bytes, ResolveReason::CpuRead)));
    const auto& bytes = fixture.backend->find(backing_a)->bytes;
    for (std::size_t i = 0; i != 16; ++i)
        CHECK(bytes[i] == (i >= 4 && i < 8 ? std::byte{0x77} : std::byte{0x11}));
}

void test_fused_acceptance_preserves_reader_hazards_and_failure() {
    Fixture fixture;
    fixture.standard_layout();
    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::ReadWrite);
    TokenSource writer(1), reader(2), failed(3);
    auto storage = std::make_shared<Storage>(range.bytes,
        std::vector<std::byte>(8, std::byte{0x33}));
    const std::array write{GpuAccessRequest{range, ResourceAccess::Write}};
    auto written = require_value(fixture.memory->accept_accesses(write, writer.token, {storage}));
    writer.submit();
    writer.succeed();
    const std::array read{GpuAccessRequest{range, ResourceAccess::Read}};
    auto read_storage = std::make_shared<ProbeLease>();
    std::weak_ptr<ProbeLease> retained_reader = read_storage;
    auto reading = require_value(fixture.memory->accept_accesses(read, reader.token, {read_storage}));
    read_storage.reset();
    CHECK(!retained_reader.expired());
    CHECK(!succeeded(fixture.memory->resolve(range.bytes, ResolveReason::CpuRead)));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->prepare_cpu_write(range.bytes)));
    CHECK(fixture.backend->writes == 0);
    reader.submit();
    reader.succeed();
    CHECK(succeeded(fixture.memory->resolve(range.bytes, ResolveReason::CpuRead)));
    CHECK(retained_reader.expired());

    auto failed_storage = std::make_shared<Storage>(range.bytes,
        std::vector<std::byte>(8, std::byte{0xee}));
    auto failure = require_value(fixture.memory->accept_accesses(write, failed.token, {failed_storage}));
    failed.submit();
    failed.fail();
    CHECK(!succeeded(fixture.memory->resolve(range.bytes, ResolveReason::CpuRead)));
    const auto current = require_value(fixture.memory->inspect_storage(range.bytes));
    CHECK(current.front().dirty.gpu_dirty);
    CHECK(current.front().storage == failed_storage);
    CHECK(fixture.backend->find(backing_a)->bytes[0] == std::byte{0x33});
}

void test_fused_acceptance_lease_retains_registry_state() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource reader(1);
    const auto range = retain(*fixture.memory, 0x1000, 4, ResourceAccess::Read);
    const std::array request{GpuAccessRequest{range, ResourceAccess::Read}};
    auto storage = std::make_shared<ProbeLease>();
    std::weak_ptr<ProbeLease> observed = storage;
    auto accepted = require_value(fixture.memory->accept_accesses(request, reader.token, {storage}));
    fixture.memory.reset();
    storage.reset();
    CHECK(!observed.expired());
    reader.submit();
    reader.succeed();
    CHECK(!observed.expired());
    accepted.lease.reset();
    CHECK(observed.expired());
}

void test_fused_acceptance_respects_pending_order_and_default_path() {
    Fixture fixture;
    fixture.standard_layout();
    const auto first = retain(*fixture.memory, 0x1000, 4, ResourceAccess::Read);
    const auto second = retain(*fixture.memory, 0x3000, 4, ResourceAccess::Read);
    const std::array first_request{GpuAccessRequest{first, ResourceAccess::Read}};
    const std::array second_request{GpuAccessRequest{second, ResourceAccess::Read}};
    TokenSource earlier(1), later(2), fallback(3);
    auto pending = reserve(*fixture.memory, first_request, earlier.token);
    CHECK(std::holds_alternative<RendererError>(fixture.memory->accept_accesses(
        second_request, later.token, {std::make_shared<ProbeLease>()})));
    CHECK(covering(inspect(*fixture.memory, second.bytes), backing_b, 0)->cpu_dirty);
    pending.reset();
    auto accepted = require_value(fixture.memory->accept_accesses(
        second_request, later.token, {std::make_shared<ProbeLease>()}));
    later.submit();
    later.succeed();
    // Exercise the compatible default used by existing derived implementations.
    auto compatible = require_value(fixture.memory->ResourceMemory::accept_accesses(
        first_request, fallback.token, {std::make_shared<ProbeLease>()}));
    CHECK(compatible.lease != nullptr);
    fallback.submit();
    fallback.succeed();
    CHECK(succeeded(fixture.memory->submission_boundary()));
}

void test_failed_resolves_retain_dirty_authority() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource token(5);
    const auto target = retain(*fixture.memory, 0x1010, 6, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(target.bytes,
        std::vector<std::byte>(6, std::byte{0x5a}));
    std::weak_ptr<const ResourceLease> authority = storage;
    commit_write(*fixture.memory, target, storage, token);
    storage->fail_reads = true;
    token.succeed();
    storage.reset();
    CHECK(!succeeded(fixture.memory->resolve(target.bytes, ResolveReason::Capture)));
    CHECK(!authority.expired());
    auto dirty = inspect(*fixture.memory, target.bytes);
    CHECK(covering(dirty, backing_a, 0x10)->gpu_dirty);
    CHECK(require_value(fixture.memory->inspect_storage(target.bytes)).front().storage ==
          authority.lock());

    auto retained_storage = std::dynamic_pointer_cast<const Storage>(authority.lock());
    CHECK(retained_storage != nullptr);
    const_cast<Storage*>(retained_storage.get())->fail_reads = false;
    fixture.backend->fail_writes = true;
    fixture.backend->partial_write_bytes = 2;
    CHECK(!succeeded(fixture.memory->resolve(target.bytes, ResolveReason::Capture)));
    dirty = inspect(*fixture.memory, target.bytes);
    CHECK(covering(dirty, backing_a, 0x10)->gpu_dirty);
    CHECK(!authority.expired());
    CHECK(require_value(fixture.memory->inspect_storage(target.bytes)).front().storage ==
          retained_storage);
    fixture.backend->fail_writes = false;
    CHECK(succeeded(fixture.memory->resolve(target.bytes, ResolveReason::Capture)));
    CHECK(!covering(inspect(*fixture.memory, target.bytes), backing_a, 0x10)->gpu_dirty);
    CHECK(!require_value(fixture.memory->inspect_storage(target.bytes)).front().storage);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(std::all_of(bytes.begin() + 0x10, bytes.begin() + 0x16,
                          [](std::byte byte) { return byte == std::byte{0x5a}; }));
    }

    TokenSource failed(6);
    const auto failed_target = retain(*fixture.memory, 0x1020, 4, ResourceAccess::Write);
    auto failed_storage = std::make_shared<Storage>(
        failed_target.bytes, std::vector<std::byte>(4, std::byte{0xf0}));
    commit_write(*fixture.memory, failed_target, failed_storage, failed);
    failed.fail();
    CHECK(!succeeded(fixture.memory->resolve(
        failed_target.bytes, ResolveReason::CpuRead)));
    CHECK(covering(inspect(*fixture.memory, failed_target.bytes), backing_a, 0x20)
              ->gpu_dirty);
    CHECK(require_value(fixture.memory->inspect_storage(failed_target.bytes)).front().storage ==
          failed_storage);
}

// A guest boundary writes buffer authorities back but keeps GPU-resident ones
// authoritative until a full submission boundary or explicit resolve.
void test_guest_boundary_keeps_resident_authority() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource buffer_token(5);
    const auto buffer_target = retain(*fixture.memory, 0x1010, 6, ResourceAccess::Write);
    auto buffer = std::make_shared<Storage>(buffer_target.bytes,
        std::vector<std::byte>(6, std::byte{0x5a}));
    commit_write(*fixture.memory, buffer_target, buffer, buffer_token);
    TokenSource texture_token(6);
    const auto texture_target = retain(*fixture.memory, 0x3008, 4, ResourceAccess::Write);
    auto texture = std::make_shared<Storage>(texture_target.bytes,
        std::vector<std::byte>(4, std::byte{0x77}));
    texture->resident = true;
    commit_write(*fixture.memory, texture_target, texture, texture_token);
    buffer_token.succeed();
    texture_token.succeed();

    CHECK(succeeded(fixture.memory->guest_boundary()));
    CHECK(!covering(inspect(*fixture.memory, buffer_target.bytes), backing_a, 0x10)->gpu_dirty);
    CHECK(covering(inspect(*fixture.memory, texture_target.bytes), backing_b, 0x08)->gpu_dirty);
    CHECK(texture->reads == 0);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(std::all_of(bytes.begin() + 0x10, bytes.begin() + 0x16,
                          [](std::byte byte) { return byte == std::byte{0x5a}; }));
        CHECK(fixture.backend->find(backing_b)->bytes[0x08] == std::byte{0});
    }
    // A second guest boundary still keeps it; the full boundary resolves it.
    CHECK(succeeded(fixture.memory->guest_boundary()));
    CHECK(covering(inspect(*fixture.memory, texture_target.bytes), backing_b, 0x08)->gpu_dirty);
    CHECK(succeeded(fixture.memory->submission_boundary()));
    CHECK(!covering(inspect(*fixture.memory, texture_target.bytes), backing_b, 0x08)->gpu_dirty);
    std::lock_guard lock(fixture.backend->data_mutex);
    const auto& bytes = fixture.backend->find(backing_b)->bytes;
    CHECK(std::all_of(bytes.begin() + 0x08, bytes.begin() + 0x0c,
                      [](std::byte byte) { return byte == std::byte{0x77}; }));
}

void test_deferred_host_store_excludes_aliases_without_waiting() {
    Fixture fixture;
    fixture.standard_layout();
    const auto range = retain(*fixture.memory, 0x1008, 8, ResourceAccess::ReadWrite);
    const auto alias = retain(*fixture.memory, 0x200a, 4, ResourceAccess::ReadWrite);
    const auto disjoint = retain(*fixture.memory, 0x1020, 8, ResourceAccess::ReadWrite);
    TokenSource prefix(1), later(2), blocked(3);
    auto storage = std::make_shared<Storage>(range.bytes,
        std::vector<std::byte>(8, std::byte{0x61}));
    commit_write(*fixture.memory, range, storage, prefix);
    auto boundary = require_value(fixture.memory->capture_guest_boundary(false));
    const auto before = require_value(fixture.memory->inspect_storage(range.bytes));
    auto guard = require_value(fixture.memory->defer_host_store(range.bytes));
    CHECK(fixture.memory->has_deferred_host_store());
    CHECK(fixture.memory->deferred_host_store(alias.bytes));
    CHECK(!fixture.memory->deferred_host_store(disjoint.bytes));
    CHECK(fixture.backend->waits == 0 && fixture.backend->writes == 0);
    CHECK(fixture.backend->find(backing_a)->bytes[8] == std::byte{});
    const auto after = require_value(fixture.memory->inspect_storage(range.bytes));
    CHECK(after.size() == before.size());
    CHECK(after[0].storage == before[0].storage && after[0].dirty.gpu_dirty);
    CHECK(after[0].dirty.version == before[0].dirty.version);
    CHECK(after[0].dirty.last_writer.id() == prefix.token.id());
    const auto epoch = fixture.memory->cpu_cache_epoch();
    CHECK(succeeded(fixture.memory->advance_cpu_epoch()));
    CHECK(fixture.memory->cpu_cache_epoch() == epoch + 1);
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->prepare_cpu_read(alias.bytes, ResolveReason::CpuRead)));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->prepare_cpu_write(alias.bytes)));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->defer_host_store(alias.bytes)));
    const std::array aliased_read{GpuAccessRequest{alias, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->accept_accesses(aliased_read, blocked.token, {storage})));
    const std::array aliased_write{GpuAccessRequest{alias, ResourceAccess::Write}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(aliased_write, blocked.token)));
    const std::array independent{GpuAccessRequest{disjoint, ResourceAccess::Read}};
    require_value(fixture.memory->accept_accesses(independent, later.token, {std::make_shared<ProbeLease>()}));
    auto read = require_value(fixture.memory->prepare_cpu_read(disjoint.bytes, ResolveReason::CpuRead));
    read.reset();
    CHECK(fixture.backend->waits == 0 && fixture.backend->writes == 0);
    // Captured prefix publication bypasses the later store's exclusion and
    // neither waits for nor admits the independent Accepted successor.
    prefix.succeed();
    CHECK(succeeded(boundary->publish()));
    CHECK(fixture.backend->find(backing_a)->bytes[8] == std::byte{0x61});
    CHECK(later.token.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(fixture.backend->waits == 0);
    guard.reset();
    CHECK(!fixture.memory->has_deferred_host_store());
    CHECK(!fixture.memory->deferred_host_store(alias.bytes));
    auto admitted = require_value(fixture.memory->prepare_cpu_read(alias.bytes, ResolveReason::CpuRead));
    admitted.reset();
    require_value(fixture.memory->accept_accesses(aliased_read, blocked.token, {storage}));
}

void test_deferred_host_store_blocks_cached_journal_admission() {
    Fixture fixture;
    fixture.standard_layout();
    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::ReadWrite);
    auto storage = std::make_shared<Storage>(range.bytes, std::vector<std::byte>(8));
    const std::array requests{GpuAccessRequest{range, ResourceAccess::Write}};
    const std::array<RetainedResource, 1> allocations{storage};
    const auto batch = std::make_shared<ProbeLease>();
    TokenSource first(1), second(2);
    require_value(fixture.memory->accept_batch_accesses(requests, first.token, allocations, batch));
    auto guard = require_value(fixture.memory->defer_host_store(range.bytes));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->accept_batch_accesses(
        requests, second.token, allocations, batch)));
    CHECK(fixture.backend->waits == 0 && fixture.backend->writes == 0);
    guard.reset();
    require_value(fixture.memory->accept_batch_accesses(requests, second.token, allocations, batch));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->defer_host_store({backing_a, 0x40, 1})));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->defer_host_store({backing_a, 0, 0})));
    auto disjoint = require_value(fixture.memory->prepare_cpu_read({backing_a, 0x20, 8}, ResolveReason::CpuRead));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->defer_host_store({backing_a, 0x24, 1})));
}

void test_deferred_host_store_invalidates_native_tracker() {
    using armigo::memory::WriteTracker;
    const auto page = static_cast<std::size_t>(getpagesize());
    void* address = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(address != MAP_FAILED);
    if (address == MAP_FAILED) return;
    auto owner = std::shared_ptr<void>(address, [page](void* pointer) {
        WriteTracker::remove_range(pointer);
        munmap(pointer, page);
    });
    auto* bytes = static_cast<std::byte*>(address);
    std::fill_n(bytes, page, std::byte{0x38});
    WriteTracker::add_range(address, page);
    auto backend = std::make_shared<Backend>();
    backend->add(backing_a, 0x40);
    ResourceMemoryOptions options{7, backend, CoherenceMode::SubmissionBoundary, 1};
    // The shader package can track code independently of the registry tracker.
    options.host_addressable_backings = true;
    auto memory = require_value(make_resource_memory(options));
    CHECK(succeeded(memory->register_backing({backing_a, 0x40, std::make_shared<ProbeLease>()})));
    auto mapping = mapped(0x1000, 0, 0x40, reinterpret_cast<std::uint64_t>(address));
    mapping.owner = owner;
    CHECK(succeeded(memory->register_mapping({mapping_a, backing_a, mapping})));
    const BackingRange range{backing_a, 8, 4};
    for (const bool protected_first : {false, true}) {
        if (protected_first) CHECK(WriteTracker::protect_checked(address, page));
        const auto old_stamp = WriteTracker::now();
        CHECK(!WriteTracker::written_since(bytes + 8, 4, old_stamp));
        auto guard = require_value(memory->defer_host_store(range));
        CHECK(WriteTracker::written_since(bytes + 8, 4, old_stamp));
        CHECK(memory->cpu_written_since(range, WriteTracker::now()));
        CHECK(std::all_of(bytes, bytes + page, [](std::byte value) { return value == std::byte{0x38}; }));
        CHECK(memory->backing_host_range(backing_a).has_value());
        guard.reset();
        CHECK(!memory->has_deferred_host_store());
    }
}

void test_deferred_host_store_invalidates_partial_tracking() {
    using armigo::memory::WriteTracker;
    const auto page = static_cast<std::size_t>(getpagesize());
    for (const bool track_second : {false, true}) {
        void* address = mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANON, -1, 0);
        CHECK(address != MAP_FAILED);
        if (address == MAP_FAILED) return;
        auto owner = std::shared_ptr<void>(address, [page](void* pointer) {
            WriteTracker::remove_range(pointer);
            WriteTracker::remove_range(static_cast<std::byte*>(pointer) + page);
            munmap(pointer, page * 2);
        });
        auto* bytes = static_cast<std::byte*>(address);
        std::fill_n(bytes, page * 2, std::byte{0x49});
        WriteTracker::add_range(address, page);
        if (track_second) WriteTracker::add_range(bytes + page, page);
        CHECK(WriteTracker::protect_checked(address, page));
        if (track_second) CHECK(WriteTracker::protect_checked(bytes + page, page));
        const auto stamp = WriteTracker::now();
        CHECK(!WriteTracker::tracked(bytes + page - 4, 8));
        CHECK(!WriteTracker::written_since(bytes + page - 4, 4, stamp));
        if (track_second) CHECK(!WriteTracker::written_since(bytes + page, 4, stamp));
        auto backend = std::make_shared<Backend>();
        backend->add(backing_a, page * 2);
        ResourceMemoryOptions options{7, backend, CoherenceMode::SubmissionBoundary, 1};
        options.host_addressable_backings = true;
        auto memory = require_value(make_resource_memory(options));
        CHECK(succeeded(memory->register_backing({backing_a, page * 2, std::make_shared<ProbeLease>()})));
        const Mapping mapping{0x100000, page * 2, 0,
            reinterpret_cast<std::uint64_t>(address), page * 2, page, owner};
        CHECK(succeeded(memory->register_mapping({mapping_a, backing_a, mapping})));
        auto guard = require_value(memory->defer_host_store({backing_a, page - 4, 8}));
        CHECK(WriteTracker::written_since(bytes + page - 4, 4, stamp));
        if (track_second) CHECK(WriteTracker::written_since(bytes + page, 4, stamp));
        CHECK(std::all_of(bytes, bytes + page * 2,
            [](std::byte value) { return value == std::byte{0x49}; }));
        CHECK(backend->waits == 0 && backend->writes == 0);
    }
}

void test_captured_boundary_preserves_later_alias_authority() {
    for (const bool journaled : {false, true}) {
        Fixture fixture;
        fixture.standard_layout();
        const auto whole = retain(*fixture.memory, 0x1000, 16, ResourceAccess::ReadWrite);
        const auto middle = retain(*fixture.memory, 0x2004, 4, ResourceAccess::ReadWrite);
        TokenSource earlier(1), later(2), reader(3);
        auto original = std::make_shared<Storage>(whole.bytes,
            std::vector<std::byte>(16, std::byte{0x31}));
        commit_write(*fixture.memory, whole, original, earlier);
        auto boundary = require_value(fixture.memory->capture_guest_boundary(false));
        auto replacement = std::make_shared<Storage>(middle.bytes,
            std::vector<std::byte>(4, std::byte{0x72}));
        const std::array writes{GpuAccessRequest{middle, ResourceAccess::Write}};
        auto batch = std::make_shared<ProbeLease>();
        if (journaled) {
            const std::array<RetainedResource, 1> storage{replacement};
            require_value(fixture.memory->accept_batch_accesses(writes, later.token, storage, batch));
        } else {
            require_value(fixture.memory->accept_accesses(writes, later.token, {replacement}));
        }
        // A later reader must not make publication wait for an unsubmitted token.
        const std::array reads{GpuAccessRequest{whole, ResourceAccess::Read}};
        require_value(fixture.memory->accept_accesses(reads, reader.token, {original}));
        earlier.succeed();
        CHECK(succeeded(boundary->publish()));
        CHECK(later.token.snapshot().phase == SubmissionPhase::Accepted);
        CHECK(reader.token.snapshot().phase == SubmissionPhase::Accepted);
        CHECK(fixture.backend->waits == 0);
        CHECK(fixture.backend->writes == 1);
        CHECK(std::all_of(fixture.backend->find(backing_a)->bytes.begin(),
                          fixture.backend->find(backing_a)->bytes.begin() + 16,
                          [](std::byte value) { return value == std::byte{0x31}; }));
        auto authority = require_value(fixture.memory->inspect_storage(whole.bytes));
        CHECK(authority.size() == 3);
        CHECK(!authority[0].dirty.gpu_dirty && !authority[2].dirty.gpu_dirty);
        CHECK(authority[1].dirty.gpu_dirty && authority[1].storage == replacement);
        CHECK(authority[1].dirty.last_writer.id() == later.token.id());
        const auto epoch = fixture.memory->cpu_cache_epoch();
        CHECK(succeeded(boundary->publish()));
        CHECK(fixture.backend->writes == 1);
        CHECK(fixture.memory->cpu_cache_epoch() == epoch);
        later.submit();
        later.succeed();
        reader.submit();
        reader.succeed();
        CHECK(succeeded(fixture.memory->submission_boundary()));
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(bytes[3] == std::byte{0x31} && bytes[4] == std::byte{0x72});
        CHECK(bytes[7] == std::byte{0x72} && bytes[8] == std::byte{0x31});
    }
}

void test_captured_boundary_same_storage_and_unpublished_journal() {
    Fixture fixture;
    fixture.standard_layout();
    const auto whole = retain(*fixture.memory, 0x1000, 16, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(whole.bytes,
        std::vector<std::byte>(16, std::byte{0x24}));
    TokenSource earlier(1), later(2);
    commit_write(*fixture.memory, whole, storage, earlier);
    auto boundary = require_value(fixture.memory->capture_guest_boundary(false));
    const std::array writes{GpuAccessRequest{whole, ResourceAccess::Write}};
    require_value(fixture.memory->accept_accesses(writes, later.token, {storage}));
    earlier.succeed();
    CHECK(succeeded(boundary->publish()));
    auto current = inspect(*fixture.memory, whole.bytes);
    CHECK(current.size() == 1 && current[0].gpu_dirty);
    CHECK(current[0].last_writer.id() == later.token.id());

    auto next = require_value(fixture.memory->capture_guest_boundary(false));
    later.submit();
    later.succeed();
    TokenSource last(3);
    auto batch = std::make_shared<ProbeLease>();
    const std::array<RetainedResource, 1> allocations{storage};
    require_value(fixture.memory->accept_batch_accesses(writes, last.token, allocations, batch));
    CHECK(succeeded(next->publish()));
    CHECK(last.token.snapshot().phase == SubmissionPhase::Accepted);
    current = inspect(*fixture.memory, whole.bytes);
    CHECK(current.size() == 1 && current[0].gpu_dirty);
    CHECK(current[0].last_writer.id() == last.token.id());
}

void test_captured_boundary_failure_retry_and_lifetime() {
    for (const bool gpu_failure : {false, true}) {
        Fixture fixture;
        fixture.standard_layout();
        const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Write);
        auto storage = std::make_shared<Storage>(range.bytes,
            std::vector<std::byte>(8, std::byte{0x63}));
        TokenSource token(1);
        commit_write(*fixture.memory, range, storage, token);
        auto boundary = require_value(fixture.memory->capture_guest_boundary(false));
        if (gpu_failure) token.fail();
        else token.succeed();
        fixture.backend->fail_writes = true;
        fixture.backend->partial_write_bytes = 3;
        CHECK(!succeeded(boundary->publish()));
        CHECK(inspect(*fixture.memory, range.bytes)[0].gpu_dirty);
        if (gpu_failure) {
            CHECK(fixture.backend->writes == 0);
            CHECK(!succeeded(boundary->publish()));
        } else {
            CHECK(fixture.backend->writes == 1);
            fixture.backend->fail_writes = false;
            // The snapshot keeps the registry state and backing ownership.
            fixture.memory.reset();
            CHECK(succeeded(boundary->publish()));
            CHECK(fixture.backend->writes == 2);
            CHECK(succeeded(boundary->publish()));
            CHECK(fixture.backend->writes == 2);
        }
    }
    // Prefix reads are ordering participants even without dirty authority.
    Fixture fixture;
    fixture.standard_layout();
    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Read);
    TokenSource reader(1);
    const std::array reads{GpuAccessRequest{range, ResourceAccess::Read}};
    require_value(fixture.memory->accept_accesses(reads, reader.token, {std::make_shared<ProbeLease>()}));
    reader.submit();
    auto boundary = require_value(fixture.memory->capture_guest_boundary(false));
    reader.fail();
    CHECK(!succeeded(boundary->publish()));
    CHECK(fixture.backend->writes == 0);
}

void test_captured_boundary_waits_only_for_its_prefix() {
    Fixture fixture;
    fixture.standard_layout();
    const auto first = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Write);
    const auto second = retain(*fixture.memory, 0x3000, 8, ResourceAccess::Write);
    TokenSource earlier(1), later(2);
    auto original = std::make_shared<Storage>(first.bytes,
        std::vector<std::byte>(8, std::byte{0x47}));
    auto replacement = std::make_shared<Storage>(second.bytes,
        std::vector<std::byte>(8, std::byte{0x82}));
    commit_write(*fixture.memory, first, original, earlier);
    auto boundary = require_value(fixture.memory->capture_guest_boundary(false));
    // Synthetic waits return without completing feedback: publication must
    // recheck that token and preserve all authority for retry.
    CHECK(!succeeded(boundary->publish()));
    CHECK(fixture.backend->waits == 1 && fixture.backend->writes == 0);
    commit_write(*fixture.memory, second, replacement, later);
    later.fail();
    earlier.succeed();
    CHECK(succeeded(boundary->publish()));
    CHECK(fixture.backend->waits == 1 && fixture.backend->writes == 1);
    CHECK(fixture.backend->find(backing_a)->bytes[0] == std::byte{0x47});
    CHECK(fixture.backend->find(backing_b)->bytes[0] == std::byte{0});
    CHECK(inspect(*fixture.memory, second.bytes)[0].gpu_dirty);
}

void test_captured_boundary_resident_and_in_place_policy() {
    Fixture fixture;
    fixture.standard_layout();
    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(range.bytes,
        std::vector<std::byte>(8, std::byte{0x53}));
    storage->resident = true;
    TokenSource token(1);
    commit_write(*fixture.memory, range, storage, token);
    auto resident = require_value(fixture.memory->capture_guest_boundary(true));
    token.succeed();
    CHECK(succeeded(resident->publish()));
    CHECK(storage->reads == 0 && fixture.backend->writes == 0);
    CHECK(inspect(*fixture.memory, range.bytes)[0].gpu_dirty);
    auto repeated = require_value(fixture.memory->capture_guest_boundary(true));
    CHECK(succeeded(repeated->publish()));
    fixture.backend->find(backing_a)->bytes[0] = std::byte{0x99};
    auto modified = require_value(fixture.memory->capture_guest_boundary(true));
    CHECK(!succeeded(modified->publish()));
    auto full = require_value(fixture.memory->capture_guest_boundary(false));
    CHECK(succeeded(full->publish()));
    CHECK(fixture.backend->find(backing_a)->bytes[0] == std::byte{0x53});

    storage->resident = false;
    storage->zero_copy = true;
    TokenSource inplace(2);
    commit_write(*fixture.memory, range, storage, inplace);
    auto direct = require_value(fixture.memory->capture_guest_boundary(false));
    inplace.succeed();
    const auto writes = fixture.backend->writes.load();
    CHECK(succeeded(direct->publish()));
    CHECK(fixture.backend->writes == writes);
    CHECK(!inspect(*fixture.memory, range.bytes)[0].gpu_dirty);
}

// A zero-copy authority wrote guest memory itself: resolving only waits for
// the writer and clears the interval, without a readback or a guest write.
// Host ranges are reported only for host-addressable registries.
void test_in_place_authority_and_host_ranges() {
    Fixture fixture;
    fixture.standard_layout();
    CHECK(!fixture.memory->backing_host_range(backing_a));
    TokenSource token(5);
    const auto target = retain(*fixture.memory, 0x1010, 6, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(target.bytes, std::vector<std::byte>(6, std::byte{0x5a}));
    storage->zero_copy = true;
    commit_write(*fixture.memory, target, storage, token);
    token.succeed();
    const unsigned writes = fixture.backend->writes;
    CHECK(succeeded(fixture.memory->resolve(target.bytes, ResolveReason::Capture)));
    CHECK(fixture.backend->writes == writes);
    CHECK(storage->reads == 0);
    CHECK(!covering(inspect(*fixture.memory, target.bytes), backing_a, 0x10)->gpu_dirty);

    auto backend = std::make_shared<Backend>();
    backend->add(backing_a, 0x40);
    ResourceMemoryOptions options{7, backend, CoherenceMode::SubmissionBoundary, 0x10};
    options.host_addressable_backings = true;
    auto host = require_value(make_resource_memory(std::move(options)));
    CHECK(succeeded(host->register_backing({backing_a, 0x40, std::make_shared<ProbeLease>()})));
    CHECK(succeeded(host->register_mapping({mapping_a, backing_a, mapped(0x1000, 0, 0x40, 0x8000)})));
    const auto range = host->backing_host_range(backing_a);
    CHECK(range && range->first == 0x8000 && range->second == 0x40);
}

// A batch publishes in order and stops at the first invalid registration;
// the view still covers the mappings published before it.
void test_batched_mapping_registration() {
    Fixture fixture;
    fixture.register_backing(backing_a);
    fixture.register_backing(backing_b);
    const std::array batch{
        MappingRegistration{mapping_a, backing_a, mapped(0x1000)},
        MappingRegistration{mapping_b, backing_b, mapped(0x1020, 0, 0x40, 0x9000)},
    };
    CHECK(!succeeded(fixture.memory->register_mappings(batch)));
    CHECK(std::holds_alternative<RetainedRange>(fixture.memory->retain(0x1010, 4, ResourceAccess::Read)));
    CHECK(!std::holds_alternative<RetainedRange>(fixture.memory->retain(0x3000, 4, ResourceAccess::Read)));
    const std::array fixed{
        MappingRegistration{mapping_b, backing_b, mapped(0x3000, 0, 0x40, 0x9000)},
        MappingRegistration{mapping_alias, backing_a, mapped(0x2000)},
    };
    CHECK(succeeded(fixture.memory->register_mappings(fixed)));
    CHECK(std::holds_alternative<RetainedRange>(fixture.memory->retain(0x3010, 4, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RetainedRange>(fixture.memory->retain(0x2010, 4, ResourceAccess::Read)));
}

void test_atomic_reservations_validation_and_cancellation() {
    Fixture fixture;
    fixture.standard_layout();
    const auto first = retain(*fixture.memory, 0x1000, 8, ResourceAccess::ReadWrite);
    const auto second = retain(*fixture.memory, 0x3000, 8, ResourceAccess::ReadWrite);
    TokenSource token(7);

    {
        const auto before_first = inspect(*fixture.memory, first.bytes);
        const auto before_second = inspect(*fixture.memory, second.bytes);
        const std::array requests{GpuAccessRequest{first, ResourceAccess::Write},
                                  GpuAccessRequest{second, ResourceAccess::Read}};
        auto cancelled = reserve(*fixture.memory, requests, token.token);
        cancelled.reset();
        const auto after_first = inspect(*fixture.memory, first.bytes);
        const auto after_second = inspect(*fixture.memory, second.bytes);
        CHECK(covering(after_first, backing_a, 0)->version ==
              covering(before_first, backing_a, 0)->version);
        CHECK(covering(after_second, backing_b, 0)->version ==
              covering(before_second, backing_b, 0)->version);
    }

    RetainedRange invalid = second;
    invalid.bytes.offset = 0x3f;
    invalid.bytes.size = 2;
    const std::array invalid_last{GpuAccessRequest{first, ResourceAccess::Write},
                                  GpuAccessRequest{invalid, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(invalid_last, token.token)));
    const std::array first_only{GpuAccessRequest{first, ResourceAccess::Write}};
    auto still_free = reserve(*fixture.memory, first_only, token.token);
    CHECK(!succeeded(still_free->commit({})));
    still_free.reset();
    CHECK(covering(inspect(*fixture.memory, first.bytes), backing_a, 0)->version ==
          ResourceVersion{});

    auto wrong_type = reserve(*fixture.memory, first_only, token.token);
    CHECK(!succeeded(wrong_type->commit({std::make_shared<ProbeLease>()})));
    wrong_type.reset();
    CHECK(covering(inspect(*fixture.memory, first.bytes), backing_a, 0)->version ==
          ResourceVersion{});

    auto wrong_coverage = reserve(*fixture.memory, first_only, token.token);
    auto too_small = std::make_shared<Storage>(
        BackingRange{backing_a, 0, 7}, std::vector<std::byte>(7));
    CHECK(!succeeded(wrong_coverage->commit({too_small})));
    wrong_coverage.reset();
    CHECK(covering(inspect(*fixture.memory, first.bytes), backing_a, 0)->version ==
          ResourceVersion{});

    const auto overlap = retain(*fixture.memory, 0x1004, 8, ResourceAccess::Write);
    const std::array ambiguous{GpuAccessRequest{first, ResourceAccess::Write},
                               GpuAccessRequest{overlap, ResourceAccess::Write}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(ambiguous, token.token)));

    const CompletionToken invalid_token;
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(first_only, invalid_token)));

    const std::array valid_requests{
        GpuAccessRequest{first, ResourceAccess::Write},
        GpuAccessRequest{second, ResourceAccess::Read}};
    auto valid = reserve(*fixture.memory, valid_requests, token.token);
    auto write_storage = std::make_shared<Storage>(
        first.bytes, std::vector<std::byte>(first.bytes.size, std::byte{0x4e}));
    CHECK(succeeded(valid->commit({write_storage, std::make_shared<ProbeLease>()})));
    token.submit();
    token.succeed();
    const auto first_dirty = inspect(*fixture.memory, first.bytes);
    const auto second_clean = inspect(*fixture.memory, second.bytes);
    CHECK(covering(first_dirty, backing_a, 0)->version.gpu_write > 0);
    CHECK(covering(first_dirty, backing_a, 0)->gpu_dirty);
    CHECK(covering(second_clean, backing_b, 0)->version.gpu_write == 0);
    CHECK(!covering(second_clean, backing_b, 0)->gpu_dirty);
}

void test_forged_foreign_and_escalated_leases_are_rejected() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource token(8);
    const auto read_only = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Read);
    const std::array escalated{GpuAccessRequest{read_only, ResourceAccess::Write}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(escalated, token.token)));

    RetainedRange forged = read_only;
    forged.lease = std::make_shared<ProbeLease>();
    const std::array forged_request{GpuAccessRequest{forged, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(forged_request, token.token)));

    Fixture foreign;
    foreign.standard_layout();
    const auto foreign_range = retain(*foreign.memory, 0x1000, 8, ResourceAccess::Read);
    const std::array foreign_request{GpuAccessRequest{foreign_range, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(foreign_request, token.token)));
}

void test_destroyed_registry_leases_stay_foreign() {
    RetainedRange stale;
    {
        Fixture old;
        old.standard_layout();
        stale = retain(*old.memory, 0x1000, 8, ResourceAccess::Read);
    }

    Fixture replacement;
    replacement.standard_layout();
    TokenSource token(9);
    const std::array request{GpuAccessRequest{stale, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        replacement.memory->reserve_accesses(request, token.token)));
    CHECK(!succeeded(replacement.memory->validate_retained(stale, ResourceAccess::Read)));
}

void test_retained_validation_checks_provenance_without_reserving() {
    Fixture fixture;
    fixture.standard_layout();
    constexpr std::array accesses{
        ResourceAccess::Read, ResourceAccess::Write, ResourceAccess::ReadWrite};
    for (const auto retained_access : accesses) {
        const auto range = retain(*fixture.memory, 0x1000, 8, retained_access);
        for (const auto requested_access : accesses) {
            CHECK(succeeded(fixture.memory->validate_retained(range, requested_access)) ==
                  (retained_access == ResourceAccess::ReadWrite ||
                   retained_access == requested_access));
        }
        CHECK(!succeeded(fixture.memory->validate_retained(
            range, static_cast<ResourceAccess>(255))));
    }

    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::ReadWrite);
    Fixture foreign;
    foreign.standard_layout();
    const auto foreign_range = retain(*foreign.memory, 0x1000, 8, ResourceAccess::ReadWrite);
    CHECK(range.mapping == foreign_range.mapping);
    CHECK(range.bytes == foreign_range.bytes);
    CHECK(!succeeded(fixture.memory->validate_retained(foreign_range, ResourceAccess::Read)));
    CHECK(!succeeded(foreign.memory->validate_retained(range, ResourceAccess::Read)));

    auto reject = [&](RetainedRange forged) {
        CHECK(!succeeded(fixture.memory->validate_retained(forged, ResourceAccess::Read)));
    };
    auto forged = range;
    forged.lease.reset();
    reject(forged);
    forged.lease = std::make_shared<ProbeLease>();
    reject(forged);
    forged = range;
    forged.bytes.offset += 1;
    reject(forged);
    forged = range;
    forged.bytes.size -= 1;
    reject(forged);
    forged = range;
    forged.bytes.backing = backing_b;
    reject(forged);
    forged = range;
    forged.mapping = mapping_alias;
    reject(forged);
    forged = range;
    ++forged.mapping.generation;
    reject(forged);

    TokenSource token(1);
    const std::array requests{GpuAccessRequest{range, ResourceAccess::Write}};
    auto reservation = reserve(*fixture.memory, requests, token.token);
    const auto alias = retain(*fixture.memory, 0x2000, 8, ResourceAccess::Read);
    const auto before = inspect(*fixture.memory, range.bytes);
    const auto epoch = fixture.memory->cpu_cache_epoch();
    CHECK(succeeded(fixture.memory->validate_retained(range, ResourceAccess::Write)));
    CHECK(succeeded(fixture.memory->validate_retained(alias, ResourceAccess::Read)));
    CHECK(fixture.memory->cpu_cache_epoch() == epoch);
    CHECK(inspect(*fixture.memory, range.bytes).front().version == before.front().version);
    CHECK(fixture.backend->reads == 0);
    CHECK(fixture.backend->writes == 0);
    CHECK(fixture.backend->waits == 0);
    auto storage = std::make_shared<Storage>(
        range.bytes, std::vector<std::byte>(8, std::byte{0x55}));
    CHECK(succeeded(reservation->commit({storage})));
    CHECK(succeeded(fixture.memory->validate_retained(range, ResourceAccess::ReadWrite)));
    token.submit();
    token.succeed();
}

void test_gpu_readers_block_cpu_writes_until_feedback() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource reader(9);
    const auto range = retain(*fixture.memory, 0x2008, 8, ResourceAccess::Read);
    const std::array requests{GpuAccessRequest{range, ResourceAccess::Read}};
    auto reservation = reserve(*fixture.memory, requests, reader.token);
    CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
    reader.submit();
    reservation.reset();
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->prepare_cpu_write({backing_a, 10, 2})));
    reader.succeed();
    auto writer = require_value(
        fixture.memory->prepare_cpu_write({backing_a, 10, 2}));
    writer->commit();
}

void test_dependencies_and_boundary_epoch() {
    Fixture fixture;
    fixture.standard_layout();
    const auto range = retain(*fixture.memory, 0x1000, 8, ResourceAccess::ReadWrite);
    TokenSource producer(10);
    auto storage = std::make_shared<Storage>(
        range.bytes, std::vector<std::byte>(8, std::byte{0x7b}));
    commit_write(*fixture.memory, range, storage, producer);

    TokenSource consumer(11);
    const std::array reads{GpuAccessRequest{range, ResourceAccess::Read}};
    auto dependency = reserve(*fixture.memory, reads, consumer.token);
    const auto dependencies = dependency->dependencies();
    CHECK(dependencies.size() == 1);
    if (dependencies.size() == 1) {
        CHECK(dependencies[0].snapshot().id == producer.token.snapshot().id);
    }
    dependency.reset();

    producer.succeed();
    const auto epoch = fixture.memory->cpu_cache_epoch();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() > epoch);
    const auto boundary_state = inspect(*fixture.memory, range.bytes);
    CHECK(!covering(boundary_state, backing_a, 0)->gpu_dirty);
    CHECK(covering(boundary_state, backing_a, 0)->cpu_dirty);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(std::all_of(bytes.begin(), bytes.begin() + 8,
                          [](std::byte byte) { return byte == std::byte{0x7b}; }));
    }

    const auto next_epoch = fixture.memory->cpu_cache_epoch();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() > next_epoch);

    TokenSource pending(12);
    const auto pending_range = retain(
        *fixture.memory, 0x1010, 4, ResourceAccess::Write);
    auto pending_storage = std::make_shared<Storage>(
        pending_range.bytes, std::vector<std::byte>(4, std::byte{0x2c}));
    commit_write(*fixture.memory, pending_range, pending_storage, pending);
    fixture.backend->fail_waits = true;
    const auto failed_epoch = fixture.memory->cpu_cache_epoch();
    CHECK(!succeeded(fixture.memory->submission_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() == failed_epoch);
    CHECK(covering(inspect(*fixture.memory, pending_range.bytes), backing_a, 0x10)
              ->gpu_dirty);
    fixture.backend->fail_waits = false;
    pending.succeed();
    CHECK(succeeded(fixture.memory->submission_boundary()));
}

void test_failed_gpu_reader_survives_pruning_until_boundary() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource reader(13);
    const auto read_range = retain(*fixture.memory, 0x1000, 4, ResourceAccess::Read);
    const std::array request{GpuAccessRequest{read_range, ResourceAccess::Read}};
    auto reservation = reserve(*fixture.memory, request, reader.token);
    CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
    reader.submit();
    reader.fail();

    CHECK(succeeded(fixture.memory->unmap(mapping_b)));
    const auto epoch = fixture.memory->cpu_cache_epoch();
    CHECK(!succeeded(fixture.memory->submission_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() == epoch);
}

void test_queue_order_applies_to_disjoint_pending_reservations() {
    Fixture fixture;
    fixture.standard_layout();
    const auto low_range = retain(*fixture.memory, 0x1000, 4, ResourceAccess::Read);
    const auto high_range = retain(*fixture.memory, 0x3010, 4, ResourceAccess::Read);
    TokenSource low(20);
    TokenSource high(21);
    const std::array low_request{GpuAccessRequest{low_range, ResourceAccess::Read}};
    const std::array high_request{GpuAccessRequest{high_range, ResourceAccess::Read}};
    auto low_reservation = reserve(*fixture.memory, low_request, low.token);
    auto high_reservation = reserve(*fixture.memory, high_request, high.token);
    auto low_storage = std::make_shared<ProbeLease>();
    auto high_storage = std::make_shared<ProbeLease>();
    CHECK(!succeeded(high_reservation->commit({high_storage})));
    CHECK(succeeded(low_reservation->commit({low_storage})));
    CHECK(succeeded(high_reservation->commit({high_storage})));
    low.submit();
    high.submit();
    low.succeed();
    high.succeed();
}

void test_accepted_gpu_accesses_never_enter_backend_wait() {
    for (const auto access : {ResourceAccess::Write, ResourceAccess::Read}) {
        Fixture fixture;
        fixture.standard_layout();
        TokenSource token(1);
        const auto range = retain(*fixture.memory, 0x1008, 8, access);
        const std::array requests{GpuAccessRequest{range, access}};
        auto reservation = reserve(*fixture.memory, requests, token.token);
        auto storage = std::make_shared<Storage>(
            range.bytes, std::vector<std::byte>(8, std::byte{0x7a}));
        CHECK(succeeded(reservation->commit({storage})));
        reservation.reset();
        CHECK(token.token.snapshot().phase == SubmissionPhase::Accepted);

        if (access == ResourceAccess::Write) {
            auto read = fixture.memory->prepare_cpu_read(range.bytes, ResolveReason::CpuRead);
            CHECK(std::holds_alternative<RendererError>(read));
            if (const auto* failure = std::get_if<RendererError>(&read)) {
                CHECK(failure->command().status == CommandStatus::Stopped);
                CHECK(failure->command().detail ==
                      "GPU access is not submitted; submit before retrying the wait");
            }
            CHECK(!succeeded(fixture.memory->resolve(range.bytes, ResolveReason::Capture)));
            CHECK(covering(inspect(*fixture.memory, range.bytes), backing_a, 8)->gpu_dirty);
        }
        CHECK(std::holds_alternative<RendererError>(
            fixture.memory->prepare_cpu_write(range.bytes)));
        const auto epoch = fixture.memory->cpu_cache_epoch();
        CHECK(!succeeded(fixture.memory->submission_boundary()));
        CHECK(fixture.memory->cpu_cache_epoch() == epoch);
        CHECK(fixture.backend->waits == 0);
        CHECK(fixture.backend->writes == 0);

        token.submit();
        token.succeed();
        auto read = require_value(fixture.memory->prepare_cpu_read(
            range.bytes, ResolveReason::CpuRead));
        read.reset();
        auto write = require_value(fixture.memory->prepare_cpu_write(range.bytes));
        write.reset();
        CHECK(succeeded(fixture.memory->submission_boundary()));
        if (access == ResourceAccess::Write) {
            CHECK(fixture.backend->find(backing_a)->bytes[8] == std::byte{0x7a});
        }
    }
}

void test_cancelled_writes_and_whole_overwrite_coalesce_segments() {
    Fixture fixture;
    constexpr std::uint64_t size = 0x1000;
    fixture.backend->find(backing_a)->bytes.resize(size);
    fixture.register_backing(backing_a, size);
    auto mapping = mapped(0x1000, 0, size);
    mapping.backing_size = size;
    fixture.register_mapping(mapping_a, backing_a, mapping);
    const BackingRange whole{backing_a, 0, size};
    const auto baseline = inspect(*fixture.memory, whole).size();
    CHECK(baseline == 1);
    for (unsigned pass = 0; pass < 2; ++pass) {
        for (std::uint64_t offset = 1; offset < size; offset += 2) {
            auto write = require_value(fixture.memory->prepare_cpu_write({backing_a, offset, 1}));
            CHECK(inspect(*fixture.memory, whole).size() <= baseline + 2);
            write.reset();
            const auto state = inspect(*fixture.memory, whole);
            CHECK(state.size() == baseline);
            CHECK(state.front().version == ResourceVersion{});
        }
    }
    for (std::uint64_t offset = 1; offset < 0x40; offset += 2) {
        auto write = require_value(fixture.memory->prepare_cpu_write({backing_a, offset, 1}));
        const std::array byte{std::byte{0x35}};
        CHECK(succeeded(fixture.backend->write_cpu({backing_a, offset, 1}, byte)));
        write->commit();
    }
    CHECK(inspect(*fixture.memory, whole).size() > baseline);
    auto overwrite = require_value(fixture.memory->prepare_cpu_write(whole));
    const std::vector bytes(size, std::byte{0x6b});
    CHECK(succeeded(fixture.backend->write_cpu(whole, bytes)));
    overwrite->commit();
    CHECK(inspect(*fixture.memory, whole).size() == baseline);
    overwrite.reset();
    CHECK(inspect(*fixture.memory, whole).size() == baseline);
}

void test_coalescing_preserves_other_prepared_write_boundaries() {
    Fixture fixture;
    fixture.standard_layout();
    auto held = require_value(fixture.memory->prepare_cpu_write({backing_a, 19, 1}));
    auto cancelled = require_value(fixture.memory->prepare_cpu_write({backing_a, 3, 1}));
    cancelled.reset();
    auto read = require_value(fixture.memory->prepare_cpu_read(
        {backing_a, 32, 1}, ResolveReason::CpuRead));
    read.reset();
    TokenSource reader(1);
    auto range = retain(*fixture.memory, 0x1030, 1, ResourceAccess::Read);
    const std::array request{GpuAccessRequest{range, ResourceAccess::Read}};
    auto reservation = reserve(*fixture.memory, request, reader.token);
    CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
    reservation.reset();
    reader.submit();
    reader.succeed();
    const std::array byte{std::byte{0x42}};
    CHECK(succeeded(fixture.backend->write_cpu({backing_a, 19, 1}, byte)));
    held->commit();
    const auto state = inspect(*fixture.memory, {backing_a, 0, 0x40});
    for (std::uint64_t offset = 0; offset < 0x40; ++offset) {
        CHECK((covering(state, backing_a, offset)->version.cpu_write != 0) == (offset == 19));
    }
}

void test_resolves_and_boundary_coalesce_redundant_segments() {
    Fixture fixture;
    fixture.standard_layout();
    const BackingRange whole{backing_a, 0, 0x40};
    const auto baseline = inspect(*fixture.memory, whole).size();
    TokenSource reader(1);
    auto range = retain(*fixture.memory, 0x1010, 0x10, ResourceAccess::Read);
    const std::array request{GpuAccessRequest{range, ResourceAccess::Read}};
    auto reservation = reserve(*fixture.memory, request, reader.token);
    CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
    // A read publishes only its intent and splits no interval.
    CHECK(inspect(*fixture.memory, whole).size() == baseline);
    reader.submit();
    reader.succeed();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    CHECK(inspect(*fixture.memory, whole).size() == baseline);

    TokenSource writer(2);
    auto target = retain(*fixture.memory, 0x1000, 0x40, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(
        target.bytes, std::vector<std::byte>(0x40, std::byte{0x71}));
    commit_write(*fixture.memory, target, storage, writer);
    writer.succeed();
    storage->fail_reads = true;
    for (std::uint64_t offset = 1; offset < whole.size; offset += 2) {
        CHECK(!succeeded(fixture.memory->resolve({backing_a, offset, 1}, ResolveReason::CpuRead)));
        CHECK(inspect(*fixture.memory, whole).size() == baseline);
    }
    storage->fail_reads = false;
    for (std::uint64_t offset = 0; offset < whole.size; ++offset) {
        CHECK(succeeded(fixture.memory->resolve({backing_a, offset, 1}, ResolveReason::CpuRead)));
        CHECK(inspect(*fixture.memory, whole).size() <= baseline + 1);
    }
    CHECK(inspect(*fixture.memory, whole).size() == baseline);
    CHECK(!inspect(*fixture.memory, whole).front().gpu_dirty);
}

void test_unaligned_guest_base_preserves_absolute_granules() {
    struct Case { std::uint64_t offset, size, preserved_offset, preserved_size; };
    constexpr std::array cases{Case{8, 1, 8, 16}, Case{0, 1, 0, 8},
                               Case{63, 1, 56, 8}, Case{7, 2, 0, 24}};
    for (const auto base : {std::uint64_t{0x8008},
                            std::numeric_limits<std::uint64_t>::max() - 0x47}) {
        for (const auto test : cases) {
            Fixture fixture;
            fixture.register_backing(backing_a);
            fixture.register_mapping(mapping_a, backing_a, mapped(0x1000, 0, 0x40, base));
            fixture.register_mapping(mapping_alias, backing_a, mapped(0x2000, 0, 0x40, base));
            TokenSource writer(1);
            auto whole = retain(*fixture.memory, 0x1000, 0x40, ResourceAccess::Write);
            auto storage = std::make_shared<Storage>(
                whole.bytes, std::vector<std::byte>(0x40, std::byte{0x7c}));
            commit_write(*fixture.memory, whole, storage, writer);
            writer.succeed();
            auto write = require_value(fixture.memory->prepare_cpu_write(
                {backing_a, test.offset, test.size}));
            const auto state = inspect(*fixture.memory, whole.bytes);
            TokenSource consumer(2);
            for (std::uint64_t offset = 0; offset < 0x40; ++offset) {
                const bool preserved = offset >= test.preserved_offset &&
                    offset - test.preserved_offset < test.preserved_size;
                CHECK(covering(state, backing_a, offset)->gpu_dirty == !preserved);
                CHECK(fixture.backend->find(backing_a)->bytes[offset] ==
                      (preserved ? std::byte{0x7c} : std::byte{0}));
                auto alias = retain(*fixture.memory, 0x2000 + offset, 1, ResourceAccess::Write);
                const std::array request{GpuAccessRequest{alias, ResourceAccess::Write}};
                auto reservation = fixture.memory->reserve_accesses(request, consumer.token);
                CHECK(std::holds_alternative<RendererError>(reservation) == preserved);
            }
        }
    }
    Fixture unmapped;
    unmapped.register_backing(backing_a);
    CHECK(std::holds_alternative<RendererError>(
        unmapped.memory->prepare_cpu_write({backing_a, 8, 1})));
}

void test_address_space_aliases_and_failed_store_accounting() {
    Fixture fixture;
    fixture.standard_layout();
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        fixture.backend->find(backing_a)->bytes[5] = std::byte{0x5d};
    }
    std::array<std::byte, 1> first{};
    std::array<std::byte, 1> alias{};
    CHECK(fixture.memory->address_space().read(0x1005, first));
    CHECK(fixture.memory->address_space().read(0x2005, alias));
    CHECK(first[0] == std::byte{0x5d});
    CHECK(alias == first);

    fixture.backend->fail_writes = true;
    fixture.backend->partial_write_bytes = 2;
    const std::array source{std::byte{0xa1}, std::byte{0xa2},
                            std::byte{0xa3}, std::byte{0xa4}};
    CHECK(fixture.memory->address_space().write(0x2008, source).code ==
          MemoryErrorCode::WriteFailed);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(bytes[8] == std::byte{0xa1});
        CHECK(bytes[9] == std::byte{0xa2});
        CHECK(bytes[10] == std::byte{0});
        CHECK(bytes[11] == std::byte{0});
    }
    const auto dirty = inspect(*fixture.memory, {backing_a, 8, 4});
    const auto* changed = covering(dirty, backing_a, 8);
    CHECK(changed != nullptr);
    if (changed) {
        CHECK(changed->version.cpu_write > 0);
        CHECK(changed->cpu_dirty);
        CHECK(!changed->gpu_dirty);
    }
}

void test_bounds_overflow_and_mapping_validation() {
    Fixture fixture;
    fixture.register_backing(backing_a);
    CHECK(!succeeded(fixture.memory->register_mapping(
        {{8, 1, 1}, backing_a, mapped(0x1000)})));
    CHECK(!succeeded(fixture.memory->register_mapping(
        {mapping_a, backing_a, mapped(0x1001)})));
    CHECK(!succeeded(fixture.memory->register_mapping(
        {mapping_a, backing_a, mapped(0x1000, 1, 0x20)})));
    const MappingId padded_id{7, 9, 1};
    CHECK(succeeded(fixture.memory->register_mapping(
        {padded_id, backing_a, mapped(0x2000, 0x30, 0x20)})));
    CHECK(std::holds_alternative<RetainedRange>(
        fixture.memory->retain(0x200f, 1, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x2010, 1, ResourceAccess::Read)));
    CHECK(succeeded(fixture.memory->unmap(padded_id)));
    fixture.register_mapping(mapping_a, backing_a, mapped(0x1000));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x1000, 0, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x103f, 2, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->retain(
        std::numeric_limits<std::uint64_t>::max(), 2, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->inspect({backing_a, 0x40, 1})));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->inspect(
        {backing_a, std::numeric_limits<std::uint64_t>::max(), 2})));
    CHECK(!succeeded(fixture.memory->resolve(
        {backing_a, 0, 0}, ResolveReason::CpuRead)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->prepare_cpu_write({backing_a, 0x3f, 2})));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->prepare_cpu_read(
        {backing_b, 0, 1}, ResolveReason::CpuRead)));

    const std::array invalid_inspections{
        BackingRange{backing_a, 0, 0},
        BackingRange{backing_a, 0x40, 1},
        BackingRange{backing_a, 0x3f, 2},
        BackingRange{backing_a, std::numeric_limits<std::uint64_t>::max(), 2},
        BackingRange{backing_b, 0, 1},
        BackingRange{{1, 1, 2}, 0, 1},
        BackingRange{{}, 0, 1}};
    for (const auto range : invalid_inspections) {
        const auto failure = fixture.memory->inspect_storage(range);
        CHECK(std::holds_alternative<RendererError>(failure));
        if (const auto* value = std::get_if<RendererError>(&failure)) {
            CHECK(value->command().status == CommandStatus::Stopped);
        }
    }
    fixture.register_backing(backing_b);
    CHECK(succeeded(fixture.memory->retire_backing(backing_b)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->inspect_storage({backing_b, 0, 1})));
}

void test_feedback_and_polling_while_backend_waits() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource token(12);
    const auto target = retain(*fixture.memory, 0x1030, 4, ResourceAccess::Write);
    auto storage = std::make_shared<Storage>(
        target.bytes, std::vector<std::byte>(4, std::byte{0xcc}));
    commit_write(*fixture.memory, target, storage, token);
    fixture.backend->block_wait = true;

    std::atomic<bool> resolved = false;
    std::atomic<bool> resolve_succeeded = false;
    std::thread coordinator([&] {
        resolve_succeeded = succeeded(fixture.memory->resolve(
            target.bytes, ResolveReason::SubmissionBoundary));
        resolved = true;
    });
    fixture.backend->await_wait_entry();
    CHECK(!resolved);

    std::atomic<bool> stop_polling = false;
    std::atomic<bool> poll_valid = true;
    std::thread poller([&] {
        while (!stop_polling.load()) {
            const auto snapshot = token.token.snapshot();
            if (snapshot.id != SubmissionId{1, 12}) {
                poll_valid = false;
            }
        }
    });
    token.succeed();
    fixture.backend->release_wait();
    coordinator.join();
    stop_polling = true;
    poller.join();
    CHECK(poll_valid);
    CHECK(resolve_succeeded);
    CHECK(resolved);
    CHECK(token.token.snapshot().phase == SubmissionPhase::GpuComplete);
}

void test_mapping_invalidation_stales_leases_and_preserves_alias_state() {
    Fixture fixture;
    fixture.standard_layout();
    const auto old = retain(*fixture.memory, 0x1010, 8, ResourceAccess::ReadWrite);
    const auto alias = retain(*fixture.memory, 0x2010, 8, ResourceAccess::ReadWrite);
    TokenSource producer(1);
    auto storage = std::make_shared<Storage>(
        old.bytes, std::vector<std::byte>(8, std::byte{0x5a}));
    commit_write(*fixture.memory, old, storage, producer);
    producer.succeed();
    const auto dirty = inspect(*fixture.memory, old.bytes);
    const auto version = covering(dirty, backing_a, 0x10)->version;
    CHECK(covering(dirty, backing_a, 0x10)->gpu_dirty);

    const std::array removed{mapping_a};
    const auto writes = fixture.backend->writes.load();
    CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    CHECK(fixture.backend->writes == writes + 1);
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x1010, 8, ResourceAccess::Read)));
    CHECK(!succeeded(fixture.memory->validate_retained(old, ResourceAccess::Read)));
    CHECK(succeeded(fixture.memory->validate_retained(alias, ResourceAccess::ReadWrite)));
    TokenSource stale_consumer(2);
    const std::array stale_request{GpuAccessRequest{old, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(stale_request, stale_consumer.token)));

    const auto resolved = inspect(*fixture.memory, old.bytes);
    const auto* current = covering(resolved, backing_a, 0x10);
    CHECK(current != nullptr);
    if (current) {
        CHECK(current->version == version);
        CHECK(!current->gpu_dirty);
    }
    CHECK(fixture.backend->find(backing_a)->bytes[0x10] == std::byte{0x5a});

    const MappingId remapped{7, 1, 2};
    fixture.register_mapping(remapped, backing_a, mapped(0x1000));
    const auto replacement = retain(
        *fixture.memory, 0x1010, 8, ResourceAccess::ReadWrite);
    CHECK(replacement.mapping == remapped);
    CHECK(replacement.bytes == old.bytes);
    CHECK(succeeded(fixture.memory->validate_retained(replacement,
                                                       ResourceAccess::ReadWrite)));
    CHECK(succeeded(fixture.memory->validate_retained(alias,
                                                       ResourceAccess::ReadWrite)));
    CHECK(covering(inspect(*fixture.memory, replacement.bytes), backing_a, 0x10)
              ->version == version);
}

void test_mapping_invalidation_rejects_pending_and_accepted_accesses() {
    const std::array removed{mapping_a};
    for (const bool write : {false, true}) {
        Fixture fixture;
        fixture.standard_layout();
        if (write) {
            auto pending = require_value(
                fixture.memory->prepare_cpu_write({backing_a, 8, 8}));
            CHECK(!succeeded(fixture.memory->invalidate_mappings(removed)));
            CHECK(succeeded(fixture.memory->validate_retained(
                retain(*fixture.memory, 0x1008, 8, ResourceAccess::Read),
                ResourceAccess::Read)));
            pending.reset();
        } else {
            auto pending = require_value(fixture.memory->prepare_cpu_read(
                {backing_a, 8, 8}, ResolveReason::CpuRead));
            CHECK(!succeeded(fixture.memory->invalidate_mappings(removed)));
            pending.reset();
        }
        CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    }

    {
        Fixture fixture;
        fixture.standard_layout();
        TokenSource token(1);
        const auto alias = retain(*fixture.memory, 0x2008, 8, ResourceAccess::Read);
        const std::array requests{GpuAccessRequest{alias, ResourceAccess::Read}};
        auto pending = reserve(*fixture.memory, requests, token.token);
        CHECK(!succeeded(fixture.memory->invalidate_mappings(removed)));
        CHECK(succeeded(fixture.memory->validate_retained(alias, ResourceAccess::Read)));
        pending.reset();
        CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    }

    for (const auto access : {ResourceAccess::Read, ResourceAccess::Write}) {
        Fixture fixture;
        fixture.standard_layout();
        TokenSource token(1);
        const auto alias = retain(*fixture.memory, 0x2008, 8, ResourceAccess::ReadWrite);
        const std::array requests{GpuAccessRequest{alias, access}};
        auto reservation = reserve(*fixture.memory, requests, token.token);
        RetainedResource storage = std::make_shared<ProbeLease>();
        if (access == ResourceAccess::Write) {
            storage = std::make_shared<Storage>(
                alias.bytes, std::vector<std::byte>(8, std::byte{0x6a}));
        }
        CHECK(succeeded(reservation->commit({storage})));
        reservation.reset();
        CHECK(!succeeded(fixture.memory->invalidate_mappings(removed)));
        CHECK(fixture.backend->waits == 0);
        CHECK(succeeded(fixture.memory->validate_retained(alias, ResourceAccess::Read)));
        token.submit();
        token.succeed();
        CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    }
}

void test_mapping_invalidation_waits_for_submitted_accesses() {
    Fixture fixture;
    fixture.standard_layout();
    const auto target = retain(*fixture.memory, 0x1018, 8, ResourceAccess::Write);
    TokenSource token(1);
    auto storage = std::make_shared<Storage>(
        target.bytes, std::vector<std::byte>(8, std::byte{0x7b}));
    commit_write(*fixture.memory, target, storage, token);
    fixture.backend->block_wait = true;
    const std::array removed{mapping_a};
    RendererStatus invalidated = synthetic_error("invalidation did not finish");
    std::thread worker([&] {
        invalidated = fixture.memory->invalidate_mappings(removed);
    });
    fixture.backend->await_wait_entry();
    CHECK(fixture.backend->waits == 1);
    token.succeed();
    fixture.backend->release_wait();
    worker.join();
    CHECK(succeeded(invalidated));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x1018, 8, ResourceAccess::Read)));
    CHECK(fixture.backend->find(backing_a)->bytes[0x18] == std::byte{0x7b});
}

void test_failed_invalidation_writeback_is_atomic_and_retryable() {
    Fixture fixture;
    fixture.standard_layout();
    const auto target = retain(*fixture.memory, 0x1010, 8, ResourceAccess::Write);
    TokenSource token(1);
    auto storage = std::make_shared<Storage>(
        target.bytes, std::vector<std::byte>(8, std::byte{0x8c}));
    commit_write(*fixture.memory, target, storage, token);
    token.succeed();
    const std::array removed{mapping_a};
    fixture.backend->fail_writes = true;
    fixture.backend->partial_write_bytes = 3;
    CHECK(!succeeded(fixture.memory->invalidate_mappings(removed)));
    CHECK(succeeded(fixture.memory->validate_retained(target, ResourceAccess::Write)));
    CHECK(require_value(fixture.memory->retain(
              0x1010, 8, ResourceAccess::Read)).mapping == mapping_a);
    const auto failed = require_value(fixture.memory->inspect_storage(target.bytes));
    CHECK(failed.size() == 1);
    if (failed.size() == 1) {
        CHECK(failed.front().dirty.gpu_dirty);
        CHECK(failed.front().storage == storage);
    }

    fixture.backend->fail_writes = false;
    CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x1010, 8, ResourceAccess::Read)));
    CHECK(succeeded(fixture.memory->validate_retained(
        retain(*fixture.memory, 0x2010, 8, ResourceAccess::Read),
        ResourceAccess::Read)));
    const auto resolved = inspect(*fixture.memory, target.bytes);
    CHECK(!covering(resolved, backing_a, 0x10)->gpu_dirty);
    CHECK(fixture.backend->find(backing_a)->bytes[0x10] == std::byte{0x8c});
}

void test_multi_backing_invalidation_preserves_lookup_on_late_failure() {
    Fixture fixture;
    fixture.standard_layout();
    const auto first = retain(*fixture.memory, 0x1010, 8, ResourceAccess::ReadWrite);
    const auto second = retain(*fixture.memory, 0x3010, 8, ResourceAccess::ReadWrite);
    TokenSource producer(1);
    auto first_storage = std::make_shared<Storage>(first.bytes,
        std::vector<std::byte>(8, std::byte{0x21}));
    auto second_storage = std::make_shared<Storage>(second.bytes,
        std::vector<std::byte>(8, std::byte{0x43}));
    const std::array requests{GpuAccessRequest{first, ResourceAccess::Write},
                              GpuAccessRequest{second, ResourceAccess::Write}};
    auto reservation = reserve(*fixture.memory, requests, producer.token);
    CHECK(succeeded(reservation->commit({first_storage, second_storage})));
    reservation.reset();
    producer.submit();
    producer.succeed();
    fixture.backend->fail_write_backing = backing_b;
    const std::array removed{mapping_a, mapping_b};
    CHECK(!succeeded(fixture.memory->invalidate_mappings(removed)));
    CHECK(succeeded(fixture.memory->validate_retained(first, ResourceAccess::Read)));
    CHECK(succeeded(fixture.memory->validate_retained(second, ResourceAccess::Read)));
    const auto resolved = require_value(fixture.memory->inspect_storage(first.bytes));
    const auto failed = require_value(fixture.memory->inspect_storage(second.bytes));
    CHECK(resolved.size() == 1 && !resolved.front().dirty.gpu_dirty && !resolved.front().storage);
    CHECK(failed.size() == 1 && failed.front().dirty.gpu_dirty && failed.front().storage == second_storage);
    CHECK(fixture.backend->find(backing_a)->bytes[0x10] == std::byte{0x21});
    CHECK(fixture.backend->find(backing_b)->bytes[0x10] == std::byte{0});
    const auto writes = fixture.backend->writes.load();
    fixture.backend->fail_write_backing = {};
    CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    CHECK(fixture.backend->writes == writes + 1);
    CHECK(fixture.backend->find(backing_b)->bytes[0x10] == std::byte{0x43});
    CHECK(!succeeded(fixture.memory->validate_retained(first, ResourceAccess::Read)));
    CHECK(!succeeded(fixture.memory->validate_retained(second, ResourceAccess::Read)));
    CHECK(require_value(fixture.memory->retain(0x2010, 8, ResourceAccess::Read)).bytes == first.bytes);
}

// A GPU write stays GPU-authoritative across internal boundaries; guest memory
// only receives the bytes at the guest-visible submission boundary.
void test_internal_boundary_preserves_gpu_authority_until_submission() {
    Fixture fixture;
    fixture.standard_layout();
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0; index != bytes.size(); ++index) {
            bytes[index] = std::byte(0x40 + index);
        }
    }

    TokenSource writer(40);
    const auto target = retain(*fixture.memory, 0x1008, 8, ResourceAccess::Write);
    std::vector<std::byte> gpu_bytes(8);
    for (std::size_t index = 0; index != gpu_bytes.size(); ++index) {
        gpu_bytes[index] = std::byte(0xa0 + index);
    }
    auto storage = std::make_shared<Storage>(target.bytes, gpu_bytes);
    commit_write(*fixture.memory, target, storage, writer);
    writer.succeed();

    const auto epoch = fixture.memory->cpu_cache_epoch();
    CHECK(succeeded(fixture.memory->internal_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() == epoch + 1);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 8; index != 16; ++index) {
            CHECK(bytes[index] == std::byte(0x40 + index));
        }
    }
    const auto deferred = inspect(*fixture.memory, target.bytes);
    const auto* interval = covering(deferred, backing_a, 8);
    CHECK(interval != nullptr);
    if (interval) {
        CHECK(interval->gpu_dirty);
        CHECK(!interval->cpu_dirty);
        CHECK(interval->version.gpu_write > 0);
    }
    const auto authority = require_value(fixture.memory->inspect_storage(target.bytes));
    CHECK(authority.size() == 1);
    if (authority.size() == 1) {
        CHECK(authority.front().dirty.bytes == target.bytes);
        CHECK(authority.front().dirty.gpu_dirty);
        CHECK(authority.front().storage == storage);
    }

    // Repeated unobservable boundaries change nothing about GPU authority.
    CHECK(succeeded(fixture.memory->internal_boundary()));
    CHECK(succeeded(fixture.memory->internal_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() == epoch + 3);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(bytes[8] == std::byte{0x48});
        CHECK(bytes[15] == std::byte{0x4f});
    }
    CHECK(require_value(fixture.memory->inspect_storage(target.bytes)).front().storage ==
          storage);

    CHECK(succeeded(fixture.memory->submission_boundary()));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 8; index != 16; ++index) {
            CHECK(bytes[index] == std::byte(0xa0 + index - 8));
        }
    }
    const auto resolved = inspect(*fixture.memory, target.bytes);
    const auto* published = covering(resolved, backing_a, 8);
    CHECK(published != nullptr);
    if (published) {
        CHECK(!published->gpu_dirty);
        CHECK(published->cpu_dirty);
    }
    const auto clean = require_value(fixture.memory->inspect_storage(target.bytes));
    CHECK(std::all_of(clean.begin(), clean.end(), [](const auto& entry) {
        return !entry.dirty.gpu_dirty && !entry.storage;
    }));
}

// The boundary epoch invalidates CPU-backed segments, but a segment still owned
// by the GPU is never reported CPU-modified just because the epoch passed it.
void test_internal_boundaries_keep_gpu_dirty_segments_cpu_clean() {
    Fixture fixture;
    fixture.standard_layout();
    TokenSource writer(41);
    const auto gpu_target = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Write);
    auto gpu_storage = std::make_shared<Storage>(
        gpu_target.bytes, std::vector<std::byte>(8, std::byte{0xb7}));
    commit_write(*fixture.memory, gpu_target, gpu_storage, writer);
    writer.succeed();

    auto cpu_write = require_value(fixture.memory->prepare_cpu_write({backing_a, 0x10, 8}));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0x10; index != 0x18; ++index) {
            bytes[index] = std::byte{0x5e};
        }
    }
    cpu_write->commit();
    cpu_write.reset();

    for (unsigned boundary = 1; boundary != 4; ++boundary) {
        const auto epoch = fixture.memory->cpu_cache_epoch();
        CHECK(succeeded(fixture.memory->internal_boundary()));
        CHECK(fixture.memory->cpu_cache_epoch() == epoch + 1);
    }

    const auto state = inspect(*fixture.memory, {backing_a, 0, 0x40});
    const auto* gpu_segment = covering(state, backing_a, 0);
    CHECK(gpu_segment != nullptr);
    if (gpu_segment) {
        CHECK(gpu_segment->gpu_dirty);
        CHECK(!gpu_segment->cpu_dirty);
    }
    const auto* cpu_segment = covering(state, backing_a, 0x10);
    CHECK(cpu_segment != nullptr);
    if (cpu_segment) {
        CHECK(!cpu_segment->gpu_dirty);
        CHECK(cpu_segment->cpu_dirty);
    }
    // Segments the CPU never wrote are stale by epoch alone.
    const auto* untouched = covering(state, backing_a, 0x20);
    CHECK(untouched != nullptr);
    if (untouched) {
        CHECK(!untouched->gpu_dirty);
        CHECK(untouched->cpu_dirty);
    }
    CHECK(require_value(fixture.memory->inspect_storage(gpu_target.bytes))
              .front().storage == gpu_storage);

    // The guest-visible boundary still resolves the deferred GPU bytes.
    CHECK(succeeded(fixture.memory->submission_boundary()));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(std::all_of(bytes.begin(), bytes.begin() + 8,
                          [](std::byte byte) { return byte == std::byte{0xb7}; }));
        CHECK(std::all_of(bytes.begin() + 0x10, bytes.begin() + 0x18,
                          [](std::byte byte) { return byte == std::byte{0x5e}; }));
    }
}

// A coordinator callback that enqueues work must observe guest CPU bytes as of
// its entry even while earlier batched submissions are pending. The epoch-only
// advance succeeds where a boundary waits or refuses: uncommitted (Accepted)
// and in-flight (Submitted) GPU work is neither rejected nor waited on, the
// GPU keeps its authority, and clean CPU-backed segments become stale.
void test_cpu_epoch_advance_is_valid_with_pending_gpu_work() {
    Fixture fixture;
    fixture.standard_layout();
    // An uncommitted batch draw: its reservation committed, token Accepted.
    TokenSource batched(30);
    const auto gpu_target = retain(*fixture.memory, 0x1008, 8, ResourceAccess::Write);
    auto gpu_storage = std::make_shared<Storage>(
        gpu_target.bytes, std::vector<std::byte>(8, std::byte{0x9d}));
    const std::array writes{GpuAccessRequest{gpu_target, ResourceAccess::Write}};
    auto batched_reservation = reserve(*fixture.memory, writes, batched.token);
    CHECK(succeeded(batched_reservation->commit({gpu_storage})));
    batched_reservation.reset();
    CHECK(batched.token.snapshot().phase == SubmissionPhase::Accepted);
    // An earlier submitted submission still holds a read of other bytes.
    TokenSource reader(31);
    const auto read_range = retain(*fixture.memory, 0x1010, 4, ResourceAccess::Read);
    const std::array reads{GpuAccessRequest{read_range, ResourceAccess::Read}};
    auto reservation = reserve(*fixture.memory, reads, reader.token);
    CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
    reservation.reset();
    reader.submit();
    reader.succeed();

    // Neither boundary variant may run under an uncommitted reservation.
    const auto epoch = fixture.memory->cpu_cache_epoch();
    CHECK(!succeeded(fixture.memory->submission_boundary()));
    CHECK(!succeeded(fixture.memory->internal_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() == epoch);
    CHECK(fixture.backend->waits == 0);

    // The epoch-only advance succeeds, waits for nothing and writes nothing.
    CHECK(succeeded(fixture.memory->advance_cpu_epoch()));
    CHECK(fixture.memory->cpu_cache_epoch() == epoch + 1);
    CHECK(fixture.backend->waits == 0);
    CHECK(fixture.backend->writes == 0);

    // The pending GPU write keeps its authority and its storage untouched.
    const auto authority = require_value(fixture.memory->inspect_storage(gpu_target.bytes));
    CHECK(authority.size() == 1);
    if (authority.size() == 1) {
        CHECK(authority.front().dirty.bytes == gpu_target.bytes);
        CHECK(authority.front().dirty.gpu_dirty);
        CHECK(!authority.front().dirty.cpu_dirty);
        CHECK(authority.front().storage == gpu_storage);
    }
    // Clean CPU-backed bytes are stale by the advanced epoch alone.
    const auto stale = inspect(*fixture.memory, read_range.bytes);
    const auto* segment = covering(stale, backing_a, 0x10);
    CHECK(segment != nullptr);
    if (segment) {
        CHECK(!segment->gpu_dirty);
        CHECK(segment->cpu_dirty);
    }

    // A prepared CPU write is exclusive; the advance refuses to run inside it.
    auto transaction = require_value(fixture.memory->prepare_cpu_write({backing_a, 0x18, 4}));
    const auto held_epoch = fixture.memory->cpu_cache_epoch();
    CHECK(!succeeded(fixture.memory->advance_cpu_epoch()));
    CHECK(fixture.memory->cpu_cache_epoch() == held_epoch);
    transaction->commit();
    transaction.reset();
    CHECK(succeeded(fixture.memory->advance_cpu_epoch()));
    CHECK(fixture.memory->cpu_cache_epoch() == held_epoch + 1);

    // Once the batched work submits and completes, the boundary waits for it
    // and keeps GPU authority, as before.
    batched.submit();
    batched.succeed();
    CHECK(succeeded(fixture.memory->internal_boundary()));
    CHECK(require_value(fixture.memory->inspect_storage(gpu_target.bytes))
              .front().storage == gpu_storage);
    CHECK(fixture.backend->writes == 0);
    CHECK(succeeded(fixture.memory->submission_boundary()));
    CHECK(fixture.backend->writes == 1);
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        CHECK(std::all_of(bytes.begin() + 8, bytes.begin() + 16,
                          [](std::byte byte) { return byte == std::byte{0x9d}; }));
    }
}

// The engine coherence callback resolves with ResolveReason::CpuEngineRead over
// exactly the range the front end is about to write; neighbouring GPU results
// keep their authority and their bytes.
void test_ranged_engine_resolve_writes_only_requested_range() {
    Fixture fixture;
    fixture.standard_layout();
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0; index != bytes.size(); ++index) {
            bytes[index] = std::byte(0x40 + index);
        }
    }

    TokenSource first_writer(42);
    const auto lower = retain(*fixture.memory, 0x1000, 8, ResourceAccess::Write);
    std::vector<std::byte> lower_bytes(8);
    for (std::size_t index = 0; index != lower_bytes.size(); ++index) {
        lower_bytes[index] = std::byte(0xb0 + index);
    }
    auto lower_storage = std::make_shared<Storage>(lower.bytes, lower_bytes);
    commit_write(*fixture.memory, lower, lower_storage, first_writer);
    first_writer.succeed();

    TokenSource second_writer(43);
    const auto upper = retain(*fixture.memory, 0x2008, 8, ResourceAccess::Write);
    std::vector<std::byte> upper_bytes(8);
    for (std::size_t index = 0; index != upper_bytes.size(); ++index) {
        upper_bytes[index] = std::byte(0xc0 + index);
    }
    auto upper_storage = std::make_shared<Storage>(upper.bytes, upper_bytes);
    commit_write(*fixture.memory, upper, upper_storage, second_writer);
    second_writer.succeed();

    CHECK(succeeded(fixture.memory->resolve({backing_a, 0xa, 4},
                                            ResolveReason::CpuEngineRead)));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0; index != 0xa; ++index) {
            CHECK(bytes[index] == std::byte(0x40 + index));
        }
        CHECK(bytes[0xa] == std::byte{0xc2});
        CHECK(bytes[0xb] == std::byte{0xc3});
        CHECK(bytes[0xc] == std::byte{0xc4});
        CHECK(bytes[0xd] == std::byte{0xc5});
        CHECK(bytes[0xe] == std::byte{0x4e});
        CHECK(bytes[0xf] == std::byte{0x4f});
    }
    const auto authority = require_value(fixture.memory->inspect_storage(upper.bytes));
    CHECK(authority.size() == 3);
    std::size_t pending = 0;
    for (const auto& entry : authority) {
        if (entry.dirty.gpu_dirty) {
            ++pending;
            CHECK(entry.storage == upper_storage);
        } else {
            CHECK(!entry.storage);
        }
    }
    CHECK(pending == 2);
    const auto lower_authority =
        require_value(fixture.memory->inspect_storage(lower.bytes));
    CHECK(lower_authority.size() == 1);
    CHECK(lower_authority.front().dirty.gpu_dirty);
    CHECK(lower_authority.front().storage == lower_storage);

    // The next submission boundary publishes everything the ranged resolve
    // left pending, completing the engine coherence sequence.
    CHECK(succeeded(fixture.memory->submission_boundary()));
    {
        std::lock_guard lock(fixture.backend->data_mutex);
        const auto& bytes = fixture.backend->find(backing_a)->bytes;
        for (std::size_t index = 0; index != 8; ++index) {
            CHECK(bytes[index] == std::byte(0xb0 + index));
        }
        for (std::size_t index = 8; index != 16; ++index) {
            CHECK(bytes[index] == std::byte(0xc0 + index - 8));
        }
    }
    CHECK(require_value(fixture.memory->inspect_storage(lower.bytes)).front().storage ==
          nullptr);
}

void test_discarded_backing_drops_authority_without_cpu_write() {
    Fixture fixture;
    fixture.standard_layout();
    const auto old = retain(*fixture.memory, 0x1008, 8, ResourceAccess::ReadWrite);
    const auto old_alias = retain(*fixture.memory, 0x2008, 8, ResourceAccess::ReadWrite);
    TokenSource producer(1);
    auto storage = std::make_shared<Storage>(
        old.bytes, std::vector<std::byte>(8, std::byte{0x9d}));
    commit_write(*fixture.memory, old, storage, producer);
    producer.succeed();

    TokenSource pending_token(2);
    const std::array pending_request{
        GpuAccessRequest{old_alias, ResourceAccess::Read}};
    auto pending = reserve(*fixture.memory, pending_request, pending_token.token);
    const std::array discarded{backing_a};
    CHECK(!succeeded(fixture.memory->invalidate_mappings(
        std::span<const MappingId>{}, discarded)));
    CHECK(fixture.memory->has_backing(backing_a));
    pending.reset();

    const auto writes = fixture.backend->writes.load();
    CHECK(succeeded(fixture.memory->invalidate_mappings(
        std::span<const MappingId>{}, discarded)));
    CHECK(fixture.backend->writes == writes);
    CHECK(!fixture.memory->has_backing(backing_a));
    CHECK(fixture.memory->has_backing(backing_b));
    CHECK(!succeeded(fixture.memory->validate_retained(old, ResourceAccess::Read)));
    CHECK(!succeeded(fixture.memory->validate_retained(old_alias, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x1008, 8, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->retain(0x2008, 8, ResourceAccess::Read)));
    CHECK(require_value(fixture.memory->retain(
              0x3008, 8, ResourceAccess::Read)).mapping == mapping_b);

    const BackingId replacement_backing{1, 1, 2};
    const MappingId replacement_mapping{7, 1, 2};
    fixture.backend->add(replacement_backing, 0x40, std::byte{0xaa});
    fixture.register_backing(replacement_backing);
    fixture.register_mapping(replacement_mapping, replacement_backing, mapped(0x1000));
    const auto replacement = retain(
        *fixture.memory, 0x1008, 8, ResourceAccess::ReadWrite);
    CHECK(replacement.mapping == replacement_mapping);
    CHECK(replacement.bytes.backing == replacement_backing);
    TokenSource stale_consumer(3);
    const std::array stale_request{GpuAccessRequest{old, ResourceAccess::Read}};
    CHECK(std::holds_alternative<RendererError>(
        fixture.memory->reserve_accesses(stale_request, stale_consumer.token)));
}

}  // namespace

int main() {
    test_deferred_host_store_excludes_aliases_without_waiting();
    test_deferred_host_store_blocks_cached_journal_admission();
    test_deferred_host_store_invalidates_native_tracker();
    test_deferred_host_store_invalidates_partial_tracking();
    test_captured_boundary_preserves_later_alias_authority();
    test_captured_boundary_same_storage_and_unpublished_journal();
    test_captured_boundary_failure_retry_and_lifetime();
    test_captured_boundary_waits_only_for_its_prefix();
    test_captured_boundary_resident_and_in_place_policy();
    test_batch_journal_repeats_publish_once();
    test_batch_journal_cpu_read_flushes_newest_writer();
    test_batch_journal_failure_and_partial_ownership();
    test_batch_journal_cross_queue_and_failed_feedback();
    test_factory_registration_aliases_and_versions();
    test_backing_identity_query_tracks_lifecycle();
    test_cpu_transactions_exclude_aliases_and_cancel_cleanly();
    test_overlapping_cpu_reads_and_pending_gpu_reader_coexist();
    test_remap_reallocation_generations_and_pins();
    test_partial_gpu_cpu_overlaps_preserve_newest_bytes();
    test_overlapping_gpu_writers_keep_newest_interval_authority();
    test_storage_inspection_tracks_clipped_current_authority();
    test_publication_boundary_does_not_wait_for_complete();
    test_submission_owner_retains_registry_and_reader_storage();
    test_deterministic_sequential_byte_oracle();
    test_fused_acceptance_before_submission_and_rejection();
    test_fused_acceptance_preserves_reader_hazards_and_failure();
    test_fused_acceptance_respects_pending_order_and_default_path();
    test_fused_acceptance_lease_retains_registry_state();
    for (const auto seed : {0xa341316cu, 0xc8013ea4u, 0xad90777du, 0x7e95761eu}) {
        test_randomized_eager_byte_model_equivalence(seed, false);
        test_randomized_eager_byte_model_equivalence(seed, true);
    }
    test_failed_resolves_retain_dirty_authority();
    test_guest_boundary_keeps_resident_authority();
    test_batched_mapping_registration();
    test_in_place_authority_and_host_ranges();
    test_atomic_reservations_validation_and_cancellation();
    test_forged_foreign_and_escalated_leases_are_rejected();
    test_destroyed_registry_leases_stay_foreign();
    test_retained_validation_checks_provenance_without_reserving();
    test_gpu_readers_block_cpu_writes_until_feedback();
    test_dependencies_and_boundary_epoch();
    test_failed_gpu_reader_survives_pruning_until_boundary();
    test_queue_order_applies_to_disjoint_pending_reservations();
    test_accepted_gpu_accesses_never_enter_backend_wait();
    test_cancelled_writes_and_whole_overwrite_coalesce_segments();
    test_coalescing_preserves_other_prepared_write_boundaries();
    test_resolves_and_boundary_coalesce_redundant_segments();
    test_unaligned_guest_base_preserves_absolute_granules();
    test_address_space_aliases_and_failed_store_accounting();
    test_bounds_overflow_and_mapping_validation();
    test_feedback_and_polling_while_backend_waits();
    test_mapping_invalidation_stales_leases_and_preserves_alias_state();
    test_mapping_invalidation_rejects_pending_and_accepted_accesses();
    test_mapping_invalidation_waits_for_submitted_accesses();
    test_failed_invalidation_writeback_is_atomic_and_retryable();
    test_multi_backing_invalidation_preserves_lookup_on_late_failure();
    test_discarded_backing_drops_authority_without_cpu_write();
    test_internal_boundary_preserves_gpu_authority_until_submission();
    test_internal_boundaries_keep_gpu_dirty_segments_cpu_clean();
    test_cpu_epoch_advance_is_valid_with_pending_gpu_work();
    test_ranged_engine_resolve_writes_only_requested_range();
    return failures == 0 ? 0 : 1;
}
