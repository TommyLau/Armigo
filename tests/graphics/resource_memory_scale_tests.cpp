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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

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

RendererError synthetic_error(const char* detail) {
    return RendererError({CommandStatus::MemoryFault, detail});
}

using BackingKey = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;

BackingKey key(BackingId id) {
    return {id.owner, id.allocation, id.generation};
}

struct ProbeLease final : ResourceLease {};

struct Backend final : ResourceMemoryBackend {
    void add(BackingId id, std::size_t size, std::byte fill = {}) {
        allocations.emplace(key(id), std::vector<std::byte>(size, fill));
    }

    std::vector<std::byte>* find(BackingId id) {
        const auto found = allocations.find(key(id));
        return found == allocations.end() ? nullptr : &found->second;
    }

    RendererStatus read_cpu(BackingRange range,
                            std::span<std::byte> destination) override {
        ++reads;
        auto* allocation = find(range.backing);
        if (!allocation || destination.size() != range.size ||
            range.offset > allocation->size() ||
            range.size > allocation->size() - range.offset) {
            return synthetic_error("synthetic scale CPU read bounds failure");
        }
        std::copy_n(allocation->begin() + range.offset, destination.size(),
                    destination.begin());
        return renderer_success();
    }

    RendererStatus write_cpu(BackingRange range,
                             std::span<const std::byte> source) override {
        ++writes;
        auto* allocation = find(range.backing);
        if (!allocation || source.size() != range.size ||
            range.offset > allocation->size() ||
            range.size > allocation->size() - range.offset) {
            return synthetic_error("synthetic scale CPU write bounds failure");
        }
        std::copy(source.begin(), source.end(), allocation->begin() + range.offset);
        return renderer_success();
    }

    RendererStatus wait_gpu(CompletionToken token) override {
        ++waits;
        if (!token.valid()) return synthetic_error("synthetic invalid GPU token");
        return renderer_success();
    }

    std::map<BackingKey, std::vector<std::byte>> allocations;
    std::uint64_t reads = 0;
    std::uint64_t writes = 0;
    std::uint64_t waits = 0;
};

struct Storage final : ResourceMemoryStorage {
    Storage(BackingRange range, std::vector<std::byte> contents)
        : range(range), contents(std::move(contents)) {}

    BackingRange coverage() const noexcept override { return range; }

    RendererStatus read_gpu(BackingRange requested,
                            std::span<std::byte> destination) const override {
        if (requested.backing != range.backing || requested.offset < range.offset ||
            destination.size() != requested.size ||
            requested.offset - range.offset > contents.size() ||
            requested.size > contents.size() - (requested.offset - range.offset)) {
            return synthetic_error("synthetic scale GPU read bounds failure");
        }
        const auto offset = requested.offset - range.offset;
        std::copy_n(contents.begin() + offset, destination.size(), destination.begin());
        return renderer_success();
    }

    BackingRange range;
    std::vector<std::byte> contents;
};

struct NoopActions final : CompletionActions {
    RendererStatus make_coherent() override { return renderer_success(); }
    RendererStatus prepare_publication() override { return renderer_success(); }
    void publish() noexcept override {}
    RendererStatus preserve_failed_authority() override {
        return renderer_success();
    }
};

struct TokenSource {
    explicit TokenSource(std::uint64_t sequence)
        : completion({1, sequence}, {}, std::make_shared<NoopActions>()),
          token(completion.token()) {}

    void submit() { CHECK(succeeded(completion.submitted())); }
    void succeed() { CHECK(succeeded(completion.feedback({}))); }

    SubmissionCompletion completion;
    CompletionToken token;
};

struct Fixture {
    explicit Fixture(std::uint64_t granule = 1)
        : backend(std::make_shared<Backend>()),
          counters(std::make_shared<ResourceMemoryWorkCounters>()),
          memory(require_value(make_resource_memory(
              {23, backend, CoherenceMode::SubmissionBoundary, granule, counters}))) {}

    void add_backing(BackingId id, std::uint64_t size, std::byte fill = {}) {
        backend->add(id, size, fill);
        CHECK(succeeded(memory->register_backing(
            {id, size, std::make_shared<ProbeLease>()})));
    }

    void add_mapping(MappingId id, BackingId backing, std::uint64_t gpu_base,
                     std::uint64_t guest_base, std::uint64_t size) {
        CHECK(succeeded(memory->register_mapping(
            {id, backing, {gpu_base, size, 0, guest_base, size, 0x10}})));
    }

    void reset_counters() { *counters = {}; }

    std::shared_ptr<Backend> backend;
    std::shared_ptr<ResourceMemoryWorkCounters> counters;
    std::shared_ptr<ResourceMemoryRegistry> memory;
};

struct LookupCounts {
    std::uint64_t retain_mappings = 0;
    std::uint64_t inspect_backings = 0;
    std::uint64_t inspect_intervals = 0;
    std::uint64_t inspect_visits = 0;
    std::uint64_t read_backings = 0;
    std::uint64_t read_intervals = 0;
    std::uint64_t read_visits = 0;
};

LookupCounts measure_lookup_scale(std::size_t count) {
    Fixture fixture;
    constexpr std::uint64_t size = 0x40;
    constexpr std::uint64_t stride = 0x100;
    constexpr std::uint64_t gpu_base = 0x100000;
    constexpr std::uint64_t guest_base = 0x10000000;
    for (std::size_t index = 0; index != count; ++index) {
        const BackingId backing{1, index + 1, 1};
        const MappingId mapping{23, index + 1, 1};
        fixture.add_backing(backing, size, std::byte(index & 0xff));
        fixture.add_mapping(mapping, backing, gpu_base + index * stride,
                            guest_base + index * stride, size);
    }

    const BackingId target_backing{1, count, 1};
    const std::uint64_t target_address = gpu_base + (count - 1) * stride + size - 1;
    fixture.reset_counters();
    const auto retained = require_value(fixture.memory->retain(
        target_address, 1, ResourceAccess::ReadWrite));
    LookupCounts result;
    result.retain_mappings = fixture.counters->mapping_comparisons;
    CHECK(retained.bytes == BackingRange{target_backing, size - 1, 1});

    fixture.reset_counters();
    const auto inspected = require_value(fixture.memory->inspect(retained.bytes));
    result.inspect_backings = fixture.counters->backing_comparisons;
    result.inspect_intervals = fixture.counters->interval_comparisons;
    result.inspect_visits = fixture.counters->interval_visits;
    CHECK(inspected.size() == 1);
    CHECK(inspected.front().bytes == retained.bytes);

    fixture.reset_counters();
    std::array<std::byte, 1> copied{};
    CHECK(succeeded(fixture.memory->read_retained(
        retained, retained.bytes, copied)));
    result.read_backings = fixture.counters->backing_comparisons;
    result.read_intervals = fixture.counters->interval_comparisons;
    result.read_visits = fixture.counters->interval_visits;
    CHECK(copied.front() == std::byte((count - 1) & 0xff));

    CHECK(std::holds_alternative<RendererError>(fixture.memory->retain(
        gpu_base - 1, 1, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->retain(
        gpu_base + count * stride, 1, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(fixture.memory->retain(
        target_address, 2, ResourceAccess::Read)));
    const auto first = require_value(fixture.memory->retain(
        gpu_base, 1, ResourceAccess::Read));
    CHECK(first.bytes == BackingRange{BackingId{1, 1, 1}, 0, 1});
    return result;
}

void test_indexed_lookup_scales_sublinearly() {
    const auto small = measure_lookup_scale(1024);
    const auto large = measure_lookup_scale(4096);
    std::fprintf(stdout,
                 "resource lookup scale: mappings=1024/4096 retain=%llu/%llu "
                 "inspect_backing=%llu/%llu inspect_interval=%llu/%llu "
                 "read_backing=%llu/%llu read_interval=%llu/%llu\n",
                 static_cast<unsigned long long>(small.retain_mappings),
                 static_cast<unsigned long long>(large.retain_mappings),
                 static_cast<unsigned long long>(small.inspect_backings),
                 static_cast<unsigned long long>(large.inspect_backings),
                 static_cast<unsigned long long>(small.inspect_intervals),
                 static_cast<unsigned long long>(large.inspect_intervals),
                 static_cast<unsigned long long>(small.read_backings),
                 static_cast<unsigned long long>(large.read_backings),
                 static_cast<unsigned long long>(small.read_intervals),
                 static_cast<unsigned long long>(large.read_intervals));

    CHECK(small.retain_mappings > 0);
    CHECK(large.retain_mappings <= small.retain_mappings * 2);
    // Retain searches both the address index and the owning identity map.
    CHECK(large.retain_mappings < 128);
    CHECK(large.inspect_backings <= small.inspect_backings * 2);
    CHECK(large.inspect_backings < 64);
    CHECK(large.inspect_intervals <= small.inspect_intervals * 2);
    CHECK(large.inspect_intervals < 64);
    CHECK(large.inspect_visits <= 4);
    CHECK(large.read_backings <= small.read_backings * 2);
    CHECK(large.read_backings < 192);
    CHECK(large.read_intervals <= small.read_intervals * 2);
    CHECK(large.read_intervals < 128);
    CHECK(large.read_visits <= 8);
}

struct IntervalCounts {
    std::uint64_t inspect_comparisons = 0;
    std::uint64_t inspect_visits = 0;
    std::uint64_t read_commit_comparisons = 0;
    std::uint64_t read_commit_visits = 0;
};

IntervalCounts measure_interval_scale(std::size_t count) {
    Fixture fixture;
    const BackingId backing{2, 1, 1};
    const MappingId mapping{23, 1, 1};
    constexpr std::uint64_t gpu_base = 0x200000;
    constexpr std::uint64_t guest_base = 0x20000000;
    fixture.add_backing(backing, count);
    fixture.add_mapping(mapping, backing, gpu_base, guest_base, count);
    for (std::size_t offset = 0; offset != count; ++offset) {
        auto write = require_value(fixture.memory->prepare_cpu_write(
            {backing, offset, 1}));
        fixture.backend->find(backing)->at(offset) = std::byte(offset & 0xff);
        write->commit();
    }

    fixture.reset_counters();
    const auto all = require_value(fixture.memory->inspect({backing, 0, count}));
    CHECK(all.size() == count);
    CHECK(fixture.counters->interval_visits <= count * 2 + 2);

    const std::size_t offset = count * 3 / 4;
    fixture.reset_counters();
    const auto one = require_value(fixture.memory->inspect({backing, offset, 1}));
    IntervalCounts result;
    result.inspect_comparisons = fixture.counters->interval_comparisons;
    result.inspect_visits = fixture.counters->interval_visits;
    CHECK(one.size() == 1);
    CHECK(one.front().bytes == BackingRange{backing, offset, 1});
    CHECK(one.front().version.cpu_write == offset + 1);

    const auto retained = require_value(fixture.memory->retain(
        gpu_base + offset, 1, ResourceAccess::Read));
    TokenSource reader(1);
    const std::array request{GpuAccessRequest{retained, ResourceAccess::Read}};
    fixture.reset_counters();
    auto reservation = require_value(
        fixture.memory->reserve_accesses(request, reader.token));
    CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
    result.read_commit_comparisons = fixture.counters->interval_comparisons;
    result.read_commit_visits = fixture.counters->interval_visits;
    reservation.reset();
    reader.submit();
    reader.succeed();
    return result;
}

void test_interval_index_scales_with_touched_ranges() {
    const auto small = measure_interval_scale(1024);
    const auto large = measure_interval_scale(4096);
    std::fprintf(stdout,
                 "resource interval scale: intervals=1024/4096 inspect=%llu/%llu "
                 "visits=%llu/%llu read_commit=%llu/%llu read_visits=%llu/%llu\n",
                 static_cast<unsigned long long>(small.inspect_comparisons),
                 static_cast<unsigned long long>(large.inspect_comparisons),
                 static_cast<unsigned long long>(small.inspect_visits),
                 static_cast<unsigned long long>(large.inspect_visits),
                 static_cast<unsigned long long>(small.read_commit_comparisons),
                 static_cast<unsigned long long>(large.read_commit_comparisons),
                 static_cast<unsigned long long>(small.read_commit_visits),
                 static_cast<unsigned long long>(large.read_commit_visits));

    CHECK(large.inspect_comparisons <= small.inspect_comparisons * 2);
    CHECK(large.inspect_comparisons < 96);
    CHECK(large.inspect_visits <= 4);
    CHECK(large.read_commit_comparisons <= small.read_commit_comparisons * 2);
    CHECK(large.read_commit_comparisons < 256);
    CHECK(large.read_commit_visits <= 12);
}

void commit_gpu_write(Fixture& fixture, RetainedRange range, std::byte value,
                      TokenSource& token) {
    const std::array request{GpuAccessRequest{range, ResourceAccess::Write}};
    auto reservation = require_value(
        fixture.memory->reserve_accesses(request, token.token));
    auto storage = std::make_shared<Storage>(
        range.bytes, std::vector<std::byte>(range.bytes.size, value));
    CHECK(succeeded(reservation->commit({storage})));
    reservation.reset();
    token.submit();
    token.succeed();
}

void test_sparse_commit_preserves_gaps_and_overlapping_request_order() {
    Fixture fixture;
    const BackingId backing{5, 1, 1};
    constexpr std::uint64_t gpu_base = 0x800000;
    fixture.add_backing(backing, 64);
    fixture.add_mapping({23, 1, 1}, backing, gpu_base, 0x80000000, 64);
    const auto whole = require_value(fixture.memory->retain(
        gpu_base, 64, ResourceAccess::ReadWrite));
    TokenSource original(1);
    commit_gpu_write(fixture, whole, std::byte{0x11}, original);
    auto cpu_write = require_value(fixture.memory->prepare_cpu_write({backing, 32, 8}));
    std::fill_n(fixture.backend->find(backing)->begin() + 32, 8, std::byte{0x77});
    cpu_write->commit();
    cpu_write.reset();
    const auto untouched = require_value(fixture.memory->inspect({backing, 32, 8}));
    CHECK(untouched.size() == 1);

    auto retained = [&](std::uint64_t offset, std::uint64_t size) {
        return require_value(fixture.memory->retain(gpu_base + offset, size, ResourceAccess::ReadWrite));
    };
    const auto first = retained(4, 4);
    const auto second = retained(48, 4);
    const std::array requests{
        GpuAccessRequest{first, ResourceAccess::Write},
        GpuAccessRequest{retained(0, 16), ResourceAccess::Read},
        GpuAccessRequest{retained(46, 10), ResourceAccess::Read},
        GpuAccessRequest{second, ResourceAccess::Write},
        GpuAccessRequest{retained(6, 4), ResourceAccess::Read}};
    TokenSource updated(2);
    auto reservation = require_value(fixture.memory->reserve_accesses(requests, updated.token));
    CHECK(succeeded(reservation->commit({
        std::make_shared<Storage>(first.bytes, std::vector<std::byte>(4, std::byte{0xa4})),
        std::make_shared<ProbeLease>(), std::make_shared<ProbeLease>(),
        std::make_shared<Storage>(second.bytes, std::vector<std::byte>(4, std::byte{0xb8})),
        std::make_shared<ProbeLease>()})));
    reservation.reset();
    updated.submit();
    updated.succeed();

    const auto gap = require_value(fixture.memory->inspect({backing, 32, 8}));
    CHECK(gap.size() == 1);
    CHECK(gap.front().version == untouched.front().version);
    CHECK(gap.front().cpu_dirty == untouched.front().cpu_dirty);
    CHECK(!gap.front().gpu_dirty);
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const auto& bytes = *fixture.backend->find(backing);
    for (std::size_t offset = 0; offset != bytes.size(); ++offset) {
        const auto expected = offset >= 4 && offset < 8 ? std::byte{0xa4}
            : offset >= 48 && offset < 52 ? std::byte{0xb8}
            : offset >= 32 && offset < 40 ? std::byte{0x77} : std::byte{0x11};
        CHECK(bytes[offset] == expected);
    }
}

void test_readback_inspection_preserves_partial_resolve_edges() {
    struct InspectingStorage final : ResourceMemoryStorage {
        BackingRange bytes;
        std::weak_ptr<ResourceMemoryRegistry> memory;
        BackingRange coverage() const noexcept override { return bytes; }
        RendererStatus read_gpu(BackingRange range, std::span<std::byte> destination) const override {
            const auto registry = memory.lock();
            if (!registry) return synthetic_error("readback registry was released");
            const auto dirty = require_value(registry->inspect(range));
            const auto stored = require_value(registry->inspect_storage(range));
            CHECK(dirty.size() == 1 && dirty.front().bytes == range);
            CHECK(stored.size() == 1 && stored.front().dirty.bytes == range);
            CHECK(stored.front().storage.get() == this);
            CHECK(stored.front().dirty.gpu_dirty);
            std::fill(destination.begin(), destination.end(), std::byte{0x5a});
            return renderer_success();
        }
    };
    Fixture fixture;
    const BackingId backing{6, 1, 1};
    constexpr std::uint64_t gpu_base = 0x900000;
    fixture.add_backing(backing, 64);
    fixture.add_mapping({23, 1, 1}, backing, gpu_base, 0x90000000, 64);
    const auto whole = require_value(fixture.memory->retain(gpu_base, 64, ResourceAccess::Write));
    auto storage = std::make_shared<InspectingStorage>();
    storage->bytes = whole.bytes;
    storage->memory = fixture.memory;
    TokenSource writer(1);
    const std::array request{GpuAccessRequest{whole, ResourceAccess::Write}};
    auto reservation = require_value(fixture.memory->reserve_accesses(request, writer.token));
    CHECK(succeeded(reservation->commit({storage})));
    reservation.reset();
    writer.submit();
    writer.succeed();
    for (const auto offset : {8u, 0u, 60u}) {
        CHECK(succeeded(fixture.memory->resolve({backing, offset, 4}, ResolveReason::Capture)));
        const auto& bytes = *fixture.backend->find(backing);
        CHECK(std::all_of(bytes.begin() + offset, bytes.begin() + offset + 4,
                          [](std::byte value) { return value == std::byte{0x5a}; }));
    }
    CHECK(fixture.backend->find(backing)->at(4) == std::byte{});
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const auto& bytes = *fixture.backend->find(backing);
    CHECK(std::all_of(bytes.begin(), bytes.end(), [](std::byte value) { return value == std::byte{0x5a}; }));
}

void test_read_only_epochs_do_not_accumulate_physical_splits() {
    Fixture fixture;
    const BackingId backing{7, 1, 1};
    constexpr std::uint64_t gpu_base = 0xa00000;
    constexpr std::uint64_t count = 1024;
    fixture.add_backing(backing, count * 4);
    fixture.add_mapping({23, 1, 1}, backing, gpu_base, 0xa0000000, count * 4);
    for (std::uint64_t index = 0; index != count; ++index) {
        const auto retained = require_value(fixture.memory->retain(
            gpu_base + index * 4 + 1, 1, ResourceAccess::Read));
        TokenSource reader(index + 1);
        const std::array request{GpuAccessRequest{retained, ResourceAccess::Read}};
        auto reservation = require_value(fixture.memory->reserve_accesses(request, reader.token));
        CHECK(succeeded(reservation->commit({std::make_shared<ProbeLease>()})));
        reservation.reset();
        if (index == 0) {
            const auto epoch = fixture.memory->cpu_cache_epoch();
            CHECK(!succeeded(fixture.memory->submission_boundary()));
            CHECK(fixture.memory->cpu_cache_epoch() == epoch);
        }
        reader.submit();
        reader.succeed();
        fixture.reset_counters();
        CHECK(succeeded(fixture.memory->submission_boundary()));
        CHECK(fixture.counters->boundary_backings == 0);
        // Reads add no epoch edges; only the boundary's own work counts.
        CHECK(fixture.counters->boundary_epoch_edges <= 2);
        CHECK(fixture.counters->interval_visits <= 4);
        fixture.reset_counters();
        const auto intervals = require_value(fixture.memory->inspect({backing, 0, count * 4}));
        CHECK(intervals.size() == 1);
        // Snapshot merging alone can hide redundant physical nodes. Count the
        // actual source intervals visited after every disjoint read-only epoch.
        CHECK(fixture.counters->interval_visits == 1);
    }
}

void test_boundary_visits_only_gpu_dirty_work() {
    Fixture fixture;
    constexpr std::size_t backing_count = 4096;
    constexpr std::size_t interval_count = 4096;
    const BackingId fragmented{3, 1, 1};
    constexpr std::uint64_t fragmented_gpu = 0x300000;
    constexpr std::uint64_t fragmented_guest = 0x30000000;
    for (std::size_t index = 0; index != backing_count; ++index) {
        fixture.add_backing({3, index + 1, 1},
                            index == 0 ? interval_count : 0x40);
    }
    fixture.add_mapping({23, 1, 1}, fragmented, fragmented_gpu,
                        fragmented_guest, interval_count);
    for (std::size_t offset = 0; offset != interval_count; ++offset) {
        auto write = require_value(fixture.memory->prepare_cpu_write(
            {fragmented, offset, 1}));
        fixture.backend->find(fragmented)->at(offset) = std::byte(offset & 0xff);
        write->commit();
    }

    fixture.reset_counters();
    const auto idle_epoch = fixture.memory->cpu_cache_epoch();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    CHECK(fixture.memory->cpu_cache_epoch() == idle_epoch + 1);
    CHECK(fixture.counters->interval_visits == 0);
    CHECK(fixture.counters->backing_comparisons == 0);
    CHECK(fixture.counters->boundary_backings == 0);
    CHECK(fixture.counters->boundary_epoch_edges == 0);

    const BackingId ordinary{3, backing_count, 1};
    constexpr std::uint64_t ordinary_gpu = 0x400000;
    fixture.add_mapping({23, 2, 1}, ordinary, ordinary_gpu,
                        0x40000000, 0x40);
    auto first = require_value(fixture.memory->retain(
        fragmented_gpu + 17, 1, ResourceAccess::Write));
    auto second = require_value(fixture.memory->retain(
        fragmented_gpu + 4000, 1, ResourceAccess::Write));
    auto third = require_value(fixture.memory->retain(
        ordinary_gpu + 31, 1, ResourceAccess::Write));
    TokenSource first_writer(1);
    TokenSource second_writer(2);
    TokenSource third_writer(3);
    commit_gpu_write(fixture, first, std::byte{0xa1}, first_writer);
    commit_gpu_write(fixture, second, std::byte{0xb2}, second_writer);
    commit_gpu_write(fixture, third, std::byte{0xc3}, third_writer);

    const auto writes_before = fixture.backend->writes;
    const auto epoch = fixture.memory->cpu_cache_epoch();
    fixture.reset_counters();
    CHECK(succeeded(fixture.memory->submission_boundary()));
    const auto measured = *fixture.counters;
    CHECK(fixture.memory->cpu_cache_epoch() == epoch + 1);
    CHECK(measured.boundary_backings == 2);
    CHECK(measured.interval_visits <= 24);
    CHECK(measured.backing_comparisons < 128);
    CHECK(fixture.backend->writes == writes_before + 3);
    CHECK(fixture.backend->find(fragmented)->at(17) == std::byte{0xa1});
    CHECK(fixture.backend->find(fragmented)->at(4000) == std::byte{0xb2});
    CHECK(fixture.backend->find(ordinary)->at(31) == std::byte{0xc3});
    const auto resolved = require_value(fixture.memory->inspect(
        {fragmented, 17, 1}));
    CHECK(resolved.size() == 1);
    CHECK(resolved.front().cpu_dirty);
    CHECK(!resolved.front().gpu_dirty);
    std::fprintf(stdout,
                 "resource boundary scale: backings=%zu intervals=%zu "
                 "dirty_backings=%llu interval_visits=%llu backing_comparisons=%llu\n",
                 backing_count, interval_count,
                 static_cast<unsigned long long>(measured.boundary_backings),
                 static_cast<unsigned long long>(measured.interval_visits),
                 static_cast<unsigned long long>(measured.backing_comparisons));
}

void test_read_retained_validation_and_mapping_lifecycle() {
    Fixture fixture;
    const BackingId backing{4, 1, 1};
    const MappingId mapping{23, 1, 1};
    constexpr std::uint64_t gpu_base = 0x500000;
    fixture.add_backing(backing, 0x40, std::byte{0x6d});
    fixture.add_mapping(mapping, backing, gpu_base, 0x50000000, 0x40);
    auto write_only = require_value(fixture.memory->retain(
        gpu_base + 8, 8, ResourceAccess::Write));
    std::array<std::byte, 4> copied{};
    CHECK(succeeded(fixture.memory->read_retained(
        write_only, {backing, 10, copied.size()}, copied)));
    CHECK(std::all_of(copied.begin(), copied.end(),
                      [](std::byte value) { return value == std::byte{0x6d}; }));

    CHECK(!succeeded(fixture.memory->read_retained(
        write_only, {backing, 7, copied.size()}, copied)));
    CHECK(!succeeded(fixture.memory->read_retained(
        write_only, {backing, 14, copied.size()}, copied)));
    std::array<std::byte, 3> short_output{};
    CHECK(!succeeded(fixture.memory->read_retained(
        write_only, {backing, 10, 4}, short_output)));
    auto mismatched = write_only;
    ++mismatched.bytes.offset;
    CHECK(!succeeded(fixture.memory->read_retained(
        mismatched, {backing, 10, copied.size()}, copied)));

    Fixture foreign;
    foreign.add_backing(backing, 0x40, std::byte{0x91});
    foreign.add_mapping(mapping, backing, gpu_base, 0x60000000, 0x40);
    CHECK(!succeeded(foreign.memory->read_retained(
        write_only, {backing, 10, copied.size()}, copied)));

    const std::array removed{mapping};
    CHECK(succeeded(fixture.memory->invalidate_mappings(removed)));
    CHECK(!succeeded(fixture.memory->read_retained(
        write_only, {backing, 10, copied.size()}, copied)));
    const MappingId remapped{23, 1, 2};
    fixture.add_mapping(remapped, backing, gpu_base, 0x50000000, 0x40);
    auto current = require_value(fixture.memory->retain(
        gpu_base + 8, 8, ResourceAccess::ReadWrite));
    CHECK(current.mapping == remapped);
    CHECK(!succeeded(fixture.memory->unmap(remapped)));
    current.lease.reset();
    CHECK(succeeded(fixture.memory->unmap(remapped)));
    CHECK(!succeeded(fixture.memory->retire_backing(backing)));
    mismatched.lease.reset();
    write_only.lease.reset();
    CHECK(succeeded(fixture.memory->retire_backing(backing)));

    const BackingId discarded{4, 2, 1};
    const MappingId discarded_mapping{23, 2, 1};
    fixture.add_backing(discarded, 0x40);
    fixture.add_mapping(discarded_mapping, discarded, 0x600000,
                        0x70000000, 0x40);
    const auto stale_discarded = require_value(fixture.memory->retain(
        0x600000, 1, ResourceAccess::Read));
    const std::array discarded_backings{discarded};
    CHECK(succeeded(fixture.memory->invalidate_mappings(
        std::span<const MappingId>{}, discarded_backings)));
    CHECK(!fixture.memory->has_backing(discarded));
    CHECK(!succeeded(fixture.memory->read_retained(
        stale_discarded, {discarded, 0, 1},
        std::span<std::byte>(copied).first<1>())));
}

}  // namespace

int main() {
    test_indexed_lookup_scales_sublinearly();
    test_interval_index_scales_with_touched_ranges();
    test_sparse_commit_preserves_gaps_and_overlapping_request_order();
    test_readback_inspection_preserves_partial_resolve_edges();
    test_read_only_epochs_do_not_accumulate_physical_splits();
    test_boundary_visits_only_gpu_dirty_work();
    test_read_retained_validation_and_mapping_lifecycle();
    return failures == 0 ? 0 : 1;
}
