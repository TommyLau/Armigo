// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/memory.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace {

using armigo::graphics::AddressSpaceResult;
using armigo::graphics::GpuAddressSpace;
using armigo::graphics::Mapping;
using armigo::graphics::MemoryCallbacks;
using armigo::graphics::MemoryErrorCode;
using armigo::graphics::MemoryResult;
using armigo::graphics::make_address_space;

unsigned failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

struct Backing {
    static constexpr std::uint64_t base = 0x8000;
    std::array<std::byte, 0x100> bytes{};
    unsigned reads = 0;
    unsigned writes = 0;
    std::uint64_t last_address = 0;
    bool fail_reads = false;
    bool fail_writes = false;

    static bool read(void* raw, std::uint64_t address,
                     std::span<std::byte> destination) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        ++self.reads;
        self.last_address = address;
        if (self.fail_reads || address < base) {
            return false;
        }
        const auto offset = address - base;
        if (offset > self.bytes.size() ||
            destination.size() > self.bytes.size() - offset) {
            return false;
        }
        std::memcpy(destination.data(), self.bytes.data() + offset,
                    destination.size());
        return true;
    }

    static bool write(void* raw, std::uint64_t address,
                      std::span<const std::byte> source) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        ++self.writes;
        self.last_address = address;
        if (self.fail_writes || address < base) {
            return false;
        }
        const auto offset = address - base;
        if (offset > self.bytes.size() ||
            source.size() > self.bytes.size() - offset) {
            return false;
        }
        std::memcpy(self.bytes.data() + offset, source.data(), source.size());
        return true;
    }

    MemoryCallbacks callbacks() noexcept {
        return {read, write, this};
    }
};

Mapping mapping(std::uint64_t gpu_base = 0x1000,
                std::uint64_t mapped_size = 0x20,
                std::uint64_t buffer_offset = 0x10) {
    return {gpu_base, mapped_size, buffer_offset, Backing::base, 0x100, 0x10};
}

GpuAddressSpace require_space(std::span<const Mapping> mappings,
                              MemoryCallbacks callbacks) {
    AddressSpaceResult result = make_address_space(mappings, callbacks);
    CHECK(std::holds_alternative<GpuAddressSpace>(result));
    if (!std::holds_alternative<GpuAddressSpace>(result)) {
        std::fprintf(stderr, "Cannot continue after address-space construction failure\n");
        std::abort();
    }
    return std::get<GpuAddressSpace>(std::move(result));
}

void expect_build_error(std::span<const Mapping> mappings,
                        MemoryErrorCode expected) {
    const AddressSpaceResult result = make_address_space(mappings, {});
    CHECK(std::holds_alternative<MemoryResult>(result));
    if (const auto* error = std::get_if<MemoryResult>(&result)) {
        CHECK(error->code == expected);
        CHECK(!error->detail.empty());
    }
}

void test_offset_translation_and_live_backing() {
    Backing backing;
    const Mapping input = mapping();
    GpuAddressSpace space = require_space({&input, 1}, backing.callbacks());

    backing.bytes[0x14] = std::byte{0x31};
    std::array<std::byte, 1> output{};
    CHECK(space.read(0x1004, output));
    CHECK(output[0] == std::byte{0x31});
    CHECK(backing.last_address == Backing::base + 0x14);

    backing.bytes[0x14] = std::byte{0x52};
    CHECK(space.read(0x1004, output));
    CHECK(output[0] == std::byte{0x52});
    CHECK(backing.reads == 2);

    const std::array source{std::byte{0xa4}, std::byte{0xb5}};
    CHECK(space.write(0x1008, source));
    CHECK(backing.last_address == Backing::base + 0x18);
    CHECK(backing.bytes[0x18] == source[0]);
    CHECK(backing.bytes[0x19] == source[1]);
}

void test_bounds_holes_and_crossings() {
    Backing backing;
    const std::array mappings{mapping(0x1000, 0x10, 0x00),
                              mapping(0x1020, 0x10, 0x20)};
    GpuAddressSpace space = require_space(mappings, backing.callbacks());
    std::array<std::byte, 1> byte{};
    CHECK(space.read(0x100f, byte));
    CHECK(space.read(0x1010, byte).code == MemoryErrorCode::UnmappedAddress);

    std::array<std::byte, 9> crossing{};
    CHECK(space.read(0x1008, crossing).code ==
          MemoryErrorCode::AccessCrossesMapping);
    CHECK(space.read(std::numeric_limits<std::uint64_t>::max() - 1, crossing)
              .code == MemoryErrorCode::GpuRangeOverflow);

    const std::array adjacent{mapping(0x2000, 0x10, 0x00),
                              mapping(0x2010, 0x10, 0x20)};
    GpuAddressSpace adjacent_space = require_space(adjacent, backing.callbacks());
    std::array<std::byte, 2> two_bytes{};
    CHECK(adjacent_space.read(0x200f, two_bytes).code ==
          MemoryErrorCode::AccessCrossesMapping);
}

void test_mapping_validation() {
    Mapping invalid = mapping();
    invalid.mapped_size = 0;
    expect_build_error({&invalid, 1}, MemoryErrorCode::ZeroMappedSize);
    invalid = mapping();
    invalid.page_size = 3;
    expect_build_error({&invalid, 1}, MemoryErrorCode::InvalidPageSize);
    invalid = mapping();
    invalid.gpu_base += 1;
    expect_build_error({&invalid, 1}, MemoryErrorCode::MisalignedGpuBase);
    invalid = mapping();
    invalid.mapped_size += 1;
    expect_build_error({&invalid, 1}, MemoryErrorCode::MisalignedMappedSize);
    invalid = mapping();
    invalid.buffer_offset += 1;
    expect_build_error({&invalid, 1}, MemoryErrorCode::MisalignedBufferOffset);
    invalid = mapping(std::numeric_limits<std::uint64_t>::max() - 0xf, 0x20, 0);
    expect_build_error({&invalid, 1}, MemoryErrorCode::GpuRangeOverflow);
    invalid = mapping();
    invalid.backing_size = 0x0f;
    expect_build_error({&invalid, 1}, MemoryErrorCode::BackingRangeExceeded);
    invalid = mapping();
    invalid.guest_address = std::numeric_limits<std::uint64_t>::max() - 7;
    expect_build_error({&invalid, 1}, MemoryErrorCode::GuestRangeOverflow);

    const std::array overlap{mapping(0x1000, 0x20, 0x00),
                             mapping(0x1010, 0x10, 0x20)};
    expect_build_error(overlap, MemoryErrorCode::OverlappingMappings);

    Mapping unaligned_backing = mapping();
    unaligned_backing.backing_size = 0x31;
    const AddressSpaceResult accepted = make_address_space(
        {&unaligned_backing, 1}, {});
    CHECK(std::holds_alternative<GpuAddressSpace>(accepted));
}

void test_padded_gpu_extent_checks_backing_per_access() {
    Backing backing;
    Mapping padded = mapping(0x4000, 0x40, 0x10);
    padded.backing_size = 0x20;
    GpuAddressSpace space = require_space({&padded, 1}, backing.callbacks());

    std::array<std::byte, 1> byte{};
    CHECK(space.read(0x400f, byte));
    const unsigned reads = backing.reads;
    CHECK(space.read(0x4010, byte).code ==
          MemoryErrorCode::BackingRangeExceeded);
    CHECK(backing.reads == reads);

    const unsigned writes = backing.writes;
    CHECK(space.write(0x4010, byte).code ==
          MemoryErrorCode::BackingRangeExceeded);
    CHECK(backing.writes == writes);
}

void test_mapping_values_are_owned() {
    Backing backing;
    std::vector<Mapping> input{mapping(0x3000, 0x10, 0x20)};
    GpuAddressSpace space = require_space(input, backing.callbacks());
    input[0].gpu_base = 0x9000;
    input.clear();

    CHECK(space.mappings().size() == 1);
    CHECK(space.mappings()[0].gpu_base == 0x3000);
    std::array<std::byte, 1> byte{};
    CHECK(space.read(0x3000, byte));
    CHECK(backing.last_address == Backing::base + 0x20);
}

void test_views_have_distinct_generations() {
    Backing backing;
    const std::vector<Mapping> input{mapping(0x3000, 0x10, 0x20)};
    GpuAddressSpace first = require_space(input, backing.callbacks());
    GpuAddressSpace second = require_space(input, backing.callbacks());
    const GpuAddressSpace copy = first;

    CHECK(first.generation() != 0);
    CHECK(second.generation() != 0);
    CHECK(first.generation() != second.generation());
    CHECK(copy.generation() == first.generation());
}

void test_callback_errors_and_empty_accesses() {
    const Mapping input = mapping();
    GpuAddressSpace missing = require_space({&input, 1}, {});
    std::array<std::byte, 1> byte{};
    const MemoryResult missing_read = missing.read(0x1000, byte);
    CHECK(missing_read.code == MemoryErrorCode::ReaderUnavailable);
    CHECK(!missing_read.detail.empty());
    const MemoryResult missing_write = missing.write(0x1000, byte);
    CHECK(missing_write.code == MemoryErrorCode::WriterUnavailable);
    CHECK(!missing_write.detail.empty());

    Backing backing;
    GpuAddressSpace space = require_space({&input, 1}, backing.callbacks());
    backing.fail_reads = true;
    const MemoryResult failed_read = space.read(0x1000, byte);
    CHECK(failed_read.code == MemoryErrorCode::ReadFailed);
    CHECK(!failed_read.detail.empty());
    backing.fail_writes = true;
    const MemoryResult failed_write = space.write(0x1000, byte);
    CHECK(failed_write.code == MemoryErrorCode::WriteFailed);
    CHECK(!failed_write.detail.empty());

    const unsigned writes = backing.writes;
    CHECK(space.write(0x2000, byte).code == MemoryErrorCode::UnmappedAddress);
    CHECK(backing.writes == writes);

    GpuAddressSpace empty = require_space({}, {});
    CHECK(empty.read(std::numeric_limits<std::uint64_t>::max(), {}));
    CHECK(empty.write(std::numeric_limits<std::uint64_t>::max(), {}));
}

}  // namespace

int main() {
    test_offset_translation_and_live_backing();
    test_bounds_holes_and_crossings();
    test_mapping_validation();
    test_padded_gpu_extent_checks_backing_per_access();
    test_mapping_values_are_owned();
    test_views_have_distinct_generations();
    test_callback_errors_and_empty_accesses();
    return failures == 0 ? 0 : 1;
}
