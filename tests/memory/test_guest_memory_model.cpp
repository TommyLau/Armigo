// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independent verification tests for the guest virtual-memory metadata model.
//
// The tests use a self-contained assertion harness (no third-party code) and
// a per-page oracle for a randomized-but-deterministic differential
// comparison. All fixtures are synthetic.

#include "armigo/memory/guest_memory_model.h"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using armigo::memory::GuestAddress;
using armigo::memory::GuestMemoryModel;
using armigo::memory::MappingInfo;
using armigo::memory::MemoryError;
using armigo::memory::Permission;

// ---------------------------------------------------------------------------
// Minimal assertion harness
// ---------------------------------------------------------------------------

int g_checks_failed = 0;
int g_checks_total = 0;

void record_check(bool passed, const std::string& description, const char* file, int line) {
    ++g_checks_total;
    if (!passed) {
        ++g_checks_failed;
        std::printf("FAIL %s:%d: %s\n", file, line, description.c_str());
    }
}

template <typename T>
void describe_value(std::ostringstream& text, const T& value) {
    text << value;
}

void describe_value(std::ostringstream& text, Permission value) {
    const auto raw = static_cast<unsigned>(static_cast<std::uint8_t>(value));
    text << "'";
    if (!armigo::memory::has_known_permission_bits(value)) {
        text << "unk";
    } else {
        text << (armigo::memory::has_permission(value, Permission::read) ? "r" : "-");
        text << (armigo::memory::has_permission(value, Permission::write) ? "w" : "-");
        text << (armigo::memory::has_permission(value, Permission::execute) ? "x" : "-");
    }
    text << "'(0x" << std::hex << raw << std::dec << ")";
}

void describe_value(std::ostringstream& text, MemoryError value) {
    text << armigo::memory::to_string(value);
}

template <typename A, typename B>
std::string describe_mismatch(const char* a_expr, const A& a, const char* b_expr, const B& b) {
    std::ostringstream text;
    text << a_expr << " == " << b_expr << " (got ";
    describe_value(text, a);
    text << " vs ";
    describe_value(text, b);
    text << ")";
    return text.str();
}

}  // namespace

#define CHECK(condition) record_check((condition), #condition, __FILE__, __LINE__)

#define CHECK_EQ(actual, expected)                                              \
    do {                                                                        \
        const auto& actual_value = (actual);                                    \
        const auto& expected_value = (expected);                                \
        record_check(                                                           \
            actual_value == expected_value,                                     \
            describe_mismatch(#actual, actual_value, #expected, expected_value), \
            __FILE__, __LINE__);                                                \
    } while (false)

// A failed precondition aborts the current test function; the harness reports
// it like any other failure.
#define REQUIRE(condition)                                         \
    do {                                                           \
        if (!(condition)) {                                        \
            record_check(false, #condition, __FILE__, __LINE__);   \
            return;                                                \
        }                                                          \
    } while (false)

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool same_region(const std::optional<MappingInfo>& mapping, GuestAddress start,
                 std::uint64_t size, Permission permissions) {
    if (!mapping.has_value()) {
        return false;
    }
    return mapping->start == start && mapping->size == size && mapping->permissions == permissions;
}

bool same_mappings(const std::vector<MappingInfo>& lhs, const std::vector<MappingInfo>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].start != rhs[i].start || lhs[i].size != rhs[i].size ||
            lhs[i].permissions != rhs[i].permissions) {
            return false;
        }
    }
    return true;
}

/// Verifies the documented structural invariants of the model: sorted
/// regions, page alignment, containment, disjointness, and the canonical
/// form (no two touching regions share permissions).
void check_model_invariants(const GuestMemoryModel& model, GuestAddress space_start,
                            std::uint64_t page_size, std::uint64_t page_count) {
    const auto snapshot = model.mappings();
    CHECK_EQ(snapshot.size(), model.mapping_count());

    const GuestAddress space_last = space_start + page_count * page_size - 1;
    std::optional<MappingInfo> previous;
    for (const MappingInfo& region : snapshot) {
        CHECK(region.size != 0);
        CHECK_EQ(region.size % page_size, std::uint64_t{0});
        CHECK_EQ(region.start % page_size, GuestAddress{0});
        CHECK(region.start >= space_start);
        CHECK(region.last_address() <= space_last);
        if (previous.has_value()) {
            CHECK(previous->last_address() < region.start);
            if (previous->last_address() + 1 == region.start) {
                CHECK(previous->permissions != region.permissions);
            }
        }
        previous = region;
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64) for the oracle comparison
// ---------------------------------------------------------------------------

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

private:
    std::uint64_t state_;
};

// ---------------------------------------------------------------------------
// Per-page oracle
// ---------------------------------------------------------------------------

class PageOracle {
public:
    static constexpr int kUnmapped = -1;

    explicit PageOracle(std::size_t page_count)
        : pages_(page_count, kUnmapped) {}

    bool is_mapped(std::size_t page) const { return pages_[page] != kUnmapped; }
    int permissions(std::size_t page) const { return pages_[page]; }

    bool any_mapped(std::size_t first, std::size_t count) const {
        for (std::size_t i = first; i < first + count; ++i) {
            if (pages_[i] != kUnmapped) {
                return true;
            }
        }
        return false;
    }

    bool fully_mapped(std::size_t first, std::size_t count) const {
        for (std::size_t i = first; i < first + count; ++i) {
            if (pages_[i] == kUnmapped) {
                return false;
            }
        }
        return true;
    }

    void map(std::size_t first, std::size_t count, int permissions) {
        for (std::size_t i = first; i < first + count; ++i) {
            pages_[i] = permissions;
        }
    }

    void unmap(std::size_t first, std::size_t count) {
        for (std::size_t i = first; i < first + count; ++i) {
            pages_[i] = kUnmapped;
        }
    }

    void protect(std::size_t first, std::size_t count, int permissions) {
        for (std::size_t i = first; i < first + count; ++i) {
            pages_[i] = permissions;
        }
    }

private:
    std::vector<int> pages_;
};

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

void test_create_valid_geometry() {
    auto model = GuestMemoryModel::create(0x10000, 0x10000, 0x400);
    REQUIRE(model.has_value());
    CHECK(model->valid());
    CHECK_EQ(model->address_space_start(), GuestAddress{0x10000});
    CHECK_EQ(model->address_space_size(), std::uint64_t{0x10000});
    CHECK_EQ(model->address_space_last(), GuestAddress{0x1FFFF});
    CHECK_EQ(model->page_size(), std::uint64_t{0x400});
    CHECK_EQ(model->mapping_count(), std::size_t{0});
    CHECK(model->mappings().empty());
}

void test_create_rejects_invalid_page_size() {
    CHECK(!GuestMemoryModel::create(0, 0x1000, 0).has_value());      // zero
    CHECK(!GuestMemoryModel::create(0, 0x1000, 3).has_value());      // not a power of two
    CHECK(!GuestMemoryModel::create(0, 0x1000, 0x600).has_value());  // not a power of two
    CHECK(!GuestMemoryModel::create(0, 0x1000, 12).has_value());     // not a power of two
    CHECK(GuestMemoryModel::create(0, 1u << 20, 1u << 20).has_value());
    // A power of two larger than the space fails the geometry checks.
    CHECK(!GuestMemoryModel::create(0, 1u << 20, 1u << 21).has_value());
}

void test_create_rejects_invalid_geometry() {
    CHECK(!GuestMemoryModel::create(0, 0, 0x1000).has_value());              // zero size
    CHECK(!GuestMemoryModel::create(1, 0x1000, 0x1000).has_value());         // misaligned start
    CHECK(!GuestMemoryModel::create(0, 0x800, 0x1000).has_value());          // misaligned size
    CHECK(!GuestMemoryModel::create(0xFFFF'FFFF'FFFF'FFFFULL, 2, 1).has_value());  // wraps past max
    CHECK(!GuestMemoryModel::create(0xFFFF'FFFF'FFFF'FFFEULL, 4, 2).has_value());  // last would be max+1
    // Geometry ending exactly at the top of the address range is legal.
    CHECK(GuestMemoryModel::create(0xFFFF'FFFF'FFFF'FFFEULL, 2, 2).has_value());
    CHECK(GuestMemoryModel::create(0, 0xFFFF'FFFF'FFFF'FFFFULL, 1).has_value());
}

void test_invalid_model_rejects_operations() {
    GuestMemoryModel model;
    CHECK(!model.valid());
    CHECK_EQ(model.page_size(), std::uint64_t{0});
    CHECK_EQ(model.map(0, 0x1000, Permission::read).error, MemoryError::invalid_address_space);
    CHECK_EQ(model.unmap(0, 0x1000).error, MemoryError::invalid_address_space);
    CHECK_EQ(model.protect(0, 0x1000, Permission::read).error,
             MemoryError::invalid_address_space);
    CHECK(!model.query(0).has_value());
    CHECK(model.mappings().empty());
}

// ---------------------------------------------------------------------------
// Map tests
// ---------------------------------------------------------------------------

void test_map_basic_query() {
    auto model = GuestMemoryModel::create(0x10000, 0x10000, 0x400);
    REQUIRE(model.has_value());
    CHECK(model->map(0x10000, 0x1000, Permission::read).ok());
    CHECK(same_region(model->query(0x10000), 0x10000, 0x1000, Permission::read));
    CHECK(same_region(model->query(0x10FFF), 0x10000, 0x1000, Permission::read));
    CHECK(!model->query(0xFFFF).has_value());   // below the region
    CHECK(!model->query(0x11000).has_value());  // above the region
    CHECK_EQ(model->mapping_count(), std::size_t{1});
}

void test_map_rejects_zero_size() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    const auto result = model->map(0x2000, 0, Permission::read);
    CHECK_EQ(result.error, MemoryError::zero_size);
    CHECK_EQ(model->mapping_count(), std::size_t{0});
}

void test_map_rejects_misalignment() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK_EQ(model->map(0x800, 0x1000, Permission::read).error, MemoryError::unaligned_address);
    CHECK_EQ(model->map(0x1000, 0x800, Permission::read).error, MemoryError::unaligned_size);
    CHECK_EQ(model->map(0x1001, 0, Permission::read).error, MemoryError::zero_size);
    CHECK_EQ(model->mapping_count(), std::size_t{0});
}

void test_map_rejects_unknown_permission_bits() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    const Permission unknown[] = {
        static_cast<Permission>(0x08),
        static_cast<Permission>(0x09),
        static_cast<Permission>(0x10 | 0x02),
        static_cast<Permission>(0xFF),
    };
    for (Permission mask : unknown) {
        CHECK_EQ(model->map(0, 0x1000, mask).error, MemoryError::invalid_permission);
    }
    CHECK_EQ(model->mapping_count(), std::size_t{0});
}

void test_map_rejects_write_plus_execute() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK_EQ(model->map(0, 0x1000, Permission::write | Permission::execute).error,
             MemoryError::write_execute_rejected);
    CHECK_EQ(
        model->map(0, 0x1000, Permission::read | Permission::write | Permission::execute).error,
        MemoryError::write_execute_rejected);
    CHECK_EQ(model->mapping_count(), std::size_t{0});
    // Write without execute and execute without write are both accepted.
    CHECK(model->map(0, 0x1000, Permission::read | Permission::write).ok());
    CHECK(model->map(0x1000, 0x1000, Permission::read | Permission::execute).ok());
}

void test_map_rejects_out_of_range() {
    auto model = GuestMemoryModel::create(0x10000, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    const auto below = model->map(0x9000, 0x1000, Permission::read);  // below the space
    CHECK_EQ(below.error, MemoryError::out_of_range);
    CHECK_EQ(below.detail, GuestAddress{0x9000});
    const auto above = model->map(0x20000, 0x1000, Permission::read);  // first page past the end
    CHECK_EQ(above.error, MemoryError::out_of_range);
    CHECK_EQ(above.detail, GuestAddress{0x20000});
    const auto straddling = model->map(0x1F000, 0x2000, Permission::read);  // runs past the end
    CHECK_EQ(straddling.error, MemoryError::out_of_range);
    CHECK_EQ(straddling.detail, GuestAddress{0x1F000});
    CHECK_EQ(model->mapping_count(), std::size_t{0});
}

void test_map_rejects_overflow() {
    auto model = GuestMemoryModel::create(0, 0xFFFF'FFFF'FFFF'FFFFULL, 1);
    REQUIRE(model.has_value());
    const auto result = model->map(0xFFFF'FFFF'FFFF'FFFFULL, 2, Permission::read);
    CHECK_EQ(result.error, MemoryError::address_overflow);
    CHECK_EQ(result.detail, GuestAddress{0xFFFF'FFFF'FFFF'FFFFULL});
    CHECK_EQ(model->mapping_count(), std::size_t{0});
}

void test_map_rejects_overlap_in_all_shapes() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0x4000, 0x2000, Permission::read).ok());  // [0x4000, 0x6000)

    struct Case {
        GuestAddress address;
        std::uint64_t size;
        const char* shape;
    };
    const Case cases[] = {
        {0x4000, 0x2000, "exact"},
        {0x3000, 0x2000, "straddles start"},
        {0x4000, 0x1000, "inner low"},
        {0x5000, 0x1000, "inner high"},
        {0x5000, 0x2000, "straddles end"},
        {0x2000, 0x4000, "encloses"},
    };
    for (const Case& entry : cases) {
        const auto result = model->map(entry.address, entry.size, Permission::read);
        CHECK_EQ(result.error, MemoryError::overlapping_map);
        CHECK_EQ(result.detail, GuestAddress{0x4000});
    }
    CHECK_EQ(model->mapping_count(), std::size_t{1});
}

void test_map_adjacent_merge() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0x1000, 0x1000, Permission::read).ok());
    CHECK(model->map(0, 0x1000, Permission::read).ok());  // left merge
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(model->map(0x2000, 0x1000, Permission::read).ok());  // right merge
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(same_region(model->query(0x2000), 0, 0x3000, Permission::read));

    // Adjacent but with different permissions: no merge.
    CHECK(model->map(0x3000, 0x1000, Permission::read | Permission::write).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{2});
    CHECK(same_region(model->query(0x2FFF), 0, 0x3000, Permission::read));
    CHECK(
        same_region(model->query(0x3000), 0x3000, 0x1000, Permission::read | Permission::write));
}

// ---------------------------------------------------------------------------
// Unmap tests
// ---------------------------------------------------------------------------

void test_unmap_whole_region() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0x2000, 0x1000, Permission::read).ok());
    CHECK(model->unmap(0x2000, 0x1000).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{0});
    CHECK(!model->query(0x2000).has_value());
}

void test_unmap_middle_splits_region() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x4000, Permission::read | Permission::write).ok());
    CHECK(model->unmap(0x1000, 0x1000).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{2});
    CHECK(same_region(model->query(0), 0, 0x1000, Permission::read | Permission::write));
    CHECK(
        same_region(model->query(0x3FFF), 0x2000, 0x2000, Permission::read | Permission::write));
    CHECK(!model->query(0x1000).has_value());
    CHECK(!model->query(0x1FFF).has_value());
    check_model_invariants(*model, 0, 0x1000, 16);
}

void test_unmap_prefix_and_suffix() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0x1000, 0x3000, Permission::read).ok());
    CHECK(model->unmap(0x1000, 0x1000).ok());  // prefix
    CHECK(same_region(model->query(0x2000), 0x2000, 0x2000, Permission::read));
    CHECK(model->unmap(0x3000, 0x1000).ok());  // suffix
    CHECK(same_region(model->query(0x2000), 0x2000, 0x1000, Permission::read));
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    check_model_invariants(*model, 0, 0x1000, 16);
}

void test_unmap_across_multiple_regions() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x800);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x1000, Permission::read).ok());
    CHECK(model->map(0x1000, 0x1000, Permission::write).ok());
    CHECK(model->map(0x2000, 0x1000, Permission::execute).ok());
    CHECK(model->unmap(0x800, 0x1800).ok());  // tail of the first, all of the second
    CHECK_EQ(model->mapping_count(), std::size_t{2});
    CHECK(same_region(model->query(0), 0, 0x800, Permission::read));
    CHECK(same_region(model->query(0x2000), 0x2000, 0x1000, Permission::execute));
    CHECK(model->map(0x800, 0x1800, Permission::none).ok());  // refill the hole
    CHECK_EQ(model->mapping_count(), std::size_t{3});
    CHECK(model->unmap(0, 0x3000).ok());  // across three touching regions
    CHECK_EQ(model->mapping_count(), std::size_t{0});
    check_model_invariants(*model, 0, 0x800, 32);
}

void test_unmap_requires_complete_coverage() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x800);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x1000, Permission::read).ok());
    CHECK(model->map(0x2000, 0x1000, Permission::read).ok());
    const auto before = model->mappings();

    const auto result = model->unmap(0, 0x3000);  // hole at [0x1000, 0x2000)
    CHECK_EQ(result.error, MemoryError::not_mapped);
    CHECK_EQ(result.detail, GuestAddress{0x1000});
    CHECK(same_mappings(model->mappings(), before));

    // Hole in the middle of the requested range.
    const auto middle = model->unmap(0x800, 0x2000);
    CHECK_EQ(middle.error, MemoryError::not_mapped);
    CHECK_EQ(middle.detail, GuestAddress{0x1000});
    CHECK(same_mappings(model->mappings(), before));

    // Range extends past the last mapping.
    const auto past_end = model->unmap(0x2000, 0x2000);
    CHECK_EQ(past_end.error, MemoryError::not_mapped);
    CHECK_EQ(past_end.detail, GuestAddress{0x3000});
    CHECK(same_mappings(model->mappings(), before));

    // Completely unmapped area.
    CHECK_EQ(model->unmap(0x8000, 0x1000).error, MemoryError::not_mapped);
    CHECK(same_mappings(model->mappings(), before));
}

void test_unmap_rejects_invalid_ranges() {
    auto model = GuestMemoryModel::create(0x10000, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0x10000, 0x1000, Permission::read).ok());
    const auto before = model->mappings();

    CHECK_EQ(model->unmap(0x10000, 0).error, MemoryError::zero_size);
    CHECK_EQ(model->unmap(0x10001, 0x1000).error, MemoryError::unaligned_address);
    CHECK_EQ(model->unmap(0x10000, 0x800).error, MemoryError::unaligned_size);
    CHECK_EQ(model->unmap(0x9000, 0x1000).error, MemoryError::out_of_range);
    CHECK_EQ(model->unmap(0x1F000, 0x2000).error, MemoryError::out_of_range);
    CHECK(same_mappings(model->mappings(), before));
}

// ---------------------------------------------------------------------------
// Protect tests
// ---------------------------------------------------------------------------

void test_protect_sets_and_merges() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x1000, Permission::read).ok());
    CHECK(model->map(0x1000, 0x1000, Permission::read | Permission::write).ok());
    CHECK(model->map(0x2000, 0x1000, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{3});

    CHECK(model->protect(0x1000, 0x1000, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(same_region(model->query(0), 0, 0x3000, Permission::read));

    // Split the merged region again by elevating the middle.
    CHECK(model->protect(0x1000, 0x1000, Permission::read | Permission::write).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{3});
    CHECK(same_region(model->query(0xFFF), 0, 0x1000, Permission::read));
    CHECK(
        same_region(model->query(0x1000), 0x1000, 0x1000, Permission::read | Permission::write));
    CHECK(same_region(model->query(0x2000), 0x2000, 0x1000, Permission::read));
    check_model_invariants(*model, 0, 0x1000, 16);
}

void test_protect_partial_edges_split_and_merge_with_neighbors() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x2000, Permission::read).ok());                          // [0, 0x2000)
    CHECK(model->map(0x2000, 0x2000, Permission::read | Permission::write).ok());  // [0x2000, 0x4000)

    // Protect the second half of the first region and the first page of the
    // second: the read piece extends right, the read-write piece shrinks.
    CHECK(model->protect(0x1000, 0x2000, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{2});
    CHECK(same_region(model->query(0), 0, 0x3000, Permission::read));
    CHECK(
        same_region(model->query(0x3000), 0x3000, 0x1000, Permission::read | Permission::write));

    // Elevate a suffix that touches the following region with equal result.
    CHECK(model->protect(0x2000, 0x2000, Permission::read | Permission::write).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{2});
    CHECK(same_region(model->query(0x1FFF), 0, 0x2000, Permission::read));
    CHECK(
        same_region(model->query(0x2000), 0x2000, 0x2000, Permission::read | Permission::write));
    check_model_invariants(*model, 0, 0x1000, 16);
}

void test_protect_requires_complete_coverage() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x4000, Permission::read).ok());
    CHECK(model->unmap(0x1000, 0x1000).ok());  // hole at [0x1000, 0x2000)
    const auto before = model->mappings();

    const auto result = model->protect(0, 0x4000, Permission::read | Permission::write);
    CHECK_EQ(result.error, MemoryError::not_mapped);
    CHECK_EQ(result.detail, GuestAddress{0x1000});
    CHECK(same_mappings(model->mappings(), before));
}

void test_protect_rejects_invalid_arguments() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x1000, Permission::read).ok());
    const auto before = model->mappings();

    CHECK_EQ(model->protect(0, 0x1000, Permission::write | Permission::execute).error,
             MemoryError::write_execute_rejected);
    CHECK_EQ(model->protect(0, 0x1000, static_cast<Permission>(0x20)).error,
             MemoryError::invalid_permission);
    CHECK_EQ(model->protect(0, 0, Permission::read).error, MemoryError::zero_size);
    CHECK_EQ(model->protect(0x100, 0x1000, Permission::read).error, MemoryError::unaligned_address);
    CHECK_EQ(model->protect(0, 0x100, Permission::read).error, MemoryError::unaligned_size);
    CHECK_EQ(model->protect(0xF000, 0x2000, Permission::read).error, MemoryError::out_of_range);
    CHECK_EQ(model->protect(0x2000, 0x1000, Permission::read).error, MemoryError::not_mapped);
    CHECK(same_mappings(model->mappings(), before));
}

// ---------------------------------------------------------------------------
// No-access and query semantics
// ---------------------------------------------------------------------------

void test_mapped_no_access_distinct_from_unmapped() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x800);
    REQUIRE(model.has_value());
    CHECK(model->map(0x1000, 0x1000, Permission::none).ok());

    const auto guard = model->query(0x1800);
    CHECK(guard.has_value());
    CHECK_EQ(guard->permissions, Permission::none);
    CHECK_EQ(guard->start, GuestAddress{0x1000});
    CHECK_EQ(guard->size, std::uint64_t{0x1000});

    // The unmapped neighbor must be distinguishable from the guard region.
    CHECK(!model->query(0xFFF).has_value());
    CHECK(!model->query(0x2000).has_value());

    // No-access regions participate in splitting, merging, and protection.
    CHECK(model->map(0, 0x1000, Permission::none).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});  // merged with the guard
    CHECK(model->map(0x2000, 0x1000, Permission::none).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(model->protect(0, 0x3000, Permission::none).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(model->protect(0x1000, 0x1000, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{3});
    CHECK(model->unmap(0x800, 0x800).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{3});
    check_model_invariants(*model, 0, 0x800, 32);
}

// ---------------------------------------------------------------------------
// Guest page size independence
// ---------------------------------------------------------------------------

void test_page_size_one_allows_byte_granularity() {
    auto model = GuestMemoryModel::create(0x10, 0x100, 1);
    REQUIRE(model.has_value());
    CHECK(model->map(0x11, 1, Permission::read).ok());
    CHECK(model->map(0x12, 2, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});  // merged
    CHECK(same_region(model->query(0x13), 0x11, 3, Permission::read));
    CHECK(model->unmap(0x12, 1).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{2});
    check_model_invariants(*model, 0x10, 1, 0x100);
}

void test_large_power_of_two_page() {
    constexpr std::uint64_t page = std::uint64_t{1} << 32;
    auto model = GuestMemoryModel::create(0, page * 4, page);
    REQUIRE(model.has_value());
    CHECK_EQ(model->page_size(), page);
    CHECK(model->map(page, page, Permission::execute).ok());
    CHECK_EQ(model->map(page, 1, Permission::execute).error, MemoryError::unaligned_size);
    CHECK_EQ(model->map(page + 1, page, Permission::execute).error, MemoryError::unaligned_address);
    CHECK(model->map(0, page, Permission::execute).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(same_region(model->query(page - 1), 0, page * 2, Permission::execute));
    check_model_invariants(*model, 0, page, 4);
}

// ---------------------------------------------------------------------------
// Boundaries near the top of the 64-bit range
// ---------------------------------------------------------------------------

void test_boundaries_near_uint64_max() {
    constexpr GuestAddress kStart = 0xFFFF'FFFF'FFFF'E000ULL;  // two pages below the top
    constexpr std::uint64_t kPage = 0x1000;
    auto model = GuestMemoryModel::create(kStart, 2 * kPage, kPage);
    REQUIRE(model.has_value());
    CHECK_EQ(model->address_space_last(), GuestAddress{0xFFFF'FFFF'FFFF'FFFFULL});

    CHECK(model->map(kStart + kPage, kPage, Permission::read).ok());
    CHECK(same_region(model->query(0xFFFF'FFFF'FFFF'FFFFULL), kStart + kPage, kPage,
                      Permission::read));
    CHECK(!model->query(kStart + kPage - 1).has_value());

    // Mapping the lower page with equal permissions must merge across the
    // top page boundary without any wrapping arithmetic.
    CHECK(model->map(kStart, kPage, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(same_region(model->query(kStart), kStart, 2 * kPage, Permission::read));

    // Ranges that would run past the top are rejected as overflow, not
    // out-of-range, and leave the state untouched.
    const auto before = model->mappings();
    CHECK_EQ(model->map(0xFFFF'FFFF'FFFF'F000ULL, 2 * kPage, Permission::read).error,
             MemoryError::address_overflow);
    CHECK_EQ(model->unmap(0xFFFF'FFFF'FFFF'F000ULL, 2 * kPage).error, MemoryError::address_overflow);
    CHECK_EQ(model->protect(0xFFFF'FFFF'FFFF'F000ULL, 2 * kPage, Permission::read).error,
             MemoryError::address_overflow);
    CHECK(same_mappings(model->mappings(), before));

    // Splits at the very top of the range.
    CHECK(model->unmap(kStart + kPage, kPage).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    CHECK(same_region(model->query(kStart), kStart, kPage, Permission::read));
    CHECK(!model->query(0xFFFF'FFFF'FFFF'FFFFULL).has_value());

    // Unmap the final page of the address space.
    CHECK(model->map(kStart + kPage, kPage, Permission::read | Permission::execute).ok());
    CHECK_EQ(model->unmap(0xFFFF'FFFF'FFFF'F000ULL, 1).error, MemoryError::unaligned_size);
    CHECK(model->unmap(0xFFFF'FFFF'FFFF'F000ULL, kPage).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    check_model_invariants(*model, kStart, kPage, 2);
}

void test_single_byte_space_at_uint64_max() {
    auto model = GuestMemoryModel::create(0xFFFF'FFFF'FFFF'FFFFULL, 1, 1);
    REQUIRE(model.has_value());
    CHECK(model->map(0xFFFF'FFFF'FFFF'FFFFULL, 1, Permission::read).ok());
    CHECK(same_region(model->query(0xFFFF'FFFF'FFFF'FFFFULL), 0xFFFF'FFFF'FFFF'FFFFULL, 1,
                      Permission::read));
    CHECK(model->protect(0xFFFF'FFFF'FFFF'FFFFULL, 1, Permission::read | Permission::write).ok());
    CHECK(model->unmap(0xFFFF'FFFF'FFFF'FFFFULL, 1).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{0});
}

void test_full_address_range_with_byte_pages() {
    constexpr GuestAddress kMax = 0xFFFF'FFFF'FFFF'FFFFULL;
    auto model = GuestMemoryModel::create(0, kMax, 1);  // everything except the top byte
    REQUIRE(model.has_value());
    CHECK(model->map(0, kMax, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    const auto whole_space = model->mappings();
    CHECK_EQ(whole_space[0].last_address(), GuestAddress{kMax - 1});
    CHECK_EQ(whole_space[0].size, std::uint64_t{kMax});

    CHECK(model->unmap(0xFF00, 0xFF).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{2});
    CHECK(model->map(0xFF00, 0xFF, Permission::write).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{3});
    CHECK(model->protect(0, 0xFF01, Permission::read).ok());  // across the seam
    CHECK_EQ(model->mapping_count(), std::size_t{3});
    CHECK(same_region(model->query(0xFF01), 0xFF01, 0xFE, Permission::write));
    CHECK(model->protect(0xFF01, 0xFE, Permission::read).ok());
    CHECK_EQ(model->mapping_count(), std::size_t{1});
    // The top byte was never part of the space.
    CHECK_EQ(model->map(kMax, 1, Permission::read).error, MemoryError::out_of_range);
    check_model_invariants(*model, 0, 1, kMax);
}

// ---------------------------------------------------------------------------
// Transactional failure behavior
// ---------------------------------------------------------------------------

void test_failures_leave_state_unchanged() {
    auto model = GuestMemoryModel::create(0, 0x10000, 0x1000);
    REQUIRE(model.has_value());
    CHECK(model->map(0, 0x1000, Permission::read).ok());
    CHECK(model->map(0x1000, 0x1000, Permission::read | Permission::write).ok());
    CHECK(model->map(0x3000, 0x1000, Permission::execute).ok());
    const auto before = model->mappings();

    // Overlapping map.
    CHECK_EQ(model->map(0, 0x2000, Permission::read).error, MemoryError::overlapping_map);
    CHECK(same_mappings(model->mappings(), before));
    // Unmap over a hole.
    CHECK_EQ(model->unmap(0, 0x4000).error, MemoryError::not_mapped);
    CHECK(same_mappings(model->mappings(), before));
    // Protect over a hole.
    CHECK_EQ(model->protect(0x1000, 0x3000, Permission::read).error, MemoryError::not_mapped);
    CHECK(same_mappings(model->mappings(), before));
    // Invalid permissions.
    CHECK_EQ(model->protect(0, 0x4000, Permission::write | Permission::execute).error,
             MemoryError::write_execute_rejected);
    CHECK(same_mappings(model->mappings(), before));
    // Geometry failures.
    CHECK_EQ(model->unmap(0, 0).error, MemoryError::zero_size);
    CHECK_EQ(model->map(0x1'0000'0000'0000ULL, 0x1000, Permission::read).error,
             MemoryError::out_of_range);
    CHECK(same_mappings(model->mappings(), before));
    check_model_invariants(*model, 0, 0x1000, 16);
}

// ---------------------------------------------------------------------------
// Differential comparison against the per-page oracle
// ---------------------------------------------------------------------------

constexpr int kValidPerms[] = {
    0x00,  // none
    0x01,  // r
    0x02,  // w
    0x04,  // x
    0x03,  // rw
    0x05,  // rx
};

constexpr int kInvalidPerms[] = {
    0x08,  // unknown bit
    0x07,  // rwx: contains write+execute
    0x06,  // wx
    0x12,  // unknown bit | write
};

MemoryError classify_permission_mask(int raw) {
    if ((raw & ~0x07) != 0) {
        return MemoryError::invalid_permission;
    }
    if ((raw & 0x06) == 0x06) {
        return MemoryError::write_execute_rejected;
    }
    return MemoryError::success;
}

/// Runs a deterministic random operation sequence against both the model and
/// a per-page oracle, verifying the error result, the full page state, the
/// structural invariants, and step-by-step determinism after every operation.
void run_oracle_round(GuestAddress space_start, std::uint64_t page_size, std::size_t page_count,
                      std::uint64_t seed, int iterations) {
    const std::uint64_t space_size = static_cast<std::uint64_t>(page_count) * page_size;
    auto model_a = GuestMemoryModel::create(space_start, space_size, page_size);
    auto model_b = GuestMemoryModel::create(space_start, space_size, page_size);
    REQUIRE(model_a.has_value());
    REQUIRE(model_b.has_value());

    PageOracle oracle(page_count);
    SplitMix64 rng(seed);

    for (int step = 0; step < iterations; ++step) {
        const int kind = static_cast<int>(rng.next() % 100);
        const std::size_t first = static_cast<std::size_t>(rng.next() % page_count);
        const std::size_t count = 1 + static_cast<std::size_t>(rng.next() % 8);
        const std::size_t len = count > page_count - first ? page_count - first : count;
        const bool invalid_perms = kind < 90 && (rng.next() % 8) == 0;
        const int raw_perms =
            invalid_perms ? kInvalidPerms[rng.next() % 4] : kValidPerms[rng.next() % 6];
        const Permission perms = static_cast<Permission>(raw_perms);
        const GuestAddress address =
            space_start + static_cast<GuestAddress>(first) * page_size;
        const std::uint64_t size = static_cast<std::uint64_t>(len) * page_size;

        // Compute the expected outcome from the oracle.
        std::optional<MemoryError> expected;
        if (kind < 40) {  // map
            const MemoryError permission_error = classify_permission_mask(raw_perms);
            if (permission_error != MemoryError::success) {
                expected = permission_error;
            } else if (oracle.any_mapped(first, len)) {
                expected = MemoryError::overlapping_map;
            }
        } else if (kind < 65) {  // unmap
            if (!oracle.fully_mapped(first, len)) {
                expected = MemoryError::not_mapped;
            }
        } else {  // protect
            const MemoryError permission_error = classify_permission_mask(raw_perms);
            if (permission_error != MemoryError::success) {
                expected = permission_error;
            } else if (!oracle.fully_mapped(first, len)) {
                expected = MemoryError::not_mapped;
            }
        }

        const auto before = model_a->mappings();
        armigo::memory::MemoryResult result{MemoryError::success, 0};
        if (kind < 40) {
            result = model_a->map(address, size, perms);
        } else if (kind < 65) {
            result = model_a->unmap(address, size);
        } else {
            result = model_a->protect(address, size, perms);
        }

        if (expected.has_value()) {
            CHECK_EQ(result.error, *expected);
            CHECK(same_mappings(model_a->mappings(), before));
        } else {
            CHECK(result.ok());
            if (kind < 40) {
                oracle.map(first, len, raw_perms);
            } else if (kind < 65) {
                oracle.unmap(first, len);
            } else {
                oracle.protect(first, len, raw_perms);
            }
        }

        // Replay the identical operation on the second model; both must agree
        // at every step.
        if (kind < 40) {
            model_b->map(address, size, perms);
        } else if (kind < 65) {
            model_b->unmap(address, size);
        } else {
            model_b->protect(address, size, perms);
        }
        CHECK(same_mappings(model_a->mappings(), model_b->mappings()));

        // Compare the complete page state and structural invariants.
        for (std::size_t page = 0; page < page_count; ++page) {
            const GuestAddress probe =
                space_start + static_cast<GuestAddress>(page) * page_size;
            const auto mapping = model_a->query(probe);
            if (oracle.is_mapped(page)) {
                CHECK(mapping.has_value());
                if (mapping.has_value()) {
                    CHECK_EQ(mapping->permissions, static_cast<Permission>(oracle.permissions(page)));
                    CHECK(mapping->start <= probe);
                    CHECK(probe <= mapping->last_address());
                }
            } else {
                CHECK(!mapping.has_value());
            }
        }
        check_model_invariants(*model_a, space_start, page_size,
                               static_cast<std::uint64_t>(page_count));
    }
}

void test_oracle_comparison() {
    // Sixty-four pages of 1 KiB at a nonzero space start.
    run_oracle_round(0x10000, 0x400, 64, 0x0B0B5EED0001ULL, 3000);
    // Byte pages in a small space.
    run_oracle_round(0, 1, 256, 0x0B0B5EED0002ULL, 3000);
    // A larger page count for wider split/merge coverage.
    run_oracle_round(0, 0x100, 96, 0x0B0B5EED0003ULL, 2000);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

void test_error_diagnostics_text() {
    CHECK(std::string(armigo::memory::to_string(MemoryError::success)) == "success");
    CHECK(std::string(armigo::memory::to_string(MemoryError::overlapping_map))
              .find("overlap") != std::string::npos);
    CHECK(std::string(armigo::memory::to_string(MemoryError::not_mapped))
              .find("not completely mapped") != std::string::npos);
    CHECK(std::string(armigo::memory::to_string(MemoryError::write_execute_rejected))
              .find("write and execute") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Test registry
// ---------------------------------------------------------------------------

struct TestCase {
    const char* name;
    void (*fn)();
};

const TestCase kTests[] = {
    {"create_valid_geometry", test_create_valid_geometry},
    {"create_rejects_invalid_page_size", test_create_rejects_invalid_page_size},
    {"create_rejects_invalid_geometry", test_create_rejects_invalid_geometry},
    {"invalid_model_rejects_operations", test_invalid_model_rejects_operations},
    {"map_basic_query", test_map_basic_query},
    {"map_rejects_zero_size", test_map_rejects_zero_size},
    {"map_rejects_misalignment", test_map_rejects_misalignment},
    {"map_rejects_unknown_permission_bits", test_map_rejects_unknown_permission_bits},
    {"map_rejects_write_plus_execute", test_map_rejects_write_plus_execute},
    {"map_rejects_out_of_range", test_map_rejects_out_of_range},
    {"map_rejects_overflow", test_map_rejects_overflow},
    {"map_rejects_overlap_in_all_shapes", test_map_rejects_overlap_in_all_shapes},
    {"map_adjacent_merge", test_map_adjacent_merge},
    {"unmap_whole_region", test_unmap_whole_region},
    {"unmap_middle_splits_region", test_unmap_middle_splits_region},
    {"unmap_prefix_and_suffix", test_unmap_prefix_and_suffix},
    {"unmap_across_multiple_regions", test_unmap_across_multiple_regions},
    {"unmap_requires_complete_coverage", test_unmap_requires_complete_coverage},
    {"unmap_rejects_invalid_ranges", test_unmap_rejects_invalid_ranges},
    {"protect_sets_and_merges", test_protect_sets_and_merges},
    {"protect_partial_edges_split_and_merge_with_neighbors",
     test_protect_partial_edges_split_and_merge_with_neighbors},
    {"protect_requires_complete_coverage", test_protect_requires_complete_coverage},
    {"protect_rejects_invalid_arguments", test_protect_rejects_invalid_arguments},
    {"mapped_no_access_distinct_from_unmapped", test_mapped_no_access_distinct_from_unmapped},
    {"page_size_one_allows_byte_granularity", test_page_size_one_allows_byte_granularity},
    {"large_power_of_two_page", test_large_power_of_two_page},
    {"boundaries_near_uint64_max", test_boundaries_near_uint64_max},
    {"single_byte_space_at_uint64_max", test_single_byte_space_at_uint64_max},
    {"full_address_range_with_byte_pages", test_full_address_range_with_byte_pages},
    {"failures_leave_state_unchanged", test_failures_leave_state_unchanged},
    {"oracle_comparison", test_oracle_comparison},
    {"error_diagnostics_text", test_error_diagnostics_text},
};

}  // namespace

int main() {
    int failed_tests = 0;
    constexpr int kTestCount = static_cast<int>(sizeof(kTests) / sizeof(kTests[0]));
    for (const TestCase& test : kTests) {
        const int before = g_checks_failed;
        test.fn();
        const bool passed = g_checks_failed == before;
        std::printf("%-56s %s\n", test.name, passed ? "PASS" : "FAIL");
        if (!passed) {
            ++failed_tests;
        }
    }
    std::printf("\n%d checks, %d failed, %d/%d tests passed\n", g_checks_total, g_checks_failed,
                kTestCount - failed_tests, kTestCount);
    return g_checks_failed == 0 ? 0 : 1;
}
