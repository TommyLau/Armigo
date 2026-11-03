// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/upload.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <utility>
#include <variant>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

struct Backing {
    static constexpr std::uint64_t guest_base = 0x8000;
    std::array<std::byte, 0x1000> bytes{};
    unsigned writes = 0;
    bool fail_write = false;

    static bool write(void* raw, std::uint64_t address,
                      std::span<const std::byte> input) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        ++self.writes;
        if (self.fail_write || address < guest_base) {
            return false;
        }
        const std::uint64_t offset = address - guest_base;
        if (offset > self.bytes.size() ||
            input.size() > self.bytes.size() - offset) {
            return false;
        }
        std::memcpy(self.bytes.data() + offset, input.data(), input.size());
        return true;
    }
};

GpuAddressSpace make_memory(Backing& backing) {
    const Mapping mapping{0x1000, 0x1000, 0, Backing::guest_base, 0x1000, 0x1000};
    AddressSpaceResult result =
        make_address_space({&mapping, 1}, {nullptr, Backing::write, &backing});
    CHECK(std::holds_alternative<GpuAddressSpace>(result));
    return std::get<GpuAddressSpace>(std::move(result));
}

void configure(InlineUpload& upload, std::uint32_t length,
               std::uint32_t destination = 0x1080) {
    CHECK(upload.set_register(0, length).accepted());
    CHECK(upload.set_register(1, 1).accepted());
    CHECK(upload.set_register(2, 0).accepted());
    CHECK(upload.set_register(3, destination).accepted());
    CHECK(upload.set_register(4, 0).accepted());
    CHECK(upload.set_register(10, 0).accepted());
    CHECK(upload.set_register(11, 0).accepted());
}

void test_exact_completion_and_partial_word() {
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    InlineUpload upload;
    configure(upload, 7);
    CHECK(upload.launch(0x1011).status == UploadStatus::Pending);
    CHECK(upload.blocked());

    CHECK(upload.push_word(0x44332211, memory).status == UploadStatus::Pending);
    CHECK(upload.blocked());
    CHECK(backing.writes == 0);

    CHECK(upload.push_word(0xddccbbaa, memory).status == UploadStatus::Complete);
    CHECK(!upload.blocked());
    CHECK(backing.writes == 1);
    const std::array expected{std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
                              std::byte{0x44}, std::byte{0xaa}, std::byte{0xbb},
                              std::byte{0xcc}};
    CHECK(std::equal(expected.begin(), expected.end(),
                     backing.bytes.begin() + 0x80));
    CHECK(backing.bytes[0x87] == std::byte{0});
    CHECK(upload.push_word(0, memory).status == UploadStatus::DataWithoutLaunch);
}

void test_snapshot_and_stale_pitch_geometry() {
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    InlineUpload upload;
    configure(upload, 4, 0x1100);
    for (std::uint32_t index = 5; index <= 9; ++index) {
        CHECK(upload.set_register(index, 0xabcdef00u + index).accepted());
    }
    CHECK(upload.launch(0x11).accepted());
    CHECK(upload.set_register(0, 8).accepted());
    CHECK(upload.set_register(3, 0x1200).accepted());
    CHECK(upload.push_word(0x78563412, memory).status == UploadStatus::Complete);
    CHECK(backing.bytes[0x100] == std::byte{0x12});
    CHECK(backing.bytes[0x200] == std::byte{0});
}

void test_launch_and_sequence_failures() {
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    InlineUpload upload;
    CHECK(upload.push_word(1, memory).status == UploadStatus::DataWithoutLaunch);
    CHECK(upload.set_register(12, 1).status == UploadStatus::InvalidRegister);

    configure(upload, 4);
    CHECK(upload.launch(0x1010).status == UploadStatus::Unsupported);
    CHECK(upload.launch(0x1013).status == UploadStatus::Unsupported);
    CHECK(upload.launch(0x1021).status == UploadStatus::Unsupported);
    CHECK(upload.launch(0x1111).status == UploadStatus::Unsupported);
    CHECK(upload.launch(0x1051).status == UploadStatus::Unsupported);
    CHECK(upload.launch(0x1011).status == UploadStatus::Pending);
    CHECK(upload.launch(0x1011).status == UploadStatus::LaunchWhilePending);
    CHECK(upload.push_word(0x12345678, memory).status == UploadStatus::Complete);

    InlineUpload dimensions;
    configure(dimensions, 4);
    CHECK(dimensions.set_register(1, 2).accepted());
    CHECK(dimensions.launch(0x1011).status == UploadStatus::Unsupported);
    CHECK(dimensions.set_register(1, 1).accepted());
    CHECK(dimensions.set_register(10, 1).accepted());
    CHECK(dimensions.launch(0x1011).status == UploadStatus::Unsupported);

    InlineUpload address_range;
    configure(address_range, 4);
    CHECK(address_range.set_register(2, 0xff).accepted());
    CHECK(address_range.set_register(3, 0xfffffffe).accepted());
    CHECK(address_range.launch(0x1011).status == UploadStatus::InvalidLaunch);
}

void test_memory_failure_stays_blocked() {
    Backing backing;
    backing.fail_write = true;
    GpuAddressSpace memory = make_memory(backing);
    InlineUpload upload;
    configure(upload, 4);
    CHECK(upload.launch(0x1011).accepted());
    CHECK(upload.push_word(0x12345678, memory).status == UploadStatus::MemoryFault);
    CHECK(upload.blocked());
    CHECK(upload.status() == UploadStatus::MemoryFault);
    CHECK(upload.set_register(0, 8).status == UploadStatus::MemoryFault);
    CHECK(upload.push_word(0, memory).status == UploadStatus::MemoryFault);
}

void test_zero_length_launch_is_a_noop() {
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    InlineUpload upload;
    configure(upload, 0);
    CHECK(upload.set_register(2, 0xffffffff).accepted());
    CHECK(upload.set_register(3, 0xffffffff).accepted());
    CHECK(upload.set_register(10, 7).accepted());
    CHECK(upload.set_register(11, 9).accepted());
    CHECK(upload.launch(0x1011).status == UploadStatus::Complete);
    CHECK(!upload.blocked());
    CHECK(backing.writes == 0);
    CHECK(upload.push_word(1, memory).status == UploadStatus::DataWithoutLaunch);
    CHECK(backing.writes == 0);
}

}  // namespace

int main() {
    test_exact_completion_and_partial_word();
    test_snapshot_and_stale_pitch_geometry();
    test_launch_and_sequence_failures();
    test_memory_failure_stays_blocked();
    test_zero_length_launch_is_a_noop();
    return failures == 0 ? 0 : 1;
}
