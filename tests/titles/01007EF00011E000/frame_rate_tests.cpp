// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/titles/01007EF00011E000/frame_rate.h>
#include <armigo/horizon/ipc.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

using armigo::titles::title_01007ef00011e000::FrameRateLayout;
using armigo::titles::title_01007ef00011e000::update_frame_rate;
using armigo::horizon::GuestBuffers;

unsigned failures = 0;

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #condition);                    \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

struct Memory {
    static constexpr std::uint64_t origin = 0x1000;
    std::array<std::byte, 0x800> bytes{};
    struct Access {
        bool write;
        std::uint64_t address;
        std::vector<std::byte> bytes;
        bool operator==(const Access&) const = default;
    };
    std::vector<Access> accesses;
    std::uint64_t failed_read = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t failed_write = std::numeric_limits<std::uint64_t>::max();

    static bool read(void* context, std::uint64_t address,
                     std::span<std::byte> destination) noexcept {
        auto& memory = *static_cast<Memory*>(context);
        memory.accesses.push_back({false, address, std::vector<std::byte>(destination.size())});
        if (address == memory.failed_read || address < origin ||
            destination.size() > memory.bytes.size() ||
            address - origin > memory.bytes.size() - destination.size()) {
            return false;
        }
        std::memcpy(destination.data(), memory.bytes.data() + (address - origin),
                    destination.size());
        return true;
    }

    static bool write(void* context, std::uint64_t address,
                      std::span<const std::byte> source) noexcept {
        auto& memory = *static_cast<Memory*>(context);
        memory.accesses.push_back({true, address, {source.begin(), source.end()}});
        if (address == memory.failed_write || address < origin ||
            source.size() > memory.bytes.size() ||
            address - origin > memory.bytes.size() - source.size()) {
            return false;
        }
        std::memcpy(memory.bytes.data() + (address - origin), source.data(), source.size());
        return true;
    }

    template <typename T> void store(std::uint64_t address, T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::memcpy(bytes.data() + (address - origin), &value, sizeof(value));
    }

    template <typename T> T load(std::uint64_t address) const {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        std::memcpy(&value, bytes.data() + (address - origin), sizeof(value));
        return value;
    }
};

constexpr std::uint64_t module_base = 0x1000;
constexpr std::uint64_t vfr_object = 0x1200;
constexpr std::uint64_t framework = 0x1500;
constexpr FrameRateLayout layout{
    .framework_pointer = 0x40,
    .framework_period = 0x28,
    .override_flag = 0x03,
    .ratio = 0x18,
    .interval = 0x1c,
    .base_interval = 0x30,
};

struct Fixture {
    Memory memory;
    GuestBuffers buffers{Memory::read, Memory::write, &memory};

    Fixture() {
        memory.store(module_base + layout.framework_pointer, framework);
        memory.store(framework + layout.framework_period, std::int64_t{640'000});
        memory.store(vfr_object + layout.override_flag, std::uint8_t{0});
        memory.store(vfr_object + layout.ratio, -7.0f);
        memory.store(vfr_object + layout.interval, std::uint32_t{99});
        memory.store(vfr_object + layout.base_interval, std::uint32_t{2});
    }
};

void test_measured_ratios() {
    Fixture fixture;
    CHECK(update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == 1.0f);
    CHECK(fixture.memory.load<std::uint32_t>(vfr_object + layout.interval) == 2);

    fixture.memory.store(framework + layout.framework_period, std::int64_t{320'000});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 1, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == 0.5f);
    CHECK(fixture.memory.load<std::uint32_t>(vfr_object + layout.interval) == 1);
}

void test_clamps() {
    Fixture fixture;
    fixture.memory.store(framework + layout.framework_period, std::int64_t{1});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == 0.1f);

    fixture.memory.store(framework + layout.framework_period, std::int64_t{1'920'000});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == 1.5f);
}

void test_override_and_base_edges() {
    Fixture fixture;
    fixture.memory.store(vfr_object + layout.override_flag, std::uint8_t{1});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 7, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);
    CHECK(fixture.memory.load<std::uint32_t>(vfr_object + layout.interval) == 99);

    fixture.memory.store(vfr_object + layout.override_flag, std::uint8_t{0});
    fixture.memory.store(vfr_object + layout.base_interval, std::uint32_t{0});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 7, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);

    fixture.memory.store(vfr_object + layout.base_interval, std::uint32_t{2});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 0, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);

    fixture.memory.store(framework + layout.framework_period, std::int64_t{0});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 7, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);
    fixture.memory.store(framework + layout.framework_period, std::int64_t{-1});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 7, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);
}

void test_null_framework() {
    Fixture fixture;
    fixture.memory.store(module_base + layout.framework_pointer, std::uint64_t{0});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);
    CHECK(fixture.memory.load<std::uint32_t>(vfr_object + layout.interval) == 99);
}

// The optional frame step receives the measured period in 60 Hz frames,
// clamped like the ratio, and is left alone by a no-op update.
void test_frame_step() {
    Fixture fixture;
    auto stepped = layout;
    stepped.frame_step = 0x80;
    fixture.memory.store(module_base + stepped.frame_step, 1.0f);
    fixture.memory.store(framework + layout.framework_period, std::int64_t{320'000});
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 1.0f);
    fixture.memory.store(framework + layout.framework_period, std::int64_t{160'000});
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 0.5f);
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == 0.25f);
    fixture.memory.store(framework + layout.framework_period, std::int64_t{1});
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 0.2f);
    fixture.memory.store(framework + layout.framework_period, std::int64_t{1'920'000});
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 2, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 3.0f);

    // The step keeps time while the ratio update is a no-op: under the
    // override and with a zero interval, it still follows the period.
    fixture.memory.store(vfr_object + layout.override_flag, std::uint8_t{1});
    fixture.memory.store(framework + layout.framework_period, std::int64_t{160'000});
    const float ratio = fixture.memory.load<float>(vfr_object + layout.ratio);
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 0.5f);
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == ratio);
    fixture.memory.store(vfr_object + layout.override_flag, std::uint8_t{0});
    fixture.memory.store(framework + layout.framework_period, std::int64_t{320'000});
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 0, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 1.0f);
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == ratio);
    // Without a positive period or a framework the step keeps its value.
    fixture.memory.store(framework + layout.framework_period, std::int64_t{-1});
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 1.0f);
    fixture.memory.store(module_base + layout.framework_pointer, std::uint64_t{0});
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));
    CHECK(fixture.memory.load<float>(module_base + stepped.frame_step) == 1.0f);
    fixture.memory.store(module_base + layout.framework_pointer, framework);
    fixture.memory.store(framework + layout.framework_period, std::int64_t{160'000});

    fixture.memory.failed_write = module_base + stepped.frame_step;
    CHECK(!update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));

    // Without a frame step nothing outside the frame-rate object changes.
    Fixture plain;
    plain.memory.store(module_base + 0x80, 1.0f);
    CHECK(update_frame_rate(layout, module_base, vfr_object, 1, plain.buffers));
    CHECK(plain.memory.load<float>(module_base + 0x80) == 1.0f);
}

void test_maximum_rate_hint() {
    Fixture fixture;
    std::uint32_t maximum_rate = 99;
    CHECK(update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers, &maximum_rate));
    CHECK(maximum_rate == 300);
    fixture.memory.store(vfr_object + layout.override_flag, std::uint8_t{1});
    CHECK(update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers, &maximum_rate));
    CHECK(maximum_rate == 0);
    fixture.memory.store(vfr_object + layout.override_flag, std::uint8_t{0});
    fixture.memory.failed_write = vfr_object + layout.ratio;
    maximum_rate = 99;
    CHECK(!update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers, &maximum_rate));
    CHECK(maximum_rate == 0);
}

// Publication order and failure boundaries are part of the guest contract.
void test_publication_order() {
    Fixture fixture;
    auto stepped = layout;
    stepped.frame_step = 0x80;
    CHECK(update_frame_rate(stepped, module_base, vfr_object, 1, fixture.buffers));
    std::vector<std::uint64_t> writes;
    for (const auto& access : fixture.memory.accesses)
        if (access.write) writes.push_back(access.address);
    CHECK((writes == std::vector<std::uint64_t>{module_base + stepped.frame_step,
        vfr_object + layout.interval, vfr_object + layout.ratio}));

    Fixture failed;
    failed.memory.failed_write = vfr_object + layout.interval;
    CHECK(!update_frame_rate(stepped, module_base, vfr_object, 1, failed.buffers));
    CHECK(failed.memory.load<float>(module_base + stepped.frame_step) == 2.0f);
    CHECK(failed.memory.load<float>(vfr_object + layout.ratio) == -7.0f);
    CHECK(failed.memory.accesses.back().address == vfr_object + layout.interval);
}

void test_memory_failures_and_address_overflow() {
    Fixture fixture;
    fixture.memory.failed_read = vfr_object + layout.base_interval;
    CHECK(!update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers));

    fixture.memory.failed_read = std::numeric_limits<std::uint64_t>::max();
    fixture.memory.failed_write = vfr_object + layout.ratio;
    CHECK(!update_frame_rate(layout, module_base, vfr_object, 2, fixture.buffers));
    CHECK(fixture.memory.load<std::uint32_t>(vfr_object + layout.interval) == 2);
    CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);

    auto overflowing = layout;
    overflowing.framework_pointer = 2;
    CHECK(!update_frame_rate(overflowing, std::numeric_limits<std::uint64_t>::max(),
                                  vfr_object, 2, fixture.buffers));
}

}  // namespace

int main() {
    test_publication_order();
    test_measured_ratios();
    test_clamps();
    test_frame_step();
    test_maximum_rate_hint();
    test_override_and_base_edges();
    test_null_framework();
    test_memory_failures_and_address_overflow();
    if (failures != 0)
        return 1;
    std::puts("01007EF00011E000 frame-rate checks passed");
    return 0;
}
