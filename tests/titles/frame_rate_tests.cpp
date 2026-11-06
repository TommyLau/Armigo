// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/titles/frame_rate.h>
#include <armigo/horizon/ipc.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

using armigo::titles::FrameRateLayout;
using armigo::titles::update_frame_rate;
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

FrameRateLayout fractional_layout() {
    auto result = layout;
    result.variant = armigo::titles::FrameRateVariant::FractionalCounter30;
    result.state = 0x40;
    result.active_state = 2;
    result.counter = 0x48;
    return result;
}

void test_fractional_time() {
    const auto selected = fractional_layout();
    for (const auto [count, ticks] : std::array<std::pair<unsigned, std::int64_t>, 3>{{
             {30, 640000}, {60, 320000}, {120, 160000}}}) {
        Fixture fixture;
        fixture.memory.store(vfr_object + selected.state, selected.active_state);
        fixture.memory.store(vfr_object + selected.counter, 0.0f);
        fixture.memory.store(framework + selected.framework_period, ticks);
        for (unsigned i = 0; i < count; ++i)
            CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
        CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 30.0f);
        CHECK(fixture.memory.load<float>(vfr_object + layout.ratio) == -7.0f);
        CHECK(fixture.memory.load<std::uint32_t>(vfr_object + layout.interval) == 99);
    }
    Fixture fixture;
    fixture.memory.store(vfr_object + selected.state, selected.active_state);
    fixture.memory.store(vfr_object + selected.counter, 0.0f);
    // Two seconds split across 70 and 140 callbacks per second; integer tick
    // rounding is below one microsecond over the complete synthetic schedule.
    for (int i = 0; i < 210; ++i) {
        fixture.memory.store(framework + selected.framework_period, std::int64_t{i < 70 ? 274286 : 137143});
        CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    }
    CHECK(std::abs(fixture.memory.load<float>(vfr_object + selected.counter) - 60.0f) < 0.0002f);
    fixture.memory.store(vfr_object + selected.counter, 0.0f);
    for (std::int64_t ticks : {960000, 1920000, 4800000}) {
        fixture.memory.store(framework + selected.framework_period, ticks);
        CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    }
    CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 4.5f);
    fixture.memory.store(vfr_object + selected.counter, 0.0f);
    fixture.memory.store(framework + selected.framework_period, std::int64_t{32000});
    CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    CHECK(std::abs(fixture.memory.load<float>(vfr_object + selected.counter) - 0.05f) < 0.000001f);
}

// The initial input delay is eight authored 30 Hz frames, independent of
// callback cadence. Saturation must leave the canonical positive-zero bits.
void test_fractional_countdown_contract() {
    auto selected = fractional_layout();
    selected.variant = armigo::titles::FrameRateVariant::FractionalCountdown30;
    for (const auto [count, ticks] : std::array<std::pair<unsigned, std::int64_t>, 3>{{
             {8, 640000}, {16, 320000}, {32, 160000}}}) {
        Fixture fixture;
        fixture.memory.store(vfr_object + selected.state, selected.active_state);
        fixture.memory.store(vfr_object + selected.counter, 8.0f);
        fixture.memory.store(framework + selected.framework_period, ticks);
        for (unsigned i = 0; i < count; ++i) {
            CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
            CHECK(fixture.memory.load<float>(vfr_object + selected.counter) ==
                  8.0f * static_cast<float>(count - i - 1) / count);
        }
        CHECK(fixture.memory.load<std::uint32_t>(vfr_object + selected.counter) == 0);
        CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
        CHECK(fixture.memory.load<std::uint32_t>(vfr_object + selected.counter) == 0);
    }
    Fixture fixture;
    fixture.memory.store(vfr_object + selected.state, selected.active_state);
    fixture.memory.store(vfr_object + selected.counter, 8.0f);
    std::int64_t elapsed_ticks = 0;
    std::int64_t previous_ticks = 0;
    unsigned index = 0;
    constexpr std::array<std::int64_t, 5> periods{274286, 137143, 960000, 1920000, 4800000};
    while (fixture.memory.load<float>(vfr_object + selected.counter) > 0 && index < 32) {
        const auto ticks = periods[index++ % periods.size()];
        fixture.memory.store(framework + selected.framework_period, ticks);
        previous_ticks = elapsed_ticks;
        elapsed_ticks += std::min<std::int64_t>(ticks, 960000);
        CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
        const auto expected = std::max(0.0, 8.0 - static_cast<double>(elapsed_ticks) / 640000.0);
        CHECK(std::abs(fixture.memory.load<float>(vfr_object + selected.counter) - expected) < 0.00001);
    }
    CHECK(index < 32 && previous_ticks < 5120000 && elapsed_ticks >= 5120000);
    CHECK(fixture.memory.load<std::uint32_t>(vfr_object + selected.counter) == 0);
}

void test_fractional_repeat_contract() {
    auto selected = fractional_layout();
    selected.variant = armigo::titles::FrameRateVariant::FractionalRepeat30;
    selected.repeat_flag = 0x4c;
    selected.repeat_frames = 5;
    for (unsigned scale : {1u, 2u, 4u}) {
        Fixture fixture;
        fixture.memory.store(vfr_object + selected.state, selected.active_state);
        fixture.memory.store(vfr_object + selected.counter, 8.0f);
        fixture.memory.store(vfr_object + selected.repeat_flag, std::uint8_t{1});
        fixture.memory.store(framework + selected.framework_period, std::int64_t{640000 / scale});
        std::vector<unsigned> expiries;
        for (unsigned frame = 1; frame <= 18 * scale; ++frame) {
            CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
            if (fixture.memory.load<float>(vfr_object + selected.counter) == 0.0f)
                expiries.push_back(frame);
        }
        CHECK(expiries == std::vector<unsigned>({8 * scale, 13 * scale, 18 * scale}));
        CHECK(fixture.memory.load<std::uint8_t>(vfr_object + selected.repeat_flag) == 1);
        fixture.memory.store(vfr_object + selected.repeat_flag, std::uint8_t{0});
        CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
        CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 0.0f);
        fixture.memory.failed_read = vfr_object + selected.repeat_flag;
        CHECK(!update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
        CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 0.0f);
    }
    Fixture fixture;
    fixture.memory.store(vfr_object + selected.state, selected.active_state);
    fixture.memory.store(vfr_object + selected.counter, 8.0f);
    fixture.memory.store(vfr_object + selected.repeat_flag, std::uint8_t{1});
    constexpr std::array<std::int64_t, 5> periods{274286, 137143, 640000, 960000, 4800000};
    std::int64_t elapsed = 0, last_expiry = 0, target = 8 * 640000;
    unsigned expiries = 0;
    for (unsigned frame = 0; frame < 100 && expiries < 4; ++frame) {
        const auto ticks = periods[frame % periods.size()];
        fixture.memory.store(framework + selected.framework_period, ticks);
        const auto previous = elapsed;
        elapsed += std::min<std::int64_t>(ticks, 960000);
        CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
        if (fixture.memory.load<float>(vfr_object + selected.counter) == 0.0f) {
            CHECK(previous - last_expiry < target + 32 && elapsed - last_expiry >= target - 32);
            last_expiry = elapsed;
            target = 5 * 640000;
            ++expiries;
        }
    }
    CHECK(expiries == 4);
}

void test_counter_state_and_failures() {
    const auto selected = fractional_layout();
    Fixture fixture;
    fixture.memory.store(vfr_object + selected.counter, 7.0f);
    fixture.memory.failed_read = module_base + layout.framework_pointer;
    // An inactive state does not touch its timing source or its counter.
    CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    CHECK(fixture.memory.accesses.size() == 1);
    CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 7.0f);
    fixture.memory.store(vfr_object + selected.state, selected.active_state);
    CHECK(!update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    fixture.memory.failed_read = UINT64_MAX;
    for (std::int64_t ticks : {0, -1}) {
        fixture.memory.store(framework + selected.framework_period, ticks);
        CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
        CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 7.0f);
    }
    fixture.memory.store(module_base + selected.framework_pointer, std::uint64_t{0});
    CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    fixture.memory.store(module_base + selected.framework_pointer, framework);
    fixture.memory.store(framework + selected.framework_period, std::int64_t{640000});
    for (float counter : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        fixture.memory.store(vfr_object + selected.counter, counter);
        CHECK(!update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    }
    fixture.memory.store(vfr_object + selected.counter, 7.0f);
    fixture.memory.failed_write = vfr_object + selected.counter;
    CHECK(!update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers));
    CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 7.0f);
    fixture.memory.failed_write = UINT64_MAX;
    std::uint32_t maximum_rate = 99;
    CHECK(update_frame_rate(selected, module_base, vfr_object, 0, fixture.buffers, &maximum_rate));
    CHECK(fixture.memory.load<float>(vfr_object + selected.counter) == 8.0f && maximum_rate == 0);
    CHECK(!update_frame_rate(selected, module_base, UINT64_MAX, 0, fixture.buffers));
}

} // namespace

int main() {
    test_fractional_time();
    test_fractional_countdown_contract();
    test_fractional_repeat_contract();
    test_counter_state_and_failures();
    if (failures) return 1;
    std::puts("Shared frame-rate layout checks passed");
    return 0;
}
