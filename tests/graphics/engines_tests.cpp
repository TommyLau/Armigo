// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/engines.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

struct Backing {
    static constexpr std::uint64_t guest_base = 0x8000;
    std::array<std::byte, 0x1000> bytes{};
    bool fail_read = false;
    bool fail_write = false;

    static bool read(void* raw, std::uint64_t address, std::span<std::byte> output) noexcept
    {
        auto& self = *static_cast<Backing*>(raw);
        if (self.fail_read || address < guest_base || address - guest_base > self.bytes.size() ||
            output.size() > self.bytes.size() - (address - guest_base)) {
            return false;
        }
        std::memcpy(output.data(), self.bytes.data() + (address - guest_base), output.size());
        return true;
    }

    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> input) noexcept
    {
        auto& self = *static_cast<Backing*>(raw);
        if (self.fail_write || address < guest_base || address - guest_base > self.bytes.size() ||
            input.size() > self.bytes.size() - (address - guest_base)) {
            return false;
        }
        std::memcpy(self.bytes.data() + (address - guest_base), input.data(), input.size());
        return true;
    }
};

GpuAddressSpace make_memory(Backing& backing)
{
    const Mapping mapping{0x1000, 0x1000, 0, Backing::guest_base, 0x1000, 0x1000};
    AddressSpaceResult result =
        make_address_space({&mapping, 1}, {Backing::read, Backing::write, &backing});
    CHECK(std::holds_alternative<GpuAddressSpace>(result));
    if (!std::holds_alternative<GpuAddressSpace>(result)) {
        std::abort();
    }
    return std::get<GpuAddressSpace>(std::move(result));
}

MethodCall call(std::uint32_t method, std::uint32_t argument, bool last = true)
{
    return {method, 2, argument, 0x4000 + method * 4, last};
}

struct Completion {
    struct OwnedIncrementState {
        std::vector<std::uint32_t> increments{};
    };

    bool allow_increment = true;
    bool is_expired = true;
    std::array<std::uint32_t, 16> increments{};
    std::size_t increment_count = 0;
    std::uint32_t expired_id = 0;
    std::uint32_t expired_threshold = 0;
    std::uint64_t tick_value = 0x1122334455667788ULL;
    std::shared_ptr<OwnedIncrementState> owned_increment_state{};

    static bool increment(void* raw, std::uint32_t id) noexcept
    {
        auto& self = *static_cast<Completion*>(raw);
        if (!self.allow_increment) {
            return false;
        }
        if (self.increment_count == self.increments.size()) {
            return false;
        }
        self.increments[self.increment_count++] = id;
        return true;
    }

    static bool expired(void* raw, std::uint32_t id, std::uint32_t threshold) noexcept
    {
        auto& self = *static_cast<Completion*>(raw);
        self.expired_id = id;
        self.expired_threshold = threshold;
        return self.is_expired;
    }

    static std::uint64_t ticks(void* raw) noexcept
    {
        return static_cast<Completion*>(raw)->tick_value;
    }

    static GuestPublication retain_increment(void* raw, std::uint32_t id,
                                             std::uint32_t count)
    {
        auto state = static_cast<Completion*>(raw)->owned_increment_state;
        if (!state) return {};
        return [state = std::move(state), id, count]() -> std::optional<CommandResult> {
            for (std::uint32_t index = 0; index != count; ++index) {
                state->increments.push_back(id);
            }
            return CommandResult{};
        };
    }

    CompletionCallbacks callbacks() noexcept { return {increment, expired, ticks, this}; }
};

struct OrderedEffectQueue {
    std::vector<GuestEffect> effects{};
    unsigned coherence_calls = 0;

    static CommandResult enqueue(void* raw, const GpuAddressSpace&, GuestEffect effect) noexcept
    {
        try {
            static_cast<OrderedEffectQueue*>(raw)->effects.push_back(std::move(effect));
            return {};
        } catch (...) {
            return {CommandStatus::BudgetExceeded, "Synthetic ordered-effect queue allocation failed"};
        }
    }

    static CommandResult coherence(void* raw, const GpuAddressSpace& memory,
                                   std::uint64_t, std::uint64_t) noexcept
    {
        auto& self = *static_cast<OrderedEffectQueue*>(raw);
        ++self.coherence_calls;
        while (!self.effects.empty()) {
            auto& effect = self.effects.front();
            if (effect.size != 0) {
                const auto written = memory.write(
                    effect.address,
                    std::span<const std::byte>(effect.payload.data(), effect.size));
                if (!written) return {CommandStatus::MemoryFault, written.detail};
            }
            if (effect.publish) {
                const auto published = effect.publish();
                if (!published) {
                    return {CommandStatus::Stopped,
                            "Synthetic ordered-effect publication remained busy"};
                }
                if (!published->ok()) return *published;
            }
            self.effects.erase(self.effects.begin());
        }
        return {};
    }
};

struct OrderedWriteQueue {
    struct Write {
        std::uint64_t address = 0;
        std::vector<std::byte> payload{};
    };

    std::vector<Write> writes{};
    unsigned coherence_calls = 0;
    unsigned read_coherence_calls = 0;
    std::optional<CommandResult> failure{};

    static CommandResult enqueue(void* raw, const GpuAddressSpace&, std::uint64_t address,
                                 std::span<const std::byte> payload) noexcept
    {
        auto& self = *static_cast<OrderedWriteQueue*>(raw);
        if (self.failure) return *self.failure;
        try {
            Write write;
            write.address = address;
            write.payload.assign(payload.begin(), payload.end());
            self.writes.push_back(std::move(write));
            return {};
        } catch (...) {
            return {CommandStatus::BudgetExceeded,
                    "Synthetic ordered-write queue allocation failed"};
        }
    }

    CommandResult flush(const GpuAddressSpace& memory, std::uint64_t address = 0,
                        std::uint64_t size = 0) noexcept
    {
        std::size_t count = writes.size();
        if (size != 0) {
            count = 0;
            for (std::size_t index = 0; index != writes.size(); ++index) {
                const auto& write = writes[index];
                if (write.address < address + size &&
                    address < write.address + write.payload.size()) {
                    count = index + 1;
                }
            }
        }
        while (count != 0) {
            --count;
            const auto written = memory.write(writes.front().address, writes.front().payload);
            if (!written) return {CommandStatus::MemoryFault, written.detail};
            writes.erase(writes.begin());
        }
        return {};
    }

    static CommandResult coherence(void* raw, const GpuAddressSpace& memory,
                                   std::uint64_t, std::uint64_t) noexcept
    {
        auto& self = *static_cast<OrderedWriteQueue*>(raw);
        ++self.coherence_calls;
        return self.flush(memory);
    }

    static CommandResult read_coherence(void* raw, const GpuAddressSpace& memory,
                                        std::uint64_t address, std::uint64_t size) noexcept
    {
        auto& self = *static_cast<OrderedWriteQueue*>(raw);
        ++self.read_coherence_calls;
        return self.flush(memory, address, size);
    }
};

constexpr std::uint32_t macro_value(std::uint32_t operation,
                                    std::uint32_t assignment,
                                    std::uint32_t destination,
                                    std::uint32_t source_a,
                                    std::uint32_t source_b,
                                    std::uint32_t alu = 0,
                                    bool exit = false)
{
    return operation | (assignment << 4) | (static_cast<std::uint32_t>(exit) << 7) |
           (destination << 8) | (source_a << 11) | (source_b << 14) | (alu << 17);
}

constexpr std::uint32_t macro_immediate(std::uint32_t assignment,
                                        std::uint32_t destination,
                                        std::uint32_t source_a,
                                        std::int32_t immediate,
                                        bool exit = false)
{
    return 1U | (assignment << 4) | (static_cast<std::uint32_t>(exit) << 7) | (destination << 8) |
           (source_a << 11) | ((static_cast<std::uint32_t>(immediate) & 0x3ffff) << 14);
}

constexpr std::uint32_t macro_nop()
{
    return macro_immediate(1, 0, 0, 0);
}

void upload_macro(EngineFrontEnd& engines,
                  const GpuAddressSpace& memory,
                  CompletionCallbacks completion,
                  std::span<const std::uint32_t> words,
                  std::uint32_t entry_slot = 0)
{
    CHECK(engines.call(call(0x45, 0), engine_class::threed, memory, completion).ok());
    for (std::uint32_t word : words) {
        CHECK(engines.call(call(0x46, word), engine_class::threed, memory, completion).ok());
    }
    CHECK(engines.call(call(0x47, entry_slot), engine_class::threed, memory, completion).ok());
    CHECK(engines.call(call(0x48, 0), engine_class::threed, memory, completion).ok());
}

void test_macro_upload_call_and_render_stop()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    const std::array words{
        macro_immediate(2, 7, 0, 0x300),
        macro_value(0, 4, 2, 1, 0, 9),
        macro_immediate(2, 7, 0, 0x674),
        macro_immediate(4, 2, 0, 1, true),
        macro_nop(),
    };
    upload_macro(engines, memory, completion.callbacks(), words);

    CHECK(
        engines.call(call(0xe00, 0x77, true), engine_class::threed, memory, completion.callbacks())
            .status == CommandStatus::Stopped);
    CHECK(engines.rendering_stop().has_value());
    if (engines.rendering_stop()) {
        const RenderStop& stop = *engines.rendering_stop();
        CHECK(stop.kind == RenderKind::Clear);
        CHECK(stop.original.method == 0x674);
        CHECK(stop.original.argument == 1);
        CHECK(stop.registers[0x300] == 0x77);
    }
    CHECK(
        engines.call(call(0x301, 9), engine_class::threed, memory, completion.callbacks()).status ==
        CommandStatus::Stopped);
}

void test_macro_argument_order_and_invalid_code()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    const std::array words{
        macro_immediate(0, 2, 0, 0),
        macro_immediate(2, 7, 0, 0x310),
        macro_value(0, 4, 3, 2, 0, 9, true),
        macro_nop(),
    };
    upload_macro(engines, memory, completion.callbacks(), words);
    CHECK(
        engines.call(call(0xe00, 0x11, false), engine_class::threed, memory, completion.callbacks())
            .ok());
    CHECK(
        engines.call(call(0xe01, 0x22, true), engine_class::threed, memory, completion.callbacks())
            .ok());
    CHECK(
        engines.call(call(0x674, 1), engine_class::threed, memory, completion.callbacks()).status ==
        CommandStatus::Stopped);
    CHECK(engines.rendering_stop()->registers[0x310] == 0x22);

    EngineFrontEnd missing;
    CHECK(
        missing.call(call(0xe00, 1), engine_class::threed, memory, completion.callbacks()).status ==
        CommandStatus::Unsupported);
}

void test_shadow_and_snapshot_descriptions()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x49, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(
        engines.call(call(0x200, 0x11), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x49, 3), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(
        engines.call(call(0x200, 0x22), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x49, 2), engine_class::threed, memory, completion.callbacks()).ok());

    CHECK(engines.call(call(0x201, 0x1234), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(engines.call(call(0x582, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x583, 0x2000), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(engines.call(call(0x800, 1), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(
        engines.call(call(0x801, 0x40), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(
        engines.call(call(0x3fd, 0xdead0000), engine_class::threed, memory, completion.callbacks())
            .ok());
    CHECK(
        engines.call(call(0x3fe, 0xbeef0000), engine_class::threed, memory, completion.callbacks())
            .ok());
    CHECK(
        engines.call(call(0x48a, 640), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(
        engines.call(call(0x48b, 360), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(
        engines.call(call(0x674, 3), engine_class::threed, memory, completion.callbacks()).status ==
        CommandStatus::Stopped);
    const RenderStop& stop = *engines.rendering_stop();
    CHECK(stop.registers[0x200] == 0x11);
    CHECK(stop.color_targets[0].address == 0x0000001100001234ULL);
    CHECK(stop.programs[0].address == 0x2040);
    CHECK(stop.depth.width == 640);
    CHECK(stop.depth.height == 360);
}

void test_constant_buffer_upload_and_fault()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x8e0, 8), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x8e1, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x8e2, 0x1080), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(engines.call(call(0x8e3, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(
        engines.call(call(0x8e4, 0x11223344), engine_class::threed, memory, completion.callbacks())
            .ok());
    CHECK(
        engines.call(call(0x8e5, 0xaabbccdd), engine_class::threed, memory, completion.callbacks())
            .ok());
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    std::memcpy(&first, backing.bytes.data() + 0x80, 4);
    std::memcpy(&second, backing.bytes.data() + 0x84, 4);
    CHECK(first == 0x11223344);
    CHECK(second == 0xaabbccdd);
    CHECK(
        engines.call(call(0x8e6, 1), engine_class::threed, memory, completion.callbacks()).status ==
        CommandStatus::Unsupported);

    EngineFrontEnd failed;
    backing.fail_write = true;
    CHECK(failed.call(call(0x8e0, 4), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(failed.call(call(0x8e2, 0x1080), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(
        failed.call(call(0x8e4, 1), engine_class::threed, memory, completion.callbacks()).status ==
        CommandStatus::MemoryFault);
}

void test_completion_and_reports()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x1c, 9), 0, memory, completion.callbacks()).ok());
    completion.is_expired = false;
    CHECK(engines.call(call(0x1d, 7U << 8), 0, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    CHECK(completion.expired_id == 7);
    CHECK(completion.expired_threshold == 9);

    completion.allow_increment = false;
    CHECK(engines.call(call(0x1d, (8U << 8) | 1), 0, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    completion.allow_increment = true;
    CHECK(engines.call(call(0xb2, 0x1234), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(completion.increment_count == 1);
    CHECK(completion.increments[0] == 0x1234);

    CHECK(engines.call(call(0x6c0, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6c1, 0x1100), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(
        engines.call(call(0x6c2, 0x55667788), engine_class::threed, memory, completion.callbacks())
            .ok());
    CHECK(engines.call(call(0x6c3, 1U << 28), engine_class::threed, memory, completion.callbacks())
              .ok());
    std::uint32_t short_value = 0;
    std::memcpy(&short_value, backing.bytes.data() + 0x100, 4);
    CHECK(short_value == 0x55667788);

    CHECK(engines.call(call(0x6c1, 0x1120), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(engines.call(call(0x6c3, 0), engine_class::threed, memory, completion.callbacks()).ok());
    std::uint64_t long_payload = 0;
    std::uint64_t ticks = 0;
    std::memcpy(&long_payload, backing.bytes.data() + 0x120, 8);
    std::memcpy(&ticks, backing.bytes.data() + 0x128, 8);
    CHECK(long_payload == 0x55667788);
    CHECK(ticks == completion.tick_value);
}

void test_constant_buffer_binding_snapshots()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    const auto write = [&](std::uint32_t method, std::uint32_t value) {
        return engines.call(call(method, value), engine_class::threed, memory, completion.callbacks());
    };
    for (std::uint32_t stage = 0; stage < 5; ++stage) {
        CHECK(write(0x8e0, 0x80 + stage * 16).ok());
        CHECK(write(0x8e1, 0x12 + stage).ok());
        CHECK(write(0x8e2, 0x4000 + stage * 0x100).ok());
        CHECK(write(0x904 + stage * 8, ((17 - stage) << 4) | 1).ok());
    }
    CHECK(write(0x8e0, 0x999).ok());
    CHECK(write(0x8e1, 0xff).ok());
    CHECK(write(0x8e2, 0xffff0000).ok());
    CHECK(write(0x904, 17 << 4).ok());
    CHECK(write(0x90c, (18 << 4) | 1).status == CommandStatus::Unsupported);
    CHECK(write(0x905, 0xffffffff).ok());
    CHECK(write(0x585, 0).status == CommandStatus::Stopped);
    const auto& snapshot = *engines.rendering_stop();
    CHECK(!snapshot.constant_buffers[0][17].enabled);
    CHECK(snapshot.constant_buffers[0][17].address == 0);
    CHECK(snapshot.constant_buffers[0][17].size == 0);
    for (std::uint32_t stage = 1; stage < 5; ++stage) {
        const auto& binding = snapshot.constant_buffers[stage][17 - stage];
        CHECK(binding.enabled);
        CHECK(binding.address == (std::uint64_t{0x12 + stage} << 32) + 0x4000 + stage * 0x100);
        CHECK(binding.size == 0x80 + stage * 16);
        CHECK(!snapshot.constant_buffers[stage][0].enabled);
    }
    CHECK(completion.increment_count == 0);
    CHECK(backing.bytes == Backing{}.bytes);

    EngineFrontEnd replay;
    const auto replay_write = [&](std::uint32_t method, std::uint32_t value) {
        return replay.call(call(method, value), engine_class::threed, memory, completion.callbacks());
    };
    CHECK(replay_write(0x8e0, 64).ok());
    CHECK(replay_write(0x8e1, 0).ok());
    CHECK(replay_write(0x8e2, 0x1000).ok());
    CHECK(replay_write(0x904, (3 << 4) | 1).ok());
    CHECK(replay_write(0x49, 2).ok());
    CHECK(replay_write(0x8e0, 128).ok());
    CHECK(replay_write(0x8e2, 0x2000).ok());
    CHECK(replay_write(0x904, (4 << 4) | 1).ok());
    CHECK(replay_write(0x49, 3).ok());
    CHECK(replay_write(0x8e0, 0x555).ok());
    CHECK(replay_write(0x8e2, 0x5000).ok());
    CHECK(replay_write(0x904, (4 << 4) | 1).ok());
    backing.bytes[0] = std::byte{0x6d};
    CHECK(replay_write(0x585, 0).status == CommandStatus::Stopped);
    const auto& replayed = *replay.rendering_stop();
    CHECK(replayed.constant_buffers[0][3].address == 0x1000);
    CHECK(replayed.constant_buffers[0][3].size == 64);
    CHECK(replayed.constant_buffers[0][4].address == 0x2000);
    CHECK(replayed.constant_buffers[0][4].size == 128);
    std::array<std::byte, 1> current{};
    CHECK(memory.read(replayed.constant_buffers[0][3].address, current));
    CHECK(current[0] == std::byte{0x6d});
}

void test_effective_clear_startup_state()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd untouched;
    CHECK(untouched.call(call(0x674, 0x3c), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    CHECK(untouched.rendering_stop()->registers[0x556] == 0);
    CHECK(untouched.rendering_stop()->effective_render_enable_mode == 1);
    CHECK(untouched.rendering_stop()->registers[0xdf] == 1);
    for (std::size_t viewport = 0; viewport != 16; ++viewport)
        CHECK(untouched.rendering_stop()->registers[0x286 + viewport * 8] == 0x6420);

    EngineFrontEnd swizzle;
    CHECK(swizzle.call(call(0x286 + 7 * 8, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(swizzle.call(call(0x585, 0), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    CHECK(swizzle.rendering_stop()->registers[0x286 + 7 * 8] == 0);
    CHECK(swizzle.rendering_stop()->registers[0x286] == 0x6420);

    EngineFrontEnd swizzle_replay;
    CHECK(swizzle_replay.call(call(0x49, 3), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(swizzle_replay.call(call(0x286 + 15 * 8, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(swizzle_replay.call(call(0x585, 0), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    CHECK(swizzle_replay.rendering_stop()->registers[0x286 + 15 * 8] == 0x6420);

    for (const auto mode : {0u, 1u, 2u, 5u}) {
        EngineFrontEnd explicit_mode;
        CHECK(explicit_mode.call(call(0x556, mode), engine_class::threed, memory, completion.callbacks()).ok());
        CHECK(explicit_mode.call(call(0xdf, 0), engine_class::threed, memory, completion.callbacks()).ok());
        CHECK(explicit_mode.call(call(0x674, 0x3c), engine_class::threed, memory, completion.callbacks()).status ==
              CommandStatus::Stopped);
        CHECK(explicit_mode.rendering_stop()->effective_render_enable_mode == mode);
        CHECK(explicit_mode.rendering_stop()->registers[0xdf] == 0);
    }

    EngineFrontEnd replay;
    CHECK(replay.call(call(0x556, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(replay.call(call(0x49, 2), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(replay.call(call(0x556, 1), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(replay.call(call(0x49, 3), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(replay.call(call(0x556, 1), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(replay.call(call(0xdf, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(replay.call(call(0x674, 0x3c), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    CHECK(replay.rendering_stop()->effective_render_enable_mode == 0);
    CHECK(replay.rendering_stop()->registers[0xdf] == 1);
}

void test_falcon_configuration_handshake()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    const auto original = backing.bytes;
    EngineFrontEnd engines;
    for (std::uint32_t index = 0; index < 0x1100; ++index) {
        CHECK(engines.call(call(0xd00, 0), engine_class::threed, memory, completion.callbacks()).ok());
        CHECK(engines.call(call(0xd01, 0x13579bdf), engine_class::threed, memory, completion.callbacks()).ok());
        CHECK(engines.call(call(0xd02, 0x2468ace0), engine_class::threed, memory, completion.callbacks()).ok());
        // Repeated requests, including argument zero, never accumulate obligations.
        CHECK(engines.call(call(0x8c4, index & 1), engine_class::threed, memory, completion.callbacks()).ok());
    }
    CHECK(engines.call(call(0x674, 0x3c), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    const auto& stop = *engines.rendering_stop();
    CHECK(stop.registers[0xd00] == 1);
    CHECK(stop.registers[0xd01] == 0x13579bdf);
    CHECK(stop.registers[0xd02] == 0x2468ace0);
    CHECK(completion.increment_count == 0);
    CHECK(backing.bytes == original);

    EngineFrontEnd ordinary;
    CHECK(ordinary.call(call(0xd00, 7), engine_class::threed, memory, completion.callbacks()).ok());
    for (std::uint32_t method = 0x8c0; method <= 0x8df; ++method) {
        if (method == 0x8c4) continue;
        CHECK(ordinary.call(call(method, method + 10), engine_class::threed, memory, completion.callbacks()).ok());
    }
    CHECK(ordinary.call(call(0x674, 0x3c), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    CHECK(ordinary.rendering_stop()->registers[0xd00] == 7);
    for (std::uint32_t method = 0x8c0; method <= 0x8df; ++method) {
        if (method != 0x8c4) CHECK(ordinary.rendering_stop()->registers[method] == method + 10);
    }

    // The acknowledgment changes active state only; replay retains tracked zero.
    EngineFrontEnd shadow;
    CHECK(shadow.call(call(0xd00, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(shadow.call(call(0x8c4, 7), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(shadow.call(call(0x49, 3), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(shadow.call(call(0xd00, 9), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(shadow.call(call(0x674, 0x3c), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    CHECK(shadow.rendering_stop()->registers[0xd00] == 0);
}

void test_other_engine_stops_and_unsupported_actions()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd compute;
    for (std::size_t index = 0; index < 0x100; ++index) {
        backing.bytes[0x200 + index] = static_cast<std::byte>(index);
    }
    const std::uint32_t program_start = 0x40;
    std::memcpy(backing.bytes.data() + 0x220, &program_start, sizeof(program_start));
    CHECK(
        compute.call(call(0xad, 0x12), engine_class::compute, memory, completion.callbacks()).ok());
    CHECK(compute.call(call(0x582, 0), engine_class::compute, memory, completion.callbacks()).ok());
    CHECK(compute.call(call(0x583, 0x5000), engine_class::compute, memory, completion.callbacks())
              .ok());
    CHECK(
        compute.call(call(0xaf, 1), engine_class::compute, memory, completion.callbacks()).status ==
        CommandStatus::Stopped);
    CHECK(compute.rendering_stop()->kind == RenderKind::Compute);
    CHECK(compute.rendering_stop()->has_compute_descriptor);
    CHECK(compute.rendering_stop()->compute_descriptor_address == 0x1200);
    CHECK(compute.rendering_stop()->compute_descriptor[7] == std::byte{7});
    CHECK(compute.rendering_stop()->programs[0].address == 0x5040);
    backing.bytes[0x207] = std::byte{0xff};
    CHECK(compute.rendering_stop()->compute_descriptor[7] == std::byte{7});

    EngineFrontEnd compute_fault;
    CHECK(
        compute_fault.call(call(0xad, 0x30), engine_class::compute, memory, completion.callbacks())
            .ok());
    CHECK(compute_fault.call(call(0xaf, 1), engine_class::compute, memory, completion.callbacks())
              .status == CommandStatus::MemoryFault);
    CHECK(!compute_fault.rendering_stop().has_value());

    EngineFrontEnd twod;
    CHECK(twod.call(call(0x237, 1), engine_class::twod, memory, completion.callbacks()).status ==
          CommandStatus::Stopped);
    CHECK(twod.rendering_stop()->kind == RenderKind::Blit);

    EngineFrontEnd actions;
    for (std::uint32_t engine : {engine_class::compute, engine_class::inline_to_memory}) {
        CHECK(actions.call(call(0x60, 4), engine, memory, completion.callbacks()).ok());
        CHECK(actions.call(call(0x61, 1), engine, memory, completion.callbacks()).ok());
        CHECK(actions.call(call(0x62, 0), engine, memory, completion.callbacks()).ok());
        CHECK(actions.call(call(0x63, engine == engine_class::compute ? 0x1300 : 0x1400),
                           engine, memory, completion.callbacks()).ok());
        CHECK(actions.call(call(0x6a, 0), engine, memory, completion.callbacks()).ok());
        CHECK(actions.call(call(0x6b, 0), engine, memory, completion.callbacks()).ok());
        CHECK(actions.call(call(0x6c, 0x1011), engine, memory, completion.callbacks()).ok());
        CHECK(actions.call(call(0x6d, 0x12345678), engine, memory, completion.callbacks()).ok());
    }
    CHECK(backing.bytes[0x300] == std::byte{0x78});
    CHECK(backing.bytes[0x400] == std::byte{0x78});
    // A DMA pitch copy moves the uploaded word; a launch of a layered
    // block-linear surface is an explicit unsupported operation.
    for (const auto [method, value] : {std::pair{0x100u, 0u}, std::pair{0x101u, 0x1300u},
                                       std::pair{0x102u, 0u}, std::pair{0x103u, 0x1500u},
                                       std::pair{0x106u, 4u}, std::pair{0x107u, 1u}})
        CHECK(actions.call(call(method, value), engine_class::dma_copy, memory, completion.callbacks()).ok());
    CHECK(actions.call(call(0xc0, 0x182), engine_class::dma_copy, memory, completion.callbacks()).ok());
    CHECK(backing.bytes[0x500] == std::byte{0x78});
    CHECK(actions.call(call(0x1ce, 1), engine_class::dma_copy, memory, completion.callbacks()).ok());
    CHECK(actions.call(call(0xc0, 0x302), engine_class::dma_copy, memory, completion.callbacks())
              .status == CommandStatus::Unsupported);
    CHECK(
        actions.call(call(0x6c, 1), engine_class::inline_to_memory, memory, completion.callbacks())
            .status == CommandStatus::Unsupported);
    CHECK(
        actions.call(call(0x8c5, 1), engine_class::threed, memory, completion.callbacks()).ok());

    EngineFrontEnd data_methods;
    CHECK(data_methods.call(call(0x35e, 4), engine_class::threed,
                            memory, completion.callbacks()).ok());
    CHECK(data_methods.call(call(0x5f8, 7), engine_class::threed,
                            memory, completion.callbacks()).ok());
    CHECK(data_methods.call(call(0x4c0, 8), engine_class::threed,
                            memory, completion.callbacks()).ok());
    CHECK(data_methods.call(call(0x57b, 9), engine_class::threed,
                            memory, completion.callbacks()).ok());
    CHECK(data_methods.call(call(0x586, 10), engine_class::threed,
                            memory, completion.callbacks()).ok());
    for (std::uint32_t method : {0x4c1u, 0x57au, 0x57cu})
        CHECK(data_methods.call(call(method, 1), engine_class::threed,
                                memory, completion.callbacks()).ok());
}

void test_inline_upload_blocks_ordered_side_effects()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x4, 0), 0, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x5, 0x1090), 0, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6, 0x12345678), 0, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6c0, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6c1, 0x1090), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6c2, 0x12345678), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x60, 7), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x61, 1), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x62, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x63, 0x1080), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(engines.call(call(0x6a, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6b, 0), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6c, 0x1011), engine_class::threed, memory, completion.callbacks())
              .ok());
    CHECK(engines.has_pending_memory_work());

    CHECK(engines.call(call(0x6d, 0x44332211, true), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.has_pending_memory_work());
    CHECK(backing.bytes[0x80] == std::byte{0});
    CHECK(engines.call(call(0x1d, (3u << 8) | 1), 0, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    CHECK(completion.increment_count == 0);
    CHECK(engines.call(call(0x1b, 0x12345678), 0, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    CHECK(engines.call(call(0x14, 0), 0, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    CHECK(engines.call(call(0x1e, 0), 0, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    CHECK(engines.call(call(0x6c3, 1u << 28), engine_class::threed,
                       memory, completion.callbacks()).status == CommandStatus::Unsupported);
    CHECK(backing.bytes[0x90] == std::byte{0});
    CHECK(engines.call(call(0x674, 1), engine_class::threed, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    CHECK(!engines.rendering_stop());

    CHECK(engines.call(call(0x6d, 0xddccbbaa, true), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(!engines.has_pending_memory_work());
    CHECK(backing.bytes[0x80] == std::byte{0x11});
    CHECK(backing.bytes[0x86] == std::byte{0xcc});
    CHECK(engines.call(call(0x1d, (3u << 8) | 1), 0, memory, completion.callbacks()).ok());
    CHECK(completion.increment_count == 1);
}

void test_inline_upload_failure_remains_blocked()
{
    Backing backing;
    backing.fail_write = true;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x60, 4), engine_class::inline_to_memory,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x61, 1), engine_class::inline_to_memory,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x62, 0), engine_class::inline_to_memory,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x63, 0x1080), engine_class::inline_to_memory,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6a, 0), engine_class::inline_to_memory,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6b, 0), engine_class::inline_to_memory,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6c, 0x1011), engine_class::inline_to_memory,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6d, 1), engine_class::inline_to_memory,
                       memory, completion.callbacks()).status == CommandStatus::MemoryFault);
    CHECK(engines.has_pending_memory_work());
    CHECK(engines.call(call(0x14, 0), 0, memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
}

void test_macro_can_finish_but_not_overtake_upload()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd finishes;
    const std::array finish_code{
        macro_immediate(2, 7, 0, 0x6d),
        macro_value(0, 4, 2, 1, 0, 9, true),
        macro_nop(),
    };
    upload_macro(finishes, memory, completion.callbacks(), finish_code);
    CHECK(finishes.call(call(0x60, 8), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.call(call(0x61, 1), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.call(call(0x62, 0), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.call(call(0x63, 0x1080), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.call(call(0x6a, 0), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.call(call(0x6b, 0), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.call(call(0x6c, 0x1011), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.call(call(0x6d, 0x44332211, true), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(finishes.has_pending_memory_work());
    CHECK(finishes.call(call(0xe00, 0x55667788, true), engine_class::threed,
                        memory, completion.callbacks()).ok());
    CHECK(!finishes.has_pending_memory_work());
    CHECK(backing.bytes[0x80] == std::byte{0x11});
    CHECK(backing.bytes[0x84] == std::byte{0x88});
    CHECK(backing.bytes[0x87] == std::byte{0x55});

    EngineFrontEnd overtakes;
    const std::array clear_code{
        macro_immediate(2, 7, 0, 0x674),
        macro_immediate(4, 2, 0, 1, true),
        macro_nop(),
    };
    upload_macro(overtakes, memory, completion.callbacks(), clear_code);
    CHECK(overtakes.call(call(0x60, 8), engine_class::threed,
                         memory, completion.callbacks()).ok());
    CHECK(overtakes.call(call(0x61, 1), engine_class::threed,
                         memory, completion.callbacks()).ok());
    CHECK(overtakes.call(call(0x62, 0), engine_class::threed,
                         memory, completion.callbacks()).ok());
    CHECK(overtakes.call(call(0x63, 0x1100), engine_class::threed,
                         memory, completion.callbacks()).ok());
    CHECK(overtakes.call(call(0x6a, 0), engine_class::threed,
                         memory, completion.callbacks()).ok());
    CHECK(overtakes.call(call(0x6b, 0), engine_class::threed,
                         memory, completion.callbacks()).ok());
    CHECK(overtakes.call(call(0x6c, 0x1011), engine_class::threed,
                         memory, completion.callbacks()).ok());
    CHECK(overtakes.call(call(0xe00, 0, true), engine_class::threed,
                         memory, completion.callbacks()).status ==
          CommandStatus::Unsupported);
    CHECK(overtakes.has_pending_memory_work());
    CHECK(!overtakes.rendering_stop());
}

struct TestRenderer {
    Completion* completion = nullptr;
    unsigned calls = 0;
    bool fail = false;
    std::uint32_t effective_argument = 0;

    static CommandResult clear(void* raw, const RenderStop& snapshot,
                               const GpuAddressSpace& memory) noexcept
    {
        auto& self = *static_cast<TestRenderer*>(raw);
        ++self.calls;
        CHECK(self.completion->increment_count == 0);
        CHECK(snapshot.kind == RenderKind::Clear);
        self.effective_argument = snapshot.registers[0x674];
        if (self.fail) return {CommandStatus::MemoryFault, "Synthetic backend failure"};
        constexpr std::array pixel{std::byte{0x12}, std::byte{0x34},
                                   std::byte{0x56}, std::byte{0x78}};
        CHECK(memory.write(0x1080, pixel).ok());
        return {};
    }
};

void test_synchronous_clear_backend()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    TestRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {TestRenderer::clear, &renderer};
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x674, 0x3c), engine_class::threed, memory, callbacks).ok());
    CHECK(renderer.calls == 1);
    CHECK(renderer.effective_argument == 0x3c);
    CHECK(!engines.rendering_stop());
    CHECK(backing.bytes[0x80] == std::byte{0x12});
    CHECK(engines.call(call(0x1d, (3u << 8) | 1u), 0, memory, callbacks).ok());
    CHECK(completion.increment_count == 1);
    CHECK(engines.call(call(0x585, 0), engine_class::threed, memory, callbacks).status ==
          CommandStatus::Stopped);
    CHECK(renderer.calls == 1);

    Completion failed_completion;
    TestRenderer failing{&failed_completion, 0, true};
    auto failed_callbacks = failed_completion.callbacks();
    failed_callbacks.rendering = {TestRenderer::clear, &failing};
    EngineFrontEnd failed;
    CHECK(failed.call(call(0x674, 0x3c), engine_class::threed, memory,
                      failed_callbacks).status == CommandStatus::MemoryFault);
    CHECK(failed.rendering_stop().has_value());
    CHECK(failed.call(call(0x1d, (3u << 8) | 1u), 0, memory,
                      failed_callbacks).status == CommandStatus::Stopped);
    CHECK(failed_completion.increment_count == 0);
    CHECK(failing.calls == 1);

    Completion shadow_completion;
    TestRenderer shadow_renderer{&shadow_completion};
    auto shadow_callbacks = shadow_completion.callbacks();
    shadow_callbacks.rendering = {TestRenderer::clear, &shadow_renderer};
    EngineFrontEnd shadow;
    CHECK(shadow.call(call(0x674, 0x3c), engine_class::threed, memory, shadow_callbacks).ok());
    CHECK(shadow.call(call(0x49, 3), engine_class::threed, memory, shadow_callbacks).ok());
    CHECK(shadow.call(call(0x674, 0), engine_class::threed, memory, shadow_callbacks).ok());
    CHECK(shadow_renderer.effective_argument == 0x3c);
}

void test_zero_length_engine_upload_is_complete()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x60, 0), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x61, 1), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x62, 0xffffffff), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x63, 0xffffffff), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6a, 3), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6b, 4), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x6c, 0x1011), engine_class::threed,
                       memory, completion.callbacks()).ok());
    CHECK(!engines.has_pending_memory_work());
}

struct DrawRenderer {
    Completion* completion = nullptr;
    std::array<DrawDescription, 64> draws{};
    std::array<std::uint32_t, 64> methods{};
    std::array<std::uint32_t, 64> index_types{};
    std::array<std::size_t, 64> inline_sizes{};
    std::array<std::array<std::uint32_t, 8>, 64> inline_values{};
    std::array<std::uint64_t, 64> dirty{}, streams{}, serials{};
    unsigned calls = 0;
    bool fail = false;
    std::uint32_t raw_end = 0;
    std::uint32_t raw_count = 0;
    std::uint32_t condition = 0;
    std::uint32_t swizzle = 0;
    std::uint32_t raster = 0;

    static CommandResult draw(void* raw, const RenderStop& snapshot,
                              const GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<DrawRenderer*>(raw);
        CHECK(snapshot.kind == RenderKind::Draw);
        CHECK(snapshot.draw.has_value());
        CHECK(self.completion->increment_count == 0);
        if (!snapshot.draw || self.calls == self.draws.size()) {
            return {CommandStatus::Unsupported, "Synthetic draw callback state is invalid"};
        }
        const auto call_index = self.calls++;
        self.draws[call_index] = *snapshot.draw;
        self.dirty[call_index] = snapshot.dirty_groups;
        self.streams[call_index] = snapshot.state_stream;
        self.serials[call_index] = snapshot.state_serial;
        self.methods[call_index] = snapshot.original.method;
        self.index_types[call_index] = snapshot.registers[0x5f6];
        self.inline_sizes[call_index] = snapshot.inline_indices.size();
        for (std::size_t index = 0;
             index < snapshot.inline_indices.size() && index < self.inline_values[call_index].size();
             ++index) {
            self.inline_values[call_index][index] = snapshot.inline_indices[index];
        }
        self.raw_end = snapshot.original.argument;
        self.raw_count = snapshot.registers[snapshot.draw->indexed ? 0x5f8 : 0x35e];
        self.condition = snapshot.effective_render_enable_mode;
        self.swizzle = snapshot.registers[0x286];
        self.raster = snapshot.registers[0xdf];
        if (self.fail) return {CommandStatus::MemoryFault, "Synthetic draw failure"};
        const std::array bytes{std::byte{0x21}, std::byte{0x43}, std::byte{0x65}, std::byte{0x87}};
        CHECK(memory.write(0x1084, bytes).ok());
        return {};
    }
};

bool same_draw(const DrawDescription& left, const DrawDescription& right)
{
    return left.indexed == right.indexed && left.topology == right.topology &&
           left.first == right.first && left.count == right.count &&
           left.base_vertex == right.base_vertex &&
           left.base_instance == right.base_instance &&
           left.instance_count == right.instance_count &&
           left.instance_index == right.instance_index;
}

struct RegisterStops {
    std::vector<RenderStop> stops;
    static CommandResult record(void* raw, const RenderStop& stop,
                                const GpuAddressSpace&) noexcept {
        static_cast<RegisterStops*>(raw)->stops.push_back(stop);
        return {};
    }
};

void test_render_stop_register_deltas()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    RegisterStops capture;
    auto callbacks = completion.callbacks();
    callbacks.rendering.context = &capture;
    callbacks.rendering.draw = RegisterStops::record;
    callbacks.rendering.clear = RegisterStops::record;
    callbacks.rendering.dispatch = RegisterStops::record;
    callbacks.rendering.blit = RegisterStops::record;
    EngineFrontEnd engines;
    const auto write = [&](std::uint32_t method, std::uint32_t argument,
                           std::uint32_t engine = engine_class::threed) {
        CHECK(engines.call(call(method, argument), engine, memory, callbacks).ok());
    };
    const auto draw = [&] { write(0x586, 4); write(0x585, 0); };
    const auto full = [](const RenderStop& stop) {
        return std::all_of(stop.changed_registers.begin(), stop.changed_registers.end(),
            [](auto word) { return word == std::numeric_limits<std::uint64_t>::max(); });
    };
    const auto changed = [](const RenderStop& stop, std::size_t method) {
        return (stop.changed_registers[method / 64] & (std::uint64_t{1} << (method % 64))) != 0;
    };
    draw();
    CHECK(full(capture.stops.back()));
    CHECK(capture.stops.back().register_stream != 0);
    CHECK(capture.stops.back().register_serial == 1);
    const auto stream = capture.stops.back().register_stream;
    draw();
    CHECK(std::all_of(capture.stops.back().changed_registers.begin(),
                      capture.stops.back().changed_registers.end(), [](auto word) { return word == 0; }));
    CHECK(capture.stops[0].registers == capture.stops[1].registers);
    write(0x35d, 17);
    write(0x702, 0x1234);
    write(0x702, 0x1234);
    draw();
    CHECK(changed(capture.stops.back(), 0x35d));
    CHECK(changed(capture.stops.back(), 0x702));
    CHECK(!changed(capture.stops.back(), 0x703));
    CHECK(capture.stops.back().registers[0x35d] == 17);
    CHECK(capture.stops.back().registers[0x286] == 0x6420);
    // A clear resets the copied-register mask but not the draw-state sequence.
    write(0x674, 0x3c);
    CHECK(capture.stops.back().kind == RenderKind::Clear);
    CHECK(changed(capture.stops.back(), 0x674));
    draw();
    CHECK(capture.stops.back().state_serial == 4);
    CHECK(capture.stops.back().register_serial == 5);
    CHECK(!changed(capture.stops.back(), 0x674));
    // Every engine switch and every non-3D stop carries a full mask.
    write(0xad, 0x12, engine_class::compute);
    write(0xaf, 1, engine_class::compute);
    CHECK(capture.stops.back().kind == RenderKind::Compute && full(capture.stops.back()));
    write(0xaf, 1, engine_class::compute);
    CHECK(full(capture.stops.back()));
    write(0x237, 1, engine_class::twod);
    CHECK(capture.stops.back().kind == RenderKind::Blit && full(capture.stops.back()));
    draw();
    CHECK(full(capture.stops.back()));
    CHECK(capture.stops.back().registers[0x702] == 0x1234);
    CHECK(capture.stops.back().registers[0x286] == 0x6420);
    draw();
    CHECK(!full(capture.stops.back()));
    for (std::size_t index = 0; index < capture.stops.size(); ++index) {
        const auto& stop = capture.stops[index];
        CHECK(stop.register_stream == stream && stop.register_serial == index + 1);
        if (index == 0) continue;
        for (std::size_t method = 0; method < stop.registers.size(); ++method)
            if (stop.registers[method] != capture.stops[index - 1].registers[method])
                CHECK(changed(stop, method));
    }
    // A receiver that skipped a stop can detect the missing delta even if its
    // two observed stops are draws from the same stream.
    CHECK(capture.stops[4].register_stream == capture.stops[2].register_stream);
    CHECK(capture.stops[4].register_serial != capture.stops[2].register_serial + 1);
    // A reset front end has a new identity and starts with a full snapshot.
    auto reset = std::make_unique<EngineFrontEnd>();
    CHECK(reset->call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(reset->call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(capture.stops.back().register_stream != stream);
    CHECK(capture.stops.back().register_serial == 1 && full(capture.stops.back()));
    const RenderStop opaque;
    CHECK(opaque.register_stream == 0 && opaque.register_serial == 0);
}

void test_render_stop_copy_metadata()
{
    RenderStop source, destination;
    source.original = call(0x674, 0x3c);
    source.engine = engine_class::compute;
    source.kind = RenderKind::Compute;
    source.registers.fill(0x12345678);
    source.changed_registers.fill(0x513);
    source.register_stream = 71;
    source.register_serial = 19;
    source.effective_render_enable_mode = 5;
    source.color_targets[7] = {0x1000, 21, 22, 23, 24, 25, 26, 27};
    source.depth = {0x2000, 31, 32, 33, 34, 35, true};
    source.programs[5] = {0x3000, 41, 42, true};
    source.constant_buffers[4][17] = {0x4000, 51, true};
    source.dirty_groups = 0xabcdef;
    source.state_stream = 81;
    source.state_serial = 29;
    source.draw = DrawDescription{true, 7, 8, 9, -10, 11, 12, 13};
    source.inline_indices = {21, 22, 23};
    source.compute_descriptor_address = 0x5000;
    source.compute_descriptor.fill(std::byte{0x73});
    source.has_compute_descriptor = true;
    source.compute = ComputeLaunch{};
    source.compute->grid = {61, 62, 63};
    source.compute->block = {64, 65, 66};
    source.compute->register_count = 67;
    destination.registers.fill(0x87654321);
    const auto registers = destination.registers;
    destination.inline_indices.reserve(64);
    const auto capacity = destination.inline_indices.capacity();
    const auto* storage = destination.inline_indices.data();
    destination.copy_metadata(source);
    CHECK(destination.registers == registers);
    CHECK(destination.original.method == source.original.method);
    CHECK(destination.original.argument == source.original.argument);
    CHECK(destination.engine == source.engine && destination.kind == source.kind);
    CHECK(destination.changed_registers == source.changed_registers);
    CHECK(destination.register_stream == 71 && destination.register_serial == 19);
    CHECK(destination.effective_render_enable_mode == 5);
    CHECK(destination.color_targets[7].address == 0x1000 && destination.color_targets[7].base_layer == 27);
    CHECK(destination.depth.address == 0x2000 && destination.depth.enabled);
    CHECK(destination.programs[5].address == 0x3000 && destination.programs[5].control == 42);
    CHECK(destination.constant_buffers[4][17].address == 0x4000 && destination.constant_buffers[4][17].enabled);
    CHECK(destination.dirty_groups == source.dirty_groups);
    CHECK(destination.state_stream == 81 && destination.state_serial == 29);
    CHECK(destination.draw && same_draw(*destination.draw, *source.draw));
    CHECK(destination.inline_indices == source.inline_indices);
    CHECK(destination.inline_indices.capacity() == capacity && destination.inline_indices.data() == storage);
    CHECK(destination.compute_descriptor_address == 0x5000);
    CHECK(destination.compute_descriptor == source.compute_descriptor && destination.has_compute_descriptor);
    CHECK(destination.compute && destination.compute->grid == source.compute->grid);
    CHECK(destination.compute && destination.compute->block == source.compute->block);
    CHECK(destination.compute && destination.compute->register_count == 67);
    destination.copy_metadata(destination);
    CHECK(destination.inline_indices == source.inline_indices && destination.registers == registers);
    source.draw.reset();
    source.compute.reset();
    source.inline_indices.clear();
    source.has_compute_descriptor = false;
    destination.copy_metadata(source);
    CHECK(!destination.draw && !destination.compute && !destination.has_compute_descriptor);
    CHECK(destination.inline_indices.empty() && destination.inline_indices.capacity() == capacity);
    CHECK(destination.registers == registers);
}

void test_draw_index_selection_and_synchronous_completion()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    EngineFrontEnd engines;
    const auto write = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call(call(method, argument), engine_class::threed, memory, callbacks);
    };
    // Index-count selection survives both a following vertex-count write and
    // Begin. It is an event latch, not a comparison of retained counts.
    CHECK(write(0x5f8, 9).ok());
    CHECK(write(0x5f7, 3).ok());
    CHECK(write(0x35d, 17).ok());
    CHECK(write(0x35e, 12).ok());
    CHECK(write(0x50d, 0xfffffffbu).ok());
    CHECK(write(0x50e, 6).ok());
    CHECK(write(0x586, 4).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(renderer.calls == 1);
    CHECK(renderer.draws[0].indexed);
    CHECK(renderer.draws[0].first == 3 && renderer.draws[0].count == 9);
    CHECK(renderer.draws[0].base_vertex == -5);
    CHECK(renderer.draws[0].base_instance == 6 && renderer.draws[0].instance_count == 1);
    CHECK(renderer.draws[0].instance_index == 0);
    CHECK(renderer.draws[0].topology == 4);
    CHECK(renderer.condition == 1 && renderer.swizzle == 0x6420 && renderer.raster == 1);
    CHECK(!engines.rendering_stop());
    CHECK(backing.bytes[0x84] == std::byte{0x21});

    // The next normal draw is an array draw despite the retained index count.
    CHECK(write(0x586, 4).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(renderer.calls == 2 && !renderer.draws[1].indexed);
    CHECK(renderer.draws[1].first == 17 && renderer.draws[1].count == 12);
    CHECK(renderer.draws[1].base_vertex == 0);

    // Zero count still selects indexed mode and reaches the backend contract.
    CHECK(write(0x586, 4).ok());
    CHECK(write(0x5f8, 0).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(renderer.calls == 3 && renderer.draws[2].indexed && renderer.draws[2].count == 0);
    CHECK(!engines.rendering_stop());
    CHECK(engines.call(call(0x1d, (3u << 8) | 1u), 0, memory, callbacks).ok());
    CHECK(completion.increment_count == 1);
    CHECK(write(0x585, 0).status == CommandStatus::Stopped);
    CHECK(renderer.calls == 3 && engines.rendering_stop() && !engines.rendering_stop()->draw);
}

void test_draw_stops_carry_changed_register_groups()
{
    namespace dirty = armigo::graphics::threed_dirty;
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    EngineFrontEnd engines;
    EngineFrontEnd other;
    const auto write = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call(call(method, argument), engine_class::threed, memory, callbacks);
    };
    const auto draw = [&] {
        CHECK(write(0x586, 4).ok());
        CHECK(write(0x585, 0).ok());
    };
    draw();
    CHECK(renderer.calls == 1 && renderer.dirty[0] == dirty::all);
    CHECK(renderer.streams[0] != 0 && renderer.serials[0] == 1);

    // Draw parameters, constant-buffer selection/data and rewrites of the
    // stored value change no group.
    CHECK(write(0x35d, 3).ok());
    CHECK(write(0x50e, 2).ok());
    CHECK(write(0x8e0, 0x100).ok());
    CHECK(write(0x8e2, 0x4000).ok());
    CHECK(write(0x800 + 0x10 * 5 + 1, 0).ok());
    draw();
    CHECK(renderer.calls == 2 && renderer.dirty[1] == 0);
    CHECK(renderer.streams[1] == renderer.streams[0] && renderer.serials[1] == 2);

    // Changed values mark exactly their groups, and a bind its stage.
    CHECK(write(0x800 + 0x10 * 5 + 1, 0x40).ok());
    CHECK(write(0x200, 0x12).ok());
    CHECK(write(0x702, 0x1000).ok());
    CHECK(write(0x55d, 0x34).ok());
    CHECK(write(0x904 + 4 * 8, (3u << 4) | 1u).ok());
    draw();
    CHECK(renderer.dirty[2] == (dirty::bit(dirty::Shaders) | dirty::bit(dirty::RenderTargets) |
                                dirty::bit(dirty::VertexStreams) | dirty::bit(dirty::TexturePools) |
                                dirty::bit(dirty::ConstantBuffers0 + 4)));
    // Rebinding the same window is not a change; other state is.
    CHECK(write(0x904 + 4 * 8, (3u << 4) | 1u).ok());
    CHECK(write(0x4b9, 1).ok());
    draw();
    CHECK(renderer.dirty[3] == dirty::bit(dirty::State));
    // A separate front end is a separate stream.
    CHECK(other.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(other.call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(renderer.calls == 5 && renderer.streams[4] != renderer.streams[0] &&
          renderer.serials[4] == 1 && renderer.dirty[4] == dirty::all);
}

void test_draw_failure_and_unproven_triggers_remain_sticky()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    EngineFrontEnd bare;
    CHECK(bare.call(call(0x5f8, 9), engine_class::threed, memory, callbacks).ok());
    CHECK(bare.call(call(0x585, 0), engine_class::threed, memory, callbacks).status == CommandStatus::Stopped);
    CHECK(renderer.calls == 0 && bare.rendering_stop() && !bare.rendering_stop()->draw);

    EngineFrontEnd other_trigger;
    CHECK(other_trigger.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(other_trigger.call(call(0x42b, 7), engine_class::threed, memory, callbacks).status == CommandStatus::Stopped);
    CHECK(renderer.calls == 0 && !other_trigger.rendering_stop()->draw);

    EngineFrontEnd failed;
    renderer.fail = true;
    CHECK(failed.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(failed.call(call(0x5f8, 9), engine_class::threed, memory, callbacks).ok());
    const auto failure = failed.call(call(0x585, 0), engine_class::threed, memory, callbacks);
    CHECK(failure.status == CommandStatus::MemoryFault && failure.method == 0x585);
    CHECK(failed.rendering_stop() && failed.rendering_stop()->draw);
    CHECK(renderer.calls == 1);
    CHECK(failed.call(call(0x1d, (3u << 8) | 1u), 0, memory, callbacks).status == CommandStatus::Stopped);
    CHECK(completion.increment_count == 0);
    CHECK(failed.call(call(0x585, 0), engine_class::threed, memory, callbacks).status == CommandStatus::Stopped);
    CHECK(renderer.calls == 1);

    EngineFrontEnd no_backend;
    CHECK(no_backend.call(call(0x586, 4), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(no_backend.call(call(0x35e, 12), engine_class::threed, memory, completion.callbacks()).ok());
    CHECK(no_backend.call(call(0x585, 0), engine_class::threed, memory, completion.callbacks()).status == CommandStatus::Stopped);
    CHECK(no_backend.rendering_stop()->draw && !no_backend.rendering_stop()->draw->indexed);
}

void test_draw_bounds_and_begin_controls()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    for (const auto begin : {4u | (1u << 29), 4u | (1u << 24)}) {
        EngineFrontEnd engines;
        CHECK(engines.call(call(0x586, begin), engine_class::threed, memory, callbacks).status == CommandStatus::Unsupported);
        CHECK(engines.call(call(0x586, 4), engine_class::threed, memory, callbacks).status == CommandStatus::Stopped);
        CHECK(engines.call(call(0x585, 0), engine_class::threed, memory, callbacks).status == CommandStatus::Stopped);
        CHECK(engines.rendering_stop() && !engines.rendering_stop()->draw);
    }
    CHECK(renderer.calls == 0);
    EngineFrontEnd planner_topology;
    CHECK(planner_topology.call(call(0x586, 5), engine_class::threed, memory, callbacks).ok());
    CHECK(planner_topology.call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(renderer.calls == 1 && renderer.draws[0].topology == 5);
    for (const bool indexed : {false, true}) {
        EngineFrontEnd engines;
        CHECK(engines.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
        CHECK(engines.call(call(indexed ? 0x5f7 : 0x35d, std::numeric_limits<std::uint32_t>::max()),
                           engine_class::threed, memory, callbacks).ok());
        CHECK(engines.call(call(indexed ? 0x5f8 : 0x35e, 2), engine_class::threed, memory, callbacks).ok());
        CHECK(engines.call(call(0x585, 0), engine_class::threed, memory, callbacks).status == CommandStatus::Unsupported);
        CHECK(engines.rendering_stop());
    }
    CHECK(renderer.calls == 1);
    EngineFrontEnd override_topology;
    CHECK(override_topology.call(call(0x652, 1), engine_class::threed, memory, callbacks).ok());
    CHECK(override_topology.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(override_topology.call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(renderer.calls == 2);
    EngineFrontEnd last_element;
    CHECK(last_element.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(last_element.call(call(0x5f7, std::numeric_limits<std::uint32_t>::max()), engine_class::threed, memory, callbacks).ok());
    CHECK(last_element.call(call(0x5f8, 1), engine_class::threed, memory, callbacks).ok());
    CHECK(last_element.call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(renderer.calls == 3 && renderer.draws[2].first == std::numeric_limits<std::uint32_t>::max());
}

void test_draw_instance_controls_and_overflow()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    EngineFrontEnd engines;
    const auto write = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call(call(method, argument), engine_class::threed, memory, callbacks);
    };
    CHECK(write(0x35d, 2).ok());
    CHECK(write(0x35e, 3).ok());
    CHECK(write(0x50e, 20).ok());

    CHECK(write(0x586, 4).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(write(0x586, (1u << 26) | 5u).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(write(0x586, (1u << 27) | 6u).ok());
    CHECK(write(0x585, 0).ok());
    // Increment takes precedence over the simultaneously requested reset.
    CHECK(write(0x586, (1u << 26) | 7u).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(write(0x586, 8).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(renderer.calls == 5);
    CHECK(renderer.draws[0].base_instance == 20 &&
          renderer.draws[0].instance_index == 0 && renderer.draws[0].topology == 4);
    CHECK(renderer.draws[1].base_instance == 20 &&
          renderer.draws[1].instance_index == 1 && renderer.draws[1].topology == 5);
    CHECK(renderer.draws[2].base_instance == 20 &&
          renderer.draws[2].instance_index == 1 && renderer.draws[2].topology == 6);
    CHECK(renderer.draws[3].base_instance == 20 &&
          renderer.draws[3].instance_index == 2 && renderer.draws[3].topology == 7);
    CHECK(renderer.draws[4].base_instance == 20 &&
          renderer.draws[4].instance_index == 0 && renderer.draws[4].topology == 8);
    for (std::size_t index = 0; index < renderer.calls; ++index)
        CHECK(renderer.draws[index].instance_count == 1);

    Completion overflow_completion;
    DrawRenderer overflow_renderer{&overflow_completion};
    auto overflow_callbacks = overflow_completion.callbacks();
    overflow_callbacks.rendering = {nullptr, &overflow_renderer, DrawRenderer::draw};
    EngineFrontEnd overflow;
    const auto overflow_write = [&](std::uint32_t method, std::uint32_t argument) {
        return overflow.call(
            call(method, argument), engine_class::threed, memory, overflow_callbacks);
    };
    CHECK(overflow_write(0x35e, 1).ok());
    CHECK(overflow_write(0x50e, std::numeric_limits<std::uint32_t>::max()).ok());
    CHECK(overflow_write(0x586, 4).ok());
    CHECK(overflow_write(0x585, 0).ok());
    CHECK(overflow_renderer.calls == 1 &&
          overflow_renderer.draws[0].base_instance ==
              std::numeric_limits<std::uint32_t>::max() &&
          overflow_renderer.draws[0].instance_index == 0);
    CHECK(overflow_write(0x586, (1u << 26) | 4u).ok());
    CHECK(overflow_write(0x585, 0).status == CommandStatus::BudgetExceeded);
    CHECK(overflow_renderer.calls == 1 && overflow.rendering_stop());
    CHECK(overflow_write(0x585, 0).status == CommandStatus::Stopped);
    CHECK(overflow_renderer.calls == 1);
}

void test_compact_draw_forms_latches_and_shadow()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    const auto packed = [](std::uint32_t topology, std::uint32_t count,
                           std::uint32_t first) {
        return (topology << 28) | (count << 16) | first;
    };

    EngineFrontEnd arrays;
    CHECK(arrays.call(call(0x50e, 30), engine_class::threed, memory, callbacks).ok());
    CHECK(arrays.call(call(0x485, packed(2, 0, 7)), engine_class::threed,
                      memory, callbacks).ok());
    CHECK(arrays.call(call(0x486, packed(3, 5, 8)), engine_class::threed,
                      memory, callbacks).ok());
    CHECK(same_draw(renderer.draws[0], {false, 2, 7, 0, 0, 30, 1, 0}));
    CHECK(same_draw(renderer.draws[1], {false, 3, 8, 5, 0, 30, 1, 1}));

    constexpr std::array first_methods{0x5f9u, 0x5fau, 0x5fbu};
    constexpr std::array subsequent_methods{0x5fcu, 0x5fdu, 0x5feu};
    for (std::size_t index = 0; index < first_methods.size(); ++index) {
        EngineFrontEnd indexed;
        CHECK(indexed.call(call(0x50d, 0xfffffffcu), engine_class::threed,
                           memory, callbacks).ok());
        CHECK(indexed.call(call(0x50e, 40), engine_class::threed,
                           memory, callbacks).ok());
        CHECK(indexed.call(call(0x5f6, 2), engine_class::threed,
                           memory, callbacks).ok());
        CHECK(indexed.call(call(first_methods[index], packed(4, 3, 1)),
                           engine_class::threed, memory, callbacks).ok());
        CHECK(indexed.call(call(subsequent_methods[index], packed(5, 4, 2)),
                           engine_class::threed, memory, callbacks).ok());
        const auto first_call = 2 + index * 2;
        CHECK(renderer.methods[first_call] == first_methods[index]);
        CHECK(renderer.methods[first_call + 1] == subsequent_methods[index]);
        CHECK(same_draw(renderer.draws[first_call],
                        {true, 4, 1, 3, -4, 40, 1, 0}));
        CHECK(same_draw(renderer.draws[first_call + 1],
                        {true, 5, 2, 4, -4, 40, 1, 1}));
        CHECK(renderer.index_types[first_call] == 2 &&
              renderer.index_types[first_call + 1] == 2);
    }

    EngineFrontEnd retained_latch;
    CHECK(retained_latch.call(call(0x5f7, 6), engine_class::threed,
                              memory, callbacks).ok());
    CHECK(retained_latch.call(call(0x5f8, 7), engine_class::threed,
                              memory, callbacks).ok());
    CHECK(retained_latch.call(call(0x586, 4), engine_class::threed,
                              memory, callbacks).ok());
    CHECK(retained_latch.call(call(0x485, packed(6, 3, 2)), engine_class::threed,
                              memory, callbacks).ok());
    CHECK(retained_latch.call(call(0x585, 0), engine_class::threed,
                              memory, callbacks).ok());
    CHECK(same_draw(renderer.draws[8], {false, 6, 2, 3, 0, 0, 1}));
    CHECK(renderer.draws[9].indexed && renderer.draws[9].topology == 6 &&
          renderer.draws[9].first == 6 && renderer.draws[9].count == 7);

    EngineFrontEnd no_latch;
    CHECK(no_latch.call(call(0x35d, 9), engine_class::threed, memory, callbacks).ok());
    CHECK(no_latch.call(call(0x35e, 2), engine_class::threed, memory, callbacks).ok());
    CHECK(no_latch.call(call(0x5f9, packed(4, 1, 0)), engine_class::threed,
                        memory, callbacks).ok());
    CHECK(no_latch.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(no_latch.call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(renderer.draws[11].indexed == false && renderer.draws[11].first == 9);

    EngineFrontEnd shadow;
    const auto replayed = packed(3, 4, 5);
    CHECK(shadow.call(call(0x49, 1), engine_class::threed, memory, callbacks).ok());
    CHECK(shadow.call(call(0x485, replayed), engine_class::threed, memory, callbacks).ok());
    CHECK(shadow.call(call(0x49, 3), engine_class::threed, memory, callbacks).ok());
    CHECK(shadow.call(call(0x485, packed(7, 8, 9)), engine_class::threed,
                      memory, callbacks).ok());
    CHECK(same_draw(renderer.draws[12], renderer.draws[13]));
    CHECK(same_draw(renderer.draws[13], {false, 3, 5, 4, 0, 0, 1}));
}

void test_inline_index_ownership_consumption_and_limits()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    EngineFrontEnd engines;
    const auto write = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call(call(method, argument), engine_class::threed, memory, callbacks);
    };
    CHECK(write(0x50d, 0xfffffffdu).ok());
    CHECK(write(0x50e, 12).ok());
    CHECK(write(0x4c1, 0x44332211).ok());
    CHECK(write(0x57a, 0xdeadbeef).ok());
    CHECK(write(0x57c, 0xbbaa6655).ok());
    CHECK(write(0x586, 4).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(renderer.calls == 1);
    CHECK(same_draw(renderer.draws[0], {true, 4, 0, 7, -3, 12, 1}));
    CHECK(renderer.inline_sizes[0] == 7);
    const std::array expected{0x11u, 0x22u, 0x33u, 0x44u,
                              0xdeadbeefu, 0x6655u, 0xbbaau};
    for (std::size_t index = 0; index < expected.size(); ++index)
        CHECK(renderer.inline_values[0][index] == expected[index]);

    // A reset Begin starts a new sequence; the consumed stream is not reused.
    CHECK(write(0x35d, 3).ok());
    CHECK(write(0x35e, 2).ok());
    CHECK(write(0x586, 4).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(renderer.calls == 2 && renderer.inline_sizes[1] == 0 &&
          !renderer.draws[1].indexed);

    EngineFrontEnd nonzero_first;
    CHECK(nonzero_first.call(call(0x4c1, 0x03020100), engine_class::threed,
                             memory, callbacks).ok());
    CHECK(nonzero_first.call(call(0x5f7, 1), engine_class::threed,
                             memory, callbacks).ok());
    CHECK(nonzero_first.call(call(0x586, 4), engine_class::threed,
                             memory, callbacks).ok());
    CHECK(nonzero_first.call(call(0x585, 0), engine_class::threed,
                             memory, callbacks).status == CommandStatus::Unsupported);
    CHECK(renderer.calls == 2 && nonzero_first.rendering_stop() &&
          nonzero_first.rendering_stop()->inline_indices.size() == 4);

    EngineFrontEnd consumed_continuation;
    CHECK(consumed_continuation.call(call(0x57a, 7), engine_class::threed,
                                     memory, callbacks).ok());
    CHECK(consumed_continuation.call(call(0x586, 4), engine_class::threed,
                                     memory, callbacks).ok());
    CHECK(consumed_continuation.call(call(0x585, 0), engine_class::threed,
                                     memory, callbacks).ok());
    CHECK(renderer.calls == 3);
    CHECK(consumed_continuation.call(call(0x586, (1u << 26) | 4u),
                                     engine_class::threed, memory, callbacks).status ==
          CommandStatus::Unsupported);
    CHECK(renderer.calls == 3 && consumed_continuation.rendering_stop());

    EngineFrontEnd live_continuation;
    CHECK(live_continuation.call(call(0x586, (1u << 27) | 4u),
                                 engine_class::threed, memory, callbacks).ok());
    CHECK(live_continuation.call(call(0x57a, 8), engine_class::threed,
                                 memory, callbacks).ok());
    CHECK(live_continuation.call(call(0x585, 0), engine_class::threed,
                                 memory, callbacks).status == CommandStatus::Unsupported);
    CHECK(renderer.calls == 3 && live_continuation.rendering_stop() &&
          live_continuation.rendering_stop()->inline_indices.size() == 1);

    Completion failed_completion;
    DrawRenderer failing{&failed_completion};
    failing.fail = true;
    auto failed_callbacks = failed_completion.callbacks();
    failed_callbacks.rendering = {nullptr, &failing, DrawRenderer::draw};
    EngineFrontEnd failed;
    CHECK(failed.call(call(0x57a, 0x1234), engine_class::threed,
                      memory, failed_callbacks).ok());
    CHECK(failed.call(call(0x586, 4), engine_class::threed,
                      memory, failed_callbacks).ok());
    CHECK(failed.call(call(0x585, 0), engine_class::threed,
                      memory, failed_callbacks).status == CommandStatus::MemoryFault);
    CHECK(failed.rendering_stop() && failed.rendering_stop()->inline_indices.size() == 1);
    CHECK(failed.call(call(0x57a, 0x5678), engine_class::threed,
                      memory, failed_callbacks).status == CommandStatus::Stopped);
    CHECK(failing.calls == 1);

    EngineFrontEnd limited;
    bool filled = true;
    for (std::size_t index = 0; index < (1u << 18); ++index) {
        if (!limited.call(call(0x4c1, 0), engine_class::threed,
                          memory, callbacks).ok()) {
            filled = false;
            break;
        }
    }
    CHECK(filled);
    CHECK(limited.call(call(0x57a, 0), engine_class::threed,
                       memory, callbacks).status ==
          CommandStatus::BudgetExceeded);
    const auto calls_before_sticky_check = renderer.calls;
    CHECK(limited.call(call(0x586, 4), engine_class::threed,
                       memory, callbacks).status == CommandStatus::Stopped);
    CHECK(renderer.calls == calls_before_sticky_check);
}

void test_nonzero_effective_draw_end_is_unsupported()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x5f8, 3), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x585, 1), engine_class::threed, memory, callbacks).status ==
          CommandStatus::Unsupported);
    CHECK(renderer.calls == 0 && engines.rendering_stop());
    CHECK(engines.call(call(0x1d, (3u << 8) | 1u), 0, memory, callbacks).status ==
          CommandStatus::Stopped);
    CHECK(completion.increment_count == 0);
}

void test_draw_effective_shadow_replay_and_explicit_defaults()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    DrawRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, DrawRenderer::draw};
    EngineFrontEnd engines;
    const auto write = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call(call(method, argument), engine_class::threed, memory, callbacks);
    };
    CHECK(write(0x556, 0).ok());
    CHECK(write(0x286, 0x1234).ok());
    CHECK(write(0xdf, 0).ok());
    CHECK(write(0x586, 4).ok());
    CHECK(write(0x5f8, 9).ok());
    CHECK(write(0x585, 0).ok());
    CHECK(renderer.condition == 0 && renderer.swizzle == 0x1234 && renderer.raster == 0);
    CHECK(write(0x49, 3).ok());
    CHECK(write(0x556, 1).ok());
    CHECK(write(0x286, 0x6420).ok());
    CHECK(write(0xdf, 1).ok());
    CHECK(write(0x586, 0xffffffffu).ok());
    CHECK(write(0x5f8, 0).ok());
    CHECK(write(0x585, 0x123u).ok());
    CHECK(renderer.calls == 2 && renderer.draws[1].indexed && renderer.draws[1].count == 9);
    CHECK(renderer.raw_end == 0x123u && renderer.raw_count == 9);
    CHECK(renderer.condition == 0 && renderer.swizzle == 0x1234 && renderer.raster == 0);
    CHECK(!engines.rendering_stop());
}

// Records the coherence requests the front end issues while a backend defers
// GPU write-back, plus the guest bytes observed when each request arrived.
struct CoherenceRenderer {
    struct Request {
        std::uint64_t gpu_address = 0;
        std::uint64_t size = 0;
        std::array<std::byte, 4> observed{};
    };

    Completion* completion = nullptr;
    std::vector<Request> coherence_calls{};
    unsigned draws = 0;
    bool fail_coherence = false;

    static CommandResult draw(void* raw, const RenderStop& snapshot,
                              const GpuAddressSpace& memory) noexcept
    {
        auto& self = *static_cast<CoherenceRenderer*>(raw);
        CHECK(snapshot.kind == RenderKind::Draw);
        CHECK(snapshot.draw.has_value());
        ++self.draws;
        const std::array marker{std::byte{0x21}, std::byte{0x43},
                                std::byte{0x65}, std::byte{0x87}};
        CHECK(memory.write(0x1084, marker).ok());
        return {};
    }

    static CommandResult coherence(void* raw, const GpuAddressSpace& memory,
                                   std::uint64_t gpu_address,
                                   std::uint64_t size) noexcept
    {
        auto& self = *static_cast<CoherenceRenderer*>(raw);
        if (self.fail_coherence) {
            return {CommandStatus::MemoryFault, "Synthetic coherence failure"};
        }
        Request request{gpu_address, size};
        const std::uint64_t probe = gpu_address != 0 ? gpu_address : 0x1088;
        CHECK(memory.read(probe, request.observed));
        self.coherence_calls.push_back(request);
        return {};
    }
};

// Rendering is followed by coherence at exactly the guest-visible points: the
// semaphore release, the syncpoint increments, and each upload word the front
// end is about to write (ranged). Ranged coherence leaves the pending latch
// set; only a full request clears it.
void test_deferred_writeback_coherence_points()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    CoherenceRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, CoherenceRenderer::draw};
    callbacks.rendering.coherence = CoherenceRenderer::coherence;
    EngineFrontEnd engines;
    const auto threed = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call(call(method, argument), engine_class::threed, memory, callbacks);
    };
    const auto puller = [&](std::uint32_t method, std::uint32_t argument) {
        return engines.call(call(method, argument), 0, memory, callbacks);
    };
    const auto draw = [&] {
        CHECK(threed(0x586, 4).ok());
        CHECK(threed(0x585, 0).ok());
    };

    // Nothing rendered yet: guest-visible effects run without coherence calls.
    CHECK(puller(0x4, 0).ok());
    CHECK(puller(0x5, 0x1088).ok());
    CHECK(puller(0x1b, 0x11111111).ok());
    CHECK(renderer.coherence_calls.empty());
    CHECK(backing.bytes[0x88] == std::byte{0x11});

    // Draw then semaphore release: one full coherence before the write.
    CHECK(threed(0x35d, 2).ok());
    CHECK(threed(0x35e, 3).ok());
    draw();
    CHECK(renderer.draws == 1);
    CHECK(puller(0x1b, 0x12345678).ok());
    CHECK(renderer.coherence_calls.size() == 1);
    CHECK(renderer.coherence_calls.front().gpu_address == 0);
    CHECK(renderer.coherence_calls.front().size == 0);
    const std::array<std::byte, 4> previous_semaphore{
        std::byte{0x11}, std::byte{0x11}, std::byte{0x11}, std::byte{0x11}};
    CHECK(renderer.coherence_calls.front().observed == previous_semaphore);
    CHECK(backing.bytes[0x88] == std::byte{0x78});

    // A second release without an intervening draw issues no further call.
    CHECK(puller(0x1b, 0x3456789a).ok());
    CHECK(renderer.coherence_calls.size() == 1);
    CHECK(backing.bytes[0x88] == std::byte{0x9a});

    // Draw then constant-buffer upload: one ranged call per uploaded word,
    // covering exactly the destination of that word.
    draw();
    CHECK(threed(0x8e0, 8).ok());
    CHECK(threed(0x8e1, 0).ok());
    CHECK(threed(0x8e2, 0x1080).ok());
    CHECK(threed(0x8e3, 0).ok());
    CHECK(threed(0x8e4, 0x11223344).ok());
    CHECK(renderer.coherence_calls.size() == 2);
    CHECK(renderer.coherence_calls.back().gpu_address == 0x1080);
    CHECK(renderer.coherence_calls.back().size == 4);
    const std::array<std::byte, 4> untouched_bytes{};
    CHECK(renderer.coherence_calls.back().observed == untouched_bytes);
    CHECK(backing.bytes[0x80] == std::byte{0x44});
    CHECK(threed(0x8e5, 0xaabbccdd).ok());
    CHECK(renderer.coherence_calls.size() == 3);
    CHECK(renderer.coherence_calls.back().gpu_address == 0x1084);
    CHECK(renderer.coherence_calls.back().size == 4);
    // The draw's own bytes were still in place when coherence ran.
    const std::array<std::byte, 4> draw_marker{
        std::byte{0x21}, std::byte{0x43}, std::byte{0x65}, std::byte{0x87}};
    CHECK(renderer.coherence_calls.back().observed == draw_marker);
    CHECK(backing.bytes[0x84] == std::byte{0xdd});

    // A ranged request leaves rendering pending: the next guest-visible point
    // still requests coherence without a new draw.
    CHECK(puller(0x1d, (3u << 8) | 1u).ok());
    CHECK(renderer.coherence_calls.size() == 4);
    CHECK(renderer.coherence_calls.back().gpu_address == 0);
    CHECK(renderer.coherence_calls.back().size == 0);
    CHECK(completion.increment_count == 1);
    CHECK(completion.increments[0] == 3);

    // Draw then inline upload: the request arrives before the final word.
    draw();
    CHECK(threed(0x60, 4).ok());
    CHECK(threed(0x61, 1).ok());
    CHECK(threed(0x62, 0).ok());
    CHECK(threed(0x63, 0x1090).ok());
    CHECK(threed(0x6a, 0).ok());
    CHECK(threed(0x6b, 0).ok());
    CHECK(threed(0x6c, 0x1011).ok());
    CHECK(renderer.coherence_calls.size() == 4);
    CHECK(threed(0x6d, 0x44332211).ok());
    CHECK(renderer.coherence_calls.size() == 5);
    CHECK(renderer.coherence_calls.back().gpu_address == 0x1090);
    CHECK(renderer.coherence_calls.back().size == 4);
    CHECK(renderer.coherence_calls.back().observed == untouched_bytes);
    CHECK(backing.bytes[0x90] == std::byte{0x11});

    // Syncpoint increments after a draw request full coherence.
    draw();
    CHECK(puller(0x1d, (4u << 8) | 1u).ok());
    CHECK(renderer.coherence_calls.size() == 6);
    CHECK(renderer.coherence_calls.back().gpu_address == 0);
    CHECK(renderer.coherence_calls.back().size == 0);
    CHECK(completion.increments[1] == 4);
    draw();
    CHECK(threed(0xb2, 0x1234).ok());
    CHECK(renderer.coherence_calls.size() == 7);
    CHECK(renderer.coherence_calls.back().gpu_address == 0);
    CHECK(renderer.coherence_calls.back().size == 0);
    CHECK(completion.increment_count == 3);
    CHECK(completion.increments[2] == 0x1234);
    CHECK(renderer.draws == 5);
}

// A failing coherence callback stops the method with that error, leaves guest
// memory untouched, and stays retryable because rendering is still pending.
void test_coherence_failure_stops_method_and_retry_succeeds()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    CoherenceRenderer renderer{&completion};
    renderer.fail_coherence = true;
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, CoherenceRenderer::draw};
    callbacks.rendering.coherence = CoherenceRenderer::coherence;
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x35e, 2), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x4, 0), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x5, 0x1088), 0, memory, callbacks).ok());
    const auto stopped = engines.call(call(0x1b, 0x12345678), 0, memory, callbacks);
    CHECK(stopped.status == CommandStatus::MemoryFault);
    CHECK(renderer.coherence_calls.empty());
    CHECK(renderer.draws == 1);
    CHECK(backing.bytes[0x88] == std::byte{0});

    renderer.fail_coherence = false;
    CHECK(engines.call(call(0x1b, 0x12345678), 0, memory, callbacks).ok());
    CHECK(renderer.coherence_calls.size() == 1);
    CHECK(renderer.coherence_calls.front().gpu_address == 0);
    CHECK(renderer.coherence_calls.front().size == 0);
    CHECK(renderer.draws == 1);
    CHECK(backing.bytes[0x88] == std::byte{0x78});
}

// Without a coherence callback every method behaves exactly as before the
// deferred write-back option existed.
void test_front_end_without_coherence_callback_unchanged()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    CoherenceRenderer renderer{&completion};
    auto callbacks = completion.callbacks();
    callbacks.rendering = {nullptr, &renderer, CoherenceRenderer::draw};
    EngineFrontEnd engines;
    CHECK(engines.call(call(0x4, 0), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x5, 0x1088), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x35e, 2), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x586, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x585, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(renderer.draws == 1);
    CHECK(engines.call(call(0x1b, 0x12345678), 0, memory, callbacks).ok());
    CHECK(backing.bytes[0x88] == std::byte{0x78});
    CHECK(engines.call(call(0x1d, (3u << 8) | 1u), 0, memory, callbacks).ok());
    CHECK(completion.increment_count == 1);
    CHECK(engines.call(call(0x8e0, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e1, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e2, 0x1080), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e3, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e4, 0x11223344), engine_class::threed, memory, callbacks).ok());
    CHECK(backing.bytes[0x80] == std::byte{0x44});
    CHECK(engines.call(call(0x60, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x61, 1), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x63, 0x1090), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x6c, 0x1011), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x6d, 0x44332211), engine_class::threed, memory, callbacks).ok());
    CHECK(backing.bytes[0x90] == std::byte{0x11});
    CHECK(renderer.coherence_calls.empty());
}

void test_ordered_front_end_writes()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    OrderedWriteQueue queue;
    auto callbacks = completion.callbacks();
    callbacks.rendering.context = &queue;
    callbacks.rendering.ordered_write = OrderedWriteQueue::enqueue;
    callbacks.rendering.coherence = OrderedWriteQueue::coherence;
    callbacks.rendering.read_coherence = OrderedWriteQueue::read_coherence;
    EngineFrontEnd engines;

    const auto upload = [&](std::uint32_t engine, std::uint64_t destination) {
        CHECK(engines.call(call(0x60, 6), engine, memory, callbacks).ok());
        CHECK(engines.call(call(0x61, 1), engine, memory, callbacks).ok());
        CHECK(engines.call(call(0x62, static_cast<std::uint32_t>(destination >> 32)),
                           engine, memory, callbacks).ok());
        CHECK(engines.call(call(0x63, static_cast<std::uint32_t>(destination)),
                           engine, memory, callbacks).ok());
        CHECK(engines.call(call(0x6a, 0), engine, memory, callbacks).ok());
        CHECK(engines.call(call(0x6b, 0), engine, memory, callbacks).ok());
        CHECK(engines.call(call(0x6c, 0x1011), engine, memory, callbacks).ok());
        CHECK(engines.call(call(0x6d, 0x44332211), engine, memory, callbacks).ok());
        CHECK(queue.writes.empty() || queue.writes.back().address != destination);
        CHECK(engines.call(call(0x6d, 0xcafebeef), engine, memory, callbacks).ok());
        CHECK(!queue.writes.empty());
        if (queue.writes.empty()) return;
        CHECK(queue.writes.back().address == destination);
        CHECK(queue.writes.back().payload.size() == 6);
        const std::array expected{std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
                                  std::byte{0x44}, std::byte{0xef}, std::byte{0xbe}};
        CHECK(std::equal(queue.writes.back().payload.begin(), queue.writes.back().payload.end(),
                         expected.begin(), expected.end()));
    };

    upload(engine_class::threed, 0x1040);
    upload(engine_class::compute, 0x1048);
    upload(engine_class::inline_to_memory, 0x1050);

    CHECK(engines.call(call(0x8e0, 8), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e1, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e2, 0x1060), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e3, 0), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e4, 0x88776655), engine_class::threed, memory, callbacks).ok());
    CHECK(engines.call(call(0x8e5, 0xccbbaa99), engine_class::threed, memory, callbacks).ok());
    CHECK(queue.writes.size() == 5);
    if (queue.writes.size() == 5) {
        CHECK(queue.writes[3].address == 0x1060);
        CHECK(queue.writes[4].address == 0x1064);
    }
    CHECK(queue.coherence_calls == 0);
    CHECK(queue.read_coherence_calls == 0);
    CHECK(backing.bytes[0x60] == std::byte{0});

    CHECK(engines.call(call(0x4, 0), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x5, 0x1060), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x1a, 0x88776655), 0, memory, callbacks).ok());
    CHECK(queue.writes.size() == 1);
    if (queue.writes.size() == 1) CHECK(queue.writes.front().address == 0x1064);
    CHECK(queue.read_coherence_calls == 1);
    CHECK(queue.coherence_calls == 0);
    CHECK(backing.bytes[0x60] == std::byte{0x55});
    CHECK(backing.bytes[0x64] == std::byte{0});
    CHECK(engines.call(call(0x5, 0x1064), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x1a, 0xccbbaa99), 0, memory, callbacks).ok());
    CHECK(queue.writes.empty());
    CHECK(queue.read_coherence_calls == 2);
    CHECK(backing.bytes[0x64] == std::byte{0x99});

    OrderedWriteQueue fallback_queue;
    auto fallback_callbacks = completion.callbacks();
    fallback_callbacks.rendering.context = &fallback_queue;
    fallback_callbacks.rendering.ordered_write = OrderedWriteQueue::enqueue;
    fallback_callbacks.rendering.coherence = OrderedWriteQueue::coherence;
    EngineFrontEnd fallback;
    CHECK(fallback.call(call(0x8e0, 4), engine_class::threed, memory, fallback_callbacks).ok());
    CHECK(fallback.call(call(0x8e2, 0x1070), engine_class::threed, memory, fallback_callbacks).ok());
    CHECK(fallback.call(call(0x8e4, 0x12345678), engine_class::threed, memory,
                        fallback_callbacks).ok());
    CHECK(fallback.call(call(0x4, 0), 0, memory, fallback_callbacks).ok());
    CHECK(fallback.call(call(0x5, 0x1070), 0, memory, fallback_callbacks).ok());
    CHECK(fallback.call(call(0x1a, 0x12345678), 0, memory, fallback_callbacks).ok());
    CHECK(fallback_queue.coherence_calls == 1);
    CHECK(fallback_queue.writes.empty());
}

void test_ordered_write_error_is_preserved()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    OrderedWriteQueue queue;
    queue.failure = CommandResult{CommandStatus::InvalidHeader, "Synthetic ordered-write failure",
                                  0x12345678, 9, 7, 6, 5, 4};
    auto callbacks = completion.callbacks();
    callbacks.rendering.context = &queue;
    callbacks.rendering.ordered_write = OrderedWriteQueue::enqueue;

    EngineFrontEnd constant_buffer;
    CHECK(constant_buffer.call(call(0x8e0, 4), engine_class::threed, memory, callbacks).ok());
    CHECK(constant_buffer.call(call(0x8e2, 0x1080), engine_class::threed, memory, callbacks).ok());
    const auto cb_error =
        constant_buffer.call(call(0x8e4, 1), engine_class::threed, memory, callbacks);
    CHECK(cb_error.status == CommandStatus::InvalidHeader);
    CHECK(cb_error.detail == "Synthetic ordered-write failure");
    CHECK(cb_error.address == 0x12345678);
    CHECK(cb_error.entry_index == 9);
    CHECK(cb_error.subchannel == 7);
    CHECK(cb_error.method == 6);
    CHECK(cb_error.argument == 5);
    CHECK(cb_error.engine == 4);

    queue.failure.reset();
    CHECK(constant_buffer.call(call(0x8e4, 0x11223344), engine_class::threed,
                               memory, callbacks).ok());
    CHECK(queue.writes.size() == 1);
    if (queue.writes.size() == 1) CHECK(queue.writes.front().address == 0x1080);
    queue.writes.clear();
    queue.failure = cb_error;

    EngineFrontEnd upload;
    for (const auto [method, argument] :
         {std::pair{0x60u, 4u}, std::pair{0x61u, 1u}, std::pair{0x62u, 0u},
          std::pair{0x63u, 0x1090u}, std::pair{0x6au, 0u}, std::pair{0x6bu, 0u},
          std::pair{0x6cu, 0x1011u}}) {
        CHECK(upload.call(call(method, argument), engine_class::inline_to_memory,
                          memory, callbacks).ok());
    }
    const auto upload_error = upload.call(call(0x6d, 0x44332211),
                                          engine_class::inline_to_memory, memory, callbacks);
    CHECK(upload_error.status == cb_error.status);
    CHECK(upload_error.detail == cb_error.detail);
    CHECK(upload_error.address == cb_error.address);
    CHECK(upload_error.entry_index == cb_error.entry_index);
    CHECK(upload_error.subchannel == cb_error.subchannel);
    CHECK(upload_error.method == cb_error.method);
    CHECK(upload_error.argument == cb_error.argument);
    CHECK(upload_error.engine == cb_error.engine);
    const auto repeated = upload.call(call(0x6d, 0xaabbccdd),
                                      engine_class::inline_to_memory, memory, callbacks);
    CHECK(repeated.status == upload_error.status);
    CHECK(repeated.detail == upload_error.detail);
    CHECK(repeated.address == upload_error.address);
    CHECK(repeated.entry_index == upload_error.entry_index);
    CHECK(repeated.subchannel == upload_error.subchannel);
    CHECK(repeated.method == upload_error.method);
    CHECK(repeated.argument == upload_error.argument);
    CHECK(repeated.engine == upload_error.engine);
    CHECK(upload.has_pending_memory_work());
    CHECK(backing.bytes[0x90] == std::byte{0});
}

void test_ordered_guest_effect_routing_and_owned_increment()
{
    Backing backing;
    auto memory = make_memory(backing);
    Completion completion;
    completion.owned_increment_state = std::make_shared<Completion::OwnedIncrementState>();
    const auto retained_state = completion.owned_increment_state;
    OrderedEffectQueue queue;
    auto callbacks = completion.callbacks();
    callbacks.retain_increment = Completion::retain_increment;
    callbacks.rendering.context = &queue;
    callbacks.rendering.ordered_effect = OrderedEffectQueue::enqueue;
    callbacks.rendering.coherence = OrderedEffectQueue::coherence;
    EngineFrontEnd engines;

    CHECK(engines.call(call(0x4, 0), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x5, 0x1088), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x1b, 0x12345678), 0, memory, callbacks).ok());
    CHECK(engines.increment_syncpoint(memory, callbacks, 7, 3).ok());
    CHECK(queue.effects.size() == 2);
    CHECK(backing.bytes[0x88] == std::byte{0});
    CHECK(completion.increment_count == 0);

    CHECK(queue.effects[0].address == 0x1088);
    CHECK(queue.effects[0].size == 4);
    std::uint32_t captured_payload = 0;
    std::memcpy(&captured_payload, queue.effects[0].payload.data(), sizeof(captured_payload));
    CHECK(captured_payload == 0x12345678);
    CHECK(!queue.effects[0].publish);
    CHECK(queue.effects[1].address == 0);
    CHECK(queue.effects[1].size == 0);
    CHECK(static_cast<bool>(queue.effects[1].publish));

    completion.owned_increment_state.reset();
    CHECK(engines.call(call(0x1c, 1), 0, memory, callbacks).ok());
    CHECK(engines.call(call(0x1d, 9u << 8), 0, memory, callbacks).ok());
    CHECK(queue.coherence_calls == 1);
    CHECK(queue.effects.empty());
    CHECK(completion.expired_id == 9);
    CHECK(completion.expired_threshold == 1);
    CHECK(retained_state->increments.size() == 3);
    CHECK(retained_state->increments[0] == 7);
    CHECK(retained_state->increments[1] == 7);
    CHECK(retained_state->increments[2] == 7);

    CHECK(backing.bytes[0x88] == std::byte{0x78});

    CHECK(engines.call(call(0x1b, 0x87654321), 0, memory, callbacks).ok());
    CHECK(queue.effects.size() == 1);
    CHECK(backing.bytes[0x88] == std::byte{0x78});
    // The acquire is an ordered check behind the release, not a drain.
    CHECK(engines.call(call(0x1a, 0x87654321), 0, memory, callbacks).ok());
    CHECK(queue.effects.size() == 2);
    CHECK(queue.effects[1].address == 0x1088);
    CHECK(queue.effects[1].size == 0);
    CHECK(queue.effects[1].acquire && queue.effects[1].acquire->value == 0x87654321 &&
          queue.effects[1].acquire->operation == 1);
    CHECK(backing.bytes[0x88] == std::byte{0x78});

    Completion fallback;
    OrderedEffectQueue unused_queue;
    auto fallback_callbacks = fallback.callbacks();
    fallback_callbacks.rendering.context = &unused_queue;
    fallback_callbacks.rendering.ordered_effect = OrderedEffectQueue::enqueue;
    EngineFrontEnd fallback_engines;
    CHECK(fallback_engines.increment_syncpoint(memory, fallback_callbacks, 11, 2).ok());
    CHECK(unused_queue.effects.empty());
    CHECK(fallback.increment_count == 2);
    CHECK(fallback.increments[0] == 11);
    CHECK(fallback.increments[1] == 11);
}

}  // namespace

int main()
{
    test_render_stop_register_deltas();
    test_render_stop_copy_metadata();
    test_macro_upload_call_and_render_stop();
    test_macro_argument_order_and_invalid_code();
    test_shadow_and_snapshot_descriptions();
    test_constant_buffer_upload_and_fault();
    test_completion_and_reports();
    test_constant_buffer_binding_snapshots();
    test_effective_clear_startup_state();
    test_falcon_configuration_handshake();
    test_other_engine_stops_and_unsupported_actions();
    test_inline_upload_blocks_ordered_side_effects();
    test_inline_upload_failure_remains_blocked();
    test_macro_can_finish_but_not_overtake_upload();
    test_zero_length_engine_upload_is_complete();
    test_synchronous_clear_backend();
    test_draw_index_selection_and_synchronous_completion();
    test_draw_stops_carry_changed_register_groups();
    test_draw_failure_and_unproven_triggers_remain_sticky();
    test_draw_bounds_and_begin_controls();
    test_draw_instance_controls_and_overflow();
    test_compact_draw_forms_latches_and_shadow();
    test_inline_index_ownership_consumption_and_limits();
    test_nonzero_effective_draw_end_is_unsupported();
    test_draw_effective_shadow_replay_and_explicit_defaults();
    test_deferred_writeback_coherence_points();
    test_coherence_failure_stops_method_and_retry_succeeds();
    test_front_end_without_coherence_callback_unchanged();
    test_ordered_front_end_writes();
    test_ordered_write_error_is_preserved();
    test_ordered_guest_effect_routing_and_owned_increment();
    return failures == 0 ? 0 : 1;
}
