// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic fixtures: a compute launch reaches the
// registered RenderCallbacks::dispatch with the fully decoded queue-meta-data
// launch, and callback outcomes drive the front end exactly like a draw.

#include <armigo/graphics/engines.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
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

    static bool read(void* raw, std::uint64_t address, std::span<std::byte> output) noexcept
    {
        auto& self = *static_cast<Backing*>(raw);
        if (address < guest_base || address - guest_base > self.bytes.size() ||
            output.size() > self.bytes.size() - (address - guest_base)) {
            return false;
        }
        std::memcpy(output.data(), self.bytes.data() + (address - guest_base), output.size());
        return true;
    }

    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> input) noexcept
    {
        auto& self = *static_cast<Backing*>(raw);
        if (address < guest_base || address - guest_base > self.bytes.size() ||
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
    if (!std::holds_alternative<GpuAddressSpace>(result)) std::abort();
    return std::get<GpuAddressSpace>(std::move(result));
}

MethodCall call(std::uint32_t method, std::uint32_t argument)
{
    return {method, 2, argument, 0x4000 + method * 4, true};
}

struct Completion {
    static bool increment(void*, std::uint32_t) noexcept { return true; }
    static bool expired(void*, std::uint32_t, std::uint32_t) noexcept { return true; }
    static std::uint64_t ticks(void*) noexcept { return 0; }
    CompletionCallbacks callbacks() noexcept { return {increment, expired, ticks, this}; }
};

// Publishes a synthetic 256-byte descriptor at GPU address 0x1200 with one
// enabled constant buffer and grid/block dimensions.
void publish_descriptor(Backing& backing)
{
    std::array<std::uint32_t, 32> words{};
    words[8] = 8;  // Bits 256..287 (word 8): program offset 8.
    // Bits 384..447: grid 2 x 3 x 1. Bits 544..561: shared memory 0x100 words.
    words[12] = 2;
    words[13] = 3 | (1 << 16);
    words[17] = 0x100;
    // Bits 592..639: block 4 x 1 x 1 (x spans word 18 bits 16..31). Bit 640
    // (word 20, bit 0): constant buffer 0 valid.
    words[18] = 4 << 16;
    words[19] = 1 | (1 << 16);
    words[20] = 1;
    // Constant buffer 0: lower at bit 928 (word 29), upper at bit 960 (word
    // 30 bits 0..7), size at bits 975..991 (word 30 bits 15..31).
    words[29] = 0x3000;
    words[30] = 0x80u << 15;
    std::memcpy(backing.bytes.data() + 0x200, words.data(), sizeof(words));
}

// Records the dispatched stop and answers with the configured status.
struct DispatchRecorder {
    std::optional<RenderStop> stop;
    CommandStatus status = CommandStatus::Complete;
    std::uint32_t calls = 0;

    static CommandResult dispatch(void* raw, const RenderStop& stop,
                                  const GpuAddressSpace&) noexcept
    {
        auto& self = *static_cast<DispatchRecorder*>(raw);
        ++self.calls;
        self.stop = stop;
        return {self.status, "recorded dispatch"};
    }
};

}  // namespace

int main()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    Completion completion;
    publish_descriptor(backing);

    EngineFrontEnd engines;
    CHECK(engines.call(call(0xad, 0x12), engine_class::compute, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x582, 0), engine_class::compute, memory, completion.callbacks()).ok());
    CHECK(engines.call(call(0x583, 0x5000), engine_class::compute, memory, completion.callbacks())
              .ok());

    // Without a registered dispatch callback the launch still stops as before.
    CHECK(engines.call(call(0xaf, 1), engine_class::compute, memory, completion.callbacks())
              .status == CommandStatus::Stopped);
    CHECK(engines.rendering_stop().has_value());

    // Each replayed launch uses a fresh front end; a refused attempt
    // deliberately keeps its stop pending.
    const auto replay = [&](const CompletionCallbacks& active) -> CommandResult {
        EngineFrontEnd launched;
        if (const auto setup = launched.call(call(0xad, 0x12), engine_class::compute, memory, active);
            !setup.ok()) return setup;
        if (const auto setup = launched.call(call(0x582, 0), engine_class::compute, memory, active);
            !setup.ok()) return setup;
        if (const auto setup = launched.call(call(0x583, 0x5000), engine_class::compute, memory, active);
            !setup.ok()) return setup;
        return launched.call(call(0xaf, 1), engine_class::compute, memory, active);
    };

    DispatchRecorder recorder;
    CompletionCallbacks callbacks = completion.callbacks();
    callbacks.rendering.dispatch = DispatchRecorder::dispatch;
    callbacks.rendering.context = &recorder;

    // A structured refusal leaves the launch pending and carries the
    // backend diagnostic on the command result.
    recorder.status = CommandStatus::Unsupported;
    const CommandResult refused = replay(callbacks);
    CHECK(refused.status == CommandStatus::Unsupported);
    CHECK(refused.detail == "recorded dispatch");
    CHECK(recorder.calls == 1);
    CHECK(engines.rendering_stop().has_value());

    // Success consumes the stop, exactly like a completed draw.
    recorder.status = CommandStatus::Complete;
    recorder.stop.reset();
    CHECK(replay(callbacks).ok());
    CHECK(recorder.calls == 2);

    const RenderStop& stop = *recorder.stop;
    CHECK(stop.kind == RenderKind::Compute);
    CHECK(stop.has_compute_descriptor);
    CHECK(stop.compute_descriptor_address == 0x1200);
    CHECK(stop.programs[0].enabled);
    CHECK(stop.programs[0].address == 0x5008);
    CHECK(stop.programs[0].offset == 8);
    CHECK(stop.compute.has_value());
    if (stop.compute) {
        const ComputeLaunch& launch = *stop.compute;
        CHECK((launch.grid == std::array<std::uint32_t, 3>{2, 3, 1}));
        CHECK((launch.block == std::array<std::uint32_t, 3>{4, 1, 1}));
        CHECK(launch.shared_memory_size == 0x100);
        CHECK(launch.program_offset == 8);
        CHECK(launch.constant_buffers[0].enabled);
        CHECK(launch.constant_buffers[0].address == 0x3000);
        CHECK(launch.constant_buffers[0].size == 0x80);
        CHECK(!launch.constant_buffers[1].enabled);
    }

    if (failures == 0) std::printf("compute launch front-end tests passed\n");
    else std::fprintf(stderr, "%u compute launch checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
