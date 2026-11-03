// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/twod_blit.h>

#include <armigo/graphics/engines.h>
#include <armigo/graphics/memory.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <variant>

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

// Installs one 2D surface register block: format, layout, block size, depth,
// layer, pitch, width, height, address upper, address lower.
void set_surface(std::array<std::uint32_t, 0x1000>& registers, std::size_t base,
                 std::uint32_t format, bool pitch, std::uint8_t block_height_log2,
                 std::uint32_t depth, std::uint32_t layer, std::uint32_t pitch_bytes,
                 std::uint32_t width, std::uint32_t height, std::uint64_t address)
{
    registers[base] = format;
    registers[base + 1] = pitch ? 1 : 0;
    registers[base + 2] = (std::uint32_t{block_height_log2} << 4);
    registers[base + 3] = depth;
    registers[base + 4] = layer;
    registers[base + 5] = pitch_bytes;
    registers[base + 6] = width;
    registers[base + 7] = height;
    registers[base + 8] = std::uint32_t(address >> 32);
    registers[base + 9] = std::uint32_t(address);
}

constexpr std::uint32_t format_a8b8g8r8 = 0xd5;  // R8G8B8A8Unorm
constexpr std::uint32_t operation_srccopy = 3;

std::array<std::uint32_t, 0x1000> copy_registers()
{
    std::array<std::uint32_t, 0x1000> registers{};
    set_surface(registers, 0x8c, format_a8b8g8r8, false, 0, 1, 0, 0, 64, 32, 0x1'0000'0000);
    set_surface(registers, 0x80, format_a8b8g8r8, false, 0, 1, 0, 0, 64, 32, 0x2'0000'0000);
    registers[0xab] = operation_srccopy;
    registers[0x223] = 0;  // center origin, point filter
    registers[0x22c] = 0;  // destination x0
    registers[0x22d] = 0;  // destination y0
    registers[0x22e] = 64; // destination width
    registers[0x22f] = 32; // destination height
    registers[0x230] = std::uint32_t(0x1'00000000ull & 0xffffffff);  // du/dx frac
    registers[0x231] = 1;                                            // du/dx int
    registers[0x232] = std::uint32_t(0x1'00000000ull & 0xffffffff);  // dv/dy frac
    registers[0x233] = 1;                                            // dv/dy int
    registers[0x234] = 0;  // source x0 frac
    registers[0x235] = 0;  // source x0 int
    registers[0x236] = 0;  // source y0 frac
    return registers;
}

void test_full_surface_copy_decode()
{
    const auto registers = copy_registers();
    const auto decoded = decode_twod_blit(registers);
    const auto* blit = std::get_if<TwodBlit>(&decoded);
    CHECK(blit != nullptr);
    if (!blit) return;
    CHECK(blit->operation == TwodOperation::SrcCopy);
    CHECK(!blit->corner_origin);
    CHECK(!blit->linear_filter);
    CHECK(!blit->flip_x && !blit->flip_y);
    CHECK(blit->source.address == 0x1'0000'0000);
    CHECK(blit->destination.address == 0x2'0000'0000);
    CHECK(blit->source.width == 64 && blit->source.height == 32);
    CHECK(blit->source.block_height_log2 == 0);
    CHECK(blit->source_x1 == 0 && blit->source_x2 == 64);
    CHECK(blit->source_y1 == 0 && blit->source_y2 == 32);
    CHECK(blit->destination_x1 == 0 && blit->destination_y1 == 0);
    CHECK(blit->destination_width == 64 && blit->destination_height == 32);
    CHECK(blit->source_address_offset == 0);
    CHECK(!blit->clip_enable);
}

void test_scaled_and_filtered_decode()
{
    auto registers = copy_registers();
    registers[0x22e] = 16;  // 64x32 -> 16x8 downscale
    registers[0x22f] = 8;
    registers[0x231] = 4;   // du/dx 4.0
    registers[0x233] = 4;   // dv/dy 4.0
    registers[0x223] = 0x11;  // corner origin, bilinear
    const auto decoded = decode_twod_blit(registers);
    const auto* blit = std::get_if<TwodBlit>(&decoded);
    CHECK(blit != nullptr);
    if (!blit) return;
    CHECK(blit->linear_filter);
    CHECK(blit->corner_origin);
    // Corner origin removes the whole part of half the step: 4.0 / 2 = 2.0.
    CHECK(blit->source_x1 == -2 && blit->source_y1 == -2);
    // (4 * 16 + 0xffffffff) >> 32 = 64.
    CHECK(blit->source_x2 == 62 && blit->source_y2 == 30);
}

void test_pitch_wrap_decode()
{
    auto registers = copy_registers();
    // A pitch-linear surface whose encoded width (96) exceeds the stride's
    // pixel count (256 / 4 = 64): the copy of [32, 96[ wraps past line one.
    set_surface(registers, 0x8c, format_a8b8g8r8, true, 0, 1, 0, 256, 96, 32,
                0x1'0000'0000);
    registers[0x22e] = 64;
    registers[0x22f] = 32;
    registers[0x235] = 32;  // source x0 = 32.0
    registers[0x230] = 0;
    registers[0x231] = 1;
    const auto decoded = decode_twod_blit(registers);
    const auto* blit = std::get_if<TwodBlit>(&decoded);
    CHECK(blit != nullptr);
    if (!blit) return;
    CHECK(blit->source.pitch);
    // The wrap skip offsets by 32 pixels and restarts the x origin at zero.
    CHECK(blit->source_address_offset == 32 * 4);
    CHECK(blit->source.address == 0x1'0000'0000 + 32 * 4);
    CHECK(blit->source.width == 64);
    CHECK(blit->source_x1 == 0 && blit->source_x2 == 64);
}

void test_reserved_encodings_are_structured_errors()
{
    auto registers = copy_registers();
    registers[0xab] = 7;  // operation encodings above BlendPremult are reserved
    const auto reserved = decode_twod_blit(registers);
    const auto* reserved_error = std::get_if<CommandResult>(&reserved);
    CHECK(reserved_error != nullptr);
    if (reserved_error) {
        CHECK(reserved_error->status == CommandStatus::Unsupported);
        CHECK(reserved_error->detail.find("operation") != std::string_view::npos);
    }
    // Rop decodes; the backend rejects it as an unsupported operation.
    registers[0xab] = 4;
    const auto rop = decode_twod_blit(registers);
    const auto* rop_blit = std::get_if<TwodBlit>(&rop);
    CHECK(rop_blit != nullptr && rop_blit->operation == TwodOperation::Rop);
    registers[0xab] = operation_srccopy;
    registers[0xb8] = 0x3;  // reserved horizontal direction encoding
    const auto direction = decode_twod_blit(registers);
    const auto* direction_error = std::get_if<CommandResult>(&direction);
    CHECK(direction_error != nullptr);
    if (direction_error) {
        CHECK(direction_error->status == CommandStatus::Unsupported);
        CHECK(direction_error->detail.find("direction") != std::string_view::npos);
    }
    registers[0xb8] = 0x2;  // right-to-left decodes as a flip
    registers[0x234] = 0;   // default source x0
    const auto flipped = decode_twod_blit(registers);
    const auto* flipped_blit = std::get_if<TwodBlit>(&flipped);
    CHECK(flipped_blit != nullptr && flipped_blit->flip_x && !flipped_blit->flip_y);
}

void test_format_decode()
{
    TextureFormat format = TextureFormat::R8Unorm;
    CHECK(decode_twod_format(0xcf, format) == false);  // BGR ordering: no entry
    CHECK(decode_twod_format(0xd5, format));
    CHECK(format == TextureFormat::R8G8B8A8Unorm);
    CHECK(decode_twod_format(0xe8, format));
    CHECK(format == TextureFormat::R5G6B5Unorm);
    CHECK(decode_twod_format(0xf3, format));
    CHECK(format == TextureFormat::R8Unorm);
}

void test_clip_and_depth_decode()
{
    auto registers = copy_registers();
    registers[0xa0] = 4;
    registers[0xa1] = 6;
    registers[0xa2] = 8;
    registers[0xa3] = 10;
    registers[0xa4] = 1;
    registers[0x8f] = 2;  // source depth two slices
    const auto decoded = decode_twod_blit(registers);
    const auto* blit = std::get_if<TwodBlit>(&decoded);
    CHECK(blit != nullptr);
    if (!blit) return;
    CHECK(blit->clip_enable);
    CHECK(blit->clip_x0 == 4 && blit->clip_y0 == 6);
    CHECK(blit->clip_width == 8 && blit->clip_height == 10);
    CHECK(blit->source.depth == 2);
}

// Records one blit callback invocation with its captured stop.
struct BlitCompletion {
    static CommandResult blit(void* raw, const RenderStop& stop,
                              const GpuAddressSpace&) noexcept {
        auto& self = *static_cast<BlitCompletion*>(raw);
        ++self.calls;
        self.stopped = stop;
        return self.result;
    }
    CompletionCallbacks callbacks() noexcept {
        CompletionCallbacks completion;
        completion.rendering.blit = blit;
        completion.rendering.context = this;
        return completion;
    }
    CommandResult result{};
    int calls = 0;
    std::optional<RenderStop> stopped;
};

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
    CHECK(std::holds_alternative<GpuAddressSpace>(result));
    if (!std::holds_alternative<GpuAddressSpace>(result)) std::abort();
    return std::get<GpuAddressSpace>(std::move(result));
}

MethodCall call(std::uint32_t method, std::uint32_t argument, bool last = true)
{
    return {method, 2, argument, 0x4000 + method * 4, last};
}

void test_front_end_dispatches_the_blit_callback()
{
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);
    BlitCompletion completion;

    // A successful callback consumes the stop and completes the method.
    EngineFrontEnd executed;
    completion.calls = 0;
    completion.stopped.reset();
    CHECK(executed.call(call(0x237, 1), engine_class::twod, memory, completion.callbacks())
              .ok());
    CHECK(completion.calls == 1);
    CHECK(!executed.rendering_stop().has_value());
    CHECK(completion.stopped->kind == RenderKind::Blit);
    CHECK(completion.stopped->engine == engine_class::twod);

    // The captured register file carries the 2D state the backend decodes.
    EngineFrontEnd state;
    CHECK(state.call(call(0x223, 0x10), engine_class::twod, memory, completion.callbacks())
              .ok());
    CHECK(state.call(call(0x237, 1), engine_class::twod, memory, completion.callbacks())
              .ok());
    const auto decoded = decode_twod_blit(completion.stopped->registers);
    const auto* blit = std::get_if<TwodBlit>(&decoded);
    CHECK(blit != nullptr && blit->linear_filter);

    // A failing callback propagates its status and retains the stop.
    EngineFrontEnd failing;
    completion.calls = 0;
    completion.result = CommandResult{CommandStatus::Unsupported, "2D blit rejected"};
    const auto outcome =
        failing.call(call(0x237, 1), engine_class::twod, memory, completion.callbacks());
    CHECK(outcome.status == CommandStatus::Unsupported);
    CHECK(outcome.detail == "2D blit rejected");
    CHECK(failing.rendering_stop().has_value());

    // Without a registered callback the run still stops at the blit.
    EngineFrontEnd halted;
    CompletionCallbacks none;
    CHECK(halted.call(call(0x237, 1), engine_class::twod, memory, none).status ==
          CommandStatus::Stopped);
    CHECK(halted.rendering_stop()->kind == RenderKind::Blit);
}

}  // namespace

int main()
{
    test_full_surface_copy_decode();
    test_scaled_and_filtered_decode();
    test_pitch_wrap_decode();
    test_reserved_encodings_are_structured_errors();
    test_format_decode();
    test_clip_and_depth_decode();
    test_front_end_dispatches_the_blit_callback();
    if (failures != 0) {
        std::fprintf(stderr, "%u twod blit decode checks failed\n", failures);
        return 1;
    }
    return 0;
}
