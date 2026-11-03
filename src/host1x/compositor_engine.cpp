// ============================================================================
// Armigo Host1x - Video Image Compositor
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "engines.h"

#include <armigo/graphics/texture_layout.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <variant>

namespace armigo::host1x {
namespace {

// A compositor working pixel: 10-bit channels in 16-bit storage. Read
// pictures put luma in r, U in g and V in b until a colour matrix converts them.
struct Pixel {
    std::uint16_t r = 0, g = 0, b = 0, a = 0;
};

// Surface pixel formats of the configuration structure (the ones handled).
enum class PixelFormat : std::uint32_t {
    a8b8g8r8 = 31,
    a8r8g8b8 = 32,
    x8b8g8r8 = 35,
    y8_v8u8_n420 = 68,
};

enum class BlockKind : std::uint32_t { pitch = 0, generic_16bx2 = 1 };

// Frame formats a slot reads its picture as.
enum class FrameFormat : std::uint32_t { progressive = 0, top_field = 3, bottom_field = 4 };

constexpr std::uint64_t field(std::uint64_t word, unsigned low, unsigned count) {
    return (word >> low) & ((std::uint64_t{1} << count) - 1);
}

constexpr std::int32_t signed_field(std::uint64_t word, unsigned low, unsigned count) {
    const auto value = field(word, low, count);
    const auto sign = std::uint64_t{1} << (count - 1);
    return static_cast<std::int32_t>(static_cast<std::int64_t>(value ^ sign) - static_cast<std::int64_t>(sign));
}

// The 0x610-byte configuration structure the Execute method names: pipe and
// output configuration, the output surface, and eight 0xb0-byte slots.
struct Configuration {
    static constexpr std::size_t size = 0x610;
    static constexpr std::size_t slot_base = 0x90, slot_size = 0xb0, slot_count = 8;

    struct OutputSurface {
        std::uint32_t format = 0, block_kind = 0, block_height_log2 = 0;
        std::uint32_t surface_width = 0, surface_height = 0;
        std::uint32_t luma_width = 0, luma_height = 0;
        std::uint32_t chroma_width = 0, chroma_height = 0;
    };
    // A 3x4 colour conversion matrix in S12.8; the fourth column is added
    // after the right shift.
    struct ColorMatrix {
        std::array<std::array<std::int32_t, 4>, 3> coefficients{};
        std::uint32_t shift = 0;
        bool enable = false;
    };
    struct Slot {
        bool enable = false;
        std::uint32_t frame_format = 0;
        std::uint32_t soft_clamp_low = 0, soft_clamp_high = 0, planar_alpha = 0;
        std::uint32_t source_left = 0, source_right = 0, source_top = 0, source_bottom = 0;
        std::uint32_t dest_left = 0, dest_right = 0, dest_top = 0, dest_bottom = 0;
        std::uint32_t surface_width = 0, surface_height = 0;
        ColorMatrix color;
    };

    std::uint32_t target_left = 0, target_right = 0, target_top = 0, target_bottom = 0;
    OutputSurface output;
    std::array<Slot, slot_count> slots;

    static std::uint64_t u64_at(std::span<const std::byte> bytes, std::size_t offset) {
        std::uint64_t value;
        std::memcpy(&value, bytes.data() + offset, sizeof value);
        return value;
    }
    static std::uint32_t u32_at(std::span<const std::byte> bytes, std::size_t offset) {
        std::uint32_t value;
        std::memcpy(&value, bytes.data() + offset, sizeof value);
        return value;
    }

    explicit Configuration(std::span<const std::byte, size> bytes) {
        const auto target_x = u32_at(bytes, 0x18), target_y = u32_at(bytes, 0x1c);
        target_left = field(target_x, 0, 14);
        target_right = field(target_x, 16, 14);
        target_top = field(target_y, 0, 14);
        target_bottom = field(target_y, 16, 14);

        const auto format = u32_at(bytes, 0x20), surface = u32_at(bytes, 0x24);
        const auto luma = u32_at(bytes, 0x28), chroma = u32_at(bytes, 0x2c);
        output.format = field(format, 0, 7);
        output.block_kind = field(format, 11, 4);
        output.block_height_log2 = field(format, 15, 4);
        output.surface_width = field(surface, 0, 14) + 1;
        output.surface_height = field(surface, 14, 14) + 1;
        output.luma_width = field(luma, 0, 14) + 1;
        output.luma_height = field(luma, 14, 14) + 1;
        output.chroma_width = field(chroma, 0, 14) + 1;
        output.chroma_height = field(chroma, 14, 14) + 1;

        for (std::size_t i = 0; i < slot_count; ++i) {
            const auto base = slot_base + i * slot_size;
            auto& slot = slots[i];
            const auto config0 = u64_at(bytes, base + 0x00), config2 = u64_at(bytes, base + 0x10);
            const auto source_x = u64_at(bytes, base + 0x20), source_y = u64_at(bytes, base + 0x28);
            const auto dest = u64_at(bytes, base + 0x30);
            slot.enable = field(config0, 0, 1) != 0;
            slot.frame_format = field(config0, 16, 4);
            slot.soft_clamp_low = field(config2, 0, 10);
            slot.soft_clamp_high = field(config2, 10, 10);
            slot.planar_alpha = field(config2, 32, 10);
            slot.source_left = field(source_x, 0, 30);
            slot.source_right = field(source_x, 32, 30);
            slot.source_top = field(source_y, 0, 30);
            slot.source_bottom = field(source_y, 32, 30);
            slot.dest_left = field(dest, 0, 14);
            slot.dest_right = field(dest, 16, 14);
            slot.dest_top = field(dest, 32, 14);
            slot.dest_bottom = field(dest, 48, 14);
            const auto slot_surface = u32_at(bytes, base + 0x44);
            slot.surface_width = field(slot_surface, 0, 14) + 1;
            slot.surface_height = field(slot_surface, 14, 14) + 1;
            for (std::size_t column = 0; column < 4; ++column) {
                const auto word = u64_at(bytes, base + 0x60 + column * 8);
                for (std::size_t row = 0; row < 3; ++row)
                    slot.color.coefficients[row][column] = signed_field(word, static_cast<unsigned>(row * 20), 20);
                if (column == 0) slot.color.shift = field(word, 60, 4);
                if (column == 1) slot.color.enable = field(word, 63, 1) != 0;
            }
        }
    }
};

// The average of every 64th luma sample of every 16th row, for tracing.
unsigned mean_luma(const DecodedFrame& frame) {
    std::uint64_t sum = 0, count = 0;
    for (std::size_t y = 0; y < frame.height; y += 16)
        for (std::size_t x = 0; x < frame.width; x += 64, ++count) sum += frame.luma[y * frame.luma_stride + x];
    return count ? static_cast<unsigned>(sum / count) : 0u;
}

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// The video image compositor engine. Its registers are 32-bit words; Execute
// reads each enabled slot's decoded picture, converts it into the output
// surface and writes that surface to the guest in the configured format and
// layout. Pictures come from the frame queue: the first slot surface the
// compositor names selects the decoder channel, whose pictures it then takes
// in order.
class CompositorEngine final : public CommandPusher {
public:
    static constexpr std::size_t register_count = 0x446;
    static constexpr std::size_t execute_method = 0xc0;
    static constexpr std::size_t surfaces = 0x100;  // [slot][surface] luma, chroma U, chroma V
    static constexpr std::size_t config_struct = 0x1c2;
    static constexpr std::size_t output_luma = 0x1c8, output_chroma = 0x1c9;

    CompositorEngine(Hub& hub, std::uint32_t syncpoint) : CommandPusher(hub, ClassId::video_compositor, syncpoint) {
        start();
    }
    ~CompositorEngine() override { stop(); }

protected:
    bool process_method(std::uint32_t method, std::uint32_t argument, std::string& failure) override {
        if (method >= registers_.size()) {
            failure = "Video compositor method " + std::to_string(method) + " is out of range";
            return false;
        }
        registers_[method] = argument;
        if (method != execute_method) return true;
        const double start = tracing() ? trace_seconds() : 0.0;
        const bool executed = execute(failure);
        if (tracing())
            armigo::diag::opt_in( "host1x %.3f compositor execution took %.2f ms\n", trace_seconds(),
                         (trace_seconds() - start) * 1000.0);
        return executed;
    }

private:
    std::uint64_t address(std::size_t word) const { return std::uint64_t{registers_[word]} << 8; }

    bool execute(std::string& failure) {
        std::array<std::byte, Configuration::size> bytes;
        if (!hub_.memory().read(hub_.memory().context, address(config_struct), bytes)) {
            failure = "Video compositor could not read its configuration at 0x" + hex(address(config_struct));
            return false;
        }
        const Configuration config(bytes);
        output_surface_.resize(std::size_t{config.output.surface_width} * config.output.surface_height);

        bool decoded = false;
        for (std::size_t i = 0; i < config.slots.size(); ++i) {
            const auto& slot = config.slots[i];
            if (!slot.enable) continue;
            const auto luma = address(surfaces + i * 8 * 3);
            // A closed decoder channel is replaced by the next one that has
            // this surface's pictures (a new movie opens a new channel).
            if (decoder_ >= 0 && !hub_.frames().is_open(decoder_)) decoder_ = -1;
            if (decoder_ < 0) decoder_ = hub_.frames().find_decoder(luma);
            const auto frame = hub_.frames().take(decoder_);
            if (tracing())
                armigo::diag::opt_in(
                             "host1x %.3f compositor slot %zu luma 0x%llx frame %s mean luma %u format %u "
                             "output 0x%llx %ux%u luma %ux%u chroma %ux%u format %u kind %u block %u\n",
                             trace_seconds(), i, static_cast<unsigned long long>(luma), frame ? "taken" : "missing",
                             frame ? mean_luma(*frame) : 0u, slot.frame_format,
                             static_cast<unsigned long long>(address(output_luma)), config.output.surface_width,
                             config.output.surface_height, config.output.luma_width, config.output.luma_height,
                             config.output.chroma_width, config.output.chroma_height, config.output.format,
                             config.output.block_kind, config.output.block_height_log2);
            if (!frame) continue;
            if (!read_frame(slot, *frame, failure)) return false;
            blend(config, slot);
            decoded = true;
        }
        if (decoded) {
            has_decoded_ = true;
        } else if (!has_decoded_) {
            return true;
        }

        switch (static_cast<PixelFormat>(config.output.format)) {
        case PixelFormat::a8b8g8r8:
        case PixelFormat::x8b8g8r8:
        case PixelFormat::a8r8g8b8:
            return write_abgr(config.output, failure);
        case PixelFormat::y8_v8u8_n420:
            return write_y8_v8u8_n420(config.output, failure);
        }
        failure = "Video compositor output format " + std::to_string(config.output.format) + " is not implemented";
        return false;
    }

    // Expands a 4:2:0 picture with interleaved chroma into the slot surface,
    // duplicating chroma horizontally and vertically. A field picture fills a
    // surface of twice the slot's height.
    bool read_frame(const Configuration::Slot& slot, const DecodedFrame& frame, std::string& failure) {
        const auto format = static_cast<FrameFormat>(slot.frame_format);
        if (format != FrameFormat::progressive && format != FrameFormat::top_field &&
            format != FrameFormat::bottom_field) {
            failure = "Video compositor frame format " + std::to_string(slot.frame_format) + " is not implemented";
            return false;
        }
        const bool interlaced = format != FrameFormat::progressive;
        const std::size_t out_width = slot.surface_width;
        const std::size_t out_height = std::size_t{slot.surface_height} * (interlaced ? 2 : 1);
        const auto in_width = std::min<std::size_t>(frame.width, out_width);
        const auto in_height = std::min<std::size_t>(frame.height, out_height);
        slot_surface_.resize(out_width * out_height);
        slot_width_ = out_width;
        slot_height_ = out_height;
        const auto alpha = static_cast<std::uint16_t>(slot.planar_alpha);
        for (std::size_t y = 0; y < in_height; ++y) {
            const auto* luma = frame.luma.data() + y * frame.luma_stride;
            const auto* chroma = frame.chroma.data() + (y / 2) * frame.chroma_stride;
            auto* out = slot_surface_.data() + y * out_width;
            for (std::size_t x = 0; x < in_width; ++x)
                out[x] = {static_cast<std::uint16_t>(luma[x] << 2),
                          static_cast<std::uint16_t>(chroma[(x & ~std::size_t{1}) + 0] << 2),
                          static_cast<std::uint16_t>(chroma[(x & ~std::size_t{1}) + 1] << 2), alpha};
        }
        return true;
    }

    // Places the slot surface in the output surface: the source rectangle,
    // limited to the destination and target rectangles and to both surfaces,
    // converted through the slot's colour matrix when it is enabled.
    void blend(const Configuration& config, const Configuration::Slot& slot) {
        const auto add_one = [](std::uint32_t value) { return value + (value > 0 ? 1u : 0u); };
        auto source_left = add_one(slot.source_left), source_right = add_one(slot.source_right);
        auto source_top = add_one(slot.source_top), source_bottom = add_one(slot.source_bottom);
        auto rect_left = std::max(add_one(config.target_left), add_one(slot.dest_left));
        auto rect_right = std::min(add_one(config.target_right), add_one(slot.dest_right));
        auto rect_top = std::max(add_one(config.target_top), add_one(slot.dest_top));
        auto rect_bottom = std::min(add_one(config.target_bottom), add_one(slot.dest_bottom));
        source_left = std::max(source_left, rect_left);
        source_right = std::min(source_right, rect_right);
        source_top = std::max(source_top, rect_top);
        source_bottom = std::min(source_bottom, rect_bottom);
        const std::uint32_t out_width = config.output.surface_width, out_height = config.output.surface_height;
        source_bottom = std::min({source_bottom, out_height, static_cast<std::uint32_t>(slot_height_)});
        source_right = std::min({source_right, out_width, static_cast<std::uint32_t>(slot_width_)});
        if (source_right <= source_left || source_bottom <= source_top) return;
        // A source column x lands at rect_left + (x - source_left).
        const std::size_t in_width = slot_width_;

        if (!slot.color.enable) {
            const auto copy_width = std::min(source_right - source_left, rect_right - rect_left);
            for (std::uint32_t y = source_top; y < source_bottom; ++y)
                std::memcpy(&output_surface_[y * std::size_t{out_width} + rect_left],
                            &slot_surface_[y * in_width + source_left], copy_width * sizeof(Pixel));
            return;
        }
        const auto& m = slot.color.coefficients;
        const auto shift = static_cast<std::int32_t>(slot.color.shift);
        const auto low = static_cast<std::int32_t>(slot.soft_clamp_low);
        const auto high = static_cast<std::int32_t>(slot.soft_clamp_high);
        const auto clamp = [&](std::int32_t value) {
            return static_cast<std::uint16_t>(std::min(std::max(value, low), high));
        };
        for (std::uint32_t y = source_top; y < source_bottom; ++y) {
            const auto* in = &slot_surface_[y * in_width];
            auto* out = &output_surface_[y * std::size_t{out_width} + rect_left];
            for (std::uint32_t x = source_left; x < source_right; ++x) {
                const std::int32_t r = in[x].r, g = in[x].g, b = in[x].b;
                const std::int32_t cr = (((r * m[0][0] + g * m[0][1] + b * m[0][2]) >> shift) + m[0][3]) >> 8;
                const std::int32_t cg = (((r * m[1][0] + g * m[1][1] + b * m[1][2]) >> shift) + m[1][3]) >> 8;
                const std::int32_t cb = (((r * m[2][0] + g * m[2][1] + b * m[2][2]) >> shift) + m[2][3]) >> 8;
                out[x - source_left] = {clamp(cr), clamp(cg), clamp(cb), clamp(in[x].a)};
            }
        }
    }

    // Writes `linear` (rows of `row_bytes`, tightly packed for block-linear
    // output, `pitch` apart for pitch output) to the surface at `address`.
    bool write_plane(std::uint64_t address, const Configuration::OutputSurface& output, std::uint32_t width,
                     std::uint32_t height, std::size_t bytes_per_pixel, std::span<const std::byte> linear,
                     std::string& failure) {
        auto& memory = hub_.memory();
        if (static_cast<BlockKind>(output.block_kind) == BlockKind::pitch) {
            if (memory.write(memory.context, address, linear)) return true;
            failure = "Video compositor could not write its output surface at 0x" + hex(address);
            return false;
        }
        if (static_cast<BlockKind>(output.block_kind) != BlockKind::generic_16bx2) {
            failure = "Video compositor output block kind " + std::to_string(output.block_kind) +
                      " is not implemented";
            return false;
        }
        graphics::TextureLayoutDescription layout;
        layout.width = width;
        layout.height = height;
        layout.depth = 1;
        layout.levels = 1;
        layout.layers = 1;
        layout.bytes_per_block = bytes_per_pixel;
        layout.gob_block_height_log2 = output.block_height_log2;
        layout.storage = graphics::TextureStorageLayout::BlockLinear;
        auto calculated = graphics::calculate_texture_size(std::move(layout));
        if (const auto* error = std::get_if<graphics::TextureLayoutError>(&calculated)) {
            failure = std::string("Video compositor output layout is invalid: ") + error->message();
            return false;
        }
        const auto& described = std::get<graphics::TextureLayoutDescription>(calculated);
        // Read-modify-write so bytes outside the picture keep their contents.
        guest_scratch_.resize(described.total_size);
        if (!memory.read(memory.context, address, guest_scratch_)) {
            failure = "Video compositor could not read its output surface at 0x" + hex(address);
            return false;
        }
        const auto written = graphics::write_level_from_linear(described, linear, guest_scratch_, 0, 0);
        if (const auto* error = std::get_if<graphics::TextureLayoutError>(&written)) {
            failure = std::string("Video compositor could not swizzle its output: ") + error->message();
            return false;
        }
        if (memory.write(memory.context, address, guest_scratch_)) return true;
        failure = "Video compositor could not write its output surface at 0x" + hex(address);
        return false;
    }

    // RGBA output (A8B8G8R8 and X8B8G8R8 store R first, A8R8G8B8 stores B first).
    bool write_abgr(const Configuration::OutputSurface& output, std::string& failure) {
        const std::size_t surface_stride = output.surface_width;
        const auto width = std::min(output.surface_width, output.luma_width);
        const auto height = std::min(output.surface_height, output.luma_height);
        const bool pitch = static_cast<BlockKind>(output.block_kind) == BlockKind::pitch;
        const std::size_t row_bytes = pitch ? align_up(std::size_t{output.luma_width} * 4, 0x10)
                                            : std::size_t{output.luma_width} * 4;
        const bool bgra = static_cast<PixelFormat>(output.format) == PixelFormat::a8r8g8b8;
        luma_scratch_.resize(row_bytes * output.luma_height);
        if (pitch && !hub_.memory().read(hub_.memory().context, address(output_luma), luma_scratch_)) {
            failure = "Video compositor could not read its output surface at 0x" + hex(address(output_luma));
            return false;
        }
        for (std::size_t y = 0; y < height; ++y) {
            const auto* in = &output_surface_[y * surface_stride];
            auto* out = reinterpret_cast<std::uint8_t*>(luma_scratch_.data() + y * row_bytes);
            for (std::size_t x = 0; x < width; ++x) {
                const auto r = static_cast<std::uint8_t>(in[x].r >> 2), g = static_cast<std::uint8_t>(in[x].g >> 2);
                const auto b = static_cast<std::uint8_t>(in[x].b >> 2), a = static_cast<std::uint8_t>(in[x].a >> 2);
                out[x * 4 + 0] = bgra ? b : r;
                out[x * 4 + 1] = g;
                out[x * 4 + 2] = bgra ? r : b;
                out[x * 4 + 3] = a;
            }
        }
        return write_plane(address(output_luma), output, output.luma_width, output.luma_height, 4, luma_scratch_,
                           failure);
    }

    // 4:2:0 output with an 8-bit luma plane and an interleaved V/U plane.
    bool write_y8_v8u8_n420(const Configuration::OutputSurface& output, std::string& failure) {
        const std::size_t surface_stride = output.surface_width;
        const auto width = std::min(output.surface_width, output.luma_width);
        const auto height = std::min(output.surface_height, output.luma_height);
        const bool pitch = static_cast<BlockKind>(output.block_kind) == BlockKind::pitch;
        const std::size_t luma_row = pitch ? align_up(output.luma_width, 0x10) : output.luma_width;
        const std::size_t chroma_row = pitch ? align_up(std::size_t{output.chroma_width} * 2, 0x10)
                                             : std::size_t{output.chroma_width} * 2;
        luma_scratch_.assign(luma_row * output.luma_height, std::byte{});
        chroma_scratch_.assign(chroma_row * output.chroma_height, std::byte{});
        auto* luma = reinterpret_cast<std::uint8_t*>(luma_scratch_.data());
        auto* chroma = reinterpret_cast<std::uint8_t*>(chroma_scratch_.data());
        for (std::size_t y = 0; y < height; ++y) {
            const auto* in = &output_surface_[y * surface_stride];
            for (std::size_t x = 0; x < width; ++x) luma[y * luma_row + x] = static_cast<std::uint8_t>(in[x].r >> 2);
            if (y / 2 >= output.chroma_height) continue;
            // Each row writes its chroma row; the odd row of a pair is the one kept.
            auto* row = chroma + (y / 2) * chroma_row;
            for (std::size_t x = 0; x + 1 < chroma_row && x < width; x += 2) {
                row[x + 0] = static_cast<std::uint8_t>(in[x].g >> 2);
                row[x + 1] = static_cast<std::uint8_t>(in[x].b >> 2);
            }
        }
        return write_plane(address(output_luma), output, output.luma_width, output.luma_height, 1, luma_scratch_,
                           failure) &&
               write_plane(address(output_chroma), output, output.chroma_width, output.chroma_height, 2,
                           chroma_scratch_, failure);
    }

    static std::string hex(std::uint64_t value) {
        char text[24];
        std::snprintf(text, sizeof text, "%llx", static_cast<unsigned long long>(value));
        return text;
    }

    std::array<std::uint32_t, register_count> registers_{};
    std::int64_t decoder_ = -1;
    bool has_decoded_ = false;
    std::vector<Pixel> slot_surface_;
    std::size_t slot_width_ = 0, slot_height_ = 0;
    std::vector<Pixel> output_surface_;
    std::vector<std::byte> luma_scratch_, chroma_scratch_, guest_scratch_;
};

} // namespace

std::unique_ptr<CommandPusher> make_video_compositor(Hub& hub, std::uint32_t, std::uint32_t syncpoint) {
    return std::make_unique<CompositorEngine>(hub, syncpoint);
}

} // namespace armigo::host1x
