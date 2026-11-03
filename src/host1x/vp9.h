// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/host1x/host1x.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace armigo::host1x {

// The decoder engine's registers, one 64-bit slot per method index. Surface
// and buffer registers hold a device address shifted right by 8.
struct DecoderRegisters {
    static constexpr std::size_t count = 0x178;
    static constexpr std::size_t codec = 0x80;
    static constexpr std::size_t execute = 0xc0;
    static constexpr std::size_t picture_info = 0x101;
    static constexpr std::size_t bitstream = 0x102;
    static constexpr std::size_t luma_surfaces = 0x10c;    // 17 entries
    static constexpr std::size_t chroma_surfaces = 0x11d;  // 17 entries
    static constexpr std::size_t vp9_probabilities = 0x170;
    std::array<std::uint64_t, count> values{};
    std::uint64_t address(std::size_t index) const noexcept { return values[index] << 8; }
};

struct Vp9FrameSize {
    std::int16_t width;
    std::int16_t height;
    std::int16_t luma_pitch;
    std::int16_t chroma_pitch;
};
static_assert(sizeof(Vp9FrameSize) == 8);

struct Vp9Segmentation {
    bool operator==(const Vp9Segmentation&) const = default;
    std::uint8_t enabled;
    std::uint8_t update_map;
    std::uint8_t temporal_update;
    std::uint8_t abs_delta;
    std::array<std::array<std::uint8_t, 4>, 8> feature_enabled;
    std::array<std::array<std::int16_t, 4>, 8> feature_data;
};
static_assert(sizeof(Vp9Segmentation) == 0x64);

struct Vp9LoopFilter {
    std::uint8_t mode_ref_delta_enabled;
    std::array<std::int8_t, 4> ref_deltas;
    std::array<std::int8_t, 2> mode_deltas;
};
static_assert(sizeof(Vp9LoopFilter) == 7);

// Entropy probabilities in the order the compressed header codes them.
struct Vp9Probabilities {
    std::array<std::uint8_t, 36> y_mode;
    std::array<std::uint8_t, 64> partition;
    std::array<std::uint8_t, 1728> coefficients;
    std::array<std::uint8_t, 8> switchable_interp;
    std::array<std::uint8_t, 28> inter_mode;
    std::array<std::uint8_t, 4> intra_inter;
    std::array<std::uint8_t, 5> comp_inter;
    std::array<std::uint8_t, 10> single_ref;
    std::array<std::uint8_t, 5> comp_ref;
    std::array<std::uint8_t, 6> tx_32x32;
    std::array<std::uint8_t, 4> tx_16x16;
    std::array<std::uint8_t, 2> tx_8x8;
    std::array<std::uint8_t, 3> skip;
    std::array<std::uint8_t, 3> mv_joints;
    std::array<std::uint8_t, 2> mv_sign;
    std::array<std::uint8_t, 20> mv_classes;
    std::array<std::uint8_t, 2> mv_class_0;
    std::array<std::uint8_t, 20> mv_bits;
    std::array<std::uint8_t, 12> mv_class_0_fr;
    std::array<std::uint8_t, 6> mv_fr;
    std::array<std::uint8_t, 2> mv_class_0_hp;
    std::array<std::uint8_t, 2> mv_hp;
};
static_assert(sizeof(Vp9Probabilities) == 0x7b4);

// The picture description the guest writes for each picture (0x100 bytes).
struct Vp9GuestPicture {
    std::array<std::uint32_t, 12> reserved_0;
    std::uint32_t bitstream_size;              // 0x30
    std::array<std::uint32_t, 5> reserved_1;
    Vp9FrameSize last_frame_size;              // 0x48
    Vp9FrameSize golden_frame_size;            // 0x50
    Vp9FrameSize alt_frame_size;               // 0x58
    Vp9FrameSize current_frame_size;           // 0x60
    std::uint32_t flags;                       // 0x68
    std::array<std::int8_t, 4> ref_frame_sign_bias;  // 0x6c
    std::uint8_t first_level;                  // 0x70
    std::uint8_t sharpness_level;
    std::uint8_t base_q_index;
    std::uint8_t y_dc_delta_q;
    std::uint8_t uv_ac_delta_q;
    std::uint8_t uv_dc_delta_q;
    std::uint8_t lossless;
    std::uint8_t tx_mode;
    std::uint8_t allow_high_precision_mv;      // 0x78
    std::uint8_t interp_filter;
    std::uint8_t reference_mode;
    std::array<std::uint8_t, 3> reserved_2;
    std::uint8_t log2_tile_cols;               // 0x7e
    std::uint8_t log2_tile_rows;
    Vp9Segmentation segmentation;              // 0x80
    Vp9LoopFilter loop_filter;                 // 0xe4
    std::array<std::uint8_t, 21> reserved_3;   // 0xeb
};
static_assert(sizeof(Vp9GuestPicture) == 0x100);
static_assert(offsetof(Vp9GuestPicture, bitstream_size) == 0x30);
static_assert(offsetof(Vp9GuestPicture, last_frame_size) == 0x48);
static_assert(offsetof(Vp9GuestPicture, first_level) == 0x70);
static_assert(offsetof(Vp9GuestPicture, log2_tile_cols) == 0x7e);
static_assert(offsetof(Vp9GuestPicture, segmentation) == 0x80);
static_assert(offsetof(Vp9GuestPicture, loop_filter) == 0xe4);

// The probability table the guest keeps for the engine (0xea0 bytes).
struct Vp9GuestProbabilities {
    std::array<std::uint8_t, 800> kf_bmode;
    std::array<std::uint8_t, 100> kf_bmode_b;
    std::array<std::uint8_t, 3> ref_pred;
    std::array<std::uint8_t, 7> segment_tree;      // 0x387
    std::array<std::uint8_t, 3> segment_pred;      // 0x38e
    std::array<std::uint8_t, 4> ref_scores;
    std::array<std::uint8_t, 2> comp_pred;
    std::array<std::uint8_t, 9> reserved_0;
    std::array<std::uint8_t, 80> kf_uv_mode;
    std::array<std::uint8_t, 10> kf_uv_mode_b;
    std::array<std::uint8_t, 6> reserved_1;
    std::array<std::uint8_t, 28> inter_mode;       // 0x400
    std::array<std::uint8_t, 4> intra_inter;
    std::array<std::uint8_t, 80> reserved_2;
    std::array<std::uint8_t, 2> tx_8x8;            // 0x470
    std::array<std::uint8_t, 4> tx_16x16;
    std::array<std::uint8_t, 6> tx_32x32;
    std::array<std::uint8_t, 4> y_mode_e8;
    std::array<std::array<std::uint8_t, 8>, 4> y_mode_e0e7;  // 0x480
    std::array<std::uint8_t, 64> reserved_3;
    std::array<std::uint8_t, 64> partition;        // 0x4e0
    std::array<std::uint8_t, 10> reserved_4;
    std::array<std::uint8_t, 8> switchable_interp; // 0x52a
    std::array<std::uint8_t, 5> comp_inter;
    std::array<std::uint8_t, 3> skip;
    std::array<std::uint8_t, 1> reserved_5;
    std::array<std::uint8_t, 3> mv_joints;         // 0x53b
    std::array<std::uint8_t, 2> mv_sign;
    std::array<std::uint8_t, 2> mv_class_0;        // 0x540
    std::array<std::uint8_t, 6> mv_fr;
    std::array<std::uint8_t, 2> mv_class_0_hp;
    std::array<std::uint8_t, 2> mv_hp;
    std::array<std::uint8_t, 20> mv_classes;
    std::array<std::uint8_t, 12> mv_class_0_fr;    // 0x560
    std::array<std::uint8_t, 20> mv_bits;
    std::array<std::uint8_t, 10> single_ref;       // 0x580
    std::array<std::uint8_t, 5> comp_ref;
    std::array<std::uint8_t, 17> reserved_6;
    std::array<std::uint8_t, 2304> coefficients;   // 0x5a0, four per entry, the fourth unused
};
static_assert(sizeof(Vp9GuestProbabilities) == 0xea0);
static_assert(offsetof(Vp9GuestProbabilities, segment_tree) == 0x387);
static_assert(offsetof(Vp9GuestProbabilities, inter_mode) == 0x400);
static_assert(offsetof(Vp9GuestProbabilities, tx_8x8) == 0x470);
static_assert(offsetof(Vp9GuestProbabilities, partition) == 0x4e0);
static_assert(offsetof(Vp9GuestProbabilities, mv_class_0) == 0x540);
static_assert(offsetof(Vp9GuestProbabilities, mv_class_0_fr) == 0x560);
static_assert(offsetof(Vp9GuestProbabilities, coefficients) == 0x5a0);

// The VP9 state of one picture, from the guest's description and probabilities.
struct Vp9Picture {
    std::uint32_t bitstream_size = 0;
    std::array<std::uint64_t, 4> surfaces{};  // last, golden, alt-ref, current luma offsets
    std::array<std::int8_t, 4> ref_frame_sign_bias{};
    std::int32_t base_q_index = 0;
    std::int32_t y_dc_delta_q = 0;
    std::int32_t uv_dc_delta_q = 0;
    std::int32_t uv_ac_delta_q = 0;
    std::int32_t transform_mode = 0;
    std::int32_t interp_filter = 0;
    std::int32_t reference_mode = 0;
    std::int32_t log2_tile_cols = 0;
    std::int32_t log2_tile_rows = 0;
    std::array<std::int8_t, 4> ref_deltas{};
    std::array<std::int8_t, 2> mode_deltas{};
    Vp9Probabilities probabilities{};
    Vp9FrameSize frame_size{};
    std::uint8_t first_level = 0;
    std::uint8_t sharpness_level = 0;
    bool key_frame = false;
    bool intra_only = false;
    bool last_frame_was_key = false;
    bool error_resilient = false;
    bool last_frame_shown = false;
    bool show_frame = true;
    bool lossless = false;
    bool high_precision_mv = false;
    bool segmentation_enabled = false;
    bool mode_ref_delta_enabled = false;
};

// Bit-exact boolean (range) coder for the compressed header.
class Vp9RangeWriter {
public:
    Vp9RangeWriter();
    void write(std::int32_t value, std::int32_t bits);
    void write(bool bit);
    void write(bool bit, std::int32_t probability);
    void end();
    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }

private:
    void put(std::uint8_t value);
    std::vector<std::uint8_t> bytes_;
    std::size_t position_ = 0;
    std::uint32_t low_ = 0;
    std::uint32_t range_ = 0xff;
    std::int32_t count_ = -24;
};

// Most-significant-bit-first writer for the uncompressed header.
class Vp9BitWriter {
public:
    void write_unsigned(std::uint32_t value, std::uint32_t bits);
    void write_signed(std::int32_t value, std::uint32_t bits);
    void write_delta_q(std::uint32_t value);
    void write_bit(bool bit);
    void flush();
    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }

private:
    void write_bits(std::uint32_t value, std::uint32_t count);
    std::int32_t free_bits();
    std::int32_t buffer_ = 0;
    std::int32_t buffer_position_ = 0;
    std::vector<std::uint8_t> bytes_;
};

// Builds complete VP9 pictures (uncompressed header, compressed header,
// tile data) from what the guest gives the decoder engine. A picture's show
// flag is only known from the next picture's description, so each picture is
// composed one execution late; the first is composed twice.
class Vp9Composer {
public:
    explicit Vp9Composer(const DeviceMemory& memory) : memory_(memory) {}
    // Composes the picture for one execution. `hidden` is set for a picture
    // that is not shown.
    bool compose(const DecoderRegisters& registers, std::vector<std::uint8_t>& picture, bool& hidden,
                 std::string& failure);
    const Vp9FrameSize& frame_size() const noexcept { return current_.frame_size; }

private:
    bool read(std::uint64_t address, void* destination, std::size_t size, const char* what, std::string& failure);
    bool read_picture(const DecoderRegisters& registers, Vp9Picture& picture, std::string& failure);
    void write_probability_update(Vp9RangeWriter& writer, std::uint8_t updated, std::uint8_t previous);
    template <std::size_t N>
    void write_probability_updates(Vp9RangeWriter& writer, const std::array<std::uint8_t, N>& updated,
                                   const std::array<std::uint8_t, N>& previous);
    template <std::size_t N>
    void write_probability_updates_aligned4(Vp9RangeWriter& writer, const std::array<std::uint8_t, N>& updated,
                                            const std::array<std::uint8_t, N>& previous);
    void write_probability_delta(Vp9RangeWriter& writer, std::uint8_t updated, std::uint8_t previous);
    void encode_term_subexp(Vp9RangeWriter& writer, std::int32_t value);
    bool write_less_than(Vp9RangeWriter& writer, std::int32_t value, std::int32_t test);
    void write_coefficient_updates(Vp9RangeWriter& writer, std::int32_t tx_mode,
                                   const std::array<std::uint8_t, 1728>& updated,
                                   const std::array<std::uint8_t, 1728>& previous);
    void write_mv_probability_update(Vp9RangeWriter& writer, std::uint8_t updated, std::uint8_t previous);
    bool write_segmentation(Vp9BitWriter& writer, const DecoderRegisters& registers, std::string& failure);
    std::vector<std::uint8_t> compressed_header();
    bool uncompressed_header(Vp9BitWriter& writer, const DecoderRegisters& registers, std::string& failure);

    DeviceMemory memory_;
    std::array<std::int8_t, 4> loop_filter_ref_deltas_{};
    std::array<std::int8_t, 2> loop_filter_mode_deltas_{};
    struct Pending {
        Vp9Picture picture;
        std::vector<std::uint8_t> bitstream;
    };
    Pending next_{};
    std::array<Vp9Probabilities, 4> frame_contexts_{};
    bool swap_ref_indices_ = false;
    Vp9Segmentation last_segmentation_{};
    Vp9GuestPicture guest_picture_{};
    Vp9Picture current_{};
    Vp9Probabilities previous_probabilities_{};
};

} // namespace armigo::host1x
