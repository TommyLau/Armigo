// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/host1x/video_decoder.h>

namespace armigo::host1x {
namespace {

// Reads AV1 syntax elements f(n) and uvlc() from one OBU payload.
class BitReader {
public:
    explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}
    bool ok() const { return ok_; }
    std::uint32_t f(unsigned bits) {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < bits; ++i) {
            if (position_ >= bytes_.size() * 8) {
                ok_ = false;
                return 0;
            }
            const auto bit = (bytes_[position_ / 8] >> (7 - position_ % 8)) & 1;
            value = (value << 1) | bit;
            ++position_;
        }
        return value;
    }
    void uvlc() {
        unsigned leading_zeros = 0;
        while (ok_ && !f(1)) ++leading_zeros;
        if (leading_zeros < 32) f(leading_zeros);
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t position_ = 0;
    bool ok_ = true;
};

struct SequenceFields {
    std::uint32_t profile = 0, level = 0, tier = 0, high_bitdepth = 0, twelve_bit = 0, monochrome = 0;
    std::uint32_t subsampling_x = 1, subsampling_y = 1, chroma_sample_position = 0;
};

// sequence_header_obu() up to and including color_config(), the fields the
// configuration record repeats (AV1 specification 5.5).
bool parse_sequence_header(std::span<const std::uint8_t> payload, SequenceFields& out) {
    BitReader r(payload);
    out.profile = r.f(3);
    r.f(1);  // still_picture
    const auto reduced = r.f(1);
    if (reduced) {
        out.level = r.f(5);
    } else {
        bool decoder_model = false;
        unsigned buffer_delay_bits = 0;
        if (r.f(1)) {  // timing_info_present_flag
            r.f(32);
            r.f(32);
            if (r.f(1)) r.uvlc();  // equal_picture_interval
            decoder_model = r.f(1);
            if (decoder_model) {
                buffer_delay_bits = r.f(5) + 1;
                r.f(32);
                r.f(5);
                r.f(5);
            }
        }
        const auto initial_display_delay = r.f(1);
        const auto operating_points = r.f(5) + 1;
        for (std::uint32_t i = 0; i < operating_points; ++i) {
            r.f(12);  // operating_point_idc
            const auto level = r.f(5);
            const auto tier = level > 7 ? r.f(1) : 0;
            if (i == 0) {
                out.level = level;
                out.tier = tier;
            }
            if (decoder_model && r.f(1)) {
                r.f(buffer_delay_bits);
                r.f(buffer_delay_bits);
                r.f(1);
            }
            if (initial_display_delay && r.f(1)) r.f(4);
        }
    }
    const auto width_bits = r.f(4) + 1, height_bits = r.f(4) + 1;
    r.f(width_bits);
    r.f(height_bits);
    if (!reduced && r.f(1)) {  // frame_id_numbers_present_flag
        r.f(4);
        r.f(3);
    }
    r.f(1);  // use_128x128_superblock
    r.f(1);  // enable_filter_intra
    r.f(1);  // enable_intra_edge_filter
    if (!reduced) {
        r.f(4);  // interintra, masked compound, warped motion, dual filter
        const auto order_hint = r.f(1);
        if (order_hint) r.f(2);  // jnt_comp, ref_frame_mvs
        std::uint32_t force_screen_content_tools = 2;
        if (!r.f(1)) force_screen_content_tools = r.f(1);  // seq_choose_screen_content_tools
        if (force_screen_content_tools > 0 && !r.f(1)) r.f(1);  // seq_choose_integer_mv
        if (order_hint) r.f(3);
    }
    r.f(3);  // superres, cdef, restoration
    // color_config()
    out.high_bitdepth = r.f(1);
    if (out.profile == 2 && out.high_bitdepth) out.twelve_bit = r.f(1);
    out.monochrome = out.profile == 1 ? 0 : r.f(1);
    std::uint32_t primaries = 2, transfer = 2, matrix = 2;
    if (r.f(1)) {  // color_description_present_flag
        primaries = r.f(8);
        transfer = r.f(8);
        matrix = r.f(8);
    }
    if (out.monochrome) {
        r.f(1);
        out.subsampling_x = out.subsampling_y = 1;
    } else if (primaries == 1 && transfer == 13 && matrix == 0) {
        out.subsampling_x = out.subsampling_y = 0;
    } else {
        r.f(1);  // color_range
        if (out.profile == 0) {
            out.subsampling_x = out.subsampling_y = 1;
        } else if (out.profile == 1) {
            out.subsampling_x = out.subsampling_y = 0;
        } else if (out.twelve_bit) {
            out.subsampling_x = r.f(1);
            out.subsampling_y = out.subsampling_x ? r.f(1) : 0;
        } else {
            out.subsampling_x = 1;
            out.subsampling_y = 0;
        }
        if (out.subsampling_x && out.subsampling_y) out.chroma_sample_position = r.f(2);
    }
    return r.ok();
}

} // namespace

std::optional<std::vector<std::uint8_t>> av1_configuration_record(std::span<const std::uint8_t> obus) {
    std::size_t offset = 0;
    while (offset < obus.size()) {
        const auto header = obus[offset];
        const auto type = (header >> 3) & 15;
        const bool extension = header & 4, has_size = header & 2;
        if ((header & 0x80) || !has_size) return std::nullopt;
        std::size_t cursor = offset + 1 + (extension ? 1 : 0);
        std::uint64_t size = 0;
        for (unsigned i = 0; i < 8; ++i) {
            if (cursor >= obus.size()) return std::nullopt;
            const auto byte = obus[cursor++];
            size |= std::uint64_t{byte & 0x7fu} << (7 * i);
            if (!(byte & 0x80)) break;
        }
        if (size > obus.size() - cursor) return std::nullopt;
        if (type == 1) {  // OBU_SEQUENCE_HEADER
            SequenceFields fields;
            if (!parse_sequence_header(obus.subspan(cursor, size), fields)) return std::nullopt;
            std::vector<std::uint8_t> record{
                0x81,  // marker, version 1
                static_cast<std::uint8_t>(fields.profile << 5 | fields.level),
                static_cast<std::uint8_t>(fields.tier << 7 | fields.high_bitdepth << 6 | fields.twelve_bit << 5 |
                                          fields.monochrome << 4 | fields.subsampling_x << 3 |
                                          fields.subsampling_y << 2 | fields.chroma_sample_position),
                0};
            record.insert(record.end(), obus.begin() + static_cast<std::ptrdiff_t>(offset),
                          obus.begin() + static_cast<std::ptrdiff_t>(cursor + size));
            return record;
        }
        offset = cursor + size;
    }
    return std::nullopt;
}

} // namespace armigo::host1x
