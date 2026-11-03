// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "vp9.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace armigo::host1x {
namespace {

constexpr std::int32_t update_probability = 252;
constexpr std::uint32_t sync_code = 0x498342;

// The VP9 default probabilities a key frame resets every context to.
constexpr Vp9Probabilities default_probabilities{
    .y_mode{
        65, 32, 18, 144, 162, 194, 41, 51, 98, 132, 68, 18, 165, 217, 196, 45, 40, 78,
        173, 80, 19, 176, 240, 193, 64, 35, 46, 221, 135, 38, 194, 248, 121, 96, 85, 29,
    },
    .partition{
        199, 122, 141, 0, 147, 63, 159, 0, 148, 133, 118, 0, 121, 104, 114, 0, 174, 73,
        87, 0, 92, 41, 83, 0, 82, 99, 50, 0, 53, 39, 39, 0, 177, 58, 59, 0,
        68, 26, 63, 0, 52, 79, 25, 0, 17, 14, 12, 0, 222, 34, 30, 0, 72, 16,
        44, 0, 58, 32, 12, 0, 10, 7, 6, 0,
    },
    .coefficients{
        195, 29, 183, 84, 49, 136, 8, 42, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        31, 107, 169, 35, 99, 159, 17, 82, 140, 8, 66, 114, 2, 44, 76, 1, 19, 32,
        40, 132, 201, 29, 114, 187, 13, 91, 157, 7, 75, 127, 3, 58, 95, 1, 28, 47,
        69, 142, 221, 42, 122, 201, 15, 91, 159, 6, 67, 121, 1, 42, 77, 1, 17, 31,
        102, 148, 228, 67, 117, 204, 17, 82, 154, 6, 59, 114, 2, 39, 75, 1, 15, 29,
        156, 57, 233, 119, 57, 212, 58, 48, 163, 29, 40, 124, 12, 30, 81, 3, 12, 31,
        191, 107, 226, 124, 117, 204, 25, 99, 155, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        29, 148, 210, 37, 126, 194, 8, 93, 157, 2, 68, 118, 1, 39, 69, 1, 17, 33,
        41, 151, 213, 27, 123, 193, 3, 82, 144, 1, 58, 105, 1, 32, 60, 1, 13, 26,
        59, 159, 220, 23, 126, 198, 4, 88, 151, 1, 66, 114, 1, 38, 71, 1, 18, 34,
        114, 136, 232, 51, 114, 207, 11, 83, 155, 3, 56, 105, 1, 33, 65, 1, 17, 34,
        149, 65, 234, 121, 57, 215, 61, 49, 166, 28, 36, 114, 12, 25, 76, 3, 16, 42,
        214, 49, 220, 132, 63, 188, 42, 65, 137, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        85, 137, 221, 104, 131, 216, 49, 111, 192, 21, 87, 155, 2, 49, 87, 1, 16, 28,
        89, 163, 230, 90, 137, 220, 29, 100, 183, 10, 70, 135, 2, 42, 81, 1, 17, 33,
        108, 167, 237, 55, 133, 222, 15, 97, 179, 4, 72, 135, 1, 45, 85, 1, 19, 38,
        124, 146, 240, 66, 124, 224, 17, 88, 175, 4, 58, 122, 1, 36, 75, 1, 18, 37,
        141, 79, 241, 126, 70, 227, 66, 58, 182, 30, 44, 136, 12, 34, 96, 2, 20, 47,
        229, 99, 249, 143, 111, 235, 46, 109, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        82, 158, 236, 94, 146, 224, 25, 117, 191, 9, 87, 149, 3, 56, 99, 1, 33, 57,
        83, 167, 237, 68, 145, 222, 10, 103, 177, 2, 72, 131, 1, 41, 79, 1, 20, 39,
        99, 167, 239, 47, 141, 224, 10, 104, 178, 2, 73, 133, 1, 44, 85, 1, 22, 47,
        127, 145, 243, 71, 129, 228, 17, 93, 177, 3, 61, 124, 1, 41, 84, 1, 21, 52,
        157, 78, 244, 140, 72, 231, 69, 58, 184, 31, 44, 137, 14, 38, 105, 8, 23, 61,
        125, 34, 187, 52, 41, 133, 6, 31, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        37, 109, 153, 51, 102, 147, 23, 87, 128, 8, 67, 101, 1, 41, 63, 1, 19, 29,
        31, 154, 185, 17, 127, 175, 6, 96, 145, 2, 73, 114, 1, 51, 82, 1, 28, 45,
        23, 163, 200, 10, 131, 185, 2, 93, 148, 1, 67, 111, 1, 41, 69, 1, 14, 24,
        29, 176, 217, 12, 145, 201, 3, 101, 156, 1, 69, 111, 1, 39, 63, 1, 14, 23,
        57, 192, 233, 25, 154, 215, 6, 109, 167, 3, 78, 118, 1, 48, 69, 1, 21, 29,
        202, 105, 245, 108, 106, 216, 18, 90, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        33, 172, 219, 64, 149, 206, 14, 117, 177, 5, 90, 141, 2, 61, 95, 1, 37, 57,
        33, 179, 220, 11, 140, 198, 1, 89, 148, 1, 60, 104, 1, 33, 57, 1, 12, 21,
        30, 181, 221, 8, 141, 198, 1, 87, 145, 1, 58, 100, 1, 31, 55, 1, 12, 20,
        32, 186, 224, 7, 142, 198, 1, 86, 143, 1, 58, 100, 1, 31, 55, 1, 12, 22,
        57, 192, 227, 20, 143, 204, 3, 96, 154, 1, 68, 112, 1, 42, 69, 1, 19, 32,
        212, 35, 215, 113, 47, 169, 29, 48, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        74, 129, 203, 106, 120, 203, 49, 107, 178, 19, 84, 144, 4, 50, 84, 1, 15, 25,
        71, 172, 217, 44, 141, 209, 15, 102, 173, 6, 76, 133, 2, 51, 89, 1, 24, 42,
        64, 185, 231, 31, 148, 216, 8, 103, 175, 3, 74, 131, 1, 46, 81, 1, 18, 30,
        65, 196, 235, 25, 157, 221, 5, 105, 174, 1, 67, 120, 1, 38, 69, 1, 15, 30,
        65, 204, 238, 30, 156, 224, 7, 107, 177, 2, 70, 124, 1, 42, 73, 1, 18, 34,
        225, 86, 251, 144, 104, 235, 42, 99, 181, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        85, 175, 239, 112, 165, 229, 29, 136, 200, 12, 103, 162, 6, 77, 123, 2, 53, 84,
        75, 183, 239, 30, 155, 221, 3, 106, 171, 1, 74, 128, 1, 44, 76, 1, 17, 28,
        73, 185, 240, 27, 159, 222, 2, 107, 172, 1, 75, 127, 1, 42, 73, 1, 17, 29,
        62, 190, 238, 21, 159, 222, 2, 107, 172, 1, 72, 122, 1, 40, 71, 1, 18, 32,
        61, 199, 240, 27, 161, 226, 4, 113, 180, 1, 76, 129, 1, 46, 80, 1, 23, 41,
        7, 27, 153, 5, 30, 95, 1, 16, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        50, 75, 127, 57, 75, 124, 27, 67, 108, 10, 54, 86, 1, 33, 52, 1, 12, 18,
        43, 125, 151, 26, 108, 148, 7, 83, 122, 2, 59, 89, 1, 38, 60, 1, 17, 27,
        23, 144, 163, 13, 112, 154, 2, 75, 117, 1, 50, 81, 1, 31, 51, 1, 14, 23,
        18, 162, 185, 6, 123, 171, 1, 78, 125, 1, 51, 86, 1, 31, 54, 1, 14, 23,
        15, 199, 227, 3, 150, 204, 1, 91, 146, 1, 55, 95, 1, 30, 53, 1, 11, 20,
        19, 55, 240, 19, 59, 196, 3, 52, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        41, 166, 207, 104, 153, 199, 31, 123, 181, 14, 101, 152, 5, 72, 106, 1, 36, 52,
        35, 176, 211, 12, 131, 190, 2, 88, 144, 1, 60, 101, 1, 36, 60, 1, 16, 28,
        28, 183, 213, 8, 134, 191, 1, 86, 142, 1, 56, 96, 1, 30, 53, 1, 12, 20,
        20, 190, 215, 4, 135, 192, 1, 84, 139, 1, 53, 91, 1, 28, 49, 1, 11, 20,
        13, 196, 216, 2, 137, 192, 1, 86, 143, 1, 57, 99, 1, 32, 56, 1, 13, 24,
        211, 29, 217, 96, 47, 156, 22, 43, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        78, 120, 193, 111, 116, 186, 46, 102, 164, 15, 80, 128, 2, 49, 76, 1, 18, 28,
        71, 161, 203, 42, 132, 192, 10, 98, 150, 3, 69, 109, 1, 44, 70, 1, 18, 29,
        57, 186, 211, 30, 140, 196, 4, 93, 146, 1, 62, 102, 1, 38, 65, 1, 16, 27,
        47, 199, 217, 14, 145, 196, 1, 88, 142, 1, 57, 98, 1, 36, 62, 1, 15, 26,
        26, 219, 229, 5, 155, 207, 1, 94, 151, 1, 60, 104, 1, 36, 62, 1, 16, 28,
        233, 29, 248, 146, 47, 220, 43, 52, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        100, 163, 232, 179, 161, 222, 63, 142, 204, 37, 113, 174, 26, 89, 137, 18, 68, 97,
        85, 181, 230, 32, 146, 209, 7, 100, 164, 3, 71, 121, 1, 45, 77, 1, 18, 30,
        65, 187, 230, 20, 148, 207, 2, 97, 159, 1, 68, 116, 1, 40, 70, 1, 14, 29,
        40, 194, 227, 8, 147, 204, 1, 94, 155, 1, 65, 112, 1, 39, 66, 1, 14, 26,
        16, 208, 228, 3, 151, 207, 1, 98, 160, 1, 67, 117, 1, 41, 74, 1, 17, 31,
        17, 38, 140, 7, 34, 80, 1, 17, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        37, 75, 128, 41, 76, 128, 26, 66, 116, 12, 52, 94, 2, 32, 55, 1, 10, 16,
        50, 127, 154, 37, 109, 152, 16, 82, 121, 5, 59, 85, 1, 35, 54, 1, 13, 20,
        40, 142, 167, 17, 110, 157, 2, 71, 112, 1, 44, 72, 1, 27, 45, 1, 11, 17,
        30, 175, 188, 9, 124, 169, 1, 74, 116, 1, 48, 78, 1, 30, 49, 1, 11, 18,
        10, 222, 223, 2, 150, 194, 1, 83, 128, 1, 48, 79, 1, 27, 45, 1, 11, 17,
        36, 41, 235, 29, 36, 193, 10, 27, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        85, 165, 222, 177, 162, 215, 110, 135, 195, 57, 113, 168, 23, 83, 120, 10, 49, 61,
        85, 190, 223, 36, 139, 200, 5, 90, 146, 1, 60, 103, 1, 38, 65, 1, 18, 30,
        72, 202, 223, 23, 141, 199, 2, 86, 140, 1, 56, 97, 1, 36, 61, 1, 16, 27,
        55, 218, 225, 13, 145, 200, 1, 86, 141, 1, 57, 99, 1, 35, 61, 1, 13, 22,
        15, 235, 212, 1, 132, 184, 1, 84, 139, 1, 57, 97, 1, 34, 56, 1, 14, 23,
        181, 21, 201, 61, 37, 123, 10, 38, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        47, 106, 172, 95, 104, 173, 42, 93, 159, 18, 77, 131, 4, 50, 81, 1, 17, 23,
        62, 147, 199, 44, 130, 189, 28, 102, 154, 18, 75, 115, 2, 44, 65, 1, 12, 19,
        55, 153, 210, 24, 130, 194, 3, 93, 146, 1, 61, 97, 1, 31, 50, 1, 10, 16,
        49, 186, 223, 17, 148, 204, 1, 96, 142, 1, 53, 83, 1, 26, 44, 1, 11, 17,
        13, 217, 212, 2, 136, 180, 1, 78, 124, 1, 50, 83, 1, 29, 49, 1, 14, 23,
        197, 13, 247, 82, 17, 222, 25, 17, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        126, 186, 247, 234, 191, 243, 176, 177, 234, 104, 158, 220, 66, 128, 186, 55, 90, 137,
        111, 197, 242, 46, 158, 219, 9, 104, 171, 2, 65, 125, 1, 44, 80, 1, 17, 91,
        104, 208, 245, 39, 168, 224, 3, 109, 162, 1, 79, 124, 1, 50, 102, 1, 43, 102,
        84, 220, 246, 31, 177, 231, 2, 115, 180, 1, 79, 134, 1, 55, 77, 1, 60, 79,
        43, 243, 240, 8, 180, 217, 1, 115, 166, 1, 84, 121, 1, 51, 67, 1, 16, 6,
    },
    .switchable_interp{
        235, 162, 36, 255, 34, 3, 149, 144,
    },
    .inter_mode{
        2, 173, 34, 0, 7, 145, 85, 0, 7, 166, 63, 0, 7, 94, 66, 0, 8, 64,
        46, 0, 17, 81, 31, 0, 25, 29, 30, 0,
    },
    .intra_inter{
        9, 102, 187, 225,
    },
    .comp_inter{
        9, 102, 187, 225, 0,
    },
    .single_ref{
        33, 16, 77, 74, 142, 142, 172, 170, 238, 247,
    },
    .comp_ref{
        50, 126, 123, 221, 226,
    },
    .tx_32x32{
        3, 136, 37, 5, 52, 13,
    },
    .tx_16x16{
        20, 152, 15, 101,
    },
    .tx_8x8{
        100, 66,
    },
    .skip{
        192, 128, 64,
    },
    .mv_joints{
        32, 64, 96,
    },
    .mv_sign{
        128, 128,
    },
    .mv_classes{
        224, 144, 192, 168, 192, 176, 192, 198, 198, 245, 216, 128, 176, 160, 176, 176, 192, 198,
        198, 208,
    },
    .mv_class_0{
        216, 208,
    },
    .mv_bits{
        136, 140, 148, 160, 176, 192, 224, 234, 234, 240, 136, 140, 148, 160, 176, 192, 224, 234,
        234, 240,
    },
    .mv_class_0_fr{
        128, 128, 64, 96, 112, 64, 128, 128, 64, 96, 112, 64,
    },
    .mv_fr{
        64, 96, 64, 64, 96, 64,
    },
    .mv_class_0_hp{
        160, 160,
    },
    .mv_hp{
        128, 128,
    },
};

// Tile column limits (VP9 specification 6.2.14).
std::int32_t min_log2_tile_cols(std::int32_t width) {
    const std::int32_t columns = (width + 63) / 64;
    std::int32_t log2 = 0;
    while ((64 << log2) < columns) ++log2;
    return log2;
}

std::int32_t max_log2_tile_cols(std::int32_t width) {
    const std::int32_t columns = (width + 63) / 64;
    std::int32_t log2 = 1;
    while ((columns >> log2) >= 4) ++log2;
    return log2 - 1;
}

// Inverse of the probability recentring of the specification (6.3.6).
std::int32_t recenter_nonnegative(std::int32_t updated, std::int32_t previous) {
    if (updated > previous * 2) return updated;
    if (updated >= previous) return (updated - previous) * 2;
    return (previous - updated) * 2 - 1;
}

// Inverse of the probability remapping of the specification (6.3.5).
std::int32_t remap_probability(std::int32_t updated, std::int32_t previous) {
    --updated;
    --previous;
    const auto index = previous * 2 <= 0xff
        ? static_cast<std::uint8_t>(std::max(0, recenter_nonnegative(updated, previous) - 1))
        : static_cast<std::uint8_t>(std::max(0, recenter_nonnegative(0xff - 1 - updated, 0xff - 1 - previous) - 1));
    return (index + 7) % 13 == 0 ? (index + 7) / 13 - 1 : index + 20 - (index + 7) / 13;
}

// The engine's probability table in the order the compressed header uses.
void convert(const Vp9GuestProbabilities& guest, Vp9Probabilities& probabilities) {
    probabilities.inter_mode = guest.inter_mode;
    probabilities.intra_inter = guest.intra_inter;
    probabilities.tx_8x8 = guest.tx_8x8;
    probabilities.tx_16x16 = guest.tx_16x16;
    probabilities.tx_32x32 = guest.tx_32x32;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 9; ++j)
            probabilities.y_mode[j + 9 * i] = j < 8 ? guest.y_mode_e0e7[i][j] : guest.y_mode_e8[i];
    probabilities.partition = guest.partition;
    probabilities.switchable_interp = guest.switchable_interp;
    probabilities.comp_inter = guest.comp_inter;
    probabilities.skip = guest.skip;
    probabilities.mv_joints = guest.mv_joints;
    probabilities.mv_sign = guest.mv_sign;
    probabilities.mv_class_0 = guest.mv_class_0;
    probabilities.mv_fr = guest.mv_fr;
    probabilities.mv_class_0_hp = guest.mv_class_0_hp;
    probabilities.mv_hp = guest.mv_hp;
    probabilities.mv_classes = guest.mv_classes;
    probabilities.mv_class_0_fr = guest.mv_class_0_fr;
    probabilities.mv_bits = guest.mv_bits;
    probabilities.single_ref = guest.single_ref;
    probabilities.comp_ref = guest.comp_ref;
    // Each coefficient entry has a fourth, unused byte.
    for (std::size_t i = 0; i < guest.coefficients.size(); i += 4) {
        const std::size_t j = i - i / 4;
        probabilities.coefficients[j] = guest.coefficients[i];
        probabilities.coefficients[j + 1] = guest.coefficients[i + 1];
        probabilities.coefficients[j + 2] = guest.coefficients[i + 2];
    }
}

} // namespace

bool Vp9Composer::read(std::uint64_t address, void* destination, std::size_t size, const char* what,
                       std::string& failure) {
    if (memory_.read && memory_.read(memory_.context, address,
                                     std::span(static_cast<std::byte*>(destination), size)))
        return true;
    failure = std::string("Video decoder could not read its ") + what;
    return false;
}

bool Vp9Composer::read_picture(const DecoderRegisters& registers, Vp9Picture& picture, std::string& failure) {
    if (!read(registers.address(DecoderRegisters::picture_info), &guest_picture_, sizeof(guest_picture_),
              "VP9 picture description", failure))
        return false;
    const auto& guest = guest_picture_;
    picture = {};
    picture.bitstream_size = guest.bitstream_size;
    picture.ref_frame_sign_bias = guest.ref_frame_sign_bias;
    picture.base_q_index = guest.base_q_index;
    picture.y_dc_delta_q = guest.y_dc_delta_q;
    picture.uv_dc_delta_q = guest.uv_dc_delta_q;
    picture.uv_ac_delta_q = guest.uv_ac_delta_q;
    picture.transform_mode = guest.tx_mode;
    picture.interp_filter = guest.interp_filter;
    picture.reference_mode = guest.reference_mode;
    picture.log2_tile_cols = guest.log2_tile_cols;
    picture.log2_tile_rows = guest.log2_tile_rows;
    picture.ref_deltas = guest.loop_filter.ref_deltas;
    picture.mode_deltas = guest.loop_filter.mode_deltas;
    picture.frame_size = guest.current_frame_size;
    picture.first_level = guest.first_level;
    picture.sharpness_level = guest.sharpness_level;
    picture.key_frame = (guest.flags & 1u) != 0;
    picture.last_frame_was_key = (guest.flags & 2u) != 0;
    picture.error_resilient = (guest.flags & 8u) != 0;
    picture.last_frame_shown = (guest.flags & 16u) != 0;
    picture.intra_only = (guest.flags & 32u) != 0;
    picture.show_frame = true;
    picture.lossless = guest.lossless != 0;
    picture.high_precision_mv = guest.allow_high_precision_mv != 0;
    picture.segmentation_enabled = guest.segmentation.enabled != 0;
    picture.mode_ref_delta_enabled = guest.loop_filter.mode_ref_delta_enabled != 0;
    Vp9GuestProbabilities probabilities;
    if (!read(registers.address(DecoderRegisters::vp9_probabilities), &probabilities, sizeof(probabilities),
              "VP9 probability table", failure))
        return false;
    convert(probabilities, picture.probabilities);
    // The first four luma surfaces are the last, golden, alt-ref and current pictures.
    for (std::size_t i = 0; i < 4; ++i)
        picture.surfaces[i] = registers.address(DecoderRegisters::luma_surfaces + i);
    return true;
}

void Vp9Composer::write_probability_update(Vp9RangeWriter& writer, std::uint8_t updated, std::uint8_t previous) {
    const bool update = updated != previous;
    writer.write(update, update_probability);
    if (update) write_probability_delta(writer, updated, previous);
}

template <std::size_t N>
void Vp9Composer::write_probability_updates(Vp9RangeWriter& writer, const std::array<std::uint8_t, N>& updated,
                                            const std::array<std::uint8_t, N>& previous) {
    for (std::size_t i = 0; i < N; ++i) write_probability_update(writer, updated[i], previous[i]);
}

template <std::size_t N>
void Vp9Composer::write_probability_updates_aligned4(Vp9RangeWriter& writer,
                                                     const std::array<std::uint8_t, N>& updated,
                                                     const std::array<std::uint8_t, N>& previous) {
    for (std::size_t i = 0; i < N; i += 4) {
        write_probability_update(writer, updated[i], previous[i]);
        write_probability_update(writer, updated[i + 1], previous[i + 1]);
        write_probability_update(writer, updated[i + 2], previous[i + 2]);
    }
}

void Vp9Composer::write_probability_delta(Vp9RangeWriter& writer, std::uint8_t updated, std::uint8_t previous) {
    encode_term_subexp(writer, remap_probability(updated, previous));
}

void Vp9Composer::encode_term_subexp(Vp9RangeWriter& writer, std::int32_t value) {
    if (write_less_than(writer, value, 16)) {
        writer.write(value, 4);
    } else if (write_less_than(writer, value, 32)) {
        writer.write(value - 16, 4);
    } else if (write_less_than(writer, value, 64)) {
        writer.write(value - 32, 5);
    } else {
        value -= 64;
        constexpr std::int32_t size = 8;
        const std::int32_t mask = (1 << size) - 191;
        const std::int32_t delta = value - mask;
        if (delta < 0) {
            writer.write(value, size - 1);
        } else {
            writer.write(delta / 2 + mask, size - 1);
            writer.write(delta & 1, 1);
        }
    }
}

bool Vp9Composer::write_less_than(Vp9RangeWriter& writer, std::int32_t value, std::int32_t test) {
    const bool less = value < test;
    writer.write(!less);
    return less;
}

void Vp9Composer::write_coefficient_updates(Vp9RangeWriter& writer, std::int32_t tx_mode,
                                            const std::array<std::uint8_t, 1728>& updated,
                                            const std::array<std::uint8_t, 1728>& previous) {
    constexpr std::uint32_t block_bytes = 2 * 2 * 6 * 6 * 3;
    for (std::uint32_t block = 0; block < 4; ++block) {
        const std::uint32_t base = block * block_bytes;
        const bool update = !std::equal(updated.begin() + base, updated.begin() + base + block_bytes,
                                        previous.begin() + base);
        writer.write(update);
        if (update) {
            std::uint32_t index = base;
            for (std::int32_t i = 0; i < 2; ++i)
                for (std::int32_t j = 0; j < 2; ++j)
                    for (std::int32_t k = 0; k < 6; ++k)
                        for (std::int32_t l = 0; l < 6; ++l) {
                            if (k != 0 || l < 3) {
                                write_probability_update(writer, updated[index], previous[index]);
                                write_probability_update(writer, updated[index + 1], previous[index + 1]);
                                write_probability_update(writer, updated[index + 2], previous[index + 2]);
                            }
                            index += 3;
                        }
        }
        if (block == static_cast<std::uint32_t>(tx_mode)) break;
    }
}

void Vp9Composer::write_mv_probability_update(Vp9RangeWriter& writer, std::uint8_t updated, std::uint8_t previous) {
    const bool update = updated != previous;
    writer.write(update, update_probability);
    if (update) writer.write(updated >> 1, 7);
}

bool Vp9Composer::write_segmentation(Vp9BitWriter& writer, const DecoderRegisters& registers,
                                     std::string& failure) {
    const auto& segmentation = guest_picture_.segmentation;
    const bool enabled = segmentation.enabled != 0;
    writer.write_bit(enabled);
    if (!enabled) return true;
    const bool update_map = segmentation.update_map != 0;
    writer.write_bit(update_map);
    if (update_map) {
        Vp9GuestProbabilities probabilities;
        if (!read(registers.address(DecoderRegisters::vp9_probabilities), &probabilities, sizeof(probabilities),
                  "VP9 probability table", failure))
            return false;
        const auto write_probability = [&](std::uint8_t probability) {
            const bool coded = probability != 255;
            writer.write_bit(coded);
            if (coded) writer.write_unsigned(probability, 8);
        };
        for (const auto probability : probabilities.segment_tree) write_probability(probability);
        const bool temporal_update = segmentation.temporal_update != 0;
        writer.write_bit(temporal_update);
        if (temporal_update)
            for (const auto probability : probabilities.segment_pred) write_probability(probability);
    }
    if (last_segmentation_ == segmentation) {
        writer.write_bit(false);
        return true;
    }
    last_segmentation_ = segmentation;
    writer.write_bit(true);
    writer.write_bit(segmentation.abs_delta != 0);
    constexpr std::array<std::uint32_t, 4> feature_bits{8, 6, 2, 0};
    for (std::size_t i = 0; i < 8; ++i) {
        const bool quantizer = segmentation.feature_enabled[i][0] != 0;
        writer.write_bit(quantizer);
        if (quantizer) writer.write_signed(segmentation.feature_data[i][0], feature_bits[0]);
        const bool loop_filter = segmentation.feature_enabled[i][1] != 0;
        writer.write_bit(loop_filter);
        if (loop_filter) writer.write_signed(segmentation.feature_data[i][1], feature_bits[1]);
        const bool reference = segmentation.feature_enabled[i][2] != 0;
        writer.write_bit(reference);
        if (reference)
            writer.write_unsigned(static_cast<std::uint32_t>(segmentation.feature_data[i][2]), feature_bits[2]);
        writer.write_bit(segmentation.feature_enabled[i][3] != 0);
    }
    return true;
}

std::vector<std::uint8_t> Vp9Composer::compressed_header() {
    Vp9RangeWriter writer;
    auto& current = current_;
    auto& previous = previous_probabilities_;
    const auto& probabilities = current.probabilities;
    const bool update_probabilities = !current.key_frame && current.show_frame;
    if (!current.lossless) {
        if (current.transform_mode >= 3) {
            writer.write(3, 2);
            writer.write(current.transform_mode == 4);
        } else {
            writer.write(current.transform_mode, 2);
        }
    }
    if (current.transform_mode == 4) {
        write_probability_updates(writer, probabilities.tx_8x8, previous.tx_8x8);
        write_probability_updates(writer, probabilities.tx_16x16, previous.tx_16x16);
        write_probability_updates(writer, probabilities.tx_32x32, previous.tx_32x32);
        if (update_probabilities) {
            previous.tx_8x8 = probabilities.tx_8x8;
            previous.tx_16x16 = probabilities.tx_16x16;
            previous.tx_32x32 = probabilities.tx_32x32;
        }
    }
    write_coefficient_updates(writer, current.transform_mode, probabilities.coefficients, previous.coefficients);
    write_probability_updates(writer, probabilities.skip, previous.skip);
    if (update_probabilities) {
        previous.coefficients = probabilities.coefficients;
        previous.skip = probabilities.skip;
    }
    if (!current.intra_only) {
        write_probability_updates_aligned4(writer, probabilities.inter_mode, previous.inter_mode);
        if (current.interp_filter == 4) {
            write_probability_updates(writer, probabilities.switchable_interp, previous.switchable_interp);
            if (update_probabilities) previous.switchable_interp = probabilities.switchable_interp;
        }
        write_probability_updates(writer, probabilities.intra_inter, previous.intra_inter);
        const auto& bias = current.ref_frame_sign_bias;
        if ((bias[1] & 1) != (bias[2] & 1) || (bias[1] & 1) != (bias[3] & 1)) {
            if (current.reference_mode >= 1) {
                writer.write(1, 1);
                writer.write(current.reference_mode == 2);
            } else {
                writer.write(0, 1);
            }
        }
        if (current.reference_mode == 2) {
            write_probability_updates(writer, probabilities.comp_inter, previous.comp_inter);
            if (update_probabilities) previous.comp_inter = probabilities.comp_inter;
        }
        if (current.reference_mode != 1) {
            write_probability_updates(writer, probabilities.single_ref, previous.single_ref);
            if (update_probabilities) previous.single_ref = probabilities.single_ref;
        }
        if (current.reference_mode != 0) {
            write_probability_updates(writer, probabilities.comp_ref, previous.comp_ref);
            if (update_probabilities) previous.comp_ref = probabilities.comp_ref;
        }
        for (std::size_t i = 0; i < probabilities.y_mode.size(); ++i)
            write_probability_update(writer, probabilities.y_mode[i], previous.y_mode[i]);
        write_probability_updates_aligned4(writer, probabilities.partition, previous.partition);
        for (std::size_t i = 0; i < 3; ++i)
            write_mv_probability_update(writer, probabilities.mv_joints[i], previous.mv_joints[i]);
        if (update_probabilities) {
            previous.inter_mode = probabilities.inter_mode;
            previous.intra_inter = probabilities.intra_inter;
            previous.y_mode = probabilities.y_mode;
            previous.partition = probabilities.partition;
            previous.mv_joints = probabilities.mv_joints;
        }
        for (std::size_t i = 0; i < 2; ++i) {
            write_mv_probability_update(writer, probabilities.mv_sign[i], previous.mv_sign[i]);
            for (std::size_t j = 0; j < 10; ++j)
                write_mv_probability_update(writer, probabilities.mv_classes[i * 10 + j],
                                            previous.mv_classes[i * 10 + j]);
            write_mv_probability_update(writer, probabilities.mv_class_0[i], previous.mv_class_0[i]);
            for (std::size_t j = 0; j < 10; ++j)
                write_mv_probability_update(writer, probabilities.mv_bits[i * 10 + j], previous.mv_bits[i * 10 + j]);
        }
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j)
                for (std::size_t k = 0; k < 3; ++k) {
                    const auto index = i * 6 + j * 3 + k;
                    write_mv_probability_update(writer, probabilities.mv_class_0_fr[index],
                                                previous.mv_class_0_fr[index]);
                }
            for (std::size_t j = 0; j < 3; ++j)
                write_mv_probability_update(writer, probabilities.mv_fr[i * 3 + j], previous.mv_fr[i * 3 + j]);
        }
        if (current.high_precision_mv) {
            for (std::size_t i = 0; i < 2; ++i) {
                write_mv_probability_update(writer, probabilities.mv_class_0_hp[i], previous.mv_class_0_hp[i]);
                write_mv_probability_update(writer, probabilities.mv_hp[i], previous.mv_hp[i]);
            }
        }
        if (update_probabilities) {
            previous.mv_sign = probabilities.mv_sign;
            previous.mv_classes = probabilities.mv_classes;
            previous.mv_class_0 = probabilities.mv_class_0;
            previous.mv_bits = probabilities.mv_bits;
            previous.mv_class_0_fr = probabilities.mv_class_0_fr;
            previous.mv_fr = probabilities.mv_fr;
            previous.mv_class_0_hp = probabilities.mv_class_0_hp;
            previous.mv_hp = probabilities.mv_hp;
        }
    }
    writer.end();
    return writer.bytes();
}

bool Vp9Composer::uncompressed_header(Vp9BitWriter& writer, const DecoderRegisters& registers,
                                      std::string& failure) {
    auto& current = current_;
    writer.write_unsigned(2, 2);  // frame marker
    writer.write_unsigned(0, 2);  // profile 0
    writer.write_bit(false);      // show existing frame
    writer.write_bit(!current.key_frame);
    writer.write_bit(current.show_frame);
    writer.write_bit(current.error_resilient);
    if (current.key_frame) {
        writer.write_unsigned(sync_code, 24);
        writer.write_unsigned(0, 3);  // colour space
        writer.write_unsigned(0, 1);  // colour range
        writer.write_unsigned(static_cast<std::uint32_t>(current.frame_size.width - 1), 16);
        writer.write_unsigned(static_cast<std::uint32_t>(current.frame_size.height - 1), 16);
        writer.write_bit(false);  // render size equals frame size
        // A key frame resets every context.
        previous_probabilities_ = default_probabilities;
        swap_ref_indices_ = false;
        loop_filter_ref_deltas_.fill(0);
        loop_filter_mode_deltas_.fill(0);
        frame_contexts_.fill(default_probabilities);
        current.intra_only = true;
    } else {
        if (!current.show_frame) writer.write_bit(current.intra_only);
        else current.intra_only = false;
        if (!current.error_resilient) writer.write_unsigned(0, 2);  // reset frame context
        const auto& surfaces = current.surfaces;
        const auto& next_surfaces = next_.picture.surfaces;
        const bool references_differ = surfaces[1] != surfaces[2];
        const bool next_swaps = next_surfaces[1] == surfaces[2] || next_surfaces[2] == surfaces[1];
        const bool needs_swap = references_differ && next_swaps;
        if (needs_swap) swap_ref_indices_ = !swap_ref_indices_;
        // Refresh the references the next picture names at this picture's surface.
        std::uint32_t refresh = 0;
        for (std::uint32_t i = 0; i < 3; ++i)
            if (surfaces[3] == next_surfaces[i]) refresh |= 1u << i;
        if (swap_ref_indices_) {
            // The two-bit golden field (bits 1..2) takes the alt-ref bit (bit 2) and
            // the alt-ref bit takes the old golden field's low bit, as the bit
            // fields of the reference overlap.
            const std::uint32_t golden = (refresh >> 1) & 3u;
            const std::uint32_t alt = (refresh >> 2) & 1u;
            refresh = (refresh & ~0x6u) | (alt << 1);
            refresh = (refresh & ~0x4u) | ((golden & 1u) << 2);
        }
        if (current.intra_only) {
            writer.write_unsigned(sync_code, 24);
            writer.write_unsigned(refresh, 8);
            writer.write_unsigned(static_cast<std::uint32_t>(current.frame_size.width - 1), 16);
            writer.write_unsigned(static_cast<std::uint32_t>(current.frame_size.height - 1), 16);
            writer.write_bit(false);
        } else {
            const bool swap_indices = needs_swap ^ swap_ref_indices_;
            const std::array<std::uint32_t, 3> ref_index =
                swap_indices ? std::array<std::uint32_t, 3>{0, 2, 1} : std::array<std::uint32_t, 3>{0, 1, 2};
            writer.write_unsigned(refresh, 8);
            for (std::size_t i = 1; i < 4; ++i) {
                writer.write_unsigned(ref_index[i - 1], 3);
                writer.write_unsigned(static_cast<std::uint32_t>(current.ref_frame_sign_bias[i]), 1);
            }
            writer.write_bit(true);   // frame size from references
            writer.write_bit(false);  // render size equals frame size
            writer.write_bit(current.high_precision_mv);
            writer.write_bit(current.interp_filter == 4);
            if (current.interp_filter != 4)
                writer.write_unsigned(static_cast<std::uint32_t>(current.interp_filter), 2);
        }
    }
    if (!current.error_resilient) {
        writer.write_bit(true);  // refresh frame context
        writer.write_bit(true);  // frame-parallel decoding mode
    }
    const std::uint32_t context = current.show_frame ? 0 : 1;
    writer.write_unsigned(context, 2);
    previous_probabilities_ = frame_contexts_[context];
    frame_contexts_[context] = current.probabilities;
    writer.write_unsigned(current.first_level, 6);
    writer.write_unsigned(current.sharpness_level, 3);
    writer.write_bit(current.mode_ref_delta_enabled);
    if (current.mode_ref_delta_enabled) {
        std::array<bool, 4> update_ref{};
        std::array<bool, 2> update_mode{};
        bool update = false;
        for (std::size_t i = 0; i < 4; ++i) {
            update_ref[i] = loop_filter_ref_deltas_[i] != current.ref_deltas[i];
            update |= update_ref[i];
        }
        for (std::size_t i = 0; i < 2; ++i) {
            update_mode[i] = loop_filter_mode_deltas_[i] != current.mode_deltas[i];
            update |= update_mode[i];
        }
        writer.write_bit(update);
        if (update) {
            for (std::size_t i = 0; i < 4; ++i) {
                writer.write_bit(update_ref[i]);
                if (update_ref[i]) writer.write_signed(current.ref_deltas[i], 6);
            }
            for (std::size_t i = 0; i < 2; ++i) {
                writer.write_bit(update_mode[i]);
                if (update_mode[i]) writer.write_signed(current.mode_deltas[i], 6);
            }
            loop_filter_ref_deltas_ = current.ref_deltas;
            loop_filter_mode_deltas_ = current.mode_deltas;
        }
    }
    writer.write_unsigned(static_cast<std::uint32_t>(current.base_q_index), 8);
    writer.write_delta_q(static_cast<std::uint32_t>(current.y_dc_delta_q));
    writer.write_delta_q(static_cast<std::uint32_t>(current.uv_dc_delta_q));
    writer.write_delta_q(static_cast<std::uint32_t>(current.uv_ac_delta_q));
    if (!write_segmentation(writer, registers, failure)) return false;
    const auto minimum = min_log2_tile_cols(current.frame_size.width);
    const auto maximum = max_log2_tile_cols(current.frame_size.width);
    const auto difference = current.log2_tile_cols - minimum;
    const auto increments = (1 << difference) - 1;
    // Below the maximum a terminating zero bit follows the increments.
    if (current.log2_tile_cols < maximum)
        writer.write_unsigned(static_cast<std::uint32_t>(increments << 1), static_cast<std::uint32_t>(difference + 1));
    else
        writer.write_unsigned(static_cast<std::uint32_t>(increments), static_cast<std::uint32_t>(difference));
    const bool tile_rows = current.log2_tile_rows != 0;
    writer.write_bit(tile_rows);
    if (tile_rows) writer.write_bit(current.log2_tile_rows > 1);
    return true;
}

bool Vp9Composer::compose(const DecoderRegisters& registers, std::vector<std::uint8_t>& picture, bool& hidden,
                          std::string& failure) {
    Pending incoming;
    if (!read_picture(registers, incoming.picture, failure)) return false;
    if (incoming.picture.bitstream_size > (64u << 20)) {
        failure = "VP9 picture bitstream is larger than 64 MiB";
        return false;
    }
    incoming.bitstream.resize(incoming.picture.bitstream_size);
    if (!incoming.bitstream.empty() &&
        !read(registers.address(DecoderRegisters::bitstream), incoming.bitstream.data(), incoming.bitstream.size(),
              "VP9 bitstream", failure))
        return false;
    // Compose the previous picture, now that this one says whether it was shown.
    Pending composed;
    if (!next_.bitstream.empty()) {
        next_.picture.show_frame = incoming.picture.last_frame_shown;
        composed = std::move(next_);
        next_ = std::move(incoming);
    } else {
        next_ = incoming;
        composed = std::move(incoming);
    }
    current_ = composed.picture;
    Vp9BitWriter uncompressed;
    if (!uncompressed_header(uncompressed, registers, failure)) return false;
    const auto compressed = compressed_header();
    uncompressed.write_unsigned(static_cast<std::uint32_t>(compressed.size()), 16);
    uncompressed.flush();
    const auto& header = uncompressed.bytes();
    picture.clear();
    picture.reserve(header.size() + compressed.size() + composed.bitstream.size());
    picture.insert(picture.end(), header.begin(), header.end());
    picture.insert(picture.end(), compressed.begin(), compressed.end());
    picture.insert(picture.end(), composed.bitstream.begin(), composed.bitstream.end());
    hidden = !current_.show_frame;
    return true;
}

Vp9RangeWriter::Vp9RangeWriter() { write(false); }

void Vp9RangeWriter::write(std::int32_t value, std::int32_t bits) {
    for (std::int32_t bit = bits - 1; bit >= 0; --bit) write(((value >> bit) & 1) != 0);
}

void Vp9RangeWriter::write(bool bit) { write(bit, 128); }

void Vp9RangeWriter::put(std::uint8_t value) {
    if (position_ == bytes_.size()) bytes_.push_back(value);
    else bytes_[position_] = value;
    ++position_;
}

void Vp9RangeWriter::write(bool bit, std::int32_t probability) {
    std::uint32_t range = range_;
    const std::uint32_t split = 1 + (((range - 1) * static_cast<std::uint32_t>(probability)) >> 8);
    range = split;
    if (bit) {
        low_ += split;
        range = range_ - split;
    }
    std::int32_t shift = range == 0 ? 0 : std::countl_zero(range) - 24;
    range <<= shift;
    count_ += shift;
    if (count_ >= 0) {
        const std::int32_t offset = shift - count_;
        if (((low_ << (offset - 1)) >> 31) != 0) {
            // Carry into the bytes already written.
            std::size_t at = position_ - 1;
            while (bytes_[at] == 0xff) {
                bytes_[at] = 0;
                --at;
            }
            ++bytes_[at];
        }
        put(static_cast<std::uint8_t>(low_ >> (24 - offset)));
        low_ <<= offset;
        shift = count_;
        low_ &= 0xffffff;
        count_ -= 8;
    }
    low_ <<= shift;
    range_ = range;
}

void Vp9RangeWriter::end() {
    for (int i = 0; i < 32; ++i) write(false);
}

void Vp9BitWriter::write_unsigned(std::uint32_t value, std::uint32_t bits) { write_bits(value, bits); }

void Vp9BitWriter::write_signed(std::int32_t value, std::uint32_t bits) {
    const bool negative = value < 0;
    if (negative) value = -value;
    write_bits(static_cast<std::uint32_t>(value << 1) | (negative ? 1u : 0u), bits + 1);
}

void Vp9BitWriter::write_delta_q(std::uint32_t value) {
    const bool coded = value != 0;
    write_bit(coded);
    if (coded) write_bits(value, 4);
}

void Vp9BitWriter::write_bit(bool bit) { write_bits(bit ? 1 : 0, 1); }

void Vp9BitWriter::write_bits(std::uint32_t value, std::uint32_t count) {
    std::int32_t position = 0;
    std::int32_t remaining = static_cast<std::int32_t>(count);
    while (remaining > 0) {
        std::int32_t size = remaining;
        const std::int32_t available = free_bits();
        if (size > available) size = available;
        const std::int32_t mask = (1 << size) - 1;
        const std::int32_t source_shift = (static_cast<std::int32_t>(count) - position) - size;
        const std::int32_t destination_shift = (8 - buffer_position_) - size;
        buffer_ |= static_cast<std::int32_t>((value >> source_shift) & static_cast<std::uint32_t>(mask))
                   << destination_shift;
        position += size;
        buffer_position_ += size;
        remaining -= size;
    }
}

std::int32_t Vp9BitWriter::free_bits() {
    if (buffer_position_ == 8) flush();
    return 8 - buffer_position_;
}

void Vp9BitWriter::flush() {
    if (buffer_position_ == 0) return;
    bytes_.push_back(static_cast<std::uint8_t>(buffer_));
    buffer_ = 0;
    buffer_position_ = 0;
}

} // namespace armigo::host1x
