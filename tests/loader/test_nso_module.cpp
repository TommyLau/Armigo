// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/nso_module.h>

#include "nso_fixture.h"

#include <algorithm>
#include <cstring>

namespace {

using namespace armigo::loader;
using namespace armigo::loader::test;

bool image_matches(const NsoModuleImage& image, const FixtureSpec& spec) {
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& content = spec.segments[i].content;
        const auto offset = spec.segments[i].image_offset;
        if (!std::equal(content.begin(), content.end(), image.bytes.begin() + offset)) {
            return false;
        }
    }
    return true;
}

bool zero_outside_content(const NsoModuleImage& image) {
    std::vector<bool> covered(image.bytes.size(), false);
    for (const auto& segment : image.segments) {
        std::fill_n(covered.begin() + static_cast<std::ptrdiff_t>(segment.image_offset),
                    segment.content_size, true);
    }
    for (std::size_t i = 0; i < image.bytes.size(); ++i) {
        if (!covered[i] && image.bytes[i] != std::byte{0}) {
            return false;
        }
    }
    return true;
}

NsoError error_of(const std::vector<std::byte>& file, const NsoReadOptions& options = {}) {
    const auto result = read_nso_module(file, options);
    return result.ok() ? NsoError::success : result.error.code;
}

void reads_build_identity() {
    auto file = build_nso(default_spec(false, false));
    std::array<std::byte, 32> identity{};
    for (std::size_t i = 0; i < identity.size(); ++i) identity[i] = std::byte(i + 1);
    std::copy(identity.begin(), identity.end(), file.begin() + 0x40);
    const auto result = read_nso_module(file);
    ARMIGO_CHECK(result.ok());
    if (result.ok()) ARMIGO_CHECK(result.image->build_id == identity);
}

void reads_execute_only_module() {
    auto file = build_nso(default_spec(false, true));
    put_u32(file, kFieldFlags, get_u32(file, kFieldFlags) | 0x40);
    const auto result = read_nso_module(file);
    ARMIGO_CHECK(result.ok());
    if (result.ok()) {
        ARMIGO_CHECK(result.image->segment(NsoSegmentKind::text).permission == SegmentPermission::execute);
        ARMIGO_CHECK(result.image->segment(NsoSegmentKind::rodata).permission == SegmentPermission::read);
        ARMIGO_CHECK(result.image->segment(NsoSegmentKind::data).permission ==
                     (SegmentPermission::read | SegmentPermission::write));
    }
}

void reads_uncompressed_module() {
    const FixtureSpec spec = default_spec(false, false);
    const auto result = read_nso_module(build_nso(spec));
    ARMIGO_CHECK(result.ok());
    if (!result.ok()) {
        return;
    }
    const NsoModuleImage& image = *result.image;
    ARMIGO_CHECK(image.bytes.size() == 0x6000); // 0x3000 + 0x400 + 0x2345, page-rounded.
    ARMIGO_CHECK(image_matches(image, spec));
    ARMIGO_CHECK(zero_outside_content(image));
    ARMIGO_CHECK(image.bss_offset == 0x3400 && image.bss_size == 0x2345);

    const auto& text = image.segment(NsoSegmentKind::text);
    const auto& rodata = image.segment(NsoSegmentKind::rodata);
    const auto& data = image.segment(NsoSegmentKind::data);
    ARMIGO_CHECK(text.image_offset == 0 && text.content_size == 0x1800 && text.mapped_size == 0x2000);
    ARMIGO_CHECK(text.permission == (SegmentPermission::read | SegmentPermission::execute));
    ARMIGO_CHECK(rodata.image_offset == 0x2000 && rodata.mapped_size == 0x1000);
    ARMIGO_CHECK(rodata.permission == SegmentPermission::read);
    ARMIGO_CHECK(data.image_offset == 0x3000 && data.mapped_size == 0x3000);
    ARMIGO_CHECK(data.permission == (SegmentPermission::read | SegmentPermission::write));
    for (const auto& segment : image.segments) {
        ARMIGO_CHECK(!segment.compressed && segment.hash == HashCheck::not_checked);
        ARMIGO_CHECK(segment.file_size == segment.content_size);
    }
    ARMIGO_CHECK(image.api_info.offset == 0x10 && image.api_info.size == 0x20);
    ARMIGO_CHECK(image.dynstr.offset == 0x100 && image.dynsym.size == 0x300);
    ARMIGO_CHECK(!image.module_header.has_value());
}

void reads_compressed_module_with_hashes() {
    const FixtureSpec spec = default_spec(true, true);
    const auto file = build_nso(spec);
    // The fixture must really be compressed for the test to mean anything.
    ARMIGO_CHECK(get_u32(file, field_file_size(0)) < 0x1800);
    const auto result = read_nso_module(file);
    ARMIGO_CHECK(result.ok());
    if (!result.ok()) {
        return;
    }
    ARMIGO_CHECK(image_matches(*result.image, spec));
    ARMIGO_CHECK(zero_outside_content(*result.image));
    for (const auto& segment : result.image->segments) {
        ARMIGO_CHECK(segment.compressed && segment.hash == HashCheck::verified);
        ARMIGO_CHECK(segment.file_size < segment.content_size);
    }
}

void mixed_flags() {
    FixtureSpec spec = default_spec(false, false);
    spec.segments[1].compress = true;
    spec.segments[2].verify_hash = true;
    const auto result = read_nso_module(build_nso(spec));
    ARMIGO_CHECK(result.ok());
    if (result.ok()) {
        const auto& segments = result.image->segments;
        ARMIGO_CHECK(!segments[0].compressed && segments[0].hash == HashCheck::not_checked);
        ARMIGO_CHECK(segments[1].compressed && segments[1].hash == HashCheck::not_checked);
        ARMIGO_CHECK(!segments[2].compressed && segments[2].hash == HashCheck::verified);
        ARMIGO_CHECK(image_matches(*result.image, spec));
    }
}

void hash_policies() {
    auto file = build_nso(default_spec(true, false));
    NsoReadOptions all;
    all.hash_policy = NsoHashPolicy::verify_all;
    const auto verified = read_nso_module(file, all);
    ARMIGO_CHECK(verified.ok());
    if (verified.ok()) {
        for (const auto& segment : verified.image->segments) {
            ARMIGO_CHECK(segment.hash == HashCheck::verified);
        }
    }

    // An unflagged, wrong hash is ignored by default but rejected when every
    // segment must be verified.
    file[field_hash(1) + 5] ^= std::byte{0x01};
    ARMIGO_CHECK(read_nso_module(file).ok());
    const auto rejected = read_nso_module(file, all);
    ARMIGO_CHECK(!rejected.ok() && rejected.error.code == NsoError::hash_mismatch &&
                 rejected.error.segment == NsoSegmentKind::rodata);
    NsoReadOptions invalid;
    invalid.hash_policy = static_cast<NsoHashPolicy>(255);
    ARMIGO_CHECK(error_of(file, invalid) == NsoError::invalid_hash_policy);

    // A flagged, wrong hash is always rejected.
    auto flagged = build_nso(default_spec(false, true));
    flagged[field_hash(2) + 31] ^= std::byte{0x80};
    const auto mismatch = read_nso_module(flagged);
    ARMIGO_CHECK(!mismatch.ok() && mismatch.error.code == NsoError::hash_mismatch &&
                 mismatch.error.segment == NsoSegmentKind::data);
    // Corrupting content instead of the stored hash is caught the same way.
    auto content = build_nso(default_spec(false, true));
    content[get_u32(content, field_file_offset(0)) + 100] ^= std::byte{0x01};
    ARMIGO_CHECK(error_of(content) == NsoError::hash_mismatch);
}

void header_rejections() {
    const auto valid = build_nso(default_spec(true, true));
    for (std::size_t size = 0; size < kNsoHeaderSize; size += 17) {
        std::vector<std::byte> truncated(valid.begin(), valid.begin() + size);
        ARMIGO_CHECK(error_of(truncated) == NsoError::truncated_header);
    }

    auto magic = valid;
    magic[3] = std::byte{'1'};
    ARMIGO_CHECK(error_of(magic) == NsoError::bad_magic);

    auto version = valid;
    put_u32(version, kFieldVersion, 1);
    ARMIGO_CHECK(error_of(version) == NsoError::unsupported_version);

    for (std::uint32_t bit = 8; bit < 32; ++bit) {
        auto flags = valid;
        put_u32(flags, kFieldFlags, get_u32(valid, kFieldFlags) | (1u << bit));
        const auto result = read_nso_module(flags);
        ARMIGO_CHECK(!result.ok() && result.error.code == NsoError::unsupported_flags &&
                     result.error.detail == (1u << bit));
    }

    NsoReadOptions small;
    small.limits.max_input_bytes = valid.size() - 1;
    ARMIGO_CHECK(error_of(valid, small) == NsoError::input_too_large);
    small.limits.max_input_bytes = valid.size();
    ARMIGO_CHECK(error_of(valid, small) == NsoError::success);
}

void file_range_rejections() {
    const auto valid = build_nso(default_spec(true, false));

    // Any truncation of the payload area must fail without reading past the
    // input; most land in the data segment's file range.
    for (std::size_t size = kNsoHeaderSize; size < valid.size(); ++size) {
        std::vector<std::byte> truncated(valid.begin(), valid.begin() + size);
        ARMIGO_CHECK(error_of(truncated) == NsoError::segment_file_range_invalid);
    }

    auto inside_header = valid;
    put_u32(inside_header, field_file_offset(0), 0xFC);
    const auto header_result = read_nso_module(inside_header);
    ARMIGO_CHECK(!header_result.ok() &&
                 header_result.error.code == NsoError::segment_file_range_invalid &&
                 header_result.error.segment == NsoSegmentKind::text);

    auto wrapping = valid;
    put_u32(wrapping, field_file_offset(2), 0xFFFFFFF0);
    ARMIGO_CHECK(error_of(wrapping) == NsoError::segment_file_range_invalid);

    auto overlap = valid;
    put_u32(overlap, field_file_offset(1), get_u32(valid, field_file_offset(0)) + 1);
    ARMIGO_CHECK(error_of(overlap) == NsoError::segment_file_overlap);

    auto uncompressed = build_nso(default_spec(false, false));
    put_u32(uncompressed, field_file_size(1), 0x8FF);
    ARMIGO_CHECK(error_of(uncompressed) == NsoError::segment_size_mismatch);

    // An empty uncompressed segment may record any file offset; it stores
    // nothing, so the offset is neither validated nor dereferenced.
    FixtureSpec empty_rodata = default_spec(false, false);
    empty_rodata.segments[1].content.clear();
    empty_rodata.rodata_extents = {};
    auto wild = build_nso(empty_rodata);
    put_u32(wild, field_file_offset(1), 0xFFFFFFFF);
    const auto wild_result = read_nso_module(wild);
    ARMIGO_CHECK(wild_result.ok());
    if (wild_result.ok()) {
        const auto& rodata = wild_result.image->segment(NsoSegmentKind::rodata);
        ARMIGO_CHECK(rodata.content_size == 0 && rodata.mapped_size == 0);
    }

    auto empty_payload = valid;
    put_u32(empty_payload, field_file_size(2), 0);
    ARMIGO_CHECK(error_of(empty_payload) == NsoError::segment_size_mismatch);
}

void layout_rejections() {
    const auto valid = build_nso(default_spec(false, false));
    auto expect_layout = [&](std::size_t field, std::uint32_t value, NsoSegmentKind segment) {
        auto file = valid;
        put_u32(file, field, value);
        const auto result = read_nso_module(file);
        ARMIGO_CHECK(!result.ok() && result.error.code == NsoError::segment_layout_invalid &&
                     result.error.segment == segment);
    };
    expect_layout(field_image_offset(0), 0x1000, NsoSegmentKind::text);   // Text not at zero.
    expect_layout(field_image_offset(1), 0x2010, NsoSegmentKind::rodata); // Misaligned.
    expect_layout(field_image_offset(1), 0x1000, NsoSegmentKind::rodata); // Shares text page.
    expect_layout(field_image_offset(2), 0x2000, NsoSegmentKind::data);   // Overlaps rodata.
    expect_layout(field_image_offset(2), 0x0000, NsoSegmentKind::data);   // Out of order.

    // Text must be non-empty: an empty uncompressed text segment.
    FixtureSpec empty_text = default_spec(false, false);
    empty_text.segments[0].content.clear();
    ARMIGO_CHECK(error_of(build_nso(empty_text)) == NsoError::segment_layout_invalid);

    // Extreme offsets and sizes are computed in 64 bits and caught by the
    // image limit rather than wrapping.
    auto huge = valid;
    put_u32(huge, field_image_offset(2), 0xFFFFF000);
    put_u32(huge, kFieldBssSize, 0xFFFFFFFF);
    const auto huge_result = read_nso_module(huge);
    ARMIGO_CHECK(!huge_result.ok() && huge_result.error.code == NsoError::image_too_large &&
                 huge_result.error.detail > 0xFFFFFFFFull);

    NsoReadOptions tight;
    tight.limits.max_image_bytes = 0x5FFF;
    ARMIGO_CHECK(error_of(valid, tight) == NsoError::image_too_large);
    tight.limits.max_image_bytes = 0x6000;
    ARMIGO_CHECK(error_of(valid, tight) == NsoError::success);

    auto extent = valid;
    put_u32(extent, field_extent(1), 0x880);
    put_u32(extent, field_extent(1) + 4, 0x81);
    ARMIGO_CHECK(error_of(extent) == NsoError::rodata_extent_invalid);
    put_u32(extent, field_extent(1), 0xFFFFFFFF);
    put_u32(extent, field_extent(1) + 4, 0x2);
    ARMIGO_CHECK(error_of(extent) == NsoError::rodata_extent_invalid);
    put_u32(extent, field_extent(1), 0x900);
    put_u32(extent, field_extent(1) + 4, 0);
    ARMIGO_CHECK(error_of(extent) == NsoError::success);
}

void decompression_rejections() {
    const FixtureSpec spec = default_spec(true, false);
    auto file = build_nso(spec);
    // Declaring a larger decompressed size than the stream produces.
    auto larger = file;
    put_u32(larger, field_size(0), 0x1801);
    const auto result = read_nso_module(larger);
    ARMIGO_CHECK(!result.ok() && result.error.code == NsoError::decompression_failed &&
                 result.error.segment == NsoSegmentKind::text &&
                 result.error.lz4 == Lz4Error::incomplete_output);
    auto smaller = file;
    put_u32(smaller, field_size(1), 0x8FF);
    const auto smaller_result = read_nso_module(smaller);
    ARMIGO_CHECK(!smaller_result.ok() && smaller_result.error.lz4 == Lz4Error::output_overflow);

    // A stream whose first match reaches before the segment start.
    auto offset = file;
    const std::size_t start = get_u32(file, field_file_offset(2));
    const std::size_t stored = get_u32(file, field_file_size(2));
    std::fill_n(offset.begin() + static_cast<std::ptrdiff_t>(start), stored, std::byte{0});
    offset[start] = std::byte{0x10};     // One literal, then a match.
    offset[start + 2] = std::byte{0x05}; // Offset 5 with only one byte produced.
    const auto bad_offset = read_nso_module(offset);
    ARMIGO_CHECK(!bad_offset.ok() && bad_offset.error.lz4 == Lz4Error::invalid_match_offset &&
                 bad_offset.error.segment == NsoSegmentKind::data);
}

void module_header() {
    FixtureSpec spec = default_spec(false, false);
    // Module header placed at the start of data (image offset 0x3000) and
    // referenced from the second word of text.
    auto& text = spec.segments[0].content;
    put_u32(text, 4, 0x3000);
    auto& data = spec.segments[2].content;
    data[0] = std::byte{'M'};
    data[1] = std::byte{'O'};
    data[2] = std::byte{'D'};
    data[3] = std::byte{'0'};
    const std::int32_t fields[6] = {0x100, 0x400, 0x2745, -0x1000, -0xF00, 0x400};
    for (int i = 0; i < 6; ++i) {
        put_u32(data, 4 + 4 * i, static_cast<std::uint32_t>(fields[i]));
    }
    const auto result = read_nso_module(build_nso(spec));
    ARMIGO_CHECK(result.ok() && result.image->module_header.has_value());
    if (result.ok() && result.image->module_header) {
        const ModuleHeaderInfo& info = *result.image->module_header;
        ARMIGO_CHECK(info.header_offset == 0x3000 && info.dynamic_offset == 0x3100);
        ARMIGO_CHECK(info.bss_start_offset == 0x3400 && info.bss_end_offset == 0x5745);
        ARMIGO_CHECK(info.eh_frame_hdr_start_offset == 0x2000);
        ARMIGO_CHECK(info.eh_frame_hdr_end_offset == 0x2100);
        ARMIGO_CHECK(info.module_object_offset == 0x3400);
    }

    // A present header with a range beyond the image is an error.
    FixtureSpec beyond = spec;
    put_u32(beyond.segments[2].content, 4, 0x3000);
    ARMIGO_CHECK(error_of(build_nso(beyond)) == NsoError::module_header_invalid);
    FixtureSpec negative = spec;
    put_u32(negative.segments[2].content, 8, static_cast<std::uint32_t>(-0x3001));
    ARMIGO_CHECK(error_of(build_nso(negative)) == NsoError::module_header_invalid);
    FixtureSpec inverted = spec;
    put_u32(inverted.segments[2].content, 12, 0x3FF);
    ARMIGO_CHECK(error_of(build_nso(inverted)) == NsoError::module_header_invalid);

    // A pointer word that does not select a module header means "absent".
    for (std::uint32_t pointer : {0x3002u, 0x5FF0u, 0x6000u, 0xFFFFFFFCu, 0x2000u}) {
        FixtureSpec absent = spec;
        put_u32(absent.segments[0].content, 4, pointer);
        const auto absent_result = read_nso_module(build_nso(absent));
        ARMIGO_CHECK(absent_result.ok() && !absent_result.image->module_header.has_value());
    }
}

void mutation_sweep() {
    // Deterministic corruption of every header word and of scattered payload
    // bytes. The reader may accept or reject; it must never read or write out
    // of bounds (checked by sanitizer builds) and must report a coherent result.
    const auto valid = build_nso(default_spec(true, true));
    std::uint32_t state = 0x1234567u;
    auto next = [&] {
        state = state * 1103515245u + 12345u;
        return state;
    };
    const std::uint32_t interesting[] = {0, 1, 0xFF, 0x100, 0xFFF, 0x1000, 0x7FFFFFFF,
                                         0x80000000u, 0xFFFFFFFFu};
    for (std::size_t word = 0; word < kNsoHeaderSize; word += 4) {
        for (std::uint32_t value : interesting) {
            auto file = valid;
            put_u32(file, word, value);
            const auto result = read_nso_module(file);
            ARMIGO_CHECK(result.ok() == (result.error.code == NsoError::success));
        }
    }
    for (int round = 0; round < 2000; ++round) {
        auto file = valid;
        const int flips = 1 + static_cast<int>(next() % 4);
        for (int i = 0; i < flips; ++i) {
            const std::size_t position = next() % file.size();
            file[position] ^= static_cast<std::byte>(1u << (next() % 8));
        }
        const auto result = read_nso_module(file);
        if (result.ok()) {
            ARMIGO_CHECK(result.image->bytes.size() % kNsoGuestPageSize == 0);
        }
    }
}

} // namespace

int main() {
    reads_build_identity();
    reads_uncompressed_module();
    reads_execute_only_module();
    reads_compressed_module_with_hashes();
    mixed_flags();
    hash_policies();
    header_rejections();
    file_range_rejections();
    layout_rejections();
    decompression_rejections();
    module_header();
    mutation_sweep();
    return armigo::loader::test::finish("loader.nso_module");
}
