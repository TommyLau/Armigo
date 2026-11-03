// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Structural NPDM parsing: a synthetic image must yield its header fields and
// section ranges, while bad magics, truncated headers, out-of-bounds sections,
// and invalid alignment must each fail with a structured, stage-named
// result. All fixtures are independently authored bytes; none is a real
// title's metadata.

#include "npdm.h"

#include "npdm_fixture.h"
#include "test_support.h"

#include <algorithm>
#include <cstddef>
#include <variant>

namespace {

using namespace npdm_fixture;

using armigo::loader::NpdmParseFailure;
using armigo::loader::ParsedNpdm;

void expect_failure(const Bytes& image, const char* operation) {
    const auto parsed = armigo::loader::parse_npdm(image);
    ARMIGO_CHECK(std::holds_alternative<NpdmParseFailure>(parsed));
    if (std::holds_alternative<NpdmParseFailure>(parsed)) {
        const NpdmParseFailure& failure = std::get<NpdmParseFailure>(parsed);
        ARMIGO_CHECK(failure.operation == operation);
        ARMIGO_CHECK(!failure.detail.empty());
    }
}

void test_valid_image() {
    const Parts parts{};
    const NpdmImage image = build_npdm(parts);
    const auto parsed = armigo::loader::parse_npdm(image.bytes);
    ARMIGO_CHECK(std::holds_alternative<ParsedNpdm>(parsed));
    if (!std::holds_alternative<ParsedNpdm>(parsed)) {
        return;
    }
    const ParsedNpdm& meta = std::get<ParsedNpdm>(parsed);
    ARMIGO_CHECK(meta.meta_flags == 0x07);
    ARMIGO_CHECK(meta.main_thread_priority == 44);
    ARMIGO_CHECK(meta.main_thread_ideal_core == 1);
    ARMIGO_CHECK(meta.system_resource_size == 0x200000);
    ARMIGO_CHECK(meta.meta_version == 0);
    ARMIGO_CHECK(meta.main_thread_stack_size == 0x40000);
    ARMIGO_CHECK(meta.aci0_program_id == kProgramId);
    ARMIGO_CHECK(meta.acid_flags == 0x01);
    ARMIGO_CHECK(meta.acid_program_id_min == kProgramIdMin);
    ARMIGO_CHECK(meta.acid_program_id_max == kProgramIdMax);

    // The ACI0 starts right after the META header; its subsections follow the
    // 0x40-byte header in file-system, service, kernel order.
    ARMIGO_CHECK(meta.aci0.offset == 0x80);
    ARMIGO_CHECK(meta.acid.offset == ((meta.aci0.offset + meta.aci0.size + 7) & ~7u));
    ARMIGO_CHECK(meta.aci0_fs_access_header.offset == meta.aci0.offset + 0x40);
    ARMIGO_CHECK(meta.aci0_fs_access_header.size == 0x1C);
    ARMIGO_CHECK(meta.aci0_service_access_control.offset ==
                 meta.aci0_fs_access_header.offset + meta.aci0_fs_access_header.size);
    ARMIGO_CHECK(meta.aci0_kernel_access_control.size ==
                 parts.aci0_capabilities.size() * 4);
    ARMIGO_CHECK(meta.acid.size >= 0x240);
    ARMIGO_CHECK(meta.acid_fs_access_control.offset == meta.acid.offset + 0x240);
    ARMIGO_CHECK(meta.acid_kernel_access_control.size ==
                 parts.acid_capabilities.size() * 4);
}

void test_bad_meta_magic() {
    NpdmImage image = profile_image_a();
    put_u8(image.bytes, 0x00, 'X');
    expect_failure(image.bytes, "npdm.meta_magic");
}

void test_truncated_image() {
    NpdmImage image = profile_image_a();
    image.bytes.resize(0x7F);
    expect_failure(image.bytes, "npdm.meta_header");
}

void test_aci0_magic() {
    NpdmImage image = profile_image_a();
    put_u8(image.bytes, 0x80, 'Z'); // first ACI0 magic byte
    expect_failure(image.bytes, "npdm.aci0_magic");
}

void test_acid_magic() {
    NpdmImage image = profile_image_a();
    // The ACID magic sits at +0x200 of the ACID section.
    const auto parsed = armigo::loader::parse_npdm(image.bytes);
    ARMIGO_CHECK(std::holds_alternative<ParsedNpdm>(parsed));
    if (!std::holds_alternative<ParsedNpdm>(parsed)) {
        return;
    }
    const std::uint32_t acid_offset = std::get<ParsedNpdm>(parsed).acid.offset;
    put_u8(image.bytes, acid_offset + 0x200, 'Q');
    expect_failure(image.bytes, "npdm.acid_magic");
}

void test_aci0_out_of_bounds() {
    NpdmImage image = profile_image_a();
    put_u32(image.bytes, kMetaAci0Offset, 0xFFFFFF00u);
    expect_failure(image.bytes, "npdm.aci0_section");
}

void test_acid_too_short() {
    NpdmImage image = profile_image_a();
    put_u32(image.bytes, kMetaAcidSize, 0x23F);
    expect_failure(image.bytes, "npdm.acid_section");
}

void expect_success(const Bytes& image) {
    ARMIGO_CHECK(std::holds_alternative<ParsedNpdm>(armigo::loader::parse_npdm(image)));
}

std::uint32_t field(const Bytes& image, std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        value |= std::uint32_t{static_cast<std::uint8_t>(image[offset + index])} << (8 * index);
    }
    return value;
}

void test_sections_overlap() {
    NpdmImage image = profile_image_a();
    const auto aci0 = field(image.bytes, kMetaAci0Offset);
    const auto acid = field(image.bytes, kMetaAcidOffset);
    const auto aci0_size = field(image.bytes, kMetaAci0Size);
    // The unsigned header and data fit inside the ACID signature area. Each
    // parent range remains independently valid; authentication is not tested.
    std::copy_n(image.bytes.begin() + aci0, aci0_size, image.bytes.begin() + acid);
    put_u32(image.bytes, kMetaAci0Offset, acid);
    expect_success(image.bytes);
}

void test_subsection_out_of_parent() {
    NpdmImage image = profile_image_a();
    // Inflate the declared kernel access control size past the ACI0 section.
    const std::size_t kac_size_field = 0x80 + 0x34;
    put_u32(image.bytes, kac_size_field, 0x1000);
    expect_failure(image.bytes, "npdm.aci0_sections");
}

void test_subsection_overlap() {
    for (bool acid : {false, true}) {
        NpdmImage image = profile_image_a();
        const auto parent = field(image.bytes, acid ? kMetaAcidOffset : kMetaAci0Offset);
        const auto fs_pair = parent + (acid ? 0x220 : 0x20);
        const auto sac_pair = fs_pair + 8;
        put_u32(image.bytes, sac_pair, field(image.bytes, fs_pair));
        expect_success(image.bytes);
        // Empty ranges can also lie inside another data range.
        put_u32(image.bytes, sac_pair, field(image.bytes, fs_pair) + 1);
        put_u32(image.bytes, sac_pair + 4, 0);
        expect_success(image.bytes);
    }
}

void test_image_size_and_flags() {
    NpdmImage image = profile_image_a();
    image.bytes.resize(0x7FFF);
    expect_success(image.bytes);
    image.bytes.resize(0x8000);
    expect_failure(image.bytes, "npdm.image_size");
    image = profile_image_a();
    put_u8(image.bytes, kMetaFlags, 0x87);
    expect_failure(image.bytes, "npdm.meta_flags");
    image.bytes.resize(0x80);
    expect_failure(image.bytes, "npdm.meta_flags");
}

void test_parent_bounds_and_alignment() {
    for (bool acid : {false, true}) {
        const auto offset_field = acid ? kMetaAcidOffset : kMetaAci0Offset;
        const auto size_field = acid ? kMetaAcidSize : kMetaAci0Size;
        const auto operation = acid ? "npdm.acid_section" : "npdm.aci0_section";
        for (std::uint32_t offset : {0x78u, 0x81u, 0xFFFFFFF8u}) {
            NpdmImage image = profile_image_a();
            put_u32(image.bytes, offset_field, offset);
            expect_failure(image.bytes, operation);
        }
        NpdmImage image = profile_image_a();
        put_u32(image.bytes, size_field, acid ? 0x23F : 0x3F);
        expect_failure(image.bytes, operation);
        image = profile_image_a();
        put_u32(image.bytes, size_field, 0xFFFFFFFFu);
        expect_failure(image.bytes, operation);
        image = profile_image_a();
        put_u32(image.bytes, size_field,
                static_cast<std::uint32_t>(image.bytes.size()) -
                    field(image.bytes, offset_field) + 1);
        expect_failure(image.bytes, operation);
    }
}

void test_subsection_boundaries() {
    for (bool acid : {false, true}) {
        const auto operation = acid ? "npdm.acid_sections" : "npdm.aci0_sections";
        const std::uint32_t header_size = acid ? 0x240 : 0x40;
        for (std::uint32_t pair_delta : {0u, 8u, 16u}) {
            NpdmImage image = profile_image_a();
            const auto parent = field(image.bytes, acid ? kMetaAcidOffset : kMetaAci0Offset);
            const auto parent_size = field(image.bytes, acid ? kMetaAcidSize : kMetaAci0Size);
            const auto pair = parent + (acid ? 0x220 : 0x20) + pair_delta;
            put_u32(image.bytes, pair, header_size - 1);
            expect_failure(image.bytes, operation);
            put_u32(image.bytes, pair, header_size);
            put_u32(image.bytes, pair + 4, 4);
            expect_success(image.bytes);
            put_u32(image.bytes, pair, 0xFFFFFFFCu);
            expect_failure(image.bytes, operation);
            put_u32(image.bytes, pair, header_size);
            put_u32(image.bytes, pair + 4, parent_size - header_size + 1);
            expect_failure(image.bytes, operation);
            if (pair_delta != 16) {
                put_u32(image.bytes, pair, parent_size);
                put_u32(image.bytes, pair + 4, 0);
                expect_success(image.bytes);
                put_u32(image.bytes, pair, parent_size + 1);
                expect_failure(image.bytes, operation);
            } else {
                put_u32(image.bytes, pair, header_size + 1);
                put_u32(image.bytes, pair + 4, 4);
                expect_failure(image.bytes, operation);
                put_u32(image.bytes, pair, header_size);
                for (std::uint32_t size : {0u, 3u}) {
                    put_u32(image.bytes, pair + 4, size);
                    expect_failure(image.bytes, operation);
                }
                // Structural validation permits a non-word-multiple byte size;
                // capability interpretation belongs to the runtime stage.
                put_u32(image.bytes, pair + 4, 5);
                expect_success(image.bytes);
            }
        }
    }
}

void test_acid_version_boundaries() {
    struct VersionCase { std::uint8_t major; std::uint8_t minor; bool accepted; };
    for (const VersionCase version : {
             VersionCase{0, 0, true}, {0, 1, false}, {18, 255, false},
             {19, 2, false}, {19, 3, true}, {19, 255, true}, {20, 0, true}}) {
        NpdmImage image = profile_image_a();
        const auto acid = field(image.bytes, kMetaAcidOffset);
        put_u8(image.bytes, acid + 0x208, version.minor);
        put_u8(image.bytes, acid + 0x209, version.major);
        if (version.accepted) {
            expect_success(image.bytes);
        } else {
            expect_failure(image.bytes, "npdm.acid_version");
        }
    }
}

void test_acid_signed_size_and_validation_order() {
    for (std::uint32_t signed_size : {0u, 0xFFFFFFFFu}) {
        NpdmImage image = profile_image_a();
        const auto acid = field(image.bytes, kMetaAcidOffset);
        put_u32(image.bytes, acid + 0x204, signed_size);
        expect_failure(image.bytes, "npdm.acid_size");
        put_u8(image.bytes, 0x80, 'Z');
        expect_failure(image.bytes, "npdm.acid_size");
        put_u32(image.bytes, acid + 0x230, 0);
        expect_failure(image.bytes, "npdm.acid_sections");
        put_u8(image.bytes, acid + 0x209, 1);
        expect_failure(image.bytes, "npdm.acid_version");
        put_u8(image.bytes, acid + 0x200, 'Z');
        expect_failure(image.bytes, "npdm.acid_magic");
    }
}

} // namespace

int main() {
    test_valid_image();
    test_bad_meta_magic();
    test_truncated_image();
    test_aci0_magic();
    test_acid_magic();
    test_aci0_out_of_bounds();
    test_acid_too_short();
    test_sections_overlap();
    test_subsection_out_of_parent();
    test_subsection_overlap();
    test_image_size_and_flags();
    test_parent_bounds_and_alignment();
    test_subsection_boundaries();
    test_acid_version_boundaries();
    test_acid_signed_size_and_validation_order();
    return armigo::loader::test::finish("loader.npdm");
}
