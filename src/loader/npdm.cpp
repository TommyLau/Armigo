// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Structural NPDM (META) reader implementation.
//
// Bounded byte-range validation reports structured failures instead of
// exceptions. Admission bounds, alignment, version checks and validation order
// follow the Horizon 20.5.0 loader. Overlapping ranges are permitted when each
// range independently satisfies its parent bounds. Validation checks the image
// and header, parent bounds, ACID magic and version, ACID subsections, signed
// size, and ACI0 magic and subsections in that order.

#include "npdm.h"

#include <string>

namespace armigo::loader {
namespace {

constexpr std::uint32_t kMetaMagic = 0x4154454Du;  // "META" little-endian
constexpr std::uint32_t kAci0Magic = 0x30494341u;  // "ACI0" little-endian
constexpr std::uint32_t kAcidMagic = 0x44494341u;  // "ACID" little-endian

// Fixed offsets inside the META header (Horizon 20.5.0 metadata layout).
constexpr std::size_t kMetaOffsetFlags = 0x0C;
constexpr std::size_t kMetaOffsetMainThreadPriority = 0x0E;
constexpr std::size_t kMetaOffsetMainThreadIdealCore = 0x0F;
constexpr std::size_t kMetaOffsetSystemResourceSize = 0x14;
constexpr std::size_t kMetaOffsetVersion = 0x18;
constexpr std::size_t kMetaOffsetMainThreadStackSize = 0x1C;
constexpr std::size_t kMetaOffsetAci0Offset = 0x70;
constexpr std::size_t kMetaOffsetAci0Size = 0x74;
constexpr std::size_t kMetaOffsetAcidOffset = 0x78;
constexpr std::size_t kMetaOffsetAcidSize = 0x7C;

// Fixed offsets inside an ACI0 section.
constexpr std::size_t kAci0OffsetProgramId = 0x10;
constexpr std::size_t kAci0OffsetFsAccessHeader = 0x20;
constexpr std::size_t kAci0OffsetServiceAccessControl = 0x28;
constexpr std::size_t kAci0OffsetKernelAccessControl = 0x30;

// Fixed offsets inside an ACID section.
constexpr std::size_t kAcidOffsetMagic = 0x200;
constexpr std::size_t kAcidOffsetSignedSize = 0x204;
constexpr std::size_t kAcidOffsetVersionMinor = 0x208;
constexpr std::size_t kAcidOffsetVersionMajor = 0x209;
constexpr std::size_t kAcidOffsetFlags = 0x20C;
constexpr std::size_t kAcidOffsetProgramIdMin = 0x210;
constexpr std::size_t kAcidOffsetProgramIdMax = 0x218;
constexpr std::size_t kAcidOffsetFsAccessControl = 0x220;
constexpr std::size_t kAcidOffsetServiceAccessControl = 0x228;
constexpr std::size_t kAcidOffsetKernelAccessControl = 0x230;

// A bounds-checked little-endian reader over the whole image.
class ImageReader {
public:
    explicit ImageReader(std::span<const std::byte> image) : image_(image) {}

    [[nodiscard]] bool read_u8(std::size_t offset, std::uint8_t& value) const {
        if (offset >= image_.size()) {
            return false;
        }
        value = static_cast<std::uint8_t>(image_[offset]);
        return true;
    }

    [[nodiscard]] bool read_u32(std::size_t offset, std::uint32_t& value) const {
        if (offset > image_.size() || image_.size() - offset < sizeof(std::uint32_t)) {
            return false;
        }
        value = 0;
        for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index) {
            value |= static_cast<std::uint32_t>(image_[offset + index]) << (8 * index);
        }
        return true;
    }

    [[nodiscard]] bool read_u64(std::size_t offset, std::uint64_t& value) const {
        if (offset > image_.size() || image_.size() - offset < sizeof(std::uint64_t)) {
            return false;
        }
        value = 0;
        for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
            value |= static_cast<std::uint64_t>(image_[offset + index]) << (8 * index);
        }
        return true;
    }

private:
    std::span<const std::byte> image_;
};

// True when [offset, offset + size) lies inside [base, base + base_size).
// All arithmetic is 64-bit; the fields are 32-bit so sums cannot wrap.
[[nodiscard]] bool inside_range(std::uint64_t base, std::uint64_t base_size,
                                std::uint64_t offset, std::uint64_t size) {
    return offset >= base && size <= base_size && offset - base <= base_size - size;
}

// Reads one subsection (offset, size) pair and bounds it inside `parent`,
// which itself must already be inside the image.
[[nodiscard]] bool read_subsection(const ImageReader& reader, std::size_t offset,
                                   const NpdmSection& parent, std::uint32_t header_size,
                                   bool kernel_access, NpdmSection& section) {
    std::uint32_t relative_offset = 0;
    std::uint32_t size = 0;
    if (!reader.read_u32(offset, relative_offset) || !reader.read_u32(offset + 4, size)) {
        return false;
    }
    if (relative_offset < header_size ||
        (kernel_access && (relative_offset % 4 != 0 || size < 4))) {
        return false;
    }
    if (!inside_range(parent.offset, parent.size, std::uint64_t{parent.offset} + relative_offset,
                      size)) {
        return false;
    }
    section.offset = parent.offset + relative_offset;
    section.size = size;
    return true;
}

} // namespace

std::variant<NpdmParseFailure, ParsedNpdm> parse_npdm(std::span<const std::byte> image) {
    const ImageReader reader(image);

    if (image.size() >= 0x8000) {
        return NpdmParseFailure{"npdm.image_size", "the image reaches the loader size limit"};
    }
    if (image.size() < kNpdmMetaHeaderSize) {
        return NpdmParseFailure{"npdm.meta_header",
                                "the image is smaller than the fixed META header"};
    }

    ParsedNpdm parsed;
    parsed.image = image;

    std::uint32_t magic = 0;
    if (!reader.read_u32(0, magic) || magic != kMetaMagic) {
        return NpdmParseFailure{"npdm.meta_magic", "the image does not start with the META magic"};
    }

    if (!reader.read_u8(kMetaOffsetFlags, parsed.meta_flags) ||
        !reader.read_u8(kMetaOffsetMainThreadPriority, parsed.main_thread_priority) ||
        !reader.read_u8(kMetaOffsetMainThreadIdealCore, parsed.main_thread_ideal_core) ||
        !reader.read_u32(kMetaOffsetSystemResourceSize, parsed.system_resource_size) ||
        !reader.read_u32(kMetaOffsetVersion, parsed.meta_version) ||
        !reader.read_u32(kMetaOffsetMainThreadStackSize, parsed.main_thread_stack_size)) {
        return NpdmParseFailure{"npdm.meta_header", "the META header fields are truncated"};
    }

    if ((parsed.meta_flags & 0x80) != 0) {
        return NpdmParseFailure{"npdm.meta_flags",
                                "the META flags set bit seven, which this loader baseline rejects"};
    }

    std::uint32_t aci0_offset = 0;
    std::uint32_t aci0_size = 0;
    std::uint32_t acid_offset = 0;
    std::uint32_t acid_size = 0;
    if (!reader.read_u32(kMetaOffsetAci0Offset, aci0_offset) ||
        !reader.read_u32(kMetaOffsetAci0Size, aci0_size) ||
        !reader.read_u32(kMetaOffsetAcidOffset, acid_offset) ||
        !reader.read_u32(kMetaOffsetAcidSize, acid_size)) {
        return NpdmParseFailure{"npdm.meta_header", "the META section table is truncated"};
    }

    // Parent ranges are checked before either access-info header is read.
    if (aci0_offset < kNpdmMetaHeaderSize || aci0_offset % 8 != 0 ||
        aci0_size < kNpdmAci0HeaderSize ||
        !inside_range(0, image.size(), aci0_offset, aci0_size)) {
        return NpdmParseFailure{"npdm.aci0_section",
                                "the ACI0 section has invalid bounds or alignment"};
    }
    if (acid_offset < kNpdmMetaHeaderSize || acid_offset % 8 != 0 ||
        acid_size < kNpdmAcidSize ||
        !inside_range(0, image.size(), acid_offset, acid_size)) {
        return NpdmParseFailure{"npdm.acid_section",
                                "the ACID section has invalid bounds or alignment"};
    }
    parsed.aci0 = NpdmSection{aci0_offset, aci0_size};
    parsed.acid = NpdmSection{acid_offset, acid_size};

    // ACID validation precedes ACI0 validation in the firmware loader.
    if (!reader.read_u32(acid_offset + kAcidOffsetMagic, magic) || magic != kAcidMagic) {
        return NpdmParseFailure{"npdm.acid_magic", "the ACID section does not carry the ACID magic"};
    }
    std::uint8_t acid_major = 0;
    std::uint8_t acid_minor = 0;
    if (!reader.read_u8(acid_offset + kAcidOffsetVersionMajor, acid_major) ||
        !reader.read_u8(acid_offset + kAcidOffsetVersionMinor, acid_minor)) {
        return NpdmParseFailure{"npdm.acid_header", "the ACID header is truncated"};
    }
    if (!((acid_major == 0 && acid_minor == 0) || acid_major > 19 ||
          (acid_major == 19 && acid_minor >= 3))) {
        return NpdmParseFailure{"npdm.acid_version",
                                "the ACID version is not admitted by this baseline"};
    }
    if (!reader.read_u32(acid_offset + kAcidOffsetFlags, parsed.acid_flags) ||
        !reader.read_u64(acid_offset + kAcidOffsetProgramIdMin, parsed.acid_program_id_min) ||
        !reader.read_u64(acid_offset + kAcidOffsetProgramIdMax, parsed.acid_program_id_max)) {
        return NpdmParseFailure{"npdm.acid_header", "the ACID header is truncated"};
    }
    if (!read_subsection(reader, acid_offset + kAcidOffsetFsAccessControl, parsed.acid,
                         kNpdmAcidSize, false, parsed.acid_fs_access_control) ||
        !read_subsection(reader, acid_offset + kAcidOffsetServiceAccessControl, parsed.acid,
                         kNpdmAcidSize, false, parsed.acid_service_access_control) ||
        !read_subsection(reader, acid_offset + kAcidOffsetKernelAccessControl, parsed.acid,
                         kNpdmAcidSize, true, parsed.acid_kernel_access_control)) {
        return NpdmParseFailure{"npdm.acid_sections",
                                "an ACID subsection has invalid bounds or alignment"};
    }
    std::uint32_t signed_size = 0;
    if (!reader.read_u32(acid_offset + kAcidOffsetSignedSize, signed_size) ||
        std::uint64_t{signed_size} + 0x100 != acid_size) {
        return NpdmParseFailure{"npdm.acid_size",
                                "the ACID signed size disagrees with its section size"};
    }

    if (!reader.read_u32(aci0_offset, magic) || magic != kAci0Magic) {
        return NpdmParseFailure{"npdm.aci0_magic", "the ACI0 section does not start with ACI0"};
    }
    if (!reader.read_u64(aci0_offset + kAci0OffsetProgramId, parsed.aci0_program_id)) {
        return NpdmParseFailure{"npdm.aci0_header", "the ACI0 header is truncated"};
    }
    if (!read_subsection(reader, aci0_offset + kAci0OffsetFsAccessHeader, parsed.aci0,
                         kNpdmAci0HeaderSize, false, parsed.aci0_fs_access_header) ||
        !read_subsection(reader, aci0_offset + kAci0OffsetServiceAccessControl, parsed.aci0,
                         kNpdmAci0HeaderSize, false, parsed.aci0_service_access_control) ||
        !read_subsection(reader, aci0_offset + kAci0OffsetKernelAccessControl, parsed.aci0,
                         kNpdmAci0HeaderSize, true, parsed.aci0_kernel_access_control)) {
        return NpdmParseFailure{"npdm.aci0_sections",
                                "an ACI0 subsection has invalid bounds or alignment"};
    }
    return parsed;
}

} // namespace armigo::loader
