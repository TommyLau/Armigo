// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/nso_module.h>

#include "sha256.h"
#include "zbic.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace armigo::loader {

namespace {

// Fixed header layout. All integers are little-endian.
constexpr std::size_t kMagicOffset = 0x00;
constexpr std::size_t kVersionOffset = 0x04;
constexpr std::size_t kFlagsOffset = 0x0C;
constexpr std::size_t kSegmentHeaderOffset = 0x10;
constexpr std::size_t kSegmentHeaderStride = 0x10;
constexpr std::size_t kBssSizeOffset = 0x3C;
constexpr std::size_t kFileSizeOffset = 0x60;
constexpr std::size_t kApiInfoOffset = 0x88;
constexpr std::size_t kDynstrOffset = 0x90;
constexpr std::size_t kDynsymOffset = 0x98;
constexpr std::size_t kHashOffset = 0xA0;
constexpr std::size_t kHashSize = 0x20;

constexpr std::uint32_t kSupportedVersion = 0;
// Bits 0-2 mark compressed segments; bits 3-5 request hash verification.
constexpr std::uint32_t kCompressedFlagShift = 0;
constexpr std::uint32_t kHashFlagShift = 3;
constexpr std::uint32_t kExecuteOnly = 1u << 6;
constexpr std::uint32_t kZbicCompression = 1u << 7;
constexpr std::uint32_t kKnownFlags = 0xFF;

constexpr std::array<std::byte, 4> kNsoMagic{std::byte{'N'}, std::byte{'S'}, std::byte{'O'},
                                             std::byte{'0'}};
constexpr std::array<std::byte, 4> kModuleMagic{std::byte{'M'}, std::byte{'O'}, std::byte{'D'},
                                                std::byte{'0'}};
// Module header: magic followed by six signed 32-bit offsets relative to the
// module header start.
constexpr std::uint64_t kModuleHeaderSize = 0x1C;
// Location of the 32-bit image offset of the module header.
constexpr std::uint64_t kModuleHeaderPointerOffset = 4;

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::int32_t read_i32(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    return static_cast<std::int32_t>(read_u32(bytes, offset));
}

constexpr std::uint64_t align_up(std::uint64_t value) noexcept {
    return (value + (kNsoGuestPageSize - 1)) & ~(kNsoGuestPageSize - 1);
}

constexpr bool is_aligned(std::uint64_t value) noexcept {
    return (value & (kNsoGuestPageSize - 1)) == 0;
}

struct SegmentHeader {
    std::uint64_t file_offset = 0;
    std::uint64_t image_offset = 0;
    std::uint64_t size = 0;
    std::uint64_t file_size = 0;
    bool compressed = false;
    bool hash_flagged = false;
};

NsoReadResult failure(NsoError code, std::uint64_t detail = 0,
                      std::optional<NsoSegmentKind> segment = std::nullopt,
                      Lz4Error lz4 = Lz4Error::success) {
    NsoReadResult result;
    result.error = NsoReadError{code, segment, lz4, detail, {}};
    return result;
}

NsoSegmentKind kind_of(std::size_t index) noexcept {
    return static_cast<NsoSegmentKind>(index);
}

// Resolves a signed module-header-relative field to an image offset, or
// returns nothing if it falls outside [0, image_size].
std::optional<std::uint64_t> resolve_relative(std::uint64_t base, std::int32_t relative,
                                              std::uint64_t image_size) noexcept {
    const std::int64_t target = static_cast<std::int64_t>(base) + relative;
    if (target < 0 || static_cast<std::uint64_t>(target) > image_size) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(target);
}

// Locates and validates the module header. A header is considered present
// only when the pointer word selects an in-bounds, 4-byte-aligned location
// that carries the module magic; any other content means "absent". A present
// header whose ranges fall outside the image is an error.
NsoError parse_module_header(const NsoModuleImage& image, std::optional<ModuleHeaderInfo>& out,
                             std::uint64_t& detail) {
    out.reset();
    const std::span<const std::byte> bytes = image.bytes;
    const std::uint64_t size = bytes.size();
    if (image.segment(NsoSegmentKind::text).content_size < kModuleHeaderPointerOffset + 4) {
        return NsoError::success;
    }
    const std::uint64_t header = read_u32(bytes, kModuleHeaderPointerOffset);
    if ((header & 3) != 0 || header > size || size - header < kModuleHeaderSize ||
        std::memcmp(bytes.data() + header, kModuleMagic.data(), kModuleMagic.size()) != 0) {
        return NsoError::success;
    }

    std::array<std::uint64_t, 6> fields{};
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto resolved = resolve_relative(header, read_i32(bytes, header + 4 + 4 * i), size);
        if (!resolved) {
            detail = header;
            return NsoError::module_header_invalid;
        }
        fields[i] = *resolved;
    }
    ModuleHeaderInfo info;
    info.header_offset = header;
    info.dynamic_offset = fields[0];
    info.bss_start_offset = fields[1];
    info.bss_end_offset = fields[2];
    info.eh_frame_hdr_start_offset = fields[3];
    info.eh_frame_hdr_end_offset = fields[4];
    info.module_object_offset = fields[5];
    if (info.dynamic_offset >= size || info.module_object_offset >= size ||
        info.bss_start_offset > info.bss_end_offset ||
        info.eh_frame_hdr_start_offset > info.eh_frame_hdr_end_offset) {
        detail = header;
        return NsoError::module_header_invalid;
    }
    out = info;
    return NsoError::success;
}

} // namespace

const char* to_string(NsoSegmentKind kind) noexcept {
    switch (kind) {
    case NsoSegmentKind::text:
        return "text";
    case NsoSegmentKind::rodata:
        return "rodata";
    case NsoSegmentKind::data:
        return "data";
    }
    return "unknown";
}

const char* to_string(NsoError error) noexcept {
    switch (error) {
    case NsoError::success:
        return "success";
    case NsoError::invalid_hash_policy:
        return "NSO hash policy is not supported";
    case NsoError::input_too_large:
        return "NSO input exceeds the configured size limit";
    case NsoError::truncated_header:
        return "NSO input is shorter than its fixed header";
    case NsoError::bad_magic:
        return "NSO magic is not NSO0";
    case NsoError::unsupported_version:
        return "NSO header version is not supported";
    case NsoError::unsupported_flags:
        return "NSO header sets unsupported flag bits";
    case NsoError::segment_file_range_invalid:
        return "NSO segment file range overlaps the header or exceeds the input";
    case NsoError::segment_file_overlap:
        return "NSO segment file ranges overlap";
    case NsoError::segment_size_mismatch:
        return "NSO segment file size is inconsistent with its storage form";
    case NsoError::segment_layout_invalid:
        return "NSO segment memory layout is misaligned, misordered, or overlapping";
    case NsoError::image_too_large:
        return "NSO module image exceeds the configured size limit";
    case NsoError::decompression_failed:
        return "NSO segment LZ4 decompression failed";
    case NsoError::zbic_decompression_failed:
        return "NSO segment ZBIC decompression failed";
    case NsoError::hash_mismatch:
        return "NSO segment SHA-256 does not match the stored hash";
    case NsoError::rodata_extent_invalid:
        return "NSO rodata-relative extent lies outside the rodata segment";
    case NsoError::module_header_invalid:
        return "NSO module header describes ranges outside the image";
    case NsoError::allocation_failed:
        return "NSO module image allocation failed";
    case NsoError::hash_unavailable:
        return "host SHA-256 implementation failed";
    }
    return "unknown NSO error";
}

NsoReadResult read_nso_module(std::span<const std::byte> file, const NsoReadOptions& options) {
    if (options.hash_policy != NsoHashPolicy::verify_flagged &&
        options.hash_policy != NsoHashPolicy::verify_all) {
        return failure(NsoError::invalid_hash_policy);
    }
    if (file.size() > options.limits.max_input_bytes) {
        return failure(NsoError::input_too_large, file.size());
    }
    if (file.size() < kNsoHeaderSize) {
        return failure(NsoError::truncated_header, file.size());
    }
    if (std::memcmp(file.data() + kMagicOffset, kNsoMagic.data(), kNsoMagic.size()) != 0) {
        return failure(NsoError::bad_magic);
    }
    const std::uint32_t version = read_u32(file, kVersionOffset);
    if (version != kSupportedVersion) {
        return failure(NsoError::unsupported_version, version);
    }
    const std::uint32_t flags = read_u32(file, kFlagsOffset);
    if ((flags & ~kKnownFlags) != 0) {
        return failure(NsoError::unsupported_flags, flags & ~kKnownFlags);
    }

    std::array<SegmentHeader, kNsoSegmentCount> headers{};
    for (std::size_t i = 0; i < kNsoSegmentCount; ++i) {
        const std::size_t base = kSegmentHeaderOffset + i * kSegmentHeaderStride;
        SegmentHeader& h = headers[i];
        h.file_offset = read_u32(file, base);
        h.image_offset = read_u32(file, base + 4);
        h.size = read_u32(file, base + 8);
        h.file_size = read_u32(file, kFileSizeOffset + 4 * i);
        h.compressed = ((flags >> (kCompressedFlagShift + i)) & 1) != 0;
        h.hash_flagged = ((flags >> (kHashFlagShift + i)) & 1) != 0;

        if (h.compressed ? h.file_size == 0 : h.file_size != h.size) {
            return failure(NsoError::segment_size_mismatch, h.file_size, kind_of(i));
        }
        if (h.file_size != 0 &&
            (h.file_offset < kNsoHeaderSize || h.file_offset > file.size() ||
             file.size() - h.file_offset < h.file_size)) {
            return failure(NsoError::segment_file_range_invalid, h.file_offset, kind_of(i));
        }
    }
    for (std::size_t i = 0; i < kNsoSegmentCount; ++i) {
        for (std::size_t j = i + 1; j < kNsoSegmentCount; ++j) {
            const SegmentHeader& a = headers[i];
            const SegmentHeader& b = headers[j];
            if (a.file_size == 0 || b.file_size == 0) {
                continue;
            }
            if (a.file_offset < b.file_offset + b.file_size &&
                b.file_offset < a.file_offset + a.file_size) {
                return failure(NsoError::segment_file_overlap,
                               a.file_offset > b.file_offset ? a.file_offset : b.file_offset,
                               kind_of(j));
            }
        }
    }

    // Guest layout: text at image offset zero, page-aligned segment starts,
    // strictly ascending order, and no two segments sharing a guest page.
    const SegmentHeader& text = headers[0];
    const SegmentHeader& rodata = headers[1];
    const SegmentHeader& data = headers[2];
    if (text.image_offset != 0 || text.size == 0) {
        return failure(NsoError::segment_layout_invalid, text.image_offset, NsoSegmentKind::text);
    }
    for (std::size_t i = 1; i < kNsoSegmentCount; ++i) {
        const SegmentHeader& previous = headers[i - 1];
        const SegmentHeader& current = headers[i];
        if (!is_aligned(current.image_offset) ||
            current.image_offset < align_up(previous.image_offset + previous.size)) {
            return failure(NsoError::segment_layout_invalid, current.image_offset, kind_of(i));
        }
    }
    const std::uint64_t bss_size = read_u32(file, kBssSizeOffset);
    const std::uint64_t bss_offset = data.image_offset + data.size;
    const std::uint64_t image_size = align_up(bss_offset + bss_size);
    if (image_size > options.limits.max_image_bytes) {
        return failure(NsoError::image_too_large, image_size);
    }

    NsoModuleImage image;
    image.zbic_compression = (flags & kZbicCompression) != 0;
    std::copy_n(file.begin() + 0x40, image.build_id.size(), image.build_id.begin());
    image.bss_offset = bss_offset;
    image.bss_size = bss_size;
    const std::array<std::pair<std::size_t, RodataExtent*>, 3> extents{{
        {kApiInfoOffset, &image.api_info},
        {kDynstrOffset, &image.dynstr},
        {kDynsymOffset, &image.dynsym},
    }};
    for (const auto& [offset, extent] : extents) {
        extent->offset = read_u32(file, offset);
        extent->size = read_u32(file, offset + 4);
        if (extent->offset > rodata.size || rodata.size - extent->offset < extent->size) {
            return failure(NsoError::rodata_extent_invalid, extent->offset,
                           NsoSegmentKind::rodata);
        }
    }

    try {
        // Value-initialized: gaps, padding, and BSS are zero.
        image.bytes.resize(image_size);
    } catch (const std::bad_alloc&) {
        return failure(NsoError::allocation_failed, image_size);
    } catch (const std::length_error&) {
        return failure(NsoError::allocation_failed, image_size);
    }

    for (std::size_t i = 0; i < kNsoSegmentCount; ++i) {
        const SegmentHeader& h = headers[i];
        const std::span<std::byte> destination =
            std::span<std::byte>(image.bytes).subspan(h.image_offset, h.size);
        // Only non-empty file ranges were validated; an empty uncompressed
        // segment may carry any file offset and must not form a span from it.
        const std::span<const std::byte> source =
            h.file_size != 0 ? file.subspan(h.file_offset, h.file_size)
                             : std::span<const std::byte>{};
        if (h.compressed && (flags & kZbicCompression)) {
            if (const auto failed = detail::decompress_zbic(source, destination)) {
                auto result = failure(NsoError::zbic_decompression_failed, 0, kind_of(i));
                result.error.decoder_detail = *failed;
                return result;
            }
        } else if (h.compressed) {
            const Lz4Result decoded = decompress_lz4_block(source, destination);
            if (!decoded.ok()) {
                return failure(NsoError::decompression_failed, decoded.input_offset, kind_of(i),
                               decoded.error);
            }
        } else if (h.size != 0) {
            std::memcpy(destination.data(), source.data(), h.size);
        }

        HashCheck check = HashCheck::not_checked;
        if (h.hash_flagged || options.hash_policy == NsoHashPolicy::verify_all) {
            const auto digest = detail::sha256(destination);
            if (!digest) {
                return failure(NsoError::hash_unavailable, 0, kind_of(i));
            }
            if (std::memcmp(digest->data(), file.data() + kHashOffset + kHashSize * i,
                            kHashSize) != 0) {
                return failure(NsoError::hash_mismatch, 0, kind_of(i));
            }
            check = HashCheck::verified;
        }

        NsoSegment& segment = image.segments[i];
        segment.kind = kind_of(i);
        segment.image_offset = h.image_offset;
        segment.content_size = h.size;
        segment.compressed = h.compressed;
        segment.file_size = h.file_size;
        segment.hash = check;
    }

    NsoSegment& text_segment = image.segments[0];
    text_segment.mapped_size = align_up(text.size);
    text_segment.permission = (flags & kExecuteOnly) ? SegmentPermission::execute
        : SegmentPermission::read | SegmentPermission::execute;
    NsoSegment& rodata_segment = image.segments[1];
    rodata_segment.mapped_size = align_up(rodata.image_offset + rodata.size) - rodata.image_offset;
    rodata_segment.permission = SegmentPermission::read;
    NsoSegment& data_segment = image.segments[2];
    data_segment.mapped_size = image_size - data.image_offset;
    data_segment.permission = SegmentPermission::read | SegmentPermission::write;

    std::uint64_t module_detail = 0;
    const NsoError module_error = parse_module_header(image, image.module_header, module_detail);
    if (module_error != NsoError::success) {
        return failure(module_error, module_detail);
    }

    NsoReadResult result;
    result.image = std::move(image);
    return result;
}

} // namespace armigo::loader
