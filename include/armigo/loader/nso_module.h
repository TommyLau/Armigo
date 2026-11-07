// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// NSO0 executable module reader.
//
// Parses an in-memory NSO0 container, validates its header, file ranges,
// guest memory layout, flags, and resource limits, decompresses its three
// segments, verifies segment hashes, and materializes a contiguous guest
// module image with BSS zero-filled.
//
// The result is plain data. The reader never maps host memory, never makes
// anything executable on the host, never applies relocations, and never runs
// guest code. Placing the image in a guest address space and linking it are
// the responsibility of a later loader stage.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <span>
#include <vector>

#include <armigo/loader/lz4_block.h>

namespace armigo::loader {

/// Guest page granularity used for NSO segment placement and image size.
inline constexpr std::uint64_t kNsoGuestPageSize = 0x1000;

/// Size of the fixed NSO0 header that precedes segment payloads.
inline constexpr std::size_t kNsoHeaderSize = 0x100;

/// The three loadable NSO segments, in header order.
enum class NsoSegmentKind : std::uint8_t {
    text = 0,
    rodata = 1,
    data = 2,
};

inline constexpr std::size_t kNsoSegmentCount = 3;

/// English segment name ("text", "rodata", "data"). Never returns null.
const char* to_string(NsoSegmentKind kind) noexcept;

/// Guest access permissions intended for an image region.
enum class SegmentPermission : std::uint8_t {
    none = 0,
    read = 1 << 0,
    write = 1 << 1,
    execute = 1 << 2,
};

constexpr SegmentPermission operator|(SegmentPermission lhs, SegmentPermission rhs) noexcept {
    return static_cast<SegmentPermission>(static_cast<std::uint8_t>(lhs) |
                                          static_cast<std::uint8_t>(rhs));
}

/// Whether a segment's stored hash was checked against its decompressed bytes.
enum class HashCheck : std::uint8_t {
    /// The header did not request a check and the policy did not force one.
    not_checked,
    /// The SHA-256 of the decompressed segment matched the stored hash.
    verified,
};

/// Placement and provenance of one loadable segment inside the image.
struct NsoSegment {
    NsoSegmentKind kind = NsoSegmentKind::text;
    /// Image offset of the segment content; a multiple of the guest page size.
    std::uint64_t image_offset = 0;
    /// Bytes of segment content produced from the file.
    std::uint64_t content_size = 0;
    /// Page-rounded extent to map with `permission`. For the data segment it
    /// also covers BSS and extends to the end of the image.
    std::uint64_t mapped_size = 0;
    SegmentPermission permission = SegmentPermission::none;
    /// Whether the file stored this segment LZ4-compressed.
    bool compressed = false;
    /// Bytes this segment occupies in the file.
    std::uint64_t file_size = 0;
    HashCheck hash = HashCheck::not_checked;
};

/// A region described relative to the start of the rodata segment.
struct RodataExtent {
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
};

/// Module header ("MOD0") located through the image's second 32-bit word.
///
/// Offsets are absolute image offsets, resolved from the signed relative
/// fields stored in the header. Every range lies within the image.
struct ModuleHeaderInfo {
    std::uint64_t header_offset = 0;
    std::uint64_t dynamic_offset = 0;
    std::uint64_t bss_start_offset = 0;
    std::uint64_t bss_end_offset = 0;
    std::uint64_t eh_frame_hdr_start_offset = 0;
    std::uint64_t eh_frame_hdr_end_offset = 0;
    std::uint64_t module_object_offset = 0;
};

/// A fully materialized, unrelocated module image.
struct NsoModuleImage {
    bool zbic_compression = false;
    /// Exact 32-byte module build ID from the NSO header.
    std::array<std::byte, 32> build_id{};
    /// Image bytes. The size is a multiple of the guest page size; bytes not
    /// covered by segment content, including BSS, are zero.
    std::vector<std::byte> bytes;
    std::array<NsoSegment, kNsoSegmentCount> segments{};
    /// BSS begins immediately after the data segment content.
    std::uint64_t bss_offset = 0;
    std::uint64_t bss_size = 0;
    /// Rodata-relative extents recorded in the header.
    RodataExtent api_info{};
    RodataExtent dynstr{};
    RodataExtent dynsym{};
    /// Present when the image carries a module header; absent otherwise.
    std::optional<ModuleHeaderInfo> module_header;

    const NsoSegment& segment(NsoSegmentKind kind) const noexcept {
        return segments[static_cast<std::size_t>(kind)];
    }
};

/// Hash verification policy.
enum class NsoHashPolicy : std::uint8_t {
    /// Verify exactly the segments whose header flag requests verification.
    verify_flagged,
    /// Verify every segment, regardless of the header flags.
    verify_all,
};

/// Resource bounds applied before any large allocation or decompression.
struct NsoReadLimits {
    std::uint64_t max_input_bytes = 512ull * 1024 * 1024;
    std::uint64_t max_image_bytes = 1024ull * 1024 * 1024;
};

struct NsoReadOptions {
    NsoHashPolicy hash_policy = NsoHashPolicy::verify_flagged;
    NsoReadLimits limits{};
};

/// Structured failure reported by `read_nso_module`.
enum class NsoError : std::uint8_t {
    success = 0,
    /// The requested hash policy is not a supported enumerator.
    invalid_hash_policy,
    /// The input exceeds `max_input_bytes`.
    input_too_large,
    /// The input is shorter than the fixed header.
    truncated_header,
    /// The magic is not "NSO0".
    bad_magic,
    /// The header version is not the supported value (0).
    unsupported_version,
    /// Flag bits outside the known compression and hash bits are set.
    /// `detail` holds the unknown bits.
    unsupported_flags,
    /// A segment's file range starts inside the header or extends past the
    /// end of the input. `detail` holds the file offset.
    segment_file_range_invalid,
    /// Two segments' file ranges overlap. `detail` holds the later offset.
    segment_file_overlap,
    /// An uncompressed segment's file size differs from its memory size, or a
    /// compressed segment has an empty payload. `detail` holds the file size.
    segment_size_mismatch,
    /// Segment memory placement violates alignment, ordering, or overlap
    /// rules. `detail` holds the offending image offset.
    segment_layout_invalid,
    /// The image would exceed `max_image_bytes`. `detail` holds the size.
    image_too_large,
    /// LZ4 decoding failed; `lz4` holds the decoder error and `detail` the
    /// position within the compressed segment.
    decompression_failed,
    /// ZBIC decoding failed; decoder_detail describes the format or build limitation.
    zbic_decompression_failed,
    /// A decompressed segment does not match its stored SHA-256 hash.
    hash_mismatch,
    /// A rodata-relative extent lies outside the rodata segment. `detail`
    /// holds the extent offset.
    rodata_extent_invalid,
    /// A module header is present but describes ranges outside the image.
    /// `detail` holds the module header offset.
    module_header_invalid,
    /// Host memory for the image could not be allocated.
    allocation_failed,
    /// The host hash implementation reported a failure.
    hash_unavailable,
};

/// English diagnostic text for an `NsoError` value. Never returns null.
const char* to_string(NsoError error) noexcept;

struct NsoReadError {
    NsoError code = NsoError::success;
    /// The segment involved, when the failure concerns one segment.
    std::optional<NsoSegmentKind> segment;
    Lz4Error lz4 = Lz4Error::success;
    std::uint64_t detail = 0;
    std::string decoder_detail;
};

/// Outcome of `read_nso_module`: an image on success, an error otherwise.
struct NsoReadResult {
    std::optional<NsoModuleImage> image;
    NsoReadError error{};

    bool ok() const noexcept { return image.has_value(); }
};

/// Reads and materializes an NSO0 module from `file`.
///
/// The input is only read; no reference to it is retained. All validation
/// that bounds allocation happens before the image is allocated.
NsoReadResult read_nso_module(std::span<const std::byte> file,
                              const NsoReadOptions& options = {});

} // namespace armigo::loader
