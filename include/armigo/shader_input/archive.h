// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Bounded archive primitives used by the offline shader input pipeline.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::shader_input {

struct ArchiveError {
    enum class Code : std::uint8_t {
        Yaz0HeaderTooSmall,
        Yaz0MagicMismatch,
        Yaz0OutputLimitExceeded,
        Yaz0AllocationFailure,
        Yaz0TruncatedStream,
        Yaz0BackReferenceBeforeOutput,
        SarcHeaderTooSmall,
        SarcMagicMismatch,
        SarcUnsupportedByteOrder,
        SarcHeaderSizeTooSmall,
        SarcHeaderOutsideInput,
        SarcDeclaredSizeBeforeHeader,
        SarcDeclaredSizeBeyondInput,
        SarcTruncatedSfat,
        SarcMissingSfat,
        SarcNodeTableOutsideInput,
        SarcTruncatedSfnt,
        SarcMissingSfnt,
        SarcDataOffsetBeforeNames,
        SarcDataOffsetOutsideInput,
        SarcNameOffsetOutsideTable,
        SarcUnterminatedName,
        SarcMemberRangeReversed,
        SarcMemberOutsideData,
    };

    Code code = Code::Yaz0HeaderTooSmall;
    // Byte offset in the compressed stream or archive at which validation
    // failed. For range errors this is the field or table entry offset.
    std::size_t offset = 0;
    // Observed value and expected limit where the error has numeric context.
    std::uint64_t detail = 0;
    std::uint64_t bound = 0;

    [[nodiscard]] const char* message() const noexcept;
};

using Yaz0Result = std::variant<std::vector<std::byte>, ArchiveError>;

// Decompresses one Yaz0 stream. The big-endian output size declared in the
// stream must not exceed max_output_size. Trailing compressed bytes are left
// uninterpreted after the declared output has been produced.
[[nodiscard]] Yaz0Result decompress_yaz0(std::span<const std::byte> input,
                                         std::size_t max_output_size);

struct SarcMember {
    // Some SARC nodes do not carry a name. In that case has_name is false and
    // name is empty. Named empty strings retain has_name == true.
    bool has_name = false;
    std::string_view name;
    std::span<const std::byte> data;
};

struct SarcArchive {
    std::span<const std::byte> source;
    std::vector<SarcMember> members;
};

using SarcResult = std::variant<SarcArchive, ArchiveError>;

// Parses a little-endian SARC and enumerates its members. The returned source
// is limited to the archive's declared file size; harmless trailing bytes in
// input are ignored. The source, names, and member data all borrow input and
// remain valid only while input is alive and unmodified. The member table
// itself is owned by SarcArchive.
[[nodiscard]] SarcResult parse_sarc(std::span<const std::byte> input);

}  // namespace armigo::shader_input
