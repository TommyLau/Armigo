// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_input/archive.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>

namespace armigo::shader_input {
namespace {

constexpr std::size_t yaz0_header_size = 16;
constexpr std::size_t sarc_header_size = 0x14;
constexpr std::size_t sarc_declared_size_offset = 0x08;
constexpr std::size_t sarc_data_offset_field = 0x0c;
constexpr std::size_t sfat_header_size = 12;
constexpr std::size_t sfat_node_size = 16;
constexpr std::size_t sfnt_header_size = 8;

constexpr std::uint8_t byte_value(std::byte value) {
    return std::to_integer<std::uint8_t>(value);
}

std::uint16_t read_le16(std::span<const std::byte> input, std::size_t offset) {
    return static_cast<std::uint16_t>(byte_value(input[offset])) |
           static_cast<std::uint16_t>(byte_value(input[offset + 1])) << 8;
}

std::uint32_t read_le32(std::span<const std::byte> input, std::size_t offset) {
    return static_cast<std::uint32_t>(byte_value(input[offset])) |
           static_cast<std::uint32_t>(byte_value(input[offset + 1])) << 8 |
           static_cast<std::uint32_t>(byte_value(input[offset + 2])) << 16 |
           static_cast<std::uint32_t>(byte_value(input[offset + 3])) << 24;
}

std::uint32_t read_be32(std::span<const std::byte> input, std::size_t offset) {
    return static_cast<std::uint32_t>(byte_value(input[offset])) << 24 |
           static_cast<std::uint32_t>(byte_value(input[offset + 1])) << 16 |
           static_cast<std::uint32_t>(byte_value(input[offset + 2])) << 8 |
           static_cast<std::uint32_t>(byte_value(input[offset + 3]));
}

bool has_magic(std::span<const std::byte> input, std::size_t offset,
               const char (&magic)[5]) {
    return offset <= input.size() && input.size() - offset >= 4 &&
           byte_value(input[offset]) == static_cast<std::uint8_t>(magic[0]) &&
           byte_value(input[offset + 1]) ==
               static_cast<std::uint8_t>(magic[1]) &&
           byte_value(input[offset + 2]) ==
               static_cast<std::uint8_t>(magic[2]) &&
           byte_value(input[offset + 3]) == static_cast<std::uint8_t>(magic[3]);
}

ArchiveError error(ArchiveError::Code code, std::size_t offset = 0,
                   std::uint64_t detail = 0, std::uint64_t bound = 0) {
    return {code, offset, detail, bound};
}

}  // namespace

const char* ArchiveError::message() const noexcept {
    switch (code) {
        case Code::Yaz0HeaderTooSmall:
            return "Yaz0 header is truncated";
        case Code::Yaz0MagicMismatch:
            return "Yaz0 magic is invalid";
        case Code::Yaz0OutputLimitExceeded:
            return "Yaz0 declared output exceeds the caller limit";
        case Code::Yaz0AllocationFailure:
            return "Yaz0 output allocation failed";
        case Code::Yaz0TruncatedStream:
            return "Yaz0 compressed stream is truncated";
        case Code::Yaz0BackReferenceBeforeOutput:
            return "Yaz0 back-reference precedes the decoded output";
        case Code::SarcHeaderTooSmall:
            return "SARC header is truncated";
        case Code::SarcMagicMismatch:
            return "SARC magic is invalid";
        case Code::SarcUnsupportedByteOrder:
            return "SARC byte order is unsupported";
        case Code::SarcHeaderSizeTooSmall:
            return "SARC declared header is too small";
        case Code::SarcHeaderOutsideInput:
            return "SARC declared header is outside the input";
        case Code::SarcDeclaredSizeBeforeHeader:
            return "SARC declared file size ends before its header";
        case Code::SarcDeclaredSizeBeyondInput:
            return "SARC declared file size exceeds the input";
        case Code::SarcTruncatedSfat:
            return "SARC SFAT header is truncated";
        case Code::SarcMissingSfat:
            return "SARC SFAT magic is missing";
        case Code::SarcNodeTableOutsideInput:
            return "SARC node table is outside the declared archive extent";
        case Code::SarcTruncatedSfnt:
            return "SARC SFNT header is truncated";
        case Code::SarcMissingSfnt:
            return "SARC SFNT magic is missing";
        case Code::SarcDataOffsetBeforeNames:
            return "SARC data begins before the name table";
        case Code::SarcDataOffsetOutsideInput:
            return "SARC data offset is outside the declared archive extent";
        case Code::SarcNameOffsetOutsideTable:
            return "SARC member name offset is outside the name table";
        case Code::SarcUnterminatedName:
            return "SARC member name is not terminated in the name table";
        case Code::SarcMemberRangeReversed:
            return "SARC member range is reversed";
        case Code::SarcMemberOutsideData:
            return "SARC member is outside the archive data extent";
    }
    return "unknown archive error";
}

Yaz0Result decompress_yaz0(std::span<const std::byte> input,
                           std::size_t max_output_size) {
    if (input.size() < yaz0_header_size) {
        return error(ArchiveError::Code::Yaz0HeaderTooSmall, input.size(),
                     input.size(), yaz0_header_size);
    }
    if (!has_magic(input, 0, "Yaz0")) {
        return error(ArchiveError::Code::Yaz0MagicMismatch);
    }

    const std::uint32_t declared_size = read_be32(input, 4);
    if (declared_size > max_output_size) {
        return error(ArchiveError::Code::Yaz0OutputLimitExceeded, 4,
                     declared_size, max_output_size);
    }

    std::vector<std::byte> output;
    try {
        output.resize(declared_size);
    } catch (const std::bad_alloc&) {
        return error(ArchiveError::Code::Yaz0AllocationFailure, 4,
                     declared_size, max_output_size);
    } catch (const std::length_error&) {
        return error(ArchiveError::Code::Yaz0AllocationFailure, 4,
                     declared_size, max_output_size);
    }

    std::size_t cursor = yaz0_header_size;
    std::size_t produced = 0;
    while (produced < output.size()) {
        if (cursor >= input.size()) {
            return error(ArchiveError::Code::Yaz0TruncatedStream, cursor,
                         produced, declared_size);
        }
        const std::uint8_t flags = byte_value(input[cursor++]);
        for (int bit = 7; bit >= 0 && produced < output.size(); --bit) {
            if ((flags & (1u << bit)) != 0) {
                if (cursor >= input.size()) {
                    return error(ArchiveError::Code::Yaz0TruncatedStream,
                                 cursor, produced, declared_size);
                }
                output[produced++] = input[cursor++];
                continue;
            }

            const std::size_t token_offset = cursor;
            if (cursor > input.size() || input.size() - cursor < 2) {
                return error(ArchiveError::Code::Yaz0TruncatedStream, cursor,
                             produced, declared_size);
            }
            const std::uint8_t first = byte_value(input[cursor]);
            const std::uint8_t second = byte_value(input[cursor + 1]);
            cursor += 2;

            const std::size_t distance =
                ((static_cast<std::size_t>(first & 0x0f) << 8) | second) + 1;
            std::size_t length = static_cast<std::size_t>(first >> 4);
            if (length == 0) {
                if (cursor >= input.size()) {
                    return error(ArchiveError::Code::Yaz0TruncatedStream,
                                 cursor, produced, declared_size);
                }
                length = static_cast<std::size_t>(byte_value(input[cursor++])) +
                         0x12;
            } else {
                length += 2;
            }

            if (distance > produced) {
                return error(ArchiveError::Code::Yaz0BackReferenceBeforeOutput,
                             token_offset, distance, produced);
            }
            const std::size_t copy_count =
                std::min(length, output.size() - produced);
            for (std::size_t i = 0; i < copy_count; ++i) {
                output[produced] = output[produced - distance];
                ++produced;
            }
        }
    }
    return output;
}

SarcResult parse_sarc(std::span<const std::byte> input) {
    if (input.size() < sarc_header_size) {
        return error(ArchiveError::Code::SarcHeaderTooSmall, input.size(),
                     input.size(), sarc_header_size);
    }
    if (!has_magic(input, 0, "SARC")) {
        return error(ArchiveError::Code::SarcMagicMismatch);
    }

    const std::size_t header_size = read_le16(input, 4);
    const std::uint16_t byte_order = read_le16(input, 6);
    if (byte_order != 0xfeffu) {
        return error(ArchiveError::Code::SarcUnsupportedByteOrder, 6,
                     byte_order, 0xfeffu);
    }
    if (header_size < sarc_header_size) {
        return error(ArchiveError::Code::SarcHeaderSizeTooSmall, 4, header_size,
                     sarc_header_size);
    }
    if (header_size > input.size()) {
        return error(ArchiveError::Code::SarcHeaderOutsideInput, 4, header_size,
                     input.size());
    }

    const std::size_t declared_size =
        read_le32(input, sarc_declared_size_offset);
    if (declared_size < header_size) {
        return error(ArchiveError::Code::SarcDeclaredSizeBeforeHeader,
                     sarc_declared_size_offset, declared_size, header_size);
    }
    if (declared_size > input.size()) {
        return error(ArchiveError::Code::SarcDeclaredSizeBeyondInput,
                     sarc_declared_size_offset, declared_size, input.size());
    }
    const std::span<const std::byte> archive_input =
        input.first(declared_size);

    const std::size_t sfat_offset = header_size;
    if (archive_input.size() - sfat_offset < sfat_header_size) {
        return error(ArchiveError::Code::SarcTruncatedSfat, sfat_offset,
                     archive_input.size() - sfat_offset, sfat_header_size);
    }
    if (!has_magic(archive_input, sfat_offset, "SFAT")) {
        return error(ArchiveError::Code::SarcMissingSfat, sfat_offset);
    }

    const std::size_t member_count =
        read_le16(archive_input, sfat_offset + 6);
    const std::size_t nodes_offset = sfat_offset + sfat_header_size;
    const std::size_t nodes_size = member_count * sfat_node_size;
    if (nodes_size > archive_input.size() - nodes_offset) {
        return error(ArchiveError::Code::SarcNodeTableOutsideInput,
                     nodes_offset, nodes_size,
                     archive_input.size() - nodes_offset);
    }

    const std::size_t sfnt_offset = nodes_offset + nodes_size;
    if (archive_input.size() - sfnt_offset < sfnt_header_size) {
        return error(ArchiveError::Code::SarcTruncatedSfnt, sfnt_offset,
                     archive_input.size() - sfnt_offset, sfnt_header_size);
    }
    if (!has_magic(archive_input, sfnt_offset, "SFNT")) {
        return error(ArchiveError::Code::SarcMissingSfnt, sfnt_offset);
    }

    const std::size_t names_offset = sfnt_offset + sfnt_header_size;
    const std::size_t data_offset =
        read_le32(archive_input, sarc_data_offset_field);
    if (data_offset > archive_input.size()) {
        return error(ArchiveError::Code::SarcDataOffsetOutsideInput,
                     sarc_data_offset_field, data_offset,
                     archive_input.size());
    }
    if (data_offset < names_offset) {
        return error(ArchiveError::Code::SarcDataOffsetBeforeNames,
                     sarc_data_offset_field, data_offset, names_offset);
    }

    SarcArchive archive{archive_input, {}};
    archive.members.reserve(member_count);
    const std::size_t name_table_size = data_offset - names_offset;
    const std::size_t data_extent = archive_input.size() - data_offset;
    for (std::size_t index = 0; index < member_count; ++index) {
        const std::size_t node_offset = nodes_offset + index * sfat_node_size;
        const std::uint32_t attributes =
            read_le32(archive_input, node_offset + 4);
        const std::uint32_t member_start =
            read_le32(archive_input, node_offset + 8);
        const std::uint32_t member_end =
            read_le32(archive_input, node_offset + 12);

        SarcMember member;
        member.has_name = (attributes >> 24) != 0;
        if (member.has_name) {
            const std::size_t name_word_offset = attributes & 0x00ffffffu;
            const std::size_t relative_name_offset = name_word_offset * 4;
            if (relative_name_offset >= name_table_size) {
                return error(ArchiveError::Code::SarcNameOffsetOutsideTable,
                             node_offset + 4, relative_name_offset,
                             name_table_size);
            }
            const std::size_t name_offset = names_offset + relative_name_offset;
            const void* terminator =
                std::memchr(archive_input.data() + name_offset, 0,
                            data_offset - name_offset);
            if (terminator == nullptr) {
                return error(ArchiveError::Code::SarcUnterminatedName,
                             name_offset, data_offset - name_offset, 0);
            }
            const auto* name_end = static_cast<const std::byte*>(terminator);
            member.name = std::string_view(
                reinterpret_cast<const char*>(archive_input.data() +
                                              name_offset),
                static_cast<std::size_t>(name_end -
                                         (archive_input.data() + name_offset)));
        }

        if (member_start > member_end) {
            return error(ArchiveError::Code::SarcMemberRangeReversed,
                         node_offset + 8, member_start, member_end);
        }
        if (member_end > data_extent) {
            return error(ArchiveError::Code::SarcMemberOutsideData,
                         node_offset + 12, member_end, data_extent);
        }
        member.data = archive_input.subspan(data_offset + member_start,
                                            member_end - member_start);
        archive.members.push_back(member);
    }
    return archive;
}

}  // namespace armigo::shader_input
