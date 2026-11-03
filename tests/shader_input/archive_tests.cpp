// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "test_support.h"

#include <armigo/shader_input/archive.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::shader_input {
namespace {

std::byte as_byte(std::uint8_t value) { return static_cast<std::byte>(value); }

void put_le16(std::vector<std::byte>& bytes, std::size_t offset,
              std::uint16_t value) {
    bytes[offset] = as_byte(static_cast<std::uint8_t>(value));
    bytes[offset + 1] = as_byte(static_cast<std::uint8_t>(value >> 8));
}

void put_le32(std::vector<std::byte>& bytes, std::size_t offset,
              std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        bytes[offset + index] =
            as_byte(static_cast<std::uint8_t>(value >> (index * 8)));
    }
}

void put_be32(std::vector<std::byte>& bytes, std::size_t offset,
              std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        bytes[offset + index] =
            as_byte(static_cast<std::uint8_t>(value >> ((3 - index) * 8)));
    }
}

void put_magic(std::vector<std::byte>& bytes, std::size_t offset,
               std::string_view magic) {
    for (std::size_t index = 0; index < magic.size(); ++index) {
        bytes[offset + index] =
            as_byte(static_cast<std::uint8_t>(magic[index]));
    }
}

std::vector<std::byte> yaz0_stream(std::uint32_t output_size,
                                   std::initializer_list<std::uint8_t> body) {
    std::vector<std::byte> bytes(16 + body.size());
    put_magic(bytes, 0, "Yaz0");
    put_be32(bytes, 4, output_size);
    std::size_t cursor = 16;
    for (std::uint8_t value : body) {
        bytes[cursor++] = as_byte(value);
    }
    return bytes;
}

const std::vector<std::byte>* must_decompress(const Yaz0Result& result,
                                              const char* file, int line) {
    if (const auto* output = std::get_if<std::vector<std::byte>>(&result)) {
        return output;
    }
    const ArchiveError& failure = std::get<ArchiveError>(result);
    test::record_failure(file, line,
                         std::string("decompress_yaz0 unexpectedly failed: ") +
                             failure.message());
    return nullptr;
}

const ArchiveError* must_fail(const Yaz0Result& result,
                              ArchiveError::Code expected, const char* file,
                              int line) {
    const ArchiveError* failure = std::get_if<ArchiveError>(&result);
    if (failure == nullptr) {
        test::record_failure(file, line,
                             "decompress_yaz0 unexpectedly succeeded");
        return nullptr;
    }
    if (failure->code != expected) {
        test::record_failure(file, line,
                             std::string("unexpected Yaz0 error: ") +
                                 failure->message());
    }
    return failure;
}

const SarcArchive* must_parse_sarc(const SarcResult& result, const char* file,
                                   int line) {
    if (const auto* archive = std::get_if<SarcArchive>(&result)) {
        return archive;
    }
    const ArchiveError& failure = std::get<ArchiveError>(result);
    test::record_failure(file, line,
                         std::string("parse_sarc unexpectedly failed: ") +
                             failure.message());
    return nullptr;
}

const ArchiveError* must_fail(const SarcResult& result,
                              ArchiveError::Code expected, const char* file,
                              int line) {
    const ArchiveError* failure = std::get_if<ArchiveError>(&result);
    if (failure == nullptr) {
        test::record_failure(file, line, "parse_sarc unexpectedly succeeded");
        return nullptr;
    }
    if (failure->code != expected) {
        test::record_failure(file, line,
                             std::string("unexpected SARC error: ") +
                                 failure->message());
    }
    return failure;
}

std::vector<std::byte> synthetic_sarc() {
    constexpr std::size_t sfat = 0x14;
    constexpr std::size_t nodes = 0x20;
    constexpr std::size_t sfnt = 0x40;
    constexpr std::size_t names = 0x48;
    constexpr std::size_t data = 0x60;
    std::vector<std::byte> bytes(data + 7);

    put_magic(bytes, 0, "SARC");
    put_le16(bytes, 4, 0x14);
    put_le16(bytes, 6, 0xfeff);
    put_le32(bytes, 8, static_cast<std::uint32_t>(bytes.size()));
    put_le32(bytes, 0x0c, data);

    put_magic(bytes, sfat, "SFAT");
    put_le16(bytes, sfat + 4, 12);
    put_le16(bytes, sfat + 6, 2);

    put_le32(bytes, nodes + 4, 0x01000000);
    put_le32(bytes, nodes + 8, 0);
    put_le32(bytes, nodes + 12, 3);
    put_le32(bytes, nodes + 16 + 4, 0x01000003);
    put_le32(bytes, nodes + 16 + 8, 3);
    put_le32(bytes, nodes + 16 + 12, 7);

    put_magic(bytes, sfnt, "SFNT");
    put_le16(bytes, sfnt + 4, 8);
    put_magic(bytes, names, "first.bin");
    bytes[names + 9] = std::byte{0};
    put_magic(bytes, names + 12, "two");
    bytes[names + 15] = std::byte{0};

    const std::array<std::uint8_t, 7> payload = {1, 2, 3, 9, 8, 7, 6};
    for (std::size_t index = 0; index < payload.size(); ++index) {
        bytes[data + index] = as_byte(payload[index]);
    }
    return bytes;
}

}  // namespace

ARMIGO_SI_TEST(yaz0, decodes_literals_and_overlapping_back_reference) {
    // Three literals "abc", then a distance-three, length-nine run. Copying
    // one byte at a time makes the source overlap produce "abcabcabcabc".
    const std::vector<std::byte> input =
        yaz0_stream(12, {0xe0, 'a', 'b', 'c', 0x70, 0x02});
    const Yaz0Result result = decompress_yaz0(input, 12);
    const std::vector<std::byte>* output =
        must_decompress(result, __FILE__, __LINE__);
    if (output != nullptr) {
        const std::string_view expected = "abcabcabcabc";
        ARMIGO_SI_CHECK_EQ(output->size(), expected.size());
        ARMIGO_SI_CHECK(
            std::memcmp(output->data(), expected.data(), expected.size()) == 0);
    }
}

ARMIGO_SI_TEST(yaz0, decodes_extended_run_and_clamps_to_declared_size) {
    const std::vector<std::byte> input =
        yaz0_stream(20, {0x80, 'x', 0x00, 0x00, 0xff});
    const Yaz0Result result = decompress_yaz0(input, 20);
    const std::vector<std::byte>* output =
        must_decompress(result, __FILE__, __LINE__);
    if (output != nullptr) {
        ARMIGO_SI_CHECK_EQ(output->size(), static_cast<std::size_t>(20));
        for (std::byte value : *output) {
            ARMIGO_SI_CHECK(value == as_byte('x'));
        }
    }
}

ARMIGO_SI_TEST(yaz0, accepts_zero_output_without_a_body) {
    const std::vector<std::byte> input = yaz0_stream(0, {});
    const Yaz0Result result = decompress_yaz0(input, 0);
    const std::vector<std::byte>* output =
        must_decompress(result, __FILE__, __LINE__);
    if (output != nullptr) {
        ARMIGO_SI_CHECK(output->empty());
    }
}

ARMIGO_SI_TEST(yaz0, enforces_the_output_cap_before_decoding) {
    const std::vector<std::byte> input = yaz0_stream(4096, {});
    const Yaz0Result result = decompress_yaz0(input, 1024);
    const ArchiveError* failure =
        must_fail(result, ArchiveError::Code::Yaz0OutputLimitExceeded, __FILE__,
                  __LINE__);
    if (failure != nullptr) {
        ARMIGO_SI_CHECK_EQ(failure->offset, static_cast<std::size_t>(4));
        ARMIGO_SI_CHECK_EQ(failure->detail, static_cast<std::uint64_t>(4096));
        ARMIGO_SI_CHECK_EQ(failure->bound, static_cast<std::uint64_t>(1024));
    }
}

ARMIGO_SI_TEST(yaz0, rejects_bad_header_and_truncated_tokens) {
    std::vector<std::byte> short_header(15);
    must_fail(decompress_yaz0(short_header, 100),
              ArchiveError::Code::Yaz0HeaderTooSmall, __FILE__, __LINE__);

    std::vector<std::byte> bad_magic = yaz0_stream(0, {});
    bad_magic[0] = as_byte('N');
    must_fail(decompress_yaz0(bad_magic, 100),
              ArchiveError::Code::Yaz0MagicMismatch, __FILE__, __LINE__);

    must_fail(decompress_yaz0(yaz0_stream(1, {0x80}), 100),
              ArchiveError::Code::Yaz0TruncatedStream, __FILE__, __LINE__);
    must_fail(decompress_yaz0(yaz0_stream(1, {0x00, 0x10}), 100),
              ArchiveError::Code::Yaz0TruncatedStream, __FILE__, __LINE__);
    must_fail(decompress_yaz0(yaz0_stream(1, {0x00, 0x00, 0x00}), 100),
              ArchiveError::Code::Yaz0TruncatedStream, __FILE__, __LINE__);
}

ARMIGO_SI_TEST(yaz0, rejects_back_reference_before_output) {
    const Yaz0Result result =
        decompress_yaz0(yaz0_stream(3, {0x00, 0x10, 0x00}), 3);
    const ArchiveError* failure =
        must_fail(result, ArchiveError::Code::Yaz0BackReferenceBeforeOutput,
                  __FILE__, __LINE__);
    if (failure != nullptr) {
        ARMIGO_SI_CHECK_EQ(failure->offset, static_cast<std::size_t>(17));
        ARMIGO_SI_CHECK_EQ(failure->detail, static_cast<std::uint64_t>(1));
        ARMIGO_SI_CHECK_EQ(failure->bound, static_cast<std::uint64_t>(0));
    }
}

ARMIGO_SI_TEST(yaz0, rejects_every_truncated_prefix) {
    const std::vector<std::byte> complete =
        yaz0_stream(12, {0xe0, 'a', 'b', 'c', 0x70, 0x02});
    for (std::size_t size = 0; size < complete.size(); ++size) {
        const Yaz0Result result =
            decompress_yaz0(std::span<const std::byte>(complete).first(size),
                            12);
        ARMIGO_SI_CHECK(std::holds_alternative<ArchiveError>(result));
    }
}

ARMIGO_SI_TEST(sarc, enumerates_borrowed_named_members) {
    const std::vector<std::byte> bytes = synthetic_sarc();
    const SarcResult result = parse_sarc(bytes);
    const SarcArchive* archive = must_parse_sarc(result, __FILE__, __LINE__);
    if (archive != nullptr) {
        ARMIGO_SI_CHECK(archive->source.data() == bytes.data());
        ARMIGO_SI_CHECK_EQ(archive->members.size(),
                           static_cast<std::size_t>(2));
        ARMIGO_SI_CHECK(archive->members[0].has_name);
        ARMIGO_SI_CHECK_EQ(archive->members[0].name,
                           std::string_view("first.bin"));
        ARMIGO_SI_CHECK(archive->members[0].name.data() ==
                        reinterpret_cast<const char*>(bytes.data() + 0x48));
        ARMIGO_SI_CHECK(archive->members[0].data.data() == bytes.data() + 0x60);
        ARMIGO_SI_CHECK_EQ(archive->members[0].data.size(),
                           static_cast<std::size_t>(3));
        ARMIGO_SI_CHECK_EQ(archive->members[1].name, std::string_view("two"));
        ARMIGO_SI_CHECK(archive->members[1].data.data() == bytes.data() + 0x63);
        ARMIGO_SI_CHECK_EQ(archive->members[1].data.size(),
                           static_cast<std::size_t>(4));
    }
}

ARMIGO_SI_TEST(sarc, represents_an_unnamed_empty_member_without_copying) {
    std::vector<std::byte> bytes = synthetic_sarc();
    put_le32(bytes, 0x20 + 4, 0);
    put_le32(bytes, 0x20 + 12, 0);
    const SarcResult result = parse_sarc(bytes);
    const SarcArchive* archive = must_parse_sarc(result, __FILE__, __LINE__);
    if (archive != nullptr) {
        ARMIGO_SI_CHECK(!archive->members[0].has_name);
        ARMIGO_SI_CHECK(archive->members[0].name.empty());
        ARMIGO_SI_CHECK(archive->members[0].data.empty());
        ARMIGO_SI_CHECK(archive->members[0].data.data() == bytes.data() + 0x60);
    }
}

ARMIGO_SI_TEST(sarc, rejects_headers_tables_and_unsupported_byte_order) {
    std::vector<std::byte> bytes = synthetic_sarc();
    must_fail(parse_sarc(std::span<const std::byte>(bytes).first(19)),
              ArchiveError::Code::SarcHeaderTooSmall, __FILE__, __LINE__);

    bytes = synthetic_sarc();
    bytes[0] = as_byte('X');
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcMagicMismatch,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le16(bytes, 6, 0xfffe);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcUnsupportedByteOrder,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le16(bytes, 4, 0x10);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcHeaderSizeTooSmall,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le16(bytes, 4, 0xffff);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcHeaderOutsideInput,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 8, 0x13);
    must_fail(parse_sarc(bytes),
              ArchiveError::Code::SarcDeclaredSizeBeforeHeader, __FILE__,
              __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 8, static_cast<std::uint32_t>(bytes.size() + 1));
    must_fail(parse_sarc(bytes),
              ArchiveError::Code::SarcDeclaredSizeBeyondInput, __FILE__,
              __LINE__);

    bytes = synthetic_sarc();
    bytes[0x14] = as_byte('X');
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcMissingSfat, __FILE__,
              __LINE__);

    bytes = synthetic_sarc();
    put_le16(bytes, 0x14 + 6, 0xffff);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcNodeTableOutsideInput,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    bytes[0x40] = as_byte('X');
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcMissingSfnt, __FILE__,
              __LINE__);
}

ARMIGO_SI_TEST(sarc, bounds_tables_and_data_by_the_declared_file_size) {
    std::vector<std::byte> bytes = synthetic_sarc();
    put_le32(bytes, 8, 0x30);
    must_fail(parse_sarc(bytes),
              ArchiveError::Code::SarcNodeTableOutsideInput, __FILE__,
              __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 8, 0x50);
    must_fail(parse_sarc(bytes),
              ArchiveError::Code::SarcDataOffsetOutsideInput, __FILE__,
              __LINE__);
}

ARMIGO_SI_TEST(sarc, rejects_invalid_name_table_references) {
    std::vector<std::byte> bytes = synthetic_sarc();
    put_le32(bytes, 0x20 + 4, 0x01000006);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcNameOffsetOutsideTable,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 0x20 + 4, 0x01000005);
    for (std::size_t offset = 0x5c; offset < 0x60; ++offset) {
        bytes[offset] = as_byte('q');
    }
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcUnterminatedName,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 0x0c, 0x47);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcDataOffsetBeforeNames,
              __FILE__, __LINE__);
}

ARMIGO_SI_TEST(sarc, rejects_members_outside_the_data_extent) {
    std::vector<std::byte> bytes = synthetic_sarc();
    put_le32(bytes, 0x20 + 8, 4);
    put_le32(bytes, 0x20 + 12, 3);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcMemberRangeReversed,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 0x20 + 12, 8);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcMemberOutsideData,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 8, 0x60);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcMemberOutsideData,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 0x0c, 0xffff);
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcDataOffsetOutsideInput,
              __FILE__, __LINE__);

    bytes = synthetic_sarc();
    put_le32(bytes, 8, static_cast<std::uint32_t>(bytes.size() - 1));
    must_fail(parse_sarc(bytes), ArchiveError::Code::SarcMemberOutsideData,
              __FILE__, __LINE__);
}

ARMIGO_SI_TEST(sarc, ignores_backing_bytes_after_the_declared_file_size) {
    std::vector<std::byte> bytes = synthetic_sarc();
    const std::size_t declared_size = bytes.size();
    bytes.insert(bytes.end(), 5, as_byte(0xee));

    const SarcResult result = parse_sarc(bytes);
    const SarcArchive* archive = must_parse_sarc(result, __FILE__, __LINE__);
    if (archive != nullptr) {
        ARMIGO_SI_CHECK_EQ(archive->source.size(), declared_size);
        ARMIGO_SI_CHECK_EQ(archive->members.size(),
                           static_cast<std::size_t>(2));
        ARMIGO_SI_CHECK_EQ(archive->members[1].data.size(),
                           static_cast<std::size_t>(4));
        ARMIGO_SI_CHECK(archive->members[1].data.data() +
                            archive->members[1].data.size() ==
                        archive->source.data() + archive->source.size());
    }
}

ARMIGO_SI_TEST(sarc, rejects_every_truncated_prefix) {
    const std::vector<std::byte> complete = synthetic_sarc();
    for (std::size_t size = 0; size < complete.size(); ++size) {
        const SarcResult result =
            parse_sarc(std::span<const std::byte>(complete).first(size));
        ARMIGO_SI_CHECK(std::holds_alternative<ArchiveError>(result));
    }
}

}  // namespace armigo::shader_input
