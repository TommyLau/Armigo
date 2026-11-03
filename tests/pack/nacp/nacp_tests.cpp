// ============================================================================
// Armigo Tests - Control Data Naming Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/pack/nacp.h>

#include <zlib.h>

#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using armigo::pack::nacp::app_file_name;
using armigo::pack::nacp::AppNameError;
using armigo::pack::nacp::AppFileNameResult;
using armigo::pack::nacp::choose_icon;
using armigo::pack::nacp::choose_title;
using armigo::pack::nacp::control_file_size;
using armigo::pack::nacp::display_version_offset;
using armigo::pack::nacp::display_version_size;
using armigo::pack::nacp::icon_file_name;
using armigo::pack::nacp::inflated_title_entry_count;
using armigo::pack::nacp::language_count;
using armigo::pack::nacp::language_name;
using armigo::pack::nacp::language_names;
using armigo::pack::nacp::Language;
using armigo::pack::nacp::max_app_file_name_bytes;
using armigo::pack::nacp::name_field_size;
using armigo::pack::nacp::NacpError;
using armigo::pack::nacp::parse_nacp;
using armigo::pack::nacp::publisher_field_size;
using armigo::pack::nacp::supported_language_flag_offset;
using armigo::pack::nacp::title_area_size;
using armigo::pack::nacp::title_entry_count;
using armigo::pack::nacp::title_entry_size;
using armigo::pack::nacp::title_format_offset;
using armigo::pack::nacp::to_string;

unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); ++failures; } } while (0)

std::string repeated(std::string_view unit, std::size_t count) {
    std::string result;
    result.reserve(unit.size() * count);
    for (std::size_t index = 0; index < count; ++index) {
        result += unit;
    }
    return result;
}

// Compresses bytes as the newer control data does: one raw DEFLATE stream,
// built with the same window bits the reader inflates with.
std::vector<std::uint8_t> deflate_raw(const std::vector<std::uint8_t>& source) {
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(compressBound(static_cast<uLongf>(source.size()))) + 16);
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(source.data());
    stream.avail_in = static_cast<uInt>(source.size());
    stream.next_out = output.data();
    stream.avail_out = static_cast<uInt>(output.size());
    CHECK(deflateInit2(&stream, 6, Z_DEFLATED, -15, 9, Z_DEFAULT_STRATEGY) == Z_OK);
    CHECK(deflate(&stream, Z_FINISH) == Z_STREAM_END);
    output.resize(stream.total_out);
    deflateEnd(&stream);
    return output;
}

// Pads a stream with zero bytes to a multiple of 256 so the length prefix's
// low byte is NUL and the owner's name heuristic reads "no entry 0 title";
// used where a test falls back to that heuristic. Inflate, like the owner's
// zlib.decompress, accepts the trailing bytes.
std::vector<std::uint8_t> pad_stream(std::vector<std::uint8_t> stream) {
    while (stream.size() % 256 != 0) {
        stream.push_back(0);
    }
    return stream;
}

// A synthetic title table: entry_count entries of 0x300 bytes, every field
// zeroed, so written values are always NUL-terminated.
class TitleArea {
public:
    explicit TitleArea(std::size_t entry_count_in)
        : bytes_(entry_count_in * title_entry_size, 0) {}

    void set_name(std::size_t index, std::string_view value) {
        write(index * title_entry_size, name_field_size, value);
    }

    void set_publisher(std::size_t index, std::string_view value) {
        write(index * title_entry_size + name_field_size, publisher_field_size, value);
    }

    void set_name(Language language, std::string_view value) {
        set_name(static_cast<std::size_t>(language), value);
    }

    void set_publisher(Language language, std::string_view value) {
        set_publisher(static_cast<std::size_t>(language), value);
    }

    // Overwrites a whole field with one non-NUL byte so it has no
    // terminator.
    void fill_name(std::size_t index, std::uint8_t byte) {
        fill(index * title_entry_size, name_field_size, byte);
    }

    void fill_publisher(std::size_t index, std::uint8_t byte) {
        fill(index * title_entry_size + name_field_size, publisher_field_size, byte);
    }

    const std::vector<std::uint8_t>& bytes() const { return bytes_; }

private:
    void write(std::size_t offset, std::size_t field_size, std::string_view value) {
        CHECK(value.size() < field_size);
        if (value.size() < field_size) {
            std::memcpy(bytes_.data() + offset, value.data(), value.size());
        }
    }

    void fill(std::size_t offset, std::size_t field_size, std::uint8_t byte) {
        std::memset(bytes_.data() + offset, byte, field_size);
    }

    std::vector<std::uint8_t> bytes_;
};

// A synthetic 0x4000-byte control data blob. The title area starts as all
// zeros, which parses as an empty classic table.
class ControlDataBlob {
public:
    ControlDataBlob() : bytes_(control_file_size, 0) {}

    // Copies a classic sixteen-entry table into the title area.
    void embed_classic(const TitleArea& area) {
        CHECK(area.bytes().size() == title_entry_count * title_entry_size);
        std::memcpy(bytes_.data(), area.bytes().data(), area.bytes().size());
    }

    // Compresses the area as the newer format stores it and selects the
    // compressed title data format.
    void embed_compressed(const TitleArea& area) {
        const std::vector<std::uint8_t> stream = deflate_raw(area.bytes());
        embed_stream(stream, stream.size());
    }

    // Writes raw bytes as the compressed stream with an explicit declared
    // length, for malformed-stream cases, and selects the compressed title
    // data format.
    void embed_stream(const std::vector<std::uint8_t>& stream, std::size_t declared_length) {
        set_title_format(1);
        bytes_[0] = static_cast<std::uint8_t>(declared_length & 0xFFu);
        bytes_[1] = static_cast<std::uint8_t>((declared_length >> 8) & 0xFFu);
        CHECK(stream.size() <= declared_length);
        if (stream.size() <= declared_length) {
            std::memcpy(bytes_.data() + 2, stream.data(), stream.size());
        }
    }

    // The blob starts as format 0, the classic table.
    void set_title_format(std::uint8_t value) {
        bytes_[title_format_offset] = value;
    }

    void set_supported_language_flag(std::uint32_t value) {
        for (unsigned index = 0; index < 4; ++index) {
            bytes_[supported_language_flag_offset + index] = static_cast<std::uint8_t>(value >> (index * 8));
        }
    }

    void set_display_version(std::string_view value) {
        CHECK(value.size() < display_version_size);
        if (value.size() < display_version_size) {
            std::memcpy(bytes_.data() + display_version_offset, value.data(), value.size());
        }
    }

    void fill_display_version(std::uint8_t byte) {
        std::memset(bytes_.data() + display_version_offset, byte, display_version_size);
    }

    std::span<const std::uint8_t> bytes() const { return bytes_; }

    std::vector<std::uint8_t> resized(std::size_t size) const {
        return {bytes_.begin(), bytes_.begin() + static_cast<std::ptrdiff_t>(size)};
    }

private:
    std::vector<std::uint8_t> bytes_;
};

void layout_constants_pin_the_control_format() {
    CHECK(title_entry_count == 16);
    CHECK(inflated_title_entry_count == 32);
    CHECK(title_entry_count * title_entry_size == title_area_size);
    CHECK(inflated_title_entry_count * title_entry_size == 0x6000);
    CHECK(name_field_size + publisher_field_size == title_entry_size);
    CHECK(display_version_offset == 0x3060);
    CHECK(display_version_offset + display_version_size <= control_file_size);
    CHECK(title_format_offset == 0x3215);
    CHECK(title_format_offset < control_file_size);
    CHECK(control_file_size == 0x4000);
    CHECK(language_count == 16);
    CHECK(to_string(NacpError::too_short) == std::string_view("too_short"));
    CHECK(to_string(NacpError::inflate_failed) == std::string_view("inflate_failed"));
}

void parses_the_classic_title_table() {
    TitleArea area(title_entry_count);
    area.set_name(Language::american_english, "Open Country");
    area.set_publisher(Language::american_english, "Sample Publisher");
    area.set_name(Language::japanese, "Far East Adventure");
    area.set_publisher(Language::japanese, "Eastern Publisher");
    area.set_name(Language::german, "Abenteuer");
    ControlDataBlob blob;
    blob.embed_classic(area);
    blob.set_supported_language_flag(0x2fbff);
    blob.set_display_version("1.5.0");

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.error == NacpError::none);
    CHECK(parsed.detail.empty());
    CHECK(parsed.warnings.empty());
    CHECK(parsed.nacp.entry_count == title_entry_count);
    CHECK(parsed.nacp.entries[0].name == "Open Country");
    CHECK(parsed.nacp.entries[0].publisher == "Sample Publisher");
    CHECK(parsed.nacp.entries[1].name.empty());
    CHECK(parsed.nacp.entries[1].publisher.empty());
    CHECK(parsed.nacp.entries[2].name == "Far East Adventure");
    CHECK(parsed.nacp.entries[2].publisher == "Eastern Publisher");
    CHECK(parsed.nacp.entries[4].name == "Abenteuer");
    CHECK(parsed.nacp.entries[15].name.empty());
    CHECK(parsed.nacp.supported_language_flag == 0x2fbff);
    CHECK(parsed.nacp.display_version == "1.5.0");
}

void american_english_title_wins_when_present() {
    TitleArea area(title_entry_count);
    area.set_name(Language::american_english, "Alpha");
    area.set_name(Language::british_english, "Beta");
    area.set_name(Language::japanese, "Gamma");
    ControlDataBlob blob;
    blob.embed_classic(area);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::american_english);
    CHECK(choice.name == "Alpha");
}

// The format field lets a classic table carry an empty AmericanEnglish
// entry, so these fallbacks run through the classic format.
void british_english_title_used_when_american_absent() {
    TitleArea area(title_entry_count);
    area.set_name(Language::british_english, "Beta");
    area.set_name(Language::french, "Bonjour");
    ControlDataBlob blob;
    blob.embed_classic(area);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::british_english);
    CHECK(choice.name == "Beta");
}

void simplified_chinese_preferred_over_traditional() {
    TitleArea area(title_entry_count);
    area.set_name(Language::traditional_chinese, "Traditional");
    area.set_name(Language::simplified_chinese, "Simplified");
    ControlDataBlob blob;
    blob.embed_classic(area);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::simplified_chinese);
    CHECK(choice.name == "Simplified");
}

void korean_and_japanese_pick_first_in_entry_order() {
    TitleArea area(title_entry_count);
    area.set_name(Language::korean, "Hanul");
    area.set_name(Language::japanese, "Higashi");
    ControlDataBlob blob;
    blob.embed_classic(area);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::japanese);
    CHECK(choice.name == "Higashi");
}

void choose_title_fails_when_every_entry_is_empty() {
    ControlDataBlob blob;
    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.nacp.entry_count == title_entry_count);

    const auto choice = choose_title(parsed.nacp);
    CHECK(!choice.ok());
    CHECK(choice.error == armigo::pack::nacp::TitleChoiceError::no_title_entry);
    CHECK(!choice.detail.empty());
    CHECK(choice.name.empty());
    CHECK(to_string(choice.error) == std::string_view("no_title_entry"));
}

void parse_rejects_short_buffers() {
    ControlDataBlob blob;

    const auto empty = parse_nacp({});
    CHECK(empty.error == NacpError::too_short);
    CHECK(!empty.detail.empty());
    CHECK(empty.nacp.entries[0].name.empty());

    const auto short_buffer = parse_nacp(blob.resized(control_file_size - 1));
    CHECK(short_buffer.error == NacpError::too_short);
    CHECK(!short_buffer.detail.empty());
}

void parse_accepts_exact_and_longer_buffers() {
    TitleArea area(title_entry_count);
    area.set_name(Language::american_english, "Alpha");
    ControlDataBlob blob;
    blob.embed_classic(area);

    const auto exact = parse_nacp(blob.bytes());
    CHECK(exact.ok());

    const auto parsed = parse_nacp(blob.resized(control_file_size + 1));
    CHECK(parsed.ok());
    CHECK(parsed.nacp.entries[0].name == "Alpha");
}

void full_width_fields_are_read_with_a_warning() {
    TitleArea area(title_entry_count);
    // Entry 0's full-width name still says "uncompressed table": the
    // detection accepts a field without a terminator.
    area.fill_name(0, 'A');
    area.fill_publisher(1, 'P');
    ControlDataBlob blob;
    blob.embed_classic(area);
    blob.fill_display_version('7');

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.nacp.entry_count == title_entry_count);
    CHECK(parsed.nacp.entries[0].name == std::string(name_field_size, 'A'));
    CHECK(parsed.nacp.entries[1].publisher == std::string(publisher_field_size, 'P'));
    CHECK(parsed.nacp.display_version == std::string(display_version_size, '7'));
    CHECK(parsed.warnings.size() == 3);
}

void unreadable_american_english_name_is_empty_with_a_warning() {
    TitleArea area(title_entry_count);
    area.set_name(Language::american_english, "\xFF");
    area.set_name(Language::british_english, "Beta");
    ControlDataBlob blob;
    blob.embed_classic(area);

    // The format field says classic, so the unreadable name is just a
    // damaged field and the choice moves to the next name.
    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.nacp.entry_count == title_entry_count);
    CHECK(parsed.nacp.entries[0].name.empty());
    CHECK(parsed.warnings.size() == 1);
    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::british_english);
    CHECK(choice.name == "Beta");
}

void unreadable_fields_become_empty_with_a_warning() {
    TitleArea area(title_entry_count);
    area.set_name(Language::american_english, "Alpha");
    area.set_name(Language::british_english, "safe\x80name");
    area.set_publisher(Language::japanese, "bad\xFFpublisher");
    ControlDataBlob blob;
    blob.embed_classic(area);
    blob.set_display_version("1.\xFF.0");

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.error == NacpError::none);
    CHECK(parsed.nacp.entries[0].name == "Alpha");
    CHECK(parsed.nacp.entries[1].name.empty());
    CHECK(parsed.nacp.entries[2].publisher.empty());
    CHECK(parsed.nacp.display_version.empty());
    CHECK(parsed.warnings.size() == 3);
    // The damaged fields do not stop the choice; American English wins.
    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::american_english);
    CHECK(choice.name == "Alpha");
}

void max_length_name_is_parsed() {
    TitleArea area(title_entry_count);
    area.set_name(Language::american_english, std::string(name_field_size - 1, 'N'));
    ControlDataBlob blob;
    blob.embed_classic(area);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.warnings.empty());
    CHECK(parsed.nacp.entries[0].name.size() == name_field_size - 1);
}

void compressed_title_area_round_trips() {
    TitleArea area(inflated_title_entry_count);
    area.set_name(Language::american_english, "Open Country");
    area.set_publisher(Language::american_english, "Sample Publisher");
    area.set_name(Language::japanese, "Far East Adventure");
    area.set_name(16, "Unnamed Extra");
    area.set_publisher(16, "Extra Publisher");
    area.set_name(17, "Eastern Title");
    ControlDataBlob blob;
    blob.embed_compressed(area);
    blob.set_display_version("1.9.0");

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.error == NacpError::none);
    CHECK(parsed.warnings.empty());
    CHECK(parsed.nacp.entry_count == inflated_title_entry_count);
    CHECK(parsed.nacp.entries[0].name == "Open Country");
    CHECK(parsed.nacp.entries[0].publisher == "Sample Publisher");
    CHECK(parsed.nacp.entries[2].name == "Far East Adventure");
    // Entries past the named languages are kept as unnamed extra entries.
    CHECK(parsed.nacp.entries[16].name == "Unnamed Extra");
    CHECK(parsed.nacp.entries[16].publisher == "Extra Publisher");
    CHECK(parsed.nacp.entries[17].name == "Eastern Title");
    CHECK(parsed.nacp.entries[31].name.empty());
    CHECK(parsed.nacp.display_version == "1.9.0");

    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::american_english);
    CHECK(choice.name == "Open Country");
}

void compressed_title_area_ignores_extra_entries_in_the_name_rule() {
    TitleArea area(inflated_title_entry_count);
    area.set_name(17, "Eastern Title");
    ControlDataBlob blob;
    blob.embed_compressed(area);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.nacp.entry_count == inflated_title_entry_count);
    CHECK(!parsed.nacp.entries[17].name.empty());
    const auto choice = choose_title(parsed.nacp);
    CHECK(!choice.ok());
    CHECK(choice.error == armigo::pack::nacp::TitleChoiceError::no_title_entry);
}

void compressed_title_area_accepts_trailing_stream_bytes() {
    TitleArea area(inflated_title_entry_count);
    area.set_name(Language::american_english, "Alpha");
    std::vector<std::uint8_t> stream = deflate_raw(area.bytes());
    stream.insert(stream.end(), 4, 0);
    ControlDataBlob blob;
    blob.embed_stream(stream, stream.size());
    blob.set_display_version("1.0.0");

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.nacp.entry_count == inflated_title_entry_count);
    CHECK(parsed.nacp.entries[0].name == "Alpha");
}

void compressed_title_area_rejects_a_truncated_stream() {
    TitleArea area(inflated_title_entry_count);
    area.set_name(Language::american_english, "Alpha");
    std::vector<std::uint8_t> stream = deflate_raw(area.bytes());
    stream.resize(stream.size() - 4);
    ControlDataBlob blob;
    blob.embed_stream(stream, stream.size());

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(!parsed.ok());
    CHECK(parsed.error == NacpError::inflate_failed);
    CHECK(!parsed.detail.empty());
}

void compressed_title_area_rejects_wrong_inflated_sizes() {
    constexpr std::size_t inflated_size = inflated_title_entry_count * title_entry_size;

    const std::vector<std::uint8_t> short_area(inflated_size - 1, 0);
    const std::vector<std::uint8_t> short_stream = deflate_raw(short_area);
    ControlDataBlob short_blob;
    short_blob.embed_stream(short_stream, short_stream.size());
    const auto too_short = parse_nacp(short_blob.bytes());
    CHECK(!too_short.ok());
    CHECK(too_short.error == NacpError::inflate_failed);
    CHECK(!too_short.detail.empty());

    const std::vector<std::uint8_t> long_area(inflated_size + 1, 0);
    const std::vector<std::uint8_t> long_stream = deflate_raw(long_area);
    ControlDataBlob long_blob;
    long_blob.embed_stream(long_stream, long_stream.size());
    const auto too_long = parse_nacp(long_blob.bytes());
    CHECK(!too_long.ok());
    CHECK(too_long.error == NacpError::inflate_failed);
    CHECK(!too_long.detail.empty());
}

// The format field decides even when the length prefix alone would read as
// a title to the name heuristic: a small stream whose size byte is
// printable ASCII still inflates.
void small_compressed_stream_is_parsed_as_compressed() {
    TitleArea area(inflated_title_entry_count);
    area.set_name(Language::american_english, "Alpha");
    const std::vector<std::uint8_t> stream = deflate_raw(area.bytes());
    ControlDataBlob blob;
    blob.embed_stream(stream, stream.size());

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.warnings.empty());
    CHECK(parsed.nacp.entry_count == inflated_title_entry_count);
    CHECK(parsed.nacp.entries[0].name == "Alpha");
}

void compressed_title_area_rejects_an_oversized_length_prefix() {
    TitleArea area(inflated_title_entry_count);
    area.set_name(Language::american_english, "Alpha");
    const std::vector<std::uint8_t> stream = deflate_raw(area.bytes());
    ControlDataBlob blob;
    blob.embed_stream(stream, title_area_size - 1);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(!parsed.ok());
    CHECK(parsed.error == NacpError::inflate_failed);
    CHECK(!parsed.detail.empty());
}

void compressed_title_area_rejects_garbage_streams() {
    const std::vector<std::uint8_t> garbage(16, 0xFF);
    ControlDataBlob blob;
    blob.embed_stream(garbage, garbage.size());

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(!parsed.ok());
    CHECK(parsed.error == NacpError::inflate_failed);
    CHECK(!parsed.detail.empty());
}

// The case the coordinator hit on a real Asian release: a classic table
// with an empty AmericanEnglish entry, which the name heuristic alone
// cannot tell from a compressed area.
void classic_table_with_empty_american_english_picks_the_next_name() {
    TitleArea area(title_entry_count);
    area.set_name(Language::japanese, "Higashi");
    ControlDataBlob blob;
    blob.embed_classic(area);

    const auto parsed = parse_nacp(blob.bytes());
    CHECK(parsed.ok());
    CHECK(parsed.warnings.empty());
    CHECK(parsed.nacp.entry_count == title_entry_count);
    const auto choice = choose_title(parsed.nacp);
    CHECK(choice.ok());
    CHECK(choice.language == Language::japanese);
    CHECK(choice.name == "Higashi");
}

// A format field the library cannot interpret falls back to the owner's
// name heuristic, on record as a warning.
void unknown_format_field_falls_back_by_the_entry_zero_name() {
    TitleArea classic_area(title_entry_count);
    classic_area.set_name(Language::american_english, "Alpha");
    ControlDataBlob classic_blob;
    classic_blob.embed_classic(classic_area);
    classic_blob.set_title_format(2);
    const auto parsed_classic = parse_nacp(classic_blob.bytes());
    CHECK(parsed_classic.ok());
    CHECK(parsed_classic.nacp.entry_count == title_entry_count);
    CHECK(parsed_classic.warnings.size() == 1);
    CHECK(parsed_classic.warnings.front().find("0x3215") != std::string::npos);
    CHECK(parsed_classic.warnings.front().find("classic") != std::string::npos);

    TitleArea empty_area(title_entry_count);
    empty_area.set_name(Language::japanese, "Higashi");
    ControlDataBlob empty_blob;
    empty_blob.embed_classic(empty_area);
    empty_blob.set_title_format(2);
    const auto parsed_empty = parse_nacp(empty_blob.bytes());
    // The heuristic sends an empty entry 0 name to the compressed path, and
    // the classic bytes are not a stream: the inherited limitation.
    CHECK(!parsed_empty.ok());
    CHECK(parsed_empty.error == NacpError::inflate_failed);

    TitleArea compressed_area(inflated_title_entry_count);
    compressed_area.set_name(Language::american_english, "Alpha");
    const std::vector<std::uint8_t> padded =
        pad_stream(deflate_raw(compressed_area.bytes()));
    ControlDataBlob compressed_blob;
    compressed_blob.embed_stream(padded, padded.size());
    compressed_blob.set_title_format(2);
    const auto parsed_compressed = parse_nacp(compressed_blob.bytes());
    CHECK(parsed_compressed.ok());
    CHECK(parsed_compressed.nacp.entry_count == inflated_title_entry_count);
    CHECK(parsed_compressed.nacp.entries[0].name == "Alpha");
    CHECK(parsed_compressed.warnings.size() == 1);
    CHECK(parsed_compressed.warnings.front().find("compressed") != std::string::npos);
}

void language_and_icon_file_names_agree() {
    CHECK(language_name(Language::american_english) == "AmericanEnglish");
    CHECK(language_name(Language::latin_american_spanish) == "LatinAmericanSpanish");
    CHECK(language_name(Language::simplified_chinese) == "SimplifiedChinese");
    CHECK(language_name(Language::brazilian_portuguese) == "BrazilianPortuguese");
    CHECK(icon_file_name(Language::american_english) == "icon_AmericanEnglish.dat");
    CHECK(icon_file_name(Language::japanese) == "icon_Japanese.dat");
    CHECK(icon_file_name(Language::latin_american_spanish) == "icon_LatinAmericanSpanish.dat");
    CHECK(icon_file_name(Language::simplified_chinese) == "icon_SimplifiedChinese.dat");

    for (std::size_t index = 0; index < language_count; ++index) {
        const auto language = static_cast<Language>(index);
        CHECK(language_name(language) == language_names[index]);
        CHECK(icon_file_name(language) ==
              "icon_" + std::string(language_names[index]) + ".dat");
    }

    const auto outside = static_cast<Language>(language_count);
    CHECK(language_name(outside).empty());
    CHECK(icon_file_name(outside).empty());
}

void icon_choice_uses_the_chosen_language_file() {
    const std::string_view available[] = {"icon_Japanese.dat", "icon_French.dat"};
    const auto choice = choose_icon(available, Language::japanese);
    CHECK(choice.ok());
    CHECK(choice.language == Language::japanese);
    CHECK(choice.file_name == "icon_Japanese.dat");
}

void icon_choice_falls_back_to_american_english() {
    const std::string_view available[] = {"icon_French.dat", "icon_AmericanEnglish.dat"};
    const auto choice = choose_icon(available, Language::korean);
    CHECK(choice.ok());
    CHECK(choice.language == Language::american_english);
    CHECK(choice.file_name == "icon_AmericanEnglish.dat");
}

void icon_choice_takes_first_available_in_entry_order() {
    const std::string_view available[] = {"icon_Korean.dat", "icon_Japanese.dat"};
    const auto choice = choose_icon(available, Language::french);
    CHECK(choice.ok());
    CHECK(choice.language == Language::japanese);
    CHECK(choice.file_name == "icon_Japanese.dat");
}

void icon_choice_without_a_match_fails() {
    const auto empty =
        choose_icon(std::span<const std::string_view>{}, Language::japanese);
    CHECK(!empty.ok());
    CHECK(empty.error == armigo::pack::nacp::IconChoiceError::no_icon_available);
    CHECK(!empty.detail.empty());
    CHECK(to_string(empty.error) == std::string_view("no_icon_available"));

    const std::string_view unrelated[] = {"control.nacp", "icon_Japanese.bak"};
    const auto no_match = choose_icon(unrelated, Language::japanese);
    CHECK(!no_match.ok());
    CHECK(no_match.error == armigo::pack::nacp::IconChoiceError::no_icon_available);
    CHECK(!no_match.detail.empty());
}

void app_name_handles_colon_forms() {
    const struct {
        std::string_view input;
        std::string_view expected;
    } cases[] = {
        {"Sample Chronicle: Open Country",
         "Sample Chronicle - Open Country"},
        {"Project: M", "Project - M"},
        {"A:B", "A-B"},
        {"A::B", "A--B"},
        {"A : B", "A  - B"},
        {"End:", "End-"},
        // The one space of ": " is consumed by the replacement; the second
        // space is untouched now that runs are not collapsed.
        {"a:  b", "a -  b"},
    };
    for (const auto& entry : cases) {
        const AppFileNameResult result = app_file_name(entry.input);
        CHECK(result.ok());
        CHECK(result.name == entry.expected);
    }
}

void app_name_handles_slashes_and_mixed_separators() {
    CHECK(app_file_name("Alpha/Beta").name == "Alpha-Beta");
    CHECK(app_file_name("a/b:c").name == "a-b-c");
    CHECK(app_file_name(" / ").name == "-");
}

void app_name_removes_control_characters() {
    CHECK(app_file_name("Sam\x01ple").name == "Sample");
    CHECK(app_file_name("A\x7F" "B").name == "AB");
    CHECK(app_file_name("A\xC2\x85" "B").name == "AB");
    CHECK(app_file_name("A\nB").name == "AB");
    CHECK(app_file_name("X \t Y").name == "X  Y");
    CHECK(app_file_name("A\xC2\x80" "z").name == "Az");
}

void app_name_trims_without_collapsing() {
    CHECK(app_file_name("  spaced  out  ").name == "spaced  out");
    CHECK(app_file_name("...Hidden").name == "Hidden");
    CHECK(app_file_name(" . name").name == "name");
    CHECK(app_file_name("name .").name == "name .");
    CHECK(app_file_name("name.").name == "name.");
    CHECK(app_file_name("...  ...dot").name == "dot");
}

void app_name_rejects_empty_results() {
    const std::string_view empty_cases[] = {"", "   ", "...", "\x01\x02", " . . "};
    for (const std::string_view input : empty_cases) {
        const AppFileNameResult result = app_file_name(input);
        CHECK(!result.ok());
        CHECK(result.error == AppNameError::empty_result);
        CHECK(!result.detail.empty());
        CHECK(result.name.empty());
    }
    CHECK(to_string(AppNameError::empty_result) == std::string_view("empty_result"));
}

void app_name_rejects_invalid_utf8() {
    const AppFileNameResult bad_byte = app_file_name("broken\xFFname");
    CHECK(!bad_byte.ok());
    CHECK(bad_byte.error == AppNameError::invalid_utf8);
    CHECK(!bad_byte.detail.empty());
    CHECK(to_string(AppNameError::invalid_utf8) == std::string_view("invalid_utf8"));

    const AppFileNameResult truncated = app_file_name("cut\xE2\x82");
    CHECK(!truncated.ok());
    CHECK(truncated.error == AppNameError::invalid_utf8);
}

void app_name_cuts_multi_byte_names_on_character_boundaries() {
    const std::string euro = "\xE2\x82\xAC";
    const std::string tree = "\xF0\x9F\x8C\xB3";

    CHECK(app_file_name("A\xC3\xA9" "B").name == "A\xC3\xA9" "B");

    const AppFileNameResult long_euro = app_file_name(repeated(euro, 121));
    CHECK(long_euro.ok());
    CHECK(long_euro.name.size() == max_app_file_name_bytes);
    CHECK(long_euro.name == repeated(euro, 80));

    const AppFileNameResult ascii_tail = app_file_name(repeated(euro, 80) + "AB");
    CHECK(ascii_tail.ok());
    CHECK(ascii_tail.name.size() == max_app_file_name_bytes);
    CHECK(ascii_tail.name == repeated(euro, 80));

    const AppFileNameResult four_byte_tail = app_file_name(repeated(euro, 80) + tree);
    CHECK(four_byte_tail.ok());
    CHECK(four_byte_tail.name.size() == max_app_file_name_bytes);
    CHECK(four_byte_tail.name == repeated(euro, 80));

    const AppFileNameResult within_limit = app_file_name(repeated(euro, 79) + "AB");
    CHECK(within_limit.ok());
    CHECK(within_limit.name == repeated(euro, 79) + "AB");
    CHECK(within_limit.name.size() == 239);

    const AppFileNameResult four_byte_blocks = app_file_name(repeated(tree, 61));
    CHECK(four_byte_blocks.ok());
    CHECK(four_byte_blocks.name.size() == max_app_file_name_bytes);
    CHECK(four_byte_blocks.name == repeated(tree, 60));

    const std::string ascii_max(max_app_file_name_bytes, 'x');
    CHECK(app_file_name(ascii_max).name == ascii_max);
    const std::string ascii_over(max_app_file_name_bytes + 1, 'x');
    const AppFileNameResult ascii_cut = app_file_name(ascii_over);
    CHECK(ascii_cut.ok());
    CHECK(ascii_cut.name.size() == max_app_file_name_bytes);
}

} // namespace

int main() {
    layout_constants_pin_the_control_format();
    parses_the_classic_title_table();
    american_english_title_wins_when_present();
    british_english_title_used_when_american_absent();
    simplified_chinese_preferred_over_traditional();
    korean_and_japanese_pick_first_in_entry_order();
    choose_title_fails_when_every_entry_is_empty();
    parse_rejects_short_buffers();
    parse_accepts_exact_and_longer_buffers();
    full_width_fields_are_read_with_a_warning();
    unreadable_american_english_name_is_empty_with_a_warning();
    unreadable_fields_become_empty_with_a_warning();
    max_length_name_is_parsed();
    compressed_title_area_round_trips();
    compressed_title_area_ignores_extra_entries_in_the_name_rule();
    compressed_title_area_accepts_trailing_stream_bytes();
    compressed_title_area_rejects_a_truncated_stream();
    compressed_title_area_rejects_wrong_inflated_sizes();
    compressed_title_area_rejects_an_oversized_length_prefix();
    compressed_title_area_rejects_garbage_streams();
    small_compressed_stream_is_parsed_as_compressed();
    classic_table_with_empty_american_english_picks_the_next_name();
    unknown_format_field_falls_back_by_the_entry_zero_name();
    language_and_icon_file_names_agree();
    icon_choice_uses_the_chosen_language_file();
    icon_choice_falls_back_to_american_english();
    icon_choice_takes_first_available_in_entry_order();
    icon_choice_without_a_match_fails();
    app_name_handles_colon_forms();
    app_name_handles_slashes_and_mixed_separators();
    app_name_removes_control_characters();
    app_name_trims_without_collapsing();
    app_name_rejects_empty_results();
    app_name_rejects_invalid_utf8();
    app_name_cuts_multi_byte_names_on_character_boundaries();
    if (failures == 0) {
        std::printf("nacp_tests: all checks passed\n");
        return 0;
    }
    std::printf("nacp_tests: %u check(s) failed\n", failures);
    return 1;
}
