// ============================================================================
// Armigo Pack - Control Data Naming
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/pack/nacp.h>

#include <zlib.h>

#include <cstring>
#include <utility>

namespace armigo::pack::nacp {

namespace {

constexpr bool is_continuation(std::uint8_t byte) noexcept {
    return (byte & 0xC0u) == 0x80u;
}

// Decodes one code point starting at text[offset]. Returns the length of the
// well-formed UTF-8 sequence, or 0 when the bytes there are not well-formed
// (stray continuation, overlong form, surrogate, above U+10FFFF, or a
// sequence truncated at the end of text). On success stores the code point.
std::size_t decode_code_point(std::string_view text, std::size_t offset,
                              std::uint32_t& code_point) noexcept {
    const auto lead = static_cast<std::uint8_t>(text[offset]);
    if (lead < 0x80u) {
        code_point = lead;
        return 1;
    }
    if (lead < 0xC2u) {
        return 0;  // stray continuation byte or overlong two-byte lead
    }
    if (lead < 0xE0u) {
        if (offset + 1 >= text.size() || !is_continuation(static_cast<std::uint8_t>(text[offset + 1]))) {
            return 0;
        }
        code_point = (static_cast<std::uint32_t>(lead & 0x1Fu) << 6) |
                     static_cast<std::uint32_t>(static_cast<std::uint8_t>(text[offset + 1]) & 0x3Fu);
        return 2;
    }
    if (lead < 0xF0u) {
        if (offset + 2 >= text.size()) {
            return 0;
        }
        const auto first = static_cast<std::uint8_t>(text[offset + 1]);
        const auto second = static_cast<std::uint8_t>(text[offset + 2]);
        if (!is_continuation(first) || !is_continuation(second)) {
            return 0;
        }
        if (lead == 0xE0u && first < 0xA0u) {
            return 0;  // overlong three-byte form
        }
        if (lead == 0xEDu && first > 0x9Fu) {
            return 0;  // surrogate code point
        }
        code_point = (static_cast<std::uint32_t>(lead & 0x0Fu) << 12) |
                     (static_cast<std::uint32_t>(first & 0x3Fu) << 6) |
                     static_cast<std::uint32_t>(second & 0x3Fu);
        return 3;
    }
    if (lead < 0xF5u) {
        if (offset + 3 >= text.size()) {
            return 0;
        }
        const auto first = static_cast<std::uint8_t>(text[offset + 1]);
        const auto second = static_cast<std::uint8_t>(text[offset + 2]);
        const auto third = static_cast<std::uint8_t>(text[offset + 3]);
        if (!is_continuation(first) || !is_continuation(second) || !is_continuation(third)) {
            return 0;
        }
        if (lead == 0xF0u && first < 0x90u) {
            return 0;  // overlong four-byte form
        }
        if (lead == 0xF4u && first > 0x8Fu) {
            return 0;  // above U+10FFFF
        }
        code_point = (static_cast<std::uint32_t>(lead & 0x07u) << 18) |
                     (static_cast<std::uint32_t>(first & 0x3Fu) << 12) |
                     (static_cast<std::uint32_t>(second & 0x3Fu) << 6) |
                     static_cast<std::uint32_t>(third & 0x3Fu);
        return 4;
    }
    return 0;
}

bool is_valid_utf8(std::string_view text) noexcept {
    std::uint32_t code_point = 0;
    for (std::size_t offset = 0; offset < text.size();) {
        const std::size_t length = decode_code_point(text, offset, code_point);
        if (length == 0) {
            return false;
        }
        offset += length;
    }
    return true;
}

bool is_control_code_point(std::uint32_t code_point) noexcept {
    return code_point < 0x20u || (code_point >= 0x7Fu && code_point <= 0x9Fu);
}

std::string_view field_text(const std::uint8_t* bytes, std::size_t length) noexcept {
    return std::string_view(reinterpret_cast<const char*>(bytes), length);
}

// The text before the field's first NUL, or the whole field when it fills
// its width without one.
struct FieldSlice {
    const std::uint8_t* text = nullptr;
    std::size_t length = 0;
    bool terminated = false;
};

FieldSlice slice_field(const std::uint8_t* bytes, std::size_t field_size) noexcept {
    const void* terminator = std::memchr(bytes, 0, field_size);
    if (terminator == nullptr) {
        return {bytes, field_size, false};
    }
    const auto* end = static_cast<const std::uint8_t*>(terminator);
    return {bytes, static_cast<std::size_t>(end - bytes), true};
}

std::string entry_field_description(std::size_t index, std::string_view field) {
    std::string description = "title entry " + std::to_string(index);
    if (index < language_count) {
        description += " (";
        description += language_names[index];
        description += ')';
    }
    description += ' ';
    description += field;
    return description;
}

// Reads one string field. A field that is not valid UTF-8 becomes empty and
// a full-width field without a NUL terminator is read whole; both record a
// warning instead of failing the parse.
std::string read_field(const std::uint8_t* bytes, std::size_t field_size,
                       std::string_view description, std::vector<std::string>& warnings) {
    const FieldSlice slice = slice_field(bytes, field_size);
    const std::string_view text = field_text(slice.text, slice.length);
    if (!is_valid_utf8(text)) {
        warnings.push_back(std::string(description) + " is not valid UTF-8; treated as empty");
        return {};
    }
    if (!slice.terminated) {
        warnings.push_back(std::string(description) + " has no NUL terminator; read at full width");
    }
    return std::string(text);
}

void parse_title_entries(const std::uint8_t* area, std::size_t entry_count, Nacp& nacp,
                         std::vector<std::string>& warnings) {
    for (std::size_t index = 0; index < entry_count; ++index) {
        const std::uint8_t* entry = area + index * title_entry_size;
        nacp.entries[index].name =
            read_field(entry, name_field_size, entry_field_description(index, "name field"),
                       warnings);
        nacp.entries[index].publisher =
            read_field(entry + name_field_size, publisher_field_size,
                       entry_field_description(index, "publisher field"), warnings);
    }
}

// The owner's detection rule, kept as the fallback for a format field this
// library cannot interpret: when entry 0's name is a non-empty, valid UTF-8
// title the title area is the classic uncompressed table; anything else is
// a compressed one.
bool has_uncompressed_title_table(const std::uint8_t* title_area) noexcept {
    const FieldSlice slice = slice_field(title_area, name_field_size);
    return slice.length != 0 && is_valid_utf8(field_text(slice.text, slice.length));
}

// The title data format field is the authority: 0 selects the classic
// table, 1 the compressed area. Any other value falls back to the name
// heuristic.
enum class TitleAreaFormat {
    classic,
    compressed,
};

TitleAreaFormat decide_title_area_format(const std::uint8_t* control,
                                         std::vector<std::string>& warnings) {
    const unsigned field = control[title_format_offset];
    if (field == 0) {
        return TitleAreaFormat::classic;
    }
    if (field == 1) {
        return TitleAreaFormat::compressed;
    }
    const TitleAreaFormat fallback =
        has_uncompressed_title_table(control) ? TitleAreaFormat::classic
                                              : TitleAreaFormat::compressed;
    warnings.push_back(
        "title data format field at offset 0x3215 holds " + std::to_string(field) +
        ", which is not a known format; detected as " +
        (fallback == TitleAreaFormat::classic ? "classic" : "compressed") +
        " by entry 0's name");
    return fallback;
}

struct InflatedTitleArea {
    bool ok = false;
    std::string detail;
    std::vector<std::uint8_t> bytes;
};

// Inflates a raw DEFLATE stream and requires exactly
// inflated_title_entry_count entries out, mirroring the owner's
// zlib.decompress(stream, -15): a truncated stream fails, trailing bytes
// after the stream's end do not.
InflatedTitleArea inflate_title_area(const std::uint8_t* input, std::size_t input_size) {
    constexpr std::size_t inflated_size = inflated_title_entry_count * title_entry_size;
    InflatedTitleArea result;
    result.bytes.assign(inflated_size, 0);

    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
    stream.avail_in = static_cast<uInt>(input_size);
    stream.next_out = result.bytes.data();
    stream.avail_out = static_cast<uInt>(result.bytes.size());
    if (inflateInit2(&stream, -15) != Z_OK) {
        result.detail = "zlib inflate initialization failed";
        return result;
    }
    const int status = inflate(&stream, Z_FINISH);
    const std::size_t produced = result.bytes.size() - stream.avail_out;
    inflateEnd(&stream);

    if (status == Z_DATA_ERROR) {
        result.detail = "the compressed title area is not valid DEFLATE data";
        return result;
    }
    if (status != Z_STREAM_END && produced == inflated_size) {
        result.detail = "the compressed title area inflates past " +
                        std::to_string(inflated_size) + " bytes";
        return result;
    }
    if (status != Z_STREAM_END) {
        result.detail = "the compressed title area is an incomplete DEFLATE stream";
        return result;
    }
    if (produced != inflated_size) {
        result.detail = "the compressed title area inflates to " + std::to_string(produced) +
                        " bytes, expected " + std::to_string(inflated_size);
        return result;
    }
    result.ok = true;
    return result;
}

// The icon file names, derived once from the language names at compile time.
constexpr std::size_t icon_file_name_capacity = 32;

struct IconFileNameTable {
    std::array<std::array<char, icon_file_name_capacity>, language_count> storage{};
    std::array<std::string_view, language_count> names{};

    constexpr IconFileNameTable() {
        for (std::size_t index = 0; index < language_count; ++index) {
            char* buffer = storage[index].data();
            std::size_t length = 0;
            for (const char character : std::string_view("icon_")) {
                buffer[length++] = character;
            }
            for (const char character : language_names[index]) {
                buffer[length++] = character;
            }
            for (const char character : std::string_view(".dat")) {
                buffer[length++] = character;
            }
            buffer[length] = '\0';
            names[index] = std::string_view(buffer, length);
        }
    }
};

constexpr IconFileNameTable icon_file_names{};

TitleChoice make_title_choice(Language language, std::string_view name) {
    TitleChoice result;
    result.language = language;
    result.name = name;
    return result;
}

IconChoice make_icon_choice(Language language) {
    IconChoice result;
    result.language = language;
    result.file_name = icon_file_name(language);
    return result;
}

void trim_file_system_edges(std::string& text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '.')) {
        text.erase(text.begin());
    }
    while (!text.empty() && text.back() == ' ') {
        text.pop_back();
    }
}

void truncate_utf8(std::string& text, std::size_t limit) {
    if (text.size() <= limit) {
        return;
    }
    std::size_t cut = limit;
    while (cut > 0 && is_continuation(static_cast<std::uint8_t>(text[cut]))) {
        --cut;
    }
    text.resize(cut);
}

} // namespace

const char* to_string(NacpError error) noexcept {
    switch (error) {
    case NacpError::none:
        return "none";
    case NacpError::too_short:
        return "too_short";
    case NacpError::inflate_failed:
        return "inflate_failed";
    }
    return "unknown";
}

NacpParseResult parse_nacp(std::span<const std::uint8_t> bytes) {
    NacpParseResult result;
    if (bytes.size() < control_file_size) {
        result.error = NacpError::too_short;
        result.detail = "control data must be at least " + std::to_string(control_file_size) +
                        " bytes; got " + std::to_string(bytes.size());
        return result;
    }

    if (decide_title_area_format(bytes.data(), result.warnings) ==
        TitleAreaFormat::classic) {
        parse_title_entries(bytes.data(), title_entry_count, result.nacp, result.warnings);
        result.nacp.entry_count = title_entry_count;
    } else {
        const std::size_t compressed_size = static_cast<std::size_t>(bytes[0]) |
                                            (static_cast<std::size_t>(bytes[1]) << 8);
        if (compressed_size + 2 > title_area_size) {
            result.error = NacpError::inflate_failed;
            result.detail = "the compressed title area length " + std::to_string(compressed_size) +
                            " exceeds the " + std::to_string(title_area_size) + "-byte title area";
            return result;
        }
        InflatedTitleArea inflated = inflate_title_area(bytes.data() + 2, compressed_size);
        if (!inflated.ok) {
            result.error = NacpError::inflate_failed;
            result.detail = std::move(inflated.detail);
            return result;
        }
        parse_title_entries(inflated.bytes.data(), inflated_title_entry_count, result.nacp,
                            result.warnings);
        result.nacp.entry_count = inflated_title_entry_count;
    }

    result.nacp.supported_language_flag =
        static_cast<std::uint32_t>(bytes[supported_language_flag_offset]) |
        (static_cast<std::uint32_t>(bytes[supported_language_flag_offset + 1]) << 8) |
        (static_cast<std::uint32_t>(bytes[supported_language_flag_offset + 2]) << 16) |
        (static_cast<std::uint32_t>(bytes[supported_language_flag_offset + 3]) << 24);
    result.nacp.display_version =
        read_field(bytes.data() + display_version_offset, display_version_size,
                   "display version field", result.warnings);
    return result;
}

std::string_view language_name(Language language) noexcept {
    const auto index = static_cast<std::size_t>(language);
    if (index >= language_count) {
        return {};
    }
    return language_names[index];
}

std::string_view icon_file_name(Language language) noexcept {
    const auto index = static_cast<std::size_t>(language);
    if (index >= language_count) {
        return {};
    }
    return icon_file_names.names[index];
}

const char* to_string(TitleChoiceError error) noexcept {
    switch (error) {
    case TitleChoiceError::none:
        return "none";
    case TitleChoiceError::no_title_entry:
        return "no_title_entry";
    }
    return "unknown";
}

TitleChoice choose_title(const Nacp& nacp) {
    const Language preferred[] = {Language::american_english, Language::british_english,
                                  Language::simplified_chinese};
    for (const Language language : preferred) {
        const TitleEntry& entry = nacp.entries[static_cast<std::size_t>(language)];
        if (!entry.name.empty()) {
            return make_title_choice(language, entry.name);
        }
    }
    for (std::size_t index = 0; index < language_count; ++index) {
        if (!nacp.entries[index].name.empty()) {
            return make_title_choice(static_cast<Language>(index), nacp.entries[index].name);
        }
    }
    TitleChoice result;
    result.error = TitleChoiceError::no_title_entry;
    result.detail = "control data has no non-empty title entry";
    return result;
}

const char* to_string(IconChoiceError error) noexcept {
    switch (error) {
    case IconChoiceError::none:
        return "none";
    case IconChoiceError::no_icon_available:
        return "no_icon_available";
    }
    return "unknown";
}

IconChoice choose_icon(std::span<const std::string_view> available_file_names,
                       Language chosen_language) {
    const auto available = [available_file_names](std::string_view file_name) {
        for (const std::string_view candidate : available_file_names) {
            if (candidate == file_name) {
                return true;
            }
        }
        return false;
    };

    const Language preferred[] = {chosen_language, Language::american_english};
    for (const Language language : preferred) {
        if (available(icon_file_name(language))) {
            return make_icon_choice(language);
        }
    }
    for (std::size_t index = 0; index < language_count; ++index) {
        const Language language = static_cast<Language>(index);
        if (available(icon_file_name(language))) {
            return make_icon_choice(language);
        }
    }
    IconChoice result;
    result.error = IconChoiceError::no_icon_available;
    result.detail = "none of the control RomFS icon files is available";
    return result;
}

const char* to_string(AppNameError error) noexcept {
    switch (error) {
    case AppNameError::none:
        return "none";
    case AppNameError::invalid_utf8:
        return "invalid_utf8";
    case AppNameError::empty_result:
        return "empty_result";
    }
    return "unknown";
}

AppFileNameResult app_file_name(std::string_view title) {
    if (!is_valid_utf8(title)) {
        AppFileNameResult result;
        result.error = AppNameError::invalid_utf8;
        result.detail = "title is not valid UTF-8";
        return result;
    }

    std::string sanitized;
    sanitized.reserve(title.size());
    std::size_t offset = 0;
    while (offset < title.size()) {
        std::uint32_t code_point = 0;
        const std::size_t length = decode_code_point(title, offset, code_point);
        if (is_control_code_point(code_point)) {
            offset += length;
            continue;
        }
        if (code_point == ':' && offset + 1 < title.size() &&
            title[offset + 1] == ' ') {
            sanitized += " - ";
            offset += 2;
            continue;
        }
        if (code_point == ':' || code_point == '/') {
            sanitized += '-';
            offset += length;
            continue;
        }
        sanitized.append(title.substr(offset, length));
        offset += length;
    }

    trim_file_system_edges(sanitized);
    truncate_utf8(sanitized, max_app_file_name_bytes);

    if (sanitized.empty()) {
        AppFileNameResult result;
        result.error = AppNameError::empty_result;
        result.detail = "title produces an empty file name";
        return result;
    }
    AppFileNameResult result;
    result.name = std::move(sanitized);
    return result;
}

} // namespace armigo::pack::nacp
