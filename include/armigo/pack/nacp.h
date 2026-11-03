// ============================================================================
// Armigo Pack - Control Data Naming
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::pack::nacp {

// Pure functions over a game's control data (control.nacp) bytes: parse the
// title table, choose the title that names the packed app, pick the control
// RomFS icon file, and turn the title into a file-system-safe app name.
// There is no file system, key, or container access here; another pack
// component supplies the bytes.

// The languages of the title table's first sixteen entries, in entry order.
enum class Language : std::uint8_t {
    american_english = 0,
    british_english = 1,
    japanese = 2,
    french = 3,
    german = 4,
    latin_american_spanish = 5,
    spanish = 6,
    italian = 7,
    dutch = 8,
    canadian_french = 9,
    portuguese = 10,
    russian = 11,
    korean = 12,
    traditional_chinese = 13,
    simplified_chinese = 14,
    brazilian_portuguese = 15,
};

inline constexpr std::size_t language_count = 16;

// The language names as they appear in control RomFS icon file names, in
// entry order ("AmericanEnglish", ..., "BrazilianPortuguese").
inline constexpr std::array<std::string_view, language_count> language_names{
    "AmericanEnglish",     "BritishEnglish",      "Japanese",
    "French",              "German",              "LatinAmericanSpanish",
    "Spanish",             "Italian",             "Dutch",
    "CanadianFrench",      "Portuguese",          "Russian",
    "Korean",              "TraditionalChinese",  "SimplifiedChinese",
    "BrazilianPortuguese",
};

// Byte layout of the control data.
inline constexpr std::size_t control_file_size = 0x4000;
// The title table (classic) or the compressed title area it inflates from.
inline constexpr std::size_t title_area_size = 0x3000;
inline constexpr std::size_t title_entry_size = 0x300;
inline constexpr std::size_t name_field_size = 0x200;
inline constexpr std::size_t publisher_field_size = 0x100;
inline constexpr std::size_t supported_language_flag_offset = 0x302c;
inline constexpr std::size_t display_version_offset = 0x3060;
inline constexpr std::size_t display_version_size = 0x10;
// The title data format field: 0 selects the classic table, 1 the
// compressed title area. Older control data without the field reads 0.
inline constexpr std::size_t title_format_offset = 0x3215;
// The classic table stores sixteen entries; a compressed title area inflates
// to thirty-two. Entries past the named languages are extra entries the
// owner's reader leaves unnamed: their text is kept, but no language names
// them and no naming rule considers them.
inline constexpr std::size_t title_entry_count = 16;
inline constexpr std::size_t inflated_title_entry_count = 32;

// Longest app file name this component produces, in UTF-8 bytes.
inline constexpr std::size_t max_app_file_name_bytes = 240;

// One parsed title entry. `name` is empty when the game ships no title for
// the entry, or when the field was unreadable (see parse_nacp).
struct TitleEntry {
    std::string name;
    std::string publisher;
};

// The parsed control data. `entries` holds `entry_count` parsed entries in
// table order; entries 0..15 are the named languages above.
struct Nacp {
    std::array<TitleEntry, inflated_title_entry_count> entries;
    // 16 for the classic table, 32 after inflating a compressed one.
    std::size_t entry_count = title_entry_count;
    std::uint32_t supported_language_flag = 0;
    std::string display_version;
};

// Structured parse failures. A packer must not refuse a lawfully owned game
// over a cosmetic field, so field problems are warnings; only a buffer too
// short to be control data, or a title area that cannot be inflated, fails.
enum class NacpError : std::uint8_t {
    none,
    // The buffer holds fewer than control_file_size bytes.
    too_short,
    // The compressed title area is not a complete DEFLATE stream, its length
    // prefix exceeds the title area, or it inflates to a size other than
    // inflated_title_entry_count entries.
    inflate_failed,
};

const char* to_string(NacpError error) noexcept;

struct NacpParseResult {
    NacpError error = NacpError::none;
    std::string detail;
    // One human-readable line per unreadable or full-width field.
    std::vector<std::string> warnings;
    Nacp nacp;

    bool ok() const noexcept { return error == NacpError::none; }
};

// Parse the control data. The buffer must hold at least control_file_size
// bytes; trailing bytes beyond the fixed layout are ignored.
//
// The title data format field at title_format_offset decides the title
// area: 0 is the classic table (sixteen entries at offset 0), 1 is the
// compressed area, where offset 0 holds a little-endian u16 length followed
// by that many bytes of raw DEFLATE, which must inflate to exactly
// inflated_title_entry_count * title_entry_size bytes; the length plus its
// prefix must fit inside the title area. A field value other than 0 or 1 is
// not interpreted: the owner's name rule then decides instead - the area is
// classic when entry 0's name is a non-empty, valid UTF-8 title, and
// compressed otherwise - and the parse records a warning saying so. That
// fallback can misread an area whose entry 0 name is empty but whose bytes
// are not a stream, so the field is the authority wherever it is valid.
//
// Every string field (names, publishers, the display version) is read up to
// its first NUL or, when a field fills its whole width, at full width; text
// that is not valid UTF-8 becomes an empty string. Both cases record a
// warning; bytes after a terminator are padding and are not inspected.
NacpParseResult parse_nacp(std::span<const std::uint8_t> bytes);

// The language's name ("Japanese"), or an empty view for a value outside
// the named languages.
std::string_view language_name(Language language) noexcept;

// The control RomFS icon file name for a language ("icon_Japanese.dat"),
// or an empty view for a value outside the named languages.
std::string_view icon_file_name(Language language) noexcept;

// The title that names the packed app. The rule considers the sixteen named
// languages only: american_english when its name is non-empty, else
// british_english, else simplified_chinese, else the first non-empty entry
// in entry order. `name` points into the passed Nacp and stays valid while
// the Nacp is alive and unmodified.
enum class TitleChoiceError : std::uint8_t {
    none,
    // No named language carries a non-empty name.
    no_title_entry,
};

const char* to_string(TitleChoiceError error) noexcept;

struct TitleChoice {
    TitleChoiceError error = TitleChoiceError::none;
    std::string detail;
    Language language = Language::american_english;
    std::string_view name;

    bool ok() const noexcept { return error == TitleChoiceError::none; }
};

TitleChoice choose_title(const Nacp& nacp);

// The icon to show on the packed app. The rule: the chosen language's icon
// file when `available_file_names` contains it, else american_english's,
// else the first language in entry order whose file is available.
// `file_name` is the chosen icon's file name and is static storage.
enum class IconChoiceError : std::uint8_t {
    none,
    // No listed file matches any language's icon file name.
    no_icon_available,
};

const char* to_string(IconChoiceError error) noexcept;

struct IconChoice {
    IconChoiceError error = IconChoiceError::none;
    std::string detail;
    Language language = Language::american_english;
    std::string_view file_name;

    bool ok() const noexcept { return error == IconChoiceError::none; }
};

IconChoice choose_icon(std::span<const std::string_view> available_file_names,
                       Language chosen_language);

// Turn a title into a macOS/iOS-safe file name (the caller appends ".app").
// Applied in this order: ": " becomes " - ", any remaining ':' or '/' becomes
// '-', Unicode control characters (U+0000-U+001F, U+007F-U+009F) are removed,
// leading spaces and dots are trimmed until neither remains, trailing spaces
// are trimmed, and the result is cut to at most max_app_file_name_bytes
// UTF-8 bytes without splitting a character. Nothing else changes: runs of
// spaces stay as they are. The title must be valid UTF-8 (parse_nacp
// guarantees this for its names).
enum class AppNameError : std::uint8_t {
    none,
    invalid_utf8,
    empty_result,
};

const char* to_string(AppNameError error) noexcept;

struct AppFileNameResult {
    AppNameError error = AppNameError::none;
    std::string detail;
    std::string name;

    bool ok() const noexcept { return error == AppNameError::none; }
};

AppFileNameResult app_file_name(std::string_view title);

} // namespace armigo::pack::nacp
