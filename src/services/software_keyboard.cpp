// ============================================================================
// Armigo Services - Software Keyboard Library Applet
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/software_keyboard.h>

#include <algorithm>

namespace armigo::services::am {
namespace {
// The applet's own input limit when the configuration sets none.
constexpr std::uint32_t default_max_length = 500;

// Configuration layout shared by every library version (the common part).
constexpr std::size_t config_type = 0x0;
constexpr std::size_t config_header = 0x24;
constexpr std::size_t config_sub = 0xa6;
constexpr std::size_t config_guide = 0x1a8;
constexpr std::size_t header_units = 65;
constexpr std::size_t sub_units = 129;
constexpr std::size_t guide_units = 257;
constexpr std::size_t config_max_length = 0x3ac;
constexpr std::size_t config_min_length = 0x3b0;
constexpr std::size_t config_password = 0x3b4;
constexpr std::size_t config_utf8 = 0x3bd;
constexpr std::size_t config_initial_offset = 0x3c0;
constexpr std::size_t config_initial_length = 0x3c4;
constexpr std::size_t config_text_check = 0x3d0;

// Library versions 6.0.0 and later send a 0x4c8-byte configuration; the
// common part is followed by text grouping and customized-dictionary fields
// the applet does not present.
struct LibraryVersion {
    std::uint32_t version;
    std::size_t config_size;
};
constexpr LibraryVersion supported_versions[] = {{0x6000b, 0x4c8}, {0x8000d, 0x4c8}};

std::uint32_t read32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + i])) << (i * 8);
    return value;
}

// A NUL-terminated UTF-16 field of at most `units` code units.
std::u16string read_text(std::span<const std::byte> bytes, std::size_t offset, std::size_t units) {
    std::u16string text;
    for (std::size_t index = 0; index < units; ++index) {
        const auto unit = static_cast<char16_t>(
            std::to_integer<unsigned>(bytes[offset + 2 * index]) |
            std::to_integer<unsigned>(bytes[offset + 2 * index + 1]) << 8);
        if (unit == 0) break;
        text.push_back(unit);
    }
    return text;
}

// UTF-16 to UTF-8; an unpaired surrogate becomes U+FFFD.
std::string to_utf8(const std::u16string& text) {
    std::string out;
    for (std::size_t index = 0; index < text.size(); ++index) {
        std::uint32_t code = text[index];
        if (code >= 0xd800 && code < 0xdc00 && index + 1 < text.size() &&
            text[index + 1] >= 0xdc00 && text[index + 1] < 0xe000) {
            code = 0x10000 + ((code - 0xd800) << 10) + (text[++index] - 0xdc00);
        } else if (code >= 0xd800 && code < 0xe000) {
            code = 0xfffd;
        }
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xc0 | code >> 6));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xe0 | code >> 12));
            out.push_back(static_cast<char>(0x80 | (code >> 6 & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | code >> 18));
            out.push_back(static_cast<char>(0x80 | (code >> 12 & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (code >> 6 & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        }
    }
    return out;
}
} // namespace

const char* parse_keyboard_request(std::span<const std::byte> common_arguments,
                                   std::span<const std::byte> config, KeyboardRequest& request) {
    // Common arguments: u32 version, u32 size, u32 library version, ...
    if (common_arguments.size() < keyboard_common_arguments_size ||
        read32(common_arguments, 4) != keyboard_common_arguments_size)
        return "Software keyboard common arguments are malformed";
    const auto version = read32(common_arguments, 8);
    const auto known = std::find_if(std::begin(supported_versions), std::end(supported_versions),
                                    [&](const LibraryVersion& entry) { return entry.version == version; });
    if (known == std::end(supported_versions))
        return "Software keyboard library version is not supported";
    if (config.size() != known->config_size)
        return "Software keyboard configuration size does not match its library version";
    KeyboardRequest parsed;
    parsed.library_version = version;
    parsed.type = read32(config, config_type);
    parsed.header = read_text(config, config_header, header_units);
    parsed.sub = read_text(config, config_sub, sub_units);
    parsed.guide = read_text(config, config_guide, guide_units);
    parsed.max_length = read32(config, config_max_length);
    parsed.min_length = read32(config, config_min_length);
    parsed.password = read32(config, config_password) != 0;
    parsed.utf8 = config[config_utf8] != std::byte{0};
    parsed.initial_offset = read32(config, config_initial_offset);
    parsed.initial_length = read32(config, config_initial_length);
    parsed.text_check = config[config_text_check] != std::byte{0};
    request = std::move(parsed);
    return nullptr;
}

std::u16string fit_keyboard_text(const KeyboardRequest& request, std::u16string text) {
    const std::size_t max_length = request.max_length ? request.max_length : default_max_length;
    if (!text.empty())
        while (text.size() < request.min_length) text = text + u" " + text;
    if (text.size() > max_length) text.resize(max_length);
    return text;
}

KeyboardResponse headless_keyboard_input(const KeyboardRequest& request) {
    return {true, fit_keyboard_text(request, request.initial.empty() ? u"Player" : request.initial)};
}

std::vector<std::byte> keyboard_output(const KeyboardRequest& request,
                                       const KeyboardResponse& response) {
    std::vector<std::byte> out(keyboard_output_size);
    out[0] = static_cast<std::byte>(response.accepted ? 0 : 1);
    if (!response.accepted) return out;
    // The text area keeps a terminating NUL.
    if (request.utf8) {
        auto text = to_utf8(response.text);
        auto length = std::min(text.size(), keyboard_string_size - 1);
        // Never end inside a multi-byte sequence.
        while (length < text.size() && length > 0 &&
               (static_cast<unsigned char>(text[length]) & 0xc0) == 0x80)
            --length;
        std::transform(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(length),
                       out.begin() + 4, [](char c) { return static_cast<std::byte>(c); });
    } else {
        const auto length = std::min(response.text.size(), keyboard_string_size / 2 - 1);
        for (std::size_t index = 0; index < length; ++index) {
            out[4 + 2 * index] = static_cast<std::byte>(response.text[index] & 0xff);
            out[5 + 2 * index] = static_cast<std::byte>(response.text[index] >> 8);
        }
    }
    return out;
}

} // namespace armigo::services::am
