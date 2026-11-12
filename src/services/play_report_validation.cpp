// ============================================================================
// Armigo Services - Offline Play Report Validation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "play_report_validation.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace armigo::services::prepo::detail {
namespace {

// The 23.0.1 report check (the same on both platforms): one MessagePack map of
// 1 to 100 pairs filling the whole report; string keys of 1 to 63 bytes drawn
// from [0-9A-Za-z_:], unique ignoring ASCII case; and each value an integer, an
// unsigned integer, a 64-bit float, a string, or an extension of type 0 with
// exactly eight data bytes. Strings must be valid UTF-8.
constexpr std::size_t maximum_report_size = 65'536;
constexpr std::size_t maximum_pairs = 100;
constexpr std::size_t maximum_key_size = 63;

unsigned byte_value(std::byte byte) {
    return std::to_integer<unsigned char>(byte);
}

unsigned lower_ascii(std::byte byte) {
    const auto value = byte_value(byte);
    return value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value;
}

bool valid_utf8(std::span<const std::byte> text) {
    std::size_t index = 0;
    const auto continuation = [&](std::size_t offset) {
        return offset < text.size() && (byte_value(text[offset]) & 0xc0) == 0x80;
    };
    while (index < text.size()) {
        const auto first = byte_value(text[index]);
        if (first <= 0x7f) {
            ++index;
            continue;
        }
        if (first >= 0xc2 && first <= 0xdf) {
            if (!continuation(index + 1)) return false;
            index += 2;
            continue;
        }
        if (first >= 0xe0 && first <= 0xef) {
            if (index + 2 >= text.size()) return false;
            const auto second = byte_value(text[index + 1]);
            if ((first == 0xe0 && (second < 0xa0 || second > 0xbf)) ||
                (first == 0xed && (second < 0x80 || second > 0x9f)) ||
                (first != 0xe0 && first != 0xed && (second & 0xc0) != 0x80) ||
                !continuation(index + 2))
                return false;
            index += 3;
            continue;
        }
        if (first >= 0xf0 && first <= 0xf4) {
            if (index + 3 >= text.size()) return false;
            const auto second = byte_value(text[index + 1]);
            if ((first == 0xf0 && (second < 0x90 || second > 0xbf)) ||
                (first == 0xf4 && (second < 0x80 || second > 0x8f)) ||
                (first != 0xf0 && first != 0xf4 && (second & 0xc0) != 0x80) ||
                !continuation(index + 2) || !continuation(index + 3))
                return false;
            index += 4;
            continue;
        }
        return false;
    }
    return true;
}


// The kinds of MessagePack element the check distinguishes.
enum class Kind { nil, boolean, binary, extension, float32, float64, unsigned_integer,
                  integer, string, array, map };

struct Element {
    Kind kind{};
    std::uint64_t count{};               // Pairs or elements of a container.
    std::span<const std::byte> data{};   // String, binary or extension payload.
    int extension_type{};
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
    bool finished() const { return offset_ == bytes_.size(); }

    // Reads one element header and skips its payload; containers report their
    // count and leave their contents unread.
    std::optional<Element> next() {
        const auto lead = take(1);
        if (!lead) return std::nullopt;
        const auto first = *lead;
        Element element;
        if (first <= 0x7f || first >= 0xe0) { element.kind = Kind::integer; return element; }
        if ((first & 0xf0) == 0x80) { element.kind = Kind::map; element.count = first & 0x0f; return element; }
        if ((first & 0xf0) == 0x90) { element.kind = Kind::array; element.count = first & 0x0f; return element; }
        if ((first & 0xe0) == 0xa0) return string(first & 0x1f);
        switch (first) {
        case 0xc0: element.kind = Kind::nil; return element;
        case 0xc2: case 0xc3: element.kind = Kind::boolean; return element;
        case 0xc4: case 0xc5: case 0xc6: {
            const auto size = take(first == 0xc4 ? 1 : first == 0xc5 ? 2 : 4);
            if (!size) return std::nullopt;
            return payload(Kind::binary, *size);
        }
        case 0xc7: case 0xc8: case 0xc9: {
            const auto size = take(first == 0xc7 ? 1 : first == 0xc8 ? 2 : 4);
            if (!size) return std::nullopt;
            return extension(*size);
        }
        case 0xca: element.kind = Kind::float32; return skip(4) ? std::optional{element} : std::nullopt;
        case 0xcb: element.kind = Kind::float64; return skip(8) ? std::optional{element} : std::nullopt;
        case 0xcc: case 0xcd: case 0xce: case 0xcf:
            element.kind = Kind::unsigned_integer;
            return skip(std::size_t{1} << (first - 0xcc)) ? std::optional{element} : std::nullopt;
        case 0xd0: case 0xd1: case 0xd2: case 0xd3:
            element.kind = Kind::integer;
            return skip(std::size_t{1} << (first - 0xd0)) ? std::optional{element} : std::nullopt;
        case 0xd4: case 0xd5: case 0xd6: case 0xd7: case 0xd8:
            return extension(std::uint64_t{1} << (first - 0xd4));
        case 0xd9: case 0xda: case 0xdb: {
            const auto size = take(first == 0xd9 ? 1 : first == 0xda ? 2 : 4);
            if (!size) return std::nullopt;
            return string(*size);
        }
        case 0xdc: case 0xdd: case 0xde: case 0xdf: {
            const auto count = take(first == 0xdc || first == 0xde ? 2 : 4);
            if (!count) return std::nullopt;
            element.kind = first <= 0xdd ? Kind::array : Kind::map;
            element.count = *count;
            return element;
        }
        default: return std::nullopt;   // 0xc1 is never used.
        }
    }

private:
    // Reads a big-endian value of `width` bytes.
    std::optional<std::uint64_t> take(std::size_t width) {
        if (bytes_.size() - offset_ < width) return std::nullopt;
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < width; ++index)
            value = value << 8 | byte_value(bytes_[offset_ + index]);
        offset_ += width;
        return value;
    }
    bool skip(std::uint64_t size) {
        if (bytes_.size() - offset_ < size) return false;
        offset_ += static_cast<std::size_t>(size);
        return true;
    }
    std::optional<Element> payload(Kind kind, std::uint64_t size) {
        if (bytes_.size() - offset_ < size) return std::nullopt;
        Element element;
        element.kind = kind;
        element.data = bytes_.subspan(offset_, static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
        return element;
    }
    std::optional<Element> string(std::uint64_t size) {
        auto element = payload(Kind::string, size);
        if (element && !valid_utf8(element->data)) return std::nullopt;
        return element;
    }
    std::optional<Element> extension(std::uint64_t size) {
        const auto type = take(1);
        if (!type) return std::nullopt;
        auto element = payload(Kind::extension, size);
        if (element) element->extension_type = static_cast<std::int8_t>(*type);
        return element;
    }

    std::span<const std::byte> bytes_;
    std::size_t offset_ = 0;
};

bool valid_key(std::span<const std::byte> key) {
    if (key.empty() || key.size() > maximum_key_size) return false;
    for (const auto byte : key) {
        const auto value = byte_value(byte);
        const bool digit = value >= '0' && value <= '9';
        const bool letter = (value | 0x20) >= 'a' && (value | 0x20) <= 'z';
        if (!digit && !letter && value != '_' && value != ':') return false;
    }
    return true;
}

bool same_key_ignoring_case(std::span<const std::byte> left, std::span<const std::byte> right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index)
        if (lower_ascii(left[index]) != lower_ascii(right[index])) return false;
    return true;
}

bool valid_value(const Element& value) {
    switch (value.kind) {
    case Kind::float64: case Kind::unsigned_integer: case Kind::integer: case Kind::string:
        return true;
    case Kind::extension:
        return value.extension_type == 0 && value.data.size() == 8;
    default:
        return false;
    }
}

} // namespace

bool valid_report(std::span<const std::byte> report) {
    if (report.size() > maximum_report_size) return false;
    Reader reader(report);
    const auto root = reader.next();
    if (!root || root->kind != Kind::map || root->count == 0 || root->count > maximum_pairs)
        return false;
    std::vector<std::span<const std::byte>> keys;
    keys.reserve(static_cast<std::size_t>(root->count));
    for (std::uint64_t pair = 0; pair < root->count; ++pair) {
        const auto key = reader.next();
        if (!key || key->kind != Kind::string || !valid_key(key->data)) return false;
        keys.push_back(key->data);
        const auto value = reader.next();
        if (!value || !valid_value(*value)) return false;
    }
    if (!reader.finished()) return false;
    for (std::size_t left = 0; left < keys.size(); ++left)
        for (std::size_t right = left + 1; right < keys.size(); ++right)
            if (same_key_ignoring_case(keys[left], keys[right])) return false;
    return true;
}

} // namespace armigo::services::prepo::detail
