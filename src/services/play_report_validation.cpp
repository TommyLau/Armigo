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

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace armigo::services::prepo::detail {
namespace {

constexpr std::size_t maximum_report_size = 65'536;
constexpr unsigned maximum_container_depth = 3;

enum class Category {
    boolean,
    integer,
    floating,
    string,
    binary,
    extension,
    array,
    map,
};

struct Shape {
    Category category{};
    std::size_t count{};
    std::span<const std::byte> key{};
    std::vector<Shape> children{};
};

enum class ValuePolicy {
    root,
    nested,
    array_element,
};

unsigned byte_value(std::byte byte) {
    return std::to_integer<unsigned char>(byte);
}

bool same_bytes(std::span<const std::byte> left, std::span<const std::byte> right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

unsigned lower_ascii(std::byte byte) {
    const auto value = byte_value(byte);
    return value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value;
}

bool key_less_case_insensitive(std::span<const std::byte> left,
                               std::span<const std::byte> right) {
    const auto common = std::min(left.size(), right.size());
    for (std::size_t index = 0; index < common; ++index) {
        const auto left_byte = lower_ascii(left[index]);
        const auto right_byte = lower_ascii(right[index]);
        if (left_byte != right_byte) return left_byte < right_byte;
    }
    return left.size() < right.size();
}

bool same_key_case_insensitive(std::span<const std::byte> left,
                               std::span<const std::byte> right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (lower_ascii(left[index]) != lower_ascii(right[index])) return false;
    }
    return true;
}

bool valid_identifier(std::span<const std::byte> key) {
    if (key.empty() || key.size() > 63) return false;
    const auto first = byte_value(key.front());
    if (!((first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z') || first == '_'))
        return false;
    for (const auto byte : key.subspan(1)) {
        const auto value = byte_value(byte);
        if (!((value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
              (value >= '0' && value <= '9') || value == '_'))
            return false;
    }
    return true;
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

bool same_shape(const Shape& left, const Shape& right) {
    if (left.category != right.category) return false;
    if (left.category != Category::array && left.category != Category::map) return true;
    if (left.count != right.count || left.children.size() != right.children.size()) return false;
    for (std::size_t index = 0; index < left.children.size(); ++index) {
        if (left.category == Category::map &&
            !same_bytes(left.children[index].key, right.children[index].key))
            return false;
        if (!same_shape(left.children[index], right.children[index])) return false;
    }
    return true;
}

class Parser {
public:
    explicit Parser(std::span<const std::byte> input) : input_(input) {}

    bool parse_root() {
        std::size_t count = 0;
        if (!take_map_header(count) || count == 0 || count > 100) return false;
        Shape root{};
        if (!parse_map(count, root, ValuePolicy::root, 1)) return false;
        return offset_ == input_.size();
    }

private:
    template <typename Integer>
    bool take_integer(Integer& value) {
        if (remaining() < sizeof(Integer)) return false;
        value = 0;
        for (std::size_t index = 0; index < sizeof(Integer); ++index)
            value = static_cast<Integer>((value << 8) | byte_value(input_[offset_++]));
        return true;
    }

    bool take_byte(unsigned& value) {
        if (remaining() == 0) return false;
        value = byte_value(input_[offset_++]);
        return true;
    }

    bool skip(std::size_t count) {
        if (count > remaining()) return false;
        offset_ += count;
        return true;
    }

    std::size_t remaining() const {
        return input_.size() - offset_;
    }

    bool take_map_header(std::size_t& count) {
        unsigned marker = 0;
        if (!take_byte(marker)) return false;
        if (marker >= 0x80 && marker <= 0x8f) {
            count = marker & 0x0f;
            return true;
        }
        if (marker == 0xde) {
            std::uint16_t value = 0;
            if (!take_integer(value)) return false;
            count = value;
            return true;
        }
        if (marker == 0xdf) {
            std::uint32_t value = 0;
            if (!take_integer(value)) return false;
            count = value;
            return true;
        }
        return false;
    }

    bool take_string_header(std::size_t& count) {
        unsigned marker = 0;
        if (!take_byte(marker)) return false;
        if (marker >= 0xa0 && marker <= 0xbf) {
            count = marker & 0x1f;
            return true;
        }
        if (marker == 0xd9) {
            unsigned value = 0;
            if (!take_byte(value)) return false;
            count = value;
            return true;
        }
        if (marker == 0xda) {
            std::uint16_t value = 0;
            if (!take_integer(value)) return false;
            count = value;
            return true;
        }
        if (marker == 0xdb) {
            std::uint32_t value = 0;
            if (!take_integer(value)) return false;
            count = value;
            return true;
        }
        return false;
    }

    bool take_string(std::span<const std::byte>& text) {
        std::size_t size = 0;
        if (!take_string_header(size) || size > remaining()) return false;
        text = input_.subspan(offset_, size);
        offset_ += size;
        return true;
    }

    bool parse_map(std::size_t count, Shape& shape, ValuePolicy values,
                   unsigned next_container_depth) {
        if (count > remaining() / 2) return false;
        shape.category = Category::map;
        shape.count = count;
        shape.children.reserve(count);
        std::vector<std::span<const std::byte>> keys;
        keys.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::span<const std::byte> key;
            if (!take_string(key) || !valid_identifier(key)) return false;
            keys.push_back(key);
            Shape value{};
            value.key = key;
            if (!parse_value(value, values, next_container_depth)) return false;
            shape.children.push_back(std::move(value));
        }
        std::sort(keys.begin(), keys.end(), key_less_case_insensitive);
        for (std::size_t index = 1; index < keys.size(); ++index) {
            if (same_key_case_insensitive(keys[index - 1], keys[index])) return false;
        }
        return true;
    }

    bool parse_array(std::size_t count, Shape& shape, unsigned next_container_depth) {
        if (count > remaining()) return false;
        shape.category = Category::array;
        shape.count = count;
        if (count == 0) return true;
        Shape first{};
        if (!parse_value(first, ValuePolicy::array_element, next_container_depth)) return false;
        shape.children.push_back(std::move(first));
        for (std::size_t index = 1; index < count; ++index) {
            Shape current{};
            if (!parse_value(current, ValuePolicy::array_element, next_container_depth) ||
                !same_shape(shape.children.front(), current))
                return false;
        }
        return true;
    }

    bool parse_string(Shape& shape) {
        --offset_;
        std::span<const std::byte> text;
        if (!take_string(text) || !valid_utf8(text)) return false;
        shape.category = Category::string;
        return true;
    }

    bool parse_binary(Shape& shape, unsigned marker) {
        std::size_t size = 0;
        if (marker == 0xc4) {
            unsigned value = 0;
            if (!take_byte(value)) return false;
            size = value;
        } else if (marker == 0xc5) {
            std::uint16_t value = 0;
            if (!take_integer(value)) return false;
            size = value;
        } else {
            std::uint32_t value = 0;
            if (!take_integer(value)) return false;
            size = value;
        }
        if (!skip(size)) return false;
        shape.category = Category::binary;
        return true;
    }

    bool parse_extension(Shape& shape, ValuePolicy policy, unsigned marker) {
        std::size_t size = 0;
        if (marker == 0xd4) size = 1;
        else if (marker == 0xd5) size = 2;
        else if (marker == 0xd6) size = 4;
        else if (marker == 0xd7) size = 8;
        else if (marker == 0xd8) size = 16;
        else if (marker == 0xc7) {
            unsigned value = 0;
            if (!take_byte(value)) return false;
            size = value;
        } else if (marker == 0xc8) {
            std::uint16_t value = 0;
            if (!take_integer(value)) return false;
            size = value;
        } else {
            std::uint32_t value = 0;
            if (!take_integer(value)) return false;
            size = value;
        }
        unsigned type = 0;
        if (!take_byte(type) || !skip(size)) return false;
        const bool valid = policy == ValuePolicy::root
            ? (type == 0 && size == 8) || (type == 64 && size == 16) ||
                  (type == 65 && size == 1) || (type == 66 && size == 10)
            : type == 0 && size == 8;
        if (!valid) return false;
        shape.category = Category::extension;
        return true;
    }

    bool parse_value(Shape& shape, ValuePolicy policy, unsigned container_depth) {
        unsigned marker = 0;
        if (!take_byte(marker)) return false;
        if (marker <= 0x7f || marker >= 0xe0) {
            shape.category = Category::integer;
            return true;
        }
        if (marker >= 0xa0 && marker <= 0xbf) return parse_string(shape);
        if (marker >= 0x80 && marker <= 0x8f) {
            if (container_depth > maximum_container_depth) return false;
            return parse_map(marker & 0x0f, shape, ValuePolicy::nested, container_depth + 1);
        }
        if (marker >= 0x90 && marker <= 0x9f) {
            if (policy == ValuePolicy::array_element ||
                container_depth > maximum_container_depth)
                return false;
            return parse_array(marker & 0x0f, shape, container_depth + 1);
        }
        switch (marker) {
        case 0xc2:
        case 0xc3:
            shape.category = Category::boolean;
            return true;
        case 0xc4:
        case 0xc5:
        case 0xc6:
            return policy == ValuePolicy::root && parse_binary(shape, marker);
        case 0xc7:
        case 0xc8:
        case 0xc9:
        case 0xd4:
        case 0xd5:
        case 0xd6:
        case 0xd7:
        case 0xd8:
            return parse_extension(shape, policy, marker);
        case 0xca: {
            std::uint32_t bits = 0;
            if (!take_integer(bits) || !std::isfinite(std::bit_cast<float>(bits))) return false;
            shape.category = Category::floating;
            return true;
        }
        case 0xcb: {
            std::uint64_t bits = 0;
            if (!take_integer(bits) || !std::isfinite(std::bit_cast<double>(bits))) return false;
            shape.category = Category::floating;
            return true;
        }
        case 0xcc:
        case 0xd0:
            if (!skip(1)) return false;
            shape.category = Category::integer;
            return true;
        case 0xcd:
        case 0xd1:
            if (!skip(2)) return false;
            shape.category = Category::integer;
            return true;
        case 0xce:
        case 0xd2:
            if (!skip(4)) return false;
            shape.category = Category::integer;
            return true;
        case 0xcf:
        case 0xd3:
            if (!skip(8)) return false;
            shape.category = Category::integer;
            return true;
        case 0xd9:
        case 0xda:
        case 0xdb:
            return parse_string(shape);
        case 0xdc: {
            if (policy == ValuePolicy::array_element ||
                container_depth > maximum_container_depth)
                return false;
            std::uint16_t count = 0;
            return take_integer(count) && parse_array(count, shape, container_depth + 1);
        }
        case 0xdd: {
            if (policy == ValuePolicy::array_element ||
                container_depth > maximum_container_depth)
                return false;
            std::uint32_t count = 0;
            return take_integer(count) && parse_array(count, shape, container_depth + 1);
        }
        case 0xde: {
            if (container_depth > maximum_container_depth) return false;
            std::uint16_t count = 0;
            return take_integer(count) &&
                   parse_map(count, shape, ValuePolicy::nested, container_depth + 1);
        }
        case 0xdf: {
            if (container_depth > maximum_container_depth) return false;
            std::uint32_t count = 0;
            return take_integer(count) &&
                   parse_map(count, shape, ValuePolicy::nested, container_depth + 1);
        }
        default:
            return false;
        }
    }

    std::span<const std::byte> input_;
    std::size_t offset_{};
};

} // namespace

bool valid_report(std::span<const std::byte> report) {
    if (report.empty() || report.size() > maximum_report_size) return false;
    return Parser(report).parse_root();
}

} // namespace armigo::services::prepo::detail
