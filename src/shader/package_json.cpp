// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "package_internal.h"

#include <charconv>
#include <cstdio>
#include <stdexcept>

namespace armigo::shader::package::detail {
namespace {

[[noreturn]] void invalid_json() {
    throw std::runtime_error("reflection.json is invalid JSON");
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output += static_cast<char>(codepoint);
    } else if (codepoint <= 0x7ff) {
        output += static_cast<char>(0xc0 | (codepoint >> 6));
        output += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else if (codepoint <= 0xffff) {
        output += static_cast<char>(0xe0 | (codepoint >> 12));
        output += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        output += static_cast<char>(0x80 | (codepoint & 0x3f));
    } else {
        output += static_cast<char>(0xf0 | (codepoint >> 18));
        output += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
        output += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
        output += static_cast<char>(0x80 | (codepoint & 0x3f));
    }
}

class Parser {
public:
    explicit Parser(std::span<const std::byte> bytes)
        : text_(reinterpret_cast<const char*>(bytes.data()), bytes.size()) {}

    Json parse() {
        space();
        Json result = value(0);
        space();
        if (position_ != text_.size()) invalid_json();
        return result;
    }

private:
    void space() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\t' || text_[position_] == '\r' ||
                text_[position_] == '\n')) {
            ++position_;
        }
    }

    char take() {
        if (position_ == text_.size()) invalid_json();
        return text_[position_++];
    }

    bool consume(char expected) {
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    std::uint32_t hex4() {
        std::uint32_t value = 0;
        for (unsigned index = 0; index < 4; ++index) {
            const char digit = take();
            value <<= 4;
            if (digit >= '0' && digit <= '9') value |= static_cast<unsigned>(digit - '0');
            else if (digit >= 'a' && digit <= 'f') value |= static_cast<unsigned>(digit - 'a' + 10);
            else if (digit >= 'A' && digit <= 'F') value |= static_cast<unsigned>(digit - 'A' + 10);
            else invalid_json();
        }
        return value;
    }

    std::string string() {
        if (take() != '"') invalid_json();
        std::string result;
        while (position_ < text_.size()) {
            const unsigned char character = static_cast<unsigned char>(take());
            if (character == '"') return result;
            if (character < 0x20) invalid_json();
            if (character == '\\') {
                switch (take()) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        std::uint32_t codepoint = hex4();
                        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                            if (take() != '\\' || take() != 'u') invalid_json();
                            const std::uint32_t low = hex4();
                            if (low < 0xdc00 || low > 0xdfff) invalid_json();
                            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                        } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                            invalid_json();
                        }
                        append_utf8(result, codepoint);
                        break;
                    }
                    default: invalid_json();
                }
                continue;
            }
            if (character < 0x80) {
                result += static_cast<char>(character);
                continue;
            }
            std::uint32_t codepoint = 0;
            unsigned continuation = 0;
            if ((character & 0xe0) == 0xc0) {
                codepoint = character & 0x1f;
                continuation = 1;
                if (codepoint < 2) invalid_json();
            } else if ((character & 0xf0) == 0xe0) {
                codepoint = character & 0x0f;
                continuation = 2;
            } else if ((character & 0xf8) == 0xf0) {
                codepoint = character & 0x07;
                continuation = 3;
            } else {
                invalid_json();
            }
            for (unsigned index = 0; index < continuation; ++index) {
                const unsigned char next = static_cast<unsigned char>(take());
                if ((next & 0xc0) != 0x80) invalid_json();
                codepoint = (codepoint << 6) | (next & 0x3f);
            }
            if ((continuation == 1 && codepoint < 0x80) || (continuation == 2 && codepoint < 0x800) ||
                (continuation == 3 && codepoint < 0x10000) || codepoint > 0x10ffff ||
                (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
                invalid_json();
            }
            result += static_cast<char>(character);
            for (unsigned index = 0; index < continuation; ++index) {
                result += text_[position_ - continuation + index];
            }
        }
        invalid_json();
    }

    Json number() {
        const std::size_t begin = position_;
        consume('-');
        if (consume('0')) {
            if (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') invalid_json();
        } else {
            if (position_ == text_.size() || text_[position_] < '1' || text_[position_] > '9') invalid_json();
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
        }
        if (consume('.')) {
            if (position_ == text_.size() || text_[position_] < '0' || text_[position_] > '9') invalid_json();
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            if (position_ == text_.size() || text_[position_] < '0' || text_[position_] > '9') invalid_json();
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
        }
        return Json{JsonNumber{text_.substr(begin, position_ - begin)}};
    }

    Json array(unsigned depth) {
        Json::Array result;
        space();
        if (consume(']')) return Json{std::move(result)};
        for (;;) {
            space();
            result.push_back(value(depth + 1));
            space();
            if (consume(']')) return Json{std::move(result)};
            if (!consume(',')) invalid_json();
        }
    }

    Json object(unsigned depth) {
        Json::Object result;
        space();
        if (consume('}')) return Json{std::move(result)};
        for (;;) {
            space();
            if (position_ == text_.size() || text_[position_] != '"') invalid_json();
            std::string key = string();
            space();
            if (!consume(':')) invalid_json();
            space();
            auto [unused, inserted] = result.emplace(std::move(key), value(depth + 1));
            static_cast<void>(unused);
            if (!inserted) invalid_json();
            space();
            if (consume('}')) return Json{std::move(result)};
            if (!consume(',')) invalid_json();
        }
    }

    Json value(unsigned depth) {
        if (depth > 128 || position_ == text_.size()) invalid_json();
        const char character = text_[position_];
        if (character == '"') return Json{string()};
        if (character == '{') {
            ++position_;
            return object(depth);
        }
        if (character == '[') {
            ++position_;
            return array(depth);
        }
        if (text_.compare(position_, 4, "null") == 0) {
            position_ += 4;
            return Json{nullptr};
        }
        if (text_.compare(position_, 4, "true") == 0) {
            position_ += 4;
            return Json{true};
        }
        if (text_.compare(position_, 5, "false") == 0) {
            position_ += 5;
            return Json{false};
        }
        if (character == '-' || (character >= '0' && character <= '9')) return number();
        invalid_json();
    }

    std::string text_;
    std::size_t position_ = 0;
};

void append_hex4(std::string& output, std::uint32_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    output += "\\u";
    output += digits[(value >> 12) & 15];
    output += digits[(value >> 8) & 15];
    output += digits[(value >> 4) & 15];
    output += digits[value & 15];
}

std::uint32_t decode_utf8(std::string_view text, std::size_t& position) {
    const auto first = static_cast<unsigned char>(text[position++]);
    if (first < 0x80) return first;
    unsigned count = (first & 0xe0) == 0xc0 ? 1 : (first & 0xf0) == 0xe0 ? 2 : 3;
    std::uint32_t result = first & (count == 1 ? 0x1f : count == 2 ? 0x0f : 0x07);
    for (unsigned index = 0; index < count; ++index) {
        result = (result << 6) | (static_cast<unsigned char>(text[position++]) & 0x3f);
    }
    return result;
}

void append_string(std::string& output, std::string_view text) {
    output += '"';
    for (std::size_t position = 0; position < text.size();) {
        const unsigned char character = static_cast<unsigned char>(text[position]);
        switch (character) {
            case '"': output += "\\\""; ++position; continue;
            case '\\': output += "\\\\"; ++position; continue;
            case '\b': output += "\\b"; ++position; continue;
            case '\f': output += "\\f"; ++position; continue;
            case '\n': output += "\\n"; ++position; continue;
            case '\r': output += "\\r"; ++position; continue;
            case '\t': output += "\\t"; ++position; continue;
            default: break;
        }
        if (character < 0x20) {
            append_hex4(output, character);
            ++position;
        } else if (character < 0x80) {
            output += static_cast<char>(character);
            ++position;
        } else {
            const std::uint32_t codepoint = decode_utf8(text, position);
            if (codepoint <= 0xffff) {
                append_hex4(output, codepoint);
            } else {
                const std::uint32_t adjusted = codepoint - 0x10000;
                append_hex4(output, 0xd800 + (adjusted >> 10));
                append_hex4(output, 0xdc00 + (adjusted & 0x3ff));
            }
        }
    }
    output += '"';
}

void append_json(std::string& output, const Json& value) {
    if (value.is_null()) {
        output += "null";
    } else if (value.is_bool()) {
        output += value.boolean() ? "true" : "false";
    } else if (value.is_number()) {
        output += std::get<JsonNumber>(value.value).text;
    } else if (value.is_string()) {
        append_string(output, value.string());
    } else if (value.is_array()) {
        output += '[';
        bool first = true;
        for (const Json& item : value.array()) {
            if (!first) output += ',';
            first = false;
            append_json(output, item);
        }
        output += ']';
    } else {
        output += '{';
        bool first = true;
        for (const auto& [key, item] : value.object()) {
            if (!first) output += ',';
            first = false;
            append_string(output, key);
            output += ':';
            append_json(output, item);
        }
        output += '}';
    }
}

} // namespace

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

Json parse_json(std::span<const std::byte> bytes) {
    return Parser{bytes}.parse();
}

std::string canonical_json(const Json& value) {
    std::string result;
    append_json(result, value);
    return result;
}

bool is_uint(const Json& value, std::uint64_t maximum) {
    if (!value.is_number()) return false;
    const std::string& text = std::get<JsonNumber>(value.value).text;
    if (text.empty() || text[0] == '-' || text.find_first_of(".eE") != std::string::npos) return false;
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && parsed <= maximum;
}

std::uint64_t uint_value(const Json& value) {
    const std::string& text = std::get<JsonNumber>(value.value).text;
    std::uint64_t result = 0;
    static_cast<void>(std::from_chars(text.data(), text.data() + text.size(), result));
    return result;
}

void require_keys(const Json& value, std::span<const std::string_view> keys, std::string_view label) {
    require(value.is_object() && value.object().size() == keys.size(),
            std::string(label) + " has an unexpected schema");
    for (const std::string_view key : keys) {
        require(value.object().contains(key), std::string(label) + " has an unexpected schema");
    }
}

const Json& member(const Json& value, std::string_view key) {
    return value.object().at(std::string(key));
}

const Json* find_member(const Json& value, std::string_view key) {
    if (!value.is_object()) return nullptr;
    const auto found = value.object().find(key);
    return found == value.object().end() ? nullptr : &found->second;
}

} // namespace armigo::shader::package::detail
