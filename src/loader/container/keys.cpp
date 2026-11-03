// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Keys-file parsing for container input. Written for Armigo from the
// prod.keys text format; no reference code is translated or vendored.
// Decoded key values never appear in diagnostics, which name keys only.

#include <armigo/loader/container.h>

#include <algorithm>
#include <cstdio>
#include <utility>

namespace armigo::loader {
namespace {

std::optional<std::vector<std::byte>> decode_hex(std::string_view text) {
    if (text.size() % 2 != 0) {
        return std::nullopt;
    }
    std::vector<std::byte> value;
    value.reserve(text.size() / 2);
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < text.size(); i += 2) {
        const int high = nibble(text[i]);
        const int low = nibble(text[i + 1]);
        if (high < 0 || low < 0) {
            return std::nullopt;
        }
        value.push_back(std::byte(static_cast<std::uint8_t>((high << 4) | low)));
    }
    return value;
}

bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}

template<typename Container>
struct ByteWiper {
    Container& bytes;

    ~ByteWiper() {
        for (volatile std::byte& byte : bytes) {
            byte = std::byte{0};
        }
    }
};

} // namespace

ContainerKeys::~ContainerKeys() {
    clear();
}

ContainerKeys& ContainerKeys::operator=(const ContainerKeys& other) {
    if (this != &other) {
        clear();
        entries_ = other.entries_;
    }
    return *this;
}

ContainerKeys& ContainerKeys::operator=(ContainerKeys&& other) noexcept {
    if (this != &other) {
        clear();
        entries_ = std::move(other.entries_);
    }
    return *this;
}

void ContainerKeys::add(std::string name, std::vector<std::byte> value) {
    entries_.emplace_back(std::move(name), std::move(value));
}

void ContainerKeys::clear() noexcept {
    for (auto& [name, value] : entries_) {
        for (volatile std::byte& byte : value) {
            byte = std::byte{0};
        }
    }
}

const std::vector<std::byte>* ContainerKeys::find(std::string_view name) const {
    const auto match = [&](const auto& entry) { return entry.first == name; };
    const auto position = std::find_if(entries_.begin(), entries_.end(), match);
    return position == entries_.end() ? nullptr : &position->second;
}

std::variant<ContainerKeysFailure, ContainerKeys> parse_container_keys(
    std::span<const std::byte> text) {
    ContainerKeys keys;
    const std::string_view content(reinterpret_cast<const char*>(text.data()), text.size());
    std::size_t line_number = 0;
    std::size_t position = 0;
    while (position <= content.size()) {
        const std::size_t end = content.find('\n', position);
        const std::size_t line_end = end == std::string::npos ? content.size() : end;
        std::string_view line(content.data() + position, line_end - position);
        ++line_number;
        position = end == std::string::npos ? content.size() + 1 : end + 1;

        // Strip the comment, then trim both sides.
        const auto comment = line.find_first_of("#;");
        if (comment != std::string_view::npos) {
            line = line.substr(0, comment);
        }
        while (!line.empty() && is_space(line.front())) line.remove_prefix(1);
        while (!line.empty() && is_space(line.back())) line.remove_suffix(1);
        if (line.empty()) {
            if (position > content.size()) break;
            continue;
        }

        const auto equals = line.find('=');
        if (equals == std::string_view::npos) {
            return ContainerKeysFailure{ContainerKeysFailure::Code::malformed_entry,
                                        "Line " + std::to_string(line_number) +
                                            " has no '=' separator."};
        }
        std::string_view name = line.substr(0, equals);
        std::string_view value = line.substr(equals + 1);
        while (!name.empty() && is_space(name.back())) name.remove_suffix(1);
        while (!value.empty() && is_space(value.front())) value.remove_prefix(1);
        while (!value.empty() && is_space(value.back())) value.remove_suffix(1);
        if (name.empty()) {
            return ContainerKeysFailure{ContainerKeysFailure::Code::malformed_entry,
                                        "Line " + std::to_string(line_number) +
                                            " has an empty key name."};
        }
        if (keys.find(name) != nullptr) {
            return ContainerKeysFailure{ContainerKeysFailure::Code::duplicate_key,
                                        "Key '" + std::string(name) + "' appears more than once."};
        }
        auto decoded = decode_hex(value);
        if (!decoded) {
            return ContainerKeysFailure{ContainerKeysFailure::Code::bad_hex_value,
                                        "Key '" + std::string(name) +
                                            "' has a malformed hex value."};
        }
        keys.add(std::string(name), std::move(*decoded));
        if (position > content.size()) break;
    }
    return keys;
}

std::variant<ContainerKeysFailure, ContainerKeys> read_container_keys_file(
    const std::filesystem::path& keys_file) {
    std::FILE* file = std::fopen(keys_file.c_str(), "rb");
    if (!file) {
        return ContainerKeysFailure{ContainerKeysFailure::Code::file_unavailable,
                                    "Could not open the keys file."};
    }
    std::vector<std::byte> text;
    ByteWiper text_wiper{text};
    constexpr std::size_t kChunk = 4096;
    std::array<std::byte, kChunk> buffer{};
    ByteWiper buffer_wiper{buffer};
    for (;;) {
        const std::size_t read = std::fread(buffer.data(), 1, kChunk, file);
        text.insert(text.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(read));
        if (read != kChunk) {
            const bool failed = std::ferror(file) != 0;
            std::fclose(file);
            if (failed) {
                return ContainerKeysFailure{ContainerKeysFailure::Code::file_unavailable,
                                            "Reading the keys file failed."};
            }
            break;
        }
    }
    return parse_container_keys(text);
}

} // namespace armigo::loader
