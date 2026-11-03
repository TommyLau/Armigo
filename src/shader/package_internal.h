// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::shader::package::detail {

inline constexpr std::size_t prefix_size = 112;
inline constexpr std::size_t maximum_program_size = 4 * 1024 * 1024;
inline constexpr std::size_t maximum_source_size = 4 * 1024 * 1024;
inline constexpr std::size_t maximum_reflection_size = 1024 * 1024;
inline constexpr std::size_t maximum_library_size = 64 * 1024 * 1024;
inline constexpr std::uint64_t maximum_input_bytes = 8ull * 1024 * 1024 * 1024;
inline constexpr std::uint64_t maximum_output_bytes = 1024ull * 1024 * 1024;
inline constexpr std::size_t maximum_identities = 1'000'000;
inline constexpr std::size_t maximum_bucket = 64;

struct JsonNumber {
    std::string text;
    bool operator==(const JsonNumber&) const = default;
};

struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json, std::less<>>;
    std::variant<std::nullptr_t, bool, JsonNumber, std::string, Array, Object> value = nullptr;

    [[nodiscard]] bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
    [[nodiscard]] bool is_bool() const { return std::holds_alternative<bool>(value); }
    [[nodiscard]] bool is_number() const { return std::holds_alternative<JsonNumber>(value); }
    [[nodiscard]] bool is_string() const { return std::holds_alternative<std::string>(value); }
    [[nodiscard]] bool is_array() const { return std::holds_alternative<Array>(value); }
    [[nodiscard]] bool is_object() const { return std::holds_alternative<Object>(value); }
    [[nodiscard]] bool boolean() const { return std::get<bool>(value); }
    [[nodiscard]] const std::string& string() const { return std::get<std::string>(value); }
    [[nodiscard]] const Array& array() const { return std::get<Array>(value); }
    [[nodiscard]] const Object& object() const { return std::get<Object>(value); }
    Array& array() { return std::get<Array>(value); }
    Object& object() { return std::get<Object>(value); }
    bool operator==(const Json&) const = default;
};

// Integers and strings share the string alternative; an integer is recognized
// by the JSON grammar retained in its text.
[[nodiscard]] Json parse_json(std::span<const std::byte> bytes);
[[nodiscard]] std::string canonical_json(const Json& value);
[[nodiscard]] bool is_uint(const Json& value, std::uint64_t maximum);
[[nodiscard]] std::uint64_t uint_value(const Json& value);

void require(bool condition, std::string_view message);
void require_keys(const Json& value, std::span<const std::string_view> keys, std::string_view label);
[[nodiscard]] const Json& member(const Json& value, std::string_view key);
[[nodiscard]] const Json* find_member(const Json& value, std::string_view key);

struct Stage {
    std::filesystem::path root;
    std::vector<std::byte> program;
    std::vector<std::byte> source;
    std::vector<std::byte> reflection;
    Json metadata;
    std::string stage;
    std::string program_key;
    std::string prefix_key;
    std::string variant_id;
};

[[nodiscard]] std::string sha256(std::span<const std::byte> bytes);
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);
[[nodiscard]] std::vector<std::byte> read_regular(const std::filesystem::path& path, std::size_t maximum,
                                                   std::string_view label);
void write_file(const std::filesystem::path& path, std::span<const std::byte> bytes);

[[nodiscard]] Stage load_stage(const std::filesystem::path& path);
[[nodiscard]] std::vector<Stage> validate_inventory(std::vector<Stage> stages);
[[nodiscard]] std::string variant_data(const Json& metadata);

} // namespace armigo::shader::package::detail
