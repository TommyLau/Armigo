// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "package_tools.h"

#include "package_internal.h"

#include <set>
#include <stdexcept>

namespace armigo::pack {
namespace {

namespace fs = std::filesystem;
namespace detail = armigo::shader::package::detail;

constexpr std::size_t maximum_manifest = 256u << 20;
constexpr std::size_t examples_per_file = 3;

detail::Json read_manifest(const fs::path& package) {
    const auto bytes = detail::read_regular(package / "manifest.json", maximum_manifest, "manifest.json");
    auto manifest = detail::parse_json(bytes);
    detail::require(manifest.is_object(), "manifest.json is not an object");
    return manifest;
}

std::string text_of(const detail::Json* value) {
    return value ? detail::canonical_json(*value) : std::string("null");
}

std::set<std::string> names_in(const fs::path& directory) {
    std::set<std::string> names;
    std::error_code error;
    if (!fs::is_directory(directory, error)) return names;
    for (const auto& entry : fs::directory_iterator(directory, error)) names.insert(entry.path().filename().string());
    if (error) throw std::runtime_error("Cannot list " + directory.string());
    return names;
}

void compare_fields(const detail::Json& old_manifest, const detail::Json& new_manifest,
                    PackageComparison& comparison) {
    std::set<std::string> keys;
    for (const auto& [key, value] : old_manifest.object()) keys.insert(key);
    for (const auto& [key, value] : new_manifest.object()) keys.insert(key);
    for (const auto& key : keys) {
        const auto* a = detail::find_member(old_manifest, key);
        const auto* b = detail::find_member(new_manifest, key);
        if (key == "profile_text") {
            const auto parse = [](const detail::Json* text) {
                if (!text || !text->is_string()) return detail::Json{detail::Json::Object{}};
                const auto& value = text->string();
                return detail::parse_json(std::as_bytes(std::span(value.data(), value.size())));
            };
            const auto pa = parse(a), pb = parse(b);
            std::set<std::string> fields;
            if (pa.is_object()) for (const auto& [field, value] : pa.object()) fields.insert(field);
            if (pb.is_object()) for (const auto& [field, value] : pb.object()) fields.insert(field);
            for (const auto& field : fields) {
                const auto* fa = pa.is_object() ? detail::find_member(pa, field) : nullptr;
                const auto* fb = pb.is_object() ? detail::find_member(pb, field) : nullptr;
                if (text_of(fa) != text_of(fb))
                    comparison.fields.push_back("profile." + field + ": " + text_of(fa) + " -> " + text_of(fb));
            }
        } else if (key == "entries") {
            const auto count = [](const detail::Json* value) {
                return value && value->is_array() ? value->array().size() : std::size_t{0};
            };
            if (count(a) != count(b) || text_of(a) != text_of(b))
                comparison.fields.push_back("entries: " + std::to_string(count(a)) + " -> " + std::to_string(count(b)));
        } else if (text_of(a) != text_of(b)) {
            comparison.fields.push_back(key + ": " + text_of(a) + " -> " + text_of(b));
        }
    }
}

} // namespace

bool PackageComparison::same() const {
    return fields.empty() && only_old == 0 && only_new == 0 && differing.empty();
}

std::variant<std::string, PackageComparison> compare_packages(const fs::path& old_package,
                                                              const fs::path& new_package) {
    try {
        PackageComparison comparison;
        compare_fields(read_manifest(old_package), read_manifest(new_package), comparison);
        const auto old_keys = names_in(old_package / "stages"), new_keys = names_in(new_package / "stages");
        comparison.old_stages = old_keys.size();
        comparison.new_stages = new_keys.size();
        std::map<std::string, std::size_t> shown;
        for (const auto& key : old_keys) {
            if (!new_keys.contains(key)) {
                ++comparison.only_old;
                continue;
            }
            ++comparison.common;
            auto names = names_in(old_package / "stages" / key);
            names.merge(names_in(new_package / "stages" / key));
            for (const auto& name : names) {
                const auto a = old_package / "stages" / key / name, b = new_package / "stages" / key / name;
                std::error_code error;
                const bool same = fs::is_regular_file(a, error) && fs::is_regular_file(b, error) &&
                    detail::sha256_file(a) == detail::sha256_file(b);
                if (same) {
                    ++comparison.identical[name];
                } else {
                    ++comparison.differing[name];
                    if (shown[name]++ < examples_per_file) comparison.examples.push_back(key + "/" + name);
                }
            }
        }
        for (const auto& key : new_keys)
            if (!old_keys.contains(key)) ++comparison.only_new;
        return comparison;
    } catch (const std::exception& failure) {
        return std::string(failure.what());
    }
}

std::string format_comparison(const PackageComparison& comparison) {
    std::string text;
    for (const auto& field : comparison.fields) text += field + "\n";
    text += "stages: old " + std::to_string(comparison.old_stages) + ", new " + std::to_string(comparison.new_stages) +
        ", common " + std::to_string(comparison.common) + ", only old " + std::to_string(comparison.only_old) +
        ", only new " + std::to_string(comparison.only_new) + "\n";
    for (const auto& example : comparison.examples) text += "differs: " + example + "\n";
    const auto counts = [](const std::map<std::string, std::size_t>& values) {
        std::string line;
        for (const auto& [name, count] : values) line += (line.empty() ? "" : ", ") + name + " " + std::to_string(count);
        return line.empty() ? std::string("none") : line;
    };
    text += "identical files: " + counts(comparison.identical) + "\n";
    text += "differing files: " + counts(comparison.differing) + "\n";
    return text;
}

} // namespace armigo::pack
