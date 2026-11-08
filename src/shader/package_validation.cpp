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

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace armigo::shader::package::detail {
namespace {

namespace fs = std::filesystem;

const Json& m(const Json& value, std::string_view key) { return member(value, key); }
const std::string& text(const Json& value, std::string_view key) { return m(value, key).string(); }
std::uint64_t number(const Json& value, std::string_view key) { return uint_value(m(value, key)); }

template <typename... Keys>
void exact(const Json& value, std::string_view label, Keys... keys) {
    const std::array<std::string_view, sizeof...(Keys)> fields{std::string_view(keys)...};
    require_keys(value, fields, label);
}

bool one_of(std::string_view value, std::initializer_list<std::string_view> choices) {
    return std::find(choices.begin(), choices.end(), value) != choices.end();
}

bool boolean(const Json& value, std::string_view key) {
    const Json& field = m(value, key);
    return field.is_bool() && field.boolean();
}

void require_hex_digest(const Json& value, std::string_view message) {
    require(value.is_string() && value.string().size() == 64 &&
                std::all_of(value.string().begin(), value.string().end(), [](char character) {
                    return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
                }),
            message);
}

void validate_texture_component_contract(const Json& value) {
    const std::string& scalar = text(value, "scalar");
    const std::string& component = text(value, "component_format");
    const bool compatible =
        (one_of(scalar, {"f16", "f32"}) && one_of(component, {"floating", "unorm", "snorm", "depth"})) ||
        (scalar == "u32" && component == "unsigned_integer") ||
        (scalar == "s32" && component == "signed_integer");
    require(compatible && (!m(value, "comparison").boolean() || (scalar == "f32" && component == "depth")),
            "texture has incompatible scalar and component format");
}

void texture_descriptor_value(const Json& value, std::string_view label) {
    exact(value, label, "dimension", "array", "multisample", "scalar", "component_format",
          "normalized_coordinates", "comparison");
    require(one_of(text(value, "dimension"), {"d1", "d2", "d3", "cube", "buffer"}) &&
                one_of(text(value, "scalar"), {"u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64",
                                                "f16", "f32", "f64", "boolean"}) &&
                one_of(text(value, "component_format"),
                       {"unorm", "snorm", "unsigned_integer", "signed_integer", "floating", "depth"}) &&
                m(value, "array").is_bool() && m(value, "multisample").is_bool() &&
                m(value, "normalized_coordinates").is_bool() && m(value, "comparison").is_bool(),
            std::string(label) + " value is invalid");
    validate_texture_component_contract(value);
}

void validate_attributes(const Json& items) {
    require(items.is_array(), "vertex_attributes must be an array");
    std::int64_t previous = -1;
    std::set<std::uint64_t> attributes;
    for (const Json& item : items.array()) {
        exact(item, "vertex attribute", "generic_slot", "attribute_index", "format");
        const auto slot = number(item, "generic_slot");
        const auto attribute = number(item, "attribute_index");
        require(is_uint(m(item, "generic_slot"), 31) && static_cast<std::int64_t>(slot) > previous &&
                    is_uint(m(item, "attribute_index"), 30) && attributes.insert(attribute).second &&
                    text(item, "format") == "float32x3",
                "vertex attribute option is unsupported");
        previous = static_cast<std::int64_t>(slot);
    }
}

void validate_constant_buffers(const Json& items) {
    require(items.is_array() && items.array().size() <= 18,
            "constant_buffers must be an array of at most 18 entries");
    std::set<std::uint64_t> guests, metals;
    for (const Json& item : items.array()) {
        exact(item, "constant buffer", "guest_binding", "minimum_size", "metal_buffer_index");
        const auto guest = number(item, "guest_binding"), size = number(item, "minimum_size");
        const auto metal = number(item, "metal_buffer_index");
        require(is_uint(m(item, "guest_binding"), 17) && is_uint(m(item, "minimum_size"), 65536) && size > 0 &&
                    size % 4 == 0 && is_uint(m(item, "metal_buffer_index"), 17) && guests.insert(guest).second &&
                    metals.insert(metal).second,
                "constant-buffer reflection is invalid");
    }
}

void validate_textures(const Json& items) {
    require(items.is_array() && items.array().size() <= 16, "textures must be an array of at most 16 entries");
    std::set<std::uint64_t> handles, textures, samplers;
    for (const Json& item : items.array()) {
        exact(item, "texture", "handle_byte_offset", "texture_index", "sampler_index");
        const auto handle = number(item, "handle_byte_offset"), texture = number(item, "texture_index");
        const auto sampler = number(item, "sampler_index");
        require(is_uint(m(item, "handle_byte_offset"), 32764) && handle % 4 == 0 &&
                    is_uint(m(item, "texture_index"), 15) && is_uint(m(item, "sampler_index"), 15) &&
                    handles.insert(handle).second && textures.insert(texture).second && samplers.insert(sampler).second,
                "texture reflection is invalid");
    }
}

void validate_generic(const Json& items, std::string_view label) {
    require(items.is_array() && items.array().size() <= 32,
            std::string(label) + " must be an array of at most 32 entries");
    std::set<std::uint64_t> slots;
    for (const Json& item : items.array()) {
        exact(item, label, "slot", "component_mask", "interpolation");
        require(is_uint(m(item, "slot"), 31) && is_uint(m(item, "component_mask"), 15) &&
                    number(item, "component_mask") != 0 && slots.insert(number(item, "slot")).second &&
                    one_of(text(item, "interpolation"), {"constant", "perspective", "screen-linear"}),
                std::string(label) + " reflection is invalid");
    }
}

void validate_generic_v2(const Json& items, std::string_view label) {
    require(items.is_array() && items.array().size() <= 32,
            std::string(label) + " must be an array of at most 32 entries");
    std::set<std::uint64_t> slots;
    for (const Json& item : items.array()) {
        exact(item, label, "slot", "component_mask", "scalar", "components", "interpolation");
        require(is_uint(m(item, "slot"), 31) && is_uint(m(item, "component_mask"), 15) &&
                    number(item, "component_mask") != 0 && slots.insert(number(item, "slot")).second &&
                    one_of(text(item, "interpolation"), {"flat", "perspective", "linear"}) &&
                    one_of(text(item, "scalar"), {"u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64",
                                                   "f16", "f32", "f64", "boolean"}) &&
                    is_uint(m(item, "components"), 4) && number(item, "components") != 0,
                std::string(label) + " reflection is invalid");
    }
}

bool validate_interface(const Json& value, std::string_view stage, const Json& attributes, const Json& textures) {
    exact(value, "interface", "position_input_mask", "position_output_mask", "color_output_mask", "generic_inputs",
          "generic_outputs");
    require(is_uint(m(value, "position_input_mask"), 15) && is_uint(m(value, "position_output_mask"), 15) &&
                is_uint(m(value, "color_output_mask"), 0xffffffff),
            "interface masks are invalid");
    validate_generic(m(value, "generic_inputs"), "generic_inputs");
    validate_generic(m(value, "generic_outputs"), "generic_outputs");
    if (stage == "vertex") {
        require(number(value, "position_input_mask") == 0 && number(value, "position_output_mask") == 15 &&
                    number(value, "color_output_mask") == 0 && textures.array().empty() &&
                    m(value, "generic_inputs").array().size() == attributes.array().size(),
                "vertex interface is outside the supported profile");
        for (const Json& option : attributes.array()) {
            bool found = false;
            for (const Json& input : m(value, "generic_inputs").array()) {
                if (number(input, "slot") == number(option, "generic_slot")) {
                    found = (number(input, "component_mask") & ~7ull) == 0;
                    break;
                }
            }
            require(found, "vertex attributes do not cover the reflected inputs");
        }
        require(std::all_of(m(value, "generic_outputs").array().begin(), m(value, "generic_outputs").array().end(),
                            [](const Json& item) { return text(item, "interpolation") == "screen-linear"; }),
                "vertex output interpolation is invalid");
    } else {
        const bool perspective = std::any_of(
            m(value, "generic_inputs").array().begin(), m(value, "generic_inputs").array().end(),
            [](const Json& item) { return text(item, "interpolation") == "perspective"; });
        require(number(value, "position_output_mask") == 0 && number(value, "color_output_mask") == 15 &&
                    m(value, "generic_outputs").array().empty() &&
                    one_of(std::to_string(number(value, "position_input_mask")), {"0", "8"}),
                "fragment interface is outside the supported profile");
        return perspective;
    }
    return false;
}

void validate_profile(const Json& value) {
    exact(value, "profile", "identity", "version");
    require(m(value, "identity").is_string() && !text(value, "identity").empty() &&
                std::all_of(text(value, "identity").begin(), text(value, "identity").end(),
                            [](unsigned char character) { return character < 0x80; }) &&
                is_uint(m(value, "version"), 0xffffffff),
            "profile reflection is invalid");
}

std::string validate_reflection_v1(const Json& metadata, std::span<const std::byte> program,
                                   std::span<const std::byte> source) {
    exact(metadata, "reflection", "schema", "source_version", "program_key", "program_size", "program_file",
          "source_file", "source_digest", "stage", "entry_point", "vertex_depth", "fragment_position",
          "vertex_attributes", "constant_buffers", "textures", "interface");
    require(text(metadata, "schema") == "armigo-shader-stage-v1", "reflection schema is unsupported");
    for (const std::string_view name : {"source_version", "program_key", "source_digest"}) {
        require_hex_digest(m(metadata, name), "reflection " + std::string(name) + " is invalid");
    }
    require(text(metadata, "program_file") == "program.bin" && text(metadata, "source_file") == "source.metal",
            "reflection filenames are invalid");
    require(number(metadata, "program_size") == program.size() && text(metadata, "program_key") == sha256(program),
            "program identity does not match reflection");
    require(text(metadata, "source_digest") == sha256(source), "source digest does not match reflection");
    const std::string& stage = text(metadata, "stage");
    require(one_of(stage, {"vertex", "fragment"}), "reflection stage is unsupported");
    require(program.size() >= prefix_size, "program is shorter than the identity prefix");
    require(program.size() <= maximum_program_size, "program exceeds the identity size policy");
    std::uint32_t sph = 0;
    std::memcpy(&sph, program.data(), sizeof sph);
    require(((sph >> 10) & 15) == (stage == "vertex" ? 1 : 5), "program SPH stage mismatch");
    validate_attributes(m(metadata, "vertex_attributes"));
    if (stage == "vertex") {
        require(text(metadata, "entry_point") == "armigo_vertex_main" &&
                    one_of(text(metadata, "vertex_depth"), {"minus-one-to-one", "zero-to-one"}) &&
                    text(metadata, "fragment_position") == "unspecified",
                "vertex explicit options are invalid");
    } else {
        require(text(metadata, "entry_point") == "armigo_fragment_main" &&
                    text(metadata, "vertex_depth") == "unspecified" &&
                    one_of(text(metadata, "fragment_position"), {"unspecified", "reciprocal-clip-w"}) &&
                    m(metadata, "vertex_attributes").array().empty(),
                "fragment explicit options are invalid");
    }
    validate_constant_buffers(m(metadata, "constant_buffers"));
    validate_textures(m(metadata, "textures"));
    const bool perspective = validate_interface(m(metadata, "interface"), stage, m(metadata, "vertex_attributes"),
                                                m(metadata, "textures"));
    if (stage == "fragment") {
        const bool needs = number(m(metadata, "interface"), "position_input_mask") == 8 || perspective;
        require((needs && text(metadata, "fragment_position") == "reciprocal-clip-w") ||
                    (!needs && text(metadata, "fragment_position") == "unspecified"),
                "fragment position option does not match its interface");
    }
    return stage;
}

std::set<std::uint64_t> validate_constant_buffers_v2(const Json& items) {
    require(items.is_array() && items.array().size() <= 18,
            "constant_buffers must be an array of at most 18 entries");
    std::set<std::uint64_t> guests, metals;
    for (const Json& item : items.array()) {
        exact(item, "constant buffer", "guest_binding", "size_kind", "window_bytes", "alignment", "policy",
              "metal_buffer_index");
        const auto guest = number(item, "guest_binding"), extent = number(item, "window_bytes");
        const auto metal = number(item, "metal_buffer_index");
        require(is_uint(m(item, "guest_binding"), 17) && is_uint(m(item, "window_bytes"), 65536) && extent >= 4 &&
                    extent % 4 == 0 && is_uint(m(item, "metal_buffer_index"), 30) && guests.insert(guest).second &&
                    metals.insert(metal).second && text(item, "size_kind") == "constant_buffer_window" &&
                    is_uint(m(item, "alignment"), 4) && number(item, "alignment") == 4 &&
                    text(item, "policy") == "runtime_zero_words_v1",
                "constant-buffer reflection is invalid");
    }
    return metals;
}

void validate_support_buffers_v2(const Json& items, const std::set<std::uint64_t>& constant_indices) {
    require(items.is_array() && items.array().size() == (constant_indices.empty() ? 0 : 1),
            "support_buffers must contain one entry exactly when constant buffers exist");
    for (const Json& item : items.array()) {
        exact(item, "support buffer", "schema", "version", "byte_extent", "alignment", "metal_buffer_index");
        require(text(item, "schema") == "armigo-cb-sizes" && is_uint(m(item, "version"), 1) &&
                    number(item, "version") == 1 && is_uint(m(item, "byte_extent"), 72) &&
                    number(item, "byte_extent") == 72 && is_uint(m(item, "alignment"), 4) &&
                    number(item, "alignment") == 4 && is_uint(m(item, "metal_buffer_index"), 30) &&
                    !constant_indices.contains(number(item, "metal_buffer_index")),
                "support-buffer reflection is invalid");
    }
}

std::set<std::uint64_t> validate_storage_buffers_v3(const Json& items, const Json& constant_buffers) {
    require(items.is_array() && items.array().size() <= 31,
            "storage_buffers must be an array of at most 31 entries");
    std::map<std::uint64_t, std::uint64_t> windows;
    for (const Json& item : constant_buffers.array()) windows[number(item, "guest_binding")] = number(item, "window_bytes");
    std::set<std::pair<std::uint64_t, std::uint64_t>> descriptors;
    std::set<std::uint64_t> metals;
    for (const Json& item : items.array()) {
        exact(item, "storage buffer", "cbuf_slot", "descriptor_byte_offset", "metal_buffer_index", "access", "alignment");
        const auto slot = number(item, "cbuf_slot"), offset = number(item, "descriptor_byte_offset");
        const auto metal = number(item, "metal_buffer_index");
        const auto window = windows.find(slot);
        require(is_uint(m(item, "cbuf_slot"), 17) && is_uint(m(item, "descriptor_byte_offset"), 65524) && offset % 8 == 0 &&
                    window != windows.end() && offset + 12 <= window->second && is_uint(m(item, "metal_buffer_index"), 30) &&
                    metals.insert(metal).second && descriptors.insert({slot, offset}).second &&
                    one_of(text(item, "access"), {"read", "write", "read_write"}) &&
                    is_uint(m(item, "alignment"), 4) && number(item, "alignment") == 4,
                "storage-buffer reflection is invalid");
    }
    return metals;
}

// Slot 31 names handle program (offset / 4), not a constant buffer. Slots 18
// through 30 stay outside the constant-buffer range. `elements` is how many
// consecutive programs the binding needs (one per array element, or one for a
// descriptor fact). The stage loader uses the same rule.
constexpr std::uint64_t handle_program_slot = 31;
constexpr std::size_t handle_program_limit = 4096;

std::size_t validate_handle_programs(const Json& metadata) {
    const Json* programs = find_member(metadata, "handle_programs");
    if (programs == nullptr) return 0;
    require(programs->is_array(), "handle_programs must be an array");
    require(programs->array().size() <= handle_program_limit,
            "handle_programs exceeds its count policy");
    for (const Json& program : programs->array()) {
        require(program.is_array(), "handle program must be an array");
        require(program.array().size() <= handle_program_limit,
                "handle program exceeds its count policy");
        for (const Json& word : program.array()) {
            require(is_uint(word, 0xffffffffu), "handle program word is invalid");
        }
    }
    return programs->array().size();
}

bool admits_texture_slot(const Json& slot_value, std::uint64_t offset, std::uint64_t elements,
                         std::size_t programs) {
    if (!is_uint(slot_value, handle_program_slot)) return false;
    const auto slot = uint_value(slot_value);
    if (slot != handle_program_slot) return slot <= 17;
    if ((offset & 3ull) != 0 || elements == 0) return false;
    return offset / 4ull + elements <= programs;
}

void validate_textures_v2(const Json& items, std::size_t programs) {
    require(items.is_array() && items.array().size() <= 32, "textures must be an array of at most 32 entries");
    std::set<std::uint64_t> texture_handles, sampler_handles, textures, samplers;
    for (const Json& item : items.array()) {
        require(item.is_object() && find_member(item, "kind") != nullptr, "texture entry is invalid");
        const std::string& kind = text(item, "kind");
        require(one_of(kind, {"texture", "sampler"}), "texture kind is unsupported");
        if (kind == "texture") {
            exact(item, "texture", "cbuf_slot", "handle_byte_offset", "kind", "dimension", "array", "multisample",
                  "scalar", "component_format", "normalized_coordinates", "comparison", "texture_index");
        } else {
            exact(item, "sampler", "cbuf_slot", "handle_byte_offset", "kind", "comparison", "normalized_coordinates",
                  "sampler_index");
        }
        const auto handle = number(item, "handle_byte_offset");
        auto& handles = kind == "texture" ? texture_handles : sampler_handles;
        require(is_uint(m(item, "handle_byte_offset"), 32764) && handle % 4 == 0 &&
                    admits_texture_slot(m(item, "cbuf_slot"), handle, 1, programs) &&
                    handles.insert(handle).second,
                "texture provenance reflection is invalid");
        if (kind == "texture") {
            require(one_of(text(item, "dimension"), {"d1", "d2", "d3", "cube", "buffer"}) &&
                        one_of(text(item, "scalar"), {"u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64",
                                                       "f16", "f32", "f64", "boolean"}) &&
                        one_of(text(item, "component_format"),
                               {"unorm", "snorm", "unsigned_integer", "signed_integer", "floating", "depth"}) &&
                        m(item, "array").is_bool() && m(item, "multisample").is_bool() &&
                        m(item, "normalized_coordinates").is_bool() && m(item, "comparison").is_bool() &&
                        is_uint(m(item, "texture_index"), 15) && textures.insert(number(item, "texture_index")).second,
                    "texture reflection is invalid");
        } else {
            require(m(item, "comparison").is_bool() && m(item, "normalized_coordinates").is_bool() &&
                        is_uint(m(item, "sampler_index"), 15) && samplers.insert(number(item, "sampler_index")).second,
                    "sampler reflection is invalid");
        }
    }
}

std::optional<std::uint64_t> validate_argument_buffer_v3(const Json& value) {
    if (value.is_null()) return std::nullopt;
    exact(value, "argument_buffer", "abi_version", "buffer_index", "byte_extent", "alignment");
    require(is_uint(m(value, "abi_version"), 1) && number(value, "abi_version") == 1 &&
                is_uint(m(value, "buffer_index"), 30) && is_uint(m(value, "byte_extent"), 0xffffffff) &&
                number(value, "byte_extent") > 0 && number(value, "byte_extent") % 8 == 0 &&
                is_uint(m(value, "alignment"), 8) && number(value, "alignment") == 8,
            "argument-buffer layout is invalid");
    return number(value, "buffer_index");
}

void validate_textures_v3(const Json& items, const Json& argument_buffer, std::size_t programs) {
    require(items.is_array() && items.array().size() <= 4096, "textures must be an array of at most 4096 entries");
    std::set<std::uint64_t> resource_ids;
    for (const Json& item : items.array()) {
        require(item.is_object() && is_uint(m(item, "resource_id"), 0xffffffff) &&
                    resource_ids.insert(number(item, "resource_id")).second,
                "resource IDs must be unique uint32 values");
    }
    std::map<std::string, std::vector<std::pair<std::uint64_t, std::uint64_t>>> direct_ranges;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> member_ranges;
    const bool indirect = !argument_buffer.is_null();
    for (const Json& item : items.array()) {
        const std::string& kind = text(item, "kind");
        require(one_of(kind, {"texture", "sampler"}), "texture kind is unsupported");
        if (kind == "texture") {
            exact(item, "texture", "cbuf_slot", "handle_byte_offset", "kind", "resource_id", "array_length", "location",
                  "dimension", "array", "multisample", "scalar", "component_format", "normalized_coordinates",
                  "comparison", "resource_kind", "access");
        } else {
            exact(item, "sampler", "cbuf_slot", "handle_byte_offset", "kind", "resource_id", "array_length", "location",
                  "comparison", "normalized_coordinates");
        }
        const auto length = number(item, "array_length"), offset = number(item, "handle_byte_offset");
        require(is_uint(m(item, "handle_byte_offset"), 65532) && offset % 4 == 0 &&
                    is_uint(m(item, "array_length"), 0xffffffff) && length > 0 && length <= 16384 &&
                    offset + length * 4 <= 65536 &&
                    admits_texture_slot(m(item, "cbuf_slot"), offset, length, programs),
                "texture provenance reflection is invalid");
        require(m(item, "comparison").is_bool() && m(item, "normalized_coordinates").is_bool(),
                kind + " reflection is invalid");
        if (kind == "texture") {
            require(one_of(text(item, "dimension"), {"d1", "d2", "d3", "cube", "buffer"}) &&
                        one_of(text(item, "scalar"), {"u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64",
                                                       "f16", "f32", "f64", "boolean"}) &&
                        one_of(text(item, "component_format"),
                               {"unorm", "snorm", "unsigned_integer", "signed_integer", "floating", "depth"}) &&
                        m(item, "array").is_bool() && m(item, "multisample").is_bool() &&
                        one_of(text(item, "resource_kind"), {"sampled", "storage_image"}) &&
                        one_of(text(item, "access"), {"read", "write", "read_write"}) &&
                        (text(item, "resource_kind") != "sampled" || text(item, "access") == "read"),
                    "texture reflection is invalid");
            Json::Object descriptor;
            for (const std::string_view name : {"dimension", "array", "multisample", "scalar", "component_format",
                                                "normalized_coordinates", "comparison"}) {
                descriptor.emplace(name, m(item, name));
            }
            texture_descriptor_value(Json{std::move(descriptor)}, "texture");
        }
        const Json& location = m(item, "location");
        require(location.is_object() && find_member(location, "kind") != nullptr, "resource location is invalid");
        if (!indirect) {
            exact(location, "resource location", "kind", "namespace", "first", "count");
            const std::string expected_namespace = kind;
            const auto first = number(location, "first"), count = number(location, "count");
            const std::uint64_t capacity = kind == "texture" ? 128 : 16;
            require(text(location, "kind") == "direct" && text(location, "namespace") == expected_namespace &&
                        is_uint(m(location, "first"), 0xffffffff) && is_uint(m(location, "count"), 0xffffffff) &&
                        count == length && first + count <= capacity,
                    "direct resource location is invalid");
            auto& ranges = direct_ranges[expected_namespace];
            require(std::all_of(ranges.begin(), ranges.end(), [&](const auto& range) {
                        return first + count <= range.first || first >= range.second;
                    }),
                    "direct resource locations overlap");
            ranges.emplace_back(first, first + count);
        } else {
            exact(location, "resource location", "kind", "byte_offset", "count", "stride");
            const auto byte_offset = number(location, "byte_offset");
            require(text(location, "kind") == "argument_buffer_member" && is_uint(m(location, "byte_offset"), 0xffffffff) &&
                        byte_offset % 8 == 0 && is_uint(m(location, "count"), 0xffffffff) &&
                        number(location, "count") == length && is_uint(m(location, "stride"), 8) &&
                        number(location, "stride") == 8,
                    "argument-buffer member location is invalid");
            const auto member_end = byte_offset + length * 8;
            require(member_end <= 0xffffffff && member_end <= number(argument_buffer, "byte_extent") &&
                        std::all_of(member_ranges.begin(), member_ranges.end(), [&](const auto& range) {
                            return member_end <= range.first || byte_offset >= range.second;
                        }),
                    "argument-buffer member exceeds its layout");
            member_ranges.emplace_back(byte_offset, member_end);
        }
    }
    if (indirect) {
        std::sort(member_ranges.begin(), member_ranges.end());
        std::uint64_t cursor = 0;
        for (const auto& [start, end] : member_ranges) {
            require(start == cursor, "argument-buffer members must be dense");
            cursor = end;
        }
        require(cursor == number(argument_buffer, "byte_extent"),
                "argument-buffer extent does not match its dense members");
    }
}

void validate_vertex_streams(const Json& items) {
    require(items.is_array() && items.array().size() <= 32, "vertex_streams must be an array of at most 32 entries");
    std::set<std::uint64_t> slots, attributes;
    for (const Json& item : items.array()) {
        exact(item, "vertex stream", "slot", "attribute_index", "format", "components", "byte_stride");
        require(is_uint(m(item, "slot"), 31) && slots.insert(number(item, "slot")).second &&
                    is_uint(m(item, "attribute_index"), 30) && attributes.insert(number(item, "attribute_index")).second &&
                    one_of(text(item, "format"), {"u8", "s8", "unorm8", "snorm8", "u16", "s16", "unorm16",
                                                  "snorm16", "f16", "u32", "s32", "f32"}) &&
                    is_uint(m(item, "components"), 4) && number(item, "components") != 0 &&
                    is_uint(m(item, "byte_stride"), 64) && number(item, "byte_stride") != 0,
                "vertex stream reflection is invalid");
    }
}

void validate_dependencies(const Json& items, std::size_t programs) {
    require(items.is_array() && items.array().size() <= 1024, "dependencies must be an array of at most 1024 entries");
    std::set<std::tuple<std::string, std::uint64_t, std::uint64_t, std::uint64_t>> keys;
    for (const Json& item : items.array()) {
        exact(item, "dependency", "kind", "index", "byte_offset", "component", "origin", "value");
        require(one_of(text(item, "kind"), {"local_memory_bytes", "constant_buffer_bound_mask", "constant_buffer_size",
                                             "constant_buffer_word", "texture_handle_cbuf", "texture_descriptor",
                                             "texture_array_length", "vertex_attribute_format", "fragment_output_type",
                                             "input_interpolation", "topology", "depth_convention", "origin_convention",
                                             "negate_y", "alpha_test", "early_z", "alpha_to_coverage", "dual_source_blend",
                                             "draw_parameters", "compute_local_size", "compute_shared_memory_bytes"}) &&
                    one_of(text(item, "origin"), {"validated_header", "captured_state", "explicit_option", "derived"}) &&
                    is_uint(m(item, "index"), 0xffffffff) && is_uint(m(item, "byte_offset"), 0xffffffff) &&
                    is_uint(m(item, "component"), 3),
                "dependency reflection is invalid");
        if (text(item, "kind") == "texture_descriptor") {
            const auto offset = number(item, "byte_offset");
            require(is_uint(m(item, "byte_offset"), 32764) && offset % 4 == 0 && number(item, "component") == 0 &&
                        admits_texture_slot(m(item, "index"), offset, 1, programs),
                    "texture descriptor dependency key is invalid");
            texture_descriptor_value(m(item, "value"), "texture descriptor dependency");
        }
        require(keys.insert({text(item, "kind"), number(item, "index"), number(item, "byte_offset"),
                             number(item, "component")}).second,
                "dependencies must be deduplicated");
    }
}

void validate_texture_descriptor_dependencies(const Json& dependencies, const Json& textures) {
    for (const Json& dependency : dependencies.array()) {
        if (text(dependency, "kind") != "texture_descriptor") continue;
        const Json* found = nullptr;
        for (const Json& texture : textures.array()) {
            if (text(texture, "kind") == "texture" && number(texture, "cbuf_slot") == number(dependency, "index") &&
                number(texture, "handle_byte_offset") == number(dependency, "byte_offset")) {
                found = &texture;
                break;
            }
        }
        require(found != nullptr, "texture descriptor dependency names no bound texture");
        for (const std::string_view name : {"dimension", "array", "multisample", "scalar", "component_format",
                                            "normalized_coordinates", "comparison"}) {
            require(m(m(dependency, "value"), name) == m(*found, name),
                    "texture descriptor dependency disagrees with the bound texture");
        }
    }
}

void validate_resource_dependencies_v3(const Json& dependencies, const Json& resources) {
    for (const Json& dependency : dependencies.array()) {
        const std::string& kind = text(dependency, "kind");
        if (kind != "texture_descriptor" && kind != "texture_array_length") continue;
        std::vector<const Json*> entries;
        for (const Json& resource : resources.array()) {
            if (number(resource, "cbuf_slot") == number(dependency, "index") &&
                number(resource, "handle_byte_offset") == number(dependency, "byte_offset")) {
                entries.push_back(&resource);
            }
        }
        require(!entries.empty() && number(dependency, "component") == 0, "resource dependency names no bound resource");
        if (kind == "texture_descriptor") {
            for (const Json* entry : entries) {
                const std::vector<std::string_view> fields = text(*entry, "kind") == "texture"
                    ? std::vector<std::string_view>{"dimension", "array", "multisample", "scalar", "component_format",
                                                    "normalized_coordinates", "comparison"}
                    : std::vector<std::string_view>{"comparison", "normalized_coordinates"};
                for (const std::string_view name : fields) {
                    require(m(m(dependency, "value"), name) == m(*entry, name),
                            "resource descriptor dependency disagrees with its provenance");
                }
            }
        } else {
            const Json& value = m(dependency, "value");
            require(is_uint(value, 0xffffffff) && uint_value(value) > 0 &&
                        std::any_of(entries.begin(), entries.end(), [&](const Json* entry) {
                            return number(*entry, "array_length") == uint_value(value);
                        }) &&
                        std::all_of(entries.begin(), entries.end(), [&](const Json* entry) {
                            const auto length = number(*entry, "array_length");
                            return length == 1 || length == uint_value(value);
                        }),
                    "resource array-length dependency disagrees with its provenance");
        }
    }
}

void validate_vertex_inputs_v2(const Json& value, const Json& streams, std::string_view stage) {
    require(value.is_array() && value.array().size() <= 32, "vertex inputs are invalid");
    std::set<std::uint64_t> seen;
    for (const Json& item : value.array()) {
        exact(item, "vertex input", "location", "class");
        require(is_uint(m(item, "location"), 31) && seen.insert(number(item, "location")).second &&
                    one_of(text(item, "class"), {"float", "uint", "sint"}),
                "vertex input is invalid");
    }
    require(value.array().empty() || (stage == "vertex" && streams.array().empty()),
            "vertex inputs are only valid for vertex stages without stream formats");
}

std::set<std::uint64_t> validate_color_outputs(const Json& items, std::string_view label) {
    require(items.is_array() && items.array().size() <= 8,
            std::string(label) + " must be an array of at most 8 entries");
    std::set<std::uint64_t> locations;
    for (const Json& item : items.array()) {
        exact(item, label, "location", "component_mask", "scalar", "components");
        const auto components = number(item, "components"), mask = number(item, "component_mask");
        require(is_uint(m(item, "location"), 7) && locations.insert(number(item, "location")).second &&
                    is_uint(m(item, "component_mask"), 15) && mask != 0 &&
                    one_of(text(item, "scalar"), {"f32", "u32", "s32"}) && is_uint(m(item, "components"), 4) &&
                    components != 0 && (mask >> components) == 0,
                std::string(label) + " reflection is invalid");
    }
    return locations;
}

bool validate_interface_v2(const Json& value, std::string_view stage, const Json& streams, const Json& dependencies,
                           const Json* fragment_output_mode) {
    exact(value, "interface", "position_input_mask", "position_output_mask", "color_output_mask", "color_outputs",
          "writes_fragment_depth", "writes_sample_mask", "generic_inputs", "generic_outputs");
    require(is_uint(m(value, "position_input_mask"), 15) && is_uint(m(value, "position_output_mask"), 15) &&
                is_uint(m(value, "color_output_mask"), 0xffffffff),
            "interface masks are invalid");
    validate_generic_v2(m(value, "generic_inputs"), "generic_inputs");
    validate_generic_v2(m(value, "generic_outputs"), "generic_outputs");
    const auto color_locations = validate_color_outputs(m(value, "color_outputs"), "color_outputs");
    require(m(value, "writes_fragment_depth").is_bool() && m(value, "writes_sample_mask").is_bool(),
            "fragment depth and sample-mask writes must be Booleans");
    std::uint64_t summary = 0;
    for (const Json& item : m(value, "color_outputs").array()) {
        summary |= number(item, "component_mask") << (4 * number(item, "location"));
    }
    require(summary == number(value, "color_output_mask"), "color outputs disagree with the summary mask");
    if (stage == "compute") {
        require(number(value, "position_input_mask") == 0 && number(value, "position_output_mask") == 0 &&
                    number(value, "color_output_mask") == 0 && m(value, "color_outputs").array().empty() &&
                    !boolean(value, "writes_fragment_depth") && !boolean(value, "writes_sample_mask") &&
                    m(value, "generic_inputs").array().empty() && m(value, "generic_outputs").array().empty(),
                "compute interface has graphics inputs or outputs");
        return false;
    }
    if (stage == "vertex") {
        // A position output may declare only some components; the stage gives
        // the rest the hardware default (0, 0, 0, 1).
        require(number(value, "position_input_mask") == 0 && number(value, "position_output_mask") != 0 &&
                    number(value, "color_output_mask") == 0 && m(value, "color_outputs").array().empty() &&
                    !boolean(value, "writes_fragment_depth") && !boolean(value, "writes_sample_mask") &&
                    m(value, "generic_inputs").array().size() == streams.array().size(),
                "vertex interface is outside the supported profile");
        for (const Json& stream : streams.array()) {
            const bool has_slot = find_member(stream, "slot") != nullptr;
            const auto slot = number(stream, has_slot ? "slot" : "location");
            const Json* input = nullptr;
            for (const Json& candidate : m(value, "generic_inputs").array()) {
                if (number(candidate, "slot") == slot) input = &candidate;
            }
            require(input != nullptr && (!has_slot || (number(*input, "component_mask") & ~7ull) == 0),
                    "vertex streams do not cover the reflected inputs");
        }
    } else {
        const bool perspective = std::any_of(
            m(value, "generic_inputs").array().begin(), m(value, "generic_inputs").array().end(),
            [](const Json& item) { return text(item, "interpolation") == "perspective"; });
        require(number(value, "position_output_mask") == 0 && m(value, "generic_outputs").array().empty() &&
                    (number(value, "position_input_mask") == 0 || number(value, "position_input_mask") == 8),
                "fragment interface is outside the supported profile");
        std::map<std::uint64_t, const Json*> facts;
        for (const Json& item : dependencies.array()) {
            if (text(item, "kind") == "fragment_output_type") {
                require(number(item, "byte_offset") == 0 && number(item, "component") == 0 &&
                            facts.emplace(number(item, "index"), &item).second,
                        "fragment_output_type dependencies must be unique per target");
            }
        }
        require(facts.size() == color_locations.size() &&
                    std::all_of(facts.begin(), facts.end(), [&](const auto& fact) { return color_locations.contains(fact.first); }),
                "written colour targets and fragment_output_type facts disagree");
        for (const Json& item : m(value, "color_outputs").array()) {
            const Json& value_fact = m(*facts.at(number(item, "location")), "value");
            Json::Object expected;
            expected.emplace("scalar", m(item, "scalar"));
            expected.emplace("components", m(item, "components"));
            require(value_fact == Json{std::move(expected)}, "colour output disagrees with its fragment_output_type fact");
            if (fragment_output_mode != nullptr && fragment_output_mode->string() == "render-target-v1") {
                require(text(item, "scalar") == "u32" && number(item, "components") == 4,
                        "render-target-v1 requires uint4 raw colour outputs");
            }
        }
        return perspective;
    }
    return false;
}

std::string validate_reflection_v2(const Json& metadata, std::span<const std::byte> program,
                                   std::span<const std::byte> source) {
    const bool v3 = text(metadata, "schema") == "armigo-shader-stage-v3";
    std::set<std::string, std::less<>> required{"schema", "source_version", "program_key", "program_size", "program_file",
        "source_file", "source_digest", "stage", "entry_point", "vertex_depth", "fragment_position", "c1_digest", "profile",
        "constant_buffers", "support_buffers", "textures", "vertex_streams", "vertex_inputs", "interface", "dependencies"};
    if (v3) required.insert("argument_buffer");
    std::set<std::string, std::less<>> optional{"fragment_output_mode", "depth_bias_mode", "y_direction", "point_size",
                                                "point_coord", "handle_programs"};
    if (v3) optional.insert({"storage_buffers", "custom_border", "draw_support", "vertex_fetch", "compute"});
    require(metadata.is_object(), "reflection has an unexpected schema");
    for (const auto& [key, value] : metadata.object()) {
        require(required.contains(key) || optional.contains(key), "reflection has an unexpected schema");
    }
    for (const auto& key : required) require(metadata.object().contains(key), "reflection has an unexpected schema");
    require(one_of(text(metadata, "schema"), {"armigo-shader-stage-v2", "armigo-shader-stage-v3"}),
            "reflection schema is unsupported");
    for (const std::string_view name : {"source_version", "program_key", "source_digest"}) {
        require_hex_digest(m(metadata, name), "reflection " + std::string(name) + " is invalid");
    }
    const Json& c1 = m(metadata, "c1_digest");
    require(c1.is_null() || (c1.is_string() && (require_hex_digest(c1, "reflection c1_digest is invalid"), true)),
            "reflection c1_digest is invalid");
    require(text(metadata, "program_file") == "program.bin" && text(metadata, "source_file") == "source.metal",
            "reflection filenames are invalid");
    require(number(metadata, "program_size") == program.size() && text(metadata, "program_key") == sha256(program),
            "program identity does not match reflection");
    require(text(metadata, "source_digest") == sha256(source), "source digest does not match reflection");
    const std::string& stage = text(metadata, "stage");
    require(one_of(stage, {"vertex", "fragment"}) || (v3 && stage == "compute"), "reflection stage is unsupported");
    require((find_member(metadata, "compute") != nullptr) == (stage == "compute"),
            "compute launch reflection does not match the stage");
    const Json* fragment_output_mode = find_member(metadata, "fragment_output_mode");
    if (fragment_output_mode != nullptr) {
        require(stage == "fragment" && fragment_output_mode->is_string() &&
                    fragment_output_mode->string() == "render-target-v1",
                "fragment output mode is invalid");
    }
    if (const Json* mode = find_member(metadata, "depth_bias_mode")) {
        require(stage == "fragment" && mode->is_string() && mode->string() == "shader-v1", "depth bias mode is invalid");
    }
    if (const Json* direction = find_member(metadata, "y_direction")) {
        require(direction->is_string() && direction->string() == "function-constant-v1", "y-direction mode is invalid");
    }
    if (const Json* size = find_member(metadata, "point_size")) {
        require(stage == "vertex" && size->is_string() && one_of(size->string(), {"fixed-one-v1", "program-v1"}),
                "point-size mode is invalid");
    }
    if (const Json* coord = find_member(metadata, "point_coord")) {
        require(stage == "fragment" && coord->is_string() && coord->string() == "attribute-v1",
                "point-coordinate mode is invalid");
    }
    require(program.size() >= prefix_size, "program is shorter than the identity prefix");
    require(program.size() <= maximum_program_size, "program exceeds the identity size policy");
    if (stage != "compute") {
        std::uint32_t sph = 0;
        std::memcpy(&sph, program.data(), sizeof sph);
        require(((sph >> 10) & 15) == (stage == "vertex" ? 1 : 5), "program SPH stage mismatch");
    }
    validate_profile(m(metadata, "profile"));
    if (v3) {
        Json::Object expected_profile;
        expected_profile.emplace("identity", Json{std::string("armigo-macos-metal4")});
        expected_profile.emplace("version", Json{JsonNumber{"3"}});
        require(m(metadata, "profile") == Json{std::move(expected_profile)}, "reflection host profile is invalid");
    }
    if (stage == "vertex") {
        require(text(metadata, "entry_point") == "armigo_vertex_main" &&
                    one_of(text(metadata, "vertex_depth"), {"minus-one-to-one", "zero-to-one"}) &&
                    text(metadata, "fragment_position") == "unspecified",
                "vertex explicit options are invalid");
    } else if (stage == "compute") {
        const Json& launch = m(metadata, "compute");
        exact(launch, "compute launch reflection", "local_size", "shared_memory_bytes");
        const Json& local_size = m(launch, "local_size");
        require(local_size.is_array() && local_size.array().size() == 3 &&
                    std::all_of(local_size.array().begin(), local_size.array().end(), [](const Json& value) {
                        return is_uint(value, 1024) && uint_value(value) >= 1;
                    }) && is_uint(m(launch, "shared_memory_bytes"), 49152),
                "compute launch reflection is invalid");
        require(text(metadata, "entry_point") == "armigo_compute_main" && text(metadata, "vertex_depth") == "unspecified" &&
                    text(metadata, "fragment_position") == "unspecified" && m(metadata, "vertex_streams").array().empty() &&
                    m(metadata, "vertex_inputs").array().empty(),
                "compute explicit options are invalid");
    } else {
        require(text(metadata, "entry_point") == "armigo_fragment_main" && text(metadata, "vertex_depth") == "unspecified" &&
                    one_of(text(metadata, "fragment_position"), {"unspecified", "reciprocal-clip-w"}) &&
                    m(metadata, "vertex_streams").array().empty(),
                "fragment explicit options are invalid");
    }
    const auto constant_indices = validate_constant_buffers_v2(m(metadata, "constant_buffers"));
    validate_support_buffers_v2(m(metadata, "support_buffers"), constant_indices);
    static const Json empty_array{Json::Array{}};
    const Json& storage_buffers = v3 && find_member(metadata, "storage_buffers") ? m(metadata, "storage_buffers") : empty_array;
    const auto storage_indices = v3 ? validate_storage_buffers_v3(storage_buffers, m(metadata, "constant_buffers"))
                                    : std::set<std::uint64_t>{};
    const auto argument_index = v3 ? validate_argument_buffer_v3(m(metadata, "argument_buffer")) : std::nullopt;
    const auto programs = validate_handle_programs(metadata);
    if (v3) validate_textures_v3(m(metadata, "textures"), m(metadata, "argument_buffer"), programs);
    else validate_textures_v2(m(metadata, "textures"), programs);
    validate_vertex_streams(m(metadata, "vertex_streams"));
    validate_vertex_inputs_v2(m(metadata, "vertex_inputs"), m(metadata, "vertex_streams"), stage);
    std::set<std::uint64_t> vertex_indices;
    for (std::size_t index = 0; index < m(metadata, "vertex_streams").array().size(); ++index) vertex_indices.insert(18 + index);
    require(std::all_of(vertex_indices.begin(), vertex_indices.end(), [](std::uint64_t index) { return index < 31; }),
            "vertex stream buffer indices exceed the Metal buffer namespace");
    std::set<std::uint64_t> support_indices;
    for (const Json& item : m(metadata, "support_buffers").array()) support_indices.insert(number(item, "metal_buffer_index"));
    std::set<std::uint64_t> reserved = constant_indices;
    reserved.insert(support_indices.begin(), support_indices.end());
    reserved.insert(storage_indices.begin(), storage_indices.end());
    reserved.insert(vertex_indices.begin(), vertex_indices.end());
    if (const Json* support = find_member(metadata, "draw_support")) {
        exact(*support, "draw support", "schema", "version", "byte_extent", "alignment", "metal_buffer_index");
        const auto index = number(*support, "metal_buffer_index");
        require(stage == "vertex" && text(*support, "schema") == "armigo-draw-parameters" &&
                    is_uint(m(*support, "version"), 1) && number(*support, "version") == 1 &&
                    is_uint(m(*support, "byte_extent"), 4) && number(*support, "byte_extent") == 4 &&
                    is_uint(m(*support, "alignment"), 4) && number(*support, "alignment") == 4 &&
                    is_uint(m(*support, "metal_buffer_index"), 30),
                "draw support ABI is invalid");
        require(!reserved.contains(index), "draw support overlaps a reflected buffer");
        reserved.insert(index);
    }
    if (const Json* fetch = find_member(metadata, "vertex_fetch")) {
        exact(*fetch, "vertex fetch", "schema", "version", "byte_extent", "alignment", "metal_buffer_index");
        const auto index = number(*fetch, "metal_buffer_index");
        require(stage == "vertex" && text(*fetch, "schema") == "armigo-vertex-fetch" &&
                    is_uint(m(*fetch, "version"), 1) && number(*fetch, "version") == 1 &&
                    is_uint(m(*fetch, "byte_extent"), 384) && number(*fetch, "byte_extent") == 384 &&
                    is_uint(m(*fetch, "alignment"), 8) && number(*fetch, "alignment") == 8 &&
                    is_uint(m(*fetch, "metal_buffer_index"), 30) && vertex_indices.empty(),
                "vertex fetch ABI is invalid");
        require(!reserved.contains(index), "vertex fetch table overlaps a reflected buffer");
        reserved.insert(index);
    }
    require(std::none_of(storage_indices.begin(), storage_indices.end(), [&](std::uint64_t index) {
                return constant_indices.contains(index) || support_indices.contains(index) || vertex_indices.contains(index);
            }),
            "storage buffer overlaps a reserved buffer index");
    if (argument_index) {
        require(!reserved.contains(*argument_index) &&
                    !(find_member(metadata, "depth_bias_mode") && *argument_index == 30),
                "argument buffer overlaps a reserved buffer index");
    }
    if (find_member(metadata, "depth_bias_mode")) {
        require(!reserved.contains(30), "depth bias buffer overlaps a reflected buffer");
    }
    if (const Json* border = find_member(metadata, "custom_border")) {
        exact(*border, "custom border support", "abi", "buffer_index", "pairs");
        const auto index = number(*border, "buffer_index");
        require(one_of(stage, {"fragment", "vertex"}) && text(*border, "abi") == "coverage-v1" &&
                    is_uint(m(*border, "buffer_index"), 30),
                "custom border support ABI is invalid");
        require(!reserved.contains(index) && (!argument_index || index != *argument_index) &&
                    !(find_member(metadata, "depth_bias_mode") && index == 30),
                "custom border support overlaps a reflected buffer");
        const Json& pairs = m(*border, "pairs");
        require(pairs.is_array() && !pairs.array().empty() && pairs.array().size() <= 4096, "custom border pairs are invalid");
        std::set<std::uint64_t> seen;
        for (const Json& pair : pairs.array()) {
            exact(pair, "custom border pair", "texture", "sampler");
            require(is_uint(m(pair, "texture"), 0xffffffff) && is_uint(m(pair, "sampler"), 0xffffffff),
                    "custom border pair is invalid");
            const Json* texture = nullptr;
            const Json* sampler = nullptr;
            for (const Json& resource : m(metadata, "textures").array()) {
                if (number(resource, "resource_id") == number(pair, "texture")) texture = &resource;
                if (number(resource, "resource_id") == number(pair, "sampler")) sampler = &resource;
            }
            require(texture != nullptr && sampler != nullptr && text(*texture, "kind") == "texture" &&
                        text(*texture, "resource_kind") == "sampled" && text(*texture, "dimension") == "d2" &&
                        !boolean(*texture, "array") && !boolean(*texture, "multisample") && text(*texture, "scalar") == "f32" &&
                        text(*texture, "component_format") != "depth" && text(*sampler, "kind") == "sampler" &&
                        number(*texture, "array_length") == 1 && number(*sampler, "array_length") == 1 &&
                        boolean(*texture, "normalized_coordinates") && boolean(*sampler, "normalized_coordinates") &&
                        !boolean(*texture, "comparison") && !boolean(*sampler, "comparison") &&
                        number(*texture, "cbuf_slot") == number(*sampler, "cbuf_slot") &&
                        number(*texture, "handle_byte_offset") == number(*sampler, "handle_byte_offset") &&
                        seen.insert(number(pair, "texture")).second,
                    "custom border pair is outside the admitted sampling profile");
        }
    }
    require(std::none_of(vertex_indices.begin(), vertex_indices.end(), [&](std::uint64_t index) {
                return constant_indices.contains(index) || support_indices.contains(index);
            }),
            "constant/support buffer indices overlap vertex streams");
    validate_dependencies(m(metadata, "dependencies"), programs);
    for (const Json& item : m(metadata, "dependencies").array()) {
        if (text(item, "kind") == "compute_local_size") {
            require(stage == "compute" && m(item, "value") == m(m(metadata, "compute"), "local_size"),
                    "compute local size dependency disagrees with the launch reflection");
        }
        if (text(item, "kind") == "compute_shared_memory_bytes") {
            require(stage == "compute" && m(item, "value") == m(m(metadata, "compute"), "shared_memory_bytes"),
                    "compute shared memory dependency disagrees with the launch reflection");
        }
    }
    if (v3) validate_resource_dependencies_v3(m(metadata, "dependencies"), m(metadata, "textures"));
    else validate_texture_descriptor_dependencies(m(metadata, "dependencies"), m(metadata, "textures"));
    const Json& streams = m(metadata, "vertex_streams").array().empty() ? m(metadata, "vertex_inputs")
                                                                       : m(metadata, "vertex_streams");
    const bool perspective = validate_interface_v2(m(metadata, "interface"), stage, streams, m(metadata, "dependencies"),
                                                   fragment_output_mode);
    if (find_member(metadata, "depth_bias_mode")) {
        require(!boolean(m(metadata, "interface"), "writes_fragment_depth"),
                "shader depth bias cannot replace a guest fragment-depth export");
    }
    if (stage == "fragment") {
        const bool needs = number(m(metadata, "interface"), "position_input_mask") == 8 || perspective;
        require((needs && text(metadata, "fragment_position") == "reciprocal-clip-w") ||
                    (!needs && text(metadata, "fragment_position") == "unspecified"),
                "fragment position option does not match its interface");
    }
    return stage;
}

std::string validate_reflection_compute(const Json& metadata, std::span<const std::byte> program,
                                        std::span<const std::byte> source) {
    exact(metadata, "reflection", "schema", "source_version", "program_key", "program_size", "program_file",
          "source_file", "source_digest", "stage", "entry_point", "local_size", "shared_memory_bytes", "profile",
          "storage_buffers", "textures");
    require(text(metadata, "schema") == "armigo-shader-stage-compute-v1", "reflection schema is unsupported");
    for (const std::string_view name : {"source_version", "program_key", "source_digest"}) {
        require_hex_digest(m(metadata, name), "reflection " + std::string(name) + " is invalid");
    }
    require(text(metadata, "program_file") == "program.bin" && text(metadata, "source_file") == "source.metal",
            "reflection filenames are invalid");
    require(number(metadata, "program_size") == program.size() && text(metadata, "program_key") == sha256(program),
            "program identity does not match reflection");
    require(text(metadata, "source_digest") == sha256(source), "source digest does not match reflection");
    require(text(metadata, "stage") == "compute" && text(metadata, "entry_point") == "armigo_compute_main",
            "compute reflection stage or entry point is invalid");
    Json::Object expected_profile;
    expected_profile.emplace("identity", Json{std::string("armigo-macos-metal4")});
    expected_profile.emplace("version", Json{JsonNumber{"3"}});
    require(m(metadata, "profile") == Json{std::move(expected_profile)}, "compute host profile is invalid");
    const Json& local_size = m(metadata, "local_size");
    require(local_size.is_array() && local_size.array().size() == 3 &&
                std::all_of(local_size.array().begin(), local_size.array().end(), [](const Json& value) {
                    return is_uint(value, 65535) && uint_value(value) >= 1;
                }),
            "compute local size is invalid");
    require(is_uint(m(metadata, "shared_memory_bytes"), 32768), "compute shared memory size is invalid");
    const Json& storage = m(metadata, "storage_buffers");
    require(storage.is_array() && storage.array().size() <= 31, "storage_buffers must be an array");
    std::set<std::uint64_t> metals;
    std::set<std::pair<std::uint64_t, std::uint64_t>> descriptors;
    for (const Json& item : storage.array()) {
        exact(item, "storage buffer", "cbuf_slot", "descriptor_byte_offset", "metal_buffer_index", "access", "alignment");
        const auto slot = number(item, "cbuf_slot"), offset = number(item, "descriptor_byte_offset");
        const auto metal = number(item, "metal_buffer_index");
        require(is_uint(m(item, "cbuf_slot"), 3) && is_uint(m(item, "descriptor_byte_offset"), 65524) && offset % 8 == 0 &&
                    is_uint(m(item, "metal_buffer_index"), 30) && metal >= 8 && metals.insert(metal).second &&
                    descriptors.insert({slot, offset}).second && one_of(text(item, "access"), {"read", "write", "read_write"}) &&
                    is_uint(m(item, "alignment"), 4) && number(item, "alignment") == 4,
                "compute storage-buffer reflection is invalid");
    }
    const Json& textures = m(metadata, "textures");
    require(textures.is_array() && textures.array().size() <= 32, "textures must be an array of at most 32 entries");
    std::set<std::pair<std::uint64_t, std::uint64_t>> texture_handles, sampler_handles;
    for (const Json& item : textures.array()) {
        require(item.is_object() && find_member(item, "kind"), "texture entry is invalid");
        const std::string& kind = text(item, "kind");
        require(one_of(kind, {"texture", "sampler"}), "texture kind is unsupported");
        require(is_uint(m(item, "cbuf_slot"), 3) && is_uint(m(item, "handle_byte_offset"), 65532) &&
                    number(item, "handle_byte_offset") % 4 == 0,
                "compute texture handle location is invalid");
        if (kind == "texture") {
            exact(item, "texture", "cbuf_slot", "handle_byte_offset", "kind", "dimension", "array", "multisample",
                  "scalar", "component_format", "normalized_coordinates", "comparison", "texture_index");
            require(is_uint(m(item, "texture_index"), 15) &&
                        texture_handles.insert({number(item, "cbuf_slot"), number(item, "handle_byte_offset")}).second,
                    "compute texture binding is invalid");
        } else {
            exact(item, "sampler", "cbuf_slot", "handle_byte_offset", "kind", "comparison", "normalized_coordinates",
                  "sampler_index");
            require(is_uint(m(item, "sampler_index"), 15) &&
                        sampler_handles.insert({number(item, "cbuf_slot"), number(item, "handle_byte_offset")}).second,
                    "compute sampler binding is invalid");
        }
    }
    return "compute";
}

std::string validate_reflection(const Json& metadata, std::span<const std::byte> program,
                                std::span<const std::byte> source) {
    require(metadata.is_object(), "reflection has an unexpected schema");
    const Json* schema = find_member(metadata, "schema");
    require(schema != nullptr && schema->is_string(), "reflection schema is unsupported");
    if (schema->string() == "armigo-shader-stage-compute-v1") return validate_reflection_compute(metadata, program, source);
    if (schema->string() == "armigo-shader-stage-v2" || schema->string() == "armigo-shader-stage-v3") {
        return validate_reflection_v2(metadata, program, source);
    }
    return validate_reflection_v1(metadata, program, source);
}

class Hasher {
public:
    Hasher() { require(CC_SHA256_Init(&context_) == 1, "System SHA-256 failed"); }
    void update(std::span<const std::byte> bytes) {
        require(CC_SHA256_Update(&context_, bytes.data(), static_cast<CC_LONG>(bytes.size())) == 1,
                "System SHA-256 failed");
    }
    std::string finish() {
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
        require(CC_SHA256_Final(digest.data(), &context_) == 1, "System SHA-256 failed");
        static constexpr char digits[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (const unsigned char byte : digest) {
            result += digits[byte >> 4];
            result += digits[byte & 15];
        }
        return result;
    }
private:
    CC_SHA256_CTX context_{};
};

std::span<const std::byte> bytes(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

} // namespace

std::string sha256(std::span<const std::byte> bytes) {
    Hasher hasher;
    hasher.update(bytes);
    return hasher.finish();
}

std::string sha256_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "Cannot hash Apple shader tool");
    Hasher hasher;
    std::array<char, 64 * 1024> block{};
    while (stream) {
        stream.read(block.data(), static_cast<std::streamsize>(block.size()));
        const auto count = stream.gcount();
        if (count > 0) hasher.update({reinterpret_cast<const std::byte*>(block.data()), static_cast<std::size_t>(count)});
    }
    require(stream.eof(), "Cannot hash Apple shader tool");
    return hasher.finish();
}

std::vector<std::byte> read_regular(const fs::path& path, std::size_t maximum, std::string_view label) {
    std::error_code error;
    require(fs::is_regular_file(path, error) && !fs::is_symlink(path, error),
            std::string(label) + " must be a regular file");
    const auto size = fs::file_size(path, error);
    require(!error && size <= maximum, std::string(label) + " exceeds its size policy");
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    require(stream.good() && (data.empty() || stream.read(reinterpret_cast<char*>(data.data()),
                                                          static_cast<std::streamsize>(data.size())).good()),
            std::string(label) + " must be a regular file");
    require(data.size() <= maximum, std::string(label) + " changed beyond its size policy");
    return data;
}

void write_file(const fs::path& path, std::span<const std::byte> data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.good() && (data.empty() || stream.write(reinterpret_cast<const char*>(data.data()),
                                                           static_cast<std::streamsize>(data.size())).good()),
            "Cannot write " + path.string());
}

std::string variant_data(const Json& metadata) {
    const std::string& schema = text(metadata, "schema");
    Json::Object result;
    auto copy = [&](std::string_view name) { result.emplace(name, m(metadata, name)); };
    if (schema == "armigo-shader-stage-compute-v1") {
        for (const std::string_view name : {"schema", "program_key", "stage", "entry_point", "profile", "local_size",
                                            "shared_memory_bytes", "storage_buffers", "textures"}) copy(name);
    } else if (schema == "armigo-shader-stage-v2" || schema == "armigo-shader-stage-v3") {
        for (const std::string_view name : {"fragment_position", "program_key", "stage", "vertex_depth", "schema",
                                            "c1_digest", "profile", "vertex_streams", "constant_buffers", "support_buffers",
                                            "vertex_inputs"}) copy(name);
        Json::Array dependencies;
        for (const Json& item : m(metadata, "dependencies").array()) {
            Json::Object projected;
            for (const std::string_view name : {"kind", "index", "byte_offset", "component", "value"}) {
                projected.emplace(name, m(item, name));
            }
            dependencies.emplace_back(std::move(projected));
        }
        result.emplace("dependencies", std::move(dependencies));
        for (const std::string_view name : {"fragment_output_mode", "depth_bias_mode", "y_direction", "point_size",
                                            "point_coord", "custom_border", "draw_support", "vertex_fetch", "compute",
                                            "handle_programs"}) {
            if (const Json* value = find_member(metadata, name)) result.emplace(name, *value);
        }
        if (schema == "armigo-shader-stage-v3") {
            copy("argument_buffer");
            static const Json empty{Json::Array{}};
            result.emplace("storage_buffers", find_member(metadata, "storage_buffers") ? m(metadata, "storage_buffers") : empty);
            copy("textures");
        }
    } else {
        for (const std::string_view name : {"fragment_position", "program_key", "stage", "vertex_attributes", "vertex_depth"}) {
            copy(name);
        }
    }
    return canonical_json(Json{std::move(result)});
}

Stage load_stage(const fs::path& path) {
    std::error_code error;
    require(fs::is_directory(path, error), "--stage must name a directory");
    Stage result;
    result.root = path;
    result.program = read_regular(path / "program.bin", maximum_program_size, "program.bin");
    result.source = read_regular(path / "source.metal", maximum_source_size, "source.metal");
    result.reflection = read_regular(path / "reflection.json", maximum_reflection_size, "reflection.json");
    result.metadata = parse_json(result.reflection);
    result.stage = validate_reflection(result.metadata, result.program, result.source);
    result.program_key = sha256(result.program);
    result.prefix_key = sha256(std::span<const std::byte>(result.program).first(prefix_size));
    const std::string variant = variant_data(result.metadata);
    result.variant_id = sha256(bytes(variant));
    return result;
}

std::vector<Stage> validate_inventory(std::vector<Stage> stages) {
    std::sort(stages.begin(), stages.end(), [](const Stage& left, const Stage& right) { return left.variant_id < right.variant_id; });
    std::vector<Stage> unique;
    for (Stage& stage : stages) {
        if (!unique.empty() && unique.back().variant_id == stage.variant_id) {
            require(unique.back().program == stage.program && unique.back().source == stage.source &&
                        unique.back().reflection == stage.reflection,
                    "duplicate variant ID has conflicting contents");
        } else {
            unique.push_back(std::move(stage));
        }
    }
    require(unique.size() <= maximum_identities, "identity count exceeds the policy");
    std::vector<Stage*> by_stage;
    by_stage.reserve(unique.size());
    for (Stage& stage : unique) by_stage.push_back(&stage);
    std::sort(by_stage.begin(), by_stage.end(), [](const Stage* left, const Stage* right) {
        return std::tie(left->stage, left->program) < std::tie(right->stage, right->program);
    });
    for (std::size_t index = 1; index < by_stage.size(); ++index) {
        const Stage& left = *by_stage[index - 1];
        const Stage& right = *by_stage[index];
        if (left.stage == right.stage && left.program.size() < right.program.size()) {
            require(!std::equal(left.program.begin(), left.program.end(), right.program.begin()),
                    "same-stage program is a byte prefix of another");
        }
    }
    std::map<std::pair<std::string, std::string>, std::set<std::pair<std::string, std::size_t>>> buckets;
    for (const Stage& stage : unique) {
        auto& bucket = buckets[{stage.stage, stage.prefix_key}];
        bucket.insert({stage.program_key, stage.program.size()});
        require(bucket.size() <= maximum_bucket, "identity prefix bucket exceeds the policy");
    }
    return unique;
}

} // namespace armigo::shader::package::detail
