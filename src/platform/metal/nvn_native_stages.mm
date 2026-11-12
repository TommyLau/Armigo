// ============================================================================
// Armigo Platform - NVN API 231 Native Shader Stages
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// API 231 programs carry native Switch 2 stages. SetShaders records them, and
// at first use the native front end translates them into the shared
// pipeline's Metal artifact; they become stages through the same reflection a
// package stage holds. The
// reflection writer below repeats the ported translator's, so the Maxwell
// translation sources (and the shader source identity packages are built
// against) stay unchanged; tests/metal/nvn_native_stages.mm keeps the two in
// the same shape.

#include "nvn_native_stages.h"
#include "../../nvn/program_capture.h"

#include <armigo/common/diagnostics.h>
#include <armigo/nvn/driver.h>
#include <armigo/platform/metal_shader_package.h>
#include <armigo/shader/source_version.h>
#include <armigo/shader/vertex_fetch.h>
#include <armigo/shader_native/frontend.h>
#include <armigo/shader_native/package.h>
#include "../../nvn/entry_table_231.h"
#include "../../nvn/programs.h"
#include "pipeline/facts.h"
#include "sha256.h"
#include <mach/mach.h>
#include <pthread.h>
#include <unistd.h>
#include "../mach_vm.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::platform::metal {
namespace {
namespace gm = armigo::shader::graphics_msl;
namespace pl = armigo::shader::pipeline;
using Bytes = std::span<const std::byte>;
using armigo::shader::vertex_fetch_bytes;
using armigo::shader::vertex_fetch_schema;
using armigo::shader::vertex_fetch_version;

std::string quote(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20 || c >= 0x7f) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

Bytes text_bytes(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

std::string digest(Bytes bytes) {
    const auto hash = armigo::loader::detail::sha256(bytes);
    if (!hash) throw std::runtime_error("System SHA-256 failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : *hash) {
        const auto value = std::to_integer<unsigned>(byte);
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}

const char* depth_name(gm::VertexDepthConvention depth) {
    switch (depth) {
    case gm::VertexDepthConvention::NegativeOneToOne: return "minus-one-to-one";
    case gm::VertexDepthConvention::ZeroToOne: return "zero-to-one";
    case gm::VertexDepthConvention::Unspecified: return "unspecified";
    }
    throw std::runtime_error("Invalid emitted depth convention");
}

const char* scalar_name(pl::ScalarType type) {
    switch (type) {
    case pl::ScalarType::u8: return "u8";
    case pl::ScalarType::s8: return "s8";
    case pl::ScalarType::u16: return "u16";
    case pl::ScalarType::s16: return "s16";
    case pl::ScalarType::u32: return "u32";
    case pl::ScalarType::s32: return "s32";
    case pl::ScalarType::u64: return "u64";
    case pl::ScalarType::s64: return "s64";
    case pl::ScalarType::f16: return "f16";
    case pl::ScalarType::f32: return "f32";
    case pl::ScalarType::f64: return "f64";
    case pl::ScalarType::boolean: return "boolean";
    }
    return "unknown";
}

std::string fact_value_json(const pl::Dependency& dependency) {
    const auto& value = dependency.value;
    std::ostringstream out;
    if (const auto* number = std::get_if<std::uint32_t>(&value)) {
        out << *number;
    } else if (const auto* size = std::get_if<pl::ComputeLocalSize>(&value)) {
        out << '[' << size->dimensions[0] << ',' << size->dimensions[1] << ',' << size->dimensions[2] << ']';
    } else if (const auto* flag = std::get_if<bool>(&value)) {
        out << (*flag ? "true" : "false");
    } else if (const auto* format = std::get_if<pl::AttributeFormat>(&value)) {
        static constexpr const char* names[] = {"u8", "s8", "unorm8", "snorm8", "u16",
            "s16", "unorm16", "snorm16", "f16", "u32", "s32", "f32"};
        out << "{\"format\":" << quote(names[unsigned(format->format)])
            << ",\"components\":" << unsigned(format->components) << '}';
    } else if (const auto* type = std::get_if<pl::IoType>(&value)) {
        out << "{\"scalar\":" << quote(scalar_name(type->scalar))
            << ",\"components\":" << unsigned(type->components) << '}';
    } else if (const auto* interpolation = std::get_if<pl::Interpolation>(&value)) {
        static constexpr const char* modes[] = {"flat", "perspective", "linear"};
        static constexpr const char* locations[] = {"center", "centroid", "sample", "offset"};
        out << "{\"mode\":" << quote(modes[unsigned(interpolation->mode)])
            << ",\"location\":" << quote(locations[unsigned(interpolation->location)]) << '}';
    } else if (const auto* descriptor = std::get_if<pl::TextureDescriptor>(&value)) {
        static constexpr const char* dimensions[] = {"d1", "d2", "d3", "cube", "buffer"};
        static constexpr const char* formats[] = {"unorm", "snorm", "unsigned_integer",
            "signed_integer", "floating", "depth"};
        out << "{\"dimension\":" << quote(dimensions[unsigned(descriptor->shape.dimension)])
            << ",\"array\":" << (descriptor->shape.array ? "true" : "false")
            << ",\"multisample\":" << (descriptor->shape.multisample ? "true" : "false")
            << ",\"scalar\":" << quote(scalar_name(descriptor->scalar))
            << ",\"component_format\":" << quote(formats[unsigned(descriptor->format)])
            << ",\"normalized_coordinates\":" << (descriptor->normalized_coordinates ? "true" : "false")
            << ",\"comparison\":" << (descriptor->comparison ? "true" : "false") << '}';
    } else if (const auto* topology = std::get_if<pl::Topology>(&value)) {
        static constexpr const char* names[] = {"points", "lines", "triangles"};
        out << quote(names[unsigned(*topology)]);
    } else if (const auto* depth = std::get_if<pl::DepthConvention>(&value)) {
        out << quote(*depth == pl::DepthConvention::zero_to_one ? "zero-to-one"
                                                                : "minus-one-to-one");
    } else if (const auto* origin = std::get_if<pl::OriginConvention>(&value)) {
        out << quote(*origin == pl::OriginConvention::upper_left ? "upper-left" : "lower-left");
    } else if (const auto* alpha = std::get_if<pl::AlphaTest>(&value)) {
        static constexpr const char* compares[] = {"never", "less", "equal", "less_equal",
            "greater", "not_equal", "greater_equal", "always"};
        out << "{\"comparison\":" << quote(compares[unsigned(alpha->comparison)])
            << ",\"reference_bits\":" << alpha->reference_bits << '}';
    } else {
        throw std::runtime_error("Ported reflection cannot serialize this fact value");
    }
    return out.str();
}

const char* fact_kind_name(pl::FactKind kind) {
    switch (kind) {
    case pl::FactKind::local_memory_bytes: return "local_memory_bytes";
    case pl::FactKind::compute_local_size: return "compute_local_size";
    case pl::FactKind::compute_shared_memory_bytes: return "compute_shared_memory_bytes";
    case pl::FactKind::bound_storage_descriptor: return "bound_storage_descriptor";
    case pl::FactKind::constant_buffer_bound_mask: return "constant_buffer_bound_mask";
    case pl::FactKind::constant_buffer_size: return "constant_buffer_size";
    case pl::FactKind::constant_buffer_word: return "constant_buffer_word";
    case pl::FactKind::texture_handle_cbuf: return "texture_handle_cbuf";
    case pl::FactKind::texture_descriptor: return "texture_descriptor";
    case pl::FactKind::texture_array_length: return "texture_array_length";
    case pl::FactKind::vertex_attribute_format: return "vertex_attribute_format";
    case pl::FactKind::fragment_output_type: return "fragment_output_type";
    case pl::FactKind::input_interpolation: return "input_interpolation";
    case pl::FactKind::topology: return "topology";
    case pl::FactKind::depth_convention: return "depth_convention";
    case pl::FactKind::origin_convention: return "origin_convention";
    case pl::FactKind::negate_y: return "negate_y";
    case pl::FactKind::alpha_test: return "alpha_test";
    case pl::FactKind::early_z: return "early_z";
    case pl::FactKind::alpha_to_coverage: return "alpha_to_coverage";
    case pl::FactKind::dual_source_blend: return "dual_source_blend";
    case pl::FactKind::draw_parameters: return "draw_parameters";
    case pl::FactKind::count: break;
    }
    throw std::runtime_error("Ported reflection cannot name this fact kind");
}

void write_resource_location(std::ostream& out, const pl::BindingLocation& location) {
    if (const auto* direct = std::get_if<pl::MetalRange>(&location)) {
        static constexpr const char* namespaces[] = {"buffer", "texture", "sampler"};
        out << "{\"kind\":\"direct\",\"namespace\":" << quote(namespaces[unsigned(direct->space)])
            << ",\"first\":" << direct->first << ",\"count\":" << direct->count << '}';
    } else {
        const auto& member = std::get<pl::ArgumentBufferMember>(location);
        out << "{\"kind\":\"argument_buffer_member\",\"byte_offset\":" << member.byte_offset
            << ",\"count\":" << member.count << ",\"stride\":" << member.stride << '}';
    }
}

// A fragment stage depends on the position convention only when it reads
// position w or interpolates an input in perspective, so only then does its
// reflection record the convention (the stage maker admits it that way).
bool uses_position_convention(const pl::ResourcePlan& plan, pl::Stage stage) {
    if (stage != pl::Stage::fragment) return false;
    for (const auto& use : plan.io) {
        if (use.access == pl::Access::write) continue;
        if (use.semantic == pl::IoSemantic::position) return true;
        if (use.semantic == pl::IoSemantic::generic && use.interpolation.mode == pl::InterpolationMode::perspective)
            return true;
    }
    return false;
}

std::string reflection(const pl::MslArtifact& artifact, const gm::EmitOptions& options,
                       const pl::Stage stage, Bytes identity, const NativeMeshLaunch* launch) {
    const auto& plan = artifact.reflection;
    const char* stage_text = launch ? (launch->task ? "task" : "mesh")
        : stage == pl::Stage::vertex ? "vertex" : stage == pl::Stage::compute ? "compute" : "fragment";
    std::ostringstream out;
    out << "{\"schema\":\"armigo-shader-stage-v3\",\"source_version\":"
        << quote(armigo::shader::source_version)
        << ",\"program_key\":" << quote(digest(identity))
        << ",\"program_size\":" << identity.size()
        << ",\"program_file\":\"program.bin\",\"source_file\":\"source.metal\""
        << ",\"source_digest\":" << quote(digest(text_bytes(artifact.source)))
        << ",\"stage\":" << quote(stage_text)
        << ",\"entry_point\":" << quote(artifact.entry_point)
        << ",\"vertex_depth\":" << quote(depth_name(options.vertex_depth))
        << ",\"fragment_position\":"
        << quote(options.fragment_position == gm::FragmentPositionConvention::Unspecified ||
                         !uses_position_convention(plan, stage)
                     ? "unspecified" : "reciprocal-clip-w")
        << (stage == pl::Stage::fragment
                ? ",\"fragment_output_mode\":\"render-target-v1\""
                : "")
        // The ported pipeline reads only the SPH and program words; the trailing
        // C1 bytes are bound at run time, so they never select a variant.
        << ",\"c1_digest\":null";
    const bool consumes_y_direction = std::any_of(plan.io.begin(), plan.io.end(),
        [](const pl::IoUse& io) {
            return io.access == pl::Access::read && io.semantic == pl::IoSemantic::y_direction;
        });
    if (consumes_y_direction) out << ",\"y_direction\":\"function-constant-v1\"";
    // A vertex stage without its own point size writes the fixed size 1.0.
    if (plan.fixed_point_size) out << ",\"point_size\":\"fixed-one-v1\"";
    else if (std::any_of(plan.io.begin(), plan.io.end(), [](const pl::IoUse& io) {
                 return io.access == pl::Access::write && io.semantic == pl::IoSemantic::point_size;
             }))
        out << ",\"point_size\":\"program-v1\"";
    if (plan.reads_point_coord) out << ",\"point_coord\":\"attribute-v1\"";
    // A vertex stage without outputs draws only with rasterizer discard.
    if (plan.requires_rasterizer_discard) out << ",\"rasterizer_discard\":\"required-v1\"";
    if (!plan.discard_entry_point.empty())
        out << ",\"discard_entry_point\":" << quote(plan.discard_entry_point);
    if (plan.emulate_depth_bias) out << ",\"depth_bias_mode\":\"shader-v1\"";
    if (plan.custom_border) {
        out << ",\"custom_border\":{\"abi\":\"coverage-v1\",\"buffer_index\":"
            << plan.custom_border->buffer_index << ",\"pairs\":[";
        bool first_pair = true;
        for (const auto& pair : plan.custom_border->pairs) {
            if (!first_pair) out << ',';
            first_pair = false;
            out << "{\"texture\":" << pair.texture.value
                << ",\"sampler\":" << pair.sampler.value << '}';
        }
        out << "]}";
    }
    if (plan.draw_support_buffer_index) {
        // A fragment reading the render-target height takes version 2.
        const bool height = plan.reads_render_target_height;
        out << ",\"draw_support\":{\"schema\":" << quote(pl::draw_support_schema)
            << ",\"version\":" << (height ? pl::draw_support_height_version : pl::draw_support_version)
            << ",\"byte_extent\":" << (height ? pl::draw_support_height_bytes : pl::draw_support_bytes)
            << ",\"alignment\":4,\"metal_buffer_index\":"
            << *plan.draw_support_buffer_index << '}';
    }
    if (plan.vertex_fetch_buffer_index) {
        out << ",\"vertex_fetch\":{\"schema\":" << quote(vertex_fetch_schema)
            << ",\"version\":" << vertex_fetch_version
            << ",\"byte_extent\":" << vertex_fetch_bytes
            << ",\"alignment\":8,\"metal_buffer_index\":"
            << *plan.vertex_fetch_buffer_index << '}';
    }
    for (const auto& planned : plan.resources) {
        // The guest-memory window table travels as its own record.
        const auto* buffer = std::get_if<pl::BufferUse>(&planned.use);
        const auto* support = buffer ? std::get_if<pl::SupportBufferSource>(&buffer->source) : nullptr;
        if (!support || support->schema != pl::guest_memory_schema) continue;
        if (support->version != pl::guest_memory_version || buffer->size.bytes != pl::guest_memory_bytes ||
            buffer->access != pl::Access::read_write)
            throw std::runtime_error("Ported reflection has an unsupported guest-memory table");
        out << ",\"guest_memory\":{\"schema\":" << quote(support->schema)
            << ",\"version\":" << support->version
            << ",\"byte_extent\":" << buffer->size.bytes
            << ",\"alignment\":" << buffer->size.alignment
            << ",\"metal_buffer_index\":" << std::get<pl::MetalRange>(planned.location).first
            << ",\"tag\":" << plan.guest_memory_tag
            << ",\"writes\":" << (plan.guest_memory_writes ? "true" : "false") << '}';
    }
    // Present only when the stage has any: a texture keyed at slot 31 takes
    // program handle_byte_offset / 4.
    if (!plan.handle_programs.empty()) {
        out << ",\"handle_programs\":[";
        bool first_program = true;
        for (const auto& program : plan.handle_programs) {
            if (!first_program) out << ',';
            first_program = false;
            out << '[';
            for (std::size_t word = 0; word < program.size(); ++word) {
                if (word) out << ',';
                out << program[word];
            }
            out << ']';
        }
        out << ']';
    }
    out << ",\"profile\":{\"identity\":" << quote(plan.profile.identity)
        << ",\"version\":" << plan.profile.version << '}';
    out << ",\"argument_buffer\":";
    if (plan.argument_buffer) {
        const auto& layout = *plan.argument_buffer;
        out << "{\"abi_version\":" << layout.abi_version
            << ",\"buffer_index\":" << layout.buffer_index
            << ",\"byte_extent\":" << layout.byte_extent
            << ",\"alignment\":" << layout.alignment << '}';
    } else out << "null";
    if (launch) {
        out << ",\"mesh_launch\":{\"threads\":" << launch->threads << ",\"payload_bytes\":" << launch->payload_bytes;
        if (!launch->task)
            out << ",\"task\":" << (launch->launched_by_task ? "true" : "false")
                << ",\"primitive_vertices\":" << unsigned(launch->primitive_vertices)
                << ",\"max_vertices\":" << launch->max_vertices << ",\"max_primitives\":" << launch->max_primitives;
        out << '}';
    } else if (plan.compute) {
        const auto& size = plan.compute->local_size.dimensions;
        out << ",\"compute\":{\"local_size\":[" << size[0] << ',' << size[1] << ',' << size[2]
            << "],\"shared_memory_bytes\":" << plan.compute->shared_memory_bytes << '}';
    }
    out << ",\"constant_buffers\":[";
    bool first = true;
    for (const auto& planned : plan.resources) {
        const auto* buffer = std::get_if<pl::BufferUse>(&planned.use);
        if (!buffer || !std::holds_alternative<pl::ConstantBufferSource>(buffer->source)) continue;
        const bool proven = buffer->size.constant_buffer_policy == pl::ConstantBufferPolicy::proven_zero_words_v2;
        if (buffer->size.kind != pl::SizeKind::constant_buffer_window ||
            (!proven && buffer->size.constant_buffer_policy != pl::ConstantBufferPolicy::padded_zero_words_v2))
            throw std::runtime_error("Ported reflection requires a runtime constant-buffer window");
        if (!first) out << ',';
        first = false;
        out << "{\"guest_binding\":"
            << unsigned(std::get<pl::ConstantBufferSource>(buffer->source).slot)
            << ",\"size_kind\":\"constant_buffer_window\""
            << ",\"window_bytes\":" << buffer->size.bytes
            << ",\"alignment\":" << buffer->size.alignment
            << ",\"policy\":\"" << (proven ? "proven_zero_words_v2" : "padded_zero_words_v2") << '"'
            << ",\"metal_buffer_index\":" << std::get<pl::MetalRange>(planned.location).first << '}';
    }
    out << "],\"support_buffers\":[";
    first = true;
    for (const auto& planned : plan.resources) {
        const auto* buffer = std::get_if<pl::BufferUse>(&planned.use);
        if (!buffer) continue;
        const auto* support = std::get_if<pl::SupportBufferSource>(&buffer->source);
        if (!support || support->schema == pl::guest_memory_schema) continue;
        if (support->schema != pl::constant_buffer_support_schema ||
            support->version != pl::constant_buffer_support_version)
            throw std::runtime_error("Ported reflection has an unsupported support-buffer schema");
        if (!first) out << ',';
        first = false;
        out << "{\"schema\":" << quote(support->schema)
            << ",\"version\":" << support->version
            << ",\"byte_extent\":" << buffer->size.bytes
            << ",\"alignment\":" << buffer->size.alignment
            << ",\"metal_buffer_index\":" << std::get<pl::MetalRange>(planned.location).first << '}';
    }
    out << "],\"storage_buffers\":[";
    first = true;
    for (const auto& planned : plan.resources) {
        const auto* buffer = std::get_if<pl::BufferUse>(&planned.use);
        if (!buffer) continue;
        const auto* storage = std::get_if<pl::StorageBufferSource>(&buffer->source);
        if (!storage) continue;
        if (buffer->size.kind != pl::SizeKind::storage_descriptor ||
            buffer->size.bytes != 4 || buffer->size.alignment != 4)
            throw std::runtime_error("Ported reflection has an unsupported storage-buffer size contract");
        const auto* location = std::get_if<pl::MetalRange>(&planned.location);
        if (!location || location->space != pl::MetalNamespace::buffer || location->count != 1)
            throw std::runtime_error("Ported reflection requires a direct storage-buffer binding");
        if (!first) out << ',';
        first = false;
        out << "{\"cbuf_slot\":" << unsigned(storage->descriptor.cbuf_slot)
            << ",\"descriptor_byte_offset\":" << storage->descriptor.handle_byte_offset
            << ",\"metal_buffer_index\":" << location->first
            << ",\"access\":" << quote(buffer->access == pl::Access::read ? "read" :
                buffer->access == pl::Access::write ? "write" : "read_write")
            << ",\"alignment\":" << buffer->size.alignment << '}';
    }
    out << "],\"textures\":[";
    first = true;
    for (const auto& planned : plan.resources) {
        const auto* texture = std::get_if<pl::TextureUse>(&planned.use);
        const auto* sampler = std::get_if<pl::SamplerUse>(&planned.use);
        if (!texture && !sampler) continue;
        if (!first) out << ',';
        first = false;
        if (texture) {
            static constexpr const char* dimensions[] = {"d1", "d2", "d3", "cube", "buffer"};
            static constexpr const char* formats[] = {"unorm", "snorm", "unsigned_integer",
                "signed_integer", "floating", "depth"};
            // A pool names its element type and buffer location. It has no
            // constant-buffer key. Normalized coordinates are not part of that
            // element type; comparison is (the project documentation).
            if (texture->pool) {
                out << "{\"kind\":\"texture_pool\""
                    << ",\"dimension\":" << quote(dimensions[unsigned(texture->descriptor.shape.dimension)])
                    << ",\"array\":" << (texture->descriptor.shape.array ? "true" : "false")
                    << ",\"multisample\":" << (texture->descriptor.shape.multisample ? "true" : "false")
                    << ",\"scalar\":" << quote(scalar_name(texture->descriptor.scalar))
                    << ",\"component_format\":" << quote(formats[unsigned(texture->descriptor.format)])
                    << ",\"comparison\":" << (texture->descriptor.comparison ? "true" : "false")
                    << ",\"resource_kind\":" << quote(texture->kind == pl::ResourceTextureKind::sampled
                        ? "sampled" : "storage_image")
                    << ",\"access\":" << quote(texture->access == pl::Access::read ? "read" :
                        texture->access == pl::Access::write ? "write" : "read_write");
            } else {
                out << "{\"cbuf_slot\":" << unsigned(texture->key.cbuf_slot)
                    << ",\"handle_byte_offset\":" << texture->key.handle_byte_offset
                    << ",\"kind\":" << quote("texture")
                    << ",\"dimension\":" << quote(dimensions[unsigned(texture->descriptor.shape.dimension)])
                    << ",\"array\":" << (texture->descriptor.shape.array ? "true" : "false")
                    << ",\"multisample\":" << (texture->descriptor.shape.multisample ? "true" : "false")
                    << ",\"scalar\":" << quote(scalar_name(texture->descriptor.scalar))
                    << ",\"component_format\":" << quote(formats[unsigned(texture->descriptor.format)])
                    << ",\"normalized_coordinates\":"
                    << (texture->descriptor.normalized_coordinates ? "true" : "false")
                    << ",\"comparison\":" << (texture->descriptor.comparison ? "true" : "false")
                    << ",\"resource_kind\":" << quote(texture->kind == pl::ResourceTextureKind::sampled
                        ? "sampled" : "storage_image")
                    << ",\"access\":" << quote(texture->access == pl::Access::read ? "read" :
                        texture->access == pl::Access::write ? "write" : "read_write");
            }
            if (!texture->pool && texture->kind == pl::ResourceTextureKind::storage_image && texture->atomic)
                out << ",\"atomic\":true";
        } else if (sampler->pool) {
            // One sampler pool for the stage: a buffer binding, with no
            // constant-buffer key. Comparison and normalized coordinates are
            // the sampler state an ID names, not fields of this record.
            out << "{\"kind\":\"sampler_pool\"";
        } else {
            out << "{\"cbuf_slot\":" << unsigned(sampler->key.cbuf_slot)
                << ",\"handle_byte_offset\":" << sampler->key.handle_byte_offset
                << ",\"kind\":" << quote("sampler")
                << ",\"comparison\":" << (sampler->comparison ? "true" : "false")
                << ",\"normalized_coordinates\":"
                << (sampler->normalized_coordinates ? "true" : "false");
        }
        out << ",\"resource_id\":" << planned.resource.value
            << ",\"array_length\":" << (texture ? texture->array_length : sampler->array_length)
            << ",\"location\":";
        write_resource_location(out, planned.location);
        out << '}';
    }
    out << "],\"vertex_streams\":[";
    first = true;
    for (const auto& use : plan.io) {
        if (use.semantic != pl::IoSemantic::generic || use.access != pl::Access::read ||
            !use.vertex_format || stage != pl::Stage::vertex)
            continue;
        if (!first) out << ',';
        first = false;
        static constexpr const char* formats[] = {"u8", "s8", "unorm8", "snorm8", "u16",
            "s16", "unorm16", "snorm16", "f16", "u32", "s32", "f32"};
        out << "{\"slot\":" << use.location
            << ",\"attribute_index\":"
            << [&] {
                   for (const auto& item : options.vertex_attributes)
                       if (item.generic_slot == use.location) return unsigned(item.attribute_index);
                   throw std::runtime_error("Ported plan reads an undeclared vertex slot");
               }()
            << ",\"format\":" << quote(formats[unsigned(use.vertex_format->format)])
            << ",\"components\":" << unsigned(use.vertex_format->components)
            << ",\"byte_stride\":" << unsigned(use.vertex_format->components) * 4 << '}';
    }
    // Inputs classified by the post-SSA use analysis instead of a recorded
    // guest format: the runtime builds the vertex descriptor from the guest
    // attribute registers and admits only formats of this class.
    out << "],\"vertex_inputs\":[";
    first = true;
    for (const auto& use : plan.io) {
        if (use.semantic != pl::IoSemantic::generic || use.access != pl::Access::read ||
            !use.vertex_class || stage != pl::Stage::vertex)
            continue;
        if (!first) out << ',';
        first = false;
        static constexpr const char* classes[] = {"float", "uint", "sint"};
        out << "{\"location\":" << use.location
            << ",\"class\":" << quote(classes[unsigned(*use.vertex_class)]) << '}';
    }
    out << "],\"interface\":{\"position_input_mask\":";
    // The summary mask packs each target's component mask into its own four-bit
    // nibble, matching the SPH export-mask layout the fragment exporter reads.
    unsigned position_input = 0, position_output = 0;
    std::uint32_t color_output = 0;
    // One typed record per written render target, in first-written order. The
    // fragment exporter stores each component separately, so the writer merges
    // them back into one record per location; every component of a location
    // was written from the same fact-proven type, and disagreement here would
    // mean the plan itself is inconsistent.
    struct ColorOutputRecord {
        std::uint16_t location;
        std::uint8_t component_mask;
        pl::IoType type;
    };
    std::vector<ColorOutputRecord> color_outputs;
    bool writes_fragment_depth = false, writes_sample_mask = false;
    for (const auto& use : plan.io) {
        if (use.semantic == pl::IoSemantic::position && use.access != pl::Access::write)
            position_input |= use.mask & 15u;
        if (use.semantic == pl::IoSemantic::position && use.access == pl::Access::write)
            position_output |= use.mask & 15u;
        if (use.semantic == pl::IoSemantic::color && use.access == pl::Access::write) {
            if (use.location >= 8)
                throw std::runtime_error("Fragment colour output exceeds the eight render targets");
            color_output |= static_cast<std::uint32_t>(use.mask) << (4 * use.location);
            const auto prior = std::find_if(color_outputs.begin(), color_outputs.end(),
                [&](const ColorOutputRecord& record) { return record.location == use.location; });
            if (prior == color_outputs.end()) {
                color_outputs.push_back({use.location, use.mask, use.type});
            } else {
                if (prior->type != use.type)
                    throw std::runtime_error("Fragment colour output components disagree on their type");
                prior->component_mask |= use.mask;
            }
        }
        if (use.semantic == pl::IoSemantic::fragment_depth && use.access != pl::Access::read)
            writes_fragment_depth = true;
        if (use.semantic == pl::IoSemantic::sample_mask && use.access != pl::Access::read)
            writes_sample_mask = true;
    }
    out << position_input << ",\"position_output_mask\":" << position_output
        << ",\"color_output_mask\":" << color_output << ",\"color_outputs\":[";
    for (std::size_t index = 0; index < color_outputs.size(); ++index) {
        if (index) out << ',';
        const auto& record = color_outputs[index];
        out << "{\"location\":" << record.location
            << ",\"component_mask\":" << unsigned(record.component_mask)
            << ",\"scalar\":" << quote(scalar_name(record.type.scalar))
            << ",\"components\":" << unsigned(record.type.components) << '}';
    }
    out << "],\"writes_fragment_depth\":" << (writes_fragment_depth ? "true" : "false")
        << ",\"writes_sample_mask\":" << (writes_sample_mask ? "true" : "false");
    const auto io_list = [&](std::string_view name, bool inputs) {
        out << ',' << quote(name) << ":[";
        bool separator = false;
        for (const auto& use : plan.io) {
            if (use.semantic != pl::IoSemantic::generic) continue;
            if (inputs != (use.access != pl::Access::write)) continue;
            if (inputs && stage != pl::Stage::fragment && !(use.vertex_format || use.vertex_class))
                continue;
            static constexpr const char* modes[] = {"flat", "perspective", "linear"};
            if (separator) out << ',';
            separator = true;
            out << "{\"slot\":" << use.location
                << ",\"component_mask\":" << unsigned(use.mask)
                << ",\"scalar\":" << quote(scalar_name(use.type.scalar))
                << ",\"components\":" << unsigned(use.type.components)
                << ",\"interpolation\":" << quote(modes[unsigned(use.interpolation.mode)]) << '}';
        }
        out << ']';
    };
    io_list("generic_inputs", true);
    io_list("generic_outputs", false);
    out << "},\"dependencies\":[";
    first = true;
    for (const auto& dependency : plan.dependencies) {
        if (!first) out << ',';
        first = false;
        static constexpr const char* origins[] = {"validated_header", "captured_state",
            "explicit_option", "derived"};
        out << "{\"kind\":" << quote(fact_kind_name(dependency.key.kind))
            << ",\"index\":" << dependency.key.index
            << ",\"byte_offset\":" << dependency.key.byte_offset
            << ",\"component\":" << unsigned(dependency.key.component)
            << ",\"origin\":" << quote(origins[unsigned(dependency.origin.kind)])
            << ",\"value\":" << fact_value_json(dependency) << '}';
    }
    out << "]}\n";
    return out.str();
}

} // namespace

std::string record_refused_stage(const std::filesystem::path& directory, unsigned selector,
                                 std::span<const std::byte> control, std::span<const std::byte> code) {
    static std::atomic<std::uint64_t> staging_count{0};
    std::vector<std::byte> bytes(control.begin(), control.end());
    bytes.insert(bytes.end(), code.begin(), code.end());
    std::string name;
    try {
        name = "231-" + std::to_string(selector) + "-" + digest(bytes);
    } catch (const std::exception& error) {
        armigo::diag::write(armigo::diag::Level::warning, "Armigo NVN 231.40: could not record a refused stage: %s\n",
                            error.what());
        return {};
    }
    const auto final_path = directory / name;
    std::error_code error;
    if (std::filesystem::exists(final_path / "code.bin", error)) return name;
    const auto staging = directory / (name + ".staging-" + std::to_string(getpid()) + "-" +
                                      std::to_string(staging_count++));
    const auto fail = [&](const char* step, const std::string& detail) {
        armigo::diag::write(armigo::diag::Level::warning, "Armigo NVN 231.40: could not record refused stage %s: %s %s\n",
                            name.c_str(), step, detail.c_str());
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
        return std::string{};
    };
    if (!std::filesystem::create_directories(staging, error))
        return fail("creating", staging.string() + ": " + (error ? error.message() : "it exists"));
    const auto write = [&](const char* file, std::span<const std::byte> data) {
        std::ofstream output(staging / file, std::ios::binary);
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        output.close();
        return !output.fail();
    };
    if (!write("control.bin", control) || !write("code.bin", code)) return fail("writing", staging.string());
    std::filesystem::rename(staging, final_path, error);
    if (error) {
        // Another thread recorded the same bytes first.
        if (std::filesystem::exists(final_path / "code.bin")) {
            std::filesystem::remove_all(staging, error);
            return name;
        }
        return fail("publishing", final_path.string() + ": " + error.message());
    }
    armigo::diag::write(armigo::diag::Level::warning, "Armigo NVN 231.40: recorded refused stage %s\n", name.c_str());
    return name;
}

std::optional<NativeMeshLaunch> native_mesh_launch(const shader_native::Program& program) {
    if ((program.stage != shader_native::Stage::task && program.stage != shader_native::Stage::mesh) || !program.mesh_launch)
        return std::nullopt;
    const auto& facts = *program.mesh_launch;
    return NativeMeshLaunch{program.stage == shader_native::Stage::task, facts.threads, facts.payload_bytes, facts.task,
                            facts.primitive_vertices, facts.max_vertices, facts.max_primitives};
}

std::string native_stage_reflection(const shader::pipeline::MslArtifact& artifact,
                                    const shader::graphics_msl::EmitOptions& emission,
                                    shader::pipeline::Stage stage, std::span<const std::byte> identity,
                                    const NativeMeshLaunch* launch) {
    return reflection(artifact, emission, stage, identity, launch);
}

bool native_texture_pools_enabled() {
    const char* value = std::getenv("ARMIGO_NVN_TEXTURE_POOLS");
    return value == nullptr || std::string_view(value) != "0";
}

} // namespace armigo::platform::metal

namespace armigo::nvn {
namespace {
namespace sn = armigo::shader_native;
namespace pl = armigo::shader::pipeline;
namespace gm = armigo::shader::graphics_msl;
using StagePointer = std::shared_ptr<const platform::CompiledGraphicsStage>;

// Native control: magic, family 0xc9 and version lead. The driver reads
// versions up to 20, each with the code extent at 0x44 and the stage selector
// at 0x56, and is given no control length (a control passed at run time holds
// zero at the container extents); the front end translates version 20, whose
// reader consumes control bytes [0, 0xe0).
constexpr std::uint32_t control_magic = 0x98761234;
constexpr std::uint32_t control_family = 0xc9;
constexpr std::uint32_t maximum_control_version = 20;
constexpr std::size_t code_extent_offset = 0x44;
constexpr std::size_t stage_selector_offset = 0x56;
// Task and mesh stages (selectors 6 and 7) take the tessellation control and
// evaluation slots (3 and 4), which API 231 programs never use, as their
// resource bindings do.
constexpr std::uint32_t last_stage_selector = 7;
constexpr std::size_t stage_slot(std::uint32_t selector) noexcept {
    return selector == 6 ? 3 : selector == 7 ? 4 : selector;
}
constexpr std::size_t maximum_code_bytes = 4u << 20;
// Translation recurses deeply, so it runs on a thread with a large stack
// rather than on the calling thread.
constexpr std::size_t translation_stack_bytes = 64u << 20;

bool read_memory(std::uint64_t address, std::span<std::byte> bytes) {
    if (!address || address > std::numeric_limits<std::uint64_t>::max() - bytes.size()) return false;
    mach_vm_size_t actual = 0;
    return mach_vm_read_overwrite(mach_task_self(), address, bytes.size(),
        reinterpret_cast<mach_vm_address_t>(bytes.data()), &actual) == KERN_SUCCESS && actual == bytes.size();
}

std::uint32_t word(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

bool vertex_fetch_requested() {
    const char* value = std::getenv("ARMIGO_VERTEX_FETCH");
    return value && std::string_view(value) == "1";
}

struct Translation {
    const sn::Program* program;
    sn::FrontendOptions options;
    gm::EmitOptions emission;
    StagePointer stage;
    std::string error;
};

void translate(Translation& work) {
    try {
        const auto& program = *work.program;
        auto translated = sn::translate_native(program, work.options);
        if (const auto* failure = std::get_if<pl::Diagnostic>(&translated)) {
            work.error = "Native shader translation refused the stage: " + failure->message;
            return;
        }
        const auto& artifact = std::get<pl::MslArtifact>(translated);
        // ARMIGO_NVN_SOURCE_DIR=DIR writes each stage translated at run time
        // under the name the package build's --source-dir gives it, so the two
        // can be compared; a mesh stage translated for its fragment stage's
        // inputs adds "-partner".
        static const char* source_directory = std::getenv("ARMIGO_NVN_SOURCE_DIR");
        if (source_directory && *source_directory) {
            const char* depth = !work.options.vertex_depth ? "stage"
                : *work.options.vertex_depth == pl::DepthConvention::zero_to_one ? "zero-to-one" : "minus-one-to-one";
            std::error_code ignored;
            std::filesystem::create_directories(source_directory, ignored);
            std::ofstream(std::filesystem::path(source_directory) /
                              (program.stage_identity + '-' + depth +
                               (work.options.mesh_fragment_reads ? "-partner" : "") + ".metal"),
                          std::ios::binary)
                << artifact.source;
        }
        // Task and mesh stages occupy the tessellation stage kinds; a mesh
        // stage's outputs are written as a vertex stage's.
        const auto stage = program.stage == sn::Stage::vertex ? shader_input::Stage::Vertex :
            program.stage == sn::Stage::fragment ? shader_input::Stage::Fragment :
            program.stage == sn::Stage::task ? shader_input::Stage::TessellationControl :
            program.stage == sn::Stage::mesh ? shader_input::Stage::TessellationEvaluation : shader_input::Stage::Compute;
        const auto pipeline_stage = program.stage == sn::Stage::vertex || program.stage == sn::Stage::mesh
            ? pl::Stage::vertex : program.stage == sn::Stage::fragment ? pl::Stage::fragment : pl::Stage::compute;
        const auto launch = platform::metal::native_mesh_launch(program);
        if ((program.stage == sn::Stage::task || program.stage == sn::Stage::mesh) && !launch) {
            work.error = "Native task or mesh stage has no launch facts";
            return;
        }
        const auto reflection = platform::metal::native_stage_reflection(artifact, work.emission, pipeline_stage,
                                                                         program.image, launch ? &*launch : nullptr);
        auto made = platform::make_source_stage(stage, program.image,
            {reinterpret_cast<const std::byte*>(reflection.data()), reflection.size()}, artifact.source);
        if (const auto* failure = std::get_if<platform::MetalShaderPackageError>(&made)) {
            work.error = "Native shader stage was not accepted: " + failure->detail;
            return;
        }
        platform::CompiledGraphicsStage copy = *std::get<StagePointer>(std::move(made));
        copy.native_stage_identity = program.stage_identity;
        work.stage = std::make_shared<const platform::CompiledGraphicsStage>(std::move(copy));
    } catch (const std::exception& error) {
        work.error = std::string("Native shader translation failed: ") + error.what();
    } catch (...) {
        work.error = "Native shader translation failed";
    }
}

// Translation recurses deeply: it runs on a thread with a large stack.
void run_translation(Translation& work) {
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, translation_stack_bytes);
    pthread_t thread;
    const int created = pthread_create(&thread, &attributes, [](void* raw) -> void* {
        translate(*static_cast<Translation*>(raw));
        return nullptr;
    }, &work);
    pthread_attr_destroy(&attributes);
    if (created == 0) pthread_join(thread, nullptr);
    else work.error = "The native shader translation thread could not start";
}

// Pixel-rate shading is disabled while the driver never applies a shading
// rate (it accepts the shading-rate commands and shades every fragment at
// full rate). Stages that need that fact translate with it only then.
bool pixel_rate_disabled() { return !v231::applies_shading_rate; }

// The translation of a stage with the device's options, and the key a
// recorded stage is kept under for the process.
Translation translation_for(const sn::Program& program, gm::VertexDepthConvention depth, std::string& key) {
    Translation work{&program, {}, {}, nullptr, {}};
    key = program.stage_identity;
    if (pixel_rate_disabled()) {
        work.options.pixel_rate = sn::PixelRateState::disabled;
        key += " pr=0";
    }
    if (program.stage == sn::Stage::vertex || program.stage == sn::Stage::mesh) {
        // A mesh stage writes clip positions as a vertex stage does; it reads no
        // vertex attributes.
        work.options.vertex_depth = depth == gm::VertexDepthConvention::ZeroToOne ? pl::DepthConvention::zero_to_one
                                                                                  : pl::DepthConvention::negative_one_to_one;
        work.emission.vertex_depth = depth;
        key += depth == gm::VertexDepthConvention::ZeroToOne ? " vd=01" : " vd=11";
        // A mesh stage fetches no vertices, as the package build translates it.
        work.options.vertex_fetch = false;
        if (program.stage == sn::Stage::vertex) {
            work.options.vertex_fetch = work.emission.vertex_fetch = vertex_fetch_requested();
            key += work.options.vertex_fetch ? " vf=1" : " vf=0";
        }
    } else {
        work.options.vertex_fetch = false;
        // Metal's fragment position w is the guest's: the native vertex stage
        // writes clip w unchanged.
        if (program.stage == sn::Stage::fragment) {
            work.options.fragment_position = pl::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
            work.emission.fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        }
    }
    work.options.texture_pools = platform::metal::native_texture_pools_enabled();
    key += work.options.texture_pools ? " tp=1" : " tp=0";
    return work;
}

const char* stage_name(std::uint32_t selector) {
    switch (selector) {
        case 0: return "vertex";
        case 1: return "fragment";
        case 2: return "geometry";
        case 3: return "tessellation control";
        case 4: return "tessellation evaluation";
        case 5: return "compute";
        case 6: return "task";
        case 7: return "mesh";
        default: return "unknown";
    }
}

// A stage SetShaders validated and recorded. It is translated when a draw or
// dispatch first uses it; a refusal is reported once, naming the stage, and
// fails every program that holds it. It keeps its own copy of the control and
// code, which the title may release once SetShaders returns.
class RecordedStage final : public PendingStage {
public:
    RecordedStage(std::vector<std::byte> control, std::vector<std::byte> code, std::uint64_t address,
                  gm::VertexDepthConvention depth)
        : control_(std::move(control)), code_(std::move(code)), address_(address), depth_(depth) {}

    // Parses the copy of a control whose magic, family and version the
    // driver reads. A malformed stage returns false with the parser's
    // diagnostic; a stage of a control version or kind the front end does not
    // take is recorded as refused, to fail at first use.
    bool parse(std::string& error) {
        sn::Record record;
        record.kind = sn::ContainerKind::runtime;
        record.control = control_;
        record.code = code_;
        auto parsed = sn::parse_program(record);
        if (auto* failure = std::get_if<sn::ProgramFailure>(&parsed)) {
            char detail[64];
            std::snprintf(detail, sizeof detail, " (control version %u, stage %u)", word(control_, 8),
                          std::to_integer<unsigned>(control_[stage_selector_offset]));
            if (failure->code == sn::ProgramError::control_family ||
                failure->code == sn::ProgramError::unsupported_stage) {
                refusal_ = "Native shader program is not translated: " + failure->message + detail;
                key_ = "refused ";
                key_.append(reinterpret_cast<const char*>(control_.data()), control_.size());
                key_.append(reinterpret_cast<const char*>(code_.data()), code_.size());
                return true;
            }
            error = "Native shader program is malformed: " + failure->message + detail;
            return false;
        }
        program_.emplace(std::get<sn::Program>(std::move(parsed)));
        work_ = translation_for(*program_, depth_, key_);
        work_.program = &*program_;
        return true;
    }
    void capture_input(program_capture::Call& capture, std::size_t index,
                       const std::shared_ptr<RecordedStage>& owner, std::uint64_t address) const noexcept {
        try {
            if (program_) capture.code(index, owner, code_, program_->payload_digest, program_->stage_identity, address);
            else capture.code(index, owner, code_, sn::digest(code_), {}, address);
        } catch (...) { program_capture::record_loss(); }
    }
    std::size_t index() const { return stage_slot(std::to_integer<std::uint32_t>(control_[stage_selector_offset])); }
    const std::string& key() const { return key_; }
    std::string name() const {
        char text[160];
        std::snprintf(text, sizeof text, "%s stage %.16s (code 0x%llx)",
                      stage_name(std::to_integer<std::uint32_t>(control_[stage_selector_offset])),
                      program_ ? program_->stage_identity.c_str() : "(not parsed)",
                      static_cast<unsigned long long>(address_));
        return text;
    }

    StagePointer wait() override {
        std::call_once(once_, [this] {
            stage_ = prepare();
            // A mesh stage from the package is registered too, so a partner
            // variant for a fragment stage is still translated at run time.
            if (stage_ && program_ && program_->stage == sn::Stage::mesh) register_mesh(stage_.get(), this);
        });
        return stage_;
    }
    const sn::Program& program() const { return *program_; }
    // This mesh stage translated for a fragment stage reading `reads`, once
    // per set of reads; null, with why in error, when that is refused.
    StagePointer for_fragment(const std::array<std::uint8_t, 32>& reads, std::string& error) {
        const std::lock_guard lock(variants_mutex_);
        auto [found, added] = variants_.try_emplace(reads);
        if (added) {
            Translation work{&*program_, work_.options, work_.emission, nullptr, {}};
            work.options.mesh_fragment_reads = reads;
            run_translation(work);
            found->second = {std::move(work.stage), std::move(work.error)};
            if (!found->second.first)
                armigo::diag::write(armigo::diag::Level::warning,
                                    "Armigo NVN 231.40: %s for its fragment stage is refused: %s\n", name().c_str(),
                                    found->second.second.c_str());
            else
                armigo::diag::write(armigo::diag::Level::info,
                                    "Armigo NVN 231.40: %s translated for its fragment stage's inputs\n", name().c_str());
        }
        error = found->second.second;
        return found->second.first;
    }

private:
    StagePointer prepare();
    void record_refused() const;
    static void register_mesh(const platform::CompiledGraphicsStage* stage, RecordedStage* recorded);

    std::vector<std::byte> control_, code_;
    std::uint64_t address_;
    gm::VertexDepthConvention depth_;
    std::optional<sn::Program> program_;
    Translation work_{};
    std::string key_, refusal_;
    std::once_flag once_;
    StagePointer stage_;
    std::mutex variants_mutex_;
    std::map<std::array<std::uint8_t, 32>, std::pair<StagePointer, std::string>> variants_;
};

// Translated mesh stages by the stage the replay holds, for the variants
// their fragment stages need. Recorded stages live for the process.
struct MeshStages {
    std::mutex mutex;
    std::map<const platform::CompiledGraphicsStage*, RecordedStage*> stages;
};
MeshStages& mesh_stages() {
    static auto* stages = new MeshStages;
    return *stages;
}
void RecordedStage::register_mesh(const platform::CompiledGraphicsStage* stage, RecordedStage* recorded) {
    auto& meshes = mesh_stages();
    const std::lock_guard lock(meshes.mutex);
    meshes.stages[stage] = recorded;
}

// Recorded stages by identity and options, and the count refused at first
// use, for the process.
struct NativeStages {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<RecordedStage>> recorded;
    std::size_t calls = 0;
    std::size_t refused = 0;
};
NativeStages& native_stages() {
    static auto* stages = new NativeStages;
    return *stages;
}

struct NativePackageState {
    std::mutex mutex;
    std::filesystem::path directory;
    bool noted = false;
    std::once_flag opened;
    bool ready = false;
    std::optional<sn::NativePackage> package;
    std::atomic<std::size_t> hits{0};
    std::atomic<std::size_t> misses{0};
    std::atomic<bool> warned{false};
};

NativePackageState& native_package_state() {
    static auto* state = new NativePackageState;
    return *state;
}

void ensure_native_package() {
    auto& state = native_package_state();
    std::filesystem::path directory;
    {
        const std::lock_guard lock(state.mutex);
        if (!state.noted) return;
        directory = state.directory;
    }
    std::call_once(state.opened, [&] {
        std::atexit([] {
            const auto& current = native_package_state();
            armigo::diag::write(armigo::diag::Level::info,
                                "Armigo NVN 231.40: native shader package hits %zu, misses %zu\n",
                                current.hits.load(), current.misses.load());
        });
        auto opened = sn::NativePackage::open(directory);
        if (const auto* error = std::get_if<std::string>(&opened)) {
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN 231.40: native shader package was not used: %s\n", error->c_str());
            return;
        }
        state.package.emplace(std::get<sn::NativePackage>(std::move(opened)));
        state.ready = true;
    });
}

const char* package_stage_name(sn::Stage stage) {
    switch (stage) {
    case sn::Stage::vertex: return "vertex";
    case sn::Stage::fragment: return "fragment";
    case sn::Stage::compute: return "compute";
    case sn::Stage::task: return "task";
    case sn::Stage::mesh: return "mesh";
    }
    return "stage";
}

const char* package_miss_reason(sn::NativeMiss::Reason reason) {
    switch (reason) {
    case sn::NativeMiss::Reason::absent: return "absent";
    case sn::NativeMiss::Reason::specialization: return "specialization";
    case sn::NativeMiss::Reason::corrupt: return "corrupt";
    }
    return "miss";
}

std::size_t identity_distance(std::string_view left, std::string_view right) {
    std::size_t mismatched = 0;
    const std::size_t limit = std::min(left.size(), right.size());
    for (std::size_t index = 0; index < limit; ++index)
        mismatched += static_cast<std::size_t>(left[index] != right[index]);
    return mismatched + (std::max(left.size(), right.size()) - limit);
}

// One line per miss: the runtime key, why it missed, and the closest packaged key.
void note_package_miss(const sn::Program& program, const char* reason, const char* detail,
                       const sn::NativePackage* package) {
    const char* nearest = "none";
    std::size_t distance = 0;
    std::string owned;
    if (package) {
        bool found = false;
        for (const auto& entry : package->section().entries) {
            const std::size_t mismatched = identity_distance(entry.stage_identity, program.stage_identity);
            if (!found || mismatched < distance) {
                found = true;
                distance = mismatched;
                owned = entry.stage_identity;
                if (distance == 0) break;
            }
        }
        if (found) nearest = owned.c_str();
    }
    armigo::diag::write(armigo::diag::Level::info,
                        "Armigo NVN 231.40: native shader package miss %s %s %s%s%s nearest %s distance %zu\n",
                        package_stage_name(program.stage), program.stage_identity.c_str(), reason,
                        detail && detail[0] ? " " : "", detail && detail[0] ? detail : "", nearest, distance);
}

StagePointer packaged_stage(const sn::Program& program, gm::VertexDepthConvention depth) {
    auto& state = native_package_state();
    bool noted = false;
    {
        const std::lock_guard lock(state.mutex);
        noted = state.noted;
    }
    if (!noted) return nullptr;
    ensure_native_package();
    if (!state.ready || !state.package) {
        note_package_miss(program, "absent", "package", nullptr);
        state.misses.fetch_add(1);
        return nullptr;
    }
    sn::NativeLookup lookup;
    lookup.pixel_rate_disabled = pixel_rate_disabled();
    if (program.stage == sn::Stage::vertex) {
        lookup.stage = sn::NativeStageKind::vertex;
        lookup.vertex_fetch = vertex_fetch_requested();
        lookup.vertex_depth_zero_to_one = depth == gm::VertexDepthConvention::ZeroToOne;
    } else if (program.stage == sn::Stage::fragment) {
        lookup.stage = sn::NativeStageKind::fragment;
        lookup.fragment_position_matches = true;
    } else if (program.stage == sn::Stage::task) {
        lookup.stage = sn::NativeStageKind::task;
    } else if (program.stage == sn::Stage::mesh) {
        // A mesh stage takes the vertex depth convention and reads no attributes.
        lookup.stage = sn::NativeStageKind::mesh;
        lookup.vertex_depth_zero_to_one = depth == gm::VertexDepthConvention::ZeroToOne;
    } else {
        lookup.stage = sn::NativeStageKind::compute;
    }
    auto found = state.package->find(program.stage_identity, lookup);
    if (const auto* miss = std::get_if<sn::NativeMiss>(&found)) {
        if (miss->reason == sn::NativeMiss::Reason::corrupt && !state.warned.exchange(true)) {
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN 231.40: native shader package entry is unusable: %s\n",
                                miss->detail.c_str());
        }
        note_package_miss(program, package_miss_reason(miss->reason), miss->detail.c_str(), &*state.package);
        state.misses.fetch_add(1);
        return nullptr;
    }
    const auto& files = std::get<sn::NativeStageFiles>(found);
    const auto stage = program.stage == sn::Stage::vertex ? shader_input::Stage::Vertex :
        program.stage == sn::Stage::fragment ? shader_input::Stage::Fragment :
        program.stage == sn::Stage::task ? shader_input::Stage::TessellationControl :
        program.stage == sn::Stage::mesh ? shader_input::Stage::TessellationEvaluation : shader_input::Stage::Compute;
    // make_source_stage hashes source text only when it is in memory. An
    // omitted source.metal still has its digest in the reflection; name a
    // stand-in file so the call accepts that, then drop it for the library.
    auto made = platform::make_source_stage(stage, files.program,
                                             {files.reflection.data(), files.reflection.size()}, files.source,
                                             files.source.empty() ? std::string("packaged") : std::string{});
    if (const auto* failure = std::get_if<platform::MetalShaderPackageError>(&made)) {
        note_package_miss(program, "stage", failure->detail.c_str(), &*state.package);
        state.misses.fetch_add(1);
        return nullptr;
    }
    platform::CompiledGraphicsStage copy = *std::get<StagePointer>(std::move(made));
    copy.native_stage_identity = program.stage_identity;
    copy.metallib = files.library;
    copy.msl_source.clear();
    copy.msl_source_file.clear();
    state.hits.fetch_add(1);
    armigo::diag::write(armigo::diag::Level::info, "Armigo NVN 231.40: native shader package hit %s %s\n",
                        package_stage_name(program.stage), program.stage_identity.c_str());
    return std::make_shared<const platform::CompiledGraphicsStage>(std::move(copy));
}

StagePointer RecordedStage::prepare() {
    if (program_) {
        if (auto packaged = packaged_stage(*program_, depth_)) return packaged;
        run_translation(work_);
        if (work_.stage) return std::move(work_.stage);
        refusal_ = std::move(work_.error);
    }
    auto& stages = native_stages();
    std::size_t refused;
    {
        const std::lock_guard lock(stages.mutex);
        refused = ++stages.refused;
    }
    armigo::diag::write(armigo::diag::Level::warning,
                        "Armigo NVN 231.40: %s is refused at first use (%zu stages refused); its draws are refused: %s\n",
                        name().c_str(), refused, refusal_.c_str());
    record_refused();
    return nullptr;
}

// ARMIGO_SHADER_MISS_DIR=DIR writes each refused stage, as the driver received
// it, so its translation can be prepared offline (record_refused_stage).
void RecordedStage::record_refused() const {
    static const char* directory = std::getenv("ARMIGO_SHADER_MISS_DIR");
    if (!directory || !*directory) return;
    armigo::platform::metal::record_refused_stage(directory, std::to_integer<unsigned>(control_[stage_selector_offset]),
                                                  control_, code_);
}

// SetShaders as the console driver takes it: each stage's control and code
// are validated and recorded, and the call succeeds for well-formed input.
// Translation waits for the stage's first use.
bool resolve_native(void*, const DeviceState* device, std::span<const NVNshaderData> descriptors,
                    std::array<StagePointer, 6>& stages, std::array<std::shared_ptr<PendingStage>, 6>& pending,
                    std::string& error) {
    const auto depth = device && device->depth_mode == 1 ? gm::VertexDepthConvention::ZeroToOne
                                                         : gm::VertexDepthConvention::NegativeOneToOne;
    for (const auto& descriptor : descriptors) {
        std::vector<std::byte> control(sn::runtime_control_bytes);
        const auto control_address = reinterpret_cast<std::uint64_t>(descriptor.control);
        if (!read_memory(control_address, control)) {
            error = "Unreadable native shader control"; return false;
        }
        if (auto* capture = program_capture::Call::active())
            capture->control(std::size_t(&descriptor - descriptors.data()), control, control_address);
        const auto selector = std::to_integer<std::uint32_t>(control[stage_selector_offset]);
        if (word(control, 0) != control_magic || word(control, 4) != control_family ||
            word(control, 8) > maximum_control_version || selector > last_stage_selector) {
            char detail[128];
            std::snprintf(detail, sizeof detail,
                          "Shader control is not a native Switch 2 stage (magic 0x%x, family 0x%x, version %u, stage %u)",
                          word(control, 0), word(control, 4), word(control, 8), selector);
            error = detail;
            return false;
        }
        const auto code_size = word(control, code_extent_offset);
        if (!code_size || code_size > maximum_code_bytes) {
            char detail[96];
            std::snprintf(detail, sizeof detail, "Native shader code extent 0x%x is outside supported bounds", code_size);
            error = detail;
            return false;
        }
        std::vector<std::byte> code(code_size);
        const auto data_address = descriptor.data;
        if (!read_memory(data_address, code)) { error = "Unreadable native shader code"; return false; }
        auto stage = std::make_shared<RecordedStage>(std::move(control), std::move(code), descriptor.data, depth);
        const bool parsed = stage->parse(error);
        if (auto* capture = program_capture::Call::active())
            stage->capture_input(*capture, std::size_t(&descriptor - descriptors.data()), stage, data_address);
        if (!parsed) return false;
        const auto index = stage->index();
        if (index >= pending.size() || stages[index] || pending[index]) {
            error = "Duplicate or invalid shader stage"; return false;
        }
        auto& recorded = native_stages();
        const std::lock_guard lock(recorded.mutex);
        stage = recorded.recorded.try_emplace(stage->key(), std::move(stage)).first->second;
        pending[index] = std::move(stage);
        // Titles record thousands of stages while they load; their progress
        // shows at the info level.
        if (++recorded.calls % 1024 == 0)
            armigo::diag::write(armigo::diag::Level::info, "Armigo NVN 231.40: %zu stages recorded (%zu distinct)\n",
                                recorded.calls, recorded.recorded.size());
    }
    return true;
}

} // namespace

void set_native_program_resolver() noexcept {
    program_capture::configure_from_environment([](void*, std::uint64_t address, std::span<std::byte> output) {
        return read_memory(address, output);
    });
    set_deferred_program_resolver(resolve_native, nullptr);
}

std::shared_ptr<const platform::CompiledGraphicsStage> mesh_stage_for(
    const std::shared_ptr<const platform::CompiledGraphicsStage>& mesh, const platform::CompiledGraphicsStage& fragment,
    std::string& error) {
    RecordedStage* recorded = nullptr;
    {
        auto& meshes = mesh_stages();
        const std::lock_guard lock(meshes.mutex);
        if (const auto found = meshes.stages.find(mesh.get()); found != meshes.stages.end()) recorded = found->second;
    }
    if (!recorded) return mesh;
    std::array<std::uint8_t, 32> reads{};
    for (const auto& input : fragment.interface.generic_inputs)
        if (input.slot < reads.size()) reads[input.slot] |= static_cast<std::uint8_t>(input.component_mask & 15u);
    const auto outputs = sn::mesh_generic_outputs(recorded->program());
    bool covered = true;
    for (std::size_t n = 0; n < reads.size(); ++n) covered &= !(reads[n] & ~outputs[n]);
    if (covered) return mesh;
    return recorded->for_fragment(reads, error);
}

void set_native_shader_package(std::filesystem::path directory) noexcept {
    try {
        auto& state = native_package_state();
        const std::lock_guard lock(state.mutex);
        if (state.noted) return;
        state.directory = std::move(directory);
        state.noted = true;
    } catch (...) {
    }
}

NativePackageUse native_shader_package_counts() noexcept {
    const auto& state = native_package_state();
    return {state.hits.load(), state.misses.load()};
}

} // namespace armigo::nvn
